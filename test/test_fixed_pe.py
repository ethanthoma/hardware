import random
import struct

from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from fixed_pe import FixedPE


def bf16_rtne(x: float) -> float:
    """Cast x to bf16 with round-to-nearest-even, matching the Rounder hardware."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    if (bits >> 23) & 0xFF == 0xFF:
        return x
    low, high = bits & 0xFFFF, bits >> 16
    if low > 0x8000 or (low == 0x8000 and (high & 1)):
        high += 1
    assert high <= 0xFFFF
    return struct.unpack("<f", struct.pack("<I", high << 16))[0]


def run_fixed_pe(pairs: list[tuple[float, float]]) -> float:
    """Cycle 0 loads acc := first product; subsequent cycles accumulate."""
    dut = FixedPE()
    captured: dict[str, float] = {}

    async def bench(ctx):
        for i, (a_val, b_val) in enumerate(pairs):
            sa, ea, ma = BF16.from_float(a_val).unpack()
            sb, eb, mb = BF16.from_float(b_val).unpack()
            ctx.set(dut.a, {"sign": sa, "exponent": ea, "mantissa": ma})
            ctx.set(dut.b, {"sign": sb, "exponent": eb, "mantissa": mb})
            ctx.set(dut.load, 1 if i == 0 else 0)
            ctx.set(dut.enable, 0 if i == 0 else 1)
            await ctx.tick()
        ctx.set(dut.load, 0)
        ctx.set(dut.enable, 0)
        r = ctx.get(dut.result)
        captured["result"] = BF16.pack(r["sign"], r["exponent"], r["mantissa"]).to_float()

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()
    return captured["result"]


def drain_once_reference(pairs: list[tuple[float, float]]) -> float:
    """FixedPE's contract: bf16-truncate operands, accumulate exactly, RTNE-cast once."""
    acc = sum(BF16.from_float(a).to_float() * BF16.from_float(b).to_float() for a, b in pairs)
    return bf16_rtne(acc)


def test_zero_product_drains_zero():
    assert run_fixed_pe([(0.0, 1.0)]) == 0.0


def test_single_product():
    assert run_fixed_pe([(1.5, 2.0)]) == 3.0
    assert run_fixed_pe([(-1.5, 2.0)]) == -3.0


def test_eight_ones_accumulate_to_eight():
    assert run_fixed_pe([(1.0, 1.0)] * 8) == 8.0


def test_alternating_signs_cancel_exactly():
    pairs = [(1.25, 4.0), (-1.25, 4.0), (1.25, 4.0), (-1.25, 4.0)]
    assert run_fixed_pe(pairs) == 0.0


def test_in_window_matches_drain_once_reference():
    rng = random.Random(0)
    for _ in range(20):
        pairs = [
            (
                rng.choice([-1.0, 1.0]) * (1.0 + rng.random()) * (2.0 ** rng.randint(-2, 2)),
                rng.choice([-1.0, 1.0]) * (1.0 + rng.random()) * (2.0 ** rng.randint(-2, 2)),
            )
            for _ in range(16)
        ]
        assert run_fixed_pe(pairs) == drain_once_reference(pairs)


def test_product_above_window_is_dropped():
    assert run_fixed_pe([(1.0, 1.0), (2.0**10, 2.0**10)]) == 1.0


def test_product_below_window_is_dropped():
    assert run_fixed_pe([(1.0, 1.0), (2.0**-20, 2.0**-20)]) == 1.0


def test_window_sweep(capsys):
    rng = random.Random(1)
    rows = []
    for exp in range(-22, 20, 2):
        a, b = (1.0 + rng.random()) * (2.0**exp), 1.0 + rng.random()
        got = run_fixed_pe([(a, b)])
        want = drain_once_reference([(a, b)])
        rel_err = abs(got - want) / abs(want) if want != 0.0 else (0.0 if got == 0.0 else float("inf"))
        rows.append((exp, want, got, rel_err))

    with capsys.disabled():
        print(f"\n{'a_exp':>6} {'reference':>14} {'fixed_pe':>14} {'rel_err':>10}  status")
        for exp, want, got, rel_err in rows:
            status = "in-window" if got == want else ("dropped" if got == 0.0 else "drift")
            print(f"{exp:>6d} {want:>14.4g} {got:>14.4g} {rel_err:>10.2e}  {status}")

    drifts = [(exp, want, got) for exp, want, got, _ in rows if got != want and got != 0.0]
    assert not drifts, f"unexpected drift: {drifts}"
