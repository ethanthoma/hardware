from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from pe_mac import PE_MAC


def run(steps):
    """Each step is a dict of controls/operands; returns (acc_wide, acc_out float) after each tick.

    keys: load_c / load_acc / enable (0/1), a / b / c (floats → bf16), acc_in (35-bit int).
    """
    dut = PE_MAC()
    snaps = []

    async def bench(ctx):
        for st in steps:
            ctx.set(dut.load_c, st.get("load_c", 0))
            ctx.set(dut.load_acc, st.get("load_acc", 0))
            ctx.set(dut.enable, st.get("enable", 0))
            ctx.set(dut.acc_in, st.get("acc_in", 0))
            for key, port in (("a", dut.a_in), ("b", dut.b_in), ("c", dut.c_in)):
                if key in st:
                    s, e, mant = BF16.from_float(st[key]).unpack()
                    ctx.set(port, {"sign": s, "exponent": e, "mantissa": mant})
            await ctx.tick()
            r = ctx.get(dut.acc_out)
            snaps.append((ctx.get(dut.acc_wide), BF16.pack(r["sign"], r["exponent"], r["mantissa"]).to_float()))

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()
    return snaps


def test_wide_load_read_round_trip():
    value = 0x5_1234_5678 & ((1 << 35) - 1)
    wide, _ = run([{"load_acc": 1, "acc_in": value}])[-1]
    assert wide == value


def test_load_acc_zero():
    wide, out = run([{"load_acc": 1, "acc_in": 0}])[-1]
    assert wide == 0
    assert out == 0.0


def test_restore_equals_continuous():
    # Accumulating continuously must equal: accumulate, spill the 35-bit acc,
    # restore it into a fresh PE, then continue. This is the wide-persist invariant.
    continuous = run(
        [
            {"load_c": 1, "c": 1.0},
            {"enable": 1, "a": 1.5, "b": 2.0},
            {"enable": 1, "a": 0.5, "b": 3.0},
        ]
    )
    expected = continuous[-1][1]

    spilled = run(
        [
            {"load_c": 1, "c": 1.0},
            {"enable": 1, "a": 1.5, "b": 2.0},
        ]
    )[-1][0]

    restored = run(
        [
            {"load_acc": 1, "acc_in": spilled},
            {"enable": 1, "a": 0.5, "b": 3.0},
        ]
    )
    assert restored[-1][1] == expected
