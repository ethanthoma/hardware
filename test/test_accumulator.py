from amaranth.hdl import Period
from amaranth.sim import Simulator

from accumulator import Accumulator
from bfloat16 import BF16


def run(dut, ops):
    """ops: list of ("load"|"add", addend). Returns (raw value, drained bf16 float)."""
    out = {}

    async def bench(ctx):
        ctx.set(dut.load, 0)
        ctx.set(dut.enable, 0)
        for kind, addend in ops:
            ctx.set(dut.load, kind == "load")
            ctx.set(dut.enable, kind == "add")
            ctx.set(dut.addend, addend)
            await ctx.tick()
        ctx.set(dut.load, 0)
        ctx.set(dut.enable, 0)
        await ctx.tick()  # drain is pipelined one cycle behind acc; hold then settle
        out["value"] = ctx.get(dut.value)
        out["result_valid"] = ctx.get(dut.result_valid)
        r = ctx.get(dut.result)
        out["result"] = BF16.pack(r["sign"], r["exponent"], r["mantissa"]).to_float()

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()
    return out


def test_integer_accumulation():
    out = run(Accumulator(width=32, lsb_exp=0), [("load", 5), ("add", 3), ("add", -2)])
    assert out["value"] == 6


def test_load_overwrites():
    out = run(Accumulator(width=32, lsb_exp=0), [("load", 100), ("add", 5), ("load", 7)])
    assert out["value"] == 7


def test_drain_exact():
    out = run(Accumulator(width=32, lsb_exp=0), [("load", 12)])
    assert out["result"] == 12.0


def test_drain_negative():
    out = run(Accumulator(width=32, lsb_exp=0), [("load", -12)])
    assert out["result"] == -12.0


def test_drain_round_to_even():
    # 257 = 256 + 1 is exactly halfway between bf16 neighbours 256 and 258;
    # round-to-nearest-even picks 256 (mantissa 0 is even)
    out = run(Accumulator(width=32, lsb_exp=0), [("load", 257)])
    assert out["result"] == 256.0


def test_drain_fractional_scale():
    # lsb_exp=-10 means value 1024 represents 1024 * 2**-10 = 1.0
    out = run(Accumulator(width=64, lsb_exp=-10), [("load", 1024)])
    assert out["result"] == 1.0


def test_accumulate_then_drain():
    out = run(Accumulator(width=32, lsb_exp=0), [("load", 8), ("add", 4), ("add", 4)])
    assert out["value"] == 16
    assert out["result"] == 16.0


def test_result_valid_after_settle():
    out = run(Accumulator(width=32, lsb_exp=0), [("load", 12)])
    assert out["result_valid"] == 1
