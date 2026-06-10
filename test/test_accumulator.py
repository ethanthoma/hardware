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
        out["any_overflow"] = ctx.get(dut.any_overflow)
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


def test_overflow_flag_on_wrap():
    # signed(10) tops out at 511; 400 + 200 wraps -> overflow flag sticks
    out = run(Accumulator(width=10, lsb_exp=0), [("load", 400), ("add", 200)])
    assert out["any_overflow"] == 1


def test_no_overflow_in_range():
    out = run(Accumulator(width=10, lsb_exp=0), [("load", 200), ("add", 100)])
    assert out["any_overflow"] == 0


def test_load_clears_overflow():
    out = run(Accumulator(width=10, lsb_exp=0), [("load", 400), ("add", 200), ("load", 5)])
    assert out["any_overflow"] == 0


def run_banked(dut, ops, read_sel):
    """ops: list of ("load"|"add", addend, acc_sel). Returns value/result/flags of read_sel after settling."""
    out = {}

    async def bench(ctx):
        for kind, addend, sel in ops:
            ctx.set(dut.acc_sel, sel)
            ctx.set(dut.load, kind == "load")
            ctx.set(dut.enable, kind == "add")
            ctx.set(dut.addend, addend)
            await ctx.tick()
        ctx.set(dut.load, 0)
        ctx.set(dut.enable, 0)
        ctx.set(dut.acc_sel, read_sel)
        await ctx.tick()  # drain is pipelined one cycle behind the selected bank
        out["value"] = ctx.get(dut.value)
        out["any_overflow"] = ctx.get(dut.any_overflow)
        r = ctx.get(dut.result)
        out["result"] = BF16.pack(r["sign"], r["exponent"], r["mantissa"]).to_float()

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()
    return out


def test_banks_accumulate_independently():
    ops = [("load", 5, 0), ("load", 100, 1), ("add", 3, 0), ("add", -50, 1)]
    assert run_banked(Accumulator(width=32, lsb_exp=0), ops, 0)["value"] == 8
    assert run_banked(Accumulator(width=32, lsb_exp=0), ops, 1)["value"] == 50


def test_drain_reads_selected_bank():
    ops = [("load", 12, 0), ("load", 7, 2)]
    assert run_banked(Accumulator(width=32, lsb_exp=0), ops, 0)["result"] == 12.0
    assert run_banked(Accumulator(width=32, lsb_exp=0), ops, 2)["result"] == 7.0


def test_overflow_sticks_per_bank():
    ops = [("load", 200, 0), ("load", 400, 1), ("add", 200, 1)]
    assert run_banked(Accumulator(width=10, lsb_exp=0), ops, 1)["any_overflow"] == 1
    assert run_banked(Accumulator(width=10, lsb_exp=0), ops, 0)["any_overflow"] == 0
