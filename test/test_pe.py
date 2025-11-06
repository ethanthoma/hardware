import sys

from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from pe import PE


def test_pe_single_mac(request):
    dut = PE()

    async def bench(ctx):
        a = BF16.from_float(2.0)
        b = BF16.from_float(3.0)
        c = BF16.from_float(1.0)

        a_s, a_e, a_m = a.unpack()
        b_s, b_e, b_m = b.unpack()
        c_s, c_e, c_m = c.unpack()

        ctx.set(dut.a_in, {"sign": a_s, "exponent": a_e, "mantissa": a_m})
        ctx.set(dut.b_in, {"sign": b_s, "exponent": b_e, "mantissa": b_m})
        ctx.set(dut.c_in, {"sign": c_s, "exponent": c_e, "mantissa": c_m})
        ctx.set(dut.load_c, 1)
        ctx.set(dut.enable, 0)

        await ctx.tick()

        ctx.set(dut.load_c, 0)
        ctx.set(dut.enable, 1)

        await ctx.tick()

        ctx.set(dut.enable, 0)

        await ctx.tick()

        result_struct = ctx.get(dut.acc_out)
        result = BF16.pack(result_struct["sign"], result_struct["exponent"], result_struct["mantissa"])
        result_f = result.to_float()

        expected = 2.0 * 3.0 + 1.0
        error = abs(result_f - expected)

        assert error < 0.01, f"Single MAC: got {result_f}, expected {expected}"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"TensorPE_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_pe_accumulation(request):
    dut = PE()

    async def bench(ctx):
        c = BF16.from_float(0.0)
        c_s, c_e, c_m = c.unpack()

        ctx.set(dut.c_in, {"sign": c_s, "exponent": c_e, "mantissa": c_m})
        ctx.set(dut.load_c, 1)
        ctx.set(dut.enable, 0)

        await ctx.tick()

        ctx.set(dut.load_c, 0)

        operations = [
            (1.0, 1.0, 1.0),
            (2.0, 2.0, 5.0),
            (1.0, 3.0, 8.0),
            (0.5, 4.0, 10.0),
        ]

        for a_f, b_f, expected in operations:
            a = BF16.from_float(a_f)
            b = BF16.from_float(b_f)

            a_s, a_e, a_m = a.unpack()
            b_s, b_e, b_m = b.unpack()

            ctx.set(dut.a_in, {"sign": a_s, "exponent": a_e, "mantissa": a_m})
            ctx.set(dut.b_in, {"sign": b_s, "exponent": b_e, "mantissa": b_m})
            ctx.set(dut.enable, 1)

            await ctx.tick()

            ctx.set(dut.enable, 0)

            await ctx.tick()

            result_struct = ctx.get(dut.acc_out)
            result = BF16.pack(result_struct["sign"], result_struct["exponent"], result_struct["mantissa"])
            result_f = result.to_float()

            error = abs(result_f - expected)
            rel_error = error / abs(expected) if expected != 0 else error

            assert rel_error < 0.02, f"Accumulation ({a_f} × {b_f}): got {result_f}, expected {expected}"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"TensorPE_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_pe_load_control(request):
    dut = PE()

    async def bench(ctx):
        c1 = BF16.from_float(5.0)
        c1_s, c1_e, c1_m = c1.unpack()

        ctx.set(dut.c_in, {"sign": c1_s, "exponent": c1_e, "mantissa": c1_m})
        ctx.set(dut.load_c, 1)
        ctx.set(dut.enable, 0)

        a = BF16.from_float(2.0)
        b = BF16.from_float(3.0)
        a_s, a_e, a_m = a.unpack()
        b_s, b_e, b_m = b.unpack()

        ctx.set(dut.a_in, {"sign": a_s, "exponent": a_e, "mantissa": a_m})
        ctx.set(dut.b_in, {"sign": b_s, "exponent": b_e, "mantissa": b_m})

        await ctx.tick()

        ctx.set(dut.load_c, 0)
        ctx.set(dut.enable, 1)

        await ctx.tick()

        ctx.set(dut.enable, 0)
        await ctx.tick()

        result_struct = ctx.get(dut.acc_out)
        result = BF16.pack(result_struct["sign"], result_struct["exponent"], result_struct["mantissa"])
        result_f = result.to_float()

        expected = 2.0 * 3.0 + 5.0
        error = abs(result_f - expected)

        assert error < 0.01, f"Load C: got {result_f}, expected {expected}"

        c2 = BF16.from_float(10.0)
        c2_s, c2_e, c2_m = c2.unpack()

        ctx.set(dut.c_in, {"sign": c2_s, "exponent": c2_e, "mantissa": c2_m})
        ctx.set(dut.load_c, 1)

        await ctx.tick()

        ctx.set(dut.load_c, 0)
        ctx.set(dut.enable, 1)

        await ctx.tick()

        ctx.set(dut.enable, 0)
        await ctx.tick()

        result_struct = ctx.get(dut.acc_out)
        result = BF16.pack(result_struct["sign"], result_struct["exponent"], result_struct["mantissa"])
        result_f = result.to_float()

        expected = 2.0 * 3.0 + 10.0
        error = abs(result_f - expected)

        assert error < 0.01, f"Reload C: got {result_f}, expected {expected}"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"TensorPE_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_pe_enable_control(request):
    dut = PE()

    async def bench(ctx):
        c = BF16.from_float(0.0)
        c_s, c_e, c_m = c.unpack()

        ctx.set(dut.c_in, {"sign": c_s, "exponent": c_e, "mantissa": c_m})
        ctx.set(dut.load_c, 1)
        ctx.set(dut.enable, 0)

        await ctx.tick()

        ctx.set(dut.load_c, 0)
        await ctx.tick()

        a = BF16.from_float(2.0)
        b = BF16.from_float(3.0)
        a_s, a_e, a_m = a.unpack()
        b_s, b_e, b_m = b.unpack()

        ctx.set(dut.a_in, {"sign": a_s, "exponent": a_e, "mantissa": a_m})
        ctx.set(dut.b_in, {"sign": b_s, "exponent": b_e, "mantissa": b_m})
        ctx.set(dut.enable, 0)

        await ctx.tick()
        await ctx.tick()

        result_struct = ctx.get(dut.acc_out)
        result = BF16.pack(result_struct["sign"], result_struct["exponent"], result_struct["mantissa"])
        result_f = result.to_float()

        assert abs(result_f) < 0.01, f"Enable=0 should not accumulate: got {result_f}, expected 0.0"

        ctx.set(dut.enable, 1)
        await ctx.tick()

        ctx.set(dut.enable, 0)
        await ctx.tick()

        result_struct = ctx.get(dut.acc_out)
        result = BF16.pack(result_struct["sign"], result_struct["exponent"], result_struct["mantissa"])
        result_f = result.to_float()

        expected = 2.0 * 3.0
        error = abs(result_f - expected)

        assert error < 0.01, f"Enable=1 should accumulate: got {result_f}, expected {expected}"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"TensorPE_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_pe_eight_macs(request):
    dut = PE()

    async def bench(ctx):
        c = BF16.from_float(0.0)
        c_s, c_e, c_m = c.unpack()

        ctx.set(dut.c_in, {"sign": c_s, "exponent": c_e, "mantissa": c_m})
        ctx.set(dut.load_c, 1)
        ctx.set(dut.enable, 0)

        await ctx.tick()

        ctx.set(dut.load_c, 0)

        expected_acc = 0.0
        for i in range(8):
            a_f = 1.0 + i * 0.5
            b_f = 1.0
            expected_acc += a_f * b_f

            a = BF16.from_float(a_f)
            b = BF16.from_float(b_f)

            a_s, a_e, a_m = a.unpack()
            b_s, b_e, b_m = b.unpack()

            ctx.set(dut.a_in, {"sign": a_s, "exponent": a_e, "mantissa": a_m})
            ctx.set(dut.b_in, {"sign": b_s, "exponent": b_e, "mantissa": b_m})
            ctx.set(dut.enable, 1)

            await ctx.tick()

            ctx.set(dut.enable, 0)
            await ctx.tick()

        result_struct = ctx.get(dut.acc_out)
        result = BF16.pack(result_struct["sign"], result_struct["exponent"], result_struct["mantissa"])
        result_f = result.to_float()

        error = abs(result_f - expected_acc)
        rel_error = error / abs(expected_acc)

        assert rel_error < 0.02, f"8 MACs: got {result_f}, expected {expected_acc}, rel_error={rel_error:.6f}"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"TensorPE_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()
