from amaranth.sim import Simulator

from mantissa_multiplier import MantissaMultiplier

dut = MantissaMultiplier()


def test_mantissa_mult_basic():
    async def bench(ctx):
        ctx.set(dut.a_mant, 0)
        ctx.set(dut.b_mant, 0)
        product = ctx.get(dut.product)
        assert product == 0x4000, f"Expected 0x4000, got 0x{product:04x}"

        ctx.set(dut.a_mant, 0x40)
        ctx.set(dut.b_mant, 0)
        product = ctx.get(dut.product)
        assert product == 0x6000, f"Expected 0x6000, got 0x{product:04x}"

        ctx.set(dut.a_mant, 0x40)
        ctx.set(dut.b_mant, 0x40)
        product = ctx.get(dut.product)
        assert product == 0x9000, f"Expected 0x9000, got 0x{product:04x}"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()
