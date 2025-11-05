from amaranth.sim import Simulator

from mantissa_multiplier import MantissaMultiplier

dut = MantissaMultiplier()


def test_mantissa_mult_basic():
    async def bench(ctx):
        # 1.0 × 1.0: mant=0 (with implicit 1 = 0x80) × mant=0 (= 0x80)
        # 0x80 × 0x80 = 0x4000 (bit 14 set, in 2.14 fixed point format)
        ctx.set(dut.a_mant, 0)
        ctx.set(dut.b_mant, 0)
        await ctx.delay(1e-6)
        product = ctx.get(dut.product)
        assert product == 0x4000, f"Expected 0x4000, got 0x{product:04x}"

        # 1.5 × 1.0: mant=0x40 (with implicit 1 = 0xC0) × mant=0 (= 0x80)
        # 0xC0 × 0x80 = 0x6000
        ctx.set(dut.a_mant, 0x40)
        ctx.set(dut.b_mant, 0)
        await ctx.delay(1e-6)
        product = ctx.get(dut.product)
        assert product == 0x6000, f"Expected 0x6000, got 0x{product:04x}"

        # 1.5 × 1.5: 0xC0 × 0xC0 = 0x9000 (2.25)
        ctx.set(dut.a_mant, 0x40)
        ctx.set(dut.b_mant, 0x40)
        await ctx.delay(1e-6)
        product = ctx.get(dut.product)
        assert product == 0x9000, f"Expected 0x9000, got 0x{product:04x}"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()
