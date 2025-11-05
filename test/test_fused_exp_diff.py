from amaranth.sim import Simulator

from fused_exp_diff import FusedExponentDifference

dut = FusedExponentDifference()


def test_fused_exp_diff_equal_exponents():
    async def bench(ctx):
        # All exponents equal: (127 + 127 - 127) - 127 = 0
        ctx.set(dut.a_exp, 127)
        ctx.set(dut.b_exp, 127)
        ctx.set(dut.c_exp, 127)
        await ctx.delay(1e-6)
        exp_diff = ctx.get(dut.exp_diff)
        shift_amt = ctx.get(dut.shift_amount)

        # Convert to signed
        if exp_diff & 0x200:  # Bit 9 set (negative)
            exp_diff_signed = exp_diff - 1024
        else:
            exp_diff_signed = exp_diff

        assert exp_diff_signed == 0, f"Expected 0, got {exp_diff_signed}"
        assert shift_amt == 0, f"Expected shift 0, got {shift_amt}"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()


def test_fused_exp_diff_product_larger():
    async def bench(ctx):
        # A=2.0 (128), B=2.0 (128), C=1.0 (127)
        # (128 + 128 - 127) - 127 = 129 - 127 = 2
        ctx.set(dut.a_exp, 128)
        ctx.set(dut.b_exp, 128)
        ctx.set(dut.c_exp, 127)
        await ctx.delay(1e-6)
        exp_diff = ctx.get(dut.exp_diff)
        shift_amt = ctx.get(dut.shift_amount)

        if exp_diff & 0x200:
            exp_diff_signed = exp_diff - 1024
        else:
            exp_diff_signed = exp_diff

        assert exp_diff_signed == 2, f"Expected 2, got {exp_diff_signed}"
        assert shift_amt == 2, f"Expected shift 2, got {shift_amt}"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()


def test_fused_exp_diff_c_larger():
    async def bench(ctx):
        # A=1.0 (127), B=1.0 (127), C=4.0 (129)
        # (127 + 127 - 127) - 129 = 127 - 129 = -2
        ctx.set(dut.a_exp, 127)
        ctx.set(dut.b_exp, 127)
        ctx.set(dut.c_exp, 129)
        await ctx.delay(1e-6)
        exp_diff = ctx.get(dut.exp_diff)
        shift_amt = ctx.get(dut.shift_amount)

        if exp_diff & 0x200:
            exp_diff_signed = exp_diff - 1024
        else:
            exp_diff_signed = exp_diff

        assert exp_diff_signed == -2, f"Expected -2, got {exp_diff_signed}"
        assert shift_amt == 2, f"Expected shift 2, got {shift_amt}"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()


def test_fused_exp_diff_large_difference():
    async def bench(ctx):
        # Large difference should clamp to 25
        ctx.set(dut.a_exp, 200)
        ctx.set(dut.b_exp, 200)
        ctx.set(dut.c_exp, 127)
        await ctx.delay(1e-6)
        shift_amt = ctx.get(dut.shift_amount)

        assert shift_amt == 25, f"Expected clamped shift 25, got {shift_amt}"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()
