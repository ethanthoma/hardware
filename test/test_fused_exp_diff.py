from amaranth.sim import Simulator

from fused_exp_diff import FusedExponentDifference

dut = FusedExponentDifference()


def test_fused_exp_diff_equal_exponents():
    async def bench(ctx):
        ctx.set(dut.a_exp, 127)
        ctx.set(dut.b_exp, 127)
        ctx.set(dut.c_exp, 127)
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
        ctx.set(dut.a_exp, 128)
        ctx.set(dut.b_exp, 128)
        ctx.set(dut.c_exp, 127)
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
        ctx.set(dut.a_exp, 127)
        ctx.set(dut.b_exp, 127)
        ctx.set(dut.c_exp, 129)
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
        ctx.set(dut.a_exp, 200)
        ctx.set(dut.b_exp, 200)
        ctx.set(dut.c_exp, 127)
        shift_amt = ctx.get(dut.shift_amount)

        assert shift_amt == 25, f"Expected clamped shift 25, got {shift_amt}"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()
