import sys

from amaranth.sim import Period, Simulator

import up_counter


def test_up_counter_overflow(request):
    dut = up_counter.UpCounter(25)

    async def bench(ctx):
        ctx.set(dut.en, 0)
        for _ in range(30):
            await ctx.tick()
            assert not ctx.get(dut.ovf)

        ctx.set(dut.en, 1)
        for _ in range(24):
            await ctx.tick()
            assert not ctx.get(dut.ovf)
        await ctx.tick()
        assert ctx.get(dut.ovf)

        await ctx.tick()
        assert not ctx.get(dut.ovf)

    sim = Simulator(dut)
    sim.add_clock(Period(MHz=1))
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"{dut.__class__.__name__}_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()
