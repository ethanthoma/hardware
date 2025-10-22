from amaranth.sim import Period, Simulator

import up_counter

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


if __name__ == "__main__":
    sim = Simulator(dut)
    sim.add_clock(Period(MHz=1))
    sim.add_testbench(bench)
    with sim.write_vcd("up_counter.vcd"):
        sim.run()
