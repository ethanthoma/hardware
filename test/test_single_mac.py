from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from mma import MMA


def test_k3_routes_a03_b30_into_d00():
    dut = MMA()

    async def bench(ctx):
        A_vals = [0.0] * 16
        B_vals = [0.0] * 16
        A_vals[0 * 4 + 3] = 10.0
        B_vals[3 * 4 + 0] = 20.0

        for idx, v in enumerate(A_vals):
            s, e, m = BF16.from_float(v).unpack()
            ctx.set(dut.a_matrix[idx], {"sign": s, "exponent": e, "mantissa": m})
        for idx, v in enumerate(B_vals):
            s, e, m = BF16.from_float(v).unpack()
            ctx.set(dut.b_matrix[idx], {"sign": s, "exponent": e, "mantissa": m})

        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)

        for _ in range(15):
            if ctx.get(dut.done):
                break
            await ctx.tick()
        assert ctx.get(dut.done), "MMA did not complete"

        r = ctx.get(dut.d_matrix[0])
        result = BF16.pack(r["sign"], r["exponent"], r["mantissa"]).to_float()
        assert result == 200.0, f"expected 200.0, got {result}"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()
