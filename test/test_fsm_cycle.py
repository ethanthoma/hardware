"""Test to verify FSM cycles through all k values and enables PE correctly"""
import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def test_fsm_cycling():
    """Verify FSM goes through all MAC states"""
    dut = TensorCore8x8()

    async def bench(ctx):
        # Use all 1.0s so each MAC adds 1.0 to accumulator
        A_vals = [1.0] * 64
        B_vals = [1.0] * 64
        C_vals = [0.0] * 64

        A_flat = [BF16.from_float(v) for v in A_vals]
        B_flat = [BF16.from_float(v) for v in B_vals]
        C_flat = [BF16.from_float(v) for v in C_vals]

        for idx, bf16 in enumerate(A_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.a_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        for idx, bf16 in enumerate(B_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.b_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        for idx, bf16 in enumerate(C_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.c_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)

        print("\nWith all 1.0s, D[0,0] should increment by 1.0 each MAC cycle:")
        print("Expected final result: 8.0 (8 MACs of 1.0 * 1.0)")

        for cycle in range(15):
            done = ctx.get(dut.done)
            result_00 = ctx.get(dut.d_matrix[0])
            result_bf16 = BF16.pack(result_00["sign"], result_00["exponent"], result_00["mantissa"])
            result_val = result_bf16.to_float()

            print(f"Cycle {cycle}: done={done}, D[0,0]={result_val:.1f}")

            if done:
                break

            await ctx.tick()

        expected = 8.0
        print(f"\nFinal: {result_val:.1f}, Expected: {expected:.1f}")
        assert abs(result_val - expected) < 0.5, f"Expected {expected}, got {result_val}"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_fsm_cycling()
