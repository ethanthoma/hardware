import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def test_negative_accumulation():
    """Test with negative values like in the failing test"""
    dut = TensorCore8x8()

    async def bench(ctx):
        # Recreate the exact scenario from failing test k=0,1
        # A[0,0]=-0.200195, B[0,0]=-0.304688, prod=0.060997
        # A[0,1]=-0.149414, B[1,0]=0.345703, prod=-0.051653
        # Expected: 0.060997 + (-0.051653) ≈ 0.009344

        A_vals = [0.0] * 64
        B_vals = [0.0] * 64
        C_vals = [0.0] * 64

        # Set only the values needed for D[0,0] first two MACs
        A_vals[0*8 + 0] = -0.200195  # A[0,0]
        A_vals[0*8 + 1] = -0.149414  # A[0,1]

        B_vals[0*8 + 0] = -0.304688  # B[0,0]
        B_vals[1*8 + 0] = 0.345703   # B[1,0]

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

        print("\nTracking D[0,0] through MAC cycles:")
        for cycle in range(15):
            done = ctx.get(dut.done)
            result_00 = ctx.get(dut.d_matrix[0])
            result_bf16 = BF16.pack(result_00["sign"], result_00["exponent"], result_00["mantissa"])
            result_val = result_bf16.to_float()

            print(f"Cycle {cycle}: done={done}, D[0,0]={result_val:.6f}")

            if done:
                break

            await ctx.tick()

        expected_k0 = BF16.from_float(-0.200195 * -0.304688).to_float()
        expected_k1_add = BF16.from_float(-0.149414 * 0.345703).to_float()
        expected_total = BF16.from_float(expected_k0 + expected_k1_add).to_float()

        print(f"\nExpected after k=0: {expected_k0:.6f}")
        print(f"Expected k=1 product: {expected_k1_add:.6f}")
        print(f"Expected after k=1: {expected_total:.6f}")
        print(f"Hardware result: {result_val:.6f}")

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_negative_accumulation()
