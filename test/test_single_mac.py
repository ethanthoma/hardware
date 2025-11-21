"""Test a single MAC operation to isolate the issue"""
import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def test_single_mac_k3():
    """Test just the k=3 MAC to see if data routing works"""
    dut = TensorCore8x8()

    async def bench(ctx):
        # Set up matrices where we can easily verify k=3 operation
        # D[0,0] with k=3 should use A[0,3] and B[3,0]
        A_vals = [0.0] * 64
        B_vals = [0.0] * 64
        C_vals = [0.0] * 64

        # Set A[0,3] = 10.0
        A_vals[0*8 + 3] = 10.0
        # Set B[3,0] = 20.0
        B_vals[3*8 + 0] = 20.0
        # Set C to 100.0 so we can see the accumulation
        C_vals[0] = 100.0

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

        print("\nCycle-by-cycle for D[0,0]:")
        print("After LOAD_C, should see 100.0")
        print("After k=0,1,2 MACs (with 0 values), should still see 100.0")
        print("After k=3 MAC (A[0,3]=10 * B[3,0]=20), should see 100 + 200 = 300.0")

        for cycle in range(15):
            done = ctx.get(dut.done)
            result_00 = ctx.get(dut.d_matrix[0])
            result_bf16 = BF16.pack(result_00["sign"], result_00["exponent"], result_00["mantissa"])
            result_val = result_bf16.to_float()

            print(f"Cycle {cycle}: done={done}, D[0,0]={result_val:.1f}")

            if done:
                break

            await ctx.tick()

        print(f"\nExpected: 300.0")
        print(f"Got: {result_val:.1f}")
        assert abs(result_val - 300.0) < 1.0, f"Expected 300, got {result_val}"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_single_mac_k3()
