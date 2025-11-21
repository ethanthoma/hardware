"""Test to verify the k signal progression during MAC states"""
import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def test_k_signal_progression():
    """Trace k signal and see which A/B values are being used"""
    dut = TensorCore8x8()

    async def bench(ctx):
        # Use simple incrementing values so we can see which are used
        A_vals = list(range(64))  # 0, 1, 2, ..., 63
        B_vals = list(range(100, 164))  # 100, 101, 102, ..., 163
        C_vals = [0.0] * 64

        A_flat = [BF16.from_float(float(v)) for v in A_vals]
        B_flat = [BF16.from_float(float(v)) for v in B_vals]
        C_flat = [BF16.from_float(float(v)) for v in C_vals]

        for idx, bf16 in enumerate(A_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.a_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        for idx, bf16 in enumerate(B_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.b_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        for idx, bf16 in enumerate(C_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.c_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        print("\nFor D[0,0], we expect:")
        print("k=0: A[0,0]=0, B[0,0]=100, product=0")
        print("k=1: A[0,1]=1, B[1,0]=108, product=108")
        print("k=2: A[0,2]=2, B[2,0]=116, product=232")
        print("k=3: A[0,3]=3, B[3,0]=124, product=372")
        print("k=4: A[0,4]=4, B[4,0]=132, product=528")
        print("k=5: A[0,5]=5, B[5,0]=140, product=700")
        print("k=6: A[0,6]=6, B[6,0]=148, product=888")
        print("k=7: A[0,7]=7, B[7,0]=156, product=1092")
        expected_sum = sum(i * (100 + i*8) for i in range(8))
        print(f"Expected sum: {expected_sum}")

        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)

        print("\nCycle-by-cycle progression:")
        for cycle in range(15):
            done = ctx.get(dut.done)
            result_00 = ctx.get(dut.d_matrix[0])
            result_bf16 = BF16.pack(result_00["sign"], result_00["exponent"], result_00["mantissa"])
            result_val = result_bf16.to_float()

            print(f"Cycle {cycle}: done={done}, D[0,0]={result_val:.0f}")

            if done:
                break

            await ctx.tick()

        print(f"\nFinal D[0,0]: {result_val:.0f}")
        print(f"Expected: {expected_sum}")
        print(f"Match: {abs(result_val - expected_sum) < 10}")

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_k_signal_progression()
