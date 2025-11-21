"""Trace seed 456 test cycle by cycle"""
import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def matrix_to_bf16_flat(matrix):
    flat = []
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            flat.append(BF16.from_float(float(matrix[i, j])))
    return flat


def test_seed456_trace():
    """Trace exact seed 456 test to see what's happening"""
    dut = TensorCore8x8()

    async def bench(ctx):
        np.random.seed(456)
        A = np.random.randn(8, 8).astype(np.float32) * 0.3
        B = np.random.randn(8, 8).astype(np.float32) * 0.3
        C = np.zeros((8, 8), dtype=np.float32)

        A_flat = matrix_to_bf16_flat(A)
        B_flat = matrix_to_bf16_flat(B)
        C_flat = matrix_to_bf16_flat(C)

        for idx, bf16 in enumerate(A_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.a_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        for idx, bf16 in enumerate(B_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.b_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        for idx, bf16 in enumerate(C_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.c_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        # Print expected values for D[0,0]
        print("\nExpected D[0,0] computation (from software):")
        acc = 0.0
        for k in range(8):
            a_val = BF16.from_float(A[0, k]).to_float()
            b_val = BF16.from_float(B[k, 0]).to_float()
            prod = a_val * b_val
            acc_new = BF16.from_float(BF16.from_float(prod).to_float() + acc).to_float()
            print(f"  k={k}: A={a_val:.6f}, B={b_val:.6f}, prod={prod:.6f}, acc={acc_new:.6f}")
            acc = acc_new

        print(f"\nExpected final D[0,0]: {acc:.6f}")

        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)

        print("\nHardware cycle-by-cycle:")
        for cycle in range(15):
            done = ctx.get(dut.done)
            result_00 = ctx.get(dut.d_matrix[0])
            result_bf16 = BF16.pack(result_00["sign"], result_00["exponent"], result_00["mantissa"])
            hw_val = result_bf16.to_float()

            print(f"  Cycle {cycle}: done={done}, D[0,0]={hw_val:.6f}")

            if done:
                break

            await ctx.tick()

        print(f"\nFinal hardware D[0,0]: {hw_val:.6f}")
        print(f"Expected: {acc:.6f}")
        print(f"Difference: {abs(hw_val - acc):.6f}")

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_seed456_trace()
