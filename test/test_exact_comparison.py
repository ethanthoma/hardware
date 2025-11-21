import itertools
import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def matrix_to_bf16_flat(matrix):
    flat = []
    rows, cols = matrix.shape
    for i, j in itertools.product(range(rows), range(cols)):
        flat.append(BF16.from_float(float(matrix[i, j])))
    return flat


def test_exact_seed_456():
    """Use exact same seed as failing test and debug"""
    dut = TensorCore8x8()

    async def bench(ctx):
        np.random.seed(456)
        A = np.random.randn(8, 8).astype(np.float32) * 0.3
        B = np.random.randn(8, 8).astype(np.float32) * 0.3
        C = np.zeros((8, 8), dtype=np.float32)

        print("A matrix (first row, BF16 quantized):")
        for j in range(8):
            val = BF16.from_float(A[0, j]).to_float()
            print(f"  A[0,{j}] = {val:.6f}")

        print("\nB matrix (first column, BF16 quantized):")
        for i in range(8):
            val = BF16.from_float(B[i, 0]).to_float()
            print(f"  B[{i},0] = {val:.6f}")

        # Manually compute D[0,0] step by step
        print("\nManual D[0,0] computation:")
        acc = 0.0
        for k in range(8):
            a_bf16 = BF16.from_float(A[0, k]).to_float()
            b_bf16 = BF16.from_float(B[k, 0]).to_float()
            # FMA: (a × b) + acc
            prod = a_bf16 * b_bf16
            fma_f64 = np.float64(prod) + np.float64(acc)
            acc = BF16.from_float(float(fma_f64)).to_float()
            print(f"  k={k}: A={a_bf16:.6f}, B={b_bf16:.6f}, prod={prod:.6f}, FMA result={acc:.6f}")

        print(f"\nSoftware D[0,0]: {acc:.6f}")

        # Now test hardware
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

        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)

        for _ in range(20):
            done = ctx.get(dut.done)
            if done:
                break
            await ctx.tick()

        result_00 = ctx.get(dut.d_matrix[0])
        result_bf16 = BF16.pack(result_00["sign"], result_00["exponent"], result_00["mantissa"])
        hw_result = result_bf16.to_float()

        print(f"Hardware D[0,0]: {hw_result:.6f}")
        print(f"\nDifference: {abs(hw_result - acc):.6f}")

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_exact_seed_456()
