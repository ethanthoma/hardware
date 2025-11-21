"""Check error distribution across all elements"""
import itertools
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


def bf16_flat_to_matrix(flat, rows, cols):
    matrix = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            matrix[i, j] = flat[idx]
    return matrix


def bf16_matmul(A, B, C):
    result = np.zeros_like(C)
    for i, j in itertools.product(range(A.shape[0]), range(B.shape[1])):
        acc = BF16.from_float(float(C[i, j])).to_float()
        for k in range(A.shape[1]):
            a_val = BF16.from_float(float(A[i, k])).to_float()
            b_val = BF16.from_float(float(B[k, j])).to_float()
            product = a_val * b_val
            acc_new = BF16.from_float(BF16.from_float(product).to_float() + acc).to_float()
            acc = acc_new
        result[i, j] = acc
    return result


def test_error_distribution():
    """Check how many elements pass with different error thresholds"""
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

        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)

        for _ in range(20):
            done = ctx.get(dut.done)
            if done:
                break
            await ctx.tick()

        result_flat = []
        for idx in range(64):
            result = ctx.get(dut.d_matrix[idx])
            result_bf16 = BF16.pack(result["sign"], result["exponent"], result["mantissa"])
            result_flat.append(result_bf16.to_float())

        result = bf16_flat_to_matrix(result_flat, 8, 8)
        expected = bf16_matmul(A, B, C)

        errors_by_threshold = {0.01: 0, 0.05: 0, 0.1: 0, 1.0: 0}
        large_errors = []

        for i, j in itertools.product(range(8), range(8)):
            error = abs(result[i, j] - expected[i, j])
            rel_error = error / abs(expected[i, j]) if abs(expected[i, j]) > 1e-6 else error

            for threshold in [0.01, 0.05, 0.1, 1.0]:
                if rel_error < threshold:
                    errors_by_threshold[threshold] += 1
                    break

            if rel_error >= 1.0:
                large_errors.append((i, j, result[i, j], expected[i, j], rel_error))

        print(f"\nError distribution (out of 64 elements):")
        print(f"  < 1% error: {errors_by_threshold[0.01]} elements")
        print(f"  < 5% error: {errors_by_threshold[0.05]} elements")
        print(f"  < 10% error: {errors_by_threshold[0.1]} elements")
        print(f"  >= 10% error: {64 - sum(errors_by_threshold.values())} elements")

        if large_errors:
            print(f"\nElements with large errors (>= 100%):")
            for i, j, hw, exp, err in large_errors[:10]:
                print(f"  D[{i},{j}]: HW={hw:.6f}, Expected={exp:.6f}, Error={err*100:.1f}%")

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_error_distribution()
