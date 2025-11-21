"""Debug D[0,3] specifically"""
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


def test_d03_debug():
    """Debug D[0,3] computation"""
    dut = TensorCore8x8()

    async def bench(ctx):
        np.random.seed(456)
        A = np.random.randn(8, 8).astype(np.float32) * 0.3
        B = np.random.randn(8, 8).astype(np.float32) * 0.3
        C = np.zeros((8, 8), dtype=np.float32)

        # D[0,3] = sum(A[0,k] * B[k,3] for k in range(8))
        print("\nExpected D[0,3] computation:")
        acc = 0.0
        for k in range(8):
            a_val = BF16.from_float(A[0, k]).to_float()
            b_val = BF16.from_float(B[k, 3]).to_float()
            prod = a_val * b_val
            acc_new = BF16.from_float(BF16.from_float(prod).to_float() + acc).to_float()
            print(f"  k={k}: A[0,{k}]={a_val:.6f}, B[{k},3]={b_val:.6f}, prod={prod:.6f}, acc={acc_new:.6f}")
            acc = acc_new

        print(f"Expected D[0,3]: {acc:.6f}")

        # Load matrices into hardware
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

        print("\nHardware D[0,3] cycle-by-cycle:")
        for cycle in range(15):
            done = ctx.get(dut.done)
            result_03 = ctx.get(dut.d_matrix[3])  # D[0,3] is at index 3 (row 0, col 3)
            result_bf16 = BF16.pack(result_03["sign"], result_03["exponent"], result_03["mantissa"])
            hw_val = result_bf16.to_float()

            print(f"  Cycle {cycle}: done={done}, D[0,3]={hw_val:.6f}")

            if done:
                break

            await ctx.tick()

        print(f"\nFinal hardware D[0,3]: {hw_val:.6f}")
        print(f"Expected: {acc:.6f}")
        print(f"Difference: {abs(hw_val - acc):.6f}")

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_d03_debug()
