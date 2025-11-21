import itertools
import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def bf16_matmul(A, B, C):
    A_bf16 = np.array(
        [[BF16.from_float(float(A[i, j])).to_float() for j in range(A.shape[1])] for i in range(A.shape[0])],
        dtype=np.float32,
    )
    B_bf16 = np.array(
        [[BF16.from_float(float(B[i, j])).to_float() for j in range(B.shape[1])] for i in range(B.shape[0])],
        dtype=np.float32,
    )
    C_bf16 = np.array(
        [[BF16.from_float(float(C[i, j])).to_float() for j in range(C.shape[1])] for i in range(C.shape[0])],
        dtype=np.float32,
    )

    result = np.zeros_like(C_bf16)
    for i, j in itertools.product(range(A.shape[0]), range(B.shape[1])):
        acc = C_bf16[i, j]
        for k in range(A.shape[1]):
            a_f64 = np.float64(A_bf16[i, k])
            b_f64 = np.float64(B_bf16[k, j])
            c_f64 = np.float64(acc)
            fma_result_f64 = (a_f64 * b_f64) + c_f64
            acc = BF16.from_float(float(fma_result_f64)).to_float()
        result[i, j] = acc
    return result


def test_debug_random():
    dut = TensorCore8x8()

    async def bench(ctx):
        np.random.seed(456)
        A = np.random.randn(8, 8).astype(np.float32) * 0.3
        B = np.random.randn(8, 8).astype(np.float32) * 0.3
        C = np.zeros((8, 8), dtype=np.float32)

        # Print first few values
        print("\nInput matrices (first 2x2):")
        print(f"A[0:2, 0:2] = \n{A[0:2, 0:2]}")
        print(f"B[0:2, 0:2] = \n{B[0:2, 0:2]}")

        A_flat = [BF16.from_float(float(A[i, j])) for i in range(8) for j in range(8)]
        B_flat = [BF16.from_float(float(B[i, j])) for i in range(8) for j in range(8)]
        C_flat = [BF16.from_float(float(C[i, j])) for i in range(8) for j in range(8)]

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

        # Get hardware result
        result_flat = []
        for idx in range(64):
            result = ctx.get(dut.d_matrix[idx])
            bf16 = BF16.pack(result["sign"], result["exponent"], result["mantissa"])
            result_flat.append(bf16)

        hw_result = np.array([[result_flat[i*8+j].to_float() for j in range(8)] for i in range(8)])

        # Get reference result
        ref_result = bf16_matmul(A, B, C)

        print("\nD[0,0] comparison:")
        print(f"HW result: {hw_result[0, 0]}")
        print(f"Ref result: {ref_result[0, 0]}")
        print(f"Error: {abs(hw_result[0, 0] - ref_result[0, 0])}")

        # Manually compute D[0,0]
        print("\nManual computation of D[0,0]:")
        manual_acc = 0.0
        for k in range(8):
            a_val = BF16.from_float(A[0, k]).to_float()
            b_val = BF16.from_float(B[k, 0]).to_float()
            prod = a_val * b_val
            manual_acc += prod
            print(f"  k={k}: A[0,{k}]={a_val:.6f}, B[{k},0]={b_val:.6f}, prod={prod:.6f}, acc={manual_acc:.6f}")

        manual_result = BF16.from_float(manual_acc).to_float()
        print(f"Manual result (direct sum): {manual_result}")

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_debug_random()
