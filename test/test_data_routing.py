"""Verify the tensor core data routing is correct"""
import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def test_data_routing_check():
    """Load a matrix where we can easily see which elements are being multiplied"""
    dut = TensorCore8x8()

    async def bench(ctx):
        # Create matrices where each element is its index * 10
        # A[i,j] = (i*8 + j) * 10
        # B[i,j] = (i*8 + j) * 10 + 1000
        # So we can tell which elements are being used

        A = np.zeros((8, 8), dtype=np.float32)
        B = np.zeros((8, 8), dtype=np.float32)

        for i in range(8):
            for j in range(8):
                A[i, j] = (i * 8 + j) * 10
                B[i, j] = (i * 8 + j) * 10 + 1000

        C = np.zeros((8, 8), dtype=np.float32)

        print("Matrix A:")
        print(A)
        print("\nMatrix B:")
        print(B)

        # For D[0,0], we expect:
        # k=0: A[0,0] * B[0,0] = 0 * 1000 = 0
        # k=1: A[0,1] * B[1,0] = 10 * 1080 = 10800
        # k=2: A[0,2] * B[2,0] = 20 * 1160 = 23200
        # ...
        # Sum should be huge and distinctive

        expected_00 = sum(A[0, k] * B[k, 0] for k in range(8))
        print(f"\nExpected D[0,0]: {expected_00}")

        # Load into hardware
        A_flat = []
        B_flat = []
        C_flat = []

        for i in range(8):
            for j in range(8):
                A_flat.append(BF16.from_float(A[i, j]))
                B_flat.append(BF16.from_float(B[i, j]))
                C_flat.append(BF16.from_float(C[i, j]))

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

        print(f"Hardware D[0,0]: {hw_result}")
        print(f"Match: {abs(hw_result - expected_00) < 100}")

        # Also check a few more elements
        for i, j in [(0, 1), (1, 0), (1, 1)]:
            expected = sum(A[i, k] * B[k, j] for k in range(8))
            idx = i * 8 + j
            result = ctx.get(dut.d_matrix[idx])
            result_bf16 = BF16.pack(result["sign"], result["exponent"], result["mantissa"])
            hw = result_bf16.to_float()
            print(f"D[{i},{j}]: HW={hw:.0f}, Expected={expected:.0f}, Match={abs(hw - expected) < 100}")

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_data_routing_check()
