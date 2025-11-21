import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def test_tensor_core_simple_known_values():
    """Test with simple known values to debug the issue"""
    dut = TensorCore8x8()

    async def bench(ctx):
        # Simple test: 2×2 submatrix in top-left, rest zeros
        # A = [[1, 2, 0...], [3, 4, 0...], [0...]]
        # B = [[5, 6, 0...], [7, 8, 0...], [0...]]
        # Expected: A @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        #                 = [[19, 22], [43, 50]]

        A_vals = [
            1, 2, 0, 0, 0, 0, 0, 0,
            3, 4, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ]

        B_vals = [
            5, 6, 0, 0, 0, 0, 0, 0,
            7, 8, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ]

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

        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)

        for _ in range(20):
            done = ctx.get(dut.done)
            if done:
                break
            await ctx.tick()

        assert done, "Computation did not complete"

        # Check the 2×2 top-left results
        expected_results = {
            (0, 0): 1*5 + 2*7,  # 19
            (0, 1): 1*6 + 2*8,  # 22
            (1, 0): 3*5 + 4*7,  # 43
            (1, 1): 3*6 + 4*8,  # 50
        }

        print("\nResults for 2×2 top-left submatrix:")
        for i in range(2):
            for j in range(2):
                idx = i * 8 + j
                result = ctx.get(dut.d_matrix[idx])
                result_bf16 = BF16.pack(result["sign"], result["exponent"], result["mantissa"])
                hw_result = result_bf16.to_float()
                expected = expected_results[(i, j)]

                print(f"D[{i},{j}]: HW={hw_result}, Expected={expected}, Match={abs(hw_result - expected) < 0.01}")

                if abs(hw_result - expected) >= 0.01:
                    # Debug: show intermediate values
                    print(f"  A[{i},0]={A_vals[i*8+0]}, A[{i},1]={A_vals[i*8+1]}")
                    print(f"  B[0,{j}]={B_vals[0*8+j]}, B[1,{j}]={B_vals[1*8+j]}")
                    print(f"  Expected: {A_vals[i*8+0]}*{B_vals[0*8+j]} + {A_vals[i*8+1]}*{B_vals[1*8+j]} = {expected}")

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_tensor_core_simple_known_values()
