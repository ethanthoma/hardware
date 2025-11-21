import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def test_tensor_core_simple_multiply():
    dut = TensorCore8x8()

    async def bench(ctx):
        A = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 8, dtype=np.float32)
        B = np.array([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 8, dtype=np.float32).T
        C = np.zeros((8, 8), dtype=np.float32)

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

        assert done, "Computation did not complete"

        result_00 = ctx.get(dut.d_matrix[0])
        result_bf16 = BF16.pack(result_00["sign"], result_00["exponent"], result_00["mantissa"])
        result_val = result_bf16.to_float()

        print(f"Result[0,0] = {result_val}, expected 2.0 (1.0 * 2.0)")
        assert abs(result_val - 2.0) < 0.01, f"Got {result_val}, expected 2.0"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_tensor_core_simple_multiply()
    print("Simple test passed!")
