import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def test_tensor_core_timing():
    dut = TensorCore8x8()

    async def bench(ctx):
        A_np = np.array([[0.1] * 8] * 8, dtype=np.float32)
        B_np = np.array([[0.1] * 8] * 8, dtype=np.float32)
        C_np = np.zeros((8, 8), dtype=np.float32)

        A_flat = [BF16.from_float(float(A_np[i, j])) for i in range(8) for j in range(8)]
        B_flat = [BF16.from_float(float(B_np[i, j])) for i in range(8) for j in range(8)]
        C_flat = [BF16.from_float(float(C_np[i, j])) for i in range(8) for j in range(8)]

        for idx, bf16 in enumerate(A_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.a_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        for idx, bf16 in enumerate(B_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.b_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        for idx, bf16 in enumerate(C_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.c_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        print("Initial state - setting start=1")
        ctx.set(dut.start, 1)
        await ctx.tick()

        print("Cycle 1 - start=0")
        ctx.set(dut.start, 0)

        for cycle in range(25):
            done = ctx.get(dut.done)
            result_00 = ctx.get(dut.d_matrix[0])
            result_bf16 = BF16.pack(result_00["sign"], result_00["exponent"], result_00["mantissa"])
            result_val = result_bf16.to_float()

            print(f"Cycle {cycle+1}: done={done}, result[0,0]={result_val}")

            if done:
                break
            await ctx.tick()

        print(f"\nFinal result[0,0]: {result_val}")
        print(f"Expected: 8 * 0.1 * 0.1 = 0.08")

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_tensor_core_timing()
