import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def test_fsm_progression():
    dut = TensorCore8x8()

    async def bench(ctx):
        # Simple test values
        A_vals = [1.0] * 64
        B_vals = [1.0] * 64
        C_vals = [0.0] * 64

        A_flat = [BF16.from_float(v) for v in A_vals]
        B_flat = [BF16.from_float(v) for v in B_vals]
        C_flat = [BF16.from_float(v) for v in C_vals]

        for idx, bf16 in enumerate(A_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.a_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        for idx, bf16 in enumerate(B_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.b_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        for idx, bf16 in enumerate(C_flat):
            sign, exp, mant = bf16.unpack()
            ctx.set(dut.c_matrix[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        print("\nFSM state progression:")
        print("Cycle | Done | D[0,0]")
        print("------|------|-------")

        ctx.set(dut.start, 1)
        await ctx.tick()

        ctx.set(dut.start, 0)

        for cycle in range(15):
            done = ctx.get(dut.done)
            result_00 = ctx.get(dut.d_matrix[0])
            result_bf16 = BF16.pack(result_00["sign"], result_00["exponent"], result_00["mantissa"])
            result_val = result_bf16.to_float()

            print(f"{cycle:5} | {done:4} | {result_val:.4f}")

            if done:
                break

            await ctx.tick()

        print(f"\nFinal D[0,0]: {result_val}")
        print(f"Expected (8 * 1.0 * 1.0): 8.0")

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_fsm_progression()
