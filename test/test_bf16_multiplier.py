import sys

from amaranth.sim import Simulator

import bf16_multiplier
from bfloat16 import BFloat16


def test_bf16_multiplier(request):
    dut = bf16_multiplier.BF16Multiplier()

    test_cases = [
        # Basic operations
        (1.0, 1.0, 1.0, "one * one"),
        (2.0, 3.0, 6.0, "basic multiply"),
        (0.5, 0.5, 0.25, "fraction multiply"),
        (1.5, 2.0, 3.0, "mixed multiply"),
        (4.0, 0.25, 1.0, "inverse multiply"),
        (10.0, 10.0, 100.0, "larger values"),
        # Sign handling
        (-1.0, 2.0, -2.0, "negative * positive"),
        (-2.0, -3.0, 6.0, "negative * negative"),
        (1.0, -1.0, -1.0, "positive * negative"),
        (-1.0, -1.0, 1.0, "neg * neg = pos"),
        # Edge cases
        (0.125, 8.0, 1.0, "small * large"),
        (16.0, 16.0, 256.0, "power of two"),
        # Zero cases
        (0.0, 0.0, 0.0, "+zero * +zero"),
        (-0.0, 0.0, 0.0, "-zero * +zero"),
        (0.0, -0.0, 0.0, "+zero * -zero"),
        (-0.0, -0.0, 0.0, "-zero * -zero"),
        (0.0, 1.0, 0.0, "zero * one"),
        (1.0, 0.0, 0.0, "one * zero"),
        (0.0, 5.0, 0.0, "zero * value"),
        (5.0, 0.0, 0.0, "value * zero"),
        (0.0, -5.5, 0.0, "zero * negative"),
        (-0.0, 33333.0, 0.0, "-zero * large"),
        (0.0, 0.125, 0.0, "zero * small"),
    ]

    async def bench(ctx):
        for a_val, b_val, expected, desc in test_cases:
            a_bits = BFloat16.from_float(a_val)
            b_bits = BFloat16.from_float(b_val)

            a_sign, a_exp, a_mant = BFloat16.unpack(a_bits)
            b_sign, b_exp, b_mant = BFloat16.unpack(b_bits)

            ctx.set(dut.a.sign, a_sign)
            ctx.set(dut.a.exponent, a_exp)
            ctx.set(dut.a.mantissa, a_mant)
            ctx.set(dut.b.sign, b_sign)
            ctx.set(dut.b.exponent, b_exp)
            ctx.set(dut.b.mantissa, b_mant)

            result_sign = ctx.get(dut.result.sign)
            result_exp = ctx.get(dut.result.exponent)
            result_mant = ctx.get(dut.result.mantissa)

            result_bits = BFloat16.pack(result_sign, result_exp, result_mant)
            result_float = BFloat16.to_float(result_bits)

            rel_error = abs(result_float - expected)
            assert abs(rel_error) < 1e-10, f"{desc}: Expected {expected}, got {result_float} (error: {rel_error:.4f})"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"{dut.__class__.__name__}_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()
