import sys

from amaranth.sim import Period, Simulator

import bf16_multiplier
from bfloat16 import BFloat16


def test_bf16_multiplier_basic(request):
    dut = bf16_multiplier.BF16Multiplier()

    test_cases = [
        (1.0, 1.0, 1.0),
        (2.0, 3.0, 6.0),
        (0.5, 0.5, 0.25),
        (1.5, 2.0, 3.0),
        (-1.0, 2.0, -2.0),
        (-2.0, -3.0, 6.0),
        (4.0, 0.25, 1.0),
        (10.0, 10.0, 100.0),
    ]

    async def bench(ctx):
        for a_val, b_val, expected in test_cases:
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

            rel_error = abs(result_float - expected) / max(abs(expected), 1e-6)
            assert rel_error < 0.02, f"Expected {expected}, got {result_float} (error: {rel_error:.4f})"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"{dut.__class__.__name__}_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_bf16_multiplier_edge_cases(request):
    dut = bf16_multiplier.BF16Multiplier()

    test_cases = [
        (0.0, 5.0, 0.0),  # zero * value = zero
        (5.0, 0.0, 0.0),  # value * zero = zero
        (1.0, -1.0, -1.0),  # sign change
        (-1.0, -1.0, 1.0),  # negative * negative
        (0.125, 8.0, 1.0),  # small * large
        (16.0, 16.0, 256.0),  # larger values
    ]

    async def bench(ctx):
        for a_val, b_val, expected in test_cases:
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

            # zero is special case that we dont handle yet
            if expected == 0.0:
                assert abs(result_float) < 1e-6, f"Expected 0, got {result_float}"
            else:
                rel_error = abs(result_float - expected) / abs(expected)
                assert rel_error < 0.02, f"Expected {expected}, got {result_float} (error: {rel_error:.4f})"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"{dut.__class__.__name__}_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()
