import sys

from amaranth.sim import Simulator

import bf16_fma
import bfloat16


def test_fma_basic_operations(request):
    """Test basic FMA operations: (a * b) + c"""
    dut = bf16_fma.BF16_FMA()

    test_cases = [
        # (a, b, c, expected_result)
        # Basic: 2.0 * 3.0 + 1.0 = 7.0
        (2.0, 3.0, 1.0, 7.0),
        # Basic: 1.0 * 1.0 + 1.0 = 2.0
        (1.0, 1.0, 1.0, 2.0),
        # Basic: 4.0 * 2.0 + 1.0 = 9.0
        (4.0, 2.0, 1.0, 9.0),
        # With fractional: 1.5 * 2.0 + 0.5 = 3.5
        (1.5, 2.0, 0.5, 3.5),
        # Small values: 0.5 * 0.5 + 0.25 = 0.5
        (0.5, 0.5, 0.25, 0.5),
    ]

    async def bench(ctx):
        for a_f, b_f, c_f, expected_f in test_cases:
            a_packed = bfloat16.BFloat16.from_float(a_f)
            b_packed = bfloat16.BFloat16.from_float(b_f)
            c_packed = bfloat16.BFloat16.from_float(c_f)

            a_sign, a_exp, a_mant = bfloat16.BFloat16.unpack(a_packed)
            b_sign, b_exp, b_mant = bfloat16.BFloat16.unpack(b_packed)
            c_sign, c_exp, c_mant = bfloat16.BFloat16.unpack(c_packed)

            ctx.set(dut.a, {"sign": a_sign, "exponent": a_exp, "mantissa": a_mant})
            ctx.set(dut.b, {"sign": b_sign, "exponent": b_exp, "mantissa": b_mant})
            ctx.set(dut.c, {"sign": c_sign, "exponent": c_exp, "mantissa": c_mant})

            result_struct = ctx.get(dut.result)
            result_packed = bfloat16.BFloat16.pack(
                result_struct["sign"], result_struct["exponent"], result_struct["mantissa"]
            )
            result_f = bfloat16.BFloat16.to_float(result_packed)

            error = abs(result_f - expected_f)
            rel_error = error / abs(expected_f) if expected_f != 0 else error

            assert rel_error < 0.01, (
                f"FMA({a_f} * {b_f} + {c_f}): got {result_f}, expected {expected_f}, error={rel_error:.6f}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"BF16_FMA_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_fma_with_zero(request):
    """Test FMA with zero operands"""
    dut = bf16_fma.BF16_FMA()

    test_cases = [
        # (a, b, c, expected_result)
        # Zero multiplication: 0.0 * 5.0 + 3.0 = 3.0
        (0.0, 5.0, 3.0, 3.0),
        # Zero multiplication: 5.0 * 0.0 + 3.0 = 3.0
        (5.0, 0.0, 3.0, 3.0),
        # Zero addend: 2.0 * 3.0 + 0.0 = 6.0
        (2.0, 3.0, 0.0, 6.0),
        # All zeros: 0.0 * 0.0 + 0.0 = 0.0
        (0.0, 0.0, 0.0, 0.0),
        # Product equals -c: 2.0 * 1.0 + (-2.0) = 0.0
        (2.0, 1.0, -2.0, 0.0),
    ]

    async def bench(ctx):
        for a_f, b_f, c_f, expected_f in test_cases:
            a_packed = bfloat16.BFloat16.from_float(a_f)
            b_packed = bfloat16.BFloat16.from_float(b_f)
            c_packed = bfloat16.BFloat16.from_float(c_f)

            a_sign, a_exp, a_mant = bfloat16.BFloat16.unpack(a_packed)
            b_sign, b_exp, b_mant = bfloat16.BFloat16.unpack(b_packed)
            c_sign, c_exp, c_mant = bfloat16.BFloat16.unpack(c_packed)

            ctx.set(dut.a, {"sign": a_sign, "exponent": a_exp, "mantissa": a_mant})
            ctx.set(dut.b, {"sign": b_sign, "exponent": b_exp, "mantissa": b_mant})
            ctx.set(dut.c, {"sign": c_sign, "exponent": c_exp, "mantissa": c_mant})

            result_struct = ctx.get(dut.result)
            result_packed = bfloat16.BFloat16.pack(
                result_struct["sign"], result_struct["exponent"], result_struct["mantissa"]
            )
            result_f = bfloat16.BFloat16.to_float(result_packed)

            error = abs(result_f - expected_f)

            assert error < 0.01, f"FMA({a_f} * {b_f} + {c_f}): got {result_f}, expected {expected_f}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"BF16_FMA_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_fma_with_negative(request):
    """Test FMA with negative operands"""
    dut = bf16_fma.BF16_FMA()

    test_cases = [
        # (a, b, c, expected_result)
        # Negative a: -2.0 * 3.0 + 1.0 = -5.0
        (-2.0, 3.0, 1.0, -5.0),
        # Negative b: 2.0 * -3.0 + 1.0 = -5.0
        (2.0, -3.0, 1.0, -5.0),
        # Negative c: 2.0 * 3.0 + (-1.0) = 5.0
        (2.0, 3.0, -1.0, 5.0),
        # Two negatives: -2.0 * -3.0 + 1.0 = 7.0
        (-2.0, -3.0, 1.0, 7.0),
        # All negative: -2.0 * -3.0 + (-1.0) = 5.0
        (-2.0, -3.0, -1.0, 5.0),
    ]

    async def bench(ctx):
        for a_f, b_f, c_f, expected_f in test_cases:
            a_packed = bfloat16.BFloat16.from_float(a_f)
            b_packed = bfloat16.BFloat16.from_float(b_f)
            c_packed = bfloat16.BFloat16.from_float(c_f)

            a_sign, a_exp, a_mant = bfloat16.BFloat16.unpack(a_packed)
            b_sign, b_exp, b_mant = bfloat16.BFloat16.unpack(b_packed)
            c_sign, c_exp, c_mant = bfloat16.BFloat16.unpack(c_packed)

            ctx.set(dut.a, {"sign": a_sign, "exponent": a_exp, "mantissa": a_mant})
            ctx.set(dut.b, {"sign": b_sign, "exponent": b_exp, "mantissa": b_mant})
            ctx.set(dut.c, {"sign": c_sign, "exponent": c_exp, "mantissa": c_mant})

            result_struct = ctx.get(dut.result)
            result_packed = bfloat16.BFloat16.pack(
                result_struct["sign"], result_struct["exponent"], result_struct["mantissa"]
            )
            result_f = bfloat16.BFloat16.to_float(result_packed)

            error = abs(result_f - expected_f)
            rel_error = error / abs(expected_f) if expected_f != 0 else error

            assert rel_error < 0.01, f"FMA({a_f} * {b_f} + {c_f}): got {result_f}, expected {expected_f}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"BF16_FMA_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_fma_alignment_cases(request):
    """Test FMA cases that require mantissa alignment"""
    dut = bf16_fma.BF16_FMA()

    test_cases = [
        # (a, b, c, expected_result)
        # Large product, small c: 8.0 * 8.0 + 1.0 = 65.0
        (8.0, 8.0, 1.0, 65.0),
        # Small product, large c: 1.0 * 1.0 + 64.0 = 65.0
        (1.0, 1.0, 64.0, 65.0),
        # Medium difference: 4.0 * 4.0 + 2.0 = 18.0
        (4.0, 4.0, 2.0, 18.0),
        # Large difference: 16.0 * 16.0 + 1.0 = 257.0
        (16.0, 16.0, 1.0, 257.0),
    ]

    async def bench(ctx):
        for a_f, b_f, c_f, expected_f in test_cases:
            a_packed = bfloat16.BFloat16.from_float(a_f)
            b_packed = bfloat16.BFloat16.from_float(b_f)
            c_packed = bfloat16.BFloat16.from_float(c_f)

            a_sign, a_exp, a_mant = bfloat16.BFloat16.unpack(a_packed)
            b_sign, b_exp, b_mant = bfloat16.BFloat16.unpack(b_packed)
            c_sign, c_exp, c_mant = bfloat16.BFloat16.unpack(c_packed)

            ctx.set(dut.a, {"sign": a_sign, "exponent": a_exp, "mantissa": a_mant})
            ctx.set(dut.b, {"sign": b_sign, "exponent": b_exp, "mantissa": b_mant})
            ctx.set(dut.c, {"sign": c_sign, "exponent": c_exp, "mantissa": c_mant})

            result_struct = ctx.get(dut.result)
            result_packed = bfloat16.BFloat16.pack(
                result_struct["sign"], result_struct["exponent"], result_struct["mantissa"]
            )
            result_f = bfloat16.BFloat16.to_float(result_packed)

            error = abs(result_f - expected_f)
            rel_error = error / abs(expected_f) if expected_f != 0 else error

            assert rel_error < 0.01, (
                f"FMA({a_f} * {b_f} + {c_f}): got {result_f}, expected {expected_f}, rel_error={rel_error:.6f}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"BF16_FMA_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_fma_edge_cases(request):
    """Test FMA edge cases"""
    dut = bf16_fma.BF16_FMA()

    test_cases = [
        # (a, b, c, expected_result)
        # Very small values
        (0.125, 0.125, 0.0625, 0.078125),
        # Powers of 2 (exact in BF16)
        (0.5, 0.5, 0.5, 0.75),
        (1.0, 2.0, 4.0, 6.0),
        # Identity: 1.0 * x + 0.0 = x
        (1.0, 5.0, 0.0, 5.0),
    ]

    async def bench(ctx):
        for a_f, b_f, c_f, expected_f in test_cases:
            a_packed = bfloat16.BFloat16.from_float(a_f)
            b_packed = bfloat16.BFloat16.from_float(b_f)
            c_packed = bfloat16.BFloat16.from_float(c_f)

            a_sign, a_exp, a_mant = bfloat16.BFloat16.unpack(a_packed)
            b_sign, b_exp, b_mant = bfloat16.BFloat16.unpack(b_packed)
            c_sign, c_exp, c_mant = bfloat16.BFloat16.unpack(c_packed)

            ctx.set(dut.a, {"sign": a_sign, "exponent": a_exp, "mantissa": a_mant})
            ctx.set(dut.b, {"sign": b_sign, "exponent": b_exp, "mantissa": b_mant})
            ctx.set(dut.c, {"sign": c_sign, "exponent": c_exp, "mantissa": c_mant})

            result_struct = ctx.get(dut.result)
            result_packed = bfloat16.BFloat16.pack(
                result_struct["sign"], result_struct["exponent"], result_struct["mantissa"]
            )
            result_f = bfloat16.BFloat16.to_float(result_packed)

            error = abs(result_f - expected_f)
            rel_error = error / abs(expected_f) if expected_f != 0 else error

            assert rel_error < 0.01, f"FMA({a_f} * {b_f} + {c_f}): got {result_f}, expected {expected_f}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"BF16_FMA_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()
