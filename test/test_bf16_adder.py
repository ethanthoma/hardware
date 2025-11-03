import sys

from amaranth.sim import Simulator

import bf16_adder
import bfloat16


def test_adder_basic_addition(request):
    """Test basic addition: a + b"""
    dut = bf16_adder.BF16Adder()

    test_cases = [
        # (a, b, expected_result)
        # Same exponent additions
        (1.0, 1.0, 2.0),
        (2.0, 3.0, 5.0),
        (4.0, 4.0, 8.0),
        # Powers of 2 (exact in BF16)
        (0.5, 0.5, 1.0),
        (0.25, 0.25, 0.5),
        (8.0, 8.0, 16.0),
        # Small values
        (0.125, 0.125, 0.25),
    ]

    async def bench(ctx):
        for a_f, b_f, expected_f in test_cases:
            a_packed = bfloat16.BFloat16.from_float(a_f)
            b_packed = bfloat16.BFloat16.from_float(b_f)

            a_sign, a_exp, a_mant = bfloat16.BFloat16.unpack(a_packed)
            b_sign, b_exp, b_mant = bfloat16.BFloat16.unpack(b_packed)

            ctx.set(dut.a, {"sign": a_sign, "exponent": a_exp, "mantissa": a_mant})
            ctx.set(dut.b, {"sign": b_sign, "exponent": b_exp, "mantissa": b_mant})

            result_struct = ctx.get(dut.result)
            result_packed = bfloat16.BFloat16.pack(
                result_struct["sign"], result_struct["exponent"], result_struct["mantissa"]
            )
            result_f = bfloat16.BFloat16.to_float(result_packed)

            error = abs(result_f - expected_f)
            rel_error = error / abs(expected_f) if expected_f != 0 else error

            assert rel_error < 0.01, f"Add({a_f} + {b_f}): got {result_f}, expected {expected_f}, error={rel_error:.6f}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"BF16Adder_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_adder_different_exponents(request):
    """Test addition with different exponents (requires alignment)"""
    dut = bf16_adder.BF16Adder()

    test_cases = [
        # (a, b, expected_result)
        # Small + large
        (1.0, 100.0, 101.0),
        (0.5, 10.0, 10.5),
        (0.125, 8.0, 8.125),
        # Large difference (c gets shifted out)
        (1.0, 1000.0, 1000.0),  # 1.0 is too small to matter
        # Medium differences
        (2.0, 0.5, 2.5),
        (4.0, 1.0, 5.0),
        (16.0, 1.0, 17.0),
    ]

    async def bench(ctx):
        for a_f, b_f, expected_f in test_cases:
            a_packed = bfloat16.BFloat16.from_float(a_f)
            b_packed = bfloat16.BFloat16.from_float(b_f)

            a_sign, a_exp, a_mant = bfloat16.BFloat16.unpack(a_packed)
            b_sign, b_exp, b_mant = bfloat16.BFloat16.unpack(b_packed)

            ctx.set(dut.a, {"sign": a_sign, "exponent": a_exp, "mantissa": a_mant})
            ctx.set(dut.b, {"sign": b_sign, "exponent": b_exp, "mantissa": b_mant})

            result_struct = ctx.get(dut.result)
            result_packed = bfloat16.BFloat16.pack(
                result_struct["sign"], result_struct["exponent"], result_struct["mantissa"]
            )
            result_f = bfloat16.BFloat16.to_float(result_packed)

            error = abs(result_f - expected_f)
            rel_error = error / abs(expected_f) if expected_f != 0 else error

            assert rel_error < 0.02, f"Add({a_f} + {b_f}): got {result_f}, expected {expected_f}, error={rel_error:.6f}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"BF16Adder_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_adder_subtraction(request):
    """Test subtraction (different signs)"""
    dut = bf16_adder.BF16Adder()

    test_cases = [
        # (a, b, expected_result)
        # Simple subtractions
        (5.0, -2.0, 3.0),  # 5 + (-2) = 5 - 2 = 3
        (10.0, -5.0, 5.0),  # 10 + (-5) = 10 - 5 = 5
        (2.0, -2.0, 0.0),  # 2 + (-2) = 0
        # Negative results
        (2.0, -5.0, -3.0),  # 2 + (-5) = 2 - 5 = -3
        (1.0, -10.0, -9.0),  # 1 + (-10) = 1 - 10 = -9
        # Both negative (actually addition of magnitudes with negative sign)
        (-2.0, -3.0, -5.0),  # (-2) + (-3) = -5
        (-1.0, -1.0, -2.0),  # (-1) + (-1) = -2
    ]

    async def bench(ctx):
        for a_f, b_f, expected_f in test_cases:
            a_packed = bfloat16.BFloat16.from_float(a_f)
            b_packed = bfloat16.BFloat16.from_float(b_f)

            a_sign, a_exp, a_mant = bfloat16.BFloat16.unpack(a_packed)
            b_sign, b_exp, b_mant = bfloat16.BFloat16.unpack(b_packed)

            ctx.set(dut.a, {"sign": a_sign, "exponent": a_exp, "mantissa": a_mant})
            ctx.set(dut.b, {"sign": b_sign, "exponent": b_exp, "mantissa": b_mant})

            result_struct = ctx.get(dut.result)
            result_packed = bfloat16.BFloat16.pack(
                result_struct["sign"], result_struct["exponent"], result_struct["mantissa"]
            )
            result_f = bfloat16.BFloat16.to_float(result_packed)

            error = abs(result_f - expected_f)
            rel_error = error / abs(expected_f) if expected_f != 0 else error

            assert rel_error < 0.01, f"Add({a_f} + {b_f}): got {result_f}, expected {expected_f}, error={rel_error:.6f}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"BF16Adder_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_adder_near_cancellation(request):
    """Test near-cancellation cases (a ≈ -b)"""
    dut = bf16_adder.BF16Adder()

    test_cases = [
        # (a, b, expected_result)
        # Exact cancellation
        (5.0, -5.0, 0.0),
        (1.0, -1.0, 0.0),
        (100.0, -100.0, 0.0),
        # Near cancellation
        (5.0, -4.0, 1.0),
        (10.0, -9.0, 1.0),
        (100.0, -99.0, 1.0),
    ]

    async def bench(ctx):
        for a_f, b_f, expected_f in test_cases:
            a_packed = bfloat16.BFloat16.from_float(a_f)
            b_packed = bfloat16.BFloat16.from_float(b_f)

            a_sign, a_exp, a_mant = bfloat16.BFloat16.unpack(a_packed)
            b_sign, b_exp, b_mant = bfloat16.BFloat16.unpack(b_packed)

            ctx.set(dut.a, {"sign": a_sign, "exponent": a_exp, "mantissa": a_mant})
            ctx.set(dut.b, {"sign": b_sign, "exponent": b_exp, "mantissa": b_mant})

            result_struct = ctx.get(dut.result)
            result_packed = bfloat16.BFloat16.pack(
                result_struct["sign"], result_struct["exponent"], result_struct["mantissa"]
            )
            result_f = bfloat16.BFloat16.to_float(result_packed)

            error = abs(result_f - expected_f)

            assert error < 0.01, f"Add({a_f} + {b_f}): got {result_f}, expected {expected_f}, error={error:.6f}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"BF16Adder_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_adder_with_zero(request):
    """Test addition with zero operands"""
    dut = bf16_adder.BF16Adder()

    test_cases = [
        # (a, b, expected_result)
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 5.0),
        (0.0, 5.0, 5.0),
        (-5.0, 0.0, -5.0),
        (0.0, -5.0, -5.0),
    ]

    async def bench(ctx):
        for a_f, b_f, expected_f in test_cases:
            a_packed = bfloat16.BFloat16.from_float(a_f)
            b_packed = bfloat16.BFloat16.from_float(b_f)

            a_sign, a_exp, a_mant = bfloat16.BFloat16.unpack(a_packed)
            b_sign, b_exp, b_mant = bfloat16.BFloat16.unpack(b_packed)

            ctx.set(dut.a, {"sign": a_sign, "exponent": a_exp, "mantissa": a_mant})
            ctx.set(dut.b, {"sign": b_sign, "exponent": b_exp, "mantissa": b_mant})

            result_struct = ctx.get(dut.result)
            result_packed = bfloat16.BFloat16.pack(
                result_struct["sign"], result_struct["exponent"], result_struct["mantissa"]
            )
            result_f = bfloat16.BFloat16.to_float(result_packed)

            error = abs(result_f - expected_f)

            assert error < 0.01, f"Add({a_f} + {b_f}): got {result_f}, expected {expected_f}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"BF16Adder_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()
