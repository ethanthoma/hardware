import sys

from amaranth.sim import Simulator

import exp_diff


def test_exp_diff_basic(request):
    dut = exp_diff.ExponentDifference()

    # BF16 bias = 127
    test_cases = [
        # (a_exp, b_exp, c_exp, expected_diff)
        # Case: A=2.0 (exp=128), B=1.0 (exp=127), C=2.0 (exp=128)
        # A*B = 2.0 (exp=128+127-127=128), diff = 128-128 = 0
        (128, 127, 128, 0),
        # Case: A=4.0 (exp=129), B=2.0 (exp=128), C=2.0 (exp=128)
        # A*B = 8.0 (exp=129+128-127=130), diff = 130-128 = 2
        (129, 128, 128, 2),
        # Case: A=1.0, B=1.0, C=0.5 (exp=126)
        # A*B = 1.0 (exp=127+127-127=127), diff = 127-126 = 1
        (127, 127, 126, 1),
        # Case: Large difference
        (135, 135, 127, 16),  # exp_sum=143, diff=16
        # Case: Negative difference (C larger)
        # Should handle as unsigned or saturate to 0
        (127, 126, 130, 0),  # exp_sum=126, C=130, diff would be -4, saturate to 0
    ]

    async def bench(ctx):
        for a_exp, b_exp, c_exp, expected in test_cases:
            ctx.set(dut.a_exp, a_exp)
            ctx.set(dut.b_exp, b_exp)
            ctx.set(dut.c_exp, c_exp)

            result = ctx.get(dut.diff)
            print(f"a_exp={a_exp}, b_exp={b_exp}, c_exp={c_exp}: diff={result} (expected {expected})")

            assert result == expected, f"Expected diff={expected}, got {result}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"ExponentDifference_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_exp_diff_edge_cases(request):
    dut = exp_diff.ExponentDifference()

    test_cases = [
        # Zero exponents (subnormal)
        (0, 0, 0, 0),
        (0, 127, 127, 0),
        # Maximum exponents
        (255, 255, 255, 128),  # 255+255-127=383, but might overflow
        (200, 200, 127, 146),  # Large values
        # C much larger than A*B (should saturate to 0)
        (127, 127, 200, 0),
        # Maximum normal range
        (254, 254, 127, 254),  # Max normal exp
    ]

    async def bench(ctx):
        for a_exp, b_exp, c_exp, expected in test_cases:
            ctx.set(dut.a_exp, a_exp)
            ctx.set(dut.b_exp, b_exp)
            ctx.set(dut.c_exp, c_exp)

            result = ctx.get(dut.diff)
            print(f"Edge: a_exp={a_exp}, b_exp={b_exp}, c_exp={c_exp}: diff={result} (expected {expected})")

            if expected <= 255:
                assert result == expected, f"Expected diff={expected}, got {result}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"ExponentDifference_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_exp_diff_alignment_cases(request):
    dut = exp_diff.ExponentDifference()

    # Realistic FMA scenarios
    test_cases = [
        # Small alignment (common case)
        (127, 127, 127, 0),  # 1.0 * 1.0 + 1.0
        (127, 128, 127, 1),  # 1.0 * 2.0 + 1.0, shift C by 1
        (128, 128, 127, 2),  # 2.0 * 2.0 + 1.0, shift C by 2
        # Medium alignment
        (130, 130, 127, 6),  # 8.0 * 8.0 + 1.0
        (127, 130, 125, 5),  # 1.0 * 8.0 + 0.25
        # Large alignment (shift all bits out)
        (127, 140, 127, 13),
        (135, 135, 120, 23),
    ]

    async def bench(ctx):
        for a_exp, b_exp, c_exp, expected in test_cases:
            ctx.set(dut.a_exp, a_exp)
            ctx.set(dut.b_exp, b_exp)
            ctx.set(dut.c_exp, c_exp)

            result = ctx.get(dut.diff)
            print(f"Align: a_exp={a_exp}, b_exp={b_exp}, c_exp={c_exp}: shift={result}")

            assert result == expected, f"Expected shift={expected}, got {result}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"ExponentDifference_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()
