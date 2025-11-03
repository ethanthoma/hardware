import sys

from amaranth.sim import Simulator

import normalizer


def test_normalizer_no_shift(request):
    dut = normalizer.Normalizer(width=26)

    test_cases = [
        (0b00000000000000000000000000, 0, 0b00000000000000000000000000),
        (0b11111111111111111111111111, 0, 0b11111111111111111111111111),
        (0b10101010101010101010101010, 0, 0b10101010101010101010101010),
        (0b01010101010101010101010101, 0, 0b01010101010101010101010101),
        (0b00000000000000000000000001, 0, 0b00000000000000000000000001),
    ]

    async def bench(ctx):
        for value, shift, expected in test_cases:
            ctx.set(dut.value_in, value)
            ctx.set(dut.shift_amount, shift)

            result = ctx.get(dut.value_out)
            assert result == expected, (
                f"No shift: input=0b{value:026b}, got=0b{result:026b}, expected=0b{expected:026b}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"Normalizer_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_normalizer_basic_shifts(request):
    dut = normalizer.Normalizer(width=26)

    test_cases = [
        # (input, shift_amount, expected_output)
        (0b00000000000000000000000001, 1, 0b00000000000000000000000010),
        (0b00000000000000000000000001, 2, 0b00000000000000000000000100),
        (0b00000000000000000000000001, 4, 0b00000000000000000000010000),
        (0b00000000000000000000000001, 8, 0b00000000000000000100000000),
        (0b00000000000000000000000001, 16, 0b00000000010000000000000000),
        (0b00000000000000000000000001, 25, 0b10000000000000000000000000),
        (0b00000000000000000000000001, 26, 0b00000000000000000000000000),  # All bits shifted out
        # Test with all bits set
        (0b11111111111111111111111111, 1, 0b11111111111111111111111110),
        (0b11111111111111111111111111, 2, 0b11111111111111111111111100),
        (0b11111111111111111111111111, 8, 0b11111111111111111100000000),
    ]

    async def bench(ctx):
        for value, shift, expected in test_cases:
            ctx.set(dut.value_in, value)
            ctx.set(dut.shift_amount, shift)

            result = ctx.get(dut.value_out)
            assert result == expected, (
                f"Shift by {shift}: input=0b{value:026b}, got=0b{result:026b}, expected=0b{expected:026b}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"Normalizer_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_normalizer_specific_patterns(request):
    dut = normalizer.Normalizer(width=26)

    test_cases = [
        # Single bit patterns
        (0b00000000000000000000000001, 1, 0b00000000000000000000000010),
        (0b00000000000000000000000001, 5, 0b00000000000000000000100000),
        (0b00000000000000000000000001, 25, 0b10000000000000000000000000),
        # Alternating patterns
        (0b01010101010101010101010101, 1, 0b10101010101010101010101010),
        (0b01010101010101010101010101, 2, 0b01010101010101010101010100),
        (0b10101010101010101010101010, 1, 0b01010101010101010101010100),
        # Edge pattern
        (0b00001111000011110000111100, 3, 0b01111000011110000111100000),
        (0b00001111000011110000111100, 7, 0b10000111100001111000000000),
    ]

    async def bench(ctx):
        for value, shift, expected in test_cases:
            ctx.set(dut.value_in, value)
            ctx.set(dut.shift_amount, shift)

            result = ctx.get(dut.value_out)
            assert result == expected, (
                f"Pattern shift by {shift}: input=0b{value:026b}, got=0b{result:026b}, expected=0b{expected:026b}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"Normalizer_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_normalizer_edge_cases(request):
    dut = normalizer.Normalizer(width=26)

    test_cases = [
        # Shift amount larger than width (should result in all zeros)
        (0b00000000000000000000000001, 26, 0b00000000000000000000000000),
        (0b00000000000000000000000001, 27, 0b00000000000000000000000000),
        (0b00000000000000000000000001, 31, 0b00000000000000000000000000),
        # Zero input
        (0b00000000000000000000000000, 5, 0b00000000000000000000000000),
        (0b00000000000000000000000000, 15, 0b00000000000000000000000000),
        # Max shift within bounds
        (0b00000000000000000000000001, 25, 0b10000000000000000000000000),
        (0b10000000000000000000000000, 0, 0b10000000000000000000000000),
        (0b10000000000000000000000000, 1, 0b00000000000000000000000000),
        # High bit patterns
        (0b00111111111111111111111111, 1, 0b01111111111111111111111110),
        (0b00111111111111111111111111, 2, 0b11111111111111111111111100),
    ]

    async def bench(ctx):
        for value, shift, expected in test_cases:
            ctx.set(dut.value_in, value)
            ctx.set(dut.shift_amount, shift)

            result = ctx.get(dut.value_out)
            assert result == expected, (
                f"Edge case shift by {shift}: input=0b{value:026b}, got=0b{result:026b}, expected=0b{expected:026b}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"Normalizer_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()
