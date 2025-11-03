import sys

from amaranth.sim import Simulator

import aligner


def test_aligner_no_shift(request):
    dut = aligner.Aligner(width=26)

    test_cases = [
        (0b00000000000000000000000000, 0, 0b00000000000000000000000000),
        (0b11111111111111111111111111, 0, 0b11111111111111111111111111),
        (0b10101010101010101010101010, 0, 0b10101010101010101010101010),
        (0b01010101010101010101010101, 0, 0b01010101010101010101010101),
        (0b10000000000000000000000001, 0, 0b10000000000000000000000001),
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
        vcd_name = f"Aligner_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_aligner_basic_shifts(request):
    dut = aligner.Aligner(width=26)

    test_cases = [
        # (input, shift_amount, expected_output)
        (0b11111111111111111111111111, 1, 0b01111111111111111111111111),
        (0b11111111111111111111111111, 2, 0b00111111111111111111111111),
        (0b11111111111111111111111111, 4, 0b00001111111111111111111111),
        (0b11111111111111111111111111, 8, 0b00000000111111111111111111),
        (0b11111111111111111111111111, 16, 0b00000000000000001111111111),
        (0b11111111111111111111111111, 25, 0b00000000000000000000000001),
        (0b11111111111111111111111111, 26, 0b00000000000000000000000000),  # All bits shifted out
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
        vcd_name = f"Aligner_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_aligner_specific_patterns(request):
    dut = aligner.Aligner(width=26)

    test_cases = [
        # Single bit patterns
        (0b10000000000000000000000000, 1, 0b01000000000000000000000000),
        (0b10000000000000000000000000, 5, 0b00000100000000000000000000),
        (0b10000000000000000000000000, 25, 0b00000000000000000000000001),
        # Alternating patterns
        (0b10101010101010101010101010, 1, 0b01010101010101010101010101),
        (0b10101010101010101010101010, 2, 0b00101010101010101010101010),
        (0b01010101010101010101010101, 1, 0b00101010101010101010101010),
        # Edge pattern
        (0b11110000111100001111000011, 3, 0b00011110000111100001111000),
        (0b11110000111100001111000011, 7, 0b00000001111000011110000111),
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
        vcd_name = f"Aligner_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_aligner_edge_cases(request):
    dut = aligner.Aligner(width=26)

    test_cases = [
        # Shift amount larger than width (should result in all zeros)
        (0b11111111111111111111111111, 26, 0b00000000000000000000000000),
        (0b11111111111111111111111111, 27, 0b00000000000000000000000000),
        (0b11111111111111111111111111, 31, 0b00000000000000000000000000),
        # Zero input
        (0b00000000000000000000000000, 5, 0b00000000000000000000000000),
        (0b00000000000000000000000000, 15, 0b00000000000000000000000000),
        # Max shift within bounds
        (0b11111111111111111111111111, 25, 0b00000000000000000000000001),
        (0b00000000000000000000000001, 0, 0b00000000000000000000000001),
        (0b00000000000000000000000001, 1, 0b00000000000000000000000000),
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
        vcd_name = f"Aligner_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()
