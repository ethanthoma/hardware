import sys

from amaranth.sim import Simulator

import lza


def test_lza_no_leading_zeros(request):
    """Test cases with no leading zeros (normalized results)"""
    dut = lza.LeadingZeroAnticipator(width=8)

    test_cases = [
        # (operand_a, operand_b, carry_in, expected_lz_count)
        # MSB is 1: no leading zeros
        (0b10000000, 0b00000001, 0, 0),
        (0b11111111, 0b00000000, 0, 0),
        (0b10101010, 0b01010101, 0, 0),
        (0b11000000, 0b00110000, 0, 0),
    ]

    async def bench(ctx):
        for a, b, cin, expected in test_cases:
            ctx.set(dut.a, a)
            ctx.set(dut.b, b)
            ctx.set(dut.carry_in, cin)

            result = ctx.get(dut.lz_count)
            actual_sum = a + b + cin
            actual_lz = count_leading_zeros(actual_sum, 8)

            assert result == expected, (
                f"a=0b{a:08b}, b=0b{b:08b}, cin={cin}: "
                f"sum=0b{actual_sum:08b} (actual_lz={actual_lz}), "
                f"predicted={result}, expected={expected}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"LZA_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_lza_single_leading_zero(request):
    """Test cases with one leading zero"""
    dut = lza.LeadingZeroAnticipator(width=8)

    test_cases = [
        # (operand_a, operand_b, carry_in, expected_lz_count)
        (0b01000000, 0b00000001, 0, 1),
        (0b01111111, 0b00000000, 0, 1),
        (0b01010101, 0b00000000, 0, 1),
    ]

    async def bench(ctx):
        for a, b, cin, expected in test_cases:
            ctx.set(dut.a, a)
            ctx.set(dut.b, b)
            ctx.set(dut.carry_in, cin)

            result = ctx.get(dut.lz_count)
            actual_sum = a + b + cin
            actual_lz = count_leading_zeros(actual_sum, 8)

            assert result == expected, (
                f"a=0b{a:08b}, b=0b{b:08b}, cin={cin}: "
                f"sum=0b{actual_sum:08b} (actual_lz={actual_lz}), "
                f"predicted={result}, expected={expected}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"LZA_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_lza_multiple_leading_zeros(request):
    """Test cases with multiple leading zeros"""
    dut = lza.LeadingZeroAnticipator(width=8)

    test_cases = [
        # (operand_a, operand_b, carry_in, expected_lz_count)
        (0b00100000, 0b00000001, 0, 2),
        (0b00010000, 0b00000001, 0, 3),
        (0b00001000, 0b00000001, 0, 4),
        (0b00000100, 0b00000001, 0, 5),
        (0b00000010, 0b00000001, 0, 6),
        (0b00000001, 0b00000000, 0, 7),
    ]

    async def bench(ctx):
        for a, b, cin, expected in test_cases:
            ctx.set(dut.a, a)
            ctx.set(dut.b, b)
            ctx.set(dut.carry_in, cin)

            result = ctx.get(dut.lz_count)
            actual_sum = a + b + cin
            actual_lz = count_leading_zeros(actual_sum, 8)

            assert result == expected, (
                f"a=0b{a:08b}, b=0b{b:08b}, cin={cin}: "
                f"sum=0b{actual_sum:08b} (actual_lz={actual_lz}), "
                f"predicted={result}, expected={expected}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"LZA_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_lza_subtraction_cases(request):
    """Test cases with subtraction (cancellation)"""
    dut = lza.LeadingZeroAnticipator(width=8)

    test_cases = [
        # (operand_a, operand_b, carry_in, expected_lz_count)
        # Subtraction via two's complement: a + ~b + 1
        # 128 + 128 + 1 = 257 = 0b00000001 (7 leading zeros)
        (0b10000000, 0b10000000, 1, 7),  # 128 + 128 + 1 = 257 = 0b00000001 (7 leading zeros)
        (0b11000000, 0b10111111, 1, 0),  # 192 + 191 + 1 = 384 = 0b10000000 (0 leading zeros)
        (0b10100000, 0b10011111, 1, 1),  # 160 + 159 + 1 = 320 = 0b01000000 (1 leading zero)
    ]

    async def bench(ctx):
        for a, b, cin, expected in test_cases:
            ctx.set(dut.a, a)
            ctx.set(dut.b, b)
            ctx.set(dut.carry_in, cin)

            result = ctx.get(dut.lz_count)
            actual_sum = (a + b + cin) & 0xFF  # Keep 8-bit
            actual_lz = count_leading_zeros(actual_sum, 8)

            assert result == expected, (
                f"a=0b{a:08b}, b=0b{b:08b}, cin={cin}: "
                f"sum=0b{actual_sum:08b} (actual_lz={actual_lz}), "
                f"predicted={result}, expected={expected}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"LZA_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_lza_edge_cases(request):
    """Test edge cases including all zeros"""
    dut = lza.LeadingZeroAnticipator(width=8)

    test_cases = [
        # (operand_a, operand_b, carry_in, expected_lz_count)
        (0b00000000, 0b00000000, 0, 8),  # All zeros
        (0b00000000, 0b00000001, 0, 7),  # Minimal non-zero
        (0b11111111, 0b11111111, 0, 0),  # 255 + 255 = 510 = 0b11111110 (0 leading zeros)
        (0b10000000, 0b10000000, 0, 8),  # 128 + 128 = 256 = 0b00000000 (8 leading zeros, overflow)
    ]

    async def bench(ctx):
        for a, b, cin, expected in test_cases:
            ctx.set(dut.a, a)
            ctx.set(dut.b, b)
            ctx.set(dut.carry_in, cin)

            result = ctx.get(dut.lz_count)
            actual_sum = (a + b + cin) & 0xFF  # Keep 8-bit
            actual_lz = count_leading_zeros(actual_sum, 8)

            assert result == expected, (
                f"Edge case: a=0b{a:08b}, b=0b{b:08b}, cin={cin}: "
                f"sum=0b{actual_sum:08b} (actual_lz={actual_lz}), "
                f"predicted={result}, expected={expected}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"LZA_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_lza_various_widths(request):
    """Test LZA with different widths"""
    test_configs = [
        # (width, a, b, cin, expected_lz)
        (16, 0b0100000000000000, 0b0000000000000001, 0, 1),
        (16, 0b0001000000000000, 0b0000000000000001, 0, 3),
        (16, 0b0000000000000001, 0b0000000000000000, 0, 15),
        (26, 0b01000000000000000000000000, 0b00000000000000000000000001, 0, 1),
        (26, 0b00000000000000000000000001, 0b00000000000000000000000000, 0, 25),
    ]

    for width, a, b, cin, expected in test_configs:
        dut = lza.LeadingZeroAnticipator(width=width)

        async def bench(ctx):
            ctx.set(dut.a, a)
            ctx.set(dut.b, b)
            ctx.set(dut.carry_in, cin)

            result = ctx.get(dut.lz_count)
            actual_sum = (a + b + cin) & ((1 << width) - 1)
            actual_lz = count_leading_zeros(actual_sum, width)

            assert result == expected, (
                f"Width={width}: a=0b{a:0{width}b}, b=0b{b:0{width}b}, cin={cin}: "
                f"sum=0b{actual_sum:0{width}b} (actual_lz={actual_lz}), "
                f"predicted={result}, expected={expected}"
            )

        sim = Simulator(dut)
        sim.add_testbench(bench)

        if request.config.getoption("--vcd"):
            vcd_name = f"LZA_{sys._getframe().f_code.co_name}_w{width}.vcd"
            with sim.write_vcd(vcd_name):
                sim.run()
        else:
            sim.run()


def count_leading_zeros(value: int, width: int) -> int:
    """Helper function to count actual leading zeros"""
    if value == 0:
        return width

    count = 0
    mask = 1 << (width - 1)
    while count < width and (value & mask) == 0:
        count += 1
        mask >>= 1

    return count
