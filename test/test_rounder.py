import sys

from amaranth.sim import Simulator

import rounder


def test_rounder_no_rounding(request):
    """Test cases where no rounding occurs (round down)"""
    dut = rounder.Rounder(width=8)

    test_cases = [
        # (mantissa, guard, round, sticky, expected_result, expected_overflow)
        # GRS = 000: always round down
        (0b10000000, 0, 0, 0, 0b10000000, 0),
        (0b11111111, 0, 0, 0, 0b11111111, 0),
        (0b10101010, 0, 0, 0, 0b10101010, 0),
        # GRS = 001: round down
        (0b10000000, 0, 0, 1, 0b10000000, 0),
        (0b11111110, 0, 0, 1, 0b11111110, 0),
        # GRS = 010: round down
        (0b10000000, 0, 1, 0, 0b10000000, 0),
        (0b11111110, 0, 1, 0, 0b11111110, 0),
        # GRS = 011: round down
        (0b10000000, 0, 1, 1, 0b10000000, 0),
        (0b11111110, 0, 1, 1, 0b11111110, 0),
    ]

    async def bench(ctx):
        for mantissa, guard, round_bit, sticky, expected, expected_ovf in test_cases:
            ctx.set(dut.mantissa_in, mantissa)
            ctx.set(dut.guard, guard)
            ctx.set(dut.round_bit, round_bit)
            ctx.set(dut.sticky, sticky)

            result = ctx.get(dut.mantissa_out)
            overflow = ctx.get(dut.overflow)

            assert result == expected, (
                f"GRS={guard}{round_bit}{sticky}: mantissa=0b{mantissa:08b}, "
                f"got=0b{result:08b}, expected=0b{expected:08b}"
            )
            assert overflow == expected_ovf, (
                f"GRS={guard}{round_bit}{sticky}: overflow={overflow}, expected={expected_ovf}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"Rounder_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_rounder_round_up(request):
    """Test cases where rounding up occurs"""
    dut = rounder.Rounder(width=8)

    test_cases = [
        # (mantissa, guard, round, sticky, expected_result, expected_overflow)
        # GRS = 101: always round up
        (0b10000000, 1, 0, 1, 0b10000001, 0),
        (0b10101010, 1, 0, 1, 0b10101011, 0),
        (0b11111110, 1, 0, 1, 0b11111111, 0),
        # GRS = 110: always round up
        (0b10000000, 1, 1, 0, 0b10000001, 0),
        (0b10101010, 1, 1, 0, 0b10101011, 0),
        (0b11111110, 1, 1, 0, 0b11111111, 0),
        # GRS = 111: always round up
        (0b10000000, 1, 1, 1, 0b10000001, 0),
        (0b10101010, 1, 1, 1, 0b10101011, 0),
        (0b11111110, 1, 1, 1, 0b11111111, 0),
    ]

    async def bench(ctx):
        for mantissa, guard, round_bit, sticky, expected, expected_ovf in test_cases:
            ctx.set(dut.mantissa_in, mantissa)
            ctx.set(dut.guard, guard)
            ctx.set(dut.round_bit, round_bit)
            ctx.set(dut.sticky, sticky)

            result = ctx.get(dut.mantissa_out)
            overflow = ctx.get(dut.overflow)

            assert result == expected, (
                f"GRS={guard}{round_bit}{sticky}: mantissa=0b{mantissa:08b}, "
                f"got=0b{result:08b}, expected=0b{expected:08b}"
            )
            assert overflow == expected_ovf, (
                f"GRS={guard}{round_bit}{sticky}: overflow={overflow}, expected={expected_ovf}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"Rounder_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_rounder_tie_to_even(request):
    """Test round-to-nearest-even (ties go to even LSB)"""
    dut = rounder.Rounder(width=8)

    test_cases = [
        # (mantissa, guard, round, sticky, expected_result, expected_overflow)
        # GRS = 100, LSB=0: round down (tie to even)
        (0b10000000, 1, 0, 0, 0b10000000, 0),
        (0b10101010, 1, 0, 0, 0b10101010, 0),
        (0b11111110, 1, 0, 0, 0b11111110, 0),
        # GRS = 100, LSB=1: round up (tie to even)
        (0b10000001, 1, 0, 0, 0b10000010, 0),
        (0b10101011, 1, 0, 0, 0b10101100, 0),
        (0b11111111, 1, 0, 0, 0b00000000, 1),  # Overflow case
    ]

    async def bench(ctx):
        for mantissa, guard, round_bit, sticky, expected, expected_ovf in test_cases:
            ctx.set(dut.mantissa_in, mantissa)
            ctx.set(dut.guard, guard)
            ctx.set(dut.round_bit, round_bit)
            ctx.set(dut.sticky, sticky)

            result = ctx.get(dut.mantissa_out)
            overflow = ctx.get(dut.overflow)

            lsb = mantissa & 1
            assert result == expected, (
                f"GRS={guard}{round_bit}{sticky}, LSB={lsb}: mantissa=0b{mantissa:08b}, "
                f"got=0b{result:08b}, expected=0b{expected:08b}"
            )
            assert overflow == expected_ovf, (
                f"GRS={guard}{round_bit}{sticky}, LSB={lsb}: overflow={overflow}, expected={expected_ovf}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"Rounder_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_rounder_overflow(request):
    """Test overflow cases when rounding all 1s"""
    dut = rounder.Rounder(width=8)

    test_cases = [
        # (mantissa, guard, round, sticky, expected_result, expected_overflow)
        # All 1s with round up: should overflow
        (0b11111111, 1, 0, 1, 0b00000000, 1),
        (0b11111111, 1, 1, 0, 0b00000000, 1),
        (0b11111111, 1, 1, 1, 0b00000000, 1),
        (0b11111111, 1, 0, 0, 0b00000000, 1),  # Tie with LSB=1
        # All 1s with round down: no overflow
        (0b11111111, 0, 0, 0, 0b11111111, 0),
        (0b11111111, 0, 1, 1, 0b11111111, 0),
        (0b11111110, 1, 0, 0, 0b11111110, 0),  # Tie with LSB=0
    ]

    async def bench(ctx):
        for mantissa, guard, round_bit, sticky, expected, expected_ovf in test_cases:
            ctx.set(dut.mantissa_in, mantissa)
            ctx.set(dut.guard, guard)
            ctx.set(dut.round_bit, round_bit)
            ctx.set(dut.sticky, sticky)

            result = ctx.get(dut.mantissa_out)
            overflow = ctx.get(dut.overflow)

            assert result == expected, (
                f"Overflow test GRS={guard}{round_bit}{sticky}: mantissa=0b{mantissa:08b}, "
                f"got=0b{result:08b}, expected=0b{expected:08b}"
            )
            assert overflow == expected_ovf, (
                f"Overflow test GRS={guard}{round_bit}{sticky}: overflow={overflow}, expected={expected_ovf}"
            )

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"Rounder_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_rounder_various_widths(request):
    """Test rounder with different mantissa widths"""
    test_configs = [
        (7, 0b1111111, 1, 0, 1, 0b0000000, 1),  # 7-bit overflow
        (7, 0b1010101, 1, 0, 0, 0b1010110, 0),  # 7-bit round up (tie, LSB=1)
        (16, 0xFFFF, 1, 0, 1, 0x0000, 1),  # 16-bit overflow
        (16, 0x8000, 1, 0, 0, 0x8000, 0),  # 16-bit tie to even (LSB=0)
        (16, 0x8001, 1, 0, 0, 0x8002, 0),  # 16-bit tie to even (LSB=1)
    ]

    for width, mantissa, guard, round_bit, sticky, expected, expected_ovf in test_configs:
        dut = rounder.Rounder(width=width)

        async def bench(ctx):
            ctx.set(dut.mantissa_in, mantissa)
            ctx.set(dut.guard, guard)
            ctx.set(dut.round_bit, round_bit)
            ctx.set(dut.sticky, sticky)

            result = ctx.get(dut.mantissa_out)
            overflow = ctx.get(dut.overflow)

            assert result == expected, (
                f"Width={width}, GRS={guard}{round_bit}{sticky}: "
                f"mantissa=0x{mantissa:0{width // 4}x}, got=0x{result:0{width // 4}x}, expected=0x{expected:0{width // 4}x}"
            )
            assert overflow == expected_ovf

        sim = Simulator(dut)
        sim.add_testbench(bench)

        if request.config.getoption("--vcd"):
            vcd_name = f"Rounder_{sys._getframe().f_code.co_name}_w{width}.vcd"
            with sim.write_vcd(vcd_name):
                sim.run()
        else:
            sim.run()
