import random

from amaranth.sim import Simulator

import parallel_prefix


def ripple_carry_reference(g, p, carry_in, width):
    """Reference ripple-carry implementation"""
    carries = [0] * (width + 1)
    carries[0] = carry_in
    for i in range(width):
        carries[i + 1] = g[i] | (p[i] & carries[i])
    return carries


def test_kogge_stone_basic():
    """Test basic Kogge-Stone operation"""
    dut = parallel_prefix.KoggeStone(width=8)

    test_cases = [
        ([0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], 0),
        ([1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1], 0),
        ([0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 1, 1, 1, 1], 1),
        ([1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0], 0),
        ([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 1),
    ]

    async def bench(ctx):
        for g_list, p_list, cin in test_cases:
            g_val = sum(bit << i for i, bit in enumerate(g_list))
            p_val = sum(bit << i for i, bit in enumerate(p_list))

            ctx.set(dut.generate, g_val)
            ctx.set(dut.propagate, p_val)
            ctx.set(dut.carry_in, cin)

            expected = ripple_carry_reference(g_list, p_list, cin, 8)

            for i in range(9):
                result = ctx.get(dut.carries[i])
                assert result == expected[i], f"Carry[{i}] mismatch: got {result}, expected {expected[i]}"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()


def test_kogge_stone_26bit():
    """Test 26-bit Kogge-Stone (actual LZA width)"""
    dut = parallel_prefix.KoggeStone(width=26)

    async def bench(ctx):
        random.seed(42)
        for _ in range(50):
            g_list = [random.randint(0, 1) for _ in range(26)]
            p_list = [random.randint(0, 1) for _ in range(26)]
            cin = random.randint(0, 1)

            g_val = sum(bit << i for i, bit in enumerate(g_list))
            p_val = sum(bit << i for i, bit in enumerate(p_list))

            ctx.set(dut.generate, g_val)
            ctx.set(dut.propagate, p_val)
            ctx.set(dut.carry_in, cin)

            expected = ripple_carry_reference(g_list, p_list, cin, 26)

            for i in range(27):
                result = ctx.get(dut.carries[i])
                assert result == expected[i], f"Carry[{i}] mismatch: got {result}, expected {expected[i]}"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()


def test_kogge_stone_all_propagate():
    """Test case where all bits propagate"""
    dut = parallel_prefix.KoggeStone(width=8)

    async def bench(ctx):
        ctx.set(dut.generate, 0b00000000)
        ctx.set(dut.propagate, 0b11111111)
        ctx.set(dut.carry_in, 1)

        for i in range(9):
            result = ctx.get(dut.carries[i])
            assert result == 1, f"Carry[{i}] should be 1 (all propagate)"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()


def test_kogge_stone_all_generate():
    """Test case where all bits generate"""
    dut = parallel_prefix.KoggeStone(width=8)

    async def bench(ctx):
        ctx.set(dut.generate, 0b11111111)
        ctx.set(dut.propagate, 0b00000000)
        ctx.set(dut.carry_in, 0)

        for i in range(1, 9):
            result = ctx.get(dut.carries[i])
            assert result == 1, f"Carry[{i}] should be 1 (all generate)"

        result = ctx.get(dut.carries[0])
        assert result == 0, "Carry[0] should be carry_in"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()
