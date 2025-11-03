import random

from amaranth.sim import Simulator

import carry_select_adder


def test_carry_select_adder_basic():
    """Test basic carry-select adder functionality"""
    dut = carry_select_adder.CarrySelectAdder(width=27, block_size=6)

    test_cases = [
        # (a, b, carry_in, expected_sum, expected_carry_out)
        (0, 0, 0, 0, 0),
        (1, 1, 0, 2, 0),
        (100, 200, 0, 300, 0),
        (2**27 - 1, 1, 0, 0, 1),  # Overflow
        (2**26, 2**26, 0, 0, 1),  # Overflow
        (0, 0, 1, 1, 0),  # Carry in only
        (2**27 - 1, 0, 1, 0, 1),  # Max + carry_in
    ]

    async def bench(ctx):
        for a, b, cin, exp_sum, exp_cout in test_cases:
            ctx.set(dut.a, a)
            ctx.set(dut.b, b)
            ctx.set(dut.carry_in, cin)

            result_sum = ctx.get(dut.sum)
            result_cout = ctx.get(dut.carry_out)

            assert result_sum == exp_sum, f"{a} + {b} + {cin}: got sum={result_sum}, expected {exp_sum}"
            assert result_cout == exp_cout, f"{a} + {b} + {cin}: got cout={result_cout}, expected {exp_cout}"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()


def test_carry_select_adder_random():
    """Test carry-select adder with random values"""
    dut = carry_select_adder.CarrySelectAdder(width=27, block_size=6)

    async def bench(ctx):
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, 2**27 - 1)
            b = random.randint(0, 2**27 - 1)
            cin = random.randint(0, 1)

            ctx.set(dut.a, a)
            ctx.set(dut.b, b)
            ctx.set(dut.carry_in, cin)

            result_sum = ctx.get(dut.sum)
            result_cout = ctx.get(dut.carry_out)

            # Compute expected result
            full_sum = a + b + cin
            exp_sum = full_sum & ((1 << 27) - 1)  # Mask to 27 bits
            exp_cout = (full_sum >> 27) & 1

            assert result_sum == exp_sum, f"{a} + {b} + {cin}: got sum={result_sum}, expected {exp_sum}"
            assert result_cout == exp_cout, f"{a} + {b} + {cin}: got cout={result_cout}, expected {exp_cout}"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()


def test_carry_select_subtractor():
    """Test carry-select subtractor"""
    dut = carry_select_adder.CarrySelectSubtractor(width=27, block_size=6)

    test_cases = [
        # (a, b, expected_diff, expected_borrow)
        (10, 5, 5, 0),
        (100, 50, 50, 0),
        (5, 10, 2**27 - 5, 1),  # Negative result (two's complement)
        (0, 1, 2**27 - 1, 1),  # 0 - 1 = -1
        (2**27 - 1, 0, 2**27 - 1, 0),  # Max value
        (1000, 1000, 0, 0),  # Equal values
    ]

    async def bench(ctx):
        for a, b, exp_diff, exp_borrow in test_cases:
            ctx.set(dut.a, a)
            ctx.set(dut.b, b)

            result_diff = ctx.get(dut.diff)
            result_borrow = ctx.get(dut.borrow)

            assert result_diff == exp_diff, f"{a} - {b}: got diff={result_diff}, expected {exp_diff}"
            assert result_borrow == exp_borrow, f"{a} - {b}: got borrow={result_borrow}, expected {exp_borrow}"

    sim = Simulator(dut)
    sim.add_testbench(bench)
    sim.run()


if __name__ == "__main__":
    test_carry_select_adder_basic()
    test_carry_select_adder_random()
    test_carry_select_subtractor()
    print("All carry-select adder tests passed!")
