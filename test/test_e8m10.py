import sys

from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16, E8M10_SW
from e8m10_convert import BF16_to_E8M10, E8M10_to_BF16


def test_bf16_to_e8m10_conversion(request):
    dut = BF16_to_E8M10()

    async def bench(ctx):
        test_values = [0.0, 1.0, -1.0, 2.5, 0.125, 3.14159]

        for val in test_values:
            bf16 = BF16.from_float(val)
            s, e, m = bf16.unpack()

            ctx.set(dut.bf16_in, {"sign": s, "exponent": e, "mantissa": m})

            result = ctx.get(dut.e8m10_out)
            result_e8m10 = E8M10_SW.pack(result["sign"], result["exponent"], result["mantissa"])

            expected_e8m10 = E8M10_SW.from_bf16(bf16)

            assert (
                result_e8m10.bits == expected_e8m10.bits
            ), f"BF16→E8M10 conversion failed for {val}: got {result_e8m10.bits:019b}, expected {expected_e8m10.bits:019b}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"E8M10_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_e8m10_to_bf16_conversion(request):
    dut = E8M10_to_BF16()

    async def bench(ctx):
        test_values = [0.0, 1.0, -1.0, 2.5, 0.125, 3.14159]

        for val in test_values:
            e8m10 = E8M10_SW.from_float(val)
            s, e, m = e8m10.unpack()

            ctx.set(dut.e8m10_in, {"sign": s, "exponent": e, "mantissa": m})

            result = ctx.get(dut.bf16_out)
            result_bf16 = BF16.pack(result["sign"], result["exponent"], result["mantissa"])

            expected_bf16 = e8m10.to_bf16()

            assert (
                result_bf16.bits == expected_bf16.bits
            ), f"E8M10→BF16 conversion failed for {val}: got {result_bf16.bits:016b}, expected {expected_bf16.bits:016b}"

    sim = Simulator(dut)
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"E8M10_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


def test_e8m10_roundtrip():
    """Test that BF16 → E8M10 → BF16 preserves value"""
    test_values = [0.0, 1.0, -1.0, 2.5, 0.125, 3.14159, -0.5, 100.0]

    for val in test_values:
        bf16_orig = BF16.from_float(val)

        e8m10 = E8M10_SW.from_bf16(bf16_orig)

        bf16_back = e8m10.to_bf16()

        assert bf16_orig.bits == bf16_back.bits, (
            f"Roundtrip failed for {val}: " f"original={bf16_orig.bits:016b}, after={bf16_back.bits:016b}"
        )


def test_e8m10_precision():
    """Verify E8M10 has 3 extra mantissa bits compared to BF16"""
    val = 1.5

    bf16 = BF16.from_float(val)
    e8m10 = E8M10_SW.from_float(val)

    _, _, m_bf16 = bf16.unpack()
    _, _, m_e8m10 = e8m10.unpack()

    assert m_e8m10 == (m_bf16 << 3), (
        f"E8M10 mantissa should be BF16 mantissa << 3: " f"got {m_e8m10:010b}, expected {m_bf16 << 3:010b}"
    )
