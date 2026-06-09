from amaranth import *
from amaranth_boards.ecp5_5g_evn import ECP55GEVNPlatform

from bfloat16 import BF16
from mma import MMA, N


def scenario_operands(s: int) -> tuple[list[float], list[float]]:
    """Flat A/B (16 each) per scenario, picked so each trips a specific saturation flag."""
    if s == 1:  # drop: 2**10 * 2**10 products land outside the alignment window -> forced to 0
        return [2.0**10] * (N * N), [2.0**10] * (N * N)
    if s == 2:  # overflow: in-window 2**13 products, K=4 sum = 2**47 -> wraps signed(48)
        return [128.0] * (N * N), [64.0] * (N * N)
    # normal: A[i][k]=i+1, B[k][j]=j+1 -> D[i][j] = 4*(i+1)*(j+1), no flags
    return [idx // N + 1 for idx in range(N * N)], [idx % N + 1 for idx in range(N * N)]


class MMAFlags(Elaboratable):
    """Auto-cycle a matmul through MMA across three scenarios (~1.4 s each) and surface its
    saturation flags -- no input needed, just watch the LED row:

        LED0 = normal result matches expected (PASS)    LED4 = scenario is normal
        LED2 = any_dropped                              LED5 = scenario is drop
        LED3 = any_overflow                             LED6 = scenario is overflow
        LED7 = heartbeat

    So it marches: (LED0+LED4) -> (LED2+LED5) -> (LED3+LED6) -> repeat. The flag LED (0/2/3)
    lighting in lockstep with its scenario LED (4/5/6) is the detection firing on silicon.
    """

    def elaborate(self, platform):
        m = Module()
        m.submodules.mma = mma = MMA()

        # one low cycle per 2**16 clocks re-pulses start so the flags track the live operands
        recompute = Signal(16)
        m.d.sync += recompute.eq(recompute + 1)
        m.d.comb += mma.start.eq(recompute != 0)

        # advance scenario 0 -> 1 -> 2 -> 0 every ~1.4 s
        phase = Signal(24)
        scenario = Signal(2)
        m.d.sync += phase.eq(phase + 1)
        with m.If(phase.all()):
            m.d.sync += scenario.eq(Mux(scenario == 2, 0, scenario + 1))

        for idx in range(N * N):
            a_opts = [BF16.from_float(scenario_operands(s)[0][idx]).to_bits() for s in range(4)]
            b_opts = [BF16.from_float(scenario_operands(s)[1][idx]).to_bits() for s in range(4)]
            m.d.comb += mma.a_matrix[idx].as_value().eq(Array(Const(x, 16) for x in a_opts)[scenario])
            m.d.comb += mma.b_matrix[idx].as_value().eq(Array(Const(x, 16) for x in b_opts)[scenario])

        expected = [BF16.from_float(4 * (idx // N + 1) * (idx % N + 1)).to_bits() for idx in range(N * N)]
        match = Signal()
        m.d.comb += match.eq(Cat(mma.d_matrix[idx].as_value() == expected[idx] for idx in range(N * N)).all())

        heartbeat = Signal(24)
        m.d.sync += heartbeat.eq(heartbeat + 1)

        leds = Signal(8)
        m.d.comb += leds[0].eq(match)
        m.d.comb += leds[2].eq(mma.any_dropped)
        m.d.comb += leds[3].eq(mma.any_overflow)
        m.d.comb += leds[4].eq(scenario == 0)
        m.d.comb += leds[5].eq(scenario == 1)
        m.d.comb += leds[6].eq(scenario == 2)
        m.d.comb += leds[7].eq(heartbeat[-1])
        for i in range(8):
            m.d.comb += platform.request("led", i).o.eq(leds[i])

        return m


if __name__ == "__main__":
    ECP55GEVNPlatform().build(MMAFlags(), name="mma_flags", do_program=False)
    print("built build/mma_flags.bit")
