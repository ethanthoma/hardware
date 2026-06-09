from amaranth import *
from amaranth_boards.ecp5_5g_evn import ECP55GEVNPlatform

from bfloat16 import BF16
from mma import MMA, N


class MMALed(Elaboratable):
    """Run a fixed bf16 matmul through MMA (start tied high -> result holds), compare to the baked
    expected tile, and report on the 8 LEDs. A[i][k]=i+1, B[k][j]=j+1 => D[i][j]=4*(i+1)*(j+1).

    switch 1 low  (status):  LED0=PASS (all 16 cells match), LED1=FAIL, LED7=heartbeat.
    switch 1 high (inspect): LEDs[7:0] = one result byte; switch 4 picks lo/hi byte, switches 5-8
                             pick the cell index 0..15.
    """

    def elaborate(self, platform):
        m = Module()
        m.submodules.mma = mma = MMA()
        m.d.comb += mma.start.eq(1)

        for i in range(N):
            for k in range(N):
                m.d.comb += mma.a_matrix[i * N + k].as_value().eq(BF16.from_float(i + 1).to_bits())
        for k in range(N):
            for j in range(N):
                m.d.comb += mma.b_matrix[k * N + j].as_value().eq(BF16.from_float(j + 1).to_bits())

        expected = [BF16.from_float(4 * (i + 1) * (j + 1)).to_bits() for i in range(N) for j in range(N)]
        match_all = Signal()
        m.d.comb += match_all.eq(Cat(mma.d_matrix[n].as_value() == expected[n] for n in range(N * N)).all())

        heartbeat = Signal(24)
        m.d.sync += heartbeat.eq(heartbeat + 1)

        switch = Array(platform.request("switch", i).i for i in (1, 4, 5, 6, 7, 8))
        byte_sel = Cat(switch[1], switch[2], switch[3], switch[4], switch[5])  # 5 bits: [lo/hi, cell0..3]
        cell = Array(mma.d_matrix[n].as_value() for n in range(N * N))[byte_sel[1:]]
        inspect_byte = Mux(byte_sel[0], cell[8:16], cell[0:8])

        leds = Signal(8)
        with m.If(switch[0]):
            m.d.comb += leds.eq(inspect_byte)
        with m.Else():
            m.d.comb += leds[0].eq(match_all)
            m.d.comb += leds[1].eq(~match_all)
            m.d.comb += leds[7].eq(heartbeat[-1])

        for i in range(8):
            m.d.comb += platform.request("led", i).o.eq(leds[i])

        return m


if __name__ == "__main__":
    expected = [4 * (i + 1) * (j + 1) for i in range(N) for j in range(N)]
    print("expected D (row-major):", expected)
    ECP55GEVNPlatform().build(MMALed(), name="mma_led", do_program=False)
    print("built build/mma_led.bit")
