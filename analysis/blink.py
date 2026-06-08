from amaranth import *
from amaranth_boards.ecp5_5g_evn import ECP55GEVNPlatform


class Blink(Elaboratable):
    """User LED 0 toggles every 2**23 cycles of the 12 MHz clock (~0.7 s) -- the build/flash smoke test."""

    def elaborate(self, platform):
        m = Module()
        led = platform.request("led", 0)
        counter = Signal(24)
        m.d.sync += counter.eq(counter + 1)
        m.d.comb += led.o.eq(counter[-1])
        return m


if __name__ == "__main__":
    ECP55GEVNPlatform().build(Blink(), name="blink", do_program=False)
    print("built build/blink.bit")
