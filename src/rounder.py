from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class Rounder(wiring.Component):
    """Mantissa rounder with round-to-nearest-even

    - Delay: 2*log2(width) gate delays for increment (ripple carry)
    - For BF16 FMA: supports 7-bit, 8-bit, and 16-bit widths
    """

    def __init__(self, width: int = 8):
        self.width = width

        super().__init__(
            {
                "mantissa_in": In(width),
                "guard": In(1),
                "round_bit": In(1),
                "sticky": In(1),
                "mantissa_out": Out(width),
                "overflow": Out(1),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        # ---- Round Decision Logic ----
        lsb = self.mantissa_in[0]

        # Determine if we should round up
        # G=1 and (R=1 or S=1): always round up
        # G=1 and R=0 and S=0 and LSB=1: tie, round to even (round up)
        round_up = Signal()
        with m.If(self.guard):
            with m.If(self.round_bit | self.sticky):
                m.d.comb += round_up.eq(1)  # G=1 and (R or S): always round up
            with m.Else():
                m.d.comb += round_up.eq(lsb)  # G=1, R=0, S=0: tie case, round to even
        with m.Else():
            m.d.comb += round_up.eq(0)  # G=0: always round down

        # ---- Increment Logic ----
        incremented = Signal(self.width + 1)
        m.d.comb += incremented.eq(self.mantissa_in + round_up)

        # ---- Output and Overflow Detection ----
        m.d.comb += self.mantissa_out.eq(incremented[0 : self.width])
        m.d.comb += self.overflow.eq(incremented[self.width])

        return m
