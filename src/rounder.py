from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class Rounder(wiring.Component):
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

        lsb = self.mantissa_in[0]

        round_up = Signal()
        with m.If(self.guard):
            with m.If(self.round_bit | self.sticky):
                m.d.comb += round_up.eq(1)
            with m.Else():
                m.d.comb += round_up.eq(lsb)
        with m.Else():
            m.d.comb += round_up.eq(0)

        incremented = Signal(self.width + 1)
        m.d.comb += incremented.eq(self.mantissa_in + round_up)

        m.d.comb += self.mantissa_out.eq(incremented[0 : self.width])
        m.d.comb += self.overflow.eq(incremented[self.width])

        return m
