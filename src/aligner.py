from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class Aligner(wiring.Component):
    def __init__(self, width: int = 26):
        self.width = width
        self.shift_bits = (width - 1).bit_length()

        super().__init__(
            {
                "value_in": In(width),
                "shift_amount": In(self.shift_bits),
                "value_out": Out(width),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()
        m.d.comb += self.value_out.eq(self.value_in >> self.shift_amount)
        return m
