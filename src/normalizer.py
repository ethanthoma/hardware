from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class Normalizer(wiring.Component):
    """Barrel shifter for mantissa normalization (left-shift)

    - Delay: 2*log2(width) gate delays (2-delta per mux stage)
    - For BF16 FMA: 26-bit width, 10-delta delay
    """

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

        # ---- Left Shift ----
        m.d.comb += self.value_out.eq(self.value_in << self.shift_amount)

        return m
