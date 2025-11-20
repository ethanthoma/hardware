from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from bfloat16 import BFloat16, E8M10


class BF16_to_E8M10(wiring.Component):
    """Convert BF16 to E8M10 by extending mantissa with 3 zero bits"""

    bf16_in: In(BFloat16)
    e8m10_out: Out(E8M10)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        m.d.comb += self.e8m10_out.sign.eq(self.bf16_in.sign)
        m.d.comb += self.e8m10_out.exponent.eq(self.bf16_in.exponent)
        m.d.comb += self.e8m10_out.mantissa.eq(self.bf16_in.mantissa << 3)

        return m


class E8M10_to_BF16(wiring.Component):
    """Convert E8M10 to BF16 by truncating mantissa (drop lower 3 bits)"""

    e8m10_in: In(E8M10)
    bf16_out: Out(BFloat16)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        m.d.comb += self.bf16_out.sign.eq(self.e8m10_in.sign)
        m.d.comb += self.bf16_out.exponent.eq(self.e8m10_in.exponent)
        m.d.comb += self.bf16_out.mantissa.eq(self.e8m10_in.mantissa >> 3)

        return m
