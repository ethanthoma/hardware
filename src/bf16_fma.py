from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from bf16_adder import BF16Adder
from bf16_multiplier import BF16Multiplier
from bfloat16 import BFloat16


class BF16_FMA(wiring.Component):
    """BFloat16 Fused Multiply-Adder: result = (a * b) + c"""

    a: In(BFloat16)
    b: In(BFloat16)
    c: In(BFloat16)
    result: Out(BFloat16)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        m.submodules.multiplier = multiplier = BF16Multiplier()
        m.submodules.adder = adder = BF16Adder()

        m.d.comb += multiplier.a.eq(self.a)
        m.d.comb += multiplier.b.eq(self.b)

        m.d.comb += adder.a.eq(multiplier.result)
        m.d.comb += adder.b.eq(self.c)

        m.d.comb += self.result.eq(adder.result)

        return m
