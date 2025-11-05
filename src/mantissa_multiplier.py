from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class MantissaMultiplier(wiring.Component):
    def __init__(self):
        super().__init__(
            {
                "a_mant": In(7),
                "b_mant": In(7),
                "product": Out(16, init=0),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        a_full = Signal(8)
        b_full = Signal(8)

        m.d.comb += a_full.eq(Cat(self.a_mant, Const(1, 1)))
        m.d.comb += b_full.eq(Cat(self.b_mant, Const(1, 1)))

        m.d.comb += self.product.eq(a_full * b_full)

        return m
