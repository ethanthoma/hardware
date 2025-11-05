from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class FusedExponentDifference(wiring.Component):
    def __init__(self):
        super().__init__(
            {
                "a_exp": In(8),
                "b_exp": In(8),
                "c_exp": In(8),
                "exp_diff": Out(10, init=0),
                "shift_amount": Out(5, init=0),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        a_ext = Signal(signed(10))
        b_ext = Signal(signed(10))
        c_ext = Signal(signed(10))

        m.d.comb += a_ext.eq(self.a_exp)
        m.d.comb += b_ext.eq(self.b_exp)
        m.d.comb += c_ext.eq(self.c_exp)

        result = Signal(signed(10))
        m.d.comb += result.eq(a_ext + b_ext - 127 - c_ext)

        m.d.comb += self.exp_diff.eq(result)

        abs_diff = Signal(10)
        with m.If(result < 0):
            m.d.comb += abs_diff.eq(-result)
        with m.Else():
            m.d.comb += abs_diff.eq(result)

        with m.If(abs_diff > 25):
            m.d.comb += self.shift_amount.eq(25)
        with m.Else():
            m.d.comb += self.shift_amount.eq(abs_diff[0:5])

        return m
