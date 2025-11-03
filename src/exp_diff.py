from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class ExponentDifference(wiring.Component):
    """Exponent difference for FMA alignment

    diff = max(0, (a_exp + b_exp - bias) - c_exp)

    - Delay: 14-delta (2*(1 + 6(FA)))
    - For BF16: 8-bit exponents, bias = 127
    """

    a_exp: In(8)
    b_exp: In(8)
    c_exp: In(8)
    diff: Out(8)

    BIAS = 127

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        # ---- Add Exponents ----
        exp_add = Signal(9)
        m.d.comb += exp_add.eq(self.a_exp + self.b_exp)

        # ---- Compute Difference with Saturation ----
        with m.If(exp_add >= self.BIAS):
            exp_sum = Signal(9)
            m.d.comb += exp_sum.eq(exp_add - self.BIAS)

            with m.If(exp_sum >= self.c_exp):
                exp_diff = Signal(9)
                m.d.comb += exp_diff.eq(exp_sum - self.c_exp)

                with m.If(exp_diff > 255):  # on overflow, saturate to 255
                    m.d.comb += self.diff.eq(255)
                with m.Else():
                    m.d.comb += self.diff.eq(exp_diff[0:8])
            with m.Else():
                m.d.comb += self.diff.eq(0)
        with m.Else():  # subnormal
            m.d.comb += self.diff.eq(0)

        return m
