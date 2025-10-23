from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from bfloat16 import BFloat16


# https://ieeexplore.ieee.org/document/9298120
class BF16Multiplier(wiring.Component):
    a: In(BFloat16)
    b: In(BFloat16)
    result: Out(BFloat16)

    def elaborate(self, platform) -> Module:
        m = Module()

        # ---- Unpack ----
        a_sign = self.a.sign
        a_exp = self.a.exponent
        a_mant = self.a.mantissa

        b_sign = self.b.sign
        b_exp = self.b.exponent
        b_mant = self.b.mantissa

        # ---- Result Sign ----
        result_sign = a_sign ^ b_sign

        # ---- Mantissa Multiply ----
        a_mant_full = Cat(a_mant, Const(1, 1))
        b_mant_full = Cat(b_mant, Const(1, 1))

        mant_product = Signal(16)
        m.d.comb += mant_product.eq(a_mant_full * b_mant_full)

        # ---- Exponent Addition ----
        exp_sum = Signal(9)
        m.d.comb += exp_sum.eq(a_exp + b_exp - 127)

        # ---- Normalization ----
        # NOTE: If bit 15 is set, we have overflow and need to shift right
        normalized_mant = Signal(8)
        normalized_exp = Signal(8)

        with m.If(mant_product[15]):  # Overflow
            m.d.comb += [
                normalized_mant.eq(mant_product[8:16]),
                normalized_exp.eq(exp_sum + 1),
            ]
        with m.Else():
            m.d.comb += [
                normalized_mant.eq(mant_product[7:15]),
                normalized_exp.eq(exp_sum),
            ]

        # ---- Pack ----
        m.d.comb += [
            self.result.mantissa.eq(normalized_mant[0:7]),
            self.result.exponent.eq(normalized_exp),
            self.result.sign.eq(result_sign),
        ]

        return m
