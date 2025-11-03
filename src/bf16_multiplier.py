from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from bfloat16 import BFloat16


class BF16Multiplier(wiring.Component):
    """BFloat16 multiplier: result = a * b

    Based on Hutchins & Swartzlander (2020)
    - https://ieeexplore.ieee.org/document/9298120
    """

    a: In(BFloat16)
    b: In(BFloat16)
    result: Out(BFloat16)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        # ---- Unpack ----
        a_sign = self.a.sign
        a_exp = self.a.exponent
        a_mant = self.a.mantissa

        b_sign = self.b.sign
        b_exp = self.b.exponent
        b_mant = self.b.mantissa

        # ---- Special case detection ----
        a_is_zero = (a_exp == 0) & (a_mant == 0)
        b_is_zero = (b_exp == 0) & (b_mant == 0)
        either_zero = a_is_zero | b_is_zero

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

        with m.If(mant_product[15]):  # Overflow: shift right by 1
            m.d.comb += [
                normalized_mant.eq(mant_product[8:16]),
                normalized_exp.eq(exp_sum + 1),
            ]
        with m.Else():
            m.d.comb += [
                normalized_mant.eq(mant_product[7:15]),
                normalized_exp.eq(exp_sum),
            ]

        # ---- Pack Result ----
        with m.If(either_zero):  # result is zero
            m.d.comb += [
                self.result.mantissa.eq(0),
                self.result.exponent.eq(0),
                self.result.sign.eq(result_sign),
            ]
        with m.Else():
            m.d.comb += [
                self.result.mantissa.eq(normalized_mant[0:7]),
                self.result.exponent.eq(normalized_exp),
                self.result.sign.eq(result_sign),
            ]

        return m
