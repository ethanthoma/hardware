from amaranth import *
from amaranth.build import Platform
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from aligner import Aligner
from bf16_multiplier import BF16Multiplier
from bfloat16 import BFloat16
from exp_diff import ExponentDifference
from lza import LeadingZeroAnticipator
from normalizer import Normalizer
from rounder import Rounder


class BF16_FMA(wiring.Component):
    """BFloat16 Fused Multiply-Adder: result = (a * b) + c

    Based on Hutchins & Swartzlander (2020)

    - Computes (a × b) + c in a single fused operation
    - Higher precision than separate multiply and add
    - Uses optimized pipeline with aligned addition
    """

    a: In(BFloat16)
    b: In(BFloat16)
    c: In(BFloat16)
    result: Out(BFloat16)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        # ---- Instantiate Submodules ----
        m.submodules.multiplier = multiplier = BF16Multiplier()
        m.submodules.exp_diff = exp_diff = ExponentDifference()
        m.submodules.aligner = aligner = Aligner(width=26)
        m.submodules.lza = lza = LeadingZeroAnticipator(width=26)
        m.submodules.normalizer = normalizer = Normalizer(width=26)
        m.submodules.rounder = rounder = Rounder(width=7)

        # ---- Multiply a × b ----
        m.d.comb += multiplier.a.eq(self.a)
        m.d.comb += multiplier.b.eq(self.b)

        product = Signal(BFloat16)
        m.d.comb += product.eq(multiplier.result)

        # ---- Unpack Product and C ----
        product_sign = product.sign
        product_exp = product.exponent
        product_mant = product.mantissa

        c_sign = self.c.sign
        c_exp = self.c.exponent
        c_mant = self.c.mantissa

        # ---- Check for Zero Operands ----
        product_is_zero = Signal()
        c_is_zero = Signal()

        m.d.comb += product_is_zero.eq((self.a.exponent == 0) | (self.b.exponent == 0))
        m.d.comb += c_is_zero.eq(self.c.exponent == 0)

        # ---- Compute Exponent Difference and Perform FMA ----
        with m.If(product_is_zero):
            m.d.comb += self.result.eq(self.c)
        with m.Elif(c_is_zero):
            m.d.comb += self.result.eq(product)
        with m.Else():
            product_mant_full = Signal(8)
            c_mant_full = Signal(8)

            m.d.comb += product_mant_full.eq(Cat(product_mant, Const(1, 1)))
            m.d.comb += c_mant_full.eq(Cat(c_mant, Const(1, 1)))

            # ---- Determine Larger Exponent ----
            product_larger = Signal()
            m.d.comb += product_larger.eq(product_exp >= c_exp)

            larger_exp = Signal(8)
            with m.If(product_larger):
                m.d.comb += larger_exp.eq(product_exp)
            with m.Else():
                m.d.comb += larger_exp.eq(c_exp)

            exp_difference = Signal(Shape(9, signed=True))
            m.d.comb += exp_difference.eq(product_exp - c_exp)

            # ---- Align Mantissas ----
            # Format: bits [25:18] = 8-bit mantissa with implicit 1
            #         bits [17:0] = fractional extension for precision

            shift_amt = Signal(5)
            abs_exp_diff = Signal(8)

            with m.If(exp_difference >= 0):
                m.d.comb += abs_exp_diff.eq(exp_difference[0:8])
            with m.Else():
                m.d.comb += abs_exp_diff.eq(-exp_difference[0:8])

            with m.If(abs_exp_diff > 25):
                m.d.comb += shift_amt.eq(25)  # Shift all bits out
            with m.Else():
                m.d.comb += shift_amt.eq(abs_exp_diff[0:5])

            product_mant_extended = Signal(26)
            c_mant_extended = Signal(26)

            with m.If(product_larger):  # Product has larger exponent, align c
                m.d.comb += product_mant_extended.eq(product_mant_full << 18)
                m.d.comb += aligner.value_in.eq(c_mant_full << 18)
                m.d.comb += aligner.shift_amount.eq(shift_amt)
                m.d.comb += c_mant_extended.eq(aligner.value_out)
            with m.Else():  # C has larger or equal exponent, align product
                m.d.comb += c_mant_extended.eq(c_mant_full << 18)
                m.d.comb += aligner.value_in.eq(product_mant_full << 18)
                m.d.comb += aligner.shift_amount.eq(shift_amt)
                m.d.comb += product_mant_extended.eq(aligner.value_out)

            # ---- Add/Subtract Mantissas ----
            sum_mant = Signal(27)  # Extra bit for overflow

            signs_match = Signal()
            m.d.comb += signs_match.eq(product_sign == c_sign)

            with m.If(signs_match):  # Same sign: add
                m.d.comb += sum_mant.eq(product_mant_extended + c_mant_extended)
            with m.Else():  # Different signs: subtract
                with m.If(product_mant_extended >= c_mant_extended):
                    m.d.comb += sum_mant.eq(product_mant_extended - c_mant_extended)
                with m.Else():
                    m.d.comb += sum_mant.eq(c_mant_extended - product_mant_extended)

            # ---- Detect Overflow and Normalize ----
            sum_overflow = Signal()
            m.d.comb += sum_overflow.eq(sum_mant[26])

            sum_mant_adjusted = Signal(26)
            exp_adjustment = Signal(Shape(9, signed=True))

            with m.If(sum_overflow):  # Overflow: shift right, increment exponent
                m.d.comb += sum_mant_adjusted.eq(sum_mant[1:27])
                m.d.comb += exp_adjustment.eq(1)
            with m.Else():  # No overflow: check for leading zeros
                larger_mant = Mux(product_mant_extended >= c_mant_extended, product_mant_extended, c_mant_extended)
                smaller_mant = Mux(product_mant_extended >= c_mant_extended, c_mant_extended, product_mant_extended)

                m.d.comb += lza.a.eq(larger_mant)
                m.d.comb += lza.b.eq(Mux(signs_match, smaller_mant, ~smaller_mant))
                m.d.comb += lza.carry_in.eq(~signs_match)

                lz_count = Signal(5)
                m.d.comb += lz_count.eq(lza.lz_count[0:5])

                m.d.comb += normalizer.value_in.eq(sum_mant[0:26])
                m.d.comb += normalizer.shift_amount.eq(lz_count)
                m.d.comb += sum_mant_adjusted.eq(normalizer.value_out)

                m.d.comb += exp_adjustment.eq(-lz_count)

            # ---- Round ----
            result_mant_unrounded = Signal(7)
            guard = Signal()
            round_bit = Signal()
            sticky = Signal()

            m.d.comb += result_mant_unrounded.eq(sum_mant_adjusted[18:25])
            m.d.comb += guard.eq(sum_mant_adjusted[17])
            m.d.comb += round_bit.eq(sum_mant_adjusted[16])
            m.d.comb += sticky.eq(sum_mant_adjusted[0:16].any())

            m.d.comb += rounder.mantissa_in.eq(result_mant_unrounded)
            m.d.comb += rounder.guard.eq(guard)
            m.d.comb += rounder.round_bit.eq(round_bit)
            m.d.comb += rounder.sticky.eq(sticky)

            rounded_mant = Signal(7)
            round_overflow = Signal()

            m.d.comb += rounded_mant.eq(rounder.mantissa_out)
            m.d.comb += round_overflow.eq(rounder.overflow)

            # ---- Compute Result Exponent ----
            result_exp = Signal(8)

            with m.If(round_overflow):  # Rounding caused overflow, increment exponent
                m.d.comb += result_exp.eq(larger_exp + exp_adjustment + 1)
            with m.Else():
                m.d.comb += result_exp.eq(larger_exp + exp_adjustment)

            # ---- Compute Result Sign ----
            result_sign = Signal()
            with m.If(signs_match):
                m.d.comb += result_sign.eq(product_sign)
            with m.Else():  # Subtraction: sign depends on which was larger
                with m.If(product_mant_extended >= c_mant_extended):
                    m.d.comb += result_sign.eq(product_sign)
                with m.Else():
                    m.d.comb += result_sign.eq(c_sign)

            # ---- Pack Result ----
            m.d.comb += self.result.mantissa.eq(rounded_mant)
            m.d.comb += self.result.exponent.eq(result_exp)
            m.d.comb += self.result.sign.eq(result_sign)

        return m
