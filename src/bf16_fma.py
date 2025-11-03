from amaranth import *
from amaranth.build import Platform
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from aligner import Aligner
from bf16_multiplier import BF16Multiplier
from bfloat16 import BFloat16
from carry_select_adder import CarrySelectAdder, CarrySelectSubtractor
from exp_diff import ExponentDifference
from lza import LeadingZeroAnticipator
from normalizer import Normalizer
from rounder import Rounder


class BF16_FMA(wiring.Component):
    """BFloat16 Fused Multiply-Adder: result = (a * b) + c"""

    a: In(BFloat16)
    b: In(BFloat16)
    c: In(BFloat16)
    result: Out(BFloat16)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        m.submodules.multiplier = multiplier = BF16Multiplier()
        m.submodules.exp_diff = exp_diff = ExponentDifference()
        m.submodules.aligner = aligner = Aligner(width=26)
        m.submodules.lza = lza = LeadingZeroAnticipator(width=26)
        m.submodules.normalizer = normalizer = Normalizer(width=26)
        m.submodules.rounder = rounder = Rounder(width=7)
        m.submodules.cs_adder = cs_adder = CarrySelectAdder(width=26, block_size=6)
        m.submodules.cs_sub = cs_sub = CarrySelectSubtractor(width=26, block_size=6)

        m.d.comb += multiplier.a.eq(self.a)
        m.d.comb += multiplier.b.eq(self.b)

        product = Signal(BFloat16)
        m.d.comb += product.eq(multiplier.result)

        product_sign = product.sign
        product_exp = product.exponent
        product_mant = product.mantissa

        c_sign = self.c.sign
        c_exp = self.c.exponent
        c_mant = self.c.mantissa

        product_is_zero = Signal()
        c_is_zero = Signal()

        m.d.comb += product_is_zero.eq((self.a.exponent == 0) | (self.b.exponent == 0))
        m.d.comb += c_is_zero.eq(self.c.exponent == 0)

        with m.If(product_is_zero):
            m.d.comb += self.result.eq(self.c)
        with m.Elif(c_is_zero):
            m.d.comb += self.result.eq(product)
        with m.Else():
            product_mant_full = Signal(8)
            c_mant_full = Signal(8)

            m.d.comb += product_mant_full.eq(Cat(product_mant, Const(1, 1)))
            m.d.comb += c_mant_full.eq(Cat(c_mant, Const(1, 1)))

            product_larger = Signal()
            m.d.comb += product_larger.eq(product_exp >= c_exp)

            larger_exp = Signal(8)
            with m.If(product_larger):
                m.d.comb += larger_exp.eq(product_exp)
            with m.Else():
                m.d.comb += larger_exp.eq(c_exp)

            exp_difference = Signal(Shape(9, signed=True))
            m.d.comb += exp_difference.eq(product_exp - c_exp)

            shift_amt = Signal(5)
            abs_exp_diff = Signal(8)

            with m.If(exp_difference >= 0):
                m.d.comb += abs_exp_diff.eq(exp_difference[0:8])
            with m.Else():
                m.d.comb += abs_exp_diff.eq(-exp_difference[0:8])

            with m.If(abs_exp_diff > 25):
                m.d.comb += shift_amt.eq(25)
            with m.Else():
                m.d.comb += shift_amt.eq(abs_exp_diff[0:5])

            product_mant_extended = Signal(26)
            c_mant_extended = Signal(26)

            with m.If(product_larger):
                m.d.comb += product_mant_extended.eq(product_mant_full << 18)
                m.d.comb += aligner.value_in.eq(c_mant_full << 18)
                m.d.comb += aligner.shift_amount.eq(shift_amt)
                m.d.comb += c_mant_extended.eq(aligner.value_out)
            with m.Else():
                m.d.comb += c_mant_extended.eq(c_mant_full << 18)
                m.d.comb += aligner.value_in.eq(product_mant_full << 18)
                m.d.comb += aligner.shift_amount.eq(shift_amt)
                m.d.comb += product_mant_extended.eq(aligner.value_out)

            sum_mant = Signal(27)

            signs_match = Signal()
            m.d.comb += signs_match.eq(product_sign == c_sign)

            product_mant_larger = Signal()
            m.d.comb += product_mant_larger.eq(product_mant_extended >= c_mant_extended)

            larger_mant_ext = Mux(product_mant_larger, product_mant_extended, c_mant_extended)
            smaller_mant_ext = Mux(product_mant_larger, c_mant_extended, product_mant_extended)

            with m.If(signs_match):
                m.d.comb += cs_adder.a.eq(product_mant_extended)
                m.d.comb += cs_adder.b.eq(c_mant_extended)
                m.d.comb += cs_adder.carry_in.eq(0)
                m.d.comb += sum_mant.eq(Cat(cs_adder.sum, cs_adder.carry_out))
            with m.Else():
                m.d.comb += cs_sub.a.eq(larger_mant_ext)
                m.d.comb += cs_sub.b.eq(smaller_mant_ext)
                m.d.comb += sum_mant[0:26].eq(cs_sub.diff)
                m.d.comb += sum_mant[26].eq(0)

            sum_overflow = Signal()
            m.d.comb += sum_overflow.eq(sum_mant[26])

            sum_mant_adjusted = Signal(26)
            exp_adjustment = Signal(Shape(9, signed=True))

            with m.If(sum_overflow):
                m.d.comb += sum_mant_adjusted.eq(sum_mant[1:27])
                m.d.comb += exp_adjustment.eq(1)
            with m.Else():
                m.d.comb += lza.a.eq(larger_mant_ext)
                m.d.comb += lza.b.eq(Mux(signs_match, smaller_mant_ext, ~smaller_mant_ext))
                m.d.comb += lza.carry_in.eq(~signs_match)

                lz_count = Signal(5)
                m.d.comb += lz_count.eq(lza.lz_count[0:5])

                m.d.comb += normalizer.value_in.eq(sum_mant[0:26])
                m.d.comb += normalizer.shift_amount.eq(lz_count)
                m.d.comb += sum_mant_adjusted.eq(normalizer.value_out)

                m.d.comb += exp_adjustment.eq(-lz_count)

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

            result_exp = Signal(8)

            with m.If(round_overflow):
                m.d.comb += result_exp.eq(larger_exp + exp_adjustment + 1)
            with m.Else():
                m.d.comb += result_exp.eq(larger_exp + exp_adjustment)

            result_sign = Signal()
            with m.If(signs_match):
                m.d.comb += result_sign.eq(product_sign)
            with m.Else():
                with m.If(product_mant_extended >= c_mant_extended):
                    m.d.comb += result_sign.eq(product_sign)
                with m.Else():
                    m.d.comb += result_sign.eq(c_sign)

            m.d.comb += self.result.mantissa.eq(rounded_mant)
            m.d.comb += self.result.exponent.eq(result_exp)
            m.d.comb += self.result.sign.eq(result_sign)

        return m
