from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from aligner import Aligner
from bfloat16 import BFloat16
from carry_select_adder import CarrySelectAdder, CarrySelectSubtractor
from fused_exp_diff import FusedExponentDifference
from lza import LeadingZeroAnticipator
from mantissa_multiplier import MantissaMultiplier
from normalizer import Normalizer
from rounder import Rounder


class BF16_MAC(wiring.Component):
    a: In(BFloat16)
    b: In(BFloat16)
    c_acc: In(35)
    result: Out(BFloat16)
    intermediate: Out(35)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        m.submodules.mult = mult = MantissaMultiplier()
        m.d.comb += mult.a_mant.eq(self.a.mantissa)
        m.d.comb += mult.b_mant.eq(self.b.mantissa)

        m.submodules.exp_diff = exp_diff = FusedExponentDifference()
        m.d.comb += exp_diff.a_exp.eq(self.a.exponent)
        m.d.comb += exp_diff.b_exp.eq(self.b.exponent)

        c_sign = self.c_acc[34]
        c_exp = self.c_acc[26:34]
        c_mant_26 = self.c_acc[0:26]

        m.d.comb += exp_diff.c_exp.eq(c_exp)

        m.submodules.aligner = aligner = Aligner(width=26)
        m.submodules.lza = lza = LeadingZeroAnticipator(width=26)
        m.submodules.normalizer = normalizer = Normalizer(width=26)
        m.submodules.rounder = rounder = Rounder(width=7)
        m.submodules.cs_adder = cs_adder = CarrySelectAdder(width=26, block_size=6)
        m.submodules.cs_sub = cs_sub = CarrySelectSubtractor(width=26, block_size=6)

        a_is_zero = self.a.exponent == 0
        b_is_zero = self.b.exponent == 0
        c_is_zero = c_exp == 0

        with m.If(a_is_zero | b_is_zero):
            c_mant_bf16 = Signal(7)
            round_up = c_mant_26[17]
            with m.If(round_up):
                mant_rounded = Signal(8)
                m.d.comb += mant_rounded.eq(c_mant_26[18:25] + 1)
                with m.If(mant_rounded[7]):
                    m.d.comb += c_mant_bf16.eq(0)
                with m.Else():
                    m.d.comb += c_mant_bf16.eq(mant_rounded[0:7])
            with m.Else():
                m.d.comb += c_mant_bf16.eq(c_mant_26[18:25])

            m.d.comb += self.result.sign.eq(c_sign)
            m.d.comb += self.result.exponent.eq(c_exp)
            m.d.comb += self.result.mantissa.eq(c_mant_bf16)
            m.d.comb += self.intermediate.eq(self.c_acc)

        with m.Elif(c_is_zero):
            product_mant = mult.product
            product_overflow = product_mant[15]

            result_mant = Signal(7)
            with m.If(product_overflow):
                m.d.comb += result_mant.eq(product_mant[8:15])
            with m.Else():
                m.d.comb += result_mant.eq(product_mant[7:14])

            base_exp = Signal(8)
            m.d.comb += base_exp.eq(self.a.exponent + self.b.exponent - 127)

            result_exp = Signal(8)
            with m.If(product_overflow):
                m.d.comb += result_exp.eq(base_exp + 1)
            with m.Else():
                m.d.comb += result_exp.eq(base_exp)

            product_sign = self.a.sign ^ self.b.sign
            m.d.comb += self.result.sign.eq(product_sign)
            m.d.comb += self.result.exponent.eq(result_exp)
            m.d.comb += self.result.mantissa.eq(result_mant)

            product_mant_26 = Signal(26)
            with m.If(product_overflow):
                m.d.comb += product_mant_26.eq(product_mant << 10)
            with m.Else():
                m.d.comb += product_mant_26.eq(product_mant << 11)
            m.d.comb += self.intermediate.eq(Cat(product_mant_26, result_exp, product_sign))

        with m.Else():
            product_mant = mult.product
            exp_difference = exp_diff.exp_diff.as_signed()

            product_overflow = product_mant[15]

            adjusted_exp_diff = Signal(signed(10))
            with m.If(product_overflow):
                m.d.comb += adjusted_exp_diff.eq(exp_difference + 1)
            with m.Else():
                m.d.comb += adjusted_exp_diff.eq(exp_difference)

            product_larger = Signal()
            m.d.comb += product_larger.eq(adjusted_exp_diff >= 0)

            adjusted_shift_amt = Signal(5)
            abs_adjusted_diff = Signal(10)
            with m.If(adjusted_exp_diff < 0):
                m.d.comb += abs_adjusted_diff.eq(-adjusted_exp_diff)
            with m.Else():
                m.d.comb += abs_adjusted_diff.eq(adjusted_exp_diff)

            with m.If(abs_adjusted_diff > 25):
                m.d.comb += adjusted_shift_amt.eq(25)
            with m.Else():
                m.d.comb += adjusted_shift_amt.eq(abs_adjusted_diff[0:5])

            product_extended = Signal(26)
            c_extended = Signal(26)

            with m.If(product_larger):
                with m.If(product_overflow):
                    m.d.comb += product_extended.eq(product_mant << 10)
                with m.Else():
                    m.d.comb += product_extended.eq(product_mant << 11)

                m.d.comb += aligner.value_in.eq(c_mant_26)
                m.d.comb += aligner.shift_amount.eq(adjusted_shift_amt)
                m.d.comb += c_extended.eq(aligner.value_out)
            with m.Else():
                m.d.comb += c_extended.eq(c_mant_26)

                product_positioned = Signal(26)
                with m.If(product_overflow):
                    m.d.comb += product_positioned.eq(product_mant << 10)
                with m.Else():
                    m.d.comb += product_positioned.eq(product_mant << 11)

                m.d.comb += aligner.value_in.eq(product_positioned)
                m.d.comb += aligner.shift_amount.eq(adjusted_shift_amt)
                m.d.comb += product_extended.eq(aligner.value_out)

            product_sign = self.a.sign ^ self.b.sign
            signs_match = product_sign == c_sign

            sum_mant = Signal(27)

            product_mant_larger = Signal()
            m.d.comb += product_mant_larger.eq(product_extended >= c_extended)

            larger_mant_ext = Mux(product_mant_larger, product_extended, c_extended)
            smaller_mant_ext = Mux(product_mant_larger, c_extended, product_extended)

            with m.If(signs_match):
                m.d.comb += cs_adder.a.eq(product_extended)
                m.d.comb += cs_adder.b.eq(c_extended)
                m.d.comb += cs_adder.carry_in.eq(0)
                m.d.comb += sum_mant.eq(Cat(cs_adder.sum, cs_adder.carry_out))
            with m.Else():
                m.d.comb += cs_sub.a.eq(larger_mant_ext)
                m.d.comb += cs_sub.b.eq(smaller_mant_ext)
                m.d.comb += sum_mant[0:26].eq(cs_sub.diff)
                m.d.comb += sum_mant[26].eq(0)

            with m.If(sum_mant == 0):
                m.d.comb += self.result.sign.eq(0)
                m.d.comb += self.result.exponent.eq(0)
                m.d.comb += self.result.mantissa.eq(0)
                m.d.comb += self.intermediate.eq(0)
            with m.Else():
                sum_overflow = sum_mant[26]
                sum_mant_adjusted = Signal(26)
                exp_adjustment = Signal(signed(9))

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

                base_larger_exp = Signal(8)
                with m.If(product_larger):
                    m.d.comb += base_larger_exp.eq(self.a.exponent + self.b.exponent - 127)
                with m.Else():
                    m.d.comb += base_larger_exp.eq(c_exp)

                larger_exp = Signal(8)
                with m.If(product_larger & product_overflow):
                    m.d.comb += larger_exp.eq(base_larger_exp + 1)
                with m.Else():
                    m.d.comb += larger_exp.eq(base_larger_exp)

                result_exp = Signal(8)
                exp_total = Signal(signed(10))
                m.d.comb += exp_total.eq(larger_exp + exp_adjustment + round_overflow)
                m.d.comb += result_exp.eq(exp_total[0:8])

                result_sign = Signal()
                with m.If(signs_match):
                    with m.If(product_larger):
                        m.d.comb += result_sign.eq(product_sign)
                    with m.Else():
                        m.d.comb += result_sign.eq(c_sign)
                with m.Else():
                    with m.If(product_mant_larger):
                        m.d.comb += result_sign.eq(product_sign)
                    with m.Else():
                        m.d.comb += result_sign.eq(c_sign)

                m.d.comb += self.result.mantissa.eq(rounded_mant)
                m.d.comb += self.result.exponent.eq(result_exp)
                m.d.comb += self.result.sign.eq(result_sign)

                intermediate_exp = Signal(8)
                intermediate_exp_calc = Signal(signed(10))
                m.d.comb += intermediate_exp_calc.eq(larger_exp + exp_adjustment)
                m.d.comb += intermediate_exp.eq(intermediate_exp_calc[0:8])
                m.d.comb += self.intermediate.eq(Cat(sum_mant_adjusted, intermediate_exp, result_sign))

        return m
