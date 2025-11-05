from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from aligner import Aligner
from bfloat16 import BFloat16
from carry_select_adder import CarrySelectAdder, CarrySelectSubtractor
from lza import LeadingZeroAnticipator
from normalizer import Normalizer
from parallel_prefix import KoggeStoneAdder, KoggeStoneSubtractor
from rounder import Rounder


class BF16Adder(wiring.Component):
    a: In(BFloat16)
    b: In(BFloat16)
    result: Out(BFloat16)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        m.submodules.aligner = aligner = Aligner(width=26)
        m.submodules.lza = lza = LeadingZeroAnticipator(width=26)
        m.submodules.normalizer = normalizer = Normalizer(width=26)
        m.submodules.rounder = rounder = Rounder(width=7)
        m.submodules.cs_adder = cs_adder = CarrySelectAdder(width=26, block_size=6)
        m.submodules.cs_sub = cs_sub = CarrySelectSubtractor(width=26, block_size=6)
        m.submodules.ks_exp_sub = ks_exp_sub = KoggeStoneSubtractor(width=8)
        m.submodules.ks_abs_sub = ks_abs_sub = KoggeStoneSubtractor(width=8)
        m.submodules.ks_exp_add = ks_exp_add = KoggeStoneAdder(width=9)

        a_sign = self.a.sign
        a_exp = self.a.exponent
        a_mant = self.a.mantissa

        b_sign = self.b.sign
        b_exp = self.b.exponent
        b_mant = self.b.mantissa

        a_is_zero = Signal()
        b_is_zero = Signal()

        m.d.comb += a_is_zero.eq(self.a.exponent == 0)
        m.d.comb += b_is_zero.eq(self.b.exponent == 0)

        with m.If(a_is_zero & b_is_zero):
            m.d.comb += self.result.sign.eq(0)
            m.d.comb += self.result.exponent.eq(0)
            m.d.comb += self.result.mantissa.eq(0)
        with m.Elif(a_is_zero):
            m.d.comb += self.result.eq(self.b)
        with m.Elif(b_is_zero):
            m.d.comb += self.result.eq(self.a)
        with m.Else():
            a_mant_full = Signal(8)
            b_mant_full = Signal(8)

            m.d.comb += a_mant_full.eq(Cat(a_mant, Const(1, 1)))
            m.d.comb += b_mant_full.eq(Cat(b_mant, Const(1, 1)))

            a_larger = Signal()
            m.d.comb += a_larger.eq(a_exp >= b_exp)

            larger_exp = Signal(8)
            with m.If(a_larger):
                m.d.comb += larger_exp.eq(a_exp)
            with m.Else():
                m.d.comb += larger_exp.eq(b_exp)

            exp_difference = Signal(Shape(9, signed=True))
            m.d.comb += ks_exp_sub.a.eq(a_exp)
            m.d.comb += ks_exp_sub.b.eq(b_exp)
            m.d.comb += exp_difference.eq(ks_exp_sub.diff.as_signed())

            shift_amt = Signal(5)
            abs_exp_diff = Signal(8)

            with m.If(exp_difference >= 0):
                m.d.comb += abs_exp_diff.eq(exp_difference[0:8])
            with m.Else():
                m.d.comb += ks_abs_sub.a.eq(0)
                m.d.comb += ks_abs_sub.b.eq(exp_difference[0:8])
                m.d.comb += abs_exp_diff.eq(ks_abs_sub.diff)

            with m.If(abs_exp_diff > 25):
                m.d.comb += shift_amt.eq(25)
            with m.Else():
                m.d.comb += shift_amt.eq(abs_exp_diff[0:5])

            a_mant_extended = Signal(26)
            b_mant_extended = Signal(26)

            with m.If(a_larger):
                m.d.comb += a_mant_extended.eq(a_mant_full << 18)
                m.d.comb += aligner.value_in.eq(b_mant_full << 18)
                m.d.comb += aligner.shift_amount.eq(shift_amt)
                m.d.comb += b_mant_extended.eq(aligner.value_out)
            with m.Else():
                m.d.comb += b_mant_extended.eq(b_mant_full << 18)
                m.d.comb += aligner.value_in.eq(a_mant_full << 18)
                m.d.comb += aligner.shift_amount.eq(shift_amt)
                m.d.comb += a_mant_extended.eq(aligner.value_out)

            sum_mant = Signal(27)

            signs_match = Signal()
            m.d.comb += signs_match.eq(a_sign == b_sign)

            a_mant_larger = Signal()
            m.d.comb += a_mant_larger.eq(a_mant_extended >= b_mant_extended)

            larger_mant_ext = Mux(a_mant_larger, a_mant_extended, b_mant_extended)
            smaller_mant_ext = Mux(a_mant_larger, b_mant_extended, a_mant_extended)

            with m.If(signs_match):
                m.d.comb += cs_adder.a.eq(a_mant_extended)
                m.d.comb += cs_adder.b.eq(b_mant_extended)
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

            with m.If(sum_mant == 0):
                m.d.comb += self.result.sign.eq(0)
                m.d.comb += self.result.exponent.eq(0)
                m.d.comb += self.result.mantissa.eq(0)
            with m.Elif(sum_overflow):
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

            with m.If(sum_mant != 0):
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

                larger_exp_extended = Signal(9)
                m.d.comb += larger_exp_extended.eq(larger_exp)

                m.d.comb += ks_exp_add.a.eq(larger_exp_extended)
                m.d.comb += ks_exp_add.b.eq(exp_adjustment)
                m.d.comb += ks_exp_add.carry_in.eq(round_overflow)
                m.d.comb += result_exp.eq(ks_exp_add.sum[0:8])

                result_sign = Signal()
                with m.If(signs_match):
                    m.d.comb += result_sign.eq(a_sign)
                with m.Else():
                    with m.If(a_exp > b_exp):
                        m.d.comb += result_sign.eq(a_sign)
                    with m.Elif(b_exp > a_exp):
                        m.d.comb += result_sign.eq(b_sign)
                    with m.Else():
                        with m.If(a_mant >= b_mant):
                            m.d.comb += result_sign.eq(a_sign)
                        with m.Else():
                            m.d.comb += result_sign.eq(b_sign)

                m.d.comb += self.result.mantissa.eq(rounded_mant)
                m.d.comb += self.result.exponent.eq(result_exp)
                m.d.comb += self.result.sign.eq(result_sign)

        return m
