from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from aligner import Aligner
from bfloat16 import BFloat16
from lza import LeadingZeroAnticipator
from normalizer import Normalizer
from rounder import Rounder


class BF16Adder(wiring.Component):
    """BFloat16 Adder: result = a + b"""

    a: In(BFloat16)
    b: In(BFloat16)
    result: Out(BFloat16)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        # ---- Instantiate Submodules ----
        m.submodules.aligner = aligner = Aligner(width=26)
        m.submodules.lza = lza = LeadingZeroAnticipator(width=26)
        m.submodules.normalizer = normalizer = Normalizer(width=26)
        m.submodules.rounder = rounder = Rounder(width=7)

        # ---- Unpack Operands ----
        a_sign = self.a.sign
        a_exp = self.a.exponent
        a_mant = self.a.mantissa

        b_sign = self.b.sign
        b_exp = self.b.exponent
        b_mant = self.b.mantissa

        # ---- Check for Zero Operands ----
        a_is_zero = Signal()
        b_is_zero = Signal()

        m.d.comb += a_is_zero.eq(self.a.exponent == 0)
        m.d.comb += b_is_zero.eq(self.b.exponent == 0)

        # ---- Special Cases ----
        with m.If(a_is_zero & b_is_zero):  # Both zero: result is zero
            m.d.comb += self.result.sign.eq(0)
            m.d.comb += self.result.exponent.eq(0)
            m.d.comb += self.result.mantissa.eq(0)
        with m.Elif(a_is_zero):  # a is zero: result is b
            m.d.comb += self.result.eq(self.b)
        with m.Elif(b_is_zero):  # b is zero: result is a
            m.d.comb += self.result.eq(self.a)
        with m.Else():
            # ---- Normal Addition ----
            a_mant_full = Signal(8)
            b_mant_full = Signal(8)

            m.d.comb += a_mant_full.eq(Cat(a_mant, Const(1, 1)))
            m.d.comb += b_mant_full.eq(Cat(b_mant, Const(1, 1)))

            # ---- Determine Larger Exponent ----
            a_larger = Signal()
            m.d.comb += a_larger.eq(a_exp >= b_exp)

            larger_exp = Signal(8)
            with m.If(a_larger):
                m.d.comb += larger_exp.eq(a_exp)
            with m.Else():
                m.d.comb += larger_exp.eq(b_exp)

            exp_difference = Signal(Shape(9, signed=True))
            m.d.comb += exp_difference.eq(a_exp - b_exp)

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

            a_mant_extended = Signal(26)
            b_mant_extended = Signal(26)

            with m.If(a_larger):  # a has larger exponent, align b
                m.d.comb += a_mant_extended.eq(a_mant_full << 18)
                m.d.comb += aligner.value_in.eq(b_mant_full << 18)
                m.d.comb += aligner.shift_amount.eq(shift_amt)
                m.d.comb += b_mant_extended.eq(aligner.value_out)
            with m.Else():  # b has larger or equal exponent, align a
                m.d.comb += b_mant_extended.eq(b_mant_full << 18)
                m.d.comb += aligner.value_in.eq(a_mant_full << 18)
                m.d.comb += aligner.shift_amount.eq(shift_amt)
                m.d.comb += a_mant_extended.eq(aligner.value_out)

            # ---- Add/Subtract Mantissas ----
            sum_mant = Signal(27)  # Extra bit for overflow

            signs_match = Signal()
            m.d.comb += signs_match.eq(a_sign == b_sign)

            with m.If(signs_match):  # Same sign: add
                m.d.comb += sum_mant.eq(a_mant_extended + b_mant_extended)
            with m.Else():  # Different signs: subtract
                with m.If(a_mant_extended >= b_mant_extended):
                    m.d.comb += sum_mant.eq(a_mant_extended - b_mant_extended)
                with m.Else():
                    m.d.comb += sum_mant.eq(b_mant_extended - a_mant_extended)

            # ---- Detect Overflow and Normalize ----
            sum_overflow = Signal()
            m.d.comb += sum_overflow.eq(sum_mant[26])

            sum_mant_adjusted = Signal(26)
            exp_adjustment = Signal(Shape(9, signed=True))

            with m.If(sum_mant == 0):  # Result is zero (exact cancellation)
                m.d.comb += self.result.sign.eq(0)
                m.d.comb += self.result.exponent.eq(0)
                m.d.comb += self.result.mantissa.eq(0)
            with m.Elif(sum_overflow):  # Overflow: shift right, increment exponent
                m.d.comb += sum_mant_adjusted.eq(sum_mant[1:27])
                m.d.comb += exp_adjustment.eq(1)
            with m.Else():  # No overflow: check for leading zeros
                larger_mant = Mux(a_mant_extended >= b_mant_extended, a_mant_extended, b_mant_extended)
                smaller_mant = Mux(a_mant_extended >= b_mant_extended, b_mant_extended, a_mant_extended)

                m.d.comb += lza.a.eq(larger_mant)
                m.d.comb += lza.b.eq(Mux(signs_match, smaller_mant, ~smaller_mant))
                m.d.comb += lza.carry_in.eq(~signs_match)

                lz_count = Signal(5)
                m.d.comb += lz_count.eq(lza.lz_count[0:5])

                m.d.comb += normalizer.value_in.eq(sum_mant[0:26])
                m.d.comb += normalizer.shift_amount.eq(lz_count)
                m.d.comb += sum_mant_adjusted.eq(normalizer.value_out)

                m.d.comb += exp_adjustment.eq(-lz_count)

            with m.If(sum_mant != 0):
                # ---- Round ----
                # Extract mantissa and GRS bits
                # Format: bit [25] = implicit 1 (should be 1 after normalization)
                #         bits [24:18] = 7-bit mantissa
                #         bit [17] = guard
                #         bit [16] = round
                #         bits [15:0] = sticky
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
                with m.If(signs_match):  # Same signs: use a's sign
                    m.d.comb += result_sign.eq(a_sign)
                with m.Else():  # Different signs
                    with m.If(a_exp > b_exp):
                        m.d.comb += result_sign.eq(a_sign)
                    with m.Elif(b_exp > a_exp):
                        m.d.comb += result_sign.eq(b_sign)
                    with m.Else():  # Equal exponents: compare mantissas
                        with m.If(a_mant >= b_mant):
                            m.d.comb += result_sign.eq(a_sign)
                        with m.Else():
                            m.d.comb += result_sign.eq(b_sign)

                # ---- Pack Result ----
                m.d.comb += self.result.mantissa.eq(rounded_mant)
                m.d.comb += self.result.exponent.eq(result_exp)
                m.d.comb += self.result.sign.eq(result_sign)

        return m
