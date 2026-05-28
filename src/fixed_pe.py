from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from accumulator import Accumulator
from bfloat16 import BFloat16
from carry_select_adder import CarrySelectAdder
from mantissa_multiplier import MantissaMultiplier

BIAS = 127
WIDTH = 48
LSB_EXP = -32
PRODUCT_FRAC_BITS = 14  # bf16 mantissas are 1.x_7bit; product is x.xx_14bit fixed
GRID_ALIGN = 2 * BIAS + PRODUCT_FRAC_BITS + LSB_EXP
MAX_SHIFT = WIDTH - 17  # WIDTH-16 for product width, -1 for the sign bit


def aligned_addend(m: Module, a, b) -> Signal:
    """Return a*b as a signed fixed-point addend on the LSB_EXP-weighted grid."""
    m.submodules.mult = mult = MantissaMultiplier()
    m.d.comb += mult.a_mant.eq(a.mantissa)
    m.d.comb += mult.b_mant.eq(b.mantissa)

    sign = a.sign ^ b.sign
    zero = (a.exponent == 0) | (b.exponent == 0)

    shift = Signal(signed(12))
    m.d.comb += shift.eq(a.exponent + b.exponent - GRID_ALIGN)

    prod = Signal(WIDTH)
    m.d.comb += prod.eq(Mux(zero, 0, mult.product))

    in_window = Signal()
    m.d.comb += in_window.eq((shift >= 0) & (shift <= MAX_SHIFT))
    shamt = Signal(6)
    m.d.comb += shamt.eq(Mux(in_window, shift[:6], 0))

    magnitude = Signal(WIDTH)
    m.d.comb += magnitude.eq(Mux(in_window, (prod << shamt)[:WIDTH], 0))

    addend = Signal(signed(WIDTH))
    m.d.comb += addend.eq(Mux(sign, -magnitude, magnitude))
    return addend


class FixedMAC(wiring.Component):
    """Combinational MAC: acc_out = acc_in + align(a*b). Depth proxy vs BF16_MAC."""

    a: In(BFloat16)
    b: In(BFloat16)
    acc_in: In(signed(WIDTH))
    acc_out: Out(signed(WIDTH))

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()
        addend = aligned_addend(m, self.a, self.b)
        m.submodules.add = add = CarrySelectAdder(WIDTH, 6)
        m.d.comb += add.a.eq(self.acc_in.as_unsigned())
        m.d.comb += add.b.eq(addend.as_unsigned())
        m.d.comb += add.carry_in.eq(0)
        m.d.comb += self.acc_out.eq(add.sum.as_signed())
        return m


class FixedPE(wiring.Component):
    """Registered PE: multiply, align, accumulate; drains to bf16."""

    a: In(BFloat16)
    b: In(BFloat16)
    load: In(1)
    enable: In(1)
    result: Out(BFloat16)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()
        addend = aligned_addend(m, self.a, self.b)
        m.submodules.acc = acc = Accumulator(width=WIDTH, lsb_exp=LSB_EXP)
        m.d.comb += acc.addend.eq(addend)
        m.d.comb += acc.load.eq(self.load)
        m.d.comb += acc.enable.eq(self.enable)
        m.d.comb += self.result.eq(acc.result)
        return m
