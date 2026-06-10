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


def aligned_addend(m: Module, a, b):
    """Return (addend, dropped): a*b as a signed fixed-point addend on the LSB_EXP-weighted grid,
    and a flag that the (non-zero) product fell outside the alignment window and was forced to 0."""
    m.submodules.mult = mult = MantissaMultiplier()
    m.d.comb += mult.a_mant.eq(a.mantissa)
    m.d.comb += mult.b_mant.eq(b.mantissa)

    sign = a.sign ^ b.sign
    # TODO: no Inf/NaN handling. The exponent==255 is treated as a finite number, so
    # Inf/NaN operands produce garbage rather than propagating. Subnormals (exponent==0,
    # mantissa!=0) flush to zero here.
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

    dropped = Signal()
    m.d.comb += dropped.eq(~in_window & ~zero)
    return addend, dropped


class FixedMAC(wiring.Component):
    """Combinational MAC: acc_out = acc_in + align(a*b). Per-cycle depth proxy for the grid PE."""

    a: In(BFloat16)
    b: In(BFloat16)
    acc_in: In(signed(WIDTH))
    acc_out: Out(signed(WIDTH))

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()
        addend, _ = aligned_addend(m, self.a, self.b)
        m.submodules.add = add = CarrySelectAdder(WIDTH, 6)
        m.d.comb += add.a.eq(self.acc_in.as_unsigned())
        m.d.comb += add.b.eq(addend.as_unsigned())
        m.d.comb += add.carry_in.eq(0)
        m.d.comb += self.acc_out.eq(add.sum.as_signed())
        return m


class FixedPE(wiring.Component):
    """Registered PE: pipelined multiply+align, accumulate, drain to bf16. The addend and its
    control register one cycle ahead of the add, so the per-cycle loop is just `acc + addend` and the
    operands accumulate one cycle behind their presentation (consumers flush a trailing cycle)."""

    a: In(BFloat16)
    b: In(BFloat16)
    acc_sel: In(2)
    load: In(1)
    enable: In(1)
    result: Out(BFloat16)
    result_valid: Out(1)
    any_dropped: Out(1)
    any_overflow: Out(1)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()
        addend, dropped = aligned_addend(m, self.a, self.b)

        addend_r = Signal(signed(WIDTH))
        dropped_r = Signal()
        acc_sel_r = Signal(2)
        load_r = Signal()
        enable_r = Signal()
        m.d.sync += addend_r.eq(addend)
        m.d.sync += dropped_r.eq(dropped)
        m.d.sync += acc_sel_r.eq(self.acc_sel)
        m.d.sync += load_r.eq(self.load)
        m.d.sync += enable_r.eq(self.enable)

        m.submodules.acc = acc = Accumulator(width=WIDTH, lsb_exp=LSB_EXP)
        m.d.comb += acc.addend.eq(addend_r)
        m.d.comb += acc.addend_dropped.eq(dropped_r)
        m.d.comb += acc.acc_sel.eq(acc_sel_r)
        m.d.comb += acc.load.eq(load_r)
        m.d.comb += acc.enable.eq(enable_r)
        m.d.comb += self.result.eq(acc.result)
        m.d.comb += self.result_valid.eq(acc.result_valid)
        m.d.comb += self.any_dropped.eq(acc.any_dropped)
        m.d.comb += self.any_overflow.eq(acc.any_overflow)
        return m
