from amaranth import *
from amaranth.build import Platform
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from bfloat16 import BFloat16
from normalizer import Normalizer
from rounder import Rounder

BF16_BIAS = 127
BF16_MANTISSA_BITS = 7


def decomposed_layout(width: int) -> data.StructLayout:
    return data.StructLayout({"sign": 1, "magnitude": width, "leading_one": range(width)})


def decompose(m: Module, value: Signal):
    """Drain stage 1: sign, |value|, and the index of its leading one."""
    out = Signal(decomposed_layout(len(value)))
    negative = value < 0
    m.d.comb += out.sign.eq(negative)
    m.d.comb += out.magnitude.eq(Mux(negative, -value, value))
    for i in range(len(value)):
        with m.If(out.magnitude[i]):
            m.d.comb += out.leading_one.eq(i)
    return out


def round_to_bf16(m: Module, d: data.View, lsb_exp: int):
    """Drain stage 2: normalize |acc| to 1.x, round-to-nearest-even, assemble the bf16."""
    width = len(d.magnitude)
    m.submodules.normalizer = normalizer = Normalizer(width=width)
    m.d.comb += normalizer.value_in.eq(d.magnitude)
    m.d.comb += normalizer.shift_amount.eq(width - 1 - d.leading_one)
    normalized = normalizer.value_out

    mantissa_lo = width - 1 - BF16_MANTISSA_BITS
    m.submodules.rounder = rounder = Rounder(width=BF16_MANTISSA_BITS)
    m.d.comb += rounder.mantissa_in.eq(normalized[mantissa_lo : width - 1])
    m.d.comb += rounder.guard.eq(normalized[mantissa_lo - 1])
    m.d.comb += rounder.round_bit.eq(normalized[mantissa_lo - 2])
    m.d.comb += rounder.sticky.eq(normalized[0 : mantissa_lo - 2].any())

    exponent = Signal(signed(12))
    m.d.comb += exponent.eq(lsb_exp + d.leading_one + BF16_BIAS + rounder.overflow)

    result = Signal(BFloat16)
    with m.If(d.magnitude == 0):
        m.d.comb += result.sign.eq(0)
        m.d.comb += result.exponent.eq(0)
        m.d.comb += result.mantissa.eq(0)
    with m.Else():
        m.d.comb += result.sign.eq(d.sign)
        m.d.comb += result.exponent.eq(exponent[0:8])
        m.d.comb += result.mantissa.eq(rounder.mantissa_out)
    return result


class Accumulator(wiring.Component):
    def __init__(self, width: int = 64, lsb_exp: int = 0):
        assert width >= BF16_MANTISSA_BITS + 3  # implicit 1 + mantissa + guard + round
        self.width = width
        self.lsb_exp = lsb_exp

        super().__init__(
            {
                "addend": In(signed(width)),
                "addend_dropped": In(1),  # this addend was an out-of-window product forced to 0
                "load": In(1),
                "enable": In(1),
                "value": Out(signed(width)),
                "result": Out(BFloat16),
                "result_valid": Out(1),
                "any_dropped": Out(1),  # sticky: some addend in this accumulation was dropped
                "any_overflow": Out(1),  # sticky: the accumulator wrapped past signed range
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        acc = Signal(signed(self.width))
        acc_next = Signal(signed(self.width + 1))
        m.d.comb += acc_next.eq(acc + self.addend)
        overflow = Signal()
        m.d.comb += overflow.eq(acc_next[self.width] != acc_next[self.width - 1])

        any_dropped = Signal()
        any_overflow = Signal()
        with m.If(self.load):
            m.d.sync += acc.eq(self.addend)
            m.d.sync += any_dropped.eq(self.addend_dropped)
            m.d.sync += any_overflow.eq(0)
        with m.Elif(self.enable):
            m.d.sync += acc.eq(acc_next[: self.width])
            m.d.sync += any_dropped.eq(any_dropped | self.addend_dropped)
            m.d.sync += any_overflow.eq(any_overflow | overflow)
        m.d.comb += self.value.eq(acc)
        m.d.comb += self.any_dropped.eq(any_dropped)
        m.d.comb += self.any_overflow.eq(any_overflow)

        # drain_latch splits the drain across a register so the normalize+round runs the cycle
        # after acc settles, keeping it off the per-cycle MAC critical path (Fmax).
        drain_latch = Signal(decomposed_layout(self.width))
        m.d.sync += drain_latch.eq(decompose(m, acc))
        m.d.comb += self.result.eq(round_to_bf16(m, drain_latch, self.lsb_exp))
        m.d.sync += self.result_valid.eq(~(self.load | self.enable))

        return m
