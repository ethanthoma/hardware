from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from bfloat16 import BFloat16
from normalizer import Normalizer
from rounder import Rounder

BF16_BIAS = 127
BF16_MANTISSA_BITS = 7


class Accumulator(wiring.Component):
    def __init__(self, width: int = 64, lsb_exp: int = 0):
        assert width >= BF16_MANTISSA_BITS + 3  # implicit 1 + mantissa + guard + round
        self.width = width
        self.lsb_exp = lsb_exp

        super().__init__(
            {
                "addend": In(signed(width)),
                "load": In(1),
                "enable": In(1),
                "value": Out(signed(width)),
                "result": Out(BFloat16),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        acc = Signal(signed(self.width))
        with m.If(self.load):
            m.d.sync += acc.eq(self.addend)
        with m.Elif(self.enable):
            m.d.sync += acc.eq(acc + self.addend)
        m.d.comb += self.value.eq(acc)

        def drain_to_bf16():
            sign = acc < 0
            magnitude = Signal(self.width)
            m.d.comb += magnitude.eq(Mux(sign, -acc, acc))

            leading_one = Signal(range(self.width))
            for i in range(self.width):
                with m.If(magnitude[i]):
                    m.d.comb += leading_one.eq(i)

            m.submodules.normalizer = normalizer = Normalizer(width=self.width)
            m.d.comb += normalizer.value_in.eq(magnitude)
            m.d.comb += normalizer.shift_amount.eq(self.width - 1 - leading_one)
            normalized = normalizer.value_out

            mantissa_lo = self.width - 1 - BF16_MANTISSA_BITS
            m.submodules.rounder = rounder = Rounder(width=BF16_MANTISSA_BITS)
            m.d.comb += rounder.mantissa_in.eq(normalized[mantissa_lo : self.width - 1])
            m.d.comb += rounder.guard.eq(normalized[mantissa_lo - 1])
            m.d.comb += rounder.round_bit.eq(normalized[mantissa_lo - 2])
            m.d.comb += rounder.sticky.eq(normalized[0 : mantissa_lo - 2].any())

            exponent = Signal(signed(12))
            m.d.comb += exponent.eq(self.lsb_exp + leading_one + BF16_BIAS + rounder.overflow)

            with m.If(magnitude == 0):
                m.d.comb += self.result.sign.eq(0)
                m.d.comb += self.result.exponent.eq(0)
                m.d.comb += self.result.mantissa.eq(0)
            with m.Else():
                m.d.comb += self.result.sign.eq(sign)
                m.d.comb += self.result.exponent.eq(exponent[0:8])
                m.d.comb += self.result.mantissa.eq(rounder.mantissa_out)

        drain_to_bf16()
        return m
