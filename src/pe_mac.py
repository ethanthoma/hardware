from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from bf16_mac import BF16_MAC
from bfloat16 import BFloat16


class PE_MAC(wiring.Component):
    a_in: In(BFloat16)
    b_in: In(BFloat16)
    c_in: In(BFloat16)
    load_c: In(1)
    enable: In(1)
    acc_out: Out(BFloat16)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        m.submodules.mac = mac = BF16_MAC()

        acc_26bit = Signal(35)
        acc_out_reg = Signal(BFloat16)

        m.d.comb += mac.a.eq(self.a_in)
        m.d.comb += mac.b.eq(self.b_in)
        m.d.comb += mac.c_acc.eq(acc_26bit)

        with m.If(self.load_c):
            c_sign = self.c_in.sign
            c_exp = self.c_in.exponent
            c_mant = self.c_in.mantissa

            c_mant_26 = Signal(26)
            c_is_zero = c_exp == 0

            with m.If(c_is_zero):
                m.d.comb += c_mant_26.eq(0)
            with m.Else():
                c_mant_with_1 = Cat(c_mant, Const(1, 1))
                m.d.comb += c_mant_26.eq(c_mant_with_1 << 18)

            m.d.sync += acc_26bit.eq(Cat(c_mant_26, c_exp, c_sign))
            m.d.sync += acc_out_reg.eq(self.c_in)

        with m.Elif(self.enable):
            m.d.sync += acc_26bit.eq(mac.intermediate)
            m.d.sync += acc_out_reg.eq(mac.result)

        m.d.comb += self.acc_out.eq(acc_out_reg)

        return m
