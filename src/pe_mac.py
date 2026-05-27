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
    acc_in: In(35)  # 35-bit packed accumulator to restore on load_acc
    load_c: In(1)  # seed acc from the bf16 c_in
    load_acc: In(1)  # restore acc from the 35-bit acc_in (spill/restore, continue-mode)
    enable: In(1)
    acc_out: Out(BFloat16)  # bf16 view of the accumulator
    acc_wide: Out(35)  # full 35-bit accumulator, for spilling to a wide buffer

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        m.submodules.mac = mac = BF16_MAC()

        acc = Signal(35)
        acc_out_reg = Signal(BFloat16)

        m.d.comb += mac.a.eq(self.a_in)
        m.d.comb += mac.b.eq(self.b_in)
        m.d.comb += mac.c_acc.eq(acc)

        with m.If(self.load_acc):
            m.d.sync += acc.eq(self.acc_in)
        with m.Elif(self.load_c):
            c_mant_26 = Signal(26)
            with m.If(self.c_in.exponent == 0):
                m.d.comb += c_mant_26.eq(0)
            with m.Else():
                m.d.comb += c_mant_26.eq(Cat(self.c_in.mantissa, Const(1, 1)) << 18)
            m.d.sync += acc.eq(Cat(c_mant_26, self.c_in.exponent, self.c_in.sign))
            m.d.sync += acc_out_reg.eq(self.c_in)
        with m.Elif(self.enable):
            m.d.sync += acc.eq(mac.intermediate)
            m.d.sync += acc_out_reg.eq(mac.result)

        m.d.comb += self.acc_out.eq(acc_out_reg)
        m.d.comb += self.acc_wide.eq(acc)

        return m
