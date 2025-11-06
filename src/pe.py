from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from bf16_fma import BF16_FMA
from bfloat16 import BFloat16


class PE(wiring.Component):
    a_in: In(BFloat16)
    b_in: In(BFloat16)
    c_in: In(BFloat16)

    load_c: In(1)
    enable: In(1)

    acc_out: Out(BFloat16)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        m.submodules.fma = fma = BF16_FMA()

        acc_reg = Signal(BFloat16)

        m.d.comb += fma.a.eq(self.a_in)
        m.d.comb += fma.b.eq(self.b_in)
        m.d.comb += fma.c.eq(acc_reg)

        with m.If(self.load_c):
            m.d.sync += acc_reg.eq(self.c_in)
        with m.Elif(self.enable):
            m.d.sync += acc_reg.eq(fma.result)

        m.d.comb += self.acc_out.eq(acc_reg)

        return m
