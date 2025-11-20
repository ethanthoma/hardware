from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from bf16_fma import BF16_FMA
from bfloat16 import BFloat16, E8M10
from e8m10_convert import BF16_to_E8M10, E8M10_to_BF16


class PE(wiring.Component):
    """Processing Element with E8M10 accumulator for improved precision

    Uses 19-bit E8M10 accumulator (1 sign + 8 exponent + 10 mantissa) instead
    of BF16. The 3 extra mantissa bits (10 vs 7) provide sufficient precision
    for 8 MAC accumulations while saving area vs full FP32 (19 vs 32 bits).
    """

    a_in: In(BFloat16)
    b_in: In(BFloat16)
    c_in: In(BFloat16)

    load_c: In(1)
    enable: In(1)

    acc_out: Out(BFloat16)

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        m.submodules.fma = fma = BF16_FMA()
        m.submodules.bf16_to_e8m10 = bf16_to_e8m10 = BF16_to_E8M10()
        m.submodules.e8m10_to_bf16 = e8m10_to_bf16 = E8M10_to_BF16()

        acc_reg = Signal(E8M10)

        m.d.comb += fma.a.eq(self.a_in)
        m.d.comb += fma.b.eq(self.b_in)

        m.d.comb += e8m10_to_bf16.e8m10_in.eq(acc_reg)
        m.d.comb += fma.c.eq(e8m10_to_bf16.bf16_out)

        with m.If(self.load_c):
            m.d.comb += bf16_to_e8m10.bf16_in.eq(self.c_in)
            m.d.sync += acc_reg.eq(bf16_to_e8m10.e8m10_out)
        with m.Elif(self.enable):
            m.d.comb += bf16_to_e8m10.bf16_in.eq(fma.result)
            m.d.sync += acc_reg.eq(bf16_to_e8m10.e8m10_out)

        m.d.comb += self.acc_out.eq(e8m10_to_bf16.bf16_out)

        return m
