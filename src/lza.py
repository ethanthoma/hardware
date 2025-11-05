from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from parallel_prefix import KoggeStone


class LeadingZeroAnticipator(wiring.Component):
    def __init__(self, width: int = 8):
        self.width = width
        self.count_bits = (width).bit_length()

        super().__init__(
            {
                "a": In(width),
                "b": In(width),
                "carry_in": In(1),
                "lz_count": Out(self.count_bits),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        m.submodules.prefix = prefix = KoggeStone(width=self.width)

        generate = Signal(self.width)
        propagate = Signal(self.width)

        m.d.comb += generate.eq(self.a & self.b)
        m.d.comb += propagate.eq(self.a ^ self.b)

        m.d.comb += prefix.generate.eq(generate)
        m.d.comb += prefix.propagate.eq(propagate)
        m.d.comb += prefix.carry_in.eq(self.carry_in)

        predicted_sum = Signal(self.width)

        for i in range(self.width):
            m.d.comb += predicted_sum[i].eq(propagate[i] ^ prefix.carries[i])

        lz_count_result = Signal(self.count_bits)

        lz_count_result = self.width

        for i in range(self.width):
            lz_count_result = Mux(
                predicted_sum[i],
                self.width - 1 - i,
                lz_count_result,
            )

        m.d.comb += self.lz_count.eq(lz_count_result)

        return m
