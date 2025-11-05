from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class CarrySelectAdder(wiring.Component):
    def __init__(self, width: int = 27, block_size: int = 6):
        self.width = width
        self.block_size = block_size

        super().__init__(
            {
                "a": In(width),
                "b": In(width),
                "carry_in": In(1),
                "sum": Out(width),
                "carry_out": Out(1),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        blocks = []
        pos = 0
        while pos < self.width:
            block_width = min(self.block_size, self.width - pos)
            blocks.append((pos, block_width))
            pos += block_width

        carries = [Signal(name=f"carry_{i}") for i in range(len(blocks) + 1)]
        m.d.comb += carries[0].eq(self.carry_in)

        for i, (start, block_width) in enumerate(blocks):
            end = start + block_width
            a_block = self.a[start:end]
            b_block = self.b[start:end]

            if i == 0:
                sum_block = Signal(block_width + 1)
                m.d.comb += sum_block.eq(a_block + b_block + carries[i])
                m.d.comb += self.sum[start:end].eq(sum_block[0:block_width])
                m.d.comb += carries[i + 1].eq(sum_block[block_width])
            else:
                sum_with_0 = Signal(block_width + 1)
                sum_with_1 = Signal(block_width + 1)

                m.d.comb += sum_with_0.eq(a_block + b_block + 0)
                m.d.comb += sum_with_1.eq(a_block + b_block + 1)

                m.d.comb += self.sum[start:end].eq(
                    Mux(carries[i], sum_with_1[0:block_width], sum_with_0[0:block_width])
                )
                m.d.comb += carries[i + 1].eq(Mux(carries[i], sum_with_1[block_width], sum_with_0[block_width]))

        m.d.comb += self.carry_out.eq(carries[len(blocks)])

        return m


class CarrySelectSubtractor(wiring.Component):
    def __init__(self, width: int = 27, block_size: int = 6):
        self.width = width
        self.block_size = block_size

        super().__init__(
            {
                "a": In(width),
                "b": In(width),
                "diff": Out(width),
                "borrow": Out(1),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        m.submodules.adder = adder = CarrySelectAdder(self.width, self.block_size)

        m.d.comb += adder.a.eq(self.a)
        m.d.comb += adder.b.eq(~self.b)
        m.d.comb += adder.carry_in.eq(1)

        m.d.comb += self.diff.eq(adder.sum)
        m.d.comb += self.borrow.eq(~adder.carry_out)

        return m
