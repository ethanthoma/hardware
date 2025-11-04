from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class KoggeStone(wiring.Component):
    """Kogge-Stone parallel prefix network for carry computation

    Computes all carries in O(log n) delay vs O(n) ripple-carry
    For 26-bit: log2(26) = 5 levels ~= 10Δ vs 26Δ ripple
    """

    def __init__(self, width: int = 26):
        self.width = width

        super().__init__(
            {
                "generate": In(width),
                "propagate": In(width),
                "carry_in": In(1),
                "carries": Out(width + 1),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        levels = (self.width - 1).bit_length()

        g = [[Signal(name=f"g_{lvl}_{i}") for i in range(self.width)] for lvl in range(levels + 1)]
        p = [[Signal(name=f"p_{lvl}_{i}") for i in range(self.width)] for lvl in range(levels + 1)]

        for i in range(self.width):
            m.d.comb += g[0][i].eq(self.generate[i])
            m.d.comb += p[0][i].eq(self.propagate[i])

        for lvl in range(levels):
            span = 1 << lvl
            for i in range(self.width):
                if i < span:
                    m.d.comb += g[lvl + 1][i].eq(g[lvl][i])
                    m.d.comb += p[lvl + 1][i].eq(p[lvl][i])
                else:
                    m.d.comb += g[lvl + 1][i].eq(g[lvl][i] | (p[lvl][i] & g[lvl][i - span]))
                    m.d.comb += p[lvl + 1][i].eq(p[lvl][i] & p[lvl][i - span])

        m.d.comb += self.carries[0].eq(self.carry_in)

        for i in range(self.width):
            m.d.comb += self.carries[i + 1].eq(g[levels][i] | (p[levels][i] & self.carry_in))

        return m


class KoggeStoneAdder(wiring.Component):
    """Kogge-Stone Adder: O(log n) delay for fast addition"""

    def __init__(self, width: int = 8):
        self.width = width

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

        m.submodules.prefix = prefix = KoggeStone(width=self.width)

        generate = Signal(self.width)
        propagate = Signal(self.width)

        for i in range(self.width):
            m.d.comb += generate[i].eq(self.a[i] & self.b[i])
            m.d.comb += propagate[i].eq(self.a[i] ^ self.b[i])

        m.d.comb += prefix.generate.eq(generate)
        m.d.comb += prefix.propagate.eq(propagate)
        m.d.comb += prefix.carry_in.eq(self.carry_in)

        for i in range(self.width):
            m.d.comb += self.sum[i].eq(propagate[i] ^ prefix.carries[i])

        m.d.comb += self.carry_out.eq(prefix.carries[self.width])

        return m


class KoggeStoneSubtractor(wiring.Component):
    """Kogge-Stone Subtractor: a - b via two's complement"""

    def __init__(self, width: int = 8):
        self.width = width

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

        m.submodules.adder = adder = KoggeStoneAdder(self.width)

        m.d.comb += adder.a.eq(self.a)
        m.d.comb += adder.b.eq(~self.b)
        m.d.comb += adder.carry_in.eq(1)

        m.d.comb += self.diff.eq(adder.sum)
        m.d.comb += self.borrow.eq(~adder.carry_out)

        return m
