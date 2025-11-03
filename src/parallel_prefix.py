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
