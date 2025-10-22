from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class UpCounter(wiring.Component):
    en: In(1)
    ovf: Out(1)

    def __init__(self, limit: int):
        self.limit = limit
        self.count = Signal(16)

        super().__init__()

    def elaborate(self, platform) -> Module:
        m = Module()

        m.d.comb += self.ovf.eq(self.count == self.limit)

        with m.If(self.en):
            with m.If(self.ovf):
                m.d.sync += self.count.eq(0)
            with m.Else():
                m.d.sync += self.count.eq(self.count + 1)

        return m
