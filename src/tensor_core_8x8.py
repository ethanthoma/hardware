from amaranth import *
from amaranth.build import Platform
from amaranth.lib import enum, wiring
from amaranth.lib.wiring import In, Out

from bfloat16 import BFloat16
from pe_mac import PE_MAC


class State(enum.Enum, shape=2):
    IDLE = 0
    LOAD_C = 1
    MAC = 2
    DONE = 3


class TensorCore8x8(wiring.Component):
    def __init__(self):
        super().__init__(
            {
                "a_matrix": In(BFloat16).array(64),
                "b_matrix": In(BFloat16).array(64),
                "c_matrix": In(BFloat16).array(64),
                "start": In(1),
                "done": Out(1),
                "d_matrix": Out(BFloat16).array(64),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        pe = Array(Array(PE_MAC() for _ in range(8)) for _ in range(8))
        for i in range(8):
            for j in range(8):
                m.submodules[f"pe_{i}_{j}"] = pe[i][j]

        state = Signal(State)
        k = Signal(range(8))

        for i in range(8):
            for j in range(8):
                idx = i * 8 + j
                m.d.comb += pe[i][j].c_in.eq(self.c_matrix[idx])

        with m.Switch(k):
            for k_val in range(8):
                with m.Case(k_val):
                    for i in range(8):
                        for j in range(8):
                            a_idx = i * 8 + k_val
                            b_idx = k_val * 8 + j
                            m.d.comb += pe[i][j].a_in.eq(self.a_matrix[a_idx])
                            m.d.comb += pe[i][j].b_in.eq(self.b_matrix[b_idx])

        with m.Switch(state):
            with m.Case(State.IDLE):
                m.d.comb += self.done.eq(0)
                for i in range(8):
                    for j in range(8):
                        m.d.comb += pe[i][j].load_c.eq(0)
                        m.d.comb += pe[i][j].enable.eq(0)

                with m.If(self.start):
                    m.d.sync += state.eq(State.LOAD_C)
                    m.d.sync += k.eq(0)

            with m.Case(State.LOAD_C):
                m.d.comb += self.done.eq(0)
                for i in range(8):
                    for j in range(8):
                        m.d.comb += pe[i][j].load_c.eq(1)
                        m.d.comb += pe[i][j].enable.eq(0)

                m.d.sync += state.eq(State.MAC)

            with m.Case(State.MAC):
                m.d.comb += self.done.eq(0)
                for i in range(8):
                    for j in range(8):
                        m.d.comb += pe[i][j].load_c.eq(0)
                        m.d.comb += pe[i][j].enable.eq(1)

                with m.If(k == 7):
                    m.d.sync += state.eq(State.DONE)
                with m.Else():
                    m.d.sync += k.eq(k + 1)

            with m.Case(State.DONE):
                m.d.comb += self.done.eq(1)
                for i in range(8):
                    for j in range(8):
                        m.d.comb += pe[i][j].load_c.eq(0)
                        m.d.comb += pe[i][j].enable.eq(0)

                with m.If(~self.start):
                    m.d.sync += state.eq(State.IDLE)

        for i in range(8):
            for j in range(8):
                idx = i * 8 + j
                m.d.comb += self.d_matrix[idx].eq(pe[i][j].acc_out)

        return m
