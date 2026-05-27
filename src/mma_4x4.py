from amaranth import *
from amaranth.build import Platform
from amaranth.lib import enum, wiring
from amaranth.lib.wiring import In, Out

from bfloat16 import BFloat16
from pe_mac import PE_MAC

N = 4  # 4x4x4 MMA: A, B, C, D are N×N and the contraction dimension K is N


class State(enum.Enum, shape=2):
    IDLE = 0
    LOAD_C = 1
    MAC = 2
    DONE = 3


class MMA4x4(wiring.Component):
    def __init__(self):
        super().__init__(
            {
                "a_matrix": In(BFloat16).array(N * N),
                "b_matrix": In(BFloat16).array(N * N),
                "c_matrix": In(BFloat16).array(N * N),
                "start": In(1),
                "done": Out(1),
                "d_matrix": Out(BFloat16).array(N * N),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        pe = Array(Array(PE_MAC() for _ in range(N)) for _ in range(N))
        for i in range(N):
            for j in range(N):
                m.submodules[f"pe_{i}_{j}"] = pe[i][j]

        state = Signal(State)
        k = Signal(range(N))

        for i in range(N):
            for j in range(N):
                m.d.comb += pe[i][j].c_in.eq(self.c_matrix[i * N + j])

        with m.Switch(k):
            for k_val in range(N):
                with m.Case(k_val):
                    for i in range(N):
                        for j in range(N):
                            m.d.comb += pe[i][j].a_in.eq(self.a_matrix[i * N + k_val])
                            m.d.comb += pe[i][j].b_in.eq(self.b_matrix[k_val * N + j])

        def set_all(load_c, enable):
            for i in range(N):
                for j in range(N):
                    m.d.comb += pe[i][j].load_c.eq(load_c)
                    m.d.comb += pe[i][j].enable.eq(enable)

        with m.Switch(state):
            with m.Case(State.IDLE):
                m.d.comb += self.done.eq(0)
                set_all(load_c=0, enable=0)
                with m.If(self.start):
                    m.d.sync += state.eq(State.LOAD_C)
                    m.d.sync += k.eq(0)

            with m.Case(State.LOAD_C):
                m.d.comb += self.done.eq(0)
                set_all(load_c=1, enable=0)
                m.d.sync += state.eq(State.MAC)

            with m.Case(State.MAC):
                m.d.comb += self.done.eq(0)
                set_all(load_c=0, enable=1)
                with m.If(k == N - 1):
                    m.d.sync += state.eq(State.DONE)
                with m.Else():
                    m.d.sync += k.eq(k + 1)

            with m.Case(State.DONE):
                m.d.comb += self.done.eq(1)
                set_all(load_c=0, enable=0)
                with m.If(~self.start):
                    m.d.sync += state.eq(State.IDLE)

        for i in range(N):
            for j in range(N):
                m.d.comb += self.d_matrix[i * N + j].eq(pe[i][j].acc_out)

        return m
