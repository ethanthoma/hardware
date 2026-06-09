from amaranth import *
from amaranth.build import Platform
from amaranth.lib import enum, wiring
from amaranth.lib.wiring import In, Out

from bfloat16 import BFloat16
from fixed_pe import FixedPE

N = 4


class State(enum.Enum, shape=3):
    IDLE = 0
    MAC = 1
    FLUSH = 2  # let the last pipelined product reach acc before draining
    DRAIN = 3  # hold acc one cycle so the pipelined drain settles before DONE reads result
    DONE = 4


class MMA(wiring.Component):
    def __init__(self):
        super().__init__(
            {
                "a_matrix": In(BFloat16).array(N * N),
                "b_matrix": In(BFloat16).array(N * N),
                "start": In(1),
                "done": Out(1),
                "d_matrix": Out(BFloat16).array(N * N),
                "any_dropped": Out(1),
                "any_overflow": Out(1),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        pe = Array(Array(FixedPE() for _ in range(N)) for _ in range(N))
        for i in range(N):
            for j in range(N):
                m.submodules[f"pe_{i}_{j}"] = pe[i][j]

        state = Signal(State)
        k = Signal(range(N))

        with m.Switch(k):
            for k_val in range(N):
                with m.Case(k_val):
                    for i in range(N):
                        for j in range(N):
                            m.d.comb += pe[i][j].a.eq(self.a_matrix[i * N + k_val])
                            m.d.comb += pe[i][j].b.eq(self.b_matrix[k_val * N + j])

        def set_all(load, enable):
            for i in range(N):
                for j in range(N):
                    m.d.comb += pe[i][j].load.eq(load)
                    m.d.comb += pe[i][j].enable.eq(enable)

        with m.Switch(state):
            with m.Case(State.IDLE):
                m.d.comb += self.done.eq(0)
                set_all(load=0, enable=0)
                with m.If(self.start):
                    m.d.sync += state.eq(State.MAC)
                    m.d.sync += k.eq(0)

            with m.Case(State.MAC):
                m.d.comb += self.done.eq(0)
                seed_first_product = k == 0
                accumulate_subsequent = k != 0
                set_all(load=seed_first_product, enable=accumulate_subsequent)
                with m.If(k == N - 1):
                    m.d.sync += state.eq(State.FLUSH)
                with m.Else():
                    m.d.sync += k.eq(k + 1)

            with m.Case(State.FLUSH):
                m.d.comb += self.done.eq(0)
                set_all(load=0, enable=0)
                m.d.sync += state.eq(State.DRAIN)

            with m.Case(State.DRAIN):
                m.d.comb += self.done.eq(0)
                set_all(load=0, enable=0)
                m.d.sync += state.eq(State.DONE)

            with m.Case(State.DONE):
                m.d.comb += self.done.eq(1)
                set_all(load=0, enable=0)
                with m.If(~self.start):
                    m.d.sync += state.eq(State.IDLE)

        for i in range(N):
            for j in range(N):
                m.d.comb += self.d_matrix[i * N + j].eq(pe[i][j].result)

        m.d.comb += self.any_dropped.eq(Cat(pe[i][j].any_dropped for i in range(N) for j in range(N)).any())
        m.d.comb += self.any_overflow.eq(Cat(pe[i][j].any_overflow for i in range(N) for j in range(N)).any())

        return m
