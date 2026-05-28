from amaranth import *
from amaranth.lib import enum, wiring
from amaranth.lib.wiring import In, Out

from bfloat16 import BFloat16
from fixed_pe import FixedPE

N = 4
TILE_BITS = N * N * 16  # one 4x4 BF16 sub-block per D-SRAM slot
MAX_KBLOCKS = 16


class State(enum.Enum, shape=3):
    IDLE = 0
    FETCH = 1
    LATCH = 2
    MAC = 3
    EVICT = 4
    DONE = 5


class MMAUnit(wiring.Component):
    """K-block streaming MMA. Speculative until cc4 pins the D-SRAM port shape.

    Contract: K = 4 * kblocks (kblocks=0 encodes 16); operands stream from
    consecutive D-SRAM slots starting at slot_a / slot_b; accumulate=1 keeps
    the FixedPE accumulator state from the prior op (no reseed); evict=1 rounds
    and writes back to slot_c.

    Stubbed: acc0..acc3 select (single accumulator only), epilogue (add_en /
    scale_en / act_sel), and BRAM read latency (assumed 1 cycle here).

    TODO: overlap FETCH+LATCH of kb+1 with MAC of kb (double-buffer a_tile /
    b_tile). Current per-kblock cost is 6 cycles (1 FETCH + 1 LATCH + 4 MAC);
    overlapping drops it to 4 (MAC only), the theoretical minimum.

    TODO (cc4 spec gap): D-SRAM port shape. This unit assumes one 256-bit read
    per slot per cycle on each of two operand ports; cc4 ISA.md pins the slot
    size (32 B = 256 b) but not the read-port width / burst pattern. Pin in
    ISA.md "D-SRAM tile slots" section before fabbing.
    """

    def __init__(self):
        super().__init__(
            {
                "start": In(1),
                "accumulate": In(1),
                "evict": In(1),
                "kblocks": In(4),
                "slot_a": In(6),
                "slot_b": In(6),
                "slot_c": In(6),
                "done": Out(1),
                "rd_addr_a": Out(6),
                "rd_addr_b": Out(6),
                "rd_data_a": In(TILE_BITS),
                "rd_data_b": In(TILE_BITS),
                "wr_addr": Out(6),
                "wr_data": Out(TILE_BITS),
                "wr_en": Out(1),
            }
        )

    def elaborate(self, _):
        m = Module()

        pe = Array(Array(FixedPE() for _ in range(N)) for _ in range(N))
        for i in range(N):
            for j in range(N):
                m.submodules[f"pe_{i}_{j}"] = pe[i][j]

        a_tile = [Signal(BFloat16) for _ in range(N * N)]
        b_tile = [Signal(BFloat16) for _ in range(N * N)]

        state = Signal(State)
        # +1 width so kb_end can hold MAX_KBLOCKS (16) when kblocks==0
        kb = Signal(range(MAX_KBLOCKS + 1))
        kb_end = Signal(range(MAX_KBLOCKS + 1))
        k = Signal(range(N))
        first_mac = Signal()

        m.d.comb += self.rd_addr_a.eq(self.slot_a + kb)
        m.d.comb += self.rd_addr_b.eq(self.slot_b + kb)

        with m.Switch(k):
            for k_val in range(N):
                with m.Case(k_val):
                    for i in range(N):
                        for j in range(N):
                            m.d.comb += pe[i][j].a.eq(a_tile[i * N + k_val])
                            m.d.comb += pe[i][j].b.eq(b_tile[k_val * N + j])

        def set_all(load, enable):
            for i in range(N):
                for j in range(N):
                    m.d.comb += pe[i][j].load.eq(load)
                    m.d.comb += pe[i][j].enable.eq(enable)

        with m.Switch(state):
            with m.Case(State.IDLE):
                set_all(load=0, enable=0)
                with m.If(self.start):
                    m.d.sync += state.eq(State.FETCH)
                    m.d.sync += kb.eq(0)
                    m.d.sync += kb_end.eq(Mux(self.kblocks == 0, MAX_KBLOCKS, self.kblocks))
                    m.d.sync += first_mac.eq(~self.accumulate)

            with m.Case(State.FETCH):
                set_all(load=0, enable=0)
                m.d.sync += state.eq(State.LATCH)

            with m.Case(State.LATCH):
                set_all(load=0, enable=0)
                for n in range(N * N):
                    m.d.sync += a_tile[n].as_value().eq(self.rd_data_a[n * 16 : (n + 1) * 16])
                    m.d.sync += b_tile[n].as_value().eq(self.rd_data_b[n * 16 : (n + 1) * 16])
                m.d.sync += k.eq(0)
                m.d.sync += state.eq(State.MAC)

            with m.Case(State.MAC):
                set_all(load=first_mac, enable=~first_mac)
                m.d.sync += first_mac.eq(0)
                with m.If(k == N - 1):
                    with m.If(kb + 1 == kb_end):
                        m.d.sync += state.eq(Mux(self.evict, State.EVICT, State.DONE))
                    with m.Else():
                        m.d.sync += kb.eq(kb + 1)
                        m.d.sync += state.eq(State.FETCH)
                with m.Else():
                    m.d.sync += k.eq(k + 1)

            with m.Case(State.EVICT):
                # TODO: epilogue (add_en + slot_c read; scale_en; act_sel) -- currently round-and-write only
                set_all(load=0, enable=0)
                m.d.comb += self.wr_addr.eq(self.slot_c)
                m.d.comb += self.wr_en.eq(1)
                for i in range(N):
                    for j in range(N):
                        m.d.comb += self.wr_data[(i * N + j) * 16 : (i * N + j + 1) * 16].eq(pe[i][j].result.as_value())
                m.d.sync += state.eq(State.DONE)

            with m.Case(State.DONE):
                set_all(load=0, enable=0)
                m.d.comb += self.done.eq(1)
                with m.If(~self.start):
                    m.d.sync += state.eq(State.IDLE)

        return m
