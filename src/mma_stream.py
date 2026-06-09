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
    FETCH0 = 1
    LATCH0 = 2
    MAC = 3
    FLUSH = 4  # let the last pipelined product reach acc before draining
    DRAIN = 5  # hold acc one cycle so the pipelined drain settles before EVICT reads result
    EVICT = 6
    DONE = 7


class MMAUnit(wiring.Component):
    """K-block streaming MMA, double-buffered tile fetch, K = 4 * kblocks (kblocks=0 means 16)."""

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

        a_tile = [[Signal(BFloat16, name=f"a_tile_{b}_{n}") for n in range(N * N)] for b in range(2)]
        b_tile = [[Signal(BFloat16, name=f"b_tile_{b}_{n}") for n in range(N * N)] for b in range(2)]

        state = Signal(State)
        # +1 width so kb_mac / kb_end can hold MAX_KBLOCKS (16) when kblocks==0
        kb_mac = Signal(range(MAX_KBLOCKS + 1))
        kb_end = Signal(range(MAX_KBLOCKS + 1))
        k = Signal(range(N))
        mac_buf = Signal()
        first_mac = Signal()
        prefetch_kb = Signal(range(MAX_KBLOCKS + 1))

        # TODO (cc4 spec gap): D-SRAM port shape (256-bit per slot per cycle, two read ports) -- pin in ISA.md.
        with m.Switch(state):
            with m.Case(State.MAC):
                m.d.comb += prefetch_kb.eq(kb_mac + 1)
            with m.Default():
                m.d.comb += prefetch_kb.eq(0)
        m.d.comb += self.rd_addr_a.eq(self.slot_a + prefetch_kb)
        m.d.comb += self.rd_addr_b.eq(self.slot_b + prefetch_kb)

        with m.Switch(k):
            for k_val in range(N):
                with m.Case(k_val):
                    for i in range(N):
                        for j in range(N):
                            a_sel = Mux(
                                mac_buf, a_tile[1][i * N + k_val].as_value(), a_tile[0][i * N + k_val].as_value()
                            )
                            b_sel = Mux(
                                mac_buf, b_tile[1][k_val * N + j].as_value(), b_tile[0][k_val * N + j].as_value()
                            )
                            m.d.comb += pe[i][j].a.as_value().eq(a_sel)
                            m.d.comb += pe[i][j].b.as_value().eq(b_sel)

        def set_all(load, enable):
            for i in range(N):
                for j in range(N):
                    m.d.comb += pe[i][j].load.eq(load)
                    m.d.comb += pe[i][j].enable.eq(enable)

        def latch_buf(buf_idx: int):
            for n in range(N * N):
                m.d.sync += a_tile[buf_idx][n].as_value().eq(self.rd_data_a[n * 16 : (n + 1) * 16])
                m.d.sync += b_tile[buf_idx][n].as_value().eq(self.rd_data_b[n * 16 : (n + 1) * 16])

        with m.Switch(state):
            with m.Case(State.IDLE):
                set_all(load=0, enable=0)
                with m.If(self.start):
                    m.d.sync += state.eq(State.FETCH0)
                    m.d.sync += kb_mac.eq(0)
                    m.d.sync += kb_end.eq(Mux(self.kblocks == 0, MAX_KBLOCKS, self.kblocks))
                    m.d.sync += mac_buf.eq(0)
                    m.d.sync += first_mac.eq(~self.accumulate)

            with m.Case(State.FETCH0):
                set_all(load=0, enable=0)
                m.d.sync += state.eq(State.LATCH0)

            with m.Case(State.LATCH0):
                set_all(load=0, enable=0)
                latch_buf(0)
                m.d.sync += k.eq(0)
                m.d.sync += state.eq(State.MAC)

            with m.Case(State.MAC):
                set_all(load=first_mac, enable=~first_mac)
                m.d.sync += first_mac.eq(0)

                # Prefetch of kb_mac+1 lands one cycle after rd_addr appears,
                # which is k==1 (rd_addr is combinational from prefetch_kb).
                with m.If((k == 1) & (kb_mac + 1 < kb_end)):
                    with m.If(mac_buf == 0):
                        latch_buf(1)
                    with m.Else():
                        latch_buf(0)

                with m.If(k == N - 1):
                    with m.If(kb_mac + 1 == kb_end):
                        m.d.sync += state.eq(State.FLUSH)
                    with m.Else():
                        m.d.sync += kb_mac.eq(kb_mac + 1)
                        m.d.sync += mac_buf.eq(~mac_buf)
                        m.d.sync += k.eq(0)
                with m.Else():
                    m.d.sync += k.eq(k + 1)

            with m.Case(State.FLUSH):
                set_all(load=0, enable=0)
                m.d.sync += state.eq(Mux(self.evict, State.DRAIN, State.DONE))

            with m.Case(State.DRAIN):
                set_all(load=0, enable=0)
                m.d.sync += state.eq(State.EVICT)

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
