import itertools
import struct

import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from mma_stream import MMAUnit, N

MAX_CYCLES = 500


def bf16_rtne(x: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    if (bits >> 23) & 0xFF == 0xFF:
        return x
    low, high = bits & 0xFFFF, bits >> 16
    if low > 0x8000 or (low == 0x8000 and (high & 1)):
        high += 1
    return struct.unpack("<f", struct.pack("<I", high << 16))[0]


def bf16_matmul(A, B):
    K = A.shape[1]
    Aq = np.array([[BF16.from_float(float(x)).to_float() for x in row] for row in A], dtype=np.float64)
    Bq = np.array([[BF16.from_float(float(x)).to_float() for x in row] for row in B], dtype=np.float64)
    out = np.zeros((N, N), dtype=np.float64)
    for i, j in itertools.product(range(N), range(N)):
        out[i, j] = bf16_rtne(sum(float(Aq[i, k]) * float(Bq[k, j]) for k in range(K)))
    return out


def pack_tile(tile: np.ndarray) -> int:
    word = 0
    for n, (i, j) in enumerate(itertools.product(range(N), range(N))):
        word |= BF16.from_float(float(tile[i, j])).to_bits() << (n * 16)
    return word


def unpack_tile(word: int) -> np.ndarray:
    out = np.zeros((N, N), dtype=np.float64)
    for n, (i, j) in enumerate(itertools.product(range(N), range(N))):
        out[i, j] = BF16.from_bits((word >> (n * 16)) & 0xFFFF).to_float()
    return out


def split_into_kblocks(M: np.ndarray, axis: str, kblocks: int):
    """A: split columns into kblocks 4x4 tiles. B: split rows into kblocks 4x4 tiles."""
    tiles = []
    for kb in range(kblocks):
        if axis == "a":
            tiles.append(M[:, kb * N : (kb + 1) * N])
        else:
            tiles.append(M[kb * N : (kb + 1) * N, :])
    return tiles


def run_stream(A, B, kblocks: int, request, kblocks_encoded: int | None = None) -> np.ndarray:
    dut = MMAUnit()
    a_tiles = split_into_kblocks(A, "a", kblocks)
    b_tiles = split_into_kblocks(B, "b", kblocks)
    slot_a, slot_b, slot_c = 8, 16, 32
    a_mem = {slot_a + kb: pack_tile(a_tiles[kb]) for kb in range(kblocks)}
    b_mem = {slot_b + kb: pack_tile(b_tiles[kb]) for kb in range(kblocks)}
    captured: dict[str, np.ndarray] = {}
    encoded = kblocks_encoded if kblocks_encoded is not None else kblocks % 16

    async def bench(ctx):
        ctx.set(dut.slot_a, slot_a)
        ctx.set(dut.slot_b, slot_b)
        ctx.set(dut.slot_c, slot_c)
        ctx.set(dut.kblocks, encoded)
        ctx.set(dut.evict, 1)
        ctx.set(dut.accumulate, 0)
        ctx.set(dut.start, 1)

        for _ in range(MAX_CYCLES):
            ctx.set(dut.rd_data_a, a_mem.get(ctx.get(dut.rd_addr_a), 0))
            ctx.set(dut.rd_data_b, b_mem.get(ctx.get(dut.rd_addr_b), 0))
            if ctx.get(dut.wr_en):
                captured["d"] = unpack_tile(ctx.get(dut.wr_data))
                captured["wr_addr"] = ctx.get(dut.wr_addr)
            if ctx.get(dut.done):
                break
            await ctx.tick()

        ctx.set(dut.start, 0)
        assert "d" in captured, "MMAUnit never asserted wr_en"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    if request.config.getoption("--vcd"):
        with sim.write_vcd(f"MMAUnit_{request.node.name}.vcd"):
            sim.run()
    else:
        sim.run()
    assert captured["wr_addr"] == slot_c
    return captured["d"]


def run_chain(ops, request):
    """Run [(A, B, kblocks, accumulate, evict, acc_d), ...] on one DUT.
    Returns (evictions, flags): evicted tiles in op order, and (any_dropped, any_overflow) at each op's done."""
    dut = MMAUnit()
    slot_a, slot_b, slot_c = 8, 16, 32
    evictions: list[np.ndarray] = []
    flags: list[tuple[bool, bool]] = []

    async def bench(ctx):
        ctx.set(dut.slot_a, slot_a)
        ctx.set(dut.slot_b, slot_b)
        ctx.set(dut.slot_c, slot_c)

        for A, B, kblocks, accumulate, evict, acc_d in ops:
            a_tiles = split_into_kblocks(A, "a", kblocks)
            b_tiles = split_into_kblocks(B, "b", kblocks)
            a_mem = {slot_a + kb: pack_tile(a_tiles[kb]) for kb in range(kblocks)}
            b_mem = {slot_b + kb: pack_tile(b_tiles[kb]) for kb in range(kblocks)}

            ctx.set(dut.kblocks, kblocks % 16)
            ctx.set(dut.accumulate, 1 if accumulate else 0)
            ctx.set(dut.evict, 1 if evict else 0)
            ctx.set(dut.acc_d, acc_d)
            ctx.set(dut.start, 1)

            evicted = None
            for _ in range(MAX_CYCLES):
                ctx.set(dut.rd_data_a, a_mem.get(ctx.get(dut.rd_addr_a), 0))
                ctx.set(dut.rd_data_b, b_mem.get(ctx.get(dut.rd_addr_b), 0))
                if ctx.get(dut.wr_en):
                    evicted = unpack_tile(ctx.get(dut.wr_data))
                if ctx.get(dut.done):
                    break
                await ctx.tick()
            assert ctx.get(dut.done), "op never completed"
            flags.append((bool(ctx.get(dut.any_dropped)), bool(ctx.get(dut.any_overflow))))
            if evicted is not None:
                evictions.append(evicted)
            ctx.set(dut.start, 0)
            await ctx.tick()

        assert evictions, "chain produced no eviction"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    if request.config.getoption("--vcd"):
        with sim.write_vcd(f"MMAUnit_{request.node.name}.vcd"):
            sim.run()
    else:
        sim.run()
    return evictions, flags


def assert_bit_exact(got, want):
    for i, j in itertools.product(range(N), range(N)):
        assert got[i, j] == want[i, j], f"[{i},{j}]: got {got[i, j]}, want {want[i, j]}"


def test_kblocks_1_matches_mma(request):
    np.random.seed(456)
    A = np.random.randn(N, N).astype(np.float32) * 0.3
    B = np.random.randn(N, N).astype(np.float32) * 0.3
    assert_bit_exact(run_stream(A, B, 1, request), bf16_matmul(A, B))


def test_kblocks_2(request):
    np.random.seed(7)
    K = N * 2
    A = np.random.randn(N, K).astype(np.float32) * 0.25
    B = np.random.randn(K, N).astype(np.float32) * 0.25
    assert_bit_exact(run_stream(A, B, 2, request), bf16_matmul(A, B))


def test_kblocks_4(request):
    np.random.seed(11)
    K = N * 4
    A = np.random.randn(N, K).astype(np.float32) * 0.2
    B = np.random.randn(K, N).astype(np.float32) * 0.2
    assert_bit_exact(run_stream(A, B, 4, request), bf16_matmul(A, B))


def test_kblocks_16_max(request):
    np.random.seed(13)
    K = N * 16
    A = np.random.randn(N, K).astype(np.float32) * 0.1
    B = np.random.randn(K, N).astype(np.float32) * 0.1
    assert_bit_exact(run_stream(A, B, 16, request), bf16_matmul(A, B))


def test_kblocks_zero_means_sixteen(request):
    np.random.seed(13)
    K = N * 16
    A = np.random.randn(N, K).astype(np.float32) * 0.1
    B = np.random.randn(K, N).astype(np.float32) * 0.1
    assert_bit_exact(run_stream(A, B, 16, request, kblocks_encoded=0), bf16_matmul(A, B))


def cycles_to_done(kblocks: int) -> int:
    dut = MMAUnit()
    K = N * kblocks
    rng = np.random.default_rng(0)
    A = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    B = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    a_tiles = split_into_kblocks(A, "a", kblocks)
    b_tiles = split_into_kblocks(B, "b", kblocks)
    slot_a, slot_b, slot_c = 8, 16, 32
    a_mem = {slot_a + kb: pack_tile(a_tiles[kb]) for kb in range(kblocks)}
    b_mem = {slot_b + kb: pack_tile(b_tiles[kb]) for kb in range(kblocks)}
    measured: dict[str, int] = {}

    async def bench(ctx):
        ctx.set(dut.slot_a, slot_a)
        ctx.set(dut.slot_b, slot_b)
        ctx.set(dut.slot_c, slot_c)
        ctx.set(dut.kblocks, kblocks % 16)
        ctx.set(dut.evict, 1)
        ctx.set(dut.accumulate, 0)
        ctx.set(dut.start, 1)

        for cyc in range(MAX_CYCLES):
            ctx.set(dut.rd_data_a, a_mem.get(ctx.get(dut.rd_addr_a), 0))
            ctx.set(dut.rd_data_b, b_mem.get(ctx.get(dut.rd_addr_b), 0))
            if ctx.get(dut.done):
                measured["cyc"] = cyc
                break
            await ctx.tick()
        assert "cyc" in measured, "never reached done"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    sim.run()
    return measured["cyc"]


def test_cycle_budget_per_kblock():
    cycles_per_kblock = 4
    base = cycles_to_done(1)
    for kblocks in (2, 4, 16):
        assert cycles_to_done(kblocks) - base == cycles_per_kblock * (kblocks - 1)


def test_accumulate_chain_matches_single_mma(request):
    # Chain (K=8 acc=0) + (K=8 acc=1, evict) == single K=16; proves no mid-chain BF16 round.
    np.random.seed(17)
    K = N * 4
    A = np.random.randn(N, K).astype(np.float32) * 0.2
    B = np.random.randn(K, N).astype(np.float32) * 0.2
    A_first, A_second = A[:, : N * 2], A[:, N * 2 :]
    B_first, B_second = B[: N * 2, :], B[N * 2 :, :]
    evictions, _ = run_chain(
        [
            (A_first, B_first, 2, False, False, 0),
            (A_second, B_second, 2, True, True, 0),
        ],
        request,
    )
    assert_bit_exact(evictions[0], bf16_matmul(A, B))


def test_interleaved_chains_stay_independent(request):
    # two accumulate chains interleaved op-by-op into acc0/acc1; each evict matches its own
    # single-K=8-mma reference, proving the banks don't bleed into each other
    np.random.seed(23)
    K = N * 2
    Ax = np.random.randn(N, K).astype(np.float32) * 0.2
    Bx = np.random.randn(K, N).astype(np.float32) * 0.2
    Ay = np.random.randn(N, K).astype(np.float32) * 0.2
    By = np.random.randn(K, N).astype(np.float32) * 0.2
    evictions, _ = run_chain(
        [
            (Ax[:, :N], Bx[:N, :], 1, False, False, 0),
            (Ay[:, :N], By[:N, :], 1, False, False, 1),
            (Ax[:, N:], Bx[N:, :], 1, True, True, 0),
            (Ay[:, N:], By[N:, :], 1, True, True, 1),
        ],
        request,
    )
    assert_bit_exact(evictions[0], bf16_matmul(Ax, Bx))
    assert_bit_exact(evictions[1], bf16_matmul(Ay, By))


def test_flags_stick_per_acc_bank(request):
    # an overflowing chain in acc1 must not contaminate acc0's sticky flags
    ones = np.ones((N, N), dtype=np.float32)
    hot_a, hot_b = ones * 128.0, ones * 64.0  # K=4 row of 2**13 products wraps signed(48)
    _, flags = run_chain(
        [
            (ones * 0.25, ones * 0.25, 1, False, False, 0),
            (hot_a, hot_b, 1, False, False, 1),
            (ones * 0.25, ones * 0.25, 1, True, True, 0),
        ],
        request,
    )
    assert flags[1] == (False, True)
    assert flags[2] == (False, False)


def fp32_matmul(A, B):
    """D = A*B with bf16 operands, exact accumulation, single RTNE to fp32 (acc_read's contract)."""
    K = A.shape[1]
    Aq = np.array([[BF16.from_float(float(x)).to_float() for x in row] for row in A], dtype=np.float64)
    Bq = np.array([[BF16.from_float(float(x)).to_float() for x in row] for row in B], dtype=np.float64)
    out = np.zeros((N, N), dtype=np.float64)
    for i, j in itertools.product(range(N), range(N)):
        out[i, j] = float(np.float32(sum(float(Aq[i, k]) * float(Bq[k, j]) for k in range(K))))
    return out


def unpack_fp32_tile(word: int) -> np.ndarray:
    out = np.zeros((N, N), dtype=np.float64)
    for n, (i, j) in enumerate(itertools.product(range(N), range(N))):
        out[i, j] = struct.unpack("<f", struct.pack("<I", (word >> (n * 32)) & 0xFFFFFFFF))[0]
    return out


def tile_memories(A, B, kblocks: int, slot_a: int, slot_b: int):
    a_tiles = split_into_kblocks(A, "a", kblocks)
    b_tiles = split_into_kblocks(B, "b", kblocks)
    a_mem = {slot_a + kb: pack_tile(a_tiles[kb]) for kb in range(kblocks)}
    b_mem = {slot_b + kb: pack_tile(b_tiles[kb]) for kb in range(kblocks)}
    return a_mem, b_mem


def run_acc_read_during_mma(filled, running, running_acc_d: int, read_delay: int, request):
    """Fill acc0 with `filled` = (A, B, kblocks) (no evict), then run `running` into `running_acc_d` with evict
    while an acc_read of acc0 -> a5 starts `read_delay` cycles after that mma's start. Returns the read tile,
    the evicted tile, the mma's start-to-done cycles, and acc0's sticky flags at acc_read_done."""
    dut = MMAUnit()
    slot_a, slot_b, slot_c = 8, 16, 32
    read_target = 5
    out: dict = {}

    async def bench(ctx):
        ctx.set(dut.slot_a, slot_a)
        ctx.set(dut.slot_b, slot_b)
        ctx.set(dut.slot_c, slot_c)

        A, B, kblocks = filled
        a_mem, b_mem = tile_memories(A, B, kblocks, slot_a, slot_b)
        ctx.set(dut.kblocks, kblocks % 16)
        ctx.set(dut.accumulate, 0)
        ctx.set(dut.evict, 0)
        ctx.set(dut.acc_d, 0)
        ctx.set(dut.start, 1)
        for _ in range(MAX_CYCLES):
            ctx.set(dut.rd_data_a, a_mem.get(ctx.get(dut.rd_addr_a), 0))
            ctx.set(dut.rd_data_b, b_mem.get(ctx.get(dut.rd_addr_b), 0))
            if ctx.get(dut.done):
                break
            await ctx.tick()
        assert ctx.get(dut.done), "fill mma never completed"
        ctx.set(dut.start, 0)
        await ctx.tick()

        A, B, kblocks = running
        a_mem, b_mem = tile_memories(A, B, kblocks, slot_a, slot_b)
        ctx.set(dut.kblocks, kblocks % 16)
        ctx.set(dut.evict, 1)
        ctx.set(dut.acc_d, running_acc_d)
        ctx.set(dut.acc_read_bank, 0)
        ctx.set(dut.acc_read_target, read_target)
        ctx.set(dut.start, 1)
        for cycle in range(MAX_CYCLES):
            if cycle == read_delay:
                ctx.set(dut.acc_read_start, 1)
            ctx.set(dut.rd_data_a, a_mem.get(ctx.get(dut.rd_addr_a), 0))
            ctx.set(dut.rd_data_b, b_mem.get(ctx.get(dut.rd_addr_b), 0))
            if ctx.get(dut.wr_en):
                out["evicted"] = unpack_tile(ctx.get(dut.wr_data))
            if ctx.get(dut.tile_wr_en):
                assert ctx.get(dut.tile_wr_addr) == read_target
                assert "read" not in out, "acc_read wrote its tile twice"
                out["read"] = unpack_fp32_tile(ctx.get(dut.tile_wr_data))
            if ctx.get(dut.done) and "mma_cycles" not in out:
                out["mma_cycles"] = cycle
                ctx.set(dut.start, 0)
            if ctx.get(dut.acc_read_done) and "read_flags" not in out:
                out["read_flags"] = (bool(ctx.get(dut.any_dropped)), bool(ctx.get(dut.any_overflow)))
                ctx.set(dut.acc_read_start, 0)
            if "mma_cycles" in out and "read_flags" in out:
                break
            await ctx.tick()
        assert "mma_cycles" in out, "running mma never completed"
        assert "read_flags" in out, "acc_read never completed"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    if request.config.getoption("--vcd"):
        with sim.write_vcd(f"MMAUnit_{request.node.name}.vcd"):
            sim.run()
    else:
        sim.run()
    return out


def test_acc_read_overlaps_mma_at_every_offset(request):
    # Start the acc_read at every cycle of a concurrent mma, including the one where its capture collides with
    # the mma's DRAIN: the read must stay exact, the evict must stay exact, and the mma must never slow down.
    rng = np.random.default_rng(31)
    A0 = rng.standard_normal((N, N)).astype(np.float32) * 0.3
    B0 = rng.standard_normal((N, N)).astype(np.float32) * 0.3
    kblocks = 2
    A1 = rng.standard_normal((N, N * kblocks)).astype(np.float32) * 0.3
    B1 = rng.standard_normal((N * kblocks, N)).astype(np.float32) * 0.3
    mma_cycles = cycles_to_done(kblocks)
    for read_delay in range(mma_cycles + 2):
        out = run_acc_read_during_mma((A0, B0, 1), (A1, B1, kblocks), 1, read_delay, request)
        assert_bit_exact(out["read"], fp32_matmul(A0, B0))
        assert_bit_exact(out["evicted"], bf16_matmul(A1, B1))
        assert out["mma_cycles"] == mma_cycles, f"read_delay={read_delay} stalled the mma"


def test_acc_read_keeps_precision_evict_rounds_away(request):
    # 1 + 2**-10 survives the fp32 read but not the bf16 evict: the software epilogue sees full precision
    A0 = np.zeros((N, N), dtype=np.float32)
    B0 = np.zeros((N, N), dtype=np.float32)
    A0[:, 0], B0[0, :] = 1.0, 1.0
    A0[:, 1], B0[1, :] = 2.0**-5, 2.0**-5
    out = run_acc_read_during_mma((A0, B0, 1), (A0, B0, 1), 1, 0, request)
    assert out["read"][0, 0] == 1.0 + 2.0**-10
    assert out["evicted"][0, 0] == 1.0


def test_acc_read_reports_the_read_banks_flags(request):
    ones = np.ones((N, N), dtype=np.float32)
    hot_a, hot_b = ones * 128.0, ones * 64.0  # K=4 row of 2**13 products wraps signed(48)
    out = run_acc_read_during_mma((hot_a, hot_b, 1), (ones * 0.25, ones * 0.25, 1), 1, 0, request)
    assert out["read_flags"] == (False, True)


def test_acc_read_captures_before_a_same_cycle_mma_overwrites_its_bank(request):
    # WAR: acc_read(acc0) and a fresh mma(acc0) start on the same cycle. The read must see the old acc0 --
    # it captures within a cycle, and the mma's first write lands at least three cycles after its start.
    rng = np.random.default_rng(37)
    A0 = rng.standard_normal((N, N)).astype(np.float32) * 0.3
    B0 = rng.standard_normal((N, N)).astype(np.float32) * 0.3
    A1 = rng.standard_normal((N, N)).astype(np.float32) * 0.3
    B1 = rng.standard_normal((N, N)).astype(np.float32) * 0.3
    out = run_acc_read_during_mma((A0, B0, 1), (A1, B1, 1), 0, 0, request)
    assert_bit_exact(out["read"], fp32_matmul(A0, B0))
    assert_bit_exact(out["evicted"], bf16_matmul(A1, B1))
