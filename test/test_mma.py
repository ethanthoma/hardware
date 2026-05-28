import itertools
import struct

import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from mma import MMA

N = 4


def to_flat(matrix):
    return [BF16.from_float(float(matrix[i, j])) for i, j in itertools.product(range(N), range(N))]


def to_matrix(flat):
    out = np.zeros((N, N), dtype=np.float32)
    for i, j in itertools.product(range(N), range(N)):
        out[i, j] = flat[i * N + j].to_float()
    return out


def bf16_rtne(x: float) -> float:
    """Cast x to bf16 with round-to-nearest-even, matching the Rounder hardware."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    if (bits >> 23) & 0xFF == 0xFF:
        return x
    low, high = bits & 0xFFFF, bits >> 16
    if low > 0x8000 or (low == 0x8000 and (high & 1)):
        high += 1
    return struct.unpack("<f", struct.pack("<I", high << 16))[0]


def bf16_matmul(A, B):
    """D = A*B with bf16 operands, exact accumulation, single RTNE at drain (FixedPE's contract)."""
    A = np.array([[BF16.from_float(float(x)).to_float() for x in row] for row in A], dtype=np.float32)
    B = np.array([[BF16.from_float(float(x)).to_float() for x in row] for row in B], dtype=np.float32)

    result = np.zeros((N, N), dtype=np.float32)
    for i, j in itertools.product(range(N), range(N)):
        result[i, j] = bf16_rtne(sum(float(A[i, k]) * float(B[k, j]) for k in range(N)))
    return result


def run_mma(request, A, B):
    dut = MMA()
    out = {}

    async def bench(ctx):
        for port, matrix in ((dut.a_matrix, A), (dut.b_matrix, B)):
            for idx, bf16 in enumerate(to_flat(matrix)):
                sign, exp, mant = bf16.unpack()
                ctx.set(port[idx], {"sign": sign, "exponent": exp, "mantissa": mant})

        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)

        for _ in range(20):
            if ctx.get(dut.done):
                break
            await ctx.tick()
        assert ctx.get(dut.done), "computation did not complete"

        flat = []
        for idx in range(N * N):
            s = ctx.get(dut.d_matrix[idx])
            flat.append(BF16.pack(s["sign"], s["exponent"], s["mantissa"]))
        out["result"] = to_matrix(flat)

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)
    if request.config.getoption("--vcd"):
        with sim.write_vcd(f"MMA_{request.node.name}.vcd"):
            sim.run()
    else:
        sim.run()
    return out["result"]


def assert_bit_exact(result, expected):
    for i, j in itertools.product(range(N), range(N)):
        assert result[i, j] == expected[i, j], f"[{i},{j}]: got {result[i, j]}, expected {expected[i, j]}"


def test_identity(request):
    np.random.seed(42)
    A = np.random.randn(N, N).astype(np.float32) * 0.5
    eye = np.eye(N, dtype=np.float32)
    assert_bit_exact(run_mma(request, eye, A), bf16_matmul(eye, A))


def test_zero(request):
    np.random.seed(123)
    A = np.random.randn(N, N).astype(np.float32) * 0.5
    zero = np.zeros((N, N), dtype=np.float32)
    result = run_mma(request, zero, A)
    for i, j in itertools.product(range(N), range(N)):
        assert result[i, j] == 0.0, f"[{i},{j}]: got {result[i, j]}, expected 0.0"


def test_basic(request):
    np.random.seed(456)
    A = np.random.randn(N, N).astype(np.float32) * 0.3
    B = np.random.randn(N, N).astype(np.float32) * 0.3
    assert_bit_exact(run_mma(request, A, B), bf16_matmul(A, B))


def test_powers_of_two(request):
    A = np.array([[2.0 ** ((i - j) % 4 - 2) for j in range(N)] for i in range(N)], dtype=np.float32)
    B = np.array([[2.0 ** ((j - i) % 4 - 2) for j in range(N)] for i in range(N)], dtype=np.float32)
    assert_bit_exact(run_mma(request, A, B), bf16_matmul(A, B))
