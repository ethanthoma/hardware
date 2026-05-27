import itertools

import numpy as np
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from mma_4x4 import MMA4x4

N = 4


def to_flat(matrix):
    return [BF16.from_float(float(matrix[i, j])) for i, j in itertools.product(range(N), range(N))]


def to_matrix(flat):
    out = np.zeros((N, N), dtype=np.float32)
    for i, j in itertools.product(range(N), range(N)):
        out[i, j] = flat[i * N + j].to_float()
    return out


def bf16_matmul(A, B, C):
    """Reference: D = A·B + C with each k-step's accumulator rounded to bf16 once (single-rounding FMA)."""
    A = np.array([[BF16.from_float(float(x)).to_float() for x in row] for row in A], dtype=np.float32)
    B = np.array([[BF16.from_float(float(x)).to_float() for x in row] for row in B], dtype=np.float32)
    C = np.array([[BF16.from_float(float(x)).to_float() for x in row] for row in C], dtype=np.float32)

    result = np.zeros_like(C)
    for i, j in itertools.product(range(N), range(N)):
        acc = np.float64(C[i, j])
        for k in range(N):
            acc = np.float64(BF16.from_float(float(np.float64(A[i, k]) * np.float64(B[k, j]) + acc)).to_float())
        result[i, j] = acc
    return result


def run_mma(request, A, B, C):
    dut = MMA4x4()
    out = {}

    async def bench(ctx):
        for port, matrix in ((dut.a_matrix, A), (dut.b_matrix, B), (dut.c_matrix, C)):
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
        with sim.write_vcd(f"MMA4x4_{request.node.name}.vcd"):
            sim.run()
    else:
        sim.run()
    return out["result"]


def assert_close(result, expected, tol):
    for i, j in itertools.product(range(N), range(N)):
        error = abs(result[i, j] - expected[i, j])
        rel = error / abs(expected[i, j]) if abs(expected[i, j]) > 1e-6 else error
        assert rel < tol, f"[{i},{j}]: got {result[i, j]}, expected {expected[i, j]}, rel={rel:.6f}"


def test_identity(request):
    np.random.seed(42)
    A = np.random.randn(N, N).astype(np.float32) * 0.5
    eye = np.eye(N, dtype=np.float32)
    zero = np.zeros((N, N), dtype=np.float32)
    assert_close(run_mma(request, eye, A, zero), bf16_matmul(eye, A, zero), 0.01)


def test_zero(request):
    np.random.seed(123)
    A = np.random.randn(N, N).astype(np.float32) * 0.5
    zero = np.zeros((N, N), dtype=np.float32)
    result = run_mma(request, zero, A, zero)
    for i, j in itertools.product(range(N), range(N)):
        assert abs(result[i, j]) < 0.01, f"[{i},{j}]: got {result[i, j]}, expected 0.0"


def test_basic(request):
    np.random.seed(456)
    A = np.random.randn(N, N).astype(np.float32) * 0.3
    B = np.random.randn(N, N).astype(np.float32) * 0.3
    zero = np.zeros((N, N), dtype=np.float32)
    assert_close(run_mma(request, A, B, zero), bf16_matmul(A, B, zero), 0.25)


def test_with_c(request):
    np.random.seed(789)
    A = np.random.randn(N, N).astype(np.float32) * 0.25
    B = np.random.randn(N, N).astype(np.float32) * 0.25
    C = np.random.randn(N, N).astype(np.float32) * 0.25
    assert_close(run_mma(request, A, B, C), bf16_matmul(A, B, C), 0.20)


def test_powers_of_two(request):
    A = np.array([[2.0 ** ((i - j) % 4 - 2) for j in range(N)] for i in range(N)], dtype=np.float32)
    B = np.array([[2.0 ** ((j - i) % 4 - 2) for j in range(N)] for i in range(N)], dtype=np.float32)
    zero = np.zeros((N, N), dtype=np.float32)
    assert_close(run_mma(request, A, B, zero), bf16_matmul(A, B, zero), 0.02)
