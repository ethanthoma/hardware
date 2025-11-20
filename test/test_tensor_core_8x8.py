import itertools
import sys

import numpy as np
import pytest
from amaranth.hdl import Period
from amaranth.sim import Simulator

from bfloat16 import BF16
from tensor_core_8x8 import TensorCore8x8


def matrix_to_bf16_flat(matrix):
    flat = []
    rows, cols = matrix.shape
    for i, j in itertools.product(range(rows), range(cols)):
        flat.append(BF16.from_float(float(matrix[i, j])))
    return flat


def bf16_flat_to_matrix(flat_bf16, rows, cols):
    matrix = np.zeros((rows, cols), dtype=np.float32)
    for i, j in itertools.product(range(rows), range(cols)):
        bf16 = flat_bf16[i * cols + j]
        matrix[i, j] = bf16.to_float()
    return matrix


def bf16_matmul(A, B, C):
    A_bf16 = np.array(
        [[BF16.from_float(float(A[i, j])).to_float() for j in range(A.shape[1])] for i in range(A.shape[0])],
        dtype=np.float32,
    )
    B_bf16 = np.array(
        [[BF16.from_float(float(B[i, j])).to_float() for j in range(B.shape[1])] for i in range(B.shape[0])],
        dtype=np.float32,
    )
    C_bf16 = np.array(
        [[BF16.from_float(float(C[i, j])).to_float() for j in range(C.shape[1])] for i in range(C.shape[0])],
        dtype=np.float32,
    )

    result = np.zeros_like(C_bf16)
    for i, j in itertools.product(range(A.shape[0]), range(B.shape[1])):
        acc = C_bf16[i, j]
        for k in range(A.shape[1]):
            prod = BF16.from_float(A_bf16[i, k] * B_bf16[k, j]).to_float()
            acc = BF16.from_float(acc + prod).to_float()
        result[i, j] = acc
    return result


def set_matrix(ctx, port, matrix_flat):
    for idx, bf16 in enumerate(matrix_flat):
        sign, exp, mant = bf16.unpack()
        ctx.set(port[idx], {"sign": sign, "exponent": exp, "mantissa": mant})


def get_matrix(ctx, port, size):
    result = []
    for idx in range(size):
        result_struct = ctx.get(port[idx])
        bf16 = BF16.pack(result_struct["sign"], result_struct["exponent"], result_struct["mantissa"])
        result.append(bf16)
    return result


@pytest.mark.slow
def test_tensor_core_8x8_identity(request):
    dut = TensorCore8x8()

    async def bench(ctx):
        np.random.seed(42)
        A = np.random.randn(8, 8).astype(np.float32) * 0.5
        I = np.eye(8, dtype=np.float32)
        C = np.zeros((8, 8), dtype=np.float32)

        A_flat = matrix_to_bf16_flat(A)
        I_flat = matrix_to_bf16_flat(I)
        C_flat = matrix_to_bf16_flat(C)

        set_matrix(ctx, dut.a_matrix, I_flat)
        set_matrix(ctx, dut.b_matrix, A_flat)
        set_matrix(ctx, dut.c_matrix, C_flat)

        ctx.set(dut.start, 1)
        await ctx.tick()

        ctx.set(dut.start, 0)

        for _ in range(20):
            done = ctx.get(dut.done)
            if done:
                break
            await ctx.tick()

        assert done, "Computation did not complete"

        result_flat = get_matrix(ctx, dut.d_matrix, 64)
        result = bf16_flat_to_matrix(result_flat, 8, 8)

        expected = bf16_matmul(I, A, C)

        for i, j in itertools.product(range(8), range(8)):
            error = abs(result[i, j] - expected[i, j])
            assert error < 0.01, f"Identity[{i},{j}]: got {result[i, j]}, expected {expected[i, j]}"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"TensorCore8x8_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


@pytest.mark.slow
def test_tensor_core_8x8_zero(request):
    dut = TensorCore8x8()

    async def bench(ctx):
        np.random.seed(123)
        A = np.random.randn(8, 8).astype(np.float32) * 0.5
        Z = np.zeros((8, 8), dtype=np.float32)
        C = np.zeros((8, 8), dtype=np.float32)

        A_flat = matrix_to_bf16_flat(A)
        Z_flat = matrix_to_bf16_flat(Z)
        C_flat = matrix_to_bf16_flat(C)

        set_matrix(ctx, dut.a_matrix, Z_flat)
        set_matrix(ctx, dut.b_matrix, A_flat)
        set_matrix(ctx, dut.c_matrix, C_flat)

        ctx.set(dut.start, 1)
        await ctx.tick()

        ctx.set(dut.start, 0)

        for _ in range(20):
            done = ctx.get(dut.done)
            if done:
                break
            await ctx.tick()

        assert done, "Computation did not complete"

        result_flat = get_matrix(ctx, dut.d_matrix, 64)
        result = bf16_flat_to_matrix(result_flat, 8, 8)

        for i, j in itertools.product(range(8), range(8)):
            assert abs(result[i, j]) < 0.01, f"Zero[{i},{j}]: got {result[i, j]}, expected 0.0"

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"TensorCore8x8_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


@pytest.mark.slow
def test_tensor_core_8x8_basic(request):
    dut = TensorCore8x8()

    async def bench(ctx):
        np.random.seed(456)
        A = np.random.randn(8, 8).astype(np.float32) * 0.3
        B = np.random.randn(8, 8).astype(np.float32) * 0.3
        C = np.zeros((8, 8), dtype=np.float32)

        A_flat = matrix_to_bf16_flat(A)
        B_flat = matrix_to_bf16_flat(B)
        C_flat = matrix_to_bf16_flat(C)

        set_matrix(ctx, dut.a_matrix, A_flat)
        set_matrix(ctx, dut.b_matrix, B_flat)
        set_matrix(ctx, dut.c_matrix, C_flat)

        ctx.set(dut.start, 1)
        await ctx.tick()

        ctx.set(dut.start, 0)

        for _ in range(20):
            done = ctx.get(dut.done)
            if done:
                break
            await ctx.tick()

        assert done, "Computation did not complete"

        result_flat = get_matrix(ctx, dut.d_matrix, 64)
        result = bf16_flat_to_matrix(result_flat, 8, 8)

        expected = bf16_matmul(A, B, C)

        for i, j in itertools.product(range(8), range(8)):
            error = abs(result[i, j] - expected[i, j])
            rel_error = error / abs(expected[i, j]) if abs(expected[i, j]) > 1e-6 else error
            assert rel_error < 0.05, (
                f"Basic[{i},{j}]: got {result[i, j]}, expected {expected[i, j]}, rel_error={rel_error:.6f}"
            )

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"TensorCore8x8_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


@pytest.mark.slow
def test_tensor_core_8x8_with_c(request):
    dut = TensorCore8x8()

    async def bench(ctx):
        np.random.seed(789)
        A = np.random.randn(8, 8).astype(np.float32) * 0.25
        B = np.random.randn(8, 8).astype(np.float32) * 0.25
        C = np.random.randn(8, 8).astype(np.float32) * 0.25

        A_flat = matrix_to_bf16_flat(A)
        B_flat = matrix_to_bf16_flat(B)
        C_flat = matrix_to_bf16_flat(C)

        set_matrix(ctx, dut.a_matrix, A_flat)
        set_matrix(ctx, dut.b_matrix, B_flat)
        set_matrix(ctx, dut.c_matrix, C_flat)

        ctx.set(dut.start, 1)
        await ctx.tick()

        ctx.set(dut.start, 0)

        for _ in range(20):
            done = ctx.get(dut.done)
            if done:
                break
            await ctx.tick()

        assert done, "Computation did not complete"

        result_flat = get_matrix(ctx, dut.d_matrix, 64)
        result = bf16_flat_to_matrix(result_flat, 8, 8)

        expected = bf16_matmul(A, B, C)

        for i, j in itertools.product(range(8), range(8)):
            error = abs(result[i, j] - expected[i, j])
            rel_error = error / abs(expected[i, j]) if abs(expected[i, j]) > 1e-6 else error
            assert rel_error < 0.05, (
                f"WithC[{i},{j}]: got {result[i, j]}, expected {expected[i, j]}, rel_error={rel_error:.6f}"
            )

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"TensorCore8x8_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()


@pytest.mark.slow
def test_tensor_core_8x8_powers_of_two(request):
    dut = TensorCore8x8()

    async def bench(ctx):
        A = np.array([[2 ** ((i - j) % 4 - 2) for j in range(8)] for i in range(8)], dtype=np.float32)
        B = np.array([[2 ** ((j - i) % 4 - 2) for j in range(8)] for i in range(8)], dtype=np.float32)
        C = np.zeros((8, 8), dtype=np.float32)

        A_flat = matrix_to_bf16_flat(A)
        B_flat = matrix_to_bf16_flat(B)
        C_flat = matrix_to_bf16_flat(C)

        set_matrix(ctx, dut.a_matrix, A_flat)
        set_matrix(ctx, dut.b_matrix, B_flat)
        set_matrix(ctx, dut.c_matrix, C_flat)

        ctx.set(dut.start, 1)
        await ctx.tick()

        ctx.set(dut.start, 0)

        for _ in range(20):
            done = ctx.get(dut.done)
            if done:
                break
            await ctx.tick()

        assert done, "Computation did not complete"

        result_flat = get_matrix(ctx, dut.d_matrix, 64)
        result = bf16_flat_to_matrix(result_flat, 8, 8)

        expected = bf16_matmul(A, B, C)

        for i, j in itertools.product(range(8), range(8)):
            error = abs(result[i, j] - expected[i, j])
            rel_error = error / abs(expected[i, j]) if abs(expected[i, j]) > 1e-6 else error
            assert rel_error < 0.02, (
                f"Powers[{i},{j}]: got {result[i, j]}, expected {expected[i, j]}, rel_error={rel_error:.6f}"
            )

    sim = Simulator(dut)
    sim.add_clock(Period(us=1))
    sim.add_testbench(bench)

    if request.config.getoption("--vcd"):
        vcd_name = f"TensorCore8x8_{sys._getframe().f_code.co_name}.vcd"
        with sim.write_vcd(vcd_name):
            sim.run()
    else:
        sim.run()
