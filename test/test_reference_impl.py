import itertools
import numpy as np
from bfloat16 import BF16


def bf16_matmul_current(A, B, C):
    """Current reference implementation from test file"""
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
            # Simulate FMA: (a × b) + c with single rounding
            a_f64 = np.float64(A_bf16[i, k])
            b_f64 = np.float64(B_bf16[k, j])
            c_f64 = np.float64(acc)
            fma_result_f64 = (a_f64 * b_f64) + c_f64
            acc = BF16.from_float(float(fma_result_f64)).to_float()
        result[i, j] = acc
    return result


def test_simple_case():
    # Same test as hardware: 2×2 in top-left
    A = np.array([
        [1, 2, 0, 0, 0, 0, 0, 0],
        [3, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)

    B = np.array([
        [5, 6, 0, 0, 0, 0, 0, 0],
        [7, 8, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)

    C = np.zeros((8, 8), dtype=np.float32)

    result = bf16_matmul_current(A, B, C)

    expected = {
        (0, 0): 19,  # 1*5 + 2*7
        (0, 1): 22,  # 1*6 + 2*8
        (1, 0): 43,  # 3*5 + 4*7
        (1, 1): 50,  # 3*6 + 4*8
    }

    print("Reference implementation results:")
    for (i, j), exp_val in expected.items():
        ref_val = result[i, j]
        print(f"D[{i},{j}]: Reference={ref_val}, Expected={exp_val}, Match={abs(ref_val - exp_val) < 0.01}")


if __name__ == "__main__":
    test_simple_case()
