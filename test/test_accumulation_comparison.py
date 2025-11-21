"""Compare BF16 vs FP32 accumulation precision"""
import numpy as np
from bfloat16 import BF16


def test_accumulation_precision():
    """Compare BF16 accumulation vs FP32 accumulation"""
    np.random.seed(456)

    # Test case: D[0,0] from our failing test
    A_row = [-0.200195, -0.149414, 0.185547, 0.169922, 0.404297, 0.488281, 0.090332, 0.134766]
    B_col = [-0.304688, 0.345703, 0.091797, 0.165039, 0.024902, -0.285156, 0.640625, -0.041748]

    # Ground truth (FP64)
    truth = sum(a * b for a, b in zip(A_row, B_col))

    # BF16 accumulation (what hardware does)
    acc_bf16 = 0.0
    for i, (a, b) in enumerate(zip(A_row, B_col)):
        a_bf16 = BF16.from_float(a).to_float()
        b_bf16 = BF16.from_float(b).to_float()
        # FMA in FP64, then quantize to BF16
        product = a_bf16 * b_bf16
        fma_result = product + acc_bf16
        acc_bf16 = BF16.from_float(fma_result).to_float()
        print(f"k={i}: BF16 acc={acc_bf16:.6f}")

    # FP32 accumulation (better!)
    acc_fp32 = np.float32(0.0)
    for i, (a, b) in enumerate(zip(A_row, B_col)):
        a_bf16 = BF16.from_float(a).to_float()
        b_bf16 = BF16.from_float(b).to_float()
        product = np.float32(a_bf16) * np.float32(b_bf16)
        acc_fp32 = acc_fp32 + product
        # Final quantize to BF16
        acc_bf16_from_fp32 = BF16.from_float(float(acc_fp32)).to_float()
        print(f"k={i}: FP32 acc={acc_fp32:.6f}, as BF16={acc_bf16_from_fp32:.6f}")

    final_bf16_from_fp32 = BF16.from_float(float(acc_fp32)).to_float()

    print(f"\n{'='*60}")
    print(f"Ground truth (FP64):        {truth:.6f}")
    print(f"BF16 accumulation:          {acc_bf16:.6f}  (error: {abs(acc_bf16 - truth):.6f})")
    print(f"FP32 accumulation:          {acc_fp32:.6f}  (error: {abs(acc_fp32 - truth):.6f})")
    print(f"FP32 → BF16:                {final_bf16_from_fp32:.6f}  (error: {abs(final_bf16_from_fp32 - truth):.6f})")
    print(f"\nBF16 error: {abs(acc_bf16 - truth) / abs(truth) * 100:.1f}%")
    print(f"FP32 error: {abs(final_bf16_from_fp32 - truth) / abs(truth) * 100:.1f}%")
    print(f"Improvement: {abs(acc_bf16 - truth) / abs(final_bf16_from_fp32 - truth):.1f}x better")


if __name__ == "__main__":
    test_accumulation_precision()
