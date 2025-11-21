"""Show where FP32 accumulation significantly helps"""
import numpy as np
from bfloat16 import BF16


def test_many_small_values():
    """Test accumulating many small values - where FP32 helps most"""
    # Create 8 small values that should sum to ~0.5
    values = [0.0625] * 8  # Each is 1/16

    truth = sum(values)  # Should be 0.5

    # BF16 accumulation
    acc_bf16 = 0.0
    for i, val in enumerate(values):
        val_bf16 = BF16.from_float(val).to_float()
        fma_result = val_bf16 + acc_bf16
        acc_bf16 = BF16.from_float(fma_result).to_float()

    # FP32 accumulation
    acc_fp32 = np.float32(0.0)
    for val in values:
        val_bf16 = BF16.from_float(val).to_float()
        acc_fp32 = acc_fp32 + np.float32(val_bf16)
    final_bf16_from_fp32 = BF16.from_float(float(acc_fp32)).to_float()

    print("Test: Accumulating 8 × 0.0625 = 0.5")
    print(f"Ground truth:        {truth:.6f}")
    print(f"BF16 accumulation:   {acc_bf16:.6f}  (error: {abs(acc_bf16 - truth):.6f})")
    print(f"FP32 → BF16:         {final_bf16_from_fp32:.6f}  (error: {abs(final_bf16_from_fp32 - truth):.6f})")


def test_cancellation():
    """Test catastrophic cancellation - where precision matters most"""
    # Large positive + large negative + small value
    # (1000 * 0.001) + (-1000 * 0.001) + (0.1 * 0.1) should be 0.01
    A = [1000.0, -1000.0, 0.1]
    B = [0.001, 0.001, 0.1]

    truth = sum(a * b for a, b in zip(A, B))

    # BF16 accumulation
    acc_bf16 = 0.0
    for a, b in zip(A, B):
        a_bf16 = BF16.from_float(a).to_float()
        b_bf16 = BF16.from_float(b).to_float()
        product = a_bf16 * b_bf16
        fma_result = product + acc_bf16
        acc_bf16 = BF16.from_float(fma_result).to_float()

    # FP32 accumulation
    acc_fp32 = np.float32(0.0)
    for a, b in zip(A, B):
        a_bf16 = BF16.from_float(a).to_float()
        b_bf16 = BF16.from_float(b).to_float()
        product = np.float32(a_bf16) * np.float32(b_bf16)
        acc_fp32 = acc_fp32 + product
    final_bf16_from_fp32 = BF16.from_float(float(acc_fp32)).to_float()

    print("\nTest: Catastrophic cancellation (1000*0.001 - 1000*0.001 + 0.1*0.1)")
    print(f"Ground truth:        {truth:.6f}")
    print(f"BF16 accumulation:   {acc_bf16:.6f}  (error: {abs(acc_bf16 - truth):.6f})")
    print(f"FP32 → BF16:         {final_bf16_from_fp32:.6f}  (error: {abs(final_bf16_from_fp32 - truth):.6f})")
    print(f"Improvement: {abs(acc_bf16 - truth) / max(abs(final_bf16_from_fp32 - truth), 1e-8):.1f}x")


if __name__ == "__main__":
    test_many_small_values()
    test_cancellation()
