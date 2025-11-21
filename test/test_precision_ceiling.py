"""Show the precision ceiling with BF16 accumulator vs ideal FP32"""
import numpy as np
from bfloat16 import BF16


def test_precision_ceiling():
    """Compare our current precision (BF16 acc) vs ceiling (FP32 acc)"""
    np.random.seed(456)
    A = np.random.randn(8, 8).astype(np.float32) * 0.3
    B = np.random.randn(8, 8).astype(np.float32) * 0.3

    errors_bf16 = []
    errors_fp32 = []

    for i in range(8):
        for j in range(8):
            # Ground truth (FP64)
            truth = sum(
                BF16.from_float(A[i, k]).to_float() * BF16.from_float(B[k, j]).to_float()
                for k in range(8)
            )

            # Current approach: BF16 accumulator (what hardware does)
            acc_bf16 = 0.0
            for k in range(8):
                a_val = BF16.from_float(A[i, k]).to_float()
                b_val = BF16.from_float(B[k, j]).to_float()
                # Simulate FMA with 26-bit internal, round to BF16
                product = a_val * b_val
                fma_result = product + acc_bf16
                acc_bf16 = BF16.from_float(fma_result).to_float()

            # Ideal approach: FP32 accumulator
            acc_fp32 = np.float32(0.0)
            for k in range(8):
                a_val = BF16.from_float(A[i, k]).to_float()
                b_val = BF16.from_float(B[k, j]).to_float()
                product = np.float32(a_val) * np.float32(b_val)
                acc_fp32 = acc_fp32 + product
            final_fp32 = BF16.from_float(float(acc_fp32)).to_float()

            # Calculate relative errors
            if abs(truth) > 1e-6:
                err_bf16 = abs(acc_bf16 - truth) / abs(truth)
                err_fp32 = abs(final_fp32 - truth) / abs(truth)
            else:
                err_bf16 = abs(acc_bf16 - truth)
                err_fp32 = abs(final_fp32 - truth)

            errors_bf16.append(err_bf16)
            errors_fp32.append(err_fp32)

    # Count by error bucket
    def count_by_threshold(errors, thresholds):
        counts = []
        for threshold in thresholds:
            counts.append(sum(1 for e in errors if e < threshold))
        return counts

    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20]
    counts_bf16 = count_by_threshold(errors_bf16, thresholds)
    counts_fp32 = count_by_threshold(errors_fp32, thresholds)

    print("Precision Analysis: BF16 Accumulator vs FP32 Accumulator")
    print("=" * 70)
    print(f"{'Threshold':<15} {'BF16 Acc':<20} {'FP32 Acc':<20} {'Improvement'}")
    print("-" * 70)

    for i, thresh in enumerate(thresholds):
        bf16_pct = counts_bf16[i] / 64 * 100
        fp32_pct = counts_fp32[i] / 64 * 100
        improvement = fp32_pct - bf16_pct
        print(f"< {thresh*100:>4.0f}% error  {counts_bf16[i]:>2d}/64 ({bf16_pct:>5.1f}%)   "
              f"{counts_fp32[i]:>2d}/64 ({fp32_pct:>5.1f}%)   +{improvement:>5.1f}%")

    print("\n" + "=" * 70)
    print(f"Max error (BF16): {max(errors_bf16)*100:.2f}%")
    print(f"Max error (FP32): {max(errors_fp32)*100:.2f}%")
    print(f"Mean error (BF16): {np.mean(errors_bf16)*100:.2f}%")
    print(f"Mean error (FP32): {np.mean(errors_fp32)*100:.2f}%")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    if max(errors_bf16) < 0.05:
        print("✅ Current BF16 accumulator: < 5% error on all elements")
    elif max(errors_bf16) < 0.10:
        print("⚠️  Current BF16 accumulator: < 10% error on all elements")
    else:
        print("❌ Current BF16 accumulator: some elements > 10% error")

    print(f"\nFP32 accumulator would improve {counts_fp32[2] - counts_bf16[2]} more elements to < 5%")


if __name__ == "__main__":
    test_precision_ceiling()
