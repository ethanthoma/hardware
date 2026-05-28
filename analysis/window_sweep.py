import math
import sys
from typing import Callable, NamedTuple

import numpy as np

from bfloat16 import BF16

K = 4
WINDOW_MAX_EXP = 13
WINDOW_MIN_EXP = -18
ROW_SUM_BOUND = 2.0**15


class Workload(NamedTuple):
    name: str
    sample: Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]


def bf16_quantize(x: np.ndarray) -> np.ndarray:
    return np.array([BF16.from_float(float(v)).to_float() for v in x.ravel()], dtype=np.float64).reshape(x.shape)


def normal_pair(rng: np.random.Generator, n: int, a_scale: float, b_scale: float) -> tuple[np.ndarray, np.ndarray]:
    return rng.standard_normal(n) * a_scale, rng.standard_normal(n) * b_scale


def bf16_uniform(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, n)


def outlier_mix(rng: np.random.Generator, n: int, fraction: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    outliers = rng.random(n) < fraction
    a = np.where(outliers, a * scale, a)
    return a, b


WORKLOADS = [
    Workload("act x weight, fan_in=128", lambda rng, n: normal_pair(rng, n, 1.0, 1.0 / math.sqrt(128))),
    Workload("act x weight, fan_in=1024", lambda rng, n: normal_pair(rng, n, 1.0, 1.0 / math.sqrt(1024))),
    Workload("attn QK, d=64 (pre-scale)", lambda rng, n: normal_pair(rng, n, 1.0, 1.0)),
    Workload("attn QK, d=64 (post-scale)", lambda rng, n: normal_pair(rng, n, 1.0, 1.0 / math.sqrt(64))),
    Workload(
        "layernorm-ish: clipped N(0,1)", lambda rng, n: (np.clip(rng.standard_normal(n), -4, 4), bf16_uniform(rng, n))
    ),
    Workload("outlier-heavy: 1% at 10x scale", lambda rng, n: outlier_mix(rng, n, 0.01, 10.0)),
]


def classify(products: np.ndarray) -> dict[str, float]:
    nonzero = products[products != 0.0]
    mag = np.abs(nonzero)
    log2 = np.log2(mag) if mag.size else np.array([])
    n = products.size
    return {
        "n": n,
        "underflow_to_zero_pct": 100.0 * (products == 0.0).sum() / n,
        "below_window_pct": 100.0 * (log2 < WINDOW_MIN_EXP).sum() / n,
        "in_window_pct": 100.0 * ((log2 >= WINDOW_MIN_EXP) & (log2 <= WINDOW_MAX_EXP)).sum() / n,
        "above_window_pct": 100.0 * (log2 > WINDOW_MAX_EXP).sum() / n,
        "p99_mag": float(np.quantile(mag, 0.99)) if mag.size else 0.0,
    }


def row_overflow_pct(products: np.ndarray, k: int) -> float:
    rows = products[: (products.size // k) * k].reshape(-1, k)
    return 100.0 * (np.abs(rows.sum(axis=1)) > ROW_SUM_BOUND).sum() / rows.shape[0]


def sweep(workload: Workload, n: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a, b = workload.sample(rng, n)
    products = bf16_quantize(a) * bf16_quantize(b)
    stats = classify(products)
    stats["row_overflow_pct"] = row_overflow_pct(products, K)
    return stats


def verdict(s: dict[str, float]) -> str:
    clipped = s["below_window_pct"] + s["above_window_pct"]
    if s["above_window_pct"] > 0.1 or s["row_overflow_pct"] > 0.1:
        return "NEEDS WIDER (high-side or row overflow)"
    if clipped < 1.0 and s["row_overflow_pct"] < 0.01:
        return "OK for fixed-point"
    return "MARGINAL"


ACCEPTABLE_VERDICT = "OK for fixed-point"


def main() -> None:
    header = f"{'workload':<32}{'in':>8}{'< 2^-18':>10}{'> 2^13':>9}{'row OF':>9}{'p99 mag':>12}  verdict"
    print(header, flush=True)
    print("-" * (len(header) + 32), flush=True)
    regressions = []
    for w in WORKLOADS:
        s = sweep(w, n=200_000, seed=0xA11)
        v = verdict(s)
        print(
            f"{w.name:<32}"
            f"{s['in_window_pct']:>7.2f}%"
            f"{s['below_window_pct'] + s['underflow_to_zero_pct']:>9.2f}%"
            f"{s['above_window_pct']:>8.3f}%"
            f"{s['row_overflow_pct']:>8.3f}%"
            f"{s['p99_mag']:>12.3g}  "
            f"{v}",
            flush=True,
        )
        if v != ACCEPTABLE_VERDICT:
            regressions.append(f"{w.name}: {v}")

    if regressions:
        print("\nFAIL: workload(s) outside the window:", *regressions, sep="\n  ", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
