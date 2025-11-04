import json
from pathlib import Path

import pytest

TIMING_BUDGETS = {
    "carry_select_26": 16.0,
    "kogge_stone_26": 11.0,
    "lza_26": 18.0,
    "bf16_adder_optimized": 125.0,
}

TIMING_TARGETS = {
    "carry_select_26": 15.0,
    "kogge_stone_26": 10.0,
    "lza_26": 17.0,
    "bf16_adder_optimized": 28.0,
}


def analyze_carry_select_adder(width, block_size):
    num_blocks = (width + block_size - 1) // block_size
    first_block_delay = block_size
    parallel_block_delay = block_size
    mux_chain_delay = (num_blocks - 1).bit_length()
    return first_block_delay + parallel_block_delay + mux_chain_delay


def analyze_kogge_stone(width):
    levels = (width - 1).bit_length()
    return levels * 2


def analyze_lza(width):
    gp_generation = 1
    kogge_stone = analyze_kogge_stone(width)
    sum_prediction = 1
    priority_encoder = width.bit_length()
    return gp_generation + kogge_stone + sum_prediction + priority_encoder


def analyze_bf16_adder_optimized():
    exp_compare = 3
    exp_diff = 8
    abs_value = 8
    shift_clamp = 2
    alignment = 10
    mag_compare = 9
    mantissa_sub = analyze_carry_select_adder(26, 6)
    lza = analyze_lza(26)
    normalize = 10
    rounding = 10
    result_exp = 8
    sign_determine = 3
    return (
        exp_compare
        + exp_diff
        + abs_value
        + shift_clamp
        + alignment
        + mag_compare
        + mantissa_sub
        + lza
        + normalize
        + rounding
        + result_exp
        + sign_determine
    )


def test_carry_select_timing(request):
    delay = analyze_carry_select_adder(width=26, block_size=6)
    request.node.timing_info = f"{delay}-delta (target: {TIMING_TARGETS['carry_select_26']}-delta, budget: {TIMING_BUDGETS['carry_select_26']}-delta)"
    assert delay <= TIMING_BUDGETS["carry_select_26"], (
        f"Carry-select: {delay}-delta > {TIMING_BUDGETS['carry_select_26']}-delta"
    )


def test_kogge_stone_timing(request):
    delay = analyze_kogge_stone(width=26)
    request.node.timing_info = f"{delay}-delta (target: {TIMING_TARGETS['kogge_stone_26']}-delta, budget: {TIMING_BUDGETS['kogge_stone_26']}-delta)"
    assert delay <= TIMING_BUDGETS["kogge_stone_26"], (
        f"Kogge-Stone: {delay}-delta > {TIMING_BUDGETS['kogge_stone_26']}-delta"
    )


def test_lza_timing(request):
    delay = analyze_lza(width=26)
    request.node.timing_info = (
        f"{delay}-delta (target: {TIMING_TARGETS['lza_26']}-delta, budget: {TIMING_BUDGETS['lza_26']}-delta)"
    )
    assert delay <= TIMING_BUDGETS["lza_26"], f"LZA: {delay}-delta > {TIMING_BUDGETS['lza_26']}-delta"


def test_bf16_adder_timing(request):
    delay = analyze_bf16_adder_optimized()
    request.node.timing_info = f"{delay}-delta (target: {TIMING_TARGETS['bf16_adder_optimized']}-delta, budget: {TIMING_BUDGETS['bf16_adder_optimized']}-delta)"
    assert delay <= TIMING_BUDGETS["bf16_adder_optimized"], (
        f"BF16 adder: {delay}-delta > {TIMING_BUDGETS['bf16_adder_optimized']}-delta"
    )


def test_no_timing_regressions(tmp_path):
    baseline_file = Path(__file__).parent / "timing_baseline.json"

    current_timing = {
        "carry_select_26": analyze_carry_select_adder(26, 6),
        "kogge_stone_26": analyze_kogge_stone(26),
        "lza_26": analyze_lza(26),
        "bf16_adder_optimized": analyze_bf16_adder_optimized(),
    }

    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)

        regressions = []
        for component, delay in current_timing.items():
            baseline_delay = baseline.get(component)
            if baseline_delay and delay > baseline_delay * 1.05:
                regressions.append(
                    f"{component}: {delay}-delta vs {baseline_delay}-delta (+{delay - baseline_delay:.1f}-delta)"
                )

        if regressions:
            pytest.fail("Timing regressions:\n" + "\n".join(f"  {r}" for r in regressions))
    else:
        with open(baseline_file, "w") as f:
            json.dump(current_timing, f, indent=2)
        print(f"Created baseline: {baseline_file}")
