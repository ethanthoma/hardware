# hardware

A bfloat16 tensor core written in
[Amaranth HDL](https://github.com/amaranth-lang/amaranth).

It builds bottom-up from arithmetic primitives to an 8×8 systolic MAC array
(`TensorCore8x8`) that computes `D = A·B + C` over bf16 matrices, accumulating
in extended (26-bit mantissa) precision and rounding to bf16 only at the output.

## Setup

```bash
uv sync
```

This installs the project editable, putting `src/` on the import path so tests
can `from bfloat16 import ...` directly.

## Test

```bash
uv run pytest test/ -v              # all tests
uv run pytest test/ -m "not slow"   # skip the full 8x8 sweeps
uv run pytest test/ --vcd           # also dump .vcd waveforms
```

## Lint

```bash
ruff check --fix && ruff format
```

## Layout

### `src/`

- `bf16_mac.py` (`BF16_MAC`) is the fused multiply-add core.
- `pe_mac.py` wraps it with a registered accumulator.
- `tensor_core_8x8.py` is the 64-PE array.

The rest are standalone arithmetic primitives (adders, aligner, normalizer, LZA,
multiplier, rounder).

### `test/`

- `amaranth.sim` benches.
- `test_tensor_core_8x8.py` holds the single-rounding FMA reference model.

The per-primitive files cover the building blocks.
