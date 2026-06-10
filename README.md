# hardware

A bfloat16 matrix multiply-accumulate (MMA) unit written in
[Amaranth HDL](https://github.com/amaranth-lang/amaranth).

It builds bottom-up from arithmetic primitives to a 4×4×4 MAC array (`MMA`) that
computes `D = A·B + C` over bf16 matrices, accumulating in extended (26-bit
mantissa) precision and rounding to bf16 only at the output.

## Setup

```bash
uv sync
```

This installs the project editable, putting `src/` on the import path so tests
can `from bfloat16 import ...` directly.

## Test

```bash
uv run pytest test/ -v              # all tests
uv run pytest test/ --vcd           # also dump .vcd waveforms
```

## Lint

```bash
ruff check --fix && ruff format
```

## Run on the board

```bash
scripts/run.sh             # build analysis/mma_flags.py and load it onto the ECP5-5G EVN
scripts/run.sh mma_led     # any analysis/ board top
scripts/run.sh blink -f    # -f writes SPI flash instead of volatile SRAM
```

Each top is self-checking on the LEDs — see its docstring for the legend.

## Layout

### `src/`

- `bf16_mac.py` (`BF16_MAC`) is the fused multiply-add core.
- `pe_mac.py` wraps it with a registered accumulator.
- `mma.py` (`MMA`) is the 16-PE array.

The rest are standalone arithmetic primitives (adders, aligner, normalizer, LZA,
multiplier, rounder).

### `test/`

- `amaranth.sim` benches.
- `test_mma.py` holds the single-rounding FMA reference model.

The per-primitive files cover the building blocks.
