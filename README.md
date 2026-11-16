# hardware

A bfloat16 matrix multiply-accumulate (MMA) unit written in
[Amaranth HDL](https://github.com/amaranth-lang/amaranth).

It builds bottom-up from arithmetic primitives to a 4×4 PE array that computes
`D = A·B` over bf16 matrices. Each PE aligns products onto a 48-bit fixed-point
grid and accumulates exactly, rounding to bf16 once on drain. `MMAUnit` streams
`K = 4·kblocks` (up to 64) from D-SRAM tiles into one of four persistent
accumulators (`acc0..acc3`) and rounds to bf16 on evict, or hands an accumulator
to a software epilogue via `acc_read` (FP32, overlapping the next matmul).

## Setup

```bash
nix develop
```

The flake builds the Python venv from `uv.lock` (uv2nix) and points `uv` at it,
with the project installed editable so `src/` is on the import path and tests
can `from bfloat16 import ...` directly. Re-enter the shell after changing
`pyproject.toml` or `uv.lock`. Without nix, `uv sync` builds a local `.venv/`.

## Test

```bash
uv run pytest test/ -v              # all tests
uv run pytest test/ --vcd           # also dump .vcd waveforms
uv run pyrefly check                # type-check
```

## Lint

```bash
ruff check --fix && ruff format
```

## Analysis

```bash
uv run python analysis/synth.py         # yosys area + LUT6 depth per block
uv run python analysis/pnr.py           # nextpnr-ecp5 Fmax + utilization, gated on Fmax floors
uv run python analysis/window_sweep.py  # fixed-point accuracy gate over representative bf16 workloads
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

- `fixed_pe.py` (`FixedPE`) is the PE: multiply, align onto the fixed-point
  grid, accumulate. Out-of-window products and accumulator wraps raise sticky
  `any_dropped` / `any_overflow` flags.
- `accumulator.py` (`Accumulator`) holds the four 48-bit banks and the pipelined
  drain to bf16 and fp32; one bank can drain while another accumulates.
- `mma.py` (`MMA`) is the fixed K=4 16-PE array (what the synth/P&R reports and
  board tops use).
- `mma_stream.py` (`MMAUnit`) is the K-block streaming unit with double-buffered
  tile fetch, plus `acc_read`, which drains an accumulator to an FP32 tile
  register for the software epilogue (bias, scale, activation run as ALU/SFU
  ops).

The rest are standalone arithmetic primitives (carry-select and Kogge-Stone
adders, mantissa multiplier, normalizer, rounder).

### `test/`

- `amaranth.sim` benches.
- `test_mma.py` holds the reference model: exact accumulation, single RTNE round
  to bf16.

The per-primitive files cover the building blocks.
