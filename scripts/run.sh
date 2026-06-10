#!/usr/bin/env bash
# usage: scripts/run.sh [top] [-f]
# build analysis/<top>.py (default mma_flags) and load build/<top>.bit onto the ECP5-5G EVN;
# -f writes SPI flash instead of volatile SRAM
set -euo pipefail
cd "$(dirname "$0")/.."

top=mma_flags
flash=()
for arg in "$@"; do
  case "$arg" in
  -f) flash=(-f) ;;
  *) top=$arg ;;
  esac
done

uv run python "analysis/$top.py"
openFPGALoader -b ecp5_evn ${flash[@]+"${flash[@]}"} "build/$top.bit"
