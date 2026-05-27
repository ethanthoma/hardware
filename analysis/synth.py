import re
import subprocess
import sys
from typing import Callable, NamedTuple

from amaranth.back import rtlil
from amaranth.lib import wiring

from aligner import Aligner
from bf16_mac import BF16_MAC
from carry_select_adder import CarrySelectAdder, CarrySelectSubtractor
from fused_exp_diff import FusedExponentDifference
from lza import LeadingZeroAnticipator
from mantissa_multiplier import MantissaMultiplier
from mma import MMA
from normalizer import Normalizer
from parallel_prefix import KoggeStone
from pe_mac import PE_MAC
from rounder import Rounder


def run_yosys(script: str) -> str:
    result = subprocess.run(["yosys", "-"], input=script, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


class Block(NamedTuple):
    name: str
    build: Callable[[], wiring.Component]
    combinational: bool
    slow: bool = False


BLOCKS = [
    Block("KoggeStone", lambda: KoggeStone(26), True),
    Block("CarrySelectAdder", lambda: CarrySelectAdder(26, 6), True),
    Block("CarrySelectSubtractor", lambda: CarrySelectSubtractor(26, 6), True),
    Block("MantissaMultiplier", MantissaMultiplier, True),
    Block("Aligner", lambda: Aligner(26), True),
    Block("Normalizer", lambda: Normalizer(26), True),
    Block("LeadingZeroAnticipator", lambda: LeadingZeroAnticipator(26), True),
    Block("Rounder", lambda: Rounder(7), True),
    Block("FusedExponentDifference", FusedExponentDifference, True),
    Block("BF16_MAC", BF16_MAC, True),
    Block("PE_MAC", PE_MAC, False),
    Block("MMA", MMA, False),
]


def synthesize(block: Block) -> str:
    il = rtlil.convert(block.build(), name=block.name)
    script = f"read_rtlil <<rtlil\n{il}\nrtlil\nsynth -top {block.name} -flatten\nstat"
    if block.combinational:
        script += "\nabc -lut 6\nltp"
    return run_yosys(script)


def cell_count(yosys_output: str) -> int:
    match = re.search(r"Number of cells:\s+(\d+)", yosys_output)
    assert match, "synth produced no cell count"
    return int(match.group(1))


def lut_depth(yosys_output: str) -> int | None:
    match = re.search(r"length=(\d+)", yosys_output)
    return int(match.group(1)) if match else None


def main() -> None:
    include_slow = "--all" in sys.argv
    print(f"{'block':<26}{'cells':>8}{'depth (LUT6)':>14}", flush=True)
    print("-" * 48, flush=True)
    for block in BLOCKS:
        if block.slow and not include_slow:
            continue
        output = synthesize(block)
        depth = lut_depth(output)
        print(f"{block.name:<26}{cell_count(output):>8}{(str(depth) if depth is not None else '—'):>14}", flush=True)


if __name__ == "__main__":
    main()
