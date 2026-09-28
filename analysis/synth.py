import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, NamedTuple

from amaranth.back import rtlil
from amaranth.lib import wiring

from carry_select_adder import CarrySelectAdder, CarrySelectSubtractor
from fixed_pe import FixedMAC, FixedPE
from mantissa_multiplier import MantissaMultiplier
from mma import MMA
from normalizer import Normalizer
from parallel_prefix import KoggeStone
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
    Block("Normalizer", lambda: Normalizer(26), True),
    Block("Rounder", lambda: Rounder(7), True),
    Block("MMA", MMA, False),
    Block("FixedMAC", FixedMAC, True),
    Block("FixedPE", FixedPE, False),
]


class Synthesis(NamedTuple):
    log: str
    cell_count: int


def synthesize(block: Block) -> Synthesis:
    il = rtlil.convert(block.build(), name=block.name)
    with tempfile.TemporaryDirectory() as directory:
        stat_path = Path(directory) / "stat.json"
        script = f"read_rtlil <<rtlil\n{il}\nrtlil\nsynth -top {block.name} -flatten\ntee -q -o {stat_path} stat -json"
        if block.combinational:
            script += "\nabc -lut 6\nltp"
        log = run_yosys(script)
        cell_count = json.loads(stat_path.read_text())["design"]["num_cells"]
    assert isinstance(cell_count, int), "yosys stat -json has no design.num_cells"
    assert cell_count > 0, f"{block.name} synthesized to zero cells"
    return Synthesis(log, cell_count)


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
        synthesis = synthesize(block)
        depth = lut_depth(synthesis.log)
        print(f"{block.name:<26}{synthesis.cell_count:>8}{(str(depth) if depth is not None else '-'):>14}", flush=True)


if __name__ == "__main__":
    main()
