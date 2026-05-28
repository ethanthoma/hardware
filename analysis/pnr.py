import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, NamedTuple

from amaranth import Cat, Module, Signal
from amaranth.back import rtlil
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from fixed_pe import FixedPE
from mma import MMA

DEVICE = "um5g-85k"
PACKAGE = "CABGA756"
SPEED = "8"
TARGET_MHZ = 100


def leaf_values(port):
    """Yield underlying Value(s) for a scalar port or list of ports."""
    items = port if isinstance(port, list) else [port]
    for sub in items:
        yield sub.as_value() if hasattr(sub, "as_value") else sub


def make_pnr_top(dut_cls: Callable[[], wiring.Component]) -> wiring.Component:
    """Wrap a DUT so its wide IO becomes internal: flop-bounded inputs from a counter,
    flop-captured XOR-reduction of outputs. Only `result` is a physical IO."""

    class PnrTop(wiring.Component):
        result: Out(1)

        def elaborate(self, _):
            m = Module()
            m.submodules.dut = dut = dut_cls()

            counter = Signal(16)
            m.d.sync += counter.eq(counter + 1)

            outputs = []
            input_idx = 0
            for name, member in dut.signature.members.items():
                attr = getattr(dut, name)
                values = list(leaf_values(attr))
                if member.flow == wiring.In:
                    for v in values:
                        reg = Signal(len(v))
                        m.d.sync += reg.eq(counter + input_idx)
                        m.d.comb += v.eq(reg)
                        input_idx += 1
                else:
                    outputs.extend(values)

            captured = Signal()
            m.d.sync += captured.eq(Cat(*outputs).xor())
            m.d.comb += self.result.eq(captured)
            return m

    return PnrTop()


class Block(NamedTuple):
    name: str
    build: Callable[[], wiring.Component]


BLOCKS = [
    Block("MMA", lambda: make_pnr_top(MMA)),
    Block("FixedPE", lambda: make_pnr_top(FixedPE)),
]


def synth_json(block: Block, out_json: Path) -> None:
    il = rtlil.convert(block.build(), name=block.name)
    script = f"read_rtlil <<RTLIL\n{il}\nRTLIL\nsynth_ecp5 -top {block.name} -json {out_json}"
    result = subprocess.run(["yosys", "-q", "-"], input=script, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def pnr(json_in: Path, report_out: Path) -> str:
    result = subprocess.run(
        [
            "nextpnr-ecp5",
            "--json",
            str(json_in),
            f"--{DEVICE}",
            "--package",
            PACKAGE,
            "--speed",
            SPEED,
            "--freq",
            str(TARGET_MHZ),
            "--report",
            str(report_out),
            "--textcfg",
            "/dev/null",
            "--seed",
            "1",
            "--timing-allow-fail",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stderr


def parse_fmax(log: str) -> float | None:
    matches = re.findall(r"Max frequency for clock[^:]*:\s+([\d.]+)\s+MHz", log)
    return float(matches[-1]) if matches else None


def parse_utilization(report_path: Path) -> dict[str, int]:
    data = json.loads(report_path.read_text())
    return {cell: counts["used"] for cell, counts in data.get("utilization", {}).items() if counts.get("used", 0) > 0}


INTERESTING_CELLS = ("TRELLIS_COMB", "TRELLIS_FF", "MULT18X18D", "DP16KD")


def main() -> None:
    print(f"{'block':<12}{'fmax MHz':>10}" + "".join(f"{c:>14}" for c in INTERESTING_CELLS), flush=True)
    print("-" * (22 + 14 * len(INTERESTING_CELLS)), flush=True)
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        for block in BLOCKS:
            json_in = tmp / f"{block.name}.json"
            report = tmp / f"{block.name}.report.json"
            synth_json(block, json_in)
            log = pnr(json_in, report)
            util = parse_utilization(report)
            fmax = parse_fmax(log)
            cells = "".join(f"{util.get(c, 0):>14}" for c in INTERESTING_CELLS)
            print(f"{block.name:<12}{(f'{fmax:.1f}' if fmax else '-'):>10}{cells}", flush=True)


if __name__ == "__main__":
    main()
