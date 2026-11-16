from amaranth import *
from amaranth.build import Platform
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out

from bfloat16 import BFloat16, Float32
from normalizer import Normalizer
from rounder import Rounder

FLOAT_BIAS = 127  # bf16 and fp32 share the 8-bit, bias-127 exponent
ACC_BANKS = 4  # acc0..acc3 -- TODO (cc4 MMA.md): confirm 4 once the streaming kernel pattern is written


def decomposed_layout(width: int) -> data.StructLayout:
    return data.StructLayout({"sign": 1, "magnitude": width, "leading_one": range(width)})


def decompose(m: Module, value: Signal):
    """Drain stage 1: sign, |value|, and the index of its leading one."""
    out = Signal(decomposed_layout(len(value)))
    negative = value < 0
    m.d.comb += out.sign.eq(negative)
    m.d.comb += out.magnitude.eq(Mux(negative, -value, value))
    for i in range(len(value)):
        with m.If(out.magnitude[i]):
            m.d.comb += out.leading_one.eq(i)
    return out


def normalize(m: Module, d: data.View) -> Value:
    """Drain stage 2a: shift |acc| so its leading one lands on the top bit."""
    width = len(d.magnitude)
    m.submodules.normalizer = normalizer = Normalizer(width=width)
    m.d.comb += normalizer.value_in.eq(d.magnitude)
    m.d.comb += normalizer.shift_amount.eq(width - 1 - d.leading_one)
    return normalizer.value_out


def round_to_float(m: Module, d: data.View, normalized: Value, lsb_exp: int, layout):
    """Drain stage 2b: round the normalized magnitude to `layout`'s mantissa (RTNE) and assemble that float."""
    fields = data.Layout.cast(layout)
    exponent_bits = Shape.cast(fields["exponent"].shape).width
    mantissa_bits = Shape.cast(fields["mantissa"].shape).width
    assert exponent_bits == 8, "exponent math assumes the 8-bit bias-127 exponent"
    assert mantissa_bits > 0, "a float layout needs a mantissa"
    padding_bits = max(0, mantissa_bits + 3 - len(normalized))  # zeros below the LSB: exact for narrow widths
    padded = Cat(Const(0, padding_bits), normalized)
    width = len(padded)

    mantissa_lo = width - 1 - mantissa_bits
    m.submodules[f"rounder_{mantissa_bits}"] = rounder = Rounder(width=mantissa_bits)
    m.d.comb += rounder.mantissa_in.eq(padded[mantissa_lo : width - 1])
    m.d.comb += rounder.guard.eq(padded[mantissa_lo - 1])
    m.d.comb += rounder.round_bit.eq(padded[mantissa_lo - 2])
    m.d.comb += rounder.sticky.eq(padded[0 : mantissa_lo - 2].any())

    exponent = Signal(signed(12), name=f"exponent_{mantissa_bits}")
    m.d.comb += exponent.eq(lsb_exp + d.leading_one + FLOAT_BIAS + rounder.overflow)

    result = Signal(layout, name=f"result_{mantissa_bits}")
    with m.If(d.magnitude == 0):
        m.d.comb += result.as_value().eq(0)
    with m.Else():
        m.d.comb += result.sign.eq(d.sign)
        m.d.comb += result.exponent.eq(exponent[0:8])
        m.d.comb += result.mantissa.eq(rounder.mantissa_out)
    return result


class Accumulator(wiring.Component):
    """Wide fixed-point accumulator, banked x4. Writes (load/enable) target `acc_sel`; the drain and every
    read-side output (`value`, `result`, `result_fp32`, flags, `result_valid`) follow `drain_sel`, so one bank
    can drain while another accumulates."""

    def __init__(self, width: int = 64, lsb_exp: int = 0):
        assert width >= 2, "a signed accumulator needs a sign bit and a magnitude bit"
        self.width = width
        self.lsb_exp = lsb_exp

        super().__init__(
            {
                "addend": In(signed(width)),
                "addend_dropped": In(1),  # this addend was an out-of-window product forced to 0
                "acc_sel": In(range(ACC_BANKS)),  # bank written by load/enable
                "drain_sel": In(range(ACC_BANKS)),  # bank read by value/result/result_fp32/flags
                "load": In(1),
                "enable": In(1),
                "value": Out(signed(width)),
                "result": Out(BFloat16),
                "result_fp32": Out(Float32),
                "result_valid": Out(1),  # result/result_fp32 reflect drain_sel's bank, settled
                "any_dropped": Out(1),  # sticky: some addend in this accumulation was dropped
                "any_overflow": Out(1),  # sticky: the accumulator wrapped past signed range
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        acc_bank = Array(Signal(signed(self.width), name=f"acc{n}") for n in range(ACC_BANKS))
        dropped_bank = Array(Signal(name=f"dropped{n}") for n in range(ACC_BANKS))
        overflow_bank = Array(Signal(name=f"overflow{n}") for n in range(ACC_BANKS))

        acc = Signal(signed(self.width))
        m.d.comb += acc.eq(acc_bank[self.acc_sel])
        acc_next = Signal(signed(self.width + 1))
        m.d.comb += acc_next.eq(acc + self.addend)
        overflow = Signal()
        m.d.comb += overflow.eq(acc_next[self.width] != acc_next[self.width - 1])

        with m.If(self.load):
            m.d.sync += acc_bank[self.acc_sel].eq(self.addend)
            m.d.sync += dropped_bank[self.acc_sel].eq(self.addend_dropped)
            m.d.sync += overflow_bank[self.acc_sel].eq(0)
        with m.Elif(self.enable):
            m.d.sync += acc_bank[self.acc_sel].eq(acc_next[: self.width])
            m.d.sync += dropped_bank[self.acc_sel].eq(dropped_bank[self.acc_sel] | self.addend_dropped)
            m.d.sync += overflow_bank[self.acc_sel].eq(overflow_bank[self.acc_sel] | overflow)

        drained = Signal(signed(self.width))
        m.d.comb += drained.eq(acc_bank[self.drain_sel])
        m.d.comb += self.value.eq(drained)
        m.d.comb += self.any_dropped.eq(dropped_bank[self.drain_sel])
        m.d.comb += self.any_overflow.eq(overflow_bank[self.drain_sel])

        # drain_latch splits the drain across a register so the normalize+round runs the cycle
        # after acc settles, keeping it off the per-cycle MAC critical path (Fmax).
        drain_latch = Signal(decomposed_layout(self.width))
        m.d.sync += drain_latch.eq(decompose(m, drained))
        normalized = normalize(m, drain_latch)
        m.d.comb += self.result.eq(round_to_float(m, drain_latch, normalized, self.lsb_exp, BFloat16))
        m.d.comb += self.result_fp32.eq(round_to_float(m, drain_latch, normalized, self.lsb_exp, Float32))

        drained_bank_written = Signal()
        m.d.comb += drained_bank_written.eq((self.load | self.enable) & (self.acc_sel == self.drain_sel))
        latched_settled = Signal()
        latched_drain_sel = Signal.like(self.drain_sel)
        m.d.sync += latched_settled.eq(~drained_bank_written)
        m.d.sync += latched_drain_sel.eq(self.drain_sel)
        m.d.comb += self.result_valid.eq(latched_settled & (latched_drain_sel == self.drain_sel))

        return m
