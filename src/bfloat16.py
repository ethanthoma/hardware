import struct

from amaranth import *
from amaranth.lib import data


class BFloat16(data.Struct):
    mantissa: 7
    exponent: 8
    sign: 1

    def is_zero(self):
        return (self.exponent == 0) & (self.mantissa == 0)

    def is_subnormal(self):
        return self.exponent == 0


class E8M10(data.Struct):
    """19-bit accumulator format: 1 sign + 8 exponent + 10 mantissa

    Extended mantissa (10 bits vs BF16's 7) provides 3 extra bits for
    accumulation error. Sufficient for 8 MAC operations (2^3 = 8).
    Same exponent range as BF16/FP32.
    """

    mantissa: 10
    exponent: 8
    sign: 1

    def is_zero(self):
        return (self.exponent == 0) & (self.mantissa == 0)

    def is_subnormal(self):
        return self.exponent == 0


class BF16:
    def __init__(self, bits: int):
        self.bits = bits

    @classmethod
    def from_float(cls, f: float):
        fp32_bits = struct.unpack(">I", struct.pack(">f", f))[0]
        bits = fp32_bits >> 16
        return cls(bits)

    @classmethod
    def from_bits(cls, bits: int):
        return cls(bits)

    def to_bits(self) -> int:
        return self.bits

    def to_float(self) -> float:
        fp32_bits = self.bits << 16
        return struct.unpack(">f", struct.pack(">I", fp32_bits))[0]

    def unpack(self) -> tuple[int, int, int]:
        sign = (self.bits >> 15) & 0x1
        exp = (self.bits >> 7) & 0xFF
        mant = self.bits & 0x7F
        return sign, exp, mant

    @classmethod
    def pack(cls, sign: int, exp: int, mant: int):
        bits = (sign << 15) | (exp << 7) | mant
        return cls(bits)


class E8M10_SW:
    """Software helper for E8M10 format (19-bit: 1 sign + 8 exponent + 10 mantissa)"""

    def __init__(self, bits: int):
        self.bits = bits

    @classmethod
    def from_float(cls, f: float):
        fp32_bits = struct.unpack(">I", struct.pack(">f", f))[0]
        sign = (fp32_bits >> 31) & 0x1
        exp = (fp32_bits >> 23) & 0xFF
        mant = (fp32_bits >> 13) & 0x3FF
        bits = (sign << 18) | (exp << 10) | mant
        return cls(bits)

    @classmethod
    def from_bf16(cls, bf16):
        sign, exp, mant = bf16.unpack()
        mant_extended = mant << 3
        bits = (sign << 18) | (exp << 10) | mant_extended
        return cls(bits)

    def to_bf16(self):
        sign = (self.bits >> 18) & 0x1
        exp = (self.bits >> 10) & 0xFF
        mant = (self.bits >> 3) & 0x7F
        return BF16.pack(sign, exp, mant)

    def to_float(self) -> float:
        sign = (self.bits >> 18) & 0x1
        exp = (self.bits >> 10) & 0xFF
        mant = (self.bits) & 0x3FF
        fp32_bits = (sign << 31) | (exp << 23) | (mant << 13)
        return struct.unpack(">f", struct.pack(">I", fp32_bits))[0]

    def unpack(self) -> tuple[int, int, int]:
        sign = (self.bits >> 18) & 0x1
        exp = (self.bits >> 10) & 0xFF
        mant = self.bits & 0x3FF
        return sign, exp, mant

    @classmethod
    def pack(cls, sign: int, exp: int, mant: int):
        bits = (sign << 18) | (exp << 10) | mant
        return cls(bits)
