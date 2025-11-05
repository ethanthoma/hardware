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
