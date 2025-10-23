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

    @staticmethod
    def from_float(f: float) -> int:
        fp32_bits = struct.unpack(">I", struct.pack(">f", f))[0]
        return fp32_bits >> 16

    @staticmethod
    def to_float(bits: int) -> float:
        fp32_bits = bits << 16
        return struct.unpack(">f", struct.pack(">I", fp32_bits))[0]

    @staticmethod
    def pack(sign: int, exponent: int, mantissa: int) -> int:
        return (sign << 15) | (exponent << 7) | mantissa

    @staticmethod
    def unpack(bits: int) -> tuple[int, int, int]:
        sign = (bits >> 15) & 0x1
        exponent = (bits >> 7) & 0xFF
        mantissa = bits & 0x7F
        return sign, exponent, mantissa
