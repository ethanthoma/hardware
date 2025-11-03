from amaranth import *
from amaranth.build import Platform
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class LeadingZeroAnticipator(wiring.Component):
    """Leading Zero Anticipator for fast normalization

    Predicts the number of leading zeros in the sum (a + b + carry_in)

    - Delay: log2(width) gate delays (much faster than full addition)
    - For BF16 FMA: supports various widths (8, 16, 26 bits)
    """

    def __init__(self, width: int = 8):
        self.width = width
        self.count_bits = (width).bit_length()

        super().__init__(
            {
                "a": In(width),
                "b": In(width),
                "carry_in": In(1),
                "lz_count": Out(self.count_bits),
            }
        )

    def elaborate(self, platform: Platform | None) -> Module:
        m = Module()

        # ---- Generate and Propagate Signals ----
        generate = Signal(self.width)
        propagate = Signal(self.width)

        m.d.comb += generate.eq(self.a & self.b)
        m.d.comb += propagate.eq(self.a ^ self.b)

        # ---- Compute Carry Chain ----
        # carry[i] is the carry into bit position i
        carries = Array([Signal(name=f"carry_{i}") for i in range(self.width + 1)])
        m.d.comb += carries[0].eq(self.carry_in)

        for i in range(self.width):
            # Carry out from position i: G[i] | (P[i] & carry_in[i])
            m.d.comb += carries[i + 1].eq(generate[i] | (propagate[i] & carries[i]))

        # ---- Predict Sum Bits ----
        predicted_sum = Signal(self.width)

        for i in range(self.width):
            # Sum bit i = P[i] ^ carry[i]
            m.d.comb += predicted_sum[i].eq(propagate[i] ^ carries[i])

        # ---- Count Leading Zeros in Predicted Sum ----
        # Use Mux tree to build priority encoder
        # Start from LSB, work up to MSB with nested Mux

        lz_count_result = Signal(self.count_bits)

        # Build from LSB to MSB
        # If bit i is 1, count is (width - 1 - i), else check higher bit
        lz_count_result = self.width  # Default: all zeros

        for i in range(self.width):
            # For bit i (LSB first), if it's 1, the count is (width - 1 - i)
            lz_count_result = Mux(
                predicted_sum[i],
                self.width - 1 - i,  # This bit is 1
                lz_count_result,  # This bit is 0, use previous result
            )

        m.d.comb += self.lz_count.eq(lz_count_result)

        return m
