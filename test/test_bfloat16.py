from bfloat16 import BF16


def test_float_to_bf16_conversion():
    test_cases = [
        (0.0, 0x0000),
        (1.0, 0x3F80),
        (2.0, 0x4000),
        (-1.0, 0xBF80),
        (0.5, 0x3F00),
    ]

    for f, expected_bits in test_cases:
        bf16 = BF16.from_float(f)
        result = bf16.to_bits()
        assert result == expected_bits, f"Expected 0x{expected_bits:04X}, got 0x{result:04X}"


def test_bf16_to_float_conversion():
    test_cases = [
        (0x0000, 0.0),
        (0x3F80, 1.0),
        (0x4000, 2.0),
        (0xBF80, -1.0),
        (0x3F00, 0.5),
    ]

    for bits, expected_f in test_cases:
        bf16 = BF16.from_bits(bits)
        result = bf16.to_float()
        assert abs(result - expected_f) < 1e-6, f"Expected {expected_f}, got {result}"


def test_pack_unpack_bf16():
    test_cases = [
        (0, 127, 0, 0x3F80),
        (1, 127, 0, 0xBF80),
        (0, 128, 0, 0x4000),
        (0, 126, 0, 0x3F00),
        (0, 0, 0, 0x0000),
    ]

    for sign, exp, mant, expected_bits in test_cases:
        bf16 = BF16.pack(sign, exp, mant)
        packed = bf16.to_bits()
        assert packed == expected_bits, (
            f"BF16.pack({sign}, {exp}, {mant}) = 0x{packed:04X}, expected 0x{expected_bits:04X}"
        )

        bf16 = BF16.from_bits(expected_bits)
        unpacked_sign, unpacked_exp, unpacked_mant = bf16.unpack()
        assert (unpacked_sign, unpacked_exp, unpacked_mant) == (sign, exp, mant), (
            f"BF16(0x{expected_bits:04X}).unpack() = ({unpacked_sign}, {unpacked_exp}, {unpacked_mant}), expected ({sign}, {exp}, {mant})"
        )


def test_bf16_roundtrip():
    test_values = [0.0, 1.0, -1.0, 2.0, 0.5, 3.5, -7.25, 100.0, -0.125]

    for original in test_values:
        bf16 = BF16.from_float(original)
        recovered = bf16.to_float()

        rel_error = abs(recovered - original) / max(abs(original), 1e-6) if original != 0 else abs(recovered)
        assert rel_error < 0.01, f"Roundtrip failed for {original}: got {recovered} (error: {rel_error:.6f})"
