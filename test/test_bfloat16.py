from bfloat16 import BFloat16


def test_float_to_bf16_conversion():
    test_cases = [
        (0.0, 0x0000),
        (1.0, 0x3F80),
        (2.0, 0x4000),
        (-1.0, 0xBF80),
        (0.5, 0x3F00),
    ]

    for f, expected_bits in test_cases:
        result = BFloat16.from_float(f)
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
        result = BFloat16.to_float(bits)
        assert abs(result - expected_f) < 1e-6, f"Expected {expected_f}, got {result}"


def test_pack_unpack_bf16():
    test_cases = [
        (0, 127, 0, 0x3F80),  # 1.0
        (1, 127, 0, 0xBF80),  # -1.0
        (0, 128, 0, 0x4000),  # 2.0
        (0, 126, 0, 0x3F00),  # 0.5
        (0, 0, 0, 0x0000),  # 0.0
    ]

    for sign, exp, mant, expected_bits in test_cases:
        packed = BFloat16.pack(sign, exp, mant)
        assert packed == expected_bits, (
            f"BFloat16.pack({sign}, {exp}, {mant}) = 0x{packed:04X}, expected 0x{expected_bits:04X}"
        )

        unpacked_sign, unpacked_exp, unpacked_mant = BFloat16.unpack(expected_bits)
        assert (unpacked_sign, unpacked_exp, unpacked_mant) == (sign, exp, mant), (
            f"BFloat16.unpack(0x{expected_bits:04X}) = ({unpacked_sign}, {unpacked_exp}, {unpacked_mant}), expected ({sign}, {exp}, {mant})"
        )


def test_bf16_roundtrip():
    test_values = [0.0, 1.0, -1.0, 2.0, 0.5, 3.5, -7.25, 100.0, -0.125]

    for original in test_values:
        bf16_bits = BFloat16.from_float(original)
        recovered = BFloat16.to_float(bf16_bits)

        rel_error = abs(recovered - original) / max(abs(original), 1e-6) if original != 0 else abs(recovered)
        assert rel_error < 0.01, f"Roundtrip failed for {original}: got {recovered} (error: {rel_error:.6f})"
