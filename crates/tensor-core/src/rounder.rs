//! Round-to-nearest-even on a bf16 mantissa. Port of `src/rounder.py`.

use rhdl::prelude::*;

/// bf16 stored mantissa width (implicit leading 1 not included).
pub const MANTISSA_BITS: usize = 7;

#[derive(Digital, Copy, Clone, PartialEq, Default)]
pub struct RoundIn {
    pub mantissa: b7,
    pub guard: bool,
    pub round_bit: bool,
    pub sticky: bool,
}

#[derive(Digital, Copy, Clone, PartialEq, Default)]
pub struct RoundOut {
    pub mantissa: b7,
    /// Mantissa rolled over; caller bumps the exponent.
    pub overflow: bool,
}

#[kernel]
pub fn round_nearest_even(i: RoundIn) -> RoundOut {
    let lsb = (i.mantissa & b7(1)).any();
    let round_up = i.guard && (i.round_bit || i.sticky || lsb);
    let incremented: b8 = i.mantissa.resize::<8>() + if round_up { b8(1) } else { b8(0) };
    RoundOut {
        mantissa: incremented.resize::<7>(),
        overflow: (incremented & b8(0x80)).any(),
    }
}

/// Reference spec in plain integers, independent of the kernel's bit tricks.
///
/// The input is the fixed-point value `mantissa + frac/8`, where frac =
/// guard·4 + round·2 + sticky. Sticky stands for "any nonzero bits below",
/// which for rounding is equivalent to a 1 in the lowest place.
/// Returns the 8-bit rounded result {overflow, mantissa}.
pub fn spec(m: u8, g: bool, r: bool, s: bool) -> u8 {
    let frac = (g as u8) * 4 + (r as u8) * 2 + (s as u8);
    let up = match frac {
        0..=3 => false,
        4 => m & 1 == 1, // exact tie: round to even
        _ => true,
    };
    m + up as u8
}

#[cfg(kani)]
mod proofs {
    use super::*;

    kernel_proof! {
        fn round_nearest_even_matches_spec() {
            let m: u8 = kani::any();
            kani::assume(m < 0x80);
            let (g, r, s): (bool, bool, bool) = (kani::any(), kani::any(), kani::any());

            let out = round_nearest_even(RoundIn {
                mantissa: b7(m as u128),
                guard: g,
                round_bit: r,
                sticky: s,
            });
            let got = ((out.overflow as u8) << 7) | out.mantissa.raw() as u8;

            assert_eq!(got, spec(m, g, r, s));
        }
    }

    /// Spec sanity: result is within half an ulp, and ties land on even.
    #[kani::proof]
    fn spec_is_nearest_even() {
        let m: u8 = kani::any();
        kani::assume(m < 0x80);
        let (g, r, s): (bool, bool, bool) = (kani::any(), kani::any(), kani::any());

        let x8 = (m as i32) * 8 + (g as i32) * 4 + (r as i32) * 2 + s as i32;
        let y = spec(m, g, r, s);
        let err = (y as i32) * 8 - x8;

        assert!((-4..=4).contains(&err));
        if err.abs() == 4 {
            assert!(y & 1 == 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive() {
        for m in 0u8..0x80 {
            for bits in 0u8..8 {
                let (g, r, s) = (bits & 4 != 0, bits & 2 != 0, bits & 1 != 0);
                let out = round_nearest_even(RoundIn {
                    mantissa: b7(m as u128),
                    guard: g,
                    round_bit: r,
                    sticky: s,
                });
                let got = ((out.overflow as u8) << 7) | out.mantissa.raw() as u8;
                assert_eq!(got, spec(m, g, r, s), "m={m:#x} grs={bits:03b}");
            }
        }
    }

    #[test]
    fn synthesizes() {
        compile_design::<round_nearest_even>(CompilationMode::Asynchronous).unwrap();
    }
}
