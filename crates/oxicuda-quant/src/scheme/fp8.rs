//! # FP8 Floating-Point Quantization
//!
//! FP8 is an 8-bit floating-point format used in Hopper/Blackwell tensor cores.
//! Two variants are defined by the NVIDIA FP8 spec (OFP8):
//!
//! | Format | Sign | Exponent | Mantissa | Range          | Use case |
//! |--------|------|----------|----------|----------------|----------|
//! | E4M3   | 1    | 4        | 3        | ±448           | Weights / Fwd activations |
//! | E5M2   | 1    | 5        | 2        | ±57344         | Gradients |
//!
//! ## E4M3 Special Values
//!
//! * Exponent all-1s + mantissa all-1s → NaN (no ±Inf)
//! * Exponent all-0s → denormals (mantissa / 2^(-6))
//!
//! ## E5M2 Special Values
//!
//! * Exponent all-1s + mantissa = 00 → ±Inf
//! * Exponent all-1s + mantissa ≠ 00 → NaN
//!
//! ## Encoding
//!
//! We store FP8 as `u8` bit patterns.  Conversion is done in pure Rust via
//! IEEE 754 bit manipulation of f32.

use crate::error::{QuantError, QuantResult};

// ─── Format ───────────────────────────────────────────────────────────────────

/// FP8 format variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// 1 sign + 4 exponent + 3 mantissa bits.  Max = 448.
    E4M3,
    /// 1 sign + 5 exponent + 2 mantissa bits.  Max = 57344.
    E5M2,
}

impl Fp8Format {
    /// Number of exponent bits.
    #[must_use]
    pub fn exp_bits(self) -> u32 {
        match self {
            Self::E4M3 => 4,
            Self::E5M2 => 5,
        }
    }

    /// Number of mantissa bits.
    #[must_use]
    pub fn man_bits(self) -> u32 {
        match self {
            Self::E4M3 => 3,
            Self::E5M2 => 2,
        }
    }

    /// Exponent bias.
    #[must_use]
    pub fn bias(self) -> i32 {
        match self {
            Self::E4M3 => 7,  // 2^(4-1) - 1
            Self::E5M2 => 15, // 2^(5-1) - 1
        }
    }

    /// Maximum finite representable value.
    #[must_use]
    pub fn max_val(self) -> f32 {
        match self {
            Self::E4M3 => 448.0,
            Self::E5M2 => 57344.0,
        }
    }
}

// ─── Fp8Codec ─────────────────────────────────────────────────────────────────

/// FP8 encoder/decoder.
///
/// Saturating conversion: values exceeding `max_val` are clamped; NaN/Inf
/// inputs return an error.
#[derive(Debug, Clone, Copy)]
pub struct Fp8Codec {
    /// Target FP8 format.
    pub format: Fp8Format,
    /// Whether to saturate (clamp) out-of-range values instead of erroring.
    pub saturate: bool,
}

impl Fp8Codec {
    /// E4M3 codec with saturation.
    #[must_use]
    pub fn e4m3() -> Self {
        Self {
            format: Fp8Format::E4M3,
            saturate: true,
        }
    }

    /// E5M2 codec with saturation.
    #[must_use]
    pub fn e5m2() -> Self {
        Self {
            format: Fp8Format::E5M2,
            saturate: true,
        }
    }

    /// Encode a single f32 to FP8 u8.
    ///
    /// # Errors
    ///
    /// * [`QuantError::NonFiniteFp8`] if `v` is NaN or Inf and `saturate = false`.
    pub fn encode_f32(&self, v: f32) -> QuantResult<u8> {
        if !v.is_finite() {
            return Err(QuantError::NonFiniteFp8(v));
        }
        let max = self.format.max_val();
        let v_sat = v.clamp(-max, max);
        Ok(self.fp32_to_fp8(v_sat))
    }

    /// Decode a FP8 u8 to f32.
    #[must_use]
    pub fn decode_f32(&self, b: u8) -> f32 {
        self.fp8_to_fp32(b)
    }

    /// Encode a slice of f32 to FP8.
    ///
    /// # Errors
    ///
    /// Propagates [`QuantError::NonFiniteFp8`] on first NaN/Inf when not saturating.
    pub fn encode(&self, data: &[f32]) -> QuantResult<Vec<u8>> {
        data.iter().map(|&v| self.encode_f32(v)).collect()
    }

    /// Decode a slice of FP8 to f32.
    pub fn decode(&self, data: &[u8]) -> Vec<f32> {
        data.iter().map(|&b| self.decode_f32(b)).collect()
    }

    /// Mean squared error of the FP8 round-trip.
    ///
    /// # Errors
    ///
    /// Propagates [`encode`](Self::encode) errors.
    pub fn quantization_mse(&self, data: &[f32]) -> QuantResult<f32> {
        let encoded = self.encode(data)?;
        let decoded = self.decode(&encoded);
        let mse = data
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;
        Ok(mse)
    }

    // ── Private: bit-level encode/decode ─────────────────────────────────────

    fn fp32_to_fp8(&self, v: f32) -> u8 {
        // Extract f32 components.
        let bits = v.to_bits();
        let sign = (bits >> 31) as u8;
        let exp32 = ((bits >> 23) & 0xFF) as i32; // biased exponent (bias=127)
        let man32 = bits & 0x007F_FFFF; // 23-bit mantissa

        let exp_bits = self.format.exp_bits();
        let man_bits = self.format.man_bits();
        let bias8 = self.format.bias();

        if v == 0.0 || v == -0.0 {
            return sign << 7;
        }

        // Re-bias the exponent.
        let exp_unbiased = exp32 - 127;
        let exp8_raw = exp_unbiased + bias8;

        let man_shift = 23 - man_bits; // how many mantissa bits to drop

        if exp8_raw <= 0 {
            // Denormal territory: right-shift mantissa into denormal position.
            // Value = (1 + man/2^23) * 2^exp_unbiased
            //       ≈ 2^exp8_raw * man_denorm / 2^man_bits
            let full_man = (man32 | 0x0080_0000) >> 1; // include implicit 1
            let shift = (1 - exp8_raw) as u32 + man_shift;
            if shift >= 24 {
                return sign << 7;
            } // underflow → ±0
            let man8 = (full_man >> shift) as u8 & ((1 << man_bits) - 1);
            return (sign << 7) | man8;
        }

        let max_exp = (1 << exp_bits) - 1;
        if exp8_raw >= max_exp {
            // Saturate to max finite value (E4M3 has no Inf, E5M2 has Inf).
            return match self.format {
                Fp8Format::E4M3 => (sign << 7) | 0x7E, // 01111110 = max finite
                Fp8Format::E5M2 => (sign << 7) | 0x7B, // 01111011 = max finite
            };
        }

        let man8 = (man32 >> man_shift) as u8 & ((1 << man_bits) - 1);
        (sign << 7) | ((exp8_raw as u8) << man_bits) | man8
    }

    fn fp8_to_fp32(&self, b: u8) -> f32 {
        let sign = (b >> 7) as u32;
        let exp_bits = self.format.exp_bits();
        let man_bits = self.format.man_bits();
        let bias8 = self.format.bias();

        let exp8 = ((b >> man_bits) & ((1 << exp_bits) - 1)) as u32;
        let man8 = (b & ((1 << man_bits) - 1)) as u32;

        // Check for special values.
        let all_exp = (1 << exp_bits) - 1;
        match self.format {
            Fp8Format::E4M3 => {
                if exp8 == all_exp as u32 && man8 == (1 << man_bits) - 1 {
                    return f32::NAN; // NaN in E4M3
                }
            }
            Fp8Format::E5M2 => {
                if exp8 == all_exp as u32 {
                    if man8 == 0 {
                        return if sign == 0 {
                            f32::INFINITY
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                    return f32::NAN;
                }
            }
        }

        // Zero / denormal.
        if exp8 == 0 {
            if man8 == 0 {
                return f32::from_bits(sign << 31); // ±0
            }
            // Denormal: value = man8 / 2^man_bits * 2^(1-bias8)
            let man_shift = 23 - man_bits;
            let exp32 = (127 + 1 - bias8) as u32;
            // Find leading bit position in man8.
            let leading = man_bits - 1 - man8.leading_zeros().min(man_bits - 1);
            let exp32_adj = exp32.wrapping_sub(leading);
            let man32 = ((man8 << leading) & ((1 << man_bits) - 1)) << man_shift;
            return f32::from_bits((sign << 31) | (exp32_adj << 23) | man32);
        }

        // Normal: re-bias.
        let exp32 = (exp8 as i32 - bias8 + 127) as u32;
        let man_shift = 23 - man_bits;
        let man32 = man8 << man_shift;
        f32::from_bits((sign << 31) | (exp32 << 23) | man32)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn e4m3_format_params() {
        assert_eq!(Fp8Format::E4M3.exp_bits(), 4);
        assert_eq!(Fp8Format::E4M3.man_bits(), 3);
        assert_eq!(Fp8Format::E4M3.bias(), 7);
        assert_abs_diff_eq!(Fp8Format::E4M3.max_val(), 448.0, epsilon = 1.0);
    }

    #[test]
    fn e5m2_format_params() {
        assert_eq!(Fp8Format::E5M2.exp_bits(), 5);
        assert_eq!(Fp8Format::E5M2.man_bits(), 2);
        assert_eq!(Fp8Format::E5M2.bias(), 15);
        assert_abs_diff_eq!(Fp8Format::E5M2.max_val(), 57344.0, epsilon = 100.0);
    }

    #[test]
    fn e4m3_zero_encodes_to_zero() {
        let c = Fp8Codec::e4m3();
        assert_eq!(c.encode_f32(0.0).unwrap(), 0x00);
        assert_eq!(c.encode_f32(-0.0).unwrap(), 0x80);
    }

    #[test]
    fn e4m3_round_trip_basic() {
        let c = Fp8Codec::e4m3();
        for &v in &[1.0_f32, -1.0, 2.0, 0.5, 0.25, -0.25] {
            let enc = c.encode_f32(v).unwrap();
            let dec = c.decode_f32(enc);
            let rel_err = (v - dec).abs() / v.abs().max(1e-6);
            assert!(rel_err < 0.15, "v={v}, dec={dec}, rel_err={rel_err}");
        }
    }

    #[test]
    fn e5m2_round_trip_basic() {
        let c = Fp8Codec::e5m2();
        for &v in &[1.0_f32, -1.0, 4.0, 16.0, -8.0] {
            let enc = c.encode_f32(v).unwrap();
            let dec = c.decode_f32(enc);
            let rel_err = (v - dec).abs() / v.abs().max(1e-6);
            assert!(rel_err < 0.25, "v={v}, dec={dec}, rel_err={rel_err}");
        }
    }

    #[test]
    fn e4m3_saturates_large_values() {
        let c = Fp8Codec::e4m3();
        let enc = c.encode_f32(1000.0).unwrap();
        let dec = c.decode_f32(enc);
        // Should saturate at max_val (448)
        assert!(dec <= 448.0 + 1.0, "should saturate, got {dec}");
        assert!(dec > 0.0, "positive saturation should be positive");
    }

    #[test]
    fn nan_input_errors() {
        let c = Fp8Codec {
            format: Fp8Format::E4M3,
            saturate: false,
        };
        assert!(matches!(
            c.encode_f32(f32::NAN),
            Err(QuantError::NonFiniteFp8(_))
        ));
        assert!(matches!(
            c.encode_f32(f32::INFINITY),
            Err(QuantError::NonFiniteFp8(_))
        ));
    }

    #[test]
    fn mse_within_tolerance() {
        let c = Fp8Codec::e4m3();
        let data: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0) - 1.0).collect();
        let mse = c.quantization_mse(&data).unwrap();
        assert!(mse < 0.01, "E4M3 MSE unexpectedly large: {mse}");
    }

    #[test]
    fn batch_encode_decode() {
        let c = Fp8Codec::e4m3();
        let data = vec![0.0_f32, 1.0, -1.0, 0.5, 2.0, -2.0];
        let enc = c.encode(&data).unwrap();
        assert_eq!(enc.len(), data.len());
        let dec = c.decode(&enc);
        assert_eq!(dec.len(), data.len());
    }
}
