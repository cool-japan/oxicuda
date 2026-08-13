//! Host-side numeric support for the extended-precision and quantised kernels.
//!
//! Two pure-Rust helpers that pair with the MSL generators in
//! [`crate::msl_nn`]:
//!
//! * [`Int8Quantizer`] — per-tensor symmetric / asymmetric INT8 dynamic
//!   quantisation: derive the scale (and optional zero point) from a slice of
//!   `f32`, quantise to `i8`, and dequantise back.  This produces exactly the
//!   `scale`/`zero` constants the [`crate::msl_nn::int8_quant_gemm_msl`] kernel
//!   consumes.
//!
//! * [`DoubleSingle`] — a double-single (`df64`) value carried as two `f32`
//!   limbs, with the same Dekker/Knuth arithmetic the
//!   [`crate::msl_nn::gemm_msl_f64_ds`] kernel performs on the GPU.  Used to
//!   prepare/verify FP64-emulated GEMM operands on the host.
//!
//! * [`pack_f16`] / [`unpack_f16`] — host-side IEEE-754 binary16 conversion, so
//!   a caller of [`crate::msl::gemm_msl_f16`] or
//!   [`crate::msl::gemm_msl_v2`]`(`[`GemmDtype::F16`](crate::msl::GemmDtype)`)`
//!   does not have to hand-roll the bit packing for the `half` buffers.
//!
//! * [`pack_bf16`] / [`unpack_bf16`] — host-side bfloat16 conversion. See
//!   [`pack_bf16`] for an honest statement of the device-side status.

use crate::error::{MetalError, MetalResult};

// ─── INT8 dynamic quantisation ─────────────────────────────────────────────────

/// Result of quantising an `f32` tensor to INT8.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    /// Quantised values in `[-127, 127]` (symmetric) or `[-128, 127]`.
    pub values: Vec<i8>,
    /// Dequantisation scale: `real ≈ (q - zero_point) * scale`.
    pub scale: f32,
    /// Zero point (0 for symmetric quantisation).
    pub zero_point: i32,
}

/// Per-tensor INT8 dynamic quantiser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8Quantizer {
    /// Symmetric: zero point is 0, range maps to `[-127, 127]`.
    Symmetric,
    /// Asymmetric (affine): zero point chosen so the min maps to `-128`.
    Asymmetric,
}

impl Int8Quantizer {
    /// Quantise `data` to INT8, deriving the scale (and zero point) from its
    /// dynamic range.
    ///
    /// Returns [`MetalError::InvalidArgument`] for an empty slice or
    /// non-finite inputs.
    pub fn quantize(self, data: &[f32]) -> MetalResult<QuantizedTensor> {
        if data.is_empty() {
            return Err(MetalError::InvalidArgument(
                "cannot quantise an empty tensor".into(),
            ));
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err(MetalError::InvalidArgument(
                "tensor contains non-finite values".into(),
            ));
        }

        match self {
            Self::Symmetric => {
                let absmax = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                // A constant-zero tensor quantises to all zeros with unit scale.
                let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                let inv = 1.0 / scale;
                let values = data.iter().map(|&v| clamp_i8((v * inv).round())).collect();
                Ok(QuantizedTensor {
                    values,
                    scale,
                    zero_point: 0,
                })
            }
            Self::Asymmetric => {
                let mut min = f32::INFINITY;
                let mut max = f32::NEG_INFINITY;
                for &v in data {
                    min = min.min(v);
                    max = max.max(v);
                }
                let range = max - min;
                let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
                // Map min → -128: zero_point = -128 - round(min/scale).
                //
                // The affine zero point is an `i32` offset (stored as `i32` in
                // [`QuantizedTensor`] and consumed as `int` by the kernel), NOT
                // an int8 quantity, so it must **not** be clamped to [-128, 127]:
                // for a tensor whose values sit far from zero the correct offset
                // is many thousands in magnitude, and clamping it would collapse
                // the representable band and silently corrupt the roundtrip. Only
                // the per-element codes below are clamped into int8 range.
                let zero_point = -128 - (min / scale).round() as i32;
                let inv = 1.0 / scale;
                let values = data
                    .iter()
                    .map(|&v| clamp_i8((v * inv).round() + zero_point as f32))
                    .collect();
                Ok(QuantizedTensor {
                    values,
                    scale,
                    zero_point,
                })
            }
        }
    }
}

impl QuantizedTensor {
    /// Dequantise back to `f32`: `(q - zero_point) * scale`.
    pub fn dequantize(&self) -> Vec<f32> {
        self.values
            .iter()
            .map(|&q| (i32::from(q) - self.zero_point) as f32 * self.scale)
            .collect()
    }

    /// Maximum absolute reconstruction error against an original tensor.
    ///
    /// Returns [`MetalError::InvalidArgument`] on a length mismatch.
    pub fn max_abs_error(&self, original: &[f32]) -> MetalResult<f32> {
        if original.len() != self.values.len() {
            return Err(MetalError::InvalidArgument(
                "length mismatch in max_abs_error".into(),
            ));
        }
        let deq = self.dequantize();
        Ok(deq
            .iter()
            .zip(original)
            .fold(0.0f32, |m, (&d, &o)| m.max((d - o).abs())))
    }
}

#[inline]
fn clamp_i8(v: f32) -> i8 {
    if v >= 127.0 {
        127
    } else if v <= -128.0 {
        -128
    } else {
        v as i8
    }
}

// ─── Double-single (df64) emulated FP64 ────────────────────────────────────────

/// A double-single value: an unevaluated sum of two `f32` limbs (`hi + lo`).
///
/// Mirrors the `df64` struct used by [`crate::msl_nn::gemm_msl_f64_ds`].  Gives
/// roughly 44 bits of mantissa precision using only `f32` storage and ops,
/// which is how the Metal kernel emulates FP64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleSingle {
    /// High-order limb (the rounded value).
    pub hi: f32,
    /// Low-order limb (the rounding error).
    pub lo: f32,
}

impl DoubleSingle {
    /// The additive identity (`0`).
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    /// Construct from a single `f32` (the low limb is zero).
    pub fn from_f32(a: f32) -> Self {
        Self { hi: a, lo: 0.0 }
    }

    /// Split an `f64` into two `f32` limbs (hi = nearest f32, lo = residual).
    ///
    /// # Domain
    ///
    /// `DoubleSingle` inherits `f32`'s **exponent** range; only the mantissa is
    /// extended. Values with `|a| > f32::MAX` are therefore not representable:
    /// `hi` saturates to `±inf`.  Previously `lo` was then computed as
    /// `(a - inf) as f32 = ∓inf`, so [`to_f64`](Self::to_f64) returned `NaN` and
    /// [`pack_df64`] wrote that straight into the GPU buffer. The low limb is now
    /// forced to zero whenever `hi` is not finite, so an out-of-range value
    /// degrades to a plain `±inf` instead of a `NaN`.
    ///
    /// Use [`try_from_f64`](Self::try_from_f64) to reject out-of-range input
    /// instead of saturating.
    pub fn from_f64(a: f64) -> Self {
        let hi = a as f32;
        if !hi.is_finite() {
            // Out of f32's exponent range (or a non-finite input): a residual
            // limb is meaningless here and `a - inf` would poison it with NaN.
            return Self { hi, lo: 0.0 };
        }
        let lo = (a - hi as f64) as f32;
        Self { hi, lo }
    }

    /// Split an `f64` into two `f32` limbs, rejecting values `DoubleSingle`
    /// cannot represent.
    ///
    /// Returns [`MetalError::InvalidArgument`] for a non-finite `a` and for any
    /// `|a| > f32::MAX` (which would otherwise saturate the high limb to
    /// infinity — see [`from_f64`](Self::from_f64)).
    pub fn try_from_f64(a: f64) -> MetalResult<Self> {
        if !a.is_finite() {
            return Err(MetalError::InvalidArgument(format!(
                "DoubleSingle cannot represent the non-finite value {a}"
            )));
        }
        let hi = a as f32;
        if !hi.is_finite() {
            return Err(MetalError::InvalidArgument(format!(
                "{a} is outside f32's exponent range; DoubleSingle extends the \
                 mantissa, not the exponent"
            )));
        }
        Ok(Self {
            hi,
            lo: (a - f64::from(hi)) as f32,
        })
    }

    /// Whether both limbs are finite.
    pub fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }

    /// Reconstruct an approximate `f64` from the two limbs.
    pub fn to_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }

    /// Knuth's two-sum: rounded sum plus its exact rounding error.
    fn two_sum(a: f32, b: f32) -> Self {
        let s = a + b;
        let bb = s - a;
        let err = (a - (s - bb)) + (b - bb);
        Self { hi: s, lo: err }
    }

    /// Dekker's two-product using fused multiply-add for the error term.
    fn two_prod(a: f32, b: f32) -> Self {
        let p = a * b;
        let err = a.mul_add(b, -p);
        Self { hi: p, lo: err }
    }
}

impl std::ops::Add for DoubleSingle {
    type Output = Self;

    /// Extended-precision addition.
    fn add(self, other: Self) -> Self {
        let s = Self::two_sum(self.hi, other.hi);
        let lo = s.lo + (self.lo + other.lo);
        Self::two_sum(s.hi, lo)
    }
}

impl std::ops::Mul for DoubleSingle {
    type Output = Self;

    /// Extended-precision multiplication.
    fn mul(self, other: Self) -> Self {
        let p = Self::two_prod(self.hi, other.hi);
        let lo = p.lo + (self.hi * other.lo + self.lo * other.hi);
        Self::two_sum(p.hi, lo)
    }
}

impl std::ops::Neg for DoubleSingle {
    type Output = Self;

    /// Exact negation — negating a float never rounds, so both limbs flip sign.
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl std::ops::Sub for DoubleSingle {
    type Output = Self;

    /// Extended-precision subtraction, `self + (-other)`.
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl std::ops::Div for DoubleSingle {
    type Output = Self;

    /// Extended-precision division by Newton-style quotient refinement.
    ///
    /// Three correction terms are accumulated: `q1 = a.hi / b.hi`, then the
    /// residual `a - q1*b` supplies `q2`, and its residual supplies `q3`. The
    /// result is renormalised with `two_sum`.
    ///
    /// Division by zero follows `f32` semantics: `hi` becomes `±inf` (or `NaN`
    /// for `0/0`) and the result is no longer [`is_finite`](Self::is_finite).
    fn div(self, other: Self) -> Self {
        let q1 = self.hi / other.hi;
        let r1 = self - Self::from_f32(q1) * other;
        let q2 = r1.hi / other.hi;
        let r2 = r1 - Self::from_f32(q2) * other;
        let q3 = r2.hi / other.hi;
        let s = Self::two_sum(q1, q2);
        Self::two_sum(s.hi, s.lo + q3)
    }
}

impl std::ops::AddAssign for DoubleSingle {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl std::ops::SubAssign for DoubleSingle {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl std::ops::MulAssign for DoubleSingle {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl std::ops::DivAssign for DoubleSingle {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

/// Split a slice of `f64` into interleaved `[hi, lo]` `f32` pairs, the storage
/// layout the `gemm_f64_ds` kernel expects for its `float2` buffers.
pub fn pack_df64(data: &[f64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        let ds = DoubleSingle::from_f64(v);
        out.push(ds.hi);
        out.push(ds.lo);
    }
    out
}

/// [`pack_df64`] that rejects values `DoubleSingle` cannot represent.
///
/// [`pack_df64`] saturates out-of-range values to `±inf` (see
/// [`DoubleSingle::from_f64`]); this variant returns
/// [`MetalError::InvalidArgument`] naming the offending index instead, so a bad
/// operand is caught on the host rather than becoming an infinity in a GPU
/// buffer.
pub fn pack_df64_checked(data: &[f64]) -> MetalResult<Vec<f32>> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for (idx, &v) in data.iter().enumerate() {
        let ds = DoubleSingle::try_from_f64(v).map_err(|e| {
            MetalError::InvalidArgument(format!("df64 element {idx} is not representable: {e}"))
        })?;
        out.push(ds.hi);
        out.push(ds.lo);
    }
    Ok(out)
}

/// Reconstruct a slice of `f64` from interleaved `[hi, lo]` `f32` pairs.
///
/// Returns [`MetalError::InvalidArgument`] if the input length is odd.
pub fn unpack_df64(data: &[f32]) -> MetalResult<Vec<f64>> {
    if data.len() % 2 != 0 {
        return Err(MetalError::InvalidArgument(
            "df64 packed buffer length must be even".into(),
        ));
    }
    Ok(data
        .chunks_exact(2)
        .map(|c| DoubleSingle { hi: c[0], lo: c[1] }.to_f64())
        .collect())
}

// ─── Half precision (IEEE-754 binary16) ────────────────────────────────────────

/// `2^-24` — the value of the least-significant bit of a binary16 subnormal.
const F16_SUBNORMAL_ULP: f32 = 1.0 / 16_777_216.0;

/// Convert an `f32` to the IEEE-754 **binary16** bit pattern MSL's `half` uses.
///
/// Rounds to nearest, ties to even, exactly as the hardware conversion does.
/// Subnormal results are produced correctly, values above binary16's range
/// saturate to `±inf`, and NaNs stay NaNs (with a non-zero payload, so a
/// signalling NaN never degrades into an infinity).
///
/// The layout is byte-compatible with `half::f16::to_bits`, so a caller already
/// using the `half` crate can interoperate without a conversion.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exp == 0xff {
        return if mantissa == 0 {
            sign | 0x7c00
        } else {
            // Preserve NaN-ness; force a non-zero payload so it cannot alias inf.
            sign | 0x7e00 | ((mantissa >> 13) as u16 & 0x03ff)
        };
    }

    // Re-bias: binary32 bias 127 -> binary16 bias 15.
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return sign | 0x7c00; // overflow -> +-inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign; // magnitude below half of the smallest subnormal
        }
        // Subnormal: restore the implicit leading 1 and shift into place.
        let m = mantissa | 0x0080_0000;
        let shift = (14 - new_exp) as u32; // 14..=24
        let half = 1u32 << (shift - 1);
        let mut result = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && (result & 1) == 1) {
            result += 1; // may carry into the smallest normal — that is correct
        }
        return sign | result as u16;
    }

    let mut h = ((new_exp as u32) << 10) | (mantissa >> 13);
    let rem = mantissa & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1; // a carry out of the mantissa correctly bumps the exponent
    }
    sign | h as u16
}

/// Convert an IEEE-754 binary16 bit pattern back to `f32` (always exact).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign_bit = bits & 0x8000;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x03ff);
    let sign = u32::from(sign_bit) << 16;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign); // +-0
        }
        let magnitude = mant as f32 * F16_SUBNORMAL_ULP;
        return if sign_bit != 0 { -magnitude } else { magnitude };
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    // Normal: bias 15 -> 127.
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

/// Pack `f32` data into the binary16 buffer an MSL `half` kernel expects.
///
/// This is the missing host half of [`crate::msl::gemm_msl_f16`] and of
/// [`crate::msl::gemm_msl_v2`] with
/// [`GemmDtype::F16`](crate::msl::GemmDtype::F16): both kernels declare
/// `device const half*` operands, so every caller previously had to hand-roll
/// binary16 bit packing.
pub fn pack_f16(data: &[f32]) -> Vec<u16> {
    data.iter().copied().map(f32_to_f16_bits).collect()
}

/// Unpack a binary16 buffer produced by an MSL `half` kernel back to `f32`.
pub fn unpack_f16(data: &[u16]) -> Vec<f32> {
    data.iter().copied().map(f16_bits_to_f32).collect()
}

// ─── bfloat16 ──────────────────────────────────────────────────────────────────

/// Convert an `f32` to a **bfloat16** bit pattern (the top 16 bits, rounded to
/// nearest with ties to even).
///
/// # Device-side status (honest)
///
/// These helpers are host-side only. MSL gained a native `bfloat` scalar type in
/// **Metal 3.1 (macOS 14 / iOS 17)**; this crate ships **no** `bfloat` kernel
/// today — a grep for `bfloat` across `msl.rs`/`msl_nn.rs` finds nothing, and
/// neither [`crate::msl::GemmDtype`] nor any dispatcher offers a bf16 path. A
/// `bfloat` GEMM would additionally need a device-family/Metal-version gate,
/// because a kernel using `bfloat` fails to *compile* on an older stack rather
/// than failing at dispatch. What these functions are good for right now is
/// converting bf16 weights (the common storage format for transformer
/// checkpoints) to `f32` or `half` before uploading them to a kernel this crate
/// actually has.
pub fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // Truncation alone can clear every payload bit and turn a NaN into an
        // infinity; force a quiet-NaN payload bit.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

/// Convert a bfloat16 bit pattern back to `f32` (always exact — bf16 is a
/// truncated `f32`).
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Pack `f32` data into bfloat16 bit patterns. See [`f32_to_bf16_bits`] for the
/// device-side status.
pub fn pack_bf16(data: &[f32]) -> Vec<u16> {
    data.iter().copied().map(f32_to_bf16_bits).collect()
}

/// Unpack bfloat16 bit patterns back to `f32`.
pub fn unpack_bf16(data: &[u16]) -> Vec<f32> {
    data.iter().copied().map(bf16_bits_to_f32).collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── INT8 quantisation ──
    #[test]
    fn symmetric_quantise_roundtrip() {
        let data = [1.0f32, -2.0, 3.0, -4.0, 0.5];
        let q = Int8Quantizer::Symmetric.quantize(&data).expect("quantise");
        assert_eq!(q.zero_point, 0);
        // absmax = 4.0 → scale = 4/127; the -4.0 maps to -127.
        assert_eq!(*q.values.iter().min().unwrap(), -127);
        let err = q.max_abs_error(&data).expect("error");
        // Error is bounded by half a quantisation step (~scale/2).
        assert!(err <= q.scale, "err {err} scale {}", q.scale);
    }

    #[test]
    fn symmetric_constant_tensor() {
        let data = [0.0f32; 8];
        let q = Int8Quantizer::Symmetric.quantize(&data).expect("quantise");
        assert_eq!(q.scale, 1.0);
        assert!(q.values.iter().all(|&v| v == 0));
        assert!(q.dequantize().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn asymmetric_quantise_covers_range() {
        let data = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        let q = Int8Quantizer::Asymmetric.quantize(&data).expect("quantise");
        let err = q.max_abs_error(&data).expect("error");
        assert!(err <= q.scale, "err {err} scale {}", q.scale);
    }

    #[test]
    fn quantise_empty_and_nonfinite_error() {
        assert!(Int8Quantizer::Symmetric.quantize(&[]).is_err());
        assert!(Int8Quantizer::Symmetric.quantize(&[1.0, f32::NAN]).is_err());
        assert!(
            Int8Quantizer::Symmetric
                .quantize(&[1.0, f32::INFINITY])
                .is_err()
        );
    }

    #[test]
    fn max_abs_error_length_mismatch() {
        let q = Int8Quantizer::Symmetric.quantize(&[1.0, 2.0]).expect("q");
        assert!(q.max_abs_error(&[1.0]).is_err());
    }

    #[test]
    fn clamp_helpers() {
        assert_eq!(clamp_i8(200.0), 127);
        assert_eq!(clamp_i8(-200.0), -128);
        assert_eq!(clamp_i8(5.0), 5);
    }

    #[test]
    fn asymmetric_quantise_range_far_from_zero() {
        // A tensor whose values are all far from zero must still round-trip
        // accurately: the affine zero point is a large-magnitude i32 offset and
        // must not be clamped into int8 range (regression for the clamp bug).
        let data = [50.0f32, 100.0, 150.0];
        let q = Int8Quantizer::Asymmetric.quantize(&data).expect("quantise");
        // zero_point must be well outside int8 range here.
        assert!(
            q.zero_point < -128,
            "zero_point {} should be < -128 (unclamped)",
            q.zero_point
        );
        let err = q.max_abs_error(&data).expect("error");
        assert!(err <= q.scale, "err {err} should be <= scale {}", q.scale);
    }

    #[test]
    fn asymmetric_quantise_all_negative() {
        // All-negative tensor: true zero_point is large positive and must not be
        // clamped to 127.
        let data = [-100.0f32, -99.5, -99.0];
        let q = Int8Quantizer::Asymmetric.quantize(&data).expect("quantise");
        assert!(
            q.zero_point > 127,
            "zero_point {} should be > 127 (unclamped)",
            q.zero_point
        );
        let err = q.max_abs_error(&data).expect("error");
        assert!(err <= q.scale, "err {err} should be <= scale {}", q.scale);
    }

    // ── Double-single ──
    #[test]
    fn df64_from_to_f32() {
        let d = DoubleSingle::from_f32(3.5);
        assert_eq!(d.hi, 3.5);
        assert_eq!(d.lo, 0.0);
        assert_eq!(DoubleSingle::ZERO.to_f64(), 0.0);
    }

    #[test]
    fn df64_add_more_precise_than_f32() {
        // 1.0 + 1e-8: a single f32 add loses the small term entirely.
        let a = DoubleSingle::from_f64(1.0);
        let b = DoubleSingle::from_f64(1e-8);
        let sum = a + b;
        let err = (sum.to_f64() - (1.0 + 1e-8)).abs();
        // Double-single retains far more than naive f32 (~1e-7 ulp at 1.0).
        assert!(err < 1e-10, "df64 add error too large: {err}");
    }

    #[test]
    fn df64_mul_two_prod_is_exact_for_small_ints() {
        let a = DoubleSingle::from_f32(123.0);
        let b = DoubleSingle::from_f32(456.0);
        let p = a * b;
        // 123 * 456 = 56088, exactly representable.
        assert_eq!(p.to_f64(), 56088.0);
    }

    #[test]
    fn df64_accumulation_beats_f32() {
        // Sum many small values onto a large one; f32 would stagnate.
        let mut acc = DoubleSingle::from_f64(1_000_000.0);
        let inc = DoubleSingle::from_f64(0.1);
        for _ in 0..1000 {
            acc += inc;
        }
        let expected = 1_000_000.0 + 0.1 * 1000.0;
        let err = (acc.to_f64() - expected).abs();
        assert!(err < 1e-3, "accumulation error {err}");
    }

    #[test]
    fn pack_unpack_df64_roundtrip() {
        let data = [1.0f64, 2.5, -3.25, 1e-9];
        let packed = pack_df64(&data);
        assert_eq!(packed.len(), 8); // 2 limbs per value
        let unpacked = unpack_df64(&packed).expect("unpack");
        for (got, want) in unpacked.iter().zip(data.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got} want {want}");
        }
    }

    #[test]
    fn unpack_df64_odd_length_errors() {
        assert!(unpack_df64(&[1.0, 2.0, 3.0]).is_err());
    }

    // ── DoubleSingle: overflow domain ──
    #[test]
    fn df64_from_f64_saturates_instead_of_producing_nan() {
        // |a| > f32::MAX: hi saturates to inf. The low limb must stay 0 so
        // to_f64() yields inf, not the NaN the old `a - inf` residual produced.
        for a in [1e40f64, -1e40] {
            let d = DoubleSingle::from_f64(a);
            assert!(d.hi.is_infinite());
            assert_eq!(d.lo, 0.0);
            assert!(d.to_f64().is_infinite(), "to_f64 must not be NaN");
            assert!(!d.is_finite());
            assert!(DoubleSingle::try_from_f64(a).is_err());
        }
        assert!(DoubleSingle::try_from_f64(f64::NAN).is_err());
        assert!(DoubleSingle::try_from_f64(f64::INFINITY).is_err());
        let ok = DoubleSingle::try_from_f64(1.0 + 1e-9).expect("in range");
        assert!(ok.is_finite());
        assert!((ok.to_f64() - (1.0 + 1e-9)).abs() < 1e-15);
    }

    #[test]
    fn pack_df64_checked_reports_the_offending_index() {
        assert!(pack_df64_checked(&[1.0, 2.0]).is_ok());
        let err = pack_df64_checked(&[1.0, 1e40, 3.0]).expect_err("out of range");
        assert!(err.to_string().contains("element 1"), "{err}");
        // The unchecked variant still saturates rather than erroring.
        let packed = pack_df64(&[1e40]);
        assert!(packed[0].is_infinite());
        assert_eq!(packed[1], 0.0);
    }

    // ── DoubleSingle: Neg / Sub / Div ──
    #[test]
    fn df64_neg_is_exact_and_sub_is_add_of_the_negation() {
        let a = DoubleSingle::from_f64(1.0 + 1e-9);
        let n = -a;
        assert_eq!(n.hi, -a.hi);
        assert_eq!(n.lo, -a.lo);
        assert_eq!(a + n, DoubleSingle::ZERO);

        let b = DoubleSingle::from_f64(1.0);
        let diff = a - b;
        // A plain f32 subtraction of these two loses the difference entirely.
        assert!(
            (diff.to_f64() - 1e-9).abs() < 1e-16,
            "df64 sub error: {}",
            diff.to_f64() - 1e-9
        );
        let mut acc = a;
        acc -= b;
        assert_eq!(acc, diff);
    }

    #[test]
    fn df64_div_beats_f32_and_round_trips_through_mul() {
        let a = DoubleSingle::from_f64(1.0);
        let b = DoubleSingle::from_f64(3.0);
        let q = a / b;
        let err = (q.to_f64() - (1.0f64 / 3.0)).abs();
        assert!(err < 1e-13, "df64 div error too large: {err}");
        // f32 alone is ~6e-8 off, so this is a real improvement.
        assert!(err < ((1.0f32 / 3.0) as f64 - 1.0f64 / 3.0).abs());

        // (a / b) * b must recover a to df64 precision.
        let back = q * b;
        assert!((back.to_f64() - 1.0).abs() < 1e-13);

        let mut acc = DoubleSingle::from_f64(10.0);
        acc /= DoubleSingle::from_f64(4.0);
        assert!((acc.to_f64() - 2.5).abs() < 1e-15);

        // Division by zero follows f32 semantics rather than panicking.
        let inf = DoubleSingle::from_f64(1.0) / DoubleSingle::ZERO;
        assert!(!inf.is_finite());
    }

    // ── binary16 ──
    #[test]
    fn f16_round_trips_exactly_representable_values() {
        // Only values whose mantissa fits binary16's 10 bits round-trip exactly.
        for &v in &[
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -2.5,
            65504.0,
            -65504.0,
            2.0f32.powi(-14),
        ] {
            let back = f16_bits_to_f32(f32_to_f16_bits(v));
            assert_eq!(back, v, "binary16 round trip failed for {v}");
        }
        // The two zeros keep their sign.
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert!(f16_bits_to_f32(0x8000).is_sign_negative());
        // Known bit patterns.
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
    }

    #[test]
    fn f16_handles_subnormals_overflow_and_nan() {
        // Smallest positive subnormal: 2^-24.
        let tiny = 2.0f32.powi(-24);
        assert_eq!(f32_to_f16_bits(tiny), 0x0001);
        assert_eq!(f16_bits_to_f32(0x0001), tiny);
        // Largest subnormal: 1023 * 2^-24.
        assert_eq!(f32_to_f16_bits(1023.0 * tiny), 0x03ff);
        // Half of the smallest subnormal rounds to even => zero.
        assert_eq!(f32_to_f16_bits(tiny * 0.5), 0x0000);
        // Just above half rounds up.
        assert_eq!(f32_to_f16_bits(tiny * 0.51), 0x0001);
        // Rounding a subnormal up into the smallest normal.
        assert_eq!(f32_to_f16_bits(1023.5 * tiny), 0x0400);

        // Overflow saturates to infinity, not to a finite maximum.
        assert!(f16_bits_to_f32(f32_to_f16_bits(1e30)).is_infinite());
        assert!(f16_bits_to_f32(f32_to_f16_bits(-1e30)).is_sign_negative());
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
        // 65520 is the round-to-inf threshold; 65504 is the largest finite.
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert!(f16_bits_to_f32(f32_to_f16_bits(65536.0)).is_infinite());

        // NaN stays NaN (never collapses into infinity).
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_rounds_to_nearest_even() {
        // 1 + 2^-11 sits exactly between 1.0 (0x3c00) and 1 + 2^-10 (0x3c01);
        // ties-to-even must pick the one with an even mantissa, i.e. 0x3c00.
        assert_eq!(f32_to_f16_bits(1.0 + 2.0f32.powi(-11)), 0x3c00);
        // 1 + 3*2^-11 ties between 0x3c01 and 0x3c02 -> the even one.
        assert_eq!(f32_to_f16_bits(1.0 + 3.0 * 2.0f32.powi(-11)), 0x3c02);
    }

    #[test]
    fn pack_unpack_f16_roundtrip() {
        let data = [1.0f32, -2.5, 0.125, 100.0, -0.0];
        let packed = pack_f16(&data);
        assert_eq!(packed.len(), data.len());
        let unpacked = unpack_f16(&packed);
        for (got, want) in unpacked.iter().zip(data.iter()) {
            assert_eq!(got, want);
        }
        assert!(pack_f16(&[]).is_empty());
    }

    // ── bfloat16 ──
    #[test]
    fn bf16_round_trips_and_rounds_to_nearest_even() {
        // bf16 keeps only 8 mantissa bits, so pick values that fit in them.
        for &v in &[0.0f32, -0.0, 1.0, -2.0, 0.5, 1.5, -256.0, 2.0f32.powi(120)] {
            assert_eq!(bf16_bits_to_f32(f32_to_bf16_bits(v)), v, "bf16 rt {v}");
        }
        // bf16 keeps f32's exponent range, so no overflow to infinity.
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(1e30)).is_finite());
        assert!(f32_to_bf16_bits(f32::INFINITY) == 0x7f80);
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::NAN)).is_nan());

        // bf16 has 7 explicit mantissa bits, so consecutive values near 1.0 are
        // 2^-7 apart and the tie point is 1 + 2^-8.
        let step = 2.0f32.powi(-7);
        // Below the midpoint: rounds down.
        assert_eq!(
            bf16_bits_to_f32(f32_to_bf16_bits(1.0 + 2.0f32.powi(-9))),
            1.0
        );
        // Exactly the midpoint, and 1.0's mantissa is even: stays 1.0.
        assert_eq!(
            bf16_bits_to_f32(f32_to_bf16_bits(1.0 + 2.0f32.powi(-8))),
            1.0
        );
        // Past the midpoint: rounds up one step.
        let up = bf16_bits_to_f32(f32_to_bf16_bits(1.0 + 3.0 * 2.0f32.powi(-9)));
        assert_eq!(up, 1.0 + step, "bf16 must round past the midpoint up");
        // A tie whose lower neighbour has an ODD mantissa must round up.
        let tie_up = bf16_bits_to_f32(f32_to_bf16_bits(1.0 + step + 2.0f32.powi(-8)));
        assert_eq!(
            tie_up,
            1.0 + 2.0 * step,
            "ties must go to the even mantissa"
        );
        // Precision is 8 mantissa bits, so relative error stays under 2^-8.
        for &v in &[1.234f32, -56.78, 9.87e10] {
            let back = bf16_bits_to_f32(f32_to_bf16_bits(v));
            assert!(
                (back - v).abs() <= v.abs() * 2.0f32.powi(-8),
                "{v} -> {back}"
            );
        }
    }

    #[test]
    fn pack_unpack_bf16_roundtrip() {
        let data = [1.0f32, -2.0, 0.25, 2.0f32.powi(60)];
        let unpacked = unpack_bf16(&pack_bf16(&data));
        for (got, want) in unpacked.iter().zip(data.iter()) {
            assert_eq!(got, want);
        }
        assert!(pack_bf16(&[]).is_empty());
    }
}
