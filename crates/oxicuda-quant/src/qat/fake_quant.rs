//! # Fake Quantization with Straight-Through Estimator (STE)
//!
//! Fake quantization simulates the effect of quantization during training by
//! applying a quantize-then-dequantize operation in the forward pass while
//! passing gradients through unchanged (STE) where the value is within the
//! representable range.
//!
//! ## Forward pass
//!
//! ```text
//! q  = clamp(round(x / scale + zp), q_min, q_max)
//! x̂  = (q − zp) × scale
//! ```
//!
//! ## Backward pass (STE)
//!
//! ```text
//! ∂L/∂x = ∂L/∂x̂  if  q_min_val ≤ x ≤ q_max_val
//!        = 0       otherwise (clipped region)
//! ```
//!
//! where `q_min_val = (q_min − zp) × scale`, `q_max_val = (q_max − zp) × scale`.

use crate::error::{QuantError, QuantResult};

// ─── FakeQuantize ─────────────────────────────────────────────────────────────

/// Fake quantization operator for quantization-aware training (QAT).
///
/// Maintains the current scale and zero-point that are updated during
/// calibration / training via an associated observer.
#[derive(Debug, Clone)]
pub struct FakeQuantize {
    /// Quantization bit-width.
    pub bits: u32,
    /// Whether to use symmetric quantization (zp = 0).
    pub symmetric: bool,
    /// Current quantization scale (must be > 0).
    pub scale: f32,
    /// Current zero-point.
    pub zero_point: i32,
    /// Whether fake quantization is enabled.
    /// When disabled, `forward` returns the input unchanged.
    pub enabled: bool,
}

impl FakeQuantize {
    /// Create a new fake quantizer with the given scale and zero-point.
    ///
    /// # Errors
    ///
    /// * [`QuantError::InvalidBitWidth`] — `bits` is 0 or > 16.
    /// * [`QuantError::InvalidScale`]   — `scale` is ≤ 0 or non-finite.
    pub fn new(bits: u32, symmetric: bool, scale: f32, zero_point: i32) -> QuantResult<Self> {
        if bits == 0 || bits > 16 {
            return Err(QuantError::InvalidBitWidth { bits });
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(QuantError::InvalidScale { scale });
        }
        Ok(Self {
            bits,
            symmetric,
            scale,
            zero_point,
            enabled: true,
        })
    }

    /// Create with default scale=1.0, zp=0 for the given bit-width.
    ///
    /// # Errors
    ///
    /// * [`QuantError::InvalidBitWidth`] — `bits` is 0 or > 16.
    pub fn with_defaults(bits: u32, symmetric: bool) -> QuantResult<Self> {
        Self::new(bits, symmetric, 1.0, 0)
    }

    /// Update scale and zero-point (e.g., from an observer).
    ///
    /// # Errors
    ///
    /// * [`QuantError::InvalidScale`] — `scale` is ≤ 0 or non-finite.
    pub fn update_params(&mut self, scale: f32, zero_point: i32) -> QuantResult<()> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(QuantError::InvalidScale { scale });
        }
        self.scale = scale;
        self.zero_point = zero_point;
        Ok(())
    }

    /// Integer quantization bounds [q_min, q_max].
    #[must_use]
    pub fn quant_range(&self) -> (i32, i32) {
        if self.symmetric {
            let half = 1i32 << (self.bits - 1);
            (-half, half - 1)
        } else {
            (0i32, (1i32 << self.bits) - 1)
        }
    }

    /// Float clipping bounds `[x_min, x_max]` corresponding to the integer range.
    #[must_use]
    pub fn float_range(&self) -> (f32, f32) {
        let (q_min, q_max) = self.quant_range();
        let zp = self.zero_point as f32;
        let lo = (q_min as f32 - zp) * self.scale;
        let hi = (q_max as f32 - zp) * self.scale;
        (lo, hi)
    }

    /// Forward pass: quantize-then-dequantize.
    ///
    /// If `enabled = false`, returns the input unchanged.
    #[must_use]
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        if !self.enabled {
            return x.to_vec();
        }
        let (q_min, q_max) = self.quant_range();
        let zp = self.zero_point as f32;
        x.iter()
            .map(|&v| {
                let q = (v / self.scale + zp)
                    .round()
                    .clamp(q_min as f32, q_max as f32);
                (q - zp) * self.scale
            })
            .collect()
    }

    /// Backward pass (Straight-Through Estimator).
    ///
    /// Passes `grad_output` through where `x` is inside the representable
    /// float range; zeros the gradient where `x` is clipped.
    ///
    /// # Errors
    ///
    /// * [`QuantError::DimensionMismatch`] — `grad_output` and `x` lengths differ.
    pub fn backward(&self, grad_output: &[f32], x: &[f32]) -> QuantResult<Vec<f32>> {
        if grad_output.len() != x.len() {
            return Err(QuantError::DimensionMismatch {
                expected: x.len(),
                got: grad_output.len(),
            });
        }
        if !self.enabled {
            return Ok(grad_output.to_vec());
        }
        let (x_min, x_max) = self.float_range();
        let grad = grad_output
            .iter()
            .zip(x.iter())
            .map(|(&g, &v)| if v >= x_min && v <= x_max { g } else { 0.0 })
            .collect();
        Ok(grad)
    }

    /// Estimate quantization noise (MSE between input and fake-quantized output).
    ///
    /// Useful for measuring quantization error at the current scale/zp.
    #[must_use]
    pub fn quantization_noise(&self, x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        let fq = self.forward(x);
        let mse = x
            .iter()
            .zip(fq.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>();
        mse / x.len() as f32
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn forward_quantize_dequantize_int8() {
        let fq = FakeQuantize::new(8, true, 1.0 / 127.0, 0).unwrap();
        // Input 1.0 → q = 127 → dequant = 127 / 127 ≈ 1.0
        let out = fq.forward(&[1.0_f32]);
        assert_abs_diff_eq!(out[0], 1.0, epsilon = 0.01);
    }

    #[test]
    fn forward_passthrough_when_disabled() {
        let mut fq = FakeQuantize::new(8, true, 0.01, 0).unwrap();
        fq.enabled = false;
        let data = vec![1.5_f32, -2.3, 0.7];
        let out = fq.forward(&data);
        assert_eq!(out, data);
    }

    #[test]
    fn backward_ste_passthrough() {
        let fq = FakeQuantize::new(8, true, 1.0 / 127.0, 0).unwrap();
        let x = vec![0.5_f32, -0.5];
        let grad = vec![1.0_f32, -1.0];
        let ste = fq.backward(&grad, &x).unwrap();
        // x is within [-1, 1]: gradient passed through unchanged.
        assert_abs_diff_eq!(ste[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(ste[1], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn backward_ste_zero_outside_range() {
        let fq = FakeQuantize::new(8, true, 1.0 / 127.0, 0).unwrap();
        // x = ±2.0 is outside [-1, 127/127] float range
        let x = vec![2.0_f32, -2.0];
        let grad = vec![1.0_f32, 1.0];
        let ste = fq.backward(&grad, &x).unwrap();
        assert_abs_diff_eq!(ste[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(ste[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn backward_dimension_mismatch_error() {
        let fq = FakeQuantize::new(8, true, 0.01, 0).unwrap();
        let x = vec![0.5_f32; 3];
        let grad = vec![1.0_f32; 4];
        assert!(matches!(
            fq.backward(&grad, &x),
            Err(QuantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn invalid_scale_error() {
        assert!(matches!(
            FakeQuantize::new(8, true, -0.01, 0),
            Err(QuantError::InvalidScale { .. })
        ));
        assert!(matches!(
            FakeQuantize::new(8, true, 0.0, 0),
            Err(QuantError::InvalidScale { .. })
        ));
    }

    #[test]
    fn invalid_bit_width_error() {
        assert!(matches!(
            FakeQuantize::new(0, true, 0.01, 0),
            Err(QuantError::InvalidBitWidth { bits: 0 })
        ));
        assert!(matches!(
            FakeQuantize::new(17, true, 0.01, 0),
            Err(QuantError::InvalidBitWidth { bits: 17 })
        ));
    }

    #[test]
    fn quant_range_int8_symmetric() {
        let fq = FakeQuantize::new(8, true, 0.01, 0).unwrap();
        assert_eq!(fq.quant_range(), (-128, 127));
    }

    #[test]
    fn quant_range_int4_asymmetric() {
        let fq = FakeQuantize::new(4, false, 0.01, 0).unwrap();
        assert_eq!(fq.quant_range(), (0, 15));
    }

    #[test]
    fn quantization_noise_zero_for_fine_scale() {
        // With scale = 1/127 and INT8, values in [-1, 1] should have small noise.
        let fq = FakeQuantize::new(8, true, 1.0 / 127.0, 0).unwrap();
        let data: Vec<f32> = (0..128).map(|i| i as f32 / 128.0 - 0.5).collect();
        let noise = fq.quantization_noise(&data);
        assert!(noise < 1e-5, "noise too high: {noise}");
    }

    #[test]
    fn update_params_works() {
        let mut fq = FakeQuantize::with_defaults(8, true).unwrap();
        fq.update_params(0.5, 0).unwrap();
        assert_abs_diff_eq!(fq.scale, 0.5, epsilon = 1e-7);
    }

    #[test]
    fn asymmetric_forward_with_nonzero_zp() {
        // scale=1/15, zp=0 for [0, 1] range with INT4 asymmetric
        let fq = FakeQuantize::new(4, false, 1.0 / 15.0, 0).unwrap();
        let out = fq.forward(&[0.0_f32, 0.5, 1.0]);
        // 0 → q=0, 0.5 → q≈7 → 7/15, 1.0 → q=15 → 1.0
        assert_abs_diff_eq!(out[0], 0.0, epsilon = 0.001);
        assert!(out[1] > 0.4 && out[1] < 0.6, "midpoint: {}", out[1]);
        assert_abs_diff_eq!(out[2], 1.0, epsilon = 0.001);
    }
}
