//! Gradient clipping utilities.
//!
//! Gradient clipping prevents gradient explosions, which are common when
//! training deep networks or RNNs.  Three clipping strategies are provided:
//!
//! | Strategy | Description |
//! |---|---|
//! | [`crate::grad_clip::GlobalNormClip`] | Rescale *all* gradients so their joint L2 norm ≤ max_norm |
//! | [`crate::grad_clip::PerLayerNormClip`] | Clip each parameter's gradient independently |
//! | [`crate::grad_clip::ValueClip`] | Hard clip each gradient element to `[−clip_val, +clip_val]` |
//!
//! ## Usage
//!
//! ```rust
//! use oxicuda_train::grad_clip::{GlobalNormClip, GradientClipper};
//! use oxicuda_train::gpu_optimizer::ParamTensor;
//!
//! let mut p = ParamTensor::new(vec![1.0, 2.0], "w");
//! p.set_grad(vec![100.0, 100.0]).unwrap();
//!
//! let clipper = GlobalNormClip::new(1.0);
//! let norm = clipper.clip(&mut [p]).unwrap();
//! assert!(norm > 1.0, "norm before clip was > max_norm");
//! ```

use crate::error::{TrainError, TrainResult};
use crate::gpu_optimizer::ParamTensor;

// ─── GradientClipper trait ────────────────────────────────────────────────────

/// Common interface for gradient clipping strategies.
pub trait GradientClipper {
    /// Clip gradients in-place.
    ///
    /// Returns the **pre-clipping** global gradient L2 norm.
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] if `params` is empty.
    /// * [`TrainError::InvalidGradNorm`] if the computed norm is NaN or Inf.
    fn clip(&self, params: &mut [ParamTensor]) -> TrainResult<f32>;
}

// ─── GlobalNormClip ───────────────────────────────────────────────────────────

/// Clip all gradients jointly so that their global L2 norm does not exceed
/// `max_norm`.
///
/// If `global_norm ≤ max_norm`, no change is made.  Otherwise every gradient
/// element is multiplied by `max_norm / global_norm`.
///
/// This preserves the direction of the gradient while bounding its magnitude.
#[derive(Debug, Clone)]
pub struct GlobalNormClip {
    /// Maximum allowed gradient L2 norm.
    pub max_norm: f32,
    /// Norm type (default 2.0 = L2 norm).
    pub norm_type: f32,
}

impl GlobalNormClip {
    /// Create a new global norm clipper.
    #[must_use]
    pub fn new(max_norm: f32) -> Self {
        Self {
            max_norm,
            norm_type: 2.0,
        }
    }

    /// Compute and return the global gradient norm without clipping.
    pub fn compute_norm(params: &[ParamTensor]) -> TrainResult<f32> {
        let mut total = 0.0_f64;
        for (i, p) in params.iter().enumerate() {
            if !p.requires_grad {
                continue;
            }
            let grad = p.grad.as_ref().ok_or(TrainError::NoGradient { index: i })?;
            for &g in grad.iter() {
                total += (g as f64) * (g as f64);
            }
        }
        let norm = (total.sqrt()) as f32;
        if norm.is_nan() || norm.is_infinite() {
            return Err(TrainError::InvalidGradNorm);
        }
        Ok(norm)
    }
}

impl GradientClipper for GlobalNormClip {
    fn clip(&self, params: &mut [ParamTensor]) -> TrainResult<f32> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }
        let norm = Self::compute_norm(params)?;
        if norm > self.max_norm {
            let scale = self.max_norm / (norm + 1e-6);
            for p in params.iter_mut() {
                if let Some(grad) = &mut p.grad {
                    for g in grad.iter_mut() {
                        *g *= scale;
                    }
                }
            }
        }
        Ok(norm)
    }
}

// ─── PerLayerNormClip ─────────────────────────────────────────────────────────

/// Clip each parameter's gradient independently to `max_norm`.
///
/// Unlike [`GlobalNormClip`], this clips each tensor's norm separately,
/// which can prevent a single large-gradient layer from dominating the
/// global clip scale.
#[derive(Debug, Clone)]
pub struct PerLayerNormClip {
    /// Per-parameter maximum norm.
    pub max_norm: f32,
}

impl PerLayerNormClip {
    /// Create a per-layer norm clipper.
    #[must_use]
    pub fn new(max_norm: f32) -> Self {
        Self { max_norm }
    }
}

impl GradientClipper for PerLayerNormClip {
    fn clip(&self, params: &mut [ParamTensor]) -> TrainResult<f32> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }

        let mut global_norm_sq = 0.0_f64;

        for (i, p) in params.iter_mut().enumerate() {
            if !p.requires_grad {
                continue;
            }
            let grad = p.grad.as_mut().ok_or(TrainError::NoGradient { index: i })?;

            // Per-layer norm
            let layer_norm_sq: f64 = grad.iter().map(|&g| (g as f64) * (g as f64)).sum();
            let layer_norm = layer_norm_sq.sqrt() as f32;

            if layer_norm > self.max_norm {
                let scale = self.max_norm / (layer_norm + 1e-6);
                for g in grad.iter_mut() {
                    *g *= scale;
                }
                global_norm_sq += (self.max_norm as f64) * (self.max_norm as f64);
            } else {
                global_norm_sq += layer_norm_sq;
            }
        }

        let global_norm = (global_norm_sq.sqrt()) as f32;
        if global_norm.is_nan() || global_norm.is_infinite() {
            return Err(TrainError::InvalidGradNorm);
        }
        Ok(global_norm)
    }
}

// ─── ValueClip ───────────────────────────────────────────────────────────────

/// Hard-clip every gradient element to `[−clip_value, +clip_value]`.
///
/// Simple and fast, but does not preserve gradient direction.
#[derive(Debug, Clone)]
pub struct ValueClip {
    /// Absolute maximum element value.
    pub clip_value: f32,
}

impl ValueClip {
    /// Create a value clipper.
    #[must_use]
    pub fn new(clip_value: f32) -> Self {
        Self { clip_value }
    }
}

impl GradientClipper for ValueClip {
    fn clip(&self, params: &mut [ParamTensor]) -> TrainResult<f32> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }

        let mut total_norm_sq = 0.0_f64;

        for (i, p) in params.iter_mut().enumerate() {
            if !p.requires_grad {
                continue;
            }
            let grad = p.grad.as_mut().ok_or(TrainError::NoGradient { index: i })?;
            for g in grad.iter_mut() {
                total_norm_sq += (*g as f64) * (*g as f64);
                *g = g.clamp(-self.clip_value, self.clip_value);
            }
        }

        Ok((total_norm_sq.sqrt()) as f32)
    }
}

// ─── Convenience function ─────────────────────────────────────────────────────

/// Clip gradients by global norm in-place.  Returns the pre-clip norm.
///
/// This is a shorthand for `GlobalNormClip::new(max_norm).clip(params)`.
pub fn clip_grad_norm(params: &mut [ParamTensor], max_norm: f32) -> TrainResult<f32> {
    GlobalNormClip::new(max_norm).clip(params)
}

/// Clip gradient values in-place.  Returns the pre-clip global L2 norm.
pub fn clip_grad_value(params: &mut [ParamTensor], clip_value: f32) -> TrainResult<f32> {
    ValueClip::new(clip_value).clip(params)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn param_with_grad(data: Vec<f32>, grad: Vec<f32>) -> ParamTensor {
        let mut p = ParamTensor::new(data, "w");
        p.set_grad(grad).unwrap();
        p
    }

    // ── GlobalNormClip ────────────────────────────────────────────────────

    #[test]
    fn global_norm_clip_no_op_when_below_max() {
        let grad = vec![3.0_f32, 4.0_f32]; // norm = 5
        let mut params = vec![param_with_grad(vec![1.0; 2], grad.clone())];
        let clipper = GlobalNormClip::new(10.0);
        let pre_norm = clipper.clip(&mut params).unwrap();
        assert_abs_diff_eq!(pre_norm, 5.0_f32, epsilon = 1e-4);
        // Gradients should be unchanged
        assert_abs_diff_eq!(params[0].grad.as_ref().unwrap()[0], grad[0], epsilon = 1e-5);
    }

    #[test]
    fn global_norm_clip_scales_down_large_grads() {
        let mut params = vec![param_with_grad(vec![1.0; 2], vec![3.0_f32, 4.0_f32])]; // norm=5
        let clipper = GlobalNormClip::new(1.0);
        let pre_norm = clipper.clip(&mut params).unwrap();
        assert_abs_diff_eq!(pre_norm, 5.0_f32, epsilon = 1e-3);

        // After clip, norm should be ≤ max_norm
        let after_norm = GlobalNormClip::compute_norm(&params).unwrap();
        assert!(
            after_norm <= 1.05,
            "clipped norm should be ≤ max_norm, got {after_norm}"
        );
    }

    #[test]
    fn global_norm_clip_multi_param() {
        // Two params, each with gradient [3, 4] → joint norm = sqrt(2*(9+16)) = sqrt(50)
        let mut params = vec![
            param_with_grad(vec![1.0, 2.0], vec![3.0_f32, 4.0_f32]),
            param_with_grad(vec![3.0, 4.0], vec![3.0_f32, 4.0_f32]),
        ];
        let max_norm = 1.0_f32;
        let clipper = GlobalNormClip::new(max_norm);
        clipper.clip(&mut params).unwrap();
        let after = GlobalNormClip::compute_norm(&params).unwrap();
        assert!(
            after <= max_norm + 1e-3,
            "after_norm={after} should be ≤ {max_norm}"
        );
    }

    #[test]
    fn global_norm_clip_empty_error() {
        let res = GlobalNormClip::new(1.0).clip(&mut []);
        assert!(matches!(res, Err(TrainError::EmptyParams)));
    }

    // ── PerLayerNormClip ─────────────────────────────────────────────────

    #[test]
    fn per_layer_clip_clips_independently() {
        // Layer 0 norm = 100, layer 1 norm = 0.1; max = 1.0
        let mut params = vec![
            param_with_grad(vec![1.0; 1], vec![100.0_f32]),
            param_with_grad(vec![1.0; 1], vec![0.1_f32]),
        ];
        let clipper = PerLayerNormClip::new(1.0);
        clipper.clip(&mut params).unwrap();
        // Layer 0 should be clipped to ≈ 1.0
        assert!(
            params[0].grad.as_ref().unwrap()[0] <= 1.1,
            "layer 0 should be clipped"
        );
        // Layer 1 should be unchanged (0.1 < 1.0)
        assert_abs_diff_eq!(params[1].grad.as_ref().unwrap()[0], 0.1_f32, epsilon = 1e-5);
    }

    // ── ValueClip ────────────────────────────────────────────────────────

    #[test]
    fn value_clip_hard_clamps() {
        let mut params = vec![param_with_grad(
            vec![0.0; 4],
            vec![10.0, -10.0, 0.5, -0.3_f32],
        )];
        ValueClip::new(1.0).clip(&mut params).unwrap();
        let g = params[0].grad.as_ref().unwrap();
        assert_abs_diff_eq!(g[0], 1.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(g[1], -1.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(g[2], 0.5_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(g[3], -0.3_f32, epsilon = 1e-5);
    }

    // ── Convenience functions ─────────────────────────────────────────────

    #[test]
    fn clip_grad_norm_function() {
        let mut params = vec![param_with_grad(vec![1.0; 2], vec![3.0_f32, 4.0_f32])];
        clip_grad_norm(&mut params, 1.0).unwrap();
        let norm = GlobalNormClip::compute_norm(&params).unwrap();
        assert!(norm <= 1.1, "norm should be clipped to ≤ 1.0, got {norm}");
    }

    #[test]
    fn clip_grad_value_function() {
        let mut params = vec![param_with_grad(vec![0.0; 3], vec![5.0, -5.0, 0.5_f32])];
        clip_grad_value(&mut params, 2.0).unwrap();
        let g = params[0].grad.as_ref().unwrap();
        assert_abs_diff_eq!(g[0], 2.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(g[1], -2.0_f32, epsilon = 1e-5);
        assert_abs_diff_eq!(g[2], 0.5_f32, epsilon = 1e-5);
    }
}
