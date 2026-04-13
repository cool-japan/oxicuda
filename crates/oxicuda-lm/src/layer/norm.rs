//! Normalization layers: RMSNorm and LayerNorm.
//!
//! Both layers carry learned scale (and, for LayerNorm, bias) weights and
//! operate over the hidden dimension of each token independently.

use crate::error::{LmError, LmResult};

// ─── Math helpers ─────────────────────────────────────────────────────────────

/// Compute the mean of a slice.
fn mean(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let sum: f32 = x.iter().sum();
    sum / x.len() as f32
}

/// Compute the variance of a slice (population variance).
fn variance(x: &[f32], mu: f32) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / x.len() as f32
}

// ─── RmsNorm ─────────────────────────────────────────────────────────────────

/// RMSNorm: `out = (x / rms(x)) * weight`
///
/// Used by LLaMA-family models.  Faster than LayerNorm because it skips mean
/// centering.  `rms(x) = sqrt(mean(x²) + eps)`.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Hidden dimension.
    pub dim: usize,
    /// Stability epsilon (typically 1e-5 or 1e-6).
    pub eps: f32,
    /// Learnable scale: `[dim]`.  Initialised to ones.
    pub weight: Vec<f32>,
}

impl RmsNorm {
    /// Construct with default scale = 1.0.
    pub fn new(dim: usize, eps: f32) -> LmResult<Self> {
        if dim == 0 {
            return Err(LmError::InvalidConfig {
                msg: "RmsNorm dim must be > 0".into(),
            });
        }
        Ok(Self {
            dim,
            eps,
            weight: vec![1.0_f32; dim],
        })
    }

    /// Construct from an existing scale vector.
    pub fn from_weight(weight: Vec<f32>, eps: f32) -> LmResult<Self> {
        let dim = weight.len();
        if dim == 0 {
            return Err(LmError::InvalidConfig {
                msg: "RmsNorm weight must be non-empty".into(),
            });
        }
        Ok(Self { dim, eps, weight })
    }

    /// Forward pass.
    ///
    /// `x` has shape `[n_tokens × dim]` (flat, row-major).
    /// Returns the same shape.
    pub fn forward(&self, x: &[f32], n_tokens: usize) -> LmResult<Vec<f32>> {
        if x.len() != n_tokens * self.dim {
            return Err(LmError::DimensionMismatch {
                expected: n_tokens * self.dim,
                got: x.len(),
            });
        }
        let mut out = vec![0.0_f32; x.len()];
        for t in 0..n_tokens {
            let row = &x[t * self.dim..(t + 1) * self.dim];
            // rms = sqrt(mean(x^2) + eps)
            let mean_sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / self.dim as f32;
            let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();
            let out_row = &mut out[t * self.dim..(t + 1) * self.dim];
            for (i, (&xi, &wi)) in row.iter().zip(self.weight.iter()).enumerate() {
                out_row[i] = xi * inv_rms * wi;
            }
        }
        Ok(out)
    }
}

// ─── LayerNorm ───────────────────────────────────────────────────────────────

/// LayerNorm: `out = (x - mean) / sqrt(var + eps) * weight + bias`
///
/// Used by GPT-2 family models.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Hidden dimension.
    pub dim: usize,
    /// Stability epsilon.
    pub eps: f32,
    /// Learnable scale: `[dim]`.  Initialised to ones.
    pub weight: Vec<f32>,
    /// Learnable bias: `[dim]`.  Initialised to zeros.
    pub bias: Vec<f32>,
}

impl LayerNorm {
    /// Construct with default scale = 1.0 and bias = 0.0.
    pub fn new(dim: usize, eps: f32) -> LmResult<Self> {
        if dim == 0 {
            return Err(LmError::InvalidConfig {
                msg: "LayerNorm dim must be > 0".into(),
            });
        }
        Ok(Self {
            dim,
            eps,
            weight: vec![1.0_f32; dim],
            bias: vec![0.0_f32; dim],
        })
    }

    /// Construct from existing weight and bias vectors.
    pub fn from_weights(weight: Vec<f32>, bias: Vec<f32>, eps: f32) -> LmResult<Self> {
        let dim = weight.len();
        if dim == 0 {
            return Err(LmError::InvalidConfig {
                msg: "LayerNorm weight must be non-empty".into(),
            });
        }
        if bias.len() != dim {
            return Err(LmError::DimensionMismatch {
                expected: dim,
                got: bias.len(),
            });
        }
        Ok(Self {
            dim,
            eps,
            weight,
            bias,
        })
    }

    /// Forward pass.
    ///
    /// `x` has shape `[n_tokens × dim]` (flat, row-major).
    /// Returns the same shape.
    pub fn forward(&self, x: &[f32], n_tokens: usize) -> LmResult<Vec<f32>> {
        if x.len() != n_tokens * self.dim {
            return Err(LmError::DimensionMismatch {
                expected: n_tokens * self.dim,
                got: x.len(),
            });
        }
        let mut out = vec![0.0_f32; x.len()];
        for t in 0..n_tokens {
            let row = &x[t * self.dim..(t + 1) * self.dim];
            let mu = mean(row);
            let var = variance(row, mu);
            let inv_std = 1.0 / (var + self.eps).sqrt();
            let out_row = &mut out[t * self.dim..(t + 1) * self.dim];
            for (i, (&xi, (&wi, &bi))) in row
                .iter()
                .zip(self.weight.iter().zip(self.bias.iter()))
                .enumerate()
            {
                out_row[i] = (xi - mu) * inv_std * wi + bi;
            }
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RmsNorm ──────────────────────────────────────────────────────────

    #[test]
    fn rms_norm_ones_weight_identity_direction() {
        // With weight=1, x / rms(x) is unit-normalised.
        let n = RmsNorm::new(4, 1e-8).unwrap();
        let x = vec![3.0_f32, 4.0, 0.0, 0.0]; // rms = sqrt((9+16)/4) = sqrt(6.25) = 2.5
        let out = n.forward(&x, 1).unwrap();
        // out[0] = 3/2.5 = 1.2, out[1] = 4/2.5 = 1.6
        assert!((out[0] - 1.2).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 1.6).abs() < 1e-5, "out[1]={}", out[1]);
        assert!(out[2].abs() < 1e-5);
    }

    #[test]
    fn rms_norm_scale_weight() {
        let mut n = RmsNorm::new(2, 1e-8).unwrap();
        n.weight = vec![2.0, 0.5];
        let x = vec![1.0_f32, 1.0]; // rms = 1.0
        let out = n.forward(&x, 1).unwrap();
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn rms_norm_batch_tokens() {
        let n = RmsNorm::new(2, 1e-8).unwrap();
        // 2 tokens, dim=2
        let x = vec![1.0_f32, 1.0, 2.0, 2.0];
        let out = n.forward(&x, 2).unwrap();
        // Each row: rms = 1.0 (all same value normalized to 1)
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-5, "v={v}");
        }
    }

    #[test]
    fn rms_norm_dim_mismatch_error() {
        let n = RmsNorm::new(4, 1e-8).unwrap();
        let err = n.forward(&[1.0, 2.0], 1).unwrap_err();
        assert!(matches!(err, LmError::DimensionMismatch { .. }));
    }

    #[test]
    fn rms_norm_zero_dim_error() {
        assert!(RmsNorm::new(0, 1e-5).is_err());
    }

    #[test]
    fn rms_norm_from_weight() {
        let n = RmsNorm::from_weight(vec![1.0, 2.0, 3.0], 1e-8).unwrap();
        assert_eq!(n.dim, 3);
    }

    // ── LayerNorm ─────────────────────────────────────────────────────────

    #[test]
    fn layer_norm_zero_centered_unit_variance() {
        // x = [1,2,3,4]; mean=2.5, var=1.25
        let ln = LayerNorm::new(4, 1e-8).unwrap();
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = ln.forward(&x, 1).unwrap();
        // Check mean ≈ 0 and std ≈ 1
        let m = out.iter().sum::<f32>() / 4.0;
        let v = out.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / 4.0;
        assert!(m.abs() < 1e-5, "mean={m}");
        assert!((v - 1.0).abs() < 1e-4, "var={v}");
    }

    #[test]
    fn layer_norm_weight_and_bias() {
        // weight = 2, bias = 1 → out = 2*(x_norm) + 1
        let ln = LayerNorm::from_weights(vec![2.0, 2.0], vec![1.0, 1.0], 1e-8).unwrap();
        let x = vec![0.0_f32, 0.0]; // both zero → norm = 0 → out = bias = 1
        let out = ln.forward(&x, 1).unwrap();
        // mean=0, var=0 → x_norm = 0/sqrt(eps) → ≈ 0, so out ≈ bias = 1
        assert!((out[0] - 1.0).abs() < 1e-3, "out[0]={}", out[0]);
    }

    #[test]
    fn layer_norm_batch_tokens() {
        let ln = LayerNorm::new(4, 1e-8).unwrap();
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let out = ln.forward(&x, 2).unwrap();
        assert_eq!(out.len(), 8);
        // First token and second token both normalized to have mean≈0
        let m1: f32 = out[..4].iter().sum::<f32>() / 4.0;
        let m2: f32 = out[4..].iter().sum::<f32>() / 4.0;
        assert!(m1.abs() < 1e-5, "token 0 mean={m1}");
        assert!(m2.abs() < 1e-5, "token 1 mean={m2}");
    }

    #[test]
    fn layer_norm_dim_mismatch_error() {
        let ln = LayerNorm::new(4, 1e-8).unwrap();
        let err = ln.forward(&[1.0, 2.0], 1).unwrap_err();
        assert!(matches!(err, LmError::DimensionMismatch { .. }));
    }

    #[test]
    fn layer_norm_weight_bias_dim_mismatch_error() {
        let err = LayerNorm::from_weights(vec![1.0, 2.0], vec![0.0], 1e-5).unwrap_err();
        assert!(matches!(err, LmError::DimensionMismatch { .. }));
    }

    #[test]
    fn layer_norm_zero_dim_error() {
        assert!(LayerNorm::new(0, 1e-5).is_err());
    }
}
