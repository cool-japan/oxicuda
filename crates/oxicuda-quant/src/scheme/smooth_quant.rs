//! # SmoothQuant — Activation–Weight Quantization Migration
//!
//! Xiao et al. (2022): "SmoothQuant: Accurate and Efficient Post-Training
//! Quantization for Large Language Models" <https://arxiv.org/abs/2211.10438>
//!
//! LLM activations often contain large per-channel outliers that make INT8
//! quantization difficult, while weights are typically well-behaved.
//! SmoothQuant migrates the quantization difficulty from activations to weights
//! via a mathematically equivalent per-channel rescaling.
//!
//! ## Migration
//!
//! ```text
//! s_j = max|X_j|^α / max|W_j|^(1−α)    (per-channel scale)
//!
//! X_smooth[:,j] = X[:,j] / s_j          (activations ÷ s)
//! W_smooth[:,j] = W[:,j] × s_j          (weights × s, column = input channel)
//!
//! Y = X W^T = X_smooth W_smooth^T       (output unchanged)
//! ```
//!
//! `α = 0.5` balances difficulty equally.  `α → 1` pushes all difficulty to
//! weights; `α → 0` leaves activations as-is.

use crate::error::{QuantError, QuantResult};

// ─── Config ───────────────────────────────────────────────────────────────────

/// SmoothQuant migration configuration.
#[derive(Debug, Clone, Copy)]
pub struct SmoothQuantConfig {
    /// Migration strength α ∈ [0, 1].
    ///
    /// * 0.5 — equal difficulty between activations and weights (default).
    /// * 1.0 — migrate all difficulty to weights (activations easy, weights hard).
    /// * 0.0 — no migration (activations carry full difficulty).
    pub alpha: f32,
}

impl Default for SmoothQuantConfig {
    fn default() -> Self {
        Self { alpha: 0.5 }
    }
}

// ─── SmoothQuantMigrator ─────────────────────────────────────────────────────

/// Applies per-channel scaling to balance quantization difficulty.
///
/// The migrator operates on linear layers:
/// * **Activations** `X` — shape `[n_tokens, n_channels]`
/// * **Weights** `W` — shape `[n_out, n_channels]` (transposed: `Y = X W^T`)
#[derive(Debug, Clone, Copy)]
pub struct SmoothQuantMigrator {
    /// Migration configuration.
    pub config: SmoothQuantConfig,
}

impl SmoothQuantMigrator {
    /// Create a migrator with the given `alpha`.
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            config: SmoothQuantConfig { alpha },
        }
    }

    /// Compute per-channel migration scales from pre-aggregated statistics.
    ///
    /// # Parameters
    ///
    /// * `act_max`    — per-channel max absolute value of activations (length `n_ch`).
    /// * `weight_max` — per-channel (column) max absolute value of weights (length `n_ch`).
    ///
    /// # Returns
    ///
    /// Scale vector `s` of length `n_ch` where
    /// `s[j] = act_max[j]^alpha / weight_max[j]^(1−alpha)`.
    ///
    /// # Errors
    ///
    /// * [`QuantError::DimensionMismatch`] — `act_max` and `weight_max` differ in length.
    /// * [`QuantError::EmptyInput`] — either slice is empty.
    pub fn compute_migration_scales(
        &self,
        act_max: &[f32],
        weight_max: &[f32],
    ) -> QuantResult<Vec<f32>> {
        if act_max.is_empty() {
            return Err(QuantError::EmptyInput(
                "SmoothQuantMigrator::compute_migration_scales",
            ));
        }
        if act_max.len() != weight_max.len() {
            return Err(QuantError::DimensionMismatch {
                expected: act_max.len(),
                got: weight_max.len(),
            });
        }
        let alpha = self.config.alpha;
        let scales = act_max
            .iter()
            .zip(weight_max.iter())
            .map(|(&a_max, &w_max)| {
                let a = a_max.abs().max(1e-8);
                let w = w_max.abs().max(1e-8);
                a.powf(alpha) / w.powf(1.0 - alpha)
            })
            .collect();
        Ok(scales)
    }

    /// Compute per-channel max absolute values from an activation tensor.
    ///
    /// # Parameters
    ///
    /// * `acts`       — row-major activation matrix `[n_tokens, n_channels]`.
    /// * `n_tokens`   — number of tokens (rows).
    /// * `n_channels` — hidden dimension (columns).
    ///
    /// # Errors
    ///
    /// * [`QuantError::DimensionMismatch`] — slice length ≠ `n_tokens × n_channels`.
    /// * [`QuantError::EmptyInput`] — either dimension is 0.
    pub fn compute_act_stats(
        acts: &[f32],
        n_tokens: usize,
        n_channels: usize,
    ) -> QuantResult<Vec<f32>> {
        if acts.is_empty() {
            return Err(QuantError::EmptyInput(
                "compute_act_stats: empty activations",
            ));
        }
        if acts.len() != n_tokens * n_channels {
            return Err(QuantError::DimensionMismatch {
                expected: n_tokens * n_channels,
                got: acts.len(),
            });
        }
        let mut stats = vec![0.0_f32; n_channels];
        for t in 0..n_tokens {
            for j in 0..n_channels {
                let v = acts[t * n_channels + j].abs();
                if v > stats[j] {
                    stats[j] = v;
                }
            }
        }
        Ok(stats)
    }

    /// Compute per-column (input-channel) max absolute values from a weight matrix.
    ///
    /// # Parameters
    ///
    /// * `weights`    — row-major weight matrix `[n_out, n_channels]`.
    /// * `n_out`      — number of output features (rows).
    /// * `n_channels` — number of input features (columns).
    ///
    /// # Errors
    ///
    /// * [`QuantError::DimensionMismatch`] — slice length ≠ `n_out × n_channels`.
    /// * [`QuantError::EmptyInput`] — either dimension is 0.
    pub fn compute_weight_stats(
        weights: &[f32],
        n_out: usize,
        n_channels: usize,
    ) -> QuantResult<Vec<f32>> {
        if weights.is_empty() {
            return Err(QuantError::EmptyInput(
                "compute_weight_stats: empty weights",
            ));
        }
        if weights.len() != n_out * n_channels {
            return Err(QuantError::DimensionMismatch {
                expected: n_out * n_channels,
                got: weights.len(),
            });
        }
        let mut stats = vec![0.0_f32; n_channels];
        for r in 0..n_out {
            for j in 0..n_channels {
                let v = weights[r * n_channels + j].abs();
                if v > stats[j] {
                    stats[j] = v;
                }
            }
        }
        Ok(stats)
    }

    /// Divide each activation channel j by `scales[j]` in-place.
    ///
    /// # Errors
    ///
    /// * [`QuantError::DimensionMismatch`] — inconsistent lengths.
    pub fn smooth_activations(
        acts: &mut [f32],
        scales: &[f32],
        n_tokens: usize,
        n_channels: usize,
    ) -> QuantResult<()> {
        if acts.len() != n_tokens * n_channels {
            return Err(QuantError::DimensionMismatch {
                expected: n_tokens * n_channels,
                got: acts.len(),
            });
        }
        if scales.len() != n_channels {
            return Err(QuantError::DimensionMismatch {
                expected: n_channels,
                got: scales.len(),
            });
        }
        for t in 0..n_tokens {
            for j in 0..n_channels {
                acts[t * n_channels + j] /= scales[j].max(1e-12);
            }
        }
        Ok(())
    }

    /// Multiply each weight column j (input channel) by `scales[j]` in-place.
    ///
    /// Weights are assumed to have shape `[n_out, n_channels]`.
    ///
    /// # Errors
    ///
    /// * [`QuantError::DimensionMismatch`] — inconsistent lengths.
    pub fn smooth_weights(
        weights: &mut [f32],
        scales: &[f32],
        n_out: usize,
        n_channels: usize,
    ) -> QuantResult<()> {
        if weights.len() != n_out * n_channels {
            return Err(QuantError::DimensionMismatch {
                expected: n_out * n_channels,
                got: weights.len(),
            });
        }
        if scales.len() != n_channels {
            return Err(QuantError::DimensionMismatch {
                expected: n_channels,
                got: scales.len(),
            });
        }
        for r in 0..n_out {
            for j in 0..n_channels {
                weights[r * n_channels + j] *= scales[j];
            }
        }
        Ok(())
    }

    /// Smooth a complete linear layer: compute scales, apply to activations and weights.
    ///
    /// # Parameters
    ///
    /// * `acts`       — mutable activation matrix `[n_tokens, n_channels]`.
    /// * `weights`    — mutable weight matrix `[n_out, n_channels]`.
    /// * `n_tokens`   — token (batch) dimension.
    /// * `n_channels` — input feature dimension.
    /// * `n_out`      — output feature dimension.
    ///
    /// # Returns
    ///
    /// The per-channel migration scales used (length `n_channels`).
    ///
    /// # Errors
    ///
    /// Propagates all dimension and empty-input errors from sub-operations.
    pub fn smooth_layer(
        &self,
        acts: &mut [f32],
        weights: &mut [f32],
        n_tokens: usize,
        n_channels: usize,
        n_out: usize,
    ) -> QuantResult<Vec<f32>> {
        let act_stats = Self::compute_act_stats(acts, n_tokens, n_channels)?;
        let weight_stats = Self::compute_weight_stats(weights, n_out, n_channels)?;
        let scales = self.compute_migration_scales(&act_stats, &weight_stats)?;
        Self::smooth_activations(acts, &scales, n_tokens, n_channels)?;
        Self::smooth_weights(weights, &scales, n_out, n_channels)?;
        Ok(scales)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Simple matrix multiply for test verification.
    fn matmul_nt(x: &[f32], w: &[f32], n_tok: usize, n_ch: usize, n_out: usize) -> Vec<f32> {
        // Y = X W^T   (X: n_tok × n_ch, W: n_out × n_ch)
        let mut y = vec![0.0_f32; n_tok * n_out];
        for t in 0..n_tok {
            for o in 0..n_out {
                let dot: f32 = (0..n_ch).map(|j| x[t * n_ch + j] * w[o * n_ch + j]).sum();
                y[t * n_out + o] = dot;
            }
        }
        y
    }

    #[test]
    fn scale_alpha_half() {
        let m = SmoothQuantMigrator::new(0.5);
        let act_max = vec![4.0_f32, 1.0, 9.0];
        let weight_max = vec![1.0_f32, 4.0, 1.0];
        let scales = m.compute_migration_scales(&act_max, &weight_max).unwrap();
        // s[0] = 4^0.5 / 1^0.5 = 2/1 = 2
        assert_abs_diff_eq!(scales[0], 2.0, epsilon = 1e-5);
        // s[1] = 1^0.5 / 4^0.5 = 1/2 = 0.5
        assert_abs_diff_eq!(scales[1], 0.5, epsilon = 1e-5);
        // s[2] = 9^0.5 / 1^0.5 = 3/1 = 3
        assert_abs_diff_eq!(scales[2], 3.0, epsilon = 1e-5);
    }

    #[test]
    fn scale_alpha_one_activations_only() {
        // alpha=1 → s = act_max / weight_max^0 = act_max
        let m = SmoothQuantMigrator::new(1.0);
        let act_max = vec![2.0_f32, 5.0];
        let weight_max = vec![3.0_f32, 7.0]; // ignored
        let scales = m.compute_migration_scales(&act_max, &weight_max).unwrap();
        assert_abs_diff_eq!(scales[0], 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(scales[1], 5.0, epsilon = 1e-5);
    }

    #[test]
    fn scale_alpha_zero_weights_only() {
        // alpha=0 → s = act_max^0 / weight_max^1 = 1 / weight_max
        let m = SmoothQuantMigrator::new(0.0);
        let act_max = vec![4.0_f32, 1.0]; // ignored
        let weight_max = vec![2.0_f32, 5.0];
        let scales = m.compute_migration_scales(&act_max, &weight_max).unwrap();
        assert_abs_diff_eq!(scales[0], 1.0 / 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(scales[1], 1.0 / 5.0, epsilon = 1e-5);
    }

    #[test]
    fn smoothing_preserves_layer_output() {
        let m = SmoothQuantMigrator::new(0.5);
        let n_tok = 3;
        let n_ch = 4;
        let n_out = 2;
        let mut acts: Vec<f32> = (0..(n_tok * n_ch))
            .map(|i| (i as f32 * 0.3) - 1.0)
            .collect();
        let mut weights: Vec<f32> = (0..(n_out * n_ch))
            .map(|i| (i as f32 * 0.2) - 0.5)
            .collect();

        // Compute original output.
        let y_orig = matmul_nt(&acts, &weights, n_tok, n_ch, n_out);

        // Smooth the layer.
        m.smooth_layer(&mut acts, &mut weights, n_tok, n_ch, n_out)
            .unwrap();

        // Compute smoothed output.
        let y_smooth = matmul_nt(&acts, &weights, n_tok, n_ch, n_out);

        // Outputs must match.
        for (a, b) in y_orig.iter().zip(y_smooth.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-4);
        }
    }

    #[test]
    fn activation_stats_max_per_channel() {
        // 2 tokens, 3 channels
        // acts = [[1, -5, 2], [-3, 4, 1]]
        let acts = vec![1.0_f32, -5.0, 2.0, -3.0, 4.0, 1.0];
        let stats = SmoothQuantMigrator::compute_act_stats(&acts, 2, 3).unwrap();
        assert_abs_diff_eq!(stats[0], 3.0, epsilon = 1e-6); // max(|1|, |-3|) = 3
        assert_abs_diff_eq!(stats[1], 5.0, epsilon = 1e-6); // max(|-5|, |4|) = 5
        assert_abs_diff_eq!(stats[2], 2.0, epsilon = 1e-6); // max(|2|, |1|) = 2
    }

    #[test]
    fn weight_stats_max_per_column() {
        // weights [2 out, 3 in] = [[0.5, -2.0, 1.0], [-1.5, 0.3, 3.0]]
        let w = vec![0.5_f32, -2.0, 1.0, -1.5, 0.3, 3.0];
        let stats = SmoothQuantMigrator::compute_weight_stats(&w, 2, 3).unwrap();
        assert_abs_diff_eq!(stats[0], 1.5, epsilon = 1e-6);
        assert_abs_diff_eq!(stats[1], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(stats[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn dimension_mismatch_error() {
        let m = SmoothQuantMigrator::new(0.5);
        let act_max = vec![1.0_f32; 3];
        let weight_max = vec![1.0_f32; 4]; // wrong
        assert!(matches!(
            m.compute_migration_scales(&act_max, &weight_max),
            Err(QuantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn empty_input_error() {
        let m = SmoothQuantMigrator::new(0.5);
        assert!(matches!(
            m.compute_migration_scales(&[], &[]),
            Err(QuantError::EmptyInput(_))
        ));
    }

    #[test]
    fn smoothing_reduces_act_channel_range_imbalance() {
        // Channel 0 has very large activations, channel 1 is normal.
        let m = SmoothQuantMigrator::new(0.5);
        let n_tok = 4;
        let n_ch = 2;
        let n_out = 2;
        let mut acts = vec![100.0_f32, 1.0, -100.0, 1.0, 100.0, -1.0, -100.0, -1.0];
        let mut weights = vec![0.5_f32, 0.5, -0.5, 0.5];

        let scales = m
            .smooth_layer(&mut acts, &mut weights, n_tok, n_ch, n_out)
            .unwrap();
        // After smoothing, channel 0 max |act| should be reduced.
        let act_max_0: f32 = (0..n_tok)
            .map(|t| acts[t * n_ch].abs())
            .fold(0.0_f32, f32::max);
        let act_max_1: f32 = (0..n_tok)
            .map(|t| acts[t * n_ch + 1].abs())
            .fold(0.0_f32, f32::max);
        // The ratio should be closer to 1 than the original 100:1.
        let ratio = act_max_0 / act_max_1.max(1e-8);
        // scale[0] > 1 means acts[:,0] was divided by > 1, reducing its range.
        assert!(
            scales[0] > 1.0,
            "scale[0] should be > 1 for outlier channel"
        );
        assert!(
            ratio < 100.0,
            "channel range imbalance should decrease after smoothing"
        );
    }
}
