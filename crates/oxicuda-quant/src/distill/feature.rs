//! # Feature-Based Knowledge Distillation
//!
//! Matches the intermediate activations (feature maps) of the student to those
//! of the teacher, enabling the student to mimic the teacher's internal
//! representations in addition to (or instead of) its final predictions.
//!
//! ## Total loss
//!
//! ```text
//! L = Σ_l  w_l × loss(teacher_features_l, student_features_l)
//! ```
//!
//! where `w_l` is the user-specified weight for layer `l`.

use crate::distill::loss::DistilLoss;
use crate::error::{QuantError, QuantResult};

// ─── FeatureDistiller ─────────────────────────────────────────────────────────

/// Feature-based knowledge distillation.
///
/// Maintains a list of `(weight, DistilLoss)` pairs, one per distillation layer.
#[derive(Debug, Clone)]
pub struct FeatureDistiller {
    /// Per-layer distillation configuration: `(weight, loss)`.
    pub layers: Vec<(f32, DistilLoss)>,
}

impl FeatureDistiller {
    /// Create a feature distiller with equal-weight MSE loss for each layer.
    ///
    /// # Parameters
    ///
    /// * `n_layers` — number of intermediate layers to distil.
    #[must_use]
    pub fn uniform_mse(n_layers: usize) -> Self {
        let weight = if n_layers == 0 {
            1.0
        } else {
            1.0 / n_layers as f32
        };
        let layers = (0..n_layers).map(|_| (weight, DistilLoss::mse())).collect();
        Self { layers }
    }

    /// Create a feature distiller with custom per-layer weights and a shared loss.
    #[must_use]
    pub fn with_weights(weights: Vec<f32>, loss: DistilLoss) -> Self {
        let layers = weights.into_iter().map(|w| (w, loss)).collect();
        Self { layers }
    }

    /// Compute the distillation loss for a single layer pair.
    ///
    /// # Errors
    ///
    /// * [`QuantError::DimensionMismatch`] — `layer_index` out of range.
    /// * Propagates errors from the underlying `DistilLoss`.
    pub fn compute_layer_loss(
        &self,
        layer_index: usize,
        teacher_feat: &[f32],
        student_feat: &[f32],
    ) -> QuantResult<f32> {
        if layer_index >= self.layers.len() {
            return Err(QuantError::DimensionMismatch {
                expected: self.layers.len(),
                got: layer_index + 1,
            });
        }
        let (w, ref loss) = self.layers[layer_index];
        let l = loss.compute(teacher_feat, student_feat)?;
        Ok(w * l)
    }

    /// Compute the total feature distillation loss across all layers.
    ///
    /// `teacher_feats[l]` and `student_feats[l]` must have matching lengths.
    ///
    /// # Errors
    ///
    /// * [`QuantError::DimensionMismatch`] — wrong number of feature arrays.
    /// * Propagates per-layer errors.
    pub fn compute_total_loss(
        &self,
        teacher_feats: &[&[f32]],
        student_feats: &[&[f32]],
    ) -> QuantResult<f32> {
        if teacher_feats.len() != self.layers.len() {
            return Err(QuantError::DimensionMismatch {
                expected: self.layers.len(),
                got: teacher_feats.len(),
            });
        }
        if student_feats.len() != self.layers.len() {
            return Err(QuantError::DimensionMismatch {
                expected: self.layers.len(),
                got: student_feats.len(),
            });
        }
        let total: f32 = (0..self.layers.len())
            .map(|l| self.compute_layer_loss(l, teacher_feats[l], student_feats[l]))
            .collect::<QuantResult<Vec<f32>>>()?
            .iter()
            .sum();
        Ok(total)
    }

    /// Number of distillation layers.
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Normalise layer weights so they sum to 1.
    pub fn normalise_weights(&mut self) {
        let sum: f32 = self
            .layers
            .iter()
            .map(|(w, _)| w.abs())
            .sum::<f32>()
            .max(1e-12);
        for (w, _) in &mut self.layers {
            *w /= sum;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn uniform_mse_layer_count() {
        let d = FeatureDistiller::uniform_mse(4);
        assert_eq!(d.n_layers(), 4);
        for (w, _) in &d.layers {
            assert_abs_diff_eq!(*w, 0.25, epsilon = 1e-6);
        }
    }

    #[test]
    fn zero_loss_for_identical_features() {
        let d = FeatureDistiller::uniform_mse(2);
        let feat = vec![1.0_f32, 2.0, 3.0];
        let t0 = feat.as_slice();
        let t1 = feat.as_slice();
        let loss = d.compute_total_loss(&[t0, t1], &[t0, t1]).unwrap();
        assert_abs_diff_eq!(loss, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn positive_loss_for_different_features() {
        let d = FeatureDistiller::uniform_mse(1);
        let teacher = vec![1.0_f32, 0.0, 0.0];
        let student = vec![0.0_f32, 1.0, 0.0];
        let loss = d.compute_total_loss(&[&teacher], &[&student]).unwrap();
        assert!(loss > 0.0, "loss should be positive for different features");
    }

    #[test]
    fn layer_count_mismatch_error() {
        let d = FeatureDistiller::uniform_mse(2);
        let feat = vec![1.0_f32; 4];
        // Provide 3 teacher arrays but distiller expects 2.
        let err = d.compute_total_loss(&[&feat, &feat, &feat], &[&feat, &feat]);
        assert!(matches!(err, Err(QuantError::DimensionMismatch { .. })));
    }

    #[test]
    fn layer_index_out_of_range_error() {
        let d = FeatureDistiller::uniform_mse(2);
        let feat = vec![1.0_f32; 4];
        assert!(matches!(
            d.compute_layer_loss(5, &feat, &feat),
            Err(QuantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn normalise_weights() {
        let mut d = FeatureDistiller::with_weights(vec![2.0, 3.0, 5.0], DistilLoss::mse());
        d.normalise_weights();
        let sum: f32 = d.layers.iter().map(|(w, _)| *w).sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn with_weights_constructs_correctly() {
        let d = FeatureDistiller::with_weights(vec![0.3, 0.7], DistilLoss::cosine());
        assert_eq!(d.n_layers(), 2);
        assert_abs_diff_eq!(d.layers[0].0, 0.3, epsilon = 1e-6);
        assert_abs_diff_eq!(d.layers[1].0, 0.7, epsilon = 1e-6);
    }

    #[test]
    fn kl_feature_distillation() {
        let loss = DistilLoss::kl_divergence(2.0);
        let d = FeatureDistiller::with_weights(vec![1.0], loss);
        let teacher = vec![0.1_f32, 0.7, 0.2];
        let student = vec![0.4_f32, 0.4, 0.2];
        let l = d.compute_total_loss(&[&teacher], &[&student]).unwrap();
        assert!(l >= 0.0, "KL loss must be non-negative");
    }
}
