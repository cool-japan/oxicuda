//! # Magnitude-Based Unstructured Pruning
//!
//! Prunes the weights with the smallest absolute value (L1) or squared value (L2),
//! achieving a target sparsity level while keeping the model architecture intact.
//!
//! ## Algorithm
//!
//! 1. Compute a scalar importance score `s_i` for each weight.
//! 2. Sort scores ascending.
//! 3. Set the bottom `target_sparsity` fraction to zero (mask = false).

use crate::error::{QuantError, QuantResult};
use crate::pruning::mask::SparseMask;

// ─── Norm ─────────────────────────────────────────────────────────────────────

/// Norm used to compute per-element importance scores for pruning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagnitudeNorm {
    /// L1 norm: importance = |w|.
    L1,
    /// L2 norm: importance = w².
    L2,
}

// ─── MagnitudePruner ─────────────────────────────────────────────────────────

/// Unstructured weight pruner based on element-wise magnitude.
///
/// Prunes the `target_sparsity` fraction of weights with the smallest
/// L1 (or L2) norm by setting them to zero and returning a [`SparseMask`].
#[derive(Debug, Clone)]
pub struct MagnitudePruner {
    /// Fraction of weights to prune ∈ (0, 1).
    pub target_sparsity: f32,
    /// Importance norm used for ranking.
    pub norm: MagnitudeNorm,
}

impl MagnitudePruner {
    /// Create a new magnitude pruner.
    ///
    /// # Panics
    ///
    /// Panics if `target_sparsity` is not in `[0, 1)`.
    #[must_use]
    pub fn new(target_sparsity: f32, norm: MagnitudeNorm) -> Self {
        assert!(
            (0.0..1.0).contains(&target_sparsity),
            "target_sparsity must be in [0, 1), got {target_sparsity}"
        );
        Self {
            target_sparsity,
            norm,
        }
    }

    /// Compute the pruning mask for `weights` at `target_sparsity`.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`]      — `weights` is empty.
    /// * [`QuantError::AllZeroPruning`]  — all weights would be zeroed.
    pub fn compute_mask(&self, weights: &[f32]) -> QuantResult<SparseMask> {
        if weights.is_empty() {
            return Err(QuantError::EmptyInput("MagnitudePruner::compute_mask"));
        }
        let n = weights.len();
        let n_prune = ((n as f32) * self.target_sparsity).ceil() as usize;

        if n_prune >= n {
            return Err(QuantError::AllZeroPruning {
                threshold: self.target_sparsity,
                n,
            });
        }

        // Compute importance scores.
        let scores: Vec<f32> = weights
            .iter()
            .map(|&w| match self.norm {
                MagnitudeNorm::L1 => w.abs(),
                MagnitudeNorm::L2 => w * w,
            })
            .collect();

        // Find the threshold (the (n_prune)-th smallest importance).
        let mut sorted = scores.clone();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[n_prune.saturating_sub(1)];

        // Build mask: keep weights whose score > threshold.
        // Tie-breaking: prune exactly n_prune entries.
        let mut n_pruned = 0_usize;
        let mask: Vec<bool> = scores
            .iter()
            .map(|&s| {
                if n_pruned < n_prune && s <= threshold {
                    n_pruned += 1;
                    false
                } else {
                    true
                }
            })
            .collect();

        Ok(SparseMask { mask })
    }

    /// Apply pruning: compute the mask and zero out pruned weights in-place.
    ///
    /// Returns the mask that was applied.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`compute_mask`](Self::compute_mask).
    pub fn prune(&self, weights: &mut [f32]) -> QuantResult<SparseMask> {
        let mask = self.compute_mask(weights)?;
        mask.apply_in_place(weights);
        Ok(mask)
    }

    /// Compute per-group masks for a weight matrix (e.g., per output channel).
    ///
    /// Prunes each group of `group_size` elements independently to the target sparsity.
    ///
    /// # Errors
    ///
    /// * [`QuantError::GroupSizeMismatch`] — `weights.len()` not divisible by `group_size`.
    /// * [`QuantError::EmptyInput`]        — `weights` is empty.
    pub fn compute_grouped_mask(
        &self,
        weights: &[f32],
        group_size: usize,
    ) -> QuantResult<SparseMask> {
        if weights.is_empty() {
            return Err(QuantError::EmptyInput(
                "MagnitudePruner::compute_grouped_mask",
            ));
        }
        if weights.len() % group_size != 0 {
            return Err(QuantError::GroupSizeMismatch {
                len: weights.len(),
                group: group_size,
            });
        }
        let mut combined = Vec::with_capacity(weights.len());
        for chunk in weights.chunks_exact(group_size) {
            let chunk_mask = self.compute_mask(chunk)?;
            combined.extend_from_slice(&chunk_mask.mask);
        }
        Ok(SparseMask { mask: combined })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn l1_prune_50_percent() {
        let p = MagnitudePruner::new(0.5, MagnitudeNorm::L1);
        let weights = vec![0.1_f32, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.6];
        let mask = p.compute_mask(&weights).unwrap();
        assert_abs_diff_eq!(mask.sparsity(), 0.5, epsilon = 0.01);
        // Smallest 4 by |w|: 0.1, 0.2, 0.3, 0.4 → pruned
        assert!(!mask.mask[0], "0.1 should be pruned");
        assert!(mask.mask[1], "0.5 should be active");
    }

    #[test]
    fn l2_prune_25_percent() {
        let p = MagnitudePruner::new(0.25, MagnitudeNorm::L2);
        let weights = vec![0.1_f32, 0.5, 0.3, 0.9];
        let mask = p.compute_mask(&weights).unwrap();
        // n_prune = ceil(4 * 0.25) = 1 → prune smallest w² = 0.01 → index 0
        assert_eq!(mask.count_pruned(), 1);
        assert!(!mask.mask[0]);
    }

    #[test]
    fn prune_in_place_zeroes_weights() {
        let p = MagnitudePruner::new(0.5, MagnitudeNorm::L1);
        let mut w = vec![1.0_f32, 0.01, 2.0, 0.02];
        let mask = p.prune(&mut w).unwrap();
        assert!(mask.count_pruned() >= 1);
        // The two smallest (0.01 and 0.02) should be zeroed.
        assert_abs_diff_eq!(w[1], 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(w[3], 0.0, epsilon = 1e-7);
    }

    #[test]
    fn empty_input_error() {
        let p = MagnitudePruner::new(0.5, MagnitudeNorm::L1);
        assert!(matches!(
            p.compute_mask(&[]),
            Err(QuantError::EmptyInput(_))
        ));
    }

    #[test]
    fn all_zero_pruning_error() {
        // sparsity=1.0 is invalid (panics), but n_prune >= n at sparsity close to 1.
        // Manually test the guard: n_prune = ceil(4 * 0.99) = 4 >= n=4.
        let p = MagnitudePruner {
            target_sparsity: 0.99,
            norm: MagnitudeNorm::L1,
        };
        let w = vec![0.1_f32; 4];
        assert!(matches!(
            p.compute_mask(&w),
            Err(QuantError::AllZeroPruning { .. })
        ));
    }

    #[test]
    fn zero_sparsity_all_active() {
        let p = MagnitudePruner::new(0.0, MagnitudeNorm::L1);
        let w = vec![0.5_f32, 1.0, 2.0];
        let mask = p.compute_mask(&w).unwrap();
        assert_eq!(mask.count_pruned(), 0);
    }

    #[test]
    fn grouped_mask_per_channel() {
        let p = MagnitudePruner::new(0.5, MagnitudeNorm::L1);
        // 2 groups of 4
        let w = vec![
            0.1_f32, 0.5, 0.2, 0.8, // group 0: prune 0.1, 0.2
            0.9_f32, 0.3, 0.7, 0.1,
        ]; // group 1: prune 0.1, 0.3
        let mask = p.compute_grouped_mask(&w, 4).unwrap();
        assert_eq!(mask.len(), 8);
        assert_abs_diff_eq!(mask.sparsity(), 0.5, epsilon = 0.01);
    }

    #[test]
    fn grouped_mask_mismatch_error() {
        let p = MagnitudePruner::new(0.5, MagnitudeNorm::L1);
        let w = vec![0.5_f32; 5];
        assert!(matches!(
            p.compute_grouped_mask(&w, 4),
            Err(QuantError::GroupSizeMismatch { .. })
        ));
    }
}
