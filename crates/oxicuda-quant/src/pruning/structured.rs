//! # Structured Pruning
//!
//! Removes entire structural units — channels, filters, or attention heads —
//! rather than individual weights.  Structured pruning produces weight matrices
//! with rows or columns of zeros that can be physically removed, yielding
//! real hardware speedups (unlike unstructured sparsity which requires
//! special sparse kernels).
//!
//! ## Granularities
//!
//! | Granularity | Unit removed        | Layout assumption                   |
//! |-------------|---------------------|-------------------------------------|
//! | `Channel`   | Output channel      | `[n_out, n_in]` row-major           |
//! | `Filter`    | Convolutional filter| `[n_filters, filter_size]` flat     |
//! | `Head`      | Attention head      | `[n_heads × head_dim, ...]`         |

use crate::error::{QuantError, QuantResult};
use crate::pruning::mask::SparseMask;

// ─── Granularity ─────────────────────────────────────────────────────────────

/// Structural unit to remove during pruning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneGranularity {
    /// Prune entire output channels (rows) of a weight matrix `[n_out, n_in]`.
    Channel {
        /// Number of output channels (rows of the weight matrix).
        n_out: usize,
        /// Number of input channels (columns of the weight matrix).
        n_in: usize,
    },
    /// Prune entire convolutional filters, each of length `filter_size`.
    Filter {
        /// Number of filters (= output channels for a conv layer).
        n_filters: usize,
        /// Flattened size of each filter (in_channels × kH × kW).
        filter_size: usize,
    },
    /// Prune entire attention heads in a projection matrix.
    Head {
        /// Number of attention heads.
        n_heads: usize,
        /// Dimension per head.
        head_dim: usize,
    },
}

// ─── StructuredPruner ────────────────────────────────────────────────────────

/// Removes structural units based on L2 norm importance.
///
/// Each unit (channel / filter / head) receives a scalar importance score
/// equal to the L2 norm of its weights.  The bottom `target_sparsity` fraction
/// of units are pruned.
#[derive(Debug, Clone)]
pub struct StructuredPruner {
    /// Fraction of structural units to prune ∈ [0, 1).
    pub target_sparsity: f32,
    /// Structural granularity.
    pub granularity: PruneGranularity,
}

impl StructuredPruner {
    /// Create a new structured pruner.
    ///
    /// # Panics
    ///
    /// Panics if `target_sparsity` is not in `[0, 1)`.
    #[must_use]
    pub fn new(target_sparsity: f32, granularity: PruneGranularity) -> Self {
        assert!(
            (0.0..1.0).contains(&target_sparsity),
            "target_sparsity must be in [0, 1), got {target_sparsity}"
        );
        Self {
            target_sparsity,
            granularity,
        }
    }

    /// Compute the pruning mask for `weights`.
    ///
    /// Returns a flat mask the same length as `weights`.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`]        — `weights` is empty.
    /// * [`QuantError::DimensionMismatch`] — `weights.len()` is inconsistent with the granularity.
    /// * [`QuantError::AllZeroPruning`]    — all units would be pruned.
    pub fn compute_mask(&self, weights: &[f32]) -> QuantResult<SparseMask> {
        if weights.is_empty() {
            return Err(QuantError::EmptyInput("StructuredPruner::compute_mask"));
        }
        match self.granularity {
            PruneGranularity::Channel { n_out, n_in } => self.channel_mask(weights, n_out, n_in),
            PruneGranularity::Filter {
                n_filters,
                filter_size,
            } => self.channel_mask(weights, n_filters, filter_size),
            PruneGranularity::Head { n_heads, head_dim } => {
                self.channel_mask(weights, n_heads, head_dim)
            }
        }
    }

    /// Apply structured pruning in-place.
    ///
    /// Returns the mask applied.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`compute_mask`](Self::compute_mask).
    pub fn prune(&self, weights: &mut [f32]) -> QuantResult<SparseMask> {
        let mask = self.compute_mask(weights)?;
        mask.apply_in_place(weights);
        Ok(mask)
    }

    /// Compute per-unit L2 norms for a matrix `[n_units, unit_size]`.
    ///
    /// Returns a `Vec<f32>` of length `n_units`.
    #[must_use]
    pub fn unit_l2_norms(weights: &[f32], n_units: usize, unit_size: usize) -> Vec<f32> {
        (0..n_units)
            .map(|u| {
                let base = u * unit_size;
                weights[base..base + unit_size]
                    .iter()
                    .map(|&w| w * w)
                    .sum::<f32>()
                    .sqrt()
            })
            .collect()
    }

    /// Return the indices of units to prune, sorted ascending by L2 norm.
    ///
    /// Prunes the `n_prune` units with the smallest L2 norm.
    #[must_use]
    pub fn pruned_unit_indices(norms: &[f32], n_prune: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..norms.len()).collect();
        idx.sort_unstable_by(|&a, &b| {
            norms[a]
                .partial_cmp(&norms[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx[..n_prune].to_vec()
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn channel_mask(
        &self,
        weights: &[f32],
        n_units: usize,
        unit_size: usize,
    ) -> QuantResult<SparseMask> {
        let expected = n_units * unit_size;
        if weights.len() != expected {
            return Err(QuantError::DimensionMismatch {
                expected,
                got: weights.len(),
            });
        }

        let n_prune = ((n_units as f32) * self.target_sparsity).ceil() as usize;
        if n_prune >= n_units {
            return Err(QuantError::AllZeroPruning {
                threshold: self.target_sparsity,
                n: n_units,
            });
        }

        let norms = Self::unit_l2_norms(weights, n_units, unit_size);
        let pruned = Self::pruned_unit_indices(&norms, n_prune);

        let mut mask = vec![true; weights.len()];
        for u in pruned {
            let base = u * unit_size;
            mask[base..base + unit_size].fill(false);
        }
        Ok(SparseMask { mask })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn channel_prune_50_percent() {
        // 4 output channels × 4 input channels = 16 weights.
        // Channels 0, 1 have small norm; channels 2, 3 have large norm.
        let n_out = 4;
        let n_in = 4;
        let mut w = vec![0.0_f32; n_out * n_in];
        // channel 2: all 1.0 → norm = 2.0
        for j in 0..n_in {
            w[2 * n_in + j] = 1.0;
        }
        // channel 3: all 2.0 → norm = 4.0
        for j in 0..n_in {
            w[3 * n_in + j] = 2.0;
        }

        let p = StructuredPruner::new(0.5, PruneGranularity::Channel { n_out, n_in });
        let mask = p.compute_mask(&w).unwrap();
        // Sparsity = 2 channels / 4 channels = 0.5 in channel count
        // = 8 weights / 16 weights = 0.5 sparsity
        assert_abs_diff_eq!(mask.sparsity(), 0.5, epsilon = 0.01);
        // Channels 0, 1 (near-zero norm) should be pruned.
        for j in 0..n_in {
            assert!(!mask.mask[j], "ch 0 elem {j} should be pruned");
            assert!(!mask.mask[n_in + j], "ch 1 elem {j} should be pruned");
            assert!(mask.mask[2 * n_in + j], "ch 2 elem {j} should be active");
            assert!(mask.mask[3 * n_in + j], "ch 3 elem {j} should be active");
        }
    }

    #[test]
    fn prune_in_place_zeroes_low_norm_units() {
        let n_out = 3;
        let n_in = 2;
        let p = StructuredPruner::new(0.33, PruneGranularity::Channel { n_out, n_in });
        let mut w = vec![
            0.01_f32, 0.01, // channel 0: near zero
            1.0, 1.0, // channel 1: large norm
            0.5, 0.5, // channel 2: medium norm
        ];
        let _mask = p.prune(&mut w).unwrap();
        // Channel 0 should be zeroed.
        assert_abs_diff_eq!(w[0], 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(w[1], 0.0, epsilon = 1e-7);
    }

    #[test]
    fn unit_l2_norms_correct() {
        let w = vec![3.0_f32, 4.0, 0.0, 1.0]; // [2 units, 2 each]
        let norms = StructuredPruner::unit_l2_norms(&w, 2, 2);
        assert_abs_diff_eq!(norms[0], 5.0, epsilon = 1e-5); // sqrt(9+16)=5
        assert_abs_diff_eq!(norms[1], 1.0, epsilon = 1e-5); // sqrt(0+1)=1
    }

    #[test]
    fn filter_pruning_behaves_as_channel() {
        let n_filters = 4;
        let filter_size = 9;
        let p = StructuredPruner::new(
            0.25,
            PruneGranularity::Filter {
                n_filters,
                filter_size,
            },
        );
        let mut w = vec![0.0_f32; n_filters * filter_size];
        // Give filter 2 large values.
        for k in 0..filter_size {
            w[2 * filter_size + k] = 1.0;
        }
        let mask = p.compute_mask(&w).unwrap();
        assert_eq!(mask.len(), n_filters * filter_size);
    }

    #[test]
    fn dimension_mismatch_error() {
        let p = StructuredPruner::new(0.5, PruneGranularity::Channel { n_out: 4, n_in: 4 });
        let w = vec![0.5_f32; 12]; // 3×4, not 4×4
        assert!(matches!(
            p.compute_mask(&w),
            Err(QuantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn empty_input_error() {
        let p = StructuredPruner::new(0.5, PruneGranularity::Channel { n_out: 4, n_in: 4 });
        assert!(matches!(
            p.compute_mask(&[]),
            Err(QuantError::EmptyInput(_))
        ));
    }

    #[test]
    fn all_zero_pruning_error() {
        let p = StructuredPruner {
            target_sparsity: 0.99,
            granularity: PruneGranularity::Channel { n_out: 2, n_in: 4 },
        };
        let w = vec![1.0_f32; 8];
        assert!(matches!(
            p.compute_mask(&w),
            Err(QuantError::AllZeroPruning { .. })
        ));
    }
}
