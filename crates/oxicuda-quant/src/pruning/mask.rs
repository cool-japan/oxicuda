//! # Sparse Mask
//!
//! A `SparseMask` is a boolean weight mask where `false` = pruned, `true` = active.
//! It is produced by a pruner and applied to weight tensors to zero out pruned entries.

/// Boolean pruning mask over a weight tensor.
///
/// `true`  → weight is active (kept).
/// `false` → weight is pruned (zeroed).
#[derive(Debug, Clone)]
pub struct SparseMask {
    /// Element-wise mask (same length as the weight tensor).
    pub mask: Vec<bool>,
}

impl SparseMask {
    /// Create a mask with all weights active (unpruned).
    #[must_use]
    pub fn all_active(n: usize) -> Self {
        Self {
            mask: vec![true; n],
        }
    }

    /// Create a mask with all weights pruned.
    #[must_use]
    pub fn all_pruned(n: usize) -> Self {
        Self {
            mask: vec![false; n],
        }
    }

    /// Number of elements in the mask.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    /// Returns `true` if the mask covers an empty weight tensor.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// Number of active (non-pruned) weights.
    #[must_use]
    pub fn count_active(&self) -> usize {
        self.mask.iter().filter(|&&m| m).count()
    }

    /// Number of pruned weights.
    #[must_use]
    pub fn count_pruned(&self) -> usize {
        self.mask.iter().filter(|&&m| !m).count()
    }

    /// Fraction of weights that are pruned ∈ [0, 1].
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        if self.mask.is_empty() {
            return 0.0;
        }
        self.count_pruned() as f32 / self.mask.len() as f32
    }

    /// Apply the mask to a weight slice: pruned weights become 0.
    ///
    /// Returns a new `Vec<f32>` of the same length.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != self.len()`.
    #[must_use]
    pub fn apply(&self, weights: &[f32]) -> Vec<f32> {
        assert_eq!(
            weights.len(),
            self.mask.len(),
            "mask/weight length mismatch"
        );
        weights
            .iter()
            .zip(self.mask.iter())
            .map(|(&w, &m)| if m { w } else { 0.0 })
            .collect()
    }

    /// Apply the mask in-place: pruned weights become 0.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != self.len()`.
    pub fn apply_in_place(&self, weights: &mut [f32]) {
        assert_eq!(
            weights.len(),
            self.mask.len(),
            "mask/weight length mismatch"
        );
        for (w, &m) in weights.iter_mut().zip(self.mask.iter()) {
            if !m {
                *w = 0.0;
            }
        }
    }

    /// Combine two masks with logical AND (both must be active to stay active).
    ///
    /// # Panics
    ///
    /// Panics if the masks have different lengths.
    #[must_use]
    pub fn and(&self, other: &Self) -> Self {
        assert_eq!(self.mask.len(), other.mask.len(), "mask length mismatch");
        let mask = self
            .mask
            .iter()
            .zip(other.mask.iter())
            .map(|(&a, &b)| a && b)
            .collect();
        Self { mask }
    }

    /// Combine two masks with logical OR (at least one active → active).
    ///
    /// # Panics
    ///
    /// Panics if the masks have different lengths.
    #[must_use]
    pub fn or(&self, other: &Self) -> Self {
        assert_eq!(self.mask.len(), other.mask.len(), "mask length mismatch");
        let mask = self
            .mask
            .iter()
            .zip(other.mask.iter())
            .map(|(&a, &b)| a || b)
            .collect();
        Self { mask }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn all_active_sparsity_zero() {
        let m = SparseMask::all_active(10);
        assert_abs_diff_eq!(m.sparsity(), 0.0, epsilon = 1e-6);
        assert_eq!(m.count_active(), 10);
        assert_eq!(m.count_pruned(), 0);
    }

    #[test]
    fn all_pruned_sparsity_one() {
        let m = SparseMask::all_pruned(8);
        assert_abs_diff_eq!(m.sparsity(), 1.0, epsilon = 1e-6);
        assert_eq!(m.count_active(), 0);
    }

    #[test]
    fn apply_zeroes_pruned() {
        let m = SparseMask {
            mask: vec![true, false, true, false],
        };
        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = m.apply(&w);
        assert_abs_diff_eq!(out[0], 1.0, epsilon = 1e-7);
        assert_abs_diff_eq!(out[1], 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(out[2], 3.0, epsilon = 1e-7);
        assert_abs_diff_eq!(out[3], 0.0, epsilon = 1e-7);
    }

    #[test]
    fn apply_in_place_modifies_weights() {
        let m = SparseMask {
            mask: vec![true, false, true],
        };
        let mut w = vec![5.0_f32, 9.0, 3.0];
        m.apply_in_place(&mut w);
        assert_abs_diff_eq!(w[1], 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(w[0], 5.0, epsilon = 1e-7);
    }

    #[test]
    fn sparsity_partial() {
        let m = SparseMask {
            mask: vec![true, false, true, false, false],
        };
        assert_abs_diff_eq!(m.sparsity(), 3.0 / 5.0, epsilon = 1e-6);
    }

    #[test]
    fn and_mask() {
        let a = SparseMask {
            mask: vec![true, true, false],
        };
        let b = SparseMask {
            mask: vec![true, false, false],
        };
        let c = a.and(&b);
        assert_eq!(c.mask, vec![true, false, false]);
    }

    #[test]
    fn or_mask() {
        let a = SparseMask {
            mask: vec![true, false, false],
        };
        let b = SparseMask {
            mask: vec![false, true, false],
        };
        let c = a.or(&b);
        assert_eq!(c.mask, vec![true, true, false]);
    }
}
