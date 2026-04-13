//! # Online Welford Mean/Variance Tracker
//!
//! Welford's online algorithm computes the running mean and variance in a single
//! pass with O(1) memory:
//!
//! ```text
//! n   += 1
//! δ    = x - mean
//! mean += δ / n
//! δ₂   = x - mean  (new mean)
//! M2  += δ * δ₂
//! var  = M2 / (n - 1)  (sample variance)
//! ```
//!
//! Supports scalar and vector (per-dimension) tracking.

use crate::error::{RlError, RlResult};

// ─── RunningStats ─────────────────────────────────────────────────────────────

/// Per-dimension running mean and variance tracker using Welford's algorithm.
#[derive(Debug, Clone)]
pub struct RunningStats {
    dim: usize,
    /// Running mean per dimension.
    mean: Vec<f64>,
    /// Running M2 (sum of squared deviations) per dimension.
    m2: Vec<f64>,
    /// Sample count.
    count: u64,
}

impl RunningStats {
    /// Create a new tracker for vectors of dimension `dim`.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "dim must be > 0");
        Self {
            dim,
            mean: vec![0.0_f64; dim],
            m2: vec![0.0_f64; dim],
            count: 0,
        }
    }

    /// Dimension (number of tracked features).
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of samples seen so far.
    #[must_use]
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Per-dimension mean.
    #[must_use]
    pub fn mean_f32(&self) -> Vec<f32> {
        self.mean.iter().map(|&m| m as f32).collect()
    }

    /// Per-dimension sample standard deviation (returns `[1.0; dim]` until `count ≥ 2`).
    #[must_use]
    pub fn std_f32(&self) -> Vec<f32> {
        if self.count < 2 {
            return vec![1.0_f32; self.dim];
        }
        let n = (self.count - 1) as f64;
        self.m2
            .iter()
            .map(|&m2| ((m2 / n).max(1e-8)).sqrt() as f32)
            .collect()
    }

    /// Per-dimension variance.
    #[must_use]
    pub fn var_f32(&self) -> Vec<f32> {
        if self.count < 2 {
            return vec![1.0_f32; self.dim];
        }
        let n = (self.count - 1) as f64;
        self.m2.iter().map(|&m2| (m2 / n) as f32).collect()
    }

    /// Update statistics with a new observation vector.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `obs.len() != dim`.
    pub fn update(&mut self, obs: &[f32]) -> RlResult<()> {
        if obs.len() != self.dim {
            return Err(RlError::DimensionMismatch {
                expected: self.dim,
                got: obs.len(),
            });
        }
        self.count += 1;
        let n = self.count as f64;
        for (i, &x) in obs.iter().enumerate() {
            let x64 = x as f64;
            let delta = x64 - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = x64 - self.mean[i];
            self.m2[i] += delta * delta2;
        }
        Ok(())
    }

    /// Update statistics with a batch of observations `[B × dim]`.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `batch.len() % dim != 0`.
    pub fn update_batch(&mut self, batch: &[f32]) -> RlResult<()> {
        if batch.len() % self.dim != 0 {
            return Err(RlError::DimensionMismatch {
                expected: self.dim,
                got: batch.len(),
            });
        }
        for chunk in batch.chunks_exact(self.dim) {
            self.update(chunk)?;
        }
        Ok(())
    }

    /// Normalise a single observation: `(obs - mean) / (std + eps)`.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `obs.len() != dim`.
    pub fn normalise(&self, obs: &[f32]) -> RlResult<Vec<f32>> {
        if obs.len() != self.dim {
            return Err(RlError::DimensionMismatch {
                expected: self.dim,
                got: obs.len(),
            });
        }
        let std = self.std_f32();
        let mean = self.mean_f32();
        Ok(obs
            .iter()
            .zip(mean.iter())
            .zip(std.iter())
            .map(|((&x, &m), &s)| (x - m) / (s + 1e-8))
            .collect())
    }

    /// Reset all statistics to zero.
    pub fn reset(&mut self) {
        self.mean.iter_mut().for_each(|v| *v = 0.0);
        self.m2.iter_mut().for_each(|v| *v = 0.0);
        self.count = 0;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_count_zero() {
        let rs = RunningStats::new(3);
        assert_eq!(rs.count(), 0);
    }

    #[test]
    fn single_update_count_one() {
        let mut rs = RunningStats::new(2);
        rs.update(&[1.0, 2.0]).unwrap();
        assert_eq!(rs.count(), 1);
    }

    #[test]
    fn mean_converges_to_true_mean() {
        let mut rs = RunningStats::new(1);
        for _ in 0..1000 {
            rs.update(&[3.0]).unwrap();
        }
        let mean = rs.mean_f32()[0];
        assert!((mean - 3.0).abs() < 0.01, "mean={mean}");
    }

    #[test]
    fn std_converges_to_true_std() {
        // Values alternating ±1 → mean=0, std≈1
        let mut rs = RunningStats::new(1);
        for i in 0..2000 {
            let v = if i % 2 == 0 { 1.0 } else { -1.0 };
            rs.update(&[v]).unwrap();
        }
        let std = rs.std_f32()[0];
        assert!((std - 1.0).abs() < 0.05, "std={std}");
    }

    #[test]
    fn normalise_close_to_zero_mean() {
        let mut rs = RunningStats::new(1);
        for i in 0..100 {
            rs.update(&[i as f32]).unwrap();
        }
        let norm = rs.normalise(&[50.0]).unwrap(); // near mean
        assert!(
            norm[0].abs() < 0.5,
            "normalised mean should be near 0, got {}",
            norm[0]
        );
    }

    #[test]
    fn normalise_dimension_error() {
        let rs = RunningStats::new(3);
        assert!(rs.normalise(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn update_batch_increments_count() {
        let mut rs = RunningStats::new(2);
        let batch = vec![1.0_f32; 10]; // 5 observations of dim 2
        rs.update_batch(&batch).unwrap();
        assert_eq!(rs.count(), 5);
    }

    #[test]
    fn reset_zeroes_stats() {
        let mut rs = RunningStats::new(2);
        rs.update(&[3.0, 4.0]).unwrap();
        rs.reset();
        assert_eq!(rs.count(), 0);
        let mean = rs.mean_f32();
        assert!(mean.iter().all(|&m| m.abs() < 1e-9));
    }

    #[test]
    fn std_default_before_two_samples() {
        let mut rs = RunningStats::new(2);
        rs.update(&[1.0, 2.0]).unwrap();
        let std = rs.std_f32();
        // With count < 2, returns [1, 1]
        assert_eq!(std, vec![1.0, 1.0]);
    }
}
