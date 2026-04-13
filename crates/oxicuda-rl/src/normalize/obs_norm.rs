//! # Observation Normalizer
//!
//! Wraps [`RunningStats`] to provide a stateful normalizer that can be
//! toggled on/off and clipped to prevent extreme values.

use crate::error::{RlError, RlResult};
use crate::normalize::running_stats::RunningStats;

/// Observation normalizer with running statistics.
#[derive(Debug, Clone)]
pub struct ObservationNormalizer {
    stats: RunningStats,
    /// Whether normalisation is active.
    enabled: bool,
    /// Clip range after normalisation (symmetric: `[-clip, clip]`).
    clip: f32,
    /// Whether to update statistics when processing observations.
    update_stats: bool,
}

impl ObservationNormalizer {
    /// Create a normalizer for observations of dimension `obs_dim`.
    ///
    /// Default: enabled, `clip = 5.0`, `update_stats = true`.
    #[must_use]
    pub fn new(obs_dim: usize) -> Self {
        Self {
            stats: RunningStats::new(obs_dim),
            enabled: true,
            clip: 5.0,
            update_stats: true,
        }
    }

    /// Disable statistics update (useful at evaluation time).
    #[must_use]
    pub fn with_no_update(mut self) -> Self {
        self.update_stats = false;
        self
    }

    /// Set the clip range.
    #[must_use]
    pub fn with_clip(mut self, clip: f32) -> Self {
        self.clip = clip;
        self
    }

    /// Disable normalisation (pass-through mode).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable normalisation.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Number of observations seen.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.stats.count()
    }

    /// Process a batch of observations: optionally update stats, normalise, clip.
    ///
    /// * `batch` — `[B × obs_dim]` flat slice.
    ///
    /// Returns a normalised `[B × obs_dim]` vector.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `batch.len() % obs_dim != 0`.
    pub fn process(&mut self, batch: &[f32]) -> RlResult<Vec<f32>> {
        let obs_dim = self.stats.dim();
        if batch.len() % obs_dim != 0 {
            return Err(RlError::DimensionMismatch {
                expected: obs_dim,
                got: batch.len() % obs_dim,
            });
        }
        if !self.enabled {
            return Ok(batch.to_vec());
        }
        if self.update_stats {
            self.stats.update_batch(batch)?;
        }
        let mut out = Vec::with_capacity(batch.len());
        for chunk in batch.chunks_exact(obs_dim) {
            let norm = self.stats.normalise(chunk)?;
            for v in norm {
                out.push(v.clamp(-self.clip, self.clip));
            }
        }
        Ok(out)
    }

    /// Process a single observation.
    ///
    /// # Errors
    ///
    /// * [`RlError::DimensionMismatch`] if `obs.len() != obs_dim`.
    pub fn process_one(&mut self, obs: &[f32]) -> RlResult<Vec<f32>> {
        let obs_dim = self.stats.dim();
        if obs.len() != obs_dim {
            return Err(RlError::DimensionMismatch {
                expected: obs_dim,
                got: obs.len(),
            });
        }
        if !self.enabled {
            return Ok(obs.to_vec());
        }
        if self.update_stats {
            self.stats.update(obs)?;
        }
        let norm = self.stats.normalise(obs)?;
        Ok(norm
            .into_iter()
            .map(|v| v.clamp(-self.clip, self.clip))
            .collect())
    }

    /// Access the underlying running statistics.
    #[must_use]
    pub fn stats(&self) -> &RunningStats {
        &self.stats
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_passthrough() {
        let mut norm = ObservationNormalizer::new(3);
        norm.disable();
        let obs = vec![1.0, 2.0, 3.0];
        let out = norm.process_one(&obs).unwrap();
        assert_eq!(out, obs);
    }

    #[test]
    fn normalise_clips_extreme() {
        let mut norm = ObservationNormalizer::new(1).with_clip(2.0);
        // Seed with many values so std is reasonable
        for _ in 0..200 {
            norm.process_one(&[0.0]).unwrap();
        }
        // Now feed an extreme value
        let out = norm.process_one(&[1000.0]).unwrap();
        assert!(out[0] <= 2.0 + 1e-3, "clipped={}", out[0]);
    }

    #[test]
    fn count_increments() {
        let mut norm = ObservationNormalizer::new(2);
        for _ in 0..10 {
            norm.process_one(&[1.0, 2.0]).unwrap();
        }
        assert_eq!(norm.count(), 10);
    }

    #[test]
    fn batch_same_as_sequential() {
        let mut norm_seq = ObservationNormalizer::new(2);
        let mut norm_bat = ObservationNormalizer::new(2);
        let obs = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 × dim 2
        for chunk in obs.chunks_exact(2) {
            norm_seq.process_one(chunk).unwrap();
        }
        norm_bat.process(&obs).unwrap();
        let std_seq = norm_seq.stats().std_f32();
        let std_bat = norm_bat.stats().std_f32();
        for (a, b) in std_seq.iter().zip(std_bat.iter()) {
            assert!((a - b).abs() < 1e-4, "seq_std={a} bat_std={b}");
        }
    }

    #[test]
    fn dimension_mismatch_error() {
        let mut norm = ObservationNormalizer::new(4);
        assert!(norm.process_one(&[1.0, 2.0]).is_err());
    }
}
