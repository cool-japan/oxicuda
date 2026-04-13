//! Problem-size interpolation for autotuning.
//!
//! When the exact problem size has not been benchmarked, this module
//! predicts the optimal kernel configuration by interpolating from
//! previously observed results.  Two strategies are provided:
//!
//! - **Nearest-neighbor** — return the config from the closest
//!   observed problem size.
//! - **Inverse-distance-weighted (IDW)** — weight nearby observations
//!   by inverse distance and pick the config with the highest weighted
//!   score.
//!
//! The distance metric operates in **log-space** (natural log of each
//! dimension) so that doubling a dimension is treated the same
//! regardless of absolute size:
//!
//! ```text
//! d(a, b) = sqrt( (ln(a.m) - ln(b.m))^2
//!               + (ln(a.n) - ln(b.n))^2
//!               + (ln(a.k) - ln(b.k))^2 )
//! ```
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::interpolation::{ProblemSize, SizeInterpolator};
//! use oxicuda_autotune::Config;
//!
//! let mut interp = SizeInterpolator::new();
//! interp.add_observation(
//!     ProblemSize { m: 1024, n: 1024, k: 1024 },
//!     Config::new().with_tile_m(128),
//!     1500.0,
//! );
//! interp.add_observation(
//!     ProblemSize { m: 512, n: 512, k: 512 },
//!     Config::new().with_tile_m(64),
//!     1200.0,
//! );
//!
//! let target = ProblemSize { m: 768, n: 768, k: 768 };
//! let result = interp.predict(&target);
//! assert!(result.is_some());
//! ```

use crate::config::Config;

// ---------------------------------------------------------------------------
// ProblemSize
// ---------------------------------------------------------------------------

/// Describes the dimensions of a matrix problem (GEMM-style).
///
/// All dimensions must be > 0 for meaningful distance computation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProblemSize {
    /// Number of rows of the output matrix.
    pub m: u32,
    /// Number of columns of the output matrix.
    pub n: u32,
    /// Inner (reduction) dimension.
    pub k: u32,
}

impl ProblemSize {
    /// Compute the normalized L2 distance in log-space to another size.
    ///
    /// Returns `f64::INFINITY` if any dimension in either size is zero.
    #[must_use]
    pub fn log_distance(&self, other: &ProblemSize) -> f64 {
        if self.m == 0 || self.n == 0 || self.k == 0 || other.m == 0 || other.n == 0 || other.k == 0
        {
            return f64::INFINITY;
        }

        let dm = (f64::from(self.m)).ln() - (f64::from(other.m)).ln();
        let dn = (f64::from(self.n)).ln() - (f64::from(other.n)).ln();
        let dk = (f64::from(self.k)).ln() - (f64::from(other.k)).ln();

        (dm * dm + dn * dn + dk * dk).sqrt()
    }
}

impl std::fmt::Display for ProblemSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}x{}", self.m, self.n, self.k)
    }
}

// ---------------------------------------------------------------------------
// SizeInterpolator
// ---------------------------------------------------------------------------

/// Predicts optimal kernel configurations for unseen problem sizes by
/// interpolating from a database of previous observations.
#[derive(Debug, Clone)]
pub struct SizeInterpolator {
    /// Observed (problem-size, config, performance-metric) triples.
    observations: Vec<(ProblemSize, Config, f64)>,
}

impl SizeInterpolator {
    /// Create a new, empty interpolator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
        }
    }

    /// Add an observation.
    ///
    /// `perf` is a performance metric where **higher is better** (e.g.
    /// GFLOPS).
    pub fn add_observation(&mut self, size: ProblemSize, config: Config, perf: f64) {
        self.observations.push((size, config, perf));
    }

    /// Nearest-neighbor prediction.
    ///
    /// Returns the config and performance of the observation whose
    /// problem size is closest (in log-space L2 distance) to `target`.
    ///
    /// Returns `None` if there are no observations.
    #[must_use]
    pub fn predict(&self, target: &ProblemSize) -> Option<(Config, f64)> {
        if self.observations.is_empty() {
            return None;
        }

        self.observations
            .iter()
            .min_by(|(a, _, _), (b, _, _)| {
                let da = target.log_distance(a);
                let db = target.log_distance(b);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, cfg, perf)| (cfg.clone(), *perf))
    }

    /// Inverse-distance-weighted (IDW) prediction.
    ///
    /// Weights each observation by `1 / d(target, obs)^2`, then returns
    /// the config whose weighted performance score is highest.  When an
    /// observation has distance = 0 (exact match), it is returned
    /// immediately.
    ///
    /// Returns `None` if there are no observations.
    #[must_use]
    pub fn predict_weighted(&self, target: &ProblemSize) -> Option<(Config, f64)> {
        if self.observations.is_empty() {
            return None;
        }

        // Compute distances; check for exact match.
        let distances: Vec<f64> = self
            .observations
            .iter()
            .map(|(size, _, _)| target.log_distance(size))
            .collect();

        // Exact match shortcut.
        for (i, &d) in distances.iter().enumerate() {
            if d < 1e-12 {
                let (_, cfg, perf) = &self.observations[i];
                return Some((cfg.clone(), *perf));
            }
        }

        // Group by unique config and accumulate weighted performance.
        // Since Config implements Hash+Eq, we can use it as a map key.
        let mut config_scores: std::collections::HashMap<&Config, (f64, f64)> =
            std::collections::HashMap::new();

        for (i, (_, cfg, perf)) in self.observations.iter().enumerate() {
            let d = distances[i];
            if !d.is_finite() || d <= 0.0 {
                continue;
            }
            let weight = 1.0 / (d * d);
            let entry = config_scores.entry(cfg).or_insert((0.0, 0.0));
            entry.0 += weight * perf;
            entry.1 += weight;
        }

        config_scores
            .into_iter()
            .filter(|(_, (_, w_sum))| *w_sum > 0.0)
            .max_by(|(_, (score_a, _)), (_, (score_b, _))| {
                // Use total weighted score (not average) so that configs
                // with strong nearby support naturally dominate over
                // distant observations with fewer data points.
                score_a
                    .partial_cmp(score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(cfg, (score, w_sum))| (cfg.clone(), score / w_sum))
    }

    /// Return the `k` nearest observations to `target`, ordered by
    /// ascending distance.
    ///
    /// If there are fewer than `k` observations, all are returned.
    #[must_use]
    pub fn nearest_observations(
        &self,
        target: &ProblemSize,
        k: usize,
    ) -> Vec<&(ProblemSize, Config, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .observations
            .iter()
            .enumerate()
            .map(|(i, (size, _, _))| (i, target.log_distance(size)))
            .collect();

        indexed.sort_by(|(_, da), (_, db)| da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal));

        indexed
            .into_iter()
            .take(k)
            .map(|(i, _)| &self.observations[i])
            .collect()
    }

    /// Return the total number of observations.
    #[must_use]
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Return a reference to all observations.
    #[must_use]
    pub fn observations(&self) -> &[(ProblemSize, Config, f64)] {
        &self.observations
    }
}

impl Default for SizeInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn size(m: u32, n: u32, k: u32) -> ProblemSize {
        ProblemSize { m, n, k }
    }

    fn cfg_tile_m(v: u32) -> Config {
        Config::new().with_tile_m(v)
    }

    // -- ProblemSize --------------------------------------------------------

    #[test]
    fn log_distance_same_point_is_zero() {
        let a = size(1024, 1024, 1024);
        let d = a.log_distance(&a);
        assert!(d.abs() < 1e-12);
    }

    #[test]
    fn log_distance_symmetric() {
        let a = size(512, 1024, 256);
        let b = size(1024, 512, 512);
        let d1 = a.log_distance(&b);
        let d2 = b.log_distance(&a);
        assert!((d1 - d2).abs() < 1e-12);
    }

    #[test]
    fn log_distance_zero_dimension_is_infinity() {
        let a = size(0, 1024, 1024);
        let b = size(1024, 1024, 1024);
        assert!(a.log_distance(&b).is_infinite());
    }

    #[test]
    fn log_distance_doubling_constant() {
        // Doubling m=512->1024 should give same distance as m=1024->2048.
        let a = size(512, 100, 100);
        let b = size(1024, 100, 100);
        let c = size(1024, 100, 100);
        let d = size(2048, 100, 100);
        let d1 = a.log_distance(&b);
        let d2 = c.log_distance(&d);
        assert!((d1 - d2).abs() < 1e-12);
    }

    #[test]
    fn problem_size_display() {
        let s = size(1024, 512, 256);
        assert_eq!(format!("{s}"), "1024x512x256");
    }

    // -- SizeInterpolator: predict (nearest-neighbor) -----------------------

    #[test]
    fn predict_empty_returns_none() {
        let interp = SizeInterpolator::new();
        assert!(interp.predict(&size(1024, 1024, 1024)).is_none());
    }

    #[test]
    fn predict_single_observation() {
        let mut interp = SizeInterpolator::new();
        interp.add_observation(size(1024, 1024, 1024), cfg_tile_m(128), 1500.0);

        let (cfg, perf) = interp
            .predict(&size(768, 768, 768))
            .expect("should predict");
        assert_eq!(cfg.tile_m, 128);
        assert!((perf - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn predict_picks_nearest() {
        let mut interp = SizeInterpolator::new();
        interp.add_observation(size(256, 256, 256), cfg_tile_m(32), 400.0);
        interp.add_observation(size(1024, 1024, 1024), cfg_tile_m(128), 1500.0);
        interp.add_observation(size(4096, 4096, 4096), cfg_tile_m(256), 2000.0);

        // 768 is closer to 1024 than to 256 or 4096 in log-space.
        let (cfg, _) = interp
            .predict(&size(768, 768, 768))
            .expect("should predict");
        assert_eq!(cfg.tile_m, 128);
    }

    // -- SizeInterpolator: predict_weighted (IDW) ---------------------------

    #[test]
    fn predict_weighted_empty_returns_none() {
        let interp = SizeInterpolator::new();
        assert!(interp.predict_weighted(&size(1024, 1024, 1024)).is_none());
    }

    #[test]
    fn predict_weighted_exact_match() {
        let mut interp = SizeInterpolator::new();
        interp.add_observation(size(1024, 1024, 1024), cfg_tile_m(128), 1500.0);
        interp.add_observation(size(512, 512, 512), cfg_tile_m(64), 1000.0);

        let (cfg, perf) = interp
            .predict_weighted(&size(1024, 1024, 1024))
            .expect("should predict");
        assert_eq!(cfg.tile_m, 128);
        assert!((perf - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn predict_weighted_prefers_nearby_config() {
        let mut interp = SizeInterpolator::new();
        // Close observation with moderate perf.
        interp.add_observation(size(900, 900, 900), cfg_tile_m(128), 1500.0);
        // Another close observation with same config, reinforcing it.
        interp.add_observation(size(1100, 1100, 1100), cfg_tile_m(128), 1400.0);
        // Far observation with slightly higher perf — but distance penalises it.
        interp.add_observation(size(64, 64, 64), cfg_tile_m(32), 1600.0);

        let target = size(1024, 1024, 1024);
        let (cfg, weighted_perf) = interp.predict_weighted(&target).expect("should predict");
        // Config 128 has two nearby observations with high weight;
        // config 32 has one far observation with negligible weight.
        assert_eq!(cfg.tile_m, 128);
        // Weighted average should be between the two nearby observations.
        assert!(weighted_perf > 1300.0 && weighted_perf < 1600.0);
    }

    // -- nearest_observations -----------------------------------------------

    #[test]
    fn nearest_observations_empty() {
        let interp = SizeInterpolator::new();
        let result = interp.nearest_observations(&size(1024, 1024, 1024), 3);
        assert!(result.is_empty());
    }

    #[test]
    fn nearest_observations_ordered() {
        let mut interp = SizeInterpolator::new();
        interp.add_observation(size(4096, 4096, 4096), cfg_tile_m(256), 2000.0);
        interp.add_observation(size(2048, 2048, 2048), cfg_tile_m(128), 1500.0);
        interp.add_observation(size(256, 256, 256), cfg_tile_m(32), 400.0);

        let target = size(512, 512, 512);
        let nearest = interp.nearest_observations(&target, 2);
        assert_eq!(nearest.len(), 2);
        // 256 is closer to 512 than 2048 in log-space (ln ratio 0.69 vs 1.39).
        assert_eq!(nearest[0].0.m, 256);
        assert_eq!(nearest[1].0.m, 2048);
    }

    #[test]
    fn nearest_observations_k_larger_than_count() {
        let mut interp = SizeInterpolator::new();
        interp.add_observation(size(512, 512, 512), cfg_tile_m(64), 800.0);
        let nearest = interp.nearest_observations(&size(1024, 1024, 1024), 10);
        assert_eq!(nearest.len(), 1);
    }

    // -- Misc ---------------------------------------------------------------

    #[test]
    fn default_impl() {
        let interp = SizeInterpolator::default();
        assert_eq!(interp.num_observations(), 0);
    }

    #[test]
    fn observations_accessor() {
        let mut interp = SizeInterpolator::new();
        interp.add_observation(size(1024, 1024, 1024), Config::new(), 1000.0);
        interp.add_observation(size(512, 512, 512), Config::new(), 800.0);
        assert_eq!(interp.observations().len(), 2);
    }
}
