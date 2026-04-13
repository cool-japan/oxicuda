//! PTX template integration for autotuning.
//!
//! Provides direct [`SearchSpace`] generation from PTX template parameter
//! ranges and a [`TemplateAutotuner`] for recording and ranking
//! benchmark results by GFLOPS performance.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::ptx_integration::{gemm_search_space, TemplateAutotuner};
//! use oxicuda_autotune::Config;
//!
//! let space = gemm_search_space(
//!     &[64, 128],
//!     &[64, 128],
//!     &[16, 32],
//!     &[2, 3],
//!     &[128, 256],
//! );
//! assert!(space.total_configs() > 0);
//!
//! let mut tuner = TemplateAutotuner::new(space);
//! tuner.record_result(Config::new().with_tile_m(64), 1200.0);
//! assert!(tuner.best_config().is_some());
//! ```

use crate::config::Config;
use crate::search_space::SearchSpace;

// ---------------------------------------------------------------------------
// SearchSpace generators for common PTX templates
// ---------------------------------------------------------------------------

/// Generate a [`SearchSpace`] from GEMM template parameter ranges.
///
/// Each slice specifies the candidate values for one GEMM dimension.
/// Warp tile values default to powers of two from 16..=64 that do not
/// exceed the maximum block tile M or N.  Tensor Core usage is always
/// explored (both on and off).
///
/// # Panics
///
/// Does not panic; empty slices produce an empty search space.
#[must_use]
pub fn gemm_search_space(
    m_range: &[u32],
    n_range: &[u32],
    k_range: &[u32],
    stages: &[u32],
    threads: &[u32],
) -> SearchSpace {
    // Derive sensible warp tile candidates from the block tile ranges.
    let max_m = m_range.iter().copied().max().unwrap_or(128);
    let max_n = n_range.iter().copied().max().unwrap_or(128);

    let warp_m_values: Vec<u32> = [16, 32, 64]
        .iter()
        .copied()
        .filter(|&v| v <= max_m)
        .collect();
    let warp_n_values: Vec<u32> = [16, 32, 64]
        .iter()
        .copied()
        .filter(|&v| v <= max_n)
        .collect();

    SearchSpace {
        tile_m_values: m_range.to_vec(),
        tile_n_values: n_range.to_vec(),
        tile_k_values: k_range.to_vec(),
        warp_m_values: if warp_m_values.is_empty() {
            vec![16]
        } else {
            warp_m_values
        },
        warp_n_values: if warp_n_values.is_empty() {
            vec![16]
        } else {
            warp_n_values
        },
        stages_values: stages.to_vec(),
        use_tensor_core_values: vec![false, true],
        block_size_values: threads.to_vec(),
    }
}

/// Generate a [`SearchSpace`] from elementwise template parameters.
///
/// Elementwise kernels are characterised by thread count and work items
/// per thread.  Block tile dimensions are set to `threads * items_per_thread`
/// to reflect the 1-D tiling.  Reduction and warp-level dimensions are
/// pinned to 1 since they are irrelevant for elementwise work.
#[must_use]
pub fn elementwise_search_space(threads: &[u32], items_per_thread: &[u32]) -> SearchSpace {
    build_1d_search_space(threads, items_per_thread)
}

/// Generate a [`SearchSpace`] from reduction template parameters.
///
/// Reduction kernels share the same 1-D parameterisation as elementwise
/// kernels (thread count x items per thread).
#[must_use]
pub fn reduction_search_space(threads: &[u32], items_per_thread: &[u32]) -> SearchSpace {
    build_1d_search_space(threads, items_per_thread)
}

/// Generate a [`SearchSpace`] from scan (prefix sum) template parameters.
///
/// Scan kernels share the same 1-D parameterisation as elementwise and
/// reduction kernels.
#[must_use]
pub fn scan_search_space(threads: &[u32], items_per_thread: &[u32]) -> SearchSpace {
    build_1d_search_space(threads, items_per_thread)
}

/// Internal helper: builds a 1-D search space for elementwise / reduction
/// / scan kernels.  The `tile_m` dimension encodes the effective tile
/// size (`threads * items_per_thread`), while other GEMM-specific
/// dimensions are pinned to neutral values.
fn build_1d_search_space(threads: &[u32], items_per_thread: &[u32]) -> SearchSpace {
    // Combine threads * items_per_thread into tile_m candidates (dedup & sort).
    let mut tile_values: Vec<u32> = Vec::new();
    for &t in threads {
        for &ipt in items_per_thread {
            tile_values.push(t.saturating_mul(ipt));
        }
    }
    tile_values.sort_unstable();
    tile_values.dedup();

    // Store items_per_thread in tile_k (repurposed for 1-D kernels).
    let mut k_values = items_per_thread.to_vec();
    k_values.sort_unstable();
    k_values.dedup();

    SearchSpace {
        tile_m_values: tile_values,
        tile_n_values: vec![1],
        tile_k_values: k_values,
        warp_m_values: vec![32],
        warp_n_values: vec![1],
        stages_values: vec![1],
        use_tensor_core_values: vec![false],
        block_size_values: threads.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// TemplateAutotuner
// ---------------------------------------------------------------------------

/// Autotuner that records benchmark results for template configurations
/// and selects the best ones by GFLOPS.
///
/// This is a lightweight in-memory structure designed for interactive
/// tuning loops — it does *not* persist results to disk.  For persistent
/// storage, feed the best config into [`ResultDb`](crate::ResultDb).
#[derive(Debug, Clone)]
pub struct TemplateAutotuner {
    /// The search space being explored.
    search_space: SearchSpace,
    /// Recorded (configuration, GFLOPS) pairs.
    results: Vec<(Config, f64)>,
}

impl TemplateAutotuner {
    /// Create a new autotuner for the given search space.
    #[must_use]
    pub fn new(search_space: SearchSpace) -> Self {
        Self {
            search_space,
            results: Vec::new(),
        }
    }

    /// Create an autotuner with a default GEMM search space.
    ///
    /// Uses [`SearchSpace::gemm_default()`] which covers common tile
    /// sizes from Volta through Hopper.
    #[must_use]
    pub fn from_gemm_defaults() -> Self {
        Self::new(SearchSpace::gemm_default())
    }

    /// Record a benchmark result for the given configuration.
    ///
    /// `gflops` is the measured throughput in GFLOPS.  Higher is better.
    pub fn record_result(&mut self, config: Config, gflops: f64) {
        self.results.push((config, gflops));
    }

    /// Return the best configuration (highest GFLOPS) recorded so far.
    ///
    /// Returns `None` if no results have been recorded.
    #[must_use]
    pub fn best_config(&self) -> Option<(&Config, f64)> {
        self.results
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(cfg, gflops)| (cfg, *gflops))
    }

    /// Return the top `n` configurations by GFLOPS (descending).
    ///
    /// If fewer than `n` results have been recorded, all are returned.
    #[must_use]
    pub fn top_n(&self, n: usize) -> Vec<(&Config, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .results
            .iter()
            .enumerate()
            .map(|(i, (_, g))| (i, *g))
            .collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .into_iter()
            .take(n)
            .map(|(i, g)| (&self.results[i].0, g))
            .collect()
    }

    /// Return a reference to the underlying search space.
    #[must_use]
    pub fn search_space(&self) -> &SearchSpace {
        &self.search_space
    }

    /// Return the number of recorded results.
    #[must_use]
    pub fn num_results(&self) -> usize {
        self.results.len()
    }

    /// Return all recorded results as a slice.
    #[must_use]
    pub fn results(&self) -> &[(Config, f64)] {
        &self.results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- gemm_search_space --------------------------------------------------

    #[test]
    fn gemm_search_space_basic() {
        let space = gemm_search_space(&[64, 128], &[64, 128], &[16, 32], &[2, 3], &[128, 256]);
        assert_eq!(space.tile_m_values, vec![64, 128]);
        assert_eq!(space.tile_n_values, vec![64, 128]);
        assert_eq!(space.tile_k_values, vec![16, 32]);
        assert_eq!(space.stages_values, vec![2, 3]);
        assert_eq!(space.block_size_values, vec![128, 256]);
        assert!(space.total_configs() > 0);
    }

    #[test]
    fn gemm_search_space_warp_tiles_clipped() {
        // If max tile_m=32, warp candidates 64 should be excluded.
        let space = gemm_search_space(&[32], &[32], &[8], &[2], &[128]);
        assert_eq!(space.warp_m_values, vec![16, 32]);
        assert_eq!(space.warp_n_values, vec![16, 32]);
    }

    #[test]
    fn gemm_search_space_very_small_tile() {
        // tile_m max = 8 — only warp 16 would be > 8, so fallback to [16].
        let space = gemm_search_space(&[8], &[8], &[4], &[1], &[64]);
        // All warp candidates (16, 32, 64) > 8 => fallback to [16]
        assert_eq!(space.warp_m_values, vec![16]);
    }

    #[test]
    fn gemm_search_space_empty_ranges() {
        let space = gemm_search_space(&[], &[], &[], &[], &[]);
        assert_eq!(space.total_configs(), 0);
    }

    #[test]
    fn gemm_search_space_enumerate_and_prune() {
        let space = gemm_search_space(&[64, 128], &[64, 128], &[16], &[2], &[128, 256]);
        let all = space.enumerate();
        assert!(all.len() > 1);
        let pruned = space.prune(48 * 1024, 255, 4);
        assert!(pruned.len() <= all.len());
    }

    // -- elementwise_search_space -------------------------------------------

    #[test]
    fn elementwise_search_space_basic() {
        let space = elementwise_search_space(&[128, 256], &[4, 8]);
        assert!(space.total_configs() > 0);
        // tile_n pinned to 1 for 1-D kernels
        assert_eq!(space.tile_n_values, vec![1]);
        assert_eq!(space.stages_values, vec![1]);
        assert_eq!(space.use_tensor_core_values, vec![false]);
    }

    #[test]
    fn elementwise_search_space_tile_values_deduped() {
        // 128*4 = 512, 256*2 = 512 — should dedup
        let space = elementwise_search_space(&[128, 256], &[2, 4]);
        // Sorted unique: 256, 512, 1024
        assert!(space.tile_m_values.windows(2).all(|w| w[0] <= w[1]));
        let unique_count = {
            let mut v = space.tile_m_values.clone();
            v.dedup();
            v.len()
        };
        assert_eq!(unique_count, space.tile_m_values.len());
    }

    // -- reduction_search_space ---------------------------------------------

    #[test]
    fn reduction_search_space_basic() {
        let space = reduction_search_space(&[64, 128, 256], &[1, 2, 4]);
        assert!(space.total_configs() > 0);
        assert_eq!(space.tile_n_values, vec![1]);
    }

    // -- scan_search_space --------------------------------------------------

    #[test]
    fn scan_search_space_basic() {
        let space = scan_search_space(&[128, 256], &[2, 4, 8]);
        assert!(space.total_configs() > 0);
        assert_eq!(space.tile_n_values, vec![1]);
    }

    // -- TemplateAutotuner --------------------------------------------------

    #[test]
    fn template_autotuner_new_empty() {
        let tuner = TemplateAutotuner::new(SearchSpace::minimal());
        assert_eq!(tuner.num_results(), 0);
        assert!(tuner.best_config().is_none());
    }

    #[test]
    fn template_autotuner_from_gemm_defaults() {
        let tuner = TemplateAutotuner::from_gemm_defaults();
        assert_eq!(
            tuner.search_space().total_configs(),
            SearchSpace::gemm_default().total_configs()
        );
    }

    #[test]
    fn template_autotuner_record_and_best() {
        let mut tuner = TemplateAutotuner::new(SearchSpace::minimal());
        tuner.record_result(Config::new().with_tile_m(64), 1000.0);
        tuner.record_result(Config::new().with_tile_m(128), 1500.0);
        tuner.record_result(Config::new().with_tile_m(256), 1200.0);

        let (best, gflops) = tuner.best_config().expect("should have results");
        assert_eq!(best.tile_m, 128);
        assert!((gflops - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn template_autotuner_top_n() {
        let mut tuner = TemplateAutotuner::new(SearchSpace::minimal());
        tuner.record_result(Config::new().with_tile_m(32), 800.0);
        tuner.record_result(Config::new().with_tile_m(64), 1200.0);
        tuner.record_result(Config::new().with_tile_m(128), 1500.0);
        tuner.record_result(Config::new().with_tile_m(256), 1100.0);

        let top2 = tuner.top_n(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0.tile_m, 128); // highest
        assert_eq!(top2[1].0.tile_m, 64); // second
    }

    #[test]
    fn template_autotuner_top_n_larger_than_results() {
        let mut tuner = TemplateAutotuner::new(SearchSpace::minimal());
        tuner.record_result(Config::new(), 100.0);
        let top5 = tuner.top_n(5);
        assert_eq!(top5.len(), 1);
    }

    #[test]
    fn template_autotuner_results_slice() {
        let mut tuner = TemplateAutotuner::new(SearchSpace::minimal());
        tuner.record_result(Config::new(), 500.0);
        tuner.record_result(Config::new().with_tile_m(64), 600.0);
        assert_eq!(tuner.results().len(), 2);
    }
}
