//! Early stopping for autotuning search.
//!
//! When tuning over a large search space, exhaustive evaluation is
//! expensive.  The [`EarlyStoppingTracker`] monitors benchmark results
//! as they arrive and signals when further exploration is unlikely to
//! yield meaningful improvement.
//!
//! Three independent stopping criteria are supported:
//!
//! 1. **Patience** — stop if no improvement for *N* consecutive trials.
//! 2. **Time budget** — stop after a wall-clock budget (in nanoseconds).
//! 3. **Convergence** — stop when the top-K results have very low
//!    variance (coefficient of variation below a threshold).
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::early_stopping::{EarlyStoppingConfig, EarlyStoppingTracker};
//! use oxicuda_autotune::BenchmarkResult;
//! use oxicuda_autotune::Config;
//!
//! let tracker_cfg = EarlyStoppingConfig {
//!     patience: 5,
//!     ..EarlyStoppingConfig::default()
//! };
//! let mut tracker = EarlyStoppingTracker::new(tracker_cfg);
//!
//! // Simulate recording results (in a real loop you'd benchmark each config).
//! let config = Config::new();
//! let result = BenchmarkResult {
//!     config: config.clone(),
//!     median_us: 100.0,
//!     min_us: 95.0,
//!     max_us: 110.0,
//!     stddev_us: 5.0,
//!     gflops: None,
//!     efficiency: None,
//! };
//! let stop = tracker.record(0, &result, 1_000_000);
//! assert!(stop.is_none()); // too few trials to stop
//! ```

use crate::benchmark::BenchmarkResult;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for early stopping behaviour.
///
/// All thresholds are non-negative.  A `time_budget_ns` of zero means
/// "no time limit".
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Minimum number of configs to evaluate before early stopping can
    /// activate.  This guarantees a baseline level of exploration.
    pub min_trials: usize,
    /// Stop if the best result has not improved for this many
    /// consecutive trials.
    pub patience: usize,
    /// Minimum *relative* improvement to count as a genuine
    /// improvement.  For example, `0.01` means the new result must be
    /// at least 1 % faster than the previous best.
    pub min_relative_improvement: f64,
    /// Maximum total time budget in **nanoseconds**.  Set to `0` to
    /// disable the time-based stopping criterion.
    pub time_budget_ns: u64,
    /// Stop if the coefficient of variation of the top-K results falls
    /// below this threshold.  A small CoV means the best configs are
    /// all roughly equal and further search adds no value.
    pub convergence_threshold: f64,
    /// Number of top results to consider for the convergence check.
    pub convergence_top_k: usize,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            min_trials: 5,
            patience: 10,
            min_relative_improvement: 0.005, // 0.5 %
            time_budget_ns: 0,               // no time limit
            convergence_threshold: 0.02,     // 2 % CoV
            convergence_top_k: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Stop reason
// ---------------------------------------------------------------------------

/// Reason why early stopping triggered.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// Patience exhausted: no improvement for *N* consecutive trials.
    PatienceExhausted {
        /// How many trials elapsed since the last improvement.
        trials_since_improvement: usize,
    },
    /// The caller-supplied time budget has been exceeded.
    TimeBudgetExhausted {
        /// Elapsed time in nanoseconds at the moment stopping was
        /// triggered.
        elapsed_ns: u64,
        /// The configured budget in nanoseconds.
        budget_ns: u64,
    },
    /// The top-K results have converged (low variance).
    Converged {
        /// Coefficient of variation of the top-K median times.
        coefficient_of_variation: f64,
    },
    /// All configurations were evaluated — no early stop was needed.
    AllEvaluated,
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

/// Summary statistics for the early-stopping search.
#[derive(Debug, Clone)]
pub struct EarlyStoppingSummary {
    /// Total number of trials completed.
    pub total_trials: usize,
    /// Index (into the original config list) of the best configuration.
    pub best_config_index: usize,
    /// Best median execution time in microseconds.
    pub best_median_us: f64,
    /// Ratio of best time to the *first* recorded time.  Values below
    /// 1.0 indicate improvement over the initial config.
    pub improvement_ratio: f64,
    /// Search efficiency: at which fraction of total trials was the
    /// best result found.  Lower is better.
    pub search_efficiency: f64,
}

// ---------------------------------------------------------------------------
// Tracker
// ---------------------------------------------------------------------------

/// Tracks autotuning progress and determines when to stop early.
///
/// Feed benchmark results via [`record()`](Self::record) and inspect
/// the return value: `Some(StopReason)` signals that the search
/// should end.
pub struct EarlyStoppingTracker {
    config: EarlyStoppingConfig,
    /// (config_index, median_us) — insertion order.
    results: Vec<(usize, f64)>,
    best_us: f64,
    best_index: usize,
    /// 1-based trial number at which the best was found.
    best_trial: usize,
    trials_since_improvement: usize,
}

impl EarlyStoppingTracker {
    /// Creates a new tracker with the given configuration.
    pub fn new(config: EarlyStoppingConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            best_us: f64::INFINITY,
            best_index: 0,
            best_trial: 0,
            trials_since_improvement: 0,
        }
    }

    /// Record a benchmark result.
    ///
    /// Returns `Some(StopReason)` if the search should stop, `None`
    /// otherwise.
    ///
    /// # Arguments
    ///
    /// * `config_index` — Index of the configuration in the original
    ///   search space.
    /// * `result` — The benchmark result for this configuration.
    /// * `elapsed_ns` — Total wall-clock time spent so far in the
    ///   search loop, in nanoseconds.  The caller is responsible for
    ///   tracking this; we do **not** use `std::time` internally.
    pub fn record(
        &mut self,
        config_index: usize,
        result: &BenchmarkResult,
        elapsed_ns: u64,
    ) -> Option<StopReason> {
        let median = result.median_us;
        self.results.push((config_index, median));

        // --- update best ---
        let improvement_threshold = self.best_us * (1.0 - self.config.min_relative_improvement);
        if median < improvement_threshold {
            self.best_us = median;
            self.best_index = config_index;
            self.best_trial = self.results.len();
            self.trials_since_improvement = 0;
        } else {
            self.trials_since_improvement += 1;
        }

        let n = self.results.len();

        // --- time budget ---
        if self.config.time_budget_ns > 0 && elapsed_ns >= self.config.time_budget_ns {
            return Some(StopReason::TimeBudgetExhausted {
                elapsed_ns,
                budget_ns: self.config.time_budget_ns,
            });
        }

        // --- minimum trials guard ---
        if n < self.config.min_trials {
            return None;
        }

        // --- patience ---
        if self.trials_since_improvement >= self.config.patience {
            return Some(StopReason::PatienceExhausted {
                trials_since_improvement: self.trials_since_improvement,
            });
        }

        // --- convergence ---
        if n >= self.config.convergence_top_k {
            let top_k = self.top_k_medians();
            let cov = coefficient_of_variation(&top_k);
            if cov < self.config.convergence_threshold {
                return Some(StopReason::Converged {
                    coefficient_of_variation: cov,
                });
            }
        }

        None
    }

    /// Returns the best result seen so far as `(config_index, median_us)`.
    ///
    /// Returns `None` if no results have been recorded yet.
    pub fn best(&self) -> Option<(usize, f64)> {
        if self.results.is_empty() {
            None
        } else {
            Some((self.best_index, self.best_us))
        }
    }

    /// Returns all results sorted by performance (lowest median first).
    pub fn sorted_results(&self) -> Vec<(usize, f64)> {
        let mut sorted = self.results.clone();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Returns the number of trials completed so far.
    pub fn trials_completed(&self) -> usize {
        self.results.len()
    }

    /// Returns summary statistics of the search.
    ///
    /// If no results have been recorded, the summary will contain
    /// zeroed fields.
    pub fn summary(&self) -> EarlyStoppingSummary {
        if self.results.is_empty() {
            return EarlyStoppingSummary {
                total_trials: 0,
                best_config_index: 0,
                best_median_us: 0.0,
                improvement_ratio: 1.0,
                search_efficiency: 0.0,
            };
        }

        let first_us = self.results[0].1;
        let improvement_ratio = if first_us > 0.0 {
            self.best_us / first_us
        } else {
            1.0
        };

        let total = self.results.len();
        let search_efficiency = if total > 0 {
            self.best_trial as f64 / total as f64
        } else {
            0.0
        };

        EarlyStoppingSummary {
            total_trials: total,
            best_config_index: self.best_index,
            best_median_us: self.best_us,
            improvement_ratio,
            search_efficiency,
        }
    }

    // -----------------------------------------------------------------------
    // helpers
    // -----------------------------------------------------------------------

    /// Extracts the top-K median values (smallest = best).
    fn top_k_medians(&self) -> Vec<f64> {
        let mut medians: Vec<f64> = self.results.iter().map(|(_, m)| *m).collect();
        medians.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        medians.truncate(self.config.convergence_top_k);
        medians
    }
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// Computes the coefficient of variation (standard deviation / mean)
/// for a slice of **positive** f64 values.
///
/// Returns `0.0` for empty slices, single-element slices, or when the
/// mean is zero or non-finite.
fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let mean = sum / n;

    if mean <= 0.0 || !mean.is_finite() {
        return 0.0;
    }

    let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    if !stddev.is_finite() {
        return 0.0;
    }

    stddev / mean
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// Helper — builds a `BenchmarkResult` with the given median_us.
    fn make_result(median_us: f64) -> BenchmarkResult {
        BenchmarkResult {
            config: Config::new(),
            median_us,
            min_us: median_us * 0.95,
            max_us: median_us * 1.05,
            stddev_us: median_us * 0.02,
            gflops: None,
            efficiency: None,
        }
    }

    // -----------------------------------------------------------------------
    // Default config
    // -----------------------------------------------------------------------

    #[test]
    fn default_config_sensible_values() {
        let cfg = EarlyStoppingConfig::default();
        assert_eq!(cfg.min_trials, 5);
        assert_eq!(cfg.patience, 10);
        assert!((cfg.min_relative_improvement - 0.005).abs() < 1e-12);
        assert_eq!(cfg.time_budget_ns, 0);
        assert!((cfg.convergence_threshold - 0.02).abs() < 1e-12);
        assert_eq!(cfg.convergence_top_k, 5);
    }

    // -----------------------------------------------------------------------
    // min_trials guard
    // -----------------------------------------------------------------------

    #[test]
    fn no_early_stop_before_min_trials() {
        let cfg = EarlyStoppingConfig {
            min_trials: 5,
            patience: 2,
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);

        // Record 4 identical results → patience *would* trigger but
        // min_trials guards against it.
        for i in 0..4 {
            let stop = tracker.record(i, &make_result(100.0), 0);
            assert!(stop.is_none(), "should not stop at trial {i}");
        }
    }

    // -----------------------------------------------------------------------
    // Patience
    // -----------------------------------------------------------------------

    #[test]
    fn patience_triggers_after_non_improving_trials() {
        let cfg = EarlyStoppingConfig {
            min_trials: 2,
            patience: 3,
            min_relative_improvement: 0.01,
            convergence_top_k: 100, // disable convergence
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);

        // Trial 0: best so far
        assert!(tracker.record(0, &make_result(100.0), 0).is_none());
        // Trial 1: improvement
        assert!(tracker.record(1, &make_result(90.0), 0).is_none());
        // Trials 2-4: no improvement (patience = 3)
        assert!(tracker.record(2, &make_result(95.0), 0).is_none());
        assert!(tracker.record(3, &make_result(92.0), 0).is_none());
        let stop = tracker.record(4, &make_result(91.0), 0);
        assert!(stop.is_some());
        match stop {
            Some(StopReason::PatienceExhausted {
                trials_since_improvement,
            }) => {
                assert_eq!(trials_since_improvement, 3);
            }
            other => panic!("expected PatienceExhausted, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Time budget
    // -----------------------------------------------------------------------

    #[test]
    fn time_budget_triggers() {
        let cfg = EarlyStoppingConfig {
            time_budget_ns: 1_000_000, // 1 ms
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);

        // Within budget
        assert!(tracker.record(0, &make_result(100.0), 500_000).is_none());
        // Exceeds budget
        let stop = tracker.record(1, &make_result(90.0), 1_500_000);
        assert!(stop.is_some());
        match stop {
            Some(StopReason::TimeBudgetExhausted {
                elapsed_ns,
                budget_ns,
            }) => {
                assert_eq!(elapsed_ns, 1_500_000);
                assert_eq!(budget_ns, 1_000_000);
            }
            other => panic!("expected TimeBudgetExhausted, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Convergence
    // -----------------------------------------------------------------------

    #[test]
    fn convergence_triggers_when_top_k_similar() {
        let cfg = EarlyStoppingConfig {
            min_trials: 1,
            patience: 100, // disable patience
            convergence_top_k: 3,
            convergence_threshold: 0.05, // 5 %
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);

        // Three very similar results
        assert!(tracker.record(0, &make_result(100.0), 0).is_none());
        assert!(tracker.record(1, &make_result(100.5), 0).is_none());
        let stop = tracker.record(2, &make_result(99.5), 0);
        assert!(stop.is_some());
        match stop {
            Some(StopReason::Converged {
                coefficient_of_variation,
            }) => {
                assert!(coefficient_of_variation < 0.05);
            }
            other => panic!("expected Converged, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // best()
    // -----------------------------------------------------------------------

    #[test]
    fn best_returns_none_before_any_recording() {
        let tracker = EarlyStoppingTracker::new(EarlyStoppingConfig::default());
        assert!(tracker.best().is_none());
    }

    #[test]
    fn best_returns_correct_after_recordings() {
        let cfg = EarlyStoppingConfig {
            convergence_top_k: 100,
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);
        tracker.record(0, &make_result(100.0), 0);
        tracker.record(1, &make_result(80.0), 0);
        tracker.record(2, &make_result(90.0), 0);

        let (idx, us) = tracker.best().expect("should have a best");
        assert_eq!(idx, 1);
        assert!((us - 80.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // sorted_results()
    // -----------------------------------------------------------------------

    #[test]
    fn sorted_results_ordering() {
        let cfg = EarlyStoppingConfig {
            convergence_top_k: 100,
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);
        tracker.record(0, &make_result(300.0), 0);
        tracker.record(1, &make_result(100.0), 0);
        tracker.record(2, &make_result(200.0), 0);

        let sorted = tracker.sorted_results();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].0, 1); // best
        assert_eq!(sorted[1].0, 2);
        assert_eq!(sorted[2].0, 0); // worst
    }

    // -----------------------------------------------------------------------
    // min_relative_improvement threshold
    // -----------------------------------------------------------------------

    #[test]
    fn improvement_requires_min_relative_threshold() {
        let cfg = EarlyStoppingConfig {
            min_trials: 1,
            patience: 2,
            min_relative_improvement: 0.10, // 10 %
            convergence_top_k: 100,
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);

        // Trial 0: baseline 100 µs
        tracker.record(0, &make_result(100.0), 0);
        // Trial 1: 95 µs — only 5 % faster, not enough
        tracker.record(1, &make_result(95.0), 0);
        // Trial 2: 94 µs — still not enough
        let stop = tracker.record(2, &make_result(94.0), 0);
        // patience=2, and 2 non-improving trials after the best (trial 0 baseline
        // counts as the first "best")
        assert!(stop.is_some());

        // Now try with a big improvement that resets patience.
        let cfg2 = EarlyStoppingConfig {
            min_trials: 1,
            patience: 2,
            min_relative_improvement: 0.10,
            convergence_top_k: 100,
            ..EarlyStoppingConfig::default()
        };
        let mut tracker2 = EarlyStoppingTracker::new(cfg2);
        tracker2.record(0, &make_result(100.0), 0);
        tracker2.record(1, &make_result(85.0), 0); // 15 % improvement → resets
        tracker2.record(2, &make_result(86.0), 0); // no improvement
        // Only 1 non-improving trial, patience=2 not reached
        assert_eq!(tracker2.trials_since_improvement, 1);
    }

    // -----------------------------------------------------------------------
    // summary()
    // -----------------------------------------------------------------------

    #[test]
    fn summary_statistics() {
        let cfg = EarlyStoppingConfig {
            convergence_top_k: 100,
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);

        tracker.record(0, &make_result(200.0), 0);
        tracker.record(1, &make_result(100.0), 0);
        tracker.record(2, &make_result(150.0), 0);

        let summary = tracker.summary();
        assert_eq!(summary.total_trials, 3);
        assert_eq!(summary.best_config_index, 1);
        assert!((summary.best_median_us - 100.0).abs() < 1e-6);
        // improvement_ratio = 100 / 200 = 0.5
        assert!((summary.improvement_ratio - 0.5).abs() < 1e-6);
        // best found at trial 2 out of 3 → efficiency = 2/3
        assert!((summary.search_efficiency - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn summary_empty_tracker() {
        let tracker = EarlyStoppingTracker::new(EarlyStoppingConfig::default());
        let summary = tracker.summary();
        assert_eq!(summary.total_trials, 0);
        assert!((summary.improvement_ratio - 1.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn edge_single_trial() {
        let cfg = EarlyStoppingConfig {
            min_trials: 1,
            patience: 1,
            convergence_top_k: 100,
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);
        // With patience=1 and min_trials=1, first trial cannot trigger
        // patience because trials_since_improvement is 0 after recording
        // the first (best) result.
        let stop = tracker.record(0, &make_result(50.0), 0);
        assert!(stop.is_none());
        assert_eq!(tracker.trials_completed(), 1);
    }

    #[test]
    fn edge_all_identical_triggers_convergence() {
        let cfg = EarlyStoppingConfig {
            min_trials: 1,
            patience: 100,
            convergence_top_k: 3,
            convergence_threshold: 0.01,
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);

        tracker.record(0, &make_result(100.0), 0);
        tracker.record(1, &make_result(100.0), 0);
        let stop = tracker.record(2, &make_result(100.0), 0);
        assert!(matches!(stop, Some(StopReason::Converged { .. })));
    }

    #[test]
    fn edge_monotonically_improving_never_triggers_patience() {
        let cfg = EarlyStoppingConfig {
            min_trials: 1,
            patience: 3,
            min_relative_improvement: 0.01, // 1 %
            convergence_top_k: 100,
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);

        // Each trial is at least 1 % better than the previous best.
        for i in 0..20 {
            let t = 100.0 * 0.95_f64.powi(i); // ~5 % improvement each
            let stop = tracker.record(i as usize, &make_result(t), 0);
            // Should never get PatienceExhausted
            if let Some(StopReason::PatienceExhausted { .. }) = stop {
                panic!("patience should not trigger during monotonic improvement");
            }
        }
    }

    #[test]
    fn edge_time_budget_zero_means_no_limit() {
        let cfg = EarlyStoppingConfig {
            time_budget_ns: 0,
            min_trials: 1,
            patience: 100,
            convergence_top_k: 100,
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);

        // Even with enormous elapsed time, no time-budget stop.
        for i in 0..5 {
            let stop = tracker.record(i, &make_result(100.0 - i as f64), u64::MAX);
            if let Some(StopReason::TimeBudgetExhausted { .. }) = stop {
                panic!("time budget 0 should mean no limit");
            }
        }
    }

    // -----------------------------------------------------------------------
    // coefficient_of_variation helper
    // -----------------------------------------------------------------------

    #[test]
    fn cov_empty_and_single() {
        assert!((coefficient_of_variation(&[]) - 0.0).abs() < 1e-12);
        assert!((coefficient_of_variation(&[42.0]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cov_identical_values() {
        let vals = vec![5.0, 5.0, 5.0, 5.0];
        assert!((coefficient_of_variation(&vals) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cov_known_values() {
        // mean = 10, stddev ≈ 1.414, cov ≈ 0.1414
        let vals = vec![8.0, 10.0, 12.0];
        let cov = coefficient_of_variation(&vals);
        // Population stddev = sqrt(((8-10)^2 + (10-10)^2 + (12-10)^2) / 3)
        //                   = sqrt(8/3) ≈ 1.6330
        // cov = 1.6330 / 10 ≈ 0.1633
        assert!((cov - 0.1633).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Patience: exact trial count at which stopping triggers.
    //
    // The tracker records consecutive non-improving trials as the
    // "patience" counter.  When the counter reaches `patience`, the
    // tracker returns `StopReason::PatienceExhausted`.
    // -----------------------------------------------------------------------

    /// After exactly `patience` non-improving trials (no improvement since
    /// the first observation), `record()` should return
    /// `PatienceExhausted`.  The trial *before* that must still return `None`.
    #[test]
    fn early_stopping_patience_triggers_at_exact_count() {
        const PATIENCE: usize = 4;
        let cfg = EarlyStoppingConfig {
            min_trials: 1,
            patience: PATIENCE,
            min_relative_improvement: 0.001,
            convergence_top_k: 100, // disable convergence check
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);

        // Trial 0: best (100 µs) — resets the patience counter.
        assert!(tracker.record(0, &make_result(100.0), 0).is_none());

        // Trials 1 .. PATIENCE-1: no improvement — counter increments.
        for i in 1..PATIENCE {
            let stop = tracker.record(i, &make_result(100.0), 0);
            assert!(
                stop.is_none(),
                "should not stop before patience is exhausted (trial {i})"
            );
        }

        // Trial PATIENCE: this is the PATIENCE-th consecutive non-improving
        // trial, so the tracker should fire now.
        let stop = tracker.record(PATIENCE, &make_result(100.0), 0);
        assert!(
            stop.is_some(),
            "tracker must fire after {PATIENCE} non-improving trials"
        );
        match stop {
            Some(StopReason::PatienceExhausted {
                trials_since_improvement,
            }) => {
                assert_eq!(
                    trials_since_improvement, PATIENCE,
                    "trials_since_improvement must equal patience"
                );
            }
            other => panic!("expected PatienceExhausted, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Time budget: stopping via wall-clock budget.
    // -----------------------------------------------------------------------

    /// Pass `elapsed_ns` well above the budget and verify `TimeBudgetExhausted`
    /// is returned immediately (before `min_trials` would prevent it).
    ///
    /// The time-budget check runs *before* the min_trials guard, so even the
    /// very first `record()` call can trigger it.
    #[test]
    fn early_stopping_time_budget_triggers_immediately() {
        // Budget: 1 µs (1_000 ns)
        let budget_ns: u64 = 1_000;
        let cfg = EarlyStoppingConfig {
            min_trials: 100, // large — would ordinarily suppress stopping
            patience: 100,
            time_budget_ns: budget_ns,
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);

        // Elapsed is 10× the budget — must stop immediately.
        let elapsed_ns = budget_ns * 10;
        let stop = tracker.record(0, &make_result(50.0), elapsed_ns);

        assert!(
            stop.is_some(),
            "tracker must stop when elapsed exceeds time budget"
        );
        match stop {
            Some(StopReason::TimeBudgetExhausted {
                elapsed_ns: got_elapsed,
                budget_ns: got_budget,
            }) => {
                assert_eq!(got_elapsed, elapsed_ns);
                assert_eq!(got_budget, budget_ns);
            }
            other => panic!("expected TimeBudgetExhausted, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Patience reset: a high-variance step resets the patience counter.
    //
    // In terms of the API: any observation whose median is significantly
    // better than the current best (relative improvement >= threshold)
    // resets `trials_since_improvement` to zero, preventing patience
    // from triggering.
    // -----------------------------------------------------------------------

    /// After `patience-1` non-improving trials, inserting one genuinely
    /// improving trial resets the counter, so patience does not trigger on
    /// the next non-improving trial.
    #[test]
    fn early_stopping_improvement_resets_patience_counter() {
        const PATIENCE: usize = 5;
        let cfg = EarlyStoppingConfig {
            min_trials: 1,
            patience: PATIENCE,
            min_relative_improvement: 0.01, // 1 %
            convergence_top_k: 100,
            ..EarlyStoppingConfig::default()
        };
        let mut tracker = EarlyStoppingTracker::new(cfg);

        // Trial 0: baseline best (200 µs).
        assert!(tracker.record(0, &make_result(200.0), 0).is_none());

        // Trials 1 .. PATIENCE-2: no improvement (patience - 2 = 3 trials).
        for i in 1..=(PATIENCE - 2) {
            assert!(
                tracker.record(i, &make_result(200.0), 0).is_none(),
                "should not stop at trial {i}"
            );
        }
        // At this point trials_since_improvement = PATIENCE-2 = 3 (out of 5).

        // Insert one genuinely improving observation (>1% better) → resets counter.
        let improving_idx = PATIENCE - 1;
        let stop = tracker.record(improving_idx, &make_result(100.0), 0); // 50 % faster
        // This is a genuine improvement — should not stop.
        assert!(
            stop.is_none() || !matches!(stop, Some(StopReason::PatienceExhausted { .. })),
            "an improving trial must not trigger PatienceExhausted"
        );
        // Counter should now be 0 (reset by the improvement).
        assert_eq!(
            tracker.trials_since_improvement, 0,
            "patience counter must be reset after a genuine improvement"
        );

        // Now add PATIENCE-2 more non-improving trials — should still not trigger.
        for i in 0..(PATIENCE - 2) {
            let stop = tracker.record(PATIENCE + i, &make_result(100.0), 0);
            assert!(
                stop.is_none() || !matches!(stop, Some(StopReason::PatienceExhausted { .. })),
                "patience must not trigger only {i} trials after reset"
            );
        }
    }
}
