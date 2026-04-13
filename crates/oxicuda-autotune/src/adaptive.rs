//! Real-time adaptive tuning for dynamic kernel configuration switching.
//!
//! This module provides runtime-adaptive kernel tuning that dynamically
//! switches between kernel configurations based on observed performance
//! metrics. Unlike offline autotuning (which runs once and persists the
//! best result), adaptive tuning continuously monitors execution metrics
//! and reacts to changing workload characteristics — e.g., problem sizes
//! that shift over time, thermal throttling, or contention from
//! co-scheduled kernels.
//!
//! ## Architecture
//!
//! ```text
//!  ┌──────────────────────────────────────────────────────────────┐
//!  │                    AdaptiveSelector                          │
//!  │                                                              │
//!  │  record_metric(RuntimeMetric)                                │
//!  │        │                                                     │
//!  │        ▼                                                     │
//!  │  MetricsWindow (sliding window per config)                   │
//!  │        │                                                     │
//!  │        ▼                                                     │
//!  │  evaluate_switch() ──► SwitchDecision                        │
//!  │        │                                                     │
//!  │        ▼                                                     │
//!  │  apply_switch() ──► update current config                    │
//!  │                                                              │
//!  │  ExplorationScheduler ──► round-robin unexplored candidates  │
//!  │                                                              │
//!  │  PerformanceRegression ──► detect config degradation         │
//!  └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use oxicuda_autotune::adaptive::*;
//! use oxicuda_autotune::Config;
//!
//! let candidates = vec![
//!     Config::new().with_tile_m(64).with_tile_n(64),
//!     Config::new().with_tile_m(128).with_tile_n(128),
//! ];
//!
//! let mut selector = AdaptiveSelector::new(AdaptivePolicy::Moderate, candidates);
//!
//! // Record runtime observations
//! selector.record_metric(RuntimeMetric {
//!     timestamp_us: 1000,
//!     kernel_name: "sgemm".to_string(),
//!     config_hash: selector.current_config_hash(),
//!     execution_time_us: 42.0,
//!     occupancy: 0.85,
//! });
//!
//! // Evaluate whether to switch
//! let decision = selector.evaluate_switch();
//! if decision.should_switch {
//!     selector.apply_switch(&decision);
//! }
//! ```
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::config::Config;

// ---------------------------------------------------------------------------
// RuntimeMetric
// ---------------------------------------------------------------------------

/// A single runtime performance observation for a kernel execution.
#[derive(Debug, Clone)]
pub struct RuntimeMetric {
    /// Timestamp in microseconds (monotonic clock).
    pub timestamp_us: u64,
    /// Name of the kernel that was executed.
    pub kernel_name: String,
    /// Hash of the [`Config`] that was used.
    pub config_hash: u64,
    /// Measured execution time in microseconds.
    pub execution_time_us: f64,
    /// Achieved occupancy (0.0–1.0).
    pub occupancy: f64,
}

// ---------------------------------------------------------------------------
// PerformanceTrend
// ---------------------------------------------------------------------------

/// Describes the direction of performance over the metrics window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceTrend {
    /// Execution times are decreasing (getting faster).
    Improving,
    /// Execution times are roughly constant.
    Stable,
    /// Execution times are increasing (getting slower).
    Degrading,
}

// ---------------------------------------------------------------------------
// MetricsWindow
// ---------------------------------------------------------------------------

/// A fixed-size sliding window of [`RuntimeMetric`] observations.
///
/// Provides statistical summaries (mean, stddev) and trend detection
/// via simple linear regression over the observation timestamps.
#[derive(Debug, Clone)]
pub struct MetricsWindow {
    /// Ring buffer of metrics.
    buffer: Vec<RuntimeMetric>,
    /// Maximum number of observations to keep.
    capacity: usize,
    /// Write cursor into the ring buffer.
    cursor: usize,
    /// Total number of observations pushed (may exceed capacity).
    total_pushed: usize,
}

impl MetricsWindow {
    /// Creates a new metrics window with the given capacity.
    ///
    /// # Panics
    ///
    /// This function does not panic. A `window_size` of 0 is treated as 1.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        let capacity = window_size.max(1);
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            cursor: 0,
            total_pushed: 0,
        }
    }

    /// Pushes a new metric into the sliding window.
    ///
    /// When the window is full, the oldest observation is overwritten.
    pub fn push(&mut self, metric: RuntimeMetric) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(metric);
        } else {
            self.buffer[self.cursor] = metric;
        }
        self.cursor = (self.cursor + 1) % self.capacity;
        self.total_pushed += 1;
    }

    /// Returns the number of observations currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the window contains no observations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns `true` if the window has reached its capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Returns the total number of observations pushed (including
    /// those that have been evicted).
    #[must_use]
    pub fn total_pushed(&self) -> usize {
        self.total_pushed
    }

    /// Computes the arithmetic mean of execution times, or `None` if empty.
    #[must_use]
    pub fn mean_time_us(&self) -> Option<f64> {
        if self.buffer.is_empty() {
            return None;
        }
        let sum: f64 = self.buffer.iter().map(|m| m.execution_time_us).sum();
        Some(sum / self.buffer.len() as f64)
    }

    /// Computes the sample standard deviation of execution times,
    /// or `None` if fewer than 2 observations.
    #[must_use]
    pub fn stddev_time_us(&self) -> Option<f64> {
        if self.buffer.len() < 2 {
            return None;
        }
        let mean = self.mean_time_us()?;
        let n = self.buffer.len() as f64;
        let variance: f64 = self
            .buffer
            .iter()
            .map(|m| {
                let d = m.execution_time_us - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        Some(variance.sqrt())
    }

    /// Determines the performance trend using linear regression of
    /// execution time over observation index.
    ///
    /// The slope of the regression line determines the trend:
    /// - Negative slope → [`PerformanceTrend::Improving`] (times decreasing)
    /// - Near-zero slope → [`PerformanceTrend::Stable`]
    /// - Positive slope → [`PerformanceTrend::Degrading`] (times increasing)
    ///
    /// With fewer than 3 observations, returns [`PerformanceTrend::Stable`].
    #[must_use]
    pub fn trend(&self) -> PerformanceTrend {
        if self.buffer.len() < 3 {
            return PerformanceTrend::Stable;
        }

        // Reconstruct chronological order from the ring buffer.
        let ordered = self.ordered_view();
        let n = ordered.len() as f64;

        // Simple linear regression: y = execution_time, x = index
        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        let mut sum_xy: f64 = 0.0;
        let mut sum_x2: f64 = 0.0;

        for (i, metric) in ordered.iter().enumerate() {
            let x = i as f64;
            let y = metric.execution_time_us;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return PerformanceTrend::Stable;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;

        // Normalize slope relative to mean to get a dimensionless measure
        let mean = sum_y / n;
        if mean.abs() < f64::EPSILON {
            return PerformanceTrend::Stable;
        }
        let normalized_slope = slope / mean;

        // Thresholds: > 1% per sample → degrading, < -1% → improving
        const TREND_THRESHOLD: f64 = 0.01;
        if normalized_slope > TREND_THRESHOLD {
            PerformanceTrend::Degrading
        } else if normalized_slope < -TREND_THRESHOLD {
            PerformanceTrend::Improving
        } else {
            PerformanceTrend::Stable
        }
    }

    /// Returns a chronologically ordered view of the ring buffer.
    fn ordered_view(&self) -> Vec<&RuntimeMetric> {
        if self.buffer.len() < self.capacity {
            // Not yet wrapped
            self.buffer.iter().collect()
        } else {
            // Wrapped: oldest element is at self.cursor
            let (tail, head) = self.buffer.split_at(self.cursor);
            head.iter().chain(tail.iter()).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// AdaptivePolicy
// ---------------------------------------------------------------------------

/// Policy governing how aggressively the adaptive tuner switches configs.
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptivePolicy {
    /// Switch only when a candidate is ≥20% better over 100 samples.
    Conservative,
    /// Switch when a candidate is ≥10% better over 50 samples.
    Moderate,
    /// Switch when a candidate is ≥5% better over 20 samples.
    Aggressive,
    /// User-defined thresholds.
    Custom {
        /// Minimum improvement fraction (e.g. 0.15 = 15%).
        improvement_threshold: f64,
        /// Minimum number of samples before a switch is considered.
        min_samples: usize,
    },
}

impl AdaptivePolicy {
    /// Returns `(improvement_threshold, min_samples)` for this policy.
    #[must_use]
    pub fn thresholds(&self) -> (f64, usize) {
        match self {
            Self::Conservative => (0.20, 100),
            Self::Moderate => (0.10, 50),
            Self::Aggressive => (0.05, 20),
            Self::Custom {
                improvement_threshold,
                min_samples,
            } => (*improvement_threshold, *min_samples),
        }
    }
}

// ---------------------------------------------------------------------------
// SwitchDecision
// ---------------------------------------------------------------------------

/// The outcome of evaluating whether to switch kernel configurations.
#[derive(Debug, Clone)]
pub struct SwitchDecision {
    /// Whether the selector recommends switching.
    pub should_switch: bool,
    /// Hash of the currently active configuration.
    pub current_config_hash: u64,
    /// Hash of the proposed replacement configuration.
    pub proposed_config_hash: u64,
    /// Expected improvement percentage (e.g., 15.0 means 15% faster).
    pub expected_improvement_pct: f64,
    /// Confidence in the decision (0.0–1.0), based on sample size and
    /// variance.
    pub confidence: f64,
    /// Human-readable explanation of the decision.
    pub reason: String,
}

// ---------------------------------------------------------------------------
// AdaptiveSelector
// ---------------------------------------------------------------------------

/// Manages adaptive kernel configuration selection with runtime feedback.
///
/// Maintains per-config metrics windows and uses the configured
/// [`AdaptivePolicy`] to decide when switching to a different config
/// is justified by sufficient evidence of improvement.
pub struct AdaptiveSelector {
    /// The tuning policy.
    policy: AdaptivePolicy,
    /// All candidate configurations, keyed by their hash.
    candidates: Vec<Config>,
    /// Hash → index into `candidates`.
    hash_to_index: HashMap<u64, usize>,
    /// Per-config sliding windows of metrics.
    windows: HashMap<u64, MetricsWindow>,
    /// Index of the currently active configuration in `candidates`.
    active_index: usize,
    /// Number of config switches performed.
    total_switches: usize,
    /// Total exploration runs.
    exploration_runs: usize,
    /// Total exploitation runs.
    exploitation_runs: usize,
    /// Performance history: (timestamp_us, execution_time_us).
    performance_history: Vec<(u64, f64)>,
}

impl AdaptiveSelector {
    /// Creates a new adaptive selector.
    ///
    /// The first candidate in `candidates` is used as the initial active
    /// config. If `candidates` is empty, a default config is inserted.
    #[must_use]
    pub fn new(policy: AdaptivePolicy, candidates: Vec<Config>) -> Self {
        let candidates = if candidates.is_empty() {
            vec![Config::default()]
        } else {
            candidates
        };

        let (_, min_samples) = policy.thresholds();
        let window_size = min_samples.max(20);

        let mut hash_to_index = HashMap::new();
        let mut windows = HashMap::new();
        for (i, cfg) in candidates.iter().enumerate() {
            let h = config_hash(cfg);
            hash_to_index.insert(h, i);
            windows.insert(h, MetricsWindow::new(window_size));
        }

        Self {
            policy,
            candidates,
            hash_to_index,
            windows,
            active_index: 0,
            total_switches: 0,
            exploration_runs: 0,
            exploitation_runs: 0,
            performance_history: Vec::new(),
        }
    }

    /// Returns a reference to the currently active configuration.
    #[must_use]
    pub fn current_config(&self) -> &Config {
        &self.candidates[self.active_index]
    }

    /// Returns the hash of the currently active configuration.
    #[must_use]
    pub fn current_config_hash(&self) -> u64 {
        config_hash(&self.candidates[self.active_index])
    }

    /// Records a runtime metric observation.
    ///
    /// The metric is routed to the appropriate per-config window based
    /// on `metric.config_hash`. If the hash is unknown, the metric is
    /// silently dropped.
    pub fn record_metric(&mut self, metric: RuntimeMetric) {
        let hash = metric.config_hash;
        let ts = metric.timestamp_us;
        let time = metric.execution_time_us;

        // Track whether this is exploration or exploitation
        let active_hash = self.current_config_hash();
        if hash == active_hash {
            self.exploitation_runs += 1;
        } else if self.hash_to_index.contains_key(&hash) {
            self.exploration_runs += 1;
        }

        self.performance_history.push((ts, time));

        if let Some(window) = self.windows.get_mut(&hash) {
            window.push(metric);
        }
    }

    /// Evaluates whether switching to a different configuration is
    /// warranted based on accumulated metrics.
    ///
    /// Compares the current config's mean execution time against all
    /// other candidates that have sufficient samples, and recommends
    /// switching if the improvement exceeds the policy threshold.
    #[must_use]
    pub fn evaluate_switch(&self) -> SwitchDecision {
        let (threshold, min_samples) = self.policy.thresholds();
        let active_hash = self.current_config_hash();

        let current_mean = self.windows.get(&active_hash).and_then(|w| {
            if w.len() >= min_samples {
                w.mean_time_us()
            } else {
                None
            }
        });

        let current_mean = match current_mean {
            Some(m) => m,
            None => {
                return SwitchDecision {
                    should_switch: false,
                    current_config_hash: active_hash,
                    proposed_config_hash: active_hash,
                    expected_improvement_pct: 0.0,
                    confidence: 0.0,
                    reason: "Insufficient samples for current config".to_string(),
                };
            }
        };

        let mut best_candidate_hash = active_hash;
        let mut best_improvement: f64 = 0.0;
        let mut best_confidence: f64 = 0.0;

        for (&candidate_hash, window) in &self.windows {
            if candidate_hash == active_hash {
                continue;
            }
            if window.len() < min_samples {
                continue;
            }
            let candidate_mean = match window.mean_time_us() {
                Some(m) => m,
                None => continue,
            };

            if current_mean.abs() < f64::EPSILON {
                continue;
            }

            let improvement = (current_mean - candidate_mean) / current_mean;

            // Confidence based on sample count and variance stability
            let confidence = compute_confidence(window, min_samples);

            if improvement > best_improvement {
                best_improvement = improvement;
                best_candidate_hash = candidate_hash;
                best_confidence = confidence;
            }
        }

        let should_switch = best_improvement >= threshold && best_candidate_hash != active_hash;
        let reason = if should_switch {
            format!(
                "Candidate 0x{:016x} is {:.1}% faster (threshold: {:.1}%, confidence: {:.2})",
                best_candidate_hash,
                best_improvement * 100.0,
                threshold * 100.0,
                best_confidence,
            )
        } else if best_candidate_hash == active_hash {
            "No candidate outperforms the current config".to_string()
        } else {
            format!(
                "Best candidate is only {:.1}% faster (need {:.1}%)",
                best_improvement * 100.0,
                threshold * 100.0,
            )
        };

        SwitchDecision {
            should_switch,
            current_config_hash: active_hash,
            proposed_config_hash: best_candidate_hash,
            expected_improvement_pct: best_improvement * 100.0,
            confidence: best_confidence,
            reason,
        }
    }

    /// Applies a switch decision, changing the active configuration.
    ///
    /// If `decision.should_switch` is `false` or the proposed config
    /// hash is unknown, no change is made.
    pub fn apply_switch(&mut self, decision: &SwitchDecision) {
        if !decision.should_switch {
            return;
        }
        if let Some(&idx) = self.hash_to_index.get(&decision.proposed_config_hash) {
            self.active_index = idx;
            self.total_switches += 1;
        }
    }

    /// Returns how many more exploratory runs should be scheduled.
    ///
    /// This is based on the minimum samples required by the policy
    /// minus the samples already collected for the least-explored
    /// candidate.
    #[must_use]
    pub fn exploration_budget(&self) -> usize {
        let (_, min_samples) = self.policy.thresholds();
        let mut budget: usize = 0;
        for window in self.windows.values() {
            if window.len() < min_samples {
                budget += min_samples - window.len();
            }
        }
        budget
    }

    /// Returns `true` if the selector is still in the exploration phase.
    ///
    /// Exploration is ongoing while any candidate has fewer than
    /// `min_samples` observations.
    #[must_use]
    pub fn is_exploring(&self) -> bool {
        let (_, min_samples) = self.policy.thresholds();
        self.windows.values().any(|w| w.len() < min_samples)
    }

    /// Generates a report summarizing the adaptive tuning session.
    #[must_use]
    pub fn report(&self) -> AdaptiveTuneReport {
        let active_hash = self.current_config_hash();

        // Find the config with the best (lowest) mean time
        let best_config_hash = self
            .windows
            .iter()
            .filter_map(|(&h, w)| w.mean_time_us().map(|m| (h, m)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(h, _)| h)
            .unwrap_or(active_hash);

        AdaptiveTuneReport {
            total_switches: self.total_switches,
            exploration_runs: self.exploration_runs,
            exploitation_runs: self.exploitation_runs,
            best_config_hash,
            performance_history: self.performance_history.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// ExplorationScheduler
// ---------------------------------------------------------------------------

/// Manages systematic exploration of candidate configurations.
///
/// During the exploration phase, the scheduler round-robins through
/// unexplored candidates. Once all candidates have been explored, it
/// switches to exploitation mode, always returning the best known config.
pub struct ExplorationScheduler {
    /// Candidate configs in exploration order.
    candidates: Vec<Config>,
    /// Results collected per config hash: (hash → best execution time).
    results: HashMap<u64, f64>,
    /// Round-robin index for unexplored configs.
    rr_index: usize,
    /// Fraction of runs to spend exploring (e.g., 0.1 = 10%).
    exploration_ratio: f64,
    /// Total calls to `next_config`.
    total_calls: usize,
}

impl ExplorationScheduler {
    /// Creates a new exploration scheduler.
    ///
    /// `exploration_ratio` should be between 0.0 and 1.0 (e.g., 0.1
    /// means 10% of runs are exploration).
    #[must_use]
    pub fn new(candidates: Vec<Config>, exploration_ratio: f64) -> Self {
        let exploration_ratio = exploration_ratio.clamp(0.0, 1.0);
        Self {
            candidates,
            results: HashMap::new(),
            rr_index: 0,
            exploration_ratio,
            total_calls: 0,
        }
    }

    /// Returns the next configuration to benchmark.
    ///
    /// During the initial round-robin phase, returns each candidate in
    /// turn. After all candidates have been explored at least once,
    /// exploration runs are interleaved at the `exploration_ratio`
    /// frequency; other runs exploit the best known config.
    #[must_use]
    pub fn next_config(&mut self) -> &Config {
        self.total_calls += 1;

        // If we haven't explored all candidates yet, round-robin
        let unexplored: Vec<usize> = self
            .candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| !self.results.contains_key(&config_hash(c)))
            .map(|(i, _)| i)
            .collect();

        if !unexplored.is_empty() {
            let idx = unexplored[self.rr_index % unexplored.len()];
            self.rr_index += 1;
            return &self.candidates[idx];
        }

        // All explored: interleave exploration at the given ratio
        let should_explore = self.exploration_ratio > 0.0
            && (self.total_calls as f64 * self.exploration_ratio) as usize
                > ((self.total_calls - 1) as f64 * self.exploration_ratio) as usize;

        if should_explore && self.candidates.len() > 1 {
            // Round-robin through all candidates for exploration
            let idx = self.rr_index % self.candidates.len();
            self.rr_index += 1;
            &self.candidates[idx]
        } else {
            // Exploit: return the best known config
            self.best_config()
        }
    }

    /// Records the result of benchmarking a configuration.
    ///
    /// Keeps track of the best (lowest) execution time for each config.
    pub fn mark_explored(&mut self, config_hash: u64, result: f64) {
        let entry = self.results.entry(config_hash).or_insert(f64::MAX);
        if result < *entry {
            *entry = result;
        }
    }

    /// Returns the best known configuration based on exploration results.
    ///
    /// Falls back to the first candidate if no results have been recorded.
    #[must_use]
    fn best_config(&self) -> &Config {
        self.results
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .and_then(|(h, _)| self.candidates.iter().find(|c| config_hash(c) == *h))
            .unwrap_or(&self.candidates[0])
    }

    /// Returns the number of distinct configs that have been explored.
    #[must_use]
    pub fn explored_count(&self) -> usize {
        self.results.len()
    }

    /// Returns the total number of candidate configs.
    #[must_use]
    pub fn total_candidates(&self) -> usize {
        self.candidates.len()
    }
}

// ---------------------------------------------------------------------------
// AdaptiveTuneReport
// ---------------------------------------------------------------------------

/// Summary statistics for an adaptive tuning session.
#[derive(Debug, Clone)]
pub struct AdaptiveTuneReport {
    /// Number of times the active configuration was switched.
    pub total_switches: usize,
    /// Number of exploration runs (benchmarking non-active configs).
    pub exploration_runs: usize,
    /// Number of exploitation runs (using the active config).
    pub exploitation_runs: usize,
    /// Hash of the best-performing configuration found.
    pub best_config_hash: u64,
    /// Chronological history of (timestamp_us, execution_time_us).
    pub performance_history: Vec<(u64, f64)>,
}

// ---------------------------------------------------------------------------
// PerformanceRegression
// ---------------------------------------------------------------------------

/// Detects when a configuration's performance has regressed beyond
/// an acceptable threshold compared to a known baseline.
#[derive(Debug, Clone)]
pub struct PerformanceRegression {
    /// How much slower the current window mean is compared to baseline,
    /// expressed as a percentage (e.g., 25.0 means 25% slower).
    pub regression_pct: f64,
    /// Number of consecutive samples that were slower than the threshold.
    pub samples_degraded: usize,
}

impl PerformanceRegression {
    /// Checks whether the given metrics window shows a regression
    /// relative to `baseline_us`.
    ///
    /// Returns `Some(PerformanceRegression)` if the window's mean
    /// execution time exceeds `baseline_us` by more than
    /// `threshold_pct` percent. Returns `None` otherwise.
    ///
    /// # Arguments
    ///
    /// * `window` — The sliding window of recent metrics.
    /// * `baseline_us` — The expected (good) execution time in µs.
    /// * `threshold_pct` — Percentage above baseline that constitutes
    ///   a regression (e.g., 20.0 for 20%).
    #[must_use]
    pub fn detect(window: &MetricsWindow, baseline_us: f64, threshold_pct: f64) -> Option<Self> {
        let mean = window.mean_time_us()?;

        if baseline_us.abs() < f64::EPSILON {
            return None;
        }

        let regression_pct = ((mean - baseline_us) / baseline_us) * 100.0;

        if regression_pct > threshold_pct {
            // Count samples above the threshold
            let threshold_time = baseline_us * (1.0 + threshold_pct / 100.0);
            let samples_degraded = window
                .buffer
                .iter()
                .filter(|m| m.execution_time_us > threshold_time)
                .count();

            Some(Self {
                regression_pct,
                samples_degraded,
            })
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Computes a deterministic hash for a [`Config`].
#[must_use]
pub fn config_hash(config: &Config) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    config.hash(&mut hasher);
    hasher.finish()
}

/// Computes a confidence score (0.0–1.0) for a metrics window based on
/// sample count and variance stability.
fn compute_confidence(window: &MetricsWindow, min_samples: usize) -> f64 {
    let n = window.len() as f64;
    let target = min_samples as f64;

    // Sample-based confidence: how close to min_samples
    let sample_confidence = (n / target).min(1.0);

    // Variance-based confidence: lower coefficient of variation → higher
    let variance_confidence = match (window.mean_time_us(), window.stddev_time_us()) {
        (Some(mean), Some(stddev)) if mean.abs() > f64::EPSILON => {
            let cv = stddev / mean; // coefficient of variation
            // CV of 0 → confidence 1.0, CV of 1+ → confidence ~0.3
            1.0 / (1.0 + cv * 2.0)
        }
        _ => 0.5,
    };

    sample_confidence * 0.6 + variance_confidence * 0.4
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a metric with defaults.
    fn make_metric(config_hash: u64, time_us: f64) -> RuntimeMetric {
        RuntimeMetric {
            timestamp_us: 0,
            kernel_name: "test_kernel".to_string(),
            config_hash,
            execution_time_us: time_us,
            occupancy: 0.8,
        }
    }

    fn make_metric_with_ts(config_hash: u64, time_us: f64, ts: u64) -> RuntimeMetric {
        RuntimeMetric {
            timestamp_us: ts,
            kernel_name: "test_kernel".to_string(),
            config_hash,
            execution_time_us: time_us,
            occupancy: 0.8,
        }
    }

    // ── MetricsWindow: push / mean / stddev ──────────────────────────

    #[test]
    fn metrics_window_push_and_mean() {
        let mut w = MetricsWindow::new(5);
        assert!(w.is_empty());
        assert_eq!(w.mean_time_us(), None);

        w.push(make_metric(1, 10.0));
        w.push(make_metric(1, 20.0));
        w.push(make_metric(1, 30.0));

        assert_eq!(w.len(), 3);
        assert!(!w.is_full());

        let mean = w.mean_time_us().expect("should have mean");
        assert!((mean - 20.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_window_stddev() {
        let mut w = MetricsWindow::new(10);
        // Push identical values — stddev should be 0
        for _ in 0..5 {
            w.push(make_metric(1, 50.0));
        }
        let sd = w.stddev_time_us().expect("should have stddev");
        assert!(sd.abs() < 1e-9, "stddev of identical values should be 0");

        // Push varied values
        let mut w2 = MetricsWindow::new(10);
        w2.push(make_metric(1, 10.0));
        w2.push(make_metric(1, 20.0));
        w2.push(make_metric(1, 30.0));
        let sd2 = w2.stddev_time_us().expect("should have stddev");
        assert!(sd2 > 0.0, "stddev of varied values should be positive");
        // sample stddev of [10, 20, 30] = 10.0
        assert!((sd2 - 10.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_window_single_element_no_stddev() {
        let mut w = MetricsWindow::new(5);
        w.push(make_metric(1, 42.0));
        assert_eq!(w.stddev_time_us(), None);
        assert_eq!(w.mean_time_us(), Some(42.0));
    }

    // ── MetricsWindow: trend detection ───────────────────────────────

    #[test]
    fn trend_improving() {
        let mut w = MetricsWindow::new(20);
        // Execution times clearly decreasing
        for i in 0..10 {
            let time = 100.0 - (i as f64) * 5.0; // 100, 95, 90, ...
            w.push(make_metric(1, time));
        }
        assert_eq!(w.trend(), PerformanceTrend::Improving);
    }

    #[test]
    fn trend_stable() {
        let mut w = MetricsWindow::new(20);
        // Execution times roughly constant with tiny noise
        for i in 0..10 {
            let noise = if i % 2 == 0 { 0.001 } else { -0.001 };
            w.push(make_metric(1, 50.0 + noise));
        }
        assert_eq!(w.trend(), PerformanceTrend::Stable);
    }

    #[test]
    fn trend_degrading() {
        let mut w = MetricsWindow::new(20);
        // Execution times clearly increasing
        for i in 0..10 {
            let time = 50.0 + (i as f64) * 5.0; // 50, 55, 60, ...
            w.push(make_metric(1, time));
        }
        assert_eq!(w.trend(), PerformanceTrend::Degrading);
    }

    #[test]
    fn trend_too_few_samples() {
        let mut w = MetricsWindow::new(20);
        w.push(make_metric(1, 100.0));
        w.push(make_metric(1, 50.0));
        // Only 2 samples → always Stable
        assert_eq!(w.trend(), PerformanceTrend::Stable);
    }

    // ── Window overflow / ring buffer ────────────────────────────────

    #[test]
    fn window_overflow_evicts_oldest() {
        let mut w = MetricsWindow::new(3);
        w.push(make_metric(1, 100.0));
        w.push(make_metric(1, 200.0));
        w.push(make_metric(1, 300.0));
        assert!(w.is_full());
        assert_eq!(w.total_pushed(), 3);

        // Push a 4th — should evict the 100.0
        w.push(make_metric(1, 400.0));
        assert_eq!(w.len(), 3);
        assert_eq!(w.total_pushed(), 4);

        // Mean should be (200 + 300 + 400) / 3 = 300
        let mean = w.mean_time_us().expect("should have mean");
        assert!((mean - 300.0).abs() < 1e-9);
    }

    // ── Empty window handling ────────────────────────────────────────

    #[test]
    fn empty_window_returns_none() {
        let w = MetricsWindow::new(10);
        assert_eq!(w.mean_time_us(), None);
        assert_eq!(w.stddev_time_us(), None);
        assert_eq!(w.trend(), PerformanceTrend::Stable);
        assert!(!w.is_full());
        assert!(w.is_empty());
    }

    // ── AdaptivePolicy thresholds ────────────────────────────────────

    #[test]
    fn policy_thresholds() {
        assert_eq!(AdaptivePolicy::Conservative.thresholds(), (0.20, 100));
        assert_eq!(AdaptivePolicy::Moderate.thresholds(), (0.10, 50));
        assert_eq!(AdaptivePolicy::Aggressive.thresholds(), (0.05, 20));

        let custom = AdaptivePolicy::Custom {
            improvement_threshold: 0.15,
            min_samples: 75,
        };
        assert_eq!(custom.thresholds(), (0.15, 75));
    }

    // ── Conservative vs Aggressive comparison ────────────────────────

    #[test]
    fn conservative_vs_aggressive_policy() {
        let configs = vec![
            Config::new().with_tile_m(64),
            Config::new().with_tile_m(128),
        ];
        let hash_a = config_hash(&configs[0]);
        let hash_b = config_hash(&configs[1]);

        // Aggressive: 5% improvement over 20 samples
        let mut aggressive = AdaptiveSelector::new(AdaptivePolicy::Aggressive, configs.clone());
        // Conservative: 20% improvement over 100 samples
        let mut conservative = AdaptiveSelector::new(AdaptivePolicy::Conservative, configs);

        // Feed both: config A at 100us, config B at 92us (8% improvement)
        for i in 0..100 {
            let m_a = make_metric_with_ts(hash_a, 100.0, i);
            let m_b = make_metric_with_ts(hash_b, 92.0, i + 1000);
            aggressive.record_metric(m_a.clone());
            aggressive.record_metric(m_b.clone());
            conservative.record_metric(RuntimeMetric {
                timestamp_us: m_a.timestamp_us,
                kernel_name: m_a.kernel_name.clone(),
                config_hash: m_a.config_hash,
                execution_time_us: m_a.execution_time_us,
                occupancy: m_a.occupancy,
            });
            conservative.record_metric(RuntimeMetric {
                timestamp_us: m_b.timestamp_us,
                kernel_name: m_b.kernel_name.clone(),
                config_hash: m_b.config_hash,
                execution_time_us: m_b.execution_time_us,
                occupancy: m_b.occupancy,
            });
        }

        let agg_decision = aggressive.evaluate_switch();
        let con_decision = conservative.evaluate_switch();

        // Aggressive should switch (8% > 5%), Conservative should not (8% < 20%)
        assert!(
            agg_decision.should_switch,
            "Aggressive should want to switch"
        );
        assert!(
            !con_decision.should_switch,
            "Conservative should not switch"
        );
    }

    // ── SwitchDecision creation ──────────────────────────────────────

    #[test]
    fn switch_decision_fields() {
        let decision = SwitchDecision {
            should_switch: true,
            current_config_hash: 0xAABB,
            proposed_config_hash: 0xCCDD,
            expected_improvement_pct: 15.5,
            confidence: 0.92,
            reason: "test reason".to_string(),
        };
        assert!(decision.should_switch);
        assert_eq!(decision.current_config_hash, 0xAABB);
        assert_eq!(decision.proposed_config_hash, 0xCCDD);
        assert!((decision.expected_improvement_pct - 15.5).abs() < 1e-9);
        assert!((decision.confidence - 0.92).abs() < 1e-9);
    }

    // ── Config switching ─────────────────────────────────────────────

    #[test]
    fn config_switching_applies_correctly() {
        let configs = vec![
            Config::new().with_tile_m(64),
            Config::new().with_tile_m(128),
        ];
        let hash_b = config_hash(&configs[1]);

        let mut selector = AdaptiveSelector::new(AdaptivePolicy::Aggressive, configs);
        assert_eq!(selector.current_config().tile_m, 64);

        let decision = SwitchDecision {
            should_switch: true,
            current_config_hash: selector.current_config_hash(),
            proposed_config_hash: hash_b,
            expected_improvement_pct: 10.0,
            confidence: 0.9,
            reason: "test".to_string(),
        };
        selector.apply_switch(&decision);
        assert_eq!(selector.current_config().tile_m, 128);
    }

    #[test]
    fn config_switching_no_op_when_false() {
        let configs = vec![
            Config::new().with_tile_m(64),
            Config::new().with_tile_m(128),
        ];
        let hash_b = config_hash(&configs[1]);

        let mut selector = AdaptiveSelector::new(AdaptivePolicy::Moderate, configs);
        let decision = SwitchDecision {
            should_switch: false,
            current_config_hash: selector.current_config_hash(),
            proposed_config_hash: hash_b,
            expected_improvement_pct: 0.0,
            confidence: 0.0,
            reason: "not enough evidence".to_string(),
        };
        selector.apply_switch(&decision);
        assert_eq!(selector.current_config().tile_m, 64);
    }

    // ── ExplorationScheduler round-robin ─────────────────────────────

    #[test]
    fn exploration_scheduler_round_robin() {
        let configs = vec![
            Config::new().with_tile_m(32),
            Config::new().with_tile_m(64),
            Config::new().with_tile_m(128),
        ];
        let mut sched = ExplorationScheduler::new(configs.clone(), 0.1);

        // Should round-robin through all 3 unexplored configs
        let first = sched.next_config().tile_m;
        let second = sched.next_config().tile_m;
        let third = sched.next_config().tile_m;

        // All three should be visited (order may vary)
        let mut seen = vec![first, second, third];
        seen.sort();
        assert_eq!(seen, vec![32, 64, 128]);
    }

    // ── Exploration budget exhaustion ────────────────────────────────

    #[test]
    fn exploration_budget_exhaustion() {
        let configs = vec![
            Config::new().with_tile_m(64),
            Config::new().with_tile_m(128),
        ];
        let hash_a = config_hash(&configs[0]);
        let hash_b = config_hash(&configs[1]);

        let mut selector = AdaptiveSelector::new(AdaptivePolicy::Aggressive, configs);
        assert!(selector.is_exploring());
        assert!(selector.exploration_budget() > 0);

        // Fill both windows to min_samples (20 for Aggressive)
        for _ in 0..20 {
            selector.record_metric(make_metric(hash_a, 100.0));
            selector.record_metric(make_metric(hash_b, 90.0));
        }

        assert!(!selector.is_exploring());
        assert_eq!(selector.exploration_budget(), 0);
    }

    // ── Performance regression detection ─────────────────────────────

    #[test]
    fn performance_regression_detection() {
        let mut w = MetricsWindow::new(20);
        // Baseline is 100us, push values around 130us (30% regression)
        for _ in 0..10 {
            w.push(make_metric(1, 130.0));
        }

        // With 20% threshold, should detect regression
        let reg = PerformanceRegression::detect(&w, 100.0, 20.0);
        assert!(reg.is_some());
        let reg = reg.expect("regression should be detected");
        assert!((reg.regression_pct - 30.0).abs() < 1e-9);
        assert_eq!(reg.samples_degraded, 10);

        // With 50% threshold, should NOT detect regression
        let no_reg = PerformanceRegression::detect(&w, 100.0, 50.0);
        assert!(no_reg.is_none());
    }

    #[test]
    fn performance_regression_empty_window() {
        let w = MetricsWindow::new(10);
        let reg = PerformanceRegression::detect(&w, 100.0, 10.0);
        assert!(reg.is_none());
    }

    // ── AdaptiveTuneReport statistics ────────────────────────────────

    #[test]
    fn report_statistics() {
        let configs = vec![
            Config::new().with_tile_m(64),
            Config::new().with_tile_m(128),
        ];
        let hash_a = config_hash(&configs[0]);
        let hash_b = config_hash(&configs[1]);

        let mut selector = AdaptiveSelector::new(AdaptivePolicy::Aggressive, configs);

        // Record some metrics
        for i in 0..25u64 {
            selector.record_metric(make_metric_with_ts(hash_a, 100.0, i * 10));
            selector.record_metric(make_metric_with_ts(hash_b, 80.0, i * 10 + 5));
        }

        let report = selector.report();
        assert_eq!(report.total_switches, 0);
        assert!(report.exploitation_runs > 0);
        assert!(report.exploration_runs > 0);
        assert!(!report.performance_history.is_empty());
        // Best config should be the one with 80us mean
        assert_eq!(report.best_config_hash, hash_b);
    }
}
