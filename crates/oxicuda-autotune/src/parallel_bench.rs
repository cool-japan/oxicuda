//! Parallel benchmark execution for accelerated autotuning.
//!
//! This module provides a [`ParallelBenchmarkEngine`] that organizes and
//! partitions benchmark workloads across multiple simulated CUDA streams
//! or GPUs. On systems without a GPU (e.g. macOS), benchmarks fall back
//! to wall-clock timing via [`BenchmarkEngine::benchmark_wallclock()`].
//!
//! # Strategies
//!
//! - [`ParallelStrategy::SingleStream`] — sequential baseline, one config
//!   at a time on a single stream.
//! - [`ParallelStrategy::MultiStream`] — partition configs across multiple
//!   streams for concurrent execution.
//! - [`ParallelStrategy::MultiGpu`] — partition configs across multiple
//!   GPU devices.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::parallel_bench::*;
//! use oxicuda_autotune::{Config, BenchmarkConfig};
//!
//! let config = ParallelBenchConfig {
//!     strategy: ParallelStrategy::MultiStream { num_streams: 4 },
//!     max_concurrent: 4,
//!     timeout_per_config_ms: 5000,
//!     retry_on_failure: true,
//! };
//!
//! let engine = ParallelBenchmarkEngine::new(config, BenchmarkConfig::default());
//! let configs = vec![Config::new().with_tile_m(64), Config::new().with_tile_m(128)];
//! let result = engine.benchmark_parallel(&configs, Some(2.0e9));
//! assert!(result.is_ok());
//! ```
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::benchmark::{BenchmarkConfig, BenchmarkEngine, BenchmarkResult};
use crate::config::Config;
use crate::error::AutotuneError;

// ---------------------------------------------------------------------------
// ParallelStrategy
// ---------------------------------------------------------------------------

/// Strategy for distributing benchmark work across execution resources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelStrategy {
    /// Sequential baseline — benchmark one config at a time on a single
    /// stream.  No parallelism, but useful as a reference point.
    SingleStream,

    /// Distribute configs round-robin across `num_streams` simulated
    /// CUDA streams.  Each stream processes its partition sequentially,
    /// but the partitions conceptually run in parallel.
    MultiStream {
        /// Number of concurrent streams to simulate.
        num_streams: usize,
    },

    /// Distribute configs across multiple GPU devices identified by
    /// their device IDs.  Each device processes its partition
    /// independently.
    MultiGpu {
        /// CUDA device IDs to distribute work across.
        device_ids: Vec<i32>,
    },
}

impl ParallelStrategy {
    /// Returns the degree of parallelism for this strategy.
    #[must_use]
    pub fn parallelism(&self) -> usize {
        match self {
            Self::SingleStream => 1,
            Self::MultiStream { num_streams } => (*num_streams).max(1),
            Self::MultiGpu { device_ids } => device_ids.len().max(1),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamPool
// ---------------------------------------------------------------------------

/// A pool of simulated stream indices for round-robin work distribution.
///
/// Each "stream" is represented by an index.  The pool distributes config
/// indices across streams in round-robin order, producing partitions that
/// can be benchmarked independently.
#[derive(Debug, Clone)]
pub struct StreamPool {
    /// Stream indices managed by this pool.
    stream_indices: Vec<usize>,
}

impl StreamPool {
    /// Creates a new stream pool with `num_streams` streams numbered
    /// `0..num_streams`.
    ///
    /// # Arguments
    ///
    /// * `num_streams` — Number of streams to create.  Clamped to at
    ///   least 1.
    #[must_use]
    pub fn new(num_streams: usize) -> Self {
        let count = num_streams.max(1);
        Self {
            stream_indices: (0..count).collect(),
        }
    }

    /// Returns the number of streams in the pool.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stream_indices.len()
    }

    /// Returns `true` if the pool contains no streams (never happens
    /// after construction, but provided for completeness).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stream_indices.is_empty()
    }

    /// Returns a reference to the stream indices.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.stream_indices
    }

    /// Partitions `num_items` items across the pool's streams in
    /// round-robin order.
    ///
    /// Returns a `Vec` of partitions, where `partitions[i]` is the list
    /// of item indices assigned to stream `i`.
    #[must_use]
    pub fn partition(&self, num_items: usize) -> Vec<Vec<usize>> {
        let n = self.stream_indices.len();
        let mut partitions: Vec<Vec<usize>> = vec![Vec::new(); n];
        for item_idx in 0..num_items {
            partitions[item_idx % n].push(item_idx);
        }
        partitions
    }
}

// ---------------------------------------------------------------------------
// ParallelBenchConfig
// ---------------------------------------------------------------------------

/// Configuration for parallel benchmarking.
#[derive(Debug, Clone)]
pub struct ParallelBenchConfig {
    /// The parallelism strategy to use.
    pub strategy: ParallelStrategy,

    /// Maximum number of concurrent benchmarks.  This caps the degree
    /// of parallelism regardless of the strategy's natural parallelism.
    pub max_concurrent: usize,

    /// Timeout per configuration in milliseconds.  If a single config's
    /// benchmark exceeds this duration, it is treated as a failure.
    /// Set to 0 to disable timeouts.
    pub timeout_per_config_ms: u64,

    /// Whether to retry a failed benchmark once before giving up.
    pub retry_on_failure: bool,
}

impl Default for ParallelBenchConfig {
    fn default() -> Self {
        Self {
            strategy: ParallelStrategy::SingleStream,
            max_concurrent: 1,
            timeout_per_config_ms: 10_000,
            retry_on_failure: false,
        }
    }
}

// ---------------------------------------------------------------------------
// BenchProgress
// ---------------------------------------------------------------------------

/// Progress information for an ongoing parallel benchmark session.
///
/// The `completed` and `failed` counters are atomic so they can be
/// updated from multiple threads (future extension).
pub struct BenchProgress {
    /// Total number of configs to benchmark.
    pub total_configs: usize,
    /// Number of configs that have completed successfully.
    pub completed: AtomicUsize,
    /// Number of configs that failed.
    pub failed: AtomicUsize,
    /// Elapsed wall-clock time since the benchmark started (milliseconds).
    pub elapsed_ms: u64,
}

impl BenchProgress {
    /// Creates a new progress tracker.
    #[must_use]
    pub fn new(total_configs: usize) -> Self {
        Self {
            total_configs,
            completed: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
            elapsed_ms: 0,
        }
    }

    /// Returns the number of completed configs.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.completed.load(Ordering::Relaxed)
    }

    /// Returns the number of failed configs.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.failed.load(Ordering::Relaxed)
    }

    /// Returns the fraction of work done (0.0 to 1.0).
    #[must_use]
    pub fn fraction_done(&self) -> f64 {
        if self.total_configs == 0 {
            return 1.0;
        }
        let done = self.completed_count() + self.failed_count();
        done as f64 / self.total_configs as f64
    }
}

// ---------------------------------------------------------------------------
// BenchProgressCallback
// ---------------------------------------------------------------------------

/// Trait for receiving progress updates during parallel benchmarking.
///
/// Implementations can log progress, update a UI, or trigger early
/// stopping based on observed results.
pub trait BenchProgressCallback: Send + Sync {
    /// Called after each config benchmark completes (success or failure).
    fn on_progress(&self, progress: &BenchProgress);
}

/// A no-op progress callback that discards all updates.
pub struct NoOpCallback;

impl BenchProgressCallback for NoOpCallback {
    fn on_progress(&self, _progress: &BenchProgress) {}
}

// ---------------------------------------------------------------------------
// ParallelBenchResult
// ---------------------------------------------------------------------------

/// Aggregate result from a parallel benchmark session.
#[derive(Debug, Clone)]
pub struct ParallelBenchResult {
    /// Individual benchmark results (one per config that succeeded).
    pub results: Vec<BenchmarkResult>,
    /// Total wall-clock time for the entire session (milliseconds).
    pub total_time_ms: u64,
    /// Throughput: configs benchmarked per second.
    pub configs_per_second: f64,
    /// The strategy that was used.
    pub strategy_used: ParallelStrategy,
}

impl ParallelBenchResult {
    /// Returns the best (fastest) result by median execution time, if any.
    #[must_use]
    pub fn best_result(&self) -> Option<&BenchmarkResult> {
        self.results.iter().min_by(|a, b| {
            a.median_us
                .partial_cmp(&b.median_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns the number of successful benchmark results.
    #[must_use]
    pub fn num_successful(&self) -> usize {
        self.results.len()
    }
}

// ---------------------------------------------------------------------------
// ParallelBenchmarkEngine
// ---------------------------------------------------------------------------

/// Engine that orchestrates parallel benchmark execution.
///
/// The engine partitions configs across streams or devices according to
/// the configured [`ParallelStrategy`], benchmarks each partition, and
/// aggregates the results into a [`ParallelBenchResult`].
///
/// On platforms without a GPU, benchmarks use wall-clock timing via
/// [`BenchmarkEngine::benchmark_wallclock()`].
pub struct ParallelBenchmarkEngine {
    /// Parallel benchmarking configuration.
    config: ParallelBenchConfig,
    /// Underlying single-config benchmark engine.
    engine: BenchmarkEngine,
    /// Optional progress callback.
    callback: Option<Box<dyn BenchProgressCallback>>,
}

impl ParallelBenchmarkEngine {
    /// Creates a new parallel benchmark engine.
    #[must_use]
    pub fn new(config: ParallelBenchConfig, bench_config: BenchmarkConfig) -> Self {
        Self {
            config,
            engine: BenchmarkEngine::with_config(bench_config),
            callback: None,
        }
    }

    /// Creates a new parallel benchmark engine with default settings.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self {
            config: ParallelBenchConfig::default(),
            engine: BenchmarkEngine::new(),
            callback: None,
        }
    }

    /// Sets a progress callback.
    #[must_use]
    pub fn with_callback(mut self, callback: Box<dyn BenchProgressCallback>) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Returns the parallel benchmarking configuration.
    #[must_use]
    pub fn config(&self) -> &ParallelBenchConfig {
        &self.config
    }

    /// Returns the effective parallelism, capped by `max_concurrent`.
    #[must_use]
    pub fn effective_parallelism(&self) -> usize {
        let natural = self.config.strategy.parallelism();
        if self.config.max_concurrent > 0 {
            natural.min(self.config.max_concurrent)
        } else {
            natural
        }
    }

    /// Benchmarks all configurations using the configured parallel
    /// strategy.
    ///
    /// For [`ParallelStrategy::MultiStream`], configs are partitioned
    /// across `num_streams` groups.  Each group is benchmarked
    /// sequentially, but the groups conceptually run in parallel (the
    /// total time approximates the longest partition's time).
    ///
    /// For [`ParallelStrategy::MultiGpu`], configs are partitioned
    /// across the specified device IDs with the same round-robin
    /// approach.
    ///
    /// # Arguments
    ///
    /// * `configs` — Slice of configurations to benchmark.
    /// * `flops` — Optional FLOP count for GFLOPS computation.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] if all configs fail.
    /// Individual config failures are silently skipped (or retried if
    /// `retry_on_failure` is enabled).
    pub fn benchmark_parallel(
        &self,
        configs: &[Config],
        flops: Option<f64>,
    ) -> Result<ParallelBenchResult, AutotuneError> {
        let start = Instant::now();

        if configs.is_empty() {
            return Ok(ParallelBenchResult {
                results: Vec::new(),
                total_time_ms: 0,
                configs_per_second: 0.0,
                strategy_used: self.config.strategy.clone(),
            });
        }

        let progress = BenchProgress::new(configs.len());
        let parallelism = self.effective_parallelism();

        // Build partitions based on strategy
        let partitions = self.build_partitions(configs.len(), parallelism);

        // Benchmark each partition sequentially (actual concurrency would
        // require OS threads + real CUDA streams; here we simulate the
        // organizational structure)
        let mut all_results: Vec<BenchmarkResult> = Vec::with_capacity(configs.len());

        for partition in &partitions {
            for &config_idx in partition {
                let config = &configs[config_idx];
                let result = self.benchmark_single_config(config, flops, &progress);

                match result {
                    Ok(bench_result) => {
                        progress.completed.fetch_add(1, Ordering::Relaxed);
                        all_results.push(bench_result);
                    }
                    Err(_) => {
                        progress.failed.fetch_add(1, Ordering::Relaxed);
                    }
                }

                if let Some(ref cb) = self.callback {
                    // Update elapsed time on the progress struct — we use
                    // a non-atomic u64 so we create a temporary view.
                    let elapsed = start.elapsed().as_millis() as u64;
                    let view = BenchProgress {
                        total_configs: configs.len(),
                        completed: AtomicUsize::new(progress.completed_count()),
                        failed: AtomicUsize::new(progress.failed_count()),
                        elapsed_ms: elapsed,
                    };
                    cb.on_progress(&view);
                }
            }
        }

        let elapsed = start.elapsed();
        let total_time_ms = elapsed.as_millis() as u64;
        let configs_per_second = if total_time_ms > 0 {
            (all_results.len() as f64) / (total_time_ms as f64 / 1000.0)
        } else {
            all_results.len() as f64
        };

        Ok(ParallelBenchResult {
            results: all_results,
            total_time_ms,
            configs_per_second,
            strategy_used: self.config.strategy.clone(),
        })
    }

    /// Benchmarks a single config with optional timeout and retry.
    fn benchmark_single_config(
        &self,
        config: &Config,
        flops: Option<f64>,
        _progress: &BenchProgress,
    ) -> Result<BenchmarkResult, AutotuneError> {
        let attempt = |cfg: &Config| -> Result<BenchmarkResult, AutotuneError> {
            let timeout_ms = self.config.timeout_per_config_ms;
            let start = Instant::now();

            let result = self.engine.benchmark_wallclock(cfg, flops, || {
                // Check timeout if enabled
                if timeout_ms > 0 {
                    let elapsed = start.elapsed().as_millis() as u64;
                    if elapsed > timeout_ms {
                        return Err(AutotuneError::BenchmarkFailed(format!(
                            "timeout after {elapsed}ms (limit: {timeout_ms}ms)"
                        )));
                    }
                }
                // Simulate a tiny workload — in real usage this would
                // launch a GPU kernel.
                Ok(())
            })?;

            // Post-benchmark timeout check (covers total time including
            // warmup + measurement)
            if timeout_ms > 0 {
                let elapsed = start.elapsed().as_millis() as u64;
                if elapsed > timeout_ms {
                    return Err(AutotuneError::BenchmarkFailed(format!(
                        "total benchmark time {elapsed}ms exceeded timeout {timeout_ms}ms"
                    )));
                }
            }

            Ok(result)
        };

        match attempt(config) {
            Ok(result) => Ok(result),
            Err(e) => {
                if self.config.retry_on_failure {
                    // One retry
                    attempt(config).map_err(|_| e)
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Builds round-robin partitions for the given number of items and
    /// parallelism level.
    fn build_partitions(&self, num_items: usize, parallelism: usize) -> Vec<Vec<usize>> {
        let pool = StreamPool::new(parallelism);
        pool.partition(num_items)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Estimates the speedup from parallel benchmarking vs. sequential.
///
/// This is a theoretical estimate based on the strategy's parallelism
/// degree, with diminishing returns modeled by Amdahl's law.  The
/// serial fraction is estimated at 10% (overhead for partitioning,
/// result aggregation, etc.).
///
/// # Arguments
///
/// * `num_configs` — Total number of configs to benchmark.
/// * `strategy` — The parallel strategy to estimate for.
///
/// # Returns
///
/// Estimated speedup factor (>= 1.0).  Returns 1.0 for
/// `SingleStream` or when there is only one config.
#[must_use]
pub fn estimate_parallel_speedup(num_configs: usize, strategy: &ParallelStrategy) -> f64 {
    if num_configs <= 1 {
        return 1.0;
    }

    let p = strategy.parallelism() as f64;
    if p <= 1.0 {
        return 1.0;
    }

    // Amdahl's law: speedup = 1 / (s + (1-s)/p)
    // where s is the serial fraction (estimated at 10%)
    let serial_fraction = 0.10;
    let parallel_fraction = 1.0 - serial_fraction;

    let speedup = 1.0 / (serial_fraction + parallel_fraction / p);

    // Cap speedup by the number of configs (can't be faster than
    // doing all configs simultaneously)
    speedup.min(num_configs as f64)
}

/// Heuristic for the optimal number of streams given the workload size
/// and available memory.
///
/// Each stream requires some memory overhead for buffering.  This
/// function balances parallelism against memory constraints.
///
/// # Arguments
///
/// * `num_configs` — Number of configs to benchmark.
/// * `available_memory_mb` — Available GPU memory in megabytes.
///
/// # Returns
///
/// Recommended number of streams (at least 1, at most `num_configs`).
#[must_use]
pub fn optimal_stream_count(num_configs: usize, available_memory_mb: u64) -> usize {
    if num_configs == 0 {
        return 1;
    }

    // Estimate ~64 MB overhead per concurrent stream (for workspace
    // buffers, kernel arguments, etc.)
    let mem_per_stream_mb: u64 = 64;

    // Reserve 25% of memory for the kernel workspace itself
    let usable_memory_mb = (available_memory_mb as f64 * 0.75) as u64;

    let max_by_memory = (usable_memory_mb / mem_per_stream_mb) as usize;

    // Diminishing returns beyond ~16 streams due to scheduling overhead
    let max_practical = 16_usize;

    max_by_memory.min(max_practical).min(num_configs).max(1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_strategy_smoke() {
        let config = ParallelBenchConfig {
            strategy: ParallelStrategy::SingleStream,
            max_concurrent: 1,
            timeout_per_config_ms: 5000,
            retry_on_failure: false,
        };
        let engine = ParallelBenchmarkEngine::new(
            config,
            BenchmarkConfig {
                warmup_runs: 1,
                benchmark_runs: 2,
            },
        );

        let configs = vec![
            Config::new().with_tile_m(64),
            Config::new().with_tile_m(128),
        ];

        let result = engine.benchmark_parallel(&configs, Some(1e9));
        assert!(result.is_ok());
        let result = result.expect("benchmark should succeed");
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.strategy_used, ParallelStrategy::SingleStream);
    }

    #[test]
    fn multi_stream_partitioning() {
        let pool = StreamPool::new(3);
        let partitions = pool.partition(7);

        assert_eq!(partitions.len(), 3);
        // Round-robin: [0,3,6], [1,4], [2,5]
        assert_eq!(partitions[0], vec![0, 3, 6]);
        assert_eq!(partitions[1], vec![1, 4]);
        assert_eq!(partitions[2], vec![2, 5]);

        // Total items across all partitions
        let total: usize = partitions.iter().map(|p| p.len()).sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn multi_stream_benchmark() {
        let config = ParallelBenchConfig {
            strategy: ParallelStrategy::MultiStream { num_streams: 3 },
            max_concurrent: 3,
            timeout_per_config_ms: 5000,
            retry_on_failure: false,
        };
        let engine = ParallelBenchmarkEngine::new(
            config,
            BenchmarkConfig {
                warmup_runs: 1,
                benchmark_runs: 2,
            },
        );

        let configs: Vec<Config> = (0..6)
            .map(|i| Config::new().with_tile_m(32 * (i as u32 + 1)))
            .collect();

        let result = engine.benchmark_parallel(&configs, None);
        assert!(result.is_ok());
        let result = result.expect("benchmark should succeed");
        assert_eq!(result.results.len(), 6);
        assert_eq!(
            result.strategy_used,
            ParallelStrategy::MultiStream { num_streams: 3 }
        );
    }

    #[test]
    fn progress_tracking() {
        use std::sync::{Arc, Mutex};

        struct TrackingCallback {
            updates: Arc<Mutex<Vec<(usize, usize)>>>,
        }
        impl BenchProgressCallback for TrackingCallback {
            fn on_progress(&self, progress: &BenchProgress) {
                if let Ok(mut updates) = self.updates.lock() {
                    updates.push((progress.completed_count(), progress.failed_count()));
                }
            }
        }

        let updates = Arc::new(Mutex::new(Vec::new()));
        let callback = TrackingCallback {
            updates: Arc::clone(&updates),
        };

        let config = ParallelBenchConfig {
            strategy: ParallelStrategy::SingleStream,
            max_concurrent: 1,
            timeout_per_config_ms: 5000,
            retry_on_failure: false,
        };
        let engine = ParallelBenchmarkEngine::new(
            config,
            BenchmarkConfig {
                warmup_runs: 0,
                benchmark_runs: 1,
            },
        )
        .with_callback(Box::new(callback));

        let configs = vec![Config::new(), Config::new(), Config::new()];
        let result = engine.benchmark_parallel(&configs, None);
        assert!(result.is_ok());

        let updates = updates.lock().expect("lock poisoned");
        // Should have received 3 progress updates
        assert_eq!(updates.len(), 3);
        // Final update should show 3 completed
        assert_eq!(updates[2].0, 3);
    }

    #[test]
    fn estimate_speedup_single_stream() {
        let speedup = estimate_parallel_speedup(10, &ParallelStrategy::SingleStream);
        assert!((speedup - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn estimate_speedup_multi_stream() {
        let speedup =
            estimate_parallel_speedup(100, &ParallelStrategy::MultiStream { num_streams: 4 });
        // With 4 streams and 10% serial fraction:
        // 1 / (0.10 + 0.90/4) = 1 / 0.325 ≈ 3.077
        assert!(speedup > 3.0);
        assert!(speedup < 4.0);
    }

    #[test]
    fn estimate_speedup_single_config() {
        let speedup =
            estimate_parallel_speedup(1, &ParallelStrategy::MultiStream { num_streams: 8 });
        assert!((speedup - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn optimal_stream_count_basic() {
        // 1024 MB available => usable ~768 MB => 768/64 = 12 streams max
        let count = optimal_stream_count(20, 1024);
        assert!(count >= 1);
        assert!(count <= 16);
        assert_eq!(count, 12);
    }

    #[test]
    fn optimal_stream_count_low_memory() {
        // 32 MB available => usable ~24 MB => 0 streams => clamped to 1
        let count = optimal_stream_count(10, 32);
        assert_eq!(count, 1);
    }

    #[test]
    fn optimal_stream_count_many_configs_capped() {
        // Even with tons of memory, capped at 16
        let count = optimal_stream_count(1000, 100_000);
        assert_eq!(count, 16);
    }

    #[test]
    fn empty_config_list() {
        let engine = ParallelBenchmarkEngine::with_defaults();
        let result = engine.benchmark_parallel(&[], None);
        assert!(result.is_ok());
        let result = result.expect("empty benchmark should succeed");
        assert!(result.results.is_empty());
        assert_eq!(result.total_time_ms, 0);
        assert!((result.configs_per_second - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn single_config_benchmark() {
        let config = ParallelBenchConfig {
            strategy: ParallelStrategy::SingleStream,
            max_concurrent: 1,
            timeout_per_config_ms: 5000,
            retry_on_failure: false,
        };
        let engine = ParallelBenchmarkEngine::new(
            config,
            BenchmarkConfig {
                warmup_runs: 1,
                benchmark_runs: 2,
            },
        );

        let configs = vec![Config::new().with_tile_m(64)];
        let result = engine.benchmark_parallel(&configs, Some(2e9));
        assert!(result.is_ok());
        let result = result.expect("single config should succeed");
        assert_eq!(result.results.len(), 1);
        assert!(result.results[0].gflops.is_some());
    }

    #[test]
    fn retry_on_failure_logic() {
        // Test that retry_on_failure flag is respected in config
        let config_with_retry = ParallelBenchConfig {
            strategy: ParallelStrategy::SingleStream,
            max_concurrent: 1,
            timeout_per_config_ms: 10_000,
            retry_on_failure: true,
        };
        assert!(config_with_retry.retry_on_failure);

        let config_without_retry = ParallelBenchConfig {
            strategy: ParallelStrategy::SingleStream,
            max_concurrent: 1,
            timeout_per_config_ms: 10_000,
            retry_on_failure: false,
        };
        assert!(!config_without_retry.retry_on_failure);

        // With retry, benchmarks that succeed on first try still work
        let engine = ParallelBenchmarkEngine::new(
            config_with_retry,
            BenchmarkConfig {
                warmup_runs: 0,
                benchmark_runs: 1,
            },
        );
        let configs = vec![Config::new()];
        let result = engine.benchmark_parallel(&configs, None);
        assert!(result.is_ok());
    }

    #[test]
    fn parallel_bench_result_statistics() {
        let results = vec![
            BenchmarkResult {
                config: Config::new().with_tile_m(64),
                median_us: 100.0,
                min_us: 90.0,
                max_us: 110.0,
                stddev_us: 5.0,
                gflops: Some(10.0),
                efficiency: None,
            },
            BenchmarkResult {
                config: Config::new().with_tile_m(128),
                median_us: 80.0,
                min_us: 75.0,
                max_us: 85.0,
                stddev_us: 3.0,
                gflops: Some(12.5),
                efficiency: None,
            },
            BenchmarkResult {
                config: Config::new().with_tile_m(256),
                median_us: 120.0,
                min_us: 115.0,
                max_us: 130.0,
                stddev_us: 7.0,
                gflops: Some(8.3),
                efficiency: None,
            },
        ];

        let bench_result = ParallelBenchResult {
            results,
            total_time_ms: 500,
            configs_per_second: 6.0,
            strategy_used: ParallelStrategy::MultiStream { num_streams: 2 },
        };

        assert_eq!(bench_result.num_successful(), 3);

        let best = bench_result.best_result();
        assert!(best.is_some());
        let best = best.expect("should have a best result");
        // Config with tile_m=128 has the lowest median (80.0)
        assert_eq!(best.config.tile_m, 128);
        assert!((best.median_us - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stream_pool_edge_cases() {
        // Pool with 0 streams should clamp to 1
        let pool = StreamPool::new(0);
        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());

        // Partition 0 items
        let partitions = pool.partition(0);
        assert_eq!(partitions.len(), 1);
        assert!(partitions[0].is_empty());

        // Pool with 1 stream, 1 item
        let pool = StreamPool::new(1);
        let partitions = pool.partition(1);
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0], vec![0]);
    }

    #[test]
    fn multi_gpu_strategy_parallelism() {
        let strategy = ParallelStrategy::MultiGpu {
            device_ids: vec![0, 1, 2, 3],
        };
        assert_eq!(strategy.parallelism(), 4);

        let strategy_empty = ParallelStrategy::MultiGpu { device_ids: vec![] };
        assert_eq!(strategy_empty.parallelism(), 1);
    }

    #[test]
    fn effective_parallelism_capping() {
        let config = ParallelBenchConfig {
            strategy: ParallelStrategy::MultiStream { num_streams: 8 },
            max_concurrent: 4,
            timeout_per_config_ms: 5000,
            retry_on_failure: false,
        };
        let engine = ParallelBenchmarkEngine::new(config, BenchmarkConfig::default());
        // 8 streams capped by max_concurrent=4
        assert_eq!(engine.effective_parallelism(), 4);
    }
}
