//! GPU benchmark engine for measuring kernel execution time.
//!
//! The [`BenchmarkEngine`] uses CUDA events for precise GPU-side timing.
//! It performs warmup iterations to stabilize GPU clock frequencies and
//! caches, then collects multiple timed samples and computes robust
//! statistics (median, min, max, standard deviation).
//!
//! # Typical usage
//!
//! ```rust,no_run
//! use oxicuda_autotune::{BenchmarkEngine, BenchmarkConfig, Config};
//! use oxicuda_driver::{Stream, Event};
//!
//! # fn example(stream: &Stream) -> Result<(), oxicuda_autotune::AutotuneError> {
//! let engine = BenchmarkEngine::with_config(BenchmarkConfig {
//!     warmup_runs: 3,
//!     benchmark_runs: 10,
//! });
//!
//! let config = Config::new();
//! let result = engine.benchmark(&stream, &config, Some(2.0e9), |s| {
//!     // Launch your kernel on stream `s` here.
//!     Ok(())
//! })?;
//!
//! println!("Median: {:.1} us, GFLOPS: {:.1}",
//!     result.median_us,
//!     result.gflops.unwrap_or(0.0));
//! # Ok(())
//! # }
//! ```

use oxicuda_driver::{Event, Stream};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::AutotuneError;

/// Result of benchmarking a single configuration.
///
/// All times are in microseconds.  GFLOPS and efficiency are
/// populated only when the caller provides a FLOP count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// The configuration that was benchmarked.
    pub config: Config,
    /// Median execution time in microseconds.
    pub median_us: f64,
    /// Minimum execution time in microseconds.
    pub min_us: f64,
    /// Maximum execution time in microseconds.
    pub max_us: f64,
    /// Standard deviation of execution times in microseconds.
    pub stddev_us: f64,
    /// Achieved GFLOPS (billions of floating-point operations per second).
    ///
    /// Populated only when the caller provides a FLOP count.
    pub gflops: Option<f64>,
    /// Efficiency vs. peak throughput (0.0–1.0).
    ///
    /// Populated only when the caller provides a FLOP count and the
    /// peak throughput is known.
    pub efficiency: Option<f64>,
}

/// Configuration for the benchmark engine.
///
/// Controls how many warmup and measurement iterations are performed.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations before measurement begins.
    ///
    /// Warmup stabilizes GPU clock frequencies and populates caches.
    /// Default: 5.
    pub warmup_runs: u32,
    /// Number of timed measurement iterations.
    ///
    /// More iterations produce more stable statistics at the cost of
    /// longer tuning time.  Default: 20.
    pub benchmark_runs: u32,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_runs: 5,
            benchmark_runs: 20,
        }
    }
}

/// GPU benchmark execution engine.
///
/// Measures kernel execution time using CUDA events for accurate
/// GPU-side timing (not wall-clock time).  The engine:
///
/// 1. Runs the kernel several times for warmup (not measured).
/// 2. Records CUDA events around each measurement iteration.
/// 3. Reads back elapsed times and computes statistics.
///
/// The `launch_fn` closure is responsible for enqueuing the kernel
/// onto the provided [`Stream`].  It will be called
/// `warmup_runs + benchmark_runs` times in total.
pub struct BenchmarkEngine {
    /// Benchmark configuration (warmup + measurement counts).
    config: BenchmarkConfig,
}

impl BenchmarkEngine {
    /// Creates a new benchmark engine with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    /// Creates a new benchmark engine with the given configuration.
    #[must_use]
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Returns the current benchmark configuration.
    #[must_use]
    pub fn config(&self) -> &BenchmarkConfig {
        &self.config
    }

    /// Benchmarks a kernel launch function and returns timing statistics.
    ///
    /// # Arguments
    ///
    /// * `stream` — The CUDA stream on which the kernel will be launched.
    /// * `config` — The tuning configuration being evaluated.
    /// * `flops` — Optional total floating-point operation count for
    ///   computing GFLOPS.
    /// * `launch_fn` — A closure that launches the kernel onto the
    ///   provided stream.  It must **not** synchronize the stream itself.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::Cuda`] if any CUDA operation fails, or
    /// [`AutotuneError::BenchmarkFailed`] if no valid timing samples
    /// could be collected.
    pub fn benchmark<F>(
        &self,
        stream: &Stream,
        config: &Config,
        flops: Option<f64>,
        launch_fn: F,
    ) -> Result<BenchmarkResult, AutotuneError>
    where
        F: Fn(&Stream) -> Result<(), oxicuda_driver::CudaError>,
    {
        // Phase 1: Warmup — run the kernel without timing.
        for _ in 0..self.config.warmup_runs {
            launch_fn(stream)?;
        }
        stream.synchronize()?;

        // Phase 2: Timed measurement — record events around each launch.
        let num_runs = self.config.benchmark_runs;
        if num_runs == 0 {
            return Err(AutotuneError::BenchmarkFailed(
                "benchmark_runs must be > 0".to_string(),
            ));
        }

        let mut times_us = Vec::with_capacity(num_runs as usize);

        for i in 0..num_runs {
            let start_event = Event::new()?;
            let end_event = Event::new()?;

            start_event.record(stream)?;
            launch_fn(stream).map_err(|e| {
                AutotuneError::BenchmarkFailed(format!("launch failed on iteration {i}: {e}"))
            })?;
            end_event.record(stream)?;
            end_event.synchronize()?;

            let elapsed_ms = Event::elapsed_time(&start_event, &end_event)?;
            times_us.push(f64::from(elapsed_ms) * 1000.0);
        }

        if times_us.is_empty() {
            return Err(AutotuneError::BenchmarkFailed(
                "no timing samples collected".to_string(),
            ));
        }

        let (median, min, max, stddev) = compute_stats(&times_us);

        // Compute GFLOPS if FLOP count is provided.
        let gflops = flops.map(|f| f / (median * 1e-6) / 1e9);

        Ok(BenchmarkResult {
            config: config.clone(),
            median_us: median,
            min_us: min,
            max_us: max,
            stddev_us: stddev,
            gflops,
            efficiency: None, // Requires peak throughput knowledge
        })
    }

    /// Benchmarks a kernel using wall-clock timing (no CUDA events).
    ///
    /// This is useful when CUDA events are not available or when
    /// benchmarking host-side overhead.  Less precise than event-based
    /// timing but works without a GPU.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] if no valid samples
    /// could be collected.
    pub fn benchmark_wallclock<F>(
        &self,
        config: &Config,
        flops: Option<f64>,
        run_fn: F,
    ) -> Result<BenchmarkResult, AutotuneError>
    where
        F: Fn() -> Result<(), AutotuneError>,
    {
        // Phase 1: Warmup
        for _ in 0..self.config.warmup_runs {
            run_fn()?;
        }

        // Phase 2: Timed measurement
        let num_runs = self.config.benchmark_runs;
        if num_runs == 0 {
            return Err(AutotuneError::BenchmarkFailed(
                "benchmark_runs must be > 0".to_string(),
            ));
        }

        let mut times_us = Vec::with_capacity(num_runs as usize);

        for _ in 0..num_runs {
            let start = std::time::Instant::now();
            run_fn()?;
            let elapsed = start.elapsed();
            times_us.push(elapsed.as_secs_f64() * 1_000_000.0);
        }

        if times_us.is_empty() {
            return Err(AutotuneError::BenchmarkFailed(
                "no timing samples collected".to_string(),
            ));
        }

        let (median, min, max, stddev) = compute_stats(&times_us);
        let gflops = flops.map(|f| f / (median * 1e-6) / 1e9);

        Ok(BenchmarkResult {
            config: config.clone(),
            median_us: median,
            min_us: min,
            max_us: max,
            stddev_us: stddev,
            gflops,
            efficiency: None,
        })
    }
}

impl Default for BenchmarkEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes median, minimum, maximum, and standard deviation from a
/// slice of timing samples.
///
/// The input slice must be non-empty.  Values are assumed to be in
/// microseconds but the function is unit-agnostic.
fn compute_stats(times: &[f64]) -> (f64, f64, f64, f64) {
    debug_assert!(!times.is_empty(), "compute_stats called with empty slice");

    let n = times.len() as f64;

    // Sort for median
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if sorted.len() % 2 == 0 {
        let mid = sorted.len() / 2;
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let min = sorted.first().copied().unwrap_or(0.0);
    let max = sorted.last().copied().unwrap_or(0.0);

    // Standard deviation (population)
    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    (median, min, max, stddev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_stats_odd_count() {
        let times = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (median, min, max, stddev) = compute_stats(&times);
        assert!((median - 30.0).abs() < 1e-9);
        assert!((min - 10.0).abs() < 1e-9);
        assert!((max - 50.0).abs() < 1e-9);
        // stddev of [10,20,30,40,50] = sqrt(200) ≈ 14.142
        assert!((stddev - 14.142_135_623_730_951).abs() < 1e-6);
    }

    #[test]
    fn compute_stats_even_count() {
        let times = vec![10.0, 20.0, 30.0, 40.0];
        let (median, min, max, _) = compute_stats(&times);
        assert!((median - 25.0).abs() < 1e-9);
        assert!((min - 10.0).abs() < 1e-9);
        assert!((max - 40.0).abs() < 1e-9);
    }

    #[test]
    fn compute_stats_single_value() {
        let times = vec![42.0];
        let (median, min, max, stddev) = compute_stats(&times);
        assert!((median - 42.0).abs() < 1e-9);
        assert!((min - 42.0).abs() < 1e-9);
        assert!((max - 42.0).abs() < 1e-9);
        assert!((stddev - 0.0).abs() < 1e-9);
    }

    #[test]
    fn benchmark_wallclock_smoke() {
        let engine = BenchmarkEngine::with_config(BenchmarkConfig {
            warmup_runs: 1,
            benchmark_runs: 3,
        });
        let cfg = Config::new();
        let result = engine
            .benchmark_wallclock(&cfg, Some(1e9), || Ok(()))
            .expect("wallclock benchmark should succeed");

        assert!(result.median_us >= 0.0);
        assert!(result.gflops.is_some());
    }

    #[test]
    fn benchmark_zero_runs_errors() {
        let engine = BenchmarkEngine::with_config(BenchmarkConfig {
            warmup_runs: 0,
            benchmark_runs: 0,
        });
        let cfg = Config::new();
        let result = engine.benchmark_wallclock(&cfg, None, || Ok(()));
        assert!(result.is_err());
    }

    #[test]
    fn test_variance_calculation_correctness() {
        // Identical values → stddev = 0, coefficient of variation = 0%
        let all_same = vec![1.0f64, 1.0, 1.0];
        let (median, _min, _max, stddev) = compute_stats(&all_same);
        assert!((median - 1.0).abs() < 1e-12);
        assert!(
            stddev.abs() < 1e-12,
            "stddev should be zero for identical values"
        );

        // [1.0, 1.1, 0.9]: mean=1.0, variance=((0)^2+(0.1)^2+(-0.1)^2)/3 = 0.02/3
        // stddev = sqrt(0.02/3) ≈ 0.08165
        let varying = vec![1.0f64, 1.1, 0.9];
        let (med2, _min2, _max2, stddev2) = compute_stats(&varying);
        assert!((med2 - 1.0).abs() < 1e-12);
        let expected_stddev = (0.02f64 / 3.0).sqrt();
        assert!(
            (stddev2 - expected_stddev).abs() < 1e-9,
            "stddev {stddev2} should ≈ {expected_stddev}"
        );

        // Five identical values of 10.0 → CV = 0%
        let uniform = vec![10.0f64, 10.0, 10.0, 10.0, 10.0];
        let (_med3, _min3, _max3, stddev3) = compute_stats(&uniform);
        assert!(stddev3.abs() < 1e-12, "CV should be 0% for uniform values");
    }

    #[test]
    fn test_gflops_formula() {
        // For M=N=K=1024: ops = 2 * 1024^3 = 2_147_483_648
        // If median_us = 1000 (1ms) → GFLOPS = 2_147_483_648 / 1000e-6 / 1e9 = 2147.48...
        let m: f64 = 1024.0;
        let n: f64 = 1024.0;
        let k: f64 = 1024.0;
        let flops: f64 = 2.0 * m * n * k;
        let median_us: f64 = 1000.0;

        // Replicate the formula used in BenchmarkEngine
        let gflops = flops / (median_us * 1e-6) / 1e9;
        assert!(
            (gflops - 2_147.483_648).abs() < 0.001,
            "gflops {gflops} should ≈ 2147.48"
        );

        // Verify via benchmark_wallclock with a mock that takes ~0 time.
        // We verify gflops is populated when flops is supplied.
        let engine = BenchmarkEngine::with_config(BenchmarkConfig {
            warmup_runs: 0,
            benchmark_runs: 5,
        });
        let cfg = Config::new();
        let result = engine
            .benchmark_wallclock(&cfg, Some(flops), || Ok(()))
            .expect("wallclock benchmark should succeed");
        assert!(
            result.gflops.is_some(),
            "gflops should be Some when flops is provided"
        );
        // The no-op run should produce some positive GFLOPS number
        assert!(
            result.gflops.unwrap_or(0.0) > 0.0,
            "gflops should be positive"
        );
    }
}
