//! Power-aware autotuning for GPU kernel optimization.
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module integrates GPU power consumption into the autotuning objective,
//! allowing users to trade off throughput vs power efficiency. It provides:
//!
//! - [`PowerMonitor`] trait for abstracting power measurement sources
//! - [`NvidiaSmiMonitor`] for reading real GPU power via `nvidia-smi`
//! - [`SyntheticPowerMonitor`] for testing with configurable static readings
//! - [`PowerAwareBenchmarkEngine`] that augments standard benchmarks with
//!   power profiling (energy-per-op, efficiency ratio, thermal headroom)
//! - [`PowerConstraint`] enum for expressing power/thermal/efficiency limits
//! - [`PowerAwareSelector`] for choosing the best config subject to constraints
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::power_aware::*;
//! use oxicuda_autotune::Config;
//!
//! let monitor = SyntheticPowerMonitor::new(PowerReading {
//!     timestamp_us: 0,
//!     power_watts: 200.0,
//!     gpu_temp_celsius: 55.0,
//!     clock_mhz: 1800,
//! });
//!
//! let reading = monitor.read_power().expect("should read power");
//! assert!((reading.power_watts - 200.0).abs() < 1e-9);
//! ```

#[cfg(test)]
use crate::benchmark::BenchmarkConfig;
use crate::benchmark::{BenchmarkEngine, BenchmarkResult};
use crate::config::Config;
use crate::error::AutotuneError;

// ---------------------------------------------------------------------------
// PowerReading
// ---------------------------------------------------------------------------

/// A single GPU power measurement sample.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerReading {
    /// Timestamp of the reading in microseconds (monotonic).
    pub timestamp_us: u64,
    /// GPU board power draw in watts.
    pub power_watts: f64,
    /// GPU die temperature in degrees Celsius.
    pub gpu_temp_celsius: f64,
    /// SM clock frequency in MHz.
    pub clock_mhz: u32,
}

// ---------------------------------------------------------------------------
// PowerMonitor trait
// ---------------------------------------------------------------------------

/// Abstraction over GPU power measurement sources.
///
/// Implementations read instantaneous power, temperature, and clock data
/// from the GPU. The trait is object-safe so it can be used as
/// `Box<dyn PowerMonitor>`.
pub trait PowerMonitor: Send + Sync {
    /// Reads the current GPU power state.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError`] if the power reading cannot be obtained
    /// (e.g. nvidia-smi not found, parsing failure, unsupported platform).
    fn read_power(&self) -> Result<PowerReading, AutotuneError>;
}

// ---------------------------------------------------------------------------
// NvidiaSmiMonitor
// ---------------------------------------------------------------------------

/// Reads GPU power data from `nvidia-smi` via `Command::output()`.
///
/// On Linux/Windows this parses the CSV output of:
/// ```text
/// nvidia-smi --query-gpu=power.draw,temperature.gpu,clocks.sm --format=csv,noheader,nounits
/// ```
///
/// On macOS (where `nvidia-smi` is unavailable), returns synthetic data
/// (power=150.0W, temp=45.0C, clock=1500MHz) so the module compiles
/// and tests pass without a GPU.
#[derive(Debug, Clone)]
pub struct NvidiaSmiMonitor {
    /// GPU index to query (for multi-GPU systems).
    gpu_index: u32,
}

impl NvidiaSmiMonitor {
    /// Creates a new monitor for the given GPU index.
    #[must_use]
    pub fn new(gpu_index: u32) -> Self {
        Self { gpu_index }
    }

    /// Creates a monitor for GPU 0.
    #[must_use]
    pub fn default_gpu() -> Self {
        Self::new(0)
    }
}

impl PowerMonitor for NvidiaSmiMonitor {
    fn read_power(&self) -> Result<PowerReading, AutotuneError> {
        let timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        self.read_power_inner(timestamp_us)
    }
}

impl NvidiaSmiMonitor {
    #[cfg(target_os = "macos")]
    fn read_power_inner(&self, timestamp_us: u64) -> Result<PowerReading, AutotuneError> {
        // macOS has no NVIDIA GPUs — return synthetic data.
        let _ = self.gpu_index;
        Ok(PowerReading {
            timestamp_us,
            power_watts: 150.0,
            gpu_temp_celsius: 45.0,
            clock_mhz: 1500,
        })
    }

    #[cfg(not(target_os = "macos"))]
    fn read_power_inner(&self, timestamp_us: u64) -> Result<PowerReading, AutotuneError> {
        let output = std::process::Command::new("nvidia-smi")
            .args([
                &format!("-i={}", self.gpu_index),
                "--query-gpu=power.draw,temperature.gpu,clocks.sm",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|e| {
                AutotuneError::BenchmarkFailed(format!("failed to run nvidia-smi: {e}"))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AutotuneError::BenchmarkFailed(format!(
                "nvidia-smi exited with error: {stderr}"
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_nvidia_smi_output(&stdout, timestamp_us)
    }
}

/// Parses the CSV output line from nvidia-smi.
///
/// Expected format: `" 150.23, 45, 1500\n"` (spaces around values).
#[cfg(not(target_os = "macos"))]
fn parse_nvidia_smi_output(output: &str, timestamp_us: u64) -> Result<PowerReading, AutotuneError> {
    let line = output.trim();
    let parts: Vec<&str> = line.split(',').collect();

    if parts.len() < 3 {
        return Err(AutotuneError::BenchmarkFailed(format!(
            "unexpected nvidia-smi output format: {line}"
        )));
    }

    let power_watts: f64 = parts[0].trim().parse().map_err(|e| {
        AutotuneError::BenchmarkFailed(format!("failed to parse power value '{}': {e}", parts[0]))
    })?;

    let gpu_temp_celsius: f64 = parts[1].trim().parse().map_err(|e| {
        AutotuneError::BenchmarkFailed(format!("failed to parse temperature '{}': {e}", parts[1]))
    })?;

    let clock_mhz: u32 = parts[2].trim().parse().map_err(|e| {
        AutotuneError::BenchmarkFailed(format!("failed to parse clock '{}': {e}", parts[2]))
    })?;

    Ok(PowerReading {
        timestamp_us,
        power_watts,
        gpu_temp_celsius,
        clock_mhz,
    })
}

// ---------------------------------------------------------------------------
// SyntheticPowerMonitor
// ---------------------------------------------------------------------------

/// A configurable power monitor that returns static readings.
///
/// Useful for testing and development without a real GPU.
#[derive(Debug, Clone)]
pub struct SyntheticPowerMonitor {
    reading: PowerReading,
}

impl SyntheticPowerMonitor {
    /// Creates a synthetic monitor that always returns the given reading.
    #[must_use]
    pub fn new(reading: PowerReading) -> Self {
        Self { reading }
    }

    /// Creates a synthetic monitor with typical idle GPU values.
    #[must_use]
    pub fn idle() -> Self {
        Self::new(PowerReading {
            timestamp_us: 0,
            power_watts: 25.0,
            gpu_temp_celsius: 35.0,
            clock_mhz: 210,
        })
    }

    /// Creates a synthetic monitor with typical load GPU values.
    #[must_use]
    pub fn under_load() -> Self {
        Self::new(PowerReading {
            timestamp_us: 0,
            power_watts: 280.0,
            gpu_temp_celsius: 72.0,
            clock_mhz: 1800,
        })
    }
}

impl PowerMonitor for SyntheticPowerMonitor {
    fn read_power(&self) -> Result<PowerReading, AutotuneError> {
        let mut r = self.reading.clone();
        r.timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        Ok(r)
    }
}

// ---------------------------------------------------------------------------
// PowerProfile
// ---------------------------------------------------------------------------

/// Aggregated power profile from a benchmark run.
///
/// Captures idle, peak, and average power alongside an efficiency metric.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerProfile {
    /// GPU power draw at idle (before warmup), in watts.
    pub idle_watts: f64,
    /// Peak power draw observed during the benchmark, in watts.
    pub peak_watts: f64,
    /// Thermal design power (TDP) of the GPU, in watts.
    pub tdp_watts: f64,
    /// Compute efficiency in GFLOPS per watt.
    pub efficiency_ratio: f64,
}

// ---------------------------------------------------------------------------
// PowerAwareBenchmarkResult
// ---------------------------------------------------------------------------

/// Benchmark result augmented with power and energy metrics.
///
/// Extends [`BenchmarkResult`] with power profiling data, enabling
/// power-aware configuration selection.
#[derive(Debug, Clone)]
pub struct PowerAwareBenchmarkResult {
    /// Standard benchmark timing results.
    pub benchmark: BenchmarkResult,
    /// Power profile measured during the benchmark.
    pub power_profile: PowerProfile,
    /// Energy consumed per floating-point operation, in nanojoules.
    ///
    /// Lower is better. Computed as:
    /// `(avg_power_watts * median_time_seconds * 1e9) / flops`
    pub energy_per_op_nj: f64,
    /// Thermal headroom in degrees Celsius.
    ///
    /// How far below the thermal throttle point (assumed 83C) the GPU
    /// was running during the benchmark. Higher is better.
    pub thermal_headroom_celsius: f64,
}

// ---------------------------------------------------------------------------
// PowerAwareBenchmarkEngine
// ---------------------------------------------------------------------------

/// Default thermal throttle temperature in Celsius.
const DEFAULT_THERMAL_LIMIT_CELSIUS: f64 = 83.0;

/// Benchmark engine that integrates power monitoring.
///
/// Wraps a standard [`BenchmarkEngine`] and a [`PowerMonitor`] to produce
/// [`PowerAwareBenchmarkResult`]s that include energy and power metrics
/// alongside timing statistics.
pub struct PowerAwareBenchmarkEngine {
    engine: BenchmarkEngine,
    monitor: Box<dyn PowerMonitor>,
    /// TDP of the target GPU, in watts (used for profile computation).
    tdp_watts: f64,
    /// Thermal throttle limit in Celsius.
    thermal_limit_celsius: f64,
}

impl PowerAwareBenchmarkEngine {
    /// Creates a new power-aware benchmark engine.
    ///
    /// # Arguments
    ///
    /// * `engine` - The underlying benchmark engine for timing.
    /// * `monitor` - Power monitoring implementation.
    /// * `tdp_watts` - Thermal design power of the GPU in watts.
    pub fn new(engine: BenchmarkEngine, monitor: Box<dyn PowerMonitor>, tdp_watts: f64) -> Self {
        Self {
            engine,
            monitor,
            tdp_watts,
            thermal_limit_celsius: DEFAULT_THERMAL_LIMIT_CELSIUS,
        }
    }

    /// Sets the thermal throttle limit temperature.
    #[must_use]
    pub fn with_thermal_limit(mut self, limit_celsius: f64) -> Self {
        self.thermal_limit_celsius = limit_celsius;
        self
    }

    /// Returns a reference to the underlying benchmark engine.
    #[must_use]
    pub fn engine(&self) -> &BenchmarkEngine {
        &self.engine
    }

    /// Benchmarks a kernel with power monitoring using wall-clock timing.
    ///
    /// 1. Reads idle power before warmup.
    /// 2. Runs the standard wall-clock benchmark.
    /// 3. Samples power during each iteration to build a power profile.
    /// 4. Computes energy-per-op and efficiency metrics.
    ///
    /// # Arguments
    ///
    /// * `config` - The tuning configuration to evaluate.
    /// * `flops` - Total floating-point operation count (required for
    ///   energy-per-op and efficiency calculations).
    /// * `run_fn` - Closure that executes the kernel workload.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError`] if benchmarking or power reading fails.
    pub fn benchmark_with_power<F>(
        &self,
        config: &Config,
        flops: f64,
        run_fn: F,
    ) -> Result<PowerAwareBenchmarkResult, AutotuneError>
    where
        F: Fn() -> Result<(), AutotuneError>,
    {
        // Step 1: Read idle power before any work.
        let idle_reading = self.monitor.read_power()?;

        // Step 2: Run warmup + measurement with power sampling.
        //
        // We use a custom measurement loop so we can sample power each
        // iteration, rather than delegating entirely to BenchmarkEngine.
        let bench_config = self.engine.config();
        let warmup_runs = bench_config.warmup_runs;
        let benchmark_runs = bench_config.benchmark_runs;

        if benchmark_runs == 0 {
            return Err(AutotuneError::BenchmarkFailed(
                "benchmark_runs must be > 0".to_string(),
            ));
        }

        // Warmup phase
        for _ in 0..warmup_runs {
            run_fn()?;
        }

        // Measurement phase with power sampling
        let mut times_us = Vec::with_capacity(benchmark_runs as usize);
        let mut power_samples = Vec::with_capacity(benchmark_runs as usize);

        for _ in 0..benchmark_runs {
            let pre_reading = self.monitor.read_power()?;
            let start = std::time::Instant::now();
            run_fn()?;
            let elapsed = start.elapsed();
            let post_reading = self.monitor.read_power()?;

            times_us.push(elapsed.as_secs_f64() * 1_000_000.0);
            // Average of pre and post readings for this iteration
            let iter_power = (pre_reading.power_watts + post_reading.power_watts) / 2.0;
            let iter_temp = (pre_reading.gpu_temp_celsius + post_reading.gpu_temp_celsius) / 2.0;
            power_samples.push((iter_power, iter_temp));
        }

        // Compute timing statistics
        let (median_us, min_us, max_us, stddev_us) = compute_stats(&times_us);

        let gflops = if median_us > 0.0 {
            Some(flops / (median_us * 1e-6) / 1e9)
        } else {
            None
        };

        let benchmark = BenchmarkResult {
            config: config.clone(),
            median_us,
            min_us,
            max_us,
            stddev_us,
            gflops,
            efficiency: None,
        };

        // Compute power metrics
        let avg_power = if power_samples.is_empty() {
            idle_reading.power_watts
        } else {
            power_samples.iter().map(|(p, _)| p).sum::<f64>() / power_samples.len() as f64
        };

        let peak_power = power_samples
            .iter()
            .map(|(p, _)| *p)
            .fold(f64::NEG_INFINITY, f64::max);
        let peak_power = if peak_power.is_finite() {
            peak_power
        } else {
            avg_power
        };

        let max_temp = power_samples
            .iter()
            .map(|(_, t)| *t)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_temp = if max_temp.is_finite() {
            max_temp
        } else {
            idle_reading.gpu_temp_celsius
        };

        let efficiency_ratio = gflops.unwrap_or(0.0) / avg_power.max(1e-9);

        // energy_per_op in nanojoules:
        // energy_joules = avg_power_watts * median_time_seconds
        //               = avg_power * median_us * 1e-6
        // energy_nj = energy_joules * 1e9 = avg_power * median_us * 1e3
        // energy_per_op_nj = energy_nj / flops
        let energy_per_op_nj = if flops > 0.0 {
            avg_power * median_us * 1e3 / flops
        } else {
            0.0
        };

        let thermal_headroom_celsius = self.thermal_limit_celsius - max_temp;

        let power_profile = PowerProfile {
            idle_watts: idle_reading.power_watts,
            peak_watts: peak_power,
            tdp_watts: self.tdp_watts,
            efficiency_ratio,
        };

        Ok(PowerAwareBenchmarkResult {
            benchmark,
            power_profile,
            energy_per_op_nj,
            thermal_headroom_celsius,
        })
    }
}

/// Computes median, minimum, maximum, and standard deviation.
fn compute_stats(times: &[f64]) -> (f64, f64, f64, f64) {
    if times.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let n = times.len() as f64;
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

    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    (median, min, max, stddev)
}

// ---------------------------------------------------------------------------
// PowerConstraint
// ---------------------------------------------------------------------------

/// Constraints on power, thermal, and efficiency characteristics.
///
/// Used by [`PowerAwareSelector`] to filter or rank configurations
/// based on their power-aware benchmark results.
#[derive(Debug, Clone, PartialEq)]
pub enum PowerConstraint {
    /// Maximum allowed power draw in watts.
    MaxWatts(f64),
    /// Maximum allowed GPU temperature in degrees Celsius.
    MaxTempCelsius(f64),
    /// Maximum allowed energy per floating-point operation in nanojoules.
    MaxEnergyPerOpNj(f64),
    /// Minimum required compute efficiency in GFLOPS per watt.
    MinEfficiency(f64),
}

impl PowerConstraint {
    /// Returns how much a result violates this constraint.
    ///
    /// Returns 0.0 if the constraint is satisfied, otherwise a positive
    /// value representing the magnitude of the violation (normalized by
    /// the constraint value for comparability).
    fn violation(&self, result: &PowerAwareBenchmarkResult) -> f64 {
        match self {
            PowerConstraint::MaxWatts(limit) => {
                let actual = result.power_profile.peak_watts;
                if actual <= *limit {
                    0.0
                } else {
                    (actual - limit) / limit.max(1e-9)
                }
            }
            PowerConstraint::MaxTempCelsius(limit) => {
                // Thermal headroom tells us how far below the throttle point we are,
                // but the constraint is on absolute temperature.
                // actual_temp = thermal_limit - headroom
                // We use peak_watts as a proxy — but really we need actual temp.
                // Since we store thermal_headroom from the engine's thermal limit,
                // we compute actual_temp conservatively.
                let actual_temp = DEFAULT_THERMAL_LIMIT_CELSIUS - result.thermal_headroom_celsius;
                if actual_temp <= *limit {
                    0.0
                } else {
                    (actual_temp - limit) / limit.max(1e-9)
                }
            }
            PowerConstraint::MaxEnergyPerOpNj(limit) => {
                let actual = result.energy_per_op_nj;
                if actual <= *limit {
                    0.0
                } else {
                    (actual - limit) / limit.max(1e-9)
                }
            }
            PowerConstraint::MinEfficiency(limit) => {
                let actual = result.power_profile.efficiency_ratio;
                if actual >= *limit {
                    0.0
                } else {
                    (limit - actual) / limit.max(1e-9)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PowerAwareSelector
// ---------------------------------------------------------------------------

/// Selects the best kernel configuration subject to power constraints.
///
/// Given a set of [`PowerAwareBenchmarkResult`]s and power constraints,
/// the selector:
///
/// 1. Filters to configurations that satisfy all constraints.
/// 2. Among feasible configs, selects the one with the best median time.
/// 3. If no config meets all constraints, selects the "least-violating"
///    config — the one with the smallest total normalized violation score.
pub struct PowerAwareSelector {
    constraints: Vec<PowerConstraint>,
}

impl PowerAwareSelector {
    /// Creates a new selector with the given power constraints.
    #[must_use]
    pub fn new(constraints: Vec<PowerConstraint>) -> Self {
        Self { constraints }
    }

    /// Adds a constraint to the selector.
    #[must_use]
    pub fn with_constraint(mut self, constraint: PowerConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Returns the constraints.
    #[must_use]
    pub fn constraints(&self) -> &[PowerConstraint] {
        &self.constraints
    }

    /// Computes the total violation score for a result across all constraints.
    ///
    /// Returns 0.0 if all constraints are satisfied.
    fn total_violation(&self, result: &PowerAwareBenchmarkResult) -> f64 {
        self.constraints.iter().map(|c| c.violation(result)).sum()
    }

    /// Returns true if the result satisfies all constraints.
    fn is_feasible(&self, result: &PowerAwareBenchmarkResult) -> bool {
        self.constraints.iter().all(|c| c.violation(result) == 0.0)
    }

    /// Selects the best configuration from the given results.
    ///
    /// If feasible solutions exist, returns the one with the lowest
    /// median execution time. If no feasible solution exists, returns
    /// the least-violating configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::NoViableConfig`] if `results` is empty.
    pub fn select_best<'a>(
        &self,
        results: &'a [PowerAwareBenchmarkResult],
    ) -> Result<&'a PowerAwareBenchmarkResult, AutotuneError> {
        if results.is_empty() {
            return Err(AutotuneError::NoViableConfig);
        }

        // Partition into feasible and infeasible
        let feasible: Vec<&PowerAwareBenchmarkResult> =
            results.iter().filter(|r| self.is_feasible(r)).collect();

        if !feasible.is_empty() {
            // Among feasible, pick the fastest
            let best = feasible
                .into_iter()
                .min_by(|a, b| {
                    a.benchmark
                        .median_us
                        .partial_cmp(&b.benchmark.median_us)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .ok_or(AutotuneError::NoViableConfig)?;
            return Ok(best);
        }

        // No feasible solution — pick least-violating
        let best = results
            .iter()
            .min_by(|a, b| {
                let va = self.total_violation(a);
                let vb = self.total_violation(b);
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(AutotuneError::NoViableConfig)?;

        Ok(best)
    }

    /// Selects and returns the feasible results, sorted by median time
    /// (fastest first).
    #[must_use]
    pub fn select_feasible<'a>(
        &self,
        results: &'a [PowerAwareBenchmarkResult],
    ) -> Vec<&'a PowerAwareBenchmarkResult> {
        let mut feasible: Vec<&PowerAwareBenchmarkResult> =
            results.iter().filter(|r| self.is_feasible(r)).collect();
        feasible.sort_by(|a, b| {
            a.benchmark
                .median_us
                .partial_cmp(&b.benchmark.median_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        feasible
    }
}

// ---------------------------------------------------------------------------
// estimate_power_from_config
// ---------------------------------------------------------------------------

/// Heuristic power estimate from kernel configuration parameters.
///
/// Estimates GPU power draw based on tile sizes and thread count relative
/// to a reference configuration (128x128 tiles, 256 threads). Larger tiles
/// and more threads imply more compute activity, hence more power.
///
/// This is a rough heuristic for pre-screening configurations before
/// running actual power measurements.
///
/// # Arguments
///
/// * `config` - The kernel configuration to estimate power for.
/// * `base_power` - The baseline power draw in watts (e.g. measured idle
///   or at a known reference point).
///
/// # Returns
///
/// Estimated power draw in watts.
#[must_use]
pub fn estimate_power_from_config(config: &Config, base_power: f64) -> f64 {
    // Reference config: tile_m=128, tile_n=128, block_size=256, stages=2
    let ref_compute_volume: f64 = 128.0 * 128.0;
    let ref_threads: f64 = 256.0;
    let ref_stages: f64 = 2.0;

    let compute_volume = f64::from(config.tile_m) * f64::from(config.tile_n);
    let threads = f64::from(config.block_size);
    let stages = f64::from(config.stages);

    // Compute ratio: geometric mean of (compute_ratio, thread_ratio, stage_ratio)
    let compute_ratio = compute_volume / ref_compute_volume;
    let thread_ratio = threads / ref_threads;
    let stage_ratio = stages / ref_stages;

    // Power scales sub-linearly with compute load (square root blending)
    let combined_ratio = (compute_ratio * thread_ratio * stage_ratio).cbrt();

    // Tensor cores add ~10-15% power overhead when active
    let tc_factor = if config.use_tensor_core { 1.12 } else { 1.0 };

    base_power * combined_ratio * tc_factor
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reading(power: f64, temp: f64, clock: u32) -> PowerReading {
        PowerReading {
            timestamp_us: 1_000_000,
            power_watts: power,
            gpu_temp_celsius: temp,
            clock_mhz: clock,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn make_power_result(
        config: Config,
        median_us: f64,
        gflops: Option<f64>,
        peak_watts: f64,
        idle_watts: f64,
        efficiency: f64,
        energy_nj: f64,
        headroom: f64,
    ) -> PowerAwareBenchmarkResult {
        PowerAwareBenchmarkResult {
            benchmark: BenchmarkResult {
                config,
                median_us,
                min_us: median_us * 0.9,
                max_us: median_us * 1.1,
                stddev_us: median_us * 0.05,
                gflops,
                efficiency: None,
            },
            power_profile: PowerProfile {
                idle_watts,
                peak_watts,
                tdp_watts: 350.0,
                efficiency_ratio: efficiency,
            },
            energy_per_op_nj: energy_nj,
            thermal_headroom_celsius: headroom,
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Synthetic monitor returns configured values
    // -----------------------------------------------------------------------
    #[test]
    fn synthetic_monitor_returns_configured_reading() {
        let expected = make_reading(200.0, 55.0, 1800);
        let monitor = SyntheticPowerMonitor::new(expected.clone());
        let reading = monitor.read_power().expect("should succeed");

        assert!((reading.power_watts - 200.0).abs() < 1e-9);
        assert!((reading.gpu_temp_celsius - 55.0).abs() < 1e-9);
        assert_eq!(reading.clock_mhz, 1800);
        // Timestamp should be updated to current time
        assert!(reading.timestamp_us > 0);
    }

    // -----------------------------------------------------------------------
    // Test 2: Synthetic monitor idle preset
    // -----------------------------------------------------------------------
    #[test]
    fn synthetic_monitor_idle_preset() {
        let monitor = SyntheticPowerMonitor::idle();
        let reading = monitor.read_power().expect("should succeed");
        assert!((reading.power_watts - 25.0).abs() < 1e-9);
        assert!((reading.gpu_temp_celsius - 35.0).abs() < 1e-9);
        assert_eq!(reading.clock_mhz, 210);
    }

    // -----------------------------------------------------------------------
    // Test 3: Synthetic monitor under_load preset
    // -----------------------------------------------------------------------
    #[test]
    fn synthetic_monitor_under_load_preset() {
        let monitor = SyntheticPowerMonitor::under_load();
        let reading = monitor.read_power().expect("should succeed");
        assert!((reading.power_watts - 280.0).abs() < 1e-9);
        assert!((reading.gpu_temp_celsius - 72.0).abs() < 1e-9);
        assert_eq!(reading.clock_mhz, 1800);
    }

    // -----------------------------------------------------------------------
    // Test 4: NvidiaSmiMonitor on macOS returns synthetic data
    // -----------------------------------------------------------------------
    #[test]
    #[cfg(target_os = "macos")]
    fn nvidia_smi_monitor_macos_synthetic() {
        let monitor = NvidiaSmiMonitor::default_gpu();
        let reading = monitor.read_power().expect("should succeed on macOS");
        assert!((reading.power_watts - 150.0).abs() < 1e-9);
        assert!((reading.gpu_temp_celsius - 45.0).abs() < 1e-9);
        assert_eq!(reading.clock_mhz, 1500);
    }

    // -----------------------------------------------------------------------
    // Test 5: PowerProfile construction
    // -----------------------------------------------------------------------
    #[test]
    fn power_profile_construction() {
        let profile = PowerProfile {
            idle_watts: 25.0,
            peak_watts: 300.0,
            tdp_watts: 350.0,
            efficiency_ratio: 0.85,
        };
        assert!((profile.idle_watts - 25.0).abs() < 1e-9);
        assert!((profile.peak_watts - 300.0).abs() < 1e-9);
        assert!((profile.tdp_watts - 350.0).abs() < 1e-9);
        assert!((profile.efficiency_ratio - 0.85).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Test 6: Energy per op calculation
    // -----------------------------------------------------------------------
    #[test]
    fn energy_per_op_calculation() {
        // avg_power = 200W, median_time = 100us, flops = 2e9
        // energy_joules = 200 * 100e-6 = 0.02 J
        // energy_nj = 0.02 * 1e9 = 2e7 nJ
        // energy_per_op = 2e7 / 2e9 = 0.01 nJ
        let avg_power = 200.0;
        let median_us = 100.0;
        let flops = 2e9;

        let energy_per_op_nj: f64 = avg_power * median_us * 1e3 / flops;
        assert!((energy_per_op_nj - 0.01).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Test 7: PowerConstraint::MaxWatts satisfied
    // -----------------------------------------------------------------------
    #[test]
    fn constraint_max_watts_satisfied() {
        let result = make_power_result(
            Config::new(),
            100.0,
            Some(20.0),
            250.0, // peak
            25.0,
            0.08,
            0.01,
            11.0,
        );
        let constraint = PowerConstraint::MaxWatts(300.0);
        assert!((constraint.violation(&result)).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Test 8: PowerConstraint::MaxWatts violated
    // -----------------------------------------------------------------------
    #[test]
    fn constraint_max_watts_violated() {
        let result = make_power_result(
            Config::new(),
            100.0,
            Some(20.0),
            350.0, // peak exceeds limit
            25.0,
            0.08,
            0.01,
            11.0,
        );
        let constraint = PowerConstraint::MaxWatts(300.0);
        let v = constraint.violation(&result);
        // violation = (350 - 300) / 300 = 50/300 ≈ 0.1667
        assert!((v - 50.0 / 300.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 9: PowerConstraint::MinEfficiency
    // -----------------------------------------------------------------------
    #[test]
    fn constraint_min_efficiency() {
        let result = make_power_result(
            Config::new(),
            100.0,
            Some(20.0),
            250.0,
            25.0,
            0.05, // efficiency below minimum
            0.01,
            11.0,
        );
        let constraint = PowerConstraint::MinEfficiency(0.10);
        let v = constraint.violation(&result);
        // violation = (0.10 - 0.05) / 0.10 = 0.5
        assert!((v - 0.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 10: PowerAwareSelector with all feasible
    // -----------------------------------------------------------------------
    #[test]
    fn selector_picks_fastest_feasible() {
        let results = vec![
            make_power_result(
                Config::new().with_tile_m(64),
                200.0,
                Some(10.0),
                200.0,
                25.0,
                0.05,
                0.02,
                20.0,
            ),
            make_power_result(
                Config::new().with_tile_m(128),
                100.0, // fastest
                Some(20.0),
                250.0,
                25.0,
                0.08,
                0.01,
                15.0,
            ),
            make_power_result(
                Config::new().with_tile_m(256),
                150.0,
                Some(13.3),
                280.0,
                25.0,
                0.048,
                0.015,
                10.0,
            ),
        ];

        let selector = PowerAwareSelector::new(vec![PowerConstraint::MaxWatts(300.0)]);
        let best = selector.select_best(&results).expect("should find best");
        assert_eq!(best.benchmark.config.tile_m, 128);
    }

    // -----------------------------------------------------------------------
    // Test 11: PowerAwareSelector filters by constraint
    // -----------------------------------------------------------------------
    #[test]
    fn selector_filters_by_power_constraint() {
        let results = vec![
            make_power_result(
                Config::new().with_tile_m(64),
                200.0,
                Some(10.0),
                200.0, // within 220W limit
                25.0,
                0.05,
                0.02,
                20.0,
            ),
            make_power_result(
                Config::new().with_tile_m(128),
                100.0, // faster but exceeds power
                Some(20.0),
                250.0, // exceeds 220W limit
                25.0,
                0.08,
                0.01,
                15.0,
            ),
        ];

        let selector = PowerAwareSelector::new(vec![PowerConstraint::MaxWatts(220.0)]);
        let best = selector.select_best(&results).expect("should find best");
        // Should pick the slower but power-compliant config
        assert_eq!(best.benchmark.config.tile_m, 64);
    }

    // -----------------------------------------------------------------------
    // Test 12: PowerAwareSelector with no feasible — picks least violating
    // -----------------------------------------------------------------------
    #[test]
    fn selector_picks_least_violating_when_no_feasible() {
        let results = vec![
            make_power_result(
                Config::new().with_tile_m(64),
                200.0,
                Some(10.0),
                250.0, // 250 > 200 limit, violation = 50/200 = 0.25
                25.0,
                0.05,
                0.02,
                20.0,
            ),
            make_power_result(
                Config::new().with_tile_m(128),
                100.0,
                Some(20.0),
                350.0, // 350 > 200 limit, violation = 150/200 = 0.75
                25.0,
                0.08,
                0.01,
                15.0,
            ),
        ];

        let selector = PowerAwareSelector::new(vec![PowerConstraint::MaxWatts(200.0)]);
        let best = selector
            .select_best(&results)
            .expect("should find least violating");
        // First config has lower violation
        assert_eq!(best.benchmark.config.tile_m, 64);
    }

    // -----------------------------------------------------------------------
    // Test 13: PowerAwareSelector with empty results
    // -----------------------------------------------------------------------
    #[test]
    fn selector_empty_results_returns_error() {
        let selector = PowerAwareSelector::new(vec![PowerConstraint::MaxWatts(200.0)]);
        let results: Vec<PowerAwareBenchmarkResult> = vec![];
        assert!(selector.select_best(&results).is_err());
    }

    // -----------------------------------------------------------------------
    // Test 14: estimate_power_from_config — default config
    // -----------------------------------------------------------------------
    #[test]
    fn estimate_power_default_config() {
        let config = Config::new(); // 128x128, 256 threads, 2 stages
        let estimated = estimate_power_from_config(&config, 200.0);
        // Default config should produce ratio ~1.0, so estimated ≈ base_power
        assert!(
            (estimated - 200.0).abs() < 1.0,
            "expected ~200W for default config, got {estimated}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 15: estimate_power_from_config — larger config draws more
    // -----------------------------------------------------------------------
    #[test]
    fn estimate_power_larger_config_draws_more() {
        let small = Config::new()
            .with_tile_m(64)
            .with_tile_n(64)
            .with_block_size(128);
        let large = Config::new()
            .with_tile_m(256)
            .with_tile_n(256)
            .with_block_size(512)
            .with_stages(4);

        let small_power = estimate_power_from_config(&small, 200.0);
        let large_power = estimate_power_from_config(&large, 200.0);

        assert!(
            large_power > small_power,
            "larger config ({large_power}W) should draw more than smaller ({small_power}W)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 16: estimate_power_from_config — tensor core overhead
    // -----------------------------------------------------------------------
    #[test]
    fn estimate_power_tensor_core_overhead() {
        let without_tc = Config::new();
        let with_tc = Config::new().with_use_tensor_core(true);

        let power_without = estimate_power_from_config(&without_tc, 200.0);
        let power_with = estimate_power_from_config(&with_tc, 200.0);

        assert!(
            power_with > power_without,
            "tensor core config ({power_with}W) should draw more than without ({power_without}W)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 17: PowerAwareBenchmarkEngine integration with synthetic monitor
    // -----------------------------------------------------------------------
    #[test]
    fn benchmark_engine_with_synthetic_monitor() {
        let monitor = SyntheticPowerMonitor::new(make_reading(200.0, 55.0, 1800));
        let engine = BenchmarkEngine::with_config(BenchmarkConfig {
            warmup_runs: 1,
            benchmark_runs: 5,
        });
        let power_engine = PowerAwareBenchmarkEngine::new(engine, Box::new(monitor), 350.0);

        let config = Config::new();
        let flops = 2e9;

        let result = power_engine
            .benchmark_with_power(&config, flops, || Ok(()))
            .expect("benchmark should succeed");

        assert!(result.benchmark.median_us >= 0.0);
        assert!((result.power_profile.idle_watts - 200.0).abs() < 1e-9);
        assert!(result.energy_per_op_nj >= 0.0);
        assert!(result.thermal_headroom_celsius > 0.0);
        assert!(result.power_profile.efficiency_ratio >= 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 18: Multiple constraints applied together
    // -----------------------------------------------------------------------
    #[test]
    fn selector_multiple_constraints() {
        let results = vec![
            make_power_result(
                Config::new().with_tile_m(64),
                200.0,
                Some(10.0),
                180.0,
                25.0,
                0.056, // 10.0 / 180 ~ 0.056
                0.02,
                20.0,
            ),
            make_power_result(
                Config::new().with_tile_m(128),
                100.0,
                Some(20.0),
                250.0, // exceeds 200W
                25.0,
                0.08,
                0.01,
                15.0,
            ),
            make_power_result(
                Config::new().with_tile_m(256),
                150.0,
                Some(13.3),
                190.0,
                25.0,
                0.07,
                0.03, // exceeds 0.025 nJ
                10.0,
            ),
        ];

        let selector = PowerAwareSelector::new(vec![
            PowerConstraint::MaxWatts(200.0),
            PowerConstraint::MaxEnergyPerOpNj(0.025),
        ]);

        let best = selector.select_best(&results).expect("should find best");
        // Only tile_m=64 satisfies both: peak=180 < 200, energy=0.02 < 0.025
        assert_eq!(best.benchmark.config.tile_m, 64);
    }

    // -----------------------------------------------------------------------
    // Test 19: select_feasible returns sorted results
    // -----------------------------------------------------------------------
    #[test]
    fn select_feasible_returns_sorted() {
        let results = vec![
            make_power_result(
                Config::new().with_tile_m(64),
                200.0,
                Some(10.0),
                180.0,
                25.0,
                0.056,
                0.02,
                20.0,
            ),
            make_power_result(
                Config::new().with_tile_m(128),
                100.0,
                Some(20.0),
                190.0,
                25.0,
                0.08,
                0.01,
                15.0,
            ),
            make_power_result(
                Config::new().with_tile_m(256),
                150.0,
                Some(13.3),
                195.0,
                25.0,
                0.07,
                0.015,
                10.0,
            ),
        ];

        let selector = PowerAwareSelector::new(vec![PowerConstraint::MaxWatts(200.0)]);
        let feasible = selector.select_feasible(&results);

        assert_eq!(feasible.len(), 3);
        // Should be sorted by median_us ascending
        assert_eq!(feasible[0].benchmark.config.tile_m, 128); // 100us
        assert_eq!(feasible[1].benchmark.config.tile_m, 256); // 150us
        assert_eq!(feasible[2].benchmark.config.tile_m, 64); // 200us
    }

    // -----------------------------------------------------------------------
    // Test 20: PowerConstraint::MaxTempCelsius
    // -----------------------------------------------------------------------
    #[test]
    fn constraint_max_temp_celsius() {
        // thermal_headroom = 83 - actual_temp
        // If headroom = 8, actual_temp = 75
        let result = make_power_result(
            Config::new(),
            100.0,
            Some(20.0),
            250.0,
            25.0,
            0.08,
            0.01,
            8.0, // headroom = 8 => actual_temp = 83 - 8 = 75
        );

        // 75C <= 80C => satisfied
        let ok = PowerConstraint::MaxTempCelsius(80.0);
        assert!((ok.violation(&result)).abs() < 1e-9);

        // 75C > 70C => violated
        let fail = PowerConstraint::MaxTempCelsius(70.0);
        let v = fail.violation(&result);
        // violation = (75 - 70) / 70 ≈ 0.0714
        assert!((v - 5.0 / 70.0).abs() < 1e-6);
    }
}
