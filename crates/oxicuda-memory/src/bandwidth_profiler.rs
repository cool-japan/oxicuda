//! Memory bandwidth profiling and throughput measurement.
//!
//! This module provides tools for measuring and analyzing transfer bandwidth
//! between host and device memory. It supports profiling of host-to-device,
//! device-to-host, device-to-device, and host-to-host transfers.
//!
//! # Overview
//!
//! The profiling workflow consists of:
//!
//! 1. **Recording** individual transfer measurements via [`BandwidthProfiler::record`].
//! 2. **Summarizing** collected data with [`BandwidthProfiler::summary`] or
//!    [`BandwidthProfiler::summary_by_direction`].
//! 3. **Estimating** transfer times and utilization with the standalone functions
//!    [`estimate_transfer_time`], [`theoretical_peak_bandwidth`], and
//!    [`bandwidth_utilization`].
//!
//! # Example
//!
//! ```rust
//! use oxicuda_memory::bandwidth_profiler::*;
//!
//! let mut profiler = BandwidthProfiler::new();
//!
//! // Record some measurements
//! let m = BandwidthMeasurement::new(
//!     TransferDirection::HostToDevice,
//!     1_048_576, // 1 MB
//!     0.5,       // 0.5 ms
//! );
//! profiler.record(m);
//!
//! let summary = profiler.summary();
//! println!("{summary}");
//! ```

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// TransferDirection
// ---------------------------------------------------------------------------

/// Direction of a memory transfer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    /// Host (CPU) memory to device (GPU) memory.
    HostToDevice,
    /// Device (GPU) memory to host (CPU) memory.
    DeviceToHost,
    /// Device (GPU) memory to device (GPU) memory.
    DeviceToDevice,
    /// Host (CPU) memory to host (CPU) memory.
    HostToHost,
}

impl fmt::Display for TransferDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "Host -> Device"),
            Self::DeviceToHost => write!(f, "Device -> Host"),
            Self::DeviceToDevice => write!(f, "Device -> Device"),
            Self::HostToHost => write!(f, "Host -> Host"),
        }
    }
}

// ---------------------------------------------------------------------------
// BandwidthMeasurement
// ---------------------------------------------------------------------------

/// A single transfer bandwidth measurement.
///
/// Each measurement captures the direction, size, elapsed time, computed
/// throughput, and a wall-clock timestamp for when it was recorded.
#[derive(Debug, Clone)]
pub struct BandwidthMeasurement {
    /// Direction of the transfer.
    pub direction: TransferDirection,
    /// Number of bytes transferred.
    pub bytes: usize,
    /// Elapsed time for the transfer in milliseconds.
    pub elapsed_ms: f64,
    /// Computed bandwidth in GB/s (10^9 bytes/s).
    pub bandwidth_gbps: f64,
    /// Wall-clock timestamp when this measurement was created.
    pub timestamp: Instant,
}

impl BandwidthMeasurement {
    /// Creates a new measurement from raw transfer parameters.
    ///
    /// The `bandwidth_gbps` field is automatically computed from `bytes` and
    /// `elapsed_ms`. If `elapsed_ms` is zero or negative, bandwidth is set
    /// to zero to avoid division-by-zero or negative values.
    pub fn new(direction: TransferDirection, bytes: usize, elapsed_ms: f64) -> Self {
        let bandwidth_gbps = if elapsed_ms > 0.0 {
            // bytes / (ms * 1e-3) = bytes * 1000 / ms  => bytes/s
            // then divide by 1e9 to get GB/s
            (bytes as f64) / (elapsed_ms * 1e-3) / 1e9
        } else {
            0.0
        };

        Self {
            direction,
            bytes,
            elapsed_ms,
            bandwidth_gbps,
            timestamp: Instant::now(),
        }
    }

    /// Returns the bandwidth in MB/s (10^6 bytes/s).
    #[inline]
    pub fn bandwidth_mbps(&self) -> f64 {
        self.bandwidth_gbps * 1000.0
    }

    /// Returns the transfer latency in microseconds.
    #[inline]
    pub fn latency_us(&self) -> f64 {
        self.elapsed_ms * 1000.0
    }
}

impl fmt::Display for BandwidthMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} bytes in {:.3} ms ({:.2} GB/s)",
            self.direction, self.bytes, self.elapsed_ms, self.bandwidth_gbps
        )
    }
}

// ---------------------------------------------------------------------------
// DirectionSummary
// ---------------------------------------------------------------------------

/// Aggregated statistics for transfers in a single direction.
#[derive(Debug, Clone)]
pub struct DirectionSummary {
    /// The transfer direction these statistics cover.
    pub direction: TransferDirection,
    /// Number of transfers recorded for this direction.
    pub transfer_count: usize,
    /// Total bytes transferred across all measurements.
    pub total_bytes: usize,
    /// Average bandwidth in GB/s across all measurements.
    pub avg_bandwidth_gbps: f64,
    /// Minimum observed bandwidth in GB/s.
    pub min_bandwidth_gbps: f64,
    /// Maximum observed bandwidth in GB/s.
    pub max_bandwidth_gbps: f64,
    /// Estimated fixed latency overhead in microseconds.
    ///
    /// Derived from the smallest transfer: this approximates the per-transfer
    /// setup cost independent of data size.
    pub latency_overhead_us: f64,
}

impl fmt::Display for DirectionSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} transfers, {} bytes total, avg {:.2} GB/s (min {:.2}, max {:.2}), \
             overhead ~{:.1} us",
            self.direction,
            self.transfer_count,
            self.total_bytes,
            self.avg_bandwidth_gbps,
            self.min_bandwidth_gbps,
            self.max_bandwidth_gbps,
            self.latency_overhead_us
        )
    }
}

// ---------------------------------------------------------------------------
// BandwidthSummary
// ---------------------------------------------------------------------------

/// Aggregated bandwidth statistics across all recorded measurements.
#[derive(Debug, Clone)]
pub struct BandwidthSummary {
    /// Total number of transfers recorded.
    pub total_transfers: usize,
    /// Total bytes transferred across all measurements.
    pub total_bytes: usize,
    /// Total wall-clock time of all transfers in milliseconds.
    pub total_time_ms: f64,
    /// Average bandwidth in GB/s across all measurements.
    pub avg_bandwidth_gbps: f64,
    /// Peak (maximum) bandwidth observed in any single measurement.
    pub peak_bandwidth_gbps: f64,
    /// Per-direction breakdown of statistics.
    pub per_direction: Vec<DirectionSummary>,
}

impl fmt::Display for BandwidthSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Bandwidth Summary ===")?;
        writeln!(
            f,
            "Total: {} transfers, {} bytes, {:.3} ms",
            self.total_transfers, self.total_bytes, self.total_time_ms
        )?;
        writeln!(
            f,
            "Avg: {:.2} GB/s, Peak: {:.2} GB/s",
            self.avg_bandwidth_gbps, self.peak_bandwidth_gbps
        )?;
        for ds in &self.per_direction {
            writeln!(f, "  {ds}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BandwidthProfiler
// ---------------------------------------------------------------------------

/// Accumulates bandwidth measurements and produces summary statistics.
///
/// The profiler stores individual [`BandwidthMeasurement`] records and can
/// compute aggregate statistics across all measurements or filtered by
/// transfer direction.
#[derive(Debug, Clone)]
pub struct BandwidthProfiler {
    /// Collected measurements.
    measurements: Vec<BandwidthMeasurement>,
    /// Number of warmup iterations (hint for benchmark drivers).
    pub warmup_iterations: u32,
    /// Number of benchmark iterations (hint for benchmark drivers).
    pub benchmark_iterations: u32,
}

impl Default for BandwidthProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl BandwidthProfiler {
    /// Creates a new profiler with default iteration counts
    /// (3 warmup, 10 benchmark).
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            warmup_iterations: 3,
            benchmark_iterations: 10,
        }
    }

    /// Creates a new profiler with custom iteration counts.
    pub fn with_iterations(warmup: u32, benchmark: u32) -> Self {
        Self {
            measurements: Vec::new(),
            warmup_iterations: warmup,
            benchmark_iterations: benchmark,
        }
    }

    /// Records a single bandwidth measurement.
    pub fn record(&mut self, measurement: BandwidthMeasurement) {
        self.measurements.push(measurement);
    }

    /// Returns the number of recorded measurements.
    #[inline]
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    /// Returns a reference to all recorded measurements.
    #[inline]
    pub fn measurements(&self) -> &[BandwidthMeasurement] {
        &self.measurements
    }

    /// Clears all recorded measurements.
    pub fn clear(&mut self) {
        self.measurements.clear();
    }

    /// Computes a summary of all recorded measurements.
    ///
    /// If no measurements have been recorded, all summary fields will be zero
    /// and `per_direction` will be empty.
    pub fn summary(&self) -> BandwidthSummary {
        if self.measurements.is_empty() {
            return BandwidthSummary {
                total_transfers: 0,
                total_bytes: 0,
                total_time_ms: 0.0,
                avg_bandwidth_gbps: 0.0,
                peak_bandwidth_gbps: 0.0,
                per_direction: Vec::new(),
            };
        }

        let total_transfers = self.measurements.len();
        let total_bytes: usize = self.measurements.iter().map(|m| m.bytes).sum();
        let total_time_ms: f64 = self.measurements.iter().map(|m| m.elapsed_ms).sum();

        let bw_sum: f64 = self.measurements.iter().map(|m| m.bandwidth_gbps).sum();
        let avg_bandwidth_gbps = bw_sum / total_transfers as f64;

        let peak_bandwidth_gbps = self
            .measurements
            .iter()
            .map(|m| m.bandwidth_gbps)
            .fold(0.0_f64, f64::max);

        // Build per-direction summaries for each direction that has data.
        let directions = [
            TransferDirection::HostToDevice,
            TransferDirection::DeviceToHost,
            TransferDirection::DeviceToDevice,
            TransferDirection::HostToHost,
        ];

        let per_direction: Vec<DirectionSummary> = directions
            .iter()
            .filter_map(|&dir| self.compute_direction_summary(dir))
            .collect();

        BandwidthSummary {
            total_transfers,
            total_bytes,
            total_time_ms,
            avg_bandwidth_gbps,
            peak_bandwidth_gbps,
            per_direction,
        }
    }

    /// Computes a summary for a single transfer direction.
    ///
    /// Returns `None` if no measurements exist for the given direction.
    pub fn summary_by_direction(&self, dir: TransferDirection) -> Option<DirectionSummary> {
        self.compute_direction_summary(dir)
    }

    /// Internal helper to compute a [`DirectionSummary`] for one direction.
    fn compute_direction_summary(&self, dir: TransferDirection) -> Option<DirectionSummary> {
        let filtered: Vec<&BandwidthMeasurement> = self
            .measurements
            .iter()
            .filter(|m| m.direction == dir)
            .collect();

        if filtered.is_empty() {
            return None;
        }

        let transfer_count = filtered.len();
        let total_bytes: usize = filtered.iter().map(|m| m.bytes).sum();

        let bw_sum: f64 = filtered.iter().map(|m| m.bandwidth_gbps).sum();
        let avg_bandwidth_gbps = bw_sum / transfer_count as f64;

        let min_bandwidth_gbps = filtered
            .iter()
            .map(|m| m.bandwidth_gbps)
            .fold(f64::INFINITY, f64::min);

        let max_bandwidth_gbps = filtered
            .iter()
            .map(|m| m.bandwidth_gbps)
            .fold(0.0_f64, f64::max);

        // Estimate latency overhead from the smallest transfer.
        // The smallest transfer is most dominated by fixed overhead, so its
        // latency serves as a reasonable approximation.
        let latency_overhead_us = filtered
            .iter()
            .min_by_key(|m| m.bytes)
            .map(|m| m.latency_us())
            .unwrap_or(0.0);

        Some(DirectionSummary {
            direction: dir,
            transfer_count,
            total_bytes,
            avg_bandwidth_gbps,
            min_bandwidth_gbps,
            max_bandwidth_gbps,
            latency_overhead_us,
        })
    }
}

// ---------------------------------------------------------------------------
// BandwidthBenchmarkConfig
// ---------------------------------------------------------------------------

/// Configuration for a bandwidth benchmark sweep.
///
/// Specifies which transfer sizes, directions, and iteration counts to use
/// when running a benchmark. The [`Default`] implementation provides a
/// standard set of sizes from 1 KB to 256 MB across all four directions.
#[derive(Debug, Clone)]
pub struct BandwidthBenchmarkConfig {
    /// Transfer sizes to benchmark (in bytes).
    pub sizes: Vec<usize>,
    /// Transfer directions to benchmark.
    pub directions: Vec<TransferDirection>,
    /// Number of warmup iterations before timing begins.
    pub warmup_iterations: u32,
    /// Number of timed benchmark iterations per size/direction pair.
    pub benchmark_iterations: u32,
    /// Whether to use pinned (page-locked) host memory for transfers.
    pub use_pinned_memory: bool,
}

impl Default for BandwidthBenchmarkConfig {
    fn default() -> Self {
        Self {
            sizes: vec![
                1 << 10,   // 1 KB
                4 << 10,   // 4 KB
                16 << 10,  // 16 KB
                64 << 10,  // 64 KB
                256 << 10, // 256 KB
                1 << 20,   // 1 MB
                4 << 20,   // 4 MB
                16 << 20,  // 16 MB
                64 << 20,  // 64 MB
                256 << 20, // 256 MB
            ],
            directions: vec![
                TransferDirection::HostToDevice,
                TransferDirection::DeviceToHost,
                TransferDirection::DeviceToDevice,
                TransferDirection::HostToHost,
            ],
            warmup_iterations: 3,
            benchmark_iterations: 10,
            use_pinned_memory: true,
        }
    }
}

impl BandwidthBenchmarkConfig {
    /// Creates a new config with custom sizes and default settings.
    pub fn with_sizes(sizes: Vec<usize>) -> Self {
        Self {
            sizes,
            ..Self::default()
        }
    }

    /// Creates a new config for a single direction.
    pub fn for_direction(direction: TransferDirection) -> Self {
        Self {
            directions: vec![direction],
            ..Self::default()
        }
    }

    /// Sets the number of warmup and benchmark iterations.
    pub fn set_iterations(&mut self, warmup: u32, benchmark: u32) {
        self.warmup_iterations = warmup;
        self.benchmark_iterations = benchmark;
    }

    /// Total number of individual transfers this config would produce.
    ///
    /// Equal to `sizes.len() * directions.len() * benchmark_iterations`.
    pub fn total_transfers(&self) -> usize {
        self.sizes.len() * self.directions.len() * self.benchmark_iterations as usize
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Estimates the transfer time in milliseconds for a given data size.
///
/// Uses a simple linear model: `time = latency + bytes / bandwidth`.
///
/// # Parameters
///
/// * `bytes` — number of bytes to transfer.
/// * `bandwidth_gbps` — sustained bandwidth in GB/s.
/// * `latency_us` — fixed per-transfer overhead in microseconds.
///
/// # Returns
///
/// Estimated transfer time in milliseconds. Returns `f64::INFINITY` if
/// `bandwidth_gbps` is zero or negative.
pub fn estimate_transfer_time(bytes: usize, bandwidth_gbps: f64, latency_us: f64) -> f64 {
    if bandwidth_gbps <= 0.0 {
        return f64::INFINITY;
    }
    let latency_ms = latency_us / 1000.0;
    // bandwidth_gbps = GB/s = 1e9 bytes/s
    // time_for_data (s) = bytes / (bandwidth_gbps * 1e9)
    // time_for_data (ms) = bytes / (bandwidth_gbps * 1e9) * 1e3
    //                    = bytes / (bandwidth_gbps * 1e6)
    let data_time_ms = bytes as f64 / (bandwidth_gbps * 1e6);
    latency_ms + data_time_ms
}

/// Returns the theoretical peak unidirectional bandwidth for a PCIe
/// configuration in GB/s.
///
/// Accounts for the 128b/130b encoding overhead used by PCIe 3.0+ (yielding
/// ~98.46% efficiency) and the per-lane raw data rates:
///
/// | Generation | Per-lane rate (GT/s) |
/// |------------|----------------------|
/// | PCIe 1.0   | 2.5                  |
/// | PCIe 2.0   | 5.0                  |
/// | PCIe 3.0   | 8.0                  |
/// | PCIe 4.0   | 16.0                 |
/// | PCIe 5.0   | 32.0                 |
/// | PCIe 6.0   | 64.0                 |
///
/// # Parameters
///
/// * `pcie_gen` — PCIe generation (1–6).
/// * `lanes` — number of PCIe lanes (typically 1, 4, 8, or 16).
///
/// # Returns
///
/// Theoretical peak bandwidth in GB/s, or `0.0` if `pcie_gen` is out of
/// range or `lanes` is zero.
pub fn theoretical_peak_bandwidth(pcie_gen: u32, lanes: u32) -> f64 {
    if lanes == 0 {
        return 0.0;
    }

    // Per-lane data rate in GT/s (gigatransfers/second)
    let rate_gtps: f64 = match pcie_gen {
        1 => 2.5,
        2 => 5.0,
        3 => 8.0,
        4 => 16.0,
        5 => 32.0,
        6 => 64.0,
        _ => return 0.0,
    };

    // PCIe 1.0 and 2.0 use 8b/10b encoding (80% efficiency).
    // PCIe 3.0+ use 128b/130b encoding (~98.46% efficiency).
    let encoding_efficiency: f64 = if pcie_gen <= 2 { 0.8 } else { 128.0 / 130.0 };

    // Each transfer moves 1 bit, so GT/s = Gbit/s.
    // Convert to GB/s: Gbit/s / 8.
    rate_gtps * lanes as f64 * encoding_efficiency / 8.0
}

/// Returns the bandwidth utilization ratio (0.0–1.0).
///
/// # Parameters
///
/// * `measured_gbps` — measured bandwidth in GB/s.
/// * `peak_gbps` — theoretical peak bandwidth in GB/s.
///
/// # Returns
///
/// The ratio `measured / peak`, clamped to `[0.0, 1.0]`. Returns `0.0` if
/// `peak_gbps` is zero or negative.
pub fn bandwidth_utilization(measured_gbps: f64, peak_gbps: f64) -> f64 {
    if peak_gbps <= 0.0 {
        return 0.0;
    }
    (measured_gbps / peak_gbps).clamp(0.0, 1.0)
}

/// Formats a byte count into a human-readable string (e.g., "1.00 MB").
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1 << 10;
    const MB: usize = 1 << 20;
    const GB: usize = 1 << 30;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Returns a human-readable description of a bandwidth value.
///
/// Useful for logging and reporting.
pub fn describe_bandwidth(gbps: f64) -> String {
    if gbps >= 1.0 {
        format!("{gbps:.2} GB/s")
    } else {
        format!("{:.2} MB/s", gbps * 1000.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- BandwidthMeasurement ------------------------------------------------

    #[test]
    fn measurement_new_computes_bandwidth() {
        // 1 GB in 1000 ms = 1 GB/s
        let m = BandwidthMeasurement::new(TransferDirection::HostToDevice, 1_000_000_000, 1000.0);
        assert!((m.bandwidth_gbps - 1.0).abs() < 1e-6);
    }

    #[test]
    fn measurement_zero_elapsed_gives_zero_bandwidth() {
        let m = BandwidthMeasurement::new(TransferDirection::HostToDevice, 1024, 0.0);
        assert!((m.bandwidth_gbps - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn measurement_negative_elapsed_gives_zero_bandwidth() {
        let m = BandwidthMeasurement::new(TransferDirection::DeviceToHost, 1024, -1.0);
        assert!((m.bandwidth_gbps - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn measurement_bandwidth_mbps() {
        let m = BandwidthMeasurement::new(TransferDirection::DeviceToDevice, 1_000_000_000, 1000.0);
        assert!((m.bandwidth_mbps() - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn measurement_latency_us() {
        let m = BandwidthMeasurement::new(TransferDirection::HostToHost, 1024, 2.5);
        assert!((m.latency_us() - 2500.0).abs() < 1e-6);
    }

    #[test]
    fn measurement_display() {
        let m = BandwidthMeasurement::new(TransferDirection::HostToDevice, 1048576, 0.5);
        let s = format!("{m}");
        assert!(s.contains("Host -> Device"));
        assert!(s.contains("1048576"));
        assert!(s.contains("0.500 ms"));
        assert!(s.contains("GB/s"));
    }

    // -- BandwidthProfiler ---------------------------------------------------

    #[test]
    fn profiler_empty_summary() {
        let profiler = BandwidthProfiler::new();
        let s = profiler.summary();
        assert_eq!(s.total_transfers, 0);
        assert_eq!(s.total_bytes, 0);
        assert!((s.avg_bandwidth_gbps - 0.0).abs() < f64::EPSILON);
        assert!(s.per_direction.is_empty());
    }

    #[test]
    fn profiler_record_and_summary() {
        let mut profiler = BandwidthProfiler::new();

        // 1 MB in 0.5 ms and 2 MB in 1.0 ms (both HtoD)
        profiler.record(BandwidthMeasurement::new(
            TransferDirection::HostToDevice,
            1 << 20,
            0.5,
        ));
        profiler.record(BandwidthMeasurement::new(
            TransferDirection::HostToDevice,
            2 << 20,
            1.0,
        ));
        // 512 KB in 0.25 ms (DtoH)
        profiler.record(BandwidthMeasurement::new(
            TransferDirection::DeviceToHost,
            512 << 10,
            0.25,
        ));

        let s = profiler.summary();
        assert_eq!(s.total_transfers, 3);
        assert_eq!(s.total_bytes, (1 << 20) + (2 << 20) + (512 << 10));
        assert!((s.total_time_ms - 1.75).abs() < 1e-9);
        assert!(s.peak_bandwidth_gbps > 0.0);
        assert_eq!(s.per_direction.len(), 2); // HtoD and DtoH
    }

    #[test]
    fn profiler_summary_by_direction() {
        let mut profiler = BandwidthProfiler::new();

        profiler.record(BandwidthMeasurement::new(
            TransferDirection::HostToDevice,
            1 << 20,
            0.5,
        ));
        profiler.record(BandwidthMeasurement::new(
            TransferDirection::DeviceToHost,
            1 << 20,
            0.6,
        ));

        assert!(
            profiler
                .summary_by_direction(TransferDirection::HostToDevice)
                .is_some()
        );
        assert!(
            profiler
                .summary_by_direction(TransferDirection::DeviceToHost)
                .is_some()
        );
        assert!(
            profiler
                .summary_by_direction(TransferDirection::DeviceToDevice)
                .is_none()
        );
    }

    #[test]
    fn profiler_direction_summary_stats() {
        let mut profiler = BandwidthProfiler::new();

        // Two HtoD transfers with different bandwidths
        let m1 = BandwidthMeasurement::new(TransferDirection::HostToDevice, 1_000_000, 1.0);
        let m2 = BandwidthMeasurement::new(TransferDirection::HostToDevice, 2_000_000, 1.0);
        let bw1 = m1.bandwidth_gbps;
        let bw2 = m2.bandwidth_gbps;
        profiler.record(m1);
        profiler.record(m2);

        let ds = profiler
            .summary_by_direction(TransferDirection::HostToDevice)
            .expect("should have HtoD summary");

        assert_eq!(ds.transfer_count, 2);
        assert_eq!(ds.total_bytes, 3_000_000);
        assert!((ds.avg_bandwidth_gbps - (bw1 + bw2) / 2.0).abs() < 1e-9);
        assert!((ds.min_bandwidth_gbps - bw1).abs() < 1e-9);
        assert!((ds.max_bandwidth_gbps - bw2).abs() < 1e-9);
    }

    #[test]
    fn profiler_with_iterations() {
        let p = BandwidthProfiler::with_iterations(5, 20);
        assert_eq!(p.warmup_iterations, 5);
        assert_eq!(p.benchmark_iterations, 20);
        assert_eq!(p.measurement_count(), 0);
    }

    #[test]
    fn profiler_clear() {
        let mut p = BandwidthProfiler::new();
        p.record(BandwidthMeasurement::new(
            TransferDirection::HostToDevice,
            1024,
            0.1,
        ));
        assert_eq!(p.measurement_count(), 1);
        p.clear();
        assert_eq!(p.measurement_count(), 0);
    }

    // -- Standalone functions ------------------------------------------------

    #[test]
    fn estimate_transfer_time_basic() {
        // 1 GB at 10 GB/s with 5 us latency
        let t = estimate_transfer_time(1_000_000_000, 10.0, 5.0);
        // data_time = 1e9 / (10 * 1e6) = 100 ms
        // latency = 5 / 1000 = 0.005 ms
        // total = 100.005 ms
        assert!((t - 100.005).abs() < 1e-6);
    }

    #[test]
    fn estimate_transfer_time_zero_bandwidth() {
        let t = estimate_transfer_time(1024, 0.0, 5.0);
        assert!(t.is_infinite());
    }

    #[test]
    fn theoretical_peak_bandwidth_pcie3_x16() {
        let bw = theoretical_peak_bandwidth(3, 16);
        // PCIe 3.0 x16: 8 GT/s * 16 * (128/130) / 8 ≈ 15.754 GB/s
        assert!((bw - 15.754).abs() < 0.01);
    }

    #[test]
    fn theoretical_peak_bandwidth_pcie4_x16() {
        let bw = theoretical_peak_bandwidth(4, 16);
        // PCIe 4.0 x16: 16 GT/s * 16 * (128/130) / 8 ≈ 31.508 GB/s
        assert!((bw - 31.508).abs() < 0.01);
    }

    #[test]
    fn theoretical_peak_bandwidth_pcie5_x16() {
        let bw = theoretical_peak_bandwidth(5, 16);
        // PCIe 5.0 x16: 32 GT/s * 16 * (128/130) / 8 ≈ 63.015 GB/s
        assert!((bw - 63.015).abs() < 0.02);
    }

    #[test]
    fn theoretical_peak_bandwidth_invalid_gen() {
        assert!((theoretical_peak_bandwidth(0, 16) - 0.0).abs() < f64::EPSILON);
        assert!((theoretical_peak_bandwidth(7, 16) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn theoretical_peak_bandwidth_zero_lanes() {
        assert!((theoretical_peak_bandwidth(3, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bandwidth_utilization_basic() {
        let u = bandwidth_utilization(12.0, 16.0);
        assert!((u - 0.75).abs() < 1e-9);
    }

    #[test]
    fn bandwidth_utilization_clamps_above_one() {
        let u = bandwidth_utilization(20.0, 16.0);
        assert!((u - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bandwidth_utilization_zero_peak() {
        let u = bandwidth_utilization(10.0, 0.0);
        assert!((u - 0.0).abs() < f64::EPSILON);
    }

    // -- BandwidthBenchmarkConfig --------------------------------------------

    #[test]
    fn benchmark_config_default_sizes() {
        let cfg = BandwidthBenchmarkConfig::default();
        assert_eq!(cfg.sizes.len(), 10);
        assert_eq!(cfg.sizes[0], 1 << 10); // 1 KB
        assert_eq!(cfg.sizes[9], 256 << 20); // 256 MB
        assert_eq!(cfg.directions.len(), 4);
        assert_eq!(cfg.warmup_iterations, 3);
        assert_eq!(cfg.benchmark_iterations, 10);
        assert!(cfg.use_pinned_memory);
    }

    #[test]
    fn benchmark_config_total_transfers() {
        let cfg = BandwidthBenchmarkConfig::default();
        // 10 sizes * 4 directions * 10 iterations = 400
        assert_eq!(cfg.total_transfers(), 400);
    }

    #[test]
    fn benchmark_config_with_sizes() {
        let cfg = BandwidthBenchmarkConfig::with_sizes(vec![1024, 2048]);
        assert_eq!(cfg.sizes.len(), 2);
        assert_eq!(cfg.directions.len(), 4); // inherits default
    }

    #[test]
    fn benchmark_config_for_direction() {
        let cfg = BandwidthBenchmarkConfig::for_direction(TransferDirection::DeviceToDevice);
        assert_eq!(cfg.directions.len(), 1);
        assert_eq!(cfg.directions[0], TransferDirection::DeviceToDevice);
    }

    // -- Display / formatting ------------------------------------------------

    #[test]
    fn summary_display_format() {
        let mut profiler = BandwidthProfiler::new();
        profiler.record(BandwidthMeasurement::new(
            TransferDirection::HostToDevice,
            1 << 20,
            0.5,
        ));
        let summary = profiler.summary();
        let display = format!("{summary}");
        assert!(display.contains("Bandwidth Summary"));
        assert!(display.contains("GB/s"));
    }

    #[test]
    fn direction_display() {
        assert_eq!(
            format!("{}", TransferDirection::HostToDevice),
            "Host -> Device"
        );
        assert_eq!(
            format!("{}", TransferDirection::DeviceToHost),
            "Device -> Host"
        );
        assert_eq!(
            format!("{}", TransferDirection::DeviceToDevice),
            "Device -> Device"
        );
        assert_eq!(format!("{}", TransferDirection::HostToHost), "Host -> Host");
    }

    #[test]
    fn format_bytes_ranges() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1 << 20), "1.00 MB");
        assert_eq!(format_bytes(1 << 30), "1.00 GB");
    }

    #[test]
    fn describe_bandwidth_formatting() {
        assert_eq!(describe_bandwidth(2.5), "2.50 GB/s");
        assert_eq!(describe_bandwidth(0.5), "500.00 MB/s");
    }

    // -- PCIe gen 1/2 encoding -----------------------------------------------

    #[test]
    fn theoretical_peak_bandwidth_pcie1_x16() {
        let bw = theoretical_peak_bandwidth(1, 16);
        // PCIe 1.0 x16: 2.5 GT/s * 16 * 0.8 / 8 = 4.0 GB/s
        assert!((bw - 4.0).abs() < 1e-6);
    }

    #[test]
    fn theoretical_peak_bandwidth_pcie2_x16() {
        let bw = theoretical_peak_bandwidth(2, 16);
        // PCIe 2.0 x16: 5.0 GT/s * 16 * 0.8 / 8 = 8.0 GB/s
        assert!((bw - 8.0).abs() < 1e-6);
    }
}
