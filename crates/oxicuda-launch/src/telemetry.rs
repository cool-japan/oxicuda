//! Launch telemetry: timing, occupancy, and register usage reporting.
//!
//! This module provides post-launch diagnostics for GPU kernel execution.
//! After a kernel launch, [`LaunchTelemetry`] captures grid/block dimensions,
//! GPU-side timing, achieved occupancy, and register usage. A
//! [`TelemetryCollector`] accumulates entries and produces a
//! [`TelemetrySummary`] with per-kernel aggregation.
//!
//! Telemetry can be exported to JSON, CSV, or Chrome trace format via
//! [`TelemetryExporter`].
//!
//! # Example
//!
//! ```
//! use oxicuda_launch::telemetry::{LaunchTelemetry, TelemetryCollector};
//!
//! let mut collector = TelemetryCollector::new(1000);
//! let entry = LaunchTelemetry::new("vector_add", (4, 1, 1), (256, 1, 1))
//!     .with_elapsed_ms(0.5)
//!     .with_achieved_occupancy(0.85);
//! collector.record(entry);
//! let summary = collector.summary();
//! assert_eq!(summary.total_launches, 1);
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// SmVersion (local, avoids oxicuda-ptx dependency)
// ---------------------------------------------------------------------------

/// GPU architecture version for occupancy estimation.
///
/// This is a local copy that avoids a dependency on `oxicuda-ptx`.
/// Each variant encodes the SM architecture parameters needed for
/// occupancy calculations (max warps per SM, register file size, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SmVersion {
    /// Turing (compute capability 7.5).
    Sm75,
    /// Ampere (compute capability 8.0).
    Sm80,
    /// Ampere GA10x (compute capability 8.6).
    Sm86,
    /// Ada Lovelace (compute capability 8.9).
    Sm89,
    /// Hopper (compute capability 9.0).
    Sm90,
    /// Blackwell (compute capability 10.0).
    Sm100,
    /// Blackwell B200 (compute capability 12.0).
    Sm120,
}

impl SmVersion {
    /// Maximum number of warps that can reside on a single SM.
    #[must_use]
    pub const fn max_warps_per_sm(self) -> u32 {
        match self {
            Self::Sm75 => 32,
            Self::Sm89 => 48,
            Self::Sm80 | Self::Sm86 | Self::Sm90 | Self::Sm100 | Self::Sm120 => 64,
        }
    }

    /// Maximum number of thread blocks that can reside on a single SM.
    #[must_use]
    pub const fn max_blocks_per_sm(self) -> u32 {
        match self {
            Self::Sm75 | Self::Sm80 | Self::Sm86 | Self::Sm89 => 16,
            Self::Sm90 | Self::Sm100 | Self::Sm120 => 32,
        }
    }

    /// Total number of 32-bit registers available per SM.
    #[must_use]
    pub const fn registers_per_sm(self) -> u32 {
        65536
    }

    /// Maximum number of registers a single thread can use.
    #[must_use]
    pub const fn max_registers_per_thread(self) -> u32 {
        255
    }

    /// Maximum shared memory per SM in bytes.
    #[must_use]
    pub const fn max_shared_mem_per_sm(self) -> u32 {
        match self {
            Self::Sm75 => 65_536,
            Self::Sm80 | Self::Sm86 => 163_840,
            Self::Sm89 => 101_376,
            Self::Sm90 | Self::Sm100 | Self::Sm120 => 232_448,
        }
    }

    /// Warp size (always 32 for NVIDIA GPUs).
    #[must_use]
    pub const fn warp_size(self) -> u32 {
        32
    }

    /// Register allocation granularity (in warps).
    ///
    /// Registers are allocated to warps in chunks of this many registers
    /// per thread, rounded up to the nearest multiple.
    #[must_use]
    pub const fn register_alloc_granularity(self) -> u32 {
        // All modern NVIDIA GPUs allocate registers in units of 256
        // (i.e., 8 registers per thread * 32 threads = 256 regs per warp).
        // The granularity for per-thread count rounding is 8.
        8
    }

    /// Shared memory allocation granularity in bytes.
    #[must_use]
    pub const fn shared_mem_alloc_granularity(self) -> u32 {
        match self {
            Self::Sm75 | Self::Sm80 | Self::Sm86 | Self::Sm89 => 256,
            Self::Sm90 | Self::Sm100 | Self::Sm120 => 128,
        }
    }
}

impl fmt::Display for SmVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Sm75 => "sm_75",
            Self::Sm80 => "sm_80",
            Self::Sm86 => "sm_86",
            Self::Sm89 => "sm_89",
            Self::Sm90 => "sm_90",
            Self::Sm100 => "sm_100",
            Self::Sm120 => "sm_120",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// LaunchTelemetry
// ---------------------------------------------------------------------------

/// Telemetry data collected after a single kernel launch.
///
/// Records dimensions, timing, occupancy, and register usage.
/// Use the builder methods (`with_*`) to set optional fields after
/// constructing with [`LaunchTelemetry::new`].
#[derive(Debug, Clone)]
pub struct LaunchTelemetry {
    /// Name of the launched kernel.
    pub kernel_name: String,
    /// Grid dimensions `(x, y, z)`.
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions `(x, y, z)`.
    pub block_dim: (u32, u32, u32),
    /// Dynamic shared memory allocated in bytes.
    pub shared_memory_bytes: u32,
    /// Number of registers used per thread, if known.
    pub register_count: Option<u32>,
    /// GPU-side elapsed time in milliseconds, if measured.
    pub elapsed_ms: Option<f64>,
    /// Achieved occupancy (0.0..=1.0), if measured.
    pub achieved_occupancy: Option<f64>,
    /// Theoretical occupancy (0.0..=1.0), if computed.
    pub theoretical_occupancy: Option<f64>,
    /// Wall-clock timestamp when the telemetry was recorded.
    pub timestamp: Instant,
}

impl LaunchTelemetry {
    /// Creates a new telemetry entry with the given kernel name and dimensions.
    ///
    /// Optional fields default to `None` / `0`. Use the `with_*` builder
    /// methods to set them.
    #[must_use]
    pub fn new(kernel_name: &str, grid_dim: (u32, u32, u32), block_dim: (u32, u32, u32)) -> Self {
        Self {
            kernel_name: kernel_name.to_owned(),
            grid_dim,
            block_dim,
            shared_memory_bytes: 0,
            register_count: None,
            elapsed_ms: None,
            achieved_occupancy: None,
            theoretical_occupancy: None,
            timestamp: Instant::now(),
        }
    }

    /// Sets the dynamic shared memory allocation.
    #[must_use]
    pub fn with_shared_memory(mut self, bytes: u32) -> Self {
        self.shared_memory_bytes = bytes;
        self
    }

    /// Sets the register count per thread.
    #[must_use]
    pub fn with_register_count(mut self, count: u32) -> Self {
        self.register_count = Some(count);
        self
    }

    /// Sets the GPU-side elapsed time in milliseconds.
    #[must_use]
    pub fn with_elapsed_ms(mut self, ms: f64) -> Self {
        self.elapsed_ms = Some(ms);
        self
    }

    /// Sets the achieved occupancy (0.0..=1.0).
    #[must_use]
    pub fn with_achieved_occupancy(mut self, occ: f64) -> Self {
        self.achieved_occupancy = Some(occ);
        self
    }

    /// Sets the theoretical occupancy (0.0..=1.0).
    #[must_use]
    pub fn with_theoretical_occupancy(mut self, occ: f64) -> Self {
        self.theoretical_occupancy = Some(occ);
        self
    }

    /// Total number of threads launched (grid_total * block_total).
    #[must_use]
    pub fn total_threads(&self) -> u64 {
        let grid_total = self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64;
        let block_total =
            self.block_dim.0 as u64 * self.block_dim.1 as u64 * self.block_dim.2 as u64;
        grid_total * block_total
    }
}

impl fmt::Display for LaunchTelemetry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Kernel '{}': grid=({},{},{}), block=({},{},{}), smem={}B",
            self.kernel_name,
            self.grid_dim.0,
            self.grid_dim.1,
            self.grid_dim.2,
            self.block_dim.0,
            self.block_dim.1,
            self.block_dim.2,
            self.shared_memory_bytes,
        )?;
        if let Some(regs) = self.register_count {
            write!(f, ", regs={regs}")?;
        }
        if let Some(ms) = self.elapsed_ms {
            write!(f, ", time={ms:.3}ms")?;
        }
        if let Some(occ) = self.achieved_occupancy {
            write!(f, ", occupancy={:.1}%", occ * 100.0)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// KernelStats
// ---------------------------------------------------------------------------

/// Aggregated statistics for a single kernel across multiple launches.
#[derive(Debug, Clone)]
pub struct KernelStats {
    /// Name of the kernel.
    pub kernel_name: String,
    /// Number of times this kernel was launched.
    pub launch_count: u32,
    /// Total GPU time across all launches in milliseconds.
    pub total_time_ms: f64,
    /// Average GPU time per launch in milliseconds.
    pub avg_time_ms: f64,
    /// Minimum GPU time observed in milliseconds.
    pub min_time_ms: f64,
    /// Maximum GPU time observed in milliseconds.
    pub max_time_ms: f64,
    /// Average achieved occupancy across launches.
    pub avg_occupancy: f64,
}

impl fmt::Display for KernelStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} launches, total={:.3}ms, avg={:.3}ms, min={:.3}ms, max={:.3}ms, occ={:.1}%",
            self.kernel_name,
            self.launch_count,
            self.total_time_ms,
            self.avg_time_ms,
            self.min_time_ms,
            self.max_time_ms,
            self.avg_occupancy * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// TelemetrySummary
// ---------------------------------------------------------------------------

/// Summary of all collected telemetry data.
///
/// Provides aggregate statistics across all recorded kernel launches
/// and per-kernel breakdowns via [`KernelStats`].
#[derive(Debug, Clone)]
pub struct TelemetrySummary {
    /// Total number of kernel launches recorded.
    pub total_launches: usize,
    /// Total GPU time across all launches in milliseconds.
    pub total_gpu_time_ms: f64,
    /// Average GPU time per launch in milliseconds.
    pub avg_gpu_time_ms: f64,
    /// Minimum GPU time observed across all launches.
    pub min_gpu_time_ms: f64,
    /// Maximum GPU time observed across all launches.
    pub max_gpu_time_ms: f64,
    /// Average achieved occupancy across all launches.
    pub avg_occupancy: f64,
    /// Kernel with the most cumulative GPU time.
    pub hottest_kernel: Option<String>,
    /// Per-kernel aggregated statistics.
    pub per_kernel_stats: Vec<KernelStats>,
}

impl fmt::Display for TelemetrySummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Telemetry Summary ===")?;
        writeln!(f, "Total launches: {}", self.total_launches)?;
        writeln!(f, "Total GPU time: {:.3} ms", self.total_gpu_time_ms)?;
        writeln!(f, "Avg GPU time:   {:.3} ms", self.avg_gpu_time_ms)?;
        writeln!(f, "Min GPU time:   {:.3} ms", self.min_gpu_time_ms)?;
        writeln!(f, "Max GPU time:   {:.3} ms", self.max_gpu_time_ms)?;
        writeln!(f, "Avg occupancy:  {:.1}%", self.avg_occupancy * 100.0)?;
        if let Some(ref hot) = self.hottest_kernel {
            writeln!(f, "Hottest kernel: {hot}")?;
        }
        if !self.per_kernel_stats.is_empty() {
            writeln!(f, "--- Per-kernel ---")?;
            for ks in &self.per_kernel_stats {
                writeln!(f, "  {ks}")?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TelemetryCollector
// ---------------------------------------------------------------------------

/// Accumulates [`LaunchTelemetry`] entries and produces summaries.
///
/// The collector can be enabled/disabled at runtime and caps the number
/// of stored entries to `max_entries` to bound memory usage.
#[derive(Debug)]
pub struct TelemetryCollector {
    entries: Vec<LaunchTelemetry>,
    enabled: bool,
    max_entries: usize,
}

impl TelemetryCollector {
    /// Creates a new collector that stores up to `max_entries` telemetry records.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            enabled: true,
            max_entries,
        }
    }

    /// Records a telemetry entry.
    ///
    /// If the collector is disabled or the entry count has reached
    /// `max_entries`, the entry is silently dropped.
    pub fn record(&mut self, telemetry: LaunchTelemetry) {
        if !self.enabled {
            return;
        }
        if self.entries.len() >= self.max_entries {
            return;
        }
        self.entries.push(telemetry);
    }

    /// Enables telemetry recording.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disables telemetry recording. Existing entries are preserved.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Returns whether the collector is currently enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Clears all recorded entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns a reference to all recorded entries.
    #[must_use]
    pub fn entries(&self) -> &[LaunchTelemetry] {
        &self.entries
    }

    /// Returns the number of recorded entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no entries have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Computes a summary of all recorded telemetry.
    ///
    /// If no entries have been recorded, returns a zeroed summary.
    #[must_use]
    pub fn summary(&self) -> TelemetrySummary {
        compute_summary(&self.entries)
    }
}

/// Computes a [`TelemetrySummary`] from a slice of telemetry entries.
fn compute_summary(entries: &[LaunchTelemetry]) -> TelemetrySummary {
    if entries.is_empty() {
        return TelemetrySummary {
            total_launches: 0,
            total_gpu_time_ms: 0.0,
            avg_gpu_time_ms: 0.0,
            min_gpu_time_ms: 0.0,
            max_gpu_time_ms: 0.0,
            avg_occupancy: 0.0,
            hottest_kernel: None,
            per_kernel_stats: Vec::new(),
        };
    }

    let mut total_time = 0.0_f64;
    let mut min_time = f64::MAX;
    let mut max_time = f64::MIN;
    let mut time_count = 0usize;
    let mut total_occ = 0.0_f64;
    let mut occ_count = 0usize;

    // Per-kernel accumulators
    struct KernelAccum {
        count: u32,
        total_time: f64,
        min_time: f64,
        max_time: f64,
        total_occ: f64,
        occ_count: u32,
    }

    let mut per_kernel: HashMap<String, KernelAccum> = HashMap::new();

    for entry in entries {
        if let Some(ms) = entry.elapsed_ms {
            total_time += ms;
            if ms < min_time {
                min_time = ms;
            }
            if ms > max_time {
                max_time = ms;
            }
            time_count += 1;
        }
        if let Some(occ) = entry.achieved_occupancy {
            total_occ += occ;
            occ_count += 1;
        }

        let acc = per_kernel
            .entry(entry.kernel_name.clone())
            .or_insert(KernelAccum {
                count: 0,
                total_time: 0.0,
                min_time: f64::MAX,
                max_time: f64::MIN,
                total_occ: 0.0,
                occ_count: 0,
            });
        acc.count += 1;
        if let Some(ms) = entry.elapsed_ms {
            acc.total_time += ms;
            if ms < acc.min_time {
                acc.min_time = ms;
            }
            if ms > acc.max_time {
                acc.max_time = ms;
            }
        }
        if let Some(occ) = entry.achieved_occupancy {
            acc.total_occ += occ;
            acc.occ_count += 1;
        }
    }

    // Fix sentinel values when no timing data was present
    if time_count == 0 {
        min_time = 0.0;
        max_time = 0.0;
    }

    // Build per-kernel stats
    let mut per_kernel_stats: Vec<KernelStats> = per_kernel
        .into_iter()
        .map(|(name, acc)| {
            let min_t = if acc.min_time == f64::MAX {
                0.0
            } else {
                acc.min_time
            };
            let max_t = if acc.max_time == f64::MIN {
                0.0
            } else {
                acc.max_time
            };
            let avg_t = if acc.count > 0 {
                acc.total_time / f64::from(acc.count)
            } else {
                0.0
            };
            let avg_o = if acc.occ_count > 0 {
                acc.total_occ / f64::from(acc.occ_count)
            } else {
                0.0
            };
            KernelStats {
                kernel_name: name,
                launch_count: acc.count,
                total_time_ms: acc.total_time,
                avg_time_ms: avg_t,
                min_time_ms: min_t,
                max_time_ms: max_t,
                avg_occupancy: avg_o,
            }
        })
        .collect();

    // Sort by total time descending for deterministic output
    per_kernel_stats.sort_by(|a, b| {
        b.total_time_ms
            .partial_cmp(&a.total_time_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let hottest_kernel = per_kernel_stats.first().map(|ks| ks.kernel_name.clone());

    let avg_gpu_time = if time_count > 0 {
        total_time / time_count as f64
    } else {
        0.0
    };
    let avg_occ = if occ_count > 0 {
        total_occ / occ_count as f64
    } else {
        0.0
    };

    TelemetrySummary {
        total_launches: entries.len(),
        total_gpu_time_ms: total_time,
        avg_gpu_time_ms: avg_gpu_time,
        min_gpu_time_ms: min_time,
        max_gpu_time_ms: max_time,
        avg_occupancy: avg_occ,
        hottest_kernel,
        per_kernel_stats,
    }
}

// ---------------------------------------------------------------------------
// TelemetryExporter
// ---------------------------------------------------------------------------

/// Export telemetry data in various formats.
///
/// All methods are stateless and operate on slices of [`LaunchTelemetry`].
pub struct TelemetryExporter;

impl TelemetryExporter {
    /// Exports telemetry entries as a JSON array.
    ///
    /// Each entry becomes a JSON object with all fields. `None` values
    /// are serialized as `null`.
    #[must_use]
    pub fn to_json(entries: &[LaunchTelemetry]) -> String {
        let mut out = String::from("[\n");
        for (i, e) in entries.iter().enumerate() {
            out.push_str("  {\n");
            json_field_str(&mut out, "kernel_name", &e.kernel_name);
            out.push_str(&format!(
                "    \"grid_dim\": [{}, {}, {}],\n",
                e.grid_dim.0, e.grid_dim.1, e.grid_dim.2
            ));
            out.push_str(&format!(
                "    \"block_dim\": [{}, {}, {}],\n",
                e.block_dim.0, e.block_dim.1, e.block_dim.2
            ));
            out.push_str(&format!(
                "    \"shared_memory_bytes\": {},\n",
                e.shared_memory_bytes
            ));
            json_field_opt_u32(&mut out, "register_count", e.register_count);
            json_field_opt_f64(&mut out, "elapsed_ms", e.elapsed_ms);
            json_field_opt_f64(&mut out, "achieved_occupancy", e.achieved_occupancy);
            json_field_opt_f64_last(&mut out, "theoretical_occupancy", e.theoretical_occupancy);
            out.push_str("  }");
            if i + 1 < entries.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push(']');
        out
    }

    /// Exports telemetry entries as CSV.
    ///
    /// The first line is a header row. Missing values are empty cells.
    #[must_use]
    pub fn to_csv(entries: &[LaunchTelemetry]) -> String {
        let mut out = String::from(
            "kernel_name,grid_x,grid_y,grid_z,block_x,block_y,block_z,\
             shared_memory_bytes,register_count,elapsed_ms,\
             achieved_occupancy,theoretical_occupancy\n",
        );
        for e in entries {
            out.push_str(&csv_escape(&e.kernel_name));
            out.push(',');
            out.push_str(&format!(
                "{},{},{},{},{},{},{},",
                e.grid_dim.0,
                e.grid_dim.1,
                e.grid_dim.2,
                e.block_dim.0,
                e.block_dim.1,
                e.block_dim.2,
                e.shared_memory_bytes,
            ));
            csv_opt_u32(&mut out, e.register_count);
            out.push(',');
            csv_opt_f64(&mut out, e.elapsed_ms);
            out.push(',');
            csv_opt_f64(&mut out, e.achieved_occupancy);
            out.push(',');
            csv_opt_f64(&mut out, e.theoretical_occupancy);
            out.push('\n');
        }
        out
    }

    /// Exports telemetry entries in Chrome `chrome://tracing` JSON format.
    ///
    /// Each kernel launch becomes a duration event (`ph: "X"`).
    /// Launches without timing data use a duration of 0.
    #[must_use]
    pub fn to_chrome_trace(entries: &[LaunchTelemetry]) -> String {
        let mut out = String::from("{\"traceEvents\":[\n");
        let mut ts_us = 0.0_f64; // cumulative timestamp in microseconds
        for (i, e) in entries.iter().enumerate() {
            let dur_us = e.elapsed_ms.unwrap_or(0.0) * 1000.0;
            out.push_str(&format!(
                "  {{\"name\":\"{}\",\"cat\":\"gpu\",\"ph\":\"X\",\
                 \"ts\":{:.3},\"dur\":{:.3},\"pid\":1,\"tid\":1,\
                 \"args\":{{\"grid\":\"{},{},{}\",\"block\":\"{},{},{}\",\
                 \"smem\":{}",
                json_escape_str(&e.kernel_name),
                ts_us,
                dur_us,
                e.grid_dim.0,
                e.grid_dim.1,
                e.grid_dim.2,
                e.block_dim.0,
                e.block_dim.1,
                e.block_dim.2,
                e.shared_memory_bytes,
            ));
            if let Some(regs) = e.register_count {
                out.push_str(&format!(",\"regs\":{regs}"));
            }
            if let Some(occ) = e.achieved_occupancy {
                out.push_str(&format!(",\"occupancy\":{occ:.4}"));
            }
            out.push_str("}}");
            if i + 1 < entries.len() {
                out.push(',');
            }
            out.push('\n');
            ts_us += dur_us;
        }
        out.push_str("]}\n");
        out
    }
}

// ---------------------------------------------------------------------------
// JSON / CSV helpers
// ---------------------------------------------------------------------------

fn json_escape_str(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn json_field_str(out: &mut String, key: &str, val: &str) {
    out.push_str(&format!("    \"{key}\": \"{}\",\n", json_escape_str(val)));
}

fn json_field_opt_u32(out: &mut String, key: &str, val: Option<u32>) {
    match val {
        Some(v) => out.push_str(&format!("    \"{key}\": {v},\n")),
        None => out.push_str(&format!("    \"{key}\": null,\n")),
    }
}

fn json_field_opt_f64(out: &mut String, key: &str, val: Option<f64>) {
    match val {
        Some(v) => out.push_str(&format!("    \"{key}\": {v},\n")),
        None => out.push_str(&format!("    \"{key}\": null,\n")),
    }
}

fn json_field_opt_f64_last(out: &mut String, key: &str, val: Option<f64>) {
    match val {
        Some(v) => out.push_str(&format!("    \"{key}\": {v}\n")),
        None => out.push_str(&format!("    \"{key}\": null\n")),
    }
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}

fn csv_opt_u32(out: &mut String, val: Option<u32>) {
    if let Some(v) = val {
        out.push_str(&v.to_string());
    }
}

fn csv_opt_f64(out: &mut String, val: Option<f64>) {
    if let Some(v) = val {
        out.push_str(&format!("{v}"));
    }
}

// ---------------------------------------------------------------------------
// Occupancy estimation
// ---------------------------------------------------------------------------

/// Estimates theoretical occupancy for a kernel launch configuration.
///
/// The occupancy is the ratio of active warps to the maximum possible
/// warps on a streaming multiprocessor. This depends on:
///
/// - `block_size`: threads per block
/// - `registers_per_thread`: registers consumed by each thread
/// - `shared_mem`: dynamic shared memory per block in bytes
/// - `sm_version`: target GPU architecture
///
/// Returns a value in the range `0.0..=1.0`.
///
/// # Example
///
/// ```
/// use oxicuda_launch::telemetry::{estimate_occupancy, SmVersion};
///
/// let occ = estimate_occupancy(256, 32, 0, SmVersion::Sm80);
/// assert!(occ > 0.0 && occ <= 1.0);
/// ```
#[must_use]
pub fn estimate_occupancy(
    block_size: u32,
    registers_per_thread: u32,
    shared_mem: u32,
    sm_version: SmVersion,
) -> f64 {
    if block_size == 0 {
        return 0.0;
    }

    let warp_size = sm_version.warp_size();
    let max_warps = sm_version.max_warps_per_sm();
    let max_blocks = sm_version.max_blocks_per_sm();
    let regs_per_sm = sm_version.registers_per_sm();
    let max_smem = sm_version.max_shared_mem_per_sm();
    let reg_granularity = sm_version.register_alloc_granularity();
    let smem_granularity = sm_version.shared_mem_alloc_granularity();

    // Warps per block
    let warps_per_block = block_size.div_ceil(warp_size);

    // --- Register limit ---
    let regs_per_thread = if registers_per_thread == 0 {
        1 // must use at least 1 register
    } else {
        registers_per_thread
    };
    // Round up to allocation granularity
    let regs_alloc = regs_per_thread.div_ceil(reg_granularity) * reg_granularity;
    let regs_per_warp = regs_alloc * warp_size;
    let warps_limited_by_regs = regs_per_sm.checked_div(regs_per_warp).unwrap_or(max_warps);

    // --- Shared memory limit ---
    let smem_per_block = if shared_mem == 0 {
        0
    } else {
        shared_mem.div_ceil(smem_granularity) * smem_granularity
    };
    let blocks_limited_by_smem = max_smem.checked_div(smem_per_block).unwrap_or(max_blocks);

    // --- Block limit ---
    let blocks_by_warps = warps_limited_by_regs
        .checked_div(warps_per_block)
        .unwrap_or(max_blocks);

    let active_blocks = max_blocks.min(blocks_by_warps).min(blocks_limited_by_smem);

    let active_warps = active_blocks * warps_per_block;
    let occupancy = active_warps as f64 / max_warps as f64;

    occupancy.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- LaunchTelemetry construction and builder --

    #[test]
    fn telemetry_new_defaults() {
        let t = LaunchTelemetry::new("kern", (4, 1, 1), (256, 1, 1));
        assert_eq!(t.kernel_name, "kern");
        assert_eq!(t.grid_dim, (4, 1, 1));
        assert_eq!(t.block_dim, (256, 1, 1));
        assert_eq!(t.shared_memory_bytes, 0);
        assert!(t.register_count.is_none());
        assert!(t.elapsed_ms.is_none());
        assert!(t.achieved_occupancy.is_none());
        assert!(t.theoretical_occupancy.is_none());
    }

    #[test]
    fn telemetry_builder_methods() {
        let t = LaunchTelemetry::new("kern", (1, 1, 1), (128, 1, 1))
            .with_shared_memory(4096)
            .with_register_count(32)
            .with_elapsed_ms(1.5)
            .with_achieved_occupancy(0.75)
            .with_theoretical_occupancy(0.80);

        assert_eq!(t.shared_memory_bytes, 4096);
        assert_eq!(t.register_count, Some(32));
        assert!((t.elapsed_ms.unwrap_or(0.0) - 1.5).abs() < f64::EPSILON);
        assert!((t.achieved_occupancy.unwrap_or(0.0) - 0.75).abs() < f64::EPSILON);
        assert!((t.theoretical_occupancy.unwrap_or(0.0) - 0.80).abs() < f64::EPSILON);
    }

    #[test]
    fn telemetry_total_threads() {
        let t = LaunchTelemetry::new("k", (4, 2, 1), (16, 16, 1));
        assert_eq!(t.total_threads(), 4 * 2 * 16 * 16);
    }

    #[test]
    fn telemetry_display() {
        let t = LaunchTelemetry::new("add", (4, 1, 1), (256, 1, 1))
            .with_elapsed_ms(0.5)
            .with_register_count(24)
            .with_achieved_occupancy(0.85);
        let s = format!("{t}");
        assert!(s.contains("add"));
        assert!(s.contains("0.500ms"));
        assert!(s.contains("regs=24"));
        assert!(s.contains("85.0%"));
    }

    // -- TelemetryCollector --

    #[test]
    fn collector_record_and_len() {
        let mut c = TelemetryCollector::new(100);
        assert!(c.is_empty());
        c.record(LaunchTelemetry::new("k", (1, 1, 1), (64, 1, 1)));
        assert_eq!(c.len(), 1);
        assert!(!c.is_empty());
    }

    #[test]
    fn collector_enable_disable() {
        let mut c = TelemetryCollector::new(100);
        assert!(c.is_enabled());

        c.disable();
        assert!(!c.is_enabled());
        c.record(LaunchTelemetry::new("k", (1, 1, 1), (64, 1, 1)));
        assert_eq!(c.len(), 0); // dropped because disabled

        c.enable();
        c.record(LaunchTelemetry::new("k", (1, 1, 1), (64, 1, 1)));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn collector_max_entries_cap() {
        let mut c = TelemetryCollector::new(3);
        for _ in 0..10 {
            c.record(LaunchTelemetry::new("k", (1, 1, 1), (64, 1, 1)));
        }
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn collector_clear() {
        let mut c = TelemetryCollector::new(100);
        c.record(LaunchTelemetry::new("k", (1, 1, 1), (64, 1, 1)));
        c.clear();
        assert!(c.is_empty());
    }

    // -- TelemetrySummary --

    #[test]
    fn summary_empty() {
        let c = TelemetryCollector::new(100);
        let s = c.summary();
        assert_eq!(s.total_launches, 0);
        assert!((s.total_gpu_time_ms).abs() < f64::EPSILON);
        assert!(s.hottest_kernel.is_none());
        assert!(s.per_kernel_stats.is_empty());
    }

    #[test]
    fn summary_single_kernel() {
        let mut c = TelemetryCollector::new(100);
        c.record(
            LaunchTelemetry::new("add", (1, 1, 1), (256, 1, 1))
                .with_elapsed_ms(1.0)
                .with_achieved_occupancy(0.8),
        );
        c.record(
            LaunchTelemetry::new("add", (1, 1, 1), (256, 1, 1))
                .with_elapsed_ms(3.0)
                .with_achieved_occupancy(0.9),
        );
        let s = c.summary();
        assert_eq!(s.total_launches, 2);
        assert!((s.total_gpu_time_ms - 4.0).abs() < f64::EPSILON);
        assert!((s.avg_gpu_time_ms - 2.0).abs() < f64::EPSILON);
        assert!((s.min_gpu_time_ms - 1.0).abs() < f64::EPSILON);
        assert!((s.max_gpu_time_ms - 3.0).abs() < f64::EPSILON);
        assert!((s.avg_occupancy - 0.85).abs() < 1e-9);
        assert_eq!(s.hottest_kernel.as_deref(), Some("add"));
        assert_eq!(s.per_kernel_stats.len(), 1);
        assert_eq!(s.per_kernel_stats[0].launch_count, 2);
    }

    #[test]
    fn summary_per_kernel_aggregation() {
        let mut c = TelemetryCollector::new(100);
        c.record(LaunchTelemetry::new("matmul", (1, 1, 1), (256, 1, 1)).with_elapsed_ms(10.0));
        c.record(LaunchTelemetry::new("add", (1, 1, 1), (128, 1, 1)).with_elapsed_ms(1.0));
        c.record(LaunchTelemetry::new("matmul", (1, 1, 1), (256, 1, 1)).with_elapsed_ms(12.0));

        let s = c.summary();
        assert_eq!(s.total_launches, 3);
        // hottest should be matmul (22ms > 1ms)
        assert_eq!(s.hottest_kernel.as_deref(), Some("matmul"));
        assert_eq!(s.per_kernel_stats.len(), 2);
        // First entry should be matmul (sorted by total time desc)
        assert_eq!(s.per_kernel_stats[0].kernel_name, "matmul");
        assert_eq!(s.per_kernel_stats[0].launch_count, 2);
        assert!((s.per_kernel_stats[0].total_time_ms - 22.0).abs() < f64::EPSILON);
    }

    #[test]
    fn summary_display() {
        let mut c = TelemetryCollector::new(100);
        c.record(
            LaunchTelemetry::new("k", (1, 1, 1), (256, 1, 1))
                .with_elapsed_ms(2.0)
                .with_achieved_occupancy(0.5),
        );
        let s = c.summary();
        let text = format!("{s}");
        assert!(text.contains("Telemetry Summary"));
        assert!(text.contains("Total launches: 1"));
        assert!(text.contains("50.0%"));
    }

    // -- TelemetryExporter: JSON --

    #[test]
    fn export_json() {
        let entries = vec![
            LaunchTelemetry::new("kern", (4, 1, 1), (256, 1, 1))
                .with_elapsed_ms(0.5)
                .with_register_count(32),
        ];
        let json = TelemetryExporter::to_json(&entries);
        assert!(json.starts_with('['));
        assert!(json.contains("\"kernel_name\": \"kern\""));
        assert!(json.contains("\"grid_dim\": [4, 1, 1]"));
        assert!(json.contains("\"elapsed_ms\": 0.5"));
        assert!(json.contains("\"register_count\": 32"));
        assert!(json.contains("\"achieved_occupancy\": null"));
    }

    // -- TelemetryExporter: CSV --

    #[test]
    fn export_csv() {
        let entries =
            vec![LaunchTelemetry::new("kern", (2, 1, 1), (128, 1, 1)).with_elapsed_ms(1.0)];
        let csv = TelemetryExporter::to_csv(&entries);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2); // header + 1 data row
        assert!(lines[0].starts_with("kernel_name,"));
        assert!(lines[1].starts_with("kern,"));
        assert!(lines[1].contains("128"));
    }

    // -- TelemetryExporter: Chrome trace --

    #[test]
    fn export_chrome_trace() {
        let entries = vec![
            LaunchTelemetry::new("k1", (1, 1, 1), (256, 1, 1)).with_elapsed_ms(1.0),
            LaunchTelemetry::new("k2", (2, 1, 1), (128, 1, 1)).with_elapsed_ms(2.0),
        ];
        let trace = TelemetryExporter::to_chrome_trace(&entries);
        assert!(trace.contains("\"traceEvents\""));
        assert!(trace.contains("\"name\":\"k1\""));
        assert!(trace.contains("\"name\":\"k2\""));
        assert!(trace.contains("\"ph\":\"X\""));
        assert!(trace.contains("\"cat\":\"gpu\""));
    }

    // -- Occupancy estimation --

    #[test]
    fn occupancy_basic() {
        let occ = estimate_occupancy(256, 32, 0, SmVersion::Sm80);
        assert!(occ > 0.0);
        assert!(occ <= 1.0);
    }

    #[test]
    fn occupancy_zero_block() {
        let occ = estimate_occupancy(0, 32, 0, SmVersion::Sm80);
        assert!((occ).abs() < f64::EPSILON);
    }

    #[test]
    fn occupancy_high_registers_lowers_occupancy() {
        let high_reg = estimate_occupancy(256, 128, 0, SmVersion::Sm80);
        let low_reg = estimate_occupancy(256, 16, 0, SmVersion::Sm80);
        assert!(high_reg < low_reg);
    }

    #[test]
    fn occupancy_large_shared_mem_lowers_occupancy() {
        let large_smem = estimate_occupancy(256, 32, 100_000, SmVersion::Sm80);
        let small_smem = estimate_occupancy(256, 32, 0, SmVersion::Sm80);
        assert!(large_smem <= small_smem);
    }

    #[test]
    fn occupancy_sm_versions() {
        for sm in [
            SmVersion::Sm75,
            SmVersion::Sm80,
            SmVersion::Sm86,
            SmVersion::Sm89,
            SmVersion::Sm90,
            SmVersion::Sm100,
            SmVersion::Sm120,
        ] {
            let occ = estimate_occupancy(128, 32, 0, sm);
            assert!(occ > 0.0, "occupancy should be positive for {sm}");
            assert!(occ <= 1.0, "occupancy should be <= 1.0 for {sm}");
        }
    }

    // -- SmVersion --

    #[test]
    fn sm_version_display() {
        assert_eq!(format!("{}", SmVersion::Sm80), "sm_80");
        assert_eq!(format!("{}", SmVersion::Sm90), "sm_90");
    }

    // -- KernelStats Display --

    #[test]
    fn kernel_stats_display() {
        let ks = KernelStats {
            kernel_name: "matmul".to_owned(),
            launch_count: 5,
            total_time_ms: 10.0,
            avg_time_ms: 2.0,
            min_time_ms: 1.0,
            max_time_ms: 4.0,
            avg_occupancy: 0.75,
        };
        let s = format!("{ks}");
        assert!(s.contains("matmul"));
        assert!(s.contains("5 launches"));
        assert!(s.contains("75.0%"));
    }
}
