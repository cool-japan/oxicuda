//! Kernel debugging utilities for OxiCUDA.
//!
//! This module provides tools for debugging GPU kernels without traditional
//! debuggers. It includes memory checking, NaN/Inf detection, printf
//! emulation, assertion support, and PTX instrumentation for automated
//! bounds/NaN checking.
//!
//! # Architecture
//!
//! The debugging system is layered:
//!
//! 1. **`KernelDebugger`** — Top-level manager that creates debug sessions and
//!    manages breakpoints / watchpoints.
//! 2. **`DebugSession`** — Collects [`DebugEvent`]s for a single kernel launch.
//! 3. **`MemoryChecker`** — Validates memory accesses against known allocations.
//! 4. **`NanInfChecker`** — Scans host-side buffers for NaN / Inf values.
//! 5. **`PrintfBuffer`** — Parses a raw byte buffer that emulates GPU `printf`.
//! 6. **`KernelAssertions`** — Convenience assertion helpers that produce
//!    [`DebugEvent`]s instead of panicking.
//! 7. **`DebugPtxInstrumenter`** — Instruments PTX source for automated
//!    bounds/NaN checking and printf support.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_driver::debug::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = KernelDebugConfig::default();
//! let mut debugger = KernelDebugger::new(config);
//! let session = debugger.attach("my_kernel")?;
//! assert_eq!(session.kernel_name(), "my_kernel");
//! # Ok(())
//! # }
//! ```

use std::fmt;

use crate::error::{CudaError, CudaResult};

// ---------------------------------------------------------------------------
// DebugLevel
// ---------------------------------------------------------------------------

/// Verbosity level for kernel debugging output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum DebugLevel {
    /// No debug output.
    Off,
    /// Only errors.
    Error,
    /// Errors and warnings.
    Warn,
    /// Errors, warnings, and informational messages.
    #[default]
    Info,
    /// Verbose debugging output.
    Debug,
    /// Maximum verbosity — every detail is logged.
    Trace,
}

impl fmt::Display for DebugLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => write!(f, "OFF"),
            Self::Error => write!(f, "ERROR"),
            Self::Warn => write!(f, "WARN"),
            Self::Info => write!(f, "INFO"),
            Self::Debug => write!(f, "DEBUG"),
            Self::Trace => write!(f, "TRACE"),
        }
    }
}

// ---------------------------------------------------------------------------
// KernelDebugConfig
// ---------------------------------------------------------------------------

/// Configuration for a kernel debug session.
#[derive(Debug, Clone)]
pub struct KernelDebugConfig {
    /// The verbosity level for debug output.
    pub debug_level: DebugLevel,
    /// Whether to instrument bounds checking on memory accesses.
    pub enable_bounds_check: bool,
    /// Whether to detect NaN values in floating-point registers.
    pub enable_nan_check: bool,
    /// Whether to detect Inf values in floating-point registers.
    pub enable_inf_check: bool,
    /// Whether to detect potential race conditions.
    pub enable_race_detection: bool,
    /// Size of the GPU-side printf buffer in bytes.
    pub print_buffer_size: usize,
    /// Maximum number of printf calls per thread before truncation.
    pub max_print_per_thread: usize,
}

impl Default for KernelDebugConfig {
    fn default() -> Self {
        Self {
            debug_level: DebugLevel::Info,
            enable_bounds_check: true,
            enable_nan_check: true,
            enable_inf_check: true,
            enable_race_detection: false,
            print_buffer_size: 1024 * 1024, // 1 MiB
            max_print_per_thread: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// DebugEventType
// ---------------------------------------------------------------------------

/// The kind of debug event captured during kernel execution.
#[derive(Debug, Clone, PartialEq)]
pub enum DebugEventType {
    /// A memory access was out of the allocated bounds.
    OutOfBounds {
        /// The faulting address.
        address: u64,
        /// The size of the attempted access in bytes.
        size: usize,
    },
    /// A NaN was detected in a floating-point register.
    NanDetected {
        /// Register or variable name.
        register: String,
        /// The NaN bit-pattern reinterpreted as f64.
        value: f64,
    },
    /// An infinity was detected in a floating-point register.
    InfDetected {
        /// Register or variable name.
        register: String,
    },
    /// A potential race condition on a shared memory address.
    RaceCondition {
        /// The conflicting address.
        address: u64,
    },
    /// A kernel-side assertion.
    Assertion {
        /// The assertion condition expression.
        condition: String,
        /// Source file name.
        file: String,
        /// Source line number.
        line: u32,
    },
    /// A kernel-side printf invocation.
    Printf {
        /// The format string.
        format: String,
    },
    /// A breakpoint was hit.
    Breakpoint {
        /// The breakpoint identifier.
        id: u32,
    },
}

impl DebugEventType {
    /// Returns a short category tag suitable for filtering.
    fn tag(&self) -> &'static str {
        match self {
            Self::OutOfBounds { .. } => "OOB",
            Self::NanDetected { .. } => "NaN",
            Self::InfDetected { .. } => "Inf",
            Self::RaceCondition { .. } => "RACE",
            Self::Assertion { .. } => "ASSERT",
            Self::Printf { .. } => "PRINTF",
            Self::Breakpoint { .. } => "BP",
        }
    }

    /// Returns `true` when this variant has the same discriminant as `other`,
    /// ignoring inner field values. Used by [`DebugSession::filter_events`].
    fn same_kind(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

// ---------------------------------------------------------------------------
// DebugEvent
// ---------------------------------------------------------------------------

/// A single debug event captured during kernel execution.
#[derive(Debug, Clone)]
pub struct DebugEvent {
    /// What kind of event occurred.
    pub event_type: DebugEventType,
    /// The CUDA thread index `(x, y, z)` that triggered the event.
    pub thread_id: (u32, u32, u32),
    /// The CUDA block index `(x, y, z)` that triggered the event.
    pub block_id: (u32, u32, u32),
    /// Timestamp in nanoseconds (monotonic, relative to session start).
    pub timestamp_ns: u64,
    /// Free-form human-readable message.
    pub message: String,
}

impl fmt::Display for DebugEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{tag}] block({bx},{by},{bz}) thread({tx},{ty},{tz}) @{ts}ns: {msg}",
            tag = self.event_type.tag(),
            bx = self.block_id.0,
            by = self.block_id.1,
            bz = self.block_id.2,
            tx = self.thread_id.0,
            ty = self.thread_id.1,
            tz = self.thread_id.2,
            ts = self.timestamp_ns,
            msg = self.message,
        )
    }
}

// ---------------------------------------------------------------------------
// WatchType
// ---------------------------------------------------------------------------

/// The kind of memory access a watchpoint monitors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WatchType {
    /// Trigger on reads.
    Read,
    /// Trigger on writes.
    Write,
    /// Trigger on both reads and writes.
    ReadWrite,
}

// ---------------------------------------------------------------------------
// Breakpoint / Watchpoint helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Breakpoint {
    id: u32,
    line: u32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Watchpoint {
    id: u32,
    address: u64,
    size: usize,
    watch_type: WatchType,
}

// ---------------------------------------------------------------------------
// KernelDebugger
// ---------------------------------------------------------------------------

/// Top-level kernel debugging manager.
///
/// Create one per debugging session group. It holds global breakpoints and
/// watchpoints and spawns [`DebugSession`] instances per kernel launch.
#[derive(Debug)]
pub struct KernelDebugger {
    config: KernelDebugConfig,
    breakpoints: Vec<Breakpoint>,
    watchpoints: Vec<Watchpoint>,
    next_bp_id: u32,
    next_wp_id: u32,
}

impl KernelDebugger {
    /// Create a new kernel debugger with the given configuration.
    pub fn new(config: KernelDebugConfig) -> Self {
        Self {
            config,
            breakpoints: Vec::new(),
            watchpoints: Vec::new(),
            next_bp_id: 1,
            next_wp_id: 1,
        }
    }

    /// Attach the debugger to a kernel launch, returning a new debug session.
    ///
    /// On macOS this always succeeds with a synthetic (empty) session because
    /// no actual GPU driver is available.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if `kernel_name` is empty.
    pub fn attach(&mut self, kernel_name: &str) -> CudaResult<DebugSession> {
        if kernel_name.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        Ok(DebugSession {
            kernel_name: kernel_name.to_owned(),
            events: Vec::new(),
            config: self.config.clone(),
        })
    }

    /// Set a breakpoint at a PTX source line. Returns the breakpoint ID.
    pub fn set_breakpoint(&mut self, line: u32) -> u32 {
        let id = self.next_bp_id;
        self.next_bp_id = self.next_bp_id.saturating_add(1);
        self.breakpoints.push(Breakpoint { id, line });
        id
    }

    /// Remove a breakpoint by ID. Returns `true` if it was found and removed.
    pub fn remove_breakpoint(&mut self, bp_id: u32) -> bool {
        let before = self.breakpoints.len();
        self.breakpoints.retain(|bp| bp.id != bp_id);
        self.breakpoints.len() < before
    }

    /// Set a memory watchpoint. Returns the watchpoint ID.
    pub fn watchpoint(&mut self, address: u64, size: usize, watch_type: WatchType) -> u32 {
        let id = self.next_wp_id;
        self.next_wp_id = self.next_wp_id.saturating_add(1);
        self.watchpoints.push(Watchpoint {
            id,
            address,
            size,
            watch_type,
        });
        id
    }

    /// Returns a reference to the current debug configuration.
    pub fn config(&self) -> &KernelDebugConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// DebugSummary
// ---------------------------------------------------------------------------

/// Aggregate statistics for a debug session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DebugSummary {
    /// Total number of debug events.
    pub total_events: usize,
    /// Number of error-level events (OOB, assertions, races).
    pub errors: usize,
    /// Number of warning-level events (NaN, Inf).
    pub warnings: usize,
    /// Number of NaN detection events.
    pub nan_count: usize,
    /// Number of Inf detection events.
    pub inf_count: usize,
    /// Number of out-of-bounds events.
    pub oob_count: usize,
    /// Number of race condition events.
    pub race_count: usize,
}

// ---------------------------------------------------------------------------
// DebugSession
// ---------------------------------------------------------------------------

/// An active debug session for a single kernel launch.
///
/// Collects [`DebugEvent`]s and provides analysis / reporting helpers.
#[derive(Debug)]
pub struct DebugSession {
    kernel_name: String,
    events: Vec<DebugEvent>,
    #[allow(dead_code)]
    config: KernelDebugConfig,
}

impl DebugSession {
    /// The kernel name this session is attached to.
    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    /// All events collected so far.
    pub fn events(&self) -> &[DebugEvent] {
        &self.events
    }

    /// Record a new debug event.
    pub fn add_event(&mut self, event: DebugEvent) {
        self.events.push(event);
    }

    /// Return references to events whose type matches the discriminant of
    /// `event_type` (field values inside variants are ignored for matching).
    pub fn filter_events(&self, event_type: &DebugEventType) -> Vec<&DebugEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type.same_kind(event_type))
            .collect()
    }

    /// Compute aggregate statistics over all collected events.
    pub fn summary(&self) -> DebugSummary {
        let mut s = DebugSummary {
            total_events: self.events.len(),
            ..DebugSummary::default()
        };
        for ev in &self.events {
            match &ev.event_type {
                DebugEventType::OutOfBounds { .. } => {
                    s.oob_count += 1;
                    s.errors += 1;
                }
                DebugEventType::NanDetected { .. } => {
                    s.nan_count += 1;
                    s.warnings += 1;
                }
                DebugEventType::InfDetected { .. } => {
                    s.inf_count += 1;
                    s.warnings += 1;
                }
                DebugEventType::RaceCondition { .. } => {
                    s.race_count += 1;
                    s.errors += 1;
                }
                DebugEventType::Assertion { .. } => {
                    s.errors += 1;
                }
                DebugEventType::Printf { .. } | DebugEventType::Breakpoint { .. } => {}
            }
        }
        s
    }

    /// Produce a human-readable debug report.
    pub fn format_report(&self) -> String {
        let summary = self.summary();
        let mut out = String::with_capacity(512);
        out.push_str(&format!("=== Debug Report: {} ===\n", self.kernel_name));
        out.push_str(&format!(
            "Total events: {}  (errors: {}, warnings: {})\n",
            summary.total_events, summary.errors, summary.warnings
        ));
        if summary.oob_count > 0 {
            out.push_str(&format!("  Out-of-bounds: {}\n", summary.oob_count));
        }
        if summary.nan_count > 0 {
            out.push_str(&format!("  NaN detected:  {}\n", summary.nan_count));
        }
        if summary.inf_count > 0 {
            out.push_str(&format!("  Inf detected:  {}\n", summary.inf_count));
        }
        if summary.race_count > 0 {
            out.push_str(&format!("  Race cond.:    {}\n", summary.race_count));
        }
        out.push_str("--- Events ---\n");
        for ev in &self.events {
            out.push_str(&format!("  {ev}\n"));
        }
        out.push_str("=== End Report ===\n");
        out
    }
}

// ---------------------------------------------------------------------------
// MemoryRegion / MemoryChecker
// ---------------------------------------------------------------------------

/// A contiguous GPU memory allocation known to the memory checker.
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Base device address of the allocation.
    pub base_address: u64,
    /// Size in bytes.
    pub size: usize,
    /// Human-readable name for diagnostics.
    pub name: String,
    /// Whether the allocation is read-only.
    pub is_readonly: bool,
}

/// Validates memory accesses against a set of known [`MemoryRegion`]s.
#[derive(Debug)]
pub struct MemoryChecker {
    allocations: Vec<MemoryRegion>,
}

impl MemoryChecker {
    /// Create a memory checker from a list of known allocations.
    pub fn new(allocations: Vec<MemoryRegion>) -> Self {
        Self { allocations }
    }

    /// Check whether a memory access is valid.
    ///
    /// Returns `Some(DebugEvent)` if the access is out of bounds or violates
    /// read-only protections; `None` if the access is valid.
    pub fn check_access(&self, address: u64, size: usize, is_write: bool) -> Option<DebugEvent> {
        // Find the allocation that contains this address.
        let region = self.allocations.iter().find(|r| {
            address >= r.base_address
                && address
                    .checked_add(size as u64)
                    .is_some_and(|end| end <= r.base_address + r.size as u64)
        });

        match region {
            Some(r) if is_write && r.is_readonly => Some(DebugEvent {
                event_type: DebugEventType::OutOfBounds { address, size },
                thread_id: (0, 0, 0),
                block_id: (0, 0, 0),
                timestamp_ns: 0,
                message: format!("Write to read-only region '{}' at {:#x}", r.name, address),
            }),
            Some(_) => None,
            None => Some(DebugEvent {
                event_type: DebugEventType::OutOfBounds { address, size },
                thread_id: (0, 0, 0),
                block_id: (0, 0, 0),
                timestamp_ns: 0,
                message: format!(
                    "Access at {:#x} (size {}) does not fall within any known allocation",
                    address, size
                ),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// NanInfChecker / NanInfLocation
// ---------------------------------------------------------------------------

/// Location of a NaN or Inf value found in a buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct NanInfLocation {
    /// Index into the buffer.
    pub index: usize,
    /// The problematic value (as f64 for uniform reporting).
    pub value: f64,
    /// `true` if NaN, `false` if Inf.
    pub is_nan: bool,
}

/// Scans host-side buffers for NaN and Inf values.
#[derive(Debug, Clone, Copy)]
pub struct NanInfChecker;

impl NanInfChecker {
    /// Check an `f32` buffer for NaN and Inf values.
    pub fn check_f32(data: &[f32]) -> Vec<NanInfLocation> {
        data.iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                if v.is_nan() {
                    Some(NanInfLocation {
                        index: i,
                        value: f64::from(v),
                        is_nan: true,
                    })
                } else if v.is_infinite() {
                    Some(NanInfLocation {
                        index: i,
                        value: f64::from(v),
                        is_nan: false,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check an `f64` buffer for NaN and Inf values.
    pub fn check_f64(data: &[f64]) -> Vec<NanInfLocation> {
        data.iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                if v.is_nan() {
                    Some(NanInfLocation {
                        index: i,
                        value: v,
                        is_nan: true,
                    })
                } else if v.is_infinite() {
                    Some(NanInfLocation {
                        index: i,
                        value: v,
                        is_nan: false,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// PrintfBuffer / PrintfEntry / PrintfArg
// ---------------------------------------------------------------------------

/// A single argument captured from a GPU printf call.
#[derive(Debug, Clone, PartialEq)]
pub enum PrintfArg {
    /// Integer argument.
    Int(i64),
    /// Floating-point argument.
    Float(f64),
    /// String argument.
    String(String),
}

/// A parsed GPU-side printf entry.
#[derive(Debug, Clone)]
pub struct PrintfEntry {
    /// Thread index that issued the printf.
    pub thread_id: (u32, u32, u32),
    /// Block index that issued the printf.
    pub block_id: (u32, u32, u32),
    /// The format string.
    pub format_string: String,
    /// Parsed arguments.
    pub args: Vec<PrintfArg>,
}

/// GPU-side printf emulation buffer.
///
/// The buffer layout is a simple framed protocol:
///
/// ```text
/// [entry_count: u32_le]
/// repeated entry_count times:
///   [thread_x: u32_le] [thread_y: u32_le] [thread_z: u32_le]
///   [block_x:  u32_le] [block_y:  u32_le] [block_z:  u32_le]
///   [fmt_len:  u32_le] [fmt_bytes: u8 * fmt_len]
///   [arg_count: u32_le]
///   repeated arg_count times:
///     [tag: u8]  0=Int(i64_le), 1=Float(f64_le), 2=String(u32_le len + bytes)
/// ```
#[derive(Debug)]
pub struct PrintfBuffer {
    buffer_size: usize,
}

impl PrintfBuffer {
    /// Create a printf buffer descriptor with the given maximum size.
    pub fn new(buffer_size: usize) -> Self {
        Self { buffer_size }
    }

    /// Returns the configured buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Parse a raw byte buffer into structured printf entries.
    ///
    /// Returns an empty vec if the buffer is too small or malformed.
    pub fn parse_entries(&self, raw: &[u8]) -> Vec<PrintfEntry> {
        let mut entries = Vec::new();
        let mut cursor = 0usize;

        let entry_count = match Self::read_u32(raw, &mut cursor) {
            Some(n) => n as usize,
            None => return entries,
        };

        for _ in 0..entry_count {
            let Some(entry) = self.parse_single_entry(raw, &mut cursor) else {
                break;
            };
            entries.push(entry);
        }

        entries
    }

    fn parse_single_entry(&self, raw: &[u8], cursor: &mut usize) -> Option<PrintfEntry> {
        let tx = Self::read_u32(raw, cursor)?;
        let ty = Self::read_u32(raw, cursor)?;
        let tz = Self::read_u32(raw, cursor)?;
        let bx = Self::read_u32(raw, cursor)?;
        let by = Self::read_u32(raw, cursor)?;
        let bz = Self::read_u32(raw, cursor)?;

        let fmt_len = Self::read_u32(raw, cursor)? as usize;
        let fmt_bytes = Self::read_bytes(raw, cursor, fmt_len)?;
        let format_string = String::from_utf8_lossy(fmt_bytes).into_owned();

        let arg_count = Self::read_u32(raw, cursor)? as usize;
        let mut args = Vec::with_capacity(arg_count);
        for _ in 0..arg_count {
            let tag = Self::read_u8(raw, cursor)?;
            let arg = match tag {
                0 => {
                    let val = Self::read_i64(raw, cursor)?;
                    PrintfArg::Int(val)
                }
                1 => {
                    let val = Self::read_f64(raw, cursor)?;
                    PrintfArg::Float(val)
                }
                2 => {
                    let slen = Self::read_u32(raw, cursor)? as usize;
                    let sbytes = Self::read_bytes(raw, cursor, slen)?;
                    PrintfArg::String(String::from_utf8_lossy(sbytes).into_owned())
                }
                _ => return None,
            };
            args.push(arg);
        }

        Some(PrintfEntry {
            thread_id: (tx, ty, tz),
            block_id: (bx, by, bz),
            format_string,
            args,
        })
    }

    // --- Low-level readers ---

    fn read_u8(raw: &[u8], cursor: &mut usize) -> Option<u8> {
        if *cursor >= raw.len() {
            return None;
        }
        let val = raw[*cursor];
        *cursor += 1;
        Some(val)
    }

    fn read_u32(raw: &[u8], cursor: &mut usize) -> Option<u32> {
        if *cursor + 4 > raw.len() {
            return None;
        }
        let bytes: [u8; 4] = raw[*cursor..*cursor + 4].try_into().ok()?;
        *cursor += 4;
        Some(u32::from_le_bytes(bytes))
    }

    fn read_i64(raw: &[u8], cursor: &mut usize) -> Option<i64> {
        if *cursor + 8 > raw.len() {
            return None;
        }
        let bytes: [u8; 8] = raw[*cursor..*cursor + 8].try_into().ok()?;
        *cursor += 8;
        Some(i64::from_le_bytes(bytes))
    }

    fn read_f64(raw: &[u8], cursor: &mut usize) -> Option<f64> {
        if *cursor + 8 > raw.len() {
            return None;
        }
        let bytes: [u8; 8] = raw[*cursor..*cursor + 8].try_into().ok()?;
        *cursor += 8;
        Some(f64::from_le_bytes(bytes))
    }

    fn read_bytes<'a>(raw: &'a [u8], cursor: &mut usize, len: usize) -> Option<&'a [u8]> {
        if *cursor + len > raw.len() {
            return None;
        }
        let slice = &raw[*cursor..*cursor + len];
        *cursor += len;
        Some(slice)
    }
}

// ---------------------------------------------------------------------------
// KernelAssertions
// ---------------------------------------------------------------------------

/// Convenience assertion helpers that produce [`DebugEvent`]s instead of
/// panicking, suitable for GPU kernel emulation / validation.
#[derive(Debug, Clone, Copy)]
pub struct KernelAssertions;

impl KernelAssertions {
    /// Assert that `index < len`. Returns an event if the assertion fails.
    pub fn assert_bounds(index: usize, len: usize, name: &str) -> Option<DebugEvent> {
        if index < len {
            return None;
        }
        Some(DebugEvent {
            event_type: DebugEventType::Assertion {
                condition: format!("{name}[{index}] < {len}"),
                file: String::new(),
                line: 0,
            },
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            timestamp_ns: 0,
            message: format!("Bounds check failed: {name}[{index}] out of range (len={len})"),
        })
    }

    /// Assert that `value` is not NaN. Returns an event if it is.
    pub fn assert_not_nan(value: f64, name: &str) -> Option<DebugEvent> {
        if !value.is_nan() {
            return None;
        }
        Some(DebugEvent {
            event_type: DebugEventType::NanDetected {
                register: name.to_owned(),
                value,
            },
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            timestamp_ns: 0,
            message: format!("NaN detected in '{name}'"),
        })
    }

    /// Assert that `value` is finite (not NaN and not Inf). Returns an event
    /// if it is not.
    pub fn assert_finite(value: f64, name: &str) -> Option<DebugEvent> {
        if value.is_finite() {
            return None;
        }
        if value.is_nan() {
            return Some(DebugEvent {
                event_type: DebugEventType::NanDetected {
                    register: name.to_owned(),
                    value,
                },
                thread_id: (0, 0, 0),
                block_id: (0, 0, 0),
                timestamp_ns: 0,
                message: format!("Non-finite (NaN) value in '{name}'"),
            });
        }
        Some(DebugEvent {
            event_type: DebugEventType::InfDetected {
                register: name.to_owned(),
            },
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            timestamp_ns: 0,
            message: format!("Non-finite (Inf) value in '{name}'"),
        })
    }

    /// Assert that `value` is strictly positive. Returns an event if it is
    /// not (including NaN, zero, and negative values).
    pub fn assert_positive(value: f64, name: &str) -> Option<DebugEvent> {
        if value > 0.0 {
            return None;
        }
        if value.is_nan() {
            return Some(DebugEvent {
                event_type: DebugEventType::NanDetected {
                    register: name.to_owned(),
                    value,
                },
                thread_id: (0, 0, 0),
                block_id: (0, 0, 0),
                timestamp_ns: 0,
                message: format!("Expected positive value for '{name}', got NaN"),
            });
        }
        Some(DebugEvent {
            event_type: DebugEventType::Assertion {
                condition: format!("{name} > 0"),
                file: String::new(),
                line: 0,
            },
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            timestamp_ns: 0,
            message: format!("Expected positive value for '{name}', got {value}"),
        })
    }
}

// ---------------------------------------------------------------------------
// DebugPtxInstrumenter
// ---------------------------------------------------------------------------

/// Instruments PTX source code with debugging checks.
///
/// This instrumenter inserts additional PTX instructions for bounds checking,
/// NaN detection, and printf buffer support. The instrumented code writes
/// diagnostic data to a designated debug buffer that the host can read back
/// after kernel execution.
#[derive(Debug)]
pub struct DebugPtxInstrumenter {
    enable_bounds_check: bool,
    enable_nan_check: bool,
    enable_printf: bool,
}

impl DebugPtxInstrumenter {
    /// Create an instrumenter from a debug configuration.
    pub fn new(config: &KernelDebugConfig) -> Self {
        Self {
            enable_bounds_check: config.enable_bounds_check,
            enable_nan_check: config.enable_nan_check,
            enable_printf: config.print_buffer_size > 0,
        }
    }

    /// Insert bounds-checking instrumentation into PTX source.
    ///
    /// Adds `setp` + `trap` sequences after every `ld.global` / `st.global`
    /// instruction to validate the address against the allocation size
    /// parameter.
    pub fn instrument_bounds_checks(&self, ptx: &str) -> String {
        if !self.enable_bounds_check {
            return ptx.to_owned();
        }

        let mut output = String::with_capacity(ptx.len() + ptx.len() / 4);
        // Add debug parameter declaration at the top of each kernel entry.
        let mut added_param = false;

        for line in ptx.lines() {
            let trimmed = line.trim();
            // Insert debug buffer param after .entry
            if trimmed.starts_with(".entry") && !added_param {
                output.push_str(line);
                output.push('\n');
                output.push_str("    // [oxicuda-debug] bounds-check instrumentation\n");
                output.push_str("    .param .u64 __oxicuda_debug_buf;\n");
                added_param = true;
                continue;
            }

            // Instrument global loads/stores
            if (trimmed.starts_with("ld.global") || trimmed.starts_with("st.global"))
                && !trimmed.starts_with("// [oxicuda-debug]")
            {
                output.push_str(line);
                output.push('\n');
                output.push_str("    // [oxicuda-debug] bounds check for above access\n");
                output.push_str("    setp.ge.u64 %p_oob, %rd_addr, %rd_alloc_end;\n");
                output.push_str("    @%p_oob trap;\n");
            } else {
                output.push_str(line);
                output.push('\n');
            }
        }

        output
    }

    /// Insert NaN-detection instrumentation into PTX source.
    ///
    /// After every floating-point arithmetic instruction (`add.f32`,
    /// `mul.f64`, etc.) a `testp.nan` check is inserted.
    pub fn instrument_nan_checks(&self, ptx: &str) -> String {
        if !self.enable_nan_check {
            return ptx.to_owned();
        }

        let fp_ops = [
            "add.f32", "add.f64", "sub.f32", "sub.f64", "mul.f32", "mul.f64", "div.f32", "div.f64",
            "fma.f32", "fma.f64",
        ];

        let mut output = String::with_capacity(ptx.len() + ptx.len() / 4);

        for line in ptx.lines() {
            output.push_str(line);
            output.push('\n');

            let trimmed = line.trim();
            if fp_ops.iter().any(|op| trimmed.starts_with(op)) {
                // Extract the destination register (first token after the op).
                if let Some(dest) = trimmed.split_whitespace().nth(1) {
                    let dest_clean = dest.trim_end_matches(',');
                    let width = if trimmed.contains(".f64") {
                        "f64"
                    } else {
                        "f32"
                    };
                    output.push_str(&format!(
                        "    // [oxicuda-debug] NaN check for {dest_clean}\n"
                    ));
                    output.push_str(&format!("    testp.nan.{width} %p_nan, {dest_clean};\n"));
                    output.push_str("    @%p_nan trap;\n");
                }
            }
        }

        output
    }

    /// Insert printf buffer support into PTX source.
    ///
    /// Adds a `.param .u64 __oxicuda_printf_buf` to each entry and inserts
    /// stub store sequences where `// PRINTF` markers appear.
    pub fn instrument_printf(&self, ptx: &str) -> String {
        if !self.enable_printf {
            return ptx.to_owned();
        }

        let mut output = String::with_capacity(ptx.len() + ptx.len() / 4);
        let mut added_param = false;

        for line in ptx.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(".entry") && !added_param {
                output.push_str(line);
                output.push('\n');
                output.push_str("    // [oxicuda-debug] printf buffer parameter\n");
                output.push_str("    .param .u64 __oxicuda_printf_buf;\n");
                added_param = true;
                continue;
            }

            if trimmed.starts_with("// PRINTF") {
                output.push_str("    // [oxicuda-debug] printf store sequence\n");
                output.push_str("    ld.param.u64 %rd_pbuf, [__oxicuda_printf_buf];\n");
                output.push_str("    atom.global.add.u32 %r_poff, [%rd_pbuf], 1;\n");
            } else {
                output.push_str(line);
                output.push('\n');
            }
        }

        output
    }

    /// Remove all OxiCUDA debug instrumentation from PTX source.
    pub fn strip_debug(&self, ptx: &str) -> String {
        let mut output = String::with_capacity(ptx.len());
        let mut skip_next = false;

        for line in ptx.lines() {
            if skip_next {
                skip_next = false;
                continue;
            }

            let trimmed = line.trim();

            // Skip debug comment lines and the instruction immediately after.
            if trimmed.starts_with("// [oxicuda-debug]") {
                // Also skip the next instrumentation line.
                skip_next = true;
                continue;
            }

            // Skip debug parameter declarations.
            if trimmed.contains("__oxicuda_debug_buf") || trimmed.contains("__oxicuda_printf_buf") {
                continue;
            }

            output.push_str(line);
            output.push('\n');
        }

        output
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config defaults --

    #[test]
    fn config_default_values() {
        let cfg = KernelDebugConfig::default();
        assert_eq!(cfg.debug_level, DebugLevel::Info);
        assert!(cfg.enable_bounds_check);
        assert!(cfg.enable_nan_check);
        assert!(cfg.enable_inf_check);
        assert!(!cfg.enable_race_detection);
        assert_eq!(cfg.print_buffer_size, 1024 * 1024);
        assert_eq!(cfg.max_print_per_thread, 32);
    }

    // -- KernelDebugger creation --

    #[test]
    fn debugger_creation_with_config() {
        let cfg = KernelDebugConfig {
            debug_level: DebugLevel::Trace,
            enable_bounds_check: false,
            ..KernelDebugConfig::default()
        };
        let debugger = KernelDebugger::new(cfg);
        assert_eq!(debugger.config().debug_level, DebugLevel::Trace);
        assert!(!debugger.config().enable_bounds_check);
    }

    // -- Session lifecycle --

    #[test]
    fn debug_session_lifecycle() {
        let cfg = KernelDebugConfig::default();
        let mut debugger = KernelDebugger::new(cfg);
        let session = debugger.attach("test_kernel");
        assert!(session.is_ok());
        let session = session.expect("session");
        assert_eq!(session.kernel_name(), "test_kernel");
        assert!(session.events().is_empty());

        // Attaching with empty name is an error.
        let err = debugger.attach("");
        assert!(err.is_err());
    }

    // -- Breakpoints --

    #[test]
    fn breakpoint_set_and_remove() {
        let mut debugger = KernelDebugger::new(KernelDebugConfig::default());
        let bp1 = debugger.set_breakpoint(42);
        let bp2 = debugger.set_breakpoint(100);
        assert_ne!(bp1, bp2);

        assert!(debugger.remove_breakpoint(bp1));
        // Removing again should return false.
        assert!(!debugger.remove_breakpoint(bp1));
        // bp2 should still be present.
        assert!(debugger.remove_breakpoint(bp2));
    }

    // -- Memory checker: valid access --

    #[test]
    fn memory_checker_valid_access() {
        let checker = MemoryChecker::new(vec![MemoryRegion {
            base_address: 0x1000,
            size: 256,
            name: "buf_a".into(),
            is_readonly: false,
        }]);
        // Read within bounds.
        assert!(checker.check_access(0x1000, 16, false).is_none());
        // Write within bounds.
        assert!(checker.check_access(0x1080, 32, true).is_none());
    }

    // -- Memory checker: OOB detection --

    #[test]
    fn memory_checker_out_of_bounds() {
        let checker = MemoryChecker::new(vec![MemoryRegion {
            base_address: 0x1000,
            size: 256,
            name: "buf_a".into(),
            is_readonly: false,
        }]);
        // Access past the end.
        let ev = checker.check_access(0x1100, 16, false);
        assert!(ev.is_some());
        let ev = ev.expect("oob event");
        assert!(matches!(ev.event_type, DebugEventType::OutOfBounds { .. }));

        // Completely outside.
        let ev2 = checker.check_access(0x5000, 4, true);
        assert!(ev2.is_some());
    }

    // -- NaN detection in f32 --

    #[test]
    fn nan_detection_f32() {
        let data = [1.0_f32, f32::NAN, 3.0, f32::NAN];
        let locs = NanInfChecker::check_f32(&data);
        assert_eq!(locs.len(), 2);
        assert_eq!(locs[0].index, 1);
        assert!(locs[0].is_nan);
        assert_eq!(locs[1].index, 3);
    }

    // -- Inf detection in f64 --

    #[test]
    fn inf_detection_f64() {
        let data = [1.0_f64, f64::INFINITY, f64::NEG_INFINITY, 4.0];
        let locs = NanInfChecker::check_f64(&data);
        assert_eq!(locs.len(), 2);
        assert!(!locs[0].is_nan);
        assert_eq!(locs[0].index, 1);
        assert!(!locs[1].is_nan);
        assert_eq!(locs[1].index, 2);
    }

    // -- Printf buffer parsing --

    #[test]
    fn printf_buffer_parsing() {
        let buf = PrintfBuffer::new(4096);

        // Build a raw buffer with one entry containing one Int arg.
        let mut raw = Vec::new();
        // entry_count = 1
        raw.extend_from_slice(&1_u32.to_le_bytes());
        // thread_id (1,0,0)
        raw.extend_from_slice(&1_u32.to_le_bytes());
        raw.extend_from_slice(&0_u32.to_le_bytes());
        raw.extend_from_slice(&0_u32.to_le_bytes());
        // block_id (0,0,0)
        raw.extend_from_slice(&0_u32.to_le_bytes());
        raw.extend_from_slice(&0_u32.to_le_bytes());
        raw.extend_from_slice(&0_u32.to_le_bytes());
        // format string "val=%d"
        let fmt = b"val=%d";
        raw.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        raw.extend_from_slice(fmt);
        // arg_count = 1
        raw.extend_from_slice(&1_u32.to_le_bytes());
        // tag=0 (Int), value=42
        raw.push(0);
        raw.extend_from_slice(&42_i64.to_le_bytes());

        let entries = buf.parse_entries(&raw);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].thread_id, (1, 0, 0));
        assert_eq!(entries[0].format_string, "val=%d");
        assert_eq!(entries[0].args.len(), 1);
        assert_eq!(entries[0].args[0], PrintfArg::Int(42));
    }

    // -- Assertions --

    #[test]
    fn assertion_checks() {
        // bounds: in range => None
        assert!(KernelAssertions::assert_bounds(5, 10, "arr").is_none());
        // bounds: out of range => Some
        let ev = KernelAssertions::assert_bounds(10, 10, "arr");
        assert!(ev.is_some());

        // NaN
        assert!(KernelAssertions::assert_not_nan(1.0, "x").is_none());
        assert!(KernelAssertions::assert_not_nan(f64::NAN, "x").is_some());

        // finite
        assert!(KernelAssertions::assert_finite(1.0, "x").is_none());
        assert!(KernelAssertions::assert_finite(f64::INFINITY, "x").is_some());
        assert!(KernelAssertions::assert_finite(f64::NAN, "x").is_some());

        // positive
        assert!(KernelAssertions::assert_positive(1.0, "x").is_none());
        assert!(KernelAssertions::assert_positive(0.0, "x").is_some());
        assert!(KernelAssertions::assert_positive(-1.0, "x").is_some());
        assert!(KernelAssertions::assert_positive(f64::NAN, "x").is_some());
    }

    // -- Event filtering --

    #[test]
    fn debug_event_filtering() {
        let cfg = KernelDebugConfig::default();
        let mut debugger = KernelDebugger::new(cfg);
        let mut session = debugger.attach("filter_test").expect("session");

        session.add_event(DebugEvent {
            event_type: DebugEventType::NanDetected {
                register: "f0".into(),
                value: f64::NAN,
            },
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            timestamp_ns: 100,
            message: "nan".into(),
        });
        session.add_event(DebugEvent {
            event_type: DebugEventType::OutOfBounds {
                address: 0xDEAD,
                size: 4,
            },
            thread_id: (1, 0, 0),
            block_id: (0, 0, 0),
            timestamp_ns: 200,
            message: "oob".into(),
        });
        session.add_event(DebugEvent {
            event_type: DebugEventType::NanDetected {
                register: "f1".into(),
                value: f64::NAN,
            },
            thread_id: (2, 0, 0),
            block_id: (0, 0, 0),
            timestamp_ns: 300,
            message: "nan2".into(),
        });

        let nans = session.filter_events(&DebugEventType::NanDetected {
            register: String::new(),
            value: 0.0,
        });
        assert_eq!(nans.len(), 2);

        let oobs = session.filter_events(&DebugEventType::OutOfBounds {
            address: 0,
            size: 0,
        });
        assert_eq!(oobs.len(), 1);
    }

    // -- Summary statistics --

    #[test]
    fn summary_statistics() {
        let cfg = KernelDebugConfig::default();
        let mut debugger = KernelDebugger::new(cfg);
        let mut session = debugger.attach("summary_test").expect("session");

        session.add_event(DebugEvent {
            event_type: DebugEventType::NanDetected {
                register: "f0".into(),
                value: f64::NAN,
            },
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            timestamp_ns: 0,
            message: String::new(),
        });
        session.add_event(DebugEvent {
            event_type: DebugEventType::InfDetected {
                register: "f1".into(),
            },
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            timestamp_ns: 0,
            message: String::new(),
        });
        session.add_event(DebugEvent {
            event_type: DebugEventType::OutOfBounds {
                address: 0x100,
                size: 4,
            },
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            timestamp_ns: 0,
            message: String::new(),
        });
        session.add_event(DebugEvent {
            event_type: DebugEventType::RaceCondition { address: 0x200 },
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            timestamp_ns: 0,
            message: String::new(),
        });

        let s = session.summary();
        assert_eq!(s.total_events, 4);
        assert_eq!(s.errors, 2); // OOB + race
        assert_eq!(s.warnings, 2); // NaN + Inf
        assert_eq!(s.nan_count, 1);
        assert_eq!(s.inf_count, 1);
        assert_eq!(s.oob_count, 1);
        assert_eq!(s.race_count, 1);
    }

    // -- Format report --

    #[test]
    fn format_report_output() {
        let cfg = KernelDebugConfig::default();
        let mut debugger = KernelDebugger::new(cfg);
        let mut session = debugger.attach("report_test").expect("session");

        session.add_event(DebugEvent {
            event_type: DebugEventType::NanDetected {
                register: "f0".into(),
                value: f64::NAN,
            },
            thread_id: (0, 0, 0),
            block_id: (0, 0, 0),
            timestamp_ns: 42,
            message: "NaN found".into(),
        });

        let report = session.format_report();
        assert!(report.contains("report_test"));
        assert!(report.contains("Total events: 1"));
        assert!(report.contains("NaN detected:  1"));
        assert!(report.contains("NaN found"));
        assert!(report.contains("=== End Report ==="));
    }

    // -- PTX instrumentation: bounds checks --

    #[test]
    fn ptx_instrumentation_bounds_checks() {
        let cfg = KernelDebugConfig::default();
        let inst = DebugPtxInstrumenter::new(&cfg);

        let ptx = ".entry my_kernel {\n    ld.global.f32 %f0, [%rd0];\n    ret;\n}\n";
        let result = inst.instrument_bounds_checks(ptx);

        assert!(result.contains("__oxicuda_debug_buf"));
        assert!(result.contains("setp.ge.u64"));
        assert!(result.contains("@%p_oob trap"));
    }

    // -- PTX strip debug --

    #[test]
    fn ptx_strip_debug_roundtrip() {
        let cfg = KernelDebugConfig::default();
        let inst = DebugPtxInstrumenter::new(&cfg);

        let original = ".entry kern {\n    add.f32 %f0, %f1, %f2;\n    ret;\n}\n";
        let instrumented = inst.instrument_nan_checks(original);
        assert!(instrumented.contains("[oxicuda-debug]"));

        let stripped = inst.strip_debug(&instrumented);
        // After stripping, no debug markers should remain.
        assert!(!stripped.contains("[oxicuda-debug]"));
        // Original instructions should still be present.
        assert!(stripped.contains("add.f32"));
        assert!(stripped.contains("ret;"));
    }
}
