//! Kernel argument serialization, Debug/Display formatting, and launch logging.
//!
//! This module provides infrastructure for serializing kernel arguments
//! into a human-readable format, logging kernel launches, and producing
//! aggregate launch summaries. It is useful for debugging, profiling,
//! and tracing GPU kernel invocations.
//!
//! # Overview
//!
//! - [`ArgType`] — describes the data type of a serialized kernel argument.
//! - [`SerializedArg`] — a single kernel argument with its type, name, and
//!   string representation of its value.
//! - [`LaunchLog`] — a complete record of a single kernel launch including
//!   the kernel name, grid/block dimensions, shared memory, and arguments.
//! - [`LaunchLogger`] — collects [`LaunchLog`] entries for analysis.
//! - [`LaunchSummary`] — aggregate statistics (per-kernel launch counts, etc.).
//! - [`SerializableKernelArgs`] — extends [`KernelArgs`]
//!   with the ability to serialize arguments into [`SerializedArg`] form.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_launch::arg_serialize::*;
//! use oxicuda_launch::{LaunchParams, Dim3};
//!
//! let arg = SerializedArg::new(Some("n".to_string()), ArgType::U32, "1024".to_string(), 4);
//! assert_eq!(arg.name(), Some("n"));
//! assert_eq!(arg.value_repr(), "1024");
//!
//! let params = LaunchParams::new(4u32, 256u32);
//! let formatted = format_launch_params(&params);
//! assert!(formatted.contains("grid"));
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use crate::grid::Dim3;
use crate::kernel::KernelArgs;
use crate::params::LaunchParams;

// ---------------------------------------------------------------------------
// ArgType
// ---------------------------------------------------------------------------

/// Describes the data type of a serialized kernel argument.
///
/// Covers the common scalar types used in GPU kernels, a generic
/// pointer type, and a [`Custom`](ArgType::Custom) variant for
/// user-defined or composite types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArgType {
    /// Unsigned 8-bit integer (`u8`).
    U8,
    /// Unsigned 16-bit integer (`u16`).
    U16,
    /// Unsigned 32-bit integer (`u32`).
    U32,
    /// Unsigned 64-bit integer (`u64`).
    U64,
    /// Signed 8-bit integer (`i8`).
    I8,
    /// Signed 16-bit integer (`i16`).
    I16,
    /// Signed 32-bit integer (`i32`).
    I32,
    /// Signed 64-bit integer (`i64`).
    I64,
    /// 32-bit floating point (`f32`).
    F32,
    /// 64-bit floating point (`f64`).
    F64,
    /// A raw pointer (device or host).
    Ptr,
    /// A user-defined or composite type with a descriptive name.
    Custom(String),
}

impl fmt::Display for ArgType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U8 => write!(f, "u8"),
            Self::U16 => write!(f, "u16"),
            Self::U32 => write!(f, "u32"),
            Self::U64 => write!(f, "u64"),
            Self::I8 => write!(f, "i8"),
            Self::I16 => write!(f, "i16"),
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
            Self::Ptr => write!(f, "ptr"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

// ---------------------------------------------------------------------------
// SerializedArg
// ---------------------------------------------------------------------------

/// A serialized representation of a single kernel argument.
///
/// Captures the argument's optional name, data type, a human-readable
/// string representation of its value, and its size in bytes.
#[derive(Debug, Clone)]
pub struct SerializedArg {
    /// Optional human-readable name for the argument (e.g., parameter name).
    name: Option<String>,
    /// The data type of the argument.
    arg_type: ArgType,
    /// A string representation of the argument's value.
    value_repr: String,
    /// Size of the argument in bytes.
    size_bytes: usize,
}

impl SerializedArg {
    /// Creates a new `SerializedArg`.
    #[inline]
    pub fn new(
        name: Option<String>,
        arg_type: ArgType,
        value_repr: String,
        size_bytes: usize,
    ) -> Self {
        Self {
            name,
            arg_type,
            value_repr,
            size_bytes,
        }
    }

    /// Returns the optional name of this argument.
    #[inline]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the data type of this argument.
    #[inline]
    pub fn arg_type(&self) -> &ArgType {
        &self.arg_type
    }

    /// Returns the string representation of the argument value.
    #[inline]
    pub fn value_repr(&self) -> &str {
        &self.value_repr
    }

    /// Returns the size of this argument in bytes.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Returns the total size of all arguments in a slice.
    pub fn total_size(args: &[Self]) -> usize {
        args.iter().map(|a| a.size_bytes).sum()
    }
}

impl fmt::Display for SerializedArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.name {
            Some(name) => write!(f, "{name}: {} = {}", self.arg_type, self.value_repr),
            None => write!(f, "{}: {}", self.arg_type, self.value_repr),
        }
    }
}

// ---------------------------------------------------------------------------
// LaunchLog
// ---------------------------------------------------------------------------

/// A complete record of a single kernel launch.
///
/// Captures the kernel name, launch configuration (grid, block, shared
/// memory), serialized arguments, and a timestamp. Named `LaunchLog`
/// to avoid conflicting with [`LaunchRecord`](crate::LaunchRecord) in
/// the `graph_launch` module.
pub struct LaunchLog {
    /// Name of the kernel function.
    kernel_name: String,
    /// Grid dimensions (number of thread blocks).
    grid: Dim3,
    /// Block dimensions (threads per block).
    block: Dim3,
    /// Dynamic shared memory in bytes.
    shared_mem: u32,
    /// Serialized kernel arguments.
    args: Vec<SerializedArg>,
    /// Timestamp when this launch was recorded.
    timestamp: Instant,
}

impl LaunchLog {
    /// Creates a new `LaunchLog` entry.
    ///
    /// The timestamp is set to the current instant.
    pub fn new(
        kernel_name: String,
        grid: Dim3,
        block: Dim3,
        shared_mem: u32,
        args: Vec<SerializedArg>,
    ) -> Self {
        Self {
            kernel_name,
            grid,
            block,
            shared_mem,
            args,
            timestamp: Instant::now(),
        }
    }

    /// Creates a new `LaunchLog` from a kernel name, [`LaunchParams`], and args.
    pub fn from_params(
        kernel_name: String,
        params: &LaunchParams,
        args: Vec<SerializedArg>,
    ) -> Self {
        Self::new(
            kernel_name,
            params.grid,
            params.block,
            params.shared_mem_bytes,
            args,
        )
    }

    /// Returns the kernel function name.
    #[inline]
    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    /// Returns the grid dimensions.
    #[inline]
    pub fn grid(&self) -> Dim3 {
        self.grid
    }

    /// Returns the block dimensions.
    #[inline]
    pub fn block(&self) -> Dim3 {
        self.block
    }

    /// Returns the shared memory size in bytes.
    #[inline]
    pub fn shared_mem(&self) -> u32 {
        self.shared_mem
    }

    /// Returns the serialized arguments.
    #[inline]
    pub fn args(&self) -> &[SerializedArg] {
        &self.args
    }

    /// Returns the timestamp when this launch was recorded.
    #[inline]
    pub fn timestamp(&self) -> Instant {
        self.timestamp
    }

    /// Returns the total number of threads in this launch.
    #[inline]
    pub fn total_threads(&self) -> u64 {
        self.grid.total() as u64 * self.block.total() as u64
    }
}

impl fmt::Display for LaunchLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let grid_str = format!("({},{},{})", self.grid.x, self.grid.y, self.grid.z);
        let block_str = format!("({},{},{})", self.block.x, self.block.y, self.block.z);
        let args_str = format_args_inner(&self.args);
        write!(
            f,
            "{}<<<{}, {}, {}>>>( {} )",
            self.kernel_name, grid_str, block_str, self.shared_mem, args_str
        )
    }
}

impl fmt::Debug for LaunchLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LaunchLog")
            .field("kernel_name", &self.kernel_name)
            .field("grid", &self.grid)
            .field("block", &self.block)
            .field("shared_mem", &self.shared_mem)
            .field("args_count", &self.args.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// LaunchLogger
// ---------------------------------------------------------------------------

/// Collects [`LaunchLog`] entries for inspection and analysis.
///
/// Provides append-only storage of launch records with methods to
/// retrieve entries, clear the log, and produce aggregate summaries.
///
/// # Example
///
/// ```rust
/// use oxicuda_launch::arg_serialize::*;
/// use oxicuda_launch::Dim3;
///
/// let mut logger = LaunchLogger::new();
/// logger.log(LaunchLog::new("kern_a".into(), Dim3::x(4), Dim3::x(256), 0, vec![]));
/// logger.log(LaunchLog::new("kern_a".into(), Dim3::x(8), Dim3::x(256), 0, vec![]));
/// logger.log(LaunchLog::new("kern_b".into(), Dim3::x(1), Dim3::x(128), 0, vec![]));
/// let summary = logger.summary();
/// assert_eq!(summary.total_launches(), 3);
/// ```
#[derive(Debug)]
pub struct LaunchLogger {
    /// Stored launch log entries.
    entries: Vec<LaunchLog>,
}

impl LaunchLogger {
    /// Creates a new empty `LaunchLogger`.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Appends a [`LaunchLog`] entry to the logger.
    #[inline]
    pub fn log(&mut self, record: LaunchLog) {
        self.entries.push(record);
    }

    /// Returns a slice of all recorded launch log entries.
    #[inline]
    pub fn entries(&self) -> &[LaunchLog] {
        &self.entries
    }

    /// Clears all recorded entries.
    #[inline]
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns the number of recorded entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no entries have been recorded.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Produces a [`LaunchSummary`] from all recorded entries.
    ///
    /// The summary aggregates per-kernel launch counts and provides
    /// the total number of launches.
    pub fn summary(&self) -> LaunchSummary {
        let mut per_kernel: HashMap<String, KernelLaunchStats> = HashMap::new();
        for entry in &self.entries {
            let stats = per_kernel
                .entry(entry.kernel_name.clone())
                .or_insert_with(|| KernelLaunchStats {
                    kernel_name: entry.kernel_name.clone(),
                    launch_count: 0,
                    total_threads: 0,
                    total_shared_mem: 0,
                });
            stats.launch_count += 1;
            stats.total_threads += entry.total_threads();
            stats.total_shared_mem += u64::from(entry.shared_mem);
        }
        LaunchSummary {
            total_launches: self.entries.len(),
            per_kernel,
        }
    }
}

impl Default for LaunchLogger {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// KernelLaunchStats
// ---------------------------------------------------------------------------

/// Per-kernel aggregate statistics within a [`LaunchSummary`].
#[derive(Debug, Clone)]
pub struct KernelLaunchStats {
    /// The kernel function name.
    kernel_name: String,
    /// Number of times this kernel was launched.
    launch_count: usize,
    /// Total threads across all launches of this kernel.
    total_threads: u64,
    /// Total shared memory bytes requested across all launches.
    total_shared_mem: u64,
}

impl KernelLaunchStats {
    /// Returns the kernel function name.
    #[inline]
    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    /// Returns the number of launches recorded for this kernel.
    #[inline]
    pub fn launch_count(&self) -> usize {
        self.launch_count
    }

    /// Returns the total number of threads across all launches.
    #[inline]
    pub fn total_threads(&self) -> u64 {
        self.total_threads
    }

    /// Returns the total shared memory bytes across all launches.
    #[inline]
    pub fn total_shared_mem(&self) -> u64 {
        self.total_shared_mem
    }
}

impl fmt::Display for KernelLaunchStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} launches, {} total threads, {} bytes shared mem",
            self.kernel_name, self.launch_count, self.total_threads, self.total_shared_mem
        )
    }
}

// ---------------------------------------------------------------------------
// LaunchSummary
// ---------------------------------------------------------------------------

/// Aggregate statistics over all recorded kernel launches.
///
/// Produced by [`LaunchLogger::summary`], this provides per-kernel
/// launch counts and total launch counts for analysis and debugging.
#[derive(Debug)]
pub struct LaunchSummary {
    /// Total number of kernel launches recorded.
    total_launches: usize,
    /// Per-kernel statistics, keyed by kernel function name.
    per_kernel: HashMap<String, KernelLaunchStats>,
}

impl LaunchSummary {
    /// Returns the total number of kernel launches across all kernels.
    #[inline]
    pub fn total_launches(&self) -> usize {
        self.total_launches
    }

    /// Returns per-kernel statistics as a map keyed by kernel name.
    #[inline]
    pub fn per_kernel(&self) -> &HashMap<String, KernelLaunchStats> {
        &self.per_kernel
    }

    /// Returns the number of distinct kernels that were launched.
    #[inline]
    pub fn unique_kernels(&self) -> usize {
        self.per_kernel.len()
    }

    /// Returns the statistics for a specific kernel by name, if present.
    #[inline]
    pub fn kernel_stats(&self, name: &str) -> Option<&KernelLaunchStats> {
        self.per_kernel.get(name)
    }
}

impl fmt::Display for LaunchSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LaunchSummary: {} total launches", self.total_launches)?;
        let mut names: Vec<&String> = self.per_kernel.keys().collect();
        names.sort();
        for name in names {
            if let Some(stats) = self.per_kernel.get(name) {
                writeln!(f, "  {stats}")?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SerializableKernelArgs trait
// ---------------------------------------------------------------------------

/// Extension trait for [`KernelArgs`] that can serialize arguments
/// into [`SerializedArg`] form for logging and debugging.
///
/// # Safety
///
/// Implementors must uphold the same invariants as [`KernelArgs`].
/// The serialized arguments must correspond one-to-one with the
/// pointers returned by `as_param_ptrs`.
pub unsafe trait SerializableKernelArgs: KernelArgs {
    /// Serializes the kernel arguments into a vector of [`SerializedArg`].
    fn serialize_args(&self) -> Vec<SerializedArg>;
}

// SerializableKernelArgs for () (no arguments)
unsafe impl SerializableKernelArgs for () {
    fn serialize_args(&self) -> Vec<SerializedArg> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Helper trait for individual argument serialization
// ---------------------------------------------------------------------------

/// Helper trait to serialize a single value into a [`SerializedArg`].
///
/// Implemented for all common scalar types used in GPU kernels.
pub trait SerializeArg: Copy {
    /// Returns the [`ArgType`] for this value.
    fn arg_type() -> ArgType;

    /// Returns a string representation of this value.
    fn value_repr(&self) -> String;

    /// Returns the size of this type in bytes.
    fn size_bytes() -> usize;

    /// Produces a [`SerializedArg`] with an optional name.
    fn to_serialized(&self, name: Option<String>) -> SerializedArg {
        SerializedArg::new(
            name,
            Self::arg_type(),
            self.value_repr(),
            Self::size_bytes(),
        )
    }
}

macro_rules! impl_serialize_arg_int {
    ($ty:ty, $variant:ident) => {
        impl SerializeArg for $ty {
            #[inline]
            fn arg_type() -> ArgType {
                ArgType::$variant
            }
            #[inline]
            fn value_repr(&self) -> String {
                self.to_string()
            }
            #[inline]
            fn size_bytes() -> usize {
                std::mem::size_of::<$ty>()
            }
        }
    };
}

impl_serialize_arg_int!(u8, U8);
impl_serialize_arg_int!(u16, U16);
impl_serialize_arg_int!(u32, U32);
impl_serialize_arg_int!(u64, U64);
impl_serialize_arg_int!(i8, I8);
impl_serialize_arg_int!(i16, I16);
impl_serialize_arg_int!(i32, I32);
impl_serialize_arg_int!(i64, I64);

impl SerializeArg for f32 {
    #[inline]
    fn arg_type() -> ArgType {
        ArgType::F32
    }
    #[inline]
    fn value_repr(&self) -> String {
        if self.fract() == 0.0 && self.is_finite() {
            format!("{self:.1}")
        } else {
            format!("{self}")
        }
    }
    #[inline]
    fn size_bytes() -> usize {
        4
    }
}

impl SerializeArg for f64 {
    #[inline]
    fn arg_type() -> ArgType {
        ArgType::F64
    }
    #[inline]
    fn value_repr(&self) -> String {
        if self.fract() == 0.0 && self.is_finite() {
            format!("{self:.1}")
        } else {
            format!("{self}")
        }
    }
    #[inline]
    fn size_bytes() -> usize {
        8
    }
}

impl SerializeArg for usize {
    #[inline]
    fn arg_type() -> ArgType {
        ArgType::Ptr
    }
    #[inline]
    fn value_repr(&self) -> String {
        format!("0x{self:x}")
    }
    #[inline]
    fn size_bytes() -> usize {
        std::mem::size_of::<usize>()
    }
}

impl SerializeArg for isize {
    #[inline]
    fn arg_type() -> ArgType {
        ArgType::Ptr
    }
    #[inline]
    fn value_repr(&self) -> String {
        format!("0x{self:x}")
    }
    #[inline]
    fn size_bytes() -> usize {
        std::mem::size_of::<isize>()
    }
}

// ---------------------------------------------------------------------------
// Macro-generated SerializableKernelArgs for tuples
// ---------------------------------------------------------------------------

macro_rules! impl_serializable_kernel_args_tuple {
    ($($idx:tt: $T:ident),+) => {
        /// # Safety
        ///
        /// The serialized arguments correspond one-to-one with the pointers
        /// from `as_param_ptrs`.
        unsafe impl<$($T: Copy + SerializeArg),+> SerializableKernelArgs for ($($T,)+) {
            fn serialize_args(&self) -> Vec<SerializedArg> {
                vec![
                    $(self.$idx.to_serialized(Some(format!("arg{}", $idx))),)+
                ]
            }
        }
    };
}

impl_serializable_kernel_args_tuple!(0: A);
impl_serializable_kernel_args_tuple!(0: A, 1: B);
impl_serializable_kernel_args_tuple!(0: A, 1: B, 2: C);
impl_serializable_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D);
impl_serializable_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_serializable_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_serializable_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_serializable_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_serializable_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_serializable_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_serializable_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_serializable_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Pretty-prints a [`LaunchParams`] configuration.
///
/// Produces a string like `"grid=(4,1,1) block=(256,1,1) smem=0"`.
pub fn format_launch_params(params: &LaunchParams) -> String {
    format!(
        "grid=({},{},{}) block=({},{},{}) smem={}",
        params.grid.x,
        params.grid.y,
        params.grid.z,
        params.block.x,
        params.block.y,
        params.block.z,
        params.shared_mem_bytes,
    )
}

/// Pretty-prints a slice of [`SerializedArg`] values.
///
/// Produces a comma-separated string of argument representations.
/// Each argument is formatted using its [`Display`](fmt::Display) impl.
pub fn format_args(args: &[SerializedArg]) -> String {
    format_args_inner(args)
}

/// Internal formatting helper shared by `format_args` and `LaunchLog::Display`.
fn format_args_inner(args: &[SerializedArg]) -> String {
    if args.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = args.iter().map(|a| a.to_string()).collect();
    parts.join(", ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::LaunchParams;

    #[test]
    fn arg_type_display() {
        assert_eq!(format!("{}", ArgType::U32), "u32");
        assert_eq!(format!("{}", ArgType::F64), "f64");
        assert_eq!(format!("{}", ArgType::Ptr), "ptr");
        assert_eq!(format!("{}", ArgType::Custom("MyType".into())), "MyType");
    }

    #[test]
    fn arg_type_equality() {
        assert_eq!(ArgType::U32, ArgType::U32);
        assert_ne!(ArgType::U32, ArgType::U64);
        assert_eq!(ArgType::Custom("Foo".into()), ArgType::Custom("Foo".into()));
    }

    #[test]
    fn serialized_arg_new_and_accessors() {
        let arg = SerializedArg::new(Some("count".into()), ArgType::U32, "42".into(), 4);
        assert_eq!(arg.name(), Some("count"));
        assert_eq!(*arg.arg_type(), ArgType::U32);
        assert_eq!(arg.value_repr(), "42");
        assert_eq!(arg.size_bytes(), 4);
    }

    #[test]
    fn serialized_arg_no_name() {
        let arg = SerializedArg::new(None, ArgType::F32, "3.14".into(), 4);
        assert_eq!(arg.name(), None);
        assert_eq!(format!("{arg}"), "f32: 3.14");
    }

    #[test]
    fn serialized_arg_with_name_display() {
        let arg = SerializedArg::new(Some("x".into()), ArgType::I64, "-100".into(), 8);
        assert_eq!(format!("{arg}"), "x: i64 = -100");
    }

    #[test]
    fn serialized_arg_total_size() {
        let args = vec![
            SerializedArg::new(None, ArgType::U32, "1".into(), 4),
            SerializedArg::new(None, ArgType::U64, "2".into(), 8),
            SerializedArg::new(None, ArgType::F32, "3.0".into(), 4),
        ];
        assert_eq!(SerializedArg::total_size(&args), 16);
    }

    #[test]
    fn launch_log_creation_and_accessors() {
        let log = LaunchLog::new(
            "vector_add".into(),
            Dim3::x(4),
            Dim3::x(256),
            1024,
            vec![SerializedArg::new(None, ArgType::U32, "42".into(), 4)],
        );
        assert_eq!(log.kernel_name(), "vector_add");
        assert_eq!(log.grid(), Dim3::x(4));
        assert_eq!(log.block(), Dim3::x(256));
        assert_eq!(log.shared_mem(), 1024);
        assert_eq!(log.args().len(), 1);
        assert_eq!(log.total_threads(), 1024);
    }

    #[test]
    fn launch_log_from_params() {
        let params = LaunchParams::new(Dim3::xy(2, 2), Dim3::x(128)).with_shared_mem(512);
        let log = LaunchLog::from_params("matmul".into(), &params, vec![]);
        assert_eq!(log.kernel_name(), "matmul");
        assert_eq!(log.grid(), Dim3::xy(2, 2));
        assert_eq!(log.shared_mem(), 512);
    }

    #[test]
    fn launch_log_display() {
        let log = LaunchLog::new(
            "my_kernel".into(),
            Dim3::x(4),
            Dim3::x(256),
            0,
            vec![
                SerializedArg::new(Some("a".into()), ArgType::U64, "0x1000".into(), 8),
                SerializedArg::new(Some("n".into()), ArgType::U32, "1024".into(), 4),
            ],
        );
        let s = format!("{log}");
        assert!(s.contains("my_kernel<<<"));
        assert!(s.contains("(4,1,1)"));
        assert!(s.contains("(256,1,1)"));
        assert!(s.contains("a: u64 = 0x1000"));
        assert!(s.contains("n: u32 = 1024"));
    }

    #[test]
    fn launch_log_debug() {
        let log = LaunchLog::new("kern".into(), Dim3::x(1), Dim3::x(1), 0, vec![]);
        let dbg = format!("{log:?}");
        assert!(dbg.contains("LaunchLog"));
        assert!(dbg.contains("kern"));
    }

    #[test]
    fn launch_logger_basic_workflow() {
        let mut logger = LaunchLogger::new();
        assert!(logger.is_empty());
        assert_eq!(logger.len(), 0);

        logger.log(LaunchLog::new(
            "kern_a".into(),
            Dim3::x(4),
            Dim3::x(256),
            0,
            vec![],
        ));
        logger.log(LaunchLog::new(
            "kern_b".into(),
            Dim3::x(8),
            Dim3::x(128),
            512,
            vec![],
        ));
        assert_eq!(logger.len(), 2);
        assert!(!logger.is_empty());
        assert_eq!(logger.entries()[0].kernel_name(), "kern_a");
        assert_eq!(logger.entries()[1].kernel_name(), "kern_b");

        logger.clear();
        assert!(logger.is_empty());
    }

    #[test]
    fn launch_logger_default() {
        let logger = LaunchLogger::default();
        assert!(logger.is_empty());
    }

    #[test]
    fn launch_summary_aggregation() {
        let mut logger = LaunchLogger::new();
        logger.log(LaunchLog::new(
            "kern_a".into(),
            Dim3::x(4),
            Dim3::x(256),
            0,
            vec![],
        ));
        logger.log(LaunchLog::new(
            "kern_a".into(),
            Dim3::x(8),
            Dim3::x(256),
            1024,
            vec![],
        ));
        logger.log(LaunchLog::new(
            "kern_b".into(),
            Dim3::x(1),
            Dim3::x(128),
            0,
            vec![],
        ));

        let summary = logger.summary();
        assert_eq!(summary.total_launches(), 3);
        assert_eq!(summary.unique_kernels(), 2);

        let a_stats = summary.kernel_stats("kern_a");
        assert!(a_stats.is_some());
        let a_stats = a_stats.expect("kern_a stats should exist in test");
        assert_eq!(a_stats.launch_count(), 2);
        assert_eq!(a_stats.total_threads(), 4 * 256 + 8 * 256);
        assert_eq!(a_stats.total_shared_mem(), 1024);

        let b_stats = summary.kernel_stats("kern_b");
        assert!(b_stats.is_some());
        let b_stats = b_stats.expect("kern_b stats should exist in test");
        assert_eq!(b_stats.launch_count(), 1);
    }

    #[test]
    fn launch_summary_display() {
        let mut logger = LaunchLogger::new();
        logger.log(LaunchLog::new(
            "kern".into(),
            Dim3::x(1),
            Dim3::x(1),
            0,
            vec![],
        ));
        let summary = logger.summary();
        let s = format!("{summary}");
        assert!(s.contains("LaunchSummary"));
        assert!(s.contains("1 total launches"));
        assert!(s.contains("kern"));
    }

    #[test]
    fn serialize_arg_trait_scalars() {
        let v: u32 = 42;
        let sa = v.to_serialized(Some("n".into()));
        assert_eq!(*sa.arg_type(), ArgType::U32);
        assert_eq!(sa.value_repr(), "42");
        assert_eq!(sa.size_bytes(), 4);

        let v: f64 = 3.15;
        let sa = v.to_serialized(None);
        assert_eq!(*sa.arg_type(), ArgType::F64);
        assert_eq!(sa.value_repr(), "3.15");
        assert_eq!(sa.size_bytes(), 8);

        let v: f32 = 1.0;
        let sa = v.to_serialized(None);
        assert_eq!(sa.value_repr(), "1.0");
    }

    #[test]
    fn serializable_kernel_args_unit() {
        let args = ();
        let serialized = args.serialize_args();
        assert!(serialized.is_empty());
    }

    #[test]
    fn serializable_kernel_args_tuple() {
        let args = (42u32, 3.15f64);
        let serialized = args.serialize_args();
        assert_eq!(serialized.len(), 2);
        assert_eq!(serialized[0].name(), Some("arg0"));
        assert_eq!(*serialized[0].arg_type(), ArgType::U32);
        assert_eq!(serialized[0].value_repr(), "42");
        assert_eq!(serialized[1].name(), Some("arg1"));
        assert_eq!(*serialized[1].arg_type(), ArgType::F64);
        assert_eq!(serialized[1].value_repr(), "3.15");
    }

    #[test]
    fn format_launch_params_output() {
        let params = LaunchParams::new(Dim3::xy(4, 2), Dim3::x(256)).with_shared_mem(4096);
        let s = format_launch_params(&params);
        assert!(s.contains("grid=(4,2,1)"));
        assert!(s.contains("block=(256,1,1)"));
        assert!(s.contains("smem=4096"));
    }

    #[test]
    fn format_args_output() {
        let args = vec![
            SerializedArg::new(Some("a".into()), ArgType::U64, "0x1000".into(), 8),
            SerializedArg::new(Some("n".into()), ArgType::U32, "1024".into(), 4),
        ];
        let s = format_args(&args);
        assert!(s.contains("a: u64 = 0x1000"));
        assert!(s.contains("n: u32 = 1024"));
    }

    #[test]
    fn format_args_empty() {
        let s = format_args(&[]);
        assert!(s.is_empty());
    }

    #[test]
    fn kernel_launch_stats_display() {
        let stats = KernelLaunchStats {
            kernel_name: "matmul".into(),
            launch_count: 5,
            total_threads: 1_000_000,
            total_shared_mem: 4096,
        };
        let s = format!("{stats}");
        assert!(s.contains("matmul"));
        assert!(s.contains("5 launches"));
        assert!(s.contains("1000000 total threads"));
    }
}
