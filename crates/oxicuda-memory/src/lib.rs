//! # OxiCUDA Memory
//!
//! **Type-safe GPU memory management with Rust ownership semantics.**
//!
//! `oxicuda-memory` provides safe, RAII-based wrappers around CUDA memory
//! allocation and transfer operations.  Every buffer type owns its GPU (or
//! pinned-host) allocation and automatically frees it on [`Drop`], preventing
//! leaks without requiring manual cleanup.
//!
//! ## Buffer types
//!
//! | Type                                  | Location       | Description                              |
//! |---------------------------------------|----------------|------------------------------------------|
//! | [`DeviceBuffer<T>`]                   | Device (VRAM)  | Primary GPU-side buffer                  |
//! | [`DeviceSlice<T>`]                    | Device (VRAM)  | Borrowed sub-range of a device buffer    |
//! | [`PinnedBuffer<T>`]                   | Host (pinned)  | Page-locked host memory for fast DMA     |
//! | [`UnifiedBuffer<T>`]                  | Unified/managed| Accessible from both host and device     |
//! | [`MappedBuffer<T>`] *(stub)*          | Host-mapped    | Zero-copy host memory (future)           |
//! | `MemoryPool` *(stub, `pool` feat)*    | Device pool    | Stream-ordered allocation (CUDA 11.2+)   |
//!
//! ## Freestanding copy helpers
//!
//! The [`copy`] module exposes explicit transfer functions that mirror the
//! CUDA driver `cuMemcpy*` family, with compile-time type safety and
//! runtime length validation.
//!
//! ## Safety philosophy
//!
//! All public APIs return [`oxicuda_driver::error::CudaResult`] rather than panicking.  Size
//! mismatches, zero-length allocations, and out-of-bounds slices are
//! reported as [`oxicuda_driver::error::CudaError::InvalidValue`].  [`Drop`] implementations log
//! errors via [`tracing::warn`] instead of panicking.
//!
//! ## Feature flags
//!
//! | Feature      | Description                                      |
//! |--------------|--------------------------------------------------|
//! | `pool`       | Enable stream-ordered memory pool (CUDA 11.2+)   |
//! | `gpu-tests`  | Enable integration tests that require a real GPU  |

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]

// ---------------------------------------------------------------------------
// Module declarations
// ---------------------------------------------------------------------------

pub mod aligned;
pub mod bandwidth_profiler;
pub mod buffer_view;
pub mod copy;
pub mod copy_2d3d;
pub mod device_buffer;
pub mod host_buffer;
pub mod host_registered;
pub mod managed_hints;
pub mod memory_info;
pub mod peer_copy;
#[cfg(feature = "pool")]
pub mod pool;
pub mod pool_stats;
pub mod unified;
pub mod virtual_memory;
pub mod zero_copy;

// ---------------------------------------------------------------------------
// Re-exports — primary buffer types
// ---------------------------------------------------------------------------

pub use bandwidth_profiler::{
    BandwidthBenchmarkConfig, BandwidthMeasurement, BandwidthProfiler, BandwidthSummary,
    DirectionSummary, TransferDirection, bandwidth_utilization, describe_bandwidth,
    estimate_transfer_time, format_bytes, theoretical_peak_bandwidth,
};
pub use buffer_view::{BufferView, BufferViewMut};
pub use copy_2d3d::{Memcpy2DParams, Memcpy3DParams};
pub use device_buffer::{DeviceBuffer, DeviceSlice};
pub use host_buffer::PinnedBuffer;
pub use host_registered::{
    RegisterFlags, RegisteredMemory, RegisteredMemoryType, RegisteredPointerInfo,
    query_registered_pointer_info, register, register_slice, register_vec,
};
pub use managed_hints::{ManagedMemoryHints, MigrationPolicy, PrefetchPlan};
pub use memory_info::{MemAdvice, MemoryInfo, mem_advise, mem_prefetch, memory_info};
pub use unified::UnifiedBuffer;
pub use virtual_memory::{
    AccessFlags, PhysicalAllocation, VirtualAddressRange, VirtualMemoryManager,
};
pub use zero_copy::MappedBuffer;

pub use aligned::{
    AlignedBuffer, Alignment, AlignmentInfo, check_alignment, coalesce_alignment,
    optimal_alignment_for_type, round_up_to_alignment, validate_alignment,
};

pub use pool_stats::{AllocationHistogram, FragmentationMetrics, PoolReport, PoolStatsTracker};

#[cfg(feature = "pool")]
pub use pool::{MemoryPool, PoolStats, PooledBuffer};

// ---------------------------------------------------------------------------
// Prelude — convenient glob import
// ---------------------------------------------------------------------------

/// Convenient glob import for common OxiCUDA Memory types.
///
/// ```rust
/// use oxicuda_memory::prelude::*;
/// ```
pub mod prelude {
    pub use crate::aligned::{AlignedBuffer, Alignment, AlignmentInfo};
    pub use crate::buffer_view::{BufferView, BufferViewMut};
    pub use crate::copy::{
        copy_dtod, copy_dtod_async, copy_dtoh, copy_dtoh_async_raw, copy_htod, copy_htod_async_raw,
    };
    pub use crate::copy_2d3d::{
        Memcpy2DParams, Memcpy3DParams, copy_2d_dtod, copy_2d_dtoh, copy_2d_htod, copy_3d_dtod,
    };
    pub use crate::device_buffer::{DeviceBuffer, DeviceSlice};
    pub use crate::host_buffer::PinnedBuffer;
    pub use crate::host_registered::{
        RegisterFlags, RegisteredMemory, RegisteredMemoryType, RegisteredPointerInfo,
        query_registered_pointer_info, register, register_slice, register_vec,
    };
    pub use crate::managed_hints::{ManagedMemoryHints, MigrationPolicy, PrefetchPlan};
    pub use crate::memory_info::{MemAdvice, MemoryInfo, mem_advise, mem_prefetch, memory_info};
    pub use crate::unified::UnifiedBuffer;
    pub use crate::virtual_memory::{AccessFlags, VirtualAddressRange, VirtualMemoryManager};
}

// ---------------------------------------------------------------------------
// Compile-time feature availability
// ---------------------------------------------------------------------------

/// Compile-time feature availability.
pub mod features {
    /// Whether the stream-ordered memory pool API is enabled.
    pub const HAS_POOL: bool = cfg!(feature = "pool");

    /// Whether GPU integration tests are enabled.
    pub const HAS_GPU_TESTS: bool = cfg!(feature = "gpu-tests");
}
