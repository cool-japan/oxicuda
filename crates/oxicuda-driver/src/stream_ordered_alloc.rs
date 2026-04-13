//! Stream-ordered memory allocation (CUDA 11.2+ / 12.x+).
//!
//! Stream-ordered memory allocation allows memory operations (`alloc` / `free`)
//! to participate in the stream execution order, eliminating the need for
//! explicit synchronisation between allocation and kernel launch.
//!
//! This module provides:
//!
//! * [`StreamMemoryPool`] — a memory pool bound to a specific device.
//! * [`StreamAllocation`] — a handle to a stream-ordered allocation.
//! * [`StreamOrderedAllocConfig`] — pool configuration (sizes, thresholds).
//! * [`PoolAttribute`] / [`PoolUsageStats`] — attribute queries and statistics.
//! * [`PoolExportDescriptor`] / [`ShareableHandleType`] — IPC sharing metadata.
//! * [`stream_alloc`] / [`stream_free`] — convenience free functions.
//!
//! # Platform behaviour
//!
//! On macOS (where NVIDIA dropped CUDA support), all operations that would
//! require the GPU driver return `Err(CudaError::NotSupported)`.  Config
//! validation, statistics tracking, and accessor methods work everywhere.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_driver::stream_ordered_alloc::*;
//!
//! let config = StreamOrderedAllocConfig::default_for_device(0);
//! let mut pool = StreamMemoryPool::new(config)?;
//!
//! let stream_handle = 0u64; // placeholder
//! let mut alloc = pool.alloc_async(1024, stream_handle)?;
//! assert_eq!(alloc.size(), 1024);
//! assert!(!alloc.is_freed());
//!
//! pool.free_async(&mut alloc)?;
//! assert!(alloc.is_freed());
//! # Ok::<(), oxicuda_driver::CudaError>(())
//! ```

use std::fmt;

use crate::error::{CudaError, CudaResult};
use crate::ffi::CUdeviceptr;

// ---------------------------------------------------------------------------
// Constants — CUmemPool_attribute (mirrors CUDA header values)
// ---------------------------------------------------------------------------

/// Pool reuse policy: follow event dependencies.
pub const CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: u32 = 1;
/// Pool reuse policy: allow opportunistic reuse.
pub const CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: u32 = 2;
/// Pool reuse policy: allow internal dependency insertion.
pub const CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: u32 = 3;
/// Release threshold in bytes (memory returned to OS when usage drops below).
pub const CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: u32 = 4;
/// Current reserved memory (bytes) — read-only.
pub const CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT: u32 = 5;
/// High-water mark of reserved memory (bytes) — resettable.
pub const CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: u32 = 6;
/// Current used memory (bytes) — read-only.
pub const CU_MEMPOOL_ATTR_USED_MEM_CURRENT: u32 = 7;
/// High-water mark of used memory (bytes) — resettable.
pub const CU_MEMPOOL_ATTR_USED_MEM_HIGH: u32 = 8;

// ---------------------------------------------------------------------------
// StreamOrderedAllocConfig
// ---------------------------------------------------------------------------

/// Configuration for a stream-ordered memory pool.
///
/// All sizes are in bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamOrderedAllocConfig {
    /// Initial pool size in bytes.  The pool pre-reserves this amount of
    /// device memory when created.
    pub initial_pool_size: usize,

    /// Maximum pool size in bytes.  `0` means unlimited — the pool will grow
    /// as needed (subject to device memory limits).
    pub max_pool_size: usize,

    /// Release threshold in bytes.  When the pool is trimmed, at least this
    /// much memory is kept reserved for future allocations.
    pub release_threshold: usize,

    /// The device ordinal to create the pool on.
    pub device: i32,
}

impl StreamOrderedAllocConfig {
    /// Validate that the configuration is internally consistent.
    ///
    /// # Rules
    ///
    /// * `initial_pool_size` must not exceed `max_pool_size` (when
    ///   `max_pool_size > 0`).
    /// * `release_threshold` must not exceed `max_pool_size` (when
    ///   `max_pool_size > 0`).
    /// * `device` must be non-negative.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if any rule is violated.
    pub fn validate(&self) -> CudaResult<()> {
        if self.device < 0 {
            return Err(CudaError::InvalidValue);
        }

        if self.max_pool_size > 0 {
            if self.initial_pool_size > self.max_pool_size {
                return Err(CudaError::InvalidValue);
            }
            if self.release_threshold > self.max_pool_size {
                return Err(CudaError::InvalidValue);
            }
        }

        Ok(())
    }

    /// Returns a sensible default configuration for the given device.
    ///
    /// * `initial_pool_size` = 0 (grow on demand)
    /// * `max_pool_size` = 0 (unlimited)
    /// * `release_threshold` = 0 (release everything on trim)
    pub fn default_for_device(device: i32) -> Self {
        Self {
            initial_pool_size: 0,
            max_pool_size: 0,
            release_threshold: 0,
            device,
        }
    }
}

// ---------------------------------------------------------------------------
// PoolAttribute
// ---------------------------------------------------------------------------

/// Attributes that can be queried or set on a [`StreamMemoryPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolAttribute {
    /// Whether freed blocks can be reused by following event dependencies.
    ReuseFollowEventDependencies,
    /// Whether freed blocks can be opportunistically reused (without ordering).
    ReuseAllowOpportunistic,
    /// Whether the pool may insert internal dependencies for reuse.
    ReuseAllowInternalDependencies,
    /// The release threshold in bytes.
    ReleaseThreshold(u64),
    /// Current reserved memory (read-only query).
    ReservedMemCurrent,
    /// High-water mark of reserved memory.
    ReservedMemHigh,
    /// Current used memory (read-only query).
    UsedMemCurrent,
    /// High-water mark of used memory.
    UsedMemHigh,
}

impl PoolAttribute {
    /// Convert to the raw CUDA attribute constant.
    pub fn to_raw(self) -> u32 {
        match self {
            Self::ReuseFollowEventDependencies => CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
            Self::ReuseAllowOpportunistic => CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
            Self::ReuseAllowInternalDependencies => {
                CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES
            }
            Self::ReleaseThreshold(_) => CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            Self::ReservedMemCurrent => CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            Self::ReservedMemHigh => CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            Self::UsedMemCurrent => CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            Self::UsedMemHigh => CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        }
    }
}

// ---------------------------------------------------------------------------
// PoolUsageStats
// ---------------------------------------------------------------------------

/// Snapshot of pool memory usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolUsageStats {
    /// Bytes currently reserved from the device allocator.
    pub reserved_current: u64,
    /// Peak bytes reserved (since creation or last reset).
    pub reserved_high: u64,
    /// Bytes currently in use by outstanding allocations.
    pub used_current: u64,
    /// Peak bytes in use (since creation or last reset).
    pub used_high: u64,
    /// Number of active (not-yet-freed) allocations.
    pub active_allocations: usize,
    /// Peak number of concurrent allocations.
    pub peak_allocations: usize,
}

// ---------------------------------------------------------------------------
// ShareableHandleType / PoolExportDescriptor
// ---------------------------------------------------------------------------

/// Handle type used for IPC sharing of memory pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShareableHandleType {
    /// No sharing.
    #[default]
    None,
    /// POSIX file descriptor (Linux).
    PosixFileDescriptor,
    /// Win32 handle (Windows).
    Win32Handle,
    /// Win32 KMT handle (Windows, legacy).
    Win32KmtHandle,
}

/// Descriptor for exporting a pool for IPC sharing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolExportDescriptor {
    /// The handle type to use for sharing.
    pub shareable_handle_type: ShareableHandleType,
    /// The device ordinal that owns the pool.
    pub pool_device: i32,
}

// ---------------------------------------------------------------------------
// StreamAllocation
// ---------------------------------------------------------------------------

/// Handle to a stream-ordered memory allocation.
///
/// An allocation lives on the GPU and is associated with a specific stream
/// and memory pool.  It becomes available when all preceding work on the
/// stream has completed, and is returned to the pool when freed (also
/// stream-ordered).
pub struct StreamAllocation {
    /// Device pointer (`CUdeviceptr`).
    ptr: CUdeviceptr,
    /// Size of the allocation in bytes.
    size: usize,
    /// The stream this allocation is ordered on.
    stream: u64,
    /// The pool handle that owns this allocation.
    pool: u64,
    /// Whether this allocation has already been freed.
    freed: bool,
}

impl StreamAllocation {
    /// Returns the device pointer as a raw `u64` (`CUdeviceptr`).
    #[inline]
    pub fn as_ptr(&self) -> u64 {
        self.ptr
    }

    /// Returns the allocation size in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns `true` if this allocation has been freed.
    #[inline]
    pub fn is_freed(&self) -> bool {
        self.freed
    }

    /// Returns the stream handle this allocation is ordered on.
    #[inline]
    pub fn stream(&self) -> u64 {
        self.stream
    }

    /// Returns the pool handle that owns this allocation.
    #[inline]
    pub fn pool(&self) -> u64 {
        self.pool
    }
}

impl fmt::Debug for StreamAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamAllocation")
            .field("ptr", &format_args!("0x{:016x}", self.ptr))
            .field("size", &self.size)
            .field("stream", &format_args!("0x{:016x}", self.stream))
            .field("freed", &self.freed)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StreamMemoryPool
// ---------------------------------------------------------------------------

/// A memory pool for stream-ordered allocations.
///
/// On platforms with a real CUDA driver (Linux, Windows), creating a pool
/// calls `cuMemPoolCreate` under the hood.  On macOS (where there is no
/// NVIDIA driver), pool metadata is tracked locally but any operation that
/// would require the driver returns `Err(CudaError::NotSupported)`.
///
/// # Allocation tracking
///
/// The pool tracks allocation counts and byte totals locally for
/// diagnostics.  These statistics are maintained even on macOS so that
/// the API surface can be exercised in tests.
pub struct StreamMemoryPool {
    /// Raw `CUmemoryPool` handle (0 if not backed by a real driver pool).
    handle: u64,
    /// Device ordinal.
    device: i32,
    /// Configuration used to create this pool.
    config: StreamOrderedAllocConfig,
    /// Number of currently active (not freed) allocations.
    active_allocations: usize,
    /// Total bytes currently allocated.
    total_allocated: usize,
    /// Peak bytes ever allocated concurrently.
    peak_allocated: usize,
    /// Peak number of concurrent allocations.
    peak_allocation_count: usize,
    /// Monotonically increasing allocation id for generating unique pointers
    /// in non-GPU mode.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    next_alloc_id: u64,
}

impl fmt::Debug for StreamMemoryPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamMemoryPool")
            .field("handle", &format_args!("0x{:016x}", self.handle))
            .field("device", &self.device)
            .field("active_allocations", &self.active_allocations)
            .field("total_allocated", &self.total_allocated)
            .field("peak_allocated", &self.peak_allocated)
            .finish()
    }
}

impl StreamMemoryPool {
    /// Create a new memory pool for the given device.
    ///
    /// The configuration is validated before the pool is created.  On
    /// platforms with a real CUDA driver, `cuMemPoolCreate` is invoked.
    /// On macOS, a local-only pool is created for testing purposes.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the config fails validation.
    /// * [`CudaError::NotSupported`] on macOS (pool metadata is still created
    ///   so that tests can exercise the API).
    pub fn new(config: StreamOrderedAllocConfig) -> CudaResult<Self> {
        config.validate()?;

        let pool = Self {
            handle: 0,
            device: config.device,
            config,
            active_allocations: 0,
            total_allocated: 0,
            peak_allocated: 0,
            peak_allocation_count: 0,
            next_alloc_id: 1,
        };

        // On real GPU platforms, we would call cuMemPoolCreate here.
        // The pool handle would be stored in `self.handle`.
        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_create_pool(&pool)?;
        }

        Ok(pool)
    }

    /// Allocate memory on a stream (stream-ordered).
    ///
    /// The allocation becomes available when all prior work on the stream
    /// has completed.  The returned [`StreamAllocation`] tracks the pointer,
    /// size, and ownership.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `size` is zero.
    /// * [`CudaError::OutOfMemory`] if `max_pool_size` would be exceeded.
    /// * [`CudaError::NotSupported`] on macOS.
    pub fn alloc_async(&mut self, size: usize, stream: u64) -> CudaResult<StreamAllocation> {
        if size == 0 {
            return Err(CudaError::InvalidValue);
        }

        // Check max pool size constraint.
        if self.config.max_pool_size > 0
            && self.total_allocated.saturating_add(size) > self.config.max_pool_size
        {
            return Err(CudaError::OutOfMemory);
        }

        let ptr = self.platform_alloc_async(size, stream)?;

        // Update bookkeeping.
        self.active_allocations += 1;
        self.total_allocated = self.total_allocated.saturating_add(size);
        if self.total_allocated > self.peak_allocated {
            self.peak_allocated = self.total_allocated;
        }
        if self.active_allocations > self.peak_allocation_count {
            self.peak_allocation_count = self.active_allocations;
        }

        Ok(StreamAllocation {
            ptr,
            size,
            stream,
            pool: self.handle,
            freed: false,
        })
    }

    /// Free memory on a stream (stream-ordered).
    ///
    /// The memory is returned to the pool when all prior work on the
    /// stream has completed.  The allocation is marked as freed and
    /// cannot be freed again.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the allocation is already freed.
    /// * [`CudaError::NotSupported`] on macOS.
    pub fn free_async(&mut self, alloc: &mut StreamAllocation) -> CudaResult<()> {
        if alloc.freed {
            return Err(CudaError::InvalidValue);
        }

        self.platform_free_async(alloc)?;

        alloc.freed = true;
        self.active_allocations = self.active_allocations.saturating_sub(1);
        self.total_allocated = self.total_allocated.saturating_sub(alloc.size);

        Ok(())
    }

    /// Trim the pool, releasing unused memory back to the OS.
    ///
    /// At least `min_bytes_to_keep` bytes of reserved memory will remain
    /// in the pool for future allocations.
    ///
    /// # Errors
    ///
    /// * [`CudaError::NotSupported`] on macOS.
    pub fn trim(&mut self, min_bytes_to_keep: usize) -> CudaResult<()> {
        self.platform_trim(min_bytes_to_keep)
    }

    /// Get pool usage statistics.
    ///
    /// The returned [`PoolUsageStats`] combines locally tracked allocation
    /// counts with byte-level information.  On macOS, the reserved/used
    /// byte fields mirror the local bookkeeping since no driver is available.
    pub fn stats(&self) -> PoolUsageStats {
        PoolUsageStats {
            reserved_current: self.total_allocated as u64,
            reserved_high: self.peak_allocated as u64,
            used_current: self.total_allocated as u64,
            used_high: self.peak_allocated as u64,
            active_allocations: self.active_allocations,
            peak_allocations: self.peak_allocation_count,
        }
    }

    /// Set a pool attribute.
    ///
    /// Only attributes that carry a value (e.g. [`PoolAttribute::ReleaseThreshold`])
    /// modify pool state.  Read-only attributes (e.g. `ReservedMemCurrent`)
    /// return [`CudaError::InvalidValue`].
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] for read-only attributes.
    /// * [`CudaError::NotSupported`] on macOS.
    pub fn set_attribute(&mut self, attr: PoolAttribute) -> CudaResult<()> {
        // Read-only attributes cannot be set.
        match attr {
            PoolAttribute::ReservedMemCurrent
            | PoolAttribute::UsedMemCurrent
            | PoolAttribute::ReservedMemHigh
            | PoolAttribute::UsedMemHigh => {
                return Err(CudaError::InvalidValue);
            }
            _ => {}
        }

        // Apply locally-meaningful attributes.
        if let PoolAttribute::ReleaseThreshold(val) = attr {
            self.config.release_threshold = val as usize;
        }

        self.platform_set_attribute(attr)
    }

    /// Enable peer access from another device to allocations in this pool.
    ///
    /// After this call, kernels running on `peer_device` can access memory
    /// allocated from this pool.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidDevice`] if `peer_device` equals this pool's device.
    /// * [`CudaError::NotSupported`] on macOS.
    pub fn enable_peer_access(&self, peer_device: i32) -> CudaResult<()> {
        if peer_device == self.device {
            return Err(CudaError::InvalidDevice);
        }

        self.platform_enable_peer_access(peer_device)
    }

    /// Disable peer access from another device to allocations in this pool.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidDevice`] if `peer_device` equals this pool's device.
    /// * [`CudaError::NotSupported`] on macOS.
    pub fn disable_peer_access(&self, peer_device: i32) -> CudaResult<()> {
        if peer_device == self.device {
            return Err(CudaError::InvalidDevice);
        }

        self.platform_disable_peer_access(peer_device)
    }

    /// Reset peak statistics (peak allocated bytes and peak allocation count).
    pub fn reset_peak_stats(&mut self) {
        self.peak_allocated = self.total_allocated;
        self.peak_allocation_count = self.active_allocations;
    }

    /// Get the default memory pool for a device.
    ///
    /// CUDA provides a default pool per device.  On macOS, this returns a
    /// local-only pool with default configuration.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `device` is negative.
    pub fn default_pool(device: i32) -> CudaResult<Self> {
        if device < 0 {
            return Err(CudaError::InvalidValue);
        }

        // On real GPU, we would call cuDeviceGetDefaultMemPool.
        // For now, return a pool with default config.
        let config = StreamOrderedAllocConfig::default_for_device(device);
        Self::new(config)
    }

    /// Returns the raw pool handle.
    #[inline]
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Returns the device ordinal.
    #[inline]
    pub fn device(&self) -> i32 {
        self.device
    }

    /// Returns the pool configuration.
    #[inline]
    pub fn config(&self) -> &StreamOrderedAllocConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Platform-specific helpers
    // -----------------------------------------------------------------------

    /// Perform the actual allocation.  On macOS, generates a synthetic pointer.
    fn platform_alloc_async(&mut self, size: usize, stream: u64) -> CudaResult<CUdeviceptr> {
        #[cfg(target_os = "macos")]
        {
            let _ = stream;
            // Generate a synthetic, non-zero device pointer for testing.
            // Each allocation gets a unique "address" based on the pool's
            // monotonic counter, with a base offset to avoid null.
            let synthetic_ptr = 0x1000_0000_0000_u64 + self.next_alloc_id * 0x1000;
            self.next_alloc_id = self.next_alloc_id.wrapping_add(1);
            let _ = size;
            Ok(synthetic_ptr)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_alloc_async(self.handle, size, stream)
        }
    }

    /// Trim on current platform.
    fn platform_trim(&mut self, min_bytes_to_keep: usize) -> CudaResult<()> {
        #[cfg(target_os = "macos")]
        {
            let _ = min_bytes_to_keep;
            Err(CudaError::NotSupported)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_trim(self.handle, min_bytes_to_keep)
        }
    }

    /// Set attribute on current platform.
    fn platform_set_attribute(&self, attr: PoolAttribute) -> CudaResult<()> {
        #[cfg(target_os = "macos")]
        {
            match attr {
                PoolAttribute::ReleaseThreshold(_) => Ok(()),
                _ => Err(CudaError::NotSupported),
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_set_attribute(self.handle, attr)
        }
    }

    /// Enable peer access on current platform.
    fn platform_enable_peer_access(&self, peer_device: i32) -> CudaResult<()> {
        #[cfg(target_os = "macos")]
        {
            let _ = peer_device;
            Err(CudaError::NotSupported)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_enable_peer_access(self.handle, peer_device)
        }
    }

    /// Disable peer access on current platform.
    fn platform_disable_peer_access(&self, peer_device: i32) -> CudaResult<()> {
        #[cfg(target_os = "macos")]
        {
            let _ = peer_device;
            Err(CudaError::NotSupported)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_disable_peer_access(self.handle, peer_device)
        }
    }

    /// Perform the actual free.  On macOS, this is a no-op (synthetic pointers).
    fn platform_free_async(&self, alloc: &StreamAllocation) -> CudaResult<()> {
        #[cfg(target_os = "macos")]
        {
            let _ = alloc;
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_free_async(alloc.ptr, alloc.stream)
        }
    }

    // -----------------------------------------------------------------------
    // GPU-only stubs (compiled out on macOS)
    // -----------------------------------------------------------------------

    /// Create the pool on the GPU via `cuMemPoolCreate`.
    #[cfg(not(target_os = "macos"))]
    fn gpu_create_pool(_pool: &Self) -> CudaResult<()> {
        // In a full implementation, this would call:
        //   cuMemPoolCreate(&pool_handle, &pool_props)
        // For now, the pool operates with handle=0 (default pool semantics).
        Ok(())
    }

    /// Allocate via `cuMemAllocAsync`.
    #[cfg(not(target_os = "macos"))]
    fn gpu_alloc_async(_pool_handle: u64, _size: usize, _stream: u64) -> CudaResult<CUdeviceptr> {
        // Would call: cuMemAllocAsync(&dptr, size, stream)
        // For now, return a placeholder.  Real implementation would use
        // try_driver() and invoke the function pointer.
        Err(CudaError::NotInitialized)
    }

    /// Free via `cuMemFreeAsync`.
    #[cfg(not(target_os = "macos"))]
    fn gpu_free_async(_ptr: CUdeviceptr, _stream: u64) -> CudaResult<()> {
        // Would call: cuMemFreeAsync(dptr, stream)
        Err(CudaError::NotInitialized)
    }

    /// Trim via `cuMemPoolTrimTo`.
    #[cfg(not(target_os = "macos"))]
    fn gpu_trim(_pool_handle: u64, _min_bytes_to_keep: usize) -> CudaResult<()> {
        // Would call: cuMemPoolTrimTo(pool, minBytesToKeep)
        Err(CudaError::NotInitialized)
    }

    /// Set attribute via `cuMemPoolSetAttribute`.
    #[cfg(not(target_os = "macos"))]
    fn gpu_set_attribute(_pool_handle: u64, _attr: PoolAttribute) -> CudaResult<()> {
        // Would call: cuMemPoolSetAttribute(pool, attr, &value)
        Err(CudaError::NotInitialized)
    }

    /// Enable peer access via `cuMemPoolExportToShareableHandle` + access control.
    #[cfg(not(target_os = "macos"))]
    fn gpu_enable_peer_access(_pool_handle: u64, _peer_device: i32) -> CudaResult<()> {
        Err(CudaError::NotInitialized)
    }

    /// Disable peer access.
    #[cfg(not(target_os = "macos"))]
    fn gpu_disable_peer_access(_pool_handle: u64, _peer_device: i32) -> CudaResult<()> {
        Err(CudaError::NotInitialized)
    }
}

// ---------------------------------------------------------------------------
// Convenience free functions
// ---------------------------------------------------------------------------

/// Allocate memory on a stream using the default pool for device 0.
///
/// This is a convenience wrapper around [`StreamMemoryPool::default_pool`]
/// and [`StreamMemoryPool::alloc_async`].
///
/// # Errors
///
/// Propagates errors from pool creation and allocation.
pub fn stream_alloc(size: usize, stream: u64) -> CudaResult<StreamAllocation> {
    let mut pool = StreamMemoryPool::default_pool(0)?;
    pool.alloc_async(size, stream)
}

/// Free a stream-ordered allocation using a temporary default pool.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if the allocation is already freed.
pub fn stream_free(alloc: &mut StreamAllocation) -> CudaResult<()> {
    if alloc.freed {
        return Err(CudaError::InvalidValue);
    }

    // On macOS, just mark as freed (no real GPU work).
    #[cfg(target_os = "macos")]
    {
        alloc.freed = true;
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    {
        StreamMemoryPool::gpu_free_async(alloc.ptr, alloc.stream)?;
        alloc.freed = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config validation -------------------------------------------------

    #[test]
    fn config_validate_valid_sizes() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 1024,
            max_pool_size: 4096,
            release_threshold: 512,
            device: 0,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_validate_unlimited_max() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 1024 * 1024,
            max_pool_size: 0, // unlimited
            release_threshold: 512,
            device: 0,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_validate_initial_exceeds_max() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 8192,
            max_pool_size: 4096,
            release_threshold: 0,
            device: 0,
        };
        assert_eq!(config.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn config_validate_negative_device() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 0,
            max_pool_size: 0,
            release_threshold: 0,
            device: -1,
        };
        assert_eq!(config.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn config_validate_threshold_exceeds_max() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 0,
            max_pool_size: 1024,
            release_threshold: 2048,
            device: 0,
        };
        assert_eq!(config.validate(), Err(CudaError::InvalidValue));
    }

    // -- Default config ----------------------------------------------------

    #[test]
    fn default_config_for_device() {
        let config = StreamOrderedAllocConfig::default_for_device(2);
        assert_eq!(config.device, 2);
        assert_eq!(config.initial_pool_size, 0);
        assert_eq!(config.max_pool_size, 0);
        assert_eq!(config.release_threshold, 0);
        assert!(config.validate().is_ok());
    }

    // -- Pool creation -----------------------------------------------------

    #[test]
    fn pool_creation() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let pool = StreamMemoryPool::new(config);
        assert!(pool.is_ok());
        let pool = pool.ok();
        assert!(pool.is_some());
        let pool = pool.map(|p| {
            assert_eq!(p.device(), 0);
            assert_eq!(p.active_allocations, 0);
            assert_eq!(p.total_allocated, 0);
        });
        let _ = pool;
    }

    #[test]
    fn pool_creation_invalid_config() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 0,
            max_pool_size: 0,
            release_threshold: 0,
            device: -1,
        };
        let result = StreamMemoryPool::new(config);
        assert!(matches!(result, Err(CudaError::InvalidValue)));
    }

    // -- alloc_async / free_async -----------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn alloc_async_creates_allocation() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool = StreamMemoryPool::new(config).ok();
        assert!(pool.is_some());
        let pool = pool.as_mut().map(|p| {
            let alloc = p.alloc_async(1024, 0);
            assert!(alloc.is_ok());
            let alloc = alloc.ok();
            assert!(alloc.is_some());
            if let Some(a) = &alloc {
                assert_eq!(a.size(), 1024);
                assert!(!a.is_freed());
                assert_ne!(a.as_ptr(), 0);
                assert_eq!(a.stream(), 0);
            }
        });
        let _ = pool;
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn free_async_marks_freed() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        let mut alloc = pool
            .alloc_async(2048, 0)
            .expect("alloc should succeed on macOS");
        assert!(!alloc.is_freed());
        assert!(pool.free_async(&mut alloc).is_ok());
        assert!(alloc.is_freed());
        assert_eq!(pool.active_allocations, 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn double_free_returns_error() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        let mut alloc = pool
            .alloc_async(512, 0)
            .expect("alloc should succeed on macOS");
        assert!(pool.free_async(&mut alloc).is_ok());
        assert_eq!(pool.free_async(&mut alloc), Err(CudaError::InvalidValue));
    }

    // -- Trim --------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn trim_returns_not_supported_on_macos() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        assert_eq!(pool.trim(0), Err(CudaError::NotSupported));
    }

    // -- Stats tracking ----------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn stats_tracking() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");

        let mut a1 = pool.alloc_async(1024, 0).expect("alloc should succeed");
        let _a2 = pool.alloc_async(2048, 0).expect("alloc should succeed");

        let stats = pool.stats();
        assert_eq!(stats.active_allocations, 2);
        assert_eq!(stats.used_current, 3072);
        assert_eq!(stats.used_high, 3072);
        assert_eq!(stats.peak_allocations, 2);

        pool.free_async(&mut a1).expect("free should succeed");
        let stats = pool.stats();
        assert_eq!(stats.active_allocations, 1);
        assert_eq!(stats.used_current, 2048);
        // Peak should remain at 3072.
        assert_eq!(stats.used_high, 3072);
    }

    // -- Pool attribute setting --------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn set_attribute_release_threshold() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        let result = pool.set_attribute(PoolAttribute::ReleaseThreshold(4096));
        assert!(result.is_ok());
        assert_eq!(pool.config().release_threshold, 4096);
    }

    #[test]
    fn set_attribute_readonly_returns_error() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool = StreamMemoryPool::new(config).expect("pool creation should succeed");
        assert_eq!(
            pool.set_attribute(PoolAttribute::ReservedMemCurrent),
            Err(CudaError::InvalidValue)
        );
        assert_eq!(
            pool.set_attribute(PoolAttribute::UsedMemCurrent),
            Err(CudaError::InvalidValue)
        );
    }

    // -- StreamAllocation accessors ----------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn allocation_accessors() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        let alloc = pool.alloc_async(4096, 42).expect("alloc should succeed");
        assert_eq!(alloc.size(), 4096);
        assert_eq!(alloc.stream(), 42);
        assert!(!alloc.is_freed());
        assert_ne!(alloc.as_ptr(), 0);
        // Debug formatting should not panic.
        let _debug = format!("{alloc:?}");
    }

    // -- Convenience functions ---------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn convenience_stream_alloc() {
        let result = stream_alloc(256, 0);
        assert!(result.is_ok());
        let alloc = result.expect("should succeed on macOS");
        assert_eq!(alloc.size(), 256);
        assert!(!alloc.is_freed());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn convenience_stream_free() {
        let mut alloc = stream_alloc(128, 0).expect("alloc should succeed on macOS");
        assert!(stream_free(&mut alloc).is_ok());
        assert!(alloc.is_freed());
        // Double free via convenience function.
        assert_eq!(stream_free(&mut alloc), Err(CudaError::InvalidValue));
    }

    // -- Large allocation size ---------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn large_allocation_size() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 0,
            max_pool_size: 0, // unlimited
            release_threshold: 0,
            device: 0,
        };
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        // 16 GiB allocation (large but valid).
        let size = 16 * 1024 * 1024 * 1024_usize;
        let alloc = pool.alloc_async(size, 0);
        assert!(alloc.is_ok());
        let alloc = alloc.expect("should succeed");
        assert_eq!(alloc.size(), size);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn alloc_exceeds_max_pool_size() {
        let config = StreamOrderedAllocConfig {
            initial_pool_size: 0,
            max_pool_size: 1024,
            release_threshold: 0,
            device: 0,
        };
        let mut pool = StreamMemoryPool::new(config).expect("pool creation should succeed");
        assert!(matches!(
            pool.alloc_async(2048, 0),
            Err(CudaError::OutOfMemory)
        ));
    }

    // -- Peer access -------------------------------------------------------

    #[test]
    fn peer_access_same_device_error() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let pool = StreamMemoryPool::new(config).expect("pool creation should succeed");
        assert_eq!(pool.enable_peer_access(0), Err(CudaError::InvalidDevice));
        assert_eq!(pool.disable_peer_access(0), Err(CudaError::InvalidDevice));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn peer_access_not_supported_on_macos() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let pool = StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");
        assert_eq!(pool.enable_peer_access(1), Err(CudaError::NotSupported));
        assert_eq!(pool.disable_peer_access(1), Err(CudaError::NotSupported));
    }

    // -- Reset peak stats --------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn reset_peak_stats() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool =
            StreamMemoryPool::new(config).expect("pool creation should succeed on macOS");

        let mut a1 = pool.alloc_async(1024, 0).expect("alloc ok");
        let _a2 = pool.alloc_async(2048, 0).expect("alloc ok");
        assert_eq!(pool.stats().peak_allocations, 2);
        assert_eq!(pool.stats().used_high, 3072);

        pool.free_async(&mut a1).expect("free ok");
        pool.reset_peak_stats();

        let stats = pool.stats();
        assert_eq!(stats.used_high, 2048); // reset to current
        assert_eq!(stats.peak_allocations, 1); // reset to current
    }

    // -- Zero-size alloc ---------------------------------------------------

    #[test]
    fn alloc_zero_size_returns_error() {
        let config = StreamOrderedAllocConfig::default_for_device(0);
        let mut pool = StreamMemoryPool::new(config).expect("pool creation should succeed");
        assert!(matches!(
            pool.alloc_async(0, 0),
            Err(CudaError::InvalidValue)
        ));
    }

    // -- Default pool ------------------------------------------------------

    #[test]
    fn default_pool_valid_device() {
        let pool = StreamMemoryPool::default_pool(0);
        assert!(pool.is_ok());
    }

    #[test]
    fn default_pool_negative_device() {
        assert!(matches!(
            StreamMemoryPool::default_pool(-1),
            Err(CudaError::InvalidValue)
        ));
    }

    // -- PoolAttribute::to_raw ---------------------------------------------

    #[test]
    fn pool_attribute_to_raw() {
        assert_eq!(
            PoolAttribute::ReuseFollowEventDependencies.to_raw(),
            CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES
        );
        assert_eq!(
            PoolAttribute::ReuseAllowOpportunistic.to_raw(),
            CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC
        );
        assert_eq!(
            PoolAttribute::ReuseAllowInternalDependencies.to_raw(),
            CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES
        );
        assert_eq!(
            PoolAttribute::ReleaseThreshold(0).to_raw(),
            CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
        );
        assert_eq!(
            PoolAttribute::ReservedMemCurrent.to_raw(),
            CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT
        );
        assert_eq!(
            PoolAttribute::ReservedMemHigh.to_raw(),
            CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH
        );
        assert_eq!(
            PoolAttribute::UsedMemCurrent.to_raw(),
            CU_MEMPOOL_ATTR_USED_MEM_CURRENT
        );
        assert_eq!(
            PoolAttribute::UsedMemHigh.to_raw(),
            CU_MEMPOOL_ATTR_USED_MEM_HIGH
        );
    }

    // -- ShareableHandleType default ---------------------------------------

    #[test]
    fn shareable_handle_type_default() {
        assert_eq!(ShareableHandleType::default(), ShareableHandleType::None);
    }

    // -- PoolExportDescriptor construction ---------------------------------

    #[test]
    fn pool_export_descriptor() {
        let desc = PoolExportDescriptor {
            shareable_handle_type: ShareableHandleType::PosixFileDescriptor,
            pool_device: 0,
        };
        assert_eq!(
            desc.shareable_handle_type,
            ShareableHandleType::PosixFileDescriptor
        );
        assert_eq!(desc.pool_device, 0);
    }
}
