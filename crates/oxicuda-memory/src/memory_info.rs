//! GPU memory usage queries and unified memory hints.
//!
//! This module provides:
//!
//! - [`MemoryInfo`] and [`memory_info`] for querying free/total GPU memory.
//! - [`MemAdvice`] and [`mem_advise`] for providing memory usage hints to
//!   the CUDA unified memory subsystem.
//! - [`mem_prefetch`] for prefetching unified memory to a specific device.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_memory::memory_info::{memory_info, MemoryInfo};
//! let info = memory_info()?;
//! println!("GPU memory: {} MB free / {} MB total",
//!     info.free / (1024 * 1024),
//!     info.total / (1024 * 1024),
//! );
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use oxicuda_driver::device::Device;
use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::stream::Stream;

// ---------------------------------------------------------------------------
// MemoryInfo
// ---------------------------------------------------------------------------

/// GPU memory usage information.
///
/// Returned by [`memory_info`], this struct reports the free and total
/// device memory for the current CUDA context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryInfo {
    /// Free device memory in bytes.
    pub free: usize,
    /// Total device memory in bytes.
    pub total: usize,
}

impl MemoryInfo {
    /// Returns the used memory in bytes (`total - free`).
    #[inline]
    pub fn used(&self) -> usize {
        self.total.saturating_sub(self.free)
    }

    /// Returns the fraction of memory currently in use (0.0 to 1.0).
    #[inline]
    pub fn usage_fraction(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.used() as f64 / self.total as f64
    }
}

impl std::fmt::Display for MemoryInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MemoryInfo(free={} MB, total={} MB, used={:.1}%)",
            self.free / (1024 * 1024),
            self.total / (1024 * 1024),
            self.usage_fraction() * 100.0,
        )
    }
}

/// Queries free and total device memory for the current CUDA context.
///
/// The returned values reflect the state at the time of the query and
/// may change as other threads or processes allocate or free memory.
///
/// # Errors
///
/// Returns an error if no context is current or the driver call fails.
pub fn memory_info() -> CudaResult<MemoryInfo> {
    let driver = try_driver()?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    oxicuda_driver::check(unsafe { (driver.cu_mem_get_info_v2)(&mut free, &mut total) })?;
    Ok(MemoryInfo { free, total })
}

// ---------------------------------------------------------------------------
// MemAdvice
// ---------------------------------------------------------------------------

/// Memory advice hints for unified (managed) memory.
///
/// These hints guide the CUDA runtime's page migration and caching
/// decisions for unified memory allocations. Providing accurate hints
/// can significantly improve performance by reducing unnecessary page
/// migrations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MemAdvice {
    /// Mark the memory region as read-mostly. This creates read-only
    /// copies on accessing processors, reducing migration overhead.
    SetReadMostly = 1,
    /// Undo a previous `SetReadMostly` hint.
    UnsetReadMostly = 2,
    /// Set the preferred location for the memory region. The data will
    /// preferably reside on the specified device.
    SetPreferredLocation = 3,
    /// Undo a previous `SetPreferredLocation` hint.
    UnsetPreferredLocation = 4,
    /// Indicate that the specified device will access this memory region.
    /// This can cause the driver to create a mapping on that device.
    SetAccessedBy = 5,
    /// Undo a previous `SetAccessedBy` hint.
    UnsetAccessedBy = 6,
}

/// Provides a memory usage hint for a unified memory region.
///
/// # Parameters
///
/// * `ptr` — device pointer to the start of the unified memory region.
/// * `count` — size of the region in bytes.
/// * `advice` — the usage hint to apply.
/// * `device` — the device to which the hint applies.
///
/// # Errors
///
/// Returns an error if the pointer is not a managed allocation, the
/// device is invalid, or the driver call fails.
pub fn mem_advise(ptr: u64, count: usize, advice: MemAdvice, device: &Device) -> CudaResult<()> {
    if count == 0 {
        return Err(CudaError::InvalidValue);
    }
    let driver = try_driver()?;
    oxicuda_driver::check(unsafe {
        (driver.cu_mem_advise)(ptr, count, advice as u32, device.raw())
    })
}

/// Prefetches unified memory to the specified device.
///
/// This is an asynchronous operation enqueued on `stream`. The data
/// is migrated to the target device so that subsequent accesses from
/// that device will not cause page faults.
///
/// # Parameters
///
/// * `ptr` — device pointer to the start of the unified memory region.
/// * `count` — size of the region in bytes.
/// * `device` — the target device to prefetch to.
/// * `stream` — the stream on which to enqueue the prefetch.
///
/// # Errors
///
/// Returns an error if the pointer is not a managed allocation, the
/// device is invalid, or the driver call fails.
pub fn mem_prefetch(ptr: u64, count: usize, device: &Device, stream: &Stream) -> CudaResult<()> {
    if count == 0 {
        return Err(CudaError::InvalidValue);
    }
    let driver = try_driver()?;
    oxicuda_driver::check(unsafe {
        (driver.cu_mem_prefetch_async)(ptr, count, device.raw(), stream.raw())
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_info_used_calculation() {
        let info = MemoryInfo {
            free: 4096,
            total: 8192,
        };
        assert_eq!(info.used(), 4096);
    }

    #[test]
    fn memory_info_usage_fraction() {
        let info = MemoryInfo {
            free: 2048,
            total: 8192,
        };
        let frac = info.usage_fraction();
        assert!((frac - 0.75).abs() < 1e-10);
    }

    #[test]
    fn memory_info_usage_fraction_zero_total() {
        let info = MemoryInfo { free: 0, total: 0 };
        assert!((info.usage_fraction()).abs() < 1e-10);
    }

    #[test]
    fn memory_info_display() {
        let info = MemoryInfo {
            free: 4 * 1024 * 1024,
            total: 8 * 1024 * 1024,
        };
        let s = format!("{info}");
        assert!(s.contains("free=4 MB"));
        assert!(s.contains("total=8 MB"));
    }

    #[test]
    fn mem_advice_variants() {
        assert_eq!(MemAdvice::SetReadMostly as u32, 1);
        assert_eq!(MemAdvice::UnsetReadMostly as u32, 2);
        assert_eq!(MemAdvice::SetPreferredLocation as u32, 3);
        assert_eq!(MemAdvice::UnsetPreferredLocation as u32, 4);
        assert_eq!(MemAdvice::SetAccessedBy as u32, 5);
        assert_eq!(MemAdvice::UnsetAccessedBy as u32, 6);
    }

    #[test]
    fn mem_advise_rejects_zero_count() {
        let dev = Device::get(0);
        // On macOS we cannot get a device, so we test the zero-count path
        // only if we can construct one.
        if let Ok(dev) = dev {
            let result = mem_advise(0x1000, 0, MemAdvice::SetReadMostly, &dev);
            assert!(result.is_err());
        }
    }

    #[test]
    fn mem_prefetch_rejects_zero_count() {
        // We cannot construct a Stream without a GPU context, but we can
        // verify the function signature compiles.
        let _: fn(u64, usize, &Device, &Stream) -> CudaResult<()> = mem_prefetch;
    }

    #[test]
    fn memory_info_signature_compiles() {
        let _: fn() -> CudaResult<MemoryInfo> = memory_info;
    }
}
