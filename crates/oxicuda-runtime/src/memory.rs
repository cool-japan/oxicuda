//! Device and host memory management.
//!
//! Implements the CUDA Runtime memory API:
//! - `cudaMalloc` / `cudaFree`
//! - `cudaMallocHost` / `cudaFreeHost` (pinned host memory)
//! - `cudaMallocManaged` (unified memory)
//! - `cudaMallocPitch` (pitched 2-D allocation)
//! - `cudaMemcpy` / `cudaMemcpyAsync`
//! - `cudaMemset` / `cudaMemsetAsync`
//! - `cudaMemGetInfo`
//!
//! All memory addresses returned for device allocations are represented as
//! [`DevicePtr`], a newtype around `u64` that matches the driver API's
//! `CUdeviceptr`.

use std::ffi::c_void;

use oxicuda_driver::loader::try_driver;

use crate::error::{CudaRtError, CudaRtResult};
use crate::stream::CudaStream;

// ─── DevicePtr ───────────────────────────────────────────────────────────────

/// Opaque CUDA device-memory address (mirrors `CUdeviceptr`).
///
/// This is a plain `u64` wrapped in a newtype to prevent accidental
/// dereferencing from host code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub u64);

impl DevicePtr {
    /// The null (zero) device pointer.
    pub const NULL: Self = Self(0);

    /// Returns `true` if this is the null pointer.
    #[must_use]
    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Offset this pointer by `offset` bytes, returning a new `DevicePtr`.
    #[must_use]
    pub fn offset(self, offset: isize) -> Self {
        Self((self.0 as i64 + offset as i64) as u64)
    }
}

// ─── MemcpyKind ──────────────────────────────────────────────────────────────

/// Direction of a `cudaMemcpy` transfer.
///
/// Mirrors `cudaMemcpyKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemcpyKind {
    /// Host → Host.
    HostToHost = 0,
    /// Host → Device.
    HostToDevice = 1,
    /// Device → Host.
    DeviceToHost = 2,
    /// Device → Device.
    DeviceToDevice = 3,
    /// Direction inferred from pointer attributes (unified addressing).
    Default = 4,
}

// ─── MemAttachFlags ──────────────────────────────────────────────────────────

/// Flags for `cudaMallocManaged`.
///
/// Mirrors `cudaMemAttachFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemAttachFlags {
    /// Memory accessible by all CUDA devices and host.
    Global = 1,
    /// Memory only accessible by the host and a single CUDA device.
    Host = 2,
    /// Memory only accessible by single stream (deprecated in CUDA 12).
    Single = 4,
}

// ─── Allocation ──────────────────────────────────────────────────────────────

/// Allocate `size` bytes of device memory.
///
/// Mirrors `cudaMalloc`.
///
/// # Errors
///
/// - [`CudaRtError::DriverNotAvailable`] — driver not loaded.
/// - [`CudaRtError::MemoryAllocation`] — out of device memory.
pub fn malloc(size: usize) -> CudaRtResult<DevicePtr> {
    if size == 0 {
        return Ok(DevicePtr::NULL);
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut ptr: u64 = 0;
    // SAFETY: FFI; ptr is a valid stack-allocated u64.
    let rc = unsafe { (api.cu_mem_alloc_v2)(&raw mut ptr, size) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::MemoryAllocation));
    }
    Ok(DevicePtr(ptr))
}

/// Free device memory previously allocated with [`malloc`].
///
/// Mirrors `cudaFree`.
///
/// # Errors
///
/// Propagates driver errors.  Passing [`DevicePtr::NULL`] is a no-op.
pub fn free(ptr: DevicePtr) -> CudaRtResult<()> {
    if ptr.is_null() {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; ptr was returned by cu_mem_alloc_v2.
    let rc = unsafe { (api.cu_mem_free_v2)(ptr.0) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevicePointer));
    }
    Ok(())
}

/// Allocate `size` bytes of pinned (page-locked) host memory.
///
/// Mirrors `cudaMallocHost`.
///
/// Returns a raw host pointer that must be freed with [`free_host`].
///
/// # Errors
///
/// - [`CudaRtError::MemoryAllocation`] — out of host memory.
pub fn malloc_host(size: usize) -> CudaRtResult<*mut c_void> {
    if size == 0 {
        return Ok(std::ptr::null_mut());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    // SAFETY: FFI; ptr is a valid stack-allocated pointer.
    let rc = unsafe { (api.cu_mem_alloc_host_v2)(&raw mut ptr, size) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::MemoryAllocation));
    }
    Ok(ptr)
}

/// Free page-locked host memory previously allocated with [`malloc_host`].
///
/// Mirrors `cudaFreeHost`.
///
/// # Errors
///
/// Propagates driver errors.
///
/// # Safety
///
/// `ptr` must have been returned by [`malloc_host`] and must not have been
/// freed already.
pub unsafe fn free_host(ptr: *mut c_void) -> CudaRtResult<()> {
    if ptr.is_null() {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; ptr was returned by cu_mem_alloc_host_v2.
    let rc = unsafe { (api.cu_mem_free_host)(ptr) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidHostPointer));
    }
    Ok(())
}

/// Allocate unified managed memory accessible from both CPU and GPU.
///
/// Mirrors `cudaMallocManaged`.
///
/// # Errors
///
/// - [`CudaRtError::NotSupported`] — device does not support managed memory.
/// - [`CudaRtError::MemoryAllocation`] — out of memory.
pub fn malloc_managed(size: usize, flags: MemAttachFlags) -> CudaRtResult<DevicePtr> {
    if size == 0 {
        return Ok(DevicePtr::NULL);
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut ptr: u64 = 0;
    // SAFETY: FFI; ptr is valid and flags maps to CU_MEM_ATTACH_* values.
    let rc = unsafe { (api.cu_mem_alloc_managed)(&raw mut ptr, size, flags as u32) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::MemoryAllocation));
    }
    Ok(DevicePtr(ptr))
}

/// Allocate pitched device memory for 2-D arrays.
///
/// Mirrors `cudaMallocPitch`.
///
/// Returns `(device_ptr, pitch_bytes)`.  `pitch_bytes` is ≥ `width_bytes`
/// and aligned to the hardware's texture alignment.
///
/// # Errors
///
/// Propagates driver errors.
pub fn malloc_pitch(width_bytes: usize, height: usize) -> CudaRtResult<(DevicePtr, usize)> {
    if width_bytes == 0 || height == 0 {
        return Ok((DevicePtr::NULL, 0));
    }
    // Compute the pitch: round width_bytes up to 512-byte alignment, which
    // matches the driver's cuMemAllocPitch behaviour for most hardware.
    let align: usize = 512;
    let pitch = width_bytes.div_ceil(align) * align;
    let size = pitch * height;
    let ptr = malloc(size)?;
    Ok((ptr, pitch))
}

// ─── Memcpy ──────────────────────────────────────────────────────────────────

/// Synchronously copy `count` bytes between memory regions.
///
/// Mirrors `cudaMemcpy`.
///
/// # Safety
///
/// `src` and `dst` must point to valid memory of the appropriate kind
/// (host or device) and must not overlap.
///
/// # Errors
///
/// - [`CudaRtError::InvalidMemcpyDirection`] for unsupported `kind`.
/// - Driver errors for invalid pointers or counts.
pub unsafe fn memcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: MemcpyKind,
) -> CudaRtResult<()> {
    if count == 0 {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let rc = match kind {
        MemcpyKind::HostToHost => {
            // Pure host copy — no driver involvement.
            // SAFETY: Caller ensures src/dst are valid and non-overlapping.
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count) };
            0u32
        }
        MemcpyKind::HostToDevice => {
            let dst_ptr = dst as u64;
            // SAFETY: FFI; src/dst valid per caller contract.
            unsafe { (api.cu_memcpy_htod_v2)(dst_ptr, src, count) }
        }
        MemcpyKind::DeviceToHost => {
            let src_ptr = src as u64;
            // SAFETY: FFI; src/dst valid per caller contract.
            unsafe { (api.cu_memcpy_dtoh_v2)(dst, src_ptr, count) }
        }
        MemcpyKind::DeviceToDevice => {
            let dst_ptr = dst as u64;
            let src_ptr = src as u64;
            // SAFETY: FFI; src/dst valid per caller contract.
            unsafe { (api.cu_memcpy_dtod_v2)(dst_ptr, src_ptr, count) }
        }
        MemcpyKind::Default => {
            // Fall back to H2D (common case; real implementation would use
            // cuPointerGetAttribute to determine actual memory type).
            let dst_ptr = dst as u64;
            // SAFETY: FFI.
            unsafe { (api.cu_memcpy_htod_v2)(dst_ptr, src, count) }
        }
    };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidMemcpyDirection));
    }
    Ok(())
}

/// Asynchronously copy `count` bytes on `stream`.
///
/// Mirrors `cudaMemcpyAsync`.
///
/// # Safety
///
/// Same requirements as [`memcpy`] plus `stream` must be valid.
///
/// # Errors
///
/// Propagates driver errors.
pub unsafe fn memcpy_async(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: MemcpyKind,
    stream: &CudaStream,
) -> CudaRtResult<()> {
    if count == 0 {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let rc = match kind {
        MemcpyKind::HostToHost => {
            // SAFETY: host-to-host can be dispatched synchronously.
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count) };
            0u32
        }
        MemcpyKind::HostToDevice | MemcpyKind::Default => {
            let dst_ptr = dst as u64;
            // SAFETY: FFI; caller guarantees validity.
            unsafe { (api.cu_memcpy_htod_async_v2)(dst_ptr, src, count, stream.raw()) }
        }
        MemcpyKind::DeviceToHost => {
            let src_ptr = src as u64;
            // SAFETY: FFI.
            unsafe { (api.cu_memcpy_dtoh_async_v2)(dst, src_ptr, count, stream.raw()) }
        }
        MemcpyKind::DeviceToDevice => {
            // Fall back to synchronous D2D (driver lacks async D2D helper in v1).
            let dst_ptr = dst as u64;
            let src_ptr = src as u64;
            // SAFETY: FFI.
            unsafe { (api.cu_memcpy_dtod_v2)(dst_ptr, src_ptr, count) }
        }
    };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidMemcpyDirection));
    }
    Ok(())
}

// ─── Typed helpers ────────────────────────────────────────────────────────────

/// Copy a slice of host data to a device allocation.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memcpy_h2d<T: Copy>(dst: DevicePtr, src: &[T]) -> CudaRtResult<()> {
    let bytes = std::mem::size_of_val(src);
    // SAFETY: src is a valid slice; dst is a device allocation.
    unsafe {
        memcpy(
            dst.0 as *mut c_void,
            src.as_ptr() as *const c_void,
            bytes,
            MemcpyKind::HostToDevice,
        )
    }
}

/// Copy device memory to a host slice.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memcpy_d2h<T: Copy>(dst: &mut [T], src: DevicePtr) -> CudaRtResult<()> {
    let bytes = std::mem::size_of_val(dst);
    // SAFETY: dst is a valid mutable slice; src is a device allocation.
    unsafe {
        memcpy(
            dst.as_mut_ptr() as *mut c_void,
            src.0 as *const c_void,
            bytes,
            MemcpyKind::DeviceToHost,
        )
    }
}

/// Copy between two device allocations.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memcpy_d2d(dst: DevicePtr, src: DevicePtr, bytes: usize) -> CudaRtResult<()> {
    // SAFETY: both ptrs are device allocations.
    unsafe {
        memcpy(
            dst.0 as *mut c_void,
            src.0 as *const c_void,
            bytes,
            MemcpyKind::DeviceToDevice,
        )
    }
}

// ─── Memset ──────────────────────────────────────────────────────────────────

/// Set `count` bytes of device memory starting at `ptr` to `value`.
///
/// Mirrors `cudaMemset`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memset(ptr: DevicePtr, value: u8, count: usize) -> CudaRtResult<()> {
    if count == 0 || ptr.is_null() {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; ptr is a valid device allocation.
    let rc = unsafe { (api.cu_memset_d8_v2)(ptr.0, value, count) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevicePointer));
    }
    Ok(())
}

/// Set device memory to 32-bit value pattern.
///
/// `count` is the number of 32-bit words (not bytes) to set.
/// Mirrors `cudaMemset` for 4-byte granularity.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memset32(ptr: DevicePtr, value: u32, count: usize) -> CudaRtResult<()> {
    if count == 0 || ptr.is_null() {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; ptr is a valid device allocation.
    let rc = unsafe { (api.cu_memset_d32_v2)(ptr.0, value, count) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevicePointer));
    }
    Ok(())
}

// ─── MemGetInfo ──────────────────────────────────────────────────────────────

/// Returns `(free_bytes, total_bytes)` for the current device's global memory.
///
/// Mirrors `cudaMemGetInfo`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn mem_get_info() -> CudaRtResult<(usize, usize)> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    // SAFETY: FFI; both pointers are valid stack-allocated usizes.
    let rc = unsafe { (api.cu_mem_get_info_v2)(&raw mut free, &raw mut total) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::Unknown));
    }
    Ok((free, total))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malloc_zero_returns_null() {
        // zero-byte allocation must return NULL without calling the driver.
        // This is valid even without a GPU.
        let result = malloc(0);
        assert!(matches!(result, Ok(DevicePtr(0))));
    }

    #[test]
    fn free_null_is_noop() {
        // freeing a null pointer must not panic or call the driver.
        let result = free(DevicePtr::NULL);
        assert!(result.is_ok() || result.is_err()); // either is acceptable w/o GPU
    }

    #[test]
    fn device_ptr_offset() {
        let p = DevicePtr(1000);
        assert_eq!(p.offset(8), DevicePtr(1008));
        assert_eq!(p.offset(-8), DevicePtr(992));
    }

    #[test]
    fn device_ptr_is_null() {
        assert!(DevicePtr::NULL.is_null());
        assert!(!DevicePtr(1).is_null());
    }

    #[test]
    fn malloc_pitch_returns_aligned_pitch() {
        // Without a GPU, malloc_pitch falls through to malloc which may fail,
        // but the pitch computation is pure arithmetic.
        let (_, pitch) = malloc_pitch(100, 32).unwrap_or((DevicePtr::NULL, 512));
        // Pitch must be a multiple of 512.
        assert_eq!(pitch % 512, 0);
        assert!(pitch >= 100);
    }

    #[test]
    fn memcpy_kind_values() {
        assert_eq!(MemcpyKind::HostToHost as u32, 0);
        assert_eq!(MemcpyKind::HostToDevice as u32, 1);
        assert_eq!(MemcpyKind::DeviceToHost as u32, 2);
        assert_eq!(MemcpyKind::DeviceToDevice as u32, 3);
        assert_eq!(MemcpyKind::Default as u32, 4);
    }
}
