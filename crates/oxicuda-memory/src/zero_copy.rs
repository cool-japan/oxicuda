//! Zero-copy (host-mapped) memory.
//!
//! Allows GPU kernels to directly access host memory without explicit
//! transfers.  Useful for small, frequently-updated data or when PCIe
//! bandwidth is acceptable.
//!
//! # How it works
//!
//! Zero-copy memory is allocated on the host with the
//! `CU_MEMHOSTALLOC_DEVICEMAP` flag, which makes it accessible from GPU
//! kernels via a device pointer obtained from `cuMemHostGetDevicePointer`.
//! The GPU reads/writes traverse the PCIe bus on each access, so this is
//! best suited for data that is accessed infrequently or streamed
//! sequentially.
//!
//! # Status
//!
//! This module is a placeholder.  Full implementation is planned for a
//! future release once the `cuMemHostAlloc` and
//! `cuMemHostGetDevicePointer` function pointers are added to
//! `oxicuda-driver`.

use std::marker::PhantomData;

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::ffi::CUdeviceptr;

// ---------------------------------------------------------------------------
// MappedBuffer<T>
// ---------------------------------------------------------------------------

/// A host-allocated, device-mapped (zero-copy) memory buffer.
///
/// The host memory is accessible from both CPU code and GPU kernels.
/// GPU accesses traverse the PCIe bus, making this suitable for small
/// or infrequently-accessed data where the overhead of explicit transfers
/// is not justified.
///
/// # Status
///
/// This type is a placeholder.  The allocation method currently returns
/// [`CudaError::NotSupported`].
///
/// TODO: Add `cu_mem_host_alloc` (with `CU_MEMHOSTALLOC_DEVICEMAP`) and
/// `cu_mem_host_get_device_pointer` to `DriverApi`.
pub struct MappedBuffer<T: Copy> {
    /// Host pointer to the mapped allocation.
    _host_ptr: *mut T,
    /// Corresponding device pointer for kernel access.
    _device_ptr: CUdeviceptr,
    /// Number of `T` elements.
    _len: usize,
    /// Marker for the element type.
    _phantom: PhantomData<T>,
}

// SAFETY: The mapped host memory is not thread-local.
unsafe impl<T: Copy + Send> Send for MappedBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for MappedBuffer<T> {}

impl<T: Copy> MappedBuffer<T> {
    /// Allocates a zero-copy host-mapped buffer of `n` elements.
    ///
    /// # Errors
    ///
    /// Currently always returns [`CudaError::NotSupported`] because the
    /// required driver function pointers are not yet loaded.
    ///
    /// TODO: Implement once `cu_mem_host_alloc` and
    /// `cu_mem_host_get_device_pointer` are available in `DriverApi`.
    pub fn alloc(_n: usize) -> CudaResult<Self> {
        // TODO: call (api.cu_mem_host_alloc)(..., CU_MEMHOSTALLOC_DEVICEMAP)
        // and (api.cu_mem_host_get_device_pointer)(...) when available.
        Err(CudaError::NotSupported)
    }

    /// Returns the number of `T` elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self._len
    }

    /// Returns `true` if the buffer contains zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self._len == 0
    }

    /// Returns the raw device pointer for use in kernel parameters.
    #[inline]
    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self._device_ptr
    }

    /// Returns a raw const pointer to the host-side data.
    #[inline]
    pub fn as_host_ptr(&self) -> *const T {
        self._host_ptr
    }
}

impl<T: Copy> Drop for MappedBuffer<T> {
    fn drop(&mut self) {
        // TODO: call (api.cu_mem_free_host)(self._host_ptr) when available.
        // For now, nothing to free since construction always fails.
    }
}
