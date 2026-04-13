//! Unified (managed) memory buffer.
//!
//! [`UnifiedBuffer<T>`] wraps `cuMemAllocManaged`, which allocates memory
//! that is automatically migrated between host and device by the CUDA
//! Unified Memory subsystem.  The allocation is accessible from both CPU
//! code (via [`as_slice`](UnifiedBuffer::as_slice) /
//! [`as_mut_slice`](UnifiedBuffer::as_mut_slice)) and GPU kernels (via
//! [`as_device_ptr`](UnifiedBuffer::as_device_ptr)).
//!
//! # Coherence caveat
//!
//! The host-side accessors are only safe to call when no GPU kernel is
//! concurrently reading or writing the same memory.  After launching a
//! kernel that touches a unified buffer, synchronise the stream (or the
//! entire context) before accessing the data from the host.
//!
//! # Ownership
//!
//! The allocation is freed with `cuMemFree_v2` on drop.  Errors during
//! drop are logged via [`tracing::warn`].
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_memory::UnifiedBuffer;
//! let mut ubuf = UnifiedBuffer::<f32>::alloc(512)?;
//! // Write from the host side (no kernel running).
//! for (i, v) in ubuf.as_mut_slice().iter_mut().enumerate() {
//!     *v = i as f32;
//! }
//! // Pass ubuf.as_device_ptr() to a kernel…
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::marker::PhantomData;

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::ffi::{CU_MEM_ATTACH_GLOBAL, CUdeviceptr};
use oxicuda_driver::loader::try_driver;

// ---------------------------------------------------------------------------
// UnifiedBuffer<T>
// ---------------------------------------------------------------------------

/// A contiguous buffer of `T` elements in CUDA unified (managed) memory.
///
/// Unified memory is accessible from both the host CPU and the GPU device.
/// The CUDA driver transparently migrates pages between host and device as
/// needed.  This simplifies programming at the cost of potential migration
/// overhead compared to explicit device buffers.
pub struct UnifiedBuffer<T: Copy> {
    /// The CUDA device pointer.  For managed memory this value is also a
    /// valid host pointer (on 64-bit systems with UVA).
    ptr: CUdeviceptr,
    /// Host-accessible pointer derived from `ptr`.
    host_ptr: *mut T,
    /// Number of `T` elements (not bytes).
    len: usize,
    /// Marker to tie the generic parameter `T` to this struct.
    _phantom: PhantomData<T>,
}

// SAFETY: Unified memory is accessible from any thread on both host and
// device.  Proper synchronisation is the caller's responsibility.
unsafe impl<T: Copy + Send> Send for UnifiedBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for UnifiedBuffer<T> {}

impl<T: Copy> UnifiedBuffer<T> {
    /// Allocates a unified memory buffer capable of holding `n` elements of
    /// type `T`.
    ///
    /// The memory is allocated with [`CU_MEM_ATTACH_GLOBAL`], making it
    /// accessible from any stream on any device in the system.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `n` is zero.
    /// * [`CudaError::OutOfMemory`] if the allocation fails.
    /// * Other driver errors from `cuMemAllocManaged`.
    pub fn alloc(n: usize) -> CudaResult<Self> {
        if n == 0 {
            return Err(CudaError::InvalidValue);
        }
        let byte_size = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CudaError::InvalidValue)?;
        let api = try_driver()?;
        let mut dev_ptr: CUdeviceptr = 0;
        // SAFETY: `cu_mem_alloc_managed` writes a valid device pointer that
        // is also host-accessible (UVA).
        let rc =
            unsafe { (api.cu_mem_alloc_managed)(&mut dev_ptr, byte_size, CU_MEM_ATTACH_GLOBAL) };
        oxicuda_driver::check(rc)?;
        // On 64-bit systems with UVA, the device pointer value is the same
        // as the host virtual address.
        let host_ptr = dev_ptr as *mut T;
        Ok(Self {
            ptr: dev_ptr,
            host_ptr,
            len: n,
            _phantom: PhantomData,
        })
    }

    /// Returns the number of `T` elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the total size of the allocation in bytes.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Returns the raw [`CUdeviceptr`] handle for use in kernel launches
    /// and other device-side operations.
    #[inline]
    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Returns a shared slice over the buffer's host-accessible contents.
    ///
    /// # Safety note
    ///
    /// This is only safe to call when no GPU kernel is concurrently
    /// reading or writing this buffer.  Synchronise the relevant stream
    /// or context before calling this method.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `host_ptr` is valid for `len` elements when no device
        // kernel is concurrently accessing the memory.  The caller is
        // responsible for proper synchronisation.
        unsafe { std::slice::from_raw_parts(self.host_ptr, self.len) }
    }

    /// Returns a mutable slice over the buffer's host-accessible contents.
    ///
    /// # Safety note
    ///
    /// This is only safe to call when no GPU kernel is concurrently
    /// reading or writing this buffer.  Synchronise the relevant stream
    /// or context before calling this method.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `host_ptr` is valid for `len` elements when no device
        // kernel is concurrently accessing the memory.  The caller is
        // responsible for proper synchronisation.
        unsafe { std::slice::from_raw_parts_mut(self.host_ptr, self.len) }
    }
}

impl<T: Copy> Drop for UnifiedBuffer<T> {
    fn drop(&mut self) {
        if let Ok(api) = try_driver() {
            // SAFETY: `self.ptr` was allocated by `cu_mem_alloc_managed`
            // and has not yet been freed.
            let rc = unsafe { (api.cu_mem_free_v2)(self.ptr) };
            if rc != 0 {
                tracing::warn!(
                    cuda_error = rc,
                    ptr = self.ptr,
                    len = self.len,
                    "cuMemFree_v2 failed during UnifiedBuffer drop"
                );
            }
        }
    }
}
