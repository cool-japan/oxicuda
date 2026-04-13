//! Pinned (page-locked) host memory buffer.
//!
//! [`PinnedBuffer<T>`] allocates host memory via `cuMemAllocHost_v2`, which
//! pins the pages so that the CUDA driver can perform DMA transfers without
//! an intermediate staging copy.  This is the recommended source/destination
//! for asynchronous host-device transfers.
//!
//! # Deref
//!
//! `PinnedBuffer<T>` implements [`Deref`] and [`DerefMut`] to `[T]`, so it
//! can be used anywhere a slice is expected.
//!
//! # Ownership
//!
//! The allocation is freed with `cuMemFreeHost` on drop.  Errors during
//! drop are logged via [`tracing::warn`].
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_memory::PinnedBuffer;
//! let mut pinned = PinnedBuffer::<f32>::alloc(256)?;
//! for (i, v) in pinned.iter_mut().enumerate() {
//!     *v = i as f32;
//! }
//! assert_eq!(pinned.len(), 256);
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::ffi::c_void;
use std::ops::{Deref, DerefMut};

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::loader::try_driver;

// ---------------------------------------------------------------------------
// PinnedBuffer<T>
// ---------------------------------------------------------------------------

/// A contiguous buffer of `T` elements in page-locked (pinned) host memory.
///
/// Pinned memory enables the CUDA driver to use DMA for host-device
/// transfers, avoiding an extra copy through a staging buffer.  This makes
/// pinned buffers the preferred choice for async copy operations.
///
/// The buffer dereferences to `&[T]` / `&mut [T]` for ergonomic access.
pub struct PinnedBuffer<T: Copy> {
    /// Pointer to the start of the pinned allocation.
    ptr: *mut T,
    /// Number of `T` elements (not bytes).
    len: usize,
}

// SAFETY: The pinned host memory is not thread-local; it is a plain heap
// allocation that is safe to access from any thread.
unsafe impl<T: Copy + Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for PinnedBuffer<T> {}

impl<T: Copy> PinnedBuffer<T> {
    /// Allocates a pinned host buffer capable of holding `n` elements of type `T`.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `n` is zero.
    /// * [`CudaError::OutOfMemory`] if the host cannot satisfy the request.
    /// * Other driver errors from `cuMemAllocHost_v2`.
    pub fn alloc(n: usize) -> CudaResult<Self> {
        if n == 0 {
            return Err(CudaError::InvalidValue);
        }
        let byte_size = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CudaError::InvalidValue)?;
        let api = try_driver()?;
        let mut raw_ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `cu_mem_alloc_host_v2` writes a valid host pointer on success.
        let rc = unsafe { (api.cu_mem_alloc_host_v2)(&mut raw_ptr, byte_size) };
        oxicuda_driver::check(rc)?;
        Ok(Self {
            ptr: raw_ptr.cast::<T>(),
            len: n,
        })
    }

    /// Allocates a pinned host buffer and copies the contents of `data` into it.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `data` is empty.
    /// * Other driver errors from allocation.
    pub fn from_slice(data: &[T]) -> CudaResult<Self> {
        let buf = Self::alloc(data.len())?;
        // SAFETY: both `data` and `buf.ptr` point to valid memory of
        // `data.len() * size_of::<T>()` bytes, and `T: Copy`.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.ptr, data.len());
        }
        Ok(buf)
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

    /// Returns a raw const pointer to the buffer's data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a raw mutable pointer to the buffer's data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns a shared slice over the buffer's contents.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `self.ptr` is a valid, aligned allocation of `self.len`
        // elements, and we have `&self` so no mutable alias exists.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns a mutable slice over the buffer's contents.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `self.ptr` is a valid, aligned allocation of `self.len`
        // elements, and we have `&mut self` so no other alias exists.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Copy> Deref for PinnedBuffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy> DerefMut for PinnedBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if let Ok(api) = try_driver() {
            // SAFETY: `self.ptr` was allocated by `cu_mem_alloc_host_v2` and
            // has not yet been freed.
            let rc = unsafe { (api.cu_mem_free_host)(self.ptr.cast::<c_void>()) };
            if rc != 0 {
                tracing::warn!(
                    cuda_error = rc,
                    len = self.len,
                    "cuMemFreeHost failed during PinnedBuffer drop"
                );
            }
        }
    }
}
