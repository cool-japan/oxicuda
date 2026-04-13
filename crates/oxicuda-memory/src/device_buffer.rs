//! Type-safe device (GPU VRAM) memory buffer.
//!
//! [`DeviceBuffer<T>`] owns a contiguous allocation of `T` elements in device
//! memory.  It supports synchronous and asynchronous copies to/from host
//! memory, device-to-device copies, and zero-initialisation via `cuMemsetD8`.
//!
//! The buffer is parameterised over `T: Copy` so that only plain-old-data
//! types can be stored — no heap pointers that would be meaningless on the
//! GPU.
//!
//! # Ownership
//!
//! The allocation is freed automatically when the buffer is dropped.  If
//! `cuMemFree_v2` fails during [`Drop`], the error is logged via
//! [`tracing::warn`] rather than panicking.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_memory::DeviceBuffer;
//! let mut buf = DeviceBuffer::<f32>::alloc(1024)?;
//! let host_data = vec![1.0_f32; 1024];
//! buf.copy_from_host(&host_data)?;
//!
//! let mut result = vec![0.0_f32; 1024];
//! buf.copy_to_host(&mut result)?;
//! assert_eq!(result, host_data);
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::ffi::c_void;
use std::marker::PhantomData;

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::stream::Stream;

// ---------------------------------------------------------------------------
// DeviceBuffer<T>
// ---------------------------------------------------------------------------

/// A contiguous buffer of `T` elements allocated in GPU device memory.
///
/// The buffer owns the underlying `CUdeviceptr` allocation and frees it on
/// drop.  All copy operations validate that source and destination lengths
/// match, returning [`CudaError::InvalidValue`] on mismatch.
pub struct DeviceBuffer<T: Copy> {
    /// Raw CUDA device pointer to the start of the allocation.
    ptr: CUdeviceptr,
    /// Number of `T` elements (not bytes).
    len: usize,
    /// Marker to tie the generic parameter `T` to this struct.
    _phantom: PhantomData<T>,
}

// SAFETY: Device memory is not bound to a specific host thread.  The raw
// pointer is a `u64` handle managed by the CUDA driver, which is thread-safe
// for memory operations when properly synchronised.
unsafe impl<T: Copy + Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocates a device buffer capable of holding `n` elements of type `T`.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `n` is zero.
    /// * [`CudaError::OutOfMemory`] if the GPU cannot satisfy the request.
    /// * Other driver errors propagated from `cuMemAlloc_v2`.
    pub fn alloc(n: usize) -> CudaResult<Self> {
        if n == 0 {
            return Err(CudaError::InvalidValue);
        }
        let byte_size = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CudaError::InvalidValue)?;
        let api = try_driver()?;
        let mut ptr: CUdeviceptr = 0;
        // SAFETY: `cu_mem_alloc_v2` writes a valid device pointer on success.
        let rc = unsafe { (api.cu_mem_alloc_v2)(&mut ptr, byte_size) };
        oxicuda_driver::check(rc)?;
        Ok(Self {
            ptr,
            len: n,
            _phantom: PhantomData,
        })
    }

    /// Allocates a device buffer of `n` elements and zero-initialises every byte.
    ///
    /// This is equivalent to [`alloc`](Self::alloc) followed by a
    /// `cuMemsetD8_v2` call that writes `0` to every byte.
    ///
    /// # Errors
    ///
    /// Same as [`alloc`](Self::alloc), plus any error from `cuMemsetD8_v2`.
    pub fn zeroed(n: usize) -> CudaResult<Self> {
        let buf = Self::alloc(n)?;
        let api = try_driver()?;
        // SAFETY: the buffer was just allocated with the correct byte size.
        let rc = unsafe { (api.cu_memset_d8_v2)(buf.ptr, 0, buf.byte_size()) };
        oxicuda_driver::check(rc)?;
        Ok(buf)
    }

    /// Allocates a device buffer and copies the contents of `data` into it.
    ///
    /// The resulting buffer has the same length as the input slice.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `data` is empty.
    /// * Other driver errors from allocation or the host-to-device copy.
    pub fn from_host(data: &[T]) -> CudaResult<Self> {
        let mut buf = Self::alloc(data.len())?;
        buf.copy_from_host(data)?;
        Ok(buf)
    }

    /// Copies data from a host slice into this device buffer (synchronous).
    ///
    /// The slice length must exactly match the buffer length.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `src.len() != self.len()`.
    /// * Other driver errors from `cuMemcpyHtoD_v2`.
    pub fn copy_from_host(&mut self, src: &[T]) -> CudaResult<()> {
        if src.len() != self.len {
            return Err(CudaError::InvalidValue);
        }
        let api = try_driver()?;
        // SAFETY: `src` is a valid host slice with the correct byte count.
        let rc = unsafe {
            (api.cu_memcpy_htod_v2)(self.ptr, src.as_ptr().cast::<c_void>(), self.byte_size())
        };
        oxicuda_driver::check(rc)
    }

    /// Copies this device buffer's contents into a host slice (synchronous).
    ///
    /// The slice length must exactly match the buffer length.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `dst.len() != self.len()`.
    /// * Other driver errors from `cuMemcpyDtoH_v2`.
    pub fn copy_to_host(&self, dst: &mut [T]) -> CudaResult<()> {
        if dst.len() != self.len {
            return Err(CudaError::InvalidValue);
        }
        let api = try_driver()?;
        // SAFETY: `dst` is a valid host slice with the correct byte count.
        let rc = unsafe {
            (api.cu_memcpy_dtoh_v2)(
                dst.as_mut_ptr().cast::<c_void>(),
                self.ptr,
                self.byte_size(),
            )
        };
        oxicuda_driver::check(rc)
    }

    /// Copies the entire contents of another device buffer into this one.
    ///
    /// Both buffers must have the same length.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `src.len() != self.len()`.
    /// * Other driver errors from `cuMemcpyDtoD_v2`.
    pub fn copy_from_device(&mut self, src: &DeviceBuffer<T>) -> CudaResult<()> {
        if src.len != self.len {
            return Err(CudaError::InvalidValue);
        }
        let api = try_driver()?;
        // SAFETY: both pointers are valid device allocations of the same size.
        let rc = unsafe { (api.cu_memcpy_dtod_v2)(self.ptr, src.ptr, self.byte_size()) };
        oxicuda_driver::check(rc)
    }

    /// Asynchronously copies data from a host slice into this device buffer.
    ///
    /// The copy is enqueued on `stream` and may not be complete when this
    /// function returns.  The caller must ensure that `src` remains valid
    /// (i.e., is not moved or dropped) until the stream has been
    /// synchronised.  For guaranteed correctness, prefer using a
    /// [`PinnedBuffer`](crate::PinnedBuffer) as the source.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `src.len() != self.len()`.
    /// * Other driver errors from `cuMemcpyHtoDAsync_v2`.
    pub fn copy_from_host_async(&mut self, src: &[T], stream: &Stream) -> CudaResult<()> {
        if src.len() != self.len {
            return Err(CudaError::InvalidValue);
        }
        let api = try_driver()?;
        // SAFETY: the caller is responsible for keeping `src` alive until
        // the stream completes.
        let rc = unsafe {
            (api.cu_memcpy_htod_async_v2)(
                self.ptr,
                src.as_ptr().cast::<c_void>(),
                self.byte_size(),
                stream.raw(),
            )
        };
        oxicuda_driver::check(rc)
    }

    /// Asynchronously copies this device buffer's contents into a host slice.
    ///
    /// The copy is enqueued on `stream` and may not be complete when this
    /// function returns.  The caller must ensure that `dst` remains valid
    /// and is not read until the stream has been synchronised.  For
    /// guaranteed correctness, prefer using a
    /// [`PinnedBuffer`](crate::PinnedBuffer) as the destination.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `dst.len() != self.len()`.
    /// * Other driver errors from `cuMemcpyDtoHAsync_v2`.
    pub fn copy_to_host_async(&self, dst: &mut [T], stream: &Stream) -> CudaResult<()> {
        if dst.len() != self.len {
            return Err(CudaError::InvalidValue);
        }
        let api = try_driver()?;
        // SAFETY: the caller is responsible for keeping `dst` alive until
        // the stream completes.
        let rc = unsafe {
            (api.cu_memcpy_dtoh_async_v2)(
                dst.as_mut_ptr().cast::<c_void>(),
                self.ptr,
                self.byte_size(),
                stream.raw(),
            )
        };
        oxicuda_driver::check(rc)
    }

    /// Returns the number of `T` elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains zero elements.
    ///
    /// In practice this is always `false` because [`alloc`](Self::alloc)
    /// rejects zero-length allocations.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the total size of the allocation in bytes.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Returns the raw [`CUdeviceptr`] handle for this buffer.
    ///
    /// This is useful when passing the pointer to kernel launch parameters
    /// or other low-level driver calls.
    #[inline]
    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Returns a borrowed [`DeviceSlice`] referencing a sub-range of this
    /// buffer starting at element `offset` and spanning `len` elements.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if the requested range exceeds
    /// the buffer bounds (i.e., `offset + len > self.len()`).
    pub fn slice(&self, offset: usize, len: usize) -> CudaResult<DeviceSlice<'_, T>> {
        let end = offset.checked_add(len).ok_or(CudaError::InvalidValue)?;
        if end > self.len {
            return Err(CudaError::InvalidValue);
        }
        let byte_offset = offset
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CudaError::InvalidValue)?;
        Ok(DeviceSlice {
            ptr: self.ptr + byte_offset as u64,
            len,
            _phantom: PhantomData,
        })
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if let Ok(api) = try_driver() {
            // SAFETY: `self.ptr` was allocated by `cu_mem_alloc_v2` and has
            // not yet been freed.
            let rc = unsafe { (api.cu_mem_free_v2)(self.ptr) };
            if rc != 0 {
                tracing::warn!(
                    cuda_error = rc,
                    ptr = self.ptr,
                    len = self.len,
                    "cuMemFree_v2 failed during DeviceBuffer drop"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceSlice<'a, T>
// ---------------------------------------------------------------------------

/// A borrowed, non-owning view into a sub-range of a [`DeviceBuffer`].
///
/// A `DeviceSlice` does not own the memory it points to — it borrows from
/// the parent [`DeviceBuffer`] and is lifetime-bound to it.  This is useful
/// for passing sub-regions of a buffer to kernels or copy operations without
/// extra allocations.
///
/// `DeviceSlice` does **not** implement [`Drop`]; the parent buffer is
/// responsible for freeing the allocation.
pub struct DeviceSlice<'a, T: Copy> {
    /// Raw device pointer to the start of this slice within the parent buffer.
    ptr: CUdeviceptr,
    /// Number of `T` elements in this slice.
    len: usize,
    /// Ties the lifetime to the parent buffer and the element type.
    _phantom: PhantomData<&'a T>,
}

impl<T: Copy> DeviceSlice<'_, T> {
    /// Returns the number of `T` elements in this slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slice contains zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the total size of this slice in bytes.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Returns the raw [`CUdeviceptr`] handle for the start of this slice.
    #[inline]
    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}
