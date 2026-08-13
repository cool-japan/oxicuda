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
use oxicuda_driver::ffi::{CUdeviceptr, CUstream};
use oxicuda_driver::loader::{DriverApi, try_driver};
use oxicuda_driver::stream::Stream;

// ---------------------------------------------------------------------------
// Legacy-default-stream synchronisation
// ---------------------------------------------------------------------------

/// Blocks until the **legacy default stream** has drained, and *only* that
/// stream.
///
/// # Why this exists (and why it is not `cuCtxSynchronize`)
///
/// The non-async driver transfer primitives this module uses — `cuMemsetD8_v2`
/// (see [`DeviceBuffer::zeroed`]) and `cuMemcpyHtoD_v2` from pageable host
/// memory (see [`DeviceBuffer::copy_from_host`]) — are documented as
/// *asynchronous with respect to the host*: they enqueue onto the legacy
/// default stream (the `NULL` stream) and return before the device-side work
/// has landed. Every OxiCUDA [`Stream`] is created `CU_STREAM_NON_BLOCKING`
/// (see [`Stream::new`]), which by definition opts **out** of the legacy
/// stream's implicit ordering, so a consumer stream can observe the buffer
/// before the memset/DMA completes — a real data race, not a theoretical one.
/// Something must block until the legacy stream has drained.
///
/// `cuCtxSynchronize()` closes that race, but far too widely: it is documented
/// as blocking "until the device has completed all preceding requested tasks"
/// in the *current context* — i.e. every stream in the context, including the
/// non-blocking ones that have nothing to do with this buffer. In a
/// multi-stream pipeline (independent models on independent streams) that turns
/// every `zeroed` / `copy_from_host` into a full device barrier and serialises
/// unrelated work.
///
/// `cuStreamSynchronize(hStream)` is documented as waiting "until the device
/// has completed all operations in the stream specified by `hStream`" — one
/// stream's queue, not the context's. Passing a `NULL` handle
/// ([`CUstream::default`], a null pointer) selects the default stream, which
/// for a driver-API symbol resolved by name (`cuStreamSynchronize`, never the
/// `_ptsz` per-thread-default alias the CUDA *Runtime* substitutes under
/// `--default-stream per-thread`) is the legacy default stream — precisely the
/// stream the memset / DMA above was enqueued on.
///
/// That makes this strictly narrower than `cuCtxSynchronize` while closing the
/// exact same gap. The legacy stream's *implicit synchronisation* rule (an
/// operation **enqueued into** it first waits for all preceding operations in
/// the context's *blocking* streams) does not widen this call: that rule
/// governs enqueued operations, not a host-side wait — and in any case OxiCUDA
/// creates no blocking streams at all, so the set of streams it could pull in
/// is empty.
///
/// Proven on-device by `tests/legacy_stream_sync_gpu.rs`: the legacy stream is
/// still awaited (so the race stays closed), and an unrelated non-blocking
/// stream no longer is (the actual speedup).
#[inline]
fn sync_legacy_stream(api: &DriverApi) -> CudaResult<()> {
    // SAFETY: `cu_stream_synchronize` was resolved from the loaded driver, and
    // a `NULL` stream handle is the driver API's legacy default stream — always
    // a valid argument, no allocation of ours is referenced.
    oxicuda_driver::check(unsafe { (api.cu_stream_synchronize)(CUstream::default()) })
}

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
    /// Whether this buffer owns its allocation and must free it on drop.
    ///
    /// `true` for buffers created via [`DeviceBuffer::alloc`],
    /// [`DeviceBuffer::zeroed`], or [`DeviceBuffer::from_host`]; `false` for
    /// non-owning views created via [`DeviceBuffer::from_raw`], which borrow an
    /// externally-owned device pointer and must NOT free it on drop.
    owned: bool,
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
            owned: true,
            _phantom: PhantomData,
        })
    }

    /// Allocates a device buffer of `n` elements and zero-initialises every byte.
    ///
    /// This is equivalent to [`alloc`](Self::alloc) followed by a
    /// `cuMemsetD8_v2` call that writes `0` to every byte.
    ///
    /// The zero-fill is **fully completed on the device before this function
    /// returns**: `cuMemsetD8_v2` is issued on the legacy default stream and is
    /// asynchronous with respect to the host for device memory, so the returned
    /// buffer would otherwise not be guaranteed zeroed relative to work later
    /// submitted on a `CU_STREAM_NON_BLOCKING` stream (which does *not*
    /// implicitly synchronise with the default stream). Draining the legacy
    /// stream after the memset makes the "every byte is 0" postcondition hold
    /// for any consumer stream, closing a data race where a kernel on a
    /// non-blocking stream could read/overwrite this buffer concurrently with
    /// the pending zero-fill.
    ///
    /// The wait is scoped to the legacy default stream alone (see
    /// `sync_legacy_stream`), so unrelated work in flight on other streams is
    /// **not** waited on.
    ///
    /// # Errors
    ///
    /// Same as [`alloc`](Self::alloc), plus any error from `cuMemsetD8_v2` or
    /// the legacy-stream synchronise.
    pub fn zeroed(n: usize) -> CudaResult<Self> {
        let buf = Self::alloc(n)?;
        let api = try_driver()?;
        // SAFETY: the buffer was just allocated with the correct byte size.
        let rc = unsafe { (api.cu_memset_d8_v2)(buf.ptr, 0, buf.byte_size()) };
        oxicuda_driver::check(rc)?;
        // The non-async memset runs on the legacy default stream and is host
        // asynchronous for device memory; block until it has actually landed so
        // the buffer is zeroed with respect to every stream, not just the
        // default one. Waits on the legacy stream specifically -- the one the
        // memset was enqueued on -- not the whole context.
        sync_legacy_stream(api)?;
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

    /// Wraps an externally-owned device pointer in a non-owning
    /// [`DeviceBuffer`] view **without allocating**.
    ///
    /// The returned buffer points at the *existing* allocation described by
    /// `ptr` and `len`, and exposes the full [`DeviceBuffer`] API (copies,
    /// slicing, [`as_device_ptr`](Self::as_device_ptr), and use as a matrix
    /// operand in `oxicuda-blas`) over that memory.  Because the view does not
    /// own the allocation, its [`Drop`] is a no-op: it will **not** call
    /// `cuMemFree_v2`.  Ownership and the lifetime of the underlying memory
    /// remain entirely with the original owner (e.g. another CUDA library,
    /// `cudarc`, or a foreign allocator).
    ///
    /// This enables zero-copy interop: a consumer that already holds a
    /// resident device allocation can wrap it here and run OxiCUDA operations
    /// in place, with no host round-trip and no extra device allocation.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all of the following:
    ///
    /// * `ptr` is a valid CUDA device pointer into an allocation of at least
    ///   `len * size_of::<T>()` bytes, correctly aligned for `T`, and
    ///   associated with the CUDA context that subsequent OxiCUDA operations
    ///   run under.
    /// * The pointed-to memory contains a valid, initialised `[T; len]` (or is
    ///   only used as a write target before being read).
    /// * The underlying allocation **outlives** this `DeviceBuffer` view: the
    ///   original owner must not free, reallocate, or invalidate `ptr` while
    ///   this view (or any [`DeviceSlice`] borrowed from it) is alive.
    /// * No other live `DeviceBuffer` owns the same `ptr` (to avoid a
    ///   double-free) and aliasing rules are respected when the view is used
    ///   mutably (e.g. as a [`MatrixDescMut`](../oxicuda_blas/struct.MatrixDescMut.html)
    ///   output operand).
    ///
    /// A zero `len` is permitted (unlike [`alloc`](Self::alloc)) since no
    /// allocation is performed; a `ptr` of `0` is also permitted for a
    /// zero-length view, but pointer/length validity is the caller's
    /// responsibility.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use oxicuda_memory::DeviceBuffer;
    /// # use oxicuda_driver::ffi::CUdeviceptr;
    /// // `raw` is a device pointer owned elsewhere (e.g. obtained from another
    /// // CUDA library) pointing at `n` resident `f32` elements.
    /// # let raw: CUdeviceptr = 0;
    /// # let n: usize = 1024;
    /// // SAFETY: `raw` is valid for `n` f32s and outlives `view`.
    /// let view = unsafe { DeviceBuffer::<f32>::from_raw(raw, n) };
    /// // `view` can now be used with oxicuda-blas / copies; dropping it does
    /// // NOT free `raw`.
    /// assert_eq!(view.len(), n);
    /// ```
    #[must_use]
    pub unsafe fn from_raw(ptr: CUdeviceptr, len: usize) -> Self {
        Self {
            ptr,
            len,
            owned: false,
            _phantom: PhantomData,
        }
    }

    /// Copies data from a host slice into this device buffer (synchronous).
    ///
    /// The slice length must exactly match the buffer length.
    ///
    /// The upload is **fully landed on the device before this function
    /// returns**, and the wait is scoped to the legacy default stream alone
    /// (see `sync_legacy_stream`) rather than to the whole context.
    ///
    /// # Performance
    ///
    /// `src` is ordinary pageable host memory, so the driver must stage it
    /// through an internal DMA buffer. For a hot path that uploads the same
    /// tensor shape every frame, a reusable page-locked staging buffer
    /// ([`crate::StagingBuffer`]) is measurably faster.
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
        oxicuda_driver::check(rc)?;
        // `cuMemcpyHtoD_v2` is only "synchronous" in the sense that it returns
        // once `src` (pageable memory) has been staged into the driver's DMA
        // buffer -- the transfer to device memory itself completes later, on the
        // legacy default stream. Every OxiCUDA `Stream` is created with
        // `CU_STREAM_NON_BLOCKING`, which by definition does *not* implicitly
        // synchronise with the default stream, so a kernel or copy issued on one
        // can observe this buffer before the upload lands and silently read
        // zeros. Block until the DMA has completed, mirroring `zeroed` -- on the
        // legacy stream specifically, not the whole context.
        sync_legacy_stream(api)
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
        // Non-owning views (created via `from_raw`) borrow an externally-owned
        // allocation and must never free it.
        if !self.owned {
            return;
        }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A `from_raw` view must be marked non-owning so that `Drop` skips the
    /// `cuMemFree_v2` call. We construct over a dummy sentinel pointer; because
    /// the view is non-owning, dropping it performs no driver call and is safe
    /// even without a CUDA device present.
    #[test]
    fn from_raw_is_non_owning() {
        let sentinel: CUdeviceptr = 0xDEAD_BEEF;
        // SAFETY: this view is never dereferenced; we only inspect metadata and
        // rely on the non-owning Drop being a no-op.
        let view = unsafe { DeviceBuffer::<f32>::from_raw(sentinel, 16) };
        assert!(!view.owned, "from_raw must produce a non-owning buffer");
        assert_eq!(view.len(), 16);
        assert_eq!(view.as_device_ptr(), sentinel);
        assert_eq!(view.byte_size(), 16 * std::mem::size_of::<f32>());
        // Dropping a non-owning view must NOT touch the driver / free memory.
        // Reaching the end of scope here exercises that path without a GPU.
        drop(view);
    }

    /// A zero-length `from_raw` view is permitted (no allocation occurs) and is
    /// reported as empty.
    #[test]
    fn from_raw_zero_len_is_empty() {
        // SAFETY: zero-length, pointer never dereferenced; Drop is a no-op.
        let view = unsafe { DeviceBuffer::<u8>::from_raw(0, 0) };
        assert!(!view.owned);
        assert!(view.is_empty());
        assert_eq!(view.len(), 0);
        assert_eq!(view.byte_size(), 0);
    }

    /// Two non-owning views may share the same pointer without risking a
    /// double-free, because neither frees on drop. This models a consumer
    /// re-wrapping the same resident allocation.
    #[test]
    fn from_raw_aliasing_views_do_not_double_free() {
        let ptr: CUdeviceptr = 0x1000;
        // SAFETY: non-owning aliases, never dereferenced; both Drops are no-ops.
        let a = unsafe { DeviceBuffer::<f64>::from_raw(ptr, 8) };
        let b = unsafe { DeviceBuffer::<f64>::from_raw(ptr, 8) };
        assert!(!a.owned);
        assert!(!b.owned);
        assert_eq!(a.as_device_ptr(), b.as_device_ptr());
        drop(a);
        drop(b);
    }

    /// A real owning allocation created via `alloc` is marked `owned` so that
    /// its memory is freed on drop. This requires a CUDA device, so it is gated
    /// behind a runtime driver check and skipped (passing) when no GPU/driver
    /// is available — keeping the test green on macOS while still proving the
    /// owned-flag wiring on real hardware.
    #[test]
    fn alloc_is_owning_when_driver_available() {
        match DeviceBuffer::<f32>::alloc(32) {
            Ok(buf) => {
                assert!(buf.owned, "alloc must produce an owning buffer");
                assert_eq!(buf.len(), 32);
                // `buf` is dropped here and frees its allocation via the driver.
            }
            Err(_) => {
                // No CUDA driver/device on this host (e.g. macOS CI): the
                // owned-flag logic is covered by the non-GPU tests above.
            }
        }
    }
}
