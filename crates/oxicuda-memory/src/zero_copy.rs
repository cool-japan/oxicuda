//! Zero-copy (host-mapped) memory.
//!
//! Allows GPU kernels to directly access host memory without explicit
//! transfers.  Useful for small, frequently-updated data or when PCIe
//! bandwidth is acceptable.
//!
//! # How it works
//!
//! Zero-copy memory is allocated on the host using `cuMemAllocHost_v2`,
//! which allocates page-locked (pinned) memory that the CUDA driver maps
//! into the device's address space.  A corresponding device pointer is
//! obtained via `cuMemHostGetDevicePointer_v2`.  GPU reads and writes
//! traverse the PCIe bus on each access, so this is best suited for data
//! that is accessed infrequently or streamed sequentially.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_memory::zero_copy::MappedBuffer;
//!
//! oxicuda_driver::init()?;
//! let _ = oxicuda_driver::primary_context::PrimaryContext::retain(
//!     &oxicuda_driver::device::Device::get(0)?
//! )?;
//!
//! let mut buf = MappedBuffer::<f32>::alloc(256)?;
//! // Write from the host.
//! for (i, val) in buf.as_host_slice_mut().iter_mut().enumerate() {
//!     *val = i as f32;
//! }
//! // `buf.as_device_ptr()` can now be passed to a kernel.
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;

use oxicuda_driver::error::CudaResult;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_driver::loader::try_driver;

// ---------------------------------------------------------------------------
// MappedBuffer<T>
// ---------------------------------------------------------------------------

/// A host-allocated, device-mapped (zero-copy) memory buffer.
///
/// The host memory is page-locked and accessible from both CPU code and GPU
/// kernels.  GPU accesses traverse the PCIe bus, making this suitable for
/// small or infrequently-accessed data where the overhead of explicit
/// transfers is not justified.
///
/// The buffer is freed automatically on drop via `cuMemFreeHost`.
pub struct MappedBuffer<T: Copy> {
    /// Host pointer to the pinned allocation.
    host_ptr: *mut T,
    /// Corresponding device pointer for kernel access.
    device_ptr: CUdeviceptr,
    /// Number of `T` elements.
    len: usize,
    /// Marker for the element type.
    _phantom: PhantomData<T>,
}

// SAFETY: The page-locked host memory is not thread-local; both the host
// and device pointers are valid for Send/Sync if T is.
unsafe impl<T: Copy + Send> Send for MappedBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for MappedBuffer<T> {}

impl<T: Copy> MappedBuffer<T> {
    /// Allocates a zero-copy host-mapped buffer of `n` elements.
    ///
    /// The allocation uses `cuMemAllocHost_v2` (page-locked pinned memory)
    /// and retrieves the corresponding device pointer via
    /// `cuMemHostGetDevicePointer_v2`.  A CUDA context must be current on
    /// the calling thread.
    ///
    /// # Errors
    ///
    /// Returns a CUDA driver error if allocation or mapping fails.
    pub fn alloc(n: usize) -> CudaResult<Self> {
        let api = try_driver()?;
        let byte_size = n.saturating_mul(size_of::<T>());

        // Allocate page-locked host memory.
        let mut raw_ptr: *mut c_void = std::ptr::null_mut();
        oxicuda_driver::error::check(unsafe {
            (api.cu_mem_alloc_host_v2)(&mut raw_ptr, byte_size)
        })?;
        let host_ptr = raw_ptr.cast::<T>();

        // Obtain the device-side pointer for this pinned region.
        let mut device_ptr: CUdeviceptr = 0;
        let result = oxicuda_driver::error::check(unsafe {
            (api.cu_mem_host_get_device_pointer_v2)(&mut device_ptr, raw_ptr, 0)
        });
        if let Err(e) = result {
            // Free the pinned allocation before propagating the error.
            unsafe { (api.cu_mem_free_host)(raw_ptr) };
            return Err(e);
        }

        Ok(Self {
            host_ptr,
            device_ptr,
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

    /// Returns the byte size of this buffer.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Returns the raw device pointer for use in kernel parameters.
    #[inline]
    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.device_ptr
    }

    /// Returns a raw const pointer to the host-side data.
    #[inline]
    pub fn as_host_ptr(&self) -> *const T {
        self.host_ptr
    }

    /// Returns a raw mutable pointer to the host-side data.
    #[inline]
    pub fn as_host_ptr_mut(&mut self) -> *mut T {
        self.host_ptr
    }

    /// Returns a shared slice over the host-side data.
    ///
    /// # Safety
    ///
    /// The caller must ensure no concurrent GPU writes are in flight.
    pub fn as_host_slice(&self) -> &[T] {
        // SAFETY: host_ptr is valid for `len` elements allocated by cuMemAllocHost.
        unsafe { std::slice::from_raw_parts(self.host_ptr, self.len) }
    }

    /// Returns a mutable slice over the host-side data.
    ///
    /// # Safety
    ///
    /// The caller must ensure no concurrent GPU reads or writes are in flight.
    pub fn as_host_slice_mut(&mut self) -> &mut [T] {
        // SAFETY: host_ptr is valid for `len` elements allocated by cuMemAllocHost.
        unsafe { std::slice::from_raw_parts_mut(self.host_ptr, self.len) }
    }
}

impl<T: Copy> Drop for MappedBuffer<T> {
    fn drop(&mut self) {
        if self.host_ptr.is_null() {
            return;
        }
        if let Ok(api) = try_driver() {
            // SAFETY: host_ptr was allocated by cuMemAllocHost_v2 and has not
            // been freed yet (Drop is called at most once).
            unsafe { (api.cu_mem_free_host)(self.host_ptr.cast::<c_void>()) };
        }
    }
}
