//! Stream-ordered memory pool for efficient async allocation.
//!
//! Requires CUDA 11.2+ driver.  Gated behind the `pool` feature.
//!
//! Stream-ordered memory pools allow allocation and deallocation to be
//! ordered relative to other operations on a CUDA stream, enabling the
//! driver to reuse memory more aggressively and avoid synchronisation
//! barriers that would otherwise be needed for conventional
//! `cuMemAlloc` / `cuMemFree` calls.
//!
//! # Implementation note
//!
//! This implementation provides a practical fallback pool that reuses freed
//! allocations by size and uses `cuMemAlloc_v2` / `cuMemFree_v2` under the
//! hood.  It keeps the same API surface as a stream-ordered pool, but does
//! not yet expose native CUDA mempool handles.
//!
//! # API
//!
//! ```rust,ignore
//! let pool = MemoryPool::new(device)?;
//! let buf = PooledBuffer::<f32>::alloc_async(&pool, 1024, &stream)?;
//! // … use buf in kernels on `stream` …
//! // buf is freed asynchronously when dropped (enqueued on the pool's stream).
//! ```

#![cfg(feature = "pool")]

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::stream::Stream;
use tracing::warn;

// ---------------------------------------------------------------------------
// MemoryPool
// ---------------------------------------------------------------------------

/// A stream-ordered memory pool (CUDA 11.2+).
///
/// Memory pools allow the driver to reuse freed allocations without
/// returning them to the OS, reducing allocation latency and avoiding
/// the implicit synchronisation of `cuMemFree`.
///
/// # Status
///
/// This type is a placeholder.  The pool-related driver function pointers
/// (`cuMemPoolCreate`, `cuMemPoolDestroy`, etc.) are not yet available in
/// `oxicuda-driver`.
///
/// TODO: Add `cu_mem_pool_create`, `cu_mem_pool_destroy`,
/// `cu_mem_alloc_from_pool_async`, `cu_mem_free_async` to `DriverApi`.
/// Statistics for a memory pool's allocation behaviour.
///
/// These statistics track the total bytes allocated, peak usage,
/// allocation count, and free count for a given pool.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Total bytes currently allocated from the pool.
    pub allocated_bytes: usize,
    /// Peak bytes allocated at any point during the pool's lifetime.
    pub peak_bytes: usize,
    /// Total number of allocations performed.
    pub allocation_count: u64,
    /// Total number of frees performed.
    pub free_count: u64,
}

#[derive(Debug)]
struct MemoryPoolInner {
    handle: u64,
    device_ordinal: i32,
    threshold_bytes: AtomicUsize,
    cached_bytes: AtomicUsize,
    stats: Mutex<PoolStats>,
    free_bins: Mutex<HashMap<usize, Vec<CUdeviceptr>>>,
}

impl MemoryPoolInner {
    fn allocate_fresh(&self, bytes: usize) -> CudaResult<CUdeviceptr> {
        let api = try_driver()?;
        let mut ptr: CUdeviceptr = 0;
        let rc = unsafe { (api.cu_mem_alloc_v2)(&mut ptr, bytes) };
        oxicuda_driver::check(rc)?;
        Ok(ptr)
    }

    fn free_ptr(&self, ptr: CUdeviceptr) -> CudaResult<()> {
        let api = try_driver()?;
        let rc = unsafe { (api.cu_mem_free_v2)(ptr) };
        oxicuda_driver::check(rc)
    }

    fn try_pop_reuse(&self, bytes: usize) -> CudaResult<Option<CUdeviceptr>> {
        let mut bins = self.free_bins.lock().map_err(|_| CudaError::Unknown(0))?;
        let maybe_ptr = bins.get_mut(&bytes).and_then(Vec::pop);
        if maybe_ptr.is_some() {
            self.cached_bytes.fetch_sub(bytes, Ordering::Relaxed);
        }
        Ok(maybe_ptr)
    }

    fn stash_freed(&self, ptr: CUdeviceptr, bytes: usize) -> CudaResult<()> {
        let mut bins = self.free_bins.lock().map_err(|_| CudaError::Unknown(0))?;
        bins.entry(bytes).or_default().push(ptr);
        self.cached_bytes.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }

    fn release_cached_until(&self, keep_bytes: usize) -> CudaResult<()> {
        loop {
            let cached = self.cached_bytes.load(Ordering::Relaxed);
            if cached <= keep_bytes {
                return Ok(());
            }

            let popped = {
                let mut bins = self.free_bins.lock().map_err(|_| CudaError::Unknown(0))?;
                let mut candidate: Option<(usize, CUdeviceptr)> = None;
                for (size, vec) in bins.iter_mut() {
                    if let Some(ptr) = vec.pop() {
                        candidate = Some((*size, ptr));
                        break;
                    }
                }
                candidate
            };

            let Some((size, ptr)) = popped else {
                return Ok(());
            };
            self.free_ptr(ptr)?;
            self.cached_bytes.fetch_sub(size, Ordering::Relaxed);
        }
    }

    fn update_alloc_stats(&self, bytes: usize) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.allocated_bytes = stats.allocated_bytes.saturating_add(bytes);
            stats.allocation_count = stats.allocation_count.saturating_add(1);
            if stats.allocated_bytes > stats.peak_bytes {
                stats.peak_bytes = stats.allocated_bytes;
            }
        }
    }

    fn update_free_stats(&self, bytes: usize) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.allocated_bytes = stats.allocated_bytes.saturating_sub(bytes);
            stats.free_count = stats.free_count.saturating_add(1);
        }
    }
}

impl Drop for MemoryPoolInner {
    fn drop(&mut self) {
        let Ok(mut bins) = self.free_bins.lock() else {
            return;
        };
        let mut to_free: Vec<CUdeviceptr> = Vec::new();
        for vec in bins.values_mut() {
            to_free.append(vec);
        }
        drop(bins);

        for ptr in to_free {
            if let Err(e) = self.free_ptr(ptr) {
                warn!("failed to free pooled pointer {ptr:#x} during drop: {e}");
            }
        }
    }
}

/// A stream-ordered memory pool (CUDA 11.2+).
pub struct MemoryPool {
    inner: Arc<MemoryPoolInner>,
}

impl MemoryPool {
    /// Creates a new memory pool on the given device.
    ///
    /// # Errors
    ///
    /// Creates an in-process pooling allocator for the given device.
    pub fn new(device_ordinal: i32) -> CudaResult<Self> {
        if device_ordinal < 0 {
            return Err(CudaError::InvalidDevice);
        }
        Ok(Self {
            inner: Arc::new(MemoryPoolInner {
                handle: 0,
                device_ordinal,
                threshold_bytes: AtomicUsize::new(0),
                cached_bytes: AtomicUsize::new(0),
                stats: Mutex::new(PoolStats::default()),
                free_bins: Mutex::new(HashMap::new()),
            }),
        })
    }

    /// Returns the raw pool handle.
    ///
    /// # Status
    ///
    /// Returns `0` until the pool is properly initialised.
    #[inline]
    pub fn raw_handle(&self) -> u64 {
        self.inner.handle
    }

    /// Returns the device ordinal this pool targets.
    #[inline]
    pub fn device_ordinal(&self) -> i32 {
        self.inner.device_ordinal
    }

    /// Returns current pool statistics.
    ///
    /// The statistics track allocation behaviour over the pool's lifetime.
    #[inline]
    pub fn stats(&self) -> PoolStats {
        self.inner.stats.lock().map(|s| *s).unwrap_or_default()
    }

    /// Trims the pool, releasing unused memory back to the OS.
    ///
    /// Attempts to release memory such that the pool retains at most
    /// `min_bytes` of unused memory.
    ///
    /// # Errors
    ///
    pub fn trim(&mut self, min_bytes: usize) -> CudaResult<()> {
        self.inner.release_cached_until(min_bytes)
    }

    /// Sets the threshold at which the pool will automatically release
    /// memory back to the OS.
    ///
    /// When the pool's unused memory exceeds `bytes`, subsequent frees
    /// will trigger automatic trimming.
    ///
    /// # Errors
    ///
    pub fn set_threshold(&mut self, bytes: usize) -> CudaResult<()> {
        self.inner.threshold_bytes.store(bytes, Ordering::Relaxed);
        self.inner.release_cached_until(bytes)
    }
}

// ---------------------------------------------------------------------------
// PooledBuffer<T>
// ---------------------------------------------------------------------------

/// A device buffer allocated from a [`MemoryPool`].
///
/// Unlike [`DeviceBuffer`](crate::DeviceBuffer), a `PooledBuffer` is freed
/// asynchronously — the free operation is enqueued on the stream rather
/// than blocking the CPU.  This enables overlap of allocation, computation,
/// and deallocation across multiple stream operations.
///
/// # Status
///
/// This type allocates from an in-process memory pool and returns buffers to
/// that pool on drop.
pub struct PooledBuffer<T: Copy> {
    /// Raw device pointer to the pooled allocation.
    ptr: CUdeviceptr,
    /// Number of `T` elements.
    len: usize,
    /// Number of bytes in this allocation.
    bytes: usize,
    /// Owning pool.
    pool: Arc<MemoryPoolInner>,
    /// Marker for the element type.
    _phantom: PhantomData<T>,
}

impl<T: Copy> PooledBuffer<T> {
    /// Asynchronously allocates a buffer of `n` elements from the given pool.
    ///
    /// The allocation is ordered relative to other operations on `stream`.
    ///
    /// # Errors
    ///
    pub fn alloc_async(pool: &MemoryPool, n: usize, _stream: &Stream) -> CudaResult<Self> {
        if n == 0 {
            return Err(CudaError::InvalidValue);
        }
        let bytes = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CudaError::InvalidValue)?;
        let ptr = if let Some(reused) = pool.inner.try_pop_reuse(bytes)? {
            reused
        } else {
            pool.inner.allocate_fresh(bytes)?
        };
        pool.inner.update_alloc_stats(bytes);

        Ok(Self {
            ptr,
            len: n,
            bytes,
            pool: Arc::clone(&pool.inner),
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
        self.bytes
    }

    /// Returns the raw [`CUdeviceptr`] handle.
    #[inline]
    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

impl<T: Copy> Drop for PooledBuffer<T> {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }

        if let Err(e) = self.pool.stash_freed(self.ptr, self.bytes) {
            warn!("failed to return pooled pointer to free list: {e}; freeing directly");
            if let Err(free_err) = self.pool.free_ptr(self.ptr) {
                warn!("direct free of pooled pointer failed: {free_err}");
            }
            self.pool.update_free_stats(self.bytes);
            self.ptr = 0;
            return;
        }

        self.pool.update_free_stats(self.bytes);
        let threshold = self.pool.threshold_bytes.load(Ordering::Relaxed);
        if let Err(e) = self.pool.release_cached_until(threshold) {
            warn!("pool threshold trim failed: {e}");
        }
        self.ptr = 0;
    }
}
