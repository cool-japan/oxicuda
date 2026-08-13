//! Metal buffer manager — allocates, copies, and frees `metal::Buffer` objects
//! using the shared-memory storage mode so the CPU can read and write them
//! directly without a staging copy.
//!
//! All buffers are tracked by opaque `u64` handles (starting at 1) that mirror
//! the CUDA device-pointer model used by the rest of OxiCUDA.

#[cfg(target_os = "macos")]
use std::sync::atomic::Ordering;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, atomic::AtomicU64},
};

use crate::{
    device::MetalDevice,
    error::{MetalError, MetalResult},
};

// ─── Internal buffer record ──────────────────────────────────────────────────

/// Provenance of a tracked Metal buffer — decides whether [`MetalMemoryManager::free`]
/// (and manager drop) is allowed to release the underlying `MTLBuffer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BufferOwnership {
    /// Allocated by this manager via [`MetalMemoryManager::alloc`]. The manager
    /// holds the sole retain and releases it on `free`/drop.
    ///
    /// Only constructed on macOS — on other platforms [`MetalMemoryManager::alloc`]
    /// returns [`MetalError::UnsupportedPlatform`] instead of producing a buffer.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    Owned,
    /// Imported from an external caller (e.g. a consumer's buffer cache). The
    /// manager holds its **own independent retain** (taken at import time) and
    /// releases only *that* retain on `free`/drop — it never deallocates the
    /// caller's buffer, which the caller keeps alive via its own retain.
    External,
}

/// Bookkeeping entry for a single tracked Metal buffer.
pub(crate) struct MetalBufferInfo {
    /// The GPU-resident buffer. For [`BufferOwnership::Owned`] entries this is a
    /// freshly allocated shared-mode buffer; for [`BufferOwnership::External`]
    /// entries it is an independent retain of a caller-provided buffer.
    #[cfg(target_os = "macos")]
    pub(crate) buffer: metal::Buffer,
    /// Byte size of the allocation (the imported logical length for external
    /// buffers, which bounds `copy_to_device` / `copy_from_device`).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) size: u64,
    /// Whether this manager owns the allocation (and may release it) or is only
    /// borrowing an external buffer via its own retain.
    pub(crate) ownership: BufferOwnership,
    /// The buffer's Metal storage mode.
    ///
    /// [`MetalMemoryManager::alloc`] always produces `Shared` buffers, but
    /// [`MetalMemoryManager::import_external`] accepts a caller-provided
    /// `Managed` buffer (Intel / discrete Macs), where a host write is **not**
    /// visible to the GPU until it is published with `didModifyRange:`.
    #[cfg(target_os = "macos")]
    pub(crate) storage_mode: metal::MTLStorageMode,
}

/// Publish a host write to `buffer` so the GPU-side copy of a `Managed` buffer
/// sees it.
///
/// On `Shared` (and `Memoryless`/`Private`, which never reach here) storage this
/// is a no-op: Apple Silicon's unified memory needs no explicit publication. On
/// a `Managed` buffer — the only CPU-writable mode on Intel / discrete Macs —
/// skipping `didModifyRange:` leaves the GPU reading the stale mirror.
#[cfg(target_os = "macos")]
fn publish_host_write(buffer: &metal::Buffer, mode: metal::MTLStorageMode, len: usize) {
    if mode == metal::MTLStorageMode::Managed && len > 0 {
        buffer.did_modify_range(metal::NSRange::new(0, len as u64));
    }
}

// ─── Buffer reuse pool ───────────────────────────────────────────────────────

/// Default byte budget retained by the free-list pool.
///
/// Reached only if a workload frees that much without re-allocating it; every
/// pooled byte is memory the process keeps mapped, so the budget is what bounds
/// the pool's growth.
pub const DEFAULT_POOL_CAPACITY_BYTES: u64 = 128 * 1024 * 1024;

/// Maximum buffers retained in one exact-size bucket.
///
/// Stops a single hot size from consuming the whole byte budget and starving
/// every other size.
///
/// Off macOS no allocation ever succeeds, so no pool exists to bound.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const MAX_BUFFERS_PER_BUCKET: usize = 16;

/// A size-bucketed free list of released `MTLBuffer`s.
///
/// Buckets are keyed on the **exact** byte length, not a rounded-up power of
/// two. Rounding would let one allocation of `n` bytes serve a later request for
/// anything in `(n/2, n]`, but it also forces every allocation to physically
/// reserve its rounded size — up to 2× the requested memory, live, for the whole
/// lifetime of the handle. On a unified-memory Mac that inflation comes straight
/// out of the same pool the application is running in, so exact buckets are the
/// right trade: no waste at all, and they still capture the pattern that
/// actually dominates (same-shape temporaries allocated and freed every
/// iteration — `dispatch_reduce_flat`'s scratch buffer is one such caller inside
/// this very crate).
#[cfg(target_os = "macos")]
#[derive(Debug)]
struct BufferPool {
    buckets: HashMap<u64, Vec<metal::Buffer>>,
    pooled_bytes: u64,
    capacity_bytes: u64,
}

#[cfg(target_os = "macos")]
impl BufferPool {
    fn new(capacity_bytes: u64) -> Self {
        Self {
            buckets: HashMap::new(),
            pooled_bytes: 0,
            capacity_bytes,
        }
    }

    /// Remove and return a buffer of exactly `size` bytes, if one is pooled.
    fn take(&mut self, size: u64) -> Option<metal::Buffer> {
        let bucket = self.buckets.get_mut(&size)?;
        let buffer = bucket.pop()?;
        if bucket.is_empty() {
            self.buckets.remove(&size);
        }
        self.pooled_bytes = self.pooled_bytes.saturating_sub(size);
        Some(buffer)
    }

    /// Offer `buffer` (of exactly `size` bytes) to the pool.
    ///
    /// Returns `false` when the pool declined it — the caller then simply drops
    /// the buffer, releasing the memory as before.
    fn put(&mut self, size: u64, buffer: metal::Buffer) -> bool {
        if size == 0 || self.pooled_bytes + size > self.capacity_bytes {
            return false;
        }
        let bucket = self.buckets.entry(size).or_default();
        if bucket.len() >= MAX_BUFFERS_PER_BUCKET {
            return false;
        }
        bucket.push(buffer);
        self.pooled_bytes += size;
        true
    }

    /// Drop every pooled buffer, returning the number of bytes released.
    fn clear(&mut self) -> u64 {
        let released = self.pooled_bytes;
        self.buckets.clear();
        self.pooled_bytes = 0;
        released
    }
}

// ─── Memory manager ──────────────────────────────────────────────────────────

/// Manages a pool of Metal buffers, returning opaque `u64` handles.
///
/// Uses `MTLResourceOptions::StorageModeShared` so the same physical pages are
/// accessible from both CPU and GPU without explicit synchronisation — the same
/// model used by Metal's unified-memory architecture on Apple Silicon.
///
/// # Buffer reuse
///
/// [`free`](Self::free) returns an owned buffer to a bounded, size-bucketed free
/// list instead of releasing it, and [`alloc`](Self::alloc) reuses a pooled
/// buffer of the same exact size when one is available — `newBufferWithLength:`
/// is a driver call that maps fresh pages, and a workload that allocates and
/// frees the same shapes every iteration pays it on every iteration otherwise.
/// The pool holds at most [`DEFAULT_POOL_CAPACITY_BYTES`] (configurable via
/// [`with_pool_capacity`](Self::with_pool_capacity)) and can be emptied at any
/// time with [`trim_pool`](Self::trim_pool).
///
/// A reused buffer is **zeroed** before it is handed out, so `alloc`'s
/// observable behaviour is exactly what it was before pooling existed and no
/// caller can accidentally read a previous allocation's data.
///
/// All public methods take `&self` so the manager can be shared behind `Arc`.
pub struct MetalMemoryManager {
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    device: Arc<MetalDevice>,
    buffers: Mutex<HashMap<u64, MetalBufferInfo>>,
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    next_handle: AtomicU64,
    /// Free list of released owned buffers, keyed by exact byte size.
    #[cfg(target_os = "macos")]
    pool: Mutex<BufferPool>,
}

impl MetalMemoryManager {
    /// Create a new memory manager backed by `device`, with the default
    /// [`DEFAULT_POOL_CAPACITY_BYTES`] reuse budget.
    pub fn new(device: Arc<MetalDevice>) -> Self {
        Self::with_pool_capacity(device, DEFAULT_POOL_CAPACITY_BYTES)
    }

    /// Create a memory manager whose free-list pool retains at most
    /// `pool_capacity_bytes` of released buffers.
    ///
    /// Pass `0` to disable reuse entirely (every [`free`](Self::free) then
    /// releases its buffer immediately, the pre-pooling behaviour).
    pub fn with_pool_capacity(device: Arc<MetalDevice>, pool_capacity_bytes: u64) -> Self {
        // Off macOS there is no Metal and `alloc` never succeeds, so the pool
        // does not exist and the capacity is accepted-and-ignored.
        #[cfg(not(target_os = "macos"))]
        let _ = pool_capacity_bytes;
        Self {
            device,
            buffers: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
            #[cfg(target_os = "macos")]
            pool: Mutex::new(BufferPool::new(pool_capacity_bytes)),
        }
    }

    /// Bytes currently held in the reuse pool (allocated but not handed out).
    ///
    /// Always `0` off macOS, where no allocation ever succeeds.
    pub fn pooled_bytes(&self) -> u64 {
        #[cfg(target_os = "macos")]
        {
            self.pool.lock().map(|p| p.pooled_bytes).unwrap_or(0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            0
        }
    }

    /// Release every buffer held in the reuse pool, returning the byte count
    /// freed.
    ///
    /// Live handles are unaffected — only buffers already returned by
    /// [`free`](Self::free) sit in the pool. Call this to hand memory back to
    /// the system under pressure, or between phases with very different
    /// allocation shapes.
    ///
    /// # Errors
    /// [`MetalError::CommandBufferError`] if the pool mutex was poisoned by a
    /// panic in another thread.
    pub fn trim_pool(&self) -> MetalResult<u64> {
        #[cfg(target_os = "macos")]
        {
            let mut pool = self
                .pool
                .lock()
                .map_err(|_| MetalError::CommandBufferError("pool mutex poisoned".into()))?;
            Ok(pool.clear())
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(0)
        }
    }

    /// Lock the internal buffer map, returning a guard for buffer access.
    ///
    /// Used by the backend to bind Metal buffers to compute command encoders.
    #[cfg(target_os = "macos")]
    pub(crate) fn lock_buffers(
        &self,
    ) -> MetalResult<std::sync::MutexGuard<'_, HashMap<u64, MetalBufferInfo>>> {
        self.buffers
            .lock()
            .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))
    }

    /// Allocate `bytes` bytes of shared-mode device memory.
    ///
    /// Returns an opaque handle.  The caller must eventually call [`free`](Self::free).
    ///
    /// # Errors
    /// * [`MetalError::InvalidArgument`] if `bytes == 0` — `newBufferWithLength:`
    ///   returns nil for a zero length.
    /// * [`MetalError::OutOfMemory`] if `bytes` exceeds the device's
    ///   `maxBufferLength`, or if the driver could not satisfy the request.
    /// * [`MetalError::UnsupportedPlatform`] on non-macOS.
    pub fn alloc(&self, bytes: usize) -> MetalResult<u64> {
        #[cfg(target_os = "macos")]
        {
            // These guards MUST run *before* `new_buffer`: metal-rs builds its
            // `Buffer` with `NonNull::new_unchecked`, so a nil return from
            // `newBufferWithLength:options:` is already wrapped in a null
            // `NonNull` by the time it reaches us and the later
            // `contents()` memcpy would write to address 0. `newBufferWithLength:`
            // returns nil for length 0, for length > maxBufferLength, and on a
            // genuine allocation failure — the first two are checkable up front.
            if bytes == 0 {
                return Err(MetalError::InvalidArgument(
                    "allocation size must be > 0".into(),
                ));
            }
            let max_len = self.device.max_buffer_length();
            if bytes as u64 > max_len {
                tracing::warn!(
                    requested = bytes,
                    max_buffer_length = max_len,
                    "Metal allocation exceeds the device's maximum buffer length"
                );
                return Err(MetalError::OutOfMemory);
            }
            // Reuse an identically-sized buffer from the free list when one is
            // available, so a workload that cycles the same shapes pays the
            // driver's `newBufferWithLength:` cost once rather than per
            // iteration.
            let pooled = self
                .pool
                .lock()
                .map_err(|_| MetalError::CommandBufferError("pool mutex poisoned".into()))?
                .take(bytes as u64);
            let buffer = match pooled {
                Some(buffer) => {
                    // Zero the reused pages so `alloc` looks exactly as it did
                    // before pooling existed: no caller can observe a previous
                    // allocation's bytes. `length() == bytes` holds because the
                    // buckets are keyed on the exact size.
                    // SAFETY: the buffer came from this manager's own `alloc`,
                    // so it is a live Shared-storage buffer of exactly `bytes`
                    // bytes with a non-null `contents()`.
                    unsafe {
                        std::ptr::write_bytes(buffer.contents() as *mut u8, 0, bytes);
                    }
                    publish_host_write(&buffer, metal::MTLStorageMode::Shared, bytes);
                    tracing::trace!(bytes, "reused a pooled Metal buffer");
                    buffer
                }
                None => self
                    .device
                    .device
                    .new_buffer(bytes as u64, metal::MTLResourceOptions::StorageModeShared),
            };
            // Defence in depth for the third case (a genuine driver-side
            // failure), which cannot be predicted from the request alone:
            // messaging a nil Objective-C receiver yields 0 / null, so a nil
            // buffer reports length 0 and null contents. Reject it here rather
            // than handing the caller a handle that segfaults on first copy.
            if buffer.length() < bytes as u64 || buffer.contents().is_null() {
                tracing::warn!(
                    requested = bytes,
                    "Metal buffer allocation failed (driver returned a nil or short buffer)"
                );
                return Err(MetalError::OutOfMemory);
            }
            let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
            self.buffers
                .lock()
                .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?
                .insert(
                    handle,
                    MetalBufferInfo {
                        buffer,
                        size: bytes as u64,
                        ownership: BufferOwnership::Owned,
                        storage_mode: metal::MTLStorageMode::Shared,
                    },
                );
            Ok(handle)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = bytes;
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// Import an **externally owned** `metal::Buffer` and return a handle usable
    /// by every compute op (`gemm`, `copy_*`, …) without copying through the host.
    ///
    /// The manager takes its **own** retain on `buffer` (so the buffer stays
    /// alive while the handle is live) but is flagged
    /// `BufferOwnership::External`: [`free`](Self::free) and manager drop
    /// release only the manager's own retain and **never deallocate the caller's
    /// buffer**, which the caller keeps alive via its own retain (e.g. an
    /// `Arc<metal::Buffer>` in a buffer cache).
    ///
    /// `len_bytes` is the logical byte length the handle exposes; it bounds
    /// host copies via [`copy_to_device`](Self::copy_to_device) /
    /// [`copy_from_device`](Self::copy_from_device). It must not exceed the
    /// buffer's actual `length()`.
    ///
    /// # Errors
    /// * [`MetalError::UnsupportedPlatform`] on non-macOS.
    /// * [`MetalError::InvalidArgument`] if `len_bytes` exceeds the buffer's
    ///   physical length.
    #[cfg(target_os = "macos")]
    pub fn import_external(&self, buffer: &metal::Buffer, len_bytes: usize) -> MetalResult<u64> {
        let physical = buffer.length();
        if len_bytes as u64 > physical {
            return Err(MetalError::InvalidArgument(format!(
                "import len_bytes {len_bytes} exceeds buffer length {physical}"
            )));
        }
        // The host copy helpers (`copy_to_device` / `copy_from_device`)
        // dereference `buffer.contents()`, which Metal returns as NULL for
        // storage modes that are not CPU-accessible (Private / Memoryless).
        // Reject those at import time so a later host copy cannot deref null.
        let mode = buffer.storage_mode();
        if mode == metal::MTLStorageMode::Private || mode == metal::MTLStorageMode::Memoryless {
            return Err(MetalError::InvalidArgument(format!(
                "import_external requires a CPU-accessible buffer \
                 (Shared or Managed); got storage mode {mode:?}"
            )));
        }
        // Take an independent retain so the buffer survives for the handle's
        // lifetime regardless of what the caller does with its own reference.
        let retained = buffer.to_owned();
        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.buffers
            .lock()
            .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?
            .insert(
                handle,
                MetalBufferInfo {
                    buffer: retained,
                    size: len_bytes as u64,
                    ownership: BufferOwnership::External,
                    storage_mode: mode,
                },
            );
        Ok(handle)
    }

    /// Release the buffer associated with `handle`.
    ///
    /// For an `BufferOwnership::Owned` allocation this drops the manager's sole
    /// retain (freeing the GPU memory). For an `BufferOwnership::External`
    /// import this drops only the manager's own retain — the caller's buffer is
    /// untouched (the caller holds an independent retain). Unknown handles are
    /// silently ignored (idempotent free).
    pub fn free(&self, handle: u64) -> MetalResult<()> {
        let removed = self
            .buffers
            .lock()
            .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?
            .remove(&handle);
        if let Some(info) = removed {
            // The dropped `MetalBufferInfo` releases exactly one retain — the one
            // this manager holds. The `ownership` flag distinguishes whether that
            // retain was a sole allocation (owned) or an independent import retain
            // (external); in the external case the caller's buffer survives.
            match info.ownership {
                BufferOwnership::Owned => {
                    // Offer the buffer to the reuse pool instead of releasing it.
                    // ONLY owned buffers are poolable: an external import's
                    // memory belongs to the caller, who may free or mutate it the
                    // moment we hand the handle back, so retaining it in a pool
                    // and later serving it to an unrelated `alloc` would alias
                    // someone else's live memory.
                    #[cfg(target_os = "macos")]
                    {
                        let pooled = match self.pool.lock() {
                            Ok(mut pool) => pool.put(info.size, info.buffer.to_owned()),
                            Err(_) => {
                                tracing::warn!("pool mutex poisoned; releasing buffer directly");
                                false
                            }
                        };
                        tracing::trace!(handle, pooled, "released owned Metal buffer");
                    }
                    #[cfg(not(target_os = "macos"))]
                    tracing::trace!(handle, "freed owned Metal buffer");
                }
                BufferOwnership::External => {
                    tracing::trace!(
                        handle,
                        "released import retain for external Metal buffer (caller's buffer untouched)"
                    );
                }
            }
            // `info` (and its `metal::Buffer`, on macOS) drops at the end of this
            // block, releasing the manager's single retain. When the pool
            // accepted the buffer it holds an independent retain of its own, so
            // the memory survives for the next `alloc`.
        }
        Ok(())
    }

    /// Report whether `handle` refers to an imported (`BufferOwnership::External`)
    /// buffer (`Some(true)`), an owned allocation (`Some(false)`), or is unknown
    /// (`None`).
    ///
    /// Lets a consumer assert that a cache-backed buffer was registered as
    /// external (so [`free`](Self::free) will not deallocate it).
    pub fn is_external(&self, handle: u64) -> MetalResult<Option<bool>> {
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?;
        Ok(buffers
            .get(&handle)
            .map(|info| info.ownership == BufferOwnership::External))
    }

    /// Upload host bytes `src` into the device buffer identified by `handle`.
    ///
    /// Because the buffer uses shared storage, this is a direct CPU `memcpy`.
    pub fn copy_to_device(&self, handle: u64, src: &[u8]) -> MetalResult<()> {
        #[cfg(target_os = "macos")]
        {
            // Resolve the buffer and validate the size under the lock, then take
            // an independent retain so the `memcpy` runs *outside* the critical
            // section — concurrent alloc/free/copy on unrelated handles are not
            // serialised behind this transfer, and the retain guarantees the
            // buffer cannot be freed out from under the copy.
            let (buffer, mode) = {
                let buffers = self
                    .buffers
                    .lock()
                    .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?;
                let info = buffers.get(&handle).ok_or_else(|| {
                    MetalError::InvalidArgument(format!("unknown handle {handle}"))
                })?;
                if src.len() as u64 > info.size {
                    return Err(MetalError::InvalidArgument(format!(
                        "copy_to_device: src length {} exceeds buffer length {}",
                        src.len(),
                        info.size
                    )));
                }
                (info.buffer.to_owned(), info.storage_mode)
            };
            // SAFETY: Metal Shared/Managed buffers are CPU-accessible; `contents()`
            // returns a valid `*mut c_void` for the buffer's lifetime, which the
            // retain above extends across this copy. `src.len() <= buffer length`
            // was verified, so the write stays in bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    buffer.contents() as *mut u8,
                    src.len(),
                );
            }
            // Publish the write for Managed (Intel / discrete Mac) buffers; a
            // no-op for the Shared buffers `alloc` produces.
            publish_host_write(&buffer, mode, src.len());
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (handle, src);
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// Download device buffer `handle` into `dst`.
    ///
    /// Because the buffer uses shared storage, this is a direct CPU `memcpy`.
    pub fn copy_from_device(&self, dst: &mut [u8], handle: u64) -> MetalResult<()> {
        #[cfg(target_os = "macos")]
        {
            // Resolve + validate under the lock, retain, then copy outside the
            // critical section (see `copy_to_device` for the rationale).
            let buffer = {
                let buffers = self
                    .buffers
                    .lock()
                    .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?;
                let info = buffers.get(&handle).ok_or_else(|| {
                    MetalError::InvalidArgument(format!("unknown handle {handle}"))
                })?;
                if dst.len() as u64 > info.size {
                    return Err(MetalError::InvalidArgument(format!(
                        "copy_from_device: dst length {} exceeds buffer length {}",
                        dst.len(),
                        info.size
                    )));
                }
                info.buffer.to_owned()
            };
            // SAFETY: Metal Shared/Managed buffers are CPU-accessible; `contents()`
            // returns a valid `*const c_void` for the buffer's lifetime, extended
            // across the copy by the retain. `dst.len() <= buffer length` was
            // verified, so the read stays in bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.contents() as *const u8,
                    dst.as_mut_ptr(),
                    dst.len(),
                );
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (dst, handle);
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// Copy `len_bytes` from device buffer `src` to device buffer `dst`,
    /// **device-to-device** with no host round-trip.
    ///
    /// Both handles may be owned or imported, in any combination. For the
    /// shared-mode buffers this manager tracks, the copy is a direct
    /// CPU-visible `memcpy` between the two buffers' unified-memory pages, so it
    /// never stages through a host `Vec`. `len_bytes` is clamped to each
    /// buffer's tracked length.
    ///
    /// # Errors
    /// * [`MetalError::UnsupportedPlatform`] on non-macOS.
    /// * [`MetalError::InvalidArgument`] for an unknown `src`/`dst` handle, if
    ///   `src == dst`, or if `len_bytes` exceeds either buffer's length.
    pub fn copy_device_to_device(&self, dst: u64, src: u64, len_bytes: usize) -> MetalResult<()> {
        #[cfg(target_os = "macos")]
        {
            if src == dst {
                return Err(MetalError::InvalidArgument(
                    "copy_device_to_device requires distinct src and dst handles".into(),
                ));
            }
            // Resolve + validate under the lock, retain both buffers, then copy
            // outside the critical section (see `copy_to_device`).
            let (src_buf, dst_buf, dst_mode) = {
                let buffers = self
                    .buffers
                    .lock()
                    .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?;
                let src_info = buffers.get(&src).ok_or_else(|| {
                    MetalError::InvalidArgument(format!("unknown src handle {src}"))
                })?;
                let dst_info = buffers.get(&dst).ok_or_else(|| {
                    MetalError::InvalidArgument(format!("unknown dst handle {dst}"))
                })?;
                if len_bytes as u64 > src_info.size {
                    return Err(MetalError::InvalidArgument(format!(
                        "len_bytes {len_bytes} exceeds src length {}",
                        src_info.size
                    )));
                }
                if len_bytes as u64 > dst_info.size {
                    return Err(MetalError::InvalidArgument(format!(
                        "len_bytes {len_bytes} exceeds dst length {}",
                        dst_info.size
                    )));
                }
                (
                    src_info.buffer.to_owned(),
                    dst_info.buffer.to_owned(),
                    dst_info.storage_mode,
                )
            };
            let src_ptr = src_buf.contents() as *const u8;
            let dst_ptr = dst_buf.contents() as *mut u8;
            // Distinct handles can still alias the *same* physical MTLBuffer
            // (e.g. the same buffer imported twice), so `src_ptr == dst_ptr` or
            // overlapping byte ranges are possible. `copy_nonoverlapping`
            // requires disjoint regions; fall back to the overlap-safe `copy`
            // (memmove) whenever the ranges overlap.
            let src_addr = src_ptr as usize;
            let dst_addr = dst_ptr as usize;
            let overlaps = src_addr < dst_addr + len_bytes && dst_addr < src_addr + len_bytes;
            // SAFETY: both Shared/Managed buffers are CPU-accessible for their
            // full lifetime (extended across the copy by the retains above); the
            // validated `len_bytes` fits within both. `copy` is used for the
            // aliasing/overlapping case, `copy_nonoverlapping` only when the
            // regions are provably disjoint.
            unsafe {
                if overlaps {
                    std::ptr::copy(src_ptr, dst_ptr, len_bytes);
                } else {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len_bytes);
                }
            }
            // The copy runs on the CPU, so the destination needs the same
            // Managed-mode publication as a host upload.
            publish_host_write(&dst_buf, dst_mode, len_bytes);
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (dst, src, len_bytes);
            Err(MetalError::UnsupportedPlatform)
        }
    }
}

impl std::fmt::Debug for MetalMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.buffers.lock().map(|b| b.len()).unwrap_or(0);
        write!(f, "MetalMemoryManager(buffers={count})")
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    fn try_get_device() -> Option<Arc<MetalDevice>> {
        MetalDevice::new().ok().map(Arc::new)
    }

    #[test]
    fn alloc_and_free_requires_device() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        let h = mm.alloc(256).expect("alloc 256 bytes");
        assert!(h > 0);
        mm.free(h).expect("free");
        // Double-free is silently ignored.
        mm.free(h).expect("double-free is a no-op");
    }

    #[test]
    fn copy_roundtrip_requires_device() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);

        let src: Vec<u8> = (0u8..64).collect();
        let h = mm.alloc(src.len()).expect("alloc");
        mm.copy_to_device(h, &src).expect("copy_to_device");

        let mut dst = vec![0u8; src.len()];
        mm.copy_from_device(&mut dst, h).expect("copy_from_device");

        assert_eq!(src, dst);
        mm.free(h).expect("free");
    }

    #[test]
    fn unknown_handle_returns_error() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        let err = mm.copy_to_device(9999, b"hello").unwrap_err();
        assert!(matches!(err, MetalError::InvalidArgument(_)));
    }

    #[test]
    fn debug_impl_smoke() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        let s = format!("{mm:?}");
        assert!(s.contains("MetalMemoryManager"));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn alloc_zero_bytes_rejected() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        // `newBufferWithLength:0` returns nil, which metal-rs wraps in a null
        // NonNull — reject it before the call instead.
        let err = mm.alloc(0).expect_err("zero-byte alloc must fail");
        assert!(matches!(err, MetalError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn alloc_oversized_returns_out_of_memory_instead_of_crashing() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let max_len = dev.max_buffer_length();
        let mm = MetalMemoryManager::new(dev);
        // One byte past the device limit, and the pathological usize::MAX: both
        // used to yield a nil MTLBuffer wrapped in NonNull that segfaulted on
        // the next `contents()` memcpy.
        for request in [max_len as usize + 1, usize::MAX] {
            let err = mm
                .alloc(request)
                .expect_err("an oversized allocation must fail");
            assert!(matches!(err, MetalError::OutOfMemory), "{err:?}");
        }
        // A copy into the (never issued) handle must also fail cleanly.
        let mut dst = [0u8; 4];
        assert!(mm.copy_from_device(&mut dst, 1).is_err());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn allocated_buffers_are_shared_mode() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        let h = mm.alloc(64).expect("alloc");
        {
            let buffers = mm.lock_buffers().expect("lock");
            let info = buffers.get(&h).expect("tracked");
            assert_eq!(info.storage_mode, metal::MTLStorageMode::Shared);
            assert_eq!(info.buffer.storage_mode(), metal::MTLStorageMode::Shared);
        }
        mm.free(h).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn imported_buffer_records_its_storage_mode() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let raw = dev
            .device
            .new_buffer(128, metal::MTLResourceOptions::StorageModeShared);
        let mm = MetalMemoryManager::new(dev);
        let h = mm.import_external(&raw, 128).expect("import");
        {
            let buffers = mm.lock_buffers().expect("lock");
            let info = buffers.get(&h).expect("tracked");
            assert_eq!(info.storage_mode, raw.storage_mode());
        }
        // A host write through the imported handle still round-trips (the
        // Managed publication path is a no-op for Shared storage).
        mm.copy_to_device(h, &[7u8; 8]).expect("upload");
        let mut back = [0u8; 8];
        mm.copy_from_device(&mut back, h).expect("download");
        assert_eq!(back, [7u8; 8]);
        mm.free(h).expect("free");
    }

    // ─── Buffer reuse pool ───────────────────────────────────────────────────

    #[test]
    #[cfg(target_os = "macos")]
    fn free_pools_owned_buffers_and_alloc_reuses_them() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        assert_eq!(mm.pooled_bytes(), 0, "a fresh manager pools nothing");

        let h1 = mm.alloc(4096).expect("alloc");
        // The physical buffer identity is what must be reused; capture it
        // before the handle goes away.
        let first = {
            let buffers = mm.lock_buffers().expect("lock");
            buffers.get(&h1).expect("tracked").buffer.to_owned()
        };
        mm.free(h1).expect("free");
        assert_eq!(
            mm.pooled_bytes(),
            4096,
            "free must return the buffer to the pool"
        );

        let h2 = mm.alloc(4096).expect("alloc");
        let second = {
            let buffers = mm.lock_buffers().expect("lock");
            buffers.get(&h2).expect("tracked").buffer.to_owned()
        };
        assert_eq!(
            mm.pooled_bytes(),
            0,
            "the pooled buffer was handed back out"
        );
        assert!(
            std::ptr::eq(
                first.as_ref() as *const metal::BufferRef,
                second.as_ref() as *const metal::BufferRef
            ),
            "alloc must reuse the pooled buffer rather than asking the driver again"
        );
        // Handles are still distinct: reuse is about the physical buffer.
        assert_ne!(h1, h2);
        mm.free(h2).expect("free");
    }

    /// A pooled buffer must never leak the previous allocation's bytes.
    #[test]
    #[cfg(target_os = "macos")]
    fn reused_buffers_are_zeroed() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        let h1 = mm.alloc(64).expect("alloc");
        mm.copy_to_device(h1, &[0xABu8; 64]).expect("upload");
        mm.free(h1).expect("free");

        let h2 = mm.alloc(64).expect("alloc");
        let mut back = [0xFFu8; 64];
        mm.copy_from_device(&mut back, h2).expect("download");
        assert_eq!(back, [0u8; 64], "a reused buffer must be handed out zeroed");
        mm.free(h2).expect("free");
    }

    /// A different size must not be served from another size's bucket.
    #[test]
    #[cfg(target_os = "macos")]
    fn pool_buckets_are_exact_size() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        let h = mm.alloc(1024).expect("alloc");
        mm.free(h).expect("free");
        assert_eq!(mm.pooled_bytes(), 1024);

        // 512 must come from the driver, leaving the 1024 bucket untouched.
        let smaller = mm.alloc(512).expect("alloc");
        assert_eq!(
            mm.pooled_bytes(),
            1024,
            "a 512-byte request must not drain the 1024 bucket"
        );
        let info_len = {
            let buffers = mm.lock_buffers().expect("lock");
            buffers.get(&smaller).expect("tracked").buffer.length()
        };
        assert_eq!(info_len, 512, "an exact-size pool never over-allocates");
        mm.free(smaller).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn trim_pool_releases_everything_and_is_idempotent() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        for size in [256usize, 512, 256] {
            let h = mm.alloc(size).expect("alloc");
            mm.free(h).expect("free");
        }
        // Only two distinct sizes are held: the third allocation *reused* the
        // pooled 256-byte buffer rather than adding a second one, which is
        // exactly the behaviour the pool exists for.
        assert_eq!(mm.pooled_bytes(), 256 + 512);
        assert_eq!(mm.trim_pool().expect("trim"), 768);
        assert_eq!(mm.pooled_bytes(), 0);
        assert_eq!(mm.trim_pool().expect("trim"), 0, "trim is idempotent");
    }

    /// The pool must be bounded: past its byte budget, `free` releases directly.
    #[test]
    #[cfg(target_os = "macos")]
    fn pool_growth_is_bounded_by_its_byte_budget() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::with_pool_capacity(dev, 4096);
        let mut handles = Vec::new();
        for _ in 0..8 {
            handles.push(mm.alloc(1024).expect("alloc"));
        }
        for h in handles {
            mm.free(h).expect("free");
        }
        assert_eq!(
            mm.pooled_bytes(),
            4096,
            "the pool must stop accepting buffers at its budget"
        );
        assert_eq!(mm.trim_pool().expect("trim"), 4096);
    }

    /// A zero budget disables reuse entirely — the pre-pooling behaviour.
    #[test]
    #[cfg(target_os = "macos")]
    fn zero_capacity_disables_the_pool() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::with_pool_capacity(dev, 0);
        let h = mm.alloc(256).expect("alloc");
        mm.free(h).expect("free");
        assert_eq!(mm.pooled_bytes(), 0);
    }

    /// Imported buffers belong to the caller; pooling one would hand the
    /// caller's live memory to an unrelated later `alloc`.
    #[test]
    #[cfg(target_os = "macos")]
    fn external_imports_are_never_pooled() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let raw = dev
            .device
            .new_buffer(2048, metal::MTLResourceOptions::StorageModeShared);
        let mm = MetalMemoryManager::new(Arc::clone(&dev));
        let h = mm.import_external(&raw, 2048).expect("import");
        mm.free(h).expect("free");
        assert_eq!(
            mm.pooled_bytes(),
            0,
            "an external import must never enter the reuse pool"
        );
        // The caller's buffer is untouched and still usable.
        assert_eq!(raw.length(), 2048);
    }
}
