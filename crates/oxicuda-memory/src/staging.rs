//! Reusable pinned host staging buffer for hot-path H2D / D2H transfers.
//!
//! [`StagingBuffer`] owns **one** page-locked ([`PinnedBuffer`]) host allocation
//! that is allocated on first use, grown only when a larger transfer arrives,
//! and reused for every subsequent transfer. It exists for the workload where
//! per-call transfer overhead dominates: the same few tensor shapes moved
//! host↔device hundreds or thousands of times (a video inference pipeline
//! running the same ONNX models once per frame), where a fresh
//! `cuMemAllocHost_v2` per call would cost more than the transfer it enables.
//!
//! # Which API to use
//!
//! The crate offers two shapes of staged transfer, and the difference between
//! them is worth understanding because it is the difference between a 2.7x
//! speedup and a 2x *slowdown*:
//!
//! * [`upload_with`](StagingBuffer::upload_with) /
//!   [`download_into`](StagingBuffer::download_into) — the producer writes its
//!   data **directly into** page-locked memory (or reads results directly out
//!   of it). No intermediate host copy exists at all, so the DMA engine
//!   transfers straight from/to the caller's working memory. **This is the fast
//!   path, and it wins at every size** (measured 1.55x–2.67x, see below).
//!
//! * [`upload`](StagingBuffer::upload) / [`download`](StagingBuffer::download) —
//!   convenience wrappers taking an ordinary `&[T]` / `&mut [T]`. These must
//!   `memcpy` between the caller's pageable slice and the pinned buffer, and
//!   that extra host copy is **not free**: a single-threaded host memcpy runs
//!   at roughly 10 GB/s, while the CUDA driver's own pageable path pipelines
//!   its chunked staging copy *against* the DMA and so sustains more. Past
//!   about 1 MiB the wrapper therefore loses badly to just letting the driver
//!   do it. Rather than expose that as a footgun, these two methods
//!   **auto-select**: they stage through pinned memory only while the transfer
//!   is at or below [`StagingBuffer::auto_stage_max_bytes`], and hand larger
//!   transfers to the driver's pageable path. They are never slower than not
//!   using a `StagingBuffer` at all.
//!
//! # Measured (NVIDIA RTX A4000, driver 550.144.03, CUDA 12.4, sm_86)
//!
//! Median of 100–400 repetitions, release build. "copy-in" is the convenience
//! wrapper (pageable slice → pinned → DMA); "resident" is
//! [`upload_with`](StagingBuffer::upload_with) (producer fills pinned memory
//! directly).
//!
//! | transfer          | size    | pageable | copy-in       | resident      |
//! |-------------------|---------|----------|---------------|---------------|
//! | H2D 112×112×3 f32 | 150 KiB | 18.2 µs  | 15.4 µs 1.18x | 10.7 µs 1.71x |
//! | H2D 128×128×3 f32 | 196 KiB | 21.9 µs  | 18.7 µs 1.17x | 12.5 µs 1.75x |
//! | H2D 640×640×3 f32 | 4.7 MiB | 311 µs   | 681 µs  0.46x | 201 µs  1.55x |
//! | D2H 112×112×3 f32 | 150 KiB | 17.8 µs  | 14.2 µs 1.25x | 10.1 µs 1.77x |
//! | D2H 128×128×3 f32 | 196 KiB | 28.1 µs  | 18.2 µs 1.54x | 11.9 µs 2.37x |
//! | D2H 640×640×3 f32 | 4.7 MiB | 509 µs   | 676 µs  0.75x | 191 µs  2.67x |
//!
//! The `copy-in` column is exactly why the auto-select threshold exists.
//!
//! # Stream semantics
//!
//! Every method here is **synchronous from the caller's thread**: it returns
//! only once the transfer has fully landed. That deliberately matches
//! [`DeviceBuffer::copy_from_host`] / [`DeviceBuffer::copy_to_host`], so a
//! `StagingBuffer` is a drop-in replacement, and it is *required* for a reused
//! staging buffer — the next call overwrites the same pinned bytes, so the DMA
//! reading them must have finished.
//!
//! All transfers are ordered against `stream`: an upload is enqueued after work
//! already queued on `stream`, and a download observes the results of work
//! already queued on `stream`. This holds on both sides of the auto-select
//! threshold (the pageable fallback synchronises `stream` explicitly), so
//! changing tensor size can never silently change ordering.
//!
//! # Example
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use oxicuda_driver::{Context, Device, Stream};
//! # use oxicuda_memory::{DeviceBuffer, StagingBuffer};
//! # fn main() -> Result<(), oxicuda_driver::error::CudaError> {
//! # let dev = Device::get(0)?;
//! # let ctx = Arc::new(Context::new(&dev)?);
//! # let stream = Stream::new(&ctx)?;
//! let mut staging = StagingBuffer::new();
//! let mut d_input = DeviceBuffer::<f32>::alloc(3 * 640 * 640)?;
//!
//! // Per frame: write preprocessed pixels straight into pinned memory.
//! staging.upload_with(&mut d_input, 3 * 640 * 640, &stream, |dst: &mut [f32]| {
//!     for (i, v) in dst.iter_mut().enumerate() {
//!         *v = i as f32; // real code: normalised pixel data
//!     }
//! })?;
//! # Ok(())
//! # }
//! ```

use std::ffi::c_void;

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::stream::Stream;

use crate::device_buffer::DeviceBuffer;
use crate::host_buffer::PinnedBuffer;

// ---------------------------------------------------------------------------
// StagingPod
// ---------------------------------------------------------------------------

/// Types for which **every** bit pattern is a valid value.
///
/// [`StagingBuffer`] hands out `&mut [T]` / `&[T]` views over page-locked bytes
/// that were either zero-filled at allocation time or last written by a DMA
/// from the device. Materialising such a view is only sound when no bit pattern
/// those bytes could hold is invalid for `T` — which rules out types with
/// niches (`bool`, `char`, `NonZeroU32`, enums, references) even though they
/// are `Copy`.
///
/// # Safety
///
/// Implementors must guarantee that `T` has no invalid bit patterns, no padding
/// whose contents could be observed as uninitialised, and no interior
/// pointers/references that would be meaningless after a device round trip.
/// Numeric primitives and `#[repr(transparent)]` wrappers over them qualify.
pub unsafe trait StagingPod: Copy {}

macro_rules! impl_staging_pod {
    ($($t:ty),* $(,)?) => {
        $(
            // SAFETY: every bit pattern of this primitive numeric type is a
            // valid value of the type, and it has no padding or interior
            // pointers.
            unsafe impl StagingPod for $t {}
        )*
    };
}

impl_staging_pod!(
    u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64
);

// SAFETY: `half::f16` is `#[repr(transparent)]` over `u16`; every 16-bit
// pattern denotes a valid IEEE half (some are NaN, which is still a value).
#[cfg(feature = "half")]
unsafe impl StagingPod for half::f16 {}

// SAFETY: `half::bf16` is `#[repr(transparent)]` over `u16`; every 16-bit
// pattern denotes a valid bfloat16.
#[cfg(feature = "half")]
unsafe impl StagingPod for half::bf16 {}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default value of [`StagingBuffer::auto_stage_max_bytes`]: 512 KiB.
///
/// At or below this size, copying a pageable slice into the pinned buffer and
/// DMA-ing from there beats handing the slice to the driver; above it, the
/// driver's pipelined pageable path wins. 512 KiB is the largest measured size
/// at which staging wins in **both** directions on the reference device (H2D
/// 1.12x, D2H 1.57x); at 1 MiB H2D has already turned negative (0.91x). See the
/// module-level table.
///
/// This is a heuristic calibrated on one machine, not a hardware invariant —
/// [`StagingBuffer::set_auto_stage_max_bytes`] overrides it (`0` disables
/// staging for the slice wrappers entirely, `usize::MAX` always stages).
pub const DEFAULT_AUTO_STAGE_MAX_BYTES: usize = 512 * 1024;

/// Capacity granularity: growth requests are rounded up to a multiple of this
/// (64 KiB).
///
/// `cuMemAllocHost_v2` pins whole pages regardless, and rounding up absorbs
/// small shape jitter (e.g. a detector whose input varies by a few rows)
/// without re-pinning — a re-pin is a millisecond-scale driver call, far more
/// expensive than the bounded ≤64 KiB of slack it avoids.
const CAPACITY_GRANULARITY: usize = 64 * 1024;

// ---------------------------------------------------------------------------
// StagingStats
// ---------------------------------------------------------------------------

/// Counters describing how a [`StagingBuffer`] has been used.
///
/// Chiefly useful for asserting in tests (and in production tracing) that the
/// buffer really is being *reused* rather than silently re-pinned every call —
/// the entire point of the type. A healthy hot loop shows `allocations == 1`
/// and a large `staged_transfers`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StagingStats {
    /// Number of times a pinned host allocation was performed (initial + grows).
    pub allocations: u64,
    /// Transfers that were staged through the pinned buffer.
    pub staged_transfers: u64,
    /// Transfers routed straight to the driver's pageable path because they
    /// exceeded [`StagingBuffer::auto_stage_max_bytes`].
    pub direct_transfers: u64,
    /// Total bytes uploaded host→device through this buffer (both paths).
    pub bytes_uploaded: u64,
    /// Total bytes downloaded device→host through this buffer (both paths).
    pub bytes_downloaded: u64,
}

// ---------------------------------------------------------------------------
// StagingBuffer
// ---------------------------------------------------------------------------

/// A grow-on-demand, reused page-locked host buffer used to stage transfers.
///
/// See the [module documentation](self) for the performance model and for which
/// method to reach for.
pub struct StagingBuffer {
    /// The pinned allocation, absent until the first transfer sizes it.
    pinned: Option<PinnedBuffer<u8>>,
    /// Transfers larger than this bypass staging (see
    /// [`DEFAULT_AUTO_STAGE_MAX_BYTES`]).
    auto_stage_max_bytes: usize,
    /// Usage counters.
    stats: StagingStats,
}

impl Default for StagingBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl StagingBuffer {
    /// Creates an empty staging buffer. **No** host memory is pinned until the
    /// first transfer (or an explicit [`reserve`](Self::reserve)).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pinned: None,
            auto_stage_max_bytes: DEFAULT_AUTO_STAGE_MAX_BYTES,
            stats: StagingStats {
                allocations: 0,
                staged_transfers: 0,
                direct_transfers: 0,
                bytes_uploaded: 0,
                bytes_downloaded: 0,
            },
        }
    }

    /// Creates a staging buffer with `bytes` of pinned host memory already
    /// allocated.
    ///
    /// Pre-sizing at start-up (to the largest tensor a pipeline will move) keeps
    /// the millisecond-scale `cuMemAllocHost_v2` out of the steady-state loop
    /// entirely.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `bytes` is zero.
    /// * Other driver errors from `cuMemAllocHost_v2`.
    pub fn with_capacity(bytes: usize) -> CudaResult<Self> {
        let mut this = Self::new();
        this.reserve(bytes)?;
        Ok(this)
    }

    /// Returns the current pinned capacity in bytes (`0` before first use).
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.pinned.as_ref().map_or(0, PinnedBuffer::len)
    }

    /// Returns the usage counters. See [`StagingStats`].
    #[inline]
    #[must_use]
    pub fn stats(&self) -> StagingStats {
        self.stats
    }

    /// Returns the size above which [`upload`](Self::upload) /
    /// [`download`](Self::download) bypass staging.
    #[inline]
    #[must_use]
    pub fn auto_stage_max_bytes(&self) -> usize {
        self.auto_stage_max_bytes
    }

    /// Overrides the auto-select threshold (see
    /// [`DEFAULT_AUTO_STAGE_MAX_BYTES`] for how it was calibrated).
    ///
    /// Affects only the slice-taking wrappers; [`upload_with`](Self::upload_with)
    /// and [`download_into`](Self::download_into) always stage, because for
    /// those the pinned buffer *is* the caller's working memory and there is no
    /// extra copy to regret.
    #[inline]
    pub fn set_auto_stage_max_bytes(&mut self, bytes: usize) {
        self.auto_stage_max_bytes = bytes;
    }

    /// Ensures at least `bytes` of pinned host memory are available.
    ///
    /// Grow-only: a request smaller than the current capacity is a no-op, so a
    /// pipeline cycling through several tensor shapes settles at the high-water
    /// mark and never re-pins again. Growth rounds up to
    /// `CAPACITY_GRANULARITY`.
    ///
    /// Any previously staged contents are discarded when the buffer grows.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `bytes` is zero, or if rounding
    ///   overflows `usize`.
    /// * Other driver errors from `cuMemAllocHost_v2`.
    pub fn reserve(&mut self, bytes: usize) -> CudaResult<()> {
        if bytes == 0 {
            return Err(CudaError::InvalidValue);
        }
        if self.capacity() >= bytes {
            return Ok(());
        }
        let rounded = bytes
            .checked_next_multiple_of(CAPACITY_GRANULARITY)
            .ok_or(CudaError::InvalidValue)?;
        // Drop the old allocation *before* pinning the new one: page-locked
        // memory is a scarce, system-wide resource, and holding two copies of a
        // large buffer at once could fail an allocation that would otherwise
        // succeed.
        self.pinned = None;
        self.pinned = Some(PinnedBuffer::<u8>::alloc(rounded)?);
        self.stats.allocations += 1;
        Ok(())
    }

    /// Releases the pinned allocation, returning capacity to `0`.
    ///
    /// Counters are preserved. The next transfer re-pins.
    pub fn shrink_to_fit(&mut self) {
        self.pinned = None;
    }

    // -- Fast path: fill / read page-locked memory in place -------------------

    /// Uploads `n` elements to `dst`, letting `fill` write them **directly into
    /// page-locked memory**.
    ///
    /// This is the fastest host→device path this crate offers (measured
    /// 1.55x–1.75x over [`DeviceBuffer::copy_from_host`] across 150 KiB–4.7 MiB)
    /// because the bytes `fill` writes are the exact bytes the DMA engine reads:
    /// there is no pageable source slice and no driver bounce buffer.
    ///
    /// `fill` receives a mutable slice of exactly `n` elements and is expected
    /// to write **all** of them; anything left untouched keeps whatever the
    /// previous transfer through this buffer left there (or zeroes, on a
    /// freshly pinned allocation), and that content is uploaded as-is.
    ///
    /// The copy is enqueued on `stream` — so it is ordered after work already
    /// queued there — and this call returns only once it has landed.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `n` is zero, if `n != dst.len()`, if the
    ///   byte size overflows, or if the pinned allocation is not aligned for `T`.
    /// * Other driver errors from `cuMemAllocHost_v2` or `cuMemcpyHtoDAsync_v2`.
    pub fn upload_with<T, F>(
        &mut self,
        dst: &mut DeviceBuffer<T>,
        n: usize,
        stream: &Stream,
        fill: F,
    ) -> CudaResult<()>
    where
        T: StagingPod,
        F: FnOnce(&mut [T]),
    {
        let byte_size = Self::check_extent::<T>(n, dst.len())?;
        self.reserve(byte_size)?;
        fill(self.typed_mut::<T>(n)?);
        self.enqueue_htod(dst.as_device_ptr(), byte_size, stream)?;
        stream.synchronize()?;
        self.stats.staged_transfers += 1;
        self.stats.bytes_uploaded += byte_size as u64;
        Ok(())
    }

    /// Downloads `n` elements from `src` into page-locked memory and returns a
    /// borrowed view of them, with **no copy out**.
    ///
    /// This is the fastest device→host path this crate offers (measured
    /// 1.77x–2.67x over [`DeviceBuffer::copy_to_host`]): the DMA engine writes
    /// straight into the memory the caller then reads. The returned slice
    /// borrows `self` and stays valid until the next call that touches the
    /// staging buffer.
    ///
    /// The copy is enqueued on `stream`, so it observes the results of work
    /// already queued there; this call returns only once the data has landed and
    /// is safe to read.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `n` is zero, if `n != src.len()`, if the
    ///   byte size overflows, or if the pinned allocation is not aligned for `T`.
    /// * Other driver errors from `cuMemAllocHost_v2` or `cuMemcpyDtoHAsync_v2`.
    pub fn download_into<T: StagingPod>(
        &mut self,
        src: &DeviceBuffer<T>,
        n: usize,
        stream: &Stream,
    ) -> CudaResult<&[T]> {
        let byte_size = Self::check_extent::<T>(n, src.len())?;
        self.reserve(byte_size)?;
        self.enqueue_dtoh(src.as_device_ptr(), byte_size, stream)?;
        stream.synchronize()?;
        self.stats.staged_transfers += 1;
        self.stats.bytes_downloaded += byte_size as u64;
        Ok(self.typed_mut::<T>(n)?)
    }

    // -- Convenience path: ordinary slices, size-aware ------------------------

    /// Uploads `src` into `dst`, staging through pinned memory when that is
    /// actually faster.
    ///
    /// A drop-in replacement for [`DeviceBuffer::copy_from_host`] with the same
    /// postcondition (the data has landed on the device when this returns) that
    /// is never slower: transfers up to
    /// [`auto_stage_max_bytes`](Self::auto_stage_max_bytes) go through the
    /// pinned buffer, larger ones go straight to the driver's pageable path,
    /// which pipelines better than a host memcpy into pinned memory can (see the
    /// module table).
    ///
    /// If you control how `src` is produced, prefer
    /// [`upload_with`](Self::upload_with) — writing the data into pinned memory
    /// in the first place removes this method's memcpy and wins at every size.
    ///
    /// Ordered against `stream` in both paths.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `src` is empty or `src.len() != dst.len()`.
    /// * Other driver errors from the allocation or copy.
    pub fn upload<T: StagingPod>(
        &mut self,
        dst: &mut DeviceBuffer<T>,
        src: &[T],
        stream: &Stream,
    ) -> CudaResult<()> {
        let byte_size = Self::check_extent::<T>(src.len(), dst.len())?;
        if byte_size > self.auto_stage_max_bytes {
            // Keep ordering identical to the staged path: the staged upload
            // would have been enqueued behind whatever is already on `stream`,
            // so drain it before a legacy-stream copy overtakes that work.
            stream.synchronize()?;
            dst.copy_from_host(src)?;
            self.stats.direct_transfers += 1;
            self.stats.bytes_uploaded += byte_size as u64;
            return Ok(());
        }
        let n = src.len();
        self.upload_with(dst, n, stream, |staged: &mut [T]| {
            staged.copy_from_slice(src);
        })
    }

    /// Downloads `src` into `dst`, staging through pinned memory when that is
    /// actually faster.
    ///
    /// A drop-in replacement for [`DeviceBuffer::copy_to_host`] that is never
    /// slower; the mirror of [`upload`](Self::upload), including the auto-select
    /// threshold. Prefer [`download_into`](Self::download_into) when the caller
    /// can consume the results in place.
    ///
    /// Unlike a bare [`DeviceBuffer::copy_to_host`], **both** paths here are
    /// ordered against `stream`, so results produced by kernels on `stream` are
    /// guaranteed visible without the caller synchronising first.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `dst` is empty or `dst.len() != src.len()`.
    /// * Other driver errors from the allocation or copy.
    pub fn download<T: StagingPod>(
        &mut self,
        dst: &mut [T],
        src: &DeviceBuffer<T>,
        stream: &Stream,
    ) -> CudaResult<()> {
        let byte_size = Self::check_extent::<T>(dst.len(), src.len())?;
        if byte_size > self.auto_stage_max_bytes {
            // `copy_to_host` runs on the legacy stream, which does NOT wait for
            // a `CU_STREAM_NON_BLOCKING` stream; drain `stream` first so this
            // path observes the same work the staged path would have.
            stream.synchronize()?;
            src.copy_to_host(dst)?;
            self.stats.direct_transfers += 1;
            self.stats.bytes_downloaded += byte_size as u64;
            return Ok(());
        }
        let n = dst.len();
        let staged = self.download_into(src, n, stream)?;
        dst.copy_from_slice(staged);
        Ok(())
    }

    // -- Internals -----------------------------------------------------------

    /// Validates a transfer extent and returns its size in bytes.
    fn check_extent<T>(n: usize, device_len: usize) -> CudaResult<usize> {
        if n == 0 || n != device_len {
            return Err(CudaError::InvalidValue);
        }
        n.checked_mul(std::mem::size_of::<T>())
            .ok_or(CudaError::InvalidValue)
    }

    /// Reinterprets the first `n` elements of the pinned allocation as `&mut [T]`.
    ///
    /// Returns [`CudaError::InvalidValue`] rather than risking undefined
    /// behaviour if the allocation is somehow under-aligned for `T`.
    /// `cuMemAllocHost_v2` returns page-aligned memory in practice, so this
    /// check never fires for any realistic `T` — it is a guard, not a code path.
    fn typed_mut<T: StagingPod>(&mut self, n: usize) -> CudaResult<&mut [T]> {
        let buf = self.pinned.as_mut().ok_or(CudaError::InvalidValue)?;
        let byte_size = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CudaError::InvalidValue)?;
        if byte_size > buf.len() {
            return Err(CudaError::InvalidValue);
        }
        let ptr = buf.as_mut_ptr();
        if !ptr.cast::<T>().is_aligned() {
            return Err(CudaError::InvalidValue);
        }
        // SAFETY: `ptr` addresses `buf.len() >= byte_size` bytes of live pinned
        // host memory that we hold `&mut` to for the returned lifetime; the
        // pointer is aligned for `T` (checked above); and `T: StagingPod`
        // guarantees every bit pattern of those bytes -- zero-filled by
        // `PinnedBuffer::alloc`, or written by a previous fill/DMA -- is a valid
        // `T`, so no invalid value can be materialised.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr.cast::<T>(), n) })
    }

    /// Enqueues `byte_size` bytes from the pinned buffer to `dst_ptr` on `stream`.
    fn enqueue_htod(
        &self,
        dst_ptr: oxicuda_driver::ffi::CUdeviceptr,
        byte_size: usize,
        stream: &Stream,
    ) -> CudaResult<()> {
        let buf = self.pinned.as_ref().ok_or(CudaError::InvalidValue)?;
        if byte_size > buf.len() {
            return Err(CudaError::InvalidValue);
        }
        let api = try_driver()?;
        // SAFETY: `buf` is page-locked host memory holding at least `byte_size`
        // valid bytes, `dst_ptr` is a live device allocation of at least that
        // size (its `DeviceBuffer` length was checked against `n`), and the
        // caller synchronises `stream` before the pinned bytes are reused.
        let rc = unsafe {
            (api.cu_memcpy_htod_async_v2)(
                dst_ptr,
                buf.as_ptr().cast::<c_void>(),
                byte_size,
                stream.raw(),
            )
        };
        oxicuda_driver::check(rc)
    }

    /// Enqueues `byte_size` bytes from `src_ptr` into the pinned buffer on `stream`.
    fn enqueue_dtoh(
        &mut self,
        src_ptr: oxicuda_driver::ffi::CUdeviceptr,
        byte_size: usize,
        stream: &Stream,
    ) -> CudaResult<()> {
        let buf = self.pinned.as_mut().ok_or(CudaError::InvalidValue)?;
        if byte_size > buf.len() {
            return Err(CudaError::InvalidValue);
        }
        let api = try_driver()?;
        // SAFETY: `buf` is page-locked host memory with room for `byte_size`
        // bytes and is exclusively borrowed here, `src_ptr` is a live device
        // allocation of at least that size, and the caller synchronises
        // `stream` before reading the result.
        let rc = unsafe {
            (api.cu_memcpy_dtoh_async_v2)(
                buf.as_mut_ptr().cast::<c_void>(),
                src_ptr,
                byte_size,
                stream.raw(),
            )
        };
        oxicuda_driver::check(rc)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_pins_nothing() {
        let staging = StagingBuffer::new();
        assert_eq!(staging.capacity(), 0);
        assert_eq!(staging.stats().allocations, 0);
        assert_eq!(staging.auto_stage_max_bytes(), DEFAULT_AUTO_STAGE_MAX_BYTES);
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(
            StagingBuffer::default().auto_stage_max_bytes(),
            StagingBuffer::new().auto_stage_max_bytes()
        );
        assert_eq!(StagingBuffer::default().capacity(), 0);
    }

    #[test]
    fn threshold_is_overridable() {
        let mut staging = StagingBuffer::new();
        staging.set_auto_stage_max_bytes(0);
        assert_eq!(staging.auto_stage_max_bytes(), 0);
        staging.set_auto_stage_max_bytes(usize::MAX);
        assert_eq!(staging.auto_stage_max_bytes(), usize::MAX);
    }

    #[test]
    fn reserve_rejects_zero() {
        let mut staging = StagingBuffer::new();
        assert_eq!(staging.reserve(0), Err(CudaError::InvalidValue));
    }

    #[test]
    fn check_extent_rejects_mismatch_and_zero() {
        assert_eq!(
            StagingBuffer::check_extent::<f32>(4, 5),
            Err(CudaError::InvalidValue)
        );
        assert_eq!(
            StagingBuffer::check_extent::<f32>(0, 0),
            Err(CudaError::InvalidValue)
        );
        assert_eq!(StagingBuffer::check_extent::<f32>(4, 4), Ok(16));
    }

    #[test]
    fn check_extent_rejects_byte_overflow() {
        assert_eq!(
            StagingBuffer::check_extent::<u64>(usize::MAX, usize::MAX),
            Err(CudaError::InvalidValue)
        );
    }

    /// Growth must round up to the granularity so shape jitter does not re-pin.
    #[test]
    fn capacity_granularity_is_a_power_of_two_page_multiple() {
        assert!(CAPACITY_GRANULARITY.is_power_of_two());
        assert_eq!(CAPACITY_GRANULARITY % 4096, 0);
    }

    #[test]
    fn stats_start_zeroed() {
        assert_eq!(StagingBuffer::new().stats(), StagingStats::default());
    }

    #[cfg(feature = "gpu-tests")]
    mod gpu_tests {
        use super::*;
        use std::sync::Arc;

        /// A live context + stream, or `None` when no GPU is present.
        fn fixture() -> Option<(Arc<oxicuda_driver::Context>, Stream)> {
            oxicuda_driver::init().ok()?;
            if oxicuda_driver::Device::count().ok()? == 0 {
                return None;
            }
            let dev = oxicuda_driver::Device::get(0).ok()?;
            let ctx = Arc::new(oxicuda_driver::Context::new(&dev).ok()?);
            let stream = Stream::new(&ctx).ok()?;
            Some((ctx, stream))
        }

        /// The fill-in-place upload and read-in-place download must round-trip
        /// exactly -- this is the fast path, so it is the one that most needs a
        /// correctness proof.
        #[test]
        fn upload_with_download_into_round_trip() {
            let Some((_ctx, stream)) = fixture() else {
                eprintln!("skipping: no CUDA driver/device");
                return;
            };
            let n = 128 * 128 * 3;
            let mut staging = StagingBuffer::new();
            let mut d = DeviceBuffer::<f32>::alloc(n).expect("alloc");

            staging
                .upload_with(&mut d, n, &stream, |dst: &mut [f32]| {
                    for (i, v) in dst.iter_mut().enumerate() {
                        *v = (i % 977) as f32 * 0.25;
                    }
                })
                .expect("upload_with");

            let got = staging
                .download_into(&d, n, &stream)
                .expect("download_into");
            for (i, &v) in got.iter().enumerate() {
                assert_eq!(v, (i % 977) as f32 * 0.25, "element {i}");
            }
        }

        /// The buffer must be allocated once and then *reused*: that is the
        /// whole reason the type exists, so assert it rather than assume it.
        #[test]
        fn repeated_transfers_reuse_one_allocation() {
            let Some((_ctx, stream)) = fixture() else {
                eprintln!("skipping: no CUDA driver/device");
                return;
            };
            let n = 4096;
            let mut staging = StagingBuffer::new();
            let mut d = DeviceBuffer::<f32>::alloc(n).expect("alloc");
            let src = vec![1.5f32; n];
            let mut dst = vec![0.0f32; n];

            for _ in 0..64 {
                staging.upload(&mut d, &src, &stream).expect("upload");
                staging.download(&mut dst, &d, &stream).expect("download");
            }
            assert_eq!(dst, src);
            let stats = staging.stats();
            assert_eq!(
                stats.allocations, 1,
                "staging buffer re-pinned host memory instead of reusing it"
            );
            assert_eq!(stats.staged_transfers, 128);
            assert_eq!(stats.direct_transfers, 0);
        }

        /// Growing to a larger shape re-pins exactly once, then shrinking back
        /// reuses the high-water allocation without pinning again.
        #[test]
        fn growth_is_high_water_marked() {
            let Some((_ctx, stream)) = fixture() else {
                eprintln!("skipping: no CUDA driver/device");
                return;
            };
            let mut staging = StagingBuffer::new();
            let small = 1024usize;
            let large = 64 * 1024usize;

            let mut d_small = DeviceBuffer::<f32>::alloc(small).expect("alloc small");
            let mut d_large = DeviceBuffer::<f32>::alloc(large).expect("alloc large");

            staging
                .upload(&mut d_small, &vec![1.0f32; small], &stream)
                .expect("small");
            assert_eq!(staging.stats().allocations, 1);
            let cap_after_small = staging.capacity();

            staging
                .upload(&mut d_large, &vec![2.0f32; large], &stream)
                .expect("large");
            assert_eq!(staging.stats().allocations, 2, "growth should re-pin once");
            assert!(staging.capacity() > cap_after_small);
            let cap_after_large = staging.capacity();

            // Back to the small shape: must not re-pin, must not shrink.
            staging
                .upload(&mut d_small, &vec![3.0f32; small], &stream)
                .expect("small again");
            assert_eq!(staging.stats().allocations, 2, "shrinking must not re-pin");
            assert_eq!(staging.capacity(), cap_after_large);
        }

        /// Transfers above the threshold must bypass staging (that is the whole
        /// point of the threshold) while still producing correct data.
        #[test]
        fn oversized_transfers_bypass_staging_and_stay_correct() {
            let Some((_ctx, stream)) = fixture() else {
                eprintln!("skipping: no CUDA driver/device");
                return;
            };
            let n = 256 * 1024; // 1 MiB of f32 -- above the 512 KiB default.
            let mut staging = StagingBuffer::new();
            let mut d = DeviceBuffer::<f32>::alloc(n).expect("alloc");
            let src: Vec<f32> = (0..n).map(|i| (i % 613) as f32).collect();
            let mut dst = vec![0.0f32; n];

            staging.upload(&mut d, &src, &stream).expect("upload");
            staging.download(&mut dst, &d, &stream).expect("download");

            assert_eq!(dst, src);
            let stats = staging.stats();
            assert_eq!(stats.direct_transfers, 2, "should have bypassed staging");
            assert_eq!(stats.staged_transfers, 0);
            assert_eq!(
                stats.allocations, 0,
                "bypassed transfers must not pin host memory"
            );
            assert_eq!(stats.bytes_uploaded, (n * 4) as u64);
            assert_eq!(stats.bytes_downloaded, (n * 4) as u64);

            // Raising the threshold routes the same transfer through staging,
            // and it must still be correct.
            staging.set_auto_stage_max_bytes(usize::MAX);
            dst.fill(0.0);
            staging
                .upload(&mut d, &src, &stream)
                .expect("upload staged");
            staging
                .download(&mut dst, &d, &stream)
                .expect("download staged");
            assert_eq!(dst, src);
            assert_eq!(staging.stats().staged_transfers, 2);
        }

        /// Length mismatches must be rejected before any driver call.
        #[test]
        fn length_mismatch_is_rejected() {
            let Some((_ctx, stream)) = fixture() else {
                eprintln!("skipping: no CUDA driver/device");
                return;
            };
            let mut staging = StagingBuffer::new();
            let mut d = DeviceBuffer::<f32>::alloc(64).expect("alloc");
            let src = vec![0.0f32; 32];
            assert_eq!(
                staging.upload(&mut d, &src, &stream),
                Err(CudaError::InvalidValue)
            );
            let mut dst = vec![0.0f32; 32];
            assert_eq!(
                staging.download(&mut dst, &d, &stream),
                Err(CudaError::InvalidValue)
            );
            assert_eq!(
                staging.upload_with(&mut d, 32, &stream, |_: &mut [f32]| {}),
                Err(CudaError::InvalidValue)
            );
        }

        /// `with_capacity` must pre-pin so the steady-state loop never allocates.
        #[test]
        fn with_capacity_preallocates() {
            let Some((_ctx, stream)) = fixture() else {
                eprintln!("skipping: no CUDA driver/device");
                return;
            };
            let n = 8192usize;
            let mut staging = StagingBuffer::with_capacity(n * 4).expect("with_capacity");
            assert!(staging.capacity() >= n * 4);
            assert_eq!(staging.stats().allocations, 1);

            let mut d = DeviceBuffer::<f32>::alloc(n).expect("alloc");
            for _ in 0..16 {
                staging
                    .upload(&mut d, &vec![7.0f32; n], &stream)
                    .expect("upload");
            }
            assert_eq!(
                staging.stats().allocations,
                1,
                "pre-sized buffer must never re-pin in the hot loop"
            );
        }
    }
}
