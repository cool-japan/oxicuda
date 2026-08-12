//! DNN handle management.
//!
//! [`DnnHandle`] is the central object for all DNN operations, analogous to
//! `cudnnHandle_t` in cuDNN.  It owns a CUDA stream, a [`BlasHandle`] for
//! matrix operations, a [`KernelCache`] of JIT-compiled kernels, a
//! [`PtxCache`] of generated PTX text on disk, and an optional workspace
//! buffer.
//!
//! # Example
//!
//! ```rust,no_run
//! # use std::sync::Arc;
//! # use oxicuda_driver::Context;
//! # use oxicuda_dnn::handle::DnnHandle;
//! # fn main() -> Result<(), oxicuda_dnn::error::DnnError> {
//! # let ctx: Arc<Context> = unimplemented!();
//! let mut handle = DnnHandle::new(&ctx)?;
//! handle.set_workspace(1 << 20)?; // 1 MiB workspace
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use oxicuda_blas::BlasHandle;
use oxicuda_driver::ffi::CU_EVENT_DISABLE_TIMING;
use oxicuda_driver::{Context, Event, Module, Stream};
use oxicuda_memory::{DeviceBuffer, StagingBuffer, StagingPod, StagingStats};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::cache::PtxCache;

use crate::error::{DnnError, DnnResult};
use crate::kernel_cache::{CachedKernel, KernelCache};

/// Central handle for DNN operations.
///
/// Every DNN routine requires a `DnnHandle`.  The handle binds operations to
/// a specific CUDA context and stream, maintains a BLAS sub-handle for GEMM
/// workloads (e.g. im2col convolutions), caches JIT-compiled kernels so a
/// repeated operation never re-runs `cuModuleLoadData`, and optionally holds a
/// pre-allocated workspace buffer.
///
/// # Thread safety
///
/// `DnnHandle` is `Send` but **not** `Sync`.  Each thread should create its
/// own handle (possibly sharing the same [`Arc<Context>`]).
pub struct DnnHandle {
    /// The CUDA context this handle is bound to.
    context: Arc<Context>,
    /// The stream on which DNN kernels are launched.
    stream: Stream,
    /// Sub-handle for BLAS operations (GEMM inside convolution, etc.).
    blas_handle: BlasHandle,
    /// On-disk cache of generated PTX **text**, keyed by
    /// [`PtxCacheKey`](oxicuda_ptx::cache::PtxCacheKey) and persisted under
    /// `~/.cache/oxicuda/ptx/`.
    ///
    /// This is *not* the hot-path cache: the expensive step in a repeated DNN
    /// call is `cuModuleLoadData` (the PTX→SASS JIT), not the string building
    /// that produces the PTX. That step is served by [`Self::kernel_cache`],
    /// which memoises the compiled [`Module`] in process memory and therefore
    /// skips PTX generation as well. The disk cache is retained for callers
    /// that want to inspect or pre-seed generated PTX across processes; it is
    /// deliberately *not* interposed in front of code generation, because its
    /// key is only invalidated by the `oxicuda-ptx` package version and would
    /// otherwise serve stale PTX after an edit to a kernel emitter in *this*
    /// crate.
    ptx_cache: PtxCache,
    /// In-memory cache of JIT-compiled modules/kernels for this handle,
    /// keyed by code-generation key. Turns the per-call `Module::from_ptx`
    /// JIT into a hash lookup for a pipeline that re-invokes the same kernels
    /// every frame.
    kernel_cache: KernelCache,
    /// SM architecture of the device, cached for kernel selection.
    sm_version: SmVersion,
    /// Optional pre-allocated workspace buffer for algorithms that need
    /// temporary storage (e.g. im2col, Winograd transforms).
    workspace: Option<DeviceBuffer<u8>>,
    /// Reused page-locked host buffer backing this handle's staged
    /// host↔device transfers (see [`Self::upload_staged_with`]).
    ///
    /// Allocated lazily on first use and grown to a high-water mark, so a
    /// pipeline moving the same tensor shapes every frame pins host memory
    /// once rather than calling `cuMemAllocHost_v2` per call.
    staging: StagingBuffer,
    /// Reusable timing-disabled event used to order this handle's two streams
    /// against each other *on the device* before a staged transfer.
    ///
    /// Held on the handle rather than created per call because
    /// `cuEventCreate`/`cuEventDestroy` are far more expensive than the
    /// record/wait pair they enable, and a staged transfer may happen on every
    /// frame. See [`Self::join_blas_stream`].
    join_event: Event,
}

impl DnnHandle {
    /// Creates a new DNN handle with a freshly-allocated default stream.
    ///
    /// The device's compute capability is queried once and cached.  A
    /// [`BlasHandle`] and [`PtxCache`] are created internally.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::Cuda`] if stream creation or device query fails.
    /// Returns [`DnnError::Blas`] if the BLAS handle cannot be created.
    /// Returns [`DnnError::Io`] if the PTX cache directory cannot be created.
    pub fn new(ctx: &Arc<Context>) -> DnnResult<Self> {
        let stream = Stream::new(ctx)?;
        Self::build(ctx, stream)
    }

    /// Creates a new DNN handle bound to an existing stream.
    ///
    /// This avoids allocating an extra stream when the caller already has
    /// one (e.g. from a training pipeline).
    ///
    /// # Errors
    ///
    /// Same as [`new`](Self::new) except stream creation cannot fail.
    pub fn with_stream(ctx: &Arc<Context>, stream: Stream) -> DnnResult<Self> {
        Self::build(ctx, stream)
    }

    /// Shared construction logic.
    fn build(ctx: &Arc<Context>, stream: Stream) -> DnnResult<Self> {
        let device = ctx.device();
        let (major, minor) = device.compute_capability()?;
        let sm_version = SmVersion::from_compute_capability(major, minor).ok_or_else(|| {
            DnnError::UnsupportedOperation(format!(
                "unsupported compute capability: {major}.{minor}"
            ))
        })?;

        // Create a separate stream for the internal BLAS handle so that
        // BLAS and DNN launches can be overlapped when appropriate.
        let blas_stream = Stream::new(ctx)?;
        let blas_handle = BlasHandle::with_stream(ctx, blas_stream)?;
        let ptx_cache = PtxCache::new()?;

        // Timing is disabled: this event is only ever used to express a
        // cross-stream dependency, never measured, and the timing-enabled
        // variant costs extra device work on every record.
        let join_event = Event::with_flags(CU_EVENT_DISABLE_TIMING)?;

        Ok(Self {
            context: Arc::clone(ctx),
            stream,
            blas_handle,
            ptx_cache,
            kernel_cache: KernelCache::new(),
            sm_version,
            workspace: None,
            staging: StagingBuffer::new(),
            join_event,
        })
    }

    // -- Compiled-kernel cache -----------------------------------------------

    /// Returns the compiled [`Module`] for `key`, generating its PTX and
    /// JIT-compiling it only on a cache miss.
    ///
    /// This is the DNN analogue of `BlasHandle::get_or_compile_module`. Without
    /// it, every DNN operation pays a full `cuModuleLoadData` JIT on *every*
    /// call — around 194 us on an RTX A4000 — which dominates the runtime of a
    /// batch-1 inference pipeline that re-invokes the same kernels once per
    /// frame.
    ///
    /// # The keying contract
    ///
    /// `key` must capture **everything the generated PTX depends on** — not
    /// merely the kernel's entry-point name. A generator that bakes, say, a
    /// channel count or an activation choice into the instruction stream while
    /// naming its entry point after only the precision emits *different* PTX
    /// under the *same* name; keying such a kernel by name alone hands back a
    /// module compiled for a different problem, producing wrong results with no
    /// error raised anywhere. Fold the target architecture into the key too: an
    /// engine carries its own SM version, independent of the handle it is
    /// executed against.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::LaunchFailed`] if the cache lock is poisoned,
    /// whatever `gen_ptx` returns on failure, or [`DnnError::Cuda`] if the
    /// generated PTX does not compile.
    pub fn get_or_compile_module(
        &self,
        key: &str,
        gen_ptx: impl FnOnce() -> DnnResult<String>,
    ) -> DnnResult<Arc<Module>> {
        self.kernel_cache.get_or_compile_module(key, gen_ptx)
    }

    /// Returns the [`CachedKernel`] for entry point `entry` in the module
    /// identified by `key`, compiling on a miss.
    ///
    /// This is what every in-crate launch site uses. Prefer it over
    /// [`Self::get_or_compile_module`]: it also memoises the
    /// `cuModuleGetFunction` lookup and gives the kernel's occupancy query
    /// somewhere stable to be cached (see [`CachedKernel::launch_1d`]).
    ///
    /// `key` carries the same obligation as
    /// [`Self::get_or_compile_module`]'s; `entry` is only the name looked up
    /// inside the compiled module, and may well be less discriminating than the
    /// key (see [`crate::kernel_cache::cache_key_with`]).
    ///
    /// # Errors
    ///
    /// As [`Self::get_or_compile_module`], plus [`DnnError::Cuda`] if `entry`
    /// is absent from the compiled module.
    pub(crate) fn get_or_compile_kernel(
        &self,
        key: &str,
        entry: &str,
        gen_ptx: impl FnOnce() -> DnnResult<String>,
    ) -> DnnResult<Arc<CachedKernel>> {
        self.kernel_cache.get_or_compile_kernel(key, entry, gen_ptx)
    }

    /// Number of distinct modules this handle has JIT-compiled and retained.
    #[cfg(all(test, feature = "gpu-tests"))]
    pub(crate) fn compiled_module_count(&self) -> usize {
        self.kernel_cache.module_count()
    }

    // -- Accessors -----------------------------------------------------------

    /// Returns a reference to the CUDA context.
    #[inline]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns a reference to the stream used for kernel launches.
    #[inline]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Returns a reference to the internal BLAS handle.
    #[inline]
    pub fn blas(&self) -> &BlasHandle {
        &self.blas_handle
    }

    /// Blocks until every operation queued on *either* of this handle's two
    /// streams has completed: [`Self::stream`] (used by DNN kernels launched
    /// directly against this handle) **and** [`BlasHandle::stream`] on
    /// [`Self::blas`] (a separate stream — see [`Self::build`] — used by GEMM
    /// launches routed through [`Self::blas`]).
    ///
    /// # Why this exists
    ///
    /// `DnnHandle` deliberately gives its BLAS sub-handle its own stream so
    /// BLAS and DNN launches can be overlapped (see [`Self::build`]'s doc
    /// comment). That is exactly what makes synchronizing only
    /// [`Self::stream`] *look* correct while actually racing: a caller that
    /// dispatches work through [`Self::blas`] and then calls
    /// `dnn_handle.stream().synchronize()` has synchronized a stream nothing
    /// was queued on. CUDA streams created `CU_STREAM_NON_BLOCKING` (as every
    /// stream here is) do not implicitly order against each other, so the
    /// host can read back a result buffer *before* the kernel that was
    /// supposed to fill it has actually run — observing whatever was in that
    /// device memory previously (for a freshly zeroed buffer, silently
    /// plausible-looking zeros, not a crash). This happened for real: see
    /// `oxionnx-cuda::matmul::cuda_matmul`'s pre-fix history.
    ///
    /// A caller that only ever dispatches through *one* of the two streams
    /// may synchronize that one directly (cheaper: one blocking driver call
    /// instead of two); this method is for callers that dispatch through
    /// [`Self::blas`] — or aren't sure which stream a helper used internally
    /// — and want a synchronization call that is correct regardless.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::Cuda`] if either stream's synchronization fails.
    pub fn synchronize_all(&self) -> DnnResult<()> {
        self.stream.synchronize()?;
        self.blas_handle.stream().synchronize()?;
        Ok(())
    }

    /// Returns a mutable reference to the internal BLAS handle.
    #[inline]
    pub fn blas_mut(&mut self) -> &mut BlasHandle {
        &mut self.blas_handle
    }

    /// Returns the SM version of the bound device.
    #[inline]
    pub fn sm_version(&self) -> SmVersion {
        self.sm_version
    }

    /// Returns a reference to the PTX cache.
    #[inline]
    pub fn ptx_cache(&self) -> &PtxCache {
        &self.ptx_cache
    }

    /// Returns a reference to the workspace buffer, if one has been allocated.
    #[inline]
    pub fn workspace(&self) -> Option<&DeviceBuffer<u8>> {
        self.workspace.as_ref()
    }

    /// Returns a mutable reference to the workspace buffer, if allocated.
    #[inline]
    pub fn workspace_mut(&mut self) -> Option<&mut DeviceBuffer<u8>> {
        self.workspace.as_mut()
    }

    // -- Mutators ------------------------------------------------------------

    /// Replaces the stream used for subsequent DNN operations.
    ///
    /// The previous stream is **not** synchronised; the caller must ensure
    /// all in-flight work has completed before swapping streams.
    pub fn set_stream(&mut self, stream: Stream) {
        self.stream = stream;
    }

    /// Allocates (or re-allocates) the workspace buffer with at least `bytes`
    /// bytes of device memory.
    ///
    /// If the current workspace is already large enough, this is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `bytes` is zero.
    /// Returns [`DnnError::Cuda`] if device memory allocation fails.
    pub fn set_workspace(&mut self, bytes: usize) -> DnnResult<()> {
        if bytes == 0 {
            return Err(DnnError::InvalidArgument(
                "workspace size must be non-zero".into(),
            ));
        }
        // Skip re-allocation if current workspace is already sufficient.
        if let Some(ref ws) = self.workspace {
            if ws.len() >= bytes {
                return Ok(());
            }
        }
        let buf = DeviceBuffer::<u8>::alloc(bytes)?;
        self.workspace = Some(buf);
        Ok(())
    }

    /// Drops the current workspace buffer, freeing device memory.
    pub fn clear_workspace(&mut self) {
        self.workspace = None;
    }

    // -- Staged host <-> device transfers ------------------------------------

    /// Returns this handle's reusable pinned staging buffer.
    #[inline]
    pub fn staging(&self) -> &StagingBuffer {
        &self.staging
    }

    /// Returns this handle's staging buffer mutably (e.g. to retune
    /// [`StagingBuffer::set_auto_stage_max_bytes`]).
    #[inline]
    pub fn staging_mut(&mut self) -> &mut StagingBuffer {
        &mut self.staging
    }

    /// Usage counters for the staging buffer; see [`StagingStats`].
    ///
    /// `allocations == 1` after a long run is the signal that the buffer is
    /// genuinely being reused rather than re-pinned per call.
    #[inline]
    pub fn staging_stats(&self) -> StagingStats {
        self.staging.stats()
    }

    /// Pre-pins `bytes` of host staging memory.
    ///
    /// Call once at model-load time with the largest tensor the pipeline will
    /// move, and `cuMemAllocHost_v2` — a millisecond-scale driver call — never
    /// runs again in the steady-state loop.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::Cuda`] if `bytes` is zero or the host allocation
    /// fails.
    pub fn reserve_staging(&mut self, bytes: usize) -> DnnResult<()> {
        self.staging.reserve(bytes)?;
        Ok(())
    }

    /// Makes this handle's launch stream wait for everything currently queued
    /// on the BLAS sub-handle's stream — **on the device**, without blocking
    /// the host.
    ///
    /// [`Self::build`] deliberately gives [`Self::blas`] its own stream so BLAS
    /// and DNN launches can overlap, and streams here are all
    /// `CU_STREAM_NON_BLOCKING`, so the two are unordered by default. A staged
    /// readback enqueued on [`Self::stream`] would therefore happily race ahead
    /// of a GEMM dispatched through [`Self::blas`] and return whatever the
    /// output buffer held beforehand — plausible-looking zeros, not a crash.
    /// That failure mode is not hypothetical; it is the bug
    /// [`Self::synchronize_all`]'s doc comment records.
    ///
    /// The cheap fix is *not* [`Self::synchronize_all`]: that blocks the calling
    /// thread on both streams. Recording an event on the BLAS stream and having
    /// the launch stream wait on it expresses the same ordering as a device-side
    /// dependency — the host does not block at all, and the DMA still cannot
    /// start before the GEMM finishes.
    fn join_blas_stream(&self) -> DnnResult<()> {
        self.join_event.record(self.blas_handle.stream())?;
        self.stream.wait_event(&self.join_event)?;
        Ok(())
    }

    /// Uploads `n` elements into `dst`, letting `fill` write them **directly
    /// into page-locked host memory**.
    ///
    /// This is the fastest host→device path available through a `DnnHandle`
    /// (measured 1.55x–1.75x over [`DeviceBuffer::copy_from_host`] from 150 KiB
    /// to 4.7 MiB on an RTX A4000): the bytes `fill` writes are the exact bytes
    /// the DMA engine reads, so there is no pageable source buffer and no
    /// driver-internal bounce copy. Preprocessing that can emit its output
    /// straight into the provided slice — normalising pixels into an NCHW input
    /// tensor, for instance — pays no copy at all.
    ///
    /// `fill` must write all `n` elements; untouched elements retain whatever
    /// the previous staged transfer left behind and are uploaded as-is.
    ///
    /// Ordered after work on **both** of this handle's streams (see
    /// [`join_blas_stream`](Self::join_blas_stream)), and returns only once the
    /// upload has landed.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::Cuda`] if `n` is zero, `n != dst.len()`, or the
    /// allocation/copy fails.
    pub fn upload_staged_with<T, F>(
        &mut self,
        dst: &mut DeviceBuffer<T>,
        n: usize,
        fill: F,
    ) -> DnnResult<()>
    where
        T: StagingPod,
        F: FnOnce(&mut [T]),
    {
        self.join_blas_stream()?;
        let Self {
            staging, stream, ..
        } = self;
        staging.upload_with(dst, n, stream, fill)?;
        Ok(())
    }

    /// Downloads `n` elements from `src` into page-locked host memory and
    /// returns a borrowed view — **no copy out**.
    ///
    /// The fastest device→host path available through a `DnnHandle` (measured
    /// 1.77x–2.67x over [`DeviceBuffer::copy_to_host`]): the DMA engine writes
    /// straight into the memory the caller reads. The returned slice borrows the
    /// handle and remains valid until the next staged transfer.
    ///
    /// Ordered after work on **both** of this handle's streams, so results
    /// produced by a GEMM dispatched through [`Self::blas`] are visible without
    /// a preceding [`Self::synchronize_all`].
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::Cuda`] if `n` is zero, `n != src.len()`, or the
    /// allocation/copy fails.
    pub fn download_staged_into<T: StagingPod>(
        &mut self,
        src: &DeviceBuffer<T>,
        n: usize,
    ) -> DnnResult<&[T]> {
        self.join_blas_stream()?;
        let Self {
            staging, stream, ..
        } = self;
        Ok(staging.download_into(src, n, stream)?)
    }

    /// Uploads `src` into `dst`, staging through pinned memory when that is
    /// actually faster.
    ///
    /// A drop-in replacement for [`DeviceBuffer::copy_from_host`] with the same
    /// postcondition that is never slower — see [`StagingBuffer::upload`] for
    /// why large transfers deliberately bypass the staging buffer. When you
    /// control how `src` is produced, prefer
    /// [`upload_staged_with`](Self::upload_staged_with), which removes this
    /// method's host memcpy and wins at every size.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::Cuda`] if `src` is empty, lengths disagree, or the
    /// transfer fails.
    pub fn upload_staged<T: StagingPod>(
        &mut self,
        dst: &mut DeviceBuffer<T>,
        src: &[T],
    ) -> DnnResult<()> {
        self.join_blas_stream()?;
        let Self {
            staging, stream, ..
        } = self;
        staging.upload(dst, src, stream)?;
        Ok(())
    }

    /// Downloads `src` into `dst`, staging through pinned memory when that is
    /// actually faster.
    ///
    /// The mirror of [`upload_staged`](Self::upload_staged). Unlike a bare
    /// [`DeviceBuffer::copy_to_host`], this is ordered against both of the
    /// handle's streams, so it cannot read back a buffer whose producing kernel
    /// has not run. Prefer
    /// [`download_staged_into`](Self::download_staged_into) when the results can
    /// be consumed in place.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::Cuda`] if `dst` is empty, lengths disagree, or the
    /// transfer fails.
    pub fn download_staged<T: StagingPod>(
        &mut self,
        dst: &mut [T],
        src: &DeviceBuffer<T>,
    ) -> DnnResult<()> {
        self.join_blas_stream()?;
        let Self {
            staging, stream, ..
        } = self;
        staging.download(dst, src, stream)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SmVersion must be Copy for the getter to return by value.
    #[test]
    fn sm_version_is_copy() {
        let v = SmVersion::Sm80;
        let v2 = v;
        assert_eq!(v, v2);
    }
}
