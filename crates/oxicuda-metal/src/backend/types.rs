//! `MetalBackend` struct, intrinsic helpers, and Metal-API dispatch helpers.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use oxicuda_backend::{BackendError, BackendResult, BackendTranspose, BinaryOp, ReduceOp, UnaryOp};

use crate::{device::MetalDevice, memory::MetalMemoryManager, pipeline::MetalComputePipeline};

#[cfg(target_os = "macos")]
use super::functions::{
    chunked_reduce_function_name, chunked_reduce_msl, clamp_1d_threadgroup, next_power_of_2,
    pow2_threadgroup, resolve_buffers, to_u32,
};
#[cfg(target_os = "macos")]
use crate::pipeline::commit_and_wait;

/// Maximum number of compiled pipelines kept alive by
/// [`MetalBackend::pipeline_cache`].
///
/// The cache must be bounded: MSL generators that bake their parameters in as
/// compile-time constants (conv2d, attention, an autotuner sweeping tile sizes)
/// produce a distinct source — hence a distinct pipeline — per configuration, so
/// an unbounded cache grows `MTLComputePipelineState` objects without limit.
const PIPELINE_CACHE_CAPACITY: usize = 64;

/// Identity of a cached compute pipeline.
///
/// Deliberately **not** a 64-bit hash of the MSL source: `DefaultHasher` is not
/// collision resistant, and a collision between two sources sharing one
/// entry-point name would silently run the wrong kernel. Both variants store the
/// full discriminating data and are compared for equality on lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum PipelineKey {
    /// A kernel this crate generates itself, identified semantically.
    ///
    /// `kind` names the dispatch family (`"unary"`, `"binary"`, `"reduce"`,
    /// `"reduce_chunked"`, `"gemm"`, …), `op` the operation within it
    /// (`"relu"`, `"add"`, `"sum"`, …) and `dtype` the element type. Keying on
    /// this tuple avoids regenerating and re-hashing multi-KB MSL source on
    /// every dispatch — the source is only built on a cache miss.
    ///
    /// INVARIANT: this key is exact only while the corresponding
    /// [`crate::msl`] generator bakes nothing into the source beyond `op`. If a
    /// generator gains a parameter (a tile size, a shape constant), that
    /// parameter **must** become part of this key, or distinct kernels will
    /// alias onto one cache entry.
    ///
    /// Only constructed on macOS — off macOS the six built-in dispatch paths
    /// are stubs that never reach the cache.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    Builtin {
        /// Dispatch family the kernel belongs to.
        kind: &'static str,
        /// Operation within the family.
        op: &'static str,
        /// Element type the kernel is compiled for.
        dtype: &'static str,
    },
    /// A caller-supplied kernel: keyed on the exact entry point **and source**,
    /// so two different sources can never share a pipeline.
    Custom {
        /// MSL entry-point name.
        function_name: String,
        /// The full MSL source text.
        source: String,
    },
}

/// A cached pipeline plus the LRU timestamp of its last use.
#[derive(Debug)]
struct CachedPipeline {
    pipeline: Arc<MetalComputePipeline>,
    last_used: u64,
}

/// A bounded, least-recently-used cache of compiled compute pipelines.
#[derive(Debug)]
pub(super) struct PipelineCache {
    entries: HashMap<PipelineKey, CachedPipeline>,
    capacity: usize,
    clock: u64,
}

impl PipelineCache {
    /// Create an empty cache holding at most `capacity` pipelines.
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity: capacity.max(1),
            clock: 0,
        }
    }

    /// Look `key` up, refreshing its LRU timestamp on a hit.
    fn get(&mut self, key: &PipelineKey) -> Option<Arc<MetalComputePipeline>> {
        self.clock = self.clock.wrapping_add(1);
        let now = self.clock;
        self.entries.get_mut(key).map(|entry| {
            entry.last_used = now;
            Arc::clone(&entry.pipeline)
        })
    }

    /// Insert `pipeline`, evicting the least recently used entries first when
    /// the cache is at capacity.
    fn insert(&mut self, key: PipelineKey, pipeline: Arc<MetalComputePipeline>) {
        self.clock = self.clock.wrapping_add(1);
        if !self.entries.contains_key(&key) {
            while self.entries.len() >= self.capacity {
                let victim = self
                    .entries
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_used)
                    .map(|(k, _)| k.clone());
                match victim {
                    Some(k) => {
                        self.entries.remove(&k);
                    }
                    // Unreachable while `capacity >= 1`, but never loop forever.
                    None => break,
                }
            }
        }
        self.entries.insert(
            key,
            CachedPipeline {
                pipeline,
                last_used: self.clock,
            },
        );
    }

    /// Number of pipelines currently held (test observability).
    ///
    /// The cache-behaviour tests need a live Metal device, so this is unused off
    /// macOS.
    #[cfg(test)]
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

/// The external Metal buffer type accepted by
/// [`MetalBackend::register_external`] / [`MetalBackend::import_buffer`].
///
/// On macOS this is `metal::Buffer`. On every other platform it is an opaque
/// placeholder so the zero-copy import API has the **same shape on all
/// targets** — callers no longer need their own `cfg(target_os = "macos")`
/// gate; the calls simply return `UnsupportedPlatform`.
#[cfg(target_os = "macos")]
pub type MetalExternalBuffer = metal::Buffer;

/// The external Metal buffer type accepted by
/// [`MetalBackend::register_external`] / [`MetalBackend::import_buffer`].
///
/// Off macOS there is no Metal, so this is an inert placeholder: it exists only
/// so the import entry points keep the same signature everywhere. Any call made
/// with it returns [`BackendError::DeviceError`] wrapping
/// [`crate::error::MetalError::UnsupportedPlatform`].
#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalExternalBuffer;

/// Packed (minimum) leading dimensions for the row-major GEMM kernels.
///
/// The v2 GEMM kernels ([`crate::msl::gemm_msl_v2`] and friends) store `op(A)`'s
/// physical buffer as `m×k` — or its transpose `k×m` — `op(B)` as `k×n` (or
/// `n×k`) and `C` as `m×n`, all row-major. The leading dimension is the physical
/// row stride, so the tightly-packed value is the width of each stored row; a
/// caller may pass a larger `ld` (a padded or sub-matrix view), which the shader
/// honours, and a smaller one is invalid.
///
/// Deliberately identical to `oxicuda_webgpu`'s `packed_gemm_lds`, so the two
/// GPU backends accept exactly the same argument space.
pub(super) fn packed_gemm_lds(
    trans_a: BackendTranspose,
    trans_b: BackendTranspose,
    m: usize,
    n: usize,
    k: usize,
) -> (usize, usize, usize) {
    let lda = if trans_a == BackendTranspose::NoTrans {
        k
    } else {
        m
    };
    let ldb = if trans_b == BackendTranspose::NoTrans {
        n
    } else {
        k
    };
    (lda, ldb, n)
}

/// Validate that a GEMM request's leading dimensions are large enough for the
/// operand extents.
///
/// Every transpose combination is supported — the v2 kernels take `trans_a`,
/// `trans_b`, `lda`, `ldb` and `ldc` from a runtime parameter buffer — so the
/// only thing left to reject is a leading dimension **smaller** than the stored
/// row it describes, which would make the kernel read across row boundaries.
///
/// Reported as [`BackendError::InvalidArgument`] (not `Unsupported`): the
/// argument is malformed, not a capability gap, and this matches the error kind
/// `oxicuda_webgpu` returns for the same condition.
pub(super) fn validate_gemm_layout(
    trans_a: BackendTranspose,
    trans_b: BackendTranspose,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) -> BackendResult<()> {
    let (min_lda, min_ldb, min_ldc) = packed_gemm_lds(trans_a, trans_b, m, n, k);
    if lda < min_lda || ldb < min_ldb || ldc < min_ldc {
        return Err(BackendError::InvalidArgument(format!(
            "Metal GEMM: leading dimension smaller than the matrix extent \
             (need lda>={min_lda}, ldb>={min_ldb}, ldc>={min_ldc}); \
             got lda={lda}, ldb={ldb}, ldc={ldc}"
        )));
    }
    Ok(())
}

/// Runtime parameter buffer shared by [`crate::msl::gemm_msl_v2`] and
/// [`crate::msl::gemm_msl_v2`]-derived FP16 kernels.
///
/// Field order and types mirror the MSL `GemmParamsV2` struct exactly; see that
/// generator's doc comment for the authoritative byte-offset table.
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GemmParamsV2 {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    trans_a: u32,
    trans_b: u32,
    alpha: f32,
    beta: f32,
}

/// Runtime parameter buffer for [`crate::msl::batched_gemm_msl_v2`]: the 40-byte
/// [`GemmParamsV2`] prefix followed by the batch descriptor.
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct BatchedGemmParamsV2 {
    base: GemmParamsV2,
    batch_count: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
}

/// Build the v2 parameter block, narrowing every dimension with a checked
/// conversion.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn gemm_params_v2(
    what: &str,
    trans_a: BackendTranspose,
    trans_b: BackendTranspose,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    alpha: f32,
    beta: f32,
) -> BackendResult<GemmParamsV2> {
    Ok(GemmParamsV2 {
        m: to_u32(m, &format!("{what} m"))?,
        n: to_u32(n, &format!("{what} n"))?,
        k: to_u32(k, &format!("{what} k"))?,
        lda: to_u32(lda, &format!("{what} lda"))?,
        ldb: to_u32(ldb, &format!("{what} ldb"))?,
        ldc: to_u32(ldc, &format!("{what} ldc"))?,
        trans_a: u32::from(trans_a != BackendTranspose::NoTrans),
        trans_b: u32::from(trans_b != BackendTranspose::NoTrans),
        alpha,
        beta,
    })
}

/// Apple Metal GPU compute backend.
///
/// On macOS this selects the system-default Metal device and allocates
/// shared-memory buffers that are directly accessible from both CPU and GPU.
///
/// On non-macOS platforms every operation returns
/// [`BackendError::DeviceError`] (wrapping [`crate::error::MetalError::UnsupportedPlatform`]).
///
/// # Lifecycle
///
/// 1. `MetalBackend::new()` — create an uninitialised backend.
/// 2. `init()` — acquire the Metal device and set up the memory manager.
/// 3. Use `alloc`, `copy_htod`, compute ops, `copy_dtoh`, `free`.
/// 4. `synchronize()` — wait for all pending GPU work to finish.
#[derive(Debug)]
pub struct MetalBackend {
    pub(super) device: Option<Arc<MetalDevice>>,
    pub(super) memory: Option<Arc<MetalMemoryManager>>,
    pub(super) initialized: bool,
    /// Bounded LRU cache of compiled pipelines, keyed by [`PipelineKey`], so
    /// repeated dispatches reuse the compiled pipeline instead of recompiling
    /// (and, for built-in kernels, without even regenerating the MSL source).
    pub(super) pipeline_cache: Mutex<PipelineCache>,
    /// Whether dispatches return before the GPU finishes them. **Off by
    /// default**; see [`MetalBackend::set_async_dispatch`].
    pub(super) async_dispatch: AtomicBool,
    /// Command buffers committed but not yet waited on, newest last, each with
    /// the label its failure should be reported under. Empty unless
    /// `async_dispatch` is on.
    #[cfg(target_os = "macos")]
    pub(super) inflight: Mutex<Vec<(metal::CommandBuffer, &'static str)>>,
}
impl MetalBackend {
    /// Create a new, uninitialised Metal backend.
    pub fn new() -> Self {
        Self {
            device: None,
            memory: None,
            initialized: false,
            pipeline_cache: Mutex::new(PipelineCache::new(PIPELINE_CACHE_CAPACITY)),
            async_dispatch: AtomicBool::new(false),
            #[cfg(target_os = "macos")]
            inflight: Mutex::new(Vec::new()),
        }
    }

    /// Whether dispatches currently return before the GPU has finished them.
    pub fn async_dispatch(&self) -> bool {
        self.async_dispatch.load(Ordering::Acquire)
    }

    /// Number of command buffers committed but not yet waited on.
    ///
    /// Always `0` in the default synchronous mode.
    pub fn inflight_count(&self) -> usize {
        #[cfg(target_os = "macos")]
        {
            self.inflight.lock().map(|q| q.len()).unwrap_or(0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            0
        }
    }

    /// Enable or disable **asynchronous dispatch** (opt-in; the default is
    /// synchronous).
    ///
    /// # What changes
    ///
    /// Synchronously (the default), every op commits its command buffer and
    /// blocks in `waitUntilCompleted` before returning, so the GPU idles between
    /// ops and a chain of small kernels is dominated by round-trip latency. With
    /// async dispatch on, an op commits and returns; the command buffer is
    /// tracked and awaited at the next **synchronisation point**, which is any
    /// of:
    ///
    /// * [`synchronize`](oxicuda_backend::ComputeBackend::synchronize),
    /// * any host read or write of device memory — `copy_dtoh`, `copy_htod`,
    ///   [`copy_dtod`](Self::copy_dtod),
    /// * [`launch_custom_kernel`](Self::launch_custom_kernel), whose kernel body
    ///   is opaque to this crate and may read or write any bound buffer,
    /// * [`free`](oxicuda_backend::ComputeBackend::free) (a buffer must not
    ///   re-enter the allocator's reuse pool while a kernel still references
    ///   it),
    /// * turning async dispatch back off, and dropping the backend.
    ///
    /// Because every path that can *observe* a buffer synchronises first,
    /// results are identical to synchronous mode; only the wall-clock schedule
    /// differs. A GPU-side failure is still never reported as success — it
    /// surfaces as an error from the synchronisation point that awaits it,
    /// rather than from the op that queued it, so the error message carries the
    /// **queuing** op's label.
    ///
    /// Ordering is safe because the whole crate submits to the single
    /// device-wide `MTLCommandQueue`, which executes command buffers in commit
    /// order.
    ///
    /// # Errors
    /// Turning it **off** flushes the queue first, so this returns any GPU
    /// failure that had not yet been reported.
    pub fn set_async_dispatch(&self, enabled: bool) -> BackendResult<()> {
        self.async_dispatch.store(enabled, Ordering::Release);
        if !enabled {
            return self.drain_inflight();
        }
        Ok(())
    }

    /// Wait for every committed-but-unawaited command buffer, reporting the
    /// first GPU-side failure among them.
    ///
    /// A no-op (one empty lock acquisition) in synchronous mode, which is why
    /// the read/write paths can call it unconditionally.
    pub(super) fn drain_inflight(&self) -> BackendResult<()> {
        #[cfg(target_os = "macos")]
        {
            let pending = {
                let mut inflight = self.inflight.lock().map_err(|_| {
                    BackendError::DeviceError("in-flight command-buffer mutex poisoned".into())
                })?;
                std::mem::take(&mut *inflight)
            };
            // Every buffer must be awaited even after one of them fails, or the
            // rest stay in flight with nothing tracking them.
            let mut first_error = None;
            for (command_buffer, label) in pending {
                command_buffer.wait_until_completed();
                if let Err(e) = crate::pipeline::status_to_result(command_buffer.status(), label) {
                    first_error.get_or_insert_with(|| BackendError::from(e));
                }
            }
            match first_error {
                Some(e) => Err(e),
                None => Ok(()),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(())
        }
    }
    /// Return an error if the backend has not been initialised yet.
    pub(super) fn check_init(&self) -> BackendResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(BackendError::NotInitialized)
        }
    }
    /// Convenience accessor: get the memory manager or return `NotInitialized`.
    pub(super) fn memory(&self) -> BackendResult<&Arc<MetalMemoryManager>> {
        self.memory.as_ref().ok_or(BackendError::NotInitialized)
    }

    /// Report whether `handle` refers to an imported (external) buffer.
    ///
    /// Returns `Some(true)` for a handle created by
    /// [`register_external`](Self::register_external) /
    /// [`import_buffer`](Self::import_buffer), `Some(false)` for an
    /// [`alloc`](oxicuda_backend::ComputeBackend::alloc)-owned handle, and
    /// `None` if the handle is unknown (e.g. already freed). Useful for
    /// asserting that a cache-backed buffer was imported (so `free` will not
    /// deallocate it).
    ///
    /// # Errors
    /// [`BackendError::NotInitialized`] if the backend is not initialised.
    pub fn is_imported(&self, handle: u64) -> BackendResult<Option<bool>> {
        self.check_init()?;
        self.memory()?
            .is_external(handle)
            .map_err(BackendError::from)
    }

    /// Copy `len_bytes` from device buffer `src` to device buffer `dst`
    /// **device-to-device**, with no host round-trip. Both handles may be
    /// [`alloc`](oxicuda_backend::ComputeBackend::alloc)-owned or imported, in
    /// any combination.
    ///
    /// Useful alongside [`register_external`](Self::register_external) for
    /// keeping data GPU-resident — e.g. copying a freshly computed result into a
    /// consumer's cached buffer without round-tripping through host memory.
    ///
    /// # Errors
    /// * [`BackendError::NotInitialized`] if the backend is not initialised.
    /// * [`BackendError::InvalidArgument`] for an unknown handle, if
    ///   `src == dst`, or if `len_bytes` exceeds either buffer's length.
    pub fn copy_dtod(&self, dst: u64, src: u64, len_bytes: usize) -> BackendResult<()> {
        self.check_init()?;
        if len_bytes == 0 {
            return Ok(());
        }
        // Synchronisation point: the copy is a host-side `memcpy` between two
        // unified-memory buffers, so an in-flight kernel writing either of them
        // must finish first.
        self.drain_inflight()?;
        self.memory()?
            .copy_device_to_device(dst, src, len_bytes)
            .map_err(BackendError::from)
    }

    /// Compile (or reuse a cached) compute pipeline from `msl_source` /
    /// `function_name` and dispatch it over `total_threads` GPU threads (1-D).
    ///
    /// Device-buffer `handles` bind to `buffer(0)`, `buffer(1)`, … in order; each
    /// `scalar_bytes` blob binds with `set_bytes` to the indices immediately
    /// following the buffers (`buffer(handles.len())`, …). This lets callers pass,
    /// for example, an element count as a `u32` and clamp constants as `f32`, each
    /// encoded as raw little-endian bytes.
    ///
    /// Compiled pipelines are cached by `(function_name, msl-source-hash)`, so
    /// repeating a call with the same kernel reuses the pipeline and command
    /// queue rather than recompiling.
    ///
    /// The dispatch is synchronous: it waits for GPU completion before returning.
    /// Because the grid is rounded up to whole threadgroups, the kernel **must**
    /// bounds-check its `thread_position_in_grid` against the element count.
    ///
    /// # Errors
    /// * [`BackendError::NotInitialized`] if [`init`](oxicuda_backend::ComputeBackend::init)
    ///   has not been called — always the case on non-macOS, where Metal is
    ///   unavailable.
    /// * [`BackendError::DeviceError`] if MSL compilation or pipeline creation fails.
    /// * [`BackendError::InvalidArgument`] for an unknown buffer handle or an
    ///   empty `scalar_bytes` entry.
    pub fn launch_custom_kernel(
        &self,
        msl_source: &str,
        function_name: &str,
        handles: &[u64],
        scalar_bytes: &[&[u8]],
        total_threads: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        if total_threads == 0 {
            return Ok(());
        }
        // Synchronisation point. `MetalComputePipeline::dispatch` commits and
        // waits on its **own** command buffer only, so without this the
        // caller-supplied kernel would run against buffers that in-flight work
        // from an earlier op may still be writing, and that earlier work's
        // status would stay unchecked. This is the one path where the kernel
        // body is opaque to us, so it gets the same treatment as a host read.
        self.drain_inflight()?;
        let pipeline = self.custom_pipeline(msl_source, function_name)?;
        let memory = self.memory()?;
        pipeline
            .dispatch(memory, handles, scalar_bytes, total_threads)
            .map_err(BackendError::from)
    }

    /// Fetch a cached compiled pipeline for `key`, compiling it on first use.
    ///
    /// `build` is only invoked on a cache **miss**, so a built-in dispatch never
    /// pays for regenerating (and hashing) its multi-kilobyte MSL source.
    pub(super) fn cached_pipeline<F>(
        &self,
        key: PipelineKey,
        function_name: &str,
        build: F,
    ) -> BackendResult<Arc<MetalComputePipeline>>
    where
        F: FnOnce() -> String,
    {
        let mut cache = self
            .pipeline_cache
            .lock()
            .map_err(|_| BackendError::DeviceError("pipeline cache mutex poisoned".into()))?;
        if let Some(existing) = cache.get(&key) {
            return Ok(existing);
        }
        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        let source = build();
        if source.is_empty() {
            return Err(BackendError::Unsupported(format!(
                "no MSL source available for Metal kernel '{function_name}'"
            )));
        }
        let pipeline = Arc::new(
            MetalComputePipeline::new(device, &source, function_name)
                .map_err(BackendError::from)?,
        );
        cache.insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
    }

    /// Fetch a cached compiled pipeline for a **caller-supplied** MSL source.
    ///
    /// The key stores the source verbatim, so two different sources declaring
    /// the same entry-point name can never share a compiled pipeline.
    fn custom_pipeline(
        &self,
        msl_source: &str,
        function_name: &str,
    ) -> BackendResult<Arc<MetalComputePipeline>> {
        let key = PipelineKey::Custom {
            function_name: function_name.to_string(),
            source: msl_source.to_string(),
        };
        self.cached_pipeline(key, function_name, || msl_source.to_string())
    }

    /// Build an uninitialised backend whose pipeline cache holds at most
    /// `capacity` entries, so LRU eviction can be exercised without compiling
    /// `PIPELINE_CACHE_CAPACITY` distinct kernels.
    ///
    /// Compiling pipelines needs a live Metal device, so this is unused off macOS.
    #[cfg(test)]
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(super) fn with_pipeline_cache_capacity(capacity: usize) -> Self {
        Self {
            device: None,
            memory: None,
            initialized: false,
            pipeline_cache: Mutex::new(PipelineCache::new(capacity)),
            async_dispatch: AtomicBool::new(false),
            #[cfg(target_os = "macos")]
            inflight: Mutex::new(Vec::new()),
        }
    }

    /// Number of compiled pipelines currently cached (test observability for
    /// cache hits and LRU eviction).
    ///
    /// Compiling pipelines needs a live Metal device, so this is unused off macOS.
    #[cfg(test)]
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(super) fn pipeline_cache_len(&self) -> usize {
        self.pipeline_cache
            .lock()
            .map(|cache| cache.entry_count())
            .unwrap_or(0)
    }
}
/// Threadgroup width targeted by the reduction kernels. Must stay a power of
/// two: the MSL tree reduction folds `for (s = tg_size / 2; s > 0; s >>= 1)`,
/// which only visits every lane for power-of-two widths.
#[cfg(target_os = "macos")]
const REDUCE_TARGET_THREADS: u64 = 256;

/// Upper bound on the number of threadgroups used by pass 1 of the two-pass
/// reduction. Caps the scratch buffer at `MAX_REDUCE_GROUPS * 4` bytes while
/// still saturating an Apple GPU.
#[cfg(target_os = "macos")]
const MAX_REDUCE_GROUPS: usize = 1024;

/// Element count at which a flat (single-output) reduction switches from the
/// one-threadgroup kernel to the two-pass, multi-threadgroup path.
#[cfg(target_os = "macos")]
const TWO_PASS_REDUCE_THRESHOLD: usize = 4096;

/// Upper bound on command buffers tracked by
/// [`MetalBackend::set_async_dispatch`]'s in-flight list.
///
/// Each entry retains its command buffer *and every resource that buffer
/// references*, so an unbounded list would pin arbitrary amounts of GPU memory
/// for a caller that never synchronises. Reaching the cap forces a drain, which
/// costs one wait but keeps the memory bounded.
#[cfg(target_os = "macos")]
const MAX_INFLIGHT_COMMAND_BUFFERS: usize = 64;

/// Pre-computed geometry for the two passes of a flat reduction.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
struct ReduceFlatParams {
    /// Number of input elements reduced by pass 1.
    count: u32,
    /// Number of threadgroups (and therefore partials) in pass 1.
    groups: u32,
    /// Threadgroup width of pass 1 (a power of two).
    tg_size: u64,
    /// Threadgroup width of pass 2 (a power of two).
    final_tg: u64,
    /// Threadgroup scratch bytes for pass 1.
    scratch_bytes: u64,
    /// Threadgroup scratch bytes for pass 2.
    final_scratch: u64,
    /// Divisor applied by pass 2 — the original element count for `Mean`,
    /// `1.0` otherwise.
    divisor: f32,
}

#[cfg(target_os = "macos")]
impl MetalBackend {
    /// Submit an encoded command buffer, honouring the current dispatch mode.
    ///
    /// Synchronously (the default) this is exactly the previous
    /// commit → `waitUntilCompleted` → `status()` sequence. In async mode the
    /// buffer is committed and queued for a later
    /// [`drain_inflight`](Self::drain_inflight); `label` is retained alongside
    /// it so a failure detected at the next synchronisation point still names
    /// the op that queued it.
    ///
    /// The queue is bounded by [`MAX_INFLIGHT_COMMAND_BUFFERS`]: reaching the
    /// cap forces a drain, so a producer that never synchronises cannot grow the
    /// list without limit.
    pub(super) fn submit(
        &self,
        command_buffer: &metal::CommandBufferRef,
        label: &'static str,
    ) -> BackendResult<()> {
        if !self.async_dispatch.load(Ordering::Acquire) {
            return commit_and_wait(command_buffer, label).map_err(BackendError::from);
        }
        command_buffer.commit();
        let at_capacity = {
            let mut inflight = self.inflight.lock().map_err(|_| {
                BackendError::DeviceError("in-flight command-buffer mutex poisoned".into())
            })?;
            inflight.push((command_buffer.to_owned(), label));
            inflight.len() >= MAX_INFLIGHT_COMMAND_BUFFERS
        };
        if at_capacity {
            return self.drain_inflight();
        }
        Ok(())
    }

    /// The device-wide command queue shared by every dispatch.
    pub(super) fn device_queue(&self) -> BackendResult<&metal::CommandQueueRef> {
        Ok(self
            .device
            .as_ref()
            .ok_or(BackendError::NotInitialized)?
            .command_queue())
    }

    /// A dispatch planner built from the live device's probed capabilities
    /// (SIMD width, `maxTotalThreadsPerThreadgroup`, threadgroup-memory budget).
    pub(super) fn planner(&self) -> BackendResult<crate::dispatch::DispatchPlanner> {
        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        Ok(crate::dispatch::DispatchPlanner::new(device.capabilities()))
    }

    /// Register an **existing, externally owned** `metal::Buffer` as an oxicuda
    /// handle so compute ops (`gemm`, `copy_*`, custom kernels) can run on it
    /// **zero-copy**, with no host round-trip.
    ///
    /// This is the recommended import entry point for callers that keep their
    /// buffers in a cache (e.g. an `Arc<metal::Buffer>`): pass a reference and
    /// oxicuda takes its **own independent retain** for the lifetime of the
    /// handle. Ownership stays with the caller —
    /// [`free`](oxicuda_backend::ComputeBackend::free) and dropping the backend
    /// release only oxicuda's retain and **never deallocate the caller's
    /// buffer**.
    ///
    /// The returned `u64` handle is interchangeable with one from
    /// [`alloc`](oxicuda_backend::ComputeBackend::alloc) and may be mixed freely
    /// with oxicuda-owned handles in the same op (e.g. external A·B → owned C, or
    /// fully external A·B·C).
    ///
    /// `len_bytes` is the logical length the handle exposes (used to bound host
    /// copies); it must not exceed `buffer.length()`. The buffer **must** belong
    /// to the same `metal::Device` oxicuda initialised with.
    ///
    /// # Errors
    /// * [`BackendError::NotInitialized`] if [`init`](oxicuda_backend::ComputeBackend::init)
    ///   has not been called.
    /// * [`BackendError::InvalidArgument`] if `len_bytes` exceeds the buffer's
    ///   physical length.
    pub fn register_external(
        &self,
        buffer: &MetalExternalBuffer,
        len_bytes: usize,
    ) -> BackendResult<u64> {
        self.check_init()?;
        self.memory()?
            .import_external(buffer, len_bytes)
            .map_err(BackendError::from)
    }

    /// Import an external `metal::Buffer` **by value**, returning a zero-copy
    /// handle. Convenience wrapper over [`register_external`](Self::register_external).
    ///
    /// oxicuda holds the buffer alive for the handle's lifetime via its own
    /// retain; the moved-in value is released once this call returns (its retain
    /// is balanced by the independent retain oxicuda takes). Callers that need to
    /// keep their own reference should clone first or use
    /// [`register_external`](Self::register_external) with a borrow. As with
    /// `register_external`, [`free`](oxicuda_backend::ComputeBackend::free) /
    /// backend drop never deallocate memory still referenced by the caller.
    ///
    /// # Errors
    /// Same as [`register_external`](Self::register_external).
    pub fn import_buffer(
        &self,
        buffer: MetalExternalBuffer,
        len_bytes: usize,
    ) -> BackendResult<u64> {
        // Borrow for registration; `buffer` is dropped (its retain released) at
        // end of scope, leaving oxicuda's own independent retain in the map.
        self.register_external(&buffer, len_bytes)
    }

    pub(super) fn dispatch_unary(
        &self,
        op: UnaryOp,
        input_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        let op_str = match op {
            UnaryOp::Relu => "relu",
            UnaryOp::Sigmoid => "sigmoid",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Abs => "abs",
            UnaryOp::Neg => "neg",
        };
        let memory = self.memory()?;
        let count = to_u32(n, "unary element count")?;
        // Semantic cache key: the MSL source is neither regenerated nor hashed
        // on a cache hit.
        let pipeline = self.cached_pipeline(
            PipelineKey::Builtin {
                kind: "unary",
                op: op_str,
                dtype: "f32",
            },
            "elementwise_f32",
            || crate::msl::elementwise_msl(op_str),
        )?;
        // Resolve handles to independent retains under the buffer-map lock, then
        // encode and block with the lock released (see
        // `MetalComputePipeline::dispatch` for the full rationale).
        let [input_buf, output_buf] = resolve_buffers(memory, [input_ptr, output_ptr])?;
        let plan = self.planner()?.plan_1d(n).map_err(BackendError::from)?;
        let tg_size = clamp_1d_threadgroup(
            u64::from(plan.threads_per_threadgroup[0]),
            pipeline.pipeline_state.max_total_threads_per_threadgroup(),
            pipeline.pipeline_state.thread_execution_width(),
            n as u64,
        );
        let groups = (n as u64).div_ceil(tg_size);
        let command_buffer = self.device_queue()?.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&input_buf), 0);
        encoder.set_buffer(1, Some(&output_buf), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const std::ffi::c_void,
        );
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        self.submit(command_buffer, "unary elementwise")
    }
    pub(super) fn dispatch_binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        let op_str = match op {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Max => "max",
            BinaryOp::Min => "min",
        };
        let memory = self.memory()?;
        let count = to_u32(n, "binary element count")?;
        let pipeline = self.cached_pipeline(
            PipelineKey::Builtin {
                kind: "binary",
                op: op_str,
                dtype: "f32",
            },
            "binary_f32",
            || crate::msl::binary_msl(op_str),
        )?;
        let [a_buf, b_buf, out_buf] = resolve_buffers(memory, [a_ptr, b_ptr, output_ptr])?;
        let plan = self.planner()?.plan_1d(n).map_err(BackendError::from)?;
        let tg_size = clamp_1d_threadgroup(
            u64::from(plan.threads_per_threadgroup[0]),
            pipeline.pipeline_state.max_total_threads_per_threadgroup(),
            pipeline.pipeline_state.thread_execution_width(),
            n as u64,
        );
        let groups = (n as u64).div_ceil(tg_size);
        let command_buffer = self.device_queue()?.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&a_buf), 0);
        encoder.set_buffer(1, Some(&b_buf), 0);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const std::ffi::c_void,
        );
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        self.submit(command_buffer, "binary elementwise")
    }
    pub(super) fn dispatch_reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        // A zero-length dimension makes the reduction degenerate: the old
        // `.max(1)` rewrote the empty products to 1, so the kernel read a tensor
        // that logically holds no elements and wrote a plausible-looking garbage
        // value with `Ok(())`, and `Mean` divided 0.0 by 0.0 into a silent NaN.
        // Fail loudly instead.
        if let Some(pos) = shape.iter().position(|&d| d == 0) {
            return Err(BackendError::InvalidArgument(format!(
                "reduce: shape {shape:?} has a zero-length dimension at index {pos}"
            )));
        }
        let op_str = match op {
            ReduceOp::Sum => "sum",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::Mean => "mean",
        };
        // Every dimension is non-zero, so both empty products are legitimately 1
        // and no `.max(1)` masking is needed.
        let outer_size: usize = shape[..axis].iter().product::<usize>();
        let reduce_size = shape[axis];
        let inner_size: usize = shape[axis + 1..].iter().product::<usize>();
        // A flat whole-tensor reduce produces a single output, which the
        // one-threadgroup-per-output kernel would run on ≤256 threads no matter
        // how large the tensor is. Spread it over the whole GPU instead.
        if outer_size == 1 && inner_size == 1 && reduce_size >= TWO_PASS_REDUCE_THRESHOLD {
            return self.dispatch_reduce_flat(op, input_ptr, output_ptr, reduce_size);
        }
        let memory = self.memory()?;
        let fn_name = crate::msl::reduction_function_name(op_str);
        let pipeline = self.cached_pipeline(
            PipelineKey::Builtin {
                kind: "reduce",
                op: op_str,
                dtype: "f32",
            },
            fn_name,
            || crate::msl::reduction_msl(op_str),
        )?;
        let outer_u32 = to_u32(outer_size, "reduce outer size")?;
        let reduce_u32 = to_u32(reduce_size, "reduce axis length")?;
        let inner_u32 = to_u32(inner_size, "reduce inner size")?;
        let total_groups = outer_size.checked_mul(inner_size).ok_or_else(|| {
            BackendError::InvalidArgument(format!(
                "reduce: output element count overflows for shape {shape:?}"
            ))
        })? as u64;
        // The MSL tree reduction requires a power-of-two threadgroup width, so
        // clamp with `pow2_threadgroup` rather than a bare `min`.
        let tg_size = pow2_threadgroup(
            next_power_of_2(reduce_size).min(REDUCE_TARGET_THREADS as usize) as u64,
            pipeline.pipeline_state.max_total_threads_per_threadgroup(),
        );
        let scratch_bytes = self
            .planner()?
            .threadgroup_scratch_bytes(tg_size as usize, std::mem::size_of::<f32>())
            .map_err(BackendError::from)?;
        let [input_buf, out_buf] = resolve_buffers(memory, [input_ptr, output_ptr])?;
        let command_buffer = self.device_queue()?.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&input_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_bytes(2, 4, &outer_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(3, 4, &reduce_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &inner_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_threadgroup_memory_length(0, scratch_bytes as u64);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(total_groups, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        self.submit(command_buffer, "reduce")
    }

    /// Two-pass reduction of a flat tensor into a single scalar.
    ///
    /// Pass 1 spreads `count` elements over up to [`MAX_REDUCE_GROUPS`]
    /// threadgroups, each writing one partial into a scratch buffer; pass 2
    /// folds the partials with a single threadgroup. Both encoders go into the
    /// **same** command buffer — Metal orders serial compute encoders and
    /// tracks the scratch-buffer write→read hazard automatically — so the whole
    /// reduction still costs one CPU↔GPU round trip.
    fn dispatch_reduce_flat(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        count: usize,
    ) -> BackendResult<()> {
        // `Mean` reuses the additive kernel; only the final pass divides.
        let chunk_op = match op {
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::Sum | ReduceOp::Mean => "sum",
        };
        let fn_name = chunked_reduce_function_name(chunk_op);
        let pipeline = self.cached_pipeline(
            PipelineKey::Builtin {
                kind: "reduce_chunked",
                op: chunk_op,
                dtype: "f32",
            },
            fn_name,
            || chunked_reduce_msl(chunk_op),
        )?;
        let max_threads = pipeline.pipeline_state.max_total_threads_per_threadgroup();
        let tg_size = pow2_threadgroup(REDUCE_TARGET_THREADS, max_threads);
        let groups = count.div_ceil(tg_size as usize).clamp(1, MAX_REDUCE_GROUPS);
        let count_u32 = to_u32(count, "reduce element count")?;
        let groups_u32 = to_u32(groups, "reduce threadgroup count")?;
        let planner = self.planner()?;
        let scratch_bytes = planner
            .threadgroup_scratch_bytes(tg_size as usize, std::mem::size_of::<f32>())
            .map_err(BackendError::from)? as u64;
        // Pass 2 folds `groups` partials with one threadgroup.
        let final_tg = pow2_threadgroup(groups as u64, max_threads);
        let final_scratch = planner
            .threadgroup_scratch_bytes(final_tg as usize, std::mem::size_of::<f32>())
            .map_err(BackendError::from)? as u64;
        let memory = self.memory()?;
        let scratch = memory
            .alloc(groups * std::mem::size_of::<f32>())
            .map_err(BackendError::from)?;
        // The scratch buffer must be released on *every* exit path, so the
        // encode runs in a helper whose result is held, freed around, and only
        // then propagated.
        let outcome = self.encode_reduce_flat(
            &pipeline,
            input_ptr,
            output_ptr,
            scratch,
            ReduceFlatParams {
                count: count_u32,
                groups: groups_u32,
                tg_size,
                final_tg,
                scratch_bytes,
                final_scratch,
                divisor: if op == ReduceOp::Mean {
                    count as f32
                } else {
                    1.0
                },
            },
        );
        // Releasing the scratch goes straight to the memory manager, bypassing
        // `ComputeBackend::free`'s synchronisation point — so in async mode the
        // GPU may still be reading these partials while `free` hands the buffer
        // to the allocator's reuse pool, from which the *next* `alloc` could
        // take it. Await the reduction first; a no-op in synchronous mode.
        let synced = self.drain_inflight();
        if let Err(e) = memory.free(scratch) {
            tracing::warn!("failed to release two-pass reduction scratch buffer: {e}");
        }
        // The encode's own error is the more informative one, so report it
        // first and only fall back to a drain failure if the encode succeeded.
        outcome.and(synced)
    }

    /// Encode and run both passes of [`Self::dispatch_reduce_flat`].
    fn encode_reduce_flat(
        &self,
        pipeline: &MetalComputePipeline,
        input_ptr: u64,
        output_ptr: u64,
        scratch_ptr: u64,
        params: ReduceFlatParams,
    ) -> BackendResult<()> {
        let memory = self.memory()?;
        let [input_buf, out_buf, scratch_buf] =
            resolve_buffers(memory, [input_ptr, output_ptr, scratch_ptr])?;
        let one = 1u32;
        let unit_divisor = 1.0f32;
        let command_buffer = self.device_queue()?.new_command_buffer();
        {
            // Pass 1: input → per-threadgroup partials (never scaled).
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
            encoder.set_buffer(0, Some(&input_buf), 0);
            encoder.set_buffer(1, Some(&scratch_buf), 0);
            encoder.set_bytes(2, 4, &params.count as *const u32 as *const std::ffi::c_void);
            encoder.set_bytes(
                3,
                4,
                &params.groups as *const u32 as *const std::ffi::c_void,
            );
            encoder.set_bytes(4, 4, &unit_divisor as *const f32 as *const std::ffi::c_void);
            encoder.set_threadgroup_memory_length(0, params.scratch_bytes);
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(u64::from(params.groups), 1, 1),
                metal::MTLSize::new(params.tg_size, 1, 1),
            );
            encoder.end_encoding();
        }
        {
            // Pass 2: partials → the single output element, applying the Mean
            // divisor (the *original* element count, matching
            // `reduction_msl`'s `sdata[0] / float(reduce_size)`).
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
            encoder.set_buffer(0, Some(&scratch_buf), 0);
            encoder.set_buffer(1, Some(&out_buf), 0);
            encoder.set_bytes(
                2,
                4,
                &params.groups as *const u32 as *const std::ffi::c_void,
            );
            encoder.set_bytes(3, 4, &one as *const u32 as *const std::ffi::c_void);
            encoder.set_bytes(
                4,
                4,
                &params.divisor as *const f32 as *const std::ffi::c_void,
            );
            encoder.set_threadgroup_memory_length(0, params.final_scratch);
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(1, 1, 1),
                metal::MTLSize::new(params.final_tg, 1, 1),
            );
            encoder.end_encoding();
        }
        self.submit(command_buffer, "reduce (two-pass)")
    }
    /// Threadgroup shape and grid for a v2 GEMM tile decomposition.
    ///
    /// The 16×16 threadgroup is a **hard precondition** of
    /// [`crate::msl::gemm_msl_v2`], not a tuning knob: the cooperative staging
    /// loop strides by a hardcoded 256 threads and every thread owns a fixed
    /// 2×2 slice of the 32×32 output tile. Deriving it from
    /// `max_total_threads_per_threadgroup()` — as the element-wise dispatchers
    /// legitimately do — would silently corrupt the result, so this asserts the
    /// shape through the kernel module's own validator instead.
    fn gemm_v2_launch(
        &self,
        pipeline: &MetalComputePipeline,
        m: usize,
        n: usize,
    ) -> BackendResult<(metal::MTLSize, metal::MTLSize)> {
        crate::msl::validate_gemm_v2_dispatch(
            crate::msl::GEMM_V2_THREADS_X,
            crate::msl::GEMM_V2_THREADS_Y,
        )
        .map_err(BackendError::from)?;
        let threads = (crate::msl::GEMM_V2_THREADS_X * crate::msl::GEMM_V2_THREADS_Y) as u64;
        let max_threads = pipeline.pipeline_state.max_total_threads_per_threadgroup();
        if threads > max_threads {
            return Err(BackendError::Unsupported(format!(
                "the tiled Metal GEMM needs a {threads}-thread threadgroup, but this pipeline \
                 allows only {max_threads}"
            )));
        }
        let groups_x = (n as u64).div_ceil(crate::msl::GEMM_V2_TILE_N as u64);
        let groups_y = (m as u64).div_ceil(crate::msl::GEMM_V2_TILE_M as u64);
        Ok((
            metal::MTLSize::new(groups_x, groups_y, 1),
            metal::MTLSize::new(
                crate::msl::GEMM_V2_THREADS_X as u64,
                crate::msl::GEMM_V2_THREADS_Y as u64,
                1,
            ),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_gemm(
        &self,
        trans_a: BackendTranspose,
        trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        lda: usize,
        b_ptr: u64,
        ldb: usize,
        beta: f64,
        c_ptr: u64,
        ldc: usize,
    ) -> BackendResult<()> {
        let memory = self.memory()?;
        let dtype = crate::msl::GemmDtype::F32;
        let pipeline = self.cached_pipeline(
            PipelineKey::Builtin {
                kind: "gemm_v2",
                op: "tiled",
                dtype: "f32",
            },
            crate::msl::gemm_v2_function_name(dtype),
            || crate::msl::gemm_msl_v2(dtype),
        )?;
        let params = gemm_params_v2(
            "gemm",
            trans_a,
            trans_b,
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            alpha as f32,
            beta as f32,
        )?;
        let [a_buf, b_buf, c_buf] = resolve_buffers(memory, [a_ptr, b_ptr, c_ptr])?;
        let (grid, threadgroup) = self.gemm_v2_launch(&pipeline, m, n)?;
        let command_buffer = self.device_queue()?.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&a_buf), 0);
        encoder.set_buffer(1, Some(&b_buf), 0);
        encoder.set_buffer(2, Some(&c_buf), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<GemmParamsV2>() as u64,
            &params as *const GemmParamsV2 as *const std::ffi::c_void,
        );
        // The staging tiles are statically declared inside the kernel — binding
        // threadgroup memory at slot 0 here would be a validation error.
        encoder.dispatch_thread_groups(grid, threadgroup);
        encoder.end_encoding();
        self.submit(command_buffer, "gemm")
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_batched_gemm(
        &self,
        trans_a: BackendTranspose,
        trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        lda: usize,
        stride_a: usize,
        b_ptr: u64,
        ldb: usize,
        stride_b: usize,
        beta: f64,
        c_ptr: u64,
        ldc: usize,
        stride_c: usize,
        batch_count: usize,
    ) -> BackendResult<()> {
        let memory = self.memory()?;
        let dtype = crate::msl::GemmDtype::F32;
        let pipeline = self.cached_pipeline(
            PipelineKey::Builtin {
                kind: "batched_gemm_v2",
                op: "tiled",
                dtype: "f32",
            },
            crate::msl::batched_gemm_v2_function_name(dtype),
            || crate::msl::batched_gemm_msl_v2(dtype),
        )?;
        // Batch strides are element counts, not byte counts, and an 8 GiB f32
        // slice already exceeds u32 on a unified-memory Mac — check, do not cast.
        let params = BatchedGemmParamsV2 {
            base: gemm_params_v2(
                "batched gemm",
                trans_a,
                trans_b,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                alpha as f32,
                beta as f32,
            )?,
            batch_count: to_u32(batch_count, "batched gemm batch_count")?,
            stride_a: to_u32(stride_a, "batched gemm stride_a")?,
            stride_b: to_u32(stride_b, "batched gemm stride_b")?,
            stride_c: to_u32(stride_c, "batched gemm stride_c")?,
        };
        let [a_buf, b_buf, c_buf] = resolve_buffers(memory, [a_ptr, b_ptr, c_ptr])?;
        let (grid, threadgroup) = self.gemm_v2_launch(&pipeline, m, n)?;
        let grid = metal::MTLSize::new(grid.width, grid.height, batch_count as u64);
        let command_buffer = self.device_queue()?.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&a_buf), 0);
        encoder.set_buffer(1, Some(&b_buf), 0);
        encoder.set_buffer(2, Some(&c_buf), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<BatchedGemmParamsV2>() as u64,
            &params as *const BatchedGemmParamsV2 as *const std::ffi::c_void,
        );
        encoder.dispatch_thread_groups(grid, threadgroup);
        encoder.end_encoding();
        self.submit(command_buffer, "batched gemm")
    }
    /// Half-precision GEMM: `C = alpha * A * B + beta * C` using FP16 storage.
    ///
    /// This is an inherent method (not on the `ComputeBackend` trait) since
    /// the trait operates on f32/f64 data. Element size is 2 bytes (half).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
        &self,
        trans_a: BackendTranspose,
        trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a_ptr: u64,
        lda: usize,
        b_ptr: u64,
        ldb: usize,
        beta: f32,
        c_ptr: u64,
        ldc: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        validate_gemm_layout(trans_a, trans_b, m, n, k, lda, ldb, ldc)?;
        let memory = self.memory()?;
        let dtype = crate::msl::GemmDtype::F16;
        let pipeline = self.cached_pipeline(
            PipelineKey::Builtin {
                kind: "gemm_v2",
                op: "tiled",
                dtype: "f16",
            },
            crate::msl::gemm_v2_function_name(dtype),
            || crate::msl::gemm_msl_v2(dtype),
        )?;
        let params = gemm_params_v2(
            "gemm_f16", trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha, beta,
        )?;
        let [a_buf, b_buf, c_buf] = resolve_buffers(memory, [a_ptr, b_ptr, c_ptr])?;
        let (grid, threadgroup) = self.gemm_v2_launch(&pipeline, m, n)?;
        let command_buffer = self.device_queue()?.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&a_buf), 0);
        encoder.set_buffer(1, Some(&b_buf), 0);
        encoder.set_buffer(2, Some(&c_buf), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<GemmParamsV2>() as u64,
            &params as *const GemmParamsV2 as *const std::ffi::c_void,
        );
        encoder.dispatch_thread_groups(grid, threadgroup);
        encoder.end_encoding();
        self.submit(command_buffer, "gemm_f16")
    }
}
#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    /// Zero-copy import of an externally owned Metal buffer.
    ///
    /// Present on every platform so callers need no `cfg` gate; off macOS there
    /// is no Metal, so this always returns [`BackendError::DeviceError`]
    /// wrapping [`crate::error::MetalError::UnsupportedPlatform`].
    ///
    /// # Errors
    /// Always [`BackendError::DeviceError`] on non-macOS targets (or
    /// [`BackendError::NotInitialized`] if `init` was never called, which is
    /// itself always the case off macOS).
    pub fn register_external(
        &self,
        _buffer: &MetalExternalBuffer,
        _len_bytes: usize,
    ) -> BackendResult<u64> {
        self.check_init()?;
        Err(BackendError::from(
            crate::error::MetalError::UnsupportedPlatform,
        ))
    }

    /// Zero-copy import of an externally owned Metal buffer, taken by value.
    ///
    /// Present on every platform so callers need no `cfg` gate; off macOS this
    /// always returns [`BackendError::DeviceError`] wrapping
    /// [`crate::error::MetalError::UnsupportedPlatform`].
    ///
    /// # Errors
    /// Same as [`register_external`](Self::register_external).
    pub fn import_buffer(
        &self,
        buffer: MetalExternalBuffer,
        len_bytes: usize,
    ) -> BackendResult<u64> {
        self.register_external(&buffer, len_bytes)
    }

    pub(super) fn dispatch_unary(
        &self,
        _op: UnaryOp,
        _input_ptr: u64,
        _output_ptr: u64,
        _n: usize,
    ) -> BackendResult<()> {
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }
    pub(super) fn dispatch_binary(
        &self,
        _op: BinaryOp,
        _a_ptr: u64,
        _b_ptr: u64,
        _output_ptr: u64,
        _n: usize,
    ) -> BackendResult<()> {
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }
    pub(super) fn dispatch_reduce(
        &self,
        _op: ReduceOp,
        _input_ptr: u64,
        _output_ptr: u64,
        _shape: &[usize],
        _axis: usize,
    ) -> BackendResult<()> {
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_gemm(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f64,
        _a_ptr: u64,
        _lda: usize,
        _b_ptr: u64,
        _ldb: usize,
        _beta: f64,
        _c_ptr: u64,
        _ldc: usize,
    ) -> BackendResult<()> {
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_batched_gemm(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f64,
        _a_ptr: u64,
        _lda: usize,
        _stride_a: usize,
        _b_ptr: u64,
        _ldb: usize,
        _stride_b: usize,
        _beta: f64,
        _c_ptr: u64,
        _ldc: usize,
        _stride_c: usize,
        _batch_count: usize,
    ) -> BackendResult<()> {
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }
    /// Half-precision GEMM: `C = alpha * A * B + beta * C` using FP16 storage.
    ///
    /// This is an inherent method (not on the `ComputeBackend` trait) since
    /// the trait operates on f32/f64 data. Element size is 2 bytes (half).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a_ptr: u64,
        _lda: usize,
        _b_ptr: u64,
        _ldb: usize,
        _beta: f32,
        _c_ptr: u64,
        _ldc: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::{PipelineKey, packed_gemm_lds, validate_gemm_layout};
    use oxicuda_backend::{BackendError, BackendTranspose};

    #[test]
    fn builtin_keys_discriminate_kind_op_and_dtype() {
        let base = PipelineKey::Builtin {
            kind: "reduce",
            op: "sum",
            dtype: "f32",
        };
        // The one-pass and two-pass reduction kernels share an op string but are
        // different kernels — the `kind` field is what keeps them apart.
        assert_ne!(
            base,
            PipelineKey::Builtin {
                kind: "reduce_chunked",
                op: "sum",
                dtype: "f32",
            }
        );
        assert_ne!(
            base,
            PipelineKey::Builtin {
                kind: "reduce",
                op: "max",
                dtype: "f32",
            }
        );
        assert_ne!(
            base,
            PipelineKey::Builtin {
                kind: "reduce",
                op: "sum",
                dtype: "f16",
            }
        );
        assert_eq!(base, base.clone());
    }

    #[test]
    fn custom_keys_compare_full_source_not_a_hash() {
        // Two different sources sharing one entry-point name must never be
        // treated as the same pipeline.
        let a = PipelineKey::Custom {
            function_name: "k".into(),
            source: "kernel void k() { /* a */ }".into(),
        };
        let b = PipelineKey::Custom {
            function_name: "k".into(),
            source: "kernel void k() { /* b */ }".into(),
        };
        assert_ne!(a, b);
        assert_eq!(a, a.clone());
        // A builtin and a custom key never collide either.
        assert_ne!(
            a,
            PipelineKey::Builtin {
                kind: "unary",
                op: "k",
                dtype: "f32",
            }
        );
    }

    /// m=5, n=4, k=3 for every transpose combination.
    const M: usize = 5;
    const N: usize = 4;
    const K: usize = 3;

    #[test]
    fn natural_layout_accepted() {
        // NoTrans/NoTrans with lda=k, ldb=n, ldc=n is the tightly-packed case.
        assert!(
            validate_gemm_layout(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                M,
                N,
                K,
                /*lda*/ K,
                /*ldb*/ N,
                /*ldc*/ N,
            )
            .is_ok()
        );
    }

    #[test]
    fn packed_leading_dims_follow_the_transpose_flags() {
        // op(A) is m×k, so a stored Aᵀ is k×m and its row stride is m — not k.
        assert_eq!(
            packed_gemm_lds(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                M,
                N,
                K
            ),
            (K, N, N)
        );
        assert_eq!(
            packed_gemm_lds(BackendTranspose::Trans, BackendTranspose::NoTrans, M, N, K),
            (M, N, N)
        );
        assert_eq!(
            packed_gemm_lds(BackendTranspose::NoTrans, BackendTranspose::Trans, M, N, K),
            (K, K, N)
        );
        assert_eq!(
            packed_gemm_lds(BackendTranspose::Trans, BackendTranspose::Trans, M, N, K),
            (M, K, N)
        );
    }

    #[test]
    fn every_transpose_combination_is_accepted() {
        // The v2 kernels take trans_a/trans_b as runtime flags, so `y = x @ W.T`
        // and the other three combinations must all be honoured rather than
        // rejected — this is the behaviour change that closes the "Metal GEMM
        // rejects transpose" gap.
        for trans_a in [BackendTranspose::NoTrans, BackendTranspose::Trans] {
            for trans_b in [BackendTranspose::NoTrans, BackendTranspose::Trans] {
                let (lda, ldb, ldc) = packed_gemm_lds(trans_a, trans_b, M, N, K);
                assert!(
                    validate_gemm_layout(trans_a, trans_b, M, N, K, lda, ldb, ldc).is_ok(),
                    "packed {trans_a:?}/{trans_b:?} must be accepted"
                );
                // A padded / sub-matrix view is legal: the kernel reads the
                // leading dimension from the parameter buffer.
                assert!(
                    validate_gemm_layout(trans_a, trans_b, M, N, K, lda + 7, ldb + 3, ldc + 1)
                        .is_ok(),
                    "padded {trans_a:?}/{trans_b:?} must be accepted"
                );
            }
        }
    }

    #[test]
    fn leading_dim_below_the_stored_row_width_is_rejected() {
        // Smaller than the stored row makes the kernel read across row
        // boundaries — malformed argument, not a capability gap.
        for (lda, ldb, ldc) in [(K - 1, N, N), (K, N - 1, N), (K, N, N - 1)] {
            assert!(
                matches!(
                    validate_gemm_layout(
                        BackendTranspose::NoTrans,
                        BackendTranspose::NoTrans,
                        M,
                        N,
                        K,
                        lda,
                        ldb,
                        ldc,
                    ),
                    Err(BackendError::InvalidArgument(_))
                ),
                "lda={lda} ldb={ldb} ldc={ldc} must be rejected"
            );
        }
        // Under trans_a the minimum lda is m, so the previously-legal lda=k is
        // now too small whenever k < m.
        assert!(matches!(
            validate_gemm_layout(
                BackendTranspose::Trans,
                BackendTranspose::NoTrans,
                M,
                N,
                K,
                /*lda*/ K,
                N,
                N,
            ),
            Err(BackendError::InvalidArgument(_))
        ));
    }
}
