//! Abstract compute backend for GPU-accelerated operations.
//!
//! This module is the facade's single entry point to the compute-backend
//! abstraction. It re-exports the whole surface of `oxicuda-backend` — the
//! [`ComputeBackend`](crate::backend::ComputeBackend) trait, the always-available [`CpuBackend`](crate::backend::CpuBackend), the
//! [`BackendRegistry`](crate::backend::BackendRegistry) control plane and its [`Capabilities`](crate::backend::Capabilities) types — and adds
//! [`CudaBackend`](crate::backend::CudaBackend), the CUDA implementation built on `oxicuda-driver`.
//!
//! # Concrete backends
//!
//! | Type | Availability |
//! |------|--------------|
//! | [`CpuBackend`](crate::backend::CpuBackend) | always (pure-Rust host reference path) |
//! | [`NullBackend`](crate::backend::NullBackend) | always (refuses every op; for dispatch tests) |
//! | [`CudaBackend`](crate::backend::CudaBackend) | always compiled; usable where `libcuda` loads |
//! | `MetalBackend` | feature `metal` (works on macOS) |
//! | `WebGpuBackend` | feature `webgpu` |
//! | `VulkanBackend` | feature `vulkan` |
//! | `RocmBackend` | feature `rocm` |
//! | `LevelZeroBackend` | feature `level-zero` |
//!
//! Rather than picking one by hand, prefer [`crate::compute::default_backend`],
//! which probes what this machine actually has and returns the best one already
//! initialised.

use std::sync::Mutex;

use oxicuda_driver::device::Device;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::primary_context::PrimaryContext;

pub use oxicuda_backend::{
    BackendEntry, BackendError, BackendKind, BackendRegistry, BackendResult, BackendTranspose,
    BinaryOp, Capabilities, ComputeBackend, CpuBackend, DeviceInfo, MemoryKind, MixedPrecision,
    NullBackend, OpClass, ReduceOp, SelectionRequest, TileShape, UnaryOp, default_tile_for,
};

/// Apple Metal compute backend (`oxicuda-metal`).
///
/// Compiled in with the `metal` feature; genuinely executes on the GPU on
/// macOS and returns [`BackendError::DeviceError`] elsewhere.
#[cfg(feature = "metal")]
pub use oxicuda_metal::MetalBackend;

/// WebGPU compute backend (`oxicuda-webgpu`), portable via `wgpu`.
#[cfg(feature = "webgpu")]
pub use oxicuda_webgpu::WebGpuBackend;

/// Vulkan compute backend (`oxicuda-vulkan`).
#[cfg(feature = "vulkan")]
pub use oxicuda_vulkan::VulkanBackend;

/// AMD ROCm/HIP compute backend (`oxicuda-rocm`).
#[cfg(feature = "rocm")]
pub use oxicuda_rocm::RocmBackend;

/// Intel Level Zero compute backend (`oxicuda-levelzero`).
#[cfg(feature = "level-zero")]
pub use oxicuda_levelzero::LevelZeroBackend;

#[cfg(feature = "ptx")]
mod ptx_ops;

/// The backend's cached PTX-launch stream plus the borrowed primary-context
/// lifetime token it was created in (see
/// [`primary_context_token`]), or `None` before the first PTX-backed op has
/// run. Aliased to keep the
/// `CudaBackend` field and [`ptx_ops`]'s accessor readable and to avoid a
/// `clippy::type_complexity` warning on the nested `Arc`/tuple/`Option`.
#[cfg(feature = "ptx")]
type PtxStreamState = Option<(
    std::sync::Arc<oxicuda_driver::Context>,
    oxicuda_driver::Stream,
)>;

// ─── CudaBackend implementation ─────────────────────────────

/// CUDA backend implementation.
///
/// Delegates to `oxicuda-driver`, `oxicuda-blas`, and `oxicuda-dnn` for
/// actual GPU computation. When those sub-crate features are not enabled,
/// the corresponding operations return [`BackendError::Unsupported`].
///
/// On systems without a CUDA GPU, [`init`](ComputeBackend::init) succeeds
/// (so the type is always usable) but GPU operations such as
/// [`alloc`](ComputeBackend::alloc) and [`copy_htod`](ComputeBackend::copy_htod)
/// return [`BackendError::DeviceError`].
///
/// # Context model
///
/// During [`init`](ComputeBackend::init) the backend retains the device's
/// **primary context** and makes it current. All device memory
/// ([`alloc`](ComputeBackend::alloc), [`copy_htod`](ComputeBackend::copy_htod),
/// …) and all compute kernels run inside that single context, so device
/// pointers are valid across every operation. The BLAS/DNN handles needed by
/// [`gemm`](ComputeBackend::gemm), [`conv2d_forward`](ComputeBackend::conv2d_forward),
/// and [`attention`](ComputeBackend::attention) require an `Arc<Context>`, as
/// does the `ptx_ops` launch stream behind [`unary`](ComputeBackend::unary),
/// [`binary`](ComputeBackend::binary), and [`reduce`](ComputeBackend::reduce);
/// the backend hands each of them a **non-owning wrapper around that same
/// primary context** (internally, `primary_context_token`) as their lifetime
/// token, so every handle, stream, and JIT-loaded module lives in the one
/// context that owns the device memory.
///
/// # Example
///
/// ```rust
/// use oxicuda::backend::{CudaBackend, ComputeBackend};
///
/// let mut backend = CudaBackend::new();
/// assert!(!backend.is_initialized());
/// backend.init().expect("init should succeed");
/// assert!(backend.is_initialized());
/// assert_eq!(backend.name(), "cuda");
/// ```
pub struct CudaBackend {
    initialized: bool,
    /// Retained primary CUDA context, set when a GPU is available. This is the
    /// single context that owns all device memory and runs all kernels.
    context: Mutex<Option<PrimaryContext>>,
    /// Cached BLAS handle plus the context lifetime-token it is bound to.
    ///
    /// Created lazily on the first [`gemm`](ComputeBackend::gemm) call (see
    /// [`with_blas_handle`]) and reused on every subsequent call, so the
    /// handle's stream, one-time compute-capability query, and JIT-compiled
    /// kernel/GEMM-tile caches are built once per backend instead of on every
    /// GEMM invocation.
    #[cfg(feature = "blas")]
    blas_state: Mutex<
        Option<(
            std::sync::Arc<oxicuda_driver::Context>,
            oxicuda_blas::BlasHandle,
        )>,
    >,
    /// Cached DNN handle plus its context lifetime-token, analogous to
    /// [`blas_state`](Self::blas_state) for
    /// [`conv2d_forward`](ComputeBackend::conv2d_forward) and
    /// [`attention`](ComputeBackend::attention) (see [`with_dnn_handle`]).
    #[cfg(feature = "dnn")]
    dnn_state: Mutex<
        Option<(
            std::sync::Arc<oxicuda_driver::Context>,
            oxicuda_dnn::handle::DnnHandle,
        )>,
    >,
    /// Cache of JIT-compiled PTX kernels for the elementwise/reduction ops in
    /// [`ptx_ops`], keyed by kernel name and target SM architecture, so a
    /// repeated op JIT-compiles its module only once per backend rather than
    /// on every call.
    #[cfg(feature = "ptx")]
    kernel_cache: Mutex<
        std::collections::HashMap<
            (String, oxicuda_ptx::arch::SmVersion),
            std::sync::Arc<oxicuda_launch::Kernel>,
        >,
    >,
    /// Cached stream (plus the borrowed primary-context lifetime token it was
    /// created in) used to launch the [`ptx_ops`] elementwise/reduction
    /// kernels, created lazily on first use and reused thereafter instead of a
    /// fresh stream per call.
    #[cfg(feature = "ptx")]
    ptx_stream: Mutex<PtxStreamState>,
}

/// Manual [`std::fmt::Debug`] implementation: the cached BLAS/DNN handles and
/// PTX kernel/stream state (gated behind the `blas`/`dnn`/`ptx` features) do
/// not themselves implement `Debug`, so this reports only the
/// backend-independent summary fields instead of deriving.
impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("initialized", &self.initialized)
            .field("has_gpu_context", &self.has_gpu_context())
            .finish_non_exhaustive()
    }
}

/// Explicit, order-sensitive teardown of every cached GPU resource.
///
/// Every cached BLAS/DNN handle, JIT-compiled kernel, and PTX launch stream
/// lives inside the backend's **primary** context (see the `# Context model`
/// section on [`CudaBackend`]), held only through a non-owning borrowed
/// lifetime token that never destroys it. If the primary context
/// were released *before* those cached resources are torn down, the driver
/// calls their destructors make (`cuModuleUnload`, `cuStreamDestroy`, …)
/// would run against an already-destroyed context: at best this surfaces as
/// `CUDA_ERROR_CONTEXT_IS_DESTROYED`, at worst the driver crashes outright
/// (observed in practice as a SIGSEGV inside `libcuda.so` during
/// `Stream::drop`). Rust drops a struct's fields in declaration order, so
/// relying on field order alone to guarantee this would silently break if a
/// field were ever reordered; this `Drop` impl makes the required order
/// explicit instead: drain every cached resource first, and release the
/// primary context last.
impl Drop for CudaBackend {
    fn drop(&mut self) {
        #[cfg(feature = "ptx")]
        {
            if let Ok(mut guard) = self.ptx_stream.lock() {
                guard.take();
            }
            if let Ok(mut guard) = self.kernel_cache.lock() {
                guard.clear();
            }
        }
        #[cfg(feature = "dnn")]
        {
            if let Ok(mut guard) = self.dnn_state.lock() {
                guard.take();
            }
        }
        #[cfg(feature = "blas")]
        {
            if let Ok(mut guard) = self.blas_state.lock() {
                guard.take();
            }
        }
        // The primary context is released last, once every cached resource
        // that actually runs inside it has already been torn down above.
        if let Ok(mut guard) = self.context.lock() {
            guard.take();
        }
    }
}

impl CudaBackend {
    /// Create a new, uninitialized CUDA backend.
    #[must_use]
    pub fn new() -> Self {
        Self {
            initialized: false,
            context: Mutex::new(None),
            #[cfg(feature = "blas")]
            blas_state: Mutex::new(None),
            #[cfg(feature = "dnn")]
            dnn_state: Mutex::new(None),
            #[cfg(feature = "ptx")]
            kernel_cache: Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "ptx")]
            ptx_stream: Mutex::new(None),
        }
    }

    /// Check that the backend is initialized, returning an error if not.
    fn check_init(&self) -> BackendResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(BackendError::NotInitialized)
        }
    }

    /// Returns `true` if a live GPU context was successfully retained during
    /// [`init`](ComputeBackend::init).
    fn has_gpu_context(&self) -> bool {
        self.context.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Ensures the retained primary context is current on the calling thread
    /// and returns the GPU [`Device`].
    ///
    /// Compute methods may be invoked from any thread; the CUDA current-context
    /// is per-thread, so this re-binds the primary context before every GPU
    /// operation. Returns [`BackendError::DeviceError`] when no GPU context is
    /// available.
    ///
    /// Only compiled when a GPU compute path (`blas`, `dnn`, or `ptx`) that
    /// needs it is enabled.
    #[cfg(any(feature = "blas", feature = "dnn", feature = "ptx"))]
    fn activate_gpu(&self) -> BackendResult<Device> {
        let guard = self
            .context
            .lock()
            .map_err(|_| BackendError::DeviceError("backend context lock poisoned".into()))?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| BackendError::DeviceError("no CUDA GPU context available".into()))?;
        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe { (api.cu_ctx_set_current)(ctx.raw()) })
            .map_err(|e| BackendError::DeviceError(e.to_string()))?;
        Ok(*ctx.device())
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }

        // Attempt real CUDA initialisation.  On systems without a GPU the
        // driver library will not be present and we fall back gracefully:
        // the backend is marked as initialised so callers can branch on
        // `is_initialized()`, but GPU operations will return a DeviceError.
        if let Ok(()) = oxicuda_driver::init() {
            if let Ok(dev) = Device::get(0) {
                if let Ok(ctx) = PrimaryContext::retain(&dev) {
                    // Make the primary context current on this thread so that
                    // subsequent driver calls (cuMemAlloc, cuMemcpy, …) use it.
                    if let Ok(api) = try_driver() {
                        let raw = ctx.raw();
                        // Ignore the error from cuCtxSetCurrent: the context
                        // may already be current, or it may be a non-fatal
                        // warning.  The important thing is that allocation
                        // attempts will produce a proper error if the context
                        // is wrong rather than silently succeeding.
                        let _ =
                            oxicuda_driver::error::check(unsafe { (api.cu_ctx_set_current)(raw) });
                    }
                    if let Ok(mut guard) = self.context.lock() {
                        *guard = Some(ctx);
                    }
                }
            }
        }

        self.initialized = true;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn gemm(
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
        self.check_init()?;
        #[cfg(feature = "blas")]
        {
            gemm_impl(
                self, trans_a, trans_b, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc,
            )
        }
        #[cfg(not(feature = "blas"))]
        {
            let _ = (
                trans_a, trans_b, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc,
            );
            Err(BackendError::Unsupported(
                "GEMM requires the 'blas' feature".into(),
            ))
        }
    }

    fn conv2d_forward(
        &self,
        input_ptr: u64,
        input_shape: &[usize],
        filter_ptr: u64,
        filter_shape: &[usize],
        output_ptr: u64,
        output_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> BackendResult<()> {
        self.check_init()?;

        // Validate shapes
        if input_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(format!(
                "input_shape must have 4 elements (NCHW), got {}",
                input_shape.len()
            )));
        }
        if filter_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(format!(
                "filter_shape must have 4 elements (KCRS), got {}",
                filter_shape.len()
            )));
        }
        if output_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(format!(
                "output_shape must have 4 elements (NKPQ), got {}",
                output_shape.len()
            )));
        }
        if stride.len() != 2 {
            return Err(BackendError::InvalidArgument(format!(
                "stride must have 2 elements, got {}",
                stride.len()
            )));
        }
        if padding.len() != 2 {
            return Err(BackendError::InvalidArgument(format!(
                "padding must have 2 elements, got {}",
                padding.len()
            )));
        }

        #[cfg(feature = "dnn")]
        {
            conv2d_forward_impl(
                self,
                input_ptr,
                input_shape,
                filter_ptr,
                filter_shape,
                output_ptr,
                output_shape,
                stride,
                padding,
            )
        }
        #[cfg(not(feature = "dnn"))]
        {
            let _ = (input_ptr, filter_ptr, output_ptr);
            Err(BackendError::Unsupported(
                "conv2d_forward requires the 'dnn' feature".into(),
            ))
        }
    }

    fn attention(
        &self,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        o_ptr: u64,
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_kv: usize,
        head_dim: usize,
        scale: f64,
        causal: bool,
    ) -> BackendResult<()> {
        self.check_init()?;

        if seq_q == 0 || seq_kv == 0 || head_dim == 0 {
            return Err(BackendError::InvalidArgument(
                "sequence lengths and head_dim must be > 0".into(),
            ));
        }
        if scale <= 0.0 || !scale.is_finite() {
            return Err(BackendError::InvalidArgument(format!(
                "scale must be a positive finite number, got {scale}"
            )));
        }

        #[cfg(feature = "dnn")]
        {
            attention_impl(
                self, q_ptr, k_ptr, v_ptr, o_ptr, batch, heads, seq_q, seq_kv, head_dim, scale,
                causal,
            )
        }
        #[cfg(not(feature = "dnn"))]
        {
            let _ = (
                q_ptr, k_ptr, v_ptr, o_ptr, batch, heads, seq_q, seq_kv, head_dim, scale, causal,
            );
            Err(BackendError::Unsupported(
                "attention requires the 'dnn' feature".into(),
            ))
        }
    }

    fn reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        self.check_init()?;

        if shape.is_empty() {
            return Err(BackendError::InvalidArgument(
                "shape must not be empty".into(),
            ));
        }
        if axis >= shape.len() {
            return Err(BackendError::InvalidArgument(format!(
                "axis {} out of bounds for shape with {} dimensions",
                axis,
                shape.len()
            )));
        }

        #[cfg(feature = "ptx")]
        {
            ptx_ops::reduce_axis(self, op, input_ptr, output_ptr, shape, axis)
        }
        #[cfg(not(feature = "ptx"))]
        {
            let _ = (op, input_ptr, output_ptr);
            Err(BackendError::Unsupported(
                "reduce requires the 'ptx' feature".into(),
            ))
        }
    }

    fn unary(&self, op: UnaryOp, input_ptr: u64, output_ptr: u64, n: usize) -> BackendResult<()> {
        self.check_init()?;

        if n == 0 {
            return Ok(()); // no-op on empty tensor
        }

        #[cfg(feature = "ptx")]
        {
            ptx_ops::unary_elementwise(self, op, input_ptr, output_ptr, n)
        }
        #[cfg(not(feature = "ptx"))]
        {
            let _ = (op, input_ptr, output_ptr);
            Err(BackendError::Unsupported(
                "unary requires the 'ptx' feature".into(),
            ))
        }
    }

    fn binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        self.check_init()?;

        if n == 0 {
            return Ok(()); // no-op on empty tensor
        }

        #[cfg(feature = "ptx")]
        {
            ptx_ops::binary_elementwise(self, op, a_ptr, b_ptr, output_ptr, n)
        }
        #[cfg(not(feature = "ptx"))]
        {
            let _ = (op, a_ptr, b_ptr, output_ptr);
            Err(BackendError::Unsupported(
                "binary requires the 'ptx' feature".into(),
            ))
        }
    }

    fn synchronize(&self) -> BackendResult<()> {
        self.check_init()?;
        // If no GPU context is available (e.g. no CUDA hardware), there is
        // nothing to synchronise — return Ok as a no-op.
        if !self.has_gpu_context() {
            return Ok(());
        }
        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe { (api.cu_ctx_synchronize)() })
            .map_err(|e| BackendError::DeviceError(e.to_string()))
    }

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        self.check_init()?;

        if bytes == 0 {
            return Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into(),
            ));
        }

        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        let mut ptr: CUdeviceptr = 0;
        oxicuda_driver::error::check(unsafe { (api.cu_mem_alloc_v2)(&mut ptr, bytes) }).map_err(
            |e| match e {
                oxicuda_driver::CudaError::OutOfMemory => BackendError::OutOfMemory,
                other => BackendError::DeviceError(other.to_string()),
            },
        )?;
        Ok(ptr)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.check_init()?;
        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe { (api.cu_mem_free_v2)(ptr) })
            .map_err(|e| BackendError::DeviceError(e.to_string()))
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        self.check_init()?;

        if src.is_empty() {
            return Ok(());
        }

        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe {
            (api.cu_memcpy_htod_v2)(dst, src.as_ptr().cast(), src.len())
        })
        .map_err(|e| BackendError::DeviceError(e.to_string()))
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        self.check_init()?;

        if dst.is_empty() {
            return Ok(());
        }

        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe {
            (api.cu_memcpy_dtoh_v2)(dst.as_mut_ptr().cast(), src, dst.len())
        })
        .map_err(|e| BackendError::DeviceError(e.to_string()))
    }
}

// ─── BLAS GEMM wiring (feature = "blas") ────────────────────

/// Builds the `Arc<Context>` lifetime token that a BLAS/DNN handle — or the
/// [`ptx_ops`] launch stream — binds to, wrapping the backend's retained
/// **primary context** without owning it.
///
/// `oxicuda-blas` / `oxicuda-dnn` handles take an `Arc<Context>` and create
/// their stream *in that context* (see `Stream::new`), as does
/// `Stream::new` itself for the PTX launch stream; the backend's persistent
/// memory context, however, is a [`PrimaryContext`]. Rather than stand up a
/// throwaway regular context (which would leave the stream bound to a context
/// that owns none of the device memory and none of the JIT-loaded modules —
/// `cuLaunchKernel` rejects such a mismatch with
/// `CUDA_ERROR_INVALID_HANDLE`), this wraps the primary context's raw handle
/// in a non-owning [`Context`](oxicuda_driver::Context) via
/// [`Context::from_raw_borrowed`](oxicuda_driver::Context::from_raw_borrowed):
/// the stream and its kernels then run in the very context where all device
/// memory lives, and the wrapper never destroys the primary context on drop
/// (that is [`PrimaryContext`]'s responsibility, ordered last in
/// [`CudaBackend`]'s `Drop`).
///
/// Called at most once per cached resource per backend, the first time
/// [`with_blas_handle`], [`with_dnn_handle`], or `ptx_ops`'s
/// `ptx_stream_state` lazily initializes its state.
#[cfg(any(feature = "blas", feature = "dnn", feature = "ptx"))]
fn primary_context_token(
    backend: &CudaBackend,
    device: Device,
) -> BackendResult<std::sync::Arc<oxicuda_driver::Context>> {
    use std::sync::Arc;

    // Read the primary context's raw handle under the context lock, re-affirming
    // it as current on this thread so the handle's stream is created in it.
    let raw = {
        let guard = backend
            .context
            .lock()
            .map_err(|_| BackendError::DeviceError("backend context lock poisoned".into()))?;
        let pctx = guard
            .as_ref()
            .ok_or_else(|| BackendError::DeviceError("no CUDA GPU context available".into()))?;
        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe { (api.cu_ctx_set_current)(pctx.raw()) })
            .map_err(|e| BackendError::DeviceError(e.to_string()))?;
        pctx.raw()
    };
    // SAFETY: the primary context is retained for the whole backend lifetime and
    // released only in `CudaBackend::drop`, after every cached handle/stream that
    // borrows it has been torn down, so `raw` stays valid for this wrapper and
    // any stream built from it.
    let ctx = unsafe { oxicuda_driver::Context::from_raw_borrowed(raw, device) };
    Ok(Arc::new(ctx))
}

/// Runs `f` with the backend's cached [`BlasHandle`](oxicuda_blas::BlasHandle),
/// creating it (and its [`primary_context_token`] lifetime token) the first
/// time this is called and reusing it on every subsequent call.
///
/// The mutex guard is held for the duration of `f`, which serializes GEMM
/// calls that share this backend across threads; each call still runs
/// [`CudaBackend::activate_gpu`] beforehand (the CUDA current-context is
/// per-thread), so the primary context is guaranteed current no matter which
/// thread invokes the operation.
#[cfg(feature = "blas")]
fn with_blas_handle<R>(
    backend: &CudaBackend,
    device: Device,
    f: impl FnOnce(&oxicuda_blas::BlasHandle) -> BackendResult<R>,
) -> BackendResult<R> {
    let mut guard = backend
        .blas_state
        .lock()
        .map_err(|_| BackendError::DeviceError("BLAS handle lock poisoned".into()))?;
    if guard.is_none() {
        let ctx = primary_context_token(backend, device)?;
        let handle = oxicuda_blas::BlasHandle::new(&ctx)
            .map_err(|e| BackendError::DeviceError(format!("BLAS handle creation failed: {e}")))?;
        *guard = Some((ctx, handle));
    }
    match guard.as_ref() {
        Some((_, handle)) => f(handle),
        None => Err(BackendError::DeviceError(
            "BLAS handle state unexpectedly empty after initialization".into(),
        )),
    }
}

/// Runs `f` with the backend's cached [`DnnHandle`](oxicuda_dnn::handle::DnnHandle),
/// analogous to [`with_blas_handle`] for `conv2d_forward`/`attention`.
#[cfg(feature = "dnn")]
fn with_dnn_handle<R>(
    backend: &CudaBackend,
    device: Device,
    f: impl FnOnce(&oxicuda_dnn::handle::DnnHandle) -> BackendResult<R>,
) -> BackendResult<R> {
    let mut guard = backend
        .dnn_state
        .lock()
        .map_err(|_| BackendError::DeviceError("DNN handle lock poisoned".into()))?;
    if guard.is_none() {
        let ctx = primary_context_token(backend, device)?;
        let handle = oxicuda_dnn::handle::DnnHandle::new(&ctx)
            .map_err(|e| BackendError::DeviceError(format!("DNN handle creation failed: {e}")))?;
        *guard = Some((ctx, handle));
    }
    match guard.as_ref() {
        Some((_, handle)) => f(handle),
        None => Err(BackendError::DeviceError(
            "DNN handle state unexpectedly empty after initialization".into(),
        )),
    }
}

/// Real GEMM implementation: delegates to `oxicuda-blas`'s GEMM dispatcher.
///
/// The abstract backend uses column-major `f64` matrices; `op(A)` is `m x k`,
/// `op(B)` is `k x n`, and `C` is `m x n`. Matrix descriptors are constructed
/// so that their post-transpose effective dimensions match this contract, and
/// the [`BackendTranspose`] flags are mapped to `oxicuda-blas`'s transpose
/// enum before forwarding pointers, leading dimensions, and the alpha/beta
/// scalars.
#[cfg(feature = "blas")]
#[allow(clippy::too_many_arguments)]
fn gemm_impl(
    backend: &CudaBackend,
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
    use oxicuda_blas::level3::gemm_api::gemm as blas_gemm;
    use oxicuda_blas::types::{Layout, MatrixDesc, MatrixDescMut, Transpose};

    if m == 0 || n == 0 || k == 0 {
        return Err(BackendError::InvalidArgument(
            "GEMM dimensions m, n, k must all be > 0".into(),
        ));
    }

    let to_u32 = |v: usize, what: &str| -> BackendResult<u32> {
        u32::try_from(v)
            .map_err(|_| BackendError::InvalidArgument(format!("GEMM {what} exceeds u32 range")))
    };
    let m_u = to_u32(m, "m")?;
    let n_u = to_u32(n, "n")?;
    let k_u = to_u32(k, "k")?;
    let lda_u = to_u32(lda, "lda")?;
    let ldb_u = to_u32(ldb, "ldb")?;
    let ldc_u = to_u32(ldc, "ldc")?;

    // `activate_gpu` still runs on every call: the CUDA current-context is
    // per-thread, so this thread's context must be (re-)bound regardless of
    // whether the BLAS handle itself is freshly created or cached.
    let device = backend.activate_gpu()?;

    let map_trans = |t: BackendTranspose| match t {
        BackendTranspose::NoTrans => Transpose::NoTrans,
        BackendTranspose::Trans => Transpose::Trans,
        BackendTranspose::ConjTrans => Transpose::ConjTrans,
    };
    let blas_trans_a = map_trans(trans_a);
    let blas_trans_b = map_trans(trans_b);

    // Build descriptors so that `effective_dims(trans)` yields the contracted
    // (rows, cols): op(A) is m x k, op(B) is k x n. For NoTrans the stored
    // dimensions equal the effective ones; for Trans/ConjTrans they are swapped.
    let stored_dims = |trans: Transpose, eff_rows: u32, eff_cols: u32| -> (u32, u32) {
        match trans {
            Transpose::NoTrans => (eff_rows, eff_cols),
            Transpose::Trans | Transpose::ConjTrans => (eff_cols, eff_rows),
        }
    };
    let (a_rows, a_cols) = stored_dims(blas_trans_a, m_u, k_u);
    let (b_rows, b_cols) = stored_dims(blas_trans_b, k_u, n_u);

    // `oxicuda_blas::gemm` operates on tight row-major operands only. Map this
    // column-major call onto it with the transpose identity
    //   C_col = op(A)*op(B)   <=>   C_row = op(B)*op(A),
    // where each column-major buffer is reinterpreted (byte-for-byte, same
    // leading dimension) as its row-major transpose — swap the stored
    // rows/cols — then the two operands are swapped and `m`/`n` exchanged. The
    // row-major result buffer is exactly the requested column-major `C`.
    // Padded (non-tight) leading dimensions are rejected by the BLAS layer with
    // a structured `InvalidArgument`, which propagates unchanged.
    let a_row = MatrixDesc::<f64>::from_raw(
        a_ptr as CUdeviceptr,
        a_cols,
        a_rows,
        lda_u,
        Layout::RowMajor,
    );
    let b_row = MatrixDesc::<f64>::from_raw(
        b_ptr as CUdeviceptr,
        b_cols,
        b_rows,
        ldb_u,
        Layout::RowMajor,
    );
    let mut c_row =
        MatrixDescMut::<f64>::from_raw(c_ptr as CUdeviceptr, n_u, m_u, ldc_u, Layout::RowMajor);

    with_blas_handle(backend, device, |handle| {
        blas_gemm(
            handle,
            blas_trans_b,
            blas_trans_a,
            alpha,
            &b_row,
            &a_row,
            beta,
            &mut c_row,
        )
        .map_err(map_blas_error)
    })
}

/// Translates an `oxicuda-blas` error into a [`BackendError`].
#[cfg(feature = "blas")]
fn map_blas_error(e: oxicuda_blas::BlasError) -> BackendError {
    use oxicuda_blas::BlasError;
    match e {
        BlasError::InvalidDimension(m) | BlasError::DimensionMismatch(m) => {
            BackendError::InvalidArgument(m)
        }
        BlasError::InvalidArgument(m) => BackendError::InvalidArgument(m),
        BlasError::UnsupportedOperation(m) => BackendError::Unsupported(m),
        BlasError::BufferTooSmall { expected, actual } => BackendError::InvalidArgument(format!(
            "BLAS buffer too small: expected {expected}, got {actual}"
        )),
        other => BackendError::DeviceError(other.to_string()),
    }
}

// ─── DNN conv2d / attention wiring (feature = "dnn") ────────

/// Real conv2d-forward implementation: delegates to `oxicuda-dnn`'s
/// `conv_forward` entry point.
///
/// The validated NCHW input / KCRS filter / NKPQ output shapes are turned into
/// `oxicuda-dnn` tensor descriptors (single-precision — the standard
/// convolution working type), a [`ConvolutionDescriptor`] carries the stride
/// and padding, and the DNN handle selects and launches the optimal
/// convolution algorithm. If the chosen algorithm needs a workspace the call
/// is retried with a freshly-allocated device buffer of the requested size.
#[cfg(feature = "dnn")]
#[allow(clippy::too_many_arguments)]
fn conv2d_forward_impl(
    backend: &CudaBackend,
    input_ptr: u64,
    input_shape: &[usize],
    filter_ptr: u64,
    filter_shape: &[usize],
    output_ptr: u64,
    output_shape: &[usize],
    stride: &[usize],
    padding: &[usize],
) -> BackendResult<()> {
    use oxicuda_dnn::conv::api::conv_forward;
    use oxicuda_dnn::handle::DnnHandle;
    use oxicuda_dnn::types::{ConvolutionDescriptor, TensorDesc, TensorDescMut, TensorLayout};
    use oxicuda_memory::DeviceBuffer;

    let to_u32 = |v: usize, what: &str| -> BackendResult<u32> {
        u32::try_from(v)
            .map_err(|_| BackendError::InvalidArgument(format!("conv2d {what} exceeds u32 range")))
    };
    let dims_u32 = |s: &[usize], what: &str| -> BackendResult<Vec<u32>> {
        s.iter().map(|&v| to_u32(v, what)).collect()
    };

    let in_dims = dims_u32(input_shape, "input dim")?;
    let filt_dims = dims_u32(filter_shape, "filter dim")?;
    let out_dims = dims_u32(output_shape, "output dim")?;

    // NCHW row-major contiguous strides for a 4-D tensor.
    let nchw_strides = |d: &[u32]| -> Vec<u32> { vec![d[1] * d[2] * d[3], d[2] * d[3], d[3], 1] };

    // `activate_gpu` still runs on every call (per-thread current-context);
    // the DNN handle itself is cached and reused across calls.
    let device = backend.activate_gpu()?;

    let conv_desc = ConvolutionDescriptor::conv2d(
        to_u32(padding[0], "pad_h")?,
        to_u32(padding[1], "pad_w")?,
        to_u32(stride[0], "stride_h")?,
        to_u32(stride[1], "stride_w")?,
        1,
        1,
        1,
    )
    .map_err(map_dnn_error)?;

    with_dnn_handle(backend, device, |handle| {
        // The conv engines may need a workspace. Try without one first; on a
        // `WorkspaceRequired` error allocate exactly the requested size and
        // retry.
        let run = |handle: &DnnHandle,
                   workspace: Option<&mut DeviceBuffer<u8>>|
         -> oxicuda_dnn::error::DnnResult<()> {
            let input = TensorDesc::<f32>::from_raw(
                input_ptr as CUdeviceptr,
                in_dims.clone(),
                nchw_strides(&in_dims),
                TensorLayout::Nchw,
            )?;
            let filter = TensorDesc::<f32>::from_raw(
                filter_ptr as CUdeviceptr,
                filt_dims.clone(),
                nchw_strides(&filt_dims),
                TensorLayout::Nchw,
            )?;
            let mut output = TensorDescMut::<f32>::from_raw(
                output_ptr as CUdeviceptr,
                out_dims.clone(),
                nchw_strides(&out_dims),
                TensorLayout::Nchw,
            )?;
            conv_forward(handle, &input, &filter, &mut output, &conv_desc, workspace)
        };

        match run(handle, None) {
            Ok(()) => Ok(()),
            Err(oxicuda_dnn::error::DnnError::WorkspaceRequired(bytes)) => {
                let mut ws = DeviceBuffer::<u8>::alloc(bytes.max(1)).map_err(|e| {
                    BackendError::DeviceError(format!("workspace alloc failed: {e}"))
                })?;
                run(handle, Some(&mut ws)).map_err(map_dnn_error)
            }
            Err(other) => Err(map_dnn_error(other)),
        }
    })
}

/// Real attention implementation: delegates to `oxicuda-dnn`'s
/// FlashAttention-2 forward kernel.
///
/// The q / k / v / o pointers are described as `[batch, heads, seq, head_dim]`
/// tensors, a [`FlashAttentionConfig`] carries the tile sizes, scale, and
/// causal flag, and a log-sum-exp scratch buffer is allocated for the kernel.
#[cfg(feature = "dnn")]
#[allow(clippy::too_many_arguments)]
fn attention_impl(
    backend: &CudaBackend,
    q_ptr: u64,
    k_ptr: u64,
    v_ptr: u64,
    o_ptr: u64,
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f64,
    causal: bool,
) -> BackendResult<()> {
    use oxicuda_dnn::attn::flash_attn::forward::{FlashAttentionConfig, flash_attention_forward};
    use oxicuda_dnn::types::{TensorDesc, TensorDescMut, TensorLayout};
    use oxicuda_memory::DeviceBuffer;

    if batch == 0 || heads == 0 {
        return Err(BackendError::InvalidArgument(
            "attention batch and heads must be > 0".into(),
        ));
    }

    let to_u32 = |v: usize, what: &str| -> BackendResult<u32> {
        u32::try_from(v).map_err(|_| {
            BackendError::InvalidArgument(format!("attention {what} exceeds u32 range"))
        })
    };
    let batch_u = to_u32(batch, "batch")?;
    let heads_u = to_u32(heads, "heads")?;
    let seq_q_u = to_u32(seq_q, "seq_q")?;
    let seq_kv_u = to_u32(seq_kv, "seq_kv")?;
    let head_dim_u = to_u32(head_dim, "head_dim")?;

    // `activate_gpu` still runs on every call (per-thread current-context);
    // the DNN handle itself is cached and reused across calls.
    let device = backend.activate_gpu()?;

    with_dnn_handle(backend, device, |handle| {
        let sm = handle.sm_version();
        // `auto` already defaults `precision` to single-precision
        // (`PtxType::F32`), which matches the f32 tensors built below.
        let mut config = FlashAttentionConfig::auto(head_dim_u, seq_q_u, seq_kv_u, causal, sm);
        config.num_heads = heads_u;
        config.sm_scale = scale as f32;

        // The FlashAttention kernel stores Q/K/V tiles in static shared
        // memory. `auto` sizes the tiles from `head_dim` alone, which can
        // exceed the device's per-block shared-memory budget for short
        // sequences or large head dimensions. Clamp the tile sizes to the
        // actual sequence lengths and then shrink them until the
        // shared-memory footprint fits, so the launch configuration is
        // always valid for the device.
        config.block_m = config.block_m.min(seq_q_u).max(1);
        config.block_n = config.block_n.min(seq_kv_u).max(1);
        // A conservative static-shared budget that every supported SM honours.
        const SHARED_BUDGET: u32 = 46 * 1024;
        while config.shared_mem_bytes() > SHARED_BUDGET
            && (config.block_m > 1 || config.block_n > 1)
        {
            if config.block_m >= config.block_n && config.block_m > 1 {
                config.block_m = (config.block_m / 2).max(1);
            } else if config.block_n > 1 {
                config.block_n = (config.block_n / 2).max(1);
            } else {
                break;
            }
        }
        if config.shared_mem_bytes() > SHARED_BUDGET {
            return Err(BackendError::Unsupported(format!(
                "attention head_dim {head_dim} too large for the device shared-memory budget"
            )));
        }

        // Attention tensors are laid out as [batch, heads, seq, head_dim].
        let attn_strides = |seq: u32| -> Vec<u32> {
            vec![heads_u * seq * head_dim_u, seq * head_dim_u, head_dim_u, 1]
        };
        let q = TensorDesc::<f32>::from_raw(
            q_ptr as CUdeviceptr,
            vec![batch_u, heads_u, seq_q_u, head_dim_u],
            attn_strides(seq_q_u),
            TensorLayout::Nchw,
        )
        .map_err(map_dnn_error)?;
        let k = TensorDesc::<f32>::from_raw(
            k_ptr as CUdeviceptr,
            vec![batch_u, heads_u, seq_kv_u, head_dim_u],
            attn_strides(seq_kv_u),
            TensorLayout::Nchw,
        )
        .map_err(map_dnn_error)?;
        let v = TensorDesc::<f32>::from_raw(
            v_ptr as CUdeviceptr,
            vec![batch_u, heads_u, seq_kv_u, head_dim_u],
            attn_strides(seq_kv_u),
            TensorLayout::Nchw,
        )
        .map_err(map_dnn_error)?;
        let mut o = TensorDescMut::<f32>::from_raw(
            o_ptr as CUdeviceptr,
            vec![batch_u, heads_u, seq_q_u, head_dim_u],
            attn_strides(seq_q_u),
            TensorLayout::Nchw,
        )
        .map_err(map_dnn_error)?;

        // Log-sum-exp scratch buffer required by the FlashAttention forward
        // kernel.
        let lse_elems = batch * heads * seq_q;
        let mut lse = DeviceBuffer::<f32>::alloc(lse_elems.max(1))
            .map_err(|e| BackendError::DeviceError(format!("LSE buffer alloc failed: {e}")))?;

        flash_attention_forward(handle, &q, &k, &v, &mut o, &mut lse, &config)
            .map_err(map_dnn_error)
    })
}

/// Translates an `oxicuda-dnn` error into a [`BackendError`].
#[cfg(feature = "dnn")]
fn map_dnn_error(e: oxicuda_dnn::error::DnnError) -> BackendError {
    use oxicuda_dnn::error::DnnError;
    match e {
        DnnError::InvalidDimension(m) | DnnError::InvalidArgument(m) => {
            BackendError::InvalidArgument(m)
        }
        DnnError::UnsupportedOperation(m) => BackendError::Unsupported(m),
        DnnError::BufferTooSmall { expected, actual } => BackendError::InvalidArgument(format!(
            "DNN buffer too small: expected {expected}, got {actual}"
        )),
        DnnError::WorkspaceRequired(bytes) => {
            BackendError::DeviceError(format!("DNN workspace required: {bytes} bytes"))
        }
        other => BackendError::DeviceError(other.to_string()),
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_backend_new_is_uninitialized() {
        let backend = CudaBackend::new();
        assert!(!backend.is_initialized());
    }

    #[test]
    fn cuda_backend_init_sets_initialized() {
        let mut backend = CudaBackend::new();
        let result = backend.init();
        assert!(result.is_ok());
        assert!(backend.is_initialized());
    }

    #[test]
    fn cuda_backend_double_init_is_noop() {
        let mut backend = CudaBackend::new();
        assert!(backend.init().is_ok());
        assert!(backend.init().is_ok());
        assert!(backend.is_initialized());
    }

    #[test]
    fn cuda_backend_name() {
        let backend = CudaBackend::new();
        assert_eq!(backend.name(), "cuda");
    }

    #[test]
    fn cuda_backend_default() {
        let backend = CudaBackend::default();
        assert!(!backend.is_initialized());
        assert_eq!(backend.name(), "cuda");
    }

    #[test]
    fn trait_is_object_safe() {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        let boxed: Box<dyn ComputeBackend> = Box::new(backend);
        assert_eq!(boxed.name(), "cuda");
        assert!(boxed.is_initialized());
        assert!(boxed.synchronize().is_ok());
    }

    #[test]
    fn operations_fail_when_not_initialized() {
        let backend = CudaBackend::new();
        assert_eq!(
            backend.synchronize().unwrap_err(),
            BackendError::NotInitialized
        );
        assert_eq!(
            backend
                .gemm(
                    BackendTranspose::NoTrans,
                    BackendTranspose::NoTrans,
                    1,
                    1,
                    1,
                    1.0,
                    0,
                    1,
                    0,
                    1,
                    0.0,
                    0,
                    1,
                )
                .unwrap_err(),
            BackendError::NotInitialized
        );
        assert_eq!(
            backend.alloc(1024).unwrap_err(),
            BackendError::NotInitialized
        );
        assert_eq!(backend.free(0).unwrap_err(), BackendError::NotInitialized);
        assert_eq!(
            backend.copy_htod(0, &[1, 2, 3]).unwrap_err(),
            BackendError::NotInitialized
        );
        let mut buf = [0u8; 4];
        assert_eq!(
            backend.copy_dtoh(&mut buf, 0).unwrap_err(),
            BackendError::NotInitialized
        );
    }

    #[test]
    fn conv2d_validates_shapes() {
        let mut backend = CudaBackend::new();
        backend.init().ok();

        // Wrong input shape length
        let result = backend.conv2d_forward(
            0,
            &[1, 3, 32], // only 3 elements
            0,
            &[64, 3, 3, 3],
            0,
            &[1, 64, 30, 30],
            &[1, 1],
            &[0, 0],
        );
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));

        // Wrong filter shape length
        let result = backend.conv2d_forward(
            0,
            &[1, 3, 32, 32],
            0,
            &[64, 3, 3], // only 3 elements
            0,
            &[1, 64, 30, 30],
            &[1, 1],
            &[0, 0],
        );
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));

        // Wrong stride length
        let result = backend.conv2d_forward(
            0,
            &[1, 3, 32, 32],
            0,
            &[64, 3, 3, 3],
            0,
            &[1, 64, 30, 30],
            &[1], // only 1 element
            &[0, 0],
        );
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn attention_validates_params() {
        let mut backend = CudaBackend::new();
        backend.init().ok();

        // Zero sequence length
        let result = backend.attention(0, 0, 0, 0, 1, 1, 0, 128, 64, 0.125, false);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));

        // Non-positive scale
        let result = backend.attention(0, 0, 0, 0, 1, 1, 128, 128, 64, 0.0, false);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));

        // NaN scale
        let result = backend.attention(0, 0, 0, 0, 1, 1, 128, 128, 64, f64::NAN, false);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn reduce_validates_axis() {
        let mut backend = CudaBackend::new();
        backend.init().ok();

        // Axis out of bounds
        let result = backend.reduce(ReduceOp::Sum, 0, 0, &[10, 20], 2);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));

        // Empty shape
        let result = backend.reduce(ReduceOp::Sum, 0, 0, &[], 0);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn unary_binary_empty_is_noop() {
        let mut backend = CudaBackend::new();
        backend.init().ok();

        // Empty tensor operations are no-ops
        assert!(backend.unary(UnaryOp::Relu, 0, 0, 0).is_ok());
        assert!(backend.binary(BinaryOp::Add, 0, 0, 0, 0).is_ok());
    }

    #[test]
    fn alloc_zero_bytes_is_error() {
        let mut backend = CudaBackend::new();
        backend.init().ok();

        let result = backend.alloc(0);
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn copy_empty_is_noop() {
        let mut backend = CudaBackend::new();
        backend.init().ok();

        assert!(backend.copy_htod(0, &[]).is_ok());
        let mut empty: [u8; 0] = [];
        assert!(backend.copy_dtoh(&mut empty, 0).is_ok());
    }

    #[test]
    fn debug_impl() {
        let backend = CudaBackend::new();
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("CudaBackend"));
        assert!(debug_str.contains("initialized"));
    }

    // ── GPU compute wiring tests (require a CUDA device) ─────
    //
    // These exercise the real BLAS / DNN / PTX pipelines. When no GPU is
    // present `init` still succeeds but no context is retained, so each test
    // returns early on `!has_gpu_context()` and stays valid on GPU-less CI.
    //
    // Past that guard there *is* a live context, and a `DeviceError` is then a
    // genuine defect, never an environment artefact — so none of these tests
    // tolerates one. They used to, and that is exactly how the PTX launch
    // stream's context mismatch (`CUDA_ERROR_INVALID_HANDLE`, see
    // `primary_context_token`) stayed green here for a whole release: the
    // wiring tests swallowed the very error the bug produced, and only
    // `compute::tests::default_backend_runs_a_real_compute_round_trip` — which
    // unwraps — caught it. `Unsupported` remains acceptable where the op
    // legitimately may not be implemented for a given shape/dtype.

    /// Returns `true` when a live CUDA context was retained.
    #[cfg(feature = "blas")]
    fn gpu_available(backend: &CudaBackend) -> bool {
        backend.has_gpu_context()
    }

    #[cfg(feature = "blas")]
    #[test]
    fn gemm_wiring_identity_multiply() {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        if !gpu_available(&backend) {
            return; // no GPU — nothing to verify
        }

        // Column-major 2x2 matrices. A = identity, B = [[1,2],[3,4]].
        // C = A * B = B.  Column-major layout: element (i,j) at j*ld + i.
        let a: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0]; // identity, col-major
        let b: Vec<f64> = vec![1.0, 3.0, 2.0, 4.0]; // [[1,2],[3,4]], col-major
        let mut c: Vec<f64> = vec![0.0; 4];

        let bytes = 4 * std::mem::size_of::<f64>();
        let a_ptr = backend.alloc(bytes).expect("alloc a");
        let b_ptr = backend.alloc(bytes).expect("alloc b");
        let c_ptr = backend.alloc(bytes).expect("alloc c");

        backend.copy_htod(a_ptr, bytemuck_cast(&a)).expect("copy a");
        backend.copy_htod(b_ptr, bytemuck_cast(&b)).expect("copy b");
        backend.copy_htod(c_ptr, bytemuck_cast(&c)).expect("zero c");

        let result = backend.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a_ptr,
            2,
            b_ptr,
            2,
            0.0,
            c_ptr,
            2,
        );

        // Free regardless of outcome.
        let read_back = if result.is_ok() {
            backend.synchronize().ok();
            let mut out = vec![0u8; bytes];
            backend.copy_dtoh(&mut out, c_ptr).ok();
            Some(out)
        } else {
            None
        };
        backend.free(a_ptr).ok();
        backend.free(b_ptr).ok();
        backend.free(c_ptr).ok();

        // The backend wiring reinterprets the column-major operands as tight
        // row-major and forwards them to `oxicuda_blas::gemm` via the
        // transpose identity. With the f64 GEMM path now correct on-device,
        // `identity * B == B` must hold exactly.
        result.expect("GEMM identity multiply should succeed");
        let out = read_back.expect("read back");
        for (i, chunk) in out.chunks_exact(8).enumerate() {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(chunk);
            c[i] = f64::from_le_bytes(arr);
        }
        assert_eq!(c, b, "identity * B must equal B (col-major)");
    }

    #[cfg(feature = "blas")]
    #[test]
    fn gemm_wiring_rejects_zero_dims() {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        if !gpu_available(&backend) {
            return;
        }
        // m = 0 must be rejected by the wiring before reaching the GPU.
        let result = backend.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            0,
            2,
            2,
            1.0,
            0,
            1,
            0,
            1,
            0.0,
            0,
            1,
        );
        assert!(matches!(result, Err(BackendError::InvalidArgument(_))));
    }

    /// Regression test for F046: calling `gemm` twice on one backend
    /// instance must reuse the cached BLAS handle (see `with_blas_handle`)
    /// rather than rebuilding a fresh context/handle per call, and the two
    /// calls must reach the same outcome. The GEMM dispatcher's acceptance
    /// or rejection of a given operand layout is orthogonal to caching (see
    /// the documented pre-existing dispatcher caveats in
    /// `gemm_wiring_identity_multiply`); what this test guards is that the
    /// SECOND call — which hits the cache — behaves exactly like the first,
    /// rather than the cache leaking or corrupting state across calls.
    #[cfg(feature = "blas")]
    #[test]
    fn gemm_wiring_repeated_calls_reuse_cached_handle() {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        if !gpu_available(&backend) {
            return;
        }

        let a: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0]; // identity, col-major
        let b: Vec<f64> = vec![1.0, 3.0, 2.0, 4.0]; // [[1,2],[3,4]], col-major
        let zero: Vec<f64> = vec![0.0; 4];
        let bytes = 4 * std::mem::size_of::<f64>();

        let mut outcomes: Vec<Option<Vec<u8>>> = Vec::new();
        for _ in 0..2 {
            let a_ptr = backend.alloc(bytes).expect("alloc a");
            let b_ptr = backend.alloc(bytes).expect("alloc b");
            let c_ptr = backend.alloc(bytes).expect("alloc c");
            backend.copy_htod(a_ptr, bytemuck_cast(&a)).expect("copy a");
            backend.copy_htod(b_ptr, bytemuck_cast(&b)).expect("copy b");
            backend
                .copy_htod(c_ptr, bytemuck_cast(&zero))
                .expect("zero c");

            let result = backend.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                2,
                2,
                2,
                1.0,
                a_ptr,
                2,
                b_ptr,
                2,
                0.0,
                c_ptr,
                2,
            );

            let outcome = match &result {
                Ok(()) => {
                    backend.synchronize().ok();
                    let mut out = vec![0u8; bytes];
                    backend.copy_dtoh(&mut out, c_ptr).ok();
                    Some(out)
                }
                Err(e) => panic!(
                    "GEMM on a live GPU context must succeed (the identical call in \
                     `gemm_wiring_identity_multiply` does), got: {e:?}"
                ),
            };
            backend.free(a_ptr).ok();
            backend.free(b_ptr).ok();
            backend.free(c_ptr).ok();
            outcomes.push(outcome);
        }

        assert_eq!(
            outcomes[0].is_some(),
            outcomes[1].is_some(),
            "second GEMM call (cached-handle path) behaved differently from the first"
        );
        if let (Some(first), Some(second)) = (&outcomes[0], &outcomes[1]) {
            assert_eq!(
                first, second,
                "repeated GEMM calls sharing a cached BLAS handle produced different output"
            );
        }
    }

    #[cfg(feature = "dnn")]
    #[test]
    fn conv2d_wiring_executes() {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        if !backend.has_gpu_context() {
            return;
        }

        // 1x1 conv: input [1,3,8,8], filter [4,3,1,1], output [1,4,8,8].
        let n = 1usize;
        let c = 3usize;
        let h = 8usize;
        let w = 8usize;
        let kk = 4usize;
        let input_elems = n * c * h * w;
        let filter_elems = kk * c;
        let output_elems = n * kk * h * w;

        let f4 = std::mem::size_of::<f32>();
        let in_ptr = backend.alloc(input_elems * f4).expect("alloc in");
        let filt_ptr = backend.alloc(filter_elems * f4).expect("alloc filt");
        let out_ptr = backend.alloc(output_elems * f4).expect("alloc out");

        let input = vec![1.0f32; input_elems];
        let filter = vec![0.5f32; filter_elems];
        backend
            .copy_htod(in_ptr, f32_bytes(&input))
            .expect("copy in");
        backend
            .copy_htod(filt_ptr, f32_bytes(&filter))
            .expect("copy filt");

        let result = backend.conv2d_forward(
            in_ptr,
            &[n, c, h, w],
            filt_ptr,
            &[kk, c, 1, 1],
            out_ptr,
            &[n, kk, h, w],
            &[1, 1],
            &[0, 0],
        );

        if result.is_ok() {
            backend.synchronize().ok();
        }
        backend.free(in_ptr).ok();
        backend.free(filt_ptr).ok();
        backend.free(out_ptr).ok();

        // With a live context the wiring must either succeed or report the op
        // as unimplemented for this configuration; a DeviceError here is a
        // real failure, and it must never silently no-op.
        match result {
            Ok(()) => {}
            Err(BackendError::Unsupported(_)) => {}
            Err(e) => panic!("conv2d on a live GPU context failed: {e:?}"),
        }
    }

    /// Regression test for F046: calling `conv2d_forward` twice on one
    /// backend instance must reuse the cached DNN handle (see
    /// `with_dnn_handle`) rather than rebuilding a fresh context/handle per
    /// call, and the two calls must produce identical results — verifying
    /// the cache does not leak or corrupt state across invocations.
    #[cfg(feature = "dnn")]
    #[test]
    fn conv2d_wiring_repeated_calls_reuse_cached_handle() {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        if !backend.has_gpu_context() {
            return;
        }

        let n = 1usize;
        let c = 3usize;
        let h = 8usize;
        let w = 8usize;
        let kk = 4usize;
        let input_elems = n * c * h * w;
        let filter_elems = kk * c;
        let output_elems = n * kk * h * w;
        let f4 = std::mem::size_of::<f32>();

        let mut outcomes: Vec<Option<Vec<u8>>> = Vec::new();
        for _ in 0..2 {
            let in_ptr = backend.alloc(input_elems * f4).expect("alloc in");
            let filt_ptr = backend.alloc(filter_elems * f4).expect("alloc filt");
            let out_ptr = backend.alloc(output_elems * f4).expect("alloc out");

            let input = vec![1.0f32; input_elems];
            let filter = vec![0.5f32; filter_elems];
            backend
                .copy_htod(in_ptr, f32_bytes(&input))
                .expect("copy in");
            backend
                .copy_htod(filt_ptr, f32_bytes(&filter))
                .expect("copy filt");

            let result = backend.conv2d_forward(
                in_ptr,
                &[n, c, h, w],
                filt_ptr,
                &[kk, c, 1, 1],
                out_ptr,
                &[n, kk, h, w],
                &[1, 1],
                &[0, 0],
            );

            let outcome = match result {
                Ok(()) => {
                    backend.synchronize().ok();
                    let mut out = vec![0u8; output_elems * f4];
                    backend.copy_dtoh(&mut out, out_ptr).ok();
                    Some(out)
                }
                Err(BackendError::Unsupported(_)) => None,
                Err(e) => panic!("conv2d on a live GPU context failed on repeated call: {e:?}"),
            };
            backend.free(in_ptr).ok();
            backend.free(filt_ptr).ok();
            backend.free(out_ptr).ok();
            outcomes.push(outcome);
        }

        assert_eq!(
            outcomes[0].is_some(),
            outcomes[1].is_some(),
            "second conv2d call (cached-handle path) behaved differently from the first"
        );
        if let (Some(first), Some(second)) = (&outcomes[0], &outcomes[1]) {
            assert_eq!(
                first, second,
                "repeated conv2d calls sharing a cached DNN handle produced different output"
            );
        }
    }

    #[cfg(feature = "dnn")]
    #[test]
    fn attention_wiring_executes() {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        if !backend.has_gpu_context() {
            return;
        }

        let batch = 1usize;
        let heads = 2usize;
        let seq = 16usize;
        let head_dim = 64usize;
        let elems = batch * heads * seq * head_dim;
        let f4 = std::mem::size_of::<f32>();

        let q_ptr = backend.alloc(elems * f4).expect("alloc q");
        let k_ptr = backend.alloc(elems * f4).expect("alloc k");
        let v_ptr = backend.alloc(elems * f4).expect("alloc v");
        let o_ptr = backend.alloc(elems * f4).expect("alloc o");

        let data = vec![0.1f32; elems];
        backend.copy_htod(q_ptr, f32_bytes(&data)).expect("copy q");
        backend.copy_htod(k_ptr, f32_bytes(&data)).expect("copy k");
        backend.copy_htod(v_ptr, f32_bytes(&data)).expect("copy v");

        let scale = 1.0 / (head_dim as f64).sqrt();
        let result = backend.attention(
            q_ptr, k_ptr, v_ptr, o_ptr, batch, heads, seq, seq, head_dim, scale, false,
        );

        if result.is_ok() {
            backend.synchronize().ok();
        }
        backend.free(q_ptr).ok();
        backend.free(k_ptr).ok();
        backend.free(v_ptr).ok();
        backend.free(o_ptr).ok();

        match result {
            Ok(()) => {}
            Err(BackendError::Unsupported(_)) => {}
            Err(e) => panic!("attention on a live GPU context failed: {e:?}"),
        }
    }

    #[cfg(feature = "ptx")]
    #[test]
    fn unary_wiring_relu() {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        if !backend.has_gpu_context() {
            return;
        }

        let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -5.0, 4.0];
        let n = input.len();
        let f4 = std::mem::size_of::<f32>();
        let in_ptr = backend.alloc(n * f4).expect("alloc in");
        let out_ptr = backend.alloc(n * f4).expect("alloc out");
        backend
            .copy_htod(in_ptr, f32_bytes(&input))
            .expect("copy in");

        let result = backend.unary(UnaryOp::Relu, in_ptr, out_ptr, n);

        let read_back = if result.is_ok() {
            backend.synchronize().ok();
            let mut out = vec![0u8; n * f4];
            backend.copy_dtoh(&mut out, out_ptr).ok();
            Some(out)
        } else {
            None
        };
        backend.free(in_ptr).ok();
        backend.free(out_ptr).ok();

        match result {
            Ok(()) => {
                let out = read_back.expect("read back");
                for (i, chunk) in out.chunks_exact(4).enumerate() {
                    let mut arr = [0u8; 4];
                    arr.copy_from_slice(chunk);
                    let got = f32::from_le_bytes(arr);
                    let want = input[i].max(0.0);
                    assert!(
                        (got - want).abs() < 1e-5,
                        "relu[{i}] = {got}, expected {want}"
                    );
                }
            }
            Err(e) => panic!("unary on a live GPU context failed: {e:?}"),
        }
    }

    #[cfg(feature = "ptx")]
    #[test]
    fn binary_wiring_add() {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        if !backend.has_gpu_context() {
            return;
        }

        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let n = a.len();
        let f4 = std::mem::size_of::<f32>();
        let a_ptr = backend.alloc(n * f4).expect("alloc a");
        let b_ptr = backend.alloc(n * f4).expect("alloc b");
        let c_ptr = backend.alloc(n * f4).expect("alloc c");
        backend.copy_htod(a_ptr, f32_bytes(&a)).expect("copy a");
        backend.copy_htod(b_ptr, f32_bytes(&b)).expect("copy b");

        let result = backend.binary(BinaryOp::Add, a_ptr, b_ptr, c_ptr, n);

        let read_back = if result.is_ok() {
            backend.synchronize().ok();
            let mut out = vec![0u8; n * f4];
            backend.copy_dtoh(&mut out, c_ptr).ok();
            Some(out)
        } else {
            None
        };
        backend.free(a_ptr).ok();
        backend.free(b_ptr).ok();
        backend.free(c_ptr).ok();

        match result {
            Ok(()) => {
                let out = read_back.expect("read back");
                for (i, chunk) in out.chunks_exact(4).enumerate() {
                    let mut arr = [0u8; 4];
                    arr.copy_from_slice(chunk);
                    let got = f32::from_le_bytes(arr);
                    let want = a[i] + b[i];
                    assert!((got - want).abs() < 1e-4, "add[{i}] = {got}, want {want}");
                }
            }
            Err(e) => panic!("binary on a live GPU context failed: {e:?}"),
        }
    }

    /// Regression test for F047: calling `binary` twice on one backend
    /// instance must reuse the cached JIT-compiled kernel (see
    /// `ptx_ops::build_kernel`) and launch stream (see
    /// `ptx_ops::ptx_stream_state`) rather than rebuilding a fresh
    /// context/stream/module per call. Each call uses distinct input data so
    /// a stale or corrupted cache entry would surface as a wrong numeric
    /// result rather than accidentally matching by coincidence.
    #[cfg(feature = "ptx")]
    #[test]
    fn binary_wiring_add_repeated_calls_reuse_cache() {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        if !backend.has_gpu_context() {
            return;
        }

        let f4 = std::mem::size_of::<f32>();
        let inputs: [(Vec<f32>, Vec<f32>); 2] = [
            (vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]),
            (vec![-1.0, 5.5, 0.0, 100.0], vec![1.0, -5.5, 2.0, -50.0]),
        ];

        for (a, b) in &inputs {
            let n = a.len();
            let a_ptr = backend.alloc(n * f4).expect("alloc a");
            let b_ptr = backend.alloc(n * f4).expect("alloc b");
            let c_ptr = backend.alloc(n * f4).expect("alloc c");
            backend.copy_htod(a_ptr, f32_bytes(a)).expect("copy a");
            backend.copy_htod(b_ptr, f32_bytes(b)).expect("copy b");

            let result = backend.binary(BinaryOp::Add, a_ptr, b_ptr, c_ptr, n);

            let read_back = if result.is_ok() {
                backend.synchronize().ok();
                let mut out = vec![0u8; n * f4];
                backend.copy_dtoh(&mut out, c_ptr).ok();
                Some(out)
            } else {
                None
            };
            backend.free(a_ptr).ok();
            backend.free(b_ptr).ok();
            backend.free(c_ptr).ok();

            match result {
                Ok(()) => {
                    let out = read_back.expect("read back");
                    for (i, chunk) in out.chunks_exact(4).enumerate() {
                        let mut arr = [0u8; 4];
                        arr.copy_from_slice(chunk);
                        let got = f32::from_le_bytes(arr);
                        let want = a[i] + b[i];
                        assert!(
                            (got - want).abs() < 1e-4,
                            "add[{i}] = {got}, want {want} (cached-kernel path)"
                        );
                    }
                }
                Err(e) => panic!("binary on a live GPU context failed on repeated call: {e:?}"),
            }
        }
    }

    #[cfg(feature = "ptx")]
    #[test]
    fn reduce_wiring_sum_axis() {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        if !backend.has_gpu_context() {
            return;
        }

        // Shape [2, 4]: reduce axis 1 (sum each row) => output length 2.
        // Row 0 = [1,2,3,4] -> 10, Row 1 = [5,6,7,8] -> 26.
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let f4 = std::mem::size_of::<f32>();
        let in_ptr = backend.alloc(input.len() * f4).expect("alloc in");
        let out_ptr = backend.alloc(2 * f4).expect("alloc out");
        backend
            .copy_htod(in_ptr, f32_bytes(&input))
            .expect("copy in");

        let result = backend.reduce(ReduceOp::Sum, in_ptr, out_ptr, &[2, 4], 1);

        let read_back = if result.is_ok() {
            backend.synchronize().ok();
            let mut out = vec![0u8; 2 * f4];
            backend.copy_dtoh(&mut out, out_ptr).ok();
            Some(out)
        } else {
            None
        };
        backend.free(in_ptr).ok();
        backend.free(out_ptr).ok();

        match result {
            Ok(()) => {
                let out = read_back.expect("read back");
                let mut got = [0.0f32; 2];
                for (i, chunk) in out.chunks_exact(4).enumerate() {
                    let mut arr = [0u8; 4];
                    arr.copy_from_slice(chunk);
                    got[i] = f32::from_le_bytes(arr);
                }
                assert!((got[0] - 10.0).abs() < 1e-4, "row0 sum = {}", got[0]);
                assert!((got[1] - 26.0).abs() < 1e-4, "row1 sum = {}", got[1]);
            }
            Err(e) => panic!("reduce on a live GPU context failed: {e:?}"),
        }
    }

    // -- test helpers ----------------------------------------

    /// Reinterprets an `f32` slice as bytes for host-to-device copies.
    #[cfg(any(feature = "ptx", feature = "dnn"))]
    fn f32_bytes(data: &[f32]) -> &[u8] {
        // SAFETY: `f32` has no padding and any bit pattern is a valid `u8`
        // sequence; the resulting slice covers exactly the same bytes.
        unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        }
    }

    /// Reinterprets an `f64` slice as bytes for host-to-device copies.
    #[cfg(feature = "blas")]
    fn bytemuck_cast(data: &[f64]) -> &[u8] {
        // SAFETY: `f64` has no padding and any bit pattern is a valid `u8`
        // sequence; the resulting slice covers exactly the same bytes.
        unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        }
    }
}
