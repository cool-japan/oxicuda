//! PTX-backed elementwise and reduction kernels for [`CudaBackend`].
//!
//! This module implements the real `reduce` / `unary` / `binary` compute path:
//! it emits PTX with `oxicuda-ptx` (the `ElementwiseTemplate` for unary/binary
//! ops and a [`KernelBuilder`]-built kernel for per-axis reductions), caches
//! the generated PTX text on disk via [`PtxCache`], JIT-compiles it into a
//! [`Module`] (itself cached per `(kernel_name, sm)` in the backend, see
//! [`build_kernel`]), and dispatches the kernel through `oxicuda-launch` on a
//! stream cached in the backend (see [`launch_with`]) with a grid/block
//! sizing that covers the work.
//!
//! All buffers are single-precision `f32` — the abstract backend's
//! element-wise operations carry no dtype, and `f32` is the working type
//! shared across the COOLJAPAN GPU stack.

use std::sync::{Arc, OnceLock};

use oxicuda_backend::{BackendError, BackendResult, BinaryOp, ReduceOp, UnaryOp};
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_driver::{Device, Module};
use oxicuda_launch::{Dim3, Kernel, LaunchParams, grid_size_for};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::cache::{PtxCache, PtxCacheKey};
use oxicuda_ptx::ir::PtxType;
use oxicuda_ptx::templates::elementwise::{ElementwiseOp, ElementwiseTemplate};

use super::CudaBackend;

/// Thread-block size used for the elementwise kernels (matches the
/// `max_threads_per_block(256)` baked into the elementwise templates).
const ELEMENTWISE_BLOCK: u32 = 256;

/// Thread-block size used for the per-axis reduction kernel (one thread per
/// output element).
const REDUCE_BLOCK: u32 = 256;

/// Resolves the [`SmVersion`] of a device for PTX target selection.
fn device_sm(device: Device) -> BackendResult<SmVersion> {
    let (major, minor) = device
        .compute_capability()
        .map_err(|e| BackendError::DeviceError(format!("compute capability query failed: {e}")))?;
    SmVersion::from_compute_capability(major, minor).ok_or_else(|| {
        BackendError::Unsupported(format!(
            "unsupported compute capability {major}.{minor} for PTX generation"
        ))
    })
}

/// Process-wide PTX disk cache, built exactly once (including its one-time
/// `create_dir_all` I/O) and shared by every [`CudaBackend`] instance in this
/// process, rather than being rebuilt on every elementwise/reduce call.
static PTX_CACHE: OnceLock<Result<PtxCache, String>> = OnceLock::new();

/// Returns the process-wide PTX disk cache, creating it on first use.
fn ptx_cache() -> BackendResult<&'static PtxCache> {
    PTX_CACHE
        .get_or_init(|| PtxCache::new().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| BackendError::DeviceError(format!("PTX cache init failed: {e}")))
}

/// JIT-compiles `ptx` into a launchable [`Kernel`] for `kernel_name`, or
/// returns the already-compiled kernel from `backend`'s cache.
///
/// The cache is keyed by `(kernel_name, sm)`: modules are loaded via
/// `cuModuleLoadData` while the backend's persistent primary context is
/// current, so a module compiled once for a given SM target remains valid for
/// the lifetime of the backend and does not need to be re-JIT-compiled on
/// every call.
fn build_kernel(
    backend: &CudaBackend,
    ptx: &str,
    kernel_name: &str,
    sm: SmVersion,
) -> BackendResult<Arc<Kernel>> {
    let key = (kernel_name.to_string(), sm);

    {
        let cache = backend
            .kernel_cache
            .lock()
            .map_err(|_| BackendError::DeviceError("kernel cache lock poisoned".into()))?;
        if let Some(kernel) = cache.get(&key) {
            return Ok(Arc::clone(kernel));
        }
    }

    // Compile outside the lock so a slow JIT compile on one thread does not
    // block lookups for unrelated kernels on other threads.
    let module = Module::from_ptx(ptx)
        .map_err(|e| BackendError::DeviceError(format!("PTX module load failed: {e}")))?;
    let kernel = Arc::new(
        Kernel::from_module(Arc::new(module), kernel_name)
            .map_err(|e| BackendError::DeviceError(format!("kernel lookup failed: {e}")))?,
    );

    let mut cache = backend
        .kernel_cache
        .lock()
        .map_err(|_| BackendError::DeviceError("kernel cache lock poisoned".into()))?;
    // Another thread may have raced us to compile the same kernel; keep
    // whichever copy is already cached so every caller observes the same
    // `Kernel` instance for a given key.
    Ok(Arc::clone(cache.entry(key).or_insert(kernel)))
}

/// Maps a backend [`UnaryOp`] to the corresponding `oxicuda-ptx` elementwise op.
fn map_unary(op: UnaryOp) -> ElementwiseOp {
    match op {
        UnaryOp::Relu => ElementwiseOp::Relu,
        UnaryOp::Sigmoid => ElementwiseOp::Sigmoid,
        UnaryOp::Tanh => ElementwiseOp::Tanh,
        UnaryOp::Exp => ElementwiseOp::Exp,
        UnaryOp::Log => ElementwiseOp::Log,
        UnaryOp::Sqrt => ElementwiseOp::Sqrt,
        UnaryOp::Abs => ElementwiseOp::Abs,
        UnaryOp::Neg => ElementwiseOp::Neg,
    }
}

/// Maps a backend [`BinaryOp`] to the corresponding `oxicuda-ptx` elementwise op.
fn map_binary(op: BinaryOp) -> ElementwiseOp {
    match op {
        BinaryOp::Add => ElementwiseOp::Add,
        BinaryOp::Sub => ElementwiseOp::Sub,
        BinaryOp::Mul => ElementwiseOp::Mul,
        BinaryOp::Div => ElementwiseOp::Div,
        BinaryOp::Max => ElementwiseOp::Max,
        BinaryOp::Min => ElementwiseOp::Min,
    }
}

/// Short lowercase name for a [`ReduceOp`], used in kernel naming.
fn reduce_op_name(op: ReduceOp) -> &'static str {
    match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
        ReduceOp::Mean => "mean",
    }
}

/// Generates the PTX source for a per-axis `f32` reduction kernel via the
/// `oxicuda-ptx` [`KernelBuilder`].
///
/// The tensor is viewed as `[outer, axis_len, inner]`. One thread computes one
/// output element: it walks the `axis_len` strided elements, combines them
/// with the reduction's operator, and writes the result. This is a correct,
/// fully-general reduction kernel; the builder places the `.maxntid`
/// directive in the kernel header so the emitted PTX assembles cleanly.
///
/// Kernel signature: `(in_ptr, out_ptr, n_out, axis_len, inner, inv)` where
/// `inv` is `1 / axis_len` and is consumed only by the `Mean` reduction.
fn generate_reduce_ptx(op: ReduceOp, sm: SmVersion) -> Result<String, oxicuda_ptx::PtxGenError> {
    let kernel_name = reduce_kernel_name(op);
    // PTX f32 hex literals for the identity element of each reduction.
    let identity = match op {
        ReduceOp::Sum | ReduceOp::Mean => "0f00000000", // +0.0
        ReduceOp::Max => "0fFF800000",                  // -infinity
        ReduceOp::Min => "0f7F800000",                  // +infinity
    };
    // PTX instruction combining the accumulator with the next element.
    let combine = match op {
        ReduceOp::Sum | ReduceOp::Mean => "add.f32",
        ReduceOp::Max => "max.f32",
        ReduceOp::Min => "min.f32",
    };
    let is_mean = matches!(op, ReduceOp::Mean);

    KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .param("n_out", PtxType::U32)
        .param("axis_len", PtxType::U32)
        .param("inner", PtxType::U32)
        .param("inv", PtxType::F32)
        .max_threads_per_block(REDUCE_BLOCK)
        .body(move |b| {
            let tid = b.global_thread_id_x();
            let tid_name = tid.to_string();
            let n_out = b.load_param_u32("n_out");
            b.if_lt_u32(tid, n_out, move |b| {
                let in_ptr = b.load_param_u64("in_ptr");
                let out_ptr = b.load_param_u64("out_ptr");
                // outer_idx = tid / inner ; inner_idx = tid % inner
                b.raw_ptx(&format!(
                    "ld.param.u32 %r_axis, [%param_axis_len];\n    \
                     ld.param.u32 %r_inner, [%param_inner];\n    \
                     div.u32 %r_outer_idx, {tid_name}, %r_inner;\n    \
                     rem.u32 %r_inner_idx, {tid_name}, %r_inner;\n    \
                     mul.lo.u32 %r_base, %r_outer_idx, %r_axis;\n    \
                     mul.lo.u32 %r_base, %r_base, %r_inner;\n    \
                     add.u32 %r_base, %r_base, %r_inner_idx;\n    \
                     mov.f32 %f_acc, {identity};\n    \
                     mov.u32 %r_k, 0;"
                ));
                // Strided accumulation loop over the reduced axis.
                b.raw_ptx(&format!(
                    "$RED_K_LOOP:\n    \
                     setp.ge.u32 %p_done, %r_k, %r_axis;\n    \
                     @%p_done bra $RED_K_END;\n    \
                     mul.lo.u32 %r_idx, %r_k, %r_inner;\n    \
                     add.u32 %r_idx, %r_idx, %r_base;\n    \
                     cvt.u64.u32 %rd_off, %r_idx;\n    \
                     mul.lo.u64 %rd_off, %rd_off, 4;\n    \
                     add.u64 %rd_addr, {in_ptr}, %rd_off;\n    \
                     ld.global.f32 %f_v, [%rd_addr];\n    \
                     {combine} %f_acc, %f_acc, %f_v;\n    \
                     add.u32 %r_k, %r_k, 1;\n    \
                     bra $RED_K_LOOP;\n    \
                     $RED_K_END:"
                ));
                if is_mean {
                    // Divide the accumulated sum by axis_len for the mean.
                    b.raw_ptx(
                        "ld.param.f32 %f_inv, [%param_inv];\n    \
                         mul.f32 %f_acc, %f_acc, %f_inv;",
                    );
                }
                // Store the result at out[tid].
                b.raw_ptx(&format!(
                    "cvt.u64.u32 %rd_ooff, {tid_name};\n    \
                     mul.lo.u64 %rd_ooff, %rd_ooff, 4;\n    \
                     add.u64 %rd_oaddr, {out_ptr}, %rd_ooff;\n    \
                     st.global.f32 [%rd_oaddr], %f_acc;"
                ));
            });
            b.ret();
        })
        .build()
}

/// Kernel function name for a per-axis reduction of the given operation.
fn reduce_kernel_name(op: ReduceOp) -> String {
    format!("reduce_axis_{}_f32", reduce_op_name(op))
}

/// A small hash of the parameters that distinguish one cached kernel from
/// another (operation identity + block size).
fn params_hash(tag: &str, block: u32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tag.hash(&mut hasher);
    block.hash(&mut hasher);
    PtxType::F32.as_ptx_str().hash(&mut hasher);
    hasher.finish()
}

/// Executes an element-wise unary operation on `n` `f32` elements.
///
/// Emits (or loads from the PTX cache) the unary kernel for `op`, compiles it,
/// and launches it with a one-thread-per-element grid.
pub(super) fn unary_elementwise(
    backend: &CudaBackend,
    op: UnaryOp,
    input_ptr: u64,
    output_ptr: u64,
    n: usize,
) -> BackendResult<()> {
    let device = backend.activate_gpu()?;
    let sm = device_sm(device)?;
    let n_u = u32::try_from(n)
        .map_err(|_| BackendError::InvalidArgument("unary element count exceeds u32".into()))?;

    let ew_op = map_unary(op);
    let template = ElementwiseTemplate::new(ew_op, PtxType::F32, sm);
    let kernel_name = template.kernel_name();

    let cache = ptx_cache()?;
    let key = PtxCacheKey {
        kernel_name: kernel_name.clone(),
        params_hash: params_hash(ew_op.as_str(), ELEMENTWISE_BLOCK),
        sm_version: sm,
    };
    let ptx = cache
        .get_or_generate(&key, || template.generate())
        .map_err(|e| BackendError::DeviceError(format!("unary PTX generation failed: {e}")))?;

    let kernel = build_kernel(backend, &ptx, &kernel_name, sm)?;
    let grid = grid_size_for(n_u, ELEMENTWISE_BLOCK);
    let params = LaunchParams::builder()
        .grid(Dim3::new(grid, 1, 1))
        .block(Dim3::new(ELEMENTWISE_BLOCK, 1, 1))
        .build();

    // Unary elementwise kernel signature: (a_ptr, b_ptr, n).
    let args = (input_ptr as CUdeviceptr, output_ptr as CUdeviceptr, n_u);
    launch_with(backend, &kernel, &params, &args)
}

/// Executes an element-wise binary operation on `n` `f32` elements.
pub(super) fn binary_elementwise(
    backend: &CudaBackend,
    op: BinaryOp,
    a_ptr: u64,
    b_ptr: u64,
    output_ptr: u64,
    n: usize,
) -> BackendResult<()> {
    let device = backend.activate_gpu()?;
    let sm = device_sm(device)?;
    let n_u = u32::try_from(n)
        .map_err(|_| BackendError::InvalidArgument("binary element count exceeds u32".into()))?;

    let ew_op = map_binary(op);
    let template = ElementwiseTemplate::new(ew_op, PtxType::F32, sm);
    let kernel_name = template.kernel_name();

    let cache = ptx_cache()?;
    let key = PtxCacheKey {
        kernel_name: kernel_name.clone(),
        params_hash: params_hash(ew_op.as_str(), ELEMENTWISE_BLOCK),
        sm_version: sm,
    };
    let ptx = cache
        .get_or_generate(&key, || template.generate())
        .map_err(|e| BackendError::DeviceError(format!("binary PTX generation failed: {e}")))?;

    let kernel = build_kernel(backend, &ptx, &kernel_name, sm)?;
    let grid = grid_size_for(n_u, ELEMENTWISE_BLOCK);
    let params = LaunchParams::builder()
        .grid(Dim3::new(grid, 1, 1))
        .block(Dim3::new(ELEMENTWISE_BLOCK, 1, 1))
        .build();

    // Binary elementwise kernel signature: (a_ptr, b_ptr, c_ptr, n).
    let args = (
        a_ptr as CUdeviceptr,
        b_ptr as CUdeviceptr,
        output_ptr as CUdeviceptr,
        n_u,
    );
    launch_with(backend, &kernel, &params, &args)
}

/// Reduces an `f32` tensor of `shape` along `axis`.
///
/// The tensor is viewed as `[outer, axis_len, inner]`; the per-axis reduction
/// kernel runs one thread per output element (`outer * inner` of them), each
/// walking the strided reduced axis. For [`ReduceOp::Mean`] the kernel divides
/// the accumulated sum by `axis_len` on-device.
pub(super) fn reduce_axis(
    backend: &CudaBackend,
    op: ReduceOp,
    input_ptr: u64,
    output_ptr: u64,
    shape: &[usize],
    axis: usize,
) -> BackendResult<()> {
    if shape.is_empty() {
        return Err(BackendError::InvalidArgument(
            "reduce shape must not be empty".into(),
        ));
    }
    if axis >= shape.len() {
        return Err(BackendError::InvalidArgument(format!(
            "reduce axis {axis} out of bounds for {}-D shape",
            shape.len()
        )));
    }

    let outer: usize = shape[..axis].iter().product();
    let axis_len: usize = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();

    if axis_len == 0 {
        return Err(BackendError::InvalidArgument(
            "reduce axis length must be > 0".into(),
        ));
    }
    // `outer` / `inner` are empty-product 1 at the boundaries, so they are
    // always >= 1; the number of output elements is `outer * inner`.
    let n_out = outer
        .checked_mul(inner)
        .ok_or_else(|| BackendError::InvalidArgument("reduce output size overflow".into()))?;
    let n_out_u = u32::try_from(n_out)
        .map_err(|_| BackendError::InvalidArgument("reduce output size exceeds u32".into()))?;
    let axis_len_u = u32::try_from(axis_len)
        .map_err(|_| BackendError::InvalidArgument("reduce axis length exceeds u32".into()))?;
    let inner_u = u32::try_from(inner)
        .map_err(|_| BackendError::InvalidArgument("reduce inner size exceeds u32".into()))?;

    let device = backend.activate_gpu()?;
    let sm = device_sm(device)?;

    let kernel_name = reduce_kernel_name(op);
    let cache = ptx_cache()?;
    let key = PtxCacheKey {
        kernel_name: kernel_name.clone(),
        params_hash: params_hash(reduce_op_name(op), REDUCE_BLOCK),
        sm_version: sm,
    };
    let ptx = cache
        .get_or_generate(&key, || generate_reduce_ptx(op, sm))
        .map_err(|e| BackendError::DeviceError(format!("reduce PTX generation failed: {e}")))?;

    let kernel = build_kernel(backend, &ptx, &kernel_name, sm)?;
    // One thread per output element.
    let grid = grid_size_for(n_out_u, REDUCE_BLOCK);
    let params = LaunchParams::builder()
        .grid(Dim3::new(grid, 1, 1))
        .block(Dim3::new(REDUCE_BLOCK, 1, 1))
        .build();

    // Kernel signature: (in_ptr, out_ptr, n_out, axis_len, inner, inv).
    let inv = 1.0f32 / axis_len as f32;
    let args = (
        input_ptr as CUdeviceptr,
        output_ptr as CUdeviceptr,
        n_out_u,
        axis_len_u,
        inner_u,
        inv,
    );
    launch_with(backend, &kernel, &params, &args)
}

/// Returns the backend's cached PTX-launch stream (with its lifetime-token
/// context), creating both lazily on first use and reusing them on every
/// subsequent call instead of standing up a fresh context and stream per
/// launch.
///
/// The returned guard must be held for the duration of the launch (and the
/// trailing synchronize): [`oxicuda-launch`](oxicuda_launch) requires a
/// [`Stream`](oxicuda_driver::Stream), which in turn needs an `Arc<Context>`.
/// That token is a **non-owning wrapper around the backend's primary context**
/// (see [`primary_context_token`](super::primary_context_token)), not a
/// throwaway regular context: `Stream::new` creates the stream *in the context
/// the token names*, so a throwaway token would leave the stream in a context
/// that owns none of the device memory and holds none of the JIT-loaded
/// modules — `cuLaunchKernel` then rejects the mismatched
/// function/stream pair with `CUDA_ERROR_INVALID_HANDLE`. Binding the token to
/// the primary context puts the stream, the modules loaded by
/// [`build_kernel`], and every device pointer in one and the same context.
fn ptx_stream_state(
    backend: &CudaBackend,
    device: Device,
) -> BackendResult<std::sync::MutexGuard<'_, super::PtxStreamState>> {
    let mut guard = backend
        .ptx_stream
        .lock()
        .map_err(|_| BackendError::DeviceError("PTX stream lock poisoned".into()))?;
    if guard.is_none() {
        // Borrows the retained primary context (and re-affirms it as current on
        // this thread), so the stream is created in the context that owns the
        // device buffers and the JIT-loaded modules.
        let token = super::primary_context_token(backend, device)?;
        let stream = oxicuda_driver::Stream::new(&token)
            .map_err(|e| BackendError::DeviceError(format!("stream creation failed: {e}")))?;
        *guard = Some((token, stream));
    }
    Ok(guard)
}

/// Launches `kernel` with `args` on the backend's cached PTX stream,
/// translating a launch failure into a [`BackendError`].
fn launch_with<A: oxicuda_launch::KernelArgs>(
    backend: &CudaBackend,
    kernel: &Kernel,
    params: &LaunchParams,
    args: &A,
) -> BackendResult<()> {
    // `activate_gpu` still runs on every call: the CUDA current-context is
    // per-thread, so this thread's context must be (re-)bound regardless of
    // whether the launch stream itself is freshly created or cached.
    let device = backend.activate_gpu()?;
    let guard = ptx_stream_state(backend, device)?;
    let (_, stream) = guard.as_ref().ok_or_else(|| {
        BackendError::DeviceError("PTX stream state unexpectedly empty after initialization".into())
    })?;

    kernel
        .launch(params, stream, args)
        .map_err(|e| BackendError::DeviceError(format!("kernel launch failed: {e}")))?;
    // Block until the kernel finishes so callers observe results immediately.
    stream
        .synchronize()
        .map_err(|e| BackendError::DeviceError(format!("stream synchronize failed: {e}")))
}
