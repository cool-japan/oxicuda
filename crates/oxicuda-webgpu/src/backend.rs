//! [`WebGpuBackend`] — the main entry point for the oxicuda-webgpu crate.
//!
//! Implements the [`ComputeBackend`] trait from `oxicuda-backend` using
//! `wgpu` for cross-platform GPU compute (Vulkan, Metal, DX12, WebGPU).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};
use wgpu;

use crate::{
    device::WebGpuDevice,
    memory::WebGpuMemoryManager,
    planner::{self, Limits},
    shader,
};

// GPU dispatch paths for conv2d_forward / attention, plus their CPU-reference
// oracles — split into a sibling file (mirroring how `tests` below is split
// into `backend_tests.rs`) purely to keep this file under the 2 000-line
// refactoring policy.
#[path = "backend_gpu_ops.rs"]
mod gpu_ops;
use gpu_ops::{
    attention_cpu_reference, attention_gpu_dispatch_grid, conv2d_cpu_reference,
    conv2d_gpu_dispatch_grid, conv2d_u32_dims,
};

// Pipeline + bind-group caching — split out for the same 2 000-line-policy
// reason as `gpu_ops` above. See that file's module doc for the caching
// design and its safety argument.
#[path = "backend_cache.rs"]
mod cache;
use cache::{BindGroupCache, CachedPipeline};

// ─── Op-mapping helpers ──────────────────────────────────────────────────────

fn map_unary_op(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Relu => "relu",
        UnaryOp::Sigmoid => "sigmoid",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Exp => "exp",
        UnaryOp::Log => "log",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Abs => "abs",
        UnaryOp::Neg => "neg",
    }
}

fn map_binary_op(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Max => "max",
        BinaryOp::Min => "min",
    }
}

fn map_reduce_op(op: ReduceOp) -> &'static str {
    match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
        ReduceOp::Mean => "mean",
    }
}

/// Packed (minimum) leading dimensions for the row-major GEMM kernel.
///
/// The WGSL kernel stores `op(A)`'s physical buffer as `m×k` (or its transpose
/// `k×m`), `op(B)` as `k×n` (or `n×k`), and `C` as `m×n`, all row-major.  The
/// leading dimension is the physical row stride, so the tightly-packed value is
/// the width of each stored row.  A caller may pass a larger `ld` (padded /
/// sub-matrix view), which the shader honours; a smaller one is invalid.
fn packed_gemm_lds(
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

/// Convert a `usize` dimension (leading dimension, matrix extent, stride,
/// batch count, …) to `u32` for a shader uniform, erroring on overflow
/// instead of silently wrapping.
///
/// `context` should read naturally as `"<context>: <name> <value> exceeds …"`,
/// e.g. `dim_u32("gemm", "m", m)` or `dim_u32("batched_gemm", "batch_count",
/// batch_count)`.
fn dim_u32(context: &str, name: &str, value: usize) -> BackendResult<u32> {
    u32::try_from(value).map_err(|_| {
        BackendError::InvalidArgument(format!("{context}: {name} {value} exceeds u32 range"))
    })
}

/// The compute-dispatch limits this backend plans against.
///
/// `WebGpuDevice::new_async` (`device.rs`) requests `required_limits:
/// adapter.limits()` (not `wgpu::Limits::default()`), so the *device* itself
/// may grant more than the WebGPU-guaranteed baseline — e.g. Apple Silicon
/// adapters typically report up to 1024 invocations per workgroup, well
/// above the 256-invocation / 16×16-tile / 65 535-workgroups-per-axis
/// portable floor.
///
/// This function nonetheless still deliberately plans against the
/// conservative [`Limits::portable_default`] rather than those real (often
/// higher) adapter limits, for two independent reasons:
///
/// 1. Every GEMM / batched-GEMM / FP16-GEMM call site passes `preferred_tile
///    = 16` into [`planner::plan_workgroup_square`], which treats that value
///    as a *ceiling* — feeding in a more permissive `Limits` cannot grow the
///    tile past 16 without also raising `preferred_tile` at the call site,
///    which needs a larger, register-blocked kernel to stay efficient (a
///    separate perf pass; "16×16, the portable max" was the explicit brief
///    for this change).
/// 2. The only other limit this module consults is `max_workgroups_per_dim`
///    (via [`planner::plan_dispatch_1d`] / `plan_dispatch_2d`), and
///    under-using a higher real value is safe by construction: it can only
///    make an extremely large 1-D dispatch (tens of millions of elements and
///    up) fold into, or get rejected as exceeding, a 2-D grid slightly
///    sooner than the hardware strictly requires — never accept a dispatch
///    size the device cannot actually run.
///
/// A future pass that raises `preferred_tile` for adapters with more
/// headroom should replace this with a real `dev.limits()`-derived
/// `planner::Limits` (see `WebGpuDevice::limits`) — the pipeline-cache keys
/// already encode `tile_size` (`"gemm:{tile_size}"` etc.), so a per-adapter
/// tile is safe to cache once that lands.
fn gpu_limits() -> Limits {
    Limits::portable_default()
}

// ─── Backend struct ──────────────────────────────────────────────────────────

/// Cross-platform GPU compute backend backed by `wgpu`.
///
/// # Lifecycle
///
/// 1. `WebGpuBackend::new()` — create an uninitialised backend.
/// 2. `init()` — select the best available adapter and create the device.
/// 3. Use `alloc`, `copy_htod`, compute ops, `copy_dtoh`, `free`.
/// 4. `synchronize()` — wait for all pending GPU work to finish.
#[derive(Debug)]
pub struct WebGpuBackend {
    device: Option<Arc<WebGpuDevice>>,
    memory: Option<Arc<WebGpuMemoryManager>>,
    initialized: bool,
    /// Cache of compiled compute pipelines (bundled with their group-0
    /// bind-group layout) keyed by a stable `(op, tile/size)` string.  WGSL
    /// front-end parsing plus backend-ISA compilation is heavyweight and
    /// depends only on the key, so every hot-path compute op reuses its
    /// pipeline instead of rebuilding one per invocation.  See `cache.rs`
    /// (`WebGpuBackend::cached_pipeline`) for the implementation.
    pipeline_cache: Mutex<HashMap<String, CachedPipeline>>,
    /// Cache of bind groups (each backed by its own dedicated, reused
    /// uniform buffer) keyed by `(pipeline, operand handles)`, so a call
    /// with the same operand buffers as a recent call — the common case in a
    /// training/inference loop — reuses both instead of allocating a fresh
    /// uniform buffer and bind group every single dispatch.  See `cache.rs`
    /// (`WebGpuBackend::cached_bind_group`) for the implementation and its
    /// safety argument.
    bind_group_cache: Mutex<BindGroupCache>,
}

impl WebGpuBackend {
    /// Create a new, uninitialised WebGPU backend.
    pub fn new() -> Self {
        Self {
            device: None,
            memory: None,
            initialized: false,
            pipeline_cache: Mutex::new(HashMap::new()),
            bind_group_cache: Mutex::new(BindGroupCache::new()),
        }
    }

    /// Return an error if the backend is not yet initialised.
    fn check_init(&self) -> BackendResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(BackendError::NotInitialized)
        }
    }

    /// Convenience accessor: get the memory manager or return `NotInitialized`.
    fn memory(&self) -> BackendResult<&Arc<WebGpuMemoryManager>> {
        self.memory.as_ref().ok_or(BackendError::NotInitialized)
    }

    /// Convenience accessor: get the device or return `NotInitialized`.
    fn device(&self) -> BackendResult<&Arc<WebGpuDevice>> {
        self.device.as_ref().ok_or(BackendError::NotInitialized)
    }

    /// Whether the initialised device enabled the `SHADER_F16` feature, i.e.
    /// whether [`gemm_f16`](Self::gemm_f16) can run instead of returning
    /// [`BackendError::Unsupported`].  Returns `false` (never errors) before
    /// `init()` — callers that want to skip f16-only tests on an
    /// uninitialised or non-f16 backend can check this directly.
    #[must_use]
    pub fn supports_f16(&self) -> bool {
        self.device.as_ref().is_some_and(|d| d.supports_f16)
    }

    /// Multi-dimensional reduce along a single axis.
    ///
    /// The tensor is logically reshaped to `[outer, dk, inner]`:
    /// * `outer` = product of dimensions before the reduce axis,
    /// * `dk`    = the reduce axis length,
    /// * `inner` = product of dimensions after the reduce axis.
    ///
    /// One workgroup of 256 threads is dispatched per `(o, j)` output slot.
    /// To stay within WebGPU's 65 535-per-axis dispatch limit a 2-D grid is
    /// used and the workgroup decodes its linear slot internally.
    ///
    /// `Mean` is handled inside the shader (divide by `dk`); the host does
    /// not need a post-pass.
    fn reduce_nd(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        // Caller (`reduce`) already validated `shape.is_empty()` and
        // `axis < shape.len()`; assert in debug to catch regressions but
        // recompute defensively in release as well.
        debug_assert!(!shape.is_empty());
        debug_assert!(axis < shape.len());

        // Output shape = shape with `axis` removed; length = outer * inner.
        let outer: usize = shape[..axis].iter().product();
        let dk: usize = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();

        // Empty tensor — nothing to do.
        if outer == 0 || dk == 0 || inner == 0 {
            return Ok(());
        }

        let total = outer.checked_mul(inner).ok_or_else(|| {
            BackendError::InvalidArgument("reduce: outer * inner overflows usize".into())
        })?;
        let in_elems = outer
            .checked_mul(dk)
            .and_then(|v| v.checked_mul(inner))
            .ok_or_else(|| {
                BackendError::InvalidArgument("reduce: outer * dk * inner overflows usize".into())
            })?;

        // Strides in elements: row-major (C order) layout.
        let inner_stride: usize = 1;
        let dk_stride: usize = inner;
        let outer_stride: usize = dk
            .checked_mul(inner)
            .ok_or_else(|| BackendError::InvalidArgument("reduce: dk * inner overflows".into()))?;

        // Plan the dispatch grid: one workgroup per output slot, folded into a
        // 2-D grid if `total` would otherwise exceed the per-axis workgroup
        // cap (see [`planner::plan_dispatch_1d`]).  `grid_x` is threaded
        // through the `ReduceNdParams` uniform so the shader can decode
        // `wgid.y * grid_x + wgid.x` back to a linear slot.
        let limits = gpu_limits();
        let (grid, grid_x) = planner::plan_dispatch_1d(&limits, total as u64, 1)
            .map_err(BackendError::InvalidArgument)?;

        let dev = self.device()?;
        let mem = self.memory()?;
        let op_str = map_reduce_op(op);
        let pipeline_key = format!("reduce_nd:{op_str}");

        let cached = self.cached_pipeline(&pipeline_key, "oxicuda-reduce-nd", || {
            shader::reduction_nd_wgsl(op_str)
        })?;

        // Build the uniform buffer: 8 × u32 = 32 bytes (16-byte aligned).
        let mut params_bytes = [0u8; 32];
        let outer_u32: u32 = outer
            .try_into()
            .map_err(|_| BackendError::InvalidArgument("reduce: outer exceeds u32 range".into()))?;
        let dk_u32: u32 = dk
            .try_into()
            .map_err(|_| BackendError::InvalidArgument("reduce: dk exceeds u32 range".into()))?;
        let inner_u32: u32 = inner
            .try_into()
            .map_err(|_| BackendError::InvalidArgument("reduce: inner exceeds u32 range".into()))?;
        let outer_stride_u32: u32 = outer_stride.try_into().map_err(|_| {
            BackendError::InvalidArgument("reduce: outer_stride exceeds u32 range".into())
        })?;
        let dk_stride_u32: u32 = dk_stride.try_into().map_err(|_| {
            BackendError::InvalidArgument("reduce: dk_stride exceeds u32 range".into())
        })?;
        let inner_stride_u32: u32 = inner_stride.try_into().map_err(|_| {
            BackendError::InvalidArgument("reduce: inner_stride exceeds u32 range".into())
        })?;
        params_bytes[0..4].copy_from_slice(&outer_u32.to_le_bytes());
        params_bytes[4..8].copy_from_slice(&dk_u32.to_le_bytes());
        params_bytes[8..12].copy_from_slice(&inner_u32.to_le_bytes());
        params_bytes[12..16].copy_from_slice(&outer_stride_u32.to_le_bytes());
        params_bytes[16..20].copy_from_slice(&dk_stride_u32.to_le_bytes());
        params_bytes[20..24].copy_from_slice(&inner_stride_u32.to_le_bytes());
        params_bytes[24..28].copy_from_slice(&grid_x.to_le_bytes());
        // bytes 28..32 are zero padding.

        // The shader trusts `shape`/`axis` to describe the buffers it is
        // bound to; an undersized buffer would otherwise silently drop
        // writes (WGSL robust access) or read stale/foreign data rather than
        // error.  `in_elems`/`total` are exactly the element counts the
        // kernel indexes into `input`/`output`.
        let need_in = (in_elems as u64) * 4;
        let need_out = (total as u64) * 4;
        let bind_group = self.cached_bind_group(
            dev,
            mem,
            &cached.bind_group_layout,
            &pipeline_key,
            &[input_ptr, output_ptr],
            &[need_in, need_out],
            &params_bytes,
            "oxicuda-reduce-nd",
        )?;

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-reduce-nd"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-reduce-nd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.x, grid.y, grid.z);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        // No per-op poll: wgpu executes queue submissions in FIFO order, so a
        // later dispatch reading this op's output, or a host readback via
        // `WebGpuMemoryManager::copy_from_device` / `synchronize()`, is
        // correctly ordered without a host-side wait here. Polling after
        // every submit was a full CPU/GPU pipeline stall on every op (see
        // "wgpu blocks on device.poll(wait_indefinitely()) after EVERY
        // submit" in the performance audit); `copy_from_device` now waits on
        // its own precise `SubmissionIndex` and `synchronize()` still waits
        // for all outstanding work.

        Ok(())
    }
}

impl WebGpuBackend {
    /// FP16 GEMM: `C = alpha * op(A) * op(B) + beta * C` with half-precision
    /// storage (accumulated in f32).
    ///
    /// This is an inherent method (not on `ComputeBackend`) because FP16
    /// support is WebGPU-specific and requires the `f16` WGSL extension.
    ///
    /// Buffers pointed to by `a_ptr`, `b_ptr`, `c_ptr` must contain `f16`
    /// elements (2 bytes each).  `lda` / `ldb` / `ldc` and `trans_a` /
    /// `trans_b` follow exactly the same convention as
    /// [`gemm`](ComputeBackend::gemm) — see [`shader::gemm_wgsl_f16`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
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
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        let dev = self.device()?;
        let mem = self.memory()?;

        // The FP16 GEMM shader declares `enable f16;`; naga rejects that module
        // unless the device enabled the SHADER_F16 feature.  Fail loudly with a
        // typed error instead of emitting an invalid module (which surfaces as a
        // process-fatal uncaptured validation error).
        if !dev.supports_f16 {
            return Err(BackendError::Unsupported(
                "f16 GEMM requires the SHADER_F16 device feature, \
                 which this adapter does not support"
                    .into(),
            ));
        }

        let trans_a_flag: u32 = u32::from(trans_a != BackendTranspose::NoTrans);
        let trans_b_flag: u32 = u32::from(trans_b != BackendTranspose::NoTrans);

        let (expected_lda, expected_ldb, expected_ldc) = packed_gemm_lds(trans_a, trans_b, m, n, k);
        if lda < expected_lda || ldb < expected_ldb || ldc < expected_ldc {
            return Err(BackendError::InvalidArgument(
                "gemm_f16: leading dimension smaller than matrix extent".into(),
            ));
        }
        let m_u32 = dim_u32("gemm_f16", "m", m)?;
        let n_u32 = dim_u32("gemm_f16", "n", n)?;
        let k_u32 = dim_u32("gemm_f16", "k", k)?;
        let lda_u32 = dim_u32("gemm_f16", "lda", lda)?;
        let ldb_u32 = dim_u32("gemm_f16", "ldb", ldb)?;
        let ldc_u32 = dim_u32("gemm_f16", "ldc", ldc)?;

        let limits = gpu_limits();
        let tile = planner::plan_workgroup_square(&limits, 16);
        let tile_size = tile.x;
        // Fail fast on a dispatch grid that would exceed the per-axis
        // workgroup cap, before creating any pipeline/buffer/bind-group GPU
        // state for a dispatch that could never legally run.
        let grid = planner::plan_dispatch_2d(&limits, m_u32, n_u32, tile, 1)
            .map_err(BackendError::InvalidArgument)?;
        let pipeline_key = format!("gemm_f16:{tile_size}");
        let cached = self.cached_pipeline(&pipeline_key, "oxicuda-gemm-f16", || {
            shader::gemm_wgsl_f16(tile_size)
        })?;

        // Build uniform buffer for GemmParams { m, n, k, alpha, beta, trans_a,
        // trans_b, lda, ldb, ldc, _pad0, _pad1 } — 12 × 4 = 48 bytes, mirroring
        // the f32 `GemmParams` layout in `shader::gemm_wgsl`.
        let mut params_bytes = [0u8; 48];
        params_bytes[0..4].copy_from_slice(&m_u32.to_le_bytes());
        params_bytes[4..8].copy_from_slice(&n_u32.to_le_bytes());
        params_bytes[8..12].copy_from_slice(&k_u32.to_le_bytes());
        params_bytes[12..16].copy_from_slice(&(alpha as f32).to_le_bytes());
        params_bytes[16..20].copy_from_slice(&(beta as f32).to_le_bytes());
        params_bytes[20..24].copy_from_slice(&trans_a_flag.to_le_bytes());
        params_bytes[24..28].copy_from_slice(&trans_b_flag.to_le_bytes());
        params_bytes[28..32].copy_from_slice(&lda_u32.to_le_bytes());
        params_bytes[32..36].copy_from_slice(&ldb_u32.to_le_bytes());
        params_bytes[36..40].copy_from_slice(&ldc_u32.to_le_bytes());
        // bytes 40..48 are zero padding.

        let bind_group = self.cached_bind_group(
            dev,
            mem,
            &cached.bind_group_layout,
            &pipeline_key,
            &[a_ptr, b_ptr, c_ptr],
            &[],
            &params_bytes,
            "oxicuda-gemm-f16",
        )?;

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-gemm-f16"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-gemm-f16"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.x, grid.y, grid.z);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        // No per-op poll: wgpu executes queue submissions in FIFO order, so a
        // later dispatch reading this op's output, or a host readback via
        // `WebGpuMemoryManager::copy_from_device` / `synchronize()`, is
        // correctly ordered without a host-side wait here. Polling after
        // every submit was a full CPU/GPU pipeline stall on every op (see
        // "wgpu blocks on device.poll(wait_indefinitely()) after EVERY
        // submit" in the performance audit); `copy_from_device` now waits on
        // its own precise `SubmissionIndex` and `synchronize()` still waits
        // for all outstanding work.

        Ok(())
    }
}

impl Default for WebGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ComputeBackend impl ─────────────────────────────────────────────────────

impl ComputeBackend for WebGpuBackend {
    fn name(&self) -> &str {
        "webgpu"
    }

    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }

        match WebGpuDevice::new() {
            Ok(dev) => {
                let dev = Arc::new(dev);
                tracing::info!("WebGPU backend initialised on: {}", dev.adapter_name);
                let memory = WebGpuMemoryManager::new(Arc::clone(&dev));
                self.device = Some(dev);
                self.memory = Some(Arc::new(memory));
                self.initialized = true;
                Ok(())
            }
            Err(e) => Err(BackendError::from(e)),
        }
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    // ── Compute operations ────────────────────────────────────────────────────

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
        // Zero-dimension matrices are trivially done.
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        // The WGSL tiled GEMM kernel handles every NN / NT / TN / TT
        // combination at runtime via the `trans_a` / `trans_b` uniforms.
        // `ConjTrans` collapses to `Trans` because the f32 buffers are real.
        let trans_a_flag: u32 = u32::from(trans_a != BackendTranspose::NoTrans);
        let trans_b_flag: u32 = u32::from(trans_b != BackendTranspose::NoTrans);

        let dev = self.device()?;
        let mem = self.memory()?;

        // Mirror the lda/ldb/ldc validation onto every other dimension that
        // feeds the shader uniform: a value above `u32::MAX` must be a clean
        // typed error, never a silent wraparound into a small (wrong) count.
        let m_u32 = dim_u32("gemm", "m", m)?;
        let n_u32 = dim_u32("gemm", "n", n)?;
        let k_u32 = dim_u32("gemm", "k", k)?;

        let limits = gpu_limits();
        let tile = planner::plan_workgroup_square(&limits, 16);
        let tile_size = tile.x;
        // Fail fast on a dispatch grid that would exceed the per-axis
        // workgroup cap, before creating any pipeline/buffer/bind-group GPU
        // state for a dispatch that could never legally run.
        let grid = planner::plan_dispatch_2d(&limits, m_u32, n_u32, tile, 1)
            .map_err(BackendError::InvalidArgument)?;
        let pipeline_key = format!("gemm:{tile_size}");
        let cached = self.cached_pipeline(&pipeline_key, "oxicuda-gemm", || {
            shader::gemm_wgsl(tile_size)
        })?;

        // The row-major WGSL kernel honours the leading dimensions carried in
        // `GemmParams`; validate they are at least the packed extent (the same
        // check the CPU reference backend performs) so a too-small stride is a
        // clean error rather than an out-of-bounds read.
        let (expected_lda, expected_ldb, expected_ldc) = packed_gemm_lds(trans_a, trans_b, m, n, k);
        if lda < expected_lda || ldb < expected_ldb || ldc < expected_ldc {
            return Err(BackendError::InvalidArgument(
                "gemm: leading dimension smaller than matrix extent".into(),
            ));
        }
        let lda_u32 = dim_u32("gemm", "lda", lda)?;
        let ldb_u32 = dim_u32("gemm", "ldb", ldb)?;
        let ldc_u32 = dim_u32("gemm", "ldc", ldc)?;

        // Build uniform buffer for GemmParams { m, n, k, alpha, beta,
        // trans_a, trans_b, lda, ldb, ldc, _pad } — 12 × 4 = 48 bytes.
        let mut params_bytes = [0u8; 48];
        params_bytes[0..4].copy_from_slice(&m_u32.to_le_bytes());
        params_bytes[4..8].copy_from_slice(&n_u32.to_le_bytes());
        params_bytes[8..12].copy_from_slice(&k_u32.to_le_bytes());
        params_bytes[12..16].copy_from_slice(&(alpha as f32).to_le_bytes());
        params_bytes[16..20].copy_from_slice(&(beta as f32).to_le_bytes());
        params_bytes[20..24].copy_from_slice(&trans_a_flag.to_le_bytes());
        params_bytes[24..28].copy_from_slice(&trans_b_flag.to_le_bytes());
        params_bytes[28..32].copy_from_slice(&lda_u32.to_le_bytes());
        params_bytes[32..36].copy_from_slice(&ldb_u32.to_le_bytes());
        params_bytes[36..40].copy_from_slice(&ldc_u32.to_le_bytes());
        // bytes 40..48 are zero padding.

        let bind_group = self.cached_bind_group(
            dev,
            mem,
            &cached.bind_group_layout,
            &pipeline_key,
            &[a_ptr, b_ptr, c_ptr],
            &[],
            &params_bytes,
            "oxicuda-gemm",
        )?;

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-gemm"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.x, grid.y, grid.z);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        // No per-op poll: wgpu executes queue submissions in FIFO order, so a
        // later dispatch reading this op's output, or a host readback via
        // `WebGpuMemoryManager::copy_from_device` / `synchronize()`, is
        // correctly ordered without a host-side wait here. Polling after
        // every submit was a full CPU/GPU pipeline stall on every op (see
        // "wgpu blocks on device.poll(wait_indefinitely()) after EVERY
        // submit" in the performance audit); `copy_from_device` now waits on
        // its own precise `SubmissionIndex` and `synchronize()` still waits
        // for all outstanding work.

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_gemm(
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
        self.check_init()?;

        if batch_count == 0 || m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        // The WGSL batched tiled GEMM kernel handles every NN / NT / TN / TT
        // combination at runtime via the `trans_a` / `trans_b` uniforms.
        // `ConjTrans` collapses to `Trans` because the f32 buffers are real.
        let trans_a_flag: u32 = u32::from(trans_a != BackendTranspose::NoTrans);
        let trans_b_flag: u32 = u32::from(trans_b != BackendTranspose::NoTrans);

        let dev = self.device()?;
        let mem = self.memory()?;

        // Mirror the lda/ldb/ldc validation onto every other dimension that
        // feeds the shader uniform or the dispatch grid.  Previously these
        // were cast with a bare `as u32`: a `stride_*` above `u32::MAX` wrapped
        // to a small value and the kernel silently read the wrong batch slice,
        // and `batch_count` fed the Z dispatch dimension completely unchecked
        // against the 65 535-per-axis limit (an oversized batch triggered a
        // fatal wgpu validation error with no uncaptured-error handler
        // installed).  `plan_dispatch_2d` below turns that overflow into a
        // clean typed `Err` instead.
        let m_u32 = dim_u32("batched_gemm", "m", m)?;
        let n_u32 = dim_u32("batched_gemm", "n", n)?;
        let k_u32 = dim_u32("batched_gemm", "k", k)?;
        let batch_u32 = dim_u32("batched_gemm", "batch_count", batch_count)?;
        let stride_a_u32 = dim_u32("batched_gemm", "stride_a", stride_a)?;
        let stride_b_u32 = dim_u32("batched_gemm", "stride_b", stride_b)?;
        let stride_c_u32 = dim_u32("batched_gemm", "stride_c", stride_c)?;

        let limits = gpu_limits();
        let tile = planner::plan_workgroup_square(&limits, 16);
        let tile_size = tile.x;
        // Fail fast — including the `batch_count > 65 535` case this
        // validation exists for — before creating any pipeline/buffer/
        // bind-group GPU state for a dispatch that could never legally run.
        let grid = planner::plan_dispatch_2d(&limits, m_u32, n_u32, tile, batch_u32)
            .map_err(BackendError::InvalidArgument)?;
        let pipeline_key = format!("batched_gemm:{tile_size}");
        let cached = self.cached_pipeline(&pipeline_key, "oxicuda-batched-gemm", || {
            shader::batched_gemm_wgsl(tile_size)
        })?;

        // Validate leading dimensions against the packed extents (per-batch row
        // strides) before threading them into the uniform.
        let (expected_lda, expected_ldb, expected_ldc) = packed_gemm_lds(trans_a, trans_b, m, n, k);
        if lda < expected_lda || ldb < expected_ldb || ldc < expected_ldc {
            return Err(BackendError::InvalidArgument(
                "batched_gemm: leading dimension smaller than matrix extent".into(),
            ));
        }
        let lda_u32 = dim_u32("batched_gemm", "lda", lda)?;
        let ldb_u32 = dim_u32("batched_gemm", "ldb", ldb)?;
        let ldc_u32 = dim_u32("batched_gemm", "ldc", ldc)?;

        // BatchedGemmParams: m, n, k, alpha, beta, batch_count, stride_a,
        // stride_b, stride_c, trans_a, trans_b, lda, ldb, ldc — 14 × 4 = 56
        // bytes.  Uniform buffers need 16-byte alignment, so 56 rounds up to 64.
        let mut params_bytes = [0u8; 64];
        params_bytes[0..4].copy_from_slice(&m_u32.to_le_bytes());
        params_bytes[4..8].copy_from_slice(&n_u32.to_le_bytes());
        params_bytes[8..12].copy_from_slice(&k_u32.to_le_bytes());
        params_bytes[12..16].copy_from_slice(&(alpha as f32).to_le_bytes());
        params_bytes[16..20].copy_from_slice(&(beta as f32).to_le_bytes());
        params_bytes[20..24].copy_from_slice(&batch_u32.to_le_bytes());
        params_bytes[24..28].copy_from_slice(&stride_a_u32.to_le_bytes());
        params_bytes[28..32].copy_from_slice(&stride_b_u32.to_le_bytes());
        params_bytes[32..36].copy_from_slice(&stride_c_u32.to_le_bytes());
        params_bytes[36..40].copy_from_slice(&trans_a_flag.to_le_bytes());
        params_bytes[40..44].copy_from_slice(&trans_b_flag.to_le_bytes());
        params_bytes[44..48].copy_from_slice(&lda_u32.to_le_bytes());
        params_bytes[48..52].copy_from_slice(&ldb_u32.to_le_bytes());
        params_bytes[52..56].copy_from_slice(&ldc_u32.to_le_bytes());
        // bytes 56..64 are padding zeros

        let bind_group = self.cached_bind_group(
            dev,
            mem,
            &cached.bind_group_layout,
            &pipeline_key,
            &[a_ptr, b_ptr, c_ptr],
            &[],
            &params_bytes,
            "oxicuda-batched-gemm",
        )?;

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-batched-gemm"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-batched-gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.x, grid.y, grid.z);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        // No per-op poll: wgpu executes queue submissions in FIFO order, so a
        // later dispatch reading this op's output, or a host readback via
        // `WebGpuMemoryManager::copy_from_device` / `synchronize()`, is
        // correctly ordered without a host-side wait here. Polling after
        // every submit was a full CPU/GPU pipeline stall on every op (see
        // "wgpu blocks on device.poll(wait_indefinitely()) after EVERY
        // submit" in the performance audit); `copy_from_device` now waits on
        // its own precise `SubmissionIndex` and `synchronize()` still waits
        // for all outstanding work.

        Ok(())
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

        if input_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "input_shape must have 4 elements (NCHW)".into(),
            ));
        }
        if filter_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "filter_shape must have 4 elements (KCFHFW)".into(),
            ));
        }
        if output_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "output_shape must have 4 elements (NKOhOw)".into(),
            ));
        }
        if stride.len() != 2 {
            return Err(BackendError::InvalidArgument(
                "stride must have 2 elements [sh, sw]".into(),
            ));
        }
        if padding.len() != 2 {
            return Err(BackendError::InvalidArgument(
                "padding must have 2 elements [ph, pw]".into(),
            ));
        }

        let batch = input_shape[0];
        let c_in = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];
        let k_out = filter_shape[0];
        let fh = filter_shape[2];
        let fw = filter_shape[3];
        let oh = output_shape[2];
        let ow = output_shape[3];
        let sh = stride[0];
        let sw = stride[1];
        let ph = padding[0];
        let pw = padding[1];

        let in_elems: usize = input_shape.iter().product();
        let f_elems: usize = filter_shape.iter().product();
        let o_elems: usize = output_shape.iter().product();

        // Prefer the GPU dispatch path; fall back to the CPU reference only
        // for configurations `shader::conv2d_wgsl`'s fixed 2-D dispatch
        // cannot address (see `conv2d_gpu_dispatch_grid`).
        if let Some((wg_x, wg_y)) = conv2d_gpu_dispatch_grid(batch, k_out, oh, ow) {
            if let Some(dims) = conv2d_u32_dims(
                batch, c_in, h_in, w_in, k_out, fh, fw, oh, ow, sh, sw, ph, pw,
            ) {
                return self.conv2d_forward_gpu(
                    input_ptr, filter_ptr, output_ptr, dims, in_elems, f_elems, o_elems, wg_x, wg_y,
                );
            }
        }

        // CPU fallback: download input + filter, compute, upload output.
        let mem = self.memory()?;
        let mut in_bytes = vec![0u8; in_elems * 4];
        let mut f_bytes = vec![0u8; f_elems * 4];
        mem.copy_from_device(&mut in_bytes, input_ptr)
            .map_err(BackendError::from)?;
        mem.copy_from_device(&mut f_bytes, filter_ptr)
            .map_err(BackendError::from)?;

        let in_f32 = bytes_to_f32_vec(&in_bytes);
        let f_f32 = bytes_to_f32_vec(&f_bytes);
        let out_f32 = conv2d_cpu_reference(
            &in_f32, &f_f32, batch, c_in, h_in, w_in, k_out, fh, fw, oh, ow, sh, sw, ph, pw,
        );

        let out_bytes = f32_slice_to_bytes(&out_f32);
        mem.copy_to_device(output_ptr, &out_bytes)
            .map_err(BackendError::from)?;

        Ok(())
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
                "seq_q, seq_kv, and head_dim must all be > 0".into(),
            ));
        }
        if scale <= 0.0 || !scale.is_finite() {
            return Err(BackendError::InvalidArgument(format!(
                "scale must be a positive finite number, got {scale}"
            )));
        }

        let batch_heads = batch * heads;
        let q_elems = batch_heads * seq_q * head_dim;
        let kv_elems = batch_heads * seq_kv * head_dim;
        let o_elems = q_elems;
        let scale_f32 = scale as f32;

        // Prefer the GPU dispatch path; fall back to the CPU reference only
        // for configurations `shader::attention_wgsl`'s fixed 1-D dispatch
        // cannot address, or whose shape overflows u32 (the shader bakes
        // every dimension as a literal).
        if let (Some(wg), Some(bh_u32), Some(seq_q_u32), Some(seq_kv_u32), Some(head_dim_u32)) = (
            attention_gpu_dispatch_grid(batch_heads, seq_q),
            u32::try_from(batch_heads).ok(),
            u32::try_from(seq_q).ok(),
            u32::try_from(seq_kv).ok(),
            u32::try_from(head_dim).ok(),
        ) {
            return self.attention_gpu(
                q_ptr,
                k_ptr,
                v_ptr,
                o_ptr,
                bh_u32,
                seq_q_u32,
                seq_kv_u32,
                head_dim_u32,
                scale_f32,
                causal,
                q_elems,
                kv_elems,
                o_elems,
                wg,
            );
        }

        // CPU fallback: download Q, K, V, compute attention, upload O.
        let mem = self.memory()?;
        let mut q_bytes = vec![0u8; q_elems * 4];
        let mut k_bytes = vec![0u8; kv_elems * 4];
        let mut v_bytes = vec![0u8; kv_elems * 4];

        mem.copy_from_device(&mut q_bytes, q_ptr)
            .map_err(BackendError::from)?;
        mem.copy_from_device(&mut k_bytes, k_ptr)
            .map_err(BackendError::from)?;
        mem.copy_from_device(&mut v_bytes, v_ptr)
            .map_err(BackendError::from)?;

        let q_f32 = bytes_to_f32_vec(&q_bytes);
        let k_f32 = bytes_to_f32_vec(&k_bytes);
        let v_f32 = bytes_to_f32_vec(&v_bytes);
        let o_f32 = attention_cpu_reference(
            &q_f32,
            &k_f32,
            &v_f32,
            batch_heads,
            seq_q,
            seq_kv,
            head_dim,
            scale_f32,
            causal,
        );

        let o_bytes = f32_slice_to_bytes(&o_f32);
        mem.copy_to_device(o_ptr, &o_bytes)
            .map_err(BackendError::from)?;

        Ok(())
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
                "axis {axis} is out of bounds for shape of length {}",
                shape.len()
            )));
        }

        // 1-D shapes (or any shape that reduces to a single scalar) take the
        // optimised two-pass scalar path.  Higher-rank shapes go through the
        // batched N-D shader below.
        if shape.len() != 1 {
            return self.reduce_nd(op, input_ptr, output_ptr, shape, axis);
        }

        let n_elements = shape[0];
        if n_elements == 0 {
            return Ok(());
        }

        let dev = self.device()?;
        let mem = self.memory()?;
        let op_str = map_reduce_op(op);

        // ── Pass 1: per-workgroup reduction ─────────────────────────────────
        // `reduction_wgsl`'s `@workgroup_size(256)` kernel decodes only
        // `global_invocation_id.x` (no 2-D dispatch fold, unlike
        // `reduction_nd_wgsl`), so `wg_count` must itself fit in one dispatch
        // axis.  `plan_dispatch_1d` both computes it and turns an
        // over-capacity `n_elements` into a clean typed error instead of an
        // invalid `dispatch_workgroups` call.
        let limits = gpu_limits();
        let (wg_grid, _) = planner::plan_dispatch_1d(&limits, n_elements as u64, 256)
            .map_err(BackendError::InvalidArgument)?;
        if wg_grid.y != 1 {
            return Err(BackendError::InvalidArgument(format!(
                "reduce: {n_elements} elements need {} workgroups, which exceeds the \
                 single-axis dispatch capacity of this 1-D reduction kernel",
                wg_grid.x as u64 * wg_grid.y as u64
            )));
        }
        let wg_count = wg_grid.x;

        let pass1_cached = self.cached_pipeline(
            &format!("reduce_pass1:{op_str}"),
            "oxicuda-reduce-pass1",
            || shader::reduction_wgsl(op_str),
        )?;

        // Partial-sums buffer (temporary).
        let partial_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-reduce-partial"),
            size: (wg_count as u64) * 4, // f32 per workgroup
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform for ReduceParams { n: u32 }.
        let mut p1_params = [0u8; 4];
        p1_params[0..4].copy_from_slice(&(n_elements as u32).to_le_bytes());
        let p1_uniform = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-reduce-p1-params"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&p1_uniform, 0, &p1_params);

        let bgl1 = &pass1_cached.bind_group_layout;

        let bg1 = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let in_info = buffers.get(&input_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {input_ptr}"))
            })?;

            let need_in = (n_elements as u64) * 4;
            if in_info.size < need_in {
                return Err(BackendError::InvalidArgument(format!(
                    "reduce: input buffer holds {} bytes, need {need_in} for {n_elements} f32 elements",
                    in_info.size
                )));
            }

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-reduce-pass1"),
                layout: bgl1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: partial_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p1_uniform.as_entire_binding(),
                    },
                ],
            })
        };

        // ── Pass 2: final reduction of partial sums ─────────────────────────
        let pass2_cached = self.cached_pipeline(
            &format!("reduce_pass2:{op_str}"),
            "oxicuda-reduce-pass2",
            || shader::reduction_final_wgsl(op_str),
        )?;

        // FinalReduceParams { num_groups: u32 }.
        let mut p2_params = [0u8; 4];
        p2_params[0..4].copy_from_slice(&wg_count.to_le_bytes());
        let p2_uniform = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-reduce-p2-params"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&p2_uniform, 0, &p2_params);

        let bgl2 = &pass2_cached.bind_group_layout;

        let bg2 = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let out_info = buffers.get(&output_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {output_ptr}"))
            })?;

            // The scalar output slot is a single f32.
            if out_info.size < 4 {
                return Err(BackendError::InvalidArgument(format!(
                    "reduce: output buffer holds {} bytes, need 4 for the scalar result",
                    out_info.size
                )));
            }

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-reduce-pass2"),
                layout: bgl2,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: partial_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p2_uniform.as_entire_binding(),
                    },
                ],
            })
        };

        // ── Encode both passes into one command buffer ──────────────────────
        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-reduce"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-reduce-pass1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pass1_cached.pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-reduce-pass2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pass2_cached.pipeline);
            pass.set_bind_group(0, &bg2, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        // No per-op poll: wgpu executes queue submissions in FIFO order, so a
        // later dispatch reading this op's output, or a host readback via
        // `WebGpuMemoryManager::copy_from_device` / `synchronize()`, is
        // correctly ordered without a host-side wait here. Polling after
        // every submit was a full CPU/GPU pipeline stall on every op (see
        // "wgpu blocks on device.poll(wait_indefinitely()) after EVERY
        // submit" in the performance audit); `copy_from_device` now waits on
        // its own precise `SubmissionIndex` and `synchronize()` still waits
        // for all outstanding work.

        // For "mean", divide the result by N on the host side.
        if op == ReduceOp::Mean && n_elements > 1 {
            let mut buf = [0u8; 4];
            mem.copy_from_device(&mut buf, output_ptr)
                .map_err(BackendError::from)?;
            let val = f32::from_le_bytes(buf);
            let mean = val / (n_elements as f32);
            mem.copy_to_device(output_ptr, &mean.to_le_bytes())
                .map_err(BackendError::from)?;
        }

        Ok(())
    }

    fn unary(&self, op: UnaryOp, input_ptr: u64, output_ptr: u64, n: usize) -> BackendResult<()> {
        self.check_init()?;
        if n == 0 {
            return Ok(());
        }
        // `elementwise_wgsl`'s bind group declares `input` (binding 0) as
        // `read` and `output` (binding 1) as `read_write`; if the two
        // handles are the same buffer, wgpu's usage-scope validation rejects
        // the dispatch outright ("conflicting usages: STORAGE_READ_ONLY vs
        // STORAGE_READ_WRITE"). Reject it here with a typed, attributable
        // error instead of letting it surface later — as a generic
        // `DeviceError` from an unrelated caller's `alloc`/`copy_*`/
        // `synchronize()`, whichever happens to be the next call that drains
        // the recorded uncaptured error — which is what happened before this
        // check existed.
        if input_ptr == output_ptr {
            return Err(BackendError::InvalidArgument(
                "unary: input_ptr and output_ptr must not alias (wgpu rejects binding the \
                 same buffer as both `read` and `read_write` within one dispatch); allocate \
                 a separate output buffer"
                    .into(),
            ));
        }

        let dev = self.device()?;
        let mem = self.memory()?;

        // `elementwise_wgsl`'s `@workgroup_size(256)` kernel decodes only
        // `global_invocation_id.x` (no 2-D dispatch fold), so `n` must map to
        // a workgroup count that fits a single dispatch axis.
        let (wg_grid, _) = planner::plan_dispatch_1d(&gpu_limits(), n as u64, 256)
            .map_err(BackendError::InvalidArgument)?;
        if wg_grid.y != 1 {
            return Err(BackendError::InvalidArgument(format!(
                "unary: {n} elements exceed the single-axis dispatch capacity of this kernel"
            )));
        }

        let op_str = map_unary_op(op);
        let pipeline_key = format!("unary:{op_str}");
        let cached = self.cached_pipeline(&pipeline_key, "oxicuda-unary", || {
            shader::elementwise_wgsl(op_str)
        })?;

        // `elementwise_wgsl` has no uniform binding at all — `n` is derived
        // in-shader via `arrayLength`, so `uniform_bytes` is empty.
        let bind_group = self.cached_bind_group(
            dev,
            mem,
            &cached.bind_group_layout,
            &pipeline_key,
            &[input_ptr, output_ptr],
            &[],
            &[],
            "oxicuda-unary",
        )?;

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-unary"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-unary"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_grid.x, 1, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        // No per-op poll: wgpu executes queue submissions in FIFO order, so a
        // later dispatch reading this op's output, or a host readback via
        // `WebGpuMemoryManager::copy_from_device` / `synchronize()`, is
        // correctly ordered without a host-side wait here. Polling after
        // every submit was a full CPU/GPU pipeline stall on every op (see
        // "wgpu blocks on device.poll(wait_indefinitely()) after EVERY
        // submit" in the performance audit); `copy_from_device` now waits on
        // its own precise `SubmissionIndex` and `synchronize()` still waits
        // for all outstanding work.

        Ok(())
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
            return Ok(());
        }
        // Same usage-scope hazard as `unary` (see the comment there):
        // `binary_wgsl` binds `lhs`/`rhs` (0/1) `read` and `output` (2)
        // `read_write`. `a_ptr == b_ptr` (both inputs aliased to each other)
        // is fine — two `read` usages of the same buffer do not conflict —
        // only the output aliasing either input is rejected.
        if a_ptr == output_ptr || b_ptr == output_ptr {
            return Err(BackendError::InvalidArgument(
                "binary: a_ptr/b_ptr must not alias output_ptr (wgpu rejects binding the \
                 same buffer as both `read` and `read_write` within one dispatch); allocate \
                 a separate output buffer"
                    .into(),
            ));
        }

        let dev = self.device()?;
        let mem = self.memory()?;

        // Same single-axis dispatch constraint as `unary` (see comment there).
        let (wg_grid, _) = planner::plan_dispatch_1d(&gpu_limits(), n as u64, 256)
            .map_err(BackendError::InvalidArgument)?;
        if wg_grid.y != 1 {
            return Err(BackendError::InvalidArgument(format!(
                "binary: {n} elements exceed the single-axis dispatch capacity of this kernel"
            )));
        }

        let op_str = map_binary_op(op);
        let pipeline_key = format!("binary:{op_str}");
        let cached = self.cached_pipeline(&pipeline_key, "oxicuda-binary", || {
            shader::binary_wgsl(op_str)
        })?;

        // `binary_wgsl` has no uniform binding either (see `unary` above).
        let bind_group = self.cached_bind_group(
            dev,
            mem,
            &cached.bind_group_layout,
            &pipeline_key,
            &[a_ptr, b_ptr, output_ptr],
            &[],
            &[],
            "oxicuda-binary",
        )?;

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-binary"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-binary"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_grid.x, 1, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        // No per-op poll: wgpu executes queue submissions in FIFO order, so a
        // later dispatch reading this op's output, or a host readback via
        // `WebGpuMemoryManager::copy_from_device` / `synchronize()`, is
        // correctly ordered without a host-side wait here. Polling after
        // every submit was a full CPU/GPU pipeline stall on every op (see
        // "wgpu blocks on device.poll(wait_indefinitely()) after EVERY
        // submit" in the performance audit); `copy_from_device` now waits on
        // its own precise `SubmissionIndex` and `synchronize()` still waits
        // for all outstanding work.

        Ok(())
    }

    // ── Synchronisation ───────────────────────────────────────────────────────

    fn synchronize(&self) -> BackendResult<()> {
        self.check_init()?;
        if let Some(dev) = &self.device {
            // This is now the *only* completion signal for a caller that
            // issues pure compute dispatches (`gemm`, `unary`, …) and skips
            // `copy_dtoh` — those ops no longer poll themselves (see the
            // comment on every `dev.queue.submit(...)` call site: wgpu's
            // queue-FIFO ordering makes a per-op poll unnecessary). A bare
            // `let _ = dev.device.poll(...)` would silently discard a
            // `PollError` (device hung or lost) exactly where a caller is
            // relying on this call to be the wait; propagate it as a typed
            // error instead, reusing the same mapping `copy_from_device`
            // uses for its own indexed wait.
            crate::memory::poll_result_to_webgpu_result(
                dev.device.poll(wgpu::PollType::wait_indefinitely()),
            )
            .map_err(BackendError::from)?;

            // Drain any uncaptured wgpu error recorded by the work this wait
            // just observed completing (validation / OOM / internal errors
            // wgpu's non-fatal handler captured instead of aborting the
            // process — see `WebGpuDevice::poll_error`) so it reaches the
            // caller instead of being silently lost.
            if let Some(msg) = dev.poll_error() {
                return Err(BackendError::from(
                    crate::error::WebGpuError::UncapturedError(msg),
                ));
            }
        }
        Ok(())
    }

    // ── Memory management ─────────────────────────────────────────────────────

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        self.check_init()?;
        if bytes == 0 {
            return Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into(),
            ));
        }
        self.memory()?.alloc(bytes).map_err(BackendError::from)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.check_init()?;
        // Evict any cached bind group that still points at `ptr` *before*
        // actually freeing it: a cached `wgpu::BindGroup` retains a strong
        // reference to every buffer it binds, so leaving a stale entry in
        // place would keep this handle's GPU memory alive indefinitely
        // despite `free()` having (logically) released it. See `cache.rs`'s
        // module doc, "Freed-buffer memory".
        self.evict_bind_group_cache(ptr)?;
        self.memory()?.free(ptr).map_err(BackendError::from)
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        self.check_init()?;
        if src.is_empty() {
            return Ok(());
        }
        self.memory()?
            .copy_to_device(dst, src)
            .map_err(BackendError::from)
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        self.check_init()?;
        if dst.is_empty() {
            return Ok(());
        }
        self.memory()?
            .copy_from_device(dst, src)
            .map_err(BackendError::from)
    }
}

// ─── Byte ↔ f32 helpers ──────────────────────────────────────────────────────

/// Convert a `&[u8]` (length must be a multiple of 4) to a `Vec<f32>`.
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Convert a `&[f32]` slice to its little-endian byte representation.
fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────
//
// The test module lives in a sibling file (`backend_tests.rs`) so the
// production code in this file stays under the 2 000-line refactoring policy.
#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;
