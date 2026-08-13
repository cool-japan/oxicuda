//! # MetalBackend - Trait Implementations
//!
//! This module contains trait implementations for `MetalBackend`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `ComputeBackend`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::Arc;

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, Capabilities, ComputeBackend,
    DeviceInfo, MemoryKind, ReduceOp, UnaryOp,
};

use crate::{device::MetalDevice, memory::MetalMemoryManager};

use super::nn::{AttentionGeometry, Conv2dGeometry};
use super::types::MetalBackend;

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Translate a live [`MetalDevice`]'s probed capability snapshot into the
/// backend-agnostic [`Capabilities`] shape.
///
/// Every value comes from `[MTLDevice supportsFamily:]` plus the driver-reported
/// threadgroup limits (see [`MetalDevice::capabilities`]), never from a guess.
/// Before the backend is initialised there is no device to ask, so the
/// conservative CPU profile is reported — the same thing the trait's own default
/// would return.
///
/// Field-by-field rationale for the non-obvious entries:
///
/// * `supports_fp16` — MSL's `half` is a first-class scalar on every
///   Metal-capable GPU, and [`MetalBackend::gemm_f16`] dispatches it.
/// * `supports_bf16` / `supports_fp8` — `false`: MSL `bfloat` needs Metal 3.1
///   and this crate ships no bf16 or fp8 kernel, so claiming either would be a
///   lie the consumer acts on.
/// * `tensor_cores` — mapped from `simdgroup_matrix` (Apple family 7+, i.e.
///   Metal 3 `simdgroup_float8x8` MMA), the closest hardware analogue to WMMA.
/// * `peer_access` — `false`: this backend drives the single system-default
///   device and has no multi-GPU path.
/// * `cluster_launch` / `async_copy` — `false`: Metal exposes no thread-block
///   clusters and no `cp.async` equivalent.
fn metal_capabilities(device: Option<&MetalDevice>) -> Capabilities {
    let Some(device) = device else {
        return Capabilities::default();
    };
    let caps = device.capabilities();
    Capabilities {
        supports_fp16: true,
        supports_bf16: false,
        supports_fp8: false,
        tensor_cores: caps.simdgroup_matrix,
        peer_access: false,
        unified_memory: caps.unified_memory,
        cluster_launch: false,
        async_copy: false,
        max_threads_per_block: u32::try_from(caps.max_threads_per_threadgroup).unwrap_or(u32::MAX),
        max_shared_mem_per_block: u32::try_from(caps.threadgroup_memory).unwrap_or(u32::MAX),
        warp_size: u32::try_from(caps.family.simd_width()).unwrap_or(32),
    }
}

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }
    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }
        match MetalDevice::new() {
            Ok(dev) => {
                let dev = Arc::new(dev);
                tracing::info!("Metal backend initialised on: {}", dev.name());
                let memory = MetalMemoryManager::new(Arc::clone(&dev));
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
    /// `C = alpha * op(A) * op(B) + beta * C` on the GPU.
    ///
    /// # Element type and layout — read this before mixing backends
    ///
    /// The operands are **`f32`, row-major**, with `lda`/`ldb`/`ldc` as physical
    /// row strides that may exceed the packed minimum (a padded or sub-matrix
    /// view). All four transpose combinations are honoured; the flags travel to
    /// the kernel in a runtime parameter buffer.
    ///
    /// This deliberately matches `oxicuda_webgpu`'s `gemm` **exactly**, so the
    /// two GPU backends accept the same argument space and produce the same
    /// numbers. It does **not** match the
    /// [`ComputeBackend::gemm`] trait
    /// doc's "column-major `f64`" wording, which describes
    /// `oxicuda_backend::CpuBackend` — the reference implementation — and not
    /// the GPU backends. (The trait is inconsistent with itself here: its own
    /// default `batched_gemm` offsets pointers with `elem_bytes = 4`, i.e.
    /// `f32`.) Metal has no `f64` type at all, so honouring the literal wording
    /// would mean either emulating it in software or refusing every call;
    /// matching the sibling GPU backend and saying so is the honest option.
    ///
    /// # Errors
    /// * [`BackendError::NotInitialized`] before `init`.
    /// * [`BackendError::InvalidArgument`] if any leading dimension is smaller
    ///   than the stored row it describes, or a dimension exceeds `u32`.
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
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        super::types::validate_gemm_layout(trans_a, trans_b, m, n, k, lda, ldb, ldc)?;
        self.dispatch_gemm(
            trans_a, trans_b, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc,
        )
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
        if stride[0] == 0 || stride[1] == 0 {
            return Err(BackendError::InvalidArgument(
                "stride entries must be non-zero; a zero stride makes every output element \
                 read the same input window"
                    .into(),
            ));
        }
        if input_shape[1] != filter_shape[1] {
            return Err(BackendError::InvalidArgument(format!(
                "input channels ({}) must equal filter channels ({})",
                input_shape[1], filter_shape[1]
            )));
        }
        if output_shape[0] != input_shape[0] || output_shape[1] != filter_shape[0] {
            return Err(BackendError::InvalidArgument(format!(
                "output_shape {output_shape:?} must be [N={}, K={}, Oh, Ow]",
                input_shape[0], filter_shape[0]
            )));
        }
        self.dispatch_conv2d(
            input_ptr,
            filter_ptr,
            output_ptr,
            Conv2dGeometry {
                n: input_shape[0],
                c_in: input_shape[1],
                h_in: input_shape[2],
                w_in: input_shape[3],
                k_out: filter_shape[0],
                fh: filter_shape[2],
                fw: filter_shape[3],
                oh: output_shape[2],
                ow: output_shape[3],
                stride_h: stride[0],
                stride_w: stride[1],
                pad_h: padding[0],
                pad_w: padding[1],
            },
        )
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
        let batch_heads = batch.checked_mul(heads).ok_or_else(|| {
            BackendError::InvalidArgument("attention: batch * heads overflows".into())
        })?;
        self.dispatch_attention(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            AttentionGeometry {
                batch_heads,
                seq_q,
                seq_kv,
                head_dim,
                scale,
                causal,
            },
        )
    }
    /// Numerically-stable softmax along `axis`, on the GPU.
    ///
    /// # Supported axes
    ///
    /// [`crate::msl_nn::softmax_msl`] is a **row-wise** kernel: one threadgroup
    /// per row of a `rows × cols` contiguous matrix. That maps exactly onto
    /// `axis == shape.len() - 1`, where the reduced elements are adjacent in
    /// memory. Any earlier axis has an inner stride the kernel cannot express,
    /// so it is rejected with [`BackendError::Unsupported`] rather than
    /// reinterpreted as a different (wrong) reduction. As the trait doc notes,
    /// such cases can still be composed from
    /// `reduce(Max) + unary(Exp) + reduce(Sum) + binary(Div)`.
    fn softmax(
        &self,
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
        if let Some(pos) = shape.iter().position(|&d| d == 0) {
            return Err(BackendError::InvalidArgument(format!(
                "softmax: shape {shape:?} has a zero-length dimension at index {pos}"
            )));
        }
        if axis != shape.len() - 1 {
            return Err(BackendError::Unsupported(format!(
                "Metal softmax reduces the last axis only (the kernel is row-contiguous); \
                 got axis {axis} of a {}-dimensional shape",
                shape.len()
            )));
        }
        let cols = shape[axis];
        let rows: usize = shape[..axis].iter().product();
        self.dispatch_softmax(input_ptr, output_ptr, rows, cols)
    }
    fn capabilities(&self) -> Capabilities {
        metal_capabilities(self.device.as_deref())
    }
    fn available_devices(&self) -> BackendResult<Vec<DeviceInfo>> {
        let Some(device) = self.device.as_deref() else {
            // Not initialised (always the case off macOS): nothing to report.
            return Ok(Vec::new());
        };
        Ok(vec![DeviceInfo {
            ordinal: 0,
            name: device.name().to_string(),
            // Metal has no CUDA-style compute capability; the GPU family is the
            // closest analogue, so report it as `(family, 0)` — Apple7 → (7, 0),
            // Mac2 → (2, 0) — rather than inventing a version number.
            compute_capability: (device.capabilities().family.generation(), 0),
            // Apple's Metal API exposes no total-VRAM query that is meaningful
            // on unified memory; the largest single allocation is the honest,
            // driver-reported bound.
            total_memory_bytes: device.max_buffer_length(),
            memory_kind: if device.capabilities().unified_memory {
                MemoryKind::Unified
            } else {
                MemoryKind::Device
            },
            capabilities: metal_capabilities(Some(device)),
        }])
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
        self.dispatch_reduce(op, input_ptr, output_ptr, shape, axis)
    }
    fn unary(&self, op: UnaryOp, input_ptr: u64, output_ptr: u64, n: usize) -> BackendResult<()> {
        self.check_init()?;
        if n == 0 {
            return Ok(());
        }
        self.dispatch_unary(op, input_ptr, output_ptr, n)
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
        self.dispatch_binary(op, a_ptr, b_ptr, output_ptr, n)
    }
    /// Strided batched GEMM: `C_b = alpha * op(A_b) * op(B_b) + beta * C_b`.
    ///
    /// Same element type, layout and transpose support as [`Self::gemm`] —
    /// `f32` row-major with runtime leading dimensions — plus per-operand
    /// element strides between consecutive matrices. One kernel launch covers
    /// the whole batch (the batch index is the grid's `z` axis), rather than the
    /// trait's default loop of `batch_count` individual `gemm` calls.
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
        super::types::validate_gemm_layout(trans_a, trans_b, m, n, k, lda, ldb, ldc)?;
        self.dispatch_batched_gemm(
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a_ptr,
            lda,
            stride_a,
            b_ptr,
            ldb,
            stride_b,
            beta,
            c_ptr,
            ldc,
            stride_c,
            batch_count,
        )
    }
    /// Block until every dispatch submitted through this backend has finished.
    ///
    /// In the default synchronous mode each op already waited before returning,
    /// so there is nothing left to await and this succeeds immediately. With
    /// [`MetalBackend::set_async_dispatch`] enabled it awaits every committed
    /// command buffer and reports the first GPU-side failure among them.
    fn synchronize(&self) -> BackendResult<()> {
        self.check_init()?;
        self.drain_inflight()
    }
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
        // Synchronisation point: the allocator's reuse pool may hand this exact
        // buffer to the next `alloc`, so it must not still be referenced by a
        // kernel that has not finished.
        self.drain_inflight()?;
        self.memory()?.free(ptr).map_err(BackendError::from)
    }
    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        self.check_init()?;
        if src.is_empty() {
            return Ok(());
        }
        // Synchronisation point: overwriting a buffer an in-flight kernel still
        // reads would corrupt that kernel's inputs.
        self.drain_inflight()?;
        self.memory()?
            .copy_to_device(dst, src)
            .map_err(BackendError::from)
    }
    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        self.check_init()?;
        if dst.is_empty() {
            return Ok(());
        }
        // Synchronisation point: this is a plain `memcpy` out of unified memory,
        // so without the wait it could read a buffer the GPU is still writing.
        self.drain_inflight()?;
        self.memory()?
            .copy_from_device(dst, src)
            .map_err(BackendError::from)
    }
}

/// Wait for any still-in-flight GPU work before the backend goes away.
///
/// Command buffers retain the resources they reference, so skipping this would
/// not be *unsafe* — but it would let a program exit with kernels still running
/// and their results never observed, and it would swallow a GPU failure that
/// nothing had reported yet. The wait is a no-op in the default synchronous
/// mode.
impl Drop for MetalBackend {
    fn drop(&mut self) {
        if let Err(e) = self.drain_inflight() {
            tracing::warn!("GPU work outstanding at MetalBackend drop failed: {e}");
        }
    }
}
