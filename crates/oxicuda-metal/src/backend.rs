//! [`MetalBackend`] — the main entry point for the oxicuda-metal crate.
//!
//! Implements the [`ComputeBackend`] trait from `oxicuda-backend` using
//! Apple's Metal API for GPU compute on macOS.

use std::sync::Arc;

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};

use crate::{device::MetalDevice, memory::MetalMemoryManager};

// ─── Backend struct ───────────────────────────────────────────────────────────

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
    device: Option<Arc<MetalDevice>>,
    memory: Option<Arc<MetalMemoryManager>>,
    initialized: bool,
}

impl MetalBackend {
    /// Create a new, uninitialised Metal backend.
    pub fn new() -> Self {
        Self {
            device: None,
            memory: None,
            initialized: false,
        }
    }

    /// Return an error if the backend has not been initialised yet.
    fn check_init(&self) -> BackendResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(BackendError::NotInitialized)
        }
    }

    /// Convenience accessor: get the memory manager or return `NotInitialized`.
    fn memory(&self) -> BackendResult<&Arc<MetalMemoryManager>> {
        self.memory.as_ref().ok_or(BackendError::NotInitialized)
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ComputeBackend impl ──────────────────────────────────────────────────────

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
        // Zero-dimension matrices are trivially complete.
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
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

        // Extract NCHW dimensions
        let n = input_shape[0];
        let c_in = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];
        let k_out = filter_shape[0];
        let fh = filter_shape[2];
        let fw = filter_shape[3];
        let oh = output_shape[2];
        let ow = output_shape[3];
        let stride_h = stride[0];
        let stride_w = stride[1];
        let pad_h = padding[0];
        let pad_w = padding[1];

        // CPU fallback: read from device, compute, write back
        let input_len = n * c_in * h_in * w_in;
        let filter_len = k_out * c_in * fh * fw;
        let output_len = n * k_out * oh * ow;

        let mut input_bytes = vec![0u8; input_len * 4];
        let mut filter_bytes = vec![0u8; filter_len * 4];
        self.copy_dtoh(&mut input_bytes, input_ptr)?;
        self.copy_dtoh(&mut filter_bytes, filter_ptr)?;

        let inp = read_f32_le(&input_bytes);
        let flt = read_f32_le(&filter_bytes);
        let mut out = vec![0.0f32; output_len];

        for b in 0..n {
            for kf in 0..k_out {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = 0.0f32;
                        for ci in 0..c_in {
                            for fy in 0..fh {
                                for fx in 0..fw {
                                    let iy = (oy * stride_h + fy) as isize - pad_h as isize;
                                    let ix = (ox * stride_w + fx) as isize - pad_w as isize;
                                    if iy >= 0
                                        && (iy as usize) < h_in
                                        && ix >= 0
                                        && (ix as usize) < w_in
                                    {
                                        let iy = iy as usize;
                                        let ix = ix as usize;
                                        acc += inp[((b * c_in + ci) * h_in + iy) * w_in + ix]
                                            * flt[((kf * c_in + ci) * fh + fy) * fw + fx];
                                    }
                                }
                            }
                        }
                        out[((b * k_out + kf) * oh + oy) * ow + ox] = acc;
                    }
                }
            }
        }

        let out_bytes = write_f32_le(&out);
        self.copy_htod(output_ptr, &out_bytes)?;
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

        // CPU fallback: read Q/K/V, compute stable softmax attention, write O
        let batch_heads = batch * heads;
        let q_len = batch_heads * seq_q * head_dim;
        let kv_len = batch_heads * seq_kv * head_dim;
        let o_len = batch_heads * seq_q * head_dim;

        let mut q_bytes = vec![0u8; q_len * 4];
        let mut k_bytes = vec![0u8; kv_len * 4];
        let mut v_bytes = vec![0u8; kv_len * 4];
        self.copy_dtoh(&mut q_bytes, q_ptr)?;
        self.copy_dtoh(&mut k_bytes, k_ptr)?;
        self.copy_dtoh(&mut v_bytes, v_ptr)?;

        let q = read_f32_le(&q_bytes);
        let k = read_f32_le(&k_bytes);
        let v = read_f32_le(&v_bytes);
        let mut o = vec![0.0f32; o_len];
        let scale_f = scale as f32;

        for bh in 0..batch_heads {
            for sq in 0..seq_q {
                let q_off = (bh * seq_q + sq) * head_dim;

                // Pass 1: find max score for numerical stability
                let mut max_score = f32::NEG_INFINITY;
                for sk in 0..seq_kv {
                    if causal && sk > sq {
                        continue;
                    }
                    let k_off = (bh * seq_kv + sk) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[k_off + d];
                    }
                    let score = dot * scale_f;
                    if score > max_score {
                        max_score = score;
                    }
                }

                // Pass 2: softmax weights + accumulate
                let mut sum_exp = 0.0f32;
                let mut acc = vec![0.0f32; head_dim];
                for sk in 0..seq_kv {
                    if causal && sk > sq {
                        continue;
                    }
                    let k_off = (bh * seq_kv + sk) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[k_off + d];
                    }
                    let w = (dot * scale_f - max_score).exp();
                    sum_exp += w;
                    let v_off = (bh * seq_kv + sk) * head_dim;
                    for d in 0..head_dim {
                        acc[d] += w * v[v_off + d];
                    }
                }

                // Normalize
                let o_off = (bh * seq_q + sq) * head_dim;
                if sum_exp > 0.0 {
                    for d in 0..head_dim {
                        o[o_off + d] = acc[d] / sum_exp;
                    }
                }
            }
        }

        let o_bytes = write_f32_le(&o);
        self.copy_htod(o_ptr, &o_bytes)?;
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

    // ── Synchronisation ───────────────────────────────────────────────────────

    fn synchronize(&self) -> BackendResult<()> {
        self.check_init()?;
        // On macOS the Metal command queue flushes synchronously when we call
        // `wait_until_completed()` on each command buffer.  For a lightweight
        // synchronise (no pending command buffers), this is a no-op.
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

// ─── Helper ───────────────────────────────────────────────────────────────────

/// Round up to the next power of 2 (minimum 1).
#[cfg(target_os = "macos")]
fn next_power_of_2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    1usize << (usize::BITS - (n - 1).leading_zeros())
}

/// Interpret a byte slice as little-endian `f32` values.
fn read_f32_le(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Encode `f32` values as little-endian bytes.
fn write_f32_le(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

// ─── Metal dispatch helpers (macOS only) ──────────────────────────────────────

#[cfg(target_os = "macos")]
impl MetalBackend {
    fn dispatch_unary(
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

        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        let memory = self.memory()?;
        let msl = crate::msl::elementwise_msl(op_str);
        let pipeline = crate::pipeline::MetalComputePipeline::new(device, &msl, "elementwise_f32")
            .map_err(BackendError::from)?;

        let buffers = memory.lock_buffers().map_err(BackendError::from)?;
        let input_info = buffers.get(&input_ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown input handle {input_ptr}"))
        })?;
        let output_info = buffers.get(&output_ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown output handle {output_ptr}"))
        })?;

        let command_buffer = pipeline.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&input_info.buffer), 0);
        encoder.set_buffer(1, Some(&output_info.buffer), 0);

        let count = n as u32;
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const std::ffi::c_void,
        );

        let tg_size = 256u64.min(n as u64);
        let groups = (n as u64).div_ceil(tg_size);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    fn dispatch_binary(
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

        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        let memory = self.memory()?;
        let msl = crate::msl::binary_msl(op_str);
        let pipeline = crate::pipeline::MetalComputePipeline::new(device, &msl, "binary_f32")
            .map_err(BackendError::from)?;

        let buffers = memory.lock_buffers().map_err(BackendError::from)?;
        let a_info = buffers
            .get(&a_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
        let b_info = buffers
            .get(&b_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
        let out_info = buffers.get(&output_ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown output handle {output_ptr}"))
        })?;

        let command_buffer = pipeline.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&a_info.buffer), 0);
        encoder.set_buffer(1, Some(&b_info.buffer), 0);
        encoder.set_buffer(2, Some(&out_info.buffer), 0);

        let count = n as u32;
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const std::ffi::c_void,
        );

        let tg_size = 256u64.min(n as u64);
        let groups = (n as u64).div_ceil(tg_size);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    fn dispatch_reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        let op_str = match op {
            ReduceOp::Sum => "sum",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::Mean => "mean",
        };

        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        let memory = self.memory()?;

        let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
        let reduce_size = shape[axis];
        let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);

        let msl = crate::msl::reduction_msl(op_str);
        if msl.is_empty() {
            return Err(BackendError::Unsupported(format!(
                "Metal reduction op '{op_str}' not supported"
            )));
        }

        let fn_name = crate::msl::reduction_function_name(op_str);
        let pipeline = crate::pipeline::MetalComputePipeline::new(device, &msl, fn_name)
            .map_err(BackendError::from)?;

        let buffers = memory.lock_buffers().map_err(BackendError::from)?;
        let input_info = buffers.get(&input_ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown input handle {input_ptr}"))
        })?;
        let out_info = buffers.get(&output_ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown output handle {output_ptr}"))
        })?;

        let command_buffer = pipeline.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&input_info.buffer), 0);
        encoder.set_buffer(1, Some(&out_info.buffer), 0);

        let outer_u32 = outer_size as u32;
        let reduce_u32 = reduce_size as u32;
        let inner_u32 = inner_size as u32;
        encoder.set_bytes(2, 4, &outer_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(3, 4, &reduce_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &inner_u32 as *const u32 as *const std::ffi::c_void);

        // Use power-of-2 threadgroup size for correct tree reduction.
        let tg_size = next_power_of_2(reduce_size).min(256) as u64;
        encoder.set_threadgroup_memory_length(0, tg_size * std::mem::size_of::<f32>() as u64);

        // 1D dispatch: one threadgroup per (outer, inner) pair.
        let total_groups = (outer_size * inner_size) as u64;
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(total_groups, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_gemm(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        _lda: usize,
        b_ptr: u64,
        _ldb: usize,
        beta: f64,
        c_ptr: u64,
        _ldc: usize,
    ) -> BackendResult<()> {
        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        let memory = self.memory()?;

        let msl = crate::msl::gemm_msl();
        let pipeline = crate::pipeline::MetalComputePipeline::new(device, msl, "gemm_f32")
            .map_err(BackendError::from)?;

        let buffers = memory.lock_buffers().map_err(BackendError::from)?;
        let a_info = buffers
            .get(&a_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
        let b_info = buffers
            .get(&b_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
        let c_info = buffers
            .get(&c_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {c_ptr}")))?;

        #[repr(C)]
        struct GemmParams {
            m: u32,
            n: u32,
            k: u32,
            alpha: f32,
            beta: f32,
        }

        let params = GemmParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            alpha: alpha as f32,
            beta: beta as f32,
        };

        let command_buffer = pipeline.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&a_info.buffer), 0);
        encoder.set_buffer(1, Some(&b_info.buffer), 0);
        encoder.set_buffer(2, Some(&c_info.buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<GemmParams>() as u64,
            &params as *const GemmParams as *const std::ffi::c_void,
        );

        let tg_w = 16u64;
        let tg_h = 16u64;
        let groups_x = (n as u64).div_ceil(tg_w);
        let groups_y = (m as u64).div_ceil(tg_h);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups_x, groups_y, 1),
            metal::MTLSize::new(tg_w, tg_h, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    fn dispatch_unary(
        &self,
        _op: UnaryOp,
        _input_ptr: u64,
        _output_ptr: u64,
        _n: usize,
    ) -> BackendResult<()> {
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }

    fn dispatch_binary(
        &self,
        _op: BinaryOp,
        _a_ptr: u64,
        _b_ptr: u64,
        _output_ptr: u64,
        _n: usize,
    ) -> BackendResult<()> {
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }

    fn dispatch_reduce(
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
    fn dispatch_gemm(
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
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_backend::{BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp};

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn metal_backend_new_uninitialized() {
        let b = MetalBackend::new();
        assert!(!b.is_initialized());
    }

    #[test]
    fn metal_backend_name() {
        let b = MetalBackend::new();
        assert_eq!(b.name(), "metal");
    }

    #[test]
    fn metal_backend_default() {
        let b = MetalBackend::default();
        assert!(!b.is_initialized());
        assert_eq!(b.name(), "metal");
    }

    #[test]
    fn backend_debug_impl() {
        let b = MetalBackend::new();
        let s = format!("{b:?}");
        assert!(s.contains("MetalBackend"));
    }

    // ── Object-safety smoke test ──────────────────────────────────────────────

    #[test]
    fn backend_object_safe() {
        let b: Box<dyn ComputeBackend> = Box::new(MetalBackend::new());
        assert_eq!(b.name(), "metal");
    }

    // ── Not-initialized guards ────────────────────────────────────────────────

    #[test]
    fn backend_not_initialized_gemm() {
        let b = MetalBackend::new();
        let result = b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            0,
            4,
            0.0,
            0,
            4,
        );
        assert_eq!(result, Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_alloc() {
        let b = MetalBackend::new();
        assert_eq!(b.alloc(1024), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_synchronize() {
        let b = MetalBackend::new();
        assert_eq!(b.synchronize(), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_free() {
        let b = MetalBackend::new();
        assert_eq!(b.free(1), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_copy_htod() {
        let b = MetalBackend::new();
        assert_eq!(b.copy_htod(1, b"hello"), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_copy_dtoh() {
        let b = MetalBackend::new();
        let mut buf = [0u8; 4];
        assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
    }

    // ── Helper: try to get an initialised backend (skip if no GPU) ────────────

    fn try_init() -> Option<MetalBackend> {
        let mut b = MetalBackend::new();
        match b.init() {
            Ok(()) => Some(b),
            Err(_) => None,
        }
    }

    // ── macOS-specific init test ──────────────────────────────────────────────

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_backend_init_on_macos() {
        let mut backend = MetalBackend::new();
        match backend.init() {
            Ok(()) => {
                assert!(backend.is_initialized());
                // Calling init() again must be a no-op.
                assert_eq!(backend.init(), Ok(()));
                assert!(backend.is_initialized());
                // Can alloc a small buffer.
                let result = backend.alloc(64);
                match result {
                    Ok(handle) => {
                        assert!(handle > 0);
                        backend.free(handle).expect("free should succeed");
                    }
                    Err(e) => {
                        // In unusual CI environments this may fail — just not a panic.
                        let _ = e;
                    }
                }
            }
            Err(e) => {
                // May fail in CI without GPU — must not panic.
                let _ = e;
            }
        }
    }

    // ── Zero-size / trivial-OK paths (post-init) ──────────────────────────────

    #[test]
    fn alloc_zero_bytes_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.alloc(0),
            Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into()
            ))
        );
    }

    #[test]
    fn copy_htod_empty_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.copy_htod(0, &[]), Ok(()));
    }

    #[test]
    fn copy_dtoh_empty_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.copy_dtoh(&mut [], 0), Ok(()));
    }

    #[test]
    fn gemm_zero_dims_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                0,
                0,
                1.0,
                0,
                1,
                0,
                1,
                0.0,
                0,
                1
            ),
            Ok(())
        );
    }

    #[test]
    fn unary_zero_n_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.unary(UnaryOp::Relu, 0, 0, 0), Ok(()));
    }

    #[test]
    fn binary_zero_n_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.binary(BinaryOp::Add, 0, 0, 0, 0), Ok(()));
    }

    #[test]
    fn synchronize_after_init() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.synchronize(), Ok(()));
    }

    // ── Argument validation (post-init) ───────────────────────────────────────

    #[test]
    fn reduce_empty_shape_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.reduce(ReduceOp::Sum, 0, 0, &[], 0),
            Err(BackendError::InvalidArgument(
                "shape must not be empty".into()
            ))
        );
    }

    #[test]
    fn reduce_axis_out_of_bounds_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.reduce(ReduceOp::Sum, 0, 0, &[4, 4], 5),
            Err(BackendError::InvalidArgument(
                "axis 5 is out of bounds for shape of length 2".into()
            ))
        );
    }

    #[test]
    fn attention_zero_seq_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 0, 8, 64, 0.125, false),
            Err(BackendError::InvalidArgument(
                "seq_q, seq_kv, and head_dim must all be > 0".into()
            ))
        );
    }

    #[test]
    fn attention_invalid_scale_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, 0.0, false),
            Err(BackendError::InvalidArgument(
                "scale must be a positive finite number, got 0".into()
            ))
        );
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, -1.0, false),
            Err(BackendError::InvalidArgument(
                "scale must be a positive finite number, got -1".into()
            ))
        );
        assert!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, f64::INFINITY, false)
                .is_err()
        );
    }

    #[test]
    fn conv2d_wrong_input_rank() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.conv2d_forward(
                0,
                &[1, 3, 32],
                0,
                &[16, 3, 3, 3],
                0,
                &[1, 16, 30, 30],
                &[1, 1],
                &[0, 0]
            ),
            Err(BackendError::InvalidArgument(
                "input_shape must have 4 elements (NCHW)".into()
            ))
        );
    }

    #[test]
    fn conv2d_wrong_filter_rank() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.conv2d_forward(
                0,
                &[1, 3, 32, 32],
                0,
                &[16, 3, 3],
                0,
                &[1, 16, 30, 30],
                &[1, 1],
                &[0, 0]
            ),
            Err(BackendError::InvalidArgument(
                "filter_shape must have 4 elements (KCFHFW)".into()
            ))
        );
    }

    // ── Init is idempotent ────────────────────────────────────────────────────

    #[test]
    fn init_idempotent() {
        let Some(mut b) = try_init() else {
            return;
        };
        assert_eq!(b.init(), Ok(()));
        assert!(b.is_initialized());
    }

    // ── Graceful failure ──────────────────────────────────────────────────────

    #[test]
    fn metal_init_graceful_failure() {
        // Verify that init() returns a Result and never panics.
        let mut b = MetalBackend::new();
        let _result = b.init();
        // Ok or Err — both are acceptable.
    }

    // ── alloc/free/copy roundtrip (macOS with Metal) ──────────────────────────

    #[test]
    fn alloc_copy_roundtrip() {
        let Some(b) = try_init() else {
            return;
        };
        let src: Vec<u8> = (0u8..64).collect();
        let handle = match b.alloc(src.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(handle, &src).expect("copy_htod");
        let mut dst = vec![0u8; src.len()];
        b.copy_dtoh(&mut dst, handle).expect("copy_dtoh");
        assert_eq!(src, dst);
        b.free(handle).expect("free");
    }

    // ── GPU compute verification (macOS only) ─────────────────────────────────

    /// Helper: encode f32 slice to bytes (little-endian).
    #[cfg(target_os = "macos")]
    fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
        let mut bytes = vec![0u8; std::mem::size_of_val(data)];
        for (i, &val) in data.iter().enumerate() {
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Helper: decode bytes to f32 vec (little-endian).
    #[cfg(target_os = "macos")]
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn unary_relu_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![-1.0f32, 0.0, 1.0, 2.0];
        let n = input.len();
        let bytes_in = f32_to_bytes(&input);

        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(ih, &bytes_in).expect("htod");
        b.unary(UnaryOp::Relu, ih, oh, n).expect("unary relu");

        let mut out = vec![0u8; bytes_in.len()];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert_eq!(result, vec![0.0f32, 0.0, 1.0, 2.0]);

        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn unary_neg_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![1.0f32, -2.0, 3.0, 0.0];
        let n = input.len();
        let bytes_in = f32_to_bytes(&input);

        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(ih, &bytes_in).expect("htod");
        b.unary(UnaryOp::Neg, ih, oh, n).expect("unary neg");

        let mut out = vec![0u8; bytes_in.len()];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert_eq!(result, vec![-1.0f32, 2.0, -3.0, -0.0]);

        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn binary_add_compute() {
        let Some(b) = try_init() else { return };
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let bv = vec![10.0f32, 20.0, 30.0, 40.0];
        let n = a.len();
        let ba = f32_to_bytes(&a);
        let bb = f32_to_bytes(&bv);

        let ah = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let bh = match b.alloc(bb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(ah, &ba).expect("htod a");
        b.copy_htod(bh, &bb).expect("htod b");
        b.binary(BinaryOp::Add, ah, bh, oh, n).expect("binary add");

        let mut out = vec![0u8; ba.len()];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert_eq!(result, vec![11.0f32, 22.0, 33.0, 44.0]);

        b.free(ah).expect("free");
        b.free(bh).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn binary_mul_compute() {
        let Some(b) = try_init() else { return };
        let a = vec![2.0f32, 3.0, 4.0, 5.0];
        let bv = vec![10.0f32, 10.0, 10.0, 10.0];
        let n = a.len();
        let ba = f32_to_bytes(&a);
        let bb = f32_to_bytes(&bv);

        let ah = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let bh = match b.alloc(bb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(ah, &ba).expect("htod a");
        b.copy_htod(bh, &bb).expect("htod b");
        b.binary(BinaryOp::Mul, ah, bh, oh, n).expect("binary mul");

        let mut out = vec![0u8; ba.len()];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert_eq!(result, vec![20.0f32, 30.0, 40.0, 50.0]);

        b.free(ah).expect("free");
        b.free(bh).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_sum_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes_in = f32_to_bytes(&input);

        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        // Output: single f32 (reduce shape [4] along axis 0).
        let oh = match b.alloc(4) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 4]).expect("zero output");
        b.reduce(ReduceOp::Sum, ih, oh, &[4], 0)
            .expect("reduce sum");

        let mut out = vec![0u8; 4];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 10.0).abs() < 1e-5,
            "expected 10.0, got {}",
            result[0]
        );

        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_max_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![3.0f32, 1.0, 4.0, 1.5];
        let bytes_in = f32_to_bytes(&input);

        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(4) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 4]).expect("zero output");
        b.reduce(ReduceOp::Max, ih, oh, &[4], 0)
            .expect("reduce max");

        let mut out = vec![0u8; 4];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 4.0).abs() < 1e-5,
            "expected 4.0, got {}",
            result[0]
        );

        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_mean_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![2.0f32, 4.0, 6.0, 8.0];
        let bytes_in = f32_to_bytes(&input);

        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(4) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 4]).expect("zero output");
        b.reduce(ReduceOp::Mean, ih, oh, &[4], 0)
            .expect("reduce mean");

        let mut out = vec![0u8; 4];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 5.0).abs() < 1e-5,
            "expected 5.0, got {}",
            result[0]
        );

        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_2d_axis1_compute() {
        let Some(b) = try_init() else { return };
        // shape [2, 3], reduce axis 1 → shape [2]
        // [[1, 2, 3], [4, 5, 6]] → [6, 15]
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes_in = f32_to_bytes(&input);

        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(8) {
            Ok(h) => h,
            Err(_) => return,
        }; // 2 floats

        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 8]).expect("zero output");
        b.reduce(ReduceOp::Sum, ih, oh, &[2, 3], 1)
            .expect("reduce sum axis=1");

        let mut out = vec![0u8; 8];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 6.0).abs() < 1e-5,
            "expected 6.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 15.0).abs() < 1e-5,
            "expected 15.0, got {}",
            result[1]
        );

        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn gemm_simple_compute() {
        let Some(b) = try_init() else { return };
        // A = [[1, 2], [3, 4]] (2×2, row-major)
        // B = [[5, 6], [7, 8]] (2×2, row-major)
        // C = alpha*A*B + beta*C = [[19, 22], [43, 50]]  (alpha=1, beta=0)
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let bm = vec![5.0f32, 6.0, 7.0, 8.0];
        let c_init = vec![0.0f32; 4];
        let ba = f32_to_bytes(&a);
        let bb = f32_to_bytes(&bm);
        let bc = f32_to_bytes(&c_init);

        let ah = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let bh = match b.alloc(bb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let ch = match b.alloc(bc.len()) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(ah, &ba).expect("htod a");
        b.copy_htod(bh, &bb).expect("htod b");
        b.copy_htod(ch, &bc).expect("htod c");

        b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            ah,
            2,
            bh,
            2,
            0.0,
            ch,
            2,
        )
        .expect("gemm");

        let mut out = vec![0u8; bc.len()];
        b.copy_dtoh(&mut out, ch).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 19.0).abs() < 1e-4,
            "C[0,0]={}, expected 19",
            result[0]
        );
        assert!(
            (result[1] - 22.0).abs() < 1e-4,
            "C[0,1]={}, expected 22",
            result[1]
        );
        assert!(
            (result[2] - 43.0).abs() < 1e-4,
            "C[1,0]={}, expected 43",
            result[2]
        );
        assert!(
            (result[3] - 50.0).abs() < 1e-4,
            "C[1,1]={}, expected 50",
            result[3]
        );

        b.free(ah).expect("free");
        b.free(bh).expect("free");
        b.free(ch).expect("free");
    }

    // ── Conv2D tests ──────────────────────────────────────────────────────────

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_conv2d_identity_1x1() {
        let Some(b) = try_init() else { return };
        // 1×1 conv, stride=1, pad=0, 1 channel → output = input * filter_val
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let filter = vec![2.0f32];
        let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();

        let ib = f32_to_bytes(&input);
        let fb = f32_to_bytes(&filter);
        let out_size = expected.len() * 4;

        let ih = match b.alloc(ib.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let fh = match b.alloc(fb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(ih, &ib).expect("htod input");
        b.copy_htod(fh, &fb).expect("htod filter");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero output");

        b.conv2d_forward(
            ih,
            &[1, 1, 3, 3],
            fh,
            &[1, 1, 1, 1],
            oh,
            &[1, 1, 3, 3],
            &[1, 1],
            &[0, 0],
        )
        .expect("conv2d 1x1");

        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "1x1 mismatch at {i}: {r} vs {e}");
        }

        b.free(ih).expect("free");
        b.free(fh).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_conv2d_3x3_basic() {
        let Some(b) = try_init() else { return };
        // 1×1×4×4 input, 1×1×3×3 all-ones filter, stride=1, pad=0 → 1×1×2×2
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let filter = vec![1.0f32; 9];
        // Expected: [54, 63, 90, 99]
        let expected = [54.0f32, 63.0, 90.0, 99.0];

        let ib = f32_to_bytes(&input);
        let fb = f32_to_bytes(&filter);
        let out_size = expected.len() * 4;

        let ih = match b.alloc(ib.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let fh = match b.alloc(fb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(ih, &ib).expect("htod");
        b.copy_htod(fh, &fb).expect("htod");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");

        b.conv2d_forward(
            ih,
            &[1, 1, 4, 4],
            fh,
            &[1, 1, 3, 3],
            oh,
            &[1, 1, 2, 2],
            &[1, 1],
            &[0, 0],
        )
        .expect("conv2d 3x3");

        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "3x3 mismatch at {i}: {r} vs {e}");
        }

        b.free(ih).expect("free");
        b.free(fh).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_conv2d_with_padding() {
        let Some(b) = try_init() else { return };
        // 1×1×3×3 input, 1×1×3×3 all-ones filter, stride=1, pad=1 → 1×1×3×3
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let filter = vec![1.0f32; 9];
        let expected = vec![12.0, 21.0, 16.0, 27.0, 45.0, 33.0, 24.0, 39.0, 28.0];

        let ib = f32_to_bytes(&input);
        let fb = f32_to_bytes(&filter);
        let out_size = expected.len() * 4;

        let ih = match b.alloc(ib.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let fh = match b.alloc(fb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(ih, &ib).expect("htod");
        b.copy_htod(fh, &fb).expect("htod");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");

        b.conv2d_forward(
            ih,
            &[1, 1, 3, 3],
            fh,
            &[1, 1, 3, 3],
            oh,
            &[1, 1, 3, 3],
            &[1, 1],
            &[1, 1],
        )
        .expect("conv2d padded");

        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "pad mismatch at {i}: {r} vs {e}");
        }

        b.free(ih).expect("free");
        b.free(fh).expect("free");
        b.free(oh).expect("free");
    }

    // ── Attention tests ───────────────────────────────────────────────────────

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_attention_uniform() {
        let Some(b) = try_init() else { return };
        // batch=1, heads=1, seq_q=2, seq_kv=3, head_dim=2
        // Q = all 1s, K = all 1s → equal scores → output ≈ mean(V)
        let q = vec![1.0f32; 2 * 2]; // [1,1, 1,1] for 2 query positions
        let k = vec![1.0f32; 3 * 2]; // [1,1, 1,1, 1,1] for 3 kv positions
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // V[0]=[1,2], V[1]=[3,4], V[2]=[5,6]
        // mean(V) = [3, 4]

        let qb = f32_to_bytes(&q);
        let kb = f32_to_bytes(&k);
        let vb = f32_to_bytes(&v);
        let out_size = q.len() * 4;

        let qh = match b.alloc(qb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let kh = match b.alloc(kb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let vh = match b.alloc(vb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(qh, &qb).expect("htod q");
        b.copy_htod(kh, &kb).expect("htod k");
        b.copy_htod(vh, &vb).expect("htod v");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");

        let scale = 1.0 / (2.0f64).sqrt();
        b.attention(qh, kh, vh, oh, 1, 1, 2, 3, 2, scale, false)
            .expect("attention uniform");

        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        // Both query positions should produce ≈ [3, 4]
        for sq in 0..2 {
            let base = sq * 2;
            assert!(
                (result[base] - 3.0).abs() < 0.1,
                "sq={sq} d=0: {} vs 3.0",
                result[base]
            );
            assert!(
                (result[base + 1] - 4.0).abs() < 0.1,
                "sq={sq} d=1: {} vs 4.0",
                result[base + 1]
            );
        }

        b.free(qh).expect("free");
        b.free(kh).expect("free");
        b.free(vh).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_attention_causal() {
        let Some(b) = try_init() else { return };
        // batch=1, heads=1, seq_q=3, seq_kv=3, head_dim=2, causal=true
        // Q=K=all 1s → equal within mask
        // V = [[10,20], [30,40], [50,60]]
        // sq=0: only sk=0 → output = V[0] = [10, 20]
        // sq=1: sk=0,1 → output = mean(V[0], V[1]) = [20, 30]
        // sq=2: sk=0,1,2 → output = mean(V[0..3]) = [30, 40]
        let q = vec![1.0f32; 3 * 2];
        let k = vec![1.0f32; 3 * 2];
        let v = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let expected = [10.0f32, 20.0, 20.0, 30.0, 30.0, 40.0];

        let qb = f32_to_bytes(&q);
        let kb = f32_to_bytes(&k);
        let vb = f32_to_bytes(&v);
        let out_size = expected.len() * 4;

        let qh = match b.alloc(qb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let kh = match b.alloc(kb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let vh = match b.alloc(vb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(qh, &qb).expect("htod");
        b.copy_htod(kh, &kb).expect("htod");
        b.copy_htod(vh, &vb).expect("htod");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");

        let scale = 1.0 / (2.0f64).sqrt();
        b.attention(qh, kh, vh, oh, 1, 1, 3, 3, 2, scale, true)
            .expect("attention causal");

        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 0.5, "causal idx {i}: {r} vs {e}");
        }

        b.free(qh).expect("free");
        b.free(kh).expect("free");
        b.free(vh).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_attention_dominant_key() {
        let Some(b) = try_init() else { return };
        // One key much larger → output ≈ that key's value
        // Q=[1,0], K=[[0,0],[0,0],[10,0]], scale=1.0
        // dots = [0, 0, 10] → softmax heavily weights k=2
        // V = [[1,2],[3,4],[5,6]] → output ≈ [5, 6]
        let q = vec![1.0f32, 0.0];
        let k = vec![0.0f32, 0.0, 0.0, 0.0, 10.0, 0.0];
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let qb = f32_to_bytes(&q);
        let kb = f32_to_bytes(&k);
        let vb = f32_to_bytes(&v);
        let out_size = q.len() * 4;

        let qh = match b.alloc(qb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let kh = match b.alloc(kb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let vh = match b.alloc(vb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };

        b.copy_htod(qh, &qb).expect("htod");
        b.copy_htod(kh, &kb).expect("htod");
        b.copy_htod(vh, &vb).expect("htod");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");

        b.attention(qh, kh, vh, oh, 1, 1, 1, 3, 2, 1.0, false)
            .expect("attention dominant");

        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!((result[0] - 5.0).abs() < 0.01, "d=0: {} vs 5.0", result[0]);
        assert!((result[1] - 6.0).abs() < 0.01, "d=1: {} vs 6.0", result[1]);

        b.free(qh).expect("free");
        b.free(kh).expect("free");
        b.free(vh).expect("free");
        b.free(oh).expect("free");
    }
}
