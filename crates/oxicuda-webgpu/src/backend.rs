//! [`WebGpuBackend`] — the main entry point for the oxicuda-webgpu crate.
//!
//! Implements the [`ComputeBackend`] trait from `oxicuda-backend` using
//! `wgpu` for cross-platform GPU compute (Vulkan, Metal, DX12, WebGPU).

use std::sync::Arc;

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};
use wgpu;

use crate::{device::WebGpuDevice, memory::WebGpuMemoryManager, shader};

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
}

impl WebGpuBackend {
    /// Create a new, uninitialised WebGPU backend.
    pub fn new() -> Self {
        Self {
            device: None,
            memory: None,
            initialized: false,
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
}

impl WebGpuBackend {
    /// FP16 GEMM: `C = alpha * A * B + beta * C` with half-precision storage.
    ///
    /// This is an inherent method (not on `ComputeBackend`) because FP16
    /// support is WebGPU-specific and requires the `f16` WGSL extension.
    ///
    /// Buffers pointed to by `a_ptr`, `b_ptr`, `c_ptr` must contain `f16`
    /// elements (2 bytes each).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        b_ptr: u64,
        beta: f64,
        c_ptr: u64,
    ) -> BackendResult<()> {
        self.check_init()?;
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        let dev = self.device()?;
        let mem = self.memory()?;

        let tile_size: u32 = 8;
        let wgsl = shader::gemm_wgsl_f16(tile_size);

        let shader_mod = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-gemm-f16"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-gemm-f16"),
                layout: None,
                module: &shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bgl = pipeline.get_bind_group_layout(0);

        // Build uniform buffer for GemmParams { m, n, k, alpha, beta }.
        let mut params_bytes = [0u8; 20];
        params_bytes[0..4].copy_from_slice(&(m as u32).to_le_bytes());
        params_bytes[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        params_bytes[8..12].copy_from_slice(&(k as u32).to_le_bytes());
        params_bytes[12..16].copy_from_slice(&(alpha as f32).to_le_bytes());
        params_bytes[16..20].copy_from_slice(&(beta as f32).to_le_bytes());

        let uniform_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-gemm-f16-params"),
            size: 20,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&uniform_buf, 0, &params_bytes);

        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let a_info = buffers
                .get(&a_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
            let b_info = buffers
                .get(&b_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
            let c_info = buffers
                .get(&c_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {c_ptr}")))?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-gemm-f16"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            })
        };

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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (n as u32).div_ceil(tile_size);
            let wg_y = (m as u32).div_ceil(tile_size);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

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
        _lda: usize,
        b_ptr: u64,
        _ldb: usize,
        beta: f64,
        c_ptr: u64,
        _ldc: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        // Zero-dimension matrices are trivially done.
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        // Transpose not yet supported in the WGSL shader.
        if trans_a != BackendTranspose::NoTrans || trans_b != BackendTranspose::NoTrans {
            return Err(BackendError::Unsupported(
                "WebGPU GEMM does not yet support transposed inputs".into(),
            ));
        }

        let dev = self.device()?;
        let mem = self.memory()?;

        let tile_size: u32 = 8;
        let wgsl = shader::gemm_wgsl(tile_size);

        let shader_mod = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-gemm"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-gemm"),
                layout: None,
                module: &shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bgl = pipeline.get_bind_group_layout(0);

        // Build uniform buffer for GemmParams { m, n, k, alpha, beta }.
        let mut params_bytes = [0u8; 20];
        params_bytes[0..4].copy_from_slice(&(m as u32).to_le_bytes());
        params_bytes[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        params_bytes[8..12].copy_from_slice(&(k as u32).to_le_bytes());
        params_bytes[12..16].copy_from_slice(&(alpha as f32).to_le_bytes());
        params_bytes[16..20].copy_from_slice(&(beta as f32).to_le_bytes());

        let uniform_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-gemm-params"),
            size: 20,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&uniform_buf, 0, &params_bytes);

        // Create bind group while holding the buffer lock.
        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let a_info = buffers
                .get(&a_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
            let b_info = buffers
                .get(&b_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
            let c_info = buffers
                .get(&c_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {c_ptr}")))?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-gemm"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            })
        };

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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (n as u32).div_ceil(tile_size);
            let wg_y = (m as u32).div_ceil(tile_size);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

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
        _lda: usize,
        stride_a: usize,
        b_ptr: u64,
        _ldb: usize,
        stride_b: usize,
        beta: f64,
        c_ptr: u64,
        _ldc: usize,
        stride_c: usize,
        batch_count: usize,
    ) -> BackendResult<()> {
        self.check_init()?;

        if batch_count == 0 || m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        if trans_a != BackendTranspose::NoTrans || trans_b != BackendTranspose::NoTrans {
            return Err(BackendError::Unsupported(
                "WebGPU batched GEMM does not yet support transposed inputs".into(),
            ));
        }

        let dev = self.device()?;
        let mem = self.memory()?;

        let tile_size: u32 = 8;
        let wgsl = shader::batched_gemm_wgsl(tile_size);

        let shader_mod = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-batched-gemm"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-batched-gemm"),
                layout: None,
                module: &shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bgl = pipeline.get_bind_group_layout(0);

        // BatchedGemmParams: m, n, k, alpha, beta, batch_count, stride_a, stride_b, stride_c
        // 9 fields: 5 x u32/f32 + 4 x u32 = 36 bytes total
        // But we need 16-byte alignment for uniform buffers. 36 rounds up to 48.
        // Actually: 3 u32 + 2 f32 + 1 u32 + 3 u32 = 9 x 4 = 36 bytes.
        // Pad to 48 for safety (16-byte aligned).
        let mut params_bytes = [0u8; 48];
        params_bytes[0..4].copy_from_slice(&(m as u32).to_le_bytes());
        params_bytes[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        params_bytes[8..12].copy_from_slice(&(k as u32).to_le_bytes());
        params_bytes[12..16].copy_from_slice(&(alpha as f32).to_le_bytes());
        params_bytes[16..20].copy_from_slice(&(beta as f32).to_le_bytes());
        params_bytes[20..24].copy_from_slice(&(batch_count as u32).to_le_bytes());
        params_bytes[24..28].copy_from_slice(&(stride_a as u32).to_le_bytes());
        params_bytes[28..32].copy_from_slice(&(stride_b as u32).to_le_bytes());
        params_bytes[32..36].copy_from_slice(&(stride_c as u32).to_le_bytes());
        // bytes 36..48 are padding zeros

        let uniform_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-batched-gemm-params"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&uniform_buf, 0, &params_bytes);

        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let a_info = buffers
                .get(&a_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
            let b_info = buffers
                .get(&b_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
            let c_info = buffers
                .get(&c_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {c_ptr}")))?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-batched-gemm"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            })
        };

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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (n as u32).div_ceil(tile_size);
            let wg_y = (m as u32).div_ceil(tile_size);
            pass.dispatch_workgroups(wg_x, wg_y, batch_count as u32);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

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

        let mem = self.memory()?;

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

        // CPU fallback: download input + filter, compute, upload output.
        let mut in_bytes = vec![0u8; in_elems * 4];
        let mut f_bytes = vec![0u8; f_elems * 4];
        mem.copy_from_device(&mut in_bytes, input_ptr)
            .map_err(BackendError::from)?;
        mem.copy_from_device(&mut f_bytes, filter_ptr)
            .map_err(BackendError::from)?;

        let in_f32 = bytes_to_f32_vec(&in_bytes);
        let f_f32 = bytes_to_f32_vec(&f_bytes);
        let mut out_f32 = vec![0.0f32; o_elems];

        for b in 0..batch {
            for kf in 0..k_out {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = 0.0f32;
                        for ci in 0..c_in {
                            for fy in 0..fh {
                                for fx in 0..fw {
                                    let iy = (oy * sh + fy) as isize - ph as isize;
                                    let ix = (ox * sw + fx) as isize - pw as isize;
                                    if iy >= 0
                                        && (iy as usize) < h_in
                                        && ix >= 0
                                        && (ix as usize) < w_in
                                    {
                                        let in_idx = ((b * c_in + ci) * h_in + iy as usize) * w_in
                                            + ix as usize;
                                        let f_idx = ((kf * c_in + ci) * fh + fy) * fw + fx;
                                        acc += in_f32[in_idx] * f_f32[f_idx];
                                    }
                                }
                            }
                        }
                        out_f32[((b * k_out + kf) * oh + oy) * ow + ox] = acc;
                    }
                }
            }
        }

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

        let mem = self.memory()?;

        let batch_heads = batch * heads;
        let q_elems = batch_heads * seq_q * head_dim;
        let kv_elems = batch_heads * seq_kv * head_dim;
        let o_elems = q_elems;

        // CPU fallback: download Q, K, V, compute attention, upload O.
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
        let mut o_f32 = vec![0.0f32; o_elems];

        let scale_f32 = scale as f32;

        for bh in 0..batch_heads {
            let q_off = bh * seq_q * head_dim;
            let k_off = bh * seq_kv * head_dim;
            let v_off = k_off;

            for sq in 0..seq_q {
                let kv_limit = if causal { (sq + 1).min(seq_kv) } else { seq_kv };

                // Pass 1: find max score for numerical stability
                let mut max_score = f32::NEG_INFINITY;
                for sk in 0..kv_limit {
                    let mut dot = 0.0f32;
                    for dd in 0..head_dim {
                        dot +=
                            q_f32[q_off + sq * head_dim + dd] * k_f32[k_off + sk * head_dim + dd];
                    }
                    let s = dot * scale_f32;
                    if s > max_score {
                        max_score = s;
                    }
                }

                // Pass 2: exp(score - max), accumulate weighted V
                let mut sum_exp = 0.0f32;
                let mut acc = vec![0.0f32; head_dim];
                for sk in 0..kv_limit {
                    let mut dot = 0.0f32;
                    for dd in 0..head_dim {
                        dot +=
                            q_f32[q_off + sq * head_dim + dd] * k_f32[k_off + sk * head_dim + dd];
                    }
                    let w = (dot * scale_f32 - max_score).exp();
                    sum_exp += w;
                    for dd in 0..head_dim {
                        acc[dd] += w * v_f32[v_off + sk * head_dim + dd];
                    }
                }

                // Normalise
                let o_base = q_off + sq * head_dim;
                if sum_exp > 0.0 {
                    for dd in 0..head_dim {
                        o_f32[o_base + dd] = acc[dd] / sum_exp;
                    }
                }
            }
        }

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

        // Only flat 1-D reduction (shape.len() == 1, axis == 0) is currently
        // supported on the GPU.  Multi-dimensional reductions require batched
        // shaders that are not yet implemented.
        if shape.len() != 1 {
            return Err(BackendError::Unsupported(
                "WebGPU reduce currently supports only 1-D shapes".into(),
            ));
        }

        let n_elements = shape[0];
        if n_elements == 0 {
            return Ok(());
        }

        let dev = self.device()?;
        let mem = self.memory()?;
        let op_str = map_reduce_op(op);

        // ── Pass 1: per-workgroup reduction ─────────────────────────────────
        let wg_count = (n_elements as u32).div_ceil(256);

        let pass1_wgsl = shader::reduction_wgsl(op_str);
        let pass1_shader = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-reduce-pass1"),
                source: wgpu::ShaderSource::Wgsl(pass1_wgsl.into()),
            });
        let pass1_pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-reduce-pass1"),
                layout: None,
                module: &pass1_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

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

        let bgl1 = pass1_pipeline.get_bind_group_layout(0);

        let bg1 = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let in_info = buffers.get(&input_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {input_ptr}"))
            })?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-reduce-pass1"),
                layout: &bgl1,
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
        let pass2_wgsl = shader::reduction_final_wgsl(op_str);
        let pass2_shader = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-reduce-pass2"),
                source: wgpu::ShaderSource::Wgsl(pass2_wgsl.into()),
            });
        let pass2_pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-reduce-pass2"),
                layout: None,
                module: &pass2_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

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

        let bgl2 = pass2_pipeline.get_bind_group_layout(0);

        let bg2 = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let out_info = buffers.get(&output_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {output_ptr}"))
            })?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-reduce-pass2"),
                layout: &bgl2,
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
            pass.set_pipeline(&pass1_pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-reduce-pass2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pass2_pipeline);
            pass.set_bind_group(0, &bg2, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

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

        let dev = self.device()?;
        let mem = self.memory()?;

        let op_str = map_unary_op(op);
        let wgsl = shader::elementwise_wgsl(op_str);

        let shader_mod = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-unary"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-unary"),
                layout: None,
                module: &shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bgl = pipeline.get_bind_group_layout(0);

        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let in_info = buffers.get(&input_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {input_ptr}"))
            })?;
            let out_info = buffers.get(&output_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {output_ptr}"))
            })?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-unary"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_info.buffer.as_entire_binding(),
                    },
                ],
            })
        };

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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32).div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

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

        let dev = self.device()?;
        let mem = self.memory()?;

        let op_str = map_binary_op(op);
        let wgsl = shader::binary_wgsl(op_str);

        let shader_mod = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-binary"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-binary"),
                layout: None,
                module: &shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bgl = pipeline.get_bind_group_layout(0);

        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let a_info = buffers
                .get(&a_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
            let b_info = buffers
                .get(&b_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
            let out_info = buffers.get(&output_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {output_ptr}"))
            })?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-binary"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_info.buffer.as_entire_binding(),
                    },
                ],
            })
        };

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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32).div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

        Ok(())
    }

    // ── Synchronisation ───────────────────────────────────────────────────────

    fn synchronize(&self) -> BackendResult<()> {
        self.check_init()?;
        if let Some(dev) = &self.device {
            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
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

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_backend::{BackendTranspose, BinaryOp, ReduceOp, UnaryOp};

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn webgpu_backend_new_uninitialized() {
        let b = WebGpuBackend::new();
        assert!(!b.is_initialized());
    }

    #[test]
    fn webgpu_backend_name() {
        let b = WebGpuBackend::new();
        assert_eq!(b.name(), "webgpu");
    }

    #[test]
    fn webgpu_backend_default() {
        let b = WebGpuBackend::default();
        assert!(!b.is_initialized());
        assert_eq!(b.name(), "webgpu");
    }

    #[test]
    fn backend_debug_impl() {
        let b = WebGpuBackend::new();
        let s = format!("{b:?}");
        assert!(s.contains("WebGpuBackend"));
    }

    // ── Object-safety smoke test ──────────────────────────────────────────────

    #[test]
    fn backend_object_safe() {
        let b: Box<dyn ComputeBackend> = Box::new(WebGpuBackend::new());
        assert_eq!(b.name(), "webgpu");
    }

    // ── Not-initialized guards ────────────────────────────────────────────────

    #[test]
    fn backend_not_initialized_gemm() {
        let b = WebGpuBackend::new();
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
        let b = WebGpuBackend::new();
        let result = b.alloc(1024);
        assert_eq!(result, Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_synchronize() {
        let b = WebGpuBackend::new();
        assert_eq!(b.synchronize(), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_free() {
        let b = WebGpuBackend::new();
        assert_eq!(b.free(1), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_copy_htod() {
        let b = WebGpuBackend::new();
        assert_eq!(b.copy_htod(1, b"hello"), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_copy_dtoh() {
        let b = WebGpuBackend::new();
        let mut buf = [0u8; 4];
        assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
    }

    // ── Zero-size / trivial-OK paths (no GPU needed) ─────────────────────────

    /// These tests exercise the "no-op for zero size" branches.  We need the
    /// backend to be initialised, but if no GPU is available we skip.
    fn try_init() -> Option<WebGpuBackend> {
        let mut b = WebGpuBackend::new();
        match b.init() {
            Ok(()) => Some(b),
            Err(_) => None,
        }
    }

    #[test]
    fn gemm_zero_size_after_init() {
        let Some(b) = try_init() else {
            return;
        };
        let result = b.gemm(
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
            1,
        );
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn unary_zero_elements_after_init() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.unary(UnaryOp::Relu, 0, 0, 0), Ok(()));
    }

    #[test]
    fn binary_zero_elements_after_init() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.binary(BinaryOp::Add, 0, 0, 0, 0), Ok(()));
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
    fn attention_nonpositive_scale_error() {
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
    fn conv2d_wrong_input_shape_error() {
        let Some(b) = try_init() else {
            return;
        };
        // 3-element input_shape — should fail.
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
    fn conv2d_wrong_filter_shape_error() {
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

    #[test]
    fn conv2d_wrong_stride_shape_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.conv2d_forward(
                0,
                &[1, 3, 32, 32],
                0,
                &[16, 3, 3, 3],
                0,
                &[1, 16, 30, 30],
                &[1], // <-- wrong
                &[0, 0],
            ),
            Err(BackendError::InvalidArgument(
                "stride must have 2 elements [sh, sw]".into()
            ))
        );
    }

    // ── Init is idempotent ────────────────────────────────────────────────────

    #[test]
    fn init_idempotent() {
        let Some(mut b) = try_init() else {
            return;
        };
        // Second call must succeed without error.
        assert_eq!(b.init(), Ok(()));
        assert!(b.is_initialized());
    }

    // ── Graceful failure ──────────────────────────────────────────────────────

    #[test]
    fn webgpu_init_graceful_failure() {
        // We cannot force a failure, but we can at least verify that init()
        // returns a Result and never panics.
        let mut b = WebGpuBackend::new();
        let _result = b.init(); // Ok or Err — both are acceptable.
        // No panic => test passes.
    }

    // ── GPU compute tests ─────────────────────────────────────────────────────
    //
    // These helpers upload f32 slices and read back results, exercising the
    // full shader → pipeline → dispatch path.

    /// Helper: upload `data` (f32 slice) to a new GPU buffer, return its handle.
    fn upload_f32(b: &WebGpuBackend, data: &[f32]) -> u64 {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let h = b.alloc(bytes.len()).expect("alloc");
        b.copy_htod(h, &bytes).expect("copy_htod");
        h
    }

    /// Helper: download `n` f32 values from a GPU buffer handle.
    fn download_f32(b: &WebGpuBackend, h: u64, n: usize) -> Vec<f32> {
        let mut bytes = vec![0u8; n * 4];
        b.copy_dtoh(&mut bytes, h).expect("copy_dtoh");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn unary_neg_small() {
        let Some(b) = try_init() else { return };
        let input = [1.0f32, -2.0, 3.0, 0.0];
        let in_h = upload_f32(&b, &input);
        let out_h = b.alloc(input.len() * 4).expect("alloc output");

        b.unary(UnaryOp::Neg, in_h, out_h, input.len())
            .expect("unary neg");

        let result = download_f32(&b, out_h, input.len());
        let expected = [-1.0f32, 2.0, -3.0, 0.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
        }

        b.free(in_h).expect("free");
        b.free(out_h).expect("free");
    }

    #[test]
    fn unary_abs_small() {
        let Some(b) = try_init() else { return };
        let input = [-3.0f32, 4.0, -5.0, 0.0];
        let in_h = upload_f32(&b, &input);
        let out_h = b.alloc(input.len() * 4).expect("alloc output");

        b.unary(UnaryOp::Abs, in_h, out_h, input.len())
            .expect("unary abs");

        let result = download_f32(&b, out_h, input.len());
        let expected = [3.0f32, 4.0, 5.0, 0.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
        }

        b.free(in_h).expect("free");
        b.free(out_h).expect("free");
    }

    #[test]
    fn binary_add_small() {
        let Some(b) = try_init() else { return };
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let bv = [10.0f32, 20.0, 30.0, 40.0];
        let a_h = upload_f32(&b, &a);
        let b_h = upload_f32(&b, &bv);
        let out_h = b.alloc(a.len() * 4).expect("alloc output");

        b.binary(BinaryOp::Add, a_h, b_h, out_h, a.len())
            .expect("binary add");

        let result = download_f32(&b, out_h, a.len());
        let expected = [11.0f32, 22.0, 33.0, 44.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
        }

        b.free(a_h).expect("free");
        b.free(b_h).expect("free");
        b.free(out_h).expect("free");
    }

    #[test]
    fn binary_mul_small() {
        let Some(b) = try_init() else { return };
        let a = [2.0f32, 3.0, 4.0, 5.0];
        let bv = [10.0f32, 10.0, 10.0, 10.0];
        let a_h = upload_f32(&b, &a);
        let b_h = upload_f32(&b, &bv);
        let out_h = b.alloc(a.len() * 4).expect("alloc output");

        b.binary(BinaryOp::Mul, a_h, b_h, out_h, a.len())
            .expect("binary mul");

        let result = download_f32(&b, out_h, a.len());
        let expected = [20.0f32, 30.0, 40.0, 50.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
        }

        b.free(a_h).expect("free");
        b.free(b_h).expect("free");
        b.free(out_h).expect("free");
    }

    #[test]
    fn reduce_sum_small() {
        let Some(b) = try_init() else { return };
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let in_h = upload_f32(&b, &input);
        let out_h = b.alloc(4).expect("alloc output"); // single f32

        b.reduce(ReduceOp::Sum, in_h, out_h, &[4], 0)
            .expect("reduce sum");

        let result = download_f32(&b, out_h, 1);
        assert!(
            (result[0] - 10.0).abs() < 1e-5,
            "expected 10.0, got {}",
            result[0]
        );

        b.free(in_h).expect("free");
        b.free(out_h).expect("free");
    }

    #[test]
    fn reduce_max_small() {
        let Some(b) = try_init() else { return };
        let input = [1.0f32, 5.0, 3.0, 2.0];
        let in_h = upload_f32(&b, &input);
        let out_h = b.alloc(4).expect("alloc output");

        b.reduce(ReduceOp::Max, in_h, out_h, &[4], 0)
            .expect("reduce max");

        let result = download_f32(&b, out_h, 1);
        assert!(
            (result[0] - 5.0).abs() < 1e-5,
            "expected 5.0, got {}",
            result[0]
        );

        b.free(in_h).expect("free");
        b.free(out_h).expect("free");
    }

    #[test]
    fn reduce_mean_small() {
        let Some(b) = try_init() else { return };
        let input = [2.0f32, 4.0, 6.0, 8.0];
        let in_h = upload_f32(&b, &input);
        let out_h = b.alloc(4).expect("alloc output");

        b.reduce(ReduceOp::Mean, in_h, out_h, &[4], 0)
            .expect("reduce mean");

        let result = download_f32(&b, out_h, 1);
        assert!(
            (result[0] - 5.0).abs() < 1e-5,
            "expected 5.0, got {}",
            result[0]
        );

        b.free(in_h).expect("free");
        b.free(out_h).expect("free");
    }

    #[test]
    fn gemm_identity_2x2() {
        let Some(b) = try_init() else { return };
        // A = [[1,2],[3,4]], B = [[1,0],[0,1]] (identity), C = zeros
        // C = 1.0 * A * I + 0.0 * C = A
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let eye = [1.0f32, 0.0, 0.0, 1.0];
        let c_init = [0.0f32; 4];

        let a_h = upload_f32(&b, &a);
        let b_h = upload_f32(&b, &eye);
        let c_h = upload_f32(&b, &c_init);

        b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a_h,
            2,
            b_h,
            2,
            0.0,
            c_h,
            2,
        )
        .expect("gemm");

        let result = download_f32(&b, c_h, 4);
        for (r, e) in result.iter().zip(a.iter()) {
            assert!((r - e).abs() < 1e-5, "got {r}, expected {e}");
        }

        b.free(a_h).expect("free");
        b.free(b_h).expect("free");
        b.free(c_h).expect("free");
    }

    #[test]
    fn gemm_2x3_times_3x2() {
        let Some(b) = try_init() else { return };
        // A 2x3, B 3x2 → C 2x2
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bm = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c_init = [0.0f32; 4];

        let a_h = upload_f32(&b, &a);
        let b_h = upload_f32(&b, &bm);
        let c_h = upload_f32(&b, &c_init);

        b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            3,
            1.0,
            a_h,
            3,
            b_h,
            2,
            0.0,
            c_h,
            2,
        )
        .expect("gemm");

        // Expected: [[58, 64], [139, 154]]
        let result = download_f32(&b, c_h, 4);
        let expected = [58.0f32, 64.0, 139.0, 154.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4, "got {r}, expected {e}");
        }

        b.free(a_h).expect("free");
        b.free(b_h).expect("free");
        b.free(c_h).expect("free");
    }

    #[test]
    fn gemm_alpha_beta() {
        let Some(b) = try_init() else { return };
        // C = 2.0 * A * B + 3.0 * C
        // A = [[1,0],[0,1]], B = [[1,0],[0,1]], C = [[1,1],[1,1]]
        // C = 2*I + 3*ones = [[5,3],[3,5]]
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let bm = [1.0f32, 0.0, 0.0, 1.0];
        let c_init = [1.0f32, 1.0, 1.0, 1.0];

        let a_h = upload_f32(&b, &a);
        let b_h = upload_f32(&b, &bm);
        let c_h = upload_f32(&b, &c_init);

        b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            2.0,
            a_h,
            2,
            b_h,
            2,
            3.0,
            c_h,
            2,
        )
        .expect("gemm alpha+beta");

        let result = download_f32(&b, c_h, 4);
        let expected = [5.0f32, 3.0, 3.0, 5.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4, "got {r}, expected {e}");
        }

        b.free(a_h).expect("free");
        b.free(b_h).expect("free");
        b.free(c_h).expect("free");
    }

    // ── Conv2D tests ──────────────────────────────────────────────────────

    #[test]
    fn conv2d_identity_1x1() {
        // 1×1 convolution with single channel, no padding, stride=1
        // input: 1×1×3×3, filter: 1×1×1×1 (weight=2.0), output: 1×1×3×3
        let Some(b) = try_init() else { return };
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let filter = [2.0f32];
        let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();

        let in_h = upload_f32(&b, &input);
        let f_h = upload_f32(&b, &filter);
        let out_h = b.alloc(9 * 4).expect("alloc output");

        b.conv2d_forward(
            in_h,
            &[1, 1, 3, 3],
            f_h,
            &[1, 1, 1, 1],
            out_h,
            &[1, 1, 3, 3],
            &[1, 1],
            &[0, 0],
        )
        .expect("conv2d");

        let result = download_f32(&b, out_h, 9);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5, "got {r}, expected {e}");
        }

        b.free(in_h).expect("free");
        b.free(f_h).expect("free");
        b.free(out_h).expect("free");
    }

    #[test]
    fn conv2d_3x3_no_padding() {
        // input: 1×1×4×4, filter: 1×1×3×3 (all ones), stride=1, pad=0
        // output: 1×1×2×2
        let Some(b) = try_init() else { return };
        let input: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let filter = [1.0f32; 9];

        let in_h = upload_f32(&b, &input);
        let f_h = upload_f32(&b, &filter);
        let out_h = b.alloc(4 * 4).expect("alloc output");

        b.conv2d_forward(
            in_h,
            &[1, 1, 4, 4],
            f_h,
            &[1, 1, 3, 3],
            out_h,
            &[1, 1, 2, 2],
            &[1, 1],
            &[0, 0],
        )
        .expect("conv2d");

        let result = download_f32(&b, out_h, 4);
        // top-left 3×3 sum: 0+1+2+4+5+6+8+9+10 = 45
        assert!((result[0] - 45.0).abs() < 1e-4, "got {}", result[0]);
        // top-right 3×3 sum: 1+2+3+5+6+7+9+10+11 = 54
        assert!((result[1] - 54.0).abs() < 1e-4, "got {}", result[1]);

        b.free(in_h).expect("free");
        b.free(f_h).expect("free");
        b.free(out_h).expect("free");
    }

    #[test]
    fn conv2d_with_padding() {
        // input: 1×1×2×2, filter: 1×1×3×3 (all ones), stride=1, pad=1
        // output: 1×1×2×2
        // With padding=1 around a 2×2 input, the output is also 2×2.
        let Some(b) = try_init() else { return };
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let filter = [1.0f32; 9];

        let in_h = upload_f32(&b, &input);
        let f_h = upload_f32(&b, &filter);
        let out_h = b.alloc(4 * 4).expect("alloc output");

        b.conv2d_forward(
            in_h,
            &[1, 1, 2, 2],
            f_h,
            &[1, 1, 3, 3],
            out_h,
            &[1, 1, 2, 2],
            &[1, 1],
            &[1, 1],
        )
        .expect("conv2d");

        let result = download_f32(&b, out_h, 4);
        // Top-left output: only 4 of 9 filter taps hit valid input
        // input[0,0]=1, input[0,1]=2, input[1,0]=3, input[1,1]=4 => sum=10
        assert!((result[0] - 10.0).abs() < 1e-4, "got {}", result[0]);

        b.free(in_h).expect("free");
        b.free(f_h).expect("free");
        b.free(out_h).expect("free");
    }

    // ── Attention tests ───────────────────────────────────────────────────

    #[test]
    fn attention_uniform_weights() {
        // 1 head, seq_q=1, seq_kv=2, head_dim=2, no causal
        // Q = [1, 0], K = [[1, 0], [1, 0]], V = [[1, 2], [3, 4]]
        // scores = [1*scale, 1*scale] => equal weights => O = mean(V) = [2, 3]
        let Some(b) = try_init() else { return };

        let q = [1.0f32, 0.0];
        let k = [1.0f32, 0.0, 1.0, 0.0];
        let v = [1.0f32, 2.0, 3.0, 4.0];

        let q_h = upload_f32(&b, &q);
        let k_h = upload_f32(&b, &k);
        let v_h = upload_f32(&b, &v);
        let o_h = b.alloc(2 * 4).expect("alloc output");

        b.attention(q_h, k_h, v_h, o_h, 1, 1, 1, 2, 2, 1.0, false)
            .expect("attention");

        let result = download_f32(&b, o_h, 2);
        // Equal scores → equal softmax weights → average of V rows
        assert!(
            (result[0] - 2.0).abs() < 1e-4,
            "got {}, expected 2.0",
            result[0]
        );
        assert!(
            (result[1] - 3.0).abs() < 1e-4,
            "got {}, expected 3.0",
            result[1]
        );

        b.free(q_h).expect("free");
        b.free(k_h).expect("free");
        b.free(v_h).expect("free");
        b.free(o_h).expect("free");
    }

    #[test]
    fn attention_causal_single_token() {
        // 1 head, seq_q=2, seq_kv=2, head_dim=1, causal
        // Q = [1, 1], K = [1, 1], V = [10, 20]
        // sq=0: only sees sk=0 → O[0] = V[0] = 10
        // sq=1: sees sk=0,1 with equal scores → O[1] = (10+20)/2 = 15
        let Some(b) = try_init() else { return };

        let q = [1.0f32, 1.0];
        let k = [1.0f32, 1.0];
        let v = [10.0f32, 20.0];

        let q_h = upload_f32(&b, &q);
        let k_h = upload_f32(&b, &k);
        let v_h = upload_f32(&b, &v);
        let o_h = b.alloc(2 * 4).expect("alloc output");

        b.attention(q_h, k_h, v_h, o_h, 1, 1, 2, 2, 1, 1.0, true)
            .expect("attention causal");

        let result = download_f32(&b, o_h, 2);
        assert!(
            (result[0] - 10.0).abs() < 1e-4,
            "got {}, expected 10.0",
            result[0]
        );
        assert!(
            (result[1] - 15.0).abs() < 1e-4,
            "got {}, expected 15.0",
            result[1]
        );

        b.free(q_h).expect("free");
        b.free(k_h).expect("free");
        b.free(v_h).expect("free");
        b.free(o_h).expect("free");
    }

    // ── Batched GEMM tests ─────────────────────────────────────────────

    #[test]
    fn batched_gemm_not_initialized() {
        let b = WebGpuBackend::new();
        let result = b.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            16,
            0,
            4,
            16,
            0.0,
            0,
            4,
            16,
            2,
        );
        assert_eq!(result, Err(BackendError::NotInitialized));
    }

    #[test]
    fn batched_gemm_zero_batch_noop() {
        let Some(b) = try_init() else { return };
        let result = b.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            16,
            0,
            4,
            16,
            0.0,
            0,
            4,
            16,
            0, // batch_count = 0
        );
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn batched_gemm_zero_dims_noop() {
        let Some(b) = try_init() else { return };
        // m = 0
        let result = b.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            0,
            4,
            4,
            1.0,
            0,
            4,
            16,
            0,
            4,
            16,
            0.0,
            0,
            4,
            16,
            2,
        );
        assert_eq!(result, Ok(()));
        // n = 0
        let result = b.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            0,
            4,
            1.0,
            0,
            4,
            16,
            0,
            4,
            16,
            0.0,
            0,
            4,
            16,
            2,
        );
        assert_eq!(result, Ok(()));
        // k = 0
        let result = b.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            0,
            1.0,
            0,
            4,
            16,
            0,
            4,
            16,
            0.0,
            0,
            4,
            16,
            2,
        );
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn batched_gemm_identity_2x2() {
        let Some(b) = try_init() else { return };
        // 2 batches of 2x2 identity multiply
        // batch 0: A0=[[1,2],[3,4]] * I = [[1,2],[3,4]]
        // batch 1: A1=[[5,6],[7,8]] * I = [[5,6],[7,8]]
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let eye = [1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let c_init = [0.0f32; 8];

        let a_h = upload_f32(&b, &a);
        let b_h = upload_f32(&b, &eye);
        let c_h = upload_f32(&b, &c_init);

        b.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a_h,
            2,
            4, // stride_a = 2*2 = 4
            b_h,
            2,
            4, // stride_b = 4
            0.0,
            c_h,
            2,
            4, // stride_c = 4
            2, // batch_count
        )
        .expect("batched_gemm");

        let result = download_f32(&b, c_h, 8);
        for (r, e) in result.iter().zip(a.iter()) {
            assert!((r - e).abs() < 1e-5, "got {r}, expected {e}");
        }

        b.free(a_h).expect("free");
        b.free(b_h).expect("free");
        b.free(c_h).expect("free");
    }

    // ── FP16 GEMM tests ─────────────────────────────────────────────────

    #[test]
    fn gemm_f16_not_initialized() {
        let b = WebGpuBackend::new();
        let result = b.gemm_f16(4, 4, 4, 1.0, 0, 0, 0.0, 0);
        assert_eq!(result, Err(BackendError::NotInitialized));
    }

    #[test]
    fn gemm_f16_zero_dims_noop() {
        let Some(b) = try_init() else { return };
        assert_eq!(b.gemm_f16(0, 4, 4, 1.0, 0, 0, 0.0, 0), Ok(()));
        assert_eq!(b.gemm_f16(4, 0, 4, 1.0, 0, 0, 0.0, 0), Ok(()));
        assert_eq!(b.gemm_f16(4, 4, 0, 1.0, 0, 0, 0.0, 0), Ok(()));
    }

    #[test]
    fn attention_dominant_key() {
        // 1 head, seq_q=1, seq_kv=2, head_dim=2, no causal
        // Q = [1, 0], K = [[10, 0], [0, 0]], V = [[100, 200], [0, 0]]
        // score[0] = 10*scale, score[1] = 0*scale
        // With large enough difference, softmax saturates → O ≈ V[0]
        let Some(b) = try_init() else { return };

        let q = [1.0f32, 0.0];
        let k = [10.0f32, 0.0, 0.0, 0.0];
        let v = [100.0f32, 200.0, 0.0, 0.0];

        let q_h = upload_f32(&b, &q);
        let k_h = upload_f32(&b, &k);
        let v_h = upload_f32(&b, &v);
        let o_h = b.alloc(2 * 4).expect("alloc output");

        // scale=1.0 gives scores 10 vs 0 → softmax ≈ [1, 0]
        b.attention(q_h, k_h, v_h, o_h, 1, 1, 1, 2, 2, 1.0, false)
            .expect("attention dominant");

        let result = download_f32(&b, o_h, 2);
        assert!(
            (result[0] - 100.0).abs() < 0.1,
            "got {}, expected ~100",
            result[0]
        );
        assert!(
            (result[1] - 200.0).abs() < 0.1,
            "got {}, expected ~200",
            result[1]
        );

        b.free(q_h).expect("free");
        b.free(k_h).expect("free");
        b.free(v_h).expect("free");
        b.free(o_h).expect("free");
    }
}
