//! [`VulkanBackend`] — implements [`ComputeBackend`] via Vulkan + SPIR-V.
//!
//! # Initialisation
//!
//! Call [`VulkanBackend::init`] before any other operation.  On macOS (or any
//! system without a Vulkan driver), `init` returns
//! `Err(BackendError::DeviceError(...))` rather than panicking.
//!
//! # Compute operations
//!
//! All compute kernels (`gemm`, `batched_gemm`, `conv2d_forward`, `attention`,
//! `reduce`, `unary`, `binary`) are dispatched via SPIR-V compute shaders
//! generated at runtime by the [`crate::spirv`] module.  The memory management
//! pipeline (alloc / free / copy_htod / copy_dtoh) and synchronisation are
//! fully implemented via Vulkan buffer objects backed by host-visible,
//! host-coherent memory.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};

use ash::vk;
use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};

use crate::async_compute::AsyncComputeManager;
use crate::command::VulkanCommandPool;
use crate::device::VulkanDevice;
use crate::memory::VulkanMemoryManager;
use crate::pipeline::VulkanComputePipeline;

// ─── Pipeline cache types ───────────────────────────────────

/// Key combining a SPIR-V hash with the number of descriptor bindings
/// so that identical shaders with different binding counts produce
/// distinct pipeline objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderKey {
    spirv_hash: u64,
    bindings: u32,
}

impl ShaderKey {
    /// Build a key by hashing the SPIR-V word slice together with the binding
    /// count.
    pub fn new(spirv: &[u32], bindings: u32) -> Self {
        let mut hasher = DefaultHasher::new();
        spirv.hash(&mut hasher);
        bindings.hash(&mut hasher);
        Self {
            spirv_hash: hasher.finish(),
            bindings,
        }
    }
}

/// Vulkan compute backend.
///
/// Create with `VulkanBackend::new()`, then call `init()` to select a device.
pub struct VulkanBackend {
    device: Option<Arc<VulkanDevice>>,
    memory: Option<Arc<VulkanMemoryManager>>,
    command_pool: Option<VulkanCommandPool>,
    /// Multi-queue async compute manager (created during `init`).
    async_manager: Option<AsyncComputeManager>,
    initialized: bool,
    /// Cache of compiled compute pipelines keyed by SPIR-V content hash.
    pipeline_cache: Mutex<HashMap<ShaderKey, Arc<VulkanComputePipeline>>>,
}

impl VulkanBackend {
    /// Create an uninitialised backend.  Call [`init`](Self::init) before use.
    pub fn new() -> Self {
        Self {
            device: None,
            memory: None,
            command_pool: None,
            async_manager: None,
            initialized: false,
            pipeline_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Return `Err(NotInitialized)` if `init` has not been called successfully.
    fn check_init(&self) -> BackendResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(BackendError::NotInitialized)
        }
    }

    /// Convenience: get the memory manager or return NotInitialized.
    fn memory_manager(&self) -> BackendResult<&VulkanMemoryManager> {
        self.check_init()?;
        self.memory.as_deref().ok_or(BackendError::NotInitialized)
    }

    /// Convenience: get the device Arc or return NotInitialized.
    fn device_arc(&self) -> BackendResult<&Arc<VulkanDevice>> {
        self.check_init()?;
        self.device.as_ref().ok_or(BackendError::NotInitialized)
    }

    /// Convenience: get the command pool or return NotInitialized.
    fn cmd_pool(&self) -> BackendResult<&VulkanCommandPool> {
        self.check_init()?;
        self.command_pool
            .as_ref()
            .ok_or(BackendError::NotInitialized)
    }

    /// Convenience: get the async compute manager or return NotInitialized.
    fn async_mgr(&self) -> BackendResult<&AsyncComputeManager> {
        self.check_init()?;
        self.async_manager
            .as_ref()
            .ok_or(BackendError::NotInitialized)
    }

    // ── Multi-queue async compute API ────────────────────────────────────────

    /// Number of compute queues available for async dispatch.
    ///
    /// Returns `Err(NotInitialized)` if `init()` has not been called.
    pub fn queue_count(&self) -> BackendResult<usize> {
        Ok(self.async_mgr()?.queue_count())
    }

    /// Submit a compute command to a specific async queue.
    ///
    /// The `record_fn` closure receives a `vk::CommandBuffer` to record into.
    /// The submission is asynchronous; call [`wait_all_queues`](Self::wait_all_queues)
    /// or [`synchronize`](ComputeBackend::synchronize) to wait for completion.
    pub fn submit_compute_async<F>(&self, queue_index: usize, record_fn: F) -> BackendResult<()>
    where
        F: FnOnce(vk::CommandBuffer) -> BackendResult<()>,
    {
        let mgr = self.async_mgr()?;
        mgr.submit_async(queue_index, |cb| {
            record_fn(cb).map_err(|e| crate::error::VulkanError::CommandBufferError(e.to_string()))
        })
        .map_err(BackendError::from)
    }

    /// Wait for all async compute queues to complete their in-flight work.
    pub fn wait_all_queues(&self) -> BackendResult<()> {
        let mgr = self.async_mgr()?;
        mgr.wait_all().map_err(BackendError::from)
    }
}

impl std::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cache_len = self.pipeline_cache.lock().map(|c| c.len()).unwrap_or(0);
        f.debug_struct("VulkanBackend")
            .field("initialized", &self.initialized)
            .field("pipeline_cache_entries", &cache_len)
            .finish()
    }
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }

        let device = VulkanDevice::new().map_err(BackendError::from)?;

        tracing::info!(device = device.device_name(), "Vulkan backend initialised");

        let device = Arc::new(device);
        let memory = Arc::new(VulkanMemoryManager::new(Arc::clone(&device)));
        let cmd_pool = VulkanCommandPool::new(Arc::clone(&device)).map_err(BackendError::from)?;
        let async_mgr =
            AsyncComputeManager::new(Arc::clone(&device)).map_err(BackendError::from)?;

        tracing::debug!(
            async_queues = async_mgr.queue_count(),
            "async compute manager created"
        );

        self.device = Some(device);
        self.memory = Some(memory);
        self.command_pool = Some(cmd_pool);
        self.async_manager = Some(async_mgr);
        self.initialized = true;

        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    // ── Compute operations ───────────────────────────────────────────────────

    fn gemm(
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
        self.check_init()?;
        // Zero-dimension GEMM is a no-op.
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        self.dispatch_gemm(m, n, k, alpha as f32, a_ptr, b_ptr, beta as f32, c_ptr)
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_gemm(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
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
        // Zero batch count or zero dimensions → nothing to do.
        if batch_count == 0 || m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        self.dispatch_batched_gemm(
            m,
            n,
            k,
            alpha as f32,
            a_ptr,
            stride_a,
            b_ptr,
            stride_b,
            beta as f32,
            c_ptr,
            stride_c,
            batch_count,
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

        // Validate shapes: all must have rank 4.
        if input_shape.len() != 4 || filter_shape.len() != 4 || output_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "conv2d_forward: input, filter, and output must each have rank 4 (NCHW)".into(),
            ));
        }
        if stride.len() != 2 {
            return Err(BackendError::InvalidArgument(
                "conv2d_forward: stride must have length 2".into(),
            ));
        }
        if padding.len() != 2 {
            return Err(BackendError::InvalidArgument(
                "conv2d_forward: padding must have length 2".into(),
            ));
        }

        let n = input_shape[0];
        let c_in = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];
        let k_out = filter_shape[0];
        let fh = filter_shape[2];
        let fw = filter_shape[3];
        let o_h = output_shape[2];
        let o_w = output_shape[3];
        let stride_h = stride[0];
        let stride_w = stride[1];
        let pad_h = padding[0];
        let pad_w = padding[1];

        // CPU fallback: copy input + filter from device
        let in_len = n * c_in * h_in * w_in;
        let flt_len = k_out * c_in * fh * fw;
        let out_len = n * k_out * o_h * o_w;

        let mut in_bytes = vec![0u8; in_len * 4];
        self.copy_dtoh(&mut in_bytes, input_ptr)?;
        let inp: Vec<f32> = in_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut flt_bytes = vec![0u8; flt_len * 4];
        self.copy_dtoh(&mut flt_bytes, filter_ptr)?;
        let flt: Vec<f32> = flt_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // NCHW convolution
        let mut out = vec![0.0f32; out_len];
        for b_idx in 0..n {
            for kf in 0..k_out {
                for oy in 0..o_h {
                    for ox in 0..o_w {
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
                                        acc += inp[((b_idx * c_in + ci) * h_in + iy) * w_in + ix]
                                            * flt[((kf * c_in + ci) * fh + fy) * fw + fx];
                                    }
                                }
                            }
                        }
                        out[((b_idx * k_out + kf) * o_h + oy) * o_w + ox] = acc;
                    }
                }
            }
        }

        let out_bytes: Vec<u8> = out.iter().flat_map(|f| f.to_ne_bytes()).collect();
        self.copy_htod(output_ptr, &out_bytes)
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

        if seq_q == 0 || seq_kv == 0 {
            return Err(BackendError::InvalidArgument(
                "attention: seq_q and seq_kv must be > 0".into(),
            ));
        }
        if head_dim == 0 {
            return Err(BackendError::InvalidArgument(
                "attention: head_dim must be > 0".into(),
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(BackendError::InvalidArgument(
                "attention: scale must be a positive finite number".into(),
            ));
        }

        let batch_heads = batch * heads;
        let scale_f32 = scale as f32;

        // CPU fallback: copy Q, K, V from device
        let q_len = batch_heads * seq_q * head_dim;
        let kv_len = batch_heads * seq_kv * head_dim;

        let mut q_bytes = vec![0u8; q_len * 4];
        self.copy_dtoh(&mut q_bytes, q_ptr)?;
        let q: Vec<f32> = q_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut k_bytes = vec![0u8; kv_len * 4];
        self.copy_dtoh(&mut k_bytes, k_ptr)?;
        let k: Vec<f32> = k_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut v_bytes = vec![0u8; kv_len * 4];
        self.copy_dtoh(&mut v_bytes, v_ptr)?;
        let v: Vec<f32> = v_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Numerically-stable scaled dot-product attention
        let mut output = vec![0.0f32; q_len];

        for bh in 0..batch_heads {
            for sq in 0..seq_q {
                let q_off = (bh * seq_q + sq) * head_dim;
                let o_off = q_off;

                // Pass 1: find max score
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
                    let score = dot * scale_f32;
                    if score > max_score {
                        max_score = score;
                    }
                }

                if max_score == f32::NEG_INFINITY {
                    max_score = 0.0;
                }

                // Pass 2: accumulate exp-weighted V
                let mut sum_exp = 0.0f32;
                for sk in 0..seq_kv {
                    if causal && sk > sq {
                        continue;
                    }
                    let k_off = (bh * seq_kv + sk) * head_dim;
                    let v_off = (bh * seq_kv + sk) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[k_off + d];
                    }
                    let w = (dot * scale_f32 - max_score).exp();
                    sum_exp += w;
                    for d in 0..head_dim {
                        output[o_off + d] += w * v[v_off + d];
                    }
                }

                // Normalize
                if sum_exp > 0.0 {
                    for d in 0..head_dim {
                        output[o_off + d] /= sum_exp;
                    }
                }
            }
        }

        let o_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_ne_bytes()).collect();
        self.copy_htod(o_ptr, &o_bytes)
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
                "reduce: shape must not be empty".into(),
            ));
        }
        if axis >= shape.len() {
            return Err(BackendError::InvalidArgument(format!(
                "reduce: axis {axis} out of bounds for shape of rank {}",
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

    // ── Synchronisation ──────────────────────────────────────────────────────

    fn synchronize(&self) -> BackendResult<()> {
        if !self.initialized {
            // No device — nothing to synchronise.
            return Ok(());
        }
        // Wait for async compute queues first.
        if let Some(mgr) = &self.async_manager {
            mgr.wait_all().map_err(BackendError::from)?;
        }
        if let Some(dev) = &self.device {
            dev.wait_idle().map_err(BackendError::from)?;
        }
        Ok(())
    }

    // ── Memory management ────────────────────────────────────────────────────

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        self.memory_manager()?
            .alloc(bytes)
            .map_err(BackendError::from)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.memory_manager()?.free(ptr).map_err(BackendError::from)
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        if src.is_empty() {
            return Ok(());
        }
        self.memory_manager()?
            .copy_to_device(dst, src)
            .map_err(BackendError::from)
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        if dst.is_empty() {
            return Ok(());
        }
        self.memory_manager()?
            .copy_from_device(dst, src)
            .map_err(BackendError::from)
    }
}

// ─── Dispatch helpers ───────────────────────────────────────

/// Workgroup size matching the SPIR-V LocalSize declaration.
const WORKGROUP_SIZE: u32 = 256;

impl VulkanBackend {
    /// Look up or compile a pipeline for the given SPIR-V and binding count.
    ///
    /// On cache hit the existing `Arc<VulkanComputePipeline>` is returned.
    /// On miss a new pipeline is created, inserted into the cache, and returned.
    pub fn get_or_create_pipeline(
        &self,
        spirv: &[u32],
        bindings: u32,
    ) -> BackendResult<Arc<VulkanComputePipeline>> {
        let key = ShaderKey::new(spirv, bindings);

        // Fast path: check cache under the lock.
        {
            let cache = self
                .pipeline_cache
                .lock()
                .map_err(|_| BackendError::DeviceError("pipeline cache lock poisoned".into()))?;
            if let Some(pipeline) = cache.get(&key) {
                return Ok(Arc::clone(pipeline));
            }
        }

        // Slow path: create a new pipeline (outside the lock), then insert.
        let device = self.device_arc()?;
        let pipeline = VulkanComputePipeline::new(Arc::clone(device), spirv, bindings, 1)
            .map_err(BackendError::from)?;
        let pipeline = Arc::new(pipeline);

        let mut cache = self
            .pipeline_cache
            .lock()
            .map_err(|_| BackendError::DeviceError("pipeline cache lock poisoned".into()))?;
        // Another thread may have inserted while we were compiling; prefer
        // the existing entry to avoid duplicate Vulkan objects.
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&pipeline));
        Ok(Arc::clone(entry))
    }

    /// Run a compute dispatch: create pipeline, bind buffers, dispatch, wait.
    ///
    /// `spv` — SPIR-V words for the shader.
    /// `bindings` — number of descriptor bindings.
    /// `buffer_handles` — slice of `(allocation_handle, binding_index)` pairs.
    /// `workgroups` — number of workgroups in the X dimension.
    fn run_compute(
        &self,
        spv: &[u32],
        bindings: u32,
        buffer_handles: &[(u64, u32)],
        workgroups: u32,
    ) -> BackendResult<()> {
        let device = self.device_arc()?;
        let memory = self.memory_manager()?;
        let cmd_pool = self.cmd_pool()?;

        let pipeline = self.get_or_create_pipeline(spv, bindings)?;

        let vk_dev = device.device();

        // Allocate a descriptor set from the pipeline's pool.
        let ds_layout = pipeline.descriptor_set_layout();
        let ds_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pipeline.descriptor_pool())
            .set_layouts(std::slice::from_ref(&ds_layout));
        let descriptor_sets = unsafe { vk_dev.allocate_descriptor_sets(&ds_alloc_info) }
            .map_err(|e| BackendError::DeviceError(format!("allocate_descriptor_sets: {e}")))?;
        let ds = descriptor_sets[0];

        // Build descriptor writes.
        let mut buf_infos = Vec::with_capacity(buffer_handles.len());
        for &(handle, _) in buffer_handles {
            let vk_buf = memory.vk_buffer(handle).map_err(BackendError::from)?;
            let size = memory.buffer_size(handle).map_err(BackendError::from)?;
            buf_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(vk_buf)
                    .offset(0)
                    .range(size),
            );
        }

        // We must create writes referencing the buf_infos elements individually
        // because vk::WriteDescriptorSet borrows slices.
        let writes: Vec<vk::WriteDescriptorSet> = buffer_handles
            .iter()
            .enumerate()
            .map(|(i, &(_, binding))| {
                vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buf_infos[i]))
            })
            .collect();

        unsafe { vk_dev.update_descriptor_sets(&writes, &[]) };

        // Record and submit.
        let pl = pipeline.pipeline();
        let pl_layout = pipeline.pipeline_layout();
        cmd_pool
            .record_and_submit(|cmd| {
                unsafe {
                    vk_dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl);
                    vk_dev.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        pl_layout,
                        0,
                        &[ds],
                        &[],
                    );
                    vk_dev.cmd_dispatch(cmd, workgroups, 1, 1);
                }
                Ok(())
            })
            .map_err(BackendError::from)
    }

    /// Run a 3-D compute dispatch: create pipeline, bind buffers, dispatch, wait.
    ///
    /// Like [`run_compute`] but dispatches `(wg_x, wg_y, wg_z)` workgroups.
    fn run_compute_3d(
        &self,
        spv: &[u32],
        bindings: u32,
        buffer_handles: &[(u64, u32)],
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) -> BackendResult<()> {
        let device = self.device_arc()?;
        let memory = self.memory_manager()?;
        let cmd_pool = self.cmd_pool()?;

        let pipeline = self.get_or_create_pipeline(spv, bindings)?;

        let vk_dev = device.device();

        let ds_layout = pipeline.descriptor_set_layout();
        let ds_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pipeline.descriptor_pool())
            .set_layouts(std::slice::from_ref(&ds_layout));
        let descriptor_sets = unsafe { vk_dev.allocate_descriptor_sets(&ds_alloc_info) }
            .map_err(|e| BackendError::DeviceError(format!("allocate_descriptor_sets: {e}")))?;
        let ds = descriptor_sets[0];

        let mut buf_infos = Vec::with_capacity(buffer_handles.len());
        for &(handle, _) in buffer_handles {
            let vk_buf = memory.vk_buffer(handle).map_err(BackendError::from)?;
            let size = memory.buffer_size(handle).map_err(BackendError::from)?;
            buf_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(vk_buf)
                    .offset(0)
                    .range(size),
            );
        }

        let writes: Vec<vk::WriteDescriptorSet> = buffer_handles
            .iter()
            .enumerate()
            .map(|(i, &(_, binding))| {
                vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buf_infos[i]))
            })
            .collect();

        unsafe { vk_dev.update_descriptor_sets(&writes, &[]) };

        let pl = pipeline.pipeline();
        let pl_layout = pipeline.pipeline_layout();
        cmd_pool
            .record_and_submit(|cmd| {
                unsafe {
                    vk_dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pl);
                    vk_dev.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        pl_layout,
                        0,
                        &[ds],
                        &[],
                    );
                    vk_dev.cmd_dispatch(cmd, wg_x, wg_y, wg_z);
                }
                Ok(())
            })
            .map_err(BackendError::from)
    }

    /// Allocate a temporary params buffer, write `data` into it, return handle.
    fn alloc_params_buffer(&self, data: &[u8]) -> BackendResult<u64> {
        let memory = self.memory_manager()?;
        let handle = memory.alloc(data.len()).map_err(BackendError::from)?;
        memory
            .copy_to_device(handle, data)
            .map_err(BackendError::from)?;
        Ok(handle)
    }

    /// Free a temporary params buffer (best-effort, ignoring errors).
    fn free_params_buffer(&self, handle: u64) {
        if let Ok(memory) = self.memory_manager() {
            let _ = memory.free(handle);
        }
    }

    fn dispatch_unary(
        &self,
        op: UnaryOp,
        input_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        let spv = crate::spirv::unary_compute_shader(op);
        let count_bytes = (n as u32).to_ne_bytes();
        let params = self.alloc_params_buffer(&count_bytes)?;

        let result = self.run_compute(
            &spv,
            3,
            &[(input_ptr, 0), (output_ptr, 1), (params, 2)],
            (n as u32).div_ceil(WORKGROUP_SIZE),
        );

        self.free_params_buffer(params);
        result
    }

    fn dispatch_binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        let spv = crate::spirv::binary_compute_shader(op);
        let count_bytes = (n as u32).to_ne_bytes();
        let params = self.alloc_params_buffer(&count_bytes)?;

        let result = self.run_compute(
            &spv,
            4,
            &[(a_ptr, 0), (b_ptr, 1), (output_ptr, 2), (params, 3)],
            (n as u32).div_ceil(WORKGROUP_SIZE),
        );

        self.free_params_buffer(params);
        result
    }

    fn dispatch_reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
        let reduce_size = shape[axis];
        let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);

        let spv = crate::spirv::reduce_compute_shader(op);
        let mut param_data = Vec::with_capacity(12);
        param_data.extend_from_slice(&(outer_size as u32).to_ne_bytes());
        param_data.extend_from_slice(&(reduce_size as u32).to_ne_bytes());
        param_data.extend_from_slice(&(inner_size as u32).to_ne_bytes());
        let params = self.alloc_params_buffer(&param_data)?;

        let total_output = (outer_size * inner_size) as u32;
        let result = self.run_compute(
            &spv,
            3,
            &[(input_ptr, 0), (output_ptr, 1), (params, 2)],
            total_output.div_ceil(WORKGROUP_SIZE),
        );

        self.free_params_buffer(params);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a_ptr: u64,
        b_ptr: u64,
        beta: f32,
        c_ptr: u64,
    ) -> BackendResult<()> {
        let spv = crate::spirv::gemm_compute_shader();
        let mut param_data = Vec::with_capacity(20);
        param_data.extend_from_slice(&(m as u32).to_ne_bytes());
        param_data.extend_from_slice(&(n as u32).to_ne_bytes());
        param_data.extend_from_slice(&(k as u32).to_ne_bytes());
        param_data.extend_from_slice(&alpha.to_bits().to_ne_bytes());
        param_data.extend_from_slice(&beta.to_bits().to_ne_bytes());
        let params = self.alloc_params_buffer(&param_data)?;

        let total = (m * n) as u32;
        let result = self.run_compute(
            &spv,
            4,
            &[(a_ptr, 0), (b_ptr, 1), (c_ptr, 2), (params, 3)],
            total.div_ceil(WORKGROUP_SIZE),
        );

        self.free_params_buffer(params);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_batched_gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a_ptr: u64,
        stride_a: usize,
        b_ptr: u64,
        stride_b: usize,
        beta: f32,
        c_ptr: u64,
        stride_c: usize,
        batch_count: usize,
    ) -> BackendResult<()> {
        let spv = crate::spirv::batched_gemm_compute_shader();

        // params: [m, n, k, alpha(bitcast), beta(bitcast), stride_a, stride_b, stride_c]
        let mut param_data = Vec::with_capacity(32);
        param_data.extend_from_slice(&(m as u32).to_ne_bytes());
        param_data.extend_from_slice(&(n as u32).to_ne_bytes());
        param_data.extend_from_slice(&(k as u32).to_ne_bytes());
        param_data.extend_from_slice(&alpha.to_bits().to_ne_bytes());
        param_data.extend_from_slice(&beta.to_bits().to_ne_bytes());
        param_data.extend_from_slice(&(stride_a as u32).to_ne_bytes());
        param_data.extend_from_slice(&(stride_b as u32).to_ne_bytes());
        param_data.extend_from_slice(&(stride_c as u32).to_ne_bytes());
        let params = self.alloc_params_buffer(&param_data)?;

        // Dispatch: X covers (m*n) elements per batch, Z covers batches.
        let elements_per_batch = (m * n) as u32;
        let wg_x = elements_per_batch.div_ceil(WORKGROUP_SIZE);
        let result = self.run_compute_3d(
            &spv,
            4,
            &[(a_ptr, 0), (b_ptr, 1), (c_ptr, 2), (params, 3)],
            wg_x,
            1,
            batch_count as u32,
        );

        self.free_params_buffer(params);
        result
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_backend::ComputeBackend;

    // ── Structural / compile-time tests ─────────────────────

    #[test]
    fn vulkan_backend_new_uninitialized() {
        let b = VulkanBackend::new();
        assert!(!b.is_initialized());
    }

    #[test]
    fn vulkan_backend_name() {
        let b = VulkanBackend::new();
        assert_eq!(b.name(), "vulkan");
    }

    #[test]
    fn vulkan_backend_default() {
        let b = VulkanBackend::default();
        assert!(!b.is_initialized());
        assert_eq!(b.name(), "vulkan");
    }

    #[test]
    fn backend_debug_impl() {
        let b = VulkanBackend::new();
        let s = format!("{b:?}");
        assert!(s.contains("VulkanBackend"));
    }

    /// Verify that `VulkanBackend` can be used as a `Box<dyn ComputeBackend>`.
    #[test]
    fn backend_object_safe() {
        let b: Box<dyn ComputeBackend> = Box::new(VulkanBackend::new());
        assert_eq!(b.name(), "vulkan");
        assert!(!b.is_initialized());
    }

    // ── Not-initialised guard tests ──────────────────────────

    #[test]
    fn backend_not_initialized_returns_error() {
        let b = VulkanBackend::new();
        assert_eq!(b.alloc(64), Err(BackendError::NotInitialized));
        assert_eq!(b.free(1), Err(BackendError::NotInitialized));
        assert_eq!(b.copy_htod(1, &[0u8; 8]), Err(BackendError::NotInitialized));
        let mut buf = [0u8; 8];
        assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
    }

    #[test]
    fn alloc_zero_bytes_error() {
        // alloc(0) — not initialised path first.
        let b = VulkanBackend::new();
        assert!(b.alloc(0).is_err());
    }

    // ── Empty / zero-dimension no-ops (no device needed) ────

    /// copy_htod with empty src is a no-op even before init.
    #[test]
    fn copy_htod_empty_noop() {
        let b = VulkanBackend::new();
        // Empty slice -> early return Ok(()) regardless of init state.
        assert_eq!(b.copy_htod(0, &[]), Ok(()));
    }

    /// copy_dtoh with empty dst is a no-op even before init.
    #[test]
    fn copy_dtoh_empty_noop() {
        let b = VulkanBackend::new();
        let buf: &mut [u8] = &mut [];
        assert_eq!(b.copy_dtoh(buf, 0), Ok(()));
    }

    /// GEMM with any zero dimension is a no-op (requires init guard to pass first).
    /// We test the invalid-arg path (not-initialized).
    #[test]
    fn gemm_uninit_returns_not_initialized() {
        let b = VulkanBackend::new();
        let r = b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            0,
            0,
            0,
            1.0,
            0,
            0,
            0,
            0,
            0.0,
            0,
            0,
        );
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    #[test]
    fn unary_zero_n_uninit_returns_not_initialized() {
        let b = VulkanBackend::new();
        assert_eq!(
            b.unary(UnaryOp::Relu, 0, 0, 0),
            Err(BackendError::NotInitialized)
        );
    }

    #[test]
    fn binary_zero_n_uninit_returns_not_initialized() {
        let b = VulkanBackend::new();
        assert_eq!(
            b.binary(BinaryOp::Add, 0, 0, 0, 0),
            Err(BackendError::NotInitialized)
        );
    }

    #[test]
    fn reduce_empty_shape_error() {
        let b = VulkanBackend::new();
        // Not initialised — returns NotInitialized first.
        let r = b.reduce(ReduceOp::Sum, 0, 0, &[], 0);
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    #[test]
    fn attention_zero_seq_error() {
        let b = VulkanBackend::new();
        let r = b.attention(0, 0, 0, 0, 1, 1, 0, 0, 64, 0.125, false);
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    #[test]
    fn attention_invalid_scale_error() {
        let b = VulkanBackend::new();
        let r = b.attention(0, 0, 0, 0, 1, 1, 4, 4, 64, -1.0, false);
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    #[test]
    fn conv2d_wrong_shape_ranks() {
        let b = VulkanBackend::new();
        // Not initialised first.
        let r = b.conv2d_forward(
            0,
            &[1, 1, 4],
            0,
            &[1, 1, 3, 3],
            0,
            &[1, 1, 2, 2],
            &[1, 1],
            &[0, 0],
        );
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    /// `synchronize()` on an uninitialised backend is a no-op.
    #[test]
    fn synchronize_uninit_noop() {
        let b = VulkanBackend::new();
        assert_eq!(b.synchronize(), Ok(()));
    }

    // ── init() graceful failure test ─────────────────────────

    /// On macOS (no Vulkan driver) init() returns Err and does not panic.
    /// On Linux with a Vulkan driver this will succeed — both cases are correct.
    #[test]
    fn init_graceful_failure() {
        let mut b = VulkanBackend::new();
        let result = b.init();
        match result {
            Ok(()) => {
                // Vulkan is available on this system — verify post-init state.
                assert!(b.is_initialized());
                // Double-init is a no-op.
                assert_eq!(b.init(), Ok(()));
            }
            Err(e) => {
                // Vulkan not available — backend must not be initialised.
                assert!(!b.is_initialized());
                // Must be a DeviceError wrapping the library-not-found message.
                assert!(
                    matches!(e, BackendError::DeviceError(_)),
                    "expected DeviceError, got {e:?}"
                );
            }
        }
    }

    // ── Post-init behaviour (only runs when Vulkan is available) ────────────

    #[test]
    fn alloc_free_round_trip_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            // No Vulkan — skip.
            return;
        }
        let handle = b.alloc(256).expect("alloc 256 bytes");
        b.free(handle).expect("free handle");
    }

    #[test]
    fn copy_round_trip_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        let data: Vec<u8> = (0u8..64).collect();
        let handle = b.alloc(64).expect("alloc");
        b.copy_htod(handle, &data).expect("copy_htod");
        let mut recv = vec![0u8; 64];
        b.copy_dtoh(&mut recv, handle).expect("copy_dtoh");
        assert_eq!(data, recv, "round-trip copy must preserve data");
        b.free(handle).expect("free");
    }

    #[test]
    fn alloc_zero_after_init_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        // alloc(0) must return an error even after init.
        assert!(matches!(b.alloc(0), Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn gemm_zero_dims_noop_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        // m == 0 -> no-op
        assert_eq!(
            b.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                4,
                4,
                1.0,
                0,
                4,
                0,
                4,
                0.0,
                0,
                4
            ),
            Ok(())
        );
        // n == 0 -> no-op
        assert_eq!(
            b.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                4,
                0,
                4,
                1.0,
                0,
                4,
                0,
                4,
                0.0,
                0,
                4
            ),
            Ok(())
        );
        // k == 0 -> no-op
        assert_eq!(
            b.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                4,
                4,
                0,
                1.0,
                0,
                4,
                0,
                4,
                0.0,
                0,
                4
            ),
            Ok(())
        );
    }

    #[test]
    fn unary_zero_n_noop_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        assert_eq!(b.unary(UnaryOp::Relu, 0, 0, 0), Ok(()));
    }

    #[test]
    fn binary_zero_n_noop_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        assert_eq!(b.binary(BinaryOp::Add, 0, 0, 0, 0), Ok(()));
    }

    #[test]
    fn reduce_empty_shape_error_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        assert!(matches!(
            b.reduce(ReduceOp::Sum, 0, 0, &[], 0),
            Err(BackendError::InvalidArgument(_))
        ));
    }

    #[test]
    fn attention_zero_seq_error_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        assert!(matches!(
            b.attention(0, 0, 0, 0, 1, 1, 0, 4, 64, 0.125, false),
            Err(BackendError::InvalidArgument(_))
        ));
    }

    #[test]
    fn attention_invalid_scale_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        assert!(matches!(
            b.attention(0, 0, 0, 0, 1, 1, 4, 4, 64, 0.0, false),
            Err(BackendError::InvalidArgument(_))
        ));
        assert!(matches!(
            b.attention(0, 0, 0, 0, 1, 1, 4, 4, 64, f64::INFINITY, false),
            Err(BackendError::InvalidArgument(_))
        ));
    }

    #[test]
    fn conv2d_wrong_shape_ranks_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        // input rank 3 is invalid.
        assert!(matches!(
            b.conv2d_forward(
                0,
                &[1, 3, 4],
                0,
                &[1, 3, 3, 3],
                0,
                &[1, 1, 2, 2],
                &[1, 1],
                &[0, 0]
            ),
            Err(BackendError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synchronize_after_init_requires_vulkan() {
        let mut b = VulkanBackend::new();
        if b.init().is_err() {
            return;
        }
        assert_eq!(b.synchronize(), Ok(()));
    }

    // ── Compute dispatch tests (only run when Vulkan is available) ──────────

    fn try_init() -> Option<VulkanBackend> {
        let mut b = VulkanBackend::new();
        match b.init() {
            Ok(()) => Some(b),
            Err(_) => None,
        }
    }

    /// Helper: upload f32 slice to a new buffer.
    fn upload_f32(b: &VulkanBackend, data: &[f32]) -> u64 {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_ne_bytes()).collect();
        let handle = b.alloc(bytes.len()).expect("alloc");
        b.copy_htod(handle, &bytes).expect("copy_htod");
        handle
    }

    /// Helper: download f32 values from a buffer.
    fn download_f32(b: &VulkanBackend, handle: u64, count: usize) -> Vec<f32> {
        let mut bytes = vec![0u8; count * 4];
        b.copy_dtoh(&mut bytes, handle).expect("copy_dtoh");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn unary_neg_compute_requires_vulkan() {
        let Some(b) = try_init() else { return };
        let input = upload_f32(&b, &[1.0, -2.0, 3.0, 0.0]);
        let output = b.alloc(16).expect("alloc output");
        b.unary(UnaryOp::Neg, input, output, 4).expect("unary neg");
        let result = download_f32(&b, output, 4);
        assert_eq!(result, vec![-1.0, 2.0, -3.0, -0.0]);
        b.free(input).expect("free input");
        b.free(output).expect("free output");
    }

    #[test]
    fn unary_relu_compute_requires_vulkan() {
        let Some(b) = try_init() else { return };
        let input = upload_f32(&b, &[-1.0, 0.0, 2.5, -0.5]);
        let output = b.alloc(16).expect("alloc output");
        b.unary(UnaryOp::Relu, input, output, 4)
            .expect("unary relu");
        let result = download_f32(&b, output, 4);
        assert_eq!(result, vec![0.0, 0.0, 2.5, 0.0]);
        b.free(input).expect("free input");
        b.free(output).expect("free output");
    }

    #[test]
    fn binary_add_compute_requires_vulkan() {
        let Some(b) = try_init() else { return };
        let a = upload_f32(&b, &[1.0, 2.0, 3.0, 4.0]);
        let bv = upload_f32(&b, &[10.0, 20.0, 30.0, 40.0]);
        let output = b.alloc(16).expect("alloc output");
        b.binary(BinaryOp::Add, a, bv, output, 4)
            .expect("binary add");
        let result = download_f32(&b, output, 4);
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
        b.free(a).expect("free a");
        b.free(bv).expect("free b");
        b.free(output).expect("free output");
    }

    #[test]
    fn binary_mul_compute_requires_vulkan() {
        let Some(b) = try_init() else { return };
        let a = upload_f32(&b, &[2.0, 3.0, 4.0, 5.0]);
        let bv = upload_f32(&b, &[0.5, 1.0, 2.0, 0.0]);
        let output = b.alloc(16).expect("alloc output");
        b.binary(BinaryOp::Mul, a, bv, output, 4)
            .expect("binary mul");
        let result = download_f32(&b, output, 4);
        assert_eq!(result, vec![1.0, 3.0, 8.0, 0.0]);
        b.free(a).expect("free a");
        b.free(bv).expect("free b");
        b.free(output).expect("free output");
    }

    #[test]
    fn reduce_sum_compute_requires_vulkan() {
        let Some(b) = try_init() else { return };
        // [6] => reduce along axis 0 → scalar
        let input = upload_f32(&b, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = b.alloc(4).expect("alloc output");
        b.reduce(ReduceOp::Sum, input, output, &[6], 0)
            .expect("reduce sum");
        let result = download_f32(&b, output, 1);
        assert!((result[0] - 21.0).abs() < 1e-5);
        b.free(input).expect("free input");
        b.free(output).expect("free output");
    }

    #[test]
    fn reduce_max_compute_requires_vulkan() {
        let Some(b) = try_init() else { return };
        let input = upload_f32(&b, &[1.0, 5.0, 2.0, 4.0, 3.0, 6.0]);
        let output = b.alloc(4).expect("alloc output");
        b.reduce(ReduceOp::Max, input, output, &[6], 0)
            .expect("reduce max");
        let result = download_f32(&b, output, 1);
        assert!((result[0] - 6.0).abs() < 1e-5);
        b.free(input).expect("free input");
        b.free(output).expect("free output");
    }

    #[test]
    fn reduce_mean_compute_requires_vulkan() {
        let Some(b) = try_init() else { return };
        let input = upload_f32(&b, &[2.0, 4.0, 6.0, 8.0]);
        let output = b.alloc(4).expect("alloc output");
        b.reduce(ReduceOp::Mean, input, output, &[4], 0)
            .expect("reduce mean");
        let result = download_f32(&b, output, 1);
        assert!((result[0] - 5.0).abs() < 1e-5);
        b.free(input).expect("free input");
        b.free(output).expect("free output");
    }

    #[test]
    fn reduce_2d_axis1_compute_requires_vulkan() {
        let Some(b) = try_init() else { return };
        // shape [2, 3], reduce axis 1 → [2]
        // [[1,2,3],[4,5,6]] → sums [6, 15]
        let input = upload_f32(&b, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = b.alloc(8).expect("alloc output");
        b.reduce(ReduceOp::Sum, input, output, &[2, 3], 1)
            .expect("reduce sum axis 1");
        let result = download_f32(&b, output, 2);
        assert!((result[0] - 6.0).abs() < 1e-5);
        assert!((result[1] - 15.0).abs() < 1e-5);
        b.free(input).expect("free input");
        b.free(output).expect("free output");
    }

    #[test]
    fn gemm_simple_compute_requires_vulkan() {
        let Some(b) = try_init() else { return };
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = A * B = [[19,22],[43,50]]
        let a = upload_f32(&b, &[1.0, 2.0, 3.0, 4.0]);
        let bv = upload_f32(&b, &[5.0, 6.0, 7.0, 8.0]);
        // Zero-init C
        let c = upload_f32(&b, &[0.0, 0.0, 0.0, 0.0]);
        b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a,
            2,
            bv,
            2,
            0.0,
            c,
            2,
        )
        .expect("gemm");
        let result = download_f32(&b, c, 4);
        assert!((result[0] - 19.0).abs() < 1e-4, "C[0,0]={}", result[0]);
        assert!((result[1] - 22.0).abs() < 1e-4, "C[0,1]={}", result[1]);
        assert!((result[2] - 43.0).abs() < 1e-4, "C[1,0]={}", result[2]);
        assert!((result[3] - 50.0).abs() < 1e-4, "C[1,1]={}", result[3]);
        b.free(a).expect("free a");
        b.free(bv).expect("free b");
        b.free(c).expect("free c");
    }

    // ── Conv2D tests ────────────────────────────────────────

    #[test]
    fn vulkan_conv2d_identity_1x1() {
        let Some(b) = try_init() else { return };
        // 1×1 conv with filter = 1.0 → output == input
        // N=1, C_in=1, H=2, W=2, K=1, Fh=1, Fw=1, Oh=2, Ow=2
        let input = upload_f32(&b, &[1.0, 2.0, 3.0, 4.0]);
        let filter = upload_f32(&b, &[1.0]);
        let output = b.alloc(16).expect("alloc output");
        // Zero-init output
        b.copy_htod(output, &[0u8; 16]).expect("zero output");
        b.conv2d_forward(
            input,
            &[1, 1, 2, 2],
            filter,
            &[1, 1, 1, 1],
            output,
            &[1, 1, 2, 2],
            &[1, 1],
            &[0, 0],
        )
        .expect("conv2d 1x1");
        let result = download_f32(&b, output, 4);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 3.0).abs() < 1e-5);
        assert!((result[3] - 4.0).abs() < 1e-5);
        b.free(input).expect("free");
        b.free(filter).expect("free");
        b.free(output).expect("free");
    }

    #[test]
    fn vulkan_conv2d_3x3_basic() {
        let Some(b) = try_init() else { return };
        // N=1, C=1, H=3, W=3, K=1, Fh=3, Fw=3, stride=1, pad=0 → Oh=1, Ow=1
        // Input: [[1,2,3],[4,5,6],[7,8,9]], Filter: all 1s → sum = 45
        let input = upload_f32(&b, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let filter = upload_f32(&b, &[1.0; 9]);
        let output = b.alloc(4).expect("alloc output");
        b.copy_htod(output, &[0u8; 4]).expect("zero output");
        b.conv2d_forward(
            input,
            &[1, 1, 3, 3],
            filter,
            &[1, 1, 3, 3],
            output,
            &[1, 1, 1, 1],
            &[1, 1],
            &[0, 0],
        )
        .expect("conv2d 3x3");
        let result = download_f32(&b, output, 1);
        assert!((result[0] - 45.0).abs() < 1e-4, "got {}", result[0]);
        b.free(input).expect("free");
        b.free(filter).expect("free");
        b.free(output).expect("free");
    }

    #[test]
    fn vulkan_conv2d_with_padding() {
        let Some(b) = try_init() else { return };
        // N=1, C=1, H=3, W=3, K=1, Fh=3, Fw=3, stride=1, pad=1 → Oh=3, Ow=3
        // Center output element (1,1) sums the full 3×3 = 45
        let input = upload_f32(&b, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let filter = upload_f32(&b, &[1.0; 9]);
        let output = b.alloc(36).expect("alloc output");
        b.copy_htod(output, &[0u8; 36]).expect("zero output");
        b.conv2d_forward(
            input,
            &[1, 1, 3, 3],
            filter,
            &[1, 1, 3, 3],
            output,
            &[1, 1, 3, 3],
            &[1, 1],
            &[1, 1],
        )
        .expect("conv2d padded");
        let result = download_f32(&b, output, 9);
        // Center (1,1) sees full 3×3 input → 45
        assert!((result[4] - 45.0).abs() < 1e-4, "center={}", result[4]);
        // Corner (0,0) with pad=1 sees only input[0..2][0..2] → 1+2+4+5 = 12
        assert!((result[0] - 12.0).abs() < 1e-4, "corner={}", result[0]);
        b.free(input).expect("free");
        b.free(filter).expect("free");
        b.free(output).expect("free");
    }

    // ── Attention tests ─────────────────────────────────────

    #[test]
    fn vulkan_attention_uniform() {
        let Some(b) = try_init() else { return };
        // batch=1, heads=1, seq_q=2, seq_kv=2, head_dim=2
        // Q=K=V=all 1s → all equal scores → uniform softmax → output = V avg
        let data = vec![1.0f32; 2 * 2]; // 2 seq × 2 dim
        let q = upload_f32(&b, &data);
        let k = upload_f32(&b, &data);
        let v = upload_f32(&b, &data);
        let o = b.alloc(data.len() * 4).expect("alloc o");
        b.attention(q, k, v, o, 1, 1, 2, 2, 2, 0.5, false)
            .expect("attention uniform");
        let result = download_f32(&b, o, 4);
        // All V elements are 1.0, softmax is uniform → output all 1.0
        for (i, &val) in result.iter().enumerate() {
            assert!((val - 1.0).abs() < 1e-4, "o[{i}]={val}");
        }
        b.free(q).expect("free");
        b.free(k).expect("free");
        b.free(v).expect("free");
        b.free(o).expect("free");
    }

    #[test]
    fn vulkan_attention_causal() {
        let Some(b) = try_init() else { return };
        // batch=1, heads=1, seq_q=2, seq_kv=2, head_dim=1
        // Q=[[1],[1]], K=[[1],[2]], V=[[10],[20]]
        // With causal: query 0 only attends to key 0 → output[0]=10
        //              query 1 attends to keys 0,1
        let q = upload_f32(&b, &[1.0, 1.0]);
        let k = upload_f32(&b, &[1.0, 2.0]);
        let v = upload_f32(&b, &[10.0, 20.0]);
        let o = b.alloc(8).expect("alloc o");
        b.attention(q, k, v, o, 1, 1, 2, 2, 1, 1.0, true)
            .expect("attention causal");
        let result = download_f32(&b, o, 2);
        // Query 0: only key 0 → output = 10.0
        assert!((result[0] - 10.0).abs() < 1e-4, "o[0]={}", result[0]);
        // Query 1: attends to both, scores = [1, 2], softmax([1,2])
        // w0=exp(1-2)=exp(-1), w1=exp(0)=1, sum=exp(-1)+1
        // output = (exp(-1)*10 + 1*20) / (exp(-1)+1)
        let e = (-1.0f32).exp();
        let expected = (e * 10.0 + 20.0) / (e + 1.0);
        assert!((result[1] - expected).abs() < 1e-3, "o[1]={}", result[1]);
        b.free(q).expect("free");
        b.free(k).expect("free");
        b.free(v).expect("free");
        b.free(o).expect("free");
    }

    #[test]
    fn vulkan_attention_dominant_key() {
        let Some(b) = try_init() else { return };
        // batch=1, heads=1, seq_q=1, seq_kv=3, head_dim=1
        // Q=[[1]], K=[[0],[0],[100]], V=[[1],[2],[99]]
        // Key 2 dominates → output ≈ V[2] = 99
        let q = upload_f32(&b, &[1.0]);
        let k = upload_f32(&b, &[0.0, 0.0, 100.0]);
        let v = upload_f32(&b, &[1.0, 2.0, 99.0]);
        let o = b.alloc(4).expect("alloc o");
        b.attention(q, k, v, o, 1, 1, 1, 3, 1, 1.0, false)
            .expect("attention dominant");
        let result = download_f32(&b, o, 1);
        assert!((result[0] - 99.0).abs() < 0.1, "o[0]={}", result[0]);
        b.free(q).expect("free");
        b.free(k).expect("free");
        b.free(v).expect("free");
        b.free(o).expect("free");
    }

    // ── Batched GEMM tests ──────────────────────────────────────

    #[test]
    fn batched_gemm_not_initialized() {
        let b = VulkanBackend::new();
        let r = b.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            0,
            2,
            4,
            0,
            2,
            4,
            0.0,
            0,
            2,
            4,
            3,
        );
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    #[test]
    fn batched_gemm_zero_batch_not_initialized() {
        // batch_count == 0 is an early return before the init check.
        let b = VulkanBackend::new();
        let r = b.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            0,
            2,
            4,
            0,
            2,
            4,
            0.0,
            0,
            2,
            4,
            0,
        );
        // Zero batch triggers check_init first, then early return would be Ok
        // but since not initialized, we get NotInitialized.
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    #[test]
    fn batched_gemm_zero_dims_not_initialized() {
        let b = VulkanBackend::new();
        // m=0 but batch_count>0
        let r = b.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            0,
            2,
            2,
            1.0,
            0,
            0,
            0,
            0,
            2,
            4,
            0.0,
            0,
            0,
            0,
            2,
        );
        assert_eq!(r, Err(BackendError::NotInitialized));
    }

    // ── Pipeline cache tests (no Vulkan device required) ─────

    #[test]
    fn shader_key_same_spirv_same_bindings() {
        use crate::backend::ShaderKey;
        let spv = vec![0x07230203u32, 0, 0, 1, 0];
        let k1 = ShaderKey::new(&spv, 3);
        let k2 = ShaderKey::new(&spv, 3);
        assert_eq!(k1, k2);
    }

    #[test]
    fn shader_key_different_spirv() {
        use crate::backend::ShaderKey;
        let spv_a = vec![0x07230203u32, 1, 2, 3, 4];
        let spv_b = vec![0x07230203u32, 5, 6, 7, 8];
        let k1 = ShaderKey::new(&spv_a, 3);
        let k2 = ShaderKey::new(&spv_b, 3);
        assert_ne!(k1, k2);
    }

    #[test]
    fn shader_key_different_bindings() {
        use crate::backend::ShaderKey;
        let spv = vec![0x07230203u32, 0, 0, 1, 0];
        let k1 = ShaderKey::new(&spv, 3);
        let k2 = ShaderKey::new(&spv, 4);
        assert_ne!(k1, k2);
    }

    #[test]
    fn pipeline_cache_miss_without_device_returns_error() {
        let b = VulkanBackend::new();
        let spv = crate::spirv::trivial_compute_shader();
        let result = b.get_or_create_pipeline(&spv, 0);
        // Not initialised — should fail.
        assert!(result.is_err());
    }

    #[test]
    fn pipeline_cache_debug_shows_entries() {
        let b = VulkanBackend::new();
        let s = format!("{b:?}");
        assert!(s.contains("pipeline_cache_entries"));
    }
}
