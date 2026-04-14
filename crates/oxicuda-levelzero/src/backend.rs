//! [`LevelZeroBackend`] — the main entry point for the oxicuda-levelzero crate.
//!
//! Implements the [`ComputeBackend`] trait from `oxicuda-backend` using
//! Intel's Level Zero API for GPU compute on Linux and Windows.

use std::sync::Arc;

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};

use crate::{device::LevelZeroDevice, memory::LevelZeroMemoryManager};

// ─── Backend struct ───────────────────────────────────────────────────────────

/// Intel Level Zero GPU compute backend.
///
/// On Linux and Windows this selects the first Intel GPU via the Level Zero
/// loader library (`libze_loader.so` / `ze_loader.dll`) and allocates device
/// memory through the Level Zero memory model.
///
/// On macOS every operation returns [`BackendError::DeviceError`] wrapping
/// [`crate::error::LevelZeroError::UnsupportedPlatform`].
///
/// # Lifecycle
///
/// 1. `LevelZeroBackend::new()` — create an uninitialised backend.
/// 2. `init()` — load the Level Zero driver and select a GPU.
/// 3. Use `alloc`, `copy_htod`, compute ops, `copy_dtoh`, `free`.
/// 4. `synchronize()` — wait for all pending GPU work to finish.
#[derive(Debug)]
pub struct LevelZeroBackend {
    device: Option<Arc<LevelZeroDevice>>,
    memory: Option<Arc<LevelZeroMemoryManager>>,
    initialized: bool,
}

impl LevelZeroBackend {
    /// Create a new, uninitialised Level Zero backend.
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
    fn memory(&self) -> BackendResult<&Arc<LevelZeroMemoryManager>> {
        self.memory.as_ref().ok_or(BackendError::NotInitialized)
    }
}

impl Default for LevelZeroBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ComputeBackend impl ──────────────────────────────────────────────────────

impl ComputeBackend for LevelZeroBackend {
    fn name(&self) -> &str {
        "level-zero"
    }

    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }
        match LevelZeroDevice::new() {
            Ok(dev) => {
                let dev = Arc::new(dev);
                tracing::info!("Level Zero backend initialised on: {}", dev.name());
                let memory = LevelZeroMemoryManager::new(Arc::clone(&dev));
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
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        self.dispatch_gemm(m, n, k, alpha as f32, a_ptr, b_ptr, beta as f32, c_ptr)
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

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            if let Some(dev) = &self.device {
                let api = &dev.api;
                let queue = dev.queue;
                // SAFETY: `queue` is a valid command queue handle and the
                // backend is initialized.  u64::MAX means "wait indefinitely".
                let rc = unsafe { (api.ze_command_queue_synchronize)(queue, u64::MAX) };
                if rc != 0 {
                    return Err(BackendError::DeviceError(format!(
                        "zeCommandQueueSynchronize failed: 0x{rc:08x}"
                    )));
                }
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

// ─── Dispatch helpers ────────────────────────────────────────────────────────

/// Workgroup size matching the SPIR-V LocalSize declaration.
const WORKGROUP_SIZE: u32 = crate::spirv::WORKGROUP_SIZE;

/// A kernel argument value for the Level Zero dispatch pipeline.
#[cfg_attr(not(any(target_os = "linux", target_os = "windows")), allow(dead_code))]
enum KernelArg {
    /// Buffer handle — resolved to a raw device pointer at dispatch time.
    Buffer(u64),
    /// 32-bit unsigned integer scalar.
    U32(u32),
    /// 32-bit float scalar.
    F32(f32),
}

impl LevelZeroBackend {
    /// Dispatch a SPIR-V compute kernel via Level Zero.
    ///
    /// 1. Build a Level Zero module from `spv_words`.
    /// 2. Create a kernel named `"main"` from the module.
    /// 3. Set group size and kernel arguments.
    /// 4. Append a launch to a command list, execute, and wait.
    /// 5. Clean up all Level Zero objects.
    fn run_kernel(
        &self,
        spv_words: &[u32],
        args: &[KernelArg],
        workgroups: u32,
    ) -> BackendResult<()> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            use std::ffi::c_void;

            use crate::device::{
                ZE_MODULE_FORMAT_IL_SPIRV, ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                ZE_STRUCTURE_TYPE_KERNEL_DESC, ZE_STRUCTURE_TYPE_MODULE_DESC, ZeCommandListDesc,
                ZeGroupCount, ZeKernelDesc, ZeKernelHandle, ZeModuleDesc, ZeModuleHandle,
            };

            let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
            let memory = self.memory()?;
            let api = &device.api;
            let context = device.context;
            let dev_handle = device.device;
            let queue = device.queue;

            // ── 1. SPIR-V words → bytes ──
            let spv_bytes: Vec<u8> = spv_words.iter().flat_map(|w| w.to_ne_bytes()).collect();

            // ── 2. Create module ──
            let module_desc = ZeModuleDesc {
                stype: ZE_STRUCTURE_TYPE_MODULE_DESC,
                p_next: std::ptr::null(),
                format: ZE_MODULE_FORMAT_IL_SPIRV,
                input_size: spv_bytes.len(),
                p_input_module: spv_bytes.as_ptr(),
                p_build_flags: std::ptr::null(),
                p_constants: std::ptr::null(),
            };
            let mut module: ZeModuleHandle = std::ptr::null_mut();
            let rc = unsafe {
                (api.ze_module_create)(
                    context,
                    dev_handle,
                    &module_desc,
                    &mut module as *mut ZeModuleHandle,
                    std::ptr::null_mut(),
                )
            };
            if rc != 0 {
                return Err(BackendError::DeviceError(format!(
                    "zeModuleCreate failed: 0x{rc:08x}"
                )));
            }

            // ── 3. Create kernel ──
            let kernel_name = b"main\0";
            let kernel_desc = ZeKernelDesc {
                stype: ZE_STRUCTURE_TYPE_KERNEL_DESC,
                p_next: std::ptr::null(),
                flags: 0,
                p_kernel_name: kernel_name.as_ptr(),
            };
            let mut kernel: ZeKernelHandle = std::ptr::null_mut();
            let rc = unsafe {
                (api.ze_kernel_create)(module, &kernel_desc, &mut kernel as *mut ZeKernelHandle)
            };
            if rc != 0 {
                unsafe { (api.ze_module_destroy)(module) };
                return Err(BackendError::DeviceError(format!(
                    "zeKernelCreate failed: 0x{rc:08x}"
                )));
            }

            // ── 4. Set group size ──
            let rc = unsafe { (api.ze_kernel_set_group_size)(kernel, WORKGROUP_SIZE, 1, 1) };
            if rc != 0 {
                unsafe {
                    (api.ze_kernel_destroy)(kernel);
                    (api.ze_module_destroy)(module);
                }
                return Err(BackendError::DeviceError(format!(
                    "zeKernelSetGroupSize failed: 0x{rc:08x}"
                )));
            }

            // ── 5. Set kernel arguments ──
            for (idx, arg) in args.iter().enumerate() {
                let rc = match arg {
                    KernelArg::Buffer(handle) => {
                        let dev_ptr = memory.device_ptr(*handle).map_err(|e| {
                            unsafe {
                                (api.ze_kernel_destroy)(kernel);
                                (api.ze_module_destroy)(module);
                            }
                            BackendError::from(e)
                        })?;
                        unsafe {
                            (api.ze_kernel_set_argument_value)(
                                kernel,
                                idx as u32,
                                std::mem::size_of::<*mut c_void>(),
                                &dev_ptr as *const *mut c_void as *const c_void,
                            )
                        }
                    }
                    KernelArg::U32(val) => unsafe {
                        (api.ze_kernel_set_argument_value)(
                            kernel,
                            idx as u32,
                            std::mem::size_of::<u32>(),
                            val as *const u32 as *const c_void,
                        )
                    },
                    KernelArg::F32(val) => unsafe {
                        (api.ze_kernel_set_argument_value)(
                            kernel,
                            idx as u32,
                            std::mem::size_of::<f32>(),
                            val as *const f32 as *const c_void,
                        )
                    },
                };
                if rc != 0 {
                    unsafe {
                        (api.ze_kernel_destroy)(kernel);
                        (api.ze_module_destroy)(module);
                    }
                    return Err(BackendError::DeviceError(format!(
                        "zeKernelSetArgumentValue(arg={idx}) failed: 0x{rc:08x}"
                    )));
                }
            }

            // ── 6. Create command list ──
            let list_desc = ZeCommandListDesc {
                stype: ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                p_next: std::ptr::null(),
                command_queue_group_ordinal: 0,
                flags: 0,
            };
            let mut list = std::ptr::null_mut();
            let rc =
                unsafe { (api.ze_command_list_create)(context, dev_handle, &list_desc, &mut list) };
            if rc != 0 {
                unsafe {
                    (api.ze_kernel_destroy)(kernel);
                    (api.ze_module_destroy)(module);
                }
                return Err(BackendError::DeviceError(format!(
                    "zeCommandListCreate failed: 0x{rc:08x}"
                )));
            }

            // ── 7. Append launch kernel ──
            let group_count = ZeGroupCount {
                group_count_x: workgroups,
                group_count_y: 1,
                group_count_z: 1,
            };
            let rc = unsafe {
                (api.ze_command_list_append_launch_kernel)(
                    list,
                    kernel,
                    &group_count,
                    0,
                    0,
                    std::ptr::null(),
                )
            };
            if rc != 0 {
                unsafe {
                    (api.ze_command_list_destroy)(list);
                    (api.ze_kernel_destroy)(kernel);
                    (api.ze_module_destroy)(module);
                }
                return Err(BackendError::DeviceError(format!(
                    "zeCommandListAppendLaunchKernel failed: 0x{rc:08x}"
                )));
            }

            // ── 8. Close + execute + wait ──
            let rc = unsafe { (api.ze_command_list_close)(list) };
            if rc != 0 {
                unsafe {
                    (api.ze_command_list_destroy)(list);
                    (api.ze_kernel_destroy)(kernel);
                    (api.ze_module_destroy)(module);
                }
                return Err(BackendError::DeviceError(format!(
                    "zeCommandListClose failed: 0x{rc:08x}"
                )));
            }

            let rc = unsafe { (api.ze_command_queue_execute_command_lists)(queue, 1, &list, 0) };
            if rc != 0 {
                unsafe {
                    (api.ze_command_list_destroy)(list);
                    (api.ze_kernel_destroy)(kernel);
                    (api.ze_module_destroy)(module);
                }
                return Err(BackendError::DeviceError(format!(
                    "zeCommandQueueExecuteCommandLists failed: 0x{rc:08x}"
                )));
            }

            let rc = unsafe { (api.ze_command_queue_synchronize)(queue, u64::MAX) };
            if rc != 0 {
                unsafe {
                    (api.ze_command_list_destroy)(list);
                    (api.ze_kernel_destroy)(kernel);
                    (api.ze_module_destroy)(module);
                }
                return Err(BackendError::DeviceError(format!(
                    "zeCommandQueueSynchronize failed: 0x{rc:08x}"
                )));
            }

            // ── 9. Clean up ──
            unsafe {
                (api.ze_command_list_destroy)(list);
                (api.ze_kernel_destroy)(kernel);
                (api.ze_module_destroy)(module);
            }

            Ok(())
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            let _ = (spv_words, args, workgroups);
            Err(BackendError::DeviceError(
                "Level Zero requires Linux or Windows".into(),
            ))
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
        let args = [
            KernelArg::Buffer(input_ptr),
            KernelArg::Buffer(output_ptr),
            KernelArg::U32(n as u32),
        ];
        self.run_kernel(&spv, &args, (n as u32).div_ceil(WORKGROUP_SIZE))
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
        let args = [
            KernelArg::Buffer(a_ptr),
            KernelArg::Buffer(b_ptr),
            KernelArg::Buffer(output_ptr),
            KernelArg::U32(n as u32),
        ];
        self.run_kernel(&spv, &args, (n as u32).div_ceil(WORKGROUP_SIZE))
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
        let total_output = (outer_size * inner_size) as u32;
        let args = [
            KernelArg::Buffer(input_ptr),
            KernelArg::Buffer(output_ptr),
            KernelArg::U32(outer_size as u32),
            KernelArg::U32(reduce_size as u32),
            KernelArg::U32(inner_size as u32),
        ];
        self.run_kernel(&spv, &args, total_output.div_ceil(WORKGROUP_SIZE))
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
        let total = (m * n) as u32;
        let args = [
            KernelArg::Buffer(a_ptr),
            KernelArg::Buffer(b_ptr),
            KernelArg::Buffer(c_ptr),
            KernelArg::U32(m as u32),
            KernelArg::U32(n as u32),
            KernelArg::U32(k as u32),
            KernelArg::F32(alpha),
            KernelArg::F32(beta),
        ];
        self.run_kernel(&spv, &args, total.div_ceil(WORKGROUP_SIZE))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_backend::{BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp};

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn level_zero_backend_new_uninitialized() {
        let b = LevelZeroBackend::new();
        assert!(!b.is_initialized());
    }

    #[test]
    fn level_zero_backend_name() {
        let b = LevelZeroBackend::new();
        assert_eq!(b.name(), "level-zero");
    }

    #[test]
    fn level_zero_backend_default() {
        let b = LevelZeroBackend::default();
        assert!(!b.is_initialized());
        assert_eq!(b.name(), "level-zero");
    }

    #[test]
    fn backend_debug_impl() {
        let b = LevelZeroBackend::new();
        let s = format!("{b:?}");
        assert!(s.contains("LevelZeroBackend"));
    }

    // ── Object-safety smoke test ──────────────────────────────────────────────

    #[test]
    fn backend_object_safe() {
        let b: Box<dyn ComputeBackend> = Box::new(LevelZeroBackend::new());
        assert_eq!(b.name(), "level-zero");
    }

    // ── Not-initialized guards ────────────────────────────────────────────────

    #[test]
    fn backend_not_initialized_gemm() {
        let b = LevelZeroBackend::new();
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
        let b = LevelZeroBackend::new();
        assert_eq!(b.alloc(1024), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_synchronize() {
        let b = LevelZeroBackend::new();
        assert_eq!(b.synchronize(), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_free() {
        let b = LevelZeroBackend::new();
        assert_eq!(b.free(1), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_copy_htod() {
        let b = LevelZeroBackend::new();
        assert_eq!(b.copy_htod(1, b"hello"), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_copy_dtoh() {
        let b = LevelZeroBackend::new();
        let mut buf = [0u8; 4];
        assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
    }

    // ── Helper: try to get an initialised backend (skip if no GPU or no loader) ─

    fn try_init() -> Option<LevelZeroBackend> {
        let mut b = LevelZeroBackend::new();
        match b.init() {
            Ok(()) => Some(b),
            Err(_) => None,
        }
    }

    // ── Graceful init failure ─────────────────────────────────────────────────

    #[test]
    fn init_graceful_failure() {
        // Verify that init() returns a Result and never panics.
        let mut b = LevelZeroBackend::new();
        let _result = b.init();
        // Ok or Err — both are acceptable.
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

    // ── alloc/free/copy roundtrip ─────────────────────────────────────────────

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

    // ── Double init is a no-op ────────────────────────────────────────────────

    #[test]
    fn double_init_is_noop() {
        let Some(mut b) = try_init() else {
            return;
        };
        let first = b.is_initialized();
        let _ = b.init();
        assert_eq!(first, b.is_initialized());
    }

    // ── alloc and free basic ──────────────────────────────────────────────────

    #[test]
    fn alloc_and_free_basic() {
        let Some(b) = try_init() else {
            return;
        };
        match b.alloc(128) {
            Ok(handle) => {
                assert!(handle > 0);
                b.free(handle).expect("free should succeed");
            }
            Err(_) => {
                // Allocation failure is acceptable in environments without GPU.
            }
        }
    }

    // ── Conv2D correctness tests ──────────────────────────────────────────────

    /// Helper: allocate device memory, store f32 data, return handle.
    fn upload_f32(b: &LevelZeroBackend, data: &[f32]) -> Option<u64> {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_ne_bytes()).collect();
        let handle = b.alloc(bytes.len()).ok()?;
        b.copy_htod(handle, &bytes).ok()?;
        Some(handle)
    }

    /// Helper: download f32 data from device.
    fn download_f32(b: &LevelZeroBackend, handle: u64, len: usize) -> Option<Vec<f32>> {
        let mut bytes = vec![0u8; len * 4];
        b.copy_dtoh(&mut bytes, handle).ok()?;
        Some(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    #[test]
    fn l0_conv2d_identity_1x1() {
        let Some(b) = try_init() else {
            return;
        };
        // 1x1x4x4 input, 1x1x1x1 filter = identity (filter=[1.0])
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let filter = vec![1.0f32];
        let output_len = 16;

        let Some(in_h) = upload_f32(&b, &input) else {
            return;
        };
        let Some(flt_h) = upload_f32(&b, &filter) else {
            return;
        };
        let Some(out_h) = b.alloc(output_len * 4).ok() else {
            return;
        };

        let result = b.conv2d_forward(
            in_h,
            &[1, 1, 4, 4],
            flt_h,
            &[1, 1, 1, 1],
            out_h,
            &[1, 1, 4, 4],
            &[1, 1],
            &[0, 0],
        );
        assert!(result.is_ok(), "conv2d_forward failed: {result:?}");

        if let Some(out) = download_f32(&b, out_h, output_len) {
            for (i, &val) in out.iter().enumerate() {
                assert!(
                    (val - input[i]).abs() < 1e-5,
                    "mismatch at {i}: expected {}, got {val}",
                    input[i]
                );
            }
        }

        let _ = b.free(in_h);
        let _ = b.free(flt_h);
        let _ = b.free(out_h);
    }

    #[test]
    fn l0_conv2d_3x3_basic() {
        let Some(b) = try_init() else {
            return;
        };
        // 1x1x4x4 input, 1x1x3x3 filter, stride=1, pad=0 → 1x1x2x2 output
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        // All-ones 3x3 filter: output[oy,ox] = sum of 3x3 window
        let filter = vec![1.0f32; 9];
        let output_len = 4;

        let Some(in_h) = upload_f32(&b, &input) else {
            return;
        };
        let Some(flt_h) = upload_f32(&b, &filter) else {
            return;
        };
        let Some(out_h) = b.alloc(output_len * 4).ok() else {
            return;
        };

        let result = b.conv2d_forward(
            in_h,
            &[1, 1, 4, 4],
            flt_h,
            &[1, 1, 3, 3],
            out_h,
            &[1, 1, 2, 2],
            &[1, 1],
            &[0, 0],
        );
        assert!(result.is_ok());

        // Expected:
        // out[0,0] = 0+1+2+4+5+6+8+9+10 = 45
        // out[0,1] = 1+2+3+5+6+7+9+10+11 = 54
        // out[1,0] = 4+5+6+8+9+10+12+13+14 = 81
        // out[1,1] = 5+6+7+9+10+11+13+14+15 = 90
        let expected = [45.0f32, 54.0, 81.0, 90.0];
        if let Some(out) = download_f32(&b, out_h, output_len) {
            for (i, &val) in out.iter().enumerate() {
                assert!(
                    (val - expected[i]).abs() < 1e-4,
                    "mismatch at {i}: expected {}, got {val}",
                    expected[i]
                );
            }
        }

        let _ = b.free(in_h);
        let _ = b.free(flt_h);
        let _ = b.free(out_h);
    }

    #[test]
    fn l0_conv2d_with_padding() {
        let Some(b) = try_init() else {
            return;
        };
        // 1x1x3x3 input, 1x1x3x3 filter (all ones), stride=1, pad=1 → 1x1x3x3 output
        let input: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        let filter = vec![1.0f32; 9];
        let output_len = 9;

        let Some(in_h) = upload_f32(&b, &input) else {
            return;
        };
        let Some(flt_h) = upload_f32(&b, &filter) else {
            return;
        };
        let Some(out_h) = b.alloc(output_len * 4).ok() else {
            return;
        };

        let result = b.conv2d_forward(
            in_h,
            &[1, 1, 3, 3],
            flt_h,
            &[1, 1, 3, 3],
            out_h,
            &[1, 1, 3, 3],
            &[1, 1],
            &[1, 1],
        );
        assert!(result.is_ok());

        // Center element: sum of all 9 = 45
        if let Some(out) = download_f32(&b, out_h, output_len) {
            assert!(
                (out[4] - 45.0).abs() < 1e-4,
                "center expected 45, got {}",
                out[4]
            );
            // Corner [0,0]: sum of [1,2,4,5] = 12
            assert!(
                (out[0] - 12.0).abs() < 1e-4,
                "corner expected 12, got {}",
                out[0]
            );
        }

        let _ = b.free(in_h);
        let _ = b.free(flt_h);
        let _ = b.free(out_h);
    }

    // ── Attention correctness tests ───────────────────────────────────────────

    #[test]
    fn l0_attention_uniform() {
        let Some(b) = try_init() else {
            return;
        };
        // batch=1, heads=1, seq_q=2, seq_kv=2, head_dim=2
        // Q=K=all zeros → uniform attention → O = mean(V)
        let seq_q = 2;
        let seq_kv = 2;
        let head_dim = 2;
        let q = vec![0.0f32; seq_q * head_dim];
        let k = vec![0.0f32; seq_kv * head_dim];
        let v = vec![1.0f32, 2.0, 3.0, 4.0]; // V[0]=[1,2], V[1]=[3,4]
        let o_len = seq_q * head_dim;

        let Some(q_h) = upload_f32(&b, &q) else {
            return;
        };
        let Some(k_h) = upload_f32(&b, &k) else {
            return;
        };
        let Some(v_h) = upload_f32(&b, &v) else {
            return;
        };
        let Some(o_h) = b.alloc(o_len * 4).ok() else {
            return;
        };
        // Zero out output
        let zeros = vec![0u8; o_len * 4];
        let _ = b.copy_htod(o_h, &zeros);

        let scale = 1.0 / (head_dim as f64).sqrt();
        let result = b.attention(
            q_h, k_h, v_h, o_h, 1, 1, seq_q, seq_kv, head_dim, scale, false,
        );
        assert!(result.is_ok(), "attention failed: {result:?}");

        // With uniform attention weights, output = mean(V rows)
        // mean = [(1+3)/2, (2+4)/2] = [2, 3]
        if let Some(out) = download_f32(&b, o_h, o_len) {
            // Both query positions should get the same result
            for sq_idx in 0..seq_q {
                let off = sq_idx * head_dim;
                assert!(
                    (out[off] - 2.0).abs() < 1e-4,
                    "q{sq_idx}[0] expected 2.0, got {}",
                    out[off]
                );
                assert!(
                    (out[off + 1] - 3.0).abs() < 1e-4,
                    "q{sq_idx}[1] expected 3.0, got {}",
                    out[off + 1]
                );
            }
        }

        let _ = b.free(q_h);
        let _ = b.free(k_h);
        let _ = b.free(v_h);
        let _ = b.free(o_h);
    }

    #[test]
    fn l0_attention_causal() {
        let Some(b) = try_init() else {
            return;
        };
        // batch=1, heads=1, seq_q=2, seq_kv=2, head_dim=2, causal=true
        let seq_q = 2;
        let seq_kv = 2;
        let head_dim = 2;
        let q = vec![0.0f32; seq_q * head_dim];
        let k = vec![0.0f32; seq_kv * head_dim];
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let o_len = seq_q * head_dim;

        let Some(q_h) = upload_f32(&b, &q) else {
            return;
        };
        let Some(k_h) = upload_f32(&b, &k) else {
            return;
        };
        let Some(v_h) = upload_f32(&b, &v) else {
            return;
        };
        let Some(o_h) = b.alloc(o_len * 4).ok() else {
            return;
        };
        let zeros = vec![0u8; o_len * 4];
        let _ = b.copy_htod(o_h, &zeros);

        let scale = 1.0 / (head_dim as f64).sqrt();
        let result = b.attention(
            q_h, k_h, v_h, o_h, 1, 1, seq_q, seq_kv, head_dim, scale, true,
        );
        assert!(result.is_ok());

        if let Some(out) = download_f32(&b, o_h, o_len) {
            // q=0 (causal: can only attend to k=0): output = V[0] = [1, 2]
            assert!(
                (out[0] - 1.0).abs() < 1e-4,
                "q0[0] expected 1.0, got {}",
                out[0]
            );
            assert!(
                (out[1] - 2.0).abs() < 1e-4,
                "q0[1] expected 2.0, got {}",
                out[1]
            );
            // q=1 (can attend to k=0,1): output = mean(V) = [2, 3]
            assert!(
                (out[2] - 2.0).abs() < 1e-4,
                "q1[0] expected 2.0, got {}",
                out[2]
            );
            assert!(
                (out[3] - 3.0).abs() < 1e-4,
                "q1[1] expected 3.0, got {}",
                out[3]
            );
        }

        let _ = b.free(q_h);
        let _ = b.free(k_h);
        let _ = b.free(v_h);
        let _ = b.free(o_h);
    }

    #[test]
    fn l0_attention_dominant_key() {
        let Some(b) = try_init() else {
            return;
        };
        // One key has a very large dot product → attention should be concentrated on it
        let seq_q = 1;
        let seq_kv = 3;
        let head_dim = 2;
        // Q = [10, 0]
        let q = vec![10.0f32, 0.0];
        // K = [[10, 0], [0, 0], [0, 0]]  → dot(Q,K[0]) = 100, others = 0
        let k = vec![10.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0]; // V[0]=[1,0], V[1]=[0,1], V[2]=[0,0]
        let o_len = seq_q * head_dim;

        let Some(q_h) = upload_f32(&b, &q) else {
            return;
        };
        let Some(k_h) = upload_f32(&b, &k) else {
            return;
        };
        let Some(v_h) = upload_f32(&b, &v) else {
            return;
        };
        let Some(o_h) = b.alloc(o_len * 4).ok() else {
            return;
        };
        let zeros = vec![0u8; o_len * 4];
        let _ = b.copy_htod(o_h, &zeros);

        let scale = 1.0;
        let result = b.attention(
            q_h, k_h, v_h, o_h, 1, 1, seq_q, seq_kv, head_dim, scale, false,
        );
        assert!(result.is_ok());

        if let Some(out) = download_f32(&b, o_h, o_len) {
            // With dot=100*scale=100 for K[0] vs 0 for others,
            // softmax should heavily favour V[0]=[1,0]
            assert!(out[0] > 0.99, "expected output[0] ≈ 1.0, got {}", out[0]);
            assert!(out[1] < 0.01, "expected output[1] ≈ 0.0, got {}", out[1]);
        }

        let _ = b.free(q_h);
        let _ = b.free(k_h);
        let _ = b.free(v_h);
        let _ = b.free(o_h);
    }
}
