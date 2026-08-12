//! GPU dispatch for the Metal backend's neural-network operations.
//!
//! Until this module existed, `MetalBackend::conv2d_forward` and
//! `MetalBackend::attention` were pure-CPU scalar loops that copied every
//! operand to the host and the result back, while the finished MSL kernels for
//! both sat in [`crate::msl`] / [`crate::msl_nn`] with no caller at all, and
//! `softmax` inherited the trait's `Unsupported` default despite a complete
//! stable-softmax shader shipping in the same crate.
//!
//! Everything here dispatches a **runtime-parameterised** kernel: shapes travel
//! in a constant buffer rather than being baked into the MSL source, so one
//! compiled pipeline per `(op, dtype)` serves every shape and the bounded
//! pipeline cache in [`super::types`] cannot be churned by varying batch or
//! sequence sizes.
//!
//! The host implementations that used to *be* these ops survive only as
//! [`attention_host`] — the genuine fallback for a configuration the GPU kernel
//! cannot express (an attention head whose accumulator exceeds the device's
//! threadgroup-memory budget).

#[cfg(target_os = "macos")]
use oxicuda_backend::BackendError;
use oxicuda_backend::BackendResult;

use super::types::MetalBackend;

#[cfg(target_os = "macos")]
use super::functions::{next_power_of_2, pow2_threadgroup, resolve_buffers, to_u32};
#[cfg(target_os = "macos")]
use super::types::PipelineKey;

/// SIMD-groups per threadgroup targeted by the attention kernel.
///
/// Four SIMD-groups (128 threads) keeps enough threadgroups in flight to cover
/// memory latency without making the per-SIMD-group accumulator slice of
/// threadgroup memory (`head_dim * 4` bytes each) dominate the budget. Clamped
/// down at dispatch by both the pipeline's thread limit and the device's
/// threadgroup-memory budget.
#[cfg(target_os = "macos")]
const ATTENTION_SIMDGROUPS: usize = 4;

/// Lanes in one Apple GPU SIMD-group. Fixed by the hardware and by
/// [`crate::msl_nn::attention_msl_v2`]'s `LANES` constant.
#[cfg(target_os = "macos")]
const SIMD_LANES: usize = 32;

/// Threadgroup width targeted by the row-wise softmax / layer-norm kernels.
#[cfg(target_os = "macos")]
const ROWWISE_TARGET_THREADS: usize = 256;

/// Runtime parameter buffer for [`crate::msl::conv2d_msl_v2`].
///
/// Field order and types mirror the MSL `ConvParamsV2` struct exactly; see that
/// generator's doc comment for the authoritative byte-offset table.
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ConvParamsV2 {
    n_batch: u32,
    c_in: u32,
    h_in: u32,
    w_in: u32,
    k_out: u32,
    fh: u32,
    fw: u32,
    oh: u32,
    ow: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
}

/// Runtime parameter buffer for [`crate::msl_nn::attention_msl_v2`].
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AttnParamsV2 {
    batch_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    head_dim: u32,
    causal: u32,
    scale: f32,
}

/// The convolution geometry `dispatch_conv2d` needs, already validated and
/// narrowed by the trait entry point.
///
/// Off macOS the dispatcher is a stub that reports `UnsupportedPlatform` without
/// looking at the geometry, so the fields are legitimately never read there.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
pub(super) struct Conv2dGeometry {
    /// Input batch size `N`.
    pub n: usize,
    /// Input channels `C`.
    pub c_in: usize,
    /// Input height `H`.
    pub h_in: usize,
    /// Input width `W`.
    pub w_in: usize,
    /// Output channels `K`.
    pub k_out: usize,
    /// Filter height.
    pub fh: usize,
    /// Filter width.
    pub fw: usize,
    /// Output height.
    pub oh: usize,
    /// Output width.
    pub ow: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
    /// Vertical zero padding.
    pub pad_h: usize,
    /// Horizontal zero padding.
    pub pad_w: usize,
}

/// The attention problem shape `dispatch_attention` needs.
///
/// As [`Conv2dGeometry`], the fields are unread off macOS.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
pub(super) struct AttentionGeometry {
    /// `batch * heads` — the leading axis of Q/K/V/O.
    pub batch_heads: usize,
    /// Query positions.
    pub seq_q: usize,
    /// Key/value positions.
    pub seq_kv: usize,
    /// Per-head feature width.
    pub head_dim: usize,
    /// Attention scale (already validated finite and positive).
    pub scale: f64,
    /// Top-left causal masking.
    pub causal: bool,
}

#[cfg(target_os = "macos")]
impl MetalBackend {
    /// Conv2D forward on the GPU (NCHW), one thread per output element.
    ///
    /// Uses [`crate::msl::conv2d_msl_v2`], whose shapes live in a constant
    /// buffer, so a single compiled pipeline serves every convolution shape.
    pub(super) fn dispatch_conv2d(
        &self,
        input_ptr: u64,
        filter_ptr: u64,
        output_ptr: u64,
        geom: Conv2dGeometry,
    ) -> BackendResult<()> {
        let total = geom
            .n
            .checked_mul(geom.k_out)
            .and_then(|v| v.checked_mul(geom.oh))
            .and_then(|v| v.checked_mul(geom.ow))
            .ok_or_else(|| {
                BackendError::InvalidArgument(
                    "conv2d: output element count overflows usize".to_string(),
                )
            })?;
        if total == 0 {
            return Ok(());
        }
        let params = ConvParamsV2 {
            n_batch: to_u32(geom.n, "conv2d N")?,
            c_in: to_u32(geom.c_in, "conv2d C_in")?,
            h_in: to_u32(geom.h_in, "conv2d H_in")?,
            w_in: to_u32(geom.w_in, "conv2d W_in")?,
            k_out: to_u32(geom.k_out, "conv2d K_out")?,
            fh: to_u32(geom.fh, "conv2d filter height")?,
            fw: to_u32(geom.fw, "conv2d filter width")?,
            oh: to_u32(geom.oh, "conv2d output height")?,
            ow: to_u32(geom.ow, "conv2d output width")?,
            stride_h: to_u32(geom.stride_h, "conv2d stride_h")?,
            stride_w: to_u32(geom.stride_w, "conv2d stride_w")?,
            pad_h: to_u32(geom.pad_h, "conv2d pad_h")?,
            pad_w: to_u32(geom.pad_w, "conv2d pad_w")?,
        };
        // Guard the grid arithmetic as well: `total` indexes the kernel's `gid`.
        to_u32(total, "conv2d output element count")?;

        let pipeline = self.cached_pipeline(
            PipelineKey::Builtin {
                kind: "conv2d",
                op: "forward",
                dtype: "f32",
            },
            crate::msl::conv2d_v2_function_name(),
            || crate::msl::conv2d_msl_v2().to_string(),
        )?;
        let memory = self.memory()?;
        let [in_buf, flt_buf, out_buf] =
            resolve_buffers(memory, [input_ptr, filter_ptr, output_ptr])?;
        let plan = self.planner()?.plan_1d(total).map_err(BackendError::from)?;
        let tg_size = super::functions::clamp_1d_threadgroup(
            u64::from(plan.threads_per_threadgroup[0]),
            pipeline.pipeline_state.max_total_threads_per_threadgroup(),
            pipeline.pipeline_state.thread_execution_width(),
            total as u64,
        );
        let groups = (total as u64).div_ceil(tg_size);
        let command_buffer = self.device_queue()?.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&in_buf), 0);
        encoder.set_buffer(1, Some(&flt_buf), 0);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<ConvParamsV2>() as u64,
            &params as *const ConvParamsV2 as *const std::ffi::c_void,
        );
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        self.submit(command_buffer, "conv2d forward")
    }

    /// Number of SIMD-groups per threadgroup the attention kernel can run for
    /// `head_dim`, or `None` when not even one accumulator slice fits.
    ///
    /// Both limits are hard: the pipeline rejects a launch wider than
    /// `maxTotalThreadsPerThreadgroup`, and binding more threadgroup memory than
    /// the device reports is a validation failure. `head_dim` is caller-supplied
    /// and unbounded, so this is the check that keeps a large head from becoming
    /// a launch failure — or, worse, a silently truncated accumulator.
    ///
    /// Visible to the sibling test module so a numeric test can assert that the
    /// configuration it exercises really takes the GPU path: comparing GPU
    /// output against [`attention_host`] proves nothing if the dispatcher
    /// quietly fell back to that very function.
    pub(super) fn attention_simdgroups(
        &self,
        head_dim: usize,
        max_total_threads: u64,
    ) -> Option<usize> {
        let by_threads = (max_total_threads as usize) / SIMD_LANES;
        if by_threads == 0 {
            return None;
        }
        let budget = self.device.as_ref()?.capabilities().threadgroup_memory;
        let bytes_per_simdgroup = crate::msl_nn::attention_v2_threadgroup_bytes(head_dim, 1);
        if bytes_per_simdgroup == 0 || bytes_per_simdgroup > budget {
            return None;
        }
        let by_memory = budget / bytes_per_simdgroup;
        Some(by_threads.min(by_memory).clamp(1, ATTENTION_SIMDGROUPS))
    }

    /// Scaled dot-product attention on the GPU, one SIMD-group per query.
    ///
    /// Uses [`crate::msl_nn::attention_msl_v2`] (single-pass online softmax).
    /// Falls back to [`attention_host`] only when the device cannot host one
    /// `head_dim`-long accumulator in threadgroup memory.
    pub(super) fn dispatch_attention(
        &self,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        o_ptr: u64,
        geom: AttentionGeometry,
    ) -> BackendResult<()> {
        let queries = geom.batch_heads.checked_mul(geom.seq_q).ok_or_else(|| {
            BackendError::InvalidArgument("attention: batch_heads * seq_q overflows".to_string())
        })?;
        if queries == 0 {
            return Ok(());
        }
        let params = AttnParamsV2 {
            batch_heads: to_u32(geom.batch_heads, "attention batch_heads")?,
            seq_q: to_u32(geom.seq_q, "attention seq_q")?,
            seq_kv: to_u32(geom.seq_kv, "attention seq_kv")?,
            head_dim: to_u32(geom.head_dim, "attention head_dim")?,
            causal: u32::from(geom.causal),
            scale: geom.scale as f32,
        };
        to_u32(queries, "attention query count")?;

        // The online-softmax recurrence rescales by `exp(m_old - m_new)`, which
        // is only contracting under strict IEEE semantics.
        let pipeline = self.cached_pipeline(
            PipelineKey::Builtin {
                kind: "attention",
                op: "online_precise",
                dtype: "f32",
            },
            crate::msl_nn::attention_v2_function_name(),
            || crate::msl_nn::attention_msl_v2(crate::msl::MslMathMode::Precise),
        )?;
        let max_total_threads = pipeline.pipeline_state.max_total_threads_per_threadgroup();
        let Some(simdgroups) = self.attention_simdgroups(geom.head_dim, max_total_threads) else {
            tracing::debug!(
                head_dim = geom.head_dim,
                "attention head_dim exceeds the threadgroup-memory budget; using the host path"
            );
            return attention_host(self, q_ptr, k_ptr, v_ptr, o_ptr, geom);
        };
        let scratch_bytes =
            crate::msl_nn::attention_v2_threadgroup_bytes(geom.head_dim, simdgroups) as u64;
        let threads = (simdgroups * SIMD_LANES) as u64;
        let groups = (queries as u64).div_ceil(simdgroups as u64);

        let memory = self.memory()?;
        let [q_buf, k_buf, v_buf, o_buf] = resolve_buffers(memory, [q_ptr, k_ptr, v_ptr, o_ptr])?;
        let command_buffer = self.device_queue()?.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&q_buf), 0);
        encoder.set_buffer(1, Some(&k_buf), 0);
        encoder.set_buffer(2, Some(&v_buf), 0);
        encoder.set_buffer(3, Some(&o_buf), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<AttnParamsV2>() as u64,
            &params as *const AttnParamsV2 as *const std::ffi::c_void,
        );
        encoder.set_threadgroup_memory_length(0, scratch_bytes);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        self.submit(command_buffer, "attention")
    }

    /// Threadgroup width for a row-wise kernel covering `cols` elements, plus
    /// the threadgroup-memory bytes its scratch needs.
    fn rowwise_threadgroup(
        &self,
        cols: usize,
        max_total_threads: u64,
    ) -> BackendResult<(u64, u64)> {
        let desired = next_power_of_2(cols).min(ROWWISE_TARGET_THREADS) as u64;
        let tg_size = pow2_threadgroup(desired, max_total_threads.min(1024));
        let scratch = self
            .planner()?
            .threadgroup_scratch_bytes(tg_size as usize, std::mem::size_of::<f32>())
            .map_err(BackendError::from)? as u64;
        Ok((tg_size, scratch))
    }

    /// Row-wise numerically-stable softmax over a `rows × cols` row-major `f32`
    /// matrix, one threadgroup per row.
    pub(super) fn dispatch_softmax(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        rows: usize,
        cols: usize,
    ) -> BackendResult<()> {
        if rows == 0 {
            return Ok(());
        }
        let rows_u32 = to_u32(rows, "softmax rows")?;
        let cols_u32 = to_u32(cols, "softmax cols")?;
        // `Precise` is part of the cache key via `op`: the math mode is baked
        // into the generated source by `with_math_mode`.
        let pipeline = self.cached_pipeline(
            PipelineKey::Builtin {
                kind: "softmax",
                op: "rows_precise",
                dtype: "f32",
            },
            "softmax_rows_f32",
            || crate::msl_nn::softmax_msl_with_mode(crate::msl::MslMathMode::Precise),
        )?;
        let (tg_size, scratch) = self.rowwise_threadgroup(
            cols,
            pipeline.pipeline_state.max_total_threads_per_threadgroup(),
        )?;
        let memory = self.memory()?;
        let [in_buf, out_buf] = resolve_buffers(memory, [input_ptr, output_ptr])?;
        let command_buffer = self.device_queue()?.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&in_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_bytes(2, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(3, 4, &cols_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_threadgroup_memory_length(0, scratch);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(rows as u64, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        self.submit(command_buffer, "softmax")
    }

    /// Row-wise layer normalisation with an affine transform:
    /// `y = (x - mean) / sqrt(var + eps) * gamma + beta`, computed across the
    /// `cols` feature axis of a `rows × cols` row-major `f32` matrix.
    ///
    /// Inherent rather than a trait method: [`oxicuda_backend::ComputeBackend`]
    /// has no layer-norm slot, so there is nothing to override.
    ///
    /// `gamma` and `beta` are each `cols` elements long. All four handles may be
    /// [`alloc`](oxicuda_backend::ComputeBackend::alloc)-owned or imported.
    ///
    /// # Errors
    /// * [`BackendError::NotInitialized`] if the backend is not initialised.
    /// * [`BackendError::InvalidArgument`] for `cols == 0` (the mean divides by
    ///   it), a non-finite or negative `eps`, an unknown handle, or a dimension
    ///   that exceeds the `u32` range the kernel uses.
    /// * [`BackendError::DeviceError`] if the shader fails to compile or the GPU
    ///   reports a command-buffer failure.
    pub fn layer_norm(
        &self,
        input_ptr: u64,
        gamma_ptr: u64,
        beta_ptr: u64,
        output_ptr: u64,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> BackendResult<()> {
        self.check_init()?;
        if cols == 0 {
            return Err(BackendError::InvalidArgument(
                "layer_norm: cols must be non-zero".into(),
            ));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(BackendError::InvalidArgument(format!(
                "layer_norm: eps must be finite and non-negative, got {eps}"
            )));
        }
        if rows == 0 {
            return Ok(());
        }
        let rows_u32 = to_u32(rows, "layer_norm rows")?;
        let cols_u32 = to_u32(cols, "layer_norm cols")?;
        let pipeline = self.cached_pipeline(
            PipelineKey::Builtin {
                kind: "layernorm",
                op: "rows_precise",
                dtype: "f32",
            },
            "layernorm_rows_f32",
            || crate::msl_nn::layernorm_msl_with_mode(crate::msl::MslMathMode::Precise),
        )?;
        let (tg_size, scratch) = self.rowwise_threadgroup(
            cols,
            pipeline.pipeline_state.max_total_threads_per_threadgroup(),
        )?;
        let memory = self.memory()?;
        let [in_buf, gamma_buf, beta_buf, out_buf] =
            resolve_buffers(memory, [input_ptr, gamma_ptr, beta_ptr, output_ptr])?;
        let command_buffer = self.device_queue()?.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&in_buf), 0);
        encoder.set_buffer(1, Some(&gamma_buf), 0);
        encoder.set_buffer(2, Some(&beta_buf), 0);
        encoder.set_buffer(3, Some(&out_buf), 0);
        encoder.set_bytes(4, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &cols_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
        encoder.set_threadgroup_memory_length(0, scratch);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(rows as u64, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        self.submit(command_buffer, "layer_norm")
    }

    /// Single-threadgroup inclusive/exclusive prefix sum over `n` `f32`
    /// elements.
    ///
    /// Inherent rather than a trait method: [`oxicuda_backend::ComputeBackend`]
    /// has no scan slot.
    ///
    /// # Limits
    ///
    /// [`crate::msl_nn::scan_msl`] is a **single-threadgroup** Hillis-Steele
    /// scan with no cross-threadgroup carry, so `n` must not exceed the
    /// pipeline's `maxTotalThreadsPerThreadgroup` (1024 on current Apple GPUs).
    /// Larger inputs are rejected with [`BackendError::InvalidArgument`] rather
    /// than silently truncated — a multi-block scan needs the standard
    /// three-kernel decomposition, which is a separate feature.
    ///
    /// # Errors
    /// * [`BackendError::NotInitialized`] if the backend is not initialised.
    /// * [`BackendError::InvalidArgument`] for `n` beyond the single-threadgroup
    ///   limit or an unknown handle.
    /// * [`BackendError::DeviceError`] on shader-compilation or GPU failure.
    pub fn scan(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        n: usize,
        exclusive: bool,
    ) -> BackendResult<()> {
        self.check_init()?;
        if n == 0 {
            return Ok(());
        }
        let n_u32 = to_u32(n, "scan element count")?;
        let pipeline = self.cached_pipeline(
            PipelineKey::Builtin {
                kind: "scan",
                op: if exclusive { "exclusive" } else { "inclusive" },
                dtype: "f32",
            },
            crate::msl_nn::scan_function_name(exclusive),
            || crate::msl_nn::scan_msl(exclusive),
        )?;
        let max_threads = pipeline.pipeline_state.max_total_threads_per_threadgroup();
        if n as u64 > max_threads {
            return Err(BackendError::InvalidArgument(format!(
                "scan: n = {n} exceeds the single-threadgroup limit of {max_threads} elements; \
                 a multi-block scan is not implemented"
            )));
        }
        // The doubling loop runs to `tg_size`, so the width must **cover** `n` —
        // rounding down would silently truncate the scan. `n <= max_threads` was
        // just checked, so the smallest covering power of two always fits after
        // the cap. The ping-pong scratch is `2 * tg_size` floats, NOT `2 * n`.
        let mut tg_size = 1u64;
        while tg_size < n as u64 {
            tg_size <<= 1;
        }
        let tg_size = tg_size.min(max_threads);
        let scratch = self
            .planner()?
            .threadgroup_scratch_bytes(2 * tg_size as usize, std::mem::size_of::<f32>())
            .map_err(BackendError::from)? as u64;
        let memory = self.memory()?;
        let [in_buf, out_buf] = resolve_buffers(memory, [input_ptr, output_ptr])?;
        let command_buffer = self.device_queue()?.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&in_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_bytes(2, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_threadgroup_memory_length(0, scratch);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        self.submit(command_buffer, "scan")
    }
}

#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    pub(super) fn dispatch_conv2d(
        &self,
        _input_ptr: u64,
        _filter_ptr: u64,
        _output_ptr: u64,
        _geom: Conv2dGeometry,
    ) -> BackendResult<()> {
        Err(oxicuda_backend::BackendError::DeviceError(
            "Metal requires macOS".into(),
        ))
    }

    pub(super) fn dispatch_attention(
        &self,
        _q_ptr: u64,
        _k_ptr: u64,
        _v_ptr: u64,
        _o_ptr: u64,
        _geom: AttentionGeometry,
    ) -> BackendResult<()> {
        Err(oxicuda_backend::BackendError::DeviceError(
            "Metal requires macOS".into(),
        ))
    }

    pub(super) fn dispatch_softmax(
        &self,
        _input_ptr: u64,
        _output_ptr: u64,
        _rows: usize,
        _cols: usize,
    ) -> BackendResult<()> {
        Err(oxicuda_backend::BackendError::DeviceError(
            "Metal requires macOS".into(),
        ))
    }

    /// Row-wise layer normalisation. Present on every platform so callers need
    /// no `cfg` gate; off macOS there is no Metal, so this always fails.
    ///
    /// # Errors
    /// Always [`oxicuda_backend::BackendError::DeviceError`] on non-macOS
    /// targets (or `NotInitialized`, which is itself always the case there).
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &self,
        _input_ptr: u64,
        _gamma_ptr: u64,
        _beta_ptr: u64,
        _output_ptr: u64,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> BackendResult<()> {
        self.check_init()?;
        Err(oxicuda_backend::BackendError::DeviceError(
            "Metal requires macOS".into(),
        ))
    }

    /// Single-threadgroup prefix sum. Present on every platform so callers need
    /// no `cfg` gate; off macOS there is no Metal, so this always fails.
    ///
    /// # Errors
    /// Always [`oxicuda_backend::BackendError::DeviceError`] on non-macOS
    /// targets (or `NotInitialized`).
    pub fn scan(
        &self,
        _input_ptr: u64,
        _output_ptr: u64,
        _n: usize,
        _exclusive: bool,
    ) -> BackendResult<()> {
        self.check_init()?;
        Err(oxicuda_backend::BackendError::DeviceError(
            "Metal requires macOS".into(),
        ))
    }
}

/// Host reference implementation of scaled dot-product attention.
///
/// This is the loop `MetalBackend::attention` used to *be*. It survives as the
/// fallback for the one configuration [`crate::msl_nn::attention_msl_v2`]
/// cannot express — a `head_dim` whose accumulator does not fit the device's
/// threadgroup-memory budget — and as the oracle the GPU path is tested against.
///
/// Causal masking is **top-left** aligned (`masked ⟺ sk > sq`), matching the GPU
/// kernel, `oxicuda_backend`'s CPU backend and [`crate::msl::attention_msl`].
///
/// Off macOS nothing dispatches at all, so this fallback is never reached.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn attention_host(
    backend: &MetalBackend,
    q_ptr: u64,
    k_ptr: u64,
    v_ptr: u64,
    o_ptr: u64,
    geom: AttentionGeometry,
) -> BackendResult<()> {
    use oxicuda_backend::ComputeBackend;

    use super::functions::{read_f32_le, write_f32_le};

    let AttentionGeometry {
        batch_heads,
        seq_q,
        seq_kv,
        head_dim,
        scale,
        causal,
    } = geom;
    let q_len = batch_heads * seq_q * head_dim;
    let kv_len = batch_heads * seq_kv * head_dim;
    let mut q_bytes = vec![0u8; q_len * 4];
    let mut k_bytes = vec![0u8; kv_len * 4];
    let mut v_bytes = vec![0u8; kv_len * 4];
    backend.copy_dtoh(&mut q_bytes, q_ptr)?;
    backend.copy_dtoh(&mut k_bytes, k_ptr)?;
    backend.copy_dtoh(&mut v_bytes, v_ptr)?;
    let q = read_f32_le(&q_bytes);
    let k = read_f32_le(&k_bytes);
    let v = read_f32_le(&v_bytes);
    let mut o = vec![0.0f32; q_len];
    let scale_f = scale as f32;
    // Reusable per-(bh, sq) score buffer: the scaled Q·Kᵀ dot products are
    // computed once in the max pass and reused in the accumulate pass instead of
    // recomputing the O(head_dim) inner product a second time.
    let mut scores = vec![0.0f32; seq_kv];
    let mut acc = vec![0.0f32; head_dim];
    for bh in 0..batch_heads {
        for sq in 0..seq_q {
            let q_off = (bh * seq_q + sq) * head_dim;
            let mut max_score = f32::NEG_INFINITY;
            for (sk, score_slot) in scores.iter_mut().enumerate() {
                if causal && sk > sq {
                    continue;
                }
                let k_off = (bh * seq_kv + sk) * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                let score = dot * scale_f;
                *score_slot = score;
                if score > max_score {
                    max_score = score;
                }
            }
            let mut sum_exp = 0.0f32;
            acc.fill(0.0);
            for (sk, &score) in scores.iter().enumerate() {
                if causal && sk > sq {
                    continue;
                }
                let w = (score - max_score).exp();
                sum_exp += w;
                let v_off = (bh * seq_kv + sk) * head_dim;
                for d in 0..head_dim {
                    acc[d] += w * v[v_off + d];
                }
            }
            let o_off = (bh * seq_q + sq) * head_dim;
            if sum_exp > 0.0 {
                for d in 0..head_dim {
                    o[o_off + d] = acc[d] / sum_exp;
                }
            }
        }
    }
    let o_bytes = write_f32_le(&o);
    backend.copy_htod(o_ptr, &o_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_params_v2_matches_the_msl_struct_size() {
        // The kernel reads a `constant ConvParamsV2&`; a Rust struct of a
        // different size would bind garbage into the trailing fields.
        assert_eq!(
            std::mem::size_of::<ConvParamsV2>(),
            crate::msl::CONV_PARAMS_V2_BYTES
        );
        assert_eq!(std::mem::align_of::<ConvParamsV2>(), 4);
    }

    #[test]
    fn attn_params_v2_matches_the_msl_struct_size() {
        assert_eq!(
            std::mem::size_of::<AttnParamsV2>(),
            crate::msl_nn::ATTN_PARAMS_V2_BYTES
        );
        assert_eq!(std::mem::align_of::<AttnParamsV2>(), 4);
    }

    /// The attention dispatcher must never pick a SIMD-group count whose
    /// accumulator exceeds the threadgroup-memory budget, and must report
    /// "impossible" rather than clamping to something that would truncate.
    #[test]
    fn attention_v2_threadgroup_bytes_scales_with_both_axes() {
        assert_eq!(crate::msl_nn::attention_v2_threadgroup_bytes(64, 1), 256);
        assert_eq!(crate::msl_nn::attention_v2_threadgroup_bytes(64, 4), 1024);
        assert_eq!(crate::msl_nn::attention_v2_threadgroup_bytes(0, 4), 0);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod device_tests {
    use super::*;
    use oxicuda_backend::ComputeBackend;

    fn try_init() -> Option<MetalBackend> {
        let mut b = MetalBackend::new();
        b.init().ok().map(|()| b)
    }

    /// A `head_dim` far beyond any device's threadgroup-memory budget must be
    /// reported as "no SIMD-group count works" so the caller routes to the host
    /// path, rather than being clamped into a launch that truncates the
    /// accumulator.
    #[test]
    fn oversized_head_dim_has_no_viable_simdgroup_count() {
        let Some(b) = try_init() else { return };
        // 1 Mi floats = 4 MiB per accumulator: larger than any Apple GPU's
        // 32 KiB threadgroup-memory budget.
        assert_eq!(b.attention_simdgroups(1 << 20, 1024), None);
    }

    /// A realistic head dimension must yield a count that satisfies *both*
    /// budgets simultaneously.
    #[test]
    fn simdgroup_count_respects_thread_and_memory_budgets() {
        let Some(b) = try_init() else { return };
        let Some(device) = b.device.as_ref() else {
            return;
        };
        let budget = device.capabilities().threadgroup_memory;
        for head_dim in [1usize, 8, 64, 128, 512] {
            let Some(s) = b.attention_simdgroups(head_dim, 1024) else {
                continue;
            };
            assert!(
                (1..=ATTENTION_SIMDGROUPS).contains(&s),
                "head_dim={head_dim}"
            );
            assert!(s * SIMD_LANES <= 1024, "head_dim={head_dim}");
            assert!(
                crate::msl_nn::attention_v2_threadgroup_bytes(head_dim, s) <= budget,
                "head_dim={head_dim} s={s} exceeds the {budget}-byte budget"
            );
        }
        // A narrow pipeline limit must shrink the count, never the memory bound.
        assert_eq!(b.attention_simdgroups(64, 32), Some(1));
        assert_eq!(b.attention_simdgroups(64, 31), None);
    }
}
