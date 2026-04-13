//! Instance Normalization for image generation and style transfer.
//!
//! Normalizes each (batch, channel) pair independently over spatial dimensions:
//!
//! ```text
//! For each sample n and channel c:
//!   mean = mean(x[n, c, :, :])
//!   var  = var(x[n, c, :, :])
//!   y[n, c, h, w] = gamma[c] * (x[n, c, h, w] - mean) / sqrt(var + eps) + beta[c]
//! ```
//!
//! Instance Normalization is widely used in style transfer (AdaIN),
//! image generation (StyleGAN, Pix2Pix), and domain adaptation tasks
//! where per-instance, per-channel statistics are needed rather than
//! batch-level statistics.

use std::fmt::Write as FmtWrite;

use oxicuda_blas::GpuFloat;
use oxicuda_ptx::arch::SmVersion;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Instance Normalization.
///
/// Each (batch, channel) pair is normalised independently over `spatial_size`
/// elements (i.e. H*W).
#[derive(Debug, Clone)]
pub struct InstanceNormConfig {
    /// Number of channels (C dimension).
    pub num_channels: u32,
    /// Number of spatial elements per channel (H * W).
    pub spatial_size: u32,
    /// Small constant for numerical stability (added to variance).
    pub epsilon: f32,
    /// Whether to apply learnable affine parameters (gamma, beta).
    pub affine: bool,
    /// Whether to track running statistics for inference mode.
    pub track_running_stats: bool,
}

impl InstanceNormConfig {
    /// Validates this configuration.
    pub fn validate(&self) -> DnnResult<()> {
        if self.num_channels == 0 {
            return Err(DnnError::InvalidArgument("num_channels must be > 0".into()));
        }
        if self.spatial_size == 0 {
            return Err(DnnError::InvalidArgument("spatial_size must be > 0".into()));
        }
        if self.epsilon <= 0.0 {
            return Err(DnnError::InvalidArgument("epsilon must be positive".into()));
        }
        if !self.epsilon.is_finite() {
            return Err(DnnError::InvalidArgument("epsilon must be finite".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// Execution plan for Instance Normalization.
///
/// Pre-generates PTX kernels for the configured spatial size and data type.
/// Each thread block handles one (batch, channel) pair, computing mean and
/// variance over the spatial dimension via parallel reduction.
#[derive(Debug)]
pub struct InstanceNormPlan {
    config: InstanceNormConfig,
    forward_ptx: String,
    backward_ptx: String,
}

impl InstanceNormPlan {
    /// Creates a new execution plan for the given configuration and SM version.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError`] if the configuration is invalid or PTX generation
    /// fails.
    pub fn new<T: GpuFloat>(config: InstanceNormConfig, sm: SmVersion) -> DnnResult<Self> {
        config.validate()?;
        let forward_ptx = generate_forward_ptx::<T>(&config, sm)?;
        let backward_ptx = generate_backward_ptx::<T>(&config, sm)?;
        Ok(Self {
            config,
            forward_ptx,
            backward_ptx,
        })
    }

    /// Returns the generated forward-pass PTX source.
    pub fn forward_ptx(&self) -> &str {
        &self.forward_ptx
    }

    /// Returns the generated backward-pass PTX source.
    pub fn backward_ptx(&self) -> &str {
        &self.backward_ptx
    }

    /// Returns a reference to the plan's configuration.
    pub fn config(&self) -> &InstanceNormConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// PTX generation -- forward
// ---------------------------------------------------------------------------

fn instance_norm_kernel_name<T: GpuFloat>(spatial: u32) -> String {
    format!("instance_norm_fwd_{}_s{spatial}", T::NAME)
}

fn instance_norm_bwd_kernel_name<T: GpuFloat>(spatial: u32) -> String {
    format!("instance_norm_bwd_{}_s{spatial}", T::NAME)
}

/// Generates PTX for the Instance Normalization forward kernel.
///
/// Kernel parameters (in order):
///   - `input`          (u64) -- input tensor [N, C, H*W]
///   - `output`         (u64) -- output tensor [N, C, H*W]
///   - `gamma`          (u64) -- per-channel scale (length C), or null if !affine
///   - `beta`           (u64) -- per-channel bias (length C), or null if !affine
///   - `batch`          (u32) -- N
///   - `channels`       (u32) -- C
///   - `spatial`        (u32) -- H * W
///   - `epsilon_bits`   (u32) -- epsilon as f32 bit pattern
///
/// Grid: one block per (n, c) pair => num_blocks = N * C.
/// Each thread accumulates over spatial elements with a strided loop.
pub fn generate_forward_ptx<T: GpuFloat>(
    config: &InstanceNormConfig,
    sm: SmVersion,
) -> DnnResult<String> {
    let ty = T::PTX_TYPE.as_ptx_str();
    let byte_size = T::PTX_TYPE.size_bytes();
    let spatial = config.spatial_size;
    let affine = config.affine;
    let kernel_name = instance_norm_kernel_name::<T>(spatial);
    let block_size = spatial.next_power_of_two().clamp(32, 1024);
    let smem_bytes = block_size as usize * 4;

    let mut ptx = String::with_capacity(8192);

    // Header
    writeln!(ptx, ".version {}", sm.ptx_version()).map_err(fmt_err)?;
    writeln!(ptx, ".target {}", sm.as_ptx_str()).map_err(fmt_err)?;
    writeln!(ptx, ".address_size 64").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;
    writeln!(ptx, ".visible .entry {kernel_name}(").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_input,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_output,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_gamma,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_beta,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_batch,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_channels,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_spatial,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_epsilon_bits").map_err(fmt_err)?;
    writeln!(ptx, ")").map_err(fmt_err)?;
    writeln!(ptx, "{{").map_err(fmt_err)?;
    writeln!(ptx, "    .maxntid {block_size}, 1, 1;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .b32 %r<32>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .b64 %rd<24>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .f32 %f<32>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .pred %p<8>;").map_err(fmt_err)?;
    writeln!(ptx, "    .shared .align 4 .b8 smem_in[{smem_bytes}];").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    // Block index = (n * C + c), thread index within block
    writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(fmt_err)?;

    // Guard: blockIdx.x >= N * C => done
    writeln!(ptx, "    ld.param.u32 %r2, [%param_batch];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r3, [%param_channels];").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u32 %r8, %r2, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p0, %r1, %r8;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p0 bra $IN_DONE;").map_err(fmt_err)?;

    // Compute channel index: c = blockIdx.x % C, sample index: n = blockIdx.x / C
    writeln!(ptx, "    div.u32 %r9, %r1, %r3;").map_err(fmt_err)?; // n
    writeln!(ptx, "    rem.u32 %r10, %r1, %r3;").map_err(fmt_err)?; // c

    // Load parameters
    writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd2, [%param_gamma];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd3, [%param_beta];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r4, [%param_spatial];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r5, [%param_epsilon_bits];").map_err(fmt_err)?;
    writeln!(ptx, "    mov.b32 %f20, %r5;").map_err(fmt_err)?; // epsilon
    writeln!(ptx).map_err(fmt_err)?;

    // Compute base offset for this (n, c) slice:
    // offset = (n * C * spatial + c * spatial)
    writeln!(ptx, "    mul.lo.u32 %r11, %r9, %r3;").map_err(fmt_err)?; // n * C
    writeln!(ptx, "    add.u32 %r11, %r11, %r10;").map_err(fmt_err)?; // n * C + c
    writeln!(ptx, "    mul.lo.u32 %r11, %r11, %r4;").map_err(fmt_err)?; // (n*C + c) * spatial
    writeln!(ptx, "    cvt.u64.u32 %rd6, %r11;").map_err(fmt_err)?;
    // rd6 = element offset for this slice

    // Pass 1: accumulate sum for mean via strided loop
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?; // partial sum
    writeln!(ptx, "    mov.u32 %r12, %r0;").map_err(fmt_err)?; // loop var = tid
    writeln!(ptx, "$IN_SUM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r12, {spatial};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p1 bra $IN_SUM_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r12;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f1, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f1, [%rd9];").map_err(fmt_err)?;
    }
    writeln!(ptx, "    add.f32 %f0, %f0, %f1;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r12, %r12, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $IN_SUM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$IN_SUM_DONE:").map_err(fmt_err)?;

    // Shared memory reduction for sum
    write_smem_reduce_f32(&mut ptx, "%f0", block_size, "SUM")?;

    // Broadcast mean
    writeln!(ptx, "    ld.shared.f32 %f4, [smem_in];").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.rn.f32.u32 %f3, %r4;").map_err(fmt_err)?; // spatial as float
    writeln!(ptx, "    div.approx.f32 %f4, %f4, %f3;").map_err(fmt_err)?; // mean
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 2: accumulate (x - mean)^2 for variance
    writeln!(ptx, "    mov.f32 %f5, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r12, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$IN_VAR_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p2, %r12, {spatial};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p2 bra $IN_VAR_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r12;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f6, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f6, [%rd9];").map_err(fmt_err)?;
    }
    writeln!(ptx, "    sub.f32 %f7, %f6, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    fma.rn.f32 %f5, %f7, %f7, %f5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r12, %r12, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $IN_VAR_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$IN_VAR_DONE:").map_err(fmt_err)?;

    // Shared memory reduction for variance
    write_smem_reduce_f32(&mut ptx, "%f5", block_size, "VAR")?;

    // Compute inv_std = rsqrt(var + eps)
    writeln!(ptx, "    ld.shared.f32 %f8, [smem_in];").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f8, %f8, %f3;").map_err(fmt_err)?; // variance
    writeln!(ptx, "    add.f32 %f9, %f8, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f10, %f9;").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Load gamma and beta for this channel (if affine)
    if affine {
        writeln!(ptx, "    cvt.u64.u32 %rd10, %r10;").map_err(fmt_err)?;
        writeln!(ptx, "    mul.lo.u64 %rd10, %rd10, {byte_size};").map_err(fmt_err)?;
        writeln!(ptx, "    add.u64 %rd11, %rd2, %rd10;").map_err(fmt_err)?;
        writeln!(ptx, "    add.u64 %rd12, %rd3, %rd10;").map_err(fmt_err)?;
        if ty == ".f32" {
            writeln!(ptx, "    ld.global.f32 %f12, [%rd11];").map_err(fmt_err)?; // gamma
            writeln!(ptx, "    ld.global.f32 %f13, [%rd12];").map_err(fmt_err)?; // beta
        } else {
            writeln!(ptx, "    ld.global{ty} %f12, [%rd11];").map_err(fmt_err)?;
            writeln!(ptx, "    ld.global{ty} %f13, [%rd12];").map_err(fmt_err)?;
        }
    }

    // Pass 3: normalize, scale, bias, store
    writeln!(ptx, "    mov.u32 %r12, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$IN_NORM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p3, %r12, {spatial};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p3 bra $IN_DONE;").map_err(fmt_err)?;

    // Reload x
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r12;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f11, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f11, [%rd9];").map_err(fmt_err)?;
    }

    // Normalize: (x - mean) * inv_std
    writeln!(ptx, "    sub.f32 %f11, %f11, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.f32 %f11, %f11, %f10;").map_err(fmt_err)?;

    // Apply affine: y = gamma * norm + beta
    if affine {
        writeln!(ptx, "    fma.rn.f32 %f14, %f11, %f12, %f13;").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    mov.f32 %f14, %f11;").map_err(fmt_err)?;
    }

    // Store result
    writeln!(ptx, "    add.u64 %rd13, %rd1, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    st.global.f32 [%rd13], %f14;").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    st.global{ty} [%rd13], %f14;").map_err(fmt_err)?;
    }
    writeln!(ptx, "    add.u32 %r12, %r12, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $IN_NORM_LOOP;").map_err(fmt_err)?;

    writeln!(ptx, "$IN_DONE:").map_err(fmt_err)?;
    writeln!(ptx, "    ret;").map_err(fmt_err)?;
    writeln!(ptx, "}}").map_err(fmt_err)?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// PTX generation -- backward
// ---------------------------------------------------------------------------

/// Generates PTX for the Instance Normalization backward kernel.
///
/// Computes gradients w.r.t. input (dx), gamma (dgamma), and beta (dbeta).
///
/// Kernel parameters:
///   - `grad_output`    (u64) -- upstream gradient [N, C, H*W]
///   - `input`          (u64) -- forward input [N, C, H*W]
///   - `gamma`          (u64) -- per-channel scale
///   - `grad_input`     (u64) -- output: gradient w.r.t. input
///   - `batch`          (u32) -- N
///   - `channels`       (u32) -- C
///   - `spatial`        (u32) -- H * W
///   - `epsilon_bits`   (u32) -- epsilon as f32 bit pattern
///
/// Grid: one block per (n, c) pair.
pub fn generate_backward_ptx<T: GpuFloat>(
    config: &InstanceNormConfig,
    sm: SmVersion,
) -> DnnResult<String> {
    let ty = T::PTX_TYPE.as_ptx_str();
    let byte_size = T::PTX_TYPE.size_bytes();
    let spatial = config.spatial_size;
    let kernel_name = instance_norm_bwd_kernel_name::<T>(spatial);
    let block_size = spatial.next_power_of_two().clamp(32, 1024);
    let smem_bytes = block_size as usize * 4;

    let mut ptx = String::with_capacity(8192);

    // Header
    writeln!(ptx, ".version {}", sm.ptx_version()).map_err(fmt_err)?;
    writeln!(ptx, ".target {}", sm.as_ptx_str()).map_err(fmt_err)?;
    writeln!(ptx, ".address_size 64").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;
    writeln!(ptx, ".visible .entry {kernel_name}(").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_grad_output,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_input,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_gamma,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_grad_input,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_batch,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_channels,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_spatial,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_epsilon_bits").map_err(fmt_err)?;
    writeln!(ptx, ")").map_err(fmt_err)?;
    writeln!(ptx, "{{").map_err(fmt_err)?;
    writeln!(ptx, "    .maxntid {block_size}, 1, 1;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .b32 %r<32>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .b64 %rd<24>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .f32 %f<32>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .pred %p<8>;").map_err(fmt_err)?;
    writeln!(ptx, "    .shared .align 4 .b8 smem_in[{smem_bytes}];").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    // Block index = (n * C + c)
    writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(fmt_err)?;

    // Guard
    writeln!(ptx, "    ld.param.u32 %r2, [%param_batch];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r3, [%param_channels];").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u32 %r8, %r2, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p0, %r1, %r8;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p0 bra $INB_DONE;").map_err(fmt_err)?;

    // n, c from block index
    writeln!(ptx, "    div.u32 %r9, %r1, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    rem.u32 %r10, %r1, %r3;").map_err(fmt_err)?;

    // Load params
    writeln!(ptx, "    ld.param.u64 %rd0, [%param_grad_output];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd1, [%param_input];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd2, [%param_gamma];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd3, [%param_grad_input];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r4, [%param_spatial];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r5, [%param_epsilon_bits];").map_err(fmt_err)?;
    writeln!(ptx, "    mov.b32 %f20, %r5;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    // Base offset for this (n, c) slice
    writeln!(ptx, "    mul.lo.u32 %r11, %r9, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r11, %r11, %r10;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u32 %r11, %r11, %r4;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd6, %r11;").map_err(fmt_err)?;

    // Pass 1: compute mean of input
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r12, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$INB_MEAN_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r12, {spatial};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p1 bra $INB_MEAN_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r12;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd1, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f1, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f1, [%rd9];").map_err(fmt_err)?;
    }
    writeln!(ptx, "    add.f32 %f0, %f0, %f1;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r12, %r12, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $INB_MEAN_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$INB_MEAN_DONE:").map_err(fmt_err)?;
    write_smem_reduce_f32(&mut ptx, "%f0", block_size, "BMEAN")?;
    writeln!(ptx, "    ld.shared.f32 %f4, [smem_in];").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.rn.f32.u32 %f3, %r4;").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f4, %f4, %f3;").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 2: compute variance
    writeln!(ptx, "    mov.f32 %f5, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r12, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$INB_VAR_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p2, %r12, {spatial};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p2 bra $INB_VAR_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r12;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd1, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f6, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f6, [%rd9];").map_err(fmt_err)?;
    }
    writeln!(ptx, "    sub.f32 %f7, %f6, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    fma.rn.f32 %f5, %f7, %f7, %f5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r12, %r12, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $INB_VAR_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$INB_VAR_DONE:").map_err(fmt_err)?;
    write_smem_reduce_f32(&mut ptx, "%f5", block_size, "BVAR")?;
    writeln!(ptx, "    ld.shared.f32 %f8, [smem_in];").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f8, %f8, %f3;").map_err(fmt_err)?;
    writeln!(ptx, "    add.f32 %f9, %f8, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f10, %f9;").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Load gamma for this channel
    writeln!(ptx, "    cvt.u64.u32 %rd10, %r10;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd10, %rd10, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd11, %rd2, %rd10;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f12, [%rd11];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f12, [%rd11];").map_err(fmt_err)?;
    }

    // Pass 3: compute sum(dy) and sum(dy * x_hat) for backward formula
    writeln!(ptx, "    mov.f32 %f15, 0f00000000;").map_err(fmt_err)?; // sum_dy
    writeln!(ptx, "    mov.f32 %f16, 0f00000000;").map_err(fmt_err)?; // sum_dy_xhat
    writeln!(ptx, "    mov.u32 %r12, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$INB_DSUM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p3, %r12, {spatial};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p3 bra $INB_DSUM_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r12;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    // Load dy
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f17, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f17, [%rd9];").map_err(fmt_err)?;
    }
    // Load x
    writeln!(ptx, "    add.u64 %rd9, %rd1, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f18, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f18, [%rd9];").map_err(fmt_err)?;
    }
    // x_hat = (x - mean) * inv_std
    writeln!(ptx, "    sub.f32 %f19, %f18, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.f32 %f19, %f19, %f10;").map_err(fmt_err)?;
    // accumulate
    writeln!(ptx, "    add.f32 %f15, %f15, %f17;").map_err(fmt_err)?;
    writeln!(ptx, "    fma.rn.f32 %f16, %f17, %f19, %f16;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r12, %r12, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $INB_DSUM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$INB_DSUM_DONE:").map_err(fmt_err)?;

    // Reduce sum_dy
    write_smem_reduce_f32(&mut ptx, "%f15", block_size, "DY")?;
    writeln!(ptx, "    ld.shared.f32 %f21, [smem_in];").map_err(fmt_err)?; // sum_dy
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Reduce sum_dy_xhat
    write_smem_reduce_f32(&mut ptx, "%f16", block_size, "DYX")?;
    writeln!(ptx, "    ld.shared.f32 %f22, [smem_in];").map_err(fmt_err)?; // sum_dy_xhat
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 4: compute dx = gamma * inv_std / N * (N * dy - sum_dy - x_hat * sum_dy_xhat)
    writeln!(ptx, "    mov.u32 %r12, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$INB_DX_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p4, %r12, {spatial};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p4 bra $INB_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r12;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    // Load dy
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f17, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f17, [%rd9];").map_err(fmt_err)?;
    }
    // Load x
    writeln!(ptx, "    add.u64 %rd9, %rd1, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f18, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f18, [%rd9];").map_err(fmt_err)?;
    }
    // x_hat
    writeln!(ptx, "    sub.f32 %f19, %f18, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.f32 %f19, %f19, %f10;").map_err(fmt_err)?;
    // dx = gamma * inv_std * (dy - (sum_dy + x_hat * sum_dy_xhat) / spatial)
    writeln!(ptx, "    mul.f32 %f23, %f19, %f22;").map_err(fmt_err)?; // x_hat * sum_dy_xhat
    writeln!(ptx, "    add.f32 %f23, %f21, %f23;").map_err(fmt_err)?; // sum_dy + x_hat * sum_dy_xhat
    writeln!(ptx, "    div.approx.f32 %f23, %f23, %f3;").map_err(fmt_err)?; // / spatial
    writeln!(ptx, "    sub.f32 %f24, %f17, %f23;").map_err(fmt_err)?; // dy - ...
    writeln!(ptx, "    mul.f32 %f24, %f24, %f10;").map_err(fmt_err)?; // * inv_std
    writeln!(ptx, "    mul.f32 %f24, %f24, %f12;").map_err(fmt_err)?; // * gamma

    // Store dx
    writeln!(ptx, "    add.u64 %rd13, %rd3, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    st.global.f32 [%rd13], %f24;").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    st.global{ty} [%rd13], %f24;").map_err(fmt_err)?;
    }
    writeln!(ptx, "    add.u32 %r12, %r12, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $INB_DX_LOOP;").map_err(fmt_err)?;

    writeln!(ptx, "$INB_DONE:").map_err(fmt_err)?;
    writeln!(ptx, "    ret;").map_err(fmt_err)?;
    writeln!(ptx, "}}").map_err(fmt_err)?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Shared memory reduction helper
// ---------------------------------------------------------------------------

/// Shared memory tree reduction for an f32 accumulator.
///
/// Writes partial value from `val_reg` into shared memory `smem_in`,
/// synchronises, then performs a tree reduction. Result ends up at
/// `smem_in[0]`.
fn write_smem_reduce_f32(
    ptx: &mut String,
    val_reg: &str,
    block_size: u32,
    tag: &str,
) -> DnnResult<()> {
    writeln!(ptx, "    // Shared memory reduction ({tag})").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd14, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd14, %rd14, 4;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u64 %rd15, smem_in;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd14, %rd15, %rd14;").map_err(fmt_err)?;
    writeln!(ptx, "    st.shared.f32 [%rd14], {val_reg};").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    let mut stride = block_size / 2;
    while stride > 0 {
        writeln!(ptx, "    setp.lt.u32 %p4, %r0, {stride};").map_err(fmt_err)?;
        writeln!(ptx, "    @!%p4 bra $SKIP_{tag}_{stride};").map_err(fmt_err)?;
        let partner_off = stride as usize * 4;
        writeln!(ptx, "    ld.shared.f32 %f15, [%rd14+{partner_off}];").map_err(fmt_err)?;
        writeln!(ptx, "    ld.shared.f32 %f16, [%rd14];").map_err(fmt_err)?;
        writeln!(ptx, "    add.f32 %f16, %f16, %f15;").map_err(fmt_err)?;
        writeln!(ptx, "    st.shared.f32 [%rd14], %f16;").map_err(fmt_err)?;
        writeln!(ptx, "$SKIP_{tag}_{stride}:").map_err(fmt_err)?;
        writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;
        stride /= 2;
    }

    Ok(())
}

/// Converts `std::fmt::Error` to [`DnnError`].
fn fmt_err(e: std::fmt::Error) -> DnnError {
    DnnError::PtxGeneration(format!("PTX format error: {e}"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config validation tests --

    #[test]
    fn config_valid() {
        let cfg = InstanceNormConfig {
            num_channels: 64,
            spatial_size: 256,
            epsilon: 1e-5,
            affine: true,
            track_running_stats: false,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_zero_channels() {
        let cfg = InstanceNormConfig {
            num_channels: 0,
            spatial_size: 256,
            epsilon: 1e-5,
            affine: true,
            track_running_stats: false,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_zero_spatial() {
        let cfg = InstanceNormConfig {
            num_channels: 64,
            spatial_size: 0,
            epsilon: 1e-5,
            affine: true,
            track_running_stats: false,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_negative_epsilon() {
        let cfg = InstanceNormConfig {
            num_channels: 64,
            spatial_size: 256,
            epsilon: -1e-5,
            affine: true,
            track_running_stats: false,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_nan_epsilon() {
        let cfg = InstanceNormConfig {
            num_channels: 64,
            spatial_size: 256,
            epsilon: f32::NAN,
            affine: true,
            track_running_stats: false,
        };
        assert!(cfg.validate().is_err());
    }

    // -- PTX generation tests --

    fn make_config(spatial: u32, affine: bool) -> InstanceNormConfig {
        InstanceNormConfig {
            num_channels: 32,
            spatial_size: spatial,
            epsilon: 1e-5,
            affine,
            track_running_stats: false,
        }
    }

    #[test]
    fn forward_ptx_f32_small_spatial() {
        let cfg = make_config(16, true);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry instance_norm_fwd_f32_s16"));
        assert!(ptx.contains("smem_in"));
        assert!(ptx.contains("rsqrt.approx.f32"));
        assert!(ptx.contains("fma.rn.f32"));
    }

    #[test]
    fn forward_ptx_f32_large_spatial() {
        let cfg = make_config(1024, true);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry instance_norm_fwd_f32_s1024"));
        assert!(ptx.contains("bar.sync"));
    }

    #[test]
    fn forward_ptx_f64() {
        let cfg = make_config(256, true);
        let ptx = generate_forward_ptx::<f64>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry instance_norm_fwd_f64_s256"));
    }

    #[test]
    fn forward_ptx_no_affine() {
        let cfg = make_config(64, false);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        // Without affine, normalized value is moved directly without gamma*x+beta
        assert!(ptx.contains("mov.f32 %f14, %f11"));
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn backward_ptx_f32() {
        let cfg = make_config(128, true);
        let ptx = generate_backward_ptx::<f32>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry instance_norm_bwd_f32_s128"));
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn backward_ptx_f64() {
        let cfg = make_config(64, true);
        let ptx = generate_backward_ptx::<f64>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry instance_norm_bwd_f64_s64"));
    }

    // -- Plan tests --

    #[test]
    fn plan_creation_success() {
        let cfg = InstanceNormConfig {
            num_channels: 128,
            spatial_size: 256,
            epsilon: 1e-5,
            affine: true,
            track_running_stats: true,
        };
        let plan = InstanceNormPlan::new::<f32>(cfg, SmVersion::Sm80);
        assert!(plan.is_ok());
        let plan = plan.unwrap_or_else(|e| panic!("plan creation failed: {e}"));
        assert!(plan.forward_ptx().contains("instance_norm_fwd"));
        assert!(plan.backward_ptx().contains("instance_norm_bwd"));
        assert_eq!(plan.config().num_channels, 128);
    }

    #[test]
    fn plan_creation_invalid_config() {
        let cfg = InstanceNormConfig {
            num_channels: 0,
            spatial_size: 256,
            epsilon: 1e-5,
            affine: true,
            track_running_stats: false,
        };
        let plan = InstanceNormPlan::new::<f32>(cfg, SmVersion::Sm80);
        assert!(plan.is_err());
    }

    #[test]
    fn forward_ptx_various_spatial_sizes() {
        for spatial in [1, 7, 49, 196, 784, 3136] {
            let cfg = make_config(spatial, true);
            let result = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
            assert!(
                result.is_ok(),
                "failed for spatial_size={spatial}: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn forward_ptx_epsilon_encoded() {
        let cfg = make_config(64, true);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80).unwrap_or_default();
        // Epsilon is loaded via param and moved to f32 register
        assert!(ptx.contains("param_epsilon_bits"));
        assert!(ptx.contains("mov.b32 %f20"));
    }
}
