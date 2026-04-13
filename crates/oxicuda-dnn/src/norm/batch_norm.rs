//! Batch Normalization for convolutional neural networks.
//!
//! Implements per-channel normalization across the `(N, H, W)` dimensions
//! of an NCHW tensor:
//!
//! ```text
//! Training:
//!   mean_c = mean(x[:, c, :, :])
//!   var_c  = var(x[:, c, :, :])
//!   y[:, c, :, :] = (x[:, c, :, :] - mean_c) / sqrt(var_c + eps) * gamma_c + beta_c
//!   running_mean = (1 - momentum) * running_mean + momentum * mean_c
//!   running_var  = (1 - momentum) * running_var  + momentum * var_c
//!
//! Inference:
//!   y[:, c, :, :] = (x[:, c, :, :] - running_mean_c) / sqrt(running_var_c + eps) * gamma_c + beta_c
//! ```

use std::fmt::Write as FmtWrite;
use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
#[cfg(test)]
use crate::types::TensorLayout;
use crate::types::{TensorDesc, TensorDescMut};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Applies Batch Normalization on an NCHW tensor.
///
/// In training mode, computes batch mean and variance per channel across
/// `N * H * W` elements, normalizes, applies affine transform, and updates
/// running statistics. In inference mode, uses the pre-computed running
/// mean and variance.
///
/// # Arguments
///
/// * `handle` -- DNN handle.
/// * `input` -- 4D tensor `[N, C, H, W]`.
/// * `gamma` -- Per-channel scale, length `C`.
/// * `beta` -- Per-channel bias, length `C`.
/// * `running_mean` -- Running mean buffer, length `C` (updated in training).
/// * `running_var` -- Running variance buffer, length `C` (updated in training).
/// * `output` -- Mutable output tensor, same shape as input.
/// * `epsilon` -- Stability constant (typically 1e-5).
/// * `momentum` -- EMA coefficient for running stats (typically 0.1).
/// * `training` -- If `true`, compute batch stats; otherwise use running stats.
/// * `save_mean` -- Optional buffer to store batch mean (training only).
/// * `save_invvar` -- Optional buffer to store inverse std-dev (training only).
///
/// # Errors
///
/// Returns [`DnnError`] on dimension/buffer validation failures or kernel
/// launch errors.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_forward<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    gamma: &DeviceBuffer<T>,
    beta: &DeviceBuffer<T>,
    running_mean: &mut DeviceBuffer<T>,
    running_var: &mut DeviceBuffer<T>,
    output: &mut TensorDescMut<T>,
    epsilon: f32,
    momentum: f32,
    training: bool,
    save_mean: Option<&mut DeviceBuffer<T>>,
    save_invvar: Option<&mut DeviceBuffer<T>>,
) -> DnnResult<()> {
    let (batch, channels, spatial) = extract_nchw_dims(input)?;
    validate_batch_norm_args(
        input,
        gamma,
        beta,
        running_mean,
        running_var,
        output,
        channels,
    )?;

    let ptx_source = generate_batch_norm_ptx::<T>(handle.sm_version(), spatial, training)?;
    let kernel_name = batch_norm_kernel_name::<T>(spatial, training);
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| DnnError::LaunchFailed(format!("module load for batch_norm: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| DnnError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;

    // One block per channel. Thread count = min(spatial * batch, 1024) rounded
    // up to next power of two, capped at 1024.
    let nhw = (batch as u64) * (spatial as u64);
    let block_size = (nhw as u32).next_power_of_two().clamp(32, 1024);
    let params = LaunchParams::new(channels, block_size);

    let eps_bits = epsilon.to_bits();
    let mom_bits = momentum.to_bits();

    let save_mean_ptr = save_mean.map(|b| b.as_device_ptr()).unwrap_or(0);
    let save_invvar_ptr = save_invvar.map(|b| b.as_device_ptr()).unwrap_or(0);

    let args = (
        input.ptr,
        output.ptr,
        gamma.as_device_ptr(),
        beta.as_device_ptr(),
        running_mean.as_device_ptr(),
        running_var.as_device_ptr(),
        batch,
        channels,
        spatial,
        eps_bits,
        mom_bits,
        save_mean_ptr,
        save_invvar_ptr,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("batch_norm: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extracts (N, C, H*W) from a 4D tensor descriptor.
fn extract_nchw_dims<T: GpuFloat>(desc: &TensorDesc<T>) -> DnnResult<(u32, u32, u32)> {
    if desc.dims.len() != 4 {
        return Err(DnnError::InvalidDimension(format!(
            "batch_norm requires 4D tensor, got {}D",
            desc.dims.len()
        )));
    }
    let n = desc.dims[0];
    let c = desc.dims[1];
    let h = desc.dims[2];
    let w = desc.dims[3];
    if n == 0 || c == 0 || h == 0 || w == 0 {
        return Err(DnnError::InvalidDimension(
            "all dimensions must be non-zero".into(),
        ));
    }
    Ok((n, c, h * w))
}

#[allow(clippy::too_many_arguments)]
fn validate_batch_norm_args<T: GpuFloat>(
    input: &TensorDesc<T>,
    gamma: &DeviceBuffer<T>,
    beta: &DeviceBuffer<T>,
    running_mean: &DeviceBuffer<T>,
    running_var: &DeviceBuffer<T>,
    output: &TensorDescMut<T>,
    channels: u32,
) -> DnnResult<()> {
    let c = channels as usize;
    for (_name, buf) in [
        ("gamma", gamma),
        ("beta", beta),
        ("running_mean", running_mean as &DeviceBuffer<T>),
        ("running_var", running_var as &DeviceBuffer<T>),
    ] {
        if buf.len() < c {
            return Err(DnnError::BufferTooSmall {
                expected: c * T::SIZE,
                actual: buf.len() * T::SIZE,
            });
        }
    }
    if output.numel() < input.numel() {
        return Err(DnnError::BufferTooSmall {
            expected: input.numel() * T::SIZE,
            actual: output.numel() * T::SIZE,
        });
    }
    Ok(())
}

fn batch_norm_kernel_name<T: GpuFloat>(spatial: u32, training: bool) -> String {
    let mode = if training { "train" } else { "infer" };
    format!("batch_norm_{mode}_{}_s{spatial}", T::NAME)
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates PTX for batch normalization.
///
/// Kernel parameters:
/// - `input`          (u64) -- input tensor ptr
/// - `output`         (u64) -- output tensor ptr
/// - `gamma`          (u64) -- scale per channel
/// - `beta`           (u64) -- bias per channel
/// - `running_mean`   (u64) -- running mean (read/write)
/// - `running_var`    (u64) -- running var (read/write)
/// - `batch`          (u32) -- N
/// - `channels`       (u32) -- C
/// - `spatial`        (u32) -- H * W
/// - `epsilon_bits`   (u32) -- eps as f32 bits
/// - `momentum_bits`  (u32) -- momentum as f32 bits
/// - `save_mean`      (u64) -- optional save mean ptr (0 if unused)
/// - `save_invvar`    (u64) -- optional save invvar ptr (0 if unused)
///
/// Grid: one block per channel (blockIdx.x = channel index).
/// Block: up to 1024 threads; each accumulates over N*HW with strided loop.
fn generate_batch_norm_ptx<T: GpuFloat>(
    sm: SmVersion,
    spatial: u32,
    training: bool,
) -> DnnResult<String> {
    let ty = T::PTX_TYPE.as_ptx_str();
    let byte_size = T::PTX_TYPE.size_bytes();
    let kernel_name = batch_norm_kernel_name::<T>(spatial, training);
    let block_size = {
        let nhw_est = (spatial as u64) * 32; // approximate
        (nhw_est as u32).next_power_of_two().clamp(32, 1024)
    };
    let smem_bytes = block_size as usize * 4; // f32 accumulator

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
    writeln!(ptx, "    .param .u64 %param_running_mean,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_running_var,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_batch,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_channels,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_spatial,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_epsilon_bits,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_momentum_bits,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_save_mean,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_save_invvar").map_err(fmt_err)?;
    writeln!(ptx, ")").map_err(fmt_err)?;
    writeln!(ptx, "{{").map_err(fmt_err)?;
    writeln!(ptx, "    .maxntid {block_size}, 1, 1;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .b32 %r<32>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .b64 %rd<24>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .f32 %f<32>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .pred %p<8>;").map_err(fmt_err)?;
    writeln!(ptx, "    .shared .align 4 .b8 smem_bn[{smem_bytes}];").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    // Channel index = blockIdx.x
    writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(fmt_err)?; // channel
    writeln!(ptx, "    ld.param.u32 %r2, [%param_channels];").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p0, %r1, %r2;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p0 bra $BN_DONE;").map_err(fmt_err)?;

    // Load params
    writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd2, [%param_gamma];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd3, [%param_beta];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd4, [%param_running_mean];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd5, [%param_running_var];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r3, [%param_batch];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r4, [%param_spatial];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r5, [%param_epsilon_bits];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r6, [%param_momentum_bits];").map_err(fmt_err)?;
    writeln!(ptx, "    mov.b32 %f20, %r5;").map_err(fmt_err)?; // epsilon
    writeln!(ptx, "    mov.b32 %f21, %r6;").map_err(fmt_err)?; // momentum
    writeln!(ptx).map_err(fmt_err)?;

    // total_elems = N * spatial (per channel)
    writeln!(ptx, "    mul.lo.u32 %r7, %r3, %r4;").map_err(fmt_err)?; // total_elems

    if training {
        write_bn_training(&mut ptx, ty, byte_size, block_size)?;
    } else {
        write_bn_inference(&mut ptx, ty, byte_size, block_size)?;
    }

    writeln!(ptx, "$BN_DONE:").map_err(fmt_err)?;
    writeln!(ptx, "    ret;").map_err(fmt_err)?;
    writeln!(ptx, "}}").map_err(fmt_err)?;

    Ok(ptx)
}

/// Training mode: compute batch stats, normalize, update running stats.
fn write_bn_training(
    ptx: &mut String,
    ty: &str,
    byte_size: usize,
    block_size: u32,
) -> DnnResult<()> {
    writeln!(ptx, "    // BatchNorm training mode").map_err(fmt_err)?;

    // Pass 1: accumulate sum for mean
    // For NCHW layout, elements for channel c are at:
    //   input[n * C * HW + c * HW + hw]
    // We iterate over n and hw with strided access.
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?; // partial sum
    writeln!(ptx, "    mov.u32 %r8, %r0;").map_err(fmt_err)?; // linear idx in [0, N*HW)
    writeln!(ptx, "$BN_SUM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r8, %r7;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p1 bra $BN_SUM_DONE;").map_err(fmt_err)?;

    // Decompose r8 into (n_idx, hw_idx): n_idx = r8 / spatial, hw_idx = r8 % spatial
    // Global offset = n_idx * C * spatial + channel * spatial + hw_idx
    writeln!(ptx, "    div.u32 %r9, %r8, %r4;").map_err(fmt_err)?; // n_idx
    writeln!(ptx, "    rem.u32 %r10, %r8, %r4;").map_err(fmt_err)?; // hw_idx
    writeln!(ptx, "    mul.lo.u32 %r11, %r9, %r2;").map_err(fmt_err)?; // n * C
    writeln!(ptx, "    add.u32 %r11, %r11, %r1;").map_err(fmt_err)?; // + c
    writeln!(ptx, "    mul.lo.u32 %r11, %r11, %r4;").map_err(fmt_err)?; // * spatial
    writeln!(ptx, "    add.u32 %r11, %r11, %r10;").map_err(fmt_err)?; // + hw

    writeln!(ptx, "    cvt.u64.u32 %rd8, %r11;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f1", "%rd9")?;
    writeln!(ptx, "    add.f32 %f0, %f0, %f1;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r8, %r8, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $BN_SUM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$BN_SUM_DONE:").map_err(fmt_err)?;

    // Reduce sum via shared memory
    write_smem_reduce_f32(ptx, "%f0", block_size, "BN_SUM")?;

    // mean = sum / total_elems
    writeln!(ptx, "    ld.shared.f32 %f2, [smem_bn];").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.rn.f32.u32 %f3, %r7;").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f4, %f2, %f3;").map_err(fmt_err)?; // mean
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 2: accumulate (x - mean)^2 for variance
    writeln!(ptx, "    mov.f32 %f5, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r8, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$BN_VAR_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p2, %r8, %r7;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p2 bra $BN_VAR_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    div.u32 %r9, %r8, %r4;").map_err(fmt_err)?;
    writeln!(ptx, "    rem.u32 %r10, %r8, %r4;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u32 %r11, %r9, %r2;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r11, %r11, %r1;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u32 %r11, %r11, %r4;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r11, %r11, %r10;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r11;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f6", "%rd9")?;
    writeln!(ptx, "    sub.f32 %f7, %f6, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    fma.rn.f32 %f5, %f7, %f7, %f5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r8, %r8, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $BN_VAR_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$BN_VAR_DONE:").map_err(fmt_err)?;

    write_smem_reduce_f32(ptx, "%f5", block_size, "BN_VAR")?;

    writeln!(ptx, "    ld.shared.f32 %f8, [smem_bn];").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f8, %f8, %f3;").map_err(fmt_err)?; // variance
    writeln!(ptx, "    add.f32 %f9, %f8, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f10, %f9;").map_err(fmt_err)?; // inv_std
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Thread 0: update running stats + save mean/invvar
    writeln!(ptx, "    setp.eq.u32 %p3, %r0, 0;").map_err(fmt_err)?;
    writeln!(ptx, "    @!%p3 bra $BN_SKIP_STATS;").map_err(fmt_err)?;

    // running_mean = (1 - momentum) * running_mean + momentum * mean
    writeln!(ptx, "    cvt.u64.u32 %rd10, %r1;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd10, %rd10, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd11, %rd4, %rd10;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f11", "%rd11")?;
    writeln!(ptx, "    mov.f32 %f12, 0f3F800000;").map_err(fmt_err)?; // 1.0
    writeln!(ptx, "    sub.f32 %f13, %f12, %f21;").map_err(fmt_err)?; // 1 - mom
    writeln!(ptx, "    mul.f32 %f11, %f11, %f13;").map_err(fmt_err)?;
    writeln!(ptx, "    fma.rn.f32 %f11, %f21, %f4, %f11;").map_err(fmt_err)?;
    store_global(ptx, ty, "%rd11", "%f11")?;

    // running_var = (1 - momentum) * running_var + momentum * var
    writeln!(ptx, "    add.u64 %rd12, %rd5, %rd10;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f14", "%rd12")?;
    writeln!(ptx, "    mul.f32 %f14, %f14, %f13;").map_err(fmt_err)?;
    writeln!(ptx, "    fma.rn.f32 %f14, %f21, %f8, %f14;").map_err(fmt_err)?;
    store_global(ptx, ty, "%rd12", "%f14")?;

    // Optionally save mean / invvar
    writeln!(ptx, "    ld.param.u64 %rd13, [%param_save_mean];").map_err(fmt_err)?;
    writeln!(ptx, "    setp.eq.u64 %p4, %rd13, 0;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p4 bra $BN_SKIP_SAVE_MEAN;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd14, %rd13, %rd10;").map_err(fmt_err)?;
    store_global(ptx, ty, "%rd14", "%f4")?;
    writeln!(ptx, "$BN_SKIP_SAVE_MEAN:").map_err(fmt_err)?;

    writeln!(ptx, "    ld.param.u64 %rd15, [%param_save_invvar];").map_err(fmt_err)?;
    writeln!(ptx, "    setp.eq.u64 %p5, %rd15, 0;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p5 bra $BN_SKIP_SAVE_INVVAR;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd16, %rd15, %rd10;").map_err(fmt_err)?;
    store_global(ptx, ty, "%rd16", "%f10")?;
    writeln!(ptx, "$BN_SKIP_SAVE_INVVAR:").map_err(fmt_err)?;
    writeln!(ptx, "$BN_SKIP_STATS:").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 3: normalize + affine transform
    write_bn_normalize_pass(ptx, ty, byte_size, block_size)?;

    Ok(())
}

/// Inference mode: use running statistics.
fn write_bn_inference(
    ptx: &mut String,
    ty: &str,
    byte_size: usize,
    block_size: u32,
) -> DnnResult<()> {
    writeln!(ptx, "    // BatchNorm inference mode").map_err(fmt_err)?;

    // Load running_mean[c] and running_var[c]
    writeln!(ptx, "    cvt.u64.u32 %rd10, %r1;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd10, %rd10, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd11, %rd4, %rd10;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f4", "%rd11")?; // mean
    writeln!(ptx, "    add.u64 %rd12, %rd5, %rd10;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f8", "%rd12")?; // var
    writeln!(ptx, "    add.f32 %f9, %f8, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f10, %f9;").map_err(fmt_err)?; // inv_std

    write_bn_normalize_pass(ptx, ty, byte_size, block_size)?;

    Ok(())
}

/// Common normalize + affine pass (used by both training and inference).
///
/// Expects `%f4` = mean, `%f10` = inv_std, channel index in `%r1`,
/// total elements in `%r7`.
fn write_bn_normalize_pass(
    ptx: &mut String,
    ty: &str,
    byte_size: usize,
    block_size: u32,
) -> DnnResult<()> {
    // Load gamma[c] and beta[c]
    writeln!(ptx, "    cvt.u64.u32 %rd17, %r1;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd17, %rd17, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd18, %rd2, %rd17;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f22", "%rd18")?; // gamma
    writeln!(ptx, "    add.u64 %rd19, %rd3, %rd17;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f23", "%rd19")?; // beta

    writeln!(ptx, "    mov.u32 %r8, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$BN_NORM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p6, %r8, %r7;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p6 bra $BN_DONE;").map_err(fmt_err)?;

    writeln!(ptx, "    div.u32 %r9, %r8, %r4;").map_err(fmt_err)?;
    writeln!(ptx, "    rem.u32 %r10, %r8, %r4;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u32 %r11, %r9, %r2;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r11, %r11, %r1;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u32 %r11, %r11, %r4;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r11, %r11, %r10;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r11;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f24", "%rd9")?;

    // y = (x - mean) * inv_std * gamma + beta
    writeln!(ptx, "    sub.f32 %f24, %f24, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.f32 %f24, %f24, %f10;").map_err(fmt_err)?;
    writeln!(ptx, "    fma.rn.f32 %f24, %f24, %f22, %f23;").map_err(fmt_err)?;

    writeln!(ptx, "    add.u64 %rd20, %rd1, %rd8;").map_err(fmt_err)?;
    store_global(ptx, ty, "%rd20", "%f24")?;

    writeln!(ptx, "    add.u32 %r8, %r8, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $BN_NORM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    Ok(())
}

/// Shared memory tree reduction (f32).
fn write_smem_reduce_f32(
    ptx: &mut String,
    val_reg: &str,
    block_size: u32,
    tag: &str,
) -> DnnResult<()> {
    writeln!(ptx, "    cvt.u64.u32 %rd6, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd6, %rd6, 4;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u64 %rd7, smem_bn;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd6, %rd7, %rd6;").map_err(fmt_err)?;
    writeln!(ptx, "    st.shared.f32 [%rd6], {val_reg};").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    let mut stride = block_size / 2;
    while stride > 0 {
        writeln!(ptx, "    setp.lt.u32 %p7, %r0, {stride};").map_err(fmt_err)?;
        writeln!(ptx, "    @!%p7 bra $SKIP_{tag}_{stride};").map_err(fmt_err)?;
        let off = stride as usize * 4;
        writeln!(ptx, "    ld.shared.f32 %f15, [%rd6+{off}];").map_err(fmt_err)?;
        writeln!(ptx, "    ld.shared.f32 %f16, [%rd6];").map_err(fmt_err)?;
        writeln!(ptx, "    add.f32 %f16, %f16, %f15;").map_err(fmt_err)?;
        writeln!(ptx, "    st.shared.f32 [%rd6], %f16;").map_err(fmt_err)?;
        writeln!(ptx, "$SKIP_{tag}_{stride}:").map_err(fmt_err)?;
        writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;
        stride /= 2;
    }

    Ok(())
}

fn load_global(ptx: &mut String, ty: &str, dst: &str, addr: &str) -> DnnResult<()> {
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 {dst}, [{addr}];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} {dst}, [{addr}];").map_err(fmt_err)?;
    }
    Ok(())
}

fn store_global(ptx: &mut String, ty: &str, addr: &str, src: &str) -> DnnResult<()> {
    if ty == ".f32" {
        writeln!(ptx, "    st.global.f32 [{addr}], {src};").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    st.global{ty} [{addr}], {src};").map_err(fmt_err)?;
    }
    Ok(())
}

fn fmt_err(e: std::fmt::Error) -> DnnError {
    DnnError::PtxGeneration(format!("PTX format error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_bn_training() {
        let ptx = generate_batch_norm_ptx::<f32>(SmVersion::Sm80, 64, true);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("batch_norm_train_f32"));
        assert!(ptx.contains("smem_bn"));
        assert!(ptx.contains("%param_running_mean"));
        assert!(ptx.contains("%param_save_mean"));
    }

    #[test]
    fn ptx_bn_inference() {
        let ptx = generate_batch_norm_ptx::<f32>(SmVersion::Sm80, 64, false);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("batch_norm_infer_f32"));
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn extract_dims_valid() {
        let desc = TensorDesc::<f32>::from_raw(
            0,
            vec![2, 64, 8, 8],
            vec![64 * 8 * 8, 8 * 8, 8, 1],
            TensorLayout::Nchw,
        );
        let desc = desc.unwrap_or_else(|_| panic!("from_raw should succeed"));
        let (n, c, hw) = extract_nchw_dims(&desc).unwrap_or((0, 0, 0));
        assert_eq!((n, c, hw), (2, 64, 64));
    }

    #[test]
    fn extract_dims_wrong_ndim() {
        let desc = TensorDesc::<f32>::from_raw(0, vec![2, 64], vec![64, 1], TensorLayout::Nchw);
        let desc = desc.unwrap_or_else(|_| panic!("from_raw should succeed"));
        assert!(extract_nchw_dims(&desc).is_err());
    }
}
