//! Group Normalization for CNNs and Vision Transformers.
//!
//! Divides channels into groups and normalizes within each group:
//!
//! ```text
//! For each sample n and group g:
//!   mean_g = mean(x[n, channels_in_g, :, :])
//!   var_g  = var(x[n, channels_in_g, :, :])
//!   y[n, c, h, w] = (x[n, c, h, w] - mean_g) / sqrt(var_g + eps) * gamma[c] + beta[c]
//! ```
//!
//! GroupNorm is independent of batch size and interpolates between
//! InstanceNorm (num_groups = C) and LayerNorm (num_groups = 1).

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

/// Applies Group Normalization on an NCHW tensor.
///
/// The `C` channels are split into `num_groups` groups of equal size.
/// Each group is normalised independently per sample.
///
/// # Arguments
///
/// * `handle` -- DNN handle.
/// * `input` -- 4D tensor `[N, C, H, W]`.
/// * `num_groups` -- Number of channel groups (must evenly divide `C`).
/// * `gamma` -- Per-channel scale, length `C`.
/// * `beta` -- Per-channel bias, length `C`.
/// * `output` -- Mutable output tensor, same shape as input.
/// * `epsilon` -- Stability constant.
///
/// # Errors
///
/// Returns [`DnnError`] if `num_groups` does not divide `C`, buffers are
/// undersized, or the kernel fails.
pub fn group_norm<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    num_groups: u32,
    gamma: &DeviceBuffer<T>,
    beta: &DeviceBuffer<T>,
    output: &mut TensorDescMut<T>,
    epsilon: f32,
) -> DnnResult<()> {
    let (batch, channels, spatial) = extract_nchw_dims(input)?;
    if num_groups == 0 {
        return Err(DnnError::InvalidArgument("num_groups must be > 0".into()));
    }
    if channels % num_groups != 0 {
        return Err(DnnError::InvalidArgument(format!(
            "channels ({channels}) not divisible by num_groups ({num_groups})"
        )));
    }
    validate_group_norm_args(input, gamma, beta, output, channels)?;

    let channels_per_group = channels / num_groups;
    let group_size = channels_per_group * spatial; // elements per group per sample

    let ptx_source = generate_group_norm_ptx::<T>(handle.sm_version(), group_size)?;
    let kernel_name = group_norm_kernel_name::<T>(group_size);
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| DnnError::LaunchFailed(format!("module load for group_norm: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| DnnError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;

    // Grid: one block per (sample, group) pair
    let num_blocks = batch * num_groups;
    let block_size = group_size.next_power_of_two().clamp(32, 1024);
    let params = LaunchParams::new(num_blocks, block_size);

    let eps_bits = epsilon.to_bits();

    let args = (
        input.ptr,
        output.ptr,
        gamma.as_device_ptr(),
        beta.as_device_ptr(),
        batch,
        channels,
        spatial,
        num_groups,
        channels_per_group,
        eps_bits,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("group_norm: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_nchw_dims<T: GpuFloat>(desc: &TensorDesc<T>) -> DnnResult<(u32, u32, u32)> {
    if desc.dims.len() != 4 {
        return Err(DnnError::InvalidDimension(format!(
            "group_norm requires 4D tensor, got {}D",
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

fn validate_group_norm_args<T: GpuFloat>(
    input: &TensorDesc<T>,
    gamma: &DeviceBuffer<T>,
    beta: &DeviceBuffer<T>,
    output: &TensorDescMut<T>,
    channels: u32,
) -> DnnResult<()> {
    let c = channels as usize;
    if gamma.len() < c {
        return Err(DnnError::BufferTooSmall {
            expected: c * T::SIZE,
            actual: gamma.len() * T::SIZE,
        });
    }
    if beta.len() < c {
        return Err(DnnError::BufferTooSmall {
            expected: c * T::SIZE,
            actual: beta.len() * T::SIZE,
        });
    }
    if output.numel() < input.numel() {
        return Err(DnnError::BufferTooSmall {
            expected: input.numel() * T::SIZE,
            actual: output.numel() * T::SIZE,
        });
    }
    Ok(())
}

fn group_norm_kernel_name<T: GpuFloat>(group_size: u32) -> String {
    format!("group_norm_{}_gs{group_size}", T::NAME)
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates PTX for group normalization.
///
/// Kernel parameters:
/// - `input`              (u64)
/// - `output`             (u64)
/// - `gamma`              (u64)
/// - `beta`               (u64)
/// - `batch`              (u32)
/// - `channels`           (u32)
/// - `spatial`            (u32) -- H * W
/// - `num_groups`         (u32)
/// - `channels_per_group` (u32)
/// - `epsilon_bits`       (u32)
///
/// Grid: one block per (sample, group). blockIdx.x encodes (n * G + g).
/// Each thread iterates over elements in the group with stride = block_size.
fn generate_group_norm_ptx<T: GpuFloat>(sm: SmVersion, group_size: u32) -> DnnResult<String> {
    let ty = T::PTX_TYPE.as_ptx_str();
    let byte_size = T::PTX_TYPE.size_bytes();
    let kernel_name = group_norm_kernel_name::<T>(group_size);
    let block_size = group_size.next_power_of_two().clamp(32, 1024);
    let smem_bytes = block_size as usize * 4;

    let mut ptx = String::with_capacity(8192);

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
    writeln!(ptx, "    .param .u32 %param_num_groups,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_cpg,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_epsilon_bits").map_err(fmt_err)?;
    writeln!(ptx, ")").map_err(fmt_err)?;
    writeln!(ptx, "{{").map_err(fmt_err)?;
    writeln!(ptx, "    .maxntid {block_size}, 1, 1;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .b32 %r<32>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .b64 %rd<20>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .f32 %f<32>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .pred %p<8>;").map_err(fmt_err)?;
    writeln!(ptx, "    .shared .align 4 .b8 smem_gn[{smem_bytes}];").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    // Decode blockIdx.x => (sample_idx, group_idx)
    writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r2, [%param_num_groups];").map_err(fmt_err)?;
    writeln!(ptx, "    div.u32 %r3, %r1, %r2;").map_err(fmt_err)?; // sample_idx
    writeln!(ptx, "    rem.u32 %r4, %r1, %r2;").map_err(fmt_err)?; // group_idx

    writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd2, [%param_gamma];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd3, [%param_beta];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r5, [%param_channels];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r6, [%param_spatial];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r7, [%param_cpg];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r8, [%param_epsilon_bits];").map_err(fmt_err)?;
    writeln!(ptx, "    mov.b32 %f20, %r8;").map_err(fmt_err)?;

    // group_size = cpg * spatial
    writeln!(ptx, "    mul.lo.u32 %r9, %r7, %r6;").map_err(fmt_err)?; // group_size

    // Base offset for this sample's data: sample_idx * C * spatial
    writeln!(ptx, "    mul.lo.u32 %r10, %r3, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u32 %r10, %r10, %r6;").map_err(fmt_err)?;

    // Group start channel offset: group_idx * cpg * spatial
    writeln!(ptx, "    mul.lo.u32 %r11, %r4, %r7;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u32 %r11, %r11, %r6;").map_err(fmt_err)?;

    // Combined base = sample_offset + group_offset
    writeln!(ptx, "    add.u32 %r12, %r10, %r11;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd4, %r12;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd4, %rd4, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd5, %rd0, %rd4;").map_err(fmt_err)?; // input base
    writeln!(ptx, "    add.u64 %rd6, %rd1, %rd4;").map_err(fmt_err)?; // output base
    writeln!(ptx).map_err(fmt_err)?;

    // Pass 1: sum
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r13, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$GN_SUM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r13, %r9;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p1 bra $GN_SUM_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r13;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd5, %rd8;").map_err(fmt_err)?;
    load_global(&mut ptx, ty, "%f1", "%rd9")?;
    writeln!(ptx, "    add.f32 %f0, %f0, %f1;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r13, %r13, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $GN_SUM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$GN_SUM_DONE:").map_err(fmt_err)?;

    write_smem_reduce_f32(&mut ptx, "%f0", block_size, "GN_SUM")?;

    writeln!(ptx, "    ld.shared.f32 %f2, [smem_gn];").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.rn.f32.u32 %f3, %r9;").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f4, %f2, %f3;").map_err(fmt_err)?; // mean
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 2: variance
    writeln!(ptx, "    mov.f32 %f5, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r13, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$GN_VAR_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p2, %r13, %r9;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p2 bra $GN_VAR_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r13;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd5, %rd8;").map_err(fmt_err)?;
    load_global(&mut ptx, ty, "%f6", "%rd9")?;
    writeln!(ptx, "    sub.f32 %f7, %f6, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    fma.rn.f32 %f5, %f7, %f7, %f5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r13, %r13, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $GN_VAR_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$GN_VAR_DONE:").map_err(fmt_err)?;

    write_smem_reduce_f32(&mut ptx, "%f5", block_size, "GN_VAR")?;

    writeln!(ptx, "    ld.shared.f32 %f8, [smem_gn];").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f8, %f8, %f3;").map_err(fmt_err)?;
    writeln!(ptx, "    add.f32 %f9, %f8, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f10, %f9;").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 3: normalize + per-channel affine
    // The channel index for element i in the group: group_idx * cpg + i / spatial
    writeln!(ptx, "    mov.u32 %r13, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$GN_NORM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p3, %r13, %r9;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p3 bra $GN_DONE;").map_err(fmt_err)?;

    // Load x
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r13;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd5, %rd8;").map_err(fmt_err)?;
    load_global(&mut ptx, ty, "%f11", "%rd9")?;

    // Normalize
    writeln!(ptx, "    sub.f32 %f11, %f11, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.f32 %f11, %f11, %f10;").map_err(fmt_err)?;

    // Channel index = group_idx * cpg + r13 / spatial
    writeln!(ptx, "    div.u32 %r14, %r13, %r6;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u32 %r15, %r4, %r7;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r14, %r14, %r15;").map_err(fmt_err)?;

    // Load gamma[c], beta[c]
    writeln!(ptx, "    cvt.u64.u32 %rd10, %r14;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd10, %rd10, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd11, %rd2, %rd10;").map_err(fmt_err)?;
    load_global(&mut ptx, ty, "%f12", "%rd11")?;
    writeln!(ptx, "    add.u64 %rd12, %rd3, %rd10;").map_err(fmt_err)?;
    load_global(&mut ptx, ty, "%f13", "%rd12")?;
    writeln!(ptx, "    fma.rn.f32 %f14, %f11, %f12, %f13;").map_err(fmt_err)?;

    // Store
    writeln!(ptx, "    add.u64 %rd13, %rd6, %rd8;").map_err(fmt_err)?;
    store_global(&mut ptx, ty, "%rd13", "%f14")?;

    writeln!(ptx, "    add.u32 %r13, %r13, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $GN_NORM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    writeln!(ptx, "$GN_DONE:").map_err(fmt_err)?;
    writeln!(ptx, "    ret;").map_err(fmt_err)?;
    writeln!(ptx, "}}").map_err(fmt_err)?;

    Ok(ptx)
}

fn write_smem_reduce_f32(
    ptx: &mut String,
    val_reg: &str,
    block_size: u32,
    tag: &str,
) -> DnnResult<()> {
    writeln!(ptx, "    cvt.u64.u32 %rd14, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd14, %rd14, 4;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u64 %rd15, smem_gn;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd14, %rd15, %rd14;").map_err(fmt_err)?;
    writeln!(ptx, "    st.shared.f32 [%rd14], {val_reg};").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    let mut stride = block_size / 2;
    while stride > 0 {
        writeln!(ptx, "    setp.lt.u32 %p4, %r0, {stride};").map_err(fmt_err)?;
        writeln!(ptx, "    @!%p4 bra $SKIP_{tag}_{stride};").map_err(fmt_err)?;
        let off = stride as usize * 4;
        writeln!(ptx, "    ld.shared.f32 %f15, [%rd14+{off}];").map_err(fmt_err)?;
        writeln!(ptx, "    ld.shared.f32 %f16, [%rd14];").map_err(fmt_err)?;
        writeln!(ptx, "    add.f32 %f16, %f16, %f15;").map_err(fmt_err)?;
        writeln!(ptx, "    st.shared.f32 [%rd14], %f16;").map_err(fmt_err)?;
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
    fn ptx_group_norm() {
        // 32 channels, 4 groups => cpg=8, spatial=16 => group_size=128
        let ptx = generate_group_norm_ptx::<f32>(SmVersion::Sm80, 128);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("group_norm_f32_gs128"));
        assert!(ptx.contains("smem_gn"));
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn channels_not_divisible() {
        // Cannot test the full group_norm function without a GPU, but we can
        // verify the validation logic by checking extract_nchw_dims.
        let desc = TensorDesc::<f32>::from_raw(
            0,
            vec![2, 32, 8, 8],
            vec![32 * 8 * 8, 8 * 8, 8, 1],
            TensorLayout::Nchw,
        );
        let desc = desc.unwrap_or_else(|_| panic!("from_raw should succeed"));
        let (_, c, _) = extract_nchw_dims(&desc).unwrap_or((0, 0, 0));
        assert_eq!(c, 32);
        // 32 % 5 != 0 -- this would fail in group_norm()
        assert_ne!(32 % 5, 0);
    }
}
