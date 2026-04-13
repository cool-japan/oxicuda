//! RMS Normalization for LLM models (LLaMA, Gemma, etc.).
//!
//! RMSNorm omits the mean-subtraction step of LayerNorm and uses the
//! root-mean-square of the input instead:
//!
//! ```text
//! y = x / sqrt(mean(x^2) + epsilon) * gamma
//! ```
//!
//! This module also provides a fused variant that first adds a residual
//! tensor to the input before normalizing, which is a common pattern in
//! LLM pre-norm architectures.

use std::fmt::Write as FmtWrite;
use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::types::{TensorDesc, TensorDescMut};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Applies RMS Normalization across the last dimension.
///
/// ```text
/// rms = sqrt(mean(x^2) + epsilon)
/// y = x / rms * gamma
/// ```
///
/// # Arguments
///
/// * `handle` -- DNN handle bound to a CUDA context and stream.
/// * `input` -- Input tensor (row-major, last dim = hidden).
/// * `gamma` -- Per-element scale (length = hidden dim).
/// * `output` -- Mutable output tensor (same shape as input).
/// * `epsilon` -- Small constant for numerical stability.
///
/// # Errors
///
/// Returns [`DnnError`] on dimension mismatch, buffer undersize, or PTX
/// generation / launch failure.
pub fn rms_norm<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    gamma: &DeviceBuffer<T>,
    output: &mut TensorDescMut<T>,
    epsilon: f32,
) -> DnnResult<()> {
    let (num_rows, hidden_dim) = extract_row_dims(input)?;
    validate_rms_args(input, gamma, output, hidden_dim)?;

    let ptx_source = generate_rms_norm_ptx::<T>(handle.sm_version(), hidden_dim, false)?;
    let kernel_name = rms_norm_kernel_name::<T>(hidden_dim, false);
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| DnnError::LaunchFailed(format!("module load for rms_norm: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| DnnError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;

    let (grid, block) = launch_config(num_rows, hidden_dim);
    let params = LaunchParams::new(grid, block);
    let eps_bits = epsilon.to_bits();

    // For non-fused variant, residual pointer is 0 (unused)
    let args = (
        input.ptr,
        output.ptr,
        gamma.as_device_ptr(),
        0u64, // residual_ptr (unused)
        num_rows,
        hidden_dim,
        eps_bits,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("rms_norm: {e}")))?;

    Ok(())
}

/// Fused residual addition + RMS Normalization.
///
/// Computes:
/// ```text
/// residual[i] = residual[i] + input[i]   (in-place update)
/// rms = sqrt(mean(residual[i]^2) + epsilon)
/// output[i] = residual[i] / rms * gamma
/// ```
///
/// This fused kernel avoids a separate element-wise add kernel launch and
/// reduces global memory traffic, which is critical for memory-bandwidth-bound
/// LLM inference.
///
/// # Arguments
///
/// * `handle` -- DNN handle.
/// * `input` -- Input tensor (the "new" activations to add).
/// * `residual` -- Mutable residual tensor (updated in-place with input + residual).
/// * `gamma` -- Per-element scale (length = hidden dim).
/// * `output` -- Mutable output tensor for normalized result.
/// * `epsilon` -- Numerical stability constant.
///
/// # Errors
///
/// Returns [`DnnError`] on validation or launch failure.
pub fn fused_add_rms_norm<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    residual: &mut TensorDescMut<T>,
    gamma: &DeviceBuffer<T>,
    output: &mut TensorDescMut<T>,
    epsilon: f32,
) -> DnnResult<()> {
    let (num_rows, hidden_dim) = extract_row_dims(input)?;
    validate_rms_args(input, gamma, output, hidden_dim)?;
    if residual.numel() < input.numel() {
        return Err(DnnError::BufferTooSmall {
            expected: input.numel() * T::SIZE,
            actual: residual.numel() * T::SIZE,
        });
    }

    let ptx_source = generate_rms_norm_ptx::<T>(handle.sm_version(), hidden_dim, true)?;
    let kernel_name = rms_norm_kernel_name::<T>(hidden_dim, true);
    let module =
        Arc::new(Module::from_ptx(&ptx_source).map_err(|e| {
            DnnError::LaunchFailed(format!("module load for fused_add_rms_norm: {e}"))
        })?);
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| DnnError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;

    let (grid, block) = launch_config(num_rows, hidden_dim);
    let params = LaunchParams::new(grid, block);
    let eps_bits = epsilon.to_bits();

    let args = (
        input.ptr,
        output.ptr,
        gamma.as_device_ptr(),
        residual.ptr,
        num_rows,
        hidden_dim,
        eps_bits,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("fused_add_rms_norm: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_row_dims<T: GpuFloat>(desc: &TensorDesc<T>) -> DnnResult<(u32, u32)> {
    let ndim = desc.dims.len();
    if ndim == 0 {
        return Err(DnnError::InvalidDimension("tensor has 0 dimensions".into()));
    }
    let hidden_dim = desc.dims[ndim - 1];
    if hidden_dim == 0 {
        return Err(DnnError::InvalidDimension(
            "hidden dimension is zero".into(),
        ));
    }
    let num_rows: u32 = desc.dims[..ndim - 1]
        .iter()
        .copied()
        .product::<u32>()
        .max(1);
    Ok((num_rows, hidden_dim))
}

fn validate_rms_args<T: GpuFloat>(
    input: &TensorDesc<T>,
    gamma: &DeviceBuffer<T>,
    output: &TensorDescMut<T>,
    hidden_dim: u32,
) -> DnnResult<()> {
    let d = hidden_dim as usize;
    if gamma.len() < d {
        return Err(DnnError::BufferTooSmall {
            expected: d * T::SIZE,
            actual: gamma.len() * T::SIZE,
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

fn launch_config(num_rows: u32, hidden_dim: u32) -> (u32, u32) {
    let block_size = if hidden_dim <= 1024 {
        hidden_dim.next_power_of_two().min(1024)
    } else {
        1024
    };
    (num_rows, block_size)
}

fn rms_norm_kernel_name<T: GpuFloat>(hidden_dim: u32, fused: bool) -> String {
    let prefix = if fused {
        "fused_add_rms_norm"
    } else {
        "rms_norm"
    };
    format!("{prefix}_{}_d{}", T::NAME, hidden_dim)
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates PTX for (optionally fused add +) RMS normalization.
///
/// Kernel parameters:
/// - `input`        (u64) -- input pointer
/// - `output`       (u64) -- output pointer
/// - `gamma`        (u64) -- scale vector pointer
/// - `residual`     (u64) -- residual pointer (0 if not fused)
/// - `n`            (u32) -- number of rows
/// - `d`            (u32) -- hidden dimension
/// - `epsilon_bits` (u32) -- epsilon as f32 bits
fn generate_rms_norm_ptx<T: GpuFloat>(
    sm: SmVersion,
    hidden_dim: u32,
    fused: bool,
) -> DnnResult<String> {
    let ptx_ty = T::PTX_TYPE;
    let ty = ptx_ty.as_ptx_str();
    let byte_size = ptx_ty.size_bytes();
    let kernel_name = rms_norm_kernel_name::<T>(hidden_dim, fused);
    let use_warp = hidden_dim <= 32;
    let block_size = if hidden_dim <= 1024 {
        hidden_dim.next_power_of_two().min(1024)
    } else {
        1024
    };
    let smem_bytes = (block_size as usize) * 4;

    let mut ptx = String::with_capacity(6144);

    // Header
    writeln!(ptx, ".version {}", sm.ptx_version()).map_err(fmt_err)?;
    writeln!(ptx, ".target {}", sm.as_ptx_str()).map_err(fmt_err)?;
    writeln!(ptx, ".address_size 64").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;
    writeln!(ptx, ".visible .entry {kernel_name}(").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_input,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_output,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_gamma,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_residual,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_n,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_d,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_epsilon_bits").map_err(fmt_err)?;
    writeln!(ptx, ")").map_err(fmt_err)?;
    writeln!(ptx, "{{").map_err(fmt_err)?;
    writeln!(ptx, "    .maxntid {block_size}, 1, 1;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .b32 %r<32>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .b64 %rd<16>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .f32 %f<32>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .pred %p<8>;").map_err(fmt_err)?;
    if !use_warp {
        writeln!(ptx, "    .shared .align 4 .b8 smem_rms[{smem_bytes}];").map_err(fmt_err)?;
    }
    writeln!(ptx).map_err(fmt_err)?;

    // Thread / row indexing
    writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r2, [%param_n];").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p0, %r1, %r2;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p0 bra $RMS_DONE;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd2, [%param_gamma];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd3, [%param_residual];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r3, [%param_d];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r4, [%param_epsilon_bits];").map_err(fmt_err)?;
    writeln!(ptx, "    mov.b32 %f20, %r4;").map_err(fmt_err)?;

    // Row element offset
    writeln!(ptx, "    cvt.u64.u32 %rd4, %r1;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd5, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd6, %rd4, %rd5;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    if use_warp {
        write_warp_rms(&mut ptx, ty, byte_size, hidden_dim, fused)?;
    } else {
        write_block_rms(&mut ptx, ty, byte_size, hidden_dim, block_size, fused)?;
    }

    writeln!(ptx, "$RMS_DONE:").map_err(fmt_err)?;
    writeln!(ptx, "    ret;").map_err(fmt_err)?;
    writeln!(ptx, "}}").map_err(fmt_err)?;

    Ok(ptx)
}

/// Warp-level RMS norm for D <= 32.
fn write_warp_rms(
    ptx: &mut String,
    ty: &str,
    byte_size: usize,
    hidden_dim: u32,
    fused: bool,
) -> DnnResult<()> {
    writeln!(ptx, "    // Warp-level RMSNorm").map_err(fmt_err)?;
    writeln!(ptx, "    setp.lt.u32 %p1, %r0, {hidden_dim};").map_err(fmt_err)?;

    // Load input
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    @!%p1 bra $WARP_RMS_SQ;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f0", "%rd9")?;

    if fused {
        // Load residual, add, store back
        writeln!(ptx, "    add.u64 %rd10, %rd3, %rd8;").map_err(fmt_err)?;
        load_global(ptx, ty, "%f1", "%rd10")?;
        writeln!(ptx, "    add.f32 %f0, %f0, %f1;").map_err(fmt_err)?;
        store_global(ptx, ty, "%rd10", "%f0")?;
    }

    writeln!(ptx, "$WARP_RMS_SQ:").map_err(fmt_err)?;

    // Pass 1: sum of squares
    writeln!(ptx, "    mul.f32 %f2, %f0, %f0;").map_err(fmt_err)?;
    writeln!(ptx, "    @!%p1 mov.f32 %f2, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.f32 %f3, %f2;").map_err(fmt_err)?;
    for offset in [16u32, 8, 4, 2, 1] {
        writeln!(
            ptx,
            "    shfl.sync.down.b32 %f4, %f3, {offset}, 31, 0xFFFFFFFF;"
        )
        .map_err(fmt_err)?;
        writeln!(ptx, "    add.f32 %f3, %f3, %f4;").map_err(fmt_err)?;
    }
    writeln!(ptx, "    shfl.sync.idx.b32 %f3, %f3, 0, 31, 0xFFFFFFFF;").map_err(fmt_err)?;

    // rms = rsqrt(mean_sq + eps)
    writeln!(ptx, "    cvt.rn.f32.u32 %f5, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f6, %f3, %f5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.f32 %f6, %f6, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f7, %f6;").map_err(fmt_err)?;

    // Pass 2: normalize + scale
    writeln!(ptx, "    @!%p1 bra $RMS_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.f32 %f8, %f0, %f7;").map_err(fmt_err)?;

    // Load gamma
    writeln!(ptx, "    cvt.u64.u32 %rd11, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd12, %rd2, %rd11;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f9", "%rd12")?;
    writeln!(ptx, "    mul.f32 %f10, %f8, %f9;").map_err(fmt_err)?;

    // Store
    writeln!(ptx, "    add.u64 %rd13, %rd1, %rd8;").map_err(fmt_err)?;
    store_global(ptx, ty, "%rd13", "%f10")?;
    writeln!(ptx).map_err(fmt_err)?;

    Ok(())
}

/// Block-level RMS norm for D > 32.
fn write_block_rms(
    ptx: &mut String,
    ty: &str,
    byte_size: usize,
    hidden_dim: u32,
    block_size: u32,
    fused: bool,
) -> DnnResult<()> {
    writeln!(ptx, "    // Block-level RMSNorm").map_err(fmt_err)?;

    // Pass 1: partial sum of squares (with optional fused add)
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$RMS_SQ_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p1 bra $RMS_SQ_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f1", "%rd9")?;

    if fused {
        writeln!(ptx, "    add.u64 %rd10, %rd3, %rd8;").map_err(fmt_err)?;
        load_global(ptx, ty, "%f2", "%rd10")?;
        writeln!(ptx, "    add.f32 %f1, %f1, %f2;").map_err(fmt_err)?;
        store_global(ptx, ty, "%rd10", "%f1")?;
    }

    writeln!(ptx, "    fma.rn.f32 %f0, %f1, %f1, %f0;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $RMS_SQ_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$RMS_SQ_DONE:").map_err(fmt_err)?;

    // Shared memory reduction
    write_smem_reduce_f32(ptx, "%f0", block_size, "RMS")?;

    // Compute rsqrt(mean_sq + eps)
    writeln!(ptx, "    ld.shared.f32 %f6, [smem_rms];").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.rn.f32.u32 %f5, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f6, %f6, %f5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.f32 %f6, %f6, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f7, %f6;").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 2: normalize + scale + store
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$RMS_NORM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p3, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p3 bra $RMS_DONE;").map_err(fmt_err)?;

    // Reload x (or residual if fused)
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    if fused {
        // Read from residual (which now contains input + residual)
        writeln!(ptx, "    add.u64 %rd9, %rd3, %rd8;").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    }
    load_global(ptx, ty, "%f8", "%rd9")?;

    writeln!(ptx, "    mul.f32 %f8, %f8, %f7;").map_err(fmt_err)?;

    // Load gamma
    writeln!(ptx, "    cvt.u64.u32 %rd11, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd11, %rd11, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd12, %rd2, %rd11;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f9", "%rd12")?;
    writeln!(ptx, "    mul.f32 %f10, %f8, %f9;").map_err(fmt_err)?;

    // Store
    writeln!(ptx, "    add.u64 %rd13, %rd1, %rd8;").map_err(fmt_err)?;
    store_global(ptx, ty, "%rd13", "%f10")?;

    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $RMS_NORM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    Ok(())
}

/// Shared memory tree reduction for f32.
fn write_smem_reduce_f32(
    ptx: &mut String,
    val_reg: &str,
    block_size: u32,
    tag: &str,
) -> DnnResult<()> {
    writeln!(ptx, "    // Shared memory reduction ({tag})").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd14, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd14, %rd14, 4;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u64 %rd15, smem_rms;").map_err(fmt_err)?;
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

/// Emit a global load into an f32 register.
fn load_global(ptx: &mut String, ty: &str, dst: &str, addr: &str) -> DnnResult<()> {
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 {dst}, [{addr}];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} {dst}, [{addr}];").map_err(fmt_err)?;
    }
    Ok(())
}

/// Emit a global store from an f32 register.
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
    fn ptx_rms_warp() {
        let ptx = generate_rms_norm_ptx::<f32>(SmVersion::Sm80, 16, false);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("rms_norm_f32_d16"));
        assert!(ptx.contains("shfl.sync"));
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn ptx_rms_block() {
        let ptx = generate_rms_norm_ptx::<f32>(SmVersion::Sm80, 256, false);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("rms_norm_f32_d256"));
        assert!(ptx.contains("smem_rms"));
    }

    #[test]
    fn ptx_fused_add_rms() {
        let ptx = generate_rms_norm_ptx::<f32>(SmVersion::Sm80, 128, true);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("fused_add_rms_norm_f32_d128"));
        assert!(ptx.contains("%param_residual"));
    }

    // -----------------------------------------------------------------------
    // Task 4: RMSNorm formula verification (CPU reference)
    // -----------------------------------------------------------------------

    /// CPU reference implementation of RMSNorm.
    ///
    /// y[i] = x[i] / sqrt(mean(x^2) + eps) * gamma[i]
    fn rms_norm_cpu(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len() as f32;
        let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / n;
        let rms = (mean_sq + eps).sqrt();
        x.iter()
            .zip(gamma)
            .map(|(&xi, &gi)| xi / rms * gi)
            .collect()
    }

    /// RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma
    #[test]
    fn test_rms_norm_formula() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let gamma = [1.0f32; 4];
        let eps = 1e-5f32;

        // rms = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) ≈ 2.7386
        let mean_sq = (1.0f32 + 4.0 + 9.0 + 16.0) / 4.0;
        let rms = (mean_sq + eps).sqrt();

        let result = rms_norm_cpu(&x, &gamma, eps);

        assert_eq!(result.len(), 4);
        let expected: Vec<f32> = x.iter().map(|&v| v / rms).collect();
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "element {i}: expected {e}, got {r}");
        }

        // Approximate expected values from the docstring: ≈ [0.365, 0.730, 1.095, 1.461]
        let approx = [0.365f32, 0.730, 1.095, 1.461];
        for (i, (&r, &a)) in result.iter().zip(approx.iter()).enumerate() {
            assert!(
                (r - a).abs() < 0.001,
                "element {i}: expected approx {a}, got {r}"
            );
        }
    }

    /// RMSNorm with non-unit gamma scales output proportionally.
    #[test]
    fn test_rms_norm_formula_with_gamma() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let gamma_unit = [1.0f32; 4];
        let gamma_scaled = [2.0f32; 4];
        let eps = 1e-5f32;

        let result_unit = rms_norm_cpu(&x, &gamma_unit, eps);
        let result_scaled = rms_norm_cpu(&x, &gamma_scaled, eps);

        for (i, (&u, &s)) in result_unit.iter().zip(result_scaled.iter()).enumerate() {
            assert!(
                (s - 2.0 * u).abs() < 1e-5,
                "element {i}: scaled should be 2x unit, {s} vs {}",
                2.0 * u
            );
        }
    }

    /// RMSNorm does NOT subtract the mean (unlike LayerNorm).
    ///
    /// Adding a constant to all inputs changes the RMS and thus the output.
    #[test]
    fn test_rms_norm_not_shift_invariant() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let x_shifted: Vec<f32> = x.iter().map(|&v| v + 10.0).collect();
        let gamma = [1.0f32; 4];
        let eps = 1e-5f32;

        let result = rms_norm_cpu(&x, &gamma, eps);
        let result_shifted = rms_norm_cpu(&x_shifted, &gamma, eps);

        // At least one element must differ (RMSNorm is NOT shift-invariant)
        let all_same = result
            .iter()
            .zip(result_shifted.iter())
            .all(|(&r, &rs)| (r - rs).abs() < 1e-5);
        assert!(
            !all_same,
            "RMSNorm must NOT be shift-invariant (unlike LayerNorm)"
        );
    }

    /// RMSNorm on uniform input: all elements get the same scale factor.
    #[test]
    fn test_rms_norm_uniform_input() {
        let x = [3.0f32; 8];
        let gamma = [1.0f32; 8];
        let eps = 1e-8f32;

        let result = rms_norm_cpu(&x, &gamma, eps);

        // rms = sqrt(9 + eps) ≈ 3.0, so y = 3 / 3 = 1.0 for each element
        for (i, &r) in result.iter().enumerate() {
            assert!(
                (r - 1.0).abs() < 1e-5,
                "element {i}: uniform input should produce ~1.0, got {r}"
            );
        }
    }

    /// RMSNorm scales proportionally with gamma.
    #[test]
    fn test_rms_norm_proportional_to_gamma() {
        let x = [1.0f32, 0.5, 2.0, 1.5];
        let eps = 1e-5f32;
        let gamma_a = [1.0f32, 2.0, 3.0, 0.5];
        let gamma_b: Vec<f32> = gamma_a.iter().map(|&g| g * 3.0).collect();

        let result_a = rms_norm_cpu(&x, &gamma_a, eps);
        let result_b = rms_norm_cpu(&x, &gamma_b, eps);

        for (i, (&a, &b)) in result_a.iter().zip(result_b.iter()).enumerate() {
            assert!(
                (b - 3.0 * a).abs() < 1e-5,
                "element {i}: 3x gamma should give 3x output, {b} vs {}",
                3.0 * a
            );
        }
    }

    /// RMSNorm output is always positive when gamma > 0 and x >= 0.
    #[test]
    fn test_rms_norm_positive_output_for_positive_input() {
        let x = [0.1f32, 0.5, 1.0, 2.0, 5.0];
        let gamma = [1.0f32; 5];
        let eps = 1e-5f32;

        let result = rms_norm_cpu(&x, &gamma, eps);
        for (i, &r) in result.iter().enumerate() {
            assert!(
                r > 0.0,
                "element {i}: positive input should give positive output, got {r}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Numerical accuracy quality-gate tests (CPU reference)
    // -----------------------------------------------------------------------

    /// RMSNorm known-values test: x=[3,4], gamma=1.
    ///
    /// rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
    /// y[0] = 3/3.5355 ≈ 0.8485, y[1] = 4/3.5355 ≈ 1.1314
    #[test]
    fn test_rms_norm_f32_known_values() {
        let x = [3.0f32, 4.0];
        let gamma = [1.0f32; 2];
        let eps = 1e-7f32;
        let result = rms_norm_cpu(&x, &gamma, eps);
        assert_eq!(result.len(), 2);
        assert!(
            (result[0] - 0.8485).abs() < 1e-3,
            "y[0]={} expected ≈0.8485",
            result[0]
        );
        assert!(
            (result[1] - 1.1314).abs() < 1e-3,
            "y[1]={} expected ≈1.1314",
            result[1]
        );
    }

    /// RMSNorm scale invariance: scaling all inputs by k scales output by same k.
    ///
    /// rms(k*x) = k * rms(x), so y_scaled[i] = k*x[i] / (k*rms(x)) * gamma = y[i].
    /// That means RMSNorm IS scale-invariant (unlike mean-invariance which it lacks).
    #[test]
    fn test_rms_norm_scale_invariance() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let x_scaled: Vec<f32> = x.iter().map(|&v| v * 5.0).collect();
        let gamma = [1.0f32; 4];
        let eps = 1e-8f32;

        let result = rms_norm_cpu(&x, &gamma, eps);
        let result_scaled = rms_norm_cpu(&x_scaled, &gamma, eps);

        for (i, (&r, &rs)) in result.iter().zip(result_scaled.iter()).enumerate() {
            assert!(
                (r - rs).abs() < 1e-5,
                "element {i}: RMSNorm should be scale-invariant, {r} vs {rs}"
            );
        }
    }

    /// RMSNorm on near-zero input: output should be near zero (eps prevents div-by-zero).
    #[test]
    fn test_rms_norm_near_zero_input() {
        let x = [1e-20f32, 1e-20, 1e-20, 1e-20];
        let gamma = [1.0f32; 4];
        let eps = 1e-5f32;
        let result = rms_norm_cpu(&x, &gamma, eps);
        for (i, &r) in result.iter().enumerate() {
            assert!(
                r.is_finite(),
                "element {i}: near-zero input must give finite output, got {r}"
            );
            // With near-zero inputs, output ≈ 1e-20 / sqrt(eps) ≈ very small
            assert!(
                r.abs() < 1.0,
                "element {i}: near-zero input should give small output, got {r}"
            );
        }
    }

    /// When the input has zero mean (like LayerNorm inputs after centering),
    /// RMSNorm and LayerNorm give different results in general.
    ///
    /// But for x = [-1, 1], mean(x) = 0, so rms = sqrt(1) = 1, and
    /// rms_norm output = x itself (with gamma=1).
    /// LayerNorm: var = 1, so output is also x (same result in this special case).
    #[test]
    fn test_rms_norm_vs_layer_norm_zero_mean_input() {
        let x = [-1.0f32, 1.0];
        let gamma = [1.0f32; 2];
        let beta = [0.0f32; 2];
        let eps = 1e-7f32;

        let rms_result = rms_norm_cpu(&x, &gamma, eps);

        // CPU reference LayerNorm for comparison
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n; // = 0.0
        let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n; // = 1.0
        let inv_std = 1.0 / (var + eps).sqrt();
        let ln_result: Vec<f32> = x
            .iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .map(|((&xi, &gi), &bi)| (xi - mean) * inv_std * gi + bi)
            .collect();

        // When mean=0 and rms=std, results should be very close
        for (i, (&r, &l)) in rms_result.iter().zip(ln_result.iter()).enumerate() {
            assert!(
                (r - l).abs() < 1e-5,
                "element {i}: RMSNorm and LayerNorm should agree for zero-mean unit-rms input, rms={r} vs ln={l}"
            );
        }
    }

    /// FP16 proxy accuracy: inputs in FP16 magnitude range, verify no precision disaster.
    #[test]
    fn test_rms_norm_fp16_proxy_accuracy() {
        // Typical LLM embedding magnitudes
        let x = [0.25f32, -0.125, 0.5, -0.375, 0.0625, 0.1875, -0.25, 0.3125];
        let gamma: Vec<f32> = vec![1.0, 0.9375, 1.0625, 0.875, 1.125, 0.9375, 1.0, 1.0625];
        let eps = 1e-5f32;
        let result = rms_norm_cpu(&x, &gamma, eps);

        for (i, &y) in result.iter().enumerate() {
            assert!(
                y.is_finite(),
                "element {i}: FP16-proxy input must give finite output"
            );
        }

        // Output RMS should be close to 1.0 when gamma ≈ 1.0 (unit gamma)
        let unit_gamma = vec![1.0f32; 8];
        let unit_result = rms_norm_cpu(&x, &unit_gamma, eps);
        let out_rms = (unit_result.iter().map(|&v| v * v).sum::<f32>() / 8.0).sqrt();
        assert!(
            (out_rms - 1.0).abs() < 1e-4,
            "RMSNorm output RMS should be ≈1.0 with unit gamma, got {out_rms}"
        );
    }
}
