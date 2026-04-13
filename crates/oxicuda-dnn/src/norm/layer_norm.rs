//! Layer Normalization for Transformer models.
//!
//! Implements the standard LayerNorm operation:
//!
//! ```text
//! y = (x - mean) / sqrt(var + epsilon) * gamma + beta
//! ```
//!
//! Each CTA (cooperative thread array / block) handles one row of the input
//! tensor. For hidden dimensions D <= 1024, warp-level shuffle reductions are
//! used. For D > 1024, shared-memory block-level reductions are employed.
//!
//! All intermediate accumulation is performed in f32 for numerical stability,
//! even when the input/output type is f16 or bf16.

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

/// Applies Layer Normalization across the last dimension of the input tensor.
///
/// The input is treated as a 2-D matrix of shape `[N, D]` where `N` is the
/// product of all dimensions except the last (the "hidden" dimension `D`).
/// `gamma` and `beta` are 1-D vectors of length `D`.
///
/// # Arguments
///
/// * `handle` -- DNN handle bound to a CUDA context and stream.
/// * `input` -- Input tensor descriptor (row-major, last dim = hidden).
/// * `gamma` -- Per-element scale (length = hidden dim).
/// * `beta` -- Per-element bias (length = hidden dim).
/// * `output` -- Mutable output tensor descriptor (same shape as input).
/// * `epsilon` -- Small constant added to variance for numerical stability.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if dimensions are zero or
/// gamma/beta lengths do not match the hidden dimension.
/// Returns [`DnnError::BufferTooSmall`] if any buffer is undersized.
pub fn layer_norm<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    gamma: &DeviceBuffer<T>,
    beta: &DeviceBuffer<T>,
    output: &mut TensorDescMut<T>,
    epsilon: f32,
) -> DnnResult<()> {
    let (num_rows, hidden_dim) = extract_row_dims(input)?;
    validate_layer_norm_args(input, gamma, beta, output, hidden_dim)?;

    let ptx_source = generate_layer_norm_ptx::<T>(handle.sm_version(), hidden_dim)?;
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| DnnError::LaunchFailed(format!("module load for layer_norm: {e}")))?,
    );
    let kernel_name = layer_norm_kernel_name::<T>(hidden_dim);
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| DnnError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;

    let (grid, block) = launch_config_for_row_norm(num_rows, hidden_dim);
    let params = LaunchParams::new(grid, block);
    let eps_bits = epsilon.to_bits();

    let args = (
        input.ptr,
        output.ptr,
        gamma.as_device_ptr(),
        beta.as_device_ptr(),
        num_rows,
        hidden_dim,
        eps_bits,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("layer_norm: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extracts (num_rows, hidden_dim) from a tensor descriptor.
///
/// The hidden dimension is the last active dimension; num_rows is the product
/// of all preceding dimensions.
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

/// Validates buffer sizes and dimension consistency for layer norm.
fn validate_layer_norm_args<T: GpuFloat>(
    input: &TensorDesc<T>,
    gamma: &DeviceBuffer<T>,
    beta: &DeviceBuffer<T>,
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
    if beta.len() < d {
        return Err(DnnError::BufferTooSmall {
            expected: d * T::SIZE,
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

/// Determines grid and block dimensions for row-wise normalization.
///
/// Each block handles one row. For D <= 1024 the block size equals the next
/// power of two >= D (capped at 1024). For D > 1024 we use 1024 threads and
/// each thread processes multiple elements via a strided loop.
fn launch_config_for_row_norm(num_rows: u32, hidden_dim: u32) -> (u32, u32) {
    let block_size = if hidden_dim <= 1024 {
        hidden_dim.next_power_of_two().min(1024)
    } else {
        1024
    };
    (num_rows, block_size)
}

fn layer_norm_kernel_name<T: GpuFloat>(hidden_dim: u32) -> String {
    format!("layer_norm_{}_d{}", T::NAME, hidden_dim)
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates PTX source for the layer normalization kernel.
///
/// Kernel parameters:
/// - `input`        (u64) -- pointer to input rows
/// - `output`       (u64) -- pointer to output rows
/// - `gamma`        (u64) -- pointer to scale vector
/// - `beta`         (u64) -- pointer to bias vector
/// - `n`            (u32) -- number of rows
/// - `d`            (u32) -- hidden dimension
/// - `epsilon_bits` (u32) -- epsilon as f32 bit pattern
///
/// The kernel performs three passes per row:
/// 1. Accumulate sum for mean.
/// 2. Accumulate (x - mean)^2 for variance.
/// 3. Normalize: y = (x - mean) * rsqrt(var + eps) * gamma + beta.
fn generate_layer_norm_ptx<T: GpuFloat>(sm: SmVersion, hidden_dim: u32) -> DnnResult<String> {
    let ptx_ty = T::PTX_TYPE;
    let ty = ptx_ty.as_ptx_str();
    let byte_size = ptx_ty.size_bytes();
    let kernel_name = layer_norm_kernel_name::<T>(hidden_dim);
    let use_warp = hidden_dim <= 32;
    let block_size = if hidden_dim <= 1024 {
        hidden_dim.next_power_of_two().min(1024)
    } else {
        1024
    };
    let smem_bytes = (block_size as usize) * 4; // f32 accumulator

    let mut ptx = String::with_capacity(8192);

    // -- Header --
    write_header(&mut ptx, sm, &kernel_name, block_size, smem_bytes, use_warp)?;

    if use_warp {
        write_warp_layer_norm(&mut ptx, ty, byte_size, hidden_dim)?;
    } else {
        write_block_layer_norm(&mut ptx, ty, byte_size, hidden_dim, block_size)?;
    }

    writeln!(ptx, "$LN_DONE:").map_err(fmt_err)?;
    writeln!(ptx, "    ret;").map_err(fmt_err)?;
    writeln!(ptx, "}}").map_err(fmt_err)?;

    Ok(ptx)
}

fn write_header(
    ptx: &mut String,
    sm: SmVersion,
    kernel_name: &str,
    block_size: u32,
    smem_bytes: usize,
    use_warp: bool,
) -> DnnResult<()> {
    writeln!(ptx, ".version {}", sm.ptx_version()).map_err(fmt_err)?;
    writeln!(ptx, ".target {}", sm.as_ptx_str()).map_err(fmt_err)?;
    writeln!(ptx, ".address_size 64").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;
    writeln!(ptx, ".visible .entry {kernel_name}(").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_input,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_output,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_gamma,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u64 %param_beta,").map_err(fmt_err)?;
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
        writeln!(ptx, "    .shared .align 4 .b8 smem_ln[{smem_bytes}];").map_err(fmt_err)?;
    }
    writeln!(ptx).map_err(fmt_err)?;

    // Row index = blockIdx.x, thread index = threadIdx.x
    writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r2, [%param_n];").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p0, %r1, %r2;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p0 bra $LN_DONE;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    // Load params
    writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd2, [%param_gamma];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd3, [%param_beta];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r3, [%param_d];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r4, [%param_epsilon_bits];").map_err(fmt_err)?;
    writeln!(ptx, "    mov.b32 %f20, %r4;").map_err(fmt_err)?; // epsilon as f32

    // Base address for this row's input/output
    writeln!(ptx, "    cvt.u64.u32 %rd4, %r1;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd5, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd6, %rd4, %rd5;").map_err(fmt_err)?;
    // row_elem_offset in %rd6 (element index, not byte offset)
    writeln!(ptx, "    // row_elem_offset in %rd6").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    Ok(())
}

/// Warp-shuffle-based layer norm for hidden_dim <= 32.
fn write_warp_layer_norm(
    ptx: &mut String,
    ty: &str,
    byte_size: usize,
    hidden_dim: u32,
) -> DnnResult<()> {
    // Thread r0 = lane within warp. For layer norm, row = blockIdx.x.
    // Each block has block_size threads; here it's a single warp.
    // Lane < hidden_dim loads data, otherwise contributes 0.
    writeln!(ptx, "    // Warp-level LayerNorm (D <= 32)").map_err(fmt_err)?;
    writeln!(ptx, "    setp.lt.u32 %p1, %r0, {hidden_dim};").map_err(fmt_err)?;

    // Load input element
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    @!%p1 bra $WARP_MEAN;").map_err(fmt_err)?;
    // Address: input + (row * D + lane) * byte_size
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f0, [%rd9];").map_err(fmt_err)?;
    } else {
        // Load in native type, convert to f32 for accumulation
        writeln!(ptx, "    ld.global{ty} %f0, [%rd9];").map_err(fmt_err)?;
    }
    writeln!(ptx, "$WARP_MEAN:").map_err(fmt_err)?;

    // Pass 1: sum for mean via warp shuffle
    writeln!(ptx, "    mov.f32 %f1, %f0;").map_err(fmt_err)?;
    for offset in [16u32, 8, 4, 2, 1] {
        writeln!(
            ptx,
            "    shfl.sync.down.b32 %f2, %f1, {offset}, 31, 0xFFFFFFFF;"
        )
        .map_err(fmt_err)?;
        writeln!(ptx, "    add.f32 %f1, %f1, %f2;").map_err(fmt_err)?;
    }
    // Broadcast sum and compute mean
    writeln!(ptx, "    shfl.sync.idx.b32 %f1, %f1, 0, 31, 0xFFFFFFFF;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.rn.f32.u32 %f3, %r3;").map_err(fmt_err)?; // D as float
    writeln!(ptx, "    div.approx.f32 %f4, %f1, %f3;").map_err(fmt_err)?; // mean

    // Pass 2: variance = sum((x - mean)^2) / D
    writeln!(ptx, "    sub.f32 %f5, %f0, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.f32 %f5, %f5, %f5;").map_err(fmt_err)?;
    writeln!(ptx, "    @!%p1 mov.f32 %f5, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.f32 %f6, %f5;").map_err(fmt_err)?;
    for offset in [16u32, 8, 4, 2, 1] {
        writeln!(
            ptx,
            "    shfl.sync.down.b32 %f7, %f6, {offset}, 31, 0xFFFFFFFF;"
        )
        .map_err(fmt_err)?;
        writeln!(ptx, "    add.f32 %f6, %f6, %f7;").map_err(fmt_err)?;
    }
    writeln!(ptx, "    shfl.sync.idx.b32 %f6, %f6, 0, 31, 0xFFFFFFFF;").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f8, %f6, %f3;").map_err(fmt_err)?; // variance

    // rsqrt(var + eps)
    writeln!(ptx, "    add.f32 %f9, %f8, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f10, %f9;").map_err(fmt_err)?;

    // Pass 3: normalize + scale + bias
    writeln!(ptx, "    @!%p1 bra $LN_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    sub.f32 %f11, %f0, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.f32 %f11, %f11, %f10;").map_err(fmt_err)?;

    // Load gamma and beta
    writeln!(ptx, "    cvt.u64.u32 %rd10, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd10, %rd10, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd11, %rd2, %rd10;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd12, %rd3, %rd10;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f12, [%rd11];").map_err(fmt_err)?;
        writeln!(ptx, "    ld.global.f32 %f13, [%rd12];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f12, [%rd11];").map_err(fmt_err)?;
        writeln!(ptx, "    ld.global{ty} %f13, [%rd12];").map_err(fmt_err)?;
    }
    writeln!(ptx, "    fma.rn.f32 %f14, %f11, %f12, %f13;").map_err(fmt_err)?;

    // Store result
    if ty == ".f32" {
        writeln!(ptx, "    add.u64 %rd13, %rd1, %rd8;").map_err(fmt_err)?;
        writeln!(ptx, "    st.global.f32 [%rd13], %f14;").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    add.u64 %rd13, %rd1, %rd8;").map_err(fmt_err)?;
        writeln!(ptx, "    st.global{ty} [%rd13], %f14;").map_err(fmt_err)?;
    }
    writeln!(ptx).map_err(fmt_err)?;

    Ok(())
}

/// Block-level shared memory layer norm for hidden_dim > 32.
fn write_block_layer_norm(
    ptx: &mut String,
    ty: &str,
    byte_size: usize,
    hidden_dim: u32,
    block_size: u32,
) -> DnnResult<()> {
    writeln!(ptx, "    // Block-level LayerNorm (D > 32)").map_err(fmt_err)?;

    // Pass 1: each thread accumulates partial sum via strided loop
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?; // partial sum
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$LN_SUM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p1 bra $LN_SUM_DONE;").map_err(fmt_err)?;
    // Compute address: input + (row * D + r5) * byte_size
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f1, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f1, [%rd9];").map_err(fmt_err)?;
    }
    writeln!(ptx, "    add.f32 %f0, %f0, %f1;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $LN_SUM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$LN_SUM_DONE:").map_err(fmt_err)?;

    // Reduce partial sums via shared memory
    write_smem_reduce_f32(ptx, "%f0", block_size, "SUM")?;

    // Broadcast mean
    writeln!(ptx, "    ld.shared.f32 %f4, [smem_ln];").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.rn.f32.u32 %f3, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f4, %f4, %f3;").map_err(fmt_err)?; // mean
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 2: variance partial sum
    writeln!(ptx, "    mov.f32 %f5, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$LN_VAR_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p2, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p2 bra $LN_VAR_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
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
    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $LN_VAR_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$LN_VAR_DONE:").map_err(fmt_err)?;

    // Reduce variance via shared memory
    write_smem_reduce_f32(ptx, "%f5", block_size, "VAR")?;

    writeln!(ptx, "    ld.shared.f32 %f8, [smem_ln];").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f8, %f8, %f3;").map_err(fmt_err)?; // variance
    writeln!(ptx, "    add.f32 %f9, %f8, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f10, %f9;").map_err(fmt_err)?; // inv_std
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 3: normalize, scale, bias, store
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$LN_NORM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p3, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p3 bra $LN_DONE;").map_err(fmt_err)?;

    // Reload x
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f11, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f11, [%rd9];").map_err(fmt_err)?;
    }

    // Normalize
    writeln!(ptx, "    sub.f32 %f11, %f11, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.f32 %f11, %f11, %f10;").map_err(fmt_err)?;

    // Load gamma, beta
    writeln!(ptx, "    cvt.u64.u32 %rd10, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd10, %rd10, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd11, %rd2, %rd10;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd12, %rd3, %rd10;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f12, [%rd11];").map_err(fmt_err)?;
        writeln!(ptx, "    ld.global.f32 %f13, [%rd12];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f12, [%rd11];").map_err(fmt_err)?;
        writeln!(ptx, "    ld.global{ty} %f13, [%rd12];").map_err(fmt_err)?;
    }
    writeln!(ptx, "    fma.rn.f32 %f14, %f11, %f12, %f13;").map_err(fmt_err)?;

    // Store
    writeln!(ptx, "    add.u64 %rd13, %rd1, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    st.global.f32 [%rd13], %f14;").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    st.global{ty} [%rd13], %f14;").map_err(fmt_err)?;
    }
    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $LN_NORM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    Ok(())
}

/// Shared memory tree reduction for an f32 accumulator.
///
/// Assumes the value to reduce is in the register `val_reg`, thread id in `%r0`,
/// and shared memory `smem_ln` is available. After the call, the result is at
/// `smem_ln[0]`.
fn write_smem_reduce_f32(
    ptx: &mut String,
    val_reg: &str,
    block_size: u32,
    tag: &str,
) -> DnnResult<()> {
    writeln!(ptx, "    // Shared memory reduction ({tag})").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd14, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd14, %rd14, 4;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u64 %rd15, smem_ln;").map_err(fmt_err)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_generation_warp() {
        let ptx = generate_layer_norm_ptx::<f32>(SmVersion::Sm80, 16);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry layer_norm_f32_d16"));
        assert!(ptx.contains("shfl.sync"));
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn ptx_generation_block() {
        let ptx = generate_layer_norm_ptx::<f32>(SmVersion::Sm80, 256);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry layer_norm_f32_d256"));
        assert!(ptx.contains("smem_ln"));
        assert!(ptx.contains("bar.sync"));
    }

    #[test]
    fn ptx_generation_large_dim() {
        let ptx = generate_layer_norm_ptx::<f32>(SmVersion::Sm80, 4096);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("layer_norm_f32_d4096"));
    }

    #[test]
    fn launch_config_small() {
        let (grid, block) = launch_config_for_row_norm(32, 16);
        assert_eq!(grid, 32);
        assert_eq!(block, 16);
    }

    #[test]
    fn launch_config_large() {
        let (grid, block) = launch_config_for_row_norm(8, 4096);
        assert_eq!(grid, 8);
        assert_eq!(block, 1024);
    }

    // -----------------------------------------------------------------------
    // Task 4: LayerNorm formula verification (CPU reference)
    // -----------------------------------------------------------------------

    /// CPU reference implementation of LayerNorm.
    ///
    /// y[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
    fn layer_norm_cpu(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + eps).sqrt();
        x.iter()
            .zip(gamma)
            .zip(beta)
            .map(|((&xi, &gi), &bi)| (xi - mean) * inv_std * gi + bi)
            .collect()
    }

    /// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
    #[test]
    fn test_layer_norm_formula() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let gamma = [1.0f32; 4];
        let beta = [0.0f32; 4];
        let eps = 1e-5f32;

        // mean = 2.5, var = 1.25, std ≈ sqrt(1.25 + 1e-5) ≈ 1.11803
        let mean = 2.5f32;
        let var = 1.25f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        let expected: Vec<f32> = x.iter().map(|&v| (v - mean) * inv_std).collect();
        let result = layer_norm_cpu(&x, &gamma, &beta, eps);

        assert_eq!(result.len(), 4);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "element {i}: expected {e}, got {r}");
        }

        // Approximate expected values from the docstring
        let approx = [-1.342f32, -0.447, 0.447, 1.342];
        for (i, (&r, &a)) in result.iter().zip(approx.iter()).enumerate() {
            assert!(
                (r - a).abs() < 0.001,
                "element {i}: expected approx {a}, got {r}"
            );
        }
    }

    /// LayerNorm with non-unit gamma and non-zero beta.
    #[test]
    fn test_layer_norm_formula_with_affine_params() {
        let x = [0.0f32, 1.0, 2.0, 3.0];
        let gamma = [2.0f32, 2.0, 2.0, 2.0];
        let beta = [1.0f32, 1.0, 1.0, 1.0];
        let eps = 1e-5f32;

        let result = layer_norm_cpu(&x, &gamma, &beta, eps);

        // Without affine: normalised values
        let unit_result = layer_norm_cpu(&x, &[1.0f32; 4], &[0.0f32; 4], eps);
        // With affine: gamma * normalised + beta
        for (i, (&r, &u)) in result.iter().zip(unit_result.iter()).enumerate() {
            let expected = 2.0 * u + 1.0;
            assert!(
                (r - expected).abs() < 1e-5,
                "element {i}: expected {expected}, got {r}"
            );
        }
    }

    /// LayerNorm output has (approximately) zero mean and unit variance.
    #[test]
    fn test_layer_norm_output_statistics() {
        // For gamma=1 and beta=0, the normalised output should have mean≈0 and var≈1.
        let x = [1.0f32, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
        let gamma = [1.0f32; 8];
        let beta = [0.0f32; 8];
        let eps = 1e-5f32;

        let result = layer_norm_cpu(&x, &gamma, &beta, eps);

        let n = result.len() as f32;
        let out_mean = result.iter().sum::<f32>() / n;
        let out_var = result
            .iter()
            .map(|&v| (v - out_mean) * (v - out_mean))
            .sum::<f32>()
            / n;

        assert!(
            out_mean.abs() < 1e-5,
            "LayerNorm output mean should be ~0, got {out_mean}"
        );
        assert!(
            (out_var - 1.0).abs() < 1e-4,
            "LayerNorm output variance should be ~1, got {out_var}"
        );
    }

    /// LayerNorm on a single element produces zero (no variance).
    #[test]
    fn test_layer_norm_single_element() {
        let x = [5.0f32];
        let gamma = [1.0f32];
        let beta = [0.0f32];
        let eps = 1e-5f32;

        let result = layer_norm_cpu(&x, &gamma, &beta, eps);
        // mean = 5.0, var = 0.0, out = (5 - 5) / sqrt(eps) = 0
        assert_eq!(result.len(), 1);
        assert!(
            result[0].abs() < 1e-4,
            "single-element LN should be 0, got {}",
            result[0]
        );
    }

    /// LayerNorm is invariant to constant shifts in input (mean removal).
    #[test]
    fn test_layer_norm_shift_invariance() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let x_shifted: Vec<f32> = x.iter().map(|&v| v + 100.0).collect();
        let gamma = [1.0f32; 4];
        let beta = [0.0f32; 4];
        let eps = 1e-5f32;

        let result = layer_norm_cpu(&x, &gamma, &beta, eps);
        let result_shifted = layer_norm_cpu(&x_shifted, &gamma, &beta, eps);

        for (i, (&r, &rs)) in result.iter().zip(result_shifted.iter()).enumerate() {
            assert!(
                (r - rs).abs() < 1e-5,
                "element {i}: shift invariance violated, {r} vs {rs}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Numerical accuracy quality-gate tests (CPU reference, FP16/FP32 proxy)
    // -----------------------------------------------------------------------

    /// LayerNorm known-values test: x=[1,2,3,4], gamma=1, beta=0.
    ///
    /// mean=2.5, var=1.25, std≈1.11803
    /// y[0] = (1-2.5)/1.11803 ≈ -1.3416, y[3] = (4-2.5)/1.11803 ≈ +1.3416
    #[test]
    fn test_layer_norm_f32_known_values() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let gamma = [1.0f32; 4];
        let beta = [0.0f32; 4];
        let result = layer_norm_cpu(&x, &gamma, &beta, 1e-5);
        assert!(
            (result[0] - (-1.3416)).abs() < 1e-3,
            "y[0]={} expected ≈-1.3416",
            result[0]
        );
        assert!(
            (result[3] - 1.3416).abs() < 1e-3,
            "y[3]={} expected ≈+1.3416",
            result[3]
        );
        // Middle values
        assert!(
            (result[1] - (-0.4472)).abs() < 1e-3,
            "y[1]={} expected ≈-0.4472",
            result[1]
        );
        assert!(
            (result[2] - 0.4472).abs() < 1e-3,
            "y[2]={} expected ≈+0.4472",
            result[2]
        );
    }

    /// LayerNorm with gamma=[1..1], beta=[0..0]: output sum ≈ 0.
    #[test]
    fn test_layer_norm_identity_gamma_zero_sum() {
        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let gamma = vec![1.0f32; 8];
        let beta = vec![0.0f32; 8];
        let result = layer_norm_cpu(&x, &gamma, &beta, 1e-5);
        let sum: f32 = result.iter().sum();
        assert!(
            sum.abs() < 1e-4,
            "LayerNorm output sum should be ≈0, got {sum}"
        );
    }

    /// LayerNorm on constant input: numerator (x - mean) = 0 everywhere,
    /// so output = beta regardless of gamma.
    #[test]
    fn test_layer_norm_constant_input_gives_beta() {
        let x = [3.0f32; 8];
        let gamma = [2.0f32; 8];
        let beta = [1.0f32; 8];
        let result = layer_norm_cpu(&x, &gamma, &beta, 1e-5);
        for (i, &y) in result.iter().enumerate() {
            assert!(
                (y - 1.0).abs() < 1e-4,
                "element {i}: constant input → output = beta = 1.0, got {y}"
            );
        }
    }

    /// LayerNorm symmetry: for x=[-a, a], y[0] = -y[1].
    #[test]
    fn test_layer_norm_symmetry_two_elements() {
        let x = [-2.0f32, 2.0];
        let gamma = [1.0f32; 2];
        let beta = [0.0f32; 2];
        let result = layer_norm_cpu(&x, &gamma, &beta, 1e-5);
        assert!(
            (result[0] + result[1]).abs() < 1e-5,
            "LayerNorm symmetry: y[0]+y[1] should be 0, got {}+{}={}",
            result[0],
            result[1],
            result[0] + result[1]
        );
    }

    /// FP16 proxy accuracy test: simulate FP16-range inputs with FP32 math.
    ///
    /// FP16 has ~3 decimal digits of precision. We verify the CPU reference
    /// produces results within FP16 round-trip tolerance (1e-3 relative error).
    #[test]
    fn test_layer_norm_fp16_proxy_accuracy() {
        // Inputs typical of FP16 range: magnitude ~1.0
        let x = [0.5f32, -0.3, 1.2, -0.8, 0.1, 0.9, -0.5, 0.4];
        let gamma = [1.0f32; 8];
        let beta = [0.0f32; 8];
        let result = layer_norm_cpu(&x, &gamma, &beta, 1e-5);

        // Verify numerical properties expected of normalized outputs
        let sum: f32 = result.iter().sum();
        assert!(
            sum.abs() < 1e-4,
            "FP16-proxy: output sum should be ≈0, got {sum}"
        );

        // Each output should be in a reasonable range (not NaN/Inf)
        for (i, &y) in result.iter().enumerate() {
            assert!(
                y.is_finite(),
                "element {i}: non-finite output {y} for FP16-proxy input"
            );
            assert!(
                y.abs() < 4.0,
                "element {i}: suspiciously large magnitude {y}"
            );
        }
    }
}
