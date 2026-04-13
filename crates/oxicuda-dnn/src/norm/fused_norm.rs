//! Fused normalization + activation kernels.
//!
//! These fused operations combine a normalization step with a nonlinear
//! activation in a single kernel launch, eliminating the intermediate
//! global-memory round-trip that would be required by separate norm + activation
//! calls. This is a significant win for memory-bandwidth-bound workloads.
//!
//! Provided fusions:
//!
//! - [`fused_layer_norm_relu`] -- LayerNorm followed by ReLU
//! - [`fused_rms_norm_silu`] -- RMSNorm followed by SiLU (Swish)

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

/// Fused Layer Normalization + ReLU activation.
///
/// Computes:
/// ```text
/// x_norm = (x - mean) / sqrt(var + eps) * gamma + beta
/// y = max(x_norm, 0)
/// ```
///
/// # Arguments
///
/// * `handle` -- DNN handle.
/// * `input` -- Input tensor (row-major, last dim = hidden).
/// * `gamma` -- Per-element scale (length = hidden dim).
/// * `beta` -- Per-element bias (length = hidden dim).
/// * `output` -- Mutable output tensor (same shape).
/// * `epsilon` -- Stability constant.
///
/// # Errors
///
/// Returns [`DnnError`] on validation or launch failure.
pub fn fused_layer_norm_relu<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    gamma: &DeviceBuffer<T>,
    beta: &DeviceBuffer<T>,
    output: &mut TensorDescMut<T>,
    epsilon: f32,
) -> DnnResult<()> {
    launch_fused_norm(
        handle,
        input,
        Some(gamma),
        Some(beta),
        None,
        output,
        epsilon,
        FusedKind::LayerNormRelu,
    )
}

/// Fused RMS Normalization + SiLU (Swish) activation.
///
/// Computes:
/// ```text
/// x_norm = x / sqrt(mean(x^2) + eps) * gamma
/// y = x_norm * sigmoid(x_norm)
/// ```
///
/// This is the pattern used in LLaMA's FFN where the gate projection
/// output goes through SiLU.
///
/// # Arguments
///
/// * `handle` -- DNN handle.
/// * `input` -- Input tensor.
/// * `gamma` -- Per-element scale (length = hidden dim).
/// * `output` -- Mutable output tensor.
/// * `epsilon` -- Stability constant.
///
/// # Errors
///
/// Returns [`DnnError`] on validation or launch failure.
pub fn fused_rms_norm_silu<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    gamma: &DeviceBuffer<T>,
    output: &mut TensorDescMut<T>,
    epsilon: f32,
) -> DnnResult<()> {
    launch_fused_norm(
        handle,
        input,
        Some(gamma),
        None,
        None,
        output,
        epsilon,
        FusedKind::RmsNormSilu,
    )
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

/// Distinguishes fused norm variants for PTX generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FusedKind {
    LayerNormRelu,
    RmsNormSilu,
}

impl FusedKind {
    fn tag(self) -> &'static str {
        match self {
            Self::LayerNormRelu => "ln_relu",
            Self::RmsNormSilu => "rms_silu",
        }
    }

    fn needs_beta(self) -> bool {
        match self {
            Self::LayerNormRelu => true,
            Self::RmsNormSilu => false,
        }
    }

    #[allow(dead_code)]
    fn needs_mean(self) -> bool {
        match self {
            Self::LayerNormRelu => true,
            Self::RmsNormSilu => false,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_fused_norm<T: GpuFloat>(
    handle: &DnnHandle,
    input: &TensorDesc<T>,
    gamma: Option<&DeviceBuffer<T>>,
    beta: Option<&DeviceBuffer<T>>,
    _residual: Option<&DeviceBuffer<T>>,
    output: &mut TensorDescMut<T>,
    epsilon: f32,
    kind: FusedKind,
) -> DnnResult<()> {
    let (num_rows, hidden_dim) = extract_row_dims(input)?;

    // Validate gamma
    if let Some(g) = gamma {
        if g.len() < hidden_dim as usize {
            return Err(DnnError::BufferTooSmall {
                expected: hidden_dim as usize * T::SIZE,
                actual: g.len() * T::SIZE,
            });
        }
    }
    // Validate beta (only for LayerNorm variants)
    if kind.needs_beta() {
        if let Some(b) = beta {
            if b.len() < hidden_dim as usize {
                return Err(DnnError::BufferTooSmall {
                    expected: hidden_dim as usize * T::SIZE,
                    actual: b.len() * T::SIZE,
                });
            }
        }
    }
    if output.numel() < input.numel() {
        return Err(DnnError::BufferTooSmall {
            expected: input.numel() * T::SIZE,
            actual: output.numel() * T::SIZE,
        });
    }

    let ptx_source = generate_fused_ptx::<T>(handle.sm_version(), hidden_dim, kind)?;
    let kernel_name = fused_kernel_name::<T>(hidden_dim, kind);
    let module = Arc::new(Module::from_ptx(&ptx_source).map_err(|e| {
        DnnError::LaunchFailed(format!("module load for fused_{}: {e}", kind.tag()))
    })?);
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| DnnError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;

    let block_size = if hidden_dim <= 1024 {
        hidden_dim.next_power_of_two().min(1024)
    } else {
        1024
    };
    let params = LaunchParams::new(num_rows, block_size);
    let eps_bits = epsilon.to_bits();

    let gamma_ptr = gamma.map(|g| g.as_device_ptr()).unwrap_or(0);
    let beta_ptr = beta.map(|b| b.as_device_ptr()).unwrap_or(0);

    let args = (
        input.ptr, output.ptr, gamma_ptr, beta_ptr, num_rows, hidden_dim, eps_bits,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| DnnError::LaunchFailed(format!("fused_{}: {e}", kind.tag())))?;

    Ok(())
}

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

fn fused_kernel_name<T: GpuFloat>(hidden_dim: u32, kind: FusedKind) -> String {
    format!("fused_{}_{}_d{}", kind.tag(), T::NAME, hidden_dim)
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

fn generate_fused_ptx<T: GpuFloat>(
    sm: SmVersion,
    hidden_dim: u32,
    kind: FusedKind,
) -> DnnResult<String> {
    let ty = T::PTX_TYPE.as_ptx_str();
    let byte_size = T::PTX_TYPE.size_bytes();
    let kernel_name = fused_kernel_name::<T>(hidden_dim, kind);
    let block_size = if hidden_dim <= 1024 {
        hidden_dim.next_power_of_two().min(1024)
    } else {
        1024
    };
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
    writeln!(ptx, "    .shared .align 4 .b8 smem_fn[{smem_bytes}];").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    // Thread / row indexing
    writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r2, [%param_n];").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p0, %r1, %r2;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p0 bra $FN_DONE;").map_err(fmt_err)?;

    writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd2, [%param_gamma];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd3, [%param_beta];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r3, [%param_d];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r4, [%param_epsilon_bits];").map_err(fmt_err)?;
    writeln!(ptx, "    mov.b32 %f20, %r4;").map_err(fmt_err)?;

    writeln!(ptx, "    cvt.u64.u32 %rd4, %r1;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd5, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd6, %rd4, %rd5;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    match kind {
        FusedKind::LayerNormRelu => {
            write_fused_layer_norm_relu(&mut ptx, ty, byte_size, hidden_dim, block_size)?;
        }
        FusedKind::RmsNormSilu => {
            write_fused_rms_norm_silu(&mut ptx, ty, byte_size, hidden_dim, block_size)?;
        }
    }

    writeln!(ptx, "$FN_DONE:").map_err(fmt_err)?;
    writeln!(ptx, "    ret;").map_err(fmt_err)?;
    writeln!(ptx, "}}").map_err(fmt_err)?;

    Ok(ptx)
}

/// Fused LayerNorm + ReLU.
fn write_fused_layer_norm_relu(
    ptx: &mut String,
    ty: &str,
    byte_size: usize,
    hidden_dim: u32,
    block_size: u32,
) -> DnnResult<()> {
    // Pass 1: sum for mean
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$FLR_SUM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p1 bra $FLR_SUM_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f1", "%rd9")?;
    writeln!(ptx, "    add.f32 %f0, %f0, %f1;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $FLR_SUM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$FLR_SUM_DONE:").map_err(fmt_err)?;

    write_smem_reduce_f32(ptx, "%f0", block_size, "FLR_SUM")?;
    writeln!(ptx, "    ld.shared.f32 %f2, [smem_fn];").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.rn.f32.u32 %f3, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f4, %f2, %f3;").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 2: variance
    writeln!(ptx, "    mov.f32 %f5, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$FLR_VAR_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p2, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p2 bra $FLR_VAR_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f6", "%rd9")?;
    writeln!(ptx, "    sub.f32 %f7, %f6, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    fma.rn.f32 %f5, %f7, %f7, %f5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $FLR_VAR_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$FLR_VAR_DONE:").map_err(fmt_err)?;

    write_smem_reduce_f32(ptx, "%f5", block_size, "FLR_VAR")?;
    writeln!(ptx, "    ld.shared.f32 %f8, [smem_fn];").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f8, %f8, %f3;").map_err(fmt_err)?;
    writeln!(ptx, "    add.f32 %f9, %f8, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f10, %f9;").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 3: normalize + scale + bias + ReLU
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$FLR_NORM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p3, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p3 bra $FN_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f11", "%rd9")?;
    writeln!(ptx, "    sub.f32 %f11, %f11, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.f32 %f11, %f11, %f10;").map_err(fmt_err)?;

    // gamma, beta
    writeln!(ptx, "    cvt.u64.u32 %rd10, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd10, %rd10, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd11, %rd2, %rd10;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f12", "%rd11")?;
    writeln!(ptx, "    add.u64 %rd12, %rd3, %rd10;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f13", "%rd12")?;
    writeln!(ptx, "    fma.rn.f32 %f14, %f11, %f12, %f13;").map_err(fmt_err)?;

    // ReLU: max(x, 0)
    writeln!(ptx, "    max.f32 %f14, %f14, 0f00000000;").map_err(fmt_err)?;

    writeln!(ptx, "    add.u64 %rd13, %rd1, %rd8;").map_err(fmt_err)?;
    store_global(ptx, ty, "%rd13", "%f14")?;
    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $FLR_NORM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    Ok(())
}

/// Fused RMSNorm + SiLU.
fn write_fused_rms_norm_silu(
    ptx: &mut String,
    ty: &str,
    byte_size: usize,
    hidden_dim: u32,
    block_size: u32,
) -> DnnResult<()> {
    // Pass 1: sum of squares
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$FRS_SQ_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p1 bra $FRS_SQ_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f1", "%rd9")?;
    writeln!(ptx, "    fma.rn.f32 %f0, %f1, %f1, %f0;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $FRS_SQ_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$FRS_SQ_DONE:").map_err(fmt_err)?;

    write_smem_reduce_f32(ptx, "%f0", block_size, "FRS_SQ")?;

    writeln!(ptx, "    ld.shared.f32 %f6, [smem_fn];").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.rn.f32.u32 %f5, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    div.approx.f32 %f6, %f6, %f5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.f32 %f6, %f6, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f7, %f6;").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 2: normalize + scale + SiLU
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$FRS_NORM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p3, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p3 bra $FN_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f8", "%rd9")?;

    // x_norm = x * inv_rms
    writeln!(ptx, "    mul.f32 %f8, %f8, %f7;").map_err(fmt_err)?;

    // gamma
    writeln!(ptx, "    cvt.u64.u32 %rd10, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd10, %rd10, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd11, %rd2, %rd10;").map_err(fmt_err)?;
    load_global(ptx, ty, "%f9", "%rd11")?;
    writeln!(ptx, "    mul.f32 %f10, %f8, %f9;").map_err(fmt_err)?;

    // SiLU: y = x * sigmoid(x) = x / (1 + exp(-x))
    // sigmoid(x) = 1 / (1 + exp(-x))
    // We compute: neg_x = -x; exp_neg = exp(neg_x); sigmoid = 1/(1+exp_neg)
    writeln!(ptx, "    neg.f32 %f11, %f10;").map_err(fmt_err)?;
    // exp via ex2: exp(x) = 2^(x * log2(e)); log2(e) = 0f3FB8AA3B
    writeln!(ptx, "    mul.f32 %f11, %f11, 0f3FB8AA3B;").map_err(fmt_err)?;
    writeln!(ptx, "    ex2.approx.f32 %f11, %f11;").map_err(fmt_err)?;
    writeln!(ptx, "    add.f32 %f12, %f11, 0f3F800000;").map_err(fmt_err)?; // 1 + exp(-x)
    writeln!(ptx, "    rcp.approx.f32 %f12, %f12;").map_err(fmt_err)?; // sigmoid
    writeln!(ptx, "    mul.f32 %f14, %f10, %f12;").map_err(fmt_err)?; // x * sigmoid(x)

    writeln!(ptx, "    add.u64 %rd13, %rd1, %rd8;").map_err(fmt_err)?;
    store_global(ptx, ty, "%rd13", "%f14")?;
    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $FRS_NORM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    Ok(())
}

fn write_smem_reduce_f32(
    ptx: &mut String,
    val_reg: &str,
    block_size: u32,
    tag: &str,
) -> DnnResult<()> {
    writeln!(ptx, "    cvt.u64.u32 %rd14, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd14, %rd14, 4;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u64 %rd15, smem_fn;").map_err(fmt_err)?;
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
    fn ptx_fused_ln_relu() {
        let ptx = generate_fused_ptx::<f32>(SmVersion::Sm80, 256, FusedKind::LayerNormRelu);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("fused_ln_relu_f32_d256"));
        assert!(ptx.contains("max.f32")); // ReLU
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn ptx_fused_rms_silu() {
        let ptx = generate_fused_ptx::<f32>(SmVersion::Sm80, 128, FusedKind::RmsNormSilu);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("fused_rms_silu_f32_d128"));
        assert!(ptx.contains("ex2.approx.f32")); // exp for sigmoid
        assert!(ptx.contains("rcp.approx.f32")); // 1/(1+exp)
    }
}
