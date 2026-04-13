//! Scale Normalization for efficient transformers.
//!
//! A simplified normalization layer that replaces LayerNorm with a
//! computationally cheaper alternative:
//!
//! ```text
//! y = g * x / ||x||_2
//! ```
//!
//! where `g` is a single learned scalar and `||x||_2` is the L2 norm of the
//! input vector. Unlike LayerNorm, there is no mean subtraction and no
//! per-element affine parameters -- only a scalar scale.
//!
//! ScaleNorm was proposed for efficient transformers where normalization
//! overhead is significant (e.g. in very deep or very wide models). It
//! provides comparable training stability to LayerNorm with fewer parameters
//! and lower computational cost.
//!
//! Reference: Nguyen & Salazar, "Transformers without Tears: Improving the
//! Normalization of Self-Attention" (2019).

use std::fmt::Write as FmtWrite;

use oxicuda_blas::GpuFloat;
use oxicuda_ptx::arch::SmVersion;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Scale Normalization.
#[derive(Debug, Clone)]
pub struct ScaleNormConfig {
    /// Hidden dimension (length of the vector to normalise).
    pub hidden_size: u32,
    /// Small constant added to the L2 norm for numerical stability.
    pub epsilon: f32,
}

impl ScaleNormConfig {
    /// Validates this configuration.
    pub fn validate(&self) -> DnnResult<()> {
        if self.hidden_size == 0 {
            return Err(DnnError::InvalidArgument("hidden_size must be > 0".into()));
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

/// Execution plan for Scale Normalization.
///
/// Pre-generates PTX for the configured hidden size and data type.
/// Each thread block handles one row (vector) of the input.
#[derive(Debug)]
pub struct ScaleNormPlan {
    config: ScaleNormConfig,
    forward_ptx: String,
}

impl ScaleNormPlan {
    /// Creates a new execution plan.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError`] if the configuration is invalid or PTX generation
    /// fails.
    pub fn new<T: GpuFloat>(config: ScaleNormConfig, sm: SmVersion) -> DnnResult<Self> {
        config.validate()?;
        let forward_ptx = generate_forward_ptx::<T>(&config, sm)?;
        Ok(Self {
            config,
            forward_ptx,
        })
    }

    /// Returns the generated forward-pass PTX source.
    pub fn forward_ptx(&self) -> &str {
        &self.forward_ptx
    }

    /// Returns a reference to the plan's configuration.
    pub fn config(&self) -> &ScaleNormConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

fn scale_norm_kernel_name<T: GpuFloat>(hidden: u32) -> String {
    format!("scale_norm_fwd_{}_d{hidden}", T::NAME)
}

/// Generates PTX for the ScaleNorm forward kernel.
///
/// Kernel parameters:
///   - `input`          (u64) -- input tensor [N, D]
///   - `output`         (u64) -- output tensor [N, D]
///   - `g_scalar`       (u64) -- pointer to learned scalar g (single element)
///   - `n`              (u32) -- number of rows
///   - `d`              (u32) -- hidden dimension
///   - `epsilon_bits`   (u32) -- epsilon as f32 bit pattern
///
/// Grid: one block per row.
/// Each thread accumulates partial sum of x^2 via strided loop, then
/// reduces to get ||x||_2. Finally normalizes: y = g * x / ||x||_2.
pub fn generate_forward_ptx<T: GpuFloat>(
    config: &ScaleNormConfig,
    sm: SmVersion,
) -> DnnResult<String> {
    let ty = T::PTX_TYPE.as_ptx_str();
    let byte_size = T::PTX_TYPE.size_bytes();
    let hidden_dim = config.hidden_size;
    let kernel_name = scale_norm_kernel_name::<T>(hidden_dim);
    let use_warp = hidden_dim <= 32;
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
    writeln!(ptx, "    .param .u64 %param_g,").map_err(fmt_err)?;
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
        writeln!(ptx, "    .shared .align 4 .b8 smem_sn[{smem_bytes}];").map_err(fmt_err)?;
    }
    writeln!(ptx).map_err(fmt_err)?;

    // Row index = blockIdx.x, thread = threadIdx.x
    writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r2, [%param_n];").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p0, %r1, %r2;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p0 bra $SN_DONE;").map_err(fmt_err)?;

    // Load params
    writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd2, [%param_g];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r3, [%param_d];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r4, [%param_epsilon_bits];").map_err(fmt_err)?;
    writeln!(ptx, "    mov.b32 %f20, %r4;").map_err(fmt_err)?; // epsilon

    // Load the learned scalar g
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f21, [%rd2];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f21, [%rd2];").map_err(fmt_err)?;
    }

    // Compute row base offset: row_offset = blockIdx.x * D
    writeln!(ptx, "    cvt.u64.u32 %rd4, %r1;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd5, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd6, %rd4, %rd5;").map_err(fmt_err)?;
    // rd6 = row element offset

    if use_warp {
        write_warp_scale_norm(&mut ptx, ty, byte_size, hidden_dim)?;
    } else {
        write_block_scale_norm(&mut ptx, ty, byte_size, hidden_dim, block_size)?;
    }

    writeln!(ptx, "$SN_DONE:").map_err(fmt_err)?;
    writeln!(ptx, "    ret;").map_err(fmt_err)?;
    writeln!(ptx, "}}").map_err(fmt_err)?;

    Ok(ptx)
}

/// Warp-level ScaleNorm for hidden_dim <= 32.
fn write_warp_scale_norm(
    ptx: &mut String,
    ty: &str,
    byte_size: usize,
    hidden_dim: u32,
) -> DnnResult<()> {
    writeln!(ptx, "    // Warp-level ScaleNorm (D <= 32)").map_err(fmt_err)?;
    writeln!(ptx, "    setp.lt.u32 %p1, %r0, {hidden_dim};").map_err(fmt_err)?;

    // Load input element (zero for out-of-range lanes)
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    @!%p1 bra $SN_WARP_NORM;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f0, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f0, [%rd9];").map_err(fmt_err)?;
    }
    writeln!(ptx, "$SN_WARP_NORM:").map_err(fmt_err)?;

    // Compute sum of squares via warp shuffle
    writeln!(ptx, "    mul.f32 %f1, %f0, %f0;").map_err(fmt_err)?; // x^2
    writeln!(ptx, "    mov.f32 %f2, %f1;").map_err(fmt_err)?;
    for offset in [16u32, 8, 4, 2, 1] {
        writeln!(
            ptx,
            "    shfl.sync.down.b32 %f3, %f2, {offset}, 31, 0xFFFFFFFF;"
        )
        .map_err(fmt_err)?;
        writeln!(ptx, "    add.f32 %f2, %f2, %f3;").map_err(fmt_err)?;
    }
    // Broadcast sum_sq and compute L2 norm
    writeln!(ptx, "    shfl.sync.idx.b32 %f2, %f2, 0, 31, 0xFFFFFFFF;").map_err(fmt_err)?;
    // norm = sqrt(sum_sq + eps), then inv_norm = 1/norm = rsqrt(sum_sq + eps)
    writeln!(ptx, "    add.f32 %f4, %f2, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f5, %f4;").map_err(fmt_err)?; // 1/||x||

    // Normalize: y = g * x / ||x|| = g * x * inv_norm
    writeln!(ptx, "    @!%p1 bra $SN_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.f32 %f6, %f21, %f5;").map_err(fmt_err)?; // g * inv_norm
    writeln!(ptx, "    mul.f32 %f7, %f0, %f6;").map_err(fmt_err)?; // x * (g * inv_norm)

    // Store
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd13, %rd1, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    st.global.f32 [%rd13], %f7;").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    st.global{ty} [%rd13], %f7;").map_err(fmt_err)?;
    }
    writeln!(ptx).map_err(fmt_err)?;

    Ok(())
}

/// Block-level ScaleNorm for hidden_dim > 32.
fn write_block_scale_norm(
    ptx: &mut String,
    ty: &str,
    byte_size: usize,
    hidden_dim: u32,
    block_size: u32,
) -> DnnResult<()> {
    writeln!(ptx, "    // Block-level ScaleNorm (D > 32)").map_err(fmt_err)?;

    // Pass 1: accumulate sum of squares via strided loop
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?; // partial sq sum
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$SN_SQ_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p1 bra $SN_SQ_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f1, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f1, [%rd9];").map_err(fmt_err)?;
    }
    writeln!(ptx, "    fma.rn.f32 %f0, %f1, %f1, %f0;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $SN_SQ_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$SN_SQ_DONE:").map_err(fmt_err)?;

    // Reduce sum of squares via shared memory
    write_smem_reduce_f32(ptx, "%f0", block_size, "SQ")?;

    // Compute inv_norm = rsqrt(sum_sq + eps)
    writeln!(ptx, "    ld.shared.f32 %f2, [smem_sn];").map_err(fmt_err)?;
    writeln!(ptx, "    add.f32 %f3, %f2, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rsqrt.approx.f32 %f4, %f3;").map_err(fmt_err)?; // 1/||x||
    // Precompute g * inv_norm
    writeln!(ptx, "    mul.f32 %f5, %f21, %f4;").map_err(fmt_err)?;
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 2: normalize and store
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$SN_NORM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p2, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p2 bra $SN_DONE;").map_err(fmt_err)?;

    // Reload x
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f6, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f6, [%rd9];").map_err(fmt_err)?;
    }

    // y = x * (g * inv_norm)
    writeln!(ptx, "    mul.f32 %f7, %f6, %f5;").map_err(fmt_err)?;

    // Store
    writeln!(ptx, "    add.u64 %rd13, %rd1, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    st.global.f32 [%rd13], %f7;").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    st.global{ty} [%rd13], %f7;").map_err(fmt_err)?;
    }
    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $SN_NORM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Shared memory reduction helper
// ---------------------------------------------------------------------------

fn write_smem_reduce_f32(
    ptx: &mut String,
    val_reg: &str,
    block_size: u32,
    tag: &str,
) -> DnnResult<()> {
    writeln!(ptx, "    // Shared memory reduction ({tag})").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd14, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd14, %rd14, 4;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u64 %rd15, smem_sn;").map_err(fmt_err)?;
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

    // -- Config validation --

    #[test]
    fn config_valid() {
        let cfg = ScaleNormConfig {
            hidden_size: 512,
            epsilon: 1e-5,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_zero_hidden() {
        let cfg = ScaleNormConfig {
            hidden_size: 0,
            epsilon: 1e-5,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_negative_epsilon() {
        let cfg = ScaleNormConfig {
            hidden_size: 512,
            epsilon: -1.0,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_inf_epsilon() {
        let cfg = ScaleNormConfig {
            hidden_size: 512,
            epsilon: f32::INFINITY,
        };
        assert!(cfg.validate().is_err());
    }

    // -- PTX generation --

    fn make_config(hidden: u32) -> ScaleNormConfig {
        ScaleNormConfig {
            hidden_size: hidden,
            epsilon: 1e-6,
        }
    }

    #[test]
    fn forward_ptx_f32_warp() {
        let cfg = make_config(16);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry scale_norm_fwd_f32_d16"));
        assert!(ptx.contains("shfl.sync"));
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn forward_ptx_f32_block() {
        let cfg = make_config(256);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry scale_norm_fwd_f32_d256"));
        assert!(ptx.contains("smem_sn"));
        assert!(ptx.contains("bar.sync"));
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn forward_ptx_f64() {
        let cfg = make_config(128);
        let ptx = generate_forward_ptx::<f64>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry scale_norm_fwd_f64_d128"));
    }

    #[test]
    fn forward_ptx_large_hidden() {
        let cfg = make_config(4096);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("scale_norm_fwd_f32_d4096"));
    }

    #[test]
    fn forward_ptx_no_mean_subtraction() {
        // ScaleNorm should NOT have mean computation -- only L2 norm
        let cfg = make_config(64);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80).unwrap_or_default();
        // No sub.f32 for mean subtraction (unlike LayerNorm)
        // ScaleNorm only computes sum of squares and divides -- no mean pass
        assert!(!ptx.contains("sub.f32"));
        // Should have rsqrt for 1/||x||
        assert!(ptx.contains("rsqrt.approx.f32"));
    }

    // -- Plan tests --

    #[test]
    fn plan_creation_success() {
        let cfg = ScaleNormConfig {
            hidden_size: 768,
            epsilon: 1e-5,
        };
        let plan = ScaleNormPlan::new::<f32>(cfg, SmVersion::Sm80);
        assert!(plan.is_ok());
        let plan = plan.unwrap_or_else(|e| panic!("plan creation failed: {e}"));
        assert!(plan.forward_ptx().contains("scale_norm_fwd"));
        assert_eq!(plan.config().hidden_size, 768);
    }

    #[test]
    fn plan_creation_invalid() {
        let cfg = ScaleNormConfig {
            hidden_size: 0,
            epsilon: 1e-5,
        };
        assert!(ScaleNormPlan::new::<f32>(cfg, SmVersion::Sm80).is_err());
    }

    #[test]
    fn forward_ptx_various_hidden_sizes() {
        for dim in [1, 8, 32, 64, 128, 512, 1024, 2048] {
            let cfg = make_config(dim);
            let result = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
            assert!(
                result.is_ok(),
                "failed for hidden_size={dim}: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn forward_ptx_epsilon_param_present() {
        let cfg = make_config(64);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80).unwrap_or_default();
        assert!(ptx.contains("param_epsilon_bits"));
        assert!(ptx.contains("mov.b32 %f20"));
    }

    #[test]
    fn forward_ptx_loads_scalar_g() {
        let cfg = make_config(64);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80).unwrap_or_default();
        assert!(ptx.contains("param_g"));
        assert!(ptx.contains("ld.global.f32 %f21"));
    }
}
