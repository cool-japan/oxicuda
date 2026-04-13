//! Power Normalization for improved training stability.
//!
//! Implements running power mean normalization, which maintains an exponential
//! moving average of power statistics for more stable deep network training.
//!
//! ```text
//! power_mean = E[|x|^p]^(1/p)
//! y = gamma * x / power_mean + beta
//! ```
//!
//! where `p` is a configurable power parameter (typically 1.0 or 2.0).
//! When p=2, this reduces to RMS-style normalization. When p=1, it uses
//! the mean absolute value.
//!
//! PowerNorm provides improved training stability compared to LayerNorm
//! for very deep transformer models by using a more robust estimate of
//! the input scale via power means.
//!
//! Reference: Shen et al., "PowerNorm: Rethinking Batch Normalization in
//! Transformers" (2020).

use std::fmt::Write as FmtWrite;

use oxicuda_blas::GpuFloat;
use oxicuda_ptx::arch::SmVersion;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Power Normalization.
#[derive(Debug, Clone)]
pub struct PowerNormConfig {
    /// Hidden dimension (length of the vector to normalise).
    pub hidden_size: u32,
    /// Small constant for numerical stability.
    pub epsilon: f32,
    /// Power parameter for the power mean (typically 1.0 or 2.0).
    ///
    /// - p = 1.0: mean absolute value normalization
    /// - p = 2.0: root mean square normalization (similar to RMSNorm)
    /// - Other positive values: generalized power mean
    pub power: f32,
}

impl PowerNormConfig {
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
        if self.power <= 0.0 {
            return Err(DnnError::InvalidArgument("power must be positive".into()));
        }
        if !self.power.is_finite() {
            return Err(DnnError::InvalidArgument("power must be finite".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// Execution plan for Power Normalization.
///
/// Pre-generates PTX for the configured hidden size, power, and data type.
/// Each thread block handles one row of the input tensor.
#[derive(Debug)]
pub struct PowerNormPlan {
    config: PowerNormConfig,
    forward_ptx: String,
}

impl PowerNormPlan {
    /// Creates a new execution plan.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError`] if the configuration is invalid or PTX generation
    /// fails.
    pub fn new<T: GpuFloat>(config: PowerNormConfig, sm: SmVersion) -> DnnResult<Self> {
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
    pub fn config(&self) -> &PowerNormConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

fn power_norm_kernel_name<T: GpuFloat>(hidden: u32) -> String {
    format!("power_norm_fwd_{}_d{hidden}", T::NAME)
}

/// Generates PTX for the PowerNorm forward kernel.
///
/// Kernel parameters:
///   - `input`          (u64) -- input tensor [N, D]
///   - `output`         (u64) -- output tensor [N, D]
///   - `gamma`          (u64) -- per-element scale (length D)
///   - `beta`           (u64) -- per-element bias (length D)
///   - `n`              (u32) -- number of rows
///   - `d`              (u32) -- hidden dimension
///   - `epsilon_bits`   (u32) -- epsilon as f32 bit pattern
///   - `power_bits`     (u32) -- power as f32 bit pattern
///   - `inv_power_bits` (u32) -- (1/power) as f32 bit pattern
///
/// Grid: one block per row.
///
/// Algorithm:
///   1. Compute sum(|x_i|^p) over the row via strided loop + reduction.
///   2. power_mean = (sum / D)^(1/p).
///   3. Normalize: y = gamma * x / (power_mean + eps) + beta.
///
/// For p=2, |x|^p = x*x and (sum/D)^(1/p) = sqrt(sum/D), which is
/// equivalent to RMSNorm. For p=1, it is the mean absolute value.
/// For general p, we use `ex2` / `lg2` to compute |x|^p and result^(1/p).
pub fn generate_forward_ptx<T: GpuFloat>(
    config: &PowerNormConfig,
    sm: SmVersion,
) -> DnnResult<String> {
    let ty = T::PTX_TYPE.as_ptx_str();
    let byte_size = T::PTX_TYPE.size_bytes();
    let hidden_dim = config.hidden_size;
    let power = config.power;
    let kernel_name = power_norm_kernel_name::<T>(hidden_dim);
    let block_size = if hidden_dim <= 1024 {
        hidden_dim.next_power_of_two().min(1024)
    } else {
        1024
    };
    let smem_bytes = block_size as usize * 4;

    // Determine if we can use a specialised path
    let is_p2 = (power - 2.0).abs() < 1e-6;
    let is_p1 = (power - 1.0).abs() < 1e-6;

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
    writeln!(ptx, "    .param .u32 %param_epsilon_bits,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_power_bits,").map_err(fmt_err)?;
    writeln!(ptx, "    .param .u32 %param_inv_power_bits").map_err(fmt_err)?;
    writeln!(ptx, ")").map_err(fmt_err)?;
    writeln!(ptx, "{{").map_err(fmt_err)?;
    writeln!(ptx, "    .maxntid {block_size}, 1, 1;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .b32 %r<32>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .b64 %rd<16>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .f32 %f<32>;").map_err(fmt_err)?;
    writeln!(ptx, "    .reg .pred %p<8>;").map_err(fmt_err)?;
    writeln!(ptx, "    .shared .align 4 .b8 smem_pn[{smem_bytes}];").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    // Row index = blockIdx.x
    writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r2, [%param_n];").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p0, %r1, %r2;").map_err(fmt_err)?;
    writeln!(ptx, "    @%p0 bra $PN_DONE;").map_err(fmt_err)?;

    // Load params
    writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd2, [%param_gamma];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u64 %rd3, [%param_beta];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r3, [%param_d];").map_err(fmt_err)?;
    writeln!(ptx, "    ld.param.u32 %r4, [%param_epsilon_bits];").map_err(fmt_err)?;
    writeln!(ptx, "    mov.b32 %f20, %r4;").map_err(fmt_err)?; // epsilon
    writeln!(ptx, "    ld.param.u32 %r6, [%param_power_bits];").map_err(fmt_err)?;
    writeln!(ptx, "    mov.b32 %f25, %r6;").map_err(fmt_err)?; // power
    writeln!(ptx, "    ld.param.u32 %r7, [%param_inv_power_bits];").map_err(fmt_err)?;
    writeln!(ptx, "    mov.b32 %f26, %r7;").map_err(fmt_err)?; // 1/power

    // Row base offset
    writeln!(ptx, "    cvt.u64.u32 %rd4, %r1;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd5, %r3;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd6, %rd4, %rd5;").map_err(fmt_err)?;
    // rd6 = row element offset

    // Pass 1: accumulate sum(|x|^p) via strided loop
    writeln!(ptx, "    mov.f32 %f0, 0f00000000;").map_err(fmt_err)?;
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$PN_POW_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p1, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p1 bra $PN_POW_DONE;").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f1, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f1, [%rd9];").map_err(fmt_err)?;
    }

    // Compute |x|^p
    writeln!(ptx, "    abs.f32 %f2, %f1;").map_err(fmt_err)?; // |x|
    if is_p2 {
        // p=2: |x|^2 = x*x (avoid log/exp overhead)
        writeln!(ptx, "    mul.f32 %f3, %f2, %f2;").map_err(fmt_err)?;
    } else if is_p1 {
        // p=1: |x|^1 = |x|
        writeln!(ptx, "    mov.f32 %f3, %f2;").map_err(fmt_err)?;
    } else {
        // General: |x|^p = exp2(p * log2(|x|))
        // Guard against log2(0): add epsilon to |x| before log
        writeln!(ptx, "    add.f32 %f27, %f2, %f20;").map_err(fmt_err)?;
        writeln!(ptx, "    lg2.approx.f32 %f28, %f27;").map_err(fmt_err)?;
        writeln!(ptx, "    mul.f32 %f28, %f28, %f25;").map_err(fmt_err)?; // p * log2(|x|+eps)
        writeln!(ptx, "    ex2.approx.f32 %f3, %f28;").map_err(fmt_err)?;
    }
    writeln!(ptx, "    add.f32 %f0, %f0, %f3;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u32 %r5, %r5, {block_size};").map_err(fmt_err)?;
    writeln!(ptx, "    bra $PN_POW_LOOP;").map_err(fmt_err)?;
    writeln!(ptx, "$PN_POW_DONE:").map_err(fmt_err)?;

    // Reduce sum(|x|^p) via shared memory
    write_smem_reduce_f32(&mut ptx, "%f0", block_size, "POW")?;

    // Compute power_mean = (sum / D)^(1/p)
    writeln!(ptx, "    ld.shared.f32 %f4, [smem_pn];").map_err(fmt_err)?;
    writeln!(ptx, "    cvt.rn.f32.u32 %f5, %r3;").map_err(fmt_err)?; // D as float
    writeln!(ptx, "    div.approx.f32 %f6, %f4, %f5;").map_err(fmt_err)?; // mean of |x|^p
    if is_p2 {
        // (mean)^(1/2) = sqrt(mean)
        writeln!(ptx, "    sqrt.approx.f32 %f7, %f6;").map_err(fmt_err)?;
    } else if is_p1 {
        // (mean)^1 = mean
        writeln!(ptx, "    mov.f32 %f7, %f6;").map_err(fmt_err)?;
    } else {
        // General: mean^(1/p) = exp2((1/p) * log2(mean))
        writeln!(ptx, "    add.f32 %f29, %f6, %f20;").map_err(fmt_err)?;
        writeln!(ptx, "    lg2.approx.f32 %f29, %f29;").map_err(fmt_err)?;
        writeln!(ptx, "    mul.f32 %f29, %f29, %f26;").map_err(fmt_err)?; // (1/p) * log2(mean+eps)
        writeln!(ptx, "    ex2.approx.f32 %f7, %f29;").map_err(fmt_err)?;
    }

    // inv_power_mean = 1 / (power_mean + eps)
    writeln!(ptx, "    add.f32 %f8, %f7, %f20;").map_err(fmt_err)?;
    writeln!(ptx, "    rcp.approx.f32 %f9, %f8;").map_err(fmt_err)?; // 1/(power_mean + eps)
    writeln!(ptx, "    bar.sync 0;").map_err(fmt_err)?;

    // Pass 2: normalize, scale, bias, store
    writeln!(ptx, "    mov.u32 %r5, %r0;").map_err(fmt_err)?;
    writeln!(ptx, "$PN_NORM_LOOP:").map_err(fmt_err)?;
    writeln!(ptx, "    setp.ge.u32 %p2, %r5, {hidden_dim};").map_err(fmt_err)?;
    writeln!(ptx, "    @%p2 bra $PN_DONE;").map_err(fmt_err)?;

    // Reload x
    writeln!(ptx, "    cvt.u64.u32 %rd8, %r5;").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd8, %rd6, %rd8;").map_err(fmt_err)?;
    writeln!(ptx, "    mul.lo.u64 %rd8, %rd8, {byte_size};").map_err(fmt_err)?;
    writeln!(ptx, "    add.u64 %rd9, %rd0, %rd8;").map_err(fmt_err)?;
    if ty == ".f32" {
        writeln!(ptx, "    ld.global.f32 %f10, [%rd9];").map_err(fmt_err)?;
    } else {
        writeln!(ptx, "    ld.global{ty} %f10, [%rd9];").map_err(fmt_err)?;
    }

    // x_norm = x * inv_power_mean
    writeln!(ptx, "    mul.f32 %f11, %f10, %f9;").map_err(fmt_err)?;

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
    writeln!(ptx, "    bra $PN_NORM_LOOP;").map_err(fmt_err)?;
    writeln!(ptx).map_err(fmt_err)?;

    writeln!(ptx, "$PN_DONE:").map_err(fmt_err)?;
    writeln!(ptx, "    ret;").map_err(fmt_err)?;
    writeln!(ptx, "}}").map_err(fmt_err)?;

    Ok(ptx)
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
    writeln!(ptx, "    mov.u64 %rd15, smem_pn;").map_err(fmt_err)?;
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
    fn config_valid_p2() {
        let cfg = PowerNormConfig {
            hidden_size: 512,
            epsilon: 1e-5,
            power: 2.0,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_valid_p1() {
        let cfg = PowerNormConfig {
            hidden_size: 256,
            epsilon: 1e-5,
            power: 1.0,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_zero_hidden() {
        let cfg = PowerNormConfig {
            hidden_size: 0,
            epsilon: 1e-5,
            power: 2.0,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_negative_epsilon() {
        let cfg = PowerNormConfig {
            hidden_size: 512,
            epsilon: -1.0,
            power: 2.0,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_zero_power() {
        let cfg = PowerNormConfig {
            hidden_size: 512,
            epsilon: 1e-5,
            power: 0.0,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_negative_power() {
        let cfg = PowerNormConfig {
            hidden_size: 512,
            epsilon: 1e-5,
            power: -1.0,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_inf_power() {
        let cfg = PowerNormConfig {
            hidden_size: 512,
            epsilon: 1e-5,
            power: f32::INFINITY,
        };
        assert!(cfg.validate().is_err());
    }

    // -- PTX generation --

    fn make_config(hidden: u32, power: f32) -> PowerNormConfig {
        PowerNormConfig {
            hidden_size: hidden,
            epsilon: 1e-5,
            power,
        }
    }

    #[test]
    fn forward_ptx_f32_p2() {
        let cfg = make_config(256, 2.0);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry power_norm_fwd_f32_d256"));
        assert!(ptx.contains("smem_pn"));
        assert!(ptx.contains("bar.sync"));
        // p=2: should use sqrt, not lg2/ex2
        assert!(ptx.contains("sqrt.approx.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
    }

    #[test]
    fn forward_ptx_f32_p1() {
        let cfg = make_config(128, 1.0);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry power_norm_fwd_f32_d128"));
        // p=1: |x|^1 = |x|, no mul for squaring
        assert!(ptx.contains("abs.f32"));
    }

    #[test]
    fn forward_ptx_f32_general_power() {
        let cfg = make_config(64, 1.5);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        // General power path uses lg2 + ex2
        assert!(ptx.contains("lg2.approx.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
    }

    #[test]
    fn forward_ptx_f64() {
        let cfg = make_config(512, 2.0);
        let ptx = generate_forward_ptx::<f64>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains(".entry power_norm_fwd_f64_d512"));
    }

    #[test]
    fn forward_ptx_large_hidden() {
        let cfg = make_config(4096, 2.0);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.unwrap_or_default();
        assert!(ptx.contains("power_norm_fwd_f32_d4096"));
    }

    // -- Plan tests --

    #[test]
    fn plan_creation_p2() {
        let cfg = PowerNormConfig {
            hidden_size: 768,
            epsilon: 1e-5,
            power: 2.0,
        };
        let plan = PowerNormPlan::new::<f32>(cfg, SmVersion::Sm80);
        assert!(plan.is_ok());
        let plan = plan.unwrap_or_else(|e| panic!("plan creation failed: {e}"));
        assert!(plan.forward_ptx().contains("power_norm_fwd"));
        assert_eq!(plan.config().hidden_size, 768);
    }

    #[test]
    fn plan_creation_invalid() {
        let cfg = PowerNormConfig {
            hidden_size: 0,
            epsilon: 1e-5,
            power: 2.0,
        };
        assert!(PowerNormPlan::new::<f32>(cfg, SmVersion::Sm80).is_err());
    }

    #[test]
    fn forward_ptx_various_hidden_sizes() {
        for dim in [1, 32, 64, 128, 512, 1024, 2048] {
            let cfg = make_config(dim, 2.0);
            let result = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80);
            assert!(
                result.is_ok(),
                "failed for hidden_size={dim}: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn forward_ptx_epsilon_and_power_params() {
        let cfg = make_config(64, 2.0);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80).unwrap_or_default();
        assert!(ptx.contains("param_epsilon_bits"));
        assert!(ptx.contains("param_power_bits"));
        assert!(ptx.contains("param_inv_power_bits"));
    }

    #[test]
    fn forward_ptx_has_affine_params() {
        let cfg = make_config(64, 2.0);
        let ptx = generate_forward_ptx::<f32>(&cfg, SmVersion::Sm80).unwrap_or_default();
        assert!(ptx.contains("param_gamma"));
        assert!(ptx.contains("param_beta"));
        assert!(ptx.contains("fma.rn.f32"));
    }
}
