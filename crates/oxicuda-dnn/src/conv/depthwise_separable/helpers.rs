//! Shared PTX helper functions for depthwise separable convolution kernels.

use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};

use super::types::ActivationType;

/// Parses precision string to PTX type and element size.
pub(super) fn parse_precision(precision: &str) -> DnnResult<(PtxType, u32)> {
    match precision {
        "f32" => Ok((PtxType::F32, 4)),
        "f64" => Ok((PtxType::F64, 8)),
        other => Err(DnnError::InvalidArgument(format!(
            "unsupported precision: {other}"
        ))),
    }
}

/// Emits inline activation PTX. Returns the register holding the result.
pub(super) fn emit_activation(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    float_type: PtxType,
    val: oxicuda_ptx::ir::Register,
    activation: ActivationType,
) -> oxicuda_ptx::ir::Register {
    match activation {
        ActivationType::None => val,
        ActivationType::Relu => {
            // max(0, val)
            let zero = alloc_zero(b, float_type);
            let result = alloc_float(b, float_type);
            if float_type == PtxType::F32 {
                b.raw_ptx(&format!("max.f32 {result}, {val}, {zero};"));
            } else {
                b.raw_ptx(&format!("max.f64 {result}, {val}, {zero};"));
            }
            result
        }
        ActivationType::Relu6 => {
            // min(6, max(0, val))
            let zero = alloc_zero(b, float_type);
            let six = alloc_float_const(b, float_type, 6.0);
            let clamped_lo = alloc_float(b, float_type);
            let result = alloc_float(b, float_type);
            if float_type == PtxType::F32 {
                b.raw_ptx(&format!("max.f32 {clamped_lo}, {val}, {zero};"));
                b.raw_ptx(&format!("min.f32 {result}, {clamped_lo}, {six};"));
            } else {
                b.raw_ptx(&format!("max.f64 {clamped_lo}, {val}, {zero};"));
                b.raw_ptx(&format!("min.f64 {result}, {clamped_lo}, {six};"));
            }
            result
        }
        ActivationType::Silu => {
            // x * sigmoid(x) = x / (1 + exp(-x))
            let neg_x = alloc_float(b, float_type);
            let exp_neg = alloc_float(b, float_type);
            let one = alloc_float_const(b, float_type, 1.0);
            let denom = alloc_float(b, float_type);
            let result = alloc_float(b, float_type);
            if float_type == PtxType::F32 {
                b.raw_ptx(&format!("neg.f32 {neg_x}, {val};"));
                b.raw_ptx(&format!("ex2.approx.f32 {exp_neg}, {neg_x};"));
                b.raw_ptx(&format!("add.rn.f32 {denom}, {exp_neg}, {one};"));
                b.raw_ptx(&format!("div.rn.f32 {result}, {val}, {denom};"));
            } else {
                // f64: use neg + approximate approach
                b.raw_ptx(&format!("neg.f64 {neg_x}, {val};"));
                // Convert to f32 for exp approx, then back
                let tmp32 = b.alloc_reg(PtxType::F32);
                let exp32 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.f64 {tmp32}, {neg_x};"));
                b.raw_ptx(&format!("ex2.approx.f32 {exp32}, {tmp32};"));
                b.raw_ptx(&format!("cvt.rn.f64.f32 {exp_neg}, {exp32};"));
                b.raw_ptx(&format!("add.rn.f64 {denom}, {exp_neg}, {one};"));
                b.raw_ptx(&format!("div.rn.f64 {result}, {val}, {denom};"));
            }
            result
        }
        ActivationType::HardSwish => {
            // x * min(6, max(0, x + 3)) / 6
            let three = alloc_float_const(b, float_type, 3.0);
            let six = alloc_float_const(b, float_type, 6.0);
            let zero = alloc_zero(b, float_type);
            let x_plus_3 = alloc_float(b, float_type);
            let clamped_lo = alloc_float(b, float_type);
            let clamped = alloc_float(b, float_type);
            let numerator = alloc_float(b, float_type);
            let result = alloc_float(b, float_type);
            if float_type == PtxType::F32 {
                b.raw_ptx(&format!("add.rn.f32 {x_plus_3}, {val}, {three};"));
                b.raw_ptx(&format!("max.f32 {clamped_lo}, {x_plus_3}, {zero};"));
                b.raw_ptx(&format!("min.f32 {clamped}, {clamped_lo}, {six};"));
                b.raw_ptx(&format!("mul.rn.f32 {numerator}, {val}, {clamped};"));
                b.raw_ptx(&format!("div.rn.f32 {result}, {numerator}, {six};"));
            } else {
                b.raw_ptx(&format!("add.rn.f64 {x_plus_3}, {val}, {three};"));
                b.raw_ptx(&format!("max.f64 {clamped_lo}, {x_plus_3}, {zero};"));
                b.raw_ptx(&format!("min.f64 {clamped}, {clamped_lo}, {six};"));
                b.raw_ptx(&format!("mul.rn.f64 {numerator}, {val}, {clamped};"));
                b.raw_ptx(&format!("div.rn.f64 {result}, {numerator}, {six};"));
            }
            result
        }
    }
}

/// Emits batch normalisation epilogue:
/// `val = gamma * (val - mean) / sqrt(var + eps) + beta`
///
/// Reads BN parameters from `bn_gamma`, `bn_beta`, `bn_mean`, `bn_var`
/// kernel params, indexed by `channel_reg`.
pub(super) fn emit_bn_epilogue(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    float_type: PtxType,
    elem_bytes: u32,
    val: &oxicuda_ptx::ir::Register,
    channel_reg: oxicuda_ptx::ir::Register,
) {
    b.comment("Batch Normalization epilogue");
    let gamma_ptr = b.load_param_u64("bn_gamma");
    let beta_ptr = b.load_param_u64("bn_beta");
    let mean_ptr = b.load_param_u64("bn_mean");
    let var_ptr = b.load_param_u64("bn_var");

    // Compute address offset for channel
    let ch64 = b.alloc_reg(PtxType::U64);
    let ch_off = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {ch64}, {channel_reg};"));
    b.raw_ptx(&format!("mul.lo.u64 {ch_off}, {ch64}, {elem_bytes};"));

    // Load gamma, beta, mean, var for this channel
    let addr = b.alloc_reg(PtxType::U64);
    let gamma = b.alloc_reg(float_type);
    let beta = b.alloc_reg(float_type);
    let mean = b.alloc_reg(float_type);
    let var = b.alloc_reg(float_type);

    let ld_suffix = if float_type == PtxType::F32 {
        "f32"
    } else {
        "f64"
    };

    b.raw_ptx(&format!("add.u64 {addr}, {gamma_ptr}, {ch_off};"));
    b.raw_ptx(&format!("ld.global.{ld_suffix} {gamma}, [{addr}];"));

    b.raw_ptx(&format!("add.u64 {addr}, {beta_ptr}, {ch_off};"));
    b.raw_ptx(&format!("ld.global.{ld_suffix} {beta}, [{addr}];"));

    b.raw_ptx(&format!("add.u64 {addr}, {mean_ptr}, {ch_off};"));
    b.raw_ptx(&format!("ld.global.{ld_suffix} {mean}, [{addr}];"));

    b.raw_ptx(&format!("add.u64 {addr}, {var_ptr}, {ch_off};"));
    b.raw_ptx(&format!("ld.global.{ld_suffix} {var}, [{addr}];"));

    // eps = 1e-5
    let eps = alloc_float_const(b, float_type, 1e-5);

    // var_eps = var + eps
    let var_eps = alloc_float(b, float_type);
    // inv_std = rsqrt(var_eps)
    let inv_std = alloc_float(b, float_type);
    // x_minus_mean = val - mean
    let x_minus_mean = alloc_float(b, float_type);
    // normed = x_minus_mean * inv_std
    let normed = alloc_float(b, float_type);

    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("add.rn.f32 {var_eps}, {var}, {eps};"));
        b.raw_ptx(&format!("rsqrt.approx.f32 {inv_std}, {var_eps};"));
        b.raw_ptx(&format!("sub.rn.f32 {x_minus_mean}, {val}, {mean};"));
        b.raw_ptx(&format!("mul.rn.f32 {normed}, {x_minus_mean}, {inv_std};"));
        // val = gamma * normed + beta
        b.raw_ptx(&format!("fma.rn.f32 {val}, {gamma}, {normed}, {beta};"));
    } else {
        b.raw_ptx(&format!("add.rn.f64 {var_eps}, {var}, {eps};"));
        // f64 doesn't have rsqrt, use rcp + sqrt
        let sqrt_val = alloc_float(b, float_type);
        b.raw_ptx(&format!("sqrt.rn.f64 {sqrt_val}, {var_eps};"));
        b.raw_ptx(&format!("rcp.rn.f64 {inv_std}, {sqrt_val};"));
        b.raw_ptx(&format!("sub.rn.f64 {x_minus_mean}, {val}, {mean};"));
        b.raw_ptx(&format!("mul.rn.f64 {normed}, {x_minus_mean}, {inv_std};"));
        b.raw_ptx(&format!("fma.rn.f64 {val}, {gamma}, {normed}, {beta};"));
    }
}

/// Allocates a zero-initialised float register.
pub(super) fn alloc_zero(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    float_type: PtxType,
) -> oxicuda_ptx::ir::Register {
    let r = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        let bits = 0f32.to_bits();
        b.raw_ptx(&format!("mov.b32 {r}, 0F{bits:08X};"));
    } else {
        let bits = 0f64.to_bits();
        b.raw_ptx(&format!("mov.b64 {r}, 0D{bits:016X};"));
    }
    r
}

/// Allocates a float register initialised to a compile-time constant.
pub(super) fn alloc_float_const(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    float_type: PtxType,
    val: f64,
) -> oxicuda_ptx::ir::Register {
    let r = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        let bits = (val as f32).to_bits();
        b.raw_ptx(&format!("mov.b32 {r}, 0F{bits:08X};"));
    } else {
        let bits = val.to_bits();
        b.raw_ptx(&format!("mov.b64 {r}, 0D{bits:016X};"));
    }
    r
}

/// Allocates an uninitialised float register.
pub(super) fn alloc_float(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    float_type: PtxType,
) -> oxicuda_ptx::ir::Register {
    b.alloc_reg(float_type)
}
