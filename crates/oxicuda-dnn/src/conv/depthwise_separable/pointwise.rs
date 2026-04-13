//! PTX generation for the pointwise (1×1) convolution stage.

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};

use super::helpers::{emit_activation, emit_bn_epilogue, parse_precision};
use super::types::{ActivationType, DepthwiseSeparableConfig};

// ---------------------------------------------------------------------------
// PTX Generation — Pointwise (1x1) Conv Kernel
// ---------------------------------------------------------------------------

/// Parameters for the pointwise convolution PTX body.
#[derive(Debug, Clone, Copy)]
struct PointwiseBodyParams {
    float_type: PtxType,
    elem_bytes: u32,
    activation: ActivationType,
    has_bn: bool,
}

/// Generates PTX for the pointwise (1×1) convolution stage.
///
/// This is essentially a GEMM: `[batch*H*W, out_channels] = [batch*H*W, in_channels] × weight^T`
/// where `weight` is `[out_channels, in_channels]`. Each thread computes one
/// output element by accumulating over the input channel dimension.
///
/// # Errors
///
/// Returns [`DnnError::PtxGeneration`] if kernel building fails.
pub fn generate_pointwise_conv_ptx(
    config: &DepthwiseSeparableConfig,
    precision: &str,
    sm_version: SmVersion,
) -> DnnResult<String> {
    let (float_type, elem_bytes) = parse_precision(precision)?;
    let act_suffix = config.pointwise_activation.kernel_suffix();
    let kernel_name = format!("depthwise_separable_pw_{act_suffix}_{precision}");

    let in_channels = config.depthwise_out_channels() as u32;

    let params = PointwiseBodyParams {
        float_type,
        elem_bytes,
        activation: config.pointwise_activation,
        has_bn: config.pointwise_bn,
    };

    let mut builder = KernelBuilder::new(&kernel_name);
    builder = builder
        .target(sm_version)
        .param("input", PtxType::U64) // intermediate DW output
        .param("weight", PtxType::U64) // [out_channels, in_channels]
        .param("bias", PtxType::U64)
        .param("output", PtxType::U64)
        .param("in_channels", PtxType::U32)
        .param("out_channels", PtxType::U32)
        .param("spatial_size", PtxType::U32) // H * W
        .param("total_outputs", PtxType::U32);

    if config.pointwise_bn {
        builder = builder
            .param("bn_gamma", PtxType::U64)
            .param("bn_beta", PtxType::U64)
            .param("bn_mean", PtxType::U64)
            .param("bn_var", PtxType::U64);
    }

    let ptx = builder
        .body(move |b| {
            emit_pointwise_conv_body(b, in_channels, &params);
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Emits pointwise convolution body.
fn emit_pointwise_conv_body(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    _compile_time_in_ch: u32,
    p: &PointwiseBodyParams,
) {
    let float_type = p.float_type;
    let elem_bytes = p.elem_bytes;

    b.comment("=== Pointwise (1x1) Convolution Stage ===");
    b.comment("Each thread: one output element (batch*h*w, oc).");

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_outputs");

    let pred = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred}, {gid}, {total};"));
    let exit_label = b.fresh_label("pw_exit");
    b.raw_ptx(&format!("@!{pred} bra {exit_label};"));

    let input_ptr = b.load_param_u64("input");
    let weight_ptr = b.load_param_u64("weight");
    let output_ptr = b.load_param_u64("output");
    let in_channels = b.load_param_u32("in_channels");
    let out_channels = b.load_param_u32("out_channels");
    let spatial_size = b.load_param_u32("spatial_size");

    b.comment("Decompose gid -> (spatial_idx, oc) where linear = spatial_idx * out_channels + oc");
    let oc = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {oc}, {gid}, {out_channels};"));
    let spatial_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {spatial_idx}, {gid}, {out_channels};"));

    let _ = spatial_size; // used implicitly via total_outputs

    // Accumulator
    let acc = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        let zero = 0f32.to_bits();
        b.raw_ptx(&format!("mov.b32 {acc}, 0F{zero:08X};"));
    } else {
        let zero = 0f64.to_bits();
        b.raw_ptx(&format!("mov.b64 {acc}, 0D{zero:016X};"));
    }

    // Loop over in_channels
    let ic = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {ic}, 0;"));

    let loop_label = b.fresh_label("pw_loop");
    let loop_end = b.fresh_label("pw_loop_end");
    b.label(&loop_label);

    let pred_loop = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_loop}, {ic}, {in_channels};"));
    b.raw_ptx(&format!("@!{pred_loop} bra {loop_end};"));

    // input[spatial_idx * in_channels + ic]
    let in_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "mul.lo.u32 {in_idx}, {spatial_idx}, {in_channels};"
    ));
    b.raw_ptx(&format!("add.u32 {in_idx}, {in_idx}, {ic};"));

    let idx64 = b.alloc_reg(PtxType::U64);
    let off64 = b.alloc_reg(PtxType::U64);
    let addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {idx64}, {in_idx};"));
    b.raw_ptx(&format!("mul.lo.u64 {off64}, {idx64}, {elem_bytes};"));
    b.raw_ptx(&format!("add.u64 {addr}, {input_ptr}, {off64};"));

    let in_val = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("ld.global.f32 {in_val}, [{addr}];"));
    } else {
        b.raw_ptx(&format!("ld.global.f64 {in_val}, [{addr}];"));
    }

    // weight[oc * in_channels + ic]
    let w_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {w_idx}, {oc}, {in_channels};"));
    b.raw_ptx(&format!("add.u32 {w_idx}, {w_idx}, {ic};"));

    let w_idx64 = b.alloc_reg(PtxType::U64);
    let w_off64 = b.alloc_reg(PtxType::U64);
    let w_addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {w_idx64}, {w_idx};"));
    b.raw_ptx(&format!("mul.lo.u64 {w_off64}, {w_idx64}, {elem_bytes};"));
    b.raw_ptx(&format!("add.u64 {w_addr}, {weight_ptr}, {w_off64};"));

    let w_val = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("ld.global.f32 {w_val}, [{w_addr}];"));
        b.raw_ptx(&format!("fma.rn.f32 {acc}, {in_val}, {w_val}, {acc};"));
    } else {
        b.raw_ptx(&format!("ld.global.f64 {w_val}, [{w_addr}];"));
        b.raw_ptx(&format!("fma.rn.f64 {acc}, {in_val}, {w_val}, {acc};"));
    }

    b.raw_ptx(&format!("add.u32 {ic}, {ic}, 1;"));
    b.raw_ptx(&format!("bra {loop_label};"));
    b.label(&loop_end);

    // Optional BN
    if p.has_bn {
        emit_bn_epilogue(b, float_type, elem_bytes, &acc, oc);
    }

    // Activation
    let activated = emit_activation(b, float_type, acc, p.activation);

    // Store
    b.comment("Store pointwise output");
    let out_idx64 = b.alloc_reg(PtxType::U64);
    let out_off64 = b.alloc_reg(PtxType::U64);
    let out_addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {out_idx64}, {gid};"));
    b.raw_ptx(&format!(
        "mul.lo.u64 {out_off64}, {out_idx64}, {elem_bytes};"
    ));
    b.raw_ptx(&format!("add.u64 {out_addr}, {output_ptr}, {out_off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("st.global.f32 [{out_addr}], {activated};"));
    } else {
        b.raw_ptx(&format!("st.global.f64 [{out_addr}], {activated};"));
    }

    b.label(&exit_label);
    b.ret();
}
