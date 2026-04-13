//! PTX generation for the depthwise convolution stage.

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};

use super::helpers::{emit_activation, emit_bn_epilogue, parse_precision};
use super::types::{ActivationType, DepthwiseSeparableConfig};

// ---------------------------------------------------------------------------
// PTX Generation — Depthwise Conv Kernel
// ---------------------------------------------------------------------------

/// Parameters for the depthwise convolution PTX body.
#[derive(Debug, Clone, Copy)]
struct DepthwiseBodyParams {
    float_type: PtxType,
    elem_bytes: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    activation: ActivationType,
    has_bn: bool,
}

/// Generates PTX for the depthwise convolution stage.
///
/// Each thread computes one output element for one channel. The kernel
/// iterates over `kernel_h × kernel_w` with padding/bounds checks and
/// applies optional batch normalisation and activation.
///
/// # Parameters
///
/// - `config` — depthwise separable configuration
/// - `precision` — `"f32"` or `"f64"`
/// - `sm_version` — target GPU architecture
///
/// # Errors
///
/// Returns [`DnnError::PtxGeneration`] if kernel building fails.
pub fn generate_depthwise_conv_ptx(
    config: &DepthwiseSeparableConfig,
    precision: &str,
    sm_version: SmVersion,
) -> DnnResult<String> {
    let (float_type, elem_bytes) = parse_precision(precision)?;
    let act_suffix = config.depthwise_activation.kernel_suffix();
    let kernel_name = format!("depthwise_separable_dw_{act_suffix}_{precision}");

    let params = DepthwiseBodyParams {
        float_type,
        elem_bytes,
        kernel_h: config.kernel_h as u32,
        kernel_w: config.kernel_w as u32,
        stride_h: config.stride_h as u32,
        stride_w: config.stride_w as u32,
        pad_h: config.pad_h as u32,
        pad_w: config.pad_w as u32,
        dilation_h: config.dilation_h as u32,
        dilation_w: config.dilation_w as u32,
        activation: config.depthwise_activation,
        has_bn: config.depthwise_bn,
    };

    let mut builder = KernelBuilder::new(&kernel_name);
    builder = builder
        .target(sm_version)
        .param("input", PtxType::U64)
        .param("filter", PtxType::U64)
        .param("output", PtxType::U64)
        .param("bias", PtxType::U64)
        .param("batch_size", PtxType::U32)
        .param("channels", PtxType::U32)
        .param("in_h", PtxType::U32)
        .param("in_w", PtxType::U32)
        .param("out_h", PtxType::U32)
        .param("out_w", PtxType::U32)
        .param("total_outputs", PtxType::U32);

    // BN parameters (scale, bias, mean, var, eps)
    if config.depthwise_bn {
        builder = builder
            .param("bn_gamma", PtxType::U64)
            .param("bn_beta", PtxType::U64)
            .param("bn_mean", PtxType::U64)
            .param("bn_var", PtxType::U64);
    }

    let ptx = builder
        .body(move |b| {
            emit_depthwise_conv_body(b, &params);
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Emits the depthwise conv kernel body.
fn emit_depthwise_conv_body(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    p: &DepthwiseBodyParams,
) {
    let float_type = p.float_type;
    let elem_bytes = p.elem_bytes;
    let kernel_h = p.kernel_h;
    let kernel_w = p.kernel_w;
    let stride_h = p.stride_h;
    let stride_w = p.stride_w;
    let pad_h = p.pad_h;
    let pad_w = p.pad_w;
    let dilation_h = p.dilation_h;
    let dilation_w = p.dilation_w;

    b.comment("=== Depthwise Convolution Stage ===");
    b.comment("Each thread computes one output (batch, channel, oh, ow).");

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_outputs");

    let pred_bounds = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_bounds}, {gid}, {total};"));
    let exit_label = b.fresh_label("dw_exit");
    b.raw_ptx(&format!("@!{pred_bounds} bra {exit_label};"));

    // Load parameters
    let input_ptr = b.load_param_u64("input");
    let filter_ptr = b.load_param_u64("filter");
    let output_ptr = b.load_param_u64("output");
    let _channels = b.load_param_u32("channels");
    let in_h = b.load_param_u32("in_h");
    let in_w = b.load_param_u32("in_w");
    let out_h = b.load_param_u32("out_h");
    let out_w = b.load_param_u32("out_w");

    b.comment("Decompose gid -> (batch, c, oh, ow)");
    let out_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {out_hw}, {out_h}, {out_w};"));

    let c_out_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {c_out_hw}, {_channels}, {out_hw};"));

    let batch_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {batch_idx}, {gid}, {c_out_hw};"));
    let rem1 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem1}, {gid}, {c_out_hw};"));

    let ch_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {ch_idx}, {rem1}, {out_hw};"));
    let rem2 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem2}, {rem1}, {out_hw};"));

    let oh = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {oh}, {rem2}, {out_w};"));
    let ow = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {ow}, {rem2}, {out_w};"));

    // Initialize accumulator
    let acc = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        let zero_bits = 0f32.to_bits();
        b.raw_ptx(&format!("mov.b32 {acc}, 0F{zero_bits:08X};"));
    } else {
        let zero_bits = 0f64.to_bits();
        b.raw_ptx(&format!("mov.b64 {acc}, 0D{zero_bits:016X};"));
    }

    // Pre-compute strides for input: batch * channels * in_h * in_w
    let in_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {in_hw}, {in_h}, {in_w};"));

    // Scratch registers
    let ih_base = b.alloc_reg(PtxType::U32);
    let iw_base = b.alloc_reg(PtxType::U32);
    let ih = b.alloc_reg(PtxType::U32);
    let iw = b.alloc_reg(PtxType::U32);
    let pred_ih_ge = b.alloc_reg(PtxType::Pred);
    let pred_ih_lt = b.alloc_reg(PtxType::Pred);
    let pred_iw_ge = b.alloc_reg(PtxType::Pred);
    let pred_iw_lt = b.alloc_reg(PtxType::Pred);
    let input_idx = b.alloc_reg(PtxType::U32);
    let filter_idx = b.alloc_reg(PtxType::U32);
    let addr64 = b.alloc_reg(PtxType::U64);
    let off64 = b.alloc_reg(PtxType::U64);
    let idx64 = b.alloc_reg(PtxType::U64);
    let in_val = b.alloc_reg(float_type);
    let f_val = b.alloc_reg(float_type);
    // oh * stride_h - pad_h (as signed: compute oh * stride_h, subtract pad later)
    b.raw_ptx(&format!("mul.lo.u32 {ih_base}, {oh}, {stride_h};"));
    b.raw_ptx(&format!("mul.lo.u32 {iw_base}, {ow}, {stride_w};"));

    // batch_offset = batch_idx * channels * in_hw
    let batch_offset = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "mul.lo.u32 {batch_offset}, {batch_idx}, {_channels};"
    ));
    b.raw_ptx(&format!(
        "mul.lo.u32 {batch_offset}, {batch_offset}, {in_hw};"
    ));

    // channel_offset = ch_idx * in_hw
    let channel_offset = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {channel_offset}, {ch_idx}, {in_hw};"));

    let kh_kw_val = kernel_h * kernel_w;
    // filter base = ch_idx * kh * kw
    let filter_base = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {filter_base}, {ch_idx}, {kh_kw_val};"));

    b.comment("Unrolled loop over (kh, kw)");
    for kh_val in 0..kernel_h {
        for kw_val in 0..kernel_w {
            let skip = b.fresh_label(&format!("dw_skip_kh{kh_val}_kw{kw_val}"));

            let kh_dil = kh_val * dilation_h;
            let kw_dil = kw_val * dilation_w;

            // ih = ih_base + kh_dil - pad_h (with underflow check)
            if pad_h > 0 {
                let has_underflow = kh_dil < pad_h;
                if has_underflow {
                    // ih_base + kh_dil may be < pad_h
                    let threshold = pad_h - kh_dil;
                    let threshold_reg = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {threshold_reg}, {threshold};"));
                    b.raw_ptx(&format!(
                        "setp.hs.u32 {pred_ih_ge}, {ih_base}, {threshold_reg};"
                    ));
                    b.raw_ptx(&format!("@!{pred_ih_ge} bra {skip};"));
                    b.raw_ptx(&format!("add.u32 {ih}, {ih_base}, {kh_dil};"));
                    b.raw_ptx(&format!("sub.u32 {ih}, {ih}, {pad_h};"));
                } else {
                    let offset_val = kh_dil - pad_h;
                    b.raw_ptx(&format!("add.u32 {ih}, {ih_base}, {offset_val};"));
                }
            } else {
                b.raw_ptx(&format!("add.u32 {ih}, {ih_base}, {kh_dil};"));
            }
            // check ih < in_h
            b.raw_ptx(&format!("setp.lo.u32 {pred_ih_lt}, {ih}, {in_h};"));
            b.raw_ptx(&format!("@!{pred_ih_lt} bra {skip};"));

            // iw = iw_base + kw_dil - pad_w
            if pad_w > 0 {
                let has_underflow = kw_dil < pad_w;
                if has_underflow {
                    let threshold = pad_w - kw_dil;
                    let threshold_reg = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {threshold_reg}, {threshold};"));
                    b.raw_ptx(&format!(
                        "setp.hs.u32 {pred_iw_ge}, {iw_base}, {threshold_reg};"
                    ));
                    b.raw_ptx(&format!("@!{pred_iw_ge} bra {skip};"));
                    b.raw_ptx(&format!("add.u32 {iw}, {iw_base}, {kw_dil};"));
                    b.raw_ptx(&format!("sub.u32 {iw}, {iw}, {pad_w};"));
                } else {
                    let offset_val = kw_dil - pad_w;
                    b.raw_ptx(&format!("add.u32 {iw}, {iw_base}, {offset_val};"));
                }
            } else {
                b.raw_ptx(&format!("add.u32 {iw}, {iw_base}, {kw_dil};"));
            }
            // check iw < in_w
            b.raw_ptx(&format!("setp.lo.u32 {pred_iw_lt}, {iw}, {in_w};"));
            b.raw_ptx(&format!("@!{pred_iw_lt} bra {skip};"));

            // input_idx = batch_offset + channel_offset + ih * in_w + iw
            let ih_times_inw = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mul.lo.u32 {ih_times_inw}, {ih}, {in_w};"));
            b.raw_ptx(&format!(
                "add.u32 {input_idx}, {batch_offset}, {channel_offset};"
            ));
            b.raw_ptx(&format!(
                "add.u32 {input_idx}, {input_idx}, {ih_times_inw};"
            ));
            b.raw_ptx(&format!("add.u32 {input_idx}, {input_idx}, {iw};"));

            // Load input
            b.raw_ptx(&format!("cvt.u64.u32 {idx64}, {input_idx};"));
            b.raw_ptx(&format!("mul.lo.u64 {off64}, {idx64}, {elem_bytes};"));
            b.raw_ptx(&format!("add.u64 {addr64}, {input_ptr}, {off64};"));
            if float_type == PtxType::F32 {
                b.raw_ptx(&format!("ld.global.f32 {in_val}, [{addr64}];"));
            } else {
                b.raw_ptx(&format!("ld.global.f64 {in_val}, [{addr64}];"));
            }

            // filter_idx = filter_base + kh_val * kernel_w + kw_val
            let filt_offset = kh_val * kernel_w + kw_val;
            b.raw_ptx(&format!(
                "add.u32 {filter_idx}, {filter_base}, {filt_offset};"
            ));

            // Load filter
            b.raw_ptx(&format!("cvt.u64.u32 {idx64}, {filter_idx};"));
            b.raw_ptx(&format!("mul.lo.u64 {off64}, {idx64}, {elem_bytes};"));
            b.raw_ptx(&format!("add.u64 {addr64}, {filter_ptr}, {off64};"));
            if float_type == PtxType::F32 {
                b.raw_ptx(&format!("ld.global.f32 {f_val}, [{addr64}];"));
            } else {
                b.raw_ptx(&format!("ld.global.f64 {f_val}, [{addr64}];"));
            }

            // acc += in_val * f_val
            if float_type == PtxType::F32 {
                b.raw_ptx(&format!("fma.rn.f32 {acc}, {in_val}, {f_val}, {acc};"));
            } else {
                b.raw_ptx(&format!("fma.rn.f64 {acc}, {in_val}, {f_val}, {acc};"));
            }

            b.label(&skip);
        }
    }

    // Optional BN: y = gamma * (acc - mean) / sqrt(var + eps) + beta
    if p.has_bn {
        emit_bn_epilogue(b, float_type, elem_bytes, &acc, ch_idx);
    }

    // Activation
    let activated = emit_activation(b, float_type, acc, p.activation);

    // Store to output
    b.comment("Store depthwise output");
    let out_idx = b.alloc_reg(PtxType::U64);
    let out_off = b.alloc_reg(PtxType::U64);
    let out_addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {out_idx}, {gid};"));
    b.raw_ptx(&format!("mul.lo.u64 {out_off}, {out_idx}, {elem_bytes};"));
    b.raw_ptx(&format!("add.u64 {out_addr}, {output_ptr}, {out_off};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("st.global.f32 [{out_addr}], {activated};"));
    } else {
        b.raw_ptx(&format!("st.global.f64 [{out_addr}], {activated};"));
    }

    b.label(&exit_label);
    b.ret();
}
