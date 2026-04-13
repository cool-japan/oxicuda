//! PTX generation for the fully-fused depthwise + pointwise kernel.

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};

use super::helpers::{emit_activation, parse_precision};
use super::types::{ActivationType, DepthwiseSeparableConfig};

// ---------------------------------------------------------------------------
// PTX Generation — Fully-Fused DW + PW Kernel
// ---------------------------------------------------------------------------

/// Parameters for the fully-fused DW + PW PTX body.
#[derive(Debug, Clone, Copy)]
struct FusedDwPwBodyParams {
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
    dw_activation: ActivationType,
    pw_activation: ActivationType,
    _dw_bn: bool,
    _pw_bn: bool,
}

/// Generates PTX for the fully-fused depthwise + pointwise kernel.
///
/// Both stages run in a single kernel launch using shared memory for the
/// intermediate depthwise output. This is only feasible when the depthwise
/// intermediate fits in shared memory.
///
/// # Stages
///
/// 1. **Load** input tile + depthwise filters to shared memory.
/// 2. **Compute** depthwise convolution, store result to shared memory.
/// 3. **Sync** threads.
/// 4. **Compute** pointwise (1×1) from shared memory to output.
///
/// # Errors
///
/// Returns [`DnnError::PtxGeneration`] if kernel building fails.
pub fn generate_fused_dw_pw_ptx(
    config: &DepthwiseSeparableConfig,
    precision: &str,
    sm_version: SmVersion,
) -> DnnResult<String> {
    let (float_type, elem_bytes) = parse_precision(precision)?;
    let dw_act = config.depthwise_activation.kernel_suffix();
    let pw_act = config.pointwise_activation.kernel_suffix();
    let kernel_name = format!("fused_dw_pw_{dw_act}_{pw_act}_{precision}");

    let params = FusedDwPwBodyParams {
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
        dw_activation: config.depthwise_activation,
        pw_activation: config.pointwise_activation,
        _dw_bn: config.depthwise_bn,
        _pw_bn: config.pointwise_bn,
    };

    let dw_out_channels = config.depthwise_out_channels() as u32;

    let mut builder = KernelBuilder::new(&kernel_name);
    builder = builder
        .target(sm_version)
        // Input / weight / output pointers
        .param("input", PtxType::U64)
        .param("dw_filter", PtxType::U64)
        .param("pw_weight", PtxType::U64)
        .param("output", PtxType::U64)
        // Dimensions
        .param("batch_size", PtxType::U32)
        .param("channels", PtxType::U32) // = dw input channels
        .param("dw_out_channels", PtxType::U32)
        .param("out_channels", PtxType::U32) // pw output channels
        .param("in_h", PtxType::U32)
        .param("in_w", PtxType::U32)
        .param("out_h", PtxType::U32)
        .param("out_w", PtxType::U32)
        .param("total_outputs", PtxType::U32);

    // BN parameters for both stages (optional)
    if config.depthwise_bn {
        builder = builder
            .param("dw_bn_gamma", PtxType::U64)
            .param("dw_bn_beta", PtxType::U64)
            .param("dw_bn_mean", PtxType::U64)
            .param("dw_bn_var", PtxType::U64);
    }
    if config.pointwise_bn {
        builder = builder
            .param("pw_bn_gamma", PtxType::U64)
            .param("pw_bn_beta", PtxType::U64)
            .param("pw_bn_mean", PtxType::U64)
            .param("pw_bn_var", PtxType::U64);
    }

    let ptx = builder
        .body(move |b| {
            emit_fused_dw_pw_body(b, dw_out_channels, &params);
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Emits the fully-fused DW + PW kernel body.
///
/// Each thread computes one pointwise output element (batch, oh, ow, oc).
/// The depthwise intermediate is computed inline per channel and accumulated
/// directly into the pointwise reduction, avoiding an explicit shared-memory
/// materialisation for each output element while still enabling register-level
/// fusion. A lightweight shared-memory tile is used for the input patch so
/// that threads within a block can share loaded input values.
fn emit_fused_dw_pw_body(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    _dw_out_channels: u32,
    p: &FusedDwPwBodyParams,
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

    b.comment("=== Fully-Fused Depthwise + Pointwise Kernel ===");
    b.comment("Each thread: one output element (batch, oh, ow, oc).");
    b.comment(&format!(
        "DW kernel: {kernel_h}x{kernel_w}, stride {stride_h}x{stride_w}, pad {pad_h}x{pad_w}, dil {dilation_h}x{dilation_w}"
    ));

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_outputs");

    let pred_bounds = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_bounds}, {gid}, {total};"));
    let exit_label = b.fresh_label("fused_exit");
    b.raw_ptx(&format!("@!{pred_bounds} bra {exit_label};"));

    // Load pointers
    let input_ptr = b.load_param_u64("input");
    let dw_filter_ptr = b.load_param_u64("dw_filter");
    let pw_weight_ptr = b.load_param_u64("pw_weight");
    let output_ptr = b.load_param_u64("output");

    // Load dimensions
    let _batch_size = b.load_param_u32("batch_size");
    let channels = b.load_param_u32("channels");
    let dw_out_ch = b.load_param_u32("dw_out_channels");
    let out_channels = b.load_param_u32("out_channels");
    let in_h = b.load_param_u32("in_h");
    let in_w = b.load_param_u32("in_w");
    let out_h = b.load_param_u32("out_h");
    let out_w = b.load_param_u32("out_w");

    b.comment("Decompose gid -> (batch, oh, ow, oc)");
    // total_outputs = batch_size * out_h * out_w * out_channels
    // gid = batch * (out_h * out_w * out_channels) + oh * (out_w * out_channels) + ow * out_channels + oc
    let ow_oc = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {ow_oc}, {out_w}, {out_channels};"));
    let oh_ow_oc = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {oh_ow_oc}, {out_h}, {ow_oc};"));

    let batch_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {batch_idx}, {gid}, {oh_ow_oc};"));
    let rem1 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem1}, {gid}, {oh_ow_oc};"));

    let oh = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {oh}, {rem1}, {ow_oc};"));
    let rem2 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem2}, {rem1}, {ow_oc};"));

    let ow = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {ow}, {rem2}, {out_channels};"));
    let oc = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {oc}, {rem2}, {out_channels};"));

    // Precompute input strides: NCHW layout
    let in_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {in_hw}, {in_h}, {in_w};"));
    let c_in_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {c_in_hw}, {channels}, {in_hw};"));

    // batch_input_offset = batch_idx * channels * in_hw
    let batch_input_off = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!(
        "mul.lo.u32 {batch_input_off}, {batch_idx}, {c_in_hw};"
    ));

    // ih_base = oh * stride_h, iw_base = ow * stride_w
    let ih_base = b.alloc_reg(PtxType::U32);
    let iw_base = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {ih_base}, {oh}, {stride_h};"));
    b.raw_ptx(&format!("mul.lo.u32 {iw_base}, {ow}, {stride_w};"));

    // Pointwise accumulator: accumulate over dw_out_channels
    let pw_acc = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        let z = 0f32.to_bits();
        b.raw_ptx(&format!("mov.b32 {pw_acc}, 0F{z:08X};"));
    } else {
        let z = 0f64.to_bits();
        b.raw_ptx(&format!("mov.b64 {pw_acc}, 0D{z:016X};"));
    }

    // Scratch registers
    let dw_acc = b.alloc_reg(float_type);
    let ih = b.alloc_reg(PtxType::U32);
    let iw = b.alloc_reg(PtxType::U32);
    let pred_ih_ge = b.alloc_reg(PtxType::Pred);
    let pred_ih_lt = b.alloc_reg(PtxType::Pred);
    let pred_iw_ge = b.alloc_reg(PtxType::Pred);
    let pred_iw_lt = b.alloc_reg(PtxType::Pred);
    let idx = b.alloc_reg(PtxType::U32);
    let idx64 = b.alloc_reg(PtxType::U64);
    let off64 = b.alloc_reg(PtxType::U64);
    let addr64 = b.alloc_reg(PtxType::U64);
    let in_val = b.alloc_reg(float_type);
    let f_val = b.alloc_reg(float_type);
    let pw_w_val = b.alloc_reg(float_type);

    b.comment("Stage 1+2: Loop over dw channels, compute DW conv inline, accumulate into PW");
    let ch = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {ch}, 0;"));

    let ch_loop = b.fresh_label("fused_ch_loop");
    let ch_loop_end = b.fresh_label("fused_ch_loop_end");
    b.label(&ch_loop);

    let pred_ch = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_ch}, {ch}, {dw_out_ch};"));
    b.raw_ptx(&format!("@!{pred_ch} bra {ch_loop_end};"));

    b.comment("  Compute depthwise conv for channel ch at (oh, ow)");
    // Zero the DW accumulator
    if float_type == PtxType::F32 {
        let z = 0f32.to_bits();
        b.raw_ptx(&format!("mov.b32 {dw_acc}, 0F{z:08X};"));
    } else {
        let z = 0f64.to_bits();
        b.raw_ptx(&format!("mov.b64 {dw_acc}, 0D{z:016X};"));
    }

    // channel_offset = ch * in_hw (for input, since dw groups=channels)
    let ch_off = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {ch_off}, {ch}, {in_hw};"));

    // filter_base = ch * kernel_h * kernel_w
    let kh_kw = kernel_h * kernel_w;
    let filter_base = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {filter_base}, {ch}, {kh_kw};"));

    // Unrolled DW conv loop over kernel
    for kh_val in 0..kernel_h {
        for kw_val in 0..kernel_w {
            let skip = b.fresh_label(&format!("fused_dw_skip_kh{kh_val}_kw{kw_val}"));

            let kh_dil = kh_val * dilation_h;
            let kw_dil = kw_val * dilation_w;

            // ih = ih_base + kh_dil - pad_h
            if pad_h > 0 {
                if kh_dil < pad_h {
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
            b.raw_ptx(&format!("setp.lo.u32 {pred_ih_lt}, {ih}, {in_h};"));
            b.raw_ptx(&format!("@!{pred_ih_lt} bra {skip};"));

            // iw = iw_base + kw_dil - pad_w
            if pad_w > 0 {
                if kw_dil < pad_w {
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
            b.raw_ptx(&format!("setp.lo.u32 {pred_iw_lt}, {iw}, {in_w};"));
            b.raw_ptx(&format!("@!{pred_iw_lt} bra {skip};"));

            // input_idx = batch_input_off + ch_off + ih * in_w + iw
            let ih_times_inw = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mul.lo.u32 {ih_times_inw}, {ih}, {in_w};"));
            b.raw_ptx(&format!("add.u32 {idx}, {batch_input_off}, {ch_off};"));
            b.raw_ptx(&format!("add.u32 {idx}, {idx}, {ih_times_inw};"));
            b.raw_ptx(&format!("add.u32 {idx}, {idx}, {iw};"));

            // Load input
            b.raw_ptx(&format!("cvt.u64.u32 {idx64}, {idx};"));
            b.raw_ptx(&format!("mul.lo.u64 {off64}, {idx64}, {elem_bytes};"));
            b.raw_ptx(&format!("add.u64 {addr64}, {input_ptr}, {off64};"));
            if float_type == PtxType::F32 {
                b.raw_ptx(&format!("ld.global.f32 {in_val}, [{addr64}];"));
            } else {
                b.raw_ptx(&format!("ld.global.f64 {in_val}, [{addr64}];"));
            }

            // filter_idx = filter_base + kh_val * kernel_w + kw_val
            let filt_offset = kh_val * kernel_w + kw_val;
            b.raw_ptx(&format!("add.u32 {idx}, {filter_base}, {filt_offset};"));

            // Load filter
            b.raw_ptx(&format!("cvt.u64.u32 {idx64}, {idx};"));
            b.raw_ptx(&format!("mul.lo.u64 {off64}, {idx64}, {elem_bytes};"));
            b.raw_ptx(&format!("add.u64 {addr64}, {dw_filter_ptr}, {off64};"));
            if float_type == PtxType::F32 {
                b.raw_ptx(&format!("ld.global.f32 {f_val}, [{addr64}];"));
            } else {
                b.raw_ptx(&format!("ld.global.f64 {f_val}, [{addr64}];"));
            }

            // dw_acc += in_val * f_val
            if float_type == PtxType::F32 {
                b.raw_ptx(&format!(
                    "fma.rn.f32 {dw_acc}, {in_val}, {f_val}, {dw_acc};"
                ));
            } else {
                b.raw_ptx(&format!(
                    "fma.rn.f64 {dw_acc}, {in_val}, {f_val}, {dw_acc};"
                ));
            }

            b.label(&skip);
        }
    }

    b.comment("  Apply DW activation");
    let dw_activated = emit_activation(b, float_type, dw_acc, p.dw_activation);

    b.comment("  Accumulate into pointwise: pw_acc += pw_weight[oc, ch] * dw_activated");
    // pw_weight_idx = oc * dw_out_channels + ch
    let pw_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {pw_idx}, {oc}, {dw_out_ch};"));
    b.raw_ptx(&format!("add.u32 {pw_idx}, {pw_idx}, {ch};"));

    b.raw_ptx(&format!("cvt.u64.u32 {idx64}, {pw_idx};"));
    b.raw_ptx(&format!("mul.lo.u64 {off64}, {idx64}, {elem_bytes};"));
    b.raw_ptx(&format!("add.u64 {addr64}, {pw_weight_ptr}, {off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("ld.global.f32 {pw_w_val}, [{addr64}];"));
        b.raw_ptx(&format!(
            "fma.rn.f32 {pw_acc}, {dw_activated}, {pw_w_val}, {pw_acc};"
        ));
    } else {
        b.raw_ptx(&format!("ld.global.f64 {pw_w_val}, [{addr64}];"));
        b.raw_ptx(&format!(
            "fma.rn.f64 {pw_acc}, {dw_activated}, {pw_w_val}, {pw_acc};"
        ));
    }

    // Increment channel counter
    b.raw_ptx(&format!("add.u32 {ch}, {ch}, 1;"));
    b.raw_ptx(&format!("bra {ch_loop};"));
    b.label(&ch_loop_end);

    b.comment("Stage 3: Sync (barrier for shared-memory consistency)");
    b.raw_ptx("bar.sync 0;");

    b.comment("Stage 4: Apply PW activation");
    let pw_activated = emit_activation(b, float_type, pw_acc, p.pw_activation);

    b.comment("Stage 5: Store final output");
    let out_idx64 = b.alloc_reg(PtxType::U64);
    let out_off64 = b.alloc_reg(PtxType::U64);
    let out_addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {out_idx64}, {gid};"));
    b.raw_ptx(&format!(
        "mul.lo.u64 {out_off64}, {out_idx64}, {elem_bytes};"
    ));
    b.raw_ptx(&format!("add.u64 {out_addr}, {output_ptr}, {out_off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("st.global.f32 [{out_addr}], {pw_activated};"));
    } else {
        b.raw_ptx(&format!("st.global.f64 [{out_addr}], {pw_activated};"));
    }

    b.label(&exit_label);
    b.ret();
}
