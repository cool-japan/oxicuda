//! PTX kernel generators for 3D volumetric convolution.
//!
//! Contains the low-level PTX emitters for:
//! - **im2col3d** — unfolds volumetric patches into columns for GEMM
//! - **col2im3d** — scatters column entries back to 3D spatial positions (backward data)
//! - **direct3d** — direct 3×3×3 convolution kernel
//! - **wgrad3d** — weight gradient kernel (backward filter)

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use super::Conv3dConfig;
use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// PTX Generation — im2col 3D kernel
// ---------------------------------------------------------------------------

/// Captured parameters for the im2col3d kernel body emission.
///
/// Bundles all compile-time constants needed by the PTX emitter to avoid
/// exceeding clippy's argument-count limit.
#[derive(Debug, Clone, Copy)]
struct Im2col3dParams {
    float_type: PtxType,
    elem_bytes: u32,
    kernel_d: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_d: u32,
    stride_h: u32,
    stride_w: u32,
    pad_d: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_d: u32,
    dilation_h: u32,
    dilation_w: u32,
    in_channels_per_group: u32,
}

/// Captured spatial dimensions for im2col 3D kernel.
#[derive(Debug, Clone, Copy)]
struct Im2col3dDims {
    input_d: u32,
    input_h: u32,
    input_w: u32,
    output_d: u32,
    output_h: u32,
    output_w: u32,
}

/// Generates PTX for the im2col 3D kernel.
///
/// Transforms a 5D input tensor `[N, C, D, H, W]` into a 2D column matrix
/// suitable for GEMM-based 3D convolution. Each thread handles one column
/// of the matrix corresponding to one output position `(od, oh, ow)`.
///
/// The column matrix has shape:
/// ```text
/// rows = C_in/groups × kD × kH × kW
/// cols = oD × oH × oW
/// ```
///
/// # Errors
///
/// Returns [`DnnError::InvalidArgument`] for unsupported precision.
/// Returns [`DnnError::PtxGeneration`] if kernel building fails.
pub fn generate_im2col3d_ptx(
    config: &Conv3dConfig,
    batch_size: usize,
    input_d: usize,
    input_h: usize,
    input_w: usize,
    precision: &str,
    sm_version: SmVersion,
) -> DnnResult<String> {
    let _ = batch_size; // used conceptually; kernel works per-sample

    let kernel_name = format!("im2col3d_{precision}");
    let float_type = match precision {
        "f32" => PtxType::F32,
        "f64" => PtxType::F64,
        other => {
            return Err(DnnError::InvalidArgument(format!(
                "unsupported precision for im2col3d: {other}"
            )));
        }
    };
    let elem_bytes: u32 = match precision {
        "f32" => 4,
        _ => 8,
    };

    let (output_d, output_h, output_w) = config.output_size(input_d, input_h, input_w);

    let params = Im2col3dParams {
        float_type,
        elem_bytes,
        kernel_d: config.kernel_d as u32,
        kernel_h: config.kernel_h as u32,
        kernel_w: config.kernel_w as u32,
        stride_d: config.stride_d as u32,
        stride_h: config.stride_h as u32,
        stride_w: config.stride_w as u32,
        pad_d: config.pad_d as u32,
        pad_h: config.pad_h as u32,
        pad_w: config.pad_w as u32,
        dilation_d: config.dilation_d as u32,
        dilation_h: config.dilation_h as u32,
        dilation_w: config.dilation_w as u32,
        in_channels_per_group: config.in_channels_per_group() as u32,
    };

    let dims = Im2col3dDims {
        input_d: input_d as u32,
        input_h: input_h as u32,
        input_w: input_w as u32,
        output_d: output_d as u32,
        output_h: output_h as u32,
        output_w: output_w as u32,
    };

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm_version)
        .param("input", PtxType::U64)
        .param("col_matrix", PtxType::U64)
        .param("total_columns", PtxType::U32)
        .body(move |b| {
            emit_im2col3d_body(b, &params, &dims);
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Emits the im2col3d kernel body.
///
/// Each thread handles one column `col_idx` → output position `(od, oh, ow)`.
/// For each `(c, kd, kh, kw)` it computes the source input position, performs
/// a bounds check (padding → zero), loads the value, and stores it into its
/// column of the output matrix.
fn emit_im2col3d_body(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    p: &Im2col3dParams,
    dims: &Im2col3dDims,
) {
    let float_type = p.float_type;
    let elem_bytes = p.elem_bytes;

    b.comment("=== im2col 3D Kernel ===");
    b.comment("Each thread: one column (od, oh, ow) of the column matrix.");

    // Global thread ID and bounds check.
    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_columns");
    let pred_bounds = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_bounds}, {gid}, {total};"));
    let exit_label = b.fresh_label("im2col3d_exit");
    b.raw_ptx(&format!("@!{pred_bounds} bra {exit_label};"));

    b.comment("Load parameters");
    let input_ptr = b.load_param_u64("input");
    let col_ptr = b.load_param_u64("col_matrix");

    b.comment("Decompose gid -> (od, oh, ow)");
    let out_hw = dims.output_h * dims.output_w;
    let od = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {od}, {gid}, {out_hw};"));
    let rem_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem_hw}, {gid}, {out_hw};"));
    let oh = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {oh}, {rem_hw}, {};", dims.output_w));
    let ow = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {ow}, {rem_hw}, {};", dims.output_w));

    // Pre-compute column stride values.
    let total_columns: u32 = dims.output_d * dims.output_h * dims.output_w;
    let _col_rows: u32 = p.in_channels_per_group * p.kernel_d * p.kernel_h * p.kernel_w;

    // Scratch registers (allocated once, reused across the unrolled body).
    let id_reg = b.alloc_reg(PtxType::U32);
    let ih_reg = b.alloc_reg(PtxType::U32);
    let iw_reg = b.alloc_reg(PtxType::U32);
    let pred_d_ok = b.alloc_reg(PtxType::Pred);
    let pred_h_ok = b.alloc_reg(PtxType::Pred);
    let pred_w_ok = b.alloc_reg(PtxType::Pred);
    let src_idx = b.alloc_reg(PtxType::U32);
    let dst_idx = b.alloc_reg(PtxType::U32);
    let addr64 = b.alloc_reg(PtxType::U64);
    let off64 = b.alloc_reg(PtxType::U64);
    let load_addr = b.alloc_reg(PtxType::U64);
    let store_addr = b.alloc_reg(PtxType::U64);
    let val = b.alloc_reg(float_type);
    let zero_val = b.alloc_reg(float_type);
    let tmp32 = b.alloc_reg(PtxType::U32);

    // Set zero_val for padding.
    if float_type == PtxType::F32 {
        let zero_bits: u32 = 0f32.to_bits();
        b.raw_ptx(&format!("mov.b32 {zero_val}, 0F{zero_bits:08X};"));
    } else {
        let zero_bits: u64 = 0f64.to_bits();
        b.raw_ptx(&format!("mov.b64 {zero_val}, 0D{zero_bits:016X};"));
    }

    // Input spatial strides: H*W, W.
    let in_hw: u32 = dims.input_h * dims.input_w;
    let in_dhw: u32 = dims.input_d * in_hw;

    b.comment("Unrolled loop over (c, kd, kh, kw)");
    // We unroll across the kernel spatial dims but keep channel as a runtime
    // iteration when in_channels_per_group is large. For moderate channel
    // counts (common in 3D CNNs) full unrolling is acceptable. However,
    // to keep PTX size bounded we generate a runtime loop over channels
    // and unroll only the 3D kernel window.

    // Runtime channel loop.
    let c_reg = b.alloc_reg(PtxType::U32);
    let c_limit = p.in_channels_per_group;
    let c_loop_label = b.fresh_label("c_loop");
    let c_done_label = b.fresh_label("c_done");
    let pred_c = b.alloc_reg(PtxType::Pred);

    b.raw_ptx(&format!("mov.u32 {c_reg}, 0;"));
    b.label(&c_loop_label);
    b.raw_ptx(&format!("setp.lo.u32 {pred_c}, {c_reg}, {c_limit};"));
    b.raw_ptx(&format!("@!{pred_c} bra {c_done_label};"));

    // row_base = c * kD * kH * kW
    let kd_kh_kw: u32 = p.kernel_d * p.kernel_h * p.kernel_w;
    let kh_kw: u32 = p.kernel_h * p.kernel_w;
    let row_base = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {row_base}, {c_reg}, {kd_kh_kw};"));

    // Unrolled kernel spatial loop.
    for kd_val in 0..p.kernel_d {
        for kh_val in 0..p.kernel_h {
            for kw_val in 0..p.kernel_w {
                let kd_dil = kd_val * p.dilation_d;
                let kh_dil = kh_val * p.dilation_h;
                let kw_dil = kw_val * p.dilation_w;
                let kernel_offset: u32 = kd_val * kh_kw + kh_val * p.kernel_w + kw_val;
                let skip = b.fresh_label(&format!("skip_kd{kd_val}_kh{kh_val}_kw{kw_val}"));
                let store_lbl = b.fresh_label(&format!("store_kd{kd_val}_kh{kh_val}_kw{kw_val}"));

                // id = od * stride_d + kd_dil - pad_d  (with underflow check)
                b.raw_ptx(&format!(
                    "mad.lo.u32 {id_reg}, {od}, {}, {kd_dil};",
                    p.stride_d
                ));
                // Check id_reg >= pad_d  (since unsigned, sub would underflow)
                b.raw_ptx(&format!("setp.hs.u32 {pred_d_ok}, {id_reg}, {};", p.pad_d));
                b.raw_ptx(&format!("@!{pred_d_ok} bra {skip};"));
                b.raw_ptx(&format!("sub.u32 {id_reg}, {id_reg}, {};", p.pad_d));
                // Check id_reg < input_d.
                b.raw_ptx(&format!(
                    "setp.lo.u32 {pred_d_ok}, {id_reg}, {};",
                    dims.input_d
                ));
                b.raw_ptx(&format!("@!{pred_d_ok} bra {skip};"));

                // ih = oh * stride_h + kh_dil - pad_h
                b.raw_ptx(&format!(
                    "mad.lo.u32 {ih_reg}, {oh}, {}, {kh_dil};",
                    p.stride_h
                ));
                b.raw_ptx(&format!("setp.hs.u32 {pred_h_ok}, {ih_reg}, {};", p.pad_h));
                b.raw_ptx(&format!("@!{pred_h_ok} bra {skip};"));
                b.raw_ptx(&format!("sub.u32 {ih_reg}, {ih_reg}, {};", p.pad_h));
                b.raw_ptx(&format!(
                    "setp.lo.u32 {pred_h_ok}, {ih_reg}, {};",
                    dims.input_h
                ));
                b.raw_ptx(&format!("@!{pred_h_ok} bra {skip};"));

                // iw = ow * stride_w + kw_dil - pad_w
                b.raw_ptx(&format!(
                    "mad.lo.u32 {iw_reg}, {ow}, {}, {kw_dil};",
                    p.stride_w
                ));
                b.raw_ptx(&format!("setp.hs.u32 {pred_w_ok}, {iw_reg}, {};", p.pad_w));
                b.raw_ptx(&format!("@!{pred_w_ok} bra {skip};"));
                b.raw_ptx(&format!("sub.u32 {iw_reg}, {iw_reg}, {};", p.pad_w));
                b.raw_ptx(&format!(
                    "setp.lo.u32 {pred_w_ok}, {iw_reg}, {};",
                    dims.input_w
                ));
                b.raw_ptx(&format!("@!{pred_w_ok} bra {skip};"));

                // src_idx = c * D*H*W + id * H*W + ih * W + iw
                b.raw_ptx(&format!("mul.lo.u32 {src_idx}, {c_reg}, {in_dhw};"));
                b.raw_ptx(&format!(
                    "mad.lo.u32 {src_idx}, {id_reg}, {in_hw}, {src_idx};"
                ));
                b.raw_ptx(&format!(
                    "mad.lo.u32 {src_idx}, {ih_reg}, {}, {src_idx};",
                    dims.input_w
                ));
                b.raw_ptx(&format!("add.u32 {src_idx}, {src_idx}, {iw_reg};"));

                // Load input[src_idx].
                b.raw_ptx(&format!("cvt.u64.u32 {addr64}, {src_idx};"));
                b.raw_ptx(&format!("mul.lo.u64 {off64}, {addr64}, {elem_bytes};"));
                b.raw_ptx(&format!("add.u64 {load_addr}, {input_ptr}, {off64};"));
                if float_type == PtxType::F32 {
                    b.raw_ptx(&format!("ld.global.f32 {val}, [{load_addr}];"));
                } else {
                    b.raw_ptx(&format!("ld.global.f64 {val}, [{load_addr}];"));
                }
                b.raw_ptx(&format!("bra {store_lbl};"));

                // Padding branch: store zero.
                b.label(&skip);
                if float_type == PtxType::F32 {
                    b.raw_ptx(&format!("mov.b32 {val}, {zero_val};"));
                } else {
                    b.raw_ptx(&format!("mov.b64 {val}, {zero_val};"));
                }

                // Store to col_matrix.
                b.label(&store_lbl);
                // dst_idx = (row_base + kernel_offset) * total_columns + gid
                b.raw_ptx(&format!("add.u32 {tmp32}, {row_base}, {kernel_offset};"));
                b.raw_ptx(&format!("mul.lo.u32 {dst_idx}, {tmp32}, {total_columns};"));
                b.raw_ptx(&format!("add.u32 {dst_idx}, {dst_idx}, {gid};"));

                b.raw_ptx(&format!("cvt.u64.u32 {addr64}, {dst_idx};"));
                b.raw_ptx(&format!("mul.lo.u64 {off64}, {addr64}, {elem_bytes};"));
                b.raw_ptx(&format!("add.u64 {store_addr}, {col_ptr}, {off64};"));
                if float_type == PtxType::F32 {
                    b.raw_ptx(&format!("st.global.f32 [{store_addr}], {val};"));
                } else {
                    b.raw_ptx(&format!("st.global.f64 [{store_addr}], {val};"));
                }
            }
        }
    }

    // Advance channel counter.
    b.raw_ptx(&format!("add.u32 {c_reg}, {c_reg}, 1;"));
    b.raw_ptx(&format!("bra {c_loop_label};"));
    b.label(&c_done_label);

    b.label(&exit_label);
    b.ret();
}

// ---------------------------------------------------------------------------
// PTX Generation — col2im 3D kernel (backward pass)
// ---------------------------------------------------------------------------

/// Captured parameters for the col2im3d kernel body emission.
#[derive(Debug, Clone, Copy)]
struct Col2im3dParams {
    float_type: PtxType,
    elem_bytes: u32,
    kernel_d: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_d: u32,
    stride_h: u32,
    stride_w: u32,
    pad_d: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_d: u32,
    dilation_h: u32,
    dilation_w: u32,
}

/// Generates PTX for the col2im 3D scatter kernel (backward pass).
///
/// The col2im kernel is the inverse of im2col: it scatters column matrix
/// elements back to 3D spatial positions. Each thread handles one element
/// in the input gradient tensor and accumulates all contributing column
/// entries.
///
/// # Errors
///
/// Returns [`DnnError::InvalidArgument`] for unsupported precision.
/// Returns [`DnnError::PtxGeneration`] if kernel building fails.
pub fn generate_col2im3d_ptx(
    config: &Conv3dConfig,
    precision: &str,
    sm_version: SmVersion,
) -> DnnResult<String> {
    let kernel_name = format!("col2im3d_{precision}");
    let float_type = match precision {
        "f32" => PtxType::F32,
        "f64" => PtxType::F64,
        other => {
            return Err(DnnError::InvalidArgument(format!(
                "unsupported precision for col2im3d: {other}"
            )));
        }
    };
    let elem_bytes: u32 = match precision {
        "f32" => 4,
        _ => 8,
    };

    let params = Col2im3dParams {
        float_type,
        elem_bytes,
        kernel_d: config.kernel_d as u32,
        kernel_h: config.kernel_h as u32,
        kernel_w: config.kernel_w as u32,
        stride_d: config.stride_d as u32,
        stride_h: config.stride_h as u32,
        stride_w: config.stride_w as u32,
        pad_d: config.pad_d as u32,
        pad_h: config.pad_h as u32,
        pad_w: config.pad_w as u32,
        dilation_d: config.dilation_d as u32,
        dilation_h: config.dilation_h as u32,
        dilation_w: config.dilation_w as u32,
    };

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm_version)
        .param("col_matrix", PtxType::U64)
        .param("output", PtxType::U64)
        .param("channels_per_group", PtxType::U32)
        .param("in_d", PtxType::U32)
        .param("in_h", PtxType::U32)
        .param("in_w", PtxType::U32)
        .param("out_d", PtxType::U32)
        .param("out_h", PtxType::U32)
        .param("out_w", PtxType::U32)
        .param("total_elements", PtxType::U32)
        .body(move |b| {
            emit_col2im3d_body(b, &params);
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Emits the col2im 3D scatter kernel body.
///
/// Each thread writes one input-gradient element `(c, id, ih, iw)` by
/// accumulating every column-matrix entry that mapped to this position
/// during im2col.
fn emit_col2im3d_body(b: &mut oxicuda_ptx::builder::BodyBuilder<'_>, p: &Col2im3dParams) {
    let float_type = p.float_type;
    let elem_bytes = p.elem_bytes;

    b.comment("=== col2im 3D Scatter Kernel ===");
    b.comment("Each thread: one input-gradient element (c, id, ih, iw).");

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_elements");
    let pred_bounds = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_bounds}, {gid}, {total};"));
    let exit_label = b.fresh_label("col2im3d_exit");
    b.raw_ptx(&format!("@!{pred_bounds} bra {exit_label};"));

    let col_ptr = b.load_param_u64("col_matrix");
    let out_ptr = b.load_param_u64("output");
    let _cpg = b.load_param_u32("channels_per_group");
    let in_d = b.load_param_u32("in_d");
    let in_h = b.load_param_u32("in_h");
    let in_w = b.load_param_u32("in_w");
    let out_d = b.load_param_u32("out_d");
    let out_h = b.load_param_u32("out_h");
    let out_w = b.load_param_u32("out_w");

    b.comment("Decompose gid -> (c, id, ih, iw)");
    let in_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {in_hw}, {in_h}, {in_w};"));
    let in_dhw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {in_dhw}, {in_d}, {in_hw};"));

    let c_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {c_reg}, {gid}, {in_dhw};"));
    let rem1 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem1}, {gid}, {in_dhw};"));
    let id_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {id_reg}, {rem1}, {in_hw};"));
    let rem2 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem2}, {rem1}, {in_hw};"));
    let ih_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {ih_reg}, {rem2}, {in_w};"));
    let iw_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {iw_reg}, {rem2}, {in_w};"));

    b.comment("Initialize accumulator to zero");
    let acc = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        let zb: u32 = 0f32.to_bits();
        b.raw_ptx(&format!("mov.b32 {acc}, 0F{zb:08X};"));
    } else {
        let zb: u64 = 0f64.to_bits();
        b.raw_ptx(&format!("mov.b64 {acc}, 0D{zb:016X};"));
    }

    // Precompute output-plane strides.
    let out_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {out_hw}, {out_h}, {out_w};"));
    let out_dhw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {out_dhw}, {out_d}, {out_hw};"));

    let kh_kw_val: u32 = p.kernel_h * p.kernel_w;
    let kd_kh_kw_val: u32 = p.kernel_d * kh_kw_val;
    let kh_kw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {kh_kw}, {kh_kw_val};"));
    let kd_kh_kw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {kd_kh_kw}, {kd_kh_kw_val};"));

    // Scratch registers.
    let id_plus_pad = b.alloc_reg(PtxType::U32);
    let ih_plus_pad = b.alloc_reg(PtxType::U32);
    let iw_plus_pad = b.alloc_reg(PtxType::U32);
    let d_off = b.alloc_reg(PtxType::U32);
    let h_off = b.alloc_reg(PtxType::U32);
    let w_off = b.alloc_reg(PtxType::U32);
    let d_mod = b.alloc_reg(PtxType::U32);
    let h_mod = b.alloc_reg(PtxType::U32);
    let w_mod = b.alloc_reg(PtxType::U32);
    let od_reg = b.alloc_reg(PtxType::U32);
    let oh_reg = b.alloc_reg(PtxType::U32);
    let ow_reg = b.alloc_reg(PtxType::U32);
    let col_row = b.alloc_reg(PtxType::U32);
    let col_idx = b.alloc_reg(PtxType::U32);
    let row_off_reg = b.alloc_reg(PtxType::U32);
    let spatial_idx = b.alloc_reg(PtxType::U32);
    let c_kd_kh_kw = b.alloc_reg(PtxType::U32);
    let pred_dge = b.alloc_reg(PtxType::Pred);
    let pred_dmod = b.alloc_reg(PtxType::Pred);
    let pred_dlt = b.alloc_reg(PtxType::Pred);
    let pred_hge = b.alloc_reg(PtxType::Pred);
    let pred_hmod = b.alloc_reg(PtxType::Pred);
    let pred_hlt = b.alloc_reg(PtxType::Pred);
    let pred_wge = b.alloc_reg(PtxType::Pred);
    let pred_wmod = b.alloc_reg(PtxType::Pred);
    let pred_wlt = b.alloc_reg(PtxType::Pred);
    let laddr = b.alloc_reg(PtxType::U64);
    let loff = b.alloc_reg(PtxType::U64);
    let lidx64 = b.alloc_reg(PtxType::U64);
    let lval = b.alloc_reg(float_type);
    let od_times_ohw = b.alloc_reg(PtxType::U32);
    let oh_times_ow = b.alloc_reg(PtxType::U32);

    b.comment("Unrolled kernel loop over (kd, kh, kw)");
    for kd_val in 0..p.kernel_d {
        for kh_val in 0..p.kernel_h {
            for kw_val in 0..p.kernel_w {
                let kd_dil = kd_val * p.dilation_d;
                let kh_dil = kh_val * p.dilation_h;
                let kw_dil = kw_val * p.dilation_w;
                let k_flat: u32 = kd_val * kh_kw_val + kh_val * p.kernel_w + kw_val;
                let skip = b.fresh_label(&format!("c2i_skip_kd{kd_val}_kh{kh_val}_kw{kw_val}"));

                // id_plus_pad = id + pad_d
                b.raw_ptx(&format!("add.u32 {id_plus_pad}, {id_reg}, {};", p.pad_d));
                b.raw_ptx(&format!("setp.hs.u32 {pred_dge}, {id_plus_pad}, {kd_dil};"));
                b.raw_ptx(&format!("@!{pred_dge} bra {skip};"));
                b.raw_ptx(&format!("sub.u32 {d_off}, {id_plus_pad}, {kd_dil};"));
                b.raw_ptx(&format!("rem.u32 {d_mod}, {d_off}, {};", p.stride_d));
                b.raw_ptx(&format!("setp.eq.u32 {pred_dmod}, {d_mod}, 0;"));
                b.raw_ptx(&format!("@!{pred_dmod} bra {skip};"));
                b.raw_ptx(&format!("div.u32 {od_reg}, {d_off}, {};", p.stride_d));
                b.raw_ptx(&format!("setp.lo.u32 {pred_dlt}, {od_reg}, {out_d};"));
                b.raw_ptx(&format!("@!{pred_dlt} bra {skip};"));

                // ih check
                b.raw_ptx(&format!("add.u32 {ih_plus_pad}, {ih_reg}, {};", p.pad_h));
                b.raw_ptx(&format!("setp.hs.u32 {pred_hge}, {ih_plus_pad}, {kh_dil};"));
                b.raw_ptx(&format!("@!{pred_hge} bra {skip};"));
                b.raw_ptx(&format!("sub.u32 {h_off}, {ih_plus_pad}, {kh_dil};"));
                b.raw_ptx(&format!("rem.u32 {h_mod}, {h_off}, {};", p.stride_h));
                b.raw_ptx(&format!("setp.eq.u32 {pred_hmod}, {h_mod}, 0;"));
                b.raw_ptx(&format!("@!{pred_hmod} bra {skip};"));
                b.raw_ptx(&format!("div.u32 {oh_reg}, {h_off}, {};", p.stride_h));
                b.raw_ptx(&format!("setp.lo.u32 {pred_hlt}, {oh_reg}, {out_h};"));
                b.raw_ptx(&format!("@!{pred_hlt} bra {skip};"));

                // iw check
                b.raw_ptx(&format!("add.u32 {iw_plus_pad}, {iw_reg}, {};", p.pad_w));
                b.raw_ptx(&format!("setp.hs.u32 {pred_wge}, {iw_plus_pad}, {kw_dil};"));
                b.raw_ptx(&format!("@!{pred_wge} bra {skip};"));
                b.raw_ptx(&format!("sub.u32 {w_off}, {iw_plus_pad}, {kw_dil};"));
                b.raw_ptx(&format!("rem.u32 {w_mod}, {w_off}, {};", p.stride_w));
                b.raw_ptx(&format!("setp.eq.u32 {pred_wmod}, {w_mod}, 0;"));
                b.raw_ptx(&format!("@!{pred_wmod} bra {skip};"));
                b.raw_ptx(&format!("div.u32 {ow_reg}, {w_off}, {};", p.stride_w));
                b.raw_ptx(&format!("setp.lo.u32 {pred_wlt}, {ow_reg}, {out_w};"));
                b.raw_ptx(&format!("@!{pred_wlt} bra {skip};"));

                // col_row = c * kD*kH*kW + k_flat
                b.raw_ptx(&format!("mul.lo.u32 {c_kd_kh_kw}, {c_reg}, {kd_kh_kw};"));
                b.raw_ptx(&format!("add.u32 {col_row}, {c_kd_kh_kw}, {k_flat};"));

                // col spatial idx = od * oH*oW + oh * oW + ow
                b.raw_ptx(&format!("mul.lo.u32 {od_times_ohw}, {od_reg}, {out_hw};"));
                b.raw_ptx(&format!("mul.lo.u32 {oh_times_ow}, {oh_reg}, {out_w};"));
                b.raw_ptx(&format!(
                    "add.u32 {spatial_idx}, {od_times_ohw}, {oh_times_ow};"
                ));
                b.raw_ptx(&format!("add.u32 {spatial_idx}, {spatial_idx}, {ow_reg};"));

                // col_idx = col_row * out_dhw + spatial_idx
                b.raw_ptx(&format!("mul.lo.u32 {row_off_reg}, {col_row}, {out_dhw};"));
                b.raw_ptx(&format!("add.u32 {col_idx}, {row_off_reg}, {spatial_idx};"));

                // Load and accumulate.
                b.raw_ptx(&format!("cvt.u64.u32 {lidx64}, {col_idx};"));
                b.raw_ptx(&format!("mul.lo.u64 {loff}, {lidx64}, {elem_bytes};"));
                b.raw_ptx(&format!("add.u64 {laddr}, {col_ptr}, {loff};"));
                if float_type == PtxType::F32 {
                    b.raw_ptx(&format!("ld.global.f32 {lval}, [{laddr}];"));
                    b.raw_ptx(&format!("add.rn.f32 {acc}, {acc}, {lval};"));
                } else {
                    b.raw_ptx(&format!("ld.global.f64 {lval}, [{laddr}];"));
                    b.raw_ptx(&format!("add.rn.f64 {acc}, {acc}, {lval};"));
                }

                b.label(&skip);
            }
        }
    }

    b.comment("Store accumulated result");
    let out_idx64 = b.alloc_reg(PtxType::U64);
    let out_off64 = b.alloc_reg(PtxType::U64);
    let out_addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {out_idx64}, {gid};"));
    b.raw_ptx(&format!(
        "mul.lo.u64 {out_off64}, {out_idx64}, {elem_bytes};"
    ));
    b.raw_ptx(&format!("add.u64 {out_addr}, {out_ptr}, {out_off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("st.global.f32 [{out_addr}], {acc};"));
    } else {
        b.raw_ptx(&format!("st.global.f64 [{out_addr}], {acc};"));
    }

    b.label(&exit_label);
    b.ret();
}

// ---------------------------------------------------------------------------
// PTX Generation — direct 3x3x3 kernel
// ---------------------------------------------------------------------------

/// Captured parameters for the direct 3×3×3 kernel body emission.
#[derive(Debug, Clone, Copy)]
struct Direct3dParams {
    float_type: PtxType,
    elem_bytes: u32,
    stride_d: u32,
    stride_h: u32,
    stride_w: u32,
    pad_d: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_d: u32,
    dilation_h: u32,
    dilation_w: u32,
    in_channels_per_group: u32,
}

/// Generates PTX for a direct 3×3×3 convolution kernel.
///
/// Each thread computes one output element. The 27-tap (3×3×3) convolution
/// is fully unrolled. A runtime loop iterates over `in_channels_per_group`.
///
/// # Errors
///
/// Returns [`DnnError::InvalidArgument`] for unsupported precision or
/// non-3×3×3 kernel.
/// Returns [`DnnError::PtxGeneration`] if kernel building fails.
pub fn generate_direct3d_ptx(
    config: &Conv3dConfig,
    precision: &str,
    sm_version: SmVersion,
) -> DnnResult<String> {
    if config.kernel_d != 3 || config.kernel_h != 3 || config.kernel_w != 3 {
        return Err(DnnError::InvalidArgument(
            "direct3d kernel requires 3x3x3 kernel size".into(),
        ));
    }

    let kernel_name = format!("direct3d_3x3x3_{precision}");
    let float_type = match precision {
        "f32" => PtxType::F32,
        "f64" => PtxType::F64,
        other => {
            return Err(DnnError::InvalidArgument(format!(
                "unsupported precision for direct3d: {other}"
            )));
        }
    };
    let elem_bytes: u32 = match precision {
        "f32" => 4,
        _ => 8,
    };

    let params = Direct3dParams {
        float_type,
        elem_bytes,
        stride_d: config.stride_d as u32,
        stride_h: config.stride_h as u32,
        stride_w: config.stride_w as u32,
        pad_d: config.pad_d as u32,
        pad_h: config.pad_h as u32,
        pad_w: config.pad_w as u32,
        dilation_d: config.dilation_d as u32,
        dilation_h: config.dilation_h as u32,
        dilation_w: config.dilation_w as u32,
        in_channels_per_group: config.in_channels_per_group() as u32,
    };

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm_version)
        .param("input", PtxType::U64)
        .param("weight", PtxType::U64)
        .param("output", PtxType::U64)
        .param("in_d", PtxType::U32)
        .param("in_h", PtxType::U32)
        .param("in_w", PtxType::U32)
        .param("out_d", PtxType::U32)
        .param("out_h", PtxType::U32)
        .param("out_w", PtxType::U32)
        .param("out_channels_per_group", PtxType::U32)
        .param("total_output_elements", PtxType::U32)
        .body(move |b| {
            emit_direct3d_body(b, &params);
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Emits the direct 3×3×3 convolution kernel body.
///
/// Each thread computes one output element `(k, od, oh, ow)` by
/// iterating over `in_channels_per_group` and the 27 taps of the
/// 3×3×3 kernel window.
fn emit_direct3d_body(b: &mut oxicuda_ptx::builder::BodyBuilder<'_>, p: &Direct3dParams) {
    let float_type = p.float_type;
    let elem_bytes = p.elem_bytes;

    b.comment("=== Direct 3x3x3 Convolution Kernel ===");
    b.comment("Each thread: one output element (k, od, oh, ow).");

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_output_elements");
    let pred_bounds = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_bounds}, {gid}, {total};"));
    let exit_label = b.fresh_label("direct3d_exit");
    b.raw_ptx(&format!("@!{pred_bounds} bra {exit_label};"));

    let input_ptr = b.load_param_u64("input");
    let weight_ptr = b.load_param_u64("weight");
    let output_ptr = b.load_param_u64("output");
    let in_d = b.load_param_u32("in_d");
    let in_h = b.load_param_u32("in_h");
    let in_w = b.load_param_u32("in_w");
    let out_d = b.load_param_u32("out_d");
    let out_h = b.load_param_u32("out_h");
    let out_w = b.load_param_u32("out_w");
    let _out_cpg = b.load_param_u32("out_channels_per_group");

    b.comment("Decompose gid -> (k, od, oh, ow)");
    let out_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {out_hw}, {out_h}, {out_w};"));
    let out_dhw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {out_dhw}, {out_d}, {out_hw};"));

    let k_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {k_reg}, {gid}, {out_dhw};"));
    let rem1 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem1}, {gid}, {out_dhw};"));
    let od = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {od}, {rem1}, {out_hw};"));
    let rem2 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem2}, {rem1}, {out_hw};"));
    let oh = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {oh}, {rem2}, {out_w};"));
    let ow = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {ow}, {rem2}, {out_w};"));

    b.comment("Initialize accumulator");
    let acc = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        let zb: u32 = 0f32.to_bits();
        b.raw_ptx(&format!("mov.b32 {acc}, 0F{zb:08X};"));
    } else {
        let zb: u64 = 0f64.to_bits();
        b.raw_ptx(&format!("mov.b64 {acc}, 0D{zb:016X};"));
    }

    // Input spatial strides.
    let in_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {in_hw}, {in_h}, {in_w};"));
    let in_dhw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {in_dhw}, {in_d}, {in_hw};"));

    // Scratch registers.
    let id_reg = b.alloc_reg(PtxType::U32);
    let ih_reg = b.alloc_reg(PtxType::U32);
    let iw_reg = b.alloc_reg(PtxType::U32);
    let pred_d = b.alloc_reg(PtxType::Pred);
    let pred_h = b.alloc_reg(PtxType::Pred);
    let pred_w = b.alloc_reg(PtxType::Pred);
    let in_idx = b.alloc_reg(PtxType::U32);
    let w_idx = b.alloc_reg(PtxType::U32);
    let addr64 = b.alloc_reg(PtxType::U64);
    let off64 = b.alloc_reg(PtxType::U64);
    let iaddr = b.alloc_reg(PtxType::U64);
    let waddr = b.alloc_reg(PtxType::U64);
    let ival = b.alloc_reg(float_type);
    let wval = b.alloc_reg(float_type);
    let prod = b.alloc_reg(float_type);

    // Channel loop.
    let c_reg = b.alloc_reg(PtxType::U32);
    let c_limit = p.in_channels_per_group;
    let c_loop = b.fresh_label("d3d_c_loop");
    let c_done = b.fresh_label("d3d_c_done");
    let pred_c = b.alloc_reg(PtxType::Pred);

    // Weight layout: [out_cpg, in_cpg, 3, 3, 3] — 27 elements per (k, c).
    // weight_idx = k * in_cpg * 27 + c * 27 + kd*9 + kh*3 + kw
    let w_base = b.alloc_reg(PtxType::U32);
    let k_times_cpg27 = b.alloc_reg(PtxType::U32);
    let cpg27: u32 = c_limit * 27;
    b.raw_ptx(&format!("mul.lo.u32 {k_times_cpg27}, {k_reg}, {cpg27};"));

    b.raw_ptx(&format!("mov.u32 {c_reg}, 0;"));
    b.label(&c_loop);
    b.raw_ptx(&format!("setp.lo.u32 {pred_c}, {c_reg}, {c_limit};"));
    b.raw_ptx(&format!("@!{pred_c} bra {c_done};"));

    // w_base = k * in_cpg * 27 + c * 27
    b.raw_ptx(&format!("mul.lo.u32 {w_base}, {c_reg}, 27;"));
    b.raw_ptx(&format!("add.u32 {w_base}, {w_base}, {k_times_cpg27};"));

    // Unrolled 3×3×3 taps.
    for kd_val in 0u32..3 {
        for kh_val in 0u32..3 {
            for kw_val in 0u32..3 {
                let kd_dil = kd_val * p.dilation_d;
                let kh_dil = kh_val * p.dilation_h;
                let kw_dil = kw_val * p.dilation_w;
                let k_flat = kd_val * 9 + kh_val * 3 + kw_val;
                let skip = b.fresh_label(&format!("d3d_skip_{kd_val}_{kh_val}_{kw_val}"));

                // id = od * stride_d + kd_dil - pad_d
                b.raw_ptx(&format!(
                    "mad.lo.u32 {id_reg}, {od}, {}, {kd_dil};",
                    p.stride_d
                ));
                b.raw_ptx(&format!("setp.hs.u32 {pred_d}, {id_reg}, {};", p.pad_d));
                b.raw_ptx(&format!("@!{pred_d} bra {skip};"));
                b.raw_ptx(&format!("sub.u32 {id_reg}, {id_reg}, {};", p.pad_d));
                b.raw_ptx(&format!("setp.lo.u32 {pred_d}, {id_reg}, {in_d};"));
                b.raw_ptx(&format!("@!{pred_d} bra {skip};"));

                // ih
                b.raw_ptx(&format!(
                    "mad.lo.u32 {ih_reg}, {oh}, {}, {kh_dil};",
                    p.stride_h
                ));
                b.raw_ptx(&format!("setp.hs.u32 {pred_h}, {ih_reg}, {};", p.pad_h));
                b.raw_ptx(&format!("@!{pred_h} bra {skip};"));
                b.raw_ptx(&format!("sub.u32 {ih_reg}, {ih_reg}, {};", p.pad_h));
                b.raw_ptx(&format!("setp.lo.u32 {pred_h}, {ih_reg}, {in_h};"));
                b.raw_ptx(&format!("@!{pred_h} bra {skip};"));

                // iw
                b.raw_ptx(&format!(
                    "mad.lo.u32 {iw_reg}, {ow}, {}, {kw_dil};",
                    p.stride_w
                ));
                b.raw_ptx(&format!("setp.hs.u32 {pred_w}, {iw_reg}, {};", p.pad_w));
                b.raw_ptx(&format!("@!{pred_w} bra {skip};"));
                b.raw_ptx(&format!("sub.u32 {iw_reg}, {iw_reg}, {};", p.pad_w));
                b.raw_ptx(&format!("setp.lo.u32 {pred_w}, {iw_reg}, {in_w};"));
                b.raw_ptx(&format!("@!{pred_w} bra {skip};"));

                // input index = c * D*H*W + id * H*W + ih * W + iw
                b.raw_ptx(&format!("mul.lo.u32 {in_idx}, {c_reg}, {in_dhw};"));
                b.raw_ptx(&format!(
                    "mad.lo.u32 {in_idx}, {id_reg}, {in_hw}, {in_idx};"
                ));
                b.raw_ptx(&format!("mad.lo.u32 {in_idx}, {ih_reg}, {in_w}, {in_idx};"));
                b.raw_ptx(&format!("add.u32 {in_idx}, {in_idx}, {iw_reg};"));

                // Load input value.
                b.raw_ptx(&format!("cvt.u64.u32 {addr64}, {in_idx};"));
                b.raw_ptx(&format!("mul.lo.u64 {off64}, {addr64}, {elem_bytes};"));
                b.raw_ptx(&format!("add.u64 {iaddr}, {input_ptr}, {off64};"));
                if float_type == PtxType::F32 {
                    b.raw_ptx(&format!("ld.global.f32 {ival}, [{iaddr}];"));
                } else {
                    b.raw_ptx(&format!("ld.global.f64 {ival}, [{iaddr}];"));
                }

                // weight index = w_base + k_flat
                b.raw_ptx(&format!("add.u32 {w_idx}, {w_base}, {k_flat};"));
                b.raw_ptx(&format!("cvt.u64.u32 {addr64}, {w_idx};"));
                b.raw_ptx(&format!("mul.lo.u64 {off64}, {addr64}, {elem_bytes};"));
                b.raw_ptx(&format!("add.u64 {waddr}, {weight_ptr}, {off64};"));
                if float_type == PtxType::F32 {
                    b.raw_ptx(&format!("ld.global.f32 {wval}, [{waddr}];"));
                    b.raw_ptx(&format!("mul.rn.f32 {prod}, {ival}, {wval};"));
                    b.raw_ptx(&format!("add.rn.f32 {acc}, {acc}, {prod};"));
                } else {
                    b.raw_ptx(&format!("ld.global.f64 {wval}, [{waddr}];"));
                    b.raw_ptx(&format!("mul.rn.f64 {prod}, {ival}, {wval};"));
                    b.raw_ptx(&format!("add.rn.f64 {acc}, {acc}, {prod};"));
                }

                b.label(&skip);
            }
        }
    }

    // Advance channel counter.
    b.raw_ptx(&format!("add.u32 {c_reg}, {c_reg}, 1;"));
    b.raw_ptx(&format!("bra {c_loop};"));
    b.label(&c_done);

    b.comment("Store output");
    let out_idx64 = b.alloc_reg(PtxType::U64);
    let out_off64 = b.alloc_reg(PtxType::U64);
    let out_addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {out_idx64}, {gid};"));
    b.raw_ptx(&format!(
        "mul.lo.u64 {out_off64}, {out_idx64}, {elem_bytes};"
    ));
    b.raw_ptx(&format!("add.u64 {out_addr}, {output_ptr}, {out_off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("st.global.f32 [{out_addr}], {acc};"));
    } else {
        b.raw_ptx(&format!("st.global.f64 [{out_addr}], {acc};"));
    }

    b.label(&exit_label);
    b.ret();
}

// ---------------------------------------------------------------------------
// PTX Generation — weight gradient (wgrad) 3D kernel
// ---------------------------------------------------------------------------

/// Captured parameters for the wgrad3d kernel body emission.
#[derive(Debug, Clone, Copy)]
struct Wgrad3dParams {
    float_type: PtxType,
    elem_bytes: u32,
    kernel_d: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_d: u32,
    stride_h: u32,
    stride_w: u32,
    pad_d: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_d: u32,
    dilation_h: u32,
    dilation_w: u32,
    in_channels_per_group: u32,
    out_channels_per_group: u32,
}

/// Captured spatial dimensions for the wgrad3d kernel.
#[derive(Debug, Clone, Copy)]
struct Wgrad3dDims {
    input_d: u32,
    input_h: u32,
    input_w: u32,
    output_d: u32,
    output_h: u32,
    output_w: u32,
}

/// Generates PTX for the weight gradient (wgrad) 3D kernel.
///
/// Each thread computes one element of `dL/dW` at position `(k, c, kd, kh, kw)`
/// by summing `input[n, c, ...] × dL/dY[n, k, ...]` over all batch samples and
/// output spatial positions.
///
/// # Errors
///
/// Returns [`DnnError::InvalidArgument`] for unsupported precision.
/// Returns [`DnnError::PtxGeneration`] if kernel building fails.
pub fn generate_wgrad3d_ptx(
    config: &Conv3dConfig,
    batch_size: usize,
    input_d: usize,
    input_h: usize,
    input_w: usize,
    precision: &str,
    sm_version: SmVersion,
) -> DnnResult<String> {
    let kernel_name = format!("wgrad3d_{precision}");
    let float_type = match precision {
        "f32" => PtxType::F32,
        "f64" => PtxType::F64,
        other => {
            return Err(DnnError::InvalidArgument(format!(
                "unsupported precision for wgrad3d: {other}"
            )));
        }
    };
    let elem_bytes: u32 = match precision {
        "f32" => 4,
        _ => 8,
    };

    let (output_d, output_h, output_w) = config.output_size(input_d, input_h, input_w);

    let params = Wgrad3dParams {
        float_type,
        elem_bytes,
        kernel_d: config.kernel_d as u32,
        kernel_h: config.kernel_h as u32,
        kernel_w: config.kernel_w as u32,
        stride_d: config.stride_d as u32,
        stride_h: config.stride_h as u32,
        stride_w: config.stride_w as u32,
        pad_d: config.pad_d as u32,
        pad_h: config.pad_h as u32,
        pad_w: config.pad_w as u32,
        dilation_d: config.dilation_d as u32,
        dilation_h: config.dilation_h as u32,
        dilation_w: config.dilation_w as u32,
        in_channels_per_group: config.in_channels_per_group() as u32,
        out_channels_per_group: config.out_channels_per_group() as u32,
    };

    let dims = Wgrad3dDims {
        input_d: input_d as u32,
        input_h: input_h as u32,
        input_w: input_w as u32,
        output_d: output_d as u32,
        output_h: output_h as u32,
        output_w: output_w as u32,
    };

    let batch_size_u32 = batch_size as u32;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm_version)
        .param("input", PtxType::U64)
        .param("grad_output", PtxType::U64)
        .param("grad_weight", PtxType::U64)
        .param("total_weight_elements", PtxType::U32)
        .body(move |b| {
            emit_wgrad3d_body(b, &params, &dims, batch_size_u32);
        })
        .build()
        .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

/// Emits the wgrad3d kernel body.
///
/// Each thread computes one weight gradient element `(k, c, kd, kh, kw)`.
/// It iterates over all batch samples and output spatial positions, accumulating
/// `input[n, c, id, ih, iw] × grad_output[n, k, od, oh, ow]` where
/// `id = od × stride_d + kd × dilation_d - pad_d` (and similarly for h, w).
fn emit_wgrad3d_body(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    p: &Wgrad3dParams,
    dims: &Wgrad3dDims,
    batch_size: u32,
) {
    let float_type = p.float_type;
    let elem_bytes = p.elem_bytes;

    b.comment("=== Weight Gradient 3D Kernel ===");
    b.comment("Each thread: one weight element dW[k, c, kd, kh, kw].");

    // Global thread ID and bounds check.
    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_weight_elements");
    let pred_bounds = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred_bounds}, {gid}, {total};"));
    let exit_label = b.fresh_label("wgrad3d_exit");
    b.raw_ptx(&format!("@!{pred_bounds} bra {exit_label};"));

    let input_ptr = b.load_param_u64("input");
    let grad_out_ptr = b.load_param_u64("grad_output");
    let grad_w_ptr = b.load_param_u64("grad_weight");

    // Decompose gid -> (k, c, kd, kh, kw).
    let kh_kw: u32 = p.kernel_h * p.kernel_w;
    let kd_kh_kw: u32 = p.kernel_d * kh_kw;
    let c_kd_kh_kw: u32 = p.in_channels_per_group * kd_kh_kw;

    let k_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {k_reg}, {gid}, {c_kd_kh_kw};"));
    let rem1 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem1}, {gid}, {c_kd_kh_kw};"));
    let c_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {c_reg}, {rem1}, {kd_kh_kw};"));
    let rem2 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem2}, {rem1}, {kd_kh_kw};"));
    let kd_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {kd_reg}, {rem2}, {kh_kw};"));
    let rem3 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {rem3}, {rem2}, {kh_kw};"));
    let kh_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {kh_reg}, {rem3}, {};", p.kernel_w));
    let kw_reg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {kw_reg}, {rem3}, {};", p.kernel_w));

    // Initialize accumulator.
    let acc = b.alloc_reg(float_type);
    if float_type == PtxType::F32 {
        let zb: u32 = 0f32.to_bits();
        b.raw_ptx(&format!("mov.b32 {acc}, 0F{zb:08X};"));
    } else {
        let zb: u64 = 0f64.to_bits();
        b.raw_ptx(&format!("mov.b64 {acc}, 0D{zb:016X};"));
    }

    // Precompute dilated kernel offsets.
    let kd_dil = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {kd_dil}, {kd_reg}, {};", p.dilation_d));
    let kh_dil = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {kh_dil}, {kh_reg}, {};", p.dilation_h));
    let kw_dil = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {kw_dil}, {kw_reg}, {};", p.dilation_w));

    // Spatial strides.
    let in_hw: u32 = dims.input_h * dims.input_w;
    let in_dhw: u32 = dims.input_d * in_hw;
    let out_hw: u32 = dims.output_h * dims.output_w;
    let out_dhw: u32 = dims.output_d * out_hw;

    // Scratch registers.
    let n_reg = b.alloc_reg(PtxType::U32);
    let od_reg = b.alloc_reg(PtxType::U32);
    let oh_reg = b.alloc_reg(PtxType::U32);
    let ow_reg = b.alloc_reg(PtxType::U32);
    let id_reg = b.alloc_reg(PtxType::U32);
    let ih_reg = b.alloc_reg(PtxType::U32);
    let iw_reg = b.alloc_reg(PtxType::U32);
    let pred_d = b.alloc_reg(PtxType::Pred);
    let pred_h = b.alloc_reg(PtxType::Pred);
    let pred_w = b.alloc_reg(PtxType::Pred);
    let in_idx = b.alloc_reg(PtxType::U32);
    let go_idx = b.alloc_reg(PtxType::U32);
    let addr64 = b.alloc_reg(PtxType::U64);
    let off64 = b.alloc_reg(PtxType::U64);
    let iaddr = b.alloc_reg(PtxType::U64);
    let goaddr = b.alloc_reg(PtxType::U64);
    let ival = b.alloc_reg(float_type);
    let goval = b.alloc_reg(float_type);
    let prod = b.alloc_reg(float_type);
    let pred_n = b.alloc_reg(PtxType::Pred);
    let pred_od = b.alloc_reg(PtxType::Pred);
    let pred_oh = b.alloc_reg(PtxType::Pred);
    let pred_ow = b.alloc_reg(PtxType::Pred);

    // Batch loop.
    let n_loop = b.fresh_label("wg_n_loop");
    let n_done = b.fresh_label("wg_n_done");
    b.raw_ptx(&format!("mov.u32 {n_reg}, 0;"));
    b.label(&n_loop);
    b.raw_ptx(&format!("setp.lo.u32 {pred_n}, {n_reg}, {batch_size};"));
    b.raw_ptx(&format!("@!{pred_n} bra {n_done};"));

    // Output depth loop.
    let od_loop = b.fresh_label("wg_od_loop");
    let od_done = b.fresh_label("wg_od_done");
    b.raw_ptx(&format!("mov.u32 {od_reg}, 0;"));
    b.label(&od_loop);
    b.raw_ptx(&format!(
        "setp.lo.u32 {pred_od}, {od_reg}, {};",
        dims.output_d
    ));
    b.raw_ptx(&format!("@!{pred_od} bra {od_done};"));

    // Compute id = od * stride_d + kd_dil - pad_d  (bounds check).
    b.raw_ptx(&format!(
        "mad.lo.u32 {id_reg}, {od_reg}, {}, {kd_dil};",
        p.stride_d
    ));
    b.raw_ptx(&format!("setp.hs.u32 {pred_d}, {id_reg}, {};", p.pad_d));
    let skip_od = b.fresh_label("wg_skip_od");
    b.raw_ptx(&format!("@!{pred_d} bra {skip_od};"));
    b.raw_ptx(&format!("sub.u32 {id_reg}, {id_reg}, {};", p.pad_d));
    b.raw_ptx(&format!(
        "setp.lo.u32 {pred_d}, {id_reg}, {};",
        dims.input_d
    ));
    b.raw_ptx(&format!("@!{pred_d} bra {skip_od};"));

    // Output height loop.
    let oh_loop = b.fresh_label("wg_oh_loop");
    let oh_done = b.fresh_label("wg_oh_done");
    b.raw_ptx(&format!("mov.u32 {oh_reg}, 0;"));
    b.label(&oh_loop);
    b.raw_ptx(&format!(
        "setp.lo.u32 {pred_oh}, {oh_reg}, {};",
        dims.output_h
    ));
    b.raw_ptx(&format!("@!{pred_oh} bra {oh_done};"));

    // Compute ih.
    b.raw_ptx(&format!(
        "mad.lo.u32 {ih_reg}, {oh_reg}, {}, {kh_dil};",
        p.stride_h
    ));
    b.raw_ptx(&format!("setp.hs.u32 {pred_h}, {ih_reg}, {};", p.pad_h));
    let skip_oh = b.fresh_label("wg_skip_oh");
    b.raw_ptx(&format!("@!{pred_h} bra {skip_oh};"));
    b.raw_ptx(&format!("sub.u32 {ih_reg}, {ih_reg}, {};", p.pad_h));
    b.raw_ptx(&format!(
        "setp.lo.u32 {pred_h}, {ih_reg}, {};",
        dims.input_h
    ));
    b.raw_ptx(&format!("@!{pred_h} bra {skip_oh};"));

    // Output width loop.
    let ow_loop = b.fresh_label("wg_ow_loop");
    let ow_done = b.fresh_label("wg_ow_done");
    b.raw_ptx(&format!("mov.u32 {ow_reg}, 0;"));
    b.label(&ow_loop);
    b.raw_ptx(&format!(
        "setp.lo.u32 {pred_ow}, {ow_reg}, {};",
        dims.output_w
    ));
    b.raw_ptx(&format!("@!{pred_ow} bra {ow_done};"));

    // Compute iw.
    b.raw_ptx(&format!(
        "mad.lo.u32 {iw_reg}, {ow_reg}, {}, {kw_dil};",
        p.stride_w
    ));
    b.raw_ptx(&format!("setp.hs.u32 {pred_w}, {iw_reg}, {};", p.pad_w));
    let skip_ow = b.fresh_label("wg_skip_ow");
    b.raw_ptx(&format!("@!{pred_w} bra {skip_ow};"));
    b.raw_ptx(&format!("sub.u32 {iw_reg}, {iw_reg}, {};", p.pad_w));
    b.raw_ptx(&format!(
        "setp.lo.u32 {pred_w}, {iw_reg}, {};",
        dims.input_w
    ));
    b.raw_ptx(&format!("@!{pred_w} bra {skip_ow};"));

    // input index = n * C_in/g * D * H * W + c * D*H*W + id * H*W + ih * W + iw
    let n_offset: u32 = p.in_channels_per_group * in_dhw;
    b.raw_ptx(&format!("mul.lo.u32 {in_idx}, {n_reg}, {n_offset};"));
    b.raw_ptx(&format!(
        "mad.lo.u32 {in_idx}, {c_reg}, {in_dhw}, {in_idx};"
    ));
    b.raw_ptx(&format!(
        "mad.lo.u32 {in_idx}, {id_reg}, {in_hw}, {in_idx};"
    ));
    b.raw_ptx(&format!(
        "mad.lo.u32 {in_idx}, {ih_reg}, {}, {in_idx};",
        dims.input_w
    ));
    b.raw_ptx(&format!("add.u32 {in_idx}, {in_idx}, {iw_reg};"));

    // grad_output index = n * C_out/g * oD * oH * oW + k * oD*oH*oW + od*oH*oW + oh*oW + ow
    let go_n_offset: u32 = p.out_channels_per_group * out_dhw;
    b.raw_ptx(&format!("mul.lo.u32 {go_idx}, {n_reg}, {go_n_offset};"));
    b.raw_ptx(&format!(
        "mad.lo.u32 {go_idx}, {k_reg}, {out_dhw}, {go_idx};"
    ));
    b.raw_ptx(&format!(
        "mad.lo.u32 {go_idx}, {od_reg}, {out_hw}, {go_idx};"
    ));
    b.raw_ptx(&format!(
        "mad.lo.u32 {go_idx}, {oh_reg}, {}, {go_idx};",
        dims.output_w
    ));
    b.raw_ptx(&format!("add.u32 {go_idx}, {go_idx}, {ow_reg};"));

    // Load input value.
    b.raw_ptx(&format!("cvt.u64.u32 {addr64}, {in_idx};"));
    b.raw_ptx(&format!("mul.lo.u64 {off64}, {addr64}, {elem_bytes};"));
    b.raw_ptx(&format!("add.u64 {iaddr}, {input_ptr}, {off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("ld.global.f32 {ival}, [{iaddr}];"));
    } else {
        b.raw_ptx(&format!("ld.global.f64 {ival}, [{iaddr}];"));
    }

    // Load grad_output value.
    b.raw_ptx(&format!("cvt.u64.u32 {addr64}, {go_idx};"));
    b.raw_ptx(&format!("mul.lo.u64 {off64}, {addr64}, {elem_bytes};"));
    b.raw_ptx(&format!("add.u64 {goaddr}, {grad_out_ptr}, {off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("ld.global.f32 {goval}, [{goaddr}];"));
        b.raw_ptx(&format!("mul.rn.f32 {prod}, {ival}, {goval};"));
        b.raw_ptx(&format!("add.rn.f32 {acc}, {acc}, {prod};"));
    } else {
        b.raw_ptx(&format!("ld.global.f64 {goval}, [{goaddr}];"));
        b.raw_ptx(&format!("mul.rn.f64 {prod}, {ival}, {goval};"));
        b.raw_ptx(&format!("add.rn.f64 {acc}, {acc}, {prod};"));
    }

    // skip_ow label and ow loop end.
    b.label(&skip_ow);
    b.raw_ptx(&format!("add.u32 {ow_reg}, {ow_reg}, 1;"));
    b.raw_ptx(&format!("bra {ow_loop};"));
    b.label(&ow_done);

    // skip_oh label and oh loop end.
    b.label(&skip_oh);
    b.raw_ptx(&format!("add.u32 {oh_reg}, {oh_reg}, 1;"));
    b.raw_ptx(&format!("bra {oh_loop};"));
    b.label(&oh_done);

    // skip_od label and od loop end.
    b.label(&skip_od);
    b.raw_ptx(&format!("add.u32 {od_reg}, {od_reg}, 1;"));
    b.raw_ptx(&format!("bra {od_loop};"));
    b.label(&od_done);

    // Batch loop end.
    b.raw_ptx(&format!("add.u32 {n_reg}, {n_reg}, 1;"));
    b.raw_ptx(&format!("bra {n_loop};"));
    b.label(&n_done);

    // Store accumulated weight gradient.
    b.comment("Store weight gradient");
    let w_idx64 = b.alloc_reg(PtxType::U64);
    let w_off64 = b.alloc_reg(PtxType::U64);
    let w_addr = b.alloc_reg(PtxType::U64);
    b.raw_ptx(&format!("cvt.u64.u32 {w_idx64}, {gid};"));
    b.raw_ptx(&format!("mul.lo.u64 {w_off64}, {w_idx64}, {elem_bytes};"));
    b.raw_ptx(&format!("add.u64 {w_addr}, {grad_w_ptr}, {w_off64};"));
    if float_type == PtxType::F32 {
        b.raw_ptx(&format!("st.global.f32 [{w_addr}], {acc};"));
    } else {
        b.raw_ptx(&format!("st.global.f64 [{w_addr}], {acc};"));
    }

    b.label(&exit_label);
    b.ret();
}
