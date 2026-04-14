//! SPIR-V compute kernel generators for neural-network operations.
//!
//! This module extends the Level Zero SPIR-V generator with Conv2D and
//! scaled dot-product attention kernels.  Both use the OpenCL execution
//! model (Kernel + Physical64 + Addresses) so they can be consumed by
//! `zeModuleCreate`.

use super::spirv::{
    EXECUTION_MODEL_KERNEL, FUNCTION_CONTROL_NONE, OP_F_ADD, OP_F_DIV, OP_F_MUL, OP_F_SUB,
    OP_I_ADD, OP_I_MUL, OP_U_DIV, OP_U_LESS_THAN, OP_U_MOD, OPENCL_EXP, OPENCL_FMAX,
    STORAGE_CLASS_FUNCTION, SpvModule, WORKGROUP_SIZE, emit_preamble, load_gid_x,
};

/// SPIR-V opcode for `OpISub` (integer subtract).
const OP_I_SUB: u32 = 130;
/// SPIR-V opcode for `OpFOrdGreaterThan` (ordered float >).
const OP_F_ORD_GT: u32 = 188;

// ─── Conv2D compute kernel ──────────────────────────────────

/// Generate an OpenCL SPIR-V compute kernel for 2-D convolution (NCHW layout).
///
/// Each work-item computes one output element.
///
/// Kernel parameters (all passed via `zeKernelSetArgumentValue`):
///
/// ```text
/// (CrossWorkgroup float* input,
///  CrossWorkgroup float* filter,
///  CrossWorkgroup float* output)
/// ```
///
/// All dimension constants are baked in as `OpConstant`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_spirv(
    n: u32,
    c_in: u32,
    h_in: u32,
    w_in: u32,
    k_out: u32,
    fh: u32,
    fw: u32,
    oh: u32,
    ow: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
) -> Vec<u32> {
    let mut m = SpvModule::new();
    let b = emit_preamble(&mut m);

    let main_fn = m.alloc_id();
    let fn_ty = m.alloc_id();
    let p_input = m.alloc_id();
    let p_filter = m.alloc_id();
    let p_output = m.alloc_id();

    // Function type: void(float*, float*, float*)
    m.emit_type_function(
        fn_ty,
        b.ty_void,
        &[
            b.ty_ptr_cross_float,
            b.ty_ptr_cross_float,
            b.ty_ptr_cross_float,
        ],
    );

    m.emit_entry_point(EXECUTION_MODEL_KERNEL, main_fn, "main", &[b.var_gid]);
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    // Emit constants for dimensions
    let c_n = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_n, n);
    let c_c_in = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_c_in, c_in);
    let c_h_in = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_h_in, h_in);
    let c_w_in = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_w_in, w_in);
    let c_k_out = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_k_out, k_out);
    let c_fh = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_fh, fh);
    let c_fw = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_fw, fw);
    let c_oh = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_oh, oh);
    let c_ow = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_ow, ow);
    let c_stride_h = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_stride_h, stride_h);
    let c_stride_w = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_stride_w, stride_w);
    let c_pad_h = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_pad_h, pad_h);
    let c_pad_w = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_pad_w, pad_w);

    // Labels
    let label_entry = m.alloc_id();
    let label_body = m.alloc_id();
    let label_merge = m.alloc_id();

    // Function
    m.emit_function(b.ty_void, main_fn, FUNCTION_CONTROL_NONE, fn_ty);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_input);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_filter);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_output);
    m.emit_label(label_entry);

    let gid = load_gid_x(&mut m, &b);

    // total = n * k_out * oh * ow
    let t1 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, t1, c_n, c_k_out]);
    let t2 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, t2, t1, c_oh]);
    let total = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, total, t2, c_ow]);

    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, total]);
    m.emit_selection_merge(label_merge);
    m.emit_branch_conditional(cond, label_body, label_merge);

    m.emit_label(label_body);

    // Decompose gid -> (b_idx, kf, oy, ox)
    let ox = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, ox, gid, c_ow]);
    let tmp1 = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, tmp1, gid, c_ow]);
    let oy = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, oy, tmp1, c_oh]);
    let tmp2 = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, tmp2, tmp1, c_oh]);
    let kf = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, kf, tmp2, c_k_out]);
    let b_idx = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, b_idx, tmp2, c_k_out]);

    // Accumulator variable
    let var_acc = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_acc, b.c_float_0);

    // Flatten ci * fh * fw
    let flat_total_id = m.alloc_id();
    let flat_t1 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, flat_t1, c_c_in, c_fh]);
    m.emit(OP_I_MUL, &[b.ty_uint, flat_total_id, flat_t1, c_fw]);

    let var_flat = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_flat, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_flat, b.c_uint_0);

    let lbl_loop_hdr = m.alloc_id();
    let lbl_loop_body = m.alloc_id();
    let lbl_loop_cont = m.alloc_id();
    let lbl_loop_merge = m.alloc_id();

    m.emit_branch(lbl_loop_hdr);

    // ── Loop header ──
    m.emit_label(lbl_loop_hdr);
    let flat_val = m.alloc_id();
    m.emit_load(b.ty_uint, flat_val, var_flat);
    let loop_cond = m.alloc_id();
    m.emit(
        OP_U_LESS_THAN,
        &[b.ty_bool, loop_cond, flat_val, flat_total_id],
    );
    m.emit_loop_merge(lbl_loop_merge, lbl_loop_cont);
    m.emit_branch_conditional(loop_cond, lbl_loop_body, lbl_loop_merge);

    // ── Loop body ──
    m.emit_label(lbl_loop_body);

    // Decompose flat_val -> (ci, fy, fx)
    let fx = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, fx, flat_val, c_fw]);
    let ftmp1 = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, ftmp1, flat_val, c_fw]);
    let fy = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, fy, ftmp1, c_fh]);
    let ci = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, ci, ftmp1, c_fh]);

    // iy_raw = oy * stride_h + fy, ix_raw = ox * stride_w + fx
    let oy_sh = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, oy_sh, oy, c_stride_h]);
    let iy_raw = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, iy_raw, oy_sh, fy]);
    let ox_sw = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, ox_sw, ox, c_stride_w]);
    let ix_raw = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ix_raw, ox_sw, fx]);

    // Bounds check: iy_raw >= pad_h  &&  (iy_raw - pad_h) < h_in
    //           &&  ix_raw >= pad_w  &&  (ix_raw - pad_w) < w_in
    let lbl_skip = m.alloc_id();

    let iy_lt_pad = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, iy_lt_pad, iy_raw, c_pad_h]);
    let lbl_iy_ok = m.alloc_id();
    m.emit_selection_merge(lbl_skip);
    m.emit_branch_conditional(iy_lt_pad, lbl_skip, lbl_iy_ok);

    m.emit_label(lbl_iy_ok);
    let iy_real = m.alloc_id();
    m.emit(OP_I_SUB, &[b.ty_uint, iy_real, iy_raw, c_pad_h]);
    let iy_in_bounds = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, iy_in_bounds, iy_real, c_h_in]);
    let lbl_ix_check = m.alloc_id();
    m.emit_selection_merge(lbl_skip);
    m.emit_branch_conditional(iy_in_bounds, lbl_ix_check, lbl_skip);

    m.emit_label(lbl_ix_check);
    let ix_lt_pad = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, ix_lt_pad, ix_raw, c_pad_w]);
    let lbl_ix_ok = m.alloc_id();
    m.emit_selection_merge(lbl_skip);
    m.emit_branch_conditional(ix_lt_pad, lbl_skip, lbl_ix_ok);

    m.emit_label(lbl_ix_ok);
    let ix_real = m.alloc_id();
    m.emit(OP_I_SUB, &[b.ty_uint, ix_real, ix_raw, c_pad_w]);
    let ix_in_bounds = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, ix_in_bounds, ix_real, c_w_in]);
    let lbl_accum = m.alloc_id();
    m.emit_selection_merge(lbl_skip);
    m.emit_branch_conditional(ix_in_bounds, lbl_accum, lbl_skip);

    m.emit_label(lbl_accum);

    // input_idx = ((b_idx * c_in + ci) * h_in + iy_real) * w_in + ix_real
    let in1 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, in1, b_idx, c_c_in]);
    let in2 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, in2, in1, ci]);
    let in3 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, in3, in2, c_h_in]);
    let in4 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, in4, in3, iy_real]);
    let in5 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, in5, in4, c_w_in]);
    let in_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, in_idx, in5, ix_real]);

    let inp_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, inp_ptr, p_input, in_idx);
    let inp_val = m.alloc_id();
    m.emit_load(b.ty_float, inp_val, inp_ptr);

    // filter_idx = ((kf * c_in + ci) * fh + fy) * fw + fx
    let f1 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, f1, kf, c_c_in]);
    let f2 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, f2, f1, ci]);
    let f3 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, f3, f2, c_fh]);
    let f4 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, f4, f3, fy]);
    let f5 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, f5, f4, c_fw]);
    let flt_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, flt_idx, f5, fx]);

    let flt_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, flt_ptr, p_filter, flt_idx);
    let flt_val = m.alloc_id();
    m.emit_load(b.ty_float, flt_val, flt_ptr);

    // acc += inp * flt
    let prod = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, prod, inp_val, flt_val]);
    let old_acc = m.alloc_id();
    m.emit_load(b.ty_float, old_acc, var_acc);
    let new_acc = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, new_acc, old_acc, prod]);
    m.emit_store(var_acc, new_acc);

    m.emit_branch(lbl_skip);

    m.emit_label(lbl_skip);
    m.emit_branch(lbl_loop_cont);

    // ── Loop continue ──
    m.emit_label(lbl_loop_cont);
    let flat_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, flat_inc, flat_val, b.c_uint_1]);
    m.emit_store(var_flat, flat_inc);
    m.emit_branch(lbl_loop_hdr);

    // ── Loop merge: store result ──
    m.emit_label(lbl_loop_merge);

    let final_acc = m.alloc_id();
    m.emit_load(b.ty_float, final_acc, var_acc);

    let out_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, out_ptr, p_output, gid);
    m.emit_store(out_ptr, final_acc);

    m.emit_branch(label_merge);

    m.emit_label(label_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── Attention compute kernel ───────────────────────────────

/// Generate an OpenCL SPIR-V compute kernel for scaled dot-product attention.
///
/// Each work-item handles one (batch_head, query_position) pair.
///
/// Kernel parameters:
///
/// ```text
/// (CrossWorkgroup float* Q,
///  CrossWorkgroup float* K,
///  CrossWorkgroup float* V,
///  CrossWorkgroup float* O)
/// ```
///
/// Dimension constants are baked in as `OpConstant`.
#[allow(clippy::too_many_arguments)]
pub fn attention_spirv(
    batch_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    head_dim: u32,
    scale: f32,
    causal: bool,
) -> Vec<u32> {
    let mut m = SpvModule::new();
    let b = emit_preamble(&mut m);

    let main_fn = m.alloc_id();
    let fn_ty = m.alloc_id();
    let p_q = m.alloc_id();
    let p_k = m.alloc_id();
    let p_v = m.alloc_id();
    let p_o = m.alloc_id();

    // Function type: void(float*, float*, float*, float*)
    m.emit_type_function(
        fn_ty,
        b.ty_void,
        &[
            b.ty_ptr_cross_float,
            b.ty_ptr_cross_float,
            b.ty_ptr_cross_float,
            b.ty_ptr_cross_float,
        ],
    );

    m.emit_entry_point(EXECUTION_MODEL_KERNEL, main_fn, "main", &[b.var_gid]);
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    // Constants
    let c_batch_heads = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_batch_heads, batch_heads);
    let c_seq_q = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_seq_q, seq_q);
    let c_seq_kv = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_seq_kv, seq_kv);
    let c_head_dim = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_head_dim, head_dim);
    let c_scale = m.alloc_id();
    m.emit_constant_f32(b.ty_float, c_scale, scale);
    let c_neg_inf = m.alloc_id();
    m.emit_constant_f32(b.ty_float, c_neg_inf, f32::NEG_INFINITY);
    // Stride: seq_kv * head_dim
    let c_skv_hd = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_skv_hd, seq_kv * head_dim);

    // Labels
    let label_entry = m.alloc_id();
    let label_body = m.alloc_id();
    let label_merge = m.alloc_id();

    m.emit_function(b.ty_void, main_fn, FUNCTION_CONTROL_NONE, fn_ty);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_q);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_k);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_v);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_o);
    m.emit_label(label_entry);

    let gid = load_gid_x(&mut m, &b);

    // total = batch_heads * seq_q
    let total = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, total, c_batch_heads, c_seq_q]);

    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, total]);
    m.emit_selection_merge(label_merge);
    m.emit_branch_conditional(cond, label_body, label_merge);

    m.emit_label(label_body);

    // bh = gid / seq_q, sq = gid % seq_q
    let bh = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, bh, gid, c_seq_q]);
    let sq = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, sq, gid, c_seq_q]);

    // q_base = gid * head_dim
    let q_base = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, q_base, gid, c_head_dim]);
    // kv_base = bh * seq_kv * head_dim
    let kv_base = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, kv_base, bh, c_skv_hd]);

    // var_max_score
    let var_max = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_max, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_max, c_neg_inf);

    // ── Pass 1: find max score ──
    emit_score_pass(
        &mut m, &b, causal, sq, c_seq_kv, c_head_dim, c_scale, q_base, kv_base, p_q, p_k, var_max,
        true, None, None, p_v, p_o,
    );

    let final_max = m.alloc_id();
    m.emit_load(b.ty_float, final_max, var_max);

    // ── Pass 2: accumulate exp-weighted V ──
    let var_sum_exp = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_sum_exp, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_sum_exp, b.c_float_0);

    emit_score_pass(
        &mut m,
        &b,
        causal,
        sq,
        c_seq_kv,
        c_head_dim,
        c_scale,
        q_base,
        kv_base,
        p_q,
        p_k,
        var_sum_exp,
        false,
        Some(final_max),
        Some(p_o),
        p_v,
        p_o,
    );

    // Normalize: O[o_base+d] /= sum_exp if sum_exp > 0
    let sum_final = m.alloc_id();
    m.emit_load(b.ty_float, sum_final, var_sum_exp);

    let sum_gt_zero = m.alloc_id();
    m.emit(
        OP_F_ORD_GT,
        &[b.ty_bool, sum_gt_zero, sum_final, b.c_float_0],
    );

    let lbl_norm = m.alloc_id();
    let lbl_norm_merge = m.alloc_id();
    m.emit_selection_merge(lbl_norm_merge);
    m.emit_branch_conditional(sum_gt_zero, lbl_norm, lbl_norm_merge);

    m.emit_label(lbl_norm);

    // Normalize loop
    let var_d4 = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_d4, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_d4, b.c_uint_0);

    let lbl_d4_hdr = m.alloc_id();
    let lbl_d4_body = m.alloc_id();
    let lbl_d4_cont = m.alloc_id();
    let lbl_d4_merge = m.alloc_id();

    m.emit_branch(lbl_d4_hdr);

    m.emit_label(lbl_d4_hdr);
    let d4_val = m.alloc_id();
    m.emit_load(b.ty_uint, d4_val, var_d4);
    let d4_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, d4_cond, d4_val, c_head_dim]);
    m.emit_loop_merge(lbl_d4_merge, lbl_d4_cont);
    m.emit_branch_conditional(d4_cond, lbl_d4_body, lbl_d4_merge);

    m.emit_label(lbl_d4_body);
    let o4_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, o4_idx, q_base, d4_val]);
    let o4_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, o4_ptr, p_o, o4_idx);
    let o4_val = m.alloc_id();
    m.emit_load(b.ty_float, o4_val, o4_ptr);
    let o4_normed = m.alloc_id();
    m.emit(OP_F_DIV, &[b.ty_float, o4_normed, o4_val, sum_final]);
    m.emit_store(o4_ptr, o4_normed);

    m.emit_branch(lbl_d4_cont);
    m.emit_label(lbl_d4_cont);
    let d4_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, d4_inc, d4_val, b.c_uint_1]);
    m.emit_store(var_d4, d4_inc);
    m.emit_branch(lbl_d4_hdr);

    m.emit_label(lbl_d4_merge);

    m.emit_branch(lbl_norm_merge);
    m.emit_label(lbl_norm_merge);

    m.emit_branch(label_merge);

    m.emit_label(label_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

/// Emit a score-computation pass (used for both max-finding and accumulation).
///
/// When `is_max_pass` is true, updates `accum_var` with fmax(accum, score).
/// When false, uses `max_val` to compute `exp(score - max)`, adds to `accum_var`,
/// and accumulates weighted V into `o_buf`.
#[allow(clippy::too_many_arguments)]
fn emit_score_pass(
    m: &mut SpvModule,
    b: &super::spirv::BaseIds,
    causal: bool,
    sq: u32,
    c_seq_kv: u32,
    c_head_dim: u32,
    c_scale: u32,
    q_base: u32,
    kv_base: u32,
    p_q: u32,
    p_k: u32,
    accum_var: u32,
    is_max_pass: bool,
    max_val: Option<u32>,
    o_buf: Option<u32>,
    p_v: u32,
    _p_o_unused: u32,
) {
    let var_sk = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_sk, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_sk, b.c_uint_0);

    let lbl_hdr = m.alloc_id();
    let lbl_body = m.alloc_id();
    let lbl_cont = m.alloc_id();
    let lbl_merge = m.alloc_id();

    m.emit_branch(lbl_hdr);

    m.emit_label(lbl_hdr);
    let sk_val = m.alloc_id();
    m.emit_load(b.ty_uint, sk_val, var_sk);
    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, sk_val, c_seq_kv]);
    m.emit_loop_merge(lbl_merge, lbl_cont);
    m.emit_branch_conditional(cond, lbl_body, lbl_merge);

    m.emit_label(lbl_body);

    let lbl_compute = m.alloc_id();
    let lbl_skip = m.alloc_id();
    if causal {
        let sk_gt_sq = m.alloc_id();
        m.emit(OP_U_LESS_THAN, &[b.ty_bool, sk_gt_sq, sq, sk_val]);
        m.emit_selection_merge(lbl_skip);
        m.emit_branch_conditional(sk_gt_sq, lbl_skip, lbl_compute);
    } else {
        m.emit_branch(lbl_compute);
    }

    m.emit_label(lbl_compute);

    // k_off = kv_base + sk * head_dim
    let sk_hd = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, sk_hd, sk_val, c_head_dim]);
    let k_off = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, k_off, kv_base, sk_hd]);

    // Inner dot product loop
    let var_d = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_d, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_d, b.c_uint_0);
    let var_dot = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_dot, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_dot, b.c_float_0);

    let lbl_d_hdr = m.alloc_id();
    let lbl_d_body = m.alloc_id();
    let lbl_d_cont = m.alloc_id();
    let lbl_d_merge = m.alloc_id();

    m.emit_branch(lbl_d_hdr);

    m.emit_label(lbl_d_hdr);
    let d_val = m.alloc_id();
    m.emit_load(b.ty_uint, d_val, var_d);
    let d_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, d_cond, d_val, c_head_dim]);
    m.emit_loop_merge(lbl_d_merge, lbl_d_cont);
    m.emit_branch_conditional(d_cond, lbl_d_body, lbl_d_merge);

    m.emit_label(lbl_d_body);
    let q_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, q_idx, q_base, d_val]);
    let q_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, q_ptr, p_q, q_idx);
    let q_val = m.alloc_id();
    m.emit_load(b.ty_float, q_val, q_ptr);
    let k_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, k_idx, k_off, d_val]);
    let k_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, k_ptr, p_k, k_idx);
    let k_val = m.alloc_id();
    m.emit_load(b.ty_float, k_val, k_ptr);

    let qk_prod = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, qk_prod, q_val, k_val]);
    let old_dot = m.alloc_id();
    m.emit_load(b.ty_float, old_dot, var_dot);
    let new_dot = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, new_dot, old_dot, qk_prod]);
    m.emit_store(var_dot, new_dot);

    m.emit_branch(lbl_d_cont);
    m.emit_label(lbl_d_cont);
    let d_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, d_inc, d_val, b.c_uint_1]);
    m.emit_store(var_d, d_inc);
    m.emit_branch(lbl_d_hdr);

    m.emit_label(lbl_d_merge);

    // score = dot * scale
    let dot_final = m.alloc_id();
    m.emit_load(b.ty_float, dot_final, var_dot);
    let score = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, score, dot_final, c_scale]);

    if is_max_pass {
        // accum = fmax(accum, score)
        let old_acc = m.alloc_id();
        m.emit_load(b.ty_float, old_acc, accum_var);
        let new_acc = m.alloc_id();
        m.emit_opencl_ext(
            b.opencl_ext,
            b.ty_float,
            new_acc,
            OPENCL_FMAX,
            &[old_acc, score],
        );
        m.emit_store(accum_var, new_acc);
    } else {
        // w = exp(score - max_score)
        let max_id = max_val.unwrap_or(b.c_float_0);
        let score_shifted = m.alloc_id();
        m.emit(OP_F_SUB, &[b.ty_float, score_shifted, score, max_id]);
        let w = m.alloc_id();
        m.emit_opencl_ext(b.opencl_ext, b.ty_float, w, OPENCL_EXP, &[score_shifted]);

        // sum_exp += w
        let old_sum = m.alloc_id();
        m.emit_load(b.ty_float, old_sum, accum_var);
        let new_sum = m.alloc_id();
        m.emit(OP_F_ADD, &[b.ty_float, new_sum, old_sum, w]);
        m.emit_store(accum_var, new_sum);

        // Accumulate weighted V
        if let Some(o_buf_id) = o_buf {
            let v_off = m.alloc_id();
            m.emit(OP_I_ADD, &[b.ty_uint, v_off, kv_base, sk_hd]);

            let var_d3 = m.alloc_id();
            m.emit_variable(b.ty_ptr_func_uint, var_d3, STORAGE_CLASS_FUNCTION);
            m.emit_store(var_d3, b.c_uint_0);

            let lbl_d3_hdr = m.alloc_id();
            let lbl_d3_body = m.alloc_id();
            let lbl_d3_cont = m.alloc_id();
            let lbl_d3_merge = m.alloc_id();

            m.emit_branch(lbl_d3_hdr);

            m.emit_label(lbl_d3_hdr);
            let d3_val = m.alloc_id();
            m.emit_load(b.ty_uint, d3_val, var_d3);
            let d3_cond = m.alloc_id();
            m.emit(OP_U_LESS_THAN, &[b.ty_bool, d3_cond, d3_val, c_head_dim]);
            m.emit_loop_merge(lbl_d3_merge, lbl_d3_cont);
            m.emit_branch_conditional(d3_cond, lbl_d3_body, lbl_d3_merge);

            m.emit_label(lbl_d3_body);
            let v_idx = m.alloc_id();
            m.emit(OP_I_ADD, &[b.ty_uint, v_idx, v_off, d3_val]);
            let v_ptr = m.alloc_id();
            m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, v_ptr, p_v, v_idx);
            let v_val = m.alloc_id();
            m.emit_load(b.ty_float, v_val, v_ptr);
            let wv = m.alloc_id();
            m.emit(OP_F_MUL, &[b.ty_float, wv, w, v_val]);

            let o_idx = m.alloc_id();
            m.emit(OP_I_ADD, &[b.ty_uint, o_idx, q_base, d3_val]);
            let o_ptr = m.alloc_id();
            m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, o_ptr, o_buf_id, o_idx);
            let o_old = m.alloc_id();
            m.emit_load(b.ty_float, o_old, o_ptr);
            let o_new = m.alloc_id();
            m.emit(OP_F_ADD, &[b.ty_float, o_new, o_old, wv]);
            m.emit_store(o_ptr, o_new);

            m.emit_branch(lbl_d3_cont);
            m.emit_label(lbl_d3_cont);
            let d3_inc = m.alloc_id();
            m.emit(OP_I_ADD, &[b.ty_uint, d3_inc, d3_val, b.c_uint_1]);
            m.emit_store(var_d3, d3_inc);
            m.emit_branch(lbl_d3_hdr);

            m.emit_label(lbl_d3_merge);
        }
    }

    m.emit_branch(lbl_skip);
    m.emit_label(lbl_skip);
    m.emit_branch(lbl_cont);

    m.emit_label(lbl_cont);
    let sk_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, sk_inc, sk_val, b.c_uint_1]);
    m.emit_store(var_sk, sk_inc);
    m.emit_branch(lbl_hdr);

    m.emit_label(lbl_merge);
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv::SPIRV_MAGIC;

    fn check_valid_spirv(words: &[u32]) {
        assert!(words.len() >= 5, "too short for SPIR-V header");
        assert_eq!(words[0], SPIRV_MAGIC, "bad magic");
        assert!(words[3] > 0, "ID bound must be > 0");
        assert_eq!(words[4], 0, "schema must be 0");
    }

    #[test]
    fn conv2d_spirv_valid() {
        let words = conv2d_spirv(1, 3, 8, 8, 16, 3, 3, 6, 6, 1, 1, 0, 0);
        check_valid_spirv(&words);
    }

    #[test]
    fn conv2d_spirv_with_padding() {
        let words = conv2d_spirv(2, 1, 5, 5, 4, 3, 3, 5, 5, 1, 1, 1, 1);
        check_valid_spirv(&words);
    }

    #[test]
    fn conv2d_spirv_1x1() {
        let words = conv2d_spirv(1, 3, 4, 4, 8, 1, 1, 4, 4, 1, 1, 0, 0);
        check_valid_spirv(&words);
    }

    #[test]
    fn attention_spirv_valid() {
        let words = attention_spirv(2, 4, 4, 8, 0.125, false);
        check_valid_spirv(&words);
    }

    #[test]
    fn attention_spirv_causal() {
        let words = attention_spirv(1, 8, 8, 16, 0.25, true);
        check_valid_spirv(&words);
    }

    #[test]
    fn attention_spirv_magic_number() {
        let words = attention_spirv(1, 4, 4, 8, 0.125, false);
        assert_eq!(words[0], 0x07230203);
    }

    #[test]
    fn conv2d_spirv_magic_number() {
        let words = conv2d_spirv(1, 1, 4, 4, 1, 1, 1, 4, 4, 1, 1, 0, 0);
        assert_eq!(words[0], 0x07230203);
    }
}
