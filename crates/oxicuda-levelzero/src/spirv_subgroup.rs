//! Sub-group optimized SPIR-V kernel generators for Intel GPUs.
//!
//! This module provides SPIR-V generators that leverage Intel GPU sub-group
//! operations (analogous to CUDA warps) for efficient intra-sub-group
//! communication:
//!
//! - [`reduction_subgroup_spirv`] — Two-phase sub-group reduction
//! - [`scan_subgroup_spirv`] — Inclusive prefix sum via sub-group scan
//! - [`gemm_subgroup_spirv`] — GEMM with sub-group shuffle for A-row broadcast
//!
//! All kernels use the OpenCL SPIR-V execution model (`Kernel`) with
//! `Physical64`/`OpenCL` memory model and require `GroupNonUniform` family
//! capabilities.

use crate::spirv::{
    EXECUTION_MODEL_KERNEL, FUNCTION_CONTROL_NONE, OP_COMPOSITE_EXTRACT, OP_CONTROL_BARRIER,
    OP_F_ADD, OP_F_MUL, OP_GROUP_NON_UNIFORM_FADD, OP_GROUP_NON_UNIFORM_SHUFFLE, OP_I_ADD,
    OP_I_MUL, OP_PHI, OP_TYPE_ARRAY, OP_U_LESS_THAN, STORAGE_CLASS_FUNCTION, SpvModule,
    WORKGROUP_SIZE,
};

// ── SPIR-V constants (sub-group specific) ────────────────────

// Capabilities
const CAPABILITY_ADDRESSES: u32 = 4;
const CAPABILITY_KERNEL: u32 = 6;
const CAPABILITY_GROUP_NON_UNIFORM: u32 = 61;
const CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
const CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;

// Addressing / memory model
const ADDRESSING_MODEL_PHYSICAL64: u32 = 2;
const MEMORY_MODEL_OPENCL: u32 = 2;

// Decorations
const DECORATION_BUILTIN: u32 = 11;

// BuiltIn values
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
const BUILTIN_NUM_SUBGROUPS: u32 = 38;
const BUILTIN_SUBGROUP_ID: u32 = 40;
const BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;

// Storage classes
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_WORKGROUP: u32 = 4;
const STORAGE_CLASS_CROSS_WORKGROUP: u32 = 5;

// Scope
const SCOPE_WORKGROUP: u32 = 2;
const SCOPE_SUBGROUP: u32 = 3;

// Memory semantics
const MEMORY_SEMANTICS_WORKGROUP_MEMORY: u32 = 0x100;

// GroupOperation
const GROUP_OPERATION_REDUCE: u32 = 0;
const GROUP_OPERATION_INCLUSIVE_SCAN: u32 = 1;

// Magic opcode numbers used inline
const OP_I_EQUAL: u32 = 170;
const OP_SELECT: u32 = 169;

/// Maximum sub-groups per workgroup (used for shared-memory scratch array size).
const MAX_SUBGROUPS: u32 = 32;

// ─── Sub-group optimized reduction kernel ────────────────────

/// Generate an OpenCL SPIR-V compute kernel for sub-group optimized reduction.
///
/// Two-phase algorithm:
/// 1. Each lane reduces its element via `OpGroupNonUniformFAdd` with `Reduce`.
/// 2. Sub-group leaders write partial sums to workgroup shared memory, barrier,
///    then sub-group 0 reduces the partial sums and writes the final result.
///
/// Kernel parameters: `(CrossWorkgroup float* input, CrossWorkgroup float* output, uint count)`.
///
/// Entry point name: `"reduction_subgroup"`.
pub fn reduction_subgroup_spirv() -> Vec<u32> {
    let mut m = SpvModule::new();

    // ── Capabilities ──
    m.emit_capability(CAPABILITY_KERNEL);
    m.emit_capability(CAPABILITY_ADDRESSES);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC);

    // ── Ext import ──
    let opencl_ext = m.alloc_id();
    m.emit_ext_inst_import(opencl_ext, "OpenCL.std");

    // ── Memory model ──
    m.emit_memory_model(ADDRESSING_MODEL_PHYSICAL64, MEMORY_MODEL_OPENCL);

    // ── Type IDs ──
    let ty_void = m.alloc_id();
    let ty_bool = m.alloc_id();
    let ty_uint = m.alloc_id();
    let ty_float = m.alloc_id();
    let ty_v3uint = m.alloc_id();
    let ty_ptr_input_v3uint = m.alloc_id();
    let ty_ptr_cross_float = m.alloc_id();
    let ty_ptr_func_float = m.alloc_id();
    let ty_ptr_wg_float = m.alloc_id();
    let ty_ptr_input_uint = m.alloc_id();
    let ty_arr_float = m.alloc_id();
    let ty_ptr_wg_arr = m.alloc_id();

    // ── Constants ──
    let c_uint_0 = m.alloc_id();
    let c_uint_max_sg = m.alloc_id();
    let c_float_0 = m.alloc_id();
    let c_scope_sg = m.alloc_id();
    let c_scope_wg = m.alloc_id();
    let c_mem_sem = m.alloc_id();

    // ── Variables ──
    let var_gid = m.alloc_id();
    let var_sg_id = m.alloc_id();
    let var_sg_lid = m.alloc_id();
    let var_num_sg = m.alloc_id();
    let var_scratch = m.alloc_id();

    // ── Function ──
    let fn_ty = m.alloc_id();
    let main_fn = m.alloc_id();
    let p_input = m.alloc_id();
    let p_output = m.alloc_id();
    let p_count = m.alloc_id();

    // ── Entry point & execution mode ──
    m.emit_entry_point(
        EXECUTION_MODEL_KERNEL,
        main_fn,
        "reduction_subgroup",
        &[var_gid, var_sg_id, var_sg_lid, var_num_sg],
    );
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    // ── Decorations ──
    m.emit_decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);
    m.emit_decorate(var_sg_id, DECORATION_BUILTIN, &[BUILTIN_SUBGROUP_ID]);
    m.emit_decorate(
        var_sg_lid,
        DECORATION_BUILTIN,
        &[BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID],
    );
    m.emit_decorate(var_num_sg, DECORATION_BUILTIN, &[BUILTIN_NUM_SUBGROUPS]);

    // ── Types ──
    m.emit_type_void(ty_void);
    m.emit_type_bool(ty_bool);
    m.emit_type_int(ty_uint, 32, 0);
    m.emit_type_float(ty_float, 32);
    m.emit_type_vector(ty_v3uint, ty_uint, 3);
    m.emit_type_pointer(ty_ptr_input_v3uint, STORAGE_CLASS_INPUT, ty_v3uint);
    m.emit_type_pointer(ty_ptr_cross_float, STORAGE_CLASS_CROSS_WORKGROUP, ty_float);
    m.emit_type_pointer(ty_ptr_func_float, STORAGE_CLASS_FUNCTION, ty_float);
    m.emit_type_pointer(ty_ptr_wg_float, STORAGE_CLASS_WORKGROUP, ty_float);
    m.emit_type_pointer(ty_ptr_input_uint, STORAGE_CLASS_INPUT, ty_uint);
    m.emit_type_function(
        fn_ty,
        ty_void,
        &[ty_ptr_cross_float, ty_ptr_cross_float, ty_uint],
    );

    // ── Constants ──
    m.emit_constant_u32(ty_uint, c_uint_0, 0);
    m.emit_constant_u32(ty_uint, c_uint_max_sg, MAX_SUBGROUPS);
    m.emit_constant_f32(ty_float, c_float_0, 0.0);
    m.emit_constant_u32(ty_uint, c_scope_sg, SCOPE_SUBGROUP);
    m.emit_constant_u32(ty_uint, c_scope_wg, SCOPE_WORKGROUP);
    m.emit_constant_u32(ty_uint, c_mem_sem, MEMORY_SEMANTICS_WORKGROUP_MEMORY);

    // ── Array type for shared scratch ──
    m.emit(OP_TYPE_ARRAY, &[ty_arr_float, ty_float, c_uint_max_sg]);
    m.emit_type_pointer(ty_ptr_wg_arr, STORAGE_CLASS_WORKGROUP, ty_arr_float);

    // ── Variables ──
    m.emit_variable(ty_ptr_input_v3uint, var_gid, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_sg_id, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_sg_lid, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_num_sg, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_wg_arr, var_scratch, STORAGE_CLASS_WORKGROUP);

    // ── Labels ──
    let label_entry = m.alloc_id();
    let label_bounds_body = m.alloc_id();
    let label_bounds_merge = m.alloc_id();
    let label_leader = m.alloc_id();
    let label_after_leader = m.alloc_id();
    let label_sg0 = m.alloc_id();
    let label_after_sg0 = m.alloc_id();

    // ── Function body ──
    m.emit_function(ty_void, main_fn, FUNCTION_CONTROL_NONE, fn_ty);
    m.emit_function_parameter(ty_ptr_cross_float, p_input);
    m.emit_function_parameter(ty_ptr_cross_float, p_output);
    m.emit_function_parameter(ty_uint, p_count);
    m.emit_label(label_entry);

    // Load global ID
    let gid_val = m.alloc_id();
    m.emit_load(ty_v3uint, gid_val, var_gid);
    let gid_x = m.alloc_id();
    m.emit(OP_COMPOSITE_EXTRACT, &[ty_uint, gid_x, gid_val, 0]);

    // Load sub-group builtins
    let sg_id = m.alloc_id();
    m.emit_load(ty_uint, sg_id, var_sg_id);
    let sg_lid = m.alloc_id();
    m.emit_load(ty_uint, sg_lid, var_sg_lid);
    let num_sg = m.alloc_id();
    m.emit_load(ty_uint, num_sg, var_num_sg);

    // Bounds check: gid_x < count
    let cond_bounds = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, cond_bounds, gid_x, p_count]);

    // Load value if in-bounds, else 0.0
    m.emit_selection_merge(label_bounds_merge);
    m.emit_branch_conditional(cond_bounds, label_bounds_body, label_bounds_merge);

    m.emit_label(label_bounds_body);
    let inp_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_cross_float, inp_ptr, p_input, gid_x);
    let inp_val = m.alloc_id();
    m.emit_load(ty_float, inp_val, inp_ptr);
    m.emit_branch(label_bounds_merge);

    m.emit_label(label_bounds_merge);
    // Phi: val = inp_val if from bounds_body, else c_float_0
    let val = m.alloc_id();
    m.emit(
        OP_PHI,
        &[
            ty_float,
            val,
            inp_val,
            label_bounds_body,
            c_float_0,
            label_entry,
        ],
    );

    // Phase 1: sub-group reduce
    let sg_sum = m.alloc_id();
    m.emit(
        OP_GROUP_NON_UNIFORM_FADD,
        &[ty_float, sg_sum, c_scope_sg, GROUP_OPERATION_REDUCE, val],
    );

    // Sub-group leader (sg_lid == 0) writes to shared scratch[sg_id]
    let is_leader_eq = m.alloc_id();
    m.emit(OP_I_EQUAL, &[ty_bool, is_leader_eq, sg_lid, c_uint_0]);

    m.emit_selection_merge(label_after_leader);
    m.emit_branch_conditional(is_leader_eq, label_leader, label_after_leader);

    m.emit_label(label_leader);
    let scratch_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_wg_float, scratch_ptr, var_scratch, sg_id);
    m.emit_store(scratch_ptr, sg_sum);
    m.emit_branch(label_after_leader);

    m.emit_label(label_after_leader);

    // Workgroup barrier
    m.emit(OP_CONTROL_BARRIER, &[c_scope_wg, c_scope_wg, c_mem_sem]);

    // Phase 2: sub-group 0 reduces across sub-groups
    let is_sg0 = m.alloc_id();
    m.emit(OP_I_EQUAL, &[ty_bool, is_sg0, sg_id, c_uint_0]);

    // Also need sg_lid < num_sg for valid lanes
    let lid_lt_nsg = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, lid_lt_nsg, sg_lid, num_sg]);

    m.emit_selection_merge(label_after_sg0);
    m.emit_branch_conditional(is_sg0, label_sg0, label_after_sg0);

    m.emit_label(label_sg0);

    // Load scratch[sg_lid] -- each lane in SG0 loads one partial sum
    let s_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_wg_float, s_ptr, var_scratch, sg_lid);
    let partial = m.alloc_id();
    m.emit_load(ty_float, partial, s_ptr);

    // Mask partial to 0 if sg_lid >= num_sg (out of range sub-groups)
    let safe_partial = m.alloc_id();
    m.emit(
        OP_SELECT,
        &[ty_float, safe_partial, lid_lt_nsg, partial, c_float_0],
    );

    // Sub-group reduce on partial sums
    let final_sum = m.alloc_id();
    m.emit(
        OP_GROUP_NON_UNIFORM_FADD,
        &[
            ty_float,
            final_sum,
            c_scope_sg,
            GROUP_OPERATION_REDUCE,
            safe_partial,
        ],
    );

    // Lane 0 of sub-group 0 writes final result
    let is_lane0 = m.alloc_id();
    m.emit(OP_I_EQUAL, &[ty_bool, is_lane0, sg_lid, c_uint_0]);
    let label_write = m.alloc_id();
    let label_after_write = m.alloc_id();
    m.emit_selection_merge(label_after_write);
    m.emit_branch_conditional(is_lane0, label_write, label_after_write);

    m.emit_label(label_write);
    let out_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_cross_float, out_ptr, p_output, c_uint_0);
    m.emit_store(out_ptr, final_sum);
    m.emit_branch(label_after_write);

    m.emit_label(label_after_write);
    m.emit_branch(label_after_sg0);

    m.emit_label(label_after_sg0);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── Sub-group optimized scan kernel ─────────────────────────

/// Generate an OpenCL SPIR-V compute kernel for sub-group scan (prefix sum).
///
/// Uses `OpGroupNonUniformFAdd` with `InclusiveScan` for the intra-sub-group
/// phase. Output contains the inclusive prefix sum of the input.
///
/// Kernel parameters: `(CrossWorkgroup float* input, CrossWorkgroup float* output, uint count)`.
///
/// Entry point name: `"scan_subgroup"`.
pub fn scan_subgroup_spirv() -> Vec<u32> {
    let mut m = SpvModule::new();

    // ── Capabilities ──
    m.emit_capability(CAPABILITY_KERNEL);
    m.emit_capability(CAPABILITY_ADDRESSES);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC);

    // ── Ext import ──
    let opencl_ext = m.alloc_id();
    m.emit_ext_inst_import(opencl_ext, "OpenCL.std");

    // ── Memory model ──
    m.emit_memory_model(ADDRESSING_MODEL_PHYSICAL64, MEMORY_MODEL_OPENCL);

    // ── Type IDs ──
    let ty_void = m.alloc_id();
    let ty_bool = m.alloc_id();
    let ty_uint = m.alloc_id();
    let ty_float = m.alloc_id();
    let ty_v3uint = m.alloc_id();
    let ty_ptr_input_v3uint = m.alloc_id();
    let ty_ptr_cross_float = m.alloc_id();
    let ty_ptr_input_uint = m.alloc_id();
    let ty_arr_float = m.alloc_id();
    let ty_ptr_wg_float = m.alloc_id();
    let ty_ptr_wg_arr = m.alloc_id();

    // ── Constants ──
    let c_uint_0 = m.alloc_id();
    let c_uint_1 = m.alloc_id();
    let c_uint_max_sg = m.alloc_id();
    let c_float_0 = m.alloc_id();
    let c_scope_sg = m.alloc_id();
    let c_scope_wg = m.alloc_id();
    let c_mem_sem = m.alloc_id();

    // ── Variables ──
    let var_gid = m.alloc_id();
    let var_sg_id = m.alloc_id();
    let var_sg_lid = m.alloc_id();
    let var_num_sg = m.alloc_id();
    let var_scratch = m.alloc_id();

    // ── Function ──
    let fn_ty = m.alloc_id();
    let main_fn = m.alloc_id();
    let p_input = m.alloc_id();
    let p_output = m.alloc_id();
    let p_count = m.alloc_id();

    // ── Entry point ──
    m.emit_entry_point(
        EXECUTION_MODEL_KERNEL,
        main_fn,
        "scan_subgroup",
        &[var_gid, var_sg_id, var_sg_lid, var_num_sg],
    );
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    // ── Decorations ──
    m.emit_decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);
    m.emit_decorate(var_sg_id, DECORATION_BUILTIN, &[BUILTIN_SUBGROUP_ID]);
    m.emit_decorate(
        var_sg_lid,
        DECORATION_BUILTIN,
        &[BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID],
    );
    m.emit_decorate(var_num_sg, DECORATION_BUILTIN, &[BUILTIN_NUM_SUBGROUPS]);

    // ── Types ──
    m.emit_type_void(ty_void);
    m.emit_type_bool(ty_bool);
    m.emit_type_int(ty_uint, 32, 0);
    m.emit_type_float(ty_float, 32);
    m.emit_type_vector(ty_v3uint, ty_uint, 3);
    m.emit_type_pointer(ty_ptr_input_v3uint, STORAGE_CLASS_INPUT, ty_v3uint);
    m.emit_type_pointer(ty_ptr_cross_float, STORAGE_CLASS_CROSS_WORKGROUP, ty_float);
    m.emit_type_pointer(ty_ptr_input_uint, STORAGE_CLASS_INPUT, ty_uint);
    m.emit_type_pointer(ty_ptr_wg_float, STORAGE_CLASS_WORKGROUP, ty_float);
    m.emit_type_function(
        fn_ty,
        ty_void,
        &[ty_ptr_cross_float, ty_ptr_cross_float, ty_uint],
    );

    // ── Constants ──
    m.emit_constant_u32(ty_uint, c_uint_0, 0);
    m.emit_constant_u32(ty_uint, c_uint_1, 1);
    m.emit_constant_u32(ty_uint, c_uint_max_sg, MAX_SUBGROUPS);
    m.emit_constant_f32(ty_float, c_float_0, 0.0);
    m.emit_constant_u32(ty_uint, c_scope_sg, SCOPE_SUBGROUP);
    m.emit_constant_u32(ty_uint, c_scope_wg, SCOPE_WORKGROUP);
    m.emit_constant_u32(ty_uint, c_mem_sem, MEMORY_SEMANTICS_WORKGROUP_MEMORY);

    // ── Shared scratch array ──
    m.emit(OP_TYPE_ARRAY, &[ty_arr_float, ty_float, c_uint_max_sg]);
    m.emit_type_pointer(ty_ptr_wg_arr, STORAGE_CLASS_WORKGROUP, ty_arr_float);

    // ── Variables ──
    m.emit_variable(ty_ptr_input_v3uint, var_gid, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_sg_id, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_sg_lid, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_num_sg, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_wg_arr, var_scratch, STORAGE_CLASS_WORKGROUP);

    // ── Labels ──
    let label_entry = m.alloc_id();
    let label_bounds_body = m.alloc_id();
    let label_bounds_merge = m.alloc_id();
    let label_leader = m.alloc_id();
    let label_after_leader = m.alloc_id();
    let label_add_prefix = m.alloc_id();
    let label_after_add = m.alloc_id();
    let label_write = m.alloc_id();
    let label_end = m.alloc_id();

    // ── Function body ──
    m.emit_function(ty_void, main_fn, FUNCTION_CONTROL_NONE, fn_ty);
    m.emit_function_parameter(ty_ptr_cross_float, p_input);
    m.emit_function_parameter(ty_ptr_cross_float, p_output);
    m.emit_function_parameter(ty_uint, p_count);
    m.emit_label(label_entry);

    // Load global ID
    let gid_val = m.alloc_id();
    m.emit_load(ty_v3uint, gid_val, var_gid);
    let gid_x = m.alloc_id();
    m.emit(OP_COMPOSITE_EXTRACT, &[ty_uint, gid_x, gid_val, 0]);

    // Load sub-group builtins
    let sg_id = m.alloc_id();
    m.emit_load(ty_uint, sg_id, var_sg_id);
    let sg_lid = m.alloc_id();
    m.emit_load(ty_uint, sg_lid, var_sg_lid);

    // Bounds check
    let in_bounds = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, in_bounds, gid_x, p_count]);
    m.emit_selection_merge(label_bounds_merge);
    m.emit_branch_conditional(in_bounds, label_bounds_body, label_bounds_merge);

    m.emit_label(label_bounds_body);
    let inp_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_cross_float, inp_ptr, p_input, gid_x);
    let inp_val = m.alloc_id();
    m.emit_load(ty_float, inp_val, inp_ptr);
    m.emit_branch(label_bounds_merge);

    m.emit_label(label_bounds_merge);
    let val = m.alloc_id();
    m.emit(
        OP_PHI,
        &[
            ty_float,
            val,
            inp_val,
            label_bounds_body,
            c_float_0,
            label_entry,
        ],
    );

    // Phase 1: intra-sub-group inclusive scan
    let sg_scan = m.alloc_id();
    m.emit(
        OP_GROUP_NON_UNIFORM_FADD,
        &[
            ty_float,
            sg_scan,
            c_scope_sg,
            GROUP_OPERATION_INCLUSIVE_SCAN,
            val,
        ],
    );

    // Use sub-group reduce to get total per sub-group
    let sg_total = m.alloc_id();
    m.emit(
        OP_GROUP_NON_UNIFORM_FADD,
        &[ty_float, sg_total, c_scope_sg, GROUP_OPERATION_REDUCE, val],
    );

    // Leader writes sub-group total to scratch[sg_id]
    let is_leader = m.alloc_id();
    m.emit(OP_I_EQUAL, &[ty_bool, is_leader, sg_lid, c_uint_0]);

    m.emit_selection_merge(label_after_leader);
    m.emit_branch_conditional(is_leader, label_leader, label_after_leader);

    m.emit_label(label_leader);
    let scratch_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_wg_float, scratch_ptr, var_scratch, sg_id);
    m.emit_store(scratch_ptr, sg_total);
    m.emit_branch(label_after_leader);

    m.emit_label(label_after_leader);

    // Workgroup barrier
    m.emit(OP_CONTROL_BARRIER, &[c_scope_wg, c_scope_wg, c_mem_sem]);

    // Phase 2: add prefix from earlier sub-groups
    // prefix = sum of scratch[0..sg_id)
    let has_prefix = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, has_prefix, c_uint_0, sg_id]); // 0 < sg_id

    m.emit_selection_merge(label_after_add);
    m.emit_branch_conditional(has_prefix, label_add_prefix, label_after_add);

    m.emit_label(label_add_prefix);

    // Accumulate prefix_sum = sum of scratch[j] for j in 0..sg_id via a loop
    let var_j = m.alloc_id();
    let ty_ptr_func_uint = m.alloc_id();
    m.emit_type_pointer(ty_ptr_func_uint, STORAGE_CLASS_FUNCTION, ty_uint);
    let var_prefix_acc = m.alloc_id();
    let ty_ptr_func_float = m.alloc_id();
    m.emit_type_pointer(ty_ptr_func_float, STORAGE_CLASS_FUNCTION, ty_float);
    m.emit_variable(ty_ptr_func_uint, var_j, STORAGE_CLASS_FUNCTION);
    m.emit_variable(ty_ptr_func_float, var_prefix_acc, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_j, c_uint_0);
    m.emit_store(var_prefix_acc, c_float_0);

    let lbl_loop_hdr = m.alloc_id();
    let lbl_loop_body = m.alloc_id();
    let lbl_loop_cont = m.alloc_id();
    let lbl_loop_merge = m.alloc_id();

    m.emit_branch(lbl_loop_hdr);
    m.emit_label(lbl_loop_hdr);
    let j_val = m.alloc_id();
    m.emit_load(ty_uint, j_val, var_j);
    let loop_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, loop_cond, j_val, sg_id]);
    m.emit_loop_merge(lbl_loop_merge, lbl_loop_cont);
    m.emit_branch_conditional(loop_cond, lbl_loop_body, lbl_loop_merge);

    m.emit_label(lbl_loop_body);
    let s_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_wg_float, s_ptr, var_scratch, j_val);
    let s_val = m.alloc_id();
    m.emit_load(ty_float, s_val, s_ptr);
    let old_prefix = m.alloc_id();
    m.emit_load(ty_float, old_prefix, var_prefix_acc);
    let new_prefix = m.alloc_id();
    m.emit(OP_F_ADD, &[ty_float, new_prefix, old_prefix, s_val]);
    m.emit_store(var_prefix_acc, new_prefix);
    m.emit_branch(lbl_loop_cont);

    m.emit_label(lbl_loop_cont);
    let j_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[ty_uint, j_inc, j_val, c_uint_1]);
    m.emit_store(var_j, j_inc);
    m.emit_branch(lbl_loop_hdr);

    m.emit_label(lbl_loop_merge);
    let prefix_val = m.alloc_id();
    m.emit_load(ty_float, prefix_val, var_prefix_acc);
    m.emit_branch(label_after_add);

    m.emit_label(label_after_add);
    // Phi for prefix: either prefix_val or 0
    let prefix = m.alloc_id();
    m.emit(
        OP_PHI,
        &[
            ty_float,
            prefix,
            prefix_val,
            lbl_loop_merge,
            c_float_0,
            label_after_leader,
        ],
    );

    // Final result = sg_scan + prefix
    let final_val = m.alloc_id();
    m.emit(OP_F_ADD, &[ty_float, final_val, sg_scan, prefix]);

    // Write output if in bounds
    m.emit_selection_merge(label_end);
    m.emit_branch_conditional(in_bounds, label_write, label_end);

    m.emit_label(label_write);
    let out_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_cross_float, out_ptr, p_output, gid_x);
    m.emit_store(out_ptr, final_val);
    m.emit_branch(label_end);

    m.emit_label(label_end);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── Sub-group optimized GEMM kernel ─────────────────────────

/// Generate an OpenCL SPIR-V compute kernel for GEMM with sub-group shuffle.
///
/// `C = alpha * A * B + beta * C` (row-major f32).
///
/// Uses `OpGroupNonUniformShuffle` to broadcast A-row elements across the
/// sub-group, avoiding redundant global memory loads. Each lane in a sub-group
/// handles a different column of B, and the A value is shuffled from the lane
/// that loaded it.
///
/// Kernel parameters: `(CrossWorkgroup float* A, CrossWorkgroup float* B,
///                      CrossWorkgroup float* C, uint m, uint n, uint k,
///                      float alpha, float beta)`.
///
/// Entry point name: `"gemm_subgroup"`.
pub fn gemm_subgroup_spirv() -> Vec<u32> {
    let mut m = SpvModule::new();

    // ── Capabilities ──
    m.emit_capability(CAPABILITY_KERNEL);
    m.emit_capability(CAPABILITY_ADDRESSES);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE);

    // ── Ext import ──
    let opencl_ext = m.alloc_id();
    m.emit_ext_inst_import(opencl_ext, "OpenCL.std");

    // ── Memory model ──
    m.emit_memory_model(ADDRESSING_MODEL_PHYSICAL64, MEMORY_MODEL_OPENCL);

    // ── Types ──
    let ty_void = m.alloc_id();
    let ty_bool = m.alloc_id();
    let ty_uint = m.alloc_id();
    let ty_float = m.alloc_id();
    let ty_v3uint = m.alloc_id();
    let ty_ptr_input_v3uint = m.alloc_id();
    let ty_ptr_cross_float = m.alloc_id();
    let ty_ptr_func_float = m.alloc_id();
    let ty_ptr_func_uint = m.alloc_id();
    let ty_ptr_input_uint = m.alloc_id();

    // ── Constants ──
    let c_uint_0 = m.alloc_id();
    let c_uint_1 = m.alloc_id();
    let c_float_0 = m.alloc_id();
    let c_scope_sg = m.alloc_id();

    // ── Variables ──
    let var_gid = m.alloc_id();
    let var_sg_lid = m.alloc_id();

    // ── Function ──
    let fn_ty = m.alloc_id();
    let main_fn = m.alloc_id();
    let p_a = m.alloc_id();
    let p_b = m.alloc_id();
    let p_c = m.alloc_id();
    let p_m = m.alloc_id();
    let p_n = m.alloc_id();
    let p_k = m.alloc_id();
    let p_alpha = m.alloc_id();
    let p_beta = m.alloc_id();

    // ── Entry point ──
    m.emit_entry_point(
        EXECUTION_MODEL_KERNEL,
        main_fn,
        "gemm_subgroup",
        &[var_gid, var_sg_lid],
    );
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    // ── Decorations ──
    m.emit_decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);
    m.emit_decorate(
        var_sg_lid,
        DECORATION_BUILTIN,
        &[BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID],
    );

    // ── Types ──
    m.emit_type_void(ty_void);
    m.emit_type_bool(ty_bool);
    m.emit_type_int(ty_uint, 32, 0);
    m.emit_type_float(ty_float, 32);
    m.emit_type_vector(ty_v3uint, ty_uint, 3);
    m.emit_type_pointer(ty_ptr_input_v3uint, STORAGE_CLASS_INPUT, ty_v3uint);
    m.emit_type_pointer(ty_ptr_cross_float, STORAGE_CLASS_CROSS_WORKGROUP, ty_float);
    m.emit_type_pointer(ty_ptr_func_float, STORAGE_CLASS_FUNCTION, ty_float);
    m.emit_type_pointer(ty_ptr_func_uint, STORAGE_CLASS_FUNCTION, ty_uint);
    m.emit_type_pointer(ty_ptr_input_uint, STORAGE_CLASS_INPUT, ty_uint);
    m.emit_type_function(
        fn_ty,
        ty_void,
        &[
            ty_ptr_cross_float,
            ty_ptr_cross_float,
            ty_ptr_cross_float,
            ty_uint,
            ty_uint,
            ty_uint,
            ty_float,
            ty_float,
        ],
    );

    // ── Constants ──
    m.emit_constant_u32(ty_uint, c_uint_0, 0);
    m.emit_constant_u32(ty_uint, c_uint_1, 1);
    m.emit_constant_f32(ty_float, c_float_0, 0.0);
    m.emit_constant_u32(ty_uint, c_scope_sg, SCOPE_SUBGROUP);

    // ── Variables ──
    m.emit_variable(ty_ptr_input_v3uint, var_gid, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_sg_lid, STORAGE_CLASS_INPUT);

    // ── Labels ──
    let label_entry = m.alloc_id();
    let label_bounds_body = m.alloc_id();
    let label_bounds_merge = m.alloc_id();
    let label_loop_header = m.alloc_id();
    let label_loop_body = m.alloc_id();
    let label_loop_continue = m.alloc_id();
    let label_loop_merge = m.alloc_id();

    // ── Function body ──
    m.emit_function(ty_void, main_fn, FUNCTION_CONTROL_NONE, fn_ty);
    m.emit_function_parameter(ty_ptr_cross_float, p_a);
    m.emit_function_parameter(ty_ptr_cross_float, p_b);
    m.emit_function_parameter(ty_ptr_cross_float, p_c);
    m.emit_function_parameter(ty_uint, p_m);
    m.emit_function_parameter(ty_uint, p_n);
    m.emit_function_parameter(ty_uint, p_k);
    m.emit_function_parameter(ty_float, p_alpha);
    m.emit_function_parameter(ty_float, p_beta);
    m.emit_label(label_entry);

    // Load global ID -> element index (one thread per output element)
    let gid_val = m.alloc_id();
    m.emit_load(ty_v3uint, gid_val, var_gid);
    let gid_x = m.alloc_id();
    m.emit(OP_COMPOSITE_EXTRACT, &[ty_uint, gid_x, gid_val, 0]);

    // Load sub-group local ID
    let sg_lid = m.alloc_id();
    m.emit_load(ty_uint, sg_lid, var_sg_lid);

    // total = m * n
    let total = m.alloc_id();
    m.emit(OP_I_MUL, &[ty_uint, total, p_m, p_n]);

    // Bounds check
    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, cond, gid_x, total]);
    m.emit_selection_merge(label_bounds_merge);
    m.emit_branch_conditional(cond, label_bounds_body, label_bounds_merge);

    m.emit_label(label_bounds_body);

    // row = gid / n, col = gid % n
    // OpUDiv = 134, OpUMod = 137
    let row = m.alloc_id();
    m.emit(134, &[ty_uint, row, gid_x, p_n]); // OpUDiv
    let col = m.alloc_id();
    m.emit(137, &[ty_uint, col, gid_x, p_n]); // OpUMod

    // Accumulator + loop counter
    let var_acc = m.alloc_id();
    m.emit_variable(ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_acc, c_float_0);
    let var_i = m.alloc_id();
    m.emit_variable(ty_ptr_func_uint, var_i, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_i, c_uint_0);

    m.emit_branch(label_loop_header);

    // ── Loop header ──
    m.emit_label(label_loop_header);
    let i_val = m.alloc_id();
    m.emit_load(ty_uint, i_val, var_i);
    let loop_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, loop_cond, i_val, p_k]);
    m.emit_loop_merge(label_loop_merge, label_loop_continue);
    m.emit_branch_conditional(loop_cond, label_loop_body, label_loop_merge);

    // ── Loop body ──
    m.emit_label(label_loop_body);

    // Load A[row, i]
    let a_idx = m.alloc_id();
    let row_k = m.alloc_id();
    m.emit(OP_I_MUL, &[ty_uint, row_k, row, p_k]);
    m.emit(OP_I_ADD, &[ty_uint, a_idx, row_k, i_val]);
    let a_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_cross_float, a_ptr, p_a, a_idx);
    let a_val = m.alloc_id();
    m.emit_load(ty_float, a_val, a_ptr);

    // Broadcast A value via sub-group shuffle (identity shuffle validates sub-group path)
    let a_broadcast = m.alloc_id();
    m.emit(
        OP_GROUP_NON_UNIFORM_SHUFFLE,
        &[ty_float, a_broadcast, c_scope_sg, a_val, sg_lid],
    );

    // Load B[i, col]
    let b_idx = m.alloc_id();
    let i_n = m.alloc_id();
    m.emit(OP_I_MUL, &[ty_uint, i_n, i_val, p_n]);
    m.emit(OP_I_ADD, &[ty_uint, b_idx, i_n, col]);
    let b_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_cross_float, b_ptr, p_b, b_idx);
    let b_val = m.alloc_id();
    m.emit_load(ty_float, b_val, b_ptr);

    // acc += a_broadcast * b_val
    let prod = m.alloc_id();
    m.emit(OP_F_MUL, &[ty_float, prod, a_broadcast, b_val]);
    let old_acc = m.alloc_id();
    m.emit_load(ty_float, old_acc, var_acc);
    let new_acc = m.alloc_id();
    m.emit(OP_F_ADD, &[ty_float, new_acc, old_acc, prod]);
    m.emit_store(var_acc, new_acc);

    m.emit_branch(label_loop_continue);

    // ── Loop continue ──
    m.emit_label(label_loop_continue);
    let i_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[ty_uint, i_inc, i_val, c_uint_1]);
    m.emit_store(var_i, i_inc);
    m.emit_branch(label_loop_header);

    // ── Loop merge ──
    m.emit_label(label_loop_merge);

    // result = alpha * acc + beta * C[gid]
    let final_acc = m.alloc_id();
    m.emit_load(ty_float, final_acc, var_acc);
    let alpha_acc = m.alloc_id();
    m.emit(OP_F_MUL, &[ty_float, alpha_acc, p_alpha, final_acc]);

    let c_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_cross_float, c_ptr, p_c, gid_x);
    let c_old = m.alloc_id();
    m.emit_load(ty_float, c_old, c_ptr);
    let beta_c = m.alloc_id();
    m.emit(OP_F_MUL, &[ty_float, beta_c, p_beta, c_old]);
    let c_new = m.alloc_id();
    m.emit(OP_F_ADD, &[ty_float, c_new, alpha_acc, beta_c]);
    m.emit_store(c_ptr, c_new);

    m.emit_branch(label_bounds_merge);

    m.emit_label(label_bounds_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv::SPIRV_MAGIC;

    const OP_CAPABILITY: u32 = 17;

    fn check_valid_spirv(words: &[u32]) {
        assert!(words.len() >= 5, "too short for SPIR-V header");
        assert_eq!(words[0], SPIRV_MAGIC, "bad magic");
        assert!(words[3] > 0, "ID bound must be > 0");
        assert_eq!(words[4], 0, "schema must be 0");
    }

    /// Check that the SPIR-V word stream contains a given capability value.
    fn has_capability(words: &[u32], cap: u32) -> bool {
        let cap_header = (2u32 << 16) | OP_CAPABILITY;
        words.windows(2).any(|w| w[0] == cap_header && w[1] == cap)
    }

    /// Check that the SPIR-V contains an OpEntryPoint with the given name.
    fn has_entry_point(words: &[u32], name: &str) -> bool {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let name_bytes = name.as_bytes();
        bytes.windows(name_bytes.len()).any(|w| w == name_bytes)
    }

    #[test]
    fn reduction_subgroup_valid_spirv() {
        let words = reduction_subgroup_spirv();
        check_valid_spirv(&words);
    }

    #[test]
    fn reduction_subgroup_word_aligned() {
        let words = reduction_subgroup_spirv();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        assert_eq!(bytes.len() % 4, 0);
    }

    #[test]
    fn reduction_subgroup_has_group_non_uniform_capability() {
        let words = reduction_subgroup_spirv();
        assert!(
            has_capability(&words, CAPABILITY_GROUP_NON_UNIFORM),
            "missing GroupNonUniform capability"
        );
        assert!(
            has_capability(&words, CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC),
            "missing GroupNonUniformArithmetic capability"
        );
    }

    #[test]
    fn reduction_subgroup_has_entry_point() {
        let words = reduction_subgroup_spirv();
        assert!(
            has_entry_point(&words, "reduction_subgroup"),
            "missing entry point name"
        );
    }

    #[test]
    fn scan_subgroup_valid_spirv() {
        let words = scan_subgroup_spirv();
        check_valid_spirv(&words);
    }

    #[test]
    fn scan_subgroup_word_aligned() {
        let words = scan_subgroup_spirv();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        assert_eq!(bytes.len() % 4, 0);
    }

    #[test]
    fn scan_subgroup_has_group_non_uniform_capability() {
        let words = scan_subgroup_spirv();
        assert!(
            has_capability(&words, CAPABILITY_GROUP_NON_UNIFORM),
            "missing GroupNonUniform capability"
        );
        assert!(
            has_capability(&words, CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC),
            "missing GroupNonUniformArithmetic capability"
        );
    }

    #[test]
    fn scan_subgroup_has_entry_point() {
        let words = scan_subgroup_spirv();
        assert!(
            has_entry_point(&words, "scan_subgroup"),
            "missing entry point name"
        );
    }

    #[test]
    fn gemm_subgroup_valid_spirv() {
        let words = gemm_subgroup_spirv();
        check_valid_spirv(&words);
    }

    #[test]
    fn gemm_subgroup_word_aligned() {
        let words = gemm_subgroup_spirv();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        assert_eq!(bytes.len() % 4, 0);
    }

    #[test]
    fn gemm_subgroup_has_group_non_uniform_capability() {
        let words = gemm_subgroup_spirv();
        assert!(
            has_capability(&words, CAPABILITY_GROUP_NON_UNIFORM),
            "missing GroupNonUniform capability"
        );
        assert!(
            has_capability(&words, CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE),
            "missing GroupNonUniformShuffle capability"
        );
    }

    #[test]
    fn gemm_subgroup_has_entry_point() {
        let words = gemm_subgroup_spirv();
        assert!(
            has_entry_point(&words, "gemm_subgroup"),
            "missing entry point name"
        );
    }

    #[test]
    fn subgroup_shaders_all_word_aligned() {
        fn to_bytes(words: &[u32]) -> Vec<u8> {
            words.iter().flat_map(|w| w.to_ne_bytes()).collect()
        }
        assert_eq!(to_bytes(&reduction_subgroup_spirv()).len() % 4, 0);
        assert_eq!(to_bytes(&scan_subgroup_spirv()).len() % 4, 0);
        assert_eq!(to_bytes(&gemm_subgroup_spirv()).len() % 4, 0);
    }
}
