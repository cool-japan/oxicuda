//! Subgroup (warp-level) optimized SPIR-V compute shader generators.
//!
//! This module provides generators that exploit Vulkan subgroup operations
//! (`OpGroupNonUniform*`) for faster reductions and prefix scans compared
//! to the shared-memory-only approach.
//!
//! Two-phase strategy:
//! 1. **Intra-subgroup** — use hardware subgroup intrinsics (one instruction
//!    per warp/wavefront).
//! 2. **Cross-subgroup** — merge partial results via shared memory + barrier.

use super::{
    // Built-ins
    BUILTIN_GLOBAL_INVOCATION_ID,
    BUILTIN_LOCAL_INVOCATION_ID,
    BUILTIN_NUM_SUBGROUPS,
    BUILTIN_SUBGROUP_ID,
    BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID,
    BUILTIN_SUBGROUP_SIZE,
    CAPABILITY_GROUP_NON_UNIFORM,
    CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC,
    CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE,
    // Capabilities
    CAPABILITY_SHADER,
    DECORATION_ARRAY_STRIDE,
    DECORATION_BINDING,
    DECORATION_BLOCK,
    // Decorations
    DECORATION_BUILTIN,
    DECORATION_DESCRIPTOR_SET,
    DECORATION_OFFSET,
    // Other
    FUNCTION_CONTROL_NONE,
    GLSL_F_MAX,
    GLSL_F_MIN,
    GROUP_OPERATION_INCLUSIVE_SCAN,
    // Group operations
    GROUP_OPERATION_REDUCE,
    MEMORY_SEMANTICS_ACQUIRE_RELEASE,
    MEMORY_SEMANTICS_WORKGROUP_MEMORY,
    OP_F_ADD,
    OP_GROUP_NON_UNIFORM_F_ADD,
    OP_GROUP_NON_UNIFORM_F_MAX,
    OP_GROUP_NON_UNIFORM_F_MIN,
    OP_GROUP_NON_UNIFORM_I_ADD,
    OP_GROUP_NON_UNIFORM_SHUFFLE,
    // Opcodes
    OP_I_ADD,
    OP_I_SUB,
    OP_U_DIV,
    OP_U_LESS_THAN,
    SCOPE_SUBGROUP,
    // Scopes & memory semantics
    SCOPE_WORKGROUP,
    SPIRV_VERSION_1_3,
    STORAGE_CLASS_FUNCTION,
    // Storage classes
    STORAGE_CLASS_INPUT,
    STORAGE_CLASS_STORAGE_BUFFER,
    STORAGE_CLASS_WORKGROUP,
    SpvModule,
    WORKGROUP_SIZE,
};

// ─── Subgroup-optimized reduction ────────────────────────────

/// Generate a SPIR-V compute shader for subgroup-optimized reduction.
///
/// Uses `OpGroupNonUniformFAdd` / `OpGroupNonUniformIAdd` for a fast
/// intra-subgroup reduce, followed by a shared-memory merge across
/// subgroups within the workgroup.
///
/// Supported `op` values: `"fadd"`, `"iadd"`, `"fmin"`, `"fmax"`.
///
/// Bindings: 0 = input `float[]`, 1 = output `float[]`,
/// 2 = params `uint[]` where `params[0] = count`.
///
/// The output buffer must hold at least `ceil(count / WORKGROUP_SIZE)` elements.
/// Each workgroup writes one reduced value at `output[workgroup_id]`.
pub fn reduction_subgroup_spirv(op: &str) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);

    // ── IDs ──
    let main_fn = m.alloc_id();
    let ty_void = m.alloc_id();
    let ty_bool = m.alloc_id();
    let ty_uint = m.alloc_id();
    let ty_float = m.alloc_id();
    let ty_v3uint = m.alloc_id();
    let ty_fn_void = m.alloc_id();
    let ty_ptr_input_v3uint = m.alloc_id();
    let ty_ptr_input_uint = m.alloc_id();
    let ty_rt_array_float = m.alloc_id();
    let ty_rt_array_uint = m.alloc_id();
    let ty_ptr_sb_float = m.alloc_id();
    let ty_ptr_sb_uint = m.alloc_id();
    let ty_ptr_func_float = m.alloc_id();
    let _ty_ptr_func_uint = m.alloc_id();
    let c_uint_0 = m.alloc_id();
    let c_uint_1 = m.alloc_id();
    let _c_float_0 = m.alloc_id();
    let _c_float_1 = m.alloc_id();
    let var_gid = m.alloc_id();
    let _glsl_ext = m.alloc_id();

    // Additional IDs for subgroup built-ins
    let var_lid = m.alloc_id();
    let var_subgroup_lid = m.alloc_id();
    let var_subgroup_size = m.alloc_id();
    let var_subgroup_id = m.alloc_id();
    let var_num_subgroups = m.alloc_id();

    // Shared memory array: float[32] (max subgroups per workgroup)
    let c_uint_32 = m.alloc_id();
    let ty_array_float_32 = m.alloc_id();
    let ty_ptr_wg_array = m.alloc_id();
    let ty_ptr_wg_float = m.alloc_id();
    let var_shared = m.alloc_id();

    // Scope / memory semantics constants
    let c_scope_subgroup = m.alloc_id();
    let c_scope_workgroup = m.alloc_id();
    let c_mem_semantics = m.alloc_id();

    // ── Capabilities ──
    m.emit_capability(CAPABILITY_SHADER);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC);

    m.emit_ext_inst_import(_glsl_ext, "GLSL.std.450");
    m.emit_memory_model();
    m.emit_entry_point(
        main_fn,
        "main",
        &[
            var_gid,
            var_lid,
            var_subgroup_lid,
            var_subgroup_size,
            var_subgroup_id,
            var_num_subgroups,
        ],
    );
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    // ── Decorations ──
    m.emit_decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);
    m.emit_decorate(var_lid, DECORATION_BUILTIN, &[BUILTIN_LOCAL_INVOCATION_ID]);
    m.emit_decorate(
        var_subgroup_lid,
        DECORATION_BUILTIN,
        &[BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID],
    );
    m.emit_decorate(
        var_subgroup_size,
        DECORATION_BUILTIN,
        &[BUILTIN_SUBGROUP_SIZE],
    );
    m.emit_decorate(var_subgroup_id, DECORATION_BUILTIN, &[BUILTIN_SUBGROUP_ID]);
    m.emit_decorate(
        var_num_subgroups,
        DECORATION_BUILTIN,
        &[BUILTIN_NUM_SUBGROUPS],
    );
    m.emit_decorate(ty_rt_array_float, DECORATION_ARRAY_STRIDE, &[4]);
    m.emit_decorate(ty_rt_array_uint, DECORATION_ARRAY_STRIDE, &[4]);

    // ── Types ──
    m.emit_type_void(ty_void);
    m.emit_type_bool(ty_bool);
    m.emit_type_int(ty_uint, 32, 0);
    m.emit_type_float(ty_float, 32);
    m.emit_type_vector(ty_v3uint, ty_uint, 3);
    m.emit_type_function(ty_fn_void, ty_void, &[]);
    m.emit_type_pointer(ty_ptr_input_v3uint, STORAGE_CLASS_INPUT, ty_v3uint);
    m.emit_type_pointer(ty_ptr_input_uint, STORAGE_CLASS_INPUT, ty_uint);
    m.emit_type_runtime_array(ty_rt_array_float, ty_float);
    m.emit_type_runtime_array(ty_rt_array_uint, ty_uint);
    m.emit_type_pointer(ty_ptr_sb_float, STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    m.emit_type_pointer(ty_ptr_sb_uint, STORAGE_CLASS_STORAGE_BUFFER, ty_uint);
    m.emit_type_pointer(ty_ptr_func_float, STORAGE_CLASS_FUNCTION, ty_float);

    // Constants
    m.emit_constant_u32(ty_uint, c_uint_0, 0);
    m.emit_constant_u32(ty_uint, c_uint_1, 1);
    m.emit_constant_f32(ty_float, _c_float_0, 0.0);
    m.emit_constant_f32(ty_float, _c_float_1, 1.0);
    m.emit_constant_u32(ty_uint, c_uint_32, 32);
    m.emit_constant_u32(ty_uint, c_scope_subgroup, SCOPE_SUBGROUP);
    m.emit_constant_u32(ty_uint, c_scope_workgroup, SCOPE_WORKGROUP);
    m.emit_constant_u32(
        ty_uint,
        c_mem_semantics,
        MEMORY_SEMANTICS_WORKGROUP_MEMORY | MEMORY_SEMANTICS_ACQUIRE_RELEASE,
    );

    // Shared memory: float[32]
    m.emit_type_array(ty_array_float_32, ty_float, c_uint_32);
    m.emit_type_pointer(ty_ptr_wg_array, STORAGE_CLASS_WORKGROUP, ty_array_float_32);
    m.emit_type_pointer(ty_ptr_wg_float, STORAGE_CLASS_WORKGROUP, ty_float);
    m.emit_variable(ty_ptr_wg_array, var_shared, STORAGE_CLASS_WORKGROUP);

    // Input variables
    m.emit_variable(ty_ptr_input_v3uint, var_gid, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_v3uint, var_lid, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_subgroup_lid, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_subgroup_size, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_subgroup_id, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_num_subgroups, STORAGE_CLASS_INPUT);

    // SSBOs
    let input_s = m.alloc_id();
    let input_sp = m.alloc_id();
    let input_var = m.alloc_id();
    m.emit_decorate(input_s, DECORATION_BLOCK, &[]);
    m.emit_member_decorate(input_s, 0, DECORATION_OFFSET, &[0]);
    m.emit_decorate(input_var, DECORATION_DESCRIPTOR_SET, &[0]);
    m.emit_decorate(input_var, DECORATION_BINDING, &[0]);
    m.emit_type_struct(input_s, &[ty_rt_array_float]);
    m.emit_type_pointer(input_sp, STORAGE_CLASS_STORAGE_BUFFER, input_s);
    m.emit_variable(input_sp, input_var, STORAGE_CLASS_STORAGE_BUFFER);

    let output_s = m.alloc_id();
    let output_sp = m.alloc_id();
    let output_var = m.alloc_id();
    m.emit_decorate(output_s, DECORATION_BLOCK, &[]);
    m.emit_member_decorate(output_s, 0, DECORATION_OFFSET, &[0]);
    m.emit_decorate(output_var, DECORATION_DESCRIPTOR_SET, &[0]);
    m.emit_decorate(output_var, DECORATION_BINDING, &[1]);
    m.emit_type_struct(output_s, &[ty_rt_array_float]);
    m.emit_type_pointer(output_sp, STORAGE_CLASS_STORAGE_BUFFER, output_s);
    m.emit_variable(output_sp, output_var, STORAGE_CLASS_STORAGE_BUFFER);

    let params_s = m.alloc_id();
    let params_sp = m.alloc_id();
    let params_var = m.alloc_id();
    m.emit_decorate(params_s, DECORATION_BLOCK, &[]);
    m.emit_member_decorate(params_s, 0, DECORATION_OFFSET, &[0]);
    m.emit_decorate(params_var, DECORATION_DESCRIPTOR_SET, &[0]);
    m.emit_decorate(params_var, DECORATION_BINDING, &[2]);
    m.emit_type_struct(params_s, &[ty_rt_array_uint]);
    m.emit_type_pointer(params_sp, STORAGE_CLASS_STORAGE_BUFFER, params_s);
    m.emit_variable(params_sp, params_var, STORAGE_CLASS_STORAGE_BUFFER);

    // Determine opcode for subgroup reduction
    let subgroup_op = match op {
        "fadd" => OP_GROUP_NON_UNIFORM_F_ADD,
        "iadd" => OP_GROUP_NON_UNIFORM_I_ADD,
        "fmin" => OP_GROUP_NON_UNIFORM_F_MIN,
        "fmax" => OP_GROUP_NON_UNIFORM_F_MAX,
        _ => OP_GROUP_NON_UNIFORM_F_ADD, // default
    };

    let identity_val = match op {
        "fmin" => f32::INFINITY,
        "fmax" => f32::NEG_INFINITY,
        _ => 0.0,
    };
    let c_identity = m.alloc_id();
    m.emit_constant_f32(ty_float, c_identity, identity_val);

    // ── Function body ──
    let label_entry = m.alloc_id();
    let label_bounds = m.alloc_id();
    let label_oob = m.alloc_id();
    let label_after_load = m.alloc_id();
    let label_sublane0 = m.alloc_id();
    let label_after_shared = m.alloc_id();
    let label_loop_h = m.alloc_id();
    let label_loop_b = m.alloc_id();
    let label_loop_c = m.alloc_id();
    let label_loop_m = m.alloc_id();
    let label_lane0_write = m.alloc_id();
    let label_end = m.alloc_id();

    m.emit_function(ty_void, main_fn, FUNCTION_CONTROL_NONE, ty_fn_void);
    m.emit_label(label_entry);

    let var_local_val = m.alloc_id();
    m.emit_variable(ty_ptr_func_float, var_local_val, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_local_val, c_identity);

    // Load GlobalInvocationId.x
    let gid_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_input_uint, gid_ptr, var_gid, &[c_uint_0]);
    let gid = m.alloc_id();
    m.emit_load(ty_uint, gid, gid_ptr);

    // Load params[0] = count
    let count_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_uint, count_ptr, params_var, &[c_uint_0, c_uint_0]);
    let count = m.alloc_id();
    m.emit_load(ty_uint, count, count_ptr);

    // Load subgroup built-ins
    let sg_lid = m.alloc_id();
    m.emit_load(ty_uint, sg_lid, var_subgroup_lid);
    let sg_id = m.alloc_id();
    m.emit_load(ty_uint, sg_id, var_subgroup_id);
    let num_sg = m.alloc_id();
    m.emit_load(ty_uint, num_sg, var_num_subgroups);

    // Bounds check: if gid < count, load input[gid]; else use identity
    let cond_bounds = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, cond_bounds, gid, count]);
    m.emit_selection_merge(label_after_load);
    m.emit_branch_conditional(cond_bounds, label_bounds, label_oob);

    // In-bounds: load input value
    m.emit_label(label_bounds);
    let inp_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_float, inp_ptr, input_var, &[c_uint_0, gid]);
    let inp_val = m.alloc_id();
    m.emit_load(ty_float, inp_val, inp_ptr);
    m.emit_store(var_local_val, inp_val);
    m.emit_branch(label_after_load);

    // Out-of-bounds: identity already stored
    m.emit_label(label_oob);
    m.emit_branch(label_after_load);

    m.emit_label(label_after_load);
    let val = m.alloc_id();
    m.emit_load(ty_float, val, var_local_val);

    // Phase 1: subgroup reduction
    let sg_reduced = m.alloc_id();
    m.emit(
        subgroup_op,
        &[
            ty_float,
            sg_reduced,
            c_scope_subgroup,
            GROUP_OPERATION_REDUCE,
            val,
        ],
    );

    // Write subgroup result to shared memory if lane 0
    let is_lane0 = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, is_lane0, sg_lid, c_uint_1]);
    m.emit_selection_merge(label_after_shared);
    m.emit_branch_conditional(is_lane0, label_sublane0, label_after_shared);

    m.emit_label(label_sublane0);
    let shared_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_wg_float, shared_ptr, var_shared, &[sg_id]);
    m.emit_store(shared_ptr, sg_reduced);
    m.emit_branch(label_after_shared);

    m.emit_label(label_after_shared);

    // Barrier: wait for all subgroups to finish writing
    m.emit_control_barrier(c_scope_workgroup, c_scope_workgroup, c_mem_semantics);

    // Phase 2: first thread in workgroup (local id.x == 0) does final merge
    let lid_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_input_uint, lid_ptr, var_lid, &[c_uint_0]);
    let lid_x = m.alloc_id();
    m.emit_load(ty_uint, lid_x, lid_ptr);

    let is_thread0 = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, is_thread0, lid_x, c_uint_1]);
    m.emit_selection_merge(label_end);
    m.emit_branch_conditional(is_thread0, label_lane0_write, label_end);

    m.emit_label(label_lane0_write);

    // Loop over shared[0..num_subgroups], accumulating
    let var_acc = m.alloc_id();
    m.emit_variable(ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_acc, c_identity);

    let var_i = m.alloc_id();
    let ty_ptr_func_uint = m.alloc_id();
    m.emit_type_pointer(ty_ptr_func_uint, STORAGE_CLASS_FUNCTION, ty_uint);
    m.emit_variable(ty_ptr_func_uint, var_i, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_i, c_uint_0);

    m.emit_branch(label_loop_h);

    m.emit_label(label_loop_h);
    let i_val = m.alloc_id();
    m.emit_load(ty_uint, i_val, var_i);
    let loop_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, loop_cond, i_val, num_sg]);
    m.emit_loop_merge(label_loop_m, label_loop_c);
    m.emit_branch_conditional(loop_cond, label_loop_b, label_loop_m);

    m.emit_label(label_loop_b);
    let s_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_wg_float, s_ptr, var_shared, &[i_val]);
    let s_val = m.alloc_id();
    m.emit_load(ty_float, s_val, s_ptr);
    let old_acc = m.alloc_id();
    m.emit_load(ty_float, old_acc, var_acc);
    let new_acc = m.alloc_id();

    match op {
        "fmin" => {
            m.emit_glsl_ext(_glsl_ext, ty_float, new_acc, GLSL_F_MIN, &[old_acc, s_val]);
        }
        "fmax" => {
            m.emit_glsl_ext(_glsl_ext, ty_float, new_acc, GLSL_F_MAX, &[old_acc, s_val]);
        }
        _ => {
            m.emit(OP_F_ADD, &[ty_float, new_acc, old_acc, s_val]);
        }
    }
    m.emit_store(var_acc, new_acc);

    m.emit_branch(label_loop_c);

    m.emit_label(label_loop_c);
    let i_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[ty_uint, i_inc, i_val, c_uint_1]);
    m.emit_store(var_i, i_inc);
    m.emit_branch(label_loop_h);

    m.emit_label(label_loop_m);

    // Compute workgroup id = gid / WORKGROUP_SIZE
    let c_wg_size = m.alloc_id();
    m.emit_constant_u32(ty_uint, c_wg_size, WORKGROUP_SIZE);
    let wg_id = m.alloc_id();
    m.emit(OP_U_DIV, &[ty_uint, wg_id, gid, c_wg_size]);

    let final_val = m.alloc_id();
    m.emit_load(ty_float, final_val, var_acc);
    let out_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_float, out_ptr, output_var, &[c_uint_0, wg_id]);
    m.emit_store(out_ptr, final_val);

    m.emit_branch(label_end);

    m.emit_label(label_end);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── Subgroup-optimized prefix scan ─────────────────────────

/// Generate a SPIR-V compute shader for subgroup-optimized inclusive prefix scan.
///
/// Uses `OpGroupNonUniformFAdd` with `InclusiveScan` for a fast intra-subgroup
/// scan, then `OpGroupNonUniformShuffle` to propagate subgroup totals.
///
/// Supported `op` values: `"fadd"`, `"iadd"`, `"fmin"`, `"fmax"`.
///
/// Bindings: 0 = input `float[]`, 1 = output `float[]`,
/// 2 = params `uint[]` where `params[0] = count`.
///
/// Each invocation writes one output element. The scan is local to each workgroup.
pub fn scan_subgroup_spirv(op: &str) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);

    // ── IDs ──
    let main_fn = m.alloc_id();
    let ty_void = m.alloc_id();
    let ty_bool = m.alloc_id();
    let ty_uint = m.alloc_id();
    let ty_float = m.alloc_id();
    let ty_v3uint = m.alloc_id();
    let ty_fn_void = m.alloc_id();
    let ty_ptr_input_v3uint = m.alloc_id();
    let ty_ptr_input_uint = m.alloc_id();
    let ty_rt_array_float = m.alloc_id();
    let ty_rt_array_uint = m.alloc_id();
    let ty_ptr_sb_float = m.alloc_id();
    let ty_ptr_sb_uint = m.alloc_id();
    let ty_ptr_func_float = m.alloc_id();
    let _ty_ptr_func_uint = m.alloc_id();

    let c_uint_0 = m.alloc_id();
    let c_uint_1 = m.alloc_id();
    let c_float_0 = m.alloc_id();
    let c_float_1 = m.alloc_id();
    let var_gid = m.alloc_id();
    let glsl_ext = m.alloc_id();

    // Subgroup built-ins
    let var_lid = m.alloc_id();
    let var_subgroup_lid = m.alloc_id();
    let var_subgroup_size = m.alloc_id();
    let var_subgroup_id = m.alloc_id();
    let var_num_subgroups = m.alloc_id();

    // Shared memory
    let c_uint_32 = m.alloc_id();
    let ty_array_float_32 = m.alloc_id();
    let ty_ptr_wg_array = m.alloc_id();
    let ty_ptr_wg_float = m.alloc_id();
    let var_shared = m.alloc_id();

    // Scope constants
    let c_scope_subgroup = m.alloc_id();
    let c_scope_workgroup = m.alloc_id();
    let c_mem_semantics = m.alloc_id();

    // ── Capabilities ──
    m.emit_capability(CAPABILITY_SHADER);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE);

    m.emit_ext_inst_import(glsl_ext, "GLSL.std.450");
    m.emit_memory_model();
    m.emit_entry_point(
        main_fn,
        "main",
        &[
            var_gid,
            var_lid,
            var_subgroup_lid,
            var_subgroup_size,
            var_subgroup_id,
            var_num_subgroups,
        ],
    );
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    // ── Decorations ──
    m.emit_decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);
    m.emit_decorate(var_lid, DECORATION_BUILTIN, &[BUILTIN_LOCAL_INVOCATION_ID]);
    m.emit_decorate(
        var_subgroup_lid,
        DECORATION_BUILTIN,
        &[BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID],
    );
    m.emit_decorate(
        var_subgroup_size,
        DECORATION_BUILTIN,
        &[BUILTIN_SUBGROUP_SIZE],
    );
    m.emit_decorate(var_subgroup_id, DECORATION_BUILTIN, &[BUILTIN_SUBGROUP_ID]);
    m.emit_decorate(
        var_num_subgroups,
        DECORATION_BUILTIN,
        &[BUILTIN_NUM_SUBGROUPS],
    );
    m.emit_decorate(ty_rt_array_float, DECORATION_ARRAY_STRIDE, &[4]);
    m.emit_decorate(ty_rt_array_uint, DECORATION_ARRAY_STRIDE, &[4]);

    // ── Types ──
    m.emit_type_void(ty_void);
    m.emit_type_bool(ty_bool);
    m.emit_type_int(ty_uint, 32, 0);
    m.emit_type_float(ty_float, 32);
    m.emit_type_vector(ty_v3uint, ty_uint, 3);
    m.emit_type_function(ty_fn_void, ty_void, &[]);
    m.emit_type_pointer(ty_ptr_input_v3uint, STORAGE_CLASS_INPUT, ty_v3uint);
    m.emit_type_pointer(ty_ptr_input_uint, STORAGE_CLASS_INPUT, ty_uint);
    m.emit_type_runtime_array(ty_rt_array_float, ty_float);
    m.emit_type_runtime_array(ty_rt_array_uint, ty_uint);
    m.emit_type_pointer(ty_ptr_sb_float, STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    m.emit_type_pointer(ty_ptr_sb_uint, STORAGE_CLASS_STORAGE_BUFFER, ty_uint);
    m.emit_type_pointer(ty_ptr_func_float, STORAGE_CLASS_FUNCTION, ty_float);

    // Constants
    m.emit_constant_u32(ty_uint, c_uint_0, 0);
    m.emit_constant_u32(ty_uint, c_uint_1, 1);
    m.emit_constant_f32(ty_float, c_float_0, 0.0);
    m.emit_constant_f32(ty_float, c_float_1, 1.0);
    m.emit_constant_u32(ty_uint, c_uint_32, 32);
    m.emit_constant_u32(ty_uint, c_scope_subgroup, SCOPE_SUBGROUP);
    m.emit_constant_u32(ty_uint, c_scope_workgroup, SCOPE_WORKGROUP);
    m.emit_constant_u32(
        ty_uint,
        c_mem_semantics,
        MEMORY_SEMANTICS_WORKGROUP_MEMORY | MEMORY_SEMANTICS_ACQUIRE_RELEASE,
    );

    let identity_val = match op {
        "fmin" => f32::INFINITY,
        "fmax" => f32::NEG_INFINITY,
        _ => 0.0,
    };
    let c_identity = m.alloc_id();
    m.emit_constant_f32(ty_float, c_identity, identity_val);

    // Shared memory
    m.emit_type_array(ty_array_float_32, ty_float, c_uint_32);
    m.emit_type_pointer(ty_ptr_wg_array, STORAGE_CLASS_WORKGROUP, ty_array_float_32);
    m.emit_type_pointer(ty_ptr_wg_float, STORAGE_CLASS_WORKGROUP, ty_float);
    m.emit_variable(ty_ptr_wg_array, var_shared, STORAGE_CLASS_WORKGROUP);

    // Input variables
    m.emit_variable(ty_ptr_input_v3uint, var_gid, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_v3uint, var_lid, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_subgroup_lid, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_subgroup_size, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_subgroup_id, STORAGE_CLASS_INPUT);
    m.emit_variable(ty_ptr_input_uint, var_num_subgroups, STORAGE_CLASS_INPUT);

    // SSBOs
    let input_s = m.alloc_id();
    let input_sp = m.alloc_id();
    let input_var = m.alloc_id();
    m.emit_decorate(input_s, DECORATION_BLOCK, &[]);
    m.emit_member_decorate(input_s, 0, DECORATION_OFFSET, &[0]);
    m.emit_decorate(input_var, DECORATION_DESCRIPTOR_SET, &[0]);
    m.emit_decorate(input_var, DECORATION_BINDING, &[0]);
    m.emit_type_struct(input_s, &[ty_rt_array_float]);
    m.emit_type_pointer(input_sp, STORAGE_CLASS_STORAGE_BUFFER, input_s);
    m.emit_variable(input_sp, input_var, STORAGE_CLASS_STORAGE_BUFFER);

    let output_s = m.alloc_id();
    let output_sp = m.alloc_id();
    let output_var = m.alloc_id();
    m.emit_decorate(output_s, DECORATION_BLOCK, &[]);
    m.emit_member_decorate(output_s, 0, DECORATION_OFFSET, &[0]);
    m.emit_decorate(output_var, DECORATION_DESCRIPTOR_SET, &[0]);
    m.emit_decorate(output_var, DECORATION_BINDING, &[1]);
    m.emit_type_struct(output_s, &[ty_rt_array_float]);
    m.emit_type_pointer(output_sp, STORAGE_CLASS_STORAGE_BUFFER, output_s);
    m.emit_variable(output_sp, output_var, STORAGE_CLASS_STORAGE_BUFFER);

    let params_s = m.alloc_id();
    let params_sp = m.alloc_id();
    let params_var = m.alloc_id();
    m.emit_decorate(params_s, DECORATION_BLOCK, &[]);
    m.emit_member_decorate(params_s, 0, DECORATION_OFFSET, &[0]);
    m.emit_decorate(params_var, DECORATION_DESCRIPTOR_SET, &[0]);
    m.emit_decorate(params_var, DECORATION_BINDING, &[2]);
    m.emit_type_struct(params_s, &[ty_rt_array_uint]);
    m.emit_type_pointer(params_sp, STORAGE_CLASS_STORAGE_BUFFER, params_s);
    m.emit_variable(params_sp, params_var, STORAGE_CLASS_STORAGE_BUFFER);

    // Select the GroupNonUniform opcode
    let subgroup_op_code = match op {
        "fadd" => OP_GROUP_NON_UNIFORM_F_ADD,
        "iadd" => OP_GROUP_NON_UNIFORM_I_ADD,
        "fmin" => OP_GROUP_NON_UNIFORM_F_MIN,
        "fmax" => OP_GROUP_NON_UNIFORM_F_MAX,
        _ => OP_GROUP_NON_UNIFORM_F_ADD,
    };

    // ── Function body ──
    let label_entry = m.alloc_id();
    let label_bounds = m.alloc_id();
    let label_merge_bounds = m.alloc_id();
    let label_sublane0 = m.alloc_id();
    let label_after_shared_write = m.alloc_id();
    let label_loop_h = m.alloc_id();
    let label_loop_b = m.alloc_id();
    let label_loop_c = m.alloc_id();
    let label_loop_m = m.alloc_id();
    let label_write_out = m.alloc_id();
    let label_end = m.alloc_id();

    m.emit_function(ty_void, main_fn, FUNCTION_CONTROL_NONE, ty_fn_void);
    m.emit_label(label_entry);

    // Local variable for this thread's value
    let var_local = m.alloc_id();
    m.emit_variable(ty_ptr_func_float, var_local, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_local, c_identity);

    // Load GlobalInvocationId.x
    let gid_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_input_uint, gid_ptr, var_gid, &[c_uint_0]);
    let gid = m.alloc_id();
    m.emit_load(ty_uint, gid, gid_ptr);

    // Load count
    let count_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_uint, count_ptr, params_var, &[c_uint_0, c_uint_0]);
    let count = m.alloc_id();
    m.emit_load(ty_uint, count, count_ptr);

    // Load subgroup built-ins
    let sg_lid = m.alloc_id();
    m.emit_load(ty_uint, sg_lid, var_subgroup_lid);
    let sg_size = m.alloc_id();
    m.emit_load(ty_uint, sg_size, var_subgroup_size);
    let sg_id = m.alloc_id();
    m.emit_load(ty_uint, sg_id, var_subgroup_id);
    let num_sg = m.alloc_id();
    m.emit_load(ty_uint, num_sg, var_num_subgroups);

    // Bounds check
    let cond_bounds = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, cond_bounds, gid, count]);
    m.emit_selection_merge(label_merge_bounds);
    m.emit_branch_conditional(cond_bounds, label_bounds, label_merge_bounds);

    m.emit_label(label_bounds);
    let inp_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_float, inp_ptr, input_var, &[c_uint_0, gid]);
    let inp_val = m.alloc_id();
    m.emit_load(ty_float, inp_val, inp_ptr);
    m.emit_store(var_local, inp_val);
    m.emit_branch(label_merge_bounds);

    m.emit_label(label_merge_bounds);
    let val = m.alloc_id();
    m.emit_load(ty_float, val, var_local);

    // Phase 1: subgroup inclusive scan
    let sg_scanned = m.alloc_id();
    m.emit(
        subgroup_op_code,
        &[
            ty_float,
            sg_scanned,
            c_scope_subgroup,
            GROUP_OPERATION_INCLUSIVE_SCAN,
            val,
        ],
    );

    // The last lane in each subgroup holds the subgroup total.
    // Get it via shuffle from lane (subgroup_size - 1).
    let last_lane = m.alloc_id();
    m.emit(OP_I_SUB, &[ty_uint, last_lane, sg_size, c_uint_1]);
    let sg_total = m.alloc_id();
    m.emit(
        OP_GROUP_NON_UNIFORM_SHUFFLE,
        &[ty_float, sg_total, c_scope_subgroup, sg_scanned, last_lane],
    );

    // Lane 0 writes subgroup total to shared[subgroup_id]
    let is_lane0 = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, is_lane0, sg_lid, c_uint_1]);
    m.emit_selection_merge(label_after_shared_write);
    m.emit_branch_conditional(is_lane0, label_sublane0, label_after_shared_write);

    m.emit_label(label_sublane0);
    let shared_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_wg_float, shared_ptr, var_shared, &[sg_id]);
    m.emit_store(shared_ptr, sg_total);
    m.emit_branch(label_after_shared_write);

    m.emit_label(label_after_shared_write);

    // Barrier
    m.emit_control_barrier(c_scope_workgroup, c_scope_workgroup, c_mem_semantics);

    // Phase 2: compute prefix of subgroup totals for subgroups before this one.
    let var_prefix = m.alloc_id();
    m.emit_variable(ty_ptr_func_float, var_prefix, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_prefix, c_identity);

    let ty_ptr_func_uint_id = m.alloc_id();
    m.emit_type_pointer(ty_ptr_func_uint_id, STORAGE_CLASS_FUNCTION, ty_uint);
    let var_j = m.alloc_id();
    m.emit_variable(ty_ptr_func_uint_id, var_j, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_j, c_uint_0);

    m.emit_branch(label_loop_h);

    m.emit_label(label_loop_h);
    let j_val = m.alloc_id();
    m.emit_load(ty_uint, j_val, var_j);
    let loop_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, loop_cond, j_val, sg_id]);
    m.emit_loop_merge(label_loop_m, label_loop_c);
    m.emit_branch_conditional(loop_cond, label_loop_b, label_loop_m);

    m.emit_label(label_loop_b);
    let s_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_wg_float, s_ptr, var_shared, &[j_val]);
    let s_val = m.alloc_id();
    m.emit_load(ty_float, s_val, s_ptr);
    let old_prefix = m.alloc_id();
    m.emit_load(ty_float, old_prefix, var_prefix);
    let new_prefix = m.alloc_id();

    match op {
        "fmin" => {
            m.emit_glsl_ext(
                glsl_ext,
                ty_float,
                new_prefix,
                GLSL_F_MIN,
                &[old_prefix, s_val],
            );
        }
        "fmax" => {
            m.emit_glsl_ext(
                glsl_ext,
                ty_float,
                new_prefix,
                GLSL_F_MAX,
                &[old_prefix, s_val],
            );
        }
        _ => {
            m.emit(OP_F_ADD, &[ty_float, new_prefix, old_prefix, s_val]);
        }
    }
    m.emit_store(var_prefix, new_prefix);

    m.emit_branch(label_loop_c);
    m.emit_label(label_loop_c);
    let j_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[ty_uint, j_inc, j_val, c_uint_1]);
    m.emit_store(var_j, j_inc);
    m.emit_branch(label_loop_h);

    m.emit_label(label_loop_m);

    // Final value = sg_scanned + prefix_from_earlier_subgroups
    let prefix_val = m.alloc_id();
    m.emit_load(ty_float, prefix_val, var_prefix);
    let final_val = m.alloc_id();
    match op {
        "fmin" => {
            m.emit_glsl_ext(
                glsl_ext,
                ty_float,
                final_val,
                GLSL_F_MIN,
                &[sg_scanned, prefix_val],
            );
        }
        "fmax" => {
            m.emit_glsl_ext(
                glsl_ext,
                ty_float,
                final_val,
                GLSL_F_MAX,
                &[sg_scanned, prefix_val],
            );
        }
        _ => {
            m.emit(OP_F_ADD, &[ty_float, final_val, sg_scanned, prefix_val]);
        }
    }

    // Write output if in bounds
    let cond_write = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, cond_write, gid, count]);
    m.emit_selection_merge(label_end);
    m.emit_branch_conditional(cond_write, label_write_out, label_end);

    m.emit_label(label_write_out);
    let out_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_float, out_ptr, output_var, &[c_uint_0, gid]);
    m.emit_store(out_ptr, final_val);
    m.emit_branch(label_end);

    m.emit_label(label_end);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::{
        CAPABILITY_GROUP_NON_UNIFORM, CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC,
        CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE, OP_CAPABILITY, OP_GROUP_NON_UNIFORM_F_ADD,
        OP_GROUP_NON_UNIFORM_F_MAX, OP_GROUP_NON_UNIFORM_F_MIN, OP_GROUP_NON_UNIFORM_I_ADD,
        OP_GROUP_NON_UNIFORM_SHUFFLE, SPIRV_MAGIC, SPIRV_VERSION_1_3,
    };
    use super::*;

    fn check_valid_spirv(words: &[u32]) {
        assert!(words.len() >= 5, "too short for SPIR-V header");
        assert_eq!(words[0], SPIRV_MAGIC, "bad magic");
        assert!(words[3] > 0, "ID bound must be > 0");
        assert_eq!(words[4], 0, "schema must be 0");
    }

    /// Helper: check that a SPIR-V word stream contains a specific capability.
    fn contains_capability(words: &[u32], cap: u32) -> bool {
        let expected_header = (2 << 16) | OP_CAPABILITY;
        words
            .windows(2)
            .any(|w| w[0] == expected_header && w[1] == cap)
    }

    /// Helper: check that a SPIR-V word stream contains a GroupNonUniform opcode.
    fn contains_opcode(words: &[u32], opcode: u32) -> bool {
        words.iter().any(|&w| (w & 0xFFFF) == opcode)
    }

    #[test]
    fn reduction_subgroup_fadd_valid() {
        let words = reduction_subgroup_spirv("fadd");
        check_valid_spirv(&words);
        assert_eq!(words[1], SPIRV_VERSION_1_3);
        assert!(words.len() > 50);
    }

    #[test]
    fn reduction_subgroup_has_capabilities() {
        let words = reduction_subgroup_spirv("fadd");
        assert!(
            contains_capability(&words, CAPABILITY_GROUP_NON_UNIFORM),
            "missing GroupNonUniform capability"
        );
        assert!(
            contains_capability(&words, CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC),
            "missing GroupNonUniformArithmetic capability"
        );
    }

    #[test]
    fn reduction_subgroup_has_group_op() {
        let words = reduction_subgroup_spirv("fadd");
        assert!(
            contains_opcode(&words, OP_GROUP_NON_UNIFORM_F_ADD),
            "missing OpGroupNonUniformFAdd"
        );
    }

    #[test]
    fn reduction_subgroup_iadd_valid() {
        let words = reduction_subgroup_spirv("iadd");
        check_valid_spirv(&words);
        assert!(
            contains_opcode(&words, OP_GROUP_NON_UNIFORM_I_ADD),
            "missing OpGroupNonUniformIAdd"
        );
    }

    #[test]
    fn reduction_subgroup_fmin_valid() {
        let words = reduction_subgroup_spirv("fmin");
        check_valid_spirv(&words);
        assert!(
            contains_opcode(&words, OP_GROUP_NON_UNIFORM_F_MIN),
            "missing OpGroupNonUniformFMin"
        );
    }

    #[test]
    fn reduction_subgroup_fmax_valid() {
        let words = reduction_subgroup_spirv("fmax");
        check_valid_spirv(&words);
        assert!(
            contains_opcode(&words, OP_GROUP_NON_UNIFORM_F_MAX),
            "missing OpGroupNonUniformFMax"
        );
    }

    #[test]
    fn scan_subgroup_fadd_valid() {
        let words = scan_subgroup_spirv("fadd");
        check_valid_spirv(&words);
        assert_eq!(words[1], SPIRV_VERSION_1_3);
        assert!(words.len() > 50);
    }

    #[test]
    fn scan_subgroup_has_capabilities() {
        let words = scan_subgroup_spirv("fadd");
        assert!(
            contains_capability(&words, CAPABILITY_GROUP_NON_UNIFORM),
            "missing GroupNonUniform capability"
        );
        assert!(
            contains_capability(&words, CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC),
            "missing GroupNonUniformArithmetic capability"
        );
        assert!(
            contains_capability(&words, CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE),
            "missing GroupNonUniformShuffle capability"
        );
    }

    #[test]
    fn scan_subgroup_has_group_ops() {
        let words = scan_subgroup_spirv("fadd");
        assert!(
            contains_opcode(&words, OP_GROUP_NON_UNIFORM_F_ADD),
            "missing OpGroupNonUniformFAdd"
        );
        assert!(
            contains_opcode(&words, OP_GROUP_NON_UNIFORM_SHUFFLE),
            "missing OpGroupNonUniformShuffle"
        );
    }

    #[test]
    fn scan_subgroup_all_ops() {
        for op in &["fadd", "iadd", "fmin", "fmax"] {
            let words = scan_subgroup_spirv(op);
            check_valid_spirv(&words);
        }
    }

    #[test]
    fn reduction_subgroup_all_ops() {
        for op in &["fadd", "iadd", "fmin", "fmax"] {
            let words = reduction_subgroup_spirv(op);
            check_valid_spirv(&words);
        }
    }
}
