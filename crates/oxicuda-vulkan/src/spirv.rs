//! SPIR-V compute shader generators for the Vulkan backend.
//!
//! This module provides:
//! - A lightweight [`SpvModule`] builder for emitting valid SPIR-V binaries.
//! - Generator functions for **unary**, **binary**, **reduce**, and **GEMM**
//!   compute shaders used by the backend dispatch methods.
//! - The original [`trivial_compute_shader`] placeholder used for validation.
//!
//! All generated shaders operate on `f32` buffers via SSBO (`StorageBuffer`)
//! bindings at descriptor set 0.  Parameters (element count, matrix dims, etc.)
//! are passed through an additional SSBO.
//!
//! The generated SPIR-V targets version 1.3 (supported by Vulkan 1.1+).

use oxicuda_backend::{BinaryOp, ReduceOp, UnaryOp};

// ─── Constants ──────────────────────────────────────────────

/// SPIR-V magic number.
pub const SPIRV_MAGIC: u32 = 0x07230203;
/// SPIR-V version 1.2.
pub const SPIRV_VERSION_1_2: u32 = 0x0001_0200;
/// SPIR-V version 1.3 (used for compute shaders with StorageBuffer).
pub const SPIRV_VERSION_1_3: u32 = 0x0001_0300;
/// Generator magic — OxiCUDA Vulkan backend.
pub const SPIRV_GENERATOR: u32 = 0x000D_0001;

// ─── SPIR-V opcodes ─────────────────────────────────────────

const OP_EXT_INST_IMPORT: u32 = 11;
const OP_EXT_INST: u32 = 12;
const OP_MEMORY_MODEL: u32 = 14;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const OP_CAPABILITY: u32 = 17;
const OP_TYPE_VOID: u32 = 19;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_TYPE_FUNCTION: u32 = 33;
const OP_CONSTANT: u32 = 43;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_END: u32 = 56;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const OP_BITCAST: u32 = 124;
const OP_F_NEGATE: u32 = 127;
const OP_I_ADD: u32 = 128;
const OP_F_ADD: u32 = 129;
const OP_I_SUB: u32 = 130;
const OP_F_SUB: u32 = 131;
const OP_I_MUL: u32 = 132;
const OP_F_MUL: u32 = 133;
const OP_U_DIV: u32 = 134;
const OP_F_DIV: u32 = 136;
const OP_U_MOD: u32 = 137;
const OP_LOGICAL_AND: u32 = 167;
const OP_U_LESS_THAN: u32 = 176;
const OP_CONVERT_U_TO_F: u32 = 112;
const OP_LOOP_MERGE: u32 = 246;
const OP_SELECTION_MERGE: u32 = 247;
const OP_LABEL: u32 = 248;
const OP_BRANCH: u32 = 249;
const OP_BRANCH_CONDITIONAL: u32 = 250;
const OP_RETURN: u32 = 253;
const OP_CONTROL_BARRIER: u32 = 224;

// Subgroup / GroupNonUniform opcodes
const OP_GROUP_NON_UNIFORM_I_ADD: u32 = 349;
const OP_GROUP_NON_UNIFORM_F_ADD: u32 = 350;
const OP_GROUP_NON_UNIFORM_F_MIN: u32 = 354;
const OP_GROUP_NON_UNIFORM_F_MAX: u32 = 356;
const OP_GROUP_NON_UNIFORM_SHUFFLE: u32 = 345;

// Capabilities
const CAPABILITY_SHADER: u32 = 1;
const CAPABILITY_GROUP_NON_UNIFORM: u32 = 61;
const CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
const CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;

// Group operation constants (used as operand to GroupNonUniform* instructions)
const GROUP_OPERATION_REDUCE: u32 = 0;
const GROUP_OPERATION_INCLUSIVE_SCAN: u32 = 1;
// Scope constants
const SCOPE_WORKGROUP: u32 = 2;
const SCOPE_SUBGROUP: u32 = 3;

// Memory semantics
const MEMORY_SEMANTICS_WORKGROUP_MEMORY: u32 = 0x100; // WorkgroupMemory
const MEMORY_SEMANTICS_ACQUIRE_RELEASE: u32 = 0x8; // AcquireRelease

// Addressing / memory model
const ADDRESSING_MODEL_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;

// Execution model / mode
const EXECUTION_MODEL_GLCOMPUTE: u32 = 5;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

// Function control
const FUNCTION_CONTROL_NONE: u32 = 0;

// Decorations
const DECORATION_BLOCK: u32 = 2;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;

// BuiltIn values
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
const BUILTIN_SUBGROUP_SIZE: u32 = 36;
const BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
const BUILTIN_NUM_SUBGROUPS: u32 = 38;
const BUILTIN_SUBGROUP_ID: u32 = 40;
const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;

// Storage class for Workgroup shared memory
const STORAGE_CLASS_WORKGROUP: u32 = 4;

// Storage classes
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_FUNCTION: u32 = 7;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// Selection/loop control
const SELECTION_CONTROL_NONE: u32 = 0;
const LOOP_CONTROL_NONE: u32 = 0;

// GLSL.std.450 extended instruction numbers
const GLSL_F_ABS: u32 = 4;
const GLSL_TANH: u32 = 21;
const GLSL_EXP: u32 = 27;
const GLSL_LOG: u32 = 28;
const GLSL_SQRT: u32 = 31;
const GLSL_F_MIN: u32 = 39;
const GLSL_F_MAX: u32 = 40;

/// Workgroup size for 1-D compute shaders.
const WORKGROUP_SIZE: u32 = 256;

// ─── Minimal SPIR-V builder ──────────────────────────────────

/// Lightweight SPIR-V word-stream builder.
///
/// Emits valid SPIR-V instructions for simple compute shaders without
/// pulling in a full compiler.
pub struct SpvModule {
    words: Vec<u32>,
    /// Next available result ID.
    id_bound: u32,
}

impl SpvModule {
    /// Create a new module targeting SPIR-V `version`.
    pub fn with_version(version: u32) -> Self {
        let words = vec![SPIRV_MAGIC, version, SPIRV_GENERATOR, 0, 0];
        Self { words, id_bound: 1 }
    }

    /// Create a new module with a placeholder header (SPIR-V 1.2).
    pub fn new() -> Self {
        Self::with_version(SPIRV_VERSION_1_2)
    }

    /// Allocate a fresh result ID.
    pub fn alloc_id(&mut self) -> u32 {
        let id = self.id_bound;
        self.id_bound += 1;
        id
    }

    /// Emit a SPIR-V instruction.
    pub fn emit(&mut self, opcode: u32, operands: &[u32]) {
        let word_count = (1 + operands.len()) as u32;
        self.words.push((word_count << 16) | opcode);
        self.words.extend_from_slice(operands);
    }

    /// Emit a string as null-terminated UTF-8 packed into 32-bit words.
    pub fn string_words(s: &str) -> Vec<u32> {
        let bytes = s.as_bytes();
        let padded_len = (bytes.len() + 4) & !3;
        let mut out = vec![0u32; padded_len / 4];
        for (i, &b) in bytes.iter().enumerate() {
            out[i / 4] |= (b as u32) << ((i % 4) * 8);
        }
        out
    }

    /// Finalise the module: patch the ID bound and return the word vector.
    pub fn finalize(mut self) -> Vec<u32> {
        self.words[3] = self.id_bound;
        self.words
    }

    // ── Convenience emitters ─────────────────────────────────

    fn emit_capability(&mut self, cap: u32) {
        self.emit(OP_CAPABILITY, &[cap]);
    }

    fn emit_ext_inst_import(&mut self, id: u32, name: &str) {
        let mut ops = vec![id];
        ops.extend(Self::string_words(name));
        self.emit(OP_EXT_INST_IMPORT, &ops);
    }

    fn emit_memory_model(&mut self) {
        self.emit(
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
        );
    }

    fn emit_entry_point(&mut self, func_id: u32, name: &str, interfaces: &[u32]) {
        let mut ops = vec![EXECUTION_MODEL_GLCOMPUTE, func_id];
        ops.extend(Self::string_words(name));
        ops.extend_from_slice(interfaces);
        self.emit(OP_ENTRY_POINT, &ops);
    }

    fn emit_execution_mode_local_size(&mut self, func_id: u32, x: u32, y: u32, z: u32) {
        self.emit(
            OP_EXECUTION_MODE,
            &[func_id, EXECUTION_MODE_LOCAL_SIZE, x, y, z],
        );
    }

    fn emit_decorate(&mut self, target: u32, decoration: u32, operands: &[u32]) {
        let mut ops = vec![target, decoration];
        ops.extend_from_slice(operands);
        self.emit(OP_DECORATE, &ops);
    }

    fn emit_member_decorate(&mut self, ty: u32, member: u32, decoration: u32, operands: &[u32]) {
        let mut ops = vec![ty, member, decoration];
        ops.extend_from_slice(operands);
        self.emit(OP_MEMBER_DECORATE, &ops);
    }

    fn emit_type_void(&mut self, id: u32) {
        self.emit(OP_TYPE_VOID, &[id]);
    }

    fn emit_type_bool(&mut self, id: u32) {
        self.emit(OP_TYPE_BOOL, &[id]);
    }

    fn emit_type_int(&mut self, id: u32, width: u32, signedness: u32) {
        self.emit(OP_TYPE_INT, &[id, width, signedness]);
    }

    fn emit_type_float(&mut self, id: u32, width: u32) {
        self.emit(OP_TYPE_FLOAT, &[id, width]);
    }

    fn emit_type_vector(&mut self, id: u32, component: u32, count: u32) {
        self.emit(OP_TYPE_VECTOR, &[id, component, count]);
    }

    fn emit_type_runtime_array(&mut self, id: u32, element: u32) {
        self.emit(OP_TYPE_RUNTIME_ARRAY, &[id, element]);
    }

    fn emit_type_struct(&mut self, id: u32, members: &[u32]) {
        let mut ops = vec![id];
        ops.extend_from_slice(members);
        self.emit(OP_TYPE_STRUCT, &ops);
    }

    fn emit_type_pointer(&mut self, id: u32, storage_class: u32, pointee: u32) {
        self.emit(OP_TYPE_POINTER, &[id, storage_class, pointee]);
    }

    fn emit_type_function(&mut self, id: u32, return_type: u32, params: &[u32]) {
        let mut ops = vec![id, return_type];
        ops.extend_from_slice(params);
        self.emit(OP_TYPE_FUNCTION, &ops);
    }

    fn emit_constant_u32(&mut self, ty: u32, id: u32, value: u32) {
        self.emit(OP_CONSTANT, &[ty, id, value]);
    }

    fn emit_constant_f32(&mut self, ty: u32, id: u32, value: f32) {
        self.emit(OP_CONSTANT, &[ty, id, value.to_bits()]);
    }

    fn emit_variable(&mut self, ty: u32, id: u32, storage_class: u32) {
        self.emit(OP_VARIABLE, &[ty, id, storage_class]);
    }

    fn emit_load(&mut self, result_ty: u32, result: u32, pointer: u32) {
        self.emit(OP_LOAD, &[result_ty, result, pointer]);
    }

    fn emit_store(&mut self, pointer: u32, value: u32) {
        self.emit(OP_STORE, &[pointer, value]);
    }

    fn emit_access_chain(&mut self, result_ty: u32, result: u32, base: u32, indices: &[u32]) {
        let mut ops = vec![result_ty, result, base];
        ops.extend_from_slice(indices);
        self.emit(OP_ACCESS_CHAIN, &ops);
    }

    fn emit_function(&mut self, result_ty: u32, result: u32, control: u32, fn_ty: u32) {
        self.emit(OP_FUNCTION, &[result_ty, result, control, fn_ty]);
    }

    fn emit_label(&mut self, id: u32) {
        self.emit(OP_LABEL, &[id]);
    }

    fn emit_return(&mut self) {
        self.emit(OP_RETURN, &[]);
    }

    fn emit_function_end(&mut self) {
        self.emit(OP_FUNCTION_END, &[]);
    }

    fn emit_branch(&mut self, target: u32) {
        self.emit(OP_BRANCH, &[target]);
    }

    fn emit_branch_conditional(&mut self, cond: u32, true_label: u32, false_label: u32) {
        self.emit(OP_BRANCH_CONDITIONAL, &[cond, true_label, false_label]);
    }

    fn emit_selection_merge(&mut self, merge_label: u32) {
        self.emit(OP_SELECTION_MERGE, &[merge_label, SELECTION_CONTROL_NONE]);
    }

    fn emit_loop_merge(&mut self, merge_label: u32, continue_label: u32) {
        self.emit(
            OP_LOOP_MERGE,
            &[merge_label, continue_label, LOOP_CONTROL_NONE],
        );
    }

    fn emit_glsl_ext(&mut self, glsl_id: u32, result_ty: u32, result: u32, ext: u32, args: &[u32]) {
        let mut ops = vec![result_ty, result, glsl_id, ext];
        ops.extend_from_slice(args);
        self.emit(OP_EXT_INST, &ops);
    }

    fn emit_type_array(&mut self, id: u32, element: u32, length: u32) {
        self.emit(OP_TYPE_ARRAY, &[id, element, length]);
    }

    fn emit_control_barrier(&mut self, execution: u32, memory: u32, semantics: u32) {
        self.emit(OP_CONTROL_BARRIER, &[execution, memory, semantics]);
    }
}

impl Default for SpvModule {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Common SSBO layout helpers ─────────────────────────────

/// IDs shared by all compute shaders.
struct BaseIds {
    ty_void: u32,
    ty_bool: u32,
    ty_uint: u32,
    ty_float: u32,
    #[allow(dead_code)]
    ty_v3uint: u32,
    ty_fn_void: u32,
    #[allow(dead_code)]
    ty_ptr_input_v3uint: u32,
    ty_ptr_input_uint: u32,
    ty_rt_array_float: u32,
    #[allow(dead_code)]
    ty_rt_array_uint: u32,
    ty_ptr_sb_float: u32,
    ty_ptr_sb_uint: u32,
    ty_ptr_func_float: u32,
    ty_ptr_func_uint: u32,
    c_uint_0: u32,
    c_uint_1: u32,
    c_float_0: u32,
    c_float_1: u32,
    var_gid: u32,
    glsl_ext: u32,
    main_fn: u32,
}

/// Emit the preamble shared by all compute shaders and return [BaseIds].
fn emit_preamble(m: &mut SpvModule) -> BaseIds {
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
    let ty_ptr_func_uint = m.alloc_id();
    let c_uint_0 = m.alloc_id();
    let c_uint_1 = m.alloc_id();
    let c_float_0 = m.alloc_id();
    let c_float_1 = m.alloc_id();
    let var_gid = m.alloc_id();
    let glsl_ext = m.alloc_id();

    m.emit_capability(CAPABILITY_SHADER);
    m.emit_ext_inst_import(glsl_ext, "GLSL.std.450");
    m.emit_memory_model();
    m.emit_entry_point(main_fn, "main", &[var_gid]);
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    m.emit_decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);
    m.emit_decorate(ty_rt_array_float, DECORATION_ARRAY_STRIDE, &[4]);
    m.emit_decorate(ty_rt_array_uint, DECORATION_ARRAY_STRIDE, &[4]);

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
    m.emit_type_pointer(ty_ptr_func_uint, STORAGE_CLASS_FUNCTION, ty_uint);

    m.emit_constant_u32(ty_uint, c_uint_0, 0);
    m.emit_constant_u32(ty_uint, c_uint_1, 1);
    m.emit_constant_f32(ty_float, c_float_0, 0.0);
    m.emit_constant_f32(ty_float, c_float_1, 1.0);

    m.emit_variable(ty_ptr_input_v3uint, var_gid, STORAGE_CLASS_INPUT);

    BaseIds {
        ty_void,
        ty_bool,
        ty_uint,
        ty_float,
        ty_v3uint,
        ty_fn_void,
        ty_ptr_input_v3uint,
        ty_ptr_input_uint,
        ty_rt_array_float,
        ty_rt_array_uint,
        ty_ptr_sb_float,
        ty_ptr_sb_uint,
        ty_ptr_func_float,
        ty_ptr_func_uint,
        c_uint_0,
        c_uint_1,
        c_float_0,
        c_float_1,
        var_gid,
        glsl_ext,
        main_fn,
    }
}

/// Emit a float SSBO and return `(struct_type, ptr_type, variable)`.
fn emit_float_ssbo(m: &mut SpvModule, b: &BaseIds, binding: u32) -> (u32, u32, u32) {
    let struct_ty = m.alloc_id();
    let ptr_ty = m.alloc_id();
    let var = m.alloc_id();

    m.emit_decorate(struct_ty, DECORATION_BLOCK, &[]);
    m.emit_member_decorate(struct_ty, 0, DECORATION_OFFSET, &[0]);
    m.emit_decorate(var, DECORATION_DESCRIPTOR_SET, &[0]);
    m.emit_decorate(var, DECORATION_BINDING, &[binding]);

    m.emit_type_struct(struct_ty, &[b.ty_rt_array_float]);
    m.emit_type_pointer(ptr_ty, STORAGE_CLASS_STORAGE_BUFFER, struct_ty);
    m.emit_variable(ptr_ty, var, STORAGE_CLASS_STORAGE_BUFFER);

    (struct_ty, ptr_ty, var)
}

/// Emit a uint SSBO (for params) and return `(struct_type, ptr_type, variable)`.
fn emit_uint_ssbo(m: &mut SpvModule, b: &BaseIds, binding: u32) -> (u32, u32, u32) {
    let struct_ty = m.alloc_id();
    let ptr_ty = m.alloc_id();
    let var = m.alloc_id();

    m.emit_decorate(struct_ty, DECORATION_BLOCK, &[]);
    m.emit_member_decorate(struct_ty, 0, DECORATION_OFFSET, &[0]);
    m.emit_decorate(var, DECORATION_DESCRIPTOR_SET, &[0]);
    m.emit_decorate(var, DECORATION_BINDING, &[binding]);

    m.emit_type_struct(struct_ty, &[b.ty_rt_array_uint]);
    m.emit_type_pointer(ptr_ty, STORAGE_CLASS_STORAGE_BUFFER, struct_ty);
    m.emit_variable(ptr_ty, var, STORAGE_CLASS_STORAGE_BUFFER);

    (struct_ty, ptr_ty, var)
}

/// Load `GlobalInvocationId.x` into a uint result.
fn load_gid_x(m: &mut SpvModule, b: &BaseIds) -> u32 {
    let ptr = m.alloc_id();
    let gid = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_input_uint, ptr, b.var_gid, &[b.c_uint_0]);
    m.emit_load(b.ty_uint, gid, ptr);
    gid
}

/// Load a uint from params SSBO at constant `idx_const`.
fn load_param_uint(m: &mut SpvModule, b: &BaseIds, params_var: u32, idx_const: u32) -> u32 {
    let ptr = m.alloc_id();
    let val = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_uint, ptr, params_var, &[b.c_uint_0, idx_const]);
    m.emit_load(b.ty_uint, val, ptr);
    val
}

// ─── Unary compute shader ───────────────────────────────────

/// Generate a SPIR-V compute shader for an element-wise unary operation.
///
/// Bindings: 0 = input `float[]`, 1 = output `float[]`, 2 = params `uint[]`
/// where `params[0] = count`.
pub fn unary_compute_shader(op: UnaryOp) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, input_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, output_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, params_var) = emit_uint_ssbo(&mut m, &b, 2);

    let label_entry = m.alloc_id();
    let label_body = m.alloc_id();
    let label_merge = m.alloc_id();

    m.emit_function(b.ty_void, b.main_fn, FUNCTION_CONTROL_NONE, b.ty_fn_void);
    m.emit_label(label_entry);

    let gid = load_gid_x(&mut m, &b);
    let count = load_param_uint(&mut m, &b, params_var, b.c_uint_0);

    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, count]);
    m.emit_selection_merge(label_merge);
    m.emit_branch_conditional(cond, label_body, label_merge);

    m.emit_label(label_body);

    let inp_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, inp_ptr, input_var, &[b.c_uint_0, gid]);
    let inp_val = m.alloc_id();
    m.emit_load(b.ty_float, inp_val, inp_ptr);

    let result = emit_unary_op(&mut m, &b, op, inp_val);

    let out_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, out_ptr, output_var, &[b.c_uint_0, gid]);
    m.emit_store(out_ptr, result);

    m.emit_branch(label_merge);

    m.emit_label(label_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

/// Emit the SPIR-V instructions for a unary operation, returning the result ID.
fn emit_unary_op(m: &mut SpvModule, b: &BaseIds, op: UnaryOp, x: u32) -> u32 {
    let result = m.alloc_id();
    match op {
        UnaryOp::Relu => {
            m.emit_glsl_ext(
                b.glsl_ext,
                b.ty_float,
                result,
                GLSL_F_MAX,
                &[b.c_float_0, x],
            );
        }
        UnaryOp::Sigmoid => {
            let neg_x = m.alloc_id();
            m.emit(OP_F_NEGATE, &[b.ty_float, neg_x, x]);
            let exp_neg_x = m.alloc_id();
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, exp_neg_x, GLSL_EXP, &[neg_x]);
            let one_plus = m.alloc_id();
            m.emit(OP_F_ADD, &[b.ty_float, one_plus, b.c_float_1, exp_neg_x]);
            m.emit(OP_F_DIV, &[b.ty_float, result, b.c_float_1, one_plus]);
        }
        UnaryOp::Tanh => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_TANH, &[x]);
        }
        UnaryOp::Exp => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_EXP, &[x]);
        }
        UnaryOp::Log => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_LOG, &[x]);
        }
        UnaryOp::Sqrt => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_SQRT, &[x]);
        }
        UnaryOp::Abs => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_F_ABS, &[x]);
        }
        UnaryOp::Neg => {
            m.emit(OP_F_NEGATE, &[b.ty_float, result, x]);
        }
    }
    result
}

// ─── Binary compute shader ──────────────────────────────────

/// Generate a SPIR-V compute shader for an element-wise binary operation.
///
/// Bindings: 0 = a `float[]`, 1 = b `float[]`, 2 = output `float[]`,
/// 3 = params `uint[]` where `params[0] = count`.
pub fn binary_compute_shader(op: BinaryOp) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, a_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, b_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, out_var) = emit_float_ssbo(&mut m, &b, 2);
    let (_, _, params_var) = emit_uint_ssbo(&mut m, &b, 3);

    let label_entry = m.alloc_id();
    let label_body = m.alloc_id();
    let label_merge = m.alloc_id();

    m.emit_function(b.ty_void, b.main_fn, FUNCTION_CONTROL_NONE, b.ty_fn_void);
    m.emit_label(label_entry);

    let gid = load_gid_x(&mut m, &b);
    let count = load_param_uint(&mut m, &b, params_var, b.c_uint_0);

    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, count]);
    m.emit_selection_merge(label_merge);
    m.emit_branch_conditional(cond, label_body, label_merge);

    m.emit_label(label_body);

    let a_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, a_ptr, a_var, &[b.c_uint_0, gid]);
    let a_val = m.alloc_id();
    m.emit_load(b.ty_float, a_val, a_ptr);

    let b_ptr_id = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, b_ptr_id, b_var, &[b.c_uint_0, gid]);
    let b_val = m.alloc_id();
    m.emit_load(b.ty_float, b_val, b_ptr_id);

    let result = emit_binary_op(&mut m, &b, op, a_val, b_val);

    let out_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, out_ptr, out_var, &[b.c_uint_0, gid]);
    m.emit_store(out_ptr, result);

    m.emit_branch(label_merge);

    m.emit_label(label_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

fn emit_binary_op(m: &mut SpvModule, b: &BaseIds, op: BinaryOp, lhs: u32, rhs: u32) -> u32 {
    let result = m.alloc_id();
    match op {
        BinaryOp::Add => m.emit(OP_F_ADD, &[b.ty_float, result, lhs, rhs]),
        BinaryOp::Sub => m.emit(OP_F_SUB, &[b.ty_float, result, lhs, rhs]),
        BinaryOp::Mul => m.emit(OP_F_MUL, &[b.ty_float, result, lhs, rhs]),
        BinaryOp::Div => m.emit(OP_F_DIV, &[b.ty_float, result, lhs, rhs]),
        BinaryOp::Max => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_F_MAX, &[lhs, rhs]);
        }
        BinaryOp::Min => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_F_MIN, &[lhs, rhs]);
        }
    }
    result
}

// ─── Reduce compute shader ──────────────────────────────────

/// Generate a SPIR-V compute shader for reduction along an axis.
///
/// Bindings: 0 = input `float[]`, 1 = output `float[]`,
/// 2 = params `uint[]` where params = [outer_size, reduce_size, inner_size].
pub fn reduce_compute_shader(op: ReduceOp) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, input_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, output_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, params_var) = emit_uint_ssbo(&mut m, &b, 2);

    let c_uint_2 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_2, 2);

    let init_val = match op {
        ReduceOp::Sum | ReduceOp::Mean => b.c_float_0,
        ReduceOp::Max => {
            let neg_inf = m.alloc_id();
            m.emit_constant_f32(b.ty_float, neg_inf, f32::NEG_INFINITY);
            neg_inf
        }
        ReduceOp::Min => {
            let pos_inf = m.alloc_id();
            m.emit_constant_f32(b.ty_float, pos_inf, f32::INFINITY);
            pos_inf
        }
    };

    let label_entry = m.alloc_id();
    let label_bounds_body = m.alloc_id();
    let label_bounds_merge = m.alloc_id();
    let label_loop_header = m.alloc_id();
    let label_loop_body = m.alloc_id();
    let label_loop_continue = m.alloc_id();
    let label_loop_merge = m.alloc_id();

    m.emit_function(b.ty_void, b.main_fn, FUNCTION_CONTROL_NONE, b.ty_fn_void);
    m.emit_label(label_entry);

    let gid = load_gid_x(&mut m, &b);

    let outer_size = load_param_uint(&mut m, &b, params_var, b.c_uint_0);
    let reduce_size = load_param_uint(&mut m, &b, params_var, b.c_uint_1);
    let inner_size = load_param_uint(&mut m, &b, params_var, c_uint_2);

    let total_output = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, total_output, outer_size, inner_size]);

    let cond_bounds = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond_bounds, gid, total_output]);
    m.emit_selection_merge(label_bounds_merge);
    m.emit_branch_conditional(cond_bounds, label_bounds_body, label_bounds_merge);

    m.emit_label(label_bounds_body);

    let outer_idx = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, outer_idx, gid, inner_size]);
    let inner_idx = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, inner_idx, gid, inner_size]);

    // base = outer_idx * reduce_size * inner_size + inner_idx
    let t1 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, t1, outer_idx, reduce_size]);
    let t2 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, t2, t1, inner_size]);
    let base_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, base_idx, t2, inner_idx]);

    let var_i = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_i, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_i, b.c_uint_0);

    let var_acc = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_acc, init_val);

    m.emit_branch(label_loop_header);

    m.emit_label(label_loop_header);
    let i_val = m.alloc_id();
    m.emit_load(b.ty_uint, i_val, var_i);
    let loop_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, loop_cond, i_val, reduce_size]);
    m.emit_loop_merge(label_loop_merge, label_loop_continue);
    m.emit_branch_conditional(loop_cond, label_loop_body, label_loop_merge);

    m.emit_label(label_loop_body);

    // input_idx = base_idx + i * inner_size
    let i_times_inner = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, i_times_inner, i_val, inner_size]);
    let input_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, input_idx, base_idx, i_times_inner]);

    let inp_ptr = m.alloc_id();
    m.emit_access_chain(
        b.ty_ptr_sb_float,
        inp_ptr,
        input_var,
        &[b.c_uint_0, input_idx],
    );
    let inp_val = m.alloc_id();
    m.emit_load(b.ty_float, inp_val, inp_ptr);

    let acc_val = m.alloc_id();
    m.emit_load(b.ty_float, acc_val, var_acc);

    let new_acc = m.alloc_id();
    match op {
        ReduceOp::Sum | ReduceOp::Mean => {
            m.emit(OP_F_ADD, &[b.ty_float, new_acc, acc_val, inp_val]);
        }
        ReduceOp::Max => {
            m.emit_glsl_ext(
                b.glsl_ext,
                b.ty_float,
                new_acc,
                GLSL_F_MAX,
                &[acc_val, inp_val],
            );
        }
        ReduceOp::Min => {
            m.emit_glsl_ext(
                b.glsl_ext,
                b.ty_float,
                new_acc,
                GLSL_F_MIN,
                &[acc_val, inp_val],
            );
        }
    }
    m.emit_store(var_acc, new_acc);

    m.emit_branch(label_loop_continue);

    m.emit_label(label_loop_continue);
    let i_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, i_inc, i_val, b.c_uint_1]);
    m.emit_store(var_i, i_inc);
    m.emit_branch(label_loop_header);

    m.emit_label(label_loop_merge);

    let final_acc = m.alloc_id();
    m.emit_load(b.ty_float, final_acc, var_acc);

    let store_val = if op == ReduceOp::Mean {
        let reduce_f = m.alloc_id();
        m.emit(OP_CONVERT_U_TO_F, &[b.ty_float, reduce_f, reduce_size]);
        let mean_val = m.alloc_id();
        m.emit(OP_F_DIV, &[b.ty_float, mean_val, final_acc, reduce_f]);
        mean_val
    } else {
        final_acc
    };

    let out_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, out_ptr, output_var, &[b.c_uint_0, gid]);
    m.emit_store(out_ptr, store_val);

    m.emit_branch(label_bounds_merge);

    m.emit_label(label_bounds_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── GEMM compute shader ────────────────────────────────────

/// Generate a SPIR-V compute shader for GEMM: `C = alpha * A * B + beta * C`.
///
/// Naive reference (one thread per output element, row-major f32 layout).
///
/// Bindings: 0 = A `float[]`, 1 = B `float[]`, 2 = C `float[]`,
/// 3 = params `uint[]` where params = [m, n, k, alpha_bits, beta_bits].
pub fn gemm_compute_shader() -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, a_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, b_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, c_var) = emit_float_ssbo(&mut m, &b, 2);
    let (_, _, params_var) = emit_uint_ssbo(&mut m, &b, 3);

    let c_uint_2 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_2, 2);
    let c_uint_3 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_3, 3);
    let c_uint_4 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_4, 4);

    let label_entry = m.alloc_id();
    let label_bounds_body = m.alloc_id();
    let label_bounds_merge = m.alloc_id();
    let label_loop_header = m.alloc_id();
    let label_loop_body = m.alloc_id();
    let label_loop_continue = m.alloc_id();
    let label_loop_merge = m.alloc_id();

    m.emit_function(b.ty_void, b.main_fn, FUNCTION_CONTROL_NONE, b.ty_fn_void);
    m.emit_label(label_entry);

    let gid = load_gid_x(&mut m, &b);

    let param_m = load_param_uint(&mut m, &b, params_var, b.c_uint_0);
    let param_n = load_param_uint(&mut m, &b, params_var, b.c_uint_1);
    let param_k = load_param_uint(&mut m, &b, params_var, c_uint_2);

    let alpha_u = load_param_uint(&mut m, &b, params_var, c_uint_3);
    let alpha = m.alloc_id();
    m.emit(OP_BITCAST, &[b.ty_float, alpha, alpha_u]);
    let beta_u = load_param_uint(&mut m, &b, params_var, c_uint_4);
    let beta = m.alloc_id();
    m.emit(OP_BITCAST, &[b.ty_float, beta, beta_u]);

    let total = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, total, param_m, param_n]);

    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, total]);
    m.emit_selection_merge(label_bounds_merge);
    m.emit_branch_conditional(cond, label_bounds_body, label_bounds_merge);

    m.emit_label(label_bounds_body);

    let row = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, row, gid, param_n]);
    let col = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, col, gid, param_n]);

    let var_i = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_i, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_i, b.c_uint_0);

    let var_acc = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_acc, b.c_float_0);

    m.emit_branch(label_loop_header);

    m.emit_label(label_loop_header);
    let i_val = m.alloc_id();
    m.emit_load(b.ty_uint, i_val, var_i);
    let loop_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, loop_cond, i_val, param_k]);
    m.emit_loop_merge(label_loop_merge, label_loop_continue);
    m.emit_branch_conditional(loop_cond, label_loop_body, label_loop_merge);

    m.emit_label(label_loop_body);

    // a_idx = row * k + i
    let row_k = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, row_k, row, param_k]);
    let a_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, a_idx, row_k, i_val]);
    // b_idx = i * n + col
    let i_n = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, i_n, i_val, param_n]);
    let b_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, b_idx, i_n, col]);

    let a_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, a_ptr, a_var, &[b.c_uint_0, a_idx]);
    let a_val = m.alloc_id();
    m.emit_load(b.ty_float, a_val, a_ptr);

    let b_ptr_id = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, b_ptr_id, b_var, &[b.c_uint_0, b_idx]);
    let b_val = m.alloc_id();
    m.emit_load(b.ty_float, b_val, b_ptr_id);

    let prod = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, prod, a_val, b_val]);
    let old_acc = m.alloc_id();
    m.emit_load(b.ty_float, old_acc, var_acc);
    let new_acc = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, new_acc, old_acc, prod]);
    m.emit_store(var_acc, new_acc);

    m.emit_branch(label_loop_continue);

    m.emit_label(label_loop_continue);
    let i_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, i_inc, i_val, b.c_uint_1]);
    m.emit_store(var_i, i_inc);
    m.emit_branch(label_loop_header);

    m.emit_label(label_loop_merge);

    // result = alpha * acc + beta * C[gid]
    let final_acc = m.alloc_id();
    m.emit_load(b.ty_float, final_acc, var_acc);
    let alpha_acc = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, alpha_acc, alpha, final_acc]);

    let c_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, c_ptr, c_var, &[b.c_uint_0, gid]);
    let c_old = m.alloc_id();
    m.emit_load(b.ty_float, c_old, c_ptr);
    let beta_c = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, beta_c, beta, c_old]);
    let c_new = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, c_new, alpha_acc, beta_c]);
    m.emit_store(c_ptr, c_new);

    m.emit_branch(label_bounds_merge);

    m.emit_label(label_bounds_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── Batched GEMM compute shader ────────────────────────────

/// Generate a SPIR-V compute shader for strided batched GEMM.
///
/// For each batch index `b` (from `GlobalInvocationId.z`), computes
/// `C_b = alpha * A_b * B_b + beta * C_b` where the batch matrices are
/// offset by `stride_a`, `stride_b`, `stride_c` elements respectively.
///
/// Bindings: 0 = A `float[]`, 1 = B `float[]`, 2 = C `float[]`,
/// 3 = params `uint[]` where:
///   `params[0..5]` = `[m, n, k, alpha(bitcast), beta(bitcast)]`
///   `params[5..8]` = `[stride_a, stride_b, stride_c]`
///
/// Dispatch: `(ceil(m*n / 256), 1, batch_count)`.
pub fn batched_gemm_compute_shader() -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, a_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, b_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, c_var) = emit_float_ssbo(&mut m, &b, 2);
    let (_, _, params_var) = emit_uint_ssbo(&mut m, &b, 3);

    // Additional uint constants for param indices 2..7
    let c_uint_2 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_2, 2);
    let c_uint_3 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_3, 3);
    let c_uint_4 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_4, 4);
    let c_uint_5 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_5, 5);
    let c_uint_6 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_6, 6);
    let c_uint_7 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_7, 7);

    // Labels
    let label_entry = m.alloc_id();
    let label_bounds_body = m.alloc_id();
    let label_bounds_merge = m.alloc_id();
    let label_loop_header = m.alloc_id();
    let label_loop_body = m.alloc_id();
    let label_loop_continue = m.alloc_id();
    let label_loop_merge = m.alloc_id();

    m.emit_function(b.ty_void, b.main_fn, FUNCTION_CONTROL_NONE, b.ty_fn_void);
    m.emit_label(label_entry);

    // Load GlobalInvocationId.x and .z
    let gid_x = load_gid_x(&mut m, &b);
    // Load GlobalInvocationId.z (batch index)
    let gid_z_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_input_uint, gid_z_ptr, b.var_gid, &[c_uint_2]);
    let gid_z = m.alloc_id();
    m.emit_load(b.ty_uint, gid_z, gid_z_ptr);

    // Load params
    let param_m = load_param_uint(&mut m, &b, params_var, b.c_uint_0);
    let param_n = load_param_uint(&mut m, &b, params_var, b.c_uint_1);
    let param_k = load_param_uint(&mut m, &b, params_var, c_uint_2);

    let alpha_u = load_param_uint(&mut m, &b, params_var, c_uint_3);
    let alpha = m.alloc_id();
    m.emit(OP_BITCAST, &[b.ty_float, alpha, alpha_u]);
    let beta_u = load_param_uint(&mut m, &b, params_var, c_uint_4);
    let beta = m.alloc_id();
    m.emit(OP_BITCAST, &[b.ty_float, beta, beta_u]);

    let p_stride_a = load_param_uint(&mut m, &b, params_var, c_uint_5);
    let p_stride_b = load_param_uint(&mut m, &b, params_var, c_uint_6);
    let p_stride_c = load_param_uint(&mut m, &b, params_var, c_uint_7);

    // Bounds check: gid_x < m * n
    let total = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, total, param_m, param_n]);

    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid_x, total]);
    m.emit_selection_merge(label_bounds_merge);
    m.emit_branch_conditional(cond, label_bounds_body, label_bounds_merge);

    m.emit_label(label_bounds_body);

    // Compute batch offsets: base_a = gid_z * stride_a, etc.
    let base_a = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, base_a, gid_z, p_stride_a]);
    let base_b = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, base_b, gid_z, p_stride_b]);
    let base_c = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, base_c, gid_z, p_stride_c]);

    // row = gid_x / n, col = gid_x % n
    let row = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, row, gid_x, param_n]);
    let col = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, col, gid_x, param_n]);

    // Loop variable and accumulator
    let var_i = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_i, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_i, b.c_uint_0);

    let var_acc = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_acc, b.c_float_0);

    m.emit_branch(label_loop_header);

    // Loop header
    m.emit_label(label_loop_header);
    let i_val = m.alloc_id();
    m.emit_load(b.ty_uint, i_val, var_i);
    let loop_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, loop_cond, i_val, param_k]);
    m.emit_loop_merge(label_loop_merge, label_loop_continue);
    m.emit_branch_conditional(loop_cond, label_loop_body, label_loop_merge);

    // Loop body
    m.emit_label(label_loop_body);

    // a_idx = base_a + row * k + i
    let row_k = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, row_k, row, param_k]);
    let a_local = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, a_local, row_k, i_val]);
    let a_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, a_idx, base_a, a_local]);

    // b_idx = base_b + i * n + col
    let i_n = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, i_n, i_val, param_n]);
    let b_local = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, b_local, i_n, col]);
    let b_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, b_idx, base_b, b_local]);

    let a_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, a_ptr, a_var, &[b.c_uint_0, a_idx]);
    let a_val = m.alloc_id();
    m.emit_load(b.ty_float, a_val, a_ptr);

    let b_ptr_id = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, b_ptr_id, b_var, &[b.c_uint_0, b_idx]);
    let b_val = m.alloc_id();
    m.emit_load(b.ty_float, b_val, b_ptr_id);

    let prod = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, prod, a_val, b_val]);
    let old_acc = m.alloc_id();
    m.emit_load(b.ty_float, old_acc, var_acc);
    let new_acc = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, new_acc, old_acc, prod]);
    m.emit_store(var_acc, new_acc);

    m.emit_branch(label_loop_continue);

    // Loop continue
    m.emit_label(label_loop_continue);
    let i_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, i_inc, i_val, b.c_uint_1]);
    m.emit_store(var_i, i_inc);
    m.emit_branch(label_loop_header);

    // Loop merge — compute result = alpha * acc + beta * C[c_idx]
    m.emit_label(label_loop_merge);

    let final_acc = m.alloc_id();
    m.emit_load(b.ty_float, final_acc, var_acc);
    let alpha_acc = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, alpha_acc, alpha, final_acc]);

    // c_idx = base_c + gid_x
    let c_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, c_idx, base_c, gid_x]);

    let c_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, c_ptr, c_var, &[b.c_uint_0, c_idx]);
    let c_old = m.alloc_id();
    m.emit_load(b.ty_float, c_old, c_ptr);
    let beta_c = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, beta_c, beta, c_old]);
    let c_new = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, c_new, alpha_acc, beta_c]);
    m.emit_store(c_ptr, c_new);

    m.emit_branch(label_bounds_merge);

    m.emit_label(label_bounds_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── Conv2D compute shader ──────────────────────────────────

/// Generate a SPIR-V compute shader for 2-D convolution (NCHW layout).
///
/// One invocation per output element.  Triple-nested accumulation loop over
/// `(in_channel, filter_y, filter_x)` with unsigned-arithmetic padding checks.
///
/// Bindings: 0 = input `float[]`, 1 = filter `float[]`, 2 = output `float[]`.
/// All dimension constants are baked into the SPIR-V binary.
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
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, input_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, filter_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, output_var) = emit_float_ssbo(&mut m, &b, 2);

    // Baked constants
    let c_cin = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_cin, c_in);
    let c_hin = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_hin, h_in);
    let c_win = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_win, w_in);
    let c_kout = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_kout, k_out);
    let c_fh = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_fh, fh);
    let c_fw = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_fw, fw);
    let c_oh = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_oh, oh);
    let c_ow = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_ow, ow);
    let c_sh = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_sh, stride_h);
    let c_sw = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_sw, stride_w);
    let c_ph = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_ph, pad_h);
    let c_pw = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_pw, pad_w);
    let total = n
        .saturating_mul(k_out)
        .saturating_mul(oh)
        .saturating_mul(ow);
    let c_total = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_total, total);

    // Labels
    let lbl_entry = m.alloc_id();
    let lbl_body = m.alloc_id();
    let lbl_merge = m.alloc_id();
    let lbl_ci_h = m.alloc_id();
    let lbl_ci_b = m.alloc_id();
    let lbl_ci_c = m.alloc_id();
    let lbl_ci_m = m.alloc_id();
    let lbl_fy_h = m.alloc_id();
    let lbl_fy_b = m.alloc_id();
    let lbl_fy_c = m.alloc_id();
    let lbl_fy_m = m.alloc_id();
    let lbl_fx_h = m.alloc_id();
    let lbl_fx_b = m.alloc_id();
    let lbl_fx_c = m.alloc_id();
    let lbl_fx_m = m.alloc_id();
    let lbl_ib = m.alloc_id();
    let lbl_ib_m = m.alloc_id();

    // ── Function ──
    m.emit_function(b.ty_void, b.main_fn, FUNCTION_CONTROL_NONE, b.ty_fn_void);
    m.emit_label(lbl_entry);

    let gid = load_gid_x(&mut m, &b);
    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, c_total]);
    m.emit_selection_merge(lbl_merge);
    m.emit_branch_conditional(cond, lbl_body, lbl_merge);

    m.emit_label(lbl_body);

    // Decompose gid → (batch, kf, oy, ox)
    let ox_val = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, ox_val, gid, c_ow]);
    let t1 = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, t1, gid, c_ow]);
    let oy_val = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, oy_val, t1, c_oh]);
    let t2 = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, t2, t1, c_oh]);
    let kf = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, kf, t2, c_kout]);
    let batch = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, batch, t2, c_kout]);

    // Local variables (Function-scope, in first block after function start)
    let var_acc = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_acc, b.c_float_0);
    let var_ci = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_ci, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_ci, b.c_uint_0);
    let var_fy = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_fy, STORAGE_CLASS_FUNCTION);
    let var_fx = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_fx, STORAGE_CLASS_FUNCTION);

    m.emit_branch(lbl_ci_h);

    // ── ci loop ──
    m.emit_label(lbl_ci_h);
    let ci = m.alloc_id();
    m.emit_load(b.ty_uint, ci, var_ci);
    let ci_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, ci_ok, ci, c_cin]);
    m.emit_loop_merge(lbl_ci_m, lbl_ci_c);
    m.emit_branch_conditional(ci_ok, lbl_ci_b, lbl_ci_m);

    m.emit_label(lbl_ci_b);
    m.emit_store(var_fy, b.c_uint_0);
    m.emit_branch(lbl_fy_h);

    // ── fy loop ──
    m.emit_label(lbl_fy_h);
    let fy = m.alloc_id();
    m.emit_load(b.ty_uint, fy, var_fy);
    let fy_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, fy_ok, fy, c_fh]);
    m.emit_loop_merge(lbl_fy_m, lbl_fy_c);
    m.emit_branch_conditional(fy_ok, lbl_fy_b, lbl_fy_m);

    m.emit_label(lbl_fy_b);
    m.emit_store(var_fx, b.c_uint_0);
    m.emit_branch(lbl_fx_h);

    // ── fx loop ──
    m.emit_label(lbl_fx_h);
    let fx = m.alloc_id();
    m.emit_load(b.ty_uint, fx, var_fx);
    let fx_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, fx_ok, fx, c_fw]);
    m.emit_loop_merge(lbl_fx_m, lbl_fx_c);
    m.emit_branch_conditional(fx_ok, lbl_fx_b, lbl_fx_m);

    m.emit_label(lbl_fx_b);

    // iy = oy*stride_h + fy − pad_h (unsigned wrapping)
    let oy_sh = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, oy_sh, oy_val, c_sh]);
    let oy_sh_fy = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, oy_sh_fy, oy_sh, fy]);
    let iy = m.alloc_id();
    m.emit(OP_I_SUB, &[b.ty_uint, iy, oy_sh_fy, c_ph]);

    // ix = ox*stride_w + fx − pad_w
    let ox_sw = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, ox_sw, ox_val, c_sw]);
    let ox_sw_fx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ox_sw_fx, ox_sw, fx]);
    let ix = m.alloc_id();
    m.emit(OP_I_SUB, &[b.ty_uint, ix, ox_sw_fx, c_pw]);

    // Bounds: iy < h_in && ix < w_in (unsigned catches underflow)
    let iy_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, iy_ok, iy, c_hin]);
    let ix_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, ix_ok, ix, c_win]);
    let ok = m.alloc_id();
    m.emit(OP_LOGICAL_AND, &[b.ty_bool, ok, iy_ok, ix_ok]);

    m.emit_selection_merge(lbl_ib_m);
    m.emit_branch_conditional(ok, lbl_ib, lbl_ib_m);

    m.emit_label(lbl_ib);

    // input_idx = ((batch*c_in + ci)*h_in + iy)*w_in + ix
    let bc = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, bc, batch, c_cin]);
    let bc_ci = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, bc_ci, bc, ci]);
    let bch = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, bch, bc_ci, c_hin]);
    let bch_iy = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, bch_iy, bch, iy]);
    let bchw = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, bchw, bch_iy, c_win]);
    let in_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, in_idx, bchw, ix]);

    // filter_idx = ((kf*c_in + ci)*fh + fy)*fw + fx
    let kc = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, kc, kf, c_cin]);
    let kc_ci = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, kc_ci, kc, ci]);
    let kcf = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, kcf, kc_ci, c_fh]);
    let kcf_fy = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, kcf_fy, kcf, fy]);
    let kcff = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, kcff, kcf_fy, c_fw]);
    let f_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, f_idx, kcff, fx]);

    // Load input and filter, accumulate
    let inp_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, inp_ptr, input_var, &[b.c_uint_0, in_idx]);
    let inp_v = m.alloc_id();
    m.emit_load(b.ty_float, inp_v, inp_ptr);
    let flt_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, flt_ptr, filter_var, &[b.c_uint_0, f_idx]);
    let flt_v = m.alloc_id();
    m.emit_load(b.ty_float, flt_v, flt_ptr);
    let prod = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, prod, inp_v, flt_v]);
    let old_acc = m.alloc_id();
    m.emit_load(b.ty_float, old_acc, var_acc);
    let new_acc = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, new_acc, old_acc, prod]);
    m.emit_store(var_acc, new_acc);

    m.emit_branch(lbl_ib_m);
    m.emit_label(lbl_ib_m);

    // fx continue / merge
    m.emit_branch(lbl_fx_c);
    m.emit_label(lbl_fx_c);
    let fx_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, fx_inc, fx, b.c_uint_1]);
    m.emit_store(var_fx, fx_inc);
    m.emit_branch(lbl_fx_h);
    m.emit_label(lbl_fx_m);

    // fy continue / merge
    m.emit_branch(lbl_fy_c);
    m.emit_label(lbl_fy_c);
    let fy_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, fy_inc, fy, b.c_uint_1]);
    m.emit_store(var_fy, fy_inc);
    m.emit_branch(lbl_fy_h);
    m.emit_label(lbl_fy_m);

    // ci continue / merge
    m.emit_branch(lbl_ci_c);
    m.emit_label(lbl_ci_c);
    let ci_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ci_inc, ci, b.c_uint_1]);
    m.emit_store(var_ci, ci_inc);
    m.emit_branch(lbl_ci_h);

    m.emit_label(lbl_ci_m);

    // Store result
    let final_acc = m.alloc_id();
    m.emit_load(b.ty_float, final_acc, var_acc);
    let out_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, out_ptr, output_var, &[b.c_uint_0, gid]);
    m.emit_store(out_ptr, final_acc);

    m.emit_branch(lbl_merge);
    m.emit_label(lbl_merge);
    m.emit_return();
    m.emit_function_end();
    m.finalize()
}

// ─── Attention compute shader ───────────────────────────────

/// Generate a SPIR-V compute shader for scaled dot-product attention.
///
/// Each invocation handles one `(batch_head, query_position)`.  The shader
/// performs numerically-stable softmax with optional causal masking.
///
/// Bindings: 0 = Q `float[]`, 1 = K `float[]`, 2 = V `float[]`,
/// 3 = O `float[]`.  All dimension constants are baked in.
pub fn attention_spirv(
    batch_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    head_dim: u32,
    scale: f32,
    causal: bool,
) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, q_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, k_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, v_var) = emit_float_ssbo(&mut m, &b, 2);
    let (_, _, o_var) = emit_float_ssbo(&mut m, &b, 3);

    // Baked constants
    let c_sq = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_sq, seq_q);
    let c_skv = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_skv, seq_kv);
    let c_hd = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_hd, head_dim);
    let c_scale = m.alloc_id();
    m.emit_constant_f32(b.ty_float, c_scale, scale);
    let c_neg_inf = m.alloc_id();
    m.emit_constant_f32(b.ty_float, c_neg_inf, f32::NEG_INFINITY);
    let c_total = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_total, batch_heads.saturating_mul(seq_q));

    // ── Labels ──
    let lbl_entry = m.alloc_id();
    let lbl_body = m.alloc_id();
    let lbl_merge = m.alloc_id();
    // Pass 1 labels (max-score)
    let lbl_s1h = m.alloc_id();
    let lbl_s1b = m.alloc_id();
    let lbl_s1c = m.alloc_id();
    let lbl_s1m = m.alloc_id();
    let lbl_s1w = m.alloc_id();
    let lbl_s1wm = m.alloc_id();
    let lbl_d1h = m.alloc_id();
    let lbl_d1b = m.alloc_id();
    let lbl_d1c = m.alloc_id();
    let lbl_d1m = m.alloc_id();
    // Zero-output labels
    let lbl_zh = m.alloc_id();
    let lbl_zb = m.alloc_id();
    let lbl_zc = m.alloc_id();
    let lbl_zm = m.alloc_id();
    // Pass 2 labels (accumulate)
    let lbl_s2h = m.alloc_id();
    let lbl_s2b = m.alloc_id();
    let lbl_s2c = m.alloc_id();
    let lbl_s2m = m.alloc_id();
    let lbl_s2w = m.alloc_id();
    let lbl_s2wm = m.alloc_id();
    let lbl_d2h = m.alloc_id();
    let lbl_d2b = m.alloc_id();
    let lbl_d2c = m.alloc_id();
    let lbl_d2m = m.alloc_id();
    let lbl_d3h = m.alloc_id();
    let lbl_d3b = m.alloc_id();
    let lbl_d3c = m.alloc_id();
    let lbl_d3m = m.alloc_id();
    // Normalize labels
    let lbl_d4h = m.alloc_id();
    let lbl_d4b = m.alloc_id();
    let lbl_d4c = m.alloc_id();
    let lbl_d4m = m.alloc_id();

    // ── Function ──
    m.emit_function(b.ty_void, b.main_fn, FUNCTION_CONTROL_NONE, b.ty_fn_void);
    m.emit_label(lbl_entry);

    let gid = load_gid_x(&mut m, &b);
    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, c_total]);
    m.emit_selection_merge(lbl_merge);
    m.emit_branch_conditional(cond, lbl_body, lbl_merge);

    m.emit_label(lbl_body);

    // bh = gid / seq_q, sq_val = gid % seq_q
    let bh = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, bh, gid, c_sq]);
    let sq_val = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, sq_val, gid, c_sq]);
    // q_base = o_base = gid * head_dim
    let q_base = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, q_base, gid, c_hd]);
    // bh_skv = bh * seq_kv  (shared prefix for k/v base)
    let bh_skv = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, bh_skv, bh, c_skv]);

    // Local variables
    let var_max = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_max, STORAGE_CLASS_FUNCTION);
    let var_sum = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_sum, STORAGE_CLASS_FUNCTION);
    let var_dot = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_dot, STORAGE_CLASS_FUNCTION);
    let var_sk = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_sk, STORAGE_CLASS_FUNCTION);
    let var_d = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_d, STORAGE_CLASS_FUNCTION);

    // ── Pass 1: find max score ──
    m.emit_store(var_max, c_neg_inf);
    m.emit_store(var_sk, b.c_uint_0);
    m.emit_branch(lbl_s1h);

    m.emit_label(lbl_s1h);
    let sk1 = m.alloc_id();
    m.emit_load(b.ty_uint, sk1, var_sk);
    let s1_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, s1_ok, sk1, c_skv]);
    m.emit_loop_merge(lbl_s1m, lbl_s1c);
    m.emit_branch_conditional(s1_ok, lbl_s1b, lbl_s1m);

    m.emit_label(lbl_s1b);
    // Causal check: skip if sq_val < sk1
    if causal {
        let skip = m.alloc_id();
        m.emit(OP_U_LESS_THAN, &[b.ty_bool, skip, sq_val, sk1]);
        m.emit_selection_merge(lbl_s1wm);
        m.emit_branch_conditional(skip, lbl_s1wm, lbl_s1w);
    } else {
        m.emit_branch(lbl_s1w);
    }

    m.emit_label(lbl_s1w);
    // k_base = (bh_skv + sk1) * head_dim
    let ks1 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ks1, bh_skv, sk1]);
    let kb1 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, kb1, ks1, c_hd]);
    // dot product loop
    m.emit_store(var_dot, b.c_float_0);
    m.emit_store(var_d, b.c_uint_0);
    m.emit_branch(lbl_d1h);

    m.emit_label(lbl_d1h);
    let d1 = m.alloc_id();
    m.emit_load(b.ty_uint, d1, var_d);
    let d1_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, d1_ok, d1, c_hd]);
    m.emit_loop_merge(lbl_d1m, lbl_d1c);
    m.emit_branch_conditional(d1_ok, lbl_d1b, lbl_d1m);

    m.emit_label(lbl_d1b);
    let qi1 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, qi1, q_base, d1]);
    let qp1 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, qp1, q_var, &[b.c_uint_0, qi1]);
    let qv1 = m.alloc_id();
    m.emit_load(b.ty_float, qv1, qp1);
    let ki1 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ki1, kb1, d1]);
    let kp1 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, kp1, k_var, &[b.c_uint_0, ki1]);
    let kv1 = m.alloc_id();
    m.emit_load(b.ty_float, kv1, kp1);
    let p1 = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, p1, qv1, kv1]);
    let od1 = m.alloc_id();
    m.emit_load(b.ty_float, od1, var_dot);
    let nd1 = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, nd1, od1, p1]);
    m.emit_store(var_dot, nd1);
    m.emit_branch(lbl_d1c);

    m.emit_label(lbl_d1c);
    let d1i = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, d1i, d1, b.c_uint_1]);
    m.emit_store(var_d, d1i);
    m.emit_branch(lbl_d1h);

    m.emit_label(lbl_d1m);
    // score = dot * scale; max_score = fmax(max_score, score)
    let dot1 = m.alloc_id();
    m.emit_load(b.ty_float, dot1, var_dot);
    let scr1 = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, scr1, dot1, c_scale]);
    let om1 = m.alloc_id();
    m.emit_load(b.ty_float, om1, var_max);
    let nm1 = m.alloc_id();
    m.emit_glsl_ext(b.glsl_ext, b.ty_float, nm1, GLSL_F_MAX, &[om1, scr1]);
    m.emit_store(var_max, nm1);

    m.emit_branch(lbl_s1wm);
    m.emit_label(lbl_s1wm);
    m.emit_branch(lbl_s1c);

    m.emit_label(lbl_s1c);
    let sk1i = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, sk1i, sk1, b.c_uint_1]);
    m.emit_store(var_sk, sk1i);
    m.emit_branch(lbl_s1h);

    m.emit_label(lbl_s1m);

    // ── Zero output ──
    m.emit_store(var_d, b.c_uint_0);
    m.emit_branch(lbl_zh);

    m.emit_label(lbl_zh);
    let zd = m.alloc_id();
    m.emit_load(b.ty_uint, zd, var_d);
    let zd_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, zd_ok, zd, c_hd]);
    m.emit_loop_merge(lbl_zm, lbl_zc);
    m.emit_branch_conditional(zd_ok, lbl_zb, lbl_zm);

    m.emit_label(lbl_zb);
    let ozi = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ozi, q_base, zd]); // o_base == q_base
    let ozp = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, ozp, o_var, &[b.c_uint_0, ozi]);
    m.emit_store(ozp, b.c_float_0);
    m.emit_branch(lbl_zc);

    m.emit_label(lbl_zc);
    let zdi = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, zdi, zd, b.c_uint_1]);
    m.emit_store(var_d, zdi);
    m.emit_branch(lbl_zh);

    m.emit_label(lbl_zm);

    // ── Pass 2: accumulate weighted V ──
    m.emit_store(var_sum, b.c_float_0);
    m.emit_store(var_sk, b.c_uint_0);
    m.emit_branch(lbl_s2h);

    m.emit_label(lbl_s2h);
    let sk2 = m.alloc_id();
    m.emit_load(b.ty_uint, sk2, var_sk);
    let s2_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, s2_ok, sk2, c_skv]);
    m.emit_loop_merge(lbl_s2m, lbl_s2c);
    m.emit_branch_conditional(s2_ok, lbl_s2b, lbl_s2m);

    m.emit_label(lbl_s2b);
    if causal {
        let skip2 = m.alloc_id();
        m.emit(OP_U_LESS_THAN, &[b.ty_bool, skip2, sq_val, sk2]);
        m.emit_selection_merge(lbl_s2wm);
        m.emit_branch_conditional(skip2, lbl_s2wm, lbl_s2w);
    } else {
        m.emit_branch(lbl_s2w);
    }

    m.emit_label(lbl_s2w);
    // k/v base
    let ks2 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ks2, bh_skv, sk2]);
    let kb2 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, kb2, ks2, c_hd]);
    // dot product d2 loop
    m.emit_store(var_dot, b.c_float_0);
    m.emit_store(var_d, b.c_uint_0);
    m.emit_branch(lbl_d2h);

    m.emit_label(lbl_d2h);
    let d2 = m.alloc_id();
    m.emit_load(b.ty_uint, d2, var_d);
    let d2_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, d2_ok, d2, c_hd]);
    m.emit_loop_merge(lbl_d2m, lbl_d2c);
    m.emit_branch_conditional(d2_ok, lbl_d2b, lbl_d2m);

    m.emit_label(lbl_d2b);
    let qi2 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, qi2, q_base, d2]);
    let qp2 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, qp2, q_var, &[b.c_uint_0, qi2]);
    let qv2 = m.alloc_id();
    m.emit_load(b.ty_float, qv2, qp2);
    let ki2 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ki2, kb2, d2]);
    let kp2 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, kp2, k_var, &[b.c_uint_0, ki2]);
    let kv2 = m.alloc_id();
    m.emit_load(b.ty_float, kv2, kp2);
    let p2 = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, p2, qv2, kv2]);
    let od2 = m.alloc_id();
    m.emit_load(b.ty_float, od2, var_dot);
    let nd2 = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, nd2, od2, p2]);
    m.emit_store(var_dot, nd2);
    m.emit_branch(lbl_d2c);

    m.emit_label(lbl_d2c);
    let d2i = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, d2i, d2, b.c_uint_1]);
    m.emit_store(var_d, d2i);
    m.emit_branch(lbl_d2h);

    m.emit_label(lbl_d2m);
    // w = exp(dot*scale − max_score); sum_exp += w
    let dot2 = m.alloc_id();
    m.emit_load(b.ty_float, dot2, var_dot);
    let scr2 = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, scr2, dot2, c_scale]);
    let mx2 = m.alloc_id();
    m.emit_load(b.ty_float, mx2, var_max);
    let diff = m.alloc_id();
    m.emit(OP_F_SUB, &[b.ty_float, diff, scr2, mx2]);
    let w = m.alloc_id();
    m.emit_glsl_ext(b.glsl_ext, b.ty_float, w, GLSL_EXP, &[diff]);
    let os2 = m.alloc_id();
    m.emit_load(b.ty_float, os2, var_sum);
    let ns2 = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, ns2, os2, w]);
    m.emit_store(var_sum, ns2);

    // V accumulation d3 loop: o[o_base+d] += w * v[kb2+d]
    m.emit_store(var_d, b.c_uint_0);
    m.emit_branch(lbl_d3h);

    m.emit_label(lbl_d3h);
    let d3 = m.alloc_id();
    m.emit_load(b.ty_uint, d3, var_d);
    let d3_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, d3_ok, d3, c_hd]);
    m.emit_loop_merge(lbl_d3m, lbl_d3c);
    m.emit_branch_conditional(d3_ok, lbl_d3b, lbl_d3m);

    m.emit_label(lbl_d3b);
    let vi3 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, vi3, kb2, d3]);
    let vp3 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, vp3, v_var, &[b.c_uint_0, vi3]);
    let vv3 = m.alloc_id();
    m.emit_load(b.ty_float, vv3, vp3);
    let wv3 = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, wv3, w, vv3]);
    let oi3 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, oi3, q_base, d3]);
    let op3 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, op3, o_var, &[b.c_uint_0, oi3]);
    let ov3 = m.alloc_id();
    m.emit_load(b.ty_float, ov3, op3);
    let nv3 = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, nv3, ov3, wv3]);
    m.emit_store(op3, nv3);
    m.emit_branch(lbl_d3c);

    m.emit_label(lbl_d3c);
    let d3i = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, d3i, d3, b.c_uint_1]);
    m.emit_store(var_d, d3i);
    m.emit_branch(lbl_d3h);

    m.emit_label(lbl_d3m);
    m.emit_branch(lbl_s2wm);

    m.emit_label(lbl_s2wm);
    m.emit_branch(lbl_s2c);

    m.emit_label(lbl_s2c);
    let sk2i = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, sk2i, sk2, b.c_uint_1]);
    m.emit_store(var_sk, sk2i);
    m.emit_branch(lbl_s2h);

    m.emit_label(lbl_s2m);

    // ── Normalize: o[o_base+d] /= sum_exp ──
    let final_sum = m.alloc_id();
    m.emit_load(b.ty_float, final_sum, var_sum);
    m.emit_store(var_d, b.c_uint_0);
    m.emit_branch(lbl_d4h);

    m.emit_label(lbl_d4h);
    let d4 = m.alloc_id();
    m.emit_load(b.ty_uint, d4, var_d);
    let d4_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, d4_ok, d4, c_hd]);
    m.emit_loop_merge(lbl_d4m, lbl_d4c);
    m.emit_branch_conditional(d4_ok, lbl_d4b, lbl_d4m);

    m.emit_label(lbl_d4b);
    let oi4 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, oi4, q_base, d4]);
    let op4 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, op4, o_var, &[b.c_uint_0, oi4]);
    let ov4 = m.alloc_id();
    m.emit_load(b.ty_float, ov4, op4);
    let nv4 = m.alloc_id();
    m.emit(OP_F_DIV, &[b.ty_float, nv4, ov4, final_sum]);
    m.emit_store(op4, nv4);
    m.emit_branch(lbl_d4c);

    m.emit_label(lbl_d4c);
    let d4i = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, d4i, d4, b.c_uint_1]);
    m.emit_store(var_d, d4i);
    m.emit_branch(lbl_d4h);

    m.emit_label(lbl_d4m);
    m.emit_branch(lbl_merge);

    m.emit_label(lbl_merge);
    m.emit_return();
    m.emit_function_end();
    m.finalize()
}

// ─── Subgroup-optimized shaders (separate file) ─────────────

#[path = "spirv_subgroup.rs"]
mod subgroup;
pub use subgroup::{reduction_subgroup_spirv, scan_subgroup_spirv};

// ─── Trivial placeholder ────────────────────────────────────

/// Build a minimal valid compute shader: `void main() {}` with `LocalSize(1,1,1)`.
pub fn trivial_compute_shader() -> Vec<u32> {
    let mut m = SpvModule::new();

    let id_main_fn = m.alloc_id();
    let id_void = m.alloc_id();
    let id_void_fn = m.alloc_id();
    let id_label = m.alloc_id();

    m.emit_capability(CAPABILITY_SHADER);
    m.emit_memory_model();

    let mut entry_words = vec![EXECUTION_MODEL_GLCOMPUTE, id_main_fn];
    entry_words.extend(SpvModule::string_words("main"));
    m.emit(OP_ENTRY_POINT, &entry_words);

    m.emit_execution_mode_local_size(id_main_fn, 1, 1, 1);

    m.emit_type_void(id_void);
    m.emit_type_function(id_void_fn, id_void, &[]);

    m.emit_function(id_void, id_main_fn, FUNCTION_CONTROL_NONE, id_void_fn);
    m.emit_label(id_label);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

/// Return the trivial compute shader as a byte slice.
pub fn trivial_compute_shader_bytes() -> Vec<u8> {
    trivial_compute_shader()
        .iter()
        .flat_map(|w| w.to_ne_bytes())
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_spv_valid_magic() {
        let words = trivial_compute_shader();
        assert!(!words.is_empty());
        assert_eq!(words[0], SPIRV_MAGIC);
    }

    #[test]
    fn placeholder_spv_word_aligned() {
        let bytes = trivial_compute_shader_bytes();
        assert_eq!(bytes.len() % 4, 0);
    }

    #[test]
    fn placeholder_spv_version_and_schema() {
        let words = trivial_compute_shader();
        assert!(words.len() >= 5);
        assert!(words[1] >= 0x0001_0000);
        assert_eq!(words[4], 0);
    }

    #[test]
    fn placeholder_spv_nonzero_bound() {
        let words = trivial_compute_shader();
        assert!(words[3] > 0);
    }

    #[test]
    fn spv_module_id_allocation_is_monotonic() {
        let mut m = SpvModule::new();
        let id1 = m.alloc_id();
        let id2 = m.alloc_id();
        assert!(id2 > id1);
    }

    #[test]
    fn string_words_null_terminated() {
        let words = SpvModule::string_words("abc");
        assert!(!words.is_empty());
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(bytes[0], b'a');
        assert_eq!(bytes[1], b'b');
        assert_eq!(bytes[2], b'c');
        assert_eq!(bytes[3], 0);
    }

    #[test]
    fn string_words_empty_string() {
        let words = SpvModule::string_words("");
        assert!(!words.is_empty());
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(bytes[0], 0);
    }

    // ── Compute shader generation ────────────────────────────

    fn check_valid_spirv(words: &[u32]) {
        assert!(words.len() >= 5, "too short for SPIR-V header");
        assert_eq!(words[0], SPIRV_MAGIC, "bad magic");
        assert!(words[3] > 0, "ID bound must be > 0");
        assert_eq!(words[4], 0, "schema must be 0");
    }

    #[test]
    fn unary_shader_all_ops() {
        let ops = [
            UnaryOp::Relu,
            UnaryOp::Sigmoid,
            UnaryOp::Tanh,
            UnaryOp::Exp,
            UnaryOp::Log,
            UnaryOp::Sqrt,
            UnaryOp::Abs,
            UnaryOp::Neg,
        ];
        for op in ops {
            let words = unary_compute_shader(op);
            check_valid_spirv(&words);
            assert_eq!(words[1], SPIRV_VERSION_1_3, "op {op:?} wrong version");
        }
    }

    #[test]
    fn binary_shader_all_ops() {
        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Max,
            BinaryOp::Min,
        ];
        for op in ops {
            let words = binary_compute_shader(op);
            check_valid_spirv(&words);
        }
    }

    #[test]
    fn reduce_shader_all_ops() {
        let ops = [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min, ReduceOp::Mean];
        for op in ops {
            let words = reduce_compute_shader(op);
            check_valid_spirv(&words);
        }
    }

    #[test]
    fn gemm_shader_valid() {
        let words = gemm_compute_shader();
        check_valid_spirv(&words);
        assert_eq!(words[1], SPIRV_VERSION_1_3);
    }

    #[test]
    fn batched_gemm_shader_valid() {
        let words = batched_gemm_compute_shader();
        check_valid_spirv(&words);
        assert_eq!(words[1], SPIRV_VERSION_1_3);
        // Batched GEMM shader must be larger than regular GEMM (extra batch logic).
        let gemm_words = gemm_compute_shader();
        assert!(
            words.len() > gemm_words.len(),
            "batched_gemm ({}) should be larger than gemm ({})",
            words.len(),
            gemm_words.len()
        );
    }

    #[test]
    fn batched_gemm_shader_contains_expected_structure() {
        let words = batched_gemm_compute_shader();
        // Magic number is correct.
        assert_eq!(words[0], 0x07230203);
        // ID bound is positive.
        assert!(words[3] > 0);
        // Schema is 0.
        assert_eq!(words[4], 0);
        // Must contain at least the capability, memory model, and entry point.
        assert!(words.len() > 50, "shader too small: {}", words.len());
    }

    #[test]
    fn conv2d_shader_valid() {
        // 1×1 identity-style convolution
        let words = conv2d_spirv(1, 1, 4, 4, 1, 1, 1, 4, 4, 1, 1, 0, 0);
        check_valid_spirv(&words);
        assert_eq!(words[1], SPIRV_VERSION_1_3);
        assert!(
            words.len() > 100,
            "conv2d shader too small: {}",
            words.len()
        );
    }

    #[test]
    fn conv2d_shader_with_padding() {
        let words = conv2d_spirv(2, 3, 8, 8, 16, 3, 3, 8, 8, 1, 1, 1, 1);
        check_valid_spirv(&words);
        assert!(words.len() > 100);
    }

    #[test]
    fn attention_shader_valid() {
        let words = attention_spirv(2, 4, 4, 8, 0.125, false);
        check_valid_spirv(&words);
        assert_eq!(words[1], SPIRV_VERSION_1_3);
        assert!(
            words.len() > 100,
            "attention shader too small: {}",
            words.len()
        );
    }

    #[test]
    fn attention_shader_causal() {
        let words = attention_spirv(4, 16, 16, 64, 0.125, true);
        check_valid_spirv(&words);
        assert!(words.len() > 100);
        // Causal shader should be larger (extra branching)
        let non_causal = attention_spirv(4, 16, 16, 64, 0.125, false);
        assert!(
            words.len() > non_causal.len(),
            "causal {} should be larger than non-causal {}",
            words.len(),
            non_causal.len()
        );
    }
}
