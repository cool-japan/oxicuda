//! SPIR-V compute kernel generators for the Level Zero backend.
//!
//! This module provides:
//! - A lightweight [`SpvModule`] builder for emitting valid SPIR-V binaries.
//! - Generator functions for **unary**, **binary**, **reduce**, and **GEMM**
//!   compute kernels consumed by Level Zero's `zeModuleCreate`.
//! - The original [`trivial_compute_shader`] placeholder used for validation.
//!
//! All generated kernels use the OpenCL SPIR-V execution model (`Kernel`)
//! with `Physical64`/`OpenCL` memory model.  Buffer parameters are
//! `CrossWorkgroup` pointers; scalar parameters are passed by value via
//! `zeKernelSetArgumentValue`.
//!
//! The generated SPIR-V targets version 1.2 (widely supported by Level Zero).

use oxicuda_backend::{BinaryOp, ReduceOp, UnaryOp};

// ─── Constants ──────────────────────────────────────────────

/// SPIR-V magic number.
pub const SPIRV_MAGIC: u32 = 0x07230203;
/// SPIR-V version 1.2.
pub const SPIRV_VERSION_1_2: u32 = 0x0001_0200;
/// Generator magic — OxiCUDA Level Zero backend.
pub const SPIRV_GENERATOR: u32 = 0x000D_0002;

// ─── SPIR-V opcodes ─────────────────────────────────────────

pub(crate) const OP_EXT_INST_IMPORT: u32 = 11;
pub(crate) const OP_EXT_INST: u32 = 12;
pub(crate) const OP_MEMORY_MODEL: u32 = 14;
pub(crate) const OP_ENTRY_POINT: u32 = 15;
pub(crate) const OP_EXECUTION_MODE: u32 = 16;
pub(crate) const OP_CAPABILITY: u32 = 17;
pub(crate) const OP_TYPE_VOID: u32 = 19;
pub(crate) const OP_TYPE_BOOL: u32 = 20;
pub(crate) const OP_TYPE_INT: u32 = 21;
pub(crate) const OP_TYPE_FLOAT: u32 = 22;
pub(crate) const OP_TYPE_VECTOR: u32 = 23;
pub(crate) const OP_TYPE_ARRAY: u32 = 28;
pub(crate) const OP_TYPE_POINTER: u32 = 32;
pub(crate) const OP_TYPE_FUNCTION: u32 = 33;
pub(crate) const OP_CONSTANT: u32 = 43;
pub(crate) const OP_FUNCTION: u32 = 54;
pub(crate) const OP_FUNCTION_PARAMETER: u32 = 55;
pub(crate) const OP_FUNCTION_END: u32 = 56;
pub(crate) const OP_VARIABLE: u32 = 59;
pub(crate) const OP_LOAD: u32 = 61;
pub(crate) const OP_STORE: u32 = 62;
pub(crate) const OP_IN_BOUNDS_PTR_ACCESS_CHAIN: u32 = 70;
pub(crate) const OP_DECORATE: u32 = 71;
pub(crate) const OP_COMPOSITE_EXTRACT: u32 = 81;
pub(crate) const OP_CONVERT_U_TO_F: u32 = 112;
pub(crate) const OP_F_NEGATE: u32 = 127;
pub(crate) const OP_I_ADD: u32 = 128;
pub(crate) const OP_F_ADD: u32 = 129;
pub(crate) const OP_F_SUB: u32 = 131;
pub(crate) const OP_I_MUL: u32 = 132;
pub(crate) const OP_F_MUL: u32 = 133;
pub(crate) const OP_U_DIV: u32 = 134;
pub(crate) const OP_F_DIV: u32 = 136;
pub(crate) const OP_U_MOD: u32 = 137;
pub(crate) const OP_U_LESS_THAN: u32 = 176;
pub(crate) const OP_LOOP_MERGE: u32 = 246;
pub(crate) const OP_SELECTION_MERGE: u32 = 247;
pub(crate) const OP_LABEL: u32 = 248;
pub(crate) const OP_BRANCH: u32 = 249;
pub(crate) const OP_BRANCH_CONDITIONAL: u32 = 250;
pub(crate) const OP_CONTROL_BARRIER: u32 = 224;
pub(crate) const OP_PHI: u32 = 245;
pub(crate) const OP_RETURN: u32 = 253;

// GroupNonUniform opcodes (SPIR-V 1.3+)
pub(crate) const OP_GROUP_NON_UNIFORM_FADD: u32 = 350;
pub(crate) const OP_GROUP_NON_UNIFORM_SHUFFLE: u32 = 345;

// Capabilities
const CAPABILITY_SHADER: u32 = 1;
const CAPABILITY_ADDRESSES: u32 = 4;
const CAPABILITY_KERNEL: u32 = 6;

// Addressing / memory model
const ADDRESSING_MODEL_LOGICAL: u32 = 0;
const ADDRESSING_MODEL_PHYSICAL64: u32 = 2;
const MEMORY_MODEL_GLSL450: u32 = 1;
const MEMORY_MODEL_OPENCL: u32 = 2;

// Execution model / mode
const EXECUTION_MODEL_GLCOMPUTE: u32 = 5;
pub(crate) const EXECUTION_MODEL_KERNEL: u32 = 6;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

// Function control
pub(crate) const FUNCTION_CONTROL_NONE: u32 = 0;

// Decorations
const DECORATION_BUILTIN: u32 = 11;

// BuiltIn values
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// Storage classes
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_CROSS_WORKGROUP: u32 = 5;
pub(crate) const STORAGE_CLASS_FUNCTION: u32 = 7;

// Selection/loop control
const SELECTION_CONTROL_NONE: u32 = 0;
const LOOP_CONTROL_NONE: u32 = 0;

// OpenCL.std extended instruction numbers
pub(crate) const OPENCL_EXP: u32 = 19;
const OPENCL_FABS: u32 = 23;
pub(crate) const OPENCL_FMAX: u32 = 27;
const OPENCL_FMIN: u32 = 28;
const OPENCL_LOG: u32 = 37;
const OPENCL_SQRT: u32 = 61;
const OPENCL_TANH: u32 = 63;

/// Workgroup size for 1-D compute kernels.
pub(crate) const WORKGROUP_SIZE: u32 = 256;

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
    /// Create a new module with a placeholder header (bound filled at finalise).
    pub fn new() -> Self {
        let words = vec![SPIRV_MAGIC, SPIRV_VERSION_1_2, SPIRV_GENERATOR, 0, 0];
        Self { words, id_bound: 1 }
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

    pub(crate) fn emit_capability(&mut self, cap: u32) {
        self.emit(OP_CAPABILITY, &[cap]);
    }

    pub(crate) fn emit_ext_inst_import(&mut self, id: u32, name: &str) {
        let mut ops = vec![id];
        ops.extend(Self::string_words(name));
        self.emit(OP_EXT_INST_IMPORT, &ops);
    }

    pub(crate) fn emit_memory_model(&mut self, addressing: u32, memory: u32) {
        self.emit(OP_MEMORY_MODEL, &[addressing, memory]);
    }

    pub(crate) fn emit_entry_point(
        &mut self,
        model: u32,
        func_id: u32,
        name: &str,
        interfaces: &[u32],
    ) {
        let mut ops = vec![model, func_id];
        ops.extend(Self::string_words(name));
        ops.extend_from_slice(interfaces);
        self.emit(OP_ENTRY_POINT, &ops);
    }

    pub(crate) fn emit_execution_mode_local_size(&mut self, func_id: u32, x: u32, y: u32, z: u32) {
        self.emit(
            OP_EXECUTION_MODE,
            &[func_id, EXECUTION_MODE_LOCAL_SIZE, x, y, z],
        );
    }

    pub(crate) fn emit_decorate(&mut self, target: u32, decoration: u32, operands: &[u32]) {
        let mut ops = vec![target, decoration];
        ops.extend_from_slice(operands);
        self.emit(OP_DECORATE, &ops);
    }

    pub(crate) fn emit_type_void(&mut self, id: u32) {
        self.emit(OP_TYPE_VOID, &[id]);
    }

    pub(crate) fn emit_type_bool(&mut self, id: u32) {
        self.emit(OP_TYPE_BOOL, &[id]);
    }

    pub(crate) fn emit_type_int(&mut self, id: u32, width: u32, signedness: u32) {
        self.emit(OP_TYPE_INT, &[id, width, signedness]);
    }

    pub(crate) fn emit_type_float(&mut self, id: u32, width: u32) {
        self.emit(OP_TYPE_FLOAT, &[id, width]);
    }

    pub(crate) fn emit_type_vector(&mut self, id: u32, component: u32, count: u32) {
        self.emit(OP_TYPE_VECTOR, &[id, component, count]);
    }

    pub(crate) fn emit_type_pointer(&mut self, id: u32, storage_class: u32, pointee: u32) {
        self.emit(OP_TYPE_POINTER, &[id, storage_class, pointee]);
    }

    pub(crate) fn emit_type_function(&mut self, id: u32, return_type: u32, params: &[u32]) {
        let mut ops = vec![id, return_type];
        ops.extend_from_slice(params);
        self.emit(OP_TYPE_FUNCTION, &ops);
    }

    pub(crate) fn emit_constant_u32(&mut self, ty: u32, id: u32, value: u32) {
        self.emit(OP_CONSTANT, &[ty, id, value]);
    }

    pub(crate) fn emit_constant_f32(&mut self, ty: u32, id: u32, value: f32) {
        self.emit(OP_CONSTANT, &[ty, id, value.to_bits()]);
    }

    pub(crate) fn emit_variable(&mut self, ty: u32, id: u32, storage_class: u32) {
        self.emit(OP_VARIABLE, &[ty, id, storage_class]);
    }

    pub(crate) fn emit_load(&mut self, result_ty: u32, result: u32, pointer: u32) {
        self.emit(OP_LOAD, &[result_ty, result, pointer]);
    }

    pub(crate) fn emit_store(&mut self, pointer: u32, value: u32) {
        self.emit(OP_STORE, &[pointer, value]);
    }

    pub(crate) fn emit_in_bounds_ptr_access_chain(
        &mut self,
        result_ty: u32,
        result: u32,
        base: u32,
        element: u32,
    ) {
        self.emit(
            OP_IN_BOUNDS_PTR_ACCESS_CHAIN,
            &[result_ty, result, base, element],
        );
    }

    pub(crate) fn emit_function(&mut self, result_ty: u32, result: u32, control: u32, fn_ty: u32) {
        self.emit(OP_FUNCTION, &[result_ty, result, control, fn_ty]);
    }

    pub(crate) fn emit_function_parameter(&mut self, result_ty: u32, result: u32) {
        self.emit(OP_FUNCTION_PARAMETER, &[result_ty, result]);
    }

    pub(crate) fn emit_label(&mut self, id: u32) {
        self.emit(OP_LABEL, &[id]);
    }

    pub(crate) fn emit_return(&mut self) {
        self.emit(OP_RETURN, &[]);
    }

    pub(crate) fn emit_function_end(&mut self) {
        self.emit(OP_FUNCTION_END, &[]);
    }

    pub(crate) fn emit_branch(&mut self, target: u32) {
        self.emit(OP_BRANCH, &[target]);
    }

    pub(crate) fn emit_branch_conditional(&mut self, cond: u32, true_label: u32, false_label: u32) {
        self.emit(OP_BRANCH_CONDITIONAL, &[cond, true_label, false_label]);
    }

    pub(crate) fn emit_selection_merge(&mut self, merge_label: u32) {
        self.emit(OP_SELECTION_MERGE, &[merge_label, SELECTION_CONTROL_NONE]);
    }

    pub(crate) fn emit_loop_merge(&mut self, merge_label: u32, continue_label: u32) {
        self.emit(
            OP_LOOP_MERGE,
            &[merge_label, continue_label, LOOP_CONTROL_NONE],
        );
    }

    pub(crate) fn emit_opencl_ext(
        &mut self,
        ext_id: u32,
        result_ty: u32,
        result: u32,
        inst: u32,
        args: &[u32],
    ) {
        let mut ops = vec![result_ty, result, ext_id, inst];
        ops.extend_from_slice(args);
        self.emit(OP_EXT_INST, &ops);
    }
}

impl Default for SpvModule {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Common preamble for OpenCL SPIR-V kernels ──────────────

/// IDs shared by all compute kernels.
pub(crate) struct BaseIds {
    pub(crate) ty_void: u32,
    pub(crate) ty_bool: u32,
    pub(crate) ty_uint: u32,
    pub(crate) ty_float: u32,
    #[allow(dead_code)]
    pub(crate) ty_v3uint: u32,
    #[allow(dead_code)]
    pub(crate) ty_fn_void: u32,
    #[allow(dead_code)]
    pub(crate) ty_ptr_input_v3uint: u32,
    pub(crate) ty_ptr_cross_float: u32,
    pub(crate) ty_ptr_func_float: u32,
    pub(crate) ty_ptr_func_uint: u32,
    pub(crate) c_uint_0: u32,
    pub(crate) c_uint_1: u32,
    pub(crate) c_float_0: u32,
    pub(crate) c_float_1: u32,
    pub(crate) var_gid: u32,
    pub(crate) opencl_ext: u32,
}

/// Emit the preamble shared by all OpenCL-style compute kernels.
///
/// This emits capabilities, memory model, types, constants, and the
/// `GlobalInvocationId` Input variable.  The caller must separately emit
/// `OpEntryPoint`, `OpExecutionMode`, and the function body.
pub(crate) fn emit_preamble(m: &mut SpvModule) -> BaseIds {
    let ty_void = m.alloc_id();
    let ty_bool = m.alloc_id();
    let ty_uint = m.alloc_id();
    let ty_float = m.alloc_id();
    let ty_v3uint = m.alloc_id();
    let ty_fn_void = m.alloc_id();
    let ty_ptr_input_v3uint = m.alloc_id();
    let ty_ptr_cross_float = m.alloc_id();
    let ty_ptr_func_float = m.alloc_id();
    let ty_ptr_func_uint = m.alloc_id();
    let c_uint_0 = m.alloc_id();
    let c_uint_1 = m.alloc_id();
    let c_float_0 = m.alloc_id();
    let c_float_1 = m.alloc_id();
    let var_gid = m.alloc_id();
    let opencl_ext = m.alloc_id();

    // Capabilities
    m.emit_capability(CAPABILITY_KERNEL);
    m.emit_capability(CAPABILITY_ADDRESSES);

    // Extension import
    m.emit_ext_inst_import(opencl_ext, "OpenCL.std");

    // Memory model: Physical64 + OpenCL
    m.emit_memory_model(ADDRESSING_MODEL_PHYSICAL64, MEMORY_MODEL_OPENCL);

    // NOTE: OpEntryPoint and OpExecutionMode are emitted by the caller after
    // allocating the main function ID, so we skip them here.

    // Decoration: GlobalInvocationId on var_gid
    m.emit_decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Types
    m.emit_type_void(ty_void);
    m.emit_type_bool(ty_bool);
    m.emit_type_int(ty_uint, 32, 0);
    m.emit_type_float(ty_float, 32);
    m.emit_type_vector(ty_v3uint, ty_uint, 3);
    m.emit_type_function(ty_fn_void, ty_void, &[]);
    m.emit_type_pointer(ty_ptr_input_v3uint, STORAGE_CLASS_INPUT, ty_v3uint);
    m.emit_type_pointer(ty_ptr_cross_float, STORAGE_CLASS_CROSS_WORKGROUP, ty_float);
    m.emit_type_pointer(ty_ptr_func_float, STORAGE_CLASS_FUNCTION, ty_float);
    m.emit_type_pointer(ty_ptr_func_uint, STORAGE_CLASS_FUNCTION, ty_uint);

    // Constants
    m.emit_constant_u32(ty_uint, c_uint_0, 0);
    m.emit_constant_u32(ty_uint, c_uint_1, 1);
    m.emit_constant_f32(ty_float, c_float_0, 0.0);
    m.emit_constant_f32(ty_float, c_float_1, 1.0);

    // GlobalInvocationId input variable
    m.emit_variable(ty_ptr_input_v3uint, var_gid, STORAGE_CLASS_INPUT);

    BaseIds {
        ty_void,
        ty_bool,
        ty_uint,
        ty_float,
        ty_v3uint,
        ty_fn_void,
        ty_ptr_input_v3uint,
        ty_ptr_cross_float,
        ty_ptr_func_float,
        ty_ptr_func_uint,
        c_uint_0,
        c_uint_1,
        c_float_0,
        c_float_1,
        var_gid,
        opencl_ext,
    }
}

/// Load `GlobalInvocationId.x` into a uint result.
pub(crate) fn load_gid_x(m: &mut SpvModule, b: &BaseIds) -> u32 {
    let gid_val = m.alloc_id();
    m.emit_load(b.ty_v3uint, gid_val, b.var_gid);
    let gid_x = m.alloc_id();
    m.emit(OP_COMPOSITE_EXTRACT, &[b.ty_uint, gid_x, gid_val, 0]);
    gid_x
}

// ─── Unary compute kernel ───────────────────────────────────

/// Generate an OpenCL SPIR-V compute kernel for an element-wise unary operation.
///
/// Kernel parameters: `(CrossWorkgroup float* input, CrossWorkgroup float* output, uint count)`.
pub fn unary_compute_shader(op: UnaryOp) -> Vec<u32> {
    let mut m = SpvModule::new();
    let b = emit_preamble(&mut m);

    let main_fn = m.alloc_id();
    let fn_ty = m.alloc_id();
    let p_input = m.alloc_id();
    let p_output = m.alloc_id();
    let p_count = m.alloc_id();

    // Function type: void(CrossWorkgroup float*, CrossWorkgroup float*, uint)
    m.emit_type_function(
        fn_ty,
        b.ty_void,
        &[b.ty_ptr_cross_float, b.ty_ptr_cross_float, b.ty_uint],
    );

    // Entry point and execution mode
    m.emit_entry_point(EXECUTION_MODEL_KERNEL, main_fn, "main", &[b.var_gid]);
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    // Labels
    let label_entry = m.alloc_id();
    let label_body = m.alloc_id();
    let label_merge = m.alloc_id();

    // Function
    m.emit_function(b.ty_void, main_fn, FUNCTION_CONTROL_NONE, fn_ty);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_input);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_output);
    m.emit_function_parameter(b.ty_uint, p_count);
    m.emit_label(label_entry);

    let gid = load_gid_x(&mut m, &b);

    // Bounds check: gid < count
    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, p_count]);
    m.emit_selection_merge(label_merge);
    m.emit_branch_conditional(cond, label_body, label_merge);

    m.emit_label(label_body);

    // input_ptr = &input[gid]
    let inp_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, inp_ptr, p_input, gid);
    let inp_val = m.alloc_id();
    m.emit_load(b.ty_float, inp_val, inp_ptr);

    let result = emit_unary_op(&mut m, &b, op, inp_val);

    // output_ptr = &output[gid]
    let out_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, out_ptr, p_output, gid);
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
            m.emit_opencl_ext(
                b.opencl_ext,
                b.ty_float,
                result,
                OPENCL_FMAX,
                &[b.c_float_0, x],
            );
        }
        UnaryOp::Sigmoid => {
            let neg_x = m.alloc_id();
            m.emit(OP_F_NEGATE, &[b.ty_float, neg_x, x]);
            let exp_neg_x = m.alloc_id();
            m.emit_opencl_ext(b.opencl_ext, b.ty_float, exp_neg_x, OPENCL_EXP, &[neg_x]);
            let one_plus = m.alloc_id();
            m.emit(OP_F_ADD, &[b.ty_float, one_plus, b.c_float_1, exp_neg_x]);
            m.emit(OP_F_DIV, &[b.ty_float, result, b.c_float_1, one_plus]);
        }
        UnaryOp::Tanh => {
            m.emit_opencl_ext(b.opencl_ext, b.ty_float, result, OPENCL_TANH, &[x]);
        }
        UnaryOp::Exp => {
            m.emit_opencl_ext(b.opencl_ext, b.ty_float, result, OPENCL_EXP, &[x]);
        }
        UnaryOp::Log => {
            m.emit_opencl_ext(b.opencl_ext, b.ty_float, result, OPENCL_LOG, &[x]);
        }
        UnaryOp::Sqrt => {
            m.emit_opencl_ext(b.opencl_ext, b.ty_float, result, OPENCL_SQRT, &[x]);
        }
        UnaryOp::Abs => {
            m.emit_opencl_ext(b.opencl_ext, b.ty_float, result, OPENCL_FABS, &[x]);
        }
        UnaryOp::Neg => {
            m.emit(OP_F_NEGATE, &[b.ty_float, result, x]);
        }
    }
    result
}

// ─── Binary compute kernel ──────────────────────────────────

/// Generate an OpenCL SPIR-V compute kernel for an element-wise binary operation.
///
/// Kernel parameters: `(CrossWorkgroup float* a, CrossWorkgroup float* b,
///                      CrossWorkgroup float* output, uint count)`.
pub fn binary_compute_shader(op: BinaryOp) -> Vec<u32> {
    let mut m = SpvModule::new();
    let b = emit_preamble(&mut m);

    let main_fn = m.alloc_id();
    let fn_ty = m.alloc_id();
    let p_a = m.alloc_id();
    let p_b = m.alloc_id();
    let p_out = m.alloc_id();
    let p_count = m.alloc_id();

    // Function type: void(float*, float*, float*, uint)
    m.emit_type_function(
        fn_ty,
        b.ty_void,
        &[
            b.ty_ptr_cross_float,
            b.ty_ptr_cross_float,
            b.ty_ptr_cross_float,
            b.ty_uint,
        ],
    );

    m.emit_entry_point(EXECUTION_MODEL_KERNEL, main_fn, "main", &[b.var_gid]);
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    let label_entry = m.alloc_id();
    let label_body = m.alloc_id();
    let label_merge = m.alloc_id();

    m.emit_function(b.ty_void, main_fn, FUNCTION_CONTROL_NONE, fn_ty);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_a);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_b);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_out);
    m.emit_function_parameter(b.ty_uint, p_count);
    m.emit_label(label_entry);

    let gid = load_gid_x(&mut m, &b);

    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, p_count]);
    m.emit_selection_merge(label_merge);
    m.emit_branch_conditional(cond, label_body, label_merge);

    m.emit_label(label_body);

    let a_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, a_ptr, p_a, gid);
    let a_val = m.alloc_id();
    m.emit_load(b.ty_float, a_val, a_ptr);

    let b_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, b_ptr, p_b, gid);
    let b_val = m.alloc_id();
    m.emit_load(b.ty_float, b_val, b_ptr);

    let result = emit_binary_op(&mut m, &b, op, a_val, b_val);

    let out_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, out_ptr, p_out, gid);
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
            m.emit_opencl_ext(b.opencl_ext, b.ty_float, result, OPENCL_FMAX, &[lhs, rhs]);
        }
        BinaryOp::Min => {
            m.emit_opencl_ext(b.opencl_ext, b.ty_float, result, OPENCL_FMIN, &[lhs, rhs]);
        }
    }
    result
}

// ─── Reduce compute kernel ──────────────────────────────────

/// Generate an OpenCL SPIR-V compute kernel for reduction along an axis.
///
/// Kernel parameters: `(CrossWorkgroup float* input, CrossWorkgroup float* output,
///                      uint outer_size, uint reduce_size, uint inner_size)`.
///
/// Each thread computes one output element by iterating over the reduce dimension.
pub fn reduce_compute_shader(op: ReduceOp) -> Vec<u32> {
    let mut m = SpvModule::new();
    let b = emit_preamble(&mut m);

    let main_fn = m.alloc_id();
    let fn_ty = m.alloc_id();
    let p_input = m.alloc_id();
    let p_output = m.alloc_id();
    let p_outer = m.alloc_id();
    let p_reduce = m.alloc_id();
    let p_inner = m.alloc_id();

    // Function type: void(float*, float*, uint, uint, uint)
    m.emit_type_function(
        fn_ty,
        b.ty_void,
        &[
            b.ty_ptr_cross_float,
            b.ty_ptr_cross_float,
            b.ty_uint,
            b.ty_uint,
            b.ty_uint,
        ],
    );

    m.emit_entry_point(EXECUTION_MODEL_KERNEL, main_fn, "main", &[b.var_gid]);
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    let label_entry = m.alloc_id();
    let label_bounds_body = m.alloc_id();
    let label_bounds_merge = m.alloc_id();
    let label_loop_header = m.alloc_id();
    let label_loop_body = m.alloc_id();
    let label_loop_continue = m.alloc_id();
    let label_loop_merge = m.alloc_id();

    m.emit_function(b.ty_void, main_fn, FUNCTION_CONTROL_NONE, fn_ty);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_input);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_output);
    m.emit_function_parameter(b.ty_uint, p_outer);
    m.emit_function_parameter(b.ty_uint, p_reduce);
    m.emit_function_parameter(b.ty_uint, p_inner);
    m.emit_label(label_entry);

    let gid = load_gid_x(&mut m, &b);

    // total_output = outer_size * inner_size
    let total_output = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, total_output, p_outer, p_inner]);

    // Bounds check: gid < total_output
    let cond_bounds = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond_bounds, gid, total_output]);
    m.emit_selection_merge(label_bounds_merge);
    m.emit_branch_conditional(cond_bounds, label_bounds_body, label_bounds_merge);

    m.emit_label(label_bounds_body);

    // outer_idx = gid / inner_size, inner_idx = gid % inner_size
    let outer_idx = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, outer_idx, gid, p_inner]);
    let inner_idx = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, inner_idx, gid, p_inner]);

    // base = outer_idx * reduce_size * inner_size + inner_idx
    let t1 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, t1, outer_idx, p_reduce]);
    let t2 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, t2, t1, p_inner]);
    let base_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, base_idx, t2, inner_idx]);

    // Loop counter
    let var_i = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_i, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_i, b.c_uint_0);

    // Accumulator
    let var_acc = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);
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
    m.emit_store(var_acc, init_val);

    m.emit_branch(label_loop_header);

    // ── Loop header ──
    m.emit_label(label_loop_header);
    let i_val = m.alloc_id();
    m.emit_load(b.ty_uint, i_val, var_i);
    let loop_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, loop_cond, i_val, p_reduce]);
    m.emit_loop_merge(label_loop_merge, label_loop_continue);
    m.emit_branch_conditional(loop_cond, label_loop_body, label_loop_merge);

    // ── Loop body ──
    m.emit_label(label_loop_body);

    // input_idx = base_idx + i * inner_size
    let i_times_inner = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, i_times_inner, i_val, p_inner]);
    let input_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, input_idx, base_idx, i_times_inner]);

    let inp_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, inp_ptr, p_input, input_idx);
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
            m.emit_opencl_ext(
                b.opencl_ext,
                b.ty_float,
                new_acc,
                OPENCL_FMAX,
                &[acc_val, inp_val],
            );
        }
        ReduceOp::Min => {
            m.emit_opencl_ext(
                b.opencl_ext,
                b.ty_float,
                new_acc,
                OPENCL_FMIN,
                &[acc_val, inp_val],
            );
        }
    }
    m.emit_store(var_acc, new_acc);

    m.emit_branch(label_loop_continue);

    // ── Loop continue ──
    m.emit_label(label_loop_continue);
    let i_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, i_inc, i_val, b.c_uint_1]);
    m.emit_store(var_i, i_inc);
    m.emit_branch(label_loop_header);

    // ── Loop merge ──
    m.emit_label(label_loop_merge);

    let final_acc = m.alloc_id();
    m.emit_load(b.ty_float, final_acc, var_acc);

    let store_val = if op == ReduceOp::Mean {
        let reduce_f = m.alloc_id();
        m.emit(OP_CONVERT_U_TO_F, &[b.ty_float, reduce_f, p_reduce]);
        let mean_val = m.alloc_id();
        m.emit(OP_F_DIV, &[b.ty_float, mean_val, final_acc, reduce_f]);
        mean_val
    } else {
        final_acc
    };

    let out_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, out_ptr, p_output, gid);
    m.emit_store(out_ptr, store_val);

    m.emit_branch(label_bounds_merge);

    m.emit_label(label_bounds_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── GEMM compute kernel ────────────────────────────────────

/// Generate an OpenCL SPIR-V compute kernel for GEMM: `C = alpha * A * B + beta * C`.
///
/// Naive reference (one thread per output element, row-major f32 layout).
///
/// Kernel parameters: `(CrossWorkgroup float* A, CrossWorkgroup float* B,
///                      CrossWorkgroup float* C, uint m, uint n, uint k,
///                      float alpha, float beta)`.
pub fn gemm_compute_shader() -> Vec<u32> {
    let mut m = SpvModule::new();
    let b = emit_preamble(&mut m);

    let main_fn = m.alloc_id();
    let fn_ty = m.alloc_id();
    let p_a = m.alloc_id();
    let p_b = m.alloc_id();
    let p_c = m.alloc_id();
    let p_m = m.alloc_id();
    let p_n = m.alloc_id();
    let p_k = m.alloc_id();
    let p_alpha = m.alloc_id();
    let p_beta = m.alloc_id();

    // Function type: void(float*, float*, float*, uint, uint, uint, float, float)
    m.emit_type_function(
        fn_ty,
        b.ty_void,
        &[
            b.ty_ptr_cross_float,
            b.ty_ptr_cross_float,
            b.ty_ptr_cross_float,
            b.ty_uint,
            b.ty_uint,
            b.ty_uint,
            b.ty_float,
            b.ty_float,
        ],
    );

    m.emit_entry_point(EXECUTION_MODEL_KERNEL, main_fn, "main", &[b.var_gid]);
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    let label_entry = m.alloc_id();
    let label_bounds_body = m.alloc_id();
    let label_bounds_merge = m.alloc_id();
    let label_loop_header = m.alloc_id();
    let label_loop_body = m.alloc_id();
    let label_loop_continue = m.alloc_id();
    let label_loop_merge = m.alloc_id();

    m.emit_function(b.ty_void, main_fn, FUNCTION_CONTROL_NONE, fn_ty);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_a);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_b);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_c);
    m.emit_function_parameter(b.ty_uint, p_m);
    m.emit_function_parameter(b.ty_uint, p_n);
    m.emit_function_parameter(b.ty_uint, p_k);
    m.emit_function_parameter(b.ty_float, p_alpha);
    m.emit_function_parameter(b.ty_float, p_beta);
    m.emit_label(label_entry);

    let gid = load_gid_x(&mut m, &b);

    // total = m * n
    let total = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, total, p_m, p_n]);

    // Bounds check
    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, total]);
    m.emit_selection_merge(label_bounds_merge);
    m.emit_branch_conditional(cond, label_bounds_body, label_bounds_merge);

    m.emit_label(label_bounds_body);

    // row = gid / n, col = gid % n
    let row = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, row, gid, p_n]);
    let col = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, col, gid, p_n]);

    // Loop counter + accumulator
    let var_i = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_i, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_i, b.c_uint_0);
    let var_acc = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_acc, b.c_float_0);

    m.emit_branch(label_loop_header);

    // ── Loop header ──
    m.emit_label(label_loop_header);
    let i_val = m.alloc_id();
    m.emit_load(b.ty_uint, i_val, var_i);
    let loop_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, loop_cond, i_val, p_k]);
    m.emit_loop_merge(label_loop_merge, label_loop_continue);
    m.emit_branch_conditional(loop_cond, label_loop_body, label_loop_merge);

    // ── Loop body ──
    m.emit_label(label_loop_body);

    // a_idx = row * k + i
    let row_k = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, row_k, row, p_k]);
    let a_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, a_idx, row_k, i_val]);

    // b_idx = i * n + col
    let i_n = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, i_n, i_val, p_n]);
    let b_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, b_idx, i_n, col]);

    let a_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, a_ptr, p_a, a_idx);
    let a_val = m.alloc_id();
    m.emit_load(b.ty_float, a_val, a_ptr);

    let b_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, b_ptr, p_b, b_idx);
    let b_val = m.alloc_id();
    m.emit_load(b.ty_float, b_val, b_ptr);

    let prod = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, prod, a_val, b_val]);
    let old_acc = m.alloc_id();
    m.emit_load(b.ty_float, old_acc, var_acc);
    let new_acc = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, new_acc, old_acc, prod]);
    m.emit_store(var_acc, new_acc);

    m.emit_branch(label_loop_continue);

    // ── Loop continue ──
    m.emit_label(label_loop_continue);
    let i_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, i_inc, i_val, b.c_uint_1]);
    m.emit_store(var_i, i_inc);
    m.emit_branch(label_loop_header);

    // ── Loop merge ──
    m.emit_label(label_loop_merge);

    // result = alpha * acc + beta * C[gid]
    let final_acc = m.alloc_id();
    m.emit_load(b.ty_float, final_acc, var_acc);
    let alpha_acc = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, alpha_acc, p_alpha, final_acc]);

    let c_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, c_ptr, p_c, gid);
    let c_old = m.alloc_id();
    m.emit_load(b.ty_float, c_old, c_ptr);
    let beta_c = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, beta_c, p_beta, c_old]);
    let c_new = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, c_new, alpha_acc, beta_c]);
    m.emit_store(c_ptr, c_new);

    m.emit_branch(label_bounds_merge);

    m.emit_label(label_bounds_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── Batched GEMM compute kernel ─────────────────────────────

/// Load `GlobalInvocationId.z` into a uint result.
fn load_gid_z(m: &mut SpvModule, b: &BaseIds) -> u32 {
    let gid_val = m.alloc_id();
    m.emit_load(b.ty_v3uint, gid_val, b.var_gid);
    let gid_z = m.alloc_id();
    m.emit(OP_COMPOSITE_EXTRACT, &[b.ty_uint, gid_z, gid_val, 2]);
    gid_z
}

/// Generate an OpenCL SPIR-V compute kernel for batched GEMM.
///
/// For each batch `b` in `0..batch_count`:
///   `C_b = alpha * A_b * B_b + beta * C_b`
/// where `A_b` starts at offset `b * stride_a`, etc.
///
/// Uses 3D global work size `(ceil(m*n / WG), 1, batch_count)`:
/// - `get_global_id(0)` = element index within a single m×n output
/// - `get_global_id(2)` = batch index
///
/// Kernel parameters:
/// `(CrossWorkgroup float* A, CrossWorkgroup float* B, CrossWorkgroup float* C,
///   uint m, uint n, uint k, float alpha, float beta,
///   uint batch_count, uint stride_a, uint stride_b, uint stride_c)`.
pub fn batched_gemm_compute_shader() -> Vec<u32> {
    let mut m = SpvModule::new();
    let b = emit_preamble(&mut m);

    let main_fn = m.alloc_id();
    let fn_ty = m.alloc_id();
    let p_a = m.alloc_id();
    let p_b = m.alloc_id();
    let p_c = m.alloc_id();
    let p_m = m.alloc_id();
    let p_n = m.alloc_id();
    let p_k = m.alloc_id();
    let p_alpha = m.alloc_id();
    let p_beta = m.alloc_id();
    let p_batch_count = m.alloc_id();
    let p_stride_a = m.alloc_id();
    let p_stride_b = m.alloc_id();
    let p_stride_c = m.alloc_id();

    // Function type: void(float*, float*, float*, uint, uint, uint, float, float,
    //                      uint, uint, uint, uint)
    m.emit_type_function(
        fn_ty,
        b.ty_void,
        &[
            b.ty_ptr_cross_float,
            b.ty_ptr_cross_float,
            b.ty_ptr_cross_float,
            b.ty_uint,
            b.ty_uint,
            b.ty_uint,
            b.ty_float,
            b.ty_float,
            b.ty_uint,
            b.ty_uint,
            b.ty_uint,
            b.ty_uint,
        ],
    );

    m.emit_entry_point(EXECUTION_MODEL_KERNEL, main_fn, "main", &[b.var_gid]);
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    let label_entry = m.alloc_id();
    let label_bounds_body = m.alloc_id();
    let label_bounds_merge = m.alloc_id();
    let label_loop_header = m.alloc_id();
    let label_loop_body = m.alloc_id();
    let label_loop_continue = m.alloc_id();
    let label_loop_merge = m.alloc_id();

    m.emit_function(b.ty_void, main_fn, FUNCTION_CONTROL_NONE, fn_ty);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_a);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_b);
    m.emit_function_parameter(b.ty_ptr_cross_float, p_c);
    m.emit_function_parameter(b.ty_uint, p_m);
    m.emit_function_parameter(b.ty_uint, p_n);
    m.emit_function_parameter(b.ty_uint, p_k);
    m.emit_function_parameter(b.ty_float, p_alpha);
    m.emit_function_parameter(b.ty_float, p_beta);
    m.emit_function_parameter(b.ty_uint, p_batch_count);
    m.emit_function_parameter(b.ty_uint, p_stride_a);
    m.emit_function_parameter(b.ty_uint, p_stride_b);
    m.emit_function_parameter(b.ty_uint, p_stride_c);
    m.emit_label(label_entry);

    // gid_x = element index within single GEMM output
    let gid = load_gid_x(&mut m, &b);
    // gid_z = batch index
    let batch_idx = load_gid_z(&mut m, &b);

    // total = m * n
    let total = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, total, p_m, p_n]);

    // Bounds check: gid < total && batch_idx < batch_count
    let cond1 = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond1, gid, total]);
    let cond2 = m.alloc_id();
    m.emit(
        OP_U_LESS_THAN,
        &[b.ty_bool, cond2, batch_idx, p_batch_count],
    );
    // Combined condition via OpLogicalAnd
    let cond = m.alloc_id();
    // OpLogicalAnd = 166
    m.emit(166, &[b.ty_bool, cond, cond1, cond2]);
    m.emit_selection_merge(label_bounds_merge);
    m.emit_branch_conditional(cond, label_bounds_body, label_bounds_merge);

    m.emit_label(label_bounds_body);

    // Compute batch offsets: a_base = batch_idx * stride_a, etc.
    let a_offset = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, a_offset, batch_idx, p_stride_a]);
    let b_offset = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, b_offset, batch_idx, p_stride_b]);
    let c_offset = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, c_offset, batch_idx, p_stride_c]);

    // Offset the base pointers
    let a_batch = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, a_batch, p_a, a_offset);
    let b_batch = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, b_batch, p_b, b_offset);
    let c_batch = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, c_batch, p_c, c_offset);

    // row = gid / n, col = gid % n
    let row = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, row, gid, p_n]);
    let col = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, col, gid, p_n]);

    // Loop counter + accumulator
    let var_i = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_i, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_i, b.c_uint_0);
    let var_acc = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_acc, b.c_float_0);

    m.emit_branch(label_loop_header);

    // ── Loop header ──
    m.emit_label(label_loop_header);
    let i_val = m.alloc_id();
    m.emit_load(b.ty_uint, i_val, var_i);
    let loop_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, loop_cond, i_val, p_k]);
    m.emit_loop_merge(label_loop_merge, label_loop_continue);
    m.emit_branch_conditional(loop_cond, label_loop_body, label_loop_merge);

    // ── Loop body ──
    m.emit_label(label_loop_body);

    // a_idx = row * k + i
    let row_k = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, row_k, row, p_k]);
    let a_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, a_idx, row_k, i_val]);

    // b_idx = i * n + col
    let i_n = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, i_n, i_val, p_n]);
    let b_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, b_idx, i_n, col]);

    let a_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, a_ptr, a_batch, a_idx);
    let a_val = m.alloc_id();
    m.emit_load(b.ty_float, a_val, a_ptr);

    let b_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, b_ptr, b_batch, b_idx);
    let b_val = m.alloc_id();
    m.emit_load(b.ty_float, b_val, b_ptr);

    let prod = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, prod, a_val, b_val]);
    let old_acc = m.alloc_id();
    m.emit_load(b.ty_float, old_acc, var_acc);
    let new_acc = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, new_acc, old_acc, prod]);
    m.emit_store(var_acc, new_acc);

    m.emit_branch(label_loop_continue);

    // ── Loop continue ──
    m.emit_label(label_loop_continue);
    let i_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, i_inc, i_val, b.c_uint_1]);
    m.emit_store(var_i, i_inc);
    m.emit_branch(label_loop_header);

    // ── Loop merge ──
    m.emit_label(label_loop_merge);

    // result = alpha * acc + beta * C_batch[gid]
    let final_acc = m.alloc_id();
    m.emit_load(b.ty_float, final_acc, var_acc);
    let alpha_acc = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, alpha_acc, p_alpha, final_acc]);

    let c_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(b.ty_ptr_cross_float, c_ptr, c_batch, gid);
    let c_old = m.alloc_id();
    m.emit_load(b.ty_float, c_old, c_ptr);
    let beta_c = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, beta_c, p_beta, c_old]);
    let c_new = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, c_new, alpha_acc, beta_c]);
    m.emit_store(c_ptr, c_new);

    m.emit_branch(label_bounds_merge);

    m.emit_label(label_bounds_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── Trivial placeholder ────────────────────────────────────

/// Build a minimal valid Shader-style compute shader: `void main() {}`.
///
/// Uses `GLCompute` / Shader capability for basic Level Zero module validation.
pub fn trivial_compute_shader() -> Vec<u32> {
    let mut m = SpvModule::new();

    let id_main_fn = m.alloc_id();
    let id_void = m.alloc_id();
    let id_void_fn = m.alloc_id();
    let id_label = m.alloc_id();

    m.emit_capability(CAPABILITY_SHADER);
    m.emit_memory_model(ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450);

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

/// Return the trivial compute shader as a byte slice suitable for
/// passing to Level Zero module creation.
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

    fn check_valid_spirv(words: &[u32]) {
        assert!(words.len() >= 5, "too short for SPIR-V header");
        assert_eq!(words[0], SPIRV_MAGIC, "bad magic");
        assert!(words[3] > 0, "ID bound must be > 0");
        assert_eq!(words[4], 0, "schema must be 0");
    }

    #[test]
    fn placeholder_spv_valid_magic() {
        let words = trivial_compute_shader();
        check_valid_spirv(&words);
    }

    #[test]
    fn placeholder_spv_word_aligned() {
        let bytes = trivial_compute_shader_bytes();
        assert_eq!(bytes.len() % 4, 0);
    }

    #[test]
    fn placeholder_spv_version_and_schema() {
        let words = trivial_compute_shader();
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

    #[test]
    fn generator_magic_is_level_zero() {
        assert_eq!(SPIRV_GENERATOR, 0x000D_0002);
        assert_ne!(SPIRV_GENERATOR, 0x000D_0001);
    }

    // ── Compute kernel generation ────────────────────────────

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
    }

    #[test]
    fn batched_gemm_shader_valid() {
        let words = batched_gemm_compute_shader();
        check_valid_spirv(&words);
    }

    #[test]
    fn batched_gemm_shader_word_aligned() {
        let words = batched_gemm_compute_shader();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        assert_eq!(bytes.len() % 4, 0);
    }

    #[test]
    fn batched_gemm_shader_uses_kernel_capability() {
        let words = batched_gemm_compute_shader();
        let cap_header = (2u32 << 16) | OP_CAPABILITY;
        assert_eq!(words[5], cap_header);
        assert_eq!(words[6], 6); // CAPABILITY_KERNEL
    }

    #[test]
    fn all_kernel_shaders_word_aligned() {
        fn to_bytes(words: &[u32]) -> Vec<u8> {
            words.iter().flat_map(|w| w.to_ne_bytes()).collect()
        }
        assert_eq!(to_bytes(&unary_compute_shader(UnaryOp::Relu)).len() % 4, 0);
        assert_eq!(to_bytes(&binary_compute_shader(BinaryOp::Add)).len() % 4, 0);
        assert_eq!(to_bytes(&reduce_compute_shader(ReduceOp::Sum)).len() % 4, 0);
        assert_eq!(to_bytes(&gemm_compute_shader()).len() % 4, 0);
        assert_eq!(to_bytes(&batched_gemm_compute_shader()).len() % 4, 0);
    }

    #[test]
    fn kernel_shaders_use_opencl_memory_model() {
        // Check that kernel shaders use Physical64 + OpenCL memory model,
        // while trivial shader uses Logical + GLSL450.
        let trivial = trivial_compute_shader();
        let unary = unary_compute_shader(UnaryOp::Relu);

        // The trivial shader should contain the Shader capability (1)
        // The unary shader should contain the Kernel capability (6)
        // These appear in OpCapability instructions after the header.

        // After header (5 words), first instruction is OpCapability
        // Format: (2 << 16) | 17 = 0x00020011
        let cap_header = (2u32 << 16) | OP_CAPABILITY;
        assert_eq!(trivial[5], cap_header);
        assert_eq!(trivial[6], CAPABILITY_SHADER);
        assert_eq!(unary[5], cap_header);
        assert_eq!(unary[6], CAPABILITY_KERNEL);
    }
}
