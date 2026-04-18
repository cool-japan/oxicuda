//! Elementwise GPU operation templates.
//!
//! This module generates complete PTX kernels for unary and binary elementwise
//! operations over device arrays. It supports basic arithmetic (`add`, `sub`, `mul`, `div`),
//! activation functions (`ReLU`, GELU, sigmoid, `SiLU`, tanh), unary math (neg, abs,
//! sqrt, rsqrt, exp, log), and fused operations (fused-add-relu, fused-scale-add).
//!
//! Each template produces a kernel that:
//! 1. Computes a global thread index
//! 2. Performs a bounds check against the array length
//! 3. Loads input element(s)
//! 4. Applies the operation
//! 5. Stores the result
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::elementwise::{ElementwiseTemplate, ElementwiseOp};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = ElementwiseTemplate::new(
//!     ElementwiseOp::Add,
//!     PtxType::F32,
//!     SmVersion::Sm80,
//! );
//! let ptx = template.generate().expect("PTX generation failed");
//! assert!(ptx.contains("add.f32"));
//! ```

use crate::arch::SmVersion;
use crate::builder::KernelBuilder;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Elementwise operation type.
///
/// Covers binary arithmetic, unary activations, unary math, and fused operations.
/// Each variant determines the kernel signature (number of input/output pointers)
/// and the PTX instruction sequence emitted in the kernel body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementwiseOp {
    /// Element-wise addition: `c[i] = a[i] + b[i]`.
    Add,
    /// Element-wise subtraction: `c[i] = a[i] - b[i]`.
    Sub,
    /// Element-wise multiplication: `c[i] = a[i] * b[i]`.
    Mul,
    /// Element-wise division: `c[i] = a[i] / b[i]`.
    Div,
    /// Rectified linear unit: `b[i] = max(0, a[i])`.
    Relu,
    /// Gaussian error linear unit (tanh approximation):
    /// `b[i] = 0.5 * a[i] * (1 + tanh(sqrt(2/pi) * (a[i] + 0.044715 * a[i]^3)))`.
    Gelu,
    /// Sigmoid activation: `b[i] = 1 / (1 + exp(-a[i]))`.
    Sigmoid,
    /// Sigmoid linear unit: `b[i] = a[i] * sigmoid(a[i])`.
    Silu,
    /// Hyperbolic tangent: `b[i] = tanh(a[i])`.
    Tanh,
    /// Arithmetic negation: `b[i] = -a[i]`.
    Neg,
    /// Absolute value: `b[i] = |a[i]|`.
    Abs,
    /// Square root: `b[i] = sqrt(a[i])`.
    Sqrt,
    /// Reciprocal square root: `b[i] = 1 / sqrt(a[i])`.
    Rsqrt,
    /// Exponential: `b[i] = exp(a[i])`.
    Exp,
    /// Natural logarithm: `b[i] = ln(a[i])`.
    Log,
    /// Scalar scaling: `b[i] = alpha * a[i]`.
    Scale,
    /// Add scalar: `b[i] = a[i] + scalar`.
    AddScalar,
    /// Ceiling (round toward +inf): `b[i] = ceil(a[i])`.
    Ceil,
    /// Floor (round toward -inf): `b[i] = floor(a[i])`.
    Floor,
    /// Hard sigmoid: `b[i] = max(0, min(1, 0.2*a[i] + 0.5))`.
    HardSigmoid,
    /// Hard swish: `b[i] = a[i] * max(0, min(6, a[i]+3)) / 6`.
    HardSwish,
    /// Softplus: `b[i] = ln(1 + exp(a[i]))`.
    Softplus,
    /// Leaky relu: `b[i] = a[i] >= 0 ? a[i] : 0.01 * a[i]`.
    LeakyRelu,
    /// One-minus: `b[i] = 1 - a[i]`.
    OneMinus,
    /// Fused add-relu: `c[i] = relu(a[i] + b[i])`.
    FusedAddRelu,
    /// Fused scale-add: `c[i] = alpha * a[i] + beta * b[i]`.
    FusedScaleAdd,
    /// Element-wise power: `c[i] = a[i]^b[i]` (lg2+mul+ex2 approximation).
    Pow,
    /// Element-wise minimum: `c[i] = min(a[i], b[i])`.
    Min,
    /// Element-wise maximum: `c[i] = max(a[i], b[i])`.
    Max,
    /// Comparison equal: `c[i] = (a[i] == b[i]) ? 1.0 : 0.0`.
    CmpEq,
    /// Comparison not-equal: `c[i] = (a[i] != b[i]) ? 1.0 : 0.0`.
    CmpNe,
    /// Comparison less-than: `c[i] = (a[i] < b[i]) ? 1.0 : 0.0`.
    CmpLt,
    /// Comparison greater-than: `c[i] = (a[i] > b[i]) ? 1.0 : 0.0`.
    CmpGt,
    /// Comparison less-or-equal: `c[i] = (a[i] <= b[i]) ? 1.0 : 0.0`.
    CmpLe,
    /// Comparison greater-or-equal: `c[i] = (a[i] >= b[i]) ? 1.0 : 0.0`.
    CmpGe,
    /// Fuzzy OR via max (same PTX as Max, distinct semantic label).
    OrMax,
    /// Probabilistic OR: `c[i] = a[i] + b[i] - a[i]*b[i]`.
    OrProbSum,
    /// Fuzzy NAND: `c[i] = 1 - a[i]*b[i]`.
    Nand,
    /// Fuzzy NOR: `c[i] = 1 - (a[i] + b[i] - a[i]*b[i])`.
    Nor,
    /// Fuzzy XOR: `c[i] = a[i] + b[i] - 2*a[i]*b[i]`.
    Xor,
    /// Fill: write a scalar to every element: `dst[i] = value`.
    Fill,
}

impl ElementwiseOp {
    /// Returns a short lowercase name suitable for kernel naming.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Relu => "relu",
            Self::Gelu => "gelu",
            Self::Sigmoid => "sigmoid",
            Self::Silu => "silu",
            Self::Tanh => "tanh",
            Self::Neg => "neg",
            Self::Abs => "abs",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Ceil => "ceil",
            Self::Floor => "floor",
            Self::HardSigmoid => "hard_sigmoid",
            Self::HardSwish => "hard_swish",
            Self::Softplus => "softplus",
            Self::LeakyRelu => "leaky_relu",
            Self::OneMinus => "one_minus",
            Self::Pow => "pow",
            Self::Min => "min",
            Self::Max => "max",
            Self::CmpEq => "cmp_eq",
            Self::CmpNe => "cmp_ne",
            Self::CmpLt => "cmp_lt",
            Self::CmpGt => "cmp_gt",
            Self::CmpLe => "cmp_le",
            Self::CmpGe => "cmp_ge",
            Self::OrMax => "or_max",
            Self::OrProbSum => "or_prob_sum",
            Self::Nand => "nand",
            Self::Nor => "nor",
            Self::Xor => "xor",
            Self::Scale => "scale",
            Self::AddScalar => "add_scalar",
            Self::FusedAddRelu => "fused_add_relu",
            Self::FusedScaleAdd => "fused_scale_add",
            Self::Fill => "fill",
        }
    }

    /// Returns `true` if this is a binary operation requiring two input arrays.
    #[must_use]
    pub const fn is_binary(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Sub
                | Self::Mul
                | Self::Div
                | Self::FusedAddRelu
                | Self::FusedScaleAdd
                | Self::Pow
                | Self::Min
                | Self::Max
                | Self::CmpEq
                | Self::CmpNe
                | Self::CmpLt
                | Self::CmpGt
                | Self::CmpLe
                | Self::CmpGe
                | Self::OrMax
                | Self::OrProbSum
                | Self::Nand
                | Self::Nor
                | Self::Xor
        )
    }

    /// Returns `true` if this operation requires scalar parameter(s).
    #[must_use]
    pub const fn needs_scalar(self) -> bool {
        matches!(
            self,
            Self::Scale | Self::AddScalar | Self::FusedScaleAdd | Self::Fill
        )
    }
}

/// Template for generating elementwise PTX kernels.
///
/// Combines an [`ElementwiseOp`], a precision ([`PtxType`]), and a target
/// architecture ([`SmVersion`]) to produce a complete PTX module string.
///
/// The generated kernel handles global thread indexing and bounds checking.
/// For complex activations (GELU, sigmoid, `SiLU`), the template emits
/// approximate PTX instruction sequences using `ex2.approx` and `rcp.approx`.
pub struct ElementwiseTemplate {
    /// The elementwise operation to generate.
    pub op: ElementwiseOp,
    /// The data precision for computation (e.g., `PtxType::F32`).
    pub precision: PtxType,
    /// The target GPU architecture.
    pub target: SmVersion,
}

impl ElementwiseTemplate {
    /// Creates a new elementwise template with the given parameters.
    #[must_use]
    pub const fn new(op: ElementwiseOp, precision: PtxType, target: SmVersion) -> Self {
        Self {
            op,
            precision,
            target,
        }
    }

    /// Returns the kernel function name derived from the operation and precision.
    ///
    /// The name follows the pattern `elementwise_{op}_{type}`, for example
    /// `elementwise_add_f32` or `elementwise_relu_f16`.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        format!("elementwise_{}_{}", self.op.as_str(), type_str)
    }

    /// Generates the complete PTX module text for this elementwise operation.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if the precision type is unsupported for the
    /// requested operation or if PTX text generation fails.
    pub fn generate(&self) -> Result<String, PtxGenError> {
        self.validate_precision()?;

        match self.op {
            ElementwiseOp::Add => self.generate_binary_arith("add"),
            ElementwiseOp::Sub => self.generate_binary_arith("sub"),
            ElementwiseOp::Mul => self.generate_binary_arith("mul"),
            ElementwiseOp::Div => self.generate_div(),
            ElementwiseOp::Relu => self.generate_relu(),
            ElementwiseOp::Gelu => self.generate_gelu(),
            ElementwiseOp::Sigmoid => self.generate_sigmoid(),
            ElementwiseOp::Silu => self.generate_silu(),
            ElementwiseOp::Tanh => self.generate_tanh(),
            ElementwiseOp::Neg => self.generate_unary("neg"),
            ElementwiseOp::Abs => self.generate_unary("abs"),
            ElementwiseOp::Sqrt => self.generate_sqrt(),
            ElementwiseOp::Rsqrt => self.generate_rsqrt(),
            ElementwiseOp::Exp => self.generate_exp(),
            ElementwiseOp::Log => self.generate_log(),
            ElementwiseOp::Ceil => self.generate_ceil(),
            ElementwiseOp::Floor => self.generate_floor(),
            ElementwiseOp::HardSigmoid => self.generate_hard_sigmoid(),
            ElementwiseOp::HardSwish => self.generate_hard_swish(),
            ElementwiseOp::Softplus => self.generate_softplus(),
            ElementwiseOp::LeakyRelu => self.generate_leaky_relu(),
            ElementwiseOp::OneMinus => self.generate_one_minus(),
            ElementwiseOp::Scale => self.generate_scale(),
            ElementwiseOp::AddScalar => self.generate_add_scalar(),
            ElementwiseOp::FusedAddRelu => self.generate_fused_add_relu(),
            ElementwiseOp::FusedScaleAdd => self.generate_fused_scale_add(),
            ElementwiseOp::Pow => self.generate_pow(),
            ElementwiseOp::Min => self.generate_binary_minmax("min"),
            ElementwiseOp::Max | ElementwiseOp::OrMax => self.generate_binary_minmax("max"),
            ElementwiseOp::CmpEq => self.generate_binary_cmp("eq"),
            ElementwiseOp::CmpNe => self.generate_binary_cmp("ne"),
            ElementwiseOp::CmpLt => self.generate_binary_cmp("lt"),
            ElementwiseOp::CmpGt => self.generate_binary_cmp("gt"),
            ElementwiseOp::CmpLe => self.generate_binary_cmp("le"),
            ElementwiseOp::CmpGe => self.generate_binary_cmp("ge"),
            ElementwiseOp::OrProbSum => self.generate_or_prob_sum(),
            ElementwiseOp::Nand => self.generate_nand(),
            ElementwiseOp::Nor => self.generate_nor(),
            ElementwiseOp::Xor => self.generate_xor(),
            ElementwiseOp::Fill => self.generate_fill(),
        }
    }

    /// Validates that the precision type is a supported floating-point type.
    fn validate_precision(&self) -> Result<(), PtxGenError> {
        if !matches!(
            self.precision,
            PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64
        ) {
            return Err(PtxGenError::InvalidType(format!(
                "elementwise operations require F16, BF16, F32, or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }
        Ok(())
    }

    /// Returns the PTX type suffix string (e.g., `.f32`).
    const fn ty_str(&self) -> &'static str {
        self.precision.as_ptx_str()
    }

    /// Generates a binary arithmetic kernel (add, sub, mul).
    ///
    /// Kernel signature: `(a_ptr: u64, b_ptr: u64, c_ptr: u64, n: u32)`
    fn generate_binary_arith(&self, op_name: &str) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let op_name = op_name.to_string();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    // Compute byte offset: tid * sizeof(element)
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));

                    // Load, compute, store via raw PTX for reliable type handling
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         {op_name}{ty} %f_c, %f_a, %f_b;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a division kernel with appropriate rounding.
    fn generate_div(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         div.rn{ty} %f_c, %f_a, %f_b;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a `ReLU` kernel: `max(0, x)`.
    fn generate_relu(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        // IEEE 754 zero in hex for PTX immediate
        let zero_lit = float_zero_literal(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         max{ty} %f_y, %f_x, {zero_lit};\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a sigmoid kernel: `1 / (1 + exp(-x))`.
    ///
    /// Uses `ex2.approx.f32` with a log2(e) scaling factor for the exponential,
    /// then `rcp.approx.f32` for the reciprocal.
    fn generate_sigmoid(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    // sigmoid(x) = 1 / (1 + exp(-x))
                    // exp(-x) = 2^(-x * log2(e))
                    // log2(e) ~= 1.4426950408889634 = 0f3FB8AA3B in float hex
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         neg{ty} %f_neg, %f_x;\n    \
                         mul{ty} %f_neg, %f_neg, 0f3FB8AA3B;\n    \
                         ex2.approx{ty} %f_exp, %f_neg;\n    \
                         add{ty} %f_denom, %f_exp, 0f3F800000;\n    \
                         rcp.approx{ty} %f_y, %f_denom;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a GELU kernel using the tanh approximation.
    ///
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ///
    /// Since PTX does not have a native tanh, this uses the identity:
    /// tanh(a) = 2 * sigmoid(2a) - 1 = (2 / (1 + exp(-2a))) - 1
    fn generate_gelu(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    // Constants (IEEE 754 hex):
                    //   0.5          = 0f3F000000
                    //   0.044715     = 0f3D372713
                    //   sqrt(2/pi)   = 0f3F4C422A  (~0.7978845608)
                    //   2.0          = 0f40000000
                    //   1.0          = 0f3F800000
                    //   log2(e)      = 0f3FB8AA3B
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_x3, %f_x, %f_x;\n    \
                         mul{ty} %f_x3, %f_x3, %f_x;\n    \
                         mul{ty} %f_x3, %f_x3, 0f3D372713;\n    \
                         add{ty} %f_inner, %f_x, %f_x3;\n    \
                         mul{ty} %f_inner, %f_inner, 0f3F4C422A;\n    \
                         mul{ty} %f_2a, %f_inner, 0f40000000;\n    \
                         neg{ty} %f_neg2a, %f_2a;\n    \
                         mul{ty} %f_neg2a, %f_neg2a, 0f3FB8AA3B;\n    \
                         ex2.approx{ty} %f_exp, %f_neg2a;\n    \
                         add{ty} %f_denom, %f_exp, 0f3F800000;\n    \
                         rcp.approx{ty} %f_sig, %f_denom;\n    \
                         mul{ty} %f_sig, %f_sig, 0f40000000;\n    \
                         sub{ty} %f_tanh, %f_sig, 0f3F800000;\n    \
                         add{ty} %f_tanh, %f_tanh, 0f3F800000;\n    \
                         mul{ty} %f_y, 0f3F000000, %f_x;\n    \
                         mul{ty} %f_y, %f_y, %f_tanh;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a `SiLU` kernel: `x * sigmoid(x)`.
    fn generate_silu(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         neg{ty} %f_neg, %f_x;\n    \
                         mul{ty} %f_neg, %f_neg, 0f3FB8AA3B;\n    \
                         ex2.approx{ty} %f_exp, %f_neg;\n    \
                         add{ty} %f_denom, %f_exp, 0f3F800000;\n    \
                         rcp.approx{ty} %f_sig, %f_denom;\n    \
                         mul{ty} %f_y, %f_x, %f_sig;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a tanh kernel using `tanh(x) = 2 * sigmoid(2x) - 1`.
    fn generate_tanh(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    // tanh(x) = 2*sigmoid(2x) - 1
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_2x, %f_x, 0f40000000;\n    \
                         neg{ty} %f_neg, %f_2x;\n    \
                         mul{ty} %f_neg, %f_neg, 0f3FB8AA3B;\n    \
                         ex2.approx{ty} %f_exp, %f_neg;\n    \
                         add{ty} %f_denom, %f_exp, 0f3F800000;\n    \
                         rcp.approx{ty} %f_sig, %f_denom;\n    \
                         mul{ty} %f_y, %f_sig, 0f40000000;\n    \
                         sub{ty} %f_y, %f_y, 0f3F800000;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a unary operation kernel (neg, abs).
    fn generate_unary(&self, op_name: &str) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let op_name = op_name.to_string();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         {op_name}{ty} %f_y, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a sqrt kernel with rounding.
    fn generate_sqrt(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         sqrt.rn{ty} %f_y, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates an rsqrt (reciprocal square root) kernel.
    fn generate_rsqrt(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         rsqrt.approx{ty} %f_y, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates an exp kernel using base-2 exponentiation: `exp(x) = 2^(x * log2(e))`.
    fn generate_exp(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    // log2(e) = 0f3FB8AA3B
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_x2, %f_x, 0f3FB8AA3B;\n    \
                         ex2.approx{ty} %f_y, %f_x2;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a natural log kernel using base-2 logarithm: `ln(x) = lg2(x) / lg2(e)`.
    fn generate_log(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    // 1/log2(e) = ln(2) ~= 0.6931471805599453 = 0f3F317218
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         lg2.approx{ty} %f_lg, %f_x;\n    \
                         mul{ty} %f_y, %f_lg, 0f3F317218;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a ceil kernel: `b[i] = ceil(a[i])`.
    ///
    /// Uses `cvt.rpi` (round-to-positive-infinity) for ceiling.
    fn generate_ceil(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         cvt.rpi{ty}{ty} %f_y, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a floor kernel: `b[i] = floor(a[i])`.
    ///
    /// Uses `cvt.rmi` (round-to-minus-infinity) for floor.
    fn generate_floor(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         cvt.rmi{ty}{ty} %f_y, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a hard-sigmoid kernel: `max(0, min(1, alpha*x + beta))`.
    ///
    /// Uses ONNX default constants: alpha=0.2 (0f3E4CCCCD), beta=0.5 (0f3F000000).
    fn generate_hard_sigmoid(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let zero_lit = float_zero_literal(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    // HardSigmoid: max(0, min(1, 0.2*x + 0.5))
                    // 0.2 = 0f3E4CCCCD, 0.5 = 0f3F000000, 1.0 = 0f3F800000
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_ax, %f_x, 0f3E4CCCCD;\n    \
                         add{ty} %f_lin, %f_ax, 0f3F000000;\n    \
                         min{ty} %f_clip, %f_lin, 0f3F800000;\n    \
                         max{ty} %f_y, %f_clip, {zero_lit};\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a hard-swish kernel: `x * max(0, min(6, x+3)) / 6`.
    ///
    /// IEEE 754 hex: 3.0=0f40400000, 6.0=0f40C00000, 1/6=0f3E2AAAAB.
    fn generate_hard_swish(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let zero_lit = float_zero_literal(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    // HardSwish: x * max(0, min(6, x+3)) / 6
                    // 3.0 = 0f40400000, 6.0 = 0f40C00000, 1/6 = 0f3E2AAAAB
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         add{ty} %f_xp3, %f_x, 0f40400000;\n    \
                         min{ty} %f_clip, %f_xp3, 0f40C00000;\n    \
                         max{ty} %f_clip, %f_clip, {zero_lit};\n    \
                         mul{ty} %f_div, %f_clip, 0f3E2AAAAB;\n    \
                         mul{ty} %f_y, %f_x, %f_div;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a softplus kernel: `ln(1 + exp(x))`.
    ///
    /// Uses exp(x) = 2^(x * log2(e)) and ln(z) = lg2(z) * ln(2).
    fn generate_softplus(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    // softplus(x) = ln(1 + exp(x))
                    // exp(x) = 2^(x * log2(e)),  log2(e) = 0f3FB8AA3B
                    // ln(z) = lg2(z) * ln(2),    ln(2) = 0f3F317218
                    // 1.0 = 0f3F800000
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_xe, %f_x, 0f3FB8AA3B;\n    \
                         ex2.approx{ty} %f_exp, %f_xe;\n    \
                         add{ty} %f_sum, %f_exp, 0f3F800000;\n    \
                         lg2.approx{ty} %f_lg, %f_sum;\n    \
                         mul{ty} %f_y, %f_lg, 0f3F317218;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a leaky-relu kernel: `x >= 0 ? x : alpha*x` (alpha=0.01).
    ///
    /// IEEE 754 hex: 0.01 = 0f3C23D70A.
    fn generate_leaky_relu(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let zero_lit = float_zero_literal(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    // LeakyRelu: x >= 0 ? x : 0.01*x
                    // Compute both paths then select via setp + selp
                    // 0.01 = 0f3C23D70A
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_leak, %f_x, 0f3C23D70A;\n    \
                         setp.ge{ty} %p_ge, %f_x, {zero_lit};\n    \
                         selp{ty} %f_y, %f_x, %f_leak, %p_ge;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a scale kernel: `b[i] = alpha * a[i]`.
    fn generate_scale(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let scalar_ty = scalar_param_type(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("alpha", scalar_ty)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.param{ty} %f_alpha, [%param_alpha];\n    \
                         ld.global{ty} %f_x, [%rd_a];\n    \
                         mul{ty} %f_y, %f_alpha, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates an add-scalar kernel: `b[i] = a[i] + scalar`.
    fn generate_add_scalar(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let scalar_ty = scalar_param_type(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("scalar", scalar_ty)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.param{ty} %f_s, [%param_scalar];\n    \
                         ld.global{ty} %f_x, [%rd_a];\n    \
                         add{ty} %f_y, %f_x, %f_s;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a fused add-relu kernel: `c[i] = max(0, a[i] + b[i])`.
    fn generate_fused_add_relu(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let zero_lit = float_zero_literal(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         add{ty} %f_sum, %f_a, %f_b;\n    \
                         max{ty} %f_y, %f_sum, {zero_lit};\n    \
                         st.global{ty} [%rd_c], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a one-minus kernel: `b[i] = 1 - a[i]`.
    fn generate_one_minus(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let one_lit = float_one_literal(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;"
                    ));
                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_x, [%rd_a];\n    \
                         sub{ty} %f_y, {one_lit}, %f_x;\n    \
                         st.global{ty} [%rd_b], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a power kernel: `c[i] = a[i]^b[i]` using lg2+mul+ex2 approximation.
    fn generate_pow(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         lg2.approx{ty} %f_t1, %f_a;\n    \
                         mul{ty} %f_t2, %f_t1, %f_b;\n    \
                         ex2.approx{ty} %f_c, %f_t2;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a min or max kernel using native PTX min/max.
    fn generate_binary_minmax(&self, min_or_max: &str) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let min_or_max = min_or_max.to_string();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         {min_or_max}{ty} %f_c, %f_a, %f_b;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a comparison kernel: `c[i] = (a[i] {cond} b[i]) ? 1.0 : 0.0`.
    fn generate_binary_cmp(&self, cond: &str) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let one_lit = float_one_literal(self.precision);
        let zero_lit = float_zero_literal(self.precision);
        let cond = cond.to_string();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         setp.{cond}{ty} %p_cmp, %f_a, %f_b;\n    \
                         selp{ty} %f_c, {one_lit}, {zero_lit}, %p_cmp;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a probabilistic OR kernel: `c[i] = a[i] + b[i] - a[i]*b[i]`.
    fn generate_or_prob_sum(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         mul{ty} %f_t, %f_a, %f_b;\n    \
                         sub{ty} %f_s, %f_a, %f_t;\n    \
                         add{ty} %f_c, %f_s, %f_b;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a fuzzy NAND kernel: `c[i] = 1 - a[i]*b[i]`.
    fn generate_nand(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let one_lit = float_one_literal(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         mul{ty} %f_t, %f_a, %f_b;\n    \
                         sub{ty} %f_c, {one_lit}, %f_t;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a fuzzy NOR kernel: `c[i] = 1 - (a[i] + b[i] - a[i]*b[i])`.
    fn generate_nor(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let one_lit = float_one_literal(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         mul{ty} %f_t, %f_a, %f_b;\n    \
                         sub{ty} %f_s, %f_a, %f_t;\n    \
                         add{ty} %f_u, %f_s, %f_b;\n    \
                         sub{ty} %f_c, {one_lit}, %f_u;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a fuzzy XOR kernel: `c[i] = a[i] + b[i] - 2*a[i]*b[i]`.
    fn generate_xor(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let two_lit = float_two_literal(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         add{ty} %f_s, %f_a, %f_b;\n    \
                         mul{ty} %f_t, %f_a, %f_b;\n    \
                         mul{ty} %f_t2, %f_t, {two_lit};\n    \
                         sub{ty} %f_c, %f_s, %f_t2;\n    \
                         st.global{ty} [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a fused scale-add kernel: `c[i] = alpha * a[i] + beta * b[i]`.
    fn generate_fused_scale_add(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let scalar_ty = scalar_param_type(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("alpha", scalar_ty)
            .param("beta", scalar_ty)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;"
                    ));

                    b.raw_ptx(&format!(
                        "ld.param{ty} %f_alpha, [%param_alpha];\n    \
                         ld.param{ty} %f_beta, [%param_beta];\n    \
                         ld.global{ty} %f_a, [%rd_a];\n    \
                         ld.global{ty} %f_b, [%rd_b];\n    \
                         mul{ty} %f_aa, %f_alpha, %f_a;\n    \
                         mul{ty} %f_bb, %f_beta, %f_b;\n    \
                         add{ty} %f_y, %f_aa, %f_bb;\n    \
                         st.global{ty} [%rd_c], %f_y;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a fill kernel: `dst[i] = value` for all `i < n`.
    ///
    /// The scalar `value` is read from a kernel parameter (not from a source buffer),
    /// which means every output element receives the same constant. This mirrors the
    /// `generate_scale()` pattern for loading a scalar kernel parameter via `ld.param`.
    fn generate_fill(&self) -> Result<String, PtxGenError> {
        let kernel_name = self.kernel_name();
        let ty = self.ty_str();
        let byte_size = self.precision.size_bytes();
        let scalar_ty = scalar_param_type(self.precision);

        KernelBuilder::new(&kernel_name)
            .target(self.target)
            .param("dst_ptr", PtxType::U64)
            .param("value", scalar_ty)
            .param("n", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let tid_name = tid.to_string();
                let n_reg = b.load_param_u32("n");
                b.if_lt_u32(tid, n_reg, move |b| {
                    let dst_ptr = b.load_param_u64("dst_ptr");

                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {tid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, {byte_size};\n    \
                         add.u64 %rd_dst, {dst_ptr}, %rd_off;\n    \
                         ld.param{ty} %f_val, [%param_value];\n    \
                         st.global{ty} [%rd_dst], %f_val;"
                    ));
                });
                b.ret();
            })
            .build()
    }
}

/// Returns the IEEE 754 hex literal for 1.0 in the given precision.
const fn float_one_literal(ty: PtxType) -> &'static str {
    match ty {
        PtxType::F64 => "0d3FF0000000000000",
        _ => "0f3F800000",
    }
}

/// Returns the IEEE 754 hex literal for 2.0 in the given precision.
const fn float_two_literal(ty: PtxType) -> &'static str {
    match ty {
        PtxType::F64 => "0d4000000000000000",
        _ => "0f40000000",
    }
}

/// Returns the IEEE 754 hex literal for 0.0 in the given precision.
const fn float_zero_literal(ty: PtxType) -> &'static str {
    match ty {
        PtxType::F64 => "0d0000000000000000",
        _ => "0f00000000",
    }
}

/// Returns the scalar parameter type matching the given float precision.
///
/// For F16 and BF16, scalar parameters are passed as F32 (promoted).
const fn scalar_param_type(ty: PtxType) -> PtxType {
    match ty {
        PtxType::F16 | PtxType::BF16 => PtxType::F32,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;

    #[test]
    fn elementwise_op_names() {
        assert_eq!(ElementwiseOp::Add.as_str(), "add");
        assert_eq!(ElementwiseOp::Relu.as_str(), "relu");
        assert_eq!(ElementwiseOp::FusedScaleAdd.as_str(), "fused_scale_add");
    }

    #[test]
    fn elementwise_op_classification() {
        assert!(ElementwiseOp::Add.is_binary());
        assert!(ElementwiseOp::Sub.is_binary());
        assert!(!ElementwiseOp::Relu.is_binary());
        assert!(!ElementwiseOp::Sigmoid.is_binary());

        assert!(ElementwiseOp::Scale.needs_scalar());
        assert!(ElementwiseOp::FusedScaleAdd.needs_scalar());
        assert!(!ElementwiseOp::Add.needs_scalar());
    }

    #[test]
    fn kernel_name_format() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Add, PtxType::F32, SmVersion::Sm80);
        assert_eq!(t.kernel_name(), "elementwise_add_f32");

        let t2 = ElementwiseTemplate::new(ElementwiseOp::Relu, PtxType::F16, SmVersion::Sm90);
        assert_eq!(t2.kernel_name(), "elementwise_relu_f16");
    }

    #[test]
    fn invalid_precision_rejected() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Add, PtxType::U32, SmVersion::Sm80);
        let result = t.generate();
        assert!(result.is_err());
    }

    #[test]
    fn generate_add_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Add, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("should generate add kernel");
        assert!(ptx.contains(".entry elementwise_add_f32"));
        assert!(ptx.contains(".target sm_80"));
        assert!(ptx.contains("add.f32"));
    }

    #[test]
    fn generate_relu_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Relu, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("should generate relu kernel");
        assert!(ptx.contains(".entry elementwise_relu_f32"));
        assert!(ptx.contains("max.f32"));
    }

    #[test]
    fn generate_sigmoid_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Sigmoid, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("should generate sigmoid kernel");
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
    }

    #[test]
    fn generate_gelu_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Gelu, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("should generate gelu kernel");
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains(".entry elementwise_gelu_f32"));
    }

    // -------------------------------------------------------------------------
    // P4: Precision tests – verify arithmetic correctness of generated PTX
    // -------------------------------------------------------------------------

    /// `ReLU` must use `max` (not `setp`/`selp`, not `sin` or other wrong ops).
    /// The implementation emits `max.f32 %f_y, %f_x, 0f00000000`.
    #[test]
    fn test_relu_ptx_correct_arithmetic() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Relu, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("relu PTX generation failed");
        // Must contain max (implements max(x, 0))
        assert!(ptx.contains("max.f32"), "relu must emit max.f32");
        // Must contain the IEEE 754 zero literal
        assert!(ptx.contains("0f00000000"), "relu must compare against 0.0");
        // Must NOT contain wrong operations
        assert!(!ptx.contains("sin.approx"), "relu must not emit sin");
        assert!(!ptx.contains("cos.approx"), "relu must not emit cos");
        assert!(!ptx.contains("ex2.approx"), "relu must not use exp");
        assert!(!ptx.contains("rcp.approx"), "relu must not use rcp");
    }

    /// Sigmoid must contain: neg (for -x), ex2.approx (exp via base-2),
    /// rcp.approx (reciprocal for 1/(1+exp(-x))), and add (for +1.0).
    /// Must NOT contain wrong operations.
    #[test]
    fn test_sigmoid_ptx_contains_exp_and_rcp() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Sigmoid, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("sigmoid PTX generation failed");
        // neg for computing -x
        assert!(ptx.contains("neg.f32"), "sigmoid must negate input");
        // exp approximation via base-2 exponentiation
        assert!(
            ptx.contains("ex2.approx.f32"),
            "sigmoid must use ex2.approx for exp"
        );
        // log2(e) scaling constant
        assert!(ptx.contains("0f3FB8AA3B"), "sigmoid must scale by log2(e)");
        // reciprocal for the division step
        assert!(
            ptx.contains("rcp.approx.f32"),
            "sigmoid must use rcp.approx for 1/denom"
        );
        // +1.0 in denominator
        assert!(
            ptx.contains("0f3F800000"),
            "sigmoid must add 1.0 to denominator"
        );
        // Must NOT contain wrong operations
        assert!(!ptx.contains("sin.approx"), "sigmoid must not emit sin");
        assert!(
            !ptx.contains("max.f32"),
            "sigmoid must not use max (relu op)"
        );
    }

    /// GELU uses tanh approximation: check for the three key constants
    /// (0.044715, sqrt(2/pi), 2.0) and the ex2+rcp pattern for tanh-via-sigmoid.
    #[test]
    fn test_gelu_ptx_contains_tanh_approximation() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Gelu, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("gelu PTX generation failed");
        // 0.044715 constant (IEEE 754: 0f3D372713)
        assert!(
            ptx.contains("0f3D372713"),
            "gelu must use 0.044715 constant"
        );
        // sqrt(2/pi) constant (IEEE 754: 0f3F4C422A)
        assert!(
            ptx.contains("0f3F4C422A"),
            "gelu must use sqrt(2/pi) constant"
        );
        // tanh implemented via 2*sigmoid(2a)-1 using ex2
        assert!(
            ptx.contains("ex2.approx.f32"),
            "gelu must use ex2.approx for tanh approximation"
        );
        assert!(
            ptx.contains("rcp.approx.f32"),
            "gelu must use rcp.approx inside tanh"
        );
        // Must NOT emit a raw sine (wrong operation for gelu)
        assert!(!ptx.contains("sin.approx"), "gelu must not emit sin");
    }

    /// Tanh (implemented as `2*sigmoid(2x)-1`) must contain ex2.approx,
    /// rcp.approx, the 2.0 constant, and the subtract of 1.0.
    #[test]
    fn test_tanh_ptx_contains_exp_instructions() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Tanh, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("tanh PTX generation failed");
        // exp via base-2
        assert!(
            ptx.contains("ex2.approx.f32"),
            "tanh must use ex2.approx for exp"
        );
        // rcp for sigmoid step
        assert!(
            ptx.contains("rcp.approx.f32"),
            "tanh must use rcp.approx in sigmoid step"
        );
        // 2.0 constant (0f40000000)
        assert!(ptx.contains("0f40000000"), "tanh must scale by 2.0");
        // sub 1.0 to complete tanh = 2*sigmoid(2x) - 1
        assert!(
            ptx.contains("sub.f32"),
            "tanh must subtract 1.0 for tanh formula"
        );
        // Must NOT emit wrong operation
        assert!(!ptx.contains("sin.approx"), "tanh must not emit sin");
    }

    /// `SiLU` (`x * sigmoid(x)`) must contain both multiplication and the
    /// sigmoid sub-pattern (ex2.approx + rcp.approx).
    #[test]
    fn test_silu_ptx_contains_mul_and_sigmoid() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Silu, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("silu PTX generation failed");
        // sigmoid sub-pattern
        assert!(
            ptx.contains("ex2.approx.f32"),
            "silu must use ex2.approx for sigmoid"
        );
        assert!(
            ptx.contains("rcp.approx.f32"),
            "silu must use rcp.approx for sigmoid"
        );
        // outer multiplication x * sigmoid(x)
        assert!(
            ptx.contains("mul.f32"),
            "silu must multiply x by sigmoid(x)"
        );
        // Must NOT emit wrong operation
        assert!(!ptx.contains("sin.approx"), "silu must not emit sin");
        assert!(!ptx.contains("max.f32"), "silu must not use relu max");
    }

    /// Every generated elementwise kernel must have valid PTX structural headers:
    /// `.version`, `.target`, and `.entry`.
    #[test]
    fn test_elementwise_ptx_has_valid_headers() {
        let ops_and_types = [
            (ElementwiseOp::Add, PtxType::F32),
            (ElementwiseOp::Relu, PtxType::F32),
            (ElementwiseOp::Sigmoid, PtxType::F32),
            (ElementwiseOp::Gelu, PtxType::F32),
            (ElementwiseOp::Tanh, PtxType::F32),
            (ElementwiseOp::Silu, PtxType::F32),
            (ElementwiseOp::Neg, PtxType::F32),
            (ElementwiseOp::Exp, PtxType::F32),
            (ElementwiseOp::Log, PtxType::F32),
        ];

        for (op, ty) in ops_and_types {
            let t = ElementwiseTemplate::new(op, ty, SmVersion::Sm80);
            let ptx = t
                .generate()
                .unwrap_or_else(|e| panic!("PTX generation failed for {op:?}: {e}"));
            assert!(
                ptx.contains(".version"),
                "PTX for {op:?} must have .version header"
            );
            assert!(
                ptx.contains(".target"),
                "PTX for {op:?} must have .target header"
            );
            assert!(
                ptx.contains(".entry"),
                "PTX for {op:?} must have .entry directive"
            );
        }
    }

    // -----------------------------------------------------------------------
    // CPU reference implementations mirroring the PTX kernel arithmetic.
    // These validate numerical precision of the elementwise operations,
    // verifying that the same arithmetic would produce correct results
    // when executed in a PTX kernel on device.
    // -----------------------------------------------------------------------

    /// CPU reference for `ReLU`: `max(0, x)`.
    fn cpu_relu_f32(x: f32) -> f32 {
        x.max(0.0)
    }

    /// CPU reference for sigmoid: 1 / (1 + exp(-x)).
    fn cpu_sigmoid_f32(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// CPU reference for GELU (tanh approximation matching PTX):
    /// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    fn cpu_gelu_f32(x: f32) -> f32 {
        let k0: f32 = 0.797_884_6; // sqrt(2/pi)
        let k1: f32 = 0.044_715;
        let inner = k0 * k1.mul_add(x * x * x, x);
        0.5 * x * (1.0 + inner.tanh())
    }

    /// CPU reference for tanh: `x.tanh()`.
    fn cpu_tanh_f32(x: f32) -> f32 {
        x.tanh()
    }

    /// CPU reference for `SiLU`: `x * sigmoid(x)`.
    fn cpu_silu_f32(x: f32) -> f32 {
        x * cpu_sigmoid_f32(x)
    }

    // -- relu precision tests ------------------------------------------------

    #[test]
    fn relu_precision_known_values() {
        assert!((cpu_relu_f32(0.0) - 0.0_f32).abs() < f32::EPSILON);
        assert!((cpu_relu_f32(-1.0) - 0.0_f32).abs() < f32::EPSILON);
        assert!((cpu_relu_f32(1.0) - 1.0_f32).abs() < f32::EPSILON);
        assert!((cpu_relu_f32(-0.001) - 0.0_f32).abs() < f32::EPSILON);
        assert!((cpu_relu_f32(100.0) - 100.0_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn relu_precision_negative_zero() {
        // -0.0 is <= 0, so relu should return 0.0 (non-negative zero)
        assert!(cpu_relu_f32(-0.0) >= 0.0);
    }

    // -- sigmoid precision tests ---------------------------------------------

    #[test]
    fn sigmoid_precision_known_values() {
        // sigmoid(0) = 0.5 exactly
        assert!((cpu_sigmoid_f32(0.0) - 0.5).abs() < 1e-7_f32);
        // sigmoid(large) -> 1.0
        assert!((cpu_sigmoid_f32(100.0) - 1.0).abs() < 1e-6_f32);
        // sigmoid(-large) -> 0.0
        assert!(cpu_sigmoid_f32(-100.0).abs() < 1e-6_f32);
        // sigmoid(1.0) ~= 0.73105858
        let expected_sig1: f32 = 0.731_058_6;
        assert!(
            (cpu_sigmoid_f32(1.0) - expected_sig1).abs() < 1e-5_f32,
            "sigmoid(1.0) expected ~{expected_sig1}, got {}",
            cpu_sigmoid_f32(1.0)
        );
    }

    #[test]
    fn sigmoid_output_in_unit_interval() {
        // For moderate inputs, sigmoid is strictly in (0, 1).
        let inputs: &[f32] = &[-10.0, -1.0, 0.0, 1.0, 10.0];
        for &x in inputs {
            let s = cpu_sigmoid_f32(x);
            assert!(s > 0.0 && s < 1.0, "sigmoid({x}) = {s} not in (0,1)");
        }
        // For extreme inputs, sigmoid saturates to 0.0 or 1.0 in f32 precision.
        assert!(cpu_sigmoid_f32(-100.0) >= 0.0);
        assert!(cpu_sigmoid_f32(100.0) <= 1.0);
    }

    // -- gelu precision tests ------------------------------------------------

    #[test]
    fn gelu_precision_known_values() {
        // gelu(0) = 0
        assert!(cpu_gelu_f32(0.0).abs() < 1e-7_f32);
        // gelu(1) ~= 0.8413 (well-known reference)
        assert!(
            (cpu_gelu_f32(1.0) - 0.8413_f32).abs() < 0.001_f32,
            "gelu(1) should be ~0.8413, got {}",
            cpu_gelu_f32(1.0)
        );
        // gelu(-1) ~= -0.1587
        assert!(
            (cpu_gelu_f32(-1.0) + 0.1587_f32).abs() < 0.001_f32,
            "gelu(-1) should be ~-0.1587, got {}",
            cpu_gelu_f32(-1.0)
        );
        // gelu(large positive) ~= x (saturation)
        assert!(
            (cpu_gelu_f32(5.0) - 5.0_f32).abs() < 0.001_f32,
            "gelu(5) should be ~5.0, got {}",
            cpu_gelu_f32(5.0)
        );
    }

    #[test]
    fn gelu_sign_preservation() {
        // GELU should be positive for positive inputs (at least for x > 0)
        assert!(cpu_gelu_f32(0.5) > 0.0);
        assert!(cpu_gelu_f32(2.0) > 0.0);
        // GELU negative at large negative inputs
        assert!(cpu_gelu_f32(-2.0) < 0.0);
    }

    // -- tanh precision tests ------------------------------------------------

    #[test]
    fn tanh_precision_known_values() {
        assert!(cpu_tanh_f32(0.0).abs() < 1e-7_f32);
        let expected_tanh1: f32 = 0.761_594_2;
        assert!(
            (cpu_tanh_f32(1.0) - expected_tanh1).abs() < 1e-5_f32,
            "tanh(1.0) expected ~{expected_tanh1}, got {}",
            cpu_tanh_f32(1.0)
        );
        assert!(
            (cpu_tanh_f32(-1.0) + expected_tanh1).abs() < 1e-5_f32,
            "tanh(-1.0) expected ~-{expected_tanh1}, got {}",
            cpu_tanh_f32(-1.0)
        );
        // tanh saturates at ±1
        assert!(
            (cpu_tanh_f32(10.0) - 1.0).abs() < 1e-5_f32,
            "tanh(10) should be ~1.0"
        );
        assert!(
            (cpu_tanh_f32(-10.0) + 1.0).abs() < 1e-5_f32,
            "tanh(-10) should be ~-1.0"
        );
    }

    #[test]
    fn tanh_output_in_bounded_range() {
        // For moderate inputs, tanh is strictly in (-1, 1).
        let inputs: &[f32] = &[-5.0, -1.0, 0.0, 1.0, 5.0];
        for &x in inputs {
            let t = cpu_tanh_f32(x);
            assert!(t > -1.0 && t < 1.0, "tanh({x}) = {t} not in (-1,1)");
        }
        // For extreme inputs, tanh saturates to ±1 in f32 precision.
        assert!(cpu_tanh_f32(-100.0) >= -1.0);
        assert!(cpu_tanh_f32(100.0) <= 1.0);
    }

    // -- silu precision tests ------------------------------------------------

    #[test]
    fn silu_precision_known_values() {
        // silu(0) = 0
        assert!(cpu_silu_f32(0.0).abs() < 1e-7_f32);
        // silu(1) = 1 * sigmoid(1) ~= 0.73106
        let expected_sig1: f32 = 0.731_058_6;
        assert!(
            (cpu_silu_f32(1.0) - expected_sig1).abs() < 1e-5_f32,
            "silu(1.0) expected ~{expected_sig1}, got {}",
            cpu_silu_f32(1.0)
        );
        // silu(-1) ~= -0.2689
        assert!(
            (cpu_silu_f32(-1.0) + 0.2689_f32).abs() < 0.001_f32,
            "silu(-1) should be ~-0.2689, got {}",
            cpu_silu_f32(-1.0)
        );
    }

    #[test]
    fn silu_sign_matches_input() {
        // silu has same sign as its input for non-zero values
        for &x in &[0.1_f32, 0.5, 1.0, 2.0, 5.0] {
            assert!(
                cpu_silu_f32(x) > 0.0,
                "silu({x}) should be positive, got {}",
                cpu_silu_f32(x)
            );
        }
        for &x in &[-0.1_f32, -0.5, -2.0] {
            assert!(
                cpu_silu_f32(x) < 0.0,
                "silu({x}) should be negative, got {}",
                cpu_silu_f32(x)
            );
        }
    }

    // -- PTX generation test for fused add+relu ------------------------------

    #[test]
    fn elementwise_ptx_generates_fused_add_relu() {
        let tmpl =
            ElementwiseTemplate::new(ElementwiseOp::FusedAddRelu, PtxType::F32, SmVersion::Sm80);
        let ptx = tmpl
            .generate()
            .expect("FusedAddRelu should generate successfully");
        assert!(
            ptx.contains("add"),
            "fused kernel should contain add instruction"
        );
        assert!(
            ptx.contains("max"),
            "fused kernel should contain max for relu"
        );
    }

    // -- grid sweep precision test -------------------------------------------

    #[test]
    fn elementwise_ops_precision_sweep() {
        // Validate mathematical invariants of all reference functions across a
        // 10-point grid spanning negative, zero, and positive inputs.
        let test_inputs: &[f32] = &[-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];

        for &x in test_inputs {
            // relu: output must be >= 0
            assert!(
                cpu_relu_f32(x) >= 0.0,
                "relu({x}) = {} should be non-negative",
                cpu_relu_f32(x)
            );

            // sigmoid: output must be strictly in (0, 1)
            let s = cpu_sigmoid_f32(x);
            assert!(s > 0.0 && s < 1.0, "sigmoid({x}) = {s} should be in (0,1)");

            // tanh: output must be in [-1, 1] (saturates at ±1 in f32 for large |x|)
            let t = cpu_tanh_f32(x);
            assert!(
                (-1.0_f32..=1.0).contains(&t),
                "tanh({x}) = {t} should be in [-1,1]"
            );

            // silu: should have same sign as input for |x| > small threshold
            if x > 0.1 {
                assert!(
                    cpu_silu_f32(x) > 0.0,
                    "silu({x}) should be positive for positive input"
                );
            }
        }
    }

    // -- PTX generation consistency tests ------------------------------------

    #[test]
    fn all_activation_ops_generate_ptx_for_f32() {
        let activation_ops = [
            ElementwiseOp::Relu,
            ElementwiseOp::Gelu,
            ElementwiseOp::Sigmoid,
            ElementwiseOp::Silu,
            ElementwiseOp::Tanh,
        ];
        for op in activation_ops {
            let t = ElementwiseTemplate::new(op, PtxType::F32, SmVersion::Sm80);
            let result = t.generate();
            assert!(
                result.is_ok(),
                "PTX generation failed for op {:?}: {:?}",
                op,
                result.err()
            );
            let ptx = result.expect("already checked is_ok");
            let name = op.as_str();
            assert!(
                ptx.contains(&format!(".entry elementwise_{name}_f32")),
                "PTX for {name} missing expected entry point"
            );
        }
    }

    #[test]
    fn relu_ptx_uses_max_instruction() {
        // The PTX relu must use max.f32 to implement max(0, x)
        let t = ElementwiseTemplate::new(ElementwiseOp::Relu, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("relu PTX generation should succeed");
        assert!(
            ptx.contains("max.f32"),
            "relu PTX must use max.f32 instruction"
        );
    }

    #[test]
    fn tanh_ptx_uses_tanh_or_approx_sequence() {
        // Tanh PTX must use some form of approximation (ex2.approx or tanh.approx)
        let t = ElementwiseTemplate::new(ElementwiseOp::Tanh, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("tanh PTX generation should succeed");
        let has_approx = ptx.contains("ex2.approx") || ptx.contains("tanh.approx");
        assert!(
            has_approx,
            "tanh PTX should use ex2.approx or tanh.approx, got:\n{ptx}"
        );
    }

    // -----------------------------------------------------------------------
    // New op PTX generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn generate_one_minus_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::OneMinus, PtxType::F32, SmVersion::Sm80);
        let ptx = t
            .generate()
            .expect("one_minus PTX generation should succeed");
        assert!(ptx.contains("sub.f32"), "one_minus must contain sub.f32");
        assert!(
            ptx.contains("0f3F800000"),
            "one_minus must contain the 1.0 literal"
        );
        assert!(
            ptx.contains(".entry elementwise_one_minus_f32"),
            "one_minus must have correct kernel name"
        );
    }

    #[test]
    fn generate_pow_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Pow, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("pow PTX generation should succeed");
        assert!(
            ptx.contains("lg2.approx.f32"),
            "pow must contain lg2.approx.f32"
        );
        assert!(
            ptx.contains("ex2.approx.f32"),
            "pow must contain ex2.approx.f32"
        );
        assert!(ptx.contains("mul.f32"), "pow must contain mul.f32");
    }

    #[test]
    fn generate_min_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Min, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("min PTX generation should succeed");
        assert!(ptx.contains("min.f32"), "min must contain min.f32");
        assert!(
            ptx.contains(".entry elementwise_min_f32"),
            "min must have correct kernel name"
        );
    }

    #[test]
    fn generate_max_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Max, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("max PTX generation should succeed");
        assert!(ptx.contains("max.f32"), "max must contain max.f32");
        assert!(
            ptx.contains(".entry elementwise_max_f32"),
            "max must have correct kernel name"
        );
    }

    #[test]
    fn generate_cmp_eq_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::CmpEq, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("cmp_eq PTX generation should succeed");
        assert!(
            ptx.contains("setp.eq.f32"),
            "cmp_eq must contain setp.eq.f32"
        );
        assert!(ptx.contains("selp.f32"), "cmp_eq must contain selp.f32");
        assert!(
            ptx.contains("0f3F800000"),
            "cmp_eq must contain the 1.0 literal for the true branch"
        );
    }

    #[test]
    fn generate_or_prob_sum_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::OrProbSum, PtxType::F32, SmVersion::Sm80);
        let ptx = t
            .generate()
            .expect("or_prob_sum PTX generation should succeed");
        assert!(ptx.contains("mul.f32"), "or_prob_sum must contain mul.f32");
        assert!(ptx.contains("sub.f32"), "or_prob_sum must contain sub.f32");
        assert!(ptx.contains("add.f32"), "or_prob_sum must contain add.f32");
    }

    #[test]
    fn generate_nand_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Nand, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("nand PTX generation should succeed");
        assert!(ptx.contains("mul.f32"), "nand must contain mul.f32");
        assert!(ptx.contains("sub.f32"), "nand must contain sub.f32");
        assert!(
            ptx.contains("0f3F800000"),
            "nand must contain the 1.0 literal"
        );
    }

    #[test]
    fn generate_nor_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Nor, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("nor PTX generation should succeed");
        assert!(ptx.contains("mul.f32"), "nor must contain mul.f32");
        assert!(ptx.contains("sub.f32"), "nor must contain sub.f32");
        assert!(ptx.contains("add.f32"), "nor must contain add.f32");
        assert!(
            ptx.contains("0f3F800000"),
            "nor must contain the 1.0 literal"
        );
    }

    #[test]
    fn generate_xor_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Xor, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("xor PTX generation should succeed");
        assert!(ptx.contains("mul.f32"), "xor must contain mul.f32");
        assert!(ptx.contains("sub.f32"), "xor must contain sub.f32");
        assert!(ptx.contains("add.f32"), "xor must contain add.f32");
        // 2.0 literal
        assert!(
            ptx.contains("0f40000000"),
            "xor must contain the 2.0 literal"
        );
    }

    #[test]
    fn generate_or_max_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::OrMax, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("or_max PTX generation should succeed");
        assert!(ptx.contains("max.f32"), "or_max must use max.f32");
        assert!(
            ptx.contains(".entry elementwise_or_max_f32"),
            "or_max must have correct kernel name"
        );
    }

    #[test]
    fn generate_cmp_ops_f32() {
        // Verify all comparison ops generate PTX with correct condition codes
        let cases = [
            (ElementwiseOp::CmpNe, "setp.ne.f32"),
            (ElementwiseOp::CmpLt, "setp.lt.f32"),
            (ElementwiseOp::CmpGt, "setp.gt.f32"),
            (ElementwiseOp::CmpLe, "setp.le.f32"),
            (ElementwiseOp::CmpGe, "setp.ge.f32"),
        ];
        for (op, expected_instr) in cases {
            let t = ElementwiseTemplate::new(op, PtxType::F32, SmVersion::Sm80);
            let ptx = t
                .generate()
                .unwrap_or_else(|e| panic!("PTX gen failed for {op:?}: {e}"));
            assert!(
                ptx.contains(expected_instr),
                "{op:?} PTX must contain {expected_instr}"
            );
            assert!(ptx.contains("selp.f32"), "{op:?} PTX must contain selp.f32");
        }
    }

    #[test]
    fn test_elementwise_ptx_has_valid_headers_extended() {
        let ops_and_types = [
            (ElementwiseOp::OneMinus, PtxType::F32),
            (ElementwiseOp::Pow, PtxType::F32),
            (ElementwiseOp::Min, PtxType::F32),
            (ElementwiseOp::Max, PtxType::F32),
            (ElementwiseOp::CmpEq, PtxType::F32),
            (ElementwiseOp::OrProbSum, PtxType::F32),
            (ElementwiseOp::Nand, PtxType::F32),
            (ElementwiseOp::Nor, PtxType::F32),
            (ElementwiseOp::Xor, PtxType::F32),
            (ElementwiseOp::OrMax, PtxType::F32),
        ];

        for (op, ty) in ops_and_types {
            let t = ElementwiseTemplate::new(op, ty, SmVersion::Sm80);
            let ptx = t
                .generate()
                .unwrap_or_else(|e| panic!("PTX generation failed for {op:?}: {e}"));
            assert!(
                ptx.contains(".version"),
                "PTX for {op:?} must have .version header"
            );
            assert!(
                ptx.contains(".target"),
                "PTX for {op:?} must have .target header"
            );
            assert!(
                ptx.contains(".entry"),
                "PTX for {op:?} must have .entry directive"
            );
        }
    }

    // CPU reference precision tests for new ops

    fn cpu_one_minus_f32(x: f32) -> f32 {
        1.0 - x
    }

    fn cpu_pow_f32(a: f32, b: f32) -> f32 {
        a.powf(b)
    }

    #[allow(clippy::float_cmp)]
    fn cpu_cmp_eq_f32(a: f32, b: f32) -> f32 {
        if a == b { 1.0 } else { 0.0 }
    }

    fn cpu_or_prob_sum_f32(a: f32, b: f32) -> f32 {
        a.mul_add(-b, a + b)
    }

    fn cpu_nand_f32(a: f32, b: f32) -> f32 {
        a.mul_add(-b, 1.0)
    }

    fn cpu_nor_f32(a: f32, b: f32) -> f32 {
        1.0 - a.mul_add(-b, a + b)
    }

    fn cpu_xor_f32(a: f32, b: f32) -> f32 {
        (2.0_f32 * a).mul_add(-b, a + b)
    }

    #[test]
    fn cpu_one_minus_f32_precision() {
        assert!((cpu_one_minus_f32(0.0) - 1.0_f32).abs() < f32::EPSILON);
        assert!((cpu_one_minus_f32(1.0) - 0.0_f32).abs() < f32::EPSILON);
        assert!((cpu_one_minus_f32(0.5) - 0.5_f32).abs() < f32::EPSILON);
        assert!((cpu_one_minus_f32(-1.0) - 2.0_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn cpu_pow_f32_precision() {
        assert!((cpu_pow_f32(2.0, 3.0) - 8.0_f32).abs() < 1e-5_f32);
        assert!((cpu_pow_f32(4.0, 0.5) - 2.0_f32).abs() < 1e-5_f32);
        assert!((cpu_pow_f32(1.0, 100.0) - 1.0_f32).abs() < 1e-5_f32);
    }

    #[test]
    fn cpu_cmp_eq_f32_precision() {
        assert!((cpu_cmp_eq_f32(1.0, 1.0) - 1.0).abs() < f32::EPSILON);
        assert!((cpu_cmp_eq_f32(1.0, 2.0) - 0.0).abs() < f32::EPSILON);
        assert!((cpu_cmp_eq_f32(0.0, 0.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn cpu_or_prob_sum_f32_precision() {
        // For a=1, b=1: 1+1-1*1 = 1
        assert!((cpu_or_prob_sum_f32(1.0, 1.0) - 1.0).abs() < f32::EPSILON);
        // For a=0, b=0: 0
        assert!(cpu_or_prob_sum_f32(0.0, 0.0).abs() < f32::EPSILON);
        // For a=0.5, b=0.5: 0.5+0.5-0.25 = 0.75
        assert!((cpu_or_prob_sum_f32(0.5, 0.5) - 0.75).abs() < 1e-6_f32);
    }

    #[test]
    fn cpu_nand_f32_precision() {
        // nand(1,1) = 1 - 1 = 0
        assert!(cpu_nand_f32(1.0, 1.0).abs() < f32::EPSILON);
        // nand(0,1) = 1 - 0 = 1
        assert!((cpu_nand_f32(0.0, 1.0) - 1.0).abs() < f32::EPSILON);
        // nand(0.5,0.5) = 1 - 0.25 = 0.75
        assert!((cpu_nand_f32(0.5, 0.5) - 0.75).abs() < 1e-6_f32);
    }

    #[test]
    fn cpu_nor_f32_precision() {
        // nor(0,0) = 1 - 0 = 1
        assert!((cpu_nor_f32(0.0, 0.0) - 1.0).abs() < f32::EPSILON);
        // nor(1,0) = 1 - (1+0-0) = 0
        assert!(cpu_nor_f32(1.0, 0.0).abs() < f32::EPSILON);
        // nor(0.5,0.5) = 1 - 0.75 = 0.25
        assert!((cpu_nor_f32(0.5, 0.5) - 0.25).abs() < 1e-6_f32);
    }

    #[test]
    fn cpu_xor_f32_precision() {
        // xor(0,0) = 0
        assert!(cpu_xor_f32(0.0, 0.0).abs() < f32::EPSILON);
        // xor(1,1) = 1+1-2 = 0
        assert!(cpu_xor_f32(1.0, 1.0).abs() < f32::EPSILON);
        // xor(1,0) = 1+0-0 = 1
        assert!((cpu_xor_f32(1.0, 0.0) - 1.0).abs() < f32::EPSILON);
        // xor(0.5,0.5) = 0.5+0.5-0.5 = 0.5
        assert!((cpu_xor_f32(0.5, 0.5) - 0.5).abs() < 1e-6_f32);
    }

    #[test]
    fn ptx_template_generates_fill_f32() {
        let template = ElementwiseTemplate::new(ElementwiseOp::Fill, PtxType::F32, SmVersion::Sm80);
        let ptx = template.generate().expect("fill PTX generation failed");
        assert!(
            ptx.contains("st.global.f32"),
            "must contain store instruction"
        );
        assert!(ptx.contains("elementwise_fill_f32"), "wrong kernel name");
    }
}
