//! Shared PTX code-generation utilities for OxiCUDA Primitives.
//!
//! This module provides helper functions and types used across the warp, block,
//! device, and sort sub-modules to generate PTX kernels for parallel primitives.
//!
//! # Design
//!
//! PTX kernels for all parallel primitives are generated at runtime using the
//! `oxicuda-ptx` DSL and then JIT-compiled via `cuModuleLoadData`.  This means:
//!
//! * Zero build-time CUDA SDK dependency
//! * No `nvcc` required
//! * Each kernel variant is generated the first time it is needed

use oxicuda_ptx::{
    arch::SmVersion,
    ir::{PtxType, Register},
};

// ─── Numeric type trait ──────────────────────────────────────────────────────

/// Types that can be used as primitive element types.
///
/// Implementations are provided for `u32`, `u64`, `i32`, `i64`, `f32`, and
/// `f64`.  Each type maps to a PTX `.type` qualifier used in kernel generation.
pub trait PrimitiveType: Copy + Send + Sync + 'static {
    /// The corresponding PTX type for kernel generation.
    fn ptx_type() -> PtxType;

    /// A human-readable suffix used in kernel names (e.g. `"f32"`, `"u32"`).
    fn type_suffix() -> &'static str;

    /// The binary representation (in bytes).
    fn size_bytes() -> usize {
        std::mem::size_of::<Self>()
    }

    /// Returns the zero / identity value for this type, serialised as a PTX
    /// immediate string (e.g. `"0"`, `"0.0"`).
    fn ptx_zero() -> &'static str;

    /// Returns the maximum value for this type as a PTX immediate string,
    /// used as the identity element for min-reductions.
    fn ptx_max() -> &'static str;

    /// Returns the minimum value for this type as a PTX immediate string,
    /// used as the identity element for max-reductions.
    fn ptx_min() -> &'static str;
}

macro_rules! impl_primitive_type {
    ($t:ty, $ptx:expr, $suffix:literal, $zero:literal, $max:literal, $min:literal) => {
        impl PrimitiveType for $t {
            fn ptx_type() -> PtxType {
                $ptx
            }
            fn type_suffix() -> &'static str {
                $suffix
            }
            fn ptx_zero() -> &'static str {
                $zero
            }
            fn ptx_max() -> &'static str {
                $max
            }
            fn ptx_min() -> &'static str {
                $min
            }
        }
    };
}

impl_primitive_type!(u32, PtxType::U32, "u32", "0", "4294967295", "0");
impl_primitive_type!(u64, PtxType::U64, "u64", "0", "18446744073709551615", "0");
impl_primitive_type!(i32, PtxType::S32, "i32", "0", "2147483647", "-2147483648");
impl_primitive_type!(
    i64,
    PtxType::S64,
    "i64",
    "0",
    "9223372036854775807",
    "-9223372036854775808"
);
impl_primitive_type!(
    f32,
    PtxType::F32,
    "f32",
    "0f00000000", // 0.0 as PTX float literal
    "0x7F800000", // +inf as hex float
    "0xFF800000"  // -inf as hex float
);
impl_primitive_type!(
    f64,
    PtxType::F64,
    "f64",
    "0d0000000000000000", // 0.0 as PTX double literal
    "0x7FF0000000000000", // +inf
    "0xFFF0000000000000"  // -inf
);

// ─── Reduction operation ─────────────────────────────────────────────────────

/// Arithmetic operation performed by a reduction or scan primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Summation (identity = 0).
    Sum,
    /// Product (identity = 1).
    Product,
    /// Minimum (identity = MAX).
    Min,
    /// Maximum (identity = MIN).
    Max,
    /// Bitwise AND (identity = all-ones).
    And,
    /// Bitwise OR (identity = all-zeros).
    Or,
    /// Bitwise XOR (identity = all-zeros).
    Xor,
}

impl ReduceOp {
    /// PTX instruction mnemonic for this operation on the given type.
    ///
    /// E.g. `Sum` on `f32` → `"add.f32"`.
    pub fn ptx_instr(self, ty: PtxType) -> &'static str {
        match (self, ty) {
            (Self::Sum, PtxType::F32) => "add.f32",
            (Self::Sum, PtxType::F64) => "add.f64",
            (Self::Sum, PtxType::U32 | PtxType::S32) => "add.s32",
            (Self::Sum, PtxType::U64 | PtxType::S64) => "add.s64",
            (Self::Product, PtxType::F32) => "mul.f32",
            (Self::Product, PtxType::F64) => "mul.f64",
            (Self::Product, PtxType::U32 | PtxType::S32) => "mul.lo.s32",
            (Self::Product, PtxType::U64 | PtxType::S64) => "mul.lo.s64",
            (Self::Min, PtxType::F32) => "min.f32",
            (Self::Min, PtxType::F64) => "min.f64",
            (Self::Min, PtxType::U32) => "min.u32",
            (Self::Min, PtxType::U64) => "min.u64",
            (Self::Min, PtxType::S32) => "min.s32",
            (Self::Min, PtxType::S64) => "min.s64",
            (Self::Max, PtxType::F32) => "max.f32",
            (Self::Max, PtxType::F64) => "max.f64",
            (Self::Max, PtxType::U32) => "max.u32",
            (Self::Max, PtxType::U64) => "max.u64",
            (Self::Max, PtxType::S32) => "max.s32",
            (Self::Max, PtxType::S64) => "max.s64",
            (Self::And, _) => "and.b32",
            (Self::Or, _) => "or.b32",
            (Self::Xor, _) => "xor.b32",
            _ => "add.s32", // fallback
        }
    }

    /// PTX immediate literal for the identity element of this operation.
    pub fn identity_literal<T: PrimitiveType>(self) -> &'static str {
        match self {
            Self::Sum | Self::Or | Self::Xor => T::ptx_zero(),
            Self::Product => "1",
            Self::Min => T::ptx_max(),
            Self::Max => T::ptx_min(),
            Self::And => "0xFFFFFFFF",
        }
    }

    /// Short lowercase name for use in generated kernel names.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Product => "prod",
            Self::Min => "min",
            Self::Max => "max",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
        }
    }
}

// ─── PTX header builder ──────────────────────────────────────────────────────

/// Build a standard PTX header string for the given SM version.
///
/// The PTX ISA version is matched to the SM family to ensure the generated
/// PTX uses available instructions.
#[must_use]
pub fn ptx_header(sm: SmVersion) -> String {
    let (ptx_major, ptx_minor) = ptx_version_for(sm);
    let sm_str = sm.as_ptx_str();
    format!(".version {ptx_major}.{ptx_minor}\n.target {sm_str}\n.address_size 64\n\n")
}

/// Map an [`SmVersion`] to the recommended PTX ISA version.
fn ptx_version_for(sm: SmVersion) -> (u32, u32) {
    match sm {
        SmVersion::Sm75 => (7, 4),
        SmVersion::Sm80 | SmVersion::Sm86 => (7, 5),
        SmVersion::Sm89 => (8, 0),
        SmVersion::Sm90 | SmVersion::Sm90a => (8, 0),
        SmVersion::Sm100 => (8, 5),
        SmVersion::Sm120 => (8, 7),
    }
}

/// Numeric SM version suffix (e.g. `80` for `Sm80`).
///
/// For `Sm90a` returns `90` — use [`ptx_header`] which calls [`SmVersion::as_ptx_str`]
/// when you need the full PTX target string (e.g. `"sm_90a"`).
pub fn sm_number(sm: SmVersion) -> u32 {
    match sm {
        SmVersion::Sm75 => 75,
        SmVersion::Sm80 => 80,
        SmVersion::Sm86 => 86,
        SmVersion::Sm89 => 89,
        SmVersion::Sm90 | SmVersion::Sm90a => 90,
        SmVersion::Sm100 => 100,
        SmVersion::Sm120 => 120,
    }
}

// ─── Register naming helpers ─────────────────────────────────────────────────

/// Build the PTX `.reg` declaration for a given register name and type.
///
/// E.g. `reg_decl("val", PtxType::F32, 4)` → `".reg .f32 %val<4>;"`.
#[must_use]
pub fn reg_decl(name: &str, ty: PtxType, count: usize) -> String {
    let type_str = ptx_type_str(ty);
    if count == 1 {
        format!("    .reg .{type_str} %{name};")
    } else {
        format!("    .reg .{type_str} %{name}<{count}>;")
    }
}

/// Map a [`PtxType`] to its PTX string representation.
#[must_use]
pub fn ptx_type_str(ty: PtxType) -> &'static str {
    match ty {
        PtxType::U8 => "u8",
        PtxType::U16 => "u16",
        PtxType::U32 => "u32",
        PtxType::U64 => "u64",
        PtxType::S8 => "s8",
        PtxType::S16 => "s16",
        PtxType::S32 => "s32",
        PtxType::S64 => "s64",
        PtxType::F16 => "f16",
        PtxType::F16x2 => "f16x2",
        PtxType::BF16 => "bf16",
        PtxType::BF16x2 => "bf16x2",
        PtxType::F32 => "f32",
        PtxType::F64 => "f64",
        PtxType::TF32 => "tf32",
        PtxType::E4M3 => "e4m3",
        PtxType::E5M2 => "e5m2",
        PtxType::E2M3 => "e2m3",
        PtxType::E3M2 => "e3m2",
        PtxType::E2M1 => "e2m1",
        PtxType::B8 => "b8",
        PtxType::B16 => "b16",
        PtxType::B32 => "b32",
        PtxType::B64 => "b64",
        PtxType::B128 => "b128",
        PtxType::Pred => "pred",
    }
}

/// Size of a PTX type in bytes.
#[must_use]
pub fn ptx_type_bytes(ty: PtxType) -> usize {
    match ty {
        PtxType::U8
        | PtxType::S8
        | PtxType::B8
        | PtxType::E4M3
        | PtxType::E5M2
        | PtxType::E2M3
        | PtxType::E3M2
        | PtxType::E2M1
        | PtxType::Pred => 1,
        PtxType::U16 | PtxType::S16 | PtxType::F16 | PtxType::BF16 | PtxType::B16 => 2,
        PtxType::U32
        | PtxType::S32
        | PtxType::F32
        | PtxType::F16x2
        | PtxType::BF16x2
        | PtxType::TF32
        | PtxType::B32 => 4,
        PtxType::U64 | PtxType::S64 | PtxType::F64 | PtxType::B64 => 8,
        PtxType::B128 => 16,
    }
}

/// Dummy usage to avoid dead-code warnings on Register in generated code.
#[allow(dead_code)]
pub(crate) fn _use_register(_r: Register) {}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_header_sm80() {
        let h = ptx_header(SmVersion::Sm80);
        assert!(h.contains(".version 7.5"), "header: {h}");
        assert!(h.contains(".target sm_80"), "header: {h}");
        assert!(h.contains(".address_size 64"), "header: {h}");
    }

    #[test]
    fn ptx_header_sm90() {
        let h = ptx_header(SmVersion::Sm90);
        assert!(h.contains(".target sm_90"), "header: {h}");
    }

    #[test]
    fn ptx_header_sm75() {
        let h = ptx_header(SmVersion::Sm75);
        assert!(h.contains(".version 7.4"), "header: {h}");
        assert!(h.contains(".target sm_75"), "header: {h}");
    }

    #[test]
    fn reduce_op_names_unique() {
        use std::collections::HashSet;
        let names: HashSet<_> = [
            ReduceOp::Sum,
            ReduceOp::Product,
            ReduceOp::Min,
            ReduceOp::Max,
            ReduceOp::And,
            ReduceOp::Or,
            ReduceOp::Xor,
        ]
        .iter()
        .map(|op| op.name())
        .collect();
        assert_eq!(names.len(), 7, "every ReduceOp must have a unique name");
    }

    #[test]
    fn reduce_op_sum_f32_instr() {
        assert_eq!(ReduceOp::Sum.ptx_instr(PtxType::F32), "add.f32");
    }

    #[test]
    fn reduce_op_min_u32_instr() {
        assert_eq!(ReduceOp::Min.ptx_instr(PtxType::U32), "min.u32");
    }

    #[test]
    fn primitive_type_suffix_u32() {
        assert_eq!(u32::type_suffix(), "u32");
        assert_eq!(f32::type_suffix(), "f32");
        assert_eq!(i64::type_suffix(), "i64");
    }

    #[test]
    fn primitive_type_ptx_type() {
        assert_eq!(u32::ptx_type(), PtxType::U32);
        assert_eq!(f32::ptx_type(), PtxType::F32);
        assert_eq!(f64::ptx_type(), PtxType::F64);
    }

    #[test]
    fn reg_decl_single() {
        let decl = reg_decl("acc", PtxType::F32, 1);
        assert_eq!(decl, "    .reg .f32 %acc;");
    }

    #[test]
    fn reg_decl_array() {
        let decl = reg_decl("tmp", PtxType::U32, 4);
        assert_eq!(decl, "    .reg .u32 %tmp<4>;");
    }

    #[test]
    fn ptx_type_bytes_correct() {
        assert_eq!(ptx_type_bytes(PtxType::F32), 4);
        assert_eq!(ptx_type_bytes(PtxType::F64), 8);
        assert_eq!(ptx_type_bytes(PtxType::U32), 4);
        assert_eq!(ptx_type_bytes(PtxType::F16), 2);
    }

    #[test]
    fn ptx_type_bytes_new_variants() {
        // Sub-byte and 1-byte FP8/FP6/FP4 types
        assert_eq!(ptx_type_bytes(PtxType::E4M3), 1);
        assert_eq!(ptx_type_bytes(PtxType::E5M2), 1);
        assert_eq!(ptx_type_bytes(PtxType::E2M3), 1);
        assert_eq!(ptx_type_bytes(PtxType::E3M2), 1);
        assert_eq!(ptx_type_bytes(PtxType::E2M1), 1);
        assert_eq!(ptx_type_bytes(PtxType::B8), 1);

        // Packed 16-bit types
        assert_eq!(ptx_type_bytes(PtxType::BF16), 2);
        assert_eq!(ptx_type_bytes(PtxType::B16), 2);

        // Packed 32-bit types
        assert_eq!(ptx_type_bytes(PtxType::F16x2), 4);
        assert_eq!(ptx_type_bytes(PtxType::BF16x2), 4);
        assert_eq!(ptx_type_bytes(PtxType::TF32), 4);
        assert_eq!(ptx_type_bytes(PtxType::B32), 4);

        // 128-bit type
        assert_eq!(ptx_type_bytes(PtxType::B128), 16);
    }

    #[test]
    fn ptx_type_str_new_variants() {
        assert_eq!(ptx_type_str(PtxType::U8), "u8");
        assert_eq!(ptx_type_str(PtxType::S8), "s8");
        assert_eq!(ptx_type_str(PtxType::F16x2), "f16x2");
        assert_eq!(ptx_type_str(PtxType::BF16), "bf16");
        assert_eq!(ptx_type_str(PtxType::BF16x2), "bf16x2");
        assert_eq!(ptx_type_str(PtxType::TF32), "tf32");
        assert_eq!(ptx_type_str(PtxType::E4M3), "e4m3");
        assert_eq!(ptx_type_str(PtxType::E5M2), "e5m2");
        assert_eq!(ptx_type_str(PtxType::E2M3), "e2m3");
        assert_eq!(ptx_type_str(PtxType::E3M2), "e3m2");
        assert_eq!(ptx_type_str(PtxType::E2M1), "e2m1");
        assert_eq!(ptx_type_str(PtxType::B8), "b8");
        assert_eq!(ptx_type_str(PtxType::B128), "b128");
        assert_eq!(ptx_type_str(PtxType::Pred), "pred");
    }

    #[test]
    fn ptx_header_sm90a_produces_sm_90a_target() {
        let h = ptx_header(SmVersion::Sm90a);
        assert!(
            h.contains(".target sm_90a"),
            "Sm90a must produce sm_90a target: {h}"
        );
        assert!(h.contains(".version 8.0"), "Sm90a header: {h}");
    }

    #[test]
    fn ptx_header_sm100_sm120() {
        let h100 = ptx_header(SmVersion::Sm100);
        assert!(h100.contains(".target sm_100"), "Sm100: {h100}");
        assert!(h100.contains(".version 8.5"), "Sm100 version: {h100}");

        let h120 = ptx_header(SmVersion::Sm120);
        assert!(h120.contains(".target sm_120"), "Sm120: {h120}");
        assert!(h120.contains(".version 8.7"), "Sm120 version: {h120}");
    }

    #[test]
    fn sm_number_values() {
        assert_eq!(sm_number(SmVersion::Sm80), 80);
        assert_eq!(sm_number(SmVersion::Sm90), 90);
        assert_eq!(sm_number(SmVersion::Sm120), 120);
    }

    #[test]
    fn sm_number_all_variants() {
        assert_eq!(sm_number(SmVersion::Sm75), 75);
        assert_eq!(sm_number(SmVersion::Sm80), 80);
        assert_eq!(sm_number(SmVersion::Sm86), 86);
        assert_eq!(sm_number(SmVersion::Sm89), 89);
        assert_eq!(sm_number(SmVersion::Sm90), 90);
        assert_eq!(sm_number(SmVersion::Sm90a), 90);
        assert_eq!(sm_number(SmVersion::Sm100), 100);
        assert_eq!(sm_number(SmVersion::Sm120), 120);
    }

    #[test]
    fn reduce_op_identity_literals() {
        // Sum/Or/Xor → zero
        assert_eq!(ReduceOp::Sum.identity_literal::<u32>(), "0");
        assert_eq!(ReduceOp::Or.identity_literal::<u32>(), "0");
        assert_eq!(ReduceOp::Xor.identity_literal::<u32>(), "0");
        // Product → 1
        assert_eq!(ReduceOp::Product.identity_literal::<u32>(), "1");
        // Min → max value; Max → min value
        assert_eq!(ReduceOp::Min.identity_literal::<u32>(), "4294967295");
        assert_eq!(ReduceOp::Max.identity_literal::<u32>(), "0");
        // And → all-ones
        assert_eq!(ReduceOp::And.identity_literal::<u32>(), "0xFFFFFFFF");
    }

    #[test]
    fn reduce_op_product_instr() {
        assert_eq!(ReduceOp::Product.ptx_instr(PtxType::F32), "mul.f32");
        assert_eq!(ReduceOp::Product.ptx_instr(PtxType::U32), "mul.lo.s32");
    }

    #[test]
    fn reduce_op_max_variants() {
        assert_eq!(ReduceOp::Max.ptx_instr(PtxType::U32), "max.u32");
        assert_eq!(ReduceOp::Max.ptx_instr(PtxType::U64), "max.u64");
        assert_eq!(ReduceOp::Max.ptx_instr(PtxType::S32), "max.s32");
        assert_eq!(ReduceOp::Max.ptx_instr(PtxType::S64), "max.s64");
        assert_eq!(ReduceOp::Max.ptx_instr(PtxType::F64), "max.f64");
    }
}
