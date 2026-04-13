//! PTX type system and supporting enumerations.
//!
//! This module defines the full set of PTX data types as specified in the PTX ISA,
//! including integer, floating-point, bit-width, and predicate types. It also
//! provides enumerations for rounding modes, comparison operators, memory spaces,
//! and special registers used throughout the IR.

use std::fmt;

/// PTX data types as defined in the PTX ISA.
///
/// Covers unsigned/signed integers, all floating-point widths (including FP8/FP6/FP4
/// formats introduced in Hopper and Blackwell architectures), untyped bit-width types,
/// and predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxType {
    // Unsigned integers
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer.
    U16,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    // Signed integers
    /// 8-bit signed integer.
    S8,
    /// 16-bit signed integer.
    S16,
    /// 32-bit signed integer.
    S32,
    /// 64-bit signed integer.
    S64,
    // Floating point
    /// IEEE 754 half-precision (16-bit) float.
    F16,
    /// Packed pair of half-precision floats.
    F16x2,
    /// Brain floating-point (16-bit, 8-bit exponent).
    BF16,
    /// Packed pair of BF16 floats.
    BF16x2,
    /// IEEE 754 single-precision (32-bit) float.
    F32,
    /// IEEE 754 double-precision (64-bit) float.
    F64,
    // Special floating point
    /// TensorFloat-32 (19-bit, used in Tensor Cores).
    TF32,
    /// FP8 E4M3 format (Hopper+).
    E4M3,
    /// FP8 E5M2 format (Hopper+).
    E5M2,
    /// FP6 E2M3 format (Blackwell).
    E2M3,
    /// FP6 E3M2 format (Blackwell).
    E3M2,
    /// FP4 E2M1 format (Blackwell).
    E2M1,
    // Bit-width types (untyped)
    /// 8-bit untyped.
    B8,
    /// 16-bit untyped.
    B16,
    /// 32-bit untyped.
    B32,
    /// 64-bit untyped.
    B64,
    /// 128-bit untyped.
    B128,
    // Predicate
    /// 1-bit predicate register type.
    Pred,
}

impl PtxType {
    /// Returns the PTX ISA string representation of this type (e.g., `".f32"`, `".u64"`).
    #[must_use]
    pub const fn as_ptx_str(&self) -> &'static str {
        match self {
            Self::U8 => ".u8",
            Self::U16 => ".u16",
            Self::U32 => ".u32",
            Self::U64 => ".u64",
            Self::S8 => ".s8",
            Self::S16 => ".s16",
            Self::S32 => ".s32",
            Self::S64 => ".s64",
            Self::F16 => ".f16",
            Self::F16x2 => ".f16x2",
            Self::BF16 => ".bf16",
            Self::BF16x2 => ".bf16x2",
            Self::F32 => ".f32",
            Self::F64 => ".f64",
            Self::TF32 => ".tf32",
            Self::E4M3 => ".e4m3",
            Self::E5M2 => ".e5m2",
            Self::E2M3 => ".e2m3",
            Self::E3M2 => ".e3m2",
            Self::E2M1 => ".e2m1",
            Self::B8 => ".b8",
            Self::B16 => ".b16",
            Self::B32 => ".b32",
            Self::B64 => ".b64",
            Self::B128 => ".b128",
            Self::Pred => ".pred",
        }
    }

    /// Returns the size in bytes of a single value of this type.
    ///
    /// Packed types (e.g., `F16x2`) return the size of the packed value.
    /// Predicates return 1 byte (the minimum addressable unit).
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        match self {
            Self::U8 | Self::S8 | Self::B8 | Self::E4M3 | Self::E5M2 | Self::E2M1 | Self::Pred => 1,
            Self::U16
            | Self::S16
            | Self::F16
            | Self::BF16
            | Self::B16
            | Self::E2M3
            | Self::E3M2 => 2,
            Self::U32
            | Self::S32
            | Self::F32
            | Self::F16x2
            | Self::BF16x2
            | Self::B32
            | Self::TF32 => 4,
            Self::U64 | Self::S64 | Self::F64 | Self::B64 => 8,
            Self::B128 => 16,
        }
    }

    /// Returns the register-width class type used in `.reg` declarations.
    ///
    /// PTX uses register classes based on width: 16-bit, 32-bit, 64-bit, and predicate.
    /// Sub-32-bit types are promoted to 32-bit registers; 128-bit uses 64-bit pairs.
    #[must_use]
    pub const fn reg_type(&self) -> Self {
        match self {
            Self::Pred => Self::Pred,
            Self::F64 | Self::U64 | Self::S64 | Self::B64 => Self::B64,
            Self::B128 => Self::B128,
            Self::F16 | Self::BF16 | Self::U16 | Self::S16 | Self::B16 => Self::B16,
            _ => Self::B32,
        }
    }

    /// Returns `true` if this is an integer type (signed or unsigned).
    #[must_use]
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::U8
                | Self::U16
                | Self::U32
                | Self::U64
                | Self::S8
                | Self::S16
                | Self::S32
                | Self::S64
        )
    }

    /// Returns `true` if this is a floating-point type (including packed and special formats).
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            Self::F16
                | Self::F16x2
                | Self::BF16
                | Self::BF16x2
                | Self::F32
                | Self::F64
                | Self::TF32
                | Self::E4M3
                | Self::E5M2
                | Self::E2M3
                | Self::E3M2
                | Self::E2M1
        )
    }

    /// Returns the bit-width of a single element of this type.
    ///
    /// For sub-byte types (E2M1 = FP4), returns 4. For packed types like
    /// `F16x2` and `BF16x2`, returns the total packed width (32 bits).
    /// Predicates are reported as 1 bit.
    #[must_use]
    pub const fn bit_width(&self) -> u32 {
        match self {
            // Sub-byte: FP4 (E2M1)
            Self::E2M1 => 4,
            // 6-bit types (stored in 8-bit containers but logically 6 bits)
            Self::E2M3 | Self::E3M2 => 6,
            // 8-bit types
            Self::U8 | Self::S8 | Self::B8 | Self::E4M3 | Self::E5M2 => 8,
            // Predicate (1 bit)
            Self::Pred => 1,
            // 16-bit types
            Self::U16 | Self::S16 | Self::F16 | Self::BF16 | Self::B16 => 16,
            // 32-bit types (including packed 16-bit pairs)
            Self::U32
            | Self::S32
            | Self::F32
            | Self::F16x2
            | Self::BF16x2
            | Self::B32
            | Self::TF32 => 32,
            // 64-bit types
            Self::U64 | Self::S64 | Self::F64 | Self::B64 => 64,
            // 128-bit types
            Self::B128 => 128,
        }
    }

    /// Returns `true` if this is a signed type (signed integers or all floats).
    #[must_use]
    pub const fn is_signed(&self) -> bool {
        matches!(
            self,
            Self::S8
                | Self::S16
                | Self::S32
                | Self::S64
                | Self::F16
                | Self::F16x2
                | Self::BF16
                | Self::BF16x2
                | Self::F32
                | Self::F64
                | Self::TF32
                | Self::E4M3
                | Self::E5M2
                | Self::E2M3
                | Self::E3M2
                | Self::E2M1
        )
    }
}

impl fmt::Display for PtxType {
    /// Formats the type as its PTX ISA string without the leading dot.
    ///
    /// For example, `PtxType::F32` displays as `"f32"`, and
    /// `PtxType::E2M1` displays as `"e2m1"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // as_ptx_str() returns ".f32" — strip the leading dot for Display
        let s = self.as_ptx_str();
        f.write_str(s.trim_start_matches('.'))
    }
}

/// Atomic operation type for `atom` and `red` instructions.
///
/// These operations are performed atomically on global or shared memory
/// locations, ensuring correctness under concurrent access from multiple threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomOp {
    /// Atomic addition.
    Add,
    /// Atomic minimum.
    Min,
    /// Atomic maximum.
    Max,
    /// Atomic increment (wraps at value).
    Inc,
    /// Atomic decrement (wraps at value).
    Dec,
    /// Atomic bitwise AND.
    And,
    /// Atomic bitwise OR.
    Or,
    /// Atomic bitwise XOR.
    Xor,
    /// Atomic exchange (swap).
    Exch,
}

impl AtomOp {
    /// Returns the PTX modifier string (e.g., `".add"`, `".exch"`).
    #[must_use]
    pub const fn as_ptx_str(&self) -> &'static str {
        match self {
            Self::Add => ".add",
            Self::Min => ".min",
            Self::Max => ".max",
            Self::Inc => ".inc",
            Self::Dec => ".dec",
            Self::And => ".and",
            Self::Or => ".or",
            Self::Xor => ".xor",
            Self::Exch => ".exch",
        }
    }
}

/// Vector width for vectorized load/store operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorWidth {
    /// Scalar (no vectorization).
    V1,
    /// 2-element vector.
    V2,
    /// 4-element vector.
    V4,
}

impl VectorWidth {
    /// Returns the PTX modifier string (e.g., `".v2"`, `".v4"`), or empty for scalar.
    #[must_use]
    pub const fn as_ptx_str(&self) -> &'static str {
        match self {
            Self::V1 => "",
            Self::V2 => ".v2",
            Self::V4 => ".v4",
        }
    }
}

/// IEEE 754 rounding modes for floating-point operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundingMode {
    /// Round to nearest even.
    Rn,
    /// Round towards zero.
    Rz,
    /// Round towards positive infinity.
    Ru,
    /// Round towards negative infinity.
    Rd,
}

impl RoundingMode {
    /// Returns the PTX modifier string (e.g., `".rn"`, `".rz"`).
    #[must_use]
    pub const fn as_ptx_str(&self) -> &'static str {
        match self {
            Self::Rn => ".rn",
            Self::Rz => ".rz",
            Self::Ru => ".ru",
            Self::Rd => ".rd",
        }
    }
}

/// Multiplication mode controlling which portion of the product is retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MulMode {
    /// Low bits of the product (default for same-width result).
    Lo,
    /// High bits of the product.
    Hi,
    /// Wide multiplication (result is twice the input width).
    Wide,
}

impl MulMode {
    /// Returns the PTX modifier string (e.g., `".lo"`, `".hi"`, `".wide"`).
    #[must_use]
    pub const fn as_ptx_str(&self) -> &'static str {
        match self {
            Self::Lo => ".lo",
            Self::Hi => ".hi",
            Self::Wide => ".wide",
        }
    }
}

/// Comparison operators for `setp` and related instructions.
///
/// The first group (Eq..Hs) are ordered comparisons; the second group
/// (Equ..Nan) are unordered comparisons for floating-point NaN handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than (signed).
    Lt,
    /// Less than or equal (signed).
    Le,
    /// Greater than (signed).
    Gt,
    /// Greater than or equal (signed).
    Ge,
    /// Lower (unsigned less than).
    Lo,
    /// Lower or same (unsigned less than or equal).
    Ls,
    /// Higher (unsigned greater than).
    Hi,
    /// Higher or same (unsigned greater than or equal).
    Hs,
    /// Equal (unordered).
    Equ,
    /// Not equal (unordered).
    Neu,
    /// Less than (unordered).
    Ltu,
    /// Less than or equal (unordered).
    Leu,
    /// Greater than (unordered).
    Gtu,
    /// Greater than or equal (unordered).
    Geu,
    /// Numeric (both operands are not NaN).
    Num,
    /// NaN (at least one operand is NaN).
    Nan,
}

impl CmpOp {
    /// Returns the PTX modifier string (e.g., `".eq"`, `".lt"`, `".geu"`).
    #[must_use]
    pub const fn as_ptx_str(&self) -> &'static str {
        match self {
            Self::Eq => ".eq",
            Self::Ne => ".ne",
            Self::Lt => ".lt",
            Self::Le => ".le",
            Self::Gt => ".gt",
            Self::Ge => ".ge",
            Self::Lo => ".lo",
            Self::Ls => ".ls",
            Self::Hi => ".hi",
            Self::Hs => ".hs",
            Self::Equ => ".equ",
            Self::Neu => ".neu",
            Self::Ltu => ".ltu",
            Self::Leu => ".leu",
            Self::Gtu => ".gtu",
            Self::Geu => ".geu",
            Self::Num => ".num",
            Self::Nan => ".nan",
        }
    }
}

/// PTX memory address spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemorySpace {
    /// Global device memory.
    Global,
    /// Shared memory (per-block scratchpad).
    Shared,
    /// Local memory (per-thread, spills to DRAM).
    Local,
    /// Constant memory (read-only, cached).
    Constant,
    /// Parameter memory (kernel arguments).
    Param,
}

impl MemorySpace {
    /// Returns the PTX modifier string (e.g., `".global"`, `".shared"`).
    #[must_use]
    pub const fn as_ptx_str(&self) -> &'static str {
        match self {
            Self::Global => ".global",
            Self::Shared => ".shared",
            Self::Local => ".local",
            Self::Constant => ".const",
            Self::Param => ".param",
        }
    }
}

/// Cache operation qualifiers for load/store instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheQualifier {
    /// No explicit cache qualifier.
    None,
    /// Cache at all levels.
    Ca,
    /// Cache at L2, bypass L1.
    Cg,
    /// Streaming (evict first).
    Cs,
    /// Last use (evict after use).
    Lu,
    /// Volatile (don't cache).
    Cv,
}

impl CacheQualifier {
    /// Returns the PTX modifier string, or empty for `None`.
    #[must_use]
    pub const fn as_ptx_str(&self) -> &'static str {
        match self {
            Self::None => "",
            Self::Ca => ".ca",
            Self::Cg => ".cg",
            Self::Cs => ".cs",
            Self::Lu => ".lu",
            Self::Cv => ".cv",
        }
    }
}

/// Scope for fence and memory ordering instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FenceScope {
    /// CTA (cooperative thread array / block) scope.
    Cta,
    /// GPU (device) scope.
    Gpu,
    /// System scope (across GPUs and host).
    Sys,
}

impl FenceScope {
    /// Returns the PTX modifier string (e.g., `".cta"`, `".gpu"`, `".sys"`).
    #[must_use]
    pub const fn as_ptx_str(&self) -> &'static str {
        match self {
            Self::Cta => ".cta",
            Self::Gpu => ".gpu",
            Self::Sys => ".sys",
        }
    }
}

/// Special registers accessible via `mov.u32` / `mov.u64` in PTX.
///
/// These provide thread identity, block identity, grid dimensions, and
/// hardware state information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialReg {
    /// Thread index X (`%tid.x`).
    TidX,
    /// Thread index Y (`%tid.y`).
    TidY,
    /// Thread index Z (`%tid.z`).
    TidZ,
    /// Block index X (`%ctaid.x`).
    CtaidX,
    /// Block index Y (`%ctaid.y`).
    CtaidY,
    /// Block index Z (`%ctaid.z`).
    CtaidZ,
    /// Block dimension X (`%ntid.x`).
    NtidX,
    /// Block dimension Y (`%ntid.y`).
    NtidY,
    /// Block dimension Z (`%ntid.z`).
    NtidZ,
    /// Grid dimension X (`%nctaid.x`).
    NctaidX,
    /// Grid dimension Y (`%nctaid.y`).
    NctaidY,
    /// Grid dimension Z (`%nctaid.z`).
    NctaidZ,
    /// Warp ID within the CTA.
    WarpId,
    /// Lane ID within the warp (0..31).
    LaneId,
    /// Streaming multiprocessor ID.
    SmId,
    /// 32-bit clock counter.
    Clock,
    /// 64-bit clock counter.
    Clock64,
    /// Dynamic shared memory size in bytes.
    DynamicSmemSize,
}

impl SpecialReg {
    /// Returns the PTX special register name (e.g., `"%tid.x"`, `"%laneid"`).
    #[must_use]
    pub const fn as_ptx_str(&self) -> &'static str {
        match self {
            Self::TidX => "%tid.x",
            Self::TidY => "%tid.y",
            Self::TidZ => "%tid.z",
            Self::CtaidX => "%ctaid.x",
            Self::CtaidY => "%ctaid.y",
            Self::CtaidZ => "%ctaid.z",
            Self::NtidX => "%ntid.x",
            Self::NtidY => "%ntid.y",
            Self::NtidZ => "%ntid.z",
            Self::NctaidX => "%nctaid.x",
            Self::NctaidY => "%nctaid.y",
            Self::NctaidZ => "%nctaid.z",
            Self::WarpId => "%warpid",
            Self::LaneId => "%laneid",
            Self::SmId => "%smid",
            Self::Clock => "%clock",
            Self::Clock64 => "%clock64",
            Self::DynamicSmemSize => "%dynamic_smem_size",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_type_as_ptx_str() {
        assert_eq!(PtxType::F32.as_ptx_str(), ".f32");
        assert_eq!(PtxType::U64.as_ptx_str(), ".u64");
        assert_eq!(PtxType::Pred.as_ptx_str(), ".pred");
        assert_eq!(PtxType::B128.as_ptx_str(), ".b128");
        assert_eq!(PtxType::E4M3.as_ptx_str(), ".e4m3");
        assert_eq!(PtxType::BF16x2.as_ptx_str(), ".bf16x2");
        assert_eq!(PtxType::S32.as_ptx_str(), ".s32");
    }

    #[test]
    fn ptx_type_size_bytes() {
        assert_eq!(PtxType::U8.size_bytes(), 1);
        assert_eq!(PtxType::F16.size_bytes(), 2);
        assert_eq!(PtxType::F32.size_bytes(), 4);
        assert_eq!(PtxType::F64.size_bytes(), 8);
        assert_eq!(PtxType::B128.size_bytes(), 16);
        assert_eq!(PtxType::Pred.size_bytes(), 1);
        assert_eq!(PtxType::F16x2.size_bytes(), 4);
        assert_eq!(PtxType::BF16x2.size_bytes(), 4);
        assert_eq!(PtxType::E2M1.size_bytes(), 1);
    }

    #[test]
    fn ptx_type_reg_type() {
        assert_eq!(PtxType::F32.reg_type(), PtxType::B32);
        assert_eq!(PtxType::F64.reg_type(), PtxType::B64);
        assert_eq!(PtxType::U64.reg_type(), PtxType::B64);
        assert_eq!(PtxType::Pred.reg_type(), PtxType::Pred);
        assert_eq!(PtxType::F16.reg_type(), PtxType::B16);
        assert_eq!(PtxType::B128.reg_type(), PtxType::B128);
        assert_eq!(PtxType::U8.reg_type(), PtxType::B32);
    }

    #[test]
    fn ptx_type_classification() {
        assert!(PtxType::U32.is_integer());
        assert!(PtxType::S64.is_integer());
        assert!(!PtxType::F32.is_integer());
        assert!(!PtxType::Pred.is_integer());

        assert!(PtxType::F32.is_float());
        assert!(PtxType::F16x2.is_float());
        assert!(PtxType::E4M3.is_float());
        assert!(!PtxType::U32.is_float());
        assert!(!PtxType::B32.is_float());

        assert!(PtxType::S32.is_signed());
        assert!(PtxType::F32.is_signed());
        assert!(!PtxType::U32.is_signed());
        assert!(!PtxType::B32.is_signed());
    }

    #[test]
    fn special_reg_ptx_str() {
        assert_eq!(SpecialReg::TidX.as_ptx_str(), "%tid.x");
        assert_eq!(SpecialReg::CtaidY.as_ptx_str(), "%ctaid.y");
        assert_eq!(SpecialReg::LaneId.as_ptx_str(), "%laneid");
        assert_eq!(SpecialReg::Clock64.as_ptx_str(), "%clock64");
        assert_eq!(
            SpecialReg::DynamicSmemSize.as_ptx_str(),
            "%dynamic_smem_size"
        );
    }

    #[test]
    fn rounding_mode_ptx_str() {
        assert_eq!(RoundingMode::Rn.as_ptx_str(), ".rn");
        assert_eq!(RoundingMode::Rz.as_ptx_str(), ".rz");
        assert_eq!(RoundingMode::Ru.as_ptx_str(), ".ru");
        assert_eq!(RoundingMode::Rd.as_ptx_str(), ".rd");
    }

    #[test]
    fn memory_space_ptx_str() {
        assert_eq!(MemorySpace::Global.as_ptx_str(), ".global");
        assert_eq!(MemorySpace::Shared.as_ptx_str(), ".shared");
        assert_eq!(MemorySpace::Constant.as_ptx_str(), ".const");
        assert_eq!(MemorySpace::Param.as_ptx_str(), ".param");
    }

    #[test]
    fn cmp_op_ptx_str() {
        assert_eq!(CmpOp::Eq.as_ptx_str(), ".eq");
        assert_eq!(CmpOp::Ltu.as_ptx_str(), ".ltu");
        assert_eq!(CmpOp::Nan.as_ptx_str(), ".nan");
    }

    #[test]
    fn vector_width_ptx_str() {
        assert_eq!(VectorWidth::V1.as_ptx_str(), "");
        assert_eq!(VectorWidth::V2.as_ptx_str(), ".v2");
        assert_eq!(VectorWidth::V4.as_ptx_str(), ".v4");
    }

    #[test]
    fn mul_mode_ptx_str() {
        assert_eq!(MulMode::Lo.as_ptx_str(), ".lo");
        assert_eq!(MulMode::Hi.as_ptx_str(), ".hi");
        assert_eq!(MulMode::Wide.as_ptx_str(), ".wide");
    }

    #[test]
    fn cache_qualifier_ptx_str() {
        assert_eq!(CacheQualifier::None.as_ptx_str(), "");
        assert_eq!(CacheQualifier::Ca.as_ptx_str(), ".ca");
        assert_eq!(CacheQualifier::Cv.as_ptx_str(), ".cv");
    }

    #[test]
    fn fence_scope_ptx_str() {
        assert_eq!(FenceScope::Cta.as_ptx_str(), ".cta");
        assert_eq!(FenceScope::Gpu.as_ptx_str(), ".gpu");
        assert_eq!(FenceScope::Sys.as_ptx_str(), ".sys");
    }

    #[test]
    fn atom_op_ptx_str() {
        assert_eq!(AtomOp::Add.as_ptx_str(), ".add");
        assert_eq!(AtomOp::Min.as_ptx_str(), ".min");
        assert_eq!(AtomOp::Max.as_ptx_str(), ".max");
        assert_eq!(AtomOp::Inc.as_ptx_str(), ".inc");
        assert_eq!(AtomOp::Dec.as_ptx_str(), ".dec");
        assert_eq!(AtomOp::And.as_ptx_str(), ".and");
        assert_eq!(AtomOp::Or.as_ptx_str(), ".or");
        assert_eq!(AtomOp::Xor.as_ptx_str(), ".xor");
        assert_eq!(AtomOp::Exch.as_ptx_str(), ".exch");
    }

    #[test]
    fn test_fp4_e2m1_type() {
        assert_eq!(PtxType::E2M1.bit_width(), 4);
        assert!(PtxType::E2M1.is_float());
        assert_eq!(format!("{}", PtxType::E2M1), "e2m1");
    }

    #[test]
    fn test_bit_width_correctness() {
        assert_eq!(PtxType::Pred.bit_width(), 1);
        assert_eq!(PtxType::E2M3.bit_width(), 6);
        assert_eq!(PtxType::E3M2.bit_width(), 6);
        assert_eq!(PtxType::E4M3.bit_width(), 8);
        assert_eq!(PtxType::E5M2.bit_width(), 8);
        assert_eq!(PtxType::U8.bit_width(), 8);
        assert_eq!(PtxType::F16.bit_width(), 16);
        assert_eq!(PtxType::BF16.bit_width(), 16);
        assert_eq!(PtxType::F16x2.bit_width(), 32);
        assert_eq!(PtxType::F32.bit_width(), 32);
        assert_eq!(PtxType::TF32.bit_width(), 32);
        assert_eq!(PtxType::F64.bit_width(), 64);
        assert_eq!(PtxType::B128.bit_width(), 128);
    }

    #[test]
    fn test_display_format() {
        assert_eq!(format!("{}", PtxType::F32), "f32");
        assert_eq!(format!("{}", PtxType::U64), "u64");
        assert_eq!(format!("{}", PtxType::E4M3), "e4m3");
        assert_eq!(format!("{}", PtxType::BF16x2), "bf16x2");
        assert_eq!(format!("{}", PtxType::B128), "b128");
        assert_eq!(format!("{}", PtxType::Pred), "pred");
    }
}
