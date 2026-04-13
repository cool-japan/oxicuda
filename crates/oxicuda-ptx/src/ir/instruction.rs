//! PTX instruction definitions and text emission.
//!
//! This module defines the [`Instruction`] enum covering arithmetic, memory,
//! comparison, control flow, synchronization, and Tensor Core operations.
//! Each variant maps directly to one or more PTX assembly instructions.
//! The [`Instruction::emit`] method produces the textual PTX representation.

use super::operand::Operand;
use super::register::Register;
use super::types::{
    AtomOp, CacheQualifier, CmpOp, FenceScope, MemorySpace, MulMode, PtxType, RoundingMode,
    SpecialReg, VectorWidth,
};

// ---------------------------------------------------------------------------
// Supporting types for Tensor Core instructions
// ---------------------------------------------------------------------------

/// WMMA (Warp Matrix Multiply-Accumulate) operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WmmaOp {
    /// Load matrix fragment A from memory.
    LoadA,
    /// Load matrix fragment B from memory.
    LoadB,
    /// Store matrix fragment D to memory.
    StoreD,
    /// Perform the matrix multiply-accumulate.
    Mma,
}

/// WMMA matrix shape (M x N x K).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WmmaShape {
    /// 16x16x16 tiles.
    M16N16K16,
    /// 8x32x16 tiles.
    M8N32K16,
    /// 32x8x16 tiles.
    M32N8K16,
}

impl WmmaShape {
    /// Returns the PTX shape suffix (e.g., `".m16n16k16"`).
    #[must_use]
    const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::M16N16K16 => ".m16n16k16",
            Self::M8N32K16 => ".m8n32k16",
            Self::M32N8K16 => ".m32n8k16",
        }
    }
}

/// Matrix layout for WMMA operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WmmaLayout {
    /// Row-major layout.
    RowMajor,
    /// Column-major layout.
    ColMajor,
}

impl WmmaLayout {
    /// Returns the PTX layout string (e.g., `".row"`, `".col"`).
    #[must_use]
    const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::RowMajor => ".row",
            Self::ColMajor => ".col",
        }
    }
}

/// MMA (Matrix Multiply-Accumulate) shape for `mma.sync` instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MmaShape {
    /// 16x8x8 tiles (Volta/Turing F16; Ampere+ TF32).
    M16N8K8,
    /// 16x8x16 tiles (Ampere+ F16/BF16/INT8).
    M16N8K16,
    /// 16x8x32 tiles (Hopper+ FP8/F16/BF16; Ampere+ INT8).
    M16N8K32,
    /// 8x8x16 tiles — Turing/Ampere INT8 (`mma.sync.aligned.m8n8k16`).
    M8N8K16,
    /// 8x8x32 tiles — Turing/Ampere INT4 (`mma.sync.aligned.m8n8k32`).
    M8N8K32,
}

impl MmaShape {
    /// Returns the PTX shape suffix (e.g., `".m16n8k16"`).
    #[must_use]
    pub const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::M16N8K8 => ".m16n8k8",
            Self::M16N8K16 => ".m16n8k16",
            Self::M16N8K32 => ".m16n8k32",
            Self::M8N8K16 => ".m8n8k16",
            Self::M8N8K32 => ".m8n8k32",
        }
    }
}

/// WGMMA (Warp Group MMA) shape for Hopper+ `wgmma` instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WgmmaShape {
    /// 64x8x16 tiles.
    M64N8K16,
    /// 64x16x16 tiles.
    M64N16K16,
    /// 64x32x16 tiles.
    M64N32K16,
    /// 64x64x16 tiles.
    M64N64K16,
    /// 64x128x16 tiles.
    M64N128K16,
    /// 64x256x16 tiles.
    M64N256K16,
}

impl WgmmaShape {
    /// Returns the PTX shape suffix (e.g., `".m64n128k16"`).
    #[must_use]
    const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::M64N8K16 => ".m64n8k16",
            Self::M64N16K16 => ".m64n16k16",
            Self::M64N32K16 => ".m64n32k16",
            Self::M64N64K16 => ".m64n64k16",
            Self::M64N128K16 => ".m64n128k16",
            Self::M64N256K16 => ".m64n256k16",
        }
    }
}

// ---------------------------------------------------------------------------
// Supporting types for PTX 8.x instructions
// ---------------------------------------------------------------------------

/// Redux (warp-level reduction) operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduxOp {
    /// Warp-level sum reduction.
    Add,
    /// Warp-level minimum reduction.
    Min,
    /// Warp-level maximum reduction.
    Max,
    /// Warp-level bitwise AND reduction.
    And,
    /// Warp-level bitwise OR reduction.
    Or,
    /// Warp-level bitwise XOR reduction.
    Xor,
}

impl ReduxOp {
    /// Returns the PTX suffix for this redux operation.
    #[must_use]
    const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::Add => ".add",
            Self::Min => ".min",
            Self::Max => ".max",
            Self::And => ".and",
            Self::Or => ".or",
            Self::Xor => ".xor",
        }
    }
}

/// Stmatrix shape for store-matrix-to-shared-memory instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StmatrixShape {
    /// Store 1 matrix fragment (m8n8).
    M8n8x1,
    /// Store 2 matrix fragments (m8n8).
    M8n8x2,
    /// Store 4 matrix fragments (m8n8).
    M8n8x4,
}

impl StmatrixShape {
    /// Returns the PTX shape suffix.
    #[must_use]
    const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::M8n8x1 => ".m8n8.x1",
            Self::M8n8x2 => ".m8n8.x2",
            Self::M8n8x4 => ".m8n8.x4",
        }
    }
}

/// Action for `setmaxnreg` instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SetmaxnregAction {
    /// Increase the maximum register count.
    Inc,
    /// Decrease the maximum register count.
    Dec,
}

impl SetmaxnregAction {
    /// Returns the PTX action suffix.
    #[must_use]
    const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::Inc => ".inc",
            Self::Dec => ".dec",
        }
    }
}

/// Action for `griddepcontrol` instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GridDepAction {
    /// Signal that dependent grids may launch.
    LaunchDependents,
    /// Wait for dependencies to complete.
    Wait,
}

impl GridDepAction {
    /// Returns the PTX action string.
    #[must_use]
    const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::LaunchDependents => ".launch_dependents",
            Self::Wait => ".wait",
        }
    }
}

// ---------------------------------------------------------------------------
// Main instruction enum
// ---------------------------------------------------------------------------

/// A single PTX instruction.
///
/// Each variant corresponds to a PTX assembly instruction (or pseudo-instruction
/// like comments and labels). The [`emit`](Instruction::emit) method converts
/// the instruction to its textual PTX representation.
///
/// # Instruction categories
///
/// - **Arithmetic**: `Add`, `Sub`, `Mul`, `Mad`, `Fma`, `Neg`, `Abs`, `Min`, `Max`
/// - **Comparison**: `SetP`
/// - **Memory**: `Load`, `Store`, `CpAsync`, `CpAsyncCommit`, `CpAsyncWait`
/// - **Type conversion**: `Cvt`
/// - **Control flow**: `Branch`, `Label`, `Return`
/// - **Synchronization**: `BarSync`, `BarArrive`, `FenceAcqRel`
/// - **Tensor Core**: `Wmma`, `Mma`, `Wgmma`, `TmaLoad`
/// - **Special**: `MovSpecial`, `LoadParam`, `Comment`, `Raw`
#[derive(Debug, Clone)]
pub enum Instruction {
    // -- Arithmetic ---------------------------------------------------------
    /// Integer or floating-point addition.
    Add {
        /// The data type.
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// First source operand.
        a: Operand,
        /// Second source operand.
        b: Operand,
    },

    /// Integer or floating-point subtraction.
    Sub {
        /// The data type.
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// First source operand.
        a: Operand,
        /// Second source operand.
        b: Operand,
    },

    /// Integer multiplication with mode selection (lo/hi/wide).
    Mul {
        /// The data type.
        ty: PtxType,
        /// Multiplication mode (lo, hi, or wide).
        mode: MulMode,
        /// Destination register.
        dst: Register,
        /// First source operand.
        a: Operand,
        /// Second source operand.
        b: Operand,
    },

    /// Multiply-and-add: `dst = a * b + c`.
    Mad {
        /// The data type.
        ty: PtxType,
        /// Multiplication mode.
        mode: MulMode,
        /// Destination register.
        dst: Register,
        /// First multiplicand.
        a: Operand,
        /// Second multiplicand.
        b: Operand,
        /// Addend.
        c: Operand,
    },

    /// Integer multiply-add, low bits: `mad.lo.{s32,u32,s64,u64} dst, a, b, c`.
    ///
    /// Computes `dst = (a * b + c)` keeping only the low 32 or 64 bits.
    MadLo {
        /// The integer data type (S32, U32, S64, or U64).
        typ: PtxType,
        /// Destination register (same width as `typ`).
        dst: Register,
        /// First multiplicand.
        a: Operand,
        /// Second multiplicand.
        b: Operand,
        /// Addend.
        c: Operand,
    },

    /// Integer multiply-add, high bits: `mad.hi.{s32,u32,s64,u64} dst, a, b, c`.
    ///
    /// Computes `a * b + c` and stores the upper half of the result.
    MadHi {
        /// The integer data type (S32, U32, S64, or U64).
        typ: PtxType,
        /// Destination register (same width as `typ`).
        dst: Register,
        /// First multiplicand.
        a: Operand,
        /// Second multiplicand.
        b: Operand,
        /// Addend.
        c: Operand,
    },

    /// Integer multiply-add, widening: `mad.wide.{s16,u16,s32,u32} dst, a, b, c`.
    ///
    /// Multiplies two N-bit values to produce a 2N-bit product, then adds `c`
    /// (which is also 2N bits). Result type is twice the width of `src_typ`.
    MadWide {
        /// The source type (S16, U16, S32, or U32).
        src_typ: PtxType,
        /// Destination register (twice the width of `src_typ`).
        dst: Register,
        /// First multiplicand (N-bit).
        a: Operand,
        /// Second multiplicand (N-bit).
        b: Operand,
        /// Addend (2N-bit).
        c: Operand,
    },

    /// Fused multiply-add with rounding: `dst = a * b + c` (no intermediate rounding).
    Fma {
        /// Rounding mode for the fused operation.
        rnd: RoundingMode,
        /// The floating-point data type.
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// First multiplicand.
        a: Operand,
        /// Second multiplicand.
        b: Operand,
        /// Addend.
        c: Operand,
    },

    /// Arithmetic negation.
    Neg {
        /// The data type.
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    /// Absolute value.
    Abs {
        /// The data type.
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    /// Minimum of two values.
    Min {
        /// The data type.
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// First source operand.
        a: Operand,
        /// Second source operand.
        b: Operand,
    },

    /// Maximum of two values.
    Max {
        /// The data type.
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// First source operand.
        a: Operand,
        /// Second source operand.
        b: Operand,
    },

    // -- Bit Manipulation ---------------------------------------------------
    /// Bit reverse: `brev.type dst, src;`
    Brev {
        /// The bit type (B32 or B64).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    /// Count leading zeros: `clz.type dst, src;`
    Clz {
        /// The source type (B32 or B64).
        ty: PtxType,
        /// Destination register (always U32).
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    /// Population count (number of 1-bits): `popc.type dst, src;`
    Popc {
        /// The source type (B32 or B64).
        ty: PtxType,
        /// Destination register (always U32).
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    /// Find most significant non-sign bit: `bfind.type dst, src;`
    Bfind {
        /// The source type (U32, S32, U64, S64).
        ty: PtxType,
        /// Destination register (always U32).
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    /// Bit field extract: `bfe.type dst, src, start, len;`
    Bfe {
        /// The data type (U32, S32, U64, S64).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand to extract from.
        src: Operand,
        /// Start bit position.
        start: Operand,
        /// Number of bits to extract.
        len: Operand,
    },

    /// Bit field insert: `bfi.type dst, insert, base, start, len;`
    Bfi {
        /// The bit type (B32 or B64).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Value to insert.
        insert: Operand,
        /// Base value to insert into.
        base: Operand,
        /// Start bit position.
        start: Operand,
        /// Number of bits to insert.
        len: Operand,
    },

    // -- Special Math -------------------------------------------------------
    /// Reciprocal: `rcp[.rnd].type dst, src;`
    Rcp {
        /// Optional rounding mode (None for approx mode).
        rnd: Option<RoundingMode>,
        /// The floating-point type.
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    /// Reciprocal square root: `rsqrt[.approx].type dst, src;`
    Rsqrt {
        /// Whether to use `.approx` qualifier.
        approx: bool,
        /// The floating-point type.
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    /// Square root: `sqrt[.rnd].type dst, src;`
    Sqrt {
        /// Optional rounding mode.
        rnd: Option<RoundingMode>,
        /// The floating-point type.
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    /// Base-2 exponential: `ex2[.approx].type dst, src;`
    Ex2 {
        /// Whether to use `.approx` qualifier.
        approx: bool,
        /// The floating-point type (typically F32).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    /// Base-2 logarithm: `lg2[.approx].type dst, src;`
    Lg2 {
        /// Whether to use `.approx` qualifier.
        approx: bool,
        /// The floating-point type (typically F32).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    /// Sine: `sin[.approx].type dst, src;`
    Sin {
        /// Whether to use `.approx` qualifier.
        approx: bool,
        /// The floating-point type (typically F32).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    /// Cosine: `cos[.approx].type dst, src;`
    Cos {
        /// Whether to use `.approx` qualifier.
        approx: bool,
        /// The floating-point type (typically F32).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    // -- Shift operations ---------------------------------------------------
    /// Left shift: `shl.type dst, src, amount;`
    Shl {
        /// The bit type (B32 or B64).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand to shift.
        src: Operand,
        /// Shift amount operand.
        amount: Operand,
    },

    /// Right shift: `shr.type dst, src, amount;`
    Shr {
        /// The bit type (B32, B64, U32, U64, S32, S64).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand to shift.
        src: Operand,
        /// Shift amount operand.
        amount: Operand,
    },

    // -- Integer Division & Modulo ------------------------------------------
    /// Integer division: `div.type dst, a, b;`
    Div {
        /// The data type (integer types only).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Dividend operand.
        a: Operand,
        /// Divisor operand.
        b: Operand,
    },

    /// Integer remainder (modulo): `rem.type dst, a, b;`
    Rem {
        /// The data type (integer types only).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Dividend operand.
        a: Operand,
        /// Divisor operand.
        b: Operand,
    },

    // -- Bitwise Logic ------------------------------------------------------
    /// Bitwise AND: `and.type dst, a, b;`
    And {
        /// The bit type (B32 or B64).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// First source operand.
        a: Operand,
        /// Second source operand.
        b: Operand,
    },

    /// Bitwise OR: `or.type dst, a, b;`
    Or {
        /// The bit type (B32 or B64).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// First source operand.
        a: Operand,
        /// Second source operand.
        b: Operand,
    },

    /// Bitwise XOR: `xor.type dst, a, b;`
    Xor {
        /// The bit type (B32 or B64).
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// First source operand.
        a: Operand,
        /// Second source operand.
        b: Operand,
    },

    // -- Comparison ---------------------------------------------------------
    /// Set predicate based on comparison: `dst = (a cmp b)`.
    SetP {
        /// The comparison operator.
        cmp: CmpOp,
        /// The data type of the operands being compared.
        ty: PtxType,
        /// Destination predicate register.
        dst: Register,
        /// First source operand.
        a: Operand,
        /// Second source operand.
        b: Operand,
    },

    // -- Memory -------------------------------------------------------------
    /// Load from memory.
    Load {
        /// Memory address space.
        space: MemorySpace,
        /// Cache qualifier.
        qualifier: CacheQualifier,
        /// Vector width (scalar, v2, or v4).
        vec: VectorWidth,
        /// Data type.
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source address operand.
        addr: Operand,
    },

    /// Store to memory.
    Store {
        /// Memory address space.
        space: MemorySpace,
        /// Cache qualifier.
        qualifier: CacheQualifier,
        /// Vector width (scalar, v2, or v4).
        vec: VectorWidth,
        /// Data type.
        ty: PtxType,
        /// Destination address operand.
        addr: Operand,
        /// Source register.
        src: Register,
    },

    /// Asynchronous copy from global to shared memory (Ampere+).
    CpAsync {
        /// Number of bytes to copy (4, 8, or 16).
        bytes: u32,
        /// Destination address in shared memory.
        dst_shared: Operand,
        /// Source address in global memory.
        src_global: Operand,
    },

    /// Commit outstanding `cp.async` operations to a group.
    CpAsyncCommit,

    /// Wait for at most `n` `cp.async` groups to remain in flight.
    CpAsyncWait {
        /// Number of groups allowed to remain in flight.
        n: u32,
    },

    // -- Type conversion ----------------------------------------------------
    /// Type conversion with optional rounding.
    Cvt {
        /// Optional rounding mode (required for float-to-float and float-to-int).
        rnd: Option<RoundingMode>,
        /// Destination type.
        dst_ty: PtxType,
        /// Source type.
        src_ty: PtxType,
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },

    // -- Control flow -------------------------------------------------------
    /// Conditional or unconditional branch.
    Branch {
        /// Target label name.
        target: String,
        /// Optional predicate: `(register, negated)`. If `negated` is true, branch
        /// is taken when the predicate is false (`@!%p0`).
        predicate: Option<(Register, bool)>,
    },

    /// A label (branch target).
    Label(String),

    /// Return from the current function.
    Return,

    // -- Synchronization ----------------------------------------------------
    /// Barrier synchronization: all threads in the CTA must reach this point.
    BarSync {
        /// Barrier ID (typically 0).
        id: u32,
    },

    /// Barrier arrive: signal arrival at a named barrier.
    BarArrive {
        /// Barrier ID.
        id: u32,
        /// Expected thread count.
        count: u32,
    },

    /// Acquire-release memory fence at the given scope.
    FenceAcqRel {
        /// Scope of the fence operation.
        scope: FenceScope,
    },

    // -- Tensor Core --------------------------------------------------------
    /// WMMA (Warp Matrix Multiply-Accumulate) instruction family.
    Wmma {
        /// The WMMA sub-operation (load, store, or mma).
        op: WmmaOp,
        /// Matrix tile shape.
        shape: WmmaShape,
        /// Matrix layout.
        layout: WmmaLayout,
        /// Element data type.
        ty: PtxType,
        /// Fragment registers.
        fragments: Vec<Register>,
        /// Optional memory address (for load/store operations).
        addr: Option<Operand>,
        /// Optional stride operand (for load/store operations).
        stride: Option<Operand>,
    },

    /// MMA (Matrix Multiply-Accumulate) via `mma.sync.aligned`.
    Mma {
        /// Matrix tile shape.
        shape: MmaShape,
        /// Type of matrix A elements.
        a_ty: PtxType,
        /// Type of matrix B elements.
        b_ty: PtxType,
        /// Type of matrix C (accumulator input) elements.
        c_ty: PtxType,
        /// Type of matrix D (accumulator output) elements.
        d_ty: PtxType,
        /// Destination (D) fragment registers.
        d_regs: Vec<Register>,
        /// Source A fragment registers.
        a_regs: Vec<Register>,
        /// Source B fragment registers.
        b_regs: Vec<Register>,
        /// Source C (accumulator) fragment registers.
        c_regs: Vec<Register>,
    },

    /// WGMMA (Warp Group MMA) for Hopper+ architectures.
    Wgmma {
        /// Matrix tile shape.
        shape: WgmmaShape,
        /// Accumulator (D) element type (always F32 per PTX ISA).
        d_ty: PtxType,
        /// A-matrix element type (F16, BF16, E4M3, E5M2).
        a_ty: PtxType,
        /// B-matrix element type (must match `a_ty`).
        b_ty: PtxType,
        /// Descriptor register for matrix A (shared-memory descriptor).
        desc_a: Register,
        /// Descriptor register for matrix B (shared-memory descriptor).
        desc_b: Register,
        /// Destination (accumulator) registers.
        d_regs: Vec<Register>,
        /// Scale factor for D output (1 = accumulate, 0 = zero-init then write).
        scale_d: i32,
        /// Immediate scale for A operand (always 1 in standard usage).
        imm_scale_a: i32,
        /// Immediate scale for B operand (always 1 in standard usage).
        imm_scale_b: i32,
        /// Transpose A from col-major to row-major (0 = no, 1 = yes).
        trans_a: i32,
        /// Transpose B from row-major to col-major (0 = no, 1 = yes).
        trans_b: i32,
    },

    /// TMA (Tensor Memory Accelerator) load from global to shared memory.
    TmaLoad {
        /// Destination address in shared memory.
        dst_shared: Operand,
        /// TMA descriptor register.
        desc: Register,
        /// Coordinate registers.
        coords: Vec<Register>,
        /// Barrier register for completion tracking.
        barrier: Register,
    },

    // -- Atomic operations ---------------------------------------------------
    /// Atomic operation on global or shared memory.
    ///
    /// `atom.space.op.type dst, [addr], operand;`
    ///
    /// Returns the old value at `[addr]` in `dst` and atomically applies `op`
    /// using `src` as the second operand.
    Atom {
        /// Memory address space (global or shared).
        space: MemorySpace,
        /// The atomic operation to perform.
        op: AtomOp,
        /// Data type of the atomic operation.
        ty: PtxType,
        /// Destination register (receives the old value).
        dst: Register,
        /// Address operand (pointer to the memory location).
        addr: Operand,
        /// Source operand (value to combine atomically).
        src: Operand,
    },

    /// Atomic compare-and-swap on global or shared memory.
    ///
    /// `atom.space.cas.type dst, [addr], compare, value;`
    ///
    /// If `[addr] == compare`, stores `value` at `[addr]`. Always returns
    /// the old value at `[addr]` in `dst`.
    AtomCas {
        /// Memory address space (global or shared).
        space: MemorySpace,
        /// Data type of the compare-and-swap.
        ty: PtxType,
        /// Destination register (receives the old value).
        dst: Register,
        /// Address operand (pointer to the memory location).
        addr: Operand,
        /// Compare operand.
        compare: Operand,
        /// Value to store if comparison succeeds.
        value: Operand,
    },

    /// Atomic reduction (no return value).
    ///
    /// `red.space.op.type [addr], operand;`
    ///
    /// Similar to `atom` but does not return the old value. This can be
    /// more efficient when the old value is not needed.
    Red {
        /// Memory address space (global or shared).
        space: MemorySpace,
        /// The reduction operation to perform.
        op: AtomOp,
        /// Data type of the reduction.
        ty: PtxType,
        /// Address operand (pointer to the memory location).
        addr: Operand,
        /// Source operand (value to combine atomically).
        src: Operand,
    },

    // -- Special registers --------------------------------------------------
    /// Move a special register value into a general-purpose register.
    MovSpecial {
        /// Destination register.
        dst: Register,
        /// The special register to read.
        special: SpecialReg,
    },

    // -- Parameter loading --------------------------------------------------
    /// Load a kernel parameter into a register.
    LoadParam {
        /// The parameter data type.
        ty: PtxType,
        /// Destination register.
        dst: Register,
        /// The parameter name as declared in the function signature.
        param_name: String,
    },

    // -- Miscellaneous ------------------------------------------------------
    /// A PTX comment (emitted as `// ...`).
    Comment(String),

    /// Raw PTX text, emitted verbatim. Use sparingly for instructions not yet
    /// modeled in this IR.
    Raw(String),

    /// PTX pragma directive (e.g., `.pragma "unroll";`).
    Pragma(String),

    // -- Video Instructions --------------------------------------------------
    /// 4-way byte dot product: `dp4a.{atype}.{btype} dst, a, b, c;`
    ///
    /// Computes `dst = c + dot(a_bytes, b_bytes)` where `a` and `b` are each
    /// treated as 4 packed signed or unsigned bytes.
    Dp4a {
        /// Destination register (S32).
        dst: Register,
        /// First source operand (4 packed bytes in a 32-bit register).
        a: Operand,
        /// Second source operand (4 packed bytes in a 32-bit register).
        b: Operand,
        /// Accumulator operand (S32).
        c: Operand,
        /// Whether `a` is treated as signed bytes (`true` = `.s32`, `false` = `.u32`).
        signed_a: bool,
        /// Whether `b` is treated as signed bytes (`true` = `.s32`, `false` = `.u32`).
        signed_b: bool,
    },

    /// 2-way halfword dot product: `dp2a.{atype}.{btype} dst, a, b, c;`
    ///
    /// Computes `dst = c + dot(a_halfwords, b_halfwords)` where `a` and `b`
    /// contain 2 packed 16-bit values.
    Dp2a {
        /// Destination register (S32).
        dst: Register,
        /// First source operand (2 packed halfwords in a 32-bit register).
        a: Operand,
        /// Second source operand (2 packed halfwords in a 32-bit register).
        b: Operand,
        /// Accumulator operand (S32).
        c: Operand,
        /// Whether `a` is treated as signed (`true` = `.s32`, `false` = `.u32`).
        signed_a: bool,
        /// Whether `b` is treated as signed (`true` = `.s32`, `false` = `.u32`).
        signed_b: bool,
        /// If `true`, use the low 16 bits of `b`; if `false`, use the high 16 bits.
        lo: bool,
    },

    // -- Texture / Surface operations ------------------------------------------
    /// 1D texture fetch: `tex.1d.v4.{ty}.s32 dst, [tex_ref, {coord}];`
    ///
    /// Fetches a texel from a 1-dimensional texture using integer coordinates.
    Tex1d {
        /// Element data type of the returned texel components.
        ty: PtxType,
        /// Destination register receiving the fetched texel.
        dst: Register,
        /// Texture reference name.
        tex_ref: String,
        /// 1D coordinate operand (`.s32`).
        coord: Operand,
    },

    /// 2D texture fetch: `tex.2d.v4.{ty}.s32 dst, [tex_ref, {{coord_x, coord_y}}];`
    ///
    /// Fetches a texel from a 2-dimensional texture using integer coordinates.
    Tex2d {
        /// Element data type of the returned texel components.
        ty: PtxType,
        /// Destination register receiving the fetched texel.
        dst: Register,
        /// Texture reference name.
        tex_ref: String,
        /// X coordinate operand (`.s32`).
        coord_x: Operand,
        /// Y coordinate operand (`.s32`).
        coord_y: Operand,
    },

    /// 3D texture fetch: `tex.3d.v4.{ty}.s32 dst, [tex_ref, {{x, y, z}}];`
    ///
    /// Fetches a texel from a 3-dimensional texture using integer coordinates.
    Tex3d {
        /// Element data type of the returned texel components.
        ty: PtxType,
        /// Destination register receiving the fetched texel.
        dst: Register,
        /// Texture reference name.
        tex_ref: String,
        /// X coordinate operand (`.s32`).
        coord_x: Operand,
        /// Y coordinate operand (`.s32`).
        coord_y: Operand,
        /// Z coordinate operand (`.s32`).
        coord_z: Operand,
    },

    /// Surface load: `suld.b.1d.{ty} dst, [surf_ref, {coord}];`
    ///
    /// Loads a value from a 1-dimensional surface at the given coordinate.
    SurfLoad {
        /// Element data type.
        ty: PtxType,
        /// Destination register receiving the loaded value.
        dst: Register,
        /// Surface reference name.
        surf_ref: String,
        /// 1D coordinate operand.
        coord: Operand,
    },

    /// Surface store: `sust.b.1d.{ty} [surf_ref, {coord}], src;`
    ///
    /// Stores a value to a 1-dimensional surface at the given coordinate.
    SurfStore {
        /// Element data type.
        ty: PtxType,
        /// Surface reference name.
        surf_ref: String,
        /// 1D coordinate operand.
        coord: Operand,
        /// Source register containing the value to store.
        src: Register,
    },

    // -- PTX 8.x Instructions (SM >= 80/90) ---------------------------------
    /// Redux warp-level reduction: `redux.sync.op.u32 dst, src, membermask;`
    ///
    /// Performs a warp-level reduction across participating threads (SM >= 80).
    Redux {
        /// The reduction operation.
        op: ReduxOp,
        /// Destination register (receives the reduced value).
        dst: Register,
        /// Source operand (each thread's value).
        src: Operand,
        /// Membership mask (which lanes participate).
        membership_mask: u32,
    },

    /// Store matrix to shared memory: `stmatrix.sync.aligned.shape[.trans] [dst], src;`
    ///
    /// Cooperatively stores matrix fragments from registers to shared memory (SM >= 90).
    Stmatrix {
        /// Destination address in shared memory.
        dst_addr: Operand,
        /// Source register containing the matrix fragment.
        src: Register,
        /// Matrix shape to store.
        shape: StmatrixShape,
        /// Whether to transpose during the store.
        trans: bool,
    },

    /// Elect warp leader: `elect.sync dst, membermask;`
    ///
    /// Elects a single thread (the lowest active lane) as the leader (SM >= 90).
    ElectSync {
        /// Destination predicate register (true for the elected thread).
        dst: Register,
        /// Membership mask (which lanes participate).
        membership_mask: u32,
    },

    /// Set maximum register count hint: `setmaxnreg.action count;`
    ///
    /// Dynamically adjusts the maximum number of registers a thread can use (SM >= 90).
    Setmaxnreg {
        /// Number of registers.
        reg_count: u32,
        /// Whether to increase or decrease.
        action: SetmaxnregAction,
    },

    /// Grid dependency control: `griddepcontrol.action;`
    ///
    /// Controls dependencies between grid launches (SM >= 90).
    Griddepcontrol {
        /// The dependency control action.
        action: GridDepAction,
    },

    /// Proxy fence for async operations: `fence.proxy.async.scope.space;`
    ///
    /// Ensures ordering of asynchronous memory operations.
    FenceProxy {
        /// Scope of the fence.
        scope: FenceScope,
        /// Memory space to fence.
        space: MemorySpace,
    },

    /// Mbarrier init: `mbarrier.init.shared.b64 [addr], count;`
    ///
    /// Initializes a shared-memory barrier with the expected arrival count (SM >= 90).
    MbarrierInit {
        /// Address of the mbarrier object in shared memory.
        addr: Operand,
        /// Expected arrival count.
        count: Operand,
    },

    /// Mbarrier arrive: `mbarrier.arrive.shared.b64 [addr];`
    ///
    /// Signals arrival at a shared-memory barrier (SM >= 90).
    MbarrierArrive {
        /// Address of the mbarrier object in shared memory.
        addr: Operand,
    },

    /// Mbarrier wait: `mbarrier.try_wait.parity.shared.b64 [addr], phase;`
    ///
    /// Waits for all expected arrivals at a shared-memory barrier (SM >= 90).
    MbarrierWait {
        /// Address of the mbarrier object in shared memory.
        addr: Operand,
        /// Phase bit to wait on.
        phase: Operand,
    },

    // -- SM 100+ (Blackwell) Tensor Core: tcgen05 ---------------------------
    /// 5th-generation Tensor Core MMA (tcgen05): `tcgen05.mma.cta_group::1.kind::f32 [a_desc], [b_desc];`
    ///
    /// Performs a bulk matrix multiply-accumulate using descriptor-based operands
    /// on SM 100 (Blackwell) and later architectures.
    Tcgen05Mma {
        /// Descriptor register pointing to the A matrix tile.
        a_desc: Register,
        /// Descriptor register pointing to the B matrix tile.
        b_desc: Register,
    },

    // -- Cluster-level barrier / fence -------------------------------------
    /// Cluster barrier arrive: `barrier.cluster.arrive;`
    ///
    /// Signals arrival at the cluster-level barrier.  All CTAs in the cluster
    /// must execute this instruction before any may proceed (SM >= 90).
    BarrierCluster,

    /// Cluster fence: `fence.mbarrier_init.release.cluster;`
    ///
    /// Issues a release fence scoped to the cluster, ensuring that all prior
    /// memory operations (including mbarrier initializations) are visible to
    /// all CTAs in the cluster before the barrier can be observed (SM >= 90).
    FenceCluster,

    // -- TMA descriptor-based bulk async copy -------------------------------
    /// Bulk async copy using a TMA descriptor for a 1-D tensor:
    /// `cp.async.bulk.tensor.1d.shared::cluster.global.tile.bulk_group [dst], [src, {desc}];`
    ///
    /// Asynchronously copies a 1-D tile from global memory to shared memory
    /// using a TMA (Tensor Memory Accelerator) descriptor (SM >= 90).
    CpAsyncBulk {
        /// Destination register holding the shared-memory address.
        dst_smem: Register,
        /// Source register holding the global-memory base address.
        src_gmem: Register,
        /// Descriptor register (coordinate / TMA descriptor).
        desc: Register,
    },

    // -- ldmatrix (SM >= 75) ------------------------------------------------
    /// Warp-cooperative load of a matrix fragment from shared memory (SM >= 75).
    ///
    /// `ldmatrix.sync.aligned.m8n8.x4.shared.b16 {d0, d1, d2, d3}, [addr];`
    ///
    /// Loads 4 matrix fragments (x4) of type b16 from shared memory in
    /// a warp-cooperative manner. Each of the 32 threads in the warp contributes
    /// to loading 8 bytes (one row) from the shared memory tile.
    Ldmatrix {
        /// Number of fragments: 1, 2, or 4 (`x1`, `x2`, `x4`).
        num_fragments: u32,
        /// Whether to transpose the loaded fragments.
        trans: bool,
        /// Destination registers (one per fragment).
        dst_regs: Vec<Register>,
        /// Source address in shared memory.
        src_addr: Operand,
    },
}

// ---------------------------------------------------------------------------
// Emit implementation
// ---------------------------------------------------------------------------

impl Instruction {
    /// Emits this instruction as a PTX assembly text string.
    ///
    /// The returned string includes the trailing semicolon (where applicable)
    /// and is suitable for direct inclusion in a PTX function body.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxicuda_ptx::ir::*;
    ///
    /// let dst = Register { name: "%f0".into(), ty: PtxType::F32 };
    /// let a = Operand::Register(Register { name: "%f1".into(), ty: PtxType::F32 });
    /// let b = Operand::Register(Register { name: "%f2".into(), ty: PtxType::F32 });
    /// let inst = Instruction::Add { ty: PtxType::F32, dst, a, b };
    /// assert_eq!(inst.emit(), "add.f32 %f0, %f1, %f2;");
    /// ```
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn emit(&self) -> String {
        match self {
            // -- Arithmetic -------------------------------------------------
            Self::Add { ty, dst, a, b } => {
                format!("add{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Sub { ty, dst, a, b } => {
                format!("sub{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Mul {
                ty,
                mode,
                dst,
                a,
                b,
            } => {
                format!(
                    "mul{}{} {dst}, {a}, {b};",
                    mode.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::Mad {
                ty,
                mode,
                dst,
                a,
                b,
                c,
            } => {
                format!(
                    "mad{}{} {dst}, {a}, {b}, {c};",
                    mode.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::MadLo { typ, dst, a, b, c } => {
                format!("mad.lo{} {dst}, {a}, {b}, {c};", typ.as_ptx_str())
            }
            Self::MadHi { typ, dst, a, b, c } => {
                format!("mad.hi{} {dst}, {a}, {b}, {c};", typ.as_ptx_str())
            }
            Self::MadWide {
                src_typ,
                dst,
                a,
                b,
                c,
            } => {
                format!("mad.wide{} {dst}, {a}, {b}, {c};", src_typ.as_ptx_str())
            }
            Self::Fma {
                rnd,
                ty,
                dst,
                a,
                b,
                c,
            } => {
                format!(
                    "fma{}{} {dst}, {a}, {b}, {c};",
                    rnd.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::Neg { ty, dst, src } => {
                format!("neg{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Abs { ty, dst, src } => {
                format!("abs{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Min { ty, dst, a, b } => {
                format!("min{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Max { ty, dst, a, b } => {
                format!("max{} {dst}, {a}, {b};", ty.as_ptx_str())
            }

            // -- Bit Manipulation -------------------------------------------
            Self::Brev { ty, dst, src } => {
                format!("brev{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Clz { ty, dst, src } => {
                format!("clz{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Popc { ty, dst, src } => {
                format!("popc{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Bfind { ty, dst, src } => {
                format!("bfind{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Bfe {
                ty,
                dst,
                src,
                start,
                len,
            } => {
                format!("bfe{} {dst}, {src}, {start}, {len};", ty.as_ptx_str())
            }
            Self::Bfi {
                ty,
                dst,
                insert,
                base,
                start,
                len,
            } => {
                format!(
                    "bfi{} {dst}, {insert}, {base}, {start}, {len};",
                    ty.as_ptx_str()
                )
            }

            // -- Special Math -----------------------------------------------
            Self::Rcp { rnd, ty, dst, src } => {
                let rnd_str = rnd.map_or(String::new(), |r| r.as_ptx_str().to_string());
                format!("rcp{rnd_str}{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Rsqrt {
                approx,
                ty,
                dst,
                src,
            } => {
                let approx_str = if *approx { ".approx" } else { "" };
                format!("rsqrt{approx_str}{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Sqrt { rnd, ty, dst, src } => {
                let rnd_str = rnd.map_or(String::new(), |r| r.as_ptx_str().to_string());
                format!("sqrt{rnd_str}{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Ex2 {
                approx,
                ty,
                dst,
                src,
            } => {
                let approx_str = if *approx { ".approx" } else { "" };
                format!("ex2{approx_str}{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Lg2 {
                approx,
                ty,
                dst,
                src,
            } => {
                let approx_str = if *approx { ".approx" } else { "" };
                format!("lg2{approx_str}{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Sin {
                approx,
                ty,
                dst,
                src,
            } => {
                let approx_str = if *approx { ".approx" } else { "" };
                format!("sin{approx_str}{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Cos {
                approx,
                ty,
                dst,
                src,
            } => {
                let approx_str = if *approx { ".approx" } else { "" };
                format!("cos{approx_str}{} {dst}, {src};", ty.as_ptx_str())
            }

            // -- Shift operations -------------------------------------------
            Self::Shl {
                ty,
                dst,
                src,
                amount,
            } => {
                format!("shl{} {dst}, {src}, {amount};", ty.as_ptx_str())
            }
            Self::Shr {
                ty,
                dst,
                src,
                amount,
            } => {
                format!("shr{} {dst}, {src}, {amount};", ty.as_ptx_str())
            }

            // -- Integer Division & Modulo ----------------------------------
            Self::Div { ty, dst, a, b } => {
                format!("div{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Rem { ty, dst, a, b } => {
                format!("rem{} {dst}, {a}, {b};", ty.as_ptx_str())
            }

            // -- Bitwise Logic ----------------------------------------------
            Self::And { ty, dst, a, b } => {
                format!("and{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Or { ty, dst, a, b } => {
                format!("or{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Xor { ty, dst, a, b } => {
                format!("xor{} {dst}, {a}, {b};", ty.as_ptx_str())
            }

            // -- Comparison -------------------------------------------------
            Self::SetP { cmp, ty, dst, a, b } => {
                format!(
                    "setp{}{} {dst}, {a}, {b};",
                    cmp.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }

            // -- Memory -----------------------------------------------------
            Self::Load {
                space,
                qualifier,
                vec,
                ty,
                dst,
                addr,
            } => {
                format!(
                    "ld{}{}{}{} {dst}, {addr};",
                    space.as_ptx_str(),
                    qualifier.as_ptx_str(),
                    vec.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::Store {
                space,
                qualifier,
                vec,
                ty,
                addr,
                src,
            } => {
                format!(
                    "st{}{}{}{} {addr}, {src};",
                    space.as_ptx_str(),
                    qualifier.as_ptx_str(),
                    vec.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::CpAsync {
                bytes,
                dst_shared,
                src_global,
            } => {
                format!("cp.async.ca.shared.global [{dst_shared}], [{src_global}], {bytes};")
            }
            Self::CpAsyncCommit => "cp.async.commit_group;".to_string(),
            Self::CpAsyncWait { n } => {
                format!("cp.async.wait_group {n};")
            }

            // -- Type conversion --------------------------------------------
            Self::Cvt {
                rnd,
                dst_ty,
                src_ty,
                dst,
                src,
            } => {
                let rnd_str = rnd.map_or(String::new(), |r| r.as_ptx_str().to_string());
                format!(
                    "cvt{rnd_str}{}{} {dst}, {src};",
                    dst_ty.as_ptx_str(),
                    src_ty.as_ptx_str()
                )
            }

            // -- Control flow -----------------------------------------------
            Self::Branch { target, predicate } => match predicate {
                Some((pred, negated)) => {
                    let neg = if *negated { "!" } else { "" };
                    format!("@{neg}{pred} bra ${target};")
                }
                None => format!("bra ${target};"),
            },
            Self::Label(name) => format!("${name}:"),
            Self::Return => "ret;".to_string(),

            // -- Synchronization --------------------------------------------
            Self::BarSync { id } => format!("bar.sync {id};"),
            Self::BarArrive { id, count } => {
                format!("bar.arrive {id}, {count};")
            }
            Self::FenceAcqRel { scope } => {
                format!("fence.acq_rel{};", scope.as_ptx_str())
            }

            // -- Tensor Core: WMMA ------------------------------------------
            Self::Wmma {
                op,
                shape,
                layout,
                ty,
                fragments,
                addr,
                stride,
            } => emit_wmma(
                *op,
                *shape,
                *layout,
                *ty,
                fragments,
                addr.as_ref(),
                stride.as_ref(),
            ),

            // -- Tensor Core: MMA -------------------------------------------
            Self::Mma {
                shape,
                a_ty,
                b_ty,
                c_ty,
                d_ty,
                d_regs,
                a_regs,
                b_regs,
                c_regs,
            } => emit_mma(
                *shape, *a_ty, *b_ty, *c_ty, *d_ty, d_regs, a_regs, b_regs, c_regs,
            ),

            // -- Tensor Core: WGMMA -----------------------------------------
            Self::Wgmma {
                shape,
                d_ty,
                a_ty,
                b_ty,
                desc_a,
                desc_b,
                d_regs,
                scale_d,
                imm_scale_a,
                imm_scale_b,
                trans_a,
                trans_b,
            } => {
                let d_list = reg_list(d_regs);
                format!(
                    "wgmma.mma_async.sync.aligned{}{}{}{} {{{d_list}}}, {desc_a}, {desc_b}, {scale_d}, {imm_scale_a}, {imm_scale_b}, {trans_a}, {trans_b};",
                    shape.as_ptx_str(),
                    d_ty.as_ptx_str(),
                    a_ty.as_ptx_str(),
                    b_ty.as_ptx_str(),
                )
            }

            // -- TMA --------------------------------------------------------
            Self::TmaLoad {
                dst_shared,
                desc,
                coords,
                barrier,
            } => {
                let coord_list = coords
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [{dst_shared}], [{desc}, {{{coord_list}}}], [{barrier}];",
                )
            }

            // -- Atomic operations -------------------------------------------
            Self::Atom {
                space,
                op,
                ty,
                dst,
                addr,
                src,
            } => {
                format!(
                    "atom{}{}{} {dst}, [{addr}], {src};",
                    space.as_ptx_str(),
                    op.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::AtomCas {
                space,
                ty,
                dst,
                addr,
                compare,
                value,
            } => {
                format!(
                    "atom{}.cas{} {dst}, [{addr}], {compare}, {value};",
                    space.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::Red {
                space,
                op,
                ty,
                addr,
                src,
            } => {
                format!(
                    "red{}{}{} [{addr}], {src};",
                    space.as_ptx_str(),
                    op.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }

            // -- Special registers ------------------------------------------
            Self::MovSpecial { dst, special } => {
                format!("mov.u32 {dst}, {};", special.as_ptx_str())
            }

            // -- Parameter loading ------------------------------------------
            Self::LoadParam {
                ty,
                dst,
                param_name,
            } => {
                format!("ld.param{} {dst}, [{param_name}];", ty.as_ptx_str())
            }

            // -- Miscellaneous ----------------------------------------------
            Self::Comment(text) => format!("// {text}"),
            Self::Raw(text) => text.clone(),
            Self::Pragma(text) => format!(".pragma \"{text}\";"),

            // -- Video Instructions -----------------------------------------
            Self::Dp4a {
                dst,
                a,
                b,
                c,
                signed_a,
                signed_b,
            } => {
                let a_ty = if *signed_a { ".s32" } else { ".u32" };
                let b_ty = if *signed_b { ".s32" } else { ".u32" };
                format!("dp4a{a_ty}{b_ty} {dst}, {a}, {b}, {c};")
            }

            Self::Dp2a {
                dst,
                a,
                b,
                c,
                signed_a,
                signed_b,
                lo,
            } => {
                let a_ty = if *signed_a { ".s32" } else { ".u32" };
                let b_ty = if *signed_b { ".s32" } else { ".u32" };
                let half = if *lo { ".lo" } else { ".hi" };
                format!("dp2a{half}{a_ty}{b_ty} {dst}, {a}, {b}, {c};")
            }

            // -- PTX 8.x Instructions ---------------------------------------
            Self::Redux {
                op,
                dst,
                src,
                membership_mask,
            } => {
                format!(
                    "redux.sync{}.u32 {dst}, {src}, 0x{membership_mask:08x};",
                    op.as_ptx_str()
                )
            }
            Self::Stmatrix {
                dst_addr,
                src,
                shape,
                trans,
            } => {
                let trans_str = if *trans { ".trans" } else { "" };
                format!(
                    "stmatrix.sync.aligned{}{trans_str}.shared.b16 [{dst_addr}], {{{src}}};",
                    shape.as_ptx_str()
                )
            }
            Self::ElectSync {
                dst,
                membership_mask,
            } => {
                format!("elect.sync {dst}, 0x{membership_mask:08x};")
            }
            Self::Setmaxnreg { reg_count, action } => {
                format!("setmaxnreg{} {reg_count};", action.as_ptx_str())
            }
            Self::Griddepcontrol { action } => {
                format!("griddepcontrol{};", action.as_ptx_str())
            }
            Self::FenceProxy { scope, space } => {
                format!(
                    "fence.proxy.async{}{};",
                    scope.as_ptx_str(),
                    space.as_ptx_str()
                )
            }
            Self::MbarrierInit { addr, count } => {
                format!("mbarrier.init.shared.b64 [{addr}], {count};")
            }
            Self::MbarrierArrive { addr } => {
                format!("mbarrier.arrive.shared.b64 [{addr}];")
            }
            Self::MbarrierWait { addr, phase } => {
                format!("mbarrier.try_wait.parity.shared.b64 [{addr}], {phase};")
            }

            // -- SM 100+ (Blackwell) tcgen05 MMA ----------------------------
            Self::Tcgen05Mma { a_desc, b_desc } => {
                format!("tcgen05.mma.cta_group::1.kind::f32 [{a_desc}], [{b_desc}];")
            }

            // -- Cluster barrier / fence ------------------------------------
            Self::BarrierCluster => "barrier.cluster.arrive;".to_string(),
            Self::FenceCluster => "fence.mbarrier_init.release.cluster;".to_string(),

            // -- TMA bulk async copy -----------------------------------------
            Self::CpAsyncBulk {
                dst_smem,
                src_gmem,
                desc,
            } => {
                format!(
                    "cp.async.bulk.tensor.1d.shared::cluster.global.tile.bulk_group [{dst_smem}], [{src_gmem}, {{{desc}}}];"
                )
            }

            // -- Texture / Surface ------------------------------------------
            Self::Tex1d {
                ty,
                dst,
                tex_ref,
                coord,
            } => {
                format!(
                    "tex.1d.v4{}.s32 {dst}, [{tex_ref}, {{{coord}}}];",
                    ty.as_ptx_str()
                )
            }
            Self::Tex2d {
                ty,
                dst,
                tex_ref,
                coord_x,
                coord_y,
            } => {
                format!(
                    "tex.2d.v4{}.s32 {dst}, [{tex_ref}, {{{coord_x}, {coord_y}}}];",
                    ty.as_ptx_str()
                )
            }
            Self::Tex3d {
                ty,
                dst,
                tex_ref,
                coord_x,
                coord_y,
                coord_z,
            } => {
                format!(
                    "tex.3d.v4{}.s32 {dst}, [{tex_ref}, {{{coord_x}, {coord_y}, {coord_z}}}];",
                    ty.as_ptx_str()
                )
            }
            Self::SurfLoad {
                ty,
                dst,
                surf_ref,
                coord,
            } => {
                format!(
                    "suld.b.1d{} {dst}, [{surf_ref}, {{{coord}}}];",
                    ty.as_ptx_str()
                )
            }
            Self::SurfStore {
                ty,
                surf_ref,
                coord,
                src,
            } => {
                format!(
                    "sust.b.1d{} [{surf_ref}, {{{coord}}}], {src};",
                    ty.as_ptx_str()
                )
            }

            // -- ldmatrix (SM >= 75) ----------------------------------------
            Self::Ldmatrix {
                num_fragments,
                trans,
                dst_regs,
                src_addr,
            } => {
                let trans_str = if *trans { ".trans" } else { "" };
                let x_str = match num_fragments {
                    2 => ".x2",
                    4 => ".x4",
                    _ => ".x1",
                };
                let dst_list = dst_regs
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "ldmatrix.sync.aligned.m8n8{x_str}{trans_str}.shared.b16 {{{dst_list}}}, [{src_addr}];"
                )
            }
        }
    }
}

/// Emit a WMMA instruction family member.
#[allow(clippy::too_many_lines)]
fn emit_wmma(
    op: WmmaOp,
    shape: WmmaShape,
    layout: WmmaLayout,
    ty: PtxType,
    fragments: &[Register],
    addr: Option<&Operand>,
    stride: Option<&Operand>,
) -> String {
    let frag_list = reg_list(fragments);
    match op {
        WmmaOp::LoadA => {
            let addr_str = addr.map_or(String::new(), |a| format!("{a}"));
            let stride_str = stride.map_or(String::new(), |s| format!(", {s}"));
            format!(
                "wmma.load.a.sync.aligned{}{}{} {{{frag_list}}}, [{addr_str}]{stride_str};",
                shape.as_ptx_str(),
                layout.as_ptx_str(),
                ty.as_ptx_str()
            )
        }
        WmmaOp::LoadB => {
            let addr_str = addr.map_or(String::new(), |a| format!("{a}"));
            let stride_str = stride.map_or(String::new(), |s| format!(", {s}"));
            format!(
                "wmma.load.b.sync.aligned{}{}{} {{{frag_list}}}, [{addr_str}]{stride_str};",
                shape.as_ptx_str(),
                layout.as_ptx_str(),
                ty.as_ptx_str()
            )
        }
        WmmaOp::StoreD => {
            let addr_str = addr.map_or(String::new(), |a| format!("{a}"));
            let stride_str = stride.map_or(String::new(), |s| format!(", {s}"));
            format!(
                "wmma.store.d.sync.aligned{}{}{} [{addr_str}], {{{frag_list}}}{stride_str};",
                shape.as_ptx_str(),
                layout.as_ptx_str(),
                ty.as_ptx_str()
            )
        }
        WmmaOp::Mma => {
            format!(
                "wmma.mma.sync.aligned{}{}{} {{{frag_list}}};",
                shape.as_ptx_str(),
                layout.as_ptx_str(),
                ty.as_ptx_str()
            )
        }
    }
}

/// Emit an `mma.sync.aligned` instruction.
#[allow(clippy::too_many_arguments)]
fn emit_mma(
    shape: MmaShape,
    a_ty: PtxType,
    b_ty: PtxType,
    c_ty: PtxType,
    d_ty: PtxType,
    d_regs: &[Register],
    a_regs: &[Register],
    b_regs: &[Register],
    c_regs: &[Register],
) -> String {
    let d_list = reg_list(d_regs);
    let a_list = reg_list(a_regs);
    let b_list = reg_list(b_regs);
    let c_list = reg_list(c_regs);
    format!(
        "mma.sync.aligned{}.row.col{}{}{}{} {{{d_list}}}, {{{a_list}}}, {{{b_list}}}, {{{c_list}}};",
        shape.as_ptx_str(),
        d_ty.as_ptx_str(),
        a_ty.as_ptx_str(),
        b_ty.as_ptx_str(),
        c_ty.as_ptx_str()
    )
}

/// Format a comma-separated list of register names for use in `{...}` groups.
fn reg_list(regs: &[Register]) -> String {
    regs.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "instruction_tests.rs"]
mod tests;
