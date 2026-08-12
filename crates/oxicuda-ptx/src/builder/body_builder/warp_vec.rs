//! `WarpVec` — a portable-SIMD-flavored warp-vector expression layer.
//!
//! A GPU warp *is* a vector unit: 32 lanes executing in lockstep, one scalar
//! register per lane. This module makes that duality first-class inside
//! [`KernelBuilder::body`] closures, in the spirit of Rust's `core::simd`
//! mapped onto warps (the "SIMT ⇄ SIMD" correspondence): a [`WarpVec`] is a
//! logical `Simd<T, N>` whose lanes live in the warp, and a [`WarpMask`] is
//! its per-lane `Mask<T, N>`.
//!
//! | `core::simd` concept      | `WarpVec` lowering                          |
//! |---------------------------|---------------------------------------------|
//! | elementwise `+ - * min…`  | per-thread scalar instructions              |
//! | `reduce_sum` / `reduce_*` | `shfl.sync.bfly` butterfly (or `redux.sync`)|
//! | `simd_swizzle!` / rotate  | `shfl.sync.idx` / `.up` / `.down`           |
//! | comparisons → `Mask`      | `setp.*` predicate registers                |
//! | `select`                  | `selp.*`                                    |
//! | mask `any` / `all`        | `vote.sync.any` / `.all`                    |
//! | `to_bitmask`              | `vote.sync.ballot.b32`                      |
//!
//! Unlike a compiler-level portable-SIMD port, this layer is runtime PTX
//! metaprogramming on stable Rust: the lowering is explicit, inspectable, and
//! architecture-aware (integer full-warp reductions use `redux.sync` on
//! `sm_80`+ automatically).
//!
//! # Lane width
//!
//! The hardware warp is always 32 lanes ([`WARP_SIZE`]), but every `WarpVec`
//! carries a *logical* width `w ∈ {2, 4, 8, 16, 32}`: horizontal operations
//! (reductions, scans, shuffles) act independently within each `w`-lane
//! segment, using the hardware's segmented-shuffle encoding. Width is part of
//! the value, checked on every binary operation — the article-era lesson that
//! warp width must be a parameter, not an assumption, applied from day one.
//!
//! # Convergence requirement
//!
//! Every lane of the warp must execute the horizontal operations (shuffles,
//! votes, reductions, scans): do not place `WarpVec` horizontal ops under
//! divergent control flow such as [`BodyBuilder::if_lt_u32`]. Elementwise
//! operations have no such requirement.
//!
//! # Example: the `relu_dot` kernel
//!
//! One warp computes `Σ max(x[i]·y[i], 0)` over its 32 elements:
//!
//! ```
//! use oxicuda_ptx::prelude::*;
//! use oxicuda_ptx::builder::warp_vec::WarpVec;
//!
//! let ptx = KernelBuilder::new("relu_dot_warp")
//!     .target(SmVersion::Sm86)
//!     .param("out", PtxType::U64)
//!     .param("x", PtxType::U64)
//!     .param("y", PtxType::U64)
//!     .body(|b| {
//!         let x_ptr = b.load_param_u64("x");
//!         let y_ptr = b.load_param_u64("y");
//!         let lane = b.lane_id();
//!         let x_addr = b.f32_elem_addr(x_ptr, lane.clone());
//!         let y_addr = b.f32_elem_addr(y_ptr, lane.clone());
//!         let x_val = b.load_global_f32(x_addr);
//!         let y_val = b.load_global_f32(y_addr);
//!         let x = WarpVec::from_register(x_val).expect("f32 vec");
//!         let y = WarpVec::from_register(y_val).expect("f32 vec");
//!
//!         let dot = x
//!             .mul(b, &y).expect("mul")
//!             .relu(b).expect("relu")
//!             .reduce_sum(b).expect("reduce");
//!
//!         // Lane 0 stores the warp's aggregate.
//!         let out_ptr = b.load_param_u64("out");
//!         let one = b.mov_imm_u32(1);
//!         b.if_lt_u32(lane, one, |b| {
//!             b.store_global_f32(out_ptr.clone(), dot.register().clone());
//!         });
//!         b.ret();
//!     })
//!     .build()
//!     .expect("PTX generation failed");
//!
//! assert!(ptx.contains("shfl.sync.bfly.b32"));
//! ```
//!
//! [`KernelBuilder::body`]: crate::builder::KernelBuilder::body
//! [`BodyBuilder::if_lt_u32`]: super::BodyBuilder::if_lt_u32

use crate::error::PtxGenError;
use crate::ir::{
    CmpOp, ImmValue, Instruction, MulMode, Operand, PtxType, ReduxOp, Register, RoundingMode,
    ShflMode,
};

use super::BodyBuilder;
use super::warp_ops::{WARP_SIZE, is_valid_warp_width};

// ═══════════════════════════════════════════════════════════════════════════
//  Operation selectors
// ═══════════════════════════════════════════════════════════════════════════

/// Combining operation for [`WarpVec::reduce`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarpReduceOp {
    /// Sum of all lanes.
    Sum,
    /// Product of all lanes.
    Prod,
    /// Minimum across lanes.
    Min,
    /// Maximum across lanes.
    Max,
    /// Bitwise AND across lanes (integer / bit types only).
    BitAnd,
    /// Bitwise OR across lanes (integer / bit types only).
    BitOr,
    /// Bitwise XOR across lanes (integer / bit types only).
    BitXor,
}

/// Scan flavor for [`WarpVec::scan_sum`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarpScanMode {
    /// Lane *i* receives `Σ lanes[0..=i]` (within its segment).
    Inclusive,
    /// Lane *i* receives `Σ lanes[0..i]`; lane 0 of each segment receives 0.
    Exclusive,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Type classification helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Element types a `WarpVec` may hold.
const fn is_warp_vec_ty(ty: PtxType) -> bool {
    matches!(
        ty,
        PtxType::F32
            | PtxType::F64
            | PtxType::U32
            | PtxType::S32
            | PtxType::B32
            | PtxType::U64
            | PtxType::S64
            | PtxType::B64
    )
}

const fn is_float(ty: PtxType) -> bool {
    matches!(ty, PtxType::F32 | PtxType::F64)
}

const fn is_signed_int(ty: PtxType) -> bool {
    matches!(ty, PtxType::S32 | PtxType::S64)
}

const fn is_unsigned_int(ty: PtxType) -> bool {
    matches!(ty, PtxType::U32 | PtxType::U64)
}

const fn is_bit(ty: PtxType) -> bool {
    matches!(ty, PtxType::B32 | PtxType::B64)
}

/// Numeric types: everything except the raw bit classes.
const fn is_numeric(ty: PtxType) -> bool {
    is_float(ty) || is_signed_int(ty) || is_unsigned_int(ty)
}

/// The bit type of matching width, for bitwise-logic instructions.
const fn bit_ty_for(ty: PtxType) -> PtxType {
    match ty {
        PtxType::F64 | PtxType::U64 | PtxType::S64 | PtxType::B64 => PtxType::B64,
        _ => PtxType::B32,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  WarpVec
// ═══════════════════════════════════════════════════════════════════════════

/// A per-lane scalar register viewed as one lane of a warp-wide vector.
///
/// See the [module documentation](self) for the programming model. Values are
/// cheap to clone (a register name and a width) and immutable: every
/// operation allocates a fresh destination register, mirroring the SSA-like
/// style of the rest of the builder.
#[derive(Debug, Clone)]
pub struct WarpVec {
    /// The per-lane value register.
    reg: Register,
    /// Logical segment width (power of two in `2..=32`).
    width: u32,
}

impl WarpVec {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Wraps an existing per-lane register as a full-warp (width 32) vector.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if the register type is not
    /// a 32-bit or 64-bit element class (`f32/f64/u32/s32/b32/u64/s64/b64`).
    pub fn from_register(reg: Register) -> Result<Self, PtxGenError> {
        Self::from_register_segmented(reg, WARP_SIZE)
    }

    /// Wraps an existing per-lane register as a segmented vector: horizontal
    /// operations act independently within each `width`-lane segment.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if `width` is not a power of
    /// two in `2..=32`, or the register type is not a supported element class.
    pub fn from_register_segmented(reg: Register, width: u32) -> Result<Self, PtxGenError> {
        if !is_valid_warp_width(width) {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec width must be a power of two in 2..=32, got {width}"
            )));
        }
        if !is_warp_vec_ty(reg.ty) {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec element type must be f32/f64/u32/s32/b32/u64/s64/b64, got {:?}",
                reg.ty
            )));
        }
        Ok(Self { reg, width })
    }

    /// Broadcasts an `f32` immediate into every lane.
    pub fn splat_f32(b: &mut BodyBuilder<'_>, value: f32) -> Self {
        let reg = b.mov_typed(PtxType::F32, Operand::Immediate(ImmValue::F32(value)));
        Self {
            reg,
            width: WARP_SIZE,
        }
    }

    /// Broadcasts an `f64` immediate into every lane.
    pub fn splat_f64(b: &mut BodyBuilder<'_>, value: f64) -> Self {
        let reg = b.mov_typed(PtxType::F64, Operand::Immediate(ImmValue::F64(value)));
        Self {
            reg,
            width: WARP_SIZE,
        }
    }

    /// Broadcasts a `u32` immediate into every lane.
    pub fn splat_u32(b: &mut BodyBuilder<'_>, value: u32) -> Self {
        let reg = b.mov_typed(PtxType::U32, Operand::Immediate(ImmValue::U32(value)));
        Self {
            reg,
            width: WARP_SIZE,
        }
    }

    /// Broadcasts an `s32` immediate into every lane.
    pub fn splat_s32(b: &mut BodyBuilder<'_>, value: i32) -> Self {
        let reg = b.mov_typed(PtxType::S32, Operand::Immediate(ImmValue::S32(value)));
        Self {
            reg,
            width: WARP_SIZE,
        }
    }

    /// The identity vector `[0, 1, 2, …, 31]` — each lane holds its own
    /// hardware lane index (`%laneid`, type `u32`, full width).
    pub fn lane_ids(b: &mut BodyBuilder<'_>) -> Self {
        let reg = b.lane_id();
        Self {
            reg,
            width: WARP_SIZE,
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /// The underlying per-lane register.
    #[must_use]
    pub const fn register(&self) -> &Register {
        &self.reg
    }

    /// Consumes the vector, returning the per-lane register.
    #[must_use]
    pub fn into_register(self) -> Register {
        self.reg
    }

    /// The element type.
    #[must_use]
    pub const fn ty(&self) -> PtxType {
        self.reg.ty
    }

    /// The logical segment width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Reinterprets this vector at a different segment width (no code is
    /// emitted; only the horizontal-operation geometry changes).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if `width` is not a power of
    /// two in `2..=32`.
    pub fn with_width(&self, width: u32) -> Result<Self, PtxGenError> {
        Self::from_register_segmented(self.reg.clone(), width)
    }

    /// Checks that `other` is elementwise-compatible (same type and width).
    fn check_binary(&self, other: &Self, what: &str) -> Result<(), PtxGenError> {
        if self.reg.ty != other.reg.ty {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::{what}: element type mismatch ({:?} vs {:?})",
                self.reg.ty, other.reg.ty
            )));
        }
        if self.width != other.width {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::{what}: segment width mismatch ({} vs {})",
                self.width, other.width
            )));
        }
        Ok(())
    }

    /// Wraps a freshly produced register with this vector's width.
    const fn derive(&self, reg: Register) -> Self {
        Self {
            reg,
            width: self.width,
        }
    }

    // ── Elementwise arithmetic ───────────────────────────────────────────

    /// Elementwise addition.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch or a
    /// non-numeric element type.
    pub fn add(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<Self, PtxGenError> {
        self.check_binary(other, "add")?;
        self.require_numeric("add")?;
        let dst = b.alloc_reg(self.reg.ty);
        b.emit(Instruction::Add {
            ty: self.reg.ty,
            dst: dst.clone(),
            a: Operand::Register(self.reg.clone()),
            b: Operand::Register(other.reg.clone()),
        });
        Ok(self.derive(dst))
    }

    /// Elementwise subtraction (`self - other`).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch or a
    /// non-numeric element type.
    pub fn sub(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<Self, PtxGenError> {
        self.check_binary(other, "sub")?;
        self.require_numeric("sub")?;
        let dst = b.alloc_reg(self.reg.ty);
        b.emit(Instruction::Sub {
            ty: self.reg.ty,
            dst: dst.clone(),
            a: Operand::Register(self.reg.clone()),
            b: Operand::Register(other.reg.clone()),
        });
        Ok(self.derive(dst))
    }

    /// Elementwise multiplication: `mul.rn` for floats, `mul.lo` for integers.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch or a
    /// non-numeric element type.
    pub fn mul(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<Self, PtxGenError> {
        self.check_binary(other, "mul")?;
        self.require_numeric("mul")?;
        let mode = if is_float(self.reg.ty) {
            MulMode::Rn
        } else {
            MulMode::Lo
        };
        let dst = b.alloc_reg(self.reg.ty);
        b.emit(Instruction::Mul {
            ty: self.reg.ty,
            mode,
            dst: dst.clone(),
            a: Operand::Register(self.reg.clone()),
            b: Operand::Register(other.reg.clone()),
        });
        Ok(self.derive(dst))
    }

    /// Elementwise fused multiply-add: `self * m + a` (floats only,
    /// round-to-nearest-even).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch or a
    /// non-float element type.
    pub fn fma(&self, b: &mut BodyBuilder<'_>, m: &Self, a: &Self) -> Result<Self, PtxGenError> {
        self.check_binary(m, "fma")?;
        self.check_binary(a, "fma")?;
        if !is_float(self.reg.ty) {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::fma requires f32/f64 elements, got {:?}",
                self.reg.ty
            )));
        }
        let dst = b.alloc_reg(self.reg.ty);
        b.emit(Instruction::Fma {
            rnd: RoundingMode::Rn,
            ty: self.reg.ty,
            dst: dst.clone(),
            a: Operand::Register(self.reg.clone()),
            b: Operand::Register(m.reg.clone()),
            c: Operand::Register(a.reg.clone()),
        });
        Ok(self.derive(dst))
    }

    /// Elementwise minimum.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch or a
    /// non-numeric element type.
    pub fn min(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<Self, PtxGenError> {
        self.check_binary(other, "min")?;
        self.require_numeric("min")?;
        let dst = b.alloc_reg(self.reg.ty);
        b.emit(Instruction::Min {
            ty: self.reg.ty,
            dst: dst.clone(),
            a: Operand::Register(self.reg.clone()),
            b: Operand::Register(other.reg.clone()),
        });
        Ok(self.derive(dst))
    }

    /// Elementwise maximum.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch or a
    /// non-numeric element type.
    pub fn max(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<Self, PtxGenError> {
        self.check_binary(other, "max")?;
        self.require_numeric("max")?;
        let dst = b.alloc_reg(self.reg.ty);
        b.emit(Instruction::Max {
            ty: self.reg.ty,
            dst: dst.clone(),
            a: Operand::Register(self.reg.clone()),
            b: Operand::Register(other.reg.clone()),
        });
        Ok(self.derive(dst))
    }

    /// Elementwise negation (floats and signed integers).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] for unsigned or bit types.
    pub fn neg(&self, b: &mut BodyBuilder<'_>) -> Result<Self, PtxGenError> {
        if !is_float(self.reg.ty) && !is_signed_int(self.reg.ty) {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::neg requires float or signed elements, got {:?}",
                self.reg.ty
            )));
        }
        let dst = b.alloc_reg(self.reg.ty);
        b.emit(Instruction::Neg {
            ty: self.reg.ty,
            dst: dst.clone(),
            src: Operand::Register(self.reg.clone()),
        });
        Ok(self.derive(dst))
    }

    /// Elementwise absolute value (floats and signed integers).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] for unsigned or bit types.
    pub fn abs(&self, b: &mut BodyBuilder<'_>) -> Result<Self, PtxGenError> {
        if !is_float(self.reg.ty) && !is_signed_int(self.reg.ty) {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::abs requires float or signed elements, got {:?}",
                self.reg.ty
            )));
        }
        let dst = b.alloc_reg(self.reg.ty);
        b.emit(Instruction::Abs {
            ty: self.reg.ty,
            dst: dst.clone(),
            src: Operand::Register(self.reg.clone()),
        });
        Ok(self.derive(dst))
    }

    /// Elementwise square root (floats, round-to-nearest-even).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] for non-float element types.
    pub fn sqrt(&self, b: &mut BodyBuilder<'_>) -> Result<Self, PtxGenError> {
        if !is_float(self.reg.ty) {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::sqrt requires f32/f64 elements, got {:?}",
                self.reg.ty
            )));
        }
        let dst = b.alloc_reg(self.reg.ty);
        b.emit(Instruction::Sqrt {
            rnd: Some(RoundingMode::Rn),
            ty: self.reg.ty,
            dst: dst.clone(),
            src: Operand::Register(self.reg.clone()),
        });
        Ok(self.derive(dst))
    }

    /// Elementwise rectified linear unit — `max(self, 0)` (floats and
    /// signed integers).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] for unsigned or bit types
    /// (where the operation would be the identity or ill-defined).
    pub fn relu(&self, b: &mut BodyBuilder<'_>) -> Result<Self, PtxGenError> {
        if !is_float(self.reg.ty) && !is_signed_int(self.reg.ty) {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::relu requires float or signed elements, got {:?}",
                self.reg.ty
            )));
        }
        let zero = b.mov_typed(self.reg.ty, Operand::Immediate(zero_imm(self.reg.ty)));
        let dst = b.alloc_reg(self.reg.ty);
        b.emit(Instruction::Max {
            ty: self.reg.ty,
            dst: dst.clone(),
            a: Operand::Register(self.reg.clone()),
            b: Operand::Register(zero),
        });
        Ok(self.derive(dst))
    }

    fn require_numeric(&self, what: &str) -> Result<(), PtxGenError> {
        if is_numeric(self.reg.ty) {
            Ok(())
        } else {
            Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::{what} requires a numeric element type, got {:?}",
                self.reg.ty
            )))
        }
    }

    // ── Comparisons → WarpMask ───────────────────────────────────────────

    /// Elementwise comparison with an explicit PTX comparison operator.
    ///
    /// Prefer the semantic wrappers ([`gt`](Self::gt), [`le`](Self::le), …)
    /// which pick the signed/unsigned/float operator automatically.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch.
    pub fn cmp(
        &self,
        b: &mut BodyBuilder<'_>,
        op: CmpOp,
        other: &Self,
    ) -> Result<WarpMask, PtxGenError> {
        self.check_binary(other, "cmp")?;
        let pred = b.alloc_reg(PtxType::Pred);
        b.emit(Instruction::SetP {
            cmp: op,
            ty: self.reg.ty,
            dst: pred.clone(),
            a: Operand::Register(self.reg.clone()),
            b: Operand::Register(other.reg.clone()),
        });
        Ok(WarpMask {
            pred,
            width: self.width,
        })
    }

    /// Elementwise `self > other`.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch or a
    /// bit-typed element (ordered comparison undefined on raw bits).
    pub fn gt(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<WarpMask, PtxGenError> {
        self.ordered_cmp(b, other, CmpOp::Gt, CmpOp::Hi, "gt")
    }

    /// Elementwise `self >= other`.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch or a
    /// bit-typed element.
    pub fn ge(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<WarpMask, PtxGenError> {
        self.ordered_cmp(b, other, CmpOp::Ge, CmpOp::Hs, "ge")
    }

    /// Elementwise `self < other`.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch or a
    /// bit-typed element.
    pub fn lt(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<WarpMask, PtxGenError> {
        self.ordered_cmp(b, other, CmpOp::Lt, CmpOp::Lo, "lt")
    }

    /// Elementwise `self <= other`.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch or a
    /// bit-typed element.
    pub fn le(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<WarpMask, PtxGenError> {
        self.ordered_cmp(b, other, CmpOp::Le, CmpOp::Ls, "le")
    }

    /// Elementwise `self == other`.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch.
    pub fn eq(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<WarpMask, PtxGenError> {
        self.cmp(b, CmpOp::Eq, other)
    }

    /// Elementwise `self != other`.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on type/width mismatch.
    pub fn ne(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<WarpMask, PtxGenError> {
        self.cmp(b, CmpOp::Ne, other)
    }

    /// Ordered comparison choosing the signed/float vs unsigned operator.
    fn ordered_cmp(
        &self,
        b: &mut BodyBuilder<'_>,
        other: &Self,
        signed_op: CmpOp,
        unsigned_op: CmpOp,
        what: &str,
    ) -> Result<WarpMask, PtxGenError> {
        if is_bit(self.reg.ty) {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::{what}: ordered comparison is undefined on bit type {:?} \
                 (use eq/ne, or reinterpret as a numeric type)",
                self.reg.ty
            )));
        }
        let op = if is_unsigned_int(self.reg.ty) {
            unsigned_op
        } else {
            signed_op
        };
        self.cmp(b, op, other)
    }

    // ── Horizontal reductions ────────────────────────────────────────────

    /// All-lanes reduction: every lane of each segment receives the
    /// segment's aggregate.
    ///
    /// Lowering:
    ///
    /// * **Fast path** — full-width `u32` `Sum`/`Min`/`Max`/`BitAnd`/
    ///   `BitOr`/`BitXor` on `sm_80`+ targets lowers to one `redux.sync`.
    /// * **General path** — a `log2(width)`-round `shfl.sync.bfly`
    ///   butterfly. Because the butterfly is an all-lanes exchange, no
    ///   broadcast step is needed afterwards.
    ///
    /// All 32 lanes must execute this operation (see the module docs on
    /// convergence).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if the operation is not
    /// defined for the element type (`Prod`/`Sum`/`Min`/`Max` need numeric
    /// elements; `Bit*` need integer or bit elements).
    pub fn reduce(&self, b: &mut BodyBuilder<'_>, op: WarpReduceOp) -> Result<Self, PtxGenError> {
        match op {
            WarpReduceOp::Sum | WarpReduceOp::Prod | WarpReduceOp::Min | WarpReduceOp::Max => {
                self.require_numeric("reduce")?;
            }
            WarpReduceOp::BitAnd | WarpReduceOp::BitOr | WarpReduceOp::BitXor => {
                if is_float(self.reg.ty) {
                    return Err(PtxGenError::GenerationFailed(format!(
                        "WarpVec::reduce: bitwise reduction is undefined on float type {:?}",
                        self.reg.ty
                    )));
                }
            }
        }

        // Fast path: redux.sync (integer, full warp, sm_80+). The IR emits
        // `redux.sync.<op>.u32`, so it is only taken for u32 elements.
        if self.reg.ty == PtxType::U32
            && self.width == WARP_SIZE
            && b.target_sm().capabilities().has_redux
        {
            if let Some(redux_op) = redux_op_for(op) {
                let dst = b.alloc_reg(PtxType::U32);
                b.emit(Instruction::Redux {
                    op: redux_op,
                    dst: dst.clone(),
                    src: Operand::Register(self.reg.clone()),
                    membership_mask: super::warp_ops::FULL_WARP_MASK,
                });
                return Ok(self.derive(dst));
            }
        }

        // General path: XOR-butterfly all-reduce.
        let mut acc = self.reg.clone();
        let mut offset = self.width / 2;
        while offset >= 1 {
            let partner = b.shfl_sync(
                ShflMode::Bfly,
                &acc,
                Operand::Immediate(ImmValue::U32(offset)),
                self.width,
            )?;
            acc = combine(b, op, self.reg.ty, &acc, &partner);
            offset /= 2;
        }
        Ok(self.derive(acc))
    }

    /// All-lanes sum — sugar for [`reduce`](Self::reduce) with
    /// [`WarpReduceOp::Sum`].
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] for non-numeric elements.
    pub fn reduce_sum(&self, b: &mut BodyBuilder<'_>) -> Result<Self, PtxGenError> {
        self.reduce(b, WarpReduceOp::Sum)
    }

    /// All-lanes product.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] for non-numeric elements.
    pub fn reduce_prod(&self, b: &mut BodyBuilder<'_>) -> Result<Self, PtxGenError> {
        self.reduce(b, WarpReduceOp::Prod)
    }

    /// All-lanes minimum.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] for non-numeric elements.
    pub fn reduce_min(&self, b: &mut BodyBuilder<'_>) -> Result<Self, PtxGenError> {
        self.reduce(b, WarpReduceOp::Min)
    }

    /// All-lanes maximum.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] for non-numeric elements.
    pub fn reduce_max(&self, b: &mut BodyBuilder<'_>) -> Result<Self, PtxGenError> {
        self.reduce(b, WarpReduceOp::Max)
    }

    // ── Prefix scan ──────────────────────────────────────────────────────

    /// Per-segment prefix sum (Hillis–Steele over `shfl.sync.up`).
    ///
    /// Each round shifts the accumulator up by a doubling offset and adds it
    /// where the shuffle's in-range predicate is true — the guarded-add
    /// formulation (`acc = valid ? acc + shifted : acc`), which needs no
    /// additive identity and therefore preserves `-0.0` in float scans.
    ///
    /// For [`WarpScanMode::Exclusive`], the inclusive result is shifted up
    /// one more lane and lane 0 of each segment receives `0`.
    ///
    /// All 32 lanes must execute this operation.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] for non-numeric elements.
    pub fn scan_sum(
        &self,
        b: &mut BodyBuilder<'_>,
        mode: WarpScanMode,
    ) -> Result<Self, PtxGenError> {
        self.require_numeric("scan_sum")?;

        let mut acc = self.reg.clone();
        let mut offset = 1;
        while offset < self.width {
            let (shifted, valid) = b.shfl_sync_with_valid(
                ShflMode::Up,
                &acc,
                Operand::Immediate(ImmValue::U32(offset)),
                self.width,
            )?;
            let summed = b.alloc_reg(self.reg.ty);
            b.emit(Instruction::Add {
                ty: self.reg.ty,
                dst: summed.clone(),
                a: Operand::Register(acc.clone()),
                b: Operand::Register(shifted),
            });
            acc = b.selp(self.reg.ty, summed, acc, valid);
            offset *= 2;
        }

        if matches!(mode, WarpScanMode::Exclusive) {
            let (shifted, valid) = b.shfl_sync_with_valid(
                ShflMode::Up,
                &acc,
                Operand::Immediate(ImmValue::U32(1)),
                self.width,
            )?;
            let zero = b.mov_typed(self.reg.ty, Operand::Immediate(zero_imm(self.reg.ty)));
            acc = b.selp(self.reg.ty, shifted, zero, valid);
        }
        Ok(self.derive(acc))
    }

    // ── Cross-lane shuffles ──────────────────────────────────────────────

    /// Every lane receives the value held by segment-relative lane `lane`
    /// (`shfl.sync.idx` — the vector `broadcast`).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if `lane >= width`.
    pub fn broadcast(&self, b: &mut BodyBuilder<'_>, lane: u32) -> Result<Self, PtxGenError> {
        if lane >= self.width {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::broadcast: lane {lane} out of range for width {}",
                self.width
            )));
        }
        let reg = b.shfl_sync(
            ShflMode::Idx,
            &self.reg,
            Operand::Immediate(ImmValue::U32(lane)),
            self.width,
        )?;
        Ok(self.derive(reg))
    }

    /// Lane *i* receives the value of lane `i - delta` within its segment
    /// (`shfl.sync.up`). The lowest `delta` lanes of each segment keep their
    /// own value (hardware clamp semantics).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if `delta >= width`.
    pub fn shuffle_up(&self, b: &mut BodyBuilder<'_>, delta: u32) -> Result<Self, PtxGenError> {
        self.relative_shuffle(b, ShflMode::Up, delta, "shuffle_up")
    }

    /// Lane *i* receives the value of lane `i + delta` within its segment
    /// (`shfl.sync.down`). The highest `delta` lanes of each segment keep
    /// their own value.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if `delta >= width`.
    pub fn shuffle_down(&self, b: &mut BodyBuilder<'_>, delta: u32) -> Result<Self, PtxGenError> {
        self.relative_shuffle(b, ShflMode::Down, delta, "shuffle_down")
    }

    /// Lane *i* exchanges with lane `i ^ xor_mask` within its segment
    /// (`shfl.sync.bfly`).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if `xor_mask >= width`.
    pub fn butterfly(&self, b: &mut BodyBuilder<'_>, xor_mask: u32) -> Result<Self, PtxGenError> {
        self.relative_shuffle(b, ShflMode::Bfly, xor_mask, "butterfly")
    }

    /// Reverses the lanes of each segment — `butterfly(width - 1)`.
    ///
    /// # Errors
    ///
    /// Propagates [`PtxGenError::GenerationFailed`] from the underlying
    /// shuffle (cannot occur for a validly constructed vector).
    pub fn reverse(&self, b: &mut BodyBuilder<'_>) -> Result<Self, PtxGenError> {
        self.butterfly(b, self.width - 1)
    }

    /// Dynamic swizzle: lane *i* receives the value of the segment-relative
    /// lane selected by `indices` lane *i* (the `simd_swizzle!` analogue with
    /// a runtime index vector). Indices are taken modulo the segment by the
    /// hardware's segmented-shuffle encoding.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if `indices` is not
    /// `u32`/`s32`-typed or the widths differ.
    pub fn shuffle_idx(
        &self,
        b: &mut BodyBuilder<'_>,
        indices: &Self,
    ) -> Result<Self, PtxGenError> {
        if !matches!(indices.reg.ty, PtxType::U32 | PtxType::S32) {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::shuffle_idx: index vector must be u32/s32, got {:?}",
                indices.reg.ty
            )));
        }
        if indices.width != self.width {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::shuffle_idx: index width {} != value width {}",
                indices.width, self.width
            )));
        }
        let reg = b.shfl_sync(
            ShflMode::Idx,
            &self.reg,
            Operand::Register(indices.reg.clone()),
            self.width,
        )?;
        Ok(self.derive(reg))
    }

    /// Shared bounds-checked relative shuffle.
    fn relative_shuffle(
        &self,
        b: &mut BodyBuilder<'_>,
        mode: ShflMode,
        amount: u32,
        what: &str,
    ) -> Result<Self, PtxGenError> {
        if amount >= self.width {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpVec::{what}: amount {amount} out of range for width {}",
                self.width
            )));
        }
        let reg = b.shfl_sync(
            mode,
            &self.reg,
            Operand::Immediate(ImmValue::U32(amount)),
            self.width,
        )?;
        Ok(self.derive(reg))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  WarpMask
// ═══════════════════════════════════════════════════════════════════════════

/// A per-lane predicate viewed as one lane of a warp-wide mask — the
/// `Mask<T, N>` counterpart of [`WarpVec`].
#[derive(Debug, Clone)]
pub struct WarpMask {
    /// The per-lane predicate register.
    pred: Register,
    /// Logical segment width (power of two in `2..=32`).
    width: u32,
}

impl WarpMask {
    /// Wraps an existing predicate register as a full-warp mask.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if the register is not a
    /// predicate.
    pub fn from_predicate(pred: Register) -> Result<Self, PtxGenError> {
        Self::from_predicate_segmented(pred, WARP_SIZE)
    }

    /// Wraps an existing predicate register as a segmented mask.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if the register is not a
    /// predicate or `width` is not a power of two in `2..=32`.
    pub fn from_predicate_segmented(pred: Register, width: u32) -> Result<Self, PtxGenError> {
        if pred.ty != PtxType::Pred {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpMask requires a predicate register, got {:?}",
                pred.ty
            )));
        }
        if !is_valid_warp_width(width) {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpMask width must be a power of two in 2..=32, got {width}"
            )));
        }
        Ok(Self { pred, width })
    }

    /// The underlying per-lane predicate register.
    #[must_use]
    pub const fn predicate(&self) -> &Register {
        &self.pred
    }

    /// The logical segment width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Per-lane select: lane *i* receives `on_true[i]` where the mask is
    /// true, else `on_false[i]` (`selp` — branch-free, no divergence).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if the value vectors differ
    /// in type or width, or their width differs from the mask's.
    pub fn select(
        &self,
        b: &mut BodyBuilder<'_>,
        on_true: &WarpVec,
        on_false: &WarpVec,
    ) -> Result<WarpVec, PtxGenError> {
        on_true.check_binary(on_false, "select")?;
        if on_true.width != self.width {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpMask::select: mask width {} != value width {}",
                self.width, on_true.width
            )));
        }
        let reg = b.selp(
            on_true.reg.ty,
            on_true.reg.clone(),
            on_false.reg.clone(),
            self.pred.clone(),
        );
        Ok(WarpVec {
            reg,
            width: self.width,
        })
    }

    /// Lanewise logical AND.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on width mismatch.
    pub fn and(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<Self, PtxGenError> {
        self.logic(b, other, "and", |dst, a, o| Instruction::And {
            ty: PtxType::Pred,
            dst,
            a,
            b: o,
        })
    }

    /// Lanewise logical OR.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on width mismatch.
    pub fn or(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<Self, PtxGenError> {
        self.logic(b, other, "or", |dst, a, o| Instruction::Or {
            ty: PtxType::Pred,
            dst,
            a,
            b: o,
        })
    }

    /// Lanewise logical XOR.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] on width mismatch.
    pub fn xor(&self, b: &mut BodyBuilder<'_>, other: &Self) -> Result<Self, PtxGenError> {
        self.logic(b, other, "xor", |dst, a, o| Instruction::Xor {
            ty: PtxType::Pred,
            dst,
            a,
            b: o,
        })
    }

    /// Lanewise logical NOT.
    #[must_use]
    pub fn not(&self, b: &mut BodyBuilder<'_>) -> Self {
        let pred = b.not_pred(&self.pred);
        Self {
            pred,
            width: self.width,
        }
    }

    fn logic(
        &self,
        b: &mut BodyBuilder<'_>,
        other: &Self,
        what: &str,
        make: impl FnOnce(Register, Operand, Operand) -> Instruction,
    ) -> Result<Self, PtxGenError> {
        if self.width != other.width {
            return Err(PtxGenError::GenerationFailed(format!(
                "WarpMask::{what}: segment width mismatch ({} vs {})",
                self.width, other.width
            )));
        }
        let dst = b.alloc_reg(PtxType::Pred);
        b.emit(make(
            dst.clone(),
            Operand::Register(self.pred.clone()),
            Operand::Register(other.pred.clone()),
        ));
        Ok(Self {
            pred: dst,
            width: self.width,
        })
    }

    /// Horizontal ANY: every lane of a segment receives `true` iff the mask
    /// is true in at least one lane of that segment.
    ///
    /// Full-width masks lower to one `vote.sync.any`; segmented masks lower
    /// to a `ballot` intersected with a per-lane segment mask.
    ///
    /// All 32 lanes must execute this operation.
    ///
    /// # Errors
    ///
    /// Propagates internal generation failures (cannot occur for a validly
    /// constructed mask).
    pub fn any(&self, b: &mut BodyBuilder<'_>) -> Result<Self, PtxGenError> {
        if self.width == WARP_SIZE {
            let pred = b.vote_any(&self.pred);
            return Ok(Self {
                pred,
                width: self.width,
            });
        }
        // Segmented: (ballot & segment_mask) != 0, evaluated per lane.
        let (masked, _seg_mask) = self.segment_ballot(b);
        let zero = b.mov_typed(PtxType::B32, Operand::Immediate(ImmValue::U32(0)));
        let pred = b.alloc_reg(PtxType::Pred);
        b.emit(Instruction::SetP {
            cmp: CmpOp::Ne,
            ty: PtxType::B32,
            dst: pred.clone(),
            a: Operand::Register(masked),
            b: Operand::Register(zero),
        });
        Ok(Self {
            pred,
            width: self.width,
        })
    }

    /// Horizontal ALL: every lane of a segment receives `true` iff the mask
    /// is true in every lane of that segment.
    ///
    /// All 32 lanes must execute this operation.
    ///
    /// # Errors
    ///
    /// Propagates internal generation failures (cannot occur for a validly
    /// constructed mask).
    pub fn all(&self, b: &mut BodyBuilder<'_>) -> Result<Self, PtxGenError> {
        if self.width == WARP_SIZE {
            let pred = b.vote_all(&self.pred);
            return Ok(Self {
                pred,
                width: self.width,
            });
        }
        // Segmented: (ballot & segment_mask) == segment_mask, per lane.
        let (masked, seg_mask) = self.segment_ballot(b);
        let pred = b.alloc_reg(PtxType::Pred);
        b.emit(Instruction::SetP {
            cmp: CmpOp::Eq,
            ty: PtxType::B32,
            dst: pred.clone(),
            a: Operand::Register(masked),
            b: Operand::Register(seg_mask),
        });
        Ok(Self {
            pred,
            width: self.width,
        })
    }

    /// The warp-wide ballot bitmask: bit *i* is set iff lane *i*'s predicate
    /// is true (`vote.sync.ballot.b32`; every lane receives the same 32-bit
    /// value, regardless of segmentation).
    ///
    /// All 32 lanes must execute this operation.
    #[must_use]
    pub fn ballot(&self, b: &mut BodyBuilder<'_>) -> Register {
        b.vote_ballot(&self.pred)
    }

    /// Population count: every lane of a segment receives the number of true
    /// lanes in its segment (`popc` of the segment-masked ballot).
    ///
    /// All 32 lanes must execute this operation.
    ///
    /// # Errors
    ///
    /// Propagates internal generation failures (cannot occur for a validly
    /// constructed mask).
    pub fn count(&self, b: &mut BodyBuilder<'_>) -> Result<WarpVec, PtxGenError> {
        let counted = if self.width == WARP_SIZE {
            let ballot = b.vote_ballot(&self.pred);
            b.popc_b32(ballot)
        } else {
            let (masked, _seg_mask) = self.segment_ballot(b);
            b.popc_b32(masked)
        };
        Ok(WarpVec {
            reg: counted,
            width: self.width,
        })
    }

    /// Computes `(ballot & segment_mask, segment_mask)` where
    /// `segment_mask = ((1 << width) - 1) << (laneid & !(width - 1))` — the
    /// 32-bit mask covering this lane's segment.
    fn segment_ballot(&self, b: &mut BodyBuilder<'_>) -> (Register, Register) {
        let ballot = b.vote_ballot(&self.pred);
        let lane = b.lane_id();
        let base_mask = b.mov_typed(
            PtxType::U32,
            Operand::Immediate(ImmValue::U32(!(self.width - 1) & (WARP_SIZE - 1))),
        );
        // Segment base lane: laneid & !(width - 1).
        let seg_base = b.alloc_reg(PtxType::B32);
        b.emit(Instruction::And {
            ty: PtxType::B32,
            dst: seg_base.clone(),
            a: Operand::Register(lane),
            b: Operand::Register(base_mask),
        });
        // Segment mask: ((1 << width) - 1) << seg_base. width < 32 here, so
        // the shifted constant fits in u32 without overflow.
        let ones = b.mov_typed(
            PtxType::B32,
            Operand::Immediate(ImmValue::U32((1_u32 << self.width) - 1)),
        );
        let seg_mask = b.shl_b32(ones, seg_base);
        let masked = b.alloc_reg(PtxType::B32);
        b.emit(Instruction::And {
            ty: PtxType::B32,
            dst: masked.clone(),
            a: Operand::Register(ballot),
            b: Operand::Register(seg_mask.clone()),
        });
        (masked, seg_mask)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Free helpers
// ═══════════════════════════════════════════════════════════════════════════

/// The all-zero immediate for a `WarpVec` element type.
const fn zero_imm(ty: PtxType) -> ImmValue {
    match ty {
        PtxType::F32 => ImmValue::F32(0.0),
        PtxType::F64 => ImmValue::F64(0.0),
        PtxType::S32 => ImmValue::S32(0),
        PtxType::S64 => ImmValue::S64(0),
        PtxType::U64 | PtxType::B64 => ImmValue::U64(0),
        // U32 / B32 and any remaining 32-bit class.
        _ => ImmValue::U32(0),
    }
}

/// Maps a [`WarpReduceOp`] to its `redux.sync` opcode, where one exists
/// (`Prod` has no hardware reduction).
const fn redux_op_for(op: WarpReduceOp) -> Option<ReduxOp> {
    match op {
        WarpReduceOp::Sum => Some(ReduxOp::Add),
        WarpReduceOp::Min => Some(ReduxOp::Min),
        WarpReduceOp::Max => Some(ReduxOp::Max),
        WarpReduceOp::BitAnd => Some(ReduxOp::And),
        WarpReduceOp::BitOr => Some(ReduxOp::Or),
        WarpReduceOp::BitXor => Some(ReduxOp::Xor),
        WarpReduceOp::Prod => None,
    }
}

#[cfg(test)]
#[path = "warp_vec_tests.rs"]
mod tests;

/// Emits one butterfly-round combine for [`WarpVec::reduce`].
fn combine(
    b: &mut BodyBuilder<'_>,
    op: WarpReduceOp,
    ty: PtxType,
    acc: &Register,
    partner: &Register,
) -> Register {
    let dst = b.alloc_reg(ty);
    let a = Operand::Register(acc.clone());
    let p = Operand::Register(partner.clone());
    let inst = match op {
        WarpReduceOp::Sum => Instruction::Add {
            ty,
            dst: dst.clone(),
            a,
            b: p,
        },
        WarpReduceOp::Prod => Instruction::Mul {
            ty,
            mode: if is_float(ty) {
                MulMode::Rn
            } else {
                MulMode::Lo
            },
            dst: dst.clone(),
            a,
            b: p,
        },
        WarpReduceOp::Min => Instruction::Min {
            ty,
            dst: dst.clone(),
            a,
            b: p,
        },
        WarpReduceOp::Max => Instruction::Max {
            ty,
            dst: dst.clone(),
            a,
            b: p,
        },
        WarpReduceOp::BitAnd => Instruction::And {
            ty: bit_ty_for(ty),
            dst: dst.clone(),
            a,
            b: p,
        },
        WarpReduceOp::BitOr => Instruction::Or {
            ty: bit_ty_for(ty),
            dst: dst.clone(),
            a,
            b: p,
        },
        WarpReduceOp::BitXor => Instruction::Xor {
            ty: bit_ty_for(ty),
            dst: dst.clone(),
            a,
            b: p,
        },
    };
    b.emit(inst);
    dst
}
