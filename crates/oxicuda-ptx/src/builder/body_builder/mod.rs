//! Instruction emission API for PTX kernel bodies.
//!
//! [`BodyBuilder`] provides the ergonomic instruction-level DSL used inside
//! [`KernelBuilder::body`] closures. It wraps a [`RegisterAllocator`] and an
//! instruction vector, exposing methods for parameter loading, thread IDs,
//! arithmetic, memory ops, control flow, synchronization, type conversions,
//! and Tensor Core MMA instructions.

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::{
    CacheQualifier, CmpOp, FenceScope, ImmValue, Instruction, MemorySpace, MmaShape, MulMode,
    Operand, PtxType, Register, RegisterAllocator, RoundingMode, SpecialReg, VectorWidth,
};

/// Instruction emission API for building the body of a PTX kernel.
///
/// `BodyBuilder` is not constructed directly — it is provided as a mutable
/// reference inside the closure passed to [`KernelBuilder::body`].
///
/// Most methods follow a consistent pattern: allocate destination register(s),
/// push the corresponding [`Instruction`] variant, and return the destination
/// register so it can be used as an operand to subsequent instructions.
///
/// [`KernelBuilder::body`]: super::KernelBuilder::body
pub struct BodyBuilder<'a> {
    /// Register allocator shared with the kernel builder.
    pub(super) regs: &'a mut RegisterAllocator,
    /// Instruction vector that accumulates the kernel body.
    pub(super) instructions: &'a mut Vec<Instruction>,
    /// Monotonically increasing label counter for generating unique labels.
    label_counter: u32,
    /// Names of the kernel parameters (for `load_param_*` methods).
    param_names: &'a [String],
    /// Target SM version (for architecture-gated instructions).
    pub(super) target: SmVersion,
}

impl<'a> BodyBuilder<'a> {
    /// Creates a new body builder.
    ///
    /// This is called internally by [`KernelBuilder::build`] — users should
    /// not need to construct this directly.
    ///
    /// [`KernelBuilder::build`]: super::KernelBuilder::build
    pub(crate) const fn new(
        regs: &'a mut RegisterAllocator,
        instructions: &'a mut Vec<Instruction>,
        param_names: &'a [String],
        target: SmVersion,
    ) -> Self {
        Self {
            regs,
            instructions,
            label_counter: 0,
            param_names,
            target,
        }
    }

    // ════════════════════════════════════════════════════════════════════
    //  Parameter Loading
    // ════════════════════════════════════════════════════════════════════

    /// Loads a `u32` kernel parameter by name.
    ///
    /// Emits a `ld.param.u32` instruction and returns the destination register.
    pub fn load_param_u32(&mut self, name: &str) -> Register {
        self.load_param(name, PtxType::U32)
    }

    /// Loads a `u64` kernel parameter by name (typically a device pointer).
    ///
    /// Emits a `ld.param.u64` instruction and returns the destination register.
    pub fn load_param_u64(&mut self, name: &str) -> Register {
        self.load_param(name, PtxType::U64)
    }

    /// Loads an `f32` kernel parameter by name.
    ///
    /// Emits a `ld.param.f32` instruction and returns the destination register.
    pub fn load_param_f32(&mut self, name: &str) -> Register {
        self.load_param(name, PtxType::F32)
    }

    /// Loads an `f64` kernel parameter by name.
    ///
    /// Emits a `ld.param.f64` instruction and returns the destination register.
    pub fn load_param_f64(&mut self, name: &str) -> Register {
        self.load_param(name, PtxType::F64)
    }

    /// Generic parameter load helper.
    ///
    /// Emits `ld.param{.ty} dst, [%param_{name}]` using the `LoadParam`
    /// instruction variant.
    fn load_param(&mut self, name: &str, ty: PtxType) -> Register {
        let dst = self.regs.alloc(ty);
        self.emit(Instruction::LoadParam {
            ty,
            dst: dst.clone(),
            param_name: format!("%param_{name}"),
        });
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Thread / Block ID Computation
    // ════════════════════════════════════════════════════════════════════

    /// Computes the global thread ID in the X dimension.
    ///
    /// Equivalent to `blockIdx.x * blockDim.x + threadIdx.x` in CUDA C.
    /// Emits:
    /// ```ptx
    /// mov.u32 %r_tid,  %tid.x;
    /// mov.u32 %r_ntid, %ntid.x;
    /// mov.u32 %r_ctaid, %ctaid.x;
    /// mad.lo.u32 %r_gid, %r_ctaid, %r_ntid, %r_tid;
    /// ```
    pub fn global_thread_id_x(&mut self) -> Register {
        let tid = self.read_special_reg(SpecialReg::TidX);
        let ntid = self.read_special_reg(SpecialReg::NtidX);
        let ctaid = self.read_special_reg(SpecialReg::CtaidX);
        let gid = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::Mad {
            ty: PtxType::U32,
            mode: MulMode::Lo,
            dst: gid.clone(),
            a: Operand::Register(ctaid),
            b: Operand::Register(ntid),
            c: Operand::Register(tid),
        });
        gid
    }

    /// Computes the global thread ID in the Y dimension.
    ///
    /// Equivalent to `blockIdx.y * blockDim.y + threadIdx.y` in CUDA C.
    pub fn global_thread_id_y(&mut self) -> Register {
        let tid = self.read_special_reg(SpecialReg::TidY);
        let ntid = self.read_special_reg(SpecialReg::NtidY);
        let ctaid = self.read_special_reg(SpecialReg::CtaidY);
        let gid = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::Mad {
            ty: PtxType::U32,
            mode: MulMode::Lo,
            dst: gid.clone(),
            a: Operand::Register(ctaid),
            b: Operand::Register(ntid),
            c: Operand::Register(tid),
        });
        gid
    }

    /// Computes both X and Y global thread IDs for 2D kernels.
    ///
    /// Returns `(row, col)` where `row` is the Y global ID and `col` is
    /// the X global ID (following matrix convention).
    pub fn global_thread_id_2d(&mut self) -> (Register, Register) {
        let col = self.global_thread_id_x();
        let row = self.global_thread_id_y();
        (row, col)
    }

    /// Reads `%tid.x` (thread index within the block, X dimension).
    pub fn thread_id_x(&mut self) -> Register {
        self.read_special_reg(SpecialReg::TidX)
    }

    /// Reads `%ctaid.x` (block index within the grid, X dimension).
    pub fn block_id_x(&mut self) -> Register {
        self.read_special_reg(SpecialReg::CtaidX)
    }

    /// Reads `%ntid.x` (number of threads per block, X dimension).
    pub fn block_dim_x(&mut self) -> Register {
        self.read_special_reg(SpecialReg::NtidX)
    }

    /// Reads a special register into a fresh `U32` register using `MovSpecial`.
    fn read_special_reg(&mut self, sreg: SpecialReg) -> Register {
        let dst = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::MovSpecial {
            dst: dst.clone(),
            special: sreg,
        });
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Integer Arithmetic
    // ════════════════════════════════════════════════════════════════════

    /// Emits `add.u32 dst, a, b`.
    pub fn add_u32(&mut self, a: Register, b: Register) -> Register {
        self.add_typed(PtxType::U32, a, b)
    }

    /// Emits `add.u64 dst, a, b`.
    pub fn add_u64(&mut self, a: Register, b: Register) -> Register {
        self.add_typed(PtxType::U64, a, b)
    }

    /// Emits `add.f32 dst, a, b`.
    pub fn add_f32(&mut self, a: Register, b: Register) -> Register {
        self.add_typed(PtxType::F32, a, b)
    }

    /// Emits `add.f64 dst, a, b`.
    pub fn add_f64(&mut self, a: Register, b: Register) -> Register {
        self.add_typed(PtxType::F64, a, b)
    }

    /// Generic typed addition helper.
    fn add_typed(&mut self, ty: PtxType, a: Register, b: Register) -> Register {
        let dst = self.regs.alloc(ty);
        self.emit(Instruction::Add {
            ty,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
        });
        dst
    }

    /// Emits `sub.f32 dst, a, b`.
    pub fn sub_f32(&mut self, a: Register, b: Register) -> Register {
        self.sub_typed(PtxType::F32, a, b)
    }

    /// Emits `sub.f64 dst, a, b`.
    pub fn sub_f64(&mut self, a: Register, b: Register) -> Register {
        self.sub_typed(PtxType::F64, a, b)
    }

    /// Generic typed subtraction helper.
    fn sub_typed(&mut self, ty: PtxType, a: Register, b: Register) -> Register {
        let dst = self.regs.alloc(ty);
        self.emit(Instruction::Sub {
            ty,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
        });
        dst
    }

    /// Emits `mul.lo.u32 dst, a, b` — low 32 bits of a u32 multiplication.
    pub fn mul_lo_u32(&mut self, a: Register, b: Register) -> Register {
        let dst = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::Mul {
            ty: PtxType::U32,
            mode: MulMode::Lo,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
        });
        dst
    }

    /// Emits `mul.wide.u32 dst, a, b` — widens two u32 operands to produce
    /// a u64 result.
    pub fn mul_wide_u32_to_u64(&mut self, a: Register, b: Register) -> Register {
        let dst = self.regs.alloc(PtxType::U64);
        self.emit(Instruction::Mul {
            ty: PtxType::U32,
            mode: MulMode::Wide,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
        });
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Integer Multiply-Add (mad.lo / mad.hi / mad.wide)
    // ════════════════════════════════════════════════════════════════════

    /// Emits `mad.lo.s32 dst, a, b, c` — low 32 bits of `a*b+c` (signed).
    pub fn mad_lo_s32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.mad_lo_typed(PtxType::S32, a, b, c)
    }

    /// Emits `mad.lo.u32 dst, a, b, c` — low 32 bits of `a*b+c` (unsigned).
    pub fn mad_lo_u32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.mad_lo_typed(PtxType::U32, a, b, c)
    }

    /// Emits `mad.lo.s64 dst, a, b, c` — low 64 bits of `a*b+c` (signed).
    pub fn mad_lo_s64(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.mad_lo_typed(PtxType::S64, a, b, c)
    }

    /// Emits `mad.lo.u64 dst, a, b, c` — low 64 bits of `a*b+c` (unsigned).
    pub fn mad_lo_u64(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.mad_lo_typed(PtxType::U64, a, b, c)
    }

    /// Generic typed `mad.lo` helper.
    fn mad_lo_typed(&mut self, typ: PtxType, a: Register, b: Register, c: Register) -> Register {
        let dst = self.regs.alloc(typ);
        self.emit(Instruction::MadLo {
            typ,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
            c: Operand::Register(c),
        });
        dst
    }

    /// Emits `mad.hi.s32 dst, a, b, c` — high 32 bits of `a*b+c` (signed).
    pub fn mad_hi_s32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.mad_hi_typed(PtxType::S32, a, b, c)
    }

    /// Emits `mad.hi.u32 dst, a, b, c` — high 32 bits of `a*b+c` (unsigned).
    pub fn mad_hi_u32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.mad_hi_typed(PtxType::U32, a, b, c)
    }

    /// Emits `mad.hi.s64 dst, a, b, c` — high 64 bits of `a*b+c` (signed).
    pub fn mad_hi_s64(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.mad_hi_typed(PtxType::S64, a, b, c)
    }

    /// Emits `mad.hi.u64 dst, a, b, c` — high 64 bits of `a*b+c` (unsigned).
    pub fn mad_hi_u64(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.mad_hi_typed(PtxType::U64, a, b, c)
    }

    /// Generic typed `mad.hi` helper.
    fn mad_hi_typed(&mut self, typ: PtxType, a: Register, b: Register, c: Register) -> Register {
        let dst = self.regs.alloc(typ);
        self.emit(Instruction::MadHi {
            typ,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
            c: Operand::Register(c),
        });
        dst
    }

    /// Emits `mad.wide.s16 dst, a, b, c` — widening multiply-add, s16 -> s32.
    pub fn mad_wide_s16(&mut self, a: Register, b: Register, c: Register) -> Register {
        let dst = self.regs.alloc(PtxType::S32);
        self.emit(Instruction::MadWide {
            src_typ: PtxType::S16,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
            c: Operand::Register(c),
        });
        dst
    }

    /// Emits `mad.wide.u16 dst, a, b, c` — widening multiply-add, u16 -> u32.
    pub fn mad_wide_u16(&mut self, a: Register, b: Register, c: Register) -> Register {
        let dst = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::MadWide {
            src_typ: PtxType::U16,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
            c: Operand::Register(c),
        });
        dst
    }

    /// Emits `mad.wide.s32 dst, a, b, c` — widening multiply-add, s32 -> s64.
    pub fn mad_wide_s32(&mut self, a: Register, b: Register, c: Register) -> Register {
        let dst = self.regs.alloc(PtxType::S64);
        self.emit(Instruction::MadWide {
            src_typ: PtxType::S32,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
            c: Operand::Register(c),
        });
        dst
    }

    /// Emits `mad.wide.u32 dst, a, b, c` — widening multiply-add, u32 -> u64.
    pub fn mad_wide_u32(&mut self, a: Register, b: Register, c: Register) -> Register {
        let dst = self.regs.alloc(PtxType::U64);
        self.emit(Instruction::MadWide {
            src_typ: PtxType::U32,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
            c: Operand::Register(c),
        });
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Floating-Point Arithmetic
    // ════════════════════════════════════════════════════════════════════

    /// Emits `fma.rn.f32 dst, a, b, c` — fused multiply-add, single precision.
    pub fn fma_f32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.fma_typed(PtxType::F32, a, b, c)
    }

    /// Emits `fma.rn.f64 dst, a, b, c` — fused multiply-add, double precision.
    pub fn fma_f64(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.fma_typed(PtxType::F64, a, b, c)
    }

    /// Generic typed FMA helper with round-to-nearest-even.
    fn fma_typed(&mut self, ty: PtxType, a: Register, b: Register, c: Register) -> Register {
        let dst = self.regs.alloc(ty);
        self.emit(Instruction::Fma {
            rnd: RoundingMode::Rn,
            ty,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
            c: Operand::Register(c),
        });
        dst
    }

    /// Emits `neg.f32 dst, src`.
    pub fn neg_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Neg {
            ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `abs.f32 dst, src`.
    pub fn abs_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Abs {
            ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `min.f32 dst, a, b`.
    pub fn min_f32(&mut self, a: Register, b: Register) -> Register {
        self.min_typed(PtxType::F32, a, b)
    }

    /// Emits `max.f32 dst, a, b`.
    pub fn max_f32(&mut self, a: Register, b: Register) -> Register {
        self.max_typed(PtxType::F32, a, b)
    }

    /// Emits `min.u32 dst, a, b`.
    pub fn min_u32(&mut self, a: Register, b: Register) -> Register {
        self.min_typed(PtxType::U32, a, b)
    }

    /// Emits `max.u32 dst, a, b`.
    pub fn max_u32(&mut self, a: Register, b: Register) -> Register {
        self.max_typed(PtxType::U32, a, b)
    }

    /// Generic typed `min` helper.
    fn min_typed(&mut self, ty: PtxType, a: Register, b: Register) -> Register {
        let dst = self.regs.alloc(ty);
        self.emit(Instruction::Min {
            ty,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
        });
        dst
    }

    /// Generic typed `max` helper.
    fn max_typed(&mut self, ty: PtxType, a: Register, b: Register) -> Register {
        let dst = self.regs.alloc(ty);
        self.emit(Instruction::Max {
            ty,
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
        });
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Bit Manipulation
    // ════════════════════════════════════════════════════════════════════

    /// Emits `brev.b32 dst, src` — reverse the bits of a 32-bit value.
    pub fn brev_b32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::B32);
        self.emit(Instruction::Brev {
            ty: PtxType::B32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `brev.b64 dst, src` — reverse the bits of a 64-bit value.
    pub fn brev_b64(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::B64);
        self.emit(Instruction::Brev {
            ty: PtxType::B64,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `clz.b32 dst, src` — count leading zeros (result is U32).
    pub fn clz_b32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::Clz {
            ty: PtxType::B32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `popc.b32 dst, src` — population count of 32-bit value (result is U32).
    pub fn popc_b32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::Popc {
            ty: PtxType::B32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `popc.b64 dst, src` — population count of 64-bit value (result is U32).
    pub fn popc_b64(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::Popc {
            ty: PtxType::B64,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `bfind.u32 dst, src` — find most significant bit (unsigned, result is U32).
    pub fn bfind_u32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::Bfind {
            ty: PtxType::U32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `bfind.s32 dst, src` — find most significant non-sign bit (signed, result is U32).
    pub fn bfind_s32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::Bfind {
            ty: PtxType::S32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `bfe.u32 dst, src, start, len` — extract a bit field (unsigned).
    pub fn bfe_u32(&mut self, src: Register, start: Register, len: Register) -> Register {
        let dst = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::Bfe {
            ty: PtxType::U32,
            dst: dst.clone(),
            src: Operand::Register(src),
            start: Operand::Register(start),
            len: Operand::Register(len),
        });
        dst
    }

    /// Emits `bfe.s32 dst, src, start, len` — extract a bit field (signed).
    pub fn bfe_s32(&mut self, src: Register, start: Register, len: Register) -> Register {
        let dst = self.regs.alloc(PtxType::S32);
        self.emit(Instruction::Bfe {
            ty: PtxType::S32,
            dst: dst.clone(),
            src: Operand::Register(src),
            start: Operand::Register(start),
            len: Operand::Register(len),
        });
        dst
    }

    /// Emits `bfi.b32 dst, insert, base, start, len` — insert a bit field.
    pub fn bfi_b32(
        &mut self,
        insert: Register,
        base: Register,
        start: Register,
        len: Register,
    ) -> Register {
        let dst = self.regs.alloc(PtxType::B32);
        self.emit(Instruction::Bfi {
            ty: PtxType::B32,
            dst: dst.clone(),
            insert: Operand::Register(insert),
            base: Operand::Register(base),
            start: Operand::Register(start),
            len: Operand::Register(len),
        });
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Shift Operations
    // ════════════════════════════════════════════════════════════════════

    /// Emits `shl.b32 dst, src, amount` — left shift, 32-bit.
    pub fn shl_b32(&mut self, src: Register, amount: Register) -> Register {
        let dst = self.regs.alloc(PtxType::B32);
        self.emit(Instruction::Shl {
            ty: PtxType::B32,
            dst: dst.clone(),
            src: Operand::Register(src),
            amount: Operand::Register(amount),
        });
        dst
    }

    /// Emits `shl.b64 dst, src, amount` — left shift, 64-bit.
    pub fn shl_b64(&mut self, src: Register, amount: Register) -> Register {
        let dst = self.regs.alloc(PtxType::B64);
        self.emit(Instruction::Shl {
            ty: PtxType::B64,
            dst: dst.clone(),
            src: Operand::Register(src),
            amount: Operand::Register(amount),
        });
        dst
    }

    /// Emits `shr.b32 dst, src, amount` — logical right shift, 32-bit.
    pub fn shr_b32(&mut self, src: Register, amount: Register) -> Register {
        let dst = self.regs.alloc(PtxType::B32);
        self.emit(Instruction::Shr {
            ty: PtxType::B32,
            dst: dst.clone(),
            src: Operand::Register(src),
            amount: Operand::Register(amount),
        });
        dst
    }

    /// Emits `shr.b64 dst, src, amount` — logical right shift, 64-bit.
    pub fn shr_b64(&mut self, src: Register, amount: Register) -> Register {
        let dst = self.regs.alloc(PtxType::B64);
        self.emit(Instruction::Shr {
            ty: PtxType::B64,
            dst: dst.clone(),
            src: Operand::Register(src),
            amount: Operand::Register(amount),
        });
        dst
    }

    /// Emits `shr.u32 dst, src, amount` — logical right shift for unsigned 32-bit.
    pub fn shr_u32(&mut self, src: Register, amount: Register) -> Register {
        let dst = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::Shr {
            ty: PtxType::U32,
            dst: dst.clone(),
            src: Operand::Register(src),
            amount: Operand::Register(amount),
        });
        dst
    }

    /// Emits `shr.s32 dst, src, amount` — arithmetic right shift for signed 32-bit.
    pub fn shr_s32(&mut self, src: Register, amount: Register) -> Register {
        let dst = self.regs.alloc(PtxType::S32);
        self.emit(Instruction::Shr {
            ty: PtxType::S32,
            dst: dst.clone(),
            src: Operand::Register(src),
            amount: Operand::Register(amount),
        });
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Special Math Functions
    // ════════════════════════════════════════════════════════════════════

    /// Emits `rcp.rn.f32 dst, src` — reciprocal, single precision.
    pub fn rcp_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Rcp {
            rnd: Some(RoundingMode::Rn),
            ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `rcp.rn.f64 dst, src` — reciprocal, double precision.
    pub fn rcp_f64(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F64);
        self.emit(Instruction::Rcp {
            rnd: Some(RoundingMode::Rn),
            ty: PtxType::F64,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `rcp.approx.ftz.f32 dst, src` — fast approximate reciprocal.
    ///
    /// Uses `rnd=None` to signal approx mode (no IEEE rounding).
    pub fn rcp_approx_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Rcp {
            rnd: None,
            ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `rsqrt.approx.f32 dst, src` — approximate reciprocal square root.
    pub fn rsqrt_approx_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Rsqrt {
            approx: true,
            ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `rsqrt.approx.f64 dst, src` — approximate reciprocal square root, double precision.
    pub fn rsqrt_approx_f64(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F64);
        self.emit(Instruction::Rsqrt {
            approx: true,
            ty: PtxType::F64,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `sqrt.rn.f32 dst, src` — square root, single precision.
    pub fn sqrt_rn_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Sqrt {
            rnd: Some(RoundingMode::Rn),
            ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `sqrt.rn.f64 dst, src` — square root, double precision.
    pub fn sqrt_rn_f64(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F64);
        self.emit(Instruction::Sqrt {
            rnd: Some(RoundingMode::Rn),
            ty: PtxType::F64,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `ex2.approx.f32 dst, src` — base-2 exponential, approximate.
    pub fn ex2_approx_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Ex2 {
            approx: true,
            ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `lg2.approx.f32 dst, src` — base-2 logarithm, approximate.
    pub fn lg2_approx_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Lg2 {
            approx: true,
            ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `sin.approx.f32 dst, src` — sine, approximate.
    pub fn sin_approx_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Sin {
            approx: true,
            ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Emits `cos.approx.f32 dst, src` — cosine, approximate.
    pub fn cos_approx_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Cos {
            approx: true,
            ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Memory Operations — Global
    // ════════════════════════════════════════════════════════════════════

    /// Loads a single `f32` from global memory.
    ///
    /// `addr` should be a `U64` register containing the global device pointer.
    /// Emits `ld.global.f32 dst, [addr]`.
    pub fn load_global_f32(&mut self, addr: Register) -> Register {
        self.load_global_scalar(PtxType::F32, addr)
    }

    /// Loads a single `f64` from global memory.
    pub fn load_global_f64(&mut self, addr: Register) -> Register {
        self.load_global_scalar(PtxType::F64, addr)
    }

    /// Loads a single signed 32-bit integer from global memory.
    ///
    /// Emits `ld.global.s32 dst, [addr]`.
    pub fn load_global_i32(&mut self, addr: Register) -> Register {
        self.load_global_scalar(PtxType::S32, addr)
    }

    /// Loads a single unsigned 32-bit integer from global memory.
    ///
    /// Emits `ld.global.u32 dst, [addr]`.
    pub fn load_global_u32(&mut self, addr: Register) -> Register {
        self.load_global_scalar(PtxType::U32, addr)
    }

    /// Loads a single scalar value from global memory.
    fn load_global_scalar(&mut self, ty: PtxType, addr: Register) -> Register {
        let dst = self.regs.alloc(ty);
        self.emit(Instruction::Load {
            space: MemorySpace::Global,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty,
            dst: dst.clone(),
            addr: Operand::Address {
                base: addr,
                offset: None,
            },
        });
        dst
    }

    /// Loads four `f32` values from global memory as a vectorized `.v4` load.
    ///
    /// Returns an array of 4 registers containing the loaded values.
    /// `addr` must be 16-byte aligned for correctness.
    ///
    /// Since the IR `Load` instruction uses a single destination register,
    /// this method emits raw PTX for the vectorized load and individual
    /// `mov` instructions to extract each element.
    pub fn load_global_f32x4(&mut self, addr: &Register) -> [Register; 4] {
        let r0 = self.regs.alloc(PtxType::F32);
        let r1 = self.regs.alloc(PtxType::F32);
        let r2 = self.regs.alloc(PtxType::F32);
        let r3 = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Raw(format!(
            "ld.global.v4.f32 {{{r0}, {r1}, {r2}, {r3}}}, [{addr}];"
        )));
        [r0, r1, r2, r3]
    }

    /// Stores a single `f32` to global memory.
    ///
    /// `addr` should be a `U64` register containing the global device pointer.
    pub fn store_global_f32(&mut self, addr: Register, val: Register) {
        self.store_global_scalar(PtxType::F32, addr, val);
    }

    /// Stores a single `f64` to global memory.
    pub fn store_global_f64(&mut self, addr: Register, val: Register) {
        self.store_global_scalar(PtxType::F64, addr, val);
    }

    /// Stores a single signed 32-bit integer to global memory.
    ///
    /// Emits `st.global.s32 [addr], val`.
    pub fn store_global_i32(&mut self, addr: Register, val: Register) {
        self.store_global_scalar(PtxType::S32, addr, val);
    }

    /// Stores a single unsigned 32-bit integer to global memory.
    ///
    /// Emits `st.global.u32 [addr], val`.
    pub fn store_global_u32(&mut self, addr: Register, val: Register) {
        self.store_global_scalar(PtxType::U32, addr, val);
    }

    /// Stores a single scalar to global memory.
    fn store_global_scalar(&mut self, ty: PtxType, addr: Register, val: Register) {
        self.emit(Instruction::Store {
            space: MemorySpace::Global,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty,
            addr: Operand::Address {
                base: addr,
                offset: None,
            },
            src: val,
        });
    }

    // ════════════════════════════════════════════════════════════════════
    //  Memory Operations — Shared
    // ════════════════════════════════════════════════════════════════════

    /// Loads a single `f32` from shared memory.
    ///
    /// `addr` should be a register containing an address in shared memory space.
    pub fn load_shared_f32(&mut self, addr: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Load {
            space: MemorySpace::Shared,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            dst: dst.clone(),
            addr: Operand::Address {
                base: addr,
                offset: None,
            },
        });
        dst
    }

    /// Stores a single `f32` to shared memory.
    pub fn store_shared_f32(&mut self, addr: Register, val: Register) {
        self.emit(Instruction::Store {
            space: MemorySpace::Shared,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            addr: Operand::Address {
                base: addr,
                offset: None,
            },
            src: val,
        });
    }

    // ════════════════════════════════════════════════════════════════════
    //  Asynchronous Copy (cp.async, Ampere+)
    // ════════════════════════════════════════════════════════════════════

    /// Emits a 32-bit (4-byte) asynchronous copy from global to shared memory.
    ///
    /// Emits: `cp.async.ca.shared.global [dst], [src], 4;`
    /// Requires `sm_80`+.
    pub fn cp_async_32bit(&mut self, dst_shared: Register, src_global: Register) {
        self.emit(Instruction::CpAsync {
            bytes: 4,
            dst_shared: Operand::Register(dst_shared),
            src_global: Operand::Register(src_global),
        });
    }

    /// Emits a 64-bit (8-byte) asynchronous copy from global to shared memory.
    ///
    /// Emits: `cp.async.ca.shared.global [dst], [src], 8;`
    /// Requires `sm_80`+.
    pub fn cp_async_64bit(&mut self, dst_shared: Register, src_global: Register) {
        self.emit(Instruction::CpAsync {
            bytes: 8,
            dst_shared: Operand::Register(dst_shared),
            src_global: Operand::Register(src_global),
        });
    }

    /// Emits a 128-bit (16-byte) asynchronous copy from global to shared memory.
    ///
    /// This is the most common `cp.async` variant, used for double-buffered
    /// data loading in high-performance kernels. Requires `sm_80`+.
    pub fn cp_async_128bit(&mut self, dst_shared: Register, src_global: Register) {
        self.emit(Instruction::CpAsync {
            bytes: 16,
            dst_shared: Operand::Register(dst_shared),
            src_global: Operand::Register(src_global),
        });
    }

    /// Emits `cp.async.commit_group` to commit all pending async copies.
    pub fn cp_async_commit(&mut self) {
        self.emit(Instruction::CpAsyncCommit);
    }

    /// Emits `cp.async.wait_group N` to wait until at most `n` copy groups
    /// are still pending.
    ///
    /// Pass `0` to wait for all pending copies to complete.
    pub fn cp_async_wait(&mut self, n: u32) {
        self.emit(Instruction::CpAsyncWait { n });
    }

    /// Emits a `ldmatrix.sync.aligned.m8n8.x4.shared.b16` instruction (SM >= 75).
    ///
    /// Loads 4 warp-cooperative 8×8 B16 matrix fragments from shared memory.
    /// Each of the 32 threads contributes to loading 8 bytes (one row) of
    /// the tile. Returns the four destination registers.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if the target architecture does not support
    /// `ldmatrix` (requires SM >= 75).
    pub fn ldmatrix_x4(&mut self, src_addr: Register) -> Result<[Register; 4], PtxGenError> {
        use crate::ir::Instruction as I;
        if !self.target.capabilities().has_ldmatrix {
            return Err(PtxGenError::UnsupportedFeature {
                arch: self.target.as_ptx_str().to_string(),
                feature: "ldmatrix (SM >= 75)".to_string(),
            });
        }
        let r0 = self.regs.alloc(PtxType::B32);
        let r1 = self.regs.alloc(PtxType::B32);
        let r2 = self.regs.alloc(PtxType::B32);
        let r3 = self.regs.alloc(PtxType::B32);
        self.emit(I::Ldmatrix {
            num_fragments: 4,
            trans: false,
            dst_regs: vec![r0.clone(), r1.clone(), r2.clone(), r3.clone()],
            src_addr: Operand::Register(src_addr),
        });
        Ok([r0, r1, r2, r3])
    }

    // ════════════════════════════════════════════════════════════════════
    //  Control Flow
    // ════════════════════════════════════════════════════════════════════

    /// Emits a conditional block that executes `body` when `a < b` (unsigned 32-bit).
    ///
    /// Generates a `setp.lo.u32` comparison, a negated conditional branch
    /// over the body, and a skip label.
    ///
    /// # Example
    ///
    /// ```ignore
    /// b.if_lt_u32(tid, n, |b| {
    ///     // Only threads with tid < n execute this
    /// });
    /// ```
    pub fn if_lt_u32<F>(&mut self, a: Register, b: Register, body: F)
    where
        F: FnOnce(&mut BodyBuilder<'_>),
    {
        let pred = self.regs.alloc(PtxType::Pred);
        self.emit(Instruction::SetP {
            cmp: CmpOp::Lo,
            ty: PtxType::U32,
            dst: pred.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
        });
        let skip_label = self.fresh_label("skip");
        // Branch to skip when predicate is false (negate = true).
        self.emit(Instruction::Branch {
            target: skip_label.clone(),
            predicate: Some((pred, true)),
        });
        body(self);
        self.emit(Instruction::Label(skip_label));
    }

    /// Emits a conditional block that executes `body` when `a >= b` (unsigned 32-bit).
    pub fn if_ge_u32<F>(&mut self, a: Register, b: Register, body: F)
    where
        F: FnOnce(&mut BodyBuilder<'_>),
    {
        let pred = self.regs.alloc(PtxType::Pred);
        self.emit(Instruction::SetP {
            cmp: CmpOp::Hs,
            ty: PtxType::U32,
            dst: pred.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
        });
        let skip_label = self.fresh_label("skip");
        self.emit(Instruction::Branch {
            target: skip_label.clone(),
            predicate: Some((pred, true)),
        });
        body(self);
        self.emit(Instruction::Label(skip_label));
    }

    /// Compile-time loop unrolling.
    ///
    /// Calls `body(i)` for `i` in `0..count`, emitting all iterations
    /// inline. This is equivalent to `#pragma unroll` in CUDA C.
    ///
    /// Each iteration gets its own comment indicating the unroll index.
    pub fn unroll<F>(&mut self, count: u32, mut body: F)
    where
        F: FnMut(&mut BodyBuilder<'_>, u32),
    {
        for i in 0..count {
            self.comment(&format!("unroll iteration {i}/{count}"));
            body(self, i);
        }
    }

    /// Emits a `.pragma "unroll N"` or `.pragma "nounroll"` directive hint.
    ///
    /// When `factor` is `Some(n)`, emits `.pragma "unroll N";` to hint the
    /// PTX assembler to unroll the following loop by factor `n`.
    /// When `factor` is `None`, emits `.pragma "nounroll";` to suppress
    /// unrolling.
    pub fn pragma_unroll(&mut self, factor: Option<u32>) {
        let text = factor.map_or_else(|| "nounroll".to_string(), |n| format!("unroll {n}"));
        self.emit(Instruction::Pragma(text));
    }

    /// Emits a label pseudo-instruction.
    ///
    /// Labels are branch targets. They appear at the start of a line
    /// without indentation in the generated PTX.
    pub fn label(&mut self, name: &str) {
        self.emit(Instruction::Label(name.to_string()));
    }

    /// Emits an unconditional branch to the given label.
    pub fn branch(&mut self, target: &str) {
        self.emit(Instruction::Branch {
            target: target.to_string(),
            predicate: None,
        });
    }

    /// Emits a conditional branch: `@pred bra target`.
    pub fn branch_if(&mut self, pred: Register, target: &str) {
        self.emit(Instruction::Branch {
            target: target.to_string(),
            predicate: Some((pred, false)),
        });
    }

    /// Emits a `ret` instruction to return from the kernel.
    pub fn ret(&mut self) {
        self.emit(Instruction::Return);
    }

    // ════════════════════════════════════════════════════════════════════
    //  Synchronization
    // ════════════════════════════════════════════════════════════════════

    /// Emits `bar.sync id` — block-level barrier synchronization.
    ///
    /// All threads in the block must reach this barrier before any can proceed.
    /// `id` is typically 0.
    pub fn bar_sync(&mut self, id: u32) {
        self.emit(Instruction::BarSync { id });
    }

    /// Emits a memory fence with acquire-release semantics at the given scope.
    ///
    /// - [`FenceScope::Cta`]: visibility within the block
    /// - [`FenceScope::Gpu`]: visibility across the entire GPU
    /// - [`FenceScope::Sys`]: visibility across GPU and host
    pub fn fence_acq_rel(&mut self, scope: FenceScope) {
        self.emit(Instruction::FenceAcqRel { scope });
    }

    // ════════════════════════════════════════════════════════════════════
    //  Type Conversion
    // ════════════════════════════════════════════════════════════════════

    /// Converts a `u32` register to `u64` (zero-extension).
    ///
    /// Emits `cvt.u64.u32 dst, src`.
    pub fn cvt_u32_to_u64(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::U64);
        self.emit(Instruction::Cvt {
            rnd: None,
            dst_ty: PtxType::U64,
            src_ty: PtxType::U32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Converts an `f32` register to `f64` (widening).
    ///
    /// Emits `cvt.f64.f32 dst, src`.
    pub fn cvt_f32_to_f64(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F64);
        self.emit(Instruction::Cvt {
            rnd: None,
            dst_ty: PtxType::F64,
            src_ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Converts an `f64` register to `f32` (narrowing, round-to-nearest-even).
    ///
    /// Emits `cvt.rn.f32.f64 dst, src`.
    pub fn cvt_f64_to_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Cvt {
            rnd: Some(RoundingMode::Rn),
            dst_ty: PtxType::F32,
            src_ty: PtxType::F64,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Converts an `f16` register to `f32` (widening).
    ///
    /// Emits `cvt.f32.f16 dst, src`.
    pub fn cvt_f16_to_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Cvt {
            rnd: None,
            dst_ty: PtxType::F32,
            src_ty: PtxType::F16,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Converts an `f32` register to `f16` (narrowing, round-to-nearest-even).
    ///
    /// Emits `cvt.rn.f16.f32 dst, src`.
    pub fn cvt_f32_to_f16(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F16);
        self.emit(Instruction::Cvt {
            rnd: Some(RoundingMode::Rn),
            dst_ty: PtxType::F16,
            src_ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Converts a `bf16` register to `f32` (widening).
    ///
    /// Emits `cvt.f32.bf16 dst, src`.
    pub fn cvt_bf16_to_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Cvt {
            rnd: None,
            dst_ty: PtxType::F32,
            src_ty: PtxType::BF16,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Converts an `f32` register to `bf16` (narrowing, round-to-nearest-even).
    ///
    /// Emits `cvt.rn.bf16.f32 dst, src`.
    pub fn cvt_f32_to_bf16(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::BF16);
        self.emit(Instruction::Cvt {
            rnd: Some(RoundingMode::Rn),
            dst_ty: PtxType::BF16,
            src_ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Converts an `f32` register to FP8 `E4M3` format (`sm_89+`, Ada/Hopper).
    ///
    /// Emits: `cvt.rn.satfinite.e4m3x2.f32 dst, src_hi, src_lo`
    /// Note: PTX packs two FP8 values per register (`e4m3x2`).
    pub fn cvt_f32_to_e4m3(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::E4M3);
        self.emit(Instruction::Cvt {
            rnd: Some(RoundingMode::Rn),
            dst_ty: PtxType::E4M3,
            src_ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Converts an FP8 `E4M3` register to `f32` (`sm_89+`).
    ///
    /// Emits `cvt.f32.e4m3 dst, src`.
    pub fn cvt_e4m3_to_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Cvt {
            rnd: None,
            dst_ty: PtxType::F32,
            src_ty: PtxType::E4M3,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Converts an `f32` register to FP8 `E5M2` format (`sm_89+`).
    ///
    /// Emits `cvt.rn.e5m2.f32 dst, src`.
    pub fn cvt_f32_to_e5m2(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::E5M2);
        self.emit(Instruction::Cvt {
            rnd: Some(RoundingMode::Rn),
            dst_ty: PtxType::E5M2,
            src_ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    /// Converts an FP8 `E5M2` register to `f32` (`sm_89+`).
    ///
    /// Emits `cvt.f32.e5m2 dst, src`.
    pub fn cvt_e5m2_to_f32(&mut self, src: Register) -> Register {
        let dst = self.regs.alloc(PtxType::F32);
        self.emit(Instruction::Cvt {
            rnd: None,
            dst_ty: PtxType::F32,
            src_ty: PtxType::E5M2,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Tensor Core (Ampere+ MMA)
    // ════════════════════════════════════════════════════════════════════

    /// Emits an `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` instruction.
    ///
    /// This is the standard Ampere tensor core MMA operation:
    /// - **Shape**: 16x8x16 tile
    /// - **A fragment**: registers holding f16 matrix A data
    /// - **B fragment**: registers holding f16 matrix B data
    /// - **C/D accumulator**: 4 f32 registers for input/output accumulator
    ///
    /// Returns the 4 destination accumulator registers.
    ///
    /// # Arguments
    ///
    /// * `a_regs` — Registers holding the A matrix fragment (f16)
    /// * `b_regs` — Registers holding the B matrix fragment (f16)
    /// * `c_regs` — Registers holding the C accumulator input (f32)
    pub fn mma_m16n8k16_f16_f32(
        &mut self,
        a_regs: &[Register],
        b_regs: &[Register],
        c_regs: &[Register],
    ) -> [Register; 4] {
        let dst = self.regs.alloc_group(PtxType::F32, 4);
        self.emit(Instruction::Mma {
            shape: MmaShape::M16N8K16,
            a_ty: PtxType::F16,
            b_ty: PtxType::F16,
            c_ty: PtxType::F32,
            d_ty: PtxType::F32,
            d_regs: dst.clone(),
            a_regs: a_regs.to_vec(),
            b_regs: b_regs.to_vec(),
            c_regs: c_regs.to_vec(),
        });
        [
            dst[0].clone(),
            dst[1].clone(),
            dst[2].clone(),
            dst[3].clone(),
        ]
    }

    /// Emits `wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16`
    /// Warpgroup MMA async for Hopper (`sm_90+`) computing a 64×128 tile.
    ///
    /// This operates on warpgroup-level fragments:
    /// - `a_desc`: A operand descriptor (shared memory descriptor string)
    /// - `b_desc`: B operand descriptor
    /// - Accumulator: 64 f32 registers (managed by the warpgroup implicitly)
    ///
    /// Emits raw PTX via `raw_ptx` since wgmma is not yet in the structured IR.
    ///
    /// # Errors
    ///
    /// Returns `PtxGenError` when the target SM is below 90 (Hopper).
    pub fn wgmma_mma_async_m64n128k16_f16(
        &mut self,
        a_desc: &str,
        b_desc: &str,
    ) -> Result<(), PtxGenError> {
        if !self.target.capabilities().has_wgmma {
            return Err(PtxGenError::GenerationFailed(format!(
                "wgmma.mma_async requires SM >= 90 (Hopper), target is {}",
                self.target
            )));
        }
        self.raw_ptx(&format!(
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {{...}}, {a_desc}, {b_desc}, 1, 1, 1, 0, 0;"
        ));
        Ok(())
    }

    // ════════════════════════════════════════════════════════════════════
    //  Video Instructions (dp4a / dp2a)
    // ════════════════════════════════════════════════════════════════════

    /// Emits `dp4a.u32.u32 dst, a, b, c` — unsigned 4-way byte dot product.
    pub fn dp4a_u32_u32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.dp4a_typed(a, b, c, false, false)
    }

    /// Emits `dp4a.s32.s32 dst, a, b, c` — signed 4-way byte dot product.
    pub fn dp4a_s32_s32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.dp4a_typed(a, b, c, true, true)
    }

    /// Emits `dp4a.s32.u32 dst, a, b, c` — mixed signed/unsigned 4-way byte dot product.
    pub fn dp4a_s32_u32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.dp4a_typed(a, b, c, true, false)
    }

    /// Emits `dp4a.u32.s32 dst, a, b, c` — mixed unsigned/signed 4-way byte dot product.
    pub fn dp4a_u32_s32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.dp4a_typed(a, b, c, false, true)
    }

    /// Generic dp4a helper.
    fn dp4a_typed(
        &mut self,
        a: Register,
        b: Register,
        c: Register,
        signed_a: bool,
        signed_b: bool,
    ) -> Register {
        let dst = self.regs.alloc(PtxType::S32);
        self.emit(Instruction::Dp4a {
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
            c: Operand::Register(c),
            signed_a,
            signed_b,
        });
        dst
    }

    /// Emits `dp2a.lo.u32.u32 dst, a, b, c` — unsigned 2-way dot product, low half.
    pub fn dp2a_lo_u32_u32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.dp2a_typed(a, b, c, false, false, true)
    }

    /// Emits `dp2a.hi.u32.u32 dst, a, b, c` — unsigned 2-way dot product, high half.
    pub fn dp2a_hi_u32_u32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.dp2a_typed(a, b, c, false, false, false)
    }

    /// Emits `dp2a.lo.s32.s32 dst, a, b, c` — signed 2-way dot product, low half.
    pub fn dp2a_lo_s32_s32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.dp2a_typed(a, b, c, true, true, true)
    }

    /// Emits `dp2a.hi.s32.s32 dst, a, b, c` — signed 2-way dot product, high half.
    pub fn dp2a_hi_s32_s32(&mut self, a: Register, b: Register, c: Register) -> Register {
        self.dp2a_typed(a, b, c, true, true, false)
    }

    /// Generic dp2a helper.
    fn dp2a_typed(
        &mut self,
        a: Register,
        b: Register,
        c: Register,
        signed_a: bool,
        signed_b: bool,
        lo: bool,
    ) -> Register {
        let dst = self.regs.alloc(PtxType::S32);
        self.emit(Instruction::Dp2a {
            dst: dst.clone(),
            a: Operand::Register(a),
            b: Operand::Register(b),
            c: Operand::Register(c),
            signed_a,
            signed_b,
            lo,
        });
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Immediate Value Helpers
    // ════════════════════════════════════════════════════════════════════

    /// Creates an unsigned 32-bit immediate operand.
    #[must_use]
    pub const fn imm_u32(&self, val: u32) -> Operand {
        Operand::Immediate(ImmValue::U32(val))
    }

    /// Loads an unsigned 32-bit immediate into a new register via `add.u32 dst, 0, val`.
    pub fn mov_imm_u32(&mut self, val: u32) -> Register {
        let dst = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::Add {
            ty: PtxType::U32,
            dst: dst.clone(),
            a: Operand::Immediate(ImmValue::U32(0)),
            b: Operand::Immediate(ImmValue::U32(val)),
        });
        dst
    }

    /// Creates an unsigned 64-bit immediate operand.
    #[must_use]
    pub const fn imm_u64(&self, val: u64) -> Operand {
        Operand::Immediate(ImmValue::U64(val))
    }

    /// Creates a 32-bit floating-point immediate operand.
    #[must_use]
    pub const fn imm_f32(&self, val: f32) -> Operand {
        Operand::Immediate(ImmValue::F32(val))
    }

    /// Creates a 64-bit floating-point immediate operand.
    #[must_use]
    pub const fn imm_f64(&self, val: f64) -> Operand {
        Operand::Immediate(ImmValue::F64(val))
    }

    // ════════════════════════════════════════════════════════════════════
    //  Miscellaneous / Escape Hatches
    // ════════════════════════════════════════════════════════════════════

    /// Emits a comment in the PTX output (for debugging / readability).
    pub fn comment(&mut self, text: &str) {
        self.emit(Instruction::Comment(text.to_string()));
    }

    /// Emits raw PTX text verbatim. Use as an escape hatch for instructions
    /// not yet modeled in the IR.
    ///
    /// The text should be a single instruction without trailing newline.
    pub fn raw_ptx(&mut self, text: &str) {
        self.emit(Instruction::Raw(text.to_string()));
    }

    // ════════════════════════════════════════════════════════════════════
    //  Address Computation Helpers
    // ════════════════════════════════════════════════════════════════════

    /// Computes a byte offset address: `base + index * stride`.
    ///
    /// Useful for computing element addresses in arrays. The index is
    /// zero-extended from `u32` to `u64` before the multiplication.
    ///
    /// Returns a `U64` register containing the computed address.
    pub fn byte_offset_addr(
        &mut self,
        base: Register,
        index: Register,
        stride_bytes: u32,
    ) -> Register {
        let idx64 = self.cvt_u32_to_u64(index);
        // Use mad.wide.u32 to compute idx * stride + base... but we need
        // mul then add since mad with mixed types isn't straightforward.
        let stride_reg = self.regs.alloc(PtxType::U64);
        self.emit(Instruction::Raw(format!(
            "mov.u64 {}, {};",
            stride_reg,
            u64::from(stride_bytes)
        )));
        let offset = self.regs.alloc(PtxType::U64);
        self.emit(Instruction::Mul {
            ty: PtxType::U64,
            mode: MulMode::Lo,
            dst: offset.clone(),
            a: Operand::Register(idx64),
            b: Operand::Register(stride_reg),
        });
        self.add_u64(base, offset)
    }

    /// Computes an element address for an `f32` array: `base + index * 4`.
    pub fn f32_elem_addr(&mut self, base: Register, index: Register) -> Register {
        self.byte_offset_addr(base, index, 4)
    }

    /// Computes an element address for an `f64` array: `base + index * 8`.
    pub fn f64_elem_addr(&mut self, base: Register, index: Register) -> Register {
        self.byte_offset_addr(base, index, 8)
    }

    // ════════════════════════════════════════════════════════════════════
    //  Register Allocation (Direct Access)
    // ════════════════════════════════════════════════════════════════════

    /// Allocates a fresh register of the given type.
    ///
    /// This is a lower-level API — most users should prefer the typed
    /// instruction methods which allocate destination registers automatically.
    pub fn alloc_reg(&mut self, ty: PtxType) -> Register {
        self.regs.alloc(ty)
    }

    // ════════════════════════════════════════════════════════════════════
    //  Internal Helpers
    // ════════════════════════════════════════════════════════════════════

    /// Appends an instruction to the body.
    fn emit(&mut self, inst: Instruction) {
        self.instructions.push(inst);
    }

    /// Generates a unique label name with the given prefix.
    ///
    /// Labels are formatted as `L__{prefix}_{counter}` to avoid
    /// collisions with user-defined labels and other generated labels.
    pub fn fresh_label(&mut self, prefix: &str) -> String {
        let id = self.label_counter;
        self.label_counter += 1;
        format!("L__{prefix}_{id}")
    }

    /// Returns the target SM version for this kernel.
    ///
    /// Useful for architecture-gated code paths within body closures.
    #[must_use]
    pub const fn target_sm(&self) -> SmVersion {
        self.target
    }

    /// Returns `true` if the given parameter name was declared on the kernel.
    #[must_use]
    pub fn has_param(&self, name: &str) -> bool {
        self.param_names.iter().any(|p| p == name)
    }
}

// Atomic/reduce, texture/surface, warp-level primitives, and barrier methods
// live in a sibling module to keep this file under the 2000-line policy.
pub(super) mod body_builder_ext;

// Extended tensor core builder: WMMA, MMA (TF32/BF16/FP8/INT8), WGMMA.
pub(super) mod tensor_core_ops;

#[cfg(test)]
#[path = "body_builder_tests.rs"]
mod tests;
