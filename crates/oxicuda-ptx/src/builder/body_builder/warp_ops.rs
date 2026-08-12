//! Warp-level instruction emission: `shfl.sync`, `vote.sync`, lane queries.
//!
//! These are the low-level building blocks beneath the SIMD-flavored
//! [`WarpVec`](super::warp_vec::WarpVec) layer: typed wrappers over the
//! [`Instruction::Shfl`] / [`Instruction::Vote`] IR variants, including the
//! composite two-half datapath that routes 64-bit values through the
//! 32-bit-only `shfl.sync` hardware.
//!
//! # Lane-selector encoding
//!
//! The PTX `c` operand of `shfl.sync` packs a source-lane clamp in bits
//! `[4:0]` and a segment mask in bits `[12:8]`. CUDA's
//! `__shfl_*_sync(mask, var, lane, width)` intrinsics encode a logical
//! sub-warp `width` (a power of two ≤ 32) as:
//!
//! ```text
//! c = ((32 - width) << 8) | clamp
//!     clamp = 0x1f  for idx / down / bfly
//!     clamp = 0     for up
//! ```
//!
//! [`shfl_c_value`] reproduces exactly that encoding, so kernels built here
//! are lane-for-lane equivalent to their CUDA C counterparts.

use crate::error::PtxGenError;
use crate::ir::{
    ImmValue, Instruction, Operand, PtxType, Register, ShflMode, SpecialReg, VoteMode,
};

use super::BodyBuilder;

/// Full-warp membership mask: all 32 lanes participate.
pub const FULL_WARP_MASK: u32 = 0xFFFF_FFFF;

/// Number of hardware lanes in a warp on every PTX target.
///
/// NVIDIA warps are 32 lanes wide on all architectures this crate supports.
/// Logical sub-warp segments (width 2 / 4 / 8 / 16) are expressed through the
/// `width` parameter of the shuffle helpers, never by changing this constant.
pub const WARP_SIZE: u32 = 32;

/// Returns `true` if `width` is a legal logical warp-segment width
/// (a power of two in `2..=32`).
#[must_use]
pub const fn is_valid_warp_width(width: u32) -> bool {
    width.is_power_of_two() && width >= 2 && width <= WARP_SIZE
}

/// Computes the packed `c` operand for a `shfl.sync` at the given logical
/// segment `width`, following the CUDA intrinsic encoding (see module docs).
#[must_use]
pub const fn shfl_c_value(mode: ShflMode, width: u32) -> u32 {
    let seg = (WARP_SIZE - width) << 8;
    match mode {
        ShflMode::Up => seg,
        ShflMode::Idx | ShflMode::Down | ShflMode::Bfly => seg | 0x1F,
    }
}

/// Returns `true` for element types that ride the `shfl.sync.b32` datapath
/// directly (any 32-bit register class).
const fn is_shfl_32bit(ty: PtxType) -> bool {
    matches!(
        ty,
        PtxType::F32 | PtxType::U32 | PtxType::S32 | PtxType::B32
    )
}

/// Returns `true` for element types shuffled as two 32-bit halves.
const fn is_shfl_64bit(ty: PtxType) -> bool {
    matches!(
        ty,
        PtxType::F64 | PtxType::U64 | PtxType::S64 | PtxType::B64
    )
}

impl BodyBuilder<'_> {
    // ════════════════════════════════════════════════════════════════════
    //  Lane queries
    // ════════════════════════════════════════════════════════════════════

    /// Reads `%laneid` — this thread's lane index within its warp (`0..32`).
    pub fn lane_id(&mut self) -> Register {
        let dst = self.regs.alloc(PtxType::U32);
        self.emit(Instruction::MovSpecial {
            dst: dst.clone(),
            special: SpecialReg::LaneId,
        });
        dst
    }

    /// Computes the warp index within the block for a 1-D thread layout:
    /// `%tid.x >> 5`.
    ///
    /// Note this is *not* `%warpid` (the hardware scheduler slot, which is
    /// neither stable nor contiguous); it is the conventional software warp
    /// index used to address per-warp scratch storage.
    pub fn warp_index_x(&mut self) -> Register {
        let tid = self.thread_id_x();
        let five = self.mov_typed(PtxType::U32, Operand::Immediate(ImmValue::U32(5)));
        self.shr_u32(tid, five)
    }

    // ════════════════════════════════════════════════════════════════════
    //  Generic register moves
    // ════════════════════════════════════════════════════════════════════

    /// Emits `mov.{ty} dst, src` — materializes an immediate or copies a
    /// register into a fresh register of type `ty`.
    ///
    /// Float immediates are emitted as PTX hex literals (`0f…` / `0d…`), so
    /// every bit pattern (±0.0, NaN, infinities, subnormals) round-trips
    /// exactly.
    pub fn mov_typed(&mut self, ty: PtxType, src: Operand) -> Register {
        let dst = self.regs.alloc(ty);
        self.emit(Instruction::Mov {
            ty,
            dst: dst.clone(),
            src,
        });
        dst
    }

    // ════════════════════════════════════════════════════════════════════
    //  Warp shuffle (shfl.sync)
    // ════════════════════════════════════════════════════════════════════

    /// Emits a warp shuffle of `val` and returns the received value.
    ///
    /// * `mode` — source-lane selection ([`ShflMode`]).
    /// * `val` — the per-lane value to exchange. 32-bit types use one
    ///   `shfl.sync.b32`; 64-bit types are split with `mov.b64 {lo, hi}`,
    ///   shuffled per half, and recombined.
    /// * `lane` — the `b` operand (delta, XOR mask, or absolute lane,
    ///   depending on `mode`); register or immediate.
    /// * `width` — logical segment width (power of two in `2..=32`).
    ///   `32` is the full-warp case.
    ///
    /// Out-of-segment sources return the calling thread's own `val`
    /// (hardware clamp semantics) — use [`shfl_sync_with_valid`] when the
    /// in-range predicate is needed.
    ///
    /// All lanes named by the membership mask (the full warp) must execute
    /// this instruction: do not place it under divergent control flow.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if `width` is not a power
    /// of two in `2..=32`, or if `val`'s type is not a 32-bit or 64-bit
    /// register class.
    ///
    /// [`shfl_sync_with_valid`]: Self::shfl_sync_with_valid
    pub fn shfl_sync(
        &mut self,
        mode: ShflMode,
        val: &Register,
        lane: Operand,
        width: u32,
    ) -> Result<Register, PtxGenError> {
        let (value, _pred) = self.shfl_sync_impl(mode, val, lane, width, false)?;
        Ok(value)
    }

    /// Like [`shfl_sync`](Self::shfl_sync), but additionally returns the
    /// in-range predicate: `true` iff the computed source lane was inside
    /// the segment (for 64-bit types the predicate comes from the low-half
    /// shuffle; both halves share the same lane geometry).
    ///
    /// This is the primitive behind guarded scan steps
    /// (`acc = valid ? acc + shifted : acc`).
    ///
    /// # Errors
    ///
    /// Same conditions as [`shfl_sync`](Self::shfl_sync).
    pub fn shfl_sync_with_valid(
        &mut self,
        mode: ShflMode,
        val: &Register,
        lane: Operand,
        width: u32,
    ) -> Result<(Register, Register), PtxGenError> {
        let (value, pred) = self.shfl_sync_impl(mode, val, lane, width, true)?;
        // `shfl_sync_impl` always yields the predicate when requested.
        pred.map_or_else(
            || {
                Err(PtxGenError::GenerationFailed(
                    "internal: shfl predicate requested but not produced".to_string(),
                ))
            },
            |p| Ok((value, p)),
        )
    }

    /// Shared implementation for the two public shuffle entry points.
    fn shfl_sync_impl(
        &mut self,
        mode: ShflMode,
        val: &Register,
        lane: Operand,
        width: u32,
        want_pred: bool,
    ) -> Result<(Register, Option<Register>), PtxGenError> {
        if !is_valid_warp_width(width) {
            return Err(PtxGenError::GenerationFailed(format!(
                "shfl.sync width must be a power of two in 2..=32, got {width}"
            )));
        }
        let c = shfl_c_value(mode, width);

        if is_shfl_32bit(val.ty) {
            let (dst, pred) = self.shfl_b32_component(mode, val, lane, c, want_pred, val.ty);
            return Ok((dst, pred));
        }

        if is_shfl_64bit(val.ty) {
            // Split → shuffle both halves → recombine. The halves are plain
            // 32-bit bit-patterns, so they travel as B32.
            let lo = self.regs.alloc(PtxType::B32);
            let hi = self.regs.alloc(PtxType::B32);
            self.emit(Instruction::UnpackB64x2 {
                lo: lo.clone(),
                hi: hi.clone(),
                src: val.clone(),
            });
            let (lo_shfl, pred) =
                self.shfl_b32_component(mode, &lo, lane.clone(), c, want_pred, PtxType::B32);
            let (hi_shfl, _) = self.shfl_b32_component(mode, &hi, lane, c, false, PtxType::B32);
            let dst = self.regs.alloc(val.ty);
            self.emit(Instruction::PackB64x2 {
                dst: dst.clone(),
                lo: lo_shfl,
                hi: hi_shfl,
            });
            return Ok((dst, pred));
        }

        Err(PtxGenError::GenerationFailed(format!(
            "shfl.sync supports 32-bit (f32/u32/s32/b32) and 64-bit \
             (f64/u64/s64/b64) register classes, got {:?}",
            val.ty
        )))
    }

    /// Emits one `shfl.sync{mode}.b32` on a 32-bit register, allocating the
    /// destination as `dst_ty` (the payload is untyped bits; keeping the
    /// source's register class avoids spurious cross-bank moves).
    fn shfl_b32_component(
        &mut self,
        mode: ShflMode,
        src: &Register,
        lane: Operand,
        c: u32,
        want_pred: bool,
        dst_ty: PtxType,
    ) -> (Register, Option<Register>) {
        let dst = self.regs.alloc(dst_ty);
        let dst_pred = if want_pred {
            Some(self.regs.alloc(PtxType::Pred))
        } else {
            None
        };
        self.emit(Instruction::Shfl {
            mode,
            dst: dst.clone(),
            dst_pred: dst_pred.clone(),
            src: Operand::Register(src.clone()),
            lane,
            c: Operand::Immediate(ImmValue::U32(c)),
            membership_mask: FULL_WARP_MASK,
        });
        (dst, dst_pred)
    }

    // ════════════════════════════════════════════════════════════════════
    //  Warp vote (vote.sync)
    // ════════════════════════════════════════════════════════════════════

    /// Emits `vote.sync.all.pred` — a predicate that is `true` iff `pred`
    /// is `true` in **all** 32 lanes.
    pub fn vote_all(&mut self, pred: &Register) -> Register {
        self.vote_pred(VoteMode::All, pred)
    }

    /// Emits `vote.sync.any.pred` — a predicate that is `true` iff `pred`
    /// is `true` in **any** lane.
    pub fn vote_any(&mut self, pred: &Register) -> Register {
        self.vote_pred(VoteMode::Any, pred)
    }

    /// Emits `vote.sync.uni.pred` — a predicate that is `true` iff `pred`
    /// has the **same** value in every lane (warp-uniformity test).
    pub fn vote_uni(&mut self, pred: &Register) -> Register {
        self.vote_pred(VoteMode::Uni, pred)
    }

    /// Emits `vote.sync.ballot.b32` — a 32-bit mask with bit *i* set iff
    /// lane *i*'s `pred` is `true`. Every lane receives the same mask.
    pub fn vote_ballot(&mut self, pred: &Register) -> Register {
        let dst = self.regs.alloc(PtxType::B32);
        self.emit(Instruction::Vote {
            mode: VoteMode::Ballot,
            dst: dst.clone(),
            src: pred.clone(),
            negate_src: false,
            membership_mask: FULL_WARP_MASK,
        });
        dst
    }

    /// Shared predicate-destination vote emitter.
    fn vote_pred(&mut self, mode: VoteMode, pred: &Register) -> Register {
        let dst = self.regs.alloc(PtxType::Pred);
        self.emit(Instruction::Vote {
            mode,
            dst: dst.clone(),
            src: pred.clone(),
            negate_src: false,
            membership_mask: FULL_WARP_MASK,
        });
        dst
    }

    /// Emits `not.pred dst, src` — logical negation of a predicate register.
    pub fn not_pred(&mut self, pred: &Register) -> Register {
        let dst = self.regs.alloc(PtxType::Pred);
        self.emit(Instruction::Not {
            ty: PtxType::Pred,
            dst: dst.clone(),
            src: Operand::Register(pred.clone()),
        });
        dst
    }
}
