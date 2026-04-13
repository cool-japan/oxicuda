//! Extension methods for [`BodyBuilder`] — atomic/reduce operations,
//! texture/surface instructions, warp-level primitives, and barrier/
//! synchronization instructions.
//!
//! Split from `body_builder.rs` to respect the 2000-line policy.
//! Declared as a child module of `body_builder` so it can access
//! `pub(super)` fields directly.

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::{
    AtomOp, FenceScope, GridDepAction, Instruction, MemorySpace, Operand, PtxType, ReduxOp,
    Register, RoundingMode, SetmaxnregAction, StmatrixShape,
};

use super::BodyBuilder;

impl BodyBuilder<'_> {
    // ════════════════════════════════════════════════════════════════════
    //  Atomic Operations
    // ════════════════════════════════════════════════════════════════════

    /// Atomic add on global memory (f32): returns old value at `[addr]`, stores `old + val`.
    pub fn atom_global_add_f32(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Global, AtomOp::Add, PtxType::F32, addr, val)
    }

    /// Atomic add on global memory (u32): returns old value at `[addr]`, stores `old + val`.
    pub fn atom_global_add_u32(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Global, AtomOp::Add, PtxType::U32, addr, val)
    }

    /// Atomic add on global memory (u64): returns old value at `[addr]`, stores `old + val`.
    pub fn atom_global_add_u64(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Global, AtomOp::Add, PtxType::U64, addr, val)
    }

    /// Atomic add on global memory (f64): returns old value at `[addr]`, stores `old + val`.
    pub fn atom_global_add_f64(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Global, AtomOp::Add, PtxType::F64, addr, val)
    }

    /// Atomic compare-and-swap on global memory (u32).
    ///
    /// If `[addr] == compare`, stores `value`. Returns the old value at `[addr]`.
    pub fn atom_global_cas_u32(
        &mut self,
        addr: Register,
        compare: Register,
        value: Register,
    ) -> Register {
        self.atom_cas_typed(MemorySpace::Global, PtxType::U32, addr, compare, value)
    }

    /// Atomic compare-and-swap on global memory (u64).
    ///
    /// If `[addr] == compare`, stores `value`. Returns the old value at `[addr]`.
    pub fn atom_global_cas_u64(
        &mut self,
        addr: Register,
        compare: Register,
        value: Register,
    ) -> Register {
        self.atom_cas_typed(MemorySpace::Global, PtxType::U64, addr, compare, value)
    }

    /// Atomic exchange on global memory (u32): stores `val`, returns old value.
    pub fn atom_global_exch_u32(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Global, AtomOp::Exch, PtxType::U32, addr, val)
    }

    /// Atomic min on global memory (u32).
    pub fn atom_global_min_u32(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Global, AtomOp::Min, PtxType::U32, addr, val)
    }

    /// Atomic max on global memory (u32).
    pub fn atom_global_max_u32(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Global, AtomOp::Max, PtxType::U32, addr, val)
    }

    /// Atomic min on global memory (s32).
    pub fn atom_global_min_s32(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Global, AtomOp::Min, PtxType::S32, addr, val)
    }

    /// Atomic max on global memory (s32).
    pub fn atom_global_max_s32(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Global, AtomOp::Max, PtxType::S32, addr, val)
    }

    /// Atomic bitwise AND on global memory (b32).
    pub fn atom_global_and_b32(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Global, AtomOp::And, PtxType::B32, addr, val)
    }

    /// Atomic bitwise OR on global memory (b32).
    pub fn atom_global_or_b32(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Global, AtomOp::Or, PtxType::B32, addr, val)
    }

    /// Atomic bitwise XOR on global memory (b32).
    pub fn atom_global_xor_b32(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Global, AtomOp::Xor, PtxType::B32, addr, val)
    }

    /// Atomic add on shared memory (f32): critical for block-level reductions.
    pub fn atom_shared_add_f32(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Shared, AtomOp::Add, PtxType::F32, addr, val)
    }

    /// Atomic add on shared memory (u32).
    pub fn atom_shared_add_u32(&mut self, addr: Register, val: Register) -> Register {
        self.atom_typed(MemorySpace::Shared, AtomOp::Add, PtxType::U32, addr, val)
    }

    /// Atomic reduction (fire-and-forget) add on global memory (f32).
    ///
    /// Unlike `atom`, this does not return the old value and may be faster.
    pub fn red_global_add_f32(&mut self, addr: Register, val: Register) {
        self.red_typed(MemorySpace::Global, AtomOp::Add, PtxType::F32, addr, val);
    }

    /// Atomic reduction (fire-and-forget) add on global memory (u32).
    pub fn red_global_add_u32(&mut self, addr: Register, val: Register) {
        self.red_typed(MemorySpace::Global, AtomOp::Add, PtxType::U32, addr, val);
    }

    /// Generic atomic operation helper.
    fn atom_typed(
        &mut self,
        space: MemorySpace,
        op: AtomOp,
        ty: PtxType,
        addr: Register,
        src: Register,
    ) -> Register {
        let dst = self.regs.alloc(ty);
        self.instructions.push(Instruction::Atom {
            space,
            op,
            ty,
            dst: dst.clone(),
            addr: Operand::Register(addr),
            src: Operand::Register(src),
        });
        dst
    }

    /// Generic atomic compare-and-swap helper.
    fn atom_cas_typed(
        &mut self,
        space: MemorySpace,
        ty: PtxType,
        addr: Register,
        compare: Register,
        value: Register,
    ) -> Register {
        let dst = self.regs.alloc(ty);
        self.instructions.push(Instruction::AtomCas {
            space,
            ty,
            dst: dst.clone(),
            addr: Operand::Register(addr),
            compare: Operand::Register(compare),
            value: Operand::Register(value),
        });
        dst
    }

    /// Generic atomic reduction helper (no return value).
    fn red_typed(
        &mut self,
        space: MemorySpace,
        op: AtomOp,
        ty: PtxType,
        addr: Register,
        src: Register,
    ) {
        self.instructions.push(Instruction::Red {
            space,
            op,
            ty,
            addr: Operand::Register(addr),
            src: Operand::Register(src),
        });
    }

    // ════════════════════════════════════════════════════════════════════
    //  Texture / Surface Operations
    // ════════════════════════════════════════════════════════════════════

    /// Emits a 1D texture fetch instruction.
    ///
    /// Fetches a texel from the named texture reference at the given
    /// integer coordinate. Returns the destination register.
    ///
    /// Emits: `tex.1d.v4.{ty}.s32 dst, [tex_ref, {coord}];`
    pub fn tex_1d(&mut self, ty: PtxType, tex_ref: &str, coord: Operand) -> Register {
        let dst = self.regs.alloc(ty);
        self.instructions.push(Instruction::Tex1d {
            ty,
            dst: dst.clone(),
            tex_ref: tex_ref.to_string(),
            coord,
        });
        dst
    }

    /// Emits a 2D texture fetch instruction.
    ///
    /// Fetches a texel from the named texture reference at the given
    /// (x, y) integer coordinates. Returns the destination register.
    ///
    /// Emits: `tex.2d.v4.{ty}.s32 dst, [tex_ref, {coord_x, coord_y}];`
    pub fn tex_2d(
        &mut self,
        ty: PtxType,
        tex_ref: &str,
        coord_x: Operand,
        coord_y: Operand,
    ) -> Register {
        let dst = self.regs.alloc(ty);
        self.instructions.push(Instruction::Tex2d {
            ty,
            dst: dst.clone(),
            tex_ref: tex_ref.to_string(),
            coord_x,
            coord_y,
        });
        dst
    }

    /// Emits a 3D texture fetch instruction.
    ///
    /// Fetches a texel from the named texture reference at the given
    /// (x, y, z) integer coordinates. Returns the destination register.
    ///
    /// Emits: `tex.3d.v4.{ty}.s32 dst, [tex_ref, {coord_x, coord_y, coord_z}];`
    pub fn tex_3d(
        &mut self,
        ty: PtxType,
        tex_ref: &str,
        coord_x: Operand,
        coord_y: Operand,
        coord_z: Operand,
    ) -> Register {
        let dst = self.regs.alloc(ty);
        self.instructions.push(Instruction::Tex3d {
            ty,
            dst: dst.clone(),
            tex_ref: tex_ref.to_string(),
            coord_x,
            coord_y,
            coord_z,
        });
        dst
    }

    /// Emits a 1D surface load instruction.
    ///
    /// Loads a value from the named surface reference at the given
    /// coordinate. Returns the destination register.
    ///
    /// Emits: `suld.b.1d.{ty} dst, [surf_ref, {coord}];`
    pub fn surf_load(&mut self, ty: PtxType, surf_ref: &str, coord: Operand) -> Register {
        let dst = self.regs.alloc(ty);
        self.instructions.push(Instruction::SurfLoad {
            ty,
            dst: dst.clone(),
            surf_ref: surf_ref.to_string(),
            coord,
        });
        dst
    }

    /// Emits a 1D surface store instruction.
    ///
    /// Stores a value to the named surface reference at the given coordinate.
    ///
    /// Emits: `sust.b.1d.{ty} [surf_ref, {coord}], src;`
    pub fn surf_store(&mut self, ty: PtxType, surf_ref: &str, coord: Operand, src: Register) {
        self.instructions.push(Instruction::SurfStore {
            ty,
            surf_ref: surf_ref.to_string(),
            coord,
            src,
        });
    }

    // ════════════════════════════════════════════════════════════════════
    //  Warp-level Primitives (SM >= 80/90)
    // ════════════════════════════════════════════════════════════════════

    /// Warp-level sum reduction on a `u32` value (SM >= 80).
    pub fn redux_add_u32(&mut self, src: &str) -> Result<String, PtxGenError> {
        self.redux_op(ReduxOp::Add, src)
    }

    /// Warp-level max reduction on a `u32` value (SM >= 80).
    pub fn redux_max_u32(&mut self, src: &str) -> Result<String, PtxGenError> {
        self.redux_op(ReduxOp::Max, src)
    }

    /// Warp-level min reduction on a `u32` value (SM >= 80).
    pub fn redux_min_u32(&mut self, src: &str) -> Result<String, PtxGenError> {
        self.redux_op(ReduxOp::Min, src)
    }

    fn redux_op(&mut self, op: ReduxOp, src: &str) -> Result<String, PtxGenError> {
        if !self.target.capabilities().has_redux {
            return Err(PtxGenError::GenerationFailed(format!(
                "redux.sync requires SM >= 80, target is {}",
                self.target
            )));
        }
        let dst = self.regs.alloc(PtxType::U32);
        let name = dst.name.clone();
        self.instructions.push(Instruction::Redux {
            op,
            dst,
            src: Operand::Register(Register {
                name: src.to_string(),
                ty: PtxType::U32,
            }),
            membership_mask: 0xFFFF_FFFF,
        });
        Ok(name)
    }

    /// Store matrix m8n8x4 to shared memory (SM >= 90).
    pub fn stmatrix_m8n8x4(&mut self, addr: &str, src: &str) -> Result<(), PtxGenError> {
        if !self.target.capabilities().has_stmatrix {
            return Err(PtxGenError::GenerationFailed(format!(
                "stmatrix requires SM >= 90, target is {}",
                self.target
            )));
        }
        self.instructions.push(Instruction::Stmatrix {
            dst_addr: Operand::Register(Register {
                name: addr.to_string(),
                ty: PtxType::U32,
            }),
            src: Register {
                name: src.to_string(),
                ty: PtxType::B32,
            },
            shape: StmatrixShape::M8n8x4,
            trans: false,
        });
        Ok(())
    }

    /// Elect a single warp leader (SM >= 90). Returns predicate register name.
    pub fn elect_sync(&mut self) -> Result<String, PtxGenError> {
        if !self.target.capabilities().has_elect_one {
            return Err(PtxGenError::GenerationFailed(format!(
                "elect.sync requires SM >= 90, target is {}",
                self.target
            )));
        }
        let dst = self.regs.alloc(PtxType::Pred);
        let name = dst.name.clone();
        self.instructions.push(Instruction::ElectSync {
            dst,
            membership_mask: 0xFFFF_FFFF,
        });
        Ok(name)
    }

    /// Increase the maximum register count (SM >= 90).
    pub fn setmaxnreg_inc(&mut self, count: u32) -> Result<(), PtxGenError> {
        self.setmaxnreg_impl(count, SetmaxnregAction::Inc)
    }

    /// Decrease the maximum register count (SM >= 90).
    pub fn setmaxnreg_dec(&mut self, count: u32) -> Result<(), PtxGenError> {
        self.setmaxnreg_impl(count, SetmaxnregAction::Dec)
    }

    fn setmaxnreg_impl(&mut self, count: u32, action: SetmaxnregAction) -> Result<(), PtxGenError> {
        if !self.target.capabilities().has_setmaxnreg {
            return Err(PtxGenError::GenerationFailed(format!(
                "setmaxnreg requires SM >= 90, target is {}",
                self.target
            )));
        }
        self.instructions.push(Instruction::Setmaxnreg {
            reg_count: count,
            action,
        });
        Ok(())
    }

    // ════════════════════════════════════════════════════════════════════
    //  Barrier / Synchronization (SM >= 90)
    // ════════════════════════════════════════════════════════════════════

    /// Signal that dependent grids may launch (SM >= 90).
    pub fn griddepcontrol_launch_dependents(&mut self) -> Result<(), PtxGenError> {
        if !self.target.capabilities().has_griddepcontrol {
            return Err(PtxGenError::GenerationFailed(format!(
                "griddepcontrol requires SM >= 90, target is {}",
                self.target
            )));
        }
        self.instructions.push(Instruction::Griddepcontrol {
            action: GridDepAction::LaunchDependents,
        });
        Ok(())
    }

    /// Wait for grid dependencies to complete (SM >= 90).
    pub fn griddepcontrol_wait(&mut self) -> Result<(), PtxGenError> {
        if !self.target.capabilities().has_griddepcontrol {
            return Err(PtxGenError::GenerationFailed(format!(
                "griddepcontrol requires SM >= 90, target is {}",
                self.target
            )));
        }
        self.instructions.push(Instruction::Griddepcontrol {
            action: GridDepAction::Wait,
        });
        Ok(())
    }

    /// Emit a proxy fence for async operations.
    pub fn fence_proxy_async(&mut self, scope: &str) -> Result<(), PtxGenError> {
        let fence_scope = match scope {
            "cta" => FenceScope::Cta,
            "gpu" => FenceScope::Gpu,
            "sys" => FenceScope::Sys,
            other => {
                return Err(PtxGenError::GenerationFailed(format!(
                    "unknown fence scope: {other}"
                )));
            }
        };
        self.instructions.push(Instruction::FenceProxy {
            scope: fence_scope,
            space: MemorySpace::Shared,
        });
        Ok(())
    }

    /// Initialize an mbarrier in shared memory (SM >= 90).
    pub fn mbarrier_init(&mut self, addr: &str, count: &str) -> Result<(), PtxGenError> {
        if !self.target.capabilities().has_cluster_barriers {
            return Err(PtxGenError::GenerationFailed(format!(
                "mbarrier requires SM >= 90, target is {}",
                self.target
            )));
        }
        self.instructions.push(Instruction::MbarrierInit {
            addr: Operand::Register(Register {
                name: addr.to_string(),
                ty: PtxType::U64,
            }),
            count: Operand::Register(Register {
                name: count.to_string(),
                ty: PtxType::U32,
            }),
        });
        Ok(())
    }

    /// Signal arrival at an mbarrier (SM >= 90).
    pub fn mbarrier_arrive(&mut self, addr: &str) -> Result<(), PtxGenError> {
        if !self.target.capabilities().has_cluster_barriers {
            return Err(PtxGenError::GenerationFailed(format!(
                "mbarrier requires SM >= 90, target is {}",
                self.target
            )));
        }
        self.instructions.push(Instruction::MbarrierArrive {
            addr: Operand::Register(Register {
                name: addr.to_string(),
                ty: PtxType::U64,
            }),
        });
        Ok(())
    }

    /// Wait on an mbarrier phase (SM >= 90).
    pub fn mbarrier_wait(&mut self, addr: &str, phase: &str) -> Result<(), PtxGenError> {
        if !self.target.capabilities().has_cluster_barriers {
            return Err(PtxGenError::GenerationFailed(format!(
                "mbarrier requires SM >= 90, target is {}",
                self.target
            )));
        }
        self.instructions.push(Instruction::MbarrierWait {
            addr: Operand::Register(Register {
                name: addr.to_string(),
                ty: PtxType::U64,
            }),
            phase: Operand::Register(Register {
                name: phase.to_string(),
                ty: PtxType::U32,
            }),
        });
        Ok(())
    }

    // ════════════════════════════════════════════════════════════════════
    //  FP4 (E2M1) Type Conversions — Blackwell (SM >= 100)
    // ════════════════════════════════════════════════════════════════════

    /// Convert an `f32` register to FP4 E2M1 format (SM >= 100, Blackwell).
    ///
    /// Emits `cvt.rn.e2m1.f32 dst, src;` and returns the destination register.
    /// The FP4 result is stored in the low 4 bits of a B32 register container.
    pub fn cvt_f32_to_e2m1(&mut self, src: Register) -> Result<Register, PtxGenError> {
        if self.target < SmVersion::Sm100 {
            return Err(PtxGenError::GenerationFailed(format!(
                "cvt.e2m1 requires SM >= 100 (Blackwell), target is {}",
                self.target
            )));
        }
        let dst = self.regs.alloc(PtxType::E2M1);
        self.instructions.push(Instruction::Cvt {
            rnd: Some(RoundingMode::Rn),
            dst_ty: PtxType::E2M1,
            src_ty: PtxType::F32,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        Ok(dst)
    }

    /// Convert an FP4 E2M1 register to `f32` (SM >= 100, Blackwell).
    ///
    /// Emits `cvt.f32.e2m1 dst, src;` and returns the destination register.
    pub fn cvt_e2m1_to_f32(&mut self, src: Register) -> Result<Register, PtxGenError> {
        if self.target < SmVersion::Sm100 {
            return Err(PtxGenError::GenerationFailed(format!(
                "cvt.f32.e2m1 requires SM >= 100 (Blackwell), target is {}",
                self.target
            )));
        }
        let dst = self.regs.alloc(PtxType::F32);
        self.instructions.push(Instruction::Cvt {
            rnd: None,
            dst_ty: PtxType::F32,
            src_ty: PtxType::E2M1,
            dst: dst.clone(),
            src: Operand::Register(src),
        });
        Ok(dst)
    }

    // ════════════════════════════════════════════════════════════════════
    //  tcgen05 MMA — 5th-gen Tensor Core (SM >= 100)
    // ════════════════════════════════════════════════════════════════════

    /// Emit a `tcgen05.mma.cta_group::1.kind::f32` instruction (SM >= 100).
    ///
    /// This is the Blackwell 5th-generation Tensor Core MMA that operates on
    /// 128×256×256 E2M1 tiles referenced by descriptors stored in 64-bit registers.
    pub fn tcgen05_mma_m128n256k256_e2m1(
        &mut self,
        a_desc: Register,
        b_desc: Register,
    ) -> Result<(), PtxGenError> {
        if self.target < SmVersion::Sm100 {
            return Err(PtxGenError::GenerationFailed(format!(
                "tcgen05.mma requires SM >= 100 (Blackwell), target is {}",
                self.target
            )));
        }
        self.instructions
            .push(Instruction::Tcgen05Mma { a_desc, b_desc });
        Ok(())
    }

    // ════════════════════════════════════════════════════════════════════
    //  Cluster-level Barrier & Fence — SM >= 90 (Hopper+)
    // ════════════════════════════════════════════════════════════════════

    /// Emit `barrier.cluster.arrive;` — signal cluster barrier (SM >= 90).
    ///
    /// All CTAs in the cluster must arrive before any may continue past
    /// the corresponding `barrier.cluster.wait`.
    pub fn barrier_cluster(&mut self) -> Result<(), PtxGenError> {
        if !self.target.capabilities().has_cluster_barriers {
            return Err(PtxGenError::GenerationFailed(format!(
                "barrier.cluster requires SM >= 90, target is {}",
                self.target
            )));
        }
        self.instructions.push(Instruction::BarrierCluster);
        Ok(())
    }

    /// Emit `fence.mbarrier_init.release.cluster;` — cluster release fence (SM >= 90).
    ///
    /// Ensures that all preceding memory operations (including mbarrier
    /// initializations) are visible cluster-wide before the barrier is observed.
    pub fn fence_cluster(&mut self) -> Result<(), PtxGenError> {
        if !self.target.capabilities().has_cluster_barriers {
            return Err(PtxGenError::GenerationFailed(format!(
                "fence.cluster requires SM >= 90, target is {}",
                self.target
            )));
        }
        self.instructions.push(Instruction::FenceCluster);
        Ok(())
    }

    // ════════════════════════════════════════════════════════════════════
    //  TMA Descriptor-based Bulk Async Copy — SM >= 90 (Hopper+)
    // ════════════════════════════════════════════════════════════════════

    /// Emit a 1-D TMA descriptor-based bulk async copy (SM >= 90).
    ///
    /// Emits:
    /// ```ptx
    /// cp.async.bulk.tensor.1d.shared::cluster.global.tile.bulk_group
    ///     [dst_smem], [src_gmem, {desc}];
    /// ```
    ///
    /// `dst_smem` is the destination shared-memory address register,
    /// `src_gmem` is the global-memory base address register, and
    /// `desc` is the coordinate / descriptor register.
    pub fn cp_async_bulk_tensor_1d(
        &mut self,
        dst_smem: Register,
        src_gmem: Register,
        desc: Register,
    ) -> Result<(), PtxGenError> {
        if !self.target.capabilities().has_bulk_copy {
            return Err(PtxGenError::GenerationFailed(format!(
                "cp.async.bulk.tensor requires SM >= 90, target is {}",
                self.target
            )));
        }
        self.instructions.push(Instruction::CpAsyncBulk {
            dst_smem,
            src_gmem,
            desc,
        });
        Ok(())
    }
}
