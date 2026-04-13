//! Extended tensor core builder methods for [`BodyBuilder`].
//!
//! This module provides the full matrix-accelerated instruction surface:
//! - **WMMA** (Volta/Turing) — `wmma.load.a`, `wmma.load.b`, `wmma.mma.sync`,
//!   `wmma.store.d` for all three shapes (m16n16k16, m8n32k16, m32n8k16).
//! - **MMA** (Ampere/Hopper) — all missing concrete wrappers: BF16, TF32, FP8,
//!   and INT8/INT4 IMMA shapes.
//! - **WGMMA** (Hopper) — parameterised warp-group MMA via the structured IR.

use super::BodyBuilder;
use crate::error::PtxGenError;
use crate::ir::{
    Instruction, MmaShape, Operand, PtxType, Register, WgmmaShape, WmmaLayout, WmmaOp, WmmaShape,
};

// ─── WMMA helpers (Volta / Turing) ───────────────────────────────────────────

/// Fragment register count per thread for A/B, by WMMA shape.
/// PTX ISA 8.x Table 108: each thread holds 8 registers for A or B.
const fn wmma_ab_regs(shape: WmmaShape) -> u32 {
    match shape {
        WmmaShape::M16N16K16 | WmmaShape::M8N32K16 | WmmaShape::M32N8K16 => 8,
    }
}

/// Fragment register count per thread for C/D accumulators.
const fn wmma_cd_regs(shape: WmmaShape, acc_ty: PtxType) -> u32 {
    match (shape, acc_ty) {
        (WmmaShape::M16N16K16 | WmmaShape::M8N32K16 | WmmaShape::M32N8K16, PtxType::F16) => 4,
        _ => 8, // F32 arms and conservative fallback
    }
}

impl BodyBuilder<'_> {
    // ════════════════════════════════════════════════════════════════════════
    //  WMMA — Load A
    // ════════════════════════════════════════════════════════════════════════

    /// Emit `wmma.load.a.sync.aligned{shape}{layout}.f16` for any WMMA shape.
    ///
    /// Returns the 8 allocated A-fragment registers.
    ///
    /// # Parameters
    /// - `shape`  — tile shape (all 3 shapes allocate 8 registers per thread).
    /// - `layout` — `RowMajor` or `ColMajor` A matrix layout.
    /// - `addr`   — address operand pointing to the matrix tile in shared/global mem.
    /// - `stride` — optional stride operand for non-contiguous storage.
    pub fn wmma_load_a_f16(
        &mut self,
        shape: WmmaShape,
        layout: WmmaLayout,
        addr: Operand,
        stride: Option<Operand>,
    ) -> Vec<Register> {
        let n = wmma_ab_regs(shape);
        let frags = self.regs.alloc_group(PtxType::B32, n);
        self.emit(Instruction::Wmma {
            op: WmmaOp::LoadA,
            shape,
            layout,
            ty: PtxType::F16,
            fragments: frags.clone(),
            addr: Some(addr),
            stride,
        });
        frags
    }

    // ════════════════════════════════════════════════════════════════════════
    //  WMMA — Load B
    // ════════════════════════════════════════════════════════════════════════

    /// Emit `wmma.load.b.sync.aligned{shape}{layout}.f16` for any WMMA shape.
    ///
    /// Returns the 8 allocated B-fragment registers.
    pub fn wmma_load_b_f16(
        &mut self,
        shape: WmmaShape,
        layout: WmmaLayout,
        addr: Operand,
        stride: Option<Operand>,
    ) -> Vec<Register> {
        let n = wmma_ab_regs(shape);
        let frags = self.regs.alloc_group(PtxType::B32, n);
        self.emit(Instruction::Wmma {
            op: WmmaOp::LoadB,
            shape,
            layout,
            ty: PtxType::F16,
            fragments: frags.clone(),
            addr: Some(addr),
            stride,
        });
        frags
    }

    // ════════════════════════════════════════════════════════════════════════
    //  WMMA — Store D
    // ════════════════════════════════════════════════════════════════════════

    /// Emit `wmma.store.d.sync.aligned{shape}{layout}.f32` for any WMMA shape.
    ///
    /// Writes the 8 F32 accumulator registers back to memory.
    ///
    /// # Parameters
    /// - `addr`  — destination address in shared/global memory.
    /// - `regs`  — the accumulator fragment registers to store.
    /// - `stride` — optional store stride.
    pub fn wmma_store_d_f32(
        &mut self,
        shape: WmmaShape,
        layout: WmmaLayout,
        addr: Operand,
        regs: Vec<Register>,
        stride: Option<Operand>,
    ) {
        self.emit(Instruction::Wmma {
            op: WmmaOp::StoreD,
            shape,
            layout,
            ty: PtxType::F32,
            fragments: regs,
            addr: Some(addr),
            stride,
        });
    }

    /// Emit `wmma.store.d.sync.aligned{shape}{layout}.f16` for any WMMA shape.
    pub fn wmma_store_d_f16(
        &mut self,
        shape: WmmaShape,
        layout: WmmaLayout,
        addr: Operand,
        regs: Vec<Register>,
        stride: Option<Operand>,
    ) {
        self.emit(Instruction::Wmma {
            op: WmmaOp::StoreD,
            shape,
            layout,
            ty: PtxType::F16,
            fragments: regs,
            addr: Some(addr),
            stride,
        });
    }

    // ════════════════════════════════════════════════════════════════════════
    //  WMMA — MMA sync (parameterised accumulator type)
    // ════════════════════════════════════════════════════════════════════════

    /// Emit `wmma.mma.sync.aligned{shape}{layout}.f32.f16.f16.f32` —
    /// F16 inputs with F32 accumulation (most common WMMA usage).
    ///
    /// Returns the 8 F32 accumulator destination registers.
    ///
    /// # Parameters
    /// - `a_regs`  — 8 fragment registers loaded by [`wmma_load_a_f16`].
    /// - `b_regs`  — 8 fragment registers loaded by [`wmma_load_b_f16`](BodyBuilder::wmma_load_b_f16).
    /// - `c_regs`  — 8 F32 accumulator input registers.
    ///
    /// [`wmma_load_a_f16`]: BodyBuilder::wmma_load_a_f16
    pub fn wmma_mma_sync_f16_f32(
        &mut self,
        shape: WmmaShape,
        layout: WmmaLayout,
        a_regs: Vec<Register>,
        b_regs: Vec<Register>,
        c_regs: Vec<Register>,
    ) -> Vec<Register> {
        let n = wmma_cd_regs(shape, PtxType::F32);
        let d_regs = self.regs.alloc_group(PtxType::F32, n);
        // PTX packs A, B, C, D into a single fragment list for wmma.mma.
        let mut all = d_regs.clone();
        all.extend(a_regs);
        all.extend(b_regs);
        all.extend(c_regs);
        self.emit(Instruction::Wmma {
            op: WmmaOp::Mma,
            shape,
            layout,
            ty: PtxType::F32,
            fragments: all,
            addr: None,
            stride: None,
        });
        d_regs
    }

    /// Emit `wmma.mma.sync.aligned{shape}{layout}.f16.f16.f16.f16` —
    /// full F16 WMMA with F16 accumulation.
    pub fn wmma_mma_sync_f16_f16(
        &mut self,
        shape: WmmaShape,
        layout: WmmaLayout,
        a_regs: Vec<Register>,
        b_regs: Vec<Register>,
        c_regs: Vec<Register>,
    ) -> Vec<Register> {
        let n = wmma_cd_regs(shape, PtxType::F16);
        let d_regs = self.regs.alloc_group(PtxType::F16, n);
        let mut all = d_regs.clone();
        all.extend(a_regs);
        all.extend(b_regs);
        all.extend(c_regs);
        self.emit(Instruction::Wmma {
            op: WmmaOp::Mma,
            shape,
            layout,
            ty: PtxType::F16,
            fragments: all,
            addr: None,
            stride: None,
        });
        d_regs
    }

    // ════════════════════════════════════════════════════════════════════════
    //  MMA sync — TF32 (Ampere+)
    // ════════════════════════════════════════════════════════════════════════

    /// Emit `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`.
    ///
    /// TF32 inputs with F32 accumulation.  Requires Ampere (`sm_80+`).
    /// Operand register counts per thread: A=2, B=1, C/D=4.
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Ampere.
    pub fn mma_m16n8k8_tf32_f32(
        &mut self,
        a_regs: &[Register],
        b_regs: &[Register],
        c_regs: &[Register],
    ) -> Result<[Register; 4], PtxGenError> {
        if !self.target.capabilities().has_ampere_mma {
            return Err(PtxGenError::UnsupportedFeature {
                arch: self.target.as_ptx_str().to_string(),
                feature: "mma.sync m16n8k8.tf32 (Ampere+)".to_string(),
            });
        }
        let dst = self.regs.alloc_group(PtxType::F32, 4);
        self.emit(Instruction::Mma {
            shape: MmaShape::M16N8K8,
            a_ty: PtxType::TF32,
            b_ty: PtxType::TF32,
            c_ty: PtxType::F32,
            d_ty: PtxType::F32,
            d_regs: dst.clone(),
            a_regs: a_regs.to_vec(),
            b_regs: b_regs.to_vec(),
            c_regs: c_regs.to_vec(),
        });
        Ok([
            dst[0].clone(),
            dst[1].clone(),
            dst[2].clone(),
            dst[3].clone(),
        ])
    }

    // ════════════════════════════════════════════════════════════════════════
    //  MMA sync — BF16 (Ampere+)
    // ════════════════════════════════════════════════════════════════════════

    /// Emit `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
    ///
    /// BF16 inputs with F32 accumulation.  Requires Ampere (`sm_80+`).
    /// Register counts per thread: A=4, B=2, C/D=4.
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Ampere.
    pub fn mma_m16n8k16_bf16_f32(
        &mut self,
        a_regs: &[Register],
        b_regs: &[Register],
        c_regs: &[Register],
    ) -> Result<[Register; 4], PtxGenError> {
        if !self.target.capabilities().has_ampere_mma {
            return Err(PtxGenError::UnsupportedFeature {
                arch: self.target.as_ptx_str().to_string(),
                feature: "mma.sync m16n8k16.bf16 (Ampere+)".to_string(),
            });
        }
        let dst = self.regs.alloc_group(PtxType::F32, 4);
        self.emit(Instruction::Mma {
            shape: MmaShape::M16N8K16,
            a_ty: PtxType::BF16,
            b_ty: PtxType::BF16,
            c_ty: PtxType::F32,
            d_ty: PtxType::F32,
            d_regs: dst.clone(),
            a_regs: a_regs.to_vec(),
            b_regs: b_regs.to_vec(),
            c_regs: c_regs.to_vec(),
        });
        Ok([
            dst[0].clone(),
            dst[1].clone(),
            dst[2].clone(),
            dst[3].clone(),
        ])
    }

    // ════════════════════════════════════════════════════════════════════════
    //  MMA sync — FP8 (Hopper+)
    // ════════════════════════════════════════════════════════════════════════

    /// Emit `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`.
    ///
    /// FP8 E4M3 inputs with F32 accumulation.  Requires Hopper (`sm_90+`).
    /// Register counts per thread: A=8, B=4, C/D=4.
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Hopper.
    pub fn mma_m16n8k32_e4m3_f32(
        &mut self,
        a_regs: &[Register],
        b_regs: &[Register],
        c_regs: &[Register],
    ) -> Result<[Register; 4], PtxGenError> {
        if self.target < crate::arch::SmVersion::Sm90 {
            return Err(PtxGenError::UnsupportedFeature {
                arch: self.target.as_ptx_str().to_string(),
                feature: "mma.sync m16n8k32.e4m3 (Hopper+)".to_string(),
            });
        }
        let dst = self.regs.alloc_group(PtxType::F32, 4);
        self.emit(Instruction::Mma {
            shape: MmaShape::M16N8K32,
            a_ty: PtxType::E4M3,
            b_ty: PtxType::E4M3,
            c_ty: PtxType::F32,
            d_ty: PtxType::F32,
            d_regs: dst.clone(),
            a_regs: a_regs.to_vec(),
            b_regs: b_regs.to_vec(),
            c_regs: c_regs.to_vec(),
        });
        Ok([
            dst[0].clone(),
            dst[1].clone(),
            dst[2].clone(),
            dst[3].clone(),
        ])
    }

    /// Emit `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32`.
    ///
    /// FP8 E5M2 inputs with F32 accumulation.  Requires Hopper (`sm_90+`).
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Hopper.
    pub fn mma_m16n8k32_e5m2_f32(
        &mut self,
        a_regs: &[Register],
        b_regs: &[Register],
        c_regs: &[Register],
    ) -> Result<[Register; 4], PtxGenError> {
        if self.target < crate::arch::SmVersion::Sm90 {
            return Err(PtxGenError::UnsupportedFeature {
                arch: self.target.as_ptx_str().to_string(),
                feature: "mma.sync m16n8k32.e5m2 (Hopper+)".to_string(),
            });
        }
        let dst = self.regs.alloc_group(PtxType::F32, 4);
        self.emit(Instruction::Mma {
            shape: MmaShape::M16N8K32,
            a_ty: PtxType::E5M2,
            b_ty: PtxType::E5M2,
            c_ty: PtxType::F32,
            d_ty: PtxType::F32,
            d_regs: dst.clone(),
            a_regs: a_regs.to_vec(),
            b_regs: b_regs.to_vec(),
            c_regs: c_regs.to_vec(),
        });
        Ok([
            dst[0].clone(),
            dst[1].clone(),
            dst[2].clone(),
            dst[3].clone(),
        ])
    }

    /// Emit `mma.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32` —
    /// the K=32 FP16 variant for Hopper.
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Hopper.
    pub fn mma_m16n8k32_f16_f32(
        &mut self,
        a_regs: &[Register],
        b_regs: &[Register],
        c_regs: &[Register],
    ) -> Result<[Register; 4], PtxGenError> {
        if self.target < crate::arch::SmVersion::Sm90 {
            return Err(PtxGenError::UnsupportedFeature {
                arch: self.target.as_ptx_str().to_string(),
                feature: "mma.sync m16n8k32.f16 (Hopper+)".to_string(),
            });
        }
        let dst = self.regs.alloc_group(PtxType::F32, 4);
        self.emit(Instruction::Mma {
            shape: MmaShape::M16N8K32,
            a_ty: PtxType::F16,
            b_ty: PtxType::F16,
            c_ty: PtxType::F32,
            d_ty: PtxType::F32,
            d_regs: dst.clone(),
            a_regs: a_regs.to_vec(),
            b_regs: b_regs.to_vec(),
            c_regs: c_regs.to_vec(),
        });
        Ok([
            dst[0].clone(),
            dst[1].clone(),
            dst[2].clone(),
            dst[3].clone(),
        ])
    }

    // ════════════════════════════════════════════════════════════════════════
    //  MMA sync — INT8 IMMA (Turing+)
    // ════════════════════════════════════════════════════════════════════════

    /// Emit `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` —
    /// INT8 IMMA for Turing/Ampere.
    ///
    /// Register counts per thread: A=1, B=1, C/D=2 (S32).
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Turing (`sm_75`).
    pub fn mma_m8n8k16_s8_s32(
        &mut self,
        a_regs: &[Register],
        b_regs: &[Register],
        c_regs: &[Register],
    ) -> Result<[Register; 2], PtxGenError> {
        if !self.target.capabilities().has_tensor_cores {
            return Err(PtxGenError::UnsupportedFeature {
                arch: self.target.as_ptx_str().to_string(),
                feature: "mma.sync m8n8k16.s8 INT8 IMMA (Turing+)".to_string(),
            });
        }
        let dst = self.regs.alloc_group(PtxType::S32, 2);
        self.emit(Instruction::Mma {
            shape: MmaShape::M8N8K16,
            a_ty: PtxType::S8,
            b_ty: PtxType::S8,
            c_ty: PtxType::S32,
            d_ty: PtxType::S32,
            d_regs: dst.clone(),
            a_regs: a_regs.to_vec(),
            b_regs: b_regs.to_vec(),
            c_regs: c_regs.to_vec(),
        });
        Ok([dst[0].clone(), dst[1].clone()])
    }

    /// Emit `mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32` —
    /// unsigned INT8 IMMA for Turing/Ampere.
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Turing.
    pub fn mma_m8n8k16_u8_s32(
        &mut self,
        a_regs: &[Register],
        b_regs: &[Register],
        c_regs: &[Register],
    ) -> Result<[Register; 2], PtxGenError> {
        if !self.target.capabilities().has_tensor_cores {
            return Err(PtxGenError::UnsupportedFeature {
                arch: self.target.as_ptx_str().to_string(),
                feature: "mma.sync m8n8k16.u8 INT8 IMMA (Turing+)".to_string(),
            });
        }
        let dst = self.regs.alloc_group(PtxType::S32, 2);
        self.emit(Instruction::Mma {
            shape: MmaShape::M8N8K16,
            a_ty: PtxType::U8,
            b_ty: PtxType::U8,
            c_ty: PtxType::S32,
            d_ty: PtxType::S32,
            d_regs: dst.clone(),
            a_regs: a_regs.to_vec(),
            b_regs: b_regs.to_vec(),
            c_regs: c_regs.to_vec(),
        });
        Ok([dst[0].clone(), dst[1].clone()])
    }

    /// Emit `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` —
    /// INT8 IMMA in the larger Ampere shape (16-wide M).
    ///
    /// Register counts per thread: A=4, B=2, C/D=4 (S32).
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Ampere.
    pub fn mma_m16n8k16_s8_s32(
        &mut self,
        a_regs: &[Register],
        b_regs: &[Register],
        c_regs: &[Register],
    ) -> Result<[Register; 4], PtxGenError> {
        if !self.target.capabilities().has_ampere_mma {
            return Err(PtxGenError::UnsupportedFeature {
                arch: self.target.as_ptx_str().to_string(),
                feature: "mma.sync m16n8k16.s8 INT8 IMMA (Ampere+)".to_string(),
            });
        }
        let dst = self.regs.alloc_group(PtxType::S32, 4);
        self.emit(Instruction::Mma {
            shape: MmaShape::M16N8K16,
            a_ty: PtxType::S8,
            b_ty: PtxType::S8,
            c_ty: PtxType::S32,
            d_ty: PtxType::S32,
            d_regs: dst.clone(),
            a_regs: a_regs.to_vec(),
            b_regs: b_regs.to_vec(),
            c_regs: c_regs.to_vec(),
        });
        Ok([
            dst[0].clone(),
            dst[1].clone(),
            dst[2].clone(),
            dst[3].clone(),
        ])
    }

    // ════════════════════════════════════════════════════════════════════════
    //  WGMMA — parameterised warp-group MMA (Hopper+)
    // ════════════════════════════════════════════════════════════════════════

    /// Emit `wgmma.mma_async.sync.aligned{shape}.f32.{a_ty}.{b_ty}` using
    /// the structured IR.
    ///
    /// This is the full parameterised WGMMA builder. It validates the target
    /// architecture, allocates the correct number of accumulator registers,
    /// and emits the structured `Instruction::Wgmma` with all required fields.
    ///
    /// # Parameters
    /// - `shape`      — one of the six M64×N×K16 shapes.
    /// - `a_ty`       — A/B element type: `F16`, `BF16`, `E4M3`, or `E5M2`.
    /// - `b_ty`       — must equal `a_ty` per WGMMA PTX ISA.
    /// - `desc_a`     — shared-memory descriptor register for A.
    /// - `desc_b`     — shared-memory descriptor register for B.
    /// - `scale_d`    — 1 to accumulate into D, 0 to zero-init D before writing.
    /// - `imm_scale_a` — immediate scale for A (typically 1).
    /// - `imm_scale_b` — immediate scale for B (typically 1).
    /// - `trans_a`    — 0 = no transpose, 1 = transpose A from col to row layout.
    /// - `trans_b`    — 0 = no transpose, 1 = transpose B from row to col layout.
    ///
    /// Returns the allocated accumulator registers (count = M×N/128).
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Hopper (`sm_90`).
    #[allow(clippy::too_many_arguments)]
    pub fn wgmma_mma_async(
        &mut self,
        shape: WgmmaShape,
        a_ty: PtxType,
        b_ty: PtxType,
        desc_a: Register,
        desc_b: Register,
        scale_d: i32,
        imm_scale_a: i32,
        imm_scale_b: i32,
        trans_a: i32,
        trans_b: i32,
    ) -> Result<Vec<Register>, PtxGenError> {
        if !self.target.capabilities().has_wgmma {
            return Err(PtxGenError::UnsupportedFeature {
                arch: self.target.as_ptx_str().to_string(),
                feature: "wgmma.mma_async (Hopper+ SM 90)".to_string(),
            });
        }
        // Accumulator count: (M × N) / 128 F32 registers per thread.
        let n_acc = wgmma_acc_regs(shape);
        let d_regs = self.regs.alloc_group(PtxType::F32, n_acc);
        self.emit(Instruction::Wgmma {
            shape,
            d_ty: PtxType::F32,
            a_ty,
            b_ty,
            desc_a,
            desc_b,
            d_regs: d_regs.clone(),
            scale_d,
            imm_scale_a,
            imm_scale_b,
            trans_a,
            trans_b,
        });
        Ok(d_regs)
    }

    /// Convenience wrapper for `wgmma.mma_async` with F16 inputs (most common).
    ///
    /// Equivalent to calling [`wgmma_mma_async`] with `a_ty = b_ty = F16` and
    /// standard scale/transpose flags (`scale_d=1`, scales=1, no transpose).
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Hopper.
    ///
    /// [`wgmma_mma_async`]: BodyBuilder::wgmma_mma_async
    pub fn wgmma_mma_async_f16(
        &mut self,
        shape: WgmmaShape,
        desc_a: Register,
        desc_b: Register,
    ) -> Result<Vec<Register>, PtxGenError> {
        self.wgmma_mma_async(
            shape,
            PtxType::F16,
            PtxType::F16,
            desc_a,
            desc_b,
            1,
            1,
            1,
            0,
            0,
        )
    }

    /// Convenience wrapper for WGMMA with BF16 inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Hopper.
    pub fn wgmma_mma_async_bf16(
        &mut self,
        shape: WgmmaShape,
        desc_a: Register,
        desc_b: Register,
    ) -> Result<Vec<Register>, PtxGenError> {
        self.wgmma_mma_async(
            shape,
            PtxType::BF16,
            PtxType::BF16,
            desc_a,
            desc_b,
            1,
            1,
            1,
            0,
            0,
        )
    }

    /// Convenience wrapper for WGMMA with FP8 E4M3 inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Hopper.
    pub fn wgmma_mma_async_e4m3(
        &mut self,
        shape: WgmmaShape,
        desc_a: Register,
        desc_b: Register,
    ) -> Result<Vec<Register>, PtxGenError> {
        self.wgmma_mma_async(
            shape,
            PtxType::E4M3,
            PtxType::E4M3,
            desc_a,
            desc_b,
            1,
            1,
            1,
            0,
            0,
        )
    }

    /// Convenience wrapper for WGMMA with FP8 E5M2 inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the target SM is below Hopper.
    pub fn wgmma_mma_async_e5m2(
        &mut self,
        shape: WgmmaShape,
        desc_a: Register,
        desc_b: Register,
    ) -> Result<Vec<Register>, PtxGenError> {
        self.wgmma_mma_async(
            shape,
            PtxType::E5M2,
            PtxType::E5M2,
            desc_a,
            desc_b,
            1,
            1,
            1,
            0,
            0,
        )
    }
}

// ─── WGMMA accumulator register counts ───────────────────────────────────────

/// Number of F32 accumulator registers per thread for each WGMMA shape.
///
/// Derived from PTX ISA: `(M × N) / 128` where 128 = threads per warpgroup.
const fn wgmma_acc_regs(shape: WgmmaShape) -> u32 {
    match shape {
        WgmmaShape::M64N8K16 => 4,     // 64 × 8  / 128 = 4
        WgmmaShape::M64N16K16 => 8,    // 64 × 16 / 128 = 8
        WgmmaShape::M64N32K16 => 16,   // 64 × 32 / 128 = 16
        WgmmaShape::M64N64K16 => 32,   // 64 × 64 / 128 = 32
        WgmmaShape::M64N128K16 => 64,  // 64 × 128/ 128 = 64
        WgmmaShape::M64N256K16 => 128, // 64 × 256/ 128 = 128
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;
    use crate::builder::KernelBuilder;
    use crate::ir::{Operand, PtxType};

    /// Verify WMMA F32 accumulator load/mma/store round-trip compiles to valid IR.
    #[test]
    fn wmma_f16_f32_round_trip() {
        let ptx = KernelBuilder::new("wmma_test")
            .target(SmVersion::Sm75)
            .param("addr_a", PtxType::U64)
            .param("addr_b", PtxType::U64)
            .param("addr_c", PtxType::U64)
            .param("addr_d", PtxType::U64)
            .body(|b| {
                let addr_a = b.load_param_u64("addr_a");
                let addr_b = b.load_param_u64("addr_b");
                let addr_c = b.load_param_u64("addr_c");
                let addr_d = b.load_param_u64("addr_d");

                let a = b.wmma_load_a_f16(
                    WmmaShape::M16N16K16,
                    WmmaLayout::RowMajor,
                    Operand::Register(addr_a),
                    None,
                );
                let bv = b.wmma_load_b_f16(
                    WmmaShape::M16N16K16,
                    WmmaLayout::ColMajor,
                    Operand::Register(addr_b),
                    None,
                );
                // Allocate 8 F32 accumulators for the C matrix.
                let c = b.regs.alloc_group(PtxType::F32, 8);
                let d =
                    b.wmma_mma_sync_f16_f32(WmmaShape::M16N16K16, WmmaLayout::RowMajor, a, bv, c);
                b.wmma_store_d_f32(
                    WmmaShape::M16N16K16,
                    WmmaLayout::RowMajor,
                    Operand::Register(addr_d),
                    d,
                    None,
                );
                let _ = addr_c;
            })
            .build()
            .expect("wmma kernel should build");

        assert!(ptx.contains("wmma.load.a"), "wmma load A must appear");
        assert!(ptx.contains("wmma.load.b"), "wmma load B must appear");
        assert!(ptx.contains("wmma.mma.sync"), "wmma mma must appear");
        assert!(ptx.contains("wmma.store.d"), "wmma store D must appear");
    }

    /// Verify TF32 MMA emits correct PTX suffix.
    #[test]
    fn mma_tf32_requires_ampere() {
        let ptx = KernelBuilder::new("tf32_test")
            .target(SmVersion::Sm80)
            .param("unused", PtxType::U32)
            .body(|b| {
                let a = b.regs.alloc_group(PtxType::B32, 2);
                let bv = b.regs.alloc_group(PtxType::B32, 1);
                let c = b.regs.alloc_group(PtxType::F32, 4);
                let _ = b.mma_m16n8k8_tf32_f32(&a, &bv, &c).expect("TF32 on Ampere");
            })
            .build()
            .expect("build");

        assert!(ptx.contains("tf32"), "PTX must contain tf32 type");
    }

    /// Verify that WGMMA allocates correct accumulator count per shape.
    #[test]
    fn wgmma_acc_reg_counts() {
        assert_eq!(wgmma_acc_regs(WgmmaShape::M64N8K16), 4);
        assert_eq!(wgmma_acc_regs(WgmmaShape::M64N16K16), 8);
        assert_eq!(wgmma_acc_regs(WgmmaShape::M64N32K16), 16);
        assert_eq!(wgmma_acc_regs(WgmmaShape::M64N64K16), 32);
        assert_eq!(wgmma_acc_regs(WgmmaShape::M64N128K16), 64);
        assert_eq!(wgmma_acc_regs(WgmmaShape::M64N256K16), 128);
    }

    /// Verify WGMMA F16 emits correct structured PTX.
    #[test]
    fn wgmma_f16_emits_correct_ptx() {
        let ptx = KernelBuilder::new("wgmma_test")
            .target(SmVersion::Sm90)
            .param("unused", PtxType::U32)
            .body(|b| {
                let desc_a = b.regs.alloc(PtxType::U64);
                let desc_b = b.regs.alloc(PtxType::U64);
                let _ = b
                    .wgmma_mma_async_f16(WgmmaShape::M64N128K16, desc_a, desc_b)
                    .expect("wgmma on sm_90");
            })
            .build()
            .expect("build");

        assert!(ptx.contains("wgmma.mma_async"), "must contain wgmma");
        assert!(ptx.contains("m64n128k16"), "must contain shape");
    }

    /// Verify INT8 IMMA m8n8k16 emits correct type suffixes.
    #[test]
    fn mma_m8n8k16_s8_emits_s8() {
        let ptx = KernelBuilder::new("imma_test")
            .target(SmVersion::Sm75)
            .param("unused", PtxType::U32)
            .body(|b| {
                let a = b.regs.alloc_group(PtxType::S8, 1);
                let bv = b.regs.alloc_group(PtxType::S8, 1);
                let c = b.regs.alloc_group(PtxType::S32, 2);
                let _ = b
                    .mma_m8n8k16_s8_s32(&a, &bv, &c)
                    .expect("INT8 IMMA on Turing");
            })
            .build()
            .expect("build");

        assert!(ptx.contains("m8n8k16"), "must contain m8n8k16 shape");
        assert!(ptx.contains(".s8"), "must contain s8 type");
        assert!(ptx.contains(".s32"), "must contain s32 accumulator");
    }

    /// Verify BF16 MMA wrapper.
    #[test]
    fn mma_m16n8k16_bf16_ok() {
        let ptx = KernelBuilder::new("bf16_test")
            .target(SmVersion::Sm80)
            .param("unused", PtxType::U32)
            .body(|b| {
                let a = b.regs.alloc_group(PtxType::BF16, 4);
                let bv = b.regs.alloc_group(PtxType::BF16, 2);
                let c = b.regs.alloc_group(PtxType::F32, 4);
                let _ = b
                    .mma_m16n8k16_bf16_f32(&a, &bv, &c)
                    .expect("BF16 on Ampere");
            })
            .build()
            .expect("build");

        assert!(ptx.contains("bf16"), "must contain bf16 type");
    }
}
