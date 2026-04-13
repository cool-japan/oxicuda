//! Dead code elimination for PTX instruction sequences.
//!
//! This module implements a fixed-point dead code elimination (DCE) pass.
//! An instruction is *dead* if it defines a register that is never used by
//! any subsequent instruction **and** the instruction has no side effects.

use std::collections::HashSet;

use crate::ir::{Instruction, Operand, Register, WmmaOp};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Remove instructions whose results are never used.
///
/// Performs iterative dead code elimination until a fixed point is reached:
/// each pass removes instructions that define registers not consumed by any
/// other instruction, provided the instruction has no side effects.
///
/// # Arguments
///
/// * `instructions` - The original instruction sequence.
///
/// # Returns
///
/// A tuple of `(optimized_instructions, eliminated_count)`.
pub fn eliminate_dead_code(instructions: &[Instruction]) -> (Vec<Instruction>, usize) {
    let mut current: Vec<Instruction> = instructions.to_vec();
    let mut total_eliminated: usize = 0;

    loop {
        let (next, eliminated) = dce_pass(&current);
        if eliminated == 0 {
            break;
        }
        total_eliminated += eliminated;
        current = next;
    }

    (current, total_eliminated)
}

// ---------------------------------------------------------------------------
// Internal pass
// ---------------------------------------------------------------------------

/// Single pass of dead code elimination.
///
/// Returns `(surviving_instructions, number_eliminated)`.
fn dce_pass(instructions: &[Instruction]) -> (Vec<Instruction>, usize) {
    // Phase 1: Collect the set of all registers that are *used* by any instruction.
    let mut used_regs: HashSet<String> = HashSet::new();
    for inst in instructions {
        for reg in uses(inst) {
            used_regs.insert(reg.name.clone());
        }
    }

    // Phase 2: Mark each instruction as live or dead.
    let mut result = Vec::with_capacity(instructions.len());
    let mut eliminated: usize = 0;

    for inst in instructions {
        if has_side_effects(inst) {
            // Side-effecting instructions are always kept.
            result.push(inst.clone());
            continue;
        }

        let defined = defs(inst);
        if defined.is_empty() {
            // Instructions that define nothing and have no side effects
            // are kept (e.g., they shouldn't exist, but be conservative).
            result.push(inst.clone());
            continue;
        }

        // The instruction is dead if *none* of its defined registers are used.
        let any_def_used = defined.iter().any(|r| used_regs.contains(&r.name));

        if any_def_used {
            result.push(inst.clone());
        } else {
            eliminated += 1;
        }
    }

    (result, eliminated)
}

// ---------------------------------------------------------------------------
// Side-effect classification
// ---------------------------------------------------------------------------

/// Check if an instruction has side effects and therefore cannot be eliminated
/// even when its result register is unused.
///
/// Side-effecting instructions include memory stores, control flow, barriers,
/// fences, TMA operations, async copies, and meta-instructions (comments, raw).
const fn has_side_effects(inst: &Instruction) -> bool {
    match inst {
        // Memory stores, async copy, control flow, synchronization, fences,
        // TMA load, atomic operations, meta-instructions — all side-effecting.
        Instruction::Store { .. }
        | Instruction::CpAsync { .. }
        | Instruction::CpAsyncCommit
        | Instruction::CpAsyncWait { .. }
        | Instruction::Branch { .. }
        | Instruction::Label(_)
        | Instruction::Return
        | Instruction::BarSync { .. }
        | Instruction::BarArrive { .. }
        | Instruction::FenceAcqRel { .. }
        | Instruction::TmaLoad { .. }
        | Instruction::Atom { .. }
        | Instruction::AtomCas { .. }
        | Instruction::Red { .. }
        | Instruction::SurfStore { .. }
        | Instruction::Stmatrix { .. }
        | Instruction::Setmaxnreg { .. }
        | Instruction::Griddepcontrol { .. }
        | Instruction::FenceProxy { .. }
        | Instruction::MbarrierInit { .. }
        | Instruction::MbarrierArrive { .. }
        | Instruction::MbarrierWait { .. }
        | Instruction::Tcgen05Mma { .. }
        | Instruction::BarrierCluster
        | Instruction::FenceCluster
        | Instruction::CpAsyncBulk { .. }
        | Instruction::Comment(_)
        | Instruction::Raw(_) => true,

        // WMMA store operations write to memory
        Instruction::Wmma { op, .. } => matches!(op, WmmaOp::StoreD),

        // Pure computation instructions: no side effects
        Instruction::Add { .. }
        | Instruction::Sub { .. }
        | Instruction::Mul { .. }
        | Instruction::Mad { .. }
        | Instruction::Fma { .. }
        | Instruction::MadLo { .. }
        | Instruction::MadHi { .. }
        | Instruction::MadWide { .. }
        | Instruction::Neg { .. }
        | Instruction::Abs { .. }
        | Instruction::Min { .. }
        | Instruction::Max { .. }
        | Instruction::Brev { .. }
        | Instruction::Clz { .. }
        | Instruction::Popc { .. }
        | Instruction::Bfind { .. }
        | Instruction::Bfe { .. }
        | Instruction::Bfi { .. }
        | Instruction::Rcp { .. }
        | Instruction::Rsqrt { .. }
        | Instruction::Sqrt { .. }
        | Instruction::Ex2 { .. }
        | Instruction::Lg2 { .. }
        | Instruction::Sin { .. }
        | Instruction::Cos { .. }
        | Instruction::Shl { .. }
        | Instruction::Shr { .. }
        | Instruction::Div { .. }
        | Instruction::Rem { .. }
        | Instruction::And { .. }
        | Instruction::Or { .. }
        | Instruction::Xor { .. }
        | Instruction::SetP { .. }
        | Instruction::Load { .. }
        | Instruction::Cvt { .. }
        | Instruction::Mma { .. }
        | Instruction::Wgmma { .. }
        | Instruction::MovSpecial { .. }
        | Instruction::LoadParam { .. }
        | Instruction::Dp4a { .. }
        | Instruction::Dp2a { .. }
        | Instruction::Tex1d { .. }
        | Instruction::Tex2d { .. }
        | Instruction::Tex3d { .. }
        | Instruction::SurfLoad { .. }
        | Instruction::Redux { .. }
        | Instruction::ElectSync { .. }
        // Pragma is a directive hint — no side effects on execution
        | Instruction::Pragma(_)
        // ldmatrix: warp-cooperative load — result registers are output side, no mem-side effect
        | Instruction::Ldmatrix { .. } => false,
    }
}

// ---------------------------------------------------------------------------
// Register extraction helpers (mirrored from register_pressure)
// ---------------------------------------------------------------------------

/// Extract registers defined (written to) by an instruction.
fn defs(inst: &Instruction) -> Vec<&Register> {
    match inst {
        Instruction::Add { dst, .. }
        | Instruction::Sub { dst, .. }
        | Instruction::Mul { dst, .. }
        | Instruction::Mad { dst, .. }
        | Instruction::MadLo { dst, .. }
        | Instruction::MadHi { dst, .. }
        | Instruction::MadWide { dst, .. }
        | Instruction::Fma { dst, .. }
        | Instruction::Neg { dst, .. }
        | Instruction::Abs { dst, .. }
        | Instruction::Min { dst, .. }
        | Instruction::Max { dst, .. }
        | Instruction::Brev { dst, .. }
        | Instruction::Clz { dst, .. }
        | Instruction::Popc { dst, .. }
        | Instruction::Bfind { dst, .. }
        | Instruction::Bfe { dst, .. }
        | Instruction::Bfi { dst, .. }
        | Instruction::Rcp { dst, .. }
        | Instruction::Rsqrt { dst, .. }
        | Instruction::Sqrt { dst, .. }
        | Instruction::Ex2 { dst, .. }
        | Instruction::Lg2 { dst, .. }
        | Instruction::Sin { dst, .. }
        | Instruction::Cos { dst, .. }
        | Instruction::Shl { dst, .. }
        | Instruction::Shr { dst, .. }
        | Instruction::Div { dst, .. }
        | Instruction::Rem { dst, .. }
        | Instruction::And { dst, .. }
        | Instruction::Or { dst, .. }
        | Instruction::Xor { dst, .. }
        | Instruction::SetP { dst, .. }
        | Instruction::Load { dst, .. }
        | Instruction::Cvt { dst, .. }
        | Instruction::MovSpecial { dst, .. }
        | Instruction::LoadParam { dst, .. }
        | Instruction::Atom { dst, .. }
        | Instruction::AtomCas { dst, .. }
        | Instruction::Dp4a { dst, .. }
        | Instruction::Dp2a { dst, .. }
        | Instruction::Tex1d { dst, .. }
        | Instruction::Tex2d { dst, .. }
        | Instruction::Tex3d { dst, .. }
        | Instruction::SurfLoad { dst, .. }
        | Instruction::Redux { dst, .. }
        | Instruction::ElectSync { dst, .. } => vec![dst],

        Instruction::Ldmatrix { dst_regs, .. } => dst_regs.iter().collect(),

        Instruction::Store { .. }
        | Instruction::CpAsync { .. }
        | Instruction::CpAsyncCommit
        | Instruction::CpAsyncWait { .. }
        | Instruction::Branch { .. }
        | Instruction::Label(_)
        | Instruction::Return
        | Instruction::BarSync { .. }
        | Instruction::BarArrive { .. }
        | Instruction::FenceAcqRel { .. }
        | Instruction::TmaLoad { .. }
        | Instruction::Red { .. }
        | Instruction::SurfStore { .. }
        | Instruction::Stmatrix { .. }
        | Instruction::Setmaxnreg { .. }
        | Instruction::Griddepcontrol { .. }
        | Instruction::FenceProxy { .. }
        | Instruction::MbarrierInit { .. }
        | Instruction::MbarrierArrive { .. }
        | Instruction::MbarrierWait { .. }
        | Instruction::Tcgen05Mma { .. }
        | Instruction::BarrierCluster
        | Instruction::FenceCluster
        | Instruction::CpAsyncBulk { .. }
        | Instruction::Comment(_)
        | Instruction::Raw(_)
        | Instruction::Pragma(_) => vec![],

        Instruction::Wmma { op, fragments, .. } => match op {
            WmmaOp::LoadA | WmmaOp::LoadB | WmmaOp::Mma => fragments.iter().collect(),
            WmmaOp::StoreD => vec![],
        },
        Instruction::Mma { d_regs, .. } | Instruction::Wgmma { d_regs, .. } => {
            d_regs.iter().collect()
        }
    }
}

/// Extract registers used (read from) by an instruction.
#[allow(clippy::too_many_lines)]
fn uses(inst: &Instruction) -> Vec<&Register> {
    match inst {
        Instruction::Add { a, b, .. }
        | Instruction::Sub { a, b, .. }
        | Instruction::Mul { a, b, .. }
        | Instruction::Min { a, b, .. }
        | Instruction::Max { a, b, .. }
        | Instruction::Div { a, b, .. }
        | Instruction::Rem { a, b, .. }
        | Instruction::And { a, b, .. }
        | Instruction::Or { a, b, .. }
        | Instruction::Xor { a, b, .. }
        | Instruction::SetP { a, b, .. }
        | Instruction::Shl {
            src: a, amount: b, ..
        }
        | Instruction::Shr {
            src: a, amount: b, ..
        } => {
            let mut regs = operand_regs(a);
            regs.extend(operand_regs(b));
            regs
        }

        Instruction::Mad { a, b, c, .. }
        | Instruction::MadLo { a, b, c, .. }
        | Instruction::MadHi { a, b, c, .. }
        | Instruction::MadWide { a, b, c, .. }
        | Instruction::Fma { a, b, c, .. }
        | Instruction::Dp4a { a, b, c, .. }
        | Instruction::Dp2a { a, b, c, .. } => {
            let mut regs = operand_regs(a);
            regs.extend(operand_regs(b));
            regs.extend(operand_regs(c));
            regs
        }

        Instruction::Neg { src, .. }
        | Instruction::Abs { src, .. }
        | Instruction::Brev { src, .. }
        | Instruction::Clz { src, .. }
        | Instruction::Popc { src, .. }
        | Instruction::Bfind { src, .. }
        | Instruction::Rcp { src, .. }
        | Instruction::Rsqrt { src, .. }
        | Instruction::Sqrt { src, .. }
        | Instruction::Ex2 { src, .. }
        | Instruction::Lg2 { src, .. }
        | Instruction::Sin { src, .. }
        | Instruction::Cos { src, .. }
        | Instruction::Cvt { src, .. }
        | Instruction::Redux { src, .. } => operand_regs(src),

        Instruction::Bfe {
            src, start, len, ..
        } => {
            let mut regs = operand_regs(src);
            regs.extend(operand_regs(start));
            regs.extend(operand_regs(len));
            regs
        }

        Instruction::Bfi {
            insert,
            base,
            start,
            len,
            ..
        } => {
            let mut regs = operand_regs(insert);
            regs.extend(operand_regs(base));
            regs.extend(operand_regs(start));
            regs.extend(operand_regs(len));
            regs
        }

        Instruction::Load { addr, .. } | Instruction::MbarrierArrive { addr } => operand_regs(addr),

        Instruction::Store { addr, src, .. } => {
            let mut regs = operand_regs(addr);
            regs.push(src);
            regs
        }

        Instruction::CpAsync {
            dst_shared,
            src_global,
            ..
        } => {
            let mut regs = operand_regs(dst_shared);
            regs.extend(operand_regs(src_global));
            regs
        }

        Instruction::CpAsyncCommit
        | Instruction::CpAsyncWait { .. }
        | Instruction::Label(_)
        | Instruction::Return
        | Instruction::BarSync { .. }
        | Instruction::BarArrive { .. }
        | Instruction::FenceAcqRel { .. }
        | Instruction::MovSpecial { .. }
        | Instruction::LoadParam { .. }
        | Instruction::ElectSync { .. }
        | Instruction::Setmaxnreg { .. }
        | Instruction::Griddepcontrol { .. }
        | Instruction::FenceProxy { .. }
        | Instruction::BarrierCluster
        | Instruction::FenceCluster
        | Instruction::Comment(_)
        | Instruction::Raw(_)
        | Instruction::Pragma(_) => vec![],

        Instruction::Branch { predicate, .. } => {
            if let Some((reg, _)) = predicate {
                vec![reg]
            } else {
                vec![]
            }
        }

        Instruction::Wmma {
            op,
            fragments,
            addr,
            stride,
            ..
        } => {
            let mut regs: Vec<&Register> = Vec::new();
            match op {
                WmmaOp::LoadA | WmmaOp::LoadB => {
                    if let Some(a) = addr {
                        regs.extend(operand_regs(a));
                    }
                    if let Some(s) = stride {
                        regs.extend(operand_regs(s));
                    }
                }
                WmmaOp::StoreD => {
                    regs.extend(fragments.iter());
                    if let Some(a) = addr {
                        regs.extend(operand_regs(a));
                    }
                    if let Some(s) = stride {
                        regs.extend(operand_regs(s));
                    }
                }
                WmmaOp::Mma => {
                    regs.extend(fragments.iter());
                }
            }
            regs
        }

        Instruction::Mma {
            a_regs,
            b_regs,
            c_regs,
            ..
        } => {
            let mut regs: Vec<&Register> = Vec::new();
            regs.extend(a_regs.iter());
            regs.extend(b_regs.iter());
            regs.extend(c_regs.iter());
            regs
        }

        Instruction::Wgmma { desc_a, desc_b, .. } => vec![desc_a, desc_b],

        Instruction::TmaLoad {
            dst_shared,
            desc,
            coords,
            barrier,
            ..
        } => {
            let mut regs = operand_regs(dst_shared);
            regs.push(desc);
            regs.extend(coords.iter());
            regs.push(barrier);
            regs
        }

        // Atomic: reads addr and src
        Instruction::Atom { addr, src, .. } | Instruction::Red { addr, src, .. } => {
            let mut regs = operand_regs(addr);
            regs.extend(operand_regs(src));
            regs
        }
        // AtomCas: reads addr, compare, and value
        Instruction::AtomCas {
            addr,
            compare,
            value,
            ..
        } => {
            let mut regs = operand_regs(addr);
            regs.extend(operand_regs(compare));
            regs.extend(operand_regs(value));
            regs
        }

        // Texture: coord registers are used
        Instruction::Tex1d { coord, .. } | Instruction::SurfLoad { coord, .. } => {
            operand_regs(coord)
        }
        Instruction::Tex2d {
            coord_x, coord_y, ..
        } => {
            let mut regs = operand_regs(coord_x);
            regs.extend(operand_regs(coord_y));
            regs
        }
        Instruction::Tex3d {
            coord_x,
            coord_y,
            coord_z,
            ..
        } => {
            let mut regs = operand_regs(coord_x);
            regs.extend(operand_regs(coord_y));
            regs.extend(operand_regs(coord_z));
            regs
        }
        Instruction::SurfStore { coord, src, .. } => {
            let mut regs = operand_regs(coord);
            regs.push(src);
            regs
        }

        // PTX 8.x instructions
        Instruction::Stmatrix { dst_addr, src, .. } => {
            let mut regs = operand_regs(dst_addr);
            regs.push(src);
            regs
        }
        Instruction::MbarrierInit { addr, count, .. } => {
            let mut regs = operand_regs(addr);
            regs.extend(operand_regs(count));
            regs
        }
        Instruction::MbarrierWait { addr, phase } => {
            let mut regs = operand_regs(addr);
            regs.extend(operand_regs(phase));
            regs
        }

        Instruction::Tcgen05Mma { a_desc, b_desc } => vec![a_desc, b_desc],

        Instruction::CpAsyncBulk {
            dst_smem,
            src_gmem,
            desc,
        } => vec![dst_smem, src_gmem, desc],

        Instruction::Ldmatrix { src_addr, .. } => operand_regs(src_addr),
    }
}

/// Extract register references from an operand.
fn operand_regs(op: &Operand) -> Vec<&Register> {
    match op {
        Operand::Register(reg) => vec![reg],
        Operand::Address { base, .. } => vec![base],
        Operand::Immediate(_) | Operand::Symbol(_) => vec![],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        CacheQualifier, FenceScope, ImmValue, Instruction, MemorySpace, MulMode, Operand, PtxType,
        Register, SpecialReg, VectorWidth, WmmaOp,
    };

    fn reg(name: &str, ty: PtxType) -> Register {
        Register {
            name: name.to_string(),
            ty,
        }
    }

    fn reg_op(name: &str, ty: PtxType) -> Operand {
        Operand::Register(reg(name, ty))
    }

    fn imm_u32(val: u32) -> Operand {
        Operand::Immediate(ImmValue::U32(val))
    }

    /// Unused register definition should be removed.
    #[test]
    fn test_unused_register_removed() {
        let instructions = vec![
            Instruction::Add {
                ty: PtxType::F32,
                dst: reg("%f0", PtxType::F32),
                a: imm_u32(1),
                b: imm_u32(2),
            },
            // %f0 is never used
        ];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 1);
        assert!(result.is_empty());
    }

    /// Used register definition should be kept.
    #[test]
    fn test_used_register_kept() {
        let instructions = vec![
            Instruction::MovSpecial {
                dst: reg("%r0", PtxType::U32),
                special: SpecialReg::TidX,
            },
            Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: None,
                },
                src: reg("%r0", PtxType::U32),
            },
        ];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 2);
    }

    /// Store instructions are never removed (side effect).
    #[test]
    fn test_stores_never_removed() {
        let instructions = vec![Instruction::Store {
            space: MemorySpace::Global,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            addr: Operand::Address {
                base: reg("%rd0", PtxType::U64),
                offset: None,
            },
            src: reg("%f0", PtxType::F32),
        }];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 1);
    }

    /// Branches are never removed (control flow side effect).
    #[test]
    fn test_branches_never_removed() {
        let instructions = vec![
            Instruction::Branch {
                target: "loop".to_string(),
                predicate: None,
            },
            Instruction::Label("loop".to_string()),
        ];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 2);
    }

    /// Barrier is never removed (synchronization side effect).
    #[test]
    fn test_barrier_never_removed() {
        let instructions = vec![Instruction::BarSync { id: 0 }];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 1);
    }

    /// `BarArrive` is never removed.
    #[test]
    fn test_bar_arrive_never_removed() {
        let instructions = vec![Instruction::BarArrive { id: 0, count: 32 }];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 1);
    }

    /// `FenceAcqRel` is never removed.
    #[test]
    fn test_fence_never_removed() {
        let instructions = vec![Instruction::FenceAcqRel {
            scope: FenceScope::Gpu,
        }];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 1);
    }

    /// Return is never removed.
    #[test]
    fn test_return_never_removed() {
        let instructions = vec![Instruction::Return];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 1);
    }

    /// Comment is never removed.
    #[test]
    fn test_comment_never_removed() {
        let instructions = vec![Instruction::Comment("keep me".to_string())];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 1);
    }

    /// Raw PTX is never removed.
    #[test]
    fn test_raw_never_removed() {
        let instructions = vec![Instruction::Raw("nop;".to_string())];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 1);
    }

    /// Chain of dead instructions: A defines X, B uses X to define Y, Y unused.
    /// Both should be eliminated (fixed-point iteration).
    #[test]
    fn test_chain_of_dead_instructions() {
        let instructions = vec![
            // %f0 = 1 + 2 (dead because only used by next instruction)
            Instruction::Add {
                ty: PtxType::F32,
                dst: reg("%f0", PtxType::F32),
                a: imm_u32(1),
                b: imm_u32(2),
            },
            // %f1 = %f0 + 3 (dead because %f1 is never used)
            Instruction::Add {
                ty: PtxType::F32,
                dst: reg("%f1", PtxType::F32),
                a: reg_op("%f0", PtxType::F32),
                b: imm_u32(3),
            },
        ];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        // First pass: %f1 unused → remove second instruction → eliminated=1
        // Second pass: %f0 now unused → remove first instruction → eliminated=1
        // Total: 2
        assert_eq!(eliminated, 2);
        assert!(result.is_empty());
    }

    /// Fixed-point: three-level chain of dead code.
    #[test]
    fn test_three_level_dead_chain() {
        let instructions = vec![
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r0", PtxType::U32),
                a: imm_u32(1),
                b: imm_u32(2),
            },
            Instruction::Mul {
                ty: PtxType::U32,
                mode: MulMode::Lo,
                dst: reg("%r1", PtxType::U32),
                a: reg_op("%r0", PtxType::U32),
                b: imm_u32(3),
            },
            Instruction::Sub {
                ty: PtxType::U32,
                dst: reg("%r2", PtxType::U32),
                a: reg_op("%r1", PtxType::U32),
                b: imm_u32(4),
            },
        ];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 3);
        assert!(result.is_empty());
    }

    /// Function with no dead code returns identical instructions.
    #[test]
    fn test_no_dead_code_unchanged() {
        let instructions = vec![
            Instruction::MovSpecial {
                dst: reg("%r0", PtxType::U32),
                special: SpecialReg::TidX,
            },
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r1", PtxType::U32),
                a: reg_op("%r0", PtxType::U32),
                b: imm_u32(1),
            },
            Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: None,
                },
                src: reg("%r1", PtxType::U32),
            },
        ];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 3);
    }

    /// `CpAsync` is never removed (DMA side effect).
    #[test]
    fn test_cp_async_never_removed() {
        let instructions = vec![
            Instruction::CpAsync {
                bytes: 16,
                dst_shared: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: None,
                },
                src_global: Operand::Address {
                    base: reg("%rd1", PtxType::U64),
                    offset: None,
                },
            },
            Instruction::CpAsyncCommit,
            Instruction::CpAsyncWait { n: 0 },
        ];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 3);
    }

    /// `TmaLoad` is never removed.
    #[test]
    fn test_tma_load_never_removed() {
        let instructions = vec![Instruction::TmaLoad {
            dst_shared: Operand::Address {
                base: reg("%rd0", PtxType::U64),
                offset: None,
            },
            desc: reg("%rd1", PtxType::U64),
            coords: vec![reg("%r0", PtxType::U32)],
            barrier: reg("%rd2", PtxType::U64),
        }];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 1);
    }

    /// Mixed live and dead instructions.
    #[test]
    fn test_mixed_live_and_dead() {
        let instructions = vec![
            // Live chain: tid → add → store
            Instruction::MovSpecial {
                dst: reg("%r0", PtxType::U32),
                special: SpecialReg::TidX,
            },
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r1", PtxType::U32),
                a: reg_op("%r0", PtxType::U32),
                b: imm_u32(1),
            },
            // Dead: %r2 is never used
            Instruction::Mul {
                ty: PtxType::U32,
                mode: MulMode::Lo,
                dst: reg("%r2", PtxType::U32),
                a: reg_op("%r0", PtxType::U32),
                b: imm_u32(2),
            },
            Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: None,
                },
                src: reg("%r1", PtxType::U32),
            },
        ];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 1);
        assert_eq!(result.len(), 3);
    }

    /// Empty input returns empty output with zero eliminated.
    #[test]
    fn test_empty_instructions() {
        let (result, eliminated) = eliminate_dead_code(&[]);
        assert_eq!(eliminated, 0);
        assert!(result.is_empty());
    }

    /// Load instruction without subsequent use is dead (loads have no side
    /// effect on other memory; they only write to the destination register).
    #[test]
    fn test_dead_load_removed() {
        let instructions = vec![Instruction::Load {
            space: MemorySpace::Global,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            addr: Operand::Address {
                base: reg("%rd0", PtxType::U64),
                offset: None,
            },
        }];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 1);
        assert!(result.is_empty());
    }

    /// `Wmma` `StoreD` is a side effect and must not be removed.
    #[test]
    fn test_wmma_store_never_removed() {
        use crate::ir::{WmmaLayout, WmmaShape};

        let instructions = vec![Instruction::Wmma {
            op: WmmaOp::StoreD,
            shape: WmmaShape::M16N16K16,
            layout: WmmaLayout::RowMajor,
            ty: PtxType::F16,
            fragments: vec![reg("%f0", PtxType::F16), reg("%f1", PtxType::F16)],
            addr: Some(Operand::Address {
                base: reg("%rd0", PtxType::U64),
                offset: None,
            }),
            stride: None,
        }];
        let (result, eliminated) = eliminate_dead_code(&instructions);
        assert_eq!(eliminated, 0);
        assert_eq!(result.len(), 1);
    }

    /// `has_side_effects` correctly classifies all instruction categories.
    #[test]
    fn test_side_effects_classification() {
        // Pure computation → no side effects
        let add = Instruction::Add {
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            a: imm_u32(0),
            b: imm_u32(0),
        };
        assert!(!has_side_effects(&add));

        // Store → side effect
        let store = Instruction::Store {
            space: MemorySpace::Global,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            addr: Operand::Address {
                base: reg("%rd0", PtxType::U64),
                offset: None,
            },
            src: reg("%f0", PtxType::F32),
        };
        assert!(has_side_effects(&store));

        // Branch → side effect
        let branch = Instruction::Branch {
            target: "L1".to_string(),
            predicate: None,
        };
        assert!(has_side_effects(&branch));

        // Label → side effect
        let label = Instruction::Label("L1".to_string());
        assert!(has_side_effects(&label));

        // BarSync → side effect
        let bar = Instruction::BarSync { id: 0 };
        assert!(has_side_effects(&bar));

        // MovSpecial → no side effect (can be DCE'd if unused)
        let mov = Instruction::MovSpecial {
            dst: reg("%r0", PtxType::U32),
            special: SpecialReg::TidX,
        };
        assert!(!has_side_effects(&mov));
    }

    // -------------------------------------------------------------------------
    // Additional DCE quality-gate tests
    // -------------------------------------------------------------------------

    /// An unreachable chain of pure computations after a branch is dead:
    /// the defined registers are never consumed, so DCE eliminates them.
    /// Note: the Branch and Label are kept (side effects); the pure
    /// computations whose results feed nowhere are eliminated.
    #[test]
    fn test_dce_removes_unreachable_block() {
        let instructions = vec![
            // Unconditional branch — side effect, kept
            Instruction::Branch {
                target: "after_dead".to_string(),
                predicate: None,
            },
            // The following pure computations are never consumed (the branch
            // causes them to be skipped at runtime, and their outputs are
            // never used anywhere in this list).
            Instruction::Add {
                ty: PtxType::F32,
                dst: reg("%f_dead0", PtxType::F32),
                a: imm_u32(1),
                b: imm_u32(2),
            },
            Instruction::Mul {
                ty: PtxType::F32,
                mode: MulMode::Lo,
                dst: reg("%f_dead1", PtxType::F32),
                a: reg_op("%f_dead0", PtxType::F32),
                b: imm_u32(3),
            },
            // Label — side effect, kept
            Instruction::Label("after_dead".to_string()),
            Instruction::Return,
        ];

        let (result, eliminated) = eliminate_dead_code(&instructions);

        // Branch, Label, Return are always kept (side effects = 3)
        // Add and Mul are dead (their outputs go nowhere) = 2 eliminated
        // Fixed-point: pass1 eliminates Mul (uses %f_dead0 which is defined but
        // then %f_dead0 has no other consumer after Mul is gone), pass2 eliminates Add
        assert_eq!(
            eliminated, 2,
            "DCE must eliminate both unreachable pure-computation instructions"
        );
        // Branch, Label, Return survive
        assert_eq!(
            result.len(),
            3,
            "Branch, Label and Return must be preserved"
        );
    }

    /// Reachable blocks (pure computations whose results feed a store) must NOT
    /// be removed by DCE.
    #[test]
    fn test_dce_keeps_reachable_blocks() {
        let instructions = vec![
            Instruction::MovSpecial {
                dst: reg("%r0", PtxType::U32),
                special: SpecialReg::TidX,
            },
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r1", PtxType::U32),
                a: reg_op("%r0", PtxType::U32),
                b: imm_u32(10),
            },
            Instruction::Mul {
                ty: PtxType::U32,
                mode: MulMode::Lo,
                dst: reg("%r2", PtxType::U32),
                a: reg_op("%r1", PtxType::U32),
                b: imm_u32(4),
            },
            // Store consumes %r2 — the whole chain is live
            Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: None,
                },
                src: reg("%r2", PtxType::U32),
            },
        ];

        let (result, eliminated) = eliminate_dead_code(&instructions);

        assert_eq!(
            eliminated, 0,
            "no instruction should be eliminated from a fully-live chain"
        );
        assert_eq!(
            result.len(),
            instructions.len(),
            "all instructions must survive DCE"
        );
    }

    /// DCE must be idempotent: running the pass twice on the same input must
    /// produce the same result as running it once.
    #[test]
    fn test_dce_idempotent() {
        let instructions = vec![
            // Live chain
            Instruction::MovSpecial {
                dst: reg("%r0", PtxType::U32),
                special: SpecialReg::TidX,
            },
            // Dead computation
            Instruction::Add {
                ty: PtxType::F32,
                dst: reg("%f_unused", PtxType::F32),
                a: imm_u32(7),
                b: imm_u32(8),
            },
            Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: None,
                },
                src: reg("%r0", PtxType::U32),
            },
        ];

        let (first_result, first_eliminated) = eliminate_dead_code(&instructions);
        // DCE already runs to fixed-point internally, so a second call changes nothing
        let (second_result, second_eliminated) = eliminate_dead_code(&first_result);

        assert_eq!(
            second_eliminated, 0,
            "second DCE pass must not eliminate anything additional (idempotent)"
        );
        assert_eq!(
            first_result.len(),
            second_result.len(),
            "result length must be the same on both passes"
        );
        assert_eq!(
            first_eliminated, 1,
            "first pass must eliminate the unused Add instruction"
        );
    }
}
