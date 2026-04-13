//! Register pressure analysis for PTX instruction sequences.
//!
//! This module performs live register analysis on a linear sequence of PTX
//! instructions, computing peak register usage per type and total, detecting
//! spill risk, and estimating occupancy impact.

use std::collections::HashMap;

use crate::ir::{Instruction, Operand, PtxType, Register, WmmaOp};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of register pressure analysis for a function.
#[derive(Debug, Clone)]
pub struct RegisterPressureReport {
    /// Maximum number of simultaneously live registers, by type.
    pub peak_by_type: HashMap<PtxType, usize>,
    /// Total peak register count (all types combined).
    pub total_peak: usize,
    /// Per-instruction live register counts.
    pub live_at_instruction: Vec<usize>,
    /// Whether the function is at risk of register spilling (> 255 regs).
    pub spill_risk: bool,
    /// Estimated maximum warps per SM based on register usage per thread.
    ///
    /// Calculated for `sm_80`: 65536 registers per SM, 32 threads per warp.
    /// Formula: `max_warps = 65536 / (32 * regs_per_thread)`.
    /// Returns `None` if the function uses zero registers.
    pub estimated_max_warps_per_sm: Option<u32>,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of registers per thread before spilling occurs on most
/// NVIDIA architectures.
const SPILL_THRESHOLD: usize = 255;

/// Total number of 32-bit registers per SM on `sm_80` (Ampere).
const SM80_REGS_PER_SM: u32 = 65536;

/// Number of threads per warp.
const THREADS_PER_WARP: u32 = 32;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Analyze register pressure for a sequence of instructions.
///
/// This performs a linear liveness analysis:
/// 1. For each register, record its first definition and last use.
/// 2. A register is *live* at instruction `i` if `def(r) <= i <= last_use(r)`.
/// 3. Peak is the maximum number of simultaneously live registers.
///
/// # Arguments
///
/// * `instructions` - The linear sequence of PTX instructions to analyze.
///
/// # Returns
///
/// A [`RegisterPressureReport`] with peak counts, per-instruction liveness,
/// spill risk assessment, and occupancy estimation.
pub fn analyze_register_pressure(instructions: &[Instruction]) -> RegisterPressureReport {
    if instructions.is_empty() {
        return RegisterPressureReport {
            peak_by_type: HashMap::new(),
            total_peak: 0,
            live_at_instruction: Vec::new(),
            spill_risk: false,
            estimated_max_warps_per_sm: None,
        };
    }

    // Phase 1: Collect def and last-use positions for each register.
    let mut first_def: HashMap<String, usize> = HashMap::new();
    let mut last_use: HashMap<String, usize> = HashMap::new();
    let mut reg_types: HashMap<String, PtxType> = HashMap::new();

    for (idx, inst) in instructions.iter().enumerate() {
        for reg in defs(inst) {
            first_def.entry(reg.name.clone()).or_insert(idx);
            reg_types.entry(reg.name.clone()).or_insert(reg.ty);
        }
        for reg in uses(inst) {
            last_use.insert(reg.name.clone(), idx);
            reg_types.entry(reg.name.clone()).or_insert(reg.ty);
        }
    }

    // Registers that are defined but never explicitly used are still live at
    // their definition point (they occupy a register slot for at least one cycle).
    // Set their last_use to their definition point if not already present.
    for (name, def_idx) in &first_def {
        last_use.entry(name.clone()).or_insert(*def_idx);
    }

    // Phase 2: For each instruction, compute the set of live registers.
    let num_instructions = instructions.len();
    let mut live_at_instruction = Vec::with_capacity(num_instructions);
    let mut peak_by_type: HashMap<PtxType, usize> = HashMap::new();
    let mut total_peak: usize = 0;

    for i in 0..num_instructions {
        let mut live_count: usize = 0;
        let mut type_counts: HashMap<PtxType, usize> = HashMap::new();

        for (name, def_idx) in &first_def {
            let use_idx = last_use.get(name).copied().unwrap_or(*def_idx);
            if *def_idx <= i && i <= use_idx {
                live_count += 1;
                if let Some(ty) = reg_types.get(name) {
                    *type_counts.entry(*ty).or_insert(0) += 1;
                }
            }
        }

        live_at_instruction.push(live_count);

        if live_count > total_peak {
            total_peak = live_count;
        }

        for (ty, count) in &type_counts {
            let current = peak_by_type.entry(*ty).or_insert(0);
            if *count > *current {
                *current = *count;
            }
        }
    }

    let spill_risk = total_peak > SPILL_THRESHOLD;

    let estimated_max_warps_per_sm = if total_peak == 0 {
        None
    } else {
        let peak_u32 = u32::try_from(total_peak).unwrap_or(u32::MAX);
        let regs_per_warp = THREADS_PER_WARP.saturating_mul(peak_u32);
        SM80_REGS_PER_SM.checked_div(regs_per_warp)
    };

    RegisterPressureReport {
        peak_by_type,
        total_peak,
        live_at_instruction,
        spill_risk,
        estimated_max_warps_per_sm,
    }
}

// ---------------------------------------------------------------------------
// Register extraction helpers
// ---------------------------------------------------------------------------

/// Extract registers defined (written to) by an instruction.
fn defs(inst: &Instruction) -> Vec<&Register> {
    match inst {
        // Arithmetic with dst, Comparison, Load, Cvt, MovSpecial, LoadParam, Atomics
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

        // ldmatrix: defines multiple destination registers
        Instruction::Ldmatrix { dst_regs, .. } => dst_regs.iter().collect(),

        // No register defs
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
        | Instruction::Comment(_)
        | Instruction::Raw(_)
        | Instruction::Pragma(_)
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
        | Instruction::CpAsyncBulk { .. } => vec![],

        // Tensor Core
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
        // Two source operands (arithmetic, comparison, min/max)
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

        // Three source operands
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

        // Load / MbarrierArrive reads the address
        Instruction::Load { addr, .. } | Instruction::MbarrierArrive { addr, .. } => {
            operand_regs(addr)
        }

        // Store reads the address and source register
        Instruction::Store { addr, src, .. } => {
            let mut regs = operand_regs(addr);
            regs.push(src);
            regs
        }

        // Async copy reads source and destination addresses
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

        // Branch may use a predicate register
        Instruction::Branch { predicate, .. } => {
            if let Some((reg, _negated)) = predicate {
                vec![reg]
            } else {
                vec![]
            }
        }

        // Tensor Core
        Instruction::Wmma {
            op,
            fragments,
            addr,
            stride,
            ..
        } => {
            let mut regs: Vec<&Register> = Vec::new();
            match op {
                // Load: reads address and stride, fragments are destinations
                WmmaOp::LoadA | WmmaOp::LoadB => {
                    if let Some(a) = addr {
                        regs.extend(operand_regs(a));
                    }
                    if let Some(s) = stride {
                        regs.extend(operand_regs(s));
                    }
                }
                // Store: reads fragments and address/stride
                WmmaOp::StoreD => {
                    regs.extend(fragments.iter());
                    if let Some(a) = addr {
                        regs.extend(operand_regs(a));
                    }
                    if let Some(s) = stride {
                        regs.extend(operand_regs(s));
                    }
                }
                // Mma: conservative — all fragments used as inputs
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

        // PTX 8.x instructions.
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
        Instruction::MbarrierWait { addr, phase, .. } => {
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

        // ldmatrix: reads the source address register
        Instruction::Ldmatrix { src_addr, .. } => operand_regs(src_addr),
    }
}

/// Extract register references from an operand.
///
/// Returns a vector containing a reference to the register if the operand is a
/// `Register` or `Address` variant, otherwise an empty vector.
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
        CacheQualifier, CmpOp, ImmValue, Instruction, MemorySpace, MulMode, Operand, PtxType,
        Register, SpecialReg, VectorWidth,
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

    /// Empty instruction sequence produces zero-pressure report.
    #[test]
    fn test_empty_instructions() {
        let report = analyze_register_pressure(&[]);
        assert_eq!(report.total_peak, 0);
        assert!(report.live_at_instruction.is_empty());
        assert!(!report.spill_risk);
        assert!(report.estimated_max_warps_per_sm.is_none());
    }

    /// Single add instruction: 3 registers live at that point (dst, a, b if regs).
    #[test]
    fn test_single_add() {
        let instructions = vec![Instruction::Add {
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            a: reg_op("%f1", PtxType::F32),
            b: reg_op("%f2", PtxType::F32),
        }];
        let report = analyze_register_pressure(&instructions);
        // %f0 defined at 0, last_use at 0 (only defined, never used later)
        // %f1 used at 0, not defined → not in first_def
        // %f2 used at 0, not defined → not in first_def
        // Only %f0 is in first_def, so live_at_instruction = [1]
        // But %f1 and %f2 are inputs (used but never defined) → they are not
        // tracked in first_def. This is a limitation of linear analysis.
        // The analysis counts registers that have a definition.
        assert_eq!(report.live_at_instruction.len(), 1);
        assert_eq!(report.total_peak, 1);
    }

    /// Linear sequence: define registers then use them, checking peak.
    #[test]
    fn test_sequence_peak_pressure() {
        // Instruction 0: %f0 = special(TidX)
        // Instruction 1: %f1 = cvt(%f0) -- %f0 alive, %f1 born
        // Instruction 2: %f2 = add(%f1, imm) -- %f1 alive, %f2 born, %f0 dead
        // Instruction 3: store %f2 -- %f2 alive
        let instructions = vec![
            Instruction::MovSpecial {
                dst: reg("%r0", PtxType::U32),
                special: SpecialReg::TidX,
            },
            Instruction::Cvt {
                rnd: None,
                dst_ty: PtxType::F32,
                src_ty: PtxType::U32,
                dst: reg("%f0", PtxType::F32),
                src: reg_op("%r0", PtxType::U32),
            },
            Instruction::Add {
                ty: PtxType::F32,
                dst: reg("%f1", PtxType::F32),
                a: reg_op("%f0", PtxType::F32),
                b: imm_u32(1),
            },
            Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::F32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: None,
                },
                src: reg("%f1", PtxType::F32),
            },
        ];
        let report = analyze_register_pressure(&instructions);
        // %r0: def=0, last_use=1 → live at [0,1]
        // %f0: def=1, last_use=2 → live at [1,2]
        // %f1: def=2, last_use=3 → live at [2,3]
        // Instruction 0: {%r0} → 1
        // Instruction 1: {%r0, %f0} → 2
        // Instruction 2: {%f0, %f1} → 2
        // Instruction 3: {%f1} → 1
        assert_eq!(report.live_at_instruction, vec![1, 2, 2, 1]);
        assert_eq!(report.total_peak, 2);
    }

    /// Register reuse reduces peak: defining to same name kills old liveness.
    #[test]
    fn test_register_reuse_reduces_peak() {
        // %f0 defined at 0, used at 1
        // %f0 redefined at 1, used at 2
        // Since we track by name, first_def stays at 0, last_use moves to 2
        // So %f0 is live across all 3 instructions
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
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r2", PtxType::U32),
                a: reg_op("%r1", PtxType::U32),
                b: imm_u32(2),
            },
        ];
        let report = analyze_register_pressure(&instructions);
        // %r0: def=0, last_use=1 → live [0,1]
        // %r1: def=1, last_use=2 → live [1,2]
        // %r2: def=2, last_use=2 → live [2]
        // Inst 0: {%r0} → 1
        // Inst 1: {%r0, %r1} → 2
        // Inst 2: {%r1, %r2} → 2
        assert_eq!(report.total_peak, 2);
        assert_eq!(report.live_at_instruction, vec![1, 2, 2]);
    }

    /// Non-overlapping lifetimes keep peak low.
    #[test]
    fn test_non_overlapping_lifetimes() {
        // %r0: def=0, used=1
        // %r1: def=2, used=3
        // These don't overlap → peak = 1
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
            Instruction::MovSpecial {
                dst: reg("%r1", PtxType::U32),
                special: SpecialReg::TidY,
            },
            Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd1", PtxType::U64),
                    offset: None,
                },
                src: reg("%r1", PtxType::U32),
            },
        ];
        let report = analyze_register_pressure(&instructions);
        // %r0: def=0, last_use=1 → live [0,1]
        // %r1: def=2, last_use=3 → live [2,3]
        // Inst 0: {%r0} → 1, Inst 1: {%r0} → 1, Inst 2: {%r1} → 1, Inst 3: {%r1} → 1
        assert_eq!(report.total_peak, 1);
    }

    /// Spill risk is flagged for high register usage.
    #[test]
    fn test_spill_risk_detection() {
        // Create 256 registers all alive at the same point
        let mut instructions = Vec::new();
        for i in 0..256 {
            instructions.push(Instruction::MovSpecial {
                dst: reg(&format!("%r{i}"), PtxType::U32),
                special: SpecialReg::TidX,
            });
        }
        // Use all of them in one instruction (we just need them all to have
        // overlapping liveness). Use a raw store with last-defined register
        // to keep all others alive.
        // Actually, registers defined but never used are only live at their def point.
        // For them to all be live simultaneously, they must all be used after the
        // last definition.
        // Add a comment to keep them alive? No, we need actual uses.
        // Let's use a different approach: one big Store does not use all registers.
        // Instead, add uses of each register after all defs.
        for i in 0..256_i32 {
            instructions.push(Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: Some(i64::from(i) * 4),
                },
                src: reg(&format!("%r{i}"), PtxType::U32),
            });
        }
        let report = analyze_register_pressure(&instructions);
        assert!(report.spill_risk);
        assert!(report.total_peak > SPILL_THRESHOLD);
    }

    /// Below spill threshold does not flag risk.
    #[test]
    fn test_no_spill_risk() {
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
        let report = analyze_register_pressure(&instructions);
        assert!(!report.spill_risk);
    }

    /// Peak by type tracks each `PtxType` separately.
    #[test]
    fn test_peak_by_type() {
        let instructions = vec![
            Instruction::MovSpecial {
                dst: reg("%r0", PtxType::U32),
                special: SpecialReg::TidX,
            },
            Instruction::Cvt {
                rnd: None,
                dst_ty: PtxType::F32,
                src_ty: PtxType::U32,
                dst: reg("%f0", PtxType::F32),
                src: reg_op("%r0", PtxType::U32),
            },
            Instruction::Add {
                ty: PtxType::F32,
                dst: reg("%f1", PtxType::F32),
                a: reg_op("%f0", PtxType::F32),
                b: imm_u32(0),
            },
        ];
        let report = analyze_register_pressure(&instructions);
        // %r0: U32, def=0, last_use=1 → live [0,1]
        // %f0: F32, def=1, last_use=2 → live [1,2]
        // %f1: F32, def=2, last_use=2 → live [2]
        // Peak U32 = 1 (at inst 0 or 1)
        // Peak F32 = 2 (at inst 2: %f0, %f1)
        assert_eq!(report.peak_by_type.get(&PtxType::U32), Some(&1));
        assert_eq!(report.peak_by_type.get(&PtxType::F32), Some(&2));
    }

    /// Occupancy estimation for moderate register usage.
    #[test]
    fn test_occupancy_estimation() {
        // With 32 regs per thread: max_warps = 65536 / (32 * 32) = 64
        let mut instructions = Vec::new();
        for i in 0..32 {
            instructions.push(Instruction::MovSpecial {
                dst: reg(&format!("%r{i}"), PtxType::U32),
                special: SpecialReg::TidX,
            });
        }
        // Use all 32 regs after all defs so they overlap
        for i in 0..32_i32 {
            instructions.push(Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: Some(i64::from(i) * 4),
                },
                src: reg(&format!("%r{i}"), PtxType::U32),
            });
        }
        let report = analyze_register_pressure(&instructions);
        assert_eq!(report.total_peak, 32);
        assert_eq!(report.estimated_max_warps_per_sm, Some(64));
    }

    /// High register usage (128) reduces occupancy.
    #[test]
    fn test_occupancy_high_register_usage() {
        let mut instructions = Vec::new();
        for i in 0..128 {
            instructions.push(Instruction::MovSpecial {
                dst: reg(&format!("%r{i}"), PtxType::U32),
                special: SpecialReg::TidX,
            });
        }
        for i in 0..128_i32 {
            instructions.push(Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: Some(i64::from(i) * 4),
                },
                src: reg(&format!("%r{i}"), PtxType::U32),
            });
        }
        let report = analyze_register_pressure(&instructions);
        assert_eq!(report.total_peak, 128);
        // 65536 / (32 * 128) = 16
        assert_eq!(report.estimated_max_warps_per_sm, Some(16));
    }

    /// Mad/Fma instructions with three source operands.
    #[test]
    fn test_mad_three_operands() {
        let instructions = vec![
            Instruction::Mad {
                ty: PtxType::S32,
                mode: MulMode::Lo,
                dst: reg("%r0", PtxType::S32),
                a: reg_op("%r1", PtxType::S32),
                b: reg_op("%r2", PtxType::S32),
                c: reg_op("%r3", PtxType::S32),
            },
            Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::S32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: None,
                },
                src: reg("%r0", PtxType::S32),
            },
        ];
        let report = analyze_register_pressure(&instructions);
        // %r0: def=0, last_use=1 → live [0,1]
        assert_eq!(report.total_peak, 1);
    }

    /// Branch with predicate register counts as a use.
    #[test]
    fn test_branch_predicate_use() {
        let instructions = vec![
            Instruction::SetP {
                cmp: CmpOp::Lt,
                ty: PtxType::U32,
                dst: reg("%p0", PtxType::Pred),
                a: reg_op("%r0", PtxType::U32),
                b: imm_u32(10),
            },
            Instruction::Branch {
                target: "label1".to_string(),
                predicate: Some((reg("%p0", PtxType::Pred), false)),
            },
        ];
        let report = analyze_register_pressure(&instructions);
        // %p0: def=0, last_use=1 → live [0,1]
        assert_eq!(report.total_peak, 1);
    }

    /// `Mma` instruction defines `d_regs` and uses `a_regs`, `b_regs`, `c_regs`.
    #[test]
    fn test_mma_register_pressure() {
        use crate::ir::MmaShape;

        let instructions = vec![Instruction::Mma {
            shape: MmaShape::M16N8K16,
            a_ty: PtxType::F16,
            b_ty: PtxType::F16,
            c_ty: PtxType::F32,
            d_ty: PtxType::F32,
            d_regs: vec![
                reg("%f0", PtxType::F32),
                reg("%f1", PtxType::F32),
                reg("%f2", PtxType::F32),
                reg("%f3", PtxType::F32),
            ],
            a_regs: vec![reg("%f10", PtxType::F16), reg("%f11", PtxType::F16)],
            b_regs: vec![reg("%f20", PtxType::F16)],
            c_regs: vec![
                reg("%f30", PtxType::F32),
                reg("%f31", PtxType::F32),
                reg("%f32", PtxType::F32),
                reg("%f33", PtxType::F32),
            ],
        }];
        let report = analyze_register_pressure(&instructions);
        // d_regs are defined: 4 registers
        assert_eq!(report.total_peak, 4);
    }

    /// Per-instruction liveness counts are correctly sized.
    #[test]
    fn test_live_at_instruction_length() {
        let instructions = vec![
            Instruction::MovSpecial {
                dst: reg("%r0", PtxType::U32),
                special: SpecialReg::TidX,
            },
            Instruction::Return,
            Instruction::Label("exit".to_string()),
        ];
        let report = analyze_register_pressure(&instructions);
        assert_eq!(report.live_at_instruction.len(), 3);
    }

    // -------------------------------------------------------------------------
    // Additional register pressure boundary tests
    // -------------------------------------------------------------------------

    /// A kernel with fewer than 255 simultaneously-live registers produces no
    /// spill warning.
    #[test]
    fn test_register_pressure_under_limit_no_warning() {
        // 10 registers, all overlapping
        let count: usize = 10;
        let mut instructions = Vec::new();
        for i in 0..count {
            instructions.push(Instruction::MovSpecial {
                dst: reg(&format!("%r{i}"), PtxType::U32),
                special: SpecialReg::TidX,
            });
        }
        for i in 0..count {
            let offset = i64::try_from(i).unwrap_or(i64::MAX) * 4;
            instructions.push(Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: Some(offset),
                },
                src: reg(&format!("%r{i}"), PtxType::U32),
            });
        }
        let report = analyze_register_pressure(&instructions);
        assert_eq!(
            report.total_peak, count,
            "peak should equal number of simultaneously-live registers"
        );
        assert!(
            !report.spill_risk,
            "10 registers must not trigger spill risk"
        );
    }

    /// A kernel with exactly 255 simultaneously-live registers must NOT trigger
    /// spill risk (the threshold is strictly greater than 255).
    #[test]
    fn test_register_pressure_at_limit_no_warning() {
        let count: usize = SPILL_THRESHOLD; // 255
        let mut instructions = Vec::new();
        for i in 0..count {
            instructions.push(Instruction::MovSpecial {
                dst: reg(&format!("%r{i}"), PtxType::U32),
                special: SpecialReg::TidX,
            });
        }
        for i in 0..count {
            let offset = i64::try_from(i).unwrap_or(i64::MAX) * 4;
            instructions.push(Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: Some(offset),
                },
                src: reg(&format!("%r{i}"), PtxType::U32),
            });
        }
        let report = analyze_register_pressure(&instructions);
        assert_eq!(
            report.total_peak, 255,
            "peak must be exactly 255 at boundary"
        );
        assert!(
            !report.spill_risk,
            "exactly 255 registers must NOT trigger spill risk (threshold is > 255)"
        );
    }

    /// A kernel with 256 simultaneously-live registers MUST trigger spill risk.
    #[test]
    fn test_register_pressure_over_limit_warns() {
        let count: usize = SPILL_THRESHOLD + 1; // 256
        let mut instructions = Vec::new();
        for i in 0..count {
            instructions.push(Instruction::MovSpecial {
                dst: reg(&format!("%r{i}"), PtxType::U32),
                special: SpecialReg::TidX,
            });
        }
        for i in 0..count {
            let offset = i64::try_from(i).unwrap_or(i64::MAX) * 4;
            instructions.push(Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: Some(offset),
                },
                src: reg(&format!("%r{i}"), PtxType::U32),
            });
        }
        let report = analyze_register_pressure(&instructions);
        assert!(
            report.total_peak > SPILL_THRESHOLD,
            "256 simultaneously-live registers must exceed spill threshold"
        );
        assert!(
            report.spill_risk,
            "256 simultaneously-live registers must trigger spill risk"
        );
    }

    /// Register count must match the number of distinct defined registers when
    /// they are all simultaneously live.
    #[test]
    fn test_register_count_matches_allocations() {
        let count: usize = 5;
        let mut instructions = Vec::new();
        for i in 0..count {
            instructions.push(Instruction::MovSpecial {
                dst: reg(&format!("%r{i}"), PtxType::U32),
                special: SpecialReg::TidX,
            });
        }
        // Use all registers after all defs so they are all live simultaneously
        // at the first use instruction
        for i in 0..count {
            let offset = i64::try_from(i).unwrap_or(i64::MAX) * 4;
            instructions.push(Instruction::Store {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::U32,
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: Some(offset),
                },
                src: reg(&format!("%r{i}"), PtxType::U32),
            });
        }
        let report = analyze_register_pressure(&instructions);
        // All 5 registers are live simultaneously during the store phase
        assert_eq!(
            report.total_peak, count,
            "peak must match the number of allocated registers"
        );
        assert_eq!(
            report.peak_by_type.get(&PtxType::U32).copied().unwrap_or(0),
            count,
            "peak U32 count must match the number of U32 allocations"
        );
    }
}
