//! Per-architecture instruction legality checking.
//!
//! This module provides a static analysis pass that validates whether each
//! instruction in a PTX sequence is legal for the target SM architecture.
//! Instructions that require newer hardware (e.g., WGMMA on SM < 90) are
//! flagged as violations, while borderline cases generate warnings.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::arch::SmVersion;
//! use oxicuda_ptx::ir::{Instruction, PtxType, Register, Operand, ImmValue};
//! use oxicuda_ptx::analysis::arch_legality::check_instruction_legality;
//!
//! let instructions = vec![
//!     Instruction::Add {
//!         ty: PtxType::S32,
//!         dst: Register { name: "%r0".into(), ty: PtxType::S32 },
//!         a: Operand::Immediate(ImmValue::S32(1)),
//!         b: Operand::Immediate(ImmValue::S32(2)),
//!     },
//! ];
//! let report = check_instruction_legality(&instructions, SmVersion::Sm75);
//! assert!(report.violations.is_empty());
//! ```

use std::fmt;

use crate::arch::SmVersion;
use crate::ir::Instruction;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of an architecture legality check over a sequence of instructions.
#[derive(Debug)]
pub struct LegalityReport {
    /// Instructions that are illegal on the target architecture.
    pub violations: Vec<LegalityViolation>,
    /// Non-fatal advisory messages.
    pub warnings: Vec<LegalityWarning>,
    /// Total number of instructions in the checked sequence.
    pub instruction_count: usize,
    /// Number of instructions that were actively checked (non-universal).
    pub checked_count: usize,
}

/// A single architecture legality violation.
#[derive(Debug)]
pub struct LegalityViolation {
    /// Zero-based index of the offending instruction in the sequence.
    pub instruction_index: usize,
    /// Human-readable description of the instruction.
    pub instruction_desc: String,
    /// Minimum SM version required for this instruction.
    pub required_sm: SmVersion,
    /// The target SM version that was checked against.
    pub target_sm: SmVersion,
    /// Explanation of why this instruction requires a newer architecture.
    pub reason: String,
}

/// A non-fatal advisory produced during legality checking.
#[derive(Debug)]
pub struct LegalityWarning {
    /// Zero-based index of the instruction that triggered the warning.
    pub instruction_index: usize,
    /// Human-readable warning message.
    pub message: String,
}

// ---------------------------------------------------------------------------
// Display implementations
// ---------------------------------------------------------------------------

impl fmt::Display for LegalityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Legality Report: {}/{} instructions checked, {} violation(s), {} warning(s)",
            self.checked_count,
            self.instruction_count,
            self.violations.len(),
            self.warnings.len(),
        )?;
        for v in &self.violations {
            writeln!(f, "  VIOLATION: {v}")?;
        }
        for w in &self.warnings {
            writeln!(f, "  WARNING: {w}")?;
        }
        Ok(())
    }
}

impl fmt::Display for LegalityViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] '{}' requires {} (target: {}): {}",
            self.instruction_index,
            self.instruction_desc,
            self.required_sm,
            self.target_sm,
            self.reason,
        )
    }
}

impl fmt::Display for LegalityWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.instruction_index, self.message)
    }
}

// ---------------------------------------------------------------------------
// Core legality logic
// ---------------------------------------------------------------------------

/// Returns the minimum SM version required for the given instruction.
///
/// Returns `None` if the instruction is universally supported on all
/// architectures from SM 75 onwards.
#[must_use]
pub const fn minimum_sm_for_instruction(instr: &Instruction) -> Option<SmVersion> {
    match instr {
        // Turing (SM 75) -- our minimum, so always legal
        Instruction::Wmma { .. } | Instruction::Dp4a { .. } | Instruction::Dp2a { .. } => {
            Some(SmVersion::Sm75)
        }

        // Ampere (SM 80)
        Instruction::Mma { .. }
        | Instruction::CpAsync { .. }
        | Instruction::CpAsyncCommit
        | Instruction::CpAsyncWait { .. }
        | Instruction::Redux { .. }
        | Instruction::FenceProxy { .. } => Some(SmVersion::Sm80),

        // Hopper (SM 90)
        Instruction::Wgmma { .. }
        | Instruction::TmaLoad { .. }
        | Instruction::Stmatrix { .. }
        | Instruction::ElectSync { .. }
        | Instruction::Setmaxnreg { .. }
        | Instruction::Griddepcontrol { .. }
        | Instruction::MbarrierInit { .. }
        | Instruction::MbarrierArrive { .. }
        | Instruction::MbarrierWait { .. }
        | Instruction::BarrierCluster
        | Instruction::FenceCluster
        | Instruction::CpAsyncBulk { .. } => Some(SmVersion::Sm90),

        // Blackwell (SM 100)
        Instruction::Tcgen05Mma { .. } => Some(SmVersion::Sm100),

        // All other instructions are universally supported (SM 75+)
        Instruction::FenceAcqRel { .. }
        | Instruction::Add { .. }
        | Instruction::Sub { .. }
        | Instruction::Mul { .. }
        | Instruction::Mad { .. }
        | Instruction::MadLo { .. }
        | Instruction::MadHi { .. }
        | Instruction::MadWide { .. }
        | Instruction::Fma { .. }
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
        | Instruction::Store { .. }
        | Instruction::Cvt { .. }
        | Instruction::Branch { .. }
        | Instruction::Label(_)
        | Instruction::Return
        | Instruction::BarSync { .. }
        | Instruction::BarArrive { .. }
        | Instruction::Atom { .. }
        | Instruction::AtomCas { .. }
        | Instruction::Red { .. }
        | Instruction::MovSpecial { .. }
        | Instruction::LoadParam { .. }
        | Instruction::Comment(_)
        | Instruction::Raw(_)
        | Instruction::Pragma(_)
        | Instruction::Tex1d { .. }
        | Instruction::Tex2d { .. }
        | Instruction::Tex3d { .. }
        | Instruction::SurfLoad { .. }
        | Instruction::SurfStore { .. } => None,

        // Turing (SM 75): ldmatrix is available from SM 75+
        Instruction::Ldmatrix { .. } => Some(SmVersion::Sm75),
    }
}

/// Returns the minimum SM version and a human-readable reason for
/// architecture-specific instructions.
///
/// Returns `None` for universally supported instructions.
#[must_use]
pub const fn instruction_arch_requirement(
    instr: &Instruction,
) -> Option<(SmVersion, &'static str)> {
    match instr {
        Instruction::Wmma { .. } => Some((
            SmVersion::Sm75,
            "WMMA (Warp Matrix Multiply-Accumulate) requires Turing (SM 75) or later",
        )),
        Instruction::Mma { .. } => Some((
            SmVersion::Sm80,
            "MMA (mma.sync.aligned) requires Ampere (SM 80) or later",
        )),
        Instruction::Wgmma { .. } => Some((
            SmVersion::Sm90,
            "WGMMA (Warp Group MMA) requires Hopper (SM 90) or later",
        )),
        Instruction::TmaLoad { .. } => Some((
            SmVersion::Sm90,
            "TMA (Tensor Memory Accelerator) load requires Hopper (SM 90) or later",
        )),
        Instruction::CpAsync { .. } => Some((
            SmVersion::Sm80,
            "cp.async (asynchronous global-to-shared copy) requires Ampere (SM 80) or later",
        )),
        Instruction::CpAsyncCommit => Some((
            SmVersion::Sm80,
            "cp.async.commit_group requires Ampere (SM 80) or later",
        )),
        Instruction::CpAsyncWait { .. } => Some((
            SmVersion::Sm80,
            "cp.async.wait_group requires Ampere (SM 80) or later",
        )),
        Instruction::Dp4a { .. } => Some((
            SmVersion::Sm75,
            "dp4a (4-way byte dot product) requires Turing (SM 75) or later",
        )),
        Instruction::Dp2a { .. } => Some((
            SmVersion::Sm75,
            "dp2a (2-way halfword dot product) requires Turing (SM 75) or later",
        )),
        Instruction::Redux { .. } => Some((
            SmVersion::Sm80,
            "redux.sync (warp-level reduction) requires Ampere (SM 80) or later",
        )),
        Instruction::Stmatrix { .. } => Some((
            SmVersion::Sm90,
            "stmatrix (cooperative store to shared memory) requires Hopper (SM 90) or later",
        )),
        Instruction::ElectSync { .. } => Some((
            SmVersion::Sm90,
            "elect.sync (warp leader election) requires Hopper (SM 90) or later",
        )),
        Instruction::Setmaxnreg { .. } => Some((
            SmVersion::Sm90,
            "setmaxnreg (dynamic register limit) requires Hopper (SM 90) or later",
        )),
        Instruction::Griddepcontrol { .. } => Some((
            SmVersion::Sm90,
            "griddepcontrol (grid dependency control) requires Hopper (SM 90) or later",
        )),
        Instruction::FenceProxy { .. } => Some((
            SmVersion::Sm80,
            "fence.proxy.async requires Ampere (SM 80) or later",
        )),
        Instruction::MbarrierInit { .. } => Some((
            SmVersion::Sm90,
            "mbarrier.init requires Hopper (SM 90) or later",
        )),
        Instruction::MbarrierArrive { .. } => Some((
            SmVersion::Sm90,
            "mbarrier.arrive requires Hopper (SM 90) or later",
        )),
        Instruction::MbarrierWait { .. } => Some((
            SmVersion::Sm90,
            "mbarrier.try_wait requires Hopper (SM 90) or later",
        )),
        Instruction::Tcgen05Mma { .. } => Some((
            SmVersion::Sm100,
            "tcgen05.mma (5th-gen Tensor Core) requires Blackwell (SM 100) or later",
        )),
        Instruction::BarrierCluster => Some((
            SmVersion::Sm90,
            "barrier.cluster requires Hopper (SM 90) or later",
        )),
        Instruction::FenceCluster => Some((
            SmVersion::Sm90,
            "fence.mbarrier_init.release.cluster requires Hopper (SM 90) or later",
        )),
        Instruction::CpAsyncBulk { .. } => Some((
            SmVersion::Sm90,
            "cp.async.bulk.tensor requires Hopper (SM 90) or later",
        )),
        Instruction::Ldmatrix { .. } => Some((
            SmVersion::Sm75,
            "ldmatrix (warp-cooperative shared memory load) requires Turing (SM 75) or later",
        )),
        _ => None,
    }
}

/// Checks whether a single instruction is legal on the target SM architecture.
#[must_use]
pub fn is_instruction_legal(instr: &Instruction, target_sm: SmVersion) -> bool {
    minimum_sm_for_instruction(instr).is_none_or(|required| target_sm >= required)
}

/// Returns a short human-readable description of an instruction for diagnostics.
fn instruction_description(instr: &Instruction) -> String {
    match instr {
        Instruction::Add { ty, .. } => format!("add{}", ty.as_ptx_str()),
        Instruction::Sub { ty, .. } => format!("sub{}", ty.as_ptx_str()),
        Instruction::Mul { ty, mode, .. } => {
            format!("mul{}{}", mode.as_ptx_str(), ty.as_ptx_str())
        }
        Instruction::Mad { ty, mode, .. } => {
            format!("mad{}{}", mode.as_ptx_str(), ty.as_ptx_str())
        }
        Instruction::MadLo { typ, .. } => format!("mad.lo{}", typ.as_ptx_str()),
        Instruction::MadHi { typ, .. } => format!("mad.hi{}", typ.as_ptx_str()),
        Instruction::MadWide { src_typ, .. } => format!("mad.wide{}", src_typ.as_ptx_str()),
        Instruction::Fma { ty, .. } => format!("fma{}", ty.as_ptx_str()),
        Instruction::Neg { ty, .. } => format!("neg{}", ty.as_ptx_str()),
        Instruction::Abs { ty, .. } => format!("abs{}", ty.as_ptx_str()),
        Instruction::Min { ty, .. } => format!("min{}", ty.as_ptx_str()),
        Instruction::Max { ty, .. } => format!("max{}", ty.as_ptx_str()),
        Instruction::Brev { .. } => "brev".into(),
        Instruction::Clz { .. } => "clz".into(),
        Instruction::Popc { .. } => "popc".into(),
        Instruction::Bfind { .. } => "bfind".into(),
        Instruction::Bfe { .. } => "bfe".into(),
        Instruction::Bfi { .. } => "bfi".into(),
        Instruction::Rcp { .. } => "rcp".into(),
        Instruction::Rsqrt { .. } => "rsqrt".into(),
        Instruction::Sqrt { .. } => "sqrt".into(),
        Instruction::Ex2 { .. } => "ex2".into(),
        Instruction::Lg2 { .. } => "lg2".into(),
        Instruction::Sin { .. } => "sin".into(),
        Instruction::Cos { .. } => "cos".into(),
        Instruction::Shl { .. } => "shl".into(),
        Instruction::Shr { .. } => "shr".into(),
        Instruction::Div { .. } => "div".into(),
        Instruction::Rem { .. } => "rem".into(),
        Instruction::And { .. } => "and".into(),
        Instruction::Or { .. } => "or".into(),
        Instruction::Xor { .. } => "xor".into(),
        Instruction::SetP { .. } => "setp".into(),
        Instruction::Load { .. } => "ld".into(),
        Instruction::Store { .. } => "st".into(),
        Instruction::CpAsync { bytes, .. } => format!("cp.async ({bytes} bytes)"),
        Instruction::CpAsyncCommit => "cp.async.commit_group".into(),
        Instruction::CpAsyncWait { n } => format!("cp.async.wait_group ({n})"),
        Instruction::Cvt { .. } => "cvt".into(),
        Instruction::Branch { target, .. } => format!("bra {target}"),
        Instruction::Label(name) => format!("label {name}"),
        Instruction::Return => "ret".into(),
        Instruction::BarSync { id } => format!("bar.sync {id}"),
        Instruction::BarArrive { id, .. } => format!("bar.arrive {id}"),
        Instruction::FenceAcqRel { scope } => format!("fence.acq_rel.{scope:?}"),
        Instruction::Wmma { op, shape, .. } => format!("wmma.{op:?}.{shape:?}"),
        Instruction::Mma { shape, .. } => format!("mma.sync.{shape:?}"),
        Instruction::Wgmma { shape, .. } => format!("wgmma.{shape:?}"),
        Instruction::TmaLoad { .. } => "cp.async.bulk (TMA load)".into(),
        Instruction::Atom { op, .. } => format!("atom.{op:?}"),
        Instruction::AtomCas { .. } => "atom.cas".into(),
        Instruction::Red { op, .. } => format!("red.{op:?}"),
        Instruction::MovSpecial { special, .. } => format!("mov.{special:?}"),
        Instruction::LoadParam { param_name, .. } => format!("ld.param {param_name}"),
        Instruction::Comment(_) => "comment".into(),
        Instruction::Raw(_) => "raw".into(),
        Instruction::Pragma(_) => "pragma".into(),
        Instruction::Dp4a { .. } => "dp4a".into(),
        Instruction::Dp2a { .. } => "dp2a".into(),
        Instruction::Redux { op, .. } => format!("redux.sync.{op:?}"),
        Instruction::Stmatrix { shape, trans, .. } => {
            let t = if *trans { ".trans" } else { "" };
            format!("stmatrix.{shape:?}{t}")
        }
        Instruction::ElectSync { .. } => "elect.sync".into(),
        Instruction::Setmaxnreg { reg_count, .. } => format!("setmaxnreg {reg_count}"),
        Instruction::Griddepcontrol { action } => format!("griddepcontrol.{action:?}"),
        Instruction::FenceProxy { scope, .. } => format!("fence.proxy.async.{scope:?}"),
        Instruction::MbarrierInit { .. } => "mbarrier.init".into(),
        Instruction::MbarrierArrive { .. } => "mbarrier.arrive".into(),
        Instruction::MbarrierWait { .. } => "mbarrier.try_wait".into(),
        Instruction::Tcgen05Mma { .. } => "tcgen05.mma".into(),
        Instruction::BarrierCluster => "barrier.cluster".into(),
        Instruction::FenceCluster => "fence.mbarrier_init.release.cluster".into(),
        Instruction::CpAsyncBulk { .. } => "cp.async.bulk.tensor.1d".into(),
        Instruction::Tex1d { .. } => "tex.1d".into(),
        Instruction::Tex2d { .. } => "tex.2d".into(),
        Instruction::Tex3d { .. } => "tex.3d".into(),
        Instruction::SurfLoad { .. } => "suld".into(),
        Instruction::SurfStore { .. } => "sust".into(),
        Instruction::Ldmatrix { num_fragments, .. } => {
            format!("ldmatrix.sync.aligned.m8n8.x{num_fragments}.shared.b16")
        }
    }
}

/// Checks all instructions in a sequence for architecture legality.
///
/// Returns a [`LegalityReport`] containing any violations (instructions that
/// require a newer SM than `target_sm`) and warnings.
#[must_use]
pub fn check_instruction_legality(
    instructions: &[Instruction],
    target_sm: SmVersion,
) -> LegalityReport {
    let instruction_count = instructions.len();
    let mut violations = Vec::new();
    let mut warnings = Vec::new();
    let mut checked_count = 0usize;

    for (idx, instr) in instructions.iter().enumerate() {
        // Check for architecture-specific requirements
        if let Some(required_sm) = minimum_sm_for_instruction(instr) {
            checked_count += 1;
            if target_sm < required_sm {
                let reason = instruction_arch_requirement(instr).map_or_else(
                    || format!("requires {required_sm} or later"),
                    |(_, reason)| reason.to_string(),
                );
                violations.push(LegalityViolation {
                    instruction_index: idx,
                    instruction_desc: instruction_description(instr),
                    required_sm,
                    target_sm,
                    reason,
                });
            }
        }

        // Generate warnings for edge cases
        generate_warnings(instr, idx, target_sm, &mut warnings);
    }

    LegalityReport {
        violations,
        warnings,
        instruction_count,
        checked_count,
    }
}

/// Generates warnings for instructions that are legal but may have
/// performance or compatibility implications on the target architecture.
fn generate_warnings(
    instr: &Instruction,
    idx: usize,
    target_sm: SmVersion,
    warnings: &mut Vec<LegalityWarning>,
) {
    match instr {
        // FenceAcqRel is always legal but memory ordering semantics
        // changed significantly between SM versions
        Instruction::FenceAcqRel { .. }
            // SM 75 is our minimum so this is always fine, but if we had
            // older architectures this would warn. We still note that
            // fence behavior varies by architecture for SM 75.
            if target_sm == SmVersion::Sm75 => {
                warnings.push(LegalityWarning {
                    instruction_index: idx,
                    message: "fence.acq_rel on SM 75 (Turing) has different memory \
                              ordering semantics than SM 80+; verify correctness"
                        .into(),
                });
            }

        // Warn about WMMA on very new architectures where WGMMA is preferred
        Instruction::Wmma { .. }
            if target_sm >= SmVersion::Sm90 => {
                warnings.push(LegalityWarning {
                    instruction_index: idx,
                    message: "WMMA is legal on SM 90+ but WGMMA is preferred for \
                              higher throughput on Hopper and newer architectures"
                        .into(),
                });
            }

        // Warn about MMA on Hopper+ where WGMMA is generally better
        Instruction::Mma { .. }
            if target_sm >= SmVersion::Sm90 => {
                warnings.push(LegalityWarning {
                    instruction_index: idx,
                    message: "mma.sync is legal on SM 90+ but WGMMA offers higher \
                              throughput on Hopper and newer architectures"
                        .into(),
                });
            }

        // Raw instructions cannot be validated -- warn unconditionally
        Instruction::Raw(_) => {
            warnings.push(LegalityWarning {
                instruction_index: idx,
                message: "Raw PTX instruction cannot be validated for architecture \
                          legality; verify manually"
                    .into(),
            });
        }

        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        FenceScope, ImmValue, MmaShape, Operand, PtxType, Register, WgmmaShape, WmmaLayout, WmmaOp,
        WmmaShape,
    };

    /// Helper to create a simple Add instruction (universally supported).
    fn make_add() -> Instruction {
        Instruction::Add {
            ty: PtxType::S32,
            dst: Register {
                name: "%r0".into(),
                ty: PtxType::S32,
            },
            a: Operand::Immediate(ImmValue::S32(1)),
            b: Operand::Immediate(ImmValue::S32(2)),
        }
    }

    /// Helper to create a WMMA instruction (requires SM 75).
    fn make_wmma() -> Instruction {
        Instruction::Wmma {
            op: WmmaOp::Mma,
            shape: WmmaShape::M16N16K16,
            layout: WmmaLayout::RowMajor,
            ty: PtxType::F16,
            fragments: vec![],
            addr: None,
            stride: None,
        }
    }

    /// Helper to create an MMA instruction (requires SM 80).
    fn make_mma() -> Instruction {
        Instruction::Mma {
            shape: MmaShape::M16N8K16,
            a_ty: PtxType::F16,
            b_ty: PtxType::F16,
            c_ty: PtxType::F32,
            d_ty: PtxType::F32,
            d_regs: vec![],
            a_regs: vec![],
            b_regs: vec![],
            c_regs: vec![],
        }
    }

    /// Helper to create a WGMMA instruction (requires SM 90).
    fn make_wgmma() -> Instruction {
        Instruction::Wgmma {
            shape: WgmmaShape::M64N128K16,
            d_ty: PtxType::F32,
            a_ty: PtxType::F16,
            b_ty: PtxType::F16,
            desc_a: Register {
                name: "%rd0".into(),
                ty: PtxType::U64,
            },
            desc_b: Register {
                name: "%rd1".into(),
                ty: PtxType::U64,
            },
            d_regs: vec![],
            scale_d: 1,
            imm_scale_a: 1,
            imm_scale_b: 1,
            trans_a: 0,
            trans_b: 0,
        }
    }

    /// Helper to create a `TmaLoad` instruction (requires SM 90).
    fn make_tma_load() -> Instruction {
        Instruction::TmaLoad {
            dst_shared: Operand::Immediate(ImmValue::U32(0)),
            desc: Register {
                name: "%rd0".into(),
                ty: PtxType::U64,
            },
            coords: vec![],
            barrier: Register {
                name: "%rd1".into(),
                ty: PtxType::U64,
            },
        }
    }

    /// Helper to create a `CpAsync` instruction (requires SM 80).
    fn make_cp_async() -> Instruction {
        Instruction::CpAsync {
            bytes: 16,
            dst_shared: Operand::Immediate(ImmValue::U32(0)),
            src_global: Operand::Immediate(ImmValue::U32(0)),
        }
    }

    /// Helper to create a Dp4a instruction (requires SM 75).
    fn make_dp4a() -> Instruction {
        Instruction::Dp4a {
            dst: Register {
                name: "%r0".into(),
                ty: PtxType::S32,
            },
            a: Operand::Immediate(ImmValue::U32(0)),
            b: Operand::Immediate(ImmValue::U32(0)),
            c: Operand::Immediate(ImmValue::U32(0)),
            signed_a: true,
            signed_b: true,
        }
    }

    // -----------------------------------------------------------------------
    // Test: Universal instructions pass on all SM versions
    // -----------------------------------------------------------------------
    #[test]
    fn universal_instructions_legal_on_all_sm() {
        let add = make_add();
        let all_sms = [
            SmVersion::Sm75,
            SmVersion::Sm80,
            SmVersion::Sm86,
            SmVersion::Sm89,
            SmVersion::Sm90,
            SmVersion::Sm90a,
            SmVersion::Sm100,
            SmVersion::Sm120,
        ];
        for sm in all_sms {
            assert!(
                is_instruction_legal(&add, sm),
                "Add should be legal on {sm}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: WMMA accepted on SM 75+ (our minimum, so always legal)
    // -----------------------------------------------------------------------
    #[test]
    fn wmma_legal_on_sm75_and_above() {
        let wmma = make_wmma();
        // SM 75 is our minimum, so WMMA is always legal
        assert!(is_instruction_legal(&wmma, SmVersion::Sm75));
        assert!(is_instruction_legal(&wmma, SmVersion::Sm80));
        assert!(is_instruction_legal(&wmma, SmVersion::Sm90));
        assert!(is_instruction_legal(&wmma, SmVersion::Sm120));
    }

    // -----------------------------------------------------------------------
    // Test: MMA rejected on SM < 80, accepted on SM 80+
    // -----------------------------------------------------------------------
    #[test]
    fn mma_rejected_below_sm80() {
        let mma = make_mma();
        assert!(!is_instruction_legal(&mma, SmVersion::Sm75));
    }

    #[test]
    fn mma_accepted_on_sm80_and_above() {
        let mma = make_mma();
        assert!(is_instruction_legal(&mma, SmVersion::Sm80));
        assert!(is_instruction_legal(&mma, SmVersion::Sm86));
        assert!(is_instruction_legal(&mma, SmVersion::Sm89));
        assert!(is_instruction_legal(&mma, SmVersion::Sm90));
        assert!(is_instruction_legal(&mma, SmVersion::Sm120));
    }

    // -----------------------------------------------------------------------
    // Test: WGMMA rejected on SM < 90, accepted on SM 90+
    // -----------------------------------------------------------------------
    #[test]
    fn wgmma_rejected_below_sm90() {
        let wgmma = make_wgmma();
        assert!(!is_instruction_legal(&wgmma, SmVersion::Sm75));
        assert!(!is_instruction_legal(&wgmma, SmVersion::Sm80));
        assert!(!is_instruction_legal(&wgmma, SmVersion::Sm86));
        assert!(!is_instruction_legal(&wgmma, SmVersion::Sm89));
    }

    #[test]
    fn wgmma_accepted_on_sm90_and_above() {
        let wgmma = make_wgmma();
        assert!(is_instruction_legal(&wgmma, SmVersion::Sm90));
        assert!(is_instruction_legal(&wgmma, SmVersion::Sm90a));
        assert!(is_instruction_legal(&wgmma, SmVersion::Sm100));
        assert!(is_instruction_legal(&wgmma, SmVersion::Sm120));
    }

    // -----------------------------------------------------------------------
    // Test: TmaLoad rejected on SM < 90
    // -----------------------------------------------------------------------
    #[test]
    fn tma_load_rejected_below_sm90() {
        let tma = make_tma_load();
        assert!(!is_instruction_legal(&tma, SmVersion::Sm75));
        assert!(!is_instruction_legal(&tma, SmVersion::Sm80));
        assert!(!is_instruction_legal(&tma, SmVersion::Sm89));
    }

    #[test]
    fn tma_load_accepted_on_sm90_and_above() {
        let tma = make_tma_load();
        assert!(is_instruction_legal(&tma, SmVersion::Sm90));
        assert!(is_instruction_legal(&tma, SmVersion::Sm100));
    }

    // -----------------------------------------------------------------------
    // Test: CpAsync rejected on SM < 80
    // -----------------------------------------------------------------------
    #[test]
    fn cp_async_rejected_below_sm80() {
        let cp = make_cp_async();
        assert!(!is_instruction_legal(&cp, SmVersion::Sm75));
    }

    #[test]
    fn cp_async_accepted_on_sm80_and_above() {
        let cp = make_cp_async();
        assert!(is_instruction_legal(&cp, SmVersion::Sm80));
        assert!(is_instruction_legal(&cp, SmVersion::Sm90));
    }

    #[test]
    fn cp_async_commit_and_wait_require_sm80() {
        let commit = Instruction::CpAsyncCommit;
        let wait = Instruction::CpAsyncWait { n: 0 };
        assert!(!is_instruction_legal(&commit, SmVersion::Sm75));
        assert!(!is_instruction_legal(&wait, SmVersion::Sm75));
        assert!(is_instruction_legal(&commit, SmVersion::Sm80));
        assert!(is_instruction_legal(&wait, SmVersion::Sm80));
    }

    // -----------------------------------------------------------------------
    // Test: Multiple violations in a single sequence
    // -----------------------------------------------------------------------
    #[test]
    fn multiple_violations_in_sequence() {
        let instructions = vec![
            make_add(),      // legal everywhere
            make_mma(),      // requires SM 80
            make_wgmma(),    // requires SM 90
            make_tma_load(), // requires SM 90
            make_cp_async(), // requires SM 80
        ];
        let report = check_instruction_legality(&instructions, SmVersion::Sm75);
        assert_eq!(report.instruction_count, 5);
        assert_eq!(report.violations.len(), 4);
        // MMA at index 1
        assert_eq!(report.violations[0].instruction_index, 1);
        assert_eq!(report.violations[0].required_sm, SmVersion::Sm80);
        // WGMMA at index 2
        assert_eq!(report.violations[1].instruction_index, 2);
        assert_eq!(report.violations[1].required_sm, SmVersion::Sm90);
        // TmaLoad at index 3
        assert_eq!(report.violations[2].instruction_index, 3);
        assert_eq!(report.violations[2].required_sm, SmVersion::Sm90);
        // CpAsync at index 4
        assert_eq!(report.violations[3].instruction_index, 4);
        assert_eq!(report.violations[3].required_sm, SmVersion::Sm80);
    }

    // -----------------------------------------------------------------------
    // Test: Empty instruction sequence
    // -----------------------------------------------------------------------
    #[test]
    fn empty_sequence() {
        let report = check_instruction_legality(&[], SmVersion::Sm75);
        assert_eq!(report.instruction_count, 0);
        assert_eq!(report.checked_count, 0);
        assert!(report.violations.is_empty());
        assert!(report.warnings.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test: All-legal sequence returns empty violations
    // -----------------------------------------------------------------------
    #[test]
    fn all_legal_sequence() {
        let instructions = vec![make_add(), make_add(), make_add()];
        let report = check_instruction_legality(&instructions, SmVersion::Sm75);
        assert_eq!(report.instruction_count, 3);
        assert!(report.violations.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test: is_instruction_legal helper
    // -----------------------------------------------------------------------
    #[test]
    fn is_instruction_legal_helper() {
        assert!(is_instruction_legal(&make_add(), SmVersion::Sm75));
        assert!(!is_instruction_legal(&make_mma(), SmVersion::Sm75));
        assert!(is_instruction_legal(&make_mma(), SmVersion::Sm80));
        assert!(!is_instruction_legal(&make_wgmma(), SmVersion::Sm89));
        assert!(is_instruction_legal(&make_wgmma(), SmVersion::Sm90));
    }

    // -----------------------------------------------------------------------
    // Test: instruction_arch_requirement returns correct reasons
    // -----------------------------------------------------------------------
    #[test]
    fn instruction_arch_requirement_reasons() {
        let (sm, reason) =
            instruction_arch_requirement(&make_wmma()).expect("WMMA should have arch requirement");
        assert_eq!(sm, SmVersion::Sm75);
        assert!(reason.contains("WMMA"));

        let (sm, reason) =
            instruction_arch_requirement(&make_mma()).expect("MMA should have arch requirement");
        assert_eq!(sm, SmVersion::Sm80);
        assert!(reason.contains("MMA"));
        assert!(reason.contains("Ampere"));

        let (sm, reason) = instruction_arch_requirement(&make_wgmma())
            .expect("WGMMA should have arch requirement");
        assert_eq!(sm, SmVersion::Sm90);
        assert!(reason.contains("WGMMA"));
        assert!(reason.contains("Hopper"));

        let (sm, reason) = instruction_arch_requirement(&make_tma_load())
            .expect("TmaLoad should have arch requirement");
        assert_eq!(sm, SmVersion::Sm90);
        assert!(reason.contains("TMA"));

        let (sm, reason) = instruction_arch_requirement(&make_cp_async())
            .expect("CpAsync should have arch requirement");
        assert_eq!(sm, SmVersion::Sm80);
        assert!(reason.contains("cp.async"));

        // Universal instruction has no requirement
        assert!(instruction_arch_requirement(&make_add()).is_none());
    }

    // -----------------------------------------------------------------------
    // Test: Dp4a arch requirement
    // -----------------------------------------------------------------------
    #[test]
    fn dp4a_arch_requirement() {
        let dp4a = make_dp4a();
        let (sm, reason) =
            instruction_arch_requirement(&dp4a).expect("Dp4a should have arch requirement");
        assert_eq!(sm, SmVersion::Sm75);
        assert!(reason.contains("dp4a"));
        // SM 75 is our minimum so dp4a is always legal
        assert!(is_instruction_legal(&dp4a, SmVersion::Sm75));
    }

    // -----------------------------------------------------------------------
    // Test: LegalityReport Display impl
    // -----------------------------------------------------------------------
    #[test]
    fn legality_report_display() {
        let instructions = vec![make_mma(), make_wgmma()];
        let report = check_instruction_legality(&instructions, SmVersion::Sm75);
        let display = format!("{report}");
        assert!(display.contains("VIOLATION"));
        assert!(display.contains("2 violation(s)"));
        assert!(display.contains("sm_80"));
        assert!(display.contains("sm_90"));
    }

    // -----------------------------------------------------------------------
    // Test: Warning generation for edge cases
    // -----------------------------------------------------------------------
    #[test]
    fn fence_acq_rel_warning_on_sm75() {
        let fence = Instruction::FenceAcqRel {
            scope: FenceScope::Gpu,
        };
        let report = check_instruction_legality(&[fence], SmVersion::Sm75);
        assert!(report.violations.is_empty());
        assert_eq!(report.warnings.len(), 1);
        assert!(report.warnings[0].message.contains("fence.acq_rel"));
        assert!(report.warnings[0].message.contains("SM 75"));
    }

    #[test]
    fn fence_acq_rel_no_warning_on_sm80() {
        let fence = Instruction::FenceAcqRel {
            scope: FenceScope::Gpu,
        };
        let report = check_instruction_legality(&[fence], SmVersion::Sm80);
        assert!(report.violations.is_empty());
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn wmma_warning_on_sm90_prefers_wgmma() {
        let wmma = make_wmma();
        let report = check_instruction_legality(&[wmma], SmVersion::Sm90);
        assert!(report.violations.is_empty());
        assert_eq!(report.warnings.len(), 1);
        assert!(report.warnings[0].message.contains("WGMMA"));
    }

    #[test]
    fn raw_instruction_warning() {
        let raw = Instruction::Raw("some.custom.instr;".into());
        let report = check_instruction_legality(&[raw], SmVersion::Sm80);
        assert!(report.violations.is_empty());
        assert_eq!(report.warnings.len(), 1);
        assert!(report.warnings[0].message.contains("Raw PTX"));
    }

    // -----------------------------------------------------------------------
    // Test: checked_count tracks arch-specific instructions
    // -----------------------------------------------------------------------
    #[test]
    fn checked_count_tracks_arch_specific() {
        let instructions = vec![
            make_add(),   // universal
            make_mma(),   // arch-specific (SM 80)
            make_add(),   // universal
            make_wgmma(), // arch-specific (SM 90)
        ];
        let report = check_instruction_legality(&instructions, SmVersion::Sm90);
        assert_eq!(report.instruction_count, 4);
        assert_eq!(report.checked_count, 2);
        assert!(report.violations.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test: minimum_sm_for_instruction returns None for universal ops
    // -----------------------------------------------------------------------
    #[test]
    fn minimum_sm_none_for_universal() {
        assert!(minimum_sm_for_instruction(&make_add()).is_none());
        assert!(minimum_sm_for_instruction(&Instruction::Return).is_none());
        assert!(minimum_sm_for_instruction(&Instruction::Comment("test".into())).is_none());
        assert!(minimum_sm_for_instruction(&Instruction::Label("L0".into())).is_none());
    }

    #[test]
    fn minimum_sm_some_for_arch_specific() {
        assert_eq!(
            minimum_sm_for_instruction(&make_mma()),
            Some(SmVersion::Sm80)
        );
        assert_eq!(
            minimum_sm_for_instruction(&make_wgmma()),
            Some(SmVersion::Sm90)
        );
        assert_eq!(
            minimum_sm_for_instruction(&make_tma_load()),
            Some(SmVersion::Sm90)
        );
        assert_eq!(
            minimum_sm_for_instruction(&make_cp_async()),
            Some(SmVersion::Sm80)
        );
    }
}
