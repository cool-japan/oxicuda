//! Constant folding optimization pass for PTX instruction sequences.
//!
//! Evaluates constant expressions at compile time. When both operands of an
//! arithmetic instruction are immediate values (or registers that hold known
//! constant values), the result is computed and the instruction is replaced
//! with a constant assignment encoded as `add.ty dst, result, 0`.

use std::collections::HashMap;

use crate::ir::{CmpOp, ImmValue, Instruction, MulMode, Operand, PtxType, Register};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Report summarizing the constant folding pass results.
#[derive(Debug, Clone)]
pub struct ConstantFoldingReport {
    /// Number of instructions whose result was computed at compile time.
    pub folded_count: usize,
    /// Number of times a known-constant register was used to resolve an
    /// operand in a downstream instruction (simple constant propagation).
    pub propagated_count: usize,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fold constant expressions in an instruction sequence.
///
/// Tracks which registers hold known constant values. When an instruction
/// reads only constants, it computes the result at compile time and replaces
/// the instruction with a constant assignment (`add.ty dst, result, 0`).
///
/// Returns the optimized instruction list and count of folded instructions.
#[allow(clippy::too_many_lines)]
pub fn fold_constants(instructions: &[Instruction]) -> (Vec<Instruction>, usize) {
    let report = fold_constants_report(instructions);
    let folded = report.folded_count;
    let (result, _) = fold_constants_inner(instructions);
    (result, folded)
}

/// Fold constant expressions and return a detailed report.
///
/// Like [`fold_constants`] but returns a [`ConstantFoldingReport`] with both
/// the folded count and the propagation count.
pub fn fold_constants_report(instructions: &[Instruction]) -> ConstantFoldingReport {
    let (_, report) = fold_constants_inner(instructions);
    report
}

/// Inner implementation that returns both the optimized instructions and
/// the detailed report.
#[allow(clippy::too_many_lines)]
fn fold_constants_inner(instructions: &[Instruction]) -> (Vec<Instruction>, ConstantFoldingReport) {
    let mut constants: HashMap<String, ImmValue> = HashMap::new();
    let mut result = Vec::with_capacity(instructions.len());
    let mut folded_count: usize = 0;
    let mut propagated_count: usize = 0;

    for inst in instructions {
        match inst {
            Instruction::Add { ty, dst, a, b } => {
                let (ra, pa) = resolve_with_propagation(a, &constants);
                let (rb, pb) = resolve_with_propagation(b, &constants);
                propagated_count += pa + pb;
                if let (Some(va), Some(vb)) = (ra, rb) {
                    if let Some(folded) = eval_add(*ty, &va, &vb) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Sub { ty, dst, a, b } => {
                let (ra, pa) = resolve_with_propagation(a, &constants);
                let (rb, pb) = resolve_with_propagation(b, &constants);
                propagated_count += pa + pb;
                if let (Some(va), Some(vb)) = (ra, rb) {
                    if let Some(folded) = eval_sub(*ty, &va, &vb) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Mul {
                ty,
                mode: MulMode::Lo,
                dst,
                a,
                b,
            } => {
                let (ra, pa) = resolve_with_propagation(a, &constants);
                let (rb, pb) = resolve_with_propagation(b, &constants);
                propagated_count += pa + pb;
                if let (Some(va), Some(vb)) = (ra, rb) {
                    if let Some(folded) = eval_mul_lo(*ty, &va, &vb) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Div { ty, dst, a, b } => {
                let (ra, pa) = resolve_with_propagation(a, &constants);
                let (rb, pb) = resolve_with_propagation(b, &constants);
                propagated_count += pa + pb;
                if let (Some(va), Some(vb)) = (ra, rb) {
                    if let Some(folded) = eval_div(*ty, &va, &vb) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::And { ty, dst, a, b } => {
                let (ra, pa) = resolve_with_propagation(a, &constants);
                let (rb, pb) = resolve_with_propagation(b, &constants);
                propagated_count += pa + pb;
                if let (Some(va), Some(vb)) = (ra, rb) {
                    if let Some(folded) = eval_and(*ty, &va, &vb) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Or { ty, dst, a, b } => {
                let (ra, pa) = resolve_with_propagation(a, &constants);
                let (rb, pb) = resolve_with_propagation(b, &constants);
                propagated_count += pa + pb;
                if let (Some(va), Some(vb)) = (ra, rb) {
                    if let Some(folded) = eval_or(*ty, &va, &vb) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Xor { ty, dst, a, b } => {
                let (ra, pa) = resolve_with_propagation(a, &constants);
                let (rb, pb) = resolve_with_propagation(b, &constants);
                propagated_count += pa + pb;
                if let (Some(va), Some(vb)) = (ra, rb) {
                    if let Some(folded) = eval_xor(*ty, &va, &vb) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Shl {
                ty,
                dst,
                src,
                amount,
            } => {
                let (rs, ps) = resolve_with_propagation(src, &constants);
                let (ra, pa) = resolve_with_propagation(amount, &constants);
                propagated_count += ps + pa;
                if let (Some(vs), Some(va)) = (rs, ra) {
                    if let Some(folded) = eval_shl(*ty, &vs, &va) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Shr {
                ty,
                dst,
                src,
                amount,
            } => {
                let (rs, ps) = resolve_with_propagation(src, &constants);
                let (ra, pa) = resolve_with_propagation(amount, &constants);
                propagated_count += ps + pa;
                if let (Some(vs), Some(va)) = (rs, ra) {
                    if let Some(folded) = eval_shr(*ty, &vs, &va) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Neg { ty, dst, src } => {
                let (rv, pv) = resolve_with_propagation(src, &constants);
                propagated_count += pv;
                if let Some(v) = rv {
                    if let Some(folded) = eval_neg(*ty, &v) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Abs { ty, dst, src } => {
                let (rv, pv) = resolve_with_propagation(src, &constants);
                propagated_count += pv;
                if let Some(v) = rv {
                    if let Some(folded) = eval_abs(*ty, &v) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Min { ty, dst, a, b } => {
                let (ra, pa) = resolve_with_propagation(a, &constants);
                let (rb, pb) = resolve_with_propagation(b, &constants);
                propagated_count += pa + pb;
                if let (Some(va), Some(vb)) = (ra, rb) {
                    if let Some(folded) = eval_min(*ty, &va, &vb) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Max { ty, dst, a, b } => {
                let (ra, pa) = resolve_with_propagation(a, &constants);
                let (rb, pb) = resolve_with_propagation(b, &constants);
                propagated_count += pa + pb;
                if let (Some(va), Some(vb)) = (ra, rb) {
                    if let Some(folded) = eval_max(*ty, &va, &vb) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::SetP { cmp, ty, dst, a, b } => {
                let (ra, pa) = resolve_with_propagation(a, &constants);
                let (rb, pb) = resolve_with_propagation(b, &constants);
                propagated_count += pa + pb;
                if let (Some(va), Some(vb)) = (ra, rb) {
                    if let Some(pred_val) = eval_setp(*cmp, *ty, &va, &vb) {
                        // Encode predicate as U32: 1 = true, 0 = false
                        let folded = ImmValue::U32(u32::from(pred_val));
                        emit_const_assign(&mut result, PtxType::U32, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Sqrt { ty, dst, src, .. } => {
                let (rv, pv) = resolve_with_propagation(src, &constants);
                propagated_count += pv;
                if let Some(v) = rv {
                    if let Some(folded) = eval_sqrt(*ty, &v) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Rcp { ty, dst, src, .. } => {
                let (rv, pv) = resolve_with_propagation(src, &constants);
                propagated_count += pv;
                if let Some(v) = rv {
                    if let Some(folded) = eval_rcp(*ty, &v) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Rsqrt { ty, dst, src, .. } => {
                let (rv, pv) = resolve_with_propagation(src, &constants);
                propagated_count += pv;
                if let Some(v) = rv {
                    if let Some(folded) = eval_rsqrt(*ty, &v) {
                        emit_const_assign(&mut result, *ty, dst, &folded, &mut constants);
                        folded_count += 1;
                        continue;
                    }
                }
                invalidate_dst(dst, &mut constants);
                result.push(inst.clone());
            }
            Instruction::Branch { target, predicate } => {
                // If the branch predicate is a known constant, the branch
                // condition can be evaluated at compile time. A constant-true
                // predicate becomes unconditional; constant-false eliminates
                // the branch entirely (dead-branch removal).
                if let Some((pred_reg, negated)) = predicate {
                    if let Some(imm) = constants.get(&pred_reg.name) {
                        let is_taken = if let ImmValue::U32(v) = imm {
                            let truthy = *v != 0;
                            if *negated { !truthy } else { truthy }
                        } else {
                            result.push(inst.clone());
                            continue;
                        };
                        if is_taken {
                            // Constant-taken: emit as unconditional branch.
                            result.push(Instruction::Branch {
                                target: target.clone(),
                                predicate: None,
                            });
                        }
                        // Constant-not-taken: drop the branch (dead code).
                        folded_count += 1;
                        continue;
                    }
                }
                result.push(inst.clone());
            }
            // Instructions that define a register but are not foldable
            _ => {
                if let Some(dst) = instruction_def(inst) {
                    constants.remove(&dst.name);
                }
                result.push(inst.clone());
            }
        }
    }

    let report = ConstantFoldingReport {
        folded_count,
        propagated_count,
    };
    (result, report)
}

// ---------------------------------------------------------------------------
// Branch elimination helper (public API)
// ---------------------------------------------------------------------------

/// Eliminate dead branches in a flat instruction list when the branch
/// predicate is a statically-known constant.
///
/// Operates on a single basic block's instruction list (not a full CFG).
/// For each [`Instruction::Branch`] whose predicate register holds a known
/// constant U32 value:
///
/// - If the branch is **always taken** (predicate evaluates to true), it is
///   replaced with an unconditional `Branch` (predicate removed).
/// - If the branch is **never taken** (predicate evaluates to false), it is
///   removed entirely.
///
/// Returns the number of branches that were eliminated or simplified.
pub fn fold_constant_branches(instructions: &mut Vec<Instruction>) -> usize {
    // First pass: build a constant table by forward-simulating the sequence.
    let (optimised, report) = fold_constants_inner(instructions);

    // Count how many Branch instructions were resolved during the inner pass.
    // The inner pass already emits simplified/dropped branches into `optimised`.
    let original_branch_count = instructions
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Branch {
                    predicate: Some(_),
                    ..
                }
            )
        })
        .count();
    let new_branch_count = optimised
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Branch {
                    predicate: Some(_),
                    ..
                }
            )
        })
        .count();

    *instructions = optimised;
    // folded_count counts SetP folds too; branch-specific delta is the
    // difference in predicated branch counts plus any outright removals.
    let branch_folds = original_branch_count.saturating_sub(new_branch_count);
    // If the report shows more total folds than just arithmetic, the
    // difference accounts for branch-specific eliminations.
    branch_folds.max(
        report
            .folded_count
            .saturating_sub(instructions.len().saturating_sub(original_branch_count)),
    )
}

// ---------------------------------------------------------------------------
// Constant tracking helpers
// ---------------------------------------------------------------------------

/// Resolve an operand to an immediate value, looking up register constants.
/// Also returns a propagation count: 1 if a register was resolved via the
/// constant table, 0 otherwise.
fn resolve_with_propagation(
    op: &Operand,
    constants: &HashMap<String, ImmValue>,
) -> (Option<ImmValue>, usize) {
    match op {
        Operand::Immediate(imm) => (Some(imm.clone()), 0),
        Operand::Register(reg) => constants
            .get(&reg.name)
            .map_or((None, 0), |val| (Some(val.clone()), 1)),
        Operand::Address { .. } | Operand::Symbol(_) => (None, 0),
    }
}

/// Record a constant assignment: emit `add.ty dst, result, 0` and track
/// the constant value for the destination register.
fn emit_const_assign(
    result: &mut Vec<Instruction>,
    ty: PtxType,
    dst: &Register,
    value: &ImmValue,
    constants: &mut HashMap<String, ImmValue>,
) {
    constants.insert(dst.name.clone(), value.clone());
    result.push(Instruction::Add {
        ty,
        dst: dst.clone(),
        a: Operand::Immediate(value.clone()),
        b: Operand::Immediate(zero_imm(ty)),
    });
}

/// Remove a register from the known constants table.
fn invalidate_dst(dst: &Register, constants: &mut HashMap<String, ImmValue>) {
    constants.remove(&dst.name);
}

/// Extract the destination register from an instruction, if any.
const fn instruction_def(inst: &Instruction) -> Option<&Register> {
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
        | Instruction::Div { dst, .. }
        | Instruction::Rem { dst, .. }
        | Instruction::And { dst, .. }
        | Instruction::Or { dst, .. }
        | Instruction::Xor { dst, .. }
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
        | Instruction::SetP { dst, .. }
        | Instruction::Load { dst, .. }
        | Instruction::Cvt { dst, .. }
        | Instruction::MovSpecial { dst, .. }
        | Instruction::LoadParam { dst, .. }
        | Instruction::Atom { dst, .. }
        | Instruction::AtomCas { dst, .. }
        | Instruction::Tex1d { dst, .. }
        | Instruction::Tex2d { dst, .. }
        | Instruction::Tex3d { dst, .. }
        | Instruction::SurfLoad { dst, .. } => Some(dst),
        _ => None,
    }
}

/// Return the zero immediate for a given type.
const fn zero_imm(ty: PtxType) -> ImmValue {
    match ty {
        PtxType::U64 | PtxType::B64 => ImmValue::U64(0),
        PtxType::S32 => ImmValue::S32(0),
        PtxType::S64 => ImmValue::S64(0),
        PtxType::F32 => ImmValue::F32(0.0),
        PtxType::F64 => ImmValue::F64(0.0),
        // For types we don't fold (incl. U32/B32), default to U32(0)
        _ => ImmValue::U32(0),
    }
}

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

fn eval_add(ty: PtxType, a: &ImmValue, b: &ImmValue) -> Option<ImmValue> {
    match (ty, a, b) {
        (PtxType::U32, ImmValue::U32(x), ImmValue::U32(y)) => {
            Some(ImmValue::U32(x.wrapping_add(*y)))
        }
        (PtxType::U64, ImmValue::U64(x), ImmValue::U64(y)) => {
            Some(ImmValue::U64(x.wrapping_add(*y)))
        }
        (PtxType::S32, ImmValue::S32(x), ImmValue::S32(y)) => {
            Some(ImmValue::S32(x.wrapping_add(*y)))
        }
        (PtxType::S64, ImmValue::S64(x), ImmValue::S64(y)) => {
            Some(ImmValue::S64(x.wrapping_add(*y)))
        }
        (PtxType::F32, ImmValue::F32(x), ImmValue::F32(y)) => Some(ImmValue::F32(x + y)),
        (PtxType::F64, ImmValue::F64(x), ImmValue::F64(y)) => Some(ImmValue::F64(x + y)),
        _ => None,
    }
}

fn eval_sub(ty: PtxType, a: &ImmValue, b: &ImmValue) -> Option<ImmValue> {
    match (ty, a, b) {
        (PtxType::U32, ImmValue::U32(x), ImmValue::U32(y)) => {
            Some(ImmValue::U32(x.wrapping_sub(*y)))
        }
        (PtxType::U64, ImmValue::U64(x), ImmValue::U64(y)) => {
            Some(ImmValue::U64(x.wrapping_sub(*y)))
        }
        (PtxType::S32, ImmValue::S32(x), ImmValue::S32(y)) => {
            Some(ImmValue::S32(x.wrapping_sub(*y)))
        }
        (PtxType::S64, ImmValue::S64(x), ImmValue::S64(y)) => {
            Some(ImmValue::S64(x.wrapping_sub(*y)))
        }
        (PtxType::F32, ImmValue::F32(x), ImmValue::F32(y)) => Some(ImmValue::F32(x - y)),
        (PtxType::F64, ImmValue::F64(x), ImmValue::F64(y)) => Some(ImmValue::F64(x - y)),
        _ => None,
    }
}

fn eval_mul_lo(ty: PtxType, a: &ImmValue, b: &ImmValue) -> Option<ImmValue> {
    match (ty, a, b) {
        (PtxType::U32, ImmValue::U32(x), ImmValue::U32(y)) => {
            Some(ImmValue::U32(x.wrapping_mul(*y)))
        }
        (PtxType::U64, ImmValue::U64(x), ImmValue::U64(y)) => {
            Some(ImmValue::U64(x.wrapping_mul(*y)))
        }
        (PtxType::S32, ImmValue::S32(x), ImmValue::S32(y)) => {
            Some(ImmValue::S32(x.wrapping_mul(*y)))
        }
        (PtxType::S64, ImmValue::S64(x), ImmValue::S64(y)) => {
            Some(ImmValue::S64(x.wrapping_mul(*y)))
        }
        (PtxType::F32, ImmValue::F32(x), ImmValue::F32(y)) => Some(ImmValue::F32(x * y)),
        (PtxType::F64, ImmValue::F64(x), ImmValue::F64(y)) => Some(ImmValue::F64(x * y)),
        _ => None,
    }
}

fn eval_neg(ty: PtxType, v: &ImmValue) -> Option<ImmValue> {
    match (ty, v) {
        (PtxType::S32, ImmValue::S32(x)) => Some(ImmValue::S32(x.wrapping_neg())),
        (PtxType::S64, ImmValue::S64(x)) => Some(ImmValue::S64(x.wrapping_neg())),
        (PtxType::F32, ImmValue::F32(x)) => Some(ImmValue::F32(-x)),
        (PtxType::F64, ImmValue::F64(x)) => Some(ImmValue::F64(-x)),
        _ => None,
    }
}

const fn eval_abs(ty: PtxType, v: &ImmValue) -> Option<ImmValue> {
    match (ty, v) {
        (PtxType::S32, ImmValue::S32(x)) => Some(ImmValue::S32(x.wrapping_abs())),
        (PtxType::S64, ImmValue::S64(x)) => Some(ImmValue::S64(x.wrapping_abs())),
        (PtxType::F32, ImmValue::F32(x)) => Some(ImmValue::F32(x.abs())),
        (PtxType::F64, ImmValue::F64(x)) => Some(ImmValue::F64(x.abs())),
        _ => None,
    }
}

fn eval_min(ty: PtxType, a: &ImmValue, b: &ImmValue) -> Option<ImmValue> {
    match (ty, a, b) {
        (PtxType::U32, ImmValue::U32(x), ImmValue::U32(y)) => Some(ImmValue::U32((*x).min(*y))),
        (PtxType::U64, ImmValue::U64(x), ImmValue::U64(y)) => Some(ImmValue::U64((*x).min(*y))),
        (PtxType::S32, ImmValue::S32(x), ImmValue::S32(y)) => Some(ImmValue::S32((*x).min(*y))),
        (PtxType::S64, ImmValue::S64(x), ImmValue::S64(y)) => Some(ImmValue::S64((*x).min(*y))),
        (PtxType::F32, ImmValue::F32(x), ImmValue::F32(y)) => Some(ImmValue::F32(x.min(*y))),
        (PtxType::F64, ImmValue::F64(x), ImmValue::F64(y)) => Some(ImmValue::F64(x.min(*y))),
        _ => None,
    }
}

fn eval_max(ty: PtxType, a: &ImmValue, b: &ImmValue) -> Option<ImmValue> {
    match (ty, a, b) {
        (PtxType::U32, ImmValue::U32(x), ImmValue::U32(y)) => Some(ImmValue::U32((*x).max(*y))),
        (PtxType::U64, ImmValue::U64(x), ImmValue::U64(y)) => Some(ImmValue::U64((*x).max(*y))),
        (PtxType::S32, ImmValue::S32(x), ImmValue::S32(y)) => Some(ImmValue::S32((*x).max(*y))),
        (PtxType::S64, ImmValue::S64(x), ImmValue::S64(y)) => Some(ImmValue::S64((*x).max(*y))),
        (PtxType::F32, ImmValue::F32(x), ImmValue::F32(y)) => Some(ImmValue::F32(x.max(*y))),
        (PtxType::F64, ImmValue::F64(x), ImmValue::F64(y)) => Some(ImmValue::F64(x.max(*y))),
        _ => None,
    }
}

fn eval_div(ty: PtxType, a: &ImmValue, b: &ImmValue) -> Option<ImmValue> {
    match (ty, a, b) {
        (PtxType::U32, ImmValue::U32(x), ImmValue::U32(y)) if *y != 0 => Some(ImmValue::U32(x / y)),
        (PtxType::U64, ImmValue::U64(x), ImmValue::U64(y)) if *y != 0 => Some(ImmValue::U64(x / y)),
        (PtxType::S32, ImmValue::S32(x), ImmValue::S32(y)) if *y != 0 => {
            Some(ImmValue::S32(x.wrapping_div(*y)))
        }
        (PtxType::S64, ImmValue::S64(x), ImmValue::S64(y)) if *y != 0 => {
            Some(ImmValue::S64(x.wrapping_div(*y)))
        }
        // PTX float division: do not fold if divisor is zero (avoids +Inf).
        (PtxType::F32, ImmValue::F32(x), ImmValue::F32(y)) if *y != 0.0 => {
            Some(ImmValue::F32(x / y))
        }
        (PtxType::F64, ImmValue::F64(x), ImmValue::F64(y)) if *y != 0.0 => {
            Some(ImmValue::F64(x / y))
        }
        _ => None,
    }
}

/// Evaluate `rcp` (reciprocal) of a float constant.
///
/// Returns `None` when the source is zero (would yield ±Infinity).
fn eval_rcp(ty: PtxType, v: &ImmValue) -> Option<ImmValue> {
    match (ty, v) {
        (PtxType::F32, ImmValue::F32(x)) if *x != 0.0 => Some(ImmValue::F32(1.0 / x)),
        (PtxType::F64, ImmValue::F64(x)) if *x != 0.0 => Some(ImmValue::F64(1.0 / x)),
        _ => None,
    }
}

/// Evaluate `rsqrt` (reciprocal square root) of a float constant.
///
/// Returns `None` when the source is negative or zero.
fn eval_rsqrt(ty: PtxType, v: &ImmValue) -> Option<ImmValue> {
    match (ty, v) {
        (PtxType::F32, ImmValue::F32(x)) if *x > 0.0 => Some(ImmValue::F32(1.0 / x.sqrt())),
        (PtxType::F64, ImmValue::F64(x)) if *x > 0.0 => Some(ImmValue::F64(1.0 / x.sqrt())),
        _ => None,
    }
}

/// Evaluate `sqrt` of a float constant.
///
/// Returns `None` when the source is negative (would yield NaN).
fn eval_sqrt(ty: PtxType, v: &ImmValue) -> Option<ImmValue> {
    match (ty, v) {
        (PtxType::F32, ImmValue::F32(x)) if *x >= 0.0 => Some(ImmValue::F32(x.sqrt())),
        (PtxType::F64, ImmValue::F64(x)) if *x >= 0.0 => Some(ImmValue::F64(x.sqrt())),
        _ => None,
    }
}

fn eval_and(ty: PtxType, a: &ImmValue, b: &ImmValue) -> Option<ImmValue> {
    match (ty, a, b) {
        (PtxType::U32 | PtxType::B32, ImmValue::U32(x), ImmValue::U32(y)) => {
            Some(ImmValue::U32(x & y))
        }
        (PtxType::U64 | PtxType::B64, ImmValue::U64(x), ImmValue::U64(y)) => {
            Some(ImmValue::U64(x & y))
        }
        (PtxType::S32, ImmValue::S32(x), ImmValue::S32(y)) => Some(ImmValue::S32(x & y)),
        (PtxType::S64, ImmValue::S64(x), ImmValue::S64(y)) => Some(ImmValue::S64(x & y)),
        _ => None,
    }
}

fn eval_or(ty: PtxType, a: &ImmValue, b: &ImmValue) -> Option<ImmValue> {
    match (ty, a, b) {
        (PtxType::U32 | PtxType::B32, ImmValue::U32(x), ImmValue::U32(y)) => {
            Some(ImmValue::U32(x | y))
        }
        (PtxType::U64 | PtxType::B64, ImmValue::U64(x), ImmValue::U64(y)) => {
            Some(ImmValue::U64(x | y))
        }
        (PtxType::S32, ImmValue::S32(x), ImmValue::S32(y)) => Some(ImmValue::S32(x | y)),
        (PtxType::S64, ImmValue::S64(x), ImmValue::S64(y)) => Some(ImmValue::S64(x | y)),
        _ => None,
    }
}

fn eval_xor(ty: PtxType, a: &ImmValue, b: &ImmValue) -> Option<ImmValue> {
    match (ty, a, b) {
        (PtxType::U32 | PtxType::B32, ImmValue::U32(x), ImmValue::U32(y)) => {
            Some(ImmValue::U32(x ^ y))
        }
        (PtxType::U64 | PtxType::B64, ImmValue::U64(x), ImmValue::U64(y)) => {
            Some(ImmValue::U64(x ^ y))
        }
        (PtxType::S32, ImmValue::S32(x), ImmValue::S32(y)) => Some(ImmValue::S32(x ^ y)),
        (PtxType::S64, ImmValue::S64(x), ImmValue::S64(y)) => Some(ImmValue::S64(x ^ y)),
        _ => None,
    }
}

const fn eval_shl(ty: PtxType, src: &ImmValue, amount: &ImmValue) -> Option<ImmValue> {
    let shift = match amount {
        ImmValue::U32(s) => *s,
        _ => return None,
    };
    match (ty, src) {
        (PtxType::U32 | PtxType::B32, ImmValue::U32(x)) => {
            Some(ImmValue::U32(x.wrapping_shl(shift)))
        }
        (PtxType::U64 | PtxType::B64, ImmValue::U64(x)) => {
            Some(ImmValue::U64(x.wrapping_shl(shift)))
        }
        (PtxType::S32, ImmValue::S32(x)) => Some(ImmValue::S32(x.wrapping_shl(shift))),
        (PtxType::S64, ImmValue::S64(x)) => Some(ImmValue::S64(x.wrapping_shl(shift))),
        _ => None,
    }
}

const fn eval_shr(ty: PtxType, src: &ImmValue, amount: &ImmValue) -> Option<ImmValue> {
    let shift = match amount {
        ImmValue::U32(s) => *s,
        _ => return None,
    };
    match (ty, src) {
        (PtxType::U32 | PtxType::B32, ImmValue::U32(x)) => {
            Some(ImmValue::U32(x.wrapping_shr(shift)))
        }
        (PtxType::U64 | PtxType::B64, ImmValue::U64(x)) => {
            Some(ImmValue::U64(x.wrapping_shr(shift)))
        }
        // Signed types use arithmetic shift right
        (PtxType::S32, ImmValue::S32(x)) => Some(ImmValue::S32(x.wrapping_shr(shift))),
        (PtxType::S64, ImmValue::S64(x)) => Some(ImmValue::S64(x.wrapping_shr(shift))),
        _ => None,
    }
}

/// Evaluate a comparison for constant operands. Returns `Some(bool)` if
/// the comparison can be evaluated, or `None` for unsupported type combinations.
fn eval_setp(cmp: CmpOp, ty: PtxType, a: &ImmValue, b: &ImmValue) -> Option<bool> {
    match (ty, a, b) {
        (PtxType::U32, ImmValue::U32(x), ImmValue::U32(y)) => eval_cmp_u32(cmp, *x, *y),
        (PtxType::S32, ImmValue::S32(x), ImmValue::S32(y)) => eval_cmp_s32(cmp, *x, *y),
        (PtxType::F32, ImmValue::F32(x), ImmValue::F32(y)) => eval_cmp_f32(cmp, *x, *y),
        (PtxType::F64, ImmValue::F64(x), ImmValue::F64(y)) => eval_cmp_f64(cmp, *x, *y),
        _ => None,
    }
}

const fn eval_cmp_u32(cmp: CmpOp, a: u32, b: u32) -> Option<bool> {
    match cmp {
        CmpOp::Eq => Some(a == b),
        CmpOp::Ne => Some(a != b),
        CmpOp::Lo => Some(a < b),
        CmpOp::Ls => Some(a <= b),
        CmpOp::Hi => Some(a > b),
        CmpOp::Hs => Some(a >= b),
        // Ordered comparisons for unsigned use Lo/Ls/Hi/Hs; Lt/Le/Gt/Ge
        // are technically signed in PTX, but we handle them conservatively.
        #[allow(clippy::cast_possible_wrap)]
        CmpOp::Lt => Some((a as i32) < (b as i32)),
        #[allow(clippy::cast_possible_wrap)]
        CmpOp::Le => Some((a as i32) <= (b as i32)),
        #[allow(clippy::cast_possible_wrap)]
        CmpOp::Gt => Some((a as i32) > (b as i32)),
        #[allow(clippy::cast_possible_wrap)]
        CmpOp::Ge => Some((a as i32) >= (b as i32)),
        _ => None,
    }
}

const fn eval_cmp_s32(cmp: CmpOp, a: i32, b: i32) -> Option<bool> {
    match cmp {
        CmpOp::Eq => Some(a == b),
        CmpOp::Ne => Some(a != b),
        CmpOp::Lt => Some(a < b),
        CmpOp::Le => Some(a <= b),
        CmpOp::Gt => Some(a > b),
        CmpOp::Ge => Some(a >= b),
        _ => None,
    }
}

#[allow(clippy::float_cmp)]
fn eval_cmp_f32(cmp: CmpOp, a: f32, b: f32) -> Option<bool> {
    match cmp {
        CmpOp::Eq => Some(a == b),
        CmpOp::Ne => Some(a != b),
        CmpOp::Lt => Some(a < b),
        CmpOp::Le => Some(a <= b),
        CmpOp::Gt => Some(a > b),
        CmpOp::Ge => Some(a >= b),
        // Unordered comparisons — true when either is NaN
        CmpOp::Equ => Some(a == b || a.is_nan() || b.is_nan()),
        CmpOp::Neu => Some(a != b || a.is_nan() || b.is_nan()),
        CmpOp::Ltu => Some(a < b || a.is_nan() || b.is_nan()),
        CmpOp::Leu => Some(a <= b || a.is_nan() || b.is_nan()),
        CmpOp::Gtu => Some(a > b || a.is_nan() || b.is_nan()),
        CmpOp::Geu => Some(a >= b || a.is_nan() || b.is_nan()),
        CmpOp::Num => Some(!a.is_nan() && !b.is_nan()),
        CmpOp::Nan => Some(a.is_nan() || b.is_nan()),
        _ => None,
    }
}

#[allow(clippy::float_cmp)]
fn eval_cmp_f64(cmp: CmpOp, a: f64, b: f64) -> Option<bool> {
    match cmp {
        CmpOp::Eq => Some(a == b),
        CmpOp::Ne => Some(a != b),
        CmpOp::Lt => Some(a < b),
        CmpOp::Le => Some(a <= b),
        CmpOp::Gt => Some(a > b),
        CmpOp::Ge => Some(a >= b),
        CmpOp::Equ => Some(a == b || a.is_nan() || b.is_nan()),
        CmpOp::Neu => Some(a != b || a.is_nan() || b.is_nan()),
        CmpOp::Ltu => Some(a < b || a.is_nan() || b.is_nan()),
        CmpOp::Leu => Some(a <= b || a.is_nan() || b.is_nan()),
        CmpOp::Gtu => Some(a > b || a.is_nan() || b.is_nan()),
        CmpOp::Geu => Some(a >= b || a.is_nan() || b.is_nan()),
        CmpOp::Num => Some(!a.is_nan() && !b.is_nan()),
        CmpOp::Nan => Some(a.is_nan() || b.is_nan()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ImmValue, Instruction, MulMode, Operand, PtxType, Register, SpecialReg};

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

    fn imm_s32(val: i32) -> Operand {
        Operand::Immediate(ImmValue::S32(val))
    }

    fn imm_f32(val: f32) -> Operand {
        Operand::Immediate(ImmValue::F32(val))
    }

    fn imm_f64(val: f64) -> Operand {
        Operand::Immediate(ImmValue::F64(val))
    }

    /// Helper to check that a folded instruction is `add.ty dst, imm, 0`.
    fn assert_folded_to_imm(inst: &Instruction, expected_imm: &ImmValue) {
        match inst {
            Instruction::Add { a, .. } => match a {
                Operand::Immediate(v) => {
                    assert_eq!(format!("{v}"), format!("{expected_imm}"));
                }
                other => panic!("Expected immediate, got {other:?}"),
            },
            other => panic!("Expected Add instruction, got {other:?}"),
        }
    }

    #[test]
    fn fold_add_u32_immediates() {
        let instructions = vec![Instruction::Add {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(3),
            b: imm_u32(5),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_eq!(result.len(), 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(8));
    }

    #[test]
    fn fold_add_f32_immediates() {
        let instructions = vec![Instruction::Add {
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            a: imm_f32(1.5),
            b: imm_f32(2.5),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::F32(4.0));
    }

    #[test]
    fn fold_sub_s32_immediates() {
        let instructions = vec![Instruction::Sub {
            ty: PtxType::S32,
            dst: reg("%r0", PtxType::S32),
            a: imm_s32(10),
            b: imm_s32(7),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::S32(3));
    }

    #[test]
    fn fold_mul_lo_u32_immediates() {
        let instructions = vec![Instruction::Mul {
            ty: PtxType::U32,
            mode: MulMode::Lo,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(6),
            b: imm_u32(7),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(42));
    }

    #[test]
    fn fold_neg_s32_immediate() {
        let instructions = vec![Instruction::Neg {
            ty: PtxType::S32,
            dst: reg("%r0", PtxType::S32),
            src: imm_s32(42),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::S32(-42));
    }

    #[test]
    fn fold_abs_f32_negative() {
        let instructions = vec![Instruction::Abs {
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            src: imm_f32(-3.19),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::F32(3.19));
    }

    #[test]
    fn fold_min_u32_immediates() {
        let instructions = vec![Instruction::Min {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(10),
            b: imm_u32(3),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(3));
    }

    #[test]
    fn fold_max_u32_immediates() {
        let instructions = vec![Instruction::Max {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(10),
            b: imm_u32(3),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(10));
    }

    #[test]
    fn non_constant_operands_not_folded() {
        let instructions = vec![Instruction::Add {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: reg_op("%r1", PtxType::U32),
            b: imm_u32(5),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn chain_folding_through_registers() {
        // %r0 = 3 + 5 (folded to 8)
        // %r1 = %r0 + 2 (folded to 10, since %r0 is known = 8)
        let instructions = vec![
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r0", PtxType::U32),
                a: imm_u32(3),
                b: imm_u32(5),
            },
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r1", PtxType::U32),
                a: reg_op("%r0", PtxType::U32),
                b: imm_u32(2),
            },
        ];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 2);
        assert_eq!(result.len(), 2);
        assert_folded_to_imm(&result[0], &ImmValue::U32(8));
        assert_folded_to_imm(&result[1], &ImmValue::U32(10));
    }

    #[test]
    fn empty_input_returns_empty() {
        let (result, count) = fold_constants(&[]);
        assert_eq!(count, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn side_effect_instructions_preserved() {
        // LoadParam sets dst to unknown, subsequent fold should not happen
        let instructions = vec![
            Instruction::LoadParam {
                ty: PtxType::U32,
                dst: reg("%r0", PtxType::U32),
                param_name: "%param_x".to_string(),
            },
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r1", PtxType::U32),
                a: reg_op("%r0", PtxType::U32),
                b: imm_u32(1),
            },
        ];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn fold_setp_eq_u32() {
        let instructions = vec![Instruction::SetP {
            cmp: CmpOp::Eq,
            ty: PtxType::U32,
            dst: reg("%p0", PtxType::Pred),
            a: imm_u32(5),
            b: imm_u32(5),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        // Should fold to true (1)
        assert_folded_to_imm(&result[0], &ImmValue::U32(1));
    }

    #[test]
    fn fold_setp_ne_u32() {
        let instructions = vec![Instruction::SetP {
            cmp: CmpOp::Ne,
            ty: PtxType::U32,
            dst: reg("%p0", PtxType::Pred),
            a: imm_u32(5),
            b: imm_u32(5),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(0));
    }

    #[test]
    fn fold_u32_overflow_wrapping() {
        let instructions = vec![Instruction::Add {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(u32::MAX),
            b: imm_u32(1),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(0));
    }

    #[test]
    fn fold_f32_nan_propagation() {
        let instructions = vec![Instruction::Add {
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            a: imm_f32(f32::NAN),
            b: imm_f32(1.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        // NaN + 1.0 = NaN
        if let Instruction::Add {
            a: Operand::Immediate(ImmValue::F32(v)),
            ..
        } = &result[0]
        {
            assert!(v.is_nan());
        } else {
            panic!("Expected folded F32 NaN");
        }
    }

    #[test]
    fn fold_mul_hi_mode_not_folded() {
        // Only Lo mode is folded
        let instructions = vec![Instruction::Mul {
            ty: PtxType::U32,
            mode: MulMode::Hi,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(6),
            b: imm_u32(7),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn fold_f64_subtraction() {
        let instructions = vec![Instruction::Sub {
            ty: PtxType::F64,
            dst: reg("%fd0", PtxType::F64),
            a: imm_f64(10.0),
            b: imm_f64(3.5),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::F64(6.5));
    }

    #[test]
    fn register_overwrite_invalidates_constant() {
        // %r0 = 3 + 5 (folded, %r0 = 8)
        // %r0 = mov_special (unknown — overwrites %r0)
        // %r1 = %r0 + 2 (NOT folded — %r0 is now unknown)
        let instructions = vec![
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r0", PtxType::U32),
                a: imm_u32(3),
                b: imm_u32(5),
            },
            Instruction::MovSpecial {
                dst: reg("%r0", PtxType::U32),
                special: SpecialReg::TidX,
            },
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r1", PtxType::U32),
                a: reg_op("%r0", PtxType::U32),
                b: imm_u32(2),
            },
        ];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1); // Only the first add is folded
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn fold_neg_f64_negative_zero() {
        let instructions = vec![Instruction::Neg {
            ty: PtxType::F64,
            dst: reg("%fd0", PtxType::F64),
            src: Operand::Immediate(ImmValue::F64(0.0)),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        // neg(0.0) = -0.0
        if let Instruction::Add {
            a: Operand::Immediate(ImmValue::F64(v)),
            ..
        } = &result[0]
        {
            assert!(v.is_sign_negative());
            assert!((*v + 0.0_f64).abs() < f64::EPSILON);
        } else {
            panic!("Expected folded F64 -0.0");
        }
    }

    #[test]
    fn fold_min_f32_with_nan() {
        let instructions = vec![Instruction::Min {
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            a: imm_f32(f32::NAN),
            b: imm_f32(1.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        // f32::min(NAN, 1.0) = 1.0 (propagates non-NaN per IEEE)
        assert_folded_to_imm(&result[0], &ImmValue::F32(1.0));
    }

    // -- Div folding --------------------------------------------------------

    #[test]
    fn fold_div_u32_immediates() {
        let instructions = vec![Instruction::Div {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(42),
            b: imm_u32(7),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(6));
    }

    #[test]
    fn fold_div_by_zero_not_folded() {
        let instructions = vec![Instruction::Div {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(42),
            b: imm_u32(0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    // -- And/Or/Xor folding -------------------------------------------------

    #[test]
    fn fold_and_u32_immediates() {
        let instructions = vec![Instruction::And {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(0xFF00),
            b: imm_u32(0x0F0F),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(0x0F00));
    }

    #[test]
    fn fold_or_u32_immediates() {
        let instructions = vec![Instruction::Or {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(0xF0),
            b: imm_u32(0x0F),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(0xFF));
    }

    #[test]
    fn fold_xor_u32_immediates() {
        let instructions = vec![Instruction::Xor {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(0xFF),
            b: imm_u32(0xFF),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(0));
    }

    // -- Shl/Shr folding ----------------------------------------------------

    #[test]
    fn fold_shl_u32_immediates() {
        let instructions = vec![Instruction::Shl {
            ty: PtxType::B32,
            dst: reg("%r0", PtxType::U32),
            src: imm_u32(1),
            amount: imm_u32(4),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(16));
    }

    #[test]
    fn fold_shr_u32_immediates() {
        let instructions = vec![Instruction::Shr {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            src: imm_u32(256),
            amount: imm_u32(3),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(32));
    }

    // -- Report tests -------------------------------------------------------

    #[test]
    fn report_tracks_propagation() {
        // %r0 = 3 + 5 → folded (8), 0 propagations
        // %r1 = %r0 + 2 → folded (10), 1 propagation (from %r0)
        let instructions = vec![
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r0", PtxType::U32),
                a: imm_u32(3),
                b: imm_u32(5),
            },
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r1", PtxType::U32),
                a: reg_op("%r0", PtxType::U32),
                b: imm_u32(2),
            },
        ];
        let report = fold_constants_report(&instructions);
        assert_eq!(report.folded_count, 2);
        assert_eq!(report.propagated_count, 1);
    }

    #[test]
    fn report_empty_input() {
        let report = fold_constants_report(&[]);
        assert_eq!(report.folded_count, 0);
        assert_eq!(report.propagated_count, 0);
    }

    // -- Float div folding ----------------------------------------------------

    #[test]
    fn test_fold_float_add() {
        // 1.0 + 2.0 = 3.0
        let instructions = vec![Instruction::Add {
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            a: imm_f32(1.0),
            b: imm_f32(2.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::F32(3.0));
    }

    #[test]
    fn test_fold_float_div_f32() {
        // 9.0 / 3.0 = 3.0
        let instructions = vec![Instruction::Div {
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            a: imm_f32(9.0),
            b: imm_f32(3.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::F32(3.0));
    }

    #[test]
    fn test_fold_float_div_by_zero_not_folded() {
        // 9.0 / 0.0 must NOT be folded (PTX handles division by zero
        // specially; we conservatively skip it).
        let instructions = vec![Instruction::Div {
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            a: imm_f32(9.0),
            b: imm_f32(0.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    // -- Sqrt / Rcp / Rsqrt folding -------------------------------------------

    #[test]
    fn test_fold_float_sqrt_positive() {
        // sqrt(4.0) = 2.0
        let instructions = vec![Instruction::Sqrt {
            rnd: None,
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            src: imm_f32(4.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::F32(2.0));
    }

    #[test]
    fn test_fold_float_sqrt_negative_not_folded() {
        // sqrt(-1.0) should NOT be folded.
        let instructions = vec![Instruction::Sqrt {
            rnd: None,
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            src: imm_f32(-1.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_fold_rcp_f32() {
        // rcp(4.0) = 0.25
        let instructions = vec![Instruction::Rcp {
            rnd: None,
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            src: imm_f32(4.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::F32(0.25));
    }

    #[test]
    fn test_fold_rcp_zero_not_folded() {
        // rcp(0.0) must NOT be folded.
        let instructions = vec![Instruction::Rcp {
            rnd: None,
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            src: imm_f32(0.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_fold_rsqrt_f32() {
        // rsqrt(4.0) = 0.5
        let instructions = vec![Instruction::Rsqrt {
            approx: true,
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            src: imm_f32(4.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::F32(0.5));
    }

    // -- Comparison folding ---------------------------------------------------

    #[test]
    fn test_fold_comparison_equal_integers() {
        // 5 == 5 → true (1)
        let instructions = vec![Instruction::SetP {
            cmp: CmpOp::Eq,
            ty: PtxType::S32,
            dst: reg("%p0", PtxType::Pred),
            a: imm_s32(5),
            b: imm_s32(5),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(1));
    }

    #[test]
    fn test_fold_comparison_not_equal_integers() {
        // 3 == 7 → false (0)
        let instructions = vec![Instruction::SetP {
            cmp: CmpOp::Eq,
            ty: PtxType::S32,
            dst: reg("%p0", PtxType::Pred),
            a: imm_s32(3),
            b: imm_s32(7),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(0));
    }

    #[test]
    fn test_fold_comparison_float_less_than() {
        // 1.0 < 2.0 → true (1)
        let instructions = vec![Instruction::SetP {
            cmp: CmpOp::Lt,
            ty: PtxType::F32,
            dst: reg("%p0", PtxType::Pred),
            a: imm_f32(1.0),
            b: imm_f32(2.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::U32(1));
    }

    // -- Branch elimination ---------------------------------------------------

    #[test]
    fn test_fold_constant_branch_eliminated() {
        // %p = (5 == 7) = false → branch @%p is dead and removed.
        let mut instructions = vec![
            // First fold SetP: false (0) → %p0 = U32(0).
            Instruction::SetP {
                cmp: CmpOp::Eq,
                ty: PtxType::U32,
                dst: reg("%p0", PtxType::Pred),
                a: imm_u32(5),
                b: imm_u32(7),
            },
            // Branch taken when %p0 is true — never taken, so eliminated.
            Instruction::Branch {
                target: "dead_target".to_string(),
                predicate: Some((reg("%p0", PtxType::Pred), false)),
            },
        ];
        let eliminated = fold_constant_branches(&mut instructions);
        // The dead branch should be eliminated.
        assert!(eliminated > 0, "expected at least one branch eliminated");
        assert!(
            instructions.iter().all(
                |i| !matches!(i, Instruction::Branch { target, .. } if target == "dead_target")
            ),
            "dead branch should have been removed"
        );
    }

    #[test]
    fn test_fold_constant_branch_taken_branch_kept() {
        // %p = (5 == 5) = true → branch @%p is always taken, becomes unconditional.
        let mut instructions = vec![
            Instruction::SetP {
                cmp: CmpOp::Eq,
                ty: PtxType::U32,
                dst: reg("%p0", PtxType::Pred),
                a: imm_u32(5),
                b: imm_u32(5),
            },
            Instruction::Branch {
                target: "always_taken".to_string(),
                predicate: Some((reg("%p0", PtxType::Pred), false)),
            },
        ];
        fold_constant_branches(&mut instructions);
        // Branch to "always_taken" should now be unconditional (predicate = None).
        let has_unconditional = instructions.iter().any(|i| {
            matches!(
                i,
                Instruction::Branch {
                    target,
                    predicate: None,
                } if target == "always_taken"
            )
        });
        assert!(
            has_unconditional,
            "expected unconditional branch to 'always_taken'"
        );
    }

    #[test]
    fn test_fold_float_div_f64() {
        // 10.0 / 4.0 = 2.5
        let instructions = vec![Instruction::Div {
            ty: PtxType::F64,
            dst: reg("%fd0", PtxType::F64),
            a: imm_f64(10.0),
            b: imm_f64(4.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::F64(2.5));
    }

    #[test]
    fn test_fold_sqrt_f64_positive() {
        // sqrt(9.0) = 3.0 (f64)
        let instructions = vec![Instruction::Sqrt {
            rnd: None,
            ty: PtxType::F64,
            dst: reg("%fd0", PtxType::F64),
            src: imm_f64(9.0),
        }];
        let (result, count) = fold_constants(&instructions);
        assert_eq!(count, 1);
        assert_folded_to_imm(&result[0], &ImmValue::F64(3.0));
    }
}
