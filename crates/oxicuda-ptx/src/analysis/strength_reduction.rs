//! Strength reduction optimization pass for PTX instruction sequences.
//!
//! Replaces expensive operations with cheaper equivalents:
//! - Multiply by power of 2 → left shift
//! - Multiply by 0 → move zero
//! - Multiply by 1 → identity copy
//! - Multiply by -1 (signed) → negation
//! - Unsigned divide by power of 2 → right shift
//! - Unsigned remainder by power of 2 → bitwise AND with mask
//! - Add 0 → identity copy

use crate::ir::{ImmValue, Instruction, MulMode, Operand, PtxType, Register};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Report summarizing the strength reduction pass results.
#[derive(Debug, Clone)]
pub struct StrengthReductionReport {
    /// Total number of instructions that were strength-reduced.
    pub reductions: usize,
    /// Human-readable description of each reduction applied.
    pub details: Vec<String>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply strength reduction optimizations to an instruction sequence.
///
/// Replaces expensive operations with cheaper equivalents:
/// - Multiply by power of 2 → left shift (`shl`)
/// - Multiply by 0 → move 0 (`add dst, 0, 0`)
/// - Multiply by 1 → identity copy (`add dst, src, 0`)
/// - Multiply by -1 (signed) → negation (`neg`)
/// - Unsigned divide by power of 2 → right shift (`shr`)
/// - Unsigned remainder by power of 2 → bitwise AND (`and dst, src, 2^n - 1`)
/// - Add by 0 → identity copy (`add dst, src, 0`)
///
/// Only integer instructions are candidates for reduction.
/// Floating-point operations are not reduced.
///
/// Returns optimized instructions and count of reduced instructions.
pub fn reduce_strength(instructions: &[Instruction]) -> (Vec<Instruction>, usize) {
    let report = reduce_strength_report(instructions);
    let count = report.reductions;
    let (result, _) = reduce_strength_inner(instructions);
    (result, count)
}

/// Apply strength reduction and return a detailed report.
pub fn reduce_strength_report(instructions: &[Instruction]) -> StrengthReductionReport {
    let (_, report) = reduce_strength_inner(instructions);
    report
}

/// Inner implementation returning both the optimized instructions and report.
fn reduce_strength_inner(
    instructions: &[Instruction],
) -> (Vec<Instruction>, StrengthReductionReport) {
    let mut result = Vec::with_capacity(instructions.len());
    let mut reductions: usize = 0;
    let mut details: Vec<String> = Vec::new();

    for inst in instructions {
        match inst {
            // -- Multiply reductions ----------------------------------------
            Instruction::Mul {
                ty,
                mode: MulMode::Lo,
                dst,
                a,
                b,
            } if ty.is_integer() => {
                if let Some((replacement, desc)) = try_reduce_mul_detailed(*ty, dst, a, b) {
                    result.push(replacement);
                    details.push(desc);
                    reductions += 1;
                    continue;
                }
                // Also check with operands swapped (commutativity)
                if let Some((replacement, desc)) = try_reduce_mul_detailed(*ty, dst, b, a) {
                    result.push(replacement);
                    details.push(desc);
                    reductions += 1;
                    continue;
                }
                result.push(inst.clone());
            }
            // -- Unsigned divide by power of 2 ------------------------------
            Instruction::Div { ty, dst, a, b } if is_unsigned_integer(*ty) => {
                if let Some((replacement, desc)) = try_reduce_div(*ty, dst, a, b) {
                    result.push(replacement);
                    details.push(desc);
                    reductions += 1;
                    continue;
                }
                result.push(inst.clone());
            }
            // -- Unsigned remainder by power of 2 ---------------------------
            Instruction::Rem { ty, dst, a, b } if is_unsigned_integer(*ty) => {
                if let Some((replacement, desc)) = try_reduce_rem(*ty, dst, a, b) {
                    result.push(replacement);
                    details.push(desc);
                    reductions += 1;
                    continue;
                }
                result.push(inst.clone());
            }
            // -- Add by 0 ---------------------------------------------------
            Instruction::Add { ty, dst, a, b } if ty.is_integer() => {
                if let Some((replacement, desc)) = try_reduce_add(*ty, dst, a, b) {
                    result.push(replacement);
                    details.push(desc);
                    reductions += 1;
                    continue;
                }
                result.push(inst.clone());
            }
            _ => {
                result.push(inst.clone());
            }
        }
    }

    let report = StrengthReductionReport {
        reductions,
        details,
    };
    (result, report)
}

// ---------------------------------------------------------------------------
// Reduction logic
// ---------------------------------------------------------------------------

/// Returns true for unsigned integer types that are safe for shift/mask
/// strength reduction of div/rem.
const fn is_unsigned_integer(ty: PtxType) -> bool {
    matches!(ty, PtxType::U8 | PtxType::U16 | PtxType::U32 | PtxType::U64)
}

/// Try to reduce a `mul.lo.ty dst, src, val` where `val` is the second operand.
/// Returns `Some((replacement, description))` if the multiply can be reduced.
fn try_reduce_mul_detailed(
    ty: PtxType,
    dst: &Register,
    src: &Operand,
    val: &Operand,
) -> Option<(Instruction, String)> {
    match val {
        Operand::Immediate(imm) => try_reduce_imm_detailed(ty, dst, src, imm),
        _ => None,
    }
}

/// Try to reduce a `div.ty dst, a, b` for unsigned power-of-2 divisors.
fn try_reduce_div(
    ty: PtxType,
    dst: &Register,
    a: &Operand,
    b: &Operand,
) -> Option<(Instruction, String)> {
    let shift = match b {
        Operand::Immediate(ImmValue::U32(v)) => log2_u32(*v),
        Operand::Immediate(ImmValue::U64(v)) => log2_u64(*v),
        _ => None,
    }?;
    let shr_ty = shr_ty_for(ty);
    let desc = format!("div by 2^{shift} -> shr by {shift}");
    Some((make_shr(shr_ty, dst, a, shift), desc))
}

/// Try to reduce a `rem.ty dst, a, b` for unsigned power-of-2 divisors.
fn try_reduce_rem(
    ty: PtxType,
    dst: &Register,
    a: &Operand,
    b: &Operand,
) -> Option<(Instruction, String)> {
    match b {
        Operand::Immediate(ImmValue::U32(v)) => log2_u32(*v).map(|_| {
            let mask = v - 1;
            let desc = format!("rem by {v} -> and with 0x{mask:x}");
            (make_and_u32(ty, dst, a, mask), desc)
        }),
        Operand::Immediate(ImmValue::U64(v)) => log2_u64(*v).map(|_| {
            let mask = v - 1;
            let desc = format!("rem by {v} -> and with 0x{mask:x}");
            (make_and_u64(ty, dst, a, mask), desc)
        }),
        _ => None,
    }
}

/// Try to reduce an `add.ty dst, a, b` when one operand is 0.
fn try_reduce_add(
    ty: PtxType,
    dst: &Register,
    a: &Operand,
    b: &Operand,
) -> Option<(Instruction, String)> {
    if is_zero_imm(b) {
        let desc = "add x, 0 -> mov dst, x".to_string();
        return Some((make_identity(ty, dst, a), desc));
    }
    if is_zero_imm(a) {
        let desc = "add 0, x -> mov dst, x".to_string();
        return Some((make_identity(ty, dst, b), desc));
    }
    None
}

/// Check if an operand is an immediate zero.
const fn is_zero_imm(op: &Operand) -> bool {
    matches!(
        op,
        Operand::Immediate(
            ImmValue::U32(0) | ImmValue::U64(0) | ImmValue::S32(0) | ImmValue::S64(0)
        )
    )
}

fn try_reduce_imm_detailed(
    ty: PtxType,
    dst: &Register,
    src: &Operand,
    imm: &ImmValue,
) -> Option<(Instruction, String)> {
    match imm {
        ImmValue::U32(v) => {
            reduce_u32(ty, dst, src, *v).map(|inst| (inst, reduce_description_u32(*v)))
        }
        ImmValue::U64(v) => {
            reduce_u64(ty, dst, src, *v).map(|inst| (inst, reduce_description_u64(*v)))
        }
        ImmValue::S32(v) => {
            reduce_s32(ty, dst, src, *v).map(|inst| (inst, reduce_description_s32(*v)))
        }
        ImmValue::S64(v) => {
            reduce_s64(ty, dst, src, *v).map(|inst| (inst, reduce_description_s64(*v)))
        }
        ImmValue::F32(_) | ImmValue::F64(_) => None,
    }
}

fn reduce_description_u32(val: u32) -> String {
    if val == 0 {
        "mul by 0 -> mov 0".to_string()
    } else if val == 1 {
        "mul by 1 -> identity".to_string()
    } else {
        format!("mul by {} -> shl by {}", val, val.trailing_zeros())
    }
}

fn reduce_description_u64(val: u64) -> String {
    if val == 0 {
        "mul by 0 -> mov 0".to_string()
    } else if val == 1 {
        "mul by 1 -> identity".to_string()
    } else {
        format!("mul by {} -> shl by {}", val, val.trailing_zeros())
    }
}

fn reduce_description_s32(val: i32) -> String {
    if val == 0 {
        "mul by 0 -> mov 0".to_string()
    } else if val == 1 {
        "mul by 1 -> identity".to_string()
    } else if val == -1 {
        "mul by -1 -> neg".to_string()
    } else {
        format!("mul by {val} -> shl")
    }
}

fn reduce_description_s64(val: i64) -> String {
    if val == 0 {
        "mul by 0 -> mov 0".to_string()
    } else if val == 1 {
        "mul by 1 -> identity".to_string()
    } else if val == -1 {
        "mul by -1 -> neg".to_string()
    } else {
        format!("mul by {val} -> shl")
    }
}

fn reduce_u32(ty: PtxType, dst: &Register, src: &Operand, val: u32) -> Option<Instruction> {
    if val == 0 {
        return Some(make_zero_move(ty, dst));
    }
    if val == 1 {
        return Some(make_identity(ty, dst, src));
    }
    if let Some(shift) = log2_u32(val) {
        return Some(make_shl(shl_ty_for(ty), dst, src, shift));
    }
    None
}

fn reduce_u64(ty: PtxType, dst: &Register, src: &Operand, val: u64) -> Option<Instruction> {
    if val == 0 {
        return Some(make_zero_move(ty, dst));
    }
    if val == 1 {
        return Some(make_identity(ty, dst, src));
    }
    if let Some(shift) = log2_u64(val) {
        return Some(make_shl(shl_ty_for(ty), dst, src, shift));
    }
    None
}

fn reduce_s32(ty: PtxType, dst: &Register, src: &Operand, val: i32) -> Option<Instruction> {
    if val == 0 {
        return Some(make_zero_move(ty, dst));
    }
    if val == 1 {
        return Some(make_identity(ty, dst, src));
    }
    if val == -1 {
        return Some(make_neg(ty, dst, src));
    }
    // Check positive power of 2
    if val > 0 {
        #[allow(clippy::cast_sign_loss)]
        if let Some(shift) = log2_u32(val as u32) {
            return Some(make_shl(shl_ty_for(ty), dst, src, shift));
        }
    }
    None
}

fn reduce_s64(ty: PtxType, dst: &Register, src: &Operand, val: i64) -> Option<Instruction> {
    if val == 0 {
        return Some(make_zero_move(ty, dst));
    }
    if val == 1 {
        return Some(make_identity(ty, dst, src));
    }
    if val == -1 {
        return Some(make_neg(ty, dst, src));
    }
    if val > 0 {
        #[allow(clippy::cast_sign_loss)]
        if let Some(shift) = log2_u64(val as u64) {
            return Some(make_shl(shl_ty_for(ty), dst, src, shift));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Helper constructors
// ---------------------------------------------------------------------------

/// `add.ty dst, 0, 0` — move zero into dst.
fn make_zero_move(ty: PtxType, dst: &Register) -> Instruction {
    let zero = zero_imm(ty);
    Instruction::Add {
        ty,
        dst: dst.clone(),
        a: Operand::Immediate(zero.clone()),
        b: Operand::Immediate(zero),
    }
}

/// `add.ty dst, src, 0` — identity copy.
fn make_identity(ty: PtxType, dst: &Register, src: &Operand) -> Instruction {
    Instruction::Add {
        ty,
        dst: dst.clone(),
        a: src.clone(),
        b: Operand::Immediate(zero_imm(ty)),
    }
}

/// `neg.ty dst, src` — negate.
fn make_neg(ty: PtxType, dst: &Register, src: &Operand) -> Instruction {
    Instruction::Neg {
        ty,
        dst: dst.clone(),
        src: src.clone(),
    }
}

/// `shl.ty dst, src, shift` — left shift.
fn make_shl(ty: PtxType, dst: &Register, src: &Operand, shift: u32) -> Instruction {
    Instruction::Shl {
        ty,
        dst: dst.clone(),
        src: src.clone(),
        amount: Operand::Immediate(ImmValue::U32(shift)),
    }
}

/// `shr.ty dst, src, shift` — right shift.
fn make_shr(ty: PtxType, dst: &Register, src: &Operand, shift: u32) -> Instruction {
    Instruction::Shr {
        ty,
        dst: dst.clone(),
        src: src.clone(),
        amount: Operand::Immediate(ImmValue::U32(shift)),
    }
}

/// `and.b32 dst, src, mask` — bitwise AND with u32 mask.
fn make_and_u32(ty: PtxType, dst: &Register, src: &Operand, mask: u32) -> Instruction {
    let and_ty = match ty {
        PtxType::U64 | PtxType::B64 => PtxType::B64,
        _ => PtxType::B32,
    };
    Instruction::And {
        ty: and_ty,
        dst: dst.clone(),
        a: src.clone(),
        b: Operand::Immediate(ImmValue::U32(mask)),
    }
}

/// `and.b64 dst, src, mask` — bitwise AND with u64 mask.
fn make_and_u64(ty: PtxType, dst: &Register, src: &Operand, mask: u64) -> Instruction {
    let and_ty = match ty {
        PtxType::U32 | PtxType::B32 => PtxType::B32,
        _ => PtxType::B64,
    };
    Instruction::And {
        ty: and_ty,
        dst: dst.clone(),
        a: src.clone(),
        b: Operand::Immediate(ImmValue::U64(mask)),
    }
}

/// Determine the appropriate bit type for shift operations.
const fn shl_ty_for(ty: PtxType) -> PtxType {
    match ty {
        PtxType::U32 | PtxType::S32 | PtxType::B32 => PtxType::B32,
        PtxType::U64 | PtxType::S64 | PtxType::B64 => PtxType::B64,
        other => other,
    }
}

/// Determine the appropriate type for right shift in strength reduction.
const fn shr_ty_for(ty: PtxType) -> PtxType {
    match ty {
        PtxType::U32 | PtxType::B32 => PtxType::U32,
        PtxType::U64 | PtxType::B64 => PtxType::U64,
        other => other,
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
        _ => ImmValue::U32(0),
    }
}

// ---------------------------------------------------------------------------
// Power-of-two detection
// ---------------------------------------------------------------------------

/// If `val` is a power of two (and > 0), return `log2(val)`.
const fn log2_u32(val: u32) -> Option<u32> {
    if val == 0 || (val & (val - 1)) != 0 {
        return None;
    }
    Some(val.trailing_zeros())
}

/// If `val` is a power of two (and > 0), return `log2(val)`.
const fn log2_u64(val: u64) -> Option<u32> {
    if val == 0 || (val & (val - 1)) != 0 {
        return None;
    }
    Some(val.trailing_zeros())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ImmValue, Instruction, MulMode, Operand, PtxType, Register};

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

    fn imm_u64(val: u64) -> Operand {
        Operand::Immediate(ImmValue::U64(val))
    }

    fn make_mul_lo_u32(dst_name: &str, src_name: &str, val: u32) -> Instruction {
        Instruction::Mul {
            ty: PtxType::U32,
            mode: MulMode::Lo,
            dst: reg(dst_name, PtxType::U32),
            a: reg_op(src_name, PtxType::U32),
            b: imm_u32(val),
        }
    }

    #[test]
    fn multiply_by_power_of_2_becomes_shift() {
        let instructions = vec![make_mul_lo_u32("%r0", "%r1", 8)];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 1);
        assert_eq!(result.len(), 1);
        match &result[0] {
            Instruction::Shl { ty, amount, .. } => {
                assert_eq!(*ty, PtxType::B32);
                assert!(matches!(amount, Operand::Immediate(ImmValue::U32(3))));
            }
            other => panic!("Expected Shl, got {other:?}"),
        }
    }

    #[test]
    fn multiply_by_zero_becomes_zero_move() {
        let instructions = vec![make_mul_lo_u32("%r0", "%r1", 0)];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 1);
        match &result[0] {
            Instruction::Add { a, b, .. } => {
                assert!(matches!(a, Operand::Immediate(ImmValue::U32(0))));
                assert!(matches!(b, Operand::Immediate(ImmValue::U32(0))));
            }
            other => panic!("Expected zero move (Add 0,0), got {other:?}"),
        }
    }

    #[test]
    fn multiply_by_one_becomes_identity() {
        let instructions = vec![make_mul_lo_u32("%r0", "%r1", 1)];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 1);
        match &result[0] {
            Instruction::Add { a, b, .. } => {
                assert!(matches!(a, Operand::Register(_)));
                assert!(matches!(b, Operand::Immediate(ImmValue::U32(0))));
            }
            other => panic!("Expected identity (Add src, 0), got {other:?}"),
        }
    }

    #[test]
    fn non_power_of_two_multiply_unchanged() {
        let instructions = vec![make_mul_lo_u32("%r0", "%r1", 7)];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
        assert!(matches!(&result[0], Instruction::Mul { .. }));
    }

    #[test]
    fn multiply_by_neg_one_becomes_neg() {
        let instructions = vec![Instruction::Mul {
            ty: PtxType::S32,
            mode: MulMode::Lo,
            dst: reg("%r0", PtxType::S32),
            a: reg_op("%r1", PtxType::S32),
            b: imm_s32(-1),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 1);
        assert!(matches!(&result[0], Instruction::Neg { .. }));
    }

    #[test]
    fn various_powers_of_two() {
        for (val, expected_shift) in [
            (2u32, 1u32),
            (4, 2),
            (16, 4),
            (32, 5),
            (64, 6),
            (128, 7),
            (256, 8),
            (512, 9),
            (1024, 10),
        ] {
            let instructions = vec![make_mul_lo_u32("%r0", "%r1", val)];
            let (result, count) = reduce_strength(&instructions);
            assert_eq!(count, 1, "power of 2 = {val}");
            match &result[0] {
                Instruction::Shl { amount, .. } => {
                    assert!(
                        matches!(amount, Operand::Immediate(ImmValue::U32(s)) if *s == expected_shift),
                        "Expected shift {expected_shift} for val {val}"
                    );
                }
                other => panic!("Expected Shl for val {val}, got {other:?}"),
            }
        }
    }

    #[test]
    fn f32_multiply_not_reduced() {
        // Float multiplies should NOT be strength-reduced
        let instructions = vec![Instruction::Mul {
            ty: PtxType::F32,
            mode: MulMode::Lo,
            dst: reg("%f0", PtxType::F32),
            a: reg_op("%f1", PtxType::F32),
            b: Operand::Immediate(ImmValue::F32(2.0)),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn empty_input_returns_empty() {
        let (result, count) = reduce_strength(&[]);
        assert_eq!(count, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn u64_shift_for_power_of_two() {
        let instructions = vec![Instruction::Mul {
            ty: PtxType::U64,
            mode: MulMode::Lo,
            dst: reg("%rd0", PtxType::U64),
            a: reg_op("%rd1", PtxType::U64),
            b: imm_u64(16),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 1);
        match &result[0] {
            Instruction::Shl { ty, amount, .. } => {
                assert_eq!(*ty, PtxType::B64);
                assert!(matches!(amount, Operand::Immediate(ImmValue::U32(4))));
            }
            other => panic!("Expected Shl.b64, got {other:?}"),
        }
    }

    #[test]
    fn multiply_hi_mode_not_reduced() {
        let instructions = vec![Instruction::Mul {
            ty: PtxType::U32,
            mode: MulMode::Hi,
            dst: reg("%r0", PtxType::U32),
            a: reg_op("%r1", PtxType::U32),
            b: imm_u32(8),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn multiply_wide_mode_not_reduced() {
        let instructions = vec![Instruction::Mul {
            ty: PtxType::U32,
            mode: MulMode::Wide,
            dst: reg("%rd0", PtxType::U64),
            a: reg_op("%r0", PtxType::U32),
            b: imm_u32(4),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn commutative_power_of_two_detection() {
        // The immediate is in the first operand position
        let instructions = vec![Instruction::Mul {
            ty: PtxType::U32,
            mode: MulMode::Lo,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(16),
            b: reg_op("%r1", PtxType::U32),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 1);
        match &result[0] {
            Instruction::Shl { amount, .. } => {
                assert!(matches!(amount, Operand::Immediate(ImmValue::U32(4))));
            }
            other => panic!("Expected Shl, got {other:?}"),
        }
    }

    #[test]
    fn non_mul_instructions_pass_through() {
        let instructions = vec![
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r0", PtxType::U32),
                a: reg_op("%r1", PtxType::U32),
                b: reg_op("%r2", PtxType::U32),
            },
            Instruction::Return,
        ];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn power_of_two_detection_edge_cases() {
        // 0 is NOT a power of two
        assert!(log2_u32(0).is_none());
        // 1 = 2^0
        assert_eq!(log2_u32(1), Some(0));
        // 3 is not a power of two
        assert!(log2_u32(3).is_none());
        // Large power of two
        assert_eq!(log2_u32(1 << 31), Some(31));
        // u64 large power
        assert_eq!(log2_u64(1u64 << 63), Some(63));
    }

    // -- Div reduction tests ------------------------------------------------

    #[test]
    fn div_by_power_of_2_becomes_shr() {
        let instructions = vec![Instruction::Div {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: reg_op("%r1", PtxType::U32),
            b: imm_u32(16),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 1);
        match &result[0] {
            Instruction::Shr { ty, amount, .. } => {
                assert_eq!(*ty, PtxType::U32);
                assert!(matches!(amount, Operand::Immediate(ImmValue::U32(4))));
            }
            other => panic!("Expected Shr, got {other:?}"),
        }
    }

    #[test]
    fn div_by_non_power_of_2_unchanged() {
        let instructions = vec![Instruction::Div {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: reg_op("%r1", PtxType::U32),
            b: imm_u32(7),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn signed_div_not_reduced() {
        // Signed div by power of 2 is NOT safe to reduce to shift
        // (negative dividend would give wrong result)
        let instructions = vec![Instruction::Div {
            ty: PtxType::S32,
            dst: reg("%r0", PtxType::S32),
            a: reg_op("%r1", PtxType::S32),
            b: Operand::Immediate(ImmValue::S32(8)),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    // -- Rem reduction tests ------------------------------------------------

    #[test]
    fn rem_by_power_of_2_becomes_and() {
        let instructions = vec![Instruction::Rem {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: reg_op("%r1", PtxType::U32),
            b: imm_u32(32),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 1);
        match &result[0] {
            Instruction::And { a, b, .. } => {
                assert!(matches!(a, Operand::Register(_)));
                assert!(matches!(b, Operand::Immediate(ImmValue::U32(31))));
            }
            other => panic!("Expected And, got {other:?}"),
        }
    }

    #[test]
    fn rem_by_non_power_of_2_unchanged() {
        let instructions = vec![Instruction::Rem {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: reg_op("%r1", PtxType::U32),
            b: imm_u32(5),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    // -- Add by 0 reduction tests -------------------------------------------

    #[test]
    fn add_by_zero_becomes_identity() {
        let instructions = vec![Instruction::Add {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: reg_op("%r1", PtxType::U32),
            b: imm_u32(0),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 1);
        match &result[0] {
            Instruction::Add { a, b, .. } => {
                assert!(matches!(a, Operand::Register(_)));
                assert!(matches!(b, Operand::Immediate(ImmValue::U32(0))));
            }
            other => panic!("Expected identity Add, got {other:?}"),
        }
    }

    #[test]
    fn add_zero_on_left_becomes_identity() {
        let instructions = vec![Instruction::Add {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: imm_u32(0),
            b: reg_op("%r1", PtxType::U32),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 1);
        match &result[0] {
            Instruction::Add { a, b, .. } => {
                assert!(matches!(a, Operand::Register(_)));
                assert!(matches!(b, Operand::Immediate(ImmValue::U32(0))));
            }
            other => panic!("Expected identity Add, got {other:?}"),
        }
    }

    #[test]
    fn add_non_zero_not_reduced() {
        let instructions = vec![Instruction::Add {
            ty: PtxType::U32,
            dst: reg("%r0", PtxType::U32),
            a: reg_op("%r1", PtxType::U32),
            b: imm_u32(5),
        }];
        let (result, count) = reduce_strength(&instructions);
        assert_eq!(count, 0);
        assert_eq!(result.len(), 1);
    }

    // -- Report tests -------------------------------------------------------

    #[test]
    fn report_contains_details() {
        let instructions = vec![
            make_mul_lo_u32("%r0", "%r1", 8),
            Instruction::Div {
                ty: PtxType::U32,
                dst: reg("%r2", PtxType::U32),
                a: reg_op("%r3", PtxType::U32),
                b: imm_u32(4),
            },
        ];
        let report = reduce_strength_report(&instructions);
        assert_eq!(report.reductions, 2);
        assert_eq!(report.details.len(), 2);
    }

    #[test]
    fn report_empty_input() {
        let report = reduce_strength_report(&[]);
        assert_eq!(report.reductions, 0);
        assert!(report.details.is_empty());
    }
}
