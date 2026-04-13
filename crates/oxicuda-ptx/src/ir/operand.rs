//! PTX instruction operands.
//!
//! Operands represent the inputs and outputs of PTX instructions beyond
//! destination registers. They can be registers, immediate constants,
//! memory addresses (base + optional offset), or symbolic references.

use std::fmt;

use super::register::Register;

/// An operand in a PTX instruction.
///
/// Operands appear as source arguments in arithmetic, memory, and control-flow
/// instructions. The [`Operand::Address`] variant models the `[base + offset]`
/// addressing syntax used in load/store instructions.
#[derive(Debug, Clone)]
pub enum Operand {
    /// A register operand.
    Register(Register),
    /// An immediate (literal) value.
    Immediate(ImmValue),
    /// A memory address with a base register and optional byte offset.
    Address {
        /// The base address register (typically 64-bit).
        base: Register,
        /// Optional byte offset added to the base address.
        offset: Option<i64>,
    },
    /// A symbolic reference (e.g., a parameter name or label).
    Symbol(String),
}

/// An immediate (literal) value embedded in a PTX instruction.
#[derive(Debug, Clone)]
pub enum ImmValue {
    /// 32-bit unsigned integer literal.
    U32(u32),
    /// 64-bit unsigned integer literal.
    U64(u64),
    /// 32-bit signed integer literal.
    S32(i32),
    /// 64-bit signed integer literal.
    S64(i64),
    /// 32-bit floating-point literal.
    F32(f32),
    /// 64-bit floating-point literal.
    F64(f64),
}

impl From<Register> for Operand {
    fn from(reg: Register) -> Self {
        Self::Register(reg)
    }
}

impl fmt::Display for ImmValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U32(v) => write!(f, "{v}"),
            Self::U64(v) => write!(f, "{v}"),
            Self::S32(v) => write!(f, "{v}"),
            Self::S64(v) => write!(f, "{v}"),
            Self::F32(v) => {
                // PTX uses C-style float literals; ensure a decimal point is present.
                if v.fract() == 0.0 {
                    write!(f, "{v:.1}")
                } else {
                    write!(f, "{v}")
                }
            }
            Self::F64(v) => {
                if v.fract() == 0.0 {
                    write!(f, "{v:.1}")
                } else {
                    write!(f, "{v}")
                }
            }
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Register(reg) => write!(f, "{reg}"),
            Self::Immediate(imm) => write!(f, "{imm}"),
            Self::Address { base, offset } => match offset {
                Some(off) if *off != 0 => write!(f, "[{base}+{off}]"),
                _ => write!(f, "[{base}]"),
            },
            Self::Symbol(sym) => write!(f, "{sym}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::PtxType;

    #[test]
    fn operand_display_register() {
        let reg = Register {
            name: "%f0".into(),
            ty: PtxType::F32,
        };
        let op = Operand::Register(reg);
        assert_eq!(format!("{op}"), "%f0");
    }

    #[test]
    fn operand_display_immediate() {
        assert_eq!(format!("{}", ImmValue::U32(42)), "42");
        assert_eq!(format!("{}", ImmValue::F32(3.0)), "3.0");
        assert_eq!(format!("{}", ImmValue::F32(1.5)), "1.5");
        assert_eq!(format!("{}", ImmValue::S32(-7)), "-7");
    }

    #[test]
    fn operand_display_address() {
        let base = Register {
            name: "%rd0".into(),
            ty: PtxType::U64,
        };
        let op_no_offset = Operand::Address {
            base: base.clone(),
            offset: None,
        };
        assert_eq!(format!("{op_no_offset}"), "[%rd0]");

        let op_with_offset = Operand::Address {
            base,
            offset: Some(16),
        };
        assert_eq!(format!("{op_with_offset}"), "[%rd0+16]");
    }

    #[test]
    fn operand_display_symbol() {
        let op = Operand::Symbol("_param_0".into());
        assert_eq!(format!("{op}"), "_param_0");
    }

    #[test]
    fn operand_from_register() {
        let reg = Register {
            name: "%r0".into(),
            ty: PtxType::U32,
        };
        let op: Operand = reg.into();
        assert!(matches!(op, Operand::Register(_)));
    }
}
