//! Basic block representation for PTX control flow.
//!
//! A [`BasicBlock`] groups a sequence of PTX instructions under an optional
//! label. Basic blocks are the fundamental unit of control flow in the IR
//! and map directly to labeled sections in PTX assembly.

use super::instruction::Instruction;

/// A basic block of PTX instructions.
///
/// Each block optionally begins with a label and contains a linear sequence
/// of instructions. Control flow between blocks is expressed through
/// [`Instruction::Branch`] and [`Instruction::Label`] instructions.
///
/// # Examples
///
/// ```
/// use oxicuda_ptx::ir::{BasicBlock, Instruction};
///
/// let block = BasicBlock {
///     label: Some("loop_body".to_string()),
///     instructions: vec![
///         Instruction::Comment("loop iteration".to_string()),
///     ],
/// };
/// assert_eq!(block.label.as_deref(), Some("loop_body"));
/// ```
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Optional label for this block (used as a branch target).
    pub label: Option<String>,
    /// The sequence of instructions in this block.
    pub instructions: Vec<Instruction>,
}

impl BasicBlock {
    /// Creates a new empty basic block with no label.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            label: None,
            instructions: Vec::new(),
        }
    }

    /// Creates a new empty basic block with the given label.
    #[must_use]
    pub fn with_label(label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            instructions: Vec::new(),
        }
    }

    /// Appends an instruction to this block.
    pub fn push(&mut self, inst: Instruction) {
        self.instructions.push(inst);
    }

    /// Returns the number of instructions in this block.
    #[must_use]
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Returns `true` if this block contains no instructions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

impl Default for BasicBlock {
    fn default() -> Self {
        Self::new()
    }
}
