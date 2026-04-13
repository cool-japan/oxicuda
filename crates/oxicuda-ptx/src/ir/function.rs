//! PTX kernel and device function definitions.
//!
//! A [`PtxFunction`] represents a complete PTX function (`.entry` kernel or
//! `.func` device function) including its parameters, body instructions,
//! shared memory declarations, and optional launch bounds.

use super::instruction::Instruction;
use super::types::PtxType;

/// A PTX kernel or device function definition.
///
/// This structure holds all the information needed to emit a complete PTX
/// function: the function signature (name and typed parameters), the instruction
/// body, any shared memory allocations, and optional performance hints.
///
/// # Examples
///
/// ```
/// use oxicuda_ptx::ir::{PtxFunction, PtxType};
///
/// let func = PtxFunction {
///     name: "vector_add".to_string(),
///     params: vec![
///         ("a_ptr".to_string(), PtxType::U64),
///         ("b_ptr".to_string(), PtxType::U64),
///         ("c_ptr".to_string(), PtxType::U64),
///         ("n".to_string(), PtxType::U32),
///     ],
///     body: Vec::new(),
///     shared_mem: Vec::new(),
///     max_threads: Some(256),
/// };
/// assert_eq!(func.params.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct PtxFunction {
    /// The function name (without leading underscore — emitter adds `$` prefix if needed).
    pub name: String,
    /// Kernel parameters as `(name, type)` pairs.
    pub params: Vec<(String, PtxType)>,
    /// The instruction body of the function.
    pub body: Vec<Instruction>,
    /// Static shared memory declarations as `(name, element_type, num_elements)`.
    pub shared_mem: Vec<(String, PtxType, usize)>,
    /// Optional `.maxnthreads` directive (launch bounds hint to `ptxas`).
    pub max_threads: Option<u32>,
}

impl PtxFunction {
    /// Creates a new empty function with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            body: Vec::new(),
            shared_mem: Vec::new(),
            max_threads: None,
        }
    }

    /// Adds a parameter to the function signature.
    pub fn add_param(&mut self, name: impl Into<String>, ty: PtxType) {
        self.params.push((name.into(), ty));
    }

    /// Adds a static shared memory allocation.
    pub fn add_shared_mem(&mut self, name: impl Into<String>, ty: PtxType, count: usize) {
        self.shared_mem.push((name.into(), ty, count));
    }

    /// Appends an instruction to the function body.
    pub fn push(&mut self, inst: Instruction) {
        self.body.push(inst);
    }
}
