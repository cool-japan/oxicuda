//! PTX module representation.
//!
//! A [`PtxModule`] represents a complete PTX compilation unit, including the
//! PTX version directive, target architecture, address size, and all function
//! definitions. It is the top-level IR node from which PTX text is emitted.

use super::function::PtxFunction;

/// A complete PTX module (`.version`, `.target`, functions).
///
/// This is the top-level container for PTX code generation. A module
/// corresponds to a single `.ptx` file and contains the metadata directives
/// required by `ptxas` as well as one or more kernel/device functions.
///
/// # Examples
///
/// ```
/// use oxicuda_ptx::ir::PtxModule;
///
/// let module = PtxModule {
///     version: "8.5".to_string(),
///     target: "sm_90a".to_string(),
///     address_size: 64,
///     functions: Vec::new(),
/// };
/// assert_eq!(module.target, "sm_90a");
/// ```
#[derive(Debug, Clone)]
pub struct PtxModule {
    /// PTX ISA version (e.g., `"8.5"`).
    pub version: String,
    /// Target architecture (e.g., `"sm_90a"`, `"sm_100a"`).
    pub target: String,
    /// Address size in bits (32 or 64; virtually always 64).
    pub address_size: u32,
    /// The functions defined in this module.
    pub functions: Vec<PtxFunction>,
}

impl PtxModule {
    /// Creates a new module targeting the given architecture with PTX 8.5 and 64-bit addressing.
    #[must_use]
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            version: "8.5".to_string(),
            target: target.into(),
            address_size: 64,
            functions: Vec::new(),
        }
    }

    /// Adds a function to this module.
    pub fn add_function(&mut self, func: PtxFunction) {
        self.functions.push(func);
    }
}
