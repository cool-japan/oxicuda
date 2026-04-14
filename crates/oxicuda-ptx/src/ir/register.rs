//! PTX register model and allocation.
//!
//! PTX uses a virtual (infinite) register model. The `ptxas` assembler maps
//! virtual registers to physical registers during compilation. This module
//! provides [`Register`] as a named, typed register and [`RegisterAllocator`]
//! for generating unique register names following PTX naming conventions.

use std::collections::HashMap;
use std::fmt;

use super::types::PtxType;

/// A named PTX register with an associated type.
///
/// Register names follow PTX conventions: `%f0` for floats, `%r0` for 32-bit
/// integers, `%rd0` for 64-bit integers/addresses, `%p0` for predicates.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Register {
    /// The register name (e.g., `"%f0"`, `"%r3"`).
    pub name: String,
    /// The PTX type of values held in this register.
    pub ty: PtxType,
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Register allocator for PTX code generation.
///
/// PTX uses an infinite register model — `ptxas` maps virtual registers to
/// physical registers. This allocator generates sequentially numbered register
/// names grouped by type class, and can emit the `.reg` declarations needed
/// at the top of a PTX function body.
///
/// # Register naming conventions
///
/// | Type class              | Prefix | Example |
/// |-------------------------|--------|---------|
/// | Predicate               | `%p`   | `%p0`   |
/// | F16, BF16, F32, F64     | `%f`   | `%f0`   |
/// | B64, U64, S64           | `%rd`  | `%rd0`  |
/// | Everything else (32-bit)| `%r`   | `%r0`   |
pub struct RegisterAllocator {
    /// Per-prefix counters for generating unique names.
    counters: HashMap<&'static str, u32>,
    /// Track which (prefix, `PtxType`) pairs have been used for declarations.
    used_types: Vec<(&'static str, PtxType)>,
    /// Explicitly named registers (e.g., `%f_x`) declared via [`Self::declare_named`].
    named_registers: Vec<(String, PtxType)>,
}

impl RegisterAllocator {
    /// Creates a new register allocator with all counters at zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            used_types: Vec::new(),
            named_registers: Vec::new(),
        }
    }

    /// Allocates a fresh register of the given type.
    ///
    /// Returns a [`Register`] with a unique name following PTX naming conventions.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxicuda_ptx::ir::{RegisterAllocator, PtxType};
    ///
    /// let mut alloc = RegisterAllocator::new();
    /// let r0 = alloc.alloc(PtxType::F32);
    /// assert_eq!(r0.name, "%f0");
    /// let r1 = alloc.alloc(PtxType::F32);
    /// assert_eq!(r1.name, "%f1");
    /// let ri = alloc.alloc(PtxType::U32);
    /// assert_eq!(ri.name, "%r0");
    /// ```
    pub fn alloc(&mut self, ty: PtxType) -> Register {
        let prefix = Self::prefix_for(ty);
        let counter = self.counters.entry(prefix).or_insert(0);
        let idx = *counter;
        *counter += 1;

        // Track this prefix/type pair for declarations (only once per pair).
        if !self
            .used_types
            .iter()
            .any(|(p, t)| *p == prefix && *t == ty)
        {
            self.used_types.push((prefix, ty));
        }

        Register {
            name: format!("%{prefix}{idx}"),
            ty,
        }
    }

    /// Allocates a group of registers of the same type.
    ///
    /// Returns `count` sequentially numbered registers.
    pub fn alloc_group(&mut self, ty: PtxType, count: u32) -> Vec<Register> {
        (0..count).map(|_| self.alloc(ty)).collect()
    }

    /// Declares a named register for use in raw PTX.
    ///
    /// Unlike [`alloc`](Self::alloc), this accepts an arbitrary register name
    /// (e.g., `%f_x`, `%rd_off`) and ensures it appears in the `.reg`
    /// declarations emitted by [`emit_declarations`](Self::emit_declarations).
    pub fn declare_named(&mut self, name: &str, ty: PtxType) {
        if !self.named_registers.iter().any(|(n, _)| n == name) {
            self.named_registers.push((name.to_string(), ty));
        }
    }

    /// Emits `.reg` declaration lines for all allocated register types.
    ///
    /// Each declaration uses the PTX range syntax (e.g., `.reg .f32 %f<4>;`)
    /// which declares registers `%f0` through `%f3`.
    ///
    /// # Returns
    ///
    /// A vector of declaration strings, one per register prefix used.
    #[must_use]
    pub fn emit_declarations(&self) -> Vec<String> {
        let mut declarations = Vec::new();

        for (prefix, count) in &self.counters {
            if *count == 0 {
                continue;
            }
            // Determine the PTX type string from the first type using this prefix.
            let ptx_type_str = self
                .used_types
                .iter()
                .find(|(p, _)| p == prefix)
                .map_or(".b32", |(_, ty)| ty.reg_type().as_ptx_str());

            declarations.push(format!(".reg {ptx_type_str} %{prefix}<{count}>;"));
        }

        // Emit named register declarations (e.g., `.reg .b64 %rd_off;`).
        for (name, ty) in &self.named_registers {
            declarations.push(format!(".reg {} {name};", ty.reg_type().as_ptx_str()));
        }

        declarations.sort();
        declarations
    }

    /// Returns the register name prefix for a given PTX type.
    const fn prefix_for(ty: PtxType) -> &'static str {
        match ty {
            PtxType::Pred => "p",
            PtxType::F16
            | PtxType::F16x2
            | PtxType::BF16
            | PtxType::BF16x2
            | PtxType::F32
            | PtxType::F64 => "f",
            PtxType::B64 | PtxType::U64 | PtxType::S64 => "rd",
            _ => "r",
        }
    }
}

impl Default for RegisterAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_float_registers() {
        let mut alloc = RegisterAllocator::new();
        let r0 = alloc.alloc(PtxType::F32);
        let r1 = alloc.alloc(PtxType::F32);
        let r2 = alloc.alloc(PtxType::F64);
        assert_eq!(r0.name, "%f0");
        assert_eq!(r1.name, "%f1");
        assert_eq!(r2.name, "%f2");
        assert_eq!(r0.ty, PtxType::F32);
        assert_eq!(r2.ty, PtxType::F64);
    }

    #[test]
    fn alloc_integer_registers() {
        let mut alloc = RegisterAllocator::new();
        let r0 = alloc.alloc(PtxType::U32);
        let r1 = alloc.alloc(PtxType::S32);
        assert_eq!(r0.name, "%r0");
        assert_eq!(r1.name, "%r1");
    }

    #[test]
    fn alloc_64bit_registers() {
        let mut alloc = RegisterAllocator::new();
        let r0 = alloc.alloc(PtxType::U64);
        let r1 = alloc.alloc(PtxType::S64);
        let r2 = alloc.alloc(PtxType::B64);
        assert_eq!(r0.name, "%rd0");
        assert_eq!(r1.name, "%rd1");
        assert_eq!(r2.name, "%rd2");
    }

    #[test]
    fn alloc_predicate_registers() {
        let mut alloc = RegisterAllocator::new();
        let p0 = alloc.alloc(PtxType::Pred);
        let p1 = alloc.alloc(PtxType::Pred);
        assert_eq!(p0.name, "%p0");
        assert_eq!(p1.name, "%p1");
    }

    #[test]
    fn alloc_group() {
        let mut alloc = RegisterAllocator::new();
        let regs = alloc.alloc_group(PtxType::F32, 4);
        assert_eq!(regs.len(), 4);
        assert_eq!(regs[0].name, "%f0");
        assert_eq!(regs[3].name, "%f3");
    }

    #[test]
    fn emit_declarations_sorted() {
        let mut alloc = RegisterAllocator::new();
        alloc.alloc(PtxType::F32);
        alloc.alloc(PtxType::F32);
        alloc.alloc(PtxType::U32);
        alloc.alloc(PtxType::Pred);
        alloc.alloc(PtxType::U64);

        let decls = alloc.emit_declarations();
        assert_eq!(decls.len(), 4);
        // Declarations are sorted alphabetically by the full string.
        // Check that all expected declarations are present.
        let joined = decls.join("\n");
        assert!(joined.contains("%f<2>"), "missing f decl: {joined}");
        assert!(joined.contains("%p<1>"), "missing p decl: {joined}");
        assert!(joined.contains("%r<1>"), "missing r decl: {joined}");
        assert!(joined.contains("%rd<1>"), "missing rd decl: {joined}");
        // Verify sorting: each decl should be <= the next.
        for pair in decls.windows(2) {
            assert!(pair[0] <= pair[1], "declarations not sorted: {decls:?}");
        }
    }

    #[test]
    fn register_display() {
        let r = Register {
            name: "%f0".to_string(),
            ty: PtxType::F32,
        };
        assert_eq!(format!("{r}"), "%f0");
    }
}
