//! Kernel-level PTX builder.
//!
//! [`KernelBuilder`] is the top-level entry point for constructing a complete
//! PTX module containing a single kernel entry point. It collects kernel
//! metadata (name, target architecture, parameters, shared memory declarations)
//! and delegates instruction generation to a [`BodyBuilder`] closure.

use std::fmt::Write;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::{Instruction, PtxType, RegisterAllocator};

use super::body_builder::BodyBuilder;

/// Type alias for the body closure to reduce type complexity.
type BodyFn = Box<dyn FnOnce(&mut BodyBuilder<'_>)>;

/// Builder for constructing complete PTX kernel modules.
///
/// `KernelBuilder` follows the fluent builder pattern: chain configuration
/// methods, supply a body closure, and call [`build`] to produce the final
/// PTX text.
///
/// # Example
///
/// ```
/// use oxicuda_ptx::builder::KernelBuilder;
/// use oxicuda_ptx::arch::SmVersion;
/// use oxicuda_ptx::ir::PtxType;
///
/// let ptx = KernelBuilder::new("vector_add")
///     .target(SmVersion::Sm80)
///     .param("a", PtxType::U64)
///     .param("b", PtxType::U64)
///     .param("c", PtxType::U64)
///     .param("n", PtxType::U32)
///     .body(|b| {
///         let tid = b.global_thread_id_x();
///         let n_reg = b.load_param_u32("n");
///         b.if_lt_u32(tid, n_reg, |b| {
///             b.comment("kernel body goes here");
///         });
///         b.ret();
///     })
///     .build()
///     .expect("PTX generation failed");
///
/// assert!(ptx.contains(".entry vector_add"));
/// assert!(ptx.contains(".target sm_80"));
/// ```
///
/// [`build`]: KernelBuilder::build
pub struct KernelBuilder {
    /// Kernel function name.
    name: String,
    /// Target GPU architecture.
    target: SmVersion,
    /// Kernel parameters as (name, type) pairs.
    params: Vec<(String, PtxType)>,
    /// Body closure that populates instructions via `BodyBuilder`.
    body_fn: Option<BodyFn>,
    /// Static shared memory declarations: (name, `element_type`, `element_count`).
    shared_mem_declarations: Vec<(String, PtxType, usize)>,
    /// Optional `.maxntid` directive (maximum threads per block).
    max_threads: Option<u32>,
}

impl KernelBuilder {
    /// Creates a new kernel builder with the given kernel name.
    ///
    /// The default target is [`SmVersion::Sm80`] (Ampere). Call [`target`]
    /// to override.
    ///
    /// [`target`]: KernelBuilder::target
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            target: SmVersion::Sm80,
            params: Vec::new(),
            body_fn: None,
            shared_mem_declarations: Vec::new(),
            max_threads: None,
        }
    }

    /// Sets the target GPU architecture for this kernel.
    ///
    /// This determines the `.target` and `.version` directives in the
    /// generated PTX, and also controls which instructions the
    /// [`BodyBuilder`] may emit.
    #[must_use]
    pub const fn target(mut self, sm: SmVersion) -> Self {
        self.target = sm;
        self
    }

    /// Adds a kernel parameter with the given name and type.
    ///
    /// Parameters are emitted in declaration order in the `.entry` signature.
    /// Common types: `PtxType::U64` for pointers, `PtxType::U32` / `PtxType::F32`
    /// for scalar arguments.
    #[must_use]
    pub fn param(mut self, name: &str, ty: PtxType) -> Self {
        self.params.push((name.to_string(), ty));
        self
    }

    /// Declares a static shared memory allocation.
    ///
    /// This generates a `.shared .align` declaration at the top of the
    /// kernel body. The total size is `count * ty.size_bytes()` bytes.
    #[must_use]
    pub fn shared_mem(mut self, name: &str, ty: PtxType, count: usize) -> Self {
        self.shared_mem_declarations
            .push((name.to_string(), ty, count));
        self
    }

    /// Sets the `.maxntid` directive, hinting to `ptxas` the maximum
    /// number of threads per block this kernel will be launched with.
    ///
    /// This can improve register allocation and occupancy planning.
    #[must_use]
    pub const fn max_threads_per_block(mut self, n: u32) -> Self {
        self.max_threads = Some(n);
        self
    }

    /// Supplies the body closure that generates the kernel's instructions.
    ///
    /// The closure receives a mutable reference to a [`BodyBuilder`] which
    /// provides the instruction emission API (loads, stores, arithmetic,
    /// control flow, tensor core ops, etc.).
    #[must_use]
    pub fn body<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut BodyBuilder<'_>) + 'static,
    {
        self.body_fn = Some(Box::new(f));
        self
    }

    /// Consumes the builder and generates the complete PTX module text.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::MissingBody`] if no body closure was provided.
    /// Returns [`PtxGenError::FormatError`] if string formatting fails.
    pub fn build(self) -> Result<String, PtxGenError> {
        let body_fn = self.body_fn.ok_or(PtxGenError::MissingBody)?;

        // Phase 1: Execute the body closure to collect instructions.
        let mut regs = RegisterAllocator::new();
        let mut instructions: Vec<Instruction> = Vec::new();
        {
            let param_names: Vec<String> = self.params.iter().map(|(n, _)| n.clone()).collect();
            let mut bb = BodyBuilder::new(&mut regs, &mut instructions, &param_names, self.target);
            body_fn(&mut bb);
        }

        // Phase 2: Generate PTX text.
        let mut ptx = String::with_capacity(4096);

        // Header directives.
        writeln!(ptx, ".version {}", self.target.ptx_version())?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str())?;
        writeln!(ptx, ".address_size 64")?;
        writeln!(ptx)?;

        // Kernel entry point.
        write!(ptx, ".visible .entry {}(", self.name)?;
        for (i, (pname, pty)) in self.params.iter().enumerate() {
            if i > 0 {
                write!(ptx, ",")?;
            }
            writeln!(ptx)?;
            write!(ptx, "    {} {}", param_type_str(*pty), param_ident(pname))?;
        }
        writeln!(ptx)?;
        writeln!(ptx, ")")?;
        writeln!(ptx, "{{")?;

        // .maxntid directive.
        if let Some(n) = self.max_threads {
            writeln!(ptx, "    .maxntid {n}, 1, 1;")?;
        }

        // Register declarations.
        let reg_decls = regs.emit_declarations();
        for decl in &reg_decls {
            writeln!(ptx, "    {decl}")?;
        }

        // Shared memory declarations.
        for (sname, sty, count) in &self.shared_mem_declarations {
            let align = sty.size_bytes().max(4);
            let total_bytes = sty.size_bytes() * count;
            writeln!(
                ptx,
                "    .shared .align {align} .b8 {sname}[{total_bytes}];"
            )?;
        }

        if !reg_decls.is_empty() || !self.shared_mem_declarations.is_empty() {
            writeln!(ptx)?;
        }

        // Instructions.
        for inst in &instructions {
            emit_instruction(&mut ptx, inst)?;
        }

        writeln!(ptx, "}}")?;

        Ok(ptx)
    }
}

/// Returns the PTX `.param` type annotation for a kernel parameter type.
const fn param_type_str(ty: PtxType) -> &'static str {
    match ty {
        PtxType::U8 => ".param .u8",
        PtxType::U16 => ".param .u16",
        PtxType::U32 | PtxType::Pred => ".param .u32",
        PtxType::U64 => ".param .u64",
        PtxType::S8 => ".param .s8",
        PtxType::S16 => ".param .s16",
        PtxType::S32 => ".param .s32",
        PtxType::S64 => ".param .s64",
        PtxType::F16 => ".param .f16",
        PtxType::BF16 | PtxType::B16 | PtxType::E2M3 | PtxType::E3M2 => ".param .b16",
        PtxType::F32 => ".param .f32",
        PtxType::F64 => ".param .f64",
        PtxType::B8 | PtxType::E4M3 | PtxType::E5M2 | PtxType::E2M1 => ".param .b8",
        PtxType::B32 | PtxType::F16x2 | PtxType::BF16x2 | PtxType::TF32 => ".param .b32",
        PtxType::B64 => ".param .b64",
        PtxType::B128 => ".param .b128",
    }
}

/// Returns the PTX-safe parameter identifier (prefixed with `%param_`).
fn param_ident(name: &str) -> String {
    format!("%param_{name}")
}

/// Emits a single PTX instruction as text, appending to `out`.
///
/// Each instruction is indented by 4 spaces (labels have no indentation).
fn emit_instruction(out: &mut String, inst: &Instruction) -> Result<(), std::fmt::Error> {
    let text = inst.emit();
    match inst {
        Instruction::Label(_) => writeln!(out, "{text}"),
        _ => writeln!(out, "    {text}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_minimal_kernel() {
        let ptx = KernelBuilder::new("test_kernel")
            .target(SmVersion::Sm80)
            .param("n", PtxType::U32)
            .body(|b| {
                b.ret();
            })
            .build();

        let ptx = ptx.expect("build should succeed");
        assert!(ptx.contains(".version 7.0"));
        assert!(ptx.contains(".target sm_80"));
        assert!(ptx.contains(".address_size 64"));
        assert!(ptx.contains(".entry test_kernel"));
        assert!(ptx.contains(".param .u32 %param_n"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn build_missing_body() {
        let result = KernelBuilder::new("no_body")
            .target(SmVersion::Sm75)
            .build();

        assert!(result.is_err());
        let err = result.expect_err("should be MissingBody");
        assert!(matches!(err, PtxGenError::MissingBody));
    }

    #[test]
    fn build_with_shared_mem() {
        let ptx = KernelBuilder::new("smem_kernel")
            .target(SmVersion::Sm80)
            .shared_mem("tile_a", PtxType::F32, 1024)
            .body(|b| {
                b.ret();
            })
            .build()
            .expect("build should succeed");

        assert!(ptx.contains(".shared .align 4 .b8 tile_a[4096];"));
    }

    #[test]
    fn build_with_max_threads() {
        let ptx = KernelBuilder::new("bounded_kernel")
            .target(SmVersion::Sm80)
            .max_threads_per_block(256)
            .body(|b| {
                b.ret();
            })
            .build()
            .expect("build should succeed");

        assert!(ptx.contains(".maxntid 256, 1, 1;"));
    }

    #[test]
    fn param_type_str_coverage() {
        assert_eq!(param_type_str(PtxType::U32), ".param .u32");
        assert_eq!(param_type_str(PtxType::U64), ".param .u64");
        assert_eq!(param_type_str(PtxType::F32), ".param .f32");
        assert_eq!(param_type_str(PtxType::F64), ".param .f64");
        assert_eq!(param_type_str(PtxType::S32), ".param .s32");
        assert_eq!(param_type_str(PtxType::B32), ".param .b32");
        assert_eq!(param_type_str(PtxType::B128), ".param .b128");
    }
}
