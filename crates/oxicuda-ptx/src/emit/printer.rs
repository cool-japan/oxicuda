//! PTX text printer for IR structures.
//!
//! Converts [`PtxModule`] and [`PtxFunction`] IR nodes into textual PTX assembly.
//! This provides an alternative to [`KernelBuilder`] for emitting PTX from
//! manually constructed IR structures.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::ir::{PtxModule, PtxFunction, PtxType, Instruction};
//! use oxicuda_ptx::emit::printer;
//!
//! let module = PtxModule {
//!     version: "8.5".to_string(),
//!     target: "sm_90a".to_string(),
//!     address_size: 64,
//!     functions: vec![],
//! };
//!
//! let ptx = printer::emit_module(&module);
//! assert!(ptx.contains(".version 8.5"));
//! assert!(ptx.contains(".target sm_90a"));
//! ```
//!
//! [`KernelBuilder`]: crate::builder::KernelBuilder

use std::fmt::Write;

use crate::ir::{Instruction, PtxFunction, PtxModule, RegisterAllocator};

/// Emits a complete PTX module as text.
///
/// The output includes the `.version`, `.target`, and `.address_size` directives
/// followed by all function definitions in the module. Each function is emitted
/// using [`emit_function`].
///
/// # Panics
///
/// This function does not panic; formatting errors are silently handled by
/// returning a partial result. Use [`try_emit_module`] for explicit error handling.
#[must_use]
pub fn emit_module(module: &PtxModule) -> String {
    let mut out = String::with_capacity(4096);

    // Header directives
    let _ = writeln!(out, ".version {}", module.version);
    let _ = writeln!(out, ".target {}", module.target);
    let _ = writeln!(out, ".address_size {}", module.address_size);
    let _ = writeln!(out);

    // Functions
    for func in &module.functions {
        let func_text = emit_function_standalone(func);
        let _ = writeln!(out, "{func_text}");
    }

    out
}

/// Emits a complete PTX module as text, returning an error on formatting failure.
///
/// # Errors
///
/// Returns `std::fmt::Error` if string formatting fails.
pub fn try_emit_module(module: &PtxModule) -> Result<String, std::fmt::Error> {
    let mut out = String::with_capacity(4096);

    writeln!(out, ".version {}", module.version)?;
    writeln!(out, ".target {}", module.target)?;
    writeln!(out, ".address_size {}", module.address_size)?;
    writeln!(out)?;

    for func in &module.functions {
        let func_text = try_emit_function_standalone(func)?;
        writeln!(out, "{func_text}")?;
    }

    Ok(out)
}

/// Emits a PTX function definition with register declarations from an allocator.
///
/// This is useful when the function was built using a [`RegisterAllocator`]
/// and you want to include the `.reg` declarations at the top of the body.
///
/// The function is emitted as a `.visible .entry` kernel.
#[must_use]
pub fn emit_function(func: &PtxFunction, regs: &RegisterAllocator) -> String {
    let mut out = String::with_capacity(2048);

    // Function header
    let _ = write!(out, ".visible .entry {}(", func.name);
    for (i, (pname, pty)) in func.params.iter().enumerate() {
        if i > 0 {
            let _ = write!(out, ",");
        }
        let _ = writeln!(out);
        let _ = write!(out, "    .param {} %param_{pname}", param_type_str(*pty));
    }
    let _ = writeln!(out);
    let _ = writeln!(out, ")");
    let _ = writeln!(out, "{{");

    // Max threads hint
    if let Some(n) = func.max_threads {
        let _ = writeln!(out, "    .maxntid {n}, 1, 1;");
    }

    // Register declarations
    for decl in regs.emit_declarations() {
        let _ = writeln!(out, "    {decl}");
    }

    // Shared memory
    for (sname, sty, count) in &func.shared_mem {
        let align = sty.size_bytes().max(4);
        let total = sty.size_bytes() * count;
        let _ = writeln!(out, "    .shared .align {align} .b8 {sname}[{total}];");
    }

    let _ = writeln!(out);

    // Instructions
    for inst in &func.body {
        emit_instruction(&mut out, inst);
    }

    let _ = writeln!(out, "}}");

    out
}

/// Emits a standalone function (without an external register allocator).
///
/// Uses the function body instructions directly without `.reg` declarations.
/// This is suitable for functions where register declarations are embedded
/// as raw PTX in the instruction body.
#[must_use]
pub fn emit_function_standalone(func: &PtxFunction) -> String {
    let mut out = String::with_capacity(2048);

    // Function header
    let _ = write!(out, ".visible .entry {}(", func.name);
    for (i, (pname, pty)) in func.params.iter().enumerate() {
        if i > 0 {
            let _ = write!(out, ",");
        }
        let _ = writeln!(out);
        let _ = write!(out, "    .param {} %param_{pname}", param_type_str(*pty));
    }
    let _ = writeln!(out);
    let _ = writeln!(out, ")");
    let _ = writeln!(out, "{{");

    // Max threads hint
    if let Some(n) = func.max_threads {
        let _ = writeln!(out, "    .maxntid {n}, 1, 1;");
    }

    // Shared memory
    for (sname, sty, count) in &func.shared_mem {
        let align = sty.size_bytes().max(4);
        let total = sty.size_bytes() * count;
        let _ = writeln!(out, "    .shared .align {align} .b8 {sname}[{total}];");
    }

    let _ = writeln!(out);

    // Instructions
    for inst in &func.body {
        emit_instruction(&mut out, inst);
    }

    let _ = writeln!(out, "}}");

    out
}

/// Emits a standalone function with error handling.
fn try_emit_function_standalone(func: &PtxFunction) -> Result<String, std::fmt::Error> {
    let mut out = String::with_capacity(2048);

    write!(out, ".visible .entry {}(", func.name)?;
    for (i, (pname, pty)) in func.params.iter().enumerate() {
        if i > 0 {
            write!(out, ",")?;
        }
        writeln!(out)?;
        write!(out, "    .param {} %param_{pname}", param_type_str(*pty))?;
    }
    writeln!(out)?;
    writeln!(out, ")")?;
    writeln!(out, "{{")?;

    if let Some(n) = func.max_threads {
        writeln!(out, "    .maxntid {n}, 1, 1;")?;
    }

    for (sname, sty, count) in &func.shared_mem {
        let align = sty.size_bytes().max(4);
        let total = sty.size_bytes() * count;
        writeln!(out, "    .shared .align {align} .b8 {sname}[{total}];")?;
    }

    writeln!(out)?;

    for inst in &func.body {
        let text = inst.emit();
        match inst {
            Instruction::Label(_) => writeln!(out, "{text}")?,
            _ => writeln!(out, "    {text}")?,
        }
    }

    writeln!(out, "}}")?;
    Ok(out)
}

/// Emits a single instruction with appropriate indentation.
fn emit_instruction(out: &mut String, inst: &Instruction) {
    let text = inst.emit();
    match inst {
        Instruction::Label(_) => {
            let _ = writeln!(out, "{text}");
        }
        _ => {
            let _ = writeln!(out, "    {text}");
        }
    }
}

/// Returns the `.param` type annotation string for a PTX type.
const fn param_type_str(ty: crate::ir::PtxType) -> &'static str {
    use crate::ir::PtxType;
    match ty {
        PtxType::U8 => ".u8",
        PtxType::U16 => ".u16",
        PtxType::U32 | PtxType::Pred => ".u32",
        PtxType::U64 => ".u64",
        PtxType::S8 => ".s8",
        PtxType::S16 => ".s16",
        PtxType::S32 => ".s32",
        PtxType::S64 => ".s64",
        PtxType::F16 => ".f16",
        PtxType::BF16 | PtxType::B16 | PtxType::E2M3 | PtxType::E3M2 => ".b16",
        PtxType::F32 => ".f32",
        PtxType::F64 => ".f64",
        PtxType::B8 | PtxType::E4M3 | PtxType::E5M2 | PtxType::E2M1 => ".b8",
        PtxType::B32 | PtxType::F16x2 | PtxType::BF16x2 | PtxType::TF32 => ".b32",
        PtxType::B64 => ".b64",
        PtxType::B128 => ".b128",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Instruction, Operand, PtxFunction, PtxModule, PtxType, Register};

    #[test]
    fn emit_empty_module() {
        let module = PtxModule {
            version: "8.5".to_string(),
            target: "sm_90a".to_string(),
            address_size: 64,
            functions: Vec::new(),
        };
        let ptx = emit_module(&module);
        assert!(ptx.contains(".version 8.5"));
        assert!(ptx.contains(".target sm_90a"));
        assert!(ptx.contains(".address_size 64"));
    }

    #[test]
    fn emit_module_with_function() {
        let mut func = PtxFunction::new("test_kernel");
        func.add_param("n", PtxType::U32);
        func.push(Instruction::Return);

        let module = PtxModule {
            version: "7.1".to_string(),
            target: "sm_80".to_string(),
            address_size: 64,
            functions: vec![func],
        };

        let ptx = emit_module(&module);
        assert!(ptx.contains(".entry test_kernel"));
        assert!(ptx.contains("%param_n"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn emit_function_with_regs() {
        let mut func = PtxFunction::new("add_kernel");
        func.add_param("a", PtxType::U64);
        func.add_param("n", PtxType::U32);
        func.max_threads = Some(256);

        let dst = Register {
            name: "%f0".into(),
            ty: PtxType::F32,
        };
        let a = Operand::Register(Register {
            name: "%f1".into(),
            ty: PtxType::F32,
        });
        let b = Operand::Register(Register {
            name: "%f2".into(),
            ty: PtxType::F32,
        });
        func.push(Instruction::Add {
            ty: PtxType::F32,
            dst,
            a,
            b,
        });
        func.push(Instruction::Return);

        let mut regs = RegisterAllocator::new();
        regs.alloc(PtxType::F32);
        regs.alloc(PtxType::F32);
        regs.alloc(PtxType::F32);

        let ptx = emit_function(&func, &regs);
        assert!(ptx.contains(".entry add_kernel"));
        assert!(ptx.contains(".maxntid 256"));
        assert!(ptx.contains(".reg"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn emit_function_with_shared_mem() {
        let mut func = PtxFunction::new("smem_test");
        func.add_shared_mem("tile", PtxType::F32, 256);
        func.push(Instruction::Return);

        let ptx = emit_function_standalone(&func);
        assert!(ptx.contains(".shared .align 4 .b8 tile[1024];"));
    }

    #[test]
    fn try_emit_module_success() {
        let module = PtxModule::new("sm_80");
        let result = try_emit_module(&module);
        assert!(result.is_ok());
        let ptx = result.expect("should succeed");
        assert!(ptx.contains(".version 8.5"));
    }
}
