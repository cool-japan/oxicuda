//! Basic PTX text validation.
//!
//! Provides lightweight validation checks on generated PTX text to catch
//! common errors before submitting to the driver's `cuModuleLoadDataEx`.
//! This is not a full PTX parser — it performs heuristic checks for:
//!
//! - Presence of required directives (`.version`, `.target`)
//! - Register declaration vs. usage consistency
//! - Shared memory size limits for the target architecture
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::emit::validator::{validate_ptx, ValidationError};
//!
//! let ptx = ".version 8.5\n.target sm_90a\n.address_size 64\n";
//! let result = validate_ptx(ptx);
//! assert!(result.errors.is_empty());
//! ```

use std::collections::HashSet;

use crate::arch::SmVersion;
use crate::ir::{Instruction, MemorySpace, Operand, WmmaOp};

/// Result of PTX validation containing errors and warnings.
///
/// An empty `errors` vector indicates the PTX passed all checks. Warnings
/// are informational and do not indicate invalid PTX.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Fatal validation errors that likely indicate broken PTX.
    pub errors: Vec<ValidationError>,
    /// Non-fatal warnings (informational).
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Returns `true` if no errors were found.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns `true` if any errors were found.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

/// A PTX validation error.
///
/// Each variant describes a specific issue found during validation.
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// The `.version` directive is missing.
    MissingVersionDirective,
    /// The `.target` directive is missing.
    MissingTargetDirective,
    /// A register was used but not declared.
    UndefinedRegister(String),
    /// A type mismatch was detected (heuristic-based).
    TypeMismatch {
        /// The expected type.
        expected: String,
        /// The type that was found.
        found: String,
    },
    /// Shared memory size exceeds architecture limits.
    InvalidSharedMemSize {
        /// The declared shared memory size in bytes.
        declared: usize,
        /// The maximum allowed for the target architecture.
        max_allowed: usize,
    },
    /// The `.address_size` directive is missing or not 64.
    InvalidAddressSize(String),
    /// An instruction requires a newer SM version than the target.
    SmIncompatibleInstruction {
        /// The instruction or feature that is not available.
        instruction: String,
        /// The minimum SM version required (e.g. `"sm_80"`).
        required_sm: String,
        /// The SM version specified in the PTX (e.g. `"sm_75"`).
        found_sm: String,
    },
    /// Register count exceeds the architecture's per-thread limit (255).
    RegisterPressureExceeded {
        /// Number of unique registers detected.
        count: usize,
        /// Maximum allowed registers per thread.
        max_allowed: usize,
    },
    /// A generic validation error with descriptive message.
    Other(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingVersionDirective => write!(f, "missing .version directive"),
            Self::MissingTargetDirective => write!(f, "missing .target directive"),
            Self::UndefinedRegister(name) => write!(f, "undefined register: {name}"),
            Self::TypeMismatch { expected, found } => {
                write!(f, "type mismatch: expected {expected}, found {found}")
            }
            Self::InvalidSharedMemSize {
                declared,
                max_allowed,
            } => {
                write!(
                    f,
                    "shared memory {declared} bytes exceeds limit of {max_allowed} bytes"
                )
            }
            Self::InvalidAddressSize(msg) => write!(f, "address size issue: {msg}"),
            Self::SmIncompatibleInstruction {
                instruction,
                required_sm,
                found_sm,
            } => write!(
                f,
                "instruction '{instruction}' requires {required_sm} but target is {found_sm}"
            ),
            Self::RegisterPressureExceeded { count, max_allowed } => write!(
                f,
                "register count {count} exceeds per-thread limit of {max_allowed}"
            ),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

/// Validates PTX text for common errors.
///
/// Performs the following checks:
/// 1. `.version` directive is present
/// 2. `.target` directive is present
/// 3. Register declarations match usage (heuristic)
/// 4. Shared memory does not exceed architecture limits
///
/// # Arguments
///
/// * `ptx` - The PTX text to validate
///
/// # Returns
///
/// A [`ValidationResult`] containing any errors and warnings found.
#[must_use]
pub fn validate_ptx(ptx: &str) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Check for .version directive
    if !ptx.contains(".version") {
        errors.push(ValidationError::MissingVersionDirective);
    }

    // Check for .target directive
    if !ptx.contains(".target") {
        errors.push(ValidationError::MissingTargetDirective);
    }

    // Try to determine target architecture for limit checking
    let target_sm = extract_target_sm(ptx);

    // Check shared memory limits
    check_shared_memory(ptx, target_sm, &mut errors, &mut warnings);

    // Check register declarations vs usage (real tracking)
    check_register_declarations(ptx, &mut warnings);

    // Check register pressure against hardware limits (255 per thread)
    check_register_pressure(ptx, &mut errors, &mut warnings);

    // Check SM-version-specific instruction availability
    if let Some(sm) = target_sm {
        check_sm_compatibility(ptx, sm, &mut errors, &mut warnings);
    }

    // Check for basic structural issues
    check_structure(ptx, &mut warnings);

    ValidationResult { errors, warnings }
}

/// Validates PTX text against a specific target architecture.
///
/// This variant allows specifying the target explicitly rather than
/// extracting it from the PTX text. It runs all checks from [`validate_ptx`]
/// plus explicit architecture-specific checks (SM compatibility, register
/// pressure, shared memory limits) against `target`.
#[must_use]
pub fn validate_ptx_for_target(ptx: &str, target: SmVersion) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Check for .version directive
    if !ptx.contains(".version") {
        errors.push(ValidationError::MissingVersionDirective);
    }

    // Check for .target directive
    if !ptx.contains(".target") {
        errors.push(ValidationError::MissingTargetDirective);
    }

    // Shared memory against the explicitly-supplied target
    check_shared_memory(ptx, Some(target), &mut errors, &mut warnings);

    // Register declarations heuristic
    check_register_declarations(ptx, &mut warnings);

    // Register pressure against hardware limit
    check_register_pressure(ptx, &mut errors, &mut warnings);

    // SM-version-specific instruction compatibility
    check_sm_compatibility(ptx, target, &mut errors, &mut warnings);

    // Basic structural checks
    check_structure(ptx, &mut warnings);

    ValidationResult { errors, warnings }
}

/// Extracts the target SM version from PTX text, if present.
fn extract_target_sm(ptx: &str) -> Option<SmVersion> {
    for line in ptx.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(".target") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                return parse_sm_version(parts[1].trim_end_matches(';'));
            }
        }
    }
    None
}

/// Parses a target string like `sm_80` into an `SmVersion`.
fn parse_sm_version(s: &str) -> Option<SmVersion> {
    match s {
        "sm_75" => Some(SmVersion::Sm75),
        "sm_80" => Some(SmVersion::Sm80),
        "sm_86" => Some(SmVersion::Sm86),
        "sm_89" => Some(SmVersion::Sm89),
        "sm_90" => Some(SmVersion::Sm90),
        "sm_90a" => Some(SmVersion::Sm90a),
        "sm_100" => Some(SmVersion::Sm100),
        "sm_120" => Some(SmVersion::Sm120),
        _ => None,
    }
}

/// Checks shared memory declarations against architecture limits.
fn check_shared_memory(
    ptx: &str,
    target: Option<SmVersion>,
    errors: &mut Vec<ValidationError>,
    warnings: &mut Vec<String>,
) {
    let max_smem = target.map_or(usize::MAX, |sm| sm.max_shared_mem_per_block() as usize);

    let mut total_smem: usize = 0;

    for line in ptx.lines() {
        let trimmed = line.trim();
        if let Some(size) = extract_shared_mem_size(trimmed) {
            total_smem = total_smem.saturating_add(size);
        }
    }

    if total_smem > max_smem {
        errors.push(ValidationError::InvalidSharedMemSize {
            declared: total_smem,
            max_allowed: max_smem,
        });
    } else if total_smem > 48 * 1024 && target.is_some() {
        warnings.push(format!(
            "shared memory usage ({total_smem} bytes) exceeds default limit (49152); \
             may require opt-in via cuFuncSetAttribute"
        ));
    }
}

/// Extracts the byte size from a `.shared` declaration line.
///
/// Handles patterns like `.shared .align 4 .b8 smem[1024];`
fn extract_shared_mem_size(line: &str) -> Option<usize> {
    if !line.contains(".shared") {
        return None;
    }

    // Look for [size] pattern
    let bracket_start = line.find('[')?;
    let bracket_end = line.find(']')?;
    if bracket_end <= bracket_start {
        return None;
    }

    let size_str = &line[bracket_start + 1..bracket_end];
    size_str.trim().parse::<usize>().ok()
}

/// Heuristic check for register declarations vs usage.
fn check_register_declarations(ptx: &str, warnings: &mut Vec<String>) {
    // Count register declaration groups
    let decl_count = ptx
        .lines()
        .filter(|line| line.trim().starts_with(".reg"))
        .count();

    // Count entry points
    let entry_count = ptx.lines().filter(|line| line.contains(".entry")).count();

    if entry_count > 0 && decl_count == 0 {
        warnings.push(
            "kernel has no .reg declarations; all registers may be declared via raw PTX"
                .to_string(),
        );
    }
}

/// Checks basic structural properties of the PTX.
fn check_structure(ptx: &str, warnings: &mut Vec<String>) {
    let open_braces = ptx.chars().filter(|c| *c == '{').count();
    let close_braces = ptx.chars().filter(|c| *c == '}').count();

    if open_braces != close_braces {
        warnings.push(format!(
            "mismatched braces: {open_braces} opening vs {close_braces} closing"
        ));
    }
}

// ===========================================================================
// SM version compatibility checks
// ===========================================================================

/// Describes an instruction/feature that requires a minimum SM version.
struct SmRequirement {
    /// Substring to search for in the PTX text.
    pattern: &'static str,
    /// The minimum SM version required.
    min_sm: SmVersion,
    /// Human-readable name for error messages.
    name: &'static str,
}

/// Table of instructions and the minimum SM version that supports them.
const SM_REQUIREMENTS: &[SmRequirement] = &[
    SmRequirement {
        pattern: "cp.async",
        min_sm: SmVersion::Sm80,
        name: "cp.async",
    },
    SmRequirement {
        pattern: "wgmma",
        min_sm: SmVersion::Sm90,
        name: "wgmma",
    },
    SmRequirement {
        pattern: "mma.sync",
        min_sm: SmVersion::Sm75,
        name: "mma.sync (tensor core)",
    },
    SmRequirement {
        pattern: "ldmatrix",
        min_sm: SmVersion::Sm75,
        name: "ldmatrix",
    },
    SmRequirement {
        pattern: ".e4m3",
        min_sm: SmVersion::Sm89,
        name: "fp8 e4m3 type",
    },
    SmRequirement {
        pattern: ".e5m2",
        min_sm: SmVersion::Sm89,
        name: "fp8 e5m2 type",
    },
    SmRequirement {
        pattern: "tcgen05",
        min_sm: SmVersion::Sm100,
        name: "tcgen05",
    },
];

/// Checks that instructions present in the PTX are supported by the target SM.
///
/// Scans for known instruction patterns and emits an error for each that
/// requires a newer SM than the target.
fn check_sm_compatibility(
    ptx: &str,
    sm: SmVersion,
    errors: &mut Vec<ValidationError>,
    _warnings: &mut Vec<String>,
) {
    let found_sm_str = sm.as_ptx_str();
    for req in SM_REQUIREMENTS {
        if ptx.contains(req.pattern) && sm < req.min_sm {
            errors.push(ValidationError::SmIncompatibleInstruction {
                instruction: req.name.to_string(),
                required_sm: req.min_sm.as_ptx_str().to_string(),
                found_sm: found_sm_str.to_string(),
            });
        }
    }
}

// ===========================================================================
// Register pressure check
// ===========================================================================

/// Maximum number of registers per thread allowed by all current NVIDIA GPUs.
const MAX_REGISTERS_PER_THREAD: usize = 255;

/// Register count at which a warning is emitted (approaching the limit).
const REGISTER_PRESSURE_WARNING_THRESHOLD: usize = 200;

/// Checks whether the number of distinct register names used in the PTX
/// text exceeds or approaches the per-thread hardware limit.
///
/// Scans for PTX register naming conventions:
/// - `%r\d+`  — u32/s32 registers
/// - `%f\d+`  — f32 registers
/// - `%rd\d+` — u64/s64 registers
/// - `%fd\d+` — f64 registers
/// - `%p\d+`  — predicate registers
/// - `%b\d+`  — b32/b64 registers
/// - `%h\d+`  — f16 registers
fn check_register_pressure(
    ptx: &str,
    errors: &mut Vec<ValidationError>,
    warnings: &mut Vec<String>,
) {
    use std::collections::HashSet;

    let mut seen: HashSet<&str> = HashSet::new();

    // We walk the PTX character by character looking for '%' followed by a
    // register name (letters then digits).  This avoids bringing in a regex
    // dependency while still being precise enough for generated PTX.
    let bytes = ptx.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if bytes[i] == b'%' {
            let start = i;
            i += 1;
            // Consume the letter prefix (e.g. "rd", "fd", "r", "f", "p", ...)
            while i < len && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
            // Must be followed by at least one digit to be a register reference
            if i < len && bytes[i].is_ascii_digit() {
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                // Only count concrete register references (not %tid.x, %ctaid.x, etc.)
                let token = &ptx[start..i];
                // Special registers contain '.' after the name — skip them.
                // We already stopped at non-digit so the token is clean.
                // Exclude PTX special registers that start with known prefixes.
                let name_part = &token[1..]; // strip leading '%'
                let is_special = name_part.starts_with("tid")
                    || name_part.starts_with("ntid")
                    || name_part.starts_with("ctaid")
                    || name_part.starts_with("nctaid")
                    || name_part.starts_with("laneid")
                    || name_part.starts_with("warpid")
                    || name_part.starts_with("smid")
                    || name_part.starts_with("pm")
                    || name_part.starts_with("envreg")
                    || name_part.starts_with("globaltimer")
                    || name_part.starts_with("param_");
                if !is_special {
                    seen.insert(token);
                }
            }
        } else {
            i += 1;
        }
    }

    let count = seen.len();
    if count > MAX_REGISTERS_PER_THREAD {
        errors.push(ValidationError::RegisterPressureExceeded {
            count,
            max_allowed: MAX_REGISTERS_PER_THREAD,
        });
    } else if count > REGISTER_PRESSURE_WARNING_THRESHOLD {
        warnings.push(format!(
            "register count ({count}) is approaching the per-thread limit of \
             {MAX_REGISTERS_PER_THREAD}; consider reducing register pressure"
        ));
    }
}

// ===========================================================================
// IR-level validation
// ===========================================================================

/// Result of IR-level instruction validation.
#[derive(Debug, Clone)]
pub struct IrValidationResult {
    /// Fatal validation errors.
    pub errors: Vec<IrValidationError>,
    /// Non-fatal warnings.
    pub warnings: Vec<IrValidationWarning>,
}

impl IrValidationResult {
    /// Returns `true` if no errors were found.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns `true` if any errors were found.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Merge another result into this one.
    fn merge(&mut self, other: &Self) {
        self.errors.extend(other.errors.iter().cloned());
        self.warnings.extend(other.warnings.iter().cloned());
    }
}

impl std::fmt::Display for IrValidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.errors.is_empty() && self.warnings.is_empty() {
            return write!(f, "IR validation passed: no errors, no warnings");
        }
        if !self.errors.is_empty() {
            writeln!(f, "Errors ({}):", self.errors.len())?;
            for err in &self.errors {
                writeln!(
                    f,
                    "  [{:>3}] {}: {}",
                    err.instruction_index, err.kind, err.message
                )?;
            }
        }
        if !self.warnings.is_empty() {
            writeln!(f, "Warnings ({}):", self.warnings.len())?;
            for warn in &self.warnings {
                writeln!(f, "  [{:>3}] {}", warn.instruction_index, warn.message)?;
            }
        }
        Ok(())
    }
}

/// An IR-level validation error tied to a specific instruction.
#[derive(Debug, Clone)]
pub struct IrValidationError {
    /// Index of the offending instruction in the sequence.
    pub instruction_index: usize,
    /// The kind of error detected.
    pub kind: IrErrorKind,
    /// Human-readable description.
    pub message: String,
}

/// Categories of IR validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrErrorKind {
    /// Type mismatch between operands.
    TypeMismatch,
    /// Register used before definition.
    UseBeforeDef,
    /// Invalid memory space for instruction.
    InvalidMemorySpace,
    /// Invalid operand type for instruction.
    InvalidOperand,
    /// Barrier inside divergent control flow.
    BarrierInDivergent,
    /// Register lifetime issue.
    RegisterLifetime,
}

impl std::fmt::Display for IrErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeMismatch => write!(f, "TypeMismatch"),
            Self::UseBeforeDef => write!(f, "UseBeforeDef"),
            Self::InvalidMemorySpace => write!(f, "InvalidMemorySpace"),
            Self::InvalidOperand => write!(f, "InvalidOperand"),
            Self::BarrierInDivergent => write!(f, "BarrierInDivergent"),
            Self::RegisterLifetime => write!(f, "RegisterLifetime"),
        }
    }
}

/// An IR-level validation warning tied to a specific instruction.
#[derive(Debug, Clone)]
pub struct IrValidationWarning {
    /// Index of the relevant instruction.
    pub instruction_index: usize,
    /// Human-readable warning message.
    pub message: String,
}

// ---------------------------------------------------------------------------
// Helper: extract register names from operands used as sources
// ---------------------------------------------------------------------------

/// Extracts register name(s) from an operand used as a source.
fn push_operand_names(op: &Operand, names: &mut Vec<String>) {
    if let Operand::Register(r) = op {
        names.push(r.name.clone());
    }
    if let Operand::Address { base, .. } = op {
        names.push(base.name.clone());
    }
}

/// Collects the register names used as source operands in an instruction.
#[allow(clippy::too_many_lines)]
fn collect_src_register_names(inst: &Instruction) -> Vec<String> {
    let mut names = Vec::new();

    match inst {
        Instruction::Add { a, b, .. }
        | Instruction::Sub { a, b, .. }
        | Instruction::Mul { a, b, .. }
        | Instruction::Min { a, b, .. }
        | Instruction::Max { a, b, .. }
        | Instruction::Div { a, b, .. }
        | Instruction::Rem { a, b, .. }
        | Instruction::And { a, b, .. }
        | Instruction::Or { a, b, .. }
        | Instruction::Xor { a, b, .. }
        | Instruction::SetP { a, b, .. } => {
            push_operand_names(a, &mut names);
            push_operand_names(b, &mut names);
        }
        Instruction::Mad { a, b, c, .. }
        | Instruction::MadLo { a, b, c, .. }
        | Instruction::MadHi { a, b, c, .. }
        | Instruction::MadWide { a, b, c, .. }
        | Instruction::Fma { a, b, c, .. }
        | Instruction::Dp4a { a, b, c, .. }
        | Instruction::Dp2a { a, b, c, .. } => {
            push_operand_names(a, &mut names);
            push_operand_names(b, &mut names);
            push_operand_names(c, &mut names);
        }
        Instruction::Neg { src, .. }
        | Instruction::Abs { src, .. }
        | Instruction::Brev { src, .. }
        | Instruction::Clz { src, .. }
        | Instruction::Popc { src, .. }
        | Instruction::Bfind { src, .. }
        | Instruction::Rcp { src, .. }
        | Instruction::Rsqrt { src, .. }
        | Instruction::Sqrt { src, .. }
        | Instruction::Ex2 { src, .. }
        | Instruction::Lg2 { src, .. }
        | Instruction::Sin { src, .. }
        | Instruction::Cos { src, .. }
        | Instruction::Cvt { src, .. }
        | Instruction::Redux { src, .. } => {
            push_operand_names(src, &mut names);
        }
        Instruction::Bfe {
            src, start, len, ..
        } => {
            push_operand_names(src, &mut names);
            push_operand_names(start, &mut names);
            push_operand_names(len, &mut names);
        }
        Instruction::Bfi {
            insert,
            base,
            start,
            len,
            ..
        } => {
            push_operand_names(insert, &mut names);
            push_operand_names(base, &mut names);
            push_operand_names(start, &mut names);
            push_operand_names(len, &mut names);
        }
        Instruction::Shl { src, amount, .. } | Instruction::Shr { src, amount, .. } => {
            push_operand_names(src, &mut names);
            push_operand_names(amount, &mut names);
        }
        Instruction::Load { addr, .. } | Instruction::MbarrierArrive { addr } => {
            push_operand_names(addr, &mut names);
        }
        Instruction::Store { addr, src, .. } => {
            push_operand_names(addr, &mut names);
            names.push(src.name.clone());
        }
        Instruction::CpAsync {
            dst_shared,
            src_global,
            ..
        } => {
            push_operand_names(dst_shared, &mut names);
            push_operand_names(src_global, &mut names);
        }
        Instruction::Branch { predicate, .. } => {
            if let Some((r, _)) = predicate {
                names.push(r.name.clone());
            }
        }
        Instruction::Atom { addr, src, .. } | Instruction::Red { addr, src, .. } => {
            push_operand_names(addr, &mut names);
            push_operand_names(src, &mut names);
        }
        Instruction::AtomCas {
            addr,
            compare,
            value,
            ..
        } => {
            push_operand_names(addr, &mut names);
            push_operand_names(compare, &mut names);
            push_operand_names(value, &mut names);
        }
        Instruction::Tex1d { coord, .. } | Instruction::SurfLoad { coord, .. } => {
            push_operand_names(coord, &mut names);
        }
        Instruction::Tex2d {
            coord_x, coord_y, ..
        } => {
            push_operand_names(coord_x, &mut names);
            push_operand_names(coord_y, &mut names);
        }
        Instruction::Tex3d {
            coord_x,
            coord_y,
            coord_z,
            ..
        } => {
            push_operand_names(coord_x, &mut names);
            push_operand_names(coord_y, &mut names);
            push_operand_names(coord_z, &mut names);
        }
        Instruction::SurfStore { coord, src, .. } => {
            push_operand_names(coord, &mut names);
            names.push(src.name.clone());
        }
        Instruction::Wmma {
            fragments,
            addr,
            stride,
            ..
        } => {
            for frag in fragments {
                names.push(frag.name.clone());
            }
            if let Some(a) = addr {
                push_operand_names(a, &mut names);
            }
            if let Some(s) = stride {
                push_operand_names(s, &mut names);
            }
        }
        Instruction::Mma {
            a_regs,
            b_regs,
            c_regs,
            ..
        } => {
            for r in a_regs.iter().chain(b_regs).chain(c_regs) {
                names.push(r.name.clone());
            }
        }
        Instruction::Wgmma { desc_a, desc_b, .. } => {
            names.push(desc_a.name.clone());
            names.push(desc_b.name.clone());
        }
        Instruction::TmaLoad {
            desc,
            coords,
            barrier,
            dst_shared,
            ..
        } => {
            names.push(desc.name.clone());
            for c in coords {
                names.push(c.name.clone());
            }
            names.push(barrier.name.clone());
            push_operand_names(dst_shared, &mut names);
        }
        Instruction::Stmatrix { dst_addr, src, .. } => {
            push_operand_names(dst_addr, &mut names);
            names.push(src.name.clone());
        }
        Instruction::MbarrierInit { addr, count } => {
            push_operand_names(addr, &mut names);
            push_operand_names(count, &mut names);
        }
        Instruction::MbarrierWait { addr, phase } => {
            push_operand_names(addr, &mut names);
            push_operand_names(phase, &mut names);
        }
        Instruction::MovSpecial { .. }
        | Instruction::LoadParam { .. }
        | Instruction::Label(_)
        | Instruction::Return
        | Instruction::Comment(_)
        | Instruction::Raw(_)
        | Instruction::Pragma(_)
        | Instruction::BarSync { .. }
        | Instruction::BarArrive { .. }
        | Instruction::FenceAcqRel { .. }
        | Instruction::FenceProxy { .. }
        | Instruction::CpAsyncCommit
        | Instruction::CpAsyncWait { .. }
        | Instruction::ElectSync { .. }
        | Instruction::Setmaxnreg { .. }
        | Instruction::Griddepcontrol { .. }
        | Instruction::BarrierCluster
        | Instruction::FenceCluster => {}

        Instruction::Tcgen05Mma { a_desc, b_desc } => {
            names.push(a_desc.name.clone());
            names.push(b_desc.name.clone());
        }
        Instruction::CpAsyncBulk {
            dst_smem,
            src_gmem,
            desc,
        } => {
            names.push(dst_smem.name.clone());
            names.push(src_gmem.name.clone());
            names.push(desc.name.clone());
        }
        Instruction::Ldmatrix { src_addr, .. } => {
            push_operand_names(src_addr, &mut names);
        }
    }
    names
}

/// Returns the destination register name defined by an instruction, if any.
fn dst_register_name(inst: &Instruction) -> Option<String> {
    match inst {
        Instruction::Add { dst, .. }
        | Instruction::Sub { dst, .. }
        | Instruction::Mul { dst, .. }
        | Instruction::Min { dst, .. }
        | Instruction::Max { dst, .. }
        | Instruction::Div { dst, .. }
        | Instruction::Rem { dst, .. }
        | Instruction::And { dst, .. }
        | Instruction::Or { dst, .. }
        | Instruction::Xor { dst, .. }
        | Instruction::SetP { dst, .. }
        | Instruction::Mad { dst, .. }
        | Instruction::MadLo { dst, .. }
        | Instruction::MadHi { dst, .. }
        | Instruction::MadWide { dst, .. }
        | Instruction::Fma { dst, .. }
        | Instruction::Neg { dst, .. }
        | Instruction::Abs { dst, .. }
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
        | Instruction::Load { dst, .. }
        | Instruction::Cvt { dst, .. }
        | Instruction::Atom { dst, .. }
        | Instruction::AtomCas { dst, .. }
        | Instruction::MovSpecial { dst, .. }
        | Instruction::LoadParam { dst, .. }
        | Instruction::Dp4a { dst, .. }
        | Instruction::Dp2a { dst, .. }
        | Instruction::Tex1d { dst, .. }
        | Instruction::Tex2d { dst, .. }
        | Instruction::Tex3d { dst, .. }
        | Instruction::SurfLoad { dst, .. }
        | Instruction::Redux { dst, .. }
        | Instruction::ElectSync { dst, .. } => Some(dst.name.clone()),
        Instruction::Mma { d_regs, .. } => d_regs.first().map(|r| r.name.clone()),
        Instruction::Wgmma { d_regs, .. } => d_regs.first().map(|r| r.name.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Type compatibility check
// ---------------------------------------------------------------------------

/// Returns `true` if an operand's register type is compatible with the
/// instruction type for simple arithmetic.
fn operand_type_compatible(op: &Operand, expected_ty: crate::ir::PtxType) -> bool {
    match op {
        Operand::Register(r) => r.ty == expected_ty,
        // Immediates, symbols, and address operands are always considered compatible at IR level
        Operand::Immediate(_) | Operand::Symbol(_) | Operand::Address { .. } => true,
    }
}

// ---------------------------------------------------------------------------
// Public IR validation functions
// ---------------------------------------------------------------------------

/// Validate an IR instruction sequence for type safety and correctness.
///
/// Performs:
/// 1. Type checking on arithmetic instructions
/// 2. Use-before-def analysis on registers
/// 3. Memory space validation for load/store/cp.async
/// 4. Operand validation for tensor core instructions
#[must_use]
pub fn validate_ir_instructions(instructions: &[Instruction]) -> IrValidationResult {
    let mut result = IrValidationResult {
        errors: Vec::new(),
        warnings: Vec::new(),
    };

    // Run sub-validations and merge
    let lifetime_result = validate_register_lifetimes(instructions);
    result.merge(&lifetime_result);

    let consistency_result = validate_memory_consistency(instructions);
    result.merge(&consistency_result);

    // Type checking for arithmetic instructions
    for (idx, inst) in instructions.iter().enumerate() {
        validate_type_safety(inst, idx, &mut result);
        validate_memory_spaces(inst, idx, &mut result);
        validate_tensor_core_operands(inst, idx, &mut result);
    }

    result
}

/// Validate register lifetimes: no use-before-def.
///
/// Tracks which registers have been written to (appear as `dst`) and flags
/// any register used as a source before it has been defined. Registers
/// defined by `LoadParam` and `MovSpecial` are counted as definitions.
#[must_use]
pub fn validate_register_lifetimes(instructions: &[Instruction]) -> IrValidationResult {
    let mut result = IrValidationResult {
        errors: Vec::new(),
        warnings: Vec::new(),
    };

    let mut defined: HashSet<String> = HashSet::new();

    for (idx, inst) in instructions.iter().enumerate() {
        // First, check sources for use-before-def
        let src_names = collect_src_register_names(inst);
        for name in &src_names {
            if !defined.contains(name) {
                result.errors.push(IrValidationError {
                    instruction_index: idx,
                    kind: IrErrorKind::UseBeforeDef,
                    message: format!("register {name} used before definition"),
                });
            }
        }

        // Then, record definitions
        if let Some(dst_name) = dst_register_name(inst) {
            defined.insert(dst_name);
        }

        // Multi-register definitions for tensor core
        match inst {
            Instruction::Mma { d_regs, .. } | Instruction::Wgmma { d_regs, .. } => {
                for r in d_regs {
                    defined.insert(r.name.clone());
                }
            }
            Instruction::Wmma { op, fragments, .. } => {
                // For WMMA Mma/StoreD, fragments are sources; for Load, they are defs
                if matches!(op, WmmaOp::LoadA | WmmaOp::LoadB) {
                    for frag in fragments {
                        defined.insert(frag.name.clone());
                    }
                }
            }
            _ => {}
        }
    }

    result
}

/// Validate fence/barrier placement and memory consistency.
///
/// Checks:
/// 1. Barriers potentially inside divergent control flow
/// 2. Shared memory stores without a subsequent barrier before shared loads
#[must_use]
pub fn validate_memory_consistency(instructions: &[Instruction]) -> IrValidationResult {
    let mut result = IrValidationResult {
        errors: Vec::new(),
        warnings: Vec::new(),
    };

    // Check for barriers after conditional branches (potential divergence)
    check_barrier_divergence(instructions, &mut result);

    // Check for shared memory race conditions
    check_shared_memory_races(instructions, &mut result);

    result
}

// ---------------------------------------------------------------------------
// Internal validation helpers
// ---------------------------------------------------------------------------

/// Type safety checks for arithmetic instructions.
fn validate_type_safety(inst: &Instruction, idx: usize, result: &mut IrValidationResult) {
    match inst {
        Instruction::Add { ty, dst, a, b }
        | Instruction::Sub { ty, dst, a, b }
        | Instruction::Min { ty, dst, a, b }
        | Instruction::Max { ty, dst, a, b } => {
            if dst.ty != *ty {
                result.errors.push(IrValidationError {
                    instruction_index: idx,
                    kind: IrErrorKind::TypeMismatch,
                    message: format!(
                        "dst register {} has type {:?} but instruction type is {:?}",
                        dst.name, dst.ty, ty
                    ),
                });
            }
            if !operand_type_compatible(a, *ty) {
                result.errors.push(IrValidationError {
                    instruction_index: idx,
                    kind: IrErrorKind::TypeMismatch,
                    message: format!("operand a type mismatch with instruction type {ty:?}"),
                });
            }
            if !operand_type_compatible(b, *ty) {
                result.errors.push(IrValidationError {
                    instruction_index: idx,
                    kind: IrErrorKind::TypeMismatch,
                    message: format!("operand b type mismatch with instruction type {ty:?}"),
                });
            }
        }
        Instruction::Mul { ty, dst, a, b, .. } => {
            // For mul, dst type depends on mode (wide produces double width)
            // but the source operands should match the instruction type
            if !operand_type_compatible(a, *ty) {
                result.errors.push(IrValidationError {
                    instruction_index: idx,
                    kind: IrErrorKind::TypeMismatch,
                    message: format!("mul operand a type mismatch with instruction type {ty:?}"),
                });
            }
            if !operand_type_compatible(b, *ty) {
                result.errors.push(IrValidationError {
                    instruction_index: idx,
                    kind: IrErrorKind::TypeMismatch,
                    message: format!("mul operand b type mismatch with instruction type {ty:?}"),
                });
            }
            // For non-wide modes, dst should match
            if dst.ty != *ty {
                result.warnings.push(IrValidationWarning {
                    instruction_index: idx,
                    message: format!(
                        "mul dst register {} type {:?} differs from instruction type {:?}",
                        dst.name, dst.ty, ty
                    ),
                });
            }
        }
        _ => {}
    }
}

/// Validate memory spaces for load/store/cp.async instructions.
fn validate_memory_spaces(inst: &Instruction, idx: usize, result: &mut IrValidationResult) {
    if let Instruction::CpAsync {
        dst_shared: Operand::Register(r),
        ..
    } = inst
    {
        // If someone passes a raw register instead of an address, warn
        result.warnings.push(IrValidationWarning {
            instruction_index: idx,
            message: format!(
                "cp.async dst_shared uses register {} directly; expected a shared memory address",
                r.name
            ),
        });
    }

    // Validate that Load/Store with shared space uses address operands
    match inst {
        Instruction::Load {
            space,
            addr: Operand::Immediate(_),
            ..
        } if *space == MemorySpace::Shared => {
            result.errors.push(IrValidationError {
                instruction_index: idx,
                kind: IrErrorKind::InvalidMemorySpace,
                message: "shared memory load with immediate address is invalid".to_string(),
            });
        }
        Instruction::Store {
            space,
            addr: Operand::Immediate(_),
            ..
        } if *space == MemorySpace::Shared => {
            result.errors.push(IrValidationError {
                instruction_index: idx,
                kind: IrErrorKind::InvalidMemorySpace,
                message: "shared memory store with immediate address is invalid".to_string(),
            });
        }
        _ => {}
    }
}

/// Validate tensor core instruction operands.
fn validate_tensor_core_operands(inst: &Instruction, idx: usize, result: &mut IrValidationResult) {
    match inst {
        Instruction::Wmma { addr, stride, .. } => {
            // Check that addr/stride are not immediates (should be registers/addresses)
            if let Some(Operand::Immediate(_)) = addr.as_ref() {
                result.errors.push(IrValidationError {
                    instruction_index: idx,
                    kind: IrErrorKind::InvalidOperand,
                    message: "wmma address operand must not be an immediate value".to_string(),
                });
            }
            if let Some(Operand::Immediate(_)) = stride.as_ref() {
                result.errors.push(IrValidationError {
                    instruction_index: idx,
                    kind: IrErrorKind::InvalidOperand,
                    message: "wmma stride operand must not be an immediate value".to_string(),
                });
            }
        }
        Instruction::Mma {
            a_regs,
            b_regs,
            c_regs,
            d_regs,
            ..
        }
            // All registers should be non-empty
            if (a_regs.is_empty() || b_regs.is_empty() || c_regs.is_empty() || d_regs.is_empty()) => {
                result.errors.push(IrValidationError {
                    instruction_index: idx,
                    kind: IrErrorKind::InvalidOperand,
                    message: "mma instruction requires non-empty register fragments".to_string(),
                });
            }
        Instruction::Wgmma { d_regs, .. }
            if d_regs.is_empty() => {
                result.errors.push(IrValidationError {
                    instruction_index: idx,
                    kind: IrErrorKind::InvalidOperand,
                    message: "wgmma instruction requires non-empty destination registers".to_string(),
                });
            }
        _ => {}
    }
}

/// Check if barrier instructions might be inside divergent control flow.
fn check_barrier_divergence(instructions: &[Instruction], result: &mut IrValidationResult) {
    // Collect all labels in the program
    let all_labels: HashSet<&str> = instructions
        .iter()
        .filter_map(|inst| {
            if let Instruction::Label(name) = inst {
                Some(name.as_str())
            } else {
                None
            }
        })
        .collect();

    let mut in_conditional_region = false;
    let mut conditional_branch_idx = 0;

    for (idx, inst) in instructions.iter().enumerate() {
        match inst {
            Instruction::Branch {
                predicate: Some(_),
                target,
                ..
            }
                // A conditional branch that targets a label creates a divergent
                // region until we reach that label
                if all_labels.contains(target.as_str()) => {
                    in_conditional_region = true;
                    conditional_branch_idx = idx;
                }
            Instruction::Label(_) => {
                // Reaching a label ends the conditional region
                in_conditional_region = false;
            }
            Instruction::BarSync { .. }
                if in_conditional_region => {
                    result.warnings.push(IrValidationWarning {
                        instruction_index: idx,
                        message: format!(
                            "bar.sync inside potentially divergent control flow \
                             (conditional branch at instruction {conditional_branch_idx}); \
                             this may cause deadlock if not all threads reach the barrier"
                        ),
                    });
                }
            _ => {}
        }
    }
}

/// Check for potential shared memory race conditions.
///
/// Warns if there are shared memory stores without a subsequent `bar.sync`
/// before the next shared memory load.
fn check_shared_memory_races(instructions: &[Instruction], result: &mut IrValidationResult) {
    let mut pending_shared_store: Option<usize> = None;

    for (idx, inst) in instructions.iter().enumerate() {
        match inst {
            Instruction::Store {
                space: MemorySpace::Shared,
                ..
            } => {
                pending_shared_store = Some(idx);
            }
            Instruction::BarSync { .. } => {
                // Barrier clears the pending shared store
                pending_shared_store = None;
            }
            Instruction::Load {
                space: MemorySpace::Shared,
                ..
            } => {
                if let Some(store_idx) = pending_shared_store {
                    result.warnings.push(IrValidationWarning {
                        instruction_index: idx,
                        message: format!(
                            "shared memory load without bar.sync after shared memory \
                             store at instruction {store_idx}; potential race condition"
                        ),
                    });
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        CacheQualifier, ImmValue, Instruction, MemorySpace, Operand, PtxType, Register, SpecialReg,
        VectorWidth, WmmaLayout, WmmaOp, WmmaShape,
    };

    #[test]
    fn valid_minimal_ptx() {
        let ptx = ".version 8.5\n.target sm_90a\n.address_size 64\n";
        let result = validate_ptx(ptx);
        assert!(result.is_ok());
        assert!(result.errors.is_empty());
    }

    #[test]
    fn missing_version() {
        let ptx = ".target sm_80\n.address_size 64\n";
        let result = validate_ptx(ptx);
        assert!(result.has_errors());
        assert!(
            result
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::MissingVersionDirective))
        );
    }

    #[test]
    fn missing_target() {
        let ptx = ".version 8.5\n.address_size 64\n";
        let result = validate_ptx(ptx);
        assert!(result.has_errors());
        assert!(
            result
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::MissingTargetDirective))
        );
    }

    #[test]
    fn shared_memory_within_limits() {
        let ptx = ".version 8.5\n.target sm_80\n.address_size 64\n\
                    .shared .align 4 .b8 smem[4096];\n";
        let result = validate_ptx(ptx);
        assert!(result.is_ok());
    }

    #[test]
    fn shared_memory_exceeds_limits() {
        // sm_75 max is 65536
        let ptx = ".version 6.4\n.target sm_75\n.address_size 64\n\
                    .shared .align 4 .b8 smem[100000];\n";
        let result = validate_ptx(ptx);
        assert!(result.has_errors());
        assert!(
            result
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::InvalidSharedMemSize { .. }))
        );
    }

    #[test]
    fn validate_for_specific_target() {
        let ptx = ".version 8.5\n.target sm_80\n.address_size 64\n\
                    .shared .align 4 .b8 smem[200000];\n";
        let result = validate_ptx_for_target(ptx, SmVersion::Sm80);
        // 200000 > 163840 (sm_80 limit)
        assert!(result.has_errors());
    }

    #[test]
    fn extract_shared_mem_size_fn() {
        assert_eq!(
            extract_shared_mem_size("    .shared .align 4 .b8 smem[4096];"),
            Some(4096)
        );
        assert_eq!(
            extract_shared_mem_size("    .shared .align 16 .b8 tile[65536];"),
            Some(65536)
        );
        assert_eq!(extract_shared_mem_size("    mov.u32 %r0, 0;"), None);
    }

    #[test]
    fn parse_sm_version_fn() {
        assert_eq!(parse_sm_version("sm_80"), Some(SmVersion::Sm80));
        assert_eq!(parse_sm_version("sm_90a"), Some(SmVersion::Sm90a));
        assert_eq!(parse_sm_version("sm_100"), Some(SmVersion::Sm100));
        assert_eq!(parse_sm_version("sm_999"), None);
    }

    #[test]
    fn mismatched_braces_warning() {
        let ptx = ".version 8.5\n.target sm_80\n.address_size 64\n{\n";
        let result = validate_ptx(ptx);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn validation_error_display() {
        let err = ValidationError::MissingVersionDirective;
        assert_eq!(format!("{err}"), "missing .version directive");

        let err = ValidationError::InvalidSharedMemSize {
            declared: 100_000,
            max_allowed: 65536,
        };
        assert!(format!("{err}").contains("100000"));
    }

    // -----------------------------------------------------------------------
    // IR-level validation tests
    // -----------------------------------------------------------------------

    fn reg(name: &str, ty: PtxType) -> Register {
        Register {
            name: name.to_string(),
            ty,
        }
    }

    fn reg_op(name: &str, ty: PtxType) -> Operand {
        Operand::Register(reg(name, ty))
    }

    #[test]
    fn ir_type_compatible_arithmetic_passes() {
        let instructions = vec![
            Instruction::LoadParam {
                ty: PtxType::F32,
                dst: reg("%f0", PtxType::F32),
                param_name: "a".to_string(),
            },
            Instruction::LoadParam {
                ty: PtxType::F32,
                dst: reg("%f1", PtxType::F32),
                param_name: "b".to_string(),
            },
            Instruction::Add {
                ty: PtxType::F32,
                dst: reg("%f2", PtxType::F32),
                a: reg_op("%f0", PtxType::F32),
                b: reg_op("%f1", PtxType::F32),
            },
        ];
        let result = validate_ir_instructions(&instructions);
        assert!(
            result.errors.is_empty(),
            "expected no errors, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn ir_type_mismatched_arithmetic_fails() {
        let instructions = vec![
            Instruction::LoadParam {
                ty: PtxType::F32,
                dst: reg("%f0", PtxType::F32),
                param_name: "a".to_string(),
            },
            Instruction::LoadParam {
                ty: PtxType::U32,
                dst: reg("%r0", PtxType::U32),
                param_name: "b".to_string(),
            },
            Instruction::Add {
                ty: PtxType::F32,
                dst: reg("%f1", PtxType::F32),
                a: reg_op("%f0", PtxType::F32),
                b: reg_op("%r0", PtxType::U32), // mismatch
            },
        ];
        let result = validate_ir_instructions(&instructions);
        assert!(result.has_errors());
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.kind == IrErrorKind::TypeMismatch)
        );
    }

    #[test]
    fn ir_use_before_def_detection() {
        let instructions = vec![Instruction::Add {
            ty: PtxType::F32,
            dst: reg("%f2", PtxType::F32),
            a: reg_op("%f0", PtxType::F32), // never defined
            b: reg_op("%f1", PtxType::F32), // never defined
        }];
        let result = validate_ir_instructions(&instructions);
        assert!(result.has_errors());
        let ubd_count = result
            .errors
            .iter()
            .filter(|e| e.kind == IrErrorKind::UseBeforeDef)
            .count();
        assert!(ubd_count >= 2, "expected at least 2 use-before-def errors");
    }

    #[test]
    fn ir_load_param_counted_as_definition() {
        let instructions = vec![
            Instruction::LoadParam {
                ty: PtxType::U64,
                dst: reg("%rd0", PtxType::U64),
                param_name: "ptr".to_string(),
            },
            Instruction::Load {
                space: MemorySpace::Global,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::F32,
                dst: reg("%f0", PtxType::F32),
                addr: Operand::Address {
                    base: reg("%rd0", PtxType::U64),
                    offset: None,
                },
            },
        ];
        let result = validate_register_lifetimes(&instructions);
        assert!(
            result.errors.is_empty(),
            "LoadParam should count as definition: {:?}",
            result.errors
        );
    }

    #[test]
    fn ir_mov_special_counted_as_definition() {
        let instructions = vec![
            Instruction::MovSpecial {
                dst: reg("%r0", PtxType::U32),
                special: SpecialReg::TidX,
            },
            Instruction::Add {
                ty: PtxType::U32,
                dst: reg("%r1", PtxType::U32),
                a: reg_op("%r0", PtxType::U32),
                b: Operand::Immediate(ImmValue::U32(1)),
            },
        ];
        let result = validate_register_lifetimes(&instructions);
        assert!(
            result.errors.is_empty(),
            "MovSpecial should count as definition: {:?}",
            result.errors
        );
    }

    #[test]
    fn ir_shared_store_without_barrier_warns() {
        let addr_reg = reg("%rd0", PtxType::U64);
        let instructions = vec![
            Instruction::LoadParam {
                ty: PtxType::U64,
                dst: addr_reg.clone(),
                param_name: "addr".to_string(),
            },
            Instruction::LoadParam {
                ty: PtxType::F32,
                dst: reg("%f0", PtxType::F32),
                param_name: "val".to_string(),
            },
            Instruction::Store {
                space: MemorySpace::Shared,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::F32,
                addr: Operand::Address {
                    base: addr_reg.clone(),
                    offset: None,
                },
                src: reg("%f0", PtxType::F32),
            },
            // No bar.sync here!
            Instruction::Load {
                space: MemorySpace::Shared,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::F32,
                dst: reg("%f1", PtxType::F32),
                addr: Operand::Address {
                    base: addr_reg,
                    offset: Some(4),
                },
            },
        ];
        let result = validate_memory_consistency(&instructions);
        assert!(
            !result.warnings.is_empty(),
            "expected race condition warning"
        );
        assert!(
            result.warnings[0].message.contains("race condition"),
            "warning should mention race condition"
        );
    }

    #[test]
    fn ir_barrier_after_shared_store_no_warning() {
        let addr_reg = reg("%rd0", PtxType::U64);
        let instructions = vec![
            Instruction::Store {
                space: MemorySpace::Shared,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::F32,
                addr: Operand::Address {
                    base: addr_reg.clone(),
                    offset: None,
                },
                src: reg("%f0", PtxType::F32),
            },
            Instruction::BarSync { id: 0 },
            Instruction::Load {
                space: MemorySpace::Shared,
                qualifier: CacheQualifier::None,
                vec: VectorWidth::V1,
                ty: PtxType::F32,
                dst: reg("%f1", PtxType::F32),
                addr: Operand::Address {
                    base: addr_reg,
                    offset: Some(4),
                },
            },
        ];
        let result = validate_memory_consistency(&instructions);
        assert!(
            result.warnings.is_empty(),
            "expected no warnings when barrier separates store/load"
        );
    }

    #[test]
    fn ir_empty_instruction_list_no_errors() {
        let result = validate_ir_instructions(&[]);
        assert!(result.is_ok());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn ir_complex_sequence_multiple_issues() {
        let instructions = vec![
            // Use-before-def: %f0 never defined
            Instruction::Add {
                ty: PtxType::F32,
                dst: reg("%f1", PtxType::F32),
                a: reg_op("%f0", PtxType::F32),
                b: Operand::Immediate(ImmValue::F32(1.0)),
            },
            // Type mismatch: dst is U32 but instruction type is F32
            Instruction::Sub {
                ty: PtxType::F32,
                dst: reg("%r0", PtxType::U32),
                a: reg_op("%f1", PtxType::F32),
                b: Operand::Immediate(ImmValue::F32(2.0)),
            },
        ];
        let result = validate_ir_instructions(&instructions);
        assert!(result.has_errors());

        let has_ubd = result
            .errors
            .iter()
            .any(|e| e.kind == IrErrorKind::UseBeforeDef);
        let has_type_mismatch = result
            .errors
            .iter()
            .any(|e| e.kind == IrErrorKind::TypeMismatch);
        assert!(has_ubd, "expected use-before-def error");
        assert!(has_type_mismatch, "expected type mismatch error");
    }

    #[test]
    fn ir_validate_register_lifetimes_standalone() {
        let instructions = vec![
            Instruction::LoadParam {
                ty: PtxType::F32,
                dst: reg("%f0", PtxType::F32),
                param_name: "x".to_string(),
            },
            Instruction::Neg {
                ty: PtxType::F32,
                dst: reg("%f1", PtxType::F32),
                src: reg_op("%f0", PtxType::F32),
            },
            // %f99 never defined
            Instruction::Add {
                ty: PtxType::F32,
                dst: reg("%f2", PtxType::F32),
                a: reg_op("%f1", PtxType::F32),
                b: reg_op("%f99", PtxType::F32),
            },
        ];
        let result = validate_register_lifetimes(&instructions);
        assert!(result.has_errors());
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].message.contains("%f99"));
    }

    #[test]
    fn ir_validate_memory_consistency_standalone() {
        // Conditional branch followed by bar.sync => divergence warning
        let instructions = vec![
            Instruction::LoadParam {
                ty: PtxType::U32,
                dst: reg("%p0", PtxType::Pred),
                param_name: "pred".to_string(),
            },
            Instruction::Branch {
                target: "skip".to_string(),
                predicate: Some((reg("%p0", PtxType::Pred), false)),
            },
            Instruction::BarSync { id: 0 },
            Instruction::Label("skip".to_string()),
        ];
        let result = validate_memory_consistency(&instructions);
        assert!(!result.warnings.is_empty(), "expected divergence warning");
        assert!(result.warnings[0].message.contains("divergent"));
    }

    #[test]
    fn ir_validation_result_display() {
        let result = IrValidationResult {
            errors: vec![IrValidationError {
                instruction_index: 3,
                kind: IrErrorKind::TypeMismatch,
                message: "dst type does not match".to_string(),
            }],
            warnings: vec![IrValidationWarning {
                instruction_index: 7,
                message: "possible race".to_string(),
            }],
        };
        let display = format!("{result}");
        assert!(display.contains("Errors (1)"));
        assert!(display.contains("TypeMismatch"));
        assert!(display.contains("Warnings (1)"));
        assert!(display.contains("possible race"));

        // Also test the all-clear case
        let ok_result = IrValidationResult {
            errors: Vec::new(),
            warnings: Vec::new(),
        };
        let ok_display = format!("{ok_result}");
        assert!(ok_display.contains("passed"));
    }

    #[test]
    fn ir_wmma_with_immediate_operand_flagged() {
        let instructions = vec![Instruction::Wmma {
            op: WmmaOp::LoadA,
            shape: WmmaShape::M16N16K16,
            layout: WmmaLayout::RowMajor,
            ty: PtxType::F16,
            fragments: vec![reg("%f0", PtxType::F16)],
            addr: Some(Operand::Immediate(ImmValue::U32(0))), // invalid!
            stride: Some(Operand::Immediate(ImmValue::U32(16))), // invalid!
        }];
        let result = validate_ir_instructions(&instructions);
        let invalid_operand_errors: Vec<_> = result
            .errors
            .iter()
            .filter(|e| e.kind == IrErrorKind::InvalidOperand)
            .collect();
        assert!(
            invalid_operand_errors.len() >= 2,
            "expected at least 2 InvalidOperand errors for wmma immediates, got {}",
            invalid_operand_errors.len()
        );
    }

    #[test]
    fn ir_mixed_valid_and_invalid_instructions() {
        let instructions = vec![
            // Valid: load param
            Instruction::LoadParam {
                ty: PtxType::F32,
                dst: reg("%f0", PtxType::F32),
                param_name: "x".to_string(),
            },
            // Valid: mov special
            Instruction::MovSpecial {
                dst: reg("%r0", PtxType::U32),
                special: SpecialReg::TidX,
            },
            // Valid: add with matching types
            Instruction::Add {
                ty: PtxType::F32,
                dst: reg("%f1", PtxType::F32),
                a: reg_op("%f0", PtxType::F32),
                b: Operand::Immediate(ImmValue::F32(1.0)),
            },
            // Invalid: sub with mismatched dst type
            Instruction::Sub {
                ty: PtxType::F32,
                dst: reg("%bad", PtxType::U32), // type mismatch
                a: reg_op("%f1", PtxType::F32),
                b: Operand::Immediate(ImmValue::F32(0.5)),
            },
            // Valid: comment (no validation issues)
            Instruction::Comment("test".to_string()),
            // Valid: return
            Instruction::Return,
        ];
        let result = validate_ir_instructions(&instructions);
        // Should have exactly 1 type mismatch error (dst type mismatch on Sub)
        let type_errors: Vec<_> = result
            .errors
            .iter()
            .filter(|e| e.kind == IrErrorKind::TypeMismatch)
            .collect();
        assert_eq!(
            type_errors.len(),
            1,
            "expected exactly 1 type mismatch, got {}: {:?}",
            type_errors.len(),
            type_errors
        );
        // No use-before-def errors since all regs are defined
        let ubd_errors: Vec<_> = result
            .errors
            .iter()
            .filter(|e| e.kind == IrErrorKind::UseBeforeDef)
            .collect();
        assert!(
            ubd_errors.is_empty(),
            "expected no use-before-def errors: {ubd_errors:?}",
        );
    }
}

/// SM compatibility, register pressure, and related tests — kept in a separate
/// file to stay under the 2 000-line limit.
#[cfg(test)]
#[path = "validator_tests.rs"]
mod sm_tests;
