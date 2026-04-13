//! Visual PTX explorer for terminal-based PTX analysis.
//!
//! This module provides a set of rendering and analysis utilities that produce
//! formatted ASCII/ANSI text output for inspecting PTX intermediate representation.
//! It analyzes [`PtxModule`], [`PtxFunction`], [`Instruction`], and [`BasicBlock`]
//! structures, producing pretty-printed PTX code, control flow graphs, register
//! lifetime timelines, instruction mix bar charts, and more.
//!
//! This is **not** a live TUI application -- it requires no TUI framework. All
//! output is returned as plain `String` values suitable for printing to a terminal
//! or writing to a file.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::tui_explorer::{ExplorerConfig, PtxExplorer};
//! use oxicuda_ptx::ir::{PtxFunction, PtxType};
//!
//! let config = ExplorerConfig::default();
//! let explorer = PtxExplorer::new(config);
//! let func = PtxFunction::new("my_kernel");
//! let output = explorer.render_function(&func);
//! assert!(!output.is_empty());
//! ```

use std::collections::HashMap;
use std::fmt::Write;

use crate::ir::{BasicBlock, Instruction, MemorySpace, Operand, PtxFunction, PtxModule};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the PTX explorer rendering engine.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ExplorerConfig {
    /// Whether to emit ANSI color codes in output.
    pub use_color: bool,
    /// Maximum output width in columns.
    pub max_width: usize,
    /// Whether to show line numbers alongside instructions.
    pub show_line_numbers: bool,
    /// Whether to annotate registers with their types.
    pub show_register_types: bool,
    /// Whether to show estimated instruction latency.
    pub show_instruction_latency: bool,
}

impl Default for ExplorerConfig {
    fn default() -> Self {
        Self {
            use_color: false,
            max_width: 120,
            show_line_numbers: false,
            show_register_types: false,
            show_instruction_latency: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction categorisation
// ---------------------------------------------------------------------------

/// Category of a PTX instruction for analysis purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstructionCategory {
    /// Arithmetic / math operations (add, mul, fma, etc.).
    Arithmetic,
    /// Memory operations (load, store, cp.async, atom, etc.).
    Memory,
    /// Control flow operations (branch, label, return).
    Control,
    /// Synchronization primitives (bar.sync, fence, mbarrier, etc.).
    Synchronization,
    /// Tensor Core operations (wmma, mma, wgmma).
    TensorCore,
    /// Special operations (mov.special, load.param, comment, raw, pragma).
    Special,
    /// Type conversion operations (cvt).
    Conversion,
}

impl InstructionCategory {
    /// Returns a human-readable label for this category.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Arithmetic => "Arithmetic",
            Self::Memory => "Memory",
            Self::Control => "Control",
            Self::Synchronization => "Sync",
            Self::TensorCore => "TensorCore",
            Self::Special => "Special",
            Self::Conversion => "Conversion",
        }
    }

    /// Returns an ANSI color escape code for this category.
    const fn ansi_color(self) -> &'static str {
        match self {
            Self::Arithmetic => "\x1b[32m",      // green
            Self::Memory => "\x1b[34m",          // blue
            Self::Control => "\x1b[33m",         // yellow
            Self::Synchronization => "\x1b[35m", // magenta
            Self::TensorCore => "\x1b[36m",      // cyan
            Self::Special => "\x1b[90m",         // bright black (gray)
            Self::Conversion => "\x1b[37m",      // white
        }
    }
}

/// Detailed information about a single instruction.
#[derive(Debug, Clone)]
pub struct InstructionInfo {
    /// The instruction text (PTX emission).
    pub instruction: String,
    /// Instruction category.
    pub category: InstructionCategory,
    /// Estimated latency in GPU clock cycles.
    pub latency_cycles: u32,
    /// Estimated throughput (instructions per SM per cycle).
    pub throughput_per_sm: f64,
    /// Registers read by this instruction.
    pub registers_read: Vec<String>,
    /// Registers written by this instruction.
    pub registers_written: Vec<String>,
}

// ---------------------------------------------------------------------------
// Register lifetime
// ---------------------------------------------------------------------------

/// Lifetime information for a single PTX register.
#[derive(Debug, Clone)]
pub struct RegisterLifetime {
    /// Register name (e.g., `%f0`).
    pub register: String,
    /// Register type string (e.g., `.f32`).
    pub reg_type: String,
    /// Instruction index of the first definition.
    pub first_def: usize,
    /// Instruction index of the last use.
    pub last_use: usize,
    /// Total number of uses (reads).
    pub num_uses: usize,
}

// ---------------------------------------------------------------------------
// Instruction mix
// ---------------------------------------------------------------------------

/// Instruction mix statistics for a function.
#[derive(Debug, Clone)]
pub struct InstructionMix {
    /// Per-category instruction counts.
    pub counts: HashMap<InstructionCategory, usize>,
    /// Total number of instructions analysed.
    pub total: usize,
}

// ---------------------------------------------------------------------------
// Memory report
// ---------------------------------------------------------------------------

/// Memory access pattern analysis report.
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Number of global memory load instructions.
    pub global_loads: usize,
    /// Number of global memory store instructions.
    pub global_stores: usize,
    /// Number of shared memory load instructions.
    pub shared_loads: usize,
    /// Number of shared memory store instructions.
    pub shared_stores: usize,
    /// Number of local memory load instructions.
    pub local_loads: usize,
    /// Number of local memory store instructions.
    pub local_stores: usize,
    /// Estimated coalescing score (0.0 = uncoalesced, 1.0 = perfectly coalesced).
    pub coalescing_score: f64,
}

// ---------------------------------------------------------------------------
// Diff report
// ---------------------------------------------------------------------------

/// Report comparing two PTX functions.
#[derive(Debug, Clone)]
pub struct DiffReport {
    /// Number of instructions present in B but not in A.
    pub added_instructions: usize,
    /// Number of instructions present in A but not in B.
    pub removed_instructions: usize,
    /// Number of basic blocks that differ between A and B.
    pub changed_blocks: usize,
    /// Change in total register count (B - A). Positive means B uses more registers.
    pub register_delta: i32,
}

// ---------------------------------------------------------------------------
// Complexity metrics
// ---------------------------------------------------------------------------

/// Kernel complexity analysis results.
#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    /// Total instruction count in the function body.
    pub instruction_count: usize,
    /// Number of branch instructions.
    pub branch_count: usize,
    /// Estimated loop count (number of back-edges detected).
    pub loop_count: usize,
    /// Maximum number of live registers at any point.
    pub max_register_pressure: usize,
    /// Estimated occupancy percentage (0.0 -- 100.0).
    pub estimated_occupancy_pct: f64,
    /// Arithmetic intensity (arithmetic ops / memory ops).
    pub arithmetic_intensity: f64,
}

// ---------------------------------------------------------------------------
// Helpers -- instruction classification
// ---------------------------------------------------------------------------

/// Categorise a PTX [`Instruction`] into an [`InstructionCategory`].
const fn categorize_instruction(inst: &Instruction) -> InstructionCategory {
    match inst {
        // Arithmetic
        Instruction::Add { .. }
        | Instruction::Sub { .. }
        | Instruction::Mul { .. }
        | Instruction::Mad { .. }
        | Instruction::MadLo { .. }
        | Instruction::MadHi { .. }
        | Instruction::MadWide { .. }
        | Instruction::Fma { .. }
        | Instruction::Neg { .. }
        | Instruction::Abs { .. }
        | Instruction::Min { .. }
        | Instruction::Max { .. }
        | Instruction::Brev { .. }
        | Instruction::Clz { .. }
        | Instruction::Popc { .. }
        | Instruction::Bfind { .. }
        | Instruction::Bfe { .. }
        | Instruction::Bfi { .. }
        | Instruction::Shl { .. }
        | Instruction::Shr { .. }
        | Instruction::Div { .. }
        | Instruction::Rem { .. }
        | Instruction::And { .. }
        | Instruction::Or { .. }
        | Instruction::Xor { .. }
        | Instruction::Rcp { .. }
        | Instruction::Rsqrt { .. }
        | Instruction::Sqrt { .. }
        | Instruction::Ex2 { .. }
        | Instruction::Lg2 { .. }
        | Instruction::Sin { .. }
        | Instruction::Cos { .. }
        | Instruction::Dp4a { .. }
        | Instruction::Dp2a { .. }
        | Instruction::SetP { .. } => InstructionCategory::Arithmetic,

        // Memory
        Instruction::Load { .. }
        | Instruction::Store { .. }
        | Instruction::CpAsync { .. }
        | Instruction::CpAsyncCommit
        | Instruction::CpAsyncWait { .. }
        | Instruction::Atom { .. }
        | Instruction::AtomCas { .. }
        | Instruction::Red { .. }
        | Instruction::TmaLoad { .. }
        | Instruction::Tex1d { .. }
        | Instruction::Tex2d { .. }
        | Instruction::Tex3d { .. }
        | Instruction::SurfLoad { .. }
        | Instruction::SurfStore { .. }
        | Instruction::Stmatrix { .. }
        | Instruction::CpAsyncBulk { .. }
        | Instruction::Ldmatrix { .. } => InstructionCategory::Memory,

        // Control
        Instruction::Branch { .. } | Instruction::Label(_) | Instruction::Return => {
            InstructionCategory::Control
        }

        // Synchronization
        Instruction::BarSync { .. }
        | Instruction::BarArrive { .. }
        | Instruction::FenceAcqRel { .. }
        | Instruction::FenceProxy { .. }
        | Instruction::MbarrierInit { .. }
        | Instruction::MbarrierArrive { .. }
        | Instruction::MbarrierWait { .. }
        | Instruction::ElectSync { .. }
        | Instruction::Griddepcontrol { .. }
        | Instruction::Redux { .. }
        | Instruction::BarrierCluster
        | Instruction::FenceCluster => InstructionCategory::Synchronization,

        // Tensor Core
        Instruction::Wmma { .. }
        | Instruction::Mma { .. }
        | Instruction::Wgmma { .. }
        | Instruction::Tcgen05Mma { .. } => InstructionCategory::TensorCore,

        // Conversion
        Instruction::Cvt { .. } => InstructionCategory::Conversion,

        // Special
        Instruction::MovSpecial { .. }
        | Instruction::LoadParam { .. }
        | Instruction::Comment(_)
        | Instruction::Raw(_)
        | Instruction::Pragma(_)
        | Instruction::Setmaxnreg { .. } => InstructionCategory::Special,
    }
}

/// Estimate latency in clock cycles for an instruction.
#[allow(clippy::match_same_arms)]
const fn estimate_latency(inst: &Instruction) -> u32 {
    match inst {
        // Arithmetic -- single-cycle ALU
        Instruction::Add { .. }
        | Instruction::Sub { .. }
        | Instruction::Neg { .. }
        | Instruction::Abs { .. }
        | Instruction::Min { .. }
        | Instruction::Max { .. }
        | Instruction::And { .. }
        | Instruction::Or { .. }
        | Instruction::Xor { .. }
        | Instruction::Shl { .. }
        | Instruction::Shr { .. }
        | Instruction::SetP { .. } => 4,

        Instruction::Mul { .. }
        | Instruction::Mad { .. }
        | Instruction::MadLo { .. }
        | Instruction::MadHi { .. }
        | Instruction::MadWide { .. }
        | Instruction::Fma { .. } => 4,

        // Special math (multi-cycle)
        Instruction::Div { .. } | Instruction::Rem { .. } => 32,
        Instruction::Rcp { .. } | Instruction::Rsqrt { .. } | Instruction::Sqrt { .. } => 8,
        Instruction::Ex2 { .. }
        | Instruction::Lg2 { .. }
        | Instruction::Sin { .. }
        | Instruction::Cos { .. } => 8,

        // Bit manipulation
        Instruction::Brev { .. }
        | Instruction::Clz { .. }
        | Instruction::Popc { .. }
        | Instruction::Bfind { .. }
        | Instruction::Bfe { .. }
        | Instruction::Bfi { .. } => 4,

        // Dot product
        Instruction::Dp4a { .. } | Instruction::Dp2a { .. } => 8,

        // Memory
        Instruction::Load { .. } => 200,
        Instruction::Store { .. } => 200,
        Instruction::CpAsync { .. } => 200,
        Instruction::CpAsyncCommit | Instruction::CpAsyncWait { .. } => 4,
        Instruction::Atom { .. } | Instruction::AtomCas { .. } | Instruction::Red { .. } => 200,
        Instruction::TmaLoad { .. } | Instruction::CpAsyncBulk { .. } => 200,
        Instruction::Tex1d { .. } | Instruction::Tex2d { .. } | Instruction::Tex3d { .. } => 200,
        Instruction::SurfLoad { .. } | Instruction::SurfStore { .. } => 200,
        Instruction::Stmatrix { .. } => 32,
        Instruction::Ldmatrix { .. } => 20,

        // Control
        Instruction::Branch { .. } => 8,
        Instruction::Label(_) | Instruction::Return => 0,

        // Synchronization
        Instruction::BarSync { .. }
        | Instruction::BarArrive { .. }
        | Instruction::FenceAcqRel { .. }
        | Instruction::FenceProxy { .. }
        | Instruction::MbarrierInit { .. }
        | Instruction::MbarrierArrive { .. }
        | Instruction::MbarrierWait { .. }
        | Instruction::ElectSync { .. }
        | Instruction::Griddepcontrol { .. }
        | Instruction::Redux { .. }
        | Instruction::BarrierCluster
        | Instruction::FenceCluster => 16,

        // Tensor Core
        Instruction::Wmma { .. } => 32,
        Instruction::Mma { .. } => 16,
        Instruction::Wgmma { .. } => 64,
        Instruction::Tcgen05Mma { .. } => 64,

        // Conversion
        Instruction::Cvt { .. } => 4,

        // Special / meta
        Instruction::MovSpecial { .. } | Instruction::LoadParam { .. } => 4,
        Instruction::Comment(_) | Instruction::Raw(_) | Instruction::Pragma(_) => 0,
        Instruction::Setmaxnreg { .. } => 0,
    }
}

/// Estimate throughput per SM per cycle for an instruction.
const fn estimate_throughput(inst: &Instruction) -> f64 {
    match categorize_instruction(inst) {
        InstructionCategory::Arithmetic => 64.0,
        InstructionCategory::Memory
        | InstructionCategory::Control
        | InstructionCategory::Special
        | InstructionCategory::Conversion => 32.0,
        InstructionCategory::Synchronization => 16.0,
        InstructionCategory::TensorCore => 1.0,
    }
}

/// Extract register names that an instruction reads.
fn registers_read(inst: &Instruction) -> Vec<String> {
    let mut regs = Vec::new();
    let mut push_operand = |op: &Operand| match op {
        Operand::Register(r) => regs.push(r.name.clone()),
        Operand::Address { base, .. } => regs.push(base.name.clone()),
        _ => {}
    };

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
            push_operand(a);
            push_operand(b);
        }
        Instruction::Mad { a, b, c, .. }
        | Instruction::MadLo { a, b, c, .. }
        | Instruction::MadHi { a, b, c, .. }
        | Instruction::MadWide { a, b, c, .. }
        | Instruction::Fma { a, b, c, .. } => {
            push_operand(a);
            push_operand(b);
            push_operand(c);
        }
        Instruction::Neg { src, .. }
        | Instruction::Abs { src, .. }
        | Instruction::Brev { src, .. }
        | Instruction::Clz { src, .. }
        | Instruction::Popc { src, .. }
        | Instruction::Bfind { src, .. }
        | Instruction::Cvt { src, .. }
        | Instruction::Rcp { src, .. }
        | Instruction::Rsqrt { src, .. }
        | Instruction::Sqrt { src, .. }
        | Instruction::Ex2 { src, .. }
        | Instruction::Lg2 { src, .. }
        | Instruction::Sin { src, .. }
        | Instruction::Cos { src, .. } => {
            push_operand(src);
        }
        Instruction::Load { addr, .. } => {
            push_operand(addr);
        }
        Instruction::Store { addr, src, .. } => {
            push_operand(addr);
            regs.push(src.name.clone());
        }
        Instruction::Branch {
            predicate: Some((pred, _)),
            ..
        } => {
            regs.push(pred.name.clone());
        }
        Instruction::Shl { src, amount, .. } | Instruction::Shr { src, amount, .. } => {
            push_operand(src);
            push_operand(amount);
        }
        _ => {}
    }
    regs
}

/// Extract register names that an instruction writes.
fn registers_written(inst: &Instruction) -> Vec<String> {
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
        | Instruction::Brev { dst, .. }
        | Instruction::Clz { dst, .. }
        | Instruction::Popc { dst, .. }
        | Instruction::Bfind { dst, .. }
        | Instruction::Bfe { dst, .. }
        | Instruction::Bfi { dst, .. }
        | Instruction::Shl { dst, .. }
        | Instruction::Shr { dst, .. }
        | Instruction::Div { dst, .. }
        | Instruction::Rem { dst, .. }
        | Instruction::And { dst, .. }
        | Instruction::Or { dst, .. }
        | Instruction::Xor { dst, .. }
        | Instruction::SetP { dst, .. }
        | Instruction::Load { dst, .. }
        | Instruction::Cvt { dst, .. }
        | Instruction::Atom { dst, .. }
        | Instruction::AtomCas { dst, .. }
        | Instruction::MovSpecial { dst, .. }
        | Instruction::LoadParam { dst, .. }
        | Instruction::Rcp { dst, .. }
        | Instruction::Rsqrt { dst, .. }
        | Instruction::Sqrt { dst, .. }
        | Instruction::Ex2 { dst, .. }
        | Instruction::Lg2 { dst, .. }
        | Instruction::Sin { dst, .. }
        | Instruction::Cos { dst, .. }
        | Instruction::Dp4a { dst, .. }
        | Instruction::Dp2a { dst, .. }
        | Instruction::Tex1d { dst, .. }
        | Instruction::Tex2d { dst, .. }
        | Instruction::Tex3d { dst, .. }
        | Instruction::SurfLoad { dst, .. }
        | Instruction::Redux { dst, .. }
        | Instruction::ElectSync { dst, .. } => vec![dst.name.clone()],
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// ANSI helpers
// ---------------------------------------------------------------------------

const ANSI_RESET: &str = "\x1b[0m";
const ANSI_BOLD: &str = "\x1b[1m";

fn colorize(text: &str, color: &str, use_color: bool) -> String {
    if use_color {
        format!("{color}{text}{ANSI_RESET}")
    } else {
        text.to_string()
    }
}

// ---------------------------------------------------------------------------
// PtxExplorer -- main analysis and rendering engine
// ---------------------------------------------------------------------------

/// Main PTX analysis and rendering engine.
///
/// `PtxExplorer` provides methods to render PTX functions and modules as
/// formatted text, including syntax-highlighted code, control flow graphs,
/// register lifetime diagrams, and instruction mix charts.
#[derive(Debug, Clone)]
pub struct PtxExplorer {
    config: ExplorerConfig,
}

impl PtxExplorer {
    /// Creates a new explorer with the given configuration.
    #[must_use]
    pub const fn new(config: ExplorerConfig) -> Self {
        Self { config }
    }

    /// Renders a single PTX function as pretty-printed text.
    ///
    /// If `use_color` is enabled, ANSI escape codes are used for syntax
    /// highlighting of different instruction categories.
    #[must_use]
    pub fn render_function(&self, func: &PtxFunction) -> String {
        let mut out = String::new();
        let header = format!(".entry {} (", func.name);
        let _ = writeln!(
            out,
            "{}",
            colorize(&header, ANSI_BOLD, self.config.use_color)
        );

        for (i, (name, ty)) in func.params.iter().enumerate() {
            let comma = if i + 1 < func.params.len() { "," } else { "" };
            let _ = writeln!(out, "    .param {} {}{}", ty.as_ptx_str(), name, comma);
        }
        let _ = writeln!(out, ")");
        let _ = writeln!(out, "{{");

        for (idx, inst) in func.body.iter().enumerate() {
            let cat = categorize_instruction(inst);
            let emitted = inst.emit();
            let line = if self.config.show_line_numbers {
                format!("{:>4}  {}", idx + 1, emitted)
            } else {
                format!("    {emitted}")
            };

            let line = if self.config.show_instruction_latency {
                let lat = estimate_latency(inst);
                if lat > 0 {
                    let pad = self.config.max_width.saturating_sub(line.len()).max(2);
                    format!("{line}{:>pad$}", format!("// ~{lat} cycles"), pad = pad)
                } else {
                    line
                }
            } else {
                line
            };

            let _ = writeln!(
                out,
                "{}",
                colorize(&line, cat.ansi_color(), self.config.use_color)
            );
        }

        let _ = writeln!(out, "}}");
        out
    }

    /// Renders an entire PTX module.
    #[must_use]
    pub fn render_module(&self, module: &PtxModule) -> String {
        let mut out = String::new();
        let _ = writeln!(out, ".version {}", module.version);
        let _ = writeln!(out, ".target {}", module.target);
        let _ = writeln!(out, ".address_size {}", module.address_size);
        let _ = writeln!(out);

        for func in &module.functions {
            out.push_str(&self.render_function(func));
            let _ = writeln!(out);
        }
        out
    }

    /// Renders a control flow graph for a function as ASCII art.
    ///
    /// Uses basic blocks derived from `Label` and `Branch` variants of
    /// [`Instruction`] in the function body.
    /// Each block is drawn as a box with its label and instruction count;
    /// edges show branch targets.
    #[must_use]
    pub fn render_cfg(&self, func: &PtxFunction) -> String {
        let blocks = split_into_blocks(&func.body);
        let renderer = CfgRenderer;
        renderer.render(&blocks)
    }

    /// Renders a register lifetime timeline for the function.
    #[must_use]
    pub fn render_register_lifetime(&self, func: &PtxFunction) -> String {
        let analyzer = RegisterLifetimeAnalyzer;
        let lifetimes = analyzer.analyze(func);
        RegisterLifetimeAnalyzer::render_timeline(&lifetimes, self.config.max_width)
    }

    /// Renders an instruction mix bar chart for the function.
    #[must_use]
    pub fn render_instruction_mix(&self, func: &PtxFunction) -> String {
        let analyzer = InstructionMixAnalyzer;
        let mix = analyzer.analyze(func);
        InstructionMixAnalyzer::render_bar_chart(&mix, self.config.max_width)
    }

    /// Renders a data dependency graph for a single basic block.
    #[must_use]
    pub fn render_dependency_graph(&self, block: &BasicBlock) -> String {
        let mut out = String::new();
        let label = block.label.as_deref().unwrap_or("(unnamed)");
        let _ = writeln!(out, "Dependency graph for block: {label}");
        let _ = writeln!(out, "{}", "-".repeat(40));

        // Build a map from register name -> instruction index that last wrote it
        let mut last_writer: HashMap<String, usize> = HashMap::new();
        // edges: (from_idx, to_idx, register)
        let mut edges: Vec<(usize, usize, String)> = Vec::new();

        for (idx, inst) in block.instructions.iter().enumerate() {
            // Check reads -- any register read that has a prior writer creates an edge
            for reg in registers_read(inst) {
                if let Some(&writer_idx) = last_writer.get(&reg) {
                    edges.push((writer_idx, idx, reg));
                }
            }
            // Record writes
            for reg in registers_written(inst) {
                last_writer.insert(reg, idx);
            }
        }

        if edges.is_empty() {
            let _ = writeln!(out, "(no data dependencies)");
        } else {
            for (from, to, reg) in &edges {
                let from_text = block
                    .instructions
                    .get(*from)
                    .map_or_else(|| "?".to_string(), |i| truncate_emit(i, 40));
                let to_text = block
                    .instructions
                    .get(*to)
                    .map_or_else(|| "?".to_string(), |i| truncate_emit(i, 40));
                let _ = writeln!(out, "[{from}] {from_text}");
                let _ = writeln!(out, "  --({reg})--> [{to}] {to_text}");
            }
        }
        out
    }
}

/// Analyse an instruction and return detailed information.
#[must_use]
pub fn analyze_instruction(inst: &Instruction) -> InstructionInfo {
    InstructionInfo {
        instruction: inst.emit(),
        category: categorize_instruction(inst),
        latency_cycles: estimate_latency(inst),
        throughput_per_sm: estimate_throughput(inst),
        registers_read: registers_read(inst),
        registers_written: registers_written(inst),
    }
}

// ---------------------------------------------------------------------------
// CfgRenderer
// ---------------------------------------------------------------------------

/// Control flow graph renderer producing ASCII box-and-arrow diagrams.
pub struct CfgRenderer;

impl CfgRenderer {
    /// Renders the given basic blocks as an ASCII CFG diagram.
    ///
    /// Each block is drawn as a bordered box containing the block label and
    /// instruction count. Edges are drawn as arrows between blocks.
    #[must_use]
    pub fn render(&self, blocks: &[BasicBlock]) -> String {
        if blocks.is_empty() {
            return "(empty CFG)\n".to_string();
        }

        let mut out = String::new();
        let _ = writeln!(out, "Control Flow Graph");
        let _ = writeln!(out, "==================");
        let _ = writeln!(out);

        // Build label -> block index map
        let mut label_to_idx: HashMap<&str, usize> = HashMap::new();
        for (idx, blk) in blocks.iter().enumerate() {
            if let Some(ref label) = blk.label {
                label_to_idx.insert(label.as_str(), idx);
            }
        }

        // Collect edges: (from_block_idx, to_block_idx)
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for (idx, blk) in blocks.iter().enumerate() {
            for inst in &blk.instructions {
                if let Instruction::Branch { target, .. } = inst {
                    if let Some(&target_idx) = label_to_idx.get(target.as_str()) {
                        edges.push((idx, target_idx));
                    }
                }
            }
            // Fall-through edge to next block (if last instruction is not an
            // unconditional branch or return)
            let is_terminal = blk.instructions.last().is_some_and(|i| {
                matches!(
                    i,
                    Instruction::Return
                        | Instruction::Branch {
                            predicate: None,
                            ..
                        }
                )
            });
            if !is_terminal && idx + 1 < blocks.len() {
                edges.push((idx, idx + 1));
            }
        }

        // Draw blocks
        for (idx, blk) in blocks.iter().enumerate() {
            let label = blk.label.as_deref().unwrap_or("(entry)");
            let box_content = format!("B{idx}: {label} ({} insts)", blk.instructions.len());
            let box_width = box_content.len() + 4;
            let border = "+".to_string() + &"-".repeat(box_width - 2) + "+";
            let _ = writeln!(out, "{border}");
            let _ = writeln!(out, "| {box_content} |");
            let _ = writeln!(out, "{border}");

            // Show outgoing edges from this block
            let outgoing: Vec<&(usize, usize)> = edges.iter().filter(|(f, _)| *f == idx).collect();
            for (_, to) in outgoing {
                let target_label = blocks
                    .get(*to)
                    .and_then(|b| b.label.as_deref())
                    .unwrap_or("(next)");
                let _ = writeln!(out, "    |");
                let _ = writeln!(out, "    +--> B{to}: {target_label}");
            }
            let _ = writeln!(out);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// RegisterLifetimeAnalyzer
// ---------------------------------------------------------------------------

/// Analyzes register lifetimes (live ranges) across a function.
pub struct RegisterLifetimeAnalyzer;

impl RegisterLifetimeAnalyzer {
    /// Analyses a PTX function and returns lifetime information for each register.
    #[must_use]
    pub fn analyze(&self, func: &PtxFunction) -> Vec<RegisterLifetime> {
        let mut first_defs: HashMap<String, (usize, String)> = HashMap::new();
        let mut last_uses: HashMap<String, usize> = HashMap::new();
        let mut use_counts: HashMap<String, usize> = HashMap::new();

        for (idx, inst) in func.body.iter().enumerate() {
            // Written registers -- first definition
            for reg in registers_written(inst) {
                first_defs.entry(reg.clone()).or_insert_with(|| {
                    let reg_type = Self::infer_type(inst, &reg);
                    (idx, reg_type)
                });
                // A write is also a "use" of the register slot
                last_uses.insert(reg, idx);
            }
            // Read registers
            for reg in registers_read(inst) {
                last_uses.insert(reg.clone(), idx);
                *use_counts.entry(reg).or_insert(0) += 1;
            }
        }

        let mut lifetimes: Vec<RegisterLifetime> = first_defs
            .into_iter()
            .map(|(reg, (def_idx, reg_type))| {
                let last = last_uses.get(&reg).copied().unwrap_or(def_idx);
                let uses = use_counts.get(&reg).copied().unwrap_or(0);
                RegisterLifetime {
                    register: reg,
                    reg_type,
                    first_def: def_idx,
                    last_use: last,
                    num_uses: uses,
                }
            })
            .collect();

        lifetimes.sort_by_key(|l| (l.first_def, l.register.clone()));
        lifetimes
    }

    /// Renders a horizontal timeline of register lifetimes.
    #[must_use]
    pub fn render_timeline(lifetimes: &[RegisterLifetime], max_width: usize) -> String {
        if lifetimes.is_empty() {
            return "(no registers)\n".to_string();
        }

        let mut out = String::new();
        let _ = writeln!(out, "Register Lifetimes");
        let _ = writeln!(out, "==================");
        let _ = writeln!(out);

        // Determine scaling
        let max_inst = lifetimes
            .iter()
            .map(|l| l.last_use)
            .max()
            .unwrap_or(0)
            .max(1);

        let name_col_width = lifetimes
            .iter()
            .map(|l| l.register.len())
            .max()
            .unwrap_or(4)
            .max(4);
        let type_col_width = lifetimes
            .iter()
            .map(|l| l.reg_type.len())
            .max()
            .unwrap_or(4)
            .max(4);

        // Available width for the bar
        let bar_width = max_width
            .saturating_sub(name_col_width + type_col_width + 10)
            .max(10);

        let _ = writeln!(
            out,
            "{:>nw$}  {:>tw$}  Lifetime",
            "Reg",
            "Type",
            nw = name_col_width,
            tw = type_col_width
        );
        let _ = writeln!(
            out,
            "{}  {}  {}",
            "-".repeat(name_col_width),
            "-".repeat(type_col_width),
            "-".repeat(bar_width),
        );

        for lt in lifetimes {
            let start_pos = (lt.first_def * bar_width) / max_inst.max(1);
            let end_pos = (lt.last_use * bar_width) / max_inst.max(1);
            let end_pos = end_pos.max(start_pos + 1).min(bar_width);

            let mut bar = vec![' '; bar_width];
            for ch in bar.iter_mut().take(end_pos).skip(start_pos) {
                *ch = '#';
            }
            let bar_str: String = bar.into_iter().collect();

            let _ = writeln!(
                out,
                "{:>nw$}  {:>tw$}  {bar_str}  (uses: {})",
                lt.register,
                lt.reg_type,
                lt.num_uses,
                nw = name_col_width,
                tw = type_col_width,
            );
        }
        out
    }

    /// Infer a type string for a register from the instruction that defines it.
    fn infer_type(inst: &Instruction, _reg: &str) -> String {
        match inst {
            Instruction::Add { ty, .. }
            | Instruction::Sub { ty, .. }
            | Instruction::Mul { ty, .. }
            | Instruction::Min { ty, .. }
            | Instruction::Max { ty, .. }
            | Instruction::Neg { ty, .. }
            | Instruction::Abs { ty, .. }
            | Instruction::Div { ty, .. }
            | Instruction::Rem { ty, .. }
            | Instruction::And { ty, .. }
            | Instruction::Or { ty, .. }
            | Instruction::Xor { ty, .. }
            | Instruction::Shl { ty, .. }
            | Instruction::Shr { ty, .. }
            | Instruction::Load { ty, .. }
            | Instruction::Brev { ty, .. }
            | Instruction::Clz { ty, .. }
            | Instruction::Popc { ty, .. }
            | Instruction::Bfind { ty, .. }
            | Instruction::Bfe { ty, .. }
            | Instruction::Bfi { ty, .. }
            | Instruction::Rcp { ty, .. }
            | Instruction::Rsqrt { ty, .. }
            | Instruction::Sqrt { ty, .. }
            | Instruction::Ex2 { ty, .. }
            | Instruction::Lg2 { ty, .. }
            | Instruction::Sin { ty, .. }
            | Instruction::Cos { ty, .. }
            | Instruction::Tex1d { ty, .. }
            | Instruction::Tex2d { ty, .. }
            | Instruction::Tex3d { ty, .. }
            | Instruction::SurfLoad { ty, .. }
            | Instruction::Atom { ty, .. }
            | Instruction::AtomCas { ty, .. }
            | Instruction::Mad { ty, .. }
            | Instruction::Fma { ty, .. }
            | Instruction::SetP { ty, .. }
            | Instruction::LoadParam { ty, .. } => ty.as_ptx_str().to_string(),
            Instruction::MadLo { typ, .. } | Instruction::MadHi { typ, .. } => {
                typ.as_ptx_str().to_string()
            }
            Instruction::MadWide { src_typ, .. } => src_typ.as_ptx_str().to_string(),
            Instruction::Cvt { dst_ty, .. } => dst_ty.as_ptx_str().to_string(),
            _ => "?".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// InstructionMixAnalyzer
// ---------------------------------------------------------------------------

/// Analyses the instruction mix (category distribution) of a PTX function.
pub struct InstructionMixAnalyzer;

impl InstructionMixAnalyzer {
    /// Analyses a PTX function and returns instruction mix statistics.
    #[must_use]
    pub fn analyze(&self, func: &PtxFunction) -> InstructionMix {
        let mut counts: HashMap<InstructionCategory, usize> = HashMap::new();
        for inst in &func.body {
            *counts.entry(categorize_instruction(inst)).or_insert(0) += 1;
        }
        InstructionMix {
            total: func.body.len(),
            counts,
        }
    }

    /// Renders a horizontal bar chart of instruction categories.
    #[must_use]
    pub fn render_bar_chart(mix: &InstructionMix, width: usize) -> String {
        if mix.total == 0 {
            return "(no instructions)\n".to_string();
        }

        let mut out = String::new();
        let _ = writeln!(out, "Instruction Mix");
        let _ = writeln!(out, "===============");
        let _ = writeln!(out);

        let label_width = 12_usize;
        let bar_width = width.saturating_sub(label_width + 20).max(10);

        let mut categories: Vec<(InstructionCategory, usize)> =
            mix.counts.iter().map(|(&cat, &cnt)| (cat, cnt)).collect();
        categories.sort_by_key(|&(_, cnt)| std::cmp::Reverse(cnt));

        for (cat, count) in &categories {
            #[allow(clippy::cast_precision_loss)]
            let pct = (*count as f64 / mix.total as f64) * 100.0;
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let filled = ((*count as f64 / mix.total as f64) * bar_width as f64) as usize;
            let bar: String = "#".repeat(filled) + &" ".repeat(bar_width.saturating_sub(filled));
            let _ = writeln!(
                out,
                "{:<lw$} [{bar}] {count:>4} ({pct:>5.1}%)",
                cat.label(),
                lw = label_width,
            );
        }

        let _ = writeln!(out);
        let _ = writeln!(out, "Total: {} instructions", mix.total);
        out
    }
}

// ---------------------------------------------------------------------------
// MemoryAccessPattern
// ---------------------------------------------------------------------------

/// Analyses memory access patterns in a PTX function.
pub struct MemoryAccessPattern;

impl MemoryAccessPattern {
    /// Analyses a PTX function and returns a memory access report.
    #[must_use]
    pub fn analyze(func: &PtxFunction) -> MemoryReport {
        let mut report = MemoryReport {
            global_loads: 0,
            global_stores: 0,
            shared_loads: 0,
            shared_stores: 0,
            local_loads: 0,
            local_stores: 0,
            coalescing_score: 1.0,
        };

        let mut total_mem_ops = 0_usize;
        let mut likely_coalesced = 0_usize;

        for inst in &func.body {
            match inst {
                Instruction::Load { space, .. } => {
                    total_mem_ops += 1;
                    match space {
                        MemorySpace::Global => {
                            report.global_loads += 1;
                            // Heuristic: global loads that use address + offset pattern
                            // are more likely to be coalesced
                            likely_coalesced += 1;
                        }
                        MemorySpace::Shared => report.shared_loads += 1,
                        MemorySpace::Local => report.local_loads += 1,
                        _ => {}
                    }
                }
                Instruction::Store { space, .. } => {
                    total_mem_ops += 1;
                    match space {
                        MemorySpace::Global => {
                            report.global_stores += 1;
                            likely_coalesced += 1;
                        }
                        MemorySpace::Shared => report.shared_stores += 1,
                        MemorySpace::Local => report.local_stores += 1,
                        _ => {}
                    }
                }
                Instruction::CpAsync { .. } | Instruction::TmaLoad { .. } => {
                    total_mem_ops += 1;
                    report.global_loads += 1;
                    report.shared_stores += 1;
                    likely_coalesced += 1;
                }
                _ => {}
            }
        }

        if total_mem_ops > 0 {
            #[allow(clippy::cast_precision_loss)]
            {
                report.coalescing_score = likely_coalesced as f64 / total_mem_ops as f64;
            }
        }

        report
    }
}

// ---------------------------------------------------------------------------
// PtxDiff
// ---------------------------------------------------------------------------

/// Compares two PTX functions and produces a diff report.
pub struct PtxDiff;

impl PtxDiff {
    /// Compares two functions and returns a diff report.
    #[must_use]
    pub fn diff(a: &PtxFunction, b: &PtxFunction) -> DiffReport {
        let a_count = a.body.len();
        let b_count = b.body.len();

        let added = b_count.saturating_sub(a_count);
        let removed = a_count.saturating_sub(b_count);

        // Count blocks
        let a_blocks = split_into_blocks(&a.body);
        let b_blocks = split_into_blocks(&b.body);

        let changed_blocks = count_changed_blocks(&a_blocks, &b_blocks);

        // Register delta
        let a_regs = count_unique_registers(&a.body);
        let b_regs = count_unique_registers(&b.body);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let register_delta = b_regs as i32 - a_regs as i32;

        DiffReport {
            added_instructions: added,
            removed_instructions: removed,
            changed_blocks,
            register_delta,
        }
    }

    /// Renders a diff report as formatted text.
    #[must_use]
    pub fn render_diff(report: &DiffReport) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "PTX Diff Report");
        let _ = writeln!(out, "===============");
        let _ = writeln!(out);
        let _ = writeln!(out, "Added instructions:   +{}", report.added_instructions);
        let _ = writeln!(
            out,
            "Removed instructions: -{}",
            report.removed_instructions
        );
        let _ = writeln!(out, "Changed blocks:       {}", report.changed_blocks);

        let sign = if report.register_delta >= 0 { "+" } else { "" };
        let _ = writeln!(out, "Register delta:       {sign}{}", report.register_delta);
        out
    }
}

// ---------------------------------------------------------------------------
// KernelComplexityScore
// ---------------------------------------------------------------------------

/// Computes complexity metrics for a PTX kernel.
pub struct KernelComplexityScore;

impl KernelComplexityScore {
    /// Analyses a PTX function and returns complexity metrics.
    #[must_use]
    pub fn analyze(func: &PtxFunction) -> ComplexityMetrics {
        let instruction_count = func.body.len();

        let mut branch_count = 0_usize;
        let mut arith_count = 0_usize;
        let mut mem_count = 0_usize;

        for inst in &func.body {
            match categorize_instruction(inst) {
                InstructionCategory::Control => {
                    if matches!(inst, Instruction::Branch { .. }) {
                        branch_count += 1;
                    }
                }
                InstructionCategory::Arithmetic => arith_count += 1,
                InstructionCategory::Memory => mem_count += 1,
                _ => {}
            }
        }

        // Detect loops via back-edges: a branch to a label that appears
        // earlier in the instruction stream.
        let loop_count = count_back_edges(&func.body);

        // Register pressure: count max live registers at any point
        let max_register_pressure = compute_max_register_pressure(&func.body);

        // Estimated occupancy: rough heuristic based on register pressure.
        // SM has 65536 registers; a warp uses 32 * regs. Max warps per SM = 64.
        // occupancy = min(64, 65536 / (32 * max_regs)) / 64 * 100
        #[allow(clippy::cast_precision_loss)]
        let estimated_occupancy_pct = if max_register_pressure > 0 {
            let warps_per_sm = 65536_f64 / (32.0 * max_register_pressure as f64);
            let warps_per_sm = warps_per_sm.min(64.0);
            (warps_per_sm / 64.0) * 100.0
        } else {
            100.0
        };

        #[allow(clippy::cast_precision_loss)]
        let arithmetic_intensity = if mem_count > 0 {
            arith_count as f64 / mem_count as f64
        } else if arith_count > 0 {
            f64::INFINITY
        } else {
            0.0
        };

        ComplexityMetrics {
            instruction_count,
            branch_count,
            loop_count,
            max_register_pressure,
            estimated_occupancy_pct,
            arithmetic_intensity,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Split a flat instruction vector into basic blocks by scanning for Label
/// instructions.
fn split_into_blocks(body: &[Instruction]) -> Vec<BasicBlock> {
    if body.is_empty() {
        return Vec::new();
    }

    let mut blocks: Vec<BasicBlock> = Vec::new();
    let mut current_label: Option<String> = None;
    let mut current_insts: Vec<Instruction> = Vec::new();

    for inst in body {
        if let Instruction::Label(lbl) = inst {
            // Flush current block
            if !current_insts.is_empty() || current_label.is_some() {
                blocks.push(BasicBlock {
                    label: current_label.take(),
                    instructions: std::mem::take(&mut current_insts),
                });
            }
            current_label = Some(lbl.clone());
        } else {
            current_insts.push(inst.clone());
        }
    }

    // Flush remaining
    if !current_insts.is_empty() || current_label.is_some() {
        blocks.push(BasicBlock {
            label: current_label,
            instructions: current_insts,
        });
    }

    blocks
}

/// Count the number of blocks that differ between two block lists.
fn count_changed_blocks(a: &[BasicBlock], b: &[BasicBlock]) -> usize {
    let max_len = a.len().max(b.len());
    let mut changed = 0_usize;

    for i in 0..max_len {
        let a_block = a.get(i);
        let b_block = b.get(i);
        match (a_block, b_block) {
            (Some(ab), Some(bb)) => {
                if ab.label != bb.label || ab.instructions.len() != bb.instructions.len() {
                    changed += 1;
                } else {
                    // Compare instruction emissions
                    let differs = ab
                        .instructions
                        .iter()
                        .zip(bb.instructions.iter())
                        .any(|(ai, bi)| ai.emit() != bi.emit());
                    if differs {
                        changed += 1;
                    }
                }
            }
            _ => changed += 1,
        }
    }
    changed
}

/// Count unique register names referenced in an instruction list.
fn count_unique_registers(body: &[Instruction]) -> usize {
    let mut regs = std::collections::HashSet::new();
    for inst in body {
        for r in registers_read(inst) {
            regs.insert(r);
        }
        for r in registers_written(inst) {
            regs.insert(r);
        }
    }
    regs.len()
}

/// Count back-edges (branches to a label that appears before the branch).
fn count_back_edges(body: &[Instruction]) -> usize {
    // Record the instruction index of each label
    let mut label_positions: HashMap<&str, usize> = HashMap::new();
    for (idx, inst) in body.iter().enumerate() {
        if let Instruction::Label(lbl) = inst {
            label_positions.insert(lbl.as_str(), idx);
        }
    }

    let mut count = 0_usize;
    for (idx, inst) in body.iter().enumerate() {
        if let Instruction::Branch { target, .. } = inst {
            if let Some(&lbl_idx) = label_positions.get(target.as_str()) {
                if lbl_idx <= idx {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Compute maximum number of simultaneously live registers.
fn compute_max_register_pressure(body: &[Instruction]) -> usize {
    if body.is_empty() {
        return 0;
    }

    // Build per-register [first_def, last_use] intervals
    let mut first_def: HashMap<String, usize> = HashMap::new();
    let mut last_use: HashMap<String, usize> = HashMap::new();

    for (idx, inst) in body.iter().enumerate() {
        for r in registers_written(inst) {
            first_def.entry(r.clone()).or_insert(idx);
            last_use.insert(r, idx);
        }
        for r in registers_read(inst) {
            last_use.insert(r, idx);
        }
    }

    // Sweep: for each instruction index, count live registers
    let intervals: Vec<(usize, usize)> = first_def
        .iter()
        .map(|(reg, &def)| {
            let use_end = last_use.get(reg).copied().unwrap_or(def);
            (def, use_end)
        })
        .collect();

    let mut max_live = 0_usize;
    for idx in 0..body.len() {
        let live = intervals
            .iter()
            .filter(|(start, end)| *start <= idx && idx <= *end)
            .count();
        if live > max_live {
            max_live = live;
        }
    }
    max_live
}

/// Truncate an instruction's emitted text to a maximum length.
fn truncate_emit(inst: &Instruction, max_len: usize) -> String {
    let s = inst.emit();
    if s.len() > max_len {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    } else {
        s
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        BasicBlock, CacheQualifier, CmpOp, ImmValue, Instruction, MemorySpace, Operand,
        PtxFunction, PtxModule, PtxType, Register, RoundingMode, SpecialReg, VectorWidth,
    };

    // -- Test helpers -------------------------------------------------------

    fn make_reg(name: &str, ty: PtxType) -> Register {
        Register {
            name: name.to_string(),
            ty,
        }
    }

    fn make_operand_reg(name: &str, ty: PtxType) -> Operand {
        Operand::Register(make_reg(name, ty))
    }

    fn make_simple_function() -> PtxFunction {
        let mut func = PtxFunction::new("test_kernel");
        func.add_param("a_ptr", PtxType::U64);
        func.add_param("n", PtxType::U32);

        // Load param
        func.push(Instruction::LoadParam {
            ty: PtxType::U64,
            dst: make_reg("%rd0", PtxType::U64),
            param_name: "a_ptr".to_string(),
        });

        // MovSpecial
        func.push(Instruction::MovSpecial {
            dst: make_reg("%r0", PtxType::U32),
            special: SpecialReg::TidX,
        });

        // Add
        func.push(Instruction::Add {
            ty: PtxType::U32,
            dst: make_reg("%r1", PtxType::U32),
            a: make_operand_reg("%r0", PtxType::U32),
            b: Operand::Immediate(ImmValue::U32(1)),
        });

        // Load
        func.push(Instruction::Load {
            space: MemorySpace::Global,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            dst: make_reg("%f0", PtxType::F32),
            addr: Operand::Address {
                base: make_reg("%rd0", PtxType::U64),
                offset: None,
            },
        });

        // Fma
        func.push(Instruction::Fma {
            rnd: RoundingMode::Rn,
            ty: PtxType::F32,
            dst: make_reg("%f1", PtxType::F32),
            a: make_operand_reg("%f0", PtxType::F32),
            b: Operand::Immediate(ImmValue::F32(2.0)),
            c: Operand::Immediate(ImmValue::F32(1.0)),
        });

        // Store
        func.push(Instruction::Store {
            space: MemorySpace::Global,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            addr: Operand::Address {
                base: make_reg("%rd0", PtxType::U64),
                offset: None,
            },
            src: make_reg("%f1", PtxType::F32),
        });

        func.push(Instruction::Return);
        func
    }

    fn make_branching_function() -> PtxFunction {
        let mut func = PtxFunction::new("branch_kernel");

        func.push(Instruction::MovSpecial {
            dst: make_reg("%r0", PtxType::U32),
            special: SpecialReg::TidX,
        });

        func.push(Instruction::SetP {
            cmp: CmpOp::Lt,
            ty: PtxType::U32,
            dst: make_reg("%p0", PtxType::Pred),
            a: make_operand_reg("%r0", PtxType::U32),
            b: Operand::Immediate(ImmValue::U32(128)),
        });

        func.push(Instruction::Branch {
            target: "skip".to_string(),
            predicate: Some((make_reg("%p0", PtxType::Pred), true)),
        });

        // Work block
        func.push(Instruction::Add {
            ty: PtxType::U32,
            dst: make_reg("%r1", PtxType::U32),
            a: make_operand_reg("%r0", PtxType::U32),
            b: Operand::Immediate(ImmValue::U32(1)),
        });

        func.push(Instruction::Label("skip".to_string()));

        func.push(Instruction::Return);
        func
    }

    // -- Tests --------------------------------------------------------------

    #[test]
    fn test_render_empty_function() {
        let config = ExplorerConfig::default();
        let explorer = PtxExplorer::new(config);
        let func = PtxFunction::new("empty");
        let output = explorer.render_function(&func);
        assert!(output.contains("empty"));
        assert!(output.contains('{'));
        assert!(output.contains('}'));
    }

    #[test]
    fn test_render_function_with_multiple_blocks() {
        let config = ExplorerConfig::default();
        let explorer = PtxExplorer::new(config);
        let func = make_branching_function();
        let output = explorer.render_function(&func);
        assert!(output.contains("branch_kernel"));
        assert!(output.contains("setp"));
        assert!(output.contains("bra"));
        assert!(output.contains("add"));
    }

    #[test]
    fn test_cfg_rendering_with_branches() {
        let config = ExplorerConfig::default();
        let explorer = PtxExplorer::new(config);
        let func = make_branching_function();
        let output = explorer.render_cfg(&func);
        assert!(output.contains("Control Flow Graph"));
        assert!(output.contains("skip"));
        assert!(output.contains("-->"));
    }

    #[test]
    fn test_register_lifetime_analysis() {
        let analyzer = RegisterLifetimeAnalyzer;
        let func = make_simple_function();
        let lifetimes = analyzer.analyze(&func);

        // We should have several registers
        assert!(!lifetimes.is_empty());

        // %rd0 is defined first (LoadParam at index 0) and used later (Load, Store)
        let rd0 = lifetimes.iter().find(|l| l.register == "%rd0");
        assert!(rd0.is_some(), "should find %rd0 lifetime");
        let rd0 = rd0.expect("checked above");
        assert_eq!(rd0.first_def, 0);
        assert!(
            rd0.last_use > rd0.first_def,
            "last_use should be after first_def"
        );
    }

    #[test]
    fn test_register_lifetime_timeline_rendering() {
        let analyzer = RegisterLifetimeAnalyzer;
        let func = make_simple_function();
        let lifetimes = analyzer.analyze(&func);
        let timeline = RegisterLifetimeAnalyzer::render_timeline(&lifetimes, 80);
        assert!(timeline.contains("Register Lifetimes"));
        assert!(timeline.contains('#')); // should have bar characters
        assert!(timeline.contains("uses:"));
    }

    #[test]
    fn test_instruction_mix_categorization() {
        let analyzer = InstructionMixAnalyzer;
        let func = make_simple_function();
        let mix = analyzer.analyze(&func);

        assert_eq!(mix.total, func.body.len());

        // Should have arithmetic, memory, special, and control categories
        let arith = mix
            .counts
            .get(&InstructionCategory::Arithmetic)
            .copied()
            .unwrap_or(0);
        let mem = mix
            .counts
            .get(&InstructionCategory::Memory)
            .copied()
            .unwrap_or(0);
        let special = mix
            .counts
            .get(&InstructionCategory::Special)
            .copied()
            .unwrap_or(0);
        assert!(arith > 0, "should have arithmetic instructions");
        assert!(mem > 0, "should have memory instructions");
        assert!(special > 0, "should have special instructions");
    }

    #[test]
    fn test_instruction_mix_bar_chart() {
        let analyzer = InstructionMixAnalyzer;
        let func = make_simple_function();
        let mix = analyzer.analyze(&func);
        let chart = InstructionMixAnalyzer::render_bar_chart(&mix, 80);
        assert!(chart.contains("Instruction Mix"));
        assert!(chart.contains('#')); // bar characters
        assert!(chart.contains('%')); // percentages
        assert!(chart.contains("Total:"));
    }

    #[test]
    fn test_memory_access_pattern_analysis() {
        let func = make_simple_function();
        let report = MemoryAccessPattern::analyze(&func);
        assert_eq!(report.global_loads, 1);
        assert_eq!(report.global_stores, 1);
        assert_eq!(report.shared_loads, 0);
        assert_eq!(report.shared_stores, 0);
        assert!(report.coalescing_score > 0.0);
        assert!(report.coalescing_score <= 1.0);
    }

    #[test]
    fn test_ptx_diff_identical_functions() {
        let func = make_simple_function();
        let report = PtxDiff::diff(&func, &func);
        assert_eq!(report.added_instructions, 0);
        assert_eq!(report.removed_instructions, 0);
        assert_eq!(report.changed_blocks, 0);
        assert_eq!(report.register_delta, 0);
    }

    #[test]
    fn test_ptx_diff_different_functions() {
        let a = make_simple_function();
        let mut b = make_simple_function();
        // Add extra instructions to b
        b.push(Instruction::Comment("extra".to_string()));
        b.push(Instruction::Add {
            ty: PtxType::U32,
            dst: make_reg("%r99", PtxType::U32),
            a: Operand::Immediate(ImmValue::U32(0)),
            b: Operand::Immediate(ImmValue::U32(1)),
        });

        let report = PtxDiff::diff(&a, &b);
        assert!(report.added_instructions > 0);
        assert!(report.register_delta > 0);

        let rendered = PtxDiff::render_diff(&report);
        assert!(rendered.contains("PTX Diff Report"));
        assert!(rendered.contains('+'));
    }

    #[test]
    fn test_kernel_complexity_scoring() {
        let func = make_branching_function();
        let metrics = KernelComplexityScore::analyze(&func);
        assert_eq!(metrics.instruction_count, func.body.len());
        assert!(metrics.branch_count > 0, "should detect branches");
        assert!(metrics.estimated_occupancy_pct > 0.0);
        assert!(metrics.estimated_occupancy_pct <= 100.0);
    }

    #[test]
    fn test_color_vs_no_color_output() {
        let func = make_simple_function();

        let no_color = PtxExplorer::new(ExplorerConfig {
            use_color: false,
            ..ExplorerConfig::default()
        });
        let with_color = PtxExplorer::new(ExplorerConfig {
            use_color: true,
            ..ExplorerConfig::default()
        });

        let plain = no_color.render_function(&func);
        let colored = with_color.render_function(&func);

        // Colored output should contain ANSI codes
        assert!(colored.contains("\x1b["));
        // Plain output should not
        assert!(!plain.contains("\x1b["));
        // Both should contain the function name
        assert!(plain.contains("test_kernel"));
        assert!(colored.contains("test_kernel"));
    }

    #[test]
    fn test_config_defaults() {
        let config = ExplorerConfig::default();
        assert!(!config.use_color);
        assert_eq!(config.max_width, 120);
        assert!(!config.show_line_numbers);
        assert!(!config.show_register_types);
        assert!(!config.show_instruction_latency);
    }

    #[test]
    fn test_large_function_handling() {
        let mut func = PtxFunction::new("big_kernel");
        // Add 500 instructions -- should not crash or truncate
        for i in 0..500 {
            func.push(Instruction::Add {
                ty: PtxType::F32,
                dst: make_reg(&format!("%f{i}"), PtxType::F32),
                a: Operand::Immediate(ImmValue::F32(1.0)),
                b: Operand::Immediate(ImmValue::F32(2.0)),
            });
        }

        let config = ExplorerConfig::default();
        let explorer = PtxExplorer::new(config);
        let output = explorer.render_function(&func);
        // Should contain all 500 instructions
        assert!(output.lines().count() > 500);

        let mix = InstructionMixAnalyzer.analyze(&func);
        assert_eq!(mix.total, 500);

        let metrics = KernelComplexityScore::analyze(&func);
        assert_eq!(metrics.instruction_count, 500);
    }

    #[test]
    fn test_line_number_rendering() {
        let config = ExplorerConfig {
            show_line_numbers: true,
            ..ExplorerConfig::default()
        };
        let explorer = PtxExplorer::new(config);
        let func = make_simple_function();
        let output = explorer.render_function(&func);
        // Should contain line numbers starting from 1
        assert!(output.contains("   1  "));
        assert!(output.contains("   2  "));
    }

    #[test]
    fn test_render_module() {
        let mut module = PtxModule::new("sm_80");
        module.add_function(make_simple_function());
        module.add_function(make_branching_function());

        let explorer = PtxExplorer::new(ExplorerConfig::default());
        let output = explorer.render_module(&module);
        assert!(output.contains(".version 8.5"));
        assert!(output.contains(".target sm_80"));
        assert!(output.contains("test_kernel"));
        assert!(output.contains("branch_kernel"));
    }

    #[test]
    fn test_dependency_graph() {
        let mut block = BasicBlock::with_label("test_block");
        block.push(Instruction::LoadParam {
            ty: PtxType::F32,
            dst: make_reg("%f0", PtxType::F32),
            param_name: "x".to_string(),
        });
        block.push(Instruction::Add {
            ty: PtxType::F32,
            dst: make_reg("%f1", PtxType::F32),
            a: make_operand_reg("%f0", PtxType::F32),
            b: Operand::Immediate(ImmValue::F32(1.0)),
        });
        block.push(Instruction::Add {
            ty: PtxType::F32,
            dst: make_reg("%f2", PtxType::F32),
            a: make_operand_reg("%f1", PtxType::F32),
            b: make_operand_reg("%f0", PtxType::F32),
        });

        let explorer = PtxExplorer::new(ExplorerConfig::default());
        let output = explorer.render_dependency_graph(&block);
        assert!(output.contains("Dependency graph"));
        assert!(output.contains("test_block"));
        assert!(output.contains("-->")); // should have dependency edges
        assert!(output.contains("%f0")); // register dependency
    }

    #[test]
    fn test_cfg_empty_function() {
        let config = ExplorerConfig::default();
        let explorer = PtxExplorer::new(config);
        let func = PtxFunction::new("empty_kernel");
        let output = explorer.render_cfg(&func);
        // Empty function body → empty CFG message
        assert!(
            output.contains("empty CFG")
                || output.contains("Control Flow Graph")
                || output.is_empty()
                || output.contains("(entry)")
        );
    }

    #[test]
    fn test_cfg_no_branch_single_block() {
        let config = ExplorerConfig::default();
        let explorer = PtxExplorer::new(config);
        let func = make_simple_function();
        let output = explorer.render_cfg(&func);
        // make_simple_function has no labels/branches → one block
        assert!(output.contains("Control Flow Graph"));
        assert!(output.contains("B0"));
    }

    #[test]
    fn test_register_lifetime_single_instruction() {
        let analyzer = RegisterLifetimeAnalyzer;
        let mut func = PtxFunction::new("single");
        func.push(Instruction::Add {
            ty: PtxType::U32,
            dst: make_reg("%r0", PtxType::U32),
            a: Operand::Immediate(ImmValue::U32(1)),
            b: Operand::Immediate(ImmValue::U32(2)),
        });
        let lifetimes = analyzer.analyze(&func);
        // %r0 is written at index 0, last_use = 0
        let r0 = lifetimes.iter().find(|l| l.register == "%r0");
        assert!(r0.is_some(), "should track %r0");
        let r0 = r0.expect("checked above");
        assert_eq!(r0.first_def, 0);
        assert_eq!(r0.last_use, 0);
    }

    #[test]
    fn test_register_lifetime_render_empty() {
        let rendered = RegisterLifetimeAnalyzer::render_timeline(&[], 80);
        assert!(rendered.contains("no registers"));
    }

    #[test]
    fn test_instruction_mix_empty_function() {
        let analyzer = InstructionMixAnalyzer;
        let func = PtxFunction::new("empty_kernel");
        let mix = analyzer.analyze(&func);
        assert_eq!(mix.total, 0);
        let chart = InstructionMixAnalyzer::render_bar_chart(&mix, 80);
        assert!(chart.contains("no instructions"));
    }

    #[test]
    fn test_dependency_graph_no_deps() {
        let mut block = BasicBlock::with_label("no_deps");
        // Instructions with no register dependencies between them
        block.push(Instruction::Add {
            ty: PtxType::U32,
            dst: make_reg("%r0", PtxType::U32),
            a: Operand::Immediate(ImmValue::U32(1)),
            b: Operand::Immediate(ImmValue::U32(2)),
        });
        block.push(Instruction::Add {
            ty: PtxType::U32,
            dst: make_reg("%r1", PtxType::U32),
            a: Operand::Immediate(ImmValue::U32(3)),
            b: Operand::Immediate(ImmValue::U32(4)),
        });
        let explorer = PtxExplorer::new(ExplorerConfig::default());
        let output = explorer.render_dependency_graph(&block);
        assert!(output.contains("no_deps"));
        // %r0 and %r1 are independent — no edges between them
        assert!(output.contains("no data dependencies"));
    }

    #[test]
    fn test_cfg_renderer_empty_blocks() {
        let renderer = CfgRenderer;
        let output = renderer.render(&[]);
        assert!(output.contains("empty CFG"));
    }
}
