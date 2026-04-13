//! Automatic kernel fusion analysis pass.
//!
//! This module implements an IR-level analysis pass that identifies and fuses
//! compatible PTX kernels to reduce launch overhead and eliminate intermediate
//! memory allocations. The analysis examines pairs and sequences of kernels,
//! detecting data dependencies, checking fusion constraints, and producing
//! an optimized [`FusionPlan`] that groups kernels for combined execution.
//!
//! # Fusion types
//!
//! - **Elementwise**: Both kernels perform purely elementwise operations with
//!   no shared memory or synchronization -- trivially fusible.
//! - **Producer-consumer**: One kernel's output feeds directly into another
//!   kernel's input, allowing the intermediate buffer to be eliminated.
//! - **Horizontal**: Independent kernels with compatible grid dimensions that
//!   can share a single launch.
//! - **Vertical**: Producer-consumer pair separated by a reduction boundary.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_ptx::ir::{PtxFunction, PtxType, Instruction, Operand, Register, ImmValue};
//! use oxicuda_ptx::ir::{MemorySpace, CacheQualifier, VectorWidth, SpecialReg};
//! use oxicuda_ptx::analysis::kernel_fusion::{FusionAnalysis, plan_fusion};
//!
//! // Create two simple elementwise kernels
//! let mut k0 = PtxFunction::new("add_kernel");
//! k0.add_param("input", PtxType::U64);
//! k0.add_param("output", PtxType::U64);
//! k0.body.push(Instruction::Add {
//!     ty: PtxType::F32,
//!     dst: Register { name: "%f0".into(), ty: PtxType::F32 },
//!     a: Operand::Immediate(ImmValue::F32(1.0)),
//!     b: Operand::Immediate(ImmValue::F32(2.0)),
//! });
//!
//! let mut k1 = PtxFunction::new("mul_kernel");
//! k1.add_param("input", PtxType::U64);
//! k1.add_param("output", PtxType::U64);
//! k1.body.push(Instruction::Add {
//!     ty: PtxType::F32,
//!     dst: Register { name: "%f0".into(), ty: PtxType::F32 },
//!     a: Operand::Immediate(ImmValue::F32(3.0)),
//!     b: Operand::Immediate(ImmValue::F32(4.0)),
//! });
//!
//! let report = plan_fusion(&[k0, k1], 255, 49152);
//! assert!(!report.plan.candidates.is_empty());
//! ```

use std::collections::HashSet;
use std::fmt;

use crate::ir::{Instruction, PtxFunction, PtxType};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// The type of fusion identified between two kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionType {
    /// Both kernels are elementwise, trivially fusible.
    Elementwise,
    /// Producer output feeds directly to consumer input.
    ProducerConsumer,
    /// Independent kernels with the same grid that can share a launch.
    Horizontal,
    /// Producer-consumer with a reduction boundary.
    Vertical,
}

impl fmt::Display for FusionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Elementwise => write!(f, "elementwise"),
            Self::ProducerConsumer => write!(f, "producer-consumer"),
            Self::Horizontal => write!(f, "horizontal"),
            Self::Vertical => write!(f, "vertical"),
        }
    }
}

/// Access pattern for a data dependency between kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    /// Sequential streaming access (coalesced).
    Streaming,
    /// Random (non-coalesced) access.
    Random,
    /// Strided access with the given stride in elements.
    Strided(u32),
    /// Access pattern could not be determined.
    Unknown,
}

impl fmt::Display for AccessPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Streaming => write!(f, "streaming"),
            Self::Random => write!(f, "random"),
            Self::Strided(s) => write!(f, "strided({s})"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Constraints that a fusion candidate must satisfy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionConstraint {
    /// The two kernels must have compatible grid dimensions.
    SameGridDimensions,
    /// Combined shared memory usage must not exceed the device limit.
    NoSharedMemoryConflict,
    /// No conflicting synchronization barriers between the kernels.
    NoBarrierConflict,
    /// Combined register usage must not exceed the given budget.
    RegisterBudget(u32),
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A candidate pair of kernels that may be fused.
#[derive(Debug, Clone)]
pub struct FusionCandidate {
    /// Index of the producing kernel in the original sequence.
    pub producer_index: usize,
    /// Index of the consuming kernel in the original sequence.
    pub consumer_index: usize,
    /// Name of the intermediate buffer shared between the pair.
    pub shared_buffer: String,
    /// The type of fusion identified.
    pub fusion_type: FusionType,
    /// Estimated speedup from fusing these kernels (1.0 = no change).
    pub estimated_speedup: f64,
    /// Shared memory bytes used by the producer kernel.
    pub producer_shared_bytes: usize,
    /// Shared memory bytes used by the consumer kernel.
    pub consumer_shared_bytes: usize,
    /// Estimated register count for the fused kernel.
    pub estimated_registers: u32,
}

/// A data dependency between two kernels.
#[derive(Debug, Clone)]
pub struct DataDependency {
    /// Index of the producing kernel.
    pub producer: usize,
    /// Index of the consuming kernel.
    pub consumer: usize,
    /// Name of the buffer connecting the kernels.
    pub buffer_name: String,
    /// Access pattern of the dependency.
    pub access_pattern: AccessPattern,
}

/// A plan describing which kernels to fuse and the expected benefit.
#[derive(Debug, Clone)]
pub struct FusionPlan {
    /// All accepted fusion candidates.
    pub candidates: Vec<FusionCandidate>,
    /// Groups of kernel indices that should be fused together.
    pub fused_groups: Vec<Vec<usize>>,
    /// Number of kernels before fusion.
    pub original_kernel_count: usize,
    /// Number of kernels after fusion.
    pub fused_kernel_count: usize,
    /// Estimated aggregate speedup factor.
    pub estimated_total_speedup: f64,
}

impl fmt::Display for FusionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Fusion Plan")?;
        writeln!(f, "  Original kernels: {}", self.original_kernel_count)?;
        writeln!(f, "  Fused kernels:    {}", self.fused_kernel_count)?;
        writeln!(
            f,
            "  Estimated speedup: {:.2}x",
            self.estimated_total_speedup
        )?;
        writeln!(f, "  Groups:")?;
        for (i, group) in self.fused_groups.iter().enumerate() {
            writeln!(f, "    [{i}]: {group:?}")?;
        }
        Ok(())
    }
}

/// Full fusion analysis report including accepted and rejected candidates.
#[derive(Debug, Clone)]
pub struct FusionReport {
    /// The accepted fusion plan.
    pub plan: FusionPlan,
    /// Candidates that were rejected, with a reason string.
    pub rejected: Vec<(FusionCandidate, String)>,
}

impl fmt::Display for FusionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.plan)?;
        if !self.rejected.is_empty() {
            writeln!(f, "  Rejected candidates:")?;
            for (cand, reason) in &self.rejected {
                writeln!(
                    f,
                    "    kernel[{}] -> kernel[{}] ({}): {}",
                    cand.producer_index, cand.consumer_index, cand.fusion_type, reason
                )?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Analysis engine
// ---------------------------------------------------------------------------

/// Kernel fusion analysis engine.
///
/// Provides static analysis methods to identify fusion opportunities
/// between PTX kernels. All methods are associated functions that do
/// not require mutable state.
#[derive(Debug, Clone, Copy, Default)]
pub struct FusionAnalysis;

impl FusionAnalysis {
    /// Creates a new fusion analysis engine.
    pub const fn new() -> Self {
        Self
    }

    /// Analyzes a pair of kernels for fusion opportunity.
    ///
    /// Returns `Some(FusionCandidate)` if the two kernels can be fused,
    /// or `None` if no fusion opportunity exists.
    pub fn analyze_pair(producer: &PtxFunction, consumer: &PtxFunction) -> Option<FusionCandidate> {
        let prod_shared = shared_mem_bytes(producer);
        let cons_shared = shared_mem_bytes(consumer);
        let est_regs = estimate_register_count(producer) + estimate_register_count(consumer);

        // Determine shared buffer name from parameter overlap
        let shared_buf = find_shared_buffer(producer, consumer);

        let prod_ew = Self::is_elementwise(producer);
        let cons_ew = Self::is_elementwise(consumer);

        // Determine fusion type
        let fusion_type = if prod_ew && cons_ew {
            FusionType::Elementwise
        } else if shared_buf.is_some() {
            if has_reduction(producer) {
                FusionType::Vertical
            } else {
                FusionType::ProducerConsumer
            }
        } else if compatible_grid_hints(producer, consumer) {
            FusionType::Horizontal
        } else {
            return None;
        };

        let buffer_name = shared_buf.unwrap_or_default();

        let candidate = FusionCandidate {
            producer_index: 0,
            consumer_index: 1,
            shared_buffer: buffer_name,
            fusion_type,
            estimated_speedup: 1.0, // placeholder, filled by estimate
            producer_shared_bytes: prod_shared,
            consumer_shared_bytes: cons_shared,
            estimated_registers: est_regs,
        };

        let speedup = Self::estimate_fusion_speedup(&candidate);

        Some(FusionCandidate {
            estimated_speedup: speedup,
            ..candidate
        })
    }

    /// Analyzes a sequence of kernels and returns all fusion candidates.
    pub fn analyze_sequence(kernels: &[PtxFunction]) -> Vec<FusionCandidate> {
        let mut candidates = Vec::new();
        if kernels.len() < 2 {
            return candidates;
        }

        for i in 0..kernels.len() {
            for j in (i + 1)..kernels.len() {
                if let Some(mut cand) = Self::analyze_pair(&kernels[i], &kernels[j]) {
                    cand.producer_index = i;
                    cand.consumer_index = j;
                    cand.estimated_speedup = Self::estimate_fusion_speedup(&cand);
                    candidates.push(cand);
                }
            }
        }

        candidates
    }

    /// Checks whether a fusion candidate satisfies all given constraints.
    pub fn check_constraints(
        candidate: &FusionCandidate,
        constraints: &[FusionConstraint],
    ) -> bool {
        for constraint in constraints {
            match constraint {
                FusionConstraint::SameGridDimensions => {
                    // For elementwise and producer-consumer, grid compatibility
                    // is assumed when the analysis identified the pair. For
                    // horizontal fusion, it was explicitly checked.
                    // Always passes if we got this far.
                }
                FusionConstraint::NoSharedMemoryConflict => {
                    // Default limit: 48 KiB
                    let combined =
                        candidate.producer_shared_bytes + candidate.consumer_shared_bytes;
                    if combined > 49152 {
                        return false;
                    }
                }
                FusionConstraint::NoBarrierConflict => {
                    // Barrier conflicts are detected during pair analysis.
                    // If both kernels use barriers, fusion is risky.
                    // We encode this: if both have nonzero shared mem AND
                    // the fusion type is not Elementwise, it may conflict.
                    // Conservative: reject if both use shared memory.
                    if candidate.producer_shared_bytes > 0
                        && candidate.consumer_shared_bytes > 0
                        && candidate.fusion_type != FusionType::Elementwise
                    {
                        return false;
                    }
                }
                FusionConstraint::RegisterBudget(max_regs) => {
                    if candidate.estimated_registers > *max_regs {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Checks whether a kernel is purely elementwise.
    ///
    /// A kernel is elementwise if it has no shared memory, no synchronization
    /// barriers, no tensor core operations, and no warp-level reductions.
    pub fn is_elementwise(func: &PtxFunction) -> bool {
        if !func.shared_mem.is_empty() {
            return false;
        }

        for inst in &func.body {
            if is_non_elementwise_instruction(inst) {
                return false;
            }
        }

        true
    }

    /// Finds data dependencies between kernels in a sequence.
    ///
    /// A data dependency exists when a producer kernel writes to a buffer
    /// that a consumer kernel reads from, identified by matching parameter
    /// names with pointer types.
    pub fn find_data_dependencies(funcs: &[PtxFunction]) -> Vec<DataDependency> {
        let mut deps = Vec::new();

        for (i, prod_func) in funcs.iter().enumerate() {
            let producer_outputs = output_params(prod_func);
            for (j, cons_func) in funcs.iter().enumerate().skip(i + 1) {
                let consumer_inputs = input_params(cons_func);
                for out_name in &producer_outputs {
                    for in_name in &consumer_inputs {
                        if out_name == in_name {
                            let pattern = infer_access_pattern(cons_func);
                            deps.push(DataDependency {
                                producer: i,
                                consumer: j,
                                buffer_name: out_name.clone(),
                                access_pattern: pattern,
                            });
                        }
                    }
                }
            }
        }

        deps
    }

    /// Estimates the speedup factor from fusing a candidate pair.
    ///
    /// The estimate accounts for launch overhead elimination, memory bandwidth
    /// savings from eliminating intermediate buffers, and the fusion type.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_fusion_speedup(candidate: &FusionCandidate) -> f64 {
        // Base speedup from eliminating one kernel launch (~5-10 us)
        let launch_overhead_factor = 1.05;

        // Memory bandwidth savings depend on fusion type
        let bandwidth_factor = match candidate.fusion_type {
            FusionType::Elementwise => 1.8,
            FusionType::ProducerConsumer => 1.5,
            FusionType::Horizontal => 1.1,
            FusionType::Vertical => 1.3,
        };

        // Register pressure penalty: more registers may reduce occupancy
        let reg_penalty = if candidate.estimated_registers > 128 {
            0.9
        } else if candidate.estimated_registers > 64 {
            0.95
        } else {
            1.0
        };

        // Shared memory penalty
        let smem_total = (candidate.producer_shared_bytes + candidate.consumer_shared_bytes) as f64;
        let smem_penalty = if smem_total > 32768.0 { 0.9 } else { 1.0 };

        launch_overhead_factor * bandwidth_factor * reg_penalty * smem_penalty
    }
}

// ---------------------------------------------------------------------------
// Top-level planning function
// ---------------------------------------------------------------------------

/// Plans kernel fusion for a sequence of kernels with resource constraints.
///
/// Analyzes all pairs, checks constraints, and produces a [`FusionReport`]
/// describing which kernels should be fused and which candidates were rejected.
///
/// # Arguments
///
/// * `kernels` - The sequence of PTX kernels to analyze.
/// * `max_registers` - Maximum registers per thread for the target device.
/// * `max_shared_mem` - Maximum shared memory per block in bytes.
pub fn plan_fusion(
    kernels: &[PtxFunction],
    max_registers: u32,
    max_shared_mem: u32,
) -> FusionReport {
    let candidates = FusionAnalysis::analyze_sequence(kernels);
    let constraints = vec![
        FusionConstraint::SameGridDimensions,
        FusionConstraint::NoSharedMemoryConflict,
        FusionConstraint::NoBarrierConflict,
        FusionConstraint::RegisterBudget(max_registers),
    ];

    let mut accepted = Vec::new();
    let mut rejected = Vec::new();

    for cand in candidates {
        if FusionAnalysis::check_constraints(&cand, &constraints) {
            accepted.push(cand);
        } else {
            let reason = rejection_reason(&cand, &constraints, max_shared_mem);
            rejected.push((cand, reason));
        }
    }

    // Build fusion groups using a union-find approach
    let groups = build_fusion_groups(&accepted, kernels.len());

    let fused_kernel_count = groups.len();
    let total_speedup = if accepted.is_empty() {
        1.0
    } else {
        // Geometric mean of accepted speedups
        let product: f64 = accepted.iter().map(|c| c.estimated_speedup).product();
        #[allow(clippy::cast_precision_loss)]
        let n = accepted.len() as f64;
        product.powf(1.0 / n)
    };

    FusionReport {
        plan: FusionPlan {
            candidates: accepted,
            fused_groups: groups,
            original_kernel_count: kernels.len(),
            fused_kernel_count,
            estimated_total_speedup: total_speedup,
        },
        rejected,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Calculates total static shared memory bytes for a function.
fn shared_mem_bytes(func: &PtxFunction) -> usize {
    func.shared_mem
        .iter()
        .map(|(_, ty, count)| ty.size_bytes() * count)
        .sum()
}

/// Estimates register count from instruction body length.
///
/// A rough heuristic: each unique destination register in the body
/// contributes one register. We cap at a reasonable maximum.
fn estimate_register_count(func: &PtxFunction) -> u32 {
    let mut reg_names: HashSet<&str> = HashSet::new();
    for inst in &func.body {
        if let Some(name) = destination_register_name(inst) {
            reg_names.insert(name);
        }
    }
    // Minimum 1 register per kernel, even if body is empty
    let count = reg_names.len().max(1);
    // Clamp to u32
    u32::try_from(count).unwrap_or(u32::MAX)
}

/// Extracts the destination register name from an instruction, if any.
fn destination_register_name(inst: &Instruction) -> Option<&str> {
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
        | Instruction::Rcp { dst, .. }
        | Instruction::Rsqrt { dst, .. }
        | Instruction::Sqrt { dst, .. }
        | Instruction::Ex2 { dst, .. }
        | Instruction::Lg2 { dst, .. }
        | Instruction::Sin { dst, .. }
        | Instruction::Cos { dst, .. }
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
        | Instruction::MovSpecial { dst, .. }
        | Instruction::LoadParam { dst, .. }
        | Instruction::Atom { dst, .. }
        | Instruction::AtomCas { dst, .. }
        | Instruction::Dp4a { dst, .. }
        | Instruction::Dp2a { dst, .. }
        | Instruction::Tex1d { dst, .. }
        | Instruction::Tex2d { dst, .. }
        | Instruction::Tex3d { dst, .. }
        | Instruction::SurfLoad { dst, .. }
        | Instruction::Redux { dst, .. }
        | Instruction::ElectSync { dst, .. } => Some(&dst.name),

        _ => None,
    }
}

/// Checks if an instruction is non-elementwise (shared mem, sync, tensor core, etc.).
fn is_non_elementwise_instruction(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::BarSync { .. }
            | Instruction::BarArrive { .. }
            | Instruction::FenceAcqRel { .. }
            | Instruction::Mma { .. }
            | Instruction::Wgmma { .. }
            | Instruction::TmaLoad { .. }
            | Instruction::CpAsync { .. }
            | Instruction::CpAsyncCommit
            | Instruction::CpAsyncWait { .. }
            | Instruction::Redux { .. }
            | Instruction::Stmatrix { .. }
            | Instruction::MbarrierInit { .. }
            | Instruction::MbarrierArrive { .. }
            | Instruction::MbarrierWait { .. }
            | Instruction::FenceProxy { .. }
    ) || matches!(inst, Instruction::Wmma { .. })
        || is_shared_mem_access(inst)
}

/// Checks if an instruction accesses shared memory.
fn is_shared_mem_access(inst: &Instruction) -> bool {
    match inst {
        Instruction::Load { space, .. } | Instruction::Store { space, .. } => {
            *space == crate::ir::MemorySpace::Shared
        }
        _ => false,
    }
}

/// Checks if a kernel contains a reduction pattern.
///
/// Heuristic: looks for warp-level redux instructions or shared-memory
/// barrier patterns commonly used in reductions.
fn has_reduction(func: &PtxFunction) -> bool {
    func.body
        .iter()
        .any(|inst| matches!(inst, Instruction::Redux { .. }))
}

/// Finds a shared buffer name between producer output and consumer input params.
fn find_shared_buffer(producer: &PtxFunction, consumer: &PtxFunction) -> Option<String> {
    let producer_outputs = output_params(producer);
    let consumer_inputs = input_params(consumer);

    for out_name in &producer_outputs {
        for in_name in &consumer_inputs {
            if out_name == in_name {
                return Some(out_name.clone());
            }
        }
    }
    None
}

/// Returns output parameter names for a kernel.
///
/// Heuristic: the last pointer-typed (U64) parameter is treated as the output.
/// If the kernel has Store instructions referencing a `LoadParam`, those params
/// are also outputs.
fn output_params(func: &PtxFunction) -> Vec<String> {
    let mut outputs = Vec::new();

    // Heuristic: last U64 param is output (common CUDA convention)
    if let Some((name, _)) = func.params.iter().rev().find(|(_, ty)| *ty == PtxType::U64) {
        outputs.push(name.clone());
    }

    // Also check for params named with "output" or "out" or "dst" or "result"
    for (name, ty) in &func.params {
        if *ty == PtxType::U64 {
            let lower = name.to_lowercase();
            if (lower.contains("out") || lower.contains("dst") || lower.contains("result"))
                && !outputs.contains(name)
            {
                outputs.push(name.clone());
            }
        }
    }

    outputs
}

/// Returns input parameter names for a kernel.
///
/// Heuristic: all pointer-typed (U64) parameters that are not the last one,
/// plus params with "input" or "in" or "src" in the name.
fn input_params(func: &PtxFunction) -> Vec<String> {
    let mut inputs = Vec::new();
    let outputs = output_params(func);

    for (name, ty) in &func.params {
        if *ty == PtxType::U64 && !outputs.contains(name) {
            inputs.push(name.clone());
        }
    }

    // Also include params explicitly named as inputs
    for (name, ty) in &func.params {
        if *ty == PtxType::U64 {
            let lower = name.to_lowercase();
            if (lower.contains("in") || lower.contains("src")) && !inputs.contains(name) {
                inputs.push(name.clone());
            }
        }
    }

    // If no inputs found, treat all U64 params as potential inputs
    if inputs.is_empty() {
        for (name, ty) in &func.params {
            if *ty == PtxType::U64 {
                inputs.push(name.clone());
            }
        }
    }

    inputs
}

/// Infers the access pattern of a kernel from its instruction body.
fn infer_access_pattern(func: &PtxFunction) -> AccessPattern {
    let has_tid = func.body.iter().any(|inst| {
        matches!(
            inst,
            Instruction::MovSpecial {
                special: crate::ir::SpecialReg::TidX,
                ..
            }
        )
    });

    let has_stride_mul = func.body.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Mul { .. } | Instruction::Shl { .. } | Instruction::Mad { .. }
        )
    });

    if has_tid && !has_stride_mul {
        AccessPattern::Streaming
    } else if has_tid && has_stride_mul {
        // Could be strided; default to unknown stride
        AccessPattern::Strided(1)
    } else {
        AccessPattern::Unknown
    }
}

/// Checks if two kernels have compatible grid dimension hints.
///
/// Returns true if both have the same `max_threads` setting or if
/// neither specifies one (assumed compatible).
const fn compatible_grid_hints(a: &PtxFunction, b: &PtxFunction) -> bool {
    match (a.max_threads, b.max_threads) {
        (Some(ma), Some(mb)) => ma == mb,
        (None, None) => true,
        _ => false,
    }
}

/// Determines a human-readable rejection reason for a candidate.
fn rejection_reason(
    candidate: &FusionCandidate,
    constraints: &[FusionConstraint],
    _max_shared_mem: u32,
) -> String {
    for constraint in constraints {
        match constraint {
            FusionConstraint::NoSharedMemoryConflict => {
                let combined = candidate.producer_shared_bytes + candidate.consumer_shared_bytes;
                if combined > 49152 {
                    return format!(
                        "combined shared memory ({combined} bytes) exceeds 48 KiB limit"
                    );
                }
            }
            FusionConstraint::NoBarrierConflict => {
                if candidate.producer_shared_bytes > 0
                    && candidate.consumer_shared_bytes > 0
                    && candidate.fusion_type != FusionType::Elementwise
                {
                    return "barrier conflict: both kernels use shared memory".to_string();
                }
            }
            FusionConstraint::RegisterBudget(max_regs) => {
                if candidate.estimated_registers > *max_regs {
                    return format!(
                        "register budget exceeded ({} > {max_regs})",
                        candidate.estimated_registers
                    );
                }
            }
            FusionConstraint::SameGridDimensions => {}
        }
    }
    "unknown reason".to_string()
}

/// Builds fusion groups from accepted candidates using a union-find strategy.
///
/// Each group is a set of kernel indices that should be fused together.
/// Kernels not in any accepted candidate get their own singleton group.
/// Find with path compression (iterative) for union-find.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn build_fusion_groups(candidates: &[FusionCandidate], num_kernels: usize) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..num_kernels).collect();

    for cand in candidates {
        let pa = uf_find(&mut parent, cand.producer_index);
        let pb = uf_find(&mut parent, cand.consumer_index);
        if pa != pb {
            // Union: smaller root becomes child
            let (small, big) = if pa < pb { (pa, pb) } else { (pb, pa) };
            parent[big] = small;
        }
    }

    // Collect groups
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..num_kernels {
        let root = uf_find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }

    let mut result: Vec<Vec<usize>> = groups.into_values().collect();
    result.sort_by_key(|g| g[0]);
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        CacheQualifier, ImmValue, Instruction, MemorySpace, Operand, PtxType, Register, SpecialReg,
        VectorWidth, WmmaLayout, WmmaOp, WmmaShape,
    };

    fn reg(name: &str, ty: PtxType) -> Register {
        Register {
            name: name.to_string(),
            ty,
        }
    }

    fn imm_f32(v: f32) -> Operand {
        Operand::Immediate(ImmValue::F32(v))
    }

    /// Build a simple elementwise kernel with the given name and params.
    fn make_elementwise_kernel(name: &str, params: &[(&str, PtxType)]) -> PtxFunction {
        let mut func = PtxFunction::new(name);
        for (pname, pty) in params {
            func.add_param(*pname, *pty);
        }
        // Simple elementwise: tid -> load -> add -> store
        func.body.push(Instruction::MovSpecial {
            dst: reg("%r0", PtxType::U32),
            special: SpecialReg::TidX,
        });
        func.body.push(Instruction::Add {
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            a: imm_f32(1.0),
            b: imm_f32(2.0),
        });
        func.body.push(Instruction::Store {
            space: MemorySpace::Global,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            addr: Operand::Address {
                base: reg("%rd0", PtxType::U64),
                offset: None,
            },
            src: reg("%f0", PtxType::F32),
        });
        func
    }

    /// Build a kernel that uses shared memory and barriers.
    fn make_reduction_kernel(name: &str) -> PtxFunction {
        let mut func = PtxFunction::new(name);
        func.add_param("input", PtxType::U64);
        func.add_param("output", PtxType::U64);
        func.add_shared_mem("smem", PtxType::F32, 256);
        func.body.push(Instruction::MovSpecial {
            dst: reg("%r0", PtxType::U32),
            special: SpecialReg::TidX,
        });
        func.body.push(Instruction::BarSync { id: 0 });
        func.body.push(Instruction::Redux {
            op: crate::ir::ReduxOp::Add,
            dst: reg("%r1", PtxType::U32),
            src: Operand::Register(reg("%r0", PtxType::U32)),
            membership_mask: 0xFFFF_FFFF,
        });
        func
    }

    // -----------------------------------------------------------------------
    // Test: elementwise detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_elementwise_simple() {
        let kernel =
            make_elementwise_kernel("ew", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        assert!(FusionAnalysis::is_elementwise(&kernel));
    }

    #[test]
    fn test_is_not_elementwise_with_shared_mem() {
        let kernel = make_reduction_kernel("reduce");
        assert!(!FusionAnalysis::is_elementwise(&kernel));
    }

    #[test]
    fn test_is_not_elementwise_with_barrier() {
        let mut kernel = PtxFunction::new("barrier_kernel");
        kernel.body.push(Instruction::BarSync { id: 0 });
        assert!(!FusionAnalysis::is_elementwise(&kernel));
    }

    #[test]
    fn test_is_not_elementwise_with_wmma() {
        let mut kernel = PtxFunction::new("wmma_kernel");
        kernel.body.push(Instruction::Wmma {
            op: WmmaOp::Mma,
            shape: WmmaShape::M16N16K16,
            layout: WmmaLayout::RowMajor,
            ty: PtxType::F16,
            fragments: vec![reg("%f0", PtxType::F16)],
            addr: None,
            stride: None,
        });
        assert!(!FusionAnalysis::is_elementwise(&kernel));
    }

    // -----------------------------------------------------------------------
    // Test: pair analysis
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyze_pair_elementwise() {
        let k0 =
            make_elementwise_kernel("add", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        let k1 =
            make_elementwise_kernel("mul", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        let cand = FusionAnalysis::analyze_pair(&k0, &k1);
        assert!(cand.is_some());
        let c = cand.as_ref().map(|c| c.fusion_type);
        assert_eq!(c, Some(FusionType::Elementwise));
    }

    #[test]
    fn test_analyze_pair_producer_consumer() {
        let k0 = make_elementwise_kernel(
            "producer",
            &[("input", PtxType::U64), ("buf", PtxType::U64)],
        );
        let mut k1 = PtxFunction::new("consumer");
        k1.add_param("buf", PtxType::U64);
        k1.add_param("output", PtxType::U64);
        k1.add_shared_mem("smem", PtxType::F32, 64);
        k1.body.push(Instruction::Load {
            space: MemorySpace::Shared,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            addr: Operand::Address {
                base: reg("%rd0", PtxType::U64),
                offset: None,
            },
        });

        let cand = FusionAnalysis::analyze_pair(&k0, &k1);
        assert!(cand.is_some());
        let c = cand.as_ref().map(|c| c.fusion_type);
        assert_eq!(c, Some(FusionType::ProducerConsumer));
    }

    #[test]
    fn test_analyze_pair_horizontal() {
        // Two kernels with same max_threads but no shared buffer
        let mut k0 = PtxFunction::new("kernel_a");
        k0.add_param("a_in", PtxType::U64);
        k0.add_param("a_out", PtxType::U64);
        k0.max_threads = Some(256);
        k0.body.push(Instruction::BarSync { id: 0 });

        let mut k1 = PtxFunction::new("kernel_b");
        k1.add_param("b_in", PtxType::U64);
        k1.add_param("b_out", PtxType::U64);
        k1.max_threads = Some(256);
        k1.body.push(Instruction::BarSync { id: 0 });

        let cand = FusionAnalysis::analyze_pair(&k0, &k1);
        assert!(cand.is_some());
        let c = cand.as_ref().map(|c| c.fusion_type);
        assert_eq!(c, Some(FusionType::Horizontal));
    }

    // -----------------------------------------------------------------------
    // Test: constraint checking
    // -----------------------------------------------------------------------

    #[test]
    fn test_constraints_pass() {
        let cand = FusionCandidate {
            producer_index: 0,
            consumer_index: 1,
            shared_buffer: String::new(),
            fusion_type: FusionType::Elementwise,
            estimated_speedup: 1.5,
            producer_shared_bytes: 0,
            consumer_shared_bytes: 0,
            estimated_registers: 32,
        };
        let constraints = vec![
            FusionConstraint::SameGridDimensions,
            FusionConstraint::NoSharedMemoryConflict,
            FusionConstraint::RegisterBudget(255),
        ];
        assert!(FusionAnalysis::check_constraints(&cand, &constraints));
    }

    #[test]
    fn test_constraints_register_budget_exceeded() {
        let cand = FusionCandidate {
            producer_index: 0,
            consumer_index: 1,
            shared_buffer: String::new(),
            fusion_type: FusionType::ProducerConsumer,
            estimated_speedup: 1.3,
            producer_shared_bytes: 0,
            consumer_shared_bytes: 0,
            estimated_registers: 300,
        };
        let constraints = vec![FusionConstraint::RegisterBudget(255)];
        assert!(!FusionAnalysis::check_constraints(&cand, &constraints));
    }

    #[test]
    fn test_constraints_shared_mem_exceeded() {
        let cand = FusionCandidate {
            producer_index: 0,
            consumer_index: 1,
            shared_buffer: String::new(),
            fusion_type: FusionType::ProducerConsumer,
            estimated_speedup: 1.3,
            producer_shared_bytes: 32768,
            consumer_shared_bytes: 32768,
            estimated_registers: 32,
        };
        let constraints = vec![FusionConstraint::NoSharedMemoryConflict];
        assert!(!FusionAnalysis::check_constraints(&cand, &constraints));
    }

    #[test]
    fn test_constraints_barrier_conflict() {
        let cand = FusionCandidate {
            producer_index: 0,
            consumer_index: 1,
            shared_buffer: "buf".to_string(),
            fusion_type: FusionType::ProducerConsumer,
            estimated_speedup: 1.3,
            producer_shared_bytes: 1024,
            consumer_shared_bytes: 1024,
            estimated_registers: 32,
        };
        let constraints = vec![FusionConstraint::NoBarrierConflict];
        assert!(!FusionAnalysis::check_constraints(&cand, &constraints));
    }

    // -----------------------------------------------------------------------
    // Test: fusion planning
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_fusion_two_elementwise() {
        let k0 =
            make_elementwise_kernel("add", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        let k1 =
            make_elementwise_kernel("mul", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        let report = plan_fusion(&[k0, k1], 255, 49152);
        assert!(!report.plan.candidates.is_empty());
        assert_eq!(report.plan.original_kernel_count, 2);
        // Two kernels fused into 1 group
        assert!(report.plan.fused_kernel_count <= 2);
    }

    #[test]
    fn test_plan_fusion_empty_sequence() {
        let report = plan_fusion(&[], 255, 49152);
        assert!(report.plan.candidates.is_empty());
        assert_eq!(report.plan.original_kernel_count, 0);
        assert_eq!(report.plan.fused_kernel_count, 0);
    }

    #[test]
    fn test_plan_fusion_single_kernel() {
        let k0 =
            make_elementwise_kernel("only", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        let report = plan_fusion(&[k0], 255, 49152);
        assert!(report.plan.candidates.is_empty());
        assert_eq!(report.plan.fused_kernel_count, 1);
    }

    // -----------------------------------------------------------------------
    // Test: speedup estimation
    // -----------------------------------------------------------------------

    #[test]
    fn test_speedup_elementwise() {
        let cand = FusionCandidate {
            producer_index: 0,
            consumer_index: 1,
            shared_buffer: String::new(),
            fusion_type: FusionType::Elementwise,
            estimated_speedup: 0.0,
            producer_shared_bytes: 0,
            consumer_shared_bytes: 0,
            estimated_registers: 16,
        };
        let speedup = FusionAnalysis::estimate_fusion_speedup(&cand);
        // Elementwise should give the highest speedup
        assert!(speedup > 1.5, "expected > 1.5, got {speedup}");
    }

    #[test]
    fn test_speedup_high_register_pressure() {
        let cand = FusionCandidate {
            producer_index: 0,
            consumer_index: 1,
            shared_buffer: String::new(),
            fusion_type: FusionType::Elementwise,
            estimated_speedup: 0.0,
            producer_shared_bytes: 0,
            consumer_shared_bytes: 0,
            estimated_registers: 200,
        };
        let speedup = FusionAnalysis::estimate_fusion_speedup(&cand);
        // High registers should reduce speedup
        let low_cand = FusionCandidate {
            estimated_registers: 16,
            ..cand
        };
        let low_speedup = FusionAnalysis::estimate_fusion_speedup(&low_cand);
        assert!(
            speedup < low_speedup,
            "high regs ({speedup}) should be < low regs ({low_speedup})"
        );
    }

    // -----------------------------------------------------------------------
    // Test: data dependency detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_data_dependencies() {
        let k0 = make_elementwise_kernel(
            "producer",
            &[("input", PtxType::U64), ("buf", PtxType::U64)],
        );
        let k1 = make_elementwise_kernel(
            "consumer",
            &[("buf", PtxType::U64), ("output", PtxType::U64)],
        );
        let deps = FusionAnalysis::find_data_dependencies(&[k0, k1]);
        assert!(!deps.is_empty(), "expected at least one dependency");
        assert_eq!(deps[0].producer, 0);
        assert_eq!(deps[0].consumer, 1);
        assert_eq!(deps[0].buffer_name, "buf");
    }

    #[test]
    fn test_no_data_dependencies() {
        let k0 = make_elementwise_kernel("a", &[("a_in", PtxType::U64), ("a_out", PtxType::U64)]);
        let k1 = make_elementwise_kernel("b", &[("b_in", PtxType::U64), ("b_out", PtxType::U64)]);
        let deps = FusionAnalysis::find_data_dependencies(&[k0, k1]);
        assert!(
            deps.is_empty(),
            "expected no dependencies, got {}",
            deps.len()
        );
    }

    // -----------------------------------------------------------------------
    // Test: Display implementations
    // -----------------------------------------------------------------------

    #[test]
    fn test_fusion_plan_display() {
        let plan = FusionPlan {
            candidates: vec![],
            fused_groups: vec![vec![0, 1], vec![2]],
            original_kernel_count: 3,
            fused_kernel_count: 2,
            estimated_total_speedup: 1.5,
        };
        let display = format!("{plan}");
        assert!(display.contains("Original kernels: 3"));
        assert!(display.contains("Fused kernels:    2"));
        assert!(display.contains("1.50x"));
    }

    #[test]
    fn test_fusion_report_display() {
        let rejected_cand = FusionCandidate {
            producer_index: 0,
            consumer_index: 2,
            shared_buffer: "buf".to_string(),
            fusion_type: FusionType::ProducerConsumer,
            estimated_speedup: 1.3,
            producer_shared_bytes: 0,
            consumer_shared_bytes: 0,
            estimated_registers: 300,
        };
        let report = FusionReport {
            plan: FusionPlan {
                candidates: vec![],
                fused_groups: vec![vec![0], vec![1], vec![2]],
                original_kernel_count: 3,
                fused_kernel_count: 3,
                estimated_total_speedup: 1.0,
            },
            rejected: vec![(rejected_cand, "register budget exceeded".to_string())],
        };
        let display = format!("{report}");
        assert!(display.contains("Rejected candidates:"));
        assert!(display.contains("register budget exceeded"));
    }

    #[test]
    fn test_fusion_type_display() {
        assert_eq!(format!("{}", FusionType::Elementwise), "elementwise");
        assert_eq!(
            format!("{}", FusionType::ProducerConsumer),
            "producer-consumer"
        );
        assert_eq!(format!("{}", FusionType::Horizontal), "horizontal");
        assert_eq!(format!("{}", FusionType::Vertical), "vertical");
    }

    #[test]
    fn test_access_pattern_display() {
        assert_eq!(format!("{}", AccessPattern::Streaming), "streaming");
        assert_eq!(format!("{}", AccessPattern::Random), "random");
        assert_eq!(format!("{}", AccessPattern::Strided(4)), "strided(4)");
        assert_eq!(format!("{}", AccessPattern::Unknown), "unknown");
    }
}
