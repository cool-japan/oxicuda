//! Profile-guided code generation for PTX kernels.
//!
//! This module uses profiling data (from autotuning or `nsight` runs) to make
//! informed decisions about PTX instruction selection, loop unrolling, memory
//! access strategies, and tile sizing.  The [`ProfileGuidedOptimizer`] ingests
//! a [`ProfileData`] snapshot and emits a set of [`CodeGenDecision`]s that
//! downstream kernel builders can apply.

use std::fmt;

use crate::arch::SmVersion;

// ---------------------------------------------------------------------------
// Profile data types
// ---------------------------------------------------------------------------

/// Collected profiling information for a single kernel invocation.
///
/// This is the primary input to the profile-guided optimizer. Typically
/// constructed from autotune results or external profiler output.
#[derive(Debug, Clone)]
pub struct ProfileData {
    /// Name of the profiled kernel.
    pub kernel_name: String,
    /// Target SM architecture the profile was gathered on.
    pub sm_version: SmVersion,
    /// Aggregate performance metrics.
    pub metrics: ProfileMetrics,
    /// Hot instruction indices with stall information.
    pub hotspots: Vec<HotSpot>,
    /// Per-branch taken/not-taken statistics.
    pub branch_stats: Vec<BranchProfile>,
    /// Memory access coalescing and caching statistics.
    pub memory_access_pattern: MemoryAccessProfile,
}

/// Aggregate GPU performance metrics (all ratios are 0.0–1.0 unless noted).
#[derive(Debug, Clone, Copy)]
pub struct ProfileMetrics {
    /// Fraction of the theoretical maximum occupancy achieved.
    pub achieved_occupancy: f64,
    /// Fraction of peak compute throughput utilised.
    pub compute_throughput: f64,
    /// Fraction of peak memory bandwidth utilised.
    pub memory_throughput: f64,
    /// L2 cache hit rate.
    pub l2_hit_rate: f64,
    /// Shared memory transaction efficiency.
    pub shared_memory_efficiency: f64,
    /// Fraction of warps with all lanes active.
    pub warp_execution_efficiency: f64,
    /// Instructions retired per clock cycle.
    pub ipc: f64,
}

/// A single hot instruction with cycle count and stall classification.
#[derive(Debug, Clone)]
pub struct HotSpot {
    /// Index into the instruction stream.
    pub instruction_index: usize,
    /// Total cycles spent at this instruction.
    pub cycle_count: u64,
    /// Dominant stall category at this instruction.
    pub stall_reason: StallReason,
}

/// Reason a warp stalled at a particular instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StallReason {
    /// No significant stall.
    None,
    /// Waiting for a memory operation to complete.
    MemoryDependency,
    /// Waiting for a prior arithmetic result.
    ExecutionDependency,
    /// Blocked on a synchronisation barrier.
    SyncBarrier,
    /// Instruction cache miss.
    InstructionFetch,
    /// Any other stall category.
    Other(String),
}

impl fmt::Display for StallReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::MemoryDependency => f.write_str("memory_dependency"),
            Self::ExecutionDependency => f.write_str("execution_dependency"),
            Self::SyncBarrier => f.write_str("sync_barrier"),
            Self::InstructionFetch => f.write_str("instruction_fetch"),
            Self::Other(s) => write!(f, "other({s})"),
        }
    }
}

/// Taken/not-taken statistics for a single branch site.
#[derive(Debug, Clone, Copy)]
pub struct BranchProfile {
    /// Index of the branch instruction.
    pub branch_index: usize,
    /// Number of times the branch was taken.
    pub taken_count: u64,
    /// Number of times the branch was *not* taken.
    pub not_taken_count: u64,
}

impl BranchProfile {
    /// Returns the fraction of executions where the branch was taken.
    ///
    /// Returns 0.0 if neither path was ever executed.
    #[must_use]
    pub fn taken_ratio(&self) -> f64 {
        let total = self.taken_count + self.not_taken_count;
        if total == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let ratio = self.taken_count as f64 / total as f64;
        ratio
    }

    /// Returns `true` if the branch is biased beyond `threshold` in
    /// either direction (taken ratio > threshold or < 1 − threshold).
    #[must_use]
    pub fn is_biased(&self, threshold: f64) -> bool {
        let ratio = self.taken_ratio();
        ratio > threshold || ratio < (1.0 - threshold)
    }
}

impl fmt::Display for BranchProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "branch[{}]: taken={} not_taken={} ratio={:.2}%",
            self.branch_index,
            self.taken_count,
            self.not_taken_count,
            self.taken_ratio() * 100.0,
        )
    }
}

/// Memory access pattern statistics.
#[derive(Debug, Clone, Copy)]
pub struct MemoryAccessProfile {
    /// Fraction of global loads that are coalesced (0.0–1.0).
    pub coalesced_ratio: f64,
    /// Fraction of shared memory accesses with bank conflicts (0.0–1.0).
    pub bank_conflict_rate: f64,
    /// Average fraction of each cache line actually consumed (0.0–1.0).
    pub cache_line_utilization: f64,
}

impl fmt::Display for MemoryAccessProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "coalesced={:.1}% bank_conflicts={:.1}% cache_util={:.1}%",
            self.coalesced_ratio * 100.0,
            self.bank_conflict_rate * 100.0,
            self.cache_line_utilization * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Bottleneck classification
// ---------------------------------------------------------------------------

/// High-level classification of a kernel's performance bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    /// The kernel is limited by arithmetic throughput.
    ComputeBound,
    /// The kernel is limited by memory bandwidth.
    MemoryBound,
    /// The kernel is limited by instruction or data latency (pipeline bubbles).
    LatencyBound,
    /// No single bottleneck dominates — the kernel is reasonably balanced.
    Balanced,
}

impl fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ComputeBound => f.write_str("compute-bound"),
            Self::MemoryBound => f.write_str("memory-bound"),
            Self::LatencyBound => f.write_str("latency-bound"),
            Self::Balanced => f.write_str("balanced"),
        }
    }
}

// ---------------------------------------------------------------------------
// Code generation decisions
// ---------------------------------------------------------------------------

/// A concrete optimisation decision derived from profiling data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodeGenDecision {
    /// Unroll a hot loop by the given factor.
    UnrollLoop {
        /// Number of iterations to unroll.
        factor: u32,
    },
    /// Convert a heavily biased branch to a predicated instruction.
    PredicateBranch,
    /// Insert prefetch instructions at the given distance (in iterations).
    PrefetchMemory {
        /// Prefetch lookahead distance.
        distance: u32,
    },
    /// Increase occupancy by targeting the given number of blocks per SM.
    IncreaseOccupancy {
        /// Desired concurrent blocks per SM.
        target_blocks: u32,
    },
    /// Use larger tile dimensions for a compute-bound GEMM.
    UseLargerTiles {
        /// Tile size in the M dimension.
        tile_m: u32,
        /// Tile size in the N dimension.
        tile_n: u32,
    },
    /// Promote global memory loads to shared memory.
    SwitchToSharedMemory,
    /// Enable split-K parallelism.
    EnableSplitK {
        /// Number of K-dimension slices.
        k_slices: u32,
    },
}

impl fmt::Display for CodeGenDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnrollLoop { factor } => write!(f, "unroll loop x{factor}"),
            Self::PredicateBranch => f.write_str("convert branch to predicated"),
            Self::PrefetchMemory { distance } => {
                write!(f, "insert prefetch (distance={distance})")
            }
            Self::IncreaseOccupancy { target_blocks } => {
                write!(f, "increase occupancy to {target_blocks} blocks/SM")
            }
            Self::UseLargerTiles { tile_m, tile_n } => {
                write!(f, "use larger tiles ({tile_m}x{tile_n})")
            }
            Self::SwitchToSharedMemory => f.write_str("switch to shared memory"),
            Self::EnableSplitK { k_slices } => {
                write!(f, "enable split-K ({k_slices} slices)")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tile configuration
// ---------------------------------------------------------------------------

/// Suggested tile configuration produced by the optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    /// Tile size in the M dimension.
    pub tile_m: u32,
    /// Tile size in the N dimension.
    pub tile_n: u32,
    /// Tile size in the K dimension.
    pub tile_k: u32,
}

impl fmt::Display for TileConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}x{}", self.tile_m, self.tile_n, self.tile_k)
    }
}

// ---------------------------------------------------------------------------
// KernelProfile — mutable configuration that decisions are applied to
// ---------------------------------------------------------------------------

/// Mutable kernel configuration that the profile-guided optimizer adjusts.
///
/// Downstream builders read these fields after optimisation to generate the
/// final PTX code.
#[derive(Debug, Clone)]
pub struct KernelProfile {
    /// Tile size in the M dimension.
    pub tile_m: u32,
    /// Tile size in the N dimension.
    pub tile_n: u32,
    /// Tile size in the K dimension.
    pub tile_k: u32,
    /// Loop unroll factor.
    pub unroll_factor: u32,
    /// Whether shared memory staging is enabled.
    pub use_shared_memory: bool,
    /// Target register count per thread (0 = no constraint).
    pub register_target: u32,
    /// Number of split-K slices (1 = disabled).
    pub split_k: u32,
}

impl KernelProfile {
    /// Creates a new `KernelProfile` with sensible defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            tile_m: 64,
            tile_n: 64,
            tile_k: 8,
            unroll_factor: 1,
            use_shared_memory: false,
            register_target: 0,
            split_k: 1,
        }
    }
}

impl Default for KernelProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for KernelProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tile={}x{}x{} unroll={} smem={} regs={} split_k={}",
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.unroll_factor,
            if self.use_shared_memory { "on" } else { "off" },
            self.register_target,
            self.split_k,
        )
    }
}

// ---------------------------------------------------------------------------
// Thresholds (internal constants)
// ---------------------------------------------------------------------------

/// Ratio above which a kernel is considered compute-bound.
const COMPUTE_BOUND_THRESHOLD: f64 = 0.7;
/// Ratio above which a kernel is considered memory-bound.
const MEMORY_BOUND_THRESHOLD: f64 = 0.7;
/// IPC below which a kernel is considered latency-bound.
const LATENCY_BOUND_IPC_THRESHOLD: f64 = 1.0;
/// Default branch bias threshold (90 %).
const DEFAULT_BRANCH_BIAS_THRESHOLD: f64 = 0.9;
/// Occupancy below which we recommend increasing it.
const LOW_OCCUPANCY_THRESHOLD: f64 = 0.5;
/// Coalescing ratio below which we recommend shared memory staging.
const POOR_COALESCING_THRESHOLD: f64 = 0.5;
/// Memory throughput above which prefetch is beneficial.
const PREFETCH_MEMORY_THROUGHPUT_THRESHOLD: f64 = 0.5;

// ---------------------------------------------------------------------------
// ProfileGuidedOptimizer
// ---------------------------------------------------------------------------

/// Analyses profiling data and produces [`CodeGenDecision`]s.
#[derive(Debug, Clone)]
pub struct ProfileGuidedOptimizer {
    profile: ProfileData,
}

impl ProfileGuidedOptimizer {
    /// Create a new optimizer from the given profile data.
    #[must_use]
    pub const fn new(profile: ProfileData) -> Self {
        Self { profile }
    }

    /// Classify the kernel's dominant bottleneck.
    #[must_use]
    pub fn classify_bottleneck(&self) -> Bottleneck {
        let m = &self.profile.metrics;

        let compute_heavy = m.compute_throughput >= COMPUTE_BOUND_THRESHOLD;
        let memory_heavy = m.memory_throughput >= MEMORY_BOUND_THRESHOLD;

        match (compute_heavy, memory_heavy) {
            (true, false) => Bottleneck::ComputeBound,
            (false, true) => Bottleneck::MemoryBound,
            (true, true) => Bottleneck::Balanced,
            (false, false) => {
                // Neither unit is saturated — check IPC for latency bound.
                if m.ipc < LATENCY_BOUND_IPC_THRESHOLD
                    && m.achieved_occupancy < LOW_OCCUPANCY_THRESHOLD
                {
                    Bottleneck::LatencyBound
                } else {
                    Bottleneck::Balanced
                }
            }
        }
    }

    /// Produce a list of optimisation decisions based on the profile.
    ///
    /// The returned decisions are ordered from most impactful to least.
    #[must_use]
    pub fn analyze(&self) -> Vec<CodeGenDecision> {
        let mut decisions = Vec::new();
        let bottleneck = self.classify_bottleneck();

        // --- Unroll hot loops ---
        let unroll = self.suggest_unroll_factor();
        if unroll > 1 {
            decisions.push(CodeGenDecision::UnrollLoop { factor: unroll });
        }

        // --- Branch predication ---
        for bp in &self.profile.branch_stats {
            if bp.is_biased(DEFAULT_BRANCH_BIAS_THRESHOLD) {
                decisions.push(CodeGenDecision::PredicateBranch);
                break; // one decision covers all biased branches
            }
        }

        // --- Memory-bound specific ---
        if bottleneck == Bottleneck::MemoryBound || bottleneck == Bottleneck::Balanced {
            let mem = &self.profile.memory_access_pattern;
            if mem.coalesced_ratio < POOR_COALESCING_THRESHOLD {
                decisions.push(CodeGenDecision::SwitchToSharedMemory);
            }
            if self.profile.metrics.memory_throughput > PREFETCH_MEMORY_THROUGHPUT_THRESHOLD {
                let distance = self.suggest_prefetch_distance();
                decisions.push(CodeGenDecision::PrefetchMemory { distance });
            }
        }

        // --- Occupancy ---
        if self.profile.metrics.achieved_occupancy < LOW_OCCUPANCY_THRESHOLD {
            let target = self.suggest_target_blocks();
            decisions.push(CodeGenDecision::IncreaseOccupancy {
                target_blocks: target,
            });
        }

        // --- Compute-bound tile sizing ---
        if bottleneck == Bottleneck::ComputeBound {
            decisions.push(CodeGenDecision::UseLargerTiles {
                tile_m: 128,
                tile_n: 128,
            });
        }

        // --- Split-K for tall-skinny K ---
        if bottleneck == Bottleneck::LatencyBound {
            decisions.push(CodeGenDecision::EnableSplitK { k_slices: 4 });
        }

        decisions
    }

    /// Suggest a tile configuration for a GEMM of the given dimensions.
    #[must_use]
    pub fn suggest_tile_config(&self, m: u32, n: u32, k: u32) -> TileConfig {
        let bottleneck = self.classify_bottleneck();
        let caps = self.profile.sm_version.capabilities();

        // Base tile sizes depend on bottleneck classification.
        let (base_m, base_n) = match bottleneck {
            Bottleneck::ComputeBound => {
                if caps.has_wgmma {
                    (256, 128) // Hopper+ can sustain larger tiles
                } else if caps.has_ampere_mma {
                    (128, 128)
                } else {
                    (128, 64)
                }
            }
            Bottleneck::MemoryBound => (64, 64),
            Bottleneck::LatencyBound => (64, 32),
            Bottleneck::Balanced => (128, 64),
        };

        // Clamp to problem dimensions.
        let tile_m = base_m.min(m);
        let tile_n = base_n.min(n);

        // K tile: for memory-bound kernels use deeper K tiles for reuse.
        let tile_k = match bottleneck {
            Bottleneck::MemoryBound => 32.min(k),
            Bottleneck::ComputeBound => 16.min(k),
            _ => 8.min(k),
        };

        TileConfig {
            tile_m,
            tile_n,
            tile_k,
        }
    }

    /// Suggest an unroll factor based on hotspot and IPC data.
    #[must_use]
    pub fn suggest_unroll_factor(&self) -> u32 {
        let m = &self.profile.metrics;

        // Count memory-dependency stalls — unrolling helps hide latency.
        let mem_stalls = self
            .profile
            .hotspots
            .iter()
            .filter(|h| h.stall_reason == StallReason::MemoryDependency)
            .count();

        if mem_stalls >= 3 {
            return 8;
        }

        if m.ipc < 1.0 {
            return 4;
        }

        if m.ipc < 2.0 {
            return 2;
        }

        1
    }

    // --- private helpers ---

    /// Suggest prefetch distance based on memory throughput and L2 hit rate.
    fn suggest_prefetch_distance(&self) -> u32 {
        let m = &self.profile.metrics;
        if m.l2_hit_rate < 0.3 {
            4 // deep prefetch for poor caching
        } else if m.l2_hit_rate < 0.6 {
            2
        } else {
            1
        }
    }

    /// Suggest target concurrent blocks per SM.
    fn suggest_target_blocks(&self) -> u32 {
        let max_threads = self.profile.sm_version.max_threads_per_sm();
        // Aim for 75 % of max threads at 128 threads/block.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let target_threads = (f64::from(max_threads) * 0.75) as u32;
        let blocks = target_threads / 128;
        blocks.max(2)
    }
}

// ---------------------------------------------------------------------------
// apply_profile_decisions
// ---------------------------------------------------------------------------

/// Apply a set of [`CodeGenDecision`]s to a mutable [`KernelProfile`].
///
/// Returns a human-readable log of every change that was made.
pub fn apply_profile_decisions(
    decisions: &[CodeGenDecision],
    config: &mut KernelProfile,
) -> Vec<String> {
    let mut log = Vec::with_capacity(decisions.len());

    for decision in decisions {
        match decision {
            CodeGenDecision::UnrollLoop { factor } => {
                let prev = config.unroll_factor;
                config.unroll_factor = *factor;
                log.push(format!("unroll factor: {prev} -> {factor}"));
            }
            CodeGenDecision::PredicateBranch => {
                log.push("enabled branch predication".to_string());
            }
            CodeGenDecision::PrefetchMemory { distance } => {
                log.push(format!("enabled prefetch with distance {distance}"));
            }
            CodeGenDecision::IncreaseOccupancy { target_blocks } => {
                // Reduce register pressure to fit more blocks.
                let new_target = 255 / target_blocks;
                let prev = config.register_target;
                config.register_target = new_target;
                log.push(format!(
                    "register target: {prev} -> {new_target} (for {target_blocks} blocks/SM)"
                ));
            }
            CodeGenDecision::UseLargerTiles { tile_m, tile_n } => {
                let prev_m = config.tile_m;
                let prev_n = config.tile_n;
                config.tile_m = *tile_m;
                config.tile_n = *tile_n;
                log.push(format!("tile size: {prev_m}x{prev_n} -> {tile_m}x{tile_n}"));
            }
            CodeGenDecision::SwitchToSharedMemory => {
                config.use_shared_memory = true;
                log.push("enabled shared memory staging".to_string());
            }
            CodeGenDecision::EnableSplitK { k_slices } => {
                let prev = config.split_k;
                config.split_k = *k_slices;
                log.push(format!("split-K: {prev} -> {k_slices} slices"));
            }
        }
    }

    log
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a `ProfileData` with the given metrics.
    fn make_profile(metrics: ProfileMetrics) -> ProfileData {
        ProfileData {
            kernel_name: "test_kernel".to_string(),
            sm_version: SmVersion::Sm80,
            metrics,
            hotspots: Vec::new(),
            branch_stats: Vec::new(),
            memory_access_pattern: MemoryAccessProfile {
                coalesced_ratio: 0.9,
                bank_conflict_rate: 0.05,
                cache_line_utilization: 0.85,
            },
        }
    }

    fn balanced_metrics() -> ProfileMetrics {
        ProfileMetrics {
            achieved_occupancy: 0.75,
            compute_throughput: 0.5,
            memory_throughput: 0.5,
            l2_hit_rate: 0.6,
            shared_memory_efficiency: 0.9,
            warp_execution_efficiency: 0.95,
            ipc: 2.5,
        }
    }

    fn compute_bound_metrics() -> ProfileMetrics {
        ProfileMetrics {
            achieved_occupancy: 0.8,
            compute_throughput: 0.85,
            memory_throughput: 0.3,
            l2_hit_rate: 0.7,
            shared_memory_efficiency: 0.9,
            warp_execution_efficiency: 0.95,
            ipc: 3.0,
        }
    }

    fn memory_bound_metrics() -> ProfileMetrics {
        ProfileMetrics {
            achieved_occupancy: 0.7,
            compute_throughput: 0.2,
            memory_throughput: 0.85,
            l2_hit_rate: 0.4,
            shared_memory_efficiency: 0.6,
            warp_execution_efficiency: 0.9,
            ipc: 1.5,
        }
    }

    fn latency_bound_metrics() -> ProfileMetrics {
        ProfileMetrics {
            achieved_occupancy: 0.3,
            compute_throughput: 0.15,
            memory_throughput: 0.2,
            l2_hit_rate: 0.5,
            shared_memory_efficiency: 0.7,
            warp_execution_efficiency: 0.8,
            ipc: 0.5,
        }
    }

    // --- Bottleneck classification tests ---

    #[test]
    fn classify_compute_bound() {
        let opt = ProfileGuidedOptimizer::new(make_profile(compute_bound_metrics()));
        assert_eq!(opt.classify_bottleneck(), Bottleneck::ComputeBound);
    }

    #[test]
    fn classify_memory_bound() {
        let opt = ProfileGuidedOptimizer::new(make_profile(memory_bound_metrics()));
        assert_eq!(opt.classify_bottleneck(), Bottleneck::MemoryBound);
    }

    #[test]
    fn classify_latency_bound() {
        let opt = ProfileGuidedOptimizer::new(make_profile(latency_bound_metrics()));
        assert_eq!(opt.classify_bottleneck(), Bottleneck::LatencyBound);
    }

    #[test]
    fn classify_balanced() {
        let opt = ProfileGuidedOptimizer::new(make_profile(balanced_metrics()));
        assert_eq!(opt.classify_bottleneck(), Bottleneck::Balanced);
    }

    #[test]
    fn classify_both_saturated_is_balanced() {
        let mut m = balanced_metrics();
        m.compute_throughput = 0.8;
        m.memory_throughput = 0.8;
        let opt = ProfileGuidedOptimizer::new(make_profile(m));
        assert_eq!(opt.classify_bottleneck(), Bottleneck::Balanced);
    }

    // --- Decision generation tests ---

    #[test]
    fn compute_bound_suggests_larger_tiles() {
        let opt = ProfileGuidedOptimizer::new(make_profile(compute_bound_metrics()));
        let decisions = opt.analyze();
        assert!(
            decisions
                .iter()
                .any(|d| matches!(d, CodeGenDecision::UseLargerTiles { .. })),
            "expected UseLargerTiles in {decisions:?}"
        );
    }

    #[test]
    fn memory_bound_with_poor_coalescing_suggests_shared_mem() {
        let mut profile = make_profile(memory_bound_metrics());
        profile.memory_access_pattern.coalesced_ratio = 0.3;
        let opt = ProfileGuidedOptimizer::new(profile);
        let decisions = opt.analyze();
        assert!(
            decisions
                .iter()
                .any(|d| matches!(d, CodeGenDecision::SwitchToSharedMemory)),
            "expected SwitchToSharedMemory in {decisions:?}"
        );
    }

    #[test]
    fn latency_bound_suggests_split_k() {
        let opt = ProfileGuidedOptimizer::new(make_profile(latency_bound_metrics()));
        let decisions = opt.analyze();
        assert!(
            decisions
                .iter()
                .any(|d| matches!(d, CodeGenDecision::EnableSplitK { .. })),
            "expected EnableSplitK in {decisions:?}"
        );
    }

    #[test]
    fn low_occupancy_suggests_increase() {
        let mut m = balanced_metrics();
        m.achieved_occupancy = 0.3;
        let opt = ProfileGuidedOptimizer::new(make_profile(m));
        let decisions = opt.analyze();
        assert!(
            decisions
                .iter()
                .any(|d| matches!(d, CodeGenDecision::IncreaseOccupancy { .. })),
            "expected IncreaseOccupancy in {decisions:?}"
        );
    }

    // --- Branch bias tests ---

    #[test]
    fn branch_profile_taken_ratio() {
        let bp = BranchProfile {
            branch_index: 0,
            taken_count: 900,
            not_taken_count: 100,
        };
        let ratio = bp.taken_ratio();
        assert!((ratio - 0.9).abs() < 1e-9);
    }

    #[test]
    fn branch_profile_zero_executions() {
        let bp = BranchProfile {
            branch_index: 0,
            taken_count: 0,
            not_taken_count: 0,
        };
        assert!((bp.taken_ratio() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn branch_bias_detection() {
        let bp = BranchProfile {
            branch_index: 0,
            taken_count: 950,
            not_taken_count: 50,
        };
        assert!(bp.is_biased(0.9));
        assert!(!bp.is_biased(0.96));
    }

    #[test]
    fn biased_branch_triggers_predication() {
        let mut profile = make_profile(balanced_metrics());
        profile.branch_stats.push(BranchProfile {
            branch_index: 0,
            taken_count: 980,
            not_taken_count: 20,
        });
        let opt = ProfileGuidedOptimizer::new(profile);
        let decisions = opt.analyze();
        assert!(
            decisions
                .iter()
                .any(|d| matches!(d, CodeGenDecision::PredicateBranch)),
            "expected PredicateBranch in {decisions:?}"
        );
    }

    // --- Unroll factor tests ---

    #[test]
    fn unroll_factor_high_mem_stalls() {
        let mut profile = make_profile(balanced_metrics());
        for i in 0..4 {
            profile.hotspots.push(HotSpot {
                instruction_index: i,
                cycle_count: 500,
                stall_reason: StallReason::MemoryDependency,
            });
        }
        let opt = ProfileGuidedOptimizer::new(profile);
        assert_eq!(opt.suggest_unroll_factor(), 8);
    }

    #[test]
    fn unroll_factor_low_ipc() {
        let mut m = balanced_metrics();
        m.ipc = 0.8;
        let opt = ProfileGuidedOptimizer::new(make_profile(m));
        assert_eq!(opt.suggest_unroll_factor(), 4);
    }

    #[test]
    fn unroll_factor_moderate_ipc() {
        let mut m = balanced_metrics();
        m.ipc = 1.5;
        let opt = ProfileGuidedOptimizer::new(make_profile(m));
        assert_eq!(opt.suggest_unroll_factor(), 2);
    }

    #[test]
    fn unroll_factor_high_ipc_no_unroll() {
        let m = balanced_metrics(); // ipc = 2.5
        let opt = ProfileGuidedOptimizer::new(make_profile(m));
        assert_eq!(opt.suggest_unroll_factor(), 1);
    }

    // --- Tile suggestion tests ---

    #[test]
    fn tile_config_compute_bound_ampere() {
        let opt = ProfileGuidedOptimizer::new(make_profile(compute_bound_metrics()));
        let tc = opt.suggest_tile_config(512, 512, 256);
        assert_eq!(tc.tile_m, 128);
        assert_eq!(tc.tile_n, 128);
        assert_eq!(tc.tile_k, 16);
    }

    #[test]
    fn tile_config_compute_bound_hopper() {
        let mut profile = make_profile(compute_bound_metrics());
        profile.sm_version = SmVersion::Sm90;
        let opt = ProfileGuidedOptimizer::new(profile);
        let tc = opt.suggest_tile_config(512, 512, 256);
        assert_eq!(tc.tile_m, 256);
        assert_eq!(tc.tile_n, 128);
    }

    #[test]
    fn tile_config_clamps_to_problem_size() {
        let opt = ProfileGuidedOptimizer::new(make_profile(compute_bound_metrics()));
        let tc = opt.suggest_tile_config(32, 16, 4);
        assert_eq!(tc.tile_m, 32);
        assert_eq!(tc.tile_n, 16);
        assert_eq!(tc.tile_k, 4);
    }

    #[test]
    fn tile_config_memory_bound_uses_deep_k() {
        let opt = ProfileGuidedOptimizer::new(make_profile(memory_bound_metrics()));
        let tc = opt.suggest_tile_config(512, 512, 256);
        assert_eq!(tc.tile_k, 32);
    }

    // --- apply_profile_decisions tests ---

    #[test]
    fn apply_decisions_updates_config() {
        let decisions = vec![
            CodeGenDecision::UnrollLoop { factor: 4 },
            CodeGenDecision::SwitchToSharedMemory,
            CodeGenDecision::EnableSplitK { k_slices: 8 },
            CodeGenDecision::UseLargerTiles {
                tile_m: 128,
                tile_n: 256,
            },
        ];
        let mut config = KernelProfile::new();
        let log = apply_profile_decisions(&decisions, &mut config);

        assert_eq!(config.unroll_factor, 4);
        assert!(config.use_shared_memory);
        assert_eq!(config.split_k, 8);
        assert_eq!(config.tile_m, 128);
        assert_eq!(config.tile_n, 256);
        assert_eq!(log.len(), 4);
    }

    #[test]
    fn apply_increase_occupancy_sets_register_target() {
        let decisions = vec![CodeGenDecision::IncreaseOccupancy { target_blocks: 4 }];
        let mut config = KernelProfile::new();
        let log = apply_profile_decisions(&decisions, &mut config);
        // 255 / 4 = 63
        assert_eq!(config.register_target, 63);
        assert_eq!(log.len(), 1);
    }

    // --- Display trait tests ---

    #[test]
    fn display_bottleneck() {
        assert_eq!(format!("{}", Bottleneck::ComputeBound), "compute-bound");
        assert_eq!(format!("{}", Bottleneck::MemoryBound), "memory-bound");
        assert_eq!(format!("{}", Bottleneck::LatencyBound), "latency-bound");
        assert_eq!(format!("{}", Bottleneck::Balanced), "balanced");
    }

    #[test]
    fn display_stall_reason() {
        assert_eq!(format!("{}", StallReason::None), "none");
        assert_eq!(
            format!("{}", StallReason::MemoryDependency),
            "memory_dependency"
        );
        assert_eq!(
            format!("{}", StallReason::Other("pipe_busy".to_string())),
            "other(pipe_busy)"
        );
    }

    #[test]
    fn display_code_gen_decision() {
        let d = CodeGenDecision::UnrollLoop { factor: 4 };
        assert_eq!(format!("{d}"), "unroll loop x4");
        let d = CodeGenDecision::EnableSplitK { k_slices: 8 };
        assert_eq!(format!("{d}"), "enable split-K (8 slices)");
    }

    #[test]
    fn display_kernel_profile() {
        let kp = KernelProfile::new();
        let s = format!("{kp}");
        assert!(s.contains("tile=64x64x8"));
        assert!(s.contains("smem=off"));
    }

    #[test]
    fn display_tile_config() {
        let tc = TileConfig {
            tile_m: 128,
            tile_n: 64,
            tile_k: 16,
        };
        assert_eq!(format!("{tc}"), "128x64x16");
    }

    #[test]
    fn display_memory_access_profile() {
        let m = MemoryAccessProfile {
            coalesced_ratio: 0.95,
            bank_conflict_rate: 0.02,
            cache_line_utilization: 0.88,
        };
        let s = format!("{m}");
        assert!(s.contains("coalesced=95.0%"));
    }

    #[test]
    fn display_branch_profile() {
        let bp = BranchProfile {
            branch_index: 3,
            taken_count: 750,
            not_taken_count: 250,
        };
        let s = format!("{bp}");
        assert!(s.contains("branch[3]"));
        assert!(s.contains("75.00%"));
    }

    // --- End-to-end: profile -> decisions -> applied config ---

    #[test]
    fn end_to_end_compute_bound_pipeline() {
        let profile = make_profile(compute_bound_metrics());
        let opt = ProfileGuidedOptimizer::new(profile);
        assert_eq!(opt.classify_bottleneck(), Bottleneck::ComputeBound);

        let decisions = opt.analyze();
        let mut config = KernelProfile::new();
        let log = apply_profile_decisions(&decisions, &mut config);

        // Compute-bound should have enlarged tiles.
        assert!(config.tile_m >= 128);
        assert!(!log.is_empty());
    }
}
