//! Pruned FFT — skip computation for known-zero input/output elements.
//!
//! When an FFT input is zero-padded (only the first M of N elements are
//! nonzero) or only a subset K of the N output elements is needed, many
//! butterfly operations produce trivially-zero results.  This module
//! analyses the Cooley-Tukey decomposition to identify and eliminate those
//! butterflies, producing a [`PrunedFftPlan`] that records per-stage
//! activity maps and the overall FLOP savings.
//!
//! # Usage
//!
//! ```rust
//! use oxicuda_fft::pruned::*;
//! use oxicuda_fft::types::{FftDirection, FftPrecision};
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let config = PrunedFftConfig {
//!     fft_size: 1024,
//!     input_nonzero_count: 64,
//!     output_needed_count: 1024,
//!     direction: FftDirection::Forward,
//!     precision: FftPrecision::Single,
//!     sm_version: SmVersion::Sm80,
//! };
//!
//! let plan = plan_pruned_fft(&config).expect("pruned plan");
//! assert!(plan.flop_savings_ratio > 0.0);
//! ```
#![allow(dead_code)]

use std::fmt;

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::error::PtxGenError;
use oxicuda_ptx::ir::PtxType;

use crate::error::{FftError, FftResult};
use crate::ptx_helpers::ptx_type_suffix;
use crate::types::{FftDirection, FftPrecision};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a pruned FFT.
///
/// Describes the full FFT size together with the sparsity pattern of the
/// input (first `input_nonzero_count` elements are nonzero, the rest are
/// zero-padded) and how many output bins are actually required.
#[derive(Debug, Clone)]
pub struct PrunedFftConfig {
    /// Full FFT size N (must be a power of 2 and >= 2).
    pub fft_size: usize,
    /// Number of nonzero input elements (first M elements, 1 <= M <= N).
    pub input_nonzero_count: usize,
    /// Number of output elements needed (first K elements, 1 <= K <= N).
    pub output_needed_count: usize,
    /// Transform direction.
    pub direction: FftDirection,
    /// Floating-point precision.
    pub precision: FftPrecision,
    /// Target GPU SM version for PTX generation.
    pub sm_version: SmVersion,
}

// ---------------------------------------------------------------------------
// Prune mode
// ---------------------------------------------------------------------------

/// Describes which dimensions of the FFT are pruned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PruneMode {
    /// Only the input is sparse (zero-padded); all outputs are needed.
    InputOnly,
    /// The input is dense; only a subset of outputs is needed.
    OutputOnly,
    /// Both input is sparse and only a subset of outputs is needed.
    Both,
}

impl fmt::Display for PruneMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputOnly => write!(f, "InputOnly"),
            Self::OutputOnly => write!(f, "OutputOnly"),
            Self::Both => write!(f, "Both"),
        }
    }
}

// ---------------------------------------------------------------------------
// Pruned stage
// ---------------------------------------------------------------------------

/// Per-stage pruning analysis.
///
/// Each Cooley-Tukey radix-2 stage has N/2 butterfly operations.  When the
/// input is zero-padded, some of those butterflies have both inputs zero
/// and can be skipped entirely.
#[derive(Debug, Clone)]
pub struct PrunedStage {
    /// Zero-based stage index (stage 0 is the first butterfly pass).
    pub stage_index: u32,
    /// Radix of this stage (always 2 for the current implementation).
    pub radix: u32,
    /// Number of butterflies that must actually be computed.
    pub active_butterflies: u64,
    /// Total butterflies in a full (unpruned) stage.
    pub total_butterflies: u64,
    /// `true` when `active_butterflies == 0`, meaning the entire stage
    /// produces only zeros and can be elided.
    pub can_skip_entirely: bool,
}

impl fmt::Display for PrunedStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pct = if self.total_butterflies > 0 {
            (self.active_butterflies as f64 / self.total_butterflies as f64) * 100.0
        } else {
            0.0
        };
        write!(
            f,
            "Stage {} (radix-{}): {}/{} butterflies ({:.1}% active){}",
            self.stage_index,
            self.radix,
            self.active_butterflies,
            self.total_butterflies,
            pct,
            if self.can_skip_entirely {
                " [SKIP]"
            } else {
                ""
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Pruned FFT plan
// ---------------------------------------------------------------------------

/// A pruned FFT execution plan.
///
/// Contains the per-stage pruning analysis and overall savings metrics.
/// Created by [`plan_pruned_fft`].
#[derive(Debug, Clone)]
pub struct PrunedFftPlan {
    /// The configuration that produced this plan.
    pub config: PrunedFftConfig,
    /// Per-stage pruning information.
    pub pruned_stages: Vec<PrunedStage>,
    /// Fraction of FLOPs saved (0.0 = no savings, 1.0 = everything skipped).
    pub flop_savings_ratio: f64,
    /// Total number of butterflies that are skipped.
    pub skipped_butterflies: u64,
    /// Total number of butterflies in the unpruned FFT.
    pub total_butterflies: u64,
}

impl fmt::Display for PrunedFftPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "PrunedFftPlan(N={}, M={}, K={})",
            self.config.fft_size, self.config.input_nonzero_count, self.config.output_needed_count,
        )?;
        writeln!(f, "  Mode: {}", optimal_prune_mode(&self.config),)?;
        writeln!(
            f,
            "  FLOP savings: {:.1}% ({} of {} butterflies skipped)",
            self.flop_savings_ratio * 100.0,
            self.skipped_butterflies,
            self.total_butterflies,
        )?;
        for stage in &self.pruned_stages {
            writeln!(f, "  {stage}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates that a [`PrunedFftConfig`] is well-formed.
///
/// # Errors
///
/// - [`FftError::InvalidSize`] if `fft_size` is zero, not a power of 2, or
///   less than 2.
/// - [`FftError::InvalidSize`] if `input_nonzero_count` or
///   `output_needed_count` is zero or exceeds `fft_size`.
pub fn validate_pruned_config(config: &PrunedFftConfig) -> FftResult<()> {
    if config.fft_size < 2 {
        return Err(FftError::InvalidSize(
            "pruned FFT size must be >= 2".to_string(),
        ));
    }
    if !config.fft_size.is_power_of_two() {
        return Err(FftError::InvalidSize(format!(
            "pruned FFT size {} is not a power of 2",
            config.fft_size,
        )));
    }
    if config.input_nonzero_count == 0 {
        return Err(FftError::InvalidSize(
            "input_nonzero_count must be >= 1".to_string(),
        ));
    }
    if config.input_nonzero_count > config.fft_size {
        return Err(FftError::InvalidSize(format!(
            "input_nonzero_count ({}) exceeds fft_size ({})",
            config.input_nonzero_count, config.fft_size,
        )));
    }
    if config.output_needed_count == 0 {
        return Err(FftError::InvalidSize(
            "output_needed_count must be >= 1".to_string(),
        ));
    }
    if config.output_needed_count > config.fft_size {
        return Err(FftError::InvalidSize(format!(
            "output_needed_count ({}) exceeds fft_size ({})",
            config.output_needed_count, config.fft_size,
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Prune mode selection
// ---------------------------------------------------------------------------

/// Determines the optimal pruning mode for a given configuration.
///
/// - If only the input is sparse: [`PruneMode::InputOnly`]
/// - If only the output is partial: [`PruneMode::OutputOnly`]
/// - If both: [`PruneMode::Both`]
pub fn optimal_prune_mode(config: &PrunedFftConfig) -> PruneMode {
    let input_sparse = config.input_nonzero_count < config.fft_size;
    let output_partial = config.output_needed_count < config.fft_size;

    match (input_sparse, output_partial) {
        (true, true) => PruneMode::Both,
        (true, false) => PruneMode::InputOnly,
        (false, true) => PruneMode::OutputOnly,
        // Even when both are full, we treat it as InputOnly (no pruning).
        (false, false) => PruneMode::InputOnly,
    }
}

// ---------------------------------------------------------------------------
// Active butterfly computation
// ---------------------------------------------------------------------------

/// Computes the number of active (non-trivially-zero) butterflies for one
/// stage of an input-pruned radix-2 Cooley-Tukey FFT.
///
/// In a decimation-in-time (DIT) FFT, stage `s` (0-indexed) operates on
/// groups of size `2^(s+1)`.  Within each group, the butterfly at position
/// `j` combines element `j` with element `j + 2^s`.  If both of those
/// elements are known to be zero (because they came from zero-padded
/// input and no earlier stage has made them nonzero), the butterfly can
/// be skipped.
///
/// We track a conservative "nonzero frontier": after stage `s`, the number
/// of potentially-nonzero elements grows because each butterfly can
/// propagate nonzero values to its partner.  The frontier at stage `s` is
/// `min(N, nonzero_count * 2^s)` for DIT (since each stage can at most
/// double the reach of nonzero data).
///
/// # Arguments
///
/// - `fft_size`: full FFT size N (must be power of 2)
/// - `nonzero_count`: number of nonzero input elements M
/// - `stage`: 0-based stage index
/// - `radix`: radix of the butterfly (currently only 2 is meaningful)
///
/// # Returns
///
/// The number of butterfly pairs that need computation in this stage.
pub fn compute_active_butterflies(
    fft_size: usize,
    nonzero_count: usize,
    stage: u32,
    _radix: u32,
) -> u64 {
    if nonzero_count >= fft_size {
        // No pruning possible — all butterflies active.
        return (fft_size / 2) as u64;
    }

    let total_butterflies = (fft_size / 2) as u64;

    // Group size at this stage (DIT): 2^(stage+1)
    let half_group = 1_usize << stage;
    let group_size = half_group * 2;

    // Conservative nonzero frontier: after stage s, nonzero elements can
    // reach at most min(N, M * 2^s) positions via butterfly propagation.
    // But we need to think in terms of which butterfly *inputs* are nonzero.
    //
    // At stage s, the inputs come from the bit-reversal permutation of
    // the original data.  For DIT, stage 0 butterflies combine pairs
    // that are N/2 apart in the original ordering.
    //
    // A simpler model: the number of groups that contain at least one
    // nonzero element.  At stage s, each group has `group_size` elements.
    // The first M elements (in natural order) are nonzero.  After
    // bit-reversal, these are scattered, but in the worst case every
    // group that overlaps with the first M elements has active butterflies.

    // Number of groups that are fully or partially nonzero
    let num_groups = fft_size / group_size;

    // For input pruning with DIT: at stage s, the "reach" of nonzero
    // data is min(N, nonzero_count << stage).
    let reach = std::cmp::min(fft_size, nonzero_count.saturating_mul(1 << stage));

    // Number of complete groups covered by the reach
    let full_groups = reach / group_size;
    // Partial group remainder
    let remainder = reach % group_size;

    // Each full group contributes half_group butterflies
    let mut active = (full_groups as u64) * (half_group as u64);

    // Partial group: butterflies where at least one input is nonzero
    if remainder > 0 && full_groups < num_groups {
        // In the partial group, elements [0..remainder) are potentially nonzero.
        // A butterfly at position j within the group pairs j with j + half_group.
        // Active if j < remainder OR j + half_group < remainder.
        let active_in_partial = if remainder >= half_group {
            // All half_group butterflies are active (the top half is at
            // least partially touched).
            half_group as u64
        } else {
            // Only butterflies where j < remainder are active (the
            // partner at j + half_group is zero but the butterfly still
            // needs to propagate the nonzero value).
            remainder as u64
        };
        active += active_in_partial;
    }

    std::cmp::min(active, total_butterflies)
}

/// Computes active butterflies for output pruning (inverse direction).
///
/// When only the first K outputs are needed, later stages can skip
/// butterflies whose results are not consumed by any needed output.
/// This is the "dual" of input pruning applied to the output side.
fn compute_output_active_butterflies(
    fft_size: usize,
    output_needed: usize,
    stage: u32,
    total_stages: u32,
) -> u64 {
    if output_needed >= fft_size {
        return (fft_size / 2) as u64;
    }

    let total_butterflies = (fft_size / 2) as u64;

    // For output pruning we work backwards: stage `total_stages - 1` is
    // the last stage (closest to the output).  At the last stage, only
    // butterflies producing the first K outputs are needed.  Going
    // backwards (towards the input), the "needed" set expands.
    let reverse_stage = total_stages.saturating_sub(1).saturating_sub(stage);
    let half_group = 1_usize << reverse_stage;
    let group_size = half_group * 2;
    let num_groups = fft_size / group_size;

    let reach = std::cmp::min(fft_size, output_needed.saturating_mul(1 << reverse_stage));

    let full_groups = reach / group_size;
    let remainder = reach % group_size;

    let mut active = (full_groups as u64) * (half_group as u64);

    if remainder > 0 && full_groups < num_groups {
        let active_in_partial = if remainder >= half_group {
            half_group as u64
        } else {
            remainder as u64
        };
        active += active_in_partial;
    }

    std::cmp::min(active, total_butterflies)
}

// ---------------------------------------------------------------------------
// Plan construction
// ---------------------------------------------------------------------------

/// Creates a pruned FFT execution plan.
///
/// Analyses the given configuration and produces a per-stage breakdown of
/// which butterflies are active, together with overall savings metrics.
///
/// # Errors
///
/// Returns an error if the configuration is invalid (see
/// [`validate_pruned_config`]).
pub fn plan_pruned_fft(config: &PrunedFftConfig) -> FftResult<PrunedFftPlan> {
    validate_pruned_config(config)?;

    let n = config.fft_size;
    let total_stages = (n as f64).log2().round() as u32;
    let mode = optimal_prune_mode(config);

    let mut stages = Vec::with_capacity(total_stages as usize);
    let mut total_active: u64 = 0;
    let mut total_full: u64 = 0;

    for s in 0..total_stages {
        let full_butterflies = (n / 2) as u64;

        let active = match mode {
            PruneMode::InputOnly => compute_active_butterflies(n, config.input_nonzero_count, s, 2),
            PruneMode::OutputOnly => {
                compute_output_active_butterflies(n, config.output_needed_count, s, total_stages)
            }
            PruneMode::Both => {
                // Take the minimum of input-pruned and output-pruned
                // active counts — a butterfly can be skipped if *either*
                // pruning strategy says it is inactive.
                let input_active = compute_active_butterflies(n, config.input_nonzero_count, s, 2);
                let output_active = compute_output_active_butterflies(
                    n,
                    config.output_needed_count,
                    s,
                    total_stages,
                );
                std::cmp::min(input_active, output_active)
            }
        };

        let stage = PrunedStage {
            stage_index: s,
            radix: 2,
            active_butterflies: active,
            total_butterflies: full_butterflies,
            can_skip_entirely: active == 0,
        };

        total_active += active;
        total_full += full_butterflies;
        stages.push(stage);
    }

    let skipped = total_full.saturating_sub(total_active);
    let savings = if total_full > 0 {
        skipped as f64 / total_full as f64
    } else {
        0.0
    };

    Ok(PrunedFftPlan {
        config: config.clone(),
        pruned_stages: stages,
        flop_savings_ratio: savings,
        skipped_butterflies: skipped,
        total_butterflies: total_full,
    })
}

// ---------------------------------------------------------------------------
// FLOP estimation
// ---------------------------------------------------------------------------

/// Estimates the number of floating-point operations for a full (unpruned)
/// FFT of size N.
///
/// Uses the standard `5 * N * log2(N)` model (each radix-2 butterfly
/// requires 1 complex multiply + 1 complex add = 10 real flops, applied to
/// N/2 butterflies per stage over log2(N) stages, giving 5*N*log2(N)).
pub fn estimate_full_flops(n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    5.0 * (n as f64) * (n as f64).log2()
}

/// Estimates the number of floating-point operations for a pruned FFT.
///
/// Sums per-stage active butterflies, each contributing 10 real FLOPs
/// (one complex multiply + one complex add).
pub fn estimate_pruned_flops(plan: &PrunedFftPlan) -> f64 {
    let active: u64 = plan
        .pruned_stages
        .iter()
        .map(|s| s.active_butterflies)
        .sum();
    // 10 real flops per butterfly (complex mul = 6, complex add = 4,
    // but with FMA the effective count is ~10).
    active as f64 * 10.0
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates a PTX kernel for one pruned butterfly stage.
///
/// The kernel processes only the active butterflies identified in `stage`,
/// loading input elements, applying twiddle factors, and storing the
/// results.  Inactive butterflies are not touched — their outputs remain
/// zero (either from initialisation or a previous stage).
///
/// # Arguments
///
/// - `config`: the pruned FFT configuration
/// - `stage`: the particular stage to generate code for
///
/// # Errors
///
/// Returns [`PtxGenError`] if PTX construction fails.
pub fn generate_pruned_butterfly_ptx(
    config: &PrunedFftConfig,
    stage: &PrunedStage,
) -> Result<String, PtxGenError> {
    let suffix = ptx_type_suffix(config.precision);
    let float_ty = match config.precision {
        FftPrecision::Single => PtxType::F32,
        FftPrecision::Double => PtxType::F64,
    };

    let n = config.fft_size;
    let stage_idx = stage.stage_index;
    let active = stage.active_butterflies;

    let kernel_name = format!("pruned_butterfly_{suffix}_n{n}_s{stage_idx}_a{active}");

    let half_group = 1_u32 << stage_idx;
    let group_size = half_group * 2;
    let dir_sign = config.direction.sign();

    // Precompute twiddle angle quantum for this stage.
    let twiddle_base_angle = dir_sign * 2.0 * std::f64::consts::PI / (group_size as f64);

    let ptx = KernelBuilder::new(&kernel_name)
        .target(config.sm_version)
        .param("input", PtxType::U64)
        .param("output", PtxType::U64)
        .param("n_active", PtxType::U32)
        .body(move |b| {
            // Thread index = global butterfly index within active set.
            let tid = b.global_thread_id_x();
            let tid_name = format!("{tid}");
            let n_active_reg = b.load_param_u32("n_active");

            b.if_lt_u32(tid, n_active_reg, |b| {
                let input_ptr = b.load_param_u64("input");
                let output_ptr = b.load_param_u64("output");

                // Map thread to butterfly index within its group.
                // group_index = tid / half_group
                // j = tid % half_group
                let half_group_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {half_group_reg}, {half_group};"));

                let group_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "div.u32 {group_idx}, {tid_name}, {half_group_reg};"
                ));

                let j = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("rem.u32 {j}, {tid_name}, {half_group_reg};"));

                // idx_a = group_index * group_size + j
                // idx_b = idx_a + half_group
                let group_size_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {group_size_reg}, {group_size};"));
                let base = b.mul_lo_u32(group_idx, group_size_reg);
                let idx_a = b.add_u32(base, j.clone());
                let idx_a_name = format!("{idx_a}");
                let idx_b = b.add_u32(idx_a.clone(), half_group_reg);
                let idx_b_name = format!("{idx_b}");

                // Byte offsets (complex = 2 floats)
                let elem_bytes = match float_ty {
                    PtxType::F32 => 8_u32,  // 2 * 4
                    PtxType::F64 => 16_u32, // 2 * 8
                    _ => 8_u32,
                };
                let elem_bytes_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {elem_bytes_reg}, {elem_bytes};"));

                // Address of element a
                let offset_a = b.mul_lo_u32(idx_a, elem_bytes_reg.clone());
                let offset_a_64 = b.cvt_u32_to_u64(offset_a);
                let addr_a = b.add_u64(input_ptr.clone(), offset_a_64);

                // Address of element b
                let offset_b = b.mul_lo_u32(idx_b, elem_bytes_reg);
                let offset_b_64 = b.cvt_u32_to_u64(offset_b);
                let addr_b = b.add_u64(input_ptr, offset_b_64);

                // Load complex elements a and b
                let (a_re, a_im) = match float_ty {
                    PtxType::F32 => {
                        let re = b.load_global_f32(addr_a.clone());
                        let offset_4 = b.mov_imm_u32(4);
                        let offset_4_64 = b.cvt_u32_to_u64(offset_4);
                        let addr_a_im = b.add_u64(addr_a, offset_4_64);
                        let im = b.load_global_f32(addr_a_im);
                        (re, im)
                    }
                    PtxType::F64 => {
                        let re = b.load_global_f64(addr_a.clone());
                        let offset_8 = b.mov_imm_u32(8);
                        let offset_8_64 = b.cvt_u32_to_u64(offset_8);
                        let addr_a_im = b.add_u64(addr_a, offset_8_64);
                        let im = b.load_global_f64(addr_a_im);
                        (re, im)
                    }
                    _ => return,
                };

                let (b_re, b_im) = match float_ty {
                    PtxType::F32 => {
                        let re = b.load_global_f32(addr_b.clone());
                        let offset_4 = b.mov_imm_u32(4);
                        let offset_4_64 = b.cvt_u32_to_u64(offset_4);
                        let addr_b_im = b.add_u64(addr_b, offset_4_64);
                        let im = b.load_global_f32(addr_b_im);
                        (re, im)
                    }
                    PtxType::F64 => {
                        let re = b.load_global_f64(addr_b.clone());
                        let offset_8 = b.mov_imm_u32(8);
                        let offset_8_64 = b.cvt_u32_to_u64(offset_8);
                        let addr_b_im = b.add_u64(addr_b, offset_8_64);
                        let im = b.load_global_f64(addr_b_im);
                        (re, im)
                    }
                    _ => return,
                };

                // Compute twiddle factor W = exp(twiddle_base_angle * j)
                // For the PTX kernel we embed cos/sin as computed from j.
                // In a real kernel this would use a twiddle table or
                // __sincosf; here we load the base angle and multiply by j.
                let j_float = match float_ty {
                    PtxType::F32 => {
                        let tmp = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {tmp}, {j};"));
                        tmp
                    }
                    PtxType::F64 => {
                        let tmp = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("cvt.rn.f64.u32 {tmp}, {j};"));
                        tmp
                    }
                    _ => return,
                };

                // Load twiddle base angle as immediate
                let angle = match float_ty {
                    PtxType::F32 => {
                        let base_reg = b.alloc_reg(PtxType::F32);
                        let bits = (twiddle_base_angle as f32).to_bits();
                        b.raw_ptx(&format!("mov.b32 {base_reg}, 0F{bits:08X};"));
                        let angle = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {angle}, {base_reg}, {j_float};"));
                        angle
                    }
                    PtxType::F64 => {
                        let base_reg = b.alloc_reg(PtxType::F64);
                        let bits = twiddle_base_angle.to_bits();
                        b.raw_ptx(&format!("mov.b64 {base_reg}, 0D{bits:016X};"));
                        let angle = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {angle}, {base_reg}, {j_float};"));
                        angle
                    }
                    _ => return,
                };

                // cos/sin approximation via PTX sin/cos intrinsics (f32)
                // or Taylor series comment for f64.
                // For f32 PTX has sin.approx.f32 and cos.approx.f32.
                let (tw_re, tw_im) = match float_ty {
                    PtxType::F32 => {
                        let cos_r = b.alloc_reg(PtxType::F32);
                        let sin_r = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cos.approx.f32 {cos_r}, {angle};"));
                        b.raw_ptx(&format!("sin.approx.f32 {sin_r}, {angle};"));
                        (cos_r, sin_r)
                    }
                    PtxType::F64 => {
                        // f64: lightweight polynomial with angle reduction
                        // x = angle - round(angle / 2pi) * 2pi, x in [-pi, pi]
                        // sin(x) ~= x - x^3/6 + x^5/120
                        // cos(x) ~= 1 - x^2/2 + x^4/24
                        let two_pi = b.alloc_reg(PtxType::F64);
                        let inv_two_pi = b.alloc_reg(PtxType::F64);
                        let one = b.alloc_reg(PtxType::F64);
                        let neg_half = b.alloc_reg(PtxType::F64);
                        let one_over_24 = b.alloc_reg(PtxType::F64);
                        let neg_one_over_6 = b.alloc_reg(PtxType::F64);
                        let one_over_120 = b.alloc_reg(PtxType::F64);

                        b.raw_ptx(&format!(
                            "mov.b64 {two_pi}, 0D{:016X};",
                            (2.0 * std::f64::consts::PI).to_bits()
                        ));
                        b.raw_ptx(&format!(
                            "mov.b64 {inv_two_pi}, 0D{:016X};",
                            (1.0 / (2.0 * std::f64::consts::PI)).to_bits()
                        ));
                        b.raw_ptx(&format!("mov.b64 {one}, 0D{:016X};", 1.0_f64.to_bits()));
                        b.raw_ptx(&format!(
                            "mov.b64 {neg_half}, 0D{:016X};",
                            (-0.5_f64).to_bits()
                        ));
                        b.raw_ptx(&format!(
                            "mov.b64 {one_over_24}, 0D{:016X};",
                            (1.0_f64 / 24.0_f64).to_bits()
                        ));
                        b.raw_ptx(&format!(
                            "mov.b64 {neg_one_over_6}, 0D{:016X};",
                            (-(1.0_f64 / 6.0_f64)).to_bits()
                        ));
                        b.raw_ptx(&format!(
                            "mov.b64 {one_over_120}, 0D{:016X};",
                            (1.0_f64 / 120.0_f64).to_bits()
                        ));

                        let scaled = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {scaled}, {angle}, {inv_two_pi};"));
                        let k_i64 = b.alloc_reg(PtxType::S64);
                        b.raw_ptx(&format!("cvt.rni.s64.f64 {k_i64}, {scaled};"));
                        let k_f64 = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("cvt.rn.f64.s64 {k_f64}, {k_i64};"));
                        let k_two_pi = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {k_two_pi}, {k_f64}, {two_pi};"));
                        let x = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("sub.rn.f64 {x}, {angle}, {k_two_pi};"));

                        let x2 = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {x2}, {x}, {x};"));
                        let x3 = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {x3}, {x2}, {x};"));
                        let x4 = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {x4}, {x2}, {x2};"));
                        let x5 = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {x5}, {x3}, {x2};"));

                        let cos_r = b.alloc_reg(PtxType::F64);
                        let sin_r = b.alloc_reg(PtxType::F64);

                        // sin(x) = x + (-1/6)*x^3 + (1/120)*x^5
                        let sin_t = b.fma_f64(neg_one_over_6, x3, x.clone());
                        b.raw_ptx(&format!(
                            "fma.rn.f64 {sin_r}, {one_over_120}, {x5}, {sin_t};"
                        ));

                        // cos(x) = 1 + (-1/2)*x^2 + (1/24)*x^4
                        let cos_t = b.fma_f64(neg_half, x2, one);
                        b.raw_ptx(&format!(
                            "fma.rn.f64 {cos_r}, {one_over_24}, {x4}, {cos_t};"
                        ));

                        (cos_r, sin_r)
                    }
                    _ => return,
                };

                // Butterfly: out_a = a + W*b,  out_b = a - W*b
                // W*b = (tw_re*b_re - tw_im*b_im, tw_re*b_im + tw_im*b_re)
                let (wb_re, wb_im) = match float_ty {
                    PtxType::F32 => {
                        let t1 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {t1}, {tw_re}, {b_re};"));
                        let t2 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("neg.f32 {t2}, {tw_im};"));
                        let wb_re = b.fma_f32(t2, b_im.clone(), t1);
                        let t3 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {t3}, {tw_re}, {b_im};"));
                        let wb_im = b.fma_f32(tw_im, b_re, t3);
                        (wb_re, wb_im)
                    }
                    PtxType::F64 => {
                        let t1 = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {t1}, {tw_re}, {b_re};"));
                        let t2 = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("neg.f64 {t2}, {tw_im};"));
                        let wb_re = b.fma_f64(t2, b_im.clone(), t1);
                        let t3 = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {t3}, {tw_re}, {b_im};"));
                        let wb_im = b.fma_f64(tw_im, b_re, t3);
                        (wb_re, wb_im)
                    }
                    _ => return,
                };

                // out_a = a + W*b
                let (out_a_re, out_a_im) = match float_ty {
                    PtxType::F32 => (
                        b.add_f32(a_re.clone(), wb_re.clone()),
                        b.add_f32(a_im.clone(), wb_im.clone()),
                    ),
                    PtxType::F64 => (
                        b.add_f64(a_re.clone(), wb_re.clone()),
                        b.add_f64(a_im.clone(), wb_im.clone()),
                    ),
                    _ => return,
                };

                // out_b = a - W*b
                let (out_b_re, out_b_im) = match float_ty {
                    PtxType::F32 => (b.sub_f32(a_re, wb_re), b.sub_f32(a_im, wb_im)),
                    PtxType::F64 => (b.sub_f64(a_re, wb_re), b.sub_f64(a_im, wb_im)),
                    _ => return,
                };

                // Store results to output — compute byte offsets via raw PTX
                // to avoid borrow-checker issues with idx_a/idx_b (already moved).
                let out_offset_a = match float_ty {
                    PtxType::F32 => {
                        let stride_reg = b.mov_imm_u32(8);
                        let oa = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mul.lo.u32 {oa}, {idx_a_name}, {stride_reg};"));
                        b.cvt_u32_to_u64(oa)
                    }
                    PtxType::F64 => {
                        let stride_reg = b.mov_imm_u32(16);
                        let oa = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mul.lo.u32 {oa}, {idx_a_name}, {stride_reg};"));
                        b.cvt_u32_to_u64(oa)
                    }
                    _ => return,
                };
                let out_addr_a = b.add_u64(output_ptr.clone(), out_offset_a);

                let out_offset_b = match float_ty {
                    PtxType::F32 => {
                        let stride_reg = b.mov_imm_u32(8);
                        let ob = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mul.lo.u32 {ob}, {idx_b_name}, {stride_reg};"));
                        b.cvt_u32_to_u64(ob)
                    }
                    PtxType::F64 => {
                        let stride_reg = b.mov_imm_u32(16);
                        let ob = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mul.lo.u32 {ob}, {idx_b_name}, {stride_reg};"));
                        b.cvt_u32_to_u64(ob)
                    }
                    _ => return,
                };
                let out_addr_b = b.add_u64(output_ptr, out_offset_b);

                match float_ty {
                    PtxType::F32 => {
                        b.store_global_f32(out_addr_a.clone(), out_a_re);
                        let imm4 = b.mov_imm_u32(4);
                        let imm4_64 = b.cvt_u32_to_u64(imm4);
                        let addr_a_im_out = b.add_u64(out_addr_a, imm4_64);
                        b.store_global_f32(addr_a_im_out, out_a_im);

                        b.store_global_f32(out_addr_b.clone(), out_b_re);
                        let imm4b = b.mov_imm_u32(4);
                        let imm4b_64 = b.cvt_u32_to_u64(imm4b);
                        let addr_b_im_out = b.add_u64(out_addr_b, imm4b_64);
                        b.store_global_f32(addr_b_im_out, out_b_im);
                    }
                    PtxType::F64 => {
                        b.store_global_f64(out_addr_a.clone(), out_a_re);
                        let imm8 = b.mov_imm_u32(8);
                        let imm8_64 = b.cvt_u32_to_u64(imm8);
                        let addr_a_im_out = b.add_u64(out_addr_a, imm8_64);
                        b.store_global_f64(addr_a_im_out, out_a_im);

                        b.store_global_f64(out_addr_b.clone(), out_b_re);
                        let imm8b = b.mov_imm_u32(8);
                        let imm8b_64 = b.cvt_u32_to_u64(imm8b);
                        let addr_b_im_out = b.add_u64(out_addr_b, imm8b_64);
                        b.store_global_f64(addr_b_im_out, out_b_im);
                    }
                    _ => {}
                }
            });

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a default test config.
    fn test_config(n: usize, m: usize, k: usize) -> PrunedFftConfig {
        PrunedFftConfig {
            fft_size: n,
            input_nonzero_count: m,
            output_needed_count: k,
            direction: FftDirection::Forward,
            precision: FftPrecision::Single,
            sm_version: SmVersion::Sm80,
        }
    }

    // -- Validation tests ---------------------------------------------------

    #[test]
    fn validate_rejects_zero_size() {
        let cfg = test_config(0, 0, 0);
        assert!(matches!(
            validate_pruned_config(&cfg),
            Err(FftError::InvalidSize(_))
        ));
    }

    #[test]
    fn validate_rejects_non_power_of_two() {
        let cfg = test_config(100, 50, 100);
        assert!(matches!(
            validate_pruned_config(&cfg),
            Err(FftError::InvalidSize(_))
        ));
    }

    #[test]
    fn validate_rejects_zero_input_count() {
        let cfg = test_config(256, 0, 256);
        assert!(matches!(
            validate_pruned_config(&cfg),
            Err(FftError::InvalidSize(_))
        ));
    }

    #[test]
    fn validate_rejects_input_exceeds_size() {
        let cfg = test_config(256, 512, 256);
        assert!(matches!(
            validate_pruned_config(&cfg),
            Err(FftError::InvalidSize(_))
        ));
    }

    #[test]
    fn validate_rejects_zero_output_count() {
        let cfg = test_config(256, 64, 0);
        assert!(matches!(
            validate_pruned_config(&cfg),
            Err(FftError::InvalidSize(_))
        ));
    }

    #[test]
    fn validate_accepts_valid_config() {
        let cfg = test_config(1024, 64, 1024);
        assert!(validate_pruned_config(&cfg).is_ok());
    }

    // -- Prune mode tests ---------------------------------------------------

    #[test]
    fn prune_mode_input_only() {
        let cfg = test_config(1024, 64, 1024);
        assert_eq!(optimal_prune_mode(&cfg), PruneMode::InputOnly);
    }

    #[test]
    fn prune_mode_output_only() {
        let cfg = test_config(1024, 1024, 128);
        assert_eq!(optimal_prune_mode(&cfg), PruneMode::OutputOnly);
    }

    #[test]
    fn prune_mode_both() {
        let cfg = test_config(1024, 64, 128);
        assert_eq!(optimal_prune_mode(&cfg), PruneMode::Both);
    }

    // -- Plan generation tests ----------------------------------------------

    #[test]
    fn plan_basic_input_pruned() {
        let cfg = test_config(1024, 64, 1024);
        let plan = plan_pruned_fft(&cfg);
        assert!(plan.is_ok());
        let plan = plan.unwrap_or_else(|e| panic!("plan failed: {e}"));
        // With 64/1024 nonzero inputs, there should be significant savings.
        assert!(
            plan.flop_savings_ratio > 0.0,
            "expected savings > 0, got {}",
            plan.flop_savings_ratio,
        );
        assert_eq!(plan.pruned_stages.len(), 10); // log2(1024) = 10
    }

    #[test]
    fn plan_no_pruning_full() {
        // M == N and K == N => no savings
        let cfg = test_config(256, 256, 256);
        let plan = plan_pruned_fft(&cfg);
        assert!(plan.is_ok());
        let plan = plan.unwrap_or_else(|e| panic!("plan failed: {e}"));
        assert!(
            (plan.flop_savings_ratio - 0.0).abs() < 1e-12,
            "expected 0 savings for full FFT, got {}",
            plan.flop_savings_ratio,
        );
        assert_eq!(plan.skipped_butterflies, 0);
    }

    #[test]
    fn plan_minimal_input() {
        // Only 1 nonzero input element
        let cfg = test_config(256, 1, 256);
        let plan = plan_pruned_fft(&cfg);
        assert!(plan.is_ok());
        let plan = plan.unwrap_or_else(|e| panic!("{e}"));
        // Stage 0: only 1 active butterfly (out of 128)
        assert_eq!(plan.pruned_stages[0].active_butterflies, 1);
        // Significant overall savings
        assert!(plan.flop_savings_ratio > 0.3);
    }

    #[test]
    fn plan_stages_monotonically_grow() {
        // Active butterflies should generally grow or stay the same
        // as we progress through stages (nonzero reach expands).
        let cfg = test_config(1024, 32, 1024);
        let plan = plan_pruned_fft(&cfg);
        assert!(plan.is_ok());
        let plan = plan.unwrap_or_else(|e| panic!("{e}"));
        for i in 1..plan.pruned_stages.len() {
            assert!(
                plan.pruned_stages[i].active_butterflies
                    >= plan.pruned_stages[i - 1].active_butterflies,
                "stage {} has fewer active butterflies than stage {}",
                i,
                i - 1,
            );
        }
    }

    // -- Active butterfly computation tests ---------------------------------

    #[test]
    fn active_butterflies_full_input() {
        // All inputs nonzero => all butterflies active
        let active = compute_active_butterflies(256, 256, 0, 2);
        assert_eq!(active, 128);
    }

    #[test]
    fn active_butterflies_stage_zero_sparse() {
        // N=16, M=2: stage 0, half_group=1, group_size=2
        // reach = min(16, 2*1) = 2, full_groups = 2/2 = 1, remainder = 0
        // active = 1 full group * 1 butterfly/group = 1
        let active = compute_active_butterflies(16, 2, 0, 2);
        assert_eq!(active, 1);
        assert!(active <= 8); // N/2 = 8 total butterflies
    }

    // -- FLOP estimation tests ----------------------------------------------

    #[test]
    fn full_flops_known_value() {
        // N=1024: 5 * 1024 * 10 = 51200
        let flops = estimate_full_flops(1024);
        let expected = 5.0 * 1024.0 * 10.0;
        assert!(
            (flops - expected).abs() < 1e-6,
            "expected {expected}, got {flops}",
        );
    }

    #[test]
    fn pruned_flops_less_than_full() {
        let cfg = test_config(1024, 64, 1024);
        let plan = plan_pruned_fft(&cfg).unwrap_or_else(|e| panic!("{e}"));
        let pruned = estimate_pruned_flops(&plan);
        let full = estimate_full_flops(1024);
        assert!(
            pruned < full,
            "pruned ({pruned}) should be less than full ({full})",
        );
    }

    // -- Display tests ------------------------------------------------------

    #[test]
    fn display_plan_contains_key_info() {
        let cfg = test_config(256, 32, 256);
        let plan = plan_pruned_fft(&cfg).unwrap_or_else(|e| panic!("{e}"));
        let s = format!("{plan}");
        assert!(s.contains("N=256"));
        assert!(s.contains("M=32"));
        assert!(s.contains("FLOP savings"));
    }

    #[test]
    fn display_stage_format() {
        let stage = PrunedStage {
            stage_index: 2,
            radix: 2,
            active_butterflies: 50,
            total_butterflies: 128,
            can_skip_entirely: false,
        };
        let s = format!("{stage}");
        assert!(s.contains("Stage 2"));
        assert!(s.contains("50/128"));
    }

    // -- PTX generation tests -----------------------------------------------

    #[test]
    fn ptx_generation_single_stage() {
        let cfg = test_config(256, 32, 256);
        let plan = plan_pruned_fft(&cfg).unwrap_or_else(|e| panic!("{e}"));
        let stage = &plan.pruned_stages[0];
        let result = generate_pruned_butterfly_ptx(&cfg, stage);
        assert!(result.is_ok(), "PTX gen failed: {result:?}");
        let ptx = result.unwrap_or_else(|e| panic!("{e}"));
        assert!(ptx.contains(".entry pruned_butterfly_f32_n256_s0_"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn ptx_generation_f64() {
        let cfg = PrunedFftConfig {
            fft_size: 128,
            input_nonzero_count: 16,
            output_needed_count: 128,
            direction: FftDirection::Forward,
            precision: FftPrecision::Double,
            sm_version: SmVersion::Sm90,
        };
        let plan = plan_pruned_fft(&cfg).unwrap_or_else(|e| panic!("{e}"));
        let stage = &plan.pruned_stages[0];
        let result = generate_pruned_butterfly_ptx(&cfg, stage);
        assert!(result.is_ok(), "PTX gen failed: {result:?}");
        let ptx = result.unwrap_or_else(|e| panic!("{e}"));
        assert!(ptx.contains("pruned_butterfly_f64_n128_s0_"));
        assert!(ptx.contains(".target sm_90"));
        assert!(!ptx.contains("cos/sin placeholder"));
        assert!(ptx.contains("cvt.rni.s64.f64"));
        assert!(ptx.contains("cvt.rn.f64.s64"));
        assert!(ptx.contains("fma.rn.f64"));
    }

    // -- Edge case tests ----------------------------------------------------

    #[test]
    fn smallest_valid_fft() {
        let cfg = test_config(2, 1, 2);
        let plan = plan_pruned_fft(&cfg);
        assert!(plan.is_ok());
        let plan = plan.unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.pruned_stages.len(), 1);
    }

    #[test]
    fn output_pruning_savings() {
        let cfg = test_config(1024, 1024, 64);
        let plan = plan_pruned_fft(&cfg).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            plan.flop_savings_ratio > 0.0,
            "output pruning should yield savings",
        );
    }

    // -----------------------------------------------------------------------
    // CPU reference DFT/IDFT for round-trip accuracy tests
    // -----------------------------------------------------------------------

    /// CPU reference DFT for round-trip testing (f32 precision).
    fn dft_cpu_f32(x: &[f32]) -> Vec<(f32, f32)> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let (re, im) = (0..n).fold((0.0_f32, 0.0_f32), |(re, im), j| {
                    let angle = -2.0 * std::f32::consts::PI * (k * j) as f32 / n as f32;
                    (re + x[j] * angle.cos(), im + x[j] * angle.sin())
                });
                (re, im)
            })
            .collect()
    }

    /// CPU reference IDFT for round-trip testing (f32 precision).
    fn idft_cpu_f32(spectrum: &[(f32, f32)]) -> Vec<f32> {
        let n = spectrum.len();
        (0..n)
            .map(|j| {
                let sum = (0..n).fold(0.0_f32, |acc, k| {
                    let angle = 2.0 * std::f32::consts::PI * (k * j) as f32 / n as f32;
                    acc + spectrum[k].0 * angle.cos() - spectrum[k].1 * angle.sin()
                });
                sum / n as f32
            })
            .collect()
    }

    /// CPU reference DFT for round-trip testing (f64 precision).
    fn dft_cpu_f64(x: &[f64]) -> Vec<(f64, f64)> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let (re, im) = (0..n).fold((0.0_f64, 0.0_f64), |(re, im), j| {
                    let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                    (re + x[j] * angle.cos(), im + x[j] * angle.sin())
                });
                (re, im)
            })
            .collect()
    }

    /// CPU reference IDFT for round-trip testing (f64 precision).
    fn idft_cpu_f64(spectrum: &[(f64, f64)]) -> Vec<f64> {
        let n = spectrum.len();
        (0..n)
            .map(|j| {
                let sum = (0..n).fold(0.0_f64, |acc, k| {
                    let angle = 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                    acc + spectrum[k].0 * angle.cos() - spectrum[k].1 * angle.sin()
                });
                sum / n as f64
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Round-trip accuracy tests (FP32 and FP64)
    // -----------------------------------------------------------------------

    #[test]
    fn fft_round_trip_fp32_n16() {
        // DFT then IDFT on a 16-element f32 signal → matches original within N*1e-6.
        let n = 16_usize;
        let tol = n as f32 * 1e-6_f32;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin() + 0.5).collect();

        let spectrum = dft_cpu_f32(&signal);
        let recovered = idft_cpu_f32(&spectrum);

        assert_eq!(recovered.len(), n);
        for i in 0..n {
            let diff = (recovered[i] - signal[i]).abs();
            assert!(
                diff < tol,
                "round-trip[{i}]: recovered={}, original={}, diff={diff} > tol={tol}",
                recovered[i],
                signal[i]
            );
        }
    }

    #[test]
    fn fft_round_trip_fp32_n64() {
        // DFT then IDFT on a 64-element f32 signal → matches original within N*1e-6.
        let n = 64_usize;
        let tol = n as f32 * 1e-6_f32;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();

        let spectrum = dft_cpu_f32(&signal);
        let recovered = idft_cpu_f32(&spectrum);

        assert_eq!(recovered.len(), n);
        for i in 0..n {
            let diff = (recovered[i] - signal[i]).abs();
            assert!(
                diff < tol,
                "round-trip[{i}]: recovered={}, original={}, diff={diff} > tol={tol}",
                recovered[i],
                signal[i]
            );
        }
    }

    #[test]
    fn fft_round_trip_delta_function() {
        // x[0]=1, rest=0 → DFT = all (1, 0) (flat spectrum), IDFT recovers original.
        let n = 16_usize;
        let mut signal = vec![0.0_f32; n];
        signal[0] = 1.0;

        let spectrum = dft_cpu_f32(&signal);

        // All spectrum entries should be (1, 0).
        let tol_spec = 1e-6_f32;
        for (k, &(re, im)) in spectrum.iter().enumerate() {
            assert!(
                (re - 1.0).abs() < tol_spec,
                "spectrum[{k}].re = {re} \u{2260} 1.0"
            );
            assert!(im.abs() < tol_spec, "spectrum[{k}].im = {im} \u{2260} 0.0");
        }

        // IDFT should recover the delta.
        let recovered = idft_cpu_f32(&spectrum);
        let tol_rec = n as f32 * 1e-6_f32;
        assert!(
            (recovered[0] - 1.0).abs() < tol_rec,
            "recovered[0] = {}",
            recovered[0]
        );
        for (i, &r) in recovered.iter().enumerate().skip(1) {
            assert!(r.abs() < tol_rec, "recovered[{i}] = {} should be ~0", r);
        }
    }

    #[test]
    fn fft_round_trip_constant_signal() {
        // x[i]=1.0 → DFT[0] = N (DC bin), all other bins = 0, IDFT recovers 1.0.
        let n = 16_usize;
        let signal = vec![1.0_f32; n];

        let spectrum = dft_cpu_f32(&signal);

        // DC bin should be N.
        let tol = 1e-5_f32;
        assert!(
            (spectrum[0].0 - n as f32).abs() < tol,
            "DFT[0].re = {} \u{2260} {n}",
            spectrum[0].0
        );
        assert!(
            spectrum[0].1.abs() < tol,
            "DFT[0].im = {} \u{2260} 0",
            spectrum[0].1
        );

        // All non-DC bins should be ~0.
        for (k, &(re, im)) in spectrum.iter().enumerate().skip(1) {
            let mag = (re * re + im * im).sqrt();
            assert!(
                mag < tol,
                "DFT[{k}] magnitude = {mag} should be ~0 for constant signal"
            );
        }

        // IDFT recovers 1.0.
        let recovered = idft_cpu_f32(&spectrum);
        let tol_rec = n as f32 * 1e-6_f32;
        for (i, &r) in recovered.iter().enumerate() {
            assert!(
                (r - 1.0).abs() < tol_rec,
                "recovered[{i}] = {r} \u{2260} 1.0"
            );
        }
    }

    #[test]
    fn fft_fp64_round_trip_n32() {
        // DFT then IDFT on a 32-element f64 signal → matches original within N*2.2e-15.
        let n = 32_usize;
        let tol = n as f64 * 2.2e-15_f64;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();

        let spectrum = dft_cpu_f64(&signal);
        let recovered = idft_cpu_f64(&spectrum);

        assert_eq!(recovered.len(), n);
        for i in 0..n {
            let diff = (recovered[i] - signal[i]).abs();
            assert!(
                diff < tol,
                "f64 round-trip[{i}]: recovered={}, original={}, diff={diff} > tol={tol}",
                recovered[i],
                signal[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // TMA FFT configuration tests (large N > 64K)
    // -----------------------------------------------------------------------

    #[test]
    fn fft_tma_config_for_large_n() {
        use crate::out_of_core::LargeFftConfig;

        // N = 2^17 = 131072 > 65536 → tma_loading should be true.
        let n: usize = 131_072;
        let gpu_mem_bytes: usize = 512 * 1024 * 1024; // 512 MiB
        let cfg = LargeFftConfig::new(n, gpu_mem_bytes);

        assert!(cfg.tma_loading, "N={n} > 65536 should enable TMA loading");
        assert_eq!(cfg.n, n);
        assert_eq!(cfg.chunk_size, gpu_mem_bytes / 8);
    }

    #[test]
    fn fft_scalar_config_for_small_n() {
        use crate::out_of_core::LargeFftConfig;

        // N = 1024 <= 65536 → tma_loading should be false.
        let n: usize = 1024;
        let gpu_mem_bytes: usize = 16 * 1024 * 1024; // 16 MiB
        let cfg = LargeFftConfig::new(n, gpu_mem_bytes);

        assert!(
            !cfg.tma_loading,
            "N={n} <= 65536 should NOT enable TMA loading"
        );
        assert_eq!(cfg.n, n);
    }

    #[test]
    fn fft_tma_threshold_boundary() {
        use crate::out_of_core::LargeFftConfig;

        // Exactly 65536 → not TMA.
        let cfg_at = LargeFftConfig::new(65_536, 1024 * 1024);
        assert!(!cfg_at.tma_loading, "N=65536 is NOT > 65536, so no TMA");

        // One above → TMA.
        let cfg_above = LargeFftConfig::new(65_537, 1024 * 1024);
        assert!(cfg_above.tma_loading, "N=65537 > 65536 → TMA enabled");
    }
}
