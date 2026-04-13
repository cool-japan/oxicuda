//! Kernel similarity detection for configuration transfer.
//!
//! When a new kernel needs autotuning, it is often structurally similar to
//! kernels that have already been tuned.  This module provides a way to
//! describe a kernel's structural properties ([`KernelSignature`]), compute
//! a similarity score between two signatures ([`compute_similarity`]), and
//! transfer a tuning configuration from a similar kernel to the new one
//! ([`ConfigAdapter`]).
//!
//! # Workflow
//!
//! 1. Register known kernels and their best configs in a
//!    [`KernelSimilarityIndex`].
//! 2. When a new kernel appears, build its [`KernelSignature`] and call
//!    [`KernelSimilarityIndex::find_similar`] to retrieve candidate matches.
//! 3. Use [`ConfigAdapter::adapt`] to transform the best matching config
//!    to suit the new kernel.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::kernel_similarity::*;
//! use oxicuda_autotune::Config;
//!
//! let sig_a = KernelSignature::new("sgemm")
//!     .with_param_count(3)
//!     .with_arithmetic_intensity(ArithmeticIntensity::High)
//!     .with_access_pattern(AccessPattern::Coalesced)
//!     .with_element_bytes(4);
//!
//! let sig_b = KernelSignature::new("hgemm")
//!     .with_param_count(3)
//!     .with_arithmetic_intensity(ArithmeticIntensity::High)
//!     .with_access_pattern(AccessPattern::Coalesced)
//!     .with_element_bytes(2);
//!
//! let score = compute_similarity(&sig_a, &sig_b);
//! assert!(score.value() > 0.5);
//! ```

#![allow(clippy::module_inception)]

use serde::{Deserialize, Serialize};

use crate::config::Config;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Classification of a kernel's compute-to-memory ratio.
///
/// This drives tile-size scaling when transferring configurations between
/// kernels with different arithmetic intensities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArithmeticIntensity {
    /// Memory-bound: more loads/stores than arithmetic (e.g. elementwise).
    Low,
    /// Balanced: roughly equal compute and memory traffic (e.g. small GEMM).
    Medium,
    /// Compute-bound: dominated by arithmetic (e.g. large GEMM, convolution).
    High,
}

impl ArithmeticIntensity {
    /// Returns a numeric rank for distance calculations.
    fn rank(self) -> u32 {
        match self {
            Self::Low => 0,
            Self::Medium => 1,
            Self::High => 2,
        }
    }

    /// Distance between two intensity classes, normalised to [0, 1].
    fn distance(self, other: Self) -> f64 {
        let diff = (self.rank() as f64 - other.rank() as f64).abs();
        diff / 2.0
    }

    /// Scale factor for tile sizes when adapting from one intensity to another.
    ///
    /// Higher intensity kernels benefit from larger tiles (more reuse),
    /// while lower intensity kernels prefer smaller tiles (less shared
    /// memory pressure).
    fn tile_scale_factor(self, target: Self) -> f64 {
        match (self, target) {
            (Self::Low, Self::High) => 2.0,
            (Self::Low, Self::Medium) => 1.5,
            (Self::Medium, Self::High) => 1.5,
            (Self::High, Self::Low) => 0.5,
            (Self::High, Self::Medium) => 0.75,
            (Self::Medium, Self::Low) => 0.75,
            _ => 1.0,
        }
    }
}

/// Memory access pattern of the kernel's primary data reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Threads in a warp access consecutive addresses (ideal for GPU).
    Coalesced,
    /// Threads access addresses with a constant stride > 1.
    Strided,
    /// Threads access non-contiguous, data-dependent addresses.
    Random,
    /// Threads access small contiguous blocks with gaps between blocks.
    Blocked,
}

impl AccessPattern {
    /// Returns a numeric rank for distance calculations.
    fn rank(self) -> u32 {
        match self {
            Self::Coalesced => 0,
            Self::Blocked => 1,
            Self::Strided => 2,
            Self::Random => 3,
        }
    }

    /// Distance between two access patterns, normalised to [0, 1].
    fn distance(self, other: Self) -> f64 {
        let diff = (self.rank() as f64 - other.rank() as f64).abs();
        diff / 3.0
    }

    /// Suggested stage adjustment when adapting from one pattern to another.
    ///
    /// More irregular access patterns benefit from deeper pipelines to
    /// hide latency, so we increase stages.  Coalesced patterns need
    /// fewer stages.
    fn stage_delta(self, target: Self) -> i32 {
        target.rank() as i32 - self.rank() as i32
    }
}

/// Type of reduction operation (if any) performed by the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReductionType {
    /// Summation reduction.
    Sum,
    /// Maximum reduction.
    Max,
    /// Minimum reduction.
    Min,
    /// Index of maximum element.
    ArgMax,
    /// Index of minimum element.
    ArgMin,
    /// Product reduction.
    Product,
}

impl ReductionType {
    /// Distance between two reduction types, normalised to [0, 1].
    ///
    /// Sum/Product are close (both associative scalars), Max/Min are close,
    /// ArgMax/ArgMin are close.  Cross-group distance is larger.
    fn distance(self, other: Self) -> f64 {
        if self == other {
            return 0.0;
        }
        match (self, other) {
            (Self::Sum, Self::Product) | (Self::Product, Self::Sum) => 0.2,
            (Self::Max, Self::Min) | (Self::Min, Self::Max) => 0.1,
            (Self::ArgMax, Self::ArgMin) | (Self::ArgMin, Self::ArgMax) => 0.1,
            (Self::Max, Self::ArgMax)
            | (Self::ArgMax, Self::Max)
            | (Self::Min, Self::ArgMin)
            | (Self::ArgMin, Self::Min) => 0.3,
            _ => 0.6,
        }
    }
}

// ---------------------------------------------------------------------------
// KernelSignature
// ---------------------------------------------------------------------------

/// Structural description of a GPU kernel for similarity matching.
///
/// Captures the key properties that influence optimal tuning parameters
/// without requiring the actual kernel source code.  Two kernels with
/// similar signatures are likely to benefit from similar configurations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelSignature {
    /// Kernel name (e.g. `"sgemm"`, `"conv2d_fprop"`).
    pub name: String,
    /// Number of kernel parameters (pointers + scalars).
    pub param_count: usize,
    /// Compute-to-memory ratio class.
    pub arithmetic_intensity: ArithmeticIntensity,
    /// Primary memory access pattern.
    pub access_pattern: AccessPattern,
    /// Reduction operation, if the kernel performs one.
    pub reduction_type: Option<ReductionType>,
    /// Size of each element in bytes (e.g. 4 for f32, 2 for f16).
    pub element_bytes: u32,
}

impl KernelSignature {
    /// Creates a new signature with default properties.
    ///
    /// Defaults: `param_count = 0`, `Medium` intensity, `Coalesced`
    /// access, no reduction, 4-byte elements.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_count: 0,
            arithmetic_intensity: ArithmeticIntensity::Medium,
            access_pattern: AccessPattern::Coalesced,
            reduction_type: None,
            element_bytes: 4,
        }
    }

    /// Sets the parameter count.
    #[must_use]
    pub fn with_param_count(mut self, count: usize) -> Self {
        self.param_count = count;
        self
    }

    /// Sets the arithmetic intensity class.
    #[must_use]
    pub fn with_arithmetic_intensity(mut self, intensity: ArithmeticIntensity) -> Self {
        self.arithmetic_intensity = intensity;
        self
    }

    /// Sets the memory access pattern.
    #[must_use]
    pub fn with_access_pattern(mut self, pattern: AccessPattern) -> Self {
        self.access_pattern = pattern;
        self
    }

    /// Sets the reduction type.
    #[must_use]
    pub fn with_reduction_type(mut self, reduction: ReductionType) -> Self {
        self.reduction_type = Some(reduction);
        self
    }

    /// Sets the element size in bytes.
    #[must_use]
    pub fn with_element_bytes(mut self, bytes: u32) -> Self {
        self.element_bytes = bytes;
        self
    }
}

// ---------------------------------------------------------------------------
// SimilarityScore
// ---------------------------------------------------------------------------

/// A similarity score in the range [0.0, 1.0].
///
/// 1.0 means identical structural properties; 0.0 means completely
/// dissimilar.  Scores are used to rank candidate kernels for
/// configuration transfer.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SimilarityScore(f64);

impl SimilarityScore {
    /// Creates a new score, clamping to [0.0, 1.0].
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Returns the raw f64 value.
    #[must_use]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl std::fmt::Display for SimilarityScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

// ---------------------------------------------------------------------------
// compute_similarity
// ---------------------------------------------------------------------------

/// Feature weights for the similarity computation.
const WEIGHT_INTENSITY: f64 = 0.30;
const WEIGHT_ACCESS: f64 = 0.25;
const WEIGHT_REDUCTION: f64 = 0.15;
const WEIGHT_ELEMENT_BYTES: f64 = 0.15;
const WEIGHT_PARAM_COUNT: f64 = 0.15;

/// Computes a weighted similarity score between two kernel signatures.
///
/// Each structural feature contributes a component score:
///
/// | Feature              | Weight |
/// |----------------------|--------|
/// | Arithmetic intensity | 0.30   |
/// | Access pattern       | 0.25   |
/// | Reduction type       | 0.15   |
/// | Element bytes        | 0.15   |
/// | Parameter count      | 0.15   |
///
/// The final score is `1.0 - weighted_distance`, clamped to [0, 1].
#[must_use]
pub fn compute_similarity(a: &KernelSignature, b: &KernelSignature) -> SimilarityScore {
    let intensity_dist = a.arithmetic_intensity.distance(b.arithmetic_intensity);
    let access_dist = a.access_pattern.distance(b.access_pattern);

    let reduction_dist = match (a.reduction_type, b.reduction_type) {
        (Some(ra), Some(rb)) => ra.distance(rb),
        (None, None) => 0.0,
        _ => 0.8, // one has reduction, the other doesn't
    };

    let element_dist = if a.element_bytes == b.element_bytes {
        0.0
    } else {
        let ratio = a.element_bytes.max(b.element_bytes) as f64
            / a.element_bytes.min(b.element_bytes).max(1) as f64;
        // ratio is >= 1.0; map it to [0, 1] via 1 - 1/ratio
        1.0 - 1.0 / ratio
    };

    let param_dist = {
        let max_p = a.param_count.max(b.param_count).max(1) as f64;
        let diff = (a.param_count as f64 - b.param_count as f64).abs();
        (diff / max_p).min(1.0)
    };

    let weighted_dist = WEIGHT_INTENSITY * intensity_dist
        + WEIGHT_ACCESS * access_dist
        + WEIGHT_REDUCTION * reduction_dist
        + WEIGHT_ELEMENT_BYTES * element_dist
        + WEIGHT_PARAM_COUNT * param_dist;

    SimilarityScore::new(1.0 - weighted_dist)
}

// ---------------------------------------------------------------------------
// SimilarityMatch
// ---------------------------------------------------------------------------

/// A match result from the similarity index, pairing a kernel signature
/// with its score and suggested configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMatch {
    /// The matching kernel's signature.
    pub signature: KernelSignature,
    /// How similar this kernel is to the query (higher is better).
    pub score: SimilarityScore,
    /// The best known configuration for the matching kernel.
    pub suggested_config: Config,
}

// ---------------------------------------------------------------------------
// KernelSimilarityIndex
// ---------------------------------------------------------------------------

/// In-memory index of known kernel signatures and their best configs.
///
/// Supports registration, similarity search, and configuration transfer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KernelSimilarityIndex {
    /// Registered (signature, best_config) pairs.
    entries: Vec<(KernelSignature, Config)>,
}

impl KernelSimilarityIndex {
    /// Creates an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a kernel signature with its best known configuration.
    ///
    /// If a kernel with the same name is already registered, its entry
    /// is replaced.
    pub fn register(&mut self, signature: KernelSignature, best_config: Config) {
        // Replace if same name already exists.
        if let Some(pos) = self
            .entries
            .iter()
            .position(|(s, _)| s.name == signature.name)
        {
            self.entries[pos] = (signature, best_config);
        } else {
            self.entries.push((signature, best_config));
        }
    }

    /// Finds all registered kernels whose similarity to `query` meets
    /// or exceeds `threshold`.
    ///
    /// Results are sorted by descending similarity score.
    #[must_use]
    pub fn find_similar(&self, query: &KernelSignature, threshold: f64) -> Vec<SimilarityMatch> {
        let mut matches: Vec<SimilarityMatch> = self
            .entries
            .iter()
            .filter_map(|(sig, cfg)| {
                let score = compute_similarity(query, sig);
                if score.value() >= threshold {
                    Some(SimilarityMatch {
                        signature: sig.clone(),
                        score,
                        suggested_config: cfg.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort descending by score.
        matches.sort_by(|a, b| {
            b.score
                .value()
                .partial_cmp(&a.score.value())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }

    /// Transfers the best configuration from the most similar registered
    /// kernel to the target, adapting it for the target's properties.
    ///
    /// Returns `None` if no registered kernel has a similarity score
    /// above the internal minimum threshold (0.3).
    #[must_use]
    pub fn transfer_config(&self, from: &KernelSignature, to: &KernelSignature) -> Option<Config> {
        // Find the best match for `from` in the index.
        let matches = self.find_similar(from, 0.3);
        let best = matches.first()?;
        let adapter = ConfigAdapter::new(from, to);
        Some(adapter.adapt(&best.suggested_config))
    }

    /// Returns the number of registered kernels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no kernels are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns an iterator over all registered (signature, config) pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(KernelSignature, Config)> {
        self.entries.iter()
    }
}

// ---------------------------------------------------------------------------
// ConfigAdapter
// ---------------------------------------------------------------------------

/// Adapts a tuning configuration from one kernel to another based on
/// their structural differences.
///
/// The adapter applies three transformations:
///
/// 1. **Tile scaling** — scales tile dimensions (`tile_m`, `tile_n`,
///    `tile_k`) based on the arithmetic intensity difference.  Compute-
///    bound targets get larger tiles; memory-bound targets get smaller.
///
/// 2. **Stage adjustment** — adjusts the pipeline depth (`stages`)
///    based on the access pattern difference.  More irregular patterns
///    get deeper pipelines to hide latency.
///
/// 3. **Tensor core preservation** — keeps the tensor core setting if
///    the source and target have matching element sizes (both must be
///    TC-compatible: 2 or 1 byte).  Otherwise tensor cores are disabled.
pub struct ConfigAdapter<'a> {
    source: &'a KernelSignature,
    target: &'a KernelSignature,
}

impl<'a> ConfigAdapter<'a> {
    /// Creates a new adapter that will transform configs from `source`
    /// to `target`.
    #[must_use]
    pub fn new(source: &'a KernelSignature, target: &'a KernelSignature) -> Self {
        Self { source, target }
    }

    /// Adapts the given configuration for the target kernel.
    ///
    /// Applies tile scaling, stage adjustment, and tensor core
    /// preservation logic.
    #[must_use]
    pub fn adapt(&self, config: &Config) -> Config {
        let mut adapted = config.clone();

        // 1. Tile scaling based on arithmetic intensity difference.
        let scale = self
            .source
            .arithmetic_intensity
            .tile_scale_factor(self.target.arithmetic_intensity);

        adapted.tile_m = Self::scale_tile(config.tile_m, scale);
        adapted.tile_n = Self::scale_tile(config.tile_n, scale);
        adapted.tile_k = Self::scale_tile(config.tile_k, scale);

        // Also scale warp tiles proportionally.
        adapted.warp_m = Self::scale_tile(config.warp_m, scale);
        adapted.warp_n = Self::scale_tile(config.warp_n, scale);

        // 2. Stage adjustment based on access pattern difference.
        let delta = self
            .source
            .access_pattern
            .stage_delta(self.target.access_pattern);
        let new_stages = (config.stages as i32 + delta).max(1) as u32;
        adapted.stages = new_stages.min(8); // cap at 8 stages

        // 3. Tensor core preservation.
        if config.use_tensor_core {
            // Tensor cores require matching element sizes that are
            // TC-compatible (fp16 = 2 bytes, fp8 = 1 byte, bf16 = 2 bytes).
            let source_tc_ok = self.source.element_bytes <= 2;
            let target_tc_ok = self.target.element_bytes <= 2;
            adapted.use_tensor_core = source_tc_ok
                && target_tc_ok
                && self.source.element_bytes == self.target.element_bytes;
        }

        adapted
    }

    /// Scales a tile dimension by a factor, rounding to the nearest
    /// power of two (common requirement for GPU tile sizes).
    fn scale_tile(tile: u32, scale: f64) -> u32 {
        let scaled = (tile as f64 * scale).round() as u32;
        let scaled = scaled.max(1);
        // Round to nearest power of two.
        let log2 = (scaled as f64).log2().round() as u32;
        1u32.checked_shl(log2).unwrap_or(tile)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sgemm_sig() -> KernelSignature {
        KernelSignature::new("sgemm")
            .with_param_count(3)
            .with_arithmetic_intensity(ArithmeticIntensity::High)
            .with_access_pattern(AccessPattern::Coalesced)
            .with_element_bytes(4)
    }

    fn hgemm_sig() -> KernelSignature {
        KernelSignature::new("hgemm")
            .with_param_count(3)
            .with_arithmetic_intensity(ArithmeticIntensity::High)
            .with_access_pattern(AccessPattern::Coalesced)
            .with_element_bytes(2)
    }

    fn reduction_sig() -> KernelSignature {
        KernelSignature::new("reduce_sum")
            .with_param_count(2)
            .with_arithmetic_intensity(ArithmeticIntensity::Low)
            .with_access_pattern(AccessPattern::Coalesced)
            .with_reduction_type(ReductionType::Sum)
            .with_element_bytes(4)
    }

    fn scatter_sig() -> KernelSignature {
        KernelSignature::new("scatter_add")
            .with_param_count(4)
            .with_arithmetic_intensity(ArithmeticIntensity::Low)
            .with_access_pattern(AccessPattern::Random)
            .with_element_bytes(4)
    }

    // -- SimilarityScore ---------------------------------------------------

    #[test]
    fn similarity_score_clamping() {
        assert!((SimilarityScore::new(1.5).value() - 1.0).abs() < f64::EPSILON);
        assert!((SimilarityScore::new(-0.5).value()).abs() < f64::EPSILON);
        assert!((SimilarityScore::new(0.7).value() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn similarity_score_display() {
        let s = SimilarityScore::new(0.8765);
        assert_eq!(format!("{s}"), "0.8765");
    }

    // -- compute_similarity ------------------------------------------------

    #[test]
    fn identical_signatures_score_one() {
        let sig = sgemm_sig();
        let score = compute_similarity(&sig, &sig);
        assert!(
            (score.value() - 1.0).abs() < 1e-9,
            "identical signatures should score 1.0, got {}",
            score.value()
        );
    }

    #[test]
    fn sgemm_vs_hgemm_high_similarity() {
        let score = compute_similarity(&sgemm_sig(), &hgemm_sig());
        assert!(
            score.value() > 0.7,
            "sgemm vs hgemm should be highly similar, got {}",
            score.value()
        );
    }

    #[test]
    fn gemm_vs_reduction_low_similarity() {
        let score = compute_similarity(&sgemm_sig(), &reduction_sig());
        assert!(
            score.value() < 0.6,
            "gemm vs reduction should be dissimilar, got {}",
            score.value()
        );
    }

    #[test]
    fn gemm_vs_scatter_low_similarity() {
        let score = compute_similarity(&sgemm_sig(), &scatter_sig());
        assert!(
            score.value() < 0.5,
            "gemm vs scatter should be very dissimilar, got {}",
            score.value()
        );
    }

    #[test]
    fn similarity_is_symmetric() {
        let a = sgemm_sig();
        let b = reduction_sig();
        let ab = compute_similarity(&a, &b);
        let ba = compute_similarity(&b, &a);
        assert!(
            (ab.value() - ba.value()).abs() < 1e-9,
            "similarity should be symmetric"
        );
    }

    // -- KernelSimilarityIndex ---------------------------------------------

    #[test]
    fn index_register_and_find() {
        let mut index = KernelSimilarityIndex::new();
        assert!(index.is_empty());

        index.register(sgemm_sig(), Config::new().with_tile_m(128));
        index.register(hgemm_sig(), Config::new().with_tile_m(64));
        assert_eq!(index.len(), 2);

        let matches = index.find_similar(&sgemm_sig(), 0.5);
        assert!(
            !matches.is_empty(),
            "should find at least one match for sgemm"
        );
        // The first match should be sgemm itself (score 1.0).
        assert!(
            (matches[0].score.value() - 1.0).abs() < 1e-9,
            "best match should be exact"
        );
    }

    #[test]
    fn index_replace_on_duplicate_name() {
        let mut index = KernelSimilarityIndex::new();
        index.register(sgemm_sig(), Config::new().with_tile_m(128));
        index.register(sgemm_sig(), Config::new().with_tile_m(64));
        assert_eq!(index.len(), 1);

        let matches = index.find_similar(&sgemm_sig(), 0.0);
        assert_eq!(matches[0].suggested_config.tile_m, 64);
    }

    #[test]
    fn index_threshold_filters() {
        let mut index = KernelSimilarityIndex::new();
        index.register(sgemm_sig(), Config::new());
        index.register(scatter_sig(), Config::new());

        // High threshold should exclude the dissimilar scatter kernel.
        let matches = index.find_similar(&sgemm_sig(), 0.9);
        assert!(
            matches.iter().all(|m| m.signature.name != "scatter_add"),
            "scatter should be below 0.9 threshold"
        );
    }

    #[test]
    fn transfer_config_returns_none_when_empty() {
        let index = KernelSimilarityIndex::new();
        let result = index.transfer_config(&sgemm_sig(), &hgemm_sig());
        assert!(result.is_none());
    }

    #[test]
    fn transfer_config_adapts_tiles() {
        let mut index = KernelSimilarityIndex::new();
        let base_cfg = Config::new()
            .with_tile_m(128)
            .with_tile_n(128)
            .with_tile_k(32)
            .with_stages(2)
            .with_use_tensor_core(false);

        // Register a high-intensity kernel.
        index.register(sgemm_sig(), base_cfg);

        // Transfer to a low-intensity kernel: tiles should shrink.
        let low_sig = KernelSignature::new("elementwise")
            .with_param_count(3)
            .with_arithmetic_intensity(ArithmeticIntensity::Low)
            .with_access_pattern(AccessPattern::Coalesced)
            .with_element_bytes(4);

        let adapted = index.transfer_config(&sgemm_sig(), &low_sig);
        assert!(adapted.is_some());
        let adapted = adapted.expect("should have adapted config");
        // Tiles should be smaller (scale factor 0.5 from High->Low).
        assert!(
            adapted.tile_m < 128,
            "tile_m should shrink for low intensity, got {}",
            adapted.tile_m
        );
    }

    // -- ConfigAdapter -----------------------------------------------------

    #[test]
    fn adapter_preserves_tensor_core_for_matching_elements() {
        let source = hgemm_sig(); // element_bytes = 2
        let target = KernelSignature::new("bf16_gemm")
            .with_param_count(3)
            .with_arithmetic_intensity(ArithmeticIntensity::High)
            .with_access_pattern(AccessPattern::Coalesced)
            .with_element_bytes(2);

        let cfg = Config::new().with_use_tensor_core(true);
        let adapter = ConfigAdapter::new(&source, &target);
        let adapted = adapter.adapt(&cfg);
        assert!(
            adapted.use_tensor_core,
            "tensor core should be preserved for matching 2-byte elements"
        );
    }

    #[test]
    fn adapter_disables_tensor_core_for_mismatched_elements() {
        let source = hgemm_sig(); // element_bytes = 2
        let target = sgemm_sig(); // element_bytes = 4

        let cfg = Config::new().with_use_tensor_core(true);
        let adapter = ConfigAdapter::new(&source, &target);
        let adapted = adapter.adapt(&cfg);
        assert!(
            !adapted.use_tensor_core,
            "tensor core should be disabled for f32 target"
        );
    }

    #[test]
    fn adapter_increases_stages_for_irregular_access() {
        let source = KernelSignature::new("src")
            .with_access_pattern(AccessPattern::Coalesced)
            .with_arithmetic_intensity(ArithmeticIntensity::Medium);

        let target = KernelSignature::new("tgt")
            .with_access_pattern(AccessPattern::Strided)
            .with_arithmetic_intensity(ArithmeticIntensity::Medium);

        let cfg = Config::new().with_stages(2);
        let adapter = ConfigAdapter::new(&source, &target);
        let adapted = adapter.adapt(&cfg);
        assert!(
            adapted.stages > 2,
            "stages should increase for strided access, got {}",
            adapted.stages
        );
    }

    #[test]
    fn scale_tile_rounds_to_power_of_two() {
        // 128 * 0.5 = 64 (already pow2)
        assert_eq!(ConfigAdapter::scale_tile(128, 0.5), 64);
        // 128 * 1.5 = 192 -> nearest pow2 = 256
        assert_eq!(ConfigAdapter::scale_tile(128, 1.5), 256);
        // 128 * 2.0 = 256 (already pow2)
        assert_eq!(ConfigAdapter::scale_tile(128, 2.0), 256);
        // 64 * 0.75 = 48 -> nearest pow2 = 32 or 64 (log2(48)=5.58 -> round=6 -> 64)
        assert_eq!(ConfigAdapter::scale_tile(64, 0.75), 64);
    }

    // -- Serde roundtrip ---------------------------------------------------

    #[test]
    fn serde_roundtrip_signature() {
        let sig = sgemm_sig();
        let json = serde_json::to_string(&sig).expect("serialize");
        let restored: KernelSignature = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(sig, restored);
    }

    #[test]
    fn serde_roundtrip_index() {
        let mut index = KernelSimilarityIndex::new();
        index.register(sgemm_sig(), Config::new().with_tile_m(128));
        index.register(hgemm_sig(), Config::new().with_tile_m(64));

        let json = serde_json::to_string(&index).expect("serialize");
        let restored: KernelSimilarityIndex = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.len(), 2);
    }
}
