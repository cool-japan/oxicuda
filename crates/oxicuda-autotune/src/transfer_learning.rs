//! Transfer learning between GPU architectures.
//!
//! When tuning results exist for one GPU architecture (e.g. sm_80 / A100),
//! this module can *warm-start* tuning on a different architecture
//! (e.g. sm_90 / H100) by transforming configurations according to the
//! hardware differences — scaling tile sizes, adjusting pipeline depth,
//! enabling Tensor Cores, etc.
//!
//! # Workflow
//!
//! 1. Build [`ArchitectureProfile`]s for the source and target GPUs.
//! 2. Call [`TransferStrategy::infer`] to auto-derive transfer rules.
//! 3. Use [`ConfigTransformer::transform`] to adapt individual configs,
//!    or use [`TransferLearningEngine`] to batch-transfer an entire
//!    [`ResultDb`].

use crate::config::Config;
use crate::result_db::ResultDb;

// ---------------------------------------------------------------------------
// ArchitectureProfile
// ---------------------------------------------------------------------------

/// Hardware profile for a specific GPU compute capability.
///
/// Captures the key parameters that influence kernel tuning: memory
/// limits, compute throughput, Tensor Core generation, and SM count.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureProfile {
    /// SM version (e.g. 80 for Ampere A100, 90 for Hopper H100).
    pub sm_version: u32,
    /// Maximum dynamic shared memory per block (bytes).
    pub max_shared_memory: u32,
    /// Maximum registers per thread.
    pub max_registers: u32,
    /// Warp size (always 32 on current NVIDIA hardware).
    pub warp_size: u32,
    /// Whether the architecture has Tensor Core units.
    pub has_tensor_cores: bool,
    /// Tensor Core generation: 0 = none, 1 = Volta, 2 = Ampere, 3 = Hopper.
    pub tensor_core_generations: u32,
    /// Peak memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,
    /// Peak FP32 compute throughput in TFLOPS.
    pub compute_tflops: f64,
    /// GPU core clock in MHz.
    pub clock_mhz: u32,
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
}

impl ArchitectureProfile {
    /// NVIDIA A100 (sm_80, Ampere).
    #[must_use]
    pub fn sm80() -> Self {
        Self {
            sm_version: 80,
            max_shared_memory: 163_840, // 160 KiB
            max_registers: 255,
            warp_size: 32,
            has_tensor_cores: true,
            tensor_core_generations: 2,
            memory_bandwidth_gbps: 2039.0,
            compute_tflops: 19.5,
            clock_mhz: 1410,
            sm_count: 108,
        }
    }

    /// NVIDIA A40 / RTX 3080-class (sm_86, Ampere).
    #[must_use]
    pub fn sm86() -> Self {
        Self {
            sm_version: 86,
            max_shared_memory: 99_304, // ~97 KiB
            max_registers: 255,
            warp_size: 32,
            has_tensor_cores: true,
            tensor_core_generations: 2,
            memory_bandwidth_gbps: 696.0,
            compute_tflops: 29.8,
            clock_mhz: 1695,
            sm_count: 84,
        }
    }

    /// NVIDIA L40S / RTX 4090-class (sm_89, Ada Lovelace).
    #[must_use]
    pub fn sm89() -> Self {
        Self {
            sm_version: 89,
            max_shared_memory: 99_304,
            max_registers: 255,
            warp_size: 32,
            has_tensor_cores: true,
            tensor_core_generations: 2,
            memory_bandwidth_gbps: 864.0,
            compute_tflops: 82.6,
            clock_mhz: 2520,
            sm_count: 128,
        }
    }

    /// NVIDIA H100 (sm_90, Hopper).
    #[must_use]
    pub fn sm90() -> Self {
        Self {
            sm_version: 90,
            max_shared_memory: 228_352, // 223 KiB
            max_registers: 255,
            warp_size: 32,
            has_tensor_cores: true,
            tensor_core_generations: 3,
            memory_bandwidth_gbps: 3350.0,
            compute_tflops: 51.2,
            clock_mhz: 1830,
            sm_count: 132,
        }
    }
}

// ---------------------------------------------------------------------------
// ArchitectureSimilarity
// ---------------------------------------------------------------------------

/// Computes a similarity score between two GPU architecture profiles.
pub struct ArchitectureSimilarity;

impl ArchitectureSimilarity {
    /// Returns a similarity score in `[0.0, 1.0]`.
    ///
    /// Factors considered (equally weighted):
    /// - SM version distance (closer = more similar)
    /// - Memory bandwidth ratio
    /// - Compute throughput ratio
    /// - Tensor Core generation match
    /// - Shared memory ratio
    #[must_use]
    pub fn compute_similarity(source: &ArchitectureProfile, target: &ArchitectureProfile) -> f64 {
        // SM version distance: each major version step reduces similarity.
        let sm_diff = (source.sm_version as f64 - target.sm_version as f64).abs();
        let sm_sim = 1.0 / (1.0 + sm_diff / 10.0);

        // Bandwidth ratio (clamped to [0,1] via min/max ratio).
        let bw_sim = ratio_similarity(source.memory_bandwidth_gbps, target.memory_bandwidth_gbps);

        // Compute ratio.
        let compute_sim = ratio_similarity(source.compute_tflops, target.compute_tflops);

        // Tensor core generation match.
        let tc_diff =
            (source.tensor_core_generations as f64 - target.tensor_core_generations as f64).abs();
        let tc_sim = 1.0 / (1.0 + tc_diff);

        // Shared memory ratio.
        let smem_sim = ratio_similarity(
            f64::from(source.max_shared_memory),
            f64::from(target.max_shared_memory),
        );

        // Equal-weight average.
        (sm_sim + bw_sim + compute_sim + tc_sim + smem_sim) / 5.0
    }
}

/// Similarity of two positive values as `min(a,b) / max(a,b)`.
fn ratio_similarity(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }
    a.min(b) / a.max(b)
}

// ---------------------------------------------------------------------------
// ConfigTransferRule
// ---------------------------------------------------------------------------

/// A rule that describes how to adapt a configuration from one
/// architecture to another.
#[derive(Debug, Clone)]
pub enum ConfigTransferRule {
    /// Keep the configuration unchanged.
    KeepAsIs,
    /// Scale shared-memory-dependent parameters.
    ScaleSharedMemory {
        /// Multiplicative factor for smem-related values.
        factor: f64,
    },
    /// Scale tile dimensions.
    ScaleTiles {
        /// Factor for the M tile dimension.
        m_factor: f64,
        /// Factor for the N tile dimension.
        n_factor: f64,
        /// Factor for the K tile dimension.
        k_factor: f64,
    },
    /// Adjust pipeline depth by a signed delta.
    AdjustPipelineDepth {
        /// Amount to add (positive) or subtract (negative).
        delta: i32,
    },
    /// Enable Tensor Core usage on the target.
    EnableTensorCores,
    /// Disable Tensor Core usage on the target.
    DisableTensorCores,
    /// Apply multiple rules in sequence.
    Composite(Vec<ConfigTransferRule>),
}

// ---------------------------------------------------------------------------
// TransferStrategy
// ---------------------------------------------------------------------------

/// A strategy for transferring tuning results between architectures.
///
/// Contains the source/target profiles, their similarity score, and
/// the set of transformation rules to apply.
#[derive(Debug, Clone)]
pub struct TransferStrategy {
    /// Source GPU profile.
    pub source: ArchitectureProfile,
    /// Target GPU profile.
    pub target: ArchitectureProfile,
    /// Architecture similarity score (0.0–1.0).
    pub similarity: f64,
    /// Ordered rules to apply during config transfer.
    pub rules: Vec<ConfigTransferRule>,
}

impl TransferStrategy {
    /// Automatically infer a transfer strategy from architecture differences.
    ///
    /// Heuristics:
    /// - If source has no Tensor Cores but target does → `EnableTensorCores`.
    /// - If source has Tensor Cores but target does not → `DisableTensorCores`.
    /// - If compute ratio differs significantly → scale tiles proportionally.
    /// - If shared memory capacity differs → scale smem-related params.
    /// - If target is Hopper (sm_90) → increase pipeline depth.
    #[must_use]
    pub fn infer(source: ArchitectureProfile, target: ArchitectureProfile) -> Self {
        let similarity = ArchitectureSimilarity::compute_similarity(&source, &target);
        let mut rules = Vec::new();

        // Tensor core enablement.
        if !source.has_tensor_cores && target.has_tensor_cores {
            rules.push(ConfigTransferRule::EnableTensorCores);
        } else if source.has_tensor_cores && !target.has_tensor_cores {
            rules.push(ConfigTransferRule::DisableTensorCores);
        }

        // Compute scaling: if the ratio differs by >20%, scale tiles.
        let compute_ratio = target.compute_tflops / source.compute_tflops.max(1e-9);
        if (compute_ratio - 1.0).abs() > 0.2 {
            // Scale M and N proportionally to sqrt of compute ratio
            // (since work ~ M*N), K stays roughly the same.
            let mn_factor = compute_ratio.sqrt().clamp(0.5, 4.0);
            rules.push(ConfigTransferRule::ScaleTiles {
                m_factor: mn_factor,
                n_factor: mn_factor,
                k_factor: 1.0,
            });
        }

        // Shared memory scaling.
        let smem_ratio =
            f64::from(target.max_shared_memory) / f64::from(source.max_shared_memory).max(1.0);
        if (smem_ratio - 1.0).abs() > 0.2 {
            rules.push(ConfigTransferRule::ScaleSharedMemory { factor: smem_ratio });
        }

        // Pipeline depth: Hopper's TMA benefits from deeper pipelines.
        if target.tensor_core_generations >= 3 && source.tensor_core_generations < 3 {
            rules.push(ConfigTransferRule::AdjustPipelineDepth { delta: 1 });
        } else if target.tensor_core_generations < 3 && source.tensor_core_generations >= 3 {
            rules.push(ConfigTransferRule::AdjustPipelineDepth { delta: -1 });
        }

        // If no rules were generated, keep as-is.
        if rules.is_empty() {
            rules.push(ConfigTransferRule::KeepAsIs);
        }

        Self {
            source,
            target,
            similarity,
            rules,
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigTransformer
// ---------------------------------------------------------------------------

/// Transforms a [`Config`] according to a [`TransferStrategy`].
pub struct ConfigTransformer;

impl ConfigTransformer {
    /// Apply all rules in a strategy to produce a transferred config.
    ///
    /// The result is clamped to the target architecture's limits.
    #[must_use]
    pub fn transform(config: &Config, strategy: &TransferStrategy) -> Config {
        let mut result = config.clone();
        for rule in &strategy.rules {
            result = Self::apply_rule(&result, rule, strategy);
        }
        Self::clamp_to_target(&result, &strategy.target)
    }

    /// Apply a single rule to a configuration.
    #[must_use]
    #[allow(clippy::only_used_in_recursion)]
    pub fn apply_rule(
        config: &Config,
        rule: &ConfigTransferRule,
        strategy: &TransferStrategy,
    ) -> Config {
        let mut c = config.clone();
        match rule {
            ConfigTransferRule::KeepAsIs => {}
            ConfigTransferRule::ScaleSharedMemory { factor } => {
                // Scale stages (more smem → can afford more stages).
                let new_stages = (f64::from(c.stages) * factor).round() as u32;
                c.stages = new_stages.max(1);
            }
            ConfigTransferRule::ScaleTiles {
                m_factor,
                n_factor,
                k_factor,
            } => {
                c.tile_m = round_to_power_of_two((f64::from(c.tile_m) * m_factor).round() as u32);
                c.tile_n = round_to_power_of_two((f64::from(c.tile_n) * n_factor).round() as u32);
                c.tile_k = round_to_power_of_two((f64::from(c.tile_k) * k_factor).round() as u32);
                // Scale warp tiles proportionally.
                c.warp_m = round_to_power_of_two((f64::from(c.warp_m) * m_factor).round() as u32);
                c.warp_n = round_to_power_of_two((f64::from(c.warp_n) * n_factor).round() as u32);
            }
            ConfigTransferRule::AdjustPipelineDepth { delta } => {
                let new_depth = (c.stages as i32 + delta).max(1) as u32;
                c.stages = new_depth;
            }
            ConfigTransferRule::EnableTensorCores => {
                c.use_tensor_core = true;
            }
            ConfigTransferRule::DisableTensorCores => {
                c.use_tensor_core = false;
            }
            ConfigTransferRule::Composite(rules) => {
                for r in rules {
                    c = Self::apply_rule(&c, r, strategy);
                }
            }
        }
        c
    }

    /// Clamp config values to the target architecture's hardware limits.
    fn clamp_to_target(config: &Config, target: &ArchitectureProfile) -> Config {
        let mut c = config.clone();
        let max_smem = u64::from(target.max_shared_memory);

        // Iteratively reduce until shared memory fits.
        // First try reducing stages, then shrink tiles.
        for _ in 0..16 {
            if c.estimated_shared_mem(4) <= max_smem {
                break;
            }
            if c.stages > 1 {
                c.stages -= 1;
            } else if c.tile_k > 8 {
                c.tile_k /= 2;
            } else if c.tile_m > 16 || c.tile_n > 16 {
                c.tile_m = (c.tile_m / 2).max(16);
                c.tile_n = (c.tile_n / 2).max(16);
            } else {
                break;
            }
        }

        // Clamp tile dimensions to reasonable bounds.
        c.tile_m = c.tile_m.clamp(16, 512);
        c.tile_n = c.tile_n.clamp(16, 512);
        c.tile_k = c.tile_k.clamp(8, 128);
        c.warp_m = c.warp_m.clamp(8, c.tile_m);
        c.warp_n = c.warp_n.clamp(8, c.tile_n);
        c.stages = c.stages.clamp(1, 8);
        c
    }
}

/// Round a value to the nearest power of two (≥ 8).
fn round_to_power_of_two(v: u32) -> u32 {
    let v = v.max(8);
    let lo = 1u32 << (31 - v.leading_zeros());
    let hi = lo.saturating_mul(2);
    if v - lo <= hi - v { lo } else { hi }
}

// ---------------------------------------------------------------------------
// TransferredConfig
// ---------------------------------------------------------------------------

/// Result of transferring a single configuration.
#[derive(Debug, Clone)]
pub struct TransferredConfig {
    /// The original configuration from the source architecture.
    pub original_config: Config,
    /// The transformed configuration for the target architecture.
    pub transferred_config: Config,
    /// Confidence that the transferred config is a good starting point (0.0–1.0).
    pub confidence: f64,
    /// The strategy that was applied.
    pub strategy_applied: TransferStrategy,
}

// ---------------------------------------------------------------------------
// TransferLearningEngine
// ---------------------------------------------------------------------------

/// Engine that batch-transfers tuning results between architectures.
///
/// Given a [`ResultDb`] containing results for a *source* architecture,
/// the engine produces warm-start configurations for a *target*
/// architecture by applying architecture-aware transfer rules.
pub struct TransferLearningEngine<'a> {
    source_db: &'a ResultDb,
    strategy: TransferStrategy,
    source_gpu_name: String,
}

impl<'a> TransferLearningEngine<'a> {
    /// Create a new transfer learning engine.
    ///
    /// The `source_gpu_name` is used to look up entries in the result DB.
    #[must_use]
    pub fn new(
        source_db: &'a ResultDb,
        source_gpu_name: &str,
        source_arch: ArchitectureProfile,
        target_arch: ArchitectureProfile,
    ) -> Self {
        let strategy = TransferStrategy::infer(source_arch, target_arch);
        Self {
            source_db,
            strategy,
            source_gpu_name: source_gpu_name.to_string(),
        }
    }

    /// Transfer all results from the source DB to warm-start configs.
    ///
    /// Returns `(kernel_name, problem_key, transferred_config, confidence)`.
    #[must_use]
    pub fn warm_start(&self) -> Vec<(String, String, Config, f64)> {
        let entries = self.source_db.list_gpu(&self.source_gpu_name);
        let mut results = Vec::new();

        for (kernel, problem, bench_result) in entries {
            let transferred = ConfigTransformer::transform(&bench_result.config, &self.strategy);
            let feasibility = Self::config_feasibility(&transferred, &self.strategy.target);
            let confidence = self.strategy.similarity * feasibility;
            results.push((
                kernel.to_string(),
                problem.to_string(),
                transferred,
                confidence,
            ));
        }
        results
    }

    /// Transfer a single (kernel, problem) entry.
    ///
    /// Returns `None` if no matching entry exists in the source DB.
    #[must_use]
    pub fn transfer_single(&self, kernel: &str, problem: &str) -> Option<TransferredConfig> {
        let bench_result = self
            .source_db
            .lookup(&self.source_gpu_name, kernel, problem)?;
        let original = bench_result.config.clone();
        let transferred = ConfigTransformer::transform(&original, &self.strategy);
        let feasibility = Self::config_feasibility(&transferred, &self.strategy.target);
        let confidence = self.strategy.similarity * feasibility;

        Some(TransferredConfig {
            original_config: original,
            transferred_config: transferred,
            confidence,
            strategy_applied: self.strategy.clone(),
        })
    }

    /// Assess how feasible a config is on the target architecture (0.0–1.0).
    ///
    /// Penalizes configs that are close to hardware limits.
    fn config_feasibility(config: &Config, target: &ArchitectureProfile) -> f64 {
        let smem_used = config.estimated_shared_mem(4);
        let smem_max = u64::from(target.max_shared_memory);

        // Shared memory utilization penalty.
        let smem_ratio = if smem_max > 0 {
            (smem_used as f64) / (smem_max as f64)
        } else {
            1.0
        };
        let smem_score = if smem_ratio > 1.0 {
            0.0
        } else {
            1.0 - (smem_ratio * 0.3) // mild penalty for high utilization
        };

        // Register pressure penalty.
        let regs = config.estimated_registers_per_thread();
        let reg_ratio = f64::from(regs) / f64::from(target.max_registers);
        let reg_score = if reg_ratio > 1.0 {
            0.0
        } else {
            1.0 - (reg_ratio * 0.2)
        };

        (smem_score + reg_score) / 2.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkResult;

    fn make_result(config: Config, median_us: f64) -> BenchmarkResult {
        BenchmarkResult {
            config,
            median_us,
            min_us: median_us - 1.0,
            max_us: median_us + 1.0,
            stddev_us: 0.5,
            gflops: Some(1000.0),
            efficiency: Some(0.8),
        }
    }

    // -- Architecture profiles ------------------------------------------------

    #[test]
    fn sm80_profile_has_correct_sm_version() {
        let p = ArchitectureProfile::sm80();
        assert_eq!(p.sm_version, 80);
        assert!(p.has_tensor_cores);
        assert_eq!(p.tensor_core_generations, 2);
    }

    #[test]
    fn sm90_profile_has_hopper_tensor_cores() {
        let p = ArchitectureProfile::sm90();
        assert_eq!(p.sm_version, 90);
        assert_eq!(p.tensor_core_generations, 3);
        assert!(p.max_shared_memory > 200_000);
    }

    #[test]
    fn all_known_profiles_have_warp_size_32() {
        for p in &[
            ArchitectureProfile::sm80(),
            ArchitectureProfile::sm86(),
            ArchitectureProfile::sm89(),
            ArchitectureProfile::sm90(),
        ] {
            assert_eq!(p.warp_size, 32);
        }
    }

    // -- Similarity -----------------------------------------------------------

    #[test]
    fn same_architecture_similarity_is_one() {
        let a = ArchitectureProfile::sm80();
        let sim = ArchitectureSimilarity::compute_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "same arch should be 1.0, got {sim}"
        );
    }

    #[test]
    fn very_different_architectures_have_low_similarity() {
        let source = ArchitectureProfile {
            sm_version: 50,
            max_shared_memory: 48_000,
            max_registers: 128,
            warp_size: 32,
            has_tensor_cores: false,
            tensor_core_generations: 0,
            memory_bandwidth_gbps: 200.0,
            compute_tflops: 2.0,
            clock_mhz: 1000,
            sm_count: 24,
        };
        let target = ArchitectureProfile::sm90();
        let sim = ArchitectureSimilarity::compute_similarity(&source, &target);
        assert!(sim < 0.5, "very different archs should be < 0.5, got {sim}");
    }

    #[test]
    fn sm80_sm86_more_similar_than_sm80_sm90() {
        let sm80 = ArchitectureProfile::sm80();
        let sm86 = ArchitectureProfile::sm86();
        let sm90 = ArchitectureProfile::sm90();
        let sim_86 = ArchitectureSimilarity::compute_similarity(&sm80, &sm86);
        let sim_90 = ArchitectureSimilarity::compute_similarity(&sm80, &sm90);
        // sm80 and sm86 are both Ampere, should be more similar
        assert!(
            sim_86 > sim_90,
            "sm80-sm86 ({sim_86}) should be more similar than sm80-sm90 ({sim_90})"
        );
    }

    // -- Config transformation ------------------------------------------------

    #[test]
    fn keep_as_is_does_not_change_config() {
        let cfg = Config::new().with_tile_m(64).with_tile_n(64);
        let strategy = TransferStrategy {
            source: ArchitectureProfile::sm80(),
            target: ArchitectureProfile::sm80(),
            similarity: 1.0,
            rules: vec![ConfigTransferRule::KeepAsIs],
        };
        let result = ConfigTransformer::transform(&cfg, &strategy);
        assert_eq!(result.tile_m, cfg.tile_m);
        assert_eq!(result.tile_n, cfg.tile_n);
    }

    #[test]
    fn enable_tensor_cores_rule() {
        let cfg = Config::new().with_use_tensor_core(false);
        let strategy = TransferStrategy {
            source: ArchitectureProfile::sm80(),
            target: ArchitectureProfile::sm90(),
            similarity: 0.8,
            rules: vec![ConfigTransferRule::EnableTensorCores],
        };
        let result = ConfigTransformer::transform(&cfg, &strategy);
        assert!(result.use_tensor_core);
    }

    #[test]
    fn disable_tensor_cores_rule() {
        let cfg = Config::new().with_use_tensor_core(true);
        let strategy = TransferStrategy {
            source: ArchitectureProfile::sm90(),
            target: ArchitectureProfile::sm80(),
            similarity: 0.8,
            rules: vec![ConfigTransferRule::DisableTensorCores],
        };
        let result = ConfigTransformer::transform(&cfg, &strategy);
        assert!(!result.use_tensor_core);
    }

    #[test]
    fn scale_tiles_rule() {
        let cfg = Config::new()
            .with_tile_m(64)
            .with_tile_n(64)
            .with_tile_k(32);
        let strategy = TransferStrategy {
            source: ArchitectureProfile::sm80(),
            target: ArchitectureProfile::sm90(),
            similarity: 0.8,
            rules: vec![ConfigTransferRule::ScaleTiles {
                m_factor: 2.0,
                n_factor: 2.0,
                k_factor: 1.0,
            }],
        };
        let result = ConfigTransformer::transform(&cfg, &strategy);
        assert_eq!(result.tile_m, 128);
        assert_eq!(result.tile_n, 128);
        assert_eq!(result.tile_k, 32);
    }

    #[test]
    fn adjust_pipeline_depth_rule() {
        let cfg = Config::new().with_stages(2);
        let strategy = TransferStrategy {
            source: ArchitectureProfile::sm80(),
            target: ArchitectureProfile::sm90(),
            similarity: 0.8,
            rules: vec![ConfigTransferRule::AdjustPipelineDepth { delta: 2 }],
        };
        let result = ConfigTransformer::transform(&cfg, &strategy);
        assert_eq!(result.stages, 4);
    }

    #[test]
    fn pipeline_depth_does_not_go_below_one() {
        let cfg = Config::new().with_stages(1);
        let strategy = TransferStrategy {
            source: ArchitectureProfile::sm90(),
            target: ArchitectureProfile::sm80(),
            similarity: 0.8,
            rules: vec![ConfigTransferRule::AdjustPipelineDepth { delta: -5 }],
        };
        let result = ConfigTransformer::transform(&cfg, &strategy);
        assert_eq!(result.stages, 1);
    }

    #[test]
    fn composite_rule_applies_all_sub_rules() {
        let cfg = Config::new()
            .with_tile_m(64)
            .with_tile_n(64)
            .with_use_tensor_core(false)
            .with_stages(2);
        let strategy = TransferStrategy {
            source: ArchitectureProfile::sm80(),
            target: ArchitectureProfile::sm90(),
            similarity: 0.8,
            rules: vec![ConfigTransferRule::Composite(vec![
                ConfigTransferRule::EnableTensorCores,
                ConfigTransferRule::AdjustPipelineDepth { delta: 1 },
            ])],
        };
        let result = ConfigTransformer::transform(&cfg, &strategy);
        assert!(result.use_tensor_core);
        assert_eq!(result.stages, 3);
    }

    #[test]
    fn clamping_respects_shared_memory_limit() {
        // Huge tiles that would exceed any reasonable smem limit.
        let cfg = Config::new()
            .with_tile_m(512)
            .with_tile_n(512)
            .with_tile_k(128)
            .with_stages(8);
        let strategy = TransferStrategy {
            source: ArchitectureProfile::sm90(),
            target: ArchitectureProfile::sm86(), // smaller smem
            similarity: 0.6,
            rules: vec![ConfigTransferRule::KeepAsIs],
        };
        let result = ConfigTransformer::transform(&cfg, &strategy);
        let smem = result.estimated_shared_mem(4);
        assert!(
            smem <= u64::from(ArchitectureProfile::sm86().max_shared_memory),
            "smem {smem} exceeds target limit"
        );
    }

    // -- Inferred strategy ----------------------------------------------------

    #[test]
    fn infer_sm80_to_sm90_increases_pipeline_depth() {
        let strategy =
            TransferStrategy::infer(ArchitectureProfile::sm80(), ArchitectureProfile::sm90());
        let has_pipeline_increase = strategy
            .rules
            .iter()
            .any(|r| matches!(r, ConfigTransferRule::AdjustPipelineDepth { delta } if *delta > 0));
        assert!(
            has_pipeline_increase,
            "sm80→sm90 should increase pipeline depth"
        );
    }

    // -- TransferLearningEngine -----------------------------------------------

    #[test]
    fn warm_start_from_result_db() {
        let dir = std::env::temp_dir().join("oxicuda_test_transfer_warm");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("results.json");

        let mut db = ResultDb::open_at(path).ok();
        if let Some(ref mut db) = db {
            let cfg = Config::new().with_tile_m(64).with_tile_n(64);
            let _ = db.save("A100", "sgemm", "1024x1024", make_result(cfg.clone(), 42.0));
            let _ = db.save("A100", "dgemm", "512x512", make_result(cfg, 100.0));

            let engine = TransferLearningEngine::new(
                db,
                "A100",
                ArchitectureProfile::sm80(),
                ArchitectureProfile::sm90(),
            );
            let results = engine.warm_start();
            assert_eq!(results.len(), 2);
            for (_, _, _, confidence) in &results {
                assert!(*confidence > 0.0);
                assert!(*confidence <= 1.0);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transfer_single_existing_entry() {
        let dir = std::env::temp_dir().join("oxicuda_test_transfer_single");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("results.json");

        let mut db = ResultDb::open_at(path).ok();
        if let Some(ref mut db) = db {
            let cfg = Config::new().with_tile_m(128).with_stages(2);
            let _ = db.save("A100", "sgemm", "2048x2048", make_result(cfg, 50.0));

            let engine = TransferLearningEngine::new(
                db,
                "A100",
                ArchitectureProfile::sm80(),
                ArchitectureProfile::sm90(),
            );
            let tc = engine.transfer_single("sgemm", "2048x2048");
            assert!(tc.is_some());
            let tc = tc.expect("checked above");
            assert!(tc.confidence > 0.0);
            assert!(tc.confidence <= 1.0);
            assert!(tc.strategy_applied.similarity > 0.0);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transfer_single_missing_returns_none() {
        let dir = std::env::temp_dir().join("oxicuda_test_transfer_none");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("results.json");

        if let Ok(db) = ResultDb::open_at(path) {
            let engine = TransferLearningEngine::new(
                &db,
                "A100",
                ArchitectureProfile::sm80(),
                ArchitectureProfile::sm90(),
            );
            let tc = engine.transfer_single("nonexistent", "1x1");
            assert!(tc.is_none());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn confidence_scoring_range() {
        let dir = std::env::temp_dir().join("oxicuda_test_confidence_range");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("results.json");

        if let Ok(mut db) = ResultDb::open_at(path) {
            let cfg = Config::new();
            let _ = db.save("A100", "sgemm", "1024", make_result(cfg, 30.0));

            let engine = TransferLearningEngine::new(
                &db,
                "A100",
                ArchitectureProfile::sm80(),
                ArchitectureProfile::sm90(),
            );
            let results = engine.warm_start();
            for (_, _, _, conf) in &results {
                assert!(
                    *conf >= 0.0 && *conf <= 1.0,
                    "confidence out of range: {conf}"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_to_power_of_two_edge_cases() {
        assert_eq!(round_to_power_of_two(1), 8);
        assert_eq!(round_to_power_of_two(8), 8);
        assert_eq!(round_to_power_of_two(9), 8);
        assert_eq!(round_to_power_of_two(12), 8); // 12 is closer to 8 than 16
        assert_eq!(round_to_power_of_two(64), 64);
        assert_eq!(round_to_power_of_two(65), 64);
        assert_eq!(round_to_power_of_two(96), 64); // equidistant, rounds down
        assert_eq!(round_to_power_of_two(97), 128);
        assert_eq!(round_to_power_of_two(128), 128);
    }
}
