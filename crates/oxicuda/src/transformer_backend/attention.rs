//! Attention dispatch for transformer inference.
//!
//! Selects the optimal attention kernel based on sequence length, hardware
//! capabilities, cache mode, and head configuration. Supports:
//!
//! - Standard O(n²) attention
//! - FlashAttention-2 (tiled, memory-efficient)
//! - FlashAttention-3 (Hopper async pipeline)
//! - PagedAttention (vLLM-style)
//! - Sliding window attention (Mistral)
//! - Linearized attention with RoPE

use super::{TransformerError, TransformerResult};

/// Attention kernel variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionKind {
    /// Standard O(n²) dot-product attention.
    Standard,
    /// FlashAttention-2 — tiled, memory-efficient O(n²) with IO-awareness.
    Flash,
    /// FlashAttention-3 — Hopper async pipeline with warp-specialization.
    FlashHopper,
    /// PagedAttention — vLLM-style with block tables.
    Paged,
    /// Sliding window attention with the given window size.
    SlidingWindow(usize),
    /// Linearized attention with RoPE (rotary positional encoding).
    LinearRope,
}

impl std::fmt::Display for AttentionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard"),
            Self::Flash => write!(f, "FlashAttention-2"),
            Self::FlashHopper => write!(f, "FlashAttention-3 (Hopper)"),
            Self::Paged => write!(f, "PagedAttention"),
            Self::SlidingWindow(w) => write!(f, "SlidingWindow({w})"),
            Self::LinearRope => write!(f, "LinearRoPE"),
        }
    }
}

/// Head configuration for attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeadConfig {
    /// Multi-head attention: all heads have separate K, V.
    Mha {
        /// Number of query/key/value heads.
        num_heads: usize,
    },
    /// Grouped-query attention: fewer KV heads than query heads.
    Gqa {
        /// Number of query heads.
        num_heads: usize,
        /// Number of key-value heads (must divide num_heads evenly).
        num_kv_heads: usize,
    },
    /// Multi-query attention: single KV head shared across all query heads.
    Mqa {
        /// Number of query heads.
        num_heads: usize,
    },
}

impl HeadConfig {
    /// Number of query heads.
    pub fn num_query_heads(&self) -> usize {
        match self {
            Self::Mha { num_heads } => *num_heads,
            Self::Gqa { num_heads, .. } => *num_heads,
            Self::Mqa { num_heads } => *num_heads,
        }
    }

    /// Number of key-value heads.
    pub fn num_kv_heads(&self) -> usize {
        match self {
            Self::Mha { num_heads } => *num_heads,
            Self::Gqa { num_kv_heads, .. } => *num_kv_heads,
            Self::Mqa { .. } => 1,
        }
    }

    /// Validate the head configuration.
    pub fn validate(&self) -> TransformerResult<()> {
        match self {
            Self::Mha { num_heads } => {
                if *num_heads == 0 {
                    return Err(TransformerError::AttentionError(
                        "MHA num_heads must be > 0".to_string(),
                    ));
                }
            }
            Self::Gqa {
                num_heads,
                num_kv_heads,
            } => {
                if *num_heads == 0 || *num_kv_heads == 0 {
                    return Err(TransformerError::AttentionError(
                        "GQA heads must be > 0".to_string(),
                    ));
                }
                if num_heads % num_kv_heads != 0 {
                    return Err(TransformerError::AttentionError(format!(
                        "GQA: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
                    )));
                }
            }
            Self::Mqa { num_heads } => {
                if *num_heads == 0 {
                    return Err(TransformerError::AttentionError(
                        "MQA num_heads must be > 0".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Number of query heads per KV head group.
    pub fn group_size(&self) -> usize {
        let q = self.num_query_heads();
        let kv = self.num_kv_heads();
        if kv == 0 {
            return 0;
        }
        q / kv
    }

    /// KV memory ratio compared to MHA.
    ///
    /// Returns a value in (0.0, 1.0] indicating the fraction of KV memory
    /// this config uses relative to full MHA.
    pub fn kv_memory_ratio(&self) -> f64 {
        let q = self.num_query_heads() as f64;
        let kv = self.num_kv_heads() as f64;
        if q == 0.0 {
            return 0.0;
        }
        kv / q
    }
}

/// Compute capability tier for hardware-based kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComputeTier {
    /// SM 7.x (Volta, Turing) — basic tensor cores.
    Volta,
    /// SM 8.x (Ampere) — BF16, improved tensor cores.
    Ampere,
    /// SM 9.0 (Hopper) — async pipeline, FP8.
    Hopper,
}

/// Configuration for attention dispatch.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Head configuration.
    pub head_config: HeadConfig,
    /// Head dimension.
    pub head_dim: usize,
    /// Whether paged cache is in use.
    pub use_paged_cache: bool,
    /// Compute capability tier.
    pub compute_tier: ComputeTier,
    /// Optional sliding window size.
    pub sliding_window: Option<usize>,
    /// Whether to use causal masking.
    pub causal: bool,
    /// Scale factor (defaults to 1/sqrt(head_dim)).
    pub scale: Option<f64>,
    /// Maximum sequence length hint (for kernel selection).
    pub max_seq_len_hint: Option<usize>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            head_config: HeadConfig::Mha { num_heads: 32 },
            head_dim: 128,
            use_paged_cache: false,
            compute_tier: ComputeTier::Ampere,
            sliding_window: None,
            causal: true,
            scale: None,
            max_seq_len_hint: None,
        }
    }
}

/// Attention dispatch engine.
///
/// Selects and configures the optimal attention kernel based on the
/// current configuration and runtime parameters.
#[derive(Debug, Clone)]
pub struct AttentionDispatch {
    /// Current configuration.
    config: AttentionConfig,
    /// Currently selected kernel.
    selected_kernel: AttentionKind,
}

/// Threshold for switching from standard to flash attention.
const FLASH_THRESHOLD: usize = 512;

/// Threshold for considering a sequence "very long".
const VERY_LONG_THRESHOLD: usize = 8192;

impl AttentionDispatch {
    /// Create a new attention dispatcher from configuration.
    pub fn new(config: AttentionConfig) -> TransformerResult<Self> {
        config.head_config.validate()?;
        if config.head_dim == 0 {
            return Err(TransformerError::AttentionError(
                "head_dim must be > 0".to_string(),
            ));
        }

        let selected_kernel = Self::select_kernel_for_config(&config, None);

        Ok(Self {
            config,
            selected_kernel,
        })
    }

    /// Select an attention kernel for the given sequence length.
    pub fn select_kernel(&mut self, seq_len: usize) -> AttentionKind {
        self.selected_kernel = Self::select_kernel_for_config(&self.config, Some(seq_len));
        self.selected_kernel
    }

    /// Get the currently selected kernel.
    pub fn current_kernel(&self) -> AttentionKind {
        self.selected_kernel
    }

    /// Get the configuration.
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    /// Get the attention scale factor.
    pub fn scale(&self) -> f64 {
        self.config
            .scale
            .unwrap_or_else(|| 1.0 / (self.config.head_dim as f64).sqrt())
    }

    /// Update configuration and re-select kernel.
    pub fn update_config(&mut self, config: AttentionConfig) -> TransformerResult<()> {
        config.head_config.validate()?;
        self.config = config;
        self.selected_kernel = Self::select_kernel_for_config(&self.config, None);
        Ok(())
    }

    /// Get memory estimate for attention computation (in bytes).
    ///
    /// For a single batch element with the given sequence lengths.
    pub fn memory_estimate(&self, seq_len: usize, past_kv_len: usize) -> usize {
        let total_len = seq_len + past_kv_len;
        let num_q = self.config.head_config.num_query_heads();
        let num_kv = self.config.head_config.num_kv_heads();
        let head_dim = self.config.head_dim;

        match self.selected_kernel {
            AttentionKind::Standard => {
                // Q*K^T: [num_q, seq_len, total_len] + softmax + output
                let qk_size = num_q * seq_len * total_len * 2; // fp16
                let output_size = num_q * seq_len * head_dim * 2;
                qk_size + output_size
            }
            AttentionKind::Flash | AttentionKind::FlashHopper => {
                // O(n) memory — only Q, K, V, O in tiles
                let q_size = num_q * seq_len * head_dim * 2;
                let kv_size = num_kv * total_len * head_dim * 2 * 2; // K+V
                let output_size = num_q * seq_len * head_dim * 2;
                q_size + kv_size + output_size
            }
            AttentionKind::Paged => {
                // Similar to flash but with block table overhead
                let base = num_q * seq_len * head_dim * 2;
                let kv = num_kv * total_len * head_dim * 2 * 2;
                let block_table = total_len * 4; // block IDs
                base + kv + block_table
            }
            AttentionKind::SlidingWindow(w) => {
                let effective_len = total_len.min(w);
                let qk_size = num_q * seq_len * effective_len * 2;
                let output_size = num_q * seq_len * head_dim * 2;
                qk_size + output_size
            }
            AttentionKind::LinearRope => {
                // Linear attention: O(n * d²) instead of O(n²)
                let feature_size = num_q * seq_len * head_dim * 2;
                let state_size = num_kv * head_dim * head_dim * 2;
                feature_size + state_size
            }
        }
    }

    fn select_kernel_for_config(config: &AttentionConfig, seq_len: Option<usize>) -> AttentionKind {
        // Priority order:
        // 1. Sliding window if configured
        // 2. Paged attention if paged cache is in use
        // 3. FlashHopper for Hopper hardware + long sequences
        // 4. Flash for Ampere+ and seq > threshold
        // 5. Standard for short sequences

        if let Some(window) = config.sliding_window {
            return AttentionKind::SlidingWindow(window);
        }

        if config.use_paged_cache {
            return AttentionKind::Paged;
        }

        let effective_len = seq_len
            .or(config.max_seq_len_hint)
            .unwrap_or(FLASH_THRESHOLD);

        if effective_len >= VERY_LONG_THRESHOLD && config.compute_tier >= ComputeTier::Hopper {
            return AttentionKind::FlashHopper;
        }

        if effective_len >= FLASH_THRESHOLD && config.compute_tier >= ComputeTier::Ampere {
            return AttentionKind::Flash;
        }

        if effective_len >= FLASH_THRESHOLD {
            return AttentionKind::Flash;
        }

        AttentionKind::Standard
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_head_config_mha() {
        let cfg = HeadConfig::Mha { num_heads: 32 };
        assert_eq!(cfg.num_query_heads(), 32);
        assert_eq!(cfg.num_kv_heads(), 32);
        assert_eq!(cfg.group_size(), 1);
        assert!((cfg.kv_memory_ratio() - 1.0).abs() < 1e-10);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_head_config_gqa() {
        let cfg = HeadConfig::Gqa {
            num_heads: 32,
            num_kv_heads: 8,
        };
        assert_eq!(cfg.num_query_heads(), 32);
        assert_eq!(cfg.num_kv_heads(), 8);
        assert_eq!(cfg.group_size(), 4);
        assert!((cfg.kv_memory_ratio() - 0.25).abs() < 1e-10);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_head_config_mqa() {
        let cfg = HeadConfig::Mqa { num_heads: 32 };
        assert_eq!(cfg.num_query_heads(), 32);
        assert_eq!(cfg.num_kv_heads(), 1);
        assert_eq!(cfg.group_size(), 32);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_head_config_validation_errors() {
        assert!(HeadConfig::Mha { num_heads: 0 }.validate().is_err());
        assert!(
            HeadConfig::Gqa {
                num_heads: 32,
                num_kv_heads: 0
            }
            .validate()
            .is_err()
        );
        assert!(
            HeadConfig::Gqa {
                num_heads: 32,
                num_kv_heads: 5
            }
            .validate()
            .is_err()
        );
        assert!(HeadConfig::Mqa { num_heads: 0 }.validate().is_err());
    }

    #[test]
    fn test_dispatch_standard_short_seq() {
        let config = AttentionConfig {
            compute_tier: ComputeTier::Ampere,
            max_seq_len_hint: Some(64),
            ..Default::default()
        };
        let dispatch = AttentionDispatch::new(config).unwrap();
        assert_eq!(dispatch.current_kernel(), AttentionKind::Standard);
    }

    #[test]
    fn test_dispatch_flash_long_seq() {
        let config = AttentionConfig {
            compute_tier: ComputeTier::Ampere,
            max_seq_len_hint: Some(2048),
            ..Default::default()
        };
        let dispatch = AttentionDispatch::new(config).unwrap();
        assert_eq!(dispatch.current_kernel(), AttentionKind::Flash);
    }

    #[test]
    fn test_dispatch_flash_hopper() {
        let config = AttentionConfig {
            compute_tier: ComputeTier::Hopper,
            max_seq_len_hint: Some(16384),
            ..Default::default()
        };
        let dispatch = AttentionDispatch::new(config).unwrap();
        assert_eq!(dispatch.current_kernel(), AttentionKind::FlashHopper);
    }

    #[test]
    fn test_dispatch_paged() {
        let config = AttentionConfig {
            use_paged_cache: true,
            ..Default::default()
        };
        let dispatch = AttentionDispatch::new(config).unwrap();
        assert_eq!(dispatch.current_kernel(), AttentionKind::Paged);
    }

    #[test]
    fn test_dispatch_sliding_window() {
        let config = AttentionConfig {
            sliding_window: Some(4096),
            ..Default::default()
        };
        let dispatch = AttentionDispatch::new(config).unwrap();
        assert_eq!(
            dispatch.current_kernel(),
            AttentionKind::SlidingWindow(4096)
        );
    }

    #[test]
    fn test_dispatch_select_kernel_runtime() {
        let config = AttentionConfig {
            compute_tier: ComputeTier::Ampere,
            ..Default::default()
        };
        let mut dispatch = AttentionDispatch::new(config).unwrap();

        // Short sequence -> standard
        let k = dispatch.select_kernel(64);
        assert_eq!(k, AttentionKind::Standard);

        // Long sequence -> flash
        let k = dispatch.select_kernel(2048);
        assert_eq!(k, AttentionKind::Flash);
    }

    #[test]
    fn test_dispatch_scale() {
        let config = AttentionConfig {
            head_dim: 64,
            ..Default::default()
        };
        let dispatch = AttentionDispatch::new(config).unwrap();
        let expected = 1.0 / 64.0_f64.sqrt();
        assert!((dispatch.scale() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dispatch_custom_scale() {
        let config = AttentionConfig {
            scale: Some(0.5),
            ..Default::default()
        };
        let dispatch = AttentionDispatch::new(config).unwrap();
        assert!((dispatch.scale() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_memory_estimate() {
        let config = AttentionConfig::default();
        let dispatch = AttentionDispatch::new(config).unwrap();
        let mem = dispatch.memory_estimate(1024, 0);
        assert!(mem > 0);
    }

    #[test]
    fn test_attention_kind_display() {
        assert_eq!(format!("{}", AttentionKind::Standard), "Standard");
        assert_eq!(format!("{}", AttentionKind::Flash), "FlashAttention-2");
        assert_eq!(
            format!("{}", AttentionKind::SlidingWindow(4096)),
            "SlidingWindow(4096)"
        );
    }

    #[test]
    fn test_update_config() {
        let config = AttentionConfig::default();
        let mut dispatch = AttentionDispatch::new(config).unwrap();

        let new_config = AttentionConfig {
            use_paged_cache: true,
            ..Default::default()
        };
        dispatch.update_config(new_config).unwrap();
        assert_eq!(dispatch.current_kernel(), AttentionKind::Paged);
    }

    #[test]
    fn test_invalid_head_dim() {
        let config = AttentionConfig {
            head_dim: 0,
            ..Default::default()
        };
        assert!(AttentionDispatch::new(config).is_err());
    }
}
