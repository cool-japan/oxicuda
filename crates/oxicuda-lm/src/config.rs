//! Model configurations for GPT-2 and LLaMA family models.
//!
//! The structs here store the *hyperparameters* (layer counts, dimensions,
//! etc.) that define a model's shape.  They contain no weights.
//!
//! Pre-built constructors provide the exact configurations published in the
//! original model papers so that tests and examples can use realistic shapes.

use crate::error::{LmError, LmResult};

// ─── GptConfig ───────────────────────────────────────────────────────────────

/// Configuration for a GPT-2 style transformer.
///
/// Architecture: token embedding + learned positional embedding →
/// N × (LayerNorm → MHA → residual → LayerNorm → MLP → residual) →
/// final LayerNorm → LM head (weight-tied to token embedding).
#[derive(Debug, Clone)]
pub struct GptConfig {
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Number of attention heads per layer.
    pub n_heads: usize,
    /// Hidden / embedding dimension.
    pub n_embd: usize,
    /// Maximum sequence length (position vocabulary size).
    pub n_positions: usize,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// FFN inner dimension (typically 4 × `n_embd`).
    pub ffn_intermediate: usize,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f32,
}

impl GptConfig {
    /// Validate consistency of the configuration.
    pub fn validate(&self) -> LmResult<()> {
        if self.n_layers == 0 {
            return Err(LmError::InvalidConfig {
                msg: "n_layers must be > 0".into(),
            });
        }
        if self.n_heads == 0 {
            return Err(LmError::InvalidConfig {
                msg: "n_heads must be > 0".into(),
            });
        }
        if self.n_embd == 0 || self.n_embd % self.n_heads != 0 {
            return Err(LmError::HeadDimMismatch {
                hidden_dim: self.n_embd,
                n_heads: self.n_heads,
            });
        }
        if self.vocab_size == 0 {
            return Err(LmError::InvalidConfig {
                msg: "vocab_size must be > 0".into(),
            });
        }
        if self.ffn_intermediate == 0 {
            return Err(LmError::InvalidConfig {
                msg: "ffn_intermediate must be > 0".into(),
            });
        }
        if self.n_positions == 0 {
            return Err(LmError::InvalidConfig {
                msg: "n_positions must be > 0".into(),
            });
        }
        Ok(())
    }

    /// Head dimension derived from `n_embd / n_heads`.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_heads
    }

    // ── Real model presets ────────────────────────────────────────────────

    /// GPT-2 Small (117 M parameters).
    pub fn gpt2_small() -> Self {
        Self {
            n_layers: 12,
            n_heads: 12,
            n_embd: 768,
            n_positions: 1024,
            vocab_size: 50257,
            ffn_intermediate: 3072,
            layer_norm_eps: 1e-5,
        }
    }

    /// GPT-2 Medium (345 M parameters).
    pub fn gpt2_medium() -> Self {
        Self {
            n_layers: 24,
            n_heads: 16,
            n_embd: 1024,
            n_positions: 1024,
            vocab_size: 50257,
            ffn_intermediate: 4096,
            layer_norm_eps: 1e-5,
        }
    }

    /// GPT-2 Large (762 M parameters).
    pub fn gpt2_large() -> Self {
        Self {
            n_layers: 36,
            n_heads: 20,
            n_embd: 1280,
            n_positions: 1024,
            vocab_size: 50257,
            ffn_intermediate: 5120,
            layer_norm_eps: 1e-5,
        }
    }

    /// GPT-2 XL (1.5 B parameters).
    pub fn gpt2_xl() -> Self {
        Self {
            n_layers: 48,
            n_heads: 25,
            n_embd: 1600,
            n_positions: 1024,
            vocab_size: 50257,
            ffn_intermediate: 6400,
            layer_norm_eps: 1e-5,
        }
    }

    /// Tiny configuration suitable for unit tests
    /// (2 layers, 2 heads, embedding dim 8, vocab 16).
    pub fn tiny() -> Self {
        Self {
            n_layers: 2,
            n_heads: 2,
            n_embd: 8,
            n_positions: 32,
            vocab_size: 16,
            ffn_intermediate: 16,
            layer_norm_eps: 1e-5,
        }
    }
}

// ─── LlamaConfig ─────────────────────────────────────────────────────────────

/// Configuration for a LLaMA-2 / LLaMA-3 / Mistral style transformer.
///
/// Architecture: token embedding →
/// N × (RMSNorm → GQA-MHA with RoPE → residual → RMSNorm → SwiGLU FFN → residual) →
/// final RMSNorm → LM head (independent weight).
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (GQA; `n_kv_heads` divides `n_heads`).
    pub n_kv_heads: usize,
    /// Hidden dimension (must be divisible by `n_heads`).
    pub hidden_dim: usize,
    /// SwiGLU FFN intermediate dimension.
    pub intermediate_dim: usize,
    /// Maximum sequence length (also limits the RoPE cache).
    pub max_position_embeddings: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency (default 10 000.0 for LLaMA-2, 500 000.0 for LLaMA-3).
    pub rope_theta: f32,
}

impl LlamaConfig {
    /// Validate consistency of the configuration.
    pub fn validate(&self) -> LmResult<()> {
        if self.n_layers == 0 {
            return Err(LmError::InvalidConfig {
                msg: "n_layers must be > 0".into(),
            });
        }
        if self.n_heads == 0 {
            return Err(LmError::InvalidConfig {
                msg: "n_heads must be > 0".into(),
            });
        }
        if self.n_kv_heads == 0 {
            return Err(LmError::InvalidConfig {
                msg: "n_kv_heads must be > 0".into(),
            });
        }
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(LmError::GqaHeadMismatch {
                n_heads: self.n_heads,
                n_kv_heads: self.n_kv_heads,
            });
        }
        if self.hidden_dim == 0 || self.hidden_dim % self.n_heads != 0 {
            return Err(LmError::HeadDimMismatch {
                hidden_dim: self.hidden_dim,
                n_heads: self.n_heads,
            });
        }
        if self.intermediate_dim == 0 {
            return Err(LmError::InvalidConfig {
                msg: "intermediate_dim must be > 0".into(),
            });
        }
        if self.vocab_size == 0 {
            return Err(LmError::InvalidConfig {
                msg: "vocab_size must be > 0".into(),
            });
        }
        if self.max_position_embeddings == 0 {
            return Err(LmError::InvalidConfig {
                msg: "max_position_embeddings must be > 0".into(),
            });
        }
        if self.rope_theta <= 0.0 {
            return Err(LmError::InvalidConfig {
                msg: "rope_theta must be > 0".into(),
            });
        }
        Ok(())
    }

    /// Head dimension: `hidden_dim / n_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.n_heads
    }

    /// GQA repeat factor: `n_heads / n_kv_heads`.
    pub fn gqa_factor(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    // ── Real model presets ────────────────────────────────────────────────

    /// LLaMA-2 7B.
    pub fn llama2_7b() -> Self {
        Self {
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 32, // no GQA in original LLaMA-2 7B
            hidden_dim: 4096,
            intermediate_dim: 11008,
            max_position_embeddings: 4096,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    /// LLaMA-2 13B.
    pub fn llama2_13b() -> Self {
        Self {
            n_layers: 40,
            n_heads: 40,
            n_kv_heads: 40,
            hidden_dim: 5120,
            intermediate_dim: 13824,
            max_position_embeddings: 4096,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    /// LLaMA-3 8B (uses GQA: 32 Q heads, 8 KV heads).
    pub fn llama3_8b() -> Self {
        Self {
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            hidden_dim: 4096,
            intermediate_dim: 14336,
            max_position_embeddings: 8192,
            vocab_size: 128256,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
        }
    }

    /// Mistral-7B-v0.1 (uses GQA: 32 Q heads, 8 KV heads).
    pub fn mistral_7b() -> Self {
        Self {
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            hidden_dim: 4096,
            intermediate_dim: 14336,
            max_position_embeddings: 32768,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    /// Phi-2 (Microsoft Phi-2 model, MHA not GQA).
    pub fn phi2() -> Self {
        Self {
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 32,
            hidden_dim: 2560,
            intermediate_dim: 10240,
            max_position_embeddings: 2048,
            vocab_size: 51200,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    /// Tiny configuration for unit tests
    /// (2 layers, 4 Q heads, 2 KV heads, hidden_dim 8).
    pub fn tiny() -> Self {
        Self {
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            hidden_dim: 8,
            intermediate_dim: 12,
            max_position_embeddings: 32,
            vocab_size: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_small_validates() {
        GptConfig::gpt2_small().validate().unwrap();
    }

    #[test]
    fn gpt2_medium_head_dim() {
        let cfg = GptConfig::gpt2_medium();
        assert_eq!(cfg.head_dim(), 64); // 1024/16
    }

    #[test]
    fn gpt2_xl_validates() {
        GptConfig::gpt2_xl().validate().unwrap();
    }

    #[test]
    fn gpt2_tiny_validates() {
        GptConfig::tiny().validate().unwrap();
    }

    #[test]
    fn gpt2_invalid_heads() {
        let mut cfg = GptConfig::tiny();
        cfg.n_heads = 3; // 8 % 3 != 0
        assert!(matches!(
            cfg.validate(),
            Err(LmError::HeadDimMismatch { .. })
        ));
    }

    #[test]
    fn gpt2_zero_vocab() {
        let mut cfg = GptConfig::tiny();
        cfg.vocab_size = 0;
        assert!(matches!(cfg.validate(), Err(LmError::InvalidConfig { .. })));
    }

    #[test]
    fn llama2_7b_validates() {
        LlamaConfig::llama2_7b().validate().unwrap();
    }

    #[test]
    fn llama3_8b_head_dim() {
        let cfg = LlamaConfig::llama3_8b();
        assert_eq!(cfg.head_dim(), 128); // 4096/32
        assert_eq!(cfg.gqa_factor(), 4); // 32/8
    }

    #[test]
    fn mistral_7b_validates() {
        LlamaConfig::mistral_7b().validate().unwrap();
    }

    #[test]
    fn phi2_validates() {
        LlamaConfig::phi2().validate().unwrap();
    }

    #[test]
    fn llama_tiny_validates() {
        LlamaConfig::tiny().validate().unwrap();
    }

    #[test]
    fn llama_invalid_gqa() {
        let mut cfg = LlamaConfig::tiny();
        cfg.n_kv_heads = 3; // 4 % 3 != 0
        assert!(matches!(
            cfg.validate(),
            Err(LmError::GqaHeadMismatch { .. })
        ));
    }

    #[test]
    fn llama_invalid_rope_theta() {
        let mut cfg = LlamaConfig::tiny();
        cfg.rope_theta = 0.0;
        assert!(matches!(cfg.validate(), Err(LmError::InvalidConfig { .. })));
    }
}
