//! Embedding layers: token embedding, learned positional embedding, and
//! Rotary Positional Embedding (RoPE).

use crate::error::{LmError, LmResult};
use crate::weights::WeightTensor;

// ─── TokenEmbedding ──────────────────────────────────────────────────────────

/// Token embedding table: maps token ids to dense vectors.
///
/// Weight shape: `[vocab_size × embed_dim]`.
/// Output shape: `[seq_len × embed_dim]`.
#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Weight table: `[vocab_size × embed_dim]`, row-major.
    pub weight: WeightTensor,
}

impl TokenEmbedding {
    /// Construct with zero-initialised weights.
    pub fn new(vocab_size: usize, embed_dim: usize) -> LmResult<Self> {
        if vocab_size == 0 || embed_dim == 0 {
            return Err(LmError::InvalidConfig {
                msg: "TokenEmbedding: vocab_size and embed_dim must be > 0".into(),
            });
        }
        let weight = WeightTensor::zeros(&[vocab_size, embed_dim]);
        Ok(Self {
            vocab_size,
            embed_dim,
            weight,
        })
    }

    /// Construct from an existing weight tensor.
    pub fn from_weight(weight: WeightTensor) -> LmResult<Self> {
        if weight.shape.len() != 2 {
            return Err(LmError::DimensionMismatch {
                expected: 2,
                got: weight.shape.len(),
            });
        }
        let vocab_size = weight.shape[0];
        let embed_dim = weight.shape[1];
        if vocab_size == 0 || embed_dim == 0 {
            return Err(LmError::InvalidConfig {
                msg: "TokenEmbedding weight must be non-empty".into(),
            });
        }
        Ok(Self {
            vocab_size,
            embed_dim,
            weight,
        })
    }

    /// Lookup embeddings for `token_ids`.
    ///
    /// Returns a flat buffer of shape `[token_ids.len() × embed_dim]`.
    pub fn forward(&self, token_ids: &[u32]) -> LmResult<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(LmError::EmptyInput {
                context: "token_ids",
            });
        }
        let mut out = vec![0.0_f32; token_ids.len() * self.embed_dim];
        for (pos, &tid) in token_ids.iter().enumerate() {
            if tid as usize >= self.vocab_size {
                return Err(LmError::OutOfVocab { token: tid });
            }
            let src_start = tid as usize * self.embed_dim;
            let dst_start = pos * self.embed_dim;
            out[dst_start..dst_start + self.embed_dim]
                .copy_from_slice(&self.weight.data[src_start..src_start + self.embed_dim]);
        }
        Ok(out)
    }
}

// ─── LearnedPositionalEmbedding ───────────────────────────────────────────────

/// Learned positional embedding table (GPT-2 style).
///
/// Weight shape: `[max_positions × embed_dim]`.
#[derive(Debug, Clone)]
pub struct LearnedPositionalEmbedding {
    /// Maximum number of positions.
    pub max_positions: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Weight table.
    pub weight: WeightTensor,
}

impl LearnedPositionalEmbedding {
    /// Construct with zero-initialised weights.
    pub fn new(max_positions: usize, embed_dim: usize) -> LmResult<Self> {
        if max_positions == 0 || embed_dim == 0 {
            return Err(LmError::InvalidConfig {
                msg: "LearnedPositionalEmbedding: max_positions and embed_dim must be > 0".into(),
            });
        }
        let weight = WeightTensor::zeros(&[max_positions, embed_dim]);
        Ok(Self {
            max_positions,
            embed_dim,
            weight,
        })
    }

    /// Construct from an existing weight tensor.
    pub fn from_weight(weight: WeightTensor) -> LmResult<Self> {
        if weight.shape.len() != 2 {
            return Err(LmError::DimensionMismatch {
                expected: 2,
                got: weight.shape.len(),
            });
        }
        let max_positions = weight.shape[0];
        let embed_dim = weight.shape[1];
        Ok(Self {
            max_positions,
            embed_dim,
            weight,
        })
    }

    /// Return positional embeddings for positions `[offset, offset + seq_len)`.
    ///
    /// Returns flat buffer of shape `[seq_len × embed_dim]`.
    pub fn forward(&self, seq_len: usize, offset: usize) -> LmResult<Vec<f32>> {
        if offset + seq_len > self.max_positions {
            return Err(LmError::SequenceTooLong {
                total_len: offset + seq_len,
                max_pos: self.max_positions,
            });
        }
        let mut out = vec![0.0_f32; seq_len * self.embed_dim];
        for i in 0..seq_len {
            let pos = offset + i;
            let src = pos * self.embed_dim;
            let dst = i * self.embed_dim;
            out[dst..dst + self.embed_dim]
                .copy_from_slice(&self.weight.data[src..src + self.embed_dim]);
        }
        Ok(out)
    }
}

// ─── RotaryEmbedding ─────────────────────────────────────────────────────────

/// Rotary Positional Embedding (RoPE).
///
/// Precomputes `cos` and `sin` tables for all positions up to `max_positions`.
/// The rotation applies to pairs of dimensions `(x_{2i}, x_{2i+1})` as:
///
/// ```text
/// x_out[2i]   = x[2i]*cos(θ_i*pos) − x[2i+1]*sin(θ_i*pos)
/// x_out[2i+1] = x[2i]*sin(θ_i*pos) + x[2i+1]*cos(θ_i*pos)
/// ```
///
/// where `θ_i = theta ^ (-2i / head_dim)`.
///
/// This embeds position information directly into the attention dot product
/// without requiring separate positional embeddings.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Head dimension (must be even).
    pub head_dim: usize,
    /// Maximum sequence length for which tables are precomputed.
    pub max_positions: usize,
    /// RoPE base frequency (typically 10 000 for LLaMA-2, 500 000 for LLaMA-3).
    pub theta: f32,
    /// Cos table: `[max_positions × head_dim/2]`, row-major.
    cos_table: Vec<f32>,
    /// Sin table: `[max_positions × head_dim/2]`, row-major.
    sin_table: Vec<f32>,
}

impl RotaryEmbedding {
    /// Build RoPE tables for the given configuration.
    pub fn new(head_dim: usize, max_positions: usize, theta: f32) -> LmResult<Self> {
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(LmError::InvalidConfig {
                msg: format!("RotaryEmbedding: head_dim={head_dim} must be even and > 0"),
            });
        }
        if max_positions == 0 {
            return Err(LmError::InvalidConfig {
                msg: "RotaryEmbedding: max_positions must be > 0".into(),
            });
        }
        if theta <= 0.0 {
            return Err(LmError::InvalidConfig {
                msg: "RotaryEmbedding: theta must be > 0".into(),
            });
        }

        let half_dim = head_dim / 2;
        let n = max_positions * half_dim;
        let mut cos_table = Vec::with_capacity(n);
        let mut sin_table = Vec::with_capacity(n);

        for pos in 0..max_positions {
            for i in 0..half_dim {
                // θ_i = theta ^ (-2i / head_dim)
                let freq = theta.powf(-((2 * i) as f32) / head_dim as f32);
                let angle = pos as f32 * freq;
                cos_table.push(angle.cos());
                sin_table.push(angle.sin());
            }
        }

        Ok(Self {
            head_dim,
            max_positions,
            theta,
            cos_table,
            sin_table,
        })
    }

    /// Apply RoPE in-place to a QKV projection.
    ///
    /// `x` has shape `[n_tokens × n_heads × head_dim]`.
    /// `offset` is the absolute position of the first token (for KV-cache decode).
    pub fn apply(
        &self,
        x: &mut [f32],
        n_heads: usize,
        n_tokens: usize,
        offset: usize,
    ) -> LmResult<()> {
        // Check positional bounds before buffer size so callers get a more
        // informative error when both conditions are violated simultaneously.
        if offset + n_tokens > self.max_positions {
            return Err(LmError::SequenceTooLong {
                total_len: offset + n_tokens,
                max_pos: self.max_positions,
            });
        }
        let expected = n_tokens * n_heads * self.head_dim;
        if x.len() != expected {
            return Err(LmError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let half_dim = self.head_dim / 2;

        for t in 0..n_tokens {
            let abs_pos = offset + t;
            let cos_row_start = abs_pos * half_dim;
            for h in 0..n_heads {
                let base = (t * n_heads + h) * self.head_dim;
                for i in 0..half_dim {
                    let cos = self.cos_table[cos_row_start + i];
                    let sin = self.sin_table[cos_row_start + i];
                    let x0 = x[base + 2 * i];
                    let x1 = x[base + 2 * i + 1];
                    x[base + 2 * i] = x0 * cos - x1 * sin;
                    x[base + 2 * i + 1] = x0 * sin + x1 * cos;
                }
            }
        }
        Ok(())
    }

    /// Cosine value for `(position, half_dim_index)`.
    pub fn cos_at(&self, pos: usize, i: usize) -> f32 {
        self.cos_table[pos * (self.head_dim / 2) + i]
    }

    /// Sine value for `(position, half_dim_index)`.
    pub fn sin_at(&self, pos: usize, i: usize) -> f32 {
        self.sin_table[pos * (self.head_dim / 2) + i]
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TokenEmbedding ────────────────────────────────────────────────────

    #[test]
    fn token_embedding_lookup() {
        let mut emb = TokenEmbedding::new(4, 3).unwrap();
        // Set row 2 to [1,2,3]
        emb.weight.data[6] = 1.0;
        emb.weight.data[7] = 2.0;
        emb.weight.data[8] = 3.0;
        let out = emb.forward(&[2]).unwrap();
        assert_eq!(out, vec![1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn token_embedding_multi_token() {
        let mut emb = TokenEmbedding::new(3, 2).unwrap();
        emb.weight.data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // tokens [0, 2] → [[1,2], [5,6]]
        let out = emb.forward(&[0, 2]).unwrap();
        assert_eq!(out, vec![1.0_f32, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn token_embedding_out_of_vocab_error() {
        let emb = TokenEmbedding::new(4, 3).unwrap();
        assert!(matches!(
            emb.forward(&[5]),
            Err(LmError::OutOfVocab { token: 5 })
        ));
    }

    #[test]
    fn token_embedding_empty_error() {
        let emb = TokenEmbedding::new(4, 3).unwrap();
        assert!(matches!(emb.forward(&[]), Err(LmError::EmptyInput { .. })));
    }

    #[test]
    fn token_embedding_from_weight() {
        let w = WeightTensor::zeros(&[10, 4]);
        let emb = TokenEmbedding::from_weight(w).unwrap();
        assert_eq!(emb.vocab_size, 10);
        assert_eq!(emb.embed_dim, 4);
    }

    // ── LearnedPositionalEmbedding ────────────────────────────────────────

    #[test]
    fn pos_embedding_lookup() {
        let mut pe = LearnedPositionalEmbedding::new(4, 2).unwrap();
        // Set position 1 to [3.0, 4.0]
        pe.weight.data[2] = 3.0;
        pe.weight.data[3] = 4.0;
        let out = pe.forward(2, 0).unwrap();
        // pos 0: [0,0], pos 1: [3,4]
        assert_eq!(out, vec![0.0_f32, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn pos_embedding_with_offset() {
        let mut pe = LearnedPositionalEmbedding::new(8, 2).unwrap();
        // positions 4,5 get value 10
        for i in 8..12 {
            pe.weight.data[i] = 10.0;
        }
        let out = pe.forward(2, 4).unwrap(); // positions 4..6
        assert!(out.iter().all(|&v| v == 10.0));
    }

    #[test]
    fn pos_embedding_too_long_error() {
        let pe = LearnedPositionalEmbedding::new(4, 2).unwrap();
        assert!(matches!(
            pe.forward(5, 0),
            Err(LmError::SequenceTooLong { .. })
        ));
    }

    // ── RotaryEmbedding ───────────────────────────────────────────────────

    #[test]
    fn rope_pos0_is_identity() {
        // At position 0, angle = 0, cos=1, sin=0 → rotation is identity
        let rope = RotaryEmbedding::new(4, 16, 10_000.0).unwrap();
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0]; // 1 token, 1 head, head_dim=4
        rope.apply(&mut x, 1, 1, 0).unwrap();
        // All cos=1 at pos=0 for i=0, so x[0]=1*1-2*sin=1-2*0=1, x[1]=1*0+2*1=2
        assert!((x[0] - 1.0).abs() < 1e-5, "x[0]={}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-5, "x[1]={}", x[1]);
        assert!((x[2] - 3.0).abs() < 1e-5, "x[2]={}", x[2]);
        assert!((x[3] - 4.0).abs() < 1e-5, "x[3]={}", x[3]);
    }

    #[test]
    fn rope_rotation_preserves_norm() {
        // Rotation is orthogonal → norm preserved.
        let rope = RotaryEmbedding::new(4, 32, 10_000.0).unwrap();
        let original = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut x = original.clone();
        rope.apply(&mut x, 1, 1, 5).unwrap(); // pos=5
        let norm_before: f32 = original.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let norm_after: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "norm {norm_before} ≠ {norm_after}"
        );
    }

    #[test]
    fn rope_multi_head_multi_token() {
        // Just check no error and correct output size.
        let rope = RotaryEmbedding::new(4, 32, 10_000.0).unwrap();
        let mut x = vec![1.0_f32; 2 * 3 * 4]; // 2 tokens, 3 heads, head_dim=4
        rope.apply(&mut x, 3, 2, 0).unwrap();
        assert_eq!(x.len(), 24);
    }

    #[test]
    fn rope_odd_head_dim_error() {
        assert!(RotaryEmbedding::new(3, 16, 10_000.0).is_err());
    }

    #[test]
    fn rope_sequence_too_long_error() {
        let rope = RotaryEmbedding::new(4, 4, 10_000.0).unwrap();
        let mut x = vec![0.0_f32; 4];
        // offset=3 + seq_len=2 = 5 > max_positions=4
        assert!(matches!(
            rope.apply(&mut x, 1, 2, 3),
            Err(LmError::SequenceTooLong { .. })
        ));
    }

    #[test]
    fn rope_cos_sin_tables_at_zero() {
        let rope = RotaryEmbedding::new(4, 8, 10_000.0).unwrap();
        // At position 0, cos=1, sin=0 for all dims
        assert!((rope.cos_at(0, 0) - 1.0).abs() < 1e-6);
        assert!(rope.sin_at(0, 0).abs() < 1e-6);
    }

    #[test]
    fn rope_tables_have_correct_dimensions() {
        let head_dim = 8;
        let max_pos = 16;
        let rope = RotaryEmbedding::new(head_dim, max_pos, 10_000.0).unwrap();
        assert_eq!(rope.cos_table.len(), max_pos * (head_dim / 2));
        assert_eq!(rope.sin_table.len(), max_pos * (head_dim / 2));
    }
}
