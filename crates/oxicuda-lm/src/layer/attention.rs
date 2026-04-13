//! Multi-head attention (MHA) with optional Grouped Query Attention (GQA)
//! and KV-cache support for incremental decoding.
//!
//! # KV Cache
//!
//! During inference, previously computed key/value tensors are stored in a
//! [`LayerKvCache`].  On each decode step the new K/V slice is appended to
//! the cache before computing attention, so each attention call only projects
//! the *new* query token(s) while attending to the full context.
//!
//! # GQA
//!
//! When `n_kv_heads < n_heads`, key/value heads are shared across groups of
//! `n_heads / n_kv_heads` query heads.  This reduces the KV-cache size
//! proportionally.

use crate::error::{LmError, LmResult};
use crate::layer::embedding::RotaryEmbedding;
use crate::layer::ffn::linear_batch;
use crate::weights::WeightTensor;

// ─── LayerKvCache ────────────────────────────────────────────────────────────

/// Key/value cache for a single transformer layer.
///
/// Keys and values are stored flat as `[past_len × n_kv_heads × head_dim]`.
#[derive(Debug, Clone)]
pub struct LayerKvCache {
    /// Cached keys: `[past_len × n_kv_heads × head_dim]`.
    pub keys: Vec<f32>,
    /// Cached values: `[past_len × n_kv_heads × head_dim]`.
    pub values: Vec<f32>,
    /// Number of tokens already in the cache.
    pub past_len: usize,
    /// Number of KV heads.
    pub n_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
}

impl LayerKvCache {
    /// Create an empty cache.
    pub fn new(n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: vec![],
            values: vec![],
            past_len: 0,
            n_kv_heads,
            head_dim,
        }
    }

    /// Append `new_k` and `new_v` (each `[seq_len × n_kv_heads × head_dim]`).
    pub fn append(&mut self, new_k: &[f32], new_v: &[f32], seq_len: usize) {
        self.keys.extend_from_slice(new_k);
        self.values.extend_from_slice(new_v);
        self.past_len += seq_len;
    }

    /// Total key/value sequence length (past + newly appended).
    pub fn total_len(&self) -> usize {
        self.past_len
    }
}

// ─── MultiHeadAttention ──────────────────────────────────────────────────────

/// Scaled dot-product multi-head (or grouped-query) attention.
///
/// Supports:
/// - Full attention (MHA): `n_kv_heads == n_heads`
/// - Grouped query attention (GQA): `n_kv_heads < n_heads`
/// - Causal masking for autoregressive decoding
/// - KV cache for incremental decoding
/// - Optional RoPE
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (for GQA, `n_kv_heads ≤ n_heads`).
    pub n_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Total hidden dimension (`n_heads × head_dim`).
    pub hidden_dim: usize,
    /// Query projection: `[hidden_dim × hidden_dim]`.
    pub w_q: WeightTensor,
    /// Key projection: `[n_kv_heads × head_dim × hidden_dim]` (flat as 2-D `[kv_proj_dim × hidden_dim]`).
    pub w_k: WeightTensor,
    /// Value projection: same shape as `w_k`.
    pub w_v: WeightTensor,
    /// Output projection: `[hidden_dim × hidden_dim]`.
    pub w_o: WeightTensor,
    /// Optional query bias: `[hidden_dim]`.
    pub b_q: Option<Vec<f32>>,
    /// Optional key bias: `[n_kv_heads × head_dim]`.
    pub b_k: Option<Vec<f32>>,
    /// Optional value bias: same shape as `b_k`.
    pub b_v: Option<Vec<f32>>,
    /// Optional output bias: `[hidden_dim]`.
    pub b_o: Option<Vec<f32>>,
    /// Whether to apply causal (lower-triangular) mask.
    pub causal: bool,
}

impl MultiHeadAttention {
    /// Construct with zero-initialised weights, no biases, causal = true.
    pub fn new(
        n_heads: usize,
        n_kv_heads: usize,
        hidden_dim: usize,
        causal: bool,
    ) -> LmResult<Self> {
        if n_heads == 0 || hidden_dim == 0 {
            return Err(LmError::InvalidConfig {
                msg: "n_heads and hidden_dim must be > 0".into(),
            });
        }
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(LmError::GqaHeadMismatch {
                n_heads,
                n_kv_heads,
            });
        }
        if hidden_dim % n_heads != 0 {
            return Err(LmError::HeadDimMismatch {
                hidden_dim,
                n_heads,
            });
        }
        let head_dim = hidden_dim / n_heads;
        let kv_proj_dim = n_kv_heads * head_dim;
        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_dim,
            w_q: WeightTensor::zeros(&[hidden_dim, hidden_dim]),
            w_k: WeightTensor::zeros(&[kv_proj_dim, hidden_dim]),
            w_v: WeightTensor::zeros(&[kv_proj_dim, hidden_dim]),
            w_o: WeightTensor::zeros(&[hidden_dim, hidden_dim]),
            b_q: None,
            b_k: None,
            b_v: None,
            b_o: None,
            causal,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// - `x`: `[seq_len × hidden_dim]` — input tokens.
    /// - `past_kv`: optional KV cache to extend.
    /// - `rope`: optional RoPE to apply to Q and K.
    ///
    /// # Returns
    ///
    /// `(output [seq_len × hidden_dim], updated LayerKvCache)`.
    pub fn forward(
        &self,
        x: &[f32],
        seq_len: usize,
        past_kv: Option<&LayerKvCache>,
        rope: Option<&RotaryEmbedding>,
    ) -> LmResult<(Vec<f32>, LayerKvCache)> {
        if seq_len == 0 {
            return Err(LmError::EmptyInput {
                context: "MultiHeadAttention::forward seq_len",
            });
        }
        let expected_len = seq_len * self.hidden_dim;
        if x.len() != expected_len {
            return Err(LmError::DimensionMismatch {
                expected: expected_len,
                got: x.len(),
            });
        }

        let past_len = past_kv.map_or(0, |c| c.past_len);
        let kv_proj_dim = self.n_kv_heads * self.head_dim;

        // ── 1. Project Q, K, V ───────────────────────────────────────────
        let mut q = linear_batch(&self.w_q, self.b_q.as_deref(), x, seq_len)?;
        let mut k_new = linear_batch(&self.w_k, self.b_k.as_deref(), x, seq_len)?;
        let v_new = linear_batch(&self.w_v, self.b_v.as_deref(), x, seq_len)?;

        // ── 2. Apply RoPE ─────────────────────────────────────────────────
        if let Some(r) = rope {
            r.apply(&mut q, self.n_heads, seq_len, past_len)?;
            r.apply(&mut k_new, self.n_kv_heads, seq_len, past_len)?;
        }

        // ── 3. Build/update KV cache ──────────────────────────────────────
        // full_k: [total_len × n_kv_heads × head_dim]
        // full_v: [total_len × n_kv_heads × head_dim]
        let (full_k, full_v) = if let Some(cache) = past_kv {
            let mut fk = cache.keys.clone();
            fk.extend_from_slice(&k_new);
            let mut fv = cache.values.clone();
            fv.extend_from_slice(&v_new);
            (fk, fv)
        } else {
            (k_new.clone(), v_new.clone())
        };
        let total_len = past_len + seq_len;

        // ── 4. Scaled dot-product attention ──────────────────────────────
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let gqa_factor = self.n_heads / self.n_kv_heads;

        // out: [seq_len × hidden_dim]
        let mut out = vec![0.0_f32; seq_len * self.hidden_dim];

        for t in 0..seq_len {
            // Absolute position of this query token.
            let abs_q_pos = past_len + t;

            for h in 0..self.n_heads {
                let kv_h = h / gqa_factor; // GQA: which KV head to use

                // Q vector for (t, h): q[t * hidden + h * head_dim .. + head_dim]
                let q_off = t * self.hidden_dim + h * self.head_dim;
                let q_vec = &q[q_off..q_off + self.head_dim];

                // Compute attention scores over full_k.
                let mut scores = vec![0.0_f32; total_len];
                for (kpos, sc) in scores.iter_mut().enumerate() {
                    // Causal mask: skip future positions.
                    if self.causal && kpos > abs_q_pos {
                        continue;
                    }
                    let k_off = kpos * kv_proj_dim + kv_h * self.head_dim;
                    let k_vec = &full_k[k_off..k_off + self.head_dim];
                    let dot: f32 = q_vec
                        .iter()
                        .zip(k_vec.iter())
                        .map(|(&qi, &ki)| qi * ki)
                        .sum();
                    *sc = dot * scale;
                }

                // Causal mask: set future positions to -∞.
                if self.causal {
                    for (kpos, sc) in scores.iter_mut().enumerate() {
                        if kpos > abs_q_pos {
                            *sc = f32::NEG_INFINITY;
                        }
                    }
                }

                // Numerically stable softmax.
                let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0_f32;
                let mut attn: Vec<f32> = scores
                    .iter()
                    .map(|&s| {
                        let e = (s - max_s).exp();
                        sum_exp += e;
                        e
                    })
                    .collect();
                if sum_exp > 0.0 {
                    for a in &mut attn {
                        *a /= sum_exp;
                    }
                }

                // Weighted sum of values.
                let out_off = t * self.hidden_dim + h * self.head_dim;
                for (kpos, &aw) in attn.iter().enumerate() {
                    if self.causal && kpos > abs_q_pos {
                        continue;
                    }
                    let v_off = kpos * kv_proj_dim + kv_h * self.head_dim;
                    let v_vec = &full_v[v_off..v_off + self.head_dim];
                    for (d, &vi) in v_vec.iter().enumerate() {
                        out[out_off + d] += aw * vi;
                    }
                }
            }
        }

        // ── 5. Output projection ──────────────────────────────────────────
        let out = linear_batch(&self.w_o, self.b_o.as_deref(), &out, seq_len)?;

        // ── 6. Build updated cache ────────────────────────────────────────
        let mut new_cache = LayerKvCache::new(self.n_kv_heads, self.head_dim);
        new_cache.keys = full_k;
        new_cache.values = full_v;
        new_cache.past_len = total_len;

        Ok((out, new_cache))
    }

    /// Residual-compatible forward: adds `x` to the attention output.
    ///
    /// Equivalent to `x + attn(x)` but avoids an extra allocation.
    pub fn forward_residual(
        &self,
        x: &[f32],
        seq_len: usize,
        past_kv: Option<&LayerKvCache>,
        rope: Option<&RotaryEmbedding>,
    ) -> LmResult<(Vec<f32>, LayerKvCache)> {
        let (attn_out, cache) = self.forward(x, seq_len, past_kv, rope)?;
        let mut out = x.to_vec();
        for (o, &a) in out.iter_mut().zip(attn_out.iter()) {
            *o += a;
        }
        Ok((out, cache))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mha(n_heads: usize, hidden_dim: usize) -> MultiHeadAttention {
        MultiHeadAttention::new(n_heads, n_heads, hidden_dim, true).unwrap()
    }

    // ── LayerKvCache ──────────────────────────────────────────────────────

    #[test]
    fn kv_cache_empty_on_init() {
        let c = LayerKvCache::new(4, 16);
        assert_eq!(c.past_len, 0);
        assert!(c.keys.is_empty());
    }

    #[test]
    fn kv_cache_append() {
        let mut c = LayerKvCache::new(2, 4);
        let k = vec![1.0_f32; 2 * 4]; // 1 token, 2 heads, head_dim=4
        c.append(&k, &k, 1);
        assert_eq!(c.past_len, 1);
        assert_eq!(c.keys.len(), 8);
    }

    // ── MultiHeadAttention ────────────────────────────────────────────────

    #[test]
    fn mha_zero_weights_zero_output() {
        let mha = make_mha(2, 4);
        let x = vec![1.0_f32; 4]; // 1 token
        let (out, _) = mha.forward(&x, 1, None, None).unwrap();
        // W_o = 0 → output must be 0
        assert!(out.iter().all(|&v| v.abs() < 1e-6), "out={out:?}");
    }

    #[test]
    fn mha_output_shape_single_token() {
        let mha = make_mha(2, 4);
        let x = vec![0.0_f32; 4];
        let (out, cache) = mha.forward(&x, 1, None, None).unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(cache.past_len, 1);
    }

    #[test]
    fn mha_output_shape_multi_token() {
        let mha = make_mha(2, 4);
        let x = vec![0.0_f32; 3 * 4];
        let (out, cache) = mha.forward(&x, 3, None, None).unwrap();
        assert_eq!(out.len(), 3 * 4);
        assert_eq!(cache.past_len, 3);
    }

    #[test]
    fn mha_kv_cache_extends() {
        let mha = make_mha(2, 4);
        let x1 = vec![0.0_f32; 4];
        let (_, cache1) = mha.forward(&x1, 1, None, None).unwrap();
        assert_eq!(cache1.past_len, 1);

        let x2 = vec![0.0_f32; 4];
        let (_, cache2) = mha.forward(&x2, 1, Some(&cache1), None).unwrap();
        assert_eq!(cache2.past_len, 2);
    }

    #[test]
    fn mha_gqa_forward_shape() {
        // 4 query heads, 2 KV heads (GQA with factor 2)
        let mha = MultiHeadAttention::new(4, 2, 8, true).unwrap();
        let x = vec![0.0_f32; 3 * 8]; // 3 tokens
        let (out, _) = mha.forward(&x, 3, None, None).unwrap();
        assert_eq!(out.len(), 3 * 8);
    }

    #[test]
    fn mha_gqa_head_mismatch_error() {
        // 5 KV heads does not divide 4 query heads
        assert!(MultiHeadAttention::new(4, 5, 8, true).is_err());
    }

    #[test]
    fn mha_invalid_config_error() {
        assert!(MultiHeadAttention::new(0, 1, 4, true).is_err());
    }

    #[test]
    fn mha_w_o_identity_propagates_value() {
        // W_q = I, W_k = I, W_v = I, W_o = I, no causal (x attends to itself fully)
        // With uniform Q=K=V=x and softmax → uniform attention → output ≈ x
        let mut mha = MultiHeadAttention::new(1, 1, 4, false).unwrap();
        mha.w_q = WeightTensor::eye(4, 4);
        mha.w_k = WeightTensor::eye(4, 4);
        mha.w_v = WeightTensor::eye(4, 4);
        mha.w_o = WeightTensor::eye(4, 4);
        // Single token: QK^T is a scalar → softmax → 1.0 → output = V = x
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let (out, _) = mha.forward(&x, 1, None, None).unwrap();
        for (&o, &xi) in out.iter().zip(x.iter()) {
            assert!((o - xi).abs() < 1e-5, "out={out:?} x={x:?}");
        }
    }

    #[test]
    fn mha_causal_mask_applied() {
        // With causal=true, token 0 only sees itself.
        // Token 1 sees tokens 0 and 1.
        // Use identity weights so we can trace values.
        let mut mha = MultiHeadAttention::new(1, 1, 4, true).unwrap();
        mha.w_q = WeightTensor::eye(4, 4);
        mha.w_k = WeightTensor::eye(4, 4);
        mha.w_v = WeightTensor::eye(4, 4);
        mha.w_o = WeightTensor::eye(4, 4);
        // x: token 0 = [1,0,0,0], token 1 = [0,2,0,0]
        let x = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (out, _) = mha.forward(&x, 2, None, None).unwrap();
        // Token 0 attends only to itself → out[0..4] ≈ [1,0,0,0]
        assert!((out[0] - 1.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!(out[1].abs() < 1e-5);
    }

    #[test]
    fn mha_empty_seq_error() {
        let mha = make_mha(2, 4);
        assert!(mha.forward(&[], 0, None, None).is_err());
    }

    #[test]
    fn mha_rope_applied_no_error() {
        let mha = make_mha(2, 4);
        let rope = RotaryEmbedding::new(2, 32, 10_000.0).unwrap();
        let x = vec![0.5_f32; 4];
        let result = mha.forward(&x, 1, None, Some(&rope));
        assert!(result.is_ok());
    }

    #[test]
    fn mha_incremental_vs_full_consistency() {
        // Full pass with 2 tokens should equal two incremental passes
        // (token-0 alone, then token-1 with cache).
        let mut mha = MultiHeadAttention::new(1, 1, 4, false).unwrap();
        mha.w_q = WeightTensor::eye(4, 4);
        mha.w_k = WeightTensor::eye(4, 4);
        mha.w_v = WeightTensor::eye(4, 4);
        mha.w_o = WeightTensor::eye(4, 4);
        let x0 = vec![1.0_f32, 0.0, 0.0, 0.0];
        let x1 = vec![0.0_f32, 1.0, 0.0, 0.0];
        let full_x = [x0.clone(), x1.clone()].concat();

        // Full 2-token pass
        let (out_full, _) = mha.forward(&full_x, 2, None, None).unwrap();

        // Incremental: token 0
        let (_, cache0) = mha.forward(&x0, 1, None, None).unwrap();
        // Incremental: token 1 with cache
        let (out_incr_1, _) = mha.forward(&x1, 1, Some(&cache0), None).unwrap();

        // Second-token outputs should match
        for (&full_v, &incr_v) in out_full[4..].iter().zip(out_incr_1.iter()) {
            assert!(
                (full_v - incr_v).abs() < 1e-4,
                "full={full_v} incr={incr_v}"
            );
        }
    }
}
