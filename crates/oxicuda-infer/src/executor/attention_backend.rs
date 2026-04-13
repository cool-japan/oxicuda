//! # PagedAttention CPU Backend
//!
//! Pure-Rust CPU reference implementation of the PagedAttention kernel.
//!
//! This module provides a numerically correct but non-optimised attention
//! computation over a paged KV cache.  It is used:
//!
//! * As a ground-truth reference for validating the GPU PTX kernel.
//! * Directly in unit tests that run without a CUDA device.
//!
//! ## Algorithm
//!
//! For each attention head `h` and query token:
//!
//! ```text
//! scores[t]  = dot(q[h, :], K[t, h, :]) * scale        (for t = 0..seq_len)
//! attn[t]    = softmax(scores)[t]
//! output[h, :] = Σ_t attn[t] * V[t, h, :]
//! ```
//!
//! Keys and values are retrieved from the paged cache using the block table,
//! which maps logical token positions to physical blocks.

use crate::cache::kv_cache::{BlockId, PagedKvCache};
use crate::error::{InferError, InferResult};

// ─── paged_attention_cpu ─────────────────────────────────────────────────────

/// CPU reference implementation of paged scaled dot-product attention.
///
/// # Parameters
///
/// * `q`           — query tensor `[n_heads, head_dim]` flattened.
/// * `kv_cache`    — the full paged KV cache.
/// * `block_table` — physical block IDs for this sequence (ordered).
/// * `seq_len`     — number of valid tokens in the KV cache for this sequence.
/// * `layer`       — which transformer layer to read K/V from.
/// * `n_heads`     — number of attention (query) heads.
/// * `n_kv_heads`  — number of KV heads (< n_heads for GQA).
/// * `head_dim`    — dimension per head.
/// * `block_size`  — tokens per KV block.
/// * `scale`       — attention scale factor (typically `1/√head_dim`).
///
/// # Returns
///
/// Output tensor `[n_heads, head_dim]` flattened.
///
/// # Errors
///
/// * [`InferError::DimensionMismatch`] — `q` has wrong length.
/// * [`InferError::EmptyBatch`]        — `seq_len == 0` or no blocks.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_cpu(
    q: &[f32],
    kv_cache: &PagedKvCache,
    block_table: &[BlockId],
    seq_len: usize,
    layer: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    scale: f32,
) -> InferResult<Vec<f32>> {
    let expected_q = n_heads * head_dim;
    if q.len() != expected_q {
        return Err(InferError::DimensionMismatch {
            expected: expected_q,
            got: q.len(),
        });
    }
    if seq_len == 0 || block_table.is_empty() {
        return Err(InferError::EmptyBatch);
    }
    if n_kv_heads == 0 || n_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(InferError::InvalidConfig(
            "n_heads must be divisible by n_kv_heads",
        ));
    }

    let gqa_ratio = n_heads / n_kv_heads; // how many Q heads share one KV head
    let mut output = vec![0.0_f32; n_heads * head_dim];

    // Process each attention head independently.
    for h in 0..n_heads {
        let kv_h = h / gqa_ratio; // corresponding KV head
        let q_off = h * head_dim;

        // Collect scores for all seq_len tokens.
        let mut scores = Vec::with_capacity(seq_len);
        for tok_idx in 0..seq_len {
            let blk_idx = tok_idx / block_size;
            let slot_idx = tok_idx % block_size;
            if blk_idx >= block_table.len() {
                break;
            }
            let block_id = block_table[blk_idx];
            let block = kv_cache.block(block_id, layer)?;
            if slot_idx >= block.filled {
                break;
            }
            let k_stride = kv_cache.n_kv_heads * head_dim;
            let k_base = slot_idx * k_stride + kv_h * head_dim;
            let k_slice = &block.keys[k_base..k_base + head_dim];

            // dot(q[h,:], k)
            let dot: f32 = q[q_off..q_off + head_dim]
                .iter()
                .zip(k_slice.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            scores.push(dot * scale);
        }

        if scores.is_empty() {
            continue;
        }

        // Stable softmax over scores.
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let exp_sum: f32 = exps.iter().sum();
        let inv_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
        exps.iter_mut().for_each(|e| *e *= inv_sum);

        // Weighted sum over values.
        let out_off = h * head_dim;
        for (tok_idx, &attn_weight) in exps.iter().enumerate() {
            let blk_idx = tok_idx / block_size;
            let slot_idx = tok_idx % block_size;
            if blk_idx >= block_table.len() {
                break;
            }
            let block_id = block_table[blk_idx];
            let block = kv_cache.block(block_id, layer)?;
            if slot_idx >= block.filled {
                break;
            }
            let v_stride = kv_cache.n_kv_heads * head_dim;
            let v_base = slot_idx * v_stride + kv_h * head_dim;
            let v_slice = &block.values[v_base..v_base + head_dim];

            for (o, &v) in output[out_off..out_off + head_dim]
                .iter_mut()
                .zip(v_slice.iter())
            {
                *o += attn_weight * v;
            }
        }
    }

    Ok(output)
}

// ─── AttentionConfig ─────────────────────────────────────────────────────────

/// Configuration for the attention backend.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
}

impl AttentionConfig {
    #[must_use]
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::kv_cache::PagedKvCache;
    use approx::assert_abs_diff_eq;

    fn tiny_cache_with_token(
        n_heads: usize,
        head_dim: usize,
        block_size: usize,
        kv_val: f32,
    ) -> (PagedKvCache, Vec<BlockId>) {
        // 1 layer, 1 KV head, head_dim, block_size=4
        let mut cache = PagedKvCache::new(1, n_heads, head_dim, block_size, 4);
        let id = cache.alloc_block().unwrap();
        // Write one token with all-kv_val K and V
        let kv = vec![kv_val; n_heads * head_dim];
        cache.append_token(id, 0, &kv, &kv).unwrap();
        (cache, vec![id])
    }

    #[test]
    fn single_token_output_equals_value() {
        // With one token, attention is trivially 1.0 on that token.
        // Output should equal V (which is all kv_val).
        let n_heads = 2;
        let head_dim = 4;
        let block_size = 4;
        let kv_val = 0.5_f32;
        let (cache, btable) = tiny_cache_with_token(n_heads, head_dim, block_size, kv_val);

        let q = vec![1.0_f32; n_heads * head_dim];
        let out = paged_attention_cpu(
            &q, &cache, &btable, 1, 0, n_heads, n_heads, head_dim, block_size, 1.0,
        )
        .unwrap();

        assert_eq!(out.len(), n_heads * head_dim);
        for &v in &out {
            assert_abs_diff_eq!(v, kv_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn two_token_attention_sums_to_v_weighted() {
        let n_heads = 1;
        let head_dim = 2;
        let block_size = 4;
        let mut cache = PagedKvCache::new(1, n_heads, head_dim, block_size, 4);
        let id = cache.alloc_block().unwrap();

        // Token 0: k=[1,0], v=[1,0]
        // Token 1: k=[0,1], v=[0,1]
        cache
            .append_token(id, 0, &[1.0_f32, 0.0], &[1.0_f32, 0.0])
            .unwrap();
        cache
            .append_token(id, 0, &[0.0_f32, 1.0], &[0.0_f32, 1.0])
            .unwrap();

        // Query q=[1,0]: dot(q, k0)=1 > dot(q, k1)=0 → mostly attend to token 0
        let q = vec![1.0_f32, 0.0];
        let out = paged_attention_cpu(
            &q,
            &cache,
            &[id],
            2,
            0,
            n_heads,
            n_heads,
            head_dim,
            block_size,
            1.0,
        )
        .unwrap();
        // Output should be close to V[0] = [1, 0] since q strongly attends to token 0
        assert!(
            out[0] > 0.5,
            "expected strong attention to token 0, got {}",
            out[0]
        );
        assert!(
            out[1] < 0.5,
            "expected weak attention to token 1, got {}",
            out[1]
        );
    }

    #[test]
    fn empty_seq_len_error() {
        let n_heads = 1;
        let head_dim = 2;
        let cache = PagedKvCache::new(1, n_heads, head_dim, 4, 4);
        let q = vec![1.0_f32; n_heads * head_dim];
        assert!(matches!(
            paged_attention_cpu(&q, &cache, &[], 0, 0, n_heads, n_heads, head_dim, 4, 1.0),
            Err(InferError::EmptyBatch)
        ));
    }

    #[test]
    fn wrong_q_length_error() {
        let n_heads = 2;
        let head_dim = 4;
        let mut cache = PagedKvCache::new(1, n_heads, head_dim, 4, 4);
        let id = cache.alloc_block().unwrap();
        let q = vec![0.0_f32; 3]; // wrong: should be n_heads * head_dim = 8
        assert!(matches!(
            paged_attention_cpu(&q, &cache, &[id], 1, 0, n_heads, n_heads, head_dim, 4, 1.0),
            Err(InferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn gqa_two_q_heads_one_kv_head() {
        // GQA: 2 Q heads, 1 KV head, head_dim=2
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 2;
        let block_size = 4;
        let mut cache = PagedKvCache::new(1, n_kv_heads, head_dim, block_size, 4);
        let id = cache.alloc_block().unwrap();
        // One token: K=[1,1], V=[2,2]
        cache
            .append_token(id, 0, &[1.0_f32, 1.0], &[2.0_f32, 2.0])
            .unwrap();
        // Query: [q0, q1] = [[1,0], [0,1]]
        let q = vec![1.0_f32, 0.0, 0.0, 1.0]; // [n_heads, head_dim]
        let out = paged_attention_cpu(
            &q,
            &cache,
            &[id],
            1,
            0,
            n_heads,
            n_kv_heads,
            head_dim,
            block_size,
            1.0,
        )
        .unwrap();
        // Both Q heads see the same KV, output = V[0] = [2,2] for each head
        assert_eq!(out.len(), n_heads * head_dim);
        for &v in &out {
            assert_abs_diff_eq!(v, 2.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn scale_factor_applied() {
        // With scale=0 (zero out all scores), output should be uniform over V
        // but zero scores → equal attention → output = mean(V).
        let n_heads = 1;
        let head_dim = 2;
        let block_size = 4;
        let mut cache = PagedKvCache::new(1, n_heads, head_dim, block_size, 4);
        let id = cache.alloc_block().unwrap();
        cache
            .append_token(id, 0, &[1.0_f32, 0.0], &[2.0_f32, 0.0])
            .unwrap();
        cache
            .append_token(id, 0, &[0.0_f32, 1.0], &[0.0_f32, 4.0])
            .unwrap();

        let q = vec![1.0_f32, 0.0];
        let out = paged_attention_cpu(
            &q,
            &cache,
            &[id],
            2,
            0,
            n_heads,
            n_heads,
            head_dim,
            block_size,
            0.0,
        )
        .unwrap();
        // scale=0: all scores=0 → uniform softmax (0.5 each)
        assert_abs_diff_eq!(out[0], 1.0, epsilon = 1e-5); // 0.5*2 + 0.5*0
        assert_abs_diff_eq!(out[1], 2.0, epsilon = 1e-5); // 0.5*0 + 0.5*4
    }
}
