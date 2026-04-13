//! Boundary exchange for sequence-parallel attention layers.
//!
//! At the boundary of an attention layer each rank needs the full sequence
//! context (for causal or full attention).  The `BoundaryExchange` type manages:
//!
//! 1. **All-gather** before attention: collect all `sp` local key/value chunks
//!    into a full K/V sequence per rank (so every rank can compute global attention).
//! 2. **Reduce-scatter** after attention: sum output partials across ranks and
//!    scatter back so each rank holds only its local output chunk.
//!
//! Communication is host-simulated (no real GPU transfer), making the code
//! runnable in pure-Rust test environments.

use crate::error::{DistInferError, DistInferResult};
use crate::handle::DistInferHandle;

// ─── BoundaryExchange ────────────────────────────────────────────────────────

/// Manages pre-attention all-gather and post-attention reduce-scatter.
#[derive(Debug, Clone)]
pub struct BoundaryExchange {
    handle: DistInferHandle,
    /// Total tokens in the full sequence.
    total_tokens: usize,
    /// Tokens per rank chunk.
    chunk_len: usize,
    /// Embedding / hidden dimension.
    hidden_dim: usize,
    /// Number of attention heads.
    n_heads: usize,
    /// Dimension per attention head (head_dim = hidden_dim / n_heads).
    head_dim: usize,
}

impl BoundaryExchange {
    /// Construct a boundary exchange manager.
    ///
    /// `total_tokens` must be divisible by `sp`.  `hidden_dim` must be
    /// divisible by `n_heads`.
    pub fn new(
        handle: DistInferHandle,
        total_tokens: usize,
        hidden_dim: usize,
        n_heads: usize,
    ) -> DistInferResult<Self> {
        let sp = handle.config.sp;
        if total_tokens % sp != 0 {
            return Err(DistInferError::SpSeqLenMisaligned {
                seq_len: total_tokens,
                degree: sp,
            });
        }
        if n_heads == 0 || hidden_dim % n_heads != 0 {
            return Err(DistInferError::DimensionMismatch {
                expected: 0, // placeholder: n_heads must divide hidden_dim
                got: hidden_dim % n_heads,
            });
        }
        let chunk_len = total_tokens / sp;
        let head_dim = hidden_dim / n_heads;
        Ok(Self {
            handle,
            total_tokens,
            chunk_len,
            hidden_dim,
            n_heads,
            head_dim,
        })
    }

    /// **Pre-attention all-gather**: collect KV chunks from all sp ranks.
    ///
    /// `local_kv` has shape `[chunk_len × hidden_dim]` for one K or V matrix.
    /// `all_local_kvs[r]` is rank `r`'s local KV chunk.
    ///
    /// Returns the full KV matrix `[total_tokens × hidden_dim]`.
    pub fn pre_attn_all_gather(&self, all_local_kvs: &[Vec<f32>]) -> DistInferResult<Vec<f32>> {
        let sp = self.handle.config.sp;
        if all_local_kvs.len() != sp {
            return Err(DistInferError::DimensionMismatch {
                expected: sp,
                got: all_local_kvs.len(),
            });
        }
        let expected_chunk = self.chunk_len * self.hidden_dim;
        let mut full = vec![0.0_f32; self.total_tokens * self.hidden_dim];
        for (rank, kv) in all_local_kvs.iter().enumerate() {
            if kv.len() != expected_chunk {
                return Err(DistInferError::DimensionMismatch {
                    expected: expected_chunk,
                    got: kv.len(),
                });
            }
            let start = rank * expected_chunk;
            full[start..start + expected_chunk].copy_from_slice(kv);
        }
        Ok(full)
    }

    /// **Post-attention reduce-scatter**: sum output contributions and scatter.
    ///
    /// `partials[r]` is rank `r`'s full-sequence output partial from the
    /// attention layer (shape `[total_tokens × hidden_dim]`).
    ///
    /// Returns this rank's local output chunk `[chunk_len × hidden_dim]`.
    pub fn post_attn_reduce_scatter(&self, partials: &[Vec<f32>]) -> DistInferResult<Vec<f32>> {
        let full_elems = self.total_tokens * self.hidden_dim;
        let sp = self.handle.config.sp;
        if partials.len() != sp {
            return Err(DistInferError::DimensionMismatch {
                expected: sp,
                got: partials.len(),
            });
        }
        // Sum all partials
        let mut acc = vec![0.0_f32; full_elems];
        for part in partials {
            if part.len() != full_elems {
                return Err(DistInferError::DimensionMismatch {
                    expected: full_elems,
                    got: part.len(),
                });
            }
            for (a, &p) in acc.iter_mut().zip(part.iter()) {
                *a += p;
            }
        }
        // Scatter: return only this rank's chunk
        let start = self.handle.sp_rank() * self.chunk_len * self.hidden_dim;
        let end = start + self.chunk_len * self.hidden_dim;
        Ok(acc[start..end].to_vec())
    }

    /// Compute a scaled dot-product attention on the **local query chunk**
    /// against the **globally all-gathered** full K and V sequences.
    ///
    /// This is the core of sequence-parallel attention: each rank holds a
    /// chunk of the query sequence but attends to all key/values.
    ///
    /// # Arguments
    ///
    /// * `q_chunk`  — `[chunk_len × hidden_dim]` query embeddings for this rank.
    /// * `k_full`   — `[total_tokens × hidden_dim]` full key sequence.
    /// * `v_full`   — `[total_tokens × hidden_dim]` full value sequence.
    /// * `causal`   — if `true`, tokens can only attend to earlier positions
    ///   (using the token's global position = `sp_rank*chunk_len + local_i`).
    ///
    /// Returns `[chunk_len × hidden_dim]` attended outputs.
    pub fn local_attention(
        &self,
        q_chunk: &[f32],
        k_full: &[f32],
        v_full: &[f32],
        causal: bool,
    ) -> DistInferResult<Vec<f32>> {
        let cl = self.chunk_len;
        let hd = self.hidden_dim;
        let tt = self.total_tokens;
        let nh = self.n_heads;
        let dh = self.head_dim;
        let scale = 1.0_f32 / (dh as f32).sqrt();

        if q_chunk.len() != cl * hd {
            return Err(DistInferError::DimensionMismatch {
                expected: cl * hd,
                got: q_chunk.len(),
            });
        }
        if k_full.len() != tt * hd {
            return Err(DistInferError::DimensionMismatch {
                expected: tt * hd,
                got: k_full.len(),
            });
        }
        if v_full.len() != tt * hd {
            return Err(DistInferError::DimensionMismatch {
                expected: tt * hd,
                got: v_full.len(),
            });
        }

        let q_start_tok = self.handle.sp_rank() * cl; // global start token index
        let mut out = vec![0.0_f32; cl * hd];

        for local_q in 0..cl {
            let global_q = q_start_tok + local_q;
            for h in 0..nh {
                // Q vector for this (local_q, head): q_chunk[(local_q*nh + h)*dh ..]
                let q_off = (local_q * nh + h) * dh;
                // Compute attention scores against all key positions
                let mut scores = vec![0.0_f32; tt];
                for (kpos, score) in scores.iter_mut().enumerate() {
                    if causal && kpos > global_q {
                        continue;
                    }
                    let k_off = (kpos * nh + h) * dh;
                    let mut dot = 0.0_f32;
                    for d in 0..dh {
                        dot += q_chunk[q_off + d] * k_full[k_off + d];
                    }
                    *score = dot * scale;
                }
                // Softmax (numerically stable)
                if causal {
                    // Mask future positions with -inf
                    for (kpos, sc) in scores.iter_mut().enumerate() {
                        if kpos > global_q {
                            *sc = f32::NEG_INFINITY;
                        }
                    }
                }
                let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0_f32;
                let mut exp_s: Vec<f32> = scores
                    .iter()
                    .map(|&s| {
                        let e = (s - max_s).exp();
                        sum_exp += e;
                        e
                    })
                    .collect();
                if sum_exp > 0.0 {
                    for e in &mut exp_s {
                        *e /= sum_exp;
                    }
                }
                // Weighted sum of V
                let out_off = (local_q * nh + h) * dh;
                for (kpos, &attn_w) in exp_s.iter().enumerate() {
                    if causal && kpos > global_q {
                        continue;
                    }
                    let v_off = (kpos * nh + h) * dh;
                    for d in 0..dh {
                        out[out_off + d] += attn_w * v_full[v_off + d];
                    }
                }
            }
        }
        Ok(out)
    }

    /// Accessors
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }
    pub fn chunk_len(&self) -> usize {
        self.chunk_len
    }
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::{DistInferHandle, ParallelismConfig, SmVersion};

    fn handle_sp(sp: usize, rank: usize) -> DistInferHandle {
        DistInferHandle::new(
            rank as i32,
            SmVersion(80),
            rank,
            ParallelismConfig { tp: 1, sp, ep: 1 },
        )
        .unwrap()
    }

    fn bex(sp: usize, rank: usize, total: usize, hd: usize, nh: usize) -> BoundaryExchange {
        BoundaryExchange::new(handle_sp(sp, rank), total, hd, nh).unwrap()
    }

    #[test]
    fn pre_attn_all_gather_correctness() {
        let sp = 2;
        let total = 4;
        let hd = 4;
        let nh = 2;
        // Each rank's chunk is filled with its rank index as float
        let chunks: Vec<Vec<f32>> = (0..sp).map(|r| vec![r as f32; (total / sp) * hd]).collect();
        let b = bex(sp, 0, total, hd, nh);
        let full = b.pre_attn_all_gather(&chunks).unwrap();
        // First half = 0s, second half = 1s
        let chunk_elems = (total / sp) * hd;
        assert_eq!(&full[..chunk_elems], chunks[0].as_slice());
        assert_eq!(&full[chunk_elems..], chunks[1].as_slice());
    }

    #[test]
    fn post_attn_reduce_scatter_correctness() {
        let sp = 4;
        let total = 4;
        let hd = 2;
        let nh = 2;
        // All partials = ones (total × hd)
        let part = vec![1.0_f32; total * hd];
        let partials = vec![part; sp];
        let b = bex(sp, 2, total, hd, nh); // rank 2
        let local = b.post_attn_reduce_scatter(&partials).unwrap();
        // sum of sp=4 ones per element = 4.0, chunk_len=1, hd=2
        assert_eq!(local, vec![4.0_f32; hd]);
    }

    #[test]
    fn local_attention_uniform_qkv_gives_avg_v() {
        // Q=K=V = ones, no causal mask → softmax is uniform → output = V avg = 1
        let sp = 2;
        let total = 4;
        let hd = 2;
        let nh = 2;
        let b = bex(sp, 0, total, hd, nh);
        let cl = b.chunk_len();
        let q = vec![1.0_f32; cl * hd];
        let kv = vec![1.0_f32; total * hd];
        let out = b.local_attention(&q, &kv, &kv, false).unwrap();
        // Uniform softmax → output = weighted sum of V = 1.0 * V = 1.0 per dim
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-5, "expected ≈1.0, got {v}");
        }
    }

    #[test]
    fn local_attention_causal_mask_rank0() {
        // With causal mask, token 0 can only attend to position 0.
        // Q[0]=K[0]=V[0]=1, K[1]=V[1]=2 (but masked).
        // Expected: output for token 0 = V[0] = 1.
        let sp = 2;
        let total = 4;
        let hd = 2;
        let nh = 1;
        let b = bex(sp, 0, total, hd, nh);
        let cl = b.chunk_len(); // 2
        let q = vec![1.0_f32; cl * hd];
        let mut k = vec![0.0_f32; total * hd];
        let mut v = vec![0.0_f32; total * hd];
        // Fill K[0] and V[0] with 1.0
        k[..hd].iter_mut().for_each(|x| *x = 1.0);
        v[..hd].iter_mut().for_each(|x| *x = 1.0);
        // K[1] = V[1] = 2.0 (causal should mask this for token 0)
        k[hd..2 * hd].iter_mut().for_each(|x| *x = 2.0);
        v[hd..2 * hd].iter_mut().for_each(|x| *x = 2.0);
        let out = b.local_attention(&q, &k, &v, true).unwrap();
        // Token 0 attends only to K[0]/V[0] → output = 1.0
        for &o in &out[..hd] {
            assert!((o - 1.0).abs() < 1e-5, "causal token0 got {o}");
        }
    }

    #[test]
    fn pre_attn_wrong_chunk_count_errors() {
        let b = bex(4, 0, 8, 4, 2);
        let three_chunks: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0_f32; 2 * 4]).collect();
        let err = b.pre_attn_all_gather(&three_chunks).unwrap_err();
        assert!(matches!(
            err,
            DistInferError::DimensionMismatch {
                expected: 4,
                got: 3
            }
        ));
    }
}
