//! Row-parallel linear layer.
//!
//! Splits the input-feature dimension across `tp` ranks.  Each rank holds a
//! shard of weight shape `[out × local_in]` where `local_in = in / tp`.
//!
//! ## Compute pattern
//!
//! 1. Rank `i` receives input slice `x[:, i*(in/tp) : (i+1)*(in/tp)]`.
//! 2. **Local GEMM**: `partial_i = x_slice @ shard_weight^T`  shape `[batch × out]`
//! 3. **All-reduce** (sum): `y = Σ_i partial_i`  (ring all-reduce).
//! 4. Bias is added **once** only on rank 0 (avoid double-counting).

use crate::error::{DistInferError, DistInferResult};
use crate::handle::DistInferHandle;

// ─── RowLinearShard ──────────────────────────────────────────────────────────

/// Weight shard for a row-parallel linear layer.
///
/// Weight shape: `[out_features × local_in]` (row-major).
#[derive(Debug, Clone)]
pub struct RowLinearShard {
    /// Total output features (identical across all shards).
    pub out_features: usize,
    /// Total input features (before sharding).
    pub total_in: usize,
    /// Number of input features handled by this shard.
    pub local_in: usize,
    /// Column offset in the input dimension: `tp_rank * local_in`.
    pub row_offset: usize,
    /// Weight data `[out_features × local_in]` row-major.
    pub weight: Vec<f32>,
    /// Bias `[out_features]` — non-empty only on rank 0.
    pub bias: Vec<f32>,
}

impl RowLinearShard {
    /// Check size invariants.
    pub fn validate(&self) -> DistInferResult<()> {
        let exp_w = self.out_features * self.local_in;
        if self.weight.len() != exp_w {
            return Err(DistInferError::ShardShapeMismatch {
                rows: self.out_features,
                cols: self.local_in,
                exp_rows: self.out_features,
                exp_cols: self.local_in,
            });
        }
        // bias is either full [out_features] or empty (all other ranks)
        if !self.bias.is_empty() && self.bias.len() != self.out_features {
            return Err(DistInferError::DimensionMismatch {
                expected: self.out_features,
                got: self.bias.len(),
            });
        }
        Ok(())
    }

    /// Weight element at `(out_row, in_col)`.
    #[inline]
    fn w(&self, out_row: usize, in_col: usize) -> f32 {
        self.weight[out_row * self.local_in + in_col]
    }

    /// Compute the local partial output for `input_slice` of shape
    /// `[batch × local_in]`.  Returns `[batch × out_features]`.
    pub fn forward_partial(&self, input_slice: &[f32], batch: usize) -> DistInferResult<Vec<f32>> {
        if input_slice.len() != batch * self.local_in {
            return Err(DistInferError::DimensionMismatch {
                expected: batch * self.local_in,
                got: input_slice.len(),
            });
        }
        let mut partial = vec![0.0_f32; batch * self.out_features];
        for b in 0..batch {
            for o in 0..self.out_features {
                let mut acc = 0.0_f32;
                for i in 0..self.local_in {
                    acc += input_slice[b * self.local_in + i] * self.w(o, i);
                }
                // Add bias on rank 0 only (determined by non-empty bias field)
                if !self.bias.is_empty() {
                    acc += self.bias[o];
                }
                partial[b * self.out_features + o] = acc;
            }
        }
        Ok(partial)
    }
}

// ─── RowLinear ───────────────────────────────────────────────────────────────

/// A row-parallel linear layer — owns one shard and can simulate the full
/// forward pass (local GEMM + all-reduce).
#[derive(Debug, Clone)]
pub struct RowLinear {
    /// Session handle.
    pub handle: DistInferHandle,
    /// Local weight shard.
    pub shard: RowLinearShard,
}

impl RowLinear {
    /// Construct a row-parallel layer by slicing the full weight matrix.
    ///
    /// `weight` must be the **full** weight matrix `[out × total_in]`.
    /// Rank `r` receives columns `[r*local_in .. (r+1)*local_in]` of each row.
    pub fn from_full_weight(
        handle: DistInferHandle,
        out_features: usize,
        total_in: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> DistInferResult<Self> {
        let tp = handle.config.tp;
        if total_in % tp != 0 {
            return Err(DistInferError::TpInputMisaligned {
                features: total_in,
                degree: tp,
            });
        }
        let local_in = total_in / tp;
        let row_offset = handle.tp_rank() * local_in;

        if weight.len() != out_features * total_in {
            return Err(DistInferError::DimensionMismatch {
                expected: out_features * total_in,
                got: weight.len(),
            });
        }

        // Slice the in-columns for this rank from every row
        let mut shard_weight = Vec::with_capacity(out_features * local_in);
        for o in 0..out_features {
            let row_start = o * total_in + row_offset;
            shard_weight.extend_from_slice(&weight[row_start..row_start + local_in]);
        }

        // Bias only on rank 0 to avoid double-counting
        let shard_bias = if handle.tp_rank() == 0 {
            match bias {
                Some(b) => {
                    if b.len() != out_features {
                        return Err(DistInferError::DimensionMismatch {
                            expected: out_features,
                            got: b.len(),
                        });
                    }
                    b.to_vec()
                }
                None => vec![],
            }
        } else {
            vec![]
        };

        let shard = RowLinearShard {
            out_features,
            total_in,
            local_in,
            row_offset,
            weight: shard_weight,
            bias: shard_bias,
        };
        shard.validate()?;
        Ok(Self { handle, shard })
    }

    /// Extract this rank's input slice from the full input tensor.
    ///
    /// Full input shape: `[batch × total_in]`.
    /// Returns the local slice: `[batch × local_in]`.
    pub fn slice_input(&self, full_input: &[f32], batch: usize) -> DistInferResult<Vec<f32>> {
        let total_in = self.shard.total_in;
        let local_in = self.shard.local_in;
        let offset = self.shard.row_offset;
        if full_input.len() != batch * total_in {
            return Err(DistInferError::DimensionMismatch {
                expected: batch * total_in,
                got: full_input.len(),
            });
        }
        let mut slice = Vec::with_capacity(batch * local_in);
        for b in 0..batch {
            let row_start = b * total_in + offset;
            slice.extend_from_slice(&full_input[row_start..row_start + local_in]);
        }
        Ok(slice)
    }

    /// Full forward pass: slice input → local GEMM → return partial.
    /// Caller must all-reduce across ranks (see `all_reduce`).
    pub fn local_forward(&self, full_input: &[f32], batch: usize) -> DistInferResult<Vec<f32>> {
        let slice = self.slice_input(full_input, batch)?;
        self.shard.forward_partial(&slice, batch)
    }

    /// Simulate ring all-reduce by summing partials from all ranks.
    ///
    /// In production this is done via NVLink P2P + PTX kernel; here we sum.
    pub fn all_reduce(partials: &[Vec<f32>]) -> DistInferResult<Vec<f32>> {
        if partials.is_empty() {
            return Err(DistInferError::TooFewRanks {
                needed: 1,
                world_size: 0,
            });
        }
        let len = partials[0].len();
        let mut acc = partials[0].clone();
        for part in &partials[1..] {
            if part.len() != len {
                return Err(DistInferError::DimensionMismatch {
                    expected: len,
                    got: part.len(),
                });
            }
            for (a, &p) in acc.iter_mut().zip(part.iter()) {
                *a += p;
            }
        }
        Ok(acc)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::{DistInferHandle, ParallelismConfig, SmVersion};

    fn make_handle(tp: usize, rank: usize) -> DistInferHandle {
        DistInferHandle::new(
            rank as i32,
            SmVersion(80),
            rank,
            ParallelismConfig { tp, sp: 1, ep: 1 },
        )
        .unwrap()
    }

    #[test]
    fn row_linear_identity_weight() {
        // tp=2, in=4, out=4 → each rank handles 2 input columns
        let tp = 2;
        let out = 4;
        let total_in = 4;
        // Identity weight 4×4
        let weight: Vec<f32> = (0..out * total_in)
            .map(|i| {
                if i / total_in == i % total_in {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let input = vec![1.0_f32, 2.0, 3.0, 4.0]; // batch=1

        let partials: Vec<Vec<f32>> = (0..tp)
            .map(|rank| {
                let h = make_handle(tp, rank);
                let layer = RowLinear::from_full_weight(h, out, total_in, &weight, None).unwrap();
                layer.local_forward(&input, 1).unwrap()
            })
            .collect();

        let result = RowLinear::all_reduce(&partials).unwrap();
        // Identity × [1,2,3,4] = [1,2,3,4]
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn row_linear_bias_on_rank0_only() {
        let tp = 2;
        let out = 2;
        let total_in = 2;
        // Zero weight
        let weight = vec![0.0_f32; out * total_in];
        let bias = vec![5.0_f32, 7.0];
        let input = vec![0.0_f32, 0.0];

        let partials: Vec<Vec<f32>> = (0..tp)
            .map(|rank| {
                let h = make_handle(tp, rank);
                let layer =
                    RowLinear::from_full_weight(h, out, total_in, &weight, Some(&bias)).unwrap();
                layer.local_forward(&input, 1).unwrap()
            })
            .collect();

        // rank 0 adds bias [5,7]; rank 1 does not. Sum = [5,7]
        let result = RowLinear::all_reduce(&partials).unwrap();
        assert_eq!(result, vec![5.0, 7.0]);
    }

    #[test]
    fn row_linear_tp4_ones_weight() {
        let tp = 4;
        let out = 3;
        let total_in = 4;
        // Ones weight (3×4)
        let weight = vec![1.0_f32; out * total_in];
        // input: batch=2, each token = [1,1,1,1]
        let input = vec![1.0_f32; 2 * total_in];

        let partials: Vec<Vec<f32>> = (0..tp)
            .map(|rank| {
                let h = make_handle(tp, rank);
                let layer = RowLinear::from_full_weight(h, out, total_in, &weight, None).unwrap();
                layer.local_forward(&input, 2).unwrap()
            })
            .collect();

        let result = RowLinear::all_reduce(&partials).unwrap();
        // Each output = sum of total_in ones = 4.0, 2 tokens, 3 outputs
        assert_eq!(result, vec![4.0; 2 * out]);
    }

    #[test]
    fn row_linear_input_misaligned() {
        let h = make_handle(3, 0); // total_in=4 % tp=3 ≠ 0
        let w = vec![0.0_f32; 4 * 4];
        let err = RowLinear::from_full_weight(h, 4, 4, &w, None).unwrap_err();
        assert!(matches!(
            err,
            DistInferError::TpInputMisaligned {
                features: 4,
                degree: 3
            }
        ));
    }

    #[test]
    fn slice_input_correctness() {
        let tp = 2;
        let h0 = make_handle(tp, 0);
        let h1 = make_handle(tp, 1);
        let w = vec![0.0_f32; 4 * 4];
        let l0 = RowLinear::from_full_weight(h0, 4, 4, &w, None).unwrap();
        let l1 = RowLinear::from_full_weight(h1, 4, 4, &w, None).unwrap();
        // input: batch=1, [10, 20, 30, 40]
        let full = vec![10.0_f32, 20.0, 30.0, 40.0];
        let s0 = l0.slice_input(&full, 1).unwrap();
        let s1 = l1.slice_input(&full, 1).unwrap();
        assert_eq!(s0, vec![10.0, 20.0]);
        assert_eq!(s1, vec![30.0, 40.0]);
    }

    #[test]
    fn all_reduce_empty_partials_errors() {
        let err = RowLinear::all_reduce(&[]).unwrap_err();
        assert!(matches!(
            err,
            DistInferError::TooFewRanks {
                needed: 1,
                world_size: 0
            }
        ));
    }
}
