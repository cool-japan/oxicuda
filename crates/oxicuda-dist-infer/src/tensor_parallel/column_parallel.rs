//! Column-parallel linear layer.
//!
//! Splits the output-feature dimension across `tp` ranks.  Each rank holds a
//! shard of shape `[local_out × in]` where `local_out = out / tp`.
//!
//! ## Compute pattern
//!
//! 1. **Local GEMM**: `partial = input @ shard_weight^T + shard_bias`  (shape `[batch × local_out]`)
//! 2. **All-gather**:  Concatenate along column dim to produce `[batch × out]`.
//!
//! The all-gather is simulated in CPU-side reference code (no real CUDA driver),
//! making the entire module pure Rust and testable without a GPU.

use crate::error::{DistInferError, DistInferResult};
use crate::handle::DistInferHandle;

// ─── ColumnLinearShard ──────────────────────────────────────────────────────

/// The weight + bias shard owned by one rank in a column-parallel linear layer.
///
/// Shapes:
/// * `weight`: `[local_out × in]`   (row-major)
/// * `bias`:   `[local_out]`         (optional; zeroes when absent)
#[derive(Debug, Clone)]
pub struct ColumnLinearShard {
    /// Number of input features (shared across all shards).
    pub in_features: usize,
    /// Number of output features handled by this shard.
    pub local_out: usize,
    /// Column offset of this shard in the full output dimension.
    pub col_offset: usize,
    /// Total output features across all ranks.
    pub total_out: usize,
    /// Row-major weight data `[local_out × in_features]`.
    pub weight: Vec<f32>,
    /// Per-output bias `[local_out]`.  All-zeros when the layer has no bias.
    pub bias: Vec<f32>,
}

impl ColumnLinearShard {
    /// Validate invariants: sizes must match declared dimensions.
    pub fn validate(&self) -> DistInferResult<()> {
        let exp_w = self.local_out * self.in_features;
        if self.weight.len() != exp_w {
            return Err(DistInferError::ShardShapeMismatch {
                rows: self.local_out,
                cols: self.in_features,
                exp_rows: self.local_out,
                exp_cols: self.in_features,
            });
        }
        if self.bias.len() != self.local_out {
            return Err(DistInferError::DimensionMismatch {
                expected: self.local_out,
                got: self.bias.len(),
            });
        }
        if self.col_offset + self.local_out > self.total_out {
            return Err(DistInferError::TpFeaturesMisaligned {
                features: self.col_offset + self.local_out,
                degree: self.total_out,
            });
        }
        Ok(())
    }

    /// Element of the weight matrix at `(row, col)`.
    #[inline]
    fn w(&self, row: usize, col: usize) -> f32 {
        self.weight[row * self.in_features + col]
    }

    /// Compute `output = input @ weight^T + bias` for the local shard.
    ///
    /// `input` shape: `[batch × in_features]` (row-major).
    /// Output shape: `[batch × local_out]` (row-major).
    pub fn forward(&self, input: &[f32], batch: usize) -> DistInferResult<Vec<f32>> {
        if input.len() != batch * self.in_features {
            return Err(DistInferError::DimensionMismatch {
                expected: batch * self.in_features,
                got: input.len(),
            });
        }
        let mut out = vec![0.0_f32; batch * self.local_out];
        for b in 0..batch {
            for o in 0..self.local_out {
                let mut acc = self.bias[o];
                for i in 0..self.in_features {
                    acc += input[b * self.in_features + i] * self.w(o, i);
                }
                out[b * self.local_out + o] = acc;
            }
        }
        Ok(out)
    }
}

// ─── ColumnLinear ────────────────────────────────────────────────────────────

/// A column-parallel linear layer — owns one shard and implements the full
/// forward pass (local GEMM + simulated all-gather).
#[derive(Debug, Clone)]
pub struct ColumnLinear {
    /// Configuration for this rank.
    pub handle: DistInferHandle,
    /// Local weight/bias shard.
    pub shard: ColumnLinearShard,
}

impl ColumnLinear {
    /// Construct a column-parallel layer.
    ///
    /// `weight` must be the **full** weight matrix `[total_out × in]`; this
    /// constructor slices the appropriate shard for `handle.tp_rank()`.
    pub fn from_full_weight(
        handle: DistInferHandle,
        in_features: usize,
        total_out: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> DistInferResult<Self> {
        let tp = handle.config.tp;
        if total_out % tp != 0 {
            return Err(DistInferError::TpFeaturesMisaligned {
                features: total_out,
                degree: tp,
            });
        }
        let local_out = total_out / tp;
        let col_offset = handle.tp_rank() * local_out;

        // Slice weight rows [col_offset .. col_offset+local_out]
        let w_start = col_offset * in_features;
        let w_end = w_start + local_out * in_features;
        if weight.len() != total_out * in_features {
            return Err(DistInferError::DimensionMismatch {
                expected: total_out * in_features,
                got: weight.len(),
            });
        }
        let shard_weight = weight[w_start..w_end].to_vec();

        let shard_bias = match bias {
            Some(b) => {
                if b.len() != total_out {
                    return Err(DistInferError::DimensionMismatch {
                        expected: total_out,
                        got: b.len(),
                    });
                }
                b[col_offset..col_offset + local_out].to_vec()
            }
            None => vec![0.0_f32; local_out],
        };

        let shard = ColumnLinearShard {
            in_features,
            local_out,
            col_offset,
            total_out,
            weight: shard_weight,
            bias: shard_bias,
        };
        shard.validate()?;
        Ok(Self { handle, shard })
    }

    /// Run the column-parallel forward pass.
    ///
    /// Returns the full output after simulated all-gather — in production each
    /// rank would all-gather over NVLink; here we return only the local shard
    /// and a separate `all_gather` helper reconstructs the full result.
    pub fn local_forward(&self, input: &[f32], batch: usize) -> DistInferResult<Vec<f32>> {
        self.shard.forward(input, batch)
    }

    /// Simulate an all-gather across `tp` ranks.
    ///
    /// `shards[r]` is the output of `local_forward` for rank `r`.
    /// Returns the concatenated full output `[batch × total_out]`.
    pub fn all_gather(
        total_out: usize,
        batch: usize,
        shards: &[Vec<f32>],
    ) -> DistInferResult<Vec<f32>> {
        let tp = shards.len();
        if tp == 0 {
            return Err(DistInferError::TooFewRanks {
                needed: 1,
                world_size: 0,
            });
        }
        let local_out = total_out / tp;
        let mut out = vec![0.0_f32; batch * total_out];
        for (rank, shard_out) in shards.iter().enumerate() {
            let col_offset = rank * local_out;
            for b in 0..batch {
                for o in 0..local_out {
                    out[b * total_out + col_offset + o] = shard_out[b * local_out + o];
                }
            }
        }
        Ok(out)
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
    fn column_linear_identity_weight() {
        // tp=2, in=4, out=4 → each rank owns 2 out columns
        let tp = 2;
        let in_f = 4;
        let total_out = 4;
        // Weight = identity matrix (4×4)
        let weight: Vec<f32> = (0..total_out * in_f)
            .map(|i| if i / in_f == i % in_f { 1.0 } else { 0.0 })
            .collect();

        let shards: Vec<Vec<f32>> = (0..tp)
            .map(|rank| {
                let h = make_handle(tp, rank);
                let layer =
                    ColumnLinear::from_full_weight(h, in_f, total_out, &weight, None).unwrap();
                let input = vec![1.0_f32, 2.0, 3.0, 4.0]; // batch=1
                layer.local_forward(&input, 1).unwrap()
            })
            .collect();

        // All-gather
        let out = ColumnLinear::all_gather(total_out, 1, &shards).unwrap();
        // Identity forward should give back the input
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn column_linear_batch2() {
        let tp = 4;
        let in_f = 2;
        let total_out = 4;
        // Ones weight matrix (out×in)
        let weight = vec![1.0_f32; total_out * in_f];
        // input: 2 tokens, each [1, 1]
        let input = vec![1.0_f32, 1.0, 1.0, 1.0]; // batch=2

        let shards: Vec<Vec<f32>> = (0..tp)
            .map(|rank| {
                let h = make_handle(tp, rank);
                let layer =
                    ColumnLinear::from_full_weight(h, in_f, total_out, &weight, None).unwrap();
                layer.local_forward(&input, 2).unwrap()
            })
            .collect();

        let out = ColumnLinear::all_gather(total_out, 2, &shards).unwrap();
        // Each output column = sum of in_f ones inputs = 2.0 per token
        assert_eq!(out, vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn column_linear_bias_added() {
        let tp = 2;
        let in_f = 2;
        let total_out = 2;
        // Zero weight, bias = [10, 20]
        let weight = vec![0.0_f32; total_out * in_f];
        let bias = vec![10.0_f32, 20.0];
        let input = vec![0.0_f32, 0.0]; // batch=1

        let shards: Vec<Vec<f32>> = (0..tp)
            .map(|rank| {
                let h = make_handle(tp, rank);
                let layer =
                    ColumnLinear::from_full_weight(h, in_f, total_out, &weight, Some(&bias))
                        .unwrap();
                layer.local_forward(&input, 1).unwrap()
            })
            .collect();

        let out = ColumnLinear::all_gather(total_out, 1, &shards).unwrap();
        assert_eq!(out, vec![10.0, 20.0]);
    }

    #[test]
    fn column_linear_misaligned_out_features_errors() {
        let h = make_handle(3, 0); // tp=3 does not divide out=4
        let w = vec![0.0_f32; 4 * 2];
        let err = ColumnLinear::from_full_weight(h, 2, 4, &w, None).unwrap_err();
        assert!(matches!(
            err,
            DistInferError::TpFeaturesMisaligned {
                features: 4,
                degree: 3
            }
        ));
    }

    #[test]
    fn shard_validate_bad_weight_len() {
        let shard = ColumnLinearShard {
            in_features: 4,
            local_out: 2,
            col_offset: 0,
            total_out: 4,
            weight: vec![0.0; 5], // wrong: should be 8
            bias: vec![0.0; 2],
        };
        assert!(shard.validate().is_err());
    }

    #[test]
    fn all_gather_empty_shards_errors() {
        let err = ColumnLinear::all_gather(4, 1, &[]).unwrap_err();
        assert!(matches!(
            err,
            DistInferError::TooFewRanks {
                needed: 1,
                world_size: 0
            }
        ));
    }
}
