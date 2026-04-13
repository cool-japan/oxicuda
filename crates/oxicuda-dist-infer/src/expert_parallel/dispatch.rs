//! Expert dispatcher — all-to-all scatter/gather for MoE inference.
//!
//! The `ExpertDispatcher` owns one rank's subset of experts.  It:
//!
//! 1. Receives a `RoutingPlan` (from `TopKRouter`).
//! 2. Scatters tokens to expert-local input buffers (all-to-all dispatch).
//! 3. Runs a user-provided expert function on each local expert's token batch.
//! 4. Gathers results back to the original token order (all-to-all gather).
//!
//! Communication is host-simulated; in production the all-to-all would use
//! NVLink P2P + PTX kernels.

use crate::error::{DistInferError, DistInferResult};
use crate::expert_parallel::router::RoutingPlan;
use crate::handle::DistInferHandle;

// ─── LocalExpertBatch ────────────────────────────────────────────────────────

/// A batch of tokens dispatched to one local expert.
#[derive(Debug, Clone)]
pub struct LocalExpertBatch {
    /// Expert index (global numbering).
    pub expert_idx: usize,
    /// Token embeddings `[n × hidden_dim]` (row-major).
    pub tokens: Vec<f32>,
    /// Number of tokens.
    pub n_tokens: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// For each row in `tokens`, the original global token index.
    pub token_indices: Vec<usize>,
    /// For each row, the routing weight.
    pub weights: Vec<f32>,
}

// ─── ExpertDispatcher ────────────────────────────────────────────────────────

/// Orchestrates token dispatch and result gather for expert-parallel inference.
#[derive(Debug, Clone)]
pub struct ExpertDispatcher {
    handle: DistInferHandle,
    /// Total number of experts across all ranks.
    pub n_experts: usize,
    /// Hidden dimension of token embeddings.
    pub hidden_dim: usize,
    /// Experts owned by this rank.
    pub local_expert_range: std::ops::Range<usize>,
}

impl ExpertDispatcher {
    /// Construct a dispatcher for rank `r` in an `ep`-way expert-parallel group.
    ///
    /// `n_experts` must be divisible by `ep`.
    pub fn new(
        handle: DistInferHandle,
        n_experts: usize,
        hidden_dim: usize,
    ) -> DistInferResult<Self> {
        let ep = handle.config.ep;
        if n_experts % ep != 0 {
            return Err(DistInferError::EpExpertsMisaligned {
                n_experts,
                degree: ep,
            });
        }
        let experts_per_rank = n_experts / ep;
        let ep_rank = handle.ep_rank();
        let start = ep_rank * experts_per_rank;
        let end = start + experts_per_rank;
        Ok(Self {
            handle,
            n_experts,
            hidden_dim,
            local_expert_range: start..end,
        })
    }

    /// This dispatcher's rank.
    pub fn rank(&self) -> usize {
        self.handle.global_rank()
    }

    /// Number of experts on this rank.
    pub fn n_local_experts(&self) -> usize {
        self.local_expert_range.len()
    }

    /// Whether `expert_idx` is handled by this rank.
    pub fn owns_expert(&self, expert_idx: usize) -> bool {
        self.local_expert_range.contains(&expert_idx)
    }

    /// Scatter token embeddings to local expert input buffers.
    ///
    /// `token_embeddings` shape: `[n_tokens × hidden_dim]` (row-major).
    /// Returns one `LocalExpertBatch` per locally owned expert that has at
    /// least one token dispatched to it.
    pub fn scatter(
        &self,
        token_embeddings: &[f32],
        n_tokens: usize,
        plan: &RoutingPlan,
    ) -> DistInferResult<Vec<LocalExpertBatch>> {
        let hd = self.hidden_dim;
        if token_embeddings.len() != n_tokens * hd {
            return Err(DistInferError::DimensionMismatch {
                expected: n_tokens * hd,
                got: token_embeddings.len(),
            });
        }

        // Build one buffer per local expert
        let mut expert_tokens: Vec<Vec<f32>> = vec![vec![]; self.n_local_experts()];
        let mut expert_tidxs: Vec<Vec<usize>> = vec![vec![]; self.n_local_experts()];
        let mut expert_wts: Vec<Vec<f32>> = vec![vec![]; self.n_local_experts()];

        for entry in &plan.entries {
            if !self.owns_expert(entry.expert_idx) {
                continue;
            }
            let local_idx = entry.expert_idx - self.local_expert_range.start;
            let tok_start = entry.token_idx * hd;
            expert_tokens[local_idx]
                .extend_from_slice(&token_embeddings[tok_start..tok_start + hd]);
            expert_tidxs[local_idx].push(entry.token_idx);
            expert_wts[local_idx].push(entry.weight);
        }

        let batches: Vec<LocalExpertBatch> = expert_tokens
            .into_iter()
            .zip(expert_tidxs)
            .zip(expert_wts)
            .enumerate()
            .filter_map(|(li, ((toks, tidxs), wts))| {
                if tidxs.is_empty() {
                    return None;
                }
                let global_expert = self.local_expert_range.start + li;
                Some(LocalExpertBatch {
                    expert_idx: global_expert,
                    n_tokens: tidxs.len(),
                    hidden_dim: hd,
                    tokens: toks,
                    token_indices: tidxs,
                    weights: wts,
                })
            })
            .collect();

        Ok(batches)
    }

    /// Gather expert outputs back to the original token order.
    ///
    /// `batches` must be the processed `LocalExpertBatch` list from `scatter`
    /// (with `tokens` replaced by the expert's output).
    /// `n_tokens` is the global token count.
    ///
    /// Returns `[n_tokens × hidden_dim]` combined output.  Each token's output
    /// is the sum of `weight * expert_output` for each expert it was routed to
    /// (weighted combination following MoE convention).
    pub fn gather(
        &self,
        batches: &[LocalExpertBatch],
        n_tokens: usize,
    ) -> DistInferResult<Vec<f32>> {
        let hd = self.hidden_dim;
        let mut out = vec![0.0_f32; n_tokens * hd];

        for batch in batches {
            if batch.tokens.len() != batch.n_tokens * hd {
                return Err(DistInferError::DimensionMismatch {
                    expected: batch.n_tokens * hd,
                    got: batch.tokens.len(),
                });
            }
            for (slot, (&tok_idx, &wt)) in batch
                .token_indices
                .iter()
                .zip(batch.weights.iter())
                .enumerate()
            {
                let src = slot * hd;
                let dst = tok_idx * hd;
                for d in 0..hd {
                    out[dst + d] += wt * batch.tokens[src + d];
                }
            }
        }
        Ok(out)
    }

    /// Run `expert_fn` for each local expert batch and gather results.
    ///
    /// `expert_fn(batch) -> Ok(output_tokens)` where `output_tokens` has the
    /// same shape as `batch.tokens` (`[n × hd]`).
    pub fn dispatch_and_gather<F>(
        &self,
        token_embeddings: &[f32],
        n_tokens: usize,
        plan: &RoutingPlan,
        mut expert_fn: F,
    ) -> DistInferResult<Vec<f32>>
    where
        F: FnMut(&LocalExpertBatch) -> DistInferResult<Vec<f32>>,
    {
        let mut batches = self.scatter(token_embeddings, n_tokens, plan)?;
        for batch in &mut batches {
            let output = expert_fn(batch)?;
            batch.tokens = output;
        }
        self.gather(&batches, n_tokens)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_parallel::router::TopKRouter;
    use crate::handle::{DistInferHandle, ParallelismConfig, SmVersion};

    fn make_handle(ep: usize, rank: usize) -> DistInferHandle {
        DistInferHandle::new(
            rank as i32,
            SmVersion(80),
            rank,
            ParallelismConfig { tp: 1, sp: 1, ep },
        )
        .unwrap()
    }

    #[test]
    fn scatter_routes_tokens_to_correct_expert() {
        // ep=2, n_experts=4, rank 0 owns experts 0,1; rank 1 owns 2,3
        let ep = 2;
        let n_experts = 4;
        let hd = 2;
        let h0 = make_handle(ep, 0);
        let dispatcher = ExpertDispatcher::new(h0, n_experts, hd).unwrap();

        let router = TopKRouter::new(n_experts, 1).unwrap();
        // 2 tokens: token 0 → expert 0, token 1 → expert 2 (remote)
        let logits = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let plan = router.route(&logits, 2).unwrap();

        let embeddings = vec![1.0_f32, 2.0, 3.0, 4.0]; // [t0=[1,2], t1=[3,4]]
        let batches = dispatcher.scatter(&embeddings, 2, &plan).unwrap();

        // Only token 0 → expert 0 is local; token 1 → expert 2 is remote
        assert_eq!(
            batches.len(),
            1,
            "only 1 local expert should receive tokens"
        );
        assert_eq!(batches[0].expert_idx, 0);
        assert_eq!(batches[0].n_tokens, 1);
        assert_eq!(batches[0].tokens, vec![1.0, 2.0]);
    }

    #[test]
    fn dispatch_and_gather_identity_expert() {
        let ep = 1;
        let n_experts = 2;
        let hd = 3;
        let h = make_handle(ep, 0);
        let dispatcher = ExpertDispatcher::new(h, n_experts, hd).unwrap();

        let router = TopKRouter::new(n_experts, 1).unwrap();
        // 2 tokens: token 0 → expert 0, token 1 → expert 1
        let logits = vec![1.0_f32, 0.0, 0.0, 1.0];
        let plan = router.route(&logits, 2).unwrap();

        let embeddings: Vec<f32> = (0..2 * hd).map(|i| i as f32).collect();

        let out = dispatcher
            .dispatch_and_gather(&embeddings, 2, &plan, |batch| {
                // Identity expert: pass through tokens unchanged
                Ok(batch.tokens.clone())
            })
            .unwrap();

        // After gather with routing weights ≈ 1.0, output ≈ input
        for (i, (&o, &e)) in out.iter().zip(embeddings.iter()).enumerate() {
            assert!(
                (o - e).abs() < 1e-5,
                "mismatch at {i}: got {o} expected {e}"
            );
        }
    }

    #[test]
    fn gather_applies_routing_weights() {
        let ep = 1;
        let n_experts = 2;
        let hd = 1;
        let h = make_handle(ep, 0);
        let dispatcher = ExpertDispatcher::new(h, n_experts, hd).unwrap();

        // Manual batch for token 0 routed to expert 0 with weight 0.5
        let batch = LocalExpertBatch {
            expert_idx: 0,
            n_tokens: 1,
            hidden_dim: 1,
            tokens: vec![10.0], // expert output
            token_indices: vec![0],
            weights: vec![0.5],
        };
        let out = dispatcher.gather(&[batch], 1).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-6, "0.5 * 10 = 5, got {}", out[0]);
    }

    #[test]
    fn dispatcher_wrong_expert_count_errors() {
        let h = make_handle(3, 0);
        let err = ExpertDispatcher::new(h, 5, 4).unwrap_err(); // 5 % 3 ≠ 0
        assert!(matches!(
            err,
            DistInferError::EpExpertsMisaligned {
                n_experts: 5,
                degree: 3
            }
        ));
    }

    #[test]
    fn owns_expert_boundary_check() {
        let ep = 4;
        let n_experts = 8;
        let h2 = make_handle(ep, 2);
        let d = ExpertDispatcher::new(h2, n_experts, 4).unwrap();
        // Rank 2 of 4 owns experts [4, 5]
        assert!(!d.owns_expert(3));
        assert!(d.owns_expert(4));
        assert!(d.owns_expert(5));
        assert!(!d.owns_expert(6));
    }
}
