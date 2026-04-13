//! Top-K expert router for MoE inference.
//!
//! Given gating logits of shape `[n_tokens × n_experts]`, selects the top-K
//! experts per token, computes softmax weights over the selected K, and
//! produces a routing plan used by `ExpertDispatcher`.

use crate::error::{DistInferError, DistInferResult};

// ─── RoutingEntry ────────────────────────────────────────────────────────────

/// Assignment of one token to one expert.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutingEntry {
    /// Source token index.
    pub token_idx: usize,
    /// Target expert index.
    pub expert_idx: usize,
    /// Routing weight (softmax probability of this expert given top-K logits).
    pub weight: f32,
}

// ─── RoutingPlan ─────────────────────────────────────────────────────────────

/// Complete routing plan for a batch of tokens.
#[derive(Debug, Clone)]
pub struct RoutingPlan {
    /// One entry per (token, expert) assignment; length = n_tokens * top_k.
    pub entries: Vec<RoutingEntry>,
    /// Total number of tokens routed.
    pub n_tokens: usize,
    /// Number of available experts.
    pub n_experts: usize,
    /// K experts selected per token.
    pub top_k: usize,
    /// For each expert, how many tokens it receives (token load).
    pub expert_load: Vec<usize>,
}

impl RoutingPlan {
    /// Returns the cumulative slot offset for expert `e` in a flat buffer.
    ///
    /// Tokens assigned to expert 0 occupy slots `[0..expert_load[0])`,
    /// expert 1 occupies `[expert_load[0]..expert_load[0]+expert_load[1])`,
    /// etc.
    pub fn expert_offset(&self, e: usize) -> usize {
        self.expert_load[..e].iter().sum()
    }

    /// Total dispatched slots = n_tokens * top_k.
    pub fn total_slots(&self) -> usize {
        self.entries.len()
    }
}

// ─── TopKRouter ──────────────────────────────────────────────────────────────

/// Computes expert routing using top-K selection + softmax normalisation.
///
/// This implements the SwitchTransformer / Mixtral routing logic:
/// 1. Select top-K experts per token based on gating logit.
/// 2. Softmax over the K selected logits → routing weights.
/// 3. (Optionally) apply auxiliary load-balance penalty (not part of inference path).
#[derive(Debug, Clone)]
pub struct TopKRouter {
    /// Total number of experts.
    pub n_experts: usize,
    /// Number of experts selected per token.
    pub top_k: usize,
    /// Noise added to logits to break ties (set to 0 for deterministic routing).
    pub jitter_noise: f32,
}

impl TopKRouter {
    /// Construct a top-K router.
    pub fn new(n_experts: usize, top_k: usize) -> DistInferResult<Self> {
        if top_k == 0 || top_k > n_experts {
            return Err(DistInferError::EpExpertsMisaligned {
                n_experts,
                degree: top_k,
            });
        }
        Ok(Self {
            n_experts,
            top_k,
            jitter_noise: 0.0,
        })
    }

    /// Compute top-K routing plan from gating logits.
    ///
    /// `logits` shape: `[n_tokens × n_experts]` row-major.
    ///
    /// Returns a `RoutingPlan` with one entry per (token, expert) pair.
    pub fn route(&self, logits: &[f32], n_tokens: usize) -> DistInferResult<RoutingPlan> {
        if logits.len() != n_tokens * self.n_experts {
            return Err(DistInferError::DimensionMismatch {
                expected: n_tokens * self.n_experts,
                got: logits.len(),
            });
        }

        let mut entries = Vec::with_capacity(n_tokens * self.top_k);
        let mut expert_load = vec![0usize; self.n_experts];

        for tok in 0..n_tokens {
            let row = &logits[tok * self.n_experts..(tok + 1) * self.n_experts];

            // Find top-K expert indices by sorting descending
            let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_k_indices = &indexed[..self.top_k];

            // Softmax over the top-K logits
            let max_l = top_k_indices
                .iter()
                .map(|&(_, v)| v)
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = top_k_indices
                .iter()
                .map(|&(_, v)| (v - max_l).exp())
                .collect();
            let sum_exp: f32 = exp_vals.iter().sum();
            let sum_exp = if sum_exp > 0.0 { sum_exp } else { 1.0 };

            for (idx, (&(expert_idx, _), &exp_v)) in
                top_k_indices.iter().zip(exp_vals.iter()).enumerate()
            {
                let _ = idx;
                let weight = exp_v / sum_exp;
                entries.push(RoutingEntry {
                    token_idx: tok,
                    expert_idx,
                    weight,
                });
                expert_load[expert_idx] += 1;
            }
        }

        Ok(RoutingPlan {
            entries,
            n_tokens,
            n_experts: self.n_experts,
            top_k: self.top_k,
            expert_load,
        })
    }

    /// Load balance metric: coefficient of variation of expert loads (lower = better).
    pub fn load_balance_cv(plan: &RoutingPlan) -> f32 {
        let loads: Vec<f32> = plan.expert_load.iter().map(|&l| l as f32).collect();
        let n = loads.len() as f32;
        if n == 0.0 {
            return 0.0;
        }
        let mean = loads.iter().sum::<f32>() / n;
        if mean == 0.0 {
            return 0.0;
        }
        let var = loads.iter().map(|&l| (l - mean).powi(2)).sum::<f32>() / n;
        var.sqrt() / mean
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top1_routing_selects_max_logit() {
        // 3 tokens, 4 experts, top_k=1
        let router = TopKRouter::new(4, 1).unwrap();
        let logits = vec![
            0.0_f32, 1.0, 0.5, -1.0, // token 0 → expert 1
            2.0, 0.0, 0.0, 0.0, // token 1 → expert 0
            -1.0, -2.0, 3.0, 0.0, // token 2 → expert 2
        ];
        let plan = router.route(&logits, 3).unwrap();
        assert_eq!(plan.entries.len(), 3);
        assert_eq!(plan.entries[0].expert_idx, 1);
        assert_eq!(plan.entries[1].expert_idx, 0);
        assert_eq!(plan.entries[2].expert_idx, 2);
    }

    #[test]
    fn top2_routing_gives_two_entries_per_token() {
        let router = TopKRouter::new(4, 2).unwrap();
        let logits = vec![1.0_f32, 2.0, 0.0, -1.0]; // 1 token, 4 experts
        let plan = router.route(&logits, 1).unwrap();
        assert_eq!(plan.entries.len(), 2);
        // Top-2 are expert 1 (logit=2) and expert 0 (logit=1)
        let experts: Vec<usize> = plan.entries.iter().map(|e| e.expert_idx).collect();
        assert!(experts.contains(&1), "expert 1 must be selected");
        assert!(experts.contains(&0), "expert 0 must be selected");
    }

    #[test]
    fn routing_weights_sum_to_one() {
        let router = TopKRouter::new(8, 3).unwrap();
        let logits: Vec<f32> = (0..8).map(|i| i as f32).collect(); // 1 token
        let plan = router.route(&logits, 1).unwrap();
        let sum: f32 = plan.entries.iter().map(|e| e.weight).sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights must sum to 1, got {sum}");
    }

    #[test]
    fn expert_load_counts_correctly() {
        let router = TopKRouter::new(2, 1).unwrap();
        // 4 tokens: expert 0 gets tokens 0,2; expert 1 gets tokens 1,3
        let logits = vec![
            1.0_f32, 0.0, // tok0 → exp0
            0.0, 1.0, // tok1 → exp1
            1.0, 0.0, // tok2 → exp0
            0.0, 1.0, // tok3 → exp1
        ];
        let plan = router.route(&logits, 4).unwrap();
        assert_eq!(plan.expert_load[0], 2);
        assert_eq!(plan.expert_load[1], 2);
        assert_eq!(plan.expert_offset(0), 0);
        assert_eq!(plan.expert_offset(1), 2);
    }

    #[test]
    fn load_balance_cv_perfectly_balanced_is_zero() {
        let router = TopKRouter::new(4, 1).unwrap();
        // 4 tokens, each going to a different expert
        let logits = vec![
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let plan = router.route(&logits, 4).unwrap();
        let cv = TopKRouter::load_balance_cv(&plan);
        assert!(cv < 1e-6, "perfectly balanced should have cv=0, got {cv}");
    }

    #[test]
    fn invalid_top_k_errors() {
        assert!(TopKRouter::new(4, 0).is_err());
        assert!(TopKRouter::new(4, 5).is_err()); // k > n_experts
    }

    #[test]
    fn logit_dimension_mismatch_errors() {
        let router = TopKRouter::new(4, 1).unwrap();
        let err = router.route(&[0.0; 3], 2).unwrap_err(); // expects 8 elements
        assert!(matches!(
            err,
            DistInferError::DimensionMismatch {
                expected: 8,
                got: 3
            }
        ));
    }
}
