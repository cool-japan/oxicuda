//! # oxicuda-dist-infer — Distributed GPU Inference Engine (Vol.12)
//!
//! Production-grade multi-GPU inference infrastructure for OxiCUDA.  Implements
//! three orthogonal parallelism strategies and the distributed KV-cache /
//! request-routing infrastructure needed to serve LLMs across GPU clusters.
//!
//! ## Parallelism axes
//!
//! | Axis | Degree | Description |
//! |------|--------|-------------|
//! | **TP** | `tp` | *Tensor parallelism* — shard weight matrices column- or row-wise |
//! | **SP** | `sp` | *Sequence parallelism* — partition the token sequence |
//! | **EP** | `ep` | *Expert parallelism* — partition MoE experts across GPUs |
//!
//! The three degrees multiply to `world_size = tp × sp × ep`.
//!
//! ## Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────────────┐
//!  │              RoutingPolicy                   │   ← request routing
//!  │  (RoundRobin | LeastLoaded | PrefixAffinity) │
//!  └──────────────────┬──────────────────────────┘
//!                     │ RoutingDecision
//!  ┌──────────────────▼──────────────────────────┐
//!  │            CachePartition                    │   ← distributed KV cache
//!  │     (sequence ownership + migration)         │
//!  └──────────────────┬──────────────────────────┘
//!                     │
//!        ┌────────────┼────────────┐
//!        │            │            │
//!   ColumnLinear  SeqSplitter  TopKRouter    ← per-rank compute
//!   RowLinear     Boundary     Dispatcher
//!  (tensor_par)  (seq_par)    (expert_par)
//! ```
//!
//! ## Crate layout
//!
//! ```text
//! src/
//! ├── error.rs                  DistInferError (27 variants), DistInferResult
//! ├── handle.rs                 DistInferHandle, ParallelismConfig, RankCoordinates
//! ├── ptx_kernels.rs            5 PTX kernel source strings
//! ├── tensor_parallel/
//! │   ├── column_parallel.rs    ColumnLinear, ColumnLinearShard
//! │   └── row_parallel.rs       RowLinear, RowLinearShard
//! ├── sequence_parallel/
//! │   ├── splitter.rs           SeqSplitter (extract/insert/all-gather/reduce-scatter)
//! │   └── boundary.rs           BoundaryExchange (pre-attn gather, post-attn scatter, QKV attn)
//! ├── expert_parallel/
//! │   ├── router.rs             TopKRouter (top-K selection + softmax weights)
//! │   └── dispatch.rs           ExpertDispatcher (scatter/gather + dispatch_and_gather)
//! ├── distributed_cache/
//! │   ├── partition.rs          CachePartition (least-loaded assign, grow, release, rebalance)
//! │   └── migration.rs          BlockMigrator (simulated P2P block migration)
//! └── router/
//!     ├── request.rs            Request, RoutingDecision, DispatchPolicy
//!     └── policy.rs             RoutingPolicy (RoundRobin/LeastLoaded/PrefixAffinity)
//! ```

pub mod distributed_cache;
pub mod error;
pub mod expert_parallel;
pub mod handle;
pub mod ptx_kernels;
pub mod router;
pub mod sequence_parallel;
pub mod tensor_parallel;

pub use error::{DistInferError, DistInferResult};
pub use handle::{DistInferHandle, ParallelismConfig, RankCoordinates, SmVersion};

// ─── Integration Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        distributed_cache::partition::CachePartition,
        expert_parallel::{dispatch::ExpertDispatcher, router::TopKRouter},
        router::{
            policy::{PolicyMode, RankLoad, RoutingPolicy},
            request::Request,
        },
        sequence_parallel::{boundary::BoundaryExchange, splitter::SeqSplitter},
        tensor_parallel::{column_parallel::ColumnLinear, row_parallel::RowLinear},
    };

    fn tp_handle(tp: usize, rank: usize) -> DistInferHandle {
        DistInferHandle::new(
            rank as i32,
            SmVersion(80),
            rank,
            ParallelismConfig { tp, sp: 1, ep: 1 },
        )
        .unwrap()
    }
    fn sp_handle(sp: usize, rank: usize) -> DistInferHandle {
        DistInferHandle::new(
            rank as i32,
            SmVersion(80),
            rank,
            ParallelismConfig { tp: 1, sp, ep: 1 },
        )
        .unwrap()
    }
    fn ep_handle(ep: usize, rank: usize) -> DistInferHandle {
        DistInferHandle::new(
            rank as i32,
            SmVersion(80),
            rank,
            ParallelismConfig { tp: 1, sp: 1, ep },
        )
        .unwrap()
    }

    // ── E2E-1: TP column + row roundtrip ─────────────────────────────────────

    #[test]
    fn e2e_tp_column_row_roundtrip() {
        // One attention projection (Q or K or V) followed by output projection.
        // Column-parallel → all-gather → row-parallel → all-reduce → output.
        let tp = 4;
        let in_f = 8;
        let hidden = 8;

        // Column-parallel: weight shape [hidden × in_f], split across tp=4
        // Use identity-ish weight for easy verification
        let col_weight: Vec<f32> = (0..hidden * in_f)
            .map(|i| if i / in_f == i % in_f { 1.0 } else { 0.0 })
            .collect();
        let input = vec![1.0_f32; in_f]; // batch=1

        let col_shards: Vec<Vec<f32>> = (0..tp)
            .map(|r| {
                let h = tp_handle(tp, r);
                let l = ColumnLinear::from_full_weight(h, in_f, hidden, &col_weight, None).unwrap();
                l.local_forward(&input, 1).unwrap()
            })
            .collect();
        let gathered = ColumnLinear::all_gather(hidden, 1, &col_shards).unwrap();
        // Identity linear on ones input → output = [1,1,1,1,1,1,1,1]
        assert_eq!(gathered, vec![1.0_f32; hidden]);

        // Row-parallel: weight shape [in_f × hidden], output projects back to in_f
        let row_weight: Vec<f32> = (0..in_f * hidden)
            .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
            .collect();
        let row_partials: Vec<Vec<f32>> = (0..tp)
            .map(|r| {
                let h = tp_handle(tp, r);
                let l = RowLinear::from_full_weight(h, in_f, hidden, &row_weight, None).unwrap();
                l.local_forward(&gathered, 1).unwrap()
            })
            .collect();
        let reduced = RowLinear::all_reduce(&row_partials).unwrap();
        // Identity × ones = ones
        assert_eq!(reduced, vec![1.0_f32; in_f]);
    }

    // ── E2E-2: SP all-gather → attention → reduce-scatter ────────────────────

    #[test]
    fn e2e_sp_attention_pipeline() {
        let sp = 2;
        let total_tokens = 4;
        let hidden_dim = 4;
        let n_heads = 2;

        // Build the full sequence as a counting tensor
        let full_seq: Vec<f32> = (0..total_tokens * hidden_dim).map(|i| i as f32).collect();

        // Extract chunks and simulate all-gather before attention
        let chunks: Vec<Vec<f32>> = (0..sp)
            .map(|r| {
                let h = sp_handle(sp, r);
                let s = SeqSplitter::new(h, total_tokens, hidden_dim).unwrap();
                s.extract_chunk(&full_seq).unwrap()
            })
            .collect();
        let regathered = SeqSplitter::all_gather(total_tokens, hidden_dim, sp, &chunks).unwrap();
        assert_eq!(
            regathered, full_seq,
            "all-gather must reconstruct the full sequence"
        );

        // Run BoundaryExchange local_attention for rank 0
        let bex =
            BoundaryExchange::new(sp_handle(sp, 0), total_tokens, hidden_dim, n_heads).unwrap();
        let q_chunk = &chunks[0];
        let kv_full = vec![1.0_f32; total_tokens * hidden_dim]; // uniform K/V
        let out = bex
            .local_attention(q_chunk, &kv_full, &kv_full, false)
            .unwrap();
        // Uniform softmax → output = weighted V = 1.0 everywhere
        assert_eq!(out.len(), bex.chunk_len() * hidden_dim);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-4, "expected 1.0, got {v}");
        }
    }

    // ── E2E-3: EP top-K routing + dispatch + gather ───────────────────────────

    #[test]
    fn e2e_ep_moe_dispatch_gather() {
        let ep = 2;
        let n_experts = 4;
        let hd = 4;
        let n_tokens = 4;

        // Route each token to top-1 expert round-robin style
        let router = TopKRouter::new(n_experts, 1).unwrap();
        let mut logits = vec![0.0_f32; n_tokens * n_experts];
        for t in 0..n_tokens {
            logits[t * n_experts + (t % n_experts)] = 1.0;
        }
        let plan = router.route(&logits, n_tokens).unwrap();

        // Token embeddings: token t = [t as f32; hd]
        let embeddings: Vec<f32> = (0..n_tokens).flat_map(|t| vec![t as f32; hd]).collect();

        // Rank 0 owns experts 0,1; run identity expert (passthrough × weight ≈ 1)
        let h0 = ep_handle(ep, 0);
        let dispatcher = ExpertDispatcher::new(h0, n_experts, hd).unwrap();
        let out = dispatcher
            .dispatch_and_gather(&embeddings, n_tokens, &plan, |batch| {
                Ok(batch.tokens.clone())
            })
            .unwrap();

        // Tokens 0 and 1 were routed to experts 0 and 1 (rank 0's experts).
        // After gather with weight=1, their output should equal their embedding.
        // Tokens 2,3 go to experts 2,3 (rank 1) → rank 0 gets zero output for them.
        for t in 0..2 {
            for d in 0..hd {
                let expected = t as f32;
                let got = out[t * hd + d];
                assert!(
                    (got - expected).abs() < 1e-5,
                    "token {t} feature {d}: expected {expected}, got {got}"
                );
            }
        }
    }

    // ── E2E-4: Cache partition assignment + release lifecycle ─────────────────

    #[test]
    fn e2e_cache_partition_lifecycle() {
        let ws = 4;
        let h = DistInferHandle::new(
            0,
            SmVersion(80),
            0,
            ParallelismConfig {
                tp: ws,
                sp: 1,
                ep: 1,
            },
        )
        .unwrap();
        let blocks_per_rank = vec![100usize; ws];
        let mut part = CachePartition::new(h, &blocks_per_rank, 0.2).unwrap();

        // Assign 8 sequences to 4 ranks
        let mut owners = Vec::new();
        for seq_id in 0..8 {
            let rank = part.assign(seq_id, 10).unwrap();
            owners.push((seq_id, rank));
        }
        // Each rank should have ~2 sequences (round-robin over equal load)
        let counts: Vec<usize> = (0..ws)
            .map(|r| owners.iter().filter(|(_, o)| *o == r).count())
            .collect();
        // All ranks should have at least 1 assignment
        for (r, &count) in counts.iter().enumerate() {
            assert!(count >= 1, "rank {r} has 0 sequences");
        }

        // Release all and verify blocks are returned
        for (seq_id, _) in &owners {
            part.release(*seq_id).unwrap();
        }
        for r in 0..ws {
            assert_eq!(
                part.stats()[r].free_blocks,
                100,
                "rank {r} should have all blocks free"
            );
        }
    }

    // ── E2E-5: Request routing prefix-affinity pipeline ───────────────────────

    #[test]
    fn e2e_routing_prefix_affinity_pipeline() {
        let n_ranks = 4;
        let free_blocks = 50;
        let loads: Vec<RankLoad> = (0..n_ranks)
            .map(|_| RankLoad {
                free_blocks,
                total_blocks: free_blocks,
                in_flight: 0,
            })
            .collect();

        let mut router = RoutingPolicy::new(n_ranks, PolicyMode::PrefixAffinity, loads, 5).unwrap();

        // First request: cache miss → least-loaded (any rank), registers prefix
        let req1 = Request {
            request_id: 1,
            token_ids: vec![1, 2, 3, 4, 5, 6],
            max_new_tokens: 32,
            priority: 0,
        };
        let dec1 = router.route(&req1).unwrap();
        assert!(!dec1.prefix_hit, "first request must be a cache miss");

        // Second request with same prefix → should hit the same rank
        let req2 = Request {
            request_id: 2,
            token_ids: vec![1, 2, 3, 4, 5, 99],
            max_new_tokens: 32,
            priority: 0,
        };
        let dec2 = router.route(&req2).unwrap();
        assert!(
            dec2.prefix_hit,
            "second request with same prefix should hit"
        );
        assert_eq!(dec2.rank, dec1.rank, "prefix hit should route to same rank");

        assert!(router.metrics().prefix_hit_rate() > 0.0);
    }

    // ── E2E-6: PTX kernels valid for all SM versions ───────────────────────────

    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        use crate::ptx_kernels::*;

        let sms = [
            SmVersion(75),
            SmVersion(80),
            SmVersion(90),
            SmVersion(100),
            SmVersion(120),
        ];
        for sm in sms {
            let kernels = [
                tp_col_scatter_ptx(sm),
                tp_row_all_reduce_ptx(sm),
                sp_seq_chunk_copy_ptx(sm),
                ep_token_scatter_ptx(sm),
                ep_token_gather_ptx(sm),
            ];
            for kernel in &kernels {
                assert!(
                    kernel.contains(".visible .entry"),
                    "sm={}: missing kernel entry",
                    sm.0
                );
                assert!(
                    kernel.contains(&format!(".target sm_{}", sm.0)),
                    "sm={}: wrong PTX target",
                    sm.0
                );
            }
        }
    }
}
