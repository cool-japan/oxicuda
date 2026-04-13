//! Routing policy implementations.
//!
//! `RoutingPolicy` selects a target rank for an incoming request.

use std::collections::HashMap;

use crate::error::{DistInferError, DistInferResult};
use crate::router::request::{DispatchPolicy, Request, RoutingDecision};

// ─── RankLoad ────────────────────────────────────────────────────────────────

/// Per-rank load state used by the router.
#[derive(Debug, Clone, Copy)]
pub struct RankLoad {
    /// Free KV-cache blocks remaining.
    pub free_blocks: usize,
    /// Total block capacity.
    pub total_blocks: usize,
    /// In-flight request count on this rank.
    pub in_flight: usize,
}

impl RankLoad {
    /// Block utilization in [0.0, 1.0].
    pub fn utilization(&self) -> f32 {
        if self.total_blocks == 0 {
            return 1.0;
        }
        let used = self.total_blocks - self.free_blocks;
        used as f32 / self.total_blocks as f32
    }
}

// ─── RouterMetrics ───────────────────────────────────────────────────────────

/// Cumulative routing statistics.
#[derive(Debug, Clone, Default)]
pub struct RouterMetrics {
    /// Requests routed by each policy.
    pub round_robin_count: u64,
    pub least_loaded_count: u64,
    pub prefix_affinity_count: u64,
    /// Total routing decisions made.
    pub total_routed: u64,
    /// Prefix cache hits.
    pub prefix_hits: u64,
}

impl RouterMetrics {
    /// Prefix hit rate in [0, 1].
    pub fn prefix_hit_rate(&self) -> f32 {
        if self.total_routed == 0 {
            return 0.0;
        }
        self.prefix_hits as f32 / self.total_routed as f32
    }
}

// ─── RoutingPolicy ───────────────────────────────────────────────────────────

/// Multi-policy request router.
///
/// The chosen mode determines how target ranks are selected:
///
/// * `RoundRobin` — ignores load, cycles rank 0..N-1.
/// * `LeastLoaded` — selects rank with most `free_blocks`.
/// * `PrefixAffinity` — checks the prefix hash map for a rank with a matching
///   entry; falls back to `LeastLoaded` on miss.
///
/// All modes reject requests if all ranks are at capacity
/// (`free_blocks == 0`).
#[derive(Debug)]
pub struct RoutingPolicy {
    world_size: usize,
    mode: PolicyMode,
    /// Current rank loads (caller must update before each route call).
    loads: Vec<RankLoad>,
    /// Prefix hash → rank with warm cache.
    prefix_map: HashMap<u64, usize>,
    /// How many leading tokens to use for prefix hashing.
    prefix_len: usize,
    /// Round-robin counter.
    rr_counter: usize,
    /// Accumulated metrics.
    metrics: RouterMetrics,
}

/// Active routing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyMode {
    RoundRobin,
    LeastLoaded,
    PrefixAffinity,
}

impl RoutingPolicy {
    /// Create a new router with the given mode.
    ///
    /// `loads` must have `world_size` entries.
    pub fn new(
        world_size: usize,
        mode: PolicyMode,
        loads: Vec<RankLoad>,
        prefix_len: usize,
    ) -> DistInferResult<Self> {
        if world_size == 0 {
            return Err(DistInferError::TooFewRanks {
                needed: 1,
                world_size: 0,
            });
        }
        if loads.len() != world_size {
            return Err(DistInferError::DimensionMismatch {
                expected: world_size,
                got: loads.len(),
            });
        }
        Ok(Self {
            world_size,
            mode,
            loads,
            prefix_map: HashMap::new(),
            prefix_len,
            rr_counter: 0,
            metrics: RouterMetrics::default(),
        })
    }

    /// Update rank load snapshot (call before `route` to reflect current state).
    pub fn update_load(&mut self, rank: usize, load: RankLoad) -> DistInferResult<()> {
        if rank >= self.world_size {
            return Err(DistInferError::RankOutOfRange {
                rank,
                world_size: self.world_size,
            });
        }
        self.loads[rank] = load;
        Ok(())
    }

    /// Register a prefix hash → rank mapping (used by PrefixAffinity mode).
    pub fn register_prefix(&mut self, token_hash: u64, rank: usize) -> DistInferResult<()> {
        if rank >= self.world_size {
            return Err(DistInferError::RankOutOfRange {
                rank,
                world_size: self.world_size,
            });
        }
        self.prefix_map.insert(token_hash, rank);
        Ok(())
    }

    /// Remove a stale prefix entry (e.g., after a block eviction).
    pub fn evict_prefix(&mut self, token_hash: u64) {
        self.prefix_map.remove(&token_hash);
    }

    /// Route `req` to a rank according to the active policy.
    pub fn route(&mut self, req: &Request) -> DistInferResult<RoutingDecision> {
        if req.token_ids.is_empty() {
            return Err(DistInferError::EmptyTokenSequence);
        }
        // Check capacity across all ranks
        if self.loads.iter().all(|l| l.free_blocks == 0) {
            return Err(DistInferError::AllRanksAtCapacity {
                n_ranks: self.world_size,
            });
        }

        let decision = match self.mode {
            PolicyMode::RoundRobin => self.route_round_robin(req)?,
            PolicyMode::LeastLoaded => self.route_least_loaded(req)?,
            PolicyMode::PrefixAffinity => self.route_prefix_affinity(req)?,
        };

        // Update metrics
        self.metrics.total_routed += 1;
        match decision.policy_used {
            DispatchPolicy::RoundRobin => self.metrics.round_robin_count += 1,
            DispatchPolicy::LeastLoaded => self.metrics.least_loaded_count += 1,
            DispatchPolicy::PrefixAffinity => self.metrics.prefix_affinity_count += 1,
        }
        if decision.prefix_hit {
            self.metrics.prefix_hits += 1;
        }

        Ok(decision)
    }

    fn route_round_robin(&mut self, _req: &Request) -> DistInferResult<RoutingDecision> {
        // Skip ranks at full capacity
        for _ in 0..self.world_size {
            let rank = self.rr_counter % self.world_size;
            self.rr_counter += 1;
            if self.loads[rank].free_blocks > 0 {
                return Ok(RoutingDecision {
                    rank,
                    policy_used: DispatchPolicy::RoundRobin,
                    prefix_hit: false,
                });
            }
        }
        Err(DistInferError::AllRanksAtCapacity {
            n_ranks: self.world_size,
        })
    }

    fn route_least_loaded(&mut self, _req: &Request) -> DistInferResult<RoutingDecision> {
        let rank = self
            .loads
            .iter()
            .enumerate()
            .filter(|(_, l)| l.free_blocks > 0)
            .max_by_key(|(_, l)| l.free_blocks)
            .map(|(r, _)| r)
            .ok_or(DistInferError::AllRanksAtCapacity {
                n_ranks: self.world_size,
            })?;
        Ok(RoutingDecision {
            rank,
            policy_used: DispatchPolicy::LeastLoaded,
            prefix_hit: false,
        })
    }

    fn route_prefix_affinity(&mut self, req: &Request) -> DistInferResult<RoutingDecision> {
        let hash = req.prefix_hash(self.prefix_len);
        if let Some(&rank) = self.prefix_map.get(&hash) {
            if self.loads[rank].free_blocks > 0 {
                return Ok(RoutingDecision {
                    rank,
                    policy_used: DispatchPolicy::PrefixAffinity,
                    prefix_hit: true,
                });
            }
        }
        // Fallback to least-loaded
        let mut decision = self.route_least_loaded(req)?;
        decision.policy_used = DispatchPolicy::PrefixAffinity;
        // Register this mapping for future requests with the same prefix
        self.prefix_map.insert(hash, decision.rank);
        Ok(decision)
    }

    /// Access accumulated metrics.
    pub fn metrics(&self) -> &RouterMetrics {
        &self.metrics
    }

    /// Current load snapshots.
    pub fn loads(&self) -> &[RankLoad] {
        &self.loads
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn loads(n: usize, free: usize) -> Vec<RankLoad> {
        (0..n)
            .map(|_| RankLoad {
                free_blocks: free,
                total_blocks: free,
                in_flight: 0,
            })
            .collect()
    }

    fn req(id: u64, tokens: Vec<u32>) -> Request {
        Request {
            request_id: id,
            token_ids: tokens,
            max_new_tokens: 32,
            priority: 0,
        }
    }

    #[test]
    fn round_robin_cycles() {
        let mut rp = RoutingPolicy::new(3, PolicyMode::RoundRobin, loads(3, 10), 4).unwrap();
        let r0 = rp.route(&req(1, vec![1, 2])).unwrap();
        let r1 = rp.route(&req(2, vec![1, 2])).unwrap();
        let r2 = rp.route(&req(3, vec![1, 2])).unwrap();
        assert_eq!(r0.rank, 0);
        assert_eq!(r1.rank, 1);
        assert_eq!(r2.rank, 2);
        assert_eq!(rp.metrics().round_robin_count, 3);
    }

    #[test]
    fn least_loaded_picks_most_free() {
        let ls = vec![
            RankLoad {
                free_blocks: 5,
                total_blocks: 10,
                in_flight: 0,
            },
            RankLoad {
                free_blocks: 9,
                total_blocks: 10,
                in_flight: 0,
            },
            RankLoad {
                free_blocks: 2,
                total_blocks: 10,
                in_flight: 0,
            },
        ];
        let mut rp = RoutingPolicy::new(3, PolicyMode::LeastLoaded, ls, 4).unwrap();
        let dec = rp.route(&req(1, vec![1])).unwrap();
        assert_eq!(dec.rank, 1, "rank 1 has most free blocks");
    }

    #[test]
    fn prefix_affinity_hits_registered_rank() {
        let mut rp = RoutingPolicy::new(4, PolicyMode::PrefixAffinity, loads(4, 10), 3).unwrap();
        let r = req(1, vec![10, 20, 30, 40]);
        let hash = r.prefix_hash(3);
        rp.register_prefix(hash, 2).unwrap();
        let dec = rp.route(&r).unwrap();
        assert_eq!(dec.rank, 2);
        assert!(dec.prefix_hit);
        assert_eq!(dec.policy_used, DispatchPolicy::PrefixAffinity);
        assert_eq!(rp.metrics().prefix_hits, 1);
    }

    #[test]
    fn prefix_affinity_falls_back_on_miss() {
        let ls = vec![
            RankLoad {
                free_blocks: 3,
                total_blocks: 10,
                in_flight: 0,
            },
            RankLoad {
                free_blocks: 8,
                total_blocks: 10,
                in_flight: 0,
            },
        ];
        let mut rp = RoutingPolicy::new(2, PolicyMode::PrefixAffinity, ls, 3).unwrap();
        let dec = rp.route(&req(1, vec![99, 98, 97])).unwrap();
        // No prefix map entry → falls back to least-loaded → rank 1
        assert_eq!(dec.rank, 1);
        assert!(!dec.prefix_hit);
        assert_eq!(dec.policy_used, DispatchPolicy::PrefixAffinity);
    }

    #[test]
    fn all_at_capacity_errors() {
        let ls = vec![
            RankLoad {
                free_blocks: 0,
                total_blocks: 5,
                in_flight: 0,
            },
            RankLoad {
                free_blocks: 0,
                total_blocks: 5,
                in_flight: 0,
            },
        ];
        let mut rp = RoutingPolicy::new(2, PolicyMode::LeastLoaded, ls, 4).unwrap();
        let err = rp.route(&req(1, vec![1])).unwrap_err();
        assert!(matches!(
            err,
            DistInferError::AllRanksAtCapacity { n_ranks: 2 }
        ));
    }

    #[test]
    fn empty_token_sequence_errors() {
        let mut rp = RoutingPolicy::new(2, PolicyMode::RoundRobin, loads(2, 10), 4).unwrap();
        let err = rp.route(&req(1, vec![])).unwrap_err();
        assert!(matches!(err, DistInferError::EmptyTokenSequence));
    }

    #[test]
    fn update_load_changes_routing() {
        let ls = loads(2, 5);
        let mut rp = RoutingPolicy::new(2, PolicyMode::LeastLoaded, ls, 4).unwrap();
        // Update rank 0 to have 10 free blocks
        rp.update_load(
            0,
            RankLoad {
                free_blocks: 10,
                total_blocks: 10,
                in_flight: 0,
            },
        )
        .unwrap();
        let dec = rp.route(&req(1, vec![1])).unwrap();
        assert_eq!(dec.rank, 0);
    }

    #[test]
    fn metrics_accumulate() {
        let mut rp = RoutingPolicy::new(2, PolicyMode::RoundRobin, loads(2, 10), 4).unwrap();
        for i in 0..6 {
            rp.route(&req(i, vec![i as u32 + 1])).unwrap();
        }
        assert_eq!(rp.metrics().total_routed, 6);
        assert_eq!(rp.metrics().round_robin_count, 6);
    }
}
