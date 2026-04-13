//! Sequence ownership partition across ranks.
//!
//! `CachePartition` assigns each new sequence to the rank with the most
//! available KV-cache blocks (least-loaded).  It tracks per-rank sequence
//! count and provides utilities for rebalancing.

use std::collections::HashMap;

use crate::error::{DistInferError, DistInferResult};
use crate::handle::DistInferHandle;

// ─── SeqOwnership ────────────────────────────────────────────────────────────

/// Records which rank owns a sequence and how many KV blocks it has used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqOwnership {
    /// Sequence identifier.
    pub seq_id: u64,
    /// Rank that owns (manages KV blocks for) this sequence.
    pub owner_rank: usize,
    /// Number of KV blocks currently allocated for this sequence.
    pub n_blocks: usize,
}

// ─── RankCacheStats ──────────────────────────────────────────────────────────

/// Per-rank cache utilization snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankCacheStats {
    /// Total KV-block capacity of this rank's pool.
    pub total_blocks: usize,
    /// Number of free blocks on this rank.
    pub free_blocks: usize,
    /// Number of sequences currently owned by this rank.
    pub n_seqs: usize,
}

impl RankCacheStats {
    /// Blocks in use.
    pub fn used_blocks(&self) -> usize {
        self.total_blocks - self.free_blocks
    }

    /// Utilization in [0.0, 1.0].
    pub fn utilization(&self) -> f32 {
        if self.total_blocks == 0 {
            return 1.0;
        }
        self.used_blocks() as f32 / self.total_blocks as f32
    }
}

// ─── CachePartition ──────────────────────────────────────────────────────────

/// Manages sequence-to-rank assignment for a distributed KV cache.
///
/// Assignment policy: **least-loaded rank** (fewest allocated blocks).
/// Migration policy: evict a sequence from the most loaded rank to the
/// least loaded rank when utilization difference exceeds a threshold.
#[derive(Debug)]
pub struct CachePartition {
    handle: DistInferHandle,
    /// Per-rank stats, indexed by rank.
    rank_stats: Vec<RankCacheStats>,
    /// Map from seq_id to its ownership record.
    ownership: HashMap<u64, SeqOwnership>,
    /// Threshold in utilization difference [0, 1] above which migration is suggested.
    rebalance_threshold: f32,
}

impl CachePartition {
    /// Construct a partition manager.
    ///
    /// `blocks_per_rank` must have `world_size` entries.
    pub fn new(
        handle: DistInferHandle,
        blocks_per_rank: &[usize],
        rebalance_threshold: f32,
    ) -> DistInferResult<Self> {
        let ws = handle.world_size();
        if blocks_per_rank.len() != ws {
            return Err(DistInferError::DimensionMismatch {
                expected: ws,
                got: blocks_per_rank.len(),
            });
        }
        let rank_stats = blocks_per_rank
            .iter()
            .map(|&cap| RankCacheStats {
                total_blocks: cap,
                free_blocks: cap,
                n_seqs: 0,
            })
            .collect();
        Ok(Self {
            handle,
            rank_stats,
            ownership: HashMap::new(),
            rebalance_threshold,
        })
    }

    /// Assign a new sequence to the least-loaded rank.
    ///
    /// Returns the rank that will own the sequence.
    pub fn assign(&mut self, seq_id: u64, initial_blocks: usize) -> DistInferResult<usize> {
        if self.ownership.contains_key(&seq_id) {
            // Already assigned — return existing owner
            return Ok(self.ownership[&seq_id].owner_rank);
        }
        // Find least-loaded rank (by free blocks)
        let (best_rank, stats) = self
            .rank_stats
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.free_blocks)
            .ok_or(DistInferError::AllRanksAtCapacity { n_ranks: 0 })?;

        if stats.free_blocks < initial_blocks {
            return Err(DistInferError::BlockPoolExhausted { rank: best_rank });
        }

        self.rank_stats[best_rank].free_blocks -= initial_blocks;
        self.rank_stats[best_rank].n_seqs += 1;

        let ownership = SeqOwnership {
            seq_id,
            owner_rank: best_rank,
            n_blocks: initial_blocks,
        };
        self.ownership.insert(seq_id, ownership);
        Ok(best_rank)
    }

    /// Allocate additional blocks for a sequence already assigned.
    pub fn grow(&mut self, seq_id: u64, additional_blocks: usize) -> DistInferResult<()> {
        let own = self
            .ownership
            .get_mut(&seq_id)
            .ok_or(DistInferError::SequenceNotOwned {
                seq_id,
                rank: self.handle.global_rank(),
            })?;
        let rank = own.owner_rank;
        if self.rank_stats[rank].free_blocks < additional_blocks {
            return Err(DistInferError::BlockPoolExhausted { rank });
        }
        self.rank_stats[rank].free_blocks -= additional_blocks;
        own.n_blocks += additional_blocks;
        Ok(())
    }

    /// Free all blocks for a completed sequence.
    pub fn release(&mut self, seq_id: u64) -> DistInferResult<()> {
        let own = self
            .ownership
            .remove(&seq_id)
            .ok_or(DistInferError::SequenceNotOwned {
                seq_id,
                rank: self.handle.global_rank(),
            })?;
        self.rank_stats[own.owner_rank].free_blocks += own.n_blocks;
        self.rank_stats[own.owner_rank].n_seqs =
            self.rank_stats[own.owner_rank].n_seqs.saturating_sub(1);
        Ok(())
    }

    /// Look up ownership for `seq_id`.
    pub fn ownership(&self, seq_id: u64) -> Option<SeqOwnership> {
        self.ownership.get(&seq_id).copied()
    }

    /// Return per-rank statistics snapshot.
    pub fn stats(&self) -> &[RankCacheStats] {
        &self.rank_stats
    }

    /// Suggest sequences to migrate for rebalancing.
    ///
    /// Returns `(seq_id, from_rank, to_rank)` tuples for sequences whose
    /// migration would reduce utilization imbalance below the threshold.
    pub fn rebalance_suggestions(&self) -> Vec<(u64, usize, usize)> {
        let utils: Vec<f32> = self.rank_stats.iter().map(|s| s.utilization()).collect();
        let max_u = utils.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_u = utils.iter().copied().fold(f32::INFINITY, f32::min);
        if max_u - min_u < self.rebalance_threshold {
            return vec![];
        }

        let from_rank = utils
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(r, _)| r)
            .unwrap_or(0);
        let to_rank = utils
            .iter()
            .copied()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(r, _)| r)
            .unwrap_or(0);

        if from_rank == to_rank {
            return vec![];
        }

        // Suggest migrating the smallest sequence from the overloaded rank
        let candidate = self
            .ownership
            .values()
            .filter(|o| o.owner_rank == from_rank)
            .min_by_key(|o| o.n_blocks);

        candidate
            .map(|o| vec![(o.seq_id, from_rank, to_rank)])
            .unwrap_or_default()
    }

    /// Apply a migration: transfer ownership of `seq_id` from `from` to `to`.
    pub fn apply_migration(
        &mut self,
        seq_id: u64,
        from_rank: usize,
        to_rank: usize,
    ) -> DistInferResult<()> {
        let ws = self.handle.world_size();
        if to_rank >= ws {
            return Err(DistInferError::MigrationTargetInvalid {
                target: to_rank,
                world_size: ws,
            });
        }
        let own = self
            .ownership
            .get_mut(&seq_id)
            .ok_or(DistInferError::SequenceNotOwned {
                seq_id,
                rank: from_rank,
            })?;
        if own.owner_rank != from_rank {
            return Err(DistInferError::SequenceNotOwned {
                seq_id,
                rank: from_rank,
            });
        }
        let blocks = own.n_blocks;
        if self.rank_stats[to_rank].free_blocks < blocks {
            return Err(DistInferError::BlockPoolExhausted { rank: to_rank });
        }
        // Update stats
        self.rank_stats[from_rank].free_blocks += blocks;
        self.rank_stats[from_rank].n_seqs = self.rank_stats[from_rank].n_seqs.saturating_sub(1);
        self.rank_stats[to_rank].free_blocks -= blocks;
        self.rank_stats[to_rank].n_seqs += 1;
        own.owner_rank = to_rank;
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::{DistInferHandle, ParallelismConfig, SmVersion};

    fn handle_world(ws: usize) -> DistInferHandle {
        DistInferHandle::new(
            0,
            SmVersion(80),
            0,
            ParallelismConfig {
                tp: ws,
                sp: 1,
                ep: 1,
            },
        )
        .unwrap()
    }

    #[test]
    fn assign_to_least_loaded() {
        let h = handle_world(4);
        let mut part = CachePartition::new(h, &[10, 20, 5, 15], 0.2).unwrap();
        // Least loaded is rank 1 (most free blocks)
        let r = part.assign(42, 3).unwrap();
        assert_eq!(r, 1);
        assert_eq!(part.stats()[1].free_blocks, 17);
    }

    #[test]
    fn release_returns_blocks() {
        let h = handle_world(2);
        let mut part = CachePartition::new(h, &[10, 10], 0.2).unwrap();
        part.assign(1, 4).unwrap();
        part.release(1).unwrap();
        assert_eq!(part.stats()[0].free_blocks, 10);
    }

    #[test]
    fn grow_allocates_extra_blocks() {
        let h = handle_world(2);
        let mut part = CachePartition::new(h, &[20, 20], 0.2).unwrap();
        part.assign(99, 5).unwrap();
        part.grow(99, 3).unwrap();
        let own = part.ownership(99).unwrap();
        assert_eq!(own.n_blocks, 8);
    }

    #[test]
    fn pool_exhausted_error() {
        let h = handle_world(2);
        let mut part = CachePartition::new(h, &[3, 3], 0.2).unwrap();
        let err = part.assign(1, 5).unwrap_err();
        assert!(matches!(
            err,
            DistInferError::BlockPoolExhausted { rank: _ }
        ));
    }

    #[test]
    fn rebalance_suggests_migration() {
        let h = handle_world(2);
        // Rank 0: 10 total, 1 free (90% utilization); rank 1: 10 total, 9 free
        let mut part = CachePartition::new(h, &[10, 10], 0.1).unwrap();
        part.assign(1, 9).unwrap(); // → rank 1 (most free)
        // Now manually skew: assign small seq to same rank with low threshold
        // The suggestion should migrate seq 1 from rank 1 to rank 0... wait,
        // rank 1 has 1 free, rank 0 has 10 free → from=rank1 to=rank0
        let suggestions = part.rebalance_suggestions();
        assert!(!suggestions.is_empty(), "should suggest migration");
        let (sid, from, _to) = suggestions[0];
        assert_eq!(sid, 1);
        assert_eq!(from, 1, "overloaded rank should be 1");
    }

    #[test]
    fn apply_migration_updates_ownership() {
        let h = handle_world(2);
        let mut part = CachePartition::new(h, &[10, 10], 0.1).unwrap();
        part.assign(77, 4).unwrap(); // → rank 1 (or 0, both equal, rank 1 is first max)
        // Determine where it went
        let owner = part.ownership(77).unwrap().owner_rank;
        let other = 1 - owner;
        part.apply_migration(77, owner, other).unwrap();
        assert_eq!(part.ownership(77).unwrap().owner_rank, other);
    }
}
