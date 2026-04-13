//! Cross-rank KV block migration.
//!
//! In production, migrating a KV block from rank A to rank B requires:
//! 1. An NVLink / PCIe P2P GPU copy (or staging through host memory).
//! 2. Updating the block table in rank B's `CacheManager`.
//!
//! Here we simulate the migration with host-side buffer operations so the
//! entire crate remains runnable without a CUDA driver.

use crate::error::{DistInferError, DistInferResult};

// ─── BlockData ───────────────────────────────────────────────────────────────

/// Serialized content of a single KV block ready for transmission.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockData {
    /// The block identifier on the source rank.
    pub block_id: u32,
    /// Layer count (the block stores K and V for each layer).
    pub n_layers: usize,
    /// Tokens stored per block.
    pub block_size: usize,
    /// Key/value dimension per layer per token.
    pub kv_dim: usize,
    /// Flat data buffer: `[n_layers, 2, block_size, kv_dim]` in row-major order.
    /// The `2` axis is [K, V].
    pub data: Vec<f32>,
}

impl BlockData {
    /// Create a zeroed block.
    pub fn zeros(block_id: u32, n_layers: usize, block_size: usize, kv_dim: usize) -> Self {
        let n = n_layers * 2 * block_size * kv_dim;
        Self {
            block_id,
            n_layers,
            block_size,
            kv_dim,
            data: vec![0.0; n],
        }
    }

    /// Expected data length.
    pub fn expected_len(n_layers: usize, block_size: usize, kv_dim: usize) -> usize {
        n_layers * 2 * block_size * kv_dim
    }

    /// Validate that `data.len()` matches the declared dimensions.
    pub fn validate(&self) -> DistInferResult<()> {
        let expected = Self::expected_len(self.n_layers, self.block_size, self.kv_dim);
        if self.data.len() != expected {
            return Err(DistInferError::DimensionMismatch {
                expected,
                got: self.data.len(),
            });
        }
        Ok(())
    }

    /// Byte size of the block data (4 bytes per f32 element).
    pub fn byte_size(&self) -> usize {
        self.data.len() * 4
    }

    /// Access the K data for a given layer.
    pub fn key_slice(&self, layer: usize) -> &[f32] {
        let stride = self.block_size * self.kv_dim;
        let start = (layer * 2) * stride;
        &self.data[start..start + stride]
    }

    /// Access the V data for a given layer.
    pub fn value_slice(&self, layer: usize) -> &[f32] {
        let stride = self.block_size * self.kv_dim;
        let start = (layer * 2 + 1) * stride;
        &self.data[start..start + stride]
    }
}

// ─── MigrationRequest ────────────────────────────────────────────────────────

/// Describes a migration of one block from one rank to another.
#[derive(Debug, Clone)]
pub struct MigrationRequest {
    /// Sequence identifier whose block is being migrated.
    pub seq_id: u64,
    /// Block identifier on the source rank.
    pub src_block_id: u32,
    /// Rank currently holding the block.
    pub src_rank: usize,
    /// Destination rank.
    pub dst_rank: usize,
}

// ─── MigrationStats ──────────────────────────────────────────────────────────

/// Running statistics for a `BlockMigrator`.
#[derive(Debug, Clone, Default)]
pub struct MigrationStats {
    /// Total blocks migrated since creation.
    pub blocks_migrated: usize,
    /// Total bytes transferred (simulated).
    pub bytes_transferred: usize,
    /// Migrations that failed (destination pool full, etc.).
    pub failures: usize,
}

// ─── BlockMigrator ───────────────────────────────────────────────────────────

/// Simulates cross-rank KV block migration.
///
/// Maintains a local "staging buffer" representing this rank's share of the
/// migration pipe.  When `migrate` is called, block data is serialised,
/// "transferred" (copied in host memory), and the source is marked free.
#[derive(Debug)]
pub struct BlockMigrator {
    /// This rank's ordinal.
    rank: usize,
    /// World size.
    world_size: usize,
    /// Simulated incoming staging buffer: `dst_block_id → BlockData`.
    incoming: std::collections::HashMap<u32, BlockData>,
    /// Accumulated statistics.
    stats: MigrationStats,
    /// Next block id to issue for newly received blocks.
    next_block_id: u32,
}

impl BlockMigrator {
    /// Create a migrator for the given rank.
    pub fn new(rank: usize, world_size: usize) -> DistInferResult<Self> {
        if rank >= world_size {
            return Err(DistInferError::RankOutOfRange { rank, world_size });
        }
        Ok(Self {
            rank,
            world_size,
            incoming: std::collections::HashMap::new(),
            stats: MigrationStats::default(),
            next_block_id: 0,
        })
    }

    /// Simulate sending `block_data` from `src_rank` to this rank.
    ///
    /// In production this would be a P2P copy or NVLink transfer.
    /// Returns the local block id assigned to the received data.
    pub fn receive_block(
        &mut self,
        req: &MigrationRequest,
        block_data: BlockData,
    ) -> DistInferResult<u32> {
        if req.dst_rank != self.rank {
            return Err(DistInferError::MigrationTargetInvalid {
                target: req.dst_rank,
                world_size: self.world_size,
            });
        }
        block_data.validate()?;
        let new_id = self.next_block_id;
        self.next_block_id += 1;
        self.stats.bytes_transferred += block_data.byte_size();
        self.stats.blocks_migrated += 1;
        self.incoming.insert(new_id, block_data);
        Ok(new_id)
    }

    /// Take ownership of a received block, removing it from the staging buffer.
    pub fn take_block(&mut self, local_block_id: u32) -> Option<BlockData> {
        self.incoming.remove(&local_block_id)
    }

    /// Number of blocks currently in the incoming staging buffer.
    pub fn pending_count(&self) -> usize {
        self.incoming.len()
    }

    /// Validate that `dst_rank` is a valid migration target.
    pub fn validate_target(&self, dst_rank: usize) -> DistInferResult<()> {
        if dst_rank >= self.world_size {
            return Err(DistInferError::MigrationTargetInvalid {
                target: dst_rank,
                world_size: self.world_size,
            });
        }
        if dst_rank == self.rank {
            return Err(DistInferError::Internal(
                "cannot migrate block to the same rank",
            ));
        }
        Ok(())
    }

    /// Retrieve statistics.
    pub fn stats(&self) -> &MigrationStats {
        &self.stats
    }

    /// This rank's ordinal.
    pub fn rank(&self) -> usize {
        self.rank
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn req(seq_id: u64, src_block_id: u32, src: usize, dst: usize) -> MigrationRequest {
        MigrationRequest {
            seq_id,
            src_block_id,
            src_rank: src,
            dst_rank: dst,
        }
    }

    #[test]
    fn block_data_validate_correct() {
        let b = BlockData::zeros(0, 2, 4, 8);
        assert!(b.validate().is_ok());
    }

    #[test]
    fn block_data_key_value_slices() {
        let n_layers = 2;
        let bs = 2;
        let kv_dim = 3;
        let mut b = BlockData::zeros(0, n_layers, bs, kv_dim);
        // Fill layer 0 K with 1.0, layer 0 V with 2.0
        let stride = bs * kv_dim;
        b.data[0..stride].iter_mut().for_each(|x| *x = 1.0);
        b.data[stride..2 * stride].iter_mut().for_each(|x| *x = 2.0);
        assert!(b.key_slice(0).iter().all(|&v| v == 1.0));
        assert!(b.value_slice(0).iter().all(|&v| v == 2.0));
    }

    #[test]
    fn receive_block_assigns_local_id() {
        let mut m = BlockMigrator::new(1, 4).unwrap();
        let block = BlockData::zeros(5, 2, 4, 8);
        let local_id = m.receive_block(&req(42, 5, 0, 1), block).unwrap();
        assert_eq!(local_id, 0);
        assert_eq!(m.pending_count(), 1);
    }

    #[test]
    fn take_block_removes_from_buffer() {
        let mut m = BlockMigrator::new(2, 4).unwrap();
        let block = BlockData::zeros(3, 1, 2, 4);
        let lid = m.receive_block(&req(10, 3, 0, 2), block.clone()).unwrap();
        let taken = m.take_block(lid).unwrap();
        assert_eq!(taken, block);
        assert_eq!(m.pending_count(), 0);
    }

    #[test]
    fn wrong_destination_rank_errors() {
        let mut m = BlockMigrator::new(2, 4).unwrap();
        let block = BlockData::zeros(0, 1, 2, 4);
        // Request says dst=3, but migrator is rank 2
        let err = m.receive_block(&req(1, 0, 0, 3), block).unwrap_err();
        assert!(matches!(
            err,
            DistInferError::MigrationTargetInvalid {
                target: 3,
                world_size: 4
            }
        ));
    }

    #[test]
    fn stats_updated_after_receive() {
        let mut m = BlockMigrator::new(1, 2).unwrap();
        let block = BlockData::zeros(0, 2, 4, 8); // 2*2*4*8=128 f32s = 512 bytes
        m.receive_block(&req(1, 0, 0, 1), block.clone()).unwrap();
        m.receive_block(&req(2, 1, 0, 1), block).unwrap();
        assert_eq!(m.stats().blocks_migrated, 2);
        assert_eq!(m.stats().bytes_transferred, 1024); // 2 × 128 × 4
    }

    #[test]
    fn validate_target_self_errors() {
        let m = BlockMigrator::new(1, 4).unwrap();
        assert!(m.validate_target(1).is_err()); // same rank
    }

    #[test]
    fn rank_out_of_range_errors() {
        assert!(BlockMigrator::new(5, 4).is_err());
    }
}
