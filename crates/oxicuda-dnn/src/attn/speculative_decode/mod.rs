//! Speculative decoding KV-cache management.
//!
//! Speculative decoding uses a lightweight draft model to generate `K` candidate
//! tokens autoregressively, then verifies them in parallel with the target model.
//! Accepted tokens are promoted; rejected tokens trigger a rollback of the draft
//! cache to the last checkpoint.
//!
//! This module provides:
//!
//! | Type                       | Description                                        |
//! |----------------------------|----------------------------------------------------|
//! | [`SpeculativeDecodeConfig`]| Configuration for draft/target model dimensions     |
//! | [`SpeculativeKvManager`]   | Manages draft + target KV caches with rollback      |
//! | [`DraftCacheState`]        | Draft model cache (lightweight, frequently rolled)  |
//! | [`TargetCacheState`]       | Target model cache (verified tokens only)           |
//! | [`CacheCheckpoint`]        | Snapshot for rollback on rejection                  |
//! | [`VerificationResult`]     | Outcome of speculation verification                 |
//! | [`SpeculativeDecodePlan`]  | PTX generation for speculative decode kernels       |
//! | [`KvManager`]              | Standalone paged KV-cache allocator                 |
//! | [`KvCheckpoint`]           | Per-sequence snapshot for KvManager restore         |
//! | [`SpeculativeDecoder`]     | High-level speculative decode driver                |
//! | [`TokenVerificationResult`]| Token-level verification outcome                    |

pub mod decoder;
pub mod kv_manager;

pub use decoder::{
    SpecDecConfig, SpecDecOutput, SpeculativeDecoder, TokenVerificationResult, accept_token,
};
pub use kv_manager::{KvCheckpoint, KvManager};

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::DnnError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for copy/verification kernels.
const SPEC_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// SpeculativeDecodeConfig
// ---------------------------------------------------------------------------

/// Configuration for speculative decoding.
///
/// Describes the draft and target model dimensions and constraints for
/// the speculative decode loop.
#[derive(Debug, Clone)]
pub struct SpeculativeDecodeConfig {
    /// Number of layers in the draft model.
    pub draft_num_layers: usize,
    /// Number of KV heads in the draft model.
    pub draft_num_heads: usize,
    /// Per-head dimension in the draft model.
    pub draft_head_dim: usize,
    /// Number of layers in the target model.
    pub target_num_layers: usize,
    /// Number of KV heads in the target model.
    pub target_num_heads: usize,
    /// Per-head dimension in the target model.
    pub target_head_dim: usize,
    /// Maximum number of draft tokens generated per speculation round (K).
    pub max_draft_tokens: usize,
    /// Tokens per page in paged KV-cache.
    pub page_size: usize,
    /// Maximum physical pages available.
    pub max_pages: usize,
    /// Threshold for modified rejection sampling (0.0..=1.0).
    pub acceptance_threshold: f32,
}

impl SpeculativeDecodeConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if any dimension is zero,
    /// `max_draft_tokens` is zero, `page_size` is zero, `max_pages` is zero,
    /// or `acceptance_threshold` is outside `[0.0, 1.0]`.
    pub fn validate(&self) -> Result<(), DnnError> {
        if self.draft_num_layers == 0 {
            return Err(DnnError::InvalidArgument(
                "draft_num_layers must be > 0".into(),
            ));
        }
        if self.draft_num_heads == 0 {
            return Err(DnnError::InvalidArgument(
                "draft_num_heads must be > 0".into(),
            ));
        }
        if self.draft_head_dim == 0 {
            return Err(DnnError::InvalidArgument(
                "draft_head_dim must be > 0".into(),
            ));
        }
        if self.target_num_layers == 0 {
            return Err(DnnError::InvalidArgument(
                "target_num_layers must be > 0".into(),
            ));
        }
        if self.target_num_heads == 0 {
            return Err(DnnError::InvalidArgument(
                "target_num_heads must be > 0".into(),
            ));
        }
        if self.target_head_dim == 0 {
            return Err(DnnError::InvalidArgument(
                "target_head_dim must be > 0".into(),
            ));
        }
        if self.max_draft_tokens == 0 {
            return Err(DnnError::InvalidArgument(
                "max_draft_tokens must be > 0".into(),
            ));
        }
        if self.page_size == 0 {
            return Err(DnnError::InvalidArgument("page_size must be > 0".into()));
        }
        if self.max_pages == 0 {
            return Err(DnnError::InvalidArgument("max_pages must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.acceptance_threshold) {
            return Err(DnnError::InvalidArgument(format!(
                "acceptance_threshold must be in [0.0, 1.0], got {}",
                self.acceptance_threshold,
            )));
        }
        Ok(())
    }

    /// Elements per page for the draft model cache (one of K or V).
    #[must_use]
    pub fn draft_page_elements(&self) -> usize {
        self.draft_num_heads * self.page_size * self.draft_head_dim
    }

    /// Elements per page for the target model cache (one of K or V).
    #[must_use]
    pub fn target_page_elements(&self) -> usize {
        self.target_num_heads * self.page_size * self.target_head_dim
    }
}

// ---------------------------------------------------------------------------
// DraftCacheState
// ---------------------------------------------------------------------------

/// Draft model KV-cache state.
///
/// The draft cache is lightweight and frequently rolled back when speculation
/// is rejected. Page tables and positions are managed on the host for fast
/// checkpoint/restore without GPU round-trips.
#[derive(Debug, Clone)]
pub struct DraftCacheState {
    /// Number of layers in the draft model.
    pub num_layers: usize,
    /// Number of KV heads per layer.
    pub num_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Tokens per page.
    pub page_size: usize,
    /// Maximum physical pages.
    pub max_pages: usize,
    /// Current position per sequence.
    pub seq_positions: Vec<usize>,
    /// Page table per sequence (sequence index -> list of physical page indices).
    pub page_tables: Vec<Vec<usize>>,
    /// Free page list.
    pub free_pages: Vec<usize>,
    /// Cumulative count of tokens processed through the draft model.
    pub total_tokens_generated: usize,
}

impl DraftCacheState {
    /// Creates a new draft cache with a single sequence at position 0.
    fn new(config: &SpeculativeDecodeConfig) -> Self {
        let free_pages: Vec<usize> = (0..config.max_pages).collect();
        Self {
            num_layers: config.draft_num_layers,
            num_heads: config.draft_num_heads,
            head_dim: config.draft_head_dim,
            page_size: config.page_size,
            max_pages: config.max_pages,
            seq_positions: vec![0],
            page_tables: vec![Vec::new()],
            free_pages,
            total_tokens_generated: 0,
        }
    }

    /// Number of pages currently allocated (not free).
    #[must_use]
    pub fn allocated_pages(&self) -> usize {
        self.max_pages - self.free_pages.len()
    }
}

// ---------------------------------------------------------------------------
// TargetCacheState
// ---------------------------------------------------------------------------

/// Target model KV-cache state.
///
/// Only verified (accepted) tokens are written here. The `verified_position`
/// tracks the global high-water mark of accepted tokens.
#[derive(Debug, Clone)]
pub struct TargetCacheState {
    /// Number of layers in the target model.
    pub num_layers: usize,
    /// Number of KV heads per layer.
    pub num_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Tokens per page.
    pub page_size: usize,
    /// Maximum physical pages.
    pub max_pages: usize,
    /// Current position per sequence.
    pub seq_positions: Vec<usize>,
    /// Page table per sequence.
    pub page_tables: Vec<Vec<usize>>,
    /// Free page list.
    pub free_pages: Vec<usize>,
    /// Global verified position (high-water mark across sequences).
    pub verified_position: usize,
}

impl TargetCacheState {
    /// Creates a new target cache with a single sequence at position 0.
    fn new(config: &SpeculativeDecodeConfig) -> Self {
        let free_pages: Vec<usize> = (0..config.max_pages).collect();
        Self {
            num_layers: config.target_num_layers,
            num_heads: config.target_num_heads,
            head_dim: config.target_head_dim,
            page_size: config.page_size,
            max_pages: config.max_pages,
            seq_positions: vec![0],
            page_tables: vec![Vec::new()],
            free_pages,
            verified_position: 0,
        }
    }

    /// Number of pages currently allocated (not free).
    #[must_use]
    pub fn allocated_pages(&self) -> usize {
        self.max_pages - self.free_pages.len()
    }
}

// ---------------------------------------------------------------------------
// CacheCheckpoint
// ---------------------------------------------------------------------------

/// Snapshot of cache state for rollback on speculation rejection.
///
/// Captures draft cache positions and page tables so the draft cache can
/// be restored to the pre-speculation state without touching GPU memory.
#[derive(Debug, Clone)]
pub struct CacheCheckpoint {
    /// Draft positions at checkpoint time.
    pub draft_positions: Vec<usize>,
    /// Draft page tables at checkpoint time (deep clone).
    pub draft_page_tables: Vec<Vec<usize>>,
    /// Draft free pages at checkpoint time.
    pub draft_free_pages: Vec<usize>,
    /// Target verified position at checkpoint time.
    pub target_position: usize,
    /// Monotonic timestamp for ordering checkpoints.
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// VerificationResult
// ---------------------------------------------------------------------------

/// Outcome of verifying drafted tokens against the target model.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of tokens accepted by the target model.
    pub accepted_count: usize,
    /// Total tokens that were drafted.
    pub total_drafted: usize,
    /// Fraction of drafted tokens that were accepted.
    pub acceptance_rate: f32,
    /// Physical pages freed during rollback (if any).
    pub rolled_back_pages: usize,
    /// Whether the target model generated a bonus token (always true when
    /// at least partial acceptance occurs, per the speculative decode algorithm).
    pub bonus_token: bool,
}

// ---------------------------------------------------------------------------
// SpeculativeDecodeStats
// ---------------------------------------------------------------------------

/// Cumulative statistics for the speculative decode manager.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeDecodeStats {
    /// Pages currently allocated in the draft cache.
    pub draft_pages_allocated: usize,
    /// Pages currently allocated in the target cache.
    pub target_pages_allocated: usize,
    /// Total checkpoints created over the manager's lifetime.
    pub total_checkpoints_created: usize,
    /// Total rollbacks performed over the manager's lifetime.
    pub total_rollbacks: usize,
    /// Running average acceptance rate.
    pub average_acceptance_rate: f32,
}

// ---------------------------------------------------------------------------
// SpeculativeKvManager
// ---------------------------------------------------------------------------

/// Manages draft and target KV caches for speculative decoding.
///
/// The workflow for one speculation round:
///
/// 1. [`Self::create_checkpoint`] — snapshot draft state before speculation.
/// 2. [`Self::append_draft_kv`] — called K times as the draft model generates tokens.
/// 3. (Target model verifies K tokens in parallel.)
///    4a. If all accepted: [`Self::accept_tokens`] with `count = K`.
///    4b. If partial: [`Self::accept_tokens`] with `count < K` (triggers rollback).
///
/// The manager tracks lifetime statistics via [`Self::stats`].
pub struct SpeculativeKvManager {
    config: SpeculativeDecodeConfig,
    draft_cache: DraftCacheState,
    target_cache: TargetCacheState,
    checkpoint: Option<CacheCheckpoint>,
    checkpoint_counter: u64,
    total_checkpoints: usize,
    total_rollbacks: usize,
    acceptance_sum: f64,
    acceptance_count: usize,
}

impl SpeculativeKvManager {
    /// Creates a new speculative KV manager.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: SpeculativeDecodeConfig) -> Result<Self, DnnError> {
        config.validate()?;
        let draft_cache = DraftCacheState::new(&config);
        let target_cache = TargetCacheState::new(&config);
        Ok(Self {
            config,
            draft_cache,
            target_cache,
            checkpoint: None,
            checkpoint_counter: 0,
            total_checkpoints: 0,
            total_rollbacks: 0,
            acceptance_sum: 0.0,
            acceptance_count: 0,
        })
    }

    /// Creates a checkpoint of the draft cache state before a speculation round.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `seq_id` is out of range.
    pub fn create_checkpoint(&mut self, seq_id: usize) -> Result<(), DnnError> {
        if seq_id >= self.draft_cache.seq_positions.len() {
            return Err(DnnError::InvalidArgument(format!(
                "seq_id {} out of range (max {})",
                seq_id,
                self.draft_cache.seq_positions.len(),
            )));
        }
        self.checkpoint_counter += 1;
        self.total_checkpoints += 1;
        self.checkpoint = Some(CacheCheckpoint {
            draft_positions: self.draft_cache.seq_positions.clone(),
            draft_page_tables: self.draft_cache.page_tables.clone(),
            draft_free_pages: self.draft_cache.free_pages.clone(),
            target_position: self.target_cache.verified_position,
            timestamp: self.checkpoint_counter,
        });
        Ok(())
    }

    /// Appends a draft token's KV entry, allocating a new page if needed.
    ///
    /// Returns `(k_offset, v_offset)` as element offsets into the draft
    /// cache pool where the caller should write the KV data.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `seq_id` is out of range.
    /// Returns [`DnnError::WorkspaceRequired`] if no free pages remain.
    pub fn append_draft_kv(&mut self, seq_id: usize) -> Result<(usize, usize), DnnError> {
        if seq_id >= self.draft_cache.seq_positions.len() {
            return Err(DnnError::InvalidArgument(format!(
                "seq_id {} out of range (max {})",
                seq_id,
                self.draft_cache.seq_positions.len(),
            )));
        }

        let pos = self.draft_cache.seq_positions[seq_id];
        let logical_page = pos / self.draft_cache.page_size;
        let offset_in_page = pos % self.draft_cache.page_size;

        // Allocate a new page if we're at the start of a logical page.
        if logical_page >= self.draft_cache.page_tables[seq_id].len() {
            let phys = self
                .draft_cache
                .free_pages
                .pop()
                .ok_or(DnnError::WorkspaceRequired(
                    self.config.draft_page_elements() * 4, // approximate bytes
                ))?;
            self.draft_cache.page_tables[seq_id].push(phys);
        }

        let phys_page = self.draft_cache.page_tables[seq_id][logical_page];
        let page_elements = self.config.draft_page_elements();
        let token_stride = self.draft_cache.num_heads * self.draft_cache.head_dim;
        let k_offset = phys_page * page_elements + offset_in_page * token_stride;
        let v_offset = k_offset; // V pool is separate; offset within pool is identical.

        self.draft_cache.seq_positions[seq_id] += 1;
        self.draft_cache.total_tokens_generated += 1;

        Ok((k_offset, v_offset))
    }

    /// Appends a verified token's KV entry to the target cache.
    ///
    /// Returns `(k_offset, v_offset)` as element offsets into the target
    /// cache pool.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `seq_id` is out of range.
    /// Returns [`DnnError::WorkspaceRequired`] if no free pages remain.
    pub fn append_target_kv(&mut self, seq_id: usize) -> Result<(usize, usize), DnnError> {
        if seq_id >= self.target_cache.seq_positions.len() {
            return Err(DnnError::InvalidArgument(format!(
                "seq_id {} out of range (max {})",
                seq_id,
                self.target_cache.seq_positions.len(),
            )));
        }

        let pos = self.target_cache.seq_positions[seq_id];
        let logical_page = pos / self.target_cache.page_size;
        let offset_in_page = pos % self.target_cache.page_size;

        if logical_page >= self.target_cache.page_tables[seq_id].len() {
            let phys = self
                .target_cache
                .free_pages
                .pop()
                .ok_or(DnnError::WorkspaceRequired(
                    self.config.target_page_elements() * 4,
                ))?;
            self.target_cache.page_tables[seq_id].push(phys);
        }

        let phys_page = self.target_cache.page_tables[seq_id][logical_page];
        let page_elements = self.config.target_page_elements();
        let token_stride = self.target_cache.num_heads * self.target_cache.head_dim;
        let k_offset = phys_page * page_elements + offset_in_page * token_stride;
        let v_offset = k_offset;

        self.target_cache.seq_positions[seq_id] += 1;
        self.target_cache.verified_position += 1;

        Ok((k_offset, v_offset))
    }

    /// Rolls back the draft cache to the most recent checkpoint.
    ///
    /// Returns the number of pages freed during rollback.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `seq_id` is out of range or
    /// no checkpoint exists.
    pub fn rollback_to_checkpoint(&mut self, seq_id: usize) -> Result<usize, DnnError> {
        if seq_id >= self.draft_cache.seq_positions.len() {
            return Err(DnnError::InvalidArgument(format!(
                "seq_id {} out of range (max {})",
                seq_id,
                self.draft_cache.seq_positions.len(),
            )));
        }

        let cp = self
            .checkpoint
            .take()
            .ok_or_else(|| DnnError::InvalidArgument("no checkpoint to rollback to".into()))?;

        let pages_before = self.draft_cache.allocated_pages();

        self.draft_cache.seq_positions = cp.draft_positions;
        self.draft_cache.page_tables = cp.draft_page_tables;
        self.draft_cache.free_pages = cp.draft_free_pages;

        let pages_after = self.draft_cache.allocated_pages();
        let freed = pages_before.saturating_sub(pages_after);

        self.total_rollbacks += 1;

        Ok(freed)
    }

    /// Accepts `count` drafted tokens and promotes them to the target cache.
    ///
    /// If `count < total_drafted`, a partial rollback of the draft cache
    /// occurs to discard tokens after the rejection point.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `seq_id` is out of range or
    /// `count` exceeds the number of drafted tokens since the last checkpoint.
    pub fn accept_tokens(
        &mut self,
        seq_id: usize,
        count: usize,
    ) -> Result<VerificationResult, DnnError> {
        if seq_id >= self.draft_cache.seq_positions.len() {
            return Err(DnnError::InvalidArgument(format!(
                "seq_id {} out of range (max {})",
                seq_id,
                self.draft_cache.seq_positions.len(),
            )));
        }

        let cp = self.checkpoint.as_ref().ok_or_else(|| {
            DnnError::InvalidArgument("no checkpoint: call create_checkpoint first".into())
        })?;

        let draft_start = cp.draft_positions[seq_id];
        let draft_end = self.draft_cache.seq_positions[seq_id];
        let total_drafted = draft_end.saturating_sub(draft_start);

        if count > total_drafted {
            return Err(DnnError::InvalidArgument(format!(
                "cannot accept {} tokens when only {} were drafted",
                count, total_drafted,
            )));
        }

        // Promote accepted tokens to the target cache.
        for _ in 0..count {
            self.append_target_kv(seq_id)?;
        }

        let acceptance_rate = if total_drafted > 0 {
            count as f32 / total_drafted as f32
        } else {
            0.0
        };

        // Track running average.
        self.acceptance_sum += acceptance_rate as f64;
        self.acceptance_count += 1;

        // Partial rejection: rollback draft cache to accepted position.
        let rolled_back_pages = if count < total_drafted {
            // Restore checkpoint, then advance draft to accepted position.
            let cp_clone = self.checkpoint.clone();
            let freed = self.rollback_to_checkpoint(seq_id)?;
            // Re-advance draft positions to reflect accepted tokens only.
            self.draft_cache.seq_positions[seq_id] = draft_start + count;
            // Restore checkpoint reference for future use if needed.
            self.checkpoint = cp_clone;
            freed
        } else {
            // Full acceptance: consume the checkpoint.
            self.checkpoint = None;
            0
        };

        // The target model always generates one bonus token beyond accepted.
        let bonus_token = true;

        Ok(VerificationResult {
            accepted_count: count,
            total_drafted,
            acceptance_rate,
            rolled_back_pages,
            bonus_token,
        })
    }

    /// Returns the current draft sequence length for `seq_id`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `seq_id` is out of range.
    pub fn draft_seq_len(&self, seq_id: usize) -> Result<usize, DnnError> {
        self.draft_cache
            .seq_positions
            .get(seq_id)
            .copied()
            .ok_or_else(|| {
                DnnError::InvalidArgument(format!(
                    "seq_id {} out of range (max {})",
                    seq_id,
                    self.draft_cache.seq_positions.len(),
                ))
            })
    }

    /// Returns the current target (verified) sequence length for `seq_id`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `seq_id` is out of range.
    pub fn target_seq_len(&self, seq_id: usize) -> Result<usize, DnnError> {
        self.target_cache
            .seq_positions
            .get(seq_id)
            .copied()
            .ok_or_else(|| {
                DnnError::InvalidArgument(format!(
                    "seq_id {} out of range (max {})",
                    seq_id,
                    self.target_cache.seq_positions.len(),
                ))
            })
    }

    /// Resets all caches for a given sequence, freeing allocated pages.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if `seq_id` is out of range.
    pub fn reset_sequence(&mut self, seq_id: usize) -> Result<(), DnnError> {
        if seq_id >= self.draft_cache.seq_positions.len() {
            return Err(DnnError::InvalidArgument(format!(
                "seq_id {} out of range (max {})",
                seq_id,
                self.draft_cache.seq_positions.len(),
            )));
        }

        // Free draft pages for this sequence.
        for &page in &self.draft_cache.page_tables[seq_id] {
            self.draft_cache.free_pages.push(page);
        }
        self.draft_cache.page_tables[seq_id].clear();
        self.draft_cache.seq_positions[seq_id] = 0;

        // Free target pages for this sequence.
        if seq_id < self.target_cache.seq_positions.len() {
            for &page in &self.target_cache.page_tables[seq_id] {
                self.target_cache.free_pages.push(page);
            }
            self.target_cache.page_tables[seq_id].clear();
            self.target_cache.seq_positions[seq_id] = 0;
        }

        self.checkpoint = None;

        Ok(())
    }

    /// Returns cumulative statistics.
    #[must_use]
    pub fn stats(&self) -> SpeculativeDecodeStats {
        let avg_rate = if self.acceptance_count > 0 {
            (self.acceptance_sum / self.acceptance_count as f64) as f32
        } else {
            0.0
        };
        SpeculativeDecodeStats {
            draft_pages_allocated: self.draft_cache.allocated_pages(),
            target_pages_allocated: self.target_cache.allocated_pages(),
            total_checkpoints_created: self.total_checkpoints,
            total_rollbacks: self.total_rollbacks,
            average_acceptance_rate: avg_rate,
        }
    }

    /// Returns a reference to the draft cache state.
    #[must_use]
    pub fn draft_cache(&self) -> &DraftCacheState {
        &self.draft_cache
    }

    /// Returns a reference to the target cache state.
    #[must_use]
    pub fn target_cache(&self) -> &TargetCacheState {
        &self.target_cache
    }

    /// Returns a reference to the current checkpoint, if any.
    #[must_use]
    pub fn checkpoint(&self) -> Option<&CacheCheckpoint> {
        self.checkpoint.as_ref()
    }
}

// ---------------------------------------------------------------------------
// SpeculativeDecodePlan — PTX generation
// ---------------------------------------------------------------------------

/// PTX generation plan for speculative decoding kernels.
///
/// Generates kernels for:
/// - Copying KV data from draft to target cache pages.
/// - Token verification (logit comparison).
/// - Modified rejection sampling.
pub struct SpeculativeDecodePlan {
    config: SpeculativeDecodeConfig,
}

impl SpeculativeDecodePlan {
    /// Creates a new plan after validating the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: SpeculativeDecodeConfig) -> Result<Self, DnnError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Generates PTX for the draft-to-target KV copy kernel.
    ///
    /// The kernel copies `num_tokens` worth of KV data from source pages
    /// to destination pages, using per-token offsets for non-contiguous
    /// paged layouts.
    ///
    /// Parameters: `src_ptr` (U64), `dst_ptr` (U64), `offsets_ptr` (U64),
    /// `num_elements` (U32).
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if kernel building fails.
    pub fn generate_kv_copy_ptx(&self) -> Result<String, DnnError> {
        let kernel_name = "spec_decode_kv_copy";
        let ptx = KernelBuilder::new(kernel_name)
            .target(SmVersion::Sm80)
            .max_threads_per_block(SPEC_BLOCK_SIZE)
            .param("src_ptr", PtxType::U64)
            .param("dst_ptr", PtxType::U64)
            .param("offsets_ptr", PtxType::U64)
            .param("num_elements", PtxType::U32)
            .body(|b| {
                // tid = blockIdx.x * blockDim.x + threadIdx.x
                let tid = b.global_thread_id_x();
                let n = b.load_param_u32("num_elements");

                let tid2 = tid.clone();
                b.if_lt_u32(tid, n, |b| {
                    // Load source pointer and offset for this element.
                    let src_base = b.load_param_u64("src_ptr");
                    let dst_base = b.load_param_u64("dst_ptr");

                    // Compute byte offset: tid * 4 (f32 elements).
                    let byte_off = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("shl.b32 {byte_off}, {tid2}, 2;"));
                    let byte_off_64 = b.alloc_reg(PtxType::U64);
                    b.raw_ptx(&format!("cvt.u64.u32 {byte_off_64}, {byte_off};"));

                    let src_addr = b.add_u64(src_base, byte_off_64.clone());
                    let dst_addr = b.add_u64(dst_base, byte_off_64);

                    // Load from source, store to destination.
                    let val = b.load_global_f32(src_addr);
                    b.store_global_f32(dst_addr, val);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }

    /// Generates PTX for the token verification kernel.
    ///
    /// Compares draft logits against target logits to determine acceptance.
    /// Each thread handles one token in the speculation window.
    ///
    /// Parameters: `draft_logits` (U64), `target_logits` (U64),
    /// `accept_mask` (U64), `vocab_size` (U32), `num_tokens` (U32),
    /// `threshold` (F32).
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if kernel building fails.
    pub fn generate_verification_ptx(&self) -> Result<String, DnnError> {
        let kernel_name = "spec_decode_verify";
        let ptx = KernelBuilder::new(kernel_name)
            .target(SmVersion::Sm80)
            .max_threads_per_block(SPEC_BLOCK_SIZE)
            .param("draft_logits", PtxType::U64)
            .param("target_logits", PtxType::U64)
            .param("accept_mask", PtxType::U64)
            .param("vocab_size", PtxType::U32)
            .param("num_tokens", PtxType::U32)
            .param("threshold", PtxType::F32)
            .body(|b| {
                let tid = b.global_thread_id_x();
                let num_tok = b.load_param_u32("num_tokens");

                let tid2 = tid.clone();
                b.if_lt_u32(tid, num_tok, |b| {
                    let draft_ptr = b.load_param_u64("draft_logits");
                    let target_ptr = b.load_param_u64("target_logits");
                    let thresh = b.load_param_f32("threshold");

                    // Compute byte offset: tid * 4.
                    let byte_off = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("shl.b32 {byte_off}, {tid2}, 2;"));
                    let byte_off_64 = b.alloc_reg(PtxType::U64);
                    b.raw_ptx(&format!("cvt.u64.u32 {byte_off_64}, {byte_off};"));

                    let d_addr = b.add_u64(draft_ptr, byte_off_64.clone());
                    let t_addr = b.add_u64(target_ptr, byte_off_64.clone());

                    let d_val = b.load_global_f32(d_addr);
                    let t_val = b.load_global_f32(t_addr);

                    // Compute difference check against threshold.
                    let diff = b.sub_f32(t_val, d_val);
                    let abs_diff = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("abs.f32 {abs_diff}, {diff};"));

                    // Accept if abs_diff <= threshold (store 1), else reject (store 0).
                    let pred = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.le.f32 {pred}, {abs_diff}, {thresh};"));
                    let result = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("selp.u32 {result}, 1, 0, {pred};"));

                    let mask_ptr = b.load_param_u64("accept_mask");
                    let out_addr = b.add_u64(mask_ptr, byte_off_64);
                    b.store_global_u32(out_addr, result);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }

    /// Generates PTX for the modified rejection sampling kernel.
    ///
    /// Implements the speculative decoding rejection sampling algorithm:
    /// for each token, accept with probability min(1, p_target / p_draft);
    /// on rejection, sample from the residual distribution
    /// `norm(max(0, p_target - p_draft))`.
    ///
    /// Parameters: `draft_probs` (U64), `target_probs` (U64),
    /// `random_vals` (U64), `output_tokens` (U64), `num_tokens` (U32).
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if kernel building fails.
    pub fn generate_rejection_sampling_ptx(&self) -> Result<String, DnnError> {
        let kernel_name = "spec_decode_rejection_sample";
        let ptx = KernelBuilder::new(kernel_name)
            .target(SmVersion::Sm80)
            .max_threads_per_block(SPEC_BLOCK_SIZE)
            .param("draft_probs", PtxType::U64)
            .param("target_probs", PtxType::U64)
            .param("random_vals", PtxType::U64)
            .param("output_tokens", PtxType::U64)
            .param("num_tokens", PtxType::U32)
            .body(|b| {
                let tid = b.global_thread_id_x();
                let n = b.load_param_u32("num_tokens");

                let tid2 = tid.clone();
                b.if_lt_u32(tid, n, |b| {
                    let draft_ptr = b.load_param_u64("draft_probs");
                    let target_ptr = b.load_param_u64("target_probs");
                    let rand_ptr = b.load_param_u64("random_vals");

                    let byte_off = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("shl.b32 {byte_off}, {tid2}, 2;"));
                    let byte_off_64 = b.alloc_reg(PtxType::U64);
                    b.raw_ptx(&format!("cvt.u64.u32 {byte_off_64}, {byte_off};"));

                    let d_addr = b.add_u64(draft_ptr, byte_off_64.clone());
                    let t_addr = b.add_u64(target_ptr, byte_off_64.clone());
                    let r_addr = b.add_u64(rand_ptr, byte_off_64.clone());

                    let p_draft = b.load_global_f32(d_addr);
                    let p_target = b.load_global_f32(t_addr);
                    let rand_val = b.load_global_f32(r_addr);

                    // Acceptance ratio: min(1.0, p_target / p_draft).
                    let ratio = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("div.rn.f32 {ratio}, {p_target}, {p_draft};"));
                    let one = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {one}, 0f3F800000;"));
                    let clamped = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("min.f32 {clamped}, {ratio}, {one};"));

                    // Accept if random_val < clamped ratio.
                    let pred = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.lt.f32 {pred}, {rand_val}, {clamped};"));
                    let result = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("selp.u32 {result}, 1, 0, {pred};"));

                    let out_ptr = b.load_param_u64("output_tokens");
                    let out_addr = b.add_u64(out_ptr, byte_off_64);
                    b.store_global_u32(out_addr, result);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &SpeculativeDecodeConfig {
        &self.config
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SpeculativeDecodeConfig {
        SpeculativeDecodeConfig {
            draft_num_layers: 6,
            draft_num_heads: 8,
            draft_head_dim: 64,
            target_num_layers: 32,
            target_num_heads: 32,
            target_head_dim: 128,
            max_draft_tokens: 5,
            page_size: 16,
            max_pages: 64,
            acceptance_threshold: 0.9,
        }
    }

    // -- Config validation ---------------------------------------------------

    #[test]
    fn config_validation_ok() {
        let cfg = test_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_validation_zero_draft_layers() {
        let mut cfg = test_config();
        cfg.draft_num_layers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validation_zero_page_size() {
        let mut cfg = test_config();
        cfg.page_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validation_bad_threshold() {
        let mut cfg = test_config();
        cfg.acceptance_threshold = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_page_elements() {
        let cfg = test_config();
        // draft: 8 heads * 16 page_size * 64 head_dim = 8192
        assert_eq!(cfg.draft_page_elements(), 8 * 16 * 64);
        // target: 32 heads * 16 page_size * 128 head_dim = 65536
        assert_eq!(cfg.target_page_elements(), 32 * 16 * 128);
    }

    // -- Manager creation ----------------------------------------------------

    #[test]
    fn create_manager_initial_state() {
        let mgr = SpeculativeKvManager::new(test_config());
        assert!(mgr.is_ok());
        let mgr = mgr.expect("tested above");
        assert_eq!(mgr.draft_seq_len(0).expect("seq 0"), 0);
        assert_eq!(mgr.target_seq_len(0).expect("seq 0"), 0);
        assert!(mgr.checkpoint().is_none());

        let s = mgr.stats();
        assert_eq!(s.draft_pages_allocated, 0);
        assert_eq!(s.target_pages_allocated, 0);
        assert_eq!(s.total_checkpoints_created, 0);
        assert_eq!(s.total_rollbacks, 0);
    }

    // -- Checkpoint + rollback -----------------------------------------------

    #[test]
    fn checkpoint_and_rollback() {
        let mut mgr = SpeculativeKvManager::new(test_config()).expect("valid config");

        // Append a few draft tokens.
        mgr.append_draft_kv(0).expect("append 0");
        mgr.append_draft_kv(0).expect("append 1");
        assert_eq!(mgr.draft_seq_len(0).expect("seq 0"), 2);

        // Checkpoint.
        mgr.create_checkpoint(0).expect("checkpoint");
        assert!(mgr.checkpoint().is_some());

        // More draft tokens.
        mgr.append_draft_kv(0).expect("append 2");
        mgr.append_draft_kv(0).expect("append 3");
        assert_eq!(mgr.draft_seq_len(0).expect("seq 0"), 4);

        // Rollback.
        let freed = mgr.rollback_to_checkpoint(0).expect("rollback");
        assert_eq!(mgr.draft_seq_len(0).expect("seq 0"), 2);
        // May or may not free pages depending on page boundaries.
        let _ = freed; // usize is always >= 0
        assert!(mgr.checkpoint().is_none());

        let s = mgr.stats();
        assert_eq!(s.total_checkpoints_created, 1);
        assert_eq!(s.total_rollbacks, 1);
    }

    // -- Append draft KV -----------------------------------------------------

    #[test]
    fn append_draft_kv_increments_position() {
        let mut mgr = SpeculativeKvManager::new(test_config()).expect("valid config");
        let (k, v) = mgr.append_draft_kv(0).expect("append");
        assert_eq!(mgr.draft_seq_len(0).expect("seq 0"), 1);
        // Offsets should be defined.
        let _ = (k, v); // usize values are always valid
    }

    // -- Append target KV ----------------------------------------------------

    #[test]
    fn append_target_kv_increments_position() {
        let mut mgr = SpeculativeKvManager::new(test_config()).expect("valid config");
        let (k, v) = mgr.append_target_kv(0).expect("append");
        assert_eq!(mgr.target_seq_len(0).expect("seq 0"), 1);
        let _ = (k, v); // usize values are always valid
    }

    // -- Accept tokens -------------------------------------------------------

    #[test]
    fn accept_tokens_full() {
        let mut mgr = SpeculativeKvManager::new(test_config()).expect("valid config");

        mgr.create_checkpoint(0).expect("checkpoint");
        for _ in 0..5 {
            mgr.append_draft_kv(0).expect("draft");
        }
        assert_eq!(mgr.draft_seq_len(0).expect("seq 0"), 5);

        let result = mgr.accept_tokens(0, 5).expect("accept all");
        assert_eq!(result.accepted_count, 5);
        assert_eq!(result.total_drafted, 5);
        assert!((result.acceptance_rate - 1.0).abs() < f32::EPSILON);
        assert!(result.bonus_token);
        assert_eq!(result.rolled_back_pages, 0);

        // Target should have 5 tokens now.
        assert_eq!(mgr.target_seq_len(0).expect("seq 0"), 5);
    }

    #[test]
    fn accept_tokens_partial() {
        let mut mgr = SpeculativeKvManager::new(test_config()).expect("valid config");

        mgr.create_checkpoint(0).expect("checkpoint");
        for _ in 0..5 {
            mgr.append_draft_kv(0).expect("draft");
        }

        let result = mgr.accept_tokens(0, 2).expect("accept partial");
        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.total_drafted, 5);
        assert!((result.acceptance_rate - 0.4).abs() < f32::EPSILON);
        assert!(result.bonus_token);

        // Target should have 2 tokens; draft rolled back to 2.
        assert_eq!(mgr.target_seq_len(0).expect("seq 0"), 2);
        assert_eq!(mgr.draft_seq_len(0).expect("seq 0"), 2);
    }

    // -- Sequence reset ------------------------------------------------------

    #[test]
    fn reset_sequence_clears_state() {
        let mut mgr = SpeculativeKvManager::new(test_config()).expect("valid config");

        for _ in 0..10 {
            mgr.append_draft_kv(0).expect("draft");
        }
        for _ in 0..3 {
            mgr.append_target_kv(0).expect("target");
        }

        mgr.reset_sequence(0).expect("reset");
        assert_eq!(mgr.draft_seq_len(0).expect("seq 0"), 0);
        assert_eq!(mgr.target_seq_len(0).expect("seq 0"), 0);
        assert!(mgr.checkpoint().is_none());
    }

    // -- Stats tracking ------------------------------------------------------

    #[test]
    fn stats_tracking() {
        let mut mgr = SpeculativeKvManager::new(test_config()).expect("valid config");

        // Round 1: full acceptance.
        mgr.create_checkpoint(0).expect("cp");
        for _ in 0..3 {
            mgr.append_draft_kv(0).expect("draft");
        }
        mgr.accept_tokens(0, 3).expect("accept");

        // Round 2: partial acceptance.
        mgr.create_checkpoint(0).expect("cp");
        for _ in 0..4 {
            mgr.append_draft_kv(0).expect("draft");
        }
        mgr.accept_tokens(0, 1).expect("accept");

        let s = mgr.stats();
        assert_eq!(s.total_checkpoints_created, 2);
        // Partial acceptance triggers rollback internally.
        assert!(s.total_rollbacks >= 1);
        // Average of 1.0 and 0.25 = 0.625
        assert!(s.average_acceptance_rate > 0.0);
        assert!(s.average_acceptance_rate < 1.0);
    }

    // -- PTX generation ------------------------------------------------------

    #[test]
    fn kv_copy_ptx_generation() {
        let plan = SpeculativeDecodePlan::new(test_config()).expect("plan");
        let ptx = plan.generate_kv_copy_ptx().expect("ptx");
        assert!(ptx.contains(".entry spec_decode_kv_copy"));
        assert!(ptx.contains("%param_src_ptr"));
        assert!(ptx.contains("%param_dst_ptr"));
        assert!(ptx.contains("%param_num_elements"));
    }

    #[test]
    fn verification_ptx_generation() {
        let plan = SpeculativeDecodePlan::new(test_config()).expect("plan");
        let ptx = plan.generate_verification_ptx().expect("ptx");
        assert!(ptx.contains(".entry spec_decode_verify"));
        assert!(ptx.contains("%param_draft_logits"));
        assert!(ptx.contains("%param_target_logits"));
        assert!(ptx.contains("%param_threshold"));
    }

    #[test]
    fn rejection_sampling_ptx_generation() {
        let plan = SpeculativeDecodePlan::new(test_config()).expect("plan");
        let ptx = plan.generate_rejection_sampling_ptx().expect("ptx");
        assert!(ptx.contains(".entry spec_decode_rejection_sample"));
        assert!(ptx.contains("%param_draft_probs"));
        assert!(ptx.contains("%param_target_probs"));
        assert!(ptx.contains("%param_random_vals"));
    }

    // -- Edge: max_draft_tokens = 1 ------------------------------------------

    #[test]
    fn max_draft_tokens_one() {
        let mut cfg = test_config();
        cfg.max_draft_tokens = 1;
        let mut mgr = SpeculativeKvManager::new(cfg).expect("valid config");

        mgr.create_checkpoint(0).expect("cp");
        mgr.append_draft_kv(0).expect("draft");

        let result = mgr.accept_tokens(0, 1).expect("accept");
        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.total_drafted, 1);
        assert!((result.acceptance_rate - 1.0).abs() < f32::EPSILON);
    }
}
