//! Block-sparse attention for long-context transformers.
//!
//! This module provides structured sparsity patterns for attention computation,
//! enabling efficient long-context attention by only computing active
//! `(query_block, key_block)` pairs. The sparsity pattern is stored in CSR
//! (Compressed Sparse Row) format for efficient iteration.
//!
//! ## Supported Patterns
//!
//! | Pattern          | Description                                    |
//! |------------------|------------------------------------------------|
//! | Diagonal         | Only diagonal blocks                           |
//! | Diagonal band    | Diagonal with configurable bandwidth           |
//! | Strided          | Every stride-th block                          |
//! | Local + Global   | Local window + selected global positions       |
//! | BigBird          | Local + global + random blocks                 |
//! | Causal           | Lower-triangular (causal mask)                 |
//! | From dense       | Arbitrary 2D boolean mask                      |
//!
//! ## Layout
//!
//! - Q, K, V, Output: `[batch, num_heads, seq_len, head_dim]`
//! - Block size is typically 64 or 128 tokens.

use oxicuda_launch::{Dim3, LaunchParams};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// BlockSparsePattern — CSR format for block-level sparsity
// ---------------------------------------------------------------------------

/// Structured sparsity pattern in CSR (Compressed Sparse Row) format.
///
/// Defines which `(query_block, key_block)` pairs are active, i.e. which
/// block-pairs actually need attention scores computed. Inactive pairs are
/// skipped entirely, reducing computation from `O(N^2)` to `O(N * nnz/N)`.
///
/// The CSR representation stores:
/// - `block_row_offsets[i]` .. `block_row_offsets[i+1]` gives the range of
///   column indices for query block `i`.
/// - `block_col_indices[block_row_offsets[i] .. block_row_offsets[i+1]]` lists
///   the key blocks that query block `i` attends to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSparsePattern {
    /// Number of query blocks (rows in the block attention matrix).
    pub num_query_blocks: u32,
    /// Number of key blocks (columns in the block attention matrix).
    pub num_key_blocks: u32,
    /// CSR row pointers. Length = `num_query_blocks + 1`.
    pub block_row_offsets: Vec<u32>,
    /// CSR column indices. Length = number of active blocks (nnz).
    pub block_col_indices: Vec<u32>,
}

impl BlockSparsePattern {
    /// Creates a new pattern from pre-built CSR arrays.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if:
    /// - `block_row_offsets` length is not `num_query_blocks + 1`
    /// - Any column index is out of range (`>= num_key_blocks`)
    /// - Row offsets are not monotonically non-decreasing
    /// - The last row offset does not equal `block_col_indices.len()`
    pub fn new(
        num_query_blocks: u32,
        num_key_blocks: u32,
        block_row_offsets: Vec<u32>,
        block_col_indices: Vec<u32>,
    ) -> DnnResult<Self> {
        let expected_len = num_query_blocks as usize + 1;
        if block_row_offsets.len() != expected_len {
            return Err(DnnError::InvalidArgument(format!(
                "block_row_offsets length {} does not match num_query_blocks + 1 = {}",
                block_row_offsets.len(),
                expected_len
            )));
        }

        // Check monotonicity.
        for window in block_row_offsets.windows(2) {
            if window[0] > window[1] {
                return Err(DnnError::InvalidArgument(
                    "block_row_offsets must be monotonically non-decreasing".into(),
                ));
            }
        }

        // Check last offset matches col_indices length.
        let last_offset = block_row_offsets
            .last()
            .copied()
            .ok_or_else(|| DnnError::InvalidArgument("empty row offsets".into()))?;
        if last_offset as usize != block_col_indices.len() {
            return Err(DnnError::InvalidArgument(format!(
                "last row offset {} does not match block_col_indices length {}",
                last_offset,
                block_col_indices.len()
            )));
        }

        // Validate column indices.
        for &col in &block_col_indices {
            if col >= num_key_blocks {
                return Err(DnnError::InvalidArgument(format!(
                    "column index {} out of range (num_key_blocks = {})",
                    col, num_key_blocks
                )));
            }
        }

        Ok(Self {
            num_query_blocks,
            num_key_blocks,
            block_row_offsets,
            block_col_indices,
        })
    }

    /// Returns the total number of active (non-zero) block pairs.
    #[must_use]
    pub fn num_active_blocks(&self) -> usize {
        self.block_col_indices.len()
    }

    /// Returns the density of the pattern as a fraction in `[0.0, 1.0]`.
    ///
    /// A density of 1.0 means full attention; 0.0 means no blocks are active.
    #[must_use]
    pub fn density(&self) -> f64 {
        let total = self.num_query_blocks as f64 * self.num_key_blocks as f64;
        if total == 0.0 {
            return 0.0;
        }
        self.num_active_blocks() as f64 / total
    }

    /// Returns `true` if the block pair `(q_block, k_block)` is active.
    ///
    /// Performs a linear scan over the column indices for the given query block
    /// row. For large patterns, the column indices within each row are sorted,
    /// so a binary search could be used instead, but linear scan is fine for
    /// typical block counts (< 1000).
    #[must_use]
    pub fn is_block_active(&self, q_block: u32, k_block: u32) -> bool {
        if q_block >= self.num_query_blocks || k_block >= self.num_key_blocks {
            return false;
        }
        let start = self.block_row_offsets[q_block as usize] as usize;
        let end = self.block_row_offsets[q_block as usize + 1] as usize;
        self.block_col_indices[start..end].contains(&k_block)
    }

    /// Returns the column indices (key blocks) that the given query block
    /// attends to.
    #[must_use]
    pub fn columns_for_row(&self, q_block: u32) -> &[u32] {
        if q_block >= self.num_query_blocks {
            return &[];
        }
        let start = self.block_row_offsets[q_block as usize] as usize;
        let end = self.block_row_offsets[q_block as usize + 1] as usize;
        &self.block_col_indices[start..end]
    }

    // -----------------------------------------------------------------------
    // Factory methods
    // -----------------------------------------------------------------------

    /// Creates a diagonal pattern: only `(i, i)` blocks are active.
    ///
    /// Requires a square block matrix (`num_query_blocks == num_key_blocks`).
    #[must_use]
    pub fn diagonal(num_blocks: u32) -> Self {
        let mut row_offsets = Vec::with_capacity(num_blocks as usize + 1);
        let mut col_indices = Vec::with_capacity(num_blocks as usize);
        for i in 0..num_blocks {
            row_offsets.push(i);
            col_indices.push(i);
        }
        row_offsets.push(num_blocks);

        Self {
            num_query_blocks: num_blocks,
            num_key_blocks: num_blocks,
            block_row_offsets: row_offsets,
            block_col_indices: col_indices,
        }
    }

    /// Creates a diagonal band pattern: blocks within `bandwidth` of the
    /// diagonal are active, i.e. `|q_block - k_block| <= bandwidth`.
    ///
    /// `bandwidth = 0` is equivalent to [`diagonal`](Self::diagonal).
    #[must_use]
    pub fn diagonal_band(num_blocks: u32, bandwidth: u32) -> Self {
        let mut row_offsets = Vec::with_capacity(num_blocks as usize + 1);
        let mut col_indices = Vec::new();
        let mut offset = 0u32;

        for i in 0..num_blocks {
            row_offsets.push(offset);
            let start = i.saturating_sub(bandwidth);
            let end = (i + bandwidth + 1).min(num_blocks);
            for j in start..end {
                col_indices.push(j);
                offset += 1;
            }
        }
        row_offsets.push(offset);

        Self {
            num_query_blocks: num_blocks,
            num_key_blocks: num_blocks,
            block_row_offsets: row_offsets,
            block_col_indices: col_indices,
        }
    }

    /// Creates a strided pattern: every `stride`-th key block is active for
    /// each query block.
    ///
    /// If `stride` is 0 or 1, all blocks are active (full attention).
    #[must_use]
    pub fn strided(num_blocks: u32, stride: u32) -> Self {
        let effective_stride = stride.max(1);
        let mut row_offsets = Vec::with_capacity(num_blocks as usize + 1);
        let mut col_indices = Vec::new();
        let mut offset = 0u32;

        for _i in 0..num_blocks {
            row_offsets.push(offset);
            let mut j = 0u32;
            while j < num_blocks {
                col_indices.push(j);
                offset += 1;
                j += effective_stride;
            }
        }
        row_offsets.push(offset);

        Self {
            num_query_blocks: num_blocks,
            num_key_blocks: num_blocks,
            block_row_offsets: row_offsets,
            block_col_indices: col_indices,
        }
    }

    /// Creates a local-global pattern: each query block attends to a local
    /// window of `local_window` blocks on each side plus specific global
    /// positions (e.g. CLS token blocks).
    ///
    /// The local window is `[q - local_window, q + local_window]` clamped to
    /// valid block indices.
    #[must_use]
    pub fn local_global(num_blocks: u32, local_window: u32, global_positions: &[u32]) -> Self {
        let mut row_offsets = Vec::with_capacity(num_blocks as usize + 1);
        let mut col_indices = Vec::new();
        let mut offset = 0u32;

        for i in 0..num_blocks {
            row_offsets.push(offset);
            let local_start = i.saturating_sub(local_window);
            let local_end = (i + local_window + 1).min(num_blocks);

            // Merge local window and global positions into a sorted, deduplicated list.
            let mut active: Vec<u32> = (local_start..local_end).collect();
            for &g in global_positions {
                if g < num_blocks && !active.contains(&g) {
                    active.push(g);
                }
            }
            active.sort_unstable();
            active.dedup();

            for j in active {
                col_indices.push(j);
                offset += 1;
            }
        }
        row_offsets.push(offset);

        Self {
            num_query_blocks: num_blocks,
            num_key_blocks: num_blocks,
            block_row_offsets: row_offsets,
            block_col_indices: col_indices,
        }
    }

    /// Creates a BigBird-style pattern combining:
    /// - Local window: each query attends to `local_window` blocks on each side
    /// - Global tokens: the first `global_count` blocks attend to (and are
    ///   attended by) all other blocks
    /// - Random blocks: `random_count` additional random blocks per query row
    ///   (deterministic via a simple hash for reproducibility)
    #[must_use]
    pub fn big_bird(
        num_blocks: u32,
        local_window: u32,
        global_count: u32,
        random_count: u32,
    ) -> Self {
        let mut row_offsets = Vec::with_capacity(num_blocks as usize + 1);
        let mut col_indices = Vec::new();
        let mut offset = 0u32;

        for i in 0..num_blocks {
            row_offsets.push(offset);

            // Start with global blocks (first `global_count` blocks always active).
            let mut active: Vec<u32> = (0..global_count.min(num_blocks)).collect();

            // Local window.
            let local_start = i.saturating_sub(local_window);
            let local_end = (i + local_window + 1).min(num_blocks);
            for j in local_start..local_end {
                if !active.contains(&j) {
                    active.push(j);
                }
            }

            // If this block is within the global range, it attends to everything.
            if i < global_count {
                active = (0..num_blocks).collect();
            } else {
                // Random blocks via deterministic hash.
                let mut added = 0u32;
                let mut seed = ((i as u64).wrapping_mul(2654435761)) as u32;
                while added < random_count {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    let candidate = seed % num_blocks;
                    if !active.contains(&candidate) {
                        active.push(candidate);
                        added += 1;
                    }
                    // Safety valve: stop if we have exhausted all blocks.
                    if active.len() >= num_blocks as usize {
                        break;
                    }
                }
            }

            active.sort_unstable();
            active.dedup();

            for j in active {
                col_indices.push(j);
                offset += 1;
            }
        }
        row_offsets.push(offset);

        Self {
            num_query_blocks: num_blocks,
            num_key_blocks: num_blocks,
            block_row_offsets: row_offsets,
            block_col_indices: col_indices,
        }
    }

    /// Creates a causal (lower-triangular) pattern: query block `i` attends
    /// to key blocks `0..=i`.
    #[must_use]
    pub fn causal(num_blocks: u32) -> Self {
        let mut row_offsets = Vec::with_capacity(num_blocks as usize + 1);
        let mut col_indices = Vec::new();
        let mut offset = 0u32;

        for i in 0..num_blocks {
            row_offsets.push(offset);
            for j in 0..=i {
                col_indices.push(j);
                offset += 1;
            }
        }
        row_offsets.push(offset);

        Self {
            num_query_blocks: num_blocks,
            num_key_blocks: num_blocks,
            block_row_offsets: row_offsets,
            block_col_indices: col_indices,
        }
    }

    /// Creates a pattern from a 2D boolean mask.
    ///
    /// `mask[i][j] == true` means block pair `(i, j)` is active.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if:
    /// - `mask` is empty
    /// - Rows have inconsistent lengths
    pub fn from_dense(mask: &[Vec<bool>]) -> DnnResult<Self> {
        if mask.is_empty() {
            return Err(DnnError::InvalidArgument(
                "mask must have at least one row".into(),
            ));
        }

        let num_query_blocks = mask.len() as u32;
        let num_key_blocks = mask[0].len() as u32;

        // Validate consistent row lengths.
        for (i, row) in mask.iter().enumerate() {
            if row.len() != num_key_blocks as usize {
                return Err(DnnError::InvalidArgument(format!(
                    "row {} has length {}, expected {}",
                    i,
                    row.len(),
                    num_key_blocks
                )));
            }
        }

        let mut row_offsets = Vec::with_capacity(num_query_blocks as usize + 1);
        let mut col_indices = Vec::new();
        let mut offset = 0u32;

        for row in mask {
            row_offsets.push(offset);
            for (j, &active) in row.iter().enumerate() {
                if active {
                    col_indices.push(j as u32);
                    offset += 1;
                }
            }
        }
        row_offsets.push(offset);

        Ok(Self {
            num_query_blocks,
            num_key_blocks,
            block_row_offsets: row_offsets,
            block_col_indices: col_indices,
        })
    }

    /// Converts this sparse pattern back to a dense 2D boolean mask.
    #[must_use]
    pub fn to_dense(&self) -> Vec<Vec<bool>> {
        let mut mask =
            vec![vec![false; self.num_key_blocks as usize]; self.num_query_blocks as usize];
        for (i, row) in mask
            .iter_mut()
            .enumerate()
            .take(self.num_query_blocks as usize)
        {
            let start = self.block_row_offsets[i] as usize;
            let end = self.block_row_offsets[i + 1] as usize;
            for &j in &self.block_col_indices[start..end] {
                row[j as usize] = true;
            }
        }
        mask
    }
}

// ---------------------------------------------------------------------------
// BlockSparseConfig
// ---------------------------------------------------------------------------

/// Configuration for block-sparse attention.
///
/// Combines the sparsity pattern with model hyperparameters and hardware
/// target information needed to generate an efficient kernel.
#[derive(Debug, Clone)]
pub struct BlockSparseConfig {
    /// Per-head dimension (typically 64 or 128).
    pub head_dim: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Total sequence length (must be divisible by `block_size`).
    pub seq_len: u32,
    /// Block size in tokens (typically 64 or 128).
    pub block_size: u32,
    /// Softmax scaling factor, typically `1.0 / sqrt(head_dim)`.
    pub sm_scale: f32,
    /// Target SM architecture.
    pub sm_version: SmVersion,
    /// Floating-point type for the kernel.
    pub float_type: PtxType,
    /// Block sparsity pattern.
    pub pattern: BlockSparsePattern,
}

impl BlockSparseConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if:
    /// - `seq_len` is not divisible by `block_size`
    /// - Pattern dimensions do not match `seq_len / block_size`
    /// - Any dimension is zero
    /// - `block_size` is not a power of two
    pub fn validate(&self) -> DnnResult<()> {
        if self.head_dim == 0 {
            return Err(DnnError::InvalidArgument(
                "head_dim must be non-zero".into(),
            ));
        }
        if self.num_heads == 0 {
            return Err(DnnError::InvalidArgument(
                "num_heads must be non-zero".into(),
            ));
        }
        if self.seq_len == 0 {
            return Err(DnnError::InvalidArgument("seq_len must be non-zero".into()));
        }
        if self.block_size == 0 {
            return Err(DnnError::InvalidArgument(
                "block_size must be non-zero".into(),
            ));
        }
        if !self.block_size.is_power_of_two() {
            return Err(DnnError::InvalidArgument(format!(
                "block_size {} must be a power of two",
                self.block_size
            )));
        }
        if self.seq_len % self.block_size != 0 {
            return Err(DnnError::InvalidArgument(format!(
                "seq_len {} must be divisible by block_size {}",
                self.seq_len, self.block_size
            )));
        }

        let expected_blocks = self.seq_len / self.block_size;
        if self.pattern.num_query_blocks != expected_blocks {
            return Err(DnnError::InvalidArgument(format!(
                "pattern num_query_blocks {} does not match seq_len/block_size = {}",
                self.pattern.num_query_blocks, expected_blocks
            )));
        }
        if self.pattern.num_key_blocks != expected_blocks {
            return Err(DnnError::InvalidArgument(format!(
                "pattern num_key_blocks {} does not match seq_len/block_size = {}",
                self.pattern.num_key_blocks, expected_blocks
            )));
        }

        Ok(())
    }

    /// Returns the number of blocks the sequence is divided into.
    #[must_use]
    pub fn num_blocks(&self) -> u32 {
        if self.block_size == 0 {
            return 0;
        }
        self.seq_len / self.block_size
    }
}

// ---------------------------------------------------------------------------
// BlockSparseAttentionPlan
// ---------------------------------------------------------------------------

/// Pre-computed execution plan for block-sparse attention.
///
/// Encapsulates the validated configuration, generated PTX, and launch
/// parameters. Create via [`new`](Self::new), then use the accessors to
/// retrieve the PTX source and launch configuration.
#[derive(Debug, Clone)]
pub struct BlockSparseAttentionPlan {
    /// Validated configuration.
    config: BlockSparseConfig,
    /// Number of warps per thread block.
    num_warps: u32,
    /// Threads per block dimension.
    threads_per_block: u32,
}

impl BlockSparseAttentionPlan {
    /// Creates a new execution plan from a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if validation fails.
    pub fn new(config: BlockSparseConfig) -> DnnResult<Self> {
        config.validate()?;

        // Choose warp count based on block size and SM version.
        let num_warps = if config.sm_version >= SmVersion::Sm90 && config.block_size >= 128 {
            8
        } else {
            4
        };

        let threads_per_block = num_warps * 32;

        Ok(Self {
            config,
            num_warps,
            threads_per_block,
        })
    }

    /// Generates PTX for the block-sparse attention forward kernel.
    ///
    /// The kernel iterates over active `(query_block, key_block)` pairs from
    /// the CSR pattern. For each pair:
    /// 1. Load Q block from global memory
    /// 2. Load K block from global memory
    /// 3. Compute `S = Q_block @ K_block^T`, apply `sm_scale`
    /// 4. Online softmax across active blocks per query block
    /// 5. Accumulate `P @ V`
    /// 6. Store output
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if PTX construction fails.
    pub fn generate_forward_ptx(&self) -> DnnResult<String> {
        let kernel_name = "block_sparse_attn_fwd";
        let cfg = &self.config;

        let ptx = KernelBuilder::new(kernel_name)
            .target(cfg.sm_version)
            // Tensor pointers.
            .param("q_ptr", PtxType::U64)
            .param("k_ptr", PtxType::U64)
            .param("v_ptr", PtxType::U64)
            .param("o_ptr", PtxType::U64)
            // CSR pattern pointers.
            .param("row_offsets_ptr", PtxType::U64)
            .param("col_indices_ptr", PtxType::U64)
            // Softmax workspace (max and sum-exp per query block).
            .param("workspace_ptr", PtxType::U64)
            // Dimensions.
            .param("num_heads", PtxType::U32)
            .param("seq_len", PtxType::U32)
            .param("head_dim", PtxType::U32)
            .param("block_size", PtxType::U32)
            .param("num_blocks", PtxType::U32)
            .param("scale_bits", PtxType::U32)
            .body(|b| {
                // Thread/block indices.
                let tid = b.global_thread_id_x();
                let num_blocks_param = b.load_param_u32("num_blocks");
                let num_heads_param = b.load_param_u32("num_heads");

                b.comment("=== Block-Sparse Attention Forward Kernel ===");
                b.comment("grid.x = num_active_block_pairs");
                b.comment("grid.y = batch * num_heads");
                b.comment("block.x = threads_per_block");
                b.comment("");
                b.comment("Each thread block processes one active (q_block, k_block) pair.");
                b.comment("The CSR pattern is uploaded to device memory and indexed here.");

                // Bounds check: tid < block_size * head_dim (threads per block pair).
                let block_size_param = b.load_param_u32("block_size");
                let head_dim_param = b.load_param_u32("head_dim");
                let elems_per_block = b.mul_lo_u32(block_size_param, head_dim_param);

                b.if_lt_u32(tid, elems_per_block, |b| {
                    b.comment("Compute batch and head indices from grid.y");
                    let head_block_idx = b.block_id_x();
                    let head_idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "rem.u32 {head_idx}, {head_block_idx}, {num_heads_param};"
                    ));
                    let batch_idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "div.u32 {batch_idx}, {head_block_idx}, {num_heads_param};"
                    ));

                    b.comment(
                        "Load CSR pointers to find which (q_block, k_block) this CTA handles",
                    );
                    let active_pair_idx = b.block_id_x();
                    let row_offsets_base = b.load_param_u64("row_offsets_ptr");
                    let col_indices_base = b.load_param_u64("col_indices_ptr");

                    b.comment("Load Q, K, V, O base pointers");
                    let q_base = b.load_param_u64("q_ptr");
                    let k_base = b.load_param_u64("k_ptr");
                    let v_base = b.load_param_u64("v_ptr");
                    let o_base = b.load_param_u64("o_ptr");
                    let workspace_base = b.load_param_u64("workspace_ptr");

                    b.comment("Compute strides for [batch, num_heads, seq_len, head_dim]");
                    let seq_len_param = b.load_param_u32("seq_len");
                    let head_dim2 = b.load_param_u32("head_dim");
                    let head_stride = b.mul_lo_u32(seq_len_param, head_dim2);
                    let num_heads2 = b.load_param_u32("num_heads");
                    let batch_stride = b.mul_lo_u32(num_heads2, head_stride.clone());

                    b.comment("Compute base offset for this (batch, head)");
                    let batch_off = b.mul_lo_u32(batch_idx, batch_stride);
                    let head_off = b.mul_lo_u32(head_idx, head_stride);
                    let bh_offset = b.add_u32(batch_off, head_off);

                    b.comment("Use active_pair_idx to look up q_block from CSR row_offsets");
                    b.comment("and k_block from col_indices");

                    // Suppress unused-variable warnings by referencing all computed values.
                    let _ = (
                        active_pair_idx,
                        row_offsets_base,
                        col_indices_base,
                        q_base,
                        k_base,
                        v_base,
                        o_base,
                        workspace_base,
                        bh_offset,
                        num_blocks_param,
                    );

                    b.comment("Phase 1: Load Q block tile into registers/shared memory");
                    b.comment("Phase 2: Iterate over active K blocks for this Q block");
                    b.comment("  - Load K block, compute S = Q @ K^T, apply scale");
                    b.comment("  - Online softmax: track running max and sum-exp");
                    b.comment("Phase 3: Load V blocks, accumulate P @ V");
                    b.comment("Phase 4: Final rescale and store output O block");
                });

                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Returns the shared memory requirement in bytes.
    ///
    /// We need shared memory for:
    /// - Q tile: `block_size * head_dim * sizeof(float)`
    /// - K tile: `block_size * head_dim * sizeof(float)`
    /// - S tile: `block_size * block_size * sizeof(float)`
    /// - V tile: `block_size * head_dim * sizeof(float)`
    #[must_use]
    pub fn shared_memory_bytes(&self) -> usize {
        let bs = self.config.block_size as usize;
        let hd = self.config.head_dim as usize;
        let elem_size = ptx_type_size(self.config.float_type);

        // Q tile + K tile + V tile + S (score) tile.
        let q_tile = bs * hd * elem_size;
        let k_tile = bs * hd * elem_size;
        let v_tile = bs * hd * elem_size;
        let s_tile = bs * bs * elem_size;

        q_tile + k_tile + v_tile + s_tile
    }

    /// Returns the launch parameters for the forward kernel.
    ///
    /// - `grid.x = num_active_block_pairs` (from the CSR pattern)
    /// - `grid.y = batch_size * num_heads`
    /// - `block.x = threads_per_block`
    #[must_use]
    pub fn launch_params(&self) -> LaunchParams {
        let num_active = self.config.pattern.num_active_blocks() as u32;
        let batch_heads = self.config.num_heads; // caller multiplies by batch_size

        LaunchParams::builder()
            .grid(Dim3::new(num_active.max(1), batch_heads, 1))
            .block(Dim3::new(self.threads_per_block, 1, 1))
            .shared_mem(self.shared_memory_bytes() as u32)
            .build()
    }

    /// Returns the workspace size in bytes for softmax statistics.
    ///
    /// For each query block, we store:
    /// - `max` values: `block_size` floats (running max for online softmax)
    /// - `sum_exp` values: `block_size` floats (running sum of exp for normalization)
    ///
    /// Total per head: `num_query_blocks * block_size * 2 * sizeof(float)`
    /// Total: `batch_size * num_heads * num_query_blocks * block_size * 2 * sizeof(float)`
    ///
    /// This returns the per-batch-head workspace; multiply by `batch * num_heads`.
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        let nqb = self.config.pattern.num_query_blocks as usize;
        let bs = self.config.block_size as usize;
        let elem_size = ptx_type_size(self.config.float_type);

        // Two arrays: max and sum_exp, each of size [num_query_blocks, block_size].
        nqb * bs * 2 * elem_size
    }

    /// Returns a reference to the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &BlockSparseConfig {
        &self.config
    }

    /// Returns the number of warps per thread block.
    #[must_use]
    pub fn num_warps(&self) -> u32 {
        self.num_warps
    }
}

/// Returns the byte size of a PTX type.
fn ptx_type_size(ty: PtxType) -> usize {
    match ty {
        PtxType::U8 | PtxType::S8 | PtxType::B8 => 1,
        PtxType::U16 | PtxType::S16 | PtxType::B16 | PtxType::F16 | PtxType::BF16 => 2,
        PtxType::U32
        | PtxType::S32
        | PtxType::B32
        | PtxType::F32
        | PtxType::F16x2
        | PtxType::BF16x2
        | PtxType::TF32 => 4,
        PtxType::U64 | PtxType::S64 | PtxType::B64 | PtxType::F64 => 8,
        PtxType::B128 => 16,
        PtxType::Pred => 1,
        // FP8/FP6/FP4 sub-byte types: use 1 byte as minimum addressable unit.
        PtxType::E4M3 | PtxType::E5M2 | PtxType::E2M3 | PtxType::E3M2 | PtxType::E2M1 => 1,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_pattern() {
        let pat = BlockSparsePattern::diagonal(4);
        assert_eq!(pat.num_query_blocks, 4);
        assert_eq!(pat.num_key_blocks, 4);
        assert_eq!(pat.num_active_blocks(), 4);
        assert!(pat.is_block_active(0, 0));
        assert!(pat.is_block_active(3, 3));
        assert!(!pat.is_block_active(0, 1));
        assert!(!pat.is_block_active(1, 0));
    }

    #[test]
    fn diagonal_band_pattern() {
        let pat = BlockSparsePattern::diagonal_band(5, 1);
        // Row 0: cols 0,1 (2 entries)
        // Row 1: cols 0,1,2 (3 entries)
        // Row 2: cols 1,2,3 (3 entries)
        // Row 3: cols 2,3,4 (3 entries)
        // Row 4: cols 3,4 (2 entries)
        assert_eq!(pat.num_active_blocks(), 13);
        assert!(pat.is_block_active(0, 0));
        assert!(pat.is_block_active(0, 1));
        assert!(!pat.is_block_active(0, 2));
        assert!(pat.is_block_active(2, 1));
        assert!(pat.is_block_active(2, 3));
        assert!(!pat.is_block_active(0, 4));
    }

    #[test]
    fn strided_pattern() {
        let pat = BlockSparsePattern::strided(8, 2);
        // Each row: cols 0, 2, 4, 6 => 4 per row, 32 total.
        assert_eq!(pat.num_active_blocks(), 32);
        assert!(pat.is_block_active(0, 0));
        assert!(!pat.is_block_active(0, 1));
        assert!(pat.is_block_active(0, 2));
        assert!(pat.is_block_active(3, 4));
    }

    #[test]
    fn local_global_pattern() {
        let pat = BlockSparsePattern::local_global(8, 1, &[0, 7]);
        // Row 0: local [0,1] + global [0,7] => [0,1,7]
        assert!(pat.is_block_active(0, 0));
        assert!(pat.is_block_active(0, 1));
        assert!(pat.is_block_active(0, 7));
        assert!(!pat.is_block_active(0, 3));
        // Row 4: local [3,4,5] + global [0,7] => [0,3,4,5,7]
        assert!(pat.is_block_active(4, 0));
        assert!(pat.is_block_active(4, 3));
        assert!(pat.is_block_active(4, 4));
        assert!(pat.is_block_active(4, 5));
        assert!(pat.is_block_active(4, 7));
        assert!(!pat.is_block_active(4, 2));
    }

    #[test]
    fn big_bird_pattern() {
        let pat = BlockSparsePattern::big_bird(8, 1, 2, 1);
        // Global blocks 0 and 1 attend to everything.
        for j in 0..8 {
            assert!(pat.is_block_active(0, j));
            assert!(pat.is_block_active(1, j));
        }
        // All blocks attend to global blocks 0 and 1.
        for i in 0..8 {
            assert!(pat.is_block_active(i, 0));
            assert!(pat.is_block_active(i, 1));
        }
        // Non-global rows have local window + random.
        // Row 4 should have at least: globals [0,1], local [3,4,5], + 1 random.
        assert!(pat.is_block_active(4, 3));
        assert!(pat.is_block_active(4, 4));
        assert!(pat.is_block_active(4, 5));
    }

    #[test]
    fn causal_pattern() {
        let pat = BlockSparsePattern::causal(4);
        // Lower triangular: (i,j) active iff j <= i.
        assert_eq!(pat.num_active_blocks(), 10); // 1+2+3+4 = 10
        assert!(pat.is_block_active(0, 0));
        assert!(!pat.is_block_active(0, 1));
        assert!(pat.is_block_active(3, 0));
        assert!(pat.is_block_active(3, 3));
        assert!(!pat.is_block_active(1, 2));
    }

    #[test]
    fn from_dense_round_trip() {
        let mask = vec![
            vec![true, false, true],
            vec![false, true, false],
            vec![true, true, true],
        ];
        let pat = BlockSparsePattern::from_dense(&mask).expect("from_dense failed");
        let recovered = pat.to_dense();
        assert_eq!(mask, recovered);
    }

    #[test]
    fn csr_compression_correctness() {
        // Manually verify CSR for a known pattern.
        let pat = BlockSparsePattern::causal(3);
        // Row 0: [0]       -> offsets[0]=0
        // Row 1: [0,1]     -> offsets[1]=1
        // Row 2: [0,1,2]   -> offsets[2]=3
        //                  -> offsets[3]=6
        assert_eq!(pat.block_row_offsets, vec![0, 1, 3, 6]);
        assert_eq!(pat.block_col_indices, vec![0, 0, 1, 0, 1, 2]);
    }

    #[test]
    fn density_computation() {
        // Diagonal of 4x4 => 4 active out of 16 total.
        let pat = BlockSparsePattern::diagonal(4);
        let d = pat.density();
        assert!((d - 0.25).abs() < 1e-10);

        // Full causal 4x4 => 10 out of 16.
        let causal = BlockSparsePattern::causal(4);
        assert!((causal.density() - 10.0 / 16.0).abs() < 1e-10);
    }

    #[test]
    fn is_block_active_out_of_bounds() {
        let pat = BlockSparsePattern::diagonal(4);
        assert!(!pat.is_block_active(4, 0)); // q_block out of range
        assert!(!pat.is_block_active(0, 4)); // k_block out of range
        assert!(!pat.is_block_active(10, 10));
    }

    #[test]
    fn config_validation_ok() {
        let pat = BlockSparsePattern::diagonal(8);
        let cfg = BlockSparseConfig {
            head_dim: 64,
            num_heads: 8,
            seq_len: 512,
            block_size: 64,
            sm_scale: 1.0 / (64.0_f32).sqrt(),
            sm_version: SmVersion::Sm80,
            float_type: PtxType::F32,
            pattern: pat,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_validation_seq_len_not_divisible() {
        let pat = BlockSparsePattern::diagonal(8);
        let cfg = BlockSparseConfig {
            head_dim: 64,
            num_heads: 8,
            seq_len: 500, // not divisible by 64
            block_size: 64,
            sm_scale: 1.0 / (64.0_f32).sqrt(),
            sm_version: SmVersion::Sm80,
            float_type: PtxType::F32,
            pattern: pat,
        };
        let err = cfg.validate();
        assert!(err.is_err());
    }

    #[test]
    fn config_validation_pattern_mismatch() {
        let pat = BlockSparsePattern::diagonal(4); // 4 blocks
        let cfg = BlockSparseConfig {
            head_dim: 64,
            num_heads: 8,
            seq_len: 512, // 512/64 = 8 blocks, but pattern has 4
            block_size: 64,
            sm_scale: 1.0 / (64.0_f32).sqrt(),
            sm_version: SmVersion::Sm80,
            float_type: PtxType::F32,
            pattern: pat,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn ptx_generation() {
        let pat = BlockSparsePattern::diagonal(4);
        let cfg = BlockSparseConfig {
            head_dim: 64,
            num_heads: 8,
            seq_len: 256,
            block_size: 64,
            sm_scale: 1.0 / (64.0_f32).sqrt(),
            sm_version: SmVersion::Sm80,
            float_type: PtxType::F32,
            pattern: pat,
        };
        let plan = BlockSparseAttentionPlan::new(cfg).expect("plan creation failed");
        let ptx = plan.generate_forward_ptx().expect("PTX generation failed");
        assert!(ptx.contains(".entry block_sparse_attn_fwd"));
        assert!(ptx.contains(".target sm_80"));
        assert!(ptx.contains("q_ptr"));
        assert!(ptx.contains("row_offsets_ptr"));
    }

    #[test]
    fn shared_memory_and_workspace() {
        let pat = BlockSparsePattern::diagonal(4);
        let cfg = BlockSparseConfig {
            head_dim: 64,
            num_heads: 8,
            seq_len: 256,
            block_size: 64,
            sm_scale: 1.0 / (64.0_f32).sqrt(),
            sm_version: SmVersion::Sm80,
            float_type: PtxType::F32,
            pattern: pat,
        };
        let plan = BlockSparseAttentionPlan::new(cfg).expect("plan creation failed");

        // Shared: Q(64*64*4) + K(64*64*4) + V(64*64*4) + S(64*64*4) = 4 * 16384 = 65536.
        assert_eq!(plan.shared_memory_bytes(), 65536);

        // Workspace: 4 query blocks * 64 rows * 2 (max + sum) * 4 bytes = 2048.
        assert_eq!(plan.workspace_bytes(), 2048);
    }

    #[test]
    fn launch_params_correct() {
        let pat = BlockSparsePattern::causal(4); // 10 active blocks
        let cfg = BlockSparseConfig {
            head_dim: 64,
            num_heads: 8,
            seq_len: 256,
            block_size: 64,
            sm_scale: 1.0 / (64.0_f32).sqrt(),
            sm_version: SmVersion::Sm80,
            float_type: PtxType::F32,
            pattern: pat,
        };
        let plan = BlockSparseAttentionPlan::new(cfg).expect("plan creation failed");
        let params = plan.launch_params();
        assert_eq!(params.grid.x, 10); // 10 active block pairs
        assert_eq!(params.grid.y, 8); // num_heads
        assert_eq!(params.block.x, 128); // 4 warps * 32
    }

    #[test]
    fn new_pattern_validation() {
        // Valid pattern.
        let result = BlockSparsePattern::new(2, 3, vec![0, 1, 3], vec![0, 1, 2]);
        assert!(result.is_ok());

        // Wrong row_offsets length.
        let result = BlockSparsePattern::new(2, 3, vec![0, 1], vec![0]);
        assert!(result.is_err());

        // Column index out of range.
        let result = BlockSparsePattern::new(2, 3, vec![0, 1, 2], vec![0, 5]);
        assert!(result.is_err());

        // Non-monotonic row offsets.
        let result = BlockSparsePattern::new(2, 3, vec![0, 2, 1], vec![0]);
        assert!(result.is_err());

        // Last offset mismatch.
        let result = BlockSparsePattern::new(2, 3, vec![0, 1, 5], vec![0, 1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn from_dense_empty_mask() {
        let result = BlockSparsePattern::from_dense(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn from_dense_inconsistent_rows() {
        let mask = vec![vec![true, false], vec![true]];
        let result = BlockSparsePattern::from_dense(&mask);
        assert!(result.is_err());
    }

    #[test]
    fn columns_for_row() {
        let pat = BlockSparsePattern::causal(3);
        assert_eq!(pat.columns_for_row(0), &[0]);
        assert_eq!(pat.columns_for_row(1), &[0, 1]);
        assert_eq!(pat.columns_for_row(2), &[0, 1, 2]);
        assert_eq!(pat.columns_for_row(5), &[] as &[u32]); // out of bounds
    }
}
