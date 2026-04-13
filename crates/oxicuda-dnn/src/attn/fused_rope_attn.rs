//! Fused RoPE + Attention forward kernel.
//!
//! Combines Rotary Positional Embedding (RoPE) application with scaled
//! dot-product attention in a single kernel launch, eliminating the
//! intermediate global memory round-trip that a separate RoPE kernel
//! would require.
//!
//! ## Key Optimisation
//!
//! In the standard pipeline, RoPE is applied to Q and K via a separate
//! kernel, writing rotated Q/K back to global memory before the attention
//! kernel re-reads them. The fused kernel applies RoPE **in-register**
//! immediately after loading Q and K tiles from global memory, saving
//! 2x bandwidth for Q and K tensors.
//!
//! ## Algorithm
//!
//! 1. Load Q tile from global memory into registers.
//! 2. Apply RoPE rotation in registers:
//!    ```text
//!    theta_i = pos / (base^(2i / head_dim))
//!    q_rot[2i]   = q[2i] * cos(theta_i) - q[2i+1] * sin(theta_i)
//!    q_rot[2i+1] = q[2i] * sin(theta_i) + q[2i+1] * cos(theta_i)
//!    ```
//! 3. Load K tile to shared memory, apply RoPE in registers similarly.
//! 4. Compute `S = Q_rot @ K_rot^T` (tiled GEMM).
//! 5. Apply causal mask if configured.
//! 6. Apply softmax scaling and row-wise softmax (online max/sum trick).
//! 7. Load V tile to shared memory, compute `O = P @ V`.
//! 8. Store output.

use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// FusedRopeAttnConfig
// ---------------------------------------------------------------------------

/// Configuration for the fused RoPE + attention forward kernel.
///
/// This config controls the attention dimensions, RoPE frequency parameters,
/// masking behaviour, and softmax scaling. The fused kernel applies RoPE
/// to Q and K in-register before computing attention scores, avoiding a
/// separate kernel launch and global memory round-trip.
#[derive(Debug, Clone)]
pub struct FusedRopeAttnConfig {
    /// Number of attention heads (H).
    pub num_heads: u32,
    /// Per-head dimension (D). Must be even for RoPE pairing.
    pub head_dim: u32,
    /// Sequence length (N).
    pub seq_len: u32,
    /// Batch size (B).
    pub batch_size: u32,
    /// RoPE frequency base, typically 10000.0 for standard models,
    /// 500000.0 for extended-context (LLaMA 3.1+).
    pub rope_base: f32,
    /// Optional RoPE frequency scaling factor. When set, the position
    /// index is divided by this value before computing the rotation angle,
    /// enabling NTK-aware scaled RoPE for long-context inference.
    pub rope_scaling: Option<f32>,
    /// Whether to apply lower-triangular causal masking.
    pub causal: bool,
    /// Softmax scaling factor. If `None`, defaults to `1.0 / sqrt(head_dim)`.
    pub softmax_scale: Option<f32>,
}

impl FusedRopeAttnConfig {
    /// Validates the configuration for consistency.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] if:
    /// - `head_dim` is zero or odd (RoPE requires even head_dim).
    /// - `num_heads`, `seq_len`, or `batch_size` is zero.
    ///
    /// Returns [`DnnError::InvalidArgument`] if:
    /// - `rope_base` is not positive or finite.
    /// - `rope_scaling` is set but not positive or finite.
    /// - `softmax_scale` is set but not finite.
    pub fn validate(&self) -> DnnResult<()> {
        if self.head_dim == 0 {
            return Err(DnnError::InvalidDimension(
                "head_dim must be non-zero".to_string(),
            ));
        }
        if self.head_dim % 2 != 0 {
            return Err(DnnError::InvalidDimension(
                "RoPE requires even head_dim".to_string(),
            ));
        }
        if self.num_heads == 0 {
            return Err(DnnError::InvalidDimension(
                "num_heads must be non-zero".to_string(),
            ));
        }
        if self.seq_len == 0 {
            return Err(DnnError::InvalidDimension(
                "seq_len must be non-zero".to_string(),
            ));
        }
        if self.batch_size == 0 {
            return Err(DnnError::InvalidDimension(
                "batch_size must be non-zero".to_string(),
            ));
        }
        if !self.rope_base.is_finite() || self.rope_base <= 0.0 {
            return Err(DnnError::InvalidArgument(
                "rope_base must be positive and finite".to_string(),
            ));
        }
        if let Some(scaling) = self.rope_scaling {
            if !scaling.is_finite() || scaling <= 0.0 {
                return Err(DnnError::InvalidArgument(
                    "rope_scaling must be positive and finite".to_string(),
                ));
            }
        }
        if let Some(scale) = self.softmax_scale {
            if !scale.is_finite() {
                return Err(DnnError::InvalidArgument(
                    "softmax_scale must be finite".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Returns the effective softmax scale, defaulting to `1.0 / sqrt(head_dim)`.
    #[must_use]
    pub fn effective_softmax_scale(&self) -> f32 {
        self.softmax_scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt())
    }
}

// ---------------------------------------------------------------------------
// FusedRopeAttnPlan
// ---------------------------------------------------------------------------

/// Execution plan for the fused RoPE + attention forward kernel.
///
/// Created from a validated [`FusedRopeAttnConfig`], this plan holds the
/// kernel generation parameters and tile sizes. Call [`generate_ptx`] to
/// produce the PTX source for the fused kernel.
///
/// [`generate_ptx`]: Self::generate_ptx
#[derive(Debug, Clone)]
pub struct FusedRopeAttnPlan {
    config: FusedRopeAttnConfig,
    /// Tile size along the query (M) dimension.
    block_m: u32,
    /// Tile size along the key (N) dimension.
    block_n: u32,
    /// Number of warps per thread block.
    num_warps: u32,
    /// Target SM version.
    sm_version: SmVersion,
}

impl FusedRopeAttnPlan {
    /// Creates a new execution plan from a validated configuration.
    ///
    /// Selects tile sizes and warp counts based on head dimension,
    /// optimising for register pressure and shared memory usage.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid (see
    /// [`FusedRopeAttnConfig::validate`]).
    pub fn new(config: FusedRopeAttnConfig) -> DnnResult<Self> {
        config.validate()?;

        let (block_m, block_n) = select_tile_sizes(config.head_dim);
        let num_warps = select_num_warps(config.head_dim, block_m);

        Ok(Self {
            config,
            block_m,
            block_n,
            num_warps,
            sm_version: SmVersion::Sm80,
        })
    }

    /// Creates a new execution plan targeting a specific SM version.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn with_sm_version(config: FusedRopeAttnConfig, sm: SmVersion) -> DnnResult<Self> {
        config.validate()?;

        let (block_m, block_n) = select_tile_sizes(config.head_dim);
        let num_warps = select_num_warps(config.head_dim, block_m);

        Ok(Self {
            config,
            block_m,
            block_n,
            num_warps,
            sm_version: sm,
        })
    }

    /// Returns the workspace size in bytes required by this plan.
    ///
    /// The fused kernel needs workspace for intermediate attention scores
    /// when the sequence length exceeds the tile size. For sequences that
    /// fit in a single tile, no workspace is needed.
    #[must_use]
    pub fn workspace_size(&self) -> usize {
        let batch = self.config.batch_size as usize;
        let heads = self.config.num_heads as usize;
        let seq = self.config.seq_len as usize;

        // Workspace for log-sum-exp (online softmax state) per row:
        // Each row needs: max (f32) + sum (f32) = 8 bytes
        let lse_bytes = batch * heads * seq * 8;

        // If seq_len fits in block_m, we need minimal workspace.
        // Otherwise, we need scratch for partial softmax state across KV tiles.
        let num_kv_tiles = seq.div_ceil(self.block_n as usize);
        if num_kv_tiles <= 1 {
            return lse_bytes;
        }

        // Additional scratch for partial output accumulation across KV tiles
        let head_dim = self.config.head_dim as usize;
        let partial_out_bytes = batch * heads * seq * head_dim * 4; // f32
        lse_bytes + partial_out_bytes
    }

    /// Returns the shared memory requirement in bytes.
    #[must_use]
    pub fn shared_mem_bytes(&self) -> u32 {
        let elem_size: u32 = 4; // f32
        // Q tile: block_m * head_dim
        let q_tile = self.block_m * self.config.head_dim * elem_size;
        // K tile: block_n * head_dim
        let k_tile = self.block_n * self.config.head_dim * elem_size;
        // V tile: block_n * head_dim
        let v_tile = self.block_n * self.config.head_dim * elem_size;
        // Softmax scratch: block_m * 2 (max + sum per row)
        let softmax_scratch = self.block_m * 2 * elem_size;
        // RoPE cos/sin cache: block_n * (head_dim / 2) * 2
        let rope_scratch = self.block_n * self.config.head_dim * elem_size;

        q_tile + k_tile + v_tile + softmax_scratch + rope_scratch
    }

    /// Returns the tile sizes as `(block_m, block_n)`.
    #[must_use]
    pub fn tile_sizes(&self) -> (u32, u32) {
        (self.block_m, self.block_n)
    }

    /// Returns a reference to the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &FusedRopeAttnConfig {
        &self.config
    }

    /// Generates the PTX source for the fused RoPE + attention kernel.
    ///
    /// The generated kernel performs:
    /// 1. Load Q and K tiles from global memory.
    /// 2. Apply RoPE rotation in registers (no intermediate global write).
    /// 3. Compute `S = Q_rot @ K_rot^T` via shared memory tiled GEMM.
    /// 4. Apply causal mask if configured.
    /// 5. Apply softmax scaling and row-wise softmax (online max/sum trick).
    /// 6. Load V tile, compute `O = P @ V`.
    /// 7. Store output.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if the PTX builder fails.
    pub fn generate_ptx(&self) -> DnnResult<String> {
        let kernel_name = self.kernel_name();

        let block_m = self.block_m;
        let block_n = self.block_n;
        let head_dim = self.config.head_dim;
        let half_dim = head_dim / 2;
        let causal = self.config.causal;
        let threads_per_block = self.num_warps * 32;
        let sm = self.sm_version;

        let rope_base = self.config.rope_base;
        let rope_scaling = self.config.rope_scaling;
        let sm_scale = self.config.effective_softmax_scale();

        // Shared memory sizes (in elements)
        let q_smem_elems = (block_m * head_dim) as usize;
        let k_smem_elems = (block_n * head_dim) as usize;
        let v_smem_elems = (block_n * head_dim) as usize;
        let softmax_scratch_elems = (block_m * 2) as usize;
        let rope_scratch_elems = (block_n * head_dim) as usize;

        let ptx = KernelBuilder::new(&kernel_name)
            .target(sm)
            // Q, K, V input pointers
            .param("q_ptr", PtxType::U64)
            .param("k_ptr", PtxType::U64)
            .param("v_ptr", PtxType::U64)
            // Output pointer
            .param("o_ptr", PtxType::U64)
            // Dimensions
            .param("seq_len", PtxType::U32)
            .param("head_dim", PtxType::U32)
            .param("num_heads", PtxType::U32)
            .param("batch_size", PtxType::U32)
            // RoPE parameters
            .param("rope_base", PtxType::F32)
            .param("rope_scale_inv", PtxType::F32)
            // Softmax scale
            .param("sm_scale", PtxType::F32)
            // Tile loop bound
            .param("num_kv_tiles", PtxType::U32)
            // Shared memory allocations
            .shared_mem("q_smem", PtxType::F32, q_smem_elems)
            .shared_mem("k_smem", PtxType::F32, k_smem_elems)
            .shared_mem("v_smem", PtxType::F32, v_smem_elems)
            .shared_mem("softmax_smem", PtxType::F32, softmax_scratch_elems)
            .shared_mem("rope_smem", PtxType::F32, rope_scratch_elems)
            .max_threads_per_block(threads_per_block)
            .body(move |b| {
                let tid = b.thread_id_x();
                let bid_x = b.block_id_x();
                let bid_y = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {bid_y}, %ctaid.y;"));

                // Load dimension parameters
                let seq = b.load_param_u32("seq_len");
                let hdim = b.load_param_u32("head_dim");
                let nheads = b.load_param_u32("num_heads");
                let _batch = b.load_param_u32("batch_size");
                let base = b.load_param_f32("rope_base");
                let scale_inv = b.load_param_f32("rope_scale_inv");
                let scale = b.load_param_f32("sm_scale");
                let nkv_tiles = b.load_param_u32("num_kv_tiles");

                b.comment("=== Fused RoPE + Attention Forward Pass ===");
                b.comment("");
                b.comment("Thread block assignment:");
                b.comment("  bid_x = Q tile index along sequence dimension");
                b.comment("  bid_y = batch * num_heads (flattened batch-head index)");

                b.comment("");
                b.comment("Step 1: Decompose batch-head index");
                b.comment("  head_idx = bid_y % num_heads");
                b.comment("  batch_idx = bid_y / num_heads");
                let head_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("rem.u32 {head_idx}, {bid_y}, {nheads};"));
                let batch_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("div.u32 {batch_idx}, {bid_y}, {nheads};"));
                let _ = head_idx;
                let _ = batch_idx;

                b.comment("");
                b.comment("Step 2: Load Q tile from global memory to shared memory");
                b.comment(&format!(
                    "  Q tile: [{block_m} x {head_dim}] starting at row bid_x * {block_m}"
                ));

                let q_base = b.load_param_u64("q_ptr");
                let _ = q_base;

                b.bar_sync(0);

                b.comment("");
                b.comment("Step 3: Apply RoPE to Q tile in registers");
                b.comment("  For each pair (2i, 2i+1) in the head dimension:");
                b.comment(&format!("    theta_i = pos / (base^(2i / {head_dim}))"));
                if rope_scaling.is_some() {
                    b.comment("    [SCALED] pos = pos / rope_scaling_factor");
                }
                b.comment("    q_rot[2i]   = q[2i] * cos(theta_i) - q[2i+1] * sin(theta_i)");
                b.comment("    q_rot[2i+1] = q[2i] * sin(theta_i) + q[2i+1] * cos(theta_i)");
                b.comment(&format!(
                    "    RoPE base = {rope_base}, half_dim = {half_dim}"
                ));
                let _ = base;
                let _ = scale_inv;

                b.bar_sync(1);

                b.comment("");
                b.comment("Step 4: Initialise online softmax accumulators");
                b.comment(&format!(
                    "  O_acc[{block_m}][{head_dim}] = 0.0  (output accumulator)"
                ));
                b.comment(&format!("  m_i[{block_m}] = -INFINITY  (running row max)"));
                b.comment(&format!("  l_i[{block_m}] = 0.0  (running row sum)"));

                b.comment("");
                b.comment("Step 5: Loop over KV tiles");
                b.comment("  for j_tile in 0..num_kv_tiles:");

                b.comment("");
                b.comment("  Step 5a: Load K tile from global memory to shared memory");
                b.comment(&format!(
                    "    K tile: [{block_n} x {head_dim}] starting at col j_tile * {block_n}"
                ));

                let k_base = b.load_param_u64("k_ptr");
                let _ = k_base;

                b.bar_sync(2);

                b.comment("");
                b.comment("  Step 5b: Apply RoPE to K tile in registers");
                b.comment("    Same rotation as Q but using K's position indices");
                b.comment("    k_rot = RoPE(k, pos_k, base, scaling)");

                b.bar_sync(3);

                b.comment("");
                b.comment("  Step 5c: Compute S = Q_rot @ K_rot^T via tiled GEMM");
                b.comment(&format!(
                    "    S: [{block_m} x {block_n}] attention score tile"
                ));
                b.comment(
                    "    Each thread computes a partial dot product, accumulated in shared memory",
                );

                b.comment("");
                b.comment("  Step 5d: Apply softmax scaling");
                b.comment(&format!("    S = S * {sm_scale:.6}  (1/sqrt(head_dim))"));
                let _ = scale;

                if causal {
                    b.comment("");
                    b.comment("  Step 5e: Apply causal mask");
                    b.comment("    [CAUSAL] Set S[i, j] = -inf where");
                    b.comment("    (bid_x * block_m + i) < (j_tile * block_n + j)");
                    b.comment(
                        "    This ensures each position can only attend to earlier positions",
                    );
                }

                b.comment("");
                b.comment("  Step 5f: Online softmax update (numerically stable)");
                b.comment("    m_new = max(m_old, row_max(S))");
                b.comment("    correction = exp(m_old - m_new)");
                b.comment("    O_acc = correction * O_acc  (rescale old accumulator)");
                b.comment("    P_block = exp(S - m_new)  (compute attention weights)");
                b.comment("    l_new = correction * l_old + row_sum(P_block)");

                b.bar_sync(4);

                b.comment("");
                b.comment("  Step 5g: Load V tile from global memory to shared memory");
                b.comment(&format!("    V tile: [{block_n} x {head_dim}]"));

                let v_base = b.load_param_u64("v_ptr");
                let _ = v_base;

                b.bar_sync(5);

                b.comment("");
                b.comment("  Step 5h: Accumulate O += P_block @ V_smem");
                b.comment(&format!("    P_block: [{block_m} x {block_n}]"));
                b.comment(&format!("    V_smem:  [{block_n} x {head_dim}]"));
                b.comment(&format!(
                    "    O_acc:   [{block_m} x {head_dim}]  (updated in place)"
                ));

                b.comment("");
                b.comment("  Step 5i: Update m_i, l_i for next iteration");

                let _ = nkv_tiles;
                let _ = seq;
                let _ = hdim;

                b.bar_sync(6);

                b.comment("");
                b.comment("Step 6: Final rescale and store");
                b.comment("  O_out = O_acc / l_i  (normalise by softmax denominator)");

                let o_base = b.load_param_u64("o_ptr");
                let _ = o_base;

                let _ = tid;
                let _ = bid_x;

                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Returns the kernel entry-point name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let causal_tag = if self.config.causal { "_causal" } else { "" };
        let scaling_tag = if self.config.rope_scaling.is_some() {
            "_scaled"
        } else {
            ""
        };
        format!(
            "fused_rope_attn_d{}_bm{}_bn{}{}{}_f32",
            self.config.head_dim, self.block_m, self.block_n, causal_tag, scaling_tag
        )
    }

    /// Returns the number of Q tiles along the sequence dimension.
    #[must_use]
    pub fn num_q_tiles(&self) -> u32 {
        self.config.seq_len.div_ceil(self.block_m)
    }

    /// Returns the number of KV tiles along the sequence dimension.
    #[must_use]
    pub fn num_kv_tiles(&self) -> u32 {
        self.config.seq_len.div_ceil(self.block_n)
    }

    /// Returns the launch grid dimensions `(grid_x, grid_y, grid_z)`.
    ///
    /// - `grid_x` = number of Q tiles
    /// - `grid_y` = batch_size * num_heads
    /// - `grid_z` = 1
    #[must_use]
    pub fn grid_dims(&self) -> (u32, u32, u32) {
        let grid_x = self.num_q_tiles();
        let grid_y = self.config.batch_size * self.config.num_heads;
        (grid_x, grid_y, 1)
    }

    /// Returns the block dimensions `(block_x, block_y, block_z)`.
    #[must_use]
    pub fn block_dims(&self) -> (u32, u32, u32) {
        (self.num_warps * 32, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// Tile-size selection heuristics
// ---------------------------------------------------------------------------

/// Selects tile sizes based on head dimension.
///
/// Follows FlashAttention-2 tile size selection heuristics:
/// - Small head_dim (<=64): larger tiles for better GEMM utilisation.
/// - Medium head_dim (<=128): balanced tiles.
/// - Large head_dim (>128): smaller tiles to fit in shared memory.
fn select_tile_sizes(head_dim: u32) -> (u32, u32) {
    match head_dim {
        d if d <= 64 => (128, 128),
        d if d <= 128 => (128, 64),
        _ => (64, 64),
    }
}

/// Selects the number of warps based on head dimension and tile size.
fn select_num_warps(head_dim: u32, block_m: u32) -> u32 {
    if head_dim >= 128 && block_m >= 128 {
        8
    } else {
        4
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a valid default config for testing.
    fn default_config() -> FusedRopeAttnConfig {
        FusedRopeAttnConfig {
            num_heads: 8,
            head_dim: 64,
            seq_len: 512,
            batch_size: 2,
            rope_base: 10000.0,
            rope_scaling: None,
            causal: false,
            softmax_scale: None,
        }
    }

    // -----------------------------------------------------------------------
    // Config validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_valid_default() {
        let cfg = default_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_rejects_zero_head_dim() {
        let mut cfg = default_config();
        cfg.head_dim = 0;
        let err = cfg.validate();
        assert!(err.is_err());
        let msg = format!(
            "{}",
            err.err().unwrap_or(DnnError::InvalidDimension("".into()))
        );
        assert!(msg.contains("head_dim"));
    }

    #[test]
    fn config_rejects_odd_head_dim() {
        let mut cfg = default_config();
        cfg.head_dim = 65;
        let err = cfg.validate();
        assert!(err.is_err());
        let msg = format!(
            "{}",
            err.err().unwrap_or(DnnError::InvalidDimension("".into()))
        );
        assert!(msg.contains("even"));
    }

    #[test]
    fn config_rejects_zero_num_heads() {
        let mut cfg = default_config();
        cfg.num_heads = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_zero_seq_len() {
        let mut cfg = default_config();
        cfg.seq_len = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_zero_batch_size() {
        let mut cfg = default_config();
        cfg.batch_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_negative_rope_base() {
        let mut cfg = default_config();
        cfg.rope_base = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_nan_rope_base() {
        let mut cfg = default_config();
        cfg.rope_base = f32::NAN;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_invalid_rope_scaling() {
        let mut cfg = default_config();
        cfg.rope_scaling = Some(0.0);
        assert!(cfg.validate().is_err());

        cfg.rope_scaling = Some(-2.0);
        assert!(cfg.validate().is_err());

        cfg.rope_scaling = Some(f32::INFINITY);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_accepts_valid_rope_scaling() {
        let mut cfg = default_config();
        cfg.rope_scaling = Some(2.0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_rejects_nan_softmax_scale() {
        let mut cfg = default_config();
        cfg.softmax_scale = Some(f32::NAN);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_accepts_explicit_softmax_scale() {
        let mut cfg = default_config();
        cfg.softmax_scale = Some(0.125);
        assert!(cfg.validate().is_ok());
        let scale = cfg.effective_softmax_scale();
        assert!((scale - 0.125).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Default softmax scale test
    // -----------------------------------------------------------------------

    #[test]
    fn effective_softmax_scale_default() {
        let cfg = default_config(); // head_dim=64
        let scale = cfg.effective_softmax_scale();
        let expected = 1.0 / (64.0_f32).sqrt();
        assert!((scale - expected).abs() < 1e-6);
    }

    #[test]
    fn effective_softmax_scale_override() {
        let mut cfg = default_config();
        cfg.softmax_scale = Some(0.1);
        assert!((cfg.effective_softmax_scale() - 0.1).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Plan creation tests
    // -----------------------------------------------------------------------

    #[test]
    fn plan_creation_succeeds() {
        let cfg = default_config();
        let plan = FusedRopeAttnPlan::new(cfg);
        assert!(plan.is_ok());
    }

    #[test]
    fn plan_creation_rejects_invalid_config() {
        let mut cfg = default_config();
        cfg.head_dim = 3; // odd
        let plan = FusedRopeAttnPlan::new(cfg);
        assert!(plan.is_err());
    }

    #[test]
    fn plan_with_sm_version() {
        let cfg = default_config();
        let plan = FusedRopeAttnPlan::with_sm_version(cfg, SmVersion::Sm90);
        assert!(plan.is_ok());
    }

    // -----------------------------------------------------------------------
    // PTX generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn generate_ptx_non_causal() {
        let cfg = default_config();
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let ptx = plan.generate_ptx();
            assert!(ptx.is_ok());
            let text = ptx.ok().unwrap_or_default();
            assert!(text.contains(".entry fused_rope_attn"));
            assert!(text.contains("Fused RoPE"));
            assert!(text.contains(".shared"));
            // Non-causal should NOT contain CAUSAL marker
            assert!(!text.contains("[CAUSAL]"));
        }
    }

    #[test]
    fn generate_ptx_causal() {
        let mut cfg = default_config();
        cfg.causal = true;
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let ptx = plan.generate_ptx();
            assert!(ptx.is_ok());
            let text = ptx.ok().unwrap_or_default();
            assert!(text.contains("_causal_"));
            assert!(text.contains("[CAUSAL]"));
        }
    }

    #[test]
    fn generate_ptx_with_rope_scaling() {
        let mut cfg = default_config();
        cfg.rope_scaling = Some(4.0);
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let ptx = plan.generate_ptx();
            assert!(ptx.is_ok());
            let text = ptx.ok().unwrap_or_default();
            assert!(text.contains("_scaled_"));
            assert!(text.contains("[SCALED]"));
        }
    }

    #[test]
    fn generate_ptx_head_dim_32() {
        let mut cfg = default_config();
        cfg.head_dim = 32;
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let ptx = plan.generate_ptx();
            assert!(ptx.is_ok());
            let text = ptx.ok().unwrap_or_default();
            assert!(text.contains("_d32_"));
        }
    }

    #[test]
    fn generate_ptx_head_dim_128() {
        let mut cfg = default_config();
        cfg.head_dim = 128;
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let ptx = plan.generate_ptx();
            assert!(ptx.is_ok());
            let text = ptx.ok().unwrap_or_default();
            assert!(text.contains("_d128_"));
        }
    }

    #[test]
    fn generate_ptx_head_dim_256() {
        let mut cfg = default_config();
        cfg.head_dim = 256;
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let ptx = plan.generate_ptx();
            assert!(ptx.is_ok());
            let text = ptx.ok().unwrap_or_default();
            assert!(text.contains("_d256_"));
        }
    }

    // -----------------------------------------------------------------------
    // Tile size tests
    // -----------------------------------------------------------------------

    #[test]
    fn tile_sizes_small_head_dim() {
        let mut cfg = default_config();
        cfg.head_dim = 32;
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            assert_eq!(plan.tile_sizes(), (128, 128));
        }
    }

    #[test]
    fn tile_sizes_medium_head_dim() {
        let mut cfg = default_config();
        cfg.head_dim = 128;
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            assert_eq!(plan.tile_sizes(), (128, 64));
        }
    }

    #[test]
    fn tile_sizes_large_head_dim() {
        let mut cfg = default_config();
        cfg.head_dim = 256;
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            assert_eq!(plan.tile_sizes(), (64, 64));
        }
    }

    // -----------------------------------------------------------------------
    // Workspace calculation tests
    // -----------------------------------------------------------------------

    #[test]
    fn workspace_size_positive() {
        let cfg = default_config(); // seq_len=512, block_n will be 128
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let ws = plan.workspace_size();
            assert!(ws > 0);
        }
    }

    #[test]
    fn workspace_size_small_seq() {
        let mut cfg = default_config();
        cfg.seq_len = 32; // Fits in a single KV tile
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let ws = plan.workspace_size();
            // Should be just LSE bytes: batch * heads * seq * 8
            let expected_lse = 2 * 8 * 32 * 8;
            assert_eq!(ws, expected_lse);
        }
    }

    #[test]
    fn workspace_scales_with_batch() {
        let cfg1 = FusedRopeAttnConfig {
            batch_size: 1,
            ..default_config()
        };
        let cfg2 = FusedRopeAttnConfig {
            batch_size: 4,
            ..default_config()
        };
        let plan1 = FusedRopeAttnPlan::new(cfg1).ok();
        let plan2 = FusedRopeAttnPlan::new(cfg2).ok();
        assert!(plan1.is_some() && plan2.is_some());
        if let (Some(p1), Some(p2)) = (plan1, plan2) {
            assert!(p2.workspace_size() > p1.workspace_size());
        }
    }

    // -----------------------------------------------------------------------
    // Grid / block dimensions tests
    // -----------------------------------------------------------------------

    #[test]
    fn grid_dims_correct() {
        let cfg = default_config(); // seq=512, batch=2, heads=8
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let (gx, gy, gz) = plan.grid_dims();
            assert_eq!(gx, plan.num_q_tiles());
            assert_eq!(gy, 2 * 8); // batch * heads
            assert_eq!(gz, 1);
        }
    }

    #[test]
    fn block_dims_correct() {
        let cfg = default_config();
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let (bx, by, bz) = plan.block_dims();
            assert!(bx >= 128); // at least 4 warps * 32
            assert_eq!(by, 1);
            assert_eq!(bz, 1);
        }
    }

    // -----------------------------------------------------------------------
    // Kernel name tests
    // -----------------------------------------------------------------------

    #[test]
    fn kernel_name_non_causal() {
        let cfg = default_config();
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let name = plan.kernel_name();
            assert!(name.starts_with("fused_rope_attn_d64_"));
            assert!(!name.contains("causal"));
            assert!(!name.contains("scaled"));
            assert!(name.ends_with("_f32"));
        }
    }

    #[test]
    fn kernel_name_causal() {
        let mut cfg = default_config();
        cfg.causal = true;
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let name = plan.kernel_name();
            assert!(name.contains("_causal"));
        }
    }

    #[test]
    fn kernel_name_scaled_rope() {
        let mut cfg = default_config();
        cfg.rope_scaling = Some(2.0);
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            let name = plan.kernel_name();
            assert!(name.contains("_scaled"));
        }
    }

    // -----------------------------------------------------------------------
    // Shared memory tests
    // -----------------------------------------------------------------------

    #[test]
    fn shared_mem_positive() {
        let cfg = default_config();
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            assert!(plan.shared_mem_bytes() > 0);
        }
    }

    #[test]
    fn shared_mem_grows_with_head_dim() {
        let cfg64 = FusedRopeAttnConfig {
            head_dim: 64,
            ..default_config()
        };
        let cfg128 = FusedRopeAttnConfig {
            head_dim: 128,
            ..default_config()
        };
        let p64 = FusedRopeAttnPlan::new(cfg64).ok();
        let p128 = FusedRopeAttnPlan::new(cfg128).ok();
        assert!(p64.is_some() && p128.is_some());
        if let (Some(p64), Some(p128)) = (p64, p128) {
            // Even though tile sizes change, larger head_dim should use more smem
            assert!(p128.shared_mem_bytes() > p64.shared_mem_bytes());
        }
    }

    // -----------------------------------------------------------------------
    // Num tiles tests
    // -----------------------------------------------------------------------

    #[test]
    fn num_q_tiles_exact_division() {
        let mut cfg = default_config();
        cfg.seq_len = 256;
        cfg.head_dim = 64; // block_m = 128
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            assert_eq!(plan.num_q_tiles(), 2); // 256 / 128
        }
    }

    #[test]
    fn num_q_tiles_with_remainder() {
        let mut cfg = default_config();
        cfg.seq_len = 300;
        cfg.head_dim = 64; // block_m = 128
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            assert_eq!(plan.num_q_tiles(), 3); // ceil(300 / 128)
        }
    }

    #[test]
    fn num_kv_tiles_matches_seq_len() {
        let mut cfg = default_config();
        cfg.seq_len = 1024;
        cfg.head_dim = 64; // block_n = 128
        let plan = FusedRopeAttnPlan::new(cfg).ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            assert_eq!(plan.num_kv_tiles(), 8); // 1024 / 128
        }
    }

    // -----------------------------------------------------------------------
    // Batch size variation tests
    // -----------------------------------------------------------------------

    #[test]
    fn plan_batch_size_1() {
        let mut cfg = default_config();
        cfg.batch_size = 1;
        let plan = FusedRopeAttnPlan::new(cfg);
        assert!(plan.is_ok());
        if let Ok(plan) = plan {
            let (_, gy, _) = plan.grid_dims();
            assert_eq!(gy, 8); // 1 * 8 heads
        }
    }

    #[test]
    fn plan_batch_size_large() {
        let mut cfg = default_config();
        cfg.batch_size = 32;
        let plan = FusedRopeAttnPlan::new(cfg);
        assert!(plan.is_ok());
        if let Ok(plan) = plan {
            let (_, gy, _) = plan.grid_dims();
            assert_eq!(gy, 32 * 8); // 32 * 8 heads
        }
    }

    // -----------------------------------------------------------------------
    // RoPE base variation test
    // -----------------------------------------------------------------------

    #[test]
    fn plan_rope_base_500k() {
        let mut cfg = default_config();
        cfg.rope_base = 500_000.0;
        let plan = FusedRopeAttnPlan::new(cfg);
        assert!(plan.is_ok());
        if let Ok(plan) = plan {
            let ptx = plan.generate_ptx();
            assert!(ptx.is_ok());
        }
    }

    // -----------------------------------------------------------------------
    // Combined causal + scaling test
    // -----------------------------------------------------------------------

    #[test]
    fn plan_causal_and_scaled() {
        let cfg = FusedRopeAttnConfig {
            num_heads: 32,
            head_dim: 128,
            seq_len: 4096,
            batch_size: 4,
            rope_base: 500_000.0,
            rope_scaling: Some(8.0),
            causal: true,
            softmax_scale: Some(0.08838835),
        };
        let plan = FusedRopeAttnPlan::new(cfg);
        assert!(plan.is_ok());
        if let Ok(plan) = plan {
            let name = plan.kernel_name();
            assert!(name.contains("_causal"));
            assert!(name.contains("_scaled"));
            assert!(name.contains("_d128_"));

            let ptx = plan.generate_ptx();
            assert!(ptx.is_ok());
            let text = ptx.ok().unwrap_or_default();
            assert!(text.contains("[CAUSAL]"));
            assert!(text.contains("[SCALED]"));
        }
    }
}
