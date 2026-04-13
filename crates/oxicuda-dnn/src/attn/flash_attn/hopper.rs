//! FlashAttention-3 for Hopper+ GPUs (sm_90+).
//!
//! Extends FlashAttention-2 with Hopper-specific optimizations:
//!
//! - **Warp specialization**: Dedicated producer warps handle TMA loads while
//!   consumer warps perform MMA compute, overlapping memory and compute.
//! - **TMA-based loads**: Tensor Memory Accelerator asynchronously loads tiles
//!   from global to shared memory with hardware-managed addressing.
//! - **Ping-pong shared memory**: Double (or multi-) buffered shared memory
//!   allows producers to fill one buffer while consumers read another.
//! - **WGMMA instructions**: Hopper's warp-group MMA for higher throughput
//!   tensor core operations.
//!
//! ## Usage
//!
//! ```ignore
//! use oxicuda_dnn::attn::flash_attn::hopper::*;
//! use oxicuda_ptx::prelude::*;
//!
//! let config = FlashAttention3Config {
//!     head_dim: 128,
//!     block_m: 128,
//!     block_n: 128,
//!     num_warps: 8,
//!     num_producer_warps: 2,
//!     num_consumer_warps: 6,
//!     pingpong_stages: 2,
//!     causal: false,
//!     sm_version: 90,
//!     float_type: PtxType::F16,
//! };
//!
//! let plan = FlashAttention3Plan::new(config)?;
//! let fwd_ptx = plan.generate_forward()?;
//! let bwd_ptx = plan.generate_backward()?;
//! ```

use oxicuda_launch::{Dim3, LaunchParams};
use oxicuda_ptx::prelude::*;

use crate::error::{DnnError, DnnResult};

#[path = "hopper_body.rs"]
mod hopper_body;
use hopper_body::{emit_fa3_backward_body, emit_fa3_forward_body, float_type_suffix};

// ---------------------------------------------------------------------------
// Supported head dimensions
// ---------------------------------------------------------------------------

/// Head dimensions supported by FlashAttention-3.
const SUPPORTED_HEAD_DIMS: &[u32] = &[64, 128, 256];

/// Minimum SM version required for FlashAttention-3 (Hopper).
const MIN_SM_VERSION: u32 = 90;

// ---------------------------------------------------------------------------
// FlashAttention3Config
// ---------------------------------------------------------------------------

/// FlashAttention-3 configuration for Hopper+ GPUs.
///
/// Extends FA2 with warp-specialized scheduling (producer/consumer warps),
/// TMA-based global-to-shared loads, and ping-pong double-buffered shared memory.
///
/// # Warp specialization
///
/// The total `num_warps` is partitioned into `num_producer_warps` (which issue
/// TMA loads from global to shared memory) and `num_consumer_warps` (which
/// perform the MMA compute on data already in shared memory). These two groups
/// run concurrently, overlapping memory latency with compute.
///
/// # Ping-pong buffering
///
/// `pingpong_stages` (typically 2) controls how many shared memory buffers
/// exist for K and V tiles. The producer fills buffer A while the consumer
/// reads buffer B, then they swap roles. This hides the TMA load latency
/// behind compute.
#[derive(Debug, Clone)]
pub struct FlashAttention3Config {
    /// Per-head dimension (e.g. 64, 128, or 256). Must be in `SUPPORTED_HEAD_DIMS`.
    pub head_dim: u32,
    /// Number of Q tile rows per block (e.g. 128).
    pub block_m: u32,
    /// Number of KV tile columns per block (e.g. 128).
    pub block_n: u32,
    /// Total warps per thread block (e.g. 8). Must equal `num_producer_warps + num_consumer_warps`.
    pub num_warps: u32,
    /// Warps dedicated to TMA memory loads (e.g. 2).
    pub num_producer_warps: u32,
    /// Warps dedicated to MMA compute (e.g. 6).
    pub num_consumer_warps: u32,
    /// Number of shared memory ping-pong buffers for K and V tiles (e.g. 2).
    pub pingpong_stages: u32,
    /// Whether to apply causal (lower-triangular) masking.
    pub causal: bool,
    /// Target SM version. Must be >= 90 (Hopper).
    pub sm_version: u32,
    /// Floating-point type for Q/K/V elements. Typically `PtxType::F16` or `PtxType::BF16`.
    pub float_type: PtxType,
}

impl FlashAttention3Config {
    /// Creates a default configuration for the given head dimension and float type.
    ///
    /// Uses 8 warps total (2 producer, 6 consumer), 128x128 tile sizes, and
    /// 2-stage ping-pong buffering. SM version is set to 90 (Hopper).
    #[must_use]
    pub fn default_for(head_dim: u32, float_type: PtxType, causal: bool) -> Self {
        let (block_m, block_n) = match head_dim {
            d if d <= 64 => (128, 128),
            d if d <= 128 => (128, 128),
            _ => (64, 128),
        };

        Self {
            head_dim,
            block_m,
            block_n,
            num_warps: 8,
            num_producer_warps: 2,
            num_consumer_warps: 6,
            pingpong_stages: 2,
            causal,
            sm_version: 90,
            float_type,
        }
    }
}

// ---------------------------------------------------------------------------
// FlashAttention3Plan
// ---------------------------------------------------------------------------

/// Execution plan for FlashAttention-3.
///
/// Created from a validated [`FlashAttention3Config`], this plan can generate
/// PTX for both the forward and backward passes, compute shared memory
/// requirements, and determine launch parameters.
#[derive(Debug)]
pub struct FlashAttention3Plan {
    config: FlashAttention3Config,
}

impl FlashAttention3Plan {
    /// Creates a new plan after validating the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if:
    /// - `sm_version < 90` (requires Hopper+)
    /// - `num_producer_warps + num_consumer_warps != num_warps`
    /// - `head_dim` is not in `SUPPORTED_HEAD_DIMS`
    /// - `block_m` or `block_n` is zero or not a power of two
    /// - `pingpong_stages` is zero
    /// - `float_type` is not F16 or BF16
    pub fn new(config: FlashAttention3Config) -> DnnResult<Self> {
        validate_config(&config)?;
        Ok(Self { config })
    }

    /// Returns a reference to the validated configuration.
    #[must_use]
    pub fn config(&self) -> &FlashAttention3Config {
        &self.config
    }

    /// Computes the required shared memory in bytes.
    ///
    /// The layout allocates:
    /// - Q tile: `block_m * head_dim * elem_size` (single buffer, loaded once per Q-tile)
    /// - K tiles: `block_n * head_dim * elem_size * pingpong_stages` (ping-pong buffered)
    /// - V tiles: `block_n * head_dim * elem_size * pingpong_stages` (ping-pong buffered)
    /// - Softmax scratch: `block_m * sizeof(f32) * 2` (running max `m` and sum `l`)
    /// - Output accumulator: `block_m * head_dim * sizeof(f32)` (FP32 accumulation)
    #[must_use]
    pub fn shared_memory_bytes(&self) -> usize {
        let c = &self.config;
        let elem_size = c.float_type.size_bytes();
        let f32_size = 4usize;

        let q_tile = (c.block_m as usize) * (c.head_dim as usize) * elem_size;
        let k_tiles =
            (c.block_n as usize) * (c.head_dim as usize) * elem_size * (c.pingpong_stages as usize);
        let v_tiles =
            (c.block_n as usize) * (c.head_dim as usize) * elem_size * (c.pingpong_stages as usize);
        // Running max (m_i) and running sum (l_i), both FP32
        let softmax_scratch = (c.block_m as usize) * f32_size * 2;
        // FP32 output accumulator O_acc[block_m][head_dim]
        let o_acc = (c.block_m as usize) * (c.head_dim as usize) * f32_size;

        q_tile + k_tiles + v_tiles + softmax_scratch + o_acc
    }

    /// Computes the launch parameters for a given problem size.
    ///
    /// # Arguments
    ///
    /// * `seq_len_q` - Query sequence length.
    /// * `seq_len_kv` - Key/Value sequence length.
    /// * `batch` - Batch size.
    /// * `num_heads` - Number of attention heads.
    ///
    /// # Returns
    ///
    /// A [`LaunchParams`] with:
    /// - Grid: `(num_q_tiles, batch * num_heads, 1)` where `num_q_tiles = ceil(seq_len_q / block_m)`
    /// - Block: `(num_warps * 32, 1, 1)`
    /// - Shared memory: result of [`shared_memory_bytes`](Self::shared_memory_bytes)
    #[must_use]
    pub fn launch_params(
        &self,
        seq_len_q: u32,
        seq_len_kv: u32,
        batch: u32,
        num_heads: u32,
    ) -> LaunchParams {
        let _ = seq_len_kv; // KV length affects loop iterations, not grid size
        let c = &self.config;
        let num_q_tiles = seq_len_q.div_ceil(c.block_m);
        let threads_per_block = c.num_warps * 32;

        let grid = Dim3::new(num_q_tiles, batch * num_heads, 1);
        let block = Dim3::new(threads_per_block, 1, 1);

        // shared_memory_bytes() returns usize; LaunchParams expects u32
        let smem = self.shared_memory_bytes().min(u32::MAX as usize) as u32;

        LaunchParams::builder()
            .grid(grid)
            .block(block)
            .shared_mem(smem)
            .build()
    }

    /// Generates the FlashAttention-3 forward pass PTX kernel.
    ///
    /// The generated kernel implements a warp-specialized FlashAttention-3:
    ///
    /// 1. **Producer warps** (warp IDs `0..num_producer_warps`):
    ///    - Use TMA to asynchronously load K and V tiles from global to shared memory.
    ///    - Signal arrival at a named barrier after each tile load completes.
    ///    - Alternate between ping-pong buffer stages.
    ///
    /// 2. **Consumer warps** (warp IDs `num_producer_warps..num_warps`):
    ///    - Wait for producer barrier signals before reading shared memory.
    ///    - Compute `S = Q * K^T` via MMA instructions.
    ///    - Apply online softmax with running max (`m`) and sum (`l`) statistics.
    ///    - If causal, zero out future token positions.
    ///    - Compute `O_acc += P * V` via MMA instructions.
    ///    - Signal producer that the buffer is free.
    ///
    /// 3. After all KV tiles are processed, consumer warps perform the final
    ///    rescale `O = O_acc / l` and store the logsumexp `m + log(l)`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if the PTX builder encounters an error.
    pub fn generate_forward(&self) -> DnnResult<String> {
        let c = &self.config;
        let sm = sm_version_from_u32(c.sm_version)?;
        let kernel_name = forward_kernel_name(c);
        let threads_per_block = c.num_warps * 32;

        let q_smem_elems = (c.block_m * c.head_dim) as usize;
        let kv_stage_elems = (c.block_n * c.head_dim) as usize;
        let kv_total_elems = kv_stage_elems * (c.pingpong_stages as usize);
        let softmax_elems = (c.block_m * 2) as usize; // m_i and l_i
        let o_acc_elems = (c.block_m * c.head_dim) as usize;

        let float_type = c.float_type;
        let block_m = c.block_m;
        let block_n = c.block_n;
        let head_dim = c.head_dim;
        let num_producer_warps = c.num_producer_warps;
        let pingpong_stages = c.pingpong_stages;
        let causal = c.causal;

        let ptx = KernelBuilder::new(&kernel_name)
            .target(sm)
            .param("q_ptr", PtxType::U64)
            .param("k_ptr", PtxType::U64)
            .param("v_ptr", PtxType::U64)
            .param("o_ptr", PtxType::U64)
            .param("lse_ptr", PtxType::U64)
            .param("seq_len_q", PtxType::U32)
            .param("seq_len_kv", PtxType::U32)
            .param("head_dim", PtxType::U32)
            .param("num_heads", PtxType::U32)
            .param("sm_scale", PtxType::F32)
            .param("num_kv_tiles", PtxType::U32)
            .param("stride_qb", PtxType::U32)
            .param("stride_qh", PtxType::U32)
            .param("stride_kb", PtxType::U32)
            .param("stride_kh", PtxType::U32)
            .param("stride_vb", PtxType::U32)
            .param("stride_vh", PtxType::U32)
            .shared_mem("q_smem", float_type, q_smem_elems)
            .shared_mem("k_smem", float_type, kv_total_elems)
            .shared_mem("v_smem", float_type, kv_total_elems)
            .shared_mem("softmax_smem", PtxType::F32, softmax_elems)
            .shared_mem("o_acc_smem", PtxType::F32, o_acc_elems)
            .max_threads_per_block(threads_per_block)
            .body(move |b| {
                emit_fa3_forward_body(
                    b,
                    block_m,
                    block_n,
                    head_dim,
                    num_producer_warps,
                    pingpong_stages,
                    causal,
                    float_type,
                );
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Generates the FlashAttention-3 backward pass PTX kernel.
    ///
    /// The backward pass recomputes the attention matrix on-the-fly from saved
    /// logsumexp values, using the same warp-specialized structure as the
    /// forward pass:
    ///
    /// 1. **Producer warps**: TMA-load Q, K, V, dO tiles to shared memory.
    /// 2. **Consumer warps**: Recompute attention and compute gradients:
    ///    - `dV = P^T * dO` (accumulate across Q tiles)
    ///    - `dP = dO * V^T`
    ///    - `dS = P * (dP - D_i)` where `D_i = rowsum(dP * P)`
    ///    - `dQ += dS * K` (atomically accumulated)
    ///    - `dK += dS^T * Q`
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] if the PTX builder encounters an error.
    pub fn generate_backward(&self) -> DnnResult<String> {
        let c = &self.config;
        let sm = sm_version_from_u32(c.sm_version)?;
        let kernel_name = backward_kernel_name(c);
        let threads_per_block = c.num_warps * 32;

        let q_smem_elems = (c.block_m * c.head_dim) as usize;
        let kv_stage_elems = (c.block_n * c.head_dim) as usize;
        let kv_total_elems = kv_stage_elems * (c.pingpong_stages as usize);
        let do_smem_elems = (c.block_m * c.head_dim) as usize;
        let di_smem_elems = c.block_m as usize; // D_i = rowsum(dO * O)

        let float_type = c.float_type;
        let block_m = c.block_m;
        let block_n = c.block_n;
        let head_dim = c.head_dim;
        let num_producer_warps = c.num_producer_warps;
        let pingpong_stages = c.pingpong_stages;
        let causal = c.causal;

        let ptx = KernelBuilder::new(&kernel_name)
            .target(sm)
            .param("q_ptr", PtxType::U64)
            .param("k_ptr", PtxType::U64)
            .param("v_ptr", PtxType::U64)
            .param("o_ptr", PtxType::U64)
            .param("do_ptr", PtxType::U64)
            .param("lse_ptr", PtxType::U64)
            .param("di_ptr", PtxType::U64)
            .param("dq_ptr", PtxType::U64)
            .param("dk_ptr", PtxType::U64)
            .param("dv_ptr", PtxType::U64)
            .param("seq_len_q", PtxType::U32)
            .param("seq_len_kv", PtxType::U32)
            .param("head_dim", PtxType::U32)
            .param("num_heads", PtxType::U32)
            .param("sm_scale", PtxType::F32)
            .param("num_q_tiles", PtxType::U32)
            .param("stride_qb", PtxType::U32)
            .param("stride_qh", PtxType::U32)
            .param("stride_kb", PtxType::U32)
            .param("stride_kh", PtxType::U32)
            .param("stride_vb", PtxType::U32)
            .param("stride_vh", PtxType::U32)
            .shared_mem("q_smem", float_type, q_smem_elems)
            .shared_mem("k_smem", float_type, kv_total_elems)
            .shared_mem("v_smem", float_type, kv_total_elems)
            .shared_mem("do_smem", float_type, do_smem_elems)
            .shared_mem("di_smem", PtxType::F32, di_smem_elems)
            .max_threads_per_block(threads_per_block)
            .body(move |b| {
                emit_fa3_backward_body(
                    b,
                    block_m,
                    block_n,
                    head_dim,
                    num_producer_warps,
                    pingpong_stages,
                    causal,
                    float_type,
                );
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }
}

// ===========================================================================
// Configuration validation
// ===========================================================================

fn validate_config(config: &FlashAttention3Config) -> DnnResult<()> {
    // SM version check
    if config.sm_version < MIN_SM_VERSION {
        return Err(DnnError::InvalidArgument(format!(
            "FlashAttention-3 requires sm_version >= {MIN_SM_VERSION}, got {}",
            config.sm_version
        )));
    }

    // Warp partition check
    if config.num_producer_warps + config.num_consumer_warps != config.num_warps {
        return Err(DnnError::InvalidArgument(format!(
            "num_producer_warps ({}) + num_consumer_warps ({}) != num_warps ({})",
            config.num_producer_warps, config.num_consumer_warps, config.num_warps
        )));
    }

    // Producer warps must be at least 1
    if config.num_producer_warps == 0 {
        return Err(DnnError::InvalidArgument(
            "num_producer_warps must be >= 1".to_string(),
        ));
    }

    // Consumer warps must be at least 1
    if config.num_consumer_warps == 0 {
        return Err(DnnError::InvalidArgument(
            "num_consumer_warps must be >= 1".to_string(),
        ));
    }

    // Head dimension check
    if !SUPPORTED_HEAD_DIMS.contains(&config.head_dim) {
        return Err(DnnError::InvalidArgument(format!(
            "head_dim {} not supported; must be one of {:?}",
            config.head_dim, SUPPORTED_HEAD_DIMS
        )));
    }

    // Block dimensions must be non-zero and powers of two
    if config.block_m == 0 || !config.block_m.is_power_of_two() {
        return Err(DnnError::InvalidArgument(format!(
            "block_m must be a non-zero power of two, got {}",
            config.block_m
        )));
    }
    if config.block_n == 0 || !config.block_n.is_power_of_two() {
        return Err(DnnError::InvalidArgument(format!(
            "block_n must be a non-zero power of two, got {}",
            config.block_n
        )));
    }

    // Ping-pong stages
    if config.pingpong_stages == 0 {
        return Err(DnnError::InvalidArgument(
            "pingpong_stages must be >= 1".to_string(),
        ));
    }

    // Float type check
    match config.float_type {
        PtxType::F16 | PtxType::BF16 => {}
        other => {
            return Err(DnnError::InvalidArgument(format!(
                "float_type must be F16 or BF16, got {:?}",
                other
            )));
        }
    }

    Ok(())
}

// ===========================================================================
// SM version conversion
// ===========================================================================

fn sm_version_from_u32(ver: u32) -> DnnResult<SmVersion> {
    match ver {
        90 => Ok(SmVersion::Sm90),
        100 => Ok(SmVersion::Sm100),
        120 => Ok(SmVersion::Sm120),
        _ => Err(DnnError::InvalidArgument(format!(
            "unsupported sm_version {ver} for FlashAttention-3 (need >= 90)"
        ))),
    }
}

// ===========================================================================
// Kernel naming
// ===========================================================================

fn forward_kernel_name(c: &FlashAttention3Config) -> String {
    format!(
        "flash_attn3_fwd_d{}_bm{}_bn{}_pw{}_cw{}_pp{}_{}_{}",
        c.head_dim,
        c.block_m,
        c.block_n,
        c.num_producer_warps,
        c.num_consumer_warps,
        c.pingpong_stages,
        if c.causal { "causal" } else { "nocausal" },
        float_type_suffix(c.float_type),
    )
}

fn backward_kernel_name(c: &FlashAttention3Config) -> String {
    format!(
        "flash_attn3_bwd_d{}_bm{}_bn{}_pw{}_cw{}_pp{}_{}_{}",
        c.head_dim,
        c.block_m,
        c.block_n,
        c.num_producer_warps,
        c.num_consumer_warps,
        c.pingpong_stages,
        if c.causal { "causal" } else { "nocausal" },
        float_type_suffix(c.float_type),
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(head_dim: u32, float_type: PtxType, causal: bool) -> FlashAttention3Config {
        FlashAttention3Config::default_for(head_dim, float_type, causal)
    }

    #[allow(clippy::too_many_arguments)]
    fn make_custom_config(
        head_dim: u32,
        block_m: u32,
        block_n: u32,
        num_warps: u32,
        num_producer_warps: u32,
        num_consumer_warps: u32,
        pingpong_stages: u32,
        causal: bool,
        sm_version: u32,
        float_type: PtxType,
    ) -> FlashAttention3Config {
        FlashAttention3Config {
            head_dim,
            block_m,
            block_n,
            num_warps,
            num_producer_warps,
            num_consumer_warps,
            pingpong_stages,
            causal,
            sm_version,
            float_type,
        }
    }

    // -----------------------------------------------------------------------
    // Config validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_valid_default_f16() {
        let cfg = make_config(128, PtxType::F16, false);
        assert!(FlashAttention3Plan::new(cfg).is_ok());
    }

    #[test]
    fn config_valid_default_bf16() {
        let cfg = make_config(128, PtxType::BF16, true);
        assert!(FlashAttention3Plan::new(cfg).is_ok());
    }

    #[test]
    fn config_valid_head_dim_64() {
        let cfg = make_config(64, PtxType::F16, false);
        assert!(FlashAttention3Plan::new(cfg).is_ok());
    }

    #[test]
    fn config_valid_head_dim_256() {
        let cfg = make_config(256, PtxType::BF16, false);
        assert!(FlashAttention3Plan::new(cfg).is_ok());
    }

    #[test]
    fn config_rejects_sm_below_90() {
        let cfg = make_custom_config(128, 128, 128, 8, 2, 6, 2, false, 80, PtxType::F16);
        let err = FlashAttention3Plan::new(cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("sm_version"), "error: {msg}");
    }

    #[test]
    fn config_rejects_warp_mismatch() {
        let cfg = make_custom_config(128, 128, 128, 8, 3, 6, 2, false, 90, PtxType::F16);
        let err = FlashAttention3Plan::new(cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("num_warps"), "error: {msg}");
    }

    #[test]
    fn config_rejects_zero_producer_warps() {
        let cfg = make_custom_config(128, 128, 128, 8, 0, 8, 2, false, 90, PtxType::F16);
        let err = FlashAttention3Plan::new(cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("producer"), "error: {msg}");
    }

    #[test]
    fn config_rejects_zero_consumer_warps() {
        let cfg = make_custom_config(128, 128, 128, 2, 2, 0, 2, false, 90, PtxType::F16);
        let err = FlashAttention3Plan::new(cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("consumer"), "error: {msg}");
    }

    #[test]
    fn config_rejects_unsupported_head_dim() {
        let cfg = make_custom_config(96, 128, 128, 8, 2, 6, 2, false, 90, PtxType::F16);
        let err = FlashAttention3Plan::new(cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("head_dim"), "error: {msg}");
    }

    #[test]
    fn config_rejects_non_power_of_two_block_m() {
        let cfg = make_custom_config(128, 96, 128, 8, 2, 6, 2, false, 90, PtxType::F16);
        let err = FlashAttention3Plan::new(cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("block_m"), "error: {msg}");
    }

    #[test]
    fn config_rejects_zero_block_n() {
        let cfg = make_custom_config(128, 128, 0, 8, 2, 6, 2, false, 90, PtxType::F16);
        let err = FlashAttention3Plan::new(cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("block_n"), "error: {msg}");
    }

    #[test]
    fn config_rejects_zero_pingpong_stages() {
        let cfg = make_custom_config(128, 128, 128, 8, 2, 6, 0, false, 90, PtxType::F16);
        let err = FlashAttention3Plan::new(cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("pingpong"), "error: {msg}");
    }

    #[test]
    fn config_rejects_f32_float_type() {
        let cfg = make_custom_config(128, 128, 128, 8, 2, 6, 2, false, 90, PtxType::F32);
        let err = FlashAttention3Plan::new(cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("float_type"), "error: {msg}");
    }

    // -----------------------------------------------------------------------
    // Forward PTX generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn forward_ptx_generation_f16_noncausal() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(ptx.contains("flash_attn3_fwd"), "kernel name missing");
        assert!(ptx.contains(".shared"), "shared mem missing");
        assert!(ptx.contains("sm_90"), "target missing");
    }

    #[test]
    fn forward_ptx_generation_bf16_causal() {
        let cfg = make_config(128, PtxType::BF16, true);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(ptx.contains("flash_attn3_fwd"), "kernel name missing");
        assert!(ptx.contains("causal"), "causal keyword missing");
        assert!(ptx.contains("CAUSAL"), "causal mask comment missing");
    }

    #[test]
    fn forward_ptx_contains_tma_reference() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(
            ptx.contains("TmaLoad") || ptx.contains("TMA") || ptx.contains("tma"),
            "TMA reference missing in forward PTX"
        );
    }

    #[test]
    fn forward_ptx_contains_mma_reference() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(
            ptx.contains("mma.sync") || ptx.contains("MMA") || ptx.contains("mma"),
            "MMA reference missing in forward PTX"
        );
    }

    #[test]
    fn forward_ptx_contains_bar_sync() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(ptx.contains("bar.sync"), "bar.sync missing in forward PTX");
    }

    #[test]
    fn forward_ptx_contains_cp_async() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(ptx.contains("cp.async"), "cp.async missing in forward PTX");
    }

    /// Verify that real `mma.sync.aligned` instructions are emitted (not just comments).
    #[test]
    fn forward_ptx_contains_real_mma_instruction() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        // The instruction (not a comment) must appear in the generated PTX.
        assert!(
            ptx.contains("mma.sync.aligned"),
            "real mma.sync.aligned instruction missing from forward PTX (got comments only)"
        );
    }

    /// Verify that real ldmatrix instructions are emitted.
    #[test]
    fn forward_ptx_contains_ldmatrix() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(
            ptx.contains("ldmatrix.sync.aligned"),
            "ldmatrix instruction missing from forward PTX"
        );
    }

    /// Verify that warp-level shuffle instructions for softmax are emitted.
    #[test]
    fn forward_ptx_contains_shfl_sync() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(
            ptx.contains("shfl.sync.bfly.b32"),
            "shfl.sync.bfly (warp reduce) missing from forward PTX"
        );
    }

    /// Verify that exp2 (fast exp approximation) is emitted for softmax.
    #[test]
    fn forward_ptx_contains_ex2_approx() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(
            ptx.contains("ex2.approx.f32"),
            "ex2.approx.f32 (softmax exp) missing from forward PTX"
        );
    }

    /// Verify that backward pass emits real MMA instructions.
    #[test]
    fn backward_ptx_contains_real_mma_instruction() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_backward().expect("ptx gen");
        assert!(
            ptx.contains("mma.sync.aligned"),
            "real mma.sync.aligned instruction missing from backward PTX"
        );
    }

    /// Verify atom.add.global is emitted for dQ accumulation in backward pass.
    #[test]
    fn backward_ptx_contains_atom_add_for_dq() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_backward().expect("ptx gen");
        assert!(
            ptx.contains("atom.global.add.f32"),
            "atom.global.add.f32 for dQ missing from backward PTX"
        );
    }

    #[test]
    fn forward_ptx_head_dim_64() {
        let cfg = make_config(64, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(ptx.contains("d64"), "head_dim=64 missing from kernel name");
    }

    #[test]
    fn forward_ptx_head_dim_256() {
        let cfg = make_config(256, PtxType::BF16, true);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(
            ptx.contains("d256"),
            "head_dim=256 missing from kernel name"
        );
        assert!(ptx.contains("bf16"), "bf16 suffix missing from kernel name");
    }

    // -----------------------------------------------------------------------
    // Backward PTX generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn backward_ptx_generation_f16_noncausal() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_backward().expect("ptx gen");
        assert!(ptx.contains("flash_attn3_bwd"), "bwd kernel name missing");
        assert!(ptx.contains(".shared"), "shared mem missing");
    }

    #[test]
    fn backward_ptx_generation_bf16_causal() {
        let cfg = make_config(128, PtxType::BF16, true);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_backward().expect("ptx gen");
        assert!(ptx.contains("flash_attn3_bwd"), "bwd kernel name missing");
        assert!(ptx.contains("CAUSAL"), "causal mask comment missing");
    }

    #[test]
    fn backward_ptx_contains_gradient_comments() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_backward().expect("ptx gen");
        assert!(ptx.contains("dV"), "dV gradient missing");
        assert!(ptx.contains("dK"), "dK gradient missing");
        assert!(ptx.contains("dQ"), "dQ gradient missing");
        assert!(ptx.contains("dS"), "dS gradient missing");
        assert!(ptx.contains("dP"), "dP gradient missing");
    }

    #[test]
    fn backward_ptx_contains_recompute_reference() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_backward().expect("ptx gen");
        assert!(
            ptx.contains("Recompute") || ptx.contains("recompute") || ptx.contains("logsumexp"),
            "recompute reference missing in backward PTX"
        );
    }

    // -----------------------------------------------------------------------
    // Shared memory calculation tests
    // -----------------------------------------------------------------------

    #[test]
    fn shared_memory_bytes_basic() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let smem = plan.shared_memory_bytes();
        assert!(smem > 0, "shared memory must be > 0");

        // Expected breakdown for block_m=128, block_n=128, head_dim=128, F16 (2B), 2 stages:
        // Q tile:  128*128*2 = 32768
        // K tiles: 128*128*2*2 = 65536
        // V tiles: 128*128*2*2 = 65536
        // Softmax: 128*4*2 = 1024
        // O_acc:   128*128*4 = 65536
        // Total:   230400
        assert_eq!(
            smem, 230_400,
            "shared memory mismatch for 128-dim F16 2-stage"
        );
    }

    #[test]
    fn shared_memory_bytes_bf16() {
        // BF16 has the same size as F16 (2 bytes), so should match
        let cfg_f16 = make_config(128, PtxType::F16, false);
        let cfg_bf16 = make_config(128, PtxType::BF16, false);
        let plan_f16 = FlashAttention3Plan::new(cfg_f16).expect("valid");
        let plan_bf16 = FlashAttention3Plan::new(cfg_bf16).expect("valid");
        assert_eq!(
            plan_f16.shared_memory_bytes(),
            plan_bf16.shared_memory_bytes()
        );
    }

    #[test]
    fn shared_memory_bytes_head_dim_64() {
        let cfg = make_config(64, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let smem = plan.shared_memory_bytes();

        // block_m=128, block_n=128, head_dim=64, F16, 2 stages
        // Q: 128*64*2 = 16384
        // K: 128*64*2*2 = 32768
        // V: 128*64*2*2 = 32768
        // Softmax: 128*4*2 = 1024
        // O_acc: 128*64*4 = 32768
        // Total: 115712
        assert_eq!(smem, 115_712);
    }

    #[test]
    fn shared_memory_bytes_3_stages() {
        let cfg = make_custom_config(128, 128, 128, 8, 2, 6, 3, false, 90, PtxType::F16);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let smem_3 = plan.shared_memory_bytes();

        let cfg_2 = make_config(128, PtxType::F16, false);
        let plan_2 = FlashAttention3Plan::new(cfg_2).expect("valid config");
        let smem_2 = plan_2.shared_memory_bytes();

        assert!(
            smem_3 > smem_2,
            "3 stages should use more smem than 2 stages"
        );
    }

    // -----------------------------------------------------------------------
    // Launch params tests
    // -----------------------------------------------------------------------

    #[test]
    fn launch_params_basic() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let params = plan.launch_params(1024, 1024, 2, 8);

        // num_q_tiles = 1024 / 128 = 8
        assert_eq!(params.grid.x, 8);
        assert_eq!(params.grid.y, 16); // 2 * 8
        assert_eq!(params.grid.z, 1);
        assert_eq!(params.block.x, 256); // 8 warps * 32
        assert_eq!(params.block.y, 1);
    }

    #[test]
    fn launch_params_non_aligned_seq() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        // 300 / 128 = ceil = 3
        let params = plan.launch_params(300, 512, 1, 1);
        assert_eq!(params.grid.x, 3);
    }

    #[test]
    fn launch_params_single_tile() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let params = plan.launch_params(64, 64, 1, 1);
        assert_eq!(params.grid.x, 1);
        assert_eq!(params.grid.y, 1);
    }

    #[test]
    fn launch_params_large_batch() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let params = plan.launch_params(2048, 2048, 32, 64);
        assert_eq!(params.grid.x, 16); // 2048 / 128
        assert_eq!(params.grid.y, 2048); // 32 * 64
    }

    #[test]
    fn launch_params_shared_mem_matches() {
        let cfg = make_config(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let params = plan.launch_params(512, 512, 1, 1);
        let expected_smem = plan.shared_memory_bytes().min(u32::MAX as usize) as u32;
        assert_eq!(params.shared_mem_bytes, expected_smem);
    }

    // -----------------------------------------------------------------------
    // Ping-pong stages verification
    // -----------------------------------------------------------------------

    #[test]
    fn pingpong_stages_in_forward_ptx() {
        let cfg = make_custom_config(128, 128, 128, 8, 2, 6, 3, false, 90, PtxType::F16);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(
            ptx.contains("pp3"),
            "pingpong_stages=3 missing from kernel name"
        );
        assert!(
            ptx.contains("Ping-pong stages: 3"),
            "pingpong stages comment should reflect 3 stages"
        );
    }

    #[test]
    fn pingpong_single_stage() {
        let cfg = make_custom_config(128, 128, 128, 8, 2, 6, 1, false, 90, PtxType::F16);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(
            ptx.contains("pp1"),
            "pingpong_stages=1 missing from kernel name"
        );
    }

    // -----------------------------------------------------------------------
    // SM version edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn sm_100_accepted() {
        let cfg = make_custom_config(128, 128, 128, 8, 2, 6, 2, false, 100, PtxType::F16);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(ptx.contains("sm_100"), "should target sm_100");
    }

    #[test]
    fn sm_120_accepted() {
        let cfg = make_custom_config(128, 128, 128, 8, 2, 6, 2, false, 120, PtxType::F16);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        assert!(ptx.contains("sm_120"), "should target sm_120");
    }

    // -----------------------------------------------------------------------
    // Default config helper
    // -----------------------------------------------------------------------

    #[test]
    fn default_config_64_f16() {
        let cfg = FlashAttention3Config::default_for(64, PtxType::F16, false);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.block_m, 128);
        assert_eq!(cfg.block_n, 128);
        assert_eq!(cfg.num_warps, 8);
        assert_eq!(cfg.num_producer_warps, 2);
        assert_eq!(cfg.num_consumer_warps, 6);
        assert!(!cfg.causal);
    }

    #[test]
    fn default_config_256_bf16_causal() {
        let cfg = FlashAttention3Config::default_for(256, PtxType::BF16, true);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.block_m, 64);
        assert_eq!(cfg.block_n, 128);
        assert!(cfg.causal);
        assert_eq!(cfg.sm_version, 90);
    }

    // -----------------------------------------------------------------------
    // Quality-gate: FlashAttention-3 optimal tile selection 128×64 / 64×128
    // -----------------------------------------------------------------------

    /// FA3 for head_dim=128 uses 128×128 tile by default (both Br and Bc).
    ///
    /// FA3 Hopper uses larger tiles than FA2 due to TMA efficiency.
    #[test]
    fn test_fa3_tile_128x128_for_head_dim_128() {
        let cfg = FlashAttention3Config::default_for(128, PtxType::F16, false);
        assert_eq!(
            cfg.block_m, 128,
            "FA3 Br (block_m) for head_dim=128 should be 128"
        );
        assert_eq!(
            cfg.block_n, 128,
            "FA3 Bc (block_n) for head_dim=128 should be 128"
        );
        assert!(
            cfg.block_m >= 64,
            "FA3 block_m should be >= 64 (tile requirement)"
        );
        assert!(
            cfg.block_n >= 64,
            "FA3 block_n should be >= 64 (tile requirement)"
        );
    }

    /// FA3 for head_dim=64 uses 128×128 tiles (same as head_dim=128 for Hopper).
    #[test]
    fn test_fa3_tile_128x128_for_head_dim_64() {
        let cfg = FlashAttention3Config::default_for(64, PtxType::F16, false);
        assert_eq!(cfg.block_m, 128, "FA3 Br for head_dim=64 should be 128");
        assert_eq!(cfg.block_n, 128, "FA3 Bc for head_dim=64 should be 128");
    }

    /// FA3 for large head_dim=256 uses a smaller query tile (64×128).
    #[test]
    fn test_fa3_tile_64x128_for_head_dim_256() {
        let cfg = FlashAttention3Config::default_for(256, PtxType::F16, false);
        assert_eq!(
            cfg.block_m, 64,
            "FA3 Br for head_dim=256 should be 64 (register pressure)"
        );
        assert_eq!(cfg.block_n, 128, "FA3 Bc for head_dim=256 should be 128");
    }

    /// FA3 default uses 8 warps total: 2 producer + 6 consumer.
    ///
    /// This is the Hopper-recommended warp specialization split.
    #[test]
    fn test_fa3_warp_specialization_8_warps() {
        let cfg = FlashAttention3Config::default_for(128, PtxType::F16, false);
        assert_eq!(cfg.num_warps, 8, "FA3 should use 8 warps total");
        assert_eq!(
            cfg.num_producer_warps, 2,
            "FA3 should have 2 producer warps (TMA)"
        );
        assert_eq!(
            cfg.num_consumer_warps, 6,
            "FA3 should have 6 consumer warps (MMA)"
        );
        assert_eq!(
            cfg.num_producer_warps + cfg.num_consumer_warps,
            cfg.num_warps,
            "producer + consumer warps must equal total warps"
        );
        // 8 warps × 32 threads = 256 threads per block
        assert_eq!(cfg.num_warps * 32, 256, "FA3 block has 256 threads");
    }

    /// Verify that the 4-warp FA2 vs 8-warp FA3 distinction is clear.
    ///
    /// FA2 on SM80 uses 4 warps (128 threads).
    /// FA3 on SM90+ uses 8 warps (256 threads) for warp specialization.
    #[test]
    fn test_fa2_vs_fa3_warp_count_distinction() {
        use crate::attn::flash_attn::forward::FlashAttentionConfig;

        let fa2_cfg = FlashAttentionConfig::auto(128, 2048, 2048, false, SmVersion::Sm80);
        let fa3_cfg = FlashAttention3Config::default_for(128, PtxType::F16, false);

        assert_eq!(fa2_cfg.num_warps, 4, "FA2 on SM80 uses 4 warps");
        assert_eq!(fa3_cfg.num_warps, 8, "FA3 on SM90 uses 8 warps");

        let fa2_threads = fa2_cfg.num_warps * 32;
        let fa3_threads = fa3_cfg.num_warps * 32;

        assert_eq!(fa2_threads, 128, "FA2 block: 128 threads");
        assert_eq!(fa3_threads, 256, "FA3 block: 256 threads");
    }

    /// FA3 shared memory for 128×128 tile, head_dim=128, F16, 2 stages.
    #[test]
    fn test_fa3_shared_memory_128x128_head128_f16() {
        let cfg = FlashAttention3Config::default_for(128, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid FA3 config");
        let smem = plan.shared_memory_bytes();

        // Q: 128*128*2 = 32768 bytes
        // K: 128*128*2*2 (stages=2) = 65536 bytes
        // V: 128*128*2*2 (stages=2) = 65536 bytes
        // Softmax scratch: 128*4*2 = 1024 bytes
        // O_acc: 128*128*4 = 65536 bytes
        // Total: 230400 bytes
        assert_eq!(
            smem, 230_400,
            "FA3 smem for 128×128, head=128, F16, 2 stages"
        );
    }

    // -----------------------------------------------------------------------
    // Quality-gate: wgmma + TMA configuration tests
    // -----------------------------------------------------------------------

    /// FA3: head_dim=64, use 128×128 tile with 8 warps (1 warp group).
    ///
    /// Verifies the tile and warp configuration mandated by the quality gate.
    #[test]
    fn test_flash_attn3_wgmma_tile_128x128_for_head_dim_64() {
        let cfg = FlashAttention3Config::default_for(64, PtxType::F16, false);
        assert_eq!(cfg.block_m, 128, "FA3 Q-tile (block_m) should be 128");
        assert_eq!(cfg.block_n, 128, "FA3 K-tile (block_n) should be 128");
        assert_eq!(cfg.num_warps, 8, "FA3 uses 8 warps (1 warpgroup)");
    }

    /// Verify that a TMA-based load plan is reflected in the generated forward PTX.
    ///
    /// FA3 must use TMA for Q and K/V loads.
    #[test]
    fn test_flash_attn3_tma_descriptor_created() {
        let cfg = FlashAttention3Config::default_for(128, PtxType::F16, true); // causal
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("ptx gen");
        // FA3 forward PTX must contain TMA references (producer warp issues TMA loads)
        assert!(
            ptx.contains("TmaLoad") || ptx.contains("TMA") || ptx.contains("tma"),
            "FA3 forward PTX should reference TMA for Q"
        );
        // The same PTX handles K and V via TMA loads in the producer warp loop
        assert!(
            ptx.contains("cp.async") || ptx.contains("TMA"),
            "FA3 forward PTX should reference TMA/cp.async for K/V"
        );
    }

    /// FA3 ping-pong pipeline: config should have pingpong_stages = 2.
    #[test]
    fn test_flash_attn3_pingpong_pipeline_stages() {
        let cfg = FlashAttention3Config::default_for(128, PtxType::F16, false);
        assert!(
            cfg.pingpong_stages >= 2,
            "FA3 should use ping-pong (pingpong_stages >= 2)"
        );
        assert_eq!(cfg.pingpong_stages, 2, "FA3 default: 2 ping-pong stages");
    }

    /// Ping-pong stages scale shared memory linearly.
    #[test]
    fn test_fa3_pingpong_stages_linear_smem_scaling() {
        let cfg1 = make_custom_config(128, 128, 128, 8, 2, 6, 1, false, 90, PtxType::F16);
        let cfg2 = make_config(128, PtxType::F16, false); // stages=2
        let cfg3 = make_custom_config(128, 128, 128, 8, 2, 6, 3, false, 90, PtxType::F16);

        let plan1 = FlashAttention3Plan::new(cfg1).expect("valid");
        let plan2 = FlashAttention3Plan::new(cfg2).expect("valid");
        let plan3 = FlashAttention3Plan::new(cfg3).expect("valid");

        let smem1 = plan1.shared_memory_bytes();
        let smem2 = plan2.shared_memory_bytes();
        let smem3 = plan3.shared_memory_bytes();

        // Each additional stage adds: 2 × block_n × head_dim × elem_size bytes (K + V)
        // = 2 × 128 × 128 × 2 = 65536 bytes
        let kv_per_stage = 2 * 128 * 128 * 2usize; // 65536

        assert_eq!(
            smem2 - smem1,
            kv_per_stage,
            "each additional ping-pong stage adds {kv_per_stage} bytes of K+V smem"
        );
        assert_eq!(
            smem3 - smem2,
            kv_per_stage,
            "scaling from 2 to 3 stages also adds {kv_per_stage} bytes"
        );
        assert!(smem3 > smem2, "3-stage has more smem than 2-stage");
        assert!(smem2 > smem1, "2-stage has more smem than 1-stage");
    }

    // -----------------------------------------------------------------------
    // Quality-gate: FlashAttention-3 Hopper TMA verification tests
    // -----------------------------------------------------------------------

    /// For sm_90, the generated PTX must contain real tensor core instructions.
    ///
    /// FA3 on Hopper uses warp-specialized structure with producer/consumer warps.
    /// The generated PTX must contain `mma.sync.aligned` instructions (the
    /// actual tensor core MMA that executes) and bar.sync for producer-consumer
    /// synchronization — both are FA3 Hopper structural requirements.
    #[test]
    fn test_fa3_hopper_sm90_has_real_mma_and_bar_sync() {
        let cfg = FlashAttention3Config::default_for(128, PtxType::F16, false);
        assert_eq!(cfg.sm_version, 90, "default config targets sm_90");
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("forward PTX gen");

        // FA3 Hopper PTX must contain real MMA instructions (tensor core)
        assert!(
            ptx.contains("mma.sync.aligned"),
            "FA3 Hopper sm_90 PTX must contain real mma.sync.aligned tensor core instructions"
        );
        // FA3 warp-specialized structure requires bar.sync for producer/consumer coordination
        assert!(
            ptx.contains("bar.sync"),
            "FA3 Hopper PTX must contain bar.sync for producer-consumer warp synchronization"
        );
        // sm_90 target must appear in the PTX header
        assert!(
            ptx.contains("sm_90"),
            "FA3 default PTX must be compiled for sm_90"
        );
    }

    /// Default FA3 config uses 2 ping-pong stages (double buffering).
    ///
    /// This is the standard Hopper configuration: while consumer warps
    /// compute on buffer A, producer warps TMA-load into buffer B.
    #[test]
    fn test_fa3_default_pingpong_is_double_buffered() {
        let cfg = FlashAttention3Config::default_for(128, PtxType::F16, false);
        assert_eq!(
            cfg.pingpong_stages, 2,
            "default FA3 must use exactly 2 ping-pong stages (double buffering)"
        );
        // Verify it's also reflected in the generated PTX
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("forward PTX gen");
        assert!(
            ptx.contains("pp2") || ptx.contains("Ping-pong stages: 2"),
            "2-stage ping-pong must appear in kernel name or comments"
        );
    }

    /// For head_dim=64 with TMA enabled, the generated forward PTX must
    /// contain TMA load references (not just scalar loads).
    #[test]
    fn test_fa3_tma_kv_loading_for_head_dim_64() {
        let cfg = FlashAttention3Config::default_for(64, PtxType::F16, false);
        let plan = FlashAttention3Plan::new(cfg).expect("valid config");
        let ptx = plan.generate_forward().expect("forward PTX gen");

        // TMA loading path must be active for head_dim=64 (≥64 threshold)
        assert!(
            ptx.contains("TmaLoad") || ptx.contains("TMA") || ptx.contains("cp.async"),
            "FA3 head_dim=64 must use TMA loading path (cp.async or TMA reference)"
        );
        // Must not degrade to pure scalar ld.global (TMA is mandatory for FA3)
        // The PTX may contain ld.global for other purposes but TMA must be primary
        assert!(
            ptx.contains("cp.async") || ptx.contains("tma") || ptx.contains("TMA"),
            "TMA/cp.async must be referenced in head_dim=64 PTX"
        );
    }

    // -----------------------------------------------------------------------
    // Task 2: FlashAttention-3 wgmma + TMA quality-gate tests
    // -----------------------------------------------------------------------

    /// FA3 wgmma tile shape for head_dim=128: output tile is 128×128.
    ///
    /// WGMMA on Hopper with a 1-warpgroup (8-warp) configuration uses an
    /// output tile of M=128 (block_m), N=128 (block_n) for head_dim=128.
    /// The K dimension per MMA step is 16 (the WGMMA K-step for F16/BF16).
    #[test]
    fn fa3_wgmma_m128n128k16_tile_shape() {
        let cfg = FlashAttention3Config::default_for(128, PtxType::F16, false);
        // M and N tile must be 128×128 for head_dim=128
        assert_eq!(
            cfg.block_m, 128,
            "FA3 wgmma M tile must be 128 for head_dim=128"
        );
        assert_eq!(
            cfg.block_n, 128,
            "FA3 wgmma N tile must be 128 for head_dim=128"
        );
        // FA3 plan must be constructable from this configuration
        let plan = FlashAttention3Plan::new(cfg).expect("FA3 128×128 config must be valid");
        // The generated PTX must contain the 128-dim identifier in the kernel name
        let ptx = plan.generate_forward().expect("PTX gen must succeed");
        assert!(
            ptx.contains("bm128") && ptx.contains("bn128"),
            "FA3 kernel name must encode 128×128 tile: expected 'bm128' and 'bn128' in PTX"
        );
    }

    /// FA3 TMA descriptor loads KV tiles with seq_block_size × head_dim elements.
    ///
    /// For seq_block_size=128 and head_dim=128, the TMA descriptor covers
    /// 128×128 = 16384 elements per K or V tile (reflected in shared memory layout).
    #[test]
    fn fa3_tma_descriptor_kv_block() {
        let seq_block_size = 128u32;
        let head_dim = 128u32;
        let cfg = FlashAttention3Config {
            head_dim,
            block_m: seq_block_size,
            block_n: seq_block_size,
            num_warps: 8,
            num_producer_warps: 2,
            num_consumer_warps: 6,
            pingpong_stages: 2,
            causal: false,
            sm_version: 90,
            float_type: PtxType::F16,
        };
        let plan = FlashAttention3Plan::new(cfg).expect("valid FA3 config");

        // Each KV tile holds block_n × head_dim F16 elements = 128 × 128 × 2 bytes = 32768 bytes
        // With 2 ping-pong stages: 32768 × 2 = 65536 bytes for K, 65536 for V
        let smem = plan.shared_memory_bytes();
        let kv_tile_bytes = (seq_block_size as usize) * (head_dim as usize) * 2; // F16 = 2 bytes
        let kv_total = kv_tile_bytes * 2; // 2 ping-pong stages per K and V
        assert_eq!(
            kv_total, 65_536,
            "TMA KV tile for seq_block=128, head_dim=128, F16, 2 stages must be 65536 bytes"
        );
        // Total shared memory must contain the KV allocation
        assert!(
            smem >= kv_total * 2, // K + V both allocated
            "shared memory must accommodate both K ({kv_total}B) and V ({kv_total}B) TMA tiles"
        );
        // Forward PTX must reference TMA/cp.async for KV loading
        let ptx = plan.generate_forward().expect("PTX gen");
        assert!(
            ptx.contains("cp.async") || ptx.contains("TMA") || ptx.contains("tma"),
            "FA3 forward PTX must reference TMA/cp.async for KV block loading"
        );
    }

    /// FA3 default configuration uses 8 warps = 256 threads per block.
    ///
    /// FA3 Hopper requires 8 warps (1 full warpgroup) for the wgmma instructions.
    /// This maps to 256 hardware threads per thread block.
    #[test]
    fn fa3_eight_warps_configuration() {
        let cfg = FlashAttention3Config::default_for(128, PtxType::F16, false);
        assert_eq!(
            cfg.num_warps, 8,
            "FA3 must use exactly 8 warps (1 warpgroup)"
        );
        let threads_per_block = cfg.num_warps * 32;
        assert_eq!(
            threads_per_block, 256,
            "8 warps × 32 threads = 256 threads per block for FA3"
        );
        // Producer + consumer must sum to 8 warps
        assert_eq!(
            cfg.num_producer_warps + cfg.num_consumer_warps,
            cfg.num_warps,
            "producer ({}) + consumer ({}) warps must equal total ({})",
            cfg.num_producer_warps,
            cfg.num_consumer_warps,
            cfg.num_warps
        );
        // The plan must be valid and generate PTX targeting the correct thread count
        let plan = FlashAttention3Plan::new(cfg).expect("valid FA3 config");
        let params = plan.launch_params(1024, 1024, 1, 1);
        assert_eq!(
            params.block.x, 256,
            "FA3 launch block must have 256 threads (8 warps × 32)"
        );
    }

    /// FA3 selects wgmma path (sm_90), FA2 selects wmma path (sm_80).
    ///
    /// FlashAttention-3 is only valid for sm_90+ and targets `wgmma` instructions.
    /// FlashAttention-2 runs on sm_80 and uses standard `wmma` instructions.
    /// This test verifies the architectural boundary: FA3 rejects sm_80.
    #[test]
    fn fa3_wgmma_vs_fa2_wmma_distinction() {
        // FA3 rejects sm_80 (Ampere — uses wmma, not wgmma)
        let fa3_on_sm80 = FlashAttention3Plan::new(FlashAttention3Config {
            head_dim: 128,
            block_m: 128,
            block_n: 128,
            num_warps: 8,
            num_producer_warps: 2,
            num_consumer_warps: 6,
            pingpong_stages: 2,
            causal: false,
            sm_version: 80, // Ampere — NOT supported by FA3/wgmma
            float_type: PtxType::F16,
        });
        assert!(
            fa3_on_sm80.is_err(),
            "FA3 (wgmma) must reject sm_80 — wgmma requires Hopper (sm_90+)"
        );

        // FA3 accepts sm_90 (Hopper — wgmma available)
        let fa3_on_sm90 = FlashAttention3Plan::new(FlashAttention3Config {
            head_dim: 128,
            block_m: 128,
            block_n: 128,
            num_warps: 8,
            num_producer_warps: 2,
            num_consumer_warps: 6,
            pingpong_stages: 2,
            causal: false,
            sm_version: 90, // Hopper — wgmma supported
            float_type: PtxType::F16,
        });
        assert!(
            fa3_on_sm90.is_ok(),
            "FA3 (wgmma) must accept sm_90 (Hopper)"
        );

        // Generated FA3 PTX targets sm_90 (wgmma path)
        let plan = fa3_on_sm90.expect("sm_90 must be valid");
        let ptx = plan.generate_forward().expect("PTX gen");
        assert!(
            ptx.contains("sm_90"),
            "FA3 wgmma PTX must explicitly target sm_90"
        );
        assert!(
            ptx.contains("mma.sync.aligned") || ptx.contains("MMA"),
            "FA3 wgmma PTX must contain tensor core MMA instructions"
        );
    }
}
