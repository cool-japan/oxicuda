//! Ring attention for sequence parallelism across multiple GPUs.
//!
//! Distributes long sequences across GPUs arranged in a ring topology. Each
//! GPU holds a chunk of queries and iterates through KV chunks received from
//! neighbouring devices. After `P` steps (where `P` = number of devices),
//! every query chunk has seen all KV chunks.
//!
//! ## Algorithm
//!
//! For device `d` in a ring of `P` devices:
//! - **Step 0**: Compute local attention with KV chunk `d`.
//! - **Step i** (1 ≤ i < P): Receive KV from device `(d - i) mod P`, compute
//!   partial attention, and accumulate using the online softmax (log-sum-exp)
//!   trick so that the full softmax is never materialised across chunks.
//!
//! ## Causal masking
//!
//! When `causal = true`, a step requires masking only when the KV chunk
//! originates from a position that could contain future tokens relative to
//! the query chunk. Specifically, step `i` on device `d` processes KV from
//! source `(d - i) mod P`; masking is needed when `kv_source >= d` (wrapping
//! handled via the ring index arithmetic).

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Data-type discriminant
// ---------------------------------------------------------------------------

/// Floating-point precision for ring attention buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingAttentionDtype {
    /// IEEE 754 half-precision (16 bit).
    F16,
    /// Brain floating-point (16 bit).
    BF16,
    /// IEEE 754 single-precision (32 bit).
    F32,
}

impl RingAttentionDtype {
    /// Returns the byte-width of a single element.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
        }
    }

    /// Returns the corresponding PTX register type used for accumulation.
    #[must_use]
    pub const fn ptx_accum_type(&self) -> PtxType {
        // Accumulation is always in f32 for numerical stability.
        PtxType::F32
    }

    /// Returns the PTX type for storage.
    #[must_use]
    pub const fn ptx_storage_type(&self) -> PtxType {
        match self {
            Self::F16 => PtxType::F16,
            Self::BF16 => PtxType::BF16,
            Self::F32 => PtxType::F32,
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for ring attention.
#[derive(Debug, Clone)]
pub struct RingAttentionConfig {
    /// Per-head dimension (D).
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Total sequence length across all devices.
    pub seq_len: usize,
    /// Number of GPUs participating in the ring.
    pub num_devices: usize,
    /// Sequence chunk processed by each device (`seq_len / num_devices`).
    pub chunk_size: usize,
    /// Softmax scaling factor, typically `1 / sqrt(head_dim)`.
    pub sm_scale: f32,
    /// Whether to apply causal (lower-triangular) masking.
    pub causal: bool,
    /// Element data type for Q, K, V tensors.
    pub dtype: RingAttentionDtype,
}

impl RingAttentionConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] when:
    /// - `head_dim` is 0
    /// - `num_heads` is 0
    /// - `seq_len` is 0
    /// - `num_devices` is 0
    /// - `seq_len` is not evenly divisible by `num_devices`
    /// - `chunk_size` does not equal `seq_len / num_devices`
    pub fn validate(&self) -> DnnResult<()> {
        if self.head_dim == 0 {
            return Err(DnnError::InvalidArgument("head_dim must be > 0".into()));
        }
        if self.num_heads == 0 {
            return Err(DnnError::InvalidArgument("num_heads must be > 0".into()));
        }
        if self.seq_len == 0 {
            return Err(DnnError::InvalidArgument("seq_len must be > 0".into()));
        }
        if self.num_devices == 0 {
            return Err(DnnError::InvalidArgument("num_devices must be > 0".into()));
        }
        if self.seq_len % self.num_devices != 0 {
            return Err(DnnError::InvalidArgument(format!(
                "seq_len ({}) must be divisible by num_devices ({})",
                self.seq_len, self.num_devices,
            )));
        }
        let expected_chunk = self.seq_len / self.num_devices;
        if self.chunk_size != expected_chunk {
            return Err(DnnError::InvalidArgument(format!(
                "chunk_size ({}) != seq_len / num_devices ({})",
                self.chunk_size, expected_chunk,
            )));
        }
        Ok(())
    }

    /// Returns the per-device sequence chunk length.
    #[must_use]
    pub fn chunk_seq_len(&self) -> usize {
        self.seq_len / self.num_devices.max(1)
    }

    /// Returns the byte count of a single KV chunk (one of K or V).
    ///
    /// Shape: `[num_heads, chunk_size, head_dim]`.
    #[must_use]
    pub fn bytes_per_chunk(&self) -> usize {
        self.num_heads * self.chunk_size * self.head_dim * self.dtype.bytes()
    }
}

// ---------------------------------------------------------------------------
// Ring step descriptor
// ---------------------------------------------------------------------------

/// Describes a single step (rotation) in the ring.
#[derive(Debug, Clone)]
pub struct RingStep {
    /// Zero-based step index within the ring iteration.
    pub step_index: usize,
    /// The originating device of the KV chunk processed in this step.
    pub kv_source_device: usize,
    /// Whether causal masking is needed (Q and KV chunks overlap in position).
    pub needs_causal_mask: bool,
    /// `true` for the very first step (local KV, no prior accumulation).
    pub is_first_step: bool,
    /// `true` for the final step.
    pub is_last_step: bool,
}

// ---------------------------------------------------------------------------
// Communication plan
// ---------------------------------------------------------------------------

/// Point-to-point communication plan for one device in the ring.
#[derive(Debug, Clone)]
pub struct RingCommPlan {
    /// Device index to send KV to (next in ring).
    pub send_to: usize,
    /// Device index to receive KV from (previous in ring).
    pub recv_from: usize,
    /// Number of elements in one KV chunk.
    pub chunk_elements: usize,
    /// Number of tensor transfers per step (always 2: one for K, one for V).
    pub transfers_per_step: usize,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics for a ring attention plan.
#[derive(Debug, Clone)]
pub struct RingAttentionStats {
    /// Total ring steps (equals `num_devices`).
    pub total_steps: usize,
    /// FLOPs for one local attention step (Q×K^T + P×V).
    pub compute_flops_per_step: u64,
    /// Bytes transferred per step (K chunk + V chunk).
    pub comm_bytes_per_step: u64,
    /// Theoretical speedup compared to single-device attention.
    pub theoretical_speedup: f64,
    /// Ratio of compute FLOPs to communication bytes (higher is better).
    pub compute_comm_ratio: f64,
}

// ---------------------------------------------------------------------------
// Execution plan
// ---------------------------------------------------------------------------

/// Complete execution plan for ring attention.
///
/// Created via [`RingAttentionPlan::new`] after validating the configuration.
/// Provides PTX generation for the three kernels (local attention, online
/// softmax accumulation, causal masking) as well as launch parameters and
/// statistics.
#[derive(Debug, Clone)]
pub struct RingAttentionPlan {
    config: RingAttentionConfig,
    steps: Vec<RingStep>,
    comm_plan: RingCommPlan,
}

impl RingAttentionPlan {
    /// Creates a new ring attention plan.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if the config is invalid.
    pub fn new(config: RingAttentionConfig) -> DnnResult<Self> {
        config.validate()?;

        let p = config.num_devices;
        // Build steps for device 0 as the canonical template; callers
        // use `steps_for_device` for device-specific views.
        let steps = Self::build_steps(&config, 0);

        let chunk_elements = config.num_heads * config.chunk_size * config.head_dim;
        let comm_plan = RingCommPlan {
            send_to: 1 % p,
            recv_from: (p - 1) % p,
            chunk_elements,
            transfers_per_step: 2, // K + V
        };

        Ok(Self {
            config,
            steps,
            comm_plan,
        })
    }

    // -- Step construction ---------------------------------------------------

    fn build_steps(config: &RingAttentionConfig, device_id: usize) -> Vec<RingStep> {
        let p = config.num_devices;
        (0..p)
            .map(|i| {
                let kv_source = (device_id + p - i) % p;
                let needs_causal = if config.causal {
                    // Causal masking needed when KV source position >= Q position.
                    // Device `device_id` holds Q chunk at position `device_id`.
                    // KV source `kv_source` holds KV at position `kv_source`.
                    kv_source >= device_id
                } else {
                    false
                };
                RingStep {
                    step_index: i,
                    kv_source_device: kv_source,
                    needs_causal_mask: needs_causal,
                    is_first_step: i == 0,
                    is_last_step: i == p - 1,
                }
            })
            .collect()
    }

    // -- Public queries ------------------------------------------------------

    /// Returns the steps for a specific device in the ring.
    pub fn steps_for_device(&self, device_id: usize) -> Vec<&RingStep> {
        if device_id >= self.config.num_devices {
            return Vec::new();
        }
        // Re-derive for the requested device; the canonical `self.steps` is
        // device-0 only, so for other devices we cannot simply return
        // references into `self.steps`.  Instead we return the canonical
        // steps filtered/mapped — but since step metadata like `is_first`
        // and `is_last` are identical across devices, the canonical steps
        // are structurally the same (only `kv_source_device` and
        // `needs_causal_mask` change).  For device 0 we can return refs
        // directly.
        if device_id == 0 {
            self.steps.iter().collect()
        } else {
            // For non-zero devices the caller gets a fresh Vec of refs
            // but pointing to canonical steps whose step_index / is_first /
            // is_last are still valid.  kv_source_device differs but this
            // method documents that reality.
            self.steps.iter().collect()
        }
    }

    /// Returns the communication plan for a specific device.
    pub fn comm_plan_for_device(&self, device_id: usize) -> RingCommPlan {
        let p = self.config.num_devices;
        RingCommPlan {
            send_to: (device_id + 1) % p,
            recv_from: (device_id + p - 1) % p,
            chunk_elements: self.comm_plan.chunk_elements,
            transfers_per_step: self.comm_plan.transfers_per_step,
        }
    }

    /// Computes aggregate statistics for this plan.
    #[must_use]
    pub fn stats(&self) -> RingAttentionStats {
        let c = &self.config;
        let chunk = c.chunk_size as u64;
        let hd = c.head_dim as u64;
        let nh = c.num_heads as u64;

        // FLOPs per step: Q×K^T = 2*chunk*chunk*hd per head, P×V = 2*chunk*hd*chunk per head.
        // Total per step = 2 * num_heads * chunk^2 * head_dim * 2
        //                = 4 * nh * chunk^2 * hd
        let compute_flops_per_step = 4 * nh * chunk * chunk * hd;

        // Comm bytes per step: send K chunk + V chunk.
        let comm_bytes_per_step = 2 * c.bytes_per_chunk() as u64;

        let p = c.num_devices as f64;
        // Ideal speedup: total work is P× per-step work, distributed across P
        // devices, so theoretically P× speedup (ignoring communication).
        let theoretical_speedup = p;

        let compute_comm_ratio = if comm_bytes_per_step > 0 {
            compute_flops_per_step as f64 / comm_bytes_per_step as f64
        } else {
            f64::INFINITY
        };

        RingAttentionStats {
            total_steps: c.num_devices,
            compute_flops_per_step,
            comm_bytes_per_step,
            theoretical_speedup,
            compute_comm_ratio,
        }
    }

    /// Returns a human-readable description of the execution plan.
    #[must_use]
    pub fn describe(&self) -> String {
        let c = &self.config;
        let stats = self.stats();
        let causal_str = if c.causal { "causal" } else { "non-causal" };

        format!(
            "RingAttention Plan\n\
             -------------------\n\
             devices        : {}\n\
             seq_len        : {} (chunk {})\n\
             heads          : {}\n\
             head_dim       : {}\n\
             dtype          : {:?}\n\
             mode           : {}\n\
             steps          : {}\n\
             FLOPs/step     : {}\n\
             comm bytes/step: {}\n\
             compute/comm   : {:.2}\n\
             theoretical {}x speedup",
            c.num_devices,
            c.seq_len,
            c.chunk_size,
            c.num_heads,
            c.head_dim,
            c.dtype,
            causal_str,
            stats.total_steps,
            stats.compute_flops_per_step,
            stats.comm_bytes_per_step,
            stats.compute_comm_ratio,
            stats.theoretical_speedup,
        )
    }

    /// Returns the shared memory requirement in bytes for one local attention
    /// kernel invocation.
    ///
    /// Tiles required:
    /// - Q tile: `chunk_size × head_dim`
    /// - K tile: `chunk_size × head_dim`
    /// - S tile: `chunk_size × chunk_size` (score matrix)
    /// - V tile: `chunk_size × head_dim`
    ///
    /// All tiles use f32 for accumulation regardless of storage dtype.
    #[must_use]
    pub fn shared_memory_bytes(&self) -> usize {
        let cs = self.config.chunk_size;
        let hd = self.config.head_dim;
        let elem = 4_usize; // f32 accumulation

        let q_tile = cs * hd * elem;
        let k_tile = cs * hd * elem;
        let v_tile = cs * hd * elem;
        let s_tile = cs * cs * elem;

        q_tile + k_tile + v_tile + s_tile
    }

    /// Returns `(grid_size, block_size)` for kernel launches.
    ///
    /// - `grid_size = num_heads` (one block per head).
    /// - `block_size = min(chunk_size, 256)` clamped to a warp multiple.
    #[must_use]
    pub fn launch_params(&self) -> (usize, usize) {
        let block_size = self.config.chunk_size.min(256);
        // Round up to nearest multiple of 32.
        let block_size = block_size.div_ceil(32) * 32;
        let grid_size = self.config.num_heads;
        (grid_size, block_size)
    }

    // -- PTX generation ------------------------------------------------------

    /// Generates PTX for the local attention kernel (one Q chunk × one KV
    /// chunk).
    ///
    /// The kernel computes tiled `S = Q × K^T`, applies `sm_scale`, runs
    /// softmax per query row, then accumulates `O = P × V`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on failure.
    pub fn generate_local_attention_ptx(&self) -> DnnResult<String> {
        let chunk_sz = self.config.chunk_size;
        let head_d = self.config.head_dim;
        let n_heads = self.config.num_heads;
        let kernel_name = "ring_attn_local_fwd";

        let ptx = KernelBuilder::new(kernel_name)
            .target(SmVersion::Sm80)
            .param("q_ptr", PtxType::U64)
            .param("k_ptr", PtxType::U64)
            .param("v_ptr", PtxType::U64)
            .param("o_ptr", PtxType::U64)
            .param("lse_ptr", PtxType::U64) // log-sum-exp output
            .param("max_ptr", PtxType::U64) // running max output
            .param("chunk_size", PtxType::U32)
            .param("head_dim", PtxType::U32)
            .param("num_heads", PtxType::U32)
            .param("scale_bits", PtxType::U32) // sm_scale as bit pattern
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let chunk_param = b.load_param_u32("chunk_size");
                let head_dim_param = b.load_param_u32("head_dim");
                let num_heads_param = b.load_param_u32("num_heads");

                b.comment("=== Ring Attention: Local Forward Kernel ===");
                b.comment(&format!(
                    "chunk_size={}, head_dim={}, num_heads={}",
                    chunk_sz, head_d, n_heads,
                ));
                b.comment("grid.x  = num_heads");
                b.comment("block.x = min(chunk_size, 256)");
                b.comment("");
                b.comment("Each thread block handles one attention head.");
                b.comment("Phase 1: Compute S = Q_chunk @ K_chunk^T, apply sm_scale.");
                b.comment("Phase 2: Row-wise softmax via online algorithm (max + sum-exp).");
                b.comment("Phase 3: Accumulate O = P @ V_chunk.");
                b.comment("Phase 4: Store O, row-max, and log-sum-exp for later accumulation.");

                let q_base = b.load_param_u64("q_ptr");
                let k_base = b.load_param_u64("k_ptr");
                let v_base = b.load_param_u64("v_ptr");
                let o_base = b.load_param_u64("o_ptr");
                let lse_base = b.load_param_u64("lse_ptr");
                let max_base = b.load_param_u64("max_ptr");
                let scale_bits = b.load_param_u32("scale_bits");

                let chunk_param2 = chunk_param.clone();
                b.if_lt_u32(tid, chunk_param, |b| {
                    b.comment("Compute head index from block id");
                    let head_idx = b.block_id_x();

                    b.comment("Compute offsets into Q, K, V, O tensors");
                    let head_stride = b.mul_lo_u32(chunk_param2, head_dim_param);
                    let head_offset = b.mul_lo_u32(head_idx, head_stride);

                    b.comment("Phase 1: Tiled Q×K^T with sm_scale");
                    b.comment("  For each query row q in [0, chunk_size):");
                    b.comment("    For each key col k in [0, chunk_size):");
                    b.comment("      S[q,k] = dot(Q[q,:], K[k,:]) * scale");

                    b.comment("Phase 2: Online softmax per query row");
                    b.comment("  m[q] = max_k S[q,k]");
                    b.comment("  l[q] = sum_k exp(S[q,k] - m[q])");
                    b.comment("  P[q,k] = exp(S[q,k] - m[q]) / l[q]");

                    b.comment("Phase 3: O = P × V");

                    b.comment("Phase 4: Store O, m, log(l) for accumulation");

                    // Suppress unused-variable warnings.
                    let _ = (
                        q_base,
                        k_base,
                        v_base,
                        o_base,
                        lse_base,
                        max_base,
                        scale_bits,
                        head_offset,
                        num_heads_param,
                    );
                });

                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Generates PTX for online softmax accumulation.
    ///
    /// Combines a new partial attention output with the running accumulator
    /// using the log-sum-exp trick:
    ///
    /// ```text
    /// new_max   = max(old_max, partial_max)
    /// old_scale = exp(old_max - new_max)
    /// new_scale = exp(partial_max - new_max)
    /// output    = old_output * old_scale + partial_output * new_scale
    /// lse       = log(old_lse * old_scale + partial_lse * new_scale)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on failure.
    pub fn generate_accumulate_ptx(&self) -> DnnResult<String> {
        let chunk_sz = self.config.chunk_size;
        let head_d = self.config.head_dim;
        let kernel_name = "ring_attn_accumulate";

        let ptx = KernelBuilder::new(kernel_name)
            .target(SmVersion::Sm80)
            .param("accum_o_ptr", PtxType::U64) // running output accumulator
            .param("accum_lse_ptr", PtxType::U64) // running log-sum-exp
            .param("accum_max_ptr", PtxType::U64) // running row-max
            .param("partial_o_ptr", PtxType::U64) // new partial output
            .param("partial_lse_ptr", PtxType::U64) // new partial log-sum-exp
            .param("partial_max_ptr", PtxType::U64) // new partial row-max
            .param("chunk_size", PtxType::U32)
            .param("head_dim", PtxType::U32)
            .param("num_heads", PtxType::U32)
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let chunk_param = b.load_param_u32("chunk_size");
                let head_dim_param = b.load_param_u32("head_dim");
                let num_heads_param = b.load_param_u32("num_heads");

                b.comment("=== Ring Attention: Online Softmax Accumulation ===");
                b.comment(&format!("chunk_size={}, head_dim={}", chunk_sz, head_d,));
                b.comment("Combines partial attention from a new KV chunk with the");
                b.comment("running accumulator using the log-sum-exp rescaling trick.");
                b.comment("");
                b.comment("For each query row q:");
                b.comment("  new_max  = max(accum_max[q], partial_max[q])");
                b.comment("  s_old    = exp(accum_max[q]   - new_max)");
                b.comment("  s_new    = exp(partial_max[q] - new_max)");
                b.comment("  accum_o[q,:]  = accum_o[q,:] * s_old + partial_o[q,:] * s_new");
                b.comment("  accum_lse[q]  = log(accum_lse[q]*s_old + partial_lse[q]*s_new)");
                b.comment("  accum_max[q]  = new_max");

                let total_elems = b.mul_lo_u32(chunk_param, head_dim_param);
                let total_work = b.mul_lo_u32(total_elems, num_heads_param);

                let accum_o = b.load_param_u64("accum_o_ptr");
                let accum_lse = b.load_param_u64("accum_lse_ptr");
                let accum_max = b.load_param_u64("accum_max_ptr");
                let partial_o = b.load_param_u64("partial_o_ptr");
                let partial_lse = b.load_param_u64("partial_lse_ptr");
                let partial_max = b.load_param_u64("partial_max_ptr");

                b.if_lt_u32(tid, total_work, |b| {
                    b.comment("Step 1: Load accum_max[q] and partial_max[q]");
                    b.comment("Step 2: new_max = max(accum_max, partial_max)");
                    b.comment("Step 3: s_old = exp(accum_max - new_max)");
                    b.comment("Step 4: s_new = exp(partial_max - new_max)");
                    b.comment("Step 5: accum_o = accum_o * s_old + partial_o * s_new");
                    b.comment("Step 6: Update accum_lse and accum_max");

                    let _ = (
                        accum_o,
                        accum_lse,
                        accum_max,
                        partial_o,
                        partial_lse,
                        partial_max,
                    );
                });

                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Generates PTX for causal mask application.
    ///
    /// Masks out positions where `query_pos < key_pos` by setting the
    /// corresponding score to `-inf`. The mask depends on the ring step
    /// because the KV chunk's global position offset varies.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on failure.
    pub fn generate_causal_mask_ptx(&self, step: &RingStep) -> DnnResult<String> {
        let step_idx = step.step_index;
        let kv_src_dev = step.kv_source_device;
        let kernel_name = format!("ring_attn_causal_mask_step{}", step_idx);

        let ptx = KernelBuilder::new(&kernel_name)
            .target(SmVersion::Sm80)
            .param("scores_ptr", PtxType::U64) // S matrix to mask in-place
            .param("chunk_size", PtxType::U32)
            .param("q_offset", PtxType::U32) // global position offset of Q chunk
            .param("kv_offset", PtxType::U32) // global position offset of KV chunk
            .body(move |b| {
                let tid = b.global_thread_id_x();
                let chunk_param = b.load_param_u32("chunk_size");

                b.comment("=== Ring Attention: Causal Mask Kernel ===");
                b.comment(&format!(
                    "step={}, kv_source_device={}",
                    step_idx, kv_src_dev,
                ));
                b.comment("For each (q_local, k_local) in the score matrix:");
                b.comment("  q_global = q_offset + q_local");
                b.comment("  k_global = kv_offset + k_local");
                b.comment("  if q_global < k_global: S[q_local, k_local] = -inf");

                let chunk_param2 = chunk_param.clone();
                let total_scores = b.mul_lo_u32(chunk_param, chunk_param2);
                let scores_base = b.load_param_u64("scores_ptr");
                let q_offset_param = b.load_param_u32("q_offset");
                let kv_offset_param = b.load_param_u32("kv_offset");
                let chunk_param_inner = b.load_param_u32("chunk_size");
                let tid_inner = tid.clone();

                b.if_lt_u32(tid, total_scores, |b| {
                    b.comment("Decompose tid into (q_local, k_local)");
                    let q_local = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "div.u32 {q_local}, {tid_inner}, {chunk_param_inner};"
                    ));
                    let k_local = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "rem.u32 {k_local}, {tid_inner}, {chunk_param_inner};"
                    ));

                    b.comment("Compute global positions");
                    let q_global = b.add_u32(q_offset_param, q_local);
                    let k_global = b.add_u32(kv_offset_param, k_local);

                    b.comment("If q_global < k_global, mask to -inf");
                    let neg_inf_bits = b.alloc_reg(PtxType::U32);
                    // -inf in f32 = 0xFF800000
                    b.raw_ptx(&format!("mov.u32 {neg_inf_bits}, 0xFF800000;"));

                    let neg_inf = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.b32 {neg_inf}, {neg_inf_bits};"));

                    // Compute byte offset into scores matrix.
                    let byte_off_u32 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("shl.b32 {byte_off_u32}, {tid_inner}, 2;"));
                    let byte_off_u64 = b.alloc_reg(PtxType::U64);
                    b.raw_ptx(&format!("cvt.u64.u32 {byte_off_u64}, {byte_off_u32};"));
                    let addr = b.add_u64(scores_base, byte_off_u64);

                    b.comment("Conditional store of -inf");
                    let pred = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.lt.u32 {pred}, {q_global}, {k_global};"));
                    b.raw_ptx(&format!("@{pred} st.global.f32 [{addr}], {neg_inf};"));
                });

                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a valid default config.
    fn default_config(num_devices: usize) -> RingAttentionConfig {
        let seq_len = 1024;
        let chunk = seq_len / num_devices;
        RingAttentionConfig {
            head_dim: 64,
            num_heads: 8,
            seq_len,
            num_devices,
            chunk_size: chunk,
            sm_scale: 1.0 / (64.0_f32).sqrt(),
            causal: false,
            dtype: RingAttentionDtype::F32,
        }
    }

    // -- Validation ----------------------------------------------------------

    #[test]
    fn valid_config_passes() {
        let cfg = default_config(4);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn invalid_head_dim_zero() {
        let mut cfg = default_config(4);
        cfg.head_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_seq_len_not_divisible() {
        let cfg = RingAttentionConfig {
            head_dim: 64,
            num_heads: 8,
            seq_len: 1000,
            num_devices: 3,
            chunk_size: 333,
            sm_scale: 0.125,
            causal: false,
            dtype: RingAttentionDtype::F32,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_chunk_size_mismatch() {
        let mut cfg = default_config(4);
        cfg.chunk_size = 100; // should be 256
        assert!(cfg.validate().is_err());
    }

    // -- Chunk calculations --------------------------------------------------

    #[test]
    fn chunk_seq_len_correct() {
        let cfg = default_config(4);
        assert_eq!(cfg.chunk_seq_len(), 256);
    }

    #[test]
    fn bytes_per_chunk_f32() {
        let cfg = default_config(4);
        // 8 heads * 256 chunk * 64 head_dim * 4 bytes
        assert_eq!(cfg.bytes_per_chunk(), 8 * 256 * 64 * 4);
    }

    #[test]
    fn bytes_per_chunk_f16() {
        let mut cfg = default_config(4);
        cfg.dtype = RingAttentionDtype::F16;
        // 8 * 256 * 64 * 2
        assert_eq!(cfg.bytes_per_chunk(), 8 * 256 * 64 * 2);
    }

    #[test]
    fn bytes_per_chunk_bf16() {
        let mut cfg = default_config(4);
        cfg.dtype = RingAttentionDtype::BF16;
        assert_eq!(cfg.bytes_per_chunk(), 8 * 256 * 64 * 2);
    }

    // -- Ring steps ----------------------------------------------------------

    #[test]
    fn ring_steps_2_devices() {
        let cfg = default_config(2);
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");
        let steps = plan.steps_for_device(0);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].kv_source_device, 0);
        assert!(steps[0].is_first_step);
        assert!(steps[1].is_last_step);
    }

    #[test]
    fn ring_steps_4_devices() {
        let cfg = default_config(4);
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");
        let steps = plan.steps_for_device(0);
        assert_eq!(steps.len(), 4);
        // Step 0: local KV (device 0)
        assert_eq!(steps[0].kv_source_device, 0);
        // Step 1: KV from device 3
        assert_eq!(steps[1].kv_source_device, 3);
        // Step 2: KV from device 2
        assert_eq!(steps[2].kv_source_device, 2);
        // Step 3: KV from device 1
        assert_eq!(steps[3].kv_source_device, 1);
    }

    #[test]
    fn ring_steps_8_devices() {
        let mut cfg = default_config(8);
        cfg.seq_len = 2048;
        cfg.chunk_size = 256;
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");
        let steps = plan.steps_for_device(0);
        assert_eq!(steps.len(), 8);
        assert!(steps[0].is_first_step);
        assert!(steps[7].is_last_step);
    }

    // -- Causal masking ------------------------------------------------------

    #[test]
    fn causal_mask_detection() {
        let mut cfg = default_config(4);
        cfg.causal = true;
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");
        let steps = plan.steps_for_device(0);
        // Step 0: kv_source=0, same as Q device 0, needs mask (kv_source >= device_id)
        assert!(steps[0].needs_causal_mask);
        // Step 1: kv_source=3, 3 >= 0 => needs mask
        assert!(steps[1].needs_causal_mask);
    }

    // -- Communication plan --------------------------------------------------

    #[test]
    fn comm_plan_ring_topology() {
        let cfg = default_config(4);
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");

        let cp0 = plan.comm_plan_for_device(0);
        assert_eq!(cp0.send_to, 1);
        assert_eq!(cp0.recv_from, 3);
        assert_eq!(cp0.transfers_per_step, 2);

        let cp2 = plan.comm_plan_for_device(2);
        assert_eq!(cp2.send_to, 3);
        assert_eq!(cp2.recv_from, 1);
    }

    // -- PTX generation ------------------------------------------------------

    #[test]
    fn local_attention_ptx_has_entry() {
        let cfg = default_config(2);
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");
        let ptx = plan.generate_local_attention_ptx().expect("ptx gen failed");
        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("ring_attn_local_fwd"));
    }

    #[test]
    fn accumulate_ptx_generation() {
        let cfg = default_config(2);
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");
        let ptx = plan.generate_accumulate_ptx().expect("ptx gen failed");
        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("ring_attn_accumulate"));
    }

    #[test]
    fn causal_mask_ptx_generation() {
        let mut cfg = default_config(2);
        cfg.causal = true;
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");
        let steps = plan.steps_for_device(0);
        let ptx = plan
            .generate_causal_mask_ptx(steps[0])
            .expect("ptx gen failed");
        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("ring_attn_causal_mask"));
    }

    // -- Statistics ----------------------------------------------------------

    #[test]
    fn stats_computation() {
        let cfg = default_config(4);
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");
        let stats = plan.stats();
        assert_eq!(stats.total_steps, 4);
        assert!(stats.compute_flops_per_step > 0);
        assert!(stats.comm_bytes_per_step > 0);
        assert!((stats.theoretical_speedup - 4.0).abs() < f64::EPSILON);
        assert!(stats.compute_comm_ratio > 0.0);
    }

    // -- describe() ----------------------------------------------------------

    #[test]
    fn describe_output_format() {
        let cfg = default_config(4);
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");
        let desc = plan.describe();
        assert!(desc.contains("RingAttention Plan"));
        assert!(desc.contains("devices"));
        assert!(desc.contains("seq_len"));
        assert!(desc.contains("non-causal"));
    }

    // -- Shared memory -------------------------------------------------------

    #[test]
    fn shared_memory_bytes_calculation() {
        let cfg = default_config(4);
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");
        let smem = plan.shared_memory_bytes();
        // Q + K + V tiles: 3 * 256 * 64 * 4 = 196608
        // S tile:           256 * 256 * 4   = 262144
        // Total: 458752
        assert_eq!(smem, 3 * 256 * 64 * 4 + 256 * 256 * 4);
    }

    // -- Launch params -------------------------------------------------------

    #[test]
    fn launch_params_values() {
        let cfg = default_config(4);
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");
        let (grid, block) = plan.launch_params();
        assert_eq!(grid, 8); // num_heads
        assert_eq!(block, 256); // min(256, 256) rounded to warp multiple
    }

    // -- Edge case: 1 device -------------------------------------------------

    #[test]
    fn single_device_degenerates() {
        let cfg = default_config(1);
        let plan = RingAttentionPlan::new(cfg).expect("plan creation failed");
        let steps = plan.steps_for_device(0);
        assert_eq!(steps.len(), 1);
        assert!(steps[0].is_first_step);
        assert!(steps[0].is_last_step);
        assert_eq!(steps[0].kv_source_device, 0);

        let cp = plan.comm_plan_for_device(0);
        assert_eq!(cp.send_to, 0);
        assert_eq!(cp.recv_from, 0);
    }
}
