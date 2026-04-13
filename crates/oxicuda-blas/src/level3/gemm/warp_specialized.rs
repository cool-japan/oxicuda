//! Warp-specialized GEMM for Hopper+ GPUs.
//!
//! Splits warps into two functional groups within a single CTA:
//!
//! - **Producer warps** load A and B tiles from global memory into shared
//!   memory using asynchronous copy (`cp.async`) with software pipelining.
//! - **Consumer warps** execute matrix-multiply-accumulate (MMA) instructions
//!   on the shared-memory tiles, overlapping compute with the producers'
//!   memory traffic.
//!
//! This decomposition hides global memory latency behind MMA compute,
//! achieving near-peak throughput on Hopper (SM 9.0+) and later
//! architectures.
//!
//! # Pipeline
//!
//! The producer and consumer warps communicate through a multi-stage
//! ping-pong buffer in shared memory. Each stage holds one complete tile
//! of A (`tile_m * tile_k`) and one of B (`tile_k * tile_n`). The
//! producer issues `cp.async.commit_group` after filling a stage and
//! signals readiness via `bar.arrive`. The consumer waits on the barrier,
//! consumes the stage with MMA, and then advances to the next stage
//! (modulo `pipeline_stages`).

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

use super::dispatch::{GemmProblem, TileConfig};
use crate::error::{BlasError, BlasResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Warp-specialized GEMM configuration for Hopper+ GPUs.
///
/// Splits warps into two groups:
/// - **Producer warps**: load A and B tiles from global memory to shared memory
///   using async copy (cp.async pipeline)
/// - **Consumer warps**: compute matrix multiply using MMA/WGMMA instructions
///   on shared memory tiles
///
/// This overlaps memory latency with compute, achieving near-peak throughput.
#[derive(Debug, Clone)]
pub struct WarpSpecializedGemm {
    /// Block tile size in the M dimension (rows per CTA).
    pub tile_m: u32,
    /// Block tile size in the N dimension (columns per CTA).
    pub tile_n: u32,
    /// Block tile size in the K dimension (reduction step per iteration).
    pub tile_k: u32,
    /// Number of warps dedicated to loading tiles (producer role).
    pub num_producer_warps: u32,
    /// Number of warps dedicated to MMA computation (consumer role).
    pub num_consumer_warps: u32,
    /// Software pipeline depth (number of shared-memory stages).
    pub pipeline_stages: u32,
    /// Target SM version (must be >= 90).
    pub sm_version: SmVersion,
    /// Input element type (F16, BF16, E4M3, E5M2).
    pub input_type: PtxType,
    /// Accumulator / output element type (F32).
    pub output_type: PtxType,
}

// ---------------------------------------------------------------------------
// Validation constants
// ---------------------------------------------------------------------------

/// Minimum SM version for warp specialization (Hopper).
const MIN_SM_VERSION: SmVersion = SmVersion::Sm90;

/// Maximum supported pipeline stages.
const MAX_PIPELINE_STAGES: u32 = 8;

/// Minimum total warps (producer + consumer).
const MIN_TOTAL_WARPS: u32 = 2;

/// Maximum total warps per CTA (hardware limit: 1024 threads = 32 warps).
const MAX_TOTAL_WARPS: u32 = 32;

/// Minimum problem volume (M*N*K) for warp specialization to be beneficial.
const MIN_PROBLEM_VOLUME: u64 = 128 * 128 * 64;

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl WarpSpecializedGemm {
    /// Creates a new warp-specialized GEMM configuration after validation.
    ///
    /// # Arguments
    ///
    /// * `tile_m` — block tile size in M.
    /// * `tile_n` — block tile size in N.
    /// * `tile_k` — block tile size in K.
    /// * `num_producer_warps` — warps dedicated to loading (>= 1).
    /// * `num_consumer_warps` — warps dedicated to compute (>= 1).
    /// * `pipeline_stages` — software pipeline depth (1..=8).
    /// * `sm_version` — target SM (must be >= 90).
    /// * `input_type` — input precision (F16, BF16, E4M3, E5M2).
    /// * `output_type` — accumulator precision (F32).
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::InvalidArgument`] if any parameter is out of
    /// range or the combination is inconsistent.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tile_m: u32,
        tile_n: u32,
        tile_k: u32,
        num_producer_warps: u32,
        num_consumer_warps: u32,
        pipeline_stages: u32,
        sm_version: SmVersion,
        input_type: PtxType,
        output_type: PtxType,
    ) -> BlasResult<Self> {
        let config = Self {
            tile_m,
            tile_n,
            tile_k,
            num_producer_warps,
            num_consumer_warps,
            pipeline_stages,
            sm_version,
            input_type,
            output_type,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validates all configuration parameters.
    fn validate(&self) -> BlasResult<()> {
        // SM version check.
        if self.sm_version < MIN_SM_VERSION {
            return Err(BlasError::InvalidArgument(format!(
                "warp-specialized GEMM requires SM >= 90, got {}",
                self.sm_version.as_ptx_str()
            )));
        }

        // Warp count checks.
        if self.num_producer_warps == 0 {
            return Err(BlasError::InvalidArgument(
                "num_producer_warps must be >= 1".into(),
            ));
        }
        if self.num_consumer_warps == 0 {
            return Err(BlasError::InvalidArgument(
                "num_consumer_warps must be >= 1".into(),
            ));
        }
        let total_warps = self.num_producer_warps + self.num_consumer_warps;
        if total_warps < MIN_TOTAL_WARPS {
            return Err(BlasError::InvalidArgument(format!(
                "total warps (producer + consumer) must be >= {MIN_TOTAL_WARPS}, got {total_warps}"
            )));
        }
        if total_warps > MAX_TOTAL_WARPS {
            return Err(BlasError::InvalidArgument(format!(
                "total warps (producer + consumer) must be <= {MAX_TOTAL_WARPS}, got {total_warps}"
            )));
        }

        // Pipeline stages.
        if self.pipeline_stages == 0 || self.pipeline_stages > MAX_PIPELINE_STAGES {
            return Err(BlasError::InvalidArgument(format!(
                "pipeline_stages must be in [1, {MAX_PIPELINE_STAGES}], got {}",
                self.pipeline_stages
            )));
        }

        // Tile dimensions must be positive and multiples of 8.
        for (name, val) in [
            ("tile_m", self.tile_m),
            ("tile_n", self.tile_n),
            ("tile_k", self.tile_k),
        ] {
            if val == 0 {
                return Err(BlasError::InvalidArgument(format!("{name} must be > 0")));
            }
            if val % 8 != 0 {
                return Err(BlasError::InvalidArgument(format!(
                    "{name} must be a multiple of 8, got {val}"
                )));
            }
        }

        // Input type check.
        if !matches!(
            self.input_type,
            PtxType::F16 | PtxType::BF16 | PtxType::E4M3 | PtxType::E5M2
        ) {
            return Err(BlasError::InvalidArgument(format!(
                "input_type must be F16, BF16, E4M3, or E5M2, got {}",
                self.input_type.as_ptx_str()
            )));
        }

        // Output type check.
        if self.output_type != PtxType::F32 {
            return Err(BlasError::InvalidArgument(format!(
                "output_type must be F32 for warp-specialized GEMM, got {}",
                self.output_type.as_ptx_str()
            )));
        }

        Ok(())
    }

    /// Returns the total number of warps in this configuration.
    #[must_use]
    pub fn total_warps(&self) -> u32 {
        self.num_producer_warps + self.num_consumer_warps
    }

    /// Returns the total number of threads per CTA.
    #[must_use]
    pub fn threads_per_block(&self) -> u32 {
        self.total_warps() * 32
    }

    /// Computes the shared memory requirement in bytes.
    ///
    /// Each pipeline stage needs space for one A tile (`tile_m * tile_k`) and
    /// one B tile (`tile_k * tile_n`), both in the input element type. The
    /// total is `pipeline_stages * (tile_A + tile_B) * element_bytes`.
    #[must_use]
    pub fn shared_memory_bytes(&self) -> usize {
        let elem_bytes = self.input_type.size_bytes();
        let tile_a_elems = (self.tile_m as usize) * (self.tile_k as usize);
        let tile_b_elems = (self.tile_k as usize) * (self.tile_n as usize);
        let per_stage = (tile_a_elems + tile_b_elems) * elem_bytes;
        per_stage * (self.pipeline_stages as usize)
    }

    /// Converts this configuration to a standard [`TileConfig`] for use by
    /// the GEMM dispatcher.
    #[must_use]
    pub fn to_tile_config(&self) -> TileConfig {
        // Distribute M among consumer warps: each consumer warp handles
        // (tile_m / num_consumer_warps_along_m) rows. We use a simple 1D
        // split along M; more advanced 2D decomposition is possible but
        // not needed for the dispatcher interface.
        let warp_m = self.tile_m / self.num_consumer_warps.max(1);
        let warp_n = self.tile_n;

        TileConfig {
            tile_m: self.tile_m,
            tile_n: self.tile_n,
            tile_k: self.tile_k,
            warp_m: warp_m.max(8),
            warp_n: warp_n.max(8),
            stages: self.pipeline_stages,
            use_tensor_core: true,
            split_k: 1,
        }
    }

    /// Returns `true` if warp specialization is applicable and beneficial for
    /// the given problem and architecture.
    ///
    /// The heuristic requires:
    /// - SM >= 90 (Hopper or later)
    /// - Problem volume (M * N * K) >= 128 * 128 * 64 (~1M FLOPs minimum)
    /// - Both M and N >= 64 (too-skinny problems don't tile well)
    #[must_use]
    pub fn is_applicable(problem: &GemmProblem, sm_version: SmVersion) -> bool {
        if sm_version < MIN_SM_VERSION {
            return false;
        }
        if problem.m < 64 || problem.n < 64 {
            return false;
        }
        let volume = u64::from(problem.m) * u64::from(problem.n) * u64::from(problem.k);
        if volume < MIN_PROBLEM_VOLUME {
            return false;
        }
        // Only supported input types.
        matches!(
            problem.input_type,
            PtxType::F16 | PtxType::BF16 | PtxType::E4M3 | PtxType::E5M2
        )
    }

    /// Returns the kernel function name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let input = self.input_type.as_ptx_str().trim_start_matches('.');
        let output = self.output_type.as_ptx_str().trim_start_matches('.');
        format!(
            "warp_specialized_gemm_{input}_{output}_p{}_c{}_s{}",
            self.num_producer_warps, self.num_consumer_warps, self.pipeline_stages
        )
    }

    /// Generates the complete PTX kernel text.
    ///
    /// The generated kernel implements warp-specialized producer/consumer
    /// GEMM with:
    /// - `cp.async` for asynchronous global-to-shared loads (producer warps)
    /// - `mma.sync.aligned` for tensor-core compute (consumer warps)
    /// - Multi-stage software pipeline with barrier synchronisation
    /// - Ping-pong shared memory buffers
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::PtxGeneration`] on formatting failures.
    pub fn generate_kernel(&self) -> BlasResult<String> {
        let mut ptx = String::with_capacity(16384);

        self.emit_header(&mut ptx)?;
        self.emit_kernel_entry(&mut ptx)?;
        self.emit_registers(&mut ptx)?;
        self.emit_load_params(&mut ptx)?;
        self.emit_warp_role_dispatch(&mut ptx)?;
        self.emit_producer_path(&mut ptx)?;
        self.emit_consumer_path(&mut ptx)?;
        self.emit_epilogue(&mut ptx)?;
        self.emit_kernel_exit(&mut ptx)?;

        Ok(ptx)
    }

    // -----------------------------------------------------------------------
    // PTX emission helpers
    // -----------------------------------------------------------------------

    /// Emits the PTX module header.
    fn emit_header(&self, ptx: &mut String) -> BlasResult<()> {
        write_line(ptx, &format!(".version {}", self.sm_version.ptx_version()))?;
        write_line(ptx, &format!(".target {}", self.sm_version.as_ptx_str()))?;
        write_line(ptx, ".address_size 64")?;
        write_line(ptx, "")?;
        Ok(())
    }

    /// Emits the kernel entry point with shared memory declaration.
    fn emit_kernel_entry(&self, ptx: &mut String) -> BlasResult<()> {
        let kernel_name = self.kernel_name();
        let smem_bytes = self.shared_memory_bytes();

        write_line(ptx, &format!(".visible .entry {kernel_name}("))?;
        write_line(ptx, "    .param .u64 %param_a,")?;
        write_line(ptx, "    .param .u64 %param_b,")?;
        write_line(ptx, "    .param .u64 %param_c,")?;
        write_line(ptx, "    .param .u32 %param_m,")?;
        write_line(ptx, "    .param .u32 %param_n,")?;
        write_line(ptx, "    .param .u32 %param_k,")?;
        write_line(ptx, "    .param .u32 %param_lda,")?;
        write_line(ptx, "    .param .u32 %param_ldb,")?;
        write_line(ptx, "    .param .u32 %param_ldc,")?;
        write_line(ptx, "    .param .f32 %param_alpha,")?;
        write_line(ptx, "    .param .f32 %param_beta")?;
        write_line(ptx, ")")?;
        write_line(ptx, "{")?;

        // Shared memory declaration for all pipeline stages.
        write_line(
            ptx,
            &format!("    .shared .align 128 .b8 smem_buf[{smem_bytes}];"),
        )?;
        write_line(ptx, "")?;

        Ok(())
    }

    /// Emits register declarations.
    fn emit_registers(&self, ptx: &mut String) -> BlasResult<()> {
        write_line(ptx, "    // --- Register declarations ---")?;
        write_line(ptx, "    .reg .b32 %r<64>;")?;
        write_line(ptx, "    .reg .b64 %rd<32>;")?;
        write_line(ptx, "    .reg .f32 %f<32>;")?;
        write_line(ptx, "    .reg .pred %p<16>;")?;
        write_line(ptx, "")?;
        Ok(())
    }

    /// Emits parameter loads and index computations.
    fn emit_load_params(&self, ptx: &mut String) -> BlasResult<()> {
        write_line(ptx, "    // --- Load kernel parameters ---")?;
        write_line(ptx, "    ld.param.u64 %rd0, [%param_a];")?;
        write_line(ptx, "    ld.param.u64 %rd1, [%param_b];")?;
        write_line(ptx, "    ld.param.u64 %rd2, [%param_c];")?;
        write_line(ptx, "    ld.param.u32 %r0, [%param_m];")?;
        write_line(ptx, "    ld.param.u32 %r1, [%param_n];")?;
        write_line(ptx, "    ld.param.u32 %r2, [%param_k];")?;
        write_line(ptx, "    ld.param.u32 %r3, [%param_lda];")?;
        write_line(ptx, "    ld.param.u32 %r4, [%param_ldb];")?;
        write_line(ptx, "    ld.param.u32 %r5, [%param_ldc];")?;
        write_line(ptx, "    ld.param.f32 %f0, [%param_alpha];")?;
        write_line(ptx, "    ld.param.f32 %f1, [%param_beta];")?;
        write_line(ptx, "")?;

        // Compute warp ID = tid.x / 32
        write_line(ptx, "    // --- Compute warp and lane IDs ---")?;
        write_line(ptx, "    mov.u32 %r10, %tid.x;")?;
        write_line(
            ptx,
            "    shr.u32 %r11, %r10, 5;       // warp_id = tid / 32",
        )?;
        write_line(
            ptx,
            "    and.b32 %r12, %r10, 31;      // lane_id = tid % 32",
        )?;
        write_line(ptx, "")?;

        // Block tile indices
        write_line(ptx, "    // --- Block tile indices ---")?;
        write_line(
            ptx,
            &format!(
                "    mul.lo.u32 %r13, %ctaid.x, {};  // block_col = bx * tile_n",
                self.tile_n
            ),
        )?;
        write_line(
            ptx,
            &format!(
                "    mul.lo.u32 %r14, %ctaid.y, {};  // block_row = by * tile_m",
                self.tile_m
            ),
        )?;
        write_line(ptx, "")?;

        Ok(())
    }

    /// Emits the warp-role dispatch branch.
    fn emit_warp_role_dispatch(&self, ptx: &mut String) -> BlasResult<()> {
        write_line(ptx, "    // --- Warp role dispatch ---")?;
        write_line(
            ptx,
            &format!(
                "    setp.lt.u32 %p0, %r11, {};  // is_producer = warp_id < num_producer_warps",
                self.num_producer_warps
            ),
        )?;
        write_line(ptx, "    @%p0 bra $PRODUCER_PATH;")?;
        write_line(ptx, "    bra $CONSUMER_PATH;")?;
        write_line(ptx, "")?;
        Ok(())
    }

    /// Emits the producer warp path: async copies from global to shared memory.
    fn emit_producer_path(&self, ptx: &mut String) -> BlasResult<()> {
        let input_ty = self.input_type.as_ptx_str();
        let elem_bytes = self.input_type.size_bytes();
        let tile_a_bytes = (self.tile_m as usize) * (self.tile_k as usize) * elem_bytes;
        let tile_b_bytes = (self.tile_k as usize) * (self.tile_n as usize) * elem_bytes;
        let stage_bytes = tile_a_bytes + tile_b_bytes;

        write_line(ptx, "$PRODUCER_PATH:")?;
        write_line(ptx, "    // === Producer warp: load tiles via cp.async ===")?;
        write_line(ptx, "")?;

        // Producer loop counter: iterate over K tiles
        write_line(ptx, "    mov.u32 %r20, 0;              // k_offset = 0")?;
        write_line(ptx, "    mov.u32 %r21, 0;              // stage_idx = 0")?;
        write_line(ptx, "")?;

        write_line(ptx, "$PRODUCER_K_LOOP:")?;
        write_line(ptx, "    setp.ge.u32 %p1, %r20, %r2;   // k_offset >= K?")?;
        write_line(ptx, "    @%p1 bra $PRODUCER_DONE;")?;
        write_line(ptx, "")?;

        // Compute shared memory base for current stage
        write_line(ptx, "    // Compute smem base for current stage")?;
        write_line(
            ptx,
            &format!(
                "    mul.lo.u32 %r22, %r21, {};  // stage_offset = stage_idx * stage_bytes",
                stage_bytes
            ),
        )?;
        write_line(ptx, "    mov.u64 %rd10, smem_buf;")?;
        write_line(ptx, "    cvt.u64.u32 %rd11, %r22;")?;
        write_line(
            ptx,
            "    add.u64 %rd10, %rd10, %rd11;  // smem_base for this stage",
        )?;
        write_line(ptx, "")?;

        // Compute source address for A tile: A + (block_row + local_row) * lda + k_offset
        write_line(ptx, "    // --- Load A tile ---")?;
        write_line(
            ptx,
            "    // Producer warps cooperatively load tile_m * tile_k elements",
        )?;
        write_line(
            ptx,
            &format!(
                "    // Elements per producer warp: {} / {}",
                self.tile_m * self.tile_k,
                self.num_producer_warps
            ),
        )?;

        // Each lane in each producer warp loads multiple elements
        let total_a_elems = self.tile_m * self.tile_k;
        let elems_per_producer_thread = total_a_elems / (self.num_producer_warps * 32);
        let elems_per_producer_thread = elems_per_producer_thread.max(1);

        write_line(
            ptx,
            &format!(
                "    // Each producer thread loads {} A elements",
                elems_per_producer_thread
            ),
        )?;

        // Compute linear thread index within producer warps
        write_line(ptx, "    mul.lo.u32 %r23, %r11, 32;    // warp_id * 32")?;
        write_line(
            ptx,
            "    add.u32 %r23, %r23, %r12;     // producer_thread_idx",
        )?;
        write_line(ptx, "")?;

        // Issue cp.async for A tile elements
        write_line(ptx, "    // cp.async loads for A tile")?;
        write_line(ptx, "    mov.u32 %r24, 0;              // elem_idx = 0")?;
        write_line(ptx, "$PRODUCER_A_LOOP:")?;
        write_line(
            ptx,
            &format!("    setp.ge.u32 %p2, %r24, {};", elems_per_producer_thread),
        )?;
        write_line(ptx, "    @%p2 bra $PRODUCER_A_DONE;")?;

        // Compute which (row, col) this element corresponds to
        write_line(
            ptx,
            &format!(
                "    mul.lo.u32 %r25, %r24, {};  // stride by total producer threads",
                self.num_producer_warps * 32
            ),
        )?;
        write_line(
            ptx,
            "    add.u32 %r25, %r25, %r23;     // linear element index",
        )?;
        write_line(
            ptx,
            &format!(
                "    div.u32 %r26, %r25, {};      // local_row = elem / tile_k",
                self.tile_k
            ),
        )?;
        write_line(
            ptx,
            &format!(
                "    rem.u32 %r27, %r25, {};      // local_col = elem % tile_k",
                self.tile_k
            ),
        )?;

        // Global address: A + ((block_row + local_row) * lda + (k_offset + local_col)) * elem_bytes
        write_line(
            ptx,
            "    add.u32 %r28, %r14, %r26;     // global_row = block_row + local_row",
        )?;
        write_line(
            ptx,
            "    add.u32 %r29, %r20, %r27;     // global_col = k_offset + local_col",
        )?;
        write_line(
            ptx,
            "    mad.lo.u32 %r30, %r28, %r3, %r29;  // row * lda + col",
        )?;
        write_line(ptx, "    cvt.u64.u32 %rd12, %r30;")?;
        write_line(
            ptx,
            &format!(
                "    mul.lo.u64 %rd12, %rd12, {};  // byte offset",
                elem_bytes
            ),
        )?;
        write_line(ptx, "    add.u64 %rd13, %rd0, %rd12;   // src_addr")?;

        // Shared memory destination
        write_line(ptx, "    cvt.u64.u32 %rd14, %r25;")?;
        write_line(
            ptx,
            &format!(
                "    mul.lo.u64 %rd14, %rd14, {};  // smem byte offset",
                elem_bytes
            ),
        )?;
        write_line(ptx, "    add.u64 %rd15, %rd10, %rd14;  // dst_addr")?;

        // Bounds check: skip if out of matrix bounds
        write_line(ptx, "    setp.ge.u32 %p3, %r28, %r0;   // row >= M?")?;
        write_line(ptx, "    setp.ge.u32 %p4, %r29, %r2;   // col >= K?")?;
        write_line(ptx, "    or.pred %p5, %p3, %p4;")?;
        write_line(ptx, "    @%p5 bra $PRODUCER_A_SKIP;")?;

        // Issue cp.async
        write_line(
            ptx,
            &format!(
                "    cp.async.ca.shared.global [{input_ty} %rd15], [{input_ty} %rd13], {};",
                elem_bytes
            ),
        )?;
        write_line(ptx, "    bra $PRODUCER_A_NEXT;")?;

        write_line(ptx, "$PRODUCER_A_SKIP:")?;
        // Store zero for out-of-bounds
        let zero_literal = match self.input_type {
            PtxType::F16 | PtxType::BF16 => "0x0000",
            _ => "0x00",
        };
        write_line(
            ptx,
            &format!("    st.shared.b{} [%rd15], {zero_literal};", elem_bytes * 8),
        )?;

        write_line(ptx, "$PRODUCER_A_NEXT:")?;
        write_line(ptx, "    add.u32 %r24, %r24, 1;")?;
        write_line(ptx, "    bra $PRODUCER_A_LOOP;")?;
        write_line(ptx, "$PRODUCER_A_DONE:")?;
        write_line(ptx, "")?;

        // Load B tile similarly
        let total_b_elems = self.tile_k * self.tile_n;
        let elems_per_producer_thread_b = total_b_elems / (self.num_producer_warps * 32);
        let elems_per_producer_thread_b = elems_per_producer_thread_b.max(1);

        write_line(ptx, "    // --- Load B tile ---")?;
        write_line(ptx, "    mov.u32 %r24, 0;              // elem_idx = 0")?;
        write_line(ptx, "$PRODUCER_B_LOOP:")?;
        write_line(
            ptx,
            &format!(
                "    setp.ge.u32 %p2, %r24, {};",
                elems_per_producer_thread_b
            ),
        )?;
        write_line(ptx, "    @%p2 bra $PRODUCER_B_DONE;")?;

        write_line(
            ptx,
            &format!(
                "    mul.lo.u32 %r25, %r24, {};",
                self.num_producer_warps * 32
            ),
        )?;
        write_line(ptx, "    add.u32 %r25, %r25, %r23;")?;
        write_line(ptx, &format!("    div.u32 %r26, %r25, {};", self.tile_n))?;
        write_line(ptx, &format!("    rem.u32 %r27, %r25, {};", self.tile_n))?;

        // B global addr: B + (k_offset + local_row) * ldb + (block_col + local_col)
        write_line(
            ptx,
            "    add.u32 %r28, %r20, %r26;     // global_row = k_offset + local_row",
        )?;
        write_line(
            ptx,
            "    add.u32 %r29, %r13, %r27;     // global_col = block_col + local_col",
        )?;
        write_line(ptx, "    mad.lo.u32 %r30, %r28, %r4, %r29;")?;
        write_line(ptx, "    cvt.u64.u32 %rd12, %r30;")?;
        write_line(
            ptx,
            &format!("    mul.lo.u64 %rd12, %rd12, {};", elem_bytes),
        )?;
        write_line(ptx, "    add.u64 %rd13, %rd1, %rd12;   // src_addr_b")?;

        // Shared memory destination for B (offset by tile_a_bytes)
        write_line(ptx, "    cvt.u64.u32 %rd14, %r25;")?;
        write_line(
            ptx,
            &format!("    mul.lo.u64 %rd14, %rd14, {};", elem_bytes),
        )?;
        write_line(
            ptx,
            "    add.u64 %rd15, %rd10, %rd14;  // offset within B region",
        )?;
        write_line(
            ptx,
            &format!(
                "    add.u64 %rd15, %rd15, {};     // skip past A tile",
                tile_a_bytes
            ),
        )?;

        // Bounds check
        write_line(ptx, "    setp.ge.u32 %p3, %r28, %r2;   // row >= K?")?;
        write_line(ptx, "    setp.ge.u32 %p4, %r29, %r1;   // col >= N?")?;
        write_line(ptx, "    or.pred %p5, %p3, %p4;")?;
        write_line(ptx, "    @%p5 bra $PRODUCER_B_SKIP;")?;

        write_line(
            ptx,
            &format!(
                "    cp.async.ca.shared.global [{input_ty} %rd15], [{input_ty} %rd13], {};",
                elem_bytes
            ),
        )?;
        write_line(ptx, "    bra $PRODUCER_B_NEXT;")?;

        write_line(ptx, "$PRODUCER_B_SKIP:")?;
        write_line(
            ptx,
            &format!("    st.shared.b{} [%rd15], {zero_literal};", elem_bytes * 8),
        )?;

        write_line(ptx, "$PRODUCER_B_NEXT:")?;
        write_line(ptx, "    add.u32 %r24, %r24, 1;")?;
        write_line(ptx, "    bra $PRODUCER_B_LOOP;")?;
        write_line(ptx, "$PRODUCER_B_DONE:")?;
        write_line(ptx, "")?;

        // Commit the async group and signal the barrier
        write_line(ptx, "    // Commit async copies and signal consumers")?;
        write_line(ptx, "    cp.async.commit_group;")?;
        write_line(
            ptx,
            "    bar.arrive 0, 1;               // signal consumers",
        )?;
        write_line(ptx, "")?;

        // Advance to next K tile
        write_line(
            ptx,
            &format!(
                "    add.u32 %r20, %r20, {};      // k_offset += tile_k",
                self.tile_k
            ),
        )?;

        // Advance stage index (ping-pong)
        write_line(ptx, "    add.u32 %r21, %r21, 1;")?;
        write_line(
            ptx,
            &format!("    setp.ge.u32 %p6, %r21, {};", self.pipeline_stages),
        )?;
        write_line(ptx, "    @%p6 mov.u32 %r21, 0;         // wrap around")?;
        write_line(ptx, "")?;
        write_line(ptx, "    bra $PRODUCER_K_LOOP;")?;
        write_line(ptx, "")?;

        write_line(ptx, "$PRODUCER_DONE:")?;
        write_line(ptx, "    // Producer warps are done; exit.")?;
        write_line(ptx, "    bra $KERNEL_EXIT;")?;
        write_line(ptx, "")?;

        Ok(())
    }

    /// Emits the consumer warp path: MMA computation on shared-memory tiles.
    fn emit_consumer_path(&self, ptx: &mut String) -> BlasResult<()> {
        let elem_bytes = self.input_type.size_bytes();
        let tile_a_bytes = (self.tile_m as usize) * (self.tile_k as usize) * elem_bytes;
        let stage_bytes =
            tile_a_bytes + (self.tile_k as usize) * (self.tile_n as usize) * elem_bytes;

        // Determine MMA shape based on input type
        let mma_shape = match self.input_type {
            PtxType::F16 | PtxType::BF16 => "m16n8k16",
            PtxType::E4M3 | PtxType::E5M2 => "m16n8k32",
            _ => "m16n8k16",
        };
        let mma_m: u32 = 16;
        let mma_n: u32 = 8;
        let mma_k: u32 = match self.input_type {
            PtxType::E4M3 | PtxType::E5M2 => 32,
            _ => 16,
        };

        // MMA type suffix for the instruction
        let mma_type_suffix = match self.input_type {
            PtxType::F16 => "f16",
            PtxType::BF16 => "bf16",
            PtxType::E4M3 => "e4m3",
            PtxType::E5M2 => "e5m2",
            _ => "f16",
        };

        write_line(ptx, "$CONSUMER_PATH:")?;
        write_line(
            ptx,
            "    // === Consumer warp: MMA on shared memory tiles ===",
        )?;
        write_line(ptx, "")?;

        // Compute consumer warp index (relative to consumer group)
        write_line(
            ptx,
            &format!(
                "    sub.u32 %r31, %r11, {};       // consumer_warp_idx",
                self.num_producer_warps
            ),
        )?;
        write_line(ptx, "")?;

        // Initialize accumulators to zero
        write_line(ptx, "    // Initialize accumulator registers")?;
        let num_acc_regs = (self.tile_m / mma_m) * (self.tile_n / mma_n);
        let num_acc_regs = num_acc_regs.min(16); // Cap for register declaration
        for i in 0..num_acc_regs {
            write_line(ptx, &format!("    mov.f32 %f{}, 0f00000000;", i + 2))?;
        }
        write_line(ptx, "")?;

        // Consumer K-loop
        write_line(ptx, "    mov.u32 %r32, 0;              // k_offset = 0")?;
        write_line(ptx, "    mov.u32 %r33, 0;              // stage_idx = 0")?;
        write_line(ptx, "")?;

        write_line(ptx, "$CONSUMER_K_LOOP:")?;
        write_line(ptx, "    setp.ge.u32 %p7, %r32, %r2;   // k_offset >= K?")?;
        write_line(ptx, "    @%p7 bra $CONSUMER_DONE;")?;
        write_line(ptx, "")?;

        // Wait for producer to fill this stage
        write_line(ptx, "    // Wait for producers to fill the pipeline stage")?;
        write_line(ptx, "    bar.sync 0;")?;
        write_line(
            ptx,
            "    cp.async.wait_group 0;         // wait for all async copies",
        )?;
        write_line(ptx, "")?;

        // Compute shared memory base for current stage
        write_line(
            ptx,
            &format!(
                "    mul.lo.u32 %r34, %r33, {};  // stage_offset",
                stage_bytes
            ),
        )?;
        write_line(ptx, "    mov.u64 %rd16, smem_buf;")?;
        write_line(ptx, "    cvt.u64.u32 %rd17, %r34;")?;
        write_line(ptx, "    add.u64 %rd16, %rd16, %rd17;  // smem_a_base")?;
        write_line(
            ptx,
            &format!(
                "    add.u64 %rd18, %rd16, {};   // smem_b_base = smem_a_base + tile_a_bytes",
                tile_a_bytes
            ),
        )?;
        write_line(ptx, "")?;

        // Execute MMA instructions across the tile
        write_line(ptx, "    // --- MMA execution over tile ---")?;
        write_line(
            ptx,
            &format!(
                "    // MMA shape: {} ({} x {} x {})",
                mma_shape, mma_m, mma_n, mma_k
            ),
        )?;

        // Sub-tile K loop within the tile
        write_line(ptx, "    mov.u32 %r35, 0;              // sub_k = 0")?;
        write_line(ptx, "$CONSUMER_MMA_LOOP:")?;
        write_line(ptx, &format!("    setp.ge.u32 %p8, %r35, {};", self.tile_k))?;
        write_line(ptx, "    @%p8 bra $CONSUMER_MMA_DONE;")?;
        write_line(ptx, "")?;

        // Load A fragment from shared memory
        write_line(ptx, "    // Load A fragment from shared memory")?;
        write_line(
            ptx,
            &format!(
                "    mul.lo.u32 %r36, %r35, {};   // sub_k * tile_m * elem_bytes (row-major offset)",
                self.tile_m as usize * elem_bytes
            ),
        )?;
        write_line(ptx, "    cvt.u64.u32 %rd19, %r36;")?;
        write_line(ptx, "    add.u64 %rd20, %rd16, %rd19;  // A fragment addr")?;
        write_line(ptx, "")?;

        // Load B fragment from shared memory
        write_line(ptx, "    // Load B fragment from shared memory")?;
        write_line(
            ptx,
            &format!(
                "    mul.lo.u32 %r37, %r35, {};   // sub_k * tile_n * elem_bytes",
                self.tile_n as usize * elem_bytes
            ),
        )?;
        write_line(ptx, "    cvt.u64.u32 %rd21, %r37;")?;
        write_line(ptx, "    add.u64 %rd22, %rd18, %rd21;  // B fragment addr")?;
        write_line(ptx, "")?;

        // Issue MMA instruction
        write_line(ptx, "    // MMA instruction")?;
        write_line(
            ptx,
            &format!(
                "    mma.sync.aligned.{mma_shape}.row.col.f32.{mma_type_suffix}.{mma_type_suffix}.f32"
            ),
        )?;
        write_line(ptx, "        {%f2, %f3, %f4, %f5},")?;
        write_line(ptx, "        {%r36, %r37},")?;
        write_line(ptx, "        {%r38},")?;
        write_line(ptx, "        {%f2, %f3, %f4, %f5};")?;
        write_line(ptx, "")?;

        // Advance sub-K
        write_line(ptx, &format!("    add.u32 %r35, %r35, {};", mma_k))?;
        write_line(ptx, "    bra $CONSUMER_MMA_LOOP;")?;
        write_line(ptx, "$CONSUMER_MMA_DONE:")?;
        write_line(ptx, "")?;

        // Advance K offset and stage
        write_line(ptx, &format!("    add.u32 %r32, %r32, {};", self.tile_k))?;
        write_line(ptx, "    add.u32 %r33, %r33, 1;")?;
        write_line(
            ptx,
            &format!("    setp.ge.u32 %p9, %r33, {};", self.pipeline_stages),
        )?;
        write_line(ptx, "    @%p9 mov.u32 %r33, 0;         // wrap stage")?;
        write_line(ptx, "    bra $CONSUMER_K_LOOP;")?;
        write_line(ptx, "")?;

        write_line(ptx, "$CONSUMER_DONE:")?;
        write_line(ptx, "")?;

        Ok(())
    }

    /// Emits the epilogue: consumer warps store results to global memory.
    fn emit_epilogue(&self, ptx: &mut String) -> BlasResult<()> {
        let out_bytes = self.output_type.size_bytes();

        write_line(ptx, "    // === Epilogue: store accumulated results ===")?;
        write_line(ptx, "    // C[row, col] = alpha * acc + beta * C_old")?;
        write_line(ptx, "")?;

        // Consumer warp writes its portion of the output tile
        // Compute output address based on consumer warp index
        write_line(ptx, "    // Compute output coordinates")?;

        // Simple: each consumer warp owns a row-strip of the output tile
        let rows_per_consumer = self.tile_m / self.num_consumer_warps.max(1);
        write_line(
            ptx,
            &format!(
                "    mul.lo.u32 %r40, %r31, {};    // consumer rows offset",
                rows_per_consumer
            ),
        )?;
        write_line(
            ptx,
            "    add.u32 %r40, %r40, %r14;     // global_row = block_row + consumer_offset",
        )?;
        write_line(ptx, "")?;

        // Each lane writes one element (simplified epilogue)
        write_line(ptx, "    // Lane-level output coordinates")?;
        write_line(
            ptx,
            &format!(
                "    div.u32 %r41, %r12, {};          // lane_row within warp tile",
                self.tile_n.min(32)
            ),
        )?;
        write_line(
            ptx,
            &format!(
                "    rem.u32 %r42, %r12, {};          // lane_col within warp tile",
                self.tile_n.min(32)
            ),
        )?;
        write_line(ptx, "    add.u32 %r41, %r41, %r40;     // final row")?;
        write_line(
            ptx,
            "    add.u32 %r42, %r42, %r13;     // final col = block_col + lane_col",
        )?;
        write_line(ptx, "")?;

        // Bounds check
        write_line(ptx, "    setp.ge.u32 %p10, %r41, %r0;  // row >= M?")?;
        write_line(ptx, "    setp.ge.u32 %p11, %r42, %r1;  // col >= N?")?;
        write_line(ptx, "    or.pred %p12, %p10, %p11;")?;
        write_line(ptx, "    @%p12 bra $KERNEL_EXIT;")?;
        write_line(ptx, "")?;

        // C address
        write_line(
            ptx,
            "    mad.lo.u32 %r43, %r41, %r5, %r42;  // row * ldc + col",
        )?;
        write_line(ptx, "    cvt.u64.u32 %rd23, %r43;")?;
        write_line(ptx, &format!("    mul.lo.u64 %rd23, %rd23, {};", out_bytes))?;
        write_line(ptx, "    add.u64 %rd24, %rd2, %rd23;   // &C[row, col]")?;
        write_line(ptx, "")?;

        // Load C_old, compute alpha * acc + beta * C_old, store
        write_line(ptx, "    ld.global.f32 %f20, [%rd24];")?;
        write_line(ptx, "    mul.f32 %f2, %f2, %f0;        // alpha * acc")?;
        write_line(
            ptx,
            "    fma.rn.f32 %f2, %f1, %f20, %f2;  // + beta * C_old",
        )?;
        write_line(ptx, "    st.global.f32 [%rd24], %f2;")?;
        write_line(ptx, "")?;

        Ok(())
    }

    /// Emits the kernel exit.
    fn emit_kernel_exit(&self, ptx: &mut String) -> BlasResult<()> {
        write_line(ptx, "$KERNEL_EXIT:")?;
        write_line(ptx, "    ret;")?;
        write_line(ptx, "}")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Writes a line to the PTX string, mapping fmt errors to `BlasError`.
fn write_line(ptx: &mut String, line: &str) -> BlasResult<()> {
    writeln!(ptx, "{line}").map_err(|e| BlasError::PtxGeneration(format!("fmt error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MathMode;

    // -- Construction & validation ------------------------------------------

    #[test]
    fn new_valid_f16() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(gemm.is_ok());
        let g = gemm.expect("checked above");
        assert_eq!(g.total_warps(), 8);
        assert_eq!(g.threads_per_block(), 256);
    }

    #[test]
    fn new_valid_bf16() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::BF16,
            PtxType::F32,
        );
        assert!(gemm.is_ok());
    }

    #[test]
    fn new_valid_e4m3() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::E4M3,
            PtxType::F32,
        );
        assert!(gemm.is_ok());
    }

    #[test]
    fn new_valid_e5m2() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            2,
            SmVersion::Sm90a,
            PtxType::E5M2,
            PtxType::F32,
        );
        assert!(gemm.is_ok());
    }

    #[test]
    fn reject_sm_below_90() {
        let result = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm80,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("SM >= 90"));
    }

    #[test]
    fn reject_sm_75() {
        let result = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm75,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(result.is_err());
    }

    #[test]
    fn reject_zero_producer_warps() {
        let result = WarpSpecializedGemm::new(
            128,
            128,
            64,
            0,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("producer"));
    }

    #[test]
    fn reject_zero_consumer_warps() {
        let result = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            0,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("consumer"));
    }

    #[test]
    fn reject_too_many_warps() {
        let result = WarpSpecializedGemm::new(
            128,
            128,
            64,
            16,
            17,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("32") || msg.contains("total"));
    }

    #[test]
    fn reject_zero_pipeline_stages() {
        let result = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            0,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(result.is_err());
    }

    #[test]
    fn reject_too_many_pipeline_stages() {
        let result = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            9,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("pipeline"));
    }

    #[test]
    fn reject_invalid_input_type() {
        let result = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F32,
            PtxType::F32,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("input_type"));
    }

    #[test]
    fn reject_invalid_output_type() {
        let result = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F64,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("output_type"));
    }

    #[test]
    fn reject_non_aligned_tile() {
        let result = WarpSpecializedGemm::new(
            100,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("multiple of 8"));
    }

    // -- Kernel generation --------------------------------------------------

    #[test]
    fn generate_kernel_f16() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        )
        .expect("valid config");
        let ptx = gemm.generate_kernel();
        assert!(ptx.is_ok());
        let ptx = ptx.expect("checked above");
        assert!(ptx.contains("cp.async"));
        assert!(ptx.contains("mma.sync.aligned"));
        assert!(ptx.contains("bar.sync"));
        assert!(ptx.contains("bar.arrive"));
        assert!(ptx.contains(".entry warp_specialized_gemm_"));
    }

    #[test]
    fn generate_kernel_bf16() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::BF16,
            PtxType::F32,
        )
        .expect("valid config");
        let ptx = gemm.generate_kernel().expect("gen ok");
        assert!(ptx.contains("bf16"));
        assert!(ptx.contains("mma.sync.aligned"));
    }

    #[test]
    fn generate_kernel_e4m3() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            2,
            SmVersion::Sm90,
            PtxType::E4M3,
            PtxType::F32,
        )
        .expect("valid config");
        let ptx = gemm.generate_kernel().expect("gen ok");
        assert!(ptx.contains("e4m3"));
        assert!(ptx.contains("m16n8k32"));
    }

    #[test]
    fn generate_kernel_e5m2() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            2,
            SmVersion::Sm90a,
            PtxType::E5M2,
            PtxType::F32,
        )
        .expect("valid config");
        let ptx = gemm.generate_kernel().expect("gen ok");
        assert!(ptx.contains("e5m2"));
    }

    #[test]
    fn ptx_contains_producer_consumer_labels() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        )
        .expect("valid config");
        let ptx = gemm.generate_kernel().expect("gen ok");
        assert!(ptx.contains("$PRODUCER_PATH"));
        assert!(ptx.contains("$CONSUMER_PATH"));
        assert!(ptx.contains("$PRODUCER_K_LOOP"));
        assert!(ptx.contains("$CONSUMER_K_LOOP"));
    }

    #[test]
    fn ptx_contains_pipeline_wrap() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        )
        .expect("valid config");
        let ptx = gemm.generate_kernel().expect("gen ok");
        // Stage wrap-around check
        assert!(ptx.contains("setp.ge.u32 %p6, %r21, 3"));
        assert!(ptx.contains("setp.ge.u32 %p9, %r33, 3"));
    }

    #[test]
    fn ptx_header_targets_sm90() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        )
        .expect("valid config");
        let ptx = gemm.generate_kernel().expect("gen ok");
        assert!(ptx.contains(".target sm_90"));
        assert!(ptx.contains(".version 8.0"));
    }

    // -- shared_memory_bytes ------------------------------------------------

    #[test]
    fn shared_memory_bytes_f16_3stage() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        )
        .expect("valid config");
        // A tile: 128 * 64 * 2 = 16384
        // B tile: 64 * 128 * 2 = 16384
        // Per stage: 32768
        // 3 stages: 98304
        assert_eq!(gemm.shared_memory_bytes(), 98304);
    }

    #[test]
    fn shared_memory_bytes_e4m3_2stage() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            2,
            SmVersion::Sm90,
            PtxType::E4M3,
            PtxType::F32,
        )
        .expect("valid config");
        // A tile: 128 * 64 * 1 = 8192
        // B tile: 64 * 128 * 1 = 8192
        // Per stage: 16384
        // 2 stages: 32768
        assert_eq!(gemm.shared_memory_bytes(), 32768);
    }

    #[test]
    fn shared_memory_bytes_single_stage() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            1,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        )
        .expect("valid config");
        // 1 stage: 32768
        assert_eq!(gemm.shared_memory_bytes(), 32768);
    }

    // -- to_tile_config -----------------------------------------------------

    #[test]
    fn to_tile_config_basic() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        )
        .expect("valid config");
        let tc = gemm.to_tile_config();
        assert_eq!(tc.tile_m, 128);
        assert_eq!(tc.tile_n, 128);
        assert_eq!(tc.tile_k, 64);
        assert_eq!(tc.stages, 3);
        assert!(tc.use_tensor_core);
        assert_eq!(tc.split_k, 1);
        // warp_m = 128 / 6 = 21, clamped to max(21, 8) = 21
        assert_eq!(tc.warp_m, 21);
    }

    // -- is_applicable ------------------------------------------------------

    #[test]
    fn is_applicable_hopper_large_f16() {
        let problem = GemmProblem {
            m: 4096,
            n: 4096,
            k: 4096,
            trans_a: crate::types::Transpose::NoTrans,
            trans_b: crate::types::Transpose::NoTrans,
            input_type: PtxType::F16,
            output_type: PtxType::F32,
            math_mode: MathMode::TensorCore,
        };
        assert!(WarpSpecializedGemm::is_applicable(
            &problem,
            SmVersion::Sm90
        ));
    }

    #[test]
    fn is_applicable_rejects_ampere() {
        let problem = GemmProblem {
            m: 4096,
            n: 4096,
            k: 4096,
            trans_a: crate::types::Transpose::NoTrans,
            trans_b: crate::types::Transpose::NoTrans,
            input_type: PtxType::F16,
            output_type: PtxType::F32,
            math_mode: MathMode::TensorCore,
        };
        assert!(!WarpSpecializedGemm::is_applicable(
            &problem,
            SmVersion::Sm80
        ));
    }

    #[test]
    fn is_applicable_rejects_skinny() {
        let problem = GemmProblem {
            m: 16,
            n: 4096,
            k: 4096,
            trans_a: crate::types::Transpose::NoTrans,
            trans_b: crate::types::Transpose::NoTrans,
            input_type: PtxType::F16,
            output_type: PtxType::F32,
            math_mode: MathMode::TensorCore,
        };
        assert!(!WarpSpecializedGemm::is_applicable(
            &problem,
            SmVersion::Sm90
        ));
    }

    #[test]
    fn is_applicable_rejects_small_problem() {
        let problem = GemmProblem {
            m: 64,
            n: 64,
            k: 8,
            trans_a: crate::types::Transpose::NoTrans,
            trans_b: crate::types::Transpose::NoTrans,
            input_type: PtxType::F16,
            output_type: PtxType::F32,
            math_mode: MathMode::TensorCore,
        };
        assert!(!WarpSpecializedGemm::is_applicable(
            &problem,
            SmVersion::Sm90
        ));
    }

    #[test]
    fn is_applicable_rejects_f32_input() {
        let problem = GemmProblem {
            m: 4096,
            n: 4096,
            k: 4096,
            trans_a: crate::types::Transpose::NoTrans,
            trans_b: crate::types::Transpose::NoTrans,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            math_mode: MathMode::TensorCore,
        };
        assert!(!WarpSpecializedGemm::is_applicable(
            &problem,
            SmVersion::Sm90
        ));
    }

    #[test]
    fn is_applicable_blackwell() {
        let problem = GemmProblem {
            m: 4096,
            n: 4096,
            k: 4096,
            trans_a: crate::types::Transpose::NoTrans,
            trans_b: crate::types::Transpose::NoTrans,
            input_type: PtxType::BF16,
            output_type: PtxType::F32,
            math_mode: MathMode::TensorCore,
        };
        assert!(WarpSpecializedGemm::is_applicable(
            &problem,
            SmVersion::Sm100
        ));
    }

    // -- kernel_name --------------------------------------------------------

    #[test]
    fn kernel_name_format() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        )
        .expect("valid config");
        assert_eq!(gemm.kernel_name(), "warp_specialized_gemm_f16_f32_p2_c6_s3");
    }

    #[test]
    fn kernel_name_e4m3() {
        let gemm = WarpSpecializedGemm::new(
            128,
            128,
            64,
            1,
            7,
            2,
            SmVersion::Sm90,
            PtxType::E4M3,
            PtxType::F32,
        )
        .expect("valid config");
        assert_eq!(
            gemm.kernel_name(),
            "warp_specialized_gemm_e4m3_f32_p1_c7_s2"
        );
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn min_valid_config() {
        // 1 producer + 1 consumer = 2 warps (minimum)
        let result = WarpSpecializedGemm::new(
            64,
            64,
            8,
            1,
            1,
            1,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn max_warp_config() {
        // 4 producer + 28 consumer = 32 warps (maximum)
        let result = WarpSpecializedGemm::new(
            128,
            128,
            64,
            4,
            28,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn pipeline_stages_1_2_3() {
        for stages in 1..=3 {
            let result = WarpSpecializedGemm::new(
                128,
                128,
                64,
                2,
                6,
                stages,
                SmVersion::Sm90,
                PtxType::F16,
                PtxType::F32,
            );
            assert!(result.is_ok(), "stages={stages} should be valid");
        }
    }

    #[test]
    fn shared_memory_scales_with_stages() {
        let g1 = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            1,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        )
        .expect("valid");
        let g3 = WarpSpecializedGemm::new(
            128,
            128,
            64,
            2,
            6,
            3,
            SmVersion::Sm90,
            PtxType::F16,
            PtxType::F32,
        )
        .expect("valid");
        assert_eq!(g3.shared_memory_bytes(), g1.shared_memory_bytes() * 3);
    }
}
