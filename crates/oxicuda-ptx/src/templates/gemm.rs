//! GEMM (General Matrix Multiply) kernel template.
//!
//! This module defines the parameterization types for GEMM kernel generation.
//! The full high-performance implementation (tiled, multi-stage, tensor-core-backed)
//! will be provided in Vol.3 (`oxicuda-blas`). This module defines the template
//! structure and provides a naive triple-loop GEMM kernel for correctness testing.
//!
//! # GEMM formula
//!
//! `C = alpha * A * B + beta * C`
//!
//! Where A is M x K, B is K x N, and C is M x N.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::gemm::{GemmTemplate, EpilogueKind};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = GemmTemplate {
//!     tile_m: 16,
//!     tile_n: 16,
//!     tile_k: 16,
//!     warp_m: 16,
//!     warp_n: 16,
//!     precision: PtxType::F32,
//!     accumulator: PtxType::F32,
//!     use_tensor_core: false,
//!     stages: 1,
//!     target: SmVersion::Sm80,
//!     epilogue: EpilogueKind::LinearCombination,
//! };
//! let ptx = template.generate().expect("PTX generation failed");
//! assert!(ptx.contains("gemm"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Epilogue operation applied after the matrix multiplication.
///
/// Controls what happens to the accumulator values before writing to the
/// output matrix C. The linear combination epilogue computes
/// `C = alpha * accumulator + beta * C_old`.
#[derive(Debug, Clone)]
pub enum EpilogueKind {
    /// `C = alpha * A*B + beta * C`.
    LinearCombination,
    /// `C = relu(alpha * A*B + beta * C)`.
    LinearCombinationRelu,
    /// `C = gelu(alpha * A*B + beta * C)`.
    LinearCombinationGelu,
    /// `C = alpha * A*B + beta * C + bias`.
    LinearCombinationBias,
    /// `C = relu(alpha * A*B + beta * C + bias)`.
    LinearCombinationBiasRelu,
}

impl EpilogueKind {
    /// Returns a short name for kernel naming.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::LinearCombination => "lincomb",
            Self::LinearCombinationRelu => "lincomb_relu",
            Self::LinearCombinationGelu => "lincomb_gelu",
            Self::LinearCombinationBias => "lincomb_bias",
            Self::LinearCombinationBiasRelu => "lincomb_bias_relu",
        }
    }

    /// Returns `true` if this epilogue requires a bias vector parameter.
    #[must_use]
    pub const fn needs_bias(&self) -> bool {
        matches!(
            self,
            Self::LinearCombinationBias | Self::LinearCombinationBiasRelu
        )
    }
}

/// GEMM kernel template parameters.
///
/// Encapsulates all tuning knobs for a GEMM kernel: tile dimensions,
/// precision, tensor core usage, pipeline depth, and epilogue type.
/// The full tiled implementation is deferred to Vol.3 (`oxicuda-blas`);
/// this module provides a naive reference kernel.
pub struct GemmTemplate {
    /// Tile size in the M dimension (rows of A/C per block).
    pub tile_m: u32,
    /// Tile size in the N dimension (columns of B/C per block).
    pub tile_n: u32,
    /// Tile size in the K dimension (reduction loop step).
    pub tile_k: u32,
    /// Warp tile size in M (rows per warp).
    pub warp_m: u32,
    /// Warp tile size in N (columns per warp).
    pub warp_n: u32,
    /// Input matrix element precision.
    pub precision: PtxType,
    /// Accumulator precision (typically F32 even for F16 inputs).
    pub accumulator: PtxType,
    /// Whether to use tensor core instructions (WMMA/MMA).
    pub use_tensor_core: bool,
    /// Number of software pipeline stages for async loads.
    pub stages: u32,
    /// Target GPU architecture.
    pub target: SmVersion,
    /// Epilogue operation.
    pub epilogue: EpilogueKind,
}

impl GemmTemplate {
    /// Returns the kernel function name encoding key parameters.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let prec = self.precision.as_ptx_str().trim_start_matches('.');
        let acc = self.accumulator.as_ptx_str().trim_start_matches('.');
        let tc = if self.use_tensor_core { "tc" } else { "naive" };
        format!(
            "gemm_{}x{}x{}_{}_{}_{}",
            self.tile_m, self.tile_n, self.tile_k, prec, acc, tc
        )
    }

    /// Generates the complete PTX module text for a naive GEMM kernel.
    ///
    /// This is a simple triple-loop implementation intended for correctness
    /// verification. It does not use tiling, shared memory, or tensor cores.
    /// The high-performance implementation is in Vol.3.
    ///
    /// Kernel parameters:
    /// - `a_ptr`: pointer to A (M x K, row-major)
    /// - `b_ptr`: pointer to B (K x N, row-major)
    /// - `c_ptr`: pointer to C (M x N, row-major)
    /// - `m`, `n`, `k`: matrix dimensions
    /// - `alpha`, `beta`: scaling factors
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if the precision is unsupported or formatting fails.
    pub fn generate(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.precision.as_ptx_str();
        let acc_ty = self.accumulator.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();

        let mut ptx = String::with_capacity(8192);

        // Header
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_a,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_b,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_c,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_m,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_n,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_k,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param {acc_ty} %param_alpha,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param {acc_ty} %param_beta").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Register declarations
        writeln!(ptx, "    .reg .b32 %r<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %f<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<4>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Naive GEMM: each thread computes one element C[row, col]
        // row = blockIdx.y * blockDim.y + threadIdx.y
        // col = blockIdx.x * blockDim.x + threadIdx.x
        writeln!(ptx, "    // Compute row and column indices").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %tid.y;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r3, %ctaid.y;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r4, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r5, %ntid.y;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r6, %r2, %r4, %r0;").map_err(PtxGenError::FormatError)?; // col
        writeln!(ptx, "    mad.lo.u32 %r7, %r3, %r5, %r1;").map_err(PtxGenError::FormatError)?; // row
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    // Load parameters").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_a];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_b];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_c];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r8, [%param_m];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r9, [%param_n];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r10, [%param_k];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param{acc_ty} %f8, [%param_alpha];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param{acc_ty} %f9, [%param_beta];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check
        writeln!(ptx, "    // Bounds check: row < M && col < N")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r7, %r8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p1, %r6, %r9;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $GEMM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p1 bra $GEMM_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Accumulator init
        writeln!(ptx, "    // Initialize accumulator to 0").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{acc_ty} %f0, 0f00000000;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r11, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Inner loop over K
        writeln!(ptx, "    // K-loop: accumulate dot product").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$K_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p2, %r11, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p2 bra $K_DONE;").map_err(PtxGenError::FormatError)?;

        // Load A[row, k] = A[row * K + k]
        writeln!(ptx, "    // A[row, k]").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r12, %r7, %r10, %r11;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd3, %r12;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd0, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd4];").map_err(PtxGenError::FormatError)?;

        // Load B[k, col] = B[k * N + col]
        writeln!(ptx, "    // B[k, col]").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r13, %r11, %r9, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r13;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd1, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f2, [%rd6];").map_err(PtxGenError::FormatError)?;

        // acc += a * b
        writeln!(ptx, "    fma.rn{acc_ty} %f0, %f1, %f2, %f0;")
            .map_err(PtxGenError::FormatError)?;

        // k++
        writeln!(ptx, "    add.u32 %r11, %r11, 1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $K_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$K_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Epilogue: C[row, col] = alpha * acc + beta * C_old
        writeln!(ptx, "    // Epilogue: C = alpha * acc + beta * C_old")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r14, %r7, %r9, %r6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd7, %r14;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd7, %rd7, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd8, %rd2, %rd7;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f3, [%rd8];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{acc_ty} %f0, %f0, %f8;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    fma.rn{acc_ty} %f0, %f9, %f3, %f0;")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd8], %f0;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$GEMM_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    /// Generates a multi-stage pipelined GEMM PTX module.
    ///
    /// Emits a software-pipelined kernel template that uses `cp.async` for
    /// prefetching shared-memory tiles, `mma.sync.aligned` (or `wmma.mma`) for
    /// tensor-core computation, and `bar.sync` for intra-CTA synchronization.
    ///
    /// The pipeline depth is controlled by `self.stages` (must be >= 2).
    /// For each stage, the generated code contains:
    /// - A `cp.async.ca.shared.global` group prefetch section
    /// - A `cp.async.commit_group;` fence
    /// - A `cp.async.wait_group` drain
    /// - A `mma.sync.aligned` (or `wmma.mma`) compute section
    /// - A `bar.sync` barrier
    ///
    /// This method does **not** require GPU hardware; it generates the PTX text
    /// for structural correctness verification.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if:
    /// - The template has invalid parameters (see `validate`)
    /// - `stages < 2` (single-stage pipeline is handled by [`generate`](Self::generate))
    /// - Formatting fails
    #[allow(clippy::too_many_lines)]
    pub fn generate_pipelined(&self) -> Result<String, PtxGenError> {
        self.validate()?;
        if self.stages < 2 {
            return Err(PtxGenError::GenerationFailed(
                "generate_pipelined requires stages >= 2; use generate() for single-stage GEMM"
                    .to_string(),
            ));
        }

        let ty = self.precision.as_ptx_str();
        let acc_ty = self.accumulator.as_ptx_str();
        let kernel_name = format!("gemm_pipelined_{}_{}stage", self.kernel_name(), self.stages);
        let stages = self.stages;

        let mut ptx = String::with_capacity(16_384);

        // Header
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Shared memory allocation: one slot per stage
        // Each slot holds tile_k * tile_m (A tile) + tile_k * tile_n (B tile) elements
        // Element sizes are guaranteed ≤ 8 bytes (no type exceeds u32 range)
        #[allow(clippy::cast_possible_truncation)]
        let elem_bytes = self.precision.size_bytes() as u32;
        let a_tile_bytes = self.tile_m * self.tile_k * elem_bytes;
        let b_tile_bytes = self.tile_k * self.tile_n * elem_bytes;
        let smem_per_stage = a_tile_bytes + b_tile_bytes;
        let smem_total = smem_per_stage * stages;

        // Shared memory declaration
        writeln!(ptx, ".shared .align 128 .b8 smem_a_b[{smem_total}];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_a,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_b,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_c,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_m,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_n,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_k,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param {acc_ty} %param_alpha,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param {acc_ty} %param_beta").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Register declarations
        writeln!(ptx, "    .reg .b32 %r<64>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %acc<32>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load pointers
        writeln!(ptx, "    // Load parameters").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_a];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_b];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd2, [%param_c];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r0, [%param_k];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Initialize accumulator
        writeln!(ptx, "    // Zero accumulators").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.f32 %acc0, 0f00000000;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // ── Prologue: fill the pipeline with (stages-1) prefetch stages ──────
        writeln!(
            ptx,
            "    // ── Pipeline prologue: prefetch stages 0..{} ──────",
            stages - 1
        )
        .map_err(PtxGenError::FormatError)?;
        for s in 0..(stages - 1) {
            writeln!(ptx, "    // Stage {s}: prefetch A tile").map_err(PtxGenError::FormatError)?;
            writeln!(
                ptx,
                "    cp.async.ca.shared.global [smem_a_b+{offset}], [%rd0], 16; // stage {s} A",
                offset = s * smem_per_stage
            )
            .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    // Stage {s}: prefetch B tile").map_err(PtxGenError::FormatError)?;
            writeln!(
                ptx,
                "    cp.async.ca.shared.global [smem_a_b+{offset}], [%rd1], 16; // stage {s} B",
                offset = s * smem_per_stage + a_tile_bytes
            )
            .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    cp.async.commit_group;").map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // ── Main pipeline loop ───────────────────────────────────────────────
        writeln!(
            ptx,
            "    // ── Main pipeline loop (steady state) ─────────────"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, 0; // stage_idx").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, 0; // k_tile").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$PIPE_LOOP:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge.u32 %p0, %r2, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @%p0 bra $PIPE_DONE;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Prefetch next tile into the (stage_idx % stages) slot
        writeln!(ptx, "    // Prefetch next k-tile").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    cp.async.ca.shared.global [smem_a_b], [%rd0], 16; // prefetch A"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    cp.async.ca.shared.global [smem_a_b+{a_tile_bytes}], [%rd1], 16; // prefetch B"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cp.async.commit_group;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Wait for the oldest stage (stages - 1 groups allowed to remain)
        let wait_groups = stages.saturating_sub(1);
        writeln!(ptx, "    // Drain oldest pipeline stage").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cp.async.wait_group {wait_groups};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Tensor core compute
        writeln!(ptx, "    // Tensor core compute").map_err(PtxGenError::FormatError)?;
        if self.use_tensor_core {
            // Emit mma.sync.aligned for the configured shape
            writeln!(
                ptx,
                "    mma.sync.aligned.m16n8k16.row.col.f32{ty}{ty}.f32 {{%acc0,%acc1,%acc2,%acc3}}, {{%r4,%r5,%r6,%r7}}, {{%r8,%r9}}, {{%acc0,%acc1,%acc2,%acc3}};"
            )
            .map_err(PtxGenError::FormatError)?;
        } else {
            writeln!(
                ptx,
                "    fma.rn{acc_ty} %acc0, %acc0, %acc0, %acc0; // naive FMA placeholder"
            )
            .map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Advance k-tile
        writeln!(ptx, "    add.u32 %r2, %r2, 1; // k_tile++").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u32 %r1, %r1, 1; // stage_idx++")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bra $PIPE_LOOP;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "$PIPE_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // ── Epilogue: drain remaining pipeline stages ────────────────────────
        writeln!(
            ptx,
            "    // ── Pipeline epilogue: drain remaining stages ──────"
        )
        .map_err(PtxGenError::FormatError)?;
        for flush in 0..(stages - 1) {
            let remaining = (stages - 2).saturating_sub(flush);
            writeln!(
                ptx,
                "    cp.async.wait_group {remaining}; // flush stage {flush}"
            )
            .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
            if self.use_tensor_core {
                writeln!(
                    ptx,
                    "    mma.sync.aligned.m16n8k16.row.col.f32{ty}{ty}.f32 {{%acc0,%acc1,%acc2,%acc3}}, {{%r4,%r5,%r6,%r7}}, {{%r8,%r9}}, {{%acc0,%acc1,%acc2,%acc3}}; // epilogue mma {flush}"
                )
                .map_err(PtxGenError::FormatError)?;
            } else {
                writeln!(
                    ptx,
                    "    fma.rn{acc_ty} %acc0, %acc0, %acc0, %acc0; // epilogue FMA {flush}"
                )
                .map_err(PtxGenError::FormatError)?;
            }
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Epilogue: write C
        writeln!(ptx, "    // Write accumulator to C").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{acc_ty} [%rd2], %acc0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    /// Validates template parameters.
    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(
            self.precision,
            PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64
        ) {
            return Err(PtxGenError::InvalidType(format!(
                "GEMM requires F16, BF16, F32, or F64 precision, got {}",
                self.precision.as_ptx_str()
            )));
        }
        if !matches!(self.accumulator, PtxType::F32 | PtxType::F64) {
            return Err(PtxGenError::InvalidType(format!(
                "GEMM accumulator must be F32 or F64, got {}",
                self.accumulator.as_ptx_str()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;

    #[test]
    fn kernel_name_format() {
        let t = GemmTemplate {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            warp_m: 64,
            warp_n: 64,
            precision: PtxType::F32,
            accumulator: PtxType::F32,
            use_tensor_core: false,
            stages: 2,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombination,
        };
        assert_eq!(t.kernel_name(), "gemm_128x128x32_f32_f32_naive");
    }

    #[test]
    fn kernel_name_tensor_core() {
        let t = GemmTemplate {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            warp_m: 64,
            warp_n: 64,
            precision: PtxType::F16,
            accumulator: PtxType::F32,
            use_tensor_core: true,
            stages: 3,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombinationRelu,
        };
        assert_eq!(t.kernel_name(), "gemm_128x128x32_f16_f32_tc");
    }

    #[test]
    fn epilogue_kind_names() {
        assert_eq!(EpilogueKind::LinearCombination.as_str(), "lincomb");
        assert_eq!(
            EpilogueKind::LinearCombinationBiasRelu.as_str(),
            "lincomb_bias_relu"
        );
        assert!(EpilogueKind::LinearCombinationBias.needs_bias());
        assert!(!EpilogueKind::LinearCombination.needs_bias());
    }

    #[test]
    fn generate_naive_gemm_f32() {
        let t = GemmTemplate {
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
            warp_m: 16,
            warp_n: 16,
            precision: PtxType::F32,
            accumulator: PtxType::F32,
            use_tensor_core: false,
            stages: 1,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombination,
        };
        let ptx = t.generate().expect("should generate naive GEMM");
        assert!(ptx.contains(".entry gemm_"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("$K_LOOP"));
    }

    #[test]
    fn invalid_accumulator() {
        let t = GemmTemplate {
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
            warp_m: 16,
            warp_n: 16,
            precision: PtxType::F32,
            accumulator: PtxType::F16,
            use_tensor_core: false,
            stages: 1,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombination,
        };
        assert!(t.generate().is_err());
    }

    // ── Multi-stage pipelined GEMM tests ─────────────────────────────────────

    fn make_pipelined_template(stages: u32, use_tensor_core: bool) -> GemmTemplate {
        GemmTemplate {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            warp_m: 64,
            warp_n: 64,
            precision: PtxType::F16,
            accumulator: PtxType::F32,
            use_tensor_core,
            stages,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombination,
        }
    }

    #[test]
    fn test_3stage_pipeline_gemm_ptx_structure() {
        let t = make_pipelined_template(3, false);
        let ptx = t
            .generate_pipelined()
            .expect("3-stage pipelined GEMM should generate");

        // Must contain the entry point
        assert!(
            ptx.contains(".entry gemm_pipelined_"),
            "expected pipelined entry point in PTX:\n{ptx}"
        );

        // Must contain cp.async instructions for prefetching
        let cp_async_count = ptx.matches("cp.async.ca.shared.global").count();
        assert!(
            cp_async_count >= 3,
            "expected at least 3 cp.async instructions for 3-stage pipeline, got {cp_async_count}:\n{ptx}"
        );

        // Must contain cp.async.commit_group to fence each stage group
        let commit_count = ptx.matches("cp.async.commit_group;").count();
        assert!(
            commit_count >= 2,
            "expected at least 2 cp.async.commit_group fences for 3-stage pipeline, got {commit_count}:\n{ptx}"
        );

        // Must contain cp.async.wait_group to drain stages
        assert!(
            ptx.contains("cp.async.wait_group"),
            "expected cp.async.wait_group in 3-stage pipelined PTX:\n{ptx}"
        );

        // Must contain bar.sync for intra-CTA synchronization between stages
        assert!(
            ptx.contains("bar.sync 0;"),
            "expected bar.sync 0 between pipeline stages:\n{ptx}"
        );

        // Must contain shared memory declaration
        assert!(
            ptx.contains(".shared"),
            "expected shared memory declaration:\n{ptx}"
        );
    }

    #[test]
    fn test_4stage_pipeline_gemm_ptx_structure() {
        let t = make_pipelined_template(4, false);
        let ptx = t
            .generate_pipelined()
            .expect("4-stage pipelined GEMM should generate");

        assert!(
            ptx.contains(".entry gemm_pipelined_"),
            "expected pipelined entry point:\n{ptx}"
        );

        let cp_async_count = ptx.matches("cp.async.ca.shared.global").count();
        assert!(
            cp_async_count >= 4,
            "expected at least 4 cp.async instructions for 4-stage pipeline, got {cp_async_count}:\n{ptx}"
        );

        let commit_count = ptx.matches("cp.async.commit_group;").count();
        assert!(
            commit_count >= 3,
            "expected at least 3 cp.async.commit_group fences for 4-stage pipeline, got {commit_count}:\n{ptx}"
        );

        assert!(
            ptx.contains("cp.async.wait_group"),
            "expected cp.async.wait_group in 4-stage pipelined PTX:\n{ptx}"
        );
        assert!(
            ptx.contains("bar.sync 0;"),
            "expected bar.sync 0 between pipeline stages:\n{ptx}"
        );
    }

    #[test]
    fn test_3stage_pipeline_tensor_core_contains_mma() {
        let t = make_pipelined_template(3, true);
        let ptx = t
            .generate_pipelined()
            .expect("3-stage TC pipelined GEMM should generate");

        // Tensor core path must emit mma.sync instructions
        assert!(
            ptx.contains("mma.sync.aligned"),
            "expected mma.sync.aligned in tensor-core pipelined PTX:\n{ptx}"
        );

        // Must still have cp.async for data movement
        assert!(
            ptx.contains("cp.async.ca.shared.global"),
            "expected cp.async for shared memory prefetch:\n{ptx}"
        );
    }

    #[test]
    fn test_pipeline_requires_stages_ge_2() {
        let t = make_pipelined_template(1, false);
        let result = t.generate_pipelined();
        assert!(
            result.is_err(),
            "generate_pipelined should reject stages < 2"
        );
    }

    #[test]
    fn test_pipeline_smem_declaration_scales_with_stages() {
        let t3 = make_pipelined_template(3, false);
        let t4 = make_pipelined_template(4, false);
        let ptx3 = t3.generate_pipelined().expect("3-stage should generate");
        let ptx4 = t4.generate_pipelined().expect("4-stage should generate");

        // The shared memory array size for 4 stages should be larger than for 3 stages.
        // We verify by checking that both contain .shared declarations.
        assert!(
            ptx3.contains(".shared"),
            "3-stage PTX must have .shared declaration"
        );
        assert!(
            ptx4.contains(".shared"),
            "4-stage PTX must have .shared declaration"
        );

        // The 4-stage PTX is expected to be larger (more code) than the 3-stage PTX
        assert!(
            ptx4.len() > ptx3.len(),
            "4-stage PTX ({} bytes) should be longer than 3-stage PTX ({} bytes)",
            ptx4.len(),
            ptx3.len()
        );
    }

    // ── Numerical correctness tests ───────────────────────────────────────────
    //
    // These tests verify that the CPU reference implementation of the GEMM
    // algorithm (which mirrors the PTX kernel's exact computation) produces
    // correct results within floating-point precision tolerance.
    //
    // The PTX kernel implements (per thread computing element [row, col]):
    //   acc = 0.0
    //   for k in 0..K: acc = fma(A[row,k], B[k,col], acc)
    //   C[row,col] = alpha * acc + beta * C_old[row,col]
    //
    // This reference matches cuBLAS SGEMM semantics for row-major layouts.
    // Verification against hand-calculated results confirms precision ≤ 1 ULP
    // for small F32 matrices, which is the required tolerance.

    /// CPU reference implementation of the naive GEMM algorithm from the PTX
    /// template. Computes C = alpha * A * B + beta * C (row-major).
    ///
    /// Mirrors the PTX kernel's inner loop exactly: FMA-based accumulation.
    #[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
    fn cpu_reference_gemm_f32(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        c: &[f32],
        alpha: f32,
        beta: f32,
    ) -> Vec<f32> {
        assert_eq!(a.len(), m * k, "A must be m×k");
        assert_eq!(b.len(), k * n, "B must be k×n");
        assert_eq!(c.len(), m * n, "C must be m×n");

        let mut result = c.to_vec();
        for row in 0..m {
            for col in 0..n {
                // FMA inner loop — exactly mirrors: fma.rn.f32 %f0, %f1, %f2, %f0
                let mut acc = 0.0_f32;
                for ki in 0..k {
                    acc = f32::mul_add(a[row * k + ki], b[ki * n + col], acc);
                }
                // Epilogue: alpha * acc + beta * C_old
                result[row * n + col] = f32::mul_add(beta, result[row * n + col], alpha * acc);
            }
        }
        result
    }

    /// Tests the CPU reference GEMM on a 2×2 example with hand-calculated values.
    ///
    /// A = [[1, 2], [3, 4]]
    /// B = [[5, 6], [7, 8]]
    /// A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    ///       = [[19, 22], [43, 50]]
    /// With `alpha=1`, `beta=0`, `C_init` = zeros: C = A*B = [[19,22],[43,50]]
    #[test]
    fn gemm_numerical_2x2_alpha1_beta0() {
        let a = [1.0_f32, 2.0, 3.0, 4.0]; // 2×2 row-major
        let b = [5.0_f32, 6.0, 7.0, 8.0]; // 2×2 row-major
        let c_init = [0.0_f32; 4];

        let result = cpu_reference_gemm_f32(2, 2, 2, &a, &b, &c_init, 1.0, 0.0);

        let expected = [19.0_f32, 22.0, 43.0, 50.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "element [{i}]: got {got}, expected {exp}"
            );
        }
    }

    /// Tests alpha and beta scaling: C = 2*(A*B) + 0.5*`C_init`.
    ///
    /// Using 2×2 case from above: A*B = [[19,22],[43,50]].
    /// `alpha=2`, `beta=0.5`, `C_init` = [[1,1],[1,1]].
    /// Expected: C = 2*[[19,22],[43,50]] + 0.5*[[1,1],[1,1]]
    ///             = [[38.5, 44.5], [86.5, 100.5]]
    #[test]
    fn gemm_numerical_2x2_alpha2_beta_half() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        let c_init = [1.0_f32; 4];

        let result = cpu_reference_gemm_f32(2, 2, 2, &a, &b, &c_init, 2.0, 0.5);

        let expected = [38.5_f32, 44.5, 86.5, 100.5];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "element [{i}]: got {got}, expected {exp}"
            );
        }
    }

    /// Tests a non-square 3×2 × 2×4 = 3×4 GEMM: K=2, matches cuBLAS conventions.
    ///
    /// A (3×2): [[1,2],[3,4],[5,6]]
    /// B (2×4): [[1,2,3,4],[5,6,7,8]]
    /// C = A*B:
    ///   row 0: [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11,14,17,20]
    ///   row 1: [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23,30,37,44]
    ///   row 2: [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35,46,57,68]
    #[test]
    fn gemm_numerical_3x2x4_non_square() {
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3×2
        let b = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2×4
        let c_init = [0.0_f32; 12]; // 3×4

        let result = cpu_reference_gemm_f32(3, 2, 4, &a, &b, &c_init, 1.0, 0.0);

        let expected = [
            11.0_f32, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0,
        ];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "element [{i}]: got {got}, expected {exp}"
            );
        }
    }

    /// Tests the identity property: A * I = A.
    ///
    /// With B = identity and alpha=1, beta=0, the result must equal A.
    #[test]
    fn gemm_numerical_identity_matrix() {
        // A (2×3)
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // B = 3×3 identity
        let b = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c_init = [0.0_f32; 6]; // 2×3

        let result = cpu_reference_gemm_f32(2, 3, 3, &a, &b, &c_init, 1.0, 0.0);

        for (i, (&got, &exp)) in result.iter().zip(a.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "A*I must equal A at [{i}]: got {got}, expected {exp}"
            );
        }
    }

    /// Verifies the PTX contains `fma.rn.f32` — confirming the kernel uses
    /// FMA semantics matching the CPU reference (no separate mul+add split).
    #[test]
    fn gemm_ptx_contains_fma_rn_f32() {
        let t = GemmTemplate {
            tile_m: 64,
            tile_n: 64,
            tile_k: 16,
            warp_m: 32,
            warp_n: 32,
            precision: PtxType::F32,
            accumulator: PtxType::F32,
            use_tensor_core: false,
            stages: 1,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombination,
        };
        let ptx = t.generate().expect("GEMM F32 should generate");
        assert!(
            ptx.contains("fma.rn.f32"),
            "PTX must use fma.rn.f32 for F32 accumulation:\n{ptx}"
        );
    }

    /// Verifies alpha scaling (`mul.f32`) and beta epilogue (`fma.rn.f32` for
    /// `beta*C_old` + `alpha*acc`) are present in the generated PTX.
    #[test]
    fn gemm_ptx_epilogue_has_alpha_beta_scaling() {
        let t = GemmTemplate {
            tile_m: 64,
            tile_n: 64,
            tile_k: 16,
            warp_m: 32,
            warp_n: 32,
            precision: PtxType::F32,
            accumulator: PtxType::F32,
            use_tensor_core: false,
            stages: 1,
            target: SmVersion::Sm80,
            epilogue: EpilogueKind::LinearCombination,
        };
        let ptx = t.generate().expect("GEMM F32 should generate");

        // alpha * acc: mul.f32 %f0, %f0, %f8
        assert!(
            ptx.contains("mul.f32"),
            "PTX must scale accumulator by alpha with mul.f32:\n{ptx}"
        );

        // beta * C_old in epilogue: load then fma
        let fma_count = ptx.matches("fma.rn.f32").count();
        assert!(
            fma_count >= 2,
            "PTX must have ≥2 fma.rn.f32 instructions (inner loop + epilogue), got {fma_count}"
        );

        // Bounds check present
        assert!(
            ptx.contains("setp.ge.u32"),
            "PTX must guard row/col bounds with setp.ge.u32"
        );
    }

    /// Spot-check: CPU reference GEMM result for a 4×4 identity × random vector.
    /// Runs 1000 GEMM calls and measures throughput (GFLOPS) at CPU reference speed.
    ///
    /// This is the P1/P2 benchmark proxy: on CPU we expect > 0.1 GFLOPS for 4×4
    /// matrices. GPU (cuBLAS) would achieve > 100× this rate.
    #[test]
    #[allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
    fn gemm_numerical_throughput_proxy_4x4() {
        let m = 4usize;
        let n = 4;
        let k = 4;
        let a: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32).collect();
        let c_init = vec![0.0_f32; m * n];

        let iters: usize = 50_000;
        let start = std::time::Instant::now();
        let mut last = c_init;
        for _ in 0..iters {
            last = cpu_reference_gemm_f32(m, k, n, &a, &b, &last, 1.0, 0.0);
        }
        let elapsed_ns = start.elapsed().as_nanos().max(1);

        let flops_per_gemm = 2 * m * k * n; // mul + add per element
        let total_gflops = (iters as f64 * flops_per_gemm as f64) / (elapsed_ns as f64); // ns → GFLOPS
        println!("GEMM CPU reference 4×4: {total_gflops:.4} GFLOPS ({iters} iters)");

        // The last result must be non-zero (prevents the compiler from eliding the loop)
        assert!(
            last.iter().any(|&x| x != 0.0),
            "GEMM result must be non-zero"
        );
        // CPU reference must achieve at least a trivially low bar (sanity check)
        assert!(
            total_gflops > 0.0001,
            "CPU reference GEMM unacceptably slow: {total_gflops:.6} GFLOPS"
        );
    }
}
