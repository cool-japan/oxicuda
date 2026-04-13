//! Cooperative GEMM across CTAs.
//!
//! Unlike Split-K (which splits K across independent CTAs with atomic
//! accumulation), cooperative GEMM uses cluster-level synchronisation (SM 90+)
//! or explicit multi-phase reduction for inter-CTA communication. Multiple
//! CTAs collaborate on the same output tile, each computing a partial result
//! over a slice of the K dimension.
//!
//! # Reduction strategies
//!
//! - **`ClusterSharedMemory`** (SM 90+): CTAs in the same cluster share
//!   distributed shared memory. After the partial GEMM, a `barrier.cluster`
//!   synchronises the cluster and partial results are reduced via
//!   `ld.shared::cluster`.
//! - **`TwoPhase`**: Each CTA writes its partial C tile to a global workspace
//!   buffer indexed by `(m_tile, n_tile, cta_id)`. A second lightweight
//!   kernel reduces along the `cta_id` dimension.
//! - **`AtomicAccumulate`**: Each CTA atomically adds its partial result to
//!   the output matrix. Simple but suffers from contention at high CTA counts.
//! - **`Auto`**: Selects the best strategy based on SM version and problem
//!   size.

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::arch::SmVersion;

use crate::error::{BlasError, BlasResult};

// ---------------------------------------------------------------------------
// Precision
// ---------------------------------------------------------------------------

/// Precision modes for cooperative GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoopPrecision {
    /// IEEE 754 half-precision (16-bit).
    F16,
    /// Brain floating-point (16-bit).
    BF16,
    /// TensorFloat-32 (19-bit mantissa, used on Ampere+).
    TF32,
    /// IEEE 754 single-precision (32-bit).
    F32,
    /// IEEE 754 double-precision (64-bit).
    F64,
}

impl CoopPrecision {
    /// PTX type string for loads/stores of the accumulator.
    fn acc_ptx_str(self) -> &'static str {
        match self {
            Self::F16 | Self::BF16 | Self::TF32 | Self::F32 => ".f32",
            Self::F64 => ".f64",
        }
    }

    /// PTX type string for the input elements.
    fn input_ptx_str(self) -> &'static str {
        match self {
            Self::F16 => ".f16",
            Self::BF16 => ".bf16",
            Self::TF32 | Self::F32 => ".f32",
            Self::F64 => ".f64",
        }
    }

    /// Byte size of the accumulator element.
    fn acc_bytes(self) -> usize {
        match self {
            Self::F16 | Self::BF16 | Self::TF32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Byte size of the input element.
    fn input_bytes(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::TF32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Reduction strategy
// ---------------------------------------------------------------------------

/// How cooperating CTAs reduce their partial results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoopReductionStrategy {
    /// Hierarchical reduction within a cluster using distributed shared
    /// memory. Requires SM >= 90.
    ClusterSharedMemory,
    /// Two-phase: partial results go to a global workspace, then a second
    /// kernel reduces them.
    TwoPhase,
    /// Each CTA atomically accumulates its partial result into the output.
    AtomicAccumulate,
    /// Automatically select the best strategy based on SM version and
    /// problem characteristics.
    Auto,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a cooperative GEMM operation.
#[derive(Debug, Clone)]
pub struct CooperativeGemmConfig {
    /// Rows of the output matrix C.
    pub m: usize,
    /// Columns of the output matrix C.
    pub n: usize,
    /// Shared (inner) dimension.
    pub k: usize,
    /// Target SM architecture.
    pub sm_version: SmVersion,
    /// Number of CTAs that cooperate on each output tile (2, 4, 8, or 16).
    pub cta_cluster_size: usize,
    /// Reduction strategy.
    pub reduction_strategy: CoopReductionStrategy,
    /// Element precision.
    pub precision: CoopPrecision,
}

impl CooperativeGemmConfig {
    /// Validates this configuration.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::InvalidArgument`] when:
    /// - `cta_cluster_size` is not a power of two in `[2, 16]`.
    /// - The SM version is too old for the requested reduction strategy.
    /// - Matrix dimensions are zero.
    pub fn validate(&self) -> BlasResult<()> {
        // Dimensions must be positive.
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(BlasError::InvalidArgument(
                "cooperative GEMM requires non-zero M, N, K".into(),
            ));
        }

        // Cluster size must be a power of two in [2, 16].
        if !matches!(self.cta_cluster_size, 2 | 4 | 8 | 16) {
            return Err(BlasError::InvalidArgument(format!(
                "cta_cluster_size must be 2, 4, 8, or 16, got {}",
                self.cta_cluster_size,
            )));
        }

        // SM requirements per strategy.
        match self.reduction_strategy {
            CoopReductionStrategy::ClusterSharedMemory => {
                if self.sm_version < SmVersion::Sm90 {
                    return Err(BlasError::UnsupportedOperation(
                        "ClusterSharedMemory reduction requires SM >= 90".into(),
                    ));
                }
            }
            CoopReductionStrategy::AtomicAccumulate => {
                if self.sm_version < SmVersion::Sm80 {
                    return Err(BlasError::UnsupportedOperation(
                        "AtomicAccumulate reduction requires SM >= 80".into(),
                    ));
                }
            }
            CoopReductionStrategy::TwoPhase | CoopReductionStrategy::Auto => {}
        }

        Ok(())
    }

    /// Resolves [`CoopReductionStrategy::Auto`] to a concrete strategy.
    fn resolve_strategy(&self) -> CoopReductionStrategy {
        match self.reduction_strategy {
            CoopReductionStrategy::Auto => {
                if self.sm_version >= SmVersion::Sm90 && self.cta_cluster_size <= 8 {
                    CoopReductionStrategy::ClusterSharedMemory
                } else if self.k >= 2048 {
                    CoopReductionStrategy::TwoPhase
                } else {
                    CoopReductionStrategy::AtomicAccumulate
                }
            }
            other => other,
        }
    }
}

// ---------------------------------------------------------------------------
// Work partitioning
// ---------------------------------------------------------------------------

/// Describes how work is distributed across cooperating CTAs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoopWorkPartition {
    /// K elements each CTA processes (except possibly the last).
    pub k_per_cta: usize,
    /// Extra K elements assigned to the last CTA in a group.
    pub k_remainder: usize,
    /// Output tile height (rows per CTA group).
    pub output_tile_m: usize,
    /// Output tile width (columns per CTA group).
    pub output_tile_n: usize,
    /// Number of independent CTA groups (`ceil(M/tile_m) * ceil(N/tile_n)`).
    pub num_cta_groups: usize,
    /// CTAs cooperating within each group (= cluster size).
    pub ctas_per_group: usize,
    /// Total CTA count (`num_cta_groups * ctas_per_group`).
    pub total_ctas: usize,
}

/// Determines whether cooperative GEMM is likely beneficial for the given
/// problem dimensions.
///
/// Cooperative GEMM adds synchronisation and reduction overhead, so it only
/// pays off when the K dimension is large enough to amortise that cost and
/// the output tile is not too small.
pub fn is_cooperative_beneficial(m: usize, n: usize, k: usize, sm_version: SmVersion) -> bool {
    // K must be large enough to split meaningfully.
    if k < 512 {
        return false;
    }

    // Output tile must have enough compute to justify coordination overhead.
    let output_elements = m.saturating_mul(n);
    if output_elements < 64 * 64 {
        return false;
    }

    // Compute density: ratio of FLOPs to output elements.  Each output
    // element requires 2*K FLOPs (multiply-add), so density = 2*K.
    // Cooperative GEMM is worthwhile when the reduction dimension dominates.
    let flops_per_output = 2 * k;

    // On Hopper+ with cluster support, a lower threshold suffices.
    let threshold = if sm_version >= SmVersion::Sm90 {
        1024
    } else {
        2048
    };

    flops_per_output >= threshold
}

/// Computes the work partition for a cooperative GEMM configuration.
pub fn partition_work(config: &CooperativeGemmConfig) -> CoopWorkPartition {
    // Choose output tile dimensions based on precision.
    let (tile_m, tile_n) = match config.precision {
        CoopPrecision::F16 | CoopPrecision::BF16 | CoopPrecision::TF32 => (128, 128),
        CoopPrecision::F32 => (64, 64),
        CoopPrecision::F64 => (32, 32),
    };

    let m_tiles = config.m.div_ceil(tile_m);
    let n_tiles = config.n.div_ceil(tile_n);
    let num_groups = m_tiles * n_tiles;

    let k_per_cta = config.k.div_ceil(config.cta_cluster_size);
    let k_remainder = config
        .k
        .saturating_sub(k_per_cta * (config.cta_cluster_size - 1));

    CoopWorkPartition {
        k_per_cta,
        k_remainder,
        output_tile_m: tile_m,
        output_tile_n: tile_n,
        num_cta_groups: num_groups,
        ctas_per_group: config.cta_cluster_size,
        total_ctas: num_groups * config.cta_cluster_size,
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics for a cooperative GEMM execution plan.
#[derive(Debug, Clone)]
pub struct CoopGemmStats {
    /// Total number of CTAs launched.
    pub total_ctas: usize,
    /// K elements processed per CTA.
    pub k_per_cta: usize,
    /// Total floating-point operations (2 * M * N * K).
    pub compute_flops: u64,
    /// Bytes needed for the reduction workspace (zero for atomic/cluster).
    pub reduction_overhead_bytes: u64,
    /// Estimated speed-up vs a single-CTA GEMM (based on K parallelism
    /// minus reduction overhead).
    pub speedup_vs_single_cta: f64,
}

// ---------------------------------------------------------------------------
// Execution plan
// ---------------------------------------------------------------------------

/// A fully resolved cooperative GEMM execution plan.
///
/// Created from a validated [`CooperativeGemmConfig`], it pre-computes the
/// work partition and exposes methods for PTX generation, workspace sizing,
/// and launch parameter calculation.
#[derive(Debug, Clone)]
pub struct CooperativeGemmPlan {
    config: CooperativeGemmConfig,
    partition: CoopWorkPartition,
    resolved_strategy: CoopReductionStrategy,
}

impl CooperativeGemmPlan {
    /// Creates a new cooperative GEMM plan from the given configuration.
    ///
    /// # Errors
    ///
    /// Propagates validation errors from [`CooperativeGemmConfig::validate`].
    pub fn new(config: CooperativeGemmConfig) -> BlasResult<Self> {
        config.validate()?;
        let partition = partition_work(&config);
        let resolved_strategy = config.resolve_strategy();
        Ok(Self {
            config,
            partition,
            resolved_strategy,
        })
    }

    /// Generates PTX for the partial GEMM kernel.
    ///
    /// Each CTA computes a partial C tile over its slice of the K dimension
    /// and writes the result to a workspace buffer (TwoPhase) or atomically
    /// accumulates (AtomicAccumulate).
    pub fn generate_partial_gemm_ptx(&self) -> BlasResult<String> {
        let acc_ty = self.config.precision.acc_ptx_str();
        let in_ty = self.config.precision.input_ptx_str();
        let elem_bytes = self.config.precision.input_bytes();
        let acc_bytes = self.config.precision.acc_bytes();

        let kernel_name = format!(
            "coop_partial_gemm_{}_c{}",
            acc_ty.trim_start_matches('.'),
            self.config.cta_cluster_size,
        );

        let mut ptx = String::with_capacity(4096);

        wl(
            &mut ptx,
            &format!(".version {}", self.config.sm_version.ptx_version()),
        )?;
        wl(
            &mut ptx,
            &format!(".target {}", self.config.sm_version.as_ptx_str()),
        )?;
        wl(&mut ptx, ".address_size 64")?;
        wl(&mut ptx, "")?;

        // Kernel signature: (A_ptr, B_ptr, workspace_ptr, M, N, K, k_start, k_end, cta_id_in_group)
        wl(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
        wl(&mut ptx, "    .param .u64 %param_a,")?;
        wl(&mut ptx, "    .param .u64 %param_b,")?;
        wl(&mut ptx, "    .param .u64 %param_ws,")?;
        wl(&mut ptx, "    .param .u32 %param_m,")?;
        wl(&mut ptx, "    .param .u32 %param_n,")?;
        wl(&mut ptx, "    .param .u32 %param_k,")?;
        wl(&mut ptx, "    .param .u32 %param_k_start,")?;
        wl(&mut ptx, "    .param .u32 %param_k_end,")?;
        wl(&mut ptx, "    .param .u32 %param_cta_id")?;
        wl(&mut ptx, ")")?;
        wl(&mut ptx, "{")?;

        // Register declarations
        wl(&mut ptx, "    .reg .b32 %r<32>;")?;
        wl(&mut ptx, "    .reg .b64 %rd<32>;")?;
        wl(&mut ptx, "    .reg .f32 %f<32>;")?;
        wl(&mut ptx, "    .reg .f64 %fd<8>;")?;
        wl(&mut ptx, "    .reg .pred %p<8>;")?;

        // Shared memory for tile loading
        let tile_m = self.partition.output_tile_m;
        let tile_n = self.partition.output_tile_n;
        let tile_k = 32usize; // K-tile for inner loop
        let smem_a = tile_m * tile_k * elem_bytes;
        let smem_b = tile_k * tile_n * elem_bytes;
        wl(
            &mut ptx,
            &format!("    .shared .align 16 .b8 smem_a[{smem_a}];"),
        )?;
        wl(
            &mut ptx,
            &format!("    .shared .align 16 .b8 smem_b[{smem_b}];"),
        )?;
        wl(&mut ptx, "")?;

        // Thread indexing
        wl(&mut ptx, "    mov.u32 %r0, %tid.x;")?;
        wl(
            &mut ptx,
            "    mov.u32 %r1, %ctaid.x;     // output tile index",
        )?;
        wl(&mut ptx, "    mov.u32 %r2, %ntid.x;")?;
        wl(&mut ptx, "")?;

        // Load parameters
        wl(&mut ptx, "    ld.param.u64 %rd0, [%param_a];")?;
        wl(&mut ptx, "    ld.param.u64 %rd1, [%param_b];")?;
        wl(&mut ptx, "    ld.param.u64 %rd2, [%param_ws];")?;
        wl(&mut ptx, "    ld.param.u32 %r3, [%param_m];")?;
        wl(&mut ptx, "    ld.param.u32 %r4, [%param_n];")?;
        wl(&mut ptx, "    ld.param.u32 %r5, [%param_k];")?;
        wl(&mut ptx, "    ld.param.u32 %r6, [%param_k_start];")?;
        wl(&mut ptx, "    ld.param.u32 %r7, [%param_k_end];")?;
        wl(&mut ptx, "    ld.param.u32 %r8, [%param_cta_id];")?;
        wl(&mut ptx, "")?;

        // Compute row and column within the output tile
        wl(
            &mut ptx,
            &format!("    // tile_m={tile_m}, tile_n={tile_n}"),
        )?;
        wl(&mut ptx, &format!("    mov.u32 %r9, {tile_n};"))?;
        wl(&mut ptx, "    div.u32 %r10, %r1, %r9;    // tile_row")?;
        wl(&mut ptx, "    rem.u32 %r11, %r1, %r9;    // tile_col")?;
        wl(&mut ptx, "")?;

        // Compute global row = tile_row * tile_m + thread_row
        wl(&mut ptx, &format!("    mov.u32 %r12, {tile_m};"))?;
        wl(
            &mut ptx,
            "    mul.lo.u32 %r13, %r10, %r12;  // tile_row * tile_m",
        )?;
        wl(
            &mut ptx,
            "    mov.u32 %r14, %r0;             // thread_id as row offset",
        )?;
        wl(&mut ptx, "    add.u32 %r15, %r13, %r14;      // global_row")?;
        wl(&mut ptx, "")?;

        // Bounds check
        wl(&mut ptx, "    setp.ge.u32 %p0, %r15, %r3;    // row >= M")?;
        wl(&mut ptx, "    @%p0 bra $PARTIAL_DONE;")?;
        wl(&mut ptx, "")?;

        // Initialise accumulator to zero
        if self.config.precision == CoopPrecision::F64 {
            wl(
                &mut ptx,
                "    mov.f64 %fd0, 0d0000000000000000;  // acc = 0.0",
            )?;
        } else {
            wl(&mut ptx, "    mov.f32 %f0, 0f00000000;  // acc = 0.0")?;
        }
        wl(&mut ptx, "")?;

        // K-loop: iterate from k_start to k_end
        wl(&mut ptx, "    mov.u32 %r16, %r6;  // k = k_start")?;
        wl(&mut ptx, "$K_LOOP:")?;
        wl(&mut ptx, "    setp.ge.u32 %p1, %r16, %r7;")?;
        wl(&mut ptx, "    @%p1 bra $K_LOOP_END;")?;
        wl(&mut ptx, "")?;

        // Load A[row, k] and B[k, col] and multiply-add
        wl(&mut ptx, "    // A[row, k]: row-major offset = row * K + k")?;
        wl(&mut ptx, "    cvt.u64.u32 %rd3, %r15;       // row")?;
        wl(&mut ptx, "    cvt.u64.u32 %rd4, %r5;        // K")?;
        wl(&mut ptx, "    mul.lo.u64 %rd5, %rd3, %rd4;")?;
        wl(&mut ptx, "    cvt.u64.u32 %rd6, %r16;       // k")?;
        wl(&mut ptx, "    add.u64 %rd5, %rd5, %rd6;")?;
        wl(
            &mut ptx,
            &format!("    mul.lo.u64 %rd5, %rd5, {elem_bytes};"),
        )?;
        wl(&mut ptx, "    add.u64 %rd7, %rd0, %rd5;")?;
        wl(&mut ptx, &format!("    ld.global{in_ty} %f1, [%rd7];"))?;
        wl(&mut ptx, "")?;

        wl(&mut ptx, "    // B[k, col]: row-major offset = k * N + col")?;
        wl(&mut ptx, "    cvt.u64.u32 %rd8, %r4;        // N")?;
        wl(&mut ptx, "    mul.lo.u64 %rd9, %rd6, %rd8;")?;
        wl(&mut ptx, "    cvt.u64.u32 %rd10, %r11;      // col")?;
        wl(&mut ptx, "    add.u64 %rd9, %rd9, %rd10;")?;
        wl(
            &mut ptx,
            &format!("    mul.lo.u64 %rd9, %rd9, {elem_bytes};"),
        )?;
        wl(&mut ptx, "    add.u64 %rd11, %rd1, %rd9;")?;
        wl(&mut ptx, &format!("    ld.global{in_ty} %f2, [%rd11];"))?;
        wl(&mut ptx, "")?;

        // FMA
        if self.config.precision == CoopPrecision::F64 {
            wl(&mut ptx, "    fma.rn.f64 %fd0, %fd1, %fd2, %fd0;")?;
        } else {
            wl(&mut ptx, "    fma.rn.f32 %f0, %f1, %f2, %f0;")?;
        }
        wl(&mut ptx, "")?;

        wl(&mut ptx, "    add.u32 %r16, %r16, 1;")?;
        wl(&mut ptx, "    bra $K_LOOP;")?;
        wl(&mut ptx, "$K_LOOP_END:")?;
        wl(&mut ptx, "")?;

        // Write partial result to workspace[cta_id * M * N + row * N + col]
        wl(
            &mut ptx,
            "    // Workspace offset: cta_id * M * N + row * N + col",
        )?;
        wl(&mut ptx, "    cvt.u64.u32 %rd12, %r8;       // cta_id")?;
        wl(&mut ptx, "    cvt.u64.u32 %rd13, %r3;       // M")?;
        wl(&mut ptx, "    cvt.u64.u32 %rd14, %r4;       // N")?;
        wl(&mut ptx, "    mul.lo.u64 %rd15, %rd13, %rd14;  // M * N")?;
        wl(
            &mut ptx,
            "    mul.lo.u64 %rd16, %rd12, %rd15;  // cta_id * M*N",
        )?;
        wl(&mut ptx, "    cvt.u64.u32 %rd17, %r15;      // row")?;
        wl(&mut ptx, "    mul.lo.u64 %rd18, %rd17, %rd14;  // row * N")?;
        wl(&mut ptx, "    add.u64 %rd16, %rd16, %rd18;")?;
        wl(&mut ptx, "    cvt.u64.u32 %rd19, %r11;      // col")?;
        wl(&mut ptx, "    add.u64 %rd16, %rd16, %rd19;")?;
        wl(
            &mut ptx,
            &format!("    mul.lo.u64 %rd16, %rd16, {acc_bytes};"),
        )?;
        wl(&mut ptx, "    add.u64 %rd20, %rd2, %rd16;")?;

        if self.config.precision == CoopPrecision::F64 {
            wl(&mut ptx, "    st.global.f64 [%rd20], %fd0;")?;
        } else {
            wl(&mut ptx, "    st.global.f32 [%rd20], %f0;")?;
        }
        wl(&mut ptx, "")?;

        wl(&mut ptx, "$PARTIAL_DONE:")?;
        wl(&mut ptx, "    ret;")?;
        wl(&mut ptx, "}")?;

        Ok(ptx)
    }

    /// Generates PTX for the reduction kernel (TwoPhase strategy).
    ///
    /// Sums `cta_cluster_size` partial results from the workspace into the
    /// final output matrix C. Each thread handles one output element.
    pub fn generate_reduction_ptx(&self) -> BlasResult<String> {
        let acc_ty = self.config.precision.acc_ptx_str();
        let acc_bytes = self.config.precision.acc_bytes();
        let cluster = self.config.cta_cluster_size;

        let kernel_name = format!(
            "coop_reduce_{}_c{}",
            acc_ty.trim_start_matches('.'),
            cluster,
        );

        let mut ptx = String::with_capacity(2048);

        wl(
            &mut ptx,
            &format!(".version {}", self.config.sm_version.ptx_version()),
        )?;
        wl(
            &mut ptx,
            &format!(".target {}", self.config.sm_version.as_ptx_str()),
        )?;
        wl(&mut ptx, ".address_size 64")?;
        wl(&mut ptx, "")?;

        // Kernel: (workspace_ptr, c_ptr, mn_count)
        wl(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
        wl(&mut ptx, "    .param .u64 %param_ws,")?;
        wl(&mut ptx, "    .param .u64 %param_c,")?;
        wl(&mut ptx, "    .param .u32 %param_mn")?;
        wl(&mut ptx, ")")?;
        wl(&mut ptx, "{")?;

        wl(&mut ptx, "    .reg .b32 %r<16>;")?;
        wl(&mut ptx, "    .reg .b64 %rd<16>;")?;
        wl(&mut ptx, "    .reg .f32 %f<16>;")?;
        wl(&mut ptx, "    .reg .f64 %fd<8>;")?;
        wl(&mut ptx, "    .reg .pred %p<4>;")?;
        wl(&mut ptx, "")?;

        // Global index
        wl(&mut ptx, "    mov.u32 %r0, %tid.x;")?;
        wl(&mut ptx, "    mov.u32 %r1, %ctaid.x;")?;
        wl(&mut ptx, "    mov.u32 %r2, %ntid.x;")?;
        wl(
            &mut ptx,
            "    mad.lo.u32 %r3, %r1, %r2, %r0;  // global idx",
        )?;
        wl(&mut ptx, "")?;

        // Bounds check
        wl(&mut ptx, "    ld.param.u32 %r4, [%param_mn];")?;
        wl(&mut ptx, "    setp.ge.u32 %p0, %r3, %r4;")?;
        wl(&mut ptx, "    @%p0 bra $COOP_REDUCE_DONE;")?;
        wl(&mut ptx, "")?;

        // Load pointers
        wl(&mut ptx, "    ld.param.u64 %rd0, [%param_ws];")?;
        wl(&mut ptx, "    ld.param.u64 %rd1, [%param_c];")?;
        wl(&mut ptx, "")?;

        // Element byte offset
        wl(&mut ptx, "    cvt.u64.u32 %rd2, %r3;")?;
        wl(
            &mut ptx,
            &format!("    mul.lo.u64 %rd2, %rd2, {acc_bytes};"),
        )?;

        // Partition stride in bytes: mn_count * acc_bytes
        wl(&mut ptx, "    cvt.u64.u32 %rd3, %r4;")?;
        wl(
            &mut ptx,
            &format!("    mul.lo.u64 %rd3, %rd3, {acc_bytes};  // partition stride"),
        )?;
        wl(&mut ptx, "")?;

        // Sum across partitions
        let is_f64 = self.config.precision == CoopPrecision::F64;
        if is_f64 {
            wl(&mut ptx, "    mov.f64 %fd0, 0d0000000000000000;  // acc")?;
        } else {
            wl(&mut ptx, "    mov.f32 %f0, 0f00000000;  // acc")?;
        }
        wl(&mut ptx, "    add.u64 %rd4, %rd0, %rd2;  // ws + offset")?;

        for _ in 0..cluster {
            if is_f64 {
                wl(&mut ptx, "    ld.global.f64 %fd1, [%rd4];")?;
                wl(&mut ptx, "    add.f64 %fd0, %fd0, %fd1;")?;
            } else {
                wl(&mut ptx, "    ld.global.f32 %f1, [%rd4];")?;
                wl(&mut ptx, "    add.f32 %f0, %f0, %f1;")?;
            }
            wl(&mut ptx, "    add.u64 %rd4, %rd4, %rd3;")?;
        }
        wl(&mut ptx, "")?;

        // Store to C
        wl(&mut ptx, "    add.u64 %rd5, %rd1, %rd2;  // c + offset")?;
        if is_f64 {
            wl(&mut ptx, "    st.global.f64 [%rd5], %fd0;")?;
        } else {
            wl(&mut ptx, "    st.global.f32 [%rd5], %f0;")?;
        }
        wl(&mut ptx, "")?;

        wl(&mut ptx, "$COOP_REDUCE_DONE:")?;
        wl(&mut ptx, "    ret;")?;
        wl(&mut ptx, "}")?;

        Ok(ptx)
    }

    /// Generates PTX for cluster-based cooperative GEMM (SM 90+).
    ///
    /// Uses `barrier.cluster` and `ld.shared::cluster` for intra-cluster
    /// reduction without a global workspace. This is a single-kernel
    /// approach available only on Hopper and later architectures.
    pub fn generate_cluster_cooperative_ptx(&self) -> BlasResult<String> {
        if self.config.sm_version < SmVersion::Sm90 {
            return Err(BlasError::UnsupportedOperation(
                "cluster cooperative GEMM requires SM >= 90".into(),
            ));
        }

        let acc_ty = self.config.precision.acc_ptx_str();
        let acc_bytes = self.config.precision.acc_bytes();
        let in_ty = self.config.precision.input_ptx_str();
        let elem_bytes = self.config.precision.input_bytes();
        let cluster = self.config.cta_cluster_size;
        let tile_m = self.partition.output_tile_m;
        let tile_n = self.partition.output_tile_n;
        let tile_k = 32usize;

        let kernel_name = format!(
            "coop_cluster_gemm_{}_c{}",
            acc_ty.trim_start_matches('.'),
            cluster,
        );

        let mut ptx = String::with_capacity(4096);

        wl(
            &mut ptx,
            &format!(".version {}", self.config.sm_version.ptx_version()),
        )?;
        wl(
            &mut ptx,
            &format!(".target {}", self.config.sm_version.as_ptx_str()),
        )?;
        wl(&mut ptx, ".address_size 64")?;
        wl(&mut ptx, "")?;

        // Cluster attribute
        wl(&mut ptx, &format!(".reqnctapercluster {cluster}"))?;
        wl(&mut ptx, "")?;

        // Kernel signature
        wl(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
        wl(&mut ptx, "    .param .u64 %param_a,")?;
        wl(&mut ptx, "    .param .u64 %param_b,")?;
        wl(&mut ptx, "    .param .u64 %param_c,")?;
        wl(&mut ptx, "    .param .u32 %param_m,")?;
        wl(&mut ptx, "    .param .u32 %param_n,")?;
        wl(&mut ptx, "    .param .u32 %param_k")?;
        wl(&mut ptx, ")")?;
        wl(&mut ptx, "{")?;

        // Registers
        wl(&mut ptx, "    .reg .b32 %r<32>;")?;
        wl(&mut ptx, "    .reg .b64 %rd<32>;")?;
        wl(&mut ptx, "    .reg .f32 %f<32>;")?;
        wl(&mut ptx, "    .reg .f64 %fd<8>;")?;
        wl(&mut ptx, "    .reg .pred %p<8>;")?;
        wl(&mut ptx, "")?;

        // Shared memory for partial accumulation and tile data
        let smem_a = tile_m * tile_k * elem_bytes;
        let smem_b = tile_k * tile_n * elem_bytes;
        let smem_acc = tile_m * tile_n * acc_bytes;
        wl(
            &mut ptx,
            &format!("    .shared .align 16 .b8 smem_a[{smem_a}];"),
        )?;
        wl(
            &mut ptx,
            &format!("    .shared .align 16 .b8 smem_b[{smem_b}];"),
        )?;
        wl(
            &mut ptx,
            &format!("    .shared .align 16 .b8 smem_acc[{smem_acc}];"),
        )?;
        wl(&mut ptx, "")?;

        // Thread / CTA indexing
        wl(&mut ptx, "    mov.u32 %r0, %tid.x;")?;
        wl(
            &mut ptx,
            "    mov.u32 %r1, %ctaid.x;     // CTA index within cluster",
        )?;
        wl(&mut ptx, "    mov.u32 %r2, %ntid.x;")?;
        wl(&mut ptx, "")?;

        // Load parameters
        wl(&mut ptx, "    ld.param.u64 %rd0, [%param_a];")?;
        wl(&mut ptx, "    ld.param.u64 %rd1, [%param_b];")?;
        wl(&mut ptx, "    ld.param.u64 %rd2, [%param_c];")?;
        wl(&mut ptx, "    ld.param.u32 %r3, [%param_m];")?;
        wl(&mut ptx, "    ld.param.u32 %r4, [%param_n];")?;
        wl(&mut ptx, "    ld.param.u32 %r5, [%param_k];")?;
        wl(&mut ptx, "")?;

        // Compute K-range for this CTA in the cluster
        let k_per_cta = self.partition.k_per_cta;
        wl(&mut ptx, &format!("    mov.u32 %r6, {k_per_cta};"))?;
        wl(&mut ptx, "    mul.lo.u32 %r7, %r1, %r6;  // k_start")?;
        wl(
            &mut ptx,
            "    add.u32 %r8, %r7, %r6;     // k_end (tentative)",
        )?;
        wl(&mut ptx, "    min.u32 %r8, %r8, %r5;     // clamp to K")?;
        wl(&mut ptx, "")?;

        // Compute partial GEMM and store to shared memory
        wl(&mut ptx, "    // -- Partial GEMM over [k_start, k_end) --")?;
        let is_f64 = self.config.precision == CoopPrecision::F64;
        if is_f64 {
            wl(&mut ptx, "    mov.f64 %fd0, 0d0000000000000000;")?;
        } else {
            wl(&mut ptx, "    mov.f32 %f0, 0f00000000;  // acc")?;
        }
        wl(&mut ptx, "")?;

        // Simplified K-loop (representative, not fully unrolled)
        wl(&mut ptx, "    mov.u32 %r9, %r7;  // k = k_start")?;
        wl(&mut ptx, "$CLUSTER_K_LOOP:")?;
        wl(&mut ptx, "    setp.ge.u32 %p0, %r9, %r8;")?;
        wl(&mut ptx, "    @%p0 bra $CLUSTER_K_DONE;")?;
        wl(&mut ptx, "")?;
        wl(
            &mut ptx,
            &format!("    // Load A and B elements (simplified {in_ty})"),
        )?;
        wl(&mut ptx, "    cvt.u64.u32 %rd3, %r0;  // thread as row")?;
        wl(&mut ptx, "    cvt.u64.u32 %rd4, %r5;  // K")?;
        wl(&mut ptx, "    mul.lo.u64 %rd5, %rd3, %rd4;")?;
        wl(&mut ptx, "    cvt.u64.u32 %rd6, %r9;  // k")?;
        wl(&mut ptx, "    add.u64 %rd5, %rd5, %rd6;")?;
        wl(
            &mut ptx,
            &format!("    mul.lo.u64 %rd5, %rd5, {elem_bytes};"),
        )?;
        wl(&mut ptx, "    add.u64 %rd7, %rd0, %rd5;")?;
        // Load A[row, k] into appropriate register based on precision
        if is_f64 {
            wl(&mut ptx, "    ld.global.f64 %fd1, [%rd7];")?;
        } else {
            wl(&mut ptx, "    ld.global.f32 %f1, [%rd7];")?;
        }
        wl(&mut ptx, "")?;

        // Compute B address: column-major B, B[k, col] = B_ptr + (col * K + k) * elem_bytes
        // col = %r0 (tid.x), K = %r5, k = %r9
        wl(
            &mut ptx,
            "    cvt.u64.u32 %rd9, %r0;           // col (tid.x)",
        )?;
        wl(&mut ptx, "    cvt.u64.u32 %rd10, %r5;          // K")?;
        wl(&mut ptx, "    mul.lo.u64 %rd11, %rd9, %rd10;   // col * K")?;
        wl(&mut ptx, "    cvt.u64.u32 %rd12, %r9;          // k")?;
        wl(
            &mut ptx,
            "    add.u64 %rd11, %rd11, %rd12;     // col * K + k",
        )?;
        wl(
            &mut ptx,
            &format!("    mul.lo.u64 %rd11, %rd11, {elem_bytes}; // byte offset B"),
        )?;
        wl(
            &mut ptx,
            "    add.u64 %rd12, %rd1, %rd11;      // B_ptr + offset",
        )?;
        // Load B[k, col] and compute FMA: acc += A[row,k] * B[k,col]
        if is_f64 {
            wl(&mut ptx, "    ld.global.f64 %fd2, [%rd12];")?;
            wl(
                &mut ptx,
                "    fma.rn.f64 %fd0, %fd1, %fd2, %fd0;  // acc += A[row,k] * B[k,col]",
            )?;
        } else {
            wl(&mut ptx, "    ld.global.f32 %f2, [%rd12];")?;
            wl(
                &mut ptx,
                "    fma.rn.f32 %f0, %f1, %f2, %f0;  // acc += A[row,k] * B[k,col]",
            )?;
        }
        wl(&mut ptx, "    add.u32 %r9, %r9, 1;")?;
        wl(&mut ptx, "    bra $CLUSTER_K_LOOP;")?;
        wl(&mut ptx, "$CLUSTER_K_DONE:")?;
        wl(&mut ptx, "")?;

        // Store partial to shared memory
        wl(&mut ptx, "    // Store partial to smem_acc")?;
        wl(&mut ptx, "    cvt.u64.u32 %rd8, %r0;")?;
        wl(
            &mut ptx,
            &format!("    mul.lo.u64 %rd8, %rd8, {acc_bytes};"),
        )?;
        if is_f64 {
            wl(&mut ptx, "    st.shared.f64 [smem_acc + %rd8], %fd0;")?;
        } else {
            wl(&mut ptx, "    st.shared.f32 [smem_acc + %rd8], %f0;")?;
        }
        wl(&mut ptx, "")?;

        // Cluster-level barrier and reduction
        wl(&mut ptx, "    // Cluster barrier for synchronisation")?;
        wl(&mut ptx, "    barrier.cluster.arrive;")?;
        wl(&mut ptx, "    barrier.cluster.wait;")?;
        wl(&mut ptx, "")?;

        // CTA 0 in the cluster reduces partial results from all cluster CTAs
        wl(&mut ptx, "    // CTA 0 reduces across cluster")?;
        wl(&mut ptx, "    setp.ne.u32 %p1, %r1, 0;")?;
        wl(&mut ptx, "    @%p1 bra $CLUSTER_DONE;")?;
        wl(&mut ptx, "")?;

        // Read from other CTAs via ld.shared::cluster
        for peer in 1..cluster {
            wl(
                &mut ptx,
                &format!("    // Reduce partial from CTA {peer} via distributed shared memory"),
            )?;
            if is_f64 {
                wl(
                    &mut ptx,
                    &format!(
                        "    ld.shared::cluster.f64 %fd1, [smem_acc + %rd8 + {peer} * {smem_acc}];"
                    ),
                )?;
                wl(&mut ptx, "    add.f64 %fd0, %fd0, %fd1;")?;
            } else {
                wl(
                    &mut ptx,
                    &format!(
                        "    ld.shared::cluster.f32 %f2, [smem_acc + %rd8 + {peer} * {smem_acc}];"
                    ),
                )?;
                wl(&mut ptx, "    add.f32 %f0, %f0, %f2;")?;
            }
        }
        wl(&mut ptx, "")?;

        // Write final result to global C
        wl(&mut ptx, "    // Write final result to C")?;
        wl(
            &mut ptx,
            "    add.u64 %rd9, %rd2, %rd8;  // C + element offset",
        )?;
        if is_f64 {
            wl(&mut ptx, "    st.global.f64 [%rd9], %fd0;")?;
        } else {
            wl(&mut ptx, "    st.global.f32 [%rd9], %f0;")?;
        }
        wl(&mut ptx, "")?;

        wl(&mut ptx, "$CLUSTER_DONE:")?;
        wl(&mut ptx, "    ret;")?;
        wl(&mut ptx, "}")?;

        Ok(ptx)
    }

    /// Returns the workspace size in bytes needed for partial results.
    ///
    /// For `ClusterSharedMemory`, no global workspace is needed (returns 0).
    /// For `TwoPhase`, workspace = `M * N * cluster_size * acc_bytes`.
    /// For `AtomicAccumulate`, no workspace is needed (returns 0).
    pub fn workspace_bytes(&self) -> usize {
        match self.resolved_strategy {
            CoopReductionStrategy::TwoPhase => {
                self.config.m
                    * self.config.n
                    * self.config.cta_cluster_size
                    * self.config.precision.acc_bytes()
            }
            CoopReductionStrategy::ClusterSharedMemory
            | CoopReductionStrategy::AtomicAccumulate => 0,
            CoopReductionStrategy::Auto => {
                // Should not happen after resolve_strategy, but be safe.
                0
            }
        }
    }

    /// Shared memory per CTA in bytes.
    ///
    /// Includes space for A and B tiles, plus (for cluster mode) the
    /// accumulator tile.
    pub fn shared_memory_bytes(&self) -> usize {
        let tile_m = self.partition.output_tile_m;
        let tile_n = self.partition.output_tile_n;
        let tile_k = 32usize;
        let elem_bytes = self.config.precision.input_bytes();
        let smem_a = tile_m * tile_k * elem_bytes;
        let smem_b = tile_k * tile_n * elem_bytes;

        let smem_acc = if self.resolved_strategy == CoopReductionStrategy::ClusterSharedMemory {
            tile_m * tile_n * self.config.precision.acc_bytes()
        } else {
            0
        };

        smem_a + smem_b + smem_acc
    }

    /// Returns `(grid_size, block_size)` for the partial GEMM kernel.
    pub fn launch_params(&self) -> (usize, usize) {
        let grid = self.partition.total_ctas;
        // Threads per CTA: enough to cover one tile row or a reasonable
        // fraction of the tile.
        let block = self.partition.output_tile_m.min(256);
        (grid, block)
    }

    /// Computes performance statistics for this plan.
    pub fn stats(&self) -> CoopGemmStats {
        let m = self.config.m as u64;
        let n = self.config.n as u64;
        let k = self.config.k as u64;
        let compute_flops = 2 * m * n * k;

        let reduction_overhead_bytes = self.workspace_bytes() as u64;

        // Estimate speed-up: ideal parallelism is cluster_size, penalised
        // by reduction overhead relative to compute.
        let cluster = self.config.cta_cluster_size as f64;
        let overhead_ratio = if compute_flops > 0 {
            reduction_overhead_bytes as f64 / compute_flops as f64
        } else {
            1.0
        };
        // Simple model: speedup = cluster / (1 + overhead_ratio * cluster)
        let speedup = cluster / (1.0 + overhead_ratio * cluster);

        CoopGemmStats {
            total_ctas: self.partition.total_ctas,
            k_per_cta: self.partition.k_per_cta,
            compute_flops,
            reduction_overhead_bytes,
            speedup_vs_single_cta: speedup,
        }
    }

    /// Returns a human-readable description of this cooperative GEMM plan.
    pub fn describe(&self) -> String {
        let stats = self.stats();
        let strategy_name = match self.resolved_strategy {
            CoopReductionStrategy::ClusterSharedMemory => "ClusterSharedMemory",
            CoopReductionStrategy::TwoPhase => "TwoPhase",
            CoopReductionStrategy::AtomicAccumulate => "AtomicAccumulate",
            CoopReductionStrategy::Auto => "Auto",
        };

        format!(
            "CooperativeGEMM: M={} N={} K={}, cluster_size={}, strategy={}, \
             total_ctas={}, k_per_cta={}, compute_flops={}, \
             workspace_bytes={}, est_speedup={:.2}x",
            self.config.m,
            self.config.n,
            self.config.k,
            self.config.cta_cluster_size,
            strategy_name,
            stats.total_ctas,
            stats.k_per_cta,
            stats.compute_flops,
            stats.reduction_overhead_bytes,
            stats.speedup_vs_single_cta,
        )
    }
}

/// Writes a line to the PTX buffer, mapping formatting errors.
fn wl(ptx: &mut String, line: &str) -> BlasResult<()> {
    writeln!(ptx, "{line}").map_err(|e| BlasError::PtxGeneration(format!("fmt error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(
        m: usize,
        n: usize,
        k: usize,
        cluster: usize,
        sm: SmVersion,
        strategy: CoopReductionStrategy,
        precision: CoopPrecision,
    ) -> CooperativeGemmConfig {
        CooperativeGemmConfig {
            m,
            n,
            k,
            sm_version: sm,
            cta_cluster_size: cluster,
            reduction_strategy: strategy,
            precision,
        }
    }

    // -- Config validation ---------------------------------------------------

    #[test]
    fn validate_valid_config() {
        let cfg = make_config(
            1024,
            1024,
            2048,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::ClusterSharedMemory,
            CoopPrecision::F32,
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_invalid_cluster_size_3() {
        let cfg = make_config(
            1024,
            1024,
            2048,
            3,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F32,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_invalid_cluster_size_32() {
        let cfg = make_config(
            1024,
            1024,
            2048,
            32,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F32,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_cluster_sm_requirement() {
        let cfg = make_config(
            1024,
            1024,
            2048,
            4,
            SmVersion::Sm80,
            CoopReductionStrategy::ClusterSharedMemory,
            CoopPrecision::F32,
        );
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_zero_dimensions() {
        let cfg = make_config(
            0,
            1024,
            2048,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F32,
        );
        assert!(cfg.validate().is_err());
    }

    // -- is_cooperative_beneficial -------------------------------------------

    #[test]
    fn beneficial_large_k() {
        assert!(is_cooperative_beneficial(1024, 1024, 4096, SmVersion::Sm90));
    }

    #[test]
    fn not_beneficial_small_k() {
        assert!(!is_cooperative_beneficial(1024, 1024, 128, SmVersion::Sm80));
    }

    #[test]
    fn not_beneficial_tiny_output() {
        assert!(!is_cooperative_beneficial(8, 8, 4096, SmVersion::Sm90));
    }

    // -- Work partitioning ---------------------------------------------------

    #[test]
    fn partition_work_correctness() {
        let cfg = make_config(
            1024,
            1024,
            2048,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F32,
        );
        let part = partition_work(&cfg);
        assert_eq!(part.ctas_per_group, 4);
        // k_per_cta * cluster_size >= K
        assert!(part.k_per_cta * cfg.cta_cluster_size >= cfg.k);
        assert_eq!(part.output_tile_m, 64); // F32 tile
        assert_eq!(part.output_tile_n, 64);
        assert_eq!(part.total_ctas, part.num_cta_groups * part.ctas_per_group,);
    }

    #[test]
    fn partition_work_f16_tiles() {
        let cfg = make_config(
            256,
            256,
            1024,
            2,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F16,
        );
        let part = partition_work(&cfg);
        assert_eq!(part.output_tile_m, 128);
        assert_eq!(part.output_tile_n, 128);
    }

    // -- PTX generation ------------------------------------------------------

    #[test]
    fn partial_gemm_ptx_contains_kernel_name() {
        let cfg = make_config(
            512,
            512,
            2048,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F32,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        let ptx = plan.generate_partial_gemm_ptx().expect("ptx gen failed");
        assert!(ptx.contains("coop_partial_gemm_f32_c4"));
        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("$PARTIAL_DONE"));
    }

    #[test]
    fn reduction_ptx_contains_kernel_name() {
        let cfg = make_config(
            512,
            512,
            2048,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F32,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        let ptx = plan.generate_reduction_ptx().expect("ptx gen failed");
        assert!(ptx.contains("coop_reduce_f32_c4"));
        assert!(ptx.contains("$COOP_REDUCE_DONE"));
    }

    #[test]
    fn cluster_cooperative_ptx_sm90() {
        let cfg = make_config(
            1024,
            1024,
            4096,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::ClusterSharedMemory,
            CoopPrecision::F32,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        let ptx = plan
            .generate_cluster_cooperative_ptx()
            .expect("ptx gen failed");
        assert!(ptx.contains("coop_cluster_gemm_f32_c4"));
        assert!(ptx.contains("barrier.cluster"));
        assert!(ptx.contains("ld.shared::cluster"));
        assert!(ptx.contains(".reqnctapercluster 4"));
    }

    #[test]
    fn cluster_cooperative_ptx_rejected_on_ampere() {
        let cfg = make_config(
            1024,
            1024,
            4096,
            4,
            SmVersion::Sm80,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F32,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        let result = plan.generate_cluster_cooperative_ptx();
        assert!(result.is_err());
    }

    // -- Workspace bytes -----------------------------------------------------

    #[test]
    fn workspace_bytes_two_phase() {
        let cfg = make_config(
            256,
            256,
            2048,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F32,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        // 256 * 256 * 4 (CTAs) * 4 (bytes) = 1_048_576
        assert_eq!(plan.workspace_bytes(), 256 * 256 * 4 * 4);
    }

    #[test]
    fn workspace_bytes_cluster_is_zero() {
        let cfg = make_config(
            256,
            256,
            2048,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::ClusterSharedMemory,
            CoopPrecision::F32,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        assert_eq!(plan.workspace_bytes(), 0);
    }

    // -- Stats ---------------------------------------------------------------

    #[test]
    fn stats_flops() {
        let cfg = make_config(
            128,
            128,
            1024,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F32,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        let stats = plan.stats();
        assert_eq!(stats.compute_flops, 2 * 128 * 128 * 1024);
        assert!(stats.speedup_vs_single_cta > 1.0);
        assert!(stats.speedup_vs_single_cta <= 4.0);
    }

    // -- Auto strategy selection ---------------------------------------------

    #[test]
    fn auto_strategy_selects_cluster_on_hopper() {
        let cfg = make_config(
            1024,
            1024,
            4096,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::Auto,
            CoopPrecision::F32,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        assert_eq!(
            plan.resolved_strategy,
            CoopReductionStrategy::ClusterSharedMemory
        );
    }

    #[test]
    fn auto_strategy_selects_two_phase_for_large_cluster() {
        // cluster_size = 16 > 8, so on Hopper should pick TwoPhase
        // since K >= 2048
        let cfg = make_config(
            1024,
            1024,
            4096,
            16,
            SmVersion::Sm90,
            CoopReductionStrategy::Auto,
            CoopPrecision::F32,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        assert_eq!(plan.resolved_strategy, CoopReductionStrategy::TwoPhase);
    }

    #[test]
    fn auto_strategy_selects_atomic_for_small_k_on_ampere() {
        let cfg = make_config(
            1024,
            1024,
            1024,
            4,
            SmVersion::Sm80,
            CoopReductionStrategy::Auto,
            CoopPrecision::F32,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        assert_eq!(
            plan.resolved_strategy,
            CoopReductionStrategy::AtomicAccumulate
        );
    }

    // -- Describe ------------------------------------------------------------

    #[test]
    fn describe_output_format() {
        let cfg = make_config(
            1024,
            1024,
            4096,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F32,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        let desc = plan.describe();
        assert!(desc.contains("CooperativeGEMM"));
        assert!(desc.contains("M=1024"));
        assert!(desc.contains("N=1024"));
        assert!(desc.contains("K=4096"));
        assert!(desc.contains("cluster_size=4"));
        assert!(desc.contains("TwoPhase"));
    }

    // -- Different precisions ------------------------------------------------

    #[test]
    fn f64_precision_partial_gemm() {
        let cfg = make_config(
            256,
            256,
            2048,
            2,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F64,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        let ptx = plan.generate_partial_gemm_ptx().expect("ptx gen failed");
        assert!(ptx.contains(".f64"));
        assert!(ptx.contains("st.global.f64"));
    }

    #[test]
    fn f16_precision_workspace() {
        let cfg = make_config(
            256,
            256,
            2048,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::F16,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        // Accumulator is F32 (4 bytes) for F16 input
        assert_eq!(plan.workspace_bytes(), 256 * 256 * 4 * 4);
    }

    #[test]
    fn bf16_precision_shared_memory() {
        let cfg = make_config(
            512,
            512,
            2048,
            4,
            SmVersion::Sm90,
            CoopReductionStrategy::TwoPhase,
            CoopPrecision::BF16,
        );
        let plan = CooperativeGemmPlan::new(cfg).expect("plan creation failed");
        let smem = plan.shared_memory_bytes();
        // tile = 128x128 for BF16, tile_k = 32, input = 2 bytes
        // smem_a = 128 * 32 * 2 = 8192, smem_b = 32 * 128 * 2 = 8192
        // No acc for TwoPhase
        assert_eq!(smem, 8192 + 8192);
    }
}
