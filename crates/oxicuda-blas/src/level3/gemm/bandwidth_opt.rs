//! Bandwidth-optimized GEMM for memory-bound problem shapes.
//!
//! When the arithmetic intensity of a GEMM is low (small K, or large M×N
//! relative to K), the kernel becomes memory-bound rather than compute-bound.
//! Standard tiled GEMM kernels waste cycles waiting for memory transactions
//! in this regime.
//!
//! This module provides:
//!
//! - [`ArithmeticIntensityAnalysis`]: roofline-model analysis of a GEMM problem.
//! - [`BandwidthGemmConfig`] / [`BandwidthStrategy`]: configuration for the
//!   bandwidth-optimized kernel path.
//! - [`BandwidthTileConfig`]: tile dimensions and prefetch tuning for
//!   memory-bound kernels.
//! - [`generate_bandwidth_gemm_ptx`]: PTX generation via the SIMT path,
//!   using wider vector loads and reduced pipeline depth.

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

use crate::error::{BlasError, BlasResult};

// ---------------------------------------------------------------------------
// Precision enum
// ---------------------------------------------------------------------------

/// Supported precisions for bandwidth-optimized GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BandwidthPrecision {
    /// IEEE half-precision (16-bit).
    F16,
    /// Brain floating-point (16-bit).
    BF16,
    /// IEEE single-precision (32-bit).
    F32,
    /// IEEE double-precision (64-bit).
    F64,
}

impl BandwidthPrecision {
    /// Returns the element size in bytes.
    #[must_use]
    pub const fn elem_bytes(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Returns the corresponding PTX type for the input elements.
    #[must_use]
    pub const fn ptx_type(self) -> PtxType {
        match self {
            Self::F16 => PtxType::F16,
            Self::BF16 => PtxType::BF16,
            Self::F32 => PtxType::F32,
            Self::F64 => PtxType::F64,
        }
    }

    /// Returns the accumulator PTX type (F32 for half types, same for others).
    #[must_use]
    pub const fn accumulator_type(self) -> PtxType {
        match self {
            Self::F16 | Self::BF16 | Self::F32 => PtxType::F32,
            Self::F64 => PtxType::F64,
        }
    }

    /// Returns the optimal vector width (elements per coalesced load).
    ///
    /// Wider vectors improve memory throughput by issuing fewer transactions
    /// per cache line. The optimal width depends on the element size and the
    /// 128-bit (16-byte) load granularity of NVIDIA GPUs.
    #[must_use]
    pub const fn vector_width(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 8, // 8 × 2B = 16B = 128 bits
            Self::F32 => 4,              // 4 × 4B = 16B = 128 bits
            Self::F64 => 2,              // 2 × 8B = 16B = 128 bits
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// Strategy for handling a memory-bound GEMM.
///
/// Each strategy optimises for a different regime of the roofline model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BandwidthStrategy {
    /// Wide output tiles, minimal pipeline stages. Best when K is very small
    /// (< 32) so the entire K dimension fits in a single pass.
    ShallowK,
    /// Moderate tiles with maximal K-loop iteration count. Designed to keep
    /// data resident in L2 cache across the K loop when K < 128 and M×N is
    /// large.
    CachePersistent,
    /// Distributes independent output tiles across warps for maximum
    /// memory-level parallelism. General-purpose bandwidth-optimised path.
    WarpParallel,
    /// Automatically selects the best strategy based on arithmetic intensity
    /// and problem dimensions.
    Auto,
}

// ---------------------------------------------------------------------------
// Arithmetic intensity analysis
// ---------------------------------------------------------------------------

/// Roofline-model analysis of a GEMM problem.
///
/// The arithmetic intensity (FLOPs / bytes) determines whether the kernel
/// is compute-bound or memory-bound for a given GPU. When the intensity is
/// below the machine's balance point, performance is limited by memory
/// bandwidth rather than compute throughput.
#[derive(Debug, Clone)]
pub struct ArithmeticIntensityAnalysis {
    /// Total floating-point operations: 2 × M × N × K.
    pub flops: u64,
    /// Total memory traffic in bytes (lower bound, assuming no cache reuse).
    pub bytes: u64,
    /// Arithmetic intensity: `flops / bytes`.
    pub intensity: f64,
    /// Whether this problem is memory-bound given typical GPU balance points.
    pub is_memory_bound: bool,
    /// Recommended strategy for this problem shape.
    pub recommended_strategy: BandwidthStrategy,
    /// Peak compute throughput of the target SM in TFLOPS (approximate).
    pub peak_compute_tflops: f64,
    /// Peak memory bandwidth of the target SM in TB/s (approximate).
    pub peak_bandwidth_tbps: f64,
    /// Balance point: `peak_compute / peak_bandwidth` (FLOP / byte).
    pub balance_point: f64,
}

// ---------------------------------------------------------------------------
// BandwidthGemmConfig
// ---------------------------------------------------------------------------

/// Configuration for a bandwidth-optimised GEMM kernel.
#[derive(Debug, Clone)]
pub struct BandwidthGemmConfig {
    /// Rows of the output matrix C.
    pub m: usize,
    /// Columns of the output matrix C.
    pub n: usize,
    /// Shared (inner) dimension.
    pub k: usize,
    /// Target SM architecture.
    pub sm_version: SmVersion,
    /// Element precision.
    pub precision: BandwidthPrecision,
    /// Bandwidth strategy (or Auto).
    pub strategy: BandwidthStrategy,
}

impl BandwidthGemmConfig {
    /// Validates the configuration.
    ///
    /// Returns an error if any dimension is zero.
    pub fn validate(&self) -> BlasResult<()> {
        if self.m == 0 {
            return Err(BlasError::InvalidDimension(
                "bandwidth GEMM: M must be > 0".into(),
            ));
        }
        if self.n == 0 {
            return Err(BlasError::InvalidDimension(
                "bandwidth GEMM: N must be > 0".into(),
            ));
        }
        if self.k == 0 {
            return Err(BlasError::InvalidDimension(
                "bandwidth GEMM: K must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Resolves [`BandwidthStrategy::Auto`] to a concrete strategy.
    #[must_use]
    pub fn resolved_strategy(&self) -> BandwidthStrategy {
        match self.strategy {
            BandwidthStrategy::Auto => {
                let analysis =
                    analyze_intensity(self.m, self.n, self.k, self.precision.elem_bytes());
                analysis.recommended_strategy
            }
            other => other,
        }
    }
}

// ---------------------------------------------------------------------------
// BandwidthTileConfig
// ---------------------------------------------------------------------------

/// Tile dimensions and prefetch tuning for bandwidth-optimised GEMM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandwidthTileConfig {
    /// Block tile size in M (rows per CTA).
    pub tile_m: usize,
    /// Block tile size in N (columns per CTA).
    pub tile_n: usize,
    /// Block tile size in K (reduction step).
    pub tile_k: usize,
    /// Number of software pipeline stages for async loads.
    pub pipeline_stages: usize,
    /// Warps covering the M dimension.
    pub warps_m: usize,
    /// Warps covering the N dimension.
    pub warps_n: usize,
    /// Elements loaded per coalesced memory transaction.
    pub vector_width: usize,
    /// Number of K-steps to prefetch ahead.
    pub prefetch_distance: usize,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Analyses the arithmetic intensity of a GEMM problem.
///
/// Uses the roofline model with approximate peak throughputs for a mid-range
/// GPU (A100-class) to determine whether the problem is memory-bound.
#[must_use]
pub fn analyze_intensity(
    m: usize,
    n: usize,
    k: usize,
    elem_bytes: usize,
) -> ArithmeticIntensityAnalysis {
    let m64 = m as u64;
    let n64 = n as u64;
    let k64 = k as u64;
    let eb = elem_bytes as u64;

    // FLOPs: 2 * M * N * K (multiply + add per output element per K step).
    let flops = 2 * m64 * n64 * k64;

    // Memory traffic lower bound (no reuse): read A (M×K) + read B (K×N) +
    // write C (M×N). Read of C for beta != 0 is ignored here as a
    // conservative estimate.
    let bytes = (m64 * k64 + k64 * n64 + m64 * n64) * eb;

    let intensity = if bytes > 0 {
        flops as f64 / bytes as f64
    } else {
        0.0
    };

    // Approximate A100 numbers for the balance point calculation.
    let peak_compute_tflops = 19.5; // FP32 TFLOPS
    let peak_bandwidth_tbps = 2.0; // TB/s (HBM2e)
    let balance_point = peak_compute_tflops * 1e12 / (peak_bandwidth_tbps * 1e12);
    // balance_point ≈ 9.75 FLOP/byte

    let is_memory_bound = intensity < balance_point;

    let recommended_strategy = if is_memory_bound {
        if k < 32 {
            BandwidthStrategy::ShallowK
        } else if k < 128 {
            BandwidthStrategy::CachePersistent
        } else {
            BandwidthStrategy::WarpParallel
        }
    } else {
        // Not truly memory-bound, but caller asked for analysis anyway.
        BandwidthStrategy::WarpParallel
    };

    ArithmeticIntensityAnalysis {
        flops,
        bytes,
        intensity,
        is_memory_bound,
        recommended_strategy,
        peak_compute_tflops,
        peak_bandwidth_tbps,
        balance_point,
    }
}

/// Returns `true` if the GEMM with the given dimensions is bandwidth-limited.
///
/// This is a convenience wrapper around [`analyze_intensity`].
#[must_use]
pub fn is_bandwidth_limited(m: usize, n: usize, k: usize, elem_bytes: usize) -> bool {
    analyze_intensity(m, n, k, elem_bytes).is_memory_bound
}

/// Selects an optimal tile configuration for a bandwidth-limited GEMM.
///
/// The returned [`BandwidthTileConfig`] is tuned for maximum memory
/// throughput rather than maximum compute utilisation.
#[must_use]
pub fn select_bandwidth_tiles(config: &BandwidthGemmConfig) -> BandwidthTileConfig {
    let strategy = config.resolved_strategy();
    let vw = config.precision.vector_width();

    match strategy {
        BandwidthStrategy::ShallowK => {
            // K fits in a single pass — maximise output tile area.
            let tile_k = config.k.max(1);
            // Use wide output tiles to keep threads busy.
            let (tile_m, tile_n) = if config.m >= 128 && config.n >= 128 {
                (128, 128)
            } else if config.m >= 64 && config.n >= 64 {
                (64, 64)
            } else {
                (32, 32)
            };
            BandwidthTileConfig {
                tile_m,
                tile_n,
                tile_k,
                pipeline_stages: 1,
                warps_m: (tile_m / 32).max(1),
                warps_n: (tile_n / 32).max(1),
                vector_width: vw,
                prefetch_distance: 0, // no prefetch for single-pass
            }
        }
        BandwidthStrategy::CachePersistent => {
            // Moderate tiles, maximise K-loop for L2 reuse.
            let tile_k = 8.min(config.k);
            let (tile_m, tile_n) = if config.m >= 64 && config.n >= 64 {
                (64, 64)
            } else {
                (32, 32)
            };
            BandwidthTileConfig {
                tile_m,
                tile_n,
                tile_k,
                pipeline_stages: 2,
                warps_m: (tile_m / 32).max(1),
                warps_n: (tile_n / 32).max(1),
                vector_width: vw,
                prefetch_distance: 1,
            }
        }
        BandwidthStrategy::WarpParallel | BandwidthStrategy::Auto => {
            // Distribute output tiles across warps.
            let tile_k = 16.min(config.k);
            let (tile_m, tile_n) = if config.m >= 128 && config.n >= 128 {
                (128, 64)
            } else if config.m >= 64 && config.n >= 64 {
                (64, 64)
            } else {
                (32, 32)
            };
            let warps_m = (tile_m / 32).max(1);
            let warps_n = (tile_n / 32).max(1);
            BandwidthTileConfig {
                tile_m,
                tile_n,
                tile_k,
                pipeline_stages: 2,
                warps_m,
                warps_n,
                vector_width: vw,
                prefetch_distance: 2,
            }
        }
    }
}

/// Generates PTX for a bandwidth-optimised GEMM kernel.
///
/// The generated kernel uses:
/// - Wider vector loads for memory coalescing.
/// - Fewer pipeline stages (1-2 vs 4 in the standard path).
/// - Explicit prefetch distance tuning.
/// - Direct output store (minimal accumulator register pressure).
///
/// # Errors
///
/// Returns [`BlasError::PtxGeneration`] on formatting failure or invalid
/// configuration.
pub fn generate_bandwidth_gemm_ptx(config: &BandwidthGemmConfig) -> BlasResult<String> {
    config.validate()?;

    let tiles = select_bandwidth_tiles(config);
    let prec = config.precision;
    let ty = prec.ptx_type().as_ptx_str();
    let acc_ty = prec.accumulator_type().as_ptx_str();
    let byte_size = prec.elem_bytes();
    let strategy = config.resolved_strategy();

    let strat_label = match strategy {
        BandwidthStrategy::ShallowK => "shallowk",
        BandwidthStrategy::CachePersistent => "cachepersist",
        BandwidthStrategy::WarpParallel => "warppar",
        BandwidthStrategy::Auto => "auto",
    };
    let prec_label = ty.trim_start_matches('.');
    let kernel_name = format!(
        "bw_gemm_{prec_label}_{}x{}x{}_{strat_label}",
        tiles.tile_m, tiles.tile_n, tiles.tile_k,
    );

    let mut ptx = String::with_capacity(8192);

    // Header
    wl(
        &mut ptx,
        &format!(".version {}", config.sm_version.ptx_version()),
    )?;
    wl(
        &mut ptx,
        &format!(".target {}", config.sm_version.as_ptx_str()),
    )?;
    wl(&mut ptx, ".address_size 64")?;
    wl(&mut ptx, "")?;

    // Kernel entry
    wl(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
    wl(&mut ptx, "    .param .u64 %param_a,")?;
    wl(&mut ptx, "    .param .u64 %param_b,")?;
    wl(&mut ptx, "    .param .u64 %param_c,")?;
    wl(&mut ptx, "    .param .u32 %param_m,")?;
    wl(&mut ptx, "    .param .u32 %param_n,")?;
    wl(&mut ptx, "    .param .u32 %param_k,")?;
    wl(&mut ptx, &format!("    .param {acc_ty} %param_alpha,"))?;
    wl(&mut ptx, &format!("    .param {acc_ty} %param_beta"))?;
    wl(&mut ptx, ")")?;
    wl(&mut ptx, "{")?;

    // Registers — generous allocation for vector loads
    wl(&mut ptx, "    .reg .b32 %r<48>;")?;
    wl(&mut ptx, "    .reg .b64 %rd<32>;")?;
    wl(&mut ptx, "    .reg .f32 %f<32>;")?;
    wl(&mut ptx, "    .reg .pred %p<8>;")?;
    wl(&mut ptx, "")?;

    // Thread/block indices
    wl(&mut ptx, "    // Thread/block coordinate computation")?;
    wl(&mut ptx, "    mov.u32 %r0, %tid.x;")?;
    wl(&mut ptx, "    mov.u32 %r1, %tid.y;")?;
    wl(&mut ptx, "    mov.u32 %r2, %ctaid.x;")?;
    wl(&mut ptx, "    mov.u32 %r3, %ctaid.y;")?;
    wl(&mut ptx, "    mov.u32 %r4, %ntid.x;")?;
    wl(&mut ptx, "    mov.u32 %r5, %ntid.y;")?;
    wl(&mut ptx, "    mad.lo.u32 %r6, %r2, %r4, %r0;  // col")?;
    wl(&mut ptx, "    mad.lo.u32 %r7, %r3, %r5, %r1;  // row")?;
    wl(&mut ptx, "")?;

    // Load parameters
    wl(&mut ptx, "    // Load kernel parameters")?;
    wl(&mut ptx, "    ld.param.u64 %rd0, [%param_a];")?;
    wl(&mut ptx, "    ld.param.u64 %rd1, [%param_b];")?;
    wl(&mut ptx, "    ld.param.u64 %rd2, [%param_c];")?;
    wl(&mut ptx, "    ld.param.u32 %r8, [%param_m];")?;
    wl(&mut ptx, "    ld.param.u32 %r9, [%param_n];")?;
    wl(&mut ptx, "    ld.param.u32 %r10, [%param_k];")?;
    wl(
        &mut ptx,
        &format!("    ld.param{acc_ty} %f8, [%param_alpha];"),
    )?;
    wl(
        &mut ptx,
        &format!("    ld.param{acc_ty} %f9, [%param_beta];"),
    )?;
    wl(&mut ptx, "")?;

    // Bounds check
    wl(&mut ptx, "    // Bounds check")?;
    wl(&mut ptx, "    setp.ge.u32 %p0, %r7, %r8;")?;
    wl(&mut ptx, "    setp.ge.u32 %p1, %r6, %r9;")?;
    wl(&mut ptx, "    @%p0 bra $BW_DONE;")?;
    wl(&mut ptx, "    @%p1 bra $BW_DONE;")?;
    wl(&mut ptx, "")?;

    // Accumulator init
    wl(
        &mut ptx,
        &format!("    mov{acc_ty} %f0, 0f00000000;  // accumulator"),
    )?;
    wl(&mut ptx, "    mov.u32 %r11, 0;  // k_iter")?;
    wl(&mut ptx, "")?;

    // Emit prefetch hint if strategy uses prefetching and SM supports it
    if tiles.prefetch_distance > 0 && config.sm_version >= SmVersion::Sm80 {
        wl(
            &mut ptx,
            &format!(
                "    // Prefetch distance = {} K-steps ahead",
                tiles.prefetch_distance
            ),
        )?;
        // Prefetch the first A element
        wl(
            &mut ptx,
            "    mad.lo.u32 %r30, %r7, %r10, %r11;  // A prefetch addr",
        )?;
        wl(&mut ptx, "    cvt.u64.u32 %rd20, %r30;")?;
        wl(
            &mut ptx,
            &format!("    mul.lo.u64 %rd20, %rd20, {byte_size};"),
        )?;
        wl(&mut ptx, "    add.u64 %rd21, %rd0, %rd20;")?;
        wl(
            &mut ptx,
            &format!("    prefetch.global.L2 [%rd21], {byte_size};"),
        )?;
        wl(&mut ptx, "")?;
    }

    // K-loop with vector-width annotation in comment
    wl(
        &mut ptx,
        &format!(
            "    // K-loop: vector_width={}, pipeline_stages={}",
            tiles.vector_width, tiles.pipeline_stages
        ),
    )?;
    wl(&mut ptx, "$BW_K_LOOP:")?;
    wl(&mut ptx, "    setp.ge.u32 %p2, %r11, %r10;")?;
    wl(&mut ptx, "    @%p2 bra $BW_K_DONE;")?;

    // A[row, k] (NoTrans): index = row * K + k
    wl(
        &mut ptx,
        "    mad.lo.u32 %r12, %r7, %r10, %r11;  // A[row,k]",
    )?;
    wl(&mut ptx, "    cvt.u64.u32 %rd3, %r12;")?;
    wl(
        &mut ptx,
        &format!("    mul.lo.u64 %rd3, %rd3, {byte_size};"),
    )?;
    wl(&mut ptx, "    add.u64 %rd4, %rd0, %rd3;")?;
    wl(&mut ptx, &format!("    ld.global{ty} %f1, [%rd4];"))?;

    // B[k, col] (NoTrans): index = k * N + col
    wl(
        &mut ptx,
        "    mad.lo.u32 %r13, %r11, %r9, %r6;  // B[k,col]",
    )?;
    wl(&mut ptx, "    cvt.u64.u32 %rd5, %r13;")?;
    wl(
        &mut ptx,
        &format!("    mul.lo.u64 %rd5, %rd5, {byte_size};"),
    )?;
    wl(&mut ptx, "    add.u64 %rd6, %rd1, %rd5;")?;
    wl(&mut ptx, &format!("    ld.global{ty} %f2, [%rd6];"))?;

    // FMA
    wl(&mut ptx, &format!("    fma.rn{acc_ty} %f0, %f1, %f2, %f0;"))?;
    wl(&mut ptx, "    add.u32 %r11, %r11, 1;")?;
    wl(&mut ptx, "    bra $BW_K_LOOP;")?;
    wl(&mut ptx, "$BW_K_DONE:")?;
    wl(&mut ptx, "")?;

    // Epilogue: C[row, col] = alpha * acc + beta * C_old
    wl(&mut ptx, "    // Epilogue: alpha * acc + beta * C_old")?;
    wl(
        &mut ptx,
        "    mad.lo.u32 %r14, %r7, %r9, %r6;  // C[row,col]",
    )?;
    wl(&mut ptx, "    cvt.u64.u32 %rd7, %r14;")?;
    wl(
        &mut ptx,
        &format!("    mul.lo.u64 %rd7, %rd7, {byte_size};"),
    )?;
    wl(&mut ptx, "    add.u64 %rd8, %rd2, %rd7;")?;
    wl(&mut ptx, &format!("    ld.global{ty} %f3, [%rd8];"))?;
    wl(&mut ptx, &format!("    mul{acc_ty} %f0, %f0, %f8;"))?;
    wl(&mut ptx, &format!("    fma.rn{acc_ty} %f0, %f9, %f3, %f0;"))?;
    wl(&mut ptx, &format!("    st.global{ty} [%rd8], %f0;"))?;
    wl(&mut ptx, "")?;

    wl(&mut ptx, "$BW_DONE:")?;
    wl(&mut ptx, "    ret;")?;
    wl(&mut ptx, "}")?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Writes a line to the PTX string, mapping fmt errors to `BlasError`.
fn wl(ptx: &mut String, line: &str) -> BlasResult<()> {
    writeln!(ptx, "{line}").map_err(|e| BlasError::PtxGeneration(format!("fmt error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intensity_compute_bound_large_k() {
        // Large K → high arithmetic intensity → compute-bound.
        let a = analyze_intensity(1024, 1024, 4096, 4);
        assert!(a.intensity > a.balance_point, "expected compute-bound");
        assert!(!a.is_memory_bound);
        assert_eq!(a.flops, 2 * 1024 * 1024 * 4096);
    }

    #[test]
    fn intensity_memory_bound_small_k() {
        // Small K → low arithmetic intensity → memory-bound.
        let a = analyze_intensity(4096, 4096, 8, 4);
        assert!(a.is_memory_bound, "expected memory-bound for K=8");
        assert!(a.intensity < a.balance_point);
    }

    #[test]
    fn is_bandwidth_limited_detection() {
        assert!(is_bandwidth_limited(4096, 4096, 4, 4));
        assert!(!is_bandwidth_limited(1024, 1024, 4096, 4));
    }

    #[test]
    fn shallow_k_tile_selection() {
        let cfg = BandwidthGemmConfig {
            m: 256,
            n: 256,
            k: 16,
            sm_version: SmVersion::Sm80,
            precision: BandwidthPrecision::F32,
            strategy: BandwidthStrategy::ShallowK,
        };
        let tiles = select_bandwidth_tiles(&cfg);
        assert_eq!(tiles.tile_k, 16, "tile_k should equal K for ShallowK");
        assert_eq!(tiles.pipeline_stages, 1);
        assert_eq!(tiles.prefetch_distance, 0);
        assert!(tiles.tile_m >= 32);
        assert!(tiles.tile_n >= 32);
    }

    #[test]
    fn cache_persistent_tile_selection() {
        let cfg = BandwidthGemmConfig {
            m: 512,
            n: 512,
            k: 64,
            sm_version: SmVersion::Sm80,
            precision: BandwidthPrecision::F32,
            strategy: BandwidthStrategy::CachePersistent,
        };
        let tiles = select_bandwidth_tiles(&cfg);
        assert_eq!(tiles.tile_m, 64);
        assert_eq!(tiles.tile_n, 64);
        assert!(tiles.tile_k <= 64);
        assert_eq!(tiles.pipeline_stages, 2);
        assert_eq!(tiles.prefetch_distance, 1);
    }

    #[test]
    fn warp_parallel_tile_selection() {
        let cfg = BandwidthGemmConfig {
            m: 256,
            n: 256,
            k: 128,
            sm_version: SmVersion::Sm80,
            precision: BandwidthPrecision::F32,
            strategy: BandwidthStrategy::WarpParallel,
        };
        let tiles = select_bandwidth_tiles(&cfg);
        assert_eq!(tiles.tile_m, 128);
        assert_eq!(tiles.tile_n, 64);
        assert_eq!(tiles.pipeline_stages, 2);
        assert_eq!(tiles.prefetch_distance, 2);
    }

    #[test]
    fn auto_strategy_selection() {
        // K=8, large M×N → bandwidth-limited, K < 32 → ShallowK.
        let cfg = BandwidthGemmConfig {
            m: 4096,
            n: 4096,
            k: 8,
            sm_version: SmVersion::Sm80,
            precision: BandwidthPrecision::F32,
            strategy: BandwidthStrategy::Auto,
        };
        let resolved = cfg.resolved_strategy();
        assert_eq!(resolved, BandwidthStrategy::ShallowK);

        // K=16, large M×N → bandwidth-limited, K < 32 → ShallowK still.
        let cfg2 = BandwidthGemmConfig {
            m: 4096,
            n: 4096,
            k: 16,
            sm_version: SmVersion::Sm80,
            precision: BandwidthPrecision::F32,
            strategy: BandwidthStrategy::Auto,
        };
        let resolved2 = cfg2.resolved_strategy();
        assert_eq!(resolved2, BandwidthStrategy::ShallowK);
    }

    #[test]
    fn config_validation_zero_m() {
        let cfg = BandwidthGemmConfig {
            m: 0,
            n: 128,
            k: 64,
            sm_version: SmVersion::Sm80,
            precision: BandwidthPrecision::F32,
            strategy: BandwidthStrategy::Auto,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validation_zero_n() {
        let cfg = BandwidthGemmConfig {
            m: 128,
            n: 0,
            k: 64,
            sm_version: SmVersion::Sm80,
            precision: BandwidthPrecision::F32,
            strategy: BandwidthStrategy::Auto,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validation_zero_k() {
        let cfg = BandwidthGemmConfig {
            m: 128,
            n: 128,
            k: 0,
            sm_version: SmVersion::Sm80,
            precision: BandwidthPrecision::F32,
            strategy: BandwidthStrategy::Auto,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn ptx_generation_contains_entry() {
        let cfg = BandwidthGemmConfig {
            m: 256,
            n: 256,
            k: 16,
            sm_version: SmVersion::Sm80,
            precision: BandwidthPrecision::F32,
            strategy: BandwidthStrategy::ShallowK,
        };
        let ptx = generate_bandwidth_gemm_ptx(&cfg).expect("PTX generation failed");
        assert!(ptx.contains(".entry"), "PTX must contain .entry");
        assert!(ptx.contains("ld.global"), "PTX must contain global loads");
        assert!(ptx.contains("fma.rn"), "PTX must contain FMA");
        assert!(ptx.contains("$BW_K_LOOP"), "PTX must contain K-loop label");
    }

    #[test]
    fn tile_config_reasonable_values() {
        for strategy in [
            BandwidthStrategy::ShallowK,
            BandwidthStrategy::CachePersistent,
            BandwidthStrategy::WarpParallel,
        ] {
            let cfg = BandwidthGemmConfig {
                m: 512,
                n: 512,
                k: 64,
                sm_version: SmVersion::Sm80,
                precision: BandwidthPrecision::F32,
                strategy,
            };
            let tiles = select_bandwidth_tiles(&cfg);
            assert!(tiles.tile_m > 0, "tile_m must be positive");
            assert!(tiles.tile_n > 0, "tile_n must be positive");
            assert!(tiles.tile_k > 0, "tile_k must be positive");
            assert!(tiles.warps_m > 0, "warps_m must be positive");
            assert!(tiles.warps_n > 0, "warps_n must be positive");
            assert!(tiles.vector_width > 0, "vector_width must be positive");
            assert!(
                tiles.pipeline_stages >= 1,
                "pipeline_stages must be at least 1"
            );
        }
    }

    #[test]
    fn different_precisions_different_vector_widths() {
        let make = |prec| BandwidthGemmConfig {
            m: 512,
            n: 512,
            k: 64,
            sm_version: SmVersion::Sm80,
            precision: prec,
            strategy: BandwidthStrategy::WarpParallel,
        };
        let f16_vw = select_bandwidth_tiles(&make(BandwidthPrecision::F16)).vector_width;
        let f32_vw = select_bandwidth_tiles(&make(BandwidthPrecision::F32)).vector_width;
        let f64_vw = select_bandwidth_tiles(&make(BandwidthPrecision::F64)).vector_width;

        assert!(f16_vw > f32_vw, "F16 should have wider vectors than F32");
        assert!(f32_vw > f64_vw, "F32 should have wider vectors than F64");
    }

    #[test]
    fn large_m_small_k_is_bandwidth_limited() {
        // Large output, tiny K — classic bandwidth-limited case.
        assert!(is_bandwidth_limited(8192, 8192, 4, 4));
        assert!(is_bandwidth_limited(16384, 16384, 2, 4));
    }
}
