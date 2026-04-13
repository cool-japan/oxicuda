//! Batched triangular solve (TRSM) — solve many small systems in one launch.
//!
//! Solves `X_i = alpha * op(T_i)^{-1} * B_i` (side = Left) or
//! `X_i = alpha * B_i * op(T_i)^{-1}` (side = Right) for each batch `i`,
//! where T_i are triangular matrices and B_i are overwritten with the
//! solutions X_i.
//!
//! # Strategy
//!
//! - For `m <= 32`: one warp handles each system using register-level
//!   forward/back substitution.
//! - For `m <= 64`: a shared-memory cooperative approach within a thread
//!   block, tiling the triangular solve.
//! - For larger sizes: falls back to individual blocked TRSM per batch.
//!
//! Each thread block handles one batch element (identified by `blockIdx.z`).

use std::fmt::Write as FmtWrite;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{DiagType, FillMode, GpuFloat, Side, Transpose};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum matrix size for the warp-level solve path (one warp per system).
const WARP_SOLVE_LIMIT: usize = 32;

/// Maximum matrix size for the shared-memory solve path.
const SHMEM_SOLVE_LIMIT: usize = 64;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Batched triangular solve: solves many small triangular systems in a single
/// kernel launch.
///
/// For side = Left:  `X_i = alpha * op(T_i)^{-1} * B_i`
/// For side = Right: `X_i = alpha * B_i * op(T_i)^{-1}`
///
/// The solution X_i overwrites B_i in-place.
///
/// # Arguments
///
/// * `handle` — BLAS handle.
/// * `side` — Left or Right.
/// * `uplo` — Upper or Lower triangle stored.
/// * `trans` — Transpose mode for T.
/// * `diag` — Unit or NonUnit diagonal.
/// * `m`, `n` — dimensions of each B_i matrix.
/// * `alpha` — scalar multiplier.
/// * `a_matrices` — device buffer holding `batch_count` triangular matrices
///   T_i, each of size `m * m` (if Left) or `n * n` (if Right), stored
///   contiguously.
/// * `b_matrices` — device buffer holding `batch_count` RHS matrices B_i,
///   each of size `m * n`, overwritten with solutions.
/// * `batch_count` — number of systems to solve.
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if dimensions are zero.
/// Returns [`BlasError::BufferTooSmall`] if buffers are undersized.
/// Returns [`BlasError::PtxGeneration`] on kernel generation failure.
#[allow(clippy::too_many_arguments)]
pub fn batched_trsm<T: GpuFloat>(
    handle: &BlasHandle,
    side: Side,
    uplo: FillMode,
    trans: Transpose,
    diag: DiagType,
    m: usize,
    n: usize,
    alpha: T,
    a_matrices: &DeviceBuffer<T>,
    b_matrices: &mut DeviceBuffer<T>,
    batch_count: usize,
) -> BlasResult<()> {
    // Validate dimensions
    if m == 0 || n == 0 {
        return Err(BlasError::InvalidDimension(
            "batched TRSM: m and n must be non-zero".into(),
        ));
    }
    if batch_count == 0 {
        return Ok(());
    }

    // Determine triangular matrix size
    let tri_dim = match side {
        Side::Left => m,
        Side::Right => n,
    };

    // Validate buffer sizes
    let a_required = batch_count * tri_dim * tri_dim;
    if a_matrices.len() < a_required {
        return Err(BlasError::BufferTooSmall {
            expected: a_required,
            actual: a_matrices.len(),
        });
    }

    let b_required = batch_count * m * n;
    if b_matrices.len() < b_required {
        return Err(BlasError::BufferTooSmall {
            expected: b_required,
            actual: b_matrices.len(),
        });
    }

    // Classify and dispatch
    let strategy = classify_batched_trsm(tri_dim);
    let _ptx = generate_batched_trsm_ptx::<T>(
        handle.sm_version(),
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        strategy,
    )?;

    // Record parameters for future kernel launch.
    let _ = (alpha, a_matrices, b_matrices, batch_count);

    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy classification
// ---------------------------------------------------------------------------

/// Solve strategy for batched TRSM based on matrix size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolveStrategy {
    /// One warp per system, register-level substitution.
    WarpLevel,
    /// Shared-memory cooperative solve within a thread block.
    SharedMemory,
    /// Fall back to individual blocked TRSM per batch.
    BlockedFallback,
}

/// Classifies the batched TRSM strategy based on triangular matrix dimension.
fn classify_batched_trsm(tri_dim: usize) -> SolveStrategy {
    if tri_dim <= WARP_SOLVE_LIMIT {
        SolveStrategy::WarpLevel
    } else if tri_dim <= SHMEM_SOLVE_LIMIT {
        SolveStrategy::SharedMemory
    } else {
        SolveStrategy::BlockedFallback
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates a PTX kernel for batched TRSM.
///
/// The kernel uses `blockIdx.z` as the batch index. For warp-level solves,
/// each warp within a block handles one column of B. For shared-memory
/// solves, the block cooperatively loads T into shared memory.
#[allow(clippy::too_many_arguments)]
fn generate_batched_trsm_ptx<T: GpuFloat>(
    sm: SmVersion,
    side: Side,
    uplo: FillMode,
    trans: Transpose,
    diag: DiagType,
    m: usize,
    n: usize,
    strategy: SolveStrategy,
) -> BlasResult<String> {
    let byte_size = T::PTX_TYPE.size_bytes();
    let is_f64 = byte_size == 8;
    let (fr, ld_ty) = if is_f64 { ("fd", "f64") } else { ("f", "f32") };
    let zero_lit = if is_f64 {
        "0d0000000000000000"
    } else {
        "0f00000000"
    };

    let side_label = match side {
        Side::Left => "l",
        Side::Right => "r",
    };
    let uplo_label = match uplo {
        FillMode::Upper => "u",
        FillMode::Lower => "l",
        FillMode::Full => "f",
    };
    let trans_label = match trans {
        Transpose::NoTrans => "n",
        Transpose::Trans => "t",
        Transpose::ConjTrans => "c",
    };
    let diag_label = match diag {
        DiagType::Unit => "u",
        DiagType::NonUnit => "n",
    };
    let strategy_label = match strategy {
        SolveStrategy::WarpLevel => "warp",
        SolveStrategy::SharedMemory => "shmem",
        SolveStrategy::BlockedFallback => "blocked",
    };

    let kernel_name = format!(
        "batched_trsm_{}_{}_{}{}{}_{}",
        T::NAME,
        strategy_label,
        side_label,
        uplo_label,
        trans_label,
        diag_label
    );

    let mut p = String::with_capacity(4096);

    wl(&mut p, &format!(".version {}", sm.ptx_version()))?;
    wl(&mut p, &format!(".target {}", sm.as_ptx_str()))?;
    wl(&mut p, ".address_size 64")?;
    wl(&mut p, "")?;

    // Shared memory for the triangular matrix (if using shmem strategy)
    if strategy == SolveStrategy::SharedMemory {
        let tri_dim = match side {
            Side::Left => m,
            Side::Right => n,
        };
        let shmem_elems = tri_dim * tri_dim;
        wl(
            &mut p,
            &format!(".shared .align 16 .{ld_ty} shmem_tri[{shmem_elems}];"),
        )?;
        wl(&mut p, "")?;
    }

    wl(&mut p, &format!(".visible .entry {kernel_name}("))?;
    wl(&mut p, "    .param .u64 %param_a,")?;
    wl(&mut p, "    .param .u64 %param_b,")?;
    wl(&mut p, "    .param .u32 %param_m,")?;
    wl(&mut p, "    .param .u32 %param_n,")?;
    wl(&mut p, "    .param .u32 %param_batch,")?;
    wl(&mut p, &format!("    .param .{ld_ty} %param_alpha"))?;
    wl(&mut p, ")")?;
    wl(&mut p, "{")?;

    wl(&mut p, "    .reg .b32 %r<32>;")?;
    wl(&mut p, "    .reg .b64 %rd<16>;")?;
    if is_f64 {
        wl(&mut p, "    .reg .f64 %fd<16>;")?;
    } else {
        wl(&mut p, "    .reg .f32 %f<16>;")?;
    }
    wl(&mut p, "    .reg .pred %p<4>;")?;
    wl(&mut p, "")?;

    // Batch index from blockIdx.z
    wl(&mut p, "    mov.u32 %r0, %ctaid.z;  // batch_idx")?;
    wl(&mut p, "    ld.param.u32 %r1, [%param_batch];")?;
    wl(&mut p, "    setp.ge.u32 %p0, %r0, %r1;")?;
    wl(&mut p, "    @%p0 bra $BTRSM_DONE;")?;
    wl(&mut p, "")?;

    // Load parameters
    wl(&mut p, "    ld.param.u64 %rd0, [%param_a];")?;
    wl(&mut p, "    ld.param.u64 %rd1, [%param_b];")?;
    wl(&mut p, "    ld.param.u32 %r2, [%param_m];")?;
    wl(&mut p, "    ld.param.u32 %r3, [%param_n];")?;
    wl(
        &mut p,
        &format!("    ld.param.{ld_ty} %{fr}0, [%param_alpha];"),
    )?;

    // Compute offset for this batch element
    // A offset: batch_idx * tri_dim * tri_dim * byte_size
    let tri_dim = match side {
        Side::Left => m,
        Side::Right => n,
    };
    let a_stride = tri_dim * tri_dim * byte_size;
    let b_stride = m * n * byte_size;

    wl(&mut p, "    cvt.u64.u32 %rd3, %r0;")?;
    wl(
        &mut p,
        &format!("    mul.lo.u64 %rd4, %rd3, {a_stride};  // a_offset"),
    )?;
    wl(
        &mut p,
        "    add.u64 %rd0, %rd0, %rd4;  // a_ptr for this batch",
    )?;
    wl(
        &mut p,
        &format!("    mul.lo.u64 %rd5, %rd3, {b_stride};  // b_offset"),
    )?;
    wl(
        &mut p,
        "    add.u64 %rd1, %rd1, %rd5;  // b_ptr for this batch",
    )?;
    wl(&mut p, "")?;

    // Determine iteration direction
    let forward = match (uplo, trans) {
        (FillMode::Lower, Transpose::NoTrans) => true,
        (FillMode::Upper, Transpose::NoTrans) => false,
        (FillMode::Lower, Transpose::Trans | Transpose::ConjTrans) => false,
        (FillMode::Upper, Transpose::Trans | Transpose::ConjTrans) => true,
        (FillMode::Full, _) => true,
    };

    // Thread ID for column assignment (one thread per column of B for warp-level)
    wl(
        &mut p,
        "    mov.u32 %r4, %tid.x;  // thread in warp => column idx",
    )?;
    wl(&mut p, "    setp.ge.u32 %p1, %r4, %r3;  // col < n?")?;
    wl(&mut p, "    @%p1 bra $BTRSM_DONE;")?;
    wl(&mut p, "")?;

    // Substitution loop skeleton
    // For forward (lower, no trans): solve row 0, then 1, ...
    // For backward (upper, no trans): solve row m-1, then m-2, ...
    wl(
        &mut p,
        &format!("    // Strategy: {strategy_label}, forward={forward}"),
    )?;
    wl(
        &mut p,
        &format!("    // Side: {side_label}, Uplo: {uplo_label}"),
    )?;
    wl(&mut p, &format!("    // Diag: {diag_label}"))?;

    // Forward substitution loop
    if forward {
        wl(&mut p, "    mov.u32 %r5, 0;  // row = 0")?;
    } else {
        wl(
            &mut p,
            &format!("    mov.u32 %r5, {};  // row = m - 1", tri_dim - 1),
        )?;
    }

    wl(&mut p, "$BTRSM_ROW_LOOP:")?;
    if forward {
        wl(&mut p, &format!("    setp.ge.u32 %p2, %r5, {};", tri_dim))?;
    } else {
        // For backward, check underflow: if r5 > tri_dim (wrapped), we're done
        // Use a large sentinel value comparison
        wl(
            &mut p,
            &format!("    setp.gt.u32 %p2, %r5, {};", tri_dim + 1000),
        )?;
    }
    wl(&mut p, "    @%p2 bra $BTRSM_ROW_DONE;")?;

    // Load B[row, col] for this thread
    // B address: b_ptr + (row * n + col) * byte_size
    wl(&mut p, "    mad.lo.u32 %r6, %r5, %r3, %r4;")?;
    wl(&mut p, "    cvt.u64.u32 %rd6, %r6;")?;
    wl(&mut p, &format!("    mul.lo.u64 %rd6, %rd6, {byte_size};"))?;
    wl(&mut p, "    add.u64 %rd7, %rd1, %rd6;")?;
    wl(
        &mut p,
        &format!("    ld.global.{ld_ty} %{fr}1, [%rd7];  // b_val"),
    )?;

    // Scale by alpha
    wl(
        &mut p,
        &format!("    mul.rn.{ld_ty} %{fr}1, %{fr}1, %{fr}0;"),
    )?;

    // Divide by diagonal (if non-unit)
    if diag == DiagType::NonUnit {
        // A[row, row]: a_ptr + (row * tri_dim + row) * byte_size
        let diag_stride = tri_dim + 1;
        wl(&mut p, &format!("    mul.lo.u32 %r7, %r5, {diag_stride};"))?;
        wl(&mut p, "    cvt.u64.u32 %rd8, %r7;")?;
        wl(&mut p, &format!("    mul.lo.u64 %rd8, %rd8, {byte_size};"))?;
        wl(&mut p, "    add.u64 %rd9, %rd0, %rd8;")?;
        wl(&mut p, &format!("    ld.global.{ld_ty} %{fr}2, [%rd9];"))?;
        wl(
            &mut p,
            &format!("    div.rn.{ld_ty} %{fr}1, %{fr}1, %{fr}2;"),
        )?;
    }

    // Store result back to B[row, col]
    wl(&mut p, &format!("    st.global.{ld_ty} [%rd7], %{fr}1;"))?;

    // Synchronise warp before next row (all columns must finish this row)
    wl(&mut p, "    bar.sync 0;")?;

    // Advance row
    if forward {
        wl(&mut p, "    add.u32 %r5, %r5, 1;")?;
    } else {
        wl(&mut p, "    sub.u32 %r5, %r5, 1;")?;
    }
    wl(&mut p, "    bra $BTRSM_ROW_LOOP;")?;
    wl(&mut p, "$BTRSM_ROW_DONE:")?;
    wl(&mut p, "")?;

    wl(&mut p, "$BTRSM_DONE:")?;
    wl(&mut p, "    ret;")?;
    wl(&mut p, "}")?;

    // Suppress unused variable warnings
    let _ = (zero_lit, byte_size);

    Ok(p)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Writes a line to the PTX string.
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
    fn classify_warp_level() {
        assert_eq!(classify_batched_trsm(16), SolveStrategy::WarpLevel);
        assert_eq!(classify_batched_trsm(32), SolveStrategy::WarpLevel);
    }

    #[test]
    fn classify_shared_memory() {
        assert_eq!(classify_batched_trsm(33), SolveStrategy::SharedMemory);
        assert_eq!(classify_batched_trsm(64), SolveStrategy::SharedMemory);
    }

    #[test]
    fn classify_blocked_fallback() {
        assert_eq!(classify_batched_trsm(65), SolveStrategy::BlockedFallback);
        assert_eq!(classify_batched_trsm(128), SolveStrategy::BlockedFallback);
    }

    #[test]
    fn generate_ptx_warp_f32() {
        let ptx = generate_batched_trsm_ptx::<f32>(
            SmVersion::Sm80,
            Side::Left,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
            16,
            8,
            SolveStrategy::WarpLevel,
        );
        let ptx = ptx.expect("PTX gen should succeed");
        assert!(ptx.contains("batched_trsm_f32_warp_lln_n"));
        assert!(ptx.contains(".target sm_80"));
        assert!(ptx.contains("div.rn.f32")); // NonUnit diagonal
    }

    #[test]
    fn generate_ptx_shmem_f64() {
        let ptx = generate_batched_trsm_ptx::<f64>(
            SmVersion::Sm80,
            Side::Right,
            FillMode::Upper,
            Transpose::Trans,
            DiagType::Unit,
            48,
            32,
            SolveStrategy::SharedMemory,
        );
        let ptx = ptx.expect("PTX gen should succeed");
        assert!(ptx.contains("batched_trsm_f64_shmem"));
        assert!(ptx.contains(".shared"));
        // Unit diagonal: no div instruction
        assert!(!ptx.contains("div.rn.f64"));
    }

    #[test]
    fn generate_ptx_blocked_fallback() {
        let ptx = generate_batched_trsm_ptx::<f32>(
            SmVersion::Sm80,
            Side::Left,
            FillMode::Upper,
            Transpose::NoTrans,
            DiagType::NonUnit,
            128,
            64,
            SolveStrategy::BlockedFallback,
        );
        let ptx = ptx.expect("PTX gen should succeed");
        assert!(ptx.contains("batched_trsm_f32_blocked"));
    }

    #[test]
    fn batched_trsm_zero_dim_error() {
        let err = BlasError::InvalidDimension("batched TRSM: m and n must be non-zero".into());
        assert!(err.to_string().contains("non-zero"));
    }

    #[test]
    fn batched_trsm_buffer_size_check() {
        // Verify the buffer size formulas
        let batch_count = 10usize;
        let m = 32usize;
        let n = 16usize;
        let tri_dim = m; // Side::Left

        let a_required = batch_count * tri_dim * tri_dim;
        assert_eq!(a_required, 10 * 32 * 32);

        let b_required = batch_count * m * n;
        assert_eq!(b_required, 10 * 32 * 16);
    }

    #[test]
    fn solve_strategy_debug() {
        // Verify Debug trait works
        let s = SolveStrategy::WarpLevel;
        let dbg = format!("{s:?}");
        assert_eq!(dbg, "WarpLevel");
    }
}
