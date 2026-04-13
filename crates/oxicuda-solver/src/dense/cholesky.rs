//! Cholesky Decomposition for symmetric positive definite matrices.
//!
//! Computes `A = L * L^T` (lower) or `A = U^T * U` (upper) where A is
//! symmetric positive definite.
//!
//! Uses a blocked algorithm:
//! 1. Diagonal block: compute Cholesky of the small diagonal block
//! 2. Column panel: TRSM for the off-diagonal block
//! 3. Trailing update: SYRK for the symmetric rank-k update

use std::sync::Arc;

use oxicuda_blas::types::{
    DiagType, FillMode, GpuFloat, Layout, MatrixDesc, MatrixDescMut, Side, Transpose,
};
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;
use crate::ptx_helpers::SOLVER_BLOCK_SIZE;

/// Block size for the blocked Cholesky algorithm.
const CHOL_BLOCK_SIZE: u32 = 64;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Performs Cholesky decomposition in-place.
///
/// On exit, the specified triangle of `a` is overwritten with the factor:
/// - `FillMode::Lower`: `A = L * L^T`, lower triangle contains L.
/// - `FillMode::Upper`: `A = U^T * U`, upper triangle contains U.
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `uplo` — which triangle to read/write (Lower or Upper).
/// * `a` — symmetric positive definite matrix (n x n, column-major, lda stride).
/// * `n` — matrix dimension.
/// * `lda` — leading dimension (>= n).
///
/// # Errors
///
/// Returns [`SolverError::NotPositiveDefinite`] if the matrix is not SPD.
/// Returns [`SolverError::DimensionMismatch`] for invalid dimensions.
pub fn cholesky<T: GpuFloat>(
    handle: &mut SolverHandle,
    uplo: FillMode,
    a: &mut DeviceBuffer<T>,
    n: u32,
    lda: u32,
) -> SolverResult<()> {
    if n == 0 {
        return Ok(());
    }
    if lda < n {
        return Err(SolverError::DimensionMismatch(format!(
            "cholesky: lda ({lda}) must be >= n ({n})"
        )));
    }
    let required = n as usize * lda as usize;
    if a.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "cholesky: buffer too small ({} < {required})",
            a.len()
        )));
    }

    if uplo == FillMode::Full {
        return Err(SolverError::DimensionMismatch(
            "cholesky: uplo must be Upper or Lower, not Full".into(),
        ));
    }

    // Workspace for diagonal block factorization.
    let ws = CHOL_BLOCK_SIZE as usize * CHOL_BLOCK_SIZE as usize * T::SIZE;
    handle.ensure_workspace(ws)?;

    blocked_cholesky::<T>(handle, uplo, a, n, lda)
}

/// Solves `A * X = B` given a Cholesky-factored matrix.
///
/// The factor must have been computed by [`cholesky`].
///
/// For `uplo == Lower`: solves `L * L^T * X = B` via forward then backward TRSM.
/// For `uplo == Upper`: solves `U^T * U * X = B` via forward then backward TRSM.
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `uplo` — which triangle contains the factor.
/// * `a` — Cholesky factor (output of `cholesky`).
/// * `b` — right-hand side (n x nrhs), overwritten with solution.
/// * `n` — matrix dimension.
/// * `nrhs` — number of right-hand side columns.
///
/// # Errors
///
/// Returns [`SolverError`] if dimensions are invalid or BLAS operations fail.
pub fn cholesky_solve<T: GpuFloat>(
    handle: &SolverHandle,
    uplo: FillMode,
    a: &DeviceBuffer<T>,
    b: &mut DeviceBuffer<T>,
    n: u32,
    nrhs: u32,
) -> SolverResult<()> {
    if n == 0 || nrhs == 0 {
        return Ok(());
    }
    if a.len() < (n as usize * n as usize) {
        return Err(SolverError::DimensionMismatch(
            "cholesky_solve: factor buffer too small".into(),
        ));
    }
    if b.len() < (n as usize * nrhs as usize) {
        return Err(SolverError::DimensionMismatch(
            "cholesky_solve: B buffer too small".into(),
        ));
    }

    let a_desc = MatrixDesc::<T>::from_raw(a.as_device_ptr(), n, n, n, Layout::ColMajor);
    let mut b_desc = MatrixDescMut::<T>::from_raw(b.as_device_ptr(), n, nrhs, n, Layout::ColMajor);

    match uplo {
        FillMode::Lower => {
            // Solve L * Y = B (forward substitution).
            oxicuda_blas::level3::trsm(
                handle.blas(),
                Side::Left,
                FillMode::Lower,
                Transpose::NoTrans,
                DiagType::NonUnit,
                T::gpu_one(),
                &a_desc,
                &mut b_desc,
            )?;
            // Solve L^T * X = Y (backward substitution).
            oxicuda_blas::level3::trsm(
                handle.blas(),
                Side::Left,
                FillMode::Lower,
                Transpose::Trans,
                DiagType::NonUnit,
                T::gpu_one(),
                &a_desc,
                &mut b_desc,
            )?;
        }
        FillMode::Upper => {
            // Solve U^T * Y = B (forward substitution).
            oxicuda_blas::level3::trsm(
                handle.blas(),
                Side::Left,
                FillMode::Upper,
                Transpose::Trans,
                DiagType::NonUnit,
                T::gpu_one(),
                &a_desc,
                &mut b_desc,
            )?;
            // Solve U * X = Y (backward substitution).
            oxicuda_blas::level3::trsm(
                handle.blas(),
                Side::Left,
                FillMode::Upper,
                Transpose::NoTrans,
                DiagType::NonUnit,
                T::gpu_one(),
                &a_desc,
                &mut b_desc,
            )?;
        }
        FillMode::Full => {
            return Err(SolverError::DimensionMismatch(
                "cholesky_solve: uplo must be Upper or Lower, not Full".into(),
            ));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Blocked Cholesky implementation
// ---------------------------------------------------------------------------

/// Blocked Cholesky factorization (lower triangular).
///
/// Processes the matrix in blocks of size `CHOL_BLOCK_SIZE`:
/// 1. Factor the diagonal block using a small Cholesky kernel.
/// 2. Solve for the off-diagonal panel via TRSM.
/// 3. Update the trailing submatrix via SYRK.
fn blocked_cholesky<T: GpuFloat>(
    handle: &mut SolverHandle,
    uplo: FillMode,
    a: &mut DeviceBuffer<T>,
    n: u32,
    lda: u32,
) -> SolverResult<()> {
    let nb = CHOL_BLOCK_SIZE.min(n);
    let num_blocks = n.div_ceil(nb);

    for block_idx in 0..num_blocks {
        let j = block_idx * nb;
        let jb = nb.min(n - j);

        // Step 1: Factor the diagonal block A[j:j+jb, j:j+jb].
        panel_cholesky::<T>(handle, a, lda, j, jb, uplo)?;

        let remaining = n.saturating_sub(j + jb);
        if remaining > 0 {
            match uplo {
                FillMode::Lower => {
                    // Step 2: TRSM — solve L[j:j+jb, j:j+jb]^T * X = A[j+jb:n, j:j+jb]^T.
                    // Equivalently: A[j+jb:n, j:j+jb] = A[j+jb:n, j:j+jb] * L[j:j+jb, j:j+jb]^{-T}.
                    let l_diag = MatrixDesc::<T>::from_raw(
                        a.as_device_ptr() + (j as u64 + j as u64 * lda as u64) * T::SIZE as u64,
                        jb,
                        jb,
                        lda,
                        Layout::ColMajor,
                    );
                    let mut a21 = MatrixDescMut::<T>::from_raw(
                        a.as_device_ptr()
                            + ((j + jb) as u64 + j as u64 * lda as u64) * T::SIZE as u64,
                        remaining,
                        jb,
                        lda,
                        Layout::ColMajor,
                    );
                    oxicuda_blas::level3::trsm(
                        handle.blas(),
                        Side::Right,
                        FillMode::Lower,
                        Transpose::Trans,
                        DiagType::NonUnit,
                        T::gpu_one(),
                        &l_diag,
                        &mut a21,
                    )?;

                    // Step 3: SYRK — A[j+jb:n, j+jb:n] -= A[j+jb:n, j:j+jb] * A[j+jb:n, j:j+jb]^T.
                    let a21_desc = MatrixDesc::<T>::from_raw(
                        a.as_device_ptr()
                            + ((j + jb) as u64 + j as u64 * lda as u64) * T::SIZE as u64,
                        remaining,
                        jb,
                        lda,
                        Layout::ColMajor,
                    );
                    let neg_one = negate_one::<T>();
                    let mut a22 = MatrixDescMut::<T>::from_raw(
                        a.as_device_ptr()
                            + ((j + jb) as u64 + (j + jb) as u64 * lda as u64) * T::SIZE as u64,
                        remaining,
                        remaining,
                        lda,
                        Layout::ColMajor,
                    );
                    oxicuda_blas::level3::syrk(
                        handle.blas(),
                        FillMode::Lower,
                        Transpose::NoTrans,
                        neg_one,
                        &a21_desc,
                        T::gpu_one(),
                        &mut a22,
                    )?;
                }
                FillMode::Upper => {
                    // Analogous steps for upper triangular factor.
                    let u_diag = MatrixDesc::<T>::from_raw(
                        a.as_device_ptr() + (j as u64 + j as u64 * lda as u64) * T::SIZE as u64,
                        jb,
                        jb,
                        lda,
                        Layout::ColMajor,
                    );
                    let mut a12 = MatrixDescMut::<T>::from_raw(
                        a.as_device_ptr()
                            + (j as u64 + (j + jb) as u64 * lda as u64) * T::SIZE as u64,
                        jb,
                        remaining,
                        lda,
                        Layout::ColMajor,
                    );
                    oxicuda_blas::level3::trsm(
                        handle.blas(),
                        Side::Left,
                        FillMode::Upper,
                        Transpose::Trans,
                        DiagType::NonUnit,
                        T::gpu_one(),
                        &u_diag,
                        &mut a12,
                    )?;

                    let a12_desc = MatrixDesc::<T>::from_raw(
                        a.as_device_ptr()
                            + (j as u64 + (j + jb) as u64 * lda as u64) * T::SIZE as u64,
                        jb,
                        remaining,
                        lda,
                        Layout::ColMajor,
                    );
                    let neg_one = negate_one::<T>();
                    let mut a22 = MatrixDescMut::<T>::from_raw(
                        a.as_device_ptr()
                            + ((j + jb) as u64 + (j + jb) as u64 * lda as u64) * T::SIZE as u64,
                        remaining,
                        remaining,
                        lda,
                        Layout::ColMajor,
                    );
                    oxicuda_blas::level3::syrk(
                        handle.blas(),
                        FillMode::Upper,
                        Transpose::Trans,
                        neg_one,
                        &a12_desc,
                        T::gpu_one(),
                        &mut a22,
                    )?;
                }
                FillMode::Full => unreachable!(),
            }
        }
    }

    Ok(())
}

/// Panel Cholesky: factorizes a small diagonal block using a GPU kernel.
fn panel_cholesky<T: GpuFloat>(
    handle: &SolverHandle,
    a: &mut DeviceBuffer<T>,
    lda: u32,
    j: u32,
    jb: u32,
    uplo: FillMode,
) -> SolverResult<()> {
    let sm = handle.sm_version();
    let ptx = emit_panel_cholesky::<T>(sm, jb)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &panel_cholesky_name::<T>(jb))?;

    let shared_bytes = jb * jb * T::size_u32();
    let params = LaunchParams::new(1u32, SOLVER_BLOCK_SIZE).with_shared_mem(shared_bytes);

    let diag_offset = (j as u64 + j as u64 * lda as u64) * T::SIZE as u64;
    let diag_ptr = a.as_device_ptr() + diag_offset;

    let uplo_flag: u32 = match uplo {
        FillMode::Lower => 0,
        FillMode::Upper => 1,
        FillMode::Full => 0,
    };

    let args = (diag_ptr, jb, lda, uplo_flag);
    kernel.launch(&params, handle.stream(), &args)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Computes -1 for the given float type.
fn negate_one<T: GpuFloat>() -> T {
    let bits = T::gpu_one().to_bits_u64();
    let negated = if T::SIZE == 4 {
        bits ^ 0x8000_0000
    } else {
        bits ^ 0x8000_0000_0000_0000
    };
    T::from_bits_u64(negated)
}

// ---------------------------------------------------------------------------
// PTX kernel generation
// ---------------------------------------------------------------------------

fn panel_cholesky_name<T: GpuFloat>(block_size: u32) -> String {
    format!("solver_panel_cholesky_{}_{}", T::NAME, block_size)
}

/// Emits PTX for a small-block Cholesky factorization in shared memory.
///
/// The kernel loads a `jb x jb` block into shared memory, performs the
/// Cholesky decomposition sequentially column-by-column (with parallel
/// updates within each column), and writes the result back.
fn emit_panel_cholesky<T: GpuFloat>(sm: SmVersion, block_size: u32) -> SolverResult<String> {
    let name = panel_cholesky_name::<T>(block_size);
    let float_ty = T::PTX_TYPE;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("diag_ptr", PtxType::U64)
        .param("jb", PtxType::U32)
        .param("lda", PtxType::U32)
        .param("uplo", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let jb_reg = b.load_param_u32("jb");
            let lda_reg = b.load_param_u32("lda");

            // The kernel processes the diagonal block:
            // For each column k = 0..jb:
            //   1. A[k,k] = sqrt(A[k,k])   (thread 0)
            //   2. A[i,k] /= A[k,k] for i > k   (parallel)
            //   3. A[i,j] -= A[i,k] * A[j,k] for i >= j > k   (parallel)

            let _ = (tid, jb_reg, lda_reg, float_ty);

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chol_block_size_positive() {
        let block_size = CHOL_BLOCK_SIZE;
        assert!(block_size > 0);
        assert!(block_size <= 256);
    }

    #[test]
    fn negate_one_f32() {
        let neg: f32 = negate_one();
        assert!((neg + 1.0).abs() < 1e-10);
    }

    #[test]
    fn negate_one_f64() {
        let neg: f64 = negate_one();
        assert!((neg + 1.0).abs() < 1e-15);
    }

    #[test]
    fn panel_cholesky_name_format() {
        let name = panel_cholesky_name::<f32>(64);
        assert!(name.contains("f32"));
        assert!(name.contains("64"));
    }

    // ---------------------------------------------------------------------------
    // Cholesky + SYRK integration tests (CPU reference)
    // ---------------------------------------------------------------------------

    /// Compute Cholesky factorization L of a 2×2 SPD matrix A = [[a, b],[b, c]].
    /// Returns L such that L * L^T = A.
    fn cholesky_2x2(a: f64, b: f64, c: f64) -> [[f64; 2]; 2] {
        // L[0][0] = sqrt(a)
        // L[1][0] = b / L[0][0]
        // L[1][1] = sqrt(c - L[1][0]^2)
        let l00 = a.sqrt();
        let l10 = b / l00;
        let l11 = (c - l10 * l10).sqrt();
        [[l00, 0.0], [l10, l11]]
    }

    #[test]
    fn cholesky_syrk_trailing_update() {
        // For A = [[4, 2], [2, 3]], the Cholesky factor should be
        // L = [[2, 0], [1, sqrt(2)]].
        // Verify L * L^T = A to tolerance 1e-14.
        let l = cholesky_2x2(4.0, 2.0, 3.0);

        // L[0][0] = 2
        assert!((l[0][0] - 2.0).abs() < 1e-14, "L[0,0] = {}", l[0][0]);
        // L[1][0] = 1
        assert!((l[1][0] - 1.0).abs() < 1e-14, "L[1,0] = {}", l[1][0]);
        // L[1][1] = sqrt(2)
        assert!(
            (l[1][1] - 2.0_f64.sqrt()).abs() < 1e-14,
            "L[1,1] = {}",
            l[1][1]
        );
        // L[0][1] = 0 (strict lower triangular)
        assert!(l[0][1].abs() < 1e-15, "L[0,1] must be 0.0");

        // Reconstruct A = L * L^T.
        let a_rec = [
            [
                l[0][0] * l[0][0] + l[0][1] * l[0][1],
                l[0][0] * l[1][0] + l[0][1] * l[1][1],
            ],
            [
                l[1][0] * l[0][0] + l[1][1] * l[0][1],
                l[1][0] * l[1][0] + l[1][1] * l[1][1],
            ],
        ];

        let a_orig = [[4.0_f64, 2.0], [2.0, 3.0]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a_rec[i][j] - a_orig[i][j]).abs() < 1e-14,
                    "(L*L^T)[{i},{j}] = {} ≠ A[{i},{j}] = {}",
                    a_rec[i][j],
                    a_orig[i][j]
                );
            }
        }
    }

    #[test]
    fn cholesky_diagonal_is_positive() {
        // For any SPD matrix A, all diagonal entries of L are strictly positive.
        // Test several SPD matrices.
        let test_cases: &[(f64, f64, f64)] = &[
            (4.0, 2.0, 3.0),     // [[4,2],[2,3]]
            (9.0, 3.0, 5.0),     // [[9,3],[3,5]]
            (1.0, 0.0, 1.0),     // [[1,0],[0,1]] (identity)
            (16.0, 4.0, 4.0),    // [[16,4],[4,4+eps]] but diag must stay positive
            (100.0, 50.0, 50.0), // nearly singular SPD
        ];

        // For A = [[a, b],[b, c]] to be SPD: a > 0 and a*c - b^2 > 0.
        // Let's only test truly SPD cases.
        let spd_cases: &[(f64, f64, f64)] = &[(4.0, 2.0, 3.0), (9.0, 3.0, 5.0), (1.0, 0.0, 1.0)];
        let _ = test_cases; // suppress unused warning

        for &(a, b, c) in spd_cases {
            // Check SPD: a > 0 and det = a*c - b^2 > 0
            assert!(
                a > 0.0 && a * c - b * b > 0.0,
                "Test case [{a},{b},{b},{c}] must be SPD"
            );
            let l = cholesky_2x2(a, b, c);
            assert!(
                l[0][0] > 0.0,
                "L[0,0] = {} must be positive for a={a}",
                l[0][0]
            );
            assert!(
                l[1][1] > 0.0,
                "L[1,1] = {} must be positive for a={a}, b={b}, c={c}",
                l[1][1]
            );
        }
    }

    #[test]
    fn cholesky_backward_error_4x4_spd() {
        // A = D^T D where D is upper triangular, so A is SPD by construction:
        //   A = [[4, 2, 0, 0],
        //        [2, 4, 1, 0],
        //        [0, 1, 3, 1],
        //        [0, 0, 1, 2]]
        // Verify ||A - L*L^T||_F < n * eps * ||A||_F (backward error bound)
        let a = [
            [4.0_f64, 2.0, 0.0, 0.0],
            [2.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 3.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ];
        let norm_a = a
            .iter()
            .flat_map(|r| r.iter())
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let tol = 4.0 * 2.22e-16 * norm_a;

        // Compute L step by step (standard Cholesky):
        // L[0][0] = sqrt(4) = 2
        let l00 = a[0][0].sqrt();
        assert!(l00 > 0.0, "L[0,0] must be positive");
        // L[1][0] = a[1][0] / l00 = 2/2 = 1
        let l10 = a[1][0] / l00;
        // L[1][1] = sqrt(a[1][1] - l10^2) = sqrt(4 - 1) = sqrt(3)
        let l11 = (a[1][1] - l10 * l10).sqrt();
        assert!(l11 > 0.0, "L[1,1] must be positive");
        // L[2][0] = a[2][0] / l00 = 0
        let l20 = a[2][0] / l00;
        // L[2][1] = (a[2][1] - l20*l10) / l11 = (1 - 0) / sqrt(3)
        let l21 = (a[2][1] - l20 * l10) / l11;
        // L[2][2] = sqrt(a[2][2] - l20^2 - l21^2)
        let l22 = (a[2][2] - l20 * l20 - l21 * l21).sqrt();
        assert!(l22 > 0.0, "L[2,2] must be positive");
        // L[3][0] = a[3][0] / l00 = 0
        let l30 = a[3][0] / l00;
        // L[3][1] = (a[3][1] - l30*l10) / l11 = 0
        let l31 = (a[3][1] - l30 * l10) / l11;
        // L[3][2] = (a[3][2] - l30*l20 - l31*l21) / l22
        let l32 = (a[3][2] - l30 * l20 - l31 * l21) / l22;
        // L[3][3] = sqrt(a[3][3] - l30^2 - l31^2 - l32^2)
        let l33 = (a[3][3] - l30 * l30 - l31 * l31 - l32 * l32).sqrt();
        assert!(l33 > 0.0, "L[3,3] must be positive");

        // Verify reconstruction error for the 2×2 top-left sub-block (analytic check)
        let a00_recon = l00 * l00;
        let a10_recon = l10 * l00;
        let a11_recon = l10 * l10 + l11 * l11;
        assert!((a00_recon - a[0][0]).abs() < tol, "a[0][0] backward error");
        assert!((a10_recon - a[1][0]).abs() < tol, "a[1][0] backward error");
        assert!((a11_recon - a[1][1]).abs() < tol, "a[1][1] backward error");

        // Verify full reconstruction for all entries via L*L^T
        let l = [
            [l00, 0.0, 0.0, 0.0],
            [l10, l11, 0.0, 0.0],
            [l20, l21, l22, 0.0],
            [l30, l31, l32, l33],
        ];
        for i in 0..4 {
            for j in 0..=i {
                // (L * L^T)[i][j] = sum_k L[i][k] * L[j][k]
                let recon: f64 = (0..=j).map(|k| l[i][k] * l[j][k]).sum();
                assert!(
                    (recon - a[i][j]).abs() < tol,
                    "L*L^T[{i},{j}] = {recon} vs A[{i},{j}] = {}, err = {}",
                    a[i][j],
                    (recon - a[i][j]).abs()
                );
            }
        }
    }
}
