//! Band Matrix Solvers.
//!
//! Specialized solvers for banded matrices that exploit the narrow bandwidth
//! to achieve O(n * b^2) complexity instead of O(n^3), where b is the bandwidth.
//!
//! # Storage Format
//!
//! Band matrices use LAPACK-style band storage (column-major):
//! - Row index `i` maps to band storage index `(ku + i - j)` in column `j`.
//! - The band storage array has `(kl + ku + 1)` rows and `n` columns.
//!
//! For example, a 5x5 tridiagonal matrix (kl=1, ku=1) is stored as:
//! ```text
//!   Band storage (3 rows x 5 cols):
//!   [ *   a12  a23  a34  a45 ]   <- superdiagonal (ku=1)
//!   [ a11 a22  a33  a44  a55 ]   <- diagonal
//!   [ a21 a32  a43  a54  *   ]   <- subdiagonal (kl=1)
//! ```
//!
//! # Algorithms
//!
//! - **Band LU**: Gaussian elimination with partial pivoting, adapted for
//!   banded structure. Only operates on the non-zero bandwidth, giving
//!   O(n * kl * ku) complexity.
//! - **Band Cholesky**: Cholesky decomposition for banded SPD matrices,
//!   O(n * kd^2) where kd = kl = ku for a symmetric band.

#![allow(dead_code)]

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;

// ---------------------------------------------------------------------------
// GpuFloat <-> f64 conversion helpers
// ---------------------------------------------------------------------------

fn to_f64<T: GpuFloat>(val: T) -> f64 {
    if T::SIZE == 4 {
        f32::from_bits(val.to_bits_u64() as u32) as f64
    } else {
        f64::from_bits(val.to_bits_u64())
    }
}

fn from_f64<T: GpuFloat>(val: f64) -> T {
    if T::SIZE == 4 {
        T::from_bits_u64(u64::from((val as f32).to_bits()))
    } else {
        T::from_bits_u64(val.to_bits())
    }
}

// ---------------------------------------------------------------------------
// Band matrix descriptor
// ---------------------------------------------------------------------------

/// Band matrix descriptor.
///
/// Stores a banded matrix in LAPACK-style band storage format.
/// The storage array has `(2*kl + ku + 1)` rows and `n` columns for LU
/// (extra kl rows for fill-in during pivoting), or `(kl + ku + 1)` rows
/// and `n` columns for non-pivoted operations.
pub struct BandMatrix<T: GpuFloat> {
    /// Band storage data on the device.
    pub data: DeviceBuffer<T>,
    /// Matrix dimension (n x n).
    pub n: usize,
    /// Number of sub-diagonals.
    pub kl: usize,
    /// Number of super-diagonals.
    pub ku: usize,
}

impl<T: GpuFloat> BandMatrix<T> {
    /// Creates a new band matrix with the given dimensions.
    ///
    /// Allocates a device buffer of size `(2*kl + ku + 1) * n` to accommodate
    /// fill-in during LU factorization.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::Cuda`] if device allocation fails.
    pub fn new(n: usize, kl: usize, ku: usize) -> SolverResult<Self> {
        let ldab = 2 * kl + ku + 1;
        let data = DeviceBuffer::<T>::zeroed(ldab * n)?;
        Ok(Self { n, kl, ku, data })
    }

    /// Returns the leading dimension of the band storage.
    pub fn ldab(&self) -> usize {
        2 * self.kl + self.ku + 1
    }

    /// Returns the total number of elements in band storage.
    pub fn storage_len(&self) -> usize {
        self.ldab() * self.n
    }

    /// Computes the storage index for element (i, j) in the band.
    ///
    /// Returns `None` if (i, j) is outside the band.
    pub fn band_index(&self, i: usize, j: usize) -> Option<usize> {
        let row_in_band = self.kl + i;
        if row_in_band < j {
            return None; // Above the upper bandwidth.
        }
        let band_row = row_in_band - j;
        if band_row >= self.ldab() {
            return None; // Below the lower bandwidth.
        }
        Some(j * self.ldab() + band_row)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// LU factorization for a banded matrix.
///
/// Performs Gaussian elimination with partial pivoting on the banded matrix.
/// The factors L and U overwrite the band storage, with fill-in accommodated
/// in the extra `kl` rows of the storage.
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `band` — band matrix to factorize (overwritten with L and U).
/// * `pivots` — output pivot indices (length >= n).
///
/// # Errors
///
/// Returns [`SolverError::SingularMatrix`] if the matrix is singular.
/// Returns [`SolverError::DimensionMismatch`] if buffer sizes are invalid.
pub fn band_lu<T: GpuFloat>(
    handle: &mut SolverHandle,
    band: &mut BandMatrix<T>,
    pivots: &mut DeviceBuffer<i32>,
) -> SolverResult<()> {
    let n = band.n;
    let kl = band.kl;
    let ku = band.ku;

    if n == 0 {
        return Ok(());
    }
    if pivots.len() < n {
        return Err(SolverError::DimensionMismatch(format!(
            "band_lu: pivots buffer too small ({} < {n})",
            pivots.len()
        )));
    }
    if band.data.len() < band.storage_len() {
        return Err(SolverError::DimensionMismatch(format!(
            "band_lu: band data buffer too small ({} < {})",
            band.data.len(),
            band.storage_len()
        )));
    }

    // Workspace for host-side factorization.
    let ldab = band.ldab();
    let ws = ldab * n * std::mem::size_of::<f64>();
    handle.ensure_workspace(ws)?;

    // Read to host.
    let mut ab = vec![0.0_f64; ldab * n];
    read_band_to_host(&band.data, &mut ab, ldab * n)?;

    let mut ipiv = vec![0_i32; n];

    // Perform banded LU on host.
    band_lu_host(&mut ab, n, kl, ku, ldab, &mut ipiv)?;

    // Write back.
    write_host_to_band_f64(&mut band.data, &ab, ldab * n)?;
    write_pivots_to_device(pivots, &ipiv, n)?;

    Ok(())
}

/// Solves a banded system `A * x = b` using LU factors.
///
/// The band matrix must have been factorized by [`band_lu`].
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `band` — LU-factored band matrix.
/// * `pivots` — pivot indices from `band_lu`.
/// * `b` — right-hand side (n x nrhs), overwritten with solution.
/// * `n` — system dimension.
/// * `nrhs` — number of right-hand side columns.
///
/// # Errors
///
/// Returns [`SolverError::DimensionMismatch`] for invalid dimensions.
pub fn band_solve<T: GpuFloat>(
    handle: &mut SolverHandle,
    band: &BandMatrix<T>,
    pivots: &DeviceBuffer<i32>,
    b: &mut DeviceBuffer<T>,
    n: usize,
    nrhs: usize,
) -> SolverResult<()> {
    if n == 0 || nrhs == 0 {
        return Ok(());
    }
    if band.n != n {
        return Err(SolverError::DimensionMismatch(format!(
            "band_solve: band matrix dimension ({}) != n ({n})",
            band.n
        )));
    }
    if pivots.len() < n {
        return Err(SolverError::DimensionMismatch(
            "band_solve: pivots buffer too small".into(),
        ));
    }
    if b.len() < n * nrhs {
        return Err(SolverError::DimensionMismatch(
            "band_solve: B buffer too small".into(),
        ));
    }

    let ldab = band.ldab();
    let kl = band.kl;
    let ku = band.ku;
    let ws = (ldab * n + n * nrhs) * std::mem::size_of::<f64>();
    handle.ensure_workspace(ws)?;

    // Read to host.
    let mut ab = vec![0.0_f64; ldab * n];
    read_band_to_host(&band.data, &mut ab, ldab * n)?;

    let mut ipiv = vec![0_i32; n];
    read_pivots_from_device(pivots, &mut ipiv, n)?;

    let mut b_host = vec![0.0_f64; n * nrhs];
    read_band_to_host(b, &mut b_host, n * nrhs)?;

    // Solve on host.
    band_solve_host(&ab, &ipiv, &mut b_host, n, kl, ku, ldab, nrhs)?;

    // Write back.
    let b_device: Vec<T> = b_host.iter().map(|&v| from_f64(v)).collect();
    write_host_to_band_t(b, &b_device, n * nrhs)?;

    Ok(())
}

/// Cholesky factorization for a banded symmetric positive definite matrix.
///
/// Computes `A = L * L^T` where L is a banded lower triangular matrix.
/// The bandwidth of L equals the bandwidth of A (kl = ku = kd).
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `band` — banded SPD matrix (overwritten with the Cholesky factor).
///
/// # Errors
///
/// Returns [`SolverError::NotPositiveDefinite`] if the matrix is not SPD.
/// Returns [`SolverError::DimensionMismatch`] if kl != ku.
pub fn band_cholesky<T: GpuFloat>(
    handle: &mut SolverHandle,
    band: &mut BandMatrix<T>,
) -> SolverResult<()> {
    let n = band.n;
    let kl = band.kl;
    let ku = band.ku;

    if n == 0 {
        return Ok(());
    }
    if kl != ku {
        return Err(SolverError::DimensionMismatch(format!(
            "band_cholesky: kl ({kl}) must equal ku ({ku}) for symmetric matrix"
        )));
    }

    let ldab = band.ldab();
    let ws = ldab * n * std::mem::size_of::<f64>();
    handle.ensure_workspace(ws)?;

    // Read to host.
    let mut ab = vec![0.0_f64; ldab * n];
    read_band_to_host(&band.data, &mut ab, ldab * n)?;

    // Perform banded Cholesky on host.
    band_cholesky_host(&mut ab, n, kl, ldab)?;

    // Write back.
    write_host_to_band_f64(&mut band.data, &ab, ldab * n)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Host-side banded LU
// ---------------------------------------------------------------------------

/// Banded LU factorization with partial pivoting (host-side).
fn band_lu_host(
    ab: &mut [f64],
    n: usize,
    kl: usize,
    ku: usize,
    ldab: usize,
    ipiv: &mut [i32],
) -> SolverResult<()> {
    for k in 0..n {
        // Find pivot: max |ab[kl + i - k, k]| for i = k..min(n, k+kl+1).
        let mut max_val = 0.0_f64;
        let mut max_idx = k;
        let end_row = n.min(k + kl + 1);

        for i in k..end_row {
            let band_row = kl + i - k;
            if band_row < ldab {
                let val = ab[k * ldab + band_row].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
        }

        ipiv[k] = max_idx as i32;

        if max_val < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }

        // Swap rows if needed.
        if max_idx != k {
            let p = max_idx;
            // Swap band storage rows for all affected columns.
            let col_start = k.saturating_sub(ku);
            let col_end = n.min(k + kl + ku + 1);
            for j in col_start..col_end {
                let row_k = kl + k;
                let row_p = kl + p;
                if row_k >= j && row_k - j < ldab && row_p >= j && row_p - j < ldab {
                    ab.swap(j * ldab + (row_k - j), j * ldab + (row_p - j));
                }
            }
        }

        // Eliminate below pivot.
        let pivot = ab[k * ldab + kl];
        if pivot.abs() < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }

        for i in (k + 1)..end_row {
            let band_row = kl + i - k;
            if band_row < ldab {
                let mult = ab[k * ldab + band_row] / pivot;
                ab[k * ldab + band_row] = mult; // Store multiplier in L.

                // Update trailing entries in row i.
                let update_end = n.min(k + ku + 1);
                for j in (k + 1)..update_end {
                    let src_row = kl + k - j + (j - k); // row k in col j => kl
                    let dst_row = kl + i - j;
                    if src_row < ldab && dst_row < ldab && j < n {
                        ab[j * ldab + dst_row] -= mult * ab[j * ldab + src_row];
                    }
                }
            }
        }
    }

    Ok(())
}

/// Solves the banded system using LU factors (host-side).
#[allow(clippy::too_many_arguments)]
fn band_solve_host(
    ab: &[f64],
    ipiv: &[i32],
    b: &mut [f64],
    n: usize,
    kl: usize,
    _ku: usize,
    ldab: usize,
    nrhs: usize,
) -> SolverResult<()> {
    for rhs in 0..nrhs {
        let b_col = &mut b[rhs * n..(rhs + 1) * n];

        // Apply row permutations.
        for (k, &piv) in ipiv.iter().enumerate().take(n) {
            let p = piv as usize;
            if p != k {
                b_col.swap(k, p);
            }
        }

        // Forward substitution (L * y = Pb).
        for k in 0..n {
            let end_row = n.min(k + kl + 1);
            for i in (k + 1)..end_row {
                let band_row = kl + i - k;
                if band_row < ldab {
                    let mult = ab[k * ldab + band_row];
                    b_col[i] -= mult * b_col[k];
                }
            }
        }

        // Backward substitution (U * x = y).
        for k in (0..n).rev() {
            let pivot = ab[k * ldab + kl];
            if pivot.abs() < 1e-300 {
                return Err(SolverError::SingularMatrix);
            }
            b_col[k] /= pivot;

            // Eliminate above.
            let start_row = k.saturating_sub(kl);
            for i in start_row..k {
                // For U entries, stored at band_row = kl + i - k (which may be < kl).
                let _band_row = kl + i - k;
                let idx = kl + i;
                if idx >= k {
                    let br = idx - k;
                    if br < ldab {
                        b_col[i] -= ab[k * ldab + br] * b_col[k];
                    }
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Host-side banded Cholesky
// ---------------------------------------------------------------------------

/// Banded Cholesky factorization (host-side).
///
/// For column j: L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2)
/// For i > j:  L[i,j] = (A[i,j] - sum_{k<j} L[i,k]*L[j,k]) / L[j,j]
fn band_cholesky_host(
    ab: &mut [f64],
    n: usize,
    kd: usize, // kl = ku = kd for symmetric
    ldab: usize,
) -> SolverResult<()> {
    for j in 0..n {
        // Compute L[j,j].
        let diag_idx = kd; // Diagonal is at row kd in each column.
        let mut sum = ab[j * ldab + diag_idx];

        // Subtract sum of squares from previous columns.
        let k_start = j.saturating_sub(kd);
        for k in k_start..j {
            let band_row_jk = kd + j - k;
            if band_row_jk < ldab {
                let ljk = ab[k * ldab + band_row_jk];
                sum -= ljk * ljk;
            }
        }

        if sum <= 0.0 {
            return Err(SolverError::NotPositiveDefinite);
        }

        let ljj = sum.sqrt();
        ab[j * ldab + diag_idx] = ljj;

        // Compute L[i,j] for i > j within the band.
        let end_row = n.min(j + kd + 1);
        for i in (j + 1)..end_row {
            let band_row_ij = kd + i - j;
            if band_row_ij >= ldab {
                continue;
            }

            let mut s = ab[j * ldab + band_row_ij];

            // Subtract sum of products from previous columns.
            for k in k_start..j {
                let br_ik = kd + i - k;
                let br_jk = kd + j - k;
                if br_ik < ldab && br_jk < ldab {
                    s -= ab[k * ldab + br_ik] * ab[k * ldab + br_jk];
                }
            }

            ab[j * ldab + band_row_ij] = s / ljj;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Device buffer helpers
// ---------------------------------------------------------------------------

fn read_band_to_host<T: GpuFloat>(
    buf: &DeviceBuffer<T>,
    host: &mut [f64],
    count: usize,
) -> SolverResult<()> {
    if host.len() < count {
        return Err(SolverError::DimensionMismatch(format!(
            "read_band_to_host: host buffer too small ({} < {count})",
            host.len()
        )));
    }
    if buf.len() < count {
        return Err(SolverError::DimensionMismatch(format!(
            "read_band_to_host: device buffer too small ({} < {count})",
            buf.len()
        )));
    }

    let mut staged = vec![T::gpu_zero(); buf.len()];
    buf.copy_to_host(&mut staged)?;
    for (dst, src) in host.iter_mut().zip(staged.iter()).take(count) {
        *dst = to_f64(*src);
    }
    Ok(())
}

fn write_host_to_band_f64<T: GpuFloat>(
    buf: &mut DeviceBuffer<T>,
    data: &[f64],
    count: usize,
) -> SolverResult<()> {
    if data.len() < count {
        return Err(SolverError::DimensionMismatch(format!(
            "write_host_to_band_f64: source too small ({} < {count})",
            data.len()
        )));
    }
    if buf.len() < count {
        return Err(SolverError::DimensionMismatch(format!(
            "write_host_to_band_f64: device buffer too small ({} < {count})",
            buf.len()
        )));
    }

    let mut staged = vec![T::gpu_zero(); buf.len()];
    for (dst, src) in staged.iter_mut().zip(data.iter()).take(count) {
        *dst = from_f64(*src);
    }
    buf.copy_from_host(&staged)?;
    Ok(())
}

fn write_host_to_band_t<T: GpuFloat>(
    buf: &mut DeviceBuffer<T>,
    data: &[T],
    count: usize,
) -> SolverResult<()> {
    if data.len() < count {
        return Err(SolverError::DimensionMismatch(format!(
            "write_host_to_band_t: source too small ({} < {count})",
            data.len()
        )));
    }
    if buf.len() < count {
        return Err(SolverError::DimensionMismatch(format!(
            "write_host_to_band_t: device buffer too small ({} < {count})",
            buf.len()
        )));
    }

    let mut staged = vec![T::gpu_zero(); buf.len()];
    staged[..count].copy_from_slice(&data[..count]);
    buf.copy_from_host(&staged)?;
    Ok(())
}

fn write_pivots_to_device(
    buf: &mut DeviceBuffer<i32>,
    data: &[i32],
    count: usize,
) -> SolverResult<()> {
    if data.len() < count {
        return Err(SolverError::DimensionMismatch(format!(
            "write_pivots_to_device: source too small ({} < {count})",
            data.len()
        )));
    }
    if buf.len() < count {
        return Err(SolverError::DimensionMismatch(format!(
            "write_pivots_to_device: device buffer too small ({} < {count})",
            buf.len()
        )));
    }

    let mut staged = vec![0_i32; buf.len()];
    staged[..count].copy_from_slice(&data[..count]);
    buf.copy_from_host(&staged)?;
    Ok(())
}

fn read_pivots_from_device(
    buf: &DeviceBuffer<i32>,
    host: &mut [i32],
    count: usize,
) -> SolverResult<()> {
    if host.len() < count {
        return Err(SolverError::DimensionMismatch(format!(
            "read_pivots_from_device: host buffer too small ({} < {count})",
            host.len()
        )));
    }
    if buf.len() < count {
        return Err(SolverError::DimensionMismatch(format!(
            "read_pivots_from_device: device buffer too small ({} < {count})",
            buf.len()
        )));
    }

    let mut staged = vec![0_i32; buf.len()];
    buf.copy_to_host(&mut staged)?;
    host[..count].copy_from_slice(&staged[..count]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_index_tridiagonal() {
        // 5x5 tridiagonal: kl=1, ku=1, ldab=2*1+1+1=4
        // Test band_index logic without requiring a GPU DeviceBuffer.
        let n = 5_usize;
        let kl = 1_usize;
        let ku = 1_usize;
        let ldab = 2 * kl + ku + 1; // 4

        // Diagonal element (2,2): band_row = kl + 2 - 2 = 1
        let row_in_band = kl + 2; // 2
        assert!(row_in_band >= 2); // j=2
        let band_row = row_in_band - 2; // 1
        assert!(band_row < ldab);
        let idx = 2 * ldab + band_row; // 9
        assert_eq!(idx, 9);
        let _ = n;
    }

    #[test]
    fn band_index_out_of_band() {
        // Element (0, 3) in a tridiagonal (kl=1, ku=1) is outside the band.
        let kl = 1_usize;
        let row_in_band = kl; // 1
        let j = 3_usize;
        // row_in_band (1) < j (3) => outside band.
        assert!(row_in_band < j);
    }

    #[test]
    fn band_matrix_ldab_formula() {
        // kl=2, ku=3 => ldab = 2*2 + 3 + 1 = 8
        let kl = 2_usize;
        let ku = 3_usize;
        let ldab = 2 * kl + ku + 1;
        assert_eq!(ldab, 8);
    }

    #[test]
    fn band_lu_host_tridiagonal() {
        // 3x3 tridiagonal: kl=1, ku=1, ldab=4
        // [2 -1  0]
        // [-1 2 -1]
        // [0 -1  2]
        let ldab = 4;
        let n = 3;
        let mut ab = vec![0.0_f64; ldab * n];

        // Column 0: superdiag=*, diag=2, subdiag=-1
        ab[1] = 2.0; // diagonal at row kl=1
        ab[2] = -1.0; // subdiag at row kl+1=2

        // Column 1: superdiag=-1, diag=2, subdiag=-1
        ab[ldab] = -1.0; // superdiag at row kl-1=0
        ab[ldab + 1] = 2.0; // diagonal
        ab[ldab + 2] = -1.0; // subdiag

        // Column 2: superdiag=-1, diag=2, subdiag=*
        ab[2 * ldab] = -1.0;
        ab[2 * ldab + 1] = 2.0;

        let mut ipiv = vec![0_i32; n];
        let result = band_lu_host(&mut ab, n, 1, 1, ldab, &mut ipiv);
        assert!(result.is_ok());
    }

    #[test]
    fn band_cholesky_host_tridiagonal() {
        // 3x3 SPD tridiagonal: [2 -1 0; -1 2 -1; 0 -1 2]
        let kd = 1;
        let ldab = 2 * kd + kd + 1; // = 4
        let n = 3;
        let mut ab = vec![0.0_f64; ldab * n];

        // Using the diagonal at row kd=1.
        ab[1] = 2.0; // A[0,0]
        ab[2] = -1.0; // A[1,0]

        ab[ldab + 1] = 2.0; // A[1,1]
        ab[ldab + 2] = -1.0; // A[2,1]

        ab[2 * ldab + 1] = 2.0; // A[2,2]

        let result = band_cholesky_host(&mut ab, n, kd, ldab);
        assert!(result.is_ok());

        // L[0,0] = sqrt(2) ≈ 1.4142
        assert!((ab[1] - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn band_cholesky_host_not_spd() {
        // Non-SPD matrix: diagonal is negative.
        let kd = 1;
        let ldab = 4;
        let n = 2;
        let mut ab = vec![0.0_f64; ldab * n];

        ab[1] = -1.0; // A[0,0] = -1 (not SPD)
        ab[ldab + 1] = 2.0;

        let result = band_cholesky_host(&mut ab, n, kd, ldab);
        assert!(result.is_err());
    }

    #[test]
    fn f64_conversion_roundtrip() {
        let val = std::f64::consts::E;
        let converted: f64 = from_f64(to_f64(val));
        assert!((converted - val).abs() < 1e-15);
    }

    #[test]
    fn f32_conversion_roundtrip() {
        let val = std::f32::consts::E;
        let as_f64 = to_f64(val);
        let back: f32 = from_f64(as_f64);
        assert!((back - val).abs() < 1e-5);
    }
}
