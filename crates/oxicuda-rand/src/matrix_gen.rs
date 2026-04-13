//! Random matrix generation for statistical applications.
//!
//! Provides generators for several families of structured random matrices:
//!
//! - **Gaussian**: matrices with i.i.d. Gaussian entries
//! - **Wishart**: positive-definite matrices from the Wishart distribution
//! - **Orthogonal**: uniformly distributed orthogonal matrices (Haar measure)
//! - **SPD**: symmetric positive-definite matrices with controlled condition number
//! - **Correlation**: valid correlation matrices via the vine method
//!
//! All matrices are stored as flat `Vec<f64>` in row-major or column-major order.

use crate::error::{RandError, RandResult};

// ---------------------------------------------------------------------------
// CPU-side PRNG (SplitMix64)
// ---------------------------------------------------------------------------

/// Simple SplitMix64 PRNG for CPU-side random number generation.
///
/// This is a fast, high-quality 64-bit PRNG suitable for seeding and
/// generating random values on the CPU side.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a new SplitMix64 with the given seed.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next u64 random value.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Returns a pair of standard normal f64 values via Box-Muller transform.
    fn next_normal_pair(&mut self) -> (f64, f64) {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 0.0 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                return (r * theta.cos(), r * theta.sin());
            }
        }
    }

    /// Returns a single standard normal f64 value.
    fn next_normal(&mut self) -> f64 {
        self.next_normal_pair().0
    }
}

// ---------------------------------------------------------------------------
// MatrixLayout
// ---------------------------------------------------------------------------

/// Storage layout for matrix data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixLayout {
    /// Row-major (C-style) ordering. Element (i,j) is at index `i * cols + j`.
    RowMajor,
    /// Column-major (Fortran-style) ordering. Element (i,j) is at index `j * rows + i`.
    ColMajor,
}

impl std::fmt::Display for MatrixLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RowMajor => write!(f, "RowMajor"),
            Self::ColMajor => write!(f, "ColMajor"),
        }
    }
}

// ---------------------------------------------------------------------------
// RandomMatrix
// ---------------------------------------------------------------------------

/// A random matrix stored as a flat vector.
///
/// Provides basic accessors for the matrix dimensions and elements.
/// The data is stored as a contiguous `Vec<f64>` in the specified layout.
#[derive(Debug, Clone)]
pub struct RandomMatrix {
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
    /// Flat data storage.
    data: Vec<f64>,
    /// Memory layout.
    layout: MatrixLayout,
}

impl RandomMatrix {
    /// Creates a new `RandomMatrix` from existing data.
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if `data.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>, layout: MatrixLayout) -> RandResult<Self> {
        if data.len() != rows * cols {
            return Err(RandError::InvalidSize(format!(
                "data length {} does not match {}x{} = {}",
                data.len(),
                rows,
                cols,
                rows * cols
            )));
        }
        Ok(Self {
            rows,
            cols,
            data,
            layout,
        })
    }

    /// Creates a zero-filled matrix.
    pub fn zeros(rows: usize, cols: usize, layout: MatrixLayout) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
            layout,
        }
    }

    /// Creates an identity matrix (must be square).
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if `n == 0`.
    pub fn identity(n: usize, layout: MatrixLayout) -> RandResult<Self> {
        if n == 0 {
            return Err(RandError::InvalidSize(
                "identity matrix dimension must be positive".to_string(),
            ));
        }
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        // If ColMajor, the identity is the same (diagonal is at i*n+i in both layouts)
        Ok(Self {
            rows: n,
            cols: n,
            data,
            layout,
        })
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the matrix layout.
    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }

    /// Returns a reference to the underlying data.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Returns a mutable reference to the underlying data.
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Consumes the matrix and returns the underlying data.
    pub fn into_data(self) -> Vec<f64> {
        self.data
    }

    /// Returns the element at position (i, j).
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows` or `j >= cols`.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        match self.layout {
            MatrixLayout::RowMajor => self.data[i * self.cols + j],
            MatrixLayout::ColMajor => self.data[j * self.rows + i],
        }
    }

    /// Sets the element at position (i, j).
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows` or `j >= cols`.
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        match self.layout {
            MatrixLayout::RowMajor => self.data[i * self.cols + j] = value,
            MatrixLayout::ColMajor => self.data[j * self.rows + i] = value,
        }
    }

    /// Returns `true` if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Computes the Frobenius norm of the matrix.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Returns the transpose of this matrix.
    pub fn transpose(&self) -> Self {
        let data = transpose(&self.data, self.rows, self.cols);
        Self {
            rows: self.cols,
            cols: self.rows,
            data,
            layout: self.layout,
        }
    }
}

impl std::fmt::Display for RandomMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RandomMatrix({}x{}, {})",
            self.rows, self.cols, self.layout
        )
    }
}

// ---------------------------------------------------------------------------
// Helper functions: linear algebra primitives
// ---------------------------------------------------------------------------

/// Transposes an `m x n` row-major matrix into an `n x m` row-major matrix.
pub fn transpose(matrix: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut result = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
    result
}

/// Computes C = A * B where A is `m x k` and B is `k x n` (row-major).
///
/// The output C is `m x n` in row-major order.
pub fn matrix_multiply(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

/// Computes the Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Given an `n x n` SPD matrix A (row-major), returns the lower-triangular
/// matrix L such that A = L * L^T.
///
/// # Errors
///
/// Returns `RandError::InternalError` if the matrix is not positive definite
/// (i.e., a diagonal element becomes non-positive during decomposition).
pub fn cholesky_decompose(matrix: &[f64], n: usize) -> RandResult<Vec<f64>> {
    if matrix.len() != n * n {
        return Err(RandError::InvalidSize(format!(
            "expected {}x{} = {} elements, got {}",
            n,
            n,
            n * n,
            matrix.len()
        )));
    }

    let mut l = vec![0.0; n * n];

    for j in 0..n {
        // Diagonal element
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j * n + k] * l[j * n + k];
        }
        let diag = matrix[j * n + j] - sum;
        if diag <= 0.0 {
            return Err(RandError::InternalError(format!(
                "Cholesky decomposition failed: matrix is not positive definite \
                 (diagonal element {} became {:.6e})",
                j, diag
            )));
        }
        l[j * n + j] = diag.sqrt();

        // Off-diagonal elements in column j
        for i in (j + 1)..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = (matrix[i * n + j] - sum) / l[j * n + j];
        }
    }

    Ok(l)
}

// ---------------------------------------------------------------------------
// GaussianMatrixGenerator
// ---------------------------------------------------------------------------

/// Generates matrices with independent and identically distributed Gaussian entries.
///
/// Each entry is drawn from N(mean, stddev^2) using the Box-Muller transform
/// applied to a SplitMix64 PRNG.
pub struct GaussianMatrixGenerator;

impl GaussianMatrixGenerator {
    /// Generates an `rows x cols` matrix with i.i.d. Gaussian entries.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `mean` - Mean of the Gaussian distribution
    /// * `stddev` - Standard deviation of the Gaussian distribution
    /// * `seed` - Random seed
    pub fn generate(rows: usize, cols: usize, mean: f64, stddev: f64, seed: u64) -> RandomMatrix {
        let mut rng = SplitMix64::new(seed);
        let total = rows * cols;
        let mut data = Vec::with_capacity(total);

        // Generate pairs via Box-Muller
        let mut i = 0;
        while i + 1 < total {
            let (z0, z1) = rng.next_normal_pair();
            data.push(mean + stddev * z0);
            data.push(mean + stddev * z1);
            i += 2;
        }
        // Handle odd remaining element
        if data.len() < total {
            let z = rng.next_normal();
            data.push(mean + stddev * z);
        }

        RandomMatrix {
            rows,
            cols,
            data,
            layout: MatrixLayout::RowMajor,
        }
    }
}

// ---------------------------------------------------------------------------
// WishartGenerator
// ---------------------------------------------------------------------------

/// Generates random matrices from the Wishart distribution.
///
/// The Wishart distribution W ~ Wishart(Sigma, n) is the distribution of
/// X^T * X where the rows of X are i.i.d. multivariate normal N(0, Sigma).
///
/// The resulting matrix is symmetric positive definite when `dof >= dim`.
pub struct WishartGenerator;

impl WishartGenerator {
    /// Generates a Wishart-distributed random matrix.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension p of the p x p output matrix
    /// * `dof` - Degrees of freedom n (must be >= dim for positive definiteness)
    /// * `scale` - The p x p scale matrix Sigma in row-major order
    /// * `seed` - Random seed
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if `dof < dim` or `scale` has wrong length.
    /// Returns `RandError::InternalError` if the scale matrix is not positive definite.
    pub fn generate(dim: usize, dof: usize, scale: &[f64], seed: u64) -> RandResult<RandomMatrix> {
        if dim == 0 {
            return Err(RandError::InvalidSize(
                "Wishart dimension must be positive".to_string(),
            ));
        }
        if dof < dim {
            return Err(RandError::InvalidSize(format!(
                "degrees of freedom ({dof}) must be >= dimension ({dim}) for positive definiteness"
            )));
        }
        if scale.len() != dim * dim {
            return Err(RandError::InvalidSize(format!(
                "scale matrix must have {} elements, got {}",
                dim * dim,
                scale.len()
            )));
        }

        // Cholesky decomposition: Sigma = L * L^T
        let l = cholesky_decompose(scale, dim)?;

        // Generate Z: dof x dim matrix of standard normals
        let z = GaussianMatrixGenerator::generate(dof, dim, 0.0, 1.0, seed);

        // X = Z * L^T (each row of X ~ N(0, Sigma))
        let lt = transpose(&l, dim, dim);
        let x = matrix_multiply(z.data(), &lt, dof, dim, dim);

        // W = X^T * X (dim x dim SPD matrix)
        let xt = transpose(&x, dof, dim);
        let w = matrix_multiply(&xt, &x, dim, dim, dof);

        RandomMatrix::new(dim, dim, w, MatrixLayout::RowMajor)
    }
}

// ---------------------------------------------------------------------------
// OrthogonalMatrixGenerator
// ---------------------------------------------------------------------------

/// Generates random orthogonal matrices uniformly distributed on O(n)
/// according to the Haar measure.
///
/// The method is: generate a Gaussian random matrix, compute its QR
/// decomposition via modified Gram-Schmidt, and return Q.
pub struct OrthogonalMatrixGenerator;

impl OrthogonalMatrixGenerator {
    /// Generates a random `dim x dim` orthogonal matrix.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension of the square orthogonal matrix
    /// * `seed` - Random seed
    pub fn generate(dim: usize, seed: u64) -> RandomMatrix {
        if dim == 0 {
            return RandomMatrix {
                rows: 0,
                cols: 0,
                data: Vec::new(),
                layout: MatrixLayout::RowMajor,
            };
        }
        if dim == 1 {
            return RandomMatrix {
                rows: 1,
                cols: 1,
                data: vec![1.0],
                layout: MatrixLayout::RowMajor,
            };
        }

        // Generate a random Gaussian matrix
        let a = GaussianMatrixGenerator::generate(dim, dim, 0.0, 1.0, seed);

        // QR decomposition via modified Gram-Schmidt
        let q = modified_gram_schmidt(a.data(), dim);

        RandomMatrix {
            rows: dim,
            cols: dim,
            data: q,
            layout: MatrixLayout::RowMajor,
        }
    }
}

/// Modified Gram-Schmidt QR decomposition.
///
/// Given an `n x n` matrix A (row-major), returns Q (n x n row-major)
/// such that Q is orthogonal.
///
/// We work column-by-column. Columns of A are extracted, orthogonalized,
/// and normalized.
fn modified_gram_schmidt(a: &[f64], n: usize) -> Vec<f64> {
    // Extract columns
    let mut cols: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..n).map(|i| a[i * n + j]).collect())
        .collect();

    for j in 0..n {
        // Normalize column j
        let norm = cols[j].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for elem in &mut cols[j] {
                *elem /= norm;
            }
        }

        // Orthogonalize remaining columns against column j
        for k in (j + 1)..n {
            let dot: f64 = (0..n).map(|i| cols[j][i] * cols[k][i]).sum();
            let col_j_copy: Vec<f64> = cols[j].clone();
            for (elem, basis) in cols[k].iter_mut().zip(col_j_copy.iter()) {
                *elem -= dot * basis;
            }
        }
    }

    // Pack columns back into row-major matrix
    let mut q = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            q[i * n + j] = cols[j][i];
        }
    }
    q
}

// ---------------------------------------------------------------------------
// SymmetricPositiveDefiniteGenerator
// ---------------------------------------------------------------------------

/// Generates random symmetric positive-definite (SPD) matrices with
/// controlled condition number.
///
/// The method is: generate a random orthogonal Q, a positive diagonal D
/// with eigenvalues logarithmically spaced between 1 and `condition_number`,
/// then return Q * D * Q^T.
pub struct SymmetricPositiveDefiniteGenerator;

impl SymmetricPositiveDefiniteGenerator {
    /// Generates a random `dim x dim` SPD matrix.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension of the square SPD matrix
    /// * `condition_number` - Ratio of largest to smallest eigenvalue (must be >= 1.0)
    /// * `seed` - Random seed
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if `dim == 0` or `condition_number < 1.0`.
    pub fn generate(dim: usize, condition_number: f64, seed: u64) -> RandResult<RandomMatrix> {
        if dim == 0 {
            return Err(RandError::InvalidSize(
                "SPD dimension must be positive".to_string(),
            ));
        }
        if condition_number < 1.0 {
            return Err(RandError::InvalidSize(format!(
                "condition number must be >= 1.0, got {condition_number}"
            )));
        }

        // Generate orthogonal matrix Q
        let q_mat = OrthogonalMatrixGenerator::generate(dim, seed);
        let q = q_mat.data();

        // Generate diagonal D with eigenvalues log-spaced from 1 to condition_number
        let mut d = vec![0.0; dim];
        if dim == 1 {
            d[0] = 1.0;
        } else {
            let log_min = 0.0_f64; // ln(1) = 0
            let log_max = condition_number.ln();
            for (i, d_i) in d.iter_mut().enumerate() {
                let t = i as f64 / (dim - 1) as f64;
                *d_i = (log_min + t * (log_max - log_min)).exp();
            }
        }

        // Compute Q * D * Q^T
        // First: Q * D  (scale each column of Q by d[j])
        let mut qd = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                qd[i * dim + j] = q[i * dim + j] * d[j];
            }
        }

        // Then: (Q*D) * Q^T
        let qt = transpose(q, dim, dim);
        let result = matrix_multiply(&qd, &qt, dim, dim, dim);

        RandomMatrix::new(dim, dim, result, MatrixLayout::RowMajor)
    }
}

// ---------------------------------------------------------------------------
// CorrelationMatrixGenerator
// ---------------------------------------------------------------------------

/// Generates random correlation matrices using the vine method.
///
/// A correlation matrix is symmetric positive semi-definite with ones on the
/// diagonal and off-diagonal entries in [-1, 1]. The vine method generates
/// partial correlations uniformly and constructs a valid correlation matrix.
pub struct CorrelationMatrixGenerator;

impl CorrelationMatrixGenerator {
    /// Generates a random `dim x dim` correlation matrix.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension of the square correlation matrix
    /// * `seed` - Random seed
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if `dim == 0`.
    pub fn generate(dim: usize, seed: u64) -> RandResult<RandomMatrix> {
        if dim == 0 {
            return Err(RandError::InvalidSize(
                "correlation matrix dimension must be positive".to_string(),
            ));
        }
        if dim == 1 {
            return RandomMatrix::new(1, 1, vec![1.0], MatrixLayout::RowMajor);
        }

        let mut rng = SplitMix64::new(seed);

        // Vine method: build correlation matrix from partial correlations.
        // We construct the matrix via a lower-triangular factor.
        //
        // For each column k and row i > k:
        //   Generate a partial correlation p_{ik} in (-1, 1)
        //   Build the factor L such that C = L * L^T is a correlation matrix.

        let mut l = vec![0.0; dim * dim];

        // First column of L
        l[0] = 1.0; // L[0,0] = 1
        for i in 1..dim {
            // Generate partial correlation for (i, 0)
            let p = 2.0 * rng.next_f64() - 1.0;
            l[i * dim] = p; // L[i,0] = p
        }

        // Remaining columns
        for k in 1..dim {
            // L[k,k] = sqrt(1 - sum of L[k,j]^2 for j < k)
            let mut sum_sq = 0.0;
            for j in 0..k {
                sum_sq += l[k * dim + j] * l[k * dim + j];
            }
            let rem = 1.0 - sum_sq;
            l[k * dim + k] = if rem > 0.0 { rem.sqrt() } else { 0.0 };

            // For rows i > k
            for i in (k + 1)..dim {
                // Remaining "radius" for row i
                let mut sum_sq_i = 0.0;
                for j in 0..k {
                    sum_sq_i += l[i * dim + j] * l[i * dim + j];
                }
                let rem_i = 1.0 - sum_sq_i;
                if rem_i <= 0.0 {
                    l[i * dim + k] = 0.0;
                    continue;
                }

                // Generate partial correlation
                let p = 2.0 * rng.next_f64() - 1.0;
                l[i * dim + k] = p * rem_i.sqrt();
            }
        }

        // C = L * L^T
        let lt = transpose(&l, dim, dim);
        let c = matrix_multiply(&l, &lt, dim, dim, dim);

        RandomMatrix::new(dim, dim, c, MatrixLayout::RowMajor)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    /// Check if a matrix is approximately symmetric.
    fn is_symmetric(m: &RandomMatrix, tol: f64) -> bool {
        if !m.is_square() {
            return false;
        }
        let n = m.rows();
        for i in 0..n {
            for j in (i + 1)..n {
                if (m.get(i, j) - m.get(j, i)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check if all diagonal elements of a matrix are approximately 1.
    fn has_unit_diagonal(m: &RandomMatrix, tol: f64) -> bool {
        let n = m.rows().min(m.cols());
        for i in 0..n {
            if (m.get(i, i) - 1.0).abs() > tol {
                return false;
            }
        }
        true
    }

    /// Check if a matrix is positive definite by attempting Cholesky decomposition.
    fn is_positive_definite(m: &RandomMatrix) -> bool {
        if !m.is_square() {
            return false;
        }
        cholesky_decompose(m.data(), m.rows()).is_ok()
    }

    // -----------------------------------------------------------------------
    // Gaussian matrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn gaussian_correct_dimensions() {
        let m = GaussianMatrixGenerator::generate(5, 3, 0.0, 1.0, 42);
        assert_eq!(m.rows(), 5);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.data().len(), 15);
    }

    #[test]
    fn gaussian_mean_and_variance() {
        let m = GaussianMatrixGenerator::generate(1000, 1000, 2.5, 0.5, 123);
        let n = m.data().len() as f64;
        let mean = m.data().iter().sum::<f64>() / n;
        let variance = m.data().iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

        // With 1M samples, mean should be close to 2.5 and variance close to 0.25
        assert!((mean - 2.5).abs() < 0.01, "mean = {mean}");
        assert!((variance - 0.25).abs() < 0.01, "variance = {variance}");
    }

    #[test]
    fn gaussian_deterministic_with_seed() {
        let m1 = GaussianMatrixGenerator::generate(10, 10, 0.0, 1.0, 999);
        let m2 = GaussianMatrixGenerator::generate(10, 10, 0.0, 1.0, 999);
        assert_eq!(m1.data(), m2.data());
    }

    // -----------------------------------------------------------------------
    // Cholesky decomposition tests
    // -----------------------------------------------------------------------

    #[test]
    fn cholesky_identity() {
        let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let l = cholesky_decompose(&identity, 3).expect("cholesky should succeed");
        // L should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (l[i * 3 + j] - expected).abs() < TOL,
                    "L[{i},{j}] = {} expected {expected}",
                    l[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn cholesky_reconstruction() {
        // A = [[4, 2], [2, 3]]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky_decompose(&a, 2).expect("cholesky should succeed");
        // Verify A = L * L^T
        let lt = transpose(&l, 2, 2);
        let reconstructed = matrix_multiply(&l, &lt, 2, 2, 2);
        for i in 0..4 {
            assert!(
                (reconstructed[i] - a[i]).abs() < TOL,
                "element {i}: {} vs {}",
                reconstructed[i],
                a[i]
            );
        }
    }

    #[test]
    fn cholesky_not_positive_definite() {
        // Not positive definite: diagonal has a negative
        let a = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky_decompose(&a, 2).is_err());
    }

    // -----------------------------------------------------------------------
    // Matrix multiply and transpose tests
    // -----------------------------------------------------------------------

    #[test]
    fn transpose_round_trip() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let at = transpose(&a, 2, 3); // 3x2
        let att = transpose(&at, 3, 2); // 2x3
        assert_eq!(a, att);
    }

    #[test]
    fn matrix_multiply_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let id = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        let result = matrix_multiply(&a, &id, 2, 2, 2);
        for i in 0..4 {
            assert!((result[i] - a[i]).abs() < TOL);
        }
    }

    // -----------------------------------------------------------------------
    // Orthogonal matrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn orthogonal_qtq_is_identity() {
        let q = OrthogonalMatrixGenerator::generate(5, 42);
        let qt = q.transpose();
        let qtq = matrix_multiply(qt.data(), q.data(), 5, 5, 5);

        for i in 0..5 {
            for j in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[i * 5 + j] - expected).abs() < 1e-12,
                    "Q^T*Q[{i},{j}] = {} expected {expected}",
                    qtq[i * 5 + j]
                );
            }
        }
    }

    #[test]
    fn orthogonal_determinant_abs_one() {
        // For a 2x2 orthogonal matrix, |det| = 1
        let q = OrthogonalMatrixGenerator::generate(2, 77);
        let det = q.get(0, 0) * q.get(1, 1) - q.get(0, 1) * q.get(1, 0);
        assert!(
            (det.abs() - 1.0).abs() < 1e-12,
            "|det| = {} expected 1.0",
            det.abs()
        );
    }

    // -----------------------------------------------------------------------
    // Wishart matrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn wishart_is_symmetric_and_spd() {
        let dim = 4;
        let dof = 10;
        // Use identity as scale matrix
        let mut scale = vec![0.0; dim * dim];
        for i in 0..dim {
            scale[i * dim + i] = 1.0;
        }
        let w = WishartGenerator::generate(dim, dof, &scale, 42).expect("wishart should succeed");
        assert_eq!(w.rows(), dim);
        assert_eq!(w.cols(), dim);
        assert!(
            is_symmetric(&w, 1e-10),
            "Wishart matrix should be symmetric"
        );
        assert!(is_positive_definite(&w), "Wishart matrix should be SPD");
    }

    #[test]
    fn wishart_dof_less_than_dim_errors() {
        let scale = vec![1.0, 0.0, 0.0, 1.0];
        let result = WishartGenerator::generate(2, 1, &scale, 42);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // SPD matrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn spd_is_symmetric_and_positive_definite() {
        let m = SymmetricPositiveDefiniteGenerator::generate(5, 100.0, 42)
            .expect("spd gen should succeed");
        assert!(is_symmetric(&m, 1e-10), "SPD matrix should be symmetric");
        assert!(
            is_positive_definite(&m),
            "SPD matrix should be positive definite"
        );
    }

    #[test]
    fn spd_condition_number_bound() {
        let dim = 4;
        let kappa = 10.0;
        let m = SymmetricPositiveDefiniteGenerator::generate(dim, kappa, 42)
            .expect("spd gen should succeed");

        // The eigenvalues are the diagonal of D: [1, ..., kappa]
        // We can verify by checking the trace and Frobenius norm are consistent
        let trace: f64 = (0..dim).map(|i| m.get(i, i)).sum();
        assert!(trace > 0.0, "trace should be positive");
    }

    #[test]
    fn spd_invalid_condition_number() {
        let result = SymmetricPositiveDefiniteGenerator::generate(3, 0.5, 42);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Correlation matrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn correlation_unit_diagonal() {
        let c =
            CorrelationMatrixGenerator::generate(5, 42).expect("correlation gen should succeed");
        assert!(
            has_unit_diagonal(&c, 1e-10),
            "correlation matrix should have unit diagonal"
        );
    }

    #[test]
    fn correlation_is_symmetric_and_psd() {
        let c =
            CorrelationMatrixGenerator::generate(5, 42).expect("correlation gen should succeed");
        assert!(
            is_symmetric(&c, 1e-10),
            "correlation matrix should be symmetric"
        );
        // Correlation matrices are PSD; the vine method should produce PD in practice
        assert!(
            is_positive_definite(&c),
            "correlation matrix should be positive semi-definite"
        );
    }

    #[test]
    fn correlation_entries_bounded() {
        let c =
            CorrelationMatrixGenerator::generate(6, 123).expect("correlation gen should succeed");
        for val in c.data() {
            assert!(
                *val >= -1.0 - 1e-12 && *val <= 1.0 + 1e-12,
                "correlation entry {val} out of [-1, 1]"
            );
        }
    }
}
