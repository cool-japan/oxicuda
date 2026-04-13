//! Tensor decomposition algorithms: CP, Tucker, and Tensor-Train (TT).
//!
//! This module provides CPU-side implementations of the three most widely used
//! tensor decomposition methods, which can be used to generate PTX kernels for
//! GPU execution.
//!
//! ## Decompositions
//!
//! - **CP (CANDECOMP/PARAFAC)** — decomposes a tensor into a sum of rank-one
//!   tensors via Alternating Least Squares (ALS).
//! - **Tucker** — decomposes a tensor into a core tensor multiplied by factor
//!   matrices along each mode, via HOSVD or HOOI.
//! - **Tensor-Train (TT)** — decomposes a tensor into a chain of 3D cores
//!   via sequential SVDs.
//!
//! ## References
//!
//! - Kolda & Bader, "Tensor Decompositions and Applications", SIAM Review 2009
//! - Oseledets, "Tensor-Train Decomposition", SIAM J. Sci. Comput. 2011

use crate::error::{SolverError, SolverResult};

// ---------------------------------------------------------------------------
// Matrix (minimal linear algebra support)
// ---------------------------------------------------------------------------

/// A dense row-major matrix.
#[derive(Debug, Clone)]
pub struct Matrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major data.
    pub data: Vec<f64>,
}

impl Matrix {
    /// Creates a new matrix from shape and data.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> SolverResult<Self> {
        if data.len() != rows * cols {
            return Err(SolverError::DimensionMismatch(format!(
                "matrix {}x{} requires {} elements, got {}",
                rows,
                cols,
                rows * cols,
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// Creates a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates an identity matrix.
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self {
            rows: n,
            cols: n,
            data,
        }
    }

    /// Element access.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Mutable element access.
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Returns the transpose.
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// Matrix multiplication: self * other.
    pub fn matmul(&self, other: &Matrix) -> SolverResult<Matrix> {
        if self.cols != other.rows {
            return Err(SolverError::DimensionMismatch(format!(
                "matmul: {}x{} * {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_ik = self.get(i, k);
                for j in 0..other.cols {
                    let cur = out.get(i, j);
                    out.set(i, j, cur + a_ik * other.get(k, j));
                }
            }
        }
        Ok(out)
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Column norms (L2 norm of each column).
    pub fn column_norms(&self) -> Vec<f64> {
        let mut norms = vec![0.0; self.cols];
        for r in 0..self.rows {
            for (c, norm) in norms.iter_mut().enumerate() {
                let v = self.get(r, c);
                *norm += v * v;
            }
        }
        norms.iter().map(|s| s.sqrt()).collect()
    }

    /// Normalize columns in-place, returning the norms.
    pub fn normalize_columns(&mut self) -> Vec<f64> {
        let norms = self.column_norms();
        for (c, &norm) in norms.iter().enumerate() {
            if norm > 1e-15 {
                for r in 0..self.rows {
                    let v = self.get(r, c) / norm;
                    self.set(r, c, v);
                }
            }
        }
        norms
    }

    /// Extract a column as a vector.
    pub fn column(&self, c: usize) -> Vec<f64> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    /// Truncated SVD via deflated power iteration.
    ///
    /// Computes the top `rank` singular triplets one at a time, deflating the
    /// matrix after each. Returns (U, sigma, V) where U is m×k, sigma has k
    /// entries, V is n×k.
    pub fn svd_truncated(&self, rank: usize) -> SolverResult<(Matrix, Vec<f64>, Matrix)> {
        let m = self.rows;
        let n = self.cols;
        let k = rank.min(m).min(n);
        if k == 0 {
            return Ok((Matrix::zeros(m, 0), Vec::new(), Matrix::zeros(n, 0)));
        }

        let mut u_mat = Matrix::zeros(m, k);
        let mut v_mat = Matrix::zeros(n, k);
        let mut sigma = vec![0.0; k];

        // Work on a deflated copy of the matrix
        let mut deflated = self.clone();

        for (s, sigma_s) in sigma.iter_mut().enumerate().take(k) {
            // Initialize v with a deterministic vector
            let mut v: Vec<f64> = (0..n)
                .map(|i| ((i + 1) as f64 * (s + 1) as f64 * 0.7 + 0.3).sin())
                .collect();
            // Normalize v
            let mut vnorm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if vnorm > 1e-15 {
                for x in &mut v {
                    *x /= vnorm;
                }
            }

            let max_iters = 200;
            for _ in 0..max_iters {
                // u = A * v
                let mut u: Vec<f64> = (0..m)
                    .map(|i| {
                        v.iter()
                            .enumerate()
                            .map(|(j, &vj)| deflated.get(i, j) * vj)
                            .sum()
                    })
                    .collect();

                // sigma_new = ||u||
                let sigma_new: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
                if sigma_new < 1e-15 {
                    break;
                }
                // Normalize u
                for x in &mut u {
                    *x /= sigma_new;
                }

                // v_new = A^T * u
                let mut v_new: Vec<f64> = (0..n)
                    .map(|j| {
                        u.iter()
                            .enumerate()
                            .map(|(i, &ui)| deflated.get(i, j) * ui)
                            .sum()
                    })
                    .collect();

                // Normalize v_new
                vnorm = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
                if vnorm < 1e-15 {
                    break;
                }
                for x in &mut v_new {
                    *x /= vnorm;
                }

                // Check convergence: |sigma_new - prev| / sigma_new
                let diff: f64 = v
                    .iter()
                    .zip(v_new.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                v = v_new;

                if diff < 1e-12 {
                    break;
                }
            }

            // Final pass: compute sigma from u = A*v
            let mut u: Vec<f64> = (0..m)
                .map(|i| {
                    v.iter()
                        .enumerate()
                        .map(|(j, &vj)| deflated.get(i, j) * vj)
                        .sum()
                })
                .collect();
            let sv = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            if sv > 1e-15 {
                for x in &mut u {
                    *x /= sv;
                }
            }

            *sigma_s = sv;
            for (i, &ui) in u.iter().enumerate() {
                u_mat.set(i, s, ui);
            }
            for (j, &vj) in v.iter().enumerate() {
                v_mat.set(j, s, vj);
            }

            // Deflate: A <- A - sigma * u * v^T
            for (i, &ui) in u.iter().enumerate() {
                for (j, &vj) in v.iter().enumerate() {
                    let old = deflated.get(i, j);
                    deflated.set(i, j, old - sv * ui * vj);
                }
            }
        }

        Ok((u_mat, sigma, v_mat))
    }
}

/// Modified Gram-Schmidt QR decomposition.
/// Returns (Q, R) where Q is m×k orthonormal, R is k×k upper triangular.
fn qr_gram_schmidt(a: &Matrix) -> (Matrix, Matrix) {
    let m = a.rows;
    let k = a.cols;
    let mut q = a.clone();
    let mut r = Matrix::zeros(k, k);

    for j in 0..k {
        // Orthogonalize against previous columns
        for i in 0..j {
            let mut dot = 0.0;
            for row in 0..m {
                dot += q.get(row, i) * q.get(row, j);
            }
            r.set(i, j, dot);
            for row in 0..m {
                let v = q.get(row, j) - dot * q.get(row, i);
                q.set(row, j, v);
            }
        }
        // Normalize
        let mut norm = 0.0;
        for row in 0..m {
            let v = q.get(row, j);
            norm += v * v;
        }
        norm = norm.sqrt();
        r.set(j, j, norm);
        if norm > 1e-15 {
            for row in 0..m {
                let v = q.get(row, j) / norm;
                q.set(row, j, v);
            }
        }
    }

    (q, r)
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

/// A dense multi-dimensional tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Shape of the tensor (dimensions along each mode).
    shape: Vec<usize>,
    /// Flattened data in row-major (C-order) layout.
    data: Vec<f64>,
}

impl Tensor {
    /// Creates a new tensor from shape and data.
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> SolverResult<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(SolverError::DimensionMismatch(format!(
                "tensor with shape {:?} requires {} elements, got {}",
                shape,
                numel,
                data.len()
            )));
        }
        if shape.is_empty() {
            return Err(SolverError::DimensionMismatch(
                "tensor must have at least one dimension".to_string(),
            ));
        }
        Ok(Self { shape, data })
    }

    /// Creates a zero tensor with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; numel],
        }
    }

    /// Returns the number of dimensions (modes).
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Returns a reference to the data.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Computes the linear index from multi-dimensional indices (row-major).
    fn linear_index(&self, indices: &[usize]) -> SolverResult<usize> {
        if indices.len() != self.shape.len() {
            return Err(SolverError::DimensionMismatch(format!(
                "expected {} indices, got {}",
                self.shape.len(),
                indices.len()
            )));
        }
        let mut idx = 0;
        let mut stride = 1;
        for d in (0..self.shape.len()).rev() {
            if indices[d] >= self.shape[d] {
                return Err(SolverError::DimensionMismatch(format!(
                    "index {} out of range for dimension {} with size {}",
                    indices[d], d, self.shape[d]
                )));
            }
            idx += indices[d] * stride;
            stride *= self.shape[d];
        }
        Ok(idx)
    }

    /// Gets an element by multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> SolverResult<f64> {
        let idx = self.linear_index(indices)?;
        Ok(self.data[idx])
    }

    /// Sets an element by multi-dimensional index.
    pub fn set(&mut self, indices: &[usize], value: f64) -> SolverResult<()> {
        let idx = self.linear_index(indices)?;
        self.data[idx] = value;
        Ok(())
    }

    /// Mode-n unfolding (matricization).
    ///
    /// Rearranges the tensor into a matrix where mode `n` becomes the row
    /// dimension and all other modes are collapsed into the column dimension.
    pub fn unfold(&self, mode: usize) -> SolverResult<Matrix> {
        if mode >= self.ndim() {
            return Err(SolverError::DimensionMismatch(format!(
                "mode {} out of range for {}-dimensional tensor",
                mode,
                self.ndim()
            )));
        }

        let rows = self.shape[mode];
        let cols = self.numel() / rows;
        let mut mat = Matrix::zeros(rows, cols);

        let ndim = self.ndim();
        let mut indices = vec![0usize; ndim];
        for flat in 0..self.numel() {
            // Compute multi-index from flat index
            let mut rem = flat;
            for d in (0..ndim).rev() {
                indices[d] = rem % self.shape[d];
                rem /= self.shape[d];
            }

            // Row = index along unfolded mode
            let row = indices[mode];

            // Column = multi-index of remaining modes, in order
            let mut col = 0;
            let mut col_stride = 1;
            for d in (0..ndim).rev() {
                if d != mode {
                    col += indices[d] * col_stride;
                    col_stride *= self.shape[d];
                }
            }

            mat.set(row, col, self.data[flat]);
        }

        Ok(mat)
    }

    /// Folds a matrix back into a tensor (inverse of unfold).
    pub fn fold(matrix: &Matrix, mode: usize, shape: &[usize]) -> SolverResult<Tensor> {
        let ndim = shape.len();
        if mode >= ndim {
            return Err(SolverError::DimensionMismatch(format!(
                "mode {} out of range for {}-dimensional tensor",
                mode, ndim
            )));
        }
        if matrix.rows != shape[mode] {
            return Err(SolverError::DimensionMismatch(format!(
                "matrix rows {} != shape[{}] = {}",
                matrix.rows, mode, shape[mode]
            )));
        }

        let numel: usize = shape.iter().product();
        let mut data = vec![0.0; numel];

        let mut indices = vec![0usize; ndim];
        for (flat, datum) in data.iter_mut().enumerate() {
            let mut rem = flat;
            for d in (0..ndim).rev() {
                indices[d] = rem % shape[d];
                rem /= shape[d];
            }

            let row = indices[mode];
            let mut col = 0;
            let mut col_stride = 1;
            for d in (0..ndim).rev() {
                if d != mode {
                    col += indices[d] * col_stride;
                    col_stride *= shape[d];
                }
            }

            *datum = matrix.get(row, col);
        }

        Ok(Tensor {
            shape: shape.to_vec(),
            data,
        })
    }

    /// Frobenius norm of the tensor.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Khatri-Rao product (column-wise Kronecker product).
///
/// Given A (I×R) and B (J×R), produces C ((I*J)×R) where
/// `c_{(i-1)*J+j, r} = a_{i,r} * b_{j,r}`.
pub fn khatri_rao_product(a: &Matrix, b: &Matrix) -> SolverResult<Matrix> {
    if a.cols != b.cols {
        return Err(SolverError::DimensionMismatch(format!(
            "khatri-rao requires same number of columns: {} vs {}",
            a.cols, b.cols
        )));
    }
    let r = a.cols;
    let rows = a.rows * b.rows;
    let mut out = Matrix::zeros(rows, r);
    for col in 0..r {
        for i in 0..a.rows {
            for j in 0..b.rows {
                out.set(i * b.rows + j, col, a.get(i, col) * b.get(j, col));
            }
        }
    }
    Ok(out)
}

/// Element-wise (Hadamard) product of two matrices.
pub fn hadamard_product(a: &Matrix, b: &Matrix) -> SolverResult<Matrix> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(SolverError::DimensionMismatch(format!(
            "hadamard requires same dimensions: {}x{} vs {}x{}",
            a.rows, a.cols, b.rows, b.cols
        )));
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x * y)
        .collect();
    Matrix::new(a.rows, a.cols, data)
}

/// Mode-n product: multiplies a tensor by a matrix along mode n.
///
/// If tensor has shape `[I_0, ..., I_n, ..., I_{N-1}]` and matrix is `J × I_n`,
/// the result has shape `[I_0, ..., J, ..., I_{N-1}]`.
pub fn mode_n_product(tensor: &Tensor, matrix: &Matrix, mode: usize) -> SolverResult<Tensor> {
    if mode >= tensor.ndim() {
        return Err(SolverError::DimensionMismatch(format!(
            "mode {} out of range for {}-dimensional tensor",
            mode,
            tensor.ndim()
        )));
    }
    if matrix.cols != tensor.shape()[mode] {
        return Err(SolverError::DimensionMismatch(format!(
            "matrix cols {} != tensor dimension {} size {}",
            matrix.cols,
            mode,
            tensor.shape()[mode]
        )));
    }

    let unfolded = tensor.unfold(mode)?;
    let result_mat = matrix.matmul(&unfolded)?;

    let mut new_shape = tensor.shape().to_vec();
    new_shape[mode] = matrix.rows;

    Tensor::fold(&result_mat, mode, &new_shape)
}

// ---------------------------------------------------------------------------
// CP Decomposition
// ---------------------------------------------------------------------------

/// CP (CANDECOMP/PARAFAC) decomposition result.
///
/// Represents a tensor as a weighted sum of rank-one tensors:
/// `X ≈ Σ_r λ_r · a_r^(1) ◦ a_r^(2) ◦ ... ◦ a_r^(N)`
#[derive(Debug, Clone)]
pub struct CpDecomposition {
    /// Weights (lambda) for each rank-one component.
    pub weights: Vec<f64>,
    /// Factor matrices, one per mode. Factor\[n\] is `I_n × R`.
    pub factors: Vec<Matrix>,
}

impl CpDecomposition {
    /// Returns the rank of the decomposition.
    pub fn rank(&self) -> usize {
        self.weights.len()
    }

    /// Reconstructs the full tensor from the CP factors.
    pub fn reconstruct(&self) -> SolverResult<Tensor> {
        if self.factors.is_empty() {
            return Err(SolverError::InternalError(
                "CP decomposition has no factors".to_string(),
            ));
        }

        let shape: Vec<usize> = self.factors.iter().map(|f| f.rows).collect();
        let numel: usize = shape.iter().product();
        let ndim = shape.len();
        let rank = self.rank();
        let mut data = vec![0.0; numel];

        let mut indices = vec![0usize; ndim];
        for (flat, datum) in data.iter_mut().enumerate() {
            let mut rem = flat;
            for d in (0..ndim).rev() {
                indices[d] = rem % shape[d];
                rem /= shape[d];
            }

            let mut val = 0.0;
            for r in 0..rank {
                let mut term = self.weights[r];
                for (d, idx) in indices.iter().enumerate() {
                    term *= self.factors[d].get(*idx, r);
                }
                val += term;
            }
            *datum = val;
        }

        Tensor::new(shape, data)
    }

    /// Computes the relative fit error: `||X - X_hat|| / ||X||`.
    pub fn fit_error(&self, original: &Tensor) -> SolverResult<f64> {
        let reconstructed = self.reconstruct()?;
        let orig_norm = original.frobenius_norm();
        if orig_norm < 1e-15 {
            return Ok(0.0);
        }
        let diff_norm: f64 = original
            .data()
            .iter()
            .zip(reconstructed.data().iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        Ok(diff_norm / orig_norm)
    }
}

/// Configuration for CP-ALS algorithm.
#[derive(Debug, Clone)]
pub struct CpAlsConfig {
    /// Target rank.
    pub rank: usize,
    /// Maximum number of ALS iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Whether to normalize factor columns after each iteration.
    pub normalize_factors: bool,
}

impl Default for CpAlsConfig {
    fn default() -> Self {
        Self {
            rank: 3,
            max_iterations: 100,
            tolerance: 1e-8,
            normalize_factors: true,
        }
    }
}

/// CP decomposition via Alternating Least Squares (ALS).
///
/// For each mode n, we fix all other factor matrices and solve the least
/// squares problem for factor matrix A^(n).
pub fn cp_als(tensor: &Tensor, config: &CpAlsConfig) -> SolverResult<CpDecomposition> {
    let ndim = tensor.ndim();
    let rank = config.rank;

    if rank == 0 {
        return Err(SolverError::InternalError(
            "CP rank must be positive".to_string(),
        ));
    }

    // Initialize factor matrices deterministically
    let mut factors: Vec<Matrix> = Vec::with_capacity(ndim);
    for n in 0..ndim {
        let rows = tensor.shape()[n];
        let mut data = vec![0.0; rows * rank];
        for i in 0..rows {
            for r in 0..rank {
                data[i * rank + r] = ((i + 1) as f64 * (r + 1) as f64 * 0.37).sin().abs() + 0.01;
            }
        }
        factors.push(Matrix::new(rows, rank, data)?);
    }

    let mut weights = vec![1.0; rank];
    let mut prev_fit = f64::MAX;

    for _iter in 0..config.max_iterations {
        for n in 0..ndim {
            // Build the list of modes excluding n, in the order that
            // matches the column layout of the mode-n unfolding.
            // The unfolding column index cycles through modes != n
            // in the order they appear (skipping n), from last to first.
            let other_modes: Vec<usize> = (0..ndim).filter(|&m| m != n).collect();

            // Khatri-Rao product of factors for other modes (in column order)
            let mut kr = factors[other_modes[0]].clone();
            for &m in &other_modes[1..] {
                kr = khatri_rao_product(&kr, &factors[m])?;
            }

            // Gram matrix = Hadamard of (F_m^T * F_m) for all m != n
            let mut gram = {
                let ft = factors[other_modes[0]].transpose();
                ft.matmul(&factors[other_modes[0]])?
            };
            for &m in &other_modes[1..] {
                let ft = factors[m].transpose();
                let g = ft.matmul(&factors[m])?;
                gram = hadamard_product(&gram, &g)?;
            }

            // Unfold tensor along mode n
            let x_n = tensor.unfold(n)?;

            // V = X_(n) * KR
            let v = x_n.matmul(&kr)?;

            // Solve A^(n) = V * gram^{-1} via normal equations
            let gram_inv = invert_small_matrix(&gram)?;
            factors[n] = v.matmul(&gram_inv)?;
        }

        // Normalize factors
        if config.normalize_factors {
            for w in weights.iter_mut() {
                *w = 1.0;
            }
            for factor in factors.iter_mut() {
                let norms = factor.normalize_columns();
                for (w, &norm) in weights.iter_mut().zip(norms.iter()) {
                    *w *= norm;
                }
            }
        }

        // Check convergence
        let decomp = CpDecomposition {
            weights: weights.clone(),
            factors: factors.clone(),
        };
        let fit = decomp.fit_error(tensor).unwrap_or(f64::MAX);
        if (prev_fit - fit).abs() < config.tolerance {
            return Ok(decomp);
        }
        prev_fit = fit;
    }

    Ok(CpDecomposition { weights, factors })
}

/// Invert a small R×R matrix via Gauss-Jordan elimination.
fn invert_small_matrix(m: &Matrix) -> SolverResult<Matrix> {
    if m.rows != m.cols {
        return Err(SolverError::DimensionMismatch(
            "matrix must be square to invert".to_string(),
        ));
    }
    let n = m.rows;
    // Augmented matrix [A | I]
    let mut aug = Matrix::zeros(n, 2 * n);
    for r in 0..n {
        for c in 0..n {
            aug.set(r, c, m.get(r, c));
        }
        aug.set(r, n + r, 1.0);
    }

    for col in 0..n {
        // Find pivot
        let mut max_val = aug.get(col, col).abs();
        let mut max_row = col;
        for r in (col + 1)..n {
            let v = aug.get(r, col).abs();
            if v > max_val {
                max_val = v;
                max_row = r;
            }
        }
        if max_val < 1e-14 {
            return Err(SolverError::SingularMatrix);
        }

        // Swap rows
        if max_row != col {
            for c in 0..(2 * n) {
                let tmp = aug.get(col, c);
                aug.set(col, c, aug.get(max_row, c));
                aug.set(max_row, c, tmp);
            }
        }

        // Scale pivot row
        let pivot = aug.get(col, col);
        for c in 0..(2 * n) {
            aug.set(col, c, aug.get(col, c) / pivot);
        }

        // Eliminate other rows
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug.get(r, col);
            for c in 0..(2 * n) {
                let v = aug.get(r, c) - factor * aug.get(col, c);
                aug.set(r, c, v);
            }
        }
    }

    // Extract inverse
    let mut inv = Matrix::zeros(n, n);
    for r in 0..n {
        for c in 0..n {
            inv.set(r, c, aug.get(r, n + c));
        }
    }
    Ok(inv)
}

// ---------------------------------------------------------------------------
// Tucker Decomposition
// ---------------------------------------------------------------------------

/// Tucker decomposition result.
///
/// Represents a tensor as a core tensor multiplied by factor matrices:
/// `X ≈ G ×_1 U^(1) ×_2 U^(2) ... ×_N U^(N)`
#[derive(Debug, Clone)]
pub struct TuckerDecomposition {
    /// Core tensor with shape `[R_1, R_2, ..., R_N]`.
    pub core: Tensor,
    /// Factor matrices, one per mode. Factor\[n\] is `I_n × R_n`.
    pub factors: Vec<Matrix>,
}

impl TuckerDecomposition {
    /// Reconstructs the full tensor from the Tucker factors.
    pub fn reconstruct(&self) -> SolverResult<Tensor> {
        let mut result = self.core.clone();
        for (n, factor) in self.factors.iter().enumerate() {
            result = mode_n_product(&result, factor, n)?;
        }
        Ok(result)
    }

    /// Computes the relative fit error: `||X - X_hat|| / ||X||`.
    pub fn fit_error(&self, original: &Tensor) -> SolverResult<f64> {
        let reconstructed = self.reconstruct()?;
        let orig_norm = original.frobenius_norm();
        if orig_norm < 1e-15 {
            return Ok(0.0);
        }
        let diff_norm: f64 = original
            .data()
            .iter()
            .zip(reconstructed.data().iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        Ok(diff_norm / orig_norm)
    }

    /// Compression ratio: original elements / decomposition elements.
    pub fn compression_ratio(&self, original_shape: &[usize]) -> f64 {
        let original_size: usize = original_shape.iter().product();
        let core_size = self.core.numel();
        let factor_size: usize = self.factors.iter().map(|f| f.rows * f.cols).sum();
        let decomp_size = core_size + factor_size;
        if decomp_size == 0 {
            return 0.0;
        }
        original_size as f64 / decomp_size as f64
    }
}

/// Configuration for Tucker decomposition.
#[derive(Debug, Clone)]
pub struct TuckerConfig {
    /// Target ranks, one per mode.
    pub ranks: Vec<usize>,
    /// Maximum number of HOOI iterations (unused for HOSVD).
    pub max_iterations: usize,
    /// Convergence tolerance for HOOI.
    pub tolerance: f64,
}

impl Default for TuckerConfig {
    fn default() -> Self {
        Self {
            ranks: vec![2, 2, 2],
            max_iterations: 50,
            tolerance: 1e-8,
        }
    }
}

/// Higher-Order SVD (HOSVD) for Tucker decomposition.
///
/// Non-iterative method: for each mode, compute the truncated SVD of the
/// mode-n unfolding and use the left singular vectors as the factor matrix.
pub fn tucker_hosvd(tensor: &Tensor, config: &TuckerConfig) -> SolverResult<TuckerDecomposition> {
    let ndim = tensor.ndim();
    if config.ranks.len() != ndim {
        return Err(SolverError::DimensionMismatch(format!(
            "Tucker config requires {} ranks (one per mode), got {}",
            ndim,
            config.ranks.len()
        )));
    }

    // Compute factor matrices via mode-n SVD
    let mut factors: Vec<Matrix> = Vec::with_capacity(ndim);
    for n in 0..ndim {
        let unfolded = tensor.unfold(n)?;
        let rank_n = config.ranks[n].min(tensor.shape()[n]);
        let (u, _sigma, _v) = unfolded.svd_truncated(rank_n)?;
        factors.push(u);
    }

    // Core tensor = X ×_1 U^(1)^T ×_2 U^(2)^T ... ×_N U^(N)^T
    let mut core = tensor.clone();
    for (n, factor) in factors.iter().enumerate() {
        let ft = factor.transpose();
        core = mode_n_product(&core, &ft, n)?;
    }

    Ok(TuckerDecomposition { core, factors })
}

/// Higher-Order Orthogonal Iteration (HOOI) for Tucker decomposition.
///
/// Iterative refinement of HOSVD. Alternately optimizes each factor matrix
/// while holding others fixed.
pub fn tucker_hooi(tensor: &Tensor, config: &TuckerConfig) -> SolverResult<TuckerDecomposition> {
    let ndim = tensor.ndim();
    if config.ranks.len() != ndim {
        return Err(SolverError::DimensionMismatch(format!(
            "Tucker config requires {} ranks (one per mode), got {}",
            ndim,
            config.ranks.len()
        )));
    }

    // Initialize with HOSVD
    let mut decomp = tucker_hosvd(tensor, config)?;
    let mut prev_core_norm = decomp.core.frobenius_norm();

    for _iter in 0..config.max_iterations {
        for n in 0..ndim {
            // Compute Y = X ×_{m≠n} U^(m)^T
            let mut y = tensor.clone();
            for (m, factor) in decomp.factors.iter().enumerate() {
                if m != n {
                    let ft = factor.transpose();
                    y = mode_n_product(&y, &ft, m)?;
                }
            }

            // SVD of mode-n unfolding of Y
            let y_n = y.unfold(n)?;
            let rank_n = config.ranks[n].min(tensor.shape()[n]);
            let (u, _sigma, _v) = y_n.svd_truncated(rank_n)?;
            decomp.factors[n] = u;
        }

        // Recompute core
        let mut core = tensor.clone();
        for (n, factor) in decomp.factors.iter().enumerate() {
            let ft = factor.transpose();
            core = mode_n_product(&core, &ft, n)?;
        }
        decomp.core = core;

        // Check convergence
        let core_norm = decomp.core.frobenius_norm();
        if (core_norm - prev_core_norm).abs() / (prev_core_norm + 1e-15) < config.tolerance {
            break;
        }
        prev_core_norm = core_norm;
    }

    Ok(decomp)
}

// ---------------------------------------------------------------------------
// Tensor-Train (TT) Decomposition
// ---------------------------------------------------------------------------

/// Tensor-Train (TT) decomposition result.
///
/// Represents a tensor as a chain of 3D cores:
/// `X(i_1, ..., i_N) = G_1(i_1) · G_2(i_2) · ... · G_N(i_N)`
/// where each `G_k(i_k)` is an `r_{k-1} × r_k` matrix slice, and
/// core k has shape `r_{k-1} × n_k × r_k`.
#[derive(Debug, Clone)]
pub struct TtDecomposition {
    /// TT-cores, each a 3D tensor of shape `[r_{k-1}, n_k, r_k]`.
    pub cores: Vec<Tensor>,
}

impl TtDecomposition {
    /// Returns the TT-ranks `[r_0, r_1, ..., r_N]` where r_0 = r_N = 1.
    pub fn ranks(&self) -> Vec<usize> {
        let mut ranks = Vec::with_capacity(self.cores.len() + 1);
        if self.cores.is_empty() {
            return ranks;
        }
        ranks.push(self.cores[0].shape()[0]);
        for core in &self.cores {
            ranks.push(core.shape()[2]);
        }
        ranks
    }

    /// Reconstructs the full tensor from TT-cores.
    pub fn reconstruct(&self) -> SolverResult<Tensor> {
        if self.cores.is_empty() {
            return Err(SolverError::InternalError(
                "TT decomposition has no cores".to_string(),
            ));
        }

        let shape: Vec<usize> = self.cores.iter().map(|c| c.shape()[1]).collect();
        let ndim = shape.len();
        let numel: usize = shape.iter().product();
        let mut data = vec![0.0; numel];

        let mut indices = vec![0usize; ndim];
        for (flat, datum) in data.iter_mut().enumerate() {
            let mut rem = flat;
            for d in (0..ndim).rev() {
                indices[d] = rem % shape[d];
                rem /= shape[d];
            }

            // Compute the matrix product G_1(i_1) * G_2(i_2) * ... * G_N(i_N)
            // Start with G_1(i_1): shape [r_0, r_1] = [1, r_1]
            let core0 = &self.cores[0];
            let r1 = core0.shape()[2];
            let mut current: Vec<f64> = (0..r1)
                .map(|j| core0.get(&[0, indices[0], j]))
                .collect::<SolverResult<_>>()?;

            for (k, &idx_k) in indices.iter().enumerate().skip(1) {
                let core_k = &self.cores[k];
                let r_next = core_k.shape()[2];
                let mut next = vec![0.0; r_next];
                for (j, nj) in next.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for (i, &ci) in current.iter().enumerate() {
                        sum += ci * core_k.get(&[i, idx_k, j])?;
                    }
                    *nj = sum;
                }
                current = next;
            }

            *datum = current[0]; // r_N = 1, so scalar result
        }

        Tensor::new(shape, data)
    }

    /// Computes the relative fit error: `||X - X_hat|| / ||X||`.
    pub fn fit_error(&self, original: &Tensor) -> SolverResult<f64> {
        let reconstructed = self.reconstruct()?;
        let orig_norm = original.frobenius_norm();
        if orig_norm < 1e-15 {
            return Ok(0.0);
        }
        let diff_norm: f64 = original
            .data()
            .iter()
            .zip(reconstructed.data().iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        Ok(diff_norm / orig_norm)
    }

    /// Compression ratio: original elements / decomposition elements.
    pub fn compression_ratio(&self, original_shape: &[usize]) -> f64 {
        let original_size: usize = original_shape.iter().product();
        let decomp_size: usize = self.cores.iter().map(|c| c.numel()).sum();
        if decomp_size == 0 {
            return 0.0;
        }
        original_size as f64 / decomp_size as f64
    }

    /// TT-rounding: truncates TT-ranks to at most `max_rank`.
    ///
    /// Performs left-to-right QR orthogonalization, then right-to-left
    /// truncated SVD to reduce ranks.
    pub fn tt_round(&self, max_rank: usize) -> SolverResult<TtDecomposition> {
        if self.cores.is_empty() {
            return Err(SolverError::InternalError(
                "TT decomposition has no cores".to_string(),
            ));
        }

        let ndim = self.cores.len();
        let mut cores = self.cores.clone();

        // Left-to-right QR sweep (orthogonalize)
        for k in 0..(ndim - 1) {
            let r_prev = cores[k].shape()[0];
            let n_k = cores[k].shape()[1];
            let r_next = cores[k].shape()[2];

            // Reshape core_k to (r_prev * n_k) × r_next matrix
            let mat = Matrix::new(r_prev * n_k, r_next, cores[k].data().to_vec())?;
            let (q, r_mat) = qr_gram_schmidt(&mat);

            let new_r = q.cols;
            cores[k] = Tensor::new(vec![r_prev, n_k, new_r], q.data.clone())?;

            // Absorb R into next core
            let r_next2 = cores[k + 1].shape()[2];
            let n_next = cores[k + 1].shape()[1];
            let next_mat = Matrix::new(r_next, n_next * r_next2, cores[k + 1].data().to_vec())?;
            let absorbed = r_mat.matmul(&next_mat)?;
            cores[k + 1] = Tensor::new(vec![new_r, n_next, r_next2], absorbed.data.clone())?;
        }

        // Right-to-left SVD sweep (truncate)
        for k in (1..ndim).rev() {
            let r_prev = cores[k].shape()[0];
            let n_k = cores[k].shape()[1];
            let r_next = cores[k].shape()[2];

            // Reshape core_k to r_prev × (n_k * r_next) matrix
            let mat = Matrix::new(r_prev, n_k * r_next, cores[k].data().to_vec())?;
            let trunc_rank = max_rank.min(r_prev).min(n_k * r_next);
            let (u, sigma, v) = mat.svd_truncated(trunc_rank)?;

            // New core_k = diag(sigma) * V^T reshaped to [trunc_rank, n_k, r_next]
            let mut sv = Matrix::zeros(trunc_rank, n_k * r_next);
            for (i, &si) in sigma.iter().enumerate().take(trunc_rank) {
                for j in 0..(n_k * r_next) {
                    sv.set(i, j, si * v.get(j, i));
                }
            }
            cores[k] = Tensor::new(vec![trunc_rank, n_k, r_next], sv.data.clone())?;

            // Absorb U into previous core
            let prev_r_prev = cores[k - 1].shape()[0];
            let prev_n = cores[k - 1].shape()[1];
            let prev_mat = Matrix::new(prev_r_prev * prev_n, r_prev, cores[k - 1].data().to_vec())?;
            let absorbed = prev_mat.matmul(&u)?;
            cores[k - 1] =
                Tensor::new(vec![prev_r_prev, prev_n, trunc_rank], absorbed.data.clone())?;
        }

        Ok(TtDecomposition { cores })
    }
}

/// Configuration for TT-SVD algorithm.
#[derive(Debug, Clone)]
pub struct TtConfig {
    /// Maximum TT-rank.
    pub max_rank: usize,
    /// Truncation tolerance.
    pub tolerance: f64,
}

impl Default for TtConfig {
    fn default() -> Self {
        Self {
            max_rank: 10,
            tolerance: 1e-8,
        }
    }
}

/// TT-SVD decomposition.
///
/// Decomposes a tensor into Tensor-Train format via sequential SVDs
/// from left to right.
pub fn tt_svd(tensor: &Tensor, config: &TtConfig) -> SolverResult<TtDecomposition> {
    let ndim = tensor.ndim();
    if ndim < 2 {
        // For 1D tensors, wrap as a single core [1, n, 1]
        let n = tensor.shape()[0];
        let core = Tensor::new(vec![1, n, 1], tensor.data().to_vec())?;
        return Ok(TtDecomposition { cores: vec![core] });
    }

    if config.max_rank == 0 {
        return Err(SolverError::InternalError(
            "TT max_rank must be positive".to_string(),
        ));
    }

    let shape = tensor.shape().to_vec();
    let mut cores: Vec<Tensor> = Vec::with_capacity(ndim);
    let mut remaining_data = tensor.data().to_vec();
    let mut r_prev = 1usize;

    for k in 0..(ndim - 1) {
        let n_k = shape[k];
        let remaining_size: usize = shape[(k + 1)..].iter().product();

        // Reshape to (r_prev * n_k) × remaining
        let rows = r_prev * n_k;
        let cols = remaining_size;

        // Handle edge case where data might have been truncated
        let actual_len = remaining_data.len();
        if actual_len != rows * cols {
            return Err(SolverError::InternalError(format!(
                "TT-SVD reshape error at mode {}: expected {} elements, have {}",
                k,
                rows * cols,
                actual_len
            )));
        }

        let mat = Matrix::new(rows, cols, remaining_data)?;

        // Truncated SVD
        let trunc_rank = config.max_rank.min(rows).min(cols);
        let (u, sigma, v) = mat.svd_truncated(trunc_rank)?;

        // Determine effective rank (truncate small singular values)
        let total_sv_norm: f64 = sigma.iter().map(|s| s * s).sum::<f64>().sqrt();
        let mut effective_rank = trunc_rank;
        if total_sv_norm > 1e-15 {
            let mut accumulated = 0.0;
            for (i, &s) in sigma.iter().enumerate().rev() {
                accumulated += s * s;
                if accumulated.sqrt() / total_sv_norm > config.tolerance {
                    effective_rank = i + 1;
                    break;
                }
            }
        }
        effective_rank = effective_rank.min(trunc_rank);
        if effective_rank == 0 {
            effective_rank = 1;
        }

        // Core_k = U reshaped to [r_prev, n_k, effective_rank]
        let mut core_data = vec![0.0; r_prev * n_k * effective_rank];
        for i in 0..rows {
            for j in 0..effective_rank {
                core_data[i * effective_rank + j] = u.get(i, j);
            }
        }
        cores.push(Tensor::new(vec![r_prev, n_k, effective_rank], core_data)?);

        // Remaining = diag(sigma) * V^T
        let new_cols = cols;
        let mut new_remaining = vec![0.0; effective_rank * new_cols];
        for i in 0..effective_rank {
            for j in 0..new_cols {
                new_remaining[i * new_cols + j] = sigma[i] * v.get(j, i);
            }
        }
        remaining_data = new_remaining;
        r_prev = effective_rank;
    }

    // Last core: [r_prev, n_{N-1}, 1]
    let n_last = shape[ndim - 1];
    if remaining_data.len() != r_prev * n_last {
        return Err(SolverError::InternalError(format!(
            "TT-SVD final reshape error: expected {} elements, have {}",
            r_prev * n_last,
            remaining_data.len()
        )));
    }
    let mut last_core_data = vec![0.0; r_prev * n_last];
    for i in 0..r_prev {
        for j in 0..n_last {
            last_core_data[i * n_last + j] = remaining_data[i * n_last + j];
        }
    }
    cores.push(Tensor::new(vec![r_prev, n_last, 1], last_core_data)?);

    Ok(TtDecomposition { cores })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: creates a small 3×4×2 test tensor.
    fn make_test_tensor_3d() -> Tensor {
        let shape = vec![3, 4, 2];
        let data: Vec<f64> = (0..24).map(|i| (i as f64) * 0.5 + 1.0).collect();
        Tensor::new(shape, data).expect("failed to create test tensor")
    }

    /// Helper: creates a rank-1 tensor: outer product of [1,2,3] x [1,2] x [1,2,3,4].
    fn make_rank1_tensor() -> Tensor {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0];
        let c = [1.0, 2.0, 3.0, 4.0];
        let shape = vec![3, 2, 4];
        let mut data = vec![0.0; 24];
        for i in 0..3 {
            for j in 0..2 {
                for k in 0..4 {
                    data[i * 8 + j * 4 + k] = a[i] * b[j] * c[k];
                }
            }
        }
        Tensor::new(shape, data).expect("failed to create rank-1 tensor")
    }

    #[test]
    fn test_tensor_creation_and_indexing() {
        let t = make_test_tensor_3d();
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.shape(), &[3, 4, 2]);
        assert_eq!(t.numel(), 24);

        // First element
        let v = t.get(&[0, 0, 0]).expect("get failed");
        assert!((v - 1.0).abs() < 1e-12);

        // Last element
        let v = t.get(&[2, 3, 1]).expect("get failed");
        assert!((v - 12.5).abs() < 1e-12);
    }

    #[test]
    fn test_tensor_set() {
        let mut t = make_test_tensor_3d();
        t.set(&[1, 2, 0], 99.0).expect("set failed");
        let v = t.get(&[1, 2, 0]).expect("get failed");
        assert!((v - 99.0).abs() < 1e-12);
    }

    #[test]
    fn test_tensor_index_out_of_range() {
        let t = make_test_tensor_3d();
        assert!(t.get(&[3, 0, 0]).is_err());
        assert!(t.get(&[0, 4, 0]).is_err());
        assert!(t.get(&[0, 0]).is_err()); // wrong ndim
    }

    #[test]
    fn test_mode_n_unfolding_and_folding_roundtrip() {
        let t = make_test_tensor_3d();
        for mode in 0..3 {
            let mat = t.unfold(mode).expect("unfold failed");

            // Check dimensions
            assert_eq!(mat.rows, t.shape()[mode]);
            assert_eq!(mat.rows * mat.cols, t.numel());

            // Fold back
            let recovered = Tensor::fold(&mat, mode, t.shape()).expect("fold failed");
            for i in 0..t.numel() {
                assert!(
                    (t.data()[i] - recovered.data()[i]).abs() < 1e-12,
                    "mismatch at element {} for mode {} unfold/fold",
                    i,
                    mode
                );
            }
        }
    }

    #[test]
    fn test_matrix_svd_truncated() {
        // Create a rank-2 matrix
        let mut data = vec![0.0; 12];
        for i in 0..4 {
            for j in 0..3 {
                data[i * 3 + j] = (i + 1) as f64 * (j + 1) as f64;
            }
        }
        let m = Matrix::new(4, 3, data).expect("matrix creation failed");
        let (u, sigma, v) = m.svd_truncated(2).expect("svd failed");

        assert_eq!(u.rows, 4);
        assert_eq!(u.cols, 2);
        assert_eq!(sigma.len(), 2);
        assert_eq!(v.rows, 3);
        assert_eq!(v.cols, 2);

        // Singular values should be non-negative and decreasing
        assert!(sigma[0] >= sigma[1]);
        assert!(sigma[0] > 0.0);

        // Reconstruct: U * diag(sigma) * V^T should approximate original
        let mut reconstructed = Matrix::zeros(4, 3);
        for i in 0..4 {
            for j in 0..3 {
                let mut val = 0.0;
                for (r, sigma_r) in sigma.iter().enumerate().take(2) {
                    val += u.get(i, r) * sigma_r * v.get(j, r);
                }
                reconstructed.set(i, j, val);
            }
        }
        let error = (m.data)
            .iter()
            .zip(reconstructed.data.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let norm = m.frobenius_norm();
        // This is rank-1 matrix, so rank-2 truncation should be exact
        assert!(
            error / norm < 0.05,
            "SVD reconstruction error too large: {}",
            error / norm
        );
    }

    #[test]
    fn test_khatri_rao_product() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("a");
        let b = Matrix::new(3, 2, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).expect("b");
        let kr = khatri_rao_product(&a, &b).expect("kr");

        assert_eq!(kr.rows, 6);
        assert_eq!(kr.cols, 2);

        // Column 0: [1*5, 1*7, 1*9, 3*5, 3*7, 3*9]
        assert!((kr.get(0, 0) - 5.0).abs() < 1e-12);
        assert!((kr.get(1, 0) - 7.0).abs() < 1e-12);
        assert!((kr.get(2, 0) - 9.0).abs() < 1e-12);
        assert!((kr.get(3, 0) - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_hadamard_product() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("a");
        let b = Matrix::new(2, 3, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).expect("b");
        let h = hadamard_product(&a, &b).expect("hadamard");

        assert_eq!(h.rows, 2);
        assert_eq!(h.cols, 3);
        assert!((h.get(0, 0) - 7.0).abs() < 1e-12);
        assert!((h.get(1, 2) - 72.0).abs() < 1e-12);
    }

    #[test]
    fn test_hadamard_product_dimension_mismatch() {
        let a = Matrix::new(2, 3, vec![0.0; 6]).expect("a");
        let b = Matrix::new(3, 2, vec![0.0; 6]).expect("b");
        assert!(hadamard_product(&a, &b).is_err());
    }

    #[test]
    fn test_mode_n_product() {
        let t = make_test_tensor_3d(); // 3×4×2
        let m = Matrix::new(5, 4, vec![0.1; 20]).expect("matrix");
        let result = mode_n_product(&t, &m, 1).expect("mode_n_product");
        assert_eq!(result.shape(), &[3, 5, 2]);
    }

    #[test]
    fn test_cp_als_rank1_tensor() {
        let t = make_rank1_tensor();
        let config = CpAlsConfig {
            rank: 1,
            max_iterations: 200,
            tolerance: 1e-10,
            normalize_factors: true,
        };
        let decomp = cp_als(&t, &config).expect("cp_als failed");

        assert_eq!(decomp.rank(), 1);
        assert_eq!(decomp.factors.len(), 3);
        assert_eq!(decomp.factors[0].rows, 3);
        assert_eq!(decomp.factors[1].rows, 2);
        assert_eq!(decomp.factors[2].rows, 4);

        // Rank-1 tensor should be recovered nearly exactly
        let error = decomp.fit_error(&t).expect("fit_error failed");
        assert!(error < 0.01, "CP-ALS rank-1 error too large: {}", error);
    }

    #[test]
    fn test_cp_als_rank3_tensor() {
        // Create a rank-3 tensor by summing 3 rank-1 components
        let shape = vec![4, 3, 5];
        let numel = 60;
        let mut data = vec![0.0; numel];
        for r in 0..3 {
            for i in 0..4 {
                for j in 0..3 {
                    for k in 0..5 {
                        let a_val = ((i + 1) as f64 * (r + 1) as f64 * 0.3).sin();
                        let b_val = ((j + 1) as f64 * (r + 1) as f64 * 0.5).cos();
                        let c_val = ((k + 1) as f64 * (r + 1) as f64 * 0.7).sin();
                        data[i * 15 + j * 5 + k] += a_val * b_val * c_val;
                    }
                }
            }
        }
        let t = Tensor::new(shape, data).expect("tensor");

        let config = CpAlsConfig {
            rank: 3,
            max_iterations: 300,
            tolerance: 1e-10,
            normalize_factors: true,
        };
        let decomp = cp_als(&t, &config).expect("cp_als failed");
        assert_eq!(decomp.rank(), 3);

        let error = decomp.fit_error(&t).expect("fit_error");
        assert!(error < 0.5, "CP-ALS rank-3 error too large: {}", error);
    }

    #[test]
    fn test_cp_reconstruction() {
        let t = make_rank1_tensor();
        let config = CpAlsConfig {
            rank: 1,
            max_iterations: 200,
            tolerance: 1e-10,
            normalize_factors: true,
        };
        let decomp = cp_als(&t, &config).expect("cp_als");
        let recon = decomp.reconstruct().expect("reconstruct");

        assert_eq!(recon.shape(), t.shape());
        assert_eq!(recon.numel(), t.numel());
    }

    #[test]
    fn test_tucker_hosvd() {
        let t = make_test_tensor_3d();
        let config = TuckerConfig {
            ranks: vec![2, 3, 2],
            max_iterations: 50,
            tolerance: 1e-8,
        };
        let decomp = tucker_hosvd(&t, &config).expect("tucker_hosvd");

        assert_eq!(decomp.core.shape(), &[2, 3, 2]);
        assert_eq!(decomp.factors.len(), 3);
        assert_eq!(decomp.factors[0].rows, 3); // original dim
        assert_eq!(decomp.factors[0].cols, 2); // rank
        assert_eq!(decomp.factors[1].rows, 4);
        assert_eq!(decomp.factors[1].cols, 3);
        assert_eq!(decomp.factors[2].rows, 2);
        assert_eq!(decomp.factors[2].cols, 2);

        // Should reconstruct reasonably well
        let error = decomp.fit_error(&t).expect("fit_error");
        assert!(error < 0.5, "Tucker HOSVD error too large: {}", error);
    }

    #[test]
    fn test_tucker_hooi_convergence() {
        // Use a larger tensor where compression actually happens
        let shape = vec![6, 6, 6];
        let data: Vec<f64> = (0..216)
            .map(|i| ((i as f64) * 0.13 + 0.7).sin() * ((i as f64) * 0.07).cos())
            .collect();
        let t = Tensor::new(shape, data).expect("tensor");

        let config = TuckerConfig {
            ranks: vec![2, 2, 2],
            max_iterations: 30,
            tolerance: 1e-8,
        };

        let hosvd_decomp = tucker_hosvd(&t, &config).expect("hosvd");
        let hooi_decomp = tucker_hooi(&t, &config).expect("hooi");

        let hosvd_error = hosvd_decomp.fit_error(&t).expect("fit");
        let hooi_error = hooi_decomp.fit_error(&t).expect("fit");

        // HOOI should be at least as good as HOSVD (or close)
        assert!(
            hooi_error <= hosvd_error + 0.05,
            "HOOI error {} should not be much worse than HOSVD error {}",
            hooi_error,
            hosvd_error
        );
    }

    #[test]
    fn test_tucker_compression_ratio() {
        let t = make_test_tensor_3d();
        let config = TuckerConfig {
            ranks: vec![2, 2, 2],
            max_iterations: 50,
            tolerance: 1e-8,
        };
        let decomp = tucker_hosvd(&t, &config).expect("tucker");
        let ratio = decomp.compression_ratio(t.shape());

        // Core: 2*2*2 = 8, Factors: 3*2 + 4*2 + 2*2 = 18, Total = 26
        // Original: 24. Ratio = 24/26 ~ 0.92
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_tt_svd_3d_tensor() {
        let t = make_test_tensor_3d();
        let config = TtConfig {
            max_rank: 10,
            tolerance: 1e-10,
        };
        let decomp = tt_svd(&t, &config).expect("tt_svd");

        assert_eq!(decomp.cores.len(), 3);

        // First core: [1, 3, r1]
        assert_eq!(decomp.cores[0].shape()[0], 1);
        assert_eq!(decomp.cores[0].shape()[1], 3);

        // Last core: [r2, 2, 1]
        assert_eq!(decomp.cores[2].shape()[1], 2);
        assert_eq!(decomp.cores[2].shape()[2], 1);

        // Ranks start with 1 and end with 1
        let ranks = decomp.ranks();
        assert_eq!(ranks[0], 1);
        assert_eq!(*ranks.last().expect("no ranks"), 1);

        // Reconstruction
        let recon = decomp.reconstruct().expect("reconstruct");
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_tt_svd_4d_tensor() {
        let shape = vec![2, 3, 2, 4];
        let data: Vec<f64> = (0..48).map(|i| (i as f64 + 1.0) * 0.1).collect();
        let t = Tensor::new(shape, data).expect("tensor");

        let config = TtConfig {
            max_rank: 10,
            tolerance: 1e-10,
        };
        let decomp = tt_svd(&t, &config).expect("tt_svd");
        assert_eq!(decomp.cores.len(), 4);

        let ranks = decomp.ranks();
        assert_eq!(ranks[0], 1);
        assert_eq!(ranks[4], 1);
    }

    #[test]
    fn test_tt_reconstruction_error() {
        let t = make_rank1_tensor();
        let config = TtConfig {
            max_rank: 5,
            tolerance: 1e-10,
        };
        let decomp = tt_svd(&t, &config).expect("tt_svd");
        let error = decomp.fit_error(&t).expect("fit_error");

        // Rank-1 tensor should have small reconstruction error
        assert!(error < 0.1, "TT reconstruction error too large: {}", error);
    }

    #[test]
    fn test_tt_rank_truncation() {
        let t = make_test_tensor_3d();
        let config = TtConfig {
            max_rank: 10,
            tolerance: 1e-10,
        };
        let decomp = tt_svd(&t, &config).expect("tt_svd");

        // Truncate to max_rank=1
        let truncated = decomp.tt_round(1).expect("tt_round");
        let trunc_ranks = truncated.ranks();
        for &r in &trunc_ranks {
            assert!(r <= 1, "rank {} exceeds max_rank 1", r);
        }
    }

    #[test]
    fn test_tt_compression_ratio() {
        let t = make_test_tensor_3d();
        let config = TtConfig {
            max_rank: 2,
            tolerance: 1e-10,
        };
        let decomp = tt_svd(&t, &config).expect("tt_svd");
        let ratio = decomp.compression_ratio(t.shape());
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_1d_tensor_vector() {
        let t = Tensor::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("1d tensor");
        assert_eq!(t.ndim(), 1);
        assert_eq!(t.numel(), 5);

        let v = t.get(&[3]).expect("get");
        assert!((v - 4.0).abs() < 1e-12);

        // TT of 1D tensor
        let config = TtConfig {
            max_rank: 5,
            tolerance: 1e-10,
        };
        let decomp = tt_svd(&t, &config).expect("tt_svd 1d");
        assert_eq!(decomp.cores.len(), 1);
        assert_eq!(decomp.cores[0].shape(), &[1, 5, 1]);
    }

    #[test]
    fn test_config_validation() {
        let t = make_test_tensor_3d();

        // CP with rank 0
        let config = CpAlsConfig {
            rank: 0,
            ..Default::default()
        };
        assert!(cp_als(&t, &config).is_err());

        // Tucker with wrong number of ranks
        let config = TuckerConfig {
            ranks: vec![2, 2], // 3D tensor needs 3 ranks
            ..Default::default()
        };
        assert!(tucker_hosvd(&t, &config).is_err());

        // TT with max_rank 0
        let config = TtConfig {
            max_rank: 0,
            tolerance: 1e-8,
        };
        assert!(tt_svd(&t, &config).is_err());
    }
}
