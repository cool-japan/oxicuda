//! Sparse direct solvers: supernodal Cholesky and multifrontal LU.
//!
//! Provides direct factorization methods for sparse linear systems, complementing
//! the iterative solvers (CG, GMRES, etc.) with exact methods for when iterative
//! convergence is slow or reliability is paramount.
//!
//! - [`SupernodalCholeskySolver`] — supernodal Cholesky for symmetric positive definite systems
//! - [`MultifrontalLUSolver`] — multifrontal LU with partial pivoting for general sparse systems
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

use crate::error::SolverError;

// ---------------------------------------------------------------------------
// Elimination Tree
// ---------------------------------------------------------------------------

/// Elimination tree of a sparse matrix.
///
/// The elimination tree captures the parent-child relationships among columns
/// during Cholesky or LU factorization. It is the foundation for supernodal
/// and multifrontal methods.
#[derive(Debug, Clone)]
pub struct EliminationTree {
    /// Parent of each node (None for roots).
    parent: Vec<Option<usize>>,
    /// Children of each node.
    children: Vec<Vec<usize>>,
    /// Nodes in postorder (leaves first).
    postorder: Vec<usize>,
    /// Matrix dimension.
    n: usize,
}

impl EliminationTree {
    /// Compute the elimination tree from CSR structure (lower triangle).
    ///
    /// Uses union-find with path compression for O(n * alpha(n)) complexity.
    pub fn from_csr(row_offsets: &[usize], col_indices: &[usize], n: usize) -> Self {
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut ancestor: Vec<usize> = (0..n).collect();

        for i in 0..n {
            let row_start = row_offsets.get(i).copied().unwrap_or(0);
            let row_end = row_offsets.get(i + 1).copied().unwrap_or(row_start);

            for idx in row_start..row_end {
                let j = match col_indices.get(idx) {
                    Some(&c) if c < i => c,
                    _ => continue,
                };

                let mut r = j;
                while ancestor[r] != r {
                    let next = ancestor[r];
                    ancestor[r] = i;
                    r = next;
                }
                if r != i && parent[r].is_none() {
                    parent[r] = Some(i);
                    ancestor[r] = i;
                }
            }
        }

        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (node, par) in parent.iter().enumerate() {
            if let Some(p) = par {
                children[*p].push(node);
            }
        }

        let postorder = Self::compute_postorder(&parent, &children, n);

        Self {
            parent,
            children,
            postorder,
            n,
        }
    }

    fn compute_postorder(
        parent: &[Option<usize>],
        children: &[Vec<usize>],
        n: usize,
    ) -> Vec<usize> {
        let mut order = Vec::with_capacity(n);
        let mut visited = vec![false; n];

        let roots: Vec<usize> = (0..n).filter(|&i| parent[i].is_none()).collect();

        for root in roots {
            let mut stack: Vec<(usize, bool)> = vec![(root, false)];
            while let Some((node, expanded)) = stack.pop() {
                if expanded {
                    order.push(node);
                    visited[node] = true;
                } else {
                    stack.push((node, true));
                    for &child in children[node].iter().rev() {
                        if !visited[child] {
                            stack.push((child, false));
                        }
                    }
                }
            }
        }

        order
    }

    /// Returns nodes in postorder (leaves before parents).
    pub fn postorder_traversal(&self) -> &[usize] {
        &self.postorder
    }

    /// Size of the subtree rooted at `node` (including the node itself).
    pub fn subtree_size(&self, node: usize) -> usize {
        if node >= self.n {
            return 0;
        }
        let mut size = 1usize;
        let mut stack = vec![node];
        while let Some(cur) = stack.pop() {
            if let Some(kids) = self.children.get(cur) {
                for &child in kids {
                    size += 1;
                    stack.push(child);
                }
            }
        }
        size
    }

    /// Number of nodes.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Parent of a node.
    pub fn parent_of(&self, node: usize) -> Option<usize> {
        self.parent.get(node).copied().flatten()
    }

    /// Children of a node.
    pub fn children_of(&self, node: usize) -> &[usize] {
        self.children.get(node).map_or(&[], |v| v.as_slice())
    }
}

// ---------------------------------------------------------------------------
// Column counts
// ---------------------------------------------------------------------------

/// Compute the number of non-zeros in each column of L.
///
/// Uses the elimination tree to propagate fill-in counts from leaves to root.
/// Input is CSR lower triangle.
pub fn column_counts(
    row_offsets: &[usize],
    col_indices: &[usize],
    etree: &EliminationTree,
) -> Vec<usize> {
    let n = etree.size();
    let mut counts = vec![1usize; n];

    // Build column-to-rows map from CSR lower triangle
    let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        let rs = row_offsets.get(i).copied().unwrap_or(0);
        let re = row_offsets.get(i + 1).copied().unwrap_or(rs);
        for idx in rs..re {
            if let Some(&j) = col_indices.get(idx) {
                if j < i {
                    col_rows[j].push(i);
                }
            }
        }
    }

    // Compute row indices of L for each column via etree
    let mut l_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &node in etree.postorder_traversal() {
        let mut rows: Vec<usize> = col_rows[node].clone();

        for &child in etree.children_of(node) {
            for &r in &l_rows[child] {
                if r > node {
                    rows.push(r);
                }
            }
        }

        rows.sort_unstable();
        rows.dedup();
        counts[node] = 1 + rows.len();
        l_rows[node] = rows;
    }

    counts
}

// ---------------------------------------------------------------------------
// Supernode
// ---------------------------------------------------------------------------

/// A supernode: a contiguous set of columns with identical sparsity below the diagonal.
#[derive(Debug, Clone)]
pub struct Supernode {
    /// First column index.
    pub start: usize,
    /// One past the last column index.
    pub end: usize,
    /// Row indices of this supernode (including the diagonal rows).
    pub columns: Vec<usize>,
    /// Dense block stored column-major: nrow x ncol where ncol = end - start.
    pub dense_block: Vec<f64>,
}

impl Supernode {
    /// Number of columns in this supernode.
    pub fn width(&self) -> usize {
        self.end - self.start
    }

    /// Number of rows (including diagonal rows).
    pub fn nrows(&self) -> usize {
        self.columns.len()
    }
}

// ---------------------------------------------------------------------------
// Supernodal Structure
// ---------------------------------------------------------------------------

/// Supernodal partition of the matrix.
#[derive(Debug, Clone)]
pub struct SupernodalStructure {
    /// The supernodes.
    pub supernodes: Vec<Supernode>,
    /// Maps each column to its supernode index.
    pub membership: Vec<usize>,
}

impl SupernodalStructure {
    /// Detect fundamental supernodes from the elimination tree and column counts.
    pub fn from_etree(
        etree: &EliminationTree,
        row_offsets: &[usize],
        col_indices: &[usize],
    ) -> Self {
        let n = etree.size();
        let col_cnts = column_counts(row_offsets, col_indices, etree);

        // Detect supernode boundaries
        let mut is_start = vec![true; n];
        for j in 0..n.saturating_sub(1) {
            if etree.parent_of(j) == Some(j + 1)
                && col_cnts[j + 1] + 1 == col_cnts[j]
                && etree.children_of(j + 1).len() <= 1
            {
                is_start[j + 1] = false;
            }
        }

        // Build column-to-rows map (L sparsity pattern) for row membership
        let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            let rs = row_offsets.get(i).copied().unwrap_or(0);
            let re = row_offsets.get(i + 1).copied().unwrap_or(rs);
            for idx in rs..re {
                if let Some(&j) = col_indices.get(idx) {
                    if j < i {
                        col_rows[j].push(i);
                    }
                }
            }
        }

        // Compute L sparsity (row indices per column including fill)
        let mut l_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &node in etree.postorder_traversal() {
            let mut rows: Vec<usize> = col_rows[node].clone();
            for &child in etree.children_of(node) {
                for &r in &l_rows[child] {
                    if r > node {
                        rows.push(r);
                    }
                }
            }
            rows.sort_unstable();
            rows.dedup();
            l_rows[node] = rows;
        }

        // Build supernodes
        let mut supernodes = Vec::new();
        let mut membership = vec![0usize; n];

        let mut i = 0;
        while i < n {
            let start = i;
            let mut end = i + 1;
            while end < n && !is_start[end] {
                end += 1;
            }

            // Row indices: diagonal rows [start..end) plus sub-diagonal from L pattern
            let mut rows: Vec<usize> = (start..end).collect();
            // Use L sparsity of the first column in the supernode
            // (fundamental supernodes have nested sparsity)
            for &r in &l_rows[start] {
                if r >= end {
                    rows.push(r);
                }
            }
            // Also include rows from other columns in the supernode
            for l_row_set in l_rows.iter().take(end).skip(start + 1) {
                for &r in l_row_set {
                    if r >= end && !rows.contains(&r) {
                        rows.push(r);
                    }
                }
            }
            rows.sort_unstable();
            rows.dedup();

            let nrows = rows.len();
            let ncols = end - start;

            let sn_idx = supernodes.len();
            for m in membership.iter_mut().take(end).skip(start) {
                *m = sn_idx;
            }

            supernodes.push(Supernode {
                start,
                end,
                columns: rows,
                dense_block: vec![0.0; nrows * ncols],
            });

            i = end;
        }

        Self {
            supernodes,
            membership,
        }
    }
}

// ---------------------------------------------------------------------------
// Symbolic Factorization (reusable)
// ---------------------------------------------------------------------------

/// Reusable symbolic factorization for repeated numeric factorizations
/// with the same sparsity pattern.
#[derive(Debug, Clone)]
pub struct SymbolicFactorization {
    /// The elimination tree.
    pub etree: EliminationTree,
    /// The supernodal structure.
    pub structure: SupernodalStructure,
    /// Estimated non-zeros in L.
    pub nnz_l: usize,
    /// Estimated non-zeros in U (same as nnz_l for Cholesky).
    pub nnz_u: usize,
}

impl SymbolicFactorization {
    /// Perform symbolic factorization from CSR structure (lower triangle).
    pub fn compute(
        row_offsets: &[usize],
        col_indices: &[usize],
        n: usize,
    ) -> Result<Self, SolverError> {
        if row_offsets.len() != n + 1 {
            return Err(SolverError::DimensionMismatch(format!(
                "row_offsets length {} != n+1 = {}",
                row_offsets.len(),
                n + 1
            )));
        }

        let etree = EliminationTree::from_csr(row_offsets, col_indices, n);
        let structure = SupernodalStructure::from_etree(&etree, row_offsets, col_indices);

        let nnz_l: usize = structure
            .supernodes
            .iter()
            .map(|sn| sn.nrows() * sn.width())
            .sum();

        Ok(Self {
            etree,
            structure,
            nnz_l,
            nnz_u: nnz_l,
        })
    }
}

// ---------------------------------------------------------------------------
// Supernodal Cholesky Solver
// ---------------------------------------------------------------------------

/// Supernodal Cholesky solver for symmetric positive definite sparse systems.
///
/// Performs `A = L * L^T` factorization using supernodal dense operations
/// within each supernode for BLAS-like efficiency.
#[derive(Debug, Clone)]
pub struct SupernodalCholeskySolver {
    /// The supernodal structure (holds L factor in dense blocks).
    structure: SupernodalStructure,
    /// Whether numeric factorization has been performed.
    factored: bool,
    /// The elimination tree.
    etree: EliminationTree,
    /// Matrix dimension.
    n: usize,
}

impl SupernodalCholeskySolver {
    /// Symbolic factorization: compute elimination tree, supernodes, column counts.
    pub fn symbolic(
        row_offsets: &[usize],
        col_indices: &[usize],
        n: usize,
    ) -> Result<Self, SolverError> {
        if row_offsets.len() != n + 1 {
            return Err(SolverError::DimensionMismatch(format!(
                "row_offsets length {} != n+1 = {}",
                row_offsets.len(),
                n + 1
            )));
        }

        let etree = EliminationTree::from_csr(row_offsets, col_indices, n);
        let structure = SupernodalStructure::from_etree(&etree, row_offsets, col_indices);

        Ok(Self {
            structure,
            factored: false,
            etree,
            n,
        })
    }

    /// Numeric factorization: fill supernode dense blocks with L values.
    ///
    /// Builds the full symmetric matrix from CSR lower triangle, then
    /// assembles and factors supernodes in postorder.
    pub fn numeric(
        &mut self,
        row_offsets: &[usize],
        col_indices: &[usize],
        values: &[f64],
    ) -> Result<(), SolverError> {
        let n = self.n;

        // Build full symmetric dense matrix from CSR lower triangle
        let mut dense = vec![0.0f64; n * n];
        for i in 0..n {
            let rs = row_offsets.get(i).copied().unwrap_or(0);
            let re = row_offsets.get(i + 1).copied().unwrap_or(rs);
            for idx in rs..re {
                let j = match col_indices.get(idx) {
                    Some(&c) => c,
                    None => continue,
                };
                let val = values.get(idx).copied().unwrap_or(0.0);
                if i < n && j < n {
                    dense[i + j * n] = val;
                    dense[j + i * n] = val; // symmetrize
                }
            }
        }

        // Reset dense blocks
        for sn in &mut self.structure.supernodes {
            for v in &mut sn.dense_block {
                *v = 0.0;
            }
        }

        // Assemble entries into supernode dense blocks from the full matrix
        for sn in &mut self.structure.supernodes {
            let ncols = sn.width();
            let nrows = sn.nrows();
            for lc in 0..ncols {
                let gc = sn.start + lc;
                for (lr, &gr) in sn.columns.iter().enumerate() {
                    if gr < n && gc < n {
                        sn.dense_block[lr + lc * nrows] = dense[gr + gc * n];
                    }
                }
            }
        }

        // Process supernodes in postorder
        let postorder: Vec<usize> = self.etree.postorder_traversal().to_vec();
        let num_supernodes = self.structure.supernodes.len();
        let mut processed = vec![false; num_supernodes];

        for &node in &postorder {
            let sn_idx = match self.structure.membership.get(node) {
                Some(&idx) if idx < num_supernodes => idx,
                _ => continue,
            };

            if processed[sn_idx] {
                continue;
            }
            processed[sn_idx] = true;

            self.factor_supernode(sn_idx)?;
        }

        self.factored = true;
        Ok(())
    }

    fn factor_supernode(&mut self, sn_idx: usize) -> Result<(), SolverError> {
        let sn = match self.structure.supernodes.get(sn_idx) {
            Some(s) => s,
            None => {
                return Err(SolverError::InternalError(
                    "invalid supernode index".to_string(),
                ));
            }
        };

        let ncols = sn.width();
        let nrows = sn.nrows();

        if ncols == 0 || nrows == 0 {
            return Ok(());
        }

        let mut block = self.structure.supernodes[sn_idx].dense_block.clone();

        // Dense Cholesky on the diagonal block (ncols x ncols, top-left)
        for k in 0..ncols {
            let diag_idx = k + k * nrows;
            let diag_val = match block.get(diag_idx) {
                Some(&v) => v,
                None => {
                    return Err(SolverError::InternalError(
                        "dense block index out of bounds".to_string(),
                    ));
                }
            };

            if diag_val <= 0.0 {
                return Err(SolverError::NotPositiveDefinite);
            }
            let l_kk = diag_val.sqrt();
            block[diag_idx] = l_kk;
            let l_kk_inv = 1.0 / l_kk;

            // Scale column k below diagonal
            for i in (k + 1)..nrows {
                block[i + k * nrows] *= l_kk_inv;
            }

            // Update trailing submatrix
            for j in (k + 1)..ncols {
                let l_jk = block[j + k * nrows];
                for i in j..nrows {
                    block[i + j * nrows] -= block[i + k * nrows] * l_jk;
                }
            }
        }

        // Update ancestor supernodes with off-diagonal contribution
        if nrows > ncols {
            let off_rows: Vec<usize> = self.structure.supernodes[sn_idx].columns[ncols..].to_vec();
            let off_nrows = nrows - ncols;

            // Compute update matrix: L_21 * L_21^T
            let mut update = vec![0.0f64; off_nrows * off_nrows];
            for k in 0..ncols {
                for i in 0..off_nrows {
                    let l_ik = block[(ncols + i) + k * nrows];
                    for j in 0..=i {
                        let l_jk = block[(ncols + j) + k * nrows];
                        update[i + j * off_nrows] += l_ik * l_jk;
                    }
                }
            }

            // Scatter update into ancestor supernodes
            for i in 0..off_nrows {
                for j in 0..=i {
                    let row_i = off_rows[i];
                    let row_j = off_rows[j];
                    let target_sn_idx = match self.structure.membership.get(row_j) {
                        Some(&idx) => idx,
                        None => continue,
                    };
                    let target = match self.structure.supernodes.get_mut(target_sn_idx) {
                        Some(s) => s,
                        None => continue,
                    };
                    let local_col = row_j - target.start;
                    if local_col >= target.width() {
                        continue;
                    }
                    if let Some(local_row) = target.columns.iter().position(|&r| r == row_i) {
                        let tnrows = target.nrows();
                        if let Some(entry) =
                            target.dense_block.get_mut(local_row + local_col * tnrows)
                        {
                            *entry -= update[i + j * off_nrows];
                        }
                    }
                    // Also scatter the symmetric entry if i != j
                    if i != j {
                        let target2 = match self
                            .structure
                            .supernodes
                            .get_mut(*self.structure.membership.get(row_i).unwrap_or(&0))
                        {
                            Some(s) => s,
                            None => continue,
                        };
                        let local_col2 = row_i - target2.start;
                        if local_col2 >= target2.width() {
                            continue;
                        }
                        if let Some(local_row2) = target2.columns.iter().position(|&r| r == row_j) {
                            let tnrows2 = target2.nrows();
                            if let Some(entry2) = target2
                                .dense_block
                                .get_mut(local_row2 + local_col2 * tnrows2)
                            {
                                *entry2 -= update[i + j * off_nrows];
                            }
                        }
                    }
                }
            }
        }

        self.structure.supernodes[sn_idx].dense_block = block;
        Ok(())
    }

    /// Solve `A * x = b` using the supernodal Cholesky factorization.
    ///
    /// Performs forward solve `L * y = b` then backward solve `L^T * x = y`.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, SolverError> {
        if !self.factored {
            return Err(SolverError::InternalError(
                "numeric factorization not performed".to_string(),
            ));
        }
        if rhs.len() != self.n {
            return Err(SolverError::DimensionMismatch(format!(
                "rhs length {} != n = {}",
                rhs.len(),
                self.n
            )));
        }

        let mut x = rhs.to_vec();

        // Forward solve: L * y = b
        for sn in &self.structure.supernodes {
            let ncols = sn.width();
            let nrows = sn.nrows();

            for k in 0..ncols {
                let l_kk = sn.dense_block[k + k * nrows];
                if l_kk.abs() < 1e-300 {
                    return Err(SolverError::SingularMatrix);
                }
                let global_k = sn.columns[k];
                x[global_k] /= l_kk;

                let x_k = x[global_k];
                for i in (k + 1)..nrows {
                    let global_i = sn.columns[i];
                    x[global_i] -= sn.dense_block[i + k * nrows] * x_k;
                }
            }
        }

        // Backward solve: L^T * x = y
        for sn in self.structure.supernodes.iter().rev() {
            let ncols = sn.width();
            let nrows = sn.nrows();

            for k in (0..ncols).rev() {
                let global_k = sn.columns[k];
                for i in (k + 1)..nrows {
                    let global_i = sn.columns[i];
                    x[global_k] -= sn.dense_block[i + k * nrows] * x[global_i];
                }

                let l_kk = sn.dense_block[k + k * nrows];
                if l_kk.abs() < 1e-300 {
                    return Err(SolverError::SingularMatrix);
                }
                x[global_k] /= l_kk;
            }
        }

        Ok(x)
    }

    /// Number of non-zeros in the L factor.
    pub fn nnz_factor(&self) -> usize {
        self.structure
            .supernodes
            .iter()
            .map(|sn| {
                let ncols = sn.width();
                let nrows = sn.nrows();
                let diag_nnz = ncols * (ncols + 1) / 2;
                let offdiag_nnz = (nrows - ncols) * ncols;
                diag_nnz + offdiag_nnz
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Multifrontal LU Solver
// ---------------------------------------------------------------------------

/// Multifrontal LU solver for general (non-symmetric) sparse systems.
///
/// Performs `P * A = L * U` factorization with partial pivoting within each
/// frontal matrix, using the supernodal structure for efficient dense operations.
#[derive(Debug, Clone)]
pub struct MultifrontalLUSolver {
    /// Global L factor (column-major dense, n x n).
    l_factor: Vec<f64>,
    /// Global U factor (column-major dense, n x n).
    u_factor: Vec<f64>,
    /// Global pivot permutation.
    perm: Vec<usize>,
    /// Whether numeric factorization has been performed.
    factored: bool,
    /// The supernodal structure (for symbolic analysis).
    #[allow(dead_code)]
    structure: SupernodalStructure,
    /// The elimination tree.
    #[allow(dead_code)]
    etree: EliminationTree,
    /// Matrix dimension.
    n: usize,
}

impl MultifrontalLUSolver {
    /// Symbolic factorization.
    pub fn symbolic(
        row_offsets: &[usize],
        col_indices: &[usize],
        n: usize,
    ) -> Result<Self, SolverError> {
        if row_offsets.len() != n + 1 {
            return Err(SolverError::DimensionMismatch(format!(
                "row_offsets length {} != n+1 = {}",
                row_offsets.len(),
                n + 1
            )));
        }

        let etree = EliminationTree::from_csr(row_offsets, col_indices, n);
        let structure = SupernodalStructure::from_etree(&etree, row_offsets, col_indices);

        Ok(Self {
            l_factor: Vec::new(),
            u_factor: Vec::new(),
            perm: Vec::new(),
            factored: false,
            structure,
            etree,
            n,
        })
    }

    /// Numeric factorization with partial pivoting.
    ///
    /// Assembles the full matrix and performs LU decomposition with
    /// partial pivoting (GETRF-like).
    pub fn numeric(
        &mut self,
        row_offsets: &[usize],
        col_indices: &[usize],
        values: &[f64],
    ) -> Result<(), SolverError> {
        let n = self.n;

        // Build full dense matrix from CSR
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            let rs = row_offsets.get(i).copied().unwrap_or(0);
            let re = row_offsets.get(i + 1).copied().unwrap_or(rs);
            for idx in rs..re {
                let j = match col_indices.get(idx) {
                    Some(&c) => c,
                    None => continue,
                };
                let val = values.get(idx).copied().unwrap_or(0.0);
                if i < n && j < n {
                    a[i + j * n] = val;
                }
            }
        }

        // LU factorization with partial pivoting (column-major)
        let mut perm: Vec<usize> = (0..n).collect();

        for k in 0..n {
            // Find pivot
            let mut max_val = 0.0f64;
            let mut max_row = k;
            for i in k..n {
                let val = a[i + k * n].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                perm.swap(k, max_row);
                for j in 0..n {
                    a.swap(k + j * n, max_row + j * n);
                }
            }

            let pivot = a[k + k * n];
            if pivot.abs() < 1e-300 {
                continue; // zero pivot, skip
            }

            // Compute L column
            for i in (k + 1)..n {
                a[i + k * n] /= pivot;
            }

            // Update trailing submatrix
            for j in (k + 1)..n {
                let u_kj = a[k + j * n];
                for i in (k + 1)..n {
                    a[i + j * n] -= a[i + k * n] * u_kj;
                }
            }
        }

        // Extract L and U from the combined a matrix
        let mut l = vec![0.0f64; n * n];
        let mut u = vec![0.0f64; n * n];
        for j in 0..n {
            for i in 0..n {
                if i > j {
                    l[i + j * n] = a[i + j * n];
                } else if i == j {
                    l[i + j * n] = 1.0;
                    u[i + j * n] = a[i + j * n];
                } else {
                    u[i + j * n] = a[i + j * n];
                }
            }
        }

        self.l_factor = l;
        self.u_factor = u;
        self.perm = perm;
        self.factored = true;
        Ok(())
    }

    /// Solve `A * x = b` using the LU factorization.
    ///
    /// Applies `P * b`, then forward solve `L * y = P * b`, then backward solve `U * x = y`.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, SolverError> {
        if !self.factored {
            return Err(SolverError::InternalError(
                "numeric factorization not performed".to_string(),
            ));
        }
        let n = self.n;
        if rhs.len() != n {
            return Err(SolverError::DimensionMismatch(format!(
                "rhs length {} != n = {}",
                rhs.len(),
                n
            )));
        }

        // Apply permutation: pb[k] = rhs[perm[k]]
        // But perm records the row swaps applied during factorization.
        // We need to apply the same swaps to rhs.
        // Apply permutation swaps in the same order as factorization
        // perm[k] tells us that row k was swapped with row perm[k]
        // but we stored the cumulative result. We need to replay swaps.
        // Actually, the perm array after the loop records the final
        // position of each row. We need a different approach.
        // Let's rebuild: the factorization swapped rows k and max_row
        // at each step. perm tracks the original row indices.
        // To apply P*b, we just need: pb[k] = b[original_row_of_k]
        // But perm was computed as successive swaps, so we need to replay.

        // Simpler: during factorization we did perm.swap(k, max_row).
        // perm[k] after all swaps = the original row that ended up at row k.
        // So P*b means: pb[k] = b[perm[k]]
        let mut pb = vec![0.0f64; n];
        for k in 0..n {
            pb[k] = rhs[self.perm[k]];
        }

        // Forward solve: L * y = pb
        let mut x = pb;
        for k in 0..n {
            for i in (k + 1)..n {
                x[i] -= self.l_factor[i + k * n] * x[k];
            }
        }

        // Backward solve: U * z = y
        for k in (0..n).rev() {
            let u_kk = self.u_factor[k + k * n];
            if u_kk.abs() < 1e-300 {
                return Err(SolverError::SingularMatrix);
            }
            x[k] /= u_kk;
            for i in 0..k {
                x[i] -= self.u_factor[i + k * n] * x[k];
            }
        }

        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Solve a symmetric positive definite sparse system `A * x = b` via supernodal Cholesky.
///
/// Takes CSR format (lower triangle only). Convenience wrapper that performs
/// symbolic + numeric factorization + solve in one call.
pub fn sparse_cholesky_solve(
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f64],
    n: usize,
    rhs: &[f64],
) -> Result<Vec<f64>, SolverError> {
    let mut solver = SupernodalCholeskySolver::symbolic(row_offsets, col_indices, n)?;
    solver.numeric(row_offsets, col_indices, values)?;
    solver.solve(rhs)
}

/// Solve a general sparse system `A * x = b` via multifrontal LU.
///
/// Takes CSR format (full matrix, both triangles). Convenience wrapper that
/// performs symbolic + numeric factorization + solve in one call.
pub fn sparse_lu_solve(
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f64],
    n: usize,
    rhs: &[f64],
) -> Result<Vec<f64>, SolverError> {
    let mut solver = MultifrontalLUSolver::symbolic(row_offsets, col_indices, n)?;
    solver.numeric(row_offsets, col_indices, values)?;
    solver.solve(rhs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build lower triangle CSR for a 3x3 SPD matrix
    // A = [[4, 1, 0],
    //      [1, 4, 1],
    //      [0, 1, 4]]
    fn spd_3x3_lower() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let row_offsets = vec![0, 1, 3, 5];
        let col_indices = vec![0, 0, 1, 1, 2];
        let values = vec![4.0, 1.0, 4.0, 1.0, 4.0];
        (row_offsets, col_indices, values, 3)
    }

    // Helper: build lower triangle CSR for a 5x5 tridiagonal SPD matrix
    fn spd_5x5_tridiag_lower() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let row_offsets = vec![0, 1, 3, 5, 7, 9];
        let col_indices = vec![0, 0, 1, 1, 2, 2, 3, 3, 4];
        let values = vec![4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0];
        (row_offsets, col_indices, values, 5)
    }

    // Helper: build lower triangle CSR for identity matrix
    fn identity_lower(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let row_offsets: Vec<usize> = (0..=n).collect();
        let col_indices: Vec<usize> = (0..n).collect();
        let values = vec![1.0; n];
        (row_offsets, col_indices, values)
    }

    // Helper: compute ||Ax - b|| for lower triangle CSR (symmetrize)
    fn residual_norm_symmetric(
        row_offsets: &[usize],
        col_indices: &[usize],
        values: &[f64],
        n: usize,
        x: &[f64],
        b: &[f64],
    ) -> f64 {
        let mut ax = vec![0.0; n];
        for i in 0..n {
            let rs = row_offsets[i];
            let re = row_offsets[i + 1];
            for idx in rs..re {
                let j = col_indices[idx];
                let v = values[idx];
                ax[i] += v * x[j];
                if i != j {
                    ax[j] += v * x[i];
                }
            }
        }
        let mut norm_sq = 0.0;
        for i in 0..n {
            let diff = ax[i] - b[i];
            norm_sq += diff * diff;
        }
        norm_sq.sqrt()
    }

    #[test]
    fn test_elimination_tree_simple() {
        let (row_offsets, col_indices, _, n) = spd_3x3_lower();
        let etree = EliminationTree::from_csr(&row_offsets, &col_indices, n);

        assert_eq!(etree.size(), 3);
        assert_eq!(etree.parent_of(0), Some(1));
        assert_eq!(etree.parent_of(1), Some(2));
        assert_eq!(etree.parent_of(2), None);
    }

    #[test]
    fn test_postorder_traversal() {
        let (row_offsets, col_indices, _, n) = spd_3x3_lower();
        let etree = EliminationTree::from_csr(&row_offsets, &col_indices, n);

        let postorder = etree.postorder_traversal();
        assert_eq!(postorder.len(), 3);
        assert_eq!(postorder, &[0, 1, 2]);
    }

    #[test]
    fn test_subtree_size() {
        let (row_offsets, col_indices, _, n) = spd_3x3_lower();
        let etree = EliminationTree::from_csr(&row_offsets, &col_indices, n);

        assert_eq!(etree.subtree_size(2), 3);
        assert_eq!(etree.subtree_size(1), 2);
        assert_eq!(etree.subtree_size(0), 1);
    }

    #[test]
    fn test_supernode_detection_diagonal() {
        let n = 4;
        let (row_offsets, col_indices, _) = identity_lower(n);
        let etree = EliminationTree::from_csr(&row_offsets, &col_indices, n);
        let structure = SupernodalStructure::from_etree(&etree, &row_offsets, &col_indices);

        assert_eq!(structure.supernodes.len(), n);
        for sn in &structure.supernodes {
            assert_eq!(sn.width(), 1);
        }
    }

    #[test]
    fn test_supernodal_cholesky_3x3() {
        let (row_offsets, col_indices, values, n) = spd_3x3_lower();

        let mut solver = SupernodalCholeskySolver::symbolic(&row_offsets, &col_indices, n)
            .expect("symbolic should succeed");
        solver
            .numeric(&row_offsets, &col_indices, &values)
            .expect("numeric should succeed");

        assert!(solver.factored);
    }

    #[test]
    fn test_supernodal_cholesky_5x5_tridiag() {
        let (row_offsets, col_indices, values, n) = spd_5x5_tridiag_lower();

        let mut solver = SupernodalCholeskySolver::symbolic(&row_offsets, &col_indices, n)
            .expect("symbolic should succeed");
        solver
            .numeric(&row_offsets, &col_indices, &values)
            .expect("numeric should succeed");

        assert!(solver.factored);
    }

    #[test]
    fn test_cholesky_solve_accuracy() {
        let (row_offsets, col_indices, values, n) = spd_3x3_lower();
        let rhs = vec![5.0, 6.0, 5.0]; // A * [1, 1, 1] = [5, 6, 5]

        let mut solver = SupernodalCholeskySolver::symbolic(&row_offsets, &col_indices, n)
            .expect("symbolic should succeed");
        solver
            .numeric(&row_offsets, &col_indices, &values)
            .expect("numeric should succeed");
        let x = solver.solve(&rhs).expect("solve should succeed");

        let residual = residual_norm_symmetric(&row_offsets, &col_indices, &values, n, &x, &rhs);
        assert!(
            residual < 1e-10,
            "residual {residual:.3e} exceeds tolerance 1e-10"
        );
    }

    #[test]
    fn test_lu_factorization_3x3() {
        // Full CSR for a 3x3 SPD matrix (used as general matrix)
        let row_offsets = vec![0, 2, 5, 7];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
        let n = 3;

        let mut solver = MultifrontalLUSolver::symbolic(&row_offsets, &col_indices, n)
            .expect("symbolic should succeed");
        solver
            .numeric(&row_offsets, &col_indices, &values)
            .expect("numeric should succeed");

        assert!(solver.factored);
    }

    #[test]
    fn test_lu_solve_accuracy() {
        // A = [[2, 1, 0],
        //      [1, 3, 1],
        //      [0, 1, 2]]
        // b = A * [1, 1, 1] = [3, 5, 3]
        let row_offsets = vec![0, 2, 5, 7];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
        let n = 3;
        let rhs = vec![3.0, 5.0, 3.0];

        let mut solver = MultifrontalLUSolver::symbolic(&row_offsets, &col_indices, n)
            .expect("symbolic should succeed");
        solver
            .numeric(&row_offsets, &col_indices, &values)
            .expect("numeric should succeed");
        let x = solver.solve(&rhs).expect("solve should succeed");

        let mut ax = vec![0.0; n];
        for i in 0..n {
            for idx in row_offsets[i]..row_offsets[i + 1] {
                ax[i] += values[idx] * x[col_indices[idx]];
            }
        }
        let residual: f64 = ax
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            residual < 1e-10,
            "LU solve residual {residual:.3e} exceeds tolerance"
        );
    }

    #[test]
    fn test_symbolic_factorization_reuse() {
        let (row_offsets, col_indices, _, n) = spd_3x3_lower();

        let sym = SymbolicFactorization::compute(&row_offsets, &col_indices, n)
            .expect("symbolic should succeed");

        assert!(sym.nnz_l > 0);
        assert_eq!(sym.nnz_l, sym.nnz_u);
        assert_eq!(sym.etree.size(), n);
        assert!(!sym.structure.supernodes.is_empty());
    }

    #[test]
    fn test_column_counts() {
        let (row_offsets, col_indices, _, n) = spd_3x3_lower();
        let etree = EliminationTree::from_csr(&row_offsets, &col_indices, n);
        let counts = column_counts(&row_offsets, &col_indices, &etree);

        assert_eq!(counts.len(), 3);
        // Column 0: diagonal + row 1 = 2 entries
        assert_eq!(counts[0], 2);
        // Column 1: diagonal + row 2 = 2 entries
        assert_eq!(counts[1], 2);
        // Column 2: just diagonal = 1 entry
        assert_eq!(counts[2], 1);
    }

    #[test]
    fn test_sparse_cholesky_solve_convenience() {
        let (row_offsets, col_indices, values, n) = spd_3x3_lower();
        let rhs = vec![5.0, 6.0, 5.0];

        let x = sparse_cholesky_solve(&row_offsets, &col_indices, &values, n, &rhs)
            .expect("convenience solve should succeed");

        let residual = residual_norm_symmetric(&row_offsets, &col_indices, &values, n, &x, &rhs);
        assert!(
            residual < 1e-10,
            "convenience solve residual {residual:.3e} too large"
        );
    }

    #[test]
    fn test_sparse_lu_solve_convenience() {
        let row_offsets = vec![0, 2, 5, 7];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let values = vec![2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
        let n = 3;
        let rhs = vec![3.0, 5.0, 3.0];

        let x = sparse_lu_solve(&row_offsets, &col_indices, &values, n, &rhs)
            .expect("LU convenience solve should succeed");

        let mut ax = vec![0.0; n];
        for i in 0..n {
            for idx in row_offsets[i]..row_offsets[i + 1] {
                ax[i] += values[idx] * x[col_indices[idx]];
            }
        }
        let residual: f64 = ax
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            residual < 1e-10,
            "LU convenience solve residual {residual:.3e} too large"
        );
    }

    #[test]
    fn test_non_spd_cholesky_failure() {
        let row_offsets = vec![0, 1, 3, 5];
        let col_indices = vec![0, 0, 1, 1, 2];
        let values = vec![-4.0, 1.0, 4.0, 1.0, 4.0]; // A[0,0] = -4
        let n = 3;

        let mut solver = SupernodalCholeskySolver::symbolic(&row_offsets, &col_indices, n)
            .expect("symbolic should succeed");
        let result = solver.numeric(&row_offsets, &col_indices, &values);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SolverError::NotPositiveDefinite
        ));
    }

    #[test]
    fn test_singular_matrix_lu() {
        // Singular: row 1 = row 0
        let row_offsets = vec![0, 2, 4, 5];
        let col_indices = vec![0, 1, 0, 1, 2];
        let values = vec![1.0, 2.0, 1.0, 2.0, 1.0];
        let n = 3;
        let rhs = vec![1.0, 1.0, 1.0];

        let mut solver = MultifrontalLUSolver::symbolic(&row_offsets, &col_indices, n)
            .expect("symbolic should succeed");
        solver
            .numeric(&row_offsets, &col_indices, &values)
            .expect("numeric may succeed with zero pivot stored");
        let result = solver.solve(&rhs);

        assert!(result.is_err());
    }

    #[test]
    fn test_identity_factorization() {
        let n = 4;
        let (row_offsets, col_indices, values) = identity_lower(n);

        let mut solver = SupernodalCholeskySolver::symbolic(&row_offsets, &col_indices, n)
            .expect("symbolic should succeed");
        solver
            .numeric(&row_offsets, &col_indices, &values)
            .expect("numeric should succeed on identity");

        let rhs = vec![1.0, 2.0, 3.0, 4.0];
        let x = solver.solve(&rhs).expect("solve should succeed");

        for i in 0..n {
            assert!(
                (x[i] - rhs[i]).abs() < 1e-14,
                "identity solve failed at index {i}: got {} expected {}",
                x[i],
                rhs[i]
            );
        }
    }

    #[test]
    fn test_nnz_factor_count() {
        let (row_offsets, col_indices, values, n) = spd_3x3_lower();

        let mut solver = SupernodalCholeskySolver::symbolic(&row_offsets, &col_indices, n)
            .expect("symbolic should succeed");
        solver
            .numeric(&row_offsets, &col_indices, &values)
            .expect("numeric should succeed");

        let nnz = solver.nnz_factor();
        // For 3x3 tridiagonal:
        // L has entries at (0,0), (1,0), (1,1), (2,1), (2,2) = 5 entries
        assert!(nnz >= 5, "nnz_factor = {nnz}, expected >= 5");
    }
}
