//! Sparse matrix reordering algorithms.
//!
//! Reordering reduces bandwidth (RCM) or fill-in (AMD) of sparse matrices,
//! improving the performance of direct and iterative solvers.
//!
//! ## Algorithms
//!
//! - **Reverse Cuthill-McKee (RCM)**: BFS from a peripheral node, reversed.
//!   Reduces matrix bandwidth, which improves cache locality for SpMV and
//!   reduces fill-in for banded preconditioners.
//!
//! - **Approximate Minimum Degree (AMD)**: Greedy elimination choosing the
//!   node with minimum degree. Reduces fill-in for Cholesky/LU factorization.
//!
//! ## Usage
//!
//! ```rust,no_run
//! # use oxicuda_sparse::format::reorder::{rcm_ordering, permute_csr};
//! # use oxicuda_sparse::format::CsrMatrix;
//! # fn example(matrix: &CsrMatrix<f64>) -> Result<(), oxicuda_sparse::error::SparseError> {
//! let perm = rcm_ordering(matrix)?;
//! let reordered = permute_csr(matrix, &perm)?;
//! # Ok(())
//! # }
//! ```
#![allow(dead_code)]

use std::collections::VecDeque;

use oxicuda_blas::GpuFloat;

use crate::error::{SparseError, SparseResult};
use crate::format::CsrMatrix;

// ---------------------------------------------------------------------------
// Reverse Cuthill-McKee (RCM)
// ---------------------------------------------------------------------------

/// Compute the Reverse Cuthill-McKee (RCM) ordering of a sparse matrix.
///
/// The RCM ordering is a BFS-based reordering that reduces the bandwidth of
/// the matrix. The algorithm:
/// 1. Find a pseudo-peripheral node (node with near-maximal eccentricity).
/// 2. Perform BFS from that node, visiting neighbors in order of increasing
///    degree.
/// 3. Reverse the resulting ordering.
///
/// # Arguments
///
/// * `matrix` -- A square CSR matrix.
///
/// # Returns
///
/// A permutation vector `perm` of length `n` where `perm[new_index] = old_index`.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if the matrix is not square.
pub fn rcm_ordering<T: GpuFloat>(matrix: &CsrMatrix<T>) -> SparseResult<Vec<usize>> {
    if matrix.rows() != matrix.cols() {
        return Err(SparseError::DimensionMismatch(format!(
            "RCM requires square matrix, got {}x{}",
            matrix.rows(),
            matrix.cols()
        )));
    }

    let n = matrix.rows() as usize;
    if n == 0 {
        return Ok(Vec::new());
    }

    let (h_row_ptr, h_col_idx, _) = matrix.to_host()?;
    rcm_ordering_host(&h_row_ptr, &h_col_idx, n)
}

/// Host-side RCM ordering computation.
pub fn rcm_ordering_host(row_ptr: &[i32], col_idx: &[i32], n: usize) -> SparseResult<Vec<usize>> {
    if n == 0 {
        return Ok(Vec::new());
    }

    // Compute degree of each node (excluding self-loops)
    let degrees: Vec<usize> = (0..n)
        .map(|i| {
            let start = row_ptr[i] as usize;
            let end = row_ptr[i + 1] as usize;
            col_idx[start..end]
                .iter()
                .filter(|&&c| c as usize != i && (c as usize) < n)
                .count()
        })
        .collect();

    // Find a pseudo-peripheral starting node
    let start_node = find_pseudo_peripheral(row_ptr, col_idx, &degrees, n);

    // BFS with neighbors sorted by degree
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);

    // Handle potentially disconnected graph
    let starts = [start_node];
    // We will add other starting nodes if components are disconnected

    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut component_start = 0;
    while order.len() < n {
        let root = if component_start < starts.len() {
            starts[component_start]
        } else {
            // Find next unvisited node
            match visited.iter().position(|&v| !v) {
                Some(node) => node,
                None => break,
            }
        };
        component_start += 1;

        if visited[root] {
            continue;
        }

        visited[root] = true;
        queue.push_back(root);

        while let Some(node) = queue.pop_front() {
            order.push(node);

            // Collect unvisited neighbors and sort by degree
            let start = row_ptr[node] as usize;
            let end = row_ptr[node + 1] as usize;
            let mut neighbors: Vec<usize> = col_idx[start..end]
                .iter()
                .map(|&c| c as usize)
                .filter(|&c| c < n && c != node && !visited[c])
                .collect();

            // Deduplicate
            neighbors.sort_unstable();
            neighbors.dedup();

            // Sort by degree (ascending) for RCM
            neighbors.sort_by_key(|&nbr| degrees[nbr]);

            for nbr in neighbors {
                if !visited[nbr] {
                    visited[nbr] = true;
                    queue.push_back(nbr);
                }
            }
        }
    }

    // Reverse the ordering (Reverse Cuthill-McKee)
    order.reverse();

    Ok(order)
}

/// Find a pseudo-peripheral node using the Gibbs-Poole-Stockmeyer approach.
///
/// Starting from the node with minimum degree, performs two BFS passes to find
/// a node with near-maximal eccentricity.
fn find_pseudo_peripheral(row_ptr: &[i32], col_idx: &[i32], degrees: &[usize], n: usize) -> usize {
    // Start from a minimum-degree node
    let mut current = 0;
    let mut min_deg = degrees[0];
    for (i, &d) in degrees.iter().enumerate().skip(1) {
        if d < min_deg {
            min_deg = d;
            current = i;
        }
    }

    // Refine: do a few BFS iterations to move toward a peripheral node
    for _ in 0..5 {
        let (last_level, _) = bfs_levels(row_ptr, col_idx, n, current);
        if last_level.is_empty() {
            break;
        }
        // Pick the node with minimum degree from the last BFS level
        let mut best = last_level[0];
        let mut best_deg = degrees[best];
        for &node in &last_level[1..] {
            if degrees[node] < best_deg {
                best_deg = degrees[node];
                best = node;
            }
        }
        if best == current {
            break;
        }
        current = best;
    }

    current
}

/// Performs BFS from `root` and returns the last level's nodes and the number of levels.
fn bfs_levels(row_ptr: &[i32], col_idx: &[i32], n: usize, root: usize) -> (Vec<usize>, usize) {
    let mut visited = vec![false; n];
    let mut current_level = Vec::new();
    let mut next_level = Vec::new();

    visited[root] = true;
    current_level.push(root);
    let mut num_levels = 1;

    loop {
        for &node in &current_level {
            let start = row_ptr[node] as usize;
            let end = row_ptr[node + 1] as usize;
            for &c in &col_idx[start..end] {
                let nbr = c as usize;
                if nbr < n && !visited[nbr] {
                    visited[nbr] = true;
                    next_level.push(nbr);
                }
            }
        }

        if next_level.is_empty() {
            break;
        }

        num_levels += 1;
        current_level.clear();
        std::mem::swap(&mut current_level, &mut next_level);
    }

    (current_level, num_levels)
}

// ---------------------------------------------------------------------------
// Approximate Minimum Degree (AMD)
// ---------------------------------------------------------------------------

/// Compute the Approximate Minimum Degree (AMD) ordering of a sparse matrix.
///
/// AMD is a greedy fill-reducing ordering for Cholesky and LU factorization.
/// At each step, the node with minimum approximate degree is eliminated.
///
/// # Arguments
///
/// * `matrix` -- A square CSR matrix.
///
/// # Returns
///
/// A permutation vector `perm` of length `n` where `perm[i]` is the `i`-th
/// node to be eliminated.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if the matrix is not square.
pub fn amd_ordering<T: GpuFloat>(matrix: &CsrMatrix<T>) -> SparseResult<Vec<usize>> {
    if matrix.rows() != matrix.cols() {
        return Err(SparseError::DimensionMismatch(format!(
            "AMD requires square matrix, got {}x{}",
            matrix.rows(),
            matrix.cols()
        )));
    }

    let n = matrix.rows() as usize;
    if n == 0 {
        return Ok(Vec::new());
    }

    let (h_row_ptr, h_col_idx, _) = matrix.to_host()?;
    amd_ordering_host(&h_row_ptr, &h_col_idx, n)
}

/// Host-side AMD ordering computation.
pub fn amd_ordering_host(row_ptr: &[i32], col_idx: &[i32], n: usize) -> SparseResult<Vec<usize>> {
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build adjacency lists (excluding self-loops, symmetric)
    let mut adj: Vec<Vec<usize>> = Vec::with_capacity(n);
    for i in 0..n {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        let mut neighbors: Vec<usize> = col_idx[start..end]
            .iter()
            .map(|&c| c as usize)
            .filter(|&c| c != i && c < n)
            .collect();
        neighbors.sort_unstable();
        neighbors.dedup();
        adj.push(neighbors);
    }

    let mut eliminated = vec![false; n];
    let mut degree: Vec<usize> = adj.iter().map(|a| a.len()).collect();
    let mut perm = Vec::with_capacity(n);

    for _ in 0..n {
        // Find node with minimum degree among non-eliminated nodes
        let mut min_node = None;
        let mut min_deg = usize::MAX;
        for (i, (&d, &elim)) in degree.iter().zip(eliminated.iter()).enumerate() {
            if !elim && d < min_deg {
                min_deg = d;
                min_node = Some(i);
            }
        }

        let node = match min_node {
            Some(v) => v,
            None => break,
        };

        perm.push(node);
        eliminated[node] = true;

        // Collect non-eliminated neighbors
        let neighbors: Vec<usize> = adj[node]
            .iter()
            .copied()
            .filter(|&nbr| !eliminated[nbr])
            .collect();

        // Mass elimination: connect all neighbors of `node` to each other
        // and update degrees
        for &nbr in &neighbors {
            // Remove `node` from nbr's adjacency
            adj[nbr].retain(|&x| x != node);

            // Add edges to all other neighbors of `node`
            for &other in &neighbors {
                if other != nbr && !adj[nbr].contains(&other) {
                    adj[nbr].push(other);
                }
            }

            // Update degree
            degree[nbr] = adj[nbr].iter().filter(|&&x| !eliminated[x]).count();
        }

        degree[node] = 0;
    }

    Ok(perm)
}

// ---------------------------------------------------------------------------
// Permutation utilities
// ---------------------------------------------------------------------------

/// Apply a permutation to a CSR matrix: `P * A * P^T`.
///
/// Given a permutation `perm` where `perm[new_index] = old_index`,
/// computes the reordered matrix.
///
/// # Arguments
///
/// * `matrix` -- CSR matrix to permute.
/// * `perm` -- Permutation vector of length `n`.
///
/// # Errors
///
/// Returns [`SparseError::InvalidArgument`] if `perm` length does not match
/// the matrix dimension or contains invalid indices.
pub fn permute_csr<T: GpuFloat>(
    matrix: &CsrMatrix<T>,
    perm: &[usize],
) -> SparseResult<CsrMatrix<T>> {
    let n = matrix.rows() as usize;
    if perm.len() != n {
        return Err(SparseError::InvalidArgument(format!(
            "permutation length ({}) must match matrix dimension ({})",
            perm.len(),
            n
        )));
    }

    // Validate permutation
    let inv_perm = inverse_permutation(perm);
    if inv_perm.len() != n {
        return Err(SparseError::InvalidArgument(
            "invalid permutation: not a valid bijection".to_string(),
        ));
    }

    let (h_row_ptr, h_col_idx, h_values) = matrix.to_host()?;

    // Build reordered matrix: new_row i corresponds to old_row perm[i]
    let mut new_row_ptr = vec![0i32; n + 1];
    let mut new_entries: Vec<Vec<(i32, T)>> = Vec::with_capacity(n);

    for new_row in 0..n {
        let old_row = perm[new_row];
        if old_row >= n {
            return Err(SparseError::InvalidArgument(format!(
                "permutation index {} out of bounds (n={})",
                old_row, n
            )));
        }

        let start = h_row_ptr[old_row] as usize;
        let end = h_row_ptr[old_row + 1] as usize;

        let mut entries: Vec<(i32, T)> = Vec::with_capacity(end - start);
        for k in start..end {
            let old_col = h_col_idx[k] as usize;
            if old_col >= n {
                return Err(SparseError::InvalidArgument(format!(
                    "column index {} out of bounds (n={})",
                    old_col, n
                )));
            }
            let new_col = inv_perm[old_col];
            entries.push((new_col as i32, h_values[k]));
        }

        // Sort by new column index
        entries.sort_by_key(|&(c, _)| c);

        new_row_ptr[new_row + 1] = new_row_ptr[new_row] + entries.len() as i32;
        new_entries.push(entries);
    }

    let nnz = new_row_ptr[n] as usize;
    if nnz == 0 {
        return Err(SparseError::ZeroNnz);
    }

    let mut new_col_idx = Vec::with_capacity(nnz);
    let mut new_values = Vec::with_capacity(nnz);
    for entries in &new_entries {
        for &(c, v) in entries {
            new_col_idx.push(c);
            new_values.push(v);
        }
    }

    CsrMatrix::from_host(
        matrix.rows(),
        matrix.cols(),
        &new_row_ptr,
        &new_col_idx,
        &new_values,
    )
}

/// Compute the inverse permutation.
///
/// Given `perm[new] = old`, returns `inv[old] = new`.
pub fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let n = perm.len();
    let mut inv = vec![0usize; n];
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        if old_idx < n {
            inv[old_idx] = new_idx;
        }
    }
    inv
}

/// Compute the bandwidth of a matrix given its CSR structure.
///
/// Bandwidth = max |i - j| over all nonzero entries (i, j).
pub fn bandwidth(row_ptr: &[i32], col_idx: &[i32], n: usize) -> usize {
    let mut bw = 0usize;
    for i in 0..n {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        for &c in &col_idx[start..end] {
            let j = c as usize;
            let diff = i.abs_diff(j);
            if diff > bw {
                bw = diff;
            }
        }
    }
    bw
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rcm_identity() {
        // Identity: any ordering is valid, bandwidth = 0
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let perm = rcm_ordering_host(&row_ptr, &col_idx, 3);
        assert!(perm.is_ok());
        let perm = perm.expect("test: should succeed");
        assert_eq!(perm.len(), 3);
        // Should be a valid permutation
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn rcm_tridiagonal() {
        // Tridiagonal 5x5
        let row_ptr = vec![0, 2, 5, 8, 11, 13];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let perm = rcm_ordering_host(&row_ptr, &col_idx, 5);
        assert!(perm.is_ok());
        let perm = perm.expect("test: should succeed");
        assert_eq!(perm.len(), 5);
        // Valid permutation
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn rcm_reduces_bandwidth() {
        // Arrow matrix: row 0 connects to all, others only to 0 and self
        // Original bandwidth is n-1.
        // [1 1 1 1 1]
        // [1 1 0 0 0]
        // [1 0 1 0 0]
        // [1 0 0 1 0]
        // [1 0 0 0 1]
        let row_ptr = vec![0, 5, 7, 9, 11, 13];
        let col_idx = vec![0, 1, 2, 3, 4, 0, 1, 0, 2, 0, 3, 0, 4];
        let n = 5;
        let orig_bw = bandwidth(&row_ptr, &col_idx, n);
        assert_eq!(orig_bw, 4);

        let perm = rcm_ordering_host(&row_ptr, &col_idx, n);
        assert!(perm.is_ok());
        let perm = perm.expect("test: should succeed");

        // Apply permutation to check new bandwidth
        let inv = inverse_permutation(&perm);
        let mut new_bw = 0;
        for (i, &old_row) in perm.iter().enumerate().take(n) {
            let start = row_ptr[old_row] as usize;
            let end = row_ptr[old_row + 1] as usize;
            for &c in &col_idx[start..end] {
                let new_col = inv[c as usize];
                let diff = i.abs_diff(new_col);
                if diff > new_bw {
                    new_bw = diff;
                }
            }
        }
        // RCM should not increase bandwidth
        assert!(new_bw <= orig_bw);
    }

    #[test]
    fn amd_identity() {
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let perm = amd_ordering_host(&row_ptr, &col_idx, 3);
        assert!(perm.is_ok());
        let perm = perm.expect("test: should succeed");
        assert_eq!(perm.len(), 3);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn amd_tridiagonal() {
        let row_ptr = vec![0, 2, 5, 8, 10];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let perm = amd_ordering_host(&row_ptr, &col_idx, 4);
        assert!(perm.is_ok());
        let perm = perm.expect("test: should succeed");
        assert_eq!(perm.len(), 4);
        // Valid permutation: all nodes appear exactly once
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn inverse_permutation_roundtrip() {
        let perm = vec![3, 1, 0, 2];
        let inv = inverse_permutation(&perm);
        assert_eq!(inv, vec![2, 1, 3, 0]);

        // inv(inv(perm)) == perm
        let inv_inv = inverse_permutation(&inv);
        assert_eq!(inv_inv, perm);
    }

    #[test]
    fn bandwidth_calculation() {
        // Tridiagonal: bandwidth = 1
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        assert_eq!(bandwidth(&row_ptr, &col_idx, 3), 1);

        // Diagonal: bandwidth = 0
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        assert_eq!(bandwidth(&row_ptr, &col_idx, 3), 0);
    }

    #[test]
    fn rcm_empty() {
        let perm = rcm_ordering_host(&[0], &[], 0);
        assert!(perm.is_ok());
        assert!(perm.expect("test: should succeed").is_empty());
    }

    #[test]
    fn amd_empty() {
        let perm = amd_ordering_host(&[0], &[], 0);
        assert!(perm.is_ok());
        assert!(perm.expect("test: should succeed").is_empty());
    }
}
