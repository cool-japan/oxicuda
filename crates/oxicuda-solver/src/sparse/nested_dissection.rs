//! Nested dissection ordering for sparse direct solvers.
//!
//! Provides graph-based fill-reducing orderings that minimize fill-in during
//! sparse Cholesky or LU factorization. The main algorithm recursively finds
//! vertex separators to split the graph, orders each partition recursively,
//! then places separator vertices last.
//!
//! # Algorithm
//!
//! 1. Build an [`AdjacencyGraph`] from a symmetric CSR matrix.
//! 2. Call [`NestedDissectionOrdering::compute`] to obtain a [`Permutation`].
//! 3. Evaluate quality with [`analyze_ordering`].
//!
//! The separator finder uses a multi-level approach: heavy-edge coarsening,
//! BFS-based bisection on the coarse graph, and Fiduccia-Mattheyses refinement.

use std::collections::VecDeque;

use crate::error::{SolverError, SolverResult};

// ---------------------------------------------------------------------------
// Permutation
// ---------------------------------------------------------------------------

/// A permutation with its inverse, for reordering matrix rows/columns.
#[derive(Debug, Clone)]
pub struct Permutation {
    /// Forward permutation: `new[i] = old[perm[i]]`.
    pub perm: Vec<usize>,
    /// Inverse permutation: `old[i] = new[iperm[i]]`.
    pub iperm: Vec<usize>,
}

impl Permutation {
    /// Create a permutation from a forward mapping. Returns an error if any
    /// index is out of range or duplicated.
    pub fn new(perm: Vec<usize>) -> SolverResult<Self> {
        let n = perm.len();
        let mut iperm = vec![0usize; n];
        let mut seen = vec![false; n];
        for (i, &p) in perm.iter().enumerate() {
            if p >= n {
                return Err(SolverError::InternalError(format!(
                    "permutation index {p} out of range for size {n}"
                )));
            }
            if seen[p] {
                return Err(SolverError::InternalError(format!(
                    "duplicate permutation index {p}"
                )));
            }
            seen[p] = true;
            iperm[p] = i;
        }
        Ok(Self { perm, iperm })
    }

    /// Identity permutation of size `n`.
    pub fn identity(n: usize) -> Self {
        Self {
            perm: (0..n).collect(),
            iperm: (0..n).collect(),
        }
    }

    /// Apply permutation to a slice: `result[i] = data[perm[i]]`.
    pub fn apply<T: Copy>(&self, data: &[T]) -> Vec<T> {
        self.perm.iter().map(|&p| data[p]).collect()
    }

    /// Compose two permutations: `result[i] = self.perm[other.perm[i]]`.
    pub fn compose(&self, other: &Permutation) -> SolverResult<Permutation> {
        let composed: Vec<usize> = other.perm.iter().map(|&i| self.perm[i]).collect();
        Permutation::new(composed)
    }

    /// Return the inverse permutation as a slice.
    pub fn inverse(&self) -> &[usize] {
        &self.iperm
    }
}

// ---------------------------------------------------------------------------
// AdjacencyGraph
// ---------------------------------------------------------------------------

/// Undirected graph of a symmetric sparse matrix in CSR-like form.
///
/// Self-loops (diagonal entries) are excluded. Each edge `(i,j)` appears in
/// both `neighbors(i)` and `neighbors(j)`.
#[derive(Debug, Clone)]
pub struct AdjacencyGraph {
    /// Number of vertices.
    pub num_vertices: usize,
    /// CSR-style pointers into `adj_list`. Length = `num_vertices + 1`.
    pub adj_ptr: Vec<usize>,
    /// Concatenated adjacency lists for all vertices.
    pub adj_list: Vec<usize>,
}

impl AdjacencyGraph {
    /// Build an adjacency graph from a symmetric CSR matrix.
    ///
    /// Diagonal entries are excluded. The CSR uses `i32` indices (matching
    /// cuSPARSE convention). Both upper and lower triangular parts should be
    /// present for a symmetric matrix.
    pub fn from_symmetric_csr(row_ptr: &[i32], col_indices: &[i32], n: usize) -> Self {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for row in 0..n {
            let start = row_ptr[row] as usize;
            let end = row_ptr[row + 1] as usize;
            for &col_i32 in &col_indices[start..end] {
                let col = col_i32 as usize;
                if col != row && col < n {
                    adj[row].push(col);
                }
            }
        }
        // Deduplicate each adjacency list
        for list in &mut adj {
            list.sort_unstable();
            list.dedup();
        }
        // Flatten to CSR
        let mut adj_ptr = Vec::with_capacity(n + 1);
        let mut adj_list = Vec::new();
        adj_ptr.push(0);
        for list in &adj {
            adj_list.extend_from_slice(list);
            adj_ptr.push(adj_list.len());
        }
        Self {
            num_vertices: n,
            adj_ptr,
            adj_list,
        }
    }

    /// Degree of vertex `v`.
    pub fn degree(&self, v: usize) -> usize {
        self.adj_ptr[v + 1] - self.adj_ptr[v]
    }

    /// Neighbors of vertex `v`.
    pub fn neighbors(&self, v: usize) -> &[usize] {
        &self.adj_list[self.adj_ptr[v]..self.adj_ptr[v + 1]]
    }
}

// ---------------------------------------------------------------------------
// SeparatorResult
// ---------------------------------------------------------------------------

/// Result of a graph separator computation.
#[derive(Debug, Clone)]
pub struct SeparatorResult {
    /// Vertices forming the separator.
    pub separator: Vec<usize>,
    /// Vertices in partition A.
    pub part_a: Vec<usize>,
    /// Vertices in partition B.
    pub part_b: Vec<usize>,
    /// Balance ratio (0..1], 1.0 = perfectly balanced.
    pub balance_ratio: f64,
}

// ---------------------------------------------------------------------------
// OrderingQuality
// ---------------------------------------------------------------------------

/// Quality metrics for an ordering.
#[derive(Debug, Clone)]
pub struct OrderingQuality {
    /// Estimated fill-in count (non-zero entries added during factorization).
    pub estimated_fill: usize,
    /// Total separator size across all recursive levels.
    pub separator_size: usize,
    /// Balance ratio of the top-level partition.
    pub balance_ratio: f64,
}

// ---------------------------------------------------------------------------
// BFS level-set
// ---------------------------------------------------------------------------

/// BFS from a source vertex within a subgraph, returning level sets.
fn bfs_levels(graph: &AdjacencyGraph, source: usize, in_subgraph: &[bool]) -> Vec<Vec<usize>> {
    let n = graph.num_vertices;
    let mut visited = vec![false; n];
    visited[source] = true;
    let mut levels: Vec<Vec<usize>> = vec![vec![source]];
    let mut queue = VecDeque::new();
    queue.push_back(source);
    while !queue.is_empty() {
        let mut next_level = Vec::new();
        let count = queue.len();
        for _ in 0..count {
            if let Some(v) = queue.pop_front() {
                for &nb in graph.neighbors(v) {
                    if in_subgraph[nb] && !visited[nb] {
                        visited[nb] = true;
                        next_level.push(nb);
                        queue.push_back(nb);
                    }
                }
            }
        }
        if !next_level.is_empty() {
            levels.push(next_level);
        }
    }
    levels
}

/// Find a pseudo-peripheral vertex in the subgraph (for good BFS root).
fn pseudo_peripheral(graph: &AdjacencyGraph, subgraph: &[usize]) -> usize {
    if subgraph.is_empty() {
        return 0;
    }
    let n = graph.num_vertices;
    let mut in_sub = vec![false; n];
    for &v in subgraph {
        in_sub[v] = true;
    }
    // Start from vertex with minimum degree in subgraph
    let mut start = subgraph[0];
    let mut min_deg = usize::MAX;
    for &v in subgraph {
        let d = graph.neighbors(v).iter().filter(|&&nb| in_sub[nb]).count();
        if d < min_deg {
            min_deg = d;
            start = v;
        }
    }
    // Run BFS twice to find pseudo-peripheral vertex
    for _ in 0..2 {
        let levels = bfs_levels(graph, start, &in_sub);
        if let Some(last) = levels.last() {
            // Pick vertex with minimum degree in last level
            let mut best = last[0];
            let mut best_deg = usize::MAX;
            for &v in last {
                let d = graph.neighbors(v).iter().filter(|&&nb| in_sub[nb]).count();
                if d < best_deg {
                    best_deg = d;
                    best = v;
                }
            }
            start = best;
        }
    }
    start
}

// ---------------------------------------------------------------------------
// Heavy-edge matching (coarsening)
// ---------------------------------------------------------------------------

/// Coarsen the subgraph via heavy-edge matching.
/// Returns (coarse_map, coarse_count) where coarse_map[v] = coarse vertex ID.
fn coarsen(graph: &AdjacencyGraph, subgraph: &[usize]) -> (Vec<i32>, usize) {
    let n = graph.num_vertices;
    let mut coarse_map = vec![-1i32; n];
    let mut matched = vec![false; n];
    let mut coarse_id = 0usize;

    // Sort by degree (ascending) for better matching
    let mut sorted = subgraph.to_vec();
    sorted.sort_unstable_by_key(|&v| graph.degree(v));

    let in_sub: Vec<bool> = {
        let mut s = vec![false; n];
        for &v in subgraph {
            s[v] = true;
        }
        s
    };

    for &v in &sorted {
        if matched[v] {
            continue;
        }
        // Find heaviest unmatched neighbor (by degree sum as weight proxy)
        let mut best_nb = None;
        let mut best_weight = 0usize;
        for &nb in graph.neighbors(v) {
            if in_sub[nb] && !matched[nb] && nb != v {
                let w = graph.degree(v) + graph.degree(nb);
                if w > best_weight {
                    best_weight = w;
                    best_nb = Some(nb);
                }
            }
        }
        let cid = coarse_id as i32;
        coarse_map[v] = cid;
        matched[v] = true;
        if let Some(nb) = best_nb {
            coarse_map[nb] = cid;
            matched[nb] = true;
        }
        coarse_id += 1;
    }
    (coarse_map, coarse_id)
}

// ---------------------------------------------------------------------------
// BFS-based bisection
// ---------------------------------------------------------------------------

/// Bisect a subgraph using BFS level sets, returning (part_a, part_b, separator).
fn bfs_bisect(graph: &AdjacencyGraph, subgraph: &[usize]) -> SeparatorResult {
    if subgraph.len() <= 1 {
        return SeparatorResult {
            separator: subgraph.to_vec(),
            part_a: Vec::new(),
            part_b: Vec::new(),
            balance_ratio: 1.0,
        };
    }
    let n = graph.num_vertices;
    let mut in_sub = vec![false; n];
    for &v in subgraph {
        in_sub[v] = true;
    }
    let start = pseudo_peripheral(graph, subgraph);
    let levels = bfs_levels(graph, start, &in_sub);

    if levels.len() <= 2 {
        // Too few levels — use simple split
        let half = subgraph.len() / 2;
        let part_a = subgraph[..half].to_vec();
        let separator = vec![subgraph[half]];
        let part_b = if half + 1 < subgraph.len() {
            subgraph[half + 1..].to_vec()
        } else {
            Vec::new()
        };
        let balance = compute_balance(part_a.len(), part_b.len());
        return SeparatorResult {
            separator,
            part_a,
            part_b,
            balance_ratio: balance,
        };
    }

    // Find the middle level as separator
    let mid = levels.len() / 2;
    let separator = levels[mid].clone();
    let mut part_a = Vec::new();
    let mut part_b = Vec::new();
    for (i, level) in levels.iter().enumerate() {
        if i < mid {
            part_a.extend_from_slice(level);
        } else if i > mid {
            part_b.extend_from_slice(level);
        }
    }
    let balance = compute_balance(part_a.len(), part_b.len());
    SeparatorResult {
        separator,
        part_a,
        part_b,
        balance_ratio: balance,
    }
}

/// Compute balance ratio: min(a,b)/max(a,b), or 1.0 if both zero.
fn compute_balance(a: usize, b: usize) -> f64 {
    if a == 0 && b == 0 {
        return 1.0;
    }
    let min_v = a.min(b) as f64;
    let max_v = a.max(b) as f64;
    if max_v == 0.0 { 1.0 } else { min_v / max_v }
}

// ---------------------------------------------------------------------------
// FM refinement
// ---------------------------------------------------------------------------

/// Single-pass Fiduccia-Mattheyses refinement to improve separator quality.
fn fm_refine(graph: &AdjacencyGraph, result: &mut SeparatorResult) {
    let n = graph.num_vertices;
    // partition[v]: 0=A, 1=B, 2=separator
    let mut partition = vec![0u8; n];
    for &v in &result.part_a {
        partition[v] = 0;
    }
    for &v in &result.part_b {
        partition[v] = 1;
    }
    for &v in &result.separator {
        partition[v] = 2;
    }

    let total = result.part_a.len() + result.part_b.len() + result.separator.len();
    if total <= 3 {
        return;
    }

    // Try moving separator vertices to A or B if it doesn't break connectivity
    let mut improved = true;
    let mut iterations = 0;
    while improved && iterations < 10 {
        improved = false;
        iterations += 1;
        let sep_copy = result.separator.clone();
        for &sv in &sep_copy {
            // Count neighbors in A, B, separator
            let mut na = 0usize;
            let mut nb = 0usize;
            for &nb_v in graph.neighbors(sv) {
                match partition[nb_v] {
                    0 => na += 1,
                    1 => nb += 1,
                    _ => {}
                }
            }
            // Move to side with more connections if safe
            if na > 0 && nb == 0 {
                partition[sv] = 0;
                result.part_a.push(sv);
                result.separator.retain(|&x| x != sv);
                improved = true;
            } else if nb > 0 && na == 0 {
                partition[sv] = 1;
                result.part_b.push(sv);
                result.separator.retain(|&x| x != sv);
                improved = true;
            }
        }
    }

    // Try moving boundary vertices of A/B into separator if it improves balance
    let a_len = result.part_a.len();
    let b_len = result.part_b.len();
    if a_len > 2 * b_len + 2 {
        // A is too large, move some A-boundary vertices to separator
        let mut candidates: Vec<usize> = result
            .part_a
            .iter()
            .copied()
            .filter(|&v| {
                graph
                    .neighbors(v)
                    .iter()
                    .any(|&nb| partition[nb] == 1 || partition[nb] == 2)
            })
            .collect();
        candidates.sort_unstable_by_key(|&v| graph.degree(v));
        let to_move = (a_len - b_len) / 4;
        for &v in candidates.iter().take(to_move) {
            partition[v] = 2;
            result.separator.push(v);
            result.part_a.retain(|&x| x != v);
        }
    } else if b_len > 2 * a_len + 2 {
        let mut candidates: Vec<usize> = result
            .part_b
            .iter()
            .copied()
            .filter(|&v| {
                graph
                    .neighbors(v)
                    .iter()
                    .any(|&nb| partition[nb] == 0 || partition[nb] == 2)
            })
            .collect();
        candidates.sort_unstable_by_key(|&v| graph.degree(v));
        let to_move = (b_len - a_len) / 4;
        for &v in candidates.iter().take(to_move) {
            partition[v] = 2;
            result.separator.push(v);
            result.part_b.retain(|&x| x != v);
        }
    }

    result.balance_ratio = compute_balance(result.part_a.len(), result.part_b.len());
}

// ---------------------------------------------------------------------------
// VertexSeparator
// ---------------------------------------------------------------------------

/// Finds graph vertex separators using a multi-level approach.
pub struct VertexSeparator;

impl VertexSeparator {
    /// Find a vertex separator for the given subgraph vertices.
    ///
    /// Uses multi-level coarsening + BFS bisection + FM refinement.
    pub fn find_separator(graph: &AdjacencyGraph, subgraph: &[usize]) -> SeparatorResult {
        if subgraph.len() <= 3 {
            return Self::trivial_separator(subgraph);
        }
        // Coarsen
        let (coarse_map, coarse_count) = coarsen(graph, subgraph);
        if coarse_count >= subgraph.len() || coarse_count <= 1 {
            // Coarsening didn't help, bisect directly
            let mut result = bfs_bisect(graph, subgraph);
            fm_refine(graph, &mut result);
            return result;
        }

        // Build coarse subgraph vertex list (representative vertices)
        let mut coarse_repr: Vec<Option<usize>> = vec![None; coarse_count];
        for &v in subgraph {
            let cid = coarse_map[v];
            if cid >= 0 {
                let cid = cid as usize;
                if coarse_repr[cid].is_none() {
                    coarse_repr[cid] = Some(v);
                }
            }
        }
        let coarse_verts: Vec<usize> = coarse_repr.iter().filter_map(|&x| x).collect();

        // Bisect coarse graph
        let coarse_result = bfs_bisect(graph, &coarse_verts);

        // Uncoarsen: map coarse partition back to fine vertices
        let n = graph.num_vertices;
        let mut fine_part = vec![0u8; n]; // 0=A, 1=B, 2=sep
        for &v in &coarse_result.part_a {
            let cid = coarse_map[v];
            for &fv in subgraph {
                if coarse_map[fv] == cid {
                    fine_part[fv] = 0;
                }
            }
        }
        for &v in &coarse_result.part_b {
            let cid = coarse_map[v];
            for &fv in subgraph {
                if coarse_map[fv] == cid {
                    fine_part[fv] = 1;
                }
            }
        }
        for &v in &coarse_result.separator {
            let cid = coarse_map[v];
            for &fv in subgraph {
                if coarse_map[fv] == cid {
                    fine_part[fv] = 2;
                }
            }
        }

        // Build boundary separator: vertices in A adjacent to B become separator
        let mut separator = Vec::new();
        let mut part_a = Vec::new();
        let mut part_b = Vec::new();
        for &v in subgraph {
            if fine_part[v] == 2 {
                separator.push(v);
            } else {
                let is_boundary = graph
                    .neighbors(v)
                    .iter()
                    .any(|&nb| fine_part[nb] != fine_part[v] && fine_part[nb] != 2);
                if is_boundary {
                    separator.push(v);
                } else if fine_part[v] == 0 {
                    part_a.push(v);
                } else {
                    part_b.push(v);
                }
            }
        }

        let balance = compute_balance(part_a.len(), part_b.len());
        let mut result = SeparatorResult {
            separator,
            part_a,
            part_b,
            balance_ratio: balance,
        };
        fm_refine(graph, &mut result);
        result
    }

    /// Trivial separator for tiny subgraphs.
    fn trivial_separator(subgraph: &[usize]) -> SeparatorResult {
        match subgraph.len() {
            0 => SeparatorResult {
                separator: Vec::new(),
                part_a: Vec::new(),
                part_b: Vec::new(),
                balance_ratio: 1.0,
            },
            1 => SeparatorResult {
                separator: subgraph.to_vec(),
                part_a: Vec::new(),
                part_b: Vec::new(),
                balance_ratio: 1.0,
            },
            2 => SeparatorResult {
                separator: vec![subgraph[0]],
                part_a: vec![subgraph[1]],
                part_b: Vec::new(),
                balance_ratio: 0.0,
            },
            _ => SeparatorResult {
                separator: vec![subgraph[1]],
                part_a: vec![subgraph[0]],
                part_b: vec![subgraph[2]],
                balance_ratio: 1.0,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// MinimumDegreeOrdering
// ---------------------------------------------------------------------------

/// Approximate minimum degree ordering for small subgraphs.
pub struct MinimumDegreeOrdering;

impl MinimumDegreeOrdering {
    /// Compute an approximate minimum degree ordering for the given vertices.
    ///
    /// Greedily eliminates the vertex with minimum degree at each step.
    pub fn compute(graph: &AdjacencyGraph, vertices: &[usize]) -> Vec<usize> {
        if vertices.is_empty() {
            return Vec::new();
        }
        let n = graph.num_vertices;
        let mut in_sub = vec![false; n];
        for &v in vertices {
            in_sub[v] = true;
        }
        let mut eliminated = vec![false; n];
        let mut order = Vec::with_capacity(vertices.len());

        for _ in 0..vertices.len() {
            // Find vertex with minimum current degree
            let mut best = None;
            let mut best_deg = usize::MAX;
            for &v in vertices {
                if eliminated[v] {
                    continue;
                }
                let deg = graph
                    .neighbors(v)
                    .iter()
                    .filter(|&&nb| in_sub[nb] && !eliminated[nb])
                    .count();
                if deg < best_deg {
                    best_deg = deg;
                    best = Some(v);
                }
            }
            if let Some(v) = best {
                order.push(v);
                eliminated[v] = true;
            }
        }
        order
    }
}

// ---------------------------------------------------------------------------
// NestedDissectionOrdering
// ---------------------------------------------------------------------------

/// Threshold below which we switch to minimum degree ordering.
const ND_THRESHOLD: usize = 32;

/// Maximum recursion depth to prevent stack overflow.
const MAX_DEPTH: usize = 64;

/// Nested dissection ordering algorithm.
///
/// Recursively finds vertex separators to split the graph, orders each
/// partition recursively, then places separator vertices last. This produces
/// a fill-reducing ordering for sparse Cholesky/LU factorization.
pub struct NestedDissectionOrdering;

impl NestedDissectionOrdering {
    /// Compute a fill-reducing permutation via nested dissection.
    pub fn compute(graph: &AdjacencyGraph) -> SolverResult<Permutation> {
        let n = graph.num_vertices;
        if n == 0 {
            return Permutation::new(Vec::new());
        }
        let vertices: Vec<usize> = (0..n).collect();
        let mut order = Vec::with_capacity(n);
        Self::recurse(graph, &vertices, &mut order, 0);
        // order now contains vertices in elimination order
        // perm[i] = which original vertex goes to position i
        Permutation::new(order)
    }

    /// Recursive nested dissection.
    fn recurse(graph: &AdjacencyGraph, subgraph: &[usize], order: &mut Vec<usize>, depth: usize) {
        if subgraph.len() <= ND_THRESHOLD || depth >= MAX_DEPTH {
            let md_order = MinimumDegreeOrdering::compute(graph, subgraph);
            order.extend(md_order);
            return;
        }

        let sep = VertexSeparator::find_separator(graph, subgraph);

        // Recurse on A, then B
        if !sep.part_a.is_empty() {
            Self::recurse(graph, &sep.part_a, order, depth + 1);
        }
        if !sep.part_b.is_empty() {
            Self::recurse(graph, &sep.part_b, order, depth + 1);
        }
        // Separator last
        order.extend(&sep.separator);
    }
}

// ---------------------------------------------------------------------------
// analyze_ordering
// ---------------------------------------------------------------------------

/// Analyze the quality of an ordering by estimating fill-in via symbolic
/// factorization.
pub fn analyze_ordering(graph: &AdjacencyGraph, perm: &Permutation) -> OrderingQuality {
    let n = graph.num_vertices;
    if n == 0 {
        return OrderingQuality {
            estimated_fill: 0,
            separator_size: 0,
            balance_ratio: 1.0,
        };
    }

    // Symbolic Cholesky: estimate fill using elimination tree
    let iperm = perm.inverse();
    let mut fill = 0usize;
    let mut parent = vec![n; n]; // elimination tree parent (n = root)

    for i in 0..n {
        let orig = perm.perm[i]; // original vertex eliminated at step i
        let mut visited = vec![false; n];
        visited[i] = true;
        for &nb in graph.neighbors(orig) {
            let mut j = iperm[nb]; // position of neighbor in elimination order
            // Walk up elimination tree
            while j < n && !visited[j] && j > i {
                visited[j] = true;
                fill += 1;
                if parent[j] == n || parent[j] > i {
                    // j has no parent yet or we found a closer one
                    // Don't overwrite if already set to something smaller
                    if parent[j] == n {
                        parent[j] = i;
                    }
                }
                j = parent[j];
            }
        }
    }

    // Get separator info from top-level partition
    let vertices: Vec<usize> = (0..n).collect();
    let sep = if n > 3 {
        VertexSeparator::find_separator(graph, &vertices)
    } else {
        SeparatorResult {
            separator: Vec::new(),
            part_a: Vec::new(),
            part_b: Vec::new(),
            balance_ratio: 1.0,
        }
    };

    OrderingQuality {
        estimated_fill: fill,
        separator_size: sep.separator.len(),
        balance_ratio: sep.balance_ratio,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple path graph: 0-1-2-..-(n-1)
    fn path_graph(n: usize) -> AdjacencyGraph {
        let mut row_ptr = vec![0i32];
        let mut col_indices = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_indices.push((i - 1) as i32);
            }
            if i + 1 < n {
                col_indices.push((i + 1) as i32);
            }
            row_ptr.push(col_indices.len() as i32);
        }
        AdjacencyGraph::from_symmetric_csr(&row_ptr, &col_indices, n)
    }

    /// Build a grid graph of size rows x cols.
    fn grid_graph(rows: usize, cols: usize) -> AdjacencyGraph {
        let n = rows * cols;
        let mut row_ptr = vec![0i32];
        let mut col_indices = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                if r > 0 {
                    col_indices.push((v - cols) as i32);
                }
                if c > 0 {
                    col_indices.push((v - 1) as i32);
                }
                if c + 1 < cols {
                    col_indices.push((v + 1) as i32);
                }
                if r + 1 < rows {
                    col_indices.push((v + cols) as i32);
                }
                row_ptr.push(col_indices.len() as i32);
            }
        }
        AdjacencyGraph::from_symmetric_csr(&row_ptr, &col_indices, n)
    }

    #[test]
    fn permutation_identity() {
        let p = Permutation::identity(5);
        assert_eq!(p.perm, vec![0, 1, 2, 3, 4]);
        assert_eq!(p.inverse(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn permutation_valid() {
        let p = Permutation::new(vec![2, 0, 1]).expect("valid perm");
        assert_eq!(p.perm, vec![2, 0, 1]);
        assert_eq!(p.iperm, vec![1, 2, 0]);
    }

    #[test]
    fn permutation_invalid_duplicate() {
        let result = Permutation::new(vec![0, 0, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn permutation_invalid_out_of_range() {
        let result = Permutation::new(vec![0, 5, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn permutation_apply() {
        let p = Permutation::new(vec![2, 0, 1]).expect("valid perm");
        let data = vec![10, 20, 30];
        let result = p.apply(&data);
        assert_eq!(result, vec![30, 10, 20]);
    }

    #[test]
    fn permutation_compose() {
        let p1 = Permutation::new(vec![1, 2, 0]).expect("valid");
        let p2 = Permutation::new(vec![2, 0, 1]).expect("valid");
        let composed = p1.compose(&p2).expect("compose");
        // composed[i] = p1.perm[p2.perm[i]]
        // p2.perm = [2,0,1], p1.perm = [1,2,0]
        // composed[0] = p1[2] = 0, composed[1] = p1[0] = 1, composed[2] = p1[1] = 2
        assert_eq!(composed.perm, vec![0, 1, 2]);
    }

    #[test]
    fn adjacency_graph_path() {
        let g = path_graph(5);
        assert_eq!(g.num_vertices, 5);
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(2), 2);
        assert_eq!(g.degree(4), 1);
        assert_eq!(g.neighbors(2), &[1, 3]);
    }

    #[test]
    fn adjacency_graph_grid() {
        let g = grid_graph(3, 3);
        assert_eq!(g.num_vertices, 9);
        // Corner has 2 neighbors
        assert_eq!(g.degree(0), 2);
        // Center has 4 neighbors
        assert_eq!(g.degree(4), 4);
    }

    #[test]
    fn separator_splits_graph() {
        let g = grid_graph(4, 4);
        let verts: Vec<usize> = (0..16).collect();
        let sep = VertexSeparator::find_separator(&g, &verts);
        // All vertices accounted for
        let total = sep.separator.len() + sep.part_a.len() + sep.part_b.len();
        assert_eq!(total, 16);
        // Separator is non-empty
        assert!(!sep.separator.is_empty());
        // No edge between part_a and part_b (they are separated)
        let n = g.num_vertices;
        let mut part = vec![0u8; n];
        for &v in &sep.part_a {
            part[v] = 1;
        }
        for &v in &sep.part_b {
            part[v] = 2;
        }
        for &v in &sep.part_a {
            for &nb in g.neighbors(v) {
                assert_ne!(part[nb], 2, "edge {v}-{nb} crosses separator");
            }
        }
    }

    #[test]
    fn nd_ordering_valid_permutation() {
        let g = grid_graph(4, 4);
        let perm = NestedDissectionOrdering::compute(&g).expect("nd ordering");
        assert_eq!(perm.perm.len(), 16);
        // Check it's a valid permutation
        let mut sorted = perm.perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn nd_ordering_empty_graph() {
        let g = AdjacencyGraph::from_symmetric_csr(&[0], &[], 0);
        let perm = NestedDissectionOrdering::compute(&g).expect("empty");
        assert!(perm.perm.is_empty());
    }

    #[test]
    fn nd_ordering_single_vertex() {
        let g = AdjacencyGraph::from_symmetric_csr(&[0, 0], &[], 1);
        let perm = NestedDissectionOrdering::compute(&g).expect("single");
        assert_eq!(perm.perm, vec![0]);
    }

    #[test]
    fn nd_reduces_fill_vs_natural() {
        let g = grid_graph(8, 8);
        let nd_perm = NestedDissectionOrdering::compute(&g).expect("nd");
        let natural_perm = Permutation::identity(64);
        let nd_quality = analyze_ordering(&g, &nd_perm);
        let nat_quality = analyze_ordering(&g, &natural_perm);
        // ND should produce less fill than natural ordering for grid
        assert!(
            nd_quality.estimated_fill <= nat_quality.estimated_fill,
            "ND fill {} should be <= natural fill {}",
            nd_quality.estimated_fill,
            nat_quality.estimated_fill
        );
    }

    #[test]
    fn minimum_degree_ordering_valid() {
        let g = path_graph(10);
        let verts: Vec<usize> = (0..10).collect();
        let order = MinimumDegreeOrdering::compute(&g, &verts);
        assert_eq!(order.len(), 10);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn ordering_quality_metrics() {
        let g = grid_graph(4, 4);
        let perm = NestedDissectionOrdering::compute(&g).expect("nd");
        let quality = analyze_ordering(&g, &perm);
        assert!(quality.estimated_fill < 200, "fill should be reasonable");
        assert!(quality.balance_ratio >= 0.0 && quality.balance_ratio <= 1.0);
    }

    #[test]
    fn larger_grid_nd() {
        let g = grid_graph(16, 16);
        let perm = NestedDissectionOrdering::compute(&g).expect("nd 16x16");
        assert_eq!(perm.perm.len(), 256);
        let mut sorted = perm.perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..256).collect::<Vec<_>>());
    }
}
