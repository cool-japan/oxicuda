//! Random graph generation for GNN workloads and network analysis.
//!
//! Provides generators for several families of random graphs stored as sparse
//! adjacency structures (CSR-like: `row_offsets`, `col_indices`):
//!
//! - **Erdős–Rényi** G(n, p): each possible edge included independently with
//!   probability p.
//! - **Stochastic Block Model**: community-structured graphs with tunable
//!   intra-/inter-community edge probabilities.
//! - **Barabási–Albert**: scale-free (power-law degree distribution) graphs via
//!   preferential attachment.
//! - **Watts–Strogatz**: small-world graphs interpolating between a ring lattice
//!   and a random graph.
//! - **Random Regular**: k-regular graphs where every vertex has the same degree.
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

use crate::error::{RandError, RandResult};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// CPU-side PRNG (SplitMix64) — same algorithm as in matrix_gen.rs
// ---------------------------------------------------------------------------

/// Simple SplitMix64 PRNG for CPU-side random number generation.
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

    /// Returns a uniform usize in [0, bound).
    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % (bound as u64)) as usize
    }
}

// ---------------------------------------------------------------------------
// GraphType
// ---------------------------------------------------------------------------

/// Whether the graph is directed or undirected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphType {
    /// Undirected graph — edges are symmetric.
    Undirected,
    /// Directed graph — edge (i, j) does not imply (j, i).
    Directed,
}

impl std::fmt::Display for GraphType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Undirected => write!(f, "Undirected"),
            Self::Directed => write!(f, "Directed"),
        }
    }
}

// ---------------------------------------------------------------------------
// GraphStats
// ---------------------------------------------------------------------------

/// Aggregate statistics of a graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Total number of vertices.
    pub num_vertices: usize,
    /// Total number of edges (for undirected graphs, each undirected edge is
    /// counted once even though both (i,j) and (j,i) appear in the adjacency
    /// list).
    pub num_edges: usize,
    /// Average degree across all vertices.
    pub avg_degree: f64,
    /// Maximum degree of any vertex.
    pub max_degree: usize,
    /// Minimum degree of any vertex.
    pub min_degree: usize,
    /// Whether the graph is approximately connected (BFS from vertex 0 reaches
    /// >90 % of vertices).
    pub is_connected: bool,
    /// Edge density: ratio of existing edges to maximum possible edges.
    pub density: f64,
}

impl std::fmt::Display for GraphStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GraphStats(V={}, E={}, avg_deg={:.2}, max_deg={}, min_deg={}, \
             connected={}, density={:.4})",
            self.num_vertices,
            self.num_edges,
            self.avg_degree,
            self.max_degree,
            self.min_degree,
            self.is_connected,
            self.density,
        )
    }
}

// ---------------------------------------------------------------------------
// AdjacencyList
// ---------------------------------------------------------------------------

/// Sparse adjacency representation of a graph in CSR-like format.
///
/// For vertex `v`, its neighbours are stored in
/// `col_indices[row_offsets[v]..row_offsets[v + 1]]`.
#[derive(Debug, Clone)]
pub struct AdjacencyList {
    /// Number of vertices.
    pub num_vertices: usize,
    /// Row offset array of length `num_vertices + 1`.
    pub row_offsets: Vec<usize>,
    /// Column index array containing neighbour vertex ids.
    pub col_indices: Vec<usize>,
    /// Whether the graph is directed or undirected.
    pub graph_type: GraphType,
}

impl AdjacencyList {
    /// Returns the degree of vertex `v`.
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if `v >= num_vertices`.
    pub fn degree(&self, v: usize) -> RandResult<usize> {
        if v >= self.num_vertices {
            return Err(RandError::InvalidSize(format!(
                "vertex {v} out of range [0, {})",
                self.num_vertices
            )));
        }
        Ok(self.row_offsets[v + 1] - self.row_offsets[v])
    }

    /// Returns the neighbours of vertex `v`.
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if `v >= num_vertices`.
    pub fn neighbors(&self, v: usize) -> RandResult<&[usize]> {
        if v >= self.num_vertices {
            return Err(RandError::InvalidSize(format!(
                "vertex {v} out of range [0, {})",
                self.num_vertices
            )));
        }
        Ok(&self.col_indices[self.row_offsets[v]..self.row_offsets[v + 1]])
    }

    /// Computes aggregate statistics of the graph.
    pub fn stats(&self) -> GraphStats {
        if self.num_vertices == 0 {
            return GraphStats {
                num_vertices: 0,
                num_edges: 0,
                avg_degree: 0.0,
                max_degree: 0,
                min_degree: 0,
                is_connected: true,
                density: 0.0,
            };
        }

        let mut max_degree: usize = 0;
        let mut min_degree: usize = usize::MAX;
        let mut total_degree: usize = 0;

        for v in 0..self.num_vertices {
            let d = self.row_offsets[v + 1] - self.row_offsets[v];
            total_degree += d;
            if d > max_degree {
                max_degree = d;
            }
            if d < min_degree {
                min_degree = d;
            }
        }

        let num_edges = match self.graph_type {
            GraphType::Undirected => total_degree / 2,
            GraphType::Directed => total_degree,
        };

        let avg_degree = total_degree as f64 / self.num_vertices as f64;

        let max_possible = match self.graph_type {
            GraphType::Undirected => {
                self.num_vertices
                    .saturating_mul(self.num_vertices.saturating_sub(1))
                    / 2
            }
            GraphType::Directed => self
                .num_vertices
                .saturating_mul(self.num_vertices.saturating_sub(1)),
        };
        let density = if max_possible > 0 {
            num_edges as f64 / max_possible as f64
        } else {
            0.0
        };

        let is_connected = is_approximately_connected(self);

        GraphStats {
            num_vertices: self.num_vertices,
            num_edges,
            avg_degree,
            max_degree,
            min_degree,
            is_connected,
            density,
        }
    }

    /// Converts to CSR format with uniform edge weights.
    ///
    /// Returns `(row_offsets, col_indices, values)` where every edge receives
    /// the supplied `default_val`.
    pub fn to_csr_values<T: Default + Copy>(
        &self,
        default_val: T,
    ) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        let values = vec![default_val; self.col_indices.len()];
        (self.row_offsets.clone(), self.col_indices.clone(), values)
    }
}

impl std::fmt::Display for AdjacencyList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let num_edges = match self.graph_type {
            GraphType::Undirected => self.col_indices.len() / 2,
            GraphType::Directed => self.col_indices.len(),
        };
        write!(
            f,
            "AdjacencyList(V={}, E={}, {})",
            self.num_vertices, num_edges, self.graph_type
        )
    }
}

// ---------------------------------------------------------------------------
// Helper: approximate connectivity check
// ---------------------------------------------------------------------------

/// Returns `true` if a BFS from vertex 0 reaches more than 90 % of the
/// vertices (an approximation of full connectivity).
pub fn is_approximately_connected(adj: &AdjacencyList) -> bool {
    if adj.num_vertices <= 1 {
        return true;
    }

    let mut visited = vec![false; adj.num_vertices];
    let mut queue = VecDeque::new();
    visited[0] = true;
    queue.push_back(0usize);
    let mut count: usize = 1;

    while let Some(v) = queue.pop_front() {
        let start = adj.row_offsets[v];
        let end = adj.row_offsets[v + 1];
        for &nbr in &adj.col_indices[start..end] {
            if !visited[nbr] {
                visited[nbr] = true;
                count += 1;
                queue.push_back(nbr);
            }
        }
    }

    let threshold = (adj.num_vertices as f64 * 0.9).ceil() as usize;
    count >= threshold
}

// ---------------------------------------------------------------------------
// Helper: build AdjacencyList from edge list
// ---------------------------------------------------------------------------

/// Builds an `AdjacencyList` from a list of directed edges.
///
/// For undirected graphs the caller must include both `(i, j)` and `(j, i)`.
fn adjacency_from_edges(
    num_vertices: usize,
    edges: &mut [(usize, usize)],
    graph_type: GraphType,
) -> AdjacencyList {
    // Sort by source vertex for CSR construction.
    edges.sort_unstable_by_key(|&(src, _)| src);

    let mut row_offsets = vec![0usize; num_vertices + 1];
    for &(src, _) in edges.iter() {
        row_offsets[src + 1] += 1;
    }
    for i in 1..=num_vertices {
        row_offsets[i] += row_offsets[i - 1];
    }

    let col_indices: Vec<usize> = edges.iter().map(|&(_, dst)| dst).collect();

    AdjacencyList {
        num_vertices,
        row_offsets,
        col_indices,
        graph_type,
    }
}

// ---------------------------------------------------------------------------
// ErdosRenyiGenerator
// ---------------------------------------------------------------------------

/// Generates random graphs according to the Erdős–Rényi G(n, p) model.
///
/// Each of the `n*(n-1)/2` possible undirected edges (or `n*(n-1)` directed
/// edges) is independently included with probability `p`.
pub struct ErdosRenyiGenerator {
    num_vertices: usize,
    edge_probability: f64,
    seed: u64,
}

impl ErdosRenyiGenerator {
    /// Creates a new Erdős–Rényi generator.
    ///
    /// # Arguments
    ///
    /// * `num_vertices` — number of vertices n
    /// * `edge_probability` — probability p that any given edge exists
    /// * `seed` — PRNG seed
    pub fn new(num_vertices: usize, edge_probability: f64, seed: u64) -> Self {
        Self {
            num_vertices,
            edge_probability,
            seed,
        }
    }

    /// Generates an undirected G(n, p) random graph.
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if `edge_probability` is not in [0, 1].
    pub fn generate(&self) -> RandResult<AdjacencyList> {
        self.validate_params()?;
        let mut rng = SplitMix64::new(self.seed);
        let n = self.num_vertices;
        let mut edges: Vec<(usize, usize)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                if rng.next_f64() < self.edge_probability {
                    edges.push((i, j));
                    edges.push((j, i));
                }
            }
        }

        Ok(adjacency_from_edges(n, &mut edges, GraphType::Undirected))
    }

    /// Generates a directed G(n, p) random graph.
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if `edge_probability` is not in [0, 1].
    pub fn generate_directed(&self) -> RandResult<AdjacencyList> {
        self.validate_params()?;
        let mut rng = SplitMix64::new(self.seed);
        let n = self.num_vertices;
        let mut edges: Vec<(usize, usize)> = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j && rng.next_f64() < self.edge_probability {
                    edges.push((i, j));
                }
            }
        }

        Ok(adjacency_from_edges(n, &mut edges, GraphType::Directed))
    }

    /// Validates generator parameters.
    fn validate_params(&self) -> RandResult<()> {
        if !(0.0..=1.0).contains(&self.edge_probability) {
            return Err(RandError::InvalidSize(format!(
                "edge_probability must be in [0, 1], got {}",
                self.edge_probability
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// StochasticBlockModelGenerator
// ---------------------------------------------------------------------------

/// Generates random graphs with community structure according to the
/// Stochastic Block Model.
///
/// Vertices are partitioned into blocks (communities). Within the same block,
/// edges appear with probability `intra_prob`; between different blocks, with
/// probability `inter_prob`.
pub struct StochasticBlockModelGenerator {
    block_sizes: Vec<usize>,
    intra_prob: f64,
    inter_prob: f64,
    seed: u64,
}

impl StochasticBlockModelGenerator {
    /// Creates a new SBM generator.
    ///
    /// # Arguments
    ///
    /// * `block_sizes` — sizes of each community
    /// * `intra_prob` — edge probability within communities
    /// * `inter_prob` — edge probability between communities
    /// * `seed` — PRNG seed
    pub fn new(block_sizes: Vec<usize>, intra_prob: f64, inter_prob: f64, seed: u64) -> Self {
        Self {
            block_sizes,
            intra_prob,
            inter_prob,
            seed,
        }
    }

    /// Generates an undirected SBM graph.
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if probabilities are outside [0, 1] or
    /// `block_sizes` is empty.
    pub fn generate(&self) -> RandResult<AdjacencyList> {
        if self.block_sizes.is_empty() {
            return Err(RandError::InvalidSize(
                "block_sizes must not be empty".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.intra_prob) {
            return Err(RandError::InvalidSize(format!(
                "intra_prob must be in [0, 1], got {}",
                self.intra_prob
            )));
        }
        if !(0.0..=1.0).contains(&self.inter_prob) {
            return Err(RandError::InvalidSize(format!(
                "inter_prob must be in [0, 1], got {}",
                self.inter_prob
            )));
        }

        let n: usize = self.block_sizes.iter().sum();
        let mut rng = SplitMix64::new(self.seed);

        // Build block assignments: block_of[v] = block index for vertex v
        let mut block_of = Vec::with_capacity(n);
        for (block_idx, &size) in self.block_sizes.iter().enumerate() {
            for _ in 0..size {
                block_of.push(block_idx);
            }
        }

        let mut edges: Vec<(usize, usize)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let prob = if block_of[i] == block_of[j] {
                    self.intra_prob
                } else {
                    self.inter_prob
                };
                if rng.next_f64() < prob {
                    edges.push((i, j));
                    edges.push((j, i));
                }
            }
        }

        Ok(adjacency_from_edges(n, &mut edges, GraphType::Undirected))
    }
}

// ---------------------------------------------------------------------------
// BarabasiAlbertGenerator
// ---------------------------------------------------------------------------

/// Generates scale-free random graphs via the Barabási–Albert preferential
/// attachment model.
///
/// Starting from a small complete graph of `m₀ = attachment_edges + 1`
/// vertices, each new vertex connects to `attachment_edges` existing vertices
/// with probability proportional to their current degree.
pub struct BarabasiAlbertGenerator {
    num_vertices: usize,
    attachment_edges: usize,
    seed: u64,
}

impl BarabasiAlbertGenerator {
    /// Creates a new Barabási–Albert generator.
    ///
    /// # Arguments
    ///
    /// * `num_vertices` — total number of vertices in the final graph
    /// * `attachment_edges` — number of edges each new vertex adds (m)
    /// * `seed` — PRNG seed
    pub fn new(num_vertices: usize, attachment_edges: usize, seed: u64) -> Self {
        Self {
            num_vertices,
            attachment_edges,
            seed,
        }
    }

    /// Generates an undirected Barabási–Albert graph.
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if `attachment_edges` is zero,
    /// `num_vertices <= attachment_edges`, or any other inconsistency.
    pub fn generate(&self) -> RandResult<AdjacencyList> {
        let m = self.attachment_edges;
        let n = self.num_vertices;

        if m == 0 {
            return Err(RandError::InvalidSize(
                "attachment_edges must be >= 1".to_string(),
            ));
        }
        let m0 = m + 1; // initial complete graph size
        if n <= m {
            return Err(RandError::InvalidSize(format!(
                "num_vertices ({n}) must be > attachment_edges ({m})"
            )));
        }

        let mut rng = SplitMix64::new(self.seed);

        // Degree array (used for preferential attachment weights).
        let mut degree = vec![0usize; n];
        let mut edges: Vec<(usize, usize)> = Vec::new();

        // Initial complete graph on vertices [0, m0).
        for i in 0..m0 {
            for j in (i + 1)..m0 {
                edges.push((i, j));
                edges.push((j, i));
                degree[i] += 1;
                degree[j] += 1;
            }
        }

        // Preferential attachment: add vertices m0..n one at a time.
        // We use a repeated-entries array for weighted sampling (fast and simple).
        // `stubs[k]` = some vertex id; vertex v appears `degree[v]` times.
        let mut stubs: Vec<usize> = Vec::new();
        for (v, &d) in degree.iter().enumerate().take(m0) {
            for _ in 0..d {
                stubs.push(v);
            }
        }

        for new_v in m0..n {
            // Select m distinct targets proportional to degree.
            let mut targets: Vec<usize> = Vec::with_capacity(m);
            let mut target_set = vec![false; n]; // avoid duplicates

            let mut attempts = 0usize;
            while targets.len() < m && attempts < m * 100 {
                attempts += 1;
                if stubs.is_empty() {
                    break;
                }
                let idx = rng.next_usize(stubs.len());
                let target = stubs[idx];
                if !target_set[target] && target != new_v {
                    target_set[target] = true;
                    targets.push(target);
                }
            }

            // Add edges from new_v to each target.
            for &t in &targets {
                edges.push((new_v, t));
                edges.push((t, new_v));
                degree[new_v] += 1;
                degree[t] += 1;
                stubs.push(new_v);
                stubs.push(t);
            }
        }

        Ok(adjacency_from_edges(n, &mut edges, GraphType::Undirected))
    }
}

// ---------------------------------------------------------------------------
// WattsStrogatzGenerator
// ---------------------------------------------------------------------------

/// Generates small-world random graphs via the Watts–Strogatz model.
///
/// Starts from a ring lattice where each vertex is connected to its
/// `k_neighbors` nearest neighbours (k/2 on each side), then rewires each
/// edge with probability `rewire_prob`.
pub struct WattsStrogatzGenerator {
    num_vertices: usize,
    k_neighbors: usize,
    rewire_prob: f64,
    seed: u64,
}

impl WattsStrogatzGenerator {
    /// Creates a new Watts–Strogatz generator.
    ///
    /// # Arguments
    ///
    /// * `num_vertices` — number of vertices
    /// * `k_neighbors` — each vertex is connected to its k nearest neighbours
    ///   in the ring (must be even and < n)
    /// * `rewire_prob` — probability of rewiring each edge
    /// * `seed` — PRNG seed
    pub fn new(num_vertices: usize, k_neighbors: usize, rewire_prob: f64, seed: u64) -> Self {
        Self {
            num_vertices,
            k_neighbors,
            rewire_prob,
            seed,
        }
    }

    /// Generates an undirected Watts–Strogatz small-world graph.
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if parameters are inconsistent.
    pub fn generate(&self) -> RandResult<AdjacencyList> {
        let n = self.num_vertices;
        let k = self.k_neighbors;

        if n == 0 {
            return Ok(AdjacencyList {
                num_vertices: 0,
                row_offsets: vec![0],
                col_indices: Vec::new(),
                graph_type: GraphType::Undirected,
            });
        }
        if k == 0 {
            // No edges
            return Ok(AdjacencyList {
                num_vertices: n,
                row_offsets: vec![0; n + 1],
                col_indices: Vec::new(),
                graph_type: GraphType::Undirected,
            });
        }
        if k % 2 != 0 {
            return Err(RandError::InvalidSize(format!(
                "k_neighbors must be even, got {k}"
            )));
        }
        if k >= n {
            return Err(RandError::InvalidSize(format!(
                "k_neighbors ({k}) must be < num_vertices ({n})"
            )));
        }
        if !(0.0..=1.0).contains(&self.rewire_prob) {
            return Err(RandError::InvalidSize(format!(
                "rewire_prob must be in [0, 1], got {}",
                self.rewire_prob
            )));
        }

        let mut rng = SplitMix64::new(self.seed);
        let half_k = k / 2;

        // Build adjacency as a set of edges per vertex for efficient rewiring.
        // Use Vec<Vec<usize>> as a simple adjacency set.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Ring lattice: connect each vertex to its k/2 nearest on each side.
        for i in 0..n {
            for offset in 1..=half_k {
                let j = (i + offset) % n;
                adj[i].push(j);
                adj[j].push(i);
            }
        }

        // Rewire: for each vertex i and each of its right-side neighbours,
        // with probability beta, replace (i, j) with (i, j') where j' is
        // a uniformly random vertex != i and not already a neighbour.
        for i in 0..n {
            for offset in 1..=half_k {
                if rng.next_f64() >= self.rewire_prob {
                    continue;
                }

                let j = (i + offset) % n;

                // Remove edge (i, j) and (j, i)
                if let Some(pos) = adj[i].iter().position(|&x| x == j) {
                    adj[i].swap_remove(pos);
                }
                if let Some(pos) = adj[j].iter().position(|&x| x == i) {
                    adj[j].swap_remove(pos);
                }

                // Pick a new target j' != i, not already a neighbour of i
                let mut attempts = 0usize;
                let mut new_target = j;
                let found = loop {
                    attempts += 1;
                    if attempts > n * 10 {
                        break false;
                    }
                    let candidate = rng.next_usize(n);
                    if candidate != i && !adj[i].contains(&candidate) {
                        new_target = candidate;
                        break true;
                    }
                };

                if found {
                    adj[i].push(new_target);
                    adj[new_target].push(i);
                } else {
                    // Couldn't find a valid target; re-add original edge.
                    adj[i].push(j);
                    adj[j].push(i);
                }
            }
        }

        // Convert adjacency lists to CSR.
        Ok(adjacency_vecs_to_csr(n, &adj, GraphType::Undirected))
    }
}

// ---------------------------------------------------------------------------
// RandomRegularGenerator
// ---------------------------------------------------------------------------

/// Generates random k-regular graphs where every vertex has the same degree.
///
/// Uses the pairing (configuration) model: create `k` stubs for each vertex,
/// then randomly match stubs. Self-loops and multi-edges are discarded and
/// re-attempted.
pub struct RandomRegularGenerator {
    num_vertices: usize,
    degree: usize,
    seed: u64,
}

impl RandomRegularGenerator {
    /// Creates a new random regular graph generator.
    ///
    /// # Arguments
    ///
    /// * `num_vertices` — number of vertices
    /// * `degree` — the common degree k (n*k must be even)
    /// * `seed` — PRNG seed
    pub fn new(num_vertices: usize, degree: usize, seed: u64) -> Self {
        Self {
            num_vertices,
            degree,
            seed,
        }
    }

    /// Generates a random k-regular undirected graph.
    ///
    /// # Errors
    ///
    /// Returns `RandError::InvalidSize` if `num_vertices * degree` is odd,
    /// or `degree >= num_vertices`.
    pub fn generate(&self) -> RandResult<AdjacencyList> {
        let n = self.num_vertices;
        let k = self.degree;

        if n == 0 {
            return Ok(AdjacencyList {
                num_vertices: 0,
                row_offsets: vec![0],
                col_indices: Vec::new(),
                graph_type: GraphType::Undirected,
            });
        }
        if k == 0 {
            return Ok(AdjacencyList {
                num_vertices: n,
                row_offsets: vec![0; n + 1],
                col_indices: Vec::new(),
                graph_type: GraphType::Undirected,
            });
        }
        if (n * k) % 2 != 0 {
            return Err(RandError::InvalidSize(format!(
                "num_vertices * degree ({} * {}) must be even",
                n, k
            )));
        }
        if k >= n {
            return Err(RandError::InvalidSize(format!(
                "degree ({k}) must be < num_vertices ({n})"
            )));
        }

        // Pairing model with retries.
        let mut rng = SplitMix64::new(self.seed);
        let max_retries = 100;

        for _ in 0..max_retries {
            match self.try_pairing(&mut rng, n, k) {
                Some(adj) => return Ok(adj),
                None => continue,
            }
        }

        // Fallback: return best-effort graph from last attempt (very unlikely
        // to reach here for reasonable parameters).
        Err(RandError::InternalError(format!(
            "failed to generate {k}-regular graph on {n} vertices after {max_retries} attempts"
        )))
    }

    /// Attempts a single pairing-model generation. Returns `None` if the
    /// attempt produced too many self-loops or multi-edges and should be
    /// retried.
    fn try_pairing(&self, rng: &mut SplitMix64, n: usize, k: usize) -> Option<AdjacencyList> {
        // Create stub list: vertex v appears k times.
        let total_stubs = n * k;
        let mut stubs: Vec<usize> = Vec::with_capacity(total_stubs);
        for v in 0..n {
            for _ in 0..k {
                stubs.push(v);
            }
        }

        // Fisher-Yates shuffle.
        for i in (1..total_stubs).rev() {
            let j = rng.next_usize(i + 1);
            stubs.swap(i, j);
        }

        // Pair up consecutive stubs.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut valid = true;
        let num_pairs = total_stubs / 2;

        for p in 0..num_pairs {
            let u = stubs[2 * p];
            let v = stubs[2 * p + 1];
            if u == v {
                // Self-loop — invalid.
                valid = false;
                break;
            }
            if adj[u].contains(&v) {
                // Multi-edge — invalid.
                valid = false;
                break;
            }
            adj[u].push(v);
            adj[v].push(u);
        }

        if !valid {
            return None;
        }

        Some(adjacency_vecs_to_csr(n, &adj, GraphType::Undirected))
    }
}

// ---------------------------------------------------------------------------
// Helper: convert Vec<Vec<usize>> adjacency to AdjacencyList
// ---------------------------------------------------------------------------

/// Converts per-vertex adjacency vectors into CSR `AdjacencyList`.
fn adjacency_vecs_to_csr(
    num_vertices: usize,
    adj: &[Vec<usize>],
    graph_type: GraphType,
) -> AdjacencyList {
    let mut row_offsets = Vec::with_capacity(num_vertices + 1);
    let mut col_indices = Vec::new();

    let mut offset = 0usize;
    for neighbours in adj.iter() {
        row_offsets.push(offset);
        let mut sorted = neighbours.clone();
        sorted.sort_unstable();
        col_indices.extend_from_slice(&sorted);
        offset += sorted.len();
    }
    row_offsets.push(offset);

    AdjacencyList {
        num_vertices,
        row_offsets,
        col_indices,
        graph_type,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // Erdos-Renyi tests
    // -------------------------------------------------------------------

    #[test]
    fn erdos_renyi_basic_edge_count() {
        let n = 200;
        let p = 0.1;
        let er = ErdosRenyiGenerator::new(n, p, 42);
        let g = er.generate().expect("ER generate failed");

        let stats = g.stats();
        let expected = (n * (n - 1)) as f64 / 2.0 * p;
        // Allow generous tolerance for random graphs.
        let tolerance = expected * 0.3 + 50.0;
        assert!(
            (stats.num_edges as f64 - expected).abs() < tolerance,
            "expected ~{expected:.0} edges, got {}",
            stats.num_edges
        );
    }

    #[test]
    fn erdos_renyi_directed() {
        let n = 50;
        let p = 0.2;
        let er = ErdosRenyiGenerator::new(n, p, 99);
        let g = er.generate_directed().expect("ER directed failed");

        assert_eq!(g.graph_type, GraphType::Directed);
        let stats = g.stats();
        let expected = (n * (n - 1)) as f64 * p;
        let tolerance = expected * 0.35 + 30.0;
        assert!(
            (stats.num_edges as f64 - expected).abs() < tolerance,
            "expected ~{expected:.0} directed edges, got {}",
            stats.num_edges
        );
    }

    #[test]
    fn erdos_renyi_p_zero_no_edges() {
        let gr = ErdosRenyiGenerator::new(50, 0.0, 1);
        let g = gr.generate().expect("ER p=0 failed");
        assert_eq!(g.col_indices.len(), 0);
        assert_eq!(g.stats().num_edges, 0);
    }

    #[test]
    fn erdos_renyi_p_one_complete() {
        let n = 20;
        let gr = ErdosRenyiGenerator::new(n, 1.0, 7);
        let g = gr.generate().expect("ER p=1 failed");
        let expected_edges = n * (n - 1) / 2;
        assert_eq!(g.stats().num_edges, expected_edges);
    }

    // -------------------------------------------------------------------
    // Stochastic Block Model tests
    // -------------------------------------------------------------------

    #[test]
    fn sbm_community_structure() {
        // Two blocks of 50 vertices each: intra=0.5, inter=0.05.
        let gr = StochasticBlockModelGenerator::new(vec![50, 50], 0.5, 0.05, 42);
        let g = gr.generate().expect("SBM generate failed");

        // Count intra vs inter edges.
        let mut intra_edges = 0usize;
        let mut inter_edges = 0usize;
        for v in 0..100 {
            let block_v = if v < 50 { 0 } else { 1 };
            let nbrs = g.neighbors(v).expect("neighbors failed");
            for &u in nbrs {
                let block_u = if u < 50 { 0 } else { 1 };
                if block_v == block_u {
                    intra_edges += 1;
                } else {
                    inter_edges += 1;
                }
            }
        }
        // Each undirected edge counted twice in adjacency.
        assert!(
            intra_edges > inter_edges,
            "intra_edges ({intra_edges}) should be > inter_edges ({inter_edges})"
        );
    }

    #[test]
    fn sbm_equal_probs_like_er() {
        // When intra_prob == inter_prob, SBM is equivalent to ER.
        let p = 0.15;
        let gr = StochasticBlockModelGenerator::new(vec![30, 30], p, p, 123);
        let g = gr.generate().expect("SBM equal probs failed");
        let stats = g.stats();
        let n = 60usize;
        let expected = (n * (n - 1)) as f64 / 2.0 * p;
        let tolerance = expected * 0.4 + 30.0;
        assert!(
            (stats.num_edges as f64 - expected).abs() < tolerance,
            "expected ~{expected:.0} edges, got {}",
            stats.num_edges
        );
    }

    // -------------------------------------------------------------------
    // Barabasi-Albert tests
    // -------------------------------------------------------------------

    #[test]
    fn barabasi_albert_degree_distribution() {
        let n = 500;
        let m = 3;
        let gr = BarabasiAlbertGenerator::new(n, m, 42);
        let g = gr.generate().expect("BA generate failed");

        // Verify power-law-ish: there should be a few high-degree hubs.
        let stats = g.stats();
        assert!(
            stats.max_degree > stats.avg_degree as usize * 2,
            "max_degree ({}) should be well above avg_degree ({:.1})",
            stats.max_degree,
            stats.avg_degree
        );
    }

    #[test]
    fn barabasi_albert_minimum_degree() {
        let n = 100;
        let m = 2;
        let gr = BarabasiAlbertGenerator::new(n, m, 77);
        let g = gr.generate().expect("BA generate failed");

        // Every vertex added after the initial clique should have degree >= m.
        let stats = g.stats();
        assert!(
            stats.min_degree >= m,
            "min_degree ({}) should be >= m ({})",
            stats.min_degree,
            m
        );
    }

    // -------------------------------------------------------------------
    // Watts-Strogatz tests
    // -------------------------------------------------------------------

    #[test]
    fn watts_strogatz_ring_p_zero() {
        let n = 20;
        let k = 4;
        let gr = WattsStrogatzGenerator::new(n, k, 0.0, 42);
        let g = gr.generate().expect("WS p=0 failed");

        // Every vertex should have degree exactly k.
        for v in 0..n {
            let d = g.degree(v).expect("degree failed");
            assert_eq!(d, k, "vertex {v} degree = {d}, expected {k}");
        }
    }

    #[test]
    fn watts_strogatz_fully_random() {
        let n = 50;
        let k = 6;
        let gr = WattsStrogatzGenerator::new(n, k, 1.0, 99);
        let g = gr.generate().expect("WS p=1 failed");

        // Total edges should be approximately n*k/2 (some may fail to rewire).
        let stats = g.stats();
        let expected = n * k / 2;
        let tolerance = (expected as f64 * 0.15) as usize + 5;
        assert!(
            (stats.num_edges as isize - expected as isize).unsigned_abs() < tolerance,
            "expected ~{expected} edges, got {}",
            stats.num_edges
        );
    }

    // -------------------------------------------------------------------
    // Random Regular tests
    // -------------------------------------------------------------------

    #[test]
    fn random_regular_degree_check() {
        let n = 50;
        let k = 4;
        let gr = RandomRegularGenerator::new(n, k, 42);
        let g = gr.generate().expect("regular graph failed");

        for v in 0..n {
            let d = g.degree(v).expect("degree failed");
            assert_eq!(d, k, "vertex {v} has degree {d}, expected {k}");
        }
    }

    // -------------------------------------------------------------------
    // GraphStats / AdjacencyList tests
    // -------------------------------------------------------------------

    #[test]
    fn graph_stats_computation() {
        let gr = ErdosRenyiGenerator::new(100, 0.15, 55);
        let g = gr.generate().expect("ER failed");
        let stats = g.stats();
        assert_eq!(stats.num_vertices, 100);
        assert!(stats.avg_degree > 0.0);
        assert!(stats.max_degree >= stats.min_degree);
        assert!(stats.density >= 0.0 && stats.density <= 1.0);
    }

    #[test]
    fn adjacency_list_degree_query() {
        let gr = ErdosRenyiGenerator::new(10, 1.0, 42);
        let g = gr.generate().expect("ER failed");
        // Complete graph: every vertex has degree n-1.
        for v in 0..10 {
            assert_eq!(
                g.degree(v).expect("degree failed"),
                9,
                "vertex {v} should have degree 9 in K_10"
            );
        }
        // Out-of-range vertex.
        assert!(g.degree(10).is_err());
    }

    #[test]
    fn adjacency_list_neighbors_query() {
        let gr = ErdosRenyiGenerator::new(5, 1.0, 42);
        let g = gr.generate().expect("ER failed");

        // Vertex 0 in K_5 should have neighbours {1, 2, 3, 4}.
        let nbrs = g.neighbors(0).expect("neighbors failed");
        assert_eq!(nbrs.len(), 4);
        for &expected in &[1usize, 2, 3, 4] {
            assert!(nbrs.contains(&expected), "missing neighbour {expected}");
        }
        // Out-of-range.
        assert!(g.neighbors(5).is_err());
    }

    #[test]
    fn adjacency_list_to_csr_values() {
        let gr = ErdosRenyiGenerator::new(5, 1.0, 42);
        let g = gr.generate().expect("ER failed");

        let (row_off, col_idx, vals) = g.to_csr_values(1.0_f64);
        assert_eq!(row_off.len(), 6);
        assert_eq!(col_idx.len(), vals.len());
        // All edge weights should be 1.0
        for &v in &vals {
            assert!((v - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn connectivity_check() {
        // Dense ER graph should be connected.
        let gr = ErdosRenyiGenerator::new(100, 0.1, 42);
        let g = gr.generate().expect("ER failed");
        assert!(
            is_approximately_connected(&g),
            "dense ER graph should be approximately connected"
        );

        // Sparse graph with p=0 is not connected (unless trivial).
        let gen2 = ErdosRenyiGenerator::new(50, 0.0, 1);
        let g2 = gen2.generate().expect("ER p=0 failed");
        assert!(
            !is_approximately_connected(&g2),
            "graph with no edges should not be connected"
        );
    }

    #[test]
    fn empty_graph() {
        let gr = ErdosRenyiGenerator::new(0, 0.5, 42);
        let g = gr.generate().expect("empty graph failed");
        assert_eq!(g.num_vertices, 0);
        assert_eq!(g.col_indices.len(), 0);
        let stats = g.stats();
        assert_eq!(stats.num_edges, 0);
        assert!(stats.is_connected);
    }

    #[test]
    fn single_vertex_graph() {
        let gr = ErdosRenyiGenerator::new(1, 0.5, 42);
        let g = gr.generate().expect("single vertex failed");
        assert_eq!(g.num_vertices, 1);
        assert_eq!(g.col_indices.len(), 0);
        assert_eq!(g.degree(0).expect("degree failed"), 0);
        assert!(is_approximately_connected(&g));
    }
}
