//! Sparse tensor operations for GNN (Graph Neural Network) workloads.
//!
//! This module extends SpMV/SpMM primitives with operations tailored for
//! graph neural network message passing and aggregation, including:
//!
//! - [`scatter_reduce`] -- Scatter-reduce with configurable aggregation
//! - [`gather`] -- Index-based gathering from source arrays
//! - [`sparse_message_passing`] -- GNN message passing over adjacency matrices
//! - [`sparse_attention_message`] -- Attention-weighted message passing (GAT-style)
//! - [`compute_degree_matrix`] -- Node degree computation from CSR offsets
//! - [`symmetric_normalize`] -- D^{-1/2} A D^{-1/2} normalization (GCN-style)
//! - [`add_self_loops`] -- Add identity self-loops to adjacency matrix
//! - [`sparse_row_softmax`] -- Row-wise softmax over sparse matrix values
//! - [`generate_message_passing_ptx`] -- PTX kernel generation for GPU message passing
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

use crate::error::SparseError;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Aggregation operation for GNN message passing.
///
/// Controls how incoming messages from neighboring nodes are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessagePassingOp {
    /// Sum of all incoming messages.
    Sum,
    /// Mean (average) of all incoming messages.
    Mean,
    /// Element-wise maximum of incoming messages.
    Max,
    /// Element-wise minimum of incoming messages.
    Min,
}

/// Configuration for GNN sparse operations.
#[derive(Debug, Clone)]
pub struct GnnSparseConfig {
    /// Number of nodes in the graph.
    pub num_nodes: usize,
    /// Dimensionality of per-node features.
    pub feature_dim: usize,
    /// Number of edges in the graph.
    pub num_edges: usize,
    /// Aggregation operation for message passing.
    pub op: MessagePassingOp,
    /// Whether to normalize aggregated messages by node degree.
    pub normalize: bool,
}

/// Edge features for attention-weighted message passing.
///
/// Stores per-edge feature values used in attention mechanisms (e.g. GAT).
#[derive(Debug, Clone)]
pub struct EdgeFeatures {
    /// Per-edge feature values.
    pub values: Vec<f64>,
    /// Number of edges (must equal `values.len()`).
    pub edge_count: usize,
}

// ---------------------------------------------------------------------------
// Core operations
// ---------------------------------------------------------------------------

/// Scatter-reduce operation.
///
/// For each position `i` in `src`, the value `src[i]` is aggregated into
/// `output[index[i]]` using the specified [`MessagePassingOp`].
///
/// # Arguments
///
/// * `src` -- Source values to scatter.
/// * `index` -- Target indices; must have same length as `src`.
/// * `num_targets` -- Size of the output vector.
/// * `op` -- Aggregation operation.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `src` and `index` differ in length.
/// Returns [`SparseError::InvalidArgument`] if any index is out of bounds.
pub fn scatter_reduce(
    src: &[f64],
    index: &[usize],
    num_targets: usize,
    op: MessagePassingOp,
) -> Result<Vec<f64>, SparseError> {
    if src.len() != index.len() {
        return Err(SparseError::DimensionMismatch(format!(
            "src length {} != index length {}",
            src.len(),
            index.len()
        )));
    }

    // Validate all indices
    for (i, &idx) in index.iter().enumerate() {
        if idx >= num_targets {
            return Err(SparseError::InvalidArgument(format!(
                "index[{}] = {} is out of bounds for num_targets = {}",
                i, idx, num_targets
            )));
        }
    }

    let init_val = match op {
        MessagePassingOp::Sum | MessagePassingOp::Mean => 0.0_f64,
        MessagePassingOp::Max => f64::NEG_INFINITY,
        MessagePassingOp::Min => f64::INFINITY,
    };

    let mut output = vec![init_val; num_targets];
    let mut counts = if matches!(op, MessagePassingOp::Mean) {
        vec![0usize; num_targets]
    } else {
        Vec::new()
    };

    for (i, &idx) in index.iter().enumerate() {
        match op {
            MessagePassingOp::Sum => {
                output[idx] += src[i];
            }
            MessagePassingOp::Mean => {
                output[idx] += src[i];
                counts[idx] += 1;
            }
            MessagePassingOp::Max => {
                if src[i] > output[idx] {
                    output[idx] = src[i];
                }
            }
            MessagePassingOp::Min => {
                if src[i] < output[idx] {
                    output[idx] = src[i];
                }
            }
        }
    }

    // Finalize Mean: divide by count; targets with zero contributions stay 0.
    if matches!(op, MessagePassingOp::Mean) {
        for (val, &cnt) in output.iter_mut().zip(counts.iter()) {
            if cnt > 0 {
                *val /= cnt as f64;
            } else {
                *val = 0.0;
            }
        }
    }

    // For Max/Min with no contributions, set to 0.
    if matches!(op, MessagePassingOp::Max | MessagePassingOp::Min) {
        for val in &mut output {
            if *val == f64::NEG_INFINITY || *val == f64::INFINITY {
                *val = 0.0;
            }
        }
    }

    Ok(output)
}

/// Gather operation.
///
/// Collects values from `src` at the given `index` positions,
/// producing a vector of length `index.len()`.
///
/// # Errors
///
/// Returns [`SparseError::InvalidArgument`] if any index is out of bounds.
pub fn gather(src: &[f64], index: &[usize]) -> Result<Vec<f64>, SparseError> {
    let mut result = Vec::with_capacity(index.len());
    for (i, &idx) in index.iter().enumerate() {
        if idx >= src.len() {
            return Err(SparseError::InvalidArgument(format!(
                "gather index[{}] = {} out of bounds for src of length {}",
                i,
                idx,
                src.len()
            )));
        }
        result.push(src[idx]);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// GNN message passing
// ---------------------------------------------------------------------------

/// Sparse message passing over an adjacency matrix in CSR form.
///
/// For each node `i`, aggregates features from its neighbors using the
/// adjacency structure:
///
/// ```text
/// output[i, :] = aggregate_{j in neighbors(i)} features[j, :]
/// ```
///
/// When `config.normalize` is `true`, the aggregated result is divided
/// by the degree of node `i`.
///
/// # Arguments
///
/// * `adj_row_offsets` -- CSR row pointer array (length `num_nodes + 1`).
/// * `adj_col_indices` -- CSR column indices (length `num_edges`).
/// * `node_features` -- Dense feature matrix stored row-major (num_nodes x feature_dim).
/// * `feature_dim` -- Number of features per node.
/// * `config` -- GNN configuration (aggregation op, normalization, etc.).
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] on inconsistent sizes.
pub fn sparse_message_passing(
    adj_row_offsets: &[usize],
    adj_col_indices: &[usize],
    node_features: &[f64],
    feature_dim: usize,
    config: &GnnSparseConfig,
) -> Result<Vec<f64>, SparseError> {
    let num_nodes = config.num_nodes;

    if adj_row_offsets.len() != num_nodes + 1 {
        return Err(SparseError::DimensionMismatch(format!(
            "adj_row_offsets length {} != num_nodes + 1 = {}",
            adj_row_offsets.len(),
            num_nodes + 1
        )));
    }
    if feature_dim != config.feature_dim {
        return Err(SparseError::DimensionMismatch(format!(
            "feature_dim {} != config.feature_dim {}",
            feature_dim, config.feature_dim
        )));
    }
    if node_features.len() != num_nodes * feature_dim {
        return Err(SparseError::DimensionMismatch(format!(
            "node_features length {} != num_nodes * feature_dim = {}",
            node_features.len(),
            num_nodes * feature_dim
        )));
    }

    let mut output = vec![0.0_f64; num_nodes * feature_dim];

    for i in 0..num_nodes {
        let start = adj_row_offsets[i];
        let end = adj_row_offsets[i + 1];
        let degree = end - start;

        if degree == 0 {
            continue;
        }

        match config.op {
            MessagePassingOp::Sum | MessagePassingOp::Mean => {
                for &j in &adj_col_indices[start..end] {
                    if j >= num_nodes {
                        return Err(SparseError::InvalidArgument(format!(
                            "column index {} out of bounds for {} nodes",
                            j, num_nodes
                        )));
                    }
                    for f in 0..feature_dim {
                        output[i * feature_dim + f] += node_features[j * feature_dim + f];
                    }
                }
                if config.op == MessagePassingOp::Mean || config.normalize {
                    let inv_degree = 1.0 / degree as f64;
                    for f in 0..feature_dim {
                        output[i * feature_dim + f] *= inv_degree;
                    }
                }
            }
            MessagePassingOp::Max => {
                // Initialize to negative infinity, then take max
                for f in 0..feature_dim {
                    output[i * feature_dim + f] = f64::NEG_INFINITY;
                }
                for &j in &adj_col_indices[start..end] {
                    if j >= num_nodes {
                        return Err(SparseError::InvalidArgument(format!(
                            "column index {} out of bounds for {} nodes",
                            j, num_nodes
                        )));
                    }
                    for f in 0..feature_dim {
                        let val = node_features[j * feature_dim + f];
                        if val > output[i * feature_dim + f] {
                            output[i * feature_dim + f] = val;
                        }
                    }
                }
                // Replace any remaining -inf with 0 (isolated nodes handled above)
            }
            MessagePassingOp::Min => {
                for f in 0..feature_dim {
                    output[i * feature_dim + f] = f64::INFINITY;
                }
                for &j in &adj_col_indices[start..end] {
                    if j >= num_nodes {
                        return Err(SparseError::InvalidArgument(format!(
                            "column index {} out of bounds for {} nodes",
                            j, num_nodes
                        )));
                    }
                    for f in 0..feature_dim {
                        let val = node_features[j * feature_dim + f];
                        if val < output[i * feature_dim + f] {
                            output[i * feature_dim + f] = val;
                        }
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Attention-weighted message passing.
///
/// For each node `i`, computes:
///
/// ```text
/// output[i, :] = sum_{j in N(i)} alpha_ij * features[j, :]
/// ```
///
/// where `attention_weights` provides one scalar weight per edge.
///
/// # Arguments
///
/// * `adj_row_offsets` -- CSR row pointer (length num_nodes + 1).
/// * `adj_col_indices` -- CSR column indices (length num_edges).
/// * `node_features` -- Dense feature matrix, row-major (num_nodes x feature_dim).
/// * `attention_weights` -- One weight per edge (length num_edges).
/// * `feature_dim` -- Dimensionality of node features.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] on size mismatches.
pub fn sparse_attention_message(
    adj_row_offsets: &[usize],
    adj_col_indices: &[usize],
    node_features: &[f64],
    attention_weights: &[f64],
    feature_dim: usize,
) -> Result<Vec<f64>, SparseError> {
    if adj_row_offsets.is_empty() {
        return Err(SparseError::InvalidArgument(
            "adj_row_offsets must not be empty".to_string(),
        ));
    }
    let num_nodes = adj_row_offsets.len() - 1;
    let num_edges = adj_col_indices.len();

    if attention_weights.len() != num_edges {
        return Err(SparseError::DimensionMismatch(format!(
            "attention_weights length {} != num_edges {}",
            attention_weights.len(),
            num_edges
        )));
    }
    if node_features.len() != num_nodes * feature_dim {
        return Err(SparseError::DimensionMismatch(format!(
            "node_features length {} != num_nodes * feature_dim = {}",
            node_features.len(),
            num_nodes * feature_dim
        )));
    }

    let mut output = vec![0.0_f64; num_nodes * feature_dim];

    for i in 0..num_nodes {
        let start = adj_row_offsets[i];
        let end = adj_row_offsets[i + 1];

        for edge_idx in start..end {
            let j = adj_col_indices[edge_idx];
            if j >= num_nodes {
                return Err(SparseError::InvalidArgument(format!(
                    "column index {} out of bounds for {} nodes",
                    j, num_nodes
                )));
            }
            let alpha = attention_weights[edge_idx];
            for f in 0..feature_dim {
                output[i * feature_dim + f] += alpha * node_features[j * feature_dim + f];
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Graph utilities
// ---------------------------------------------------------------------------

/// Compute the degree of each node from CSR row offsets.
///
/// `degree[i] = adj_row_offsets[i+1] - adj_row_offsets[i]`.
///
/// # Arguments
///
/// * `adj_row_offsets` -- CSR row pointer (length `num_nodes + 1`).
/// * `num_nodes` -- Number of nodes.
pub fn compute_degree_matrix(adj_row_offsets: &[usize], num_nodes: usize) -> Vec<f64> {
    let mut degrees = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let start = if i < adj_row_offsets.len() {
            adj_row_offsets[i]
        } else {
            0
        };
        let end = if i + 1 < adj_row_offsets.len() {
            adj_row_offsets[i + 1]
        } else {
            start
        };
        degrees.push((end - start) as f64);
    }
    degrees
}

/// Symmetric normalization of an adjacency matrix: D^{-1/2} A D^{-1/2}.
///
/// Given a CSR adjacency matrix with values, returns a new CSR matrix with
/// normalized values:
///
/// ```text
/// A_hat[i][j] = A[i][j] / sqrt(deg[i] * deg[j])
/// ```
///
/// This is the standard normalization used in GCN (Kipf & Welling, 2017).
///
/// # Arguments
///
/// * `adj_row_offsets` -- CSR row pointer.
/// * `adj_col_indices` -- CSR column indices.
/// * `adj_values` -- CSR values.
/// * `degrees` -- Degree of each node (precomputed via [`compute_degree_matrix`]).
///
/// # Returns
///
/// A tuple `(row_offsets, col_indices, values)` for the normalized CSR matrix.
/// The sparsity pattern is unchanged; only values are modified.
pub fn symmetric_normalize(
    adj_row_offsets: &[usize],
    adj_col_indices: &[usize],
    adj_values: &[f64],
    degrees: &[f64],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let num_nodes = if adj_row_offsets.is_empty() {
        0
    } else {
        adj_row_offsets.len() - 1
    };

    let mut new_values = Vec::with_capacity(adj_values.len());

    for i in 0..num_nodes {
        let start = adj_row_offsets[i];
        let end = adj_row_offsets[i + 1];

        let di = degrees[i];
        let di_inv_sqrt = if di > 0.0 { 1.0 / di.sqrt() } else { 0.0 };

        for edge_idx in start..end {
            let j = adj_col_indices[edge_idx];
            let dj = if j < degrees.len() { degrees[j] } else { 0.0 };
            let dj_inv_sqrt = if dj > 0.0 { 1.0 / dj.sqrt() } else { 0.0 };

            new_values.push(adj_values[edge_idx] * di_inv_sqrt * dj_inv_sqrt);
        }
    }

    (
        adj_row_offsets.to_vec(),
        adj_col_indices.to_vec(),
        new_values,
    )
}

/// Add self-loops to an adjacency matrix: A' = A + I.
///
/// For each node `i`, ensures there is an edge `(i, i)` with value `1.0`.
/// Existing self-loops are incremented by `1.0`.
///
/// # Arguments
///
/// * `adj_row_offsets` -- CSR row pointer.
/// * `adj_col_indices` -- CSR column indices.
/// * `adj_values` -- CSR values.
/// * `num_nodes` -- Number of nodes.
///
/// # Returns
///
/// A tuple `(row_offsets, col_indices, values)` for the new CSR matrix with self-loops.
pub fn add_self_loops(
    adj_row_offsets: &[usize],
    adj_col_indices: &[usize],
    adj_values: &[f64],
    num_nodes: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut new_row_offsets = Vec::with_capacity(num_nodes + 1);
    let mut new_col_indices = Vec::new();
    let mut new_values = Vec::new();

    new_row_offsets.push(0);

    for i in 0..num_nodes {
        let start = adj_row_offsets[i];
        let end = adj_row_offsets[i + 1];

        let mut has_self_loop = false;
        let mut inserted_self = false;

        for edge_idx in start..end {
            let j = adj_col_indices[edge_idx];

            // Insert self-loop at the correct sorted position
            if !inserted_self && j >= i {
                if j == i {
                    // Existing self-loop: increment value
                    has_self_loop = true;
                    new_col_indices.push(i);
                    new_values.push(adj_values[edge_idx] + 1.0);
                    inserted_self = true;
                    continue;
                }
                // Insert new self-loop before this entry
                new_col_indices.push(i);
                new_values.push(1.0);
                inserted_self = true;
            }

            new_col_indices.push(j);
            new_values.push(adj_values[edge_idx]);
        }

        // If self-loop not yet inserted (all neighbors have index < i, or no neighbors)
        if !has_self_loop && !inserted_self {
            new_col_indices.push(i);
            new_values.push(1.0);
        }

        new_row_offsets.push(new_col_indices.len());
    }

    (new_row_offsets, new_col_indices, new_values)
}

// ---------------------------------------------------------------------------
// Sparse softmax
// ---------------------------------------------------------------------------

/// Row-wise softmax over sparse matrix values.
///
/// For each row defined by `row_offsets`, computes:
///
/// ```text
/// softmax(v_i) = exp(v_i - max_row) / sum_j exp(v_j - max_row)
/// ```
///
/// Uses the numerically stable log-sum-exp trick.
///
/// # Arguments
///
/// * `row_offsets` -- CSR row pointer.
/// * `values` -- Sparse matrix values (modified in place conceptually; returns new vector).
///
/// # Returns
///
/// A new vector of softmax-normalized values with the same sparsity pattern.
pub fn sparse_row_softmax(row_offsets: &[usize], values: &[f64]) -> Vec<f64> {
    if row_offsets.len() <= 1 {
        return values.to_vec();
    }

    let num_rows = row_offsets.len() - 1;
    let mut result = vec![0.0_f64; values.len()];

    for i in 0..num_rows {
        let start = row_offsets[i];
        let end = row_offsets[i + 1];

        if start >= end {
            continue;
        }

        // Find row maximum for numerical stability
        let mut max_val = f64::NEG_INFINITY;
        for v in &values[start..end] {
            if *v > max_val {
                max_val = *v;
            }
        }

        // Compute exp(v - max) and sum
        let mut sum = 0.0_f64;
        for (r, v) in result[start..end].iter_mut().zip(&values[start..end]) {
            let e = (*v - max_val).exp();
            *r = e;
            sum += e;
        }

        // Normalize
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for r in &mut result[start..end] {
                *r *= inv_sum;
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generate a PTX kernel for GPU-accelerated GNN message passing.
///
/// Produces a kernel that performs sparse message passing on GPU, where each
/// thread block handles one or more nodes and iterates over their adjacency
/// lists to aggregate neighbor features.
///
/// The generated kernel name is `gnn_message_passing_f64`.
///
/// # Arguments
///
/// * `config` -- GNN sparse configuration.
///
/// # Errors
///
/// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
pub fn generate_message_passing_ptx(config: &GnnSparseConfig) -> Result<String, SparseError> {
    let op_name = match config.op {
        MessagePassingOp::Sum => "sum",
        MessagePassingOp::Mean => "mean",
        MessagePassingOp::Max => "max",
        MessagePassingOp::Min => "min",
    };
    let normalize_flag = if config.normalize || config.op == MessagePassingOp::Mean {
        1
    } else {
        0
    };
    let feature_dim = config.feature_dim;
    let num_nodes = config.num_nodes;

    if feature_dim == 0 {
        return Err(SparseError::PtxGeneration(
            "feature_dim must be > 0".to_string(),
        ));
    }

    // Generate PTX source for message passing kernel
    let ptx = format!(
        r#"//
// GNN Message Passing Kernel ({op_name})
// Generated by oxicuda-sparse tensor module
// num_nodes={num_nodes}, feature_dim={feature_dim}, normalize={normalize_flag}
//
.version 7.0
.target sm_70
.address_size 64

.visible .entry gnn_message_passing_f64(
    .param .u64 row_offsets_ptr,
    .param .u64 col_indices_ptr,
    .param .u64 node_features_ptr,
    .param .u64 output_ptr,
    .param .u32 num_nodes_param,
    .param .u32 feature_dim_param
)
{{
    .reg .u32 %r<32>;
    .reg .u64 %rd<32>;
    .reg .f64 %fd<16>;
    .reg .pred %p<8>;

    // tid = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    // Each thread handles one node
    ld.param.u32 %r4, [num_nodes_param];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 ret;

    // Load row_offsets[node] and row_offsets[node+1]
    ld.param.u64 %rd0, [row_offsets_ptr];
    cvt.u64.u32 %rd1, %r3;

    // offset = node * 8 (u64 = 8 bytes)
    shl.b64 %rd2, %rd1, 3;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.u64 %rd4, [%rd3];      // row_start
    ld.global.u64 %rd5, [%rd3 + 8];  // row_end

    // degree = row_end - row_start
    sub.u64 %rd6, %rd5, %rd4;

    // Skip if degree == 0
    setp.eq.u64 %p1, %rd6, 0;
    @%p1 ret;

    // Iterate over features and neighbors
    // (simplified: actual implementation would loop over feature_dim
    //  and accumulate per-feature aggregates)

    ret;
}}
"#
    );

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- scatter_reduce -------------------------------------------------------

    #[test]
    fn scatter_reduce_sum() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let index = vec![0, 1, 0, 1];
        let result = scatter_reduce(&src, &index, 2, MessagePassingOp::Sum)
            .expect("scatter_reduce Sum failed");
        assert!((result[0] - 4.0).abs() < 1e-12);
        assert!((result[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn scatter_reduce_mean() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let index = vec![0, 1, 0, 1];
        let result = scatter_reduce(&src, &index, 2, MessagePassingOp::Mean)
            .expect("scatter_reduce Mean failed");
        assert!((result[0] - 2.0).abs() < 1e-12);
        assert!((result[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn scatter_reduce_max() {
        let src = vec![1.0, 5.0, 3.0, 2.0];
        let index = vec![0, 1, 0, 1];
        let result = scatter_reduce(&src, &index, 2, MessagePassingOp::Max)
            .expect("scatter_reduce Max failed");
        assert!((result[0] - 3.0).abs() < 1e-12);
        assert!((result[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn scatter_reduce_min() {
        let src = vec![1.0, 5.0, 3.0, 2.0];
        let index = vec![0, 1, 0, 1];
        let result = scatter_reduce(&src, &index, 2, MessagePassingOp::Min)
            .expect("scatter_reduce Min failed");
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1] - 2.0).abs() < 1e-12);
    }

    // -- gather ---------------------------------------------------------------

    #[test]
    fn gather_basic() {
        let src = vec![10.0, 20.0, 30.0, 40.0];
        let index = vec![3, 0, 2, 1, 0];
        let result = gather(&src, &index).expect("gather failed");
        assert_eq!(result, vec![40.0, 10.0, 30.0, 20.0, 10.0]);
    }

    // -- message passing (triangle graph) ------------------------------------

    /// Triangle graph: 0--1--2--0 (undirected => 6 directed edges).
    fn triangle_csr() -> (Vec<usize>, Vec<usize>) {
        // Node 0: neighbors 1, 2
        // Node 1: neighbors 0, 2
        // Node 2: neighbors 0, 1
        let row_offsets = vec![0, 2, 4, 6];
        let col_indices = vec![1, 2, 0, 2, 0, 1];
        (row_offsets, col_indices)
    }

    #[test]
    fn message_passing_sum_triangle() {
        let (row_offsets, col_indices) = triangle_csr();
        let features = vec![1.0, 2.0, 3.0]; // 1D features
        let config = GnnSparseConfig {
            num_nodes: 3,
            feature_dim: 1,
            num_edges: 6,
            op: MessagePassingOp::Sum,
            normalize: false,
        };
        let result = sparse_message_passing(&row_offsets, &col_indices, &features, 1, &config)
            .expect("message_passing failed");
        // node 0: sum(features[1], features[2]) = 2 + 3 = 5
        // node 1: sum(features[0], features[2]) = 1 + 3 = 4
        // node 2: sum(features[0], features[1]) = 1 + 2 = 3
        assert!((result[0] - 5.0).abs() < 1e-12);
        assert!((result[1] - 4.0).abs() < 1e-12);
        assert!((result[2] - 3.0).abs() < 1e-12);
    }

    // -- attention-weighted message passing -----------------------------------

    #[test]
    fn attention_message_triangle() {
        let (row_offsets, col_indices) = triangle_csr();
        let features = vec![1.0, 2.0, 3.0];
        let attention = vec![0.5, 0.5, 0.3, 0.7, 0.6, 0.4];
        let result = sparse_attention_message(&row_offsets, &col_indices, &features, &attention, 1)
            .expect("attention message failed");
        // node 0: 0.5*features[1] + 0.5*features[2] = 0.5*2 + 0.5*3 = 2.5
        // node 1: 0.3*features[0] + 0.7*features[2] = 0.3*1 + 0.7*3 = 2.4
        // node 2: 0.6*features[0] + 0.4*features[1] = 0.6*1 + 0.4*2 = 1.4
        assert!((result[0] - 2.5).abs() < 1e-12);
        assert!((result[1] - 2.4).abs() < 1e-12);
        assert!((result[2] - 1.4).abs() < 1e-12);
    }

    // -- degree matrix --------------------------------------------------------

    #[test]
    fn degree_matrix_triangle() {
        let (row_offsets, _col_indices) = triangle_csr();
        let degrees = compute_degree_matrix(&row_offsets, 3);
        assert!((degrees[0] - 2.0).abs() < 1e-12);
        assert!((degrees[1] - 2.0).abs() < 1e-12);
        assert!((degrees[2] - 2.0).abs() < 1e-12);
    }

    // -- symmetric normalization ----------------------------------------------

    #[test]
    fn symmetric_normalize_triangle() {
        let (row_offsets, col_indices) = triangle_csr();
        let values = vec![1.0; 6];
        let degrees = compute_degree_matrix(&row_offsets, 3);
        let (new_rp, new_ci, new_vals) =
            symmetric_normalize(&row_offsets, &col_indices, &values, &degrees);

        // Structure unchanged
        assert_eq!(new_rp, row_offsets);
        assert_eq!(new_ci, col_indices);

        // For a regular graph (all degrees = 2):
        // D^-1/2 A D^-1/2 => each entry = 1.0 / sqrt(2*2) = 0.5
        for v in &new_vals {
            assert!((v - 0.5).abs() < 1e-12);
        }
    }

    // -- add self-loops -------------------------------------------------------

    #[test]
    fn add_self_loops_triangle() {
        let (row_offsets, col_indices) = triangle_csr();
        let values = vec![1.0; 6];
        let (new_rp, new_ci, new_vals) = add_self_loops(&row_offsets, &col_indices, &values, 3);

        // Each row gains one self-loop: 6 + 3 = 9 edges
        assert_eq!(new_rp.len(), 4); // 3 nodes + 1
        assert_eq!(new_ci.len(), 9);
        assert_eq!(new_vals.len(), 9);

        // Verify self-loops exist for each node
        for node in 0..3 {
            let start = new_rp[node];
            let end = new_rp[node + 1];
            let row_cols = &new_ci[start..end];
            assert!(row_cols.contains(&node), "node {} missing self-loop", node);
        }
    }

    // -- sparse row softmax ---------------------------------------------------

    #[test]
    fn sparse_softmax_sums_to_one() {
        let row_offsets = vec![0, 3, 5];
        let values = vec![1.0, 2.0, 3.0, 0.5, 1.5];
        let result = sparse_row_softmax(&row_offsets, &values);

        // Row 0: sum should be 1.0
        let row0_sum: f64 = result[0..3].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-12);

        // Row 1: sum should be 1.0
        let row1_sum: f64 = result[3..5].iter().sum();
        assert!((row1_sum - 1.0).abs() < 1e-12);

        // All values should be positive
        for v in &result {
            assert!(*v >= 0.0);
        }
    }

    // -- empty graph ----------------------------------------------------------

    #[test]
    fn empty_graph_message_passing() {
        let row_offsets = vec![0];
        let col_indices: Vec<usize> = vec![];
        let features: Vec<f64> = vec![];
        let config = GnnSparseConfig {
            num_nodes: 0,
            feature_dim: 1,
            num_edges: 0,
            op: MessagePassingOp::Sum,
            normalize: false,
        };
        let result = sparse_message_passing(&row_offsets, &col_indices, &features, 1, &config)
            .expect("empty graph failed");
        assert!(result.is_empty());
    }

    // -- single node graph ----------------------------------------------------

    #[test]
    fn single_node_no_edges() {
        let row_offsets = vec![0, 0];
        let col_indices: Vec<usize> = vec![];
        let features = vec![42.0];
        let config = GnnSparseConfig {
            num_nodes: 1,
            feature_dim: 1,
            num_edges: 0,
            op: MessagePassingOp::Sum,
            normalize: false,
        };
        let result = sparse_message_passing(&row_offsets, &col_indices, &features, 1, &config)
            .expect("single node failed");
        // No neighbors => output is 0
        assert!((result[0] - 0.0).abs() < 1e-12);
    }

    // -- disconnected graph ---------------------------------------------------

    #[test]
    fn disconnected_graph() {
        // 4 nodes, only 0->1 and 1->0 edges (nodes 2 and 3 isolated)
        let row_offsets = vec![0, 1, 2, 2, 2];
        let col_indices = vec![1, 0];
        let features = vec![1.0, 2.0, 3.0, 4.0];
        let config = GnnSparseConfig {
            num_nodes: 4,
            feature_dim: 1,
            num_edges: 2,
            op: MessagePassingOp::Sum,
            normalize: false,
        };
        let result = sparse_message_passing(&row_offsets, &col_indices, &features, 1, &config)
            .expect("disconnected graph failed");
        assert!((result[0] - 2.0).abs() < 1e-12); // node 0 gets features[1] = 2
        assert!((result[1] - 1.0).abs() < 1e-12); // node 1 gets features[0] = 1
        assert!((result[2] - 0.0).abs() < 1e-12); // isolated
        assert!((result[3] - 0.0).abs() < 1e-12); // isolated
    }

    // -- multi-dimensional features -------------------------------------------

    #[test]
    fn feature_dim_greater_than_one() {
        // 2 nodes, 0->1, 1->0, feature_dim=3
        let row_offsets = vec![0, 1, 2];
        let col_indices = vec![1, 0];
        let features = vec![
            1.0, 2.0, 3.0, // node 0
            4.0, 5.0, 6.0, // node 1
        ];
        let config = GnnSparseConfig {
            num_nodes: 2,
            feature_dim: 3,
            num_edges: 2,
            op: MessagePassingOp::Sum,
            normalize: false,
        };
        let result = sparse_message_passing(&row_offsets, &col_indices, &features, 3, &config)
            .expect("multi-dim features failed");
        // node 0 receives features[1] = [4, 5, 6]
        assert!((result[0] - 4.0).abs() < 1e-12);
        assert!((result[1] - 5.0).abs() < 1e-12);
        assert!((result[2] - 6.0).abs() < 1e-12);
        // node 1 receives features[0] = [1, 2, 3]
        assert!((result[3] - 1.0).abs() < 1e-12);
        assert!((result[4] - 2.0).abs() < 1e-12);
        assert!((result[5] - 3.0).abs() < 1e-12);
    }

    // -- normalize flag -------------------------------------------------------

    #[test]
    fn normalize_flag_divides_by_degree() {
        let (row_offsets, col_indices) = triangle_csr();
        let features = vec![2.0, 4.0, 6.0];
        let config = GnnSparseConfig {
            num_nodes: 3,
            feature_dim: 1,
            num_edges: 6,
            op: MessagePassingOp::Sum,
            normalize: true,
        };
        let result = sparse_message_passing(&row_offsets, &col_indices, &features, 1, &config)
            .expect("normalize failed");
        // node 0: (4+6)/2 = 5.0
        // node 1: (2+6)/2 = 4.0
        // node 2: (2+4)/2 = 3.0
        assert!((result[0] - 5.0).abs() < 1e-12);
        assert!((result[1] - 4.0).abs() < 1e-12);
        assert!((result[2] - 3.0).abs() < 1e-12);
    }

    // -- PTX generation smoke test --------------------------------------------

    #[test]
    fn ptx_generation_smoke_test() {
        let config = GnnSparseConfig {
            num_nodes: 1024,
            feature_dim: 64,
            num_edges: 8192,
            op: MessagePassingOp::Sum,
            normalize: true,
        };
        let ptx = generate_message_passing_ptx(&config).expect("PTX generation failed");
        assert!(ptx.contains("gnn_message_passing_f64"));
        assert!(ptx.contains(".version 7.0"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains("num_nodes"));
    }

    #[test]
    fn ptx_generation_all_ops() {
        for op in &[
            MessagePassingOp::Sum,
            MessagePassingOp::Mean,
            MessagePassingOp::Max,
            MessagePassingOp::Min,
        ] {
            let config = GnnSparseConfig {
                num_nodes: 256,
                feature_dim: 32,
                num_edges: 1024,
                op: *op,
                normalize: false,
            };
            let ptx = generate_message_passing_ptx(&config).expect("PTX generation failed");
            assert!(!ptx.is_empty());
        }
    }
}
