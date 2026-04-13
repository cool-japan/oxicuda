//! Error types for the OxiCUDA graph execution engine.

use thiserror::Error;

use crate::node::NodeId;

/// Result type for graph operations.
pub type GraphResult<T> = Result<T, GraphError>;

/// Errors that can occur during graph construction, analysis, or execution.
#[derive(Debug, Error)]
pub enum GraphError {
    /// A node referenced by ID does not exist in the graph.
    #[error("node {0:?} not found in graph")]
    NodeNotFound(NodeId),

    /// An edge would create a cycle in the computation DAG.
    #[error("adding edge from {from:?} to {to:?} would create a cycle")]
    CycleDetected { from: NodeId, to: NodeId },

    /// Graph has no nodes.
    #[error("graph is empty")]
    EmptyGraph,

    /// Two nodes that should be compatible have incompatible buffer shapes.
    #[error("buffer shape mismatch at node {node:?}: expected {expected} bytes, got {got} bytes")]
    BufferSizeMismatch {
        node: NodeId,
        expected: usize,
        got: usize,
    },

    /// Fusion of two nodes was requested but they are not fusible.
    #[error("nodes {a:?} and {b:?} cannot be fused: {reason}")]
    FusionNotPossible {
        a: NodeId,
        b: NodeId,
        reason: &'static str,
    },

    /// A memory planning constraint could not be satisfied.
    #[error("memory planning failed: {0}")]
    MemoryPlanningFailed(String),

    /// Stream assignment produced an invalid schedule.
    #[error("stream partitioning failed: {0}")]
    StreamPartitioningFailed(String),

    /// Execution plan validation error.
    #[error("invalid execution plan: {0}")]
    InvalidPlan(String),

    /// PTX code generation for a fused kernel failed.
    #[error("PTX codegen failed for fusion group {group}: {reason}")]
    PtxCodegenFailed { group: usize, reason: String },

    /// General internal error (should not occur in correct usage).
    #[error("internal graph error: {0}")]
    Internal(String),
}
