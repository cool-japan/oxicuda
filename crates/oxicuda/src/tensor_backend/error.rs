//! Error types for the tensor backend.

use std::fmt;

/// Error type for tensor operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    /// Shape mismatch between operands.
    ShapeMismatch {
        /// Expected total elements.
        expected: usize,
        /// Actual total elements.
        got: usize,
    },
    /// Dimension index out of range.
    InvalidDimension {
        /// Requested dimension.
        dim: usize,
        /// Number of dimensions.
        ndim: usize,
    },
    /// Index out of bounds for a flat access.
    IndexOutOfBounds {
        /// Requested index.
        index: usize,
        /// Tensor size on that axis.
        size: usize,
    },
    /// Data type mismatch.
    DtypeMismatch {
        /// Expected dtype name.
        expected: String,
        /// Actual dtype name.
        got: String,
    },
    /// A generic invalid operation.
    InvalidOperation(String),
    /// Device mismatch (tensors live on different GPUs).
    DeviceMismatch {
        /// First tensor device.
        a: usize,
        /// Second tensor device.
        b: usize,
    },
    /// Gradient computation error.
    AutogradError(String),
    /// Optimizer error.
    OptimizerError(String),
    /// Mixed precision / loss scaling error.
    MixedPrecisionError(String),
    /// Incompatible shapes for broadcast.
    BroadcastError(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected} elements, got {got}")
            }
            Self::InvalidDimension { dim, ndim } => {
                write!(f, "invalid dimension {dim} for {ndim}-d tensor")
            }
            Self::IndexOutOfBounds { index, size } => {
                write!(f, "index {index} out of bounds for size {size}")
            }
            Self::DtypeMismatch { expected, got } => {
                write!(f, "dtype mismatch: expected {expected}, got {got}")
            }
            Self::InvalidOperation(msg) => write!(f, "invalid operation: {msg}"),
            Self::DeviceMismatch { a, b } => {
                write!(f, "device mismatch: device {a} vs device {b}")
            }
            Self::AutogradError(msg) => write!(f, "autograd error: {msg}"),
            Self::OptimizerError(msg) => write!(f, "optimizer error: {msg}"),
            Self::MixedPrecisionError(msg) => write!(f, "mixed precision error: {msg}"),
            Self::BroadcastError(msg) => write!(f, "broadcast error: {msg}"),
        }
    }
}

impl std::error::Error for TensorError {}
