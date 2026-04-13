//! ONNX Intermediate Representation types.
//!
//! Defines the core data structures for representing ONNX graphs:
//! [`DataType`], [`TensorShape`], [`TensorInfo`], [`AttributeValue`],
//! [`Node`], [`Graph`], and [`OnnxTensor`].

use std::collections::HashMap;
use std::fmt;

// ─── Error types ────────────────────────────────────────────

/// Error type for ONNX backend operations.
#[derive(Debug, Clone, PartialEq)]
pub enum OnnxError {
    /// Graph structure is invalid (cycles, missing connections).
    InvalidGraph(String),
    /// Encountered an unsupported ONNX operator.
    UnsupportedOp(String),
    /// Tensor shape mismatch or incompatible shapes.
    ShapeMismatch(String),
    /// Data type error or incompatible types.
    TypeError(String),
    /// Invalid or missing attribute value.
    InvalidAttribute(String),
    /// Runtime execution error.
    ExecutionError(String),
    /// Invalid tensor data (wrong size, corrupt bytes).
    InvalidData(String),
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(msg) => write!(f, "invalid graph: {msg}"),
            Self::UnsupportedOp(msg) => write!(f, "unsupported op: {msg}"),
            Self::ShapeMismatch(msg) => write!(f, "shape mismatch: {msg}"),
            Self::TypeError(msg) => write!(f, "type error: {msg}"),
            Self::InvalidAttribute(msg) => write!(f, "invalid attribute: {msg}"),
            Self::ExecutionError(msg) => write!(f, "execution error: {msg}"),
            Self::InvalidData(msg) => write!(f, "invalid data: {msg}"),
        }
    }
}

impl std::error::Error for OnnxError {}

/// Result type for ONNX backend operations.
pub type OnnxResult<T> = Result<T, OnnxError>;

// ─── Data types ─────────────────────────────────────────────

/// ONNX tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// 32-bit floating point.
    Float32,
    /// 64-bit floating point.
    Float64,
    /// 16-bit floating point.
    Float16,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 8-bit signed integer.
    Int8,
    /// 8-bit unsigned integer.
    Uint8,
    /// Boolean.
    Bool,
}

impl DataType {
    /// Size in bytes of a single element of this type.
    pub fn size_bytes(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Float16 => 2,
            Self::Int32 => 4,
            Self::Int64 => 8,
            Self::Int8 | Self::Uint8 | Self::Bool => 1,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float32 => write!(f, "float32"),
            Self::Float64 => write!(f, "float64"),
            Self::Float16 => write!(f, "float16"),
            Self::Int32 => write!(f, "int32"),
            Self::Int64 => write!(f, "int64"),
            Self::Int8 => write!(f, "int8"),
            Self::Uint8 => write!(f, "uint8"),
            Self::Bool => write!(f, "bool"),
        }
    }
}

// ─── Tensor shape ───────────────────────────────────────────

/// Tensor shape with optional dynamic dimensions.
///
/// Each dimension is `Some(size)` for a known static dimension, or
/// `None` for a dynamic (symbolic) dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    /// Dimension sizes (`None` = dynamic).
    pub dims: Vec<Option<usize>>,
}

impl TensorShape {
    /// Create a shape with all dimensions known.
    pub fn fixed(dims: Vec<usize>) -> Self {
        Self {
            dims: dims.into_iter().map(Some).collect(),
        }
    }

    /// Create a shape from mixed known/unknown dimensions.
    pub fn new(dims: Vec<Option<usize>>) -> Self {
        Self { dims }
    }

    /// Whether all dimensions are known (no dynamic dims).
    pub fn is_fully_known(&self) -> bool {
        self.dims.iter().all(Option::is_some)
    }

    /// Number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Convert to concrete dimensions, failing if any are dynamic.
    pub fn to_concrete(&self) -> OnnxResult<Vec<usize>> {
        self.dims
            .iter()
            .enumerate()
            .map(|(i, d)| {
                d.ok_or_else(|| OnnxError::ShapeMismatch(format!("dimension {i} is dynamic")))
            })
            .collect()
    }

    /// Total number of elements (product of all dimensions).
    /// Returns `None` if any dimension is dynamic or overflow occurs.
    pub fn element_count(&self) -> Option<usize> {
        self.dims
            .iter()
            .copied()
            .try_fold(1usize, |acc, d| d.and_then(|s| acc.checked_mul(s)))
    }
}

// ─── Tensor info ────────────────────────────────────────────

/// Metadata about a tensor (name, type, shape) without actual data.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (used for graph connections).
    pub name: String,
    /// Data type.
    pub dtype: DataType,
    /// Shape.
    pub shape: TensorShape,
}

// ─── Attribute values ───────────────────────────────────────

/// Attribute value for ONNX node attributes.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    /// Single integer value.
    Int(i64),
    /// Single floating-point value.
    Float(f64),
    /// String value.
    String(String),
    /// Tensor value (for constant folding).
    Tensor(OnnxTensor),
    /// List of integers.
    Ints(Vec<i64>),
    /// List of floating-point values.
    Floats(Vec<f64>),
    /// List of strings.
    Strings(Vec<String>),
}

impl AttributeValue {
    /// Extract as integer.
    pub fn as_int(&self) -> OnnxResult<i64> {
        match self {
            Self::Int(v) => Ok(*v),
            other => Err(OnnxError::InvalidAttribute(format!(
                "expected Int, got {other:?}"
            ))),
        }
    }

    /// Extract as float.
    pub fn as_float(&self) -> OnnxResult<f64> {
        match self {
            Self::Float(v) => Ok(*v),
            other => Err(OnnxError::InvalidAttribute(format!(
                "expected Float, got {other:?}"
            ))),
        }
    }

    /// Extract as string reference.
    pub fn as_string(&self) -> OnnxResult<&str> {
        match self {
            Self::String(v) => Ok(v.as_str()),
            other => Err(OnnxError::InvalidAttribute(format!(
                "expected String, got {other:?}"
            ))),
        }
    }

    /// Extract as integer slice.
    pub fn as_ints(&self) -> OnnxResult<&[i64]> {
        match self {
            Self::Ints(v) => Ok(v.as_slice()),
            other => Err(OnnxError::InvalidAttribute(format!(
                "expected Ints, got {other:?}"
            ))),
        }
    }

    /// Extract as float slice.
    pub fn as_floats(&self) -> OnnxResult<&[f64]> {
        match self {
            Self::Floats(v) => Ok(v.as_slice()),
            other => Err(OnnxError::InvalidAttribute(format!(
                "expected Floats, got {other:?}"
            ))),
        }
    }
}

// ─── Node ───────────────────────────────────────────────────

/// A single operation in an ONNX graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// ONNX operator type (e.g., "Conv", "Relu", "MatMul").
    pub op_type: String,
    /// Human-readable node name.
    pub name: String,
    /// Input tensor names (empty string = optional absent input).
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Operator attributes.
    pub attributes: HashMap<String, AttributeValue>,
}

// ─── Graph ──────────────────────────────────────────────────

/// A complete ONNX computation graph.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Graph name.
    pub name: String,
    /// Nodes in the graph (not necessarily in execution order).
    pub nodes: Vec<Node>,
    /// Graph input tensor metadata.
    pub inputs: Vec<TensorInfo>,
    /// Graph output tensor metadata.
    pub outputs: Vec<TensorInfo>,
    /// Pre-loaded weight tensors (graph initializers).
    pub initializers: HashMap<String, OnnxTensor>,
}

// ─── Tensor ─────────────────────────────────────────────────

/// A concrete tensor with data stored as raw bytes (little-endian).
#[derive(Debug, Clone, PartialEq)]
pub struct OnnxTensor {
    /// Raw byte data (little-endian).
    pub data: Vec<u8>,
    /// Data type.
    pub dtype: DataType,
    /// Shape (all dimensions are concrete).
    pub shape: Vec<usize>,
}

impl OnnxTensor {
    /// Total number of elements.
    pub fn element_count(&self) -> usize {
        if self.shape.is_empty() {
            1 // scalar
        } else {
            self.shape.iter().product()
        }
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.element_count() * self.dtype.size_bytes()
    }

    /// Create a zero-filled tensor.
    pub fn zeros(shape: Vec<usize>, dtype: DataType) -> Self {
        let count: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let data = vec![0u8; count * dtype.size_bytes()];
        Self { data, dtype, shape }
    }

    /// Create from f32 slice.
    pub fn from_f32(values: &[f32], shape: Vec<usize>) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data,
            dtype: DataType::Float32,
            shape,
        }
    }

    /// Create from f64 slice.
    pub fn from_f64(values: &[f64], shape: Vec<usize>) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data,
            dtype: DataType::Float64,
            shape,
        }
    }

    /// Create from i32 slice.
    pub fn from_i32(values: &[i32], shape: Vec<usize>) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data,
            dtype: DataType::Int32,
            shape,
        }
    }

    /// Create from i64 slice.
    pub fn from_i64(values: &[i64], shape: Vec<usize>) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data,
            dtype: DataType::Int64,
            shape,
        }
    }

    /// Create from bool slice.
    pub fn from_bool(values: &[bool], shape: Vec<usize>) -> Self {
        let data: Vec<u8> = values.iter().map(|&v| u8::from(v)).collect();
        Self {
            data,
            dtype: DataType::Bool,
            shape,
        }
    }

    /// Create a scalar f32 tensor.
    pub fn scalar_f32(value: f32) -> Self {
        Self::from_f32(&[value], vec![])
    }

    /// Create a scalar i64 tensor.
    pub fn scalar_i64(value: i64) -> Self {
        Self::from_i64(&[value], vec![])
    }

    /// Read data as f32 values.
    pub fn as_f32(&self) -> OnnxResult<Vec<f32>> {
        if self.dtype != DataType::Float32 {
            return Err(OnnxError::TypeError(format!(
                "expected Float32, got {:?}",
                self.dtype
            )));
        }
        Ok(self
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Read data as f64 values.
    pub fn as_f64(&self) -> OnnxResult<Vec<f64>> {
        if self.dtype != DataType::Float64 {
            return Err(OnnxError::TypeError(format!(
                "expected Float64, got {:?}",
                self.dtype
            )));
        }
        Ok(self
            .data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    /// Read data as i32 values.
    pub fn as_i32(&self) -> OnnxResult<Vec<i32>> {
        if self.dtype != DataType::Int32 {
            return Err(OnnxError::TypeError(format!(
                "expected Int32, got {:?}",
                self.dtype
            )));
        }
        Ok(self
            .data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Read data as i64 values.
    pub fn as_i64(&self) -> OnnxResult<Vec<i64>> {
        if self.dtype != DataType::Int64 {
            return Err(OnnxError::TypeError(format!(
                "expected Int64, got {:?}",
                self.dtype
            )));
        }
        Ok(self
            .data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    /// Read data as bool values.
    pub fn as_bool(&self) -> OnnxResult<Vec<bool>> {
        if self.dtype != DataType::Bool {
            return Err(OnnxError::TypeError(format!(
                "expected Bool, got {:?}",
                self.dtype
            )));
        }
        Ok(self.data.iter().map(|&b| b != 0).collect())
    }
}

// ─── Broadcasting helpers ───────────────────────────────────

/// Compute the broadcast shape from two input shapes (numpy-style).
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> OnnxResult<Vec<usize>> {
    let max_rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_rank);
    for i in 0..max_rank {
        let da = if i < max_rank - a.len() {
            1
        } else {
            a[i + a.len() - max_rank]
        };
        let db = if i < max_rank - b.len() {
            1
        } else {
            b[i + b.len() - max_rank]
        };
        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(OnnxError::ShapeMismatch(format!(
                "cannot broadcast shapes {a:?} and {b:?} at dimension {i}"
            )));
        }
    }
    Ok(result)
}

/// Convert a flat index to multi-dimensional indices.
pub fn flat_to_multi(flat: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut indices = vec![0usize; shape.len()];
    let mut remaining = flat;
    for i in (0..shape.len()).rev() {
        if shape[i] > 0 {
            indices[i] = remaining % shape[i];
            remaining /= shape[i];
        }
    }
    indices
}

/// Convert multi-dimensional indices to a flat index.
pub fn multi_to_flat(indices: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        flat += indices[i] * stride;
        stride *= shape[i];
    }
    flat
}

/// Map output indices to input flat index with broadcasting.
pub fn broadcast_index(out_indices: &[usize], in_shape: &[usize], out_shape: &[usize]) -> usize {
    if in_shape.is_empty() {
        return 0; // scalar
    }
    let offset = out_shape.len() - in_shape.len();
    let mut mapped = vec![0usize; in_shape.len()];
    for i in 0..in_shape.len() {
        mapped[i] = if in_shape[i] == 1 {
            0
        } else {
            out_indices[i + offset]
        };
    }
    multi_to_flat(&mapped, in_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_sizes() {
        assert_eq!(DataType::Float32.size_bytes(), 4);
        assert_eq!(DataType::Float64.size_bytes(), 8);
        assert_eq!(DataType::Float16.size_bytes(), 2);
        assert_eq!(DataType::Int32.size_bytes(), 4);
        assert_eq!(DataType::Int64.size_bytes(), 8);
        assert_eq!(DataType::Int8.size_bytes(), 1);
        assert_eq!(DataType::Uint8.size_bytes(), 1);
        assert_eq!(DataType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_tensor_shape_fixed() {
        let shape = TensorShape::fixed(vec![2, 3, 4]);
        assert!(shape.is_fully_known());
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.element_count(), Some(24));
        assert_eq!(shape.to_concrete().ok(), Some(vec![2, 3, 4]));
    }

    #[test]
    fn test_tensor_shape_dynamic() {
        let shape = TensorShape::new(vec![Some(2), None, Some(4)]);
        assert!(!shape.is_fully_known());
        assert_eq!(shape.element_count(), None);
        assert!(shape.to_concrete().is_err());
    }

    #[test]
    fn test_tensor_f32_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = OnnxTensor::from_f32(&values, vec![2, 3]);
        assert_eq!(t.element_count(), 6);
        assert_eq!(t.dtype, DataType::Float32);
        assert_eq!(t.as_f32().ok(), Some(values));
    }

    #[test]
    fn test_tensor_i64_roundtrip() {
        let values = vec![10i64, 20, 30];
        let t = OnnxTensor::from_i64(&values, vec![3]);
        assert_eq!(t.as_i64().ok(), Some(values));
    }

    #[test]
    fn test_tensor_bool_roundtrip() {
        let values = vec![true, false, true, false];
        let t = OnnxTensor::from_bool(&values, vec![4]);
        assert_eq!(t.as_bool().ok(), Some(values));
    }

    #[test]
    fn test_tensor_type_mismatch() {
        let t = OnnxTensor::from_f32(&[1.0], vec![1]);
        assert!(t.as_i64().is_err());
        assert!(t.as_bool().is_err());
    }

    #[test]
    fn test_scalar_tensor() {
        let t = OnnxTensor::scalar_f32(7.125);
        assert_eq!(t.shape, Vec::<usize>::new());
        assert_eq!(t.element_count(), 1);
        assert_eq!(t.as_f32().ok(), Some(vec![7.125]));
    }

    #[test]
    fn test_zeros() {
        let t = OnnxTensor::zeros(vec![2, 3], DataType::Float32);
        let data = t.as_f32().ok();
        assert_eq!(data, Some(vec![0.0; 6]));
    }

    #[test]
    fn test_attribute_accessors() {
        let a = AttributeValue::Int(42);
        assert_eq!(a.as_int().ok(), Some(42));
        assert!(a.as_float().is_err());

        let b = AttributeValue::Float(7.125);
        assert!((b.as_float().ok().unwrap_or(0.0) - 7.125).abs() < 1e-10);

        let c = AttributeValue::String("relu".into());
        assert_eq!(c.as_string().ok(), Some("relu"));

        let d = AttributeValue::Ints(vec![1, 2, 3]);
        assert_eq!(d.as_ints().ok(), Some(&[1i64, 2, 3][..]));
    }

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(broadcast_shapes(&[3, 4], &[3, 4]).ok(), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[1, 4], &[3, 1]).ok(), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[4], &[3, 4]).ok(), Some(vec![3, 4]));
        assert_eq!(
            broadcast_shapes(&[2, 1, 4], &[3, 1]).ok(),
            Some(vec![2, 3, 4])
        );
        assert!(broadcast_shapes(&[3, 4], &[3, 5]).is_err());
    }

    #[test]
    fn test_flat_to_multi() {
        assert_eq!(flat_to_multi(0, &[2, 3]), vec![0, 0]);
        assert_eq!(flat_to_multi(1, &[2, 3]), vec![0, 1]);
        assert_eq!(flat_to_multi(3, &[2, 3]), vec![1, 0]);
        assert_eq!(flat_to_multi(5, &[2, 3]), vec![1, 2]);
    }

    #[test]
    fn test_multi_to_flat() {
        assert_eq!(multi_to_flat(&[0, 0], &[2, 3]), 0);
        assert_eq!(multi_to_flat(&[0, 1], &[2, 3]), 1);
        assert_eq!(multi_to_flat(&[1, 0], &[2, 3]), 3);
        assert_eq!(multi_to_flat(&[1, 2], &[2, 3]), 5);
    }

    #[test]
    fn test_broadcast_index_scalar() {
        assert_eq!(broadcast_index(&[0, 0], &[], &[2, 3]), 0);
    }

    #[test]
    fn test_graph_construction() {
        let graph = Graph {
            name: "test".into(),
            nodes: vec![Node {
                op_type: "Relu".into(),
                name: "relu0".into(),
                inputs: vec!["x".into()],
                outputs: vec!["y".into()],
                attributes: HashMap::new(),
            }],
            inputs: vec![TensorInfo {
                name: "x".into(),
                dtype: DataType::Float32,
                shape: TensorShape::fixed(vec![1, 3]),
            }],
            outputs: vec![TensorInfo {
                name: "y".into(),
                dtype: DataType::Float32,
                shape: TensorShape::fixed(vec![1, 3]),
            }],
            initializers: HashMap::new(),
        };
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op_type, "Relu");
    }
}
