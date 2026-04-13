//! Model weight storage.
//!
//! `WeightTensor` is a named flat `f32` buffer with shape metadata.
//! `ModelWeights` is a dictionary mapping string names to weight tensors,
//! mirroring the PyTorch / HuggingFace naming convention so that callers
//! can address weights with familiar names like `"model.layers.0.self_attn.q_proj.weight"`.

use std::collections::HashMap;

use crate::error::{LmError, LmResult};

// ─── WeightTensor ────────────────────────────────────────────────────────────

/// A named flat `f32` weight tensor.
///
/// Weights are stored in **row-major** order.  For a 2-D matrix `W[m, n]`,
/// the element at row `i` column `j` is at `data[i * n + j]`.
#[derive(Debug, Clone)]
pub struct WeightTensor {
    /// Raw data in row-major order.
    pub data: Vec<f32>,
    /// Shape of the tensor (product equals `data.len()`).
    pub shape: Vec<usize>,
}

impl WeightTensor {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Construct from existing data, validating that `data.len()` equals the
    /// product of `shape`.
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> LmResult<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(LmError::WeightDataLengthMismatch {
                data_len: data.len(),
                shape: shape.clone(),
                expected,
            });
        }
        Ok(Self { data, shape })
    }

    /// Tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![0.0_f32; n],
            shape: shape.to_vec(),
        }
    }

    /// Tensor filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![1.0_f32; n],
            shape: shape.to_vec(),
        }
    }

    /// Identity-like weight: ones on the "diagonal" of a 2-D matrix.
    ///
    /// For a non-square matrix the identity is placed in the top-left corner.
    pub fn eye(rows: usize, cols: usize) -> Self {
        let mut data = vec![0.0_f32; rows * cols];
        for i in 0..rows.min(cols) {
            data[i * cols + i] = 1.0;
        }
        Self {
            data,
            shape: vec![rows, cols],
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /// Total number of elements.
    pub fn n_elements(&self) -> usize {
        self.data.len()
    }

    /// Number of tensor dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Borrow the underlying data.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable borrow of the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// For a 2-D weight matrix `[rows × cols]`, return the `row`-th row slice.
    pub fn row_slice(&self, row: usize) -> LmResult<&[f32]> {
        if self.shape.len() != 2 {
            return Err(LmError::DimensionMismatch {
                expected: 2,
                got: self.shape.len(),
            });
        }
        let cols = self.shape[1];
        let start = row * cols;
        if start + cols > self.data.len() {
            return Err(LmError::DimensionMismatch {
                expected: row,
                got: self.shape[0],
            });
        }
        Ok(&self.data[start..start + cols])
    }

    /// Validate that this tensor has the given shape.
    pub fn validate_shape(&self, expected: &[usize]) -> LmResult<()> {
        if self.shape != expected {
            return Err(LmError::WeightShapeMismatch {
                name: String::new(),
                expected: expected.to_vec(),
                got: self.shape.clone(),
            });
        }
        Ok(())
    }
}

// ─── ModelWeights ────────────────────────────────────────────────────────────

/// A dictionary of named weight tensors.
///
/// Weight names follow the HuggingFace naming convention, e.g.:
/// - `"model.embed_tokens.weight"`
/// - `"model.layers.0.self_attn.q_proj.weight"`
/// - `"model.layers.0.mlp.gate_proj.weight"`
/// - `"lm_head.weight"`
#[derive(Debug, Clone, Default)]
pub struct ModelWeights {
    weights: HashMap<String, WeightTensor>,
}

impl ModelWeights {
    /// Create an empty weight store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or overwrite) a tensor under `name`.
    pub fn insert(&mut self, name: impl Into<String>, tensor: WeightTensor) {
        self.weights.insert(name.into(), tensor);
    }

    /// Retrieve a tensor by name.
    pub fn get(&self, name: &str) -> LmResult<&WeightTensor> {
        self.weights
            .get(name)
            .ok_or_else(|| LmError::WeightNotFound { name: name.into() })
    }

    /// Retrieve a tensor by name and validate its shape.
    pub fn get_checked(&self, name: &str, expected_shape: &[usize]) -> LmResult<&WeightTensor> {
        let t = self.get(name)?;
        if t.shape != expected_shape {
            return Err(LmError::WeightShapeMismatch {
                name: name.into(),
                expected: expected_shape.to_vec(),
                got: t.shape.clone(),
            });
        }
        Ok(t)
    }

    /// Whether `name` is present in the store.
    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// Iterator over all `(name, tensor)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &WeightTensor)> {
        self.weights.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Total number of scalar parameters across all tensors.
    pub fn n_params(&self) -> usize {
        self.weights.values().map(|t| t.n_elements()).sum()
    }

    /// Number of named weight entries.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_tensor_zeros() {
        let w = WeightTensor::zeros(&[4, 8]);
        assert_eq!(w.n_elements(), 32);
        assert!(w.data.iter().all(|&x| x == 0.0));
        assert_eq!(w.shape, vec![4, 8]);
    }

    #[test]
    fn weight_tensor_ones() {
        let w = WeightTensor::ones(&[3, 3]);
        assert_eq!(w.n_elements(), 9);
        assert!(w.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn weight_tensor_eye() {
        let w = WeightTensor::eye(3, 3);
        assert_eq!(w.data[0], 1.0);
        assert_eq!(w.data[1], 0.0);
        assert_eq!(w.data[4], 1.0); // [1,1]
        assert_eq!(w.data[8], 1.0); // [2,2]
    }

    #[test]
    fn weight_tensor_from_data_ok() {
        let d = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w = WeightTensor::from_data(d.clone(), vec![2, 2]).unwrap();
        assert_eq!(w.data, d);
    }

    #[test]
    fn weight_tensor_from_data_shape_mismatch() {
        let d = vec![1.0_f32; 5];
        let err = WeightTensor::from_data(d, vec![2, 2]).unwrap_err();
        assert!(matches!(err, LmError::WeightDataLengthMismatch { .. }));
    }

    #[test]
    fn weight_tensor_row_slice() {
        let w = WeightTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(w.row_slice(0).unwrap(), &[1.0_f32, 2.0]);
        assert_eq!(w.row_slice(1).unwrap(), &[3.0_f32, 4.0]);
    }

    #[test]
    fn weight_tensor_row_slice_non_2d_errors() {
        let w = WeightTensor::zeros(&[8]);
        assert!(w.row_slice(0).is_err());
    }

    #[test]
    fn weight_tensor_validate_shape_ok() {
        let w = WeightTensor::zeros(&[4, 8]);
        w.validate_shape(&[4, 8]).unwrap();
    }

    #[test]
    fn weight_tensor_validate_shape_fail() {
        let w = WeightTensor::zeros(&[4, 8]);
        assert!(w.validate_shape(&[8, 4]).is_err());
    }

    #[test]
    fn model_weights_insert_and_get() {
        let mut mw = ModelWeights::new();
        mw.insert("embed", WeightTensor::zeros(&[10, 4]));
        let t = mw.get("embed").unwrap();
        assert_eq!(t.shape, vec![10, 4]);
    }

    #[test]
    fn model_weights_get_missing_errors() {
        let mw = ModelWeights::new();
        assert!(matches!(
            mw.get("missing"),
            Err(LmError::WeightNotFound { .. })
        ));
    }

    #[test]
    fn model_weights_get_checked_shape_error() {
        let mut mw = ModelWeights::new();
        mw.insert("w", WeightTensor::zeros(&[4, 8]));
        assert!(matches!(
            mw.get_checked("w", &[8, 4]),
            Err(LmError::WeightShapeMismatch { .. })
        ));
    }

    #[test]
    fn model_weights_n_params() {
        let mut mw = ModelWeights::new();
        mw.insert("a", WeightTensor::zeros(&[4, 4])); // 16
        mw.insert("b", WeightTensor::zeros(&[3, 3])); // 9
        assert_eq!(mw.n_params(), 25);
    }

    #[test]
    fn model_weights_len_and_empty() {
        let mut mw = ModelWeights::new();
        assert!(mw.is_empty());
        mw.insert("x", WeightTensor::zeros(&[1]));
        assert_eq!(mw.len(), 1);
        assert!(!mw.is_empty());
    }
}
