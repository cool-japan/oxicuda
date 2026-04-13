//! GPU Tensor type with shape, stride, dtype, and autograd support.
//!
//! The [`GpuTensor`] struct is the primary data type for the tensor backend.
//! It represents a multi-dimensional array stored in device memory.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use super::autograd::GradFn;
use super::dtype::TensorDtype;
use super::error::TensorError;

// ─── Tensor ID ──────────────────────────────────────────────

/// Unique identifier for a tensor in the autograd graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub(crate) u64);

impl TensorId {
    /// Generate a new unique tensor id.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for TensorId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor#{}", self.0)
    }
}

// ─── Saved tensor for backward ──────────────────────────────

/// A lightweight reference to tensor data saved for the backward pass.
///
/// In a real GPU implementation this would hold a device pointer and ref-count;
/// here we store the essential metadata and a simulated host-side snapshot
/// for correctness testing.
#[derive(Debug, Clone)]
pub struct SavedTensor {
    /// Tensor ID of the saved tensor.
    pub id: TensorId,
    /// Shape at the time of saving.
    pub shape: Vec<usize>,
    /// Data type at the time of saving.
    pub dtype: TensorDtype,
    /// Host-side data snapshot (f64 for numerical precision in backward).
    pub data: Vec<f64>,
}

impl SavedTensor {
    /// Create a saved tensor from a GPU tensor snapshot.
    pub fn from_tensor(tensor: &GpuTensor) -> Self {
        Self {
            id: tensor.id,
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
            data: tensor.host_data.clone(),
        }
    }
}

// ─── GPU Tensor ─────────────────────────────────────────────

/// A GPU tensor with shape, strides, dtype, device id, and autograd metadata.
///
/// This type represents a dense tensor stored on a GPU device. It supports
/// automatic differentiation through the [`GradFn`] and the autograd tape.
///
/// # Example
///
/// ```rust
/// use oxicuda::tensor_backend::{GpuTensor, TensorDtype};
///
/// let t = GpuTensor::zeros(&[2, 3], TensorDtype::Float32, 0).unwrap();
/// assert_eq!(t.shape(), &[2, 3]);
/// assert_eq!(t.numel(), 6);
/// ```
#[derive(Debug, Clone)]
pub struct GpuTensor {
    /// Unique tensor id for the autograd graph.
    pub(crate) id: TensorId,
    /// Shape of the tensor (e.g. [batch, channels, height, width]).
    pub(crate) shape: Vec<usize>,
    /// Strides in number of elements (row-major by default).
    pub(crate) strides: Vec<usize>,
    /// Element data type.
    pub(crate) dtype: TensorDtype,
    /// GPU device index this tensor lives on.
    pub(crate) device_id: usize,
    /// Opaque device memory pointer. 0 when simulated on CPU.
    pub(crate) data_ptr: u64,
    /// Total number of elements.
    pub(crate) numel: usize,
    /// Whether gradients should be tracked.
    pub(crate) requires_grad: bool,
    /// Accumulated gradient (same shape & dtype).
    pub(crate) grad: Option<Box<GpuTensor>>,
    /// The autograd function that produced this tensor.
    pub(crate) grad_fn: Option<GradFn>,

    // Host-side data for CPU-simulated mode (testing without a GPU).
    pub(crate) host_data: Vec<f64>,
}

impl GpuTensor {
    // ── Construction ────────────────────────────────────────

    /// Create a new zero-filled tensor with the given shape and dtype.
    pub fn zeros(
        shape: &[usize],
        dtype: TensorDtype,
        device_id: usize,
    ) -> Result<Self, TensorError> {
        let numel = shape_numel(shape);
        let strides = compute_strides(shape);
        Ok(Self {
            id: TensorId::new(),
            shape: shape.to_vec(),
            strides,
            dtype,
            device_id,
            data_ptr: 0,
            numel,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            host_data: vec![0.0; numel],
        })
    }

    /// Create a new tensor filled with ones.
    pub fn ones(
        shape: &[usize],
        dtype: TensorDtype,
        device_id: usize,
    ) -> Result<Self, TensorError> {
        let numel = shape_numel(shape);
        let strides = compute_strides(shape);
        Ok(Self {
            id: TensorId::new(),
            shape: shape.to_vec(),
            strides,
            dtype,
            device_id,
            data_ptr: 0,
            numel,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            host_data: vec![1.0; numel],
        })
    }

    /// Create a tensor filled with a constant value.
    pub fn full(
        shape: &[usize],
        value: f64,
        dtype: TensorDtype,
        device_id: usize,
    ) -> Result<Self, TensorError> {
        let numel = shape_numel(shape);
        let strides = compute_strides(shape);
        Ok(Self {
            id: TensorId::new(),
            shape: shape.to_vec(),
            strides,
            dtype,
            device_id,
            data_ptr: 0,
            numel,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            host_data: vec![value; numel],
        })
    }

    /// Create a tensor from host f32 data and upload to the given device.
    pub fn from_host_f32(
        data: &[f32],
        shape: &[usize],
        device_id: usize,
    ) -> Result<Self, TensorError> {
        let numel = shape_numel(shape);
        if data.len() != numel {
            return Err(TensorError::ShapeMismatch {
                expected: numel,
                got: data.len(),
            });
        }
        let strides = compute_strides(shape);
        Ok(Self {
            id: TensorId::new(),
            shape: shape.to_vec(),
            strides,
            dtype: TensorDtype::Float32,
            device_id,
            data_ptr: 0,
            numel,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            host_data: data.iter().map(|&v| f64::from(v)).collect(),
        })
    }

    /// Create a tensor from host f64 data.
    pub fn from_host_f64(
        data: &[f64],
        shape: &[usize],
        device_id: usize,
    ) -> Result<Self, TensorError> {
        let numel = shape_numel(shape);
        if data.len() != numel {
            return Err(TensorError::ShapeMismatch {
                expected: numel,
                got: data.len(),
            });
        }
        let strides = compute_strides(shape);
        Ok(Self {
            id: TensorId::new(),
            shape: shape.to_vec(),
            strides,
            dtype: TensorDtype::Float64,
            device_id,
            data_ptr: 0,
            numel,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            host_data: data.to_vec(),
        })
    }

    /// Internal constructor from raw parts.
    pub(crate) fn from_parts(
        shape: Vec<usize>,
        dtype: TensorDtype,
        device_id: usize,
        host_data: Vec<f64>,
        requires_grad: bool,
        grad_fn: Option<GradFn>,
    ) -> Self {
        let numel = shape_numel(&shape);
        let strides = compute_strides(&shape);
        Self {
            id: TensorId::new(),
            shape,
            strides,
            dtype,
            device_id,
            data_ptr: 0,
            numel,
            requires_grad,
            grad: None,
            grad_fn,
            host_data,
        }
    }

    // ── Accessors ───────────────────────────────────────────

    /// Unique tensor ID.
    #[must_use]
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Shape of the tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Strides in elements.
    #[must_use]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Element data type.
    #[must_use]
    pub fn dtype(&self) -> TensorDtype {
        self.dtype
    }

    /// GPU device index.
    #[must_use]
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Opaque device pointer (0 when CPU-simulated).
    #[must_use]
    pub fn data_ptr(&self) -> u64 {
        self.data_ptr
    }

    /// Total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Whether this tensor has gradient tracking enabled.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Access the accumulated gradient, if any.
    #[must_use]
    pub fn grad(&self) -> Option<&GpuTensor> {
        self.grad.as_deref()
    }

    /// Access the grad function that created this tensor.
    #[must_use]
    pub fn grad_fn(&self) -> Option<&GradFn> {
        self.grad_fn.as_ref()
    }

    /// Read-only view of the host-side data (f64).
    #[must_use]
    pub fn host_data(&self) -> &[f64] {
        &self.host_data
    }

    // ── Mutation helpers ────────────────────────────────────

    /// Enable gradient tracking on this tensor.
    pub fn set_requires_grad(&mut self, requires: bool) {
        self.requires_grad = requires;
    }

    /// Set the grad_fn (called by ops when building the autograd graph).
    pub(crate) fn set_grad_fn(&mut self, grad_fn: GradFn) {
        self.grad_fn = Some(grad_fn);
    }

    /// Zero out the accumulated gradient.
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Accumulate a gradient into `.grad`.
    pub fn accumulate_grad(&mut self, grad: &GpuTensor) -> Result<(), TensorError> {
        if grad.shape() != self.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.numel,
                got: grad.numel,
            });
        }
        match &mut self.grad {
            Some(existing) => {
                for (a, b) in existing.host_data.iter_mut().zip(grad.host_data.iter()) {
                    *a += b;
                }
            }
            None => {
                self.grad = Some(Box::new(grad.clone()));
            }
        }
        Ok(())
    }

    // ── Shape operations ────────────────────────────────────

    /// Download tensor data to host as `Vec<f32>`.
    pub fn to_host_f32(&self) -> Vec<f32> {
        self.host_data.iter().map(|&v| v as f32).collect()
    }

    /// Download tensor data to host as `Vec<f64>`.
    pub fn to_host_f64(&self) -> Vec<f64> {
        self.host_data.clone()
    }

    /// Reshape the tensor to a new shape (total elements must match).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<GpuTensor, TensorError> {
        let new_numel = shape_numel(new_shape);
        if new_numel != self.numel {
            return Err(TensorError::ShapeMismatch {
                expected: self.numel,
                got: new_numel,
            });
        }
        let strides = compute_strides(new_shape);
        Ok(GpuTensor {
            id: TensorId::new(),
            shape: new_shape.to_vec(),
            strides,
            dtype: self.dtype,
            device_id: self.device_id,
            data_ptr: self.data_ptr,
            numel: self.numel,
            requires_grad: self.requires_grad,
            grad: None,
            grad_fn: None,
            host_data: self.host_data.clone(),
        })
    }

    /// Transpose two dimensions.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<GpuTensor, TensorError> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Err(TensorError::InvalidDimension {
                dim: dim0.max(dim1),
                ndim: self.ndim(),
            });
        }
        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);
        let mut new_strides = self.strides.clone();
        new_strides.swap(dim0, dim1);

        // Rearrange host data according to transpose
        let new_numel = self.numel;
        let mut new_data = vec![0.0; new_numel];
        let ndim = self.ndim();

        // Iterate over all indices in the original layout
        let mut idx = vec![0usize; ndim];
        let transposed_strides = compute_strides(&new_shape);
        for flat in 0..new_numel {
            // Compute multi-index from flat (row-major in original shape)
            let mut remaining = flat;
            for (idx_d, &stride) in idx.iter_mut().zip(self.strides.iter()) {
                *idx_d = remaining / stride;
                remaining %= stride;
            }
            // Compute flat index in transposed layout
            let mut transposed_idx = idx.clone();
            transposed_idx.swap(dim0, dim1);
            let new_flat: usize = transposed_idx
                .iter()
                .zip(transposed_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            new_data[new_flat] = self.host_data[flat];
        }

        Ok(GpuTensor {
            id: TensorId::new(),
            strides: compute_strides(&new_shape),
            shape: new_shape,
            dtype: self.dtype,
            device_id: self.device_id,
            data_ptr: 0,
            numel: new_numel,
            requires_grad: self.requires_grad,
            grad: None,
            grad_fn: None,
            host_data: new_data,
        })
    }

    /// Returns the contiguous version of this tensor.
    ///
    /// If the tensor is already contiguous, returns a clone. Otherwise,
    /// copies data into a new contiguous buffer.
    pub fn contiguous(&self) -> Result<GpuTensor, TensorError> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        // For CPU-simulated mode, the host_data is always "contiguous"
        // in the sense that it's a flat Vec. The strides are recomputed.
        let strides = compute_strides(&self.shape);
        Ok(GpuTensor {
            id: TensorId::new(),
            shape: self.shape.clone(),
            strides,
            dtype: self.dtype,
            device_id: self.device_id,
            data_ptr: 0,
            numel: self.numel,
            requires_grad: self.requires_grad,
            grad: None,
            grad_fn: None,
            host_data: self.host_data.clone(),
        })
    }

    /// Check whether the tensor has a standard row-major contiguous layout.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.strides == compute_strides(&self.shape)
    }

    /// Detach from the autograd computation graph.
    ///
    /// Returns a new tensor with the same data but no grad_fn or grad tracking.
    pub fn detach(&self) -> GpuTensor {
        GpuTensor {
            id: TensorId::new(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            device_id: self.device_id,
            data_ptr: self.data_ptr,
            numel: self.numel,
            requires_grad: false,
            grad: None,
            grad_fn: None,
            host_data: self.host_data.clone(),
        }
    }

    /// Return a scalar value for a 0-d or 1-element tensor.
    pub fn item(&self) -> Result<f64, TensorError> {
        if self.numel != 1 {
            return Err(TensorError::InvalidOperation(format!(
                "item() requires 1 element, got {}",
                self.numel
            )));
        }
        self.host_data
            .first()
            .copied()
            .ok_or_else(|| TensorError::InvalidOperation("empty tensor".into()))
    }

    /// Get element at a flat index.
    pub fn get_flat(&self, index: usize) -> Result<f64, TensorError> {
        if index >= self.numel {
            return Err(TensorError::IndexOutOfBounds {
                index,
                size: self.numel,
            });
        }
        Ok(self.host_data[index])
    }

    /// Set element at a flat index.
    pub fn set_flat(&mut self, index: usize, value: f64) -> Result<(), TensorError> {
        if index >= self.numel {
            return Err(TensorError::IndexOutOfBounds {
                index,
                size: self.numel,
            });
        }
        self.host_data[index] = value;
        Ok(())
    }

    /// Size along a given dimension.
    pub fn size(&self, dim: usize) -> Result<usize, TensorError> {
        self.shape
            .get(dim)
            .copied()
            .ok_or(TensorError::InvalidDimension {
                dim,
                ndim: self.ndim(),
            })
    }

    /// Total size in bytes of the underlying data.
    #[must_use]
    pub fn nbytes(&self) -> usize {
        self.numel * self.dtype.size_bytes()
    }
}

impl fmt::Display for GpuTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GpuTensor(shape={:?}, dtype={}, device={}, requires_grad={})",
            self.shape, self.dtype, self.device_id, self.requires_grad
        )
    }
}

// ─── Helper functions ───────────────────────────────────────

/// Compute total number of elements from a shape.
#[must_use]
pub fn shape_numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        0
    } else {
        shape.iter().product()
    }
}

/// Compute row-major strides from a shape.
#[must_use]
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        let t = GpuTensor::zeros(&[2, 3], TensorDtype::Float32, 0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
        assert!(t.host_data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_tensor_ones() {
        let t = GpuTensor::ones(&[4], TensorDtype::Float32, 0).unwrap();
        assert_eq!(t.numel(), 4);
        assert!(t.host_data.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_tensor_from_host_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = GpuTensor::from_host_f32(&data, &[2, 3], 0).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert!((t.host_data[0] - 1.0).abs() < 1e-7);
        assert!((t.host_data[5] - 6.0).abs() < 1e-7);
    }

    #[test]
    fn test_tensor_from_host_shape_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0];
        let res = GpuTensor::from_host_f32(&data, &[2, 3], 0);
        assert!(res.is_err());
    }

    #[test]
    fn test_tensor_reshape() {
        let t = GpuTensor::zeros(&[2, 3], TensorDtype::Float32, 0).unwrap();
        let r = t.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.numel(), 6);
    }

    #[test]
    fn test_tensor_reshape_mismatch() {
        let t = GpuTensor::zeros(&[2, 3], TensorDtype::Float32, 0).unwrap();
        assert!(t.reshape(&[4, 2]).is_err());
    }

    #[test]
    fn test_tensor_transpose() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = GpuTensor::from_host_f32(&data, &[2, 3], 0).unwrap();
        let tr = t.transpose(0, 1).unwrap();
        assert_eq!(tr.shape(), &[3, 2]);
        // Original: [[1,2,3],[4,5,6]] -> Transposed: [[1,4],[2,5],[3,6]]
        let expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        for (a, b) in tr.host_data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_tensor_contiguous() {
        let t = GpuTensor::zeros(&[2, 3], TensorDtype::Float32, 0).unwrap();
        assert!(t.is_contiguous());
        let c = t.contiguous().unwrap();
        assert!(c.is_contiguous());
    }

    #[test]
    fn test_tensor_detach() {
        let mut t = GpuTensor::zeros(&[2, 3], TensorDtype::Float32, 0).unwrap();
        t.set_requires_grad(true);
        let d = t.detach();
        assert!(!d.requires_grad());
        assert!(d.grad_fn().is_none());
    }

    #[test]
    fn test_tensor_item() {
        let t = GpuTensor::full(&[1], 42.0, TensorDtype::Float32, 0).unwrap();
        assert!((t.item().unwrap() - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_item_not_scalar() {
        let t = GpuTensor::zeros(&[2, 3], TensorDtype::Float32, 0).unwrap();
        assert!(t.item().is_err());
    }

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert_eq!(compute_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_shape_numel() {
        assert_eq!(shape_numel(&[2, 3, 4]), 24);
        assert_eq!(shape_numel(&[1]), 1);
        assert_eq!(shape_numel(&[]), 0);
    }

    #[test]
    fn test_tensor_display() {
        let t = GpuTensor::zeros(&[2, 3], TensorDtype::Float32, 0).unwrap();
        let s = format!("{t}");
        assert!(s.contains("GpuTensor"));
        assert!(s.contains("[2, 3]"));
    }

    #[test]
    fn test_accumulate_grad() {
        let mut t = GpuTensor::zeros(&[2], TensorDtype::Float32, 0).unwrap();
        t.set_requires_grad(true);
        let g1 = GpuTensor::from_host_f64(&[1.0, 2.0], &[2], 0).unwrap();
        let g2 = GpuTensor::from_host_f64(&[3.0, 4.0], &[2], 0).unwrap();
        t.accumulate_grad(&g1).unwrap();
        t.accumulate_grad(&g2).unwrap();
        let grad = t.grad().unwrap();
        assert!((grad.host_data[0] - 4.0).abs() < 1e-10);
        assert!((grad.host_data[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_set_flat() {
        let mut t = GpuTensor::zeros(&[3], TensorDtype::Float32, 0).unwrap();
        t.set_flat(1, 5.0).unwrap();
        assert!((t.get_flat(1).unwrap() - 5.0).abs() < 1e-10);
        assert!(t.get_flat(10).is_err());
        assert!(t.set_flat(10, 1.0).is_err());
    }

    #[test]
    fn test_nbytes() {
        let t = GpuTensor::zeros(&[100], TensorDtype::Float32, 0).unwrap();
        assert_eq!(t.nbytes(), 400);
        let t64 = GpuTensor::zeros(&[100], TensorDtype::Float64, 0).unwrap();
        assert_eq!(t64.nbytes(), 800);
    }

    #[test]
    fn test_tensor_to_host() {
        let data = vec![1.0f32, 2.0, 3.0];
        let t = GpuTensor::from_host_f32(&data, &[3], 0).unwrap();
        let f32_data = t.to_host_f32();
        assert!((f32_data[0] - 1.0).abs() < 1e-6);
        assert!((f32_data[2] - 3.0).abs() < 1e-6);
    }
}
