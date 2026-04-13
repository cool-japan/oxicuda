//! Tensor shape extraction utilities.
//!
//! Provides helper functions for extracting NCHW dimensions from the
//! generic Vec-based [`TensorDesc`] and [`TensorDescMut`] descriptors.

use oxicuda_blas::GpuFloat;

use crate::error::{DnnError, DnnResult};
use crate::types::{TensorDesc, TensorDescMut};

/// Extracts `(n, c, h, w)` from a 4-D tensor descriptor.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if the tensor is not 4-D.
pub(crate) fn nchw_dims<T: GpuFloat>(desc: &TensorDesc<T>) -> DnnResult<(u32, u32, u32, u32)> {
    if desc.dims.len() != 4 {
        return Err(DnnError::InvalidDimension(format!(
            "expected 4-D tensor, got {}-D",
            desc.dims.len()
        )));
    }
    Ok((desc.dims[0], desc.dims[1], desc.dims[2], desc.dims[3]))
}

/// Extracts `(n, c, h, w)` from a mutable 4-D tensor descriptor.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if the tensor is not 4-D.
pub(crate) fn nchw_dims_mut<T: GpuFloat>(
    desc: &TensorDescMut<T>,
) -> DnnResult<(u32, u32, u32, u32)> {
    if desc.dims.len() != 4 {
        return Err(DnnError::InvalidDimension(format!(
            "expected 4-D tensor, got {}-D",
            desc.dims.len()
        )));
    }
    Ok((desc.dims[0], desc.dims[1], desc.dims[2], desc.dims[3]))
}

/// Extracts `(batch, heads, seq_len, head_dim)` from a 4-D attention tensor.
///
/// This is an alias for [`nchw_dims`] with attention-specific semantics:
/// `dims[0]` = batch, `dims[1]` = heads, `dims[2]` = seq_len, `dims[3]` = head_dim.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if the tensor is not 4-D.
pub(crate) fn attn_dims<T: GpuFloat>(desc: &TensorDesc<T>) -> DnnResult<(u32, u32, u32, u32)> {
    nchw_dims(desc)
}

/// Extracts `(batch, heads, seq_len, head_dim)` from a mutable 4-D attention tensor.
///
/// # Errors
///
/// Returns [`DnnError::InvalidDimension`] if the tensor is not 4-D.
pub(crate) fn attn_dims_mut<T: GpuFloat>(
    desc: &TensorDescMut<T>,
) -> DnnResult<(u32, u32, u32, u32)> {
    nchw_dims_mut(desc)
}
