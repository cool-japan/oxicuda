//! Type-safe buffer reinterpretation for device memory.
//!
//! This module provides [`BufferView`] and [`BufferViewMut`], which allow
//! reinterpreting a [`DeviceBuffer<T>`] as a different element type `U`
//! without copying data. This is useful for viewing a buffer of `f32`
//! values as `u32` (e.g., for bitwise operations in a kernel), or for
//! interpreting raw byte buffers as structured types.
//!
//! # Size constraints
//!
//! The total byte size of the original buffer must be evenly divisible
//! by `std::mem::size_of::<U>()`. If not, the view creation returns
//! [`CudaError::InvalidValue`].
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_memory::DeviceBuffer;
//! # use oxicuda_memory::buffer_view::BufferView;
//! let buf = DeviceBuffer::<f32>::alloc(256)?;
//! // Reinterpret as u32 (same size, different type)
//! let view: BufferView<'_, u32> = buf.view_as::<u32>()?;
//! assert_eq!(view.len(), 256);
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::marker::PhantomData;

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::ffi::CUdeviceptr;

use crate::device_buffer::DeviceBuffer;

// ---------------------------------------------------------------------------
// BufferView<'a, U>
// ---------------------------------------------------------------------------

/// An immutable, type-reinterpreted view into a [`DeviceBuffer`].
///
/// This struct borrows the underlying device allocation and exposes it
/// as a different element type `U`. No data is copied; only the
/// pointer arithmetic changes.
///
/// The view is lifetime-bound to the original buffer, preventing use
/// after the buffer is freed.
pub struct BufferView<'a, U: Copy> {
    /// Device pointer to the start of the buffer.
    ptr: CUdeviceptr,
    /// Number of `U` elements in the reinterpreted view.
    len: usize,
    /// Ties the lifetime to the parent buffer.
    _phantom: PhantomData<&'a U>,
}

impl<U: Copy> BufferView<'_, U> {
    /// Returns the number of `U` elements in this view.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the view contains zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the total byte size of this view.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<U>()
    }

    /// Returns the raw [`CUdeviceptr`] for this view.
    #[inline]
    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

impl<U: Copy> std::fmt::Debug for BufferView<'_, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferView")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("elem_size", &std::mem::size_of::<U>())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// BufferViewMut<'a, U>
// ---------------------------------------------------------------------------

/// A mutable, type-reinterpreted view into a [`DeviceBuffer`].
///
/// Like [`BufferView`] but allows mutable operations (e.g., passing
/// to a kernel that writes through this reinterpreted pointer).
pub struct BufferViewMut<'a, U: Copy> {
    /// Device pointer to the start of the buffer.
    ptr: CUdeviceptr,
    /// Number of `U` elements in the reinterpreted view.
    len: usize,
    /// Ties the lifetime to the parent buffer (mutable borrow).
    _phantom: PhantomData<&'a mut U>,
}

impl<U: Copy> BufferViewMut<'_, U> {
    /// Returns the number of `U` elements in this view.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the view contains zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the total byte size of this view.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<U>()
    }

    /// Returns the raw [`CUdeviceptr`] for this view.
    #[inline]
    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

impl<U: Copy> std::fmt::Debug for BufferViewMut<'_, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferViewMut")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("elem_size", &std::mem::size_of::<U>())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// DeviceBuffer extensions
// ---------------------------------------------------------------------------

impl<T: Copy> DeviceBuffer<T> {
    /// Reinterprets this buffer as a different element type `U` (immutable).
    ///
    /// The total byte size of the buffer must be evenly divisible by
    /// `size_of::<U>()`. The resulting view has `byte_size / size_of::<U>()`
    /// elements.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if:
    /// - `size_of::<U>()` is zero (ZST).
    /// - The buffer's byte size is not divisible by `size_of::<U>()`.
    pub fn view_as<U: Copy>(&self) -> CudaResult<BufferView<'_, U>> {
        let u_size = std::mem::size_of::<U>();
        if u_size == 0 {
            return Err(CudaError::InvalidValue);
        }
        let byte_size = self.byte_size();
        if byte_size % u_size != 0 {
            return Err(CudaError::InvalidValue);
        }
        Ok(BufferView {
            ptr: self.as_device_ptr(),
            len: byte_size / u_size,
            _phantom: PhantomData,
        })
    }

    /// Reinterprets this buffer as a different element type `U` (mutable).
    ///
    /// The total byte size of the buffer must be evenly divisible by
    /// `size_of::<U>()`. The resulting view has `byte_size / size_of::<U>()`
    /// elements.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if:
    /// - `size_of::<U>()` is zero (ZST).
    /// - The buffer's byte size is not divisible by `size_of::<U>()`.
    pub fn view_as_mut<U: Copy>(&mut self) -> CudaResult<BufferViewMut<'_, U>> {
        let u_size = std::mem::size_of::<U>();
        if u_size == 0 {
            return Err(CudaError::InvalidValue);
        }
        let byte_size = self.byte_size();
        if byte_size % u_size != 0 {
            return Err(CudaError::InvalidValue);
        }
        Ok(BufferViewMut {
            ptr: self.as_device_ptr(),
            len: byte_size / u_size,
            _phantom: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_view_debug() {
        let view: BufferView<'_, u32> = BufferView {
            ptr: 0x1000,
            len: 64,
            _phantom: PhantomData,
        };
        let dbg = format!("{view:?}");
        assert!(dbg.contains("BufferView"));
        assert!(dbg.contains("64"));
    }

    #[test]
    fn buffer_view_mut_debug() {
        let view: BufferViewMut<'_, f32> = BufferViewMut {
            ptr: 0x2000,
            len: 128,
            _phantom: PhantomData,
        };
        let dbg = format!("{view:?}");
        assert!(dbg.contains("BufferViewMut"));
        assert!(dbg.contains("128"));
    }

    #[test]
    fn buffer_view_len_and_byte_size() {
        let view: BufferView<'_, u64> = BufferView {
            ptr: 0x3000,
            len: 32,
            _phantom: PhantomData,
        };
        assert_eq!(view.len(), 32);
        assert_eq!(view.byte_size(), 32 * 8);
        assert!(!view.is_empty());
        assert_eq!(view.as_device_ptr(), 0x3000);
    }

    #[test]
    fn buffer_view_mut_len_and_byte_size() {
        let view: BufferViewMut<'_, u16> = BufferViewMut {
            ptr: 0x4000,
            len: 100,
            _phantom: PhantomData,
        };
        assert_eq!(view.len(), 100);
        assert_eq!(view.byte_size(), 200);
        assert!(!view.is_empty());
        assert_eq!(view.as_device_ptr(), 0x4000);
    }

    #[test]
    fn buffer_view_empty() {
        let view: BufferView<'_, f64> = BufferView {
            ptr: 0,
            len: 0,
            _phantom: PhantomData,
        };
        assert!(view.is_empty());
        assert_eq!(view.byte_size(), 0);
    }

    #[test]
    fn view_as_signature_compiles() {
        let _: fn(&DeviceBuffer<f32>) -> CudaResult<BufferView<'_, u32>> = DeviceBuffer::view_as;
    }

    #[test]
    fn view_as_mut_signature_compiles() {
        let _: fn(&mut DeviceBuffer<f32>) -> CudaResult<BufferViewMut<'_, u32>> =
            DeviceBuffer::view_as_mut;
    }
}
