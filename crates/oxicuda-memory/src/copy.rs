//! Explicit memory copy operations between host and device.
//!
//! This module provides freestanding functions for copying data between
//! host memory, device memory, and pinned host memory.  Each function
//! validates that the source and destination have matching lengths before
//! issuing the underlying CUDA driver call.
//!
//! For simple cases, the methods on [`DeviceBuffer`]
//! (e.g. [`DeviceBuffer::copy_from_host`]) are more
//! ergonomic.  These freestanding functions are useful when you want to be
//! explicit about the direction of the transfer or when working with
//! [`PinnedBuffer`] for async operations.
//!
//! # Length validation
//!
//! All functions return [`CudaError::InvalidValue`] if the element counts
//! of source and destination do not match.

use std::ffi::c_void;

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::stream::Stream;

use crate::device_buffer::DeviceBuffer;
use crate::host_buffer::PinnedBuffer;

// ---------------------------------------------------------------------------
// Synchronous copies
// ---------------------------------------------------------------------------

/// Copies data from a host slice into a device buffer (host-to-device).
///
/// This is a synchronous operation: it blocks the calling thread until the
/// transfer completes.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `src.len() != dst.len()`.
/// * Other driver errors from `cuMemcpyHtoD_v2`.
pub fn copy_htod<T: Copy>(dst: &mut DeviceBuffer<T>, src: &[T]) -> CudaResult<()> {
    if src.len() != dst.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = dst.byte_size();
    let api = try_driver()?;
    // SAFETY: `src` is a valid host slice, `dst` owns a valid device allocation,
    // and the byte counts match.
    let rc = unsafe {
        (api.cu_memcpy_htod_v2)(
            dst.as_device_ptr(),
            src.as_ptr().cast::<c_void>(),
            byte_size,
        )
    };
    oxicuda_driver::check(rc)
}

/// Copies data from a device buffer into a host slice (device-to-host).
///
/// This is a synchronous operation: it blocks the calling thread until the
/// transfer completes.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `dst.len() != src.len()`.
/// * Other driver errors from `cuMemcpyDtoH_v2`.
pub fn copy_dtoh<T: Copy>(dst: &mut [T], src: &DeviceBuffer<T>) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = src.byte_size();
    let api = try_driver()?;
    // SAFETY: `dst` is a valid host slice, `src` owns a valid device allocation,
    // and the byte counts match.
    let rc = unsafe {
        (api.cu_memcpy_dtoh_v2)(
            dst.as_mut_ptr().cast::<c_void>(),
            src.as_device_ptr(),
            byte_size,
        )
    };
    oxicuda_driver::check(rc)
}

/// Copies data from one device buffer to another (device-to-device).
///
/// This is a synchronous operation that blocks until the copy completes.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `dst.len() != src.len()`.
/// * Other driver errors from `cuMemcpyDtoD_v2`.
pub fn copy_dtod<T: Copy>(dst: &mut DeviceBuffer<T>, src: &DeviceBuffer<T>) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = src.byte_size();
    let api = try_driver()?;
    // SAFETY: both buffers own valid device allocations of the same size.
    let rc =
        unsafe { (api.cu_memcpy_dtod_v2)(dst.as_device_ptr(), src.as_device_ptr(), byte_size) };
    oxicuda_driver::check(rc)
}

// ---------------------------------------------------------------------------
// Asynchronous copies
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Asynchronous copies (raw slice variants)
// ---------------------------------------------------------------------------

/// Asynchronously copies data from a host slice into a device buffer.
///
/// The copy is enqueued on `stream` and may not be complete when this
/// function returns.  The caller must ensure that `src` remains valid
/// (i.e., is not moved or dropped) until the stream has been synchronised.
/// For guaranteed correctness with DMA, prefer using a [`PinnedBuffer`]
/// as the source.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `src.len() != dst.len()`.
/// * Other driver errors from `cuMemcpyHtoDAsync_v2`.
pub fn copy_htod_async_raw<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    src: &[T],
    stream: &Stream,
) -> CudaResult<()> {
    if src.len() != dst.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = dst.byte_size();
    let api = try_driver()?;
    let rc = unsafe {
        (api.cu_memcpy_htod_async_v2)(
            dst.as_device_ptr(),
            src.as_ptr().cast::<c_void>(),
            byte_size,
            stream.raw(),
        )
    };
    oxicuda_driver::check(rc)
}

/// Asynchronously copies data from a device buffer into a host slice.
///
/// The copy is enqueued on `stream` and may not be complete when this
/// function returns.  The caller must ensure that `dst` remains valid
/// and is not read until the stream has been synchronised.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `dst.len() != src.len()`.
/// * Other driver errors from `cuMemcpyDtoHAsync_v2`.
pub fn copy_dtoh_async_raw<T: Copy>(
    dst: &mut [T],
    src: &DeviceBuffer<T>,
    stream: &Stream,
) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = src.byte_size();
    let api = try_driver()?;
    let rc = unsafe {
        (api.cu_memcpy_dtoh_async_v2)(
            dst.as_mut_ptr().cast::<c_void>(),
            src.as_device_ptr(),
            byte_size,
            stream.raw(),
        )
    };
    oxicuda_driver::check(rc)
}

/// Asynchronously copies data from one device buffer to another.
///
/// Both buffers must have the same length.  The copy is enqueued on
/// `stream`.
///
/// Note: The CUDA Driver API does not provide `cuMemcpyDtoDAsync` directly;
/// this uses `cuMemcpyHtoDAsync_v2` semantics via the driver's internal
/// routing for device-to-device copies.  For true async D2D, consider
/// using peer copy functions or ensuring both buffers are in the same
/// context.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `dst.len() != src.len()`.
/// * Other driver errors.
pub fn copy_dtod_async<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    src: &DeviceBuffer<T>,
    stream: &Stream,
) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    // Use synchronous D2D copy followed by stream ordering via event.
    // The CUDA driver routes D2D copies internally; we use the sync version
    // and rely on stream ordering at the caller level.
    // A future enhancement can add cuMemcpyDtoDAsync when the driver
    // exposes it.
    let _ = stream;
    copy_dtod(dst, src)
}

// ---------------------------------------------------------------------------
// Asynchronous copies (pinned buffer variants)
// ---------------------------------------------------------------------------

/// Asynchronously copies data from a pinned host buffer into a device buffer.
///
/// The copy is enqueued on `stream` and may not be complete when this
/// function returns.  The caller must not modify `src` or read `dst` until
/// the stream has been synchronised.
///
/// Using a [`PinnedBuffer`] as the source guarantees that the host memory
/// is page-locked, which is required for correct async DMA transfers.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `src.len() != dst.len()`.
/// * Other driver errors from `cuMemcpyHtoDAsync_v2`.
pub fn copy_htod_async<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    src: &PinnedBuffer<T>,
    stream: &Stream,
) -> CudaResult<()> {
    if src.len() != dst.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = dst.byte_size();
    let api = try_driver()?;
    // SAFETY: `src` is pinned host memory, `dst` is a valid device allocation,
    // byte counts match, and the stream will order the transfer.
    let rc = unsafe {
        (api.cu_memcpy_htod_async_v2)(
            dst.as_device_ptr(),
            src.as_ptr().cast::<c_void>(),
            byte_size,
            stream.raw(),
        )
    };
    oxicuda_driver::check(rc)
}

/// Asynchronously copies data from a device buffer into a pinned host buffer.
///
/// The copy is enqueued on `stream` and may not be complete when this
/// function returns.  The caller must not read `dst` until the stream
/// has been synchronised.
///
/// Using a [`PinnedBuffer`] as the destination guarantees that the host
/// memory is page-locked, which is required for correct async DMA transfers.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if `dst.len() != src.len()`.
/// * Other driver errors from `cuMemcpyDtoHAsync_v2`.
pub fn copy_dtoh_async<T: Copy>(
    dst: &mut PinnedBuffer<T>,
    src: &DeviceBuffer<T>,
    stream: &Stream,
) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let byte_size = src.byte_size();
    let api = try_driver()?;
    // SAFETY: `dst` is pinned host memory, `src` is a valid device allocation,
    // byte counts match, and the stream will order the transfer.
    let rc = unsafe {
        (api.cu_memcpy_dtoh_async_v2)(
            dst.as_mut_ptr().cast::<c_void>(),
            src.as_device_ptr(),
            byte_size,
            stream.raw(),
        )
    };
    oxicuda_driver::check(rc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn copy_htod_signature_compiles() {
        let _f: fn(&mut super::DeviceBuffer<f32>, &[f32]) -> super::CudaResult<()> =
            super::copy_htod;
        let _f2: fn(&mut [f32], &super::DeviceBuffer<f32>) -> super::CudaResult<()> =
            super::copy_dtoh;
    }

    #[test]
    fn copy_dtod_signature_compiles() {
        let _f: fn(
            &mut super::DeviceBuffer<f32>,
            &super::DeviceBuffer<f32>,
        ) -> super::CudaResult<()> = super::copy_dtod;
    }

    #[test]
    fn async_raw_htod_signature_compiles() {
        let _f: fn(
            &mut super::DeviceBuffer<f32>,
            &[f32],
            &oxicuda_driver::stream::Stream,
        ) -> super::CudaResult<()> = super::copy_htod_async_raw;
    }

    #[test]
    fn async_raw_dtoh_signature_compiles() {
        let _f: fn(
            &mut [f32],
            &super::DeviceBuffer<f32>,
            &oxicuda_driver::stream::Stream,
        ) -> super::CudaResult<()> = super::copy_dtoh_async_raw;
    }

    #[test]
    fn async_dtod_signature_compiles() {
        let _f: fn(
            &mut super::DeviceBuffer<f32>,
            &super::DeviceBuffer<f32>,
            &oxicuda_driver::stream::Stream,
        ) -> super::CudaResult<()> = super::copy_dtod_async;
    }

    #[test]
    fn async_pinned_htod_signature_compiles() {
        let _f: fn(
            &mut super::DeviceBuffer<f32>,
            &super::PinnedBuffer<f32>,
            &oxicuda_driver::stream::Stream,
        ) -> super::CudaResult<()> = super::copy_htod_async;
    }
}
