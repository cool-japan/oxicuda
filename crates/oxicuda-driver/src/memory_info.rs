//! Safe wrappers for GPU memory information and bulk memory operations.
//!
//! This module provides convenient functions for querying device memory
//! usage and performing device-to-device copies and memsets.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_driver::memory_info;
//! # fn main() -> Result<(), oxicuda_driver::CudaError> {
//! let (free, total) = memory_info::device_memory_info()?;
//! println!("GPU memory: {free} / {total} bytes free");
//! # Ok(())
//! # }
//! ```

use crate::error::{CudaError, CudaResult};
use crate::ffi::CUdeviceptr;
use crate::loader::try_driver;
use crate::stream::Stream;

/// Returns the amount of free and total device memory in bytes.
///
/// This queries the current context's device for its memory utilisation.
///
/// # Returns
///
/// A tuple `(free_bytes, total_bytes)`.
///
/// # Errors
///
/// Returns a [`CudaError`] if no context is current or the driver call fails.
pub fn device_memory_info() -> CudaResult<(usize, usize)> {
    let api = try_driver()?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    crate::cuda_call!((api.cu_mem_get_info_v2)(&mut free, &mut total))?;
    Ok((free, total))
}

/// Copies `bytes` bytes from one device pointer to another.
///
/// Both `src` and `dst` must point to valid device memory allocations of
/// at least `bytes` bytes.
///
/// # Errors
///
/// Returns a [`CudaError`] if the copy fails.
pub fn memcpy_device_to_device(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> CudaResult<()> {
    let api = try_driver()?;
    crate::cuda_call!((api.cu_memcpy_dtod_v2)(dst, src, bytes))
}

/// Asynchronously copies `bytes` bytes from one device pointer to another.
///
/// The copy is enqueued on the given `stream` and returns immediately.
///
/// # Errors
///
/// Returns [`CudaError::NotSupported`] if the driver does not export the
/// async DtoD copy entry point, or another [`CudaError`] on failure.
pub fn memcpy_device_to_device_async(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    bytes: usize,
    stream: &Stream,
) -> CudaResult<()> {
    let api = try_driver()?;
    let f = api.cu_memcpy_dtod_async_v2.ok_or(CudaError::NotSupported)?;
    crate::cuda_call!(f(dst, src, bytes, stream.raw()))
}

/// Sets `count` 32-bit elements starting at `ptr` to `value`.
///
/// # Errors
///
/// Returns a [`CudaError`] if the memset fails.
pub fn memset_d32(ptr: CUdeviceptr, value: u32, count: usize) -> CudaResult<()> {
    let api = try_driver()?;
    crate::cuda_call!((api.cu_memset_d32_v2)(ptr, value, count))
}

/// Asynchronously sets `count` 32-bit elements starting at `ptr` to `value`.
///
/// The operation is enqueued on the given `stream`.
///
/// # Errors
///
/// Returns [`CudaError::NotSupported`] if the driver does not export the
/// async memset entry point, or another [`CudaError`] on failure.
pub fn memset_d32_async(
    ptr: CUdeviceptr,
    value: u32,
    count: usize,
    stream: &Stream,
) -> CudaResult<()> {
    let api = try_driver()?;
    let f = api.cu_memset_d32_async.ok_or(CudaError::NotSupported)?;
    crate::cuda_call!(f(ptr, value, count, stream.raw()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_memory_info_returns_error_without_gpu() {
        // On macOS or systems without a GPU, this should return an error
        // rather than panicking.
        let result = device_memory_info();
        // We just verify it does not panic; on CI without a GPU it will be Err.
        let _ = result;
    }

    #[test]
    fn memcpy_dtod_returns_error_without_gpu() {
        let result = memcpy_device_to_device(0, 0, 0);
        let _ = result;
    }

    #[test]
    fn memset_d32_returns_error_without_gpu() {
        let result = memset_d32(0, 0, 0);
        let _ = result;
    }
}
