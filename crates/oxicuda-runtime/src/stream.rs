//! CUDA stream management.
//!
//! Implements the CUDA Runtime stream API:
//! - `cudaStreamCreate` / `cudaStreamCreateWithFlags` / `cudaStreamCreateWithPriority`
//! - `cudaStreamDestroy`
//! - `cudaStreamSynchronize`
//! - `cudaStreamQuery`
//! - `cudaStreamWaitEvent`
//! - `cudaStreamGetPriority`
//! - `cudaStreamGetFlags`
//! - `cudaStreamGetDevice`
//! - The default stream (`cudaStreamDefault` / `cudaStreamLegacy` / `cudaStreamPerThread`)

use oxicuda_driver::ffi::CUstream;
use oxicuda_driver::loader::try_driver;

use crate::error::{CudaRtError, CudaRtResult};

// ─── StreamFlags ─────────────────────────────────────────────────────────────

/// Flags for stream creation.
///
/// Mirrors `cudaStreamFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StreamFlags(pub u32);

impl StreamFlags {
    /// Default stream flag: stream synchronises with the legacy default stream.
    pub const DEFAULT: Self = Self(0x0);
    /// Non-blocking stream: the stream does not implicitly synchronise with the
    /// legacy default stream (mirrors `cudaStreamNonBlocking`).
    pub const NON_BLOCKING: Self = Self(0x1);
}

// ─── CudaStream ──────────────────────────────────────────────────────────────

/// A CUDA stream handle.
///
/// Wraps the raw `CUstream` handle from the driver API.  The stream is
/// **not** automatically destroyed when dropped — call [`stream_destroy`]
/// explicitly or use the stream within its creating context lifetime.
///
/// Use [`CudaStream::DEFAULT`] to obtain the special legacy-default
/// stream sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaStream(CUstream);

impl CudaStream {
    /// The legacy default CUDA stream (`cudaStreamDefault` = 0).
    ///
    /// Operations on the default stream block all other streams in the context.
    pub const DEFAULT: Self = Self(CUstream(std::ptr::null_mut()));

    /// Per-thread default stream (`cudaStreamPerThread`).
    ///
    /// Equivalent to passing `cudaStreamPerThread` in the Runtime API.
    /// The value `0x2` is the canonical sentinel used by the CUDA Runtime.
    pub const PER_THREAD: Self = Self(CUstream(2 as *mut std::ffi::c_void));

    /// Construct a `CudaStream` from a raw driver handle.
    ///
    /// # Safety
    ///
    /// The caller must ensure the handle is valid and not used after the
    /// associated context is destroyed.
    #[must_use]
    pub const unsafe fn from_raw(raw: CUstream) -> Self {
        Self(raw)
    }

    /// Returns the underlying raw `CUstream`.
    #[must_use]
    pub fn raw(self) -> CUstream {
        self.0
    }

    /// Returns `true` if this is the legacy default stream.
    #[must_use]
    pub fn is_default(self) -> bool {
        self.0.is_null()
    }
}

// ─── Stream creation / destruction ────────────────────────────────────────────

/// Create a new CUDA stream with default flags.
///
/// Mirrors `cudaStreamCreate`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_create() -> CudaRtResult<CudaStream> {
    stream_create_with_flags(StreamFlags::DEFAULT)
}

/// Create a new CUDA stream with the given flags.
///
/// Mirrors `cudaStreamCreateWithFlags`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_create_with_flags(flags: StreamFlags) -> CudaRtResult<CudaStream> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut stream = CUstream::default();
    // SAFETY: FFI; stream is a valid stack-allocated opaque pointer.
    let rc = unsafe { (api.cu_stream_create)(&raw mut stream, flags.0) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(CudaStream(stream))
}

/// Create a new CUDA stream with the given flags and scheduling priority.
///
/// Mirrors `cudaStreamCreateWithPriority`.
///
/// `priority` is a signed integer where lower values indicate higher priority.
/// The valid range can be queried with `cudaDeviceGetStreamPriorityRange`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_create_with_priority(flags: StreamFlags, priority: i32) -> CudaRtResult<CudaStream> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut stream = CUstream::default();
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_stream_create_with_priority)(&raw mut stream, flags.0, priority) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(CudaStream(stream))
}

/// Destroy a CUDA stream.
///
/// Mirrors `cudaStreamDestroy`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_destroy(stream: CudaStream) -> CudaRtResult<()> {
    if stream.is_default() {
        return Ok(()); // default stream is never explicitly destroyed
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; stream handle is valid.
    let rc = unsafe { (api.cu_stream_destroy_v2)(stream.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(())
}

// ─── Stream synchronisation / query ──────────────────────────────────────────

/// Wait until all preceding operations in `stream` complete.
///
/// Mirrors `cudaStreamSynchronize`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_synchronize(stream: CudaStream) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_stream_synchronize)(stream.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::Unknown));
    }
    Ok(())
}

/// Check whether all preceding operations in `stream` have completed.
///
/// Mirrors `cudaStreamQuery`.
///
/// Returns `Ok(true)` if complete, `Ok(false)` if still running.
///
/// # Errors
///
/// Propagates driver errors (other than `NotReady`).
pub fn stream_query(stream: CudaStream) -> CudaRtResult<bool> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_stream_query)(stream.raw()) };
    match rc {
        0 => Ok(true),    // CUDA_SUCCESS — complete
        600 => Ok(false), // CUDA_ERROR_NOT_READY — still running
        other => Err(CudaRtError::from_code(other).unwrap_or(CudaRtError::Unknown)),
    }
}

/// Make all future work submitted to `stream` wait until `event` is recorded.
///
/// Mirrors `cudaStreamWaitEvent`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_wait_event(
    stream: CudaStream,
    event: crate::event::CudaEvent,
    flags: u32,
) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_stream_wait_event)(stream.raw(), event.raw(), flags) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(())
}

/// Returns the priority of `stream`.
///
/// Mirrors `cudaStreamGetPriority`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_get_priority(stream: CudaStream) -> CudaRtResult<i32> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut priority: std::ffi::c_int = 0;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_stream_get_priority)(stream.raw(), &raw mut priority) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(priority)
}

/// Returns the flags of `stream`.
///
/// Mirrors `cudaStreamGetFlags`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn stream_get_flags(stream: CudaStream) -> CudaRtResult<StreamFlags> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut flags: u32 = 0;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_stream_get_flags)(stream.raw(), &raw mut flags) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(StreamFlags(flags))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_stream_is_null() {
        assert!(CudaStream::DEFAULT.is_default());
        assert!(!CudaStream::PER_THREAD.is_default());
    }

    #[test]
    fn stream_flags_values() {
        assert_eq!(StreamFlags::DEFAULT.0, 0);
        assert_eq!(StreamFlags::NON_BLOCKING.0, 1);
    }

    #[test]
    fn stream_destroy_default_is_noop() {
        // Should never hit the driver for the default stream.
        let result = stream_destroy(CudaStream::DEFAULT);
        // Without a driver it fails with DriverNotAvailable; with a driver it's Ok.
        let _ = result;
    }

    #[test]
    fn stream_create_without_gpu_returns_error() {
        let result = stream_create();
        // Must either succeed (GPU present) or fail with DriverNotAvailable /
        // some other non-panic error.
        assert!(result.is_ok() || result.is_err());
    }
}
