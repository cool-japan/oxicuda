//! CUDA event management.
//!
//! Implements the CUDA Runtime event API:
//! - `cudaEventCreate` / `cudaEventCreateWithFlags`
//! - `cudaEventDestroy`
//! - `cudaEventRecord`
//! - `cudaEventSynchronize`
//! - `cudaEventQuery`
//! - `cudaEventElapsedTime`

use oxicuda_driver::ffi::CUevent;
use oxicuda_driver::loader::try_driver;

use crate::error::{CudaRtError, CudaRtResult};
use crate::stream::CudaStream;

// ─── EventFlags ──────────────────────────────────────────────────────────────

/// Flags for event creation.
///
/// Mirrors `cudaEventFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EventFlags(pub u32);

impl EventFlags {
    /// Default flags.
    pub const DEFAULT: Self = Self(0x0);
    /// Event will not record timing data (lower overhead).
    pub const DISABLE_TIMING: Self = Self(0x2);
    /// Event can be used for interprocess synchronisation.
    pub const INTERPROCESS: Self = Self(0x4);
}

// ─── CudaEvent ───────────────────────────────────────────────────────────────

/// A CUDA event handle.
///
/// Wraps the raw `CUevent` from the driver API.  The event is **not**
/// automatically destroyed on drop — call [`event_destroy`] explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaEvent(CUevent);

impl CudaEvent {
    /// Construct from a raw driver event handle.
    ///
    /// # Safety
    ///
    /// The handle must be valid and not be used after the owning context is
    /// destroyed.
    #[must_use]
    pub const unsafe fn from_raw(raw: CUevent) -> Self {
        Self(raw)
    }

    /// Returns the underlying raw `CUevent`.
    #[must_use]
    pub fn raw(self) -> CUevent {
        self.0
    }

    /// Returns `true` if the event handle is null (invalid).
    #[must_use]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

// ─── Event creation / destruction ────────────────────────────────────────────

/// Create a CUDA event with default flags.
///
/// Mirrors `cudaEventCreate`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_create() -> CudaRtResult<CudaEvent> {
    event_create_with_flags(EventFlags::DEFAULT)
}

/// Create a CUDA event with the given flags.
///
/// Mirrors `cudaEventCreateWithFlags`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_create_with_flags(flags: EventFlags) -> CudaRtResult<CudaEvent> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut event = CUevent::default();
    // SAFETY: FFI; event pointer is valid.
    let rc = unsafe { (api.cu_event_create)(&raw mut event, flags.0) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(CudaEvent(event))
}

/// Destroy a CUDA event.
///
/// Mirrors `cudaEventDestroy`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_destroy(event: CudaEvent) -> CudaRtResult<()> {
    if event.is_null() {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; event is valid.
    let rc = unsafe { (api.cu_event_destroy_v2)(event.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(())
}

// ─── Event recording and synchronisation ─────────────────────────────────────

/// Record `event` at the current position in `stream`.
///
/// Mirrors `cudaEventRecord`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_record(event: CudaEvent, stream: CudaStream) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; event and stream are valid.
    let rc = unsafe { (api.cu_event_record)(event.raw(), stream.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(())
}

/// Record `event` at the current position in `stream` with flags.
///
/// Mirrors `cudaEventRecordWithFlags`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_record_with_flags(
    event: CudaEvent,
    stream: CudaStream,
    flags: u32,
) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI. cu_event_record_with_flags is optional (CUDA 11.1+).
    let f = api
        .cu_event_record_with_flags
        .ok_or(CudaRtError::NotSupported)?;
    let rc = unsafe { f(event.raw(), stream.raw(), flags) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(())
}

/// Block the calling thread until `event` is recorded.
///
/// Mirrors `cudaEventSynchronize`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_synchronize(event: CudaEvent) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_event_synchronize)(event.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::NotReady));
    }
    Ok(())
}

/// Query whether `event` has been recorded.
///
/// Mirrors `cudaEventQuery`.
///
/// Returns `Ok(true)` if complete, `Ok(false)` if not yet reached.
///
/// # Errors
///
/// Propagates driver errors other than `NotReady`.
pub fn event_query(event: CudaEvent) -> CudaRtResult<bool> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_event_query)(event.raw()) };
    match rc {
        0 => Ok(true),
        600 => Ok(false), // CUDA_ERROR_NOT_READY
        other => Err(CudaRtError::from_code(other).unwrap_or(CudaRtError::Unknown)),
    }
}

/// Compute the elapsed time between two events in milliseconds.
///
/// Mirrors `cudaEventElapsedTime`.
///
/// Both events must have been recorded.  If either was created with
/// `EventFlags::DISABLE_TIMING`, this returns [`CudaRtError::InvalidResourceHandle`].
///
/// # Errors
///
/// Propagates driver errors.
pub fn event_elapsed_time(start: CudaEvent, end: CudaEvent) -> CudaRtResult<f32> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut ms: f32 = 0.0;
    // SAFETY: FFI; ms is a valid stack-allocated f32.
    let rc = unsafe { (api.cu_event_elapsed_time)(&raw mut ms, start.raw(), end.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidResourceHandle));
    }
    Ok(ms)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_flags_values() {
        assert_eq!(EventFlags::DEFAULT.0, 0x0);
        assert_eq!(EventFlags::DISABLE_TIMING.0, 0x2);
        assert_eq!(EventFlags::INTERPROCESS.0, 0x4);
    }

    #[test]
    fn event_create_without_gpu_returns_error() {
        // Without a GPU this will fail, but must not panic.
        let _ = event_create();
    }

    #[test]
    fn event_destroy_null_is_noop() {
        // SAFETY: null is a well-defined state; no FFI call made.
        let ev = unsafe { CudaEvent::from_raw(CUevent::default()) };
        let _ = event_destroy(ev); // must not panic
    }
}
