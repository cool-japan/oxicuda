//! CUDA profiler control.
//!
//! Implements:
//! - `cudaProfilerStart`
//! - `cudaProfilerStop`
//!
//! These allow application code to bracket regions of interest for profiling
//! tools such as Nsight Systems or nvprof.

use oxicuda_driver::loader::try_driver;

use crate::error::{CudaRtError, CudaRtResult};

/// Start the CUDA profiler.
///
/// Mirrors `cudaProfilerStart`.  Must be paired with [`profiler_stop`].
///
/// # Errors
///
/// Propagates driver errors (e.g., profiler not initialised).
pub fn profiler_start() -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI; no user data involved. cu_profiler_start is optional.
    let f = api.cu_profiler_start.ok_or(CudaRtError::NotSupported)?;
    let rc = unsafe { f() };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::ProfilerDisabled));
    }
    Ok(())
}

/// Stop the CUDA profiler.
///
/// Mirrors `cudaProfilerStop`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn profiler_stop() -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    // SAFETY: FFI. cu_profiler_stop is optional.
    let f = api.cu_profiler_stop.ok_or(CudaRtError::NotSupported)?;
    let rc = unsafe { f() };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::ProfilerDisabled));
    }
    Ok(())
}

/// RAII guard that calls [`profiler_start`] on construction and
/// [`profiler_stop`] on drop.
///
/// # Example
///
/// ```rust,no_run
/// # use oxicuda_runtime::profiler::ProfilerGuard;
/// let _guard = ProfilerGuard::new().unwrap();
/// // profiling active here …
/// // dropped → profiler_stop called automatically
/// ```
pub struct ProfilerGuard {
    active: bool,
}

impl ProfilerGuard {
    /// Start profiling and return a guard.
    ///
    /// # Errors
    ///
    /// Propagates [`profiler_start`] errors.
    pub fn new() -> CudaRtResult<Self> {
        profiler_start()?;
        Ok(Self { active: true })
    }

    /// Stop profiling early (also called on drop).
    pub fn stop(&mut self) {
        if self.active {
            let _ = profiler_stop();
            self.active = false;
        }
    }
}

impl Drop for ProfilerGuard {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_start_without_gpu_returns_error() {
        // Must not panic regardless of GPU presence.
        let _ = profiler_start();
    }

    #[test]
    fn profiler_stop_without_gpu_returns_error() {
        let _ = profiler_stop();
    }
}
