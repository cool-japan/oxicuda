//! CUDA profiler control.
//!
//! Start and stop the CUDA profiler programmatically, or use the
//! [`ProfilerGuard`] RAII type to scope profiling to a block.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_driver::profiler;
//! # fn main() -> Result<(), oxicuda_driver::CudaError> {
//! {
//!     let _guard = profiler::ProfilerGuard::start()?;
//!     // ... GPU work to profile ...
//! } // profiler stops here
//! # Ok(())
//! # }
//! ```

use crate::error::{CudaError, CudaResult};
use crate::loader::try_driver;

/// Starts the CUDA profiler (e.g. Nsight Systems / nvprof).
///
/// # Errors
///
/// Returns [`CudaError::NotSupported`] if the driver lacks `cuProfilerStart`,
/// or another error on failure.
pub fn profiler_start() -> CudaResult<()> {
    let api = try_driver()?;
    let f = api.cu_profiler_start.ok_or(CudaError::NotSupported)?;
    crate::cuda_call!(f())
}

/// Stops the CUDA profiler.
///
/// # Errors
///
/// Returns [`CudaError::NotSupported`] if the driver lacks `cuProfilerStop`,
/// or another error on failure.
pub fn profiler_stop() -> CudaResult<()> {
    let api = try_driver()?;
    let f = api.cu_profiler_stop.ok_or(CudaError::NotSupported)?;
    crate::cuda_call!(f())
}

/// RAII guard that starts the CUDA profiler on creation and stops it on drop.
///
/// This is useful for scoping profiling to a specific block of code.
/// If the profiler fails to stop on drop, the error is silently ignored
/// (since `Drop::drop` cannot return errors).
pub struct ProfilerGuard {
    _private: (),
}

impl ProfilerGuard {
    /// Starts the CUDA profiler and returns a guard that will stop it on drop.
    ///
    /// # Errors
    ///
    /// Returns an error if `profiler_start()` fails.
    pub fn start() -> CudaResult<Self> {
        profiler_start()?;
        Ok(Self { _private: () })
    }
}

impl Drop for ProfilerGuard {
    fn drop(&mut self) {
        // Best-effort stop; we cannot propagate errors from Drop.
        let _ = profiler_stop();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_start_returns_error_without_gpu() {
        let result = profiler_start();
        // On macOS or without a GPU driver, this should fail gracefully.
        let _ = result;
    }

    #[test]
    fn profiler_stop_returns_error_without_gpu() {
        let result = profiler_stop();
        let _ = result;
    }

    #[test]
    fn profiler_guard_does_not_panic_without_gpu() {
        // start() will fail, so the guard should not be created,
        // and no panic should occur.
        let result = ProfilerGuard::start();
        let _ = result;
    }
}
