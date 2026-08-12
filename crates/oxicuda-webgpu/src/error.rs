//! Error types for the oxicuda-webgpu backend.

use oxicuda_backend::BackendError;

/// Errors specific to the WebGPU backend.
#[derive(Debug, thiserror::Error)]
pub enum WebGpuError {
    /// No compatible WebGPU adapter was found on this system.
    #[error("no WebGPU adapter found")]
    NoAdapter,

    /// The device request to the adapter failed.
    #[error("device request failed: {0}")]
    DeviceRequest(String),

    /// The GPU ran out of memory during buffer allocation.
    #[error("out of device memory")]
    OutOfMemory,

    /// WGSL shader source failed to compile.
    #[error("WGSL shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// A compute pipeline could not be created.
    #[error("pipeline creation failed: {0}")]
    PipelineCreation(String),

    /// The backend has not been initialized yet.
    #[error("not initialized")]
    NotInitialized,

    /// The requested operation is not supported by this backend.
    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// An invalid argument was passed to an operation.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A buffer mapping operation failed.
    #[error("buffer mapping failed: {0}")]
    BufferMapping(String),

    /// An async operation timed out waiting for the adapter.
    #[error("adapter timeout")]
    Timeout,

    /// A wgpu validation, out-of-memory, or internal error was captured by
    /// the non-fatal `on_uncaptured_error` handler installed in
    /// [`crate::device::WebGpuDevice::new`], instead of aborting the process
    /// (wgpu's default handler is fatal). See
    /// [`crate::device::WebGpuDevice::poll_error`].
    #[error("uncaptured wgpu error: {0}")]
    UncapturedError(String),

    /// The device was reported lost (GPU reset, driver failure, or an
    /// external `Device::destroy()` call) by the callback installed via
    /// `Device::set_device_lost_callback` in
    /// [`crate::device::WebGpuDevice::new`].
    #[error("device lost: {0}")]
    DeviceLost(String),
}

/// Convenience result alias for WebGPU operations.
pub type WebGpuResult<T> = Result<T, WebGpuError>;

impl From<WebGpuError> for BackendError {
    fn from(e: WebGpuError) -> Self {
        match e {
            WebGpuError::NoAdapter => BackendError::DeviceError("no WebGPU adapter found".into()),
            WebGpuError::DeviceRequest(msg) => {
                BackendError::DeviceError(format!("device request failed: {msg}"))
            }
            WebGpuError::OutOfMemory => BackendError::OutOfMemory,
            WebGpuError::ShaderCompilation(msg) => {
                BackendError::DeviceError(format!("WGSL shader compilation failed: {msg}"))
            }
            WebGpuError::PipelineCreation(msg) => {
                BackendError::DeviceError(format!("pipeline creation failed: {msg}"))
            }
            WebGpuError::NotInitialized => BackendError::NotInitialized,
            WebGpuError::Unsupported(msg) => BackendError::Unsupported(msg),
            WebGpuError::InvalidArgument(msg) => BackendError::InvalidArgument(msg),
            WebGpuError::BufferMapping(msg) => {
                BackendError::DeviceError(format!("buffer mapping failed: {msg}"))
            }
            WebGpuError::Timeout => BackendError::DeviceError("adapter timeout".into()),
            WebGpuError::UncapturedError(msg) => {
                BackendError::DeviceError(format!("uncaptured wgpu error: {msg}"))
            }
            WebGpuError::DeviceLost(msg) => {
                BackendError::DeviceError(format!("device lost: {msg}"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webgpu_error_display() {
        assert_eq!(
            WebGpuError::NoAdapter.to_string(),
            "no WebGPU adapter found"
        );
        assert_eq!(
            WebGpuError::DeviceRequest("oops".into()).to_string(),
            "device request failed: oops"
        );
        assert_eq!(WebGpuError::OutOfMemory.to_string(), "out of device memory");
        assert_eq!(
            WebGpuError::ShaderCompilation("syntax error".into()).to_string(),
            "WGSL shader compilation failed: syntax error"
        );
        assert_eq!(
            WebGpuError::PipelineCreation("invalid layout".into()).to_string(),
            "pipeline creation failed: invalid layout"
        );
        assert_eq!(WebGpuError::NotInitialized.to_string(), "not initialized");
        assert_eq!(
            WebGpuError::Unsupported("foo".into()).to_string(),
            "unsupported operation: foo"
        );
        assert_eq!(
            WebGpuError::InvalidArgument("bad arg".into()).to_string(),
            "invalid argument: bad arg"
        );
        assert_eq!(
            WebGpuError::BufferMapping("lock poisoned".into()).to_string(),
            "buffer mapping failed: lock poisoned"
        );
        assert_eq!(WebGpuError::Timeout.to_string(), "adapter timeout");
        assert_eq!(
            WebGpuError::UncapturedError("oom".into()).to_string(),
            "uncaptured wgpu error: oom"
        );
        assert_eq!(
            WebGpuError::DeviceLost("reset".into()).to_string(),
            "device lost: reset"
        );
    }

    #[test]
    fn webgpu_error_from_backend_error() {
        // Verify the From conversion produces the correct BackendError variants.
        let e = BackendError::from(WebGpuError::OutOfMemory);
        assert_eq!(e, BackendError::OutOfMemory);

        let e = BackendError::from(WebGpuError::NotInitialized);
        assert_eq!(e, BackendError::NotInitialized);

        let e = BackendError::from(WebGpuError::Unsupported("bar".into()));
        assert_eq!(e, BackendError::Unsupported("bar".into()));

        let e = BackendError::from(WebGpuError::InvalidArgument("baz".into()));
        assert_eq!(e, BackendError::InvalidArgument("baz".into()));

        let e = BackendError::from(WebGpuError::NoAdapter);
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(WebGpuError::DeviceRequest("x".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(WebGpuError::BufferMapping("m".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(WebGpuError::Timeout);
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(WebGpuError::UncapturedError("oom".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(WebGpuError::DeviceLost("reset".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));
    }
}
