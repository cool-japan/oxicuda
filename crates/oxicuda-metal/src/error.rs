//! Error types for the `oxicuda-metal` backend.

use oxicuda_backend::BackendError;

/// Errors specific to the Metal compute backend.
#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    /// Metal requires macOS — this platform is not supported.
    #[error("Metal requires macOS")]
    UnsupportedPlatform,

    /// No Metal-capable device was found on this system.
    #[error("no Metal device found")]
    NoDevice,

    /// The device ran out of memory during buffer allocation.
    #[error("out of device memory")]
    OutOfMemory,

    /// An MSL shader failed to compile.
    #[error("MSL shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// A compute pipeline could not be created.
    #[error("pipeline creation failed: {0}")]
    PipelineCreation(String),

    /// The backend has not been initialized yet.
    #[error("not initialized")]
    NotInitialized,

    /// The requested operation is not supported by this backend.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// An invalid argument was passed to an operation.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A Metal command buffer encountered an error.
    #[error("command buffer error: {0}")]
    CommandBufferError(String),
}

/// Convenience result alias for Metal operations.
pub type MetalResult<T> = Result<T, MetalError>;

impl From<MetalError> for BackendError {
    fn from(e: MetalError) -> Self {
        match e {
            MetalError::UnsupportedPlatform => {
                BackendError::DeviceError("Metal requires macOS".into())
            }
            MetalError::NoDevice => BackendError::DeviceError("no Metal device found".into()),
            MetalError::OutOfMemory => BackendError::OutOfMemory,
            MetalError::ShaderCompilation(msg) => {
                BackendError::DeviceError(format!("MSL shader compilation failed: {msg}"))
            }
            MetalError::PipelineCreation(msg) => {
                BackendError::DeviceError(format!("pipeline creation failed: {msg}"))
            }
            MetalError::NotInitialized => BackendError::NotInitialized,
            MetalError::Unsupported(msg) => BackendError::Unsupported(msg),
            MetalError::InvalidArgument(msg) => BackendError::InvalidArgument(msg),
            MetalError::CommandBufferError(msg) => {
                BackendError::DeviceError(format!("command buffer error: {msg}"))
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_error_display() {
        assert_eq!(
            MetalError::UnsupportedPlatform.to_string(),
            "Metal requires macOS"
        );
        assert_eq!(MetalError::NoDevice.to_string(), "no Metal device found");
        assert_eq!(MetalError::OutOfMemory.to_string(), "out of device memory");
        assert_eq!(
            MetalError::ShaderCompilation("syntax error".into()).to_string(),
            "MSL shader compilation failed: syntax error"
        );
        assert_eq!(
            MetalError::PipelineCreation("bad layout".into()).to_string(),
            "pipeline creation failed: bad layout"
        );
        assert_eq!(MetalError::NotInitialized.to_string(), "not initialized");
        assert_eq!(
            MetalError::Unsupported("op".into()).to_string(),
            "unsupported: op"
        );
        assert_eq!(
            MetalError::InvalidArgument("arg".into()).to_string(),
            "invalid argument: arg"
        );
        assert_eq!(
            MetalError::CommandBufferError("fail".into()).to_string(),
            "command buffer error: fail"
        );
    }

    #[test]
    fn backend_error_from_metal_error() {
        let e = BackendError::from(MetalError::OutOfMemory);
        assert_eq!(e, BackendError::OutOfMemory);

        let e = BackendError::from(MetalError::NotInitialized);
        assert_eq!(e, BackendError::NotInitialized);

        let e = BackendError::from(MetalError::Unsupported("foo".into()));
        assert_eq!(e, BackendError::Unsupported("foo".into()));

        let e = BackendError::from(MetalError::InvalidArgument("bar".into()));
        assert_eq!(e, BackendError::InvalidArgument("bar".into()));

        let e = BackendError::from(MetalError::UnsupportedPlatform);
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(MetalError::NoDevice);
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(MetalError::ShaderCompilation("x".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(MetalError::PipelineCreation("y".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(MetalError::CommandBufferError("z".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));
    }
}
