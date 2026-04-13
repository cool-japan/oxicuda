//! Error types for the `oxicuda-rocm` backend.

use oxicuda_backend::BackendError;

/// Errors specific to the ROCm/HIP compute backend.
#[derive(Debug, thiserror::Error)]
pub enum RocmError {
    /// ROCm/HIP is only supported on Linux.
    #[error("ROCm/HIP requires Linux")]
    UnsupportedPlatform,

    /// The HIP runtime shared library was not found or failed to load.
    #[error("HIP library not found: {0}")]
    LibraryNotFound(String),

    /// A HIP runtime API returned a non-zero error code.
    #[error("HIP error code {0}: {1}")]
    HipError(i32, String),

    /// No AMD GPU was discovered on this system.
    #[error("no AMD GPU found")]
    NoSuitableDevice,

    /// Device memory allocation failed due to insufficient resources.
    #[error("out of device memory")]
    OutOfMemory,

    /// The backend has not been initialized yet.
    #[error("not initialized")]
    NotInitialized,

    /// The requested operation is not yet supported by this backend.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// An invalid argument was passed to an operation.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A device-level error occurred.
    #[error("device error: {0}")]
    DeviceError(String),
}

/// Convenience result alias for ROCm operations.
pub type RocmResult<T> = Result<T, RocmError>;

impl From<RocmError> for BackendError {
    fn from(e: RocmError) -> Self {
        match e {
            RocmError::UnsupportedPlatform => {
                BackendError::DeviceError("ROCm/HIP requires Linux".into())
            }
            RocmError::LibraryNotFound(s) => {
                BackendError::DeviceError(format!("HIP library not found: {s}"))
            }
            RocmError::HipError(code, msg) => {
                BackendError::DeviceError(format!("hip({code}): {msg}"))
            }
            RocmError::NoSuitableDevice => BackendError::DeviceError("no AMD GPU found".into()),
            RocmError::OutOfMemory => BackendError::OutOfMemory,
            RocmError::NotInitialized => BackendError::NotInitialized,
            RocmError::Unsupported(s) => BackendError::Unsupported(s),
            RocmError::InvalidArgument(s) => BackendError::InvalidArgument(s),
            RocmError::DeviceError(s) => BackendError::DeviceError(s),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_error_display() {
        assert_eq!(
            RocmError::UnsupportedPlatform.to_string(),
            "ROCm/HIP requires Linux"
        );
        assert_eq!(
            RocmError::LibraryNotFound("libamdhip64.so".into()).to_string(),
            "HIP library not found: libamdhip64.so"
        );
        assert_eq!(
            RocmError::HipError(1, "invalid device".into()).to_string(),
            "HIP error code 1: invalid device"
        );
        assert_eq!(RocmError::NoSuitableDevice.to_string(), "no AMD GPU found");
        assert_eq!(RocmError::OutOfMemory.to_string(), "out of device memory");
        assert_eq!(RocmError::NotInitialized.to_string(), "not initialized");
        assert_eq!(
            RocmError::Unsupported("gemm".into()).to_string(),
            "unsupported: gemm"
        );
        assert_eq!(
            RocmError::InvalidArgument("bad ptr".into()).to_string(),
            "invalid argument: bad ptr"
        );
        assert_eq!(
            RocmError::DeviceError("kernel launch failed".into()).to_string(),
            "device error: kernel launch failed"
        );
    }

    #[test]
    fn backend_error_from_rocm_error() {
        let e = BackendError::from(RocmError::OutOfMemory);
        assert_eq!(e, BackendError::OutOfMemory);

        let e = BackendError::from(RocmError::NotInitialized);
        assert_eq!(e, BackendError::NotInitialized);

        let e = BackendError::from(RocmError::Unsupported("foo".into()));
        assert_eq!(e, BackendError::Unsupported("foo".into()));

        let e = BackendError::from(RocmError::InvalidArgument("bar".into()));
        assert_eq!(e, BackendError::InvalidArgument("bar".into()));

        let e = BackendError::from(RocmError::UnsupportedPlatform);
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(RocmError::LibraryNotFound("x".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(RocmError::HipError(42, "something".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(RocmError::NoSuitableDevice);
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(RocmError::DeviceError("boom".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));
    }
}
