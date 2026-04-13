//! Error types for the `oxicuda-levelzero` backend.

use oxicuda_backend::BackendError;

/// Errors specific to the Level Zero compute backend.
#[derive(Debug, thiserror::Error)]
pub enum LevelZeroError {
    /// Level Zero requires Linux or Windows.
    #[error("Level Zero requires Linux or Windows")]
    UnsupportedPlatform,

    /// The Level Zero loader library could not be found or loaded.
    #[error("Level Zero loader not found: {0}")]
    LibraryNotFound(String),

    /// A Level Zero API call returned a non-success result code.
    #[error("Level Zero error 0x{0:08x}: {1}")]
    ZeError(u32, String),

    /// No Intel GPU device was found on this system.
    #[error("no Intel GPU found")]
    NoSuitableDevice,

    /// Device ran out of memory during allocation.
    #[error("out of device memory")]
    OutOfMemory,

    /// A SPIR-V shader module error occurred.
    #[error("SPIR-V error: {0}")]
    ShaderError(String),

    /// An error occurred while operating on a command list.
    #[error("command list error: {0}")]
    CommandListError(String),

    /// The backend has not been initialized yet.
    #[error("not initialized")]
    NotInitialized,

    /// The requested operation is not supported.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// An invalid argument was passed.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

/// Convenience result alias for Level Zero operations.
pub type LevelZeroResult<T> = Result<T, LevelZeroError>;

impl From<LevelZeroError> for BackendError {
    fn from(e: LevelZeroError) -> Self {
        match e {
            LevelZeroError::UnsupportedPlatform => {
                BackendError::DeviceError("Level Zero requires Linux or Windows".into())
            }
            LevelZeroError::LibraryNotFound(msg) => {
                BackendError::DeviceError(format!("Level Zero loader not found: {msg}"))
            }
            LevelZeroError::ZeError(code, msg) => {
                BackendError::DeviceError(format!("Level Zero error 0x{code:08x}: {msg}"))
            }
            LevelZeroError::NoSuitableDevice => {
                BackendError::DeviceError("no Intel GPU found".into())
            }
            LevelZeroError::ShaderError(msg) => {
                BackendError::DeviceError(format!("SPIR-V error: {msg}"))
            }
            LevelZeroError::CommandListError(msg) => {
                BackendError::DeviceError(format!("command list error: {msg}"))
            }
            LevelZeroError::OutOfMemory => BackendError::OutOfMemory,
            LevelZeroError::NotInitialized => BackendError::NotInitialized,
            LevelZeroError::Unsupported(msg) => BackendError::Unsupported(msg),
            LevelZeroError::InvalidArgument(msg) => BackendError::InvalidArgument(msg),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_zero_error_display() {
        assert_eq!(
            LevelZeroError::UnsupportedPlatform.to_string(),
            "Level Zero requires Linux or Windows"
        );
        assert_eq!(
            LevelZeroError::LibraryNotFound("libze_loader.so".into()).to_string(),
            "Level Zero loader not found: libze_loader.so"
        );
        assert_eq!(
            LevelZeroError::ZeError(0x7800_0001, "device lost".into()).to_string(),
            "Level Zero error 0x78000001: device lost"
        );
        assert_eq!(
            LevelZeroError::NoSuitableDevice.to_string(),
            "no Intel GPU found"
        );
        assert_eq!(
            LevelZeroError::OutOfMemory.to_string(),
            "out of device memory"
        );
        assert_eq!(
            LevelZeroError::ShaderError("invalid binary".into()).to_string(),
            "SPIR-V error: invalid binary"
        );
        assert_eq!(
            LevelZeroError::CommandListError("closed".into()).to_string(),
            "command list error: closed"
        );
        assert_eq!(
            LevelZeroError::NotInitialized.to_string(),
            "not initialized"
        );
        assert_eq!(
            LevelZeroError::Unsupported("op".into()).to_string(),
            "unsupported: op"
        );
        assert_eq!(
            LevelZeroError::InvalidArgument("arg".into()).to_string(),
            "invalid argument: arg"
        );
    }

    #[test]
    fn backend_error_from_level_zero_error() {
        let e = BackendError::from(LevelZeroError::OutOfMemory);
        assert_eq!(e, BackendError::OutOfMemory);

        let e = BackendError::from(LevelZeroError::NotInitialized);
        assert_eq!(e, BackendError::NotInitialized);

        let e = BackendError::from(LevelZeroError::Unsupported("foo".into()));
        assert_eq!(e, BackendError::Unsupported("foo".into()));

        let e = BackendError::from(LevelZeroError::InvalidArgument("bar".into()));
        assert_eq!(e, BackendError::InvalidArgument("bar".into()));

        let e = BackendError::from(LevelZeroError::UnsupportedPlatform);
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(LevelZeroError::LibraryNotFound("ze.so".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(LevelZeroError::ZeError(1, "bad".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(LevelZeroError::NoSuitableDevice);
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(LevelZeroError::ShaderError("x".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));

        let e = BackendError::from(LevelZeroError::CommandListError("y".into()));
        assert!(matches!(e, BackendError::DeviceError(_)));
    }
}
