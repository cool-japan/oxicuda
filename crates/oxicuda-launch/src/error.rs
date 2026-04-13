//! Launch validation error types.
//!
//! These errors are returned by [`crate::LaunchParams::validate`] when launch
//! parameters exceed device hardware limits.

use std::fmt;

/// Error type for launch parameter validation.
///
/// These variants describe specific constraint violations that would
/// cause a kernel launch to fail at the driver level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchError {
    /// The total number of threads per block exceeds the device maximum.
    BlockSizeExceedsLimit {
        /// Requested total threads per block.
        requested: u32,
        /// Maximum allowed by the device.
        max: u32,
    },
    /// A grid dimension exceeds the device maximum.
    GridSizeExceedsLimit {
        /// Requested grid size in the offending dimension.
        requested: u32,
        /// Maximum allowed by the device.
        max: u32,
    },
    /// The requested dynamic shared memory exceeds the device maximum.
    SharedMemoryExceedsLimit {
        /// Requested shared memory in bytes.
        requested: u32,
        /// Maximum allowed by the device.
        max: u32,
    },
    /// A dimension value is invalid (e.g., zero).
    InvalidDimension {
        /// Name of the dimension (e.g., "block.x", "grid.z").
        dim: &'static str,
        /// The invalid value.
        value: u32,
    },
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlockSizeExceedsLimit { requested, max } => {
                write!(
                    f,
                    "block size {requested} exceeds device maximum {max} threads per block"
                )
            }
            Self::GridSizeExceedsLimit { requested, max } => {
                write!(f, "grid dimension {requested} exceeds device maximum {max}")
            }
            Self::SharedMemoryExceedsLimit { requested, max } => {
                write!(
                    f,
                    "shared memory {requested} bytes exceeds device maximum {max} bytes"
                )
            }
            Self::InvalidDimension { dim, value } => {
                write!(f, "invalid dimension {dim} = {value} (must be > 0)")
            }
        }
    }
}

impl std::error::Error for LaunchError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_exceeds_display() {
        let err = LaunchError::BlockSizeExceedsLimit {
            requested: 2048,
            max: 1024,
        };
        let msg = format!("{err}");
        assert!(msg.contains("2048"));
        assert!(msg.contains("1024"));
    }

    #[test]
    fn grid_size_exceeds_display() {
        let err = LaunchError::GridSizeExceedsLimit {
            requested: 100_000,
            max: 65535,
        };
        let msg = format!("{err}");
        assert!(msg.contains("100000"));
        assert!(msg.contains("65535"));
    }

    #[test]
    fn shared_memory_exceeds_display() {
        let err = LaunchError::SharedMemoryExceedsLimit {
            requested: 65536,
            max: 49152,
        };
        let msg = format!("{err}");
        assert!(msg.contains("65536"));
        assert!(msg.contains("49152"));
    }

    #[test]
    fn invalid_dimension_display() {
        let err = LaunchError::InvalidDimension {
            dim: "block.x",
            value: 0,
        };
        let msg = format!("{err}");
        assert!(msg.contains("block.x"));
        assert!(msg.contains("0"));
    }

    #[test]
    fn launch_error_implements_std_error() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<LaunchError>();
    }

    #[test]
    fn launch_error_eq() {
        let a = LaunchError::BlockSizeExceedsLimit {
            requested: 512,
            max: 256,
        };
        let b = LaunchError::BlockSizeExceedsLimit {
            requested: 512,
            max: 256,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn launch_error_debug() {
        let err = LaunchError::InvalidDimension {
            dim: "grid.z",
            value: 0,
        };
        let dbg = format!("{err:?}");
        assert!(dbg.contains("InvalidDimension"));
    }
}
