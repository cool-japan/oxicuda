//! Error types for OxiCUDA Primitives operations.
//!
//! [`PrimitivesError`] covers all failure modes for GPU-accelerated parallel
//! primitives: invalid arguments, buffer validation, PTX generation failures,
//! kernel launch errors, and underlying CUDA driver errors.

use oxicuda_driver::CudaError;
use thiserror::Error;

/// Primitives-specific error type.
///
/// Every fallible operation in `oxicuda-primitives` returns
/// [`PrimitivesResult<T>`] which aliases this enum as the error variant.
#[derive(Debug, Error)]
pub enum PrimitivesError {
    /// An underlying CUDA driver call failed.
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] CudaError),

    /// A device buffer provided to an operation is too small.
    #[error("buffer too small: operation requires at least {expected} elements, got {actual}")]
    BufferTooSmall {
        /// Minimum required element count.
        expected: usize,
        /// Actual element count in the buffer.
        actual: usize,
    },

    /// The input element count exceeds an implementation limit.
    #[error("input too large: maximum supported size is {limit}, got {n}")]
    InputTooLarge {
        /// Maximum supported element count.
        limit: usize,
        /// Actual element count requested.
        n: usize,
    },

    /// An argument is invalid for the operation.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The requested operation or precision is not supported on this device.
    #[error("unsupported operation on {device}: {reason}")]
    UnsupportedOperation {
        /// Device description (e.g. `"sm_75"`).
        device: String,
        /// Why the operation is unsupported.
        reason: String,
    },

    /// PTX kernel source generation failed.
    #[error("PTX generation error for '{operation}': {msg}")]
    PtxGeneration {
        /// Name of the operation whose PTX could not be generated.
        operation: String,
        /// Underlying generation error message.
        msg: String,
    },

    /// Kernel module loading or symbol lookup failed.
    #[error("kernel load failed for '{kernel}': {msg}")]
    KernelLoad {
        /// Name of the kernel that could not be loaded.
        kernel: String,
        /// Underlying load error message.
        msg: String,
    },

    /// Kernel execution (launch or synchronisation) failed.
    #[error("kernel launch failed for '{kernel}': {msg}")]
    KernelLaunch {
        /// Name of the launched kernel.
        kernel: String,
        /// Underlying launch error message.
        msg: String,
    },

    /// Two operands have incompatible dimensions.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// The temporary work-space buffer passed by the caller is too small.
    #[error("workspace too small: need at least {needed} bytes, got {available}")]
    WorkspaceTooSmall {
        /// Bytes required.
        needed: usize,
        /// Bytes in the caller-supplied workspace.
        available: usize,
    },
}

/// Result alias for primitives operations.
pub type PrimitivesResult<T> = Result<T, PrimitivesError>;

// ─── Helper constructors ────────────────────────────────────────────────────

#[allow(dead_code)]
impl PrimitivesError {
    /// Construct a [`PrimitivesError::PtxGeneration`] from an operation name
    /// and an error value that implements `Display`.
    pub(crate) fn ptx(operation: &str, source: impl std::fmt::Display) -> Self {
        Self::PtxGeneration {
            operation: operation.to_string(),
            msg: source.to_string(),
        }
    }

    /// Construct a [`PrimitivesError::KernelLoad`] from a kernel name and
    /// an error value that implements `Display`.
    pub(crate) fn load(kernel: &str, source: impl std::fmt::Display) -> Self {
        Self::KernelLoad {
            kernel: kernel.to_string(),
            msg: source.to_string(),
        }
    }

    /// Construct a [`PrimitivesError::KernelLaunch`] from a kernel name and
    /// an error value that implements `Display`.
    pub(crate) fn launch(kernel: &str, source: impl std::fmt::Display) -> Self {
        Self::KernelLaunch {
            kernel: kernel.to_string(),
            msg: source.to_string(),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_too_small_display() {
        let e = PrimitivesError::BufferTooSmall {
            expected: 512,
            actual: 64,
        };
        let s = e.to_string();
        assert!(s.contains("512"), "should mention expected: {s}");
        assert!(s.contains("64"), "should mention actual: {s}");
    }

    #[test]
    fn ptx_generation_display() {
        let e = PrimitivesError::ptx("radix_sort", "bad opcode");
        let s = e.to_string();
        assert!(s.contains("radix_sort"), "{s}");
        assert!(s.contains("bad opcode"), "{s}");
    }

    #[test]
    fn kernel_load_display() {
        let e = PrimitivesError::load("warp_reduce_f32", "symbol not found");
        let s = e.to_string();
        assert!(s.contains("warp_reduce_f32"), "{s}");
    }

    #[test]
    fn kernel_launch_display() {
        let e = PrimitivesError::launch("block_scan_u32", "invalid grid dims");
        let s = e.to_string();
        assert!(s.contains("block_scan_u32"), "{s}");
    }

    #[test]
    fn unsupported_operation_display() {
        let e = PrimitivesError::UnsupportedOperation {
            device: "sm_70".into(),
            reason: "requires sm_80+".into(),
        };
        let s = e.to_string();
        assert!(s.contains("sm_70"), "{s}");
        assert!(s.contains("sm_80+"), "{s}");
    }

    #[test]
    fn workspace_too_small_display() {
        let e = PrimitivesError::WorkspaceTooSmall {
            needed: 4096,
            available: 1024,
        };
        let s = e.to_string();
        assert!(s.contains("4096"), "{s}");
        assert!(s.contains("1024"), "{s}");
    }

    #[test]
    fn input_too_large_display() {
        let e = PrimitivesError::InputTooLarge {
            limit: 1 << 30,
            n: usize::MAX,
        };
        let s = e.to_string();
        assert!(s.contains("1073741824"), "{s}");
    }

    #[test]
    fn from_cuda_error() {
        let cuda_err = CudaError::NoDevice;
        let prim_err = PrimitivesError::from(cuda_err);
        assert!(matches!(
            prim_err,
            PrimitivesError::Cuda(CudaError::NoDevice)
        ));
    }
}
