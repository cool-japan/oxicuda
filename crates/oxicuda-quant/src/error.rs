//! # Error types for `oxicuda-quant`

use thiserror::Error;

/// Alias for `Result<T, QuantError>`.
pub type QuantResult<T> = Result<T, QuantError>;

/// All errors that may be produced by `oxicuda-quant`.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum QuantError {
    /// Tensor dimension does not match the expected value.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// The input slice is empty when at least one element is required.
    #[error("empty input: {0}")]
    EmptyInput(&'static str),

    /// A quantization scale is zero or negative (would produce NaN/Inf).
    #[error("invalid scale {scale}: must be finite and positive")]
    InvalidScale { scale: f32 },

    /// A quantization bit-width is outside the valid range [1, 16].
    #[error("invalid bit-width {bits}: must be in [1, 16]")]
    InvalidBitWidth { bits: u32 },

    /// The requested group size does not evenly divide the tensor length.
    #[error("tensor length {len} is not divisible by group size {group}")]
    GroupSizeMismatch { len: usize, group: usize },

    /// A calibration dataset is required but was not provided.
    #[error("calibration data required for {0}")]
    CalibrationRequired(&'static str),

    /// The Hessian matrix is singular or near-singular.
    #[error("Hessian is near-singular (min diagonal = {min_diag:.3e})")]
    SingularHessian { min_diag: f32 },

    /// Distillation teacher and student outputs have incompatible shapes.
    #[error("teacher output length {teacher} != student output length {student}")]
    TeacherStudentMismatch { teacher: usize, student: usize },

    /// A pruning threshold produced a 100% sparse tensor (no weights remain).
    #[error("pruning threshold {threshold:.4} zeroed all {n} weights")]
    AllZeroPruning { threshold: f32, n: usize },

    /// FP8 format encoding received a NaN/Inf value.
    #[error("FP8 encoding received non-finite value {0}")]
    NonFiniteFp8(f32),

    /// Mixed-precision policy could not satisfy the requested compression ratio.
    #[error("cannot achieve target ratio {target:.2}× with given sensitivity")]
    InfeasibleCompressionTarget { target: f32 },

    /// Generic configuration error.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_dimension_mismatch() {
        let e = QuantError::DimensionMismatch {
            expected: 4,
            got: 3,
        };
        assert!(e.to_string().contains("dimension mismatch"));
    }

    #[test]
    fn error_display_invalid_scale() {
        let e = QuantError::InvalidScale { scale: -0.5 };
        assert!(e.to_string().contains("invalid scale"));
    }

    #[test]
    fn error_display_group_size() {
        let e = QuantError::GroupSizeMismatch { len: 10, group: 3 };
        assert!(e.to_string().contains("not divisible"));
    }

    #[test]
    fn result_alias_ok() {
        let r: QuantResult<i32> = Ok(42);
        assert!(r.is_ok());
    }
}
