//! Data type system for GPU tensors.
//!
//! Provides [`TensorDtype`] which represents the element types supported
//! by the tensor backend, including half-precision and brain floating-point
//! formats commonly used in deep learning.

use std::fmt;

// ─── Tensor data types ──────────────────────────────────────

/// Element data type of a GPU tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TensorDtype {
    /// 16-bit IEEE 754 half precision floating point.
    Float16,
    /// 16-bit brain floating point (bfloat16).
    BFloat16,
    /// 32-bit IEEE 754 single precision floating point.
    #[default]
    Float32,
    /// 64-bit IEEE 754 double precision floating point.
    Float64,
    /// 8-bit signed integer.
    Int8,
    /// 16-bit signed integer.
    Int16,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 8-bit unsigned integer.
    Uint8,
    /// Boolean (stored as 1 byte).
    Bool,
}

impl TensorDtype {
    /// Size in bytes of a single element of this type.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::Uint8 => 1,
            Self::Float16 | Self::BFloat16 | Self::Int16 => 2,
            Self::Float32 | Self::Int32 => 4,
            Self::Float64 | Self::Int64 => 8,
        }
    }

    /// Returns `true` if this is a floating-point type.
    #[must_use]
    pub const fn is_floating_point(self) -> bool {
        matches!(
            self,
            Self::Float16 | Self::BFloat16 | Self::Float32 | Self::Float64
        )
    }

    /// Returns `true` if this is an integer type.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64 | Self::Uint8
        )
    }

    /// Returns `true` if this is a half-precision type (FP16 or BF16).
    #[must_use]
    pub const fn is_half(self) -> bool {
        matches!(self, Self::Float16 | Self::BFloat16)
    }

    /// Returns the recommended compute dtype for mixed-precision training.
    ///
    /// Half-precision types promote to Float32 for reductions / losses.
    #[must_use]
    pub const fn compute_dtype(self) -> Self {
        match self {
            Self::Float16 | Self::BFloat16 => Self::Float32,
            other => other,
        }
    }

    /// Returns a short string name (e.g. `"f32"`, `"bf16"`).
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::Float16 => "f16",
            Self::BFloat16 => "bf16",
            Self::Float32 => "f32",
            Self::Float64 => "f64",
            Self::Int8 => "i8",
            Self::Int16 => "i16",
            Self::Int32 => "i32",
            Self::Int64 => "i64",
            Self::Uint8 => "u8",
            Self::Bool => "bool",
        }
    }

    /// Whether this dtype can be used as a gradient type.
    #[must_use]
    pub const fn supports_grad(self) -> bool {
        self.is_floating_point()
    }
}

impl fmt::Display for TensorDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.short_name())
    }
}

// ─── Precision category for autocast ────────────────────────

/// Classification of operations by their numerical precision requirements.
///
/// Used by [`Autocast`](super::mixed_precision::Autocast) to decide
/// which precision to run each operation in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionCategory {
    /// Operations safe in low precision (matmul, conv2d).
    LowPrecision,
    /// Operations requiring full precision (reduction, softmax, loss).
    FullPrecision,
    /// Operations that should keep their input precision.
    PassThrough,
}

impl fmt::Display for PrecisionCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LowPrecision => write!(f, "low_precision"),
            Self::FullPrecision => write!(f, "full_precision"),
            Self::PassThrough => write!(f, "pass_through"),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(TensorDtype::Bool.size_bytes(), 1);
        assert_eq!(TensorDtype::Float16.size_bytes(), 2);
        assert_eq!(TensorDtype::BFloat16.size_bytes(), 2);
        assert_eq!(TensorDtype::Float32.size_bytes(), 4);
        assert_eq!(TensorDtype::Float64.size_bytes(), 8);
        assert_eq!(TensorDtype::Int8.size_bytes(), 1);
        assert_eq!(TensorDtype::Int16.size_bytes(), 2);
        assert_eq!(TensorDtype::Int32.size_bytes(), 4);
        assert_eq!(TensorDtype::Int64.size_bytes(), 8);
        assert_eq!(TensorDtype::Uint8.size_bytes(), 1);
    }

    #[test]
    fn test_dtype_classification() {
        assert!(TensorDtype::Float32.is_floating_point());
        assert!(TensorDtype::Float64.is_floating_point());
        assert!(TensorDtype::Float16.is_floating_point());
        assert!(TensorDtype::BFloat16.is_floating_point());
        assert!(!TensorDtype::Int32.is_floating_point());
        assert!(!TensorDtype::Bool.is_floating_point());

        assert!(TensorDtype::Int32.is_integer());
        assert!(TensorDtype::Int8.is_integer());
        assert!(!TensorDtype::Float32.is_integer());

        assert!(TensorDtype::Float16.is_half());
        assert!(TensorDtype::BFloat16.is_half());
        assert!(!TensorDtype::Float32.is_half());
    }

    #[test]
    fn test_compute_dtype_promotion() {
        assert_eq!(TensorDtype::Float16.compute_dtype(), TensorDtype::Float32);
        assert_eq!(TensorDtype::BFloat16.compute_dtype(), TensorDtype::Float32);
        assert_eq!(TensorDtype::Float32.compute_dtype(), TensorDtype::Float32);
        assert_eq!(TensorDtype::Float64.compute_dtype(), TensorDtype::Float64);
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", TensorDtype::Float32), "f32");
        assert_eq!(format!("{}", TensorDtype::BFloat16), "bf16");
        assert_eq!(format!("{}", TensorDtype::Bool), "bool");
    }

    #[test]
    fn test_supports_grad() {
        assert!(TensorDtype::Float32.supports_grad());
        assert!(TensorDtype::Float64.supports_grad());
        assert!(TensorDtype::Float16.supports_grad());
        assert!(!TensorDtype::Int32.supports_grad());
        assert!(!TensorDtype::Bool.supports_grad());
    }

    #[test]
    fn test_default_dtype() {
        assert_eq!(TensorDtype::default(), TensorDtype::Float32);
    }

    #[test]
    fn test_precision_category_display() {
        assert_eq!(
            format!("{}", PrecisionCategory::LowPrecision),
            "low_precision"
        );
        assert_eq!(
            format!("{}", PrecisionCategory::FullPrecision),
            "full_precision"
        );
        assert_eq!(
            format!("{}", PrecisionCategory::PassThrough),
            "pass_through"
        );
    }
}
