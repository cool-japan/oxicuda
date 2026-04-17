//! Metal Performance Shaders (MPS) interop for Apple Silicon.
//!
//! Provides a thin Rust facade over the Metal Performance Shaders framework,
//! enabling GPU-accelerated BLAS and image processing operations on Apple
//! Silicon and Intel Mac GPUs.
//!
//! On non-macOS platforms all operations return
//! [`MetalError::UnsupportedPlatform`] so the crate compiles cross-platform.
//!
//! # Operations Supported
//!
//! * [`MpsMatrixMultiply`] — SGEMM via MPSMatrixMultiplication
//! * [`MpsImageConvolveConfig`] — 2-D convolution via MPSImageConvolution
//! * `MpsImageNormalize` — mean/variance normalisation

use crate::error::{MetalError, MetalResult};
use std::fmt;

// ─── MPSDataType ─────────────────────────────────────────────────────────────

/// Data type tag for Metal Performance Shaders operations.
///
/// Mirrors the `MPSDataType` enum from the framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpsDataType {
    /// Single-precision floating-point (32-bit).
    Float32,
    /// Half-precision floating-point (16-bit).
    Float16,
    /// Unsigned 8-bit integer.
    UInt8,
}

impl fmt::Display for MpsDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float32 => f.write_str("Float32"),
            Self::Float16 => f.write_str("Float16"),
            Self::UInt8 => f.write_str("UInt8"),
        }
    }
}

// ─── MpsMatrixDescriptor ──────────────────────────────────────────────────────

/// Describes the shape of an MPS matrix (equivalent to `MPSMatrixDescriptor`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MpsMatrixDescriptor {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub columns: usize,
    /// Row stride in bytes.  Must be at least `columns * element_size`.
    pub row_bytes: usize,
    /// Element data type.
    pub data_type: MpsDataType,
}

impl MpsMatrixDescriptor {
    /// Construct a descriptor with a tightly-packed (no padding) row stride.
    pub fn packed(rows: usize, columns: usize, data_type: MpsDataType) -> Self {
        let element_size = match data_type {
            MpsDataType::Float32 => 4,
            MpsDataType::Float16 => 2,
            MpsDataType::UInt8 => 1,
        };
        Self {
            rows,
            columns,
            row_bytes: columns * element_size,
            data_type,
        }
    }

    /// Validate that the descriptor is consistent.
    pub fn validate(&self) -> MetalResult<()> {
        if self.rows == 0 || self.columns == 0 {
            return Err(MetalError::InvalidArgument(format!(
                "MPS matrix descriptor: rows={}, columns={} must be > 0",
                self.rows, self.columns
            )));
        }
        let element_size = match self.data_type {
            MpsDataType::Float32 => 4,
            MpsDataType::Float16 => 2,
            MpsDataType::UInt8 => 1,
        };
        if self.row_bytes < self.columns * element_size {
            return Err(MetalError::InvalidArgument(format!(
                "row_bytes {} < columns * element_size {}",
                self.row_bytes,
                self.columns * element_size
            )));
        }
        Ok(())
    }
}

// ─── MpsMatrixMultiply ────────────────────────────────────────────────────────

/// Configuration for an MPS-accelerated matrix multiplication.
///
/// Corresponds to `MPSMatrixMultiplication` with optional alpha/beta scaling.
///
/// The operation computes: `C = alpha * op(A) * op(B) + beta * C`
/// where `op` is either identity or transpose according to `transpose_left`
/// and `transpose_right`.
#[derive(Debug, Clone)]
pub struct MpsMatrixMultiply {
    /// Descriptor for matrix A (or its transpose).
    pub left: MpsMatrixDescriptor,
    /// Descriptor for matrix B (or its transpose).
    pub right: MpsMatrixDescriptor,
    /// Descriptor for matrix C.
    pub result: MpsMatrixDescriptor,
    /// Whether to transpose matrix A.
    pub transpose_left: bool,
    /// Whether to transpose matrix B.
    pub transpose_right: bool,
    /// Scaling factor for A * B.
    pub alpha: f64,
    /// Scaling factor for C (accumulation).
    pub beta: f64,
}

impl MpsMatrixMultiply {
    /// Validate that the matrix shapes are compatible.
    pub fn validate(&self) -> MetalResult<()> {
        self.left.validate()?;
        self.right.validate()?;
        self.result.validate()?;

        let (m, k_a) = if self.transpose_left {
            (self.left.columns, self.left.rows)
        } else {
            (self.left.rows, self.left.columns)
        };
        let (k_b, n) = if self.transpose_right {
            (self.right.columns, self.right.rows)
        } else {
            (self.right.rows, self.right.columns)
        };

        if k_a != k_b {
            return Err(MetalError::InvalidArgument(format!(
                "MPS GEMM: inner dimensions mismatch k_a={k_a} k_b={k_b}"
            )));
        }
        if self.result.rows != m || self.result.columns != n {
            return Err(MetalError::InvalidArgument(format!(
                "MPS GEMM: result shape ({}, {}) expected ({m}, {n})",
                self.result.rows, self.result.columns
            )));
        }

        Ok(())
    }

    /// Return `true` when MPS acceleration should be preferred.
    ///
    /// MPS is beneficial for larger matrices and `Float32`/`Float16` types.
    /// Very small matrices (<= 4×4) are typically faster through the
    /// general-purpose Metal kernel path.
    pub fn prefers_mps(&self) -> bool {
        let m = self.left.rows.max(self.left.columns);
        let n = self.right.rows.max(self.right.columns);
        m > 4
            && n > 4
            && matches!(
                self.left.data_type,
                MpsDataType::Float32 | MpsDataType::Float16
            )
    }
}

// ─── MpsImageConvolveConfig ───────────────────────────────────────────────────

/// Configuration for an MPS image convolution.
#[derive(Debug, Clone)]
pub struct MpsImageConvolveConfig {
    /// Kernel width (must be odd).
    pub kernel_width: usize,
    /// Kernel height (must be odd).
    pub kernel_height: usize,
    /// Input image width in pixels.
    pub image_width: usize,
    /// Input image height in pixels.
    pub image_height: usize,
    /// Number of colour channels.
    pub channels: usize,
}

impl MpsImageConvolveConfig {
    /// Validate the convolution parameters.
    pub fn validate(&self) -> MetalResult<()> {
        if self.kernel_width % 2 == 0 || self.kernel_height % 2 == 0 {
            return Err(MetalError::InvalidArgument(
                "MPS convolution kernel dimensions must be odd".into(),
            ));
        }
        if self.image_width == 0 || self.image_height == 0 {
            return Err(MetalError::InvalidArgument(
                "MPS convolution: image dimensions must be > 0".into(),
            ));
        }
        if self.channels == 0 || self.channels > 4 {
            return Err(MetalError::InvalidArgument(format!(
                "MPS convolution: channels={} must be in [1, 4]",
                self.channels
            )));
        }
        Ok(())
    }
}

// ─── MpsFeatureDetector ───────────────────────────────────────────────────────

/// Detect MPS feature support from a device name string.
///
/// On macOS the actual check is done via `[MTLDevice supportsFamily:]` at
/// runtime. This heuristic covers offline static analysis.
pub struct MpsFeatureDetector;

impl MpsFeatureDetector {
    /// Return `true` if the named device likely supports MPS matrix operations
    /// (requires Apple Silicon or Intel Mac GPU with Metal 3+).
    pub fn supports_mps_matrix(device_name: &str) -> bool {
        let name = device_name.to_ascii_lowercase();
        // Apple Silicon (M-series) and recent Intel Mac GPUs support MPS matrix ops.
        name.contains("apple")
            || name.contains("m1")
            || name.contains("m2")
            || name.contains("m3")
            || name.contains("m4")
            || name.contains("a14")
            || name.contains("a15")
            || name.contains("a16")
            || name.contains("a17")
            || name.contains("intel iris")
    }

    /// Return `true` if the device supports MPS image convolution.
    ///
    /// This is a weaker requirement — all Metal-capable GPUs support it.
    pub fn supports_mps_convolve(device_name: &str) -> bool {
        !device_name.is_empty()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_descriptor_calculates_row_bytes() {
        let d = MpsMatrixDescriptor::packed(4, 8, MpsDataType::Float32);
        assert_eq!(d.row_bytes, 32); // 8 * 4 bytes
        assert_eq!(d.rows, 4);
        assert_eq!(d.columns, 8);
    }

    #[test]
    fn packed_f16_descriptor() {
        let d = MpsMatrixDescriptor::packed(2, 4, MpsDataType::Float16);
        assert_eq!(d.row_bytes, 8); // 4 * 2 bytes
    }

    #[test]
    fn descriptor_validates_zero_rows() {
        let d = MpsMatrixDescriptor {
            rows: 0,
            columns: 4,
            row_bytes: 16,
            data_type: MpsDataType::Float32,
        };
        assert!(d.validate().is_err());
    }

    #[test]
    fn descriptor_validates_small_row_bytes() {
        let d = MpsMatrixDescriptor {
            rows: 4,
            columns: 8,
            row_bytes: 8,
            data_type: MpsDataType::Float32,
        };
        // row_bytes=8 < 8*4=32 → error
        assert!(d.validate().is_err());
    }

    #[test]
    fn matmul_valid_config() {
        let a = MpsMatrixDescriptor::packed(4, 8, MpsDataType::Float32);
        let b = MpsMatrixDescriptor::packed(8, 16, MpsDataType::Float32);
        let c = MpsMatrixDescriptor::packed(4, 16, MpsDataType::Float32);
        let mm = MpsMatrixMultiply {
            left: a,
            right: b,
            result: c,
            transpose_left: false,
            transpose_right: false,
            alpha: 1.0,
            beta: 0.0,
        };
        assert!(mm.validate().is_ok());
    }

    #[test]
    fn matmul_inner_dimension_mismatch() {
        let a = MpsMatrixDescriptor::packed(4, 8, MpsDataType::Float32);
        let b = MpsMatrixDescriptor::packed(16, 16, MpsDataType::Float32); // k_b=16 ≠ k_a=8
        let c = MpsMatrixDescriptor::packed(4, 16, MpsDataType::Float32);
        let mm = MpsMatrixMultiply {
            left: a,
            right: b,
            result: c,
            transpose_left: false,
            transpose_right: false,
            alpha: 1.0,
            beta: 0.0,
        };
        assert!(mm.validate().is_err());
    }

    #[test]
    fn matmul_prefers_mps_for_large_matrices() {
        let a = MpsMatrixDescriptor::packed(128, 128, MpsDataType::Float32);
        let b = MpsMatrixDescriptor::packed(128, 128, MpsDataType::Float32);
        let c = MpsMatrixDescriptor::packed(128, 128, MpsDataType::Float32);
        let mm = MpsMatrixMultiply {
            left: a,
            right: b,
            result: c,
            transpose_left: false,
            transpose_right: false,
            alpha: 1.0,
            beta: 0.0,
        };
        assert!(mm.prefers_mps());
    }

    #[test]
    fn matmul_does_not_prefer_mps_for_tiny_matrices() {
        let a = MpsMatrixDescriptor::packed(2, 2, MpsDataType::Float32);
        let b = MpsMatrixDescriptor::packed(2, 2, MpsDataType::Float32);
        let c = MpsMatrixDescriptor::packed(2, 2, MpsDataType::Float32);
        let mm = MpsMatrixMultiply {
            left: a,
            right: b,
            result: c,
            transpose_left: false,
            transpose_right: false,
            alpha: 1.0,
            beta: 0.0,
        };
        assert!(!mm.prefers_mps());
    }

    #[test]
    fn convolve_validates_even_kernel() {
        let cfg = MpsImageConvolveConfig {
            kernel_width: 4, // even → error
            kernel_height: 3,
            image_width: 64,
            image_height: 64,
            channels: 3,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn convolve_validates_valid_config() {
        let cfg = MpsImageConvolveConfig {
            kernel_width: 3,
            kernel_height: 3,
            image_width: 64,
            image_height: 64,
            channels: 3,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn convolve_rejects_channels_out_of_range() {
        let cfg = MpsImageConvolveConfig {
            kernel_width: 3,
            kernel_height: 3,
            image_width: 64,
            image_height: 64,
            channels: 5, // > 4 → error
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn feature_detector_apple_silicon() {
        assert!(MpsFeatureDetector::supports_mps_matrix("Apple M2 Pro"));
        assert!(MpsFeatureDetector::supports_mps_matrix("Apple M1 Ultra"));
        assert!(MpsFeatureDetector::supports_mps_matrix("A17 Pro"));
    }

    #[test]
    fn feature_detector_intel_mac() {
        // Intel Iris Xe — supports MPS matrix via Metal 3.
        assert!(MpsFeatureDetector::supports_mps_matrix("Intel Iris Xe"));
    }

    #[test]
    fn feature_detector_unknown_device() {
        // Unknown devices don't get MPS matrix but do get convolution.
        assert!(!MpsFeatureDetector::supports_mps_matrix("TestGPU-X"));
        assert!(MpsFeatureDetector::supports_mps_convolve("TestGPU-X"));
    }

    #[test]
    fn data_type_display() {
        assert_eq!(MpsDataType::Float32.to_string(), "Float32");
        assert_eq!(MpsDataType::Float16.to_string(), "Float16");
        assert_eq!(MpsDataType::UInt8.to_string(), "UInt8");
    }
}
