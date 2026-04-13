//! Tensor Memory Accelerator (TMA) descriptor types for CUDA 12.x / sm_90+.
//!
//! The Tensor Memory Accelerator (TMA) is a hardware unit introduced on
//! Hopper GPUs (sm_90) and extended on Blackwell (sm_100 / sm_120). It
//! enables high-throughput bulk copies between global and shared memory
//! using pre-built *tensor map* descriptors that encode address layout,
//! swizzle, and out-of-bounds fill modes.
//!
//! # Descriptor creation
//!
//! Descriptors are created on the host by calling
//! `cuTensorMapEncodeTiled` (exposed as an optional driver function pointer
//! in [`DriverApi`](crate::loader::DriverApi)). This module provides:
//!
//! - The opaque [`CuTensorMap`] container (64 bytes, 64-byte aligned).
//! - Configuration enums for data type, interleave, swizzle, etc.
//! - [`TmaDescriptorBuilder`] — a typed builder that collects parameters
//!   and produces a [`TmaEncodeTiledParams`] ready to pass to the driver.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_driver::tma::{
//!     CuTensorMapDataType, CuTensorMapSwizzle, TmaDescriptorBuilder,
//! };
//!
//! // Build a descriptor for a row-major f16 matrix of shape 1024×2048,
//! // tiled with 64-row × 64-col shared-memory tiles.
//! let params = TmaDescriptorBuilder::new_2d(
//!     CuTensorMapDataType::Float16,
//!     1024,           // rows
//!     2048,           // cols
//!     2048 * 2,       // row stride in bytes (2 bytes per f16)
//!     64, 64,         // tile rows, tile cols
//! )
//! .with_swizzle(CuTensorMapSwizzle::B128)
//! .params();
//!
//! assert_eq!(params.num_dims, 2);
//! assert_eq!(params.global_dims[0], 2048); // cols first in CUDA convention
//! ```
//!
//! # CUDA 12+ and sm_90+ requirement
//!
//! TMA hardware is only present on Hopper (sm_90), Blackwell B100 (sm_100),
//! and Blackwell B200 (sm_120) GPUs. On older devices the descriptor can
//! still be built on the host but will not be usable in a kernel.

// =========================================================================
// CuTensorMap — the opaque descriptor container
// =========================================================================

/// Number of 64-bit words in the opaque tensor-map blob.
pub const CU_TENSOR_MAP_NUM_QWORDS: usize = 16;

/// Opaque TMA tensor map descriptor (64 bytes, 64-byte aligned).
///
/// Passed to CUDA kernels via kernel arguments so the TMA hardware can
/// read its encoding. Created on the host with `cuTensorMapEncodeTiled`
/// and must not be modified after the driver populates it.
///
/// # Layout
///
/// The 128-byte structure matches CUDA's `CUtensorMap` exactly. The
/// internal encoding is private to the driver; user code should treat
/// this as an opaque blob.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct CuTensorMap {
    /// Opaque 128-byte payload.
    pub opaque: [u64; CU_TENSOR_MAP_NUM_QWORDS],
}

impl Default for CuTensorMap {
    fn default() -> Self {
        Self {
            opaque: [0u64; CU_TENSOR_MAP_NUM_QWORDS],
        }
    }
}

impl std::fmt::Debug for CuTensorMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CuTensorMap")
            .field("opaque[0]", &self.opaque[0])
            .field("opaque[1]", &self.opaque[1])
            .finish_non_exhaustive()
    }
}

// =========================================================================
// Configuration enums
// =========================================================================

/// Element data type for the TMA descriptor.
///
/// Corresponds to `CUtensorMapDataType` in the CUDA Driver API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CuTensorMapDataType {
    /// Unsigned 8-bit integer.
    Uint8 = 0,
    /// Unsigned 16-bit integer.
    Uint16 = 1,
    /// Unsigned 32-bit integer.
    Uint32 = 2,
    /// Signed 32-bit integer.
    Int32 = 3,
    /// Unsigned 64-bit integer.
    Uint64 = 4,
    /// Signed 64-bit integer.
    Int64 = 5,
    /// IEEE-754 half-precision float (f16).
    Float16 = 6,
    /// IEEE-754 single-precision float (f32).
    Float32 = 7,
    /// IEEE-754 double-precision float (f64).
    Float64 = 8,
    /// Brain float 16 (bfloat16).
    Bfloat16 = 9,
    /// f32 with flush-to-zero for subnormals.
    Float32Ftz = 10,
    /// TensorFloat-32 (TF32).
    TF32 = 11,
    /// TF32 with flush-to-zero.
    TF32Ftz = 12,
}

impl CuTensorMapDataType {
    /// Returns the element size in bytes.
    #[must_use]
    pub const fn element_size_bytes(self) -> u32 {
        match self {
            Self::Uint8 => 1,
            Self::Uint16 | Self::Float16 | Self::Bfloat16 => 2,
            Self::Uint32
            | Self::Int32
            | Self::Float32
            | Self::Float32Ftz
            | Self::TF32
            | Self::TF32Ftz => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 => 8,
        }
    }
}

/// Interleave pattern for the TMA descriptor.
///
/// Controls how elements from different warp lanes are interleaved.
/// `None` is safe for most use cases; `B16`/`B32` are only needed for
/// 1D interleaving of narrow data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CuTensorMapInterleave {
    /// No interleaving.
    None = 0,
    /// 16-byte interleaving stride.
    B16 = 1,
    /// 32-byte interleaving stride.
    B32 = 2,
}

/// Swizzle pattern applied to shared-memory tiles.
///
/// Swizzling re-maps addresses within the tile to avoid shared-memory
/// bank conflicts. Choose based on the tile width and element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CuTensorMapSwizzle {
    /// No swizzle (row-major, bank conflicts possible).
    None = 0,
    /// 32-byte swizzle sector.
    B32 = 1,
    /// 64-byte swizzle sector.
    B64 = 2,
    /// 128-byte swizzle sector.  Default for most f16/bf16 workloads.
    B128 = 3,
}

/// L2 cache promotion hint for TMA loads.
///
/// Instructs the L2 cache to pre-fetch TMA data in larger chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CuTensorMapL2Promotion {
    /// No L2 promotion.
    None = 0,
    /// Promote to 64-byte L2 lines.
    L2B64 = 1,
    /// Promote to 128-byte L2 lines.
    L2B128 = 2,
    /// Promote to 256-byte L2 lines.
    L2B256 = 3,
}

/// Out-of-bounds fill mode for TMA loads.
///
/// When a TMA access goes out of the declared tensor bounds, this controls
/// whether out-of-bounds elements return zero or NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CuTensorMapFloatOobFill {
    /// No fill — out-of-bounds reads are undefined.
    None = 0,
    /// Out-of-bounds float reads return NaN; FMA returns zero.
    NanRequestZeroFma = 1,
}

// =========================================================================
// TmaEncodeTiledParams — flattened parameters for the driver call
// =========================================================================

/// All parameters required by `cuTensorMapEncodeTiled`.
///
/// Produced by [`TmaDescriptorBuilder`]. Pass these to the driver function
/// pointer available in
/// [`DriverApi::cu_tensor_map_encode_tiled`](crate::loader::DriverApi::cu_tensor_map_encode_tiled).
///
/// # Dimension ordering
///
/// CUDA TMA uses **column-major** ordering for dimensions: `global_dims[0]`
/// is the *innermost* (fastest-varying) dimension. For a row-major matrix
/// of shape `R × C`, set `global_dims[0] = C` (cols) and
/// `global_dims[1] = R` (rows).
#[derive(Debug, Clone)]
pub struct TmaEncodeTiledParams {
    /// Element data type.
    pub data_type: CuTensorMapDataType,
    /// Number of tensor dimensions (1–5).
    pub num_dims: u32,
    /// Size of each global tensor dimension (innermost first).
    pub global_dims: [u64; 5],
    /// Byte stride between elements in outer dimensions (innermost stride omitted).
    pub global_strides: [u64; 4],
    /// Size of each tile dimension (must fit in shared memory).
    pub box_dims: [u32; 5],
    /// Element stride within each tile dimension (typically all-ones).
    pub element_strides: [u32; 5],
    /// Interleave mode.
    pub interleave: CuTensorMapInterleave,
    /// Swizzle mode.
    pub swizzle: CuTensorMapSwizzle,
    /// L2 promotion hint.
    pub l2_promotion: CuTensorMapL2Promotion,
    /// Out-of-bounds fill mode for float elements.
    pub oob_fill: CuTensorMapFloatOobFill,
}

// =========================================================================
// TmaDescriptorBuilder
// =========================================================================

/// Typed builder for TMA tensor-map descriptors.
///
/// Collects parameters in a convenient Rust API and produces a
/// [`TmaEncodeTiledParams`] struct suitable for passing to the CUDA driver's
/// `cuTensorMapEncodeTiled` entry point.
///
/// # Example
///
/// ```rust
/// use oxicuda_driver::tma::{
///     CuTensorMapDataType, CuTensorMapSwizzle, TmaDescriptorBuilder,
/// };
///
/// let params = TmaDescriptorBuilder::new_2d(
///     CuTensorMapDataType::Bfloat16,
///     512, 1024,           // rows × cols
///     1024 * 2,            // row stride in bytes
///     64, 64,              // tile rows × tile cols
/// )
/// .with_swizzle(CuTensorMapSwizzle::B128)
/// .params();
///
/// assert_eq!(params.num_dims, 2);
/// assert_eq!(params.global_dims[0], 1024); // cols (innermost)
/// assert_eq!(params.global_dims[1], 512);  // rows
/// assert_eq!(params.box_dims[0], 64);      // tile cols
/// assert_eq!(params.box_dims[1], 64);      // tile rows
/// ```
#[derive(Debug, Clone)]
pub struct TmaDescriptorBuilder {
    data_type: CuTensorMapDataType,
    num_dims: u32,
    global_dims: [u64; 5],
    global_strides: [u64; 4],
    box_dims: [u32; 5],
    element_strides: [u32; 5],
    interleave: CuTensorMapInterleave,
    swizzle: CuTensorMapSwizzle,
    l2_promotion: CuTensorMapL2Promotion,
    oob_fill: CuTensorMapFloatOobFill,
}

impl TmaDescriptorBuilder {
    /// Create a 2-D tiled TMA descriptor for a row-major matrix.
    ///
    /// # Parameters
    ///
    /// * `data_type` — element type.
    /// * `rows` — number of rows in the global tensor.
    /// * `cols` — number of columns in the global tensor.
    /// * `row_stride_bytes` — byte offset between consecutive rows in global
    ///   memory (often `cols * element_size`).
    /// * `box_rows` — tile height (rows per block in shared memory).
    /// * `box_cols` — tile width (cols per block in shared memory).
    ///
    /// # Panics
    ///
    /// Does not panic; invalid parameters will be caught by the driver when
    /// `cuTensorMapEncodeTiled` is called.
    #[must_use]
    pub fn new_2d(
        data_type: CuTensorMapDataType,
        rows: u64,
        cols: u64,
        row_stride_bytes: u64,
        box_rows: u32,
        box_cols: u32,
    ) -> Self {
        Self {
            data_type,
            num_dims: 2,
            // CUDA uses innermost-first ordering.
            global_dims: [cols, rows, 1, 1, 1],
            global_strides: [row_stride_bytes, 0, 0, 0],
            box_dims: [box_cols, box_rows, 1, 1, 1],
            element_strides: [1, 1, 1, 1, 1],
            interleave: CuTensorMapInterleave::None,
            swizzle: CuTensorMapSwizzle::B128,
            l2_promotion: CuTensorMapL2Promotion::L2B128,
            oob_fill: CuTensorMapFloatOobFill::None,
        }
    }

    /// Create an N-dimensional tiled TMA descriptor (N ≤ 5).
    ///
    /// # Parameters
    ///
    /// * `data_type` — element type.
    /// * `num_dims` — number of tensor dimensions (1–5).
    /// * `global_dims` — size of each dimension, innermost (col) first.
    /// * `global_strides` — byte stride for each outer dimension (`num_dims - 1` entries).
    /// * `box_dims` — tile extent per dimension.
    /// * `element_strides` — stride between elements in each tile dimension.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new_nd(
        data_type: CuTensorMapDataType,
        num_dims: u32,
        global_dims: [u64; 5],
        global_strides: [u64; 4],
        box_dims: [u32; 5],
        element_strides: [u32; 5],
    ) -> Self {
        Self {
            data_type,
            num_dims,
            global_dims,
            global_strides,
            box_dims,
            element_strides,
            interleave: CuTensorMapInterleave::None,
            swizzle: CuTensorMapSwizzle::B128,
            l2_promotion: CuTensorMapL2Promotion::L2B128,
            oob_fill: CuTensorMapFloatOobFill::None,
        }
    }

    /// Override the swizzle pattern (default: [`CuTensorMapSwizzle::B128`]).
    #[must_use]
    pub fn with_swizzle(mut self, swizzle: CuTensorMapSwizzle) -> Self {
        self.swizzle = swizzle;
        self
    }

    /// Override the interleave mode (default: [`CuTensorMapInterleave::None`]).
    #[must_use]
    pub fn with_interleave(mut self, interleave: CuTensorMapInterleave) -> Self {
        self.interleave = interleave;
        self
    }

    /// Override the L2 promotion hint (default: [`CuTensorMapL2Promotion::L2B128`]).
    #[must_use]
    pub fn with_l2_promotion(mut self, l2_promotion: CuTensorMapL2Promotion) -> Self {
        self.l2_promotion = l2_promotion;
        self
    }

    /// Override the out-of-bounds fill mode
    /// (default: [`CuTensorMapFloatOobFill::None`]).
    #[must_use]
    pub fn with_oob_fill(mut self, oob_fill: CuTensorMapFloatOobFill) -> Self {
        self.oob_fill = oob_fill;
        self
    }

    /// Finalise the builder and return the flat parameter struct.
    ///
    /// Pass the fields of the returned [`TmaEncodeTiledParams`] directly to
    /// `cuTensorMapEncodeTiled`.
    #[must_use]
    pub fn params(self) -> TmaEncodeTiledParams {
        TmaEncodeTiledParams {
            data_type: self.data_type,
            num_dims: self.num_dims,
            global_dims: self.global_dims,
            global_strides: self.global_strides,
            box_dims: self.box_dims,
            element_strides: self.element_strides,
            interleave: self.interleave,
            swizzle: self.swizzle,
            l2_promotion: self.l2_promotion,
            oob_fill: self.oob_fill,
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cu_tensor_map_size_and_alignment() {
        // Must be exactly 128 bytes and 64-byte aligned.
        assert_eq!(std::mem::size_of::<CuTensorMap>(), 128);
        assert_eq!(std::mem::align_of::<CuTensorMap>(), 64);
    }

    #[test]
    fn test_cu_tensor_map_default_is_zero() {
        let m = CuTensorMap::default();
        assert!(m.opaque.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_data_type_element_sizes() {
        assert_eq!(CuTensorMapDataType::Uint8.element_size_bytes(), 1);
        assert_eq!(CuTensorMapDataType::Float16.element_size_bytes(), 2);
        assert_eq!(CuTensorMapDataType::Bfloat16.element_size_bytes(), 2);
        assert_eq!(CuTensorMapDataType::Float32.element_size_bytes(), 4);
        assert_eq!(CuTensorMapDataType::Int32.element_size_bytes(), 4);
        assert_eq!(CuTensorMapDataType::Float64.element_size_bytes(), 8);
        assert_eq!(CuTensorMapDataType::Uint64.element_size_bytes(), 8);
    }

    #[test]
    fn test_tma_builder_2d_dimension_ordering() {
        // CUDA uses innermost-first — cols come before rows.
        let params = TmaDescriptorBuilder::new_2d(
            CuTensorMapDataType::Float16,
            1024, // rows
            2048, // cols
            2048 * 2,
            64,
            128,
        )
        .params();

        assert_eq!(params.num_dims, 2);
        assert_eq!(params.global_dims[0], 2048); // cols first
        assert_eq!(params.global_dims[1], 1024); // rows second
        assert_eq!(params.box_dims[0], 128); // tile cols
        assert_eq!(params.box_dims[1], 64); // tile rows
    }

    #[test]
    fn test_tma_builder_swizzle_override() {
        let params =
            TmaDescriptorBuilder::new_2d(CuTensorMapDataType::Float32, 64, 64, 64 * 4, 16, 16)
                .with_swizzle(CuTensorMapSwizzle::B64)
                .params();

        assert!(matches!(params.swizzle, CuTensorMapSwizzle::B64));
    }

    #[test]
    fn test_tma_builder_interleave_and_oob() {
        let params =
            TmaDescriptorBuilder::new_2d(CuTensorMapDataType::Uint8, 256, 256, 256, 32, 32)
                .with_interleave(CuTensorMapInterleave::B16)
                .with_oob_fill(CuTensorMapFloatOobFill::NanRequestZeroFma)
                .params();

        assert!(matches!(params.interleave, CuTensorMapInterleave::B16));
        assert!(matches!(
            params.oob_fill,
            CuTensorMapFloatOobFill::NanRequestZeroFma
        ));
    }

    #[test]
    fn test_tma_builder_l2_promotion() {
        let params = TmaDescriptorBuilder::new_2d(
            CuTensorMapDataType::Bfloat16,
            512,
            1024,
            1024 * 2,
            64,
            64,
        )
        .with_l2_promotion(CuTensorMapL2Promotion::L2B256)
        .params();

        assert!(matches!(
            params.l2_promotion,
            CuTensorMapL2Promotion::L2B256
        ));
    }

    #[test]
    fn test_enum_repr_values() {
        assert_eq!(CuTensorMapDataType::Uint8 as u32, 0);
        assert_eq!(CuTensorMapDataType::Float16 as u32, 6);
        assert_eq!(CuTensorMapDataType::Bfloat16 as u32, 9);
        assert_eq!(CuTensorMapDataType::TF32 as u32, 11);
        assert_eq!(CuTensorMapInterleave::None as u32, 0);
        assert_eq!(CuTensorMapInterleave::B32 as u32, 2);
        assert_eq!(CuTensorMapSwizzle::B128 as u32, 3);
        assert_eq!(CuTensorMapL2Promotion::L2B256 as u32, 3);
        assert_eq!(CuTensorMapFloatOobFill::NanRequestZeroFma as u32, 1);
    }

    #[test]
    fn test_nd_builder() {
        let params = TmaDescriptorBuilder::new_nd(
            CuTensorMapDataType::Float32,
            3,
            [512, 256, 128, 1, 1],
            [512 * 4, 512 * 256 * 4, 0, 0],
            [32, 16, 8, 1, 1],
            [1, 1, 1, 1, 1],
        )
        .params();

        assert_eq!(params.num_dims, 3);
        assert_eq!(params.global_dims[0], 512);
        assert_eq!(params.global_dims[1], 256);
        assert_eq!(params.global_dims[2], 128);
    }
}
