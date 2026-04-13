//! Core DNN type definitions.
//!
//! Provides tensor descriptors ([`TensorDesc`], [`TensorDescMut`]),
//! layout conventions ([`TensorLayout`]), activation functions
//! ([`Activation`]), convolution parameters ([`ConvolutionDescriptor`]),
//! and algorithm selection ([`ConvAlgorithm`]).

use std::marker::PhantomData;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_memory::DeviceBuffer;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// TensorLayout
// ---------------------------------------------------------------------------

/// Memory layout convention for multi-dimensional tensors.
///
/// The layout determines how logical indices map to linear memory offsets.
/// NHWC layouts are generally preferred on modern NVIDIA GPUs because they
/// enable Tensor Core utilisation, while NCHW is the traditional PyTorch
/// default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorLayout {
    /// Batch, Channels, Height, Width -- PyTorch default.
    Nchw,
    /// Batch, Height, Width, Channels -- Tensor Core optimal.
    Nhwc,
    /// Batch, Channels, Depth, Height, Width -- 3-D volumetric.
    Ncdhw,
    /// Batch, Depth, Height, Width, Channels -- 3-D channels-last.
    Ndhwc,
    /// Generic row-major layout for 2-D tensors (matrices) and MoE intermediates.
    RowMajor,
}

impl TensorLayout {
    /// Returns the number of spatial dimensions implied by this layout.
    #[inline]
    #[must_use]
    pub const fn spatial_dims(self) -> usize {
        match self {
            Self::Nchw | Self::Nhwc => 2,
            Self::Ncdhw | Self::Ndhwc => 3,
            Self::RowMajor => 0,
        }
    }

    /// Returns the expected number of tensor dimensions (including N and C).
    #[inline]
    #[must_use]
    pub const fn expected_ndim(self) -> usize {
        match self {
            Self::Nchw | Self::Nhwc => 4,
            Self::Ncdhw | Self::Ndhwc => 5,
            Self::RowMajor => 2,
        }
    }

    /// Returns `true` if this layout places channels last (NHWC or NDHWC).
    #[inline]
    #[must_use]
    pub const fn is_channels_last(self) -> bool {
        matches!(self, Self::Nhwc | Self::Ndhwc)
    }
}

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

/// Activation function types supported by DNN kernels.
///
/// These correspond to the most common activation functions used in deep
/// learning. Fused activation (e.g. conv + bias + ReLU) avoids extra
/// memory round-trips and is a key optimisation target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activation {
    /// Rectified Linear Unit: `max(0, x)`.
    Relu,
    /// Gaussian Error Linear Unit (exact): `x * Phi(x)`.
    Gelu,
    /// GELU approximated via tanh.
    GeluTanh,
    /// Sigmoid Linear Unit (SiLU / Swish): `x * sigmoid(x)`.
    Silu,
    /// Logistic sigmoid: `1 / (1 + exp(-x))`.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Identity (no activation applied).
    None,
}

// ---------------------------------------------------------------------------
// TensorDesc (immutable)
// ---------------------------------------------------------------------------

/// Immutable tensor descriptor binding a device pointer to shape metadata.
///
/// `TensorDesc` does **not** own the device memory; it merely borrows the
/// raw pointer for the duration of a DNN operation.  The caller must ensure
/// that the referenced [`DeviceBuffer`] outlives any computation that uses
/// this descriptor.
pub struct TensorDesc<T: GpuFloat> {
    /// Raw device pointer to the first element.
    pub ptr: CUdeviceptr,
    /// Shape (one entry per dimension).
    pub dims: Vec<u32>,
    /// Strides (one entry per dimension, in **elements** not bytes).
    pub strides: Vec<u32>,
    /// Memory layout convention.
    pub layout: TensorLayout,
    _phantom: PhantomData<T>,
}

impl<T: GpuFloat> TensorDesc<T> {
    /// Creates an NCHW tensor descriptor from a device buffer.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] if any dimension is zero.
    /// Returns [`DnnError::BufferTooSmall`] if the buffer cannot hold
    /// `n * c * h * w` elements.
    pub fn nchw(buf: &DeviceBuffer<T>, n: u32, c: u32, h: u32, w: u32) -> DnnResult<Self> {
        Self::validate_dims(&[n, c, h, w])?;
        let dims = vec![n, c, h, w];
        let strides = nchw_strides(c, h, w);
        let desc = Self {
            ptr: buf.as_device_ptr(),
            dims,
            strides,
            layout: TensorLayout::Nchw,
            _phantom: PhantomData,
        };
        desc.validate_buffer_size(buf)?;
        Ok(desc)
    }

    /// Creates an NHWC tensor descriptor from a device buffer.
    ///
    /// # Errors
    ///
    /// Same as [`nchw`](Self::nchw).
    pub fn nhwc(buf: &DeviceBuffer<T>, n: u32, c: u32, h: u32, w: u32) -> DnnResult<Self> {
        Self::validate_dims(&[n, c, h, w])?;
        let dims = vec![n, c, h, w];
        let strides = nhwc_strides(c, h, w);
        let desc = Self {
            ptr: buf.as_device_ptr(),
            dims,
            strides,
            layout: TensorLayout::Nhwc,
            _phantom: PhantomData,
        };
        desc.validate_buffer_size(buf)?;
        Ok(desc)
    }

    /// Creates an NCDHW (3-D volumetric) tensor descriptor.
    ///
    /// # Errors
    ///
    /// Same as [`nchw`](Self::nchw).
    pub fn ncdhw(buf: &DeviceBuffer<T>, n: u32, c: u32, d: u32, h: u32, w: u32) -> DnnResult<Self> {
        Self::validate_dims(&[n, c, d, h, w])?;
        let dims = vec![n, c, d, h, w];
        let strides = vec![c * d * h * w, d * h * w, h * w, w, 1];
        let desc = Self {
            ptr: buf.as_device_ptr(),
            dims,
            strides,
            layout: TensorLayout::Ncdhw,
            _phantom: PhantomData,
        };
        desc.validate_buffer_size(buf)?;
        Ok(desc)
    }

    /// Creates a 2-D matrix descriptor (rows x cols, row-major).
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] if either dimension is zero.
    /// Returns [`DnnError::BufferTooSmall`] if the buffer is too small.
    pub fn matrix(buf: &DeviceBuffer<T>, rows: u32, cols: u32) -> DnnResult<Self> {
        Self::validate_dims(&[rows, cols])?;
        let dims = vec![rows, cols];
        let strides = vec![cols, 1];
        let desc = Self {
            ptr: buf.as_device_ptr(),
            dims,
            strides,
            layout: TensorLayout::Nchw, // row-major, analogous to NCHW
            _phantom: PhantomData,
        };
        desc.validate_buffer_size(buf)?;
        Ok(desc)
    }

    /// Constructs a descriptor from raw components without buffer validation.
    ///
    /// The caller must ensure that `ptr` points to a valid device allocation
    /// large enough for the described tensor.
    pub fn from_raw(
        ptr: CUdeviceptr,
        dims: Vec<u32>,
        strides: Vec<u32>,
        layout: TensorLayout,
    ) -> DnnResult<Self> {
        if dims.len() != strides.len() {
            return Err(DnnError::InvalidDimension(format!(
                "dims length ({}) != strides length ({})",
                dims.len(),
                strides.len()
            )));
        }
        if dims.is_empty() {
            return Err(DnnError::InvalidDimension("empty dims".into()));
        }
        Ok(Self {
            ptr,
            dims,
            strides,
            layout,
            _phantom: PhantomData,
        })
    }

    /// Returns the total number of elements in the tensor.
    #[inline]
    #[must_use]
    pub fn numel(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Returns the number of dimensions.
    #[inline]
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Validates that `buf` is large enough to hold this tensor.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::BufferTooSmall`] if the buffer has fewer elements
    /// than [`numel`](Self::numel).
    pub fn validate_buffer_size(&self, buf: &DeviceBuffer<T>) -> DnnResult<()> {
        let required = self.numel() * T::SIZE;
        let actual = buf.len() * T::SIZE;
        if actual < required {
            return Err(DnnError::BufferTooSmall {
                expected: required,
                actual,
            });
        }
        Ok(())
    }

    /// Checks that no dimension is zero.
    fn validate_dims(dims: &[u32]) -> DnnResult<()> {
        for (i, &d) in dims.iter().enumerate() {
            if d == 0 {
                return Err(DnnError::InvalidDimension(format!("dimension {i} is zero")));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TensorDescMut (mutable output)
// ---------------------------------------------------------------------------

/// Mutable tensor descriptor for output buffers.
///
/// Identical to [`TensorDesc`] but signals that the referenced memory will
/// be written to.  Having a separate type prevents accidentally aliasing an
/// input and output tensor at the type level.
pub struct TensorDescMut<T: GpuFloat> {
    /// Raw device pointer to the first element (will be written).
    pub ptr: CUdeviceptr,
    /// Shape (one entry per dimension).
    pub dims: Vec<u32>,
    /// Strides (one entry per dimension, in elements).
    pub strides: Vec<u32>,
    /// Memory layout convention.
    pub layout: TensorLayout,
    _phantom: PhantomData<T>,
}

impl<T: GpuFloat> TensorDescMut<T> {
    /// Creates a mutable NCHW tensor descriptor from a device buffer.
    ///
    /// # Errors
    ///
    /// Same validation as [`TensorDesc::nchw`].
    pub fn nchw(buf: &mut DeviceBuffer<T>, n: u32, c: u32, h: u32, w: u32) -> DnnResult<Self> {
        validate_dims_helper(&[n, c, h, w])?;
        let numel = (n as usize) * (c as usize) * (h as usize) * (w as usize);
        validate_buf_size::<T>(buf.len(), numel)?;
        Ok(Self {
            ptr: buf.as_device_ptr(),
            dims: vec![n, c, h, w],
            strides: nchw_strides(c, h, w),
            layout: TensorLayout::Nchw,
            _phantom: PhantomData,
        })
    }

    /// Creates a mutable NHWC tensor descriptor from a device buffer.
    ///
    /// # Errors
    ///
    /// Same validation as [`TensorDesc::nhwc`].
    pub fn nhwc(buf: &mut DeviceBuffer<T>, n: u32, c: u32, h: u32, w: u32) -> DnnResult<Self> {
        validate_dims_helper(&[n, c, h, w])?;
        let numel = (n as usize) * (c as usize) * (h as usize) * (w as usize);
        validate_buf_size::<T>(buf.len(), numel)?;
        Ok(Self {
            ptr: buf.as_device_ptr(),
            dims: vec![n, c, h, w],
            strides: nhwc_strides(c, h, w),
            layout: TensorLayout::Nhwc,
            _phantom: PhantomData,
        })
    }

    /// Creates a mutable 2-D matrix descriptor (rows x cols, row-major).
    ///
    /// # Errors
    ///
    /// Same validation as [`TensorDesc::matrix`].
    pub fn matrix(buf: &mut DeviceBuffer<T>, rows: u32, cols: u32) -> DnnResult<Self> {
        validate_dims_helper(&[rows, cols])?;
        let numel = (rows as usize) * (cols as usize);
        validate_buf_size::<T>(buf.len(), numel)?;
        Ok(Self {
            ptr: buf.as_device_ptr(),
            dims: vec![rows, cols],
            strides: vec![cols, 1],
            layout: TensorLayout::Nchw,
            _phantom: PhantomData,
        })
    }

    /// Constructs a mutable descriptor from raw components.
    pub fn from_raw(
        ptr: CUdeviceptr,
        dims: Vec<u32>,
        strides: Vec<u32>,
        layout: TensorLayout,
    ) -> DnnResult<Self> {
        if dims.len() != strides.len() {
            return Err(DnnError::InvalidDimension(format!(
                "dims length ({}) != strides length ({})",
                dims.len(),
                strides.len()
            )));
        }
        if dims.is_empty() {
            return Err(DnnError::InvalidDimension("empty dims".into()));
        }
        Ok(Self {
            ptr,
            dims,
            strides,
            layout,
            _phantom: PhantomData,
        })
    }

    /// Returns the total number of elements in the tensor.
    #[inline]
    #[must_use]
    pub fn numel(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Returns the number of dimensions.
    #[inline]
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Borrows this mutable descriptor as an immutable [`TensorDesc`].
    #[must_use]
    pub fn as_immutable(&self) -> TensorDesc<T> {
        TensorDesc {
            ptr: self.ptr,
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            layout: self.layout,
            _phantom: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// ConvolutionDescriptor
// ---------------------------------------------------------------------------

/// Describes a convolution operation's hyper-parameters.
///
/// All vectors are indexed by spatial dimension (e.g. for 2-D convolutions
/// they have length 2, for 3-D length 3).
#[derive(Debug, Clone)]
pub struct ConvolutionDescriptor {
    /// Zero-padding applied to each spatial dimension (symmetric).
    pub padding: Vec<u32>,
    /// Stride of the convolution kernel in each spatial dimension.
    pub stride: Vec<u32>,
    /// Dilation factor in each spatial dimension.
    pub dilation: Vec<u32>,
    /// Number of groups for grouped/depthwise convolution.
    pub groups: u32,
}

impl ConvolutionDescriptor {
    /// Creates a standard 2-D convolution descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if stride or dilation contains
    /// a zero value, or if groups is zero.
    pub fn conv2d(
        pad_h: u32,
        pad_w: u32,
        stride_h: u32,
        stride_w: u32,
        dilation_h: u32,
        dilation_w: u32,
        groups: u32,
    ) -> DnnResult<Self> {
        if stride_h == 0 || stride_w == 0 {
            return Err(DnnError::InvalidArgument("stride must be non-zero".into()));
        }
        if dilation_h == 0 || dilation_w == 0 {
            return Err(DnnError::InvalidArgument(
                "dilation must be non-zero".into(),
            ));
        }
        if groups == 0 {
            return Err(DnnError::InvalidArgument("groups must be non-zero".into()));
        }
        Ok(Self {
            padding: vec![pad_h, pad_w],
            stride: vec![stride_h, stride_w],
            dilation: vec![dilation_h, dilation_w],
            groups,
        })
    }

    /// Returns the number of spatial dimensions this descriptor covers.
    #[inline]
    #[must_use]
    pub fn spatial_dims(&self) -> usize {
        self.padding.len()
    }

    /// Computes the output spatial size for a single dimension.
    ///
    /// Formula: `floor((input + 2*pad - dilation*(kernel-1) - 1) / stride) + 1`
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] if the computation underflows
    /// (i.e. the kernel is too large for the padded input).
    pub fn output_size(
        input: u32,
        kernel: u32,
        pad: u32,
        stride: u32,
        dilation: u32,
    ) -> DnnResult<u32> {
        let effective_kernel = dilation
            .checked_mul(kernel.saturating_sub(1))
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| DnnError::InvalidDimension("effective kernel size overflow".into()))?;
        let padded_input = input
            .checked_add(2 * pad)
            .ok_or_else(|| DnnError::InvalidDimension("padded input overflow".into()))?;
        if padded_input < effective_kernel {
            return Err(DnnError::InvalidDimension(format!(
                "padded input ({padded_input}) < effective kernel ({effective_kernel})"
            )));
        }
        Ok((padded_input - effective_kernel) / stride + 1)
    }
}

// ---------------------------------------------------------------------------
// ConvAlgorithm
// ---------------------------------------------------------------------------

/// Convolution algorithm selection.
///
/// Different algorithms offer different trade-offs between workspace memory
/// and compute throughput.  The optimal choice depends on tensor sizes, GPU
/// architecture, and available workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvAlgorithm {
    /// Implicit GEMM -- low workspace, moderate speed.
    ImplicitGemm,
    /// Im2col followed by explicit GEMM -- higher workspace, often fastest
    /// for medium-sized feature maps.
    Im2colGemm,
    /// Winograd transform -- fastest for 3x3 kernels with stride 1, but
    /// requires workspace and may reduce numerical precision.
    Winograd,
    /// Direct convolution -- no workspace, straightforward nested loops.
    Direct,
    /// FFT-based convolution -- fastest for very large kernels.
    FftConv,
}

// ---------------------------------------------------------------------------
// TileConfig
// ---------------------------------------------------------------------------

/// Tile configuration for tiled convolution kernels.
///
/// Controls work decomposition across thread blocks and warps.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Tile size in the M dimension (output spatial points per block).
    pub tile_m: u32,
    /// Tile size in the N dimension (output channels per block).
    pub tile_n: u32,
    /// Tile size in the K dimension (reduction loop step).
    pub tile_k: u32,
    /// Warp-level tile in M.
    pub warp_m: u32,
    /// Warp-level tile in N.
    pub warp_n: u32,
    /// Number of software pipeline stages.
    pub stages: u32,
}

impl TileConfig {
    /// Returns a default tile configuration for the given SM version.
    #[must_use]
    pub fn default_conv(sm: oxicuda_ptx::arch::SmVersion) -> Self {
        use oxicuda_ptx::arch::SmVersion;
        match sm {
            SmVersion::Sm90 | SmVersion::Sm90a | SmVersion::Sm100 | SmVersion::Sm120 => Self {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 64,
                warp_n: 64,
                stages: 4,
            },
            SmVersion::Sm80 | SmVersion::Sm86 | SmVersion::Sm89 => Self {
                tile_m: 128,
                tile_n: 128,
                tile_k: 32,
                warp_m: 64,
                warp_n: 64,
                stages: 3,
            },
            SmVersion::Sm75 => Self {
                tile_m: 64,
                tile_n: 64,
                tile_k: 32,
                warp_m: 32,
                warp_n: 32,
                stages: 2,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: pool / resize output size
// ---------------------------------------------------------------------------

/// Computes the output spatial dimension for a pooling operation.
///
/// `output_dim = floor((input_dim + 2 * padding - kernel_size) / stride) + 1`
///
/// Returns `None` if the resulting dimension would be zero or negative.
#[must_use]
pub fn pool_output_size(
    input_dim: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
) -> Option<u32> {
    if stride == 0 || kernel_size == 0 {
        return None;
    }
    let effective = input_dim + 2 * padding;
    if effective < kernel_size {
        return None;
    }
    Some((effective - kernel_size) / stride + 1)
}

// ---------------------------------------------------------------------------
// Stride helpers (private)
// ---------------------------------------------------------------------------

/// Computes NCHW strides: `[C*H*W, H*W, W, 1]`.
fn nchw_strides(c: u32, h: u32, w: u32) -> Vec<u32> {
    vec![c * h * w, h * w, w, 1]
}

/// Computes NHWC strides: `[H*W*C, 1, W*C, C]`.
fn nhwc_strides(c: u32, h: u32, w: u32) -> Vec<u32> {
    vec![h * w * c, 1, w * c, c]
}

/// Shared dimension validation.
fn validate_dims_helper(dims: &[u32]) -> DnnResult<()> {
    for (i, &d) in dims.iter().enumerate() {
        if d == 0 {
            return Err(DnnError::InvalidDimension(format!("dimension {i} is zero")));
        }
    }
    Ok(())
}

/// Validates buffer size against required element count.
fn validate_buf_size<T: GpuFloat>(buf_len: usize, required_numel: usize) -> DnnResult<()> {
    let required = required_numel * T::SIZE;
    let actual = buf_len * T::SIZE;
    if actual < required {
        return Err(DnnError::BufferTooSmall {
            expected: required,
            actual,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nchw_stride_order() {
        let s = nchw_strides(3, 4, 5);
        assert_eq!(s, vec![60, 20, 5, 1]);
    }

    #[test]
    fn nhwc_stride_order() {
        let s = nhwc_strides(3, 4, 5);
        // N-stride = H*W*C = 60, C-stride = 1, H-stride = W*C = 15, W-stride = C = 3
        assert_eq!(s, vec![60, 1, 15, 3]);
    }

    #[test]
    fn conv_output_size_basic() {
        // 32x32 input, 3x3 kernel, pad=1, stride=1, dilation=1 => 32
        let out = ConvolutionDescriptor::output_size(32, 3, 1, 1, 1);
        assert_eq!(out.ok(), Some(32));
    }

    #[test]
    fn conv_output_size_strided() {
        // 32x32 input, 3x3 kernel, pad=1, stride=2 => 16
        let out = ConvolutionDescriptor::output_size(32, 3, 1, 2, 1);
        assert_eq!(out.ok(), Some(16));
    }

    #[test]
    fn conv_output_size_dilated() {
        // 32x32 input, 3x3 kernel, pad=2, stride=1, dilation=2 => 32
        let out = ConvolutionDescriptor::output_size(32, 3, 2, 1, 2);
        assert_eq!(out.ok(), Some(32));
    }

    #[test]
    fn conv_output_size_too_small() {
        let out = ConvolutionDescriptor::output_size(3, 5, 0, 1, 1);
        assert!(out.is_err());
    }

    #[test]
    fn conv2d_zero_stride_rejected() {
        let r = ConvolutionDescriptor::conv2d(0, 0, 0, 1, 1, 1, 1);
        assert!(r.is_err());
    }

    #[test]
    fn conv2d_zero_groups_rejected() {
        let r = ConvolutionDescriptor::conv2d(0, 0, 1, 1, 1, 1, 0);
        assert!(r.is_err());
    }

    #[test]
    fn tensor_layout_spatial_dims() {
        assert_eq!(TensorLayout::Nchw.spatial_dims(), 2);
        assert_eq!(TensorLayout::Nhwc.spatial_dims(), 2);
        assert_eq!(TensorLayout::Ncdhw.spatial_dims(), 3);
        assert_eq!(TensorLayout::Ndhwc.spatial_dims(), 3);
    }

    #[test]
    fn tensor_layout_expected_ndim() {
        assert_eq!(TensorLayout::Nchw.expected_ndim(), 4);
        assert_eq!(TensorLayout::Ncdhw.expected_ndim(), 5);
    }

    #[test]
    fn from_raw_mismatched_lengths() {
        let r = TensorDesc::<f32>::from_raw(0, vec![1, 2], vec![1], TensorLayout::Nchw);
        assert!(r.is_err());
    }

    #[test]
    fn from_raw_empty_dims() {
        let r = TensorDesc::<f32>::from_raw(0, vec![], vec![], TensorLayout::Nchw);
        assert!(r.is_err());
    }

    #[test]
    fn activation_variants_are_distinct() {
        assert_ne!(Activation::Relu, Activation::Gelu);
        assert_ne!(Activation::Gelu, Activation::GeluTanh);
        assert_ne!(Activation::Silu, Activation::Sigmoid);
        assert_eq!(Activation::None, Activation::None);
    }

    #[test]
    fn conv_algorithm_debug() {
        let _ = format!("{:?}", ConvAlgorithm::Winograd);
    }

    #[test]
    fn pool_output_basic() {
        assert_eq!(pool_output_size(4, 2, 2, 0), Some(2));
        assert_eq!(pool_output_size(5, 3, 1, 1), Some(5));
    }

    #[test]
    fn pool_output_zero_stride() {
        assert_eq!(pool_output_size(4, 2, 0, 0), None);
    }

    #[test]
    fn pool_output_kernel_too_large() {
        assert_eq!(pool_output_size(2, 5, 1, 0), None);
    }
}
