//! Texture, array, and resource descriptor types for the CUDA Driver API.
//!
//! CUDA array handles, array/resource descriptors, texture descriptors,
//! resource views, and related enums for texture object creation.

use std::ffi::c_void;
use std::fmt;

use super::CUdeviceptr;

// =========================================================================
// CUarray / CUmipmappedArray — opaque CUDA array handles
// =========================================================================

/// Opaque handle to a CUDA array (1-D, 2-D, or 3-D texture memory).
///
/// Allocated by `cuArrayCreate_v2` / `cuArray3DCreate_v2` and freed by
/// `cuArrayDestroy`. Arrays can be bound to texture objects via
/// [`CUDA_RESOURCE_DESC`].
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CUarray(pub *mut c_void);

// SAFETY: CUDA handles are thread-safe when used with proper
// synchronisation via the driver API.
unsafe impl Send for CUarray {}
unsafe impl Sync for CUarray {}

impl fmt::Debug for CUarray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUarray({:p})", self.0)
    }
}

impl Default for CUarray {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl CUarray {
    /// Returns `true` if the handle is null (uninitialised).
    #[inline]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

/// Opaque handle to a CUDA mipmapped array (Mip-mapped texture memory).
///
/// Allocated by `cuMipmappedArrayCreate` and freed by
/// `cuMipmappedArrayDestroy`.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CUmipmappedArray(pub *mut c_void);

// SAFETY: CUDA handles are thread-safe when used with proper
// synchronisation via the driver API.
unsafe impl Send for CUmipmappedArray {}
unsafe impl Sync for CUmipmappedArray {}

impl fmt::Debug for CUmipmappedArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUmipmappedArray({:p})", self.0)
    }
}

impl Default for CUmipmappedArray {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl CUmipmappedArray {
    /// Returns `true` if the handle is null (uninitialised).
    #[inline]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

// =========================================================================
// CUarray_format — channel element format for CUDA arrays
// =========================================================================

/// Element format for CUDA arrays.  Mirrors `CUarray_format_enum` in the
/// CUDA driver API header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum CUarray_format {
    /// 8-bit unsigned integer channel.
    UnsignedInt8 = 0x01,
    /// 16-bit unsigned integer channel.
    UnsignedInt16 = 0x02,
    /// 32-bit unsigned integer channel.
    UnsignedInt32 = 0x03,
    /// 8-bit signed integer channel.
    SignedInt8 = 0x08,
    /// 16-bit signed integer channel.
    SignedInt16 = 0x09,
    /// 32-bit signed integer channel.
    SignedInt32 = 0x0a,
    /// 16-bit IEEE 754 half-precision float channel.
    Half = 0x10,
    /// 32-bit IEEE 754 single-precision float channel.
    Float = 0x20,
    /// NV12 planar YUV format (special 2-plane layout).
    Nv12 = 0xb0,
    /// 8-bit unsigned normalized integer (1 channel).
    UnormInt8X1 = 0xc0,
    /// 8-bit unsigned normalized integer (2 channels).
    UnormInt8X2 = 0xc1,
    /// 8-bit unsigned normalized integer (4 channels).
    UnormInt8X4 = 0xc2,
    /// 16-bit unsigned normalized integer (1 channel).
    UnormInt16X1 = 0xc3,
    /// 16-bit unsigned normalized integer (2 channels).
    UnormInt16X2 = 0xc4,
    /// 16-bit unsigned normalized integer (4 channels).
    UnormInt16X4 = 0xc5,
    /// 8-bit signed normalized integer (1 channel).
    SnormInt8X1 = 0xc6,
    /// 8-bit signed normalized integer (2 channels).
    SnormInt8X2 = 0xc7,
    /// 8-bit signed normalized integer (4 channels).
    SnormInt8X4 = 0xc8,
    /// 16-bit signed normalized integer (1 channel).
    SnormInt16X1 = 0xc9,
    /// 16-bit signed normalized integer (2 channels).
    SnormInt16X2 = 0xca,
    /// 16-bit signed normalized integer (4 channels).
    SnormInt16X4 = 0xcb,
    /// BC1 compressed (DXT1) unsigned.
    Bc1Unorm = 0x91,
    /// BC1 compressed (DXT1) unsigned, sRGB.
    Bc1UnormSrgb = 0x92,
    /// BC2 compressed (DXT3) unsigned.
    Bc2Unorm = 0x93,
    /// BC2 compressed (DXT3) unsigned, sRGB.
    Bc2UnormSrgb = 0x94,
    /// BC3 compressed (DXT5) unsigned.
    Bc3Unorm = 0x95,
    /// BC3 compressed (DXT5) unsigned, sRGB.
    Bc3UnormSrgb = 0x96,
    /// BC4 unsigned.
    Bc4Unorm = 0x97,
    /// BC4 signed.
    Bc4Snorm = 0x98,
    /// BC5 unsigned.
    Bc5Unorm = 0x99,
    /// BC5 signed.
    Bc5Snorm = 0x9a,
    /// BC6H unsigned 16-bit float.
    Bc6hUf16 = 0x9b,
    /// BC6H signed 16-bit float.
    Bc6hSf16 = 0x9c,
    /// BC7 unsigned.
    Bc7Unorm = 0x9d,
    /// BC7 unsigned, sRGB.
    Bc7UnormSrgb = 0x9e,
}

// =========================================================================
// CUresourcetype — resource type for texture/surface objects
// =========================================================================

/// Resource type discriminant for [`CUDA_RESOURCE_DESC`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
pub enum CUresourcetype {
    /// CUDA array resource.
    Array = 0x00,
    /// CUDA mipmapped array resource.
    MipmappedArray = 0x01,
    /// Linear memory resource (1-D, no filtering beyond point).
    Linear = 0x02,
    /// Pitched 2-D linear memory resource.
    Pitch2d = 0x03,
}

// =========================================================================
// CUaddress_mode — texture coordinate wrapping mode
// =========================================================================

/// Texture coordinate address-wrap mode for [`CUDA_TEXTURE_DESC`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum CUaddress_mode {
    /// Wrap (tiles) — coordinates outside [0, dim) wrap around.
    Wrap = 0,
    /// Clamp — coordinates are clamped to [0, dim-1].
    Clamp = 1,
    /// Mirror — coordinates are mirrored across array boundaries.
    Mirror = 2,
    /// Border — out-of-range coordinates return the border color.
    Border = 3,
}

// =========================================================================
// CUfilter_mode — texture / mipmap filtering mode
// =========================================================================

/// Texture / mipmap sampling filter mode for [`CUDA_TEXTURE_DESC`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum CUfilter_mode {
    /// Nearest-neighbor (point) sampling.
    Point = 0,
    /// Bilinear (linear) filtering.
    Linear = 1,
}

// =========================================================================
// CUresourceViewFormat — re-interpretation format for resource views
// =========================================================================

/// Format used to re-interpret a CUDA array in a resource view.
///
/// Mirrors `CUresourceViewFormat_enum` in the CUDA driver API header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
pub enum CUresourceViewFormat {
    /// No re-interpretation (use the array's own format).
    None = 0x00,
    /// Re-interpret as 1×8-bit unsigned integer.
    Uint1x8 = 0x01,
    /// Re-interpret as 2×8-bit unsigned integer.
    Uint2x8 = 0x02,
    /// Re-interpret as 4×8-bit unsigned integer.
    Uint4x8 = 0x03,
    /// Re-interpret as 1×8-bit signed integer.
    Sint1x8 = 0x04,
    /// Re-interpret as 2×8-bit signed integer.
    Sint2x8 = 0x05,
    /// Re-interpret as 4×8-bit signed integer.
    Sint4x8 = 0x06,
    /// Re-interpret as 1×16-bit unsigned integer.
    Uint1x16 = 0x07,
    /// Re-interpret as 2×16-bit unsigned integer.
    Uint2x16 = 0x08,
    /// Re-interpret as 4×16-bit unsigned integer.
    Uint4x16 = 0x09,
    /// Re-interpret as 1×16-bit signed integer.
    Sint1x16 = 0x0a,
    /// Re-interpret as 2×16-bit signed integer.
    Sint2x16 = 0x0b,
    /// Re-interpret as 4×16-bit signed integer.
    Sint4x16 = 0x0c,
    /// Re-interpret as 1×32-bit unsigned integer.
    Uint1x32 = 0x0d,
    /// Re-interpret as 2×32-bit unsigned integer.
    Uint2x32 = 0x0e,
    /// Re-interpret as 4×32-bit unsigned integer.
    Uint4x32 = 0x0f,
    /// Re-interpret as 1×32-bit signed integer.
    Sint1x32 = 0x10,
    /// Re-interpret as 2×32-bit signed integer.
    Sint2x32 = 0x11,
    /// Re-interpret as 4×32-bit signed integer.
    Sint4x32 = 0x12,
    /// Re-interpret as 1×16-bit float.
    Float1x16 = 0x13,
    /// Re-interpret as 2×16-bit float.
    Float2x16 = 0x14,
    /// Re-interpret as 4×16-bit float.
    Float4x16 = 0x15,
    /// Re-interpret as 1×32-bit float.
    Float1x32 = 0x16,
    /// Re-interpret as 2×32-bit float.
    Float2x32 = 0x17,
    /// Re-interpret as 4×32-bit float.
    Float4x32 = 0x18,
    /// BC1 unsigned normal compressed.
    UnsignedBc1 = 0x19,
    /// BC2 unsigned normal compressed.
    UnsignedBc2 = 0x1a,
    /// BC3 unsigned normal compressed.
    UnsignedBc3 = 0x1b,
    /// BC4 unsigned normal compressed.
    UnsignedBc4 = 0x1c,
    /// BC4 signed normal compressed.
    SignedBc4 = 0x1d,
    /// BC5 unsigned normal compressed.
    UnsignedBc5 = 0x1e,
    /// BC5 signed normal compressed.
    SignedBc5 = 0x1f,
    /// BC6H unsigned half-float.
    UnsignedBc6h = 0x20,
    /// BC6H signed half-float.
    SignedBc6h = 0x21,
    /// BC7 unsigned.
    UnsignedBc7 = 0x22,
    /// NV12 planar YUV.
    Nv12 = 0x23,
}

// =========================================================================
// CUDA_ARRAY_DESCRIPTOR — descriptor for 1-D and 2-D CUDA arrays
// =========================================================================

/// Descriptor passed to `cuArrayCreate_v2` / `cuArrayGetDescriptor_v2`.
///
/// Mirrors `CUDA_ARRAY_DESCRIPTOR_v2` in the CUDA driver API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CUDA_ARRAY_DESCRIPTOR {
    /// Width of the array in elements.
    pub width: usize,
    /// Height of the array in elements (0 for 1-D arrays).
    pub height: usize,
    /// Element format (data type of each channel).
    pub format: CUarray_format,
    /// Number of channels (1, 2, or 4).
    pub num_channels: u32,
}

// =========================================================================
// CUDA_ARRAY3D_DESCRIPTOR — descriptor for 3-D CUDA arrays
// =========================================================================

/// Descriptor passed to `cuArray3DCreate_v2` / `cuArray3DGetDescriptor_v2`.
///
/// Mirrors `CUDA_ARRAY3D_DESCRIPTOR_v2` in the CUDA driver API.  The `flags`
/// field accepts constants such as `CUDA_ARRAY3D_LAYERED` (0x01),
/// `CUDA_ARRAY3D_SURFACE_LDST` (0x02), `CUDA_ARRAY3D_CUBEMAP` (0x04), and
/// `CUDA_ARRAY3D_TEXTURE_GATHER` (0x08).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CUDA_ARRAY3D_DESCRIPTOR {
    /// Width of the array in elements.
    pub width: usize,
    /// Height of the array in elements (0 for 1-D arrays).
    pub height: usize,
    /// Depth of the array in elements (0 for 1-D and 2-D arrays).
    pub depth: usize,
    /// Element format.
    pub format: CUarray_format,
    /// Number of channels (1, 2, or 4).
    pub num_channels: u32,
    /// Creation flags (see [`CUDA_ARRAY3D_LAYERED`] etc.).
    pub flags: u32,
}

/// Flag: allocate a layered CUDA array (`CUDA_ARRAY3D_LAYERED`).
pub const CUDA_ARRAY3D_LAYERED: u32 = 0x01;
/// Flag: array usable as a surface load/store target (`CUDA_ARRAY3D_SURFACE_LDST`).
pub const CUDA_ARRAY3D_SURFACE_LDST: u32 = 0x02;
/// Flag: allocate a cubemap array (`CUDA_ARRAY3D_CUBEMAP`).
pub const CUDA_ARRAY3D_CUBEMAP: u32 = 0x04;
/// Flag: array usable with `cudaTextureGather` (`CUDA_ARRAY3D_TEXTURE_GATHER`).
pub const CUDA_ARRAY3D_TEXTURE_GATHER: u32 = 0x08;

// =========================================================================
// CUDA_RESOURCE_DESC — resource descriptor union for tex/surf objects
// =========================================================================

/// Inner data for an `Array` resource (variant of [`CudaResourceDescRes`]).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CudaResourceDescArray {
    /// CUDA array handle.
    pub h_array: CUarray,
}

/// Inner data for a `MipmappedArray` resource.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CudaResourceDescMipmap {
    /// Mipmapped array handle.
    pub h_mipmapped_array: CUmipmappedArray,
}

/// Inner data for a `Linear` (1-D linear memory) resource.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CudaResourceDescLinear {
    /// Device pointer to the linear region.
    pub dev_ptr: CUdeviceptr,
    /// Channel element format.
    pub format: CUarray_format,
    /// Number of channels.
    pub num_channels: u32,
    /// Total size in bytes.
    pub size_in_bytes: usize,
}

/// Inner data for a `Pitch2D` (2-D pitched linear memory) resource.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CudaResourceDescPitch2d {
    /// Device pointer to the pitched region (first row).
    pub dev_ptr: CUdeviceptr,
    /// Channel element format.
    pub format: CUarray_format,
    /// Number of channels.
    pub num_channels: u32,
    /// Width of the array in elements.
    pub width_in_elements: usize,
    /// Height of the array in elements.
    pub height: usize,
    /// Row pitch in bytes (stride between rows).
    pub pitch_in_bytes: usize,
}

/// Union of resource descriptors for [`CUDA_RESOURCE_DESC`].
///
/// # Safety
///
/// Callers must only read the field whose discriminant matches the
/// `res_type` field of the enclosing [`CUDA_RESOURCE_DESC`].
#[repr(C)]
pub union CudaResourceDescRes {
    /// Array resource.
    pub array: CudaResourceDescArray,
    /// Mipmapped array resource.
    pub mipmap: CudaResourceDescMipmap,
    /// 1-D linear memory resource.
    pub linear: CudaResourceDescLinear,
    /// 2-D pitched linear memory resource.
    pub pitch2d: CudaResourceDescPitch2d,
    /// Padding: ensures the union is 128 bytes (32 × i32), matching the ABI.
    pub reserved: [i32; 32],
}

/// Resource descriptor passed to `cuTexObjectCreate` / `cuSurfObjectCreate`.
///
/// Mirrors `CUDA_RESOURCE_DESC` in the CUDA driver API header.
#[repr(C)]
pub struct CUDA_RESOURCE_DESC {
    /// Identifies which union field inside `res` is valid.
    pub res_type: CUresourcetype,
    /// Resource payload — interpret via `res_type`.
    pub res: CudaResourceDescRes,
    /// Reserved flags (must be zero).
    pub flags: u32,
}

// =========================================================================
// CUDA_TEXTURE_DESC — texture object sampling parameters
// =========================================================================

/// Texture object descriptor passed to `cuTexObjectCreate`.
///
/// Mirrors `CUDA_TEXTURE_DESC` in the CUDA driver API.  All fields that the
/// caller does not set explicitly should be zeroed.
///
/// # Layout
///
/// The struct is `#[repr(C)]` and contains 64 bytes of reserved padding so
/// that it matches the binary ABI expected by the driver.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CUDA_TEXTURE_DESC {
    /// Address mode for each coordinate dimension (`[U, V, W]`).
    pub address_mode: [CUaddress_mode; 3],
    /// Texture filter mode (point or linear).
    pub filter_mode: CUfilter_mode,
    /// Flags: bit 0 = `CU_TRSF_READ_AS_INTEGER`, bit 1 = `CU_TRSF_NORMALIZED_COORDINATES`,
    /// bit 2 = `CU_TRSF_SRGB`, bit 3 = `CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION`.
    pub flags: u32,
    /// Maximum anisotropy ratio (1–16; 1 disables anisotropy).
    pub max_anisotropy: u32,
    /// Mipmap filter mode.
    pub mipmap_filter_mode: CUfilter_mode,
    /// Mipmap level-of-detail bias.
    pub mipmap_level_bias: f32,
    /// Minimum mipmap LOD clamp value.
    pub min_mipmap_level_clamp: f32,
    /// Maximum mipmap LOD clamp value.
    pub max_mipmap_level_clamp: f32,
    /// Border color (RGBA, applied when address mode is `Border`).
    pub border_color: [f32; 4],
    /// Reserved: must be zero.
    pub reserved: [i32; 12],
}

/// Flag: texture reads return raw integers (no type conversion).
pub const CU_TRSF_READ_AS_INTEGER: u32 = 0x01;
/// Flag: texture coordinates are normalized to [0, 1).
pub const CU_TRSF_NORMALIZED_COORDINATES: u32 = 0x02;
/// Flag: sRGB gamma encoding is applied during sampling.
pub const CU_TRSF_SRGB: u32 = 0x10;
/// Flag: disable hardware trilinear optimisation.
pub const CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION: u32 = 0x20;

// =========================================================================
// CUDA_RESOURCE_VIEW_DESC — optional re-interpretation of array resources
// =========================================================================

/// Optional resource view descriptor for `cuTexObjectCreate`.
///
/// Allows the caller to specify a sub-region, a different channel
/// interpretation format, or a mipmap range for a [`CUDA_RESOURCE_DESC`] that
/// wraps a CUDA array.  Pass a null pointer to `cuTexObjectCreate` to skip the
/// view override.
///
/// Mirrors `CUDA_RESOURCE_VIEW_DESC` in the CUDA driver API.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CUDA_RESOURCE_VIEW_DESC {
    /// Format to use for the resource view (re-interpretation).
    pub format: CUresourceViewFormat,
    /// Width of the view in elements.
    pub width: usize,
    /// Height of the view in elements.
    pub height: usize,
    /// Depth of the view in elements.
    pub depth: usize,
    /// First mipmap level included in the view.
    pub first_mipmap_level: u32,
    /// Last mipmap level included in the view.
    pub last_mipmap_level: u32,
    /// First array layer in a layered resource.
    pub first_layer: u32,
    /// Last array layer in a layered resource.
    pub last_layer: u32,
    /// Reserved: must be zero.
    pub reserved: [u32; 16],
}
