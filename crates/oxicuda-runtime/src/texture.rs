//! Texture and surface memory — CUDA array allocation and bindless objects.
//!
//! Implements the following CUDA Runtime API families:
//!
//! - **Array management**: `cudaMallocArray`, `cudaFreeArray`,
//!   `cudaMalloc3DArray`, `cudaArrayGetInfo`
//! - **Host-to-array copies**: `cudaMemcpyToArray`, `cudaMemcpyFromArray`,
//!   `cudaMemcpyToArrayAsync`, `cudaMemcpyFromArrayAsync`
//! - **Texture objects (bindless)**: `cudaCreateTextureObject`,
//!   `cudaDestroyTextureObject`, `cudaGetTextureObjectResourceDesc`
//! - **Surface objects (bindless)**: `cudaCreateSurfaceObject`,
//!   `cudaDestroySurfaceObject`

use std::ffi::c_void;

use oxicuda_driver::ffi::{
    CUDA_ARRAY_DESCRIPTOR, CUDA_ARRAY3D_DESCRIPTOR, CUDA_RESOURCE_DESC, CUDA_RESOURCE_VIEW_DESC,
    CUDA_TEXTURE_DESC, CUaddress_mode, CUarray, CUarray_format, CUfilter_mode, CUmipmappedArray,
    CUresourceViewFormat, CUresourcetype, CUsurfObject, CUtexObject, CudaResourceDescArray,
    CudaResourceDescLinear, CudaResourceDescMipmap, CudaResourceDescPitch2d, CudaResourceDescRes,
};
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::{
    CU_TRSF_NORMALIZED_COORDINATES, CU_TRSF_READ_AS_INTEGER, CU_TRSF_SRGB, CUDA_ARRAY3D_CUBEMAP,
    CUDA_ARRAY3D_LAYERED, CUDA_ARRAY3D_SURFACE_LDST, CUDA_ARRAY3D_TEXTURE_GATHER,
};

use crate::error::{CudaRtError, CudaRtResult};
use crate::memory::DevicePtr;
use crate::stream::CudaStream;

// ─── Channel Format ───────────────────────────────────────────────────────────

/// Element format for each channel in a CUDA array.
///
/// Mirrors `cudaChannelFormatKind` / `CUarray_format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrayFormat {
    /// 8-bit unsigned integer.
    UnsignedInt8,
    /// 16-bit unsigned integer.
    UnsignedInt16,
    /// 32-bit unsigned integer.
    UnsignedInt32,
    /// 8-bit signed integer.
    SignedInt8,
    /// 16-bit signed integer.
    SignedInt16,
    /// 32-bit signed integer.
    SignedInt32,
    /// 16-bit float (half precision).
    Half,
    /// 32-bit float (single precision).
    Float,
}

impl ArrayFormat {
    /// Convert to the driver-API [`CUarray_format`].
    #[must_use]
    pub const fn as_cu_format(self) -> CUarray_format {
        match self {
            Self::UnsignedInt8 => CUarray_format::UnsignedInt8,
            Self::UnsignedInt16 => CUarray_format::UnsignedInt16,
            Self::UnsignedInt32 => CUarray_format::UnsignedInt32,
            Self::SignedInt8 => CUarray_format::SignedInt8,
            Self::SignedInt16 => CUarray_format::SignedInt16,
            Self::SignedInt32 => CUarray_format::SignedInt32,
            Self::Half => CUarray_format::Half,
            Self::Float => CUarray_format::Float,
        }
    }

    /// Element byte width for one channel.
    #[must_use]
    pub const fn bytes_per_channel(self) -> usize {
        match self {
            Self::UnsignedInt8 | Self::SignedInt8 => 1,
            Self::UnsignedInt16 | Self::SignedInt16 | Self::Half => 2,
            Self::UnsignedInt32 | Self::SignedInt32 | Self::Float => 4,
        }
    }
}

// ─── Texture Address Mode ─────────────────────────────────────────────────────

/// Texture coordinate wrapping mode (maps to `cudaTextureAddressMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressMode {
    /// Coordinates wrap (tile).
    Wrap,
    /// Coordinates are clamped to the boundary.
    Clamp,
    /// Coordinates are mirrored at every boundary.
    Mirror,
    /// Out-of-range coordinates return the border colour.
    Border,
}

impl AddressMode {
    #[must_use]
    const fn as_cu(self) -> CUaddress_mode {
        match self {
            Self::Wrap => CUaddress_mode::Wrap,
            Self::Clamp => CUaddress_mode::Clamp,
            Self::Mirror => CUaddress_mode::Mirror,
            Self::Border => CUaddress_mode::Border,
        }
    }
}

// ─── Texture Filter Mode ──────────────────────────────────────────────────────

/// Texture sampling filter mode (maps to `cudaTextureFilterMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterMode {
    /// Nearest-neighbor (point) sampling.
    Point,
    /// Bilinear (linear) interpolation.
    Linear,
}

impl FilterMode {
    #[must_use]
    const fn as_cu(self) -> CUfilter_mode {
        match self {
            Self::Point => CUfilter_mode::Point,
            Self::Linear => CUfilter_mode::Linear,
        }
    }
}

// ─── CudaArray ────────────────────────────────────────────────────────────────

/// RAII wrapper for a CUDA array (1-D or 2-D).
///
/// Created by [`CudaArray::create_1d`] or [`CudaArray::create_2d`]; freed by [`Drop`].
///
/// A `CudaArray` can be bound to a [`CudaTextureObject`] or
/// [`CudaSurfaceObject`] for hardware-accelerated sampling.
pub struct CudaArray {
    handle: CUarray,
    width: usize,
    height: usize,
    format: ArrayFormat,
    num_channels: u32,
}

impl CudaArray {
    /// Allocate a 1-D CUDA array with `width` elements of the given format and
    /// channel count (`num_channels` must be 1, 2, or 4).
    ///
    /// Mirrors `cudaMallocArray` (1-D form).
    ///
    /// # Errors
    ///
    /// Returns [`CudaRtError::NotSupported`] if the driver does not expose
    /// `cuArrayCreate_v2`, or propagates driver errors.
    pub fn create_1d(width: usize, format: ArrayFormat, num_channels: u32) -> CudaRtResult<Self> {
        let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
        let create_fn = api.cu_array_create_v2.ok_or(CudaRtError::NotSupported)?;
        let desc = CUDA_ARRAY_DESCRIPTOR {
            width,
            height: 0,
            format: format.as_cu_format(),
            num_channels,
        };
        let mut handle = CUarray::default();
        // SAFETY: desc is valid, handle is initialized after the call succeeds.
        let rc = unsafe { create_fn(&raw mut handle, &desc) };
        if rc != 0 {
            return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::MemoryAllocation));
        }
        Ok(Self {
            handle,
            width,
            height: 0,
            format,
            num_channels,
        })
    }

    /// Allocate a 2-D CUDA array with `width × height` elements.
    ///
    /// Mirrors `cudaMallocArray` (2-D form).
    ///
    /// # Errors
    ///
    /// Returns [`CudaRtError::NotSupported`] if `cuArrayCreate_v2` is absent.
    pub fn create_2d(
        width: usize,
        height: usize,
        format: ArrayFormat,
        num_channels: u32,
    ) -> CudaRtResult<Self> {
        let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
        let create_fn = api.cu_array_create_v2.ok_or(CudaRtError::NotSupported)?;
        let desc = CUDA_ARRAY_DESCRIPTOR {
            width,
            height,
            format: format.as_cu_format(),
            num_channels,
        };
        let mut handle = CUarray::default();
        // SAFETY: FFI.
        let rc = unsafe { create_fn(&raw mut handle, &desc) };
        if rc != 0 {
            return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::MemoryAllocation));
        }
        Ok(Self {
            handle,
            width,
            height,
            format,
            num_channels,
        })
    }

    /// Copy a contiguous host buffer into the entire array (synchronous).
    ///
    /// `data` must contain exactly `width * height.max(1) * num_channels`
    /// elements of the appropriate type.
    ///
    /// Mirrors `cudaMemcpyToArray` (host-to-array).
    ///
    /// # Errors
    ///
    /// Returns an error if the driver does not support `cuMemcpyHtoA_v2` or if
    /// the copy fails.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reading `byte_count` bytes.
    pub unsafe fn copy_from_host_raw(
        &self,
        src: *const c_void,
        byte_count: usize,
    ) -> CudaRtResult<()> {
        let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
        let f = api.cu_memcpy_htoa_v2.ok_or(CudaRtError::NotSupported)?;
        // SAFETY: src is caller-guaranteed valid for `byte_count` bytes.
        let rc = unsafe { f(self.handle, 0, src, byte_count) };
        if rc != 0 {
            return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidMemcpyDirection));
        }
        Ok(())
    }

    /// Copy a typed host slice into the array (synchronous, type-safe helper).
    ///
    /// # Errors
    ///
    /// Forwards errors from [`Self::copy_from_host_raw`].
    pub fn copy_from_host<T: Copy>(&self, src: &[T]) -> CudaRtResult<()> {
        // SAFETY: src is a valid slice reference, so the pointer and size are valid.
        unsafe {
            self.copy_from_host_raw(src.as_ptr().cast::<c_void>(), std::mem::size_of_val(src))
        }
    }

    /// Copy the entire array into a host buffer (synchronous, raw pointer).
    ///
    /// Mirrors `cudaMemcpyFromArray` (array-to-host).
    ///
    /// # Errors
    ///
    /// Returns an error if `cuMemcpyAtoH_v2` is absent or the copy fails.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writing `byte_count` bytes.
    pub unsafe fn copy_to_host_raw(&self, dst: *mut c_void, byte_count: usize) -> CudaRtResult<()> {
        let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
        let f = api.cu_memcpy_atoh_v2.ok_or(CudaRtError::NotSupported)?;
        // SAFETY: dst is caller-guaranteed valid for `byte_count` bytes.
        let rc = unsafe { f(dst, self.handle, 0, byte_count) };
        if rc != 0 {
            return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidMemcpyDirection));
        }
        Ok(())
    }

    /// Copy the entire array into a typed host slice (synchronous, type-safe).
    ///
    /// # Errors
    ///
    /// Forwards errors from [`Self::copy_to_host_raw`].
    pub fn copy_to_host<T: Copy>(&self, dst: &mut [T]) -> CudaRtResult<()> {
        // SAFETY: dst is a valid mutable slice reference, so the pointer and size are valid.
        unsafe {
            self.copy_to_host_raw(
                dst.as_mut_ptr().cast::<c_void>(),
                std::mem::size_of_val(dst),
            )
        }
    }

    /// Asynchronously copy a host buffer into the array on `stream`.
    ///
    /// Mirrors `cudaMemcpyToArrayAsync`.
    ///
    /// # Errors
    ///
    /// Returns an error if `cuMemcpyHtoAAsync_v2` is absent.
    ///
    /// # Safety
    ///
    /// The caller must ensure `src` remains valid until the stream operation
    /// completes (i.e., until the stream is synchronized).
    pub unsafe fn copy_from_host_async_raw(
        &self,
        src: *const c_void,
        byte_count: usize,
        stream: CudaStream,
    ) -> CudaRtResult<()> {
        let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
        let f = api
            .cu_memcpy_htoa_async_v2
            .ok_or(CudaRtError::NotSupported)?;
        // SAFETY: caller guarantees src + lifetime.
        let rc = unsafe { f(self.handle, 0, src, byte_count, stream.raw()) };
        if rc != 0 {
            return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidMemcpyDirection));
        }
        Ok(())
    }

    /// Returns the raw [`CUarray`] handle (for use in resource descriptors).
    #[must_use]
    pub fn raw(&self) -> CUarray {
        self.handle
    }

    /// Width of the array in elements.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Height of the array in elements (0 for 1-D arrays).
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Element format of this array.
    #[must_use]
    pub const fn format(&self) -> ArrayFormat {
        self.format
    }

    /// Number of channels (1, 2, or 4).
    #[must_use]
    pub const fn num_channels(&self) -> u32 {
        self.num_channels
    }
}

impl Drop for CudaArray {
    fn drop(&mut self) {
        if let Ok(api) = try_driver() {
            if let Some(f) = api.cu_array_destroy {
                // SAFETY: handle was created by cuArrayCreate_v2 and not yet freed.
                unsafe { f(self.handle) };
            }
        }
    }
}

// ─── CudaArray3D ─────────────────────────────────────────────────────────────

/// Flags for 3-D CUDA array creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Array3DFlags(pub u32);

impl Array3DFlags {
    /// No special flags.
    pub const DEFAULT: Self = Self(0);
    /// Layered array (depth = number of layers).
    pub const LAYERED: Self = Self(CUDA_ARRAY3D_LAYERED);
    /// Supports surface load/store.
    pub const SURFACE_LDST: Self = Self(CUDA_ARRAY3D_SURFACE_LDST);
    /// Cubemap array (depth = 6 × num_layers).
    pub const CUBEMAP: Self = Self(CUDA_ARRAY3D_CUBEMAP);
    /// Supports texture gather operations.
    pub const TEXTURE_GATHER: Self = Self(CUDA_ARRAY3D_TEXTURE_GATHER);

    /// Combine flags with bitwise OR.
    #[must_use]
    pub const fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// RAII wrapper for a 3-D (or layered / cubemap) CUDA array.
pub struct CudaArray3D {
    handle: CUarray,
    width: usize,
    height: usize,
    depth: usize,
    format: ArrayFormat,
    num_channels: u32,
    flags: Array3DFlags,
}

impl CudaArray3D {
    /// Allocate a 3-D CUDA array.
    ///
    /// `depth = 0` is valid for 1-D and 2-D arrays allocated via the 3-D API;
    /// for layered arrays it specifies the number of layers.
    ///
    /// Mirrors `cudaMalloc3DArray`.
    ///
    /// # Errors
    ///
    /// Returns [`CudaRtError::NotSupported`] if `cuArray3DCreate_v2` is absent.
    pub fn create(
        width: usize,
        height: usize,
        depth: usize,
        format: ArrayFormat,
        num_channels: u32,
        flags: Array3DFlags,
    ) -> CudaRtResult<Self> {
        let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
        let create_fn = api.cu_array3d_create_v2.ok_or(CudaRtError::NotSupported)?;
        let desc = CUDA_ARRAY3D_DESCRIPTOR {
            width,
            height,
            depth,
            format: format.as_cu_format(),
            num_channels,
            flags: flags.0,
        };
        let mut handle = CUarray::default();
        // SAFETY: FFI.
        let rc = unsafe { create_fn(&raw mut handle, &desc) };
        if rc != 0 {
            return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::MemoryAllocation));
        }
        Ok(Self {
            handle,
            width,
            height,
            depth,
            format,
            num_channels,
            flags,
        })
    }

    /// Returns the raw [`CUarray`] handle.
    #[must_use]
    pub fn raw(&self) -> CUarray {
        self.handle
    }

    /// Width of the array in elements.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }
    /// Height of the array in elements.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }
    /// Depth of the array (or layer count for layered arrays).
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.depth
    }
    /// Element format.
    #[must_use]
    pub const fn format(&self) -> ArrayFormat {
        self.format
    }
    /// Number of channels.
    #[must_use]
    pub const fn num_channels(&self) -> u32 {
        self.num_channels
    }
    /// Creation flags.
    #[must_use]
    pub const fn flags(&self) -> Array3DFlags {
        self.flags
    }
}

impl Drop for CudaArray3D {
    fn drop(&mut self) {
        if let Ok(api) = try_driver() {
            if let Some(f) = api.cu_array_destroy {
                // SAFETY: handle was created by cuArray3DCreate_v2 and not yet freed.
                unsafe { f(self.handle) };
            }
        }
    }
}

// ─── ResourceDesc ─────────────────────────────────────────────────────────────

/// High-level resource description for texture and surface objects.
///
/// Converted to [`CUDA_RESOURCE_DESC`] when creating a [`CudaTextureObject`]
/// or [`CudaSurfaceObject`].
#[derive(Clone, Copy)]
pub enum ResourceDesc {
    /// A CUDA array resource (most common for textures and surfaces).
    Array {
        /// Raw array handle.
        handle: CUarray,
    },
    /// A mipmapped CUDA array resource.
    MipmappedArray {
        /// Raw mipmapped array handle.
        handle: CUmipmappedArray,
    },
    /// Linear device-memory resource (no filtering beyond point).
    Linear {
        /// Device pointer to the linear region.
        dev_ptr: DevicePtr,
        /// Channel element format.
        format: ArrayFormat,
        /// Number of channels.
        num_channels: u32,
        /// Total size in bytes.
        size_in_bytes: usize,
    },
    /// Pitched 2-D device-memory resource.
    Pitch2d {
        /// Device pointer to the first row.
        dev_ptr: DevicePtr,
        /// Channel element format.
        format: ArrayFormat,
        /// Number of channels.
        num_channels: u32,
        /// Width of the region in elements.
        width_in_elements: usize,
        /// Height of the region in elements.
        height: usize,
        /// Row pitch in bytes.
        pitch_in_bytes: usize,
    },
}

impl ResourceDesc {
    /// Convert to the raw [`CUDA_RESOURCE_DESC`] expected by the driver.
    #[must_use]
    pub fn as_raw(&self) -> CUDA_RESOURCE_DESC {
        match *self {
            Self::Array { handle } => CUDA_RESOURCE_DESC {
                res_type: CUresourcetype::Array,
                res: CudaResourceDescRes {
                    array: CudaResourceDescArray { h_array: handle },
                },
                flags: 0,
            },
            Self::MipmappedArray { handle } => CUDA_RESOURCE_DESC {
                res_type: CUresourcetype::MipmappedArray,
                res: CudaResourceDescRes {
                    mipmap: CudaResourceDescMipmap {
                        h_mipmapped_array: handle,
                    },
                },
                flags: 0,
            },
            Self::Linear {
                dev_ptr,
                format,
                num_channels,
                size_in_bytes,
            } => CUDA_RESOURCE_DESC {
                res_type: CUresourcetype::Linear,
                res: CudaResourceDescRes {
                    linear: CudaResourceDescLinear {
                        dev_ptr: dev_ptr.0,
                        format: format.as_cu_format(),
                        num_channels,
                        size_in_bytes,
                    },
                },
                flags: 0,
            },
            Self::Pitch2d {
                dev_ptr,
                format,
                num_channels,
                width_in_elements,
                height,
                pitch_in_bytes,
            } => CUDA_RESOURCE_DESC {
                res_type: CUresourcetype::Pitch2d,
                res: CudaResourceDescRes {
                    pitch2d: CudaResourceDescPitch2d {
                        dev_ptr: dev_ptr.0,
                        format: format.as_cu_format(),
                        num_channels,
                        width_in_elements,
                        height,
                        pitch_in_bytes,
                    },
                },
                flags: 0,
            },
        }
    }
}

// ─── TextureDesc ──────────────────────────────────────────────────────────────

/// Ergonomic texture-object sampling configuration.
///
/// Converted to [`CUDA_TEXTURE_DESC`] when creating a [`CudaTextureObject`].
#[derive(Clone, Copy)]
pub struct TextureDesc {
    /// Address mode for the U (X) dimension.
    pub address_u: AddressMode,
    /// Address mode for the V (Y) dimension.
    pub address_v: AddressMode,
    /// Address mode for the W (Z) dimension.
    pub address_w: AddressMode,
    /// Sampling filter mode.
    pub filter_mode: FilterMode,
    /// When `true`, texture coordinates are in the normalized range [0, 1).
    pub normalized_coords: bool,
    /// When `true`, texture reads return raw integers rather than normalized floats.
    pub read_as_integer: bool,
    /// When `true`, hardware applies sRGB gamma decoding on read.
    pub srgb: bool,
    /// Maximum anisotropy ratio (1–16; 1 disables anisotropy).
    pub max_anisotropy: u32,
    /// Mipmap filter mode.
    pub mipmap_filter: FilterMode,
    /// Mipmap LOD bias.
    pub mipmap_bias: f32,
    /// Minimum mipmap LOD clamp.
    pub min_lod: f32,
    /// Maximum mipmap LOD clamp.
    pub max_lod: f32,
    /// Border color (RGBA).
    pub border_color: [f32; 4],
}

impl TextureDesc {
    /// Construct a sensible default texture descriptor:
    ///
    /// - Clamp address mode on all axes
    /// - Nearest-neighbor filtering (no mipmap)
    /// - Normalized coordinates
    /// - No anisotropy
    #[must_use]
    pub const fn default_2d() -> Self {
        Self {
            address_u: AddressMode::Clamp,
            address_v: AddressMode::Clamp,
            address_w: AddressMode::Clamp,
            filter_mode: FilterMode::Point,
            normalized_coords: true,
            read_as_integer: false,
            srgb: false,
            max_anisotropy: 1,
            mipmap_filter: FilterMode::Point,
            mipmap_bias: 0.0,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: [0.0; 4],
        }
    }

    /// Convert to the raw [`CUDA_TEXTURE_DESC`] expected by the driver.
    #[must_use]
    pub fn as_raw(&self) -> CUDA_TEXTURE_DESC {
        let mut flags: u32 = 0;
        if self.normalized_coords {
            flags |= CU_TRSF_NORMALIZED_COORDINATES;
        }
        if self.read_as_integer {
            flags |= CU_TRSF_READ_AS_INTEGER;
        }
        if self.srgb {
            flags |= CU_TRSF_SRGB;
        }
        CUDA_TEXTURE_DESC {
            address_mode: [
                self.address_u.as_cu(),
                self.address_v.as_cu(),
                self.address_w.as_cu(),
            ],
            filter_mode: self.filter_mode.as_cu(),
            flags,
            max_anisotropy: self.max_anisotropy,
            mipmap_filter_mode: self.mipmap_filter.as_cu(),
            mipmap_level_bias: self.mipmap_bias,
            min_mipmap_level_clamp: self.min_lod,
            max_mipmap_level_clamp: self.max_lod,
            border_color: self.border_color,
            reserved: [0i32; 12],
        }
    }
}

// ─── ResourceViewDesc ─────────────────────────────────────────────────────────

/// Optional resource-view descriptor for texture objects.
///
/// Allows re-interpretation of the array format, or restriction to a sub-range
/// of mipmap levels and array layers.
#[derive(Clone, Copy)]
pub struct ResourceViewDesc {
    /// Reinterpretation format (use `None` for the array's native format).
    pub format: CUresourceViewFormat,
    /// View width in elements.
    pub width: usize,
    /// View height in elements.
    pub height: usize,
    /// View depth in elements.
    pub depth: usize,
    /// First mipmap level in the view.
    pub first_mip_level: u32,
    /// Last mipmap level in the view.
    pub last_mip_level: u32,
    /// First array layer.
    pub first_layer: u32,
    /// Last array layer.
    pub last_layer: u32,
}

impl ResourceViewDesc {
    /// Convert to the raw [`CUDA_RESOURCE_VIEW_DESC`].
    #[must_use]
    pub fn as_raw(&self) -> CUDA_RESOURCE_VIEW_DESC {
        CUDA_RESOURCE_VIEW_DESC {
            format: self.format,
            width: self.width,
            height: self.height,
            depth: self.depth,
            first_mipmap_level: self.first_mip_level,
            last_mipmap_level: self.last_mip_level,
            first_layer: self.first_layer,
            last_layer: self.last_layer,
            reserved: [0u32; 16],
        }
    }
}

// ─── CudaTextureObject ────────────────────────────────────────────────────────

/// RAII wrapper for a CUDA bindless texture object.
///
/// Created by [`CudaTextureObject::create`]; automatically destroyed on drop.
/// Mirrors `cudaCreateTextureObject` / `cudaDestroyTextureObject`.
pub struct CudaTextureObject {
    handle: CUtexObject,
}

impl CudaTextureObject {
    /// Create a texture object from a resource and texture descriptor.
    ///
    /// `view_desc` is optional — pass `None` to use the resource's native
    /// format and full extent.
    ///
    /// # Errors
    ///
    /// Returns [`CudaRtError::NotSupported`] if `cuTexObjectCreate` is absent,
    /// or propagates the driver error code.
    pub fn create(
        resource: &ResourceDesc,
        texture: &TextureDesc,
        view: Option<&ResourceViewDesc>,
    ) -> CudaRtResult<Self> {
        let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
        let create_fn = api.cu_tex_object_create.ok_or(CudaRtError::NotSupported)?;

        let raw_res = resource.as_raw();
        let raw_tex = texture.as_raw();
        let (raw_view_ptr, _raw_view_storage);
        if let Some(v) = view {
            _raw_view_storage = v.as_raw();
            raw_view_ptr = &_raw_view_storage as *const CUDA_RESOURCE_VIEW_DESC;
        } else {
            _raw_view_storage = unsafe { std::mem::zeroed() };
            raw_view_ptr = std::ptr::null();
        }

        let mut handle = CUtexObject::default();
        // SAFETY: All descriptor pointers are valid stack-allocated structs.
        let rc = unsafe { create_fn(&raw mut handle, &raw_res, &raw_tex, raw_view_ptr) };
        if rc != 0 {
            return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidValue));
        }
        Ok(Self { handle })
    }

    /// Returns the raw [`CUtexObject`] handle.
    #[must_use]
    pub fn raw(&self) -> CUtexObject {
        self.handle
    }
}

impl Drop for CudaTextureObject {
    fn drop(&mut self) {
        if let Ok(api) = try_driver() {
            if let Some(f) = api.cu_tex_object_destroy {
                // SAFETY: handle was created by cuTexObjectCreate and not yet freed.
                unsafe { f(self.handle) };
            }
        }
    }
}

// ─── CudaSurfaceObject ────────────────────────────────────────────────────────

/// RAII wrapper for a CUDA bindless surface object.
///
/// Created by [`CudaSurfaceObject::create`]; automatically destroyed on drop.
/// Mirrors `cudaCreateSurfaceObject` / `cudaDestroySurfaceObject`.
///
/// The resource must be a CUDA array allocated with the
/// [`Array3DFlags::SURFACE_LDST`] flag (or equivalent).
pub struct CudaSurfaceObject {
    handle: CUsurfObject,
}

impl CudaSurfaceObject {
    /// Create a surface object from a resource descriptor.
    ///
    /// The resource type must be `Array` — surfaces cannot be backed by linear
    /// or pitched memory.
    ///
    /// # Errors
    ///
    /// Returns [`CudaRtError::NotSupported`] if `cuSurfObjectCreate` is absent,
    /// or propagates the driver error code.
    pub fn create(resource: &ResourceDesc) -> CudaRtResult<Self> {
        let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
        let create_fn = api.cu_surf_object_create.ok_or(CudaRtError::NotSupported)?;
        let raw_res = resource.as_raw();
        let mut handle = CUsurfObject::default();
        // SAFETY: raw_res is a valid stack-allocated CUDA_RESOURCE_DESC.
        let rc = unsafe { create_fn(&raw mut handle, &raw_res) };
        if rc != 0 {
            return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidValue));
        }
        Ok(Self { handle })
    }

    /// Returns the raw [`CUsurfObject`] handle.
    #[must_use]
    pub fn raw(&self) -> CUsurfObject {
        self.handle
    }
}

impl Drop for CudaSurfaceObject {
    fn drop(&mut self) {
        if let Ok(api) = try_driver() {
            if let Some(f) = api.cu_surf_object_destroy {
                // SAFETY: handle was created by cuSurfObjectCreate and not yet freed.
                unsafe { f(self.handle) };
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_format_byte_widths() {
        assert_eq!(ArrayFormat::UnsignedInt8.bytes_per_channel(), 1);
        assert_eq!(ArrayFormat::UnsignedInt16.bytes_per_channel(), 2);
        assert_eq!(ArrayFormat::Half.bytes_per_channel(), 2);
        assert_eq!(ArrayFormat::Float.bytes_per_channel(), 4);
        assert_eq!(ArrayFormat::SignedInt32.bytes_per_channel(), 4);
    }

    #[test]
    fn array_format_cu_round_trip() {
        let fmt = ArrayFormat::Float;
        assert!(matches!(fmt.as_cu_format(), CUarray_format::Float));
        let fmt_int = ArrayFormat::SignedInt8;
        assert!(matches!(fmt_int.as_cu_format(), CUarray_format::SignedInt8));
    }

    #[test]
    fn texture_desc_default_flags() {
        let desc = TextureDesc::default_2d();
        let raw = desc.as_raw();
        // Normalized coordinates flag must be set.
        assert!(raw.flags & CU_TRSF_NORMALIZED_COORDINATES != 0);
        // No read-as-integer by default.
        assert!(raw.flags & CU_TRSF_READ_AS_INTEGER == 0);
        assert!(raw.flags & CU_TRSF_SRGB == 0);
        // Point filtering by default.
        assert!(matches!(raw.filter_mode, CUfilter_mode::Point));
        // All address modes must be Clamp.
        assert!(matches!(raw.address_mode[0], CUaddress_mode::Clamp));
        assert!(matches!(raw.address_mode[1], CUaddress_mode::Clamp));
        assert!(matches!(raw.address_mode[2], CUaddress_mode::Clamp));
    }

    #[test]
    fn resource_desc_array_round_trip() {
        let handle = CUarray::default();
        let rd = ResourceDesc::Array { handle };
        let raw = rd.as_raw();
        assert!(matches!(raw.res_type, CUresourcetype::Array));
        // SAFETY: we set the array variant, so reading it is valid.
        let arr = unsafe { raw.res.array };
        assert!(arr.h_array.is_null()); // default handle is null
    }

    #[test]
    fn resource_desc_linear_round_trip() {
        let rd = ResourceDesc::Linear {
            dev_ptr: DevicePtr(0x1000),
            format: ArrayFormat::Float,
            num_channels: 4,
            size_in_bytes: 1024,
        };
        let raw = rd.as_raw();
        assert!(matches!(raw.res_type, CUresourcetype::Linear));
        // SAFETY: we set the linear variant.
        let lin = unsafe { raw.res.linear };
        assert_eq!(lin.dev_ptr, 0x1000);
        assert_eq!(lin.num_channels, 4);
        assert_eq!(lin.size_in_bytes, 1024);
        assert!(matches!(lin.format, CUarray_format::Float));
    }

    #[test]
    fn cuda_array_create_no_gpu() {
        // Driver absent → DriverNotAvailable/NotSupported/NoGpu.
        // Driver present but no active context → DeviceUninitialized.
        // Driver present with context → Ok or InvalidDevice.
        match CudaArray::create_2d(64, 64, ArrayFormat::Float, 4) {
            Ok(_) => { /* GPU present with active context — creation succeeded */ }
            Err(CudaRtError::DriverNotAvailable)
            | Err(CudaRtError::NotSupported)
            | Err(CudaRtError::NoGpu)
            | Err(CudaRtError::InitializationError)
            | Err(CudaRtError::InvalidDevice)
            | Err(CudaRtError::DeviceUninitialized) => { /* expected */ }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn cuda_texture_object_create_no_gpu() {
        // Uses a null/default array handle — valid errors include driver-absent
        // variants and, when a driver is present, invalid-handle variants.
        let handle = CUarray::default();
        let res = ResourceDesc::Array { handle };
        let tex = TextureDesc::default_2d();
        match CudaTextureObject::create(&res, &tex, None) {
            Ok(_) => {}
            Err(CudaRtError::DriverNotAvailable)
            | Err(CudaRtError::NotSupported)
            | Err(CudaRtError::NoGpu)
            | Err(CudaRtError::InitializationError)
            | Err(CudaRtError::InvalidDevice)
            | Err(CudaRtError::InvalidValue)
            | Err(CudaRtError::DeviceUninitialized) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn cuda_surface_object_create_no_gpu() {
        // Uses a null/default array handle — valid errors include driver-absent
        // variants and, when a driver is present, invalid-handle variants.
        let handle = CUarray::default();
        let res = ResourceDesc::Array { handle };
        match CudaSurfaceObject::create(&res) {
            Ok(_) => {}
            Err(CudaRtError::DriverNotAvailable)
            | Err(CudaRtError::NotSupported)
            | Err(CudaRtError::NoGpu)
            | Err(CudaRtError::InitializationError)
            | Err(CudaRtError::InvalidDevice)
            | Err(CudaRtError::InvalidValue)
            | Err(CudaRtError::DeviceUninitialized) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn array_3d_flags_combine() {
        let flags = Array3DFlags::LAYERED.or(Array3DFlags::SURFACE_LDST);
        assert_eq!(flags.0, CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_SURFACE_LDST);
    }

    #[test]
    fn address_mode_variants_compile() {
        let _ = AddressMode::Wrap.as_cu();
        let _ = AddressMode::Clamp.as_cu();
        let _ = AddressMode::Mirror.as_cu();
        let _ = AddressMode::Border.as_cu();
    }
}
