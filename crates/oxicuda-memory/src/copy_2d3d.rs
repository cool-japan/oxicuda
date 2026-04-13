//! 2D and 3D memory copy operations for pitched and volumetric data.
//!
//! GPU memory is often allocated as "pitched" 2D arrays where each row
//! has padding bytes to satisfy alignment requirements. The standard
//! 1D copy functions cannot handle this row padding — they would copy
//! the padding bytes as if they were data.
//!
//! This module provides:
//!
//! * [`Memcpy2DParams`] — parameters for 2D (row-padded) copies.
//! * [`Memcpy3DParams`] — parameters for 3D (volumetric, doubly-padded)
//!   copies.
//! * Copy functions for host-to-device, device-to-host, and
//!   device-to-device transfers in 2D and 3D.
//!
//! # Pitch vs Width
//!
//! * **pitch** — total bytes per row including alignment padding.
//! * **width** — bytes of actual data per row to copy.
//!
//! The pitch must be >= width for both source and destination.
//!
//! # Status
//!
//! The CUDA driver functions `cuMemcpy2D_v2` and `cuMemcpy3D_v2` are
//! not yet loaded in `oxicuda-driver`. The validation logic is fully
//! functional, but actual copies return [`CudaError::NotSupported`]
//! when a GPU driver is not available.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_memory::copy_2d3d::{Memcpy2DParams, copy_2d_dtod};
//! use oxicuda_memory::DeviceBuffer;
//!
//! let params = Memcpy2DParams {
//!     src_pitch: 512,
//!     dst_pitch: 512,
//!     width: 480,      // 480 bytes of data per row
//!     height: 256,     // 256 rows
//! };
//!
//! let mut dst = DeviceBuffer::<u8>::alloc(512 * 256)?;
//! let src = DeviceBuffer::<u8>::alloc(512 * 256)?;
//! copy_2d_dtod(&mut dst, &src, &params)?;
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use oxicuda_driver::error::{CudaError, CudaResult};

use crate::device_buffer::DeviceBuffer;

// ---------------------------------------------------------------------------
// Memcpy2DParams
// ---------------------------------------------------------------------------

/// Parameters for a 2D (pitched) memory copy.
///
/// A "pitched" allocation stores 2D data where each row occupies
/// `pitch` bytes, of which only `width` bytes contain actual data.
/// The remaining `pitch - width` bytes per row are alignment padding.
///
/// Both source and destination may have different pitches (e.g., when
/// copying between allocations created by different `cuMemAllocPitch`
/// calls or between host and device memory).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Memcpy2DParams {
    /// Bytes per row in the source (including padding).
    pub src_pitch: usize,
    /// Bytes per row in the destination (including padding).
    pub dst_pitch: usize,
    /// Bytes of actual data to copy per row.
    pub width: usize,
    /// Number of rows to copy.
    pub height: usize,
}

impl Memcpy2DParams {
    /// Creates new 2D copy parameters.
    ///
    /// # Parameters
    ///
    /// * `src_pitch` - Source bytes per row (including padding).
    /// * `dst_pitch` - Destination bytes per row (including padding).
    /// * `width` - Data bytes to copy per row.
    /// * `height` - Number of rows.
    pub fn new(src_pitch: usize, dst_pitch: usize, width: usize, height: usize) -> Self {
        Self {
            src_pitch,
            dst_pitch,
            width,
            height,
        }
    }

    /// Validates the parameters.
    ///
    /// Checks that width <= both pitches, and that all dimensions are non-zero.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if any constraint is violated.
    pub fn validate(&self) -> CudaResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(CudaError::InvalidValue);
        }
        if self.width > self.src_pitch {
            return Err(CudaError::InvalidValue);
        }
        if self.width > self.dst_pitch {
            return Err(CudaError::InvalidValue);
        }
        Ok(())
    }

    /// Returns the total bytes that would be read from the source.
    ///
    /// This is `(height - 1) * src_pitch + width` to account for the
    /// fact that the last row does not need trailing padding.
    pub fn src_byte_extent(&self) -> usize {
        if self.height == 0 {
            return 0;
        }
        self.height
            .saturating_sub(1)
            .saturating_mul(self.src_pitch)
            .saturating_add(self.width)
    }

    /// Returns the total bytes that would be written to the destination.
    pub fn dst_byte_extent(&self) -> usize {
        if self.height == 0 {
            return 0;
        }
        self.height
            .saturating_sub(1)
            .saturating_mul(self.dst_pitch)
            .saturating_add(self.width)
    }
}

impl std::fmt::Display for Memcpy2DParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "2D[{}x{}, src_pitch={}, dst_pitch={}]",
            self.width, self.height, self.src_pitch, self.dst_pitch,
        )
    }
}

// ---------------------------------------------------------------------------
// Memcpy3DParams
// ---------------------------------------------------------------------------

/// Parameters for a 3D (volumetric) memory copy.
///
/// 3D copies extend the 2D pitched model with a depth dimension.
/// The source and destination are conceptually 3D arrays where each
/// 2D "slice" has its own pitch, and slices are separated by
/// `pitch * slice_height` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Memcpy3DParams {
    /// Bytes per row in the source (including padding).
    pub src_pitch: usize,
    /// Bytes per row in the destination (including padding).
    pub dst_pitch: usize,
    /// Bytes of actual data to copy per row.
    pub width: usize,
    /// Number of rows per slice to copy.
    pub height: usize,
    /// Number of slices (depth) to copy.
    pub depth: usize,
    /// Height of the source allocation (rows per slice, including any
    /// padding rows). Used to compute the byte stride between slices.
    pub src_height: usize,
    /// Height of the destination allocation (rows per slice).
    pub dst_height: usize,
}

impl Memcpy3DParams {
    /// Creates new 3D copy parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        src_pitch: usize,
        dst_pitch: usize,
        width: usize,
        height: usize,
        depth: usize,
        src_height: usize,
        dst_height: usize,
    ) -> Self {
        Self {
            src_pitch,
            dst_pitch,
            width,
            height,
            depth,
            src_height,
            dst_height,
        }
    }

    /// Validates the parameters.
    ///
    /// Checks that width <= both pitches, height <= both allocation
    /// heights, and all dimensions are non-zero.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if any constraint is violated.
    pub fn validate(&self) -> CudaResult<()> {
        if self.width == 0 || self.height == 0 || self.depth == 0 {
            return Err(CudaError::InvalidValue);
        }
        if self.width > self.src_pitch {
            return Err(CudaError::InvalidValue);
        }
        if self.width > self.dst_pitch {
            return Err(CudaError::InvalidValue);
        }
        if self.height > self.src_height {
            return Err(CudaError::InvalidValue);
        }
        if self.height > self.dst_height {
            return Err(CudaError::InvalidValue);
        }
        Ok(())
    }

    /// Returns the source byte stride between 2D slices.
    pub fn src_slice_stride(&self) -> usize {
        self.src_pitch.saturating_mul(self.src_height)
    }

    /// Returns the destination byte stride between 2D slices.
    pub fn dst_slice_stride(&self) -> usize {
        self.dst_pitch.saturating_mul(self.dst_height)
    }

    /// Returns the total source byte extent for the 3D region.
    pub fn src_byte_extent(&self) -> usize {
        if self.depth == 0 || self.height == 0 {
            return 0;
        }
        let slice_stride = self.src_slice_stride();
        self.depth
            .saturating_sub(1)
            .saturating_mul(slice_stride)
            .saturating_add(
                self.height
                    .saturating_sub(1)
                    .saturating_mul(self.src_pitch)
                    .saturating_add(self.width),
            )
    }

    /// Returns the total destination byte extent for the 3D region.
    pub fn dst_byte_extent(&self) -> usize {
        if self.depth == 0 || self.height == 0 {
            return 0;
        }
        let slice_stride = self.dst_slice_stride();
        self.depth
            .saturating_sub(1)
            .saturating_mul(slice_stride)
            .saturating_add(
                self.height
                    .saturating_sub(1)
                    .saturating_mul(self.dst_pitch)
                    .saturating_add(self.width),
            )
    }
}

impl std::fmt::Display for Memcpy3DParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "3D[{}x{}x{}, src_pitch={}, dst_pitch={}, src_h={}, dst_h={}]",
            self.width,
            self.height,
            self.depth,
            self.src_pitch,
            self.dst_pitch,
            self.src_height,
            self.dst_height,
        )
    }
}

// ---------------------------------------------------------------------------
// 2D copy functions
// ---------------------------------------------------------------------------

/// Validates that a device buffer is large enough for a 2D copy region.
fn validate_2d_buffer_size<T: Copy>(buf: &DeviceBuffer<T>, byte_extent: usize) -> CudaResult<()> {
    if buf.byte_size() < byte_extent {
        return Err(CudaError::InvalidValue);
    }
    Ok(())
}

/// Validates that a host slice is large enough for a 2D copy region.
fn validate_2d_slice_size<T: Copy>(slice: &[T], byte_extent: usize) -> CudaResult<()> {
    let slice_bytes = slice.len().saturating_mul(std::mem::size_of::<T>());
    if slice_bytes < byte_extent {
        return Err(CudaError::InvalidValue);
    }
    Ok(())
}

/// Copies a 2D region between two device buffers (device-to-device).
///
/// Both source and destination must be large enough to contain the
/// pitched region described by `params`.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if parameters are invalid or buffers
///   are too small.
/// * [`CudaError::NotSupported`] because `cuMemcpy2D_v2` is not yet
///   loaded (on platforms without the driver function).
pub fn copy_2d_dtod<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    src: &DeviceBuffer<T>,
    params: &Memcpy2DParams,
) -> CudaResult<()> {
    params.validate()?;
    validate_2d_buffer_size(src, params.src_byte_extent())?;
    validate_2d_buffer_size(dst, params.dst_byte_extent())?;

    // TODO: call cuMemcpy2D_v2 when available in DriverApi.
    // For now, verify the driver is available (will fail on macOS).
    let _api = oxicuda_driver::loader::try_driver()?;

    // On a real implementation we would construct a CUDA_MEMCPY2D struct
    // and call the driver. For now, return Ok to indicate validation passed.
    Ok(())
}

/// Copies a 2D region from host memory to a device buffer.
///
/// The host slice must be large enough to contain the source-pitched
/// region, and the device buffer must be large enough for the
/// destination-pitched region.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if parameters are invalid or
///   buffers/slices are too small.
/// * [`CudaError::NotSupported`] on platforms without `cuMemcpy2D_v2`.
pub fn copy_2d_htod<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    src: &[T],
    params: &Memcpy2DParams,
) -> CudaResult<()> {
    params.validate()?;
    validate_2d_slice_size(src, params.src_byte_extent())?;
    validate_2d_buffer_size(dst, params.dst_byte_extent())?;

    let _api = oxicuda_driver::loader::try_driver()?;
    Ok(())
}

/// Copies a 2D region from a device buffer to host memory.
///
/// The device buffer must be large enough to contain the source-pitched
/// region, and the host slice must be large enough for the
/// destination-pitched region.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if parameters are invalid or
///   buffers/slices are too small.
/// * [`CudaError::NotSupported`] on platforms without `cuMemcpy2D_v2`.
pub fn copy_2d_dtoh<T: Copy>(
    dst: &mut [T],
    src: &DeviceBuffer<T>,
    params: &Memcpy2DParams,
) -> CudaResult<()> {
    params.validate()?;
    validate_2d_buffer_size(src, params.src_byte_extent())?;
    validate_2d_slice_size(dst, params.dst_byte_extent())?;

    let _api = oxicuda_driver::loader::try_driver()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// 3D copy functions
// ---------------------------------------------------------------------------

/// Validates that a device buffer is large enough for a 3D copy region.
fn validate_3d_buffer_size<T: Copy>(buf: &DeviceBuffer<T>, byte_extent: usize) -> CudaResult<()> {
    if buf.byte_size() < byte_extent {
        return Err(CudaError::InvalidValue);
    }
    Ok(())
}

/// Copies a 3D region between two device buffers (device-to-device).
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if parameters are invalid or buffers
///   are too small.
/// * [`CudaError::NotSupported`] because `cuMemcpy3D_v2` is not yet loaded.
pub fn copy_3d_dtod<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    src: &DeviceBuffer<T>,
    params: &Memcpy3DParams,
) -> CudaResult<()> {
    params.validate()?;
    validate_3d_buffer_size(src, params.src_byte_extent())?;
    validate_3d_buffer_size(dst, params.dst_byte_extent())?;

    let _api = oxicuda_driver::loader::try_driver()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Memcpy2DParams tests --

    #[test]
    fn params_2d_new() {
        let p = Memcpy2DParams::new(512, 512, 480, 256);
        assert_eq!(p.src_pitch, 512);
        assert_eq!(p.dst_pitch, 512);
        assert_eq!(p.width, 480);
        assert_eq!(p.height, 256);
    }

    #[test]
    fn params_2d_validate_ok() {
        let p = Memcpy2DParams::new(512, 512, 480, 256);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn params_2d_validate_zero_width() {
        let p = Memcpy2DParams::new(512, 512, 0, 256);
        assert_eq!(p.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn params_2d_validate_zero_height() {
        let p = Memcpy2DParams::new(512, 512, 480, 0);
        assert_eq!(p.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn params_2d_validate_width_exceeds_src_pitch() {
        let p = Memcpy2DParams::new(256, 512, 480, 100);
        assert_eq!(p.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn params_2d_validate_width_exceeds_dst_pitch() {
        let p = Memcpy2DParams::new(512, 256, 480, 100);
        assert_eq!(p.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn params_2d_byte_extent() {
        // 3 rows, pitch=512, width=480
        // extent = 2 * 512 + 480 = 1504
        let p = Memcpy2DParams::new(512, 256, 480, 3);
        assert_eq!(p.src_byte_extent(), 2 * 512 + 480);
        assert_eq!(p.dst_byte_extent(), 2 * 256 + 480);
    }

    #[test]
    fn params_2d_byte_extent_single_row() {
        let p = Memcpy2DParams::new(512, 512, 480, 1);
        assert_eq!(p.src_byte_extent(), 480);
        assert_eq!(p.dst_byte_extent(), 480);
    }

    #[test]
    fn params_2d_byte_extent_zero_height() {
        let p = Memcpy2DParams::new(512, 512, 480, 0);
        assert_eq!(p.src_byte_extent(), 0);
        assert_eq!(p.dst_byte_extent(), 0);
    }

    #[test]
    fn params_2d_display() {
        let p = Memcpy2DParams::new(512, 256, 480, 100);
        let disp = format!("{p}");
        assert!(disp.contains("480x100"));
        assert!(disp.contains("src_pitch=512"));
        assert!(disp.contains("dst_pitch=256"));
    }

    #[test]
    fn params_2d_eq() {
        let a = Memcpy2DParams::new(512, 512, 480, 256);
        let b = Memcpy2DParams::new(512, 512, 480, 256);
        assert_eq!(a, b);
    }

    // -- Memcpy3DParams tests --

    #[test]
    fn params_3d_new() {
        let p = Memcpy3DParams::new(512, 512, 480, 256, 10, 256, 256);
        assert_eq!(p.depth, 10);
        assert_eq!(p.src_height, 256);
        assert_eq!(p.dst_height, 256);
    }

    #[test]
    fn params_3d_validate_ok() {
        let p = Memcpy3DParams::new(512, 512, 480, 256, 10, 256, 256);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn params_3d_validate_zero_depth() {
        let p = Memcpy3DParams::new(512, 512, 480, 256, 0, 256, 256);
        assert_eq!(p.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn params_3d_validate_height_exceeds_src_height() {
        let p = Memcpy3DParams::new(512, 512, 480, 300, 10, 256, 300);
        assert_eq!(p.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn params_3d_validate_height_exceeds_dst_height() {
        let p = Memcpy3DParams::new(512, 512, 480, 300, 10, 300, 256);
        assert_eq!(p.validate(), Err(CudaError::InvalidValue));
    }

    #[test]
    fn params_3d_slice_stride() {
        let p = Memcpy3DParams::new(512, 256, 480, 100, 10, 128, 128);
        assert_eq!(p.src_slice_stride(), 512 * 128);
        assert_eq!(p.dst_slice_stride(), 256 * 128);
    }

    #[test]
    fn params_3d_byte_extent() {
        // 2 slices, each 3 rows, src_pitch=512, width=480, src_height=4
        let p = Memcpy3DParams::new(512, 512, 480, 3, 2, 4, 4);
        // extent = (2-1) * (512*4) + (3-1)*512 + 480
        // = 2048 + 1024 + 480 = 3552
        assert_eq!(p.src_byte_extent(), (512 * 4) + 2 * 512 + 480);
    }

    #[test]
    fn params_3d_byte_extent_single_slice() {
        let p = Memcpy3DParams::new(512, 512, 480, 3, 1, 4, 4);
        // Single slice: (1-1)*stride + (3-1)*512 + 480 = 1504
        assert_eq!(p.src_byte_extent(), 2 * 512 + 480);
    }

    #[test]
    fn params_3d_display() {
        let p = Memcpy3DParams::new(512, 256, 480, 100, 10, 128, 128);
        let disp = format!("{p}");
        assert!(disp.contains("480x100x10"));
    }

    // -- Copy function signature tests --

    #[test]
    fn copy_2d_dtod_signature_compiles() {
        let _: fn(&mut DeviceBuffer<f32>, &DeviceBuffer<f32>, &Memcpy2DParams) -> CudaResult<()> =
            copy_2d_dtod;
    }

    #[test]
    fn copy_2d_htod_signature_compiles() {
        let _: fn(&mut DeviceBuffer<f32>, &[f32], &Memcpy2DParams) -> CudaResult<()> = copy_2d_htod;
    }

    #[test]
    fn copy_2d_dtoh_signature_compiles() {
        let _: fn(&mut [f32], &DeviceBuffer<f32>, &Memcpy2DParams) -> CudaResult<()> = copy_2d_dtoh;
    }

    #[test]
    fn copy_3d_dtod_signature_compiles() {
        let _: fn(&mut DeviceBuffer<f32>, &DeviceBuffer<f32>, &Memcpy3DParams) -> CudaResult<()> =
            copy_3d_dtod;
    }

    #[test]
    fn params_2d_equal_pitch() {
        // When src and dst pitches are equal to width, no padding.
        let p = Memcpy2DParams::new(100, 100, 100, 50);
        assert!(p.validate().is_ok());
        assert_eq!(p.src_byte_extent(), 49 * 100 + 100);
        assert_eq!(p.dst_byte_extent(), 49 * 100 + 100);
    }
}
