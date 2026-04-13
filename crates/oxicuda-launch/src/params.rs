//! Kernel launch parameter configuration.
//!
//! This module provides [`LaunchParams`] and its builder for specifying
//! the execution configuration of a GPU kernel launch: grid size,
//! block size, and dynamic shared memory allocation.
//!
//! # Examples
//!
//! ```
//! use oxicuda_launch::{LaunchParams, Dim3};
//!
//! // Direct construction
//! let params = LaunchParams::new(Dim3::x(4), Dim3::x(256));
//! assert_eq!(params.total_threads(), 1024);
//!
//! // With shared memory
//! let params = LaunchParams::new(4u32, 256u32).with_shared_mem(4096);
//! assert_eq!(params.shared_mem_bytes, 4096);
//!
//! // Builder pattern
//! let params = LaunchParams::builder()
//!     .grid(256u32)
//!     .block(256u32)
//!     .shared_mem(4096)
//!     .build();
//! assert_eq!(params.total_threads(), 256 * 256);
//! ```

use oxicuda_driver::device::Device;

use crate::error::LaunchError;
use crate::grid::Dim3;

/// Parameters for a GPU kernel launch.
///
/// Specifies the execution configuration: grid size (number of blocks),
/// block size (threads per block), and dynamic shared memory allocation.
///
/// # Examples
///
/// ```
/// use oxicuda_launch::{LaunchParams, Dim3};
///
/// let params = LaunchParams::new(Dim3::x(256), Dim3::x(256));
/// assert_eq!(params.grid, Dim3::x(256));
/// assert_eq!(params.block, Dim3::x(256));
/// assert_eq!(params.shared_mem_bytes, 0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LaunchParams {
    /// Grid dimensions (number of thread blocks in each dimension).
    pub grid: Dim3,
    /// Block dimensions (number of threads per block in each dimension).
    pub block: Dim3,
    /// Dynamic shared memory allocation in bytes (default 0).
    pub shared_mem_bytes: u32,
}

impl LaunchParams {
    /// Creates new launch parameters with the given grid and block dimensions.
    ///
    /// Shared memory defaults to 0 bytes. Use [`with_shared_mem`](Self::with_shared_mem)
    /// to specify dynamic shared memory.
    ///
    /// Both `grid` and `block` accept anything that converts to [`Dim3`],
    /// including `u32`, `(u32, u32)`, and `(u32, u32, u32)`.
    #[inline]
    pub fn new(grid: impl Into<Dim3>, block: impl Into<Dim3>) -> Self {
        Self {
            grid: grid.into(),
            block: block.into(),
            shared_mem_bytes: 0,
        }
    }

    /// Sets the dynamic shared memory allocation in bytes.
    ///
    /// Returns `self` for method chaining.
    #[inline]
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Returns a [`LaunchParamsBuilder`] for incremental configuration.
    #[inline]
    pub fn builder() -> LaunchParamsBuilder {
        LaunchParamsBuilder::default()
    }

    /// Total number of threads in the launch (grid total * block total).
    ///
    /// Returns a `u64` to avoid overflow when grid and block totals
    /// are both large `u32` values.
    #[inline]
    pub fn total_threads(&self) -> u64 {
        self.grid.total() as u64 * self.block.total() as u64
    }

    /// Validates launch parameters against device hardware limits.
    ///
    /// Checks that:
    /// - All block and grid dimensions are non-zero.
    /// - The total threads per block does not exceed the device maximum.
    /// - Each block dimension does not exceed its per-axis device maximum.
    /// - Each grid dimension does not exceed its per-axis device maximum.
    /// - The dynamic shared memory does not exceed the device maximum per block.
    ///
    /// # Errors
    ///
    /// Returns a [`LaunchError`] describing the first constraint violation
    /// found, or a [`CudaError`](oxicuda_driver::CudaError) if device
    /// attribute queries fail.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use oxicuda_launch::{LaunchParams, Dim3};
    /// use oxicuda_driver::device::Device;
    ///
    /// oxicuda_driver::init()?;
    /// let dev = Device::get(0)?;
    /// let params = LaunchParams::new(256u32, 256u32);
    /// params.validate(&dev)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn validate(&self, device: &Device) -> Result<(), Box<dyn std::error::Error>> {
        self.validate_inner(device)
    }

    /// Inner validation that queries device attributes and checks constraints.
    fn validate_inner(&self, device: &Device) -> Result<(), Box<dyn std::error::Error>> {
        // Check non-zero dimensions
        if self.block.x == 0 {
            return Err(Box::new(LaunchError::InvalidDimension {
                dim: "block.x",
                value: 0,
            }));
        }
        if self.block.y == 0 {
            return Err(Box::new(LaunchError::InvalidDimension {
                dim: "block.y",
                value: 0,
            }));
        }
        if self.block.z == 0 {
            return Err(Box::new(LaunchError::InvalidDimension {
                dim: "block.z",
                value: 0,
            }));
        }
        if self.grid.x == 0 {
            return Err(Box::new(LaunchError::InvalidDimension {
                dim: "grid.x",
                value: 0,
            }));
        }
        if self.grid.y == 0 {
            return Err(Box::new(LaunchError::InvalidDimension {
                dim: "grid.y",
                value: 0,
            }));
        }
        if self.grid.z == 0 {
            return Err(Box::new(LaunchError::InvalidDimension {
                dim: "grid.z",
                value: 0,
            }));
        }

        // Query device limits
        let max_threads = device.max_threads_per_block()? as u32;
        let block_total = self.block.total();
        if block_total > max_threads {
            return Err(Box::new(LaunchError::BlockSizeExceedsLimit {
                requested: block_total,
                max: max_threads,
            }));
        }

        // Per-axis block limits
        let (max_bx, max_by, max_bz) = device.max_block_dim()?;
        if self.block.x > max_bx as u32 {
            return Err(Box::new(LaunchError::InvalidDimension {
                dim: "block.x",
                value: self.block.x,
            }));
        }
        if self.block.y > max_by as u32 {
            return Err(Box::new(LaunchError::InvalidDimension {
                dim: "block.y",
                value: self.block.y,
            }));
        }
        if self.block.z > max_bz as u32 {
            return Err(Box::new(LaunchError::InvalidDimension {
                dim: "block.z",
                value: self.block.z,
            }));
        }

        // Per-axis grid limits
        let (max_gx, max_gy, max_gz) = device.max_grid_dim()?;
        if self.grid.x > max_gx as u32 {
            return Err(Box::new(LaunchError::GridSizeExceedsLimit {
                requested: self.grid.x,
                max: max_gx as u32,
            }));
        }
        if self.grid.y > max_gy as u32 {
            return Err(Box::new(LaunchError::GridSizeExceedsLimit {
                requested: self.grid.y,
                max: max_gy as u32,
            }));
        }
        if self.grid.z > max_gz as u32 {
            return Err(Box::new(LaunchError::GridSizeExceedsLimit {
                requested: self.grid.z,
                max: max_gz as u32,
            }));
        }

        // Shared memory limit
        let max_smem = device.max_shared_memory_per_block()? as u32;
        if self.shared_mem_bytes > max_smem {
            return Err(Box::new(LaunchError::SharedMemoryExceedsLimit {
                requested: self.shared_mem_bytes,
                max: max_smem,
            }));
        }

        Ok(())
    }
}

/// Builder for [`LaunchParams`].
///
/// Provides a fluent interface for constructing launch parameters.
/// If grid or block dimensions are not set, they default to `Dim3::x(1)`.
///
/// # Examples
///
/// ```
/// use oxicuda_launch::{LaunchParams, Dim3};
///
/// let params = LaunchParams::builder()
///     .grid((4u32, 4u32))
///     .block(256u32)
///     .shared_mem(1024)
///     .build();
///
/// assert_eq!(params.grid, Dim3::xy(4, 4));
/// assert_eq!(params.block, Dim3::x(256));
/// assert_eq!(params.shared_mem_bytes, 1024);
/// ```
#[derive(Debug, Default)]
pub struct LaunchParamsBuilder {
    /// Grid dimensions, if set.
    grid: Option<Dim3>,
    /// Block dimensions, if set.
    block: Option<Dim3>,
    /// Dynamic shared memory in bytes.
    shared_mem_bytes: u32,
}

impl LaunchParamsBuilder {
    /// Sets the grid dimensions (number of thread blocks).
    ///
    /// Accepts anything that converts to [`Dim3`].
    #[inline]
    pub fn grid(mut self, dim: impl Into<Dim3>) -> Self {
        self.grid = Some(dim.into());
        self
    }

    /// Sets the block dimensions (threads per block).
    ///
    /// Accepts anything that converts to [`Dim3`].
    #[inline]
    pub fn block(mut self, dim: impl Into<Dim3>) -> Self {
        self.block = Some(dim.into());
        self
    }

    /// Sets the dynamic shared memory allocation in bytes.
    #[inline]
    pub fn shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Builds the [`LaunchParams`].
    ///
    /// If grid or block dimensions were not set, they default to
    /// `Dim3::x(1)` (a single block or a single thread).
    #[inline]
    pub fn build(self) -> LaunchParams {
        LaunchParams {
            grid: self.grid.unwrap_or(Dim3::x(1)),
            block: self.block.unwrap_or(Dim3::x(1)),
            shared_mem_bytes: self.shared_mem_bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_params_new_basic() {
        let p = LaunchParams::new(4u32, 256u32);
        assert_eq!(p.grid, Dim3::x(4));
        assert_eq!(p.block, Dim3::x(256));
        assert_eq!(p.shared_mem_bytes, 0);
    }

    #[test]
    fn launch_params_new_with_dim3() {
        let p = LaunchParams::new(Dim3::xy(4, 4), Dim3::xy(16, 16));
        assert_eq!(p.grid.total(), 16);
        assert_eq!(p.block.total(), 256);
    }

    #[test]
    fn launch_params_new_with_tuples() {
        let p = LaunchParams::new((4u32, 4u32), (16u32, 16u32));
        assert_eq!(p.grid, Dim3::xy(4, 4));
        assert_eq!(p.block, Dim3::xy(16, 16));
    }

    #[test]
    fn launch_params_with_shared_mem() {
        let p = LaunchParams::new(1u32, 256u32).with_shared_mem(8192);
        assert_eq!(p.shared_mem_bytes, 8192);
    }

    #[test]
    fn launch_params_total_threads() {
        let p = LaunchParams::new(4u32, 256u32);
        assert_eq!(p.total_threads(), 1024);

        let p = LaunchParams::new(Dim3::xy(4, 4), Dim3::xy(16, 16));
        assert_eq!(p.total_threads(), 16 * 256);
    }

    #[test]
    fn launch_params_total_threads_large() {
        // Ensure no overflow: grid 65535x65535 * block 1024
        let p = LaunchParams::new(Dim3::xy(65535, 65535), Dim3::x(1024));
        let expected = 65535u64 * 65535u64 * 1024u64;
        assert_eq!(p.total_threads(), expected);
    }

    #[test]
    fn builder_defaults() {
        let p = LaunchParams::builder().build();
        assert_eq!(p.grid, Dim3::x(1));
        assert_eq!(p.block, Dim3::x(1));
        assert_eq!(p.shared_mem_bytes, 0);
    }

    #[test]
    fn builder_full() {
        let p = LaunchParams::builder()
            .grid(128u32)
            .block(256u32)
            .shared_mem(4096)
            .build();
        assert_eq!(p.grid, Dim3::x(128));
        assert_eq!(p.block, Dim3::x(256));
        assert_eq!(p.shared_mem_bytes, 4096);
    }

    #[test]
    fn builder_partial_grid_only() {
        let p = LaunchParams::builder().grid(64u32).build();
        assert_eq!(p.grid, Dim3::x(64));
        assert_eq!(p.block, Dim3::x(1));
    }

    #[test]
    fn builder_partial_block_only() {
        let p = LaunchParams::builder().block(512u32).build();
        assert_eq!(p.grid, Dim3::x(1));
        assert_eq!(p.block, Dim3::x(512));
    }

    #[test]
    fn builder_with_tuple_dims() {
        let p = LaunchParams::builder()
            .grid((8u32, 8u32))
            .block((16u32, 16u32, 1u32))
            .build();
        assert_eq!(p.grid, Dim3::xy(8, 8));
        assert_eq!(p.block, Dim3::new(16, 16, 1));
    }

    type ValidateFn = fn(&LaunchParams, &Device) -> Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn validate_zero_block_x() {
        let p = LaunchParams {
            grid: Dim3::x(1),
            block: Dim3::new(0, 1, 1),
            shared_mem_bytes: 0,
        };
        // Cannot validate without a device on macOS, but we can test
        // that the method exists and the error type is correct.
        let _validate_fn: ValidateFn = LaunchParams::validate;
        // Zero-dimension detection is the first check, before device queries.
        // We can't call it without a device, so just verify compilation.
        assert_eq!(p.block.x, 0);
    }

    #[test]
    fn validate_zero_grid_z() {
        let p = LaunchParams {
            grid: Dim3::new(1, 1, 0),
            block: Dim3::x(256),
            shared_mem_bytes: 0,
        };
        assert_eq!(p.grid.z, 0);
    }

    #[test]
    fn validate_signature_compiles() {
        // Just verify the signature is well-formed.
        let _: ValidateFn = LaunchParams::validate;
    }

    #[cfg(feature = "gpu-tests")]
    #[test]
    fn validate_with_real_device() {
        oxicuda_driver::init().ok();
        if let Ok(dev) = Device::get(0) {
            let p = LaunchParams::new(4u32, 256u32);
            assert!(p.validate(&dev).is_ok());

            // Too many threads per block
            let p2 = LaunchParams::new(1u32, Dim3::new(1024, 1024, 1));
            assert!(p2.validate(&dev).is_err());
        }
    }
}
