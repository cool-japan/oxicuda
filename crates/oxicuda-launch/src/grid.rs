//! Grid and block dimension types for kernel launch configuration.
//!
//! CUDA kernels are launched with a grid of thread blocks.
//! Each block contains threads organized in up to 3 dimensions.
//!
//! # Dimension model
//!
//! The CUDA execution model uses a two-level hierarchy:
//!
//! - **Grid**: A collection of thread blocks, specified as up to 3D dimensions.
//! - **Block**: A collection of threads within a block, also up to 3D.
//!
//! Both are described by [`Dim3`], which defaults unused dimensions to 1.
//!
//! # Helper function
//!
//! The [`grid_size_for`] function computes the minimum grid size needed
//! to cover a given number of elements with a given block size (ceiling
//! division).

use std::fmt;

use oxicuda_driver::error::CudaResult;
use oxicuda_driver::module::Function;

/// 3-dimensional size specification for grids and blocks.
///
/// Used to specify the number of thread blocks in a grid
/// and the number of threads in a block. Dimensions default
/// to 1 when not explicitly provided.
///
/// # Examples
///
/// ```
/// use oxicuda_launch::Dim3;
///
/// // 1D: 256 threads
/// let block = Dim3::x(256);
/// assert_eq!(block.x, 256);
/// assert_eq!(block.y, 1);
/// assert_eq!(block.z, 1);
///
/// // 2D: 16x16 threads
/// let block = Dim3::xy(16, 16);
/// assert_eq!(block.total(), 256);
///
/// // 3D
/// let block = Dim3::new(8, 8, 4);
/// assert_eq!(block.total(), 256);
///
/// // From conversions
/// let block: Dim3 = 256u32.into();
/// assert_eq!(block, Dim3::x(256));
///
/// let block: Dim3 = (16u32, 16u32).into();
/// assert_eq!(block, Dim3::xy(16, 16));
///
/// let block: Dim3 = (8u32, 8u32, 4u32).into();
/// assert_eq!(block, Dim3::new(8, 8, 4));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dim3 {
    /// X dimension.
    pub x: u32,
    /// Y dimension (default 1).
    pub y: u32,
    /// Z dimension (default 1).
    pub z: u32,
}

impl Dim3 {
    /// Creates a new `Dim3` with explicit values for all three dimensions.
    #[inline]
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Creates a 1-dimensional `Dim3` with the given X value.
    ///
    /// Y and Z are set to 1.
    #[inline]
    pub fn x(x: u32) -> Self {
        Self::new(x, 1, 1)
    }

    /// Creates a 2-dimensional `Dim3` with the given X and Y values.
    ///
    /// Z is set to 1.
    #[inline]
    pub fn xy(x: u32, y: u32) -> Self {
        Self::new(x, y, 1)
    }

    /// Total number of elements (`x * y * z`).
    ///
    /// For a grid dimension, this is the total number of thread blocks.
    /// For a block dimension, this is the total number of threads per block.
    #[inline]
    pub fn total(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl From<u32> for Dim3 {
    /// Converts a single `u32` into a 1D `Dim3`.
    #[inline]
    fn from(x: u32) -> Self {
        Self::x(x)
    }
}

impl From<(u32, u32)> for Dim3 {
    /// Converts a `(u32, u32)` tuple into a 2D `Dim3`.
    #[inline]
    fn from((x, y): (u32, u32)) -> Self {
        Self::xy(x, y)
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    /// Converts a `(u32, u32, u32)` tuple into a 3D `Dim3`.
    #[inline]
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self::new(x, y, z)
    }
}

impl fmt::Display for Dim3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.z != 1 {
            write!(f, "({}, {}, {})", self.x, self.y, self.z)
        } else if self.y != 1 {
            write!(f, "({}, {})", self.x, self.y)
        } else {
            write!(f, "{}", self.x)
        }
    }
}

// ---------------------------------------------------------------------------
// Occupancy-based auto grid sizing
// ---------------------------------------------------------------------------

/// Computes optimal grid and block dimensions for a 1D problem of `n` elements.
///
/// Queries the CUDA occupancy API to determine the block size that
/// maximises multiprocessor occupancy for the given kernel function,
/// then calculates the grid size needed to cover `n` work items.
///
/// Returns `(grid_dim, block_dim)` suitable for use with [`LaunchParams`](crate::LaunchParams).
///
/// # Errors
///
/// Returns a [`CudaError`](oxicuda_driver::CudaError) if the occupancy
/// query fails (e.g., invalid function handle, driver not loaded).
///
/// # Examples
///
/// ```rust,no_run
/// use oxicuda_launch::grid::auto_grid_for;
/// # use oxicuda_driver::module::Module;
///
/// # let module: Module = todo!();
/// let func = module.get_function("my_kernel")?;
/// let (grid, block) = auto_grid_for(&func, 100_000)?;
/// # Ok::<(), oxicuda_driver::error::CudaError>(())
/// ```
pub fn auto_grid_for(func: &Function, n: usize) -> CudaResult<(Dim3, Dim3)> {
    let (_min_grid, optimal_block) = func.optimal_block_size(0)?;
    let block_size = optimal_block as u32;
    let grid_x = if n == 0 {
        0
    } else {
        (n as u32).div_ceil(block_size)
    };
    Ok((Dim3::x(grid_x), Dim3::x(block_size)))
}

/// Computes optimal grid and block dimensions for a 2D problem.
///
/// Given a kernel function and problem dimensions `(width, height)`,
/// this function determines a 2D block size and the corresponding
/// grid dimensions.  The block is sized as a square (or near-square)
/// tile whose total thread count respects the occupancy-optimal value.
///
/// Returns `(grid_dim, block_dim)` as 2D [`Dim3`] values.
///
/// # Errors
///
/// Returns a [`CudaError`](oxicuda_driver::CudaError) if the occupancy
/// query fails.
///
/// # Examples
///
/// ```rust,no_run
/// use oxicuda_launch::grid::auto_grid_2d;
/// # use oxicuda_driver::module::Module;
///
/// # let module: Module = todo!();
/// let func = module.get_function("my_kernel_2d")?;
/// let (grid, block) = auto_grid_2d(&func, 1920, 1080)?;
/// # Ok::<(), oxicuda_driver::error::CudaError>(())
/// ```
pub fn auto_grid_2d(func: &Function, width: usize, height: usize) -> CudaResult<(Dim3, Dim3)> {
    let (_min_grid, optimal_block) = func.optimal_block_size(0)?;
    let total = optimal_block as u32;

    // Find a near-square block tile. Start from sqrt and round down to
    // powers-of-two-friendly values.
    let sqrt_approx = (total as f64).sqrt() as u32;
    let block_x = nearest_power_of_two_le(sqrt_approx).max(1);
    let block_y = (total / block_x).max(1);

    let grid_x = if width == 0 {
        0
    } else {
        (width as u32).div_ceil(block_x)
    };
    let grid_y = if height == 0 {
        0
    } else {
        (height as u32).div_ceil(block_y)
    };

    Ok((Dim3::xy(grid_x, grid_y), Dim3::xy(block_x, block_y)))
}

/// Returns the largest power of two that is less than or equal to `n`.
///
/// Returns 1 if `n` is 0.
fn nearest_power_of_two_le(n: u32) -> u32 {
    if n == 0 {
        return 1;
    }
    // Highest bit position gives the largest power-of-2 <= n.
    1u32 << (31 - n.leading_zeros())
}

// ---------------------------------------------------------------------------
// Simple grid sizing helper
// ---------------------------------------------------------------------------

/// Calculate the grid size needed to cover `n` elements with `block_size` threads.
///
/// Returns `(n + block_size - 1) / block_size`, i.e., ceiling division.
/// This is the standard formula for determining how many thread blocks
/// are needed to process `n` work items when each block handles
/// `block_size` items.
///
/// # Panics
///
/// Panics if `block_size` is zero.
///
/// # Examples
///
/// ```
/// use oxicuda_launch::grid_size_for;
///
/// assert_eq!(grid_size_for(1000, 256), 4);  // 4 * 256 = 1024 >= 1000
/// assert_eq!(grid_size_for(256, 256), 1);
/// assert_eq!(grid_size_for(257, 256), 2);
/// assert_eq!(grid_size_for(0, 256), 0);
/// assert_eq!(grid_size_for(1, 256), 1);
/// ```
#[inline]
pub fn grid_size_for(n: u32, block_size: u32) -> u32 {
    n.div_ceil(block_size)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim3_new() {
        let d = Dim3::new(4, 5, 6);
        assert_eq!(d.x, 4);
        assert_eq!(d.y, 5);
        assert_eq!(d.z, 6);
    }

    #[test]
    fn dim3_x() {
        let d = Dim3::x(128);
        assert_eq!(d, Dim3::new(128, 1, 1));
    }

    #[test]
    fn dim3_xy() {
        let d = Dim3::xy(16, 16);
        assert_eq!(d, Dim3::new(16, 16, 1));
    }

    #[test]
    fn dim3_total() {
        assert_eq!(Dim3::x(256).total(), 256);
        assert_eq!(Dim3::xy(16, 16).total(), 256);
        assert_eq!(Dim3::new(8, 8, 4).total(), 256);
        assert_eq!(Dim3::new(1, 1, 1).total(), 1);
    }

    #[test]
    fn dim3_from_u32() {
        let d: Dim3 = 512u32.into();
        assert_eq!(d, Dim3::x(512));
    }

    #[test]
    fn dim3_from_tuple2() {
        let d: Dim3 = (32u32, 8u32).into();
        assert_eq!(d, Dim3::xy(32, 8));
    }

    #[test]
    fn dim3_from_tuple3() {
        let d: Dim3 = (4u32, 4u32, 4u32).into();
        assert_eq!(d, Dim3::new(4, 4, 4));
    }

    #[test]
    fn dim3_display_1d() {
        assert_eq!(format!("{}", Dim3::x(256)), "256");
    }

    #[test]
    fn dim3_display_2d() {
        assert_eq!(format!("{}", Dim3::xy(16, 16)), "(16, 16)");
    }

    #[test]
    fn dim3_display_3d() {
        assert_eq!(format!("{}", Dim3::new(8, 8, 4)), "(8, 8, 4)");
    }

    #[test]
    fn dim3_eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Dim3::x(256));
        assert!(set.contains(&Dim3::new(256, 1, 1)));
        assert!(!set.contains(&Dim3::x(128)));
    }

    #[test]
    fn grid_size_for_exact() {
        assert_eq!(grid_size_for(256, 256), 1);
        assert_eq!(grid_size_for(512, 256), 2);
    }

    #[test]
    fn grid_size_for_remainder() {
        assert_eq!(grid_size_for(257, 256), 2);
        assert_eq!(grid_size_for(1000, 256), 4);
        assert_eq!(grid_size_for(1, 256), 1);
    }

    #[test]
    fn grid_size_for_zero_elements() {
        assert_eq!(grid_size_for(0, 256), 0);
    }

    #[test]
    fn grid_size_for_one_block() {
        assert_eq!(grid_size_for(1, 1), 1);
        assert_eq!(grid_size_for(100, 100), 1);
    }

    #[test]
    fn nearest_power_of_two_le_values() {
        assert_eq!(super::nearest_power_of_two_le(0), 1);
        assert_eq!(super::nearest_power_of_two_le(1), 1);
        assert_eq!(super::nearest_power_of_two_le(2), 2);
        assert_eq!(super::nearest_power_of_two_le(3), 2);
        assert_eq!(super::nearest_power_of_two_le(4), 4);
        assert_eq!(super::nearest_power_of_two_le(5), 4);
        assert_eq!(super::nearest_power_of_two_le(16), 16);
        assert_eq!(super::nearest_power_of_two_le(17), 16);
        assert_eq!(super::nearest_power_of_two_le(255), 128);
        assert_eq!(super::nearest_power_of_two_le(256), 256);
    }

    #[test]
    fn auto_grid_for_signature_compiles() {
        let _f: fn(
            &oxicuda_driver::module::Function,
            usize,
        ) -> oxicuda_driver::error::CudaResult<(Dim3, Dim3)> = super::auto_grid_for;
    }

    #[test]
    fn auto_grid_2d_signature_compiles() {
        let _f: fn(
            &oxicuda_driver::module::Function,
            usize,
            usize,
        ) -> oxicuda_driver::error::CudaResult<(Dim3, Dim3)> = super::auto_grid_2d;
    }

    #[cfg(feature = "gpu-tests")]
    #[test]
    fn auto_grid_for_with_real_kernel() {
        use std::sync::Arc;
        oxicuda_driver::init().ok();
        if let Ok(dev) = oxicuda_driver::device::Device::get(0) {
            let _ctx = Arc::new(oxicuda_driver::context::Context::new(&dev).expect("ctx"));
            let ptx = ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry test_kernel(.param .u32 n) { ret; }";
            if let Ok(module) = oxicuda_driver::module::Module::from_ptx(ptx) {
                let func = module.get_function("test_kernel").expect("func");
                let (grid, block) = super::auto_grid_for(&func, 10000).expect("auto_grid");
                assert!(grid.x > 0);
                assert!(block.x > 0);
            }
        }
    }
}
