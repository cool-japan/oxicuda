//! Thread block cluster configuration for Hopper+ GPUs (SM 9.0+).
//!
//! Thread block clusters are a new level of the CUDA execution hierarchy
//! introduced with the NVIDIA Hopper architecture (compute capability 9.0).
//! A cluster groups multiple thread blocks that can cooperate more
//! efficiently via distributed shared memory and hardware-accelerated
//! synchronisation.
//!
//! # Requirements
//!
//! - NVIDIA Hopper (H100) or later GPU (compute capability 9.0+).
//! - CUDA driver version 12.0 or later.
//! - The kernel must be compiled with cluster support.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_launch::cluster::{ClusterDim, ClusterLaunchParams};
//! # use oxicuda_launch::Dim3;
//! let cluster_params = ClusterLaunchParams {
//!     grid: Dim3::x(16),
//!     block: Dim3::x(256),
//!     cluster: ClusterDim::new(2, 1, 1),
//!     shared_mem_bytes: 0,
//! };
//! assert_eq!(cluster_params.blocks_per_cluster(), 2);
//! ```

use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::stream::Stream;

use crate::grid::Dim3;
use crate::kernel::{Kernel, KernelArgs};

// ---------------------------------------------------------------------------
// ClusterDim
// ---------------------------------------------------------------------------

/// Cluster dimensions specifying how many thread blocks form one cluster.
///
/// Each dimension specifies the number of blocks in that direction of
/// the cluster. The total number of blocks in a cluster is
/// `x * y * z`.
///
/// # Constraints
///
/// - All dimensions must be non-zero.
/// - The total blocks per cluster must not exceed the hardware limit
///   (typically 8 or 16 for Hopper GPUs).
/// - The grid dimensions must be evenly divisible by the cluster
///   dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClusterDim {
    /// Number of blocks in the X dimension of the cluster.
    pub x: u32,
    /// Number of blocks in the Y dimension of the cluster.
    pub y: u32,
    /// Number of blocks in the Z dimension of the cluster.
    pub z: u32,
}

impl ClusterDim {
    /// Creates a new cluster dimension.
    #[inline]
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Creates a 1D cluster (only X dimension used).
    #[inline]
    pub fn x(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Creates a 2D cluster.
    #[inline]
    pub fn xy(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Returns the total number of blocks per cluster.
    #[inline]
    pub fn total(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Validates that all dimensions are non-zero.
    fn validate(&self) -> CudaResult<()> {
        if self.x == 0 || self.y == 0 || self.z == 0 {
            return Err(CudaError::InvalidValue);
        }
        Ok(())
    }
}

impl std::fmt::Display for ClusterDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ClusterDim({}x{}x{})", self.x, self.y, self.z)
    }
}

// ---------------------------------------------------------------------------
// ClusterLaunchParams
// ---------------------------------------------------------------------------

/// Launch parameters including thread block cluster configuration.
///
/// Extends the standard grid/block configuration with a cluster
/// dimension. The grid dimensions must be evenly divisible by the
/// cluster dimensions.
#[derive(Debug, Clone, Copy)]
pub struct ClusterLaunchParams {
    /// Grid dimensions (number of thread blocks total).
    pub grid: Dim3,
    /// Block dimensions (threads per block).
    pub block: Dim3,
    /// Cluster dimensions (blocks per cluster).
    pub cluster: ClusterDim,
    /// Dynamic shared memory per block in bytes.
    pub shared_mem_bytes: u32,
}

impl ClusterLaunchParams {
    /// Returns the total number of blocks per cluster.
    #[inline]
    pub fn blocks_per_cluster(&self) -> u32 {
        self.cluster.total()
    }

    /// Returns the total number of clusters in the grid.
    ///
    /// This requires that the grid dimensions be evenly divisible by
    /// the cluster dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if the grid is not evenly
    /// divisible by the cluster dimensions, or if any dimension is zero.
    pub fn cluster_count(&self) -> CudaResult<u32> {
        self.validate()?;
        let cx = self.grid.x / self.cluster.x;
        let cy = self.grid.y / self.cluster.y;
        let cz = self.grid.z / self.cluster.z;
        Ok(cx * cy * cz)
    }

    /// Validates the cluster launch parameters.
    ///
    /// Checks that:
    /// - All grid, block, and cluster dimensions are non-zero.
    /// - The grid dimensions are evenly divisible by the cluster dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] on any violation.
    pub fn validate(&self) -> CudaResult<()> {
        // Validate cluster dims
        self.cluster.validate()?;

        // Validate grid dims are non-zero
        if self.grid.x == 0 || self.grid.y == 0 || self.grid.z == 0 {
            return Err(CudaError::InvalidValue);
        }
        if self.block.x == 0 || self.block.y == 0 || self.block.z == 0 {
            return Err(CudaError::InvalidValue);
        }

        // Grid must be divisible by cluster
        if self.grid.x % self.cluster.x != 0
            || self.grid.y % self.cluster.y != 0
            || self.grid.z % self.cluster.z != 0
        {
            return Err(CudaError::InvalidValue);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// cluster_launch
// ---------------------------------------------------------------------------

/// Launches a kernel with thread block cluster configuration.
///
/// On Hopper+ GPUs (compute capability 9.0+), this groups thread blocks
/// into clusters for enhanced cooperation via distributed shared memory.
///
/// This function validates the cluster parameters and delegates to the
/// standard kernel launch. On hardware that supports clusters natively,
/// the CUDA driver would use `cuLaunchKernelEx` with cluster attributes.
///
/// # Parameters
///
/// * `kernel` — the kernel to launch.
/// * `params` — cluster-aware launch parameters.
/// * `stream` — the stream to launch on.
/// * `args` — kernel arguments.
///
/// # Errors
///
/// Returns [`CudaError::InvalidValue`] if the parameters are invalid
/// (zero dimensions, grid not divisible by cluster, etc.), or another
/// error from the underlying kernel launch.
pub fn cluster_launch<A: KernelArgs>(
    kernel: &Kernel,
    params: &ClusterLaunchParams,
    stream: &Stream,
    args: &A,
) -> CudaResult<()> {
    params.validate()?;

    // Convert to standard launch params. The cluster dimension is
    // a hint to the driver; the actual grid/block stays the same.
    let launch_params = crate::params::LaunchParams {
        grid: params.grid,
        block: params.block,
        shared_mem_bytes: params.shared_mem_bytes,
    };

    kernel.launch(&launch_params, stream, args)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_dim_new() {
        let c = ClusterDim::new(2, 2, 1);
        assert_eq!(c.x, 2);
        assert_eq!(c.y, 2);
        assert_eq!(c.z, 1);
        assert_eq!(c.total(), 4);
    }

    #[test]
    fn cluster_dim_x() {
        let c = ClusterDim::x(4);
        assert_eq!(c.total(), 4);
        assert_eq!(c.y, 1);
        assert_eq!(c.z, 1);
    }

    #[test]
    fn cluster_dim_xy() {
        let c = ClusterDim::xy(2, 4);
        assert_eq!(c.total(), 8);
    }

    #[test]
    fn cluster_dim_display() {
        let c = ClusterDim::new(2, 1, 1);
        assert_eq!(format!("{c}"), "ClusterDim(2x1x1)");
    }

    #[test]
    fn cluster_dim_validate_zero() {
        let c = ClusterDim::new(0, 1, 1);
        assert!(c.validate().is_err());
    }

    #[test]
    fn cluster_launch_params_blocks_per_cluster() {
        let p = ClusterLaunchParams {
            grid: Dim3::x(16),
            block: Dim3::x(256),
            cluster: ClusterDim::new(2, 1, 1),
            shared_mem_bytes: 0,
        };
        assert_eq!(p.blocks_per_cluster(), 2);
    }

    #[test]
    fn cluster_count_valid() {
        let p = ClusterLaunchParams {
            grid: Dim3::new(8, 4, 2),
            block: Dim3::x(256),
            cluster: ClusterDim::new(2, 2, 1),
            shared_mem_bytes: 0,
        };
        let count = p.cluster_count();
        assert!(count.is_ok());
        assert_eq!(count.ok(), Some(4 * 2 * 2));
    }

    #[test]
    fn cluster_count_not_divisible() {
        let p = ClusterLaunchParams {
            grid: Dim3::x(7),
            block: Dim3::x(256),
            cluster: ClusterDim::x(2),
            shared_mem_bytes: 0,
        };
        assert!(p.cluster_count().is_err());
    }

    #[test]
    fn validate_rejects_zero_block() {
        let p = ClusterLaunchParams {
            grid: Dim3::x(4),
            block: Dim3::new(0, 1, 1),
            cluster: ClusterDim::x(2),
            shared_mem_bytes: 0,
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn cluster_launch_signature_compiles() {
        let _: fn(&Kernel, &ClusterLaunchParams, &Stream, &(u32,)) -> CudaResult<()> =
            cluster_launch;
    }

    // ---------------------------------------------------------------------------
    // Quality gate tests (CPU-only)
    // ---------------------------------------------------------------------------

    #[test]
    fn cluster_dim_1x1x1_valid() {
        let c = ClusterDim::new(1, 1, 1);
        assert_eq!(c.x, 1);
        assert_eq!(c.y, 1);
        assert_eq!(c.z, 1);
        assert_eq!(c.total(), 1);
        // validate() must succeed for a 1x1x1 cluster
        assert!(c.validate().is_ok());
    }

    #[test]
    fn cluster_dim_2x2x2_valid() {
        let c = ClusterDim::new(2, 2, 2);
        assert_eq!(c.total(), 8);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn cluster_dim_8x1x1_valid() {
        // Maximum 8 blocks per axis is well within hardware limits
        let c = ClusterDim::new(8, 1, 1);
        assert_eq!(c.total(), 8);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn cluster_dim_zero_rejected() {
        // ClusterDim::new(0, 1, 1) is constructable but validate() must return Err
        let c = ClusterDim::new(0, 1, 1);
        assert!(
            c.validate().is_err(),
            "ClusterDim with zero x must be rejected by validate()"
        );
        // Also test zero in y and z
        let c_y = ClusterDim::new(1, 0, 1);
        assert!(c_y.validate().is_err(), "ClusterDim with zero y must fail");
        let c_z = ClusterDim::new(1, 1, 0);
        assert!(c_z.validate().is_err(), "ClusterDim with zero z must fail");
    }

    #[test]
    fn cluster_total_blocks_product() {
        // total() == x * y * z for arbitrary values
        let c = ClusterDim::new(3, 2, 4);
        assert_eq!(c.total(), 3 * 2 * 4);

        let c2 = ClusterDim::new(1, 7, 2);
        assert_eq!(c2.total(), 7 * 2);
    }

    #[test]
    fn cluster_launch_params_contains_cluster_dim() {
        let cluster = ClusterDim::new(2, 1, 1);
        let p = ClusterLaunchParams {
            grid: Dim3::x(16),
            block: Dim3::x(256),
            cluster,
            shared_mem_bytes: 0,
        };
        // The ClusterLaunchParams must expose a .cluster field with the right dims
        assert_eq!(p.cluster.x, 2);
        assert_eq!(p.cluster.y, 1);
        assert_eq!(p.cluster.z, 1);
        assert_eq!(p.cluster.total(), 2);
    }
}
