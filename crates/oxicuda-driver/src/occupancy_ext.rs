//! Extended occupancy helpers for CPU-side occupancy estimation.
//!
//! Unlike the GPU-side queries in [`crate::occupancy`] that call
//! `cuOccupancy*` driver functions, this module provides **pure computation**
//! for analysing occupancy trade-offs without requiring a live GPU.
//!
//! # Features
//!
//! - [`OccupancyCalculator`] — CPU-side occupancy estimation
//! - [`OccupancyGrid`] — sweep block sizes to find the optimum
//! - [`DynamicSmemOccupancy`] — occupancy with shared-memory callbacks
//! - [`ClusterOccupancy`] — Hopper+ thread block cluster support
//!
//! # Example
//!
//! ```rust
//! use oxicuda_driver::occupancy_ext::*;
//!
//! let info = DeviceOccupancyInfo {
//!     sm_count: 84,
//!     max_threads_per_sm: 1536,
//!     max_blocks_per_sm: 16,
//!     max_registers_per_sm: 65536,
//!     max_shared_memory_per_sm: 102400,
//!     warp_size: 32,
//! };
//! let calc = OccupancyCalculator::new(info);
//! let est = calc.estimate_occupancy(256, 32, 0);
//! assert!(est.occupancy_ratio > 0.0);
//! ```

use crate::device::Device;
#[cfg(not(target_os = "macos"))]
use crate::error::CudaError;
use crate::error::CudaResult;

// ---------------------------------------------------------------------------
// DeviceOccupancyInfo
// ---------------------------------------------------------------------------

/// Hardware parameters needed for CPU-side occupancy estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceOccupancyInfo {
    /// Number of streaming multiprocessors on the device.
    pub sm_count: u32,
    /// Maximum resident threads per SM.
    pub max_threads_per_sm: u32,
    /// Maximum concurrent blocks per SM.
    pub max_blocks_per_sm: u32,
    /// Total 32-bit registers available per SM.
    pub max_registers_per_sm: u32,
    /// Shared memory capacity per SM in bytes.
    pub max_shared_memory_per_sm: u32,
    /// Threads per warp (typically 32).
    pub warp_size: u32,
}

impl DeviceOccupancyInfo {
    /// Maximum number of warps that can be resident on one SM.
    fn max_warps_per_sm(&self) -> u32 {
        if self.warp_size == 0 {
            return 0;
        }
        self.max_threads_per_sm / self.warp_size
    }

    /// Return synthetic [`DeviceOccupancyInfo`] for a given SM compute
    /// capability, enabling CPU-side occupancy analysis without a live GPU.
    ///
    /// Covers all major NVIDIA GPU architectures from Turing through Blackwell.
    /// Unknown architectures fall back to Ampere SM 8.6 defaults.
    ///
    /// # SM capability table
    ///
    /// | Architecture      | sm_major | sm_minor | SMs | Threads/SM | Smem/SM |
    /// |-------------------|----------|----------|-----|------------|---------|
    /// | Turing            | 7        | 5        | 68  | 1024       | 65536   |
    /// | Ampere A100       | 8        | 0        | 108 | 2048       | 167936  |
    /// | Ampere GA10x      | 8        | 6        | 84  | 1536       | 102400  |
    /// | Ada Lovelace      | 8        | 9        | 76  | 1536       | 101376  |
    /// | Hopper H100       | 9        | 0        | 132 | 2048       | 232448  |
    /// | Blackwell B100    | 10       | 0        | 132 | 2048       | 262144  |
    /// | Blackwell B200    | 12       | 0        | 148 | 2048       | 262144  |
    #[must_use]
    pub fn for_compute_capability(sm_major: u32, sm_minor: u32) -> Self {
        match (sm_major, sm_minor) {
            // Turing (sm_75)
            (7, 5) => Self {
                sm_count: 68,
                max_threads_per_sm: 1024,
                max_blocks_per_sm: 16,
                max_registers_per_sm: 65536,
                max_shared_memory_per_sm: 65536,
                warp_size: 32,
            },
            // Ampere A100 (sm_80)
            (8, 0) => Self {
                sm_count: 108,
                max_threads_per_sm: 2048,
                max_blocks_per_sm: 32,
                max_registers_per_sm: 65536,
                max_shared_memory_per_sm: 167936,
                warp_size: 32,
            },
            // Ampere GA10x (sm_86, e.g. RTX 3090)
            (8, 6) => Self {
                sm_count: 84,
                max_threads_per_sm: 1536,
                max_blocks_per_sm: 16,
                max_registers_per_sm: 65536,
                max_shared_memory_per_sm: 102400,
                warp_size: 32,
            },
            // Ada Lovelace (sm_89, e.g. RTX 4090)
            (8, 9) => Self {
                sm_count: 76,
                max_threads_per_sm: 1536,
                max_blocks_per_sm: 24,
                max_registers_per_sm: 65536,
                max_shared_memory_per_sm: 101376,
                warp_size: 32,
            },
            // Hopper H100 (sm_90)
            (9, 0) => Self {
                sm_count: 132,
                max_threads_per_sm: 2048,
                max_blocks_per_sm: 32,
                max_registers_per_sm: 65536,
                max_shared_memory_per_sm: 232448,
                warp_size: 32,
            },
            // Blackwell B100 (sm_100) — 132 SMs, 256KB shared/SM
            (10, 0) => Self {
                sm_count: 132,
                max_threads_per_sm: 2048,
                max_blocks_per_sm: 32,
                max_registers_per_sm: 65536,
                max_shared_memory_per_sm: 262144,
                warp_size: 32,
            },
            // Blackwell B200 (sm_120) — 148 SMs, 256KB shared/SM
            (12, 0) => Self {
                sm_count: 148,
                max_threads_per_sm: 2048,
                max_blocks_per_sm: 32,
                max_registers_per_sm: 65536,
                max_shared_memory_per_sm: 262144,
                warp_size: 32,
            },
            // Unknown / future — fall back to Ampere GA10x defaults.
            _ => Self {
                sm_count: 84,
                max_threads_per_sm: 1536,
                max_blocks_per_sm: 16,
                max_registers_per_sm: 65536,
                max_shared_memory_per_sm: 102400,
                warp_size: 32,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// LimitingFactor
// ---------------------------------------------------------------------------

/// The resource that limits occupancy the most.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LimitingFactor {
    /// Block size limits the number of warps.
    Threads,
    /// Register pressure limits concurrent warps.
    Registers,
    /// Shared memory exhaustion limits concurrent blocks.
    SharedMemory,
    /// Hardware block-per-SM cap is the bottleneck.
    Blocks,
    /// All resources have headroom (or estimation was trivial).
    None,
}

// ---------------------------------------------------------------------------
// OccupancyEstimate
// ---------------------------------------------------------------------------

/// Result of a CPU-side occupancy estimation for one configuration.
#[derive(Debug, Clone, Copy)]
pub struct OccupancyEstimate {
    /// Active warps per SM for this configuration.
    pub active_warps_per_sm: u32,
    /// Maximum possible warps per SM (hardware limit).
    pub max_warps_per_sm: u32,
    /// Fraction of max warps that are active (0.0 .. 1.0).
    pub occupancy_ratio: f64,
    /// Which resource is the tightest bottleneck.
    pub limiting_factor: LimitingFactor,
}

// ---------------------------------------------------------------------------
// OccupancyCalculator
// ---------------------------------------------------------------------------

/// CPU-side occupancy estimator — no GPU calls required.
///
/// Given device hardware parameters, this struct computes how many warps
/// can be concurrently resident for a given kernel configuration.
#[derive(Debug, Clone)]
pub struct OccupancyCalculator {
    info: DeviceOccupancyInfo,
}

impl OccupancyCalculator {
    /// Create a new calculator from device occupancy information.
    pub fn new(device_info: DeviceOccupancyInfo) -> Self {
        Self { info: device_info }
    }

    /// Return a reference to the underlying device info.
    pub fn device_info(&self) -> &DeviceOccupancyInfo {
        &self.info
    }

    /// Estimate occupancy for the given kernel configuration.
    ///
    /// # Parameters
    ///
    /// * `block_size` — threads per block.
    /// * `registers_per_thread` — registers consumed by each thread.
    /// * `shared_memory` — shared memory per block in bytes.
    pub fn estimate_occupancy(
        &self,
        block_size: u32,
        registers_per_thread: u32,
        shared_memory: u32,
    ) -> OccupancyEstimate {
        let max_warps = self.info.max_warps_per_sm();

        // Degenerate cases
        if block_size == 0 || self.info.warp_size == 0 || max_warps == 0 {
            return OccupancyEstimate {
                active_warps_per_sm: 0,
                max_warps_per_sm: max_warps,
                occupancy_ratio: 0.0,
                limiting_factor: LimitingFactor::None,
            };
        }

        let warps_per_block = block_size.div_ceil(self.info.warp_size);

        // --- Limit 1: max blocks per SM (hardware cap) -----------------------
        let blocks_by_block_limit = self.info.max_blocks_per_sm;

        // --- Limit 2: threads (warps) ----------------------------------------
        let blocks_by_threads = max_warps.checked_div(warps_per_block).unwrap_or(0);

        // --- Limit 3: registers -----------------------------------------------
        let blocks_by_registers = if registers_per_thread == 0 || warps_per_block == 0 {
            u32::MAX // registers not a bottleneck
        } else {
            let regs_per_block = registers_per_thread * warps_per_block * self.info.warp_size;
            self.info
                .max_registers_per_sm
                .checked_div(regs_per_block)
                .unwrap_or(u32::MAX)
        };

        // --- Limit 4: shared memory -------------------------------------------
        let blocks_by_smem = if shared_memory == 0 {
            u32::MAX // smem not a bottleneck
        } else if self.info.max_shared_memory_per_sm == 0 {
            0
        } else {
            self.info.max_shared_memory_per_sm / shared_memory
        };

        // Take the minimum across all limits
        let active_blocks = blocks_by_block_limit
            .min(blocks_by_threads)
            .min(blocks_by_registers)
            .min(blocks_by_smem);

        let active_warps = active_blocks * warps_per_block;
        let clamped_warps = active_warps.min(max_warps);
        let ratio = if max_warps > 0 {
            clamped_warps as f64 / max_warps as f64
        } else {
            0.0
        };

        // Determine limiting factor
        let effective = active_blocks;
        let limiting_factor = if effective == 0 {
            if blocks_by_smem == 0 {
                LimitingFactor::SharedMemory
            } else if blocks_by_registers == 0 {
                LimitingFactor::Registers
            } else if blocks_by_threads == 0 {
                LimitingFactor::Threads
            } else {
                LimitingFactor::Blocks
            }
        } else if effective == blocks_by_smem
            && blocks_by_smem
                <= blocks_by_registers
                    .min(blocks_by_threads)
                    .min(blocks_by_block_limit)
        {
            LimitingFactor::SharedMemory
        } else if effective == blocks_by_registers
            && blocks_by_registers <= blocks_by_threads.min(blocks_by_block_limit)
        {
            LimitingFactor::Registers
        } else if effective == blocks_by_threads && blocks_by_threads <= blocks_by_block_limit {
            LimitingFactor::Threads
        } else if effective == blocks_by_block_limit {
            LimitingFactor::Blocks
        } else {
            LimitingFactor::None
        };

        OccupancyEstimate {
            active_warps_per_sm: clamped_warps,
            max_warps_per_sm: max_warps,
            occupancy_ratio: ratio,
            limiting_factor,
        }
    }
}

// ---------------------------------------------------------------------------
// OccupancyPoint / OccupancyGrid
// ---------------------------------------------------------------------------

/// A single data point from a block-size sweep.
#[derive(Debug, Clone, Copy)]
pub struct OccupancyPoint {
    /// Block size (threads per block) for this point.
    pub block_size: u32,
    /// Occupancy ratio (0.0 .. 1.0).
    pub occupancy: f64,
    /// Active warps per SM.
    pub active_warps: u32,
    /// Limiting resource at this block size.
    pub limiting_factor: LimitingFactor,
}

/// Sweep block sizes to find the configuration that maximises occupancy.
pub struct OccupancyGrid;

impl OccupancyGrid {
    /// Sweep block sizes from `warp_size` to `max_threads_per_sm` in
    /// increments of `warp_size` and return occupancy at each step.
    pub fn sweep(
        calculator: &OccupancyCalculator,
        registers_per_thread: u32,
        shared_memory: u32,
    ) -> Vec<OccupancyPoint> {
        let ws = calculator.info.warp_size;
        if ws == 0 {
            return Vec::new();
        }
        let max_threads = calculator.info.max_threads_per_sm;
        let mut points = Vec::new();
        let mut bs = ws;
        while bs <= max_threads {
            let est = calculator.estimate_occupancy(bs, registers_per_thread, shared_memory);
            points.push(OccupancyPoint {
                block_size: bs,
                occupancy: est.occupancy_ratio,
                active_warps: est.active_warps_per_sm,
                limiting_factor: est.limiting_factor,
            });
            bs += ws;
        }
        points
    }

    /// Pick the block size with the highest occupancy.
    ///
    /// Ties are broken by choosing the **smallest** block size.
    /// Returns `0` if the slice is empty.
    pub fn best_block_size(points: &[OccupancyPoint]) -> u32 {
        let mut best: Option<&OccupancyPoint> = Option::None;
        for pt in points {
            best = Some(match best {
                Option::None => pt,
                Some(prev) => {
                    if pt.occupancy > prev.occupancy
                        || (pt.occupancy == prev.occupancy && pt.block_size < prev.block_size)
                    {
                        pt
                    } else {
                        prev
                    }
                }
            });
        }
        best.map_or(0, |p| p.block_size)
    }
}

// ---------------------------------------------------------------------------
// DynamicSmemOccupancy
// ---------------------------------------------------------------------------

/// Occupancy estimation where shared memory varies with block size.
pub struct DynamicSmemOccupancy;

impl DynamicSmemOccupancy {
    /// Sweep block sizes using a callback `smem_fn(block_size) -> smem_bytes`.
    pub fn with_smem_function<F>(
        calculator: &OccupancyCalculator,
        smem_fn: F,
        registers_per_thread: u32,
    ) -> Vec<OccupancyPoint>
    where
        F: Fn(u32) -> u32,
    {
        let ws = calculator.info.warp_size;
        if ws == 0 {
            return Vec::new();
        }
        let max_threads = calculator.info.max_threads_per_sm;
        let mut points = Vec::new();
        let mut bs = ws;
        while bs <= max_threads {
            let smem = smem_fn(bs);
            let est = calculator.estimate_occupancy(bs, registers_per_thread, smem);
            points.push(OccupancyPoint {
                block_size: bs,
                occupancy: est.occupancy_ratio,
                active_warps: est.active_warps_per_sm,
                limiting_factor: est.limiting_factor,
            });
            bs += ws;
        }
        points
    }

    /// A shared-memory function that scales linearly with block size.
    ///
    /// Returns `block_size * bytes_per_thread`.
    pub fn linear_smem(bytes_per_thread: u32) -> impl Fn(u32) -> u32 {
        move |block_size: u32| block_size * bytes_per_thread
    }

    /// A shared-memory function for tile-based kernels.
    ///
    /// Returns `tile_size * tile_size * element_size` (constant per block).
    pub fn tile_smem(tile_size: u32, element_size: u32) -> impl Fn(u32) -> u32 {
        move |_block_size: u32| tile_size * tile_size * element_size
    }
}

// ---------------------------------------------------------------------------
// ClusterOccupancy (Hopper+)
// ---------------------------------------------------------------------------

/// Thread block cluster configuration for Hopper+ GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClusterConfig {
    /// Cluster extent in X dimension (number of blocks).
    pub cluster_x: u32,
    /// Cluster extent in Y dimension.
    pub cluster_y: u32,
    /// Cluster extent in Z dimension.
    pub cluster_z: u32,
}

impl ClusterConfig {
    /// Total blocks in one cluster.
    pub fn total_blocks(&self) -> u32 {
        self.cluster_x * self.cluster_y * self.cluster_z
    }
}

/// Result of a cluster occupancy estimation.
#[derive(Debug, Clone, Copy)]
pub struct ClusterOccupancyEstimate {
    /// Number of blocks per cluster.
    pub blocks_per_cluster: u32,
    /// Maximum clusters that fit per SM (fractional blocks accounted for).
    pub clusters_per_sm: u32,
    /// Effective occupancy ratio (0.0 .. 1.0).
    pub effective_occupancy: f64,
    /// Total shared memory consumed by one cluster (bytes).
    pub cluster_smem_total: u32,
}

/// Hopper+ thread block cluster occupancy estimation.
pub struct ClusterOccupancy;

impl ClusterOccupancy {
    /// Estimate occupancy when blocks are grouped into clusters.
    ///
    /// A cluster of `cluster_size` blocks must all reside on the same GPC
    /// (GPU Processing Cluster). This effectively reduces the number of
    /// independent blocks schedulable per SM.
    ///
    /// # Parameters
    ///
    /// * `calculator` — occupancy calculator with device info.
    /// * `block_size` — threads per block.
    /// * `cluster_size` — number of blocks per cluster.
    /// * `registers_per_thread` — registers per thread.
    /// * `shared_memory` — shared memory per block (bytes).
    pub fn estimate_cluster_occupancy(
        calculator: &OccupancyCalculator,
        block_size: u32,
        cluster_size: u32,
        registers_per_thread: u32,
        shared_memory: u32,
    ) -> ClusterOccupancyEstimate {
        if cluster_size == 0 {
            return ClusterOccupancyEstimate {
                blocks_per_cluster: 0,
                clusters_per_sm: 0,
                effective_occupancy: 0.0,
                cluster_smem_total: 0,
            };
        }

        // First, get the per-block occupancy estimate
        let est = calculator.estimate_occupancy(block_size, registers_per_thread, shared_memory);

        let max_warps = est.max_warps_per_sm;
        let warps_per_block = if calculator.info.warp_size == 0 {
            0
        } else {
            block_size.div_ceil(calculator.info.warp_size)
        };

        // How many blocks could fit per SM (from the standard estimate)?
        let blocks_per_sm = est
            .active_warps_per_sm
            .checked_div(warps_per_block)
            .unwrap_or(0);

        // Clusters must schedule in whole units
        let clusters_per_sm = blocks_per_sm / cluster_size;
        let active_blocks = clusters_per_sm * cluster_size;
        let active_warps = active_blocks * warps_per_block;

        let effective_occupancy = if max_warps > 0 {
            (active_warps.min(max_warps)) as f64 / max_warps as f64
        } else {
            0.0
        };

        ClusterOccupancyEstimate {
            blocks_per_cluster: cluster_size,
            clusters_per_sm,
            effective_occupancy,
            cluster_smem_total: cluster_size * shared_memory,
        }
    }
}

// ---------------------------------------------------------------------------
// Device convenience extension
// ---------------------------------------------------------------------------

impl Device {
    /// Gather all occupancy-relevant hardware attributes into a
    /// [`DeviceOccupancyInfo`] struct.
    ///
    /// On macOS (where no NVIDIA driver is available) this returns
    /// synthetic values for a typical SM 8.6 (Ampere) GPU so that
    /// CPU-side occupancy analysis can still run.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`](crate::error::CudaError) if an attribute query fails on a real GPU.
    pub fn occupancy_info(&self) -> CudaResult<DeviceOccupancyInfo> {
        // On macOS the driver is never present — return synthetic defaults.
        #[cfg(target_os = "macos")]
        {
            let _ = self; // suppress unused warning
            Ok(DeviceOccupancyInfo {
                sm_count: 84,
                max_threads_per_sm: 1536,
                max_blocks_per_sm: 16,
                max_registers_per_sm: 65536,
                max_shared_memory_per_sm: 102400,
                warp_size: 32,
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            let sm_count = self
                .multiprocessor_count()
                .map(|v| v as u32)
                .map_err(|_| CudaError::NotInitialized)?;
            let max_threads_per_sm = self
                .max_threads_per_multiprocessor()
                .map(|v| v as u32)
                .map_err(|_| CudaError::NotInitialized)?;
            let max_blocks_per_sm = self
                .max_blocks_per_multiprocessor()
                .map(|v| v as u32)
                .map_err(|_| CudaError::NotInitialized)?;
            let max_registers_per_sm = self
                .max_registers_per_multiprocessor()
                .map(|v| v as u32)
                .map_err(|_| CudaError::NotInitialized)?;
            let max_shared_memory_per_sm = self
                .max_shared_memory_per_multiprocessor()
                .map(|v| v as u32)
                .map_err(|_| CudaError::NotInitialized)?;
            let warp_size = self
                .warp_size()
                .map(|v| v as u32)
                .map_err(|_| CudaError::NotInitialized)?;

            Ok(DeviceOccupancyInfo {
                sm_count,
                max_threads_per_sm,
                max_blocks_per_sm,
                max_registers_per_sm,
                max_shared_memory_per_sm,
                warp_size,
            })
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A typical SM 8.6 (e.g. RTX 3090) device info for testing.
    fn ampere_info() -> DeviceOccupancyInfo {
        DeviceOccupancyInfo {
            sm_count: 82,
            max_threads_per_sm: 1536,
            max_blocks_per_sm: 16,
            max_registers_per_sm: 65536,
            max_shared_memory_per_sm: 102400,
            warp_size: 32,
        }
    }

    // --- Basic occupancy estimation ------------------------------------------

    #[test]
    fn test_basic_occupancy_estimation() {
        let calc = OccupancyCalculator::new(ampere_info());
        let est = calc.estimate_occupancy(256, 32, 0);
        // 256 threads = 8 warps/block, max 48 warps, 48/8 = 6 blocks,
        // but limited by max_blocks_per_sm = 16 (not limiting here)
        // registers: 32 * 8 * 32 = 8192 per block, 65536/8192 = 8 blocks
        // min(16, 6, 8) = 6 blocks => 6*8 = 48 warps => 100%
        assert_eq!(est.max_warps_per_sm, 48);
        assert!(est.occupancy_ratio > 0.0);
        assert!(est.active_warps_per_sm > 0);
    }

    #[test]
    fn test_full_occupancy() {
        let calc = OccupancyCalculator::new(ampere_info());
        // 32 threads, 0 registers pressure, 0 smem => many blocks fit
        let est = calc.estimate_occupancy(32, 16, 0);
        // 1 warp/block, max 48 warps => 48 blocks needed,
        // but max_blocks_per_sm = 16 => 16 warps => 16/48 = 33%
        assert_eq!(est.active_warps_per_sm, 16);
    }

    // --- Limiting factor detection -------------------------------------------

    #[test]
    fn test_limiting_factor_threads() {
        let calc = OccupancyCalculator::new(ampere_info());
        // Large block (1024 threads = 32 warps), low registers, no smem
        // 48 / 32 = 1 block => 32 warps => threads is the limit
        let est = calc.estimate_occupancy(1024, 16, 0);
        assert_eq!(est.limiting_factor, LimitingFactor::Threads);
    }

    #[test]
    fn test_limiting_factor_registers() {
        let calc = OccupancyCalculator::new(ampere_info());
        // 256 threads = 8 warps, 128 registers each
        // regs per block = 128 * 8 * 32 = 32768, 65536/32768 = 2 blocks
        // threads: 48/8 = 6 blocks, blocks: 16 => min(16, 6, 2) = 2
        let est = calc.estimate_occupancy(256, 128, 0);
        assert_eq!(est.limiting_factor, LimitingFactor::Registers);
    }

    #[test]
    fn test_limiting_factor_shared_memory() {
        let calc = OccupancyCalculator::new(ampere_info());
        // 128 threads = 4 warps, low regs, 51200 bytes smem
        // smem: 102400 / 51200 = 2 blocks
        // threads: 48 / 4 = 12, blocks: 16 => min(16, 12, oo, 2) = 2
        let est = calc.estimate_occupancy(128, 16, 51200);
        assert_eq!(est.limiting_factor, LimitingFactor::SharedMemory);
    }

    #[test]
    fn test_limiting_factor_blocks() {
        let info = DeviceOccupancyInfo {
            max_blocks_per_sm: 4,
            ..ampere_info()
        };
        let calc = OccupancyCalculator::new(info);
        // 64 threads = 2 warps, low regs, no smem
        // threads: 48/2 = 24, blocks: 4 => min(4, 24) = 4
        let est = calc.estimate_occupancy(64, 16, 0);
        assert_eq!(est.limiting_factor, LimitingFactor::Blocks);
    }

    #[test]
    fn test_limiting_factor_none_zero_block() {
        let calc = OccupancyCalculator::new(ampere_info());
        let est = calc.estimate_occupancy(0, 32, 0);
        assert_eq!(est.limiting_factor, LimitingFactor::None);
        assert_eq!(est.active_warps_per_sm, 0);
        assert_eq!(est.occupancy_ratio, 0.0);
    }

    // --- Block size sweep ----------------------------------------------------

    #[test]
    fn test_sweep_returns_points() {
        let calc = OccupancyCalculator::new(ampere_info());
        let points = OccupancyGrid::sweep(&calc, 32, 0);
        // warp_size=32, max_threads=1536 => 1536/32 = 48 points
        assert_eq!(points.len(), 48);
        assert_eq!(points[0].block_size, 32);
        assert_eq!(points[47].block_size, 1536);
    }

    #[test]
    fn test_best_block_size() {
        let calc = OccupancyCalculator::new(ampere_info());
        let points = OccupancyGrid::sweep(&calc, 32, 0);
        let best = OccupancyGrid::best_block_size(&points);
        // Should pick a block size that gives 100% occupancy
        assert!(best > 0);
        assert_eq!(best % 32, 0);
    }

    #[test]
    fn test_best_block_size_empty() {
        assert_eq!(OccupancyGrid::best_block_size(&[]), 0);
    }

    // --- Dynamic shared memory -----------------------------------------------

    #[test]
    fn test_dynamic_smem_linear() {
        let calc = OccupancyCalculator::new(ampere_info());
        let smem_fn = DynamicSmemOccupancy::linear_smem(8); // 8 bytes per thread
        let points = DynamicSmemOccupancy::with_smem_function(&calc, smem_fn, 32);
        assert!(!points.is_empty());
        // At block_size = 32 => 256 bytes smem; at 1024 => 8192 bytes
        // Verify smem increases with block size by checking occupancy trend
        // (larger blocks with more smem should eventually reduce occupancy)
        let first_occ = points[0].occupancy;
        let last_occ = points[points.len() - 1].occupancy;
        // Just verify we got valid data
        assert!((0.0..=1.0).contains(&first_occ));
        assert!((0.0..=1.0).contains(&last_occ));
    }

    #[test]
    fn test_dynamic_smem_tile() {
        let calc = OccupancyCalculator::new(ampere_info());
        let smem_fn = DynamicSmemOccupancy::tile_smem(16, 4); // 16x16 * 4B = 1024 bytes
        let points = DynamicSmemOccupancy::with_smem_function(&calc, smem_fn, 32);
        // Tile smem is constant (1024) regardless of block size
        assert!(!points.is_empty());
    }

    // --- Cluster occupancy ---------------------------------------------------

    #[test]
    fn test_cluster_occupancy_basic() {
        let calc = OccupancyCalculator::new(ampere_info());
        let result = ClusterOccupancy::estimate_cluster_occupancy(&calc, 128, 2, 32, 4096);
        assert_eq!(result.blocks_per_cluster, 2);
        assert!(result.effective_occupancy >= 0.0 && result.effective_occupancy <= 1.0);
        assert_eq!(result.cluster_smem_total, 2 * 4096);
    }

    #[test]
    fn test_cluster_occupancy_zero_cluster() {
        let calc = OccupancyCalculator::new(ampere_info());
        let result = ClusterOccupancy::estimate_cluster_occupancy(&calc, 128, 0, 32, 0);
        assert_eq!(result.clusters_per_sm, 0);
        assert_eq!(result.effective_occupancy, 0.0);
    }

    // --- DeviceOccupancyInfo from Device (macOS synthetic) --------------------

    #[test]
    fn test_cluster_config_total_blocks() {
        let cfg = ClusterConfig {
            cluster_x: 2,
            cluster_y: 3,
            cluster_z: 4,
        };
        assert_eq!(cfg.total_blocks(), 24);
    }

    // --- Edge cases ----------------------------------------------------------

    #[test]
    fn test_block_size_exceeds_max() {
        let calc = OccupancyCalculator::new(ampere_info());
        // Block size larger than max_threads_per_sm (not strictly invalid for
        // the estimator but should still produce a reasonable result).
        let est = calc.estimate_occupancy(2048, 32, 0);
        // 2048 / 32 = 64 warps per block, but only 48 max => 0 blocks fit
        assert_eq!(est.active_warps_per_sm, 0);
        assert_eq!(est.occupancy_ratio, 0.0);
    }

    // --- SM100 / SM120 (Blackwell) occupancy coverage ----------------------------

    fn sm100_info() -> DeviceOccupancyInfo {
        DeviceOccupancyInfo::for_compute_capability(10, 0)
    }

    fn sm120_info() -> DeviceOccupancyInfo {
        DeviceOccupancyInfo::for_compute_capability(12, 0)
    }

    #[test]
    fn test_sm100_device_info_attributes() {
        let info = sm100_info();
        assert_eq!(info.sm_count, 132, "Blackwell B100 has 132 SMs");
        assert_eq!(info.max_threads_per_sm, 2048);
        assert_eq!(info.max_blocks_per_sm, 32);
        assert_eq!(info.max_shared_memory_per_sm, 262144, "256 KiB shared/SM");
        assert_eq!(info.warp_size, 32);
    }

    #[test]
    fn test_sm120_device_info_attributes() {
        let info = sm120_info();
        assert_eq!(info.sm_count, 148, "Blackwell B200 has 148 SMs");
        assert_eq!(info.max_threads_per_sm, 2048);
        assert_eq!(info.max_blocks_per_sm, 32);
        assert_eq!(info.max_shared_memory_per_sm, 262144, "256 KiB shared/SM");
        assert_eq!(info.warp_size, 32);
    }

    #[test]
    fn test_sm100_occupancy_estimation() {
        let calc = OccupancyCalculator::new(sm100_info());
        // 256 threads = 8 warps/block; max_warps = 2048/32 = 64
        // 64 / 8 = 8 blocks; limited by max_blocks_per_sm = 32 → 8 blocks fit
        // active_warps = 8 * 8 = 64; ratio = 64/64 = 1.0 (full occupancy)
        let est = calc.estimate_occupancy(256, 0, 0);
        assert!(
            est.occupancy_ratio > 0.0,
            "Blackwell B100 must report positive occupancy"
        );
        assert!(
            est.active_warps_per_sm <= 64,
            "Active warps must not exceed hardware limit"
        );
    }

    #[test]
    fn test_sm120_full_occupancy() {
        let calc = OccupancyCalculator::new(sm120_info());
        // 64-thread block = 2 warps; 64 max warps → 32 blocks, limited by
        // max_blocks_per_sm = 32 ⇒ 32 blocks × 2 warps = 64 warps ⇒ ratio=1.0
        let est = calc.estimate_occupancy(64, 0, 0);
        assert_eq!(est.occupancy_ratio, 1.0, "Should reach full occupancy");
        assert_eq!(est.active_warps_per_sm, 64);
    }

    #[test]
    fn test_sm100_large_shared_memory_limit() {
        let calc = OccupancyCalculator::new(sm100_info());
        // 128 KiB per block: 262144 / 131072 = 2 blocks fit
        let smem_per_block = 131_072u32;
        let est = calc.estimate_occupancy(1024, 0, smem_per_block);
        // active_blocks ≤ 2; active_warps = 2 × (1024/32) = 2 × 32 = 64
        assert!(
            matches!(est.limiting_factor, LimitingFactor::SharedMemory),
            "Large smem must be the bottleneck"
        );
    }

    #[test]
    fn test_for_compute_capability_unknown_falls_back() {
        // An architecture not in the table must return a sane fallback.
        let info = DeviceOccupancyInfo::for_compute_capability(99, 99);
        let calc = OccupancyCalculator::new(info);
        let est = calc.estimate_occupancy(256, 0, 0);
        assert!(est.occupancy_ratio > 0.0);
    }

    #[test]
    fn test_sm100_vs_sm90_shared_memory_capacity() {
        let hopper = DeviceOccupancyInfo::for_compute_capability(9, 0);
        let blackwell = sm100_info();
        // Blackwell has strictly more shared memory per SM than Hopper.
        assert!(
            blackwell.max_shared_memory_per_sm > hopper.max_shared_memory_per_sm,
            "Blackwell B100 must have larger smem than Hopper H100"
        );
    }

    #[test]
    fn test_sm120_vs_sm100_sm_count() {
        let b100 = sm100_info();
        let b200 = sm120_info();
        // B200 has more SMs than B100.
        assert!(
            b200.sm_count > b100.sm_count,
            "Blackwell B200 must have more SMs than B100"
        );
    }

    #[test]
    fn test_for_compute_capability_all_known_arches() {
        // Ensure all known architectures parse without panic and return
        // warp_size == 32.
        let arches = [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)];
        for (major, minor) in arches {
            let info = DeviceOccupancyInfo::for_compute_capability(major, minor);
            assert_eq!(info.warp_size, 32, "sm_{major}{minor} warp_size must be 32");
            assert!(info.sm_count > 0);
            assert!(info.max_threads_per_sm > 0);
        }
    }
}
