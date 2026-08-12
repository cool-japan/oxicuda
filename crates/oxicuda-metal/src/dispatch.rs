//! Threadgroup and grid-size planning for compute dispatch.
//!
//! Every `dispatchThreadgroups:threadsPerThreadgroup:` call needs a grid and a
//! threadgroup shape.  Choosing them well — respecting the pipeline's
//! `maxTotalThreadsPerThreadgroup`, the SIMD width, and the threadgroup-memory
//! budget — is pure arithmetic and fully unit-testable without a device.
//!
//! This module provides [`DispatchPlanner`], which turns a logical problem
//! extent (1-D element count, 2-D matrix tile, or 3-D batched grid) into a
//! [`DispatchPlan`] of threadgroup and grid sizes, plus helpers for picking a
//! SIMD-aligned threadgroup width and sizing threadgroup scratch memory.

use crate::device_family::MetalDeviceCapabilities;
use crate::error::{MetalError, MetalResult};

/// Narrow a grid extent to the `u32` Metal's dispatch descriptors use, failing
/// loudly instead of wrapping.
///
/// An unchecked `as u32` turns a grid of more than `2^32` threadgroups into a
/// small number, silently launching a fraction of the requested work.
fn grid_extent(value: u64, what: &str) -> MetalResult<u32> {
    u32::try_from(value).map_err(|_| {
        MetalError::InvalidArgument(format!(
            "{what} = {value} exceeds the u32 grid extent a Metal dispatch accepts"
        ))
    })
}

/// A planned dispatch: threadgroup shape and grid (threadgroups-per-grid) shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchPlan {
    /// Threads per threadgroup `(x, y, z)`.
    pub threads_per_threadgroup: [u32; 3],
    /// Threadgroups per grid `(x, y, z)`.
    pub threadgroups_per_grid: [u32; 3],
}

impl DispatchPlan {
    /// Total threads launched (`grid * group`, all dimensions).
    pub fn total_threads(&self) -> u64 {
        let group: u64 = self
            .threads_per_threadgroup
            .iter()
            .map(|&d| u64::from(d))
            .product();
        let grid: u64 = self
            .threadgroups_per_grid
            .iter()
            .map(|&d| u64::from(d))
            .product();
        group * grid
    }

    /// Total threads in a single threadgroup (`x * y * z`).
    pub fn threads_per_group(&self) -> u32 {
        self.threads_per_threadgroup.iter().product()
    }
}

/// Plans dispatch geometry for a device's capabilities.
#[derive(Debug, Clone, Copy)]
pub struct DispatchPlanner {
    max_threads: u32,
    simd_width: u32,
    threadgroup_memory: usize,
}

impl DispatchPlanner {
    /// Build a planner from device capabilities.
    pub fn new(caps: MetalDeviceCapabilities) -> Self {
        Self {
            max_threads: caps.max_threads_per_threadgroup as u32,
            simd_width: caps.family.simd_width() as u32,
            threadgroup_memory: caps.threadgroup_memory,
        }
    }

    /// SIMD-group width for this device (typically 32 on Apple GPUs).
    pub fn simd_width(&self) -> u32 {
        self.simd_width
    }

    /// Maximum threads per threadgroup.
    pub fn max_threads_per_threadgroup(&self) -> u32 {
        self.max_threads
    }

    /// Plan a 1-D dispatch over `n` elements.
    ///
    /// The threadgroup width is the largest power-of-two multiple of the SIMD
    /// width that does not exceed `max_threads` nor `n`.  The grid covers all
    /// `n` elements with a ceiling division.
    pub fn plan_1d(&self, n: usize) -> MetalResult<DispatchPlan> {
        if n == 0 {
            return Err(MetalError::InvalidArgument(
                "dispatch element count must be > 0".into(),
            ));
        }
        let tg = self.pick_1d_threadgroup(n);
        let groups = grid_extent((n as u64).div_ceil(u64::from(tg)), "1-D threadgroup count")?;
        Ok(DispatchPlan {
            threads_per_threadgroup: [tg, 1, 1],
            threadgroups_per_grid: [groups, 1, 1],
        })
    }

    /// Plan a 2-D dispatch over a `rows × cols` grid using a square
    /// `tile × tile` threadgroup (clamped so `tile*tile <= max_threads`).
    pub fn plan_2d(&self, rows: usize, cols: usize) -> MetalResult<DispatchPlan> {
        if rows == 0 || cols == 0 {
            return Err(MetalError::InvalidArgument(
                "dispatch dimensions must be > 0".into(),
            ));
        }
        // Largest power-of-two tile with tile^2 <= max_threads, capped at 16.
        // A 16x16 (=256-thread) tile is the crate-wide GEMM convention and keeps
        // occupancy high; larger square tiles waste threads on ragged edges.
        const MAX_TILE: u32 = 16;
        let mut tile = 1u32;
        while (tile * 2) * (tile * 2) <= self.max_threads && tile < MAX_TILE {
            tile *= 2;
        }
        let gx = grid_extent((cols as u64).div_ceil(u64::from(tile)), "2-D grid width")?;
        let gy = grid_extent((rows as u64).div_ceil(u64::from(tile)), "2-D grid height")?;
        Ok(DispatchPlan {
            threads_per_threadgroup: [tile, tile, 1],
            threadgroups_per_grid: [gx, gy, 1],
        })
    }

    /// Plan a 3-D batched dispatch: a 2-D `rows × cols` tile per batch slice,
    /// with `batch` slices along `z`.
    pub fn plan_batched_2d(
        &self,
        rows: usize,
        cols: usize,
        batch: usize,
    ) -> MetalResult<DispatchPlan> {
        if batch == 0 {
            return Err(MetalError::InvalidArgument(
                "batch count must be > 0".into(),
            ));
        }
        let mut plan = self.plan_2d(rows, cols)?;
        plan.threadgroups_per_grid[2] = grid_extent(batch as u64, "batch grid depth")?;
        Ok(plan)
    }

    /// Pick a 1-D threadgroup width: a power-of-two multiple of the SIMD width,
    /// capped at `max_threads` and at `n`.
    fn pick_1d_threadgroup(&self, n: usize) -> u32 {
        let mut tg = self.simd_width.max(1);
        while tg * 2 <= self.max_threads && (tg * 2) as usize <= n {
            tg *= 2;
        }
        // Never launch a group wider than the work itself. `n` can exceed
        // `u32::MAX`, where a bare `n as u32` would truncate to an arbitrarily
        // small width; saturate instead so the loop's choice stands.
        tg.min(u32::try_from(n).unwrap_or(u32::MAX)).max(1)
    }

    /// Bytes of threadgroup scratch required for `elements` of `elem_size`,
    /// validated against the device budget.
    ///
    /// Returns [`MetalError::OutOfMemory`] if the request exceeds the
    /// threadgroup-memory budget.
    pub fn threadgroup_scratch_bytes(
        &self,
        elements: usize,
        elem_size: usize,
    ) -> MetalResult<usize> {
        let bytes = elements.saturating_mul(elem_size);
        if bytes > self.threadgroup_memory {
            return Err(MetalError::OutOfMemory);
        }
        Ok(bytes)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_family::MetalGpuFamily;

    fn planner() -> DispatchPlanner {
        DispatchPlanner::new(MetalDeviceCapabilities::from_family(MetalGpuFamily::Apple8))
    }

    #[test]
    fn plan_1d_covers_all_elements() {
        let p = planner();
        let plan = p.plan_1d(1000).expect("plan");
        // Must cover at least 1000 threads.
        assert!(plan.total_threads() >= 1000);
        // Threadgroup width is a multiple of the SIMD width (32).
        assert_eq!(plan.threads_per_threadgroup[0] % 32, 0);
        assert!(plan.threads_per_threadgroup[0] <= 1024);
    }

    #[test]
    fn plan_1d_small_n_caps_threadgroup() {
        let p = planner();
        let plan = p.plan_1d(10).expect("plan");
        // Width never exceeds the work; here SIMD width 32 caps to 10.
        assert!(plan.threads_per_threadgroup[0] <= 32);
        assert_eq!(plan.threadgroups_per_grid[0], 1);
    }

    #[test]
    fn plan_1d_zero_errors() {
        assert!(planner().plan_1d(0).is_err());
    }

    #[test]
    fn plan_2d_square_tile_within_budget() {
        let p = planner();
        let plan = p.plan_2d(100, 200).expect("plan");
        // 16x16 = 256 <= 1024 max threads.
        assert_eq!(plan.threads_per_threadgroup[0], 16);
        assert_eq!(plan.threads_per_threadgroup[1], 16);
        assert!(plan.threads_per_group() <= 1024);
        // Grid covers the full extent.
        assert_eq!(plan.threadgroups_per_grid[0], 200u64.div_ceil(16) as u32);
        assert_eq!(plan.threadgroups_per_grid[1], 100u64.div_ceil(16) as u32);
    }

    #[test]
    fn plan_2d_zero_errors() {
        assert!(planner().plan_2d(0, 4).is_err());
        assert!(planner().plan_2d(4, 0).is_err());
    }

    #[test]
    fn plan_batched_sets_z() {
        let p = planner();
        let plan = p.plan_batched_2d(32, 32, 8).expect("plan");
        assert_eq!(plan.threadgroups_per_grid[2], 8);
        assert!(planner().plan_batched_2d(32, 32, 0).is_err());
    }

    #[test]
    fn smaller_family_has_lower_thread_cap() {
        let small =
            DispatchPlanner::new(MetalDeviceCapabilities::from_family(MetalGpuFamily::Apple4));
        assert_eq!(small.max_threads_per_threadgroup(), 512);
        let plan = small.plan_2d(64, 64).expect("plan");
        // 16x16=256 fits; would not exceed 512.
        assert!(plan.threads_per_group() <= 512);
    }

    #[test]
    fn threadgroup_scratch_budget() {
        let p = planner();
        // 256 floats = 1024 bytes, well within 32 KiB.
        assert_eq!(p.threadgroup_scratch_bytes(256, 4).expect("ok"), 1024);
        // Exceeding the budget errors.
        assert!(p.threadgroup_scratch_bytes(100_000, 4).is_err());
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn oversized_extents_error_instead_of_truncating() {
        let p = planner();
        // 2^63 elements at a 1024-wide threadgroup needs 2^53 threadgroups —
        // `as u32` used to wrap this to 0 and launch nothing.
        assert!(matches!(
            p.plan_1d(1usize << 63),
            Err(MetalError::InvalidArgument(_))
        ));
        assert!(matches!(
            p.plan_2d(1usize << 40, 4),
            Err(MetalError::InvalidArgument(_))
        ));
        assert!(matches!(
            p.plan_batched_2d(4, 4, 1usize << 40),
            Err(MetalError::InvalidArgument(_))
        ));
        // A huge-but-representable extent still yields a sane threadgroup width.
        let plan = p.plan_1d(u32::MAX as usize).expect("plan");
        assert_eq!(plan.threads_per_threadgroup[0], 1024);
    }

    #[test]
    fn dispatch_plan_total_threads() {
        let plan = DispatchPlan {
            threads_per_threadgroup: [16, 16, 1],
            threadgroups_per_grid: [2, 3, 1],
        };
        assert_eq!(plan.threads_per_group(), 256);
        assert_eq!(plan.total_threads(), 256 * 6);
    }
}
