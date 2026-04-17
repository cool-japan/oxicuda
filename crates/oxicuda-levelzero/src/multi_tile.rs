//! Multi-tile / multi-device dispatch for Intel Level Zero GPUs.
//!
//! Intel Xe-HPC (Ponte Vecchio) and other large-tile Intel GPUs expose individual
//! compute tiles as Level Zero "sub-devices". This module discovers those
//! sub-devices and distributes matrix work across them.
//!
//! # Overview
//!
//! ```text
//! LevelZeroDevice (root)
//!   ├── Tile 0  (sub-device handle)
//!   ├── Tile 1
//!   └── Tile N-1
//! ```
//!
//! When sub-devices are not available (older GPUs, consumer Xe Arc), the
//! [`MultiTileDispatcher`] transparently falls back to single-device dispatch.

use std::fmt;

// ─── Work Distribution Strategy ───────────────────────────────────────────────

/// How to partition a matrix operation across multiple compute tiles.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum WorkDistribution {
    /// Divide the M dimension of the output matrix into equal-sized row slabs,
    /// one per tile. If `m` is not evenly divisible, the last tile gets the
    /// remainder rows.
    #[default]
    EvenSplit,
    /// Assign exactly `rows_per_tile` rows to each tile except the last.
    RowSlab { rows_per_tile: usize },
}

// ─── Sub-device Discovery ─────────────────────────────────────────────────────

/// Metadata for a single Level Zero sub-device (compute tile).
#[derive(Debug, Clone)]
pub struct SubDeviceInfo {
    /// Zero-based tile index within the parent device.
    pub index: usize,
    /// Human-readable device name (may include tile index suffix).
    pub name: String,
    /// Reported number of EU/XVE execution units on this tile.
    pub eu_count: u32,
}

/// Work assignment for one tile during a GEMM dispatch.
#[derive(Debug, Clone)]
pub struct TileWorkSlice {
    /// Tile that will execute this slice.
    pub tile_index: usize,
    /// Starting row index in the M dimension (inclusive).
    pub row_start: usize,
    /// Exclusive end row (i.e., this tile processes rows `row_start..row_end`).
    pub row_end: usize,
}

impl TileWorkSlice {
    /// Number of rows assigned to this tile.
    #[inline]
    pub fn rows(&self) -> usize {
        self.row_end - self.row_start
    }
}

// ─── MultiTileConfig ───────────────────────────────────────────────────────────

/// Configuration for the multi-tile dispatcher.
#[derive(Debug, Clone)]
pub struct MultiTileConfig {
    /// Work partitioning strategy.
    pub strategy: WorkDistribution,
    /// Maximum number of tiles to use (0 = use all available).
    pub max_tiles: usize,
    /// Minimum problem size (M rows) below which single-device dispatch is
    /// preferred over multi-tile dispatch to avoid scheduling overhead.
    pub min_rows_for_multi_tile: usize,
}

impl Default for MultiTileConfig {
    fn default() -> Self {
        Self {
            strategy: WorkDistribution::EvenSplit,
            max_tiles: 0,
            min_rows_for_multi_tile: 64,
        }
    }
}

// ─── MultiTileDispatcher ──────────────────────────────────────────────────────

/// Enumerates Level Zero sub-devices and partitions matrix work across tiles.
///
/// On devices without sub-devices (single-tile GPUs), the dispatcher degrades
/// gracefully to single-device behaviour — callers do not need to special-case
/// this.
#[derive(Debug)]
pub struct MultiTileDispatcher {
    /// Discovered sub-devices. Empty means single-device fallback.
    pub sub_devices: Vec<SubDeviceInfo>,
    /// Active configuration.
    pub config: MultiTileConfig,
}

impl MultiTileDispatcher {
    /// Construct a dispatcher with pre-discovered sub-device information.
    ///
    /// Typically called by the backend after enumerating the Level Zero
    /// device tree. Pass an empty `sub_devices` vec for single-tile GPUs.
    pub fn new(sub_devices: Vec<SubDeviceInfo>, config: MultiTileConfig) -> Self {
        Self {
            sub_devices,
            config,
        }
    }

    /// Construct a single-device dispatcher (no sub-device enumeration).
    pub fn single_device() -> Self {
        Self::new(Vec::new(), MultiTileConfig::default())
    }

    /// Return how many tiles are available for dispatch.
    ///
    /// Returns 1 when no sub-devices were discovered (single-tile path).
    pub fn tile_count(&self) -> usize {
        let n = self.sub_devices.len().max(1);
        if self.config.max_tiles == 0 {
            n
        } else {
            n.min(self.config.max_tiles)
        }
    }

    /// Return `true` when multi-tile dispatch should be used for a problem of
    /// size `m` (number of output matrix rows).
    pub fn should_use_multi_tile(&self, m: usize) -> bool {
        self.sub_devices.len() > 1 && m >= self.config.min_rows_for_multi_tile
    }

    /// Partition `m` rows across available tiles according to the configured
    /// `WorkDistribution` strategy.
    ///
    /// Returns a `Vec<TileWorkSlice>` with one entry per active tile. The
    /// slices are non-overlapping and together cover the full `[0, m)` range.
    ///
    /// If there are no sub-devices or `m == 0`, returns a single slice for
    /// tile 0 covering the entire row range.
    pub fn partition(&self, m: usize) -> Vec<TileWorkSlice> {
        if m == 0 {
            return vec![TileWorkSlice {
                tile_index: 0,
                row_start: 0,
                row_end: 0,
            }];
        }

        let n_tiles = self.tile_count();

        if n_tiles <= 1 {
            return vec![TileWorkSlice {
                tile_index: 0,
                row_start: 0,
                row_end: m,
            }];
        }

        let rows_per_tile = match &self.config.strategy {
            WorkDistribution::EvenSplit => m.div_ceil(n_tiles),
            WorkDistribution::RowSlab { rows_per_tile } => *rows_per_tile,
        };

        let mut slices = Vec::with_capacity(n_tiles);
        let mut row_start = 0usize;

        for i in 0..n_tiles {
            if row_start >= m {
                break;
            }
            let row_end = if i == n_tiles - 1 {
                m // last tile gets remainder
            } else {
                (row_start + rows_per_tile).min(m)
            };
            slices.push(TileWorkSlice {
                tile_index: i,
                row_start,
                row_end,
            });
            row_start = row_end;
        }

        slices
    }

    /// Simulate sub-device enumeration from a list of synthetic device names.
    ///
    /// This is useful for unit tests that cannot call into Level Zero hardware.
    pub fn from_synthetic(names: &[&str]) -> Self {
        let sub_devices = names
            .iter()
            .enumerate()
            .map(|(i, &name)| SubDeviceInfo {
                index: i,
                name: name.to_string(),
                eu_count: 512,
            })
            .collect();
        Self::new(sub_devices, MultiTileConfig::default())
    }
}

impl fmt::Display for MultiTileDispatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiTileDispatcher {{ tiles: {}, strategy: {:?} }}",
            self.tile_count(),
            self.config.strategy
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_device_returns_one_tile() {
        let d = MultiTileDispatcher::single_device();
        assert_eq!(d.tile_count(), 1);
        assert!(!d.should_use_multi_tile(1024));
    }

    #[test]
    fn four_tile_even_split() {
        let d = MultiTileDispatcher::from_synthetic(&["tile0", "tile1", "tile2", "tile3"]);
        assert_eq!(d.tile_count(), 4);

        let slices = d.partition(256);
        assert_eq!(slices.len(), 4);
        // Each tile should get 64 rows.
        for (i, s) in slices.iter().enumerate() {
            assert_eq!(s.tile_index, i);
            assert_eq!(s.rows(), 64, "tile {i} expected 64 rows");
        }
        // Coverage must span [0, 256).
        assert_eq!(slices.first().unwrap().row_start, 0);
        assert_eq!(slices.last().unwrap().row_end, 256);
    }

    #[test]
    fn uneven_split_last_tile_gets_remainder() {
        let d = MultiTileDispatcher::from_synthetic(&["t0", "t1", "t2"]);
        let slices = d.partition(100); // 100 / 3 = 33 rem 1
        assert_eq!(slices.len(), 3);
        assert_eq!(slices[0].rows(), 34); // ceil(100/3)
        assert_eq!(slices[1].rows(), 34);
        assert_eq!(slices[2].rows(), 32); // remainder
        assert_eq!(slices[2].row_end, 100);
    }

    #[test]
    fn row_slab_strategy() {
        let mut d = MultiTileDispatcher::from_synthetic(&["a", "b", "c"]);
        d.config.strategy = WorkDistribution::RowSlab { rows_per_tile: 50 };
        let slices = d.partition(120);
        assert_eq!(slices[0].rows(), 50);
        assert_eq!(slices[1].rows(), 50);
        assert_eq!(slices[2].rows(), 20); // remainder
    }

    #[test]
    fn max_tiles_cap() {
        let mut d = MultiTileDispatcher::from_synthetic(&["a", "b", "c", "d"]);
        d.config.max_tiles = 2;
        assert_eq!(d.tile_count(), 2);
        let slices = d.partition(200);
        assert_eq!(slices.len(), 2);
        assert_eq!(slices.last().unwrap().row_end, 200);
    }

    #[test]
    fn zero_rows_returns_empty_slice() {
        let d = MultiTileDispatcher::from_synthetic(&["a", "b"]);
        let slices = d.partition(0);
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].rows(), 0);
    }

    #[test]
    fn should_use_multi_tile_threshold() {
        let d = MultiTileDispatcher::from_synthetic(&["a", "b"]);
        assert!(!d.should_use_multi_tile(32)); // below threshold (64)
        assert!(d.should_use_multi_tile(64));
        assert!(d.should_use_multi_tile(512));
    }

    #[test]
    fn display_format() {
        let d = MultiTileDispatcher::single_device();
        let s = format!("{d}");
        assert!(s.contains("MultiTileDispatcher"));
        assert!(s.contains("tiles: 1"));
    }

    #[test]
    fn sub_device_info_fields() {
        let info = SubDeviceInfo {
            index: 2,
            name: "Intel Xe-HPC Tile 2".to_string(),
            eu_count: 448,
        };
        assert_eq!(info.index, 2);
        assert_eq!(info.eu_count, 448);
    }

    #[test]
    fn work_slice_rows_calculation() {
        let slice = TileWorkSlice {
            tile_index: 0,
            row_start: 100,
            row_end: 200,
        };
        assert_eq!(slice.rows(), 100);
    }
}
