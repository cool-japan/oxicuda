//! Multi-GPU HIP dispatch for AMD ROCm workloads.
//!
//! Enumerates all HIP-capable GPUs on the system and distributes matrix
//! work across them using a row-slab partitioning strategy. When only one
//! GPU is available, the dispatcher transparently falls through to
//! single-device execution.
//!
//! # Architecture
//!
//! ```text
//! MultiDeviceDispatcher
//!   ├── DeviceInfo { id, name, total_memory, compute_units }
//!   ├── DeviceInfo { ... }
//!   └── DeviceInfo { ... }
//! ```

use crate::error::{RocmError, RocmResult};
use std::fmt;

// ─── DeviceInfo ───────────────────────────────────────────────────────────────

/// Properties of a single HIP-visible GPU.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// HIP device ordinal (0-based).
    pub id: i32,
    /// Device name as reported by `hipGetDeviceProperties`.
    pub name: String,
    /// Total device memory in bytes.
    pub total_memory: u64,
    /// Number of compute units (CUs / SMs).
    pub compute_units: u32,
    /// Whether this device is part of an xGMI/NVLink peer group.
    pub peer_accessible: bool,
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPU[{}] {} ({} MB, {} CUs)",
            self.id,
            self.name,
            self.total_memory / (1024 * 1024),
            self.compute_units,
        )
    }
}

// ─── WorkSlice ─────────────────────────────────────────────────────────────────

/// A contiguous slice of M rows assigned to one GPU.
#[derive(Debug, Clone)]
pub struct WorkSlice {
    /// Device ordinal to execute this slice on.
    pub device_id: i32,
    /// First row index (inclusive).
    pub row_start: usize,
    /// Last row index (exclusive).
    pub row_end: usize,
}

impl WorkSlice {
    /// Number of rows in this slice.
    #[inline]
    pub fn rows(&self) -> usize {
        self.row_end - self.row_start
    }
}

// ─── PartitionStrategy ────────────────────────────────────────────────────────

/// How to divide work across multiple GPUs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PartitionStrategy {
    /// Divide M rows equally (ceiling division) across all GPUs.
    #[default]
    EqualRows,
    /// Divide proportionally to each GPU's reported compute unit count.
    ProportionalToCUs,
}

// ─── MultiDeviceConfig ────────────────────────────────────────────────────────

/// Configuration for multi-GPU dispatch.
#[derive(Debug, Clone)]
pub struct MultiDeviceConfig {
    /// Partitioning algorithm.
    pub strategy: PartitionStrategy,
    /// Maximum number of GPUs to use (0 = all discovered).
    pub max_devices: usize,
    /// Minimum M rows required before multi-GPU dispatch is preferred.
    pub min_rows_for_multi_device: usize,
}

impl Default for MultiDeviceConfig {
    fn default() -> Self {
        Self {
            strategy: PartitionStrategy::EqualRows,
            max_devices: 0,
            min_rows_for_multi_device: 128,
        }
    }
}

// ─── MultiDeviceDispatcher ────────────────────────────────────────────────────

/// Manages multi-GPU dispatch for ROCm/HIP workloads.
///
/// Construct via [`MultiDeviceDispatcher::from_devices`] with pre-discovered
/// device info (from `hipGetDeviceCount` / `hipGetDeviceProperties`), or
/// use [`MultiDeviceDispatcher::single_device`] when only one GPU is present.
#[derive(Debug)]
pub struct MultiDeviceDispatcher {
    /// All discovered GPUs.
    pub devices: Vec<DeviceInfo>,
    /// Dispatch configuration.
    pub config: MultiDeviceConfig,
}

impl MultiDeviceDispatcher {
    /// Construct a dispatcher with pre-enumerated device info.
    pub fn from_devices(devices: Vec<DeviceInfo>, config: MultiDeviceConfig) -> Self {
        Self { devices, config }
    }

    /// Construct a single-GPU dispatcher (no enumeration needed).
    pub fn single_device(device_id: i32, name: impl Into<String>) -> Self {
        Self {
            devices: vec![DeviceInfo {
                id: device_id,
                name: name.into(),
                total_memory: 0,
                compute_units: 0,
                peer_accessible: false,
            }],
            config: MultiDeviceConfig::default(),
        }
    }

    /// Number of GPUs that will be used by the dispatcher.
    pub fn active_device_count(&self) -> usize {
        let n = self.devices.len();
        if self.config.max_devices == 0 {
            n
        } else {
            n.min(self.config.max_devices)
        }
    }

    /// Return `true` when multi-GPU dispatch should be activated for `m` rows.
    pub fn should_use_multi_device(&self, m: usize) -> bool {
        self.active_device_count() > 1 && m >= self.config.min_rows_for_multi_device
    }

    /// Partition `m` output rows across the active GPUs.
    ///
    /// Returns a [`Vec<WorkSlice>`] covering `[0, m)` with no gaps or overlap.
    pub fn partition(&self, m: usize) -> RocmResult<Vec<WorkSlice>> {
        if m == 0 {
            return Ok(vec![WorkSlice {
                device_id: self.devices[0].id,
                row_start: 0,
                row_end: 0,
            }]);
        }

        let n = self.active_device_count();
        if n == 0 {
            return Err(RocmError::NoSuitableDevice);
        }
        if n == 1 {
            let id = self.devices[0].id;
            return Ok(vec![WorkSlice {
                device_id: id,
                row_start: 0,
                row_end: m,
            }]);
        }

        let active: &[DeviceInfo] = &self.devices[..n];
        let slices = match &self.config.strategy {
            PartitionStrategy::EqualRows => Self::partition_equal(active, m),
            PartitionStrategy::ProportionalToCUs => Self::partition_proportional(active, m),
        };
        Ok(slices)
    }

    fn partition_equal(devices: &[DeviceInfo], m: usize) -> Vec<WorkSlice> {
        let n = devices.len();
        let rows_each = m.div_ceil(n);
        let mut slices = Vec::with_capacity(n);
        let mut start = 0usize;
        for (i, dev) in devices.iter().enumerate() {
            if start >= m {
                break;
            }
            let end = if i == n - 1 {
                m
            } else {
                (start + rows_each).min(m)
            };
            slices.push(WorkSlice {
                device_id: dev.id,
                row_start: start,
                row_end: end,
            });
            start = end;
        }
        slices
    }

    fn partition_proportional(devices: &[DeviceInfo], m: usize) -> Vec<WorkSlice> {
        let total_cu: u64 = devices.iter().map(|d| d.compute_units as u64).sum();
        if total_cu == 0 {
            // Fall back to equal split.
            return Self::partition_equal(devices, m);
        }

        let mut slices = Vec::with_capacity(devices.len());
        let mut start = 0usize;
        let n = devices.len();

        for (i, dev) in devices.iter().enumerate() {
            if start >= m {
                break;
            }
            let rows = if i == n - 1 {
                m - start
            } else {
                ((m as u64 * dev.compute_units as u64).div_ceil(total_cu)) as usize
            };
            let end = (start + rows).min(m);
            slices.push(WorkSlice {
                device_id: dev.id,
                row_start: start,
                row_end: end,
            });
            start = end;
        }
        slices
    }

    /// Return a summary string listing all discovered devices.
    pub fn device_summary(&self) -> String {
        self.devices
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl fmt::Display for MultiDeviceDispatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiDeviceDispatcher {{ gpus: {}, strategy: {:?} }}",
            self.active_device_count(),
            self.config.strategy
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_devices(count: usize) -> Vec<DeviceInfo> {
        (0..count)
            .map(|i| DeviceInfo {
                id: i as i32,
                name: format!("AMD Instinct MI250 [{i}]"),
                total_memory: 64 * 1024 * 1024 * 1024,
                compute_units: 110,
                peer_accessible: true,
            })
            .collect()
    }

    #[test]
    fn single_device_covers_full_range() {
        let d = MultiDeviceDispatcher::single_device(0, "TestGPU");
        let slices = d.partition(512).unwrap();
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].row_end, 512);
        assert_eq!(slices[0].device_id, 0);
    }

    #[test]
    fn four_gpu_equal_split() {
        let d = MultiDeviceDispatcher::from_devices(make_devices(4), MultiDeviceConfig::default());
        let slices = d.partition(400).unwrap();
        assert_eq!(slices.len(), 4);
        for s in &slices {
            assert_eq!(s.rows(), 100);
        }
        assert_eq!(slices[0].row_start, 0);
        assert_eq!(slices[3].row_end, 400);
    }

    #[test]
    fn uneven_rows_last_device_gets_remainder() {
        let d = MultiDeviceDispatcher::from_devices(make_devices(3), MultiDeviceConfig::default());
        let slices = d.partition(100).unwrap();
        assert_eq!(slices.len(), 3);
        let sum: usize = slices.iter().map(|s| s.rows()).sum();
        assert_eq!(sum, 100);
        assert_eq!(slices.last().unwrap().row_end, 100);
    }

    #[test]
    fn max_devices_cap() {
        let cfg = MultiDeviceConfig {
            max_devices: 2,
            ..Default::default()
        };
        let d = MultiDeviceDispatcher::from_devices(make_devices(4), cfg);
        assert_eq!(d.active_device_count(), 2);
        let slices = d.partition(200).unwrap();
        assert_eq!(slices.len(), 2);
        assert_eq!(slices.last().unwrap().row_end, 200);
    }

    #[test]
    fn zero_rows_returns_single_empty_slice() {
        let d = MultiDeviceDispatcher::from_devices(make_devices(2), MultiDeviceConfig::default());
        let slices = d.partition(0).unwrap();
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].rows(), 0);
    }

    #[test]
    fn should_use_multi_device_threshold() {
        let d = MultiDeviceDispatcher::from_devices(make_devices(2), MultiDeviceConfig::default());
        assert!(!d.should_use_multi_device(64)); // below min_rows=128
        assert!(d.should_use_multi_device(128));
        assert!(d.should_use_multi_device(1024));
    }

    #[test]
    fn proportional_partition_sums_to_m() {
        let devices: Vec<DeviceInfo> = vec![
            DeviceInfo {
                id: 0,
                name: "A".into(),
                total_memory: 0,
                compute_units: 200,
                peer_accessible: false,
            },
            DeviceInfo {
                id: 1,
                name: "B".into(),
                total_memory: 0,
                compute_units: 100,
                peer_accessible: false,
            },
        ];
        let cfg = MultiDeviceConfig {
            strategy: PartitionStrategy::ProportionalToCUs,
            ..Default::default()
        };
        let d = MultiDeviceDispatcher::from_devices(devices, cfg);
        let slices = d.partition(300).unwrap();
        let sum: usize = slices.iter().map(|s| s.rows()).sum();
        assert_eq!(sum, 300);
    }

    #[test]
    fn device_summary_includes_all_names() {
        let d = MultiDeviceDispatcher::from_devices(make_devices(3), MultiDeviceConfig::default());
        let summary = d.device_summary();
        assert!(summary.contains("MI250"));
        assert!(summary.contains("[0]"));
        assert!(summary.contains("[2]"));
    }

    #[test]
    fn display_format() {
        let d = MultiDeviceDispatcher::from_devices(make_devices(2), MultiDeviceConfig::default());
        let s = format!("{d}");
        assert!(s.contains("MultiDeviceDispatcher"));
        assert!(s.contains("gpus: 2"));
    }

    #[test]
    fn work_slice_rows() {
        let s = WorkSlice {
            device_id: 0,
            row_start: 50,
            row_end: 150,
        };
        assert_eq!(s.rows(), 100);
    }
}
