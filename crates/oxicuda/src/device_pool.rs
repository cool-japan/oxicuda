//! Thread-safe multi-GPU device pool with workload-aware scheduling.
//!
//! This module provides a higher-level abstraction over
//! [`oxicuda_driver::DevicePool`] for workload scheduling across multiple
//! GPUs. It supports round-robin, least-loaded, memory-aware, compute-aware,
//! weighted-random, and custom selection policies.
//!
//! # macOS
//!
//! On macOS (where NVIDIA drivers are unavailable), [`MultiGpuPool::new`](crate::device_pool::MultiGpuPool::new)
//! creates a synthetic pool with two fake devices so that tests and
//! downstream code can exercise scheduling logic without real hardware.
//!
//! # Example
//!
//! ```no_run
//! use oxicuda::device_pool::{MultiGpuPool, DeviceSelectionPolicy};
//!
//! let pool = MultiGpuPool::new(DeviceSelectionPolicy::RoundRobin)?;
//! let lease = pool.acquire()?;
//! println!("Acquired GPU {}", lease.ordinal());
//! pool.release(lease);
//! # Ok::<(), oxicuda_driver::CudaError>(())
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use oxicuda_driver::{CudaError, CudaResult};

// ---------------------------------------------------------------------------
// DeviceSelectionPolicy
// ---------------------------------------------------------------------------

/// Strategy for selecting a GPU from the pool.
pub enum DeviceSelectionPolicy {
    /// Cycle through devices sequentially.
    RoundRobin,
    /// Pick the device with the fewest active tasks.
    LeastLoaded,
    /// Pick the device with the most total memory.
    MostMemoryFree,
    /// Pick the device with the highest compute capability.
    BestCompute,
    /// Random selection weighted by the provided weights.
    ///
    /// The weight vector must have the same length as the device count.
    /// Higher weights increase the probability of selection.
    WeightedRandom {
        /// Per-device selection weights.
        weights: Vec<f64>,
    },
    /// User-defined selection function.
    ///
    /// The closure receives the current device statuses and returns the
    /// index into the status slice of the chosen device.
    #[allow(clippy::type_complexity)]
    Custom(Box<dyn Fn(&[DeviceStatus]) -> usize + Send + Sync>),
}

impl std::fmt::Debug for DeviceSelectionPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RoundRobin => write!(f, "RoundRobin"),
            Self::LeastLoaded => write!(f, "LeastLoaded"),
            Self::MostMemoryFree => write!(f, "MostMemoryFree"),
            Self::BestCompute => write!(f, "BestCompute"),
            Self::WeightedRandom { weights } => f
                .debug_struct("WeightedRandom")
                .field("weights", weights)
                .finish(),
            Self::Custom(_) => write!(f, "Custom(<closure>)"),
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceStatus
// ---------------------------------------------------------------------------

/// Snapshot of a device's current status within the pool.
#[derive(Debug, Clone)]
pub struct DeviceStatus {
    /// Device ordinal (0-based).
    pub ordinal: i32,
    /// Human-readable device name.
    pub name: String,
    /// Total device memory in bytes.
    pub total_memory: usize,
    /// Number of tasks currently active on this device.
    pub active_tasks: u32,
    /// Compute capability as `(major, minor)`.
    pub compute_capability: (u32, u32),
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
    /// Whether the device is available for new work.
    pub is_available: bool,
}

// ---------------------------------------------------------------------------
// GpuTask
// ---------------------------------------------------------------------------

/// Metadata for an active GPU task.
#[derive(Debug, Clone)]
pub struct GpuTask {
    /// Unique task identifier.
    pub id: u64,
    /// Device ordinal this task is running on.
    pub device_ordinal: i32,
    /// Human-readable description.
    pub description: String,
    /// When the task was started.
    pub started_at: Instant,
}

// ---------------------------------------------------------------------------
// GpuLease
// ---------------------------------------------------------------------------

/// RAII handle for a leased GPU device.
///
/// When dropped, the lease automatically decrements the active-task count
/// on the device it was acquired from.
pub struct GpuLease {
    /// The device ordinal this lease represents.
    device_ordinal: i32,
    /// Unique task id for this lease.
    task_id: u64,
    /// Description of the work.
    description: String,
    /// Reference back to the pool for auto-release.
    pool: Arc<RwLock<PoolInner>>,
}

impl GpuLease {
    /// Returns the device ordinal for this lease.
    #[inline]
    pub fn ordinal(&self) -> i32 {
        self.device_ordinal
    }

    /// Returns the task description.
    #[inline]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns the unique task id.
    #[inline]
    pub fn task_id(&self) -> u64 {
        self.task_id
    }
}

impl Drop for GpuLease {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.pool.write() {
            inner.decrement_tasks(self.device_ordinal);
        }
    }
}

impl std::fmt::Debug for GpuLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuLease")
            .field("device_ordinal", &self.device_ordinal)
            .field("task_id", &self.task_id)
            .field("description", &self.description)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// PoolInner — shared mutable state
// ---------------------------------------------------------------------------

/// Internal mutable state protected by `RwLock`.
struct PoolInner {
    /// Per-device status entries.
    devices: Vec<DeviceStatus>,
    /// Current selection policy.
    policy: DeviceSelectionPolicy,
    /// Round-robin cursor.
    rr_cursor: usize,
}

impl PoolInner {
    /// Select a device index according to the current policy.
    fn select(&mut self) -> CudaResult<usize> {
        if self.devices.is_empty() {
            return Err(CudaError::NoDevice);
        }
        let idx = match &self.policy {
            DeviceSelectionPolicy::RoundRobin => {
                let idx = self.rr_cursor % self.devices.len();
                self.rr_cursor = self.rr_cursor.wrapping_add(1);
                idx
            }
            DeviceSelectionPolicy::LeastLoaded => self
                .devices
                .iter()
                .enumerate()
                .filter(|(_, d)| d.is_available)
                .min_by_key(|(_, d)| d.active_tasks)
                .map(|(i, _)| i)
                .unwrap_or(0),
            DeviceSelectionPolicy::MostMemoryFree => self
                .devices
                .iter()
                .enumerate()
                .filter(|(_, d)| d.is_available)
                .max_by_key(|(_, d)| d.total_memory)
                .map(|(i, _)| i)
                .unwrap_or(0),
            DeviceSelectionPolicy::BestCompute => self
                .devices
                .iter()
                .enumerate()
                .filter(|(_, d)| d.is_available)
                .max_by_key(|(_, d)| d.compute_capability)
                .map(|(i, _)| i)
                .unwrap_or(0),
            DeviceSelectionPolicy::WeightedRandom { weights } => {
                weighted_random_select(weights, self.devices.len())
            }
            DeviceSelectionPolicy::Custom(f) => {
                let idx = f(&self.devices);
                if idx >= self.devices.len() { 0 } else { idx }
            }
        };
        Ok(idx)
    }

    /// Increment active_tasks for a device ordinal.
    fn increment_tasks(&mut self, ordinal: i32) {
        if let Some(dev) = self.devices.iter_mut().find(|d| d.ordinal == ordinal) {
            dev.active_tasks = dev.active_tasks.saturating_add(1);
        }
    }

    /// Decrement active_tasks for a device ordinal.
    fn decrement_tasks(&mut self, ordinal: i32) {
        if let Some(dev) = self.devices.iter_mut().find(|d| d.ordinal == ordinal) {
            dev.active_tasks = dev.active_tasks.saturating_sub(1);
        }
    }
}

/// Simple deterministic-ish weighted selection using the system clock as
/// entropy source. We avoid pulling in a full RNG crate.
fn weighted_random_select(weights: &[f64], device_count: usize) -> usize {
    if weights.is_empty() || device_count == 0 {
        return 0;
    }
    let total: f64 = weights.iter().take(device_count).sum();
    if total <= 0.0 {
        return 0;
    }
    // Use nanosecond timestamp as cheap pseudo-random source.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let r = (f64::from(nanos % 1_000_000) / 1_000_000.0) * total;
    let mut cumulative = 0.0;
    for (i, w) in weights.iter().take(device_count).enumerate() {
        cumulative += w;
        if r < cumulative {
            return i;
        }
    }
    device_count.saturating_sub(1)
}

// ---------------------------------------------------------------------------
// MultiGpuPool
// ---------------------------------------------------------------------------

/// Thread-safe multi-GPU device pool with policy-based scheduling.
///
/// Wraps a set of GPU devices and tracks per-device active-task counts.
/// Use [`acquire`](Self::acquire) to lease a GPU and
/// [`release`](Self::release) (or drop the [`GpuLease`]) to return it.
///
/// On macOS, `new()` creates a synthetic 2-device pool since no real
/// NVIDIA driver is available.
pub struct MultiGpuPool {
    /// Shared mutable pool state.
    inner: Arc<RwLock<PoolInner>>,
    /// Monotonically increasing task-id generator.
    next_task_id: AtomicU64,
}

impl MultiGpuPool {
    /// Create a pool that discovers all available GPUs.
    ///
    /// On macOS (or when no GPU is found), a synthetic 2-device pool is
    /// created so that scheduling logic can still be exercised.
    ///
    /// # Errors
    ///
    /// Returns an error only if internal lock poisoning occurs (should
    /// not happen under normal circumstances).
    pub fn new(policy: DeviceSelectionPolicy) -> CudaResult<Self> {
        let devices = Self::discover_devices();
        let inner = PoolInner {
            devices,
            policy,
            rr_cursor: 0,
        };
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
            next_task_id: AtomicU64::new(1),
        })
    }

    /// Create a pool for specific device ordinals.
    ///
    /// On macOS the ordinals are used to create synthetic entries.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if `ordinals` is empty.
    pub fn with_devices(ordinals: Vec<i32>, policy: DeviceSelectionPolicy) -> CudaResult<Self> {
        if ordinals.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        let devices: Vec<DeviceStatus> = ordinals
            .iter()
            .map(|&ord| Self::device_status_for(ord))
            .collect();
        let inner = PoolInner {
            devices,
            policy,
            rr_cursor: 0,
        };
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
            next_task_id: AtomicU64::new(1),
        })
    }

    /// Acquire a GPU lease according to the current policy.
    ///
    /// Increments the active-task count on the selected device.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NoDevice`] if the pool is empty or the lock
    /// is poisoned.
    pub fn acquire(&self) -> CudaResult<GpuLease> {
        self.acquire_with_description(String::new())
    }

    /// Acquire a GPU lease with a task description.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NoDevice`] if the pool is empty.
    pub fn acquire_with_description(&self, description: String) -> CudaResult<GpuLease> {
        let mut inner = self.inner.write().map_err(|_| CudaError::InvalidValue)?;
        let idx = inner.select()?;
        let ordinal = inner.devices[idx].ordinal;
        inner.increment_tasks(ordinal);
        let task_id = self.next_task_id.fetch_add(1, Ordering::Relaxed);
        Ok(GpuLease {
            device_ordinal: ordinal,
            task_id,
            description,
            pool: Arc::clone(&self.inner),
        })
    }

    /// Explicitly release a GPU lease.
    ///
    /// This is equivalent to dropping the lease, but makes intent clearer.
    pub fn release(&self, lease: GpuLease) {
        drop(lease);
    }

    /// Returns the number of devices in the pool.
    pub fn device_count(&self) -> usize {
        self.inner
            .read()
            .map(|inner| inner.devices.len())
            .unwrap_or(0)
    }

    /// Returns a snapshot of all device statuses.
    pub fn status(&self) -> Vec<DeviceStatus> {
        self.inner
            .read()
            .map(|inner| inner.devices.clone())
            .unwrap_or_default()
    }

    /// Change the selection policy at runtime.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock is poisoned.
    pub fn set_policy(&self, policy: DeviceSelectionPolicy) -> CudaResult<()> {
        let mut inner = self.inner.write().map_err(|_| CudaError::InvalidValue)?;
        inner.policy = policy;
        Ok(())
    }

    // -- private helpers --

    /// Discover real GPUs or fall back to synthetic devices.
    fn discover_devices() -> Vec<DeviceStatus> {
        // Try real GPU discovery first.
        if let Ok(devices) = Self::try_discover_real() {
            if !devices.is_empty() {
                return devices;
            }
        }
        // Fallback: synthetic 2-device pool for macOS / no-GPU environments.
        Self::synthetic_devices(2)
    }

    /// Attempt to discover real GPU devices via the driver.
    fn try_discover_real() -> CudaResult<Vec<DeviceStatus>> {
        oxicuda_driver::init()?;
        let count = oxicuda_driver::Device::count()?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count {
            let dev = oxicuda_driver::Device::get(i)?;
            let name = dev.name().unwrap_or_else(|_| format!("GPU-{i}"));
            let total_memory = dev.total_memory().unwrap_or(0);
            let cc = dev.compute_capability().unwrap_or((0, 0));
            let sm = dev.multiprocessor_count().unwrap_or(0);
            out.push(DeviceStatus {
                ordinal: i,
                name,
                total_memory,
                active_tasks: 0,
                compute_capability: (cc.0 as u32, cc.1 as u32),
                sm_count: sm as u32,
                is_available: true,
            });
        }
        Ok(out)
    }

    /// Create `n` synthetic device entries for testing.
    fn synthetic_devices(n: usize) -> Vec<DeviceStatus> {
        (0..n)
            .map(|i| DeviceStatus {
                ordinal: i as i32,
                name: format!("Synthetic GPU {i}"),
                total_memory: if i == 0 {
                    16 * 1024 * 1024 * 1024 // 16 GiB
                } else {
                    8 * 1024 * 1024 * 1024 // 8 GiB
                },
                active_tasks: 0,
                compute_capability: (8, i as u32),
                sm_count: (108 - (i as u32) * 24),
                is_available: true,
            })
            .collect()
    }

    /// Build a `DeviceStatus` for a specific ordinal, real or synthetic.
    fn device_status_for(ordinal: i32) -> DeviceStatus {
        // Try real device first.
        if oxicuda_driver::init().is_ok() {
            if let Ok(dev) = oxicuda_driver::Device::get(ordinal) {
                let name = dev.name().unwrap_or_else(|_| format!("GPU-{ordinal}"));
                let total_memory = dev.total_memory().unwrap_or(0);
                let cc = dev.compute_capability().unwrap_or((0, 0));
                let sm = dev.multiprocessor_count().unwrap_or(0);
                return DeviceStatus {
                    ordinal,
                    name,
                    total_memory,
                    active_tasks: 0,
                    compute_capability: (cc.0 as u32, cc.1 as u32),
                    sm_count: sm as u32,
                    is_available: true,
                };
            }
        }
        // Synthetic fallback — vary stats by ordinal for testing diversity.
        DeviceStatus {
            ordinal,
            name: format!("Synthetic GPU {ordinal}"),
            total_memory: if ordinal == 0 {
                16 * 1024 * 1024 * 1024 // 16 GiB
            } else {
                8 * 1024 * 1024 * 1024 // 8 GiB
            },
            active_tasks: 0,
            compute_capability: (8, ordinal as u32),
            sm_count: (108_u32).saturating_sub(ordinal as u32 * 24),
            is_available: true,
        }
    }
}

impl std::fmt::Debug for MultiGpuPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.device_count();
        f.debug_struct("MultiGpuPool")
            .field("device_count", &count)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// WorkloadBalancer
// ---------------------------------------------------------------------------

/// Distributes work items across GPUs in a [`MultiGpuPool`].
///
/// Provides batch-distribution and parallel-map utilities for spreading
/// computation across multiple devices.
pub struct WorkloadBalancer {
    /// Snapshot of device ordinals for distribution.
    ordinals: Vec<i32>,
}

impl WorkloadBalancer {
    /// Create a balancer from a pool's current device set.
    pub fn new(pool: &MultiGpuPool) -> Self {
        let ordinals: Vec<i32> = pool.status().iter().map(|d| d.ordinal).collect();
        Self { ordinals }
    }

    /// Assign work items to devices using round-robin distribution.
    ///
    /// Returns a vector of `(device_ordinal, item)` pairs.
    pub fn distribute_batch<T: Send>(&self, items: Vec<T>) -> Vec<(i32, T)> {
        if self.ordinals.is_empty() {
            return items.into_iter().map(|item| (-1, item)).collect();
        }
        items
            .into_iter()
            .enumerate()
            .map(|(i, item)| {
                let ord = self.ordinals[i % self.ordinals.len()];
                (ord, item)
            })
            .collect()
    }

    /// Apply a function to each item in parallel across GPUs.
    ///
    /// Items are distributed round-robin across devices, then each chunk
    /// is processed on a separate thread. Results are returned in the
    /// original item order.
    pub fn parallel_map<T, R, F>(pool: &MultiGpuPool, items: Vec<T>, f: F) -> Vec<R>
    where
        T: Send + 'static,
        R: Send + 'static,
        F: Fn(i32, T) -> R + Send + Sync + 'static,
    {
        let device_count = pool.device_count();
        if device_count == 0 || items.is_empty() {
            return Vec::new();
        }
        let ordinals: Vec<i32> = pool.status().iter().map(|d| d.ordinal).collect();
        let f = Arc::new(f);

        // Tag each item with its original index and assigned device.
        let tagged: Vec<(usize, i32, T)> = items
            .into_iter()
            .enumerate()
            .map(|(i, item)| {
                let ord = ordinals[i % ordinals.len()];
                (i, ord, item)
            })
            .collect();

        // Group by device ordinal.
        let mut buckets: Vec<Vec<(usize, i32, T)>> =
            (0..device_count).map(|_| Vec::new()).collect();
        for entry in tagged {
            let bucket_idx = ordinals.iter().position(|&o| o == entry.1).unwrap_or(0);
            buckets[bucket_idx].push(entry);
        }

        // Spawn one thread per device bucket.
        let mut handles = Vec::with_capacity(device_count);
        for bucket in buckets {
            let f = Arc::clone(&f);
            handles.push(std::thread::spawn(move || {
                bucket
                    .into_iter()
                    .map(|(idx, ord, item)| (idx, f(ord, item)))
                    .collect::<Vec<(usize, R)>>()
            }));
        }

        // Collect and re-order results.
        let mut results: Vec<(usize, R)> = Vec::new();
        for handle in handles {
            if let Ok(partial) = handle.join() {
                results.extend(partial);
            }
        }
        results.sort_by_key(|(idx, _)| *idx);
        results.into_iter().map(|(_, r)| r).collect()
    }
}

impl std::fmt::Debug for WorkloadBalancer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkloadBalancer")
            .field("device_count", &self.ordinals.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a pool with guaranteed synthetic devices, independent of
    /// real GPU hardware. Bypasses `device_status_for` so that tests exercising
    /// scheduling logic always see the same predictable device statistics.
    fn synthetic_pool(policy: DeviceSelectionPolicy) -> MultiGpuPool {
        let devices = MultiGpuPool::synthetic_devices(2);
        let inner = PoolInner {
            devices,
            policy,
            rr_cursor: 0,
        };
        MultiGpuPool {
            inner: Arc::new(RwLock::new(inner)),
            next_task_id: AtomicU64::new(1),
        }
    }

    #[test]
    fn pool_creation_with_new() {
        let pool = MultiGpuPool::new(DeviceSelectionPolicy::RoundRobin);
        assert!(pool.is_ok());
        let pool = pool.expect("pool should be created");
        // On real-GPU machines `new()` returns the actual device count (≥ 1);
        // on no-GPU / macOS systems the synthetic fallback creates 2 devices.
        assert!(pool.device_count() >= 1);
    }

    #[test]
    fn pool_creation_with_specific_devices() {
        let pool = MultiGpuPool::with_devices(vec![0, 1, 2], DeviceSelectionPolicy::LeastLoaded);
        assert!(pool.is_ok());
        let pool = pool.expect("pool should be created");
        assert_eq!(pool.device_count(), 3);
    }

    #[test]
    fn pool_empty_devices_returns_error() {
        let result = MultiGpuPool::with_devices(vec![], DeviceSelectionPolicy::RoundRobin);
        assert!(result.is_err());
    }

    #[test]
    fn round_robin_cycles_correctly() {
        let pool = synthetic_pool(DeviceSelectionPolicy::RoundRobin);
        let l0 = pool.acquire().expect("acquire 0");
        let l1 = pool.acquire().expect("acquire 1");
        let l2 = pool.acquire().expect("acquire 2");
        let l3 = pool.acquire().expect("acquire 3");
        assert_eq!(l0.ordinal(), 0);
        assert_eq!(l1.ordinal(), 1);
        assert_eq!(l2.ordinal(), 0);
        assert_eq!(l3.ordinal(), 1);
    }

    #[test]
    fn least_loaded_selects_idle_device() {
        let pool = synthetic_pool(DeviceSelectionPolicy::LeastLoaded);
        // Acquire on device 0 first (round-robin initially picks 0 for least-loaded tie).
        let lease0 = pool.acquire().expect("first acquire");
        // Now device with ordinal from lease0 has 1 task; the other should be picked.
        let lease1 = pool.acquire().expect("second acquire");
        assert_ne!(lease0.ordinal(), lease1.ordinal());
    }

    #[test]
    fn acquire_release_task_counting() {
        let pool = synthetic_pool(DeviceSelectionPolicy::RoundRobin);
        let lease = pool.acquire().expect("acquire");
        let ord = lease.ordinal();
        let status_before: Vec<_> = pool
            .status()
            .into_iter()
            .filter(|d| d.ordinal == ord)
            .collect();
        assert_eq!(status_before[0].active_tasks, 1);

        pool.release(lease);
        let status_after: Vec<_> = pool
            .status()
            .into_iter()
            .filter(|d| d.ordinal == ord)
            .collect();
        assert_eq!(status_after[0].active_tasks, 0);
    }

    #[test]
    fn gpu_lease_drop_auto_releases() {
        let pool = synthetic_pool(DeviceSelectionPolicy::RoundRobin);
        let ord;
        {
            let lease = pool.acquire().expect("acquire");
            ord = lease.ordinal();
            let tasks = pool
                .status()
                .iter()
                .find(|d| d.ordinal == ord)
                .map(|d| d.active_tasks)
                .unwrap_or(0);
            assert_eq!(tasks, 1);
        } // lease dropped here
        let tasks = pool
            .status()
            .iter()
            .find(|d| d.ordinal == ord)
            .map(|d| d.active_tasks)
            .unwrap_or(0);
        assert_eq!(tasks, 0);
    }

    #[test]
    fn device_status_reporting() {
        let pool = synthetic_pool(DeviceSelectionPolicy::RoundRobin);
        let statuses = pool.status();
        assert_eq!(statuses.len(), 2);
        for s in &statuses {
            assert!(s.is_available);
            assert!(!s.name.is_empty());
            assert!(s.total_memory > 0);
            assert!(s.sm_count > 0);
        }
    }

    #[test]
    fn workload_balancer_distribution() {
        let pool = synthetic_pool(DeviceSelectionPolicy::RoundRobin);
        let balancer = WorkloadBalancer::new(&pool);
        let items: Vec<i32> = (0..6).collect();
        let distributed = balancer.distribute_batch(items);
        assert_eq!(distributed.len(), 6);
        // Even indices -> device 0, odd indices -> device 1
        assert_eq!(distributed[0].0, 0);
        assert_eq!(distributed[1].0, 1);
        assert_eq!(distributed[2].0, 0);
        assert_eq!(distributed[3].0, 1);
    }

    #[test]
    fn policy_switching_at_runtime() {
        let pool = synthetic_pool(DeviceSelectionPolicy::RoundRobin);
        let l0 = pool.acquire().expect("rr acquire");
        assert_eq!(l0.ordinal(), 0);
        pool.release(l0);

        pool.set_policy(DeviceSelectionPolicy::MostMemoryFree)
            .expect("set_policy");
        // Device 0 has 16 GiB, device 1 has 8 GiB in synthetic pool.
        let l1 = pool.acquire().expect("most-memory acquire");
        assert_eq!(l1.ordinal(), 0); // device 0 has more memory
    }

    #[test]
    fn single_device_pool() {
        let pool = MultiGpuPool::with_devices(vec![0], DeviceSelectionPolicy::RoundRobin)
            .expect("single-device pool");
        assert_eq!(pool.device_count(), 1);
        let l0 = pool.acquire().expect("acquire 0");
        let l1 = pool.acquire().expect("acquire 1");
        assert_eq!(l0.ordinal(), 0);
        assert_eq!(l1.ordinal(), 0);
    }

    #[test]
    fn best_compute_selects_highest() {
        let pool = synthetic_pool(DeviceSelectionPolicy::BestCompute);
        // Device 1 has compute (8,1) vs device 0 with (8,0).
        let lease = pool.acquire().expect("best compute acquire");
        assert_eq!(lease.ordinal(), 1);
    }

    #[test]
    fn custom_policy_selects_correctly() {
        let policy = DeviceSelectionPolicy::Custom(Box::new(|statuses: &[DeviceStatus]| {
            // Always pick the last device.
            statuses.len().saturating_sub(1)
        }));
        let pool = synthetic_pool(policy);
        let lease = pool.acquire().expect("custom acquire");
        assert_eq!(lease.ordinal(), 1);
    }

    #[test]
    fn parallel_map_preserves_order() {
        let pool = synthetic_pool(DeviceSelectionPolicy::RoundRobin);
        let items: Vec<i32> = (0..8).collect();
        let results = WorkloadBalancer::parallel_map(&pool, items, |_device, x| x * 2);
        assert_eq!(results, vec![0, 2, 4, 6, 8, 10, 12, 14]);
    }

    #[test]
    fn acquire_with_description() {
        let pool = synthetic_pool(DeviceSelectionPolicy::RoundRobin);
        let lease = pool
            .acquire_with_description("matrix multiply".into())
            .expect("acquire with desc");
        assert_eq!(lease.description(), "matrix multiply");
        assert!(lease.task_id() > 0);
    }
}
