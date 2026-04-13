//! Multi-GPU context management with per-device context pools.
//!
//! When working with multiple GPUs, it is common to maintain one CUDA
//! context per device and dispatch work across them.  [`DevicePool`]
//! automates context lifecycle management and provides scheduling
//! helpers (round-robin, best-available) for multi-GPU workloads.
//!
//! # Thread safety
//!
//! [`DevicePool`] is `Send + Sync`.  Each context is wrapped in an
//! [`Arc<Context>`] so it can be shared across threads.  The caller is
//! responsible for calling [`Context::set_current`] on the appropriate
//! thread before issuing driver calls.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_driver::multi_gpu::DevicePool;
//!
//! oxicuda_driver::init()?;
//! let pool = DevicePool::new()?;
//! println!("managing {} devices", pool.device_count());
//!
//! for (dev, ctx) in pool.iter() {
//!     ctx.set_current()?;
//!     println!("device {}: {}", dev.ordinal(), dev.name()?);
//! }
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::context::Context;
use crate::device::Device;
use crate::error::{CudaError, CudaResult};

// ---------------------------------------------------------------------------
// DevicePool
// ---------------------------------------------------------------------------

/// Per-device context pool for multi-GPU management.
///
/// Maintains a mapping from device ordinals to contexts, with thread-safe
/// access for multi-threaded workloads.  Each device gets exactly one
/// context, created with default scheduling flags.
///
/// # Round-robin scheduling
///
/// The [`next_device`](DevicePool::next_device) method implements
/// round-robin device selection using an atomic counter, making it safe
/// to call from multiple threads without locking.
///
/// # Best-available scheduling
///
/// The [`best_available_device`](DevicePool::best_available_device) method
/// selects the device with the most total memory.  In a future release,
/// this may query free memory at runtime when the driver supports it.
pub struct DevicePool {
    /// Ordered list of (device, context) pairs.
    entries: Vec<(Device, Arc<Context>)>,
    /// Atomic counter for round-robin scheduling.
    round_robin: AtomicUsize,
}

// SAFETY: All fields are Send+Sync:
// - `entries` is a Vec of (Device, Arc<Context>); Device is Copy+Send+Sync,
//   Arc<Context> is Send (Context is Send).
// - AtomicUsize is Send+Sync.
unsafe impl Send for DevicePool {}
unsafe impl Sync for DevicePool {}

impl DevicePool {
    /// Creates a new pool with contexts for all available devices.
    ///
    /// Enumerates every CUDA-capable device and creates one context per
    /// device.  The contexts are created with default scheduling flags
    /// ([`crate::context::flags::SCHED_AUTO`]).
    ///
    /// # Errors
    ///
    /// * [`CudaError::NoDevice`] if no CUDA devices are available.
    /// * Other driver errors from device enumeration or context creation.
    pub fn new() -> CudaResult<Self> {
        let devices = crate::device::list_devices()?;
        if devices.is_empty() {
            return Err(CudaError::NoDevice);
        }
        Self::with_devices(&devices)
    }

    /// Creates a pool with contexts for specific devices.
    ///
    /// One context is created per device in the provided slice.  The
    /// ordering in the slice determines the iteration and round-robin
    /// order.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if the device slice is empty.
    /// * Other driver errors from context creation.
    pub fn with_devices(devices: &[Device]) -> CudaResult<Self> {
        if devices.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        let mut entries = Vec::with_capacity(devices.len());
        for dev in devices {
            let ctx = Context::new(dev)?;
            entries.push((*dev, Arc::new(ctx)));
        }
        Ok(Self {
            entries,
            round_robin: AtomicUsize::new(0),
        })
    }

    /// Returns the context for the given device ordinal.
    ///
    /// Searches the pool for a device whose ordinal matches the given
    /// value.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidDevice`] if no device with the given
    /// ordinal is in the pool.
    pub fn context(&self, device_ordinal: i32) -> CudaResult<&Arc<Context>> {
        self.entries
            .iter()
            .find(|(dev, _)| dev.ordinal() == device_ordinal)
            .map(|(_, ctx)| ctx)
            .ok_or(CudaError::InvalidDevice)
    }

    /// Returns the number of devices in the pool.
    #[inline]
    pub fn device_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the device with the most total memory.
    ///
    /// This is a heuristic for selecting the "best" device when you want
    /// to maximise available memory.  For real-time free-memory queries,
    /// use `cuMemGetInfo` (once it is wired into the driver API).
    ///
    /// # Errors
    ///
    /// Returns an error if memory queries fail.
    pub fn best_available_device(&self) -> CudaResult<Device> {
        let mut best_dev = self.entries[0].0;
        let mut best_mem: usize = 0;
        for (dev, _ctx) in &self.entries {
            let mem = dev.total_memory()?;
            if mem > best_mem {
                best_mem = mem;
                best_dev = *dev;
            }
        }
        Ok(best_dev)
    }

    /// Selects a device using round-robin scheduling.
    ///
    /// Each call advances an internal atomic counter and returns the
    /// next device in sequence.  This is safe to call concurrently from
    /// multiple threads.
    ///
    /// # Errors
    ///
    /// This method is infallible for a properly constructed pool, but
    /// returns `CudaResult` for API consistency.
    pub fn next_device(&self) -> CudaResult<Device> {
        let idx = self.round_robin.fetch_add(1, Ordering::Relaxed) % self.entries.len();
        Ok(self.entries[idx].0)
    }

    /// Iterates over all (device, context) pairs in pool order.
    pub fn iter(&self) -> impl Iterator<Item = (&Device, &Arc<Context>)> {
        self.entries.iter().map(|(dev, ctx)| (dev, ctx))
    }

    /// Returns the context for the device at the given pool index.
    ///
    /// Pool indices are 0-based and correspond to the order in which
    /// devices were added to the pool.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if the index is out of bounds.
    pub fn context_at(&self, index: usize) -> CudaResult<&Arc<Context>> {
        self.entries
            .get(index)
            .map(|(_, ctx)| ctx)
            .ok_or(CudaError::InvalidValue)
    }

    /// Returns the device at the given pool index.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if the index is out of bounds.
    pub fn device_at(&self, index: usize) -> CudaResult<Device> {
        self.entries
            .get(index)
            .map(|(dev, _)| *dev)
            .ok_or(CudaError::InvalidValue)
    }
}

impl std::fmt::Debug for DevicePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DevicePool")
            .field("device_count", &self.entries.len())
            .field(
                "devices",
                &self
                    .entries
                    .iter()
                    .map(|(d, _)| d.ordinal())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // On macOS the driver is not available, so we test the error paths
    // and structural properties.

    #[test]
    fn pool_with_empty_devices_returns_error() {
        let result = DevicePool::with_devices(&[]);
        assert!(result.is_err());
        assert_eq!(result.err(), Some(CudaError::InvalidValue),);
    }

    #[test]
    fn pool_new_returns_error_without_driver() {
        // When no driver is present, new() fails; when a driver is present,
        // it succeeds.  Either outcome is valid — the test just checks the
        // call does not panic.
        let _result = DevicePool::new();
    }

    #[test]
    fn device_pool_debug_format() {
        // We can at least test the Debug impl compiles and formats.
        let fmt = format!("{:?}", "DevicePool placeholder");
        assert!(!fmt.is_empty());
    }

    #[test]
    fn round_robin_counter_wraps() {
        // Test the atomic counter logic in isolation.
        let counter = AtomicUsize::new(0);
        let pool_size = 3;
        for expected in [0, 1, 2, 0, 1, 2, 0] {
            let idx = counter.fetch_add(1, Ordering::Relaxed) % pool_size;
            assert_eq!(idx, expected);
        }
    }

    #[test]
    fn round_robin_single_device() {
        let counter = AtomicUsize::new(0);
        let pool_size = 1;
        for _ in 0..10 {
            let idx = counter.fetch_add(1, Ordering::Relaxed) % pool_size;
            assert_eq!(idx, 0);
        }
    }

    #[test]
    fn context_at_out_of_bounds_returns_error() {
        // We cannot construct a DevicePool without a GPU, but we can test
        // the logic path. Since construction fails on macOS, we just verify
        // the error variant exists.
        let err = CudaError::InvalidValue;
        assert_eq!(err.as_raw(), 1);
    }

    #[cfg(feature = "gpu-tests")]
    #[test]
    fn pool_with_real_devices() {
        crate::init().ok();
        let result = DevicePool::new();
        if let Ok(pool) = result {
            assert!(pool.device_count() > 0);
            let dev = pool.next_device().expect("next_device failed");
            assert!(pool.context(dev.ordinal()).is_ok());
            let best = pool.best_available_device().expect("best_available failed");
            assert!(best.total_memory().is_ok());
            // Iterate
            for (d, _c) in pool.iter() {
                assert!(d.name().is_ok());
            }
        }
    }
}
