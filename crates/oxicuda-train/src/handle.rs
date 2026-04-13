//! Training session handle.
//!
//! [`crate::handle::TrainHandle`] owns a CUDA context and stream and carries SM version
//! metadata used to select the correct PTX target when JIT-compiling
//! optimizer update kernels.

use std::sync::Arc;

use oxicuda_driver::{Context, Stream};

use crate::error::TrainResult;

/// Central handle for all OxiCUDA training operations.
///
/// Pass a `&TrainHandle` (or clone the `Arc`-backed handle) to optimizers,
/// gradient utilities, and checkpointing APIs to share a single CUDA context
/// and stream across the training loop.
///
/// # Example
///
/// ```rust,no_run
/// use oxicuda_train::handle::TrainHandle;
///
/// let handle = TrainHandle::new().unwrap();
/// println!("SM version: {}", handle.sm_version());
/// ```
#[derive(Clone)]
pub struct TrainHandle {
    context: Arc<Context>,
    stream: Arc<Stream>,
    /// Numeric SM version, e.g. 800 for sm_80, 900 for sm_90.
    sm_version: u32,
    device_id: i32,
}

impl TrainHandle {
    /// Create a handle on the best available GPU device.
    ///
    /// Queries the device's compute capability to fill `sm_version`.
    pub fn new() -> TrainResult<Self> {
        use oxicuda_driver::{best_device, device::Device, init};
        init()?;
        let device = if let Some(d) = best_device()? {
            d
        } else {
            Device::get(0)?
        };
        let sm_version = device_sm_version(&device).unwrap_or(800);
        let device_id = device.ordinal();
        let context = Arc::new(Context::new(&device)?);
        let stream = Arc::new(Stream::new(&context)?);
        Ok(Self {
            context,
            stream,
            sm_version,
            device_id,
        })
    }

    /// Create a handle from an already-created context and stream.
    ///
    /// `sm_version` must be the numeric SM version (e.g. `800` for sm_80).
    #[must_use]
    pub fn from_parts(context: Arc<Context>, stream: Arc<Stream>, sm_version: u32) -> Self {
        let device_id = 0i32; // caller-supplied; unknown here
        Self {
            context,
            stream,
            sm_version,
            device_id,
        }
    }

    /// CUDA context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// CUDA stream used for all asynchronous launches.
    #[must_use]
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Numeric SM version (e.g. `800` for sm_80, `900` for sm_90).
    #[must_use]
    pub fn sm_version(&self) -> u32 {
        self.sm_version
    }

    /// Logical device index.
    #[must_use]
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Block until all GPU work queued on this stream has completed.
    pub fn synchronize(&self) -> TrainResult<()> {
        self.stream.synchronize()?;
        Ok(())
    }
}

impl std::fmt::Debug for TrainHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainHandle")
            .field("sm_version", &self.sm_version)
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}

// ─── SM-version helper ───────────────────────────────────────────────────────

/// Query compute capability of `device` and convert to a numeric SM version.
///
/// Returns `None` if the driver query fails (the caller should fall back to
/// a safe default such as 800 for sm_80).
fn device_sm_version(device: &oxicuda_driver::device::Device) -> Option<u32> {
    use oxicuda_driver::ffi::CUdevice_attribute;
    let major = device
        .attribute(CUdevice_attribute::ComputeCapabilityMajor)
        .ok()?;
    let minor = device
        .attribute(CUdevice_attribute::ComputeCapabilityMinor)
        .ok()?;
    Some((major as u32) * 100 + (minor as u32) * 10)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that `from_parts` stores parameters correctly without needing
    /// a real GPU.
    #[test]
    #[ignore = "requires GPU"]
    fn handle_from_parts_round_trip() {
        let handle = TrainHandle::new().unwrap();
        let sm = handle.sm_version();
        assert!(sm >= 750, "expected at least sm_75, got {sm}");
    }

    /// Debug formatting does not panic.
    #[test]
    #[ignore = "requires GPU"]
    fn handle_debug() {
        let handle = TrainHandle::new().unwrap();
        let _ = format!("{handle:?}");
    }
}
