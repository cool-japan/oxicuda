//! Metal device wrapper — owns the `metal::Device` handle and exposes a safe
//! Rust API. Only compiled on macOS; on all other platforms every constructor
//! returns [`MetalError::UnsupportedPlatform`].

use crate::device_family::MetalDeviceCapabilities;
use crate::error::{MetalError, MetalResult};

// ─── MetalDevice ─────────────────────────────────────────────────────────────

/// A Metal GPU device.
///
/// On non-macOS platforms, [`MetalDevice::new`] always returns
/// [`MetalError::UnsupportedPlatform`].
///
/// The device owns the **single** `MTLCommandQueue` shared by every compute
/// pipeline created against it (crate-internal `command_queue()` accessor).
/// Apple recommends one to three queues per device: a queue carries kernel-side
/// scheduling state, and — more importantly — command buffers submitted to
/// *different* queues have no ordering guarantee, so a per-pipeline queue would
/// make cross-op batching impossible.
pub struct MetalDevice {
    /// The underlying `metal::Device` — only present on macOS.
    #[cfg(target_os = "macos")]
    pub(crate) device: metal::Device,
    /// The one command queue shared by all pipelines on this device.
    #[cfg(target_os = "macos")]
    command_queue: metal::CommandQueue,
    /// Human-readable device name (e.g. `"Apple M3"`).
    name: String,
    /// Maximum single buffer length in bytes.
    max_buffer_length: u64,
    /// Resolved hardware capability snapshot (GPU family, threadgroup budgets).
    capabilities: MetalDeviceCapabilities,
}

impl MetalDevice {
    /// Acquire the system-default Metal device.
    ///
    /// Returns [`MetalError::NoDevice`] when no Metal-capable GPU is present,
    /// and [`MetalError::UnsupportedPlatform`] on non-macOS targets.
    pub fn new() -> MetalResult<Self> {
        #[cfg(target_os = "macos")]
        {
            let device = metal::Device::system_default().ok_or(MetalError::NoDevice)?;
            let name = device.name().to_string();
            let max_buffer_length = device.max_buffer_length();
            let capabilities = Self::probe_capabilities(&device, &name);
            // One queue per device, retained by every pipeline built from it.
            let command_queue = device.new_command_queue();
            tracing::info!(
                "Metal device selected: {name} (family {}, max threads/threadgroup {})",
                capabilities.family,
                capabilities.max_threads_per_threadgroup
            );
            Ok(Self {
                device,
                command_queue,
                name,
                max_buffer_length,
                capabilities,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// Resolve the GPU family and threadgroup budgets from the live device,
    /// falling back to the device-name heuristic when `supportsFamily:` reports
    /// nothing we model.
    #[cfg(target_os = "macos")]
    fn probe_capabilities(device: &metal::DeviceRef, name: &str) -> MetalDeviceCapabilities {
        use crate::device_family::MetalGpuFamily;
        use metal::MTLGPUFamily;
        // `supportsFamily:` is monotone along the Apple line (a device that
        // supports Apple9 also supports Apple7), so probe newest first.
        const APPLE_FAMILIES: [(MTLGPUFamily, MetalGpuFamily); 6] = [
            (MTLGPUFamily::Apple9, MetalGpuFamily::Apple9),
            (MTLGPUFamily::Apple8, MetalGpuFamily::Apple8),
            (MTLGPUFamily::Apple7, MetalGpuFamily::Apple7),
            (MTLGPUFamily::Apple6, MetalGpuFamily::Apple6),
            (MTLGPUFamily::Apple5, MetalGpuFamily::Apple5),
            (MTLGPUFamily::Apple4, MetalGpuFamily::Apple4),
        ];
        let mut family = None;
        for (probe, mapped) in APPLE_FAMILIES {
            if device.supports_family(probe) {
                family = Some(mapped);
                break;
            }
        }
        if family.is_none() && device.supports_family(MTLGPUFamily::Mac2) {
            family = Some(MetalGpuFamily::Mac2);
        }
        // The family table supplies documented minimums; refine them with the
        // values the driver actually reports for this device.
        let mut caps = match family {
            Some(f) => MetalDeviceCapabilities::from_family(f),
            None => MetalDeviceCapabilities::from_device_name(name),
        };
        let max_threads = device.max_threads_per_threadgroup().width as usize;
        if max_threads > 0 {
            caps.max_threads_per_threadgroup = max_threads;
        }
        let threadgroup_memory = device.max_threadgroup_memory_length() as usize;
        if threadgroup_memory > 0 {
            caps.threadgroup_memory = threadgroup_memory;
        }
        caps.unified_memory = device.has_unified_memory();
        caps
    }

    /// Human-readable device name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Maximum single Metal buffer length in bytes.
    pub fn max_buffer_length(&self) -> u64 {
        self.max_buffer_length
    }

    /// Resolved capability snapshot for this device (GPU family,
    /// `simdgroup_matrix` support, threadgroup budgets, unified memory).
    ///
    /// On macOS this is probed from `[MTLDevice supportsFamily:]` and the
    /// driver-reported threadgroup limits rather than guessed from the device
    /// name.
    pub fn capabilities(&self) -> MetalDeviceCapabilities {
        self.capabilities
    }

    /// The single `MTLCommandQueue` shared by every pipeline on this device.
    ///
    /// Sharing one queue keeps submission order well defined across ops (a
    /// prerequisite for batching several dispatches into one command buffer)
    /// and avoids allocating a heavyweight queue per compiled kernel variant.
    #[cfg(target_os = "macos")]
    pub(crate) fn command_queue(&self) -> &metal::CommandQueueRef {
        &self.command_queue
    }
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({})", self.name)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_device_new_graceful() {
        match MetalDevice::new() {
            Ok(dev) => {
                assert!(!dev.name().is_empty());
                assert!(dev.max_buffer_length() > 0);
                let dbg = format!("{dev:?}");
                assert!(dbg.contains("MetalDevice"));
            }
            Err(MetalError::NoDevice) => {
                // Acceptable on CI without a GPU.
            }
            Err(e) => {
                // Any other error is also acceptable — just must not panic.
                let _ = format!("Metal device init error (non-fatal): {e}");
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn capabilities_probed_from_device() {
        let Ok(dev) = MetalDevice::new() else {
            return;
        };
        let caps = dev.capabilities();
        // The driver always reports a positive threadgroup budget; the family
        // table's documented minimum is 512 threads / 16 KiB.
        assert!(caps.max_threads_per_threadgroup >= 512);
        assert!(caps.threadgroup_memory >= 16 * 1024);
        assert_eq!(
            caps.simdgroup_matrix,
            caps.family.supports_simdgroup_matrix()
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn command_queue_is_stable_per_device() {
        let Ok(dev) = MetalDevice::new() else {
            return;
        };
        // Repeated accessor calls must hand out the *same* queue object, not a
        // freshly created one.
        let first: *const metal::CommandQueueRef = dev.command_queue();
        let second: *const metal::CommandQueueRef = dev.command_queue();
        assert!(std::ptr::eq(first, second));
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn metal_device_unsupported_on_non_macos() {
        let result = MetalDevice::new();
        assert!(matches!(result, Err(MetalError::UnsupportedPlatform)));
    }
}
