//! WebGPU device wrapper — owns the wgpu instance, adapter, device, and queue.

use wgpu;

use crate::error::{WebGpuError, WebGpuResult};

/// A fully initialised WebGPU device together with its submit queue.
///
/// Created via [`WebGpuDevice::new`] which blocks the calling thread using
/// [`pollster`] until the async device request completes.
pub struct WebGpuDevice {
    /// The wgpu instance used to enumerate adapters.
    /// Kept alive to ensure the adapter and device remain valid.
    #[allow(dead_code)]
    pub(crate) instance: wgpu::Instance,
    /// The selected GPU adapter.
    /// Kept alive to ensure the device remains valid.
    #[allow(dead_code)]
    pub(crate) adapter: wgpu::Adapter,
    /// The logical device (command encoder, buffer allocator, …).
    pub(crate) device: wgpu::Device,
    /// The queue for submitting command buffers.
    pub(crate) queue: wgpu::Queue,
    /// Human-readable adapter name for diagnostics.
    pub adapter_name: String,
}

impl WebGpuDevice {
    /// Create a WebGPU device by selecting the highest-performance adapter.
    ///
    /// Blocks the calling thread until the device is ready.
    pub fn new() -> WebGpuResult<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> WebGpuResult<Self> {
        // wgpu 29: `InstanceDescriptor` does not impl `Default`; use the
        // provided constructor instead.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| WebGpuError::DeviceRequest(e.to_string()))?;

        let adapter_name = adapter.get_info().name.clone();

        // `DeviceDescriptor` does implement `Default` in wgpu-types 29 so we
        // can use struct-update syntax.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("oxicuda-webgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| WebGpuError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            adapter_name,
        })
    }
}

impl std::fmt::Debug for WebGpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WebGpuDevice({})", self.adapter_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirm that WebGpuDevice::new() does not panic — it may return Ok or Err
    /// depending on whether a GPU is available in the test environment.
    #[test]
    fn webgpu_device_new_graceful() {
        match WebGpuDevice::new() {
            Ok(dev) => {
                assert!(!dev.adapter_name.is_empty());
                // Debug impl should not panic.
                let _ = format!("{dev:?}");
            }
            Err(WebGpuError::NoAdapter) => {
                // Expected on headless CI without a GPU.
            }
            Err(e) => {
                // Any other error is also acceptable; we just must not panic.
                let _ = format!("device init error (non-fatal): {e}");
            }
        }
    }
}
