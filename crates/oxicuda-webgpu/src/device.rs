//! WebGPU device wrapper — owns the wgpu instance, adapter, device, and queue.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

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
    /// Whether the `SHADER_F16` feature was successfully enabled on the device.
    /// Gates the FP16 GEMM path, whose WGSL declares `enable f16;`.
    pub(crate) supports_f16: bool,
    /// Effective device limits, resolved from `adapter.limits()` and
    /// requested verbatim when the device was created (see
    /// [`WebGpuDevice::new_async`]).
    ///
    /// Requesting `wgpu::Limits::default()` (the previous behaviour) silently
    /// caps every allocation and dispatch at the WebGPU conformance
    /// *baseline* (e.g. a 256 MiB `max_buffer_size`, a 65535
    /// workgroups-per-dimension cap) even when the real adapter — Metal, on
    /// this machine — supports far more.  [`crate::memory::WebGpuMemoryManager::alloc`]
    /// validates against this field instead of the baseline.
    pub(crate) limits: wgpu::Limits,
    /// Most recent uncaptured wgpu error, if any.
    ///
    /// wgpu's default uncaptured-error handler is fatal to the process — the
    /// handler installed in [`WebGpuDevice::new_async`] records the message
    /// here instead. Drained (and cleared) by [`WebGpuDevice::poll_error`].
    last_error: Arc<Mutex<Option<String>>>,
    /// Set by the device-lost callback installed in [`WebGpuDevice::new_async`]
    /// once wgpu reports this device unusable (GPU reset, driver failure, or
    /// an external `Device::destroy()` call).
    device_lost: Arc<AtomicBool>,
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
            .map_err(|e| {
                // `RequestAdapterError` is `#[non_exhaustive]`, but every
                // variant it currently has (`NotFound`, `EnvNotSet`) means
                // the same thing to a caller: no usable adapter was
                // obtained. Preserve the detail via `tracing` and report the
                // typed, matchable `NoAdapter` variant so headless-CI skip
                // branches (see `webgpu_device_new_graceful` below) are
                // actually reachable instead of dead code.
                tracing::warn!("wgpu adapter request failed: {e}");
                WebGpuError::NoAdapter
            })?;

        let adapter_name = adapter.get_info().name.clone();

        // Enable FP16 shader support when the adapter advertises it, so the
        // `gemm_f16` path (whose WGSL declares `enable f16;`) validates instead
        // of being rejected for a missing capability.  When the adapter lacks
        // it we simply do not request it, and `gemm_f16` returns a typed
        // `Unsupported` error rather than emitting an invalid module.
        let supports_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
        let required_features = if supports_f16 {
            wgpu::Features::SHADER_F16
        } else {
            wgpu::Features::empty()
        };

        // Request the limits the adapter itself reports rather than
        // `wgpu::Limits::default()` (the WebGPU conformance *baseline* — a
        // 256 MiB `max_buffer_size`, a 65535 workgroups-per-dimension cap,
        // etc. — regardless of what the hardware can actually do). Per the
        // wgpu contract, requesting exactly `adapter.limits()` can never
        // fail where `Limits::default()` would have succeeded: only
        // requesting limits *better* than the adapter supports can fail, and
        // an adapter's own reported limits are always achievable on it.
        let adapter_limits = adapter.limits();

        // `DeviceDescriptor` does implement `Default` in wgpu-types 29 so we
        // can use struct-update syntax.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("oxicuda-webgpu"),
                required_features,
                required_limits: adapter_limits.clone(),
                memory_hints: wgpu::MemoryHints::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| WebGpuError::DeviceRequest(e.to_string()))?;

        // Install a non-fatal uncaptured-error handler. wgpu's default
        // handler panics/aborts the process on any validation, out-of-memory,
        // or internal error that is not caught by an explicit error scope —
        // the CHANGELOG records three separate past point-fixes (write_buffer
        // overrun, SHADER_F16 probe, copy_htod overrun) that were all
        // symptoms of this one missing handler. Route the message into a
        // shared slot instead, so callers can observe it via `poll_error()`
        // and return a typed `Err` rather than crashing the process.
        let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let error_slot = Arc::clone(&last_error);
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            tracing::error!("wgpu uncaptured error: {e}");
            if let Ok(mut slot) = error_slot.lock() {
                *slot = Some(e.to_string());
            }
        }));

        // Install a device-lost callback so a GPU reset, driver failure, or
        // an external `Device::destroy()` call becomes observable via
        // `is_device_lost()` instead of leaving subsequent operations to fail
        // later in confusing, un-attributable ways.
        let device_lost = Arc::new(AtomicBool::new(false));
        let lost_flag = Arc::clone(&device_lost);
        device.set_device_lost_callback(move |reason: wgpu::DeviceLostReason, message: String| {
            tracing::error!("wgpu device lost ({reason:?}): {message}");
            lost_flag.store(true, Ordering::Release);
        });

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            adapter_name,
            supports_f16,
            limits: adapter_limits,
            last_error,
            device_lost,
        })
    }

    /// Effective device limits (`max_buffer_size`,
    /// `max_storage_buffer_binding_size`,
    /// `max_compute_workgroups_per_dimension`, …), resolved from the adapter
    /// at device-creation time rather than the WebGPU conformance baseline.
    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    /// Drain and return the most recent uncaptured wgpu error recorded since
    /// the last call, if any.
    ///
    /// wgpu delivers validation, out-of-memory, and internal errors that were
    /// not caught by an explicit error scope through the non-fatal handler
    /// installed in [`WebGpuDevice::new`] rather than aborting the process.
    /// Call this immediately after an operation that might have triggered one
    /// — on the native wgpu-core backends the handler fires synchronously,
    /// before the triggering call returns — and turn `Some(_)` into a typed
    /// `Err(WebGpuError::UncapturedError(_))`.
    pub fn poll_error(&self) -> Option<String> {
        self.last_error
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
    }

    /// Returns `true` once wgpu has reported this device as lost (GPU reset,
    /// driver failure, or an external `Device::destroy()` call).
    pub fn is_device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Acquire)
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

    /// A freshly created device starts with no recorded error and is not lost.
    #[test]
    fn poll_error_and_device_lost_start_clean() {
        let Ok(dev) = WebGpuDevice::new() else {
            return; // No GPU — skip.
        };
        assert!(dev.poll_error().is_none());
        assert!(!dev.is_device_lost());
    }

    /// Resolved limits must be at least the WebGPU conformance baseline
    /// (`adapter.limits()` is defined to be >= the baseline on every
    /// conformant adapter) and must be what was actually granted to the
    /// `wgpu::Device` — i.e. `device.limits()` must not silently fall back to
    /// the baseline internally despite what we requested.
    #[test]
    fn limits_are_adapter_derived_not_baseline_default() {
        let Ok(dev) = WebGpuDevice::new() else {
            return; // No GPU — skip.
        };
        let baseline = wgpu::Limits::default();
        assert!(dev.limits().max_buffer_size >= baseline.max_buffer_size);
        assert!(
            dev.limits().max_storage_buffer_binding_size
                >= baseline.max_storage_buffer_binding_size
        );
        assert_eq!(
            dev.device.limits().max_buffer_size,
            dev.limits().max_buffer_size
        );
    }

    /// Deliberately requests an absurd buffer size directly against the raw
    /// `wgpu::Device` (bypassing every higher-level guard in
    /// `WebGpuMemoryManager`) to prove the handler installed in `new_async`
    /// catches the resulting validation error instead of letting wgpu's fatal
    /// default handler abort the process — the whole point of this finding.
    #[test]
    fn uncaptured_error_handler_is_non_fatal_and_drains() {
        let Ok(dev) = WebGpuDevice::new() else {
            return; // No GPU — skip.
        };
        assert!(dev.poll_error().is_none(), "no error recorded yet");

        let _bogus = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-webgpu-test-oversize"),
            size: u64::MAX,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        assert!(
            dev.poll_error().is_some(),
            "expected the uncaptured-error handler to record an error instead of aborting"
        );
        assert!(
            dev.poll_error().is_none(),
            "poll_error() should drain the slot"
        );
    }

    /// `Device::destroy()` must reach the `set_device_lost_callback` we
    /// install, flipping `is_device_lost()`.
    #[test]
    fn device_lost_callback_fires_on_destroy() {
        let Ok(dev) = WebGpuDevice::new() else {
            return; // No GPU — skip.
        };
        assert!(!dev.is_device_lost());

        dev.device.destroy();
        // `destroy()` only flips the device to invalid; wgpu-core actually
        // invokes the lost closure from `maintain()`, which runs
        // synchronously inside `poll()`. Retry briefly for robustness
        // against timing differences across wgpu backends/versions.
        for _ in 0..20 {
            if dev.is_device_lost() {
                break;
            }
            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            dev.is_device_lost(),
            "Device::destroy() should trigger the device-lost callback"
        );
    }
}
