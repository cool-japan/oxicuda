//! Global initialization with device auto-selection.
//!
//! Provides a one-call initialization path that selects a GPU, creates a
//! context and (optionally) a default stream, then stores them in a
//! process-wide singleton accessible from any thread.
//!
//! # Quick Start
//!
//! ```no_run
//! use oxicuda::prelude::*;
//!
//! // Lazily initialize with defaults (best compute capability)
//! let rt = lazy_init()?;
//! println!("Using GPU: {}", rt.device_name());
//! # Ok::<(), oxicuda_driver::CudaError>(())
//! ```
//!
//! # Custom Configuration
//!
//! ```no_run
//! use oxicuda::global_init::*;
//!
//! let rt = OxiCudaRuntimeBuilder::new()
//!     .device(DeviceSelection::BestMemory)
//!     .with_stream(true)
//!     .build()?;
//! # Ok::<(), oxicuda_driver::CudaError>(())
//! ```

use std::fmt;
use std::sync::{Arc, OnceLock};

use oxicuda_driver::{Context, CudaError, CudaResult, Device, Stream};

// ─── DeviceSelection ────────────────────────────────────────

/// Strategy for selecting a GPU device during initialization.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum DeviceSelection {
    /// Pick the device with the highest compute capability (SM version).
    /// Ties are broken by total memory.  This is the default.
    #[default]
    BestComputeCapability,
    /// Pick the device with the most VRAM.
    /// Ties are broken by compute capability.
    BestMemory,
    /// Use the device with the given ordinal.
    Specific(i32),
    /// Always use device 0.
    First,
}

impl DeviceSelection {
    /// Apply the selection strategy and return the chosen [`Device`].
    ///
    /// Returns `Err(CudaError::NoDevice)` when the system has no CUDA devices
    /// and `Err(CudaError::InvalidDevice)` for an out-of-range ordinal in
    /// [`Specific`](DeviceSelection::Specific).
    pub fn select(&self) -> CudaResult<Device> {
        match self {
            Self::First => Device::get(0),
            Self::Specific(ordinal) => {
                let count = Device::count()?;
                if *ordinal < 0 || *ordinal >= count {
                    return Err(CudaError::InvalidDevice);
                }
                Device::get(*ordinal)
            }
            Self::BestComputeCapability => {
                let devices = oxicuda_driver::list_devices()?;
                if devices.is_empty() {
                    return Err(CudaError::NoDevice);
                }
                select_best_compute_capability(&devices)
            }
            Self::BestMemory => {
                let devices = oxicuda_driver::list_devices()?;
                if devices.is_empty() {
                    return Err(CudaError::NoDevice);
                }
                select_best_memory(&devices)
            }
        }
    }
}

/// Pick device with highest compute capability, tie-breaking on memory.
fn select_best_compute_capability(devices: &[Device]) -> CudaResult<Device> {
    let mut best = devices[0];
    let mut best_cc = best.compute_capability()?;
    let mut best_mem = best.total_memory()?;

    for dev in devices.iter().skip(1) {
        let cc = dev.compute_capability()?;
        let mem = dev.total_memory()?;
        if cc > best_cc || (cc == best_cc && mem > best_mem) {
            best = *dev;
            best_cc = cc;
            best_mem = mem;
        }
    }
    Ok(best)
}

/// Pick device with most VRAM, tie-breaking on compute capability.
fn select_best_memory(devices: &[Device]) -> CudaResult<Device> {
    let mut best = devices[0];
    let mut best_mem = best.total_memory()?;
    let mut best_cc = best.compute_capability()?;

    for dev in devices.iter().skip(1) {
        let mem = dev.total_memory()?;
        let cc = dev.compute_capability()?;
        if mem > best_mem || (mem == best_mem && cc > best_cc) {
            best = *dev;
            best_mem = mem;
            best_cc = cc;
        }
    }
    Ok(best)
}

// ─── RuntimeConfig ──────────────────────────────────────────

/// Configuration for the global OxiCUDA runtime.
#[derive(Debug, Clone)]
pub struct OxiCudaRuntimeConfig {
    /// How to pick the device (default: [`DeviceSelection::BestComputeCapability`]).
    pub device_selection: DeviceSelection,
    /// Whether to create a default stream (default: `true`).
    pub create_default_stream: bool,
}

impl Default for OxiCudaRuntimeConfig {
    fn default() -> Self {
        Self {
            device_selection: DeviceSelection::default(),
            create_default_stream: true,
        }
    }
}

// ─── RuntimeInfo ────────────────────────────────────────────

/// Summary information about the initialized runtime.
#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    /// Human-readable device name.
    pub device_name: String,
    /// Compute capability `(major, minor)`.
    pub compute_capability: (i32, i32),
    /// Zero-based device ordinal.
    pub device_ordinal: i32,
}

impl fmt::Display for RuntimeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OxiCUDA runtime: {} (SM {}.{}, ordinal {})",
            self.device_name,
            self.compute_capability.0,
            self.compute_capability.1,
            self.device_ordinal,
        )
    }
}

// ─── OxiCudaRuntime ─────────────────────────────────────────

/// Process-wide CUDA runtime singleton.
///
/// Holds the selected device, its primary context, and an optional default
/// stream.  Created via [`lazy_init`], [`init_with`], or
/// [`OxiCudaRuntimeBuilder`].
pub struct OxiCudaRuntime {
    device: Device,
    context: Arc<Context>,
    default_stream: Option<Stream>,
    device_name: String,
    compute_capability: (i32, i32),
}

// Manual Debug because Stream may not impl Debug.
impl fmt::Debug for OxiCudaRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OxiCudaRuntime")
            .field("device_name", &self.device_name)
            .field("compute_capability", &self.compute_capability)
            .field("has_default_stream", &self.default_stream.is_some())
            .finish()
    }
}

impl OxiCudaRuntime {
    /// Build a new runtime from the given config.
    fn new(config: OxiCudaRuntimeConfig) -> CudaResult<Self> {
        // Ensure the CUDA driver is initialized.
        oxicuda_driver::init()?;

        let device = config.device_selection.select()?;
        let device_name = device.name()?;
        let compute_capability = device.compute_capability()?;
        let context = Arc::new(Context::new(&device)?);
        let default_stream = if config.create_default_stream {
            Some(Stream::new(&context)?)
        } else {
            None
        };

        Ok(Self {
            device,
            context,
            default_stream,
            device_name,
            compute_capability,
        })
    }

    // ── Accessors ───────────────────────────────────────────

    /// The selected GPU device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Thread-safe handle to the primary context.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// The default stream, if one was created.
    pub fn stream(&self) -> Option<&Stream> {
        self.default_stream.as_ref()
    }

    /// Human-readable device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Compute capability `(major, minor)`.
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// Build a [`RuntimeInfo`] snapshot.
    pub fn info(&self) -> RuntimeInfo {
        RuntimeInfo {
            device_name: self.device_name.clone(),
            compute_capability: self.compute_capability,
            device_ordinal: self.device.ordinal(),
        }
    }
}

// ─── Global singleton ───────────────────────────────────────

/// The process-wide runtime instance, initialized at most once.
static RUNTIME: OnceLock<Result<OxiCudaRuntime, CudaError>> = OnceLock::new();

/// Initialize with the default configuration on the first call and return a
/// reference to the global runtime.
///
/// Subsequent calls return the same instance (even if the first call failed).
pub fn lazy_init() -> CudaResult<&'static OxiCudaRuntime> {
    let result = RUNTIME.get_or_init(|| OxiCudaRuntime::new(OxiCudaRuntimeConfig::default()));
    match result {
        Ok(rt) => Ok(rt),
        Err(e) => Err(*e),
    }
}

/// Initialize with a custom configuration.
///
/// If the runtime is already initialized (by a prior call to [`lazy_init`] or
/// [`init_with`]) the existing instance is returned and `config` is ignored.
pub fn init_with(config: OxiCudaRuntimeConfig) -> CudaResult<&'static OxiCudaRuntime> {
    let result = RUNTIME.get_or_init(|| OxiCudaRuntime::new(config));
    match result {
        Ok(rt) => Ok(rt),
        Err(e) => Err(*e),
    }
}

/// Returns `true` if the global runtime has been initialized (successfully or
/// not).
pub fn is_initialized() -> bool {
    RUNTIME.get().is_some()
}

/// Convenience: get a reference to the default device.
///
/// Calls [`lazy_init`] if the runtime has not been initialized yet.
pub fn default_device() -> CudaResult<&'static Device> {
    lazy_init().map(|rt| rt.device())
}

/// Convenience: get an `Arc<Context>` for the default device.
///
/// Calls [`lazy_init`] if the runtime has not been initialized yet.
pub fn default_context() -> CudaResult<&'static Arc<Context>> {
    lazy_init().map(|rt| rt.context())
}

/// Convenience: get the default stream.
///
/// Returns `Err(CudaError::NotInitialized)` if the runtime was created
/// without a default stream.
///
/// Calls [`lazy_init`] if the runtime has not been initialized yet.
pub fn default_stream() -> CudaResult<&'static Stream> {
    let rt = lazy_init()?;
    rt.stream().ok_or(CudaError::NotInitialized)
}

/// Convenience: get a [`RuntimeInfo`] summary.
///
/// Calls [`lazy_init`] if the runtime has not been initialized yet.
pub fn runtime_info() -> CudaResult<RuntimeInfo> {
    lazy_init().map(|rt| rt.info())
}

// ─── Builder ────────────────────────────────────────────────

/// Fluent builder for configuring and initializing the global runtime.
///
/// ```no_run
/// use oxicuda::global_init::*;
///
/// let rt = OxiCudaRuntimeBuilder::new()
///     .device(DeviceSelection::Specific(1))
///     .with_stream(false)
///     .build()?;
/// # Ok::<(), oxicuda_driver::CudaError>(())
/// ```
#[derive(Debug, Clone)]
pub struct OxiCudaRuntimeBuilder {
    config: OxiCudaRuntimeConfig,
}

impl Default for OxiCudaRuntimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OxiCudaRuntimeBuilder {
    /// Create a builder with default settings.
    pub fn new() -> Self {
        Self {
            config: OxiCudaRuntimeConfig::default(),
        }
    }

    /// Set the device selection strategy.
    pub fn device(mut self, selection: DeviceSelection) -> Self {
        self.config.device_selection = selection;
        self
    }

    /// Enable or disable creation of a default stream.
    pub fn with_stream(mut self, enabled: bool) -> Self {
        self.config.create_default_stream = enabled;
        self
    }

    /// Consume the builder and initialize the global runtime.
    ///
    /// If the runtime is already initialized the existing instance is returned
    /// and this builder's configuration is ignored.
    pub fn build(self) -> CudaResult<&'static OxiCudaRuntime> {
        init_with(self.config)
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DeviceSelection unit tests ─────────────────────────

    #[test]
    fn device_selection_default_is_best_compute_capability() {
        assert_eq!(
            DeviceSelection::default(),
            DeviceSelection::BestComputeCapability
        );
    }

    #[test]
    fn device_selection_eq() {
        assert_eq!(DeviceSelection::First, DeviceSelection::First);
        assert_eq!(DeviceSelection::BestMemory, DeviceSelection::BestMemory);
        assert_eq!(DeviceSelection::Specific(0), DeviceSelection::Specific(0));
        assert_ne!(DeviceSelection::Specific(0), DeviceSelection::Specific(1));
        assert_ne!(DeviceSelection::First, DeviceSelection::BestMemory);
    }

    #[test]
    fn device_selection_debug() {
        let s = format!("{:?}", DeviceSelection::BestComputeCapability);
        assert!(s.contains("BestComputeCapability"));
    }

    #[test]
    fn device_selection_clone() {
        let a = DeviceSelection::Specific(42);
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── OxiCudaRuntimeConfig tests ─────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = OxiCudaRuntimeConfig::default();
        assert_eq!(cfg.device_selection, DeviceSelection::BestComputeCapability);
        assert!(cfg.create_default_stream);
    }

    #[test]
    fn config_custom_values() {
        let cfg = OxiCudaRuntimeConfig {
            device_selection: DeviceSelection::BestMemory,
            create_default_stream: false,
        };
        assert_eq!(cfg.device_selection, DeviceSelection::BestMemory);
        assert!(!cfg.create_default_stream);
    }

    #[test]
    fn config_debug() {
        let cfg = OxiCudaRuntimeConfig::default();
        let s = format!("{:?}", cfg);
        assert!(s.contains("BestComputeCapability"));
        assert!(s.contains("true"));
    }

    // ── RuntimeInfo tests ──────────────────────────────────

    #[test]
    fn runtime_info_display() {
        let info = RuntimeInfo {
            device_name: "Test GPU".to_string(),
            compute_capability: (8, 6),
            device_ordinal: 0,
        };
        let s = format!("{info}");
        assert!(s.contains("Test GPU"));
        assert!(s.contains("SM 8.6"));
        assert!(s.contains("ordinal 0"));
    }

    #[test]
    fn runtime_info_debug() {
        let info = RuntimeInfo {
            device_name: "RTX 4090".to_string(),
            compute_capability: (9, 0),
            device_ordinal: 1,
        };
        let s = format!("{info:?}");
        assert!(s.contains("RTX 4090"));
        assert!(s.contains("(9, 0)"));
    }

    #[test]
    fn runtime_info_clone() {
        let info = RuntimeInfo {
            device_name: "A100".to_string(),
            compute_capability: (8, 0),
            device_ordinal: 0,
        };
        let cloned = info.clone();
        assert_eq!(cloned.device_name, "A100");
        assert_eq!(cloned.compute_capability, (8, 0));
        assert_eq!(cloned.device_ordinal, 0);
    }

    // ── Builder tests ──────────────────────────────────────

    #[test]
    fn builder_default_config() {
        let builder = OxiCudaRuntimeBuilder::new();
        assert_eq!(
            builder.config.device_selection,
            DeviceSelection::BestComputeCapability
        );
        assert!(builder.config.create_default_stream);
    }

    #[test]
    fn builder_chained() {
        let builder = OxiCudaRuntimeBuilder::new()
            .device(DeviceSelection::Specific(2))
            .with_stream(false);
        assert_eq!(
            builder.config.device_selection,
            DeviceSelection::Specific(2)
        );
        assert!(!builder.config.create_default_stream);
    }

    #[test]
    fn builder_default_trait() {
        let builder = OxiCudaRuntimeBuilder::default();
        assert_eq!(
            builder.config.device_selection,
            DeviceSelection::BestComputeCapability
        );
    }

    #[test]
    fn builder_clone() {
        let a = OxiCudaRuntimeBuilder::new().device(DeviceSelection::BestMemory);
        let b = a.clone();
        assert_eq!(b.config.device_selection, DeviceSelection::BestMemory);
    }

    #[test]
    fn builder_debug() {
        let builder = OxiCudaRuntimeBuilder::new();
        let s = format!("{builder:?}");
        assert!(s.contains("OxiCudaRuntimeBuilder"));
    }

    // ── is_initialized (static state) ──────────────────────

    // Note: Because OnceLock is process-wide and tests run in the same
    // process, we can only reliably test the *type* of `is_initialized`.
    // On macOS (no GPU) the init will fail, but the lock will still be set.
    #[test]
    fn is_initialized_returns_bool() {
        // Just verify the function compiles and returns a bool.
        let _val: bool = is_initialized();
    }

    // ── select logic (negative ordinal) ────────────────────

    #[test]
    fn specific_negative_ordinal_without_gpu() {
        // Without a GPU, Device::count() itself will fail, which is fine.
        // We just verify that the code path doesn't panic.
        let sel = DeviceSelection::Specific(-1);
        let _result = sel.select(); // may be Err on any platform
    }

    // ── GPU-gated tests ────────────────────────────────────

    #[cfg(feature = "gpu-tests")]
    mod gpu {
        use super::super::*;

        /// Returns true if a GPU is actually available on this system.
        fn gpu_available() -> bool {
            oxicuda_driver::init().is_ok() && Device::count().unwrap_or(0) > 0
        }

        #[test]
        fn select_first_device() {
            if !gpu_available() {
                return; // skip on macOS / systems without NVIDIA GPU
            }
            let dev = DeviceSelection::First.select();
            assert!(dev.is_ok());
        }

        #[test]
        fn select_specific_zero() {
            if !gpu_available() {
                return;
            }
            let dev = DeviceSelection::Specific(0).select();
            assert!(dev.is_ok());
        }

        #[test]
        fn select_best_cc() {
            if !gpu_available() {
                return;
            }
            let dev = DeviceSelection::BestComputeCapability.select();
            assert!(dev.is_ok());
        }

        #[test]
        fn select_best_memory() {
            if !gpu_available() {
                return;
            }
            let dev = DeviceSelection::BestMemory.select();
            assert!(dev.is_ok());
        }

        #[test]
        fn specific_out_of_range() {
            if !gpu_available() {
                return;
            }
            let count = Device::count().unwrap_or(0);
            let dev = DeviceSelection::Specific(count + 10).select();
            assert!(dev.is_err());
        }
    }
}
