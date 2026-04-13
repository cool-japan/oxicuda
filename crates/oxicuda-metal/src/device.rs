//! Metal device wrapper — owns the `metal::Device` handle and exposes a safe
//! Rust API. Only compiled on macOS; on all other platforms every constructor
//! returns [`MetalError::UnsupportedPlatform`].

use crate::error::{MetalError, MetalResult};

// ─── MetalDevice ─────────────────────────────────────────────────────────────

/// A Metal GPU device.
///
/// On non-macOS platforms, [`MetalDevice::new`] always returns
/// [`MetalError::UnsupportedPlatform`].
pub struct MetalDevice {
    /// The underlying `metal::Device` — only present on macOS.
    #[cfg(target_os = "macos")]
    pub(crate) device: metal::Device,
    /// Human-readable device name (e.g. `"Apple M3"`).
    name: String,
    /// Maximum single buffer length in bytes.
    max_buffer_length: u64,
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
            tracing::info!("Metal device selected: {name}");
            Ok(Self {
                device,
                name,
                max_buffer_length,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// Human-readable device name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Maximum single Metal buffer length in bytes.
    pub fn max_buffer_length(&self) -> u64 {
        self.max_buffer_length
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
    #[cfg(not(target_os = "macos"))]
    fn metal_device_unsupported_on_non_macos() {
        let result = MetalDevice::new();
        assert!(matches!(result, Err(MetalError::UnsupportedPlatform)));
    }
}
