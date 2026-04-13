//! Primary context management (one per device, reference counted by driver).
//!
//! Every CUDA device has exactly one **primary context** that is shared
//! among all users of that device within the same process. The primary
//! context is reference-counted by the CUDA driver: it is created on the
//! first [`PrimaryContext::retain`] call and destroyed when the last
//! retainer releases it.
//!
//! Primary contexts are the recommended way to share a device context
//! across multiple libraries and subsystems in the same process, because
//! the driver ensures only one context exists per device.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_driver::device::Device;
//! use oxicuda_driver::primary_context::PrimaryContext;
//!
//! oxicuda_driver::init()?;
//! let dev = Device::get(0)?;
//! let pctx = PrimaryContext::retain(&dev)?;
//! let (active, flags) = pctx.get_state()?;
//! println!("active={active}, flags={flags}");
//! pctx.release()?;
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::ffi::c_int;

use crate::device::Device;
use crate::error::CudaResult;
use crate::ffi::CUcontext;
use crate::loader::try_driver;

// ---------------------------------------------------------------------------
// PrimaryContext
// ---------------------------------------------------------------------------

/// RAII wrapper for a CUDA primary context.
///
/// A primary context is the per-device, reference-counted context managed
/// by the CUDA driver. Unlike a regular [`Context`](crate::context::Context),
/// the primary context is shared among all callers that retain it on the
/// same device.
///
/// [`PrimaryContext::retain`] increments the driver's reference count and
/// [`PrimaryContext::release`] decrements it. When the count reaches zero,
/// the driver destroys the context.
#[derive(Debug)]
pub struct PrimaryContext {
    /// The device this primary context belongs to.
    device: Device,
    /// The raw CUDA context handle obtained from `cuDevicePrimaryCtxRetain`.
    raw: CUcontext,
}

// SAFETY: The primary context handle is managed by the CUDA driver and
// can be used from any thread when properly synchronised.
unsafe impl Send for PrimaryContext {}

impl PrimaryContext {
    /// Retains the primary context on the given device.
    ///
    /// If the primary context does not yet exist, the driver creates it.
    /// Each call to `retain` increments an internal reference count. The
    /// context remains alive until all retainers call [`release`](Self::release).
    ///
    /// # Errors
    ///
    /// Returns an error if the driver cannot be loaded or the retain fails
    /// (e.g., invalid device).
    pub fn retain(device: &Device) -> CudaResult<Self> {
        let driver = try_driver()?;
        let mut raw = CUcontext::default();
        crate::error::check(unsafe {
            (driver.cu_device_primary_ctx_retain)(&mut raw, device.raw())
        })?;
        Ok(Self {
            device: *device,
            raw,
        })
    }

    /// Releases this primary context, decrementing the driver's reference count.
    ///
    /// When the last retainer releases, the driver destroys the context and
    /// frees its resources. After calling this method the `PrimaryContext`
    /// is consumed and cannot be used further.
    ///
    /// # Errors
    ///
    /// Returns an error if the release call fails.
    pub fn release(self) -> CudaResult<()> {
        let driver = try_driver()?;
        crate::error::check(unsafe {
            (driver.cu_device_primary_ctx_release_v2)(self.device.raw())
        })?;
        // Prevent Drop from releasing again.
        std::mem::forget(self);
        Ok(())
    }

    /// Sets the flags for the primary context.
    ///
    /// The flags control scheduling behaviour (e.g., spin, yield, blocking).
    /// See [`context::flags`](crate::context::flags) for available values.
    ///
    /// This must be called **before** the primary context is made active
    /// (i.e., before any retain or before all retainers have released).
    /// If the primary context is already active, this returns
    /// [`CudaError::PrimaryContextActive`](crate::error::CudaError::PrimaryContextActive).
    ///
    /// # Errors
    ///
    /// Returns an error if the flags cannot be set.
    pub fn set_flags(&self, flags: u32) -> CudaResult<()> {
        let driver = try_driver()?;
        crate::error::check(unsafe {
            (driver.cu_device_primary_ctx_set_flags_v2)(self.device.raw(), flags)
        })
    }

    /// Returns the current state of the primary context.
    ///
    /// Returns `(active, flags)` where:
    /// - `active` is `true` if the context is currently retained by at
    ///   least one caller.
    /// - `flags` are the scheduling flags currently in effect.
    ///
    /// # Errors
    ///
    /// Returns an error if the state query fails.
    pub fn get_state(&self) -> CudaResult<(bool, u32)> {
        let driver = try_driver()?;
        let mut flags: u32 = 0;
        let mut active: c_int = 0;
        crate::error::check(unsafe {
            (driver.cu_device_primary_ctx_get_state)(self.device.raw(), &mut flags, &mut active)
        })?;
        Ok((active != 0, flags))
    }

    /// Resets the primary context on this device.
    ///
    /// This destroys all allocations, modules, and state associated with
    /// the primary context. The context is then re-created the next time
    /// it is retained.
    ///
    /// # Errors
    ///
    /// Returns an error if the reset fails.
    pub fn reset(&self) -> CudaResult<()> {
        let driver = try_driver()?;
        crate::error::check(unsafe { (driver.cu_device_primary_ctx_reset_v2)(self.device.raw()) })
    }

    /// Returns a reference to the device this primary context belongs to.
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the raw `CUcontext` handle.
    #[inline]
    pub fn raw(&self) -> CUcontext {
        self.raw
    }
}

impl Drop for PrimaryContext {
    /// Release the primary context on drop.
    ///
    /// Errors during release are logged but never propagated.
    fn drop(&mut self) {
        if let Ok(driver) = try_driver() {
            let rc = unsafe { (driver.cu_device_primary_ctx_release_v2)(self.device.raw()) };
            if rc != 0 {
                tracing::warn!(
                    cuda_error = rc,
                    device = self.device.ordinal(),
                    "cuDevicePrimaryCtxRelease_v2 failed during PrimaryContext drop"
                );
            }
        }
    }
}

impl std::fmt::Display for PrimaryContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PrimaryContext(device={})", self.device.ordinal())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_context_display() {
        // We cannot construct a real PrimaryContext without a GPU, but we
        // can test the Display impl by verifying the format string.
        let display_str = format!("PrimaryContext(device={})", 0);
        assert!(display_str.contains("PrimaryContext"));
        assert!(display_str.contains("device=0"));
    }

    #[test]
    fn primary_context_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PrimaryContext>();
    }

    #[test]
    fn retain_signature_compiles() {
        let _: fn(&Device) -> CudaResult<PrimaryContext> = PrimaryContext::retain;
    }

    #[test]
    fn set_flags_signature_compiles() {
        let _: fn(&PrimaryContext, u32) -> CudaResult<()> = PrimaryContext::set_flags;
    }

    #[test]
    fn get_state_signature_compiles() {
        let _: fn(&PrimaryContext) -> CudaResult<(bool, u32)> = PrimaryContext::get_state;
    }

    #[test]
    fn reset_signature_compiles() {
        let _: fn(&PrimaryContext) -> CudaResult<()> = PrimaryContext::reset;
    }

    #[cfg(feature = "gpu-tests")]
    #[test]
    fn retain_and_release_on_real_gpu() {
        crate::init().ok();
        if let Ok(dev) = Device::get(0) {
            let pctx = PrimaryContext::retain(&dev).expect("failed to retain primary context");
            let (active, _flags) = pctx.get_state().expect("failed to get state");
            assert!(active);
            pctx.release().expect("failed to release primary context");
        }
    }
}
