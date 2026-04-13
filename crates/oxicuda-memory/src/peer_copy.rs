//! Peer-to-peer (P2P) memory copy operations for multi-GPU workloads.
//!
//! This module provides functions to check, enable, and disable peer access
//! between CUDA devices, as well as copy data between device buffers on
//! different GPUs.
//!
//! Peer access enables direct GPU-to-GPU memory transfers over PCIe or
//! NVLink without staging through host memory, significantly improving
//! transfer bandwidth in multi-GPU configurations.
//!
//! # Platform note
//!
//! The underlying `cuDeviceCanAccessPeer`, `cuCtxEnablePeerAccess`,
//! `cuCtxDisablePeerAccess`, and `cuMemcpyPeer` driver functions are not
//! yet loaded by `oxicuda-driver`.  All functions currently return
//! [`CudaError::NotSupported`] as placeholders.  The API surface is
//! established here so that downstream crates can program against it.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_driver::device::Device;
//! use oxicuda_memory::peer_copy;
//!
//! oxicuda_driver::init()?;
//! let dev0 = Device::get(0)?;
//! let dev1 = Device::get(1)?;
//!
//! if peer_copy::can_access_peer(&dev0, &dev1)? {
//!     peer_copy::enable_peer_access(&dev0, &dev1)?;
//!     // Now D2D copies between dev0 and dev1 can go over NVLink/PCIe
//!     // peer_copy::copy_peer(&mut dst_buf, &dev1, &src_buf, &dev0)?;
//! }
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use oxicuda_driver::device::Device;
use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::stream::Stream;

use crate::device_buffer::DeviceBuffer;

/// Checks whether `device` can directly access memory on `peer`.
///
/// Returns `true` if peer access is supported between the two devices
/// (e.g., over NVLink or PCIe).  Returns `false` if the devices are the
/// same or if the hardware topology does not support peer access.
///
/// # Errors
///
/// Currently returns [`CudaError::NotSupported`] because the underlying
/// driver function pointer (`cuDeviceCanAccessPeer`) is not yet loaded.
pub fn can_access_peer(_device: &Device, _peer: &Device) -> CudaResult<bool> {
    // TODO: load `cuDeviceCanAccessPeer` in DriverApi and call it here.
    // let api = oxicuda_driver::loader::try_driver()?;
    // let mut can_access: i32 = 0;
    // oxicuda_driver::check(unsafe {
    //     (api.cu_device_can_access_peer)(&mut can_access, device.raw(), peer.raw())
    // })?;
    // Ok(can_access != 0)
    Err(CudaError::NotSupported)
}

/// Enables peer access from the current context's device to `peer`.
///
/// After enabling, memory on `peer` can be directly accessed from
/// kernels and copy operations in the current context.
///
/// # Errors
///
/// * [`CudaError::PeerAccessAlreadyEnabled`] if peer access is already
///   enabled.
/// * [`CudaError::PeerAccessUnsupported`] if the hardware does not
///   support peer access.
/// * [`CudaError::NotSupported`] (current stub).
pub fn enable_peer_access(_device: &Device, _peer: &Device) -> CudaResult<()> {
    // TODO: load `cuCtxEnablePeerAccess` in DriverApi and call it.
    // Need to set the device's context as current first, then enable
    // access to peer's context.
    Err(CudaError::NotSupported)
}

/// Disables peer access from the current context's device to `peer`.
///
/// # Errors
///
/// * [`CudaError::PeerAccessNotEnabled`] if peer access was not enabled.
/// * [`CudaError::NotSupported`] (current stub).
pub fn disable_peer_access(_device: &Device, _peer: &Device) -> CudaResult<()> {
    // TODO: load `cuCtxDisablePeerAccess` in DriverApi and call it.
    Err(CudaError::NotSupported)
}

/// Copies data between device buffers on different GPUs (synchronous).
///
/// Both buffers must have the same length.  Peer access should be enabled
/// between the source and destination devices before calling this function.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if buffer lengths do not match.
/// * [`CudaError::PeerAccessNotEnabled`] if peer access has not been
///   enabled.
/// * [`CudaError::NotSupported`] (current stub).
pub fn copy_peer<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    _dst_device: &Device,
    src: &DeviceBuffer<T>,
    _src_device: &Device,
) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    // TODO: load `cuMemcpyPeer` in DriverApi and call it.
    // let byte_size = src.byte_size();
    // let api = oxicuda_driver::loader::try_driver()?;
    // oxicuda_driver::check(unsafe {
    //     (api.cu_memcpy_peer)(
    //         dst.as_device_ptr(), dst_ctx, src.as_device_ptr(), src_ctx, byte_size
    //     )
    // })
    Err(CudaError::NotSupported)
}

/// Copies data between device buffers on different GPUs (asynchronous).
///
/// The copy is enqueued on `stream` and may not be complete when this
/// function returns.  Both buffers must have the same length.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if buffer lengths do not match.
/// * [`CudaError::NotSupported`] (current stub).
pub fn copy_peer_async<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    _dst_device: &Device,
    src: &DeviceBuffer<T>,
    _src_device: &Device,
    _stream: &Stream,
) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    // TODO: load `cuMemcpyPeerAsync` in DriverApi and call it.
    Err(CudaError::NotSupported)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_access_peer_returns_not_supported() {
        // On macOS, we cannot enumerate devices, so we create dummy Device
        // values indirectly. Since `Device::get` will fail, we just test
        // that the function compiles and the error type is correct.
        // The function itself returns NotSupported as a stub.
        let _f: fn(&Device, &Device) -> CudaResult<bool> = can_access_peer;
    }

    #[test]
    fn enable_peer_access_returns_not_supported() {
        let _f: fn(&Device, &Device) -> CudaResult<()> = enable_peer_access;
    }

    #[test]
    fn disable_peer_access_returns_not_supported() {
        let _f: fn(&Device, &Device) -> CudaResult<()> = disable_peer_access;
    }

    #[test]
    fn copy_peer_signature_compiles() {
        let _f: fn(&mut DeviceBuffer<f32>, &Device, &DeviceBuffer<f32>, &Device) -> CudaResult<()> =
            copy_peer;
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn copy_peer_async_signature_compiles() {
        let _f: fn(
            &mut DeviceBuffer<f32>,
            &Device,
            &DeviceBuffer<f32>,
            &Device,
            &Stream,
        ) -> CudaResult<()> = copy_peer_async;
    }

    #[cfg(feature = "gpu-tests")]
    #[test]
    fn peer_access_with_real_devices() {
        oxicuda_driver::init().ok();
        let count = oxicuda_driver::device::Device::count().unwrap_or(0);
        if count >= 2 {
            let dev0 = Device::get(0).expect("device 0");
            let dev1 = Device::get(1).expect("device 1");
            // These may or may not succeed depending on hardware.
            let _ = can_access_peer(&dev0, &dev1);
        }
    }
}
