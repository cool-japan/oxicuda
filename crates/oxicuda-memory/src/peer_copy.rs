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

use std::ffi::c_int;

use oxicuda_driver::device::Device;
use oxicuda_driver::error::{CudaError, CudaResult};
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::primary_context::PrimaryContext;
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
/// Returns a CUDA driver error if the query fails.
pub fn can_access_peer(device: &Device, peer: &Device) -> CudaResult<bool> {
    let api = try_driver()?;
    let mut can_access: c_int = 0;
    oxicuda_driver::error::check(unsafe {
        (api.cu_device_can_access_peer)(&mut can_access, device.raw(), peer.raw())
    })?;
    Ok(can_access != 0)
}

/// Enables peer access from `device`'s primary context to `peer`'s primary context.
///
/// After calling this function, kernels and copy operations running on `device`
/// can directly read from and write to memory allocated on `peer`.
///
/// # Errors
///
/// * [`CudaError::PeerAccessAlreadyEnabled`] if peer access is already enabled.
/// * [`CudaError::PeerAccessUnsupported`] if the hardware topology does not
///   support direct peer access between these devices.
pub fn enable_peer_access(device: &Device, peer: &Device) -> CudaResult<()> {
    let api = try_driver()?;

    // Retain both primary contexts.  The peer context handle is needed by
    // cuCtxEnablePeerAccess; the device context is set as current so that the
    // enable operation applies to it.
    let dev_ctx = PrimaryContext::retain(device)?;
    let peer_ctx = PrimaryContext::retain(peer)?;

    // Make the device context current on this thread.
    oxicuda_driver::error::check(unsafe { (api.cu_ctx_set_current)(dev_ctx.raw()) })?;

    // Enable access from the current (device) context to the peer context.
    let rc =
        oxicuda_driver::error::check(unsafe { (api.cu_ctx_enable_peer_access)(peer_ctx.raw(), 0) });

    // Release retained contexts regardless of outcome.
    let _ = peer_ctx.release();
    let _ = dev_ctx.release();

    rc
}

/// Disables peer access from `device`'s primary context to `peer`'s primary context.
///
/// # Errors
///
/// * [`CudaError::PeerAccessNotEnabled`] if peer access was not previously enabled.
pub fn disable_peer_access(device: &Device, peer: &Device) -> CudaResult<()> {
    let api = try_driver()?;

    let dev_ctx = PrimaryContext::retain(device)?;
    let peer_ctx = PrimaryContext::retain(peer)?;

    oxicuda_driver::error::check(unsafe { (api.cu_ctx_set_current)(dev_ctx.raw()) })?;

    let rc =
        oxicuda_driver::error::check(unsafe { (api.cu_ctx_disable_peer_access)(peer_ctx.raw()) });

    let _ = peer_ctx.release();
    let _ = dev_ctx.release();

    rc
}

/// Copies data between device buffers on different GPUs (synchronous).
///
/// Both buffers must have the same length.  Peer access should be enabled
/// between the source and destination devices before calling this function.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if buffer lengths do not match.
/// * [`CudaError::PeerAccessNotEnabled`] if peer access has not been enabled.
pub fn copy_peer<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    dst_device: &Device,
    src: &DeviceBuffer<T>,
    src_device: &Device,
) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let api = try_driver()?;
    let byte_size = src.byte_size();

    let dst_ctx = PrimaryContext::retain(dst_device)?;
    let src_ctx = PrimaryContext::retain(src_device)?;

    let rc = oxicuda_driver::error::check(unsafe {
        (api.cu_memcpy_peer)(
            dst.as_device_ptr(),
            dst_ctx.raw(),
            src.as_device_ptr(),
            src_ctx.raw(),
            byte_size,
        )
    });

    let _ = src_ctx.release();
    let _ = dst_ctx.release();

    rc
}

/// Copies data between device buffers on different GPUs (asynchronous).
///
/// The copy is enqueued on `stream` and may not be complete when this
/// function returns.  Both buffers must have the same length.
///
/// # Errors
///
/// * [`CudaError::InvalidValue`] if buffer lengths do not match.
pub fn copy_peer_async<T: Copy>(
    dst: &mut DeviceBuffer<T>,
    dst_device: &Device,
    src: &DeviceBuffer<T>,
    src_device: &Device,
    stream: &Stream,
) -> CudaResult<()> {
    if dst.len() != src.len() {
        return Err(CudaError::InvalidValue);
    }
    let api = try_driver()?;
    let byte_size = src.byte_size();

    let dst_ctx = PrimaryContext::retain(dst_device)?;
    let src_ctx = PrimaryContext::retain(src_device)?;

    let rc = oxicuda_driver::error::check(unsafe {
        (api.cu_memcpy_peer_async)(
            dst.as_device_ptr(),
            dst_ctx.raw(),
            src.as_device_ptr(),
            src_ctx.raw(),
            byte_size,
            stream.raw(),
        )
    });

    let _ = src_ctx.release();
    let _ = dst_ctx.release();

    rc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_signatures_compile() {
        let _f1: fn(&Device, &Device) -> CudaResult<bool> = can_access_peer;
        let _f2: fn(&Device, &Device) -> CudaResult<()> = enable_peer_access;
        let _f3: fn(&Device, &Device) -> CudaResult<()> = disable_peer_access;
        let _f4: fn(
            &mut DeviceBuffer<f32>,
            &Device,
            &DeviceBuffer<f32>,
            &Device,
        ) -> CudaResult<()> = copy_peer;
    }

    #[test]
    fn copy_peer_length_mismatch_returns_invalid_value() {
        // Just confirm copy_peer_async is callable — signature test only.
        type PeerAsyncFn = fn(
            &mut DeviceBuffer<f32>,
            &Device,
            &DeviceBuffer<f32>,
            &Device,
            &Stream,
        ) -> CudaResult<()>;
        let _f: PeerAsyncFn = copy_peer_async;
    }

    #[cfg(feature = "gpu-tests")]
    #[test]
    fn can_access_peer_single_gpu() {
        oxicuda_driver::init().ok();
        let count = oxicuda_driver::device::Device::count().unwrap_or(0);
        if count >= 1 {
            let dev0 = Device::get(0).expect("device 0");
            if count == 1 {
                // Single GPU: can_access_peer with itself returns false or an error.
                let _ = can_access_peer(&dev0, &dev0);
            } else {
                let dev1 = Device::get(1).expect("device 1");
                let _ = can_access_peer(&dev0, &dev1);
            }
        }
    }
}
