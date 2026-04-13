//! Peer-to-peer device access.
//!
//! Implements:
//! - `cudaDeviceCanAccessPeer`
//! - `cudaDeviceEnablePeerAccess`
//! - `cudaDeviceDisablePeerAccess`
//! - `cudaMemcpyPeer` / `cudaMemcpyPeerAsync`

use std::ffi::c_int;

use oxicuda_driver::loader::try_driver;

use crate::error::{CudaRtError, CudaRtResult};
use crate::memory::DevicePtr;
use crate::stream::CudaStream;

/// Check whether `device` can directly access the memory of `peer_device`.
///
/// Mirrors `cudaDeviceCanAccessPeer`.
///
/// Returns `Ok(true)` if peer access is supported.
///
/// # Errors
///
/// Propagates driver errors.
pub fn device_can_access_peer(device: u32, peer_device: u32) -> CudaRtResult<bool> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut can_access: c_int = 0;
    // SAFETY: FFI; both ordinals are checked against count by caller if needed.
    let rc = unsafe {
        (api.cu_device_can_access_peer)(&raw mut can_access, device as c_int, peer_device as c_int)
    };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
    }
    Ok(can_access != 0)
}

/// Enable peer access from the current context to the context owning `peer_device`.
///
/// Mirrors `cudaDeviceEnablePeerAccess`.
///
/// # Errors
///
/// - [`CudaRtError::PeerAccessUnsupported`] — link does not support peer access.
/// - [`CudaRtError::PeerAccessAlreadyEnabled`] — already enabled.
/// - Other driver errors.
pub fn device_enable_peer_access(peer_device: u32, flags: u32) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut peer_ctx = oxicuda_driver::ffi::CUcontext::default();
    // Retain the primary context of the peer device.
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_device_primary_ctx_retain)(&raw mut peer_ctx, peer_device as c_int) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
    }
    // Enable peer access to that context.
    // SAFETY: FFI.
    let rc2 = unsafe { (api.cu_ctx_enable_peer_access)(peer_ctx, flags) };
    if rc2 != 0 {
        // Release the retained context regardless.
        // SAFETY: FFI.
        unsafe { (api.cu_device_primary_ctx_release_v2)(peer_device as c_int) };
        return Err(CudaRtError::from_code(rc2).unwrap_or(CudaRtError::PeerAccessUnsupported));
    }
    Ok(())
}

/// Disable peer access from the current context to `peer_device`.
///
/// Mirrors `cudaDeviceDisablePeerAccess`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn device_disable_peer_access(peer_device: u32) -> CudaRtResult<()> {
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut peer_ctx = oxicuda_driver::ffi::CUcontext::default();
    // SAFETY: FFI.
    let rc = unsafe { (api.cu_device_primary_ctx_retain)(&raw mut peer_ctx, peer_device as c_int) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidDevice));
    }
    // SAFETY: FFI.
    let rc2 = unsafe { (api.cu_ctx_disable_peer_access)(peer_ctx) };
    if rc2 != 0 {
        // SAFETY: FFI.
        unsafe { (api.cu_device_primary_ctx_release_v2)(peer_device as c_int) };
        return Err(CudaRtError::from_code(rc2).unwrap_or(CudaRtError::PeerAccessNotEnabled));
    }
    Ok(())
}

/// Copy `count` bytes from `src` on `src_device` to `dst` on `dst_device`.
///
/// Mirrors `cudaMemcpyPeer`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memcpy_peer(
    dst: DevicePtr,
    dst_device: u32,
    src: DevicePtr,
    src_device: u32,
    count: usize,
) -> CudaRtResult<()> {
    if count == 0 {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut dst_ctx = oxicuda_driver::ffi::CUcontext::default();
    let mut src_ctx = oxicuda_driver::ffi::CUcontext::default();
    // SAFETY: FFI.
    unsafe { (api.cu_device_primary_ctx_retain)(&raw mut dst_ctx, dst_device as c_int) };
    unsafe { (api.cu_device_primary_ctx_retain)(&raw mut src_ctx, src_device as c_int) };
    // SAFETY: FFI; pointers are valid device allocations on the specified devices.
    let rc = unsafe { (api.cu_memcpy_peer)(dst.0, dst_ctx, src.0, src_ctx, count) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidMemcpyDirection));
    }
    Ok(())
}

/// Asynchronously copy across devices on `stream`.
///
/// Mirrors `cudaMemcpyPeerAsync`.
///
/// # Errors
///
/// Propagates driver errors.
pub fn memcpy_peer_async(
    dst: DevicePtr,
    dst_device: u32,
    src: DevicePtr,
    src_device: u32,
    count: usize,
    stream: CudaStream,
) -> CudaRtResult<()> {
    if count == 0 {
        return Ok(());
    }
    let api = try_driver().map_err(|_| CudaRtError::DriverNotAvailable)?;
    let mut dst_ctx = oxicuda_driver::ffi::CUcontext::default();
    let mut src_ctx = oxicuda_driver::ffi::CUcontext::default();
    // SAFETY: FFI.
    unsafe { (api.cu_device_primary_ctx_retain)(&raw mut dst_ctx, dst_device as c_int) };
    unsafe { (api.cu_device_primary_ctx_retain)(&raw mut src_ctx, src_device as c_int) };
    // SAFETY: FFI.
    let rc =
        unsafe { (api.cu_memcpy_peer_async)(dst.0, dst_ctx, src.0, src_ctx, count, stream.raw()) };
    if rc != 0 {
        return Err(CudaRtError::from_code(rc).unwrap_or(CudaRtError::InvalidMemcpyDirection));
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_access_self_check() {
        // Without GPU, driver returns DriverNotAvailable.
        // With GPU, peer access with itself should return false or succeed.
        match device_can_access_peer(0, 0) {
            Ok(v) => {
                // Self-access should typically be false for P2P (same device).
                let _ = v;
            }
            // Driver absent or not initialised — both are expected without a GPU.
            Err(CudaRtError::DriverNotAvailable)
            | Err(CudaRtError::NoGpu)
            | Err(CudaRtError::InitializationError)
            | Err(CudaRtError::InvalidDevice) => {}
            Err(e) => panic!("unexpected: {e}"),
        }
    }
}
