//! Metal buffer manager — allocates, copies, and frees `metal::Buffer` objects
//! using the shared-memory storage mode so the CPU can read and write them
//! directly without a staging copy.
//!
//! All buffers are tracked by opaque `u64` handles (starting at 1) that mirror
//! the CUDA device-pointer model used by the rest of OxiCUDA.

#[cfg(target_os = "macos")]
use std::sync::atomic::Ordering;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, atomic::AtomicU64},
};

use crate::{
    device::MetalDevice,
    error::{MetalError, MetalResult},
};

// ─── Internal buffer record ──────────────────────────────────────────────────

/// Bookkeeping entry for a single allocated Metal buffer.
struct MetalBufferInfo {
    /// The GPU-resident (shared-mode) buffer.
    #[cfg(target_os = "macos")]
    buffer: metal::Buffer,
    /// Byte size of the allocation.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    size: u64,
}

// ─── Memory manager ──────────────────────────────────────────────────────────

/// Manages a pool of Metal buffers, returning opaque `u64` handles.
///
/// Uses `MTLResourceOptions::StorageModeShared` so the same physical pages are
/// accessible from both CPU and GPU without explicit synchronisation — the same
/// model used by Metal's unified-memory architecture on Apple Silicon.
///
/// All public methods take `&self` so the manager can be shared behind `Arc`.
pub struct MetalMemoryManager {
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    device: Arc<MetalDevice>,
    buffers: Mutex<HashMap<u64, MetalBufferInfo>>,
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    next_handle: AtomicU64,
}

impl MetalMemoryManager {
    /// Create a new memory manager backed by `device`.
    pub fn new(device: Arc<MetalDevice>) -> Self {
        Self {
            device,
            buffers: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        }
    }

    /// Allocate `bytes` bytes of shared-mode device memory.
    ///
    /// Returns an opaque handle.  The caller must eventually call [`free`](Self::free).
    pub fn alloc(&self, bytes: usize) -> MetalResult<u64> {
        #[cfg(target_os = "macos")]
        {
            let buffer = self
                .device
                .device
                .new_buffer(bytes as u64, metal::MTLResourceOptions::StorageModeShared);
            let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
            self.buffers
                .lock()
                .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?
                .insert(
                    handle,
                    MetalBufferInfo {
                        buffer,
                        size: bytes as u64,
                    },
                );
            Ok(handle)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = bytes;
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// Release the buffer associated with `handle`.
    ///
    /// Unknown handles are silently ignored (idempotent free).
    pub fn free(&self, handle: u64) -> MetalResult<()> {
        self.buffers
            .lock()
            .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?
            .remove(&handle);
        Ok(())
    }

    /// Upload host bytes `src` into the device buffer identified by `handle`.
    ///
    /// Because the buffer uses shared storage, this is a direct CPU `memcpy`.
    pub fn copy_to_device(&self, handle: u64, src: &[u8]) -> MetalResult<()> {
        #[cfg(target_os = "macos")]
        {
            let buffers = self
                .buffers
                .lock()
                .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?;
            let info = buffers
                .get(&handle)
                .ok_or_else(|| MetalError::InvalidArgument(format!("unknown handle {handle}")))?;
            let copy_len = src.len().min(info.size as usize);
            // SAFETY: Metal shared-mode buffers are CPU-accessible; `contents()`
            // returns a valid `*mut c_void` for the entire buffer lifetime.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    info.buffer.contents() as *mut u8,
                    copy_len,
                );
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (handle, src);
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// Download device buffer `handle` into `dst`.
    ///
    /// Because the buffer uses shared storage, this is a direct CPU `memcpy`.
    pub fn copy_from_device(&self, dst: &mut [u8], handle: u64) -> MetalResult<()> {
        #[cfg(target_os = "macos")]
        {
            let buffers = self
                .buffers
                .lock()
                .map_err(|_| MetalError::CommandBufferError("mutex poisoned".into()))?;
            let info = buffers
                .get(&handle)
                .ok_or_else(|| MetalError::InvalidArgument(format!("unknown handle {handle}")))?;
            let copy_len = dst.len().min(info.size as usize);
            // SAFETY: Metal shared-mode buffers are CPU-accessible; `contents()`
            // returns a valid `*const c_void` for the entire buffer lifetime.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    info.buffer.contents() as *const u8,
                    dst.as_mut_ptr(),
                    copy_len,
                );
            }
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (dst, handle);
            Err(MetalError::UnsupportedPlatform)
        }
    }
}

impl std::fmt::Debug for MetalMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.buffers.lock().map(|b| b.len()).unwrap_or(0);
        write!(f, "MetalMemoryManager(buffers={count})")
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    fn try_get_device() -> Option<Arc<MetalDevice>> {
        MetalDevice::new().ok().map(Arc::new)
    }

    #[test]
    fn alloc_and_free_requires_device() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        let h = mm.alloc(256).expect("alloc 256 bytes");
        assert!(h > 0);
        mm.free(h).expect("free");
        // Double-free is silently ignored.
        mm.free(h).expect("double-free is a no-op");
    }

    #[test]
    fn copy_roundtrip_requires_device() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);

        let src: Vec<u8> = (0u8..64).collect();
        let h = mm.alloc(src.len()).expect("alloc");
        mm.copy_to_device(h, &src).expect("copy_to_device");

        let mut dst = vec![0u8; src.len()];
        mm.copy_from_device(&mut dst, h).expect("copy_from_device");

        assert_eq!(src, dst);
        mm.free(h).expect("free");
    }

    #[test]
    fn unknown_handle_returns_error() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        let err = mm.copy_to_device(9999, b"hello").unwrap_err();
        assert!(matches!(err, MetalError::InvalidArgument(_)));
    }

    #[test]
    fn debug_impl_smoke() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = MetalMemoryManager::new(dev);
        let s = format!("{mm:?}");
        assert!(s.contains("MetalMemoryManager"));
    }
}
