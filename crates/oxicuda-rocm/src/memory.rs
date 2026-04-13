//! ROCm/HIP device memory manager.
//!
//! Allocates GPU memory via `hipMalloc`, tracks allocations by opaque `u64`
//! handles (starting at 1), and provides explicit copy operations between
//! host and device.
//!
//! On non-Linux platforms every method returns
//! [`RocmError::UnsupportedPlatform`].

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[cfg(target_os = "linux")]
use std::sync::atomic::AtomicU64;

#[cfg(target_os = "linux")]
use std::sync::atomic::Ordering;

use crate::{
    device::RocmDevice,
    error::{RocmError, RocmResult},
};

#[cfg(target_os = "linux")]
use crate::device::{HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE, HIP_SUCCESS};

// ─── Internal buffer record ──────────────────────────────────────────────────

/// Bookkeeping entry for a single HIP device allocation.
struct RocmBufferRecord {
    /// The raw device pointer returned by `hipMalloc` (Linux only).
    #[cfg(target_os = "linux")]
    device_ptr: *mut std::ffi::c_void,
    /// Byte size of the allocation.
    #[cfg(target_os = "linux")]
    size: u64,
}

// SAFETY: `device_ptr` is a raw GPU pointer whose lifetime is managed
// exclusively by `RocmMemoryManager`.  No CPU thread should ever dereference
// it; it is only passed back to HIP API calls.  Therefore cross-thread
// ownership transfer is safe.
#[cfg(target_os = "linux")]
unsafe impl Send for RocmBufferRecord {}
#[cfg(target_os = "linux")]
// SAFETY: See `Send` impl above.
unsafe impl Sync for RocmBufferRecord {}

// ─── Memory manager ──────────────────────────────────────────────────────────

/// Manages a pool of HIP device allocations, returning opaque `u64` handles.
///
/// All public methods take `&self` so the manager can be shared behind `Arc`.
pub struct RocmMemoryManager {
    #[cfg(target_os = "linux")]
    device: Arc<RocmDevice>,
    buffers: Mutex<HashMap<u64, RocmBufferRecord>>,
    #[cfg(target_os = "linux")]
    next_handle: AtomicU64,
}

// SAFETY: `RocmMemoryManager` is safe to send and share across threads.
// The `Mutex` serialises all HashMap accesses.  `Arc<RocmDevice>` is already
// `Send + Sync`.  The raw device pointers inside `RocmBufferRecord` are never
// dereferenced by the CPU — they are opaque handles passed to HIP API calls
// which do their own internal locking.
unsafe impl Send for RocmMemoryManager {}
// SAFETY: See `Send` impl above.
unsafe impl Sync for RocmMemoryManager {}

impl RocmMemoryManager {
    /// Create a new memory manager backed by `device`.
    #[cfg(target_os = "linux")]
    pub fn new(device: Arc<RocmDevice>) -> Self {
        Self {
            device,
            buffers: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        }
    }

    /// Stub constructor on non-Linux platforms.
    ///
    /// All methods return [`RocmError::UnsupportedPlatform`].
    #[cfg(not(target_os = "linux"))]
    pub fn new(_device: Arc<RocmDevice>) -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
        }
    }

    /// Allocate `bytes` bytes of device memory via `hipMalloc`.
    ///
    /// Returns an opaque handle.  The caller must eventually call
    /// [`free`](Self::free) to avoid leaking device memory.
    pub fn alloc(&self, bytes: usize) -> RocmResult<u64> {
        #[cfg(target_os = "linux")]
        {
            let mut raw_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: `hip_malloc` is a valid fn ptr from the loaded HIP
            // library.  `raw_ptr` is properly aligned and ready to receive
            // the device allocation address.
            let rc = unsafe { (self.device.api.hip_malloc)(&mut raw_ptr, bytes) };
            if rc != HIP_SUCCESS {
                return Err(RocmError::HipError(rc, "hipMalloc failed".into()));
            }
            if raw_ptr.is_null() {
                return Err(RocmError::OutOfMemory);
            }
            let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
            self.buffers
                .lock()
                .map_err(|_| RocmError::DeviceError("mutex poisoned".into()))?
                .insert(
                    handle,
                    RocmBufferRecord {
                        device_ptr: raw_ptr,
                        size: bytes as u64,
                    },
                );
            Ok(handle)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = bytes;
            Err(RocmError::UnsupportedPlatform)
        }
    }

    /// Free the device allocation associated with `handle`.
    ///
    /// Unknown handles are silently ignored (idempotent free).
    pub fn free(&self, handle: u64) -> RocmResult<()> {
        #[cfg(target_os = "linux")]
        {
            let record = self
                .buffers
                .lock()
                .map_err(|_| RocmError::DeviceError("mutex poisoned".into()))?
                .remove(&handle);
            if let Some(rec) = record {
                // SAFETY: `hip_free` is a valid fn ptr.  `rec.device_ptr` was
                // returned by a prior `hipMalloc` and has not been freed yet
                // (we just removed it from the map).
                let rc = unsafe { (self.device.api.hip_free)(rec.device_ptr) };
                if rc != HIP_SUCCESS {
                    return Err(RocmError::HipError(rc, "hipFree failed".into()));
                }
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = handle;
            Err(RocmError::UnsupportedPlatform)
        }
    }

    /// Copy host bytes `src` into the device buffer identified by `handle`.
    ///
    /// The copy length is `src.len()` bytes (capped to the buffer size).
    pub fn copy_to_device(&self, handle: u64, src: &[u8]) -> RocmResult<()> {
        #[cfg(target_os = "linux")]
        {
            let buffers = self
                .buffers
                .lock()
                .map_err(|_| RocmError::DeviceError("mutex poisoned".into()))?;
            let rec = buffers
                .get(&handle)
                .ok_or_else(|| RocmError::InvalidArgument(format!("unknown handle {handle}")))?;
            let copy_len = src.len().min(rec.size as usize);
            if copy_len == 0 {
                return Ok(());
            }
            // SAFETY: `hip_memcpy` is a valid fn ptr.  `rec.device_ptr` is a
            // live HIP allocation; `src.as_ptr()` points to a valid CPU slice
            // for `copy_len` bytes.  `HIP_MEMCPY_HOST_TO_DEVICE` is the
            // correct transfer direction.
            let rc = unsafe {
                (self.device.api.hip_memcpy)(
                    rec.device_ptr,
                    src.as_ptr().cast(),
                    copy_len,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                )
            };
            if rc != HIP_SUCCESS {
                return Err(RocmError::HipError(rc, "hipMemcpy H2D failed".into()));
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (handle, src);
            Err(RocmError::UnsupportedPlatform)
        }
    }

    /// Copy device buffer `handle` into host slice `dst`.
    ///
    /// The copy length is `dst.len()` bytes (capped to the buffer size).
    pub fn copy_from_device(&self, dst: &mut [u8], handle: u64) -> RocmResult<()> {
        #[cfg(target_os = "linux")]
        {
            let buffers = self
                .buffers
                .lock()
                .map_err(|_| RocmError::DeviceError("mutex poisoned".into()))?;
            let rec = buffers
                .get(&handle)
                .ok_or_else(|| RocmError::InvalidArgument(format!("unknown handle {handle}")))?;
            let copy_len = dst.len().min(rec.size as usize);
            if copy_len == 0 {
                return Ok(());
            }
            // SAFETY: `hip_memcpy` is a valid fn ptr.  `rec.device_ptr` is a
            // live HIP allocation; `dst.as_mut_ptr()` points to a valid
            // writable CPU slice for `copy_len` bytes.
            // `HIP_MEMCPY_DEVICE_TO_HOST` is the correct direction.
            let rc = unsafe {
                (self.device.api.hip_memcpy)(
                    dst.as_mut_ptr().cast(),
                    rec.device_ptr.cast_const(),
                    copy_len,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                )
            };
            if rc != HIP_SUCCESS {
                return Err(RocmError::HipError(rc, "hipMemcpy D2H failed".into()));
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (dst, handle);
            Err(RocmError::UnsupportedPlatform)
        }
    }
}

impl Drop for RocmMemoryManager {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        {
            // Attempt to free any remaining allocations.  Failures are logged
            // as warnings; we cannot propagate errors from Drop.
            let Ok(mut map) = self.buffers.lock() else {
                tracing::warn!("RocmMemoryManager: mutex poisoned during drop — memory may leak");
                return;
            };
            for (handle, rec) in map.drain() {
                // SAFETY: `hip_free` is a valid fn ptr.  `rec.device_ptr` was
                // returned by `hipMalloc` and has not been freed yet.
                let rc = unsafe { (self.device.api.hip_free)(rec.device_ptr) };
                if rc != HIP_SUCCESS {
                    tracing::warn!(
                        "RocmMemoryManager: hipFree failed for leaked handle {handle} (rc={rc})"
                    );
                }
            }
        }
        // On non-Linux there are no device allocations to release.
    }
}

impl std::fmt::Debug for RocmMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.buffers.lock().map(|b| b.len()).unwrap_or(0);
        write!(f, "RocmMemoryManager(buffers={count})")
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn try_get_device() -> Option<Arc<RocmDevice>> {
        RocmDevice::new().ok().map(Arc::new)
    }

    #[test]
    fn alloc_and_free_requires_device() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = RocmMemoryManager::new(dev);
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
        let mm = RocmMemoryManager::new(dev);

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
        let mm = RocmMemoryManager::new(dev);
        let err = mm.copy_to_device(9999, b"hello").unwrap_err();
        assert!(matches!(err, RocmError::InvalidArgument(_)));
    }

    #[test]
    fn debug_impl_smoke() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = RocmMemoryManager::new(dev);
        let s = format!("{mm:?}");
        assert!(s.contains("RocmMemoryManager"));
    }
}
