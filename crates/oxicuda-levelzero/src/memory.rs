//! Level Zero memory manager — allocates, copies, and frees device memory
//! buffers using the Level Zero API with host-staging for transfers.
//!
//! Device memory is not directly CPU-accessible; all host↔device copies
//! use a temporary host-side staging allocation and a command list.
//!
//! All buffers are tracked by opaque `u64` handles (starting at 1) that
//! mirror the CUDA device-pointer model used by the rest of OxiCUDA.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::{
    device::LevelZeroDevice,
    error::{LevelZeroError, LevelZeroResult},
};

// ─── Platform-specific imports ───────────────────────────────────────────────

#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::ffi::c_void;

#[cfg(any(target_os = "linux", target_os = "windows"))]
use crate::device::{
    ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
    ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, ZeCommandListDesc, ZeCommandListHandle,
    ZeDeviceMemAllocDesc, ZeHostMemAllocDesc,
};

// ─── Internal buffer record ──────────────────────────────────────────────────

/// Bookkeeping entry for a single allocated Level Zero device buffer.
struct L0BufferRecord {
    /// Raw device pointer (Linux and Windows only).
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    device_ptr: *mut c_void,
    /// Byte size of the allocation.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    size: u64,
}

// SAFETY: `L0BufferRecord` contains a raw pointer that is logically owned
// by the `LevelZeroMemoryManager`.  Access is serialized through a `Mutex`.
#[cfg(any(target_os = "linux", target_os = "windows"))]
unsafe impl Send for L0BufferRecord {}

// ─── Memory manager ──────────────────────────────────────────────────────────

/// Manages a pool of Level Zero device buffers, returning opaque `u64` handles.
///
/// Uses explicit host-staging buffers and command lists for host↔device
/// data transfers, matching the Level Zero programming model.
///
/// All public methods take `&self` so the manager can be shared behind `Arc`.
pub struct LevelZeroMemoryManager {
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    device: Arc<LevelZeroDevice>,
    buffers: Mutex<HashMap<u64, L0BufferRecord>>,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    next_handle: AtomicU64,
}

impl LevelZeroMemoryManager {
    /// Create a new memory manager backed by `device`.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn new(device: Arc<LevelZeroDevice>) -> Self {
        Self {
            device,
            buffers: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        }
    }

    /// Stub constructor on unsupported platforms.
    ///
    /// All methods return [`LevelZeroError::UnsupportedPlatform`].
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    pub fn new(_device: Arc<LevelZeroDevice>) -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
        }
    }

    /// Allocate `bytes` bytes of device memory.
    ///
    /// Returns an opaque handle.  The caller must eventually call [`free`](Self::free).
    pub fn alloc(&self, bytes: usize) -> LevelZeroResult<u64> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            let api = &self.device.api;
            let context = self.device.context;
            let device_handle = self.device.device;

            let desc = ZeDeviceMemAllocDesc {
                stype: ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                p_next: std::ptr::null(),
                flags: 0,
                ordinal: 0,
            };

            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: `context` and `device_handle` are valid Level Zero handles;
            // `desc` is properly initialized; `ptr` is a valid output pointer.
            let rc = unsafe {
                (api.ze_mem_alloc_device)(
                    context,
                    &desc,
                    bytes,
                    64, // 64-byte alignment
                    device_handle,
                    &mut ptr as *mut *mut c_void,
                )
            };

            if rc != 0 {
                return Err(LevelZeroError::ZeError(
                    rc,
                    "zeMemAllocDevice failed".into(),
                ));
            }

            let handle = self.next_handle.fetch_add(1, Relaxed);
            self.buffers
                .lock()
                .map_err(|_| LevelZeroError::CommandListError("mutex poisoned".into()))?
                .insert(
                    handle,
                    L0BufferRecord {
                        device_ptr: ptr,
                        size: bytes as u64,
                    },
                );

            Ok(handle)
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            let _ = bytes;
            Err(LevelZeroError::UnsupportedPlatform)
        }
    }

    /// Release the device buffer associated with `handle`.
    ///
    /// Unknown handles are silently ignored (idempotent free).
    pub fn free(&self, handle: u64) -> LevelZeroResult<()> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            let record = self
                .buffers
                .lock()
                .map_err(|_| LevelZeroError::CommandListError("mutex poisoned".into()))?
                .remove(&handle);

            if let Some(rec) = record {
                let api = &self.device.api;
                let context = self.device.context;
                // SAFETY: `rec.device_ptr` was allocated by `zeMemAllocDevice`
                // and has not been freed yet (we just removed it from the map).
                let rc = unsafe { (api.ze_mem_free)(context, rec.device_ptr) };
                if rc != 0 {
                    return Err(LevelZeroError::ZeError(rc, "zeMemFree failed".into()));
                }
            }
            Ok(())
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            let _ = handle;
            Err(LevelZeroError::UnsupportedPlatform)
        }
    }

    /// Upload host bytes `src` into the device buffer identified by `handle`.
    ///
    /// Allocates a temporary host-side staging buffer, copies the data into it,
    /// then uses a command list to schedule the device copy and waits for completion.
    pub fn copy_to_device(&self, handle: u64, src: &[u8]) -> LevelZeroResult<()> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            let device_ptr = {
                let buffers = self
                    .buffers
                    .lock()
                    .map_err(|_| LevelZeroError::CommandListError("mutex poisoned".into()))?;
                let rec = buffers.get(&handle).ok_or_else(|| {
                    LevelZeroError::InvalidArgument(format!("unknown handle {handle}"))
                })?;
                rec.device_ptr
            };

            let api = &self.device.api;
            let context = self.device.context;
            let device_handle = self.device.device;
            let queue = self.device.queue;
            let copy_len = src.len();

            // Allocate a host staging buffer.
            let host_desc = ZeHostMemAllocDesc {
                stype: ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                p_next: std::ptr::null(),
                flags: 0,
            };
            let mut host_ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: `context` is valid; `host_desc` is properly initialized;
            // `host_ptr` is a valid output pointer.
            let rc = unsafe {
                (api.ze_mem_alloc_host)(
                    context,
                    &host_desc,
                    copy_len,
                    64,
                    &mut host_ptr as *mut *mut c_void,
                )
            };
            if rc != 0 {
                return Err(LevelZeroError::ZeError(
                    rc,
                    "zeMemAllocHost (staging) failed".into(),
                ));
            }

            // Copy host data into the staging buffer.
            // SAFETY: `host_ptr` is a valid CPU-accessible pointer allocated
            // by `zeMemAllocHost`; `src` is a valid slice of `copy_len` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), host_ptr as *mut u8, copy_len);
            }

            // Create a command list for the copy.
            let list_desc = ZeCommandListDesc {
                stype: ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                p_next: std::ptr::null(),
                command_queue_group_ordinal: 0,
                flags: 0,
            };
            let mut list: ZeCommandListHandle = std::ptr::null_mut();
            // SAFETY: `context` and `device_handle` are valid; `list_desc` is
            // properly initialized; `list` is a valid output pointer.
            let rc = unsafe {
                (api.ze_command_list_create)(
                    context,
                    device_handle,
                    &list_desc,
                    &mut list as *mut ZeCommandListHandle,
                )
            };
            if rc != 0 {
                // SAFETY: host_ptr was successfully allocated above.
                unsafe { (api.ze_mem_free)(context, host_ptr) };
                return Err(LevelZeroError::CommandListError(format!(
                    "zeCommandListCreate failed: 0x{rc:08x}"
                )));
            }

            // Append the host→device memory copy to the command list.
            // SAFETY: `list`, `device_ptr`, and `host_ptr` are valid;
            // the copy length matches the data we staged.
            let rc = unsafe {
                (api.ze_command_list_append_memory_copy)(
                    list,
                    device_ptr,
                    host_ptr as *const c_void,
                    copy_len,
                    0, // no signal event
                    0, // no wait events
                    std::ptr::null(),
                )
            };
            if rc != 0 {
                // SAFETY: list and host_ptr were allocated above.
                unsafe {
                    (api.ze_command_list_destroy)(list);
                    (api.ze_mem_free)(context, host_ptr);
                }
                return Err(LevelZeroError::CommandListError(format!(
                    "zeCommandListAppendMemoryCopy failed: 0x{rc:08x}"
                )));
            }

            // Close and execute the command list.
            // SAFETY: `list` is in the recording state.
            let rc = unsafe { (api.ze_command_list_close)(list) };
            if rc != 0 {
                unsafe {
                    (api.ze_command_list_destroy)(list);
                    (api.ze_mem_free)(context, host_ptr);
                }
                return Err(LevelZeroError::CommandListError(format!(
                    "zeCommandListClose failed: 0x{rc:08x}"
                )));
            }

            // SAFETY: `queue` is valid; `list` is closed and ready for submission.
            let rc = unsafe { (api.ze_command_queue_execute_command_lists)(queue, 1, &list, 0) };
            if rc != 0 {
                unsafe {
                    (api.ze_command_list_destroy)(list);
                    (api.ze_mem_free)(context, host_ptr);
                }
                return Err(LevelZeroError::CommandListError(format!(
                    "zeCommandQueueExecuteCommandLists failed: 0x{rc:08x}"
                )));
            }

            // Wait for completion.
            // SAFETY: `queue` is valid; u64::MAX means "wait indefinitely".
            let rc = unsafe { (api.ze_command_queue_synchronize)(queue, u64::MAX) };
            if rc != 0 {
                unsafe {
                    (api.ze_command_list_destroy)(list);
                    (api.ze_mem_free)(context, host_ptr);
                }
                return Err(LevelZeroError::CommandListError(format!(
                    "zeCommandQueueSynchronize failed: 0x{rc:08x}"
                )));
            }

            // Clean up.
            // SAFETY: `list` was created above and is no longer needed.
            unsafe {
                (api.ze_command_list_destroy)(list);
                (api.ze_mem_free)(context, host_ptr);
            }

            Ok(())
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            let _ = (handle, src);
            Err(LevelZeroError::UnsupportedPlatform)
        }
    }

    /// Download device buffer `handle` into `dst`.
    ///
    /// Uses a host-staging buffer and a command list for the device→host copy.
    pub fn copy_from_device(&self, dst: &mut [u8], handle: u64) -> LevelZeroResult<()> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            let device_ptr = {
                let buffers = self
                    .buffers
                    .lock()
                    .map_err(|_| LevelZeroError::CommandListError("mutex poisoned".into()))?;
                let rec = buffers.get(&handle).ok_or_else(|| {
                    LevelZeroError::InvalidArgument(format!("unknown handle {handle}"))
                })?;
                rec.device_ptr
            };

            let api = &self.device.api;
            let context = self.device.context;
            let device_handle = self.device.device;
            let queue = self.device.queue;
            let copy_len = dst.len();

            // Allocate a host staging buffer.
            let host_desc = ZeHostMemAllocDesc {
                stype: ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                p_next: std::ptr::null(),
                flags: 0,
            };
            let mut host_ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: `context` is valid; `host_desc` is properly initialized;
            // `host_ptr` is a valid output pointer.
            let rc = unsafe {
                (api.ze_mem_alloc_host)(
                    context,
                    &host_desc,
                    copy_len,
                    64,
                    &mut host_ptr as *mut *mut c_void,
                )
            };
            if rc != 0 {
                return Err(LevelZeroError::ZeError(
                    rc,
                    "zeMemAllocHost (staging) failed".into(),
                ));
            }

            // Create a command list for the copy.
            let list_desc = ZeCommandListDesc {
                stype: ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                p_next: std::ptr::null(),
                command_queue_group_ordinal: 0,
                flags: 0,
            };
            let mut list: ZeCommandListHandle = std::ptr::null_mut();
            // SAFETY: `context` and `device_handle` are valid; `list_desc` is
            // properly initialized; `list` is a valid output pointer.
            let rc = unsafe {
                (api.ze_command_list_create)(
                    context,
                    device_handle,
                    &list_desc,
                    &mut list as *mut ZeCommandListHandle,
                )
            };
            if rc != 0 {
                unsafe { (api.ze_mem_free)(context, host_ptr) };
                return Err(LevelZeroError::CommandListError(format!(
                    "zeCommandListCreate failed: 0x{rc:08x}"
                )));
            }

            // Append the device→host memory copy to the command list.
            // SAFETY: `list`, `host_ptr`, and `device_ptr` are valid;
            // the copy length matches the destination buffer.
            let rc = unsafe {
                (api.ze_command_list_append_memory_copy)(
                    list,
                    host_ptr,
                    device_ptr as *const c_void,
                    copy_len,
                    0, // no signal event
                    0, // no wait events
                    std::ptr::null(),
                )
            };
            if rc != 0 {
                unsafe {
                    (api.ze_command_list_destroy)(list);
                    (api.ze_mem_free)(context, host_ptr);
                }
                return Err(LevelZeroError::CommandListError(format!(
                    "zeCommandListAppendMemoryCopy failed: 0x{rc:08x}"
                )));
            }

            // Close and execute the command list.
            // SAFETY: `list` is in the recording state.
            let rc = unsafe { (api.ze_command_list_close)(list) };
            if rc != 0 {
                unsafe {
                    (api.ze_command_list_destroy)(list);
                    (api.ze_mem_free)(context, host_ptr);
                }
                return Err(LevelZeroError::CommandListError(format!(
                    "zeCommandListClose failed: 0x{rc:08x}"
                )));
            }

            // SAFETY: `queue` is valid; `list` is closed and ready for submission.
            let rc = unsafe { (api.ze_command_queue_execute_command_lists)(queue, 1, &list, 0) };
            if rc != 0 {
                unsafe {
                    (api.ze_command_list_destroy)(list);
                    (api.ze_mem_free)(context, host_ptr);
                }
                return Err(LevelZeroError::CommandListError(format!(
                    "zeCommandQueueExecuteCommandLists failed: 0x{rc:08x}"
                )));
            }

            // Wait for completion.
            // SAFETY: `queue` is valid; u64::MAX means "wait indefinitely".
            let rc = unsafe { (api.ze_command_queue_synchronize)(queue, u64::MAX) };
            if rc != 0 {
                unsafe {
                    (api.ze_command_list_destroy)(list);
                    (api.ze_mem_free)(context, host_ptr);
                }
                return Err(LevelZeroError::CommandListError(format!(
                    "zeCommandQueueSynchronize failed: 0x{rc:08x}"
                )));
            }

            // Copy staging buffer to destination.
            // SAFETY: `host_ptr` is valid and contains `copy_len` bytes of data
            // transferred from the device; `dst` is a valid mutable slice.
            unsafe {
                std::ptr::copy_nonoverlapping(host_ptr as *const u8, dst.as_mut_ptr(), copy_len);
            }

            // Clean up.
            // SAFETY: `list` and `host_ptr` were allocated above.
            unsafe {
                (api.ze_command_list_destroy)(list);
                (api.ze_mem_free)(context, host_ptr);
            }

            Ok(())
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            let _ = (dst, handle);
            Err(LevelZeroError::UnsupportedPlatform)
        }
    }
}

// ─── Drop ────────────────────────────────────────────────────────────────────

impl Drop for LevelZeroMemoryManager {
    fn drop(&mut self) {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            let api = &self.device.api;
            let context = self.device.context;

            if let Ok(mut map) = self.buffers.lock() {
                for (handle, rec) in map.drain() {
                    tracing::warn!(
                        "LevelZeroMemoryManager: leaked buffer handle {handle} ({} bytes)",
                        rec.size
                    );
                    // SAFETY: `rec.device_ptr` is a valid outstanding allocation.
                    unsafe { (api.ze_mem_free)(context, rec.device_ptr) };
                }
            }
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            // Nothing to do on unsupported platforms.
        }
    }
}

// ─── Debug ───────────────────────────────────────────────────────────────────

impl std::fmt::Debug for LevelZeroMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.buffers.lock().map(|b| b.len()).unwrap_or(0);
        write!(f, "LevelZeroMemoryManager(buffers={count})")
    }
}

// ─── Send + Sync ─────────────────────────────────────────────────────────────

// SAFETY: `LevelZeroMemoryManager` serializes all access through a `Mutex`.
// The raw pointer inside `L0BufferRecord` is owned and not aliased.
unsafe impl Send for LevelZeroMemoryManager {}
// SAFETY: See `Send` impl above.  All mutable operations go through a `Mutex`.
unsafe impl Sync for LevelZeroMemoryManager {}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn try_get_device() -> Option<Arc<LevelZeroDevice>> {
        LevelZeroDevice::new().ok().map(Arc::new)
    }

    #[test]
    fn alloc_and_free_requires_device() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = LevelZeroMemoryManager::new(dev);
        let h = mm.alloc(256).expect("alloc 256 bytes");
        assert!(h > 0);
        mm.free(h).expect("free");
        // Double-free: the handle is gone from the map, so it silently ignores.
        mm.free(h).expect("double-free is a no-op");
    }

    #[test]
    fn copy_roundtrip_requires_device() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = LevelZeroMemoryManager::new(dev);

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
        let mm = LevelZeroMemoryManager::new(dev);
        let err = mm.copy_to_device(9999, b"hello").unwrap_err();
        assert!(matches!(err, LevelZeroError::InvalidArgument(_)));
    }

    #[test]
    fn debug_impl_smoke() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = LevelZeroMemoryManager::new(dev);
        let s = format!("{mm:?}");
        assert!(s.contains("LevelZeroMemoryManager"));
    }
}
