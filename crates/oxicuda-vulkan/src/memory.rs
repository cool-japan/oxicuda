//! Device memory management for the Vulkan compute backend.
//!
//! [`VulkanMemoryManager`] allocates Vulkan buffers backed by host-visible,
//! host-coherent memory so that `copy_htod` and `copy_dtoh` can be
//! implemented via `vkMapMemory` / `memcpy` without an explicit staging
//! buffer.  For a production backend you would separate device-local buffers
//! (fast) from staging buffers (host-visible), but for this backend the
//! simpler approach is sufficient.

use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::device::VulkanDevice;
use crate::error::{VulkanError, VulkanResult};

// ─── Internal buffer record ──────────────────────────────────

struct BufferRecord {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

// ─── VulkanMemoryManager ─────────────────────────────────────

/// Thread-safe Vulkan buffer allocator.
///
/// Each allocation is identified by an opaque `u64` handle that maps to a
/// `(vk::Buffer, vk::DeviceMemory)` pair backed by host-visible,
/// host-coherent memory.
pub struct VulkanMemoryManager {
    device: Arc<VulkanDevice>,
    /// Live buffer records keyed by handle.
    buffers: Mutex<HashMap<u64, BufferRecord>>,
    /// Monotonically increasing handle counter.
    next_handle: AtomicU64,
}

// SAFETY: The inner `Mutex` guards all mutable access to the buffer map;
// `VulkanDevice` is already `Send + Sync`.
unsafe impl Send for VulkanMemoryManager {}
unsafe impl Sync for VulkanMemoryManager {}

impl VulkanMemoryManager {
    /// Create a new memory manager backed by `device`.
    pub fn new(device: Arc<VulkanDevice>) -> Self {
        Self {
            device,
            buffers: Mutex::new(HashMap::new()),
            // Start at 1 — handle 0 is reserved as "null".
            next_handle: AtomicU64::new(1),
        }
    }

    /// Allocate `bytes` bytes of device memory and return an opaque handle.
    ///
    /// Returns `Err(InvalidArgument)` if `bytes == 0`.
    pub fn alloc(&self, bytes: usize) -> VulkanResult<u64> {
        if bytes == 0 {
            return Err(VulkanError::InvalidArgument(
                "allocation size must be > 0".into(),
            ));
        }

        let size = bytes as u64;
        let vk_dev = self.device.device();

        // 1. Create the buffer.
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { vk_dev.create_buffer(&buffer_info, None) }
            .map_err(|e| VulkanError::VkError(e.as_raw(), "create_buffer".into()))?;

        // 2. Query memory requirements.
        let mem_reqs = unsafe { vk_dev.get_buffer_memory_requirements(buffer) };

        // 3. Find a host-visible, host-coherent memory type.
        let mem_type = self
            .device
            .find_memory_type(
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .inspect_err(|_| {
                // Clean up the buffer on failure.
                unsafe { vk_dev.destroy_buffer(buffer, None) };
            })?;

        // 4. Allocate.
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type);

        let memory = unsafe { vk_dev.allocate_memory(&alloc_info, None) }.map_err(|e| {
            unsafe { vk_dev.destroy_buffer(buffer, None) };
            VulkanError::VkError(e.as_raw(), "allocate_memory".into())
        })?;

        // 5. Bind memory to buffer.
        unsafe { vk_dev.bind_buffer_memory(buffer, memory, 0) }.map_err(|e| {
            unsafe {
                vk_dev.free_memory(memory, None);
                vk_dev.destroy_buffer(buffer, None);
            };
            VulkanError::VkError(e.as_raw(), "bind_buffer_memory".into())
        })?;

        // 6. Register and return handle.
        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);
        let mut guard = self.buffers.lock().map_err(|_| {
            // Mutex poisoned — should not happen in correct usage.
            unsafe {
                vk_dev.free_memory(memory, None);
                vk_dev.destroy_buffer(buffer, None);
            };
            VulkanError::InvalidArgument("memory manager lock poisoned".into())
        })?;
        guard.insert(
            handle,
            BufferRecord {
                buffer,
                memory,
                size,
            },
        );

        tracing::trace!(handle, bytes, "Vulkan buffer allocated");
        Ok(handle)
    }

    /// Free a previously allocated buffer.
    ///
    /// Returns `Err(InvalidArgument)` if the handle is not recognised.
    pub fn free(&self, handle: u64) -> VulkanResult<()> {
        let mut guard = self
            .buffers
            .lock()
            .map_err(|_| VulkanError::InvalidArgument("memory manager lock poisoned".into()))?;

        let record = guard
            .remove(&handle)
            .ok_or_else(|| VulkanError::InvalidArgument(format!("unknown handle {handle}")))?;

        let vk_dev = self.device.device();
        unsafe {
            vk_dev.free_memory(record.memory, None);
            vk_dev.destroy_buffer(record.buffer, None);
        }

        tracing::trace!(handle, "Vulkan buffer freed");
        Ok(())
    }

    /// Copy `src` bytes from the host into the buffer identified by `handle`.
    ///
    /// The source slice must be no larger than the allocated buffer size.
    pub fn copy_to_device(&self, handle: u64, src: &[u8]) -> VulkanResult<()> {
        if src.is_empty() {
            return Ok(());
        }

        let guard = self
            .buffers
            .lock()
            .map_err(|_| VulkanError::InvalidArgument("memory manager lock poisoned".into()))?;

        let record = guard
            .get(&handle)
            .ok_or_else(|| VulkanError::InvalidArgument(format!("unknown handle {handle}")))?;

        if src.len() as u64 > record.size {
            return Err(VulkanError::InvalidArgument(format!(
                "copy_to_device: src ({} bytes) > buffer ({} bytes)",
                src.len(),
                record.size
            )));
        }

        let vk_dev = self.device.device();
        let ptr = unsafe {
            vk_dev.map_memory(
                record.memory,
                0,
                src.len() as u64,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| VulkanError::MemoryMapError(format!("map_memory: {e}")))?;

        // SAFETY: `ptr` is a valid, mapped, host-visible memory region of at
        // least `src.len()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), ptr as *mut u8, src.len());
            vk_dev.unmap_memory(record.memory);
        }

        Ok(())
    }

    /// Copy `buffer` bytes from device memory into `dst`.
    ///
    /// `dst` must be at least as large as the allocated buffer.
    pub fn copy_from_device(&self, dst: &mut [u8], handle: u64) -> VulkanResult<()> {
        if dst.is_empty() {
            return Ok(());
        }

        let guard = self
            .buffers
            .lock()
            .map_err(|_| VulkanError::InvalidArgument("memory manager lock poisoned".into()))?;

        let record = guard
            .get(&handle)
            .ok_or_else(|| VulkanError::InvalidArgument(format!("unknown handle {handle}")))?;

        if dst.len() as u64 > record.size {
            return Err(VulkanError::InvalidArgument(format!(
                "copy_from_device: dst ({} bytes) > buffer ({} bytes)",
                dst.len(),
                record.size
            )));
        }

        let vk_dev = self.device.device();
        let ptr = unsafe {
            vk_dev.map_memory(
                record.memory,
                0,
                dst.len() as u64,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| VulkanError::MemoryMapError(format!("map_memory: {e}")))?;

        // SAFETY: same as copy_to_device.
        unsafe {
            std::ptr::copy_nonoverlapping(ptr as *const u8, dst.as_mut_ptr(), dst.len());
            vk_dev.unmap_memory(record.memory);
        }

        Ok(())
    }

    /// Returns the size in bytes of the buffer identified by `handle`.
    pub fn buffer_size(&self, handle: u64) -> VulkanResult<u64> {
        let guard = self
            .buffers
            .lock()
            .map_err(|_| VulkanError::InvalidArgument("memory manager lock poisoned".into()))?;
        guard
            .get(&handle)
            .map(|r| r.size)
            .ok_or_else(|| VulkanError::InvalidArgument(format!("unknown handle {handle}")))
    }

    /// Return the number of live allocations (useful for leak detection in tests).
    pub fn live_count(&self) -> usize {
        self.buffers.lock().map(|g| g.len()).unwrap_or(0)
    }
}

impl Drop for VulkanMemoryManager {
    fn drop(&mut self) {
        // Free any buffers that were not explicitly freed.
        let Ok(mut guard) = self.buffers.lock() else {
            return;
        };
        let vk_dev = self.device.device();
        for (handle, record) in guard.drain() {
            tracing::warn!(handle, "Vulkan buffer leaked — freeing in drop");
            unsafe {
                vk_dev.free_memory(record.memory, None);
                vk_dev.destroy_buffer(record.buffer, None);
            }
        }
    }
}

impl std::fmt::Debug for VulkanMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanMemoryManager")
            .field("live_count", &self.live_count())
            .finish()
    }
}
