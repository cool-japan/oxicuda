//! WebGPU buffer manager — allocates, copies, and frees `wgpu::Buffer` objects
//! through an opaque `u64` handle interface that mirrors the CUDA device-pointer
//! model used by the rest of OxiCUDA.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use wgpu;

use crate::{
    device::WebGpuDevice,
    error::{WebGpuError, WebGpuResult},
};

// ─── Buffer bookkeeping ──────────────────────────────────────────────────────

/// Internal record for a single allocated `wgpu::Buffer`.
pub struct WebGpuBufferInfo {
    /// The GPU-resident buffer.
    pub buffer: wgpu::Buffer,
    /// Byte size of the buffer.
    pub size: u64,
}

// ─── Memory manager ──────────────────────────────────────────────────────────

/// Manages a pool of device-resident `wgpu::Buffer` objects, returning opaque
/// `u64` handles to callers.
///
/// All public methods are `&self` to allow shared references from the backend.
pub struct WebGpuMemoryManager {
    device: Arc<WebGpuDevice>,
    buffers: Mutex<HashMap<u64, WebGpuBufferInfo>>,
    next_handle: AtomicU64,
}

impl WebGpuMemoryManager {
    /// Create a new memory manager backed by `device`.
    pub fn new(device: Arc<WebGpuDevice>) -> Self {
        Self {
            device,
            buffers: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        }
    }

    /// Allocate a new device buffer of `bytes` bytes.
    ///
    /// Returns an opaque handle that identifies the buffer.
    pub fn alloc(&self, bytes: usize) -> WebGpuResult<u64> {
        let size = bytes as u64;
        let buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-webgpu-buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);

        self.buffers
            .lock()
            .map_err(|_| WebGpuError::BufferMapping("mutex poisoned".into()))?
            .insert(handle, WebGpuBufferInfo { buffer, size });

        Ok(handle)
    }

    /// Release the buffer associated with `handle`.
    ///
    /// The handle is silently ignored if it is unknown (already freed).
    pub fn free(&self, handle: u64) -> WebGpuResult<()> {
        self.buffers
            .lock()
            .map_err(|_| WebGpuError::BufferMapping("mutex poisoned".into()))?
            .remove(&handle);
        Ok(())
    }

    /// Upload `src` (host bytes) into the device buffer identified by `handle`.
    pub fn copy_to_device(&self, handle: u64, src: &[u8]) -> WebGpuResult<()> {
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| WebGpuError::BufferMapping("mutex poisoned".into()))?;

        let buf_info = buffers
            .get(&handle)
            .ok_or_else(|| WebGpuError::InvalidArgument(format!("unknown handle {handle}")))?;

        self.device.queue.write_buffer(&buf_info.buffer, 0, src);
        Ok(())
    }

    /// Lock the internal buffer map and return a guard for direct access.
    ///
    /// Used by the backend to look up multiple buffers within a single lock scope
    /// (e.g. when building wgpu bind groups for compute passes).
    pub(crate) fn lock_buffers(
        &self,
    ) -> WebGpuResult<std::sync::MutexGuard<'_, HashMap<u64, WebGpuBufferInfo>>> {
        self.buffers
            .lock()
            .map_err(|_| WebGpuError::BufferMapping("mutex poisoned".into()))
    }

    /// Download the device buffer identified by `handle` into `dst` (host bytes).
    ///
    /// Uses a temporary `MAP_READ` staging buffer and blocks until the GPU
    /// work completes.
    pub fn copy_from_device(&self, dst: &mut [u8], handle: u64) -> WebGpuResult<()> {
        // Phase 1: acquire the lock, build a staging buffer + command encoder,
        // and submit the copy.  The lock is dropped at the end of this block so
        // that `device.poll()` (Phase 2) does not hold the mutex.
        let staging = {
            let buffers = self
                .buffers
                .lock()
                .map_err(|_| WebGpuError::BufferMapping("mutex poisoned".into()))?;

            let buf_info = buffers
                .get(&handle)
                .ok_or_else(|| WebGpuError::InvalidArgument(format!("unknown handle {handle}")))?;

            let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oxicuda-webgpu-staging"),
                size: buf_info.size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder =
                self.device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("oxicuda-webgpu-readback"),
                    });

            encoder.copy_buffer_to_buffer(&buf_info.buffer, 0, &staging, 0, buf_info.size);
            self.device.queue.submit(std::iter::once(encoder.finish()));

            staging
            // Mutex guard dropped here — lock released before poll.
        };

        // Phase 2: map the staging buffer and read the data back to the host.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // Ignore send errors — the receiver may have been dropped.
            let _ = tx.send(result);
        });

        // Block the calling thread until all submitted GPU work (including the
        // copy) is complete.
        let _ = self.device.device.poll(wgpu::PollType::wait_indefinitely());

        rx.recv()
            .map_err(|_| WebGpuError::BufferMapping("channel closed before map completed".into()))?
            .map_err(|e| WebGpuError::BufferMapping(format!("{e:?}")))?;

        let data = slice.get_mapped_range();
        let copy_len = dst.len().min(data.len());
        dst[..copy_len].copy_from_slice(&data[..copy_len]);
        drop(data);
        staging.unmap();

        Ok(())
    }
}

impl std::fmt::Debug for WebGpuMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.buffers.lock().map(|b| b.len()).unwrap_or(0);
        write!(f, "WebGpuMemoryManager(buffers={})", count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::WebGpuDevice;

    fn try_get_device() -> Option<Arc<WebGpuDevice>> {
        WebGpuDevice::new().ok().map(Arc::new)
    }

    #[test]
    fn alloc_and_free_requires_device() {
        let Some(dev) = try_get_device() else {
            // No GPU — skip.
            return;
        };
        let mm = WebGpuMemoryManager::new(dev);
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
        let mm = WebGpuMemoryManager::new(dev);

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
        let mm = WebGpuMemoryManager::new(dev);
        let err = mm.copy_to_device(9999, b"hello").unwrap_err();
        assert!(matches!(err, WebGpuError::InvalidArgument(_)));
    }
}
