//! WebGPU buffer manager — allocates, copies, and frees `wgpu::Buffer` objects
//! through an opaque `u64` handle interface that mirrors the CUDA device-pointer
//! model used by the rest of OxiCUDA.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
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
    /// Byte size of the buffer, rounded up to `wgpu::COPY_BUFFER_ALIGNMENT`
    /// (4 bytes) by [`WebGpuMemoryManager::alloc`] — the *physical* size, not
    /// necessarily the exact byte count the caller requested.
    pub size: u64,
}

/// Convert a raw `Device::poll` result into our typed result, distinguishing a
/// genuine timeout (device hung or lost) from any other poll failure.
///
/// Factored out as a free function so the mapping itself is unit-testable
/// without a real GPU — see the `poll_*_maps_to_*` tests below. `pub(crate)`
/// so [`crate::backend::WebGpuBackend::synchronize`] can reuse the same
/// tested mapping instead of duplicating it.
pub(crate) fn poll_result_to_webgpu_result(
    result: Result<wgpu::PollStatus, wgpu::PollError>,
) -> WebGpuResult<()> {
    match result {
        Ok(_status) => Ok(()),
        Err(wgpu::PollError::Timeout) => Err(WebGpuError::Timeout),
        Err(e) => Err(WebGpuError::BufferMapping(format!("poll failed: {e:?}"))),
    }
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
    /// Bounded wait applied to GPU readbacks: long enough not to trip on slow
    /// (but legitimate) workloads or a loaded CI runner, short enough to
    /// eventually convert a genuinely stuck or lost device into a typed
    /// [`WebGpuError::Timeout`] instead of blocking the caller forever.
    const READBACK_POLL_TIMEOUT: Duration = Duration::from_secs(60);

    /// Create a new memory manager backed by `device`.
    pub fn new(device: Arc<WebGpuDevice>) -> Self {
        Self {
            device,
            buffers: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        }
    }

    /// Return an error instead of attempting a GPU operation that can no
    /// longer succeed, once wgpu has reported the device lost (GPU reset,
    /// driver failure, or an external `Device::destroy()` call — see the
    /// device-lost callback installed in [`WebGpuDevice::new`]).
    fn ensure_device_alive(&self) -> WebGpuResult<()> {
        if self.device.is_device_lost() {
            return Err(WebGpuError::DeviceLost(
                "device was lost before the operation could run".into(),
            ));
        }
        Ok(())
    }

    /// Drain the device's uncaptured-error slot (see
    /// [`WebGpuDevice::poll_error`]) and convert a recorded error into a
    /// typed `Err` instead of letting the caller proceed as though the
    /// (non-fatal, but real) wgpu error never happened.
    fn check_uncaptured_error(&self) -> WebGpuResult<()> {
        if let Some(msg) = self.device.poll_error() {
            return Err(WebGpuError::UncapturedError(msg));
        }
        Ok(())
    }

    /// Block until the specific submission identified by `submission_index`
    /// completes, bounded by [`Self::READBACK_POLL_TIMEOUT`].
    ///
    /// Replaces a bare `let _ = device.poll(wait_indefinitely())`, which
    /// silently discarded a `PollError` (or an indefinite hang) and let the
    /// caller read out of a staging buffer that may never have been written.
    ///
    /// Waiting on the *specific* [`wgpu::SubmissionIndex`] returned by the
    /// copy's own `queue.submit(...)` — rather than `submission_index: None`
    /// ("the most recent submission at the time of the poll") — ties this
    /// wait to exactly the work this readback depends on, regardless of what
    /// other threads submit concurrently in between. Because a `wgpu::Queue`
    /// executes submissions in FIFO order, waiting for this copy's index is
    /// also sufficient to guarantee every compute dispatch that produced the
    /// data being read back (always submitted earlier, on the same queue) has
    /// completed — those dispatches no longer poll themselves, see
    /// `WebGpuBackend`'s compute-op methods.
    fn wait_for_gpu(&self, submission_index: wgpu::SubmissionIndex) -> WebGpuResult<()> {
        poll_result_to_webgpu_result(self.device.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(Self::READBACK_POLL_TIMEOUT),
        }))
    }

    /// Allocate a new device buffer of at least `bytes` bytes.
    ///
    /// The physical buffer size is rounded up to `wgpu::COPY_BUFFER_ALIGNMENT`
    /// (4 bytes): WebGPU requires `copy_buffer_to_buffer` sizes, `map_async`
    /// ranges, and STORAGE-bound bindings to be multiples of 4, so an
    /// odd-sized allocation (3 bytes, or an odd count of 2-byte f16 elements)
    /// would otherwise pass `alloc()` cleanly and only fail later — fatally,
    /// with no handler installed — on the first readback or bind. The
    /// rounded-up size is what later `copy_to_device`/`copy_from_device`
    /// calls validate against, which is strictly more permissive than the
    /// caller's requested size, never less.
    ///
    /// Returns [`WebGpuError::InvalidArgument`] for a zero-byte request and
    /// [`WebGpuError::OutOfMemory`] if the (rounded) size exceeds what this
    /// device can bind — checked against the adapter-derived limits resolved
    /// in [`WebGpuDevice::new`], instead of discovered via a fatal wgpu
    /// validation abort inside `create_buffer`.
    pub fn alloc(&self, bytes: usize) -> WebGpuResult<u64> {
        self.ensure_device_alive()?;

        if bytes == 0 {
            return Err(WebGpuError::InvalidArgument(
                "alloc: cannot allocate a zero-byte buffer".into(),
            ));
        }

        let size = (bytes as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);

        let limits = self.device.limits();
        if size > limits.max_buffer_size || size > limits.max_storage_buffer_binding_size {
            return Err(WebGpuError::OutOfMemory);
        }

        let buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-webgpu-buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Defensive: the checks above should make wgpu's own validation a
        // no-op, but if some constraint we did not anticipate fires anyway,
        // surface it as a typed error instead of handing back a handle to a
        // buffer wgpu silently rejected.
        self.check_uncaptured_error()?;

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
        self.ensure_device_alive()?;

        let buffers = self
            .buffers
            .lock()
            .map_err(|_| WebGpuError::BufferMapping("mutex poisoned".into()))?;

        let buf_info = buffers
            .get(&handle)
            .ok_or_else(|| WebGpuError::InvalidArgument(format!("unknown handle {handle}")))?;

        // Reject oversize uploads before touching wgpu: `Queue::write_buffer`
        // validates `offset + src.len() <= buffer.size` and, with no custom
        // uncaptured-error handler installed, an overrun aborts the process via
        // wgpu's default fatal handler.  Surface it as a clean typed error.
        if src.len() as u64 > buf_info.size {
            return Err(WebGpuError::InvalidArgument(format!(
                "copy_to_device: source is {} bytes but buffer holds only {} bytes",
                src.len(),
                buf_info.size
            )));
        }

        // `Queue::write_buffer` separately validates the *copy size* itself
        // against `wgpu::COPY_BUFFER_ALIGNMENT` (4 bytes) — independent of
        // the destination buffer's own (already alignment-padded) size, a
        // 3-byte write into a legally allocated 4-byte buffer is still
        // rejected ("Copy size 3 does not respect COPY_BUFFER_ALIGNMENT").
        // Pad the write up to the alignment with zero bytes when needed;
        // `alloc()` guarantees the destination buffer is at least
        // `src.len()` rounded up to that same alignment, so this never
        // overruns it. The common (already-aligned) case takes the
        // zero-copy path.
        if src.len() as u64 % wgpu::COPY_BUFFER_ALIGNMENT == 0 {
            self.device.queue.write_buffer(&buf_info.buffer, 0, src);
        } else {
            let padded_len =
                (src.len() as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) as usize;
            let mut padded = vec![0u8; padded_len];
            padded[..src.len()].copy_from_slice(src);
            self.device.queue.write_buffer(&buf_info.buffer, 0, &padded);
        }
        drop(buffers);

        self.check_uncaptured_error()?;
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
    /// Only the bytes actually requested (`dst.len()`, rounded up to
    /// `wgpu::COPY_BUFFER_ALIGNMENT`) are staged and copied — previously this
    /// always staged and DMA'd the *entire* source buffer regardless of how
    /// much the caller asked for, so e.g. reading a single scalar out of a
    /// multi-MiB reduction output moved the whole buffer across the copy
    /// engine to deliver 4 bytes.
    ///
    /// Uses a temporary `MAP_READ` staging buffer and blocks — bounded by a
    /// generous internal timeout, see [`WebGpuError::Timeout`] — until the
    /// GPU work completes.
    pub fn copy_from_device(&self, dst: &mut [u8], handle: u64) -> WebGpuResult<()> {
        self.ensure_device_alive()?;

        // Phase 1: acquire the lock, build a staging buffer sized to exactly
        // what the caller asked for, and submit the copy.  The lock is
        // dropped at the end of this block so that `wait_for_gpu` (Phase 2)
        // does not hold the mutex.
        let (staging, submission_index) = {
            let buffers = self
                .buffers
                .lock()
                .map_err(|_| WebGpuError::BufferMapping("mutex poisoned".into()))?;

            let buf_info = buffers
                .get(&handle)
                .ok_or_else(|| WebGpuError::InvalidArgument(format!("unknown handle {handle}")))?;

            // Match the `copy_dtoh` contract of the CPU reference backend: an
            // oversized destination is a sizing error, not something to
            // silently paper over by truncating (which would leave the tail
            // of `dst` stale while still reporting success). A destination
            // *smaller* than the buffer is the intended "sized-by-dst" read,
            // and is exactly the case this narrowed staging path optimises.
            if dst.len() as u64 > buf_info.size {
                return Err(WebGpuError::InvalidArgument(format!(
                    "copy_from_device: destination is {} bytes but buffer holds only {} bytes",
                    dst.len(),
                    buf_info.size
                )));
            }

            // Round up to the copy-alignment requirement. `.min(buf_info.size)`
            // is a defensive bound that is a no-op today: `alloc()` always
            // stores a 4-byte-aligned physical size, and the oversize check
            // above already guarantees `dst.len() <= buf_info.size`, so the
            // rounded value can never exceed it.
            let copy_len = (dst.len() as u64)
                .next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
                .min(buf_info.size);

            let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oxicuda-webgpu-staging"),
                size: copy_len,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder =
                self.device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("oxicuda-webgpu-readback"),
                    });

            encoder.copy_buffer_to_buffer(&buf_info.buffer, 0, &staging, 0, copy_len);
            let submission_index = self.device.queue.submit(std::iter::once(encoder.finish()));

            (staging, submission_index)
            // Mutex guard dropped here — lock released before the wait.
        };

        // Phase 2: map the staging buffer and read the data back to the host.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // Ignore send errors — the receiver may have been dropped.
            let _ = tx.send(result);
        });

        // Block the calling thread until this specific submission (the copy,
        // and — by queue-FIFO-ordering — everything it depends on) completes,
        // or the bounded timeout elapses.
        self.wait_for_gpu(submission_index)?;
        self.check_uncaptured_error()?;

        rx.recv()
            .map_err(|_| WebGpuError::BufferMapping("channel closed before map completed".into()))?
            .map_err(|e| WebGpuError::BufferMapping(format!("{e:?}")))?;

        let data = slice.get_mapped_range();
        let data_len = data.len() as u64;
        // Belt-and-braces: the staging buffer was created at exactly
        // `copy_len` and `copy_len >= dst.len()` was established above, so
        // this should never trip — but return a clean error rather than
        // panic on an out-of-bounds slice if some future change violates
        // that invariant.
        if data_len < dst.len() as u64 {
            drop(data);
            staging.unmap();
            return Err(WebGpuError::BufferMapping(format!(
                "copy_from_device: mapped range is {data_len} bytes but the destination needs {} bytes",
                dst.len(),
            )));
        }
        dst.copy_from_slice(&data[..dst.len()]);
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

    #[test]
    fn copy_to_device_oversize_errors() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = WebGpuMemoryManager::new(dev);
        let h = mm.alloc(16).expect("alloc 16 bytes");
        // 64 bytes into a 16-byte buffer must return a clean error, not panic.
        let err = mm.copy_to_device(h, &[0u8; 64]).unwrap_err();
        assert!(matches!(err, WebGpuError::InvalidArgument(_)));
        mm.free(h).expect("free");
    }

    #[test]
    fn copy_from_device_oversize_dst_errors() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = WebGpuMemoryManager::new(dev);
        let h = mm.alloc(16).expect("alloc 16 bytes");
        // Destination larger than the source buffer must error rather than
        // silently truncate and report success.
        let mut dst = vec![0u8; 64];
        let err = mm.copy_from_device(&mut dst, h).unwrap_err();
        assert!(matches!(err, WebGpuError::InvalidArgument(_)));
        mm.free(h).expect("free");
    }

    #[test]
    fn alloc_rejects_zero_bytes() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = WebGpuMemoryManager::new(dev);
        let err = mm.alloc(0).unwrap_err();
        assert!(matches!(err, WebGpuError::InvalidArgument(_)));
    }

    /// A non-multiple-of-4 allocation (e.g. 3 f16 elements = 6 bytes) must
    /// round up cleanly and round-trip a full copy_htod/copy_dtoh, instead of
    /// hitting wgpu's fatal validation path on the first readback.
    #[test]
    fn alloc_odd_size_roundtrips() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = WebGpuMemoryManager::new(dev);

        for &n in &[3usize, 6, 7] {
            let src: Vec<u8> = (0..n as u8).collect();
            let h = mm
                .alloc(n)
                .unwrap_or_else(|e| panic!("alloc({n}) failed: {e}"));
            mm.copy_to_device(h, &src).expect("copy_to_device");

            let mut dst = vec![0u8; n];
            mm.copy_from_device(&mut dst, h).expect("copy_from_device");
            assert_eq!(src, dst, "roundtrip mismatch for a {n}-byte allocation");

            mm.free(h).expect("free");
        }
    }

    #[test]
    fn alloc_rejects_oversize_allocation() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let too_big = dev.limits().max_buffer_size.saturating_add(4);
        let mm = WebGpuMemoryManager::new(Arc::clone(&dev));
        let err = mm.alloc(too_big as usize).unwrap_err();
        assert!(matches!(err, WebGpuError::OutOfMemory));
    }

    /// A destination smaller than the source buffer is the intended
    /// "sized-by-dst" read: only the first `dst.len()` bytes should come
    /// back, correctly, even though only that narrower range is now staged
    /// and DMA'd (previously the whole buffer was always copied).
    #[test]
    fn copy_from_device_reads_only_requested_prefix() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = WebGpuMemoryManager::new(dev);

        let src: Vec<u8> = (0..=255u8).collect(); // 256 distinct bytes.
        let h = mm.alloc(src.len()).expect("alloc 256 bytes");
        mm.copy_to_device(h, &src).expect("copy_to_device");

        let mut dst = vec![0u8; 8];
        mm.copy_from_device(&mut dst, h).expect("copy_from_device");
        assert_eq!(dst, src[..8]);

        mm.free(h).expect("free");
    }

    #[test]
    fn check_uncaptured_error_surfaces_recorded_error() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = WebGpuMemoryManager::new(Arc::clone(&dev));

        // Trigger a real uncaptured wgpu error directly against the raw
        // device (bypassing `alloc()`'s own guards) to prove
        // `check_uncaptured_error` surfaces it as a typed `Err`.
        let _bogus = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-webgpu-test-oversize"),
            size: u64::MAX,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let err = mm.check_uncaptured_error().unwrap_err();
        assert!(matches!(err, WebGpuError::UncapturedError(_)));
    }

    #[test]
    fn operations_fail_fast_once_device_is_lost() {
        let Some(dev) = try_get_device() else {
            return;
        };
        let mm = WebGpuMemoryManager::new(Arc::clone(&dev));

        dev.device.destroy();
        for _ in 0..20 {
            if dev.is_device_lost() {
                break;
            }
            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(dev.is_device_lost(), "precondition: device should be lost");

        let err = mm.alloc(64).unwrap_err();
        assert!(matches!(err, WebGpuError::DeviceLost(_)));
    }

    #[test]
    fn poll_timeout_maps_to_webgpu_timeout_error() {
        let err = poll_result_to_webgpu_result(Err(wgpu::PollError::Timeout)).unwrap_err();
        assert!(matches!(err, WebGpuError::Timeout));
    }

    #[test]
    fn poll_wrong_submission_index_maps_to_buffer_mapping_error() {
        let err = poll_result_to_webgpu_result(Err(wgpu::PollError::WrongSubmissionIndex(2, 1)))
            .unwrap_err();
        assert!(matches!(err, WebGpuError::BufferMapping(_)));
    }

    #[test]
    fn poll_ok_maps_to_ok() {
        assert!(poll_result_to_webgpu_result(Ok(wgpu::PollStatus::QueueEmpty)).is_ok());
    }
}
