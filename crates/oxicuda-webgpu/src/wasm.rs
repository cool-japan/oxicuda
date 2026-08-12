//! WASM target support for browser-based GPU compute via WebGPU.
//!
//! This module is conditionally compiled on `wasm32` targets (or when the `wasm`
//! feature is enabled for native testing) and provides browser-friendly wrappers
//! around the `wgpu` WebGPU backend.
//!
//! # Architecture
//!
//! ```text
//! +-------------------------------------------+
//! |         JavaScript / Browser              |
//! +-------------------+-----------------------+
//!                     |
//! +-------------------v-----------------------+
//! |   WasmGpuDevice / WasmBackend (wasm32)    |
//! +-------------------+-----------------------+
//!                     |  delegates to
//! +-------------------v-----------------------+
//! |   WebGpuBackend (wgpu web-sys backend)    |
//! +-------------------------------------------+
//! ```
//!
//! # Usage
//!
//! The [`WasmBackend`] wraps the existing [`WebGpuBackend`]
//! and adds browser-specific initialisation methods such as
//! [`init_from_canvas`](WasmBackend::init_from_canvas).
//!
//! The [`WasmMemoryManager`] provides async-friendly buffer staging suited to the
//! browser event loop.
//!
//! # Known limitation: `WasmBackend` does not actually use `WasmMemoryManager`
//!
//! Despite the name, **every [`WasmBackend`] compute and memory operation
//! forwards to [`WebGpuBackend`]**, which is backed by
//! [`WebGpuMemoryManager`](crate::memory::WebGpuMemoryManager) — not by
//! [`WasmMemoryManager`] in this module.  `WasmMemoryManager` (and
//! [`WasmGpuDevice`]) exist, are async-safe, and are unit-tested, but nothing
//! in `WasmBackend` constructs or calls them.  This matters because:
//!
//! * `WebGpuBackend`'s compute methods end in a **blocking**
//!   `Device::poll(PollType::wait_indefinitely())` (see `backend.rs`), and
//!   `WebGpuMemoryManager::copy_from_device`'s readback blocks on an
//!   `mpsc::channel` `recv()` that only resolves once that same poll drives
//!   the `map_async` callback to completion. On the single-threaded browser
//!   main thread, `Device::poll` cannot make progress without yielding back
//!   to the event loop that would deliver that callback — so this **would
//!   deadlock the tab**, exactly the failure `WasmMemoryManager::copy_dtoh`'s
//!   own `#[cfg(target_arch = "wasm32")]` guard (in this file) already exists
//!   to prevent, just on the wrong type.
//! * Fixing this by swapping `WasmBackend`'s memory calls (`alloc`,
//!   `copy_htod`, `copy_dtoh`) over to `WasmMemoryManager` while leaving the
//!   compute calls (`gemm`, `unary`, …) on `self.inner: WebGpuBackend` is
//!   **not a valid partial fix**: the two memory managers keep independent
//!   `HashMap<u64, Buffer>` handle tables, each with its own
//!   `next_handle`/`AtomicU64` counter starting at 1. A buffer allocated
//!   through `WasmMemoryManager` would not exist in
//!   `WebGpuMemoryManager`'s map (or worse, its handle number would collide
//!   with an unrelated `WebGpuMemoryManager` buffer), so every compute call
//!   would either fail with "unknown handle" or silently operate on the
//!   wrong buffer.  Correctly fixing this requires `WasmBackend` to own one
//!   coherent device + buffer table end-to-end and give every compute
//!   dispatch (not just readback) an async, non-blocking form — a genuine
//!   rework of this module and `backend.rs` together, out of scope here.
//!
//! Until that rework lands: `WasmBackend` is appropriate for **native
//! testing** of the WASM code paths (via the `wasm` feature, where blocking
//! is safe) and for **non-browser wasm32 hosts** (e.g. a WASI runtime driving
//! its own event loop outside a browser tab). On an actual browser main
//! thread, treat every `WasmBackend` compute/readback call as unsafe to call
//! synchronously; [`WasmMemoryManager::copy_dtoh_async`] is the
//! already-implemented pattern a real async rework would extend to the rest
//! of the surface.
//!
//! # Deeper pre-existing gap: this crate does not compile for `wasm32-unknown-unknown` at all
//!
//! `cargo check -p oxicuda-webgpu --target wasm32-unknown-unknown` fails
//! today (independent of anything in this module): `oxicuda_backend::
//! ComputeBackend` requires `Send + Sync`, but on the real `wasm32` target
//! `wgpu`'s WebGPU backend represents `Device`/`Buffer` using `Rc`/`RefCell`
//! internally (browser JS handles are not thread-safe), which makes
//! `WebGpuDevice`/`WebGpuBufferInfo` — and therefore `WebGpuBackend`, which
//! every `WasmBackend` method forwards to — not `Send`. So `WasmBackend`
//! (and `WebGpuBackend`) cannot implement `ComputeBackend` on that target
//! today at all; this is a compile error, not a runtime one, so nothing in
//! this crate has ever actually run compiled-for-wasm32. Fixing it needs
//! either a `?Send` carve-out on `ComputeBackend` for wasm32 (a change to
//! `oxicuda-backend`, out of scope for this crate) or a non-`ComputeBackend`
//! wasm32-native entry point built directly on `WasmGpuDevice`; both are
//! part of the same async rework noted above.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use oxicuda_backend::{
    BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};

use crate::WebGpuBackend;
use crate::error::{WebGpuError, WebGpuResult};
use crate::memory::WebGpuBufferInfo;

// ---- WasmGpuDevice --------------------------------------------------------

/// A WebGPU device obtained from the browser's `navigator.gpu` API.
///
/// Wraps the `wgpu` adapter and device objects and provides async construction
/// methods appropriate for the browser environment.
#[derive(Debug)]
pub struct WasmGpuDevice {
    /// The wgpu instance.
    #[allow(dead_code)]
    pub(crate) instance: wgpu::Instance,
    /// The selected GPU adapter.
    #[allow(dead_code)]
    pub(crate) adapter: wgpu::Adapter,
    /// The logical device.
    pub(crate) device: wgpu::Device,
    /// The queue for submitting command buffers.
    pub(crate) queue: wgpu::Queue,
    /// Human-readable adapter name.
    pub adapter_name: String,
}

impl WasmGpuDevice {
    /// Create a new [`WasmGpuDevice`] from an already-obtained adapter.
    ///
    /// This is the async path used by browser callers. On native targets this
    /// may not be exercised directly, but it is the intended entry point for
    /// WASM builds.
    pub async fn from_adapter(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
    ) -> WebGpuResult<Self> {
        let adapter_name = adapter.get_info().name.clone();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("oxicuda-webgpu-wasm"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| WebGpuError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            adapter_name,
        })
    }
}

// ---- request_adapter -------------------------------------------------------

/// Request a WebGPU adapter from the browser.
///
/// On `wasm32` this goes through the browser's `navigator.gpu` API via the
/// `wgpu` web-sys backend.
pub async fn request_adapter() -> WebGpuResult<wgpu::Adapter> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| WebGpuError::DeviceRequest(e.to_string()))
}

// ---- WasmMemoryManager -----------------------------------------------------

/// Browser-side buffer manager that uses async `map_async` staging.
///
/// This mirrors [`WebGpuMemoryManager`](crate::memory::WebGpuMemoryManager) but
/// is designed to work within the single-threaded browser event loop where
/// blocking calls are not allowed.
pub struct WasmMemoryManager {
    device: Arc<WasmGpuDevice>,
    buffers: Mutex<HashMap<u64, WebGpuBufferInfo>>,
    next_handle: AtomicU64,
}

impl WasmMemoryManager {
    /// Create a new WASM memory manager backed by `device`.
    pub fn new(device: Arc<WasmGpuDevice>) -> Self {
        Self {
            device,
            buffers: Mutex::new(HashMap::new()),
            next_handle: AtomicU64::new(1),
        }
    }

    /// Allocate a device buffer of `bytes` bytes.
    pub fn alloc(&self, bytes: usize) -> WebGpuResult<u64> {
        let size = bytes as u64;
        let buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-wasm-buffer"),
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

    /// Free the buffer identified by `handle`.
    pub fn free(&self, handle: u64) -> WebGpuResult<()> {
        self.buffers
            .lock()
            .map_err(|_| WebGpuError::BufferMapping("mutex poisoned".into()))?
            .remove(&handle);
        Ok(())
    }

    /// Upload host bytes to the device buffer (host-to-device copy).
    ///
    /// Uses `Queue::write_buffer` which is available in both native and WASM.
    pub fn copy_htod(&self, handle: u64, src: &[u8]) -> WebGpuResult<()> {
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| WebGpuError::BufferMapping("mutex poisoned".into()))?;

        let buf_info = buffers
            .get(&handle)
            .ok_or_else(|| WebGpuError::InvalidArgument(format!("unknown handle {handle}")))?;

        // `Queue::write_buffer` validates `offset + src.len() <= buffer.size`;
        // an overrun is delivered to wgpu's default uncaptured-error handler
        // which panics.  Reject it up front with a typed error instead.
        if src.len() as u64 > buf_info.size {
            return Err(WebGpuError::InvalidArgument(format!(
                "copy_htod: source is {} bytes but buffer holds only {} bytes",
                src.len(),
                buf_info.size
            )));
        }

        self.device.queue.write_buffer(&buf_info.buffer, 0, src);
        Ok(())
    }

    /// Submit a device→staging copy and return the mappable staging buffer.
    ///
    /// Shared by the synchronous [`copy_dtoh`](Self::copy_dtoh) and asynchronous
    /// [`copy_dtoh_async`](Self::copy_dtoh_async) readback paths.
    fn submit_readback(&self, handle: u64) -> WebGpuResult<wgpu::Buffer> {
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| WebGpuError::BufferMapping("mutex poisoned".into()))?;

        let buf_info = buffers
            .get(&handle)
            .ok_or_else(|| WebGpuError::InvalidArgument(format!("unknown handle {handle}")))?;

        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-wasm-staging"),
            size: buf_info.size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("oxicuda-wasm-readback"),
                });

        encoder.copy_buffer_to_buffer(&buf_info.buffer, 0, &staging, 0, buf_info.size);
        self.device.queue.submit(std::iter::once(encoder.finish()));

        Ok(staging)
    }

    /// Copy the mapped staging data into `dst`, enforcing the sized-by-`dst`
    /// contract: an oversized destination is rejected rather than silently
    /// truncated (which would leave `dst`'s tail stale while reporting success).
    fn drain_mapped(dst: &mut [u8], staging: wgpu::Buffer) -> WebGpuResult<()> {
        let slice = staging.slice(..);
        let data = slice.get_mapped_range();
        let data_len = data.len();
        if dst.len() > data_len {
            drop(data);
            staging.unmap();
            return Err(WebGpuError::InvalidArgument(format!(
                "copy_dtoh: destination is {} bytes but buffer holds only {data_len} bytes",
                dst.len(),
            )));
        }
        let copy_len = dst.len();
        dst[..copy_len].copy_from_slice(&data[..copy_len]);
        drop(data);
        staging.unmap();
        Ok(())
    }

    /// Download a device buffer to host bytes (device-to-host copy).
    ///
    /// This is a *blocking* readback: it maps a staging buffer and waits on the
    /// map callback.  On a native target (including the `wasm` feature used for
    /// testing) `Device::poll` drives completion, so this works.  On the real
    /// `wasm32` browser main thread, however, blocking would starve the event
    /// loop that delivers the map callback and freeze the tab — so there this
    /// method returns [`WebGpuError::Unsupported`] and callers must use
    /// [`copy_dtoh_async`](Self::copy_dtoh_async) instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn copy_dtoh(&self, dst: &mut [u8], handle: u64) -> WebGpuResult<()> {
        let staging = self.submit_readback(handle)?;

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let _ = self.device.device.poll(wgpu::PollType::wait_indefinitely());

        rx.recv()
            .map_err(|_| WebGpuError::BufferMapping("channel closed before map completed".into()))?
            .map_err(|e| WebGpuError::BufferMapping(format!("{e:?}")))?;

        Self::drain_mapped(dst, staging)
    }

    /// Blocking readback is unavailable on the `wasm32` browser main thread — it
    /// would deadlock the single event loop that delivers the buffer-map
    /// callback.  Use [`copy_dtoh_async`](Self::copy_dtoh_async) instead.
    #[cfg(target_arch = "wasm32")]
    pub fn copy_dtoh(&self, _dst: &mut [u8], _handle: u64) -> WebGpuResult<()> {
        Err(WebGpuError::Unsupported(
            "synchronous copy_dtoh would deadlock the browser event loop; \
             use copy_dtoh_async"
                .into(),
        ))
    }

    /// Asynchronously download a device buffer to host bytes.
    ///
    /// Unlike [`copy_dtoh`](Self::copy_dtoh) this never blocks the calling
    /// thread: it awaits the buffer-map completion via a future resolved from
    /// the `map_async` callback, making it the correct readback path on the
    /// single-threaded browser event loop.  Like `copy_dtoh`, an oversized
    /// destination is rejected with [`WebGpuError::InvalidArgument`].
    pub async fn copy_dtoh_async(&self, dst: &mut [u8], handle: u64) -> WebGpuResult<()> {
        let staging = self.submit_readback(handle)?;

        {
            let slice = staging.slice(..);
            let state = Arc::new(Mutex::new(MapState::default()));
            let cb_state = Arc::clone(&state);
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let mut guard = match cb_state.lock() {
                    Ok(g) => g,
                    Err(poisoned) => poisoned.into_inner(),
                };
                guard.result = Some(result);
                if let Some(waker) = guard.waker.take() {
                    waker.wake();
                }
            });

            // Nudge the device once so the callback can be delivered on native
            // executors; in the browser the event loop drives this itself.
            let _ = self.device.device.poll(wgpu::PollType::wait_indefinitely());

            MapWait { state }
                .await
                .map_err(|e| WebGpuError::BufferMapping(format!("{e:?}")))?;
        }

        Self::drain_mapped(dst, staging)
    }
}

/// Shared state between a `map_async` callback and the [`MapWait`] future that
/// awaits it.
#[derive(Default)]
struct MapState {
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    waker: Option<std::task::Waker>,
}

/// A minimal future that resolves once a `map_async` callback has stored its
/// result.  Used by [`WasmMemoryManager::copy_dtoh_async`] to await a buffer
/// map without blocking the browser event loop.
struct MapWait {
    state: Arc<Mutex<MapState>>,
}

impl std::future::Future for MapWait {
    type Output = Result<(), wgpu::BufferAsyncError>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let mut guard = match self.state.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(result) = guard.result.take() {
            std::task::Poll::Ready(result)
        } else {
            guard.waker = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }
}

impl std::fmt::Debug for WasmMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.buffers.lock().map(|b| b.len()).unwrap_or(0);
        write!(f, "WasmMemoryManager(buffers={count})")
    }
}

// ---- WasmBackend -----------------------------------------------------------

/// WebGPU compute backend for WASM (browser) targets.
///
/// Wraps [`WebGpuBackend`] and adds browser-specific initialisation paths.
/// Implements [`ComputeBackend`] by delegating **every** operation —
/// compute, allocation, and readback — to the inner [`WebGpuBackend`].
///
/// # Notes
///
/// See the [module-level documentation](self) for why this delegation makes
/// every blocking call (readback, and every compute op via its trailing
/// `Device::poll`) unsafe to call synchronously from a real browser main
/// thread today, why `WasmMemoryManager` cannot simply be swapped in as a
/// partial fix (it would split the buffer-handle table in two), and what a
/// correct fix requires.
#[derive(Debug)]
pub struct WasmBackend {
    inner: WebGpuBackend,
}

impl WasmBackend {
    /// Create a new, uninitialised WASM backend.
    pub fn new() -> Self {
        Self {
            inner: WebGpuBackend::new(),
        }
    }

    /// Initialise the backend from an HTML canvas element by ID.
    ///
    /// This is the recommended browser entry point. The canvas is not used for
    /// rendering but is required by some WebGPU implementations to obtain a
    /// valid adapter.
    ///
    /// # Errors
    ///
    /// Returns an error if no WebGPU adapter is available or device creation fails.
    pub async fn init_from_canvas(_canvas_id: &str) -> Result<Self, WebGpuError> {
        // In the browser, wgpu's web-sys backend goes through navigator.gpu
        // which does not actually require a canvas for compute-only usage.
        // We accept the canvas_id for forward compatibility (e.g. surface-based
        // adapters) but currently initialise via the standard path.
        let mut backend = Self::new();
        backend
            .inner
            .init()
            .map_err(|e| WebGpuError::DeviceRequest(e.to_string()))?;
        Ok(backend)
    }
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ---- ComputeBackend for WasmBackend ----------------------------------------

impl ComputeBackend for WasmBackend {
    fn name(&self) -> &str {
        "webgpu-wasm"
    }

    fn init(&mut self) -> BackendResult<()> {
        self.inner.init()
    }

    fn is_initialized(&self) -> bool {
        self.inner.is_initialized()
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        trans_a: BackendTranspose,
        trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        lda: usize,
        b_ptr: u64,
        ldb: usize,
        beta: f64,
        c_ptr: u64,
        ldc: usize,
    ) -> BackendResult<()> {
        self.inner.gemm(
            trans_a, trans_b, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc,
        )
    }

    // `batched_gemm` MUST be forwarded explicitly (finding webgpu-7): without
    // this override, `WasmBackend` inherits `ComputeBackend`'s default
    // `batched_gemm` (`oxicuda-backend/src/lib.rs`), which loops calling
    // `self.gemm(...)` with `a_ptr + b * stride_a * elem_bytes`-style pointer
    // *arithmetic* on `a_ptr`/`b_ptr`/`c_ptr`.  Those are not addresses here —
    // `WebGpuMemoryManager::alloc` (this backend's memory manager) hands out
    // opaque monotonic `u64` handles from a `HashMap<u64, Buffer>`, so adding
    // a stride offset to one either misses the map entirely (`batch_count >=
    // 2` fails with "unknown handle") or, worse, silently collides with an
    // unrelated live handle and multiplies the wrong buffers.  `batch_count
    // == 1` happens to work by accident (offset 0), which is why this is easy
    // to miss in ad hoc testing.  `WebGpuBackend::batched_gemm` (this
    // backend's `self.inner`) already implements the real batched-strided
    // dispatch correctly; this is purely a missing delegation, mirroring
    // every other method in this `impl` block.
    #[allow(clippy::too_many_arguments)]
    fn batched_gemm(
        &self,
        trans_a: BackendTranspose,
        trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        lda: usize,
        stride_a: usize,
        b_ptr: u64,
        ldb: usize,
        stride_b: usize,
        beta: f64,
        c_ptr: u64,
        ldc: usize,
        stride_c: usize,
        batch_count: usize,
    ) -> BackendResult<()> {
        self.inner.batched_gemm(
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a_ptr,
            lda,
            stride_a,
            b_ptr,
            ldb,
            stride_b,
            beta,
            c_ptr,
            ldc,
            stride_c,
            batch_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d_forward(
        &self,
        input_ptr: u64,
        input_shape: &[usize],
        filter_ptr: u64,
        filter_shape: &[usize],
        output_ptr: u64,
        output_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> BackendResult<()> {
        self.inner.conv2d_forward(
            input_ptr,
            input_shape,
            filter_ptr,
            filter_shape,
            output_ptr,
            output_shape,
            stride,
            padding,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        o_ptr: u64,
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_kv: usize,
        head_dim: usize,
        scale: f64,
        causal: bool,
    ) -> BackendResult<()> {
        self.inner.attention(
            q_ptr, k_ptr, v_ptr, o_ptr, batch, heads, seq_q, seq_kv, head_dim, scale, causal,
        )
    }

    fn reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        self.inner.reduce(op, input_ptr, output_ptr, shape, axis)
    }

    fn unary(&self, op: UnaryOp, input_ptr: u64, output_ptr: u64, n: usize) -> BackendResult<()> {
        self.inner.unary(op, input_ptr, output_ptr, n)
    }

    fn binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        self.inner.binary(op, a_ptr, b_ptr, output_ptr, n)
    }

    fn synchronize(&self) -> BackendResult<()> {
        self.inner.synchronize()
    }

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        self.inner.alloc(bytes)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.inner.free(ptr)
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        self.inner.copy_htod(dst, src)
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        self.inner.copy_dtoh(dst, src)
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_backend::BackendError;

    /// Basic compilation test: the wasm module types exist and are constructible.
    #[test]
    fn wasm_module_compiles() {
        let backend = WasmBackend::new();
        assert!(!backend.is_initialized());
        assert_eq!(backend.name(), "webgpu-wasm");

        // Debug impl works.
        let debug_str = format!("{backend:?}");
        assert!(debug_str.contains("WasmBackend"));
    }

    /// Verify conditional compilation: wasm types implement expected traits.
    #[test]
    fn wasm_feature_flag_gating() {
        // WasmBackend implements ComputeBackend.
        let backend = WasmBackend::new();
        let _: &dyn ComputeBackend = &backend;

        // WasmBackend implements Default.
        let _default = WasmBackend::default();
    }

    /// All public types and functions are accessible when `wasm` feature is enabled.
    #[test]
    fn wasm_public_api_accessible() {
        // WasmGpuDevice is a public type.
        fn _assert_wasm_gpu_device_exists(_: &WasmGpuDevice) {}

        // WasmMemoryManager is a public type.
        fn _assert_wasm_memory_manager_exists(_: &WasmMemoryManager) {}

        // WasmBackend is a public type with new() and default().
        let _b = WasmBackend::new();
        let _b2 = WasmBackend::default();

        // request_adapter is a public async fn (we can reference it).
        let _fn_ptr: fn() -> _ = || request_adapter();
    }

    /// Not-initialised guards return proper errors.
    #[test]
    fn wasm_backend_not_initialized_guards() {
        let b = WasmBackend::new();
        assert_eq!(b.alloc(1024), Err(BackendError::NotInitialized));
        assert_eq!(b.free(1), Err(BackendError::NotInitialized));
        assert_eq!(b.copy_htod(1, b"hello"), Err(BackendError::NotInitialized));

        let mut buf = [0u8; 4];
        assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
        assert_eq!(b.synchronize(), Err(BackendError::NotInitialized));
    }

    /// Init may fail gracefully (no GPU) but must not panic.
    #[test]
    fn wasm_backend_init_graceful() {
        let mut b = WasmBackend::new();
        let _result = b.init();
    }

    /// Try to build a `WasmGpuDevice` for device-backed tests; returns `None`
    /// when no adapter is available so the test skips gracefully.
    fn try_wasm_device() -> Option<Arc<WasmGpuDevice>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let dev = pollster::block_on(WasmGpuDevice::from_adapter(instance, adapter)).ok()?;
        Some(Arc::new(dev))
    }

    /// Oversized host→device upload must return a typed error, not panic via
    /// wgpu's default uncaptured-error handler.
    #[test]
    fn wasm_copy_htod_oversize_errors() {
        let Some(dev) = try_wasm_device() else {
            return;
        };
        let mm = WasmMemoryManager::new(dev);
        let h = mm.alloc(16).expect("alloc 16 bytes");
        let err = mm.copy_htod(h, &[0u8; 64]).unwrap_err();
        assert!(matches!(err, WebGpuError::InvalidArgument(_)));
        mm.free(h).expect("free");
    }

    /// Device→host readback into an oversized destination must error rather than
    /// silently truncate and report success.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn wasm_copy_dtoh_oversize_dst_errors() {
        let Some(dev) = try_wasm_device() else {
            return;
        };
        let mm = WasmMemoryManager::new(dev);
        let h = mm.alloc(16).expect("alloc 16 bytes");
        let mut dst = vec![0u8; 64];
        let err = mm.copy_dtoh(&mut dst, h).unwrap_err();
        assert!(matches!(err, WebGpuError::InvalidArgument(_)));
        mm.free(h).expect("free");
    }

    /// Try to build an initialised `WasmBackend`; returns `None` when no GPU
    /// is available so device-backed tests skip gracefully.
    fn try_init_wasm_backend() -> Option<WasmBackend> {
        let mut b = WasmBackend::new();
        b.init().ok()?;
        Some(b)
    }

    /// Regression for finding webgpu-7: before `batched_gemm` was forwarded
    /// explicitly, `WasmBackend` inherited `ComputeBackend`'s default
    /// implementation, which does pointer arithmetic
    /// (`a_ptr + batch * stride_a * elem_bytes`) on what this backend's
    /// memory manager hands out as *opaque* monotonic handles — not
    /// addresses.  `batch_count == 1` (offset 0) happened to work by
    /// accident; `batch_count >= 2` did not (either "unknown handle" or,
    /// worse, a silent collision with a different live buffer).  Drive
    /// `batch_count = 2` end-to-end through `WasmBackend` itself and check
    /// the numeric result, proving the override is wired and correct — not
    /// merely present.
    #[test]
    fn wasm_backend_batched_gemm_matches_reference() {
        let Some(wasm_b) = try_init_wasm_backend() else {
            return;
        };

        // 2 batches of 2×2 identity multiply, matching
        // `backend_tests.rs::batched_gemm_identity_2x2`.
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let eye = [1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let c_init = [0.0f32; 8];
        let to_bytes = |d: &[f32]| -> Vec<u8> { d.iter().flat_map(|v| v.to_le_bytes()).collect() };

        let a_h = wasm_b.alloc(32).expect("alloc a");
        let b_h = wasm_b.alloc(32).expect("alloc b");
        let c_h = wasm_b.alloc(32).expect("alloc c");
        wasm_b.copy_htod(a_h, &to_bytes(&a)).expect("htod a");
        wasm_b.copy_htod(b_h, &to_bytes(&eye)).expect("htod b");
        wasm_b.copy_htod(c_h, &to_bytes(&c_init)).expect("htod c");

        let nt = BackendTranspose::NoTrans;
        wasm_b
            .batched_gemm(
                nt, nt, 2, 2, 2, 1.0, a_h, 2, 4, b_h, 2, 4, 0.0, c_h, 2, 4,
                2, // batch_count >= 2 — the case the missing override broke.
            )
            .expect("wasm batched_gemm");

        let mut result_bytes = vec![0u8; 32];
        wasm_b
            .copy_dtoh(&mut result_bytes, c_h)
            .expect("dtoh result");
        let result: Vec<f32> = result_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // C = A * I = A for both batches.
        for (r, e) in result.iter().zip(a.iter()) {
            assert!((r - e).abs() < 1e-5, "got {r}, expected {e}");
        }

        wasm_b.free(a_h).expect("free");
        wasm_b.free(b_h).expect("free");
        wasm_b.free(c_h).expect("free");
    }
}
