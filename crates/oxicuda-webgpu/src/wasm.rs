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

        self.device.queue.write_buffer(&buf_info.buffer, 0, src);
        Ok(())
    }

    /// Download device buffer to host bytes (device-to-host copy).
    ///
    /// Uses a staging buffer with `map_async` and blocks via `pollster::block_on`.
    /// On native WASM, callers should prefer the async variant or schedule this
    /// on a web worker to avoid blocking the main thread.
    pub fn copy_dtoh(&self, dst: &mut [u8], handle: u64) -> WebGpuResult<()> {
        let staging = {
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

            staging
        };

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

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
/// Implements [`ComputeBackend`] by delegating all compute operations to the
/// inner [`WebGpuBackend`], which already supports WASM via `wgpu`'s web-sys
/// backend.
///
/// # Notes
///
/// Synchronous [`ComputeBackend`] trait methods use `pollster::block_on` to
/// bridge async wgpu calls. In production browser deployments, prefer using
/// the async initialisation helpers directly and scheduling GPU work on web
/// workers where blocking is acceptable.
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
}
