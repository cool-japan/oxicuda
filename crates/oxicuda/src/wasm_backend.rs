//! WASM + WebGPU compute backend for browser environments.
//!
//! Wraps [`WebGpuBackend`](oxicuda_webgpu::WebGpuBackend) with WASM-specific bindings so that OxiCUDA's
//! compute API can be used from a browser via WebAssembly.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │         JavaScript / Browser            │
//! └─────────────────┬───────────────────────┘
//!                   │  wasm-bindgen ABI
//! ┌─────────────────▼───────────────────────┐
//! │       WasmComputeBackend (wasm32)       │
//! │  - #[wasm_bindgen] exported API         │
//! └─────────────────┬───────────────────────┘
//!                   │  delegates to
//! ┌─────────────────▼───────────────────────┐
//! │            WebGpuBackend                │
//! │  (wgpu: Vulkan/Metal/DX12/WebGPU)       │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Feature flag
//!
//! This module is behind the `wasm-backend` feature flag.
//!
//! # Platform notes
//!
//! - On `wasm32`: exposes a `#[wasm_bindgen]` constructor and JS-friendly methods.
//! - On native targets: the struct compiles and is fully usable via the
//!   [`ComputeBackend`](oxicuda_backend::ComputeBackend) trait; the wasm_bindgen exports are omitted.
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]

use oxicuda_backend::{
    BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};
use oxicuda_webgpu::WebGpuBackend;

#[cfg(target_arch = "wasm32")]
use js_sys::{Function, Promise};
#[cfg(target_arch = "wasm32")]
use oxicuda_backend::BackendError;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// ─── BrowserComputeError ────────────────────────────────────────────────────

/// A JS-visible error type for browser-facing WASM compute operations.
///
/// Wraps a [`BackendError`] as a plain string message exposed to JavaScript.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct BrowserComputeError {
    message: String,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl BrowserComputeError {
    /// Returns the human-readable error message.
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

#[cfg(target_arch = "wasm32")]
impl From<BackendError> for BrowserComputeError {
    fn from(e: BackendError) -> Self {
        Self {
            message: e.to_string(),
        }
    }
}

// ─── WasmComputeBackend ──────────────────────────────────────────────────────

/// WASM/browser compute backend backed by WebGPU via `wgpu`.
///
/// This struct wraps [`WebGpuBackend`] and provides:
///
/// 1. A full [`ComputeBackend`] trait implementation (works on all targets).
/// 2. On `wasm32`, a `#[wasm_bindgen]`-exported interface with JS-friendly
///    method signatures, enabling direct use from browser JavaScript.
///
/// # Lifecycle
///
/// ```text
/// // Rust / Wasm32
/// let mut backend = WasmComputeBackend::new();
/// backend.init()?;                    // select WebGPU adapter
/// let ptr = backend.alloc(1024)?;     // device allocation
/// backend.synchronize()?;
/// backend.free(ptr)?;
/// ```
///
/// # Browser usage (JavaScript)
///
/// ```javascript
/// // ES module, built with wasm-pack
/// import init, { WasmComputeBackend } from './oxicuda_wasm.js';
/// await init();
/// const backend = new WasmComputeBackend();
/// await WasmComputeBackend.init_browser();
/// const ptr = backend.alloc(1024n);
/// backend.free(ptr);
/// backend.synchronize();
/// ```
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug)]
pub struct WasmComputeBackend {
    inner: WebGpuBackend,
}

// ─── Non-wasm32 inherent methods ────────────────────────────────────────────

impl WasmComputeBackend {
    /// Create a new, uninitialised WASM compute backend.
    ///
    /// On `wasm32` this is also exported as the `#[wasm_bindgen(constructor)]`.
    pub fn new() -> Self {
        Self {
            inner: WebGpuBackend::new(),
        }
    }
}

impl Default for WasmComputeBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ─── wasm32-only inherent methods (JS-facing) ───────────────────────────────

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmComputeBackend {
    /// Create a new, uninitialised WASM compute backend (JS constructor).
    #[wasm_bindgen(constructor)]
    pub fn new_js() -> Self {
        Self::new()
    }

    /// Backend name string (always `"webgpu"`).
    #[wasm_bindgen(js_name = "name")]
    pub fn name_js(&self) -> String {
        self.inner.name().to_owned()
    }

    /// Returns `true` if the backend has been successfully initialised.
    #[wasm_bindgen(js_name = "isInitialized")]
    pub fn is_initialized_js(&self) -> bool {
        self.inner.is_initialized()
    }

    /// Allocate `size` bytes on the GPU and return an opaque device pointer.
    ///
    /// Returns a `BigInt` (u64) on success or throws a JS error string.
    #[wasm_bindgen(js_name = "alloc")]
    pub fn alloc_js(&self, size: usize) -> Result<u64, JsValue> {
        self.inner.alloc(size).map_err(err_to_js)
    }

    /// Free a device allocation previously returned by [`alloc_js`].
    #[wasm_bindgen(js_name = "free")]
    pub fn free_js(&self, ptr: u64) -> Result<(), JsValue> {
        self.inner.free(ptr).map_err(err_to_js)
    }

    /// Block until all pending GPU work on this backend completes.
    #[wasm_bindgen(js_name = "synchronize")]
    pub fn synchronize_js(&self) -> Result<(), JsValue> {
        self.inner.synchronize().map_err(err_to_js)
    }

    /// Asynchronously initialise the WebGPU adapter in the browser.
    ///
    /// Returns a `Promise<void>` that resolves when the backend is ready, or
    /// rejects with an error message string if WebGPU is unavailable.
    ///
    /// This function takes ownership of `self` and moves it into the promise
    /// closure, returning a new backend handle on resolution. For a simpler
    /// synchronous initialisation, call `init()` via the Rust trait instead.
    #[wasm_bindgen(js_name = "initBrowser")]
    pub fn init_browser(&mut self) -> Promise {
        let result = self.inner.init();
        match result {
            Ok(()) => Promise::new(&mut |resolve: Function, _reject: Function| {
                let _ = resolve.call0(&JsValue::UNDEFINED);
            }),
            Err(e) => {
                let msg = JsValue::from_str(&e.to_string());
                Promise::new(&mut |_resolve: Function, reject: Function| {
                    let _ = reject.call1(&JsValue::UNDEFINED, &msg);
                })
            }
        }
    }
}

/// Convert a [`BackendError`] to a JS-compatible [`JsValue`] string.
#[cfg(target_arch = "wasm32")]
fn err_to_js(e: BackendError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

// ─── ComputeBackend trait impl ───────────────────────────────────────────────

impl ComputeBackend for WasmComputeBackend {
    fn name(&self) -> &str {
        self.inner.name()
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_backend::{
        BackendError, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
    };

    // ── Helper: try to initialise the backend, skip test if unavailable ───────

    fn try_init_wasm() -> Option<WasmComputeBackend> {
        let mut b = WasmComputeBackend::new();
        match b.init() {
            Ok(()) => Some(b),
            Err(_) => None,
        }
    }

    // ── 1. Structural tests (no GPU needed) ───────────────────────────────────

    #[test]
    fn wasm_backend_new_not_initialized() {
        let b = WasmComputeBackend::new();
        assert!(!b.is_initialized());
    }

    #[test]
    fn wasm_backend_default_not_initialized() {
        let b = WasmComputeBackend::default();
        assert!(!b.is_initialized());
        assert_eq!(b.name(), "webgpu");
    }

    #[test]
    fn wasm_backend_name_correct() {
        let b = WasmComputeBackend::new();
        assert_eq!(b.name(), "webgpu");
    }

    #[test]
    fn wasm_backend_debug_impl() {
        let b = WasmComputeBackend::new();
        let s = format!("{b:?}");
        assert!(s.contains("WasmComputeBackend"));
    }

    #[test]
    fn wasm_backend_object_safe_check() {
        let backend = WasmComputeBackend::new();
        let _: &dyn ComputeBackend = &backend;
    }

    // ── 2. Not-initialized guards ─────────────────────────────────────────────

    #[test]
    fn alloc_before_init_err() {
        let b = WasmComputeBackend::new();
        assert_eq!(b.alloc(1024), Err(BackendError::NotInitialized));
    }

    #[test]
    fn free_before_init_err() {
        let b = WasmComputeBackend::new();
        assert_eq!(b.free(1), Err(BackendError::NotInitialized));
    }

    #[test]
    fn copy_htod_before_init_err() {
        let b = WasmComputeBackend::new();
        assert_eq!(b.copy_htod(1, b"hello"), Err(BackendError::NotInitialized));
    }

    #[test]
    fn copy_dtoh_before_init_err() {
        let b = WasmComputeBackend::new();
        let mut buf = [0u8; 4];
        assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
    }

    #[test]
    fn synchronize_before_init_err() {
        let b = WasmComputeBackend::new();
        assert_eq!(b.synchronize(), Err(BackendError::NotInitialized));
    }

    #[test]
    fn gemm_before_init_err() {
        let b = WasmComputeBackend::new();
        let result = b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            0,
            4,
            0.0,
            0,
            4,
        );
        assert_eq!(result, Err(BackendError::NotInitialized));
    }

    // ── 3. Graceful init failure ──────────────────────────────────────────────

    #[test]
    fn wasm_backend_init_may_fail() {
        // Either Ok(()) or Err(_) is acceptable — must not panic.
        let mut b = WasmComputeBackend::new();
        let _result = b.init();
    }

    // ── 4. Post-init tests (skip if no WebGPU adapter is available) ──────────

    #[test]
    fn wasm_backend_is_initialized_after_init() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert!(b.is_initialized());
    }

    #[test]
    fn wasm_backend_alloc_small() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        let handle = match b.alloc(256) {
            Ok(h) => h,
            Err(_) => return,
        };
        assert!(handle > 0);
        b.free(handle).expect("free should succeed");
    }

    #[test]
    fn wasm_backend_alloc_zero() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(
            b.alloc(0),
            Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into()
            ))
        );
    }

    #[test]
    fn wasm_backend_alloc_large() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        let handle = match b.alloc(1 << 20) {
            Ok(h) => h,
            Err(_) => return,
        };
        assert!(handle > 0);
        b.free(handle).expect("free of large alloc should succeed");
    }

    #[test]
    fn wasm_backend_free_valid() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        let handle = match b.alloc(512) {
            Ok(h) => h,
            Err(_) => return,
        };
        assert!(b.free(handle).is_ok());
    }

    #[test]
    fn wasm_backend_free_invalid_handle() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        // A handle of 0xdeadbeef is almost certainly not a live allocation.
        let result = b.free(0xdead_beef);
        // May succeed (noop) or fail — must not panic.
        let _ = result;
    }

    #[test]
    fn wasm_backend_copy_htod_and_dtoh_roundtrip() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        let src: Vec<u8> = (0u8..64).collect();
        let handle = match b.alloc(src.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(handle, &src).expect("copy_htod");
        let mut dst = vec![0u8; src.len()];
        b.copy_dtoh(&mut dst, handle).expect("copy_dtoh");
        assert_eq!(src, dst);
        b.free(handle).expect("free");
    }

    #[test]
    fn wasm_backend_synchronize() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(b.synchronize(), Ok(()));
    }

    #[test]
    fn wasm_backend_gemm_unsupported() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        // Non-zero dims → should return Unsupported (not panic).
        let result = b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            0,
            4,
            0.0,
            0,
            4,
        );
        match result {
            Err(BackendError::Unsupported(_)) | Ok(()) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn wasm_backend_gemm_zero_dims_ok() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(
            b.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                0,
                0,
                1.0,
                0,
                1,
                0,
                1,
                0.0,
                0,
                1,
            ),
            Ok(())
        );
    }

    #[test]
    fn wasm_backend_conv2d_unsupported() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        let result = b.conv2d_forward(
            0,
            &[1, 3, 32, 32],
            0,
            &[16, 3, 3, 3],
            0,
            &[1, 16, 30, 30],
            &[1, 1],
            &[0, 0],
        );
        match result {
            Err(BackendError::Unsupported(_)) | Ok(()) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn wasm_backend_conv2d_bad_input_shape() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(
            b.conv2d_forward(
                0,
                &[1, 3, 32], // 3 elements — wrong
                0,
                &[16, 3, 3, 3],
                0,
                &[1, 16, 30, 30],
                &[1, 1],
                &[0, 0],
            ),
            Err(BackendError::InvalidArgument(
                "input_shape must have 4 elements (NCHW)".into()
            ))
        );
    }

    #[test]
    fn wasm_backend_conv2d_bad_filter_shape() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(
            b.conv2d_forward(
                0,
                &[1, 3, 32, 32],
                0,
                &[16, 3, 3], // 3 elements — wrong
                0,
                &[1, 16, 30, 30],
                &[1, 1],
                &[0, 0],
            ),
            Err(BackendError::InvalidArgument(
                "filter_shape must have 4 elements (KCFHFW)".into()
            ))
        );
    }

    #[test]
    fn wasm_backend_reduce_empty_shape_error() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(
            b.reduce(ReduceOp::Sum, 0, 0, &[], 0),
            Err(BackendError::InvalidArgument(
                "shape must not be empty".into()
            ))
        );
    }

    #[test]
    fn wasm_backend_reduce_axis_oob_error() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(
            b.reduce(ReduceOp::Sum, 0, 0, &[4, 4], 5),
            Err(BackendError::InvalidArgument(
                "axis 5 is out of bounds for shape of length 2".into()
            ))
        );
    }

    #[test]
    fn wasm_backend_unary_zero_n_ok() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(b.unary(UnaryOp::Relu, 0, 0, 0), Ok(()));
    }

    #[test]
    fn wasm_backend_binary_zero_n_ok() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(b.binary(BinaryOp::Add, 0, 0, 0, 0), Ok(()));
    }

    #[test]
    fn wasm_backend_attention_zero_seq_error() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 0, 8, 64, 0.125, false),
            Err(BackendError::InvalidArgument(
                "seq_q, seq_kv, and head_dim must all be > 0".into()
            ))
        );
    }

    #[test]
    fn wasm_backend_attention_bad_scale_error() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, 0.0, false),
            Err(BackendError::InvalidArgument(
                "scale must be a positive finite number, got 0".into()
            ))
        );
    }

    #[test]
    fn wasm_backend_init_idempotent() {
        let Some(mut b) = try_init_wasm() else {
            return;
        };
        assert_eq!(b.init(), Ok(()));
        assert!(b.is_initialized());
    }

    #[test]
    fn wasm_backend_copy_htod_empty_noop() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(b.copy_htod(0, &[]), Ok(()));
    }

    #[test]
    fn wasm_backend_copy_dtoh_empty_noop() {
        let Some(b) = try_init_wasm() else {
            return;
        };
        assert_eq!(b.copy_dtoh(&mut [], 0), Ok(()));
    }
}
