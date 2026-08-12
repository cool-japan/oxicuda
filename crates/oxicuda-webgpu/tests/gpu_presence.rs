//! Device-presence assertion, gated behind the `gpu-tests` Cargo feature.
//!
//! Mirrors `oxicuda-metal/tests/gpu_presence.rs`. Every GPU-path test
//! elsewhere in this crate degrades gracefully to a no-op skip when no wgpu
//! adapter is available, so the suite stays green on headless CI. That
//! portability has a real cost: if [`WebGpuDevice::new`] silently starts
//! failing (a regression in adapter selection, a driver change, a broken CI
//! image), every one of those tests keeps reporting PASS while genuinely
//! executing nothing on a GPU.
//!
//! This file closes that hole with the opposite policy: **when the
//! `gpu-tests` feature is enabled, the absence of a real wgpu adapter is a
//! hard test failure, not a skip.** Run it explicitly on a machine that is
//! known to have a GPU wgpu can drive (Metal on macOS, Vulkan/DX12 on
//! Linux/Windows):
//!
//! ```text
//! cargo nextest run -p oxicuda-webgpu --features gpu-tests
//! ```
//!
//! Unlike the `oxicuda-metal` counterpart, this file is **not** further
//! gated by `target_os`: `wgpu` targets Vulkan, Metal, and DX12 across every
//! desktop platform, so there is no platform on which `--features
//! gpu-tests` is expected to compile but never find an adapter — enabling
//! the feature is itself the opt-in to "this environment must have a GPU".
//!
//! This is a *compile-time feature* deliberately distinct from the
//! `OXICUDA_REQUIRE_GPU=1` *runtime* env-var switch used by the crate's
//! in-tree `#[cfg(test)]` suites (see `src/backend_tests.rs`). The two are
//! complementary: the env var flips the behaviour of tests that already
//! exist and run by default, while this feature adds a test that exists
//! *only* when explicitly opted into.

#![cfg(feature = "gpu-tests")]

use oxicuda_backend::ComputeBackend;
use oxicuda_webgpu::WebGpuBackend;
use oxicuda_webgpu::device::WebGpuDevice;

/// A real wgpu adapter must be acquirable in this environment. If this
/// fails, every self-skipping GPU test in the crate is silently vacuous.
#[test]
fn webgpu_device_must_be_available() {
    let device = WebGpuDevice::new().unwrap_or_else(|e| {
        panic!(
            "gpu-tests feature requires a real wgpu adapter, but \
             WebGpuDevice::new() failed: {e}. Either this environment \
             genuinely has no usable GPU (wrong feature to enable here), or \
             adapter acquisition has regressed."
        )
    });
    assert!(
        !device.adapter_name.is_empty(),
        "an acquired wgpu adapter must report a non-empty name"
    );
    assert!(
        !device.is_device_lost(),
        "a freshly acquired device must not already be lost"
    );
}

/// The same assertion through the public `WebGpuBackend` / `ComputeBackend`
/// entry point every real caller uses, so a regression that is specific to
/// the backend's `init()` wiring (rather than `WebGpuDevice::new()` itself)
/// is caught too.
#[test]
fn webgpu_backend_init_must_succeed() {
    let mut backend = WebGpuBackend::new();
    backend.init().unwrap_or_else(|e| {
        panic!(
            "gpu-tests feature requires WebGpuBackend::init() to succeed, \
             but it returned: {e}"
        )
    });
    assert!(backend.is_initialized());

    // A trivial real dispatch, not just adapter enumeration: alloc, upload,
    // run a unary op, download, free. If this silently no-oped, the earlier
    // assertions would not have caught it.
    let ptr = backend.alloc(4 * 4).expect("alloc 4 f32s");
    let data = [1.0f32, -2.0, 3.0, -4.0];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    backend.copy_htod(ptr, &bytes).expect("copy_htod");
    backend
        .unary(oxicuda_backend::UnaryOp::Relu, ptr, ptr, 4)
        .expect("unary relu dispatch");
    let mut out_bytes = vec![0u8; 16];
    backend.copy_dtoh(&mut out_bytes, ptr).expect("copy_dtoh");
    let out: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(out, vec![1.0, 0.0, 3.0, 0.0], "ReLU must actually execute");
    backend.free(ptr).expect("free");
}
