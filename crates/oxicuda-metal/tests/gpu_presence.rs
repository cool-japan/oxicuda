//! Device-presence assertion, gated behind the `gpu-tests` Cargo feature.
//!
//! Every GPU-path test elsewhere in this crate (and in `oxicuda-webgpu`)
//! degrades gracefully to a no-op skip when no device is available, so the
//! suite stays green on headless CI. That portability has a real cost: if
//! [`MetalDevice::new`] silently starts failing (a regression in device
//! selection, a driver change, a broken CI image), every one of those tests
//! keeps reporting PASS while genuinely executing nothing on a GPU.
//!
//! This file closes that hole with the opposite policy: **when the
//! `gpu-tests` feature is enabled, the absence of a real Metal device is a
//! hard test failure, not a skip.** Run it explicitly on a machine that is
//! known to have a Metal-capable GPU:
//!
//! ```text
//! cargo nextest run -p oxicuda-metal --features gpu-tests
//! ```
//!
//! The whole file is `#[cfg(target_os = "macos")]` in addition to the
//! feature gate: off macOS, `MetalDevice::new()` always returns
//! `MetalError::UnsupportedPlatform` by design (see `oxicuda_metal::device`),
//! so demanding a live device there would fail for a reason that has nothing
//! to do with GPU availability. A workspace-wide `--all-features` run on
//! Linux therefore stays green; the hard-failure guarantee applies on the
//! platform this crate actually targets.
//!
//! This is a *compile-time feature* deliberately distinct from the
//! `OXICUDA_REQUIRE_GPU=1` *runtime* env-var switch used by the crate's
//! in-tree `#[cfg(test)]` suites (see `src/backend/gpu_tests.rs`). The two
//! are complementary: the env var flips the behaviour of tests that already
//! exist and run by default, while this feature adds a test that exists
//! *only* when explicitly opted into — a CI job can enable `gpu-tests` on a
//! GPU runner without needing to also thread an environment variable through.

#![cfg(all(feature = "gpu-tests", target_os = "macos"))]

use oxicuda_backend::ComputeBackend;
use oxicuda_metal::MetalBackend;
use oxicuda_metal::device::MetalDevice;

/// A real Metal device must be acquirable on this platform. If this fails,
/// every self-skipping GPU test in the crate is silently vacuous.
#[test]
fn metal_device_must_be_available() {
    let device = MetalDevice::new().unwrap_or_else(|e| {
        panic!(
            "gpu-tests feature requires a real Metal device on macOS, but \
             MetalDevice::new() failed: {e}. Either this machine genuinely has \
             no Metal-capable GPU (wrong feature to enable here), or Metal \
             device acquisition has regressed."
        )
    });
    assert!(
        !device.name().is_empty(),
        "an acquired Metal device must report a non-empty name"
    );
    assert!(
        device.max_buffer_length() > 0,
        "an acquired Metal device must report a positive max buffer length"
    );
}

/// The same assertion through the public `MetalBackend` / `ComputeBackend`
/// entry point every real caller uses, so a regression that is specific to
/// the backend's `init()` wiring (rather than `MetalDevice::new()` itself)
/// is caught too.
#[test]
fn metal_backend_init_must_succeed() {
    let mut backend = MetalBackend::new();
    backend.init().unwrap_or_else(|e| {
        panic!(
            "gpu-tests feature requires MetalBackend::init() to succeed on \
             macOS, but it returned: {e}"
        )
    });
    assert!(backend.is_initialized());

    let devices = backend
        .available_devices()
        .expect("available_devices must succeed once initialised");
    assert!(
        !devices.is_empty(),
        "an initialised backend must report at least one device"
    );

    // A trivial real dispatch, not just device enumeration: alloc, upload,
    // run a unary op, download, free. If this silently no-ops the earlier
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
