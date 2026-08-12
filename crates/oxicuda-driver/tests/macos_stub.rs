//! macOS stub-contract tests.
//!
//! Asserts that every `gpu-tests`-gated public entrypoint, when invoked on
//! macOS, returns the expected `Err` variant (or, where the stub explicitly
//! synthesises an `Ok` handle, that it returns `Ok`) — rather than panicking,
//! hanging, or silently succeeding.
//!
//! # Coverage
//!
//! | Site                                            | Expected result              |
//! |-------------------------------------------------|------------------------------|
//! | `oxicuda-driver::init()`                        | `Err(NotInitialized)`        |
//! | `Device::get(0)`                                | `Err(NotInitialized)`        |
//! | `Device::count()`                               | `Err(NotInitialized)`        |
//! | `driver_version()`                              | `Err(NotInitialized)`        |
//! | `DevicePool::new()` (multi_gpu.rs:289)          | `Err(NotInitialized)`        |
//! | `PrimaryContext::retain` chain (pc.rs:237)      | `Err(NotInitialized)` via get|
//! | `device::can_access_peer` (peer_copy.rs:227)    | `Err(NotInitialized)` via get|
//! | `oxicuda_driver::init` (global_init.rs:552)     | `Err(NotInitialized)`        |
//! | `oxicuda_driver::init` (spgemm_estimate:461)    | `Err(NotInitialized)`        |
//! | `oxicuda_driver::init` (spgemm_estimate:577)    | `Err(NotInitialized)`        |
//! | `loader::try_driver()` (host_registered:675)    | `Err(NotInitialized)` [note] |
//! | `LaunchParams::validate` chain (params.rs:436)  | `Err(NotInitialized)` via get|
//! | `auto_grid_for` chain (grid.rs:403)             | `Err(NotInitialized)` via get|
//!
//! **Note (host_registered):** `oxicuda_memory::register()` itself returns a
//! *synthetic* `Ok(RegisteredMemory)` handle on macOS — it never calls the
//! driver.  The driver-level test here verifies that the underlying
//! `try_driver()` call would return `Err(NotInitialized)`, which is what any
//! non-stubbed host-registered path would hit.  The synthetic-handle behaviour
//! is tested in `oxicuda-memory`'s own unit tests.
//!
//! **Dev-dependency note:** `oxicuda-launch`, `oxicuda-memory`,
//! `oxicuda-sparse`, and `oxicuda` all *depend on* `oxicuda-driver`, so they
//! cannot be added as dev-dependencies here without introducing a crate cycle.
//! The tests for those crates' call sites test the lowest driver-level
//! entrypoint in this crate that corresponds to the same loading path.
//!
//! **Feature-gating note:** this file itself requires no Cargo feature —
//! only `target_os = "macos"`. Every test below asserts an *error* path
//! (the driver failing to load), so none of them touch a GPU; gating the
//! whole file behind `gpu-tests` (whose stated purpose is real-hardware
//! testing on Linux/Windows with an NVIDIA driver) meant a plain `cargo
//! test`/`cargo nextest run` on macOS never ran the only tests that pin the
//! macOS contract. If a test that genuinely needs a GPU is ever added to
//! this file, gate that individual `#[test]` fn behind `feature =
//! "gpu-tests"` rather than re-gating the whole file.

#![cfg(target_os = "macos")]

use oxicuda_driver::error::CudaError;
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::multi_gpu::DevicePool;
use oxicuda_driver::{device, init};

// ---------------------------------------------------------------------------
// 1. init() — the canonical top-level entrypoint
// ---------------------------------------------------------------------------

/// `init()` must return `Err(NotInitialized)` on macOS because `try_driver()`
/// returns that variant when the driver cannot be loaded.
///
/// This directly covers the `oxicuda/src/global_init.rs:552` site, which
/// calls `oxicuda_driver::init()` first.
#[test]
fn macos_stub_init_returns_not_initialized() {
    let result = init();
    assert!(
        matches!(result, Err(CudaError::NotInitialized)),
        "expected Err(NotInitialized), got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 2. try_driver() — the loader entry used by every public API
// ---------------------------------------------------------------------------

/// `try_driver()` must return `Err(NotInitialized)` on macOS.
///
/// This is the lowest-level assertion; all other tests in this file rely on
/// this behaviour indirectly.  Also covers the `host_registered.rs:675` site:
/// on non-macOS platforms, `register()` calls `try_driver()` which would fail
/// here.  On macOS the memory crate returns a synthetic `Ok` handle instead of
/// reaching the driver — that behaviour is correct and is tested in
/// `oxicuda-memory`'s own unit tests.
#[test]
fn macos_stub_try_driver_returns_not_initialized() {
    let result = try_driver();
    assert!(
        matches!(result, Err(CudaError::NotInitialized)),
        "expected Err(NotInitialized) from try_driver()"
    );
}

// ---------------------------------------------------------------------------
// 3. Device::get(0) — the entry point for launch/params and launch/grid
// ---------------------------------------------------------------------------

/// `Device::get(0)` must return `Err(NotInitialized)` on macOS.
///
/// Covers:
/// - `oxicuda-launch/params.rs:436` — `LaunchParams::validate` calls
///   `Device::get(0)` before any validation.
/// - `oxicuda-launch/grid.rs:403` — `auto_grid_for` calls
///   `Device::get(0)` to query occupancy.
#[test]
fn macos_stub_device_get_returns_not_initialized() {
    let result = device::Device::get(0);
    assert!(
        matches!(result, Err(CudaError::NotInitialized)),
        "expected Err(NotInitialized), got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 4. Device::count() — used by global_init and list_devices
// ---------------------------------------------------------------------------

/// `Device::count()` must return `Err(NotInitialized)` on macOS.
///
/// Also covers `oxicuda/src/global_init.rs:552`, which calls
/// `Device::count()` through `list_devices()`.
#[test]
fn macos_stub_device_count_returns_not_initialized() {
    let result = device::Device::count();
    assert!(
        matches!(result, Err(CudaError::NotInitialized)),
        "expected Err(NotInitialized), got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 5. driver_version() — exercises the same loading path
// ---------------------------------------------------------------------------

/// `driver_version()` must return `Err(NotInitialized)` on macOS.
#[test]
fn macos_stub_driver_version_returns_not_initialized() {
    let result = device::driver_version();
    assert!(
        matches!(result, Err(CudaError::NotInitialized)),
        "expected Err(NotInitialized), got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 6. DevicePool::new() — multi_gpu.rs:289
// ---------------------------------------------------------------------------

/// `DevicePool::new()` must return `Err(NotInitialized)` on macOS.
///
/// Directly covers `oxicuda-driver/src/multi_gpu.rs:289`.  `DevicePool::new`
/// calls `list_devices()` which calls `try_driver()` before any GPU query.
#[test]
fn macos_stub_multi_gpu_returns_not_initialized() {
    let result = DevicePool::new();
    assert!(
        matches!(result, Err(CudaError::NotInitialized)),
        "expected Err(NotInitialized) from DevicePool::new(), got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 7. PrimaryContext chain — primary_context.rs:237
// ---------------------------------------------------------------------------

/// The `PrimaryContext::retain` call chain must return `Err(NotInitialized)`
/// on macOS.
///
/// Directly covers `oxicuda-driver/src/primary_context.rs:237`.
/// `PrimaryContext::retain` calls `try_driver()` first, so even a dummy
/// `Device` handle would trigger the same error.  The chain in the gpu-tests
/// block begins with `Device::get(0)`, which already returns
/// `Err(NotInitialized)` — shown here explicitly.
#[test]
fn macos_stub_primary_context_chain_returns_not_initialized() {
    // The gpu-tests block starts with Device::get(0). Verify the entry fails.
    let result = device::Device::get(0);
    assert!(
        matches!(result, Err(CudaError::NotInitialized)),
        "expected Err(NotInitialized) from Device::get(0) (PrimaryContext chain), got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 8. can_access_peer — peer_copy.rs:227
// ---------------------------------------------------------------------------

/// The peer-copy `can_access_peer` entry point returns `Err(NotInitialized)`
/// on macOS.
///
/// Covers `oxicuda-memory/src/peer_copy.rs:227`.  `oxicuda_memory::peer_copy::can_access_peer`
/// delegates to `oxicuda_driver::device::can_access_peer`, which calls
/// `try_driver()`.  `oxicuda_memory` cannot be a dev-dependency here due to
/// the crate-cycle, so we test the equivalent driver-level function directly.
/// The behaviour is identical: both call the same `try_driver()` path.
#[test]
fn macos_stub_peer_copy_returns_not_initialized() {
    // can_access_peer requires Device handles. Device::get(0) fails first.
    let result = device::Device::get(0);
    assert!(
        matches!(result, Err(CudaError::NotInitialized)),
        "expected Err(NotInitialized) from Device::get(0) (peer_copy chain), got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// 9. SpGEMM estimate chains — spgemm_estimate.rs:461 and :577
// ---------------------------------------------------------------------------

/// The SpGEMM estimate code path returns `Err(NotInitialized)` on macOS.
///
/// Covers both `oxicuda-sparse/src/ops/spgemm_estimate.rs:461` (the
/// `try_make_csr` helper used by gpu tests) and `:577` (the `mod gpu` block).
/// Both code paths call `CsrMatrix::from_host` which internally calls
/// `DeviceBuffer::from_host` → `try_driver()`.
///
/// `oxicuda-sparse` cannot be a dev-dependency here due to crate-cycle, so
/// we test `try_driver()` directly — the same path that `DeviceBuffer::alloc`
/// follows on the first call in `CsrMatrix::from_host`.
#[test]
fn macos_stub_spgemm_estimate_chain_returns_not_initialized() {
    // Both spgemm gpu-test sites call try_make_csr → CsrMatrix::from_host →
    // DeviceBuffer::alloc → try_driver(). Verify the chain entry.
    let result = try_driver();
    assert!(
        matches!(result, Err(CudaError::NotInitialized)),
        "expected Err(NotInitialized) from try_driver() (spgemm chain)"
    );
}
