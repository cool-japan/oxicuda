//! One-call compute-backend selection: "give me something that computes here".
//!
//! [`crate::backend`] exposes the [`ComputeBackend`](crate::backend::ComputeBackend) trait and every concrete
//! implementation, but picking one by hand means knowing which GPU stack this
//! machine has, whether its driver loads, and which feature flags the build
//! turned on. This module does that for you: it probes the backends compiled
//! into the build, ranks them with the [`BackendRegistry`](crate::backend::BackendRegistry) control plane, and
//! hands back the best one **already initialised**.
//!
//! ```no_run
//! use oxicuda::backend::ComputeBackend;
//!
//! # fn main() -> oxicuda::backend::BackendResult<()> {
//! let backend = oxicuda::compute::default_backend()?;
//! println!("computing on the {} backend", backend.name());
//! # Ok(())
//! # }
//! ```
//!
//! # What gets selected
//!
//! | Kind | Constructed when | Considered available when |
//! |------|------------------|---------------------------|
//! | [`BackendKind::Cuda`](crate::backend::BackendKind::Cuda) | always | `libcuda` loads **and** reports ≥ 1 device |
//! | [`BackendKind::Rocm`](crate::backend::BackendKind::Rocm) | feature `rocm` | `RocmDevice::new()` succeeds |
//! | [`BackendKind::Metal`](crate::backend::BackendKind::Metal) | feature `metal` | `MetalDevice` opens (macOS) |
//! | [`BackendKind::LevelZero`](crate::backend::BackendKind::LevelZero) | feature `level-zero` | Level Zero driver opens |
//! | [`BackendKind::Vulkan`](crate::backend::BackendKind::Vulkan) | feature `vulkan` | a Vulkan device opens |
//! | [`BackendKind::WebGpu`](crate::backend::BackendKind::WebGpu) | feature `webgpu` | `wgpu` finds an adapter |
//! | [`BackendKind::Cpu`](crate::backend::BackendKind::Cpu) | always | always — the guaranteed fallback |
//!
//! Ordering follows [`BackendKind::default_priority`](crate::backend::BackendKind::default_priority), so a native GPU stack
//! wins over a portable one and the host path is always last. The CUDA entry
//! is special-cased: [`CudaBackend::init`](crate::backend::CudaBackend) succeeds
//! even with no NVIDIA GPU present, so availability is decided by an explicit
//! driver probe instead of by `init`.
//!
//! # macOS
//!
//! With the `metal` feature this returns a `MetalBackend` running on the Apple
//! GPU; without it — or on a Mac where the Metal device cannot be opened — it
//! returns the [`CpuBackend`](crate::backend::CpuBackend), never an error. The CUDA driver path is not
//! available on macOS at all, so [`crate::init`] there returns
//! `Err(CudaError::NotInitialized)` while this module keeps working.
//!
//! ```toml
//! [dependencies]
//! oxicuda = { version = "0.5", features = ["metal"] }
//! ```
//!
//! Metal covers `gemm`/`batched_gemm`, the element-wise unary and binary ops
//! and the axis reductions; ops it does not implement return
//! [`BackendError::Unsupported`](crate::backend::BackendError::Unsupported) rather than falling back silently, so a
//! consumer that needs them should keep a [`CpuBackend`](crate::backend::CpuBackend) alongside.

use std::ops::{Deref, DerefMut};

use crate::backend::{
    BackendEntry, BackendError, BackendKind, BackendRegistry, BackendResult, ComputeBackend,
    CpuBackend, SelectionRequest,
};

// ─── Selected backend handle ────────────────────────────────

/// An initialised compute backend plus the [`BackendKind`] it was selected as.
///
/// Dereferences to `dyn ComputeBackend`, so every trait method can be called
/// directly on the handle:
///
/// ```no_run
/// use oxicuda::backend::ComputeBackend;
///
/// # fn main() -> oxicuda::backend::BackendResult<()> {
/// let backend = oxicuda::compute::default_backend()?;
/// let ptr = backend.alloc(1024)?;
/// backend.free(ptr)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SelectedBackend {
    kind: BackendKind,
    backend: Box<dyn ComputeBackend>,
}

impl SelectedBackend {
    /// Which concrete backend was selected.
    #[must_use]
    pub fn kind(&self) -> BackendKind {
        self.kind
    }

    /// Borrow the initialised backend for dynamic dispatch.
    #[must_use]
    pub fn backend(&self) -> &dyn ComputeBackend {
        self.backend.as_ref()
    }

    /// Mutably borrow the initialised backend.
    pub fn backend_mut(&mut self) -> &mut dyn ComputeBackend {
        self.backend.as_mut()
    }

    /// Take ownership of the boxed backend, e.g. to share it as an
    /// `Arc<dyn ComputeBackend>` via `Arc::from(sel.into_inner())`.
    #[must_use]
    pub fn into_inner(self) -> Box<dyn ComputeBackend> {
        self.backend
    }
}

impl Deref for SelectedBackend {
    type Target = dyn ComputeBackend;

    fn deref(&self) -> &Self::Target {
        self.backend.as_ref()
    }
}

impl DerefMut for SelectedBackend {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.backend.as_mut()
    }
}

// ─── Per-kind construction ──────────────────────────────────

/// `true` if the CUDA driver library loaded and reports at least one device.
///
/// [`CudaBackend`](crate::backend::CudaBackend) deliberately lets `init`
/// succeed without a GPU (so the type is always constructible), which makes
/// `init` useless as an availability probe — hence this explicit check.
fn cuda_driver_present() -> bool {
    crate::Device::count().is_ok_and(|count| count > 0)
}

fn new_cuda() -> BackendResult<Box<dyn ComputeBackend>> {
    if cuda_driver_present() {
        Ok(Box::new(crate::backend::CudaBackend::new()))
    } else {
        Err(BackendError::Unsupported(
            "no CUDA driver with a usable device on this machine".into(),
        ))
    }
}

#[cfg(feature = "metal")]
fn new_metal() -> BackendResult<Box<dyn ComputeBackend>> {
    Ok(Box::new(crate::backend::MetalBackend::new()))
}

#[cfg(not(feature = "metal"))]
fn new_metal() -> BackendResult<Box<dyn ComputeBackend>> {
    Err(BackendError::Unsupported(
        "not compiled in (enable feature \"metal\")".into(),
    ))
}

#[cfg(feature = "webgpu")]
fn new_webgpu() -> BackendResult<Box<dyn ComputeBackend>> {
    Ok(Box::new(crate::backend::WebGpuBackend::new()))
}

#[cfg(not(feature = "webgpu"))]
fn new_webgpu() -> BackendResult<Box<dyn ComputeBackend>> {
    Err(BackendError::Unsupported(
        "not compiled in (enable feature \"webgpu\")".into(),
    ))
}

#[cfg(feature = "vulkan")]
fn new_vulkan() -> BackendResult<Box<dyn ComputeBackend>> {
    Ok(Box::new(crate::backend::VulkanBackend::new()))
}

#[cfg(not(feature = "vulkan"))]
fn new_vulkan() -> BackendResult<Box<dyn ComputeBackend>> {
    Err(BackendError::Unsupported(
        "not compiled in (enable feature \"vulkan\")".into(),
    ))
}

#[cfg(feature = "rocm")]
fn new_rocm() -> BackendResult<Box<dyn ComputeBackend>> {
    Ok(Box::new(crate::backend::RocmBackend::new()))
}

#[cfg(not(feature = "rocm"))]
fn new_rocm() -> BackendResult<Box<dyn ComputeBackend>> {
    Err(BackendError::Unsupported(
        "not compiled in (enable feature \"rocm\")".into(),
    ))
}

#[cfg(feature = "level-zero")]
fn new_level_zero() -> BackendResult<Box<dyn ComputeBackend>> {
    Ok(Box::new(crate::backend::LevelZeroBackend::new()))
}

#[cfg(not(feature = "level-zero"))]
fn new_level_zero() -> BackendResult<Box<dyn ComputeBackend>> {
    Err(BackendError::Unsupported(
        "not compiled in (enable feature \"level-zero\")".into(),
    ))
}

/// Construct an **uninitialised** backend of `kind`, or explain why this build
/// or machine cannot provide one.
fn instantiate(kind: BackendKind) -> BackendResult<Box<dyn ComputeBackend>> {
    match kind {
        BackendKind::Cuda => new_cuda(),
        BackendKind::Rocm => new_rocm(),
        BackendKind::LevelZero => new_level_zero(),
        BackendKind::Vulkan => new_vulkan(),
        BackendKind::Metal => new_metal(),
        BackendKind::WebGpu => new_webgpu(),
        BackendKind::Cpu => Ok(Box::new(CpuBackend::new())),
    }
}

/// Registry entry describing a backend that has just initialised successfully,
/// with the capabilities it reports now that it owns a device.
fn live_entry(kind: BackendKind, backend: &dyn ComputeBackend) -> BackendEntry {
    BackendEntry::new(kind, true).with_capabilities(backend.capabilities())
}

// ─── Public API ─────────────────────────────────────────────

/// The backend kinds this build can construct at all, most-preferred first.
///
/// "Compiled in" is a build-time fact and says nothing about whether the
/// machine has the hardware — use [`default_registry`] for that.
#[must_use]
pub fn compiled_in_kinds() -> Vec<BackendKind> {
    let mut kinds: Vec<BackendKind> = BackendKind::ALL
        .into_iter()
        .filter(|kind| match kind {
            // Always constructible: the CUDA driver is loaded at runtime and
            // the host backend is pure Rust.
            BackendKind::Cuda | BackendKind::Cpu => true,
            BackendKind::Rocm => cfg!(feature = "rocm"),
            BackendKind::LevelZero => cfg!(feature = "level-zero"),
            BackendKind::Vulkan => cfg!(feature = "vulkan"),
            BackendKind::Metal => cfg!(feature = "metal"),
            BackendKind::WebGpu => cfg!(feature = "webgpu"),
        })
        .collect();
    kinds.sort_by_key(|kind| std::cmp::Reverse(kind.default_priority()));
    kinds
}

/// Probe this machine and describe every backend in a [`BackendRegistry`].
///
/// Each compiled-in backend is constructed and initialised once; the ones that
/// come up are registered as available with the capabilities they report, and
/// everything else is registered as unavailable so the registry still lists it.
/// The probe instances are dropped before returning, so this costs one device
/// open (and close) per compiled-in backend — call it for diagnostics, and use
/// [`select_backend`] when you actually want to compute, since that stops at
/// the first backend that works.
#[must_use]
pub fn default_registry() -> BackendRegistry {
    let mut registry = BackendRegistry::new();
    for kind in BackendKind::ALL {
        let entry = match instantiate(kind) {
            Ok(mut backend) => match backend.init() {
                Ok(()) => live_entry(kind, backend.as_ref()),
                Err(_) => BackendEntry::new(kind, false),
            },
            Err(_) => BackendEntry::new(kind, false),
        };
        registry.register(entry);
    }
    registry
}

/// Select, initialise and return the best backend satisfying `req`.
///
/// Candidates are walked in [`BackendRegistry::fallback_chain`] order and the
/// first one that initialises **and** satisfies `req` wins, so a GPU that is
/// compiled in but absent (or broken) degrades to the next choice and finally
/// to the [`CpuBackend`].
///
/// Only the *structural* constraints — [`SelectionRequest::pin`] and
/// [`SelectionRequest::require_gpu`], which are facts about the backend kind —
/// are applied while building that chain. The capability constraints
/// (`require_fp16`, `require_tensor_cores`, …) are checked afterwards against
/// the live [`ComputeBackend::capabilities`] of an initialised backend, because
/// an unopened device cannot report what it can do.
///
/// # Errors
///
/// [`BackendError::Unsupported`] listing every candidate and why it was
/// rejected, when nothing qualifies (only reachable for a constrained request:
/// the unconstrained one always has the host fallback).
pub fn select_backend(req: &SelectionRequest) -> BackendResult<SelectedBackend> {
    let structural = SelectionRequest {
        require_gpu: req.require_gpu,
        pin: req.pin,
        ..SelectionRequest::any()
    };
    let mut candidates = BackendRegistry::new();
    for kind in compiled_in_kinds() {
        candidates.register(BackendEntry::new(kind, true));
    }

    let mut rejected: Vec<String> = Vec::new();
    for kind in candidates.fallback_chain(&structural) {
        let mut backend = match instantiate(kind) {
            Ok(backend) => backend,
            Err(e) => {
                rejected.push(format!("{kind}: {e}"));
                continue;
            }
        };
        if let Err(e) = backend.init() {
            rejected.push(format!("{kind}: init failed ({e})"));
            continue;
        }
        let entry = live_entry(kind, backend.as_ref());
        if !req.is_satisfied_by(&entry) {
            rejected.push(format!("{kind}: [{}] does not satisfy", entry.capabilities));
            continue;
        }
        return Ok(SelectedBackend { kind, backend });
    }

    Err(BackendError::Unsupported(format!(
        "no compute backend satisfies {req:?} — rejected: {}",
        if rejected.is_empty() {
            "nothing was compiled in".to_string()
        } else {
            rejected.join("; ")
        }
    )))
}

/// The best backend this machine offers, already initialised.
///
/// Never fails on a supported platform: with no GPU (or no GPU feature
/// enabled) it returns the pure-Rust [`CpuBackend`].
///
/// # Errors
///
/// [`BackendError::Unsupported`] only if even the host backend refuses to
/// initialise.
pub fn default_backend() -> BackendResult<SelectedBackend> {
    select_backend(&SelectionRequest::any())
}

/// The best **GPU** backend, refusing to fall back to the host.
///
/// # Errors
///
/// [`BackendError::Unsupported`] when this machine has no usable GPU backend
/// compiled in; the message lists what was tried.
pub fn gpu_backend() -> BackendResult<SelectedBackend> {
    select_backend(&SelectionRequest::require_gpu())
}

/// Pick a backend for a workload of `workload_bytes`, keeping small workloads
/// on the host.
///
/// Below [`crate::AUTO_SELECT_THRESHOLD_BYTES`] the host↔device copies and the
/// dispatch round-trip cost more than the kernel saves, so the
/// [`CpuBackend`] is selected; at or above it, selection is the same as
/// [`default_backend`]. `workload_bytes` should be the total footprint of the
/// buffers the caller is about to allocate.
///
/// # Errors
///
/// As [`default_backend`].
pub fn backend_for_workload(workload_bytes: usize) -> BackendResult<SelectedBackend> {
    let req =
        SelectionRequest::any().for_workload(workload_bytes, crate::AUTO_SELECT_THRESHOLD_BYTES);
    match select_backend(&req) {
        Ok(selected) => Ok(selected),
        // The narrowed request pins the host backend; if that is somehow
        // unusable, honour the original unconstrained request instead.
        Err(e) if req != SelectionRequest::any() => {
            select_backend(&SelectionRequest::any()).map_err(|_| e)
        }
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::UnaryOp;

    /// Host bytes for `values`, in the device's native element order.
    fn to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    /// Decode `bytes` back into `f32` elements.
    fn from_bytes(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // ── Registry / selection order ───────────────────────────────────────────

    #[test]
    fn compiled_in_kinds_always_offers_the_host_fallback() {
        let kinds = compiled_in_kinds();
        assert!(
            kinds.contains(&BackendKind::Cpu),
            "the CPU reference backend must always be constructible"
        );
        assert_eq!(
            kinds.last(),
            Some(&BackendKind::Cpu),
            "the host backend must rank last: {kinds:?}"
        );
    }

    #[test]
    fn compiled_in_kinds_track_the_enabled_features() {
        let kinds = compiled_in_kinds();
        assert_eq!(kinds.contains(&BackendKind::Metal), cfg!(feature = "metal"));
        assert_eq!(
            kinds.contains(&BackendKind::WebGpu),
            cfg!(feature = "webgpu")
        );
    }

    #[test]
    fn compiled_in_kinds_are_ordered_by_priority() {
        let kinds = compiled_in_kinds();
        for pair in kinds.windows(2) {
            assert!(
                pair[0].default_priority() >= pair[1].default_priority(),
                "selection order must be non-increasing in priority: {kinds:?}"
            );
        }
    }

    #[test]
    fn default_registry_lists_every_kind_with_cpu_available() {
        let registry = default_registry();
        assert_eq!(registry.len(), BackendKind::ALL.len());
        let cpu = registry
            .get(BackendKind::Cpu)
            .expect("the CPU entry must be registered");
        assert!(cpu.available, "the CPU backend is always available");
        assert_eq!(
            registry.select_best().expect("selection must succeed"),
            default_backend()
                .expect("default_backend must succeed")
                .kind(),
            "the probing registry and the lazy selection must agree"
        );
    }

    #[test]
    fn unavailable_backends_are_registered_but_not_selected() {
        let registry = default_registry();
        for kind in BackendKind::ALL {
            let entry = registry.get(kind).expect("every kind must be registered");
            assert_eq!(entry.kind, kind);
        }
        // Nothing compiled out can ever be reported as available.
        if !cfg!(feature = "vulkan") {
            let vulkan = registry
                .get(BackendKind::Vulkan)
                .expect("Vulkan must still be listed");
            assert!(!vulkan.available);
        }
    }

    #[test]
    fn cuda_is_never_selected_without_a_driver() {
        if cuda_driver_present() {
            return; // A real NVIDIA box: CUDA legitimately wins.
        }
        let selected = default_backend().expect("a backend must always be available");
        assert_ne!(
            selected.kind(),
            BackendKind::Cuda,
            "CudaBackend::init succeeds without a GPU, so it must be filtered by the probe"
        );
    }

    // ── Platform expectations ────────────────────────────────────────────────

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn macos_with_metal_feature_selects_metal() {
        let metal_opens = {
            let mut probe = crate::backend::MetalBackend::new();
            probe.init().is_ok()
        };
        let selected = default_backend().expect("a backend must always be available");
        if metal_opens {
            assert_eq!(
                selected.kind(),
                BackendKind::Metal,
                "with a working Metal device the facade must select it"
            );
        } else {
            assert_eq!(
                selected.kind(),
                BackendKind::Cpu,
                "without a Metal device the facade must degrade to the host"
            );
        }
    }

    /// Run `relu` over `values` on `backend`, returning the result.
    fn relu_through(backend: &dyn ComputeBackend, values: &[f32]) -> Vec<f32> {
        let bytes = std::mem::size_of_val(values);
        let input = backend.alloc(bytes).expect("alloc must succeed");
        let output = backend.alloc(bytes).expect("alloc must succeed");
        backend
            .copy_htod(input, &to_bytes(values))
            .expect("host→device copy must succeed");
        backend
            .unary(UnaryOp::Relu, input, output, values.len())
            .expect("relu must succeed");
        backend.synchronize().expect("synchronize must succeed");
        let mut host = vec![0u8; bytes];
        backend
            .copy_dtoh(&mut host, output)
            .expect("device→host copy must succeed");
        backend.free(input).expect("free must succeed");
        backend.free(output).expect("free must succeed");
        from_bytes(&host)
    }

    /// The selected GPU path must agree with the CPU reference — on a real
    /// Apple GPU, not just in the selection logic.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn metal_agrees_with_the_cpu_reference_on_negative_inputs() {
        let mut metal = crate::backend::MetalBackend::new();
        if metal.init().is_err() {
            return; // No Metal device here (headless VM): nothing to compare.
        }
        assert_eq!(metal.name(), "metal");

        let mut cpu = CpuBackend::new();
        cpu.init().expect("the host backend must initialise");

        let values = [-4.0f32, -0.25, 0.0, 0.5, 9.0, -1e-3, 1e3, -7.5];
        let on_metal = relu_through(&metal, &values);
        let on_cpu = relu_through(&cpu, &values);

        assert_eq!(on_metal.len(), values.len());
        for (i, (m, c)) in on_metal.iter().zip(on_cpu.iter()).enumerate() {
            assert!(
                (m - c).abs() < 1e-6,
                "relu[{i}]: Metal produced {m}, the CPU reference {c}"
            );
        }
        // Guards against a silently-dispatched identity kernel: the negative
        // inputs must have been clamped, so the results really differ.
        assert_ne!(
            on_metal.as_slice(),
            values.as_slice(),
            "Metal returned its input unchanged — relu did not run"
        );
    }

    #[cfg(all(target_os = "macos", not(feature = "metal")))]
    #[test]
    fn macos_without_metal_feature_selects_cpu() {
        let selected = default_backend().expect("a backend must always be available");
        assert_eq!(
            selected.kind(),
            BackendKind::Cpu,
            "macOS has no CUDA driver, so the host backend must win"
        );
    }

    #[test]
    fn gpu_backend_returns_a_gpu_or_an_explanatory_error() {
        match gpu_backend() {
            Ok(selected) => assert!(
                selected.kind().is_gpu(),
                "require_gpu must never yield the host backend"
            ),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("rejected") || msg.contains("no compute backend"),
                    "the error must explain what was tried, got: {msg}"
                );
            }
        }
    }

    // ── Workload-size routing ────────────────────────────────────────────────

    #[test]
    fn small_workloads_stay_on_the_host() {
        let selected = backend_for_workload(1024).expect("selection must succeed");
        assert_eq!(
            selected.kind(),
            BackendKind::Cpu,
            "1 KiB is below AUTO_SELECT_THRESHOLD_BYTES"
        );
        assert!(selected.is_initialized());
    }

    #[test]
    fn large_workloads_use_the_default_backend() {
        let large = backend_for_workload(crate::AUTO_SELECT_THRESHOLD_BYTES * 16)
            .expect("selection must succeed");
        let default = default_backend().expect("selection must succeed");
        assert_eq!(large.kind(), default.kind());
    }

    #[test]
    fn threshold_boundary_is_inclusive_for_the_gpu_side() {
        let at = backend_for_workload(crate::AUTO_SELECT_THRESHOLD_BYTES)
            .expect("selection must succeed");
        let default = default_backend().expect("selection must succeed");
        assert_eq!(
            at.kind(),
            default.kind(),
            "exactly at the threshold is not below it"
        );
        let below = backend_for_workload(crate::AUTO_SELECT_THRESHOLD_BYTES - 1)
            .expect("selection must succeed");
        assert_eq!(below.kind(), BackendKind::Cpu);
    }

    // ── End-to-end compute ───────────────────────────────────────────────────

    #[test]
    fn default_backend_runs_a_real_compute_round_trip() {
        let backend = default_backend().expect("a backend must always be available");
        assert!(
            backend.is_initialized(),
            "the selected backend must come back initialised"
        );

        // Negative inputs matter: a backend that silently dispatched an
        // identity kernel would still pass an all-positive ReLU check.
        let input_values = [-2.5f32, -0.5, 0.0, 1.5, 3.25, -7.0, 0.125, 42.0];
        let expected = [0.0f32, 0.0, 0.0, 1.5, 3.25, 0.0, 0.125, 42.0];

        let got = relu_through(backend.backend(), &input_values);
        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "relu[{i}] on the {} backend: got {g}, expected {e}",
                backend.name()
            );
        }
    }

    #[test]
    fn selected_backend_can_be_shared_as_an_arc() {
        let selected = default_backend().expect("a backend must always be available");
        let name = selected.name().to_string();
        let shared: std::sync::Arc<dyn ComputeBackend> =
            std::sync::Arc::from(selected.into_inner());
        assert_eq!(shared.name(), name, "sharing must not change the backend");
        assert!(
            shared.is_initialized(),
            "the shared backend stays initialised"
        );
    }
}
