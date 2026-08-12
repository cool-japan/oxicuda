//! Backend registry, capability-based selection, and fallback chains.
//!
//! This is the host-side "control plane" of the abstraction layer. It does
//! **not** execute any GPU work; it decides *which* backend should run a
//! given task, given:
//!
//! * which backends a build registered ([`BackendRegistry::register`]),
//! * whether each is actually present at runtime (its `available` flag,
//!   set from a probe in the concrete backend crate), and
//! * what each one can do ([`Capabilities`]).
//!
//! The selection logic is fully deterministic and testable without any GPU:
//! it always prefers the highest-priority *available* backend that satisfies
//! a [`SelectionRequest`], and degrades down a [`fallback_chain`] that ends
//! at the CPU reference backend whenever one is registered.

use crate::backend_kind::BackendKind;
use crate::capabilities::Capabilities;
use crate::error::{BackendError, BackendResult};

/// A backend known to the registry, plus the runtime facts needed to pick it.
///
/// `available` is supplied by the caller (typically the result of a cheap
/// driver probe such as "did `cuInit` succeed?"). The registry never sets it
/// itself, so this crate stays free of any device dependency.
#[derive(Debug, Clone)]
pub struct BackendEntry {
    /// Which concrete backend this describes.
    pub kind: BackendKind,
    /// Whether the backend is usable on this machine right now.
    pub available: bool,
    /// Selection priority — higher wins. Defaults to
    /// [`BackendKind::default_priority`].
    pub priority: u32,
    /// What the backend can do (filled from a driver query, or a sensible
    /// default for the CPU/portable backends).
    pub capabilities: Capabilities,
}

impl BackendEntry {
    /// Create an entry using the kind's default priority and CPU-profile
    /// capabilities. Callers override `capabilities` after construction for
    /// real GPU backends.
    #[must_use]
    pub fn new(kind: BackendKind, available: bool) -> Self {
        Self {
            kind,
            available,
            priority: kind.default_priority(),
            capabilities: if kind == BackendKind::Cpu {
                Capabilities::cpu()
            } else {
                Capabilities::default()
            },
        }
    }

    /// Builder-style override of the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Builder-style override of the capability report.
    #[must_use]
    pub fn with_capabilities(mut self, capabilities: Capabilities) -> Self {
        self.capabilities = capabilities;
        self
    }
}

/// Required-feature predicate used to filter backends during selection.
///
/// A backend is eligible only if it is available **and** its capabilities
/// satisfy every requested flag. All fields default to "don't care".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SelectionRequest {
    /// Require a GPU backend (exclude the CPU fallback from the primary pick).
    pub require_gpu: bool,
    /// Require native FP16 compute.
    pub require_fp16: bool,
    /// Require native BF16 compute.
    pub require_bf16: bool,
    /// Require FP8 compute.
    pub require_fp8: bool,
    /// Require Tensor-Core / matrix units.
    pub require_tensor_cores: bool,
    /// Require unified / managed memory.
    pub require_unified_memory: bool,
    /// Require peer-to-peer device access.
    pub require_peer_access: bool,
    /// If `Some`, only this exact backend kind is acceptable.
    pub pin: Option<BackendKind>,
}

impl SelectionRequest {
    /// A request with no constraints (any available backend qualifies).
    #[must_use]
    pub const fn any() -> Self {
        Self {
            require_gpu: false,
            require_fp16: false,
            require_bf16: false,
            require_fp8: false,
            require_tensor_cores: false,
            require_unified_memory: false,
            require_peer_access: false,
            pin: None,
        }
    }

    /// A request that demands a GPU (used by callers that must not silently
    /// fall back to the CPU reference path).
    #[must_use]
    pub const fn require_gpu() -> Self {
        let mut r = Self::any();
        r.require_gpu = true;
        r
    }

    /// Pin selection to one specific backend kind.
    #[must_use]
    pub const fn pinned(kind: BackendKind) -> Self {
        let mut r = Self::any();
        r.pin = Some(kind);
        r
    }

    /// Narrow this request to the CPU reference backend when the workload is
    /// too small to pay for a GPU dispatch.
    ///
    /// A GPU dispatch costs a host→device copy, a command submission and a
    /// synchronisation; below some byte count that overhead dominates the
    /// kernel and the host path finishes first. This helper expresses that
    /// policy *at selection time* — i.e. before any memory is allocated — so
    /// that the whole workload (allocations included) lands on one backend.
    /// Per-operation routing is deliberately **not** offered: a device pointer
    /// belongs to the backend that allocated it, so a per-op switch would
    /// require duplicating every buffer.
    ///
    /// The request is returned unchanged when the caller has already expressed
    /// an explicit preference ([`require_gpu`](Self::require_gpu) or a
    /// [`pin`](Self::pin)), or when `workload_bytes >= gpu_threshold_bytes`.
    ///
    /// # Example
    ///
    /// ```
    /// use oxicuda_backend::{BackendKind, SelectionRequest};
    ///
    /// // 1 KiB with a 64 KiB threshold → pinned to the host backend.
    /// let small = SelectionRequest::any().for_workload(1024, 64 * 1024);
    /// assert_eq!(small.pin, Some(BackendKind::Cpu));
    ///
    /// // 1 MiB → unchanged, so the GPU can win the selection.
    /// let large = SelectionRequest::any().for_workload(1024 * 1024, 64 * 1024);
    /// assert_eq!(large, SelectionRequest::any());
    /// ```
    #[must_use]
    pub const fn for_workload(mut self, workload_bytes: usize, gpu_threshold_bytes: usize) -> Self {
        if !self.require_gpu && self.pin.is_none() && workload_bytes < gpu_threshold_bytes {
            self.pin = Some(BackendKind::Cpu);
        }
        self
    }

    /// Returns `true` if `entry` satisfies every constraint in this request.
    #[must_use]
    pub fn is_satisfied_by(&self, entry: &BackendEntry) -> bool {
        if !entry.available {
            return false;
        }
        if let Some(pin) = self.pin {
            if entry.kind != pin {
                return false;
            }
        }
        let caps = &entry.capabilities;
        if self.require_gpu && !entry.kind.is_gpu() {
            return false;
        }
        if self.require_fp16 && !caps.supports_fp16 {
            return false;
        }
        if self.require_bf16 && !caps.supports_bf16 {
            return false;
        }
        if self.require_fp8 && !caps.supports_fp8 {
            return false;
        }
        if self.require_tensor_cores && !caps.tensor_cores {
            return false;
        }
        if self.require_unified_memory && !caps.unified_memory {
            return false;
        }
        if self.require_peer_access && !caps.peer_access {
            return false;
        }
        true
    }
}

/// Routable operation classes used by [`BackendRegistry::route`].
///
/// This is a coarse routing key, not the full op set: it lets a consumer ask
/// "which backend should run my GEMMs?" so that, for example, dense linear
/// algebra can be pinned to a Tensor-Core backend while element-wise work
/// goes to whatever is cheapest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpClass {
    /// Dense matrix multiply (`gemm` / `batched_gemm`).
    MatMul,
    /// Convolution (`conv2d_forward`).
    Convolution,
    /// Scaled dot-product attention.
    Attention,
    /// Axis reductions.
    Reduction,
    /// Element-wise unary / binary / softmax.
    Elementwise,
    /// Host/device memory transfers.
    Memory,
}

impl OpClass {
    /// Every routable op class.
    pub const ALL: [OpClass; 6] = [
        OpClass::MatMul,
        OpClass::Convolution,
        OpClass::Attention,
        OpClass::Reduction,
        OpClass::Elementwise,
        OpClass::Memory,
    ];

    /// `true` if this op class benefits from Tensor-Core units, so the
    /// router prefers a matrix-capable backend when one is available.
    #[must_use]
    pub const fn prefers_tensor_cores(self) -> bool {
        matches!(self, Self::MatMul | Self::Convolution | Self::Attention)
    }
}

/// A registry of compute backends with capability-aware selection.
///
/// Construction registers nothing; call [`register`](Self::register) (or
/// [`with_defaults`](Self::with_defaults)) to populate it. Selection is pure
/// and side-effect-free, so it is trivially unit-testable.
#[derive(Debug, Clone, Default)]
pub struct BackendRegistry {
    entries: Vec<BackendEntry>,
}

impl BackendRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// A registry pre-populated with one entry per [`BackendKind`], all
    /// flagged **unavailable** except the CPU reference backend.
    ///
    /// Concrete crates then flip the `available` flag (and refine the
    /// capabilities) for backends they detect via
    /// [`set_available`](Self::set_available) /
    /// [`set_capabilities`](Self::set_capabilities). This guarantees there
    /// is always at least one usable backend (the CPU fallback).
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        for kind in BackendKind::ALL {
            reg.register(BackendEntry::new(kind, kind == BackendKind::Cpu));
        }
        reg
    }

    /// Register (or replace) an entry. If a backend of the same kind is
    /// already present, it is overwritten so a later, more-informed probe
    /// wins.
    pub fn register(&mut self, entry: BackendEntry) {
        if let Some(slot) = self.entries.iter_mut().find(|e| e.kind == entry.kind) {
            *slot = entry;
        } else {
            self.entries.push(entry);
        }
    }

    /// Number of registered backends.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if no backend is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Borrow the entry for `kind`, if registered.
    #[must_use]
    pub fn get(&self, kind: BackendKind) -> Option<&BackendEntry> {
        self.entries.iter().find(|e| e.kind == kind)
    }

    /// Mark `kind`'s availability, returning `true` if the kind was present.
    pub fn set_available(&mut self, kind: BackendKind, available: bool) -> bool {
        if let Some(e) = self.entries.iter_mut().find(|e| e.kind == kind) {
            e.available = available;
            true
        } else {
            false
        }
    }

    /// Update `kind`'s capability report, returning `true` if present.
    pub fn set_capabilities(&mut self, kind: BackendKind, caps: Capabilities) -> bool {
        if let Some(e) = self.entries.iter_mut().find(|e| e.kind == kind) {
            e.capabilities = caps;
            true
        } else {
            false
        }
    }

    /// All currently-available backend kinds.
    #[must_use]
    pub fn available_kinds(&self) -> Vec<BackendKind> {
        self.entries
            .iter()
            .filter(|e| e.available)
            .map(|e| e.kind)
            .collect()
    }

    /// Select the single best backend satisfying `req`, or
    /// [`BackendError::Unsupported`] if none qualifies.
    ///
    /// "Best" = highest `priority` among satisfying entries; ties break by
    /// [`BackendKind::default_priority`] and then declaration order, so the
    /// result is deterministic.
    pub fn select(&self, req: &SelectionRequest) -> BackendResult<BackendKind> {
        self.entries
            .iter()
            .filter(|e| req.is_satisfied_by(e))
            .max_by(|a, b| {
                a.priority
                    .cmp(&b.priority)
                    .then(a.kind.default_priority().cmp(&b.kind.default_priority()))
            })
            .map(|e| e.kind)
            .ok_or_else(|| {
                BackendError::Unsupported(format!(
                    "no registered backend satisfies the request {req:?}"
                ))
            })
    }

    /// Convenience: select the best available backend with no constraints,
    /// preferring a GPU but accepting the CPU fallback.
    pub fn select_best(&self) -> BackendResult<BackendKind> {
        self.select(&SelectionRequest::any())
    }

    /// Select the best backend for a workload of `workload_bytes`, sending
    /// workloads smaller than `gpu_threshold_bytes` to the CPU reference
    /// backend (see [`SelectionRequest::for_workload`] for why this is a
    /// selection-time and not a per-operation decision).
    ///
    /// If the narrowed request finds nothing — e.g. a small workload on a
    /// registry with no CPU entry — the unnarrowed `req` is retried, so this
    /// never fails where [`select`](Self::select) would have succeeded.
    pub fn select_for_workload(
        &self,
        req: &SelectionRequest,
        workload_bytes: usize,
        gpu_threshold_bytes: usize,
    ) -> BackendResult<BackendKind> {
        let narrowed = req.for_workload(workload_bytes, gpu_threshold_bytes);
        match self.select(&narrowed) {
            Ok(kind) => Ok(kind),
            Err(_) if narrowed != *req => self.select(req),
            Err(e) => Err(e),
        }
    }

    /// The ordered fallback chain for `req`: every satisfying backend, most-
    /// to least-preferred, with the CPU reference backend forced to the end
    /// if it is available (even when it would also satisfy `req` earlier).
    ///
    /// Consumers walk this chain, trying each backend until one initializes
    /// and runs the op, so a transient GPU failure degrades gracefully to
    /// the host.
    #[must_use]
    pub fn fallback_chain(&self, req: &SelectionRequest) -> Vec<BackendKind> {
        // Collect satisfying entries together with their configured priority,
        // then sort most-preferred first: by explicit priority desc, then by
        // the kind's default priority desc as a deterministic tie-break.
        let mut ranked: Vec<(u32, BackendKind)> = self
            .entries
            .iter()
            .filter(|e| req.is_satisfied_by(e))
            .map(|e| (e.priority, e.kind))
            .collect();
        ranked.sort_by(|(pa, ka), (pb, kb)| {
            pb.cmp(pa)
                .then(kb.default_priority().cmp(&ka.default_priority()))
        });
        let mut chain: Vec<BackendKind> = ranked.into_iter().map(|(_, k)| k).collect();
        // Force the CPU reference backend to the very end if it is present,
        // so the host path is always the last resort.
        if let Some(pos) = chain.iter().position(|&k| k == BackendKind::Cpu) {
            let cpu = chain.remove(pos);
            chain.push(cpu);
        }
        chain
    }

    /// Route an [`OpClass`] to the best backend.
    ///
    /// For Tensor-Core-friendly classes the router first tries to find an
    /// available matrix-capable backend; if none exists it falls back to the
    /// plain best-available selection (so routing never fails when *any*
    /// backend is available).
    pub fn route(&self, op: OpClass) -> BackendResult<BackendKind> {
        if op.prefers_tensor_cores() {
            let tc_req = SelectionRequest {
                require_tensor_cores: true,
                ..SelectionRequest::any()
            };
            if let Ok(kind) = self.select(&tc_req) {
                return Ok(kind);
            }
        }
        self.select_best()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a registry with CUDA (no tensor cores), ROCm (tensor cores),
    /// and CPU, all available.
    fn three_backend_registry() -> BackendRegistry {
        let mut reg = BackendRegistry::new();
        reg.register(BackendEntry::new(BackendKind::Cuda, true));
        let rocm_caps = Capabilities {
            tensor_cores: true,
            supports_fp16: true,
            ..Capabilities::default()
        };
        reg.register(BackendEntry::new(BackendKind::Rocm, true).with_capabilities(rocm_caps));
        reg.register(BackendEntry::new(BackendKind::Cpu, true));
        reg
    }

    #[test]
    fn defaults_have_cpu_available_and_gpus_absent() {
        let reg = BackendRegistry::with_defaults();
        assert_eq!(reg.len(), 7);
        assert!(reg.get(BackendKind::Cpu).unwrap().available);
        assert!(!reg.get(BackendKind::Cuda).unwrap().available);
        // With only the CPU available, best selection is CPU.
        assert_eq!(reg.select_best().unwrap(), BackendKind::Cpu);
    }

    #[test]
    fn select_picks_highest_priority_available() {
        let reg = three_backend_registry();
        // CUDA has the highest default priority and is available.
        assert_eq!(reg.select_best().unwrap(), BackendKind::Cuda);
    }

    #[test]
    fn select_skips_unavailable_highest_priority() {
        let mut reg = three_backend_registry();
        reg.set_available(BackendKind::Cuda, false);
        // Now ROCm (next highest) wins.
        assert_eq!(reg.select_best().unwrap(), BackendKind::Rocm);
        reg.set_available(BackendKind::Rocm, false);
        // Only CPU left.
        assert_eq!(reg.select_best().unwrap(), BackendKind::Cpu);
    }

    #[test]
    fn select_honors_capability_requirements() {
        let reg = three_backend_registry();
        // Require tensor cores → only ROCm qualifies (CUDA entry has none).
        let req = SelectionRequest {
            require_tensor_cores: true,
            ..SelectionRequest::any()
        };
        assert_eq!(reg.select(&req).unwrap(), BackendKind::Rocm);
    }

    #[test]
    fn select_require_gpu_excludes_cpu() {
        let mut reg = BackendRegistry::new();
        reg.register(BackendEntry::new(BackendKind::Cpu, true));
        // Only CPU available; require_gpu must fail rather than pick CPU.
        assert!(reg.select(&SelectionRequest::require_gpu()).is_err());
        // ... but a constraint-free request happily returns CPU.
        assert_eq!(reg.select_best().unwrap(), BackendKind::Cpu);
    }

    #[test]
    fn select_pinned_backend() {
        let reg = three_backend_registry();
        let req = SelectionRequest::pinned(BackendKind::Cpu);
        assert_eq!(reg.select(&req).unwrap(), BackendKind::Cpu);
        // Pinning an unavailable backend fails.
        let mut reg2 = reg.clone();
        reg2.set_available(BackendKind::Rocm, false);
        assert!(
            reg2.select(&SelectionRequest::pinned(BackendKind::Rocm))
                .is_err()
        );
    }

    #[test]
    fn higher_explicit_priority_overrides_default() {
        let mut reg = BackendRegistry::new();
        // Give CPU an absurd priority so it beats CUDA's default 100.
        reg.register(BackendEntry::new(BackendKind::Cuda, true));
        reg.register(BackendEntry::new(BackendKind::Cpu, true).with_priority(1000));
        assert_eq!(reg.select_best().unwrap(), BackendKind::Cpu);
    }

    #[test]
    fn fallback_chain_orders_by_priority_with_cpu_last() {
        let reg = three_backend_registry();
        let chain = reg.fallback_chain(&SelectionRequest::any());
        assert_eq!(
            chain,
            vec![BackendKind::Cuda, BackendKind::Rocm, BackendKind::Cpu]
        );
        // CPU is always the final element when present.
        assert_eq!(*chain.last().unwrap(), BackendKind::Cpu);
    }

    #[test]
    fn fallback_chain_respects_constraints() {
        let reg = three_backend_registry();
        let req = SelectionRequest {
            require_tensor_cores: true,
            ..SelectionRequest::any()
        };
        // Only ROCm has tensor cores; CPU/CUDA excluded.
        assert_eq!(reg.fallback_chain(&req), vec![BackendKind::Rocm]);
    }

    #[test]
    fn fallback_chain_empty_when_nothing_qualifies() {
        let mut reg = three_backend_registry();
        reg.set_available(BackendKind::Cuda, false);
        reg.set_available(BackendKind::Rocm, false);
        reg.set_available(BackendKind::Cpu, false);
        assert!(reg.fallback_chain(&SelectionRequest::any()).is_empty());
        assert!(reg.select_best().is_err());
    }

    #[test]
    fn route_matmul_prefers_tensor_core_backend() {
        let reg = three_backend_registry();
        // CUDA has higher priority but no tensor cores; ROCm has them.
        // MatMul prefers tensor cores → ROCm.
        assert_eq!(reg.route(OpClass::MatMul).unwrap(), BackendKind::Rocm);
        // Elementwise does not prefer tensor cores → highest priority (CUDA).
        assert_eq!(reg.route(OpClass::Elementwise).unwrap(), BackendKind::Cuda);
    }

    #[test]
    fn route_falls_back_when_no_tensor_cores_anywhere() {
        let mut reg = BackendRegistry::new();
        reg.register(BackendEntry::new(BackendKind::Cuda, true)); // no TC
        reg.register(BackendEntry::new(BackendKind::Cpu, true));
        // No tensor-core backend exists → MatMul routes to best available.
        assert_eq!(reg.route(OpClass::MatMul).unwrap(), BackendKind::Cuda);
    }

    #[test]
    fn register_replaces_same_kind() {
        let mut reg = BackendRegistry::new();
        reg.register(BackendEntry::new(BackendKind::Cuda, false));
        assert!(!reg.get(BackendKind::Cuda).unwrap().available);
        reg.register(BackendEntry::new(BackendKind::Cuda, true));
        assert_eq!(reg.len(), 1);
        assert!(reg.get(BackendKind::Cuda).unwrap().available);
    }

    #[test]
    fn available_kinds_lists_only_available() {
        let mut reg = three_backend_registry();
        reg.set_available(BackendKind::Rocm, false);
        let avail = reg.available_kinds();
        assert!(avail.contains(&BackendKind::Cuda));
        assert!(avail.contains(&BackendKind::Cpu));
        assert!(!avail.contains(&BackendKind::Rocm));
    }

    #[test]
    fn set_methods_report_presence() {
        let mut reg = BackendRegistry::new();
        assert!(!reg.set_available(BackendKind::Cuda, true));
        reg.register(BackendEntry::new(BackendKind::Cuda, false));
        assert!(reg.set_available(BackendKind::Cuda, true));
        assert!(reg.set_capabilities(BackendKind::Cuda, Capabilities::default()));
        assert!(!reg.set_capabilities(BackendKind::Metal, Capabilities::default()));
    }

    #[test]
    fn op_class_tensor_core_preference() {
        assert!(OpClass::MatMul.prefers_tensor_cores());
        assert!(OpClass::Attention.prefers_tensor_cores());
        assert!(!OpClass::Elementwise.prefers_tensor_cores());
        assert!(!OpClass::Memory.prefers_tensor_cores());
        assert_eq!(OpClass::ALL.len(), 6);
    }

    // ── Workload-size-aware selection ────────────────────────────────────────

    /// 64 KiB, mirroring `oxicuda::AUTO_SELECT_THRESHOLD_BYTES`.
    const THRESHOLD: usize = 64 * 1024;

    #[test]
    fn for_workload_pins_cpu_below_threshold_only() {
        let below = SelectionRequest::any().for_workload(THRESHOLD - 1, THRESHOLD);
        assert_eq!(below.pin, Some(BackendKind::Cpu));
        // Exactly at the threshold is *not* below it → GPU still eligible.
        let at = SelectionRequest::any().for_workload(THRESHOLD, THRESHOLD);
        assert_eq!(at, SelectionRequest::any());
        let above = SelectionRequest::any().for_workload(THRESHOLD + 1, THRESHOLD);
        assert_eq!(above, SelectionRequest::any());
    }

    #[test]
    fn for_workload_respects_explicit_preferences() {
        // An explicit require_gpu is never downgraded to the host backend.
        let gpu = SelectionRequest::require_gpu().for_workload(16, THRESHOLD);
        assert_eq!(gpu.pin, None);
        assert!(gpu.require_gpu);
        // An explicit pin is never overwritten.
        let pinned = SelectionRequest::pinned(BackendKind::Metal).for_workload(16, THRESHOLD);
        assert_eq!(pinned.pin, Some(BackendKind::Metal));
    }

    #[test]
    fn select_for_workload_routes_small_to_cpu_and_large_to_gpu() {
        let reg = three_backend_registry();
        let any = SelectionRequest::any();
        assert_eq!(
            reg.select_for_workload(&any, 1024, THRESHOLD)
                .expect("small workload must select a backend"),
            BackendKind::Cpu,
            "a 1 KiB workload must stay on the host"
        );
        assert_eq!(
            reg.select_for_workload(&any, 1024 * 1024, THRESHOLD)
                .expect("large workload must select a backend"),
            BackendKind::Cuda,
            "a 1 MiB workload must reach the highest-priority GPU"
        );
    }

    #[test]
    fn select_for_workload_falls_back_when_no_cpu_entry() {
        let mut reg = BackendRegistry::new();
        reg.register(BackendEntry::new(BackendKind::Cuda, true));
        // No CPU entry at all: the narrowed (CPU-pinned) request cannot be
        // satisfied, so the original request must still be honoured.
        assert_eq!(
            reg.select_for_workload(&SelectionRequest::any(), 16, THRESHOLD)
                .expect("must fall back to the unnarrowed request"),
            BackendKind::Cuda
        );
    }

    #[test]
    fn select_for_workload_still_fails_when_nothing_qualifies() {
        let mut reg = three_backend_registry();
        for kind in BackendKind::ALL {
            reg.set_available(kind, false);
        }
        assert!(
            reg.select_for_workload(&SelectionRequest::any(), 16, THRESHOLD)
                .is_err()
        );
    }

    #[test]
    fn selection_request_constructors() {
        assert!(SelectionRequest::require_gpu().require_gpu);
        assert_eq!(
            SelectionRequest::pinned(BackendKind::Metal).pin,
            Some(BackendKind::Metal)
        );
        assert_eq!(SelectionRequest::any(), SelectionRequest::default());
    }
}
