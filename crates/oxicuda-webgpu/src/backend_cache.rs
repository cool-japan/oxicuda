//! Dispatch-side caching for [`WebGpuBackend`]: compiled pipelines (bundled
//! with their group-0 bind-group layout), and — for calls that recur with
//! the same operand handles, the common case in a training/inference loop
//! that dispatches the same op on the same tensors every iteration — a
//! reusable bind group backed by a dedicated, `write_buffer`-refreshed
//! uniform buffer.
//!
//! Split out of `backend.rs` (as `backend::cache`, via `#[path = ...] mod
//! cache;`, mirroring how `gpu_ops` is split out) purely to keep that file
//! under the workspace's 2 000-line refactoring policy — everything here is
//! still logically part of `WebGpuBackend`'s internals, and uses `backend`'s
//! private fields/helpers via ordinary Rust module-privacy rules (a private
//! item is visible to its defining module *and all descendants*, and `cache`
//! is a child of `backend`).
//!
//! # Why this is safe to cache
//!
//! * **Pipelines** never change once compiled for a given key, so caching
//!   them (already done before this file existed) is unconditionally safe.
//!   Caching their bind-group layout alongside them is exactly as safe: the
//!   layout is a pure function of the pipeline's shader module, and
//!   `pipeline.get_bind_group_layout(0)` would return an equivalent layout
//!   every time it was called — this just avoids calling it more than once.
//! * **Bind groups + their dedicated uniform buffer** are more subtle: a
//!   cache hit `queue.write_buffer`s fresh parameter bytes into the *same*
//!   uniform buffer a previous call used, then reuses the *same*
//!   `wgpu::BindGroup` (which still points at that buffer, and at the same
//!   operand buffers). This relies on `wgpu::Queue` executing `write_buffer`
//!   and `submit` calls in the exact order they were issued from the host
//!   (the WebGPU/wgpu "queue timeline" guarantee) — verified against real
//!   Metal hardware by
//!   `backend_tests_pipeline::chained_gemm_reusing_same_handles_observes_latest_uniform_each_call`.
//! * **Handle reuse safety**: `WebGpuMemoryManager`'s handles are a
//!   monotonically increasing `AtomicU64` counter that is never reused, so a
//!   given handle value can only ever refer to one logical allocation for
//!   the lifetime of the process — a stale cache entry can never be
//!   silently reinterpreted as pointing at a *different* buffer than the one
//!   it was built for.
//! * **Freed-buffer memory**: `wgpu::BindGroup` internally retains a strong
//!   reference to every buffer it binds (the same guarantee this crate
//!   already relies on for an in-flight dispatch surviving a concurrent
//!   `free()` on an unrelated handle). Left unmanaged, a cached bind group
//!   would therefore keep a "freed" buffer's GPU memory alive indefinitely.
//!   [`BindGroupCache::evict_handle`] closes this: [`WebGpuBackend::free`]
//!   calls it before releasing the handle, dropping every cache entry that
//!   mentions it (and, with it, the cache's own reference to the
//!   now-orphaned `wgpu::Buffer`/`wgpu::BindGroup`).

use std::collections::{HashMap, VecDeque};

use oxicuda_backend::{BackendError, BackendResult};

use super::WebGpuBackend;
use crate::device::WebGpuDevice;
use crate::memory::WebGpuMemoryManager;

// ─── Pipeline + bind-group-layout cache ──────────────────────────────────────

/// A compiled compute pipeline plus its (single, group 0) bind-group layout.
///
/// Every dispatch call site in this crate uses exactly one bind group at
/// `@group(0)`, so caching the layout alongside the pipeline that produced it
/// means `pipeline.get_bind_group_layout(0)` — a wgpu-core call, not a free
/// field access — is made at most once per distinct pipeline key instead of
/// once per dispatch.
#[derive(Debug, Clone)]
pub(super) struct CachedPipeline {
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
}

// ─── Bind-group + dedicated-uniform-buffer cache ─────────────────────────────

/// Identifies one bind-group cache entry: the pipeline it was built against,
/// plus every operand-buffer handle it binds, in binding-index order.
///
/// A named struct (rather than a raw tuple) documents the fields and avoids
/// a `clippy::type_complexity` trigger on a multi-`u64`-wide tuple key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BindGroupKey {
    pipeline_key: String,
    handles: Vec<u64>,
}

/// One cached bind group: the `wgpu::BindGroup` itself, plus the dedicated
/// uniform buffer it was built with (`None` for a binding layout with no
/// uniform at all, e.g. `unary` / `binary`, whose shaders derive `n` from
/// `arrayLength` instead of a parameter buffer).
#[derive(Debug)]
struct CachedBindGroup {
    bind_group: wgpu::BindGroup,
    uniform_buffer: Option<wgpu::Buffer>,
    /// Byte length `uniform_buffer` was created at (`0` when there is none).
    /// A cache hit whose caller now wants a *different* length cannot reuse
    /// the entry — never happens today (one pipeline key always pairs with
    /// one fixed uniform-struct size) but checked defensively rather than
    /// silently truncating/overrunning the `write_buffer` call.
    uniform_size: u64,
}

/// Small bounded cache of `(pipeline, operand handles) -> bind group`.
///
/// Capped at [`BindGroupCache::MAX_ENTRIES`] with FIFO eviction (oldest
/// *inserted* key first, not oldest-*used*) once full: this is a perf cache
/// serving a working set that is, in the intended training/inference-loop
/// use case, far smaller than the cap, so exact LRU recency tracking would
/// add a second data structure for no measurable benefit at this size.
#[derive(Debug)]
pub(super) struct BindGroupCache {
    entries: HashMap<BindGroupKey, CachedBindGroup>,
    /// Insertion order, oldest first, for FIFO eviction. May contain keys
    /// already removed by [`Self::evict_handle`]; pruned lazily wherever it
    /// is walked so no removal path needs to scan it eagerly.
    order: VecDeque<BindGroupKey>,
}

impl BindGroupCache {
    /// Comfortably covers a loop's working set of distinct (op,
    /// operand-handle) combinations while keeping the `retain` scan in
    /// [`Self::evict_handle`] (run on every [`WebGpuBackend::free`]) cheap.
    const MAX_ENTRIES: usize = 64;

    pub(super) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Evict every entry whose key mentions `handle` — see the module doc's
    /// "Freed-buffer memory" note for why this must run before a handle is
    /// actually freed.
    ///
    /// Also prunes `order` of the same now-dead keys. Without this, a
    /// workload that allocates a fresh handle, dispatches once, and frees it
    /// every iteration — exactly the alloc/dispatch/free loop this cache
    /// exists to speed up on a *repeated*-handle workload, and a perfectly
    /// normal one on a fresh-handle workload — would leave `entries` empty
    /// but grow `order` (each element owning a `String` + `Vec<u64>`)
    /// without bound: `insert`'s own pruning only runs once `order.len() >=
    /// MAX_ENTRIES`, and only ever looks at the front, so a cold key that is
    /// used once and freed left a permanent dead entry behind every time.
    pub(super) fn evict_handle(&mut self, handle: u64) {
        self.entries.retain(|k, _| !k.handles.contains(&handle));
        self.order.retain(|k| self.entries.contains_key(k));
    }

    /// Number of live cache entries. `pub(super)` (visible to `backend` and
    /// its descendants, including the test modules) purely for testability —
    /// production code never needs to inspect cache occupancy directly.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Number of tracked insertion-order keys, *including* any not yet
    /// pruned from a prior `evict_handle` call that predates this accessor.
    /// `pub(super)`/test-only, same rationale as [`Self::len`] — used to
    /// prove `evict_handle` keeps `order` from growing without bound.
    #[cfg(test)]
    pub(super) fn order_len(&self) -> usize {
        self.order.len()
    }
}

impl WebGpuBackend {
    /// Return a compiled compute pipeline (and its group-0 bind-group
    /// layout) for `key`, building both from the WGSL produced by `build` on
    /// the first request and caching them together for reuse.
    ///
    /// `key` must uniquely identify the shader source (e.g. `"gemm:16"`,
    /// `"unary:relu"`); `label` is the wgpu debug label.
    pub(super) fn cached_pipeline(
        &self,
        key: &str,
        label: &str,
        build: impl FnOnce() -> String,
    ) -> BackendResult<CachedPipeline> {
        let mut cache = self
            .pipeline_cache
            .lock()
            .map_err(|_| BackendError::DeviceError("pipeline cache mutex poisoned".into()))?;

        if let Some(cached) = cache.get(key) {
            return Ok(cached.clone());
        }

        let dev = self.device()?;
        let wgsl = build();
        let shader_mod = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let cached = CachedPipeline {
            pipeline,
            bind_group_layout,
        };
        cache.insert(key.to_string(), cached.clone());
        Ok(cached)
    }

    /// Resolve — building once, then reusing on every subsequent call with
    /// the same `pipeline_key` and `handles` — the bind group for a compute
    /// dispatch whose `@group(0)` binding layout is `[operand buffers, in
    /// `handles` order, at bindings 0..handles.len()]` optionally followed by
    /// `[one uniform buffer at binding handles.len()]` when `uniform_bytes`
    /// is non-empty. This covers the hot dispatch call sites in
    /// `backend.rs`: `gemm` / `batched_gemm` / `gemm_f16` / `reduce_nd`
    /// (operands + a trailing uniform) and `unary` / `binary` (operands
    /// only).
    ///
    /// On a cache **hit**: `queue.write_buffer`s `uniform_bytes` into the
    /// entry's existing dedicated uniform buffer (if any) and clones the
    /// existing `wgpu::BindGroup` — no buffer lookup, no
    /// `device.create_bind_group` call.
    ///
    /// On a **miss**: resolves every handle in `handles` under
    /// `mem.lock_buffers()`, validating each is at least the corresponding
    /// entry of `min_sizes` bytes (pass `&[]` to skip size validation
    /// entirely — used where the caller already validated by other means);
    /// builds a fresh dedicated uniform buffer (if `uniform_bytes` is
    /// non-empty) and bind group; and inserts the entry for next time.
    ///
    /// Never holds the bind-group-cache lock and the buffers lock at the
    /// same time (the miss path drops the former before taking the latter,
    /// and re-acquires the former only after releasing the latter) —
    /// [`WebGpuBackend::free`] locks them in the opposite order
    /// (cache-then-`memory.free`), so acquiring both at once here would be a
    /// lock-order inversion. The cost is a benign, rare race: two threads
    /// missing on the same cold key may each build and insert independently,
    /// with the second insert's value winning and the first's freshly built
    /// (but now-orphaned) `wgpu::Buffer`/`wgpu::BindGroup` simply dropped.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cached_bind_group(
        &self,
        dev: &WebGpuDevice,
        mem: &WebGpuMemoryManager,
        bind_group_layout: &wgpu::BindGroupLayout,
        pipeline_key: &str,
        handles: &[u64],
        min_sizes: &[u64],
        uniform_bytes: &[u8],
        label: &'static str,
    ) -> BackendResult<wgpu::BindGroup> {
        let key = BindGroupKey {
            pipeline_key: pipeline_key.to_string(),
            handles: handles.to_vec(),
        };

        // Fast path: cache hit only refreshes the uniform (if any) and
        // clones the existing bind-group handle.
        {
            let cache = self
                .bind_group_cache
                .lock()
                .map_err(|_| BackendError::DeviceError("bind-group cache mutex poisoned".into()))?;
            if let Some(cached) = cache.entries.get(&key) {
                if cached.uniform_size == uniform_bytes.len() as u64 {
                    if let Some(buf) = &cached.uniform_buffer {
                        dev.queue.write_buffer(buf, 0, uniform_bytes);
                    }
                    return Ok(cached.bind_group.clone());
                }
                // Uniform size mismatch for an otherwise-matching key: fall
                // through and rebuild below (should not happen in practice —
                // one pipeline key always pairs with one fixed uniform-struct
                // size — but handled rather than silently corrupting a
                // `write_buffer` call).
            }
        } // cache lock dropped here.

        // Miss (or size-mismatch rebuild): create the dedicated uniform
        // buffer first (needs only the device, not the buffers lock), then
        // resolve operand handles under `mem.lock_buffers()` and build the
        // bind group, then re-lock the cache just to insert.
        let uniform_buffer = if uniform_bytes.is_empty() {
            None
        } else {
            let buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: uniform_bytes.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            dev.queue.write_buffer(&buf, 0, uniform_bytes);
            Some(buf)
        };

        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let mut resolved = Vec::with_capacity(handles.len());
            for (i, &h) in handles.iter().enumerate() {
                let info = buffers
                    .get(&h)
                    .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {h}")))?;
                if let Some(&need) = min_sizes.get(i) {
                    if info.size < need {
                        return Err(BackendError::InvalidArgument(format!(
                            "{label}: handle {h} holds {} bytes, need {need}",
                            info.size
                        )));
                    }
                }
                resolved.push(&info.buffer);
            }

            let mut entries: Vec<wgpu::BindGroupEntry> = resolved
                .iter()
                .enumerate()
                .map(|(i, buf)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buf.as_entire_binding(),
                })
                .collect();
            if let Some(u) = &uniform_buffer {
                entries.push(wgpu::BindGroupEntry {
                    binding: handles.len() as u32,
                    resource: u.as_entire_binding(),
                });
            }

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: bind_group_layout,
                entries: &entries,
            })
        }; // buffers lock dropped here.

        {
            let mut cache = self
                .bind_group_cache
                .lock()
                .map_err(|_| BackendError::DeviceError("bind-group cache mutex poisoned".into()))?;
            if !cache.entries.contains_key(&key) {
                // Prune already-evicted (by `evict_handle`) stale front
                // entries first so FIFO eviction only ever drops a key that
                // is actually still live.
                while cache.order.len() >= BindGroupCache::MAX_ENTRIES {
                    match cache.order.front() {
                        Some(front) if !cache.entries.contains_key(front) => {
                            cache.order.pop_front();
                        }
                        Some(_) => {
                            if let Some(oldest) = cache.order.pop_front() {
                                cache.entries.remove(&oldest);
                            }
                            break;
                        }
                        None => break,
                    }
                }
                cache.order.push_back(key.clone());
            }
            cache.entries.insert(
                key,
                CachedBindGroup {
                    bind_group: bind_group.clone(),
                    uniform_buffer,
                    uniform_size: uniform_bytes.len() as u64,
                },
            );
        }

        Ok(bind_group)
    }

    /// Evict any bind-group cache entry that references `handle` — must be
    /// called before the handle is actually freed; see the module doc's
    /// "Freed-buffer memory" note.
    pub(super) fn evict_bind_group_cache(&self, handle: u64) -> BackendResult<()> {
        let mut cache = self
            .bind_group_cache
            .lock()
            .map_err(|_| BackendError::DeviceError("bind-group cache mutex poisoned".into()))?;
        cache.evict_handle(handle);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BindGroupCache::evict_handle` must never panic on an empty cache —
    /// the boundary case `WebGpuBackend::free` hits whenever a freed handle
    /// was never actually used in a cached dispatch. Building a real
    /// `CachedBindGroup` needs a live `wgpu::Device`, so the end-to-end
    /// behaviour (a handle's entry is actually gone after `free()`, and an
    /// unrelated entry survives) is covered against a real device by
    /// `bind_group_cache_entry_is_evicted_on_free` in
    /// `backend_tests_pipeline.rs`, via `BindGroupCache::len`.
    #[test]
    fn evict_handle_on_empty_cache_does_not_panic() {
        let mut cache = BindGroupCache::new();
        cache.evict_handle(42);
        assert_eq!(cache.len(), 0);
    }
}
