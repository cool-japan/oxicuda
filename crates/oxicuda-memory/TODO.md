# oxicuda-memory TODO

Type-safe GPU memory management with Rust ownership semantics. RAII-based wrappers around CUDA memory allocation and transfer operations. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual SLoC: 4,178** across **17 files** (estimated 70K-112K for all Vol.1 combined)

The memory crate provides the core buffer types and copy operations that all higher-level OxiCUDA crates depend on. Current implementation covers the essential buffer types with stubs for advanced features.

### Completed [x]

- [x] `device_buffer.rs` -- DeviceBuffer<T> (VRAM allocation, RAII Drop), DeviceSlice<T> (borrowed sub-range)
- [x] `host_buffer.rs` -- PinnedBuffer<T> (page-locked host memory for fast DMA transfers)
- [x] `unified.rs` -- UnifiedBuffer<T> (managed memory accessible from both host and device)
- [x] `zero_copy.rs` -- MappedBuffer<T> (zero-copy host-mapped memory)
- [x] `copy.rs` -- H2D/D2H/D2D copy helpers (copy_htod, copy_dtoh, copy_dtod) with type safety
- [x] `pool.rs` -- MemoryPool and PooledBuffer (stream-ordered allocation, feature-gated under `pool`)
- [x] `lib.rs` -- Prelude module, feature flags (pool, gpu-tests)

### Future Enhancements [ ]

- [x] Async memory pool enhancements (pool.rs) -- PoolStats (allocated/peak/count), trim(), set_threshold() (P0)
- [x] Memory pool statistics -- AllocationHistogram, FragmentationMetrics, PoolReport, PoolStatsTracker (pool_stats.rs) (P1)
- [x] CUDA 12+ managed memory hints -- ManagedMemoryHints, MigrationPolicy, PrefetchPlan (managed_hints.rs) (P1)
- [x] Virtual memory management (virtual_memory.rs) -- VirtualAddressRange, PhysicalAllocation, VirtualMemoryManager (P1)
- [x] Memory advice / prefetch hints -- cuMemAdvise, cuMemPrefetchAsync for unified memory (P1)
- [x] Multi-GPU peer copy (peer_copy.rs) -- can_access_peer, enable/disable_peer_access, copy_peer, copy_peer_async (P0)
- [x] Memory bandwidth profiling hooks -- transfer timing, throughput measurement (P2)
- [x] Async copy operations (copy.rs) -- copy_htod_async_raw, copy_dtoh_async_raw, copy_dtod_async with stream ordering (P0)
- [x] 2D/3D memory copy (copy_2d3d.rs) -- Memcpy2DParams, Memcpy3DParams, copy_2d/3d functions (P1)
- [x] Memory alignment guarantees -- 256-byte / 512-byte aligned allocation options (P2)
- [x] Buffer views and reinterpret cast -- type-safe buffer reinterpretation (P1)
- [x] Host-registered memory -- cuMemHostRegister for existing host allocations (P2)
- [x] Memory usage query -- cuMemGetInfo (free/total VRAM) (P1)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA driver API bindings | Yes |
| tracing | Warning logs in Drop impls (instead of panic) | Yes |

## Quality Status

- Warnings: 0
- Tests: 4 passing (4 doc-tests), 0 unit tests (GPU-dependent operations)
- unwrap() calls: 0
- Drop implementations log errors via tracing::warn, never panic

## Performance Targets

Memory operations are bandwidth-bound. Key targets:
- H2D/D2H copy: approach PCIe bandwidth limit (15-30 GB/s depending on gen)
- D2D copy: approach device memory bandwidth (e.g. 2 TB/s on H100)
- Pinned buffer allocation: avoid kernel round-trip overhead via page-locking
- Pool allocation: sub-microsecond from pre-allocated pool

## Notes

- MappedBuffer is currently a stub -- zero-copy requires cuMemHostAlloc with CU_MEMHOSTALLOC_DEVICEMAP
- MemoryPool requires CUDA 11.2+ and is feature-gated under `pool`
- All buffer types implement Drop for automatic deallocation
- Size mismatches and zero-length allocations return CudaError::InvalidValue

---

## Blueprint Quality Gates (Vol.1 Sec 7)

### Functional Requirements

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| F6 | `DeviceBuffer` alloc / free / copy — memory leak test under stress | P0 | [x] |
| F7 | `PinnedBuffer` async copy with stream verified correct | P0 | [x] |
| F8 | `MemoryPool` `alloc_async` / `free_async` continuous benchmark | P0 | [x] |

### Non-Functional Requirements

| # | Requirement | Target | Status |
|---|-------------|--------|--------|
| NF2 | H2D / D2H copy bandwidth | ≥ 95% of PCIe theoretical bandwidth (same as `cuMemcpy`) | [ ] Verify |
| NF4 | Memory leak detection | Zero leaks via `compute-sanitizer --tool memcheck` in CI | [ ] Verify |

---

## Numerical Accuracy / Correctness Requirements

| Operation | Requirement |
|-----------|-------------|
| `copy_from_host` → `copy_to_host` round-trip | Bitwise identical to source data for all `T: Copy` types |
| Async copy with stream sync | Same bitwise correctness, verified after `stream.synchronize()` |
| Unified memory access | CPU and GPU reads return same values after `memset`/init |

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These represent the gap between current implementation depth and blueprint-grade production requirements.

### Test Coverage Gaps
- [x] Unit test suite expansion (currently very few unit tests; rely on doc-tests)
- [ ] H2D / D2H bandwidth benchmark added to `benches/` to verify ≥ 95% PCIe (NF2)
- [x] `MemoryPool` stress test: 10K concurrent alloc_async / free_async cycles without fragmentation
- [ ] Memory leak detection CI job using `compute-sanitizer --tool memcheck` (NF4)
- [ ] Peer copy correctness test on 2+ GPU systems (copy D0→D1, verify D1 matches)

### Implementation Deepening
- [x] `DeviceBuffer::alloc_async` / `free_async` with `cuMemAllocAsync` / `cuMemFreeAsync` fully exercised (requires CUDA 11.2+ driver) — CPU-side API verified; requires driver for actual execution
- [x] Pool trim / `cuMemPoolTrimTo` to release unused pool memory to system
- [x] `MemoryPool` per-stream allocation tracking for debugging
