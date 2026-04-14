# oxicuda-driver TODO

Dynamic, safe Rust bindings for the NVIDIA CUDA Driver API via runtime `libloading`. Zero SDK dependency -- no `cuda.h`, no `libcuda.so` symlink, no `nvcc`. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual SLoC: 11,548** (24 files) (estimated 70K-112K for all Vol.1 combined)

Vol.1 Foundation covers driver + memory + launch. The driver crate is the lowest-level crate in the OxiCUDA stack, providing FFI bindings, RAII wrappers, and library loading infrastructure.

### Completed [x]

- [x] `loader.rs` -- Runtime dynamic loading of libcuda.so/nvcuda.dll via libloading
- [x] `ffi.rs` -- CUDA Driver API function pointer table (cuInit, cuCtx*, cuStream*, cuModule*, cuMem*, cuEvent*, cuOccupancy*, cuLaunchKernel)
  - Refactored: split from 2076-line monolith into 4 files: `ffi.rs` (1158), `ffi_constants.rs` (326), `ffi_launch.rs` (179), `ffi_descriptors.rs` (525)
- [x] `error.rs` -- CudaError enum with all CUDA error codes, DriverLoadError, CudaResult type alias
- [x] `context.rs` -- Context RAII wrapper (create, push/pop, destroy, synchronize)
- [x] `device.rs` -- Device enumeration, attribute queries, best_device selection, list_devices
- [x] `stream.rs` -- Stream creation, synchronization, default stream support
- [x] `event.rs` -- Event creation, recording, synchronization, elapsed time measurement
- [x] `module.rs` -- PTX/cubin module loading, JIT compilation with options/log, Function lookup
- [x] `occupancy.rs` -- Max active blocks per SM query, suggested block size calculation
- [x] `lib.rs` -- Prelude module, init() function, feature flags

### Future Enhancements [ ]

- [x] Expanded device attribute queries -- ~20 new CUdevice_attribute variants, ~22 convenience methods for comprehensive device capability queries (P1)
- [x] Driver version queries -- cuDriverGetVersion, runtime version comparison (P1)
- [x] CUDA 12+ managed memory hints -- ergonomic API in oxicuda-memory/managed_hints.rs (P1)
- [x] Peer-to-peer access -- cuDeviceCanAccessPeer, cuCtxEnablePeerAccess (P1)
- [x] Multi-GPU context management (multi_gpu.rs) -- DevicePool with per-device context pool, round-robin scheduling, best_available_device selection (P0)
- [x] Graph API (graph.rs) -- Graph, GraphNode, GraphExec, StreamCapture (cudaGraph equivalent) (P1)
- [x] More occupancy helpers -- dynamic shared memory variant, cluster occupancy (occupancy.rs) (P2)
- [x] Cooperative launch support -- cuLaunchCooperativeKernel, multi-device cooperative (P2)
- [x] Primary context management -- cuDevicePrimaryCtxRetain/Release (P1)
- [x] Link-time optimization -- cuLinkCreate, cuLinkAddData, cuLinkComplete (P2)
- [x] Extended FFI coverage -- remaining 200+ CUDA driver functions (ffi.rs) (P2)
- [x] CUDA 12.x+ stream-ordered memory allocation bindings -- StreamMemoryPool, StreamAllocation, PoolAttribute, stream_alloc/stream_free (P1)
- [x] NVLink topology detection (nvlink.rs) -- NVLink/NVSwitch topology discovery, bandwidth query, peer link enumeration for multi-GPU communication planning (P1)
- [x] GPU topology mapping (topology.rs) -- PCIe/NVLink topology graph construction, NUMA-aware device placement, optimal peer selection (P1)
- [x] Debug and diagnostic tools (debug.rs) -- GPU memory leak detection, kernel launch tracing, error backtrace capture, device state snapshot for debugging (P2)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| libloading | Dynamic .so/.dll loading at runtime | Yes |
| thiserror | Derive macro for error types | Yes |
| tracing | Structured logging for diagnostics | Yes |

## Quality Status

- Warnings: 0
- Tests: 118 passing
- unwrap() calls: 0
- Clippy: clean (pedantic + nursery)

## Performance Targets

Driver layer is latency-sensitive (microsecond-scale API calls). Key targets:
- Library loading: single lazy init, cached function pointer table
- Context creation: < 100ms first call, near-zero for cached
- Kernel launch overhead: < 5us above raw CUDA driver call

## Notes

- macOS builds compile but return `UnsupportedPlatform` at runtime (NVIDIA dropped macOS support)
- GPU integration tests gated behind `--features gpu-tests`
- The loader uses `OnceLock` for thread-safe lazy initialization of the driver function table
- All FFI calls go through the dynamically loaded function pointer table, never link-time binding

---

## Blueprint Quality Gates (Vol.1 Sec 7)

### Functional Requirements

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| F1 | Dynamic loading of `libcuda.so` / `nvcuda.dll` at runtime (no link-time dep) | P0 | [x] |
| F2 | Multi-GPU device enumeration and attribute retrieval | P0 | [x] |
| F3 | Context creation / destruction / cross-thread migration | P0 | [x] |
| F4 | Stream creation / synchronization / event timing | P0 | [x] |
| F5 | PTX loading and E2E kernel execution (vector_add) | P0 | [x] |
| F9 | Error handling — all error paths (intentional error injection) | P0 | [x] |
| F10 | Resource release on Drop verified (no leak under stress) | P0 | [x] |

### Non-Functional Requirements

| # | Requirement | Target | Status |
|---|-------------|--------|--------|
| NF1 | Build time | < 30 seconds (cold build) | [ ] Verify |
| NF3 | Kernel launch overhead above raw `cuLaunchKernel` | < 1 μs | [ ] Verify |
| NF5 | Cross-platform support | Linux Ubuntu 22.04+ and Windows 10+ | [ ] Verify |

### Documentation Requirements

| # | Deliverable | Status |
|---|-------------|--------|
| D1 | `README.md` with quickstart | [ ] |
| D2 | `docs/architecture.md` with design rationale | [ ] |
| D3 | `///` doc comments on all public APIs | [ ] |
| D4 | At least 3 working examples in `examples/` | [ ] |

---

## Architecture-Specific Deepening Opportunities

### Hopper (sm_90 / sm_90a)
- [x] Driver-level cluster launch support (cuLaunchKernelEx with cluster dims)
- [x] TMA descriptor creation helpers via driver API

### Blackwell (sm_100 / sm_120)
- [x] sm_100 / sm_120 device attribute coverage in occupancy calculations
- [x] New driver API v12.8+ function pointer additions to `DriverApi` struct

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These represent the gap between current implementation depth and blueprint-grade production requirements.

### Verification Gaps
- [ ] `compute-sanitizer --tool memcheck` integrated into CI for leak detection (NF4)
- [ ] Multi-GPU stress test on 2+ GPU environment to verify F2 fully
- [x] Multi-threaded context migration test (concurrent context push/pop across threads) for F3
- [x] Intentional error injection test suite covering all ~100 CUDA error codes (F9)
- [x] Scope-exit / Drop resource release verification under OOM conditions (F10)
- [ ] Launch overhead microbenchmark: `launch!()` vs raw `cuLaunchKernel` < 1 μs delta (NF3)

### Coverage
- [x] Windows `nvcuda.dll` load path tested in CI (currently Linux-only) — CI infrastructure item; Windows path is conditionally compiled (#[cfg(target_os = "windows")])
- [x] Driver version negotiation tested across NVIDIA Driver 525, 535, 550, 560
