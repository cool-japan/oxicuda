# oxicuda-launch TODO

Type-safe GPU kernel launch infrastructure. Provides ergonomic abstractions for configuring and dispatching CUDA kernels with compile-time argument safety. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual SLoC: 4,728** (15 files; estimated 70K-112K for all Vol.1 combined)

The launch crate ties together driver (module/function) and memory (buffers) into a safe kernel dispatch API. It provides the `launch!` macro and typed kernel wrappers that prevent common launch errors.

### Completed [x]

- [x] `grid.rs` -- Dim3 type (3D grid/block dimensions), From conversions for u32/(u32,u32)/(u32,u32,u32), grid_size_for() helper
- [x] `kernel.rs` -- Kernel wrapper (Arc<Module> + Function), KernelArgs trait with tuple impls up to 12 elements, occupancy query delegation
- [x] `params.rs` -- LaunchParams struct, LaunchParamsBuilder with builder pattern (grid, block, shared_mem)
- [x] `macros.rs` -- launch!() macro for concise kernel invocation syntax
- [x] `lib.rs` -- Prelude module, re-exports of all public types

### Future Enhancements [ ]

- [x] Cooperative launch (cooperative.rs) -- CooperativeLaunch with max_active_blocks, optimal_block_size (P1)
- [x] Graph-based launch -- GraphLaunchCapture for launch recording and replay (P1)
- [x] Dynamic parallelism support -- device-side kernel launch (nested kernels) (P2)
- [x] Occupancy-based auto grid sizing (grid.rs) -- auto_grid_for(), auto_grid_2d() using occupancy API (P0)
- [x] Kernel argument serialization -- Debug/Display for kernel args, launch parameter logging (P2)
- [x] Cluster launch (Hopper+) -- ClusterDim, ClusterLaunchParams for thread block clusters (P1)
- [x] Multi-stream launch -- multi_stream_launch across streams (P1)
- [x] Launch bounds validation (error.rs, params.rs) -- LaunchParams::validate() checks block/grid/shared memory against device limits (P0)
- [x] KernelArgs for larger tuples -- extended to 24 elements via macro (P2)
- [x] Named kernel arguments -- NamedKernelArgs trait, ArgBuilder (P1)
- [x] Async launch with completion future -- integrate with Rust async/await (P2)
- [x] Launch telemetry -- timing, occupancy achieved, register usage reporting (P2)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | Module/Function/Stream types, cuLaunchKernel FFI | Yes |
| oxicuda-memory | Buffer types for kernel argument passing | Yes |

## Quality Status

- Warnings: 0
- Tests: 42 passing (31 unit + 11 doc-tests)
- unwrap() calls: 0
- Kernel owns Module via Arc -- safe shared ownership across threads

## Performance Targets

Launch overhead must be minimal to avoid becoming a bottleneck for small kernels:
- launch!() macro: zero-cost abstraction over raw cuLaunchKernel
- Argument packing: stack-allocated void* array, no heap allocation
- Grid size calculation: O(1) integer division

## Notes

- KernelArgs trait is implemented via macro for tuples of 1-12 Copy types
- The launch!() macro expands to Kernel::launch() with LaunchParams construction
- Module lifetime is managed via Arc<Module> inside Kernel, enabling safe sharing
- grid_size_for(n, block_size) computes ceil(n / block_size) for 1D launches

---

## Blueprint Quality Gates (Vol.1 Sec 7)

### Functional Requirements

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| F5 | PTX loading and E2E kernel execution — `vector_add` full round-trip test | P0 | [x] |

### Non-Functional Requirements

| # | Requirement | Target | Status |
|---|-------------|--------|--------|
| NF3 | `launch!()` macro overhead above raw `cuLaunchKernel` | < 1 μs | [ ] Verify |

---

## Architecture-Specific Deepening Opportunities

### Hopper (sm_90)
- [x] Cluster launch (`cuLaunchKernelEx`) support with cluster_dim_x/y/z parameters
- [x] Thread block cluster cooperative groups integration

### Ampere (sm_80) / Ada (sm_89)
- [x] Cooperative launch with `cuLaunchCooperativeKernel` fully tested
- [x] Graph-based launch via `CUgraph` exercised for inference loop patterns

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These represent the gap between current implementation depth and blueprint-grade production requirements.

### Benchmark / Verification Gaps
- [ ] Launch overhead microbenchmark: `launch!()` vs raw `cuLaunchKernel` — verified < 1 μs delta (NF3)
- [x] E2E vector_add integration test: PTX load via `oxicuda-ptx` → `launch!()` → verify results (F5) — CPU-side parameter chain verified; full GPU E2E requires NVIDIA hardware
- [x] Zero-allocation verified: named args same memory footprint as positional tuple
- [x] Order preservation verified in named args

### Missing Features (from Blueprint)
- [x] `KernelArgs` implementation for larger argument tuples (current maximum verified)
- [x] Named kernel argument API: named_args!() macro and FixedNamedArgs<N> struct implemented
- [x] launch_named!() ergonomic named launch syntax implemented
- [x] Async kernel launch with future/poll-based completion notification
- [x] Launch telemetry / tracing integration (`tracing` crate span per kernel launch)
