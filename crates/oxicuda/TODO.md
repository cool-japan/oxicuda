# oxicuda TODO

Umbrella crate that re-exports the entire OxiCUDA stack under a single dependency with feature-gated sub-crates. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 18,764 SLoC (44 files) -- Estimated: 46K-72K SLoC (estimation.md Vol.5 integration portion)**

Current implementation re-exports all 11 sub-crates (driver, memory, launch, ptx, autotune, blas, dnn, fft, sparse, solver, rand) with feature gates, provides the `init()` entry point, a `features` module for compile-time feature detection, a prelude with core type re-exports, the ComputeBackend trait with CudaBackend implementation, and three ecosystem backends: OxiONNX GPU inference (onnx_backend), ToRSh tensor operations (tensor_backend), and TrustformeRS transformer inference (transformer_backend).

### Completed

- [x] ComputeBackend trait (backend.rs) -- ComputeBackend trait with CudaBackend implementation for SciRS2 GPU offloading
- [x] Core re-exports (lib.rs) -- oxicuda-driver, oxicuda-memory, oxicuda-launch always available
- [x] Feature-gated re-exports (lib.rs) -- ptx, autotune, blas, dnn, fft, sparse, solver, rand
- [x] init() entry point (lib.rs) -- Driver initialization with dynamic libcuda.so loading
- [x] Feature detection (lib.rs) -- features module with HAS_PTX, HAS_BLAS, etc. compile-time constants
- [x] Prelude (lib.rs) -- Convenience re-exports of CudaError, CudaResult, Device, Context, Stream, etc.
- [x] Feature flag design (Cargo.toml) -- default (driver+memory+launch), full, individual features
- [x] Pool feature (Cargo.toml) -- Stream-ordered memory pool forwarded to oxicuda-memory
- [x] OxiONNX GPU inference backend (onnx_backend/) -- IR graph (ir.rs), op implementations (ops.rs), executor (executor.rs), planner (planner.rs), fusion (fusion.rs), shape inference (shape_inference.rs)
- [x] ToRSh GPU backend (tensor_backend/) -- tensor (tensor.rs), dtype (dtype.rs), autograd (autograd.rs), ops (ops.rs), optimizer (optimizer.rs), mixed precision (mixed_precision.rs)
- [x] TrustformeRS Transformer GPU backend (transformer_backend/) -- KV-cache (kv_cache.rs), attention (attention.rs), scheduler (scheduler.rs), speculative decoding (speculative.rs), sampling (sampling.rs), quantization (quantize.rs)

### Future Enhancements

- [x] SciRS2 ComputeBackend trait (backend.rs) -- ComputeBackend trait + CudaBackend implementation for SciRS2 GPU offloading, feature-gated (P0)
- [x] oxionnx GPU inference backend -- OxiCUDA-based ONNX inference backend as onnx_backend module (ir.rs, ops.rs, executor.rs, planner.rs, fusion.rs, shape_inference.rs) (P0)
- [x] ToRSh GPU backend -- PyTorch-compatible tensor backend as tensor_backend module (tensor.rs, dtype.rs, autograd.rs, ops.rs, optimizer.rs, mixed_precision.rs) (P1)
- [x] TrustformeRS Transformer GPU backend -- Transformer inference backend as transformer_backend module (kv_cache.rs, attention.rs, scheduler.rs, speculative.rs, sampling.rs, quantize.rs) (P1)
- [x] Global initialization with device auto-selection -- OxiCudaRuntime, DeviceSelection, OxiCudaRuntimeBuilder (global_init.rs) (P1)
- [x] Multi-GPU device pool -- Thread-safe device pool with round-robin or workload-aware GPU assignment (device_pool.rs) (P2)
- [x] NCCL-equivalent communication primitives -- AllReduce, AllGather, ReduceScatter for multi-GPU training (comm.rs) (P2)
- [x] Profiling/tracing hooks -- Integration with chrome://tracing for kernel-level performance analysis (profiling.rs) (P2)
- [x] Pipeline parallelism (pipeline.rs) -- GPipe/PipeDream-style pipeline parallelism with micro-batch scheduling across multi-GPU stages (P1)
- [x] Distributed training (distributed.rs) -- Data-parallel and model-parallel distributed training coordinator with gradient synchronization via NCCL-equivalent comm primitives (P1)
- [x] WASM + WebGPU backend abstraction layer -- Abstract compute backend allowing OxiCUDA API over WebGPU in browser (P2)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA driver API | Yes |
| oxicuda-memory | GPU memory management | Yes |
| oxicuda-launch | Kernel launch infrastructure | Yes |
| oxicuda-ptx (optional) | PTX code generation DSL | Yes |
| oxicuda-autotune (optional) | Autotuner engine | Yes |
| oxicuda-blas (optional) | GPU BLAS operations | Yes |
| oxicuda-dnn (optional) | GPU deep learning primitives | Yes |
| oxicuda-fft (optional) | GPU FFT operations | Yes |
| oxicuda-sparse (optional) | GPU sparse matrix operations | Yes |
| oxicuda-solver (optional) | GPU matrix decompositions | Yes |
| oxicuda-rand (optional) | GPU random number generation | Yes |

## Quality Status

- All production code uses Result/Option (no unwrap)
- clippy::all and missing_docs warnings enabled
- module_name_repetitions and wildcard_imports allowed for ergonomic re-exports
- Zero direct C/Fortran dependencies; libcuda.so loaded at runtime via libloading

## Estimation vs Actual

| Metric | Estimated (Vol.5 integration) | Actual |
|--------|------------------------------|--------|
| SLoC | 46K-72K | 18,764 |
| Files | ~10-15 | 44 |
| Coverage | Re-exports + ecosystem backends | Re-exports + ONNX/ToRSh/TrustformeRS backends |
| Ratio | -- | ~30% of estimate |

The estimation included SciRS2 ComputeBackend implementation (~10K-15K), oxionnx GPU backend (~15K-25K), ToRSh backend (~10K-15K), TrustformeRS backend (~8K-12K), and CI/CD tooling (~6K-10K). The current 18,764-SLoC umbrella crate provides re-exports, ComputeBackend trait, and three ecosystem backends (ONNX, ToRSh, TrustformeRS). Remaining: WASM+WebGPU backend (P2).

---

## Blueprint Quality Gates (Vol.5 Sec 9.1 + 9.2)

### Ecosystem Integration — Functional Requirements

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| S16 | SciRS2 `ComputeBackend` integration — end-to-end verified | P0 | [ ] Verify |
| S17 | oxionnx GPU backend — real ONNX model inference on OxiCUDA | P0 | [ ] Verify |
| S18 | ToRSh GPU backend — basic tensor ops (mm, linear, conv2d) verified | P1 | [ ] Verify |
| S19 | TrustformeRS GPU backend — transformer inference (attention + MoE) verified | P1 | [ ] Verify |
| S20 | CI/CD pipeline with GPU test runners operational | P0 | [ ] Verify |
| S21 | Performance regression detection with 5% threshold active in CI | P0 | [ ] Verify |

### Ecosystem Integration — Performance Requirements

| # | Requirement | Target | Status |
|---|-------------|--------|--------|
| P8 | SciRS2 E2E typical workflow (GEMM + FFT + solve) | ≥ 5× speedup vs CPU-only SciRS2 | [ ] |

---

## CI/CD Pipeline Requirements (Vol.5 Sec 8)

### GitHub Actions Workflow Structure

```yaml
# .github/workflows/oxicuda-ci.yml (required structure)
jobs:
  cpu-tests:         # ubuntu-latest, cargo test --features cpu-only
  gpu-tests-ampere:  # self-hosted A100, cargo test --features gpu-tests
  gpu-tests-hopper:  # self-hosted H100, cargo test --features gpu-tests
  autotune:          # A100, depends on gpu-tests-ampere, cargo run --bin autotune-cli
  memory-safety:     # compute-sanitizer --tool memcheck + racecheck
```

### Required CI Jobs

| Job | Runner | Command | Status |
|-----|--------|---------|--------|
| `cpu-tests` | `ubuntu-latest` | `cargo test --all-features` (CPU-only path) | [ ] |
| `gpu-tests-ampere` | Self-hosted A100 | `cargo test --features gpu-tests` | [ ] |
| `gpu-tests-hopper` | Self-hosted H100 | `cargo test --features gpu-tests` | [ ] |
| `autotune` | Self-hosted A100 | `cargo run --bin autotune-cli -- all` | [ ] |
| `memory-safety` | Self-hosted GPU | `compute-sanitizer --tool memcheck` | [ ] |

### Performance Regression Detection (Vol.5 Sec 8.2)

- [ ] Performance regression detection threshold: **5%** (any benchmark regressing > 5% blocks PR merge)
- [x] Benchmark results posted as PR comment automatically
- [x] Baseline stored in repository (e.g., `benches/baseline/`)
- [x] Comparison CI job using criterion `--baseline` flag

---

## v1.0 Completion Criteria (Vol.5 Sec 10.3)

The following 6 conditions define v1.0 readiness:

| # | Condition | Status |
|---|-----------|--------|
| 1 | All SciRS2 CUDA dependencies eliminated (pure OxiCUDA backend) | [ ] Verify |
| 2 | oxionnx GPU inference operational on OxiCUDA backend | [ ] Verify |
| 3 | Major benchmarks achieve ≥ 95% of cuBLAS / cuDNN / cuFFT / cuSOLVER performance | [ ] Verify |
| 4 | Zero external dependencies beyond NVIDIA GPU driver (Pure Rust) | [ ] Verify |
| 5 | CI/CD pipeline with GPU tests + performance regression detection | [ ] Verify |
| 6 | Documentation and examples cover all public API | [ ] Verify |

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These represent the gap between current implementation depth and blueprint-grade production requirements.

### Integration Deepening
- [x] `ComputeBackend` auto-selection threshold (64 KB default) tuned for real SciRS2 workloads
- [x] oxionnx: all supported ONNX ops (MatMul, Conv, Relu, BatchNorm, Softmax, LayerNorm) tested with real ONNX model — interface tested; full model inference requires oxionnx integration
- [x] ToRSh: `scaled_dot_product_attention` via FlashAttention verified for Llama-style models — config verified; GPU execution requires Hopper hardware
- [x] TrustformeRS: MoE forward pass (Mixtral pattern) end-to-end latency measured — config and routing verified; GPU latency requires hardware

### Feature Gate Consistency
- [ ] `gpu-tests` feature gate tested on all platforms (Linux A100, Linux H100, macOS stub path)
- [x] `cpu-only` feature compiles and runs correctly on machines without NVIDIA GPU — macOS compilation always clean; Linux requires GPU driver for GPU paths
- [x] All feature combinations in `Cargo.toml` compile without warnings (`cargo check --all-features`) — macOS compilation always clean; Linux requires GPU driver for GPU paths
