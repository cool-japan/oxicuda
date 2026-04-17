# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-04-17

### Added

- Version bump release with documentation and quality improvements across all crates

### Changed

- Updated all internal dependency versions to 0.1.3

## [0.1.2] - 2026-04-14

### Added

- Version bump release with documentation and quality improvements across all crates

### Changed

- Updated all internal dependency versions to 0.1.2

## [0.1.1] - 2026-04-14

### Added

- `oxicuda-blas`: New elementwise operations — `Ceil`, `Floor`, `HardSigmoid`, `HardSwish`, `Softplus`, and `LeakyRelu`

### Changed

- General enhancements across crates: improved robustness, performance, and internal code quality

## [0.1.0] - 2026-04-13

### Added

**Foundation (Vol.1 — 4 crates, 22,972 SLoC)**
- `oxicuda-driver` (11,548 SLoC, 333 tests): CUDA Driver API wrapper with dynamic loading via libloading, device/context/stream/event/module management, multi-GPU context pool, occupancy queries
- `oxicuda-memory` (4,178 SLoC, 204 tests): Type-safe GPU memory management — DeviceBuffer<T>, PinnedBuffer<T>, unified memory, async pool, virtual memory, 2D/3D copies, peer transfer
- `oxicuda-launch` (4,728 SLoC, 207 tests): Type-safe kernel launch — Dim3, LaunchParams, launch! macro, cooperative launch, cluster launch (Hopper+), graph-based launch
- `oxicuda-runtime` (2,518 SLoC, 46 tests): High-level CUDA runtime wrapper — streams, events, texture objects, surface objects

**PTX Codegen & Autotuner (Vol.2 — 2 crates, 43,122 SLoC)**
- `oxicuda-ptx` (29,206 SLoC, 873 tests): Full PTX IR type system, Rust DSL for SM 7.5–SM 10.0, Tensor Core support (WMMA/MMA/WGMMA), kernel templates (GEMM, elementwise, reduction, softmax, scan, transpose, attention, BN, MoE, convolution), register pressure analysis, dead code elimination, constant folding, strength reduction
- `oxicuda-autotune` (13,916 SLoC, 408 tests): Search space definition, GPU benchmarking with statistical analysis, Bayesian optimization, simulated annealing, genetic algorithm, result DB (JSON), problem size interpolation, early stopping

**Linear Algebra (Vol.3 — 1 crate, 21,845 SLoC)**
- `oxicuda-blas` (21,845 SLoC, 604 tests): Full cuBLAS equivalent — BLAS Level 1/2/3, GEMM (SIMT/Tensor Core/Split-K), batched GEMM (standard/strided/grouped), precision coverage (F16/BF16/TF32/F32/F64/FP8), elementwise ops, reductions, epilogue fusion

**Deep Learning (Vol.4 — 1 crate, 34,711 SLoC)**
- `oxicuda-dnn` (34,711 SLoC, 960 tests): Full cuDNN equivalent — convolution (implicit GEMM/im2col/Winograd/direct/fused), FlashAttention v2 (forward/backward), PagedAttention, MoE (top-k routing, permutation, fusion), normalization (BN/LN/RMSNorm/GroupNorm), pooling, resize, speculative decoding, linear layers

**Scientific Computing (Vol.5 — 4 crates, 47,946 SLoC)**
- `oxicuda-fft` (9,749 SLoC, 295 tests): Stockham FFT, radix-2/4/8, mixed-radix, Bluestein, C2C/R2C/C2R, pruned FFT, 1D/2D/3D
- `oxicuda-sparse` (12,278 SLoC, 320 tests): CSR/CSC/COO/BSR/ELL/HYB/CSR5 formats, SpMV/SpMM/SpGEMM/SDDMM, ILU(0)/IC(0), Krylov solvers, auto-dispatch
- `oxicuda-solver` (15,804 SLoC, 373 tests): Dense LU/QR/SVD/Cholesky/eigendecomp, CG/BiCGSTAB/GMRES, tensor decomposition, matrix functions (exp/log/sqrt)
- `oxicuda-rand` (10,115 SLoC, 264 tests): Philox/MRG32k3a/XORWOW/Sobol PRNGs, uniform/normal/Poisson/exponential/gamma distributions, NIST statistical tests

**Signal Processing (Vol.6 — 1 crate, 6,037 SLoC)**
- `oxicuda-signal` (6,037 SLoC, 231 tests): Audio (MFCC, STFT, Mel filterbank), image processing (Gaussian blur, Sobel, morphology), DCT (types I–IV), DWT (Haar, Daubechies), IIR/FIR filtering, correlation

**Computation Graph (Vol.7 — 1 crate, 4,784 SLoC)**
- `oxicuda-graph` (4,784 SLoC, 175 tests): CUDA Graph capture, execution plan with dependency sorting, event synchronization, sequential/parallel executors

**GPU Training (Vol.8 — 2 crates, 10,244 SLoC)**
- `oxicuda-train` (5,927 SLoC, 165 tests): Mixed precision AMP (FP16/BF16 + loss scaling), gradient accumulation/clipping, EMA, LR schedulers (cosine/warmup/cyclic/polynomial), GPU-fused optimizers (Adam/AdamW/SGD/RMSProp/LAMB), checkpointing
- `oxicuda-quant` (4,317 SLoC, 150 tests): INT8/INT4/FP8 weight quantization, block-scaled FP4, GPTQ-style post-training quantization

**Inference Engine (Vol.9 — 3 crates, 11,929 SLoC)**
- `oxicuda-infer` (4,256 SLoC, 137 tests): PagedKvCache, prefix caching, speculative decoding, continuous batching
- `oxicuda-dist-infer` (3,279 SLoC, 80 tests): Distributed inference with tensor/pipeline parallelism, all-reduce primitives
- `oxicuda-lm` (4,394 SLoC, 182 tests): BPE tokenizer, vocabulary management, sampling strategies (greedy/top-k/top-p/beam)

**Reinforcement Learning (Vol.10 — 1 crate, 4,234 SLoC)**
- `oxicuda-rl` (4,234 SLoC, 164 tests): Replay buffers (Uniform/PER/N-step), policy distributions (Categorical/Gaussian/Deterministic), advantage estimators (GAE/TD-λ/V-trace/Retrace-λ), loss functions (PPO/DQN/SAC/TD3), observation/reward normalization, Env/VecEnv abstractions

**Backends & Primitives (7 crates, 11,234 SLoC)**
- `oxicuda-backend` (271 SLoC, 7 tests): ComputeBackend trait definition
- `oxicuda-primitives` (4,372 SLoC, 142 tests): CUB-equivalent parallel primitives (block reduce/scan/sort, warp ops)
- `oxicuda-metal` (1,186 SLoC, 52 tests): Apple Metal GPU backend (macOS/iOS)
- `oxicuda-vulkan` (1,445 SLoC, 38 tests): Vulkan Compute backend (cross-platform)
- `oxicuda-webgpu` (1,108 SLoC, 42 tests): WebGPU backend (browser/WASM)
- `oxicuda-rocm` (1,087 SLoC, 36 tests): AMD ROCm/HIP backend
- `oxicuda-levelzero` (1,765 SLoC, 44 tests): Intel oneAPI Level Zero backend

**Umbrella (1 crate)**
- `oxicuda` (19,614 SLoC, 494 tests): Re-exports all sub-crates, ComputeBackend trait with CudaBackend, OxiONNX GPU inference backend, ToRSh tensor backend, TrustformeRS transformer backend, global init/device pool
