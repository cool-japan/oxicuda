# OxiCUDA

[![Crates.io](https://img.shields.io/crates/v/oxicuda.svg)](https://crates.io/crates/oxicuda)
[![Documentation](https://docs.rs/oxicuda/badge.svg)](https://docs.rs/oxicuda)
[![CI](https://github.com/cool-japan/oxicuda/workflows/CI/badge.svg)](https://github.com/cool-japan/oxicuda/actions)
[![License](https://img.shields.io/crates/l/oxicuda.svg)](LICENSE)

**Pure Rust CUDA replacement -- cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSOLVER, cuRAND and beyond in ~253K lines of safe Rust across 28 crates.**

OxiCUDA replaces the entire NVIDIA CUDA Toolkit software stack with type-safe,
memory-safe Rust code. The only runtime dependency is the NVIDIA driver
(`libcuda.so` / `nvcuda.dll`); no CUDA SDK, no `nvcc`, no C/C++ toolchain is
needed at build time. Optimized PTX assembly is generated directly from Rust
data structures, and a built-in autotuner benchmarks kernel variants per GPU
architecture to achieve near-peak throughput from Turing through Blackwell.

## Architecture

```
+---------------------------------------------------------------+
|   SciRS2  |  OxiONNX  |  TrustformeRS  |  ToRSh              |
|   (Scientific Computing / ML / Inference Ecosystem)           |
+-------------------------------+-------------------------------+
                                |
+-------------------------------v-------------------------------+
|                         OxiCUDA                               |
|                     (Pure Rust GPU)                            |
|                                                               |
|  Vol.1 Foundation (4 crates)                                  |
|  +----------+ +--------+ +---------+ +---------+             |
|  | Driver   | | Memory | | Launch  | | Runtime |             |
|  +----------+ +--------+ +---------+ +---------+             |
|                                                               |
|  Vol.2 Codegen (2 crates)                                     |
|  +-----------+ +------------+                                 |
|  | PTX Gen   | | Autotune   |                                 |
|  +-----------+ +------------+                                 |
|                                                               |
|  Vol.3 Linear Algebra    Vol.4 Deep Learning                  |
|  +-------------+         +-------------+                      |
|  | BLAS        |         | DNN         |                      |
|  +-------------+         +-------------+                      |
|                                                               |
|  Vol.5 Scientific Computing (4 crates)                        |
|  +------+ +--------+ +--------+ +------+                     |
|  | FFT  | | Sparse | | Solver | | Rand |                     |
|  +------+ +--------+ +--------+ +------+                     |
|                                                               |
|  Vol.6 Signal    Vol.7 Comp.Graph  Vol.8 Training (2)         |
|  +---------+     +----------+      +-------+ +-------+        |
|  | Signal  |     | Graph    |      | Train | | Quant |        |
|  +---------+     +----------+      +-------+ +-------+        |
|                                                               |
|  Vol.9 Inference (3 crates)        Vol.10 RL                  |
|  +-------+ +------------+ +----+   +------+                   |
|  | Infer | | Dist-Infer | | LM |   |  RL  |                   |
|  +-------+ +------------+ +----+   +------+                   |
|                                                               |
|  Backends (7 crates)                                          |
|  +----------+ +--------+ +-------+ +--------+                 |
|  | backend  | | prims  | | Metal | | Vulkan |                 |
|  +----------+ +--------+ +-------+ +--------+                 |
|  +--------+ +-------+ +-----------+                           |
|  | WebGPU | | ROCm  | | LevelZero |                           |
|  +--------+ +-------+ +-----------+                           |
+-------------------------------+-------------------------------+
                                |
+-------------------------------v-------------------------------+
|              libcuda.so  (NVIDIA Driver, runtime only)        |
|              No SDK  /  No nvcc  /  No C Toolchain            |
+---------------------------------------------------------------+
```

## Feature Highlights

**Vol.1 -- Foundation** (4 crates, 23,025 SLoC)
- Dynamic driver loading via `libloading` -- zero build-time SDK dependency
- `DeviceBuffer<T>` with Rust ownership semantics -- `Send + Sync`, RAII
- Type-safe `launch!` macro with compile-time grid/block validation
- CUDA Runtime API layer for high-level device management

**Vol.2 -- PTX Codegen & Autotuner** (2 crates, 43,354 SLoC)
- Rust DSL that generates PTX IR covering SM 7.5 through SM 10.0
- Tensor Core support: WMMA, MMA, WGMMA instruction generation
- Built-in autotuner with 3-tier dispatch (cached / tuned / default)
- Disk-based PTX cache keyed by kernel hash + GPU architecture

**Vol.3 -- BLAS** (1 crate, 21,845 SLoC)
- Full BLAS Level 1/2/3 (axpy, gemv, gemm, trsm, syrk, ...)
- GEMM dispatch: SIMT, Tensor Core, Split-K paths
- Batched GEMM: standard, strided, grouped
- Precision coverage: F16, BF16, TF32, F32, F64, FP8
- Elementwise ops (relu, gelu, sigmoid, silu) and reductions (softmax, variance)

**Vol.4 -- DNN** (1 crate, 34,711 SLoC)
- Convolution: implicit GEMM, im2col, Winograd 3x3, direct, fused Conv+BN+Act
- FlashAttention forward/backward, PagedAttention, decode attention
- MoE: top-k routing, token permutation, fused MoE kernel
- Normalization: BatchNorm, LayerNorm, RMSNorm, GroupNorm
- Pooling: max, average, adaptive, global
- Resize: nearest, bilinear, bicubic
- Quantization: FP8, INT8, block-scaled FP4

**Vol.5 -- Scientific Computing** (4 crates, 47,946 SLoC)
- FFT: Stockham, radix-2/4/8, mixed-radix, Bluestein, C2C/R2C/C2R, 2D/3D
- Sparse: CSR/CSC/COO/BSR/ELL, SpMV, SpMM, SpGEMM, SDDMM, ILU(0)/IC(0)
- Solver: LU, QR, SVD, Cholesky, eigendecomp, CG, BiCGSTAB, GMRES
- Rand: Philox, MRG32k3a, XORWOW, Sobol, uniform/normal/Poisson

**Vol.6 -- Signal Processing** (1 crate, 6,061 SLoC)
- Audio: MFCC, STFT, Mel filterbank, spectral features
- Image: Gaussian blur, Sobel edge detection, morphological ops
- DCT: Types I-IV with fast algorithms
- DWT: Haar, Daubechies wavelets
- Filtering: IIR/FIR filters, Butterworth, Chebyshev
- Correlation: cross-correlation, autocorrelation

**Vol.7 -- Computation Graph** (1 crate, 4,802 SLoC)
- CUDA Graph capture API (StreamCapture, GraphCapture)
- Execution plan with dependency-sorted node scheduling
- Event-based inter-node synchronization
- Sequential + parallel graph executors

**Vol.8 -- GPU Training** (2 crates, 10,247 SLoC)
- Mixed precision training (AMP): FP16/BF16 + loss scaling
- Gradient accumulation and clipping; EMA (exponential moving average)
- LR schedulers: cosine, warmup, cyclic, polynomial
- GPU-fused optimizers: Adam, AdamW, SGD, RMSProp, LAMB
- Checkpointing (model save/load)
- Quantization: INT8/INT4/FP8 weight quantization, block-scaled

**Vol.9 -- Inference Engine** (3 crates, 11,929 SLoC)
- KV-cache with paged attention (PagedKvCache) and prefix caching
- Speculative decoding
- Distributed inference pipeline (tensor/pipeline parallelism)
- LM inference: BPE tokenizer, vocabulary management, sampling strategies

**Vol.10 -- Reinforcement Learning** (1 crate, 4,522 SLoC)
- Replay buffers: Uniform, Prioritized (PER), N-step
- Policy distributions: Categorical, Gaussian (SAC reparameterization), Deterministic
- Advantage estimators: GAE, TD(λ), V-trace, Retrace(λ)
- Loss functions: PPO, DQN, Double-DQN, SAC, TD3
- Observation/reward normalization with Welford running stats
- Environment abstractions: Env, VecEnv (auto-reset)

**Backends** (7 crates, 19,665 SLoC)
- Backend trait abstraction for multi-GPU-runtime portability
- CUB-equivalent GPU primitives (scan, reduce, sort, histogram)
- Metal (macOS), Vulkan Compute, WebGPU, AMD ROCm, Intel oneAPI (LevelZero)

## Pure Rust, Minimal Dependencies

OxiCUDA is built on a strict **Pure Rust** policy with minimal external
dependencies. The entire codebase compiles with `cargo build` alone -- no
C compiler, no Fortran runtime, no CUDA SDK, no `nvcc`, no `pkg-config`.

| Dependency | Purpose | Type |
|------------|---------|------|
| `libloading` | Dynamic `.so`/`.dll` loading at runtime | Pure Rust |
| `thiserror` | Ergonomic error type derivation | Pure Rust |
| `num-complex` | Complex number types (FFT) | Pure Rust |
| `half` | FP16/BF16 types (optional) | Pure Rust |
| `serde` / `serde_json` | Autotune result DB (optional) | Pure Rust |

The only runtime requirement is the NVIDIA GPU driver (`libcuda.so` on Linux,
`nvcuda.dll` on Windows). On macOS the crate compiles but returns
`UnsupportedPlatform` at runtime.

## Quick Start

```rust
use oxicuda::prelude::*;

fn main() -> Result<(), oxicuda::Error> {
    // Initialize driver and select GPU device
    let device = Device::get(0)?;
    let ctx = Context::new(device)?;
    let stream = Stream::new(&ctx)?;

    // Allocate device memory
    let mut d_a = DeviceBuffer::<f32>::zeroed(1024)?;
    let mut d_b = DeviceBuffer::<f32>::zeroed(1024)?;
    let mut d_c = DeviceBuffer::<f32>::zeroed(1024)?;

    // Copy host data to device
    d_a.copy_from_host(&host_a)?;
    d_b.copy_from_host(&host_b)?;

    // Launch a GEMM: C = alpha * A @ B + beta * C
    let handle = BlasHandle::new(&stream)?;
    handle.gemm(
        Transpose::None, Transpose::None,
        m, n, k,
        1.0f32,            // alpha
        &d_a, lda,
        &d_b, ldb,
        0.0f32,            // beta
        &mut d_c, ldc,
    )?;

    stream.synchronize()?;

    // Copy result back to host
    let mut result = vec![0.0f32; m * n];
    d_c.copy_to_host(&mut result)?;
    Ok(())
}
```

## Crate Overview

| Crate | CUDA Equivalent | Description | SLoC | Tests |
|-------|-----------------|-------------|------|-------|
| **Vol.1 -- Foundation** | | | | |
| `oxicuda-driver` | Driver API | FFI, device/context/stream/event/module | 11,601 | 333 |
| `oxicuda-memory` | cuMemAlloc | DeviceBuffer, PinnedBuffer, unified, pool | 4,178 | 204 |
| `oxicuda-launch` | cuLaunchKernel | Dim3, LaunchParams, `launch!` macro | 4,728 | 207 |
| `oxicuda-runtime` | CUDA Runtime | High-level cudaRT API layer | 2,518 | 46 |
| **Vol.2 -- PTX Codegen & Autotuner** | | | | |
| `oxicuda-ptx` | nvcc / CUTLASS | PTX IR, codegen DSL, Tensor Core gen | 29,438 | 873 |
| `oxicuda-autotune` | -- | Search space, benchmark, tuning DB | 13,916 | 408 |
| **Vol.3 -- Linear Algebra** | | | | |
| `oxicuda-blas` | cuBLAS | BLAS L1/L2/L3, GEMM, batched, elementwise | 21,845 | 604 |
| **Vol.4 -- Deep Learning** | | | | |
| `oxicuda-dnn` | cuDNN | Conv, attention, MoE, norm, pool, quantize | 34,711 | 960 |
| **Vol.5 -- Scientific Computing** | | | | |
| `oxicuda-fft` | cuFFT | Stockham, radix-2/4/8, Bluestein, 1D/2D/3D | 9,745 | 295 |
| `oxicuda-sparse` | cuSPARSE | CSR/CSC/COO/BSR/ELL, SpMV, SpMM, SpGEMM | 12,278 | 320 |
| `oxicuda-solver` | cuSOLVER | LU, QR, SVD, Cholesky, eig, CG, GMRES | 15,804 | 373 |
| `oxicuda-rand` | cuRAND | Philox, MRG32k3a, Sobol, distributions | 10,115 | 264 |
| **Vol.6 -- Signal Processing** | | | | |
| `oxicuda-signal` | -- | Audio/image DSP, DCT, DWT, IIR/FIR filters | 6,061 | 231 |
| **Vol.7 -- Computation Graph** | | | | |
| `oxicuda-graph` | CUDA Graphs | Graph capture, dep-sorted exec, events | 4,802 | 175 |
| **Vol.8 -- GPU Training** | | | | |
| `oxicuda-train` | -- | AMP, grad accum/clip, LR schedulers, optimizers | 5,929 | 165 |
| `oxicuda-quant` | -- | INT8/INT4/FP8 quantization, block-scaled | 4,318 | 150 |
| **Vol.9 -- Inference Engine** | | | | |
| `oxicuda-infer` | -- | KV-cache, paged attention, speculative decode | 4,256 | 137 |
| `oxicuda-dist-infer` | -- | Tensor/pipeline parallelism, distributed infer | 3,279 | 80 |
| `oxicuda-lm` | -- | BPE tokenizer, vocab, sampling strategies | 4,394 | 182 |
| **Vol.10 -- Reinforcement Learning** | | | | |
| `oxicuda-rl` | -- | Replay buffers, policy dists, PPO/DQN/SAC/TD3 | 4,522 | 164 |
| **Backends** | | | | |
| `oxicuda-backend` | -- | Backend trait abstraction | 271 | 10 |
| `oxicuda-primitives` | CUB | GPU scan, reduce, sort, histogram | 4,446 | 142 |
| `oxicuda-metal` | -- | Metal compute backend (macOS) | 3,328 | 119 |
| `oxicuda-vulkan` | -- | Vulkan Compute backend | 3,377 | 86 |
| `oxicuda-webgpu` | -- | WebGPU backend | 2,334 | 91 |
| `oxicuda-rocm` | -- | AMD ROCm backend | 1,995 | 68 |
| `oxicuda-levelzero` | -- | Intel oneAPI / LevelZero backend | 3,914 | 82 |
| **Umbrella** | | | | |
| `oxicuda` | -- | Umbrella re-export crate | 19,614 | 494 |
| | | **Total** | **253,125** | **7,263** |

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `driver` | on | CUDA driver API layer |
| `memory` | on | Device/pinned/unified memory |
| `launch` | on | Kernel launch primitives |
| `ptx` | off | PTX IR codegen DSL |
| `autotune` | off | Runtime autotuner with disk cache |
| `blas` | off | BLAS L1/L2/L3 and GEMM |
| `dnn` | off | Deep learning ops (conv, attention, MoE, norm) |
| `fft` | off | FFT transforms |
| `sparse` | off | Sparse matrix operations |
| `solver` | off | Linear solvers (LU, QR, SVD, Cholesky, CG) |
| `rand` | off | GPU random number generation |
| `primitives` | off | CUB-equivalent GPU primitives |
| `pool` | off | Async memory pool (CUDA 11.2+) |
| `vulkan` | off | Vulkan Compute backend |
| `metal` | off | Metal backend (macOS) |
| `webgpu` | off | WebGPU backend |
| `rocm` | off | AMD ROCm backend |
| `level-zero` | off | Intel oneAPI / LevelZero backend |
| `wasm-backend` | off | WebAssembly + WebGPU browser target |
| `gpu-tests` | off | Enable GPU hardware tests |
| `full` | off | Enable all features |

## Performance Targets

| Operation | Target vs CUDA | Notes |
|-----------|----------------|-------|
| SGEMM (FP32) | >= 95% cuBLAS | Autotuned tile sizes |
| HGEMM (FP16) | >= 95% cuBLAS | Tensor Core WMMA/MMA |
| Batch GEMM | >= 95% cuBLAS | Stream-K scheduling |
| Convolution (FP16) | >= 90% cuDNN | Implicit GEMM + Winograd |
| FlashAttention | >= 90% FA2 | Tiled, causal mask |
| FFT (power-of-2) | >= 90% cuFFT | Stockham radix-2/4/8 |
| SpMV (CSR) | >= 85% cuSPARSE | Architecture-tuned |
| LU / QR / SVD | >= 85% cuSOLVER | Blocked panel factorization |

## Supported GPU Architectures

| Architecture | SM | Codename | Key Features |
|--------------|----|----------|--------------|
| Turing | 7.5 | TU10x | INT8 Tensor Cores, RT Cores |
| Ampere | 8.0 | GA100 | TF32, FP64 Tensor Cores, Async Copy |
| Ampere | 8.6 | GA10x | Third-gen Tensor Cores |
| Ada Lovelace | 8.9 | AD10x | FP8 Tensor Cores |
| Hopper | 9.0 | GH100 | WGMMA, TMA, FP8, DPX |
| Blackwell | 10.0 | GB10x | FP4, Fifth-gen Tensor Cores |

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux x86_64 | Full support | Primary development target |
| Windows x86_64 | Full support | nvcuda.dll loaded at runtime |
| macOS (ARM/x86) | Compile-only | Returns `UnsupportedPlatform` at runtime |

## Building

```bash
# Default build (no GPU features)
cargo build

# With all GPU features
cargo build --features "ptx,autotune,blas,dnn,fft,sparse,solver,rand"

# Full build (all features including backends)
cargo build --features full

# Check without GPU
cargo check --all-targets
```

## Testing

```bash
# Unit tests (no GPU required)
cargo test

# Full test suite with GPU hardware
cargo test --features gpu-tests

# Run with nextest
cargo nextest run --all-features
```

## Roadmap

**Released (v0.1.2) -- 2026-04-14**
- Vol.1: Driver, Memory, Launch, Runtime -- foundation layer (4 crates, 23,025 SLoC)
- Vol.2: PTX codegen DSL, autotuner engine (2 crates, 43,354 SLoC)
- Vol.3: Full BLAS L1/L2/L3 with Tensor Core GEMM (21,845 SLoC)
- Vol.4: Convolution, FlashAttention, MoE, normalization, pooling, quantization (34,711 SLoC)
- Vol.5: FFT, sparse, solver, RNG (4 crates, 47,946 SLoC)
- Vol.6: Signal processing -- audio/image DSP, DCT, DWT, IIR/FIR filters (6,061 SLoC)
- Vol.7: Computation graph -- capture API, dep-sorted scheduling, parallel executor (4,802 SLoC)
- Vol.8: GPU training -- AMP, optimizers, LR schedulers, checkpointing, quantization (2 crates, 10,247 SLoC)
- Vol.9: Inference engine -- KV-cache, speculative decode, distributed infer, LM (3 crates, 11,929 SLoC)
- Vol.10: Reinforcement learning -- replay buffers, policy dists, PPO/DQN/SAC/TD3 (4,522 SLoC)
- Backends: Metal, Vulkan, WebGPU, ROCm, LevelZero (7 crates, 19,665 SLoC)

**Next**
- Published documentation on docs.rs
- GPU hardware benchmark validation (CI regression tracking)
- v1.0 completion criteria verification (see TODO.md)

## Quick Links

- [API Documentation](https://docs.rs/oxicuda)
- [GitHub Repository](https://github.com/cool-japan/oxicuda)
- [COOLJAPAN Ecosystem](https://github.com/cool-japan)
- [Changelog](CHANGELOG.md)

## Related COOLJAPAN Projects

| Project | Description |
|---------|-------------|
| [SciRS2](https://github.com/cool-japan/scirs2) | Scientific computing (NumPy/SciPy equivalent) |
| [ToRSh](https://github.com/cool-japan/torsh) | Tensor operations (PyTorch equivalent) |
| [TrustformeRS](https://github.com/cool-japan/trustformers) | Transformer models |
| [OxiONNX](https://github.com/cool-japan/oxionnx) | ONNX neural network inference |
| [OxiBLAS](https://github.com/cool-japan/oxiblas) | Pure Rust BLAS |
| [OxiFFT](https://github.com/cool-japan/oxifft) | Pure Rust FFT |

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Copyright

(C) 2026 COOLJAPAN OU (Team KitaSan)
