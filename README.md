# OxiCUDA

[![Crates.io](https://img.shields.io/crates/v/oxicuda.svg)](https://crates.io/crates/oxicuda)
[![Documentation](https://docs.rs/oxicuda/badge.svg)](https://docs.rs/oxicuda)
[![CI](https://github.com/cool-japan/oxicuda/workflows/CI/badge.svg)](https://github.com/cool-japan/oxicuda/actions)
[![License](https://img.shields.io/crates/l/oxicuda.svg)](LICENSE)

**Pure Rust CUDA replacement -- cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSOLVER, cuRAND and beyond in ~1.30M SLoC of safe Rust across 74 crates.**

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
|  Vol.1 Foundation (5 crates)                                  |
|  +----------+ +--------+ +---------+ +---------+ +---------+  |
|  | Driver   | | Memory | | Launch  | | Runtime | | NVRTC   |  |
|  +----------+ +--------+ +---------+ +---------+ +---------+  |
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

**Vol.1 -- Foundation** (5 crates, 27,080 SLoC)
- Dynamic driver loading via `libloading` -- zero build-time SDK dependency
- `DeviceBuffer<T>` with Rust ownership semantics -- `Send + Sync`, RAII
- Type-safe `launch!` macro with compile-time grid/block validation
- CUDA Runtime API layer for high-level device management
- NVRTC runtime JIT (CUDA-C to PTX) loaded the same way -- no `nvcc`, and
  absence degrades to a typed error rather than a panic

**Vol.2 -- PTX Codegen & Autotuner** (2 crates, 47,429 SLoC)
- Rust DSL that generates PTX IR covering SM 7.5 through SM 10.0
- Tensor Core support: WMMA, MMA, WGMMA instruction generation
- Built-in autotuner with 3-tier dispatch (cached / tuned / default)
- Disk-based PTX cache keyed by kernel hash + GPU architecture

**Vol.3 -- BLAS** (1 crate, 28,379 SLoC)
- Full BLAS Level 1/2/3 (axpy, gemv, gemm, trsm, syrk, ...)
- GEMM dispatch: SIMT, Tensor Core, Split-K paths
- Batched GEMM: standard, strided, grouped
- Precision coverage: F16, BF16, TF32, F32, F64, FP8
- Elementwise ops (relu, gelu, sigmoid, silu) and reductions (softmax, variance)

**Vol.4 -- DNN** (1 crate, 39,297 SLoC)
- Convolution: implicit GEMM, im2col, Winograd 3x3, direct, fused Conv+BN+Act
- FlashAttention forward/backward, PagedAttention, decode attention
- MoE: top-k routing, token permutation, fused MoE kernel
- Normalization: BatchNorm, LayerNorm, RMSNorm, GroupNorm
- Pooling: max, average, adaptive, global
- Resize: nearest, bilinear, bicubic
- Quantization: FP8, INT8, block-scaled FP4

**Vol.5 -- Scientific Computing** (4 crates, 62,511 SLoC)
- FFT: Stockham, radix-2/4/8, mixed-radix, Bluestein, C2C/R2C/C2R, 2D/3D
- Sparse: CSR/CSC/COO/BSR/ELL, SpMV, SpMM, SpGEMM, SDDMM, ILU(0)/IC(0)
- Solver: LU, QR, SVD, Cholesky, eigendecomp, CG, BiCGSTAB, GMRES
- Rand: Philox, MRG32k3a, XORWOW, Sobol, uniform/normal/Poisson

**Vol.6 -- Signal Processing** (1 crate, 12,276 SLoC)
- Audio: MFCC, STFT, Mel filterbank, spectral features
- Image: Gaussian blur, Sobel edge detection, morphological ops
- DCT: Types I-IV with fast algorithms
- DWT: Haar, Daubechies wavelets
- Filtering: IIR/FIR filters, Butterworth, Chebyshev
- Correlation: cross-correlation, autocorrelation

**Vol.7 -- Computation Graph** (1 crate, 6,563 SLoC)
- CUDA Graph capture API (StreamCapture, GraphCapture)
- Execution plan with dependency-sorted node scheduling
- Event-based inter-node synchronization
- Sequential + parallel graph executors

**Vol.8 -- GPU Training** (2 crates, 13,832 SLoC)
- Mixed precision training (AMP): FP16/BF16 + loss scaling
- Gradient accumulation and clipping; EMA (exponential moving average)
- LR schedulers: cosine, warmup, cyclic, polynomial
- GPU-fused optimizers: Adam, AdamW, SGD, RMSProp, LAMB
- Checkpointing (model save/load)
- Quantization: INT8/INT4/FP8 weight quantization, block-scaled

**Vol.9 -- Inference Engine** (3 crates, 17,909 SLoC)
- KV-cache with paged attention (PagedKvCache) and prefix caching
- Speculative decoding
- Distributed inference pipeline (tensor/pipeline parallelism)
- LM inference: BPE tokenizer, vocabulary management, sampling strategies

**Vol.10 -- Reinforcement Learning** (1 crate, 11,280 SLoC)
- Replay buffers: Uniform, Prioritized (PER), N-step
- Policy distributions: Categorical, Gaussian (SAC reparameterization), Deterministic
- Advantage estimators: GAE, TD(λ), V-trace, Retrace(λ)
- Loss functions: PPO, DQN, Double-DQN, SAC, TD3
- Observation/reward normalization with Welford running stats
- Environment abstractions: Env, VecEnv (auto-reset)

**Backends** (7 crates, 28,400 SLoC)
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
| `oxicuda-driver` | Driver API | FFI, device/context/stream/event/module | 16,160 | 461 |
| `oxicuda-memory` | cuMemAlloc | DeviceBuffer, PinnedBuffer, unified, pool | 6,812 | 301 |
| `oxicuda-launch` | cuLaunchKernel | Dim3, LaunchParams, `launch!` macro | 5,585 | 233 |
| `oxicuda-runtime` | CUDA Runtime | High-level cudaRT API layer | 4,955 | 126 |
| `oxicuda-nvrtc` | NVRTC | Runtime CUDA-C to PTX JIT, dlopen'd libnvrtc | 642 | 20 |
| **Vol.2 -- PTX Codegen & Autotuner** | | | | |
| `oxicuda-ptx` | nvcc / CUTLASS | PTX IR, codegen DSL, Tensor Core gen | 37,287 | 1,061 |
| `oxicuda-autotune` | -- | Search space, benchmark, tuning DB | 16,500 | 472 |
| **Vol.3 -- Linear Algebra** | | | | |
| `oxicuda-blas` | cuBLAS | BLAS L1/L2/L3, GEMM, batched, elementwise | 33,597 | 990 |
| **Vol.4 -- Deep Learning** | | | | |
| `oxicuda-dnn` | cuDNN | Conv, attention, MoE, norm, pool, quantize | 47,562 | 1,262 |
| **Vol.5 -- Scientific Computing** | | | | |
| `oxicuda-fft` | cuFFT | Stockham, radix-2/4/8, Bluestein, 1D/2D/3D | 15,182 | 437 |
| `oxicuda-sparse` | cuSPARSE | CSR/CSC/COO/BSR/ELL, SpMV, SpMM, SpGEMM | 17,851 | 463 |
| `oxicuda-solver` | cuSOLVER | LU, QR, SVD, Cholesky, eig, CG, GMRES | 24,858 | 534 |
| `oxicuda-rand` | cuRAND | Philox, MRG32k3a, Sobol, distributions | 14,300 | 435 |
| **Vol.6 -- Signal Processing** | | | | |
| `oxicuda-signal` | -- | Audio/image DSP, DCT, DWT, IIR/FIR filters | 14,440 | 512 |
| **Vol.7 -- Computation Graph** | | | | |
| `oxicuda-graph` | CUDA Graphs | Graph capture, dep-sorted exec, events | 8,122 | 299 |
| **Vol.8 -- GPU Training** | | | | |
| `oxicuda-train` | -- | AMP, grad accum/clip, LR schedulers, optimizers | 11,760 | 364 |
| `oxicuda-quant` | -- | INT8/INT4/FP8 quantization, block-scaled | 8,811 | 288 |
| **Vol.9 -- Inference Engine** | | | | |
| `oxicuda-infer` | -- | KV-cache, paged attention, speculative decode | 10,687 | 405 |
| `oxicuda-dist-infer` | -- | Tensor/pipeline parallelism, distributed infer | 7,735 | 239 |
| `oxicuda-lm` | -- | BPE tokenizer, vocab, sampling strategies | 7,025 | 275 |
| **Vol.10 -- Reinforcement Learning** | | | | |
| `oxicuda-rl` | -- | Replay buffers, policy dists, PPO/DQN/SAC/TD3 | 12,473 | 453 |
| **Backends** | | | | |
| `oxicuda-backend` | -- | Backend trait abstraction | 4,038 | 101 |
| `oxicuda-primitives` | CUB | GPU scan, reduce, sort, histogram | 10,712 | 271 |
| `oxicuda-metal` | -- | Metal compute backend (macOS) | 7,456 | 262 |
| `oxicuda-vulkan` | -- | Vulkan Compute backend | 7,493 | 151 |
| `oxicuda-webgpu` | -- | WebGPU backend | 5,736 | 226 |
| `oxicuda-rocm` | -- | AMD ROCm backend | 6,755 | 217 |
| `oxicuda-levelzero` | -- | Intel oneAPI / LevelZero backend | 8,290 | 155 |
| **Vol.17 -- Generative AI** | | | | |
| `oxicuda-gen` | -- | Diffusion (DDPM/DDIM/DPM-Solver++/Flow Matching), CFG, VAE, LoRA | 16,532 | 596 |
| **Vol.18 -- Graph Neural Networks** | | | | |
| `oxicuda-gnn` | -- | CSR/COO/Hetero graphs, GCN/GAT/GraphSAGE/GIN, pooling | 19,328 | 670 |
| **Vol.19 -- State Space Models** | | | | |
| `oxicuda-mamba` | -- | HiPPO-NPLR, S4D/S5 selective scan, Mamba SSM, RWKV | 16,546 | 680 |
| **Vol.20 -- Vision Transformers** | | | | |
| `oxicuda-vision` | -- | ViT, patch embedding, CLIP towers | 22,188 | 853 |
| **Vol.21 -- Audio/Speech ML** | | | | |
| `oxicuda-audio` | -- | Conformer, Wav2Vec2, CTC/RNN-T, WaveNet, SpecAugment, x-vector | 24,247 | 853 |
| **Vol.22 -- Time-Series Forecasting** | | | | |
| `oxicuda-timeseries` | -- | TCN, NHiTS, PatchTST, TimesNet, iTransformer, RevIN | 22,739 | 711 |
| **Vol.23 -- Bayesian Deep Learning** | | | | |
| `oxicuda-bayes` | -- | Variational inference, MC Dropout, Deep Ensembles, SWAG, Laplace | 21,192 | 675 |
| **Vol.24 -- Federated Learning** | | | | |
| `oxicuda-federated` | -- | FedAvg/FedProx/SCAFFOLD/FedAdam, DP, secure aggregation | 11,941 | 502 |
| **Vol.25 -- Neural Architecture Search** | | | | |
| `oxicuda-nas` | -- | DARTS, supernet, NSGA-II, hardware-aware FLOPs predictor | 12,000 | 389 |
| **Vol.26 -- Self-Supervised Learning** | | | | |
| `oxicuda-ssl` | -- | SimCLR/MoCo/BYOL/Barlow Twins/MAE/DINO | 14,885 | 459 |
| **Vol.27 -- Adversarial Robustness** | | | | |
| `oxicuda-adversarial` | -- | FGSM/PGD/CW/TRADES/MART | 13,971 | 549 |
| **Vol.28 -- Multi-Modal Learning** | | | | |
| `oxicuda-multimodal` | -- | Cross-modal attention, CLIP/ImageBind | 15,112 | 480 |
| **Vol.29 -- Continual Learning** | | | | |
| `oxicuda-continual` | -- | EWC/SI/PackNet/GEM/DER++ | 16,164 | 548 |
| **Vol.30 -- 3D Geometry & Point Clouds** | | | | |
| `oxicuda-geometry3d` | -- | FPS/kNN/PointNet/DGCNN/ICP | 18,373 | 557 |
| **Vol.31 -- Physics-Informed Neural Networks** | | | | |
| `oxicuda-pinn` | -- | PINN/NeuralODE/FNO/DeepONet | 22,082 | 727 |
| **Vol.32 -- RLHF & Alignment** | | | | |
| `oxicuda-rlhf` | -- | DPO/IPO/KTO/ORPO/PPO-RLHF/reward-model | 17,018 | 667 |
| **Vol.33 -- Meta-Learning** | | | | |
| `oxicuda-meta` | -- | MAML/FOMAML/ANIL/Reptile/ProtoNet | 17,440 | 530 |
| **Vol.34 -- Neural Radiance Fields** | | | | |
| `oxicuda-nerf` | -- | NeRF/Instant-NGP/Mip-NeRF/TensoRF | 14,289 | 395 |
| **Vol.35 -- Mixture of Experts** | | | | |
| `oxicuda-moe` | -- | Switch/Top-K/Expert-Choice/Soft-MoE | 11,698 | 352 |
| **Vol.36 -- Tabular Deep Learning** | | | | |
| `oxicuda-tabular` | -- | TabNet/SAINT/FT-Transformer/NODE | 22,938 | 564 |
| **Vol.37 -- Anomaly Detection** | | | | |
| `oxicuda-anomaly` | -- | DeepSVDD/LOF/COPOD/Mahalanobis/IsoForest | 25,368 | 611 |
| **Vol.38 -- Quantum Simulation** | | | | |
| `oxicuda-quantum` | -- | State-vector/VQE/QAOA/QML-kernels | 16,687 | 491 |
| **Vol.39 -- Approximate Nearest Neighbor** | | | | |
| `oxicuda-ann` | -- | HNSW/IVF/PQ/IVFPQ/LSH | 16,462 | 465 |
| **Vol.40 -- Recommender Systems** | | | | |
| `oxicuda-recsys` | -- | ALS/BPR/NCF/DeepFM/SASRec/LightGCN | 19,181 | 597 |
| **Vol.41 -- Causal Inference** | | | | |
| `oxicuda-causal` | -- | NOTEARS/IPW/S-T-X-learners/DML/CausalForest | 28,424 | 788 |
| **Vol.42 -- Parameter-Efficient Fine-Tuning** | | | | |
| `oxicuda-peft` | -- | LoRA/QLoRA/AdaLoRA/Prefix-Tuning | 23,507 | 791 |
| **Vol.43 -- Knowledge Distillation** | | | | |
| `oxicuda-distill` | -- | Hinton/FitNets/AT/CRD/DML/ZSKD | 14,792 | 530 |
| **Vol.44 -- Optimal Transport** | | | | |
| `oxicuda-ot` | -- | Sinkhorn/EMD/Gromov-Wasserstein/Wasserstein-kmeans | 26,402 | 657 |
| **Vol.45 -- Spiking Neural Networks** | | | | |
| `oxicuda-snn` | -- | LIF/IF/BPTT/STBP/SLAYER/STDP/ANN→SNN | 25,999 | 845 |
| **Vol.46 -- Differential Privacy** | | | | |
| `oxicuda-privacy` | -- | DP-FTRL/DP-Adam/RDP/zCDP/PRV/OUE/RAPPOR | 21,690 | 823 |
| **Vol.47 -- Hyperdimensional Computing** | | | | |
| `oxicuda-hdc` | -- | Binary/integer/complex HVs, AM/classifier | 16,573 | 609 |
| **Vol.48 -- Evolutionary Algorithms** | | | | |
| `oxicuda-evol` | -- | CMA-ES/NSGA-II/MOEA-D/NEAT/DE/PSO/ACO | 22,613 | 612 |
| **Vol.49 -- Topological Data Analysis** | | | | |
| `oxicuda-tda` | -- | Vietoris-Rips/persistent-homology/Mapper | 13,502 | 402 |
| **Vol.50 -- Tensor Networks** | | | | |
| `oxicuda-tn` | -- | MPS/MPO/DMRG/TEBD/PEPS/TT-cross/CP-ALS/einsum | 27,707 | 540 |
| **Vol.51 -- Sequence Models** | | | | |
| `oxicuda-seq` | -- | HMM/CRF/Kalman/EKF/Viterbi/Baum-Welch | 24,983 | 706 |
| **Vol.52 -- Numerical PDE Solvers** | | | | |
| `oxicuda-pde` | -- | FDM/FEM/spectral/multigrid/CG | 26,402 | 725 |
| **Vol.53 -- Manifold Learning** | | | | |
| `oxicuda-manifold` | -- | t-SNE/UMAP/LLE/Isomap/Diffusion-Maps/SMACOF | 28,950 | 620 |
| **Vol.54 -- Statistical Inference** | | | | |
| `oxicuda-stats` | -- | t-test/ANOVA/KS/bootstrap/regression/power | 34,954 | 1,015 |
| **Vol.55 -- Streaming Sketches** | | | | |
| `oxicuda-sketch` | -- | HyperLogLog/Count-Min/Bloom/t-Digest/MinHash | 15,757 | 583 |
| **Vol.56 -- Survival Analysis** | | | | |
| `oxicuda-survival` | -- | Kaplan-Meier/Cox-PH/AFT/Fine-Gray/Brier | 33,231 | 819 |
| **Vol.57 -- Convex Optimization** | | | | |
| `oxicuda-cvx` | -- | LP/QP/SOCP/SDP/ADMM/FISTA/proximal-gradient | 23,736 | 669 |
| **Vol.58 -- Compressed Sensing** | | | | |
| `oxicuda-cs` | -- | OMP/CoSaMP/IHT/AMP/K-SVD/LASSO/nuclear-norm | 12,350 | 291 |
| **Vol.59 -- Graph Algorithms** | | | | |
| `oxicuda-graphalg` | -- | BFS/DFS/Dijkstra/MST/flow/matching/SCC/TSP | 13,143 | 358 |
| **Vol.60 -- Numerical Analysis** | | | | |
| `oxicuda-numeric` | -- | Root-finding/quadrature/special-functions/ODE/interpolation | 17,025 | 545 |
| **Vol.61 -- 2D Computational Geometry** | | | | |
| `oxicuda-geom2d` | -- | Delaunay/Voronoi/convex-hull/sweep-line | 11,071 | 301 |
| **Umbrella** | | | | |
| `oxicuda` | -- | Umbrella re-export crate | 21,496 | 526 |
| | | **Total** | **~1,299,146** | **38,689** |

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `driver` | on | CUDA driver API layer |
| `memory` | on | Device/pinned/unified memory |
| `launch` | on | Kernel launch primitives |
| `nvrtc` | off | NVRTC runtime JIT compiler (CUDA-C to PTX) |
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

**Released (v0.2.0) -- 2026-06-16** *(32,320 tests passing, ~1.06M SLoC, 73 crates)*
- Vol.1: Driver, Memory, Launch, Runtime -- foundation layer (4 crates)
- Vol.2: PTX codegen DSL, autotuner engine (2 crates)
- Vol.3: Full BLAS L1/L2/L3 with Tensor Core GEMM, SYR2K two-operand cross-product variant
- Vol.4: Convolution, FlashAttention, MoE, normalization, pooling, quantization
- Vol.5: FFT, sparse, solver, RNG (4 crates)
- Vol.6: Signal processing -- audio/image DSP, DCT, DWT, IIR/FIR filters
- Vol.7: Computation graph -- capture API, dep-sorted scheduling, parallel executor
- Vol.8: GPU training -- AMP, optimizers, LR schedulers, checkpointing, quantization (2 crates)
- Vol.9: Inference engine -- KV-cache, speculative decode, distributed infer, LM (3 crates)
- Vol.10: Reinforcement learning -- replay buffers, policy dists, PPO/DQN/SAC/TD3
- Backends: Metal, Vulkan, WebGPU, ROCm, LevelZero (7 crates)
- Vol.17: Generative AI -- diffusion schedulers, CFG, VAE, LoRA
- Vol.18: Graph Neural Networks -- GCN/GAT/GraphSAGE/GIN, pooling
- Vol.19: State Space Models -- HiPPO-NPLR, S4D/S5, Mamba SSM, RWKV
- Vol.20: Vision Transformers & CLIP -- ViT, patch embedding, dual-tower CLIP
- Vol.21: Audio/Speech ML -- Conformer, Wav2Vec2, CTC/RNN-T, WaveNet, SpecAugment
- Vol.22: Time-Series Forecasting -- TCN, NHiTS, PatchTST, TimesNet, iTransformer, RevIN
- Vol.23: Bayesian Deep Learning -- variational inference, MC Dropout, Ensembles, Laplace
- Vol.24: Federated Learning -- FedAvg/FedProx/SCAFFOLD/FedAdam, DP, secure aggregation
- Vol.25: Neural Architecture Search -- DARTS, supernet, NSGA-II, hardware-aware predictor
- Vol.26--61: SSL, Adversarial, Multimodal, Continual, 3D Geometry, PINN, RLHF, Meta-Learning, NeRF, MoE, Tabular, Anomaly, Quantum, ANN, RecSys, Causal, PEFT, Distillation, OT, SNN, DP, HDC, Evolutionary, TDA, Tensor Networks, Sequence Models, PDE, Manifold, Statistics, Sketches, Survival, CVX, Compressed Sensing, Graph Algorithms, Numerical Analysis, 2D Geometry

**Released (v0.3.0) -- 2026-06-25** *(36,984 tests passing, ~1.23M SLoC, 73 crates)*
- Workspace-wide CPU algorithm implementation depth across all volumes
- `oxicuda-rlhf` now fully gradient-capable: 20+ loss gradients verified against finite differences
- Cross-crate integration: `oxicuda-gnn` -> `oxicuda-sparse` SpMM, `oxicuda-timeseries` -> `oxicuda-fft`
- Research cores: PointFlow CNF (`oxicuda-geometry3d`), Bark RVQ neural codec (`oxicuda-audio`)
- `oxicuda-ptx`: CpAsyncGenerator (cp.async codegen) and cost-gated FusionCostModel
- Latent-bug fixes, notably the `oxicuda-ot` network-simplex EMD solver that was silently broken for all n>=4

**Released (v0.4.0) -- 2026-07-01** *(38,093 tests passing, ~1.27M SLoC, 73 crates)*
- On-device GPU validation harness rolled out workspace-wide (60+ crates): hand-written PTX kernels JIT-compiled via `Module::from_ptx` and executed on real NVIDIA hardware (RTX A4000, sm_86, CUDA 12.4) for the first time, asserting numerical equivalence to CPU oracles rather than CPU-only parity checks
- Register-shadowing of CUDA's built-in special registers (`%tid`/`%ntid`/`%ctaid`/`%warpid`) was the single most common defect class found, affecting `oxicuda-primitives`, `oxicuda-train`, `oxicuda-ann`, `oxicuda-rl`, `oxicuda-dist-infer`, and `oxicuda-timeseries`
- Base-2/base-e log-exp mixups silently produced plausible-but-wrong results in `oxicuda-survival` (Cox model), `oxicuda-seq` (HMM), `oxicuda-ot` (Sinkhorn), `oxicuda-rlhf`, `oxicuda-nerf`, `oxicuda-gnn`, and `oxicuda-audio` -- all fixed with correct `log2(e)`/`ln(2)` scaling
- Algorithm completions: `oxicuda-pde` `fem_assemble_kernel` now a full unconstrained P1 stiffness assembly; `oxicuda-tabular` sparsemax/quantile-norm/NODE-tree kernels now exact (plus new `VarObliviousLayer` and `TabRecordLayer`); `oxicuda-moe` `soft_moe_dispatch_kernel` now the real 3-pass slot-softmax; `oxicuda-geometry3d` Gaussian-splatting `project_kernel`/`sh_eval_kernel` now full EWA covariance + 9-term spherical harmonics
- New: preconditioned conjugate gradient (`oxicuda-cvx`), GPT-NeoX half-split RoPE (`oxicuda-dnn`), pure-Rust FFT for FNO spectral convolution (`oxicuda-pinn`)
- Test suite expanded to 38,093 passing tests (`--all-features`; 37,166 with default features), up from 36,984 at 0.3.0, including large analytic test-coverage expansions across `oxicuda-rlhf`, `oxicuda-recsys`, `oxicuda-peft`, `oxicuda-meta`, `oxicuda-numeric`, `oxicuda-evol`, `oxicuda-solver`, and `oxicuda-ann`

**Released (v0.4.1) -- 2026-07-07** *(38,612 tests passing, ~1.28M SLoC, 73 crates)*
- On-device GPU validation sweep continues into `oxicuda-blas` (Level-1/Level-2/GEMM/reduction kernels plus Ampere `mma.sync` tensor-core paths), `oxicuda-dnn` (attention, RNN, convolution, normalization, pooling, quantization), and the `oxicuda-driver`/`oxicuda-runtime`/`oxicuda-memory`/`oxicuda-launch` stack itself -- context/stream/graph lifetime, peer access, and pooled-buffer reuse races that only surface under real concurrent device execution
- Sweep also reaches `oxicuda-sparse` (all major formats/ops), `oxicuda-solver`'s batched LU/Cholesky factorization (previously stub bodies), and `oxicuda-autotune`'s benchmark timing and result-database concurrency
- First-ever correctness pass across the non-CUDA GPU backends -- Vulkan/SPIR-V, WebGPU/WGSL, Metal/MSL, ROCm/HIP, Level Zero/SPIR-V -- against real backend compilers/validators, turning up dozens of invalid-shader-module and wrong-ISA-encoding bugs plus several silently-wrong GEMM, attention, and quantization computations
- Real driver-backed CUDA Graph stream capture (`cuStreamBeginCapture_v2`/`cuStreamEndCapture`, finalized into a launchable graph) lands for the first time
- Security hardening: PTX cache symlink/cache-poisoning fix (`oxicuda-ptx`), plus untrusted-input allocation-size fixes in device-printf parsing (`oxicuda-driver`) and index/adapter deserializers (`oxicuda-ann`, `oxicuda-peft`)
- Test suite expanded to 38,612 passing tests (`--all-features`; 37,288 with default features), up from 38,093/37,166 at 0.4.0

**Released (v0.5.0) -- 2026-07-14** *(38,622 tests passing, ~1.30M SLoC, 73 crates)*
- `oxicuda-ptx`: F64-precision elementwise PTX kernels declared the wrong register width (`.b32` instead of `.b64`) via a hardcoded value-register prefix, so `ptxas` rejected every F64 elementwise kernel outright -- fixed via a new precision-aware register prefix (`ElementwiseTemplate::vreg_prefix()`)
- `oxicuda-ptx`: transcendental elementwise ops (exp/log/sigmoid/gelu/silu/tanh/softplus/pow), which rely on F32-only special-function-unit instructions, now correctly reject non-F32 precision instead of silently emitting invalid PTX
- `oxicuda-ptx`: block-level reduction (sum/mean/L2-norm/etc.) hardcoded an F32 register bank and a 32-bit warp-shuffle tail, so F64 reductions failed `ptxas`; F64 now reduces via a shared-memory tree instead
- `oxicuda-blas`: GEMM `alpha`/`beta` scalars were encoded at the wrong (input, not accumulator) precision for F16/BF16/FP8 inputs, silently corrupting mixed-precision GEMM results instead of erroring -- fixed via a new `GpuFloat::to_accumulator_bits()` conversion
- `oxicuda-blas`'s `bf16_gemm_error` and `oxicuda-gen`'s `lora::merge::scale_adapter` now honor their documented contracts instead of panicking: the former returns `0.0` for degenerate (empty-product) dimensions, the latter returns `GenResult<LoraLinear>` on a reconstruction failure
- Workspace-wide: eliminated the remaining 42 production-code `.expect()` call sites across 22 crates, completing the workspace's zero-unwrap policy (0 unwrap/expect in library code now, verified)
- Test suite expanded to 38,622 passing tests (`--all-features`; 37,296 with default features), up from 38,612/37,288 at 0.4.1

**Released (v0.5.1) -- 2026-07-22** *(38,646 tests passing, ~1.30M SLoC, 74 crates)*
- New crate `oxicuda-nvrtc` -- a runtime loader for NVIDIA's NVRTC (the CUDA-C to PTX JIT compiler), completing the runtime-JIT half of the zero-SDK-dependency story alongside `oxicuda-driver`: it `dlopen`s `libnvrtc` via `libloading` with no `#[link]`, no `build.rs`, and no `-lnvrtc`, and caches the resolved function table process-wide
- Degrades gracefully on a host without NVRTC -- `is_available()` returns `false` and every entry point returns a typed `NvrtcError::Unavailable` naming the libraries that were tried; optional entry points (CUBIN retrieval, C++ name expressions, supported-arch queries) return `NvrtcError::NotSupported` on older runtimes rather than failing the whole library load
- `Ptx::as_str()` feeds `oxicuda_driver::Module::from_ptx` directly, so hand-written CUDA-C can be compiled and launched without leaving the workspace
- Exposed from the umbrella as `oxicuda::nvrtc` behind the `nvrtc` feature (off by default); 20 new tests, bringing the workspace to 74 members

**Released (v0.5.2) -- 2026-07-27** *(38,675 tests passing, ~1.30M SLoC, 74 crates)*
- `oxicuda-driver`: `Module::from_ptx`/`from_ptx_with_options` now scrub non-ASCII bytes from PTX source (new `ascii_only()` helper) before submitting it to the JIT -- `ptxas` and the driver's JIT reject any non-ASCII byte anywhere in a module, even inside `//` comments, which CUDA 11.x tolerated silently but CUDA 12.9+ toolchains reject outright with an opaque `CUDA_ERROR_INVALID_PTX`
- `oxicuda-ptx`/`oxicuda-fft`/`oxicuda-launch`/`oxicuda-signal`: generated PTX comments no longer contain typographic Unicode (em dashes, ×, →, π, √, Σ, box-drawing rules) -- replaced with ASCII equivalents, fixing the same CUDA 12.9+ rejection at its source rather than relying only on `oxicuda-driver`'s new scrubber
- `oxicuda-tn`: the DMRG excited-states penalty method now rotates each prior state's two-site tensor into the current state's block basis (new `overlap_env_left`/`overlap_env_right`/`prior_two_site_projector`) before applying the penalty, instead of using the prior state's raw two-site tensor directly -- that was only valid if both states shared a bond basis, which they don't, since each MPS is canonicalised independently
- `oxicuda`/`oxicuda-memory`: `FileLockGuard`'s retry loop now also treats Windows' `ErrorKind::PermissionDenied` as retryable lock contention rather than a fatal error, since a released lock file stays "delete pending" on Windows until the last handle closes; a `managed_hints` test now checks device capability before asserting `cuMemAdvise` succeeds, since WDDM (Windows) GPUs correctly reject it with `InvalidDevice`
- Windows portability fixes across `ptxas`-invoking tests in `oxicuda-blas`, `oxicuda-dnn`, `oxicuda-ptx`, `oxicuda-solver`, `oxicuda-sparse`, and `oxicuda-signal`: tests now find `ptxas.exe`, assemble to a throwaway file instead of `/dev/null` (not openable on Windows), and surface `ptxas`'s stdout in failure messages
- Test suite expanded to 38,675 passing tests (`--all-features`; 37,320 with default features), up from 38,646/37,316 at 0.5.1

**Released (v0.5.3) -- 2026-07-27** *(38,675 tests passing, ~1.30M SLoC, 74 crates)*
- `oxicuda-fft` (`transforms::c2c`, `c2r`, `fft2d`, `fft3d`, `r2c`): fixed an async-copy race where the shared `copy_dtoh_async`/`copy_htod_async` helpers enqueued `cuMemcpyDtoHAsync`/`cuMemcpyHtoDAsync` but returned before the copy actually completed, letting the caller read a still-in-flight (effectively zeroed) buffer or drop the source before the upload landed -- both helpers now call `stream.synchronize()` before returning
- `oxicuda-memory`: `DeviceBuffer::copy_from_host` only blocked until the host source was staged into the driver's DMA buffer, not until the transfer to device memory itself completed -- since every OxiCUDA `Stream` uses `CU_STREAM_NON_BLOCKING`, a kernel issued right after `copy_from_host` on such a stream could observe pre-upload zeros; it now also calls `cuCtxSynchronize`, mirroring `zeroed`'s existing behavior
- Test count unchanged at 38,675 passing (`--all-features`; 37,320 with default features) -- this release is a correctness fix, not a feature addition

**Released (v0.5.4) -- 2026-08-11** *(38,689 tests passing, ~1.30M SLoC, 74 crates)*
- `oxicuda-ptx`: new `WarpVec`/`WarpMask` SIMD-flavored warp-vector expression layer (`builder::warp_vec`, exported from the prelude) -- a warp's 32 lanes as a first-class `Simd`/`Mask`-style value with elementwise arithmetic, comparisons, butterfly all-reduce reductions (`redux.sync` fast path on sm_80+), Hillis-Steele scans, and the full shuffle family, built entirely on stable Rust via runtime PTX codegen
- `oxicuda-ptx`: six new IR instructions back the warp-vector layer -- `Shfl`/`Vote` (warp shuffle/vote), `Mov`/`Not` (typed moves/logic), `PackB64x2`/`UnpackB64x2` (64-bit-through-32-bit-shuffle routing) -- fully wired into the validator, dead-code, register-pressure, scheduling, and arch-legality passes
- `oxicuda-launch`: new `rustc_ptx_interop` gpu-tests launch PTX compiled by upstream nightly `rustc`'s NVPTX backend (not `oxicuda-ptx`'s own generator) directly through the driver stack, proving the driver/launch stack hosts foreign-compiler PTX -- including a `core::simd` (portable-SIMD) kernel
- Fixed a latent `Instruction::Redux` bug: bitwise warp reductions (`and`/`or`/`xor`) were emitted as `.u32` when the PTX ISA requires `.b32`, which `ptxas` rejected
- Test suite expanded to 38,689 passing tests (`--all-features`; 37,346 with default features), up from 38,675/37,320 at 0.5.3

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
