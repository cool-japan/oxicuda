# OxiCUDA Architecture

Pure Rust CUDA replacement for the COOLJAPAN ecosystem.
(C) 2026 COOLJAPAN OU (Team KitaSan) — Apache-2.0

---

## Table of Contents

1. [Overview](#overview)
2. [Crate Dependency Graph](#crate-dependency-graph)
3. [Driver Layer Architecture](#driver-layer-architecture)
4. [PTX IR Design](#ptx-ir-design)
5. [Autotuner Architecture](#autotuner-architecture)
6. [BLAS Architecture](#blas-architecture)
7. [DNN Architecture](#dnn-architecture)
8. [FFT Architecture](#fft-architecture)
9. [Sparse Architecture](#sparse-architecture)
10. [Solver Architecture](#solver-architecture)
11. [GPU Architecture Coverage](#gpu-architecture-coverage)
12. [Pure Rust Design Decisions](#pure-rust-design-decisions)
13. [Testing Strategy](#testing-strategy)

---

## Overview

OxiCUDA is a complete, pure Rust GPU compute stack designed as a drop-in replacement
for NVIDIA's CUDA ecosystem (cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSOLVER, cuRAND). It
exposes all GPU operations through safe Rust APIs backed by dynamically-loaded CUDA
driver symbols — no CUDA SDK, no `nvcc`, no C toolchain required at compile time.

**Why Pure Rust?**

- **Supply-chain safety**: All library code is auditable Rust. No opaque C/Fortran blobs.
- **Cross-platform compilation**: Builds on any platform (macOS, Linux, Windows) even without a
  GPU installed; GPU operations return `UnsupportedPlatform` at runtime on non-NVIDIA systems.
- **Fearless concurrency**: Rust's ownership and lifetime system enforces correct stream
  ordering and prevents data races on device memory at compile time.
- **No-unwrap contract**: Every fallible operation returns `Result<T, E>`. The entire
  183 K SLoC codebase contains zero `unwrap()` or `expect()` calls in library code.
- **COOLJAPAN ecosystem fit**: Designed to replace NVIDIA's proprietary libraries inside
  SciRS2, ToRSh, TrustformeRS, and OxiONNX without linking against any closed binaries.

**Project statistics (v0.1.0, 2026-04-11)**

**Project statistics (v0.1.2, 2026-04-14)**

| Metric | Value |
|--------|-------|
| Workspace crates | 28 (27 library crates + 1 umbrella) |
| Rust source files | 755 |
| Source lines of code | 253,125 SLoC |
| Test cases | 7,263 passing, 2 skipped (GPU-only on macOS) |
| Compiler warnings | 0 |
| Clippy warnings | 0 |
| `unwrap()` calls | 0 |
| C / Fortran build deps | 0 (default features) |

---

## Crate Dependency Graph

OxiCUDA is organized into five volumes of increasing abstraction, plus an umbrella
crate and a set of alternative compute backends.

OxiCUDA is organized into **ten volumes** of increasing abstraction, plus an umbrella
crate and a set of alternative compute backends.

```
                        oxicuda  (umbrella)
                            |
        ┌───────────────────┼───────────────────────┐
        │                   │                       │
   oxicuda-blas        oxicuda-dnn            oxicuda-fft
   oxicuda-sparse      oxicuda-solver         oxicuda-rand
        │                   │                       │
        └───────────────────┴───────────────────────┘
                            │
                    oxicuda-autotune
                            │
                      oxicuda-ptx
                            │
                    oxicuda-launch
                            │
                    oxicuda-memory
                            │
                    oxicuda-driver
                            │
                   oxicuda-backend  (trait crate)

Alternative compute backends (feature-gated):
  oxicuda-vulkan     (Vulkan Compute / SPIR-V)
  oxicuda-metal      (Apple Metal, macOS)
  oxicuda-webgpu     (wgpu / WebGPU / WASM)
  oxicuda-rocm       (AMD HIP runtime)
  oxicuda-levelzero  (Intel oneAPI / Level Zero)
```
                    ```
                          oxicuda  (umbrella)
                              │
                         ┌──────────────────────┼──────────────────────────┐
                         │                      │                          │
                      oxicuda-lm         oxicuda-infer           oxicuda-dist-infer   (Vol.9 Inference)
                      oxicuda-rl         oxicuda-train           oxicuda-quant        (Vol.8/10)
                      oxicuda-graph      oxicuda-signal                               (Vol.6/7)
                         │                      │                          │
                      oxicuda-blas        oxicuda-dnn            oxicuda-fft
                      oxicuda-sparse      oxicuda-solver         oxicuda-rand          (Vol.3-5)
                         │                      │                          │
                         └──────────────────────┴──────────────────────────┘
                              │
                            oxicuda-autotune
                              │
                              oxicuda-ptx
                              │
                            oxicuda-launch
                              │
                            oxicuda-memory
                              │
                            oxicuda-driver
                              │
                           oxicuda-backend + oxicuda-primitives  (trait + GPU primitives)

                    Alternative compute backends (feature-gated):
                      oxicuda-vulkan     (Vulkan Compute / SPIR-V)
                      oxicuda-metal      (Apple Metal, macOS)
                      oxicuda-webgpu     (wgpu / WebGPU / WASM)
                      oxicuda-rocm       (AMD HIP runtime)
                      oxicuda-levelzero  (Intel oneAPI / Level Zero)
                    ```

### Vol.1 — Foundation

| Crate | SLoC | Role |
|-------|------|------|
| `oxicuda-backend` | — | `ComputeBackend` trait definition |
| `oxicuda-driver` | 11,601 | Runtime CUDA driver loader, context/stream/event/graph management |
| `oxicuda-memory` | 4,178 | Typed GPU allocations, host-pinned buffers, unified memory, virtual mem |
| `oxicuda-launch` | 4,728 | Kernel launch DSL, `launch!` macro, cooperative/cluster/graph launch |
| `oxicuda-runtime` | 2,518 | High-level CUDA Runtime API layer |

### Vol.2 — PTX Generator & Autotuner

| Crate | SLoC | Role |
|-------|------|------|
| `oxicuda-ptx` | 29,438 | PTX IR, builder DSL, templates, analysis passes, emitter, PTX cache |
| `oxicuda-autotune` | 13,916 | Search-space exploration, Bayesian/genetic/SA strategies, result DB |

### Vol.3 — Linear Algebra (cuBLAS)

| Crate | SLoC | Role |
|-------|------|------|
| `oxicuda-blas` | 21,845 | BLAS L1/L2/L3, GEMM dispatch, Tensor Core paths, batched ops, FP8/INT4 |

### Vol.4 — Deep Learning (cuDNN)

| Crate | SLoC | Role |
|-------|------|------|
| `oxicuda-dnn` | 34,711 | Convolution, FlashAttention-2/3, normalization, pooling, quantization, MoE |

### Vol.5 — Scientific Computing

| Crate | SLoC | Role |
|-------|------|------|
| `oxicuda-fft` | 9,749 | Multi-radix FFT, 1D/2D/3D, batched, pruned FFT, multi-GPU |
| `oxicuda-sparse` | 12,278 | CSR/CSC/COO/BSR/ELL/CSR5/HYB, SpMV/SpMM/SpGEMM/SDDMM, preconditioners |
| `oxicuda-solver` | 15,804 | Dense LU/QR/SVD/Cholesky, iterative CG/BiCGSTAB/GMRES, tensor decomp |
| `oxicuda-rand` | 10,115 | Philox/XORWOW/MRG32k3a/Sobol engines, distributions, Monte Carlo |

### Vol.6 — Signal Processing

| Crate | SLoC | Role |
|-------|------|------|
| `oxicuda-signal` | 6,061 | MFCC/STFT/Mel, Gaussian blur/Sobel/morphology, DCT I-IV, DWT, IIR/FIR |

### Vol.7 — Computation Graph

| Crate | SLoC | Role |
|-------|------|------|
| `oxicuda-graph` | 4,802 | CUDA Graph capture, dependency-sorted execution, event synchronization |

### Vol.8 — GPU Training

| Crate | SLoC | Role |
|-------|------|------|
| `oxicuda-train` | 5,927 | AMP (FP16/BF16 + loss scaling), gradient accumulation, LR schedulers, GPU-fused optimizers |
| `oxicuda-quant` | 4,317 | INT8/INT4/FP8 weight quantization, block-scaled FP4, GPTQ-style PTQ |

### Vol.9 — Inference Engine

| Crate | SLoC | Role |
|-------|------|------|
| `oxicuda-infer` | 4,256 | PagedKvCache, prefix caching, speculative decoding, continuous batching |
| `oxicuda-dist-infer` | 3,279 | Distributed inference (tensor/pipeline parallelism), all-reduce |
| `oxicuda-lm` | 4,394 | BPE tokenizer, vocabulary management, sampling strategies |

### Vol.10 — Reinforcement Learning

| Crate | SLoC | Role |
|-------|------|------|
| `oxicuda-rl` | 4,522 | Replay buffers (Uniform/PER/N-step), policy distributions, GAE/TD-λ/PPO/SAC |

### Backends & Primitives

| Crate | SLoC | Role |
|-------|------|------|
| `oxicuda-primitives` | 4,372 | CUB-equivalent: block reduce/scan/sort, warp ops, histogram |
| `oxicuda-vulkan` | 1,445 | Vulkan Compute / SPIR-V backend |
| `oxicuda-metal` | 1,186 | Apple Metal compute backend (macOS/iOS) |
| `oxicuda-webgpu` | 1,108 | WebGPU / wgpu backend (browser + WASM) |
| `oxicuda-rocm` | 1,087 | AMD ROCm / HIP backend |
| `oxicuda-levelzero` | 1,765 | Intel oneAPI Level Zero backend |

### Umbrella

| Crate | SLoC | Role |
|-------|------|------|
| `oxicuda` | 19,614 | Re-exports, global init, multi-GPU device pool, NCCL-equivalent collectives, OxiONNX/ToRSh/TrustformeRS ecosystem backends |

---

## Driver Layer Architecture

The driver layer (`oxicuda-driver`) is the bedrock of the entire stack. It loads the
CUDA driver library at runtime and wraps every raw FFI call behind safe Rust types.

### Runtime Loading

```rust
// Simplified runtime loading in loader.rs
pub struct CudaLoader {
    lib: libloading::Library,
}

impl CudaLoader {
    pub fn new() -> Result<Self, CudaError> {
        let lib = unsafe {
            libloading::Library::new(platform_lib_name())
        }?;
        Ok(Self { lib })
    }
}
```

The platform name resolution tries the following paths in order:

| Platform | Primary | Fallback |
|----------|---------|----------|
| Linux | `libcuda.so.1` | `libcuda.so` |
| Windows | `nvcuda.dll` | — |
| macOS | returns `UnsupportedPlatform` immediately | — |

On macOS the library still compiles cleanly and all tests run (returning the error
`CudaError::UnsupportedPlatform`), which makes it possible to develop and unit-test
on Apple hardware without any GPU.

### Context and Stream Hierarchy

```
CudaLoader  (singleton per process)
    │
    ├── CudaDevice  (cuDeviceGet, per physical GPU)
    │       │
    │       └── CudaContext  (RAII push/pop, per-thread primary context)
    │               │
    │               ├── CudaStream  (async command queue)
    │               │       └── CudaEvent  (timing fence, inter-stream sync)
    │               │
    │               ├── CudaModule  (loaded PTX/cubin, function extraction)
    │               │       └── CudaFunction  (kernel handle)
    │               │
    │               └── CudaGraph  (Graph API for kernel capture + replay)
```

**Key design invariants:**

- `CudaContext` implements `Drop` → automatically pops the context from the CUDA stack.
- `CudaStream` is `Send + Sync` — safe to pass across threads when properly synchronized.
- `CudaEvent` records a timeline marker and supports `elapsed_time()` for microbenchmarking.
- `DevicePool` (multi_gpu.rs) owns a `Vec<CudaContext>` with round-robin scheduling and
  `best_available_device()` for work-stealing across a multi-GPU node.

### Graph API

The CUDA Graph API is exposed through `CudaGraph` / `CudaGraphExec`. The `StreamCapture`
helper wraps a stream in capture mode, accumulates kernel nodes, and instantiates an
executable graph in a single RAII scope. This allows the driver to replay complex
multi-kernel pipelines with reduced per-launch overhead.

### Occupancy API

`occupancy.rs` wraps `cuOccupancyMaxActiveBlocksPerMultiprocessor` and
`cuOccupancyMaxPotentialBlockSize`, providing high-level helpers used by
`oxicuda-launch` for automatic grid sizing:

```
auto_grid_for(kernel, elements, device) -> Dim3
auto_grid_2d(kernel, rows, cols, device) -> (Dim3, Dim3)
```

---

## PTX IR Design

`oxicuda-ptx` is the code-generation engine of OxiCUDA. It provides a structured IR
that maps directly onto NVIDIA PTX assembly, a builder DSL for constructing kernels in
Rust, a suite of high-level templates, several analysis and optimization passes, and a
text emitter that serializes the IR back to valid PTX source.

### Type System

The `PtxType` enum covers every type in the PTX ISA:

```
Integer:   U8  U16  U32  U64   S8  S16  S32  S64
Float:     F16  F16x2  BF16  BF16x2  F32  F64  TF32
Low-prec:  E4M3  E5M2  E2M3  E3M2  E2M1   (FP8 / FP6 / FP4)
Bit-width: B8  B16  B32  B64  B128         (untyped)
Special:   Pred                             (predicate register)
```

Each variant carries `as_ptx_str()`, `size_bytes()`, `reg_type()`, `is_float()`,
`is_integer()`, and `is_signed()` — all `const fn` for zero-cost use in const contexts.

The `reg_type()` method encodes PTX register promotion rules: sub-32-bit values are
stored in `.b32` registers; 64-bit values in `.b64`; 128-bit in paired `.b64`; predicates
in `.pred`.

### Instruction Set

The `Instruction` enum mirrors the PTX ISA at the instruction granularity. Selected
instruction families:

| Family | Examples |
|--------|---------|
| Arithmetic | `Mad`, `Mul`, `Add`, `Sub`, `Neg`, `Abs`, `Min`, `Max` |
| Floating-point special | `Rcp`, `Rsqrt`, `Sqrt`, `Ex2`, `Lg2`, `Sin`, `Cos` |
| Atomic | `Atom`, `AtomCas`, `Red` with `AtomOp` (Add/Min/Max/Inc/Dec/And/Or/Xor/Exch) |
| Bit manipulation | `Brev`, `Clz`, `Popc`, `Bfind`, `Bfe`, `Bfi` |
| Memory | `Ld`, `St`, `Mov`, `Cvt`, `Tex`, `Suld`, `Sust` |
| Control flow | `Bra`, `Call`, `Ret`, `Setp`, `Selp` |
| Tensor Core | `Wmma` (WMMA), `Mma` (MMA), `Wgmma` (WGMMA — Hopper) |
| Video / DP | `Dp4a`, `Dp2a` |
| Loop | `Pragma` (`.pragma "unroll"`) |

### Builder DSL

`KernelBuilder` and `BodyBuilder` provide a fluent API for constructing PTX kernels
entirely in Rust without manually writing PTX text:

```rust
let mut kb = KernelBuilder::new("vector_add", SmVersion::Sm80);
let (a, b, c, n) = kb.params_4::<*const f32, *const f32, *mut f32, u32>();
kb.body(|bb| {
    let tid = bb.tid_x();
    let stride = bb.ntid_x();
    bb.grid_stride_loop(tid, stride, n, |bb, i| {
        let ai = bb.ld_global_f32(a, i)?;
        let bi = bb.ld_global_f32(b, i)?;
        let ci = bb.add_f32(ai, bi)?;
        bb.st_global_f32(c, i, ci)
    })
})?;
let ptx_text = kb.emit()?;
```

### PTX Templates

Pre-built parameterized templates cover the most common GPU kernel patterns:

| Template | File | Description |
|----------|------|-------------|
| GEMM | `templates/gemm.rs` | Tiled shared-memory GEMM |
| Elementwise | `templates/elementwise.rs` | Unary / binary map kernels |
| Reduction | `templates/reduction.rs` | Parallel tree reduction |
| Softmax | `templates/softmax.rs` | Numerically stable online softmax |
| Scan | `templates/scan.rs` | Blelloch work-efficient prefix scan |
| Transpose | `templates/transpose.rs` | Bank-conflict-free shared-memory transpose |
| Attention | `templates/attention.rs` | FlashAttention-style fused attention |
| BatchNorm | `templates/batch_norm.rs` | Training + inference BN kernels |
| MoE | `templates/moe.rs` | Top-k gating, permute, expert GEMM, unpermute |
| Convolution | `templates/convolution.rs` | im2col, direct conv, backward data/filter |

### Analysis Passes

| Pass | File | What it does |
|------|------|-------------|
| Register pressure | `analysis/register_pressure.rs` | Peak live-register tracking, spill risk, occupancy estimation |
| Dead code elimination | `analysis/dead_code.rs` | Fixed-point DCE with liveness analysis |
| Constant folding | `analysis/constant_folding.rs` | Fold compile-time-constant expressions in the IR |
| Strength reduction | `analysis/strength_reduction.rs` | Replace multiply-by-power-of-two with shift, etc. |
| Arch legality | `analysis/arch_legality.rs` | Reject instructions not supported on target SM |
| Instruction scheduling | `analysis/instruction_scheduling.rs` | Latency-aware reordering |

The PTX validator (`emit/validator.rs`) runs structural and semantic checks before
emitting text, catching type mismatches, undefined registers, and illegal instruction
combinations for the target architecture.

### PTX Cache

`cache.rs` implements a disk-backed cache keyed by SHA-256 hash of the PTX source.
Cached cubin blobs are stored in the OS temp directory under `oxicuda-ptx-cache/`.
This avoids re-compiling identical kernels across process restarts.

---

## Autotuner Architecture

`oxicuda-autotune` implements an offline GPU autotuning framework. It explores
parameterized kernel search spaces to find the best configuration for a given device
and problem size, persists results to a JSON database, and dispatches at runtime through
a 3-tier fallback hierarchy.

### Search Space

A `SearchSpace` is a Cartesian product of discrete parameter axes:

```
tile_m: [64, 128, 256]
tile_n: [64, 128]
tile_k: [16, 32]
vector_width: [1, 2, 4, 8]
unroll_factor: [1, 2, 4]
num_warps: [4, 8, 16]
```

For a GEMM kernel this produces up to 3 × 2 × 2 × 4 × 3 × 3 = 432 candidate
configurations. The `TunableKernel` trait defines how each configuration is compiled
(via PTX template) and benchmarked on GPU.

### Search Strategies

Three interchangeable exploration strategies share a common `Optimizer` trait:

**Bayesian Optimization** (`bayesian.rs`)
- Gaussian Process surrogate with squared-exponential kernel.
- Acquisition functions: Expected Improvement (EI), Upper Confidence Bound (UCB), Probability of Improvement (PI).
- Efficient for small-to-medium search spaces (< 1,000 candidates) with expensive evaluations.

**Simulated Annealing** (`simulated_annealing.rs`)
- Geometric temperature schedule; Metropolis acceptance criterion.
- Handles large discrete search spaces (> 10,000 candidates) where GP fitting is costly.
- Configurable cooling rate, restart count, and initial temperature.

**Genetic Algorithm** (`genetic.rs`)
- Population-based search with tournament selection, uniform crossover, and point mutation.
- Effective at escaping local optima in structured parameter spaces.
- Elitism ensures the best configuration is never lost between generations.

### Early Stopping

`early_stopping.rs` avoids wasting GPU time on clearly inferior configurations:

- **Patience-based**: stop if no improvement after N evaluations.
- **Time-budget**: hard wall-clock deadline.
- **Convergence detection**: stop when variance of top-k scores falls below a threshold.

### Runtime Dispatcher

The 3-tier dispatcher (`dispatch.rs`) resolves the best kernel configuration at
inference time without triggering a new autotune run:

```
Tier 1 — Exact cache hit:   db[device_uuid][problem_shape] → cached config
Tier 2 — Interpolation:     inverse-distance-weighted from nearby problem sizes
Tier 3 — Default fallback:  compile-time heuristic (e.g., 128×128×32 tile, 8 warps)
```

`interpolation.rs` implements both nearest-neighbor and inverse-distance-weighted
strategies for Tier 2, enabling good performance on problem sizes not in the database.

### Result Database

Tuning results are persisted as JSON files under `$HOME/.cache/oxicuda/autotune/`.
Each entry records the device UUID, SM version, problem parameters, best config,
achieved throughput (TFLOP/s or GB/s), and the timestamp of the measurement.

---

## BLAS Architecture

`oxicuda-blas` provides a full BLAS-equivalent library with precision-aware dispatch,
Tensor Core acceleration, and epilogue fusion.

### Dispatch Hierarchy

```
gemm(handle, A, B, C, alpha, beta)
    │
    ├── precision = F16 / BF16 / TF32?
    │       └── arch >= SM 70?
    │               └── TensorCoreGemm  (WMMA / MMA / WGMMA)
    │
    ├── precision = FP8?
    │       └── arch >= SM 90 (Hopper)?
    │               └── Fp8Gemm  (E4M3 / E5M2 operands)
    │
    ├── shape is tall-skinny (K >> M, N)?
    │       └── SplitKGemm  (split-K across CTAs)
    │
    └── SimtGemm  (fallback: CUDA Core tiled GEMM)
```

### BLAS Level Summary

**Level 1 — Vector-vector** (`level1/`): axpy, scal, dot, nrm2, asum, iamax, copy, swap.

**Level 2 — Matrix-vector** (`level2/`): gemv, symv, trmv, trsv, ger, syr.

**Level 3 — Matrix-matrix** (`level3/`): gemm (with full dispatch hierarchy), symm, trsm,
syrk/syr2k (with Tensor Core triangle-masked paths), trmm.

### Tensor Core Paths

| Precision | Min SM | API | Tile |
|-----------|--------|-----|------|
| F16/F32 acc | SM 7.0 | WMMA | 16×16×16 |
| F16/BF16/TF32 | SM 8.0 | MMA | 16×8×16 |
| FP8 E4M3/E5M2 | SM 9.0 | WGMMA | 64×8×16 |
| FP4/FP6 | SM 10.0 | WGMMA | micro-scaled |

The warp-specialized GEMM (`gemm/warp_specialized.rs`) divides warps within a CTA into
producers (fetching tiles from global memory via `ld.global.nc`) and consumers
(performing `mma` instructions), overlapping memory latency with computation.

### Advanced GEMM Variants

| Variant | File | Description |
|---------|------|-------------|
| Stream-K | `stream_k.rs` | Dynamic work partitioning, eliminates load imbalance on last wave |
| Persistent | `persistent_gemm.rs` | Work-stealing via atomic counter; SM stays occupied for multiple tiles |
| Split-K | `level3/gemm/splitk.rs` | Splits K-dimension across CTAs, accumulates with atomic add |
| Batched | `batched/batched_gemm.rs` | Independent GEMM batch execution |
| Strided batched | `batched/strided_gemm.rs` | Fixed stride between batch elements |
| Grouped | `batched/grouped_gemm.rs` | Variable-size GEMM groups (MoE expert dispatch) |
| Complex | `complex_gemm.rs` | CGEMM / ZGEMM with interleaved real/imag storage |
| Warp-specialized | `gemm/warp_specialized.rs` | Producer/consumer warp roles with ping-pong pipelining |

### Epilogue Fusion

The epilogue module (`level3/gemm/epilogue.rs`) fuses post-GEMM operations into the
store phase without a separate kernel:

```
D = alpha * A @ B + beta * C + bias + activation(...)
```

Supported activations in the fused epilogue: ReLU, GELU, SiLU, tanh, sigmoid.

---

## DNN Architecture

`oxicuda-dnn` (89 files, 31,293 SLoC) is the deepest crate in the stack, covering
convolution, attention, normalization, pooling, quantization, MoE, and RNN.

### FlashAttention-3 on Hopper

`attn/flash_attn/hopper.rs` implements the FlashAttention-3 algorithm with
Hopper-specific optimizations:

- **TMA (Tensor Memory Accelerator)**: uses `cp.async.bulk` PTX instructions to issue
  asynchronous bulk copies from global to shared memory, decoupling memory access from
  computation.
- **WGMMA**: Hopper's warpgroup-level MMA instruction operates on entire 64×8×16 tiles
  in a single instruction, replacing the older per-warp WMMA.
- **Ping-pong pipeline**: two shared-memory tile buffers alternate between producer
  (TMA load) and consumer (WGMMA compute) phases, hiding memory latency.
- **Warp specialization**: separate warp roles for Q-load, KV-load, and softmax /
  attention score computation.
- Both forward and backward passes are implemented.

### Convolution Stack

```
ConvolutionDescriptor + FilterDescriptor
    │
    ├── algo_select.rs  (heuristic + autotuned algorithm selection)
    │
    └── Forward algorithms:
          implicit_gemm.rs  (NHWC layout, direct GEMM over unrolled input patches)
          im2col_gemm.rs    (classical im2col + cuBLAS GEMM)
          winograd.rs       (Winograd F(2×2, 3×3) and F(4×4, 3×3) fast convolution)
          direct.rs         (direct 1×1 and depthwise separable)
          fft_conv.rs       (frequency-domain for large kernels ≥ 7×7)

    └── Backward algorithms:
          dgrad/implicit_gemm.rs
          dgrad/winograd.rs
          wgrad/implicit_gemm.rs
          wgrad/winograd.rs

    └── 3D convolution: conv/conv3d/ (im2col3d + GEMM, forward/backward/wgrad)
    └── Transposed convolution: transpose_conv.rs (col2im PTX, weight reshape)
    └── Deformable convolution: deformable.rs (DCNv2, bilinear interpolation)
```

### Mixture of Experts (MoE)

The MoE module implements the full routing and computation pipeline for sparse MoE
transformer layers:

```
input tokens
    │
    ├── routing.rs   — softmax gating, top-k expert selection per token
    ├── permute.rs   — scatter tokens into expert-contiguous layout
    ├── capacity.rs  — enforce expert capacity factor (drop overflow tokens)
    ├── fused_moe.rs — fused single-pass MoE kernel (route + expert GEMM)
    ├── grouped_gemm.rs — variable-size expert GEMM batches
    ├── aux_loss.rs  — Switch Transformer load-balance loss + z-loss
    └── monitoring.rs — runtime utilization tracking across experts
```

`permute.rs` outputs tokens in expert order (scatter) which is then undone after
expert computation (gather), all using PTX atomics for conflict-free address resolution.

### Normalization Suite

| Norm | File | Notes |
|------|------|-------|
| BatchNorm | `norm/batch_norm.rs` | Training (running stats) + inference (frozen stats) |
| LayerNorm | `norm/layer_norm.rs` | Per-sample normalization, standard in transformers |
| RMSNorm | `norm/rms_norm.rs` | LLaMA / Mistral; no mean subtraction |
| GroupNorm | `norm/group_norm.rs` | Per-group within channel dimension |
| InstanceNorm | `norm/instance_norm.rs` | Per (batch, channel) normalization |
| ScaleNorm | `norm/scale_norm.rs` | Simplified L2 normalization |
| PowerNorm | `norm/power_norm.rs` | Running power-mean normalization |
| Fused norm | `norm/fused_norm.rs` | Norm + activation in a single kernel pass |

### Quantization

| Scheme | File | Target |
|--------|------|--------|
| FP8 E4M3 / E5M2 | `quantize/fp8_quantize.rs` | Hopper (SM 9.0+) training/inference |
| INT8 symmetric | `quantize/int8_quantize.rs` | Post-training quantization |
| INT4 / NF4 | `quantize/int4_quantize.rs` | QLoRA fine-tuning (group scaling) |
| Block-scaled FP4 | `quantize/block_scale.rs` | Blackwell micro-scaled sub-byte |
| GPTQ / AWQ | `quantize/gptq_awq.rs` | Activation-aware weight quantization |
| QAT (fake quant) | `quantize/qat.rs` | Straight-through estimator for training |

---

## FFT Architecture

`oxicuda-fft` (35 files, 8,853 SLoC) implements a full GPU FFT library supporting 1D,
2D, 3D, batched, real, and arbitrary-size transforms.

### Radix Strategies

The `FftPlan` selects the decomposition strategy based on the input size:

```
size is power of 2, fits in shared memory (≤ 4096)?
    └── Stockham auto-sort (bank-conflict-free padding, in-place)

size is composite (2, 3, 5, 7 factors)?
    └── Mixed-radix (combine radix-2/4/8 stages)

size is composite with coprime factors?
    └── Prime-Factor Algorithm (Good-Thomas CRT mapping)

size is power of 2, needs split across phases?
    └── Split-radix (radix-2/4 hybrid, ~10% fewer operations than pure radix-2)

size is arbitrary (prime)?
    └── Bluestein / Chirp-Z (embed in next power of 2, O(N log N))
```

Radix-2, radix-4, and radix-8 butterfly kernels are in `radix/radix{2,4,8}.rs`.
Each is hand-optimized for coalesced global loads and bank-conflict-free shared memory.

### Transform Types

| Transform | File | Description |
|-----------|------|-------------|
| C2C | `transforms/c2c.rs` | Complex-to-complex (forward + inverse) |
| R2C | `transforms/r2c.rs` | Real-to-complex (exploits Hermitian symmetry) |
| C2R | `transforms/c2r.rs` | Complex-to-real (conjugate-symmetric input) |
| 2D FFT | `transforms/fft2d.rs` | Row-major 2D via row + column passes + transpose |
| 3D FFT | `transforms/fft3d.rs` | Three-pass slab decomposition |

The real FFT path (`real_fft.rs`) packs two real transforms into one complex FFT using
the conjugate-symmetry trick, halving the compute cost for real-valued data.

### Batched FFT Fusion

`fused_batch.rs` maps multiple small FFTs (N ≤ 1024) onto a single thread block,
keeping all data in shared memory. This avoids global memory round-trips for small
transform sizes and achieves memory-bound performance for batch sizes in the thousands.

### Multi-GPU FFT

`multi_gpu.rs` implements 1D slab decomposition across P GPU devices:

```
Input array (length N) split into P slabs of N/P elements
    │
    ├── Local FFT on each slab (intra-device)
    ├── All-to-all peer copy (transposes frequency domain)
    └── Local FFT on transposed slabs (inter-device phase)
```

The peer copies use `cuMemcpyPeerAsync` with stream ordering to overlap compute and
communication where device topology permits.

---

## Sparse Architecture

`oxicuda-sparse` (36 files, 11,021 SLoC) provides GPU-accelerated sparse linear
algebra across multiple storage formats and operation types.

### Format Selection Heuristics

The auto-format selector (`ops/auto_spmv.rs`) chooses the storage format based on
matrix properties at runtime:

```
nnz / rows < 4 (very sparse rows)?
    └── COO (simpler indexing, coalesced segmented scan)

nnz / rows is uniform and small (≤ ELL_MAX_NNZ_PER_ROW)?
    └── ELL (coalesced column-major access, no row pointer overhead)

block structure detected (block_size ∈ {2, 4, 8, 16})?
    └── BSR (dense sub-block multiply via register tiles)

density variance high (irregular sparsity)?
    └── CSR5 (tile-based, load-balanced, two-phase execution)

default fallback?
    └── CSR (universal, thread-per-row or vector-per-row kernel)
```

### SpMV Dispatch

| Kernel | Format | Strategy |
|--------|--------|----------|
| `spmv_csr.rs` | CSR | Thread-per-row or warp-per-row depending on avg nnz/row |
| `spmv_ell.rs` | ELL | Coalesced column-major access with sentinel-based padding |
| `spmv_bsr.rs` | BSR | Block-aware: one CTA per block-row, dense sub-block multiply |
| `spmv_csr5.rs` | CSR5 | Tile-phase + calibration-phase, handles irregular rows |

### Higher-Level Operations

| Operation | File | Algorithm |
|-----------|------|-----------|
| SpMM | `ops/spmm.rs` | Sparse-matrix × dense-matrix (CSR row-panel tiling) |
| SpGEMM | `ops/spgemm.rs` | Sparse × sparse: symbolic + numeric phases |
| SpGEMM merge | `ops/spgemm_merge.rs` | Merge-path load balancing |
| SDDMM | `ops/sddmm.rs` | Sampled dense-dense matrix multiply |
| SpTRSV | `ops/sptrsv.rs` | Level-scheduled sparse triangular solve |
| Krylov | `ops/krylov.rs` | Lanczos and Arnoldi iteration |

### Preconditioners

ILU(0) and IC(0) incomplete factorizations are in `preconditioner/ilu0.rs` and
`preconditioner/ic0.rs`. Graph coloring (`graph_coloring.rs`) assigns wavefronts for
parallel triangular solve, enabling concurrent ILU applications across independent
color classes. `preconditioner/iluk.rs` extends this to multi-level ILU(k) with
configurable fill levels.

---

## Solver Architecture

`oxicuda-solver` (40 files, 13,981 SLoC) provides a cuSOLVER-equivalent library
spanning dense factorizations, iterative Krylov solvers, and sparse direct methods.

### Dense Solver Stack

```
oxicuda-solver dense/
├── lu.rs          — LU factorization with partial pivoting (Bunch-Kaufman variant)
├── qr.rs          — Householder QR with column pivoting
├── svd.rs         — Full / economy SVD (bidiagonalization + implicit QR shift)
├── dc_svd.rs      — Divide-and-conquer SVD (recursive bidiagonal splitting)
├── cholesky.rs    — Cholesky factorization (positive-definite systems)
├── ldlt.rs        — Symmetric indefinite LDL^T (Bunch-Kaufman)
├── eig.rs         — Eigenvalue decomposition (symmetric + general)
├── inverse.rs     — Matrix inverse via LU
├── det.rs         — Determinant via LU log-product
├── lstsq.rs       — Least squares (QR-based, rank-revealing)
├── band.rs        — Banded LU, Cholesky, solve
├── matrix_functions.rs — expm, logm, sqrtm via Padé approximation
└── tridiagonal.rs — Thomas algorithm, cyclic reduction, batched tridiagonal
```

### Iterative Solvers

| Solver | File | Use case |
|--------|------|----------|
| CG | `sparse/cg.rs` | Symmetric positive-definite systems |
| PCG | `preconditioned.rs` | CG with Jacobi or ILU preconditioner |
| BiCGSTAB | `sparse/bicgstab.rs` | Non-symmetric systems |
| GMRES | `sparse/gmres.rs` | General non-symmetric, memory-bounded restart |
| FGMRES | `sparse/fgmres.rs` | Flexible GMRES (variable preconditioner per iteration) |
| PGMRES | `preconditioned.rs` | Preconditioned GMRES |
| Direct | `sparse/direct.rs` | Sparse direct solver (fill-reducing ordering + factorization) |

### Batched Solvers

`batched.rs` handles many small matrices (4×4 to 64×64) in a single kernel launch:

- Each CUDA thread block processes one small matrix using register-resident storage.
- Supports batched LU, QR, and Cholesky with back-substitution.
- Target use case: per-pixel / per-cell solves in scientific simulation codes.

### Randomized SVD

`randomized_svd.rs` implements the Halko-Martinsson-Tropp 2011 algorithm:

```
1. Draw Gaussian sketch matrix Ω ∈ R^{n×(k+p)}
2. Form Y = A @ Ω  (matrix-matrix multiply on GPU)
3. Power iteration: Y = (A @ A^T)^q @ Y
4. QR decomposition of Y → orthonormal basis Q
5. B = Q^T @ A  (project A to low-rank basis)
6. SVD of small matrix B (on device)
7. Recover left singular vectors: U = Q @ U_B
```

This achieves O(mn(k+p)) instead of O(mn·min(m,n)) for rank-k approximations.

---

## GPU Architecture Coverage

OxiCUDA targets NVIDIA GPUs from Turing (SM 7.5) through Blackwell (SM 10.0).

| SM Version | Microarchitecture | GPU Examples | Tensor Core | FP8 | FP4/FP6 | TMA |
|------------|-------------------|--------------|-------------|-----|----------|-----|
| SM 7.5 | Turing | RTX 2080, T4 | 1st gen (WMMA F16) | — | — | — |
| SM 8.0 | Ampere | A100, A30 | 3rd gen (MMA TF32/BF16) | — | — | — |
| SM 8.6 | Ampere | RTX 3090, A10 | 3rd gen | — | — | — |
| SM 8.9 | Ada Lovelace | RTX 4090, L4 | 4th gen | — | — | — |
| SM 9.0 | Hopper | H100, H200 | 4th gen (WGMMA) | E4M3/E5M2 | — | Yes |
| SM 10.0 | Blackwell | B100, B200 | 5th gen | E4M3/E5M2 | E2M1/E2M3/E3M2 | Yes |

The architecture rules (`oxicuda-ptx/src/arch.rs`) encode per-SM feature flags and
validate at IR construction time that instructions are legal for the target architecture.
For example, WGMMA instructions are rejected on SM < 9.0, and FP4 operands require SM 10.0.

---

## Pure Rust Design Decisions

### No Compile-Time CUDA SDK

The traditional CUDA workflow requires installing the NVIDIA CUDA Toolkit (nvcc, cuda.h,
libcudart.a) at compile time. OxiCUDA eliminates this dependency entirely:

- The CUDA driver API (`libcuda.so` / `nvcuda.dll`) is loaded at **runtime** via
  `libloading`. No `cuda.h` is included; all function signatures are declared directly
  in `oxicuda-driver/src/ffi.rs` as `unsafe extern "C"` function pointer types.
- PTX is assembled to cubin by the driver itself (`cuModuleLoadData`), so `ptxas` is
  also not needed at compile time.
- The `oxicuda-metal` crate links against Apple's Metal framework only on macOS (via the
  `metal` crate), and only when the `metal` feature is enabled.
- The `oxicuda-vulkan` crate uses the `ash` runtime loader — no compile-time `libvulkan`
  dependency; the Vulkan library is discovered at runtime via `vkGetInstanceProcAddr`.

### COOLJAPAN Policy Compliance

| Policy | Implementation |
|--------|---------------|
| No openblas | All linear algebra goes through PTX-generated GPU kernels or SciRS2 |
| No bincode | Autotune DB and PTX cache use `serde_json` |
| No rustfft | FFT uses OxiFFT where needed; GPU path uses PTX-generated Stockham kernels |
| No flate2/zstd | Compression (if needed) uses `oxiarc-*` crates |
| No zip | Archive operations use `oxiarc-archive` |

### Platform Behavior

| Platform | GPU available | Behavior |
|----------|--------------|----------|
| Linux + NVIDIA GPU + driver ≥ 525 | Yes | Full GPU operation |
| Windows + NVIDIA GPU + driver ≥ 525 | Yes | Full GPU operation |
| Linux / Windows, no GPU | No | `CudaError::NoDevicesFound` at runtime |
| macOS (any) | No (MPS not used) | `CudaError::UnsupportedPlatform` at runtime |
| Any platform (Metal feature) | macOS only | `oxicuda-metal` uses Apple Metal directly |
| Any platform (WebGPU feature) | Any | `oxicuda-webgpu` uses `wgpu` cross-platform |

---

## Testing Strategy

### CPU-testable Tests (No GPU Required)

The vast majority of OxiCUDA's 5,139 tests run without any GPU hardware:

- **PTX IR tests**: construction, validation, emission, analysis passes — all exercise
  the pure-Rust PTX IR with no GPU calls.
- **Builder DSL tests**: verify that the fluent API produces syntactically valid PTX text.
- **Template tests**: instantiate templates with various parameters and validate the
  generated PTX against expected instruction patterns.
- **Type system tests**: `PtxType` methods, register class promotion, size computations.
- **Autotuner unit tests**: search space enumeration, interpolation, early stopping logic,
  database serialization round-trips — all run against mock `TunableKernel` implementations.
- **Format tests** (`oxicuda-sparse`): matrix construction, format conversion (CSR↔CSC,
  COO↔CSR), validity checks — all in pure Rust.
- **BLAS dispatch logic tests**: precision × architecture decision tree, epilogue fusion
  parameter validation.
- **Error handling tests**: every `CudaError` variant, proper `Drop` on resource handles
  (verified to not call real GPU APIs via the mock driver).
- **macOS CI**: the full test suite runs on Ubuntu GitHub Actions workers; macOS tests
  additionally exercise the `UnsupportedPlatform` error paths.

### GPU-Gated Tests (`#[cfg(feature = "gpu-tests")]`)

Tests that require a real GPU are isolated behind the `gpu-tests` Cargo feature:

```toml
# Cargo.toml (per-crate)
[features]
gpu-tests = []
```

These tests are tagged `#[cfg(feature = "gpu-tests")]` and cover:

- End-to-end kernel launch via `cuLaunchKernel` (launch overhead, stream ordering).
- Memory bandwidth: H2D / D2H with PinnedBuffer, D2D peer copy.
- GEMM correctness: FP16/BF16/TF32/FP32/FP64 against CPU reference.
- FFT round-trip accuracy: complex→FFT→IFFT→complex (tolerance < N × eps).
- SpMV correctness: CSR against SciRS2 CPU sparse reference.
- RNG distribution tests: chi-squared goodness-of-fit for uniform and normal samples.
- FlashAttention numerical accuracy vs. naive attention at FP16 and FP32.

### CI Pipeline Overview

```
GitHub Actions (ubuntu-latest, every push to main / 0.*.*)
    │
    ├── cargo fmt --all -- --check          (formatting gate)
    ├── cargo build --workspace --all-features
    ├── cargo clippy -- -D warnings         (zero-warning policy)
    ├── cargo nextest run --workspace --all-features   (5,139 CPU tests)
    └── cargo doc --workspace --no-deps     (doc completeness)
```

The `gpu-tests` feature is intentionally excluded from the public CI job because GitHub
Actions does not provide GPU runners. GPU tests are expected to be run locally on
developer workstations with NVIDIA GPUs (driver ≥ 525) before merging feature branches.

Performance regression detection (≥ 5% threshold) is planned for v1.0 via criterion
benchmark history tracked in the `benches/` directory.
