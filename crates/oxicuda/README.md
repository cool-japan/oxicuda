# oxicuda

Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

**Version:** 0.1.4 — 2026-04-18 — 496 tests passing

`oxicuda` is the umbrella crate that re-exports all OxiCUDA sub-crates behind
feature flags. It provides a single dependency entry point for applications that
need GPU compute capabilities without installing the CUDA Toolkit -- `libcuda.so`
(or `nvcuda.dll`) is loaded dynamically at runtime.

The core crates (driver, memory, launch) are enabled by default. Higher-level
libraries -- BLAS, DNN, FFT, sparse, solver, and random number generation -- are
opt-in via feature flags. Enable `full` to get everything.

A `prelude` module and `init()` function provide convenient imports and
one-call CUDA driver initialization. Additional built-in modules cover
profiling, multi-GPU device pools, collective communication (NCCL equivalent),
pipeline parallelism, and multi-node distributed training.

## Architecture

```text
                    oxicuda (umbrella)
     +---------+---------+---------+---------+
     |         |         |         |         |
  driver   memory    launch      ptx    autotune
     |         |         |         |         |
     +----+----+---------+---------+---------+
          |
   +------+------+------+------+------+
   |      |      |      |      |      |
  blas   dnn    fft   sparse solver  rand
```

## Quick Start

```rust,no_run
use oxicuda::prelude::*;

fn main() -> CudaResult<()> {
    oxicuda::init()?;

    let device = Device::get(0)?;
    let ctx = std::sync::Arc::new(Context::new(&device)?);
    let stream = Stream::new(&ctx)?;

    let mut buf = DeviceBuffer::<f32>::alloc(1024)?;
    let host = vec![1.0f32; 1024];
    buf.copy_from_host(&host)?;

    Ok(())
}
```

## Feature Flags

| Feature                | Description                                      | Default |
|------------------------|--------------------------------------------------|---------|
| `driver`               | CUDA driver API wrapper                          | Yes     |
| `memory`               | GPU memory management                            | Yes     |
| `launch`               | Kernel launch infrastructure                     | Yes     |
| `ptx`                  | PTX code generation DSL                          | No      |
| `autotune`             | Autotuner engine (implies `ptx`)                 | No      |
| `blas`                 | cuBLAS equivalent                                | No      |
| `dnn`                  | cuDNN equivalent (implies `blas`)                | No      |
| `fft`                  | cuFFT equivalent                                 | No      |
| `sparse`               | cuSPARSE equivalent                              | No      |
| `solver`               | cuSOLVER equivalent                              | No      |
| `rand`                 | cuRAND equivalent                                | No      |
| `pool`                 | Stream-ordered memory pool                       | No      |
| `backend`              | Abstract compute backend trait                   | No      |
| `primitives`           | CUB-equivalent parallel GPU primitives           | No      |
| `vulkan`               | Vulkan compute backend (cross-vendor)            | No      |
| `metal`                | Apple Metal compute backend (macOS/iOS)          | No      |
| `webgpu`               | WebGPU compute backend (via wgpu)                | No      |
| `rocm`                 | AMD ROCm/HIP backend (Linux + AMD GPU)           | No      |
| `level-zero`           | Intel Level Zero backend                         | No      |
| `onnx-backend`         | ONNX operator runtime and graph executor         | No      |
| `tensor-backend`       | ToRSh GPU tensor backend with autograd           | No      |
| `transformer-backend`  | TrustformeRS transformer inference backend       | No      |
| `wasm-backend`         | WASM + WebGPU backend for browser environments   | No      |
| `full`                 | Enable all optional features                     | No      |

## Sub-crates

| Crate                | Volume | Description                                        |
|----------------------|--------|-----------------------------------------------------|
| `oxicuda-driver`     | Vol.1  | CUDA driver API bindings                           |
| `oxicuda-memory`     | Vol.1  | Device, pinned, unified memory                     |
| `oxicuda-launch`     | Vol.1  | Kernel launch and grid configuration               |
| `oxicuda-ptx`        | Vol.2  | PTX code generation DSL                            |
| `oxicuda-autotune`   | Vol.2  | Autotuner for kernel parameters                    |
| `oxicuda-blas`       | Vol.3  | Dense linear algebra (GEMM, etc.)                  |
| `oxicuda-dnn`        | Vol.4  | Deep learning primitives                           |
| `oxicuda-fft`        | Vol.5  | Fast Fourier Transform                             |
| `oxicuda-sparse`     | Vol.5  | Sparse matrix operations                           |
| `oxicuda-solver`     | Vol.5  | Matrix decompositions and solvers                  |
| `oxicuda-rand`       | Vol.5  | Random number generation                           |
| `oxicuda-primitives` | Vol.5  | CUB-equivalent warp/block/device primitives        |
| `oxicuda-backend`    | —      | Abstract compute backend trait                     |
| `oxicuda-vulkan`     | —      | Vulkan compute backend                             |
| `oxicuda-metal`      | —      | Apple Metal compute backend                        |
| `oxicuda-webgpu`     | —      | WebGPU compute backend                             |
| `oxicuda-rocm`       | —      | AMD ROCm/HIP backend                               |
| `oxicuda-levelzero`  | —      | Intel Level Zero backend                           |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
