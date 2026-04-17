# oxicuda-rocm TODO

AMD ROCm compute backend with HIP kernel string generators and host-side dispatch,
providing GPU-accelerated operations on AMD GPUs. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Tests**: 104 passing
- **Status**: HIP kernel generators + host-side dispatch

### Completed

#### Core Infrastructure
- [x] HIP runtime abstraction
- [x] Memory allocation and buffer management
- [x] HIP kernel string generation framework
- [x] ComputeBackend trait implementation

#### Compute Operations -- HIP Kernel Generators
- [x] GEMM -- HIP tiled matrix multiply kernel generator
- [x] Unary elementwise -- HIP kernel generator for relu, sigmoid, tanh, exp, log, sqrt, abs, neg, gelu, silu
- [x] Binary elementwise -- HIP kernel generator for add, sub, mul, div, max, min, pow
- [x] Reduction -- HIP kernel generator for sum, max, min, mean
- [x] Attention -- HIP fused attention kernel generator
- [x] Conv2D -- HIP convolution kernel generator

### Future Enhancements

- [x] Actual GPU dispatch via hiprtc when HIP runtime available
- [x] ROCm-specific optimizations (wave64 vs wave32)
- [x] Batched GEMM support (`batched_gemm_hip` kernel + `batched_gemm()` trait override)
- [x] FP16/BF16 support via AMD matrix cores (MFMA) (`gemm_hip_f16`, `gemm_hip_bf16`, `elementwise_hip_f16` kernels)
- [x] hipBLAS interop for performance-critical paths
- [x] Multi-GPU HIP support

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-backend | Common backend traits | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0
- Tests: 104 passing
- unwrap() calls: 0
- Clippy: clean
