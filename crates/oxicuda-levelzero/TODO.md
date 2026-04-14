# oxicuda-levelzero TODO

Intel Level Zero / oneAPI compute backend with OpenCL SPIR-V kernel generators
and dispatch framework. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Tests**: 82 passing
- **Status**: OpenCL SPIR-V kernel generators + dispatch framework

### Completed

#### Core Infrastructure
- [x] Level Zero runtime abstraction
- [x] Memory allocation and buffer management
- [x] OpenCL SPIR-V kernel dispatch framework
- [x] ComputeBackend trait implementation

#### Compute Operations -- OpenCL SPIR-V Generators
- [x] GEMM -- OpenCL SPIR-V tiled matrix multiply kernel generator
- [x] Unary elementwise -- OpenCL SPIR-V generator for relu, sigmoid, tanh, exp, log, sqrt, abs, neg, gelu, silu
- [x] Binary elementwise -- OpenCL SPIR-V generator for add, sub, mul, div, max, min, pow
- [x] Reduction -- OpenCL SPIR-V generator for sum, max, min, mean
- [x] Conv2D forward -- OpenCL SPIR-V NCHW compute shader + CPU fallback
- [x] Attention -- OpenCL SPIR-V scaled dot-product + stable softmax + causal masking

### Future Enhancements

- [x] Batched GEMM support (`batched_gemm_compute_shader` SPIR-V + `batched_gemm()` trait override)
- [ ] Intel Xe matrix extensions (XMX) for Tensor Core equivalent
- [x] Intel GPU sub-group optimizations (`reduction_subgroup_spirv`, `scan_subgroup_spirv`, `gemm_subgroup_spirv`)
- [ ] Multi-tile/multi-device support

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-backend | Common backend traits | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0
- Tests: 82 passing
- unwrap() calls: 0
- Clippy: clean
