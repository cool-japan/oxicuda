# oxicuda-metal TODO

Apple Metal compute backend via metal-rs, providing GPU-accelerated operations
on macOS through MSL shader dispatch. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Tests**: 121 passing
- **Status**: Memory + Compute operations for macOS

### Completed

#### Core Infrastructure
- [x] Metal device/command queue management via metal-rs
- [x] Memory allocation and buffer management
- [x] ComputeBackend trait implementation
- [x] macOS-specific Metal API integration

#### Compute Operations -- MSL Shader Dispatch
- [x] GEMM -- tiled matrix multiply MSL compute shader
- [x] Unary elementwise -- relu, sigmoid, tanh, exp, log, sqrt, abs, neg, gelu, silu
- [x] Binary elementwise -- add, sub, mul, div, max, min, pow (with `binary_msl` generator)
- [x] Reduction -- sum, max, min, mean (with dedicated MSL max/min/mean shaders)
- [x] Conv2D forward -- MSL NCHW compute shader + CPU fallback
- [x] Attention -- MSL scaled dot-product + stable softmax + causal masking

#### FFT Pipeline State Caching
- [x] `MetalFftPlan` now compiles MSL shaders **once** at `::new()` time
- [x] `ComputePipelineState` objects for butterfly + bit-reversal cached in struct fields
- [x] `CommandQueue` cached — created once per plan, reused across all `execute()` calls
- [x] `execute()` reuses pre-compiled pipelines; eliminates 100 ms+ per-call recompilation
- [x] All new struct fields properly `#[cfg(target_os = "macos")]` gated
- [x] Manual `Debug` impl using `finish_non_exhaustive()` (ComputePipelineState has no Debug)

#### MSL Shader Generators
- [x] `gemm_msl` -- tiled GEMM with threadgroup shared memory
- [x] `elementwise_msl` -- unary elementwise operations
- [x] `binary_msl` -- binary elementwise operations
- [x] `reduction_msl` -- reduction with sum/max/min/mean MSL shaders

### Future Enhancements

- [x] Batched GEMM support (`batched_gemm_msl` shader + `batched_gemm()` trait override)
- [x] FP16 support via Metal half-precision (`gemm_msl_f16` shader + `gemm_f16()` method)
- [ ] Apple Silicon neural engine integration
- [ ] Metal Performance Shaders (MPS) interop

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| metal | Metal API bindings | Yes (Obj-C FFI) |
| oxicuda-backend | Common backend traits | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0
- Tests: 121 passing
- unwrap() calls: 0
- Clippy: clean
