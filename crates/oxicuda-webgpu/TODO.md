# oxicuda-webgpu TODO

Cross-platform WebGPU compute backend via wgpu, providing GPU-accelerated operations
through WGSL shader dispatch. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Tests**: 91 passing
- **Status**: Memory + Compute operations fully wired

### Completed

#### Core Infrastructure
- [x] WebGPU device/queue management via wgpu
- [x] Memory allocation and buffer management
- [x] ComputeBackend trait implementation

#### Compute Operations -- WGSL Shader Dispatch
- [x] GEMM -- tiled matrix multiply WGSL compute shader
- [x] Unary elementwise -- relu, sigmoid, tanh, exp, log, sqrt, abs, neg, gelu, silu
- [x] Binary elementwise -- add, sub, mul, div, max, min, pow (with `binary_wgsl` generator)
- [x] Reduction -- sum, max, min, mean (with `reduction_wgsl` + `reduction_final_wgsl` two-pass pipeline)
- [x] Conv2D forward -- WGSL NCHW compute shader + CPU fallback
- [x] Attention -- WGSL scaled dot-product + stable softmax + causal masking

#### WGSL Shader Generators
- [x] `gemm_wgsl` -- tiled GEMM with workgroup shared memory
- [x] `elementwise_wgsl` -- unary elementwise operations
- [x] `binary_wgsl` -- binary elementwise operations
- [x] `reduction_wgsl` -- block-level partial reduction
- [x] `reduction_final_wgsl` -- final scalar reduction from partial results

### Future Enhancements

- [x] Batched GEMM support (`batched_gemm_wgsl` shader + `batched_gemm()` trait override)
- [x] FP16 support via wgpu f16 extension (`gemm_wgsl_f16` shader + `gemm_f16()` method)
- [x] WebAssembly target support for browser-based GPU compute (`WasmBackend`, `WasmMemoryManager`, `wasm` feature flag)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| wgpu | WebGPU implementation | Yes |
| oxicuda-backend | Common backend traits | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0
- Tests: 91 passing
- unwrap() calls: 0
- Clippy: clean
