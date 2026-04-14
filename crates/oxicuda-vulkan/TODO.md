# oxicuda-vulkan TODO

Vulkan compute backend via ash, providing GPU-accelerated operations through
SPIR-V compute shader dispatch. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Tests**: 61 passing
- **Status**: Memory + Compute operations via SPIR-V

### Completed

#### Core Infrastructure
- [x] Vulkan instance/device/queue management via ash
- [x] Memory allocation and buffer management
- [x] Compute pipeline creation and dispatch
- [x] ComputeBackend trait implementation

#### Compute Operations -- SPIR-V Compute Shader Dispatch
- [x] GEMM -- tiled matrix multiply SPIR-V compute shader
- [x] Unary elementwise -- SPIR-V generator for relu, sigmoid, tanh, exp, log, sqrt, abs, neg, gelu, silu
- [x] Binary elementwise -- SPIR-V generator for add, sub, mul, div, max, min, pow
- [x] Reduction -- SPIR-V generator for sum, max, min, mean
- [x] Conv2D forward -- SPIR-V NCHW compute shader + CPU fallback
- [x] Attention -- SPIR-V scaled dot-product + stable softmax + causal masking

#### SPIR-V Generators
- [x] `elementwise_spirv` -- unary elementwise SPIR-V module generation
- [x] `binary_spirv` -- binary elementwise SPIR-V module generation
- [x] `reduction_spirv` -- reduction SPIR-V module generation
- [x] `gemm_spirv` -- tiled GEMM SPIR-V module generation

### Future Enhancements

- [ ] Batched GEMM support
- [ ] Subgroup (warp-level) optimizations
- [ ] Vulkan descriptor set caching / pipeline caching
- [ ] Multi-queue async compute

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| ash | Vulkan API bindings | Yes |
| oxicuda-backend | Common backend traits | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0
- Tests: 61 passing
- unwrap() calls: 0
- Clippy: clean
