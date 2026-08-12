# oxicuda-metal

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-metal` provides a `MetalBackend` that implements the `ComputeBackend` trait from `oxicuda-backend` using Apple's Metal API. It targets Apple Silicon and Intel Mac GPUs through Metal compute pipelines, enabling GPU-accelerated compute on macOS without any CUDA dependency. On non-macOS platforms the crate compiles cleanly: `MetalDevice::new()` returns `Err(MetalError::UnsupportedPlatform)` directly, and `MetalBackend::init()` (the `ComputeBackend` trait entry point) surfaces the same underlying cause as `Err(BackendError::DeviceError("Metal requires macOS"))` — either way, cross-platform workspaces that depend on this crate keep compiling and get a typed error instead of a build failure.

## Features

- `MetalBackend` implementing 15 of the 24 `ComputeBackend` trait methods — see [Op Coverage](#op-coverage) below for exactly which
- Shared-mode `MTLBuffer` pool via `MetalMemoryManager` for efficient host-visible GPU allocation
- `MetalDevice` wraps the system-default Metal device (`metal::Device::system_default()`) and owns the single `MTLCommandQueue` shared by every compute pipeline built against it; there is no multi-device enumeration
- `msl` module with MSL source-string generators for GEMM, element-wise ops, reductions, conv2D, and attention — usable directly for custom kernel compilation
- `msl_nn` module with additional MSL kernels (softmax, layernorm, scan, `simdgroup_matrix` GEMM, extended-precision GEMM, INT8 GEMM)
- `mps` module: **parameter-validation descriptors only** for Metal Performance Shaders — no MPS framework calls exist yet (see the module's own doc comment for the roadmap)
- `ane` module: heuristic Apple Neural Engine generation detection and CoreML/GPU dispatch-hint scheduling metadata — the ANE has no Metal-visible dispatch path, so this module never executes anything on it
- Metal compute `pipeline` module for shader compilation and dispatch
- Conditional macOS compilation: the `metal` crate is the only platform-gated (`target_os = "macos"`) dependency

## Op Coverage

Honest scope, as of this crate's current source (not aspirational):

| `ComputeBackend` op | Status |
|---|---|
| `gemm`, `batched_gemm` | Metal-executed, f32. Operand layouts are validated up front (`validate_gemm_layout` in `backend/types.rs`) — an unsupported transpose / leading-dimension combination returns `BackendError::Unsupported` rather than a silently wrong answer. As of this writing the accepted layout is `NoTrans` operands with natural (contiguous) leading dimensions only; a wider-layout `_v2` GEMM path exists in `msl::gemm_v2` but is not yet wired into this dispatcher — check `validate_gemm_layout` directly for the current exact rule |
| `unary` | Metal-executed (relu, sigmoid, tanh, exp, log, sqrt, abs, neg — the full `UnaryOp` enum) |
| `binary` | Metal-executed (add, sub, mul, div, max, min — the full `BinaryOp` enum) |
| `reduce` | Metal-executed (sum, max, min, mean along one axis — the full `ReduceOp` enum) |
| `conv2d_forward`, `attention` | Implemented, but **not** Metal-accelerated: both round-trip every operand to host memory (`copy_dtoh`) and run as scalar Rust loops, even though finished MSL kernels for both (`msl::conv2d_msl`, `msl::attention_msl`) exist in this crate |
| `softmax`, `gather`, `scatter`, `gemm_mixed_precision`, `conv2d_backward_data`, `conv2d_backward_filter` | Not overridden — inherit the trait default, `Err(BackendError::Unsupported)` |
| `capabilities`, `available_devices`, `recommended_tile_for` | Not overridden — inherit the CPU-profile trait default (reports no FP16/tensor-core support, zero devices), not the real Apple Silicon numbers |

## Platform Support

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon / Intel) | GPU compute via Metal, scoped as in [Op Coverage](#op-coverage) above |
| Linux / Windows | Compile-only; `MetalDevice::new()` returns `Err(MetalError::UnsupportedPlatform)`, and `MetalBackend::init()` surfaces it through the trait as `Err(BackendError::DeviceError(_))` |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxicuda-metal = "0.5.5"
```

```rust
use oxicuda_metal::MetalBackend;
use oxicuda_backend::ComputeBackend;

let mut backend = MetalBackend::new();
backend.init()?;

let ptr = backend.alloc(256)?;
// ... copy data, launch Metal kernels ...
backend.free(ptr)?;
```

Most users should reach this crate through the `oxicuda` facade's `metal` feature (`oxicuda::backend::MetalBackend`, or `oxicuda::compute::default_backend()` for automatic backend selection) rather than depending on it directly — see the root [README](../../README.md#using-oxicuda-on-macos).

## Status

- **Version**: 0.5.5 (2026-08-12)
- **Tests**: 343 passing (measured 2026-08-12; concurrent work in this crate means this count moves quickly — re-run `cargo nextest run -p oxicuda-metal --all-features` for the current figure)

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
