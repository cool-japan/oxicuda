# oxicuda-webgpu

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-webgpu` is the cross-platform GPU compute backend for OxiCUDA, targeting Vulkan, Metal, Direct3D 12, and the browser WebGPU API from a single Rust crate via [`wgpu`](https://wgpu.rs/). It implements the `ComputeBackend` trait from `oxicuda-backend` and provides a `WebGpuBackend` with a pooled buffer allocator and a WGSL shader generator for GEMM, element-wise ops, reductions, convolution, and attention. On macOS it is a second GPU path alongside the [`oxicuda-metal`](../oxicuda-metal/README.md) feature — `wgpu` itself runs on top of Metal there, so both features ultimately drive the same Apple GPU through different API surfaces; see [Op Coverage](#op-coverage) below for exactly which ops each one reaches.

## Features

- **Cross-platform** — Single backend targets Vulkan (Linux/Windows), Metal (macOS/iOS), Direct3D 12 (Windows), and browser WebGPU via `wgpu`
- `WebGpuBackend` implementing 15 of the 24 `ComputeBackend` trait methods — see [Op Coverage](#op-coverage) below for exactly which
- **WGSL shader generation** — The `shader` module generates WGSL source strings at runtime for GEMM (including an FP16 `enable f16` variant), element-wise, reduction, convolution, and attention kernels
- **Buffer pool** — `WebGpuMemoryManager` maintains a `u64` handle → `wgpu::Buffer` map for efficient allocation and reuse, validated against the adapter's *real* reported limits (`adapter.limits()`), not the WebGPU conformance baseline
- **Pipeline + bind-group dispatch cache** — `backend_cache` caches each compiled `wgpu::ComputePipeline` with its bind-group layout, and reuses the `wgpu::BindGroup` itself (via a refreshed dedicated uniform buffer) across repeated dispatches on the same operand handles — the common shape of a training/inference loop
- **Non-fatal GPU error capture** — installs a `wgpu` uncaptured-error handler so a device-side validation failure surfaces as a typed `Err` instead of the default (process-fatal) behaviour
- **Backend abstraction** — Implements `oxicuda-backend`'s `ComputeBackend` for interoperability with the rest of the OxiCUDA ecosystem
- **Pure Rust** — Zero C/Fortran in the default feature set; `wgpu` handles all native API calls
- `wasm` feature — compiles for `wasm32` with `wasm-bindgen`/`js-sys` for browser WebGPU targets (enabled transitively by the `oxicuda` umbrella's `wasm-backend` feature)
- `gpu-tests` feature — makes the device-presence tests in `tests/gpu_presence.rs` fail (instead of silently skipping) when no real `wgpu` adapter can be acquired, for use on a machine known to have a GPU

## Architecture

```
WebGpuBackend  (implements ComputeBackend)
      │
  WebGpuDevice       ← wgpu Instance + Adapter + Device + Queue
      │
  WebGpuMemoryManager  ← buffer pool (u64 handle → wgpu::Buffer)
```

## Op Coverage

Honest scope, as of this crate's current source (not aspirational). Updated
2026-08-12 after an adversarial audit pass (see `TODO.md`/`CHANGELOG.md`) —
unlike the sibling `oxicuda-metal` crate audited the same session, this
crate's `conv2d_forward`/`attention` were already real GPU dispatch before
that pass; the pass instead found and fixed a real-device-limits bug, a
process-fatal error handler, and several integer-overflow / bounds bugs in
generated shaders:

| `ComputeBackend` op | Status |
|---|---|
| `gemm`, `batched_gemm` | WebGPU-executed, f32, via `shader::gemm_wgsl` / `batched_gemm_wgsl`. All four transpose combinations are honoured, including padded (non-tightly-packed) leading dimensions — `packed_gemm_lds` is deliberately identical to `oxicuda-metal`'s, so the two GPU backends accept the same argument space and produce the same numbers |
| `unary` | WebGPU-executed (relu, sigmoid, tanh, exp, log, sqrt, abs, neg — the full `UnaryOp` enum). Unlike `oxicuda-metal`, this crate has **no** `gelu`/`silu` WGSL generator at all |
| `binary` | WebGPU-executed (add, sub, mul, div, max, min — the full `BinaryOp` enum) with the `binary_wgsl` generator. The generator also emits a `pow` op, but `BinaryOp` has no `Pow` variant so it is not reachable through this trait method |
| `reduce` | WebGPU-executed (sum, max, min, mean along one axis — the full `ReduceOp` enum) via `reduction_wgsl` + `reduction_final_wgsl` |
| `conv2d_forward` | WebGPU-executed (NCHW compute shader, `shader::conv2d_wgsl`), with a CPU fallback for shapes the shader's fixed-literal 2-D dispatch grid cannot express — the fallback is also the correctness oracle GPU results are checked against, so the two paths cannot silently drift apart |
| `attention` | WebGPU-executed (scaled dot-product, stable softmax, causal masking), with the same class of CPU fallback as `conv2d_forward` for oversized configurations |
| `softmax`, `gather`, `scatter`, `gemm_mixed_precision`, `conv2d_backward_data`, `conv2d_backward_filter` | Not overridden — inherit the trait default, `Err(BackendError::Unsupported)` |
| `capabilities`, `available_devices`, `recommended_tile_for` | Not overridden — inherit the CPU-profile trait default, not the real adapter numbers reported in `WebGpuDevice::limits`/`supports_f16` |

FP16 GEMM (`WebGpuBackend::gemm_f16`) is an inherent method (the trait's
`gemm`/`batched_gemm` are f32-only), gated on the adapter actually supporting
`wgpu::Features::SHADER_F16` (`WebGpuDevice::supports_f16`) — it returns a
typed error rather than silently running in reduced precision on hardware
that lacks the feature.

## Platform Support

| Platform | Status |
|----------|--------|
| Linux / Windows | GPU compute via Vulkan or Direct3D 12 (whichever `wgpu` selects), scoped as in [Op Coverage](#op-coverage) above |
| macOS (Apple Silicon / Intel) | GPU compute via `wgpu`'s Metal backend — the same physical GPU the [`oxicuda-metal`](../oxicuda-metal/README.md) feature drives, through a different API surface and a different (WGSL, not MSL) op coverage; see the root [README](../../README.md#using-oxicuda-on-macos) |
| Browser (`wasm32`, `wasm` feature) | GPU compute via the browser's native WebGPU implementation |
| No adapter available | `WebGpuBackend::init()` returns a typed `Err` rather than panicking |

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
oxicuda-webgpu = "0.5.5"
```

```rust
use oxicuda_webgpu::WebGpuBackend;
use oxicuda_backend::ComputeBackend;

fn main() -> oxicuda_backend::BackendResult<()> {
    let mut backend = WebGpuBackend::new();
    backend.init()?;

    let ptr = backend.alloc(1024)?;
    backend.free(ptr)?;
    Ok(())
}
```

Most users should reach this crate through the `oxicuda` facade's `webgpu` feature (`oxicuda::backend::WebGpuBackend`, or `oxicuda::compute::default_backend()` for automatic backend selection) rather than depending on it directly.

## Status

- **Version**: 0.5.5 (2026-08-12)
- **Tests**: 293 of 293 passing, clippy-clean (measured 2026-08-12, `--all-features`, real Apple M3 hardware running the Metal-backed `wgpu` path). Concurrent work in this crate means this count moves quickly — re-run `cargo nextest run -p oxicuda-webgpu --all-features` for the current figure.

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
