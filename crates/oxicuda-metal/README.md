# oxicuda-metal

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-metal` provides a `MetalBackend` that implements the `ComputeBackend` trait from `oxicuda-backend` using Apple's Metal API. It targets Apple Silicon and Intel Mac GPUs through Metal compute pipelines, enabling GPU-accelerated compute on macOS without any CUDA dependency. On non-macOS platforms the crate compiles cleanly: `MetalDevice::new()` returns `Err(MetalError::UnsupportedPlatform)` directly, and `MetalBackend::init()` (the `ComputeBackend` trait entry point) surfaces the same underlying cause as `Err(BackendError::DeviceError("Metal requires macOS"))` — either way, cross-platform workspaces that depend on this crate keep compiling and get a typed error instead of a build failure.

## Features

- `MetalBackend` implementing 18 of the 24 `ComputeBackend` trait methods — see [Op Coverage](#op-coverage) below for exactly which
- Shared-mode `MTLBuffer` pool via `MetalMemoryManager` for efficient host-visible GPU allocation
- `MetalDevice` wraps the system-default Metal device (`metal::Device::system_default()`) and owns the single `MTLCommandQueue` shared by every compute pipeline built against it; there is no multi-device enumeration
- `msl` module with MSL source-string generators for GEMM (including a runtime-parameterised v2 kernel with full transpose support), element-wise ops, reductions, conv2D, and attention — usable directly for custom kernel compilation, and now also what `backend/nn.rs` dispatches for real for `conv2d_forward`/`attention`
- `msl_nn` module with additional MSL kernels (softmax, layernorm, scan, `simdgroup_matrix` GEMM, extended-precision GEMM, INT8 GEMM), plus a single-pass online-softmax attention v2 kernel
- `mps` module: **parameter-validation descriptors only** for Metal Performance Shaders — no MPS framework calls exist yet (see the module's own doc comment for the roadmap)
- `ane` module: heuristic Apple Neural Engine generation detection and CoreML/GPU dispatch-hint scheduling metadata — the ANE has no Metal-visible dispatch path, so this module never executes anything on it
- Metal compute `pipeline` module for shader compilation and dispatch
- Conditional macOS compilation: the `metal` crate is the only platform-gated (`target_os = "macos"`) dependency

## Op Coverage

Honest scope, as of this crate's current source (not aspirational). Updated
2026-08-12 after an adversarial audit found and fixed two false completions
(`conv2d_forward` and `attention` were checked in as "Metal-accelerated" while
actually running as host-side scalar loops) — see `TODO.md`/`CHANGELOG.md` for
the full writeup:

| `ComputeBackend` op | Status |
|---|---|
| `gemm`, `batched_gemm` | Metal-executed, f32, via a runtime-parameterised v2 kernel (`gemm_msl_v2`/`batched_gemm_msl_v2`) that supports **all four transpose combinations**, including padded (non-tightly-packed) leading dimensions. `validate_gemm_layout` (`backend/types.rs`) now only rejects a leading dimension smaller than the operand it describes (`BackendError::InvalidArgument`) — every transpose combination is accepted and computed, where earlier releases accepted `NoTrans`-only and correctly (not silently) rejected the rest as `Unsupported` |
| `unary` | Metal-executed (relu, sigmoid, tanh, exp, log, sqrt, abs, neg — the full `UnaryOp` enum). `msl` also generates `gelu`/`silu` MSL source, but `UnaryOp` has no such variants, so they are not reachable through this method today |
| `binary` | Metal-executed (add, sub, mul, div, max, min — the full `BinaryOp` enum). `msl` also generates a `pow` op, unreachable for the same reason (`BinaryOp` has no `Pow` variant) |
| `reduce` | Metal-executed (sum, max, min, mean along one axis — the full `ReduceOp` enum); flat 1-D reductions of >= 4096 elements take a two-pass GPU path (`chunked_reduce_msl`) instead of one oversized threadgroup |
| `conv2d_forward` | **Metal-executed** (`conv2d_msl_v2`, NCHW, one thread per output element), unconditionally — no host fallback. Before 2026-08-12 this ran as a host-side scalar loop despite a finished MSL kernel already existing in the crate with no caller |
| `attention` | **Metal-executed** (`attention_msl_v2`, single-pass online-softmax, one SIMD-group per query); falls back to a host implementation only when `head_dim`'s accumulator does not fit in one threadgroup-memory slice for any SIMD-group count — a real, caller-reachable limit, not a silent no-op. Same pre-2026-08-12 history as `conv2d_forward` above |
| `softmax` | **Metal-executed**, numerically stable, **last axis only** — an earlier axis returns `BackendError::Unsupported` rather than silently reducing the wrong dimension. Before 2026-08-12 this inherited the trait's `Unsupported` default despite a complete shader already shipping in the crate |
| `capabilities`, `available_devices` | Overridden with real, device-probed values (`MetalDevice::capabilities()`, driven by `supportsFamily:` plus the driver-reported threadgroup limits): FP16 support, `simdgroup_matrix` (Apple7+) mapped to `tensor_cores`, the real unified-memory flag, and real threadgroup/shared-memory limits — not the CPU-profile guess |
| `gather`, `scatter`, `gemm_mixed_precision`, `conv2d_backward_data`, `conv2d_backward_filter` | Not overridden — inherit the trait default, `Err(BackendError::Unsupported)` |
| `recommended_tile_for` | Not overridden — inherits the CPU-profile trait default |

Two capabilities exist but are opt-in / incomplete, so they are not claimed as
plain "done" above:

- **Asynchronous dispatch** (`MetalBackend::set_async_dispatch`) — commits a
  command buffer and returns immediately instead of blocking on
  `waitUntilCompleted`, awaited at the next real synchronisation point
  (`synchronize`, a host read/write, `free`, a custom-kernel launch, or
  backend drop). **Off by default** — every op above still blocks per-call
  unless a caller opts in.
- **FP16 GEMM** (`MetalBackend::gemm_f16`) — an inherent method (the trait's
  `gemm`/`batched_gemm` are f32-only), dispatching the same v2 kernel family
  with `GemmDtype::F16`.

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
- **Tests**: 401 passing, clippy-clean (measured 2026-08-12, `--all-features`, real Apple M3 hardware; concurrent work in this crate means this count moves quickly — re-run `cargo nextest run -p oxicuda-metal --all-features` for the current figure)

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
