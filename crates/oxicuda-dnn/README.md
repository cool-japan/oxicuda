# oxicuda-dnn

GPU-accelerated deep learning primitives -- a pure Rust cuDNN equivalent.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-dnn` delivers the core building blocks for training and inference of
deep neural networks on NVIDIA GPUs, implemented entirely in Rust with no
C/Fortran dependencies. It covers convolution (five algorithm families with
forward and backward passes), multi-head attention with FlashAttention-2, MoE
routing, normalization layers, pooling, resize, and quantization.

The crate builds on `oxicuda-blas` for GEMM-based algorithms (im2col
convolution, MoE grouped GEMM) and on `oxicuda-ptx` for runtime PTX kernel
generation. `DnnHandle` manages a CUDA stream, a BLAS sub-handle, and a PTX
cache so that compiled kernels are reused across calls.

Algorithm selection is automatic: the convolution dispatcher tries a
CTA-tiled implicit-GEMM fast path first (`conv::fprop::tiled_implicit_gemm`,
wired transparently inside the implicit-GEMM engine -- 5.7-8.0 TFLOPS on
real face-pipeline 3x3 shapes vs. a ~900 GFLOPS scalar baseline when the
shape qualifies), then Winograd F(2,3) once the shape clears a measured
profitability threshold (`conv::algo_select::winograd_forward_implemented`
is `true` -- forward is a real, hardware-validated implementation, not a
skeleton), then falls back through im2col+GEMM, direct, and FFT-based
strategies. Winograd F(4,3) forward and the Winograd *backward* passes
(dgrad/wgrad) remain unimplemented skeletons -- see "Supported Operations"
below. Fused epilogues (conv+BN+ReLU, LayerNorm+activation,
fused_add_rms_norm) are provided to minimize global memory traffic;
conv+BN+ReLU in particular is a decomposed real convolution plus a combined
BN-affine + activation kernel pass, not a single monolithic kernel (see
"Supported Operations" below).

## Modules

| Module | Description |
|--------|-------------|
| `handle` | `DnnHandle` -- central entry point binding context, stream, BLAS, PTX cache |
| `types` | `TensorDesc`, `TensorLayout` (NCHW/NHWC), `Activation`, `ConvolutionDescriptor` |
| `conv` | Convolution forward (5 algos), backward dgrad/wgrad, fused conv+BN+ReLU |
| `attn` | FlashAttention-2 (fwd+bwd), PagedAttention, decode attention, RoPE, KV-cache |
| `moe` | MoE routing (top-k softmax), token permutation, fused MoE kernel |
| `norm` | LayerNorm, RMSNorm, BatchNorm, GroupNorm, fused variants |
| `pool` | MaxPool2D, AvgPool2D, AdaptivePool, GlobalPool |
| `resize` | Nearest, bilinear, bicubic interpolation |
| `quantize` | FP8 (E4M3/E5M2), INT8 (symmetric/asymmetric), block-scaled FP4 |
| `error` | `DnnError` / `DnnResult` |

## Quick Start

```rust,no_run
use std::sync::Arc;
use oxicuda_driver::Context;
use oxicuda_dnn::prelude::*;

fn main() -> DnnResult<()> {
    let ctx: Arc<Context> = unimplemented!();

    // Create a DNN handle with a 1 MiB workspace
    let mut handle = DnnHandle::new(&ctx)?;
    handle.set_workspace(1 << 20)?;

    // Convolution forward pass (algorithm auto-selected):
    //   conv::api::conv_forward(&mut handle, &conv_desc,
    //       ConvAlgorithm::ImplicitGemm,
    //       &x_desc, &w_desc, &y_desc)?;

    // FlashAttention-2 forward:
    //   attn::flash_attn::forward::flash_attention_forward(
    //       &mut handle, &q, &k, &v, &out, scale, causal)?;

    Ok(())
}
```

## Supported Operations

### Convolution

| Algorithm | Forward | dgrad | wgrad |
|-----------|---------|-------|-------|
| Implicit GEMM¹ | yes | yes | yes |
| im2col + GEMM | yes | -- | -- |
| Winograd F(2,3) | yes² | skeleton³ | skeleton³ |
| Winograd F(4,3) | not implemented⁴ | -- | -- |
| Direct (1x1, depthwise) | yes | -- | -- |
| FFT-based | yes | -- | -- |

¹ Forward transparently dispatches through a CTA-tiled, register-blocked
GEMM mainloop (`conv/fprop/tiled_implicit_gemm.rs`) when the shape
qualifies (F32/NCHW/`groups==1`/2-D, above minimum K/output-channel/CTA-count
thresholds) -- 5.7-8.0 TFLOPS vs. a ~900 GFLOPS scalar baseline on real
face-pipeline 3x3 shapes; declined shapes fall back to the scalar,
one-thread-per-output-element kernel. `ImplicitGemmConv::scalar_only` pins
the scalar path directly, and the `OXICUDA_DISABLE_TILED_CONV` environment
variable disables tiling process-wide (A/B measurement, bisecting a
suspected miscompare).

² A genuine F(2x2,3x3) implementation (`conv/fprop/winograd/`): input
transform, filter transform, a shared-memory-tiled batched GEMM over the 16
transform-domain positions, output transform + bias. Hardware-validated on
an RTX A4000 against an `f64` CPU oracle (relative L2 `1.0e-7`..`1.7e-7`)
and `ImplicitGemmConv` (`7.9e-8`..`1.4e-6`); `conv::algo_select` routes
eligible shapes to it only above a measured profitability threshold --
below it, Winograd's four kernel launches lose to implicit-GEMM's one.

³ Tile selection, workspace sizing, and transform-matrix constants exist
for both (`conv/dgrad/winograd.rs`, `conv/wgrad/winograd.rs`), but the
kernel bodies are load/launch-only skeletons that perform no numeric work
(they leave the output buffer untouched rather than computing a wrong
answer). Not wired into `conv_backward_data`/`conv_backward_filter` at all
(those use separate implicit-GEMM dgrad/wgrad engines) and only reachable
by constructing `WinogradDgrad`/`WinogradWgrad` directly. See each file's
"Implementation status" module docs.

⁴ Explicitly rejected by `WinogradTileSize::forward_supported` -- the
larger transform's coefficients (`1/24`, `1/12`, ...) amplify FP32
round-off enough to need its own error budget, so it was deliberately not
enabled by inheritance from F(2,3).

Fused: conv + BatchNorm + ReLU via `conv_bn_relu` -- decomposed into a real
convolution dispatch plus one combined BN-affine + activation kernel pass,
not a single monolithic kernel launch. (The single-kernel `FusedConvBnAct`
engine also exists in `conv/fused.rs`, but its kernel body is currently a
load/launch-only skeleton and is not used by `conv_bn_relu`.)

### Attention

- FlashAttention-2 forward and backward with online softmax
- PagedAttention for LLM KV-cache serving
- Decode attention for autoregressive generation
- Rotary Position Embedding (RoPE)

### Mixture of Experts (MoE)

- Top-k softmax routing with load balancing
- Token permutation / unpermutation
- Fused MoE kernel (TokenParallel / ExpertParallel strategies)
- Grouped GEMM backend

### Normalization

| Layer | Training | Inference | Fused Variants |
|-------|----------|-----------|----------------|
| LayerNorm | yes | yes | LN + ReLU |
| RMSNorm | yes | yes | fused_add_rms_norm, RMSNorm + SiLU |
| BatchNorm | yes | yes | conv + BN + ReLU |
| GroupNorm | yes | yes | -- |

### Pooling and Resize

- MaxPool2D, AvgPool2D, AdaptivePool, GlobalPool
- Nearest, bilinear, bicubic interpolation resize

### Quantization

- FP8: E4M3 and E5M2 quantize / dequantize
- INT8: symmetric and asymmetric per-tensor / per-channel
- Block-scaled FP4 for weight compression

## Feature Flags

| Feature | Description |
|---------|-------------|
| `f16` | Enable FP16 / BF16 tensor support (enables `oxicuda-blas/f16`) |

## Tensor Layouts

NCHW (PyTorch default), NHWC (Tensor Core optimal), NCDHW, NDHWC (3-D), and
generic RowMajor for 2-D intermediates. Channels-last layouts (NHWC/NDHWC) are
recommended for best Tensor Core utilization.

## Performance Targets

Convolution forward aims for 90% of cuDNN throughput on common CNN shapes
(ResNet-50, EfficientNet). FlashAttention-2 targets parity with the reference
Tri Dao kernel at sequence lengths 512--8192.

## Status

| Item | Value |
|------|-------|
| Version | 0.5.5 |
| Release date | 2026-08-13 |
| Tests | 1,291 passing |
| Warnings | 0 (clippy clean) |
| `unwrap()` | 0 (production code) |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
