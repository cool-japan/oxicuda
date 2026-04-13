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

Algorithm selection is automatic: the convolution dispatcher benchmarks
implicit-GEMM, im2col+GEMM, Winograd, direct, and FFT-based strategies and
picks the fastest for each problem shape. Fused kernels (conv+BN+ReLU,
LayerNorm+activation, fused_add_rms_norm) are provided to minimize global
memory traffic.

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
| Implicit GEMM | yes | yes | yes |
| im2col + GEMM | yes | -- | -- |
| Winograd F(2,3) / F(4,3) | yes | -- | -- |
| Direct (1x1, depthwise) | yes | -- | -- |
| FFT-based | yes | -- | -- |

Fused: conv + BatchNorm + ReLU in a single kernel launch.

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

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
