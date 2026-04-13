# oxicuda-quant

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-quant` is a GPU-accelerated quantization and model compression engine for OxiCUDA. It provides a comprehensive suite of post-training quantization (PTQ), quantization-aware training (QAT), pruning, knowledge distillation, and mixed-precision analysis tools — all backed by PTX kernels for GPU-side execution.

## Features

- **Quantization schemes** (`scheme`): MinMax INT4/INT8, NF4 (QLoRA-style), FP8 E4M3/E5M2, GPTQ, and SmoothQuant — with per-tensor and per-channel granularity
- **QAT support** (`qat`): MinMax, MovingAvg, and Histogram observers; `FakeQuantize` with Straight-Through Estimator (STE) for gradient-preserving fake-quantization
- **Pruning** (`pruning`): magnitude-based unstructured pruning; structured channel, filter, and attention-head pruning
- **Knowledge distillation** (`distill`): KL-divergence, MSE, and cosine response distillation; intermediate feature distillation for layer-to-layer alignment
- **Sensitivity analysis** (`analysis`): per-layer quantization sensitivity scores, compression metrics (ratio, sparsity, bit-width), and mixed-precision policy generation
- **PTX kernels** (`ptx_kernels`): GPU kernel source strings for fake-quant, INT8 quant/dequant, NF4 encoding, and pruning mask application

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxicuda-quant = "0.1.0"
```

```rust
use oxicuda_quant::scheme::minmax::{MinMaxQuantizer, QuantScheme, QuantGranularity};

// Calibrate an INT8 symmetric quantizer on a set of activations.
let q = MinMaxQuantizer::int8_symmetric();
let data = vec![-1.0_f32, 0.0, 0.5, 1.0];
let params = q.calibrate(&data)?;
let codes  = q.quantize(&data, &params)?;
let deq    = q.dequantize(&codes, &params);
```

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
