# oxicuda-train

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-train` is a GPU-accelerated training engine providing the full optimizer and scheduling stack needed for deep learning. It delivers PTX-generated fused optimizer update kernels that keep all optimizer state on-device, minimising memory traffic, alongside gradient clipping, accumulation, checkpointing, mixed-precision training, and ZeRO-1/2/3 optimizer state sharding for memory-efficient distributed training.

## Features

- **Fused optimizer kernels** — Adam, AdamW, Lion, CAME, Muon, RAdam, RMSProp, AdaGrad; all parameter update kernels are PTX-generated and run entirely on-device
- **Gradient clipping** — Global norm clip, per-layer norm clip, and element-wise value clip
- **Gradient accumulation** — Micro-batch accumulation with configurable step count and averaging/summing modes
- **Gradient checkpointing** — Activation recomputation (uniform, selective, offload policies) to trade compute for memory
- **LR schedulers** — 11 built-in schedules: constant, step, multi-step, exponential, cosine, warmup+cosine, polynomial, 1cycle, cyclic, reduce-on-plateau, and linear warmup
- **ZeRO sharding** — ZeRO Stage 1/2/3 optimizer state, gradient, and parameter partitioning
- **AMP / GradScaler** — Automatic mixed-precision with dynamic loss scaling and overflow detection
- **EMA** — Exponential Moving Average of model parameters with configurable decay modes

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
oxicuda-train = "0.1.3"
```

```rust
use oxicuda_train::gpu_optimizer::{GpuOptimizer, ParamTensor};
use oxicuda_train::gpu_optimizer::adamw::GpuAdamW;
use oxicuda_train::grad_clip::clip_grad_norm;
use oxicuda_train::lr_scheduler::{LrScheduler, WarmupCosine};

let mut params = vec![
    ParamTensor::new(vec![0.5f32; 1024], "embed"),
    ParamTensor::new(vec![0.1f32; 4096], "ffn"),
];

let mut opt = GpuAdamW::new(3e-4).with_weight_decay(0.01);
let mut sched = WarmupCosine::new(3e-4, 500, 10_000);

for step in 0..10_000u64 {
    // ... populate param gradients ...
    clip_grad_norm(&mut params, 1.0).unwrap();
    let lr = sched.step();
    opt.set_lr(lr);
    opt.step(&mut params).unwrap();
    opt.zero_grad(&mut params);
}
```

## Status

**v0.1.3** (2026-04-17) — 165 tests passing

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
