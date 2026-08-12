# oxicuda-gen

Generative AI primitives for OxiCUDA -- diffusion schedulers, classifier-free
guidance, VAE codec, LoRA adapters, and score-network building blocks, in
pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project. See the
[workspace README](../../README.md) for the full crate map.

## Overview

`oxicuda-gen` ships the algorithmic core of modern diffusion and flow-based
generative models: DDPM forward/reverse processes, deterministic DDIM, fast
DPM-Solver++ first/second-order steps, rectified Flow Matching schedulers,
classifier-free guidance (with adaptive policies and Perp-Neg), the encoder /
decoder / KL / VQ pieces of a VAE codec, low-rank LoRA adapters with merge /
unmerge round-trip, and timestep + attention blocks for score networks.

The reference implementation runs on CPU `Vec<f32>` tensors and emits PTX
strings (`ddpm_step`, `cfg_combine`, `lora_apply`, `flow_velocity`,
`vae_kl_loss`, `timestep_embed`) targeting SM 7.5 through SM 12.0. Default
features are 100 % pure Rust; the only required dependency is `thiserror`.

## Modules

| Module | Description |
|--------|-------------|
| `error` | `GenError` / `GenResult` |
| `handle` | `GenHandle`, `SmVersion`, `LcgRng` |
| `scheduler::beta_schedule` | `BetaSchedule` linear / cosine / sigmoid betas, `BetaScheduleType` |
| `scheduler::ddpm` | `DdpmScheduler` forward `add_noise` and `step` |
| `scheduler::ddim` | `DdimScheduler` deterministic / stochastic ODE step |
| `scheduler::dpm_solver` | `DpmSolverScheduler` (`DpmOrder::First`/`Second`) high-order ODE solver |
| `scheduler::flow_matching` | `FlowMatchingScheduler`, `FlowMatchingPath` rectified-flow interpolation |
| `guidance::cfg` | `CfgGuidance`, `CfgConfig` classifier-free guidance |
| `guidance` | `AdaptiveCfgScheduler`, `AdaptiveCfgPolicy`, `PerpNegGuidance` |
| `vae::encoder` / `vae::decoder` | `Encoder`, `Decoder` with weights and configs |
| `vae::kl` | `GaussianLatent::kl_loss` for the standard ELBO term |
| `vae::quantize` | `VqCodebook` nearest-neighbour quantizer |
| `lora::adapter` | `LoraConfig`, `LoraLinear`, `LoraModel`, `merge_lora` / `unmerge_lora` |
| `score::timestep` | `SinusoidalEmbedding`, `FourierEmbedding` |
| `score::blocks` | `UNetResBlock`, `SelfAttentionBlock`, `CrossAttentionBlock` |
| `ptx_kernels` | GPU PTX kernel strings (six entry points) |

## Quick Start

```rust,no_run
use oxicuda_gen::prelude::*;

// 50-step DDIM sampling over 1000 training timesteps, eta = 0 (deterministic).
let scheduler = DdimScheduler::new(1000, 50, 0.0)?;

// Classifier-free guidance with scale = 7.5 (typical Stable Diffusion).
let guide = CfgGuidance::new(CfgConfig::new(7.5)?);

let cond   = vec![0.5_f32; 64];
let uncond = vec![0.0_f32; 64];
let combined = guide.apply(&cond, &uncond)?;
assert_eq!(combined.len(), 64);
# Ok::<(), GenError>(())
```

## Status

| Item | Value |
|------|-------|
| Version | 0.5.2 |
| Release date | 2026-07-27 |
| Default features | Pure Rust (`thiserror` only) |
| `unwrap()` | 0 in production code |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
