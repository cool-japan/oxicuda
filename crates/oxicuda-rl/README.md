# oxicuda-rl

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-rl` provides a comprehensive set of GPU-ready reinforcement learning building blocks, covering experience replay buffers, policy distributions, return/advantage estimators, RL algorithm loss functions, and environment abstractions. All components are designed for on-device operation to minimise host-device memory traffic, with PTX kernel sources for GPU-accelerated RL operations.

## Status

| Version | Tests | Date |
|---------|-------|------|
| 0.1.3 | 164 passing | 2026-04-17 |

## Features

- **Replay Buffers** — Uniform replay (DQN, SAC, TD3), Prioritized Experience Replay with segment-tree PER and IS weight computation (PER-DQN, PER-SAC), and N-step return accumulation
- **Policy Distributions** — Categorical (discrete actions, Gumbel-max sampling), Gaussian (continuous actions with reparameterisation trick and optional Tanh squashing for SAC), and Deterministic (DDPG/TD3 with Ornstein-Uhlenbeck noise)
- **Return & Advantage Estimators** — GAE (PPO/A3C), TD(λ), V-trace off-policy correction (IMPALA), and Retrace(λ) safe off-policy Q-targets
- **Loss Functions** — PPO clip+value+entropy, DQN/Double-DQN Bellman MSE/Huber, SAC soft Q + actor losses with automatic temperature tuning, TD3 twin-Q critic and deterministic actor losses
- **Normalization** — Running mean/variance observation normalization, return-based reward normalization, Welford online statistics
- **Environment Abstractions** — `Env` trait, `VecEnv` vectorized wrapper with auto-reset, `LinearQuadraticEnv` reference environment
- **PTX Kernels** — GPU PTX source strings for TD-error, PPO ratio, SAC target, PER IS weight computation, and advantage normalization

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
oxicuda-rl = "0.1.3"
```

```rust
use oxicuda_rl::buffer::UniformReplayBuffer;
use oxicuda_rl::estimator::{GaeConfig, compute_gae};
use oxicuda_rl::loss::{PpoConfig, ppo_loss};
use oxicuda_rl::handle::RlHandle;

// Set up replay buffer
let mut buf = UniformReplayBuffer::new(10_000, 8, 4);
let mut handle = RlHandle::default_handle();

// Push experience and sample a mini-batch
for i in 0..100_usize {
    buf.push(vec![i as f32; 8], vec![0.0_f32; 4], 1.0, vec![i as f32 + 1.0; 8], false);
}
let batch = buf.sample(32, &mut handle).unwrap();

// Compute GAE for a 5-step rollout
let rewards   = vec![1.0_f32; 5];
let values    = vec![0.5_f32; 5];
let next_vals = vec![0.5_f32; 5];
let dones     = vec![0.0_f32; 5];
let gae = compute_gae(&rewards, &values, &next_vals, &dones, GaeConfig::default()).unwrap();
assert_eq!(gae.advantages.len(), 5);
```

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
