# oxicuda-rand

GPU-accelerated random number generation -- pure Rust cuRAND equivalent.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-rand` provides GPU-accelerated random number generation implemented entirely
in Rust, targeting feature parity with NVIDIA's cuRAND. It includes three PRNG engines,
four distribution transforms, and Sobol quasi-random sequences, all generating numbers
in parallel on the GPU via PTX kernels.

Each engine is counter-based or state-based, producing statistically independent streams
across GPU threads. Philox-4x32-10 is the default (matching cuRAND), offering an
excellent balance of speed and quality. XORWOW provides fast generation for Monte Carlo
workloads. MRG32k3a delivers the highest statistical quality for applications that
demand it.

The high-level `RngGenerator` API dispatches to any engine and automatically tracks
per-thread offsets to ensure non-overlapping sequences across successive generation
calls. Sobol quasi-random sequences support up to 20 dimensions with Gray-code
generation for low-discrepancy sampling in Monte Carlo integration.

## Engines

| Engine        | Module               | Description                            |
|---------------|----------------------|----------------------------------------|
| Philox-4x32-10 | `engines::philox`  | Counter-based, cuRAND default, 2^128 period |
| XORWOW        | `engines::xorwow`   | XORshift + Weyl sequence, fast         |
| MRG32k3a      | `engines::mrg32k3a` | Combined MRG, highest statistical quality |

## Distributions

| Distribution | Module                    | Method                              |
|--------------|---------------------------|-------------------------------------|
| Uniform      | `distributions::uniform`  | [0, 1) in f32/f64                   |
| Normal       | `distributions::normal`   | Box-Muller transform                |
| Log-normal   | `distributions::log_normal` | exp(Normal(mu, sigma))            |
| Poisson      | `distributions::poisson`  | Knuth (small lambda), normal approx (large) |

## Quasi-Random

| Sequence | Module         | Description                                |
|----------|----------------|--------------------------------------------|
| Sobol    | `quasi::sobol` | Gray-code Sobol, up to 20 dimensions       |

## Quick Start

```rust,no_run
use oxicuda_rand::prelude::*;

// With a GPU context:
// let mut gen = RngGenerator::new(&ctx, RngEngine::Philox, 42)?;
//
// // Generate 1M uniform f32 values on the GPU
// let uniform = gen.generate_uniform_f32(1_000_000)?;
//
// // Generate normal-distributed f64 values
// let normal = gen.generate_normal_f64(1_000_000, 0.0, 1.0)?;
//
// // Sobol quasi-random sequence (5 dimensions)
// let sobol = SobolGenerator::new(&ctx, 5)?;
// let points = sobol.generate(10_000)?;
```

## Feature Flags

| Feature    | Description                          |
|------------|--------------------------------------|
| `f16`      | Half-precision (fp16) RNG support    |
| `gpu-tests`| Enable GPU integration tests         |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
