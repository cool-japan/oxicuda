# oxicuda-fft

GPU-accelerated Fast Fourier Transform operations -- pure Rust cuFFT equivalent.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-fft` provides a complete GPU-accelerated FFT library implemented entirely in Rust,
targeting feature parity with NVIDIA's cuFFT. It generates PTX kernels at runtime using
the Stockham auto-sort algorithm, which is optimal for GPU execution because it avoids
bit-reversal permutations entirely.

The crate supports 1-D, 2-D, and 3-D transforms across complex-to-complex (C2C),
real-to-complex (R2C), and complex-to-real (C2R) modes, with both in-place and
out-of-place execution. For sizes up to 4096, a single-kernel strategy uses shared
memory for maximum throughput. Larger transforms employ a multi-stage ping-pong
approach with explicit transpose kernels.

Arbitrary-size FFTs are handled via the Bluestein/Chirp-Z algorithm, which reduces
any size N to a power-of-two convolution. Mixed-radix support (radix-3, 5, 7) is
also available for composite sizes.

## Modules

| Module       | Description                                              |
|--------------|----------------------------------------------------------|
| `types`      | Core types: `Complex<T>`, `FftType`, `FftDirection`, `FftPrecision` |
| `error`      | Error types and `FftResult<T>` alias                     |
| `plan`       | FFT plan creation with automatic strategy selection      |
| `execute`    | High-level `FftHandle` executor with GPU context         |
| `transforms` | C2C, R2C, C2R, 2-D, and 3-D transform dispatch          |
| `kernels`    | PTX kernel generators (Stockham, batch, large, transpose)|
| `radix`      | Butterfly implementations (radix-2/4/8, mixed, Bluestein)|

## Supported Transforms

- **C2C** -- Complex-to-complex, forward and inverse, in-place or out-of-place
- **R2C** -- Real-to-complex with Hermitian-symmetric output (N/2+1 complex values)
- **C2R** -- Complex-to-real inverse transform
- **2-D FFT** -- Row-wise FFT + transpose + column-wise FFT
- **3-D FFT** -- Extension of 2-D to volumetric data
- **Batched FFT** -- Multiple independent transforms in a single launch

## Quick Start

```rust,no_run
use oxicuda_fft::prelude::*;

// Create a 1-D complex-to-complex plan for 1024 elements
let plan = FftPlan::new_1d(1024, FftType::C2C, 1).expect("plan creation failed");

// Create a 2-D plan for a 256x256 grid
let plan_2d = FftPlan::new_2d(256, 256, FftType::C2C).expect("2d plan");

// With a GPU context:
// let handle = FftHandle::new(&ctx)?;
// handle.execute(&plan, input, output, FftDirection::Forward)?;
```

## Feature Flags

| Feature | Description                        |
|---------|------------------------------------|
| `f16`   | Half-precision (fp16) FFT support  |

## Status

| Metric | Value |
|--------|-------|
| Version | 0.1.4 |
| Tests passing | 314 |
| Release date | 2026-04-18 |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
