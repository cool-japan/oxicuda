# oxicuda-signal

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-signal` is a GPU-accelerated signal, audio, and image processing library for the OxiCUDA ecosystem. It provides CPU reference implementations alongside PTX-generating GPU kernels for a wide range of DSP operations — from spectral transforms and wavelet decompositions to audio feature extraction and computer vision primitives. Kernels are emitted as WGSL/PTX strings at runtime and compiled via the CUDA JIT; no pre-compiled `.ptx` files are shipped.

## Features

- **Spectral transforms** — DCT-II/III/IV, MDCT/IMDCT via the `dct` module; Haar, Daubechies db2–db10, Symlets, Biorthogonal, and Coiflet wavelets with multi-level DWT via the `dwt` module
- **Audio processing** — STFT, mel filterbank, MFCC, chroma features, and spectrogram variants (magnitude, power, mel)
- **Filtering** — FIR (windowed-sinc, raised-cosine), IIR (biquad SOS cascade), and Wiener filter
- **Correlation** — Autocorrelation, cross-correlation, GCC-PHAT time-delay estimation, circular convolution, PACF, Ljung-Box Q
- **Image processing** — Non-Maximum Suppression (greedy, soft, heatmap), morphological operations (erode, dilate, open, close, gradient), Gaussian blur, Sobel edge detection
- **Window functions** — Hann, Hamming, Blackman, Kaiser, flat-top and others with ENBW/CG metrics
- **Zero-copy PTX** — All GPU kernels generated as `String` at runtime for JIT compilation

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
oxicuda-signal = "0.1.4"
```

```rust
use oxicuda_signal::prelude::*;

// Compute MFCC features from a raw audio frame
let samples: Vec<f32> = vec![0.0; 16000];
let config = MfccConfig::default();
let mfcc_features = mfcc(&samples, config).unwrap();

// Multi-level DWT decomposition
let signal = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let decomp = multilevel_forward(&signal, WaveletFamily::Haar, 2).unwrap();
let reconstructed = multilevel_inverse(&decomp, WaveletFamily::Haar).unwrap();
```

## Status

**v0.1.4** (2026-04-18) — 240 tests passing

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
