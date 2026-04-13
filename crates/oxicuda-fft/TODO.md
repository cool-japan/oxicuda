# oxicuda-fft TODO

GPU-accelerated Fast Fourier Transform operations, serving as a pure Rust equivalent to NVIDIA's cuFFT library. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.5).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 9,749 SLoC (35 files) -- Estimated: 63K-103K SLoC (estimation.md Vol.5 FFT portion)**

Current implementation covers the core FFT pipeline: plan creation, Stockham kernel generation, radix-2/4/8/mixed/Bluestein butterflies, C2C/R2C/C2R/2D/3D transforms, batch FFT, large FFT, and matrix transpose kernels.

### Completed

- [x] Core types (types.rs) -- Complex<T>, FftType, FftDirection, FftPrecision
- [x] Error handling (error.rs) -- FftError, FftResult<T>
- [x] FFT plan (plan.rs) -- FftPlan with 1D/2D/3D creation and strategy selection
- [x] FFT executor (execute.rs) -- FftHandle high-level execution dispatch
- [x] PTX helpers (ptx_helpers.rs) -- Shared PTX code generation utilities
- [x] Stockham kernel (kernels/stockham.rs) -- Stockham auto-sort FFT kernel generator
- [x] Batch FFT kernel (kernels/batch_fft.rs) -- Batched FFT kernel for multiple transforms
- [x] Large FFT kernel (kernels/large_fft.rs) -- Multi-pass FFT for large sizes
- [x] Transpose kernel (kernels/transpose.rs) -- Matrix transpose for multi-dim FFT
- [x] Radix-2 butterfly (radix/radix2.rs) -- DFT-2 butterfly PTX generation
- [x] Radix-4 butterfly (radix/radix4.rs) -- DFT-4 butterfly PTX generation
- [x] Radix-8 butterfly (radix/radix8.rs) -- DFT-8 butterfly PTX generation
- [x] Mixed-radix (radix/mixed_radix.rs) -- Composite-size FFT via mixed radixes
- [x] Bluestein algorithm (radix/bluestein.rs) -- Arbitrary-size FFT via chirp-Z
- [x] C2C transform (transforms/c2c.rs) -- Complex-to-complex forward/inverse
- [x] R2C transform (transforms/r2c.rs) -- Real-to-complex (Hermitian output)
- [x] C2R transform (transforms/c2r.rs) -- Complex-to-real (inverse of R2C)
- [x] 2D FFT (transforms/fft2d.rs) -- 2D FFT via row/column passes + transpose
- [x] 3D FFT (transforms/fft3d.rs) -- 3D FFT via three axis passes + transposes

### Future Enhancements

- [x] Stockham shared memory bank conflict avoidance (bank_conflict_free.rs) -- padded layout (addr + addr/32), power-of-2 sizes 64-4096 (P0)
- [x] Batched FFT kernel fusion (fused_batch.rs) -- multiple small FFTs per thread block, N<=1024, shared memory only (P0)
- [x] Prime-factor algorithm (pfa.rs) -- Good-Thomas FFT with CRT mapping for coprime factor decomposition (P1)
- [x] Split-radix FFT (split_radix.rs) -- radix-2/4 hybrid for ~10% fewer operations than pure radix-2 (P1)
- [x] Real-valued FFT optimization (real_fft.rs) -- pack/unpack exploiting conjugate symmetry to halve memory and compute (P1)
- [x] Multi-GPU FFT -- 1D slab decomposition across multiple GPUs (multi_gpu.rs) (P1)
- [x] Out-of-core FFT -- Handle transforms exceeding single GPU memory via staged host-device transfers (P1)
- [x] Pruned FFT -- Skip computation for known-zero input/output elements in zero-padded signals (P2)
- [x] Inverse FFT scaling -- Configurable 1/N scaling for inverse transforms (cuFFT compatibility) (P2)
- [x] Convolution via FFT helper (conv_fft.rs) -- 1D/2D convolution + cross-correlation using FFT multiply IFFT pattern (P2)
- [x] Half-precision FFT (FP16) -- FP16 storage with FP32 accumulation for memory-bound workloads (P2)
- [x] Callback functions -- User-defined load/store callbacks during FFT execution for fused operations (P2)

## Benchmark Coverage

- [x] Criterion benchmarks (benches/) -- CPU-side planning and dispatch heuristics

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA driver API (runtime loading) | Yes |
| oxicuda-memory | GPU memory management | Yes |
| oxicuda-launch | Kernel launch infrastructure | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| oxicuda-blas | BLAS operations (used by Bluestein) | Yes |
| num-complex | Complex number types | Yes |
| thiserror | Error derive macros | Yes |
| half (optional) | FP16 support | Yes |

## Quality Status

- All production code uses Result/Option (no unwrap)
- clippy::all and missing_docs warnings enabled
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- macOS compiles but returns UnsupportedPlatform at runtime

## Estimation vs Actual

| Metric | Estimated (Vol.5 FFT) | Actual |
|--------|----------------------|--------|
| SLoC | 63K-103K | 8,853 |
| Files | ~15-20 | 35 |
| Coverage | Full cuFFT parity | Core pipeline |
| Ratio | -- | ~3.5% of estimate |

The gap reflects the estimation targeting full cuFFT feature parity with exhaustive optimization, while the current implementation covers the functional core. The P0/P1 items above represent the path toward closing this gap.

---

## Blueprint Quality Gates (Vol.5 Sec 9)

### Functional Requirements

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| S1 | FFT 1D C2C — power-of-2 sizes correctness | P0 | [x] |
| S2 | FFT 1D R2C and C2R — Hermitian symmetry verified | P0 | [x] |
| S3 | FFT 2D — correctness via round-trip | P0 | [x] |
| S4 | FFT 3D — correctness via round-trip | P0 | [x] |
| S5 | FFT batched — correctness at batch size ≥ 32 | P0 | [x] |
| S6 | FFT arbitrary size — Bluestein algorithm for large primes | P1 | [x] |

### Performance Requirements

| # | Requirement | Target | Status |
|---|-------------|--------|--------|
| P1 | FFT 1D C2C, N=2²⁰ (1M elements) | ≥ 90% cuFFT throughput | [ ] |
| P2 | FFT 2D, 1024×1024 | ≥ 85% cuFFT throughput | [ ] |

---

## Numerical Accuracy Requirements (Vol.5 Sec 9)

| Transform | Precision | Acceptable Round-Trip Error |
|-----------|-----------|----------------------------|
| FFT 1D C2C | FP32 | \|IFFT(FFT(x)) - x\| / \|x\| < N × ε_machine |
| FFT 1D C2C | FP64 | \|IFFT(FFT(x)) - x\| / \|x\| < N × ε_machine |
| FFT 2D | FP32 | \|IFFT2(FFT2(x)) - x\| / \|x\| < N × ε_machine |
| R2C + C2R | FP32 | Round-trip within FP32 precision bound |

where ε_machine = 1.19e-7 for FP32, 2.22e-16 for FP64.

---

## Architecture-Specific Deepening Opportunities

### Ampere (sm_80) / Ada (sm_89)
- [x] `cp.async` used for global→shared twiddle factor prefetch in large FFT stages
- [x] Bank conflict avoidance in shared memory via padding verified for radix-4/8

### Hopper (sm_90)
- [x] TMA-based data loading for large FFT N > 64K
- [ ] Multi-GPU FFT via NVLink with overlapping compute/transfer

---

## Deepening Opportunities

- [ ] All S1–S6 functional requirements verified on GPU hardware
- [x] Round-trip accuracy verified for FP32 and FP64 across all transform types (S1–S5)
- [x] Bluestein algorithm correctness for non-power-of-2 sizes including prime N (S6)
- [ ] Performance benchmarks P1 and P2 measured and documented
- [x] Pruned FFT: only compute non-zero output elements (useful for zero-padded signals)
- [x] Out-of-core FFT: N larger than GPU memory via host-device streaming
