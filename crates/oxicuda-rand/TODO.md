# oxicuda-rand TODO

GPU-accelerated random number generation, serving as a pure Rust equivalent to NVIDIA's cuRAND library. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.5).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual: 10,115 SLoC (27 files) -- Estimated: 15K-24K SLoC (estimation.md Vol.5 rand portion)**

Current implementation covers three PRNG engines (Philox-4x32-10, XORWOW, MRG32k3a), four distributions (uniform, normal, log-normal, Poisson), and one quasi-random sequence (Sobol).

### Completed

- [x] Error handling (error.rs) -- RandError, RandResult<T>
- [x] Generator abstraction (generator.rs) -- RngEngine enum, RngGenerator with engine selection
- [x] Philox engine (engines/philox.rs) -- Philox-4x32-10 counter-based PRNG (cuRAND default)
- [x] XORWOW engine (engines/xorwow.rs) -- XORshift with Weyl sequence addition
- [x] MRG32k3a engine (engines/mrg32k3a.rs) -- Combined multiple recursive generator
- [x] Uniform distribution (distributions/uniform.rs) -- u32 to uniform float [0,1) in f32/f64
- [x] Normal distribution (distributions/normal.rs) -- Box-Muller transform for Gaussian samples
- [x] Log-normal distribution (distributions/log_normal.rs) -- Exponentiation of normal samples
- [x] Poisson distribution (distributions/poisson.rs) -- Knuth (small lambda) + normal approximation (large lambda)
- [x] Sobol sequences (quasi/sobol.rs) -- Sobol low-discrepancy sequences via Joe-Kuo direction numbers

### Future Enhancements

- [x] Philox kernel optimization (philox_optimized.rs) -- 4 values per thread, grid-stride loop, Box-Muller pair generation (P0)
- [x] Scrambled Sobol sequences (scrambled_sobol.rs) -- Owen's scrambling for improved equidistribution without losing low-discrepancy property (P1)
- [x] Halton sequences (quasi/halton.rs) -- multi-dimensional quasi-random Halton sequences (P1)
- [x] Latin Hypercube sampling (quasi/latin_hypercube.rs) -- stratified space-filling design (P1)
- [x] Binomial distribution (distributions/binomial.rs) -- direct inversion + BTPE algorithm (P1)
- [x] Geometric distribution (distributions/geometric.rs) -- inverse CDF method (P1)
- [x] Multinomial distribution (distributions/multinomial.rs) -- conditional-binomial decomposition (P1)
- [x] Truncated normal (distributions/truncated_normal.rs) -- accept-reject Box-Muller for [a,b] interval (P1)
- [x] MRG32k3a skip-ahead (engines/mrg32k3a.rs) -- Efficient stream splitting for parallel Monte Carlo via matrix power skip-ahead (P1)
- [x] AES-CTR CSPRNG -- Cryptographically secure PRNG based on AES counter mode for security-sensitive applications (P2)
- [x] cuRAND host API compatibility -- Host-side API matching cuRAND's curandGenerator interface for drop-in usage (P2)
- [x] Random matrix generation -- Wishart matrices, random orthogonal matrices (QR of Gaussian) for statistical applications (P2)
- [x] Random graph generation -- Erdos-Renyi, stochastic block model, Barabasi-Albert, Watts-Strogatz, random regular (graph_gen.rs) (P2)
- [x] Monte Carlo methods (monte_carlo.rs) -- GPU-accelerated Monte Carlo integration, importance sampling, Markov chain Monte Carlo (MCMC) with Metropolis-Hastings and Hamiltonian MC (P1)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA driver API (runtime loading) | Yes |
| oxicuda-memory | GPU memory management | Yes |
| oxicuda-launch | Kernel launch infrastructure | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |
| half (optional) | FP16 support | Yes |

## Quality Status

- All production code uses Result/Option (no unwrap)
- clippy::all and missing_docs warnings enabled
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- All engines produce PTX via oxicuda-ptx (no nvcc required)

## Estimation vs Actual

| Metric | Estimated (Vol.5 rand) | Actual |
|--------|----------------------|--------|
| SLoC | 15K-24K | 9,064 |
| Files | ~10-15 | 27 |
| Coverage | Full cuRAND parity | Core engines + distributions |
| Ratio | -- | ~11.5% of estimate |

The rand crate has the highest actual-to-estimate ratio among Vol.5 crates, reflecting the relatively straightforward nature of RNG algorithms. P0/P1 items focus on throughput optimization and distribution coverage.

---

## Blueprint Quality Gates (Vol.5 Sec 9)

### Functional Requirements

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| S15 | Philox-4x32-10 uniform and normal distribution generation correctness | P0 | [x] Done |

### Performance Requirements

| # | Requirement | Target | Status |
|---|-------------|--------|--------|
| P7 | Philox, 100M uniform F32 samples | ≥ 95% cuRAND throughput | [ ] |

---

## Statistical Quality Requirements (Vol.5 Sec 9)

### PRNG Engine Quality (per BigCrush / NIST SP 800-22)

| Engine | Period | Statistical Tests | Status |
|--------|--------|------------------|--------|
| Philox-4x32-10 | 2¹²⁸ | BigCrush-passing per design | [x] |
| XORWOW | 2¹⁹⁹³⁷ | NIST SP 800-22 basic | [x] |
| MRG32k3a | 2¹⁹¹ | L'Ecuyer-verified | [x] |

### Required Statistical Tests

| Test | Description | Status |
|------|-------------|--------|
| Frequency (monobit) | Distribution of 0s and 1s | [x] |
| Block frequency | Frequency within M-bit blocks | [x] |
| Runs | Length of consecutive identical bits | [x] |
| Longest run | Longest run of 1s in a block | [x] |
| Chi-squared (uniform) | Uniform distribution goodness-of-fit | [x] |
| Kolmogorov-Smirnov (normal) | Normal distribution goodness-of-fit | [x] |

---

## Numerical Accuracy Requirements

| Distribution | FP32 Tolerance | Reference |
|-------------|---------------|-----------|
| Uniform [0, 1) | Bitwise exact to Philox spec | Philox-4x32-10 paper |
| Normal (Box-Muller) | KS test p-value > 0.01 | scipy.stats.kstest |
| Log-normal | KS test p-value > 0.01 | scipy.stats.kstest |
| Poisson | Chi-squared p-value > 0.01 | scipy.stats.chisquare |

---

## Deepening Opportunities

- [x] S15 correctness: Philox output matches reference implementation bit-for-bit (same seed → same sequence) — 5 tests added
- [ ] Performance P7: 100M F32 uniform samples, compare with cuRAND throughput
- [x] NIST SP 800-22 frequency, block frequency, runs, longest-run tests pass for Philox — implemented in statistical_tests.rs
- [x] Box-Muller normal distribution: KS test p-value > 0.01 for 1M samples — implemented; also box_muller_lcg_accuracy() verifies mean ±0.1 and std within 10% for LCG-based Box-Muller (3 tests)
- [x] Philox counter-mode independence verification with different seed/subsequence offsets — implemented; philox_counter_offset_independence() verifies that offset-N sequences give different results (5 tests including error handling)
- [x] Skip-ahead for MRG32k3a: verified correct for parallel reproducible simulation — skip_ahead(100) + 10 sequential values match; 4-stream pairwise independence verified (< 10% identical values)
