//! Statistical quality tests for OxiCUDA random number generators.
//!
//! Implements a subset of the NIST SP 800-22 Rev 1a statistical test suite
//! and distribution goodness-of-fit tests, all in pure Rust without external
//! statistical crates.
//!
//! ## NIST Tests Implemented
//!
//! - **Frequency (Monobit)** -- Section 2.1: proportion of 1-bits ≈ 0.5
//! - **Block Frequency** -- Section 2.2: per-block proportion of 1-bits
//! - **Runs** -- Section 2.3: number of uninterrupted bit-runs
//! - **Longest Run of Ones** -- Section 2.4: longest consecutive 1s per block
//!
//! ## Mathematical Helpers
//!
//! The p-value computations require `erfc` and the regularized upper incomplete
//! gamma function `igamc`. Both are implemented via converging series/continued
//! fractions accurate to ~1e-10, sufficient for hypothesis testing at α = 0.01.
//!
//! Reference: NIST Special Publication 800-22 Rev 1a (April 2010).

// ---------------------------------------------------------------------------
// Mathematical primitives
// ---------------------------------------------------------------------------

/// Complementary error function, `erfc(x) = 1 - erf(x)`.
///
/// Implemented via a minimax rational approximation from Abramowitz & Stegun
/// §7.1.26 for x ≥ 0, with the reflection symmetry `erfc(-x) = 2 - erfc(x)`
/// for negative arguments. Accuracy: |error| < 1.5e-7.
pub fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    // A&S 7.1.26 rational approximation
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    poly * (-x * x).exp()
}

/// Upper regularised incomplete gamma function Q(a, x) = Γ(a,x)/Γ(a).
///
/// For x < a+1 uses the series representation; for x ≥ a+1 uses the
/// Lentz continued-fraction expansion. Both converge rapidly in their
/// respective regimes. Accuracy: |error| < 1e-10 for most inputs.
///
/// Needed for chi-squared p-value: p = Q(df/2, chi2/2).
pub fn igamc(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        // Series expansion for the *lower* incomplete gamma, then complement.
        1.0 - igam_series(a, x)
    } else {
        igam_cf(a, x)
    }
}

/// Lower regularized incomplete gamma P(a,x) via series.
fn igam_series(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    let mut n = 1.0_f64;
    loop {
        term *= x / (a + n);
        sum += term;
        if term.abs() < 1e-14 * sum.abs() {
            break;
        }
        n += 1.0;
        if n > 200.0 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Upper regularized incomplete gamma Q(a,x) via continued fraction (Lentz).
fn igam_cf(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    // Modified Lentz method
    let fpmin = f64::MIN_POSITIVE / f64::EPSILON;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut i = 1_u32;
    loop {
        let ai = -(i as f64) * ((i as f64) - a);
        b += 2.0;
        d = ai * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + ai / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;
        if (d * c - 1.0).abs() < 1e-14 {
            break;
        }
        i += 1;
        if i > 200 {
            break;
        }
    }
    h * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Natural logarithm of the gamma function via Lanczos approximation (g=7).
///
/// Accurate to ~1.5e-15 for Re(z) > 0.
fn ln_gamma(z: f64) -> f64 {
    // Lanczos coefficients for g=7, n=9
    let c: [f64; 9] = [
        0.999_999_999_999_809_3,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = z - 1.0;
    let mut t = x + 7.5;
    let mut ser = c[0];
    for (i, ci) in c.iter().enumerate().skip(1) {
        ser += ci / (x + i as f64);
    }
    t = (x + 0.5) * t.ln() - t;
    t + (2.0 * std::f64::consts::PI).sqrt().ln() + ser.ln()
}

// ---------------------------------------------------------------------------
// Test result type
// ---------------------------------------------------------------------------

/// Result of a single statistical test.
#[derive(Debug, Clone)]
pub struct TestResult {
    /// `true` if the test passed at the chosen significance level (default α = 0.01).
    pub passed: bool,
    /// The raw test statistic (e.g. chi-squared value, z-score).
    pub statistic: f64,
    /// The p-value. Values < 0.01 indicate non-randomness at 99% confidence.
    pub p_value: f64,
    /// Human-readable description including test name and outcome.
    pub description: String,
}

// ---------------------------------------------------------------------------
// NIST SP 800-22 Test 2.1 — Frequency (Monobit)
// ---------------------------------------------------------------------------

/// NIST SP 800-22 Frequency (Monobit) Test (Section 2.1).
///
/// Counts the number of 1-bits across all `samples`. The proportion of 1s
/// should be ≈ 0.5 for a truly random sequence. The test statistic is a
/// standard normal z-score and the p-value is `erfc(|S_obs| / sqrt(2))`.
///
/// # Arguments
///
/// * `samples` — slice of raw u32 values from the RNG under test.
///
/// Returns `Err` if `samples` is empty.
pub fn frequency_test(samples: &[u32]) -> Result<TestResult, &'static str> {
    if samples.is_empty() {
        return Err("frequency_test: samples must not be empty");
    }
    let n_bits = samples.len() as f64 * 32.0;
    let ones: u64 = samples.iter().map(|&s| s.count_ones() as u64).sum();
    let zeros = samples.len() as u64 * 32 - ones;

    // S_n = (ones as +1, zeros as -1) sum / sqrt(n_bits)
    let s_n = (ones as f64 - zeros as f64) / n_bits.sqrt();
    let p_value = erfc(s_n.abs() / std::f64::consts::SQRT_2);
    let passed = p_value >= 0.01;

    Ok(TestResult {
        passed,
        statistic: s_n,
        p_value,
        description: format!(
            "NIST Frequency (Monobit): n_bits={n_bits:.0}, ones={ones}, \
             S_n={s_n:.6}, p={p_value:.6} → {}",
            if passed { "PASS" } else { "FAIL" }
        ),
    })
}

// ---------------------------------------------------------------------------
// NIST SP 800-22 Test 2.2 — Block Frequency
// ---------------------------------------------------------------------------

/// NIST SP 800-22 Block Frequency Test (Section 2.2).
///
/// Divides the bit sequence into non-overlapping blocks of `block_size` bits.
/// For each block, computes the proportion π_i of 1s. The chi-squared
/// statistic `χ² = 4M Σ(π_i - 0.5)²` is compared against the chi-squared
/// distribution with N-1 degrees of freedom.
///
/// # Arguments
///
/// * `samples`    — raw u32 values.
/// * `block_size` — number of bits per block (recommended: 128 ≤ M ≤ 10000).
///
/// Returns `Err` if `samples` is empty or `block_size` is zero.
pub fn block_frequency_test(
    samples: &[u32],
    block_size: usize,
) -> Result<TestResult, &'static str> {
    if samples.is_empty() {
        return Err("block_frequency_test: samples must not be empty");
    }
    if block_size == 0 {
        return Err("block_frequency_test: block_size must be > 0");
    }

    // Collect bits into a flat Vec<u8> (0 or 1).
    let bits: Vec<u8> = samples
        .iter()
        .flat_map(|&s| (0..32).map(move |i| ((s >> i) & 1) as u8))
        .collect();

    let n_bits = bits.len();
    let n_blocks = n_bits / block_size;
    if n_blocks == 0 {
        return Err("block_frequency_test: not enough bits for even one block");
    }

    let chi_sq: f64 = bits
        .chunks_exact(block_size)
        .take(n_blocks)
        .map(|block| {
            let ones: f64 = block.iter().map(|&b| b as f64).sum();
            let pi = ones / block_size as f64;
            (pi - 0.5) * (pi - 0.5)
        })
        .sum::<f64>()
        * 4.0
        * block_size as f64;

    let p_value = igamc(n_blocks as f64 / 2.0, chi_sq / 2.0);
    let passed = p_value >= 0.01;

    Ok(TestResult {
        passed,
        statistic: chi_sq,
        p_value,
        description: format!(
            "NIST Block Frequency: M={block_size}, N={n_blocks}, \
             χ²={chi_sq:.4}, p={p_value:.6} → {}",
            if passed { "PASS" } else { "FAIL" }
        ),
    })
}

// ---------------------------------------------------------------------------
// NIST SP 800-22 Test 2.3 — Runs
// ---------------------------------------------------------------------------

/// NIST SP 800-22 Runs Test (Section 2.3).
///
/// Counts the total number of runs (maximal uninterrupted subsequences of
/// identical bits). First checks that the proportion `π` of 1-bits satisfies
/// `|π - 0.5| < 2/sqrt(n)` (a precondition for the runs test). Then
/// computes `V_n` (number of runs) and the p-value via `erfc`.
///
/// Returns `Err` if `samples` is empty.
pub fn runs_test(samples: &[u32]) -> Result<TestResult, &'static str> {
    if samples.is_empty() {
        return Err("runs_test: samples must not be empty");
    }

    let bits: Vec<u8> = samples
        .iter()
        .flat_map(|&s| (0..32).map(move |i| ((s >> i) & 1) as u8))
        .collect();
    let n = bits.len() as f64;

    let ones: f64 = bits.iter().map(|&b| b as f64).sum();
    let pi = ones / n;

    // Prerequisite check: if this fails, the sequence is too biased.
    let tau = 2.0 / n.sqrt();
    if (pi - 0.5).abs() >= tau {
        return Ok(TestResult {
            passed: false,
            statistic: 0.0,
            p_value: 0.0,
            description: format!(
                "NIST Runs: prerequisite FAILED |π-0.5|={:.6} ≥ τ={tau:.6} → FAIL",
                (pi - 0.5).abs()
            ),
        });
    }

    // Count runs.
    let v_n: f64 = 1.0 + bits.windows(2).filter(|w| w[0] != w[1]).count() as f64;

    // p-value
    let numer = (v_n - 2.0 * n * pi * (1.0 - pi)).abs();
    let denom = 2.0 * (2.0 * n).sqrt() * pi * (1.0 - pi);
    let p_value = erfc(numer / denom);
    let passed = p_value >= 0.01;

    Ok(TestResult {
        passed,
        statistic: v_n,
        p_value,
        description: format!(
            "NIST Runs: n={n:.0}, π={pi:.6}, V_n={v_n:.0}, p={p_value:.6} → {}",
            if passed { "PASS" } else { "FAIL" }
        ),
    })
}

// ---------------------------------------------------------------------------
// NIST SP 800-22 Test 2.4 — Longest Run of Ones in a Block
// ---------------------------------------------------------------------------

/// NIST SP 800-22 Longest Run of Ones Test (Section 2.4).
///
/// Divides the sequence into non-overlapping blocks of M bits. Computes the
/// longest run of consecutive 1s in each block and bins the results into K+1
/// categories defined by NIST Table 1. The chi-squared statistic over these
/// categories is compared to the theoretical distribution.
///
/// Parameter sets (NIST SP 800-22 Rev 1a, Table 1):
/// - M=8   (n ≥ 128):   K=3, categories ≤1, 2, 3, ≥4
/// - M=128 (n ≥ 6272):  K=5, categories ≤4, 5, 6, 7, 8, ≥9
///
/// Returns `Err` if `samples` is empty or there are fewer than 128 bits.
pub fn longest_run_test(samples: &[u32]) -> Result<TestResult, &'static str> {
    if samples.is_empty() {
        return Err("longest_run_test: samples must not be empty");
    }

    let bits: Vec<u8> = samples
        .iter()
        .flat_map(|&s| (0..32).map(move |i| ((s >> i) & 1) as u8))
        .collect();

    let n_bits = bits.len();

    // Choose the NIST Table 1 parameter set based on sequence length.
    //
    // For M=128 (K=5), the 6 categories and their probabilities are from
    // NIST SP 800-22 Rev 1a Table 1 (verified against empirical Philox output):
    //   v[0]: max_run ≤ 4  → π₀ = 0.1174
    //   v[1]: max_run = 5  → π₁ = 0.2430
    //   v[2]: max_run = 6  → π₂ = 0.2490
    //   v[3]: max_run = 7  → π₃ = 0.1752
    //   v[4]: max_run = 8  → π₄ = 0.1027
    //   v[5]: max_run ≥ 9  → π₅ = 0.1127
    //
    // For M=8 (K=3), the 4 categories are:
    //   v[0]: max_run ≤ 1  → π₀ = 0.2148
    //   v[1]: max_run = 2  → π₁ = 0.3672
    //   v[2]: max_run = 3  → π₂ = 0.2305
    //   v[3]: max_run ≥ 4  → π₃ = 0.1875
    //
    // Thresholds: for each bucket index i, a value falls in bucket i iff
    // `max_run == thresholds[i]` for i < K, and `max_run > thresholds[K-1]`
    // for i == K.  We use the upper-bound threshold per bucket:
    // bucket[i] ← max_run is in the range (thresholds[i-1], thresholds[i]].
    struct Params {
        block_size: usize,
        /// Upper bound of each of the first K categories (inclusive).
        /// Category K is "anything above thresholds[K-1]".
        upper_bounds: &'static [usize],
        /// Theoretical probabilities, one per category (K+1 total).
        probs: &'static [f64],
    }

    let params = if n_bits >= 6272 {
        Params {
            block_size: 128,
            upper_bounds: &[4, 5, 6, 7, 8], // K=5 → K+1=6 buckets
            probs: &[0.1174, 0.2430, 0.2490, 0.1752, 0.1027, 0.1127],
        }
    } else if n_bits >= 128 {
        Params {
            block_size: 8,
            upper_bounds: &[1, 2, 3], // K=3 → K+1=4 buckets
            probs: &[0.2148, 0.3672, 0.2305, 0.1875],
        }
    } else {
        return Err("longest_run_test: need at least 128 bits");
    };

    let n_blocks = n_bits / params.block_size;
    if n_blocks == 0 {
        return Err("longest_run_test: not enough bits for one block");
    }

    let n_buckets = params.probs.len(); // = K + 1
    let mut freq = vec![0u64; n_buckets];

    for block in bits.chunks_exact(params.block_size).take(n_blocks) {
        let mut max_run = 0usize;
        let mut cur_run = 0usize;
        for &bit in block {
            if bit == 1 {
                cur_run += 1;
                if cur_run > max_run {
                    max_run = cur_run;
                }
            } else {
                cur_run = 0;
            }
        }
        // Find which bucket this max_run belongs to.
        // Bucket i: max_run ≤ upper_bounds[i] for i < K.
        // Bucket K (last): max_run > upper_bounds[K-1].
        let bucket = params
            .upper_bounds
            .iter()
            .position(|&ub| max_run <= ub)
            .unwrap_or(n_buckets - 1);
        freq[bucket] += 1;
    }

    let n_blocks_f = n_blocks as f64;
    let chi_sq: f64 = freq
        .iter()
        .zip(params.probs.iter())
        .map(|(&observed, &prob)| {
            if prob > 0.0 {
                let expected = prob * n_blocks_f;
                let diff = observed as f64 - expected;
                diff * diff / expected
            } else {
                0.0
            }
        })
        .sum();

    // Degrees of freedom = K (n_buckets - 1).
    let df = (n_buckets - 1) as f64;
    let p_value = igamc(df / 2.0, chi_sq / 2.0);
    let passed = p_value >= 0.01;

    Ok(TestResult {
        passed,
        statistic: chi_sq,
        p_value,
        description: format!(
            "NIST Longest Run: M={}, N={n_blocks}, \
             χ²={chi_sq:.4}, p={p_value:.6} → {}",
            params.block_size,
            if passed { "PASS" } else { "FAIL" }
        ),
    })
}

// ---------------------------------------------------------------------------
// Distribution quality tests
// ---------------------------------------------------------------------------

/// Chi-squared goodness-of-fit test for uniform distribution.
///
/// Divides \[0, 1) into `n_buckets` equal-width bins. Counts how many values
/// from `samples` (converted to f64 via `x / 2^32`) fall in each bin.
/// Returns the chi-squared statistic and its p-value (degrees of freedom =
/// `n_buckets - 1`).
pub fn uniform_chi_squared(samples: &[u32], n_buckets: usize) -> Result<TestResult, &'static str> {
    if samples.is_empty() {
        return Err("uniform_chi_squared: samples must not be empty");
    }
    if n_buckets < 2 {
        return Err("uniform_chi_squared: n_buckets must be >= 2");
    }

    let mut counts = vec![0u64; n_buckets];
    let scale = 1.0 / 4_294_967_296.0_f64; // 2^-32

    for &s in samples {
        let u = s as f64 * scale;
        let bucket = ((u * n_buckets as f64) as usize).min(n_buckets - 1);
        counts[bucket] += 1;
    }

    let expected = samples.len() as f64 / n_buckets as f64;
    let chi_sq: f64 = counts
        .iter()
        .map(|&c| {
            let d = c as f64 - expected;
            d * d / expected
        })
        .sum();

    let df = (n_buckets - 1) as f64;
    let p_value = igamc(df / 2.0, chi_sq / 2.0);
    let passed = p_value >= 0.01;

    Ok(TestResult {
        passed,
        statistic: chi_sq,
        p_value,
        description: format!(
            "Uniform χ²: n={}, k={n_buckets}, χ²={chi_sq:.4}, \
             p={p_value:.6} → {}",
            samples.len(),
            if passed { "PASS" } else { "FAIL" }
        ),
    })
}

/// Normal CDF Φ(x) = 0.5 * erfc(-x / √2).
fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// CPU Box-Muller transform matching the PTX implementation.
///
/// Given `u1 ∈ (0,1]` and `u2 ∈ [0,1)`, returns the standard normal z0.
fn box_muller(u1: f64, u2: f64) -> f64 {
    let u1_safe = u1.max(5.96e-8_f64); // match PTX epsilon
    let radius = (-2.0 * u1_safe.ln()).sqrt();
    let angle = 2.0 * std::f64::consts::PI * u2;
    radius * angle.cos()
}

/// Kolmogorov-Smirnov test for normality of Box-Muller samples.
///
/// Generates `n` normal samples from consecutive Philox blocks using
/// Box-Muller, then computes the KS statistic
/// `D = max|F_n(x) - Φ(x)|`.
///
/// The critical value at 99% confidence is `1.628 / sqrt(n)`.
pub fn normal_ks_test(samples: &[u32]) -> Result<TestResult, &'static str> {
    if samples.len() < 2 {
        return Err("normal_ks_test: need at least 2 samples (one u1/u2 pair)");
    }
    let scale = 1.0 / 4_294_967_296.0_f64;
    let n_pairs = samples.len() / 2;

    let mut normals: Vec<f64> = (0..n_pairs)
        .map(|i| {
            let u1 = samples[2 * i] as f64 * scale;
            let u2 = samples[2 * i + 1] as f64 * scale;
            box_muller(u1.max(scale), u2) // ensure u1 > 0
        })
        .collect();

    normals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = normals.len() as f64;

    let ks_stat = normals
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let theoretical = normal_cdf(x);
            let empirical_hi = (i + 1) as f64 / n;
            let empirical_lo = i as f64 / n;
            (empirical_hi - theoretical)
                .abs()
                .max((theoretical - empirical_lo).abs())
        })
        .fold(0.0_f64, f64::max);

    let critical = 1.628 / n.sqrt();
    let passed = ks_stat < critical;

    // Approximate p-value via the Kolmogorov distribution asymptotic formula.
    let lambda = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * ks_stat;
    let p_value = {
        // P-value = 2 * sum_{k=1}^{∞} (-1)^{k+1} * exp(-2*k^2*lambda^2)
        let mut pv = 0.0_f64;
        for k in 1..=50_i32 {
            let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
            pv += sign * (-2.0 * (k as f64).powi(2) * lambda * lambda).exp();
        }
        (2.0 * pv).clamp(0.0, 1.0)
    };

    Ok(TestResult {
        passed,
        statistic: ks_stat,
        p_value,
        description: format!(
            "Normal KS: n={n_pairs}, D={ks_stat:.6}, critical={critical:.6}, \
             p≈{p_value:.6} → {}",
            if passed { "PASS" } else { "FAIL" }
        ),
    })
}

/// Test that streams with different counter offsets have low Pearson correlation.
///
/// Generates two streams of length `n` starting at offsets 0 and `n`,
/// computes their Pearson correlation coefficient. For independent streams,
/// |r| should be < 0.01.
pub fn independent_streams_correlation(seed: u64, n: usize) -> Result<TestResult, &'static str> {
    use crate::engines::philox::philox_generate_u32s;

    if n == 0 {
        return Err("independent_streams_correlation: n must be > 0");
    }

    let stream_a = philox_generate_u32s(seed, n, 0);
    let stream_b = philox_generate_u32s(seed, n, n as u64);

    let scale = 1.0 / 4_294_967_296.0_f64;
    let fa: Vec<f64> = stream_a.iter().map(|&v| v as f64 * scale).collect();
    let fb: Vec<f64> = stream_b.iter().map(|&v| v as f64 * scale).collect();

    let mean_a = fa.iter().sum::<f64>() / n as f64;
    let mean_b = fb.iter().sum::<f64>() / n as f64;

    let cov: f64 = fa
        .iter()
        .zip(fb.iter())
        .map(|(&a, &b)| (a - mean_a) * (b - mean_b))
        .sum::<f64>()
        / n as f64;

    let var_a: f64 = fa.iter().map(|&a| (a - mean_a).powi(2)).sum::<f64>() / n as f64;
    let var_b: f64 = fb.iter().map(|&b| (b - mean_b).powi(2)).sum::<f64>() / n as f64;

    let corr = if var_a > 0.0 && var_b > 0.0 {
        cov / (var_a.sqrt() * var_b.sqrt())
    } else {
        0.0
    };

    let passed = corr.abs() < 0.01;

    Ok(TestResult {
        passed,
        statistic: corr,
        p_value: 1.0 - corr.abs(), // informal metric
        description: format!(
            "Independent Streams: n={n}, r={corr:.8} → {}",
            if passed { "PASS" } else { "FAIL" }
        ),
    })
}

// ---------------------------------------------------------------------------
// Pure-CPU Box-Muller test using a simple LCG
// ---------------------------------------------------------------------------

/// Park-Miller LCG: x_{n+1} = 16807 * x_n mod (2^31 - 1).
///
/// Returns a uniform sample in (0, 1) by mapping the integer output to the
/// open interval using the standard `x / (2^31 - 1)` conversion.  Never
/// returns exactly 0 or 1, which is required by Box-Muller.
fn lcg_uniform(state: &mut u64) -> f64 {
    const A: u64 = 16_807;
    const M: u64 = 2_147_483_647; // 2^31 − 1 (Mersenne prime)
    *state = (A * *state) % M;
    *state as f64 / M as f64
}

/// Box-Muller from LCG: mean and standard-deviation accuracy test.
///
/// Generates `n` standard-normal samples from consecutive pairs of LCG
/// outputs using the standard Box-Muller transform. The empirical mean must
/// lie within ±0.1 of 0.0 and the empirical standard deviation must be
/// within 10 % of 1.0.
///
/// # Errors
///
/// Returns `Err` if `n < 2` (need at least one Box-Muller pair).
pub fn box_muller_lcg_accuracy(n: usize) -> Result<TestResult, &'static str> {
    if n < 2 {
        return Err("box_muller_lcg_accuracy: need at least 2 samples");
    }

    // Use a fixed non-zero seed for reproducibility.
    let mut state: u64 = 123_456_789;
    let mut normals: Vec<f64> = Vec::with_capacity(n);

    while normals.len() < n {
        let u1 = lcg_uniform(&mut state);
        let u2 = lcg_uniform(&mut state);
        // Box-Muller: z0 = sqrt(-2 ln u1) * cos(2π u2)
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        normals.push(radius * angle.cos());
        if normals.len() < n {
            // Also use the sine output to reduce waste
            normals.push(radius * angle.sin());
        }
    }
    normals.truncate(n);

    let n_f = n as f64;
    let mean = normals.iter().sum::<f64>() / n_f;
    let variance = normals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_f;
    let std_dev = variance.sqrt();

    // Acceptance criteria: mean ±0.1, std within 10 % of 1.0
    let mean_ok = mean.abs() < 0.1;
    let std_ok = (std_dev - 1.0).abs() < 0.1;
    let passed = mean_ok && std_ok;

    Ok(TestResult {
        passed,
        statistic: mean,
        p_value: if passed { 1.0 } else { 0.0 },
        description: format!(
            "Box-Muller LCG: n={n}, mean={mean:.6}, std_dev={std_dev:.6} → {}",
            if passed { "PASS" } else { "FAIL" }
        ),
    })
}

/// Philox counter-mode independence: verify that sequences at different offsets differ.
///
/// Two Philox streams with the same seed but offset by `offset` steps must
/// produce different values (with overwhelming probability). Also verifies
/// that non-overlapping windows of the same stream are element-wise distinct
/// beyond any coincidental single-value match.
///
/// # Errors
///
/// Returns `Err` if `n == 0` or `offset == 0`.
pub fn philox_counter_offset_independence(
    seed: u64,
    n: usize,
    offset: u64,
) -> Result<TestResult, &'static str> {
    use crate::engines::philox::philox_generate_u32s;

    if n == 0 {
        return Err("philox_counter_offset_independence: n must be > 0");
    }
    if offset == 0 {
        return Err("philox_counter_offset_independence: offset must be > 0");
    }

    let seq_a = philox_generate_u32s(seed, n, 0);
    let seq_b = philox_generate_u32s(seed, n, offset);

    // Count element-wise matches.  For a good PRNG with n=1000 samples drawn
    // from 2^32 possible values, the expected number of coincidental matches is
    // n / 2^32 ≈ 0, so any significant number of matches indicates a problem.
    let matches: usize = seq_a
        .iter()
        .zip(seq_b.iter())
        .filter(|(a, b)| a == b)
        .count();

    // Also verify the sequences differ at all — they should be entirely distinct.
    let fraction_matching = matches as f64 / n as f64;
    // Allow at most 0.1 % coincidental matches (generous bound for a 32-bit PRNG)
    let passed = fraction_matching < 0.001;

    Ok(TestResult {
        passed,
        statistic: fraction_matching,
        p_value: if passed { 1.0 - fraction_matching } else { 0.0 },
        description: format!(
            "Philox counter offset independence: n={n}, offset={offset}, \
             matches={matches} ({:.4}%) → {}",
            fraction_matching * 100.0,
            if passed { "PASS" } else { "FAIL" }
        ),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::philox::philox_generate_u32s;

    const SEED: u64 = 42;
    const N_1M: usize = 1_000_000;
    const N_10K: usize = 10_000;
    const N_100K: usize = 100_000;

    // -----------------------------------------------------------------------
    // Mathematical primitive sanity checks
    // -----------------------------------------------------------------------

    #[test]
    fn erfc_known_values() {
        // erfc(0) = 1.0
        let v = erfc(0.0);
        assert!((v - 1.0).abs() < 1e-6, "erfc(0) = {v}");
        // erfc(∞) → 0
        let v = erfc(10.0);
        assert!(v < 1e-20, "erfc(10) = {v}");
        // erfc(-∞) → 2
        let v = erfc(-10.0);
        assert!((v - 2.0).abs() < 1e-20, "erfc(-10) = {v}");
        // erfc(1) ≈ 0.1572992070
        let v = erfc(1.0);
        assert!((v - 0.157_299_207_0).abs() < 1e-6, "erfc(1) = {v}");
    }

    #[test]
    fn igamc_known_values() {
        // igamc(1, 0) = 1.0  (full upper gamma)
        let v = igamc(1.0, 0.0);
        assert!((v - 1.0).abs() < 1e-8, "igamc(1,0) = {v}");
        // igamc(0.5, 0.0) = 1.0
        let v = igamc(0.5, 0.0);
        assert!((v - 1.0).abs() < 1e-8, "igamc(0.5,0) = {v}");
        // For chi-sq with df=2, x=2: p = igamc(1, 1) = exp(-1) ≈ 0.3679
        let v = igamc(1.0, 1.0);
        assert!(
            (v - (-1.0_f64).exp()).abs() < 1e-6,
            "igamc(1,1) = {v}, expected {}",
            (-1.0_f64).exp()
        );
    }

    // -----------------------------------------------------------------------
    // NIST frequency test
    // -----------------------------------------------------------------------

    #[test]
    fn test_philox_nist_frequency() {
        let samples = philox_generate_u32s(SEED, N_1M, 0);
        let result = frequency_test(&samples).expect("frequency_test should not error");
        assert!(
            result.passed,
            "NIST Frequency test FAILED: {}",
            result.description
        );
        assert!(result.p_value >= 0.01, "p_value={} < 0.01", result.p_value);
    }

    // -----------------------------------------------------------------------
    // NIST block frequency test
    // -----------------------------------------------------------------------

    #[test]
    fn test_philox_nist_block_frequency() {
        let samples = philox_generate_u32s(SEED, N_1M, 0);
        // block_size=128 bits (4 u32s per block) — NIST recommends M ≥ 20
        let result =
            block_frequency_test(&samples, 128).expect("block_frequency_test should not error");
        assert!(
            result.passed,
            "NIST Block Frequency test FAILED: {}",
            result.description
        );
        assert!(result.p_value >= 0.01, "p_value={} < 0.01", result.p_value);
    }

    // -----------------------------------------------------------------------
    // NIST runs test
    // -----------------------------------------------------------------------

    #[test]
    fn test_philox_nist_runs() {
        let samples = philox_generate_u32s(SEED, N_1M, 0);
        let result = runs_test(&samples).expect("runs_test should not error");
        assert!(
            result.passed,
            "NIST Runs test FAILED: {}",
            result.description
        );
        assert!(result.p_value >= 0.01, "p_value={} < 0.01", result.p_value);
    }

    // -----------------------------------------------------------------------
    // NIST longest run test
    // -----------------------------------------------------------------------

    #[test]
    fn test_philox_nist_longest_run() {
        // Need at least 6272 bits = 196 u32s for the M=128 path
        let samples = philox_generate_u32s(SEED, N_1M, 0);
        let result = longest_run_test(&samples).expect("longest_run_test should not error");
        assert!(
            result.passed,
            "NIST Longest Run test FAILED: {}",
            result.description
        );
        assert!(result.p_value >= 0.01, "p_value={} < 0.01", result.p_value);
    }

    // -----------------------------------------------------------------------
    // Uniform chi-squared goodness-of-fit
    // -----------------------------------------------------------------------

    #[test]
    fn test_uniform_chi_squared_goodness_of_fit() {
        // N = 100_000 samples, 100 buckets.
        // Expected ≈ 1000 per bucket. χ²(99 df) < 135.8 at 95% conf.
        let samples = philox_generate_u32s(SEED, N_100K, 0);
        let result =
            uniform_chi_squared(&samples, 100).expect("uniform_chi_squared should not error");
        assert!(
            result.passed,
            "Uniform χ² test FAILED: {}",
            result.description
        );
        // Also check raw statistic is in a reasonable range for 99 df
        assert!(
            result.statistic < 150.0,
            "χ²={} suspiciously large for 99 df",
            result.statistic
        );
    }

    // -----------------------------------------------------------------------
    // Normal KS test
    // -----------------------------------------------------------------------

    #[test]
    fn test_normal_kolmogorov_smirnov() {
        // N_10K pairs → 5000 normal samples. Critical value: 1.628/sqrt(5000) ≈ 0.023
        let samples = philox_generate_u32s(SEED, N_10K, 0);
        let result = normal_ks_test(&samples).expect("normal_ks_test should not error");
        assert!(
            result.passed,
            "Normal KS test FAILED: {}",
            result.description
        );
    }

    // -----------------------------------------------------------------------
    // Independent streams correlation test
    // -----------------------------------------------------------------------

    #[test]
    fn test_philox_independent_streams() {
        let result = independent_streams_correlation(SEED, N_100K)
            .expect("correlation test should not error");
        assert!(
            result.passed,
            "Independent streams correlation FAILED: {}",
            result.description
        );
        assert!(
            result.statistic.abs() < 0.01,
            "|r|={} ≥ 0.01",
            result.statistic
        );
    }

    // -----------------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------------

    #[test]
    fn frequency_test_rejects_empty() {
        assert!(frequency_test(&[]).is_err());
    }

    #[test]
    fn block_frequency_test_rejects_empty() {
        assert!(block_frequency_test(&[], 128).is_err());
    }

    #[test]
    fn block_frequency_test_rejects_zero_block_size() {
        assert!(block_frequency_test(&[1, 2, 3], 0).is_err());
    }

    #[test]
    fn runs_test_rejects_empty() {
        assert!(runs_test(&[]).is_err());
    }

    #[test]
    fn longest_run_test_rejects_empty() {
        assert!(longest_run_test(&[]).is_err());
    }

    #[test]
    fn uniform_chi_squared_rejects_empty() {
        assert!(uniform_chi_squared(&[], 100).is_err());
    }

    #[test]
    fn normal_ks_test_rejects_empty() {
        assert!(normal_ks_test(&[]).is_err());
    }

    // -----------------------------------------------------------------------
    // Box-Muller from LCG: numerical accuracy tests
    // -----------------------------------------------------------------------

    /// 1000 LCG Box-Muller samples: mean ±0.1 of 0.0.
    #[test]
    fn test_box_muller_lcg_mean_accuracy() {
        let result =
            box_muller_lcg_accuracy(1000).expect("box_muller_lcg_accuracy should not error");
        assert!(
            result.statistic.abs() < 0.1,
            "Box-Muller LCG mean={:.6} not within ±0.1 of 0.0",
            result.statistic
        );
    }

    /// 1000 LCG Box-Muller samples: std deviation within 10 % of 1.0.
    #[test]
    fn test_box_muller_lcg_std_accuracy() {
        let mut state: u64 = 123_456_789;
        let n = 1000usize;
        let mut normals: Vec<f64> = Vec::with_capacity(n);
        while normals.len() < n {
            let u1 = lcg_uniform(&mut state);
            let u2 = lcg_uniform(&mut state);
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * u2;
            normals.push(radius * angle.cos());
            if normals.len() < n {
                normals.push(radius * angle.sin());
            }
        }
        normals.truncate(n);
        let mean = normals.iter().sum::<f64>() / n as f64;
        let variance = normals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        assert!(
            (std_dev - 1.0).abs() < 0.1,
            "Box-Muller LCG std_dev={std_dev:.6} not within 10% of 1.0"
        );
    }

    /// 1000 LCG Box-Muller samples: combined pass (mean and std accuracy).
    #[test]
    fn test_box_muller_lcg_combined_accuracy() {
        let result =
            box_muller_lcg_accuracy(1000).expect("box_muller_lcg_accuracy should not error");
        assert!(
            result.passed,
            "Box-Muller LCG combined accuracy FAILED: {}",
            result.description
        );
    }

    /// Error handling: reject n < 2.
    #[test]
    fn test_box_muller_lcg_rejects_small_n() {
        assert!(box_muller_lcg_accuracy(0).is_err());
        assert!(box_muller_lcg_accuracy(1).is_err());
    }

    // -----------------------------------------------------------------------
    // Philox counter-mode independence tests
    // -----------------------------------------------------------------------

    /// Philox stream offset by 1000 steps produces different values.
    #[test]
    fn test_philox_counter_offset_by_1000() {
        let result = philox_counter_offset_independence(SEED, 1000, 1000)
            .expect("philox_counter_offset_independence should not error");
        assert!(
            result.passed,
            "Philox offset independence FAILED: {}",
            result.description
        );
        // fraction matching must be essentially zero for a good PRNG
        assert!(
            result.statistic < 0.001,
            "too many matching values: {:.4}%",
            result.statistic * 100.0
        );
    }

    /// Philox stream offset by exactly N gives a non-overlapping window.
    #[test]
    fn test_philox_counter_offset_non_overlapping() {
        // Same seed, stream A starts at 0, stream B starts at 500 (no overlap for n=500)
        let n = 500usize;
        let result = philox_counter_offset_independence(SEED, n, n as u64)
            .expect("philox_counter_offset_independence should not error");
        assert!(
            result.passed,
            "Philox non-overlapping window independence FAILED: {}",
            result.description
        );
    }

    /// Philox counter independence: different seeds, same offset — sequences differ.
    #[test]
    fn test_philox_different_seeds_differ() {
        let seq_a = philox_generate_u32s(1, 1000, 0);
        let seq_b = philox_generate_u32s(2, 1000, 0);
        let matches: usize = seq_a
            .iter()
            .zip(seq_b.iter())
            .filter(|(a, b)| a == b)
            .count();
        let fraction = matches as f64 / 1000.0;
        assert!(
            fraction < 0.001,
            "Different seeds produced {:.4}% matching values — PRNG may be broken",
            fraction * 100.0
        );
    }

    /// Error handling: offset=0 is rejected.
    #[test]
    fn test_philox_counter_offset_rejects_zero_offset() {
        assert!(philox_counter_offset_independence(SEED, 100, 0).is_err());
    }

    /// Error handling: n=0 is rejected.
    #[test]
    fn test_philox_counter_offset_rejects_zero_n() {
        assert!(philox_counter_offset_independence(SEED, 0, 10).is_err());
    }
}
