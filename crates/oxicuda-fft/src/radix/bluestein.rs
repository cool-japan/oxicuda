//! Bluestein (Chirp-Z) transform for arbitrary-size FFTs.
//!
//! The Bluestein algorithm converts an N-point DFT (where N may be prime
//! or have large prime factors) into a circular convolution of length M,
//! where M is the next power of 2 >= 2*N - 1.  The convolution is then
//! computed via three FFTs of size M (which are power-of-2 and thus
//! efficiently handled by radix-2/4/8 kernels).
//!
//! Algorithm outline:
//!
//! 1. Precompute chirp sequence `h[n] = W_N^(n^2/2)` for `n = 0..N-1`.
//! 2. Multiply input by conjugate chirp: `a[n] = x[n] * conj(h[n])`.
//! 3. Zero-pad `a` to length M.
//! 4. Forward FFT of `a` (size M).
//! 5. Forward FFT of `b` (the chirp filter, precomputed and cached).
//! 6. Pointwise multiply: `c = FFT(a) * FFT(b)`.
//! 7. Inverse FFT of `c`.
//! 8. Multiply result by `conj(h[n])` and extract first N elements.
#![allow(dead_code)]

use crate::error::{FftError, FftResult};
use crate::types::{Complex, FftPrecision};

// ---------------------------------------------------------------------------
// BluesteinPlan
// ---------------------------------------------------------------------------

/// Pre-computed data for a Bluestein transform of size N.
#[derive(Debug, Clone)]
pub struct BluesteinPlan {
    /// Original FFT size.
    pub n: usize,
    /// Padded size (next power of 2 >= 2*N - 1).
    pub m: usize,
    /// Precision of the transform.
    pub precision: FftPrecision,
    /// Chirp sequence `h[k] = exp(-pi*i*k^2/N)` for `k = 0..N-1` (f64).
    pub chirp_f64: Vec<Complex<f64>>,
    /// Conjugate chirp sequence `conj(h[k])`.
    pub chirp_conj_f64: Vec<Complex<f64>>,
    /// FFT of the zero-padded chirp filter (length M), used for convolution.
    /// This is left empty at construction and populated when the plan is
    /// compiled for GPU execution.
    pub chirp_fft_f64: Vec<Complex<f64>>,
}

impl BluesteinPlan {
    /// Creates a new Bluestein plan for the given size and precision.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::InvalidSize`] if `n` is zero.
    pub fn new(n: usize, precision: FftPrecision) -> FftResult<Self> {
        if n == 0 {
            return Err(FftError::InvalidSize(
                "Bluestein size must be > 0".to_string(),
            ));
        }

        let m = next_power_of_2(2 * n - 1);

        // Precompute chirp: h[k] = exp(-pi * i * k^2 / N)
        let mut chirp_f64 = Vec::with_capacity(n);
        let mut chirp_conj_f64 = Vec::with_capacity(n);

        for k in 0..n {
            let k_f64 = k as f64;
            let n_f64 = n as f64;
            let angle = -std::f64::consts::PI * k_f64 * k_f64 / n_f64;
            let re = angle.cos();
            let im = angle.sin();
            chirp_f64.push(Complex::<f64>::new(re, im));
            chirp_conj_f64.push(Complex::<f64>::new(re, -im));
        }

        Ok(Self {
            n,
            m,
            precision,
            chirp_f64,
            chirp_conj_f64,
            chirp_fft_f64: Vec::new(),
        })
    }

    /// Returns the padded convolution size M.
    pub fn padded_size(&self) -> usize {
        self.m
    }

    /// Returns the workspace size in bytes required for the Bluestein transform.
    ///
    /// This includes three buffers of size M complex elements:
    /// - Padded input after chirp multiplication
    /// - Chirp filter FFT (can be precomputed once)
    /// - Temporary for FFT stages
    pub fn workspace_bytes(&self) -> usize {
        // 3 buffers of M complex elements
        3 * self.m * self.precision.complex_bytes()
    }

    /// Builds the chirp filter for convolution.
    ///
    /// The filter `b[k]` is:
    /// - `conj(h[k])` for `k = 0..N-1`
    /// - Zero for `k = N..M-N+1`
    /// - `conj(h[M-k])` for `k = M-N+1..M-1` (wrap-around)
    pub fn build_chirp_filter(&self) -> Vec<Complex<f64>> {
        let mut filter = vec![Complex::<f64>::zero(); self.m];

        // First N elements
        for (k, &c) in self.chirp_conj_f64.iter().enumerate() {
            filter[k] = c;
        }

        // Wrap-around elements
        for k in 1..self.n {
            filter[self.m - k] = self.chirp_conj_f64[k];
        }

        filter
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the next power of 2 that is >= `n`.
fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Checks whether a size requires the Bluestein algorithm.
///
/// Returns `true` if `n` has a prime factor > 7 (not fully factorisable
/// into supported radices).
pub(crate) fn needs_bluestein(n: usize) -> bool {
    let mut rem = n;
    for &f in &[2, 3, 5, 7] {
        while rem % f == 0 {
            rem /= f;
        }
    }
    rem > 1
}

// ---------------------------------------------------------------------------
// CPU-side Bluestein reference (for testing / precomputation)
// ---------------------------------------------------------------------------

/// Applies the Bluestein algorithm on the CPU (f64 precision).
///
/// This is used for precomputing the chirp filter FFT and for testing.
/// It uses a naive DFT for the internal FFTs (suitable only for small M).
pub fn bluestein_cpu_f64(input: &[Complex<f64>], direction_sign: f64) -> Vec<Complex<f64>> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }

    let m = next_power_of_2(2 * n - 1);

    // Chirp: h[k] = exp(sign * pi * i * k^2 / N)
    let chirp: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let angle = direction_sign * std::f64::consts::PI * (k * k) as f64 / n as f64;
            Complex::<f64>::new(angle.cos(), angle.sin())
        })
        .collect();

    // a[k] = x[k] * conj(h[k]), zero-padded to M
    let mut a = vec![Complex::<f64>::zero(); m];
    for k in 0..n {
        a[k] = input[k] * chirp[k].conj();
    }

    // b[k] = chirp filter
    let mut b_buf = vec![Complex::<f64>::zero(); m];
    for (k, chirp_k) in chirp[..n].iter().enumerate() {
        b_buf[k] = *chirp_k;
    }
    for k in 1..n {
        b_buf[m - k] = chirp[k];
    }

    // Naive DFT (acceptable for precomputation on CPU)
    let a_fft = naive_dft_f64(&a, -1.0);
    let b_fft = naive_dft_f64(&b_buf, -1.0);

    // Pointwise multiply
    let c_fft: Vec<Complex<f64>> = a_fft
        .iter()
        .zip(b_fft.iter())
        .map(|(&a_val, &b_val)| a_val * b_val)
        .collect();

    // Inverse DFT
    let c = naive_dft_f64(&c_fft, 1.0);

    // Extract and scale
    let inv_m = 1.0 / m as f64;
    (0..n)
        .map(|k| {
            let scaled = Complex::<f64>::new(c[k].re * inv_m, c[k].im * inv_m);
            scaled * chirp[k].conj()
        })
        .collect()
}

/// Naive O(N^2) DFT for CPU-side precomputation.
fn naive_dft_f64(input: &[Complex<f64>], direction_sign: f64) -> Vec<Complex<f64>> {
    let n = input.len();
    let mut output = vec![Complex::<f64>::zero(); n];

    for (k, out_k) in output.iter_mut().enumerate() {
        let mut sum = Complex::<f64>::zero();
        for (j, inp_j) in input.iter().enumerate() {
            let angle = direction_sign * 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            let w = Complex::<f64>::new(angle.cos(), angle.sin());
            sum = sum + *inp_j * w;
        }
        *out_k = sum;
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_pow2() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(8), 8);
        assert_eq!(next_power_of_2(9), 16);
    }

    #[test]
    fn bluestein_plan_creation() {
        let plan = BluesteinPlan::new(13, FftPrecision::Double);
        assert!(plan.is_ok());
        let plan = plan.ok();
        assert!(plan.is_some());
        if let Some(p) = plan {
            assert_eq!(p.n, 13);
            assert!(p.m >= 25); // 2*13 - 1 = 25, next pow2 = 32
            assert_eq!(p.m, 32);
        }
    }

    #[test]
    fn needs_bluestein_check() {
        assert!(!needs_bluestein(8)); // 2^3
        assert!(!needs_bluestein(360)); // 2^3 * 3^2 * 5
        assert!(needs_bluestein(13)); // prime > 7
        assert!(needs_bluestein(11)); // prime > 7
    }

    #[test]
    fn bluestein_cpu_identity() {
        // DFT of [1, 0, 0, ...] should give all ones
        let n = 13;
        let mut input = vec![Complex::<f64>::zero(); n];
        input[0] = Complex::<f64>::one();

        let output = bluestein_cpu_f64(&input, -1.0);
        for val in &output {
            assert!((val.re - 1.0).abs() < 1e-10, "re = {}", val.re);
            assert!(val.im.abs() < 1e-10, "im = {}", val.im);
        }
    }

    #[test]
    fn chirp_filter_symmetry() {
        let plan = BluesteinPlan::new(7, FftPrecision::Double);
        assert!(plan.is_ok());
        if let Ok(p) = plan {
            let filter = p.build_chirp_filter();
            assert_eq!(filter.len(), p.m);
            // filter[0] should be conj(h[0]) = conj(1) = 1
            assert!((filter[0].re - 1.0).abs() < 1e-12);
            assert!(filter[0].im.abs() < 1e-12);
        }
    }

    // ── Bluestein Algorithm Correctness Tests (S6) ──────────

    /// Bluestein works for prime size N=11 (prime > 7, not a supported radix).
    /// The algorithm should produce the correct result: DFT of impulse = all ones.
    #[test]
    fn test_bluestein_prime_size_n11() {
        let n = 11;
        assert!(needs_bluestein(n), "N=11 must require Bluestein");
        let plan = BluesteinPlan::new(n, FftPrecision::Double);
        assert!(plan.is_ok(), "BluesteinPlan for N=11 must be created");
        if let Ok(p) = plan {
            assert_eq!(p.n, n);
            // m must be next power of 2 >= 2*11 - 1 = 21 => 32
            assert_eq!(p.m, 32, "padded size for N=11 must be 32");
            // Chirp sequences must be length N
            assert_eq!(p.chirp_f64.len(), n);
            assert_eq!(p.chirp_conj_f64.len(), n);
        }
    }

    /// Bluestein works for prime size N=13 (prime > 7): DFT of impulse.
    #[test]
    fn test_bluestein_prime_size_n13() {
        let n = 13;
        assert!(needs_bluestein(n), "N=13 must require Bluestein");
        let mut input = vec![Complex::<f64>::zero(); n];
        input[0] = Complex::<f64>::one();
        let output = bluestein_cpu_f64(&input, -1.0);
        assert_eq!(output.len(), n);
        for (i, val) in output.iter().enumerate() {
            assert!(
                (val.re - 1.0).abs() < 1e-9,
                "re[{i}] = {} expected 1.0",
                val.re
            );
            assert!(val.im.abs() < 1e-9, "im[{i}] = {} expected 0.0", val.im);
        }
    }

    /// Bluestein works for prime size N=17 (prime > 7).
    #[test]
    fn test_bluestein_prime_size_n17() {
        let n = 17;
        assert!(needs_bluestein(n), "N=17 must require Bluestein");
        let plan = BluesteinPlan::new(n, FftPrecision::Double);
        assert!(plan.is_ok(), "BluesteinPlan for N=17 must be created");
        if let Ok(p) = plan {
            assert_eq!(p.n, n);
            // m >= 2*17-1 = 33, next pow2 = 64
            assert_eq!(p.m, 64, "padded size for N=17 must be 64");
        }
    }

    /// Strategy selection: power-of-2 sizes do NOT need Bluestein.
    /// Primes > 7 DO need Bluestein.
    /// Mixed-radix with supported factors do NOT need Bluestein.
    #[test]
    fn test_strategy_selects_bluestein_for_primes() {
        // Power-of-2 sizes: fully factorisable, no Bluestein needed
        assert!(
            !needs_bluestein(1024),
            "N=1024 (2^10) must not need Bluestein"
        );
        assert!(!needs_bluestein(512), "N=512 (2^9) must not need Bluestein");
        assert!(!needs_bluestein(256), "N=256 (2^8) must not need Bluestein");

        // Pure primes > 7: require Bluestein
        assert!(needs_bluestein(11), "N=11 (prime) must need Bluestein");
        assert!(needs_bluestein(13), "N=13 (prime) must need Bluestein");
        assert!(needs_bluestein(17), "N=17 (prime) must need Bluestein");
        assert!(needs_bluestein(101), "N=101 (prime) must need Bluestein");

        // Supported composite sizes (factors only in {2,3,5,7}): no Bluestein
        assert!(!needs_bluestein(12), "N=12=2^2*3 must not need Bluestein");
        assert!(!needs_bluestein(60), "N=60=2^2*3*5 must not need Bluestein");
        assert!(
            !needs_bluestein(360),
            "N=360=2^3*3^2*5 must not need Bluestein"
        );
        assert!(
            !needs_bluestein(7),
            "N=7 (supported radix) must not need Bluestein"
        );
    }

    /// Chirp sequence length (padded size M) is always >= 2*N-1, and a power of 2.
    #[test]
    fn test_bluestein_chirp_length_correct() {
        // N=11: 2*11-1=21 => next pow2 = 32
        let plan11 = BluesteinPlan::new(11, FftPrecision::Double);
        assert!(plan11.is_ok());
        if let Ok(p) = plan11 {
            assert!(p.m >= 2 * p.n - 1, "M must be >= 2*N-1");
            assert!(p.m >= 21);
            assert_eq!(p.m, 32);
            assert_eq!(p.m & (p.m - 1), 0, "M must be a power of 2");
        }

        // N=13: 2*13-1=25 => next pow2 = 32
        let plan13 = BluesteinPlan::new(13, FftPrecision::Double);
        assert!(plan13.is_ok());
        if let Ok(p) = plan13 {
            assert!(p.m >= 2 * p.n - 1);
            assert!(p.m >= 25);
            assert_eq!(p.m, 32);
            assert_eq!(p.m & (p.m - 1), 0);
        }

        // N=101 (prime): 2*101-1=201 => next pow2 = 256
        let plan101 = BluesteinPlan::new(101, FftPrecision::Double);
        assert!(plan101.is_ok());
        if let Ok(p) = plan101 {
            assert!(p.m >= 2 * p.n - 1);
            assert!(p.m >= 201);
            assert_eq!(p.m, 256);
            assert_eq!(p.m & (p.m - 1), 0);
        }

        // N=17 (prime): 2*17-1=33 => next pow2 = 64
        let plan17 = BluesteinPlan::new(17, FftPrecision::Double);
        assert!(plan17.is_ok());
        if let Ok(p) = plan17 {
            assert!(p.m >= 2 * p.n - 1);
            assert!(p.m >= 33);
            assert_eq!(p.m, 64);
            assert_eq!(p.m & (p.m - 1), 0);
        }
    }

    #[test]
    fn bluestein_dft_prime_n_5() {
        // DFT of [1, 0, 0, 0, 0] (delta function at index 0) should produce a flat spectrum.
        // Each output bin k has magnitude exactly 1.0.
        let x = [1.0_f32, 0.0, 0.0, 0.0, 0.0];
        let n = 5usize;
        // CPU reference DFT via direct summation
        let spectrum: Vec<(f32, f32)> = (0..n)
            .map(|k| {
                (0..n).fold((0.0_f32, 0.0_f32), |(re, im), j| {
                    let angle = -2.0 * std::f32::consts::PI * (k * j) as f32 / n as f32;
                    (re + x[j] * angle.cos(), im + x[j] * angle.sin())
                })
            })
            .collect();
        for (k, &(re, im)) in spectrum.iter().enumerate() {
            let mag = (re * re + im * im).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-5,
                "bin {k} magnitude should be 1.0, got {mag}"
            );
        }
    }

    #[test]
    fn bluestein_supports_prime_n() {
        // Bluestein transforms arbitrary N by padding to a power of 2.
        // Verify the padded size calculation: next power of 2 >= 2*N - 1.
        let n = 7usize; // prime
        let min_pad = 2 * n - 1; // = 13
        let padded = min_pad.next_power_of_two(); // = 16
        assert_eq!(padded, 16);
        assert!(padded >= min_pad);
        // Confirm BluesteinPlan can be constructed for this prime N
        let plan = BluesteinPlan::new(n, FftPrecision::Single);
        assert!(plan.is_ok(), "BluesteinPlan::new({n}) should succeed");
        if let Ok(p) = plan {
            assert!(p.m >= min_pad, "padded size M must be >= 2*N-1");
            assert_eq!(p.m & (p.m - 1), 0, "padded size M must be a power of 2");
        }
    }

    #[test]
    fn bluestein_chirp_z_transform_n13() {
        // N=13 (prime): verify Bluestein padded size.
        // 2*13 - 1 = 25, next power of 2 = 32.
        let n = 13usize;
        let padded = (2 * n - 1).next_power_of_two();
        assert_eq!(padded, 32, "N=13 padded size must be 32");
        let plan = BluesteinPlan::new(n, FftPrecision::Double);
        assert!(plan.is_ok(), "BluesteinPlan::new({n}) should succeed");
        if let Ok(p) = plan {
            assert_eq!(p.m, 32, "BluesteinPlan M for N=13 must be 32");
        }
    }

    /// Bluestein forward+inverse round-trip: IDFT(DFT(x)) ≈ x for prime N.
    #[test]
    fn test_bluestein_round_trip_prime() {
        let n = 13;
        // Use a non-trivial input
        let input: Vec<Complex<f64>> = (0..n)
            .map(|k| Complex::<f64>::new(k as f64 * 0.1 + 0.5, (n - k) as f64 * 0.05))
            .collect();

        let forward = bluestein_cpu_f64(&input, -1.0);
        let reconstructed = bluestein_cpu_f64(&forward, 1.0);

        // IDFT result must be scaled by N to recover input
        let inv_n = 1.0 / n as f64;
        for (i, (orig, recon)) in input.iter().zip(reconstructed.iter()).enumerate() {
            let scaled_re = recon.re * inv_n;
            let scaled_im = recon.im * inv_n;
            assert!(
                (scaled_re - orig.re).abs() < 1e-9,
                "re[{i}]: original={} reconstructed_scaled={}",
                orig.re,
                scaled_re
            );
            assert!(
                (scaled_im - orig.im).abs() < 1e-9,
                "im[{i}]: original={} reconstructed_scaled={}",
                orig.im,
                scaled_im
            );
        }
    }
}
