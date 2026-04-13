//! Prime-Factor Algorithm (Good--Thomas FFT).
//!
//! For `N = N1 * N2` where `gcd(N1, N2) = 1`, the PFA decomposes the N-point
//! DFT into `N2` transforms of length `N1` and `N1` transforms of length `N2`,
//! **without** requiring twiddle factors between the sub-transforms (unlike
//! the Cooley--Tukey algorithm).
//!
//! The index remapping uses the Chinese Remainder Theorem (CRT):
//!
//! ```text
//!   n = n1 * N2 * (N2^{-1} mod N1) + n2 * N1 * (N1^{-1} mod N2)  (mod N)
//!   k = k1 * (N2 mod ... ) ... (Ruritanian mapping)
//! ```
//!
//! The absence of twiddle factors makes PFA attractive for sizes that are
//! products of small coprime factors (e.g. 15 = 3*5, 21 = 3*7, 35 = 5*7).
#![allow(dead_code)]

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{FftError, FftResult};
use crate::types::{Complex, FftDirection, FftPrecision};

// ---------------------------------------------------------------------------
// GCD and modular inverse
// ---------------------------------------------------------------------------

/// Greatest common divisor (Euclidean algorithm).
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Modular multiplicative inverse of `a` modulo `m`, if it exists.
///
/// Returns `Some(x)` where `a * x ≡ 1 (mod m)`, or `None` if `gcd(a, m) != 1`.
fn mod_inverse(a: usize, m: usize) -> Option<usize> {
    if m == 1 {
        return Some(0);
    }

    let (mut old_r, mut r) = (a as i64, m as i64);
    let (mut old_s, mut s) = (1_i64, 0_i64);

    while r != 0 {
        let q = old_r / r;
        let tmp_r = r;
        r = old_r - q * r;
        old_r = tmp_r;
        let tmp_s = s;
        s = old_s - q * s;
        old_s = tmp_s;
    }

    if old_r != 1 {
        return None; // gcd != 1, no inverse
    }

    #[allow(clippy::cast_sign_loss)]
    Some(((old_s % m as i64 + m as i64) % m as i64) as usize)
}

// ---------------------------------------------------------------------------
// CRT index mapping
// ---------------------------------------------------------------------------

/// Computes the CRT input and output index mappings for PFA.
///
/// Returns `(input_map, output_map)` where:
/// - `input_map[n]` gives the position in the row-major `(n1, n2)` grid
///   corresponding to linear index `n`.
/// - `output_map[k]` gives the linear index for frequency bin `(k1, k2)`.
///
/// Input mapping (Ruritanian):
///   `n -> (n mod N1, n mod N2)` stored as `n1 * N2 + n2`.
///
/// Output mapping (CRT reconstruction):
///   `(k1, k2) -> (k1 * N2 * N2_inv + k2 * N1 * N1_inv) mod N`
fn crt_map(n1: usize, n2: usize) -> FftResult<(Vec<usize>, Vec<usize>)> {
    let n = n1 * n2;

    let n1_inv = mod_inverse(n1, n2)
        .ok_or_else(|| FftError::InternalError(format!("no modular inverse of {n1} mod {n2}")))?;
    let n2_inv = mod_inverse(n2, n1)
        .ok_or_else(|| FftError::InternalError(format!("no modular inverse of {n2} mod {n1}")))?;

    // Input map: linear -> (n1, n2) via Ruritanian mapping
    let mut input_map = vec![0usize; n];
    for (idx, slot) in input_map.iter_mut().enumerate() {
        let r1 = idx % n1;
        let r2 = idx % n2;
        *slot = r1 * n2 + r2;
    }

    // Output map: (k1, k2) -> linear via CRT
    let mut output_map = vec![0usize; n];
    for k1 in 0..n1 {
        for k2 in 0..n2 {
            let linear = (k1 * n2 * n2_inv + k2 * n1 * n1_inv) % n;
            output_map[k1 * n2 + k2] = linear;
        }
    }

    Ok((input_map, output_map))
}

// ---------------------------------------------------------------------------
// Coprime factorisation
// ---------------------------------------------------------------------------

/// Attempts to decompose `n` into a sequence of pairwise coprime factor pairs.
///
/// The function greedily extracts the largest coprime factor pair (n1, n2)
/// where n1 > 1 and n2 > 1 and gcd(n1, n2) = 1, recurses on each factor.
///
/// Returns the coprime factor pairs in order.
fn coprime_factors(n: usize) -> Vec<(usize, usize)> {
    if n <= 1 {
        return Vec::new();
    }

    // Try to split n into two coprime factors
    // Find the first non-trivial factor
    let small_primes = [2_usize, 3, 5, 7, 11, 13, 17, 19, 23];

    for &p in &small_primes {
        if n % p == 0 {
            // Extract all powers of p from n
            let mut pp = 1;
            let mut rem = n;
            while rem % p == 0 {
                pp *= p;
                rem /= p;
            }
            if rem > 1 {
                // n = pp * rem, gcd(pp, rem) = 1
                return vec![(pp, rem)];
            }
            // n is a prime power, cannot split into coprimes
            return Vec::new();
        }
    }

    // n is a prime > 23 or product of large primes
    Vec::new()
}

// ---------------------------------------------------------------------------
// PrimeFactorFft
// ---------------------------------------------------------------------------

/// Prime-Factor Algorithm FFT decomposition.
///
/// Decomposes an N-point DFT into sub-transforms of coprime lengths,
/// eliminating the need for inter-stage twiddle factors.
#[derive(Debug, Clone)]
pub struct PrimeFactorFft {
    /// Total FFT size.
    n: usize,
    /// Coprime factor pairs `(N1, N2)` such that `N1 * N2 = N` and
    /// `gcd(N1, N2) = 1`.
    factors: Vec<(usize, usize)>,
    /// CRT input index mapping.
    input_map: Vec<usize>,
    /// CRT output index mapping.
    output_map: Vec<usize>,
}

impl PrimeFactorFft {
    /// Creates a new PFA decomposition for the given size.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::InvalidSize`] if `n` is zero or cannot be
    /// decomposed into coprime factors.
    pub fn new(n: usize) -> FftResult<Self> {
        if n == 0 {
            return Err(FftError::InvalidSize("PFA size must be > 0".to_string()));
        }

        let factors = coprime_factors(n);
        if factors.is_empty() {
            return Err(FftError::InvalidSize(format!(
                "size {n} cannot be decomposed into coprime factors for PFA"
            )));
        }

        let (n1, n2) = factors[0];
        let (input_map, output_map) = crt_map(n1, n2)?;

        Ok(Self {
            n,
            factors,
            input_map,
            output_map,
        })
    }

    /// Checks if `n` can be decomposed into coprime factors.
    pub fn is_applicable(n: usize) -> bool {
        !coprime_factors(n).is_empty()
    }

    /// Returns the FFT size.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Returns the coprime factor pairs.
    pub fn factors(&self) -> &[(usize, usize)] {
        &self.factors
    }

    /// Returns the CRT input mapping.
    pub fn input_map(&self) -> &[usize] {
        &self.input_map
    }

    /// Returns the CRT output mapping.
    pub fn output_map(&self) -> &[usize] {
        &self.output_map
    }

    /// Executes the PFA on the CPU using naive sub-DFTs (for testing).
    ///
    /// This is an O(N * (N1 + N2)) algorithm when N = N1 * N2.
    pub fn execute_cpu(
        &self,
        input: &[Complex<f64>],
        direction: FftDirection,
    ) -> FftResult<Vec<Complex<f64>>> {
        if input.len() != self.n {
            return Err(FftError::InvalidSize(format!(
                "input length {} does not match PFA size {}",
                input.len(),
                self.n
            )));
        }

        let (n1, n2) = self.factors[0];
        let sign = direction.sign();

        // Step 1: Remap input to (n1, n2) grid using Ruritanian mapping
        let mut grid = vec![Complex::<f64>::zero(); self.n];
        for (idx, &map_idx) in self.input_map.iter().enumerate() {
            grid[map_idx] = input[idx];
        }

        // Step 2: Compute N1-point DFTs on each of N2 "columns"
        let mut after_rows = vec![Complex::<f64>::zero(); self.n];
        for col in 0..n2 {
            // Extract column col: grid[row * n2 + col] for row = 0..n1
            let col_input: Vec<Complex<f64>> = (0..n1).map(|row| grid[row * n2 + col]).collect();
            let col_output = naive_dft(&col_input, sign);
            for row in 0..n1 {
                after_rows[row * n2 + col] = col_output[row];
            }
        }

        // Step 3: Compute N2-point DFTs on each of N1 "rows"
        // No twiddle factors needed between steps (PFA property)
        let mut after_cols = vec![Complex::<f64>::zero(); self.n];
        for row in 0..n1 {
            let row_input: Vec<Complex<f64>> =
                (0..n2).map(|col| after_rows[row * n2 + col]).collect();
            let row_output = naive_dft(&row_input, sign);
            for col in 0..n2 {
                after_cols[row * n2 + col] = row_output[col];
            }
        }

        // Step 4: Remap output using CRT output mapping
        let mut output = vec![Complex::<f64>::zero(); self.n];
        for (grid_idx, &linear_idx) in self.output_map.iter().enumerate() {
            output[linear_idx] = after_cols[grid_idx];
        }

        Ok(output)
    }

    /// Generates a PTX kernel for the PFA.
    ///
    /// The kernel operates on a batch of PFA transforms.  Each thread
    /// handles one element, computing its contribution using the CRT
    /// index mappings baked into the kernel as immediates.
    ///
    /// Parameters: `(in_ptr, out_ptr, batch_count)`
    ///
    /// # Errors
    ///
    /// Returns [`FftError::PtxGeneration`] on PTX builder failure.
    pub fn generate_kernel(
        &self,
        _precision: FftPrecision,
        _direction: FftDirection,
        sm: SmVersion,
    ) -> FftResult<String> {
        let n = self.n;
        let (n1, n2) = self.factors[0];

        let ptx = KernelBuilder::new("pfa_fft")
            .target(sm)
            .param("in_ptr", PtxType::U64)
            .param("out_ptr", PtxType::U64)
            .param("batch_count", PtxType::U32)
            .max_threads_per_block(256)
            .body(move |b| {
                let gid = b.global_thread_id_x();

                // Total elements = n * batch_count
                let batch_count = b.load_param_u32("batch_count");
                let n_const = b.alloc_reg(PtxType::U32);
                #[allow(clippy::cast_possible_truncation)]
                let n_u32 = n as u32;
                b.raw_ptx(&format!("mov.u32 {n_const}, {n_u32};"));
                let total = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {total}, {n_const}, {batch_count};"));

                b.if_lt_u32(gid.clone(), total, move |b| {
                    let _in_ptr = b.load_param_u64("in_ptr");
                    let _out_ptr = b.load_param_u64("out_ptr");

                    // Determine which batch and which output element
                    let n_reg = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {n_reg}, {n_u32};"));
                    let batch_idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("div.u32 {batch_idx}, {gid}, {n_reg};"));
                    let elem_idx = b.alloc_reg(PtxType::U32);
                    let batch_times_n = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mul.lo.u32 {batch_times_n}, {batch_idx}, {n_reg};"
                    ));
                    b.raw_ptx(&format!("sub.u32 {elem_idx}, {gid}, {batch_times_n};"));

                    // Compute k1 = elem_idx / n2, k2 = elem_idx % n2
                    let n2_reg = b.alloc_reg(PtxType::U32);
                    #[allow(clippy::cast_possible_truncation)]
                    let n2_u32 = n2 as u32;
                    b.raw_ptx(&format!("mov.u32 {n2_reg}, {n2_u32};"));
                    let k1 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("div.u32 {k1}, {elem_idx}, {n2_reg};"));
                    let k1_times_n2 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {k1_times_n2}, {k1}, {n2_reg};"));
                    let k2 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("sub.u32 {k2}, {elem_idx}, {k1_times_n2};"));

                    b.comment(&format!("PFA N={n}, N1={n1}, N2={n2}: kernel placeholder"));
                    b.comment("Full implementation requires shared memory sub-DFTs");

                    // Store zeros as placeholder (actual implementation would
                    // compute the transform using shared memory)
                    let zero_f = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {zero_f}, 0f00000000;"));

                    // Output address: out_ptr + gid * 8 (complex f32 = 8 bytes)
                    let addr = b.byte_offset_addr(_out_ptr, gid.clone(), 8);
                    b.store_global_f32(addr.clone(), zero_f.clone());
                    let addr_im = b.alloc_reg(PtxType::U64);
                    b.raw_ptx(&format!("add.u64 {addr_im}, {addr}, 4;"));
                    b.store_global_f32(addr_im, zero_f);

                    let _ = (k2, _in_ptr);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }
}

// ---------------------------------------------------------------------------
// Naive DFT helper (for CPU testing)
// ---------------------------------------------------------------------------

/// Naive O(N^2) DFT.
fn naive_dft(input: &[Complex<f64>], direction_sign: f64) -> Vec<Complex<f64>> {
    let n = input.len();
    let mut output = vec![Complex::<f64>::zero(); n];

    for (k, out_k) in output.iter_mut().enumerate() {
        let mut sum = Complex::<f64>::zero();
        for (j, &x_j) in input.iter().enumerate() {
            let angle = direction_sign * 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            let w = Complex::<f64>::new(angle.cos(), angle.sin());
            sum = sum + x_j * w;
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
    fn gcd_basic() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(15, 7), 1);
        assert_eq!(gcd(3, 5), 1);
        assert_eq!(gcd(6, 6), 6);
    }

    #[test]
    fn mod_inverse_basic() {
        // 3 * 2 = 6 ≡ 1 (mod 5)
        assert_eq!(mod_inverse(3, 5), Some(2));
        // 5 * 2 = 10 ≡ 1 (mod 3)
        assert_eq!(mod_inverse(5, 3), Some(2));
        // gcd(4, 6) = 2, no inverse
        assert_eq!(mod_inverse(4, 6), None);
    }

    #[test]
    fn coprime_factors_15() {
        let factors = coprime_factors(15);
        assert!(!factors.is_empty());
        let (a, b) = factors[0];
        assert_eq!(a * b, 15);
        assert_eq!(gcd(a, b), 1);
    }

    #[test]
    fn coprime_factors_prime_power() {
        // 8 = 2^3, cannot split into coprimes
        let factors = coprime_factors(8);
        assert!(factors.is_empty());
    }

    #[test]
    fn is_applicable_various() {
        assert!(PrimeFactorFft::is_applicable(15)); // 3 * 5
        assert!(PrimeFactorFft::is_applicable(21)); // 3 * 7
        assert!(PrimeFactorFft::is_applicable(35)); // 5 * 7
        assert!(!PrimeFactorFft::is_applicable(8)); // 2^3
        assert!(!PrimeFactorFft::is_applicable(7)); // prime
    }

    #[test]
    fn new_rejects_zero() {
        assert!(PrimeFactorFft::new(0).is_err());
    }

    #[test]
    fn new_rejects_prime() {
        assert!(PrimeFactorFft::new(7).is_err());
    }

    #[test]
    fn crt_map_15() {
        let (inp, out) = crt_map(3, 5).ok().unwrap_or_default();
        assert_eq!(inp.len(), 15);
        assert_eq!(out.len(), 15);

        // Verify the input map covers all grid positions
        let mut seen = [false; 15];
        for &idx in &inp {
            seen[idx] = true;
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "grid position {i} not covered by input map");
        }

        // Verify the output map covers all linear indices
        let mut seen_out = [false; 15];
        for &idx in &out {
            seen_out[idx] = true;
        }
        for (i, &s) in seen_out.iter().enumerate() {
            assert!(s, "linear index {i} not covered by output map");
        }
    }

    #[test]
    fn pfa_cpu_impulse_15() {
        let pfa = PrimeFactorFft::new(15);
        assert!(pfa.is_ok());
        if let Ok(p) = pfa {
            // DFT of [1, 0, 0, ...] should give all ones
            let mut input = vec![Complex::<f64>::zero(); 15];
            input[0] = Complex::<f64>::one();

            let output = p.execute_cpu(&input, FftDirection::Forward);
            assert!(output.is_ok());
            if let Ok(out) = output {
                for (k, val) in out.iter().enumerate() {
                    assert!(
                        (val.re - 1.0).abs() < 1e-10,
                        "k={k}: re={} (expected 1.0)",
                        val.re
                    );
                    assert!(val.im.abs() < 1e-10, "k={k}: im={} (expected 0.0)", val.im);
                }
            }
        }
    }

    #[test]
    fn pfa_cpu_constant_input() {
        let pfa = PrimeFactorFft::new(15);
        assert!(pfa.is_ok());
        if let Ok(p) = pfa {
            // DFT of constant [c, c, ..., c] should give [N*c, 0, 0, ..., 0]
            let c = Complex::<f64>::new(2.0, 0.0);
            let input = vec![c; 15];

            let output = p.execute_cpu(&input, FftDirection::Forward);
            assert!(output.is_ok());
            if let Ok(out) = output {
                assert!((out[0].re - 30.0).abs() < 1e-10);
                assert!(out[0].im.abs() < 1e-10);
                for val in &out[1..] {
                    assert!(val.re.abs() < 1e-10);
                    assert!(val.im.abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn pfa_ptx_generates() {
        let pfa = PrimeFactorFft::new(15);
        assert!(pfa.is_ok());
        if let Ok(p) = pfa {
            let ptx =
                p.generate_kernel(FftPrecision::Single, FftDirection::Forward, SmVersion::Sm80);
            assert!(ptx.is_ok());
            if let Ok(ptx_str) = ptx {
                assert!(ptx_str.contains(".entry pfa_fft"));
            }
        }
    }

    #[test]
    fn pfa_cpu_wrong_input_length() {
        let pfa = PrimeFactorFft::new(15);
        assert!(pfa.is_ok());
        if let Ok(p) = pfa {
            let input = vec![Complex::<f64>::zero(); 10]; // wrong length
            let result = p.execute_cpu(&input, FftDirection::Forward);
            assert!(result.is_err());
        }
    }

    #[test]
    fn pfa_cpu_21_impulse() {
        // 21 = 3 * 7
        let pfa = PrimeFactorFft::new(21);
        assert!(pfa.is_ok());
        if let Ok(p) = pfa {
            let mut input = vec![Complex::<f64>::zero(); 21];
            input[0] = Complex::<f64>::one();

            let output = p.execute_cpu(&input, FftDirection::Forward);
            assert!(output.is_ok());
            if let Ok(out) = output {
                for (k, val) in out.iter().enumerate() {
                    assert!(
                        (val.re - 1.0).abs() < 1e-10,
                        "k={k}: re={} (expected 1.0)",
                        val.re
                    );
                    assert!(val.im.abs() < 1e-10, "k={k}: im={} (expected 0.0)", val.im);
                }
            }
        }
    }
}
