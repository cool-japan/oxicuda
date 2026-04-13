//! Split-radix butterfly PTX generation.
//!
//! The split-radix FFT decomposes a length-N DFT into one length-N/2 DFT
//! and two length-N/4 DFTs, requiring approximately 10% fewer multiplications
//! than a pure radix-2 decomposition.
//!
//! The L-shaped computation pattern is:
//! ```text
//!   X[k]       = E[k] + (W^k * O1[k] + W^{3k} * O3[k])
//!   X[k+N/2]   = E[k] - (W^k * O1[k] + W^{3k} * O3[k])
//!   X[k+N/4]   = E[k+N/4] - j*(W^k * O1[k] - W^{3k} * O3[k])
//!   X[k+3N/4]  = E[k+N/4] + j*(W^k * O1[k] - W^{3k} * O3[k])
//! ```
//!
//! where `E[k]` is the length-N/2 DFT of even-indexed elements,
//! `O1[k]` is the length-N/4 DFT of elements at indices 1 mod 4,
//! and `O3[k]` is the length-N/4 DFT of elements at indices 3 mod 4.
#![allow(dead_code)]

use oxicuda_ptx::builder::BodyBuilder;

use crate::error::{FftError, FftResult};
use crate::ptx_helpers::{
    ComplexRegs, complex_add, complex_mul, complex_mul_j, complex_mul_neg_j, complex_sub,
    load_twiddle_imm,
};
use crate::types::FftPrecision;

// ---------------------------------------------------------------------------
// Split-radix butterfly
// ---------------------------------------------------------------------------

/// Split-radix butterfly generator.
///
/// Decomposes an N-point DFT into one N/2-point DFT and two N/4-point DFTs,
/// achieving approximately 4N*log2(N) - 6N + 8 real multiplications, compared
/// to 5N*log2(N) for pure radix-2.
#[derive(Debug, Clone)]
pub struct SplitRadixButterfly {
    /// Total FFT size (must be >= 16 and a power of 2).
    fft_size: usize,
    /// Floating-point precision (f32 or f64).
    precision: FftPrecision,
}

impl SplitRadixButterfly {
    /// Creates a new split-radix butterfly for the given FFT size.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::InvalidSize`] if `fft_size` is not a power of 2,
    /// or is less than 16.
    pub fn new(fft_size: usize, precision: FftPrecision) -> FftResult<Self> {
        if !Self::should_use_split_radix(fft_size) {
            return Err(FftError::InvalidSize(format!(
                "split-radix requires N >= 16 and power of 2, got {fft_size}"
            )));
        }
        Ok(Self {
            fft_size,
            precision,
        })
    }

    /// Returns whether split-radix should be used for the given FFT size.
    ///
    /// Split-radix is beneficial when N >= 16 and N is a power of 2.
    /// For smaller sizes, radix-2 or radix-4 butterflies are more efficient
    /// due to lower overhead.
    pub fn should_use_split_radix(fft_size: usize) -> bool {
        fft_size >= 16 && fft_size.is_power_of_two()
    }

    /// Returns the FFT size.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Returns the precision.
    pub fn precision(&self) -> FftPrecision {
        self.precision
    }

    /// Emits the split-radix butterfly for a single group of 4 related outputs.
    ///
    /// Given:
    /// - `even_k`: E[k] from the N/2-point DFT of even-indexed elements
    /// - `even_k_quarter`: E[k + N/4] from the same N/2-point DFT
    /// - `odd1_k`: O1[k] from the N/4-point DFT of elements at indices 1 mod 4
    /// - `odd3_k`: O3[k] from the N/4-point DFT of elements at indices 3 mod 4
    /// - `k`: the frequency index within the quarter-length (0 <= k < N/4)
    /// - `direction_sign`: -1.0 for forward, +1.0 for inverse
    ///
    /// Produces outputs for X[k], X[k+N/4], X[k+N/2], X[k+3N/4].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_butterfly(
        &self,
        b: &mut BodyBuilder<'_>,
        even_k: &ComplexRegs,
        even_k_quarter: &ComplexRegs,
        odd1_k: &ComplexRegs,
        odd3_k: &ComplexRegs,
        k: u32,
        direction_sign: f64,
    ) -> FftResult<SplitRadixOutputs> {
        let n = self.fft_size;

        #[allow(clippy::cast_possible_truncation)]
        let n_u32 = n as u32;

        b.comment(&format!("split-radix butterfly k={k}, N={n}"));

        // Compute twiddle-multiplied odd terms:
        //   tw1 = W_N^k * O1[k]
        //   tw3 = W_N^{3k} * O3[k]
        let (tw1, tw3) = if k == 0 {
            // W^0 = 1, no multiplication needed
            (odd1_k.clone(), odd3_k.clone())
        } else {
            let w1 = load_twiddle_imm(b, self.precision, k, n_u32, direction_sign);
            let tw1 = complex_mul(b, self.precision, odd1_k, &w1);

            let w3 = load_twiddle_imm(b, self.precision, 3 * k, n_u32, direction_sign);
            let tw3 = complex_mul(b, self.precision, odd3_k, &w3);

            (tw1, tw3)
        };

        // sum_odd = tw1 + tw3  (used for X[k] and X[k+N/2])
        let sum_odd = complex_add(b, self.precision, &tw1, &tw3);

        // diff_odd = tw1 - tw3  (used with j-rotation for X[k+N/4] and X[k+3N/4])
        let diff_odd = complex_sub(b, self.precision, &tw1, &tw3);

        // X[k]     = E[k] + sum_odd
        let x_k = complex_add(b, self.precision, even_k, &sum_odd);

        // X[k+N/2] = E[k] - sum_odd
        let x_k_half = complex_sub(b, self.precision, even_k, &sum_odd);

        // For forward transform (sign = -1): multiply diff_odd by -j
        // For inverse transform (sign = +1): multiply diff_odd by +j
        let forward = direction_sign < 0.0;
        let j_diff = if forward {
            complex_mul_neg_j(b, self.precision, &diff_odd)
        } else {
            complex_mul_j(b, self.precision, &diff_odd)
        };

        // X[k+N/4]   = E[k+N/4] - j * diff_odd
        let x_k_quarter = complex_sub(b, self.precision, even_k_quarter, &j_diff);

        // X[k+3N/4]  = E[k+N/4] + j * diff_odd
        let x_k_three_quarter = complex_add(b, self.precision, even_k_quarter, &j_diff);

        Ok(SplitRadixOutputs {
            x_k,
            x_k_quarter,
            x_k_half,
            x_k_three_quarter,
        })
    }

    /// Emits a complete split-radix stage operating on the `data` slice.
    ///
    /// Processes N/4 butterfly groups, each producing 4 outputs according
    /// to the split-radix recurrence.
    ///
    /// `data` must have length == `self.fft_size`.
    /// The even-indexed sub-DFT and two odd sub-DFTs are assumed to already
    /// be in their respective positions within the data array.
    pub(crate) fn emit_stage(
        &self,
        b: &mut BodyBuilder<'_>,
        data: &mut [ComplexRegs],
        direction_sign: f64,
    ) -> FftResult<()> {
        let n = self.fft_size;
        if data.len() != n {
            return Err(FftError::InvalidSize(format!(
                "split-radix stage expects {n} elements, got {}",
                data.len()
            )));
        }

        let quarter = n / 4;
        let half = n / 2;

        b.comment(&format!("split-radix stage: N={n}"));

        for k in 0..quarter {
            let even_k = data[k].clone();
            let even_k_quarter = data[k + quarter].clone();
            let odd1_k = data[half + k].clone();
            let odd3_k = data[half + quarter + k].clone();

            #[allow(clippy::cast_possible_truncation)]
            let k_u32 = k as u32;

            let outputs = self.emit_butterfly(
                b,
                &even_k,
                &even_k_quarter,
                &odd1_k,
                &odd3_k,
                k_u32,
                direction_sign,
            )?;

            data[k] = outputs.x_k;
            data[k + quarter] = outputs.x_k_quarter;
            data[half + k] = outputs.x_k_half;
            data[half + quarter + k] = outputs.x_k_three_quarter;
        }

        Ok(())
    }
}

/// Outputs from a single split-radix butterfly.
#[derive(Debug, Clone)]
pub(crate) struct SplitRadixOutputs {
    /// X[k]
    pub(crate) x_k: ComplexRegs,
    /// X[k + N/4]
    pub(crate) x_k_quarter: ComplexRegs,
    /// X[k + N/2]
    pub(crate) x_k_half: ComplexRegs,
    /// X[k + 3N/4]
    pub(crate) x_k_three_quarter: ComplexRegs,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_use_split_radix_basic() {
        assert!(!SplitRadixButterfly::should_use_split_radix(0));
        assert!(!SplitRadixButterfly::should_use_split_radix(1));
        assert!(!SplitRadixButterfly::should_use_split_radix(4));
        assert!(!SplitRadixButterfly::should_use_split_radix(8));
        assert!(SplitRadixButterfly::should_use_split_radix(16));
        assert!(SplitRadixButterfly::should_use_split_radix(32));
        assert!(SplitRadixButterfly::should_use_split_radix(1024));
        assert!(SplitRadixButterfly::should_use_split_radix(4096));
    }

    #[test]
    fn should_use_split_radix_non_power_of_two() {
        assert!(!SplitRadixButterfly::should_use_split_radix(24));
        assert!(!SplitRadixButterfly::should_use_split_radix(48));
        assert!(!SplitRadixButterfly::should_use_split_radix(100));
    }

    #[test]
    fn new_rejects_small_sizes() {
        let result = SplitRadixButterfly::new(4, FftPrecision::Single);
        assert!(result.is_err());
        let result = SplitRadixButterfly::new(8, FftPrecision::Single);
        assert!(result.is_err());
    }

    #[test]
    fn new_rejects_non_power_of_two() {
        let result = SplitRadixButterfly::new(24, FftPrecision::Single);
        assert!(result.is_err());
    }

    #[test]
    fn new_accepts_valid_sizes() {
        let result = SplitRadixButterfly::new(16, FftPrecision::Single);
        assert!(result.is_ok());
        if let Ok(sr) = result {
            assert_eq!(sr.fft_size(), 16);
            assert_eq!(sr.precision(), FftPrecision::Single);
        }

        let result = SplitRadixButterfly::new(1024, FftPrecision::Double);
        assert!(result.is_ok());
    }

    #[test]
    fn stage_rejects_wrong_data_length() {
        use oxicuda_ptx::arch::SmVersion;
        use oxicuda_ptx::builder::KernelBuilder;
        use oxicuda_ptx::ir::PtxType;

        let sr = SplitRadixButterfly::new(16, FftPrecision::Single);
        assert!(sr.is_ok());
        if let Ok(sr) = sr {
            let result = KernelBuilder::new("test_sr_wrong_len")
                .target(SmVersion::Sm80)
                .param("dummy", PtxType::U32)
                .body(move |b| {
                    // Create 8 elements instead of 16
                    let mut data: Vec<ComplexRegs> = (0..8)
                        .map(|_| {
                            let re = b.alloc_reg(PtxType::F32);
                            let im = b.alloc_reg(PtxType::F32);
                            b.raw_ptx(&format!("mov.f32 {re}, 0f00000000;"));
                            b.raw_ptx(&format!("mov.f32 {im}, 0f00000000;"));
                            ComplexRegs { re, im }
                        })
                        .collect();

                    let stage_result = sr.emit_stage(b, &mut data, -1.0);
                    assert!(stage_result.is_err());
                    b.ret();
                })
                .build();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn butterfly_emits_ptx_for_n16() {
        use oxicuda_ptx::arch::SmVersion;
        use oxicuda_ptx::builder::KernelBuilder;
        use oxicuda_ptx::ir::PtxType;

        let sr = SplitRadixButterfly::new(16, FftPrecision::Single);
        assert!(sr.is_ok());
        if let Ok(sr) = sr {
            let ptx = KernelBuilder::new("test_sr_butterfly")
                .target(SmVersion::Sm80)
                .param("dummy", PtxType::U32)
                .body(move |b| {
                    let alloc_complex = |b: &mut BodyBuilder<'_>| {
                        let re = b.alloc_reg(PtxType::F32);
                        let im = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {re}, 0f3F800000;"));
                        b.raw_ptx(&format!("mov.f32 {im}, 0f00000000;"));
                        ComplexRegs { re, im }
                    };

                    let even_k = alloc_complex(b);
                    let even_kq = alloc_complex(b);
                    let odd1 = alloc_complex(b);
                    let odd3 = alloc_complex(b);

                    let result = sr.emit_butterfly(b, &even_k, &even_kq, &odd1, &odd3, 1, -1.0);
                    assert!(result.is_ok());
                    b.ret();
                })
                .build();

            assert!(ptx.is_ok());
            if let Ok(ptx_str) = ptx {
                // Should contain twiddle factor loading and complex arithmetic
                assert!(ptx_str.contains("split-radix butterfly"));
            }
        }
    }

    #[test]
    fn stage_emits_ptx_for_n16() {
        use oxicuda_ptx::arch::SmVersion;
        use oxicuda_ptx::builder::KernelBuilder;
        use oxicuda_ptx::ir::PtxType;

        let sr = SplitRadixButterfly::new(16, FftPrecision::Single);
        assert!(sr.is_ok());
        if let Ok(sr) = sr {
            let ptx = KernelBuilder::new("test_sr_stage")
                .target(SmVersion::Sm80)
                .param("dummy", PtxType::U32)
                .body(move |b| {
                    let mut data: Vec<ComplexRegs> = (0..16)
                        .map(|_| {
                            let re = b.alloc_reg(PtxType::F32);
                            let im = b.alloc_reg(PtxType::F32);
                            b.raw_ptx(&format!("mov.f32 {re}, 0f3F800000;"));
                            b.raw_ptx(&format!("mov.f32 {im}, 0f00000000;"));
                            ComplexRegs { re, im }
                        })
                        .collect();

                    let result = sr.emit_stage(b, &mut data, -1.0);
                    assert!(result.is_ok());
                    b.ret();
                })
                .build();
            assert!(ptx.is_ok());
            if let Ok(ptx_str) = ptx {
                assert!(ptx_str.contains("split-radix stage"));
            }
        }
    }

    #[test]
    fn double_precision_butterfly() {
        use oxicuda_ptx::arch::SmVersion;
        use oxicuda_ptx::builder::KernelBuilder;
        use oxicuda_ptx::ir::PtxType;

        let sr = SplitRadixButterfly::new(16, FftPrecision::Double);
        assert!(sr.is_ok());
        if let Ok(sr) = sr {
            let ptx = KernelBuilder::new("test_sr_f64")
                .target(SmVersion::Sm80)
                .param("dummy", PtxType::U32)
                .body(move |b| {
                    let alloc_complex = |b: &mut BodyBuilder<'_>| {
                        let re = b.alloc_reg(PtxType::F64);
                        let im = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mov.f64 {re}, 0d3FF0000000000000;"));
                        b.raw_ptx(&format!("mov.f64 {im}, 0d0000000000000000;"));
                        ComplexRegs { re, im }
                    };

                    let even_k = alloc_complex(b);
                    let even_kq = alloc_complex(b);
                    let odd1 = alloc_complex(b);
                    let odd3 = alloc_complex(b);

                    let result = sr.emit_butterfly(b, &even_k, &even_kq, &odd1, &odd3, 2, -1.0);
                    assert!(result.is_ok());
                    b.ret();
                })
                .build();
            assert!(ptx.is_ok());
        }
    }
}
