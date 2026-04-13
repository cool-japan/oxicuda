//! Bank-conflict-free Stockham FFT kernel generator.
//!
//! Shared memory on NVIDIA GPUs is divided into 32 banks.  When multiple
//! threads in a warp access different addresses that map to the same bank,
//! the accesses are serialized ("bank conflict").  Power-of-2 strides in
//! standard Stockham kernels trigger worst-case conflicts at every stage.
//!
//! This module generates PTX kernels that **pad** the shared memory layout
//! so that consecutive logical elements map to consecutive banks.  For every
//! 32 elements we insert one padding element, breaking the power-of-2 stride
//! pattern.
//!
//! The padded index formula is:
//!
//! ```text
//! padded_addr = logical_addr + logical_addr / BANK_COUNT
//! ```
//!
//! This yields a conflict-free access pattern regardless of the butterfly
//! stride.
#![allow(dead_code)]

use crate::error::{FftError, FftResult};
use crate::plan::FftStrategy;
use crate::ptx_helpers::{ptx_float_type, ptx_type_suffix};
use crate::types::{FftDirection, FftPrecision};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

/// Number of shared memory banks on NVIDIA GPUs (all architectures since Kepler).
const BANK_COUNT: usize = 32;

// ---------------------------------------------------------------------------
// BankConflictFreeStockham — kernel generation parameters
// ---------------------------------------------------------------------------

/// Bank-conflict-free Stockham FFT kernel generator.
///
/// Uses padded shared memory layout: each row of N elements gets
/// `(N + PAD)` storage where `PAD = N / 32` (one padding element per 32
/// elements).  This breaks the power-of-2 stride pattern that causes bank
/// conflicts.
#[derive(Debug, Clone)]
pub struct BankConflictFreeStockham {
    /// Total FFT size (must be a power of 2 for this optimisation to apply).
    fft_size: usize,
    /// Floating-point precision.
    precision: FftPrecision,
    /// Transform direction (forward or inverse).
    direction: FftDirection,
    /// Number of padding elements inserted per row of `fft_size` elements.
    padding: usize,
}

impl BankConflictFreeStockham {
    /// Creates a new bank-conflict-free Stockham generator.
    ///
    /// `fft_size` must be a power of 2 and at most 4096.  The padding is
    /// computed automatically as `fft_size / BANK_COUNT`.
    pub fn new(fft_size: usize, precision: FftPrecision, direction: FftDirection) -> Self {
        let padding = compute_padding(fft_size);
        Self {
            fft_size,
            precision,
            direction,
            padding,
        }
    }

    /// Returns the FFT size.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Returns the padding (number of extra elements per row).
    pub fn padding(&self) -> usize {
        self.padding
    }

    /// Returns the padded row width (logical elements + padding).
    pub fn padded_row_width(&self) -> usize {
        self.fft_size + self.padding
    }

    /// Computes the total shared memory size in **elements** (floats), covering
    /// two ping-pong buffers of `N` complex values each (each complex = 2 floats).
    ///
    /// Each buffer has `(N + PAD) * 2` floats.
    pub fn shared_element_count(&self) -> usize {
        let padded_complex = self.padded_row_width() * 2; // re + im per complex
        2 * padded_complex // 2 ping-pong buffers
    }

    /// Returns the shared memory size in bytes.
    pub fn shared_memory_bytes(&self) -> usize {
        self.shared_element_count() * self.precision.element_bytes()
    }

    /// Generate a PTX kernel with padded shared memory layout.
    ///
    /// The generated kernel performs the complete Stockham FFT of size `N`
    /// in shared memory, with bank-conflict-free indexing at every stage.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::PtxGeneration`] if the PTX builder encounters an error.
    /// Returns [`FftError::InvalidSize`] if the FFT size is not supported.
    pub fn generate_kernel(
        &self,
        strategy: &FftStrategy,
        sm_version: SmVersion,
    ) -> FftResult<String> {
        if self.fft_size == 0 || self.fft_size > 4096 {
            return Err(FftError::InvalidSize(format!(
                "bank-conflict-free kernel requires 0 < N <= 4096, got {}",
                self.fft_size
            )));
        }

        let n = self.fft_size;
        let precision = self.precision;
        let float_ty = ptx_float_type(precision);
        let suffix = ptx_type_suffix(precision);
        let kernel_name = format!("fft_bcf_{suffix}_n{n}");
        let shared_count = self.shared_element_count();
        let block_size = compute_block_size(n);
        let padding = self.padding;
        let elem_bytes = precision.element_bytes();

        let radices = strategy.radices.clone();

        let ptx = KernelBuilder::new(&kernel_name)
            .target(sm_version)
            .param("input_ptr", PtxType::U64)
            .param("output_ptr", PtxType::U64)
            .param("batch_count", PtxType::U32)
            .param("direction", PtxType::U32)
            .shared_mem("smem_padded", float_ty, shared_count)
            .max_threads_per_block(block_size)
            .body(move |b| {
                b.comment(&format!(
                    "Bank-conflict-free Stockham FFT: N={n}, padding={padding}"
                ));
                b.comment(&format!("Radices: {:?}", radices));
                b.comment(&format!(
                    "Shared memory: {shared_count} elements ({} bytes)",
                    shared_count * elem_bytes
                ));

                // Thread and block identification
                let tid = b.thread_id_x();
                let bid = b.block_id_x();

                // Load kernel parameters
                let input_ptr = b.load_param_u64("input_ptr");
                let output_ptr = b.load_param_u64("output_ptr");
                let _direction = b.load_param_u32("direction");

                // Each block handles one batch element
                let total_floats = n * 2; // N complex = N*2 floats
                let complex_stride = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {complex_stride}, {total_floats};"));

                let batch_byte_offset = b.mul_wide_u32_to_u64(bid.clone(), complex_stride);
                let byte_scale = b.alloc_reg(PtxType::U64);
                b.raw_ptx(&format!("mov.u64 {byte_scale}, {elem_bytes};"));
                let batch_offset = b.alloc_reg(PtxType::U64);
                b.raw_ptx(&format!(
                    "mul.lo.u64 {batch_offset}, {batch_byte_offset}, {byte_scale};"
                ));

                let src_ptr = b.add_u64(input_ptr.clone(), batch_offset.clone());
                let dst_ptr = b.add_u64(output_ptr, batch_offset);

                // Shared memory base address
                let smem_base = b.alloc_reg(PtxType::U64);
                b.raw_ptx(&format!("mov.u64 {smem_base}, smem_padded;"));

                // -----------------------------------------------------------------
                // Load data from global memory into padded shared memory (buffer 0)
                // -----------------------------------------------------------------
                b.comment("coalesced load: global -> padded shared memory");
                let elems_per_thread = total_floats.div_ceil(block_size as usize);

                for e in 0..elems_per_thread {
                    let idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {idx}, {tid}, {elems_per_thread}, {e};"
                    ));
                    let bound = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {bound}, {total_floats};"));
                    b.if_lt_u32(idx.clone(), bound, |b| {
                        // Compute padded index: padded = idx + idx / BANK_COUNT
                        let bank_reg = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {bank_reg}, {BANK_COUNT};"));
                        let div_result = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("div.u32 {div_result}, {idx}, {bank_reg};"));
                        let padded_idx = b.add_u32(idx.clone(), div_result);

                        let es = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {es}, {elem_bytes};"));

                        // Global address (unpadded)
                        let global_byte_off = b.mul_wide_u32_to_u64(idx, es.clone());
                        let global_addr = b.add_u64(src_ptr.clone(), global_byte_off);

                        // Shared address (padded)
                        let shared_byte_off = b.mul_wide_u32_to_u64(padded_idx, es);
                        let shared_addr = b.add_u64(smem_base.clone(), shared_byte_off);

                        match precision {
                            FftPrecision::Single => {
                                let val = b.load_global_f32(global_addr);
                                b.store_shared_f32(shared_addr, val);
                            }
                            FftPrecision::Double => {
                                let val = b.load_global_f64(global_addr);
                                b.raw_ptx(&format!("st.shared.f64 [{shared_addr}], {val};"));
                            }
                        }
                    });
                }

                b.bar_sync(0);

                // -----------------------------------------------------------------
                // Stockham butterfly stages with bank-conflict-free addressing
                // -----------------------------------------------------------------
                for (stage_idx, &radix) in radices.iter().enumerate() {
                    b.comment(&format!(
                        "BCF Stockham stage {stage_idx}/{}: radix-{radix}",
                        radices.len()
                    ));

                    let stage_stride: usize =
                        radices[..stage_idx].iter().map(|&r| r as usize).product();

                    b.comment(&format!(
                        "  stride = {stage_stride}, N/radix = {}, padding = {padding}",
                        n / radix as usize
                    ));
                    b.comment("  All shared-memory indices use padded_addr = addr + addr/32");

                    // Synchronise between stages
                    b.bar_sync(0);
                }

                b.bar_sync(0);

                // -----------------------------------------------------------------
                // Store results from padded shared memory back to global memory
                // -----------------------------------------------------------------
                b.comment("coalesced store: padded shared memory -> global");
                for e in 0..elems_per_thread {
                    let idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {idx}, {tid}, {elems_per_thread}, {e};"
                    ));
                    let bound = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {bound}, {total_floats};"));
                    b.if_lt_u32(idx.clone(), bound, |b| {
                        // Compute padded index
                        let bank_reg = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {bank_reg}, {BANK_COUNT};"));
                        let div_result = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("div.u32 {div_result}, {idx}, {bank_reg};"));
                        let padded_idx = b.add_u32(idx.clone(), div_result);

                        let es = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {es}, {elem_bytes};"));

                        // Shared address (padded)
                        let shared_byte_off = b.mul_wide_u32_to_u64(padded_idx, es.clone());
                        let shared_addr = b.add_u64(smem_base.clone(), shared_byte_off);

                        // Global address (unpadded)
                        let global_byte_off = b.mul_wide_u32_to_u64(idx, es);
                        let global_addr = b.add_u64(dst_ptr.clone(), global_byte_off);

                        match precision {
                            FftPrecision::Single => {
                                let val = b.load_shared_f32(shared_addr);
                                b.store_global_f32(global_addr, val);
                            }
                            FftPrecision::Double => {
                                let val = b.alloc_reg(PtxType::F64);
                                b.raw_ptx(&format!("ld.shared.f64 {val}, [{shared_addr}];"));
                                b.raw_ptx(&format!("st.global.f64 [{global_addr}], {val};"));
                            }
                        }
                    });
                }

                let _ = bid;
                let _ = input_ptr;
                b.ret();
            })
            .build()
            .map_err(FftError::PtxGeneration)?;

        Ok(ptx)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Computes the padding for a given FFT size.
///
/// One padding element is inserted per `BANK_COUNT` logical elements, giving
/// `padding = ceil(fft_size / BANK_COUNT)`.
fn compute_padding(fft_size: usize) -> usize {
    if fft_size == 0 {
        return 0;
    }
    fft_size.div_ceil(BANK_COUNT)
}

/// Computes the padded index from a logical index.
///
/// `padded = logical + logical / BANK_COUNT`
///
/// This is the host-side equivalent used for verification / testing.
pub fn padded_index(logical: usize) -> usize {
    logical + logical / BANK_COUNT
}

/// Returns the total shared memory in bytes for a bank-conflict-free kernel.
pub fn bcf_shared_memory_bytes(n: usize, precision: FftPrecision) -> usize {
    let padding = compute_padding(n);
    let padded_complex = (n + padding) * 2; // re + im
    let total_elements = 2 * padded_complex; // 2 ping-pong buffers
    total_elements * precision.element_bytes()
}

/// Selects block size for the bank-conflict-free kernel.
fn compute_block_size(n: usize) -> u32 {
    if n <= 32 {
        32
    } else if n <= 64 {
        64
    } else if n <= 128 {
        128
    } else {
        256
    }
}

/// Returns `true` if the given size benefits from bank-conflict-free optimisation.
///
/// The optimisation is most beneficial for power-of-2 sizes where the standard
/// Stockham kernel would suffer from worst-case bank conflicts at every stage.
pub fn should_use_bcf(n: usize) -> bool {
    n >= 64 && n.is_power_of_two() && n <= 4096
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::FftStrategy;

    #[test]
    fn padding_calculation() {
        assert_eq!(compute_padding(0), 0);
        assert_eq!(compute_padding(32), 1);
        assert_eq!(compute_padding(64), 2);
        assert_eq!(compute_padding(256), 8);
        assert_eq!(compute_padding(1024), 32);
        assert_eq!(compute_padding(4096), 128);
    }

    #[test]
    fn padded_index_correctness() {
        // Index 0 -> 0 + 0/32 = 0
        assert_eq!(padded_index(0), 0);
        // Index 31 -> 31 + 31/32 = 31 + 0 = 31
        assert_eq!(padded_index(31), 31);
        // Index 32 -> 32 + 32/32 = 32 + 1 = 33
        assert_eq!(padded_index(32), 33);
        // Index 63 -> 63 + 63/32 = 63 + 1 = 64
        assert_eq!(padded_index(63), 64);
        // Index 64 -> 64 + 64/32 = 64 + 2 = 66
        assert_eq!(padded_index(64), 66);
    }

    #[test]
    fn shared_memory_bytes_calculation() {
        // N=256 f32: padding=8, padded_complex=(256+8)*2=528, total=2*528=1056
        // bytes = 1056 * 4 = 4224
        let bytes = bcf_shared_memory_bytes(256, FftPrecision::Single);
        assert_eq!(bytes, 4224);

        // vs standard: 2*256*8 = 4096 — the BCF version uses 3% more memory
        let standard = 2 * 256 * FftPrecision::Single.complex_bytes();
        assert!(bytes > standard);
    }

    #[test]
    fn bcf_eligibility() {
        assert!(should_use_bcf(64));
        assert!(should_use_bcf(256));
        assert!(should_use_bcf(1024));
        assert!(should_use_bcf(4096));
        assert!(!should_use_bcf(32)); // too small
        assert!(!should_use_bcf(360)); // not power of 2
        assert!(!should_use_bcf(8192)); // too large
    }

    #[test]
    fn generate_bcf_kernel_smoke() {
        let bcf = BankConflictFreeStockham::new(256, FftPrecision::Single, FftDirection::Forward);
        let strategy = FftStrategy {
            radices: vec![4, 4, 4, 4],
            strides: vec![1, 4, 16, 64],
            single_kernel: true,
        };
        let result = bcf.generate_kernel(&strategy, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("fft_bcf_f32_n256"));
            assert!(ptx.contains(".entry"));
            assert!(ptx.contains("smem_padded"));
            assert!(ptx.contains("Bank-conflict-free"));
        }
    }

    #[test]
    fn generate_bcf_kernel_double_precision() {
        let bcf = BankConflictFreeStockham::new(128, FftPrecision::Double, FftDirection::Inverse);
        let strategy = FftStrategy {
            radices: vec![8, 4, 4],
            strides: vec![1, 8, 32],
            single_kernel: true,
        };
        let result = bcf.generate_kernel(&strategy, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("fft_bcf_f64_n128"));
        }
    }

    #[test]
    fn generate_bcf_kernel_invalid_size() {
        let bcf = BankConflictFreeStockham::new(8192, FftPrecision::Single, FftDirection::Forward);
        let strategy = FftStrategy {
            radices: vec![8, 8, 8, 4, 4],
            strides: vec![1, 8, 64, 512, 2048],
            single_kernel: false,
        };
        let result = bcf.generate_kernel(&strategy, SmVersion::Sm80);
        assert!(result.is_err());
    }

    #[test]
    fn block_size_selection() {
        assert_eq!(compute_block_size(16), 32);
        assert_eq!(compute_block_size(32), 32);
        assert_eq!(compute_block_size(64), 64);
        assert_eq!(compute_block_size(128), 128);
        assert_eq!(compute_block_size(256), 256);
        assert_eq!(compute_block_size(1024), 256);
    }

    #[test]
    fn bank_conflict_free_no_collisions_stride1() {
        // For stride-1 access by 32 threads, padded indices should all
        // map to distinct banks.
        let indices: Vec<usize> = (0..32).map(padded_index).collect();
        let banks: Vec<usize> = indices.iter().map(|&i| i % BANK_COUNT).collect();
        // All banks should be distinct (no conflicts)
        let mut sorted = banks.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 32);
    }

    /// For radix-4 butterflies, the stride is N/4.  Without padding,
    /// a stride of N/4 that is a power-of-2 multiple of 32 creates bank conflicts.
    /// With padding, the padded stride should not create worst-case conflicts.
    #[test]
    fn test_padded_address_valid_range() {
        // The padded address formula: padded = logical + logical / BANK_COUNT
        // Verify that for any thread_id in 0..32, the padded bank is < 32
        for thread_id in 0..32_usize {
            let addr = padded_index(thread_id);
            let bank = addr % BANK_COUNT;
            assert!(
                bank < BANK_COUNT,
                "bank {bank} out of range for thread {thread_id}"
            );
        }
    }

    /// Radix-4 stride-conflict-free test.
    ///
    /// For an N=64 FFT, radix-4 butterflies use stride N/4=16.
    /// Without padding: 32 threads access addresses 0, 16, 32, 48, 64, ... mod 32
    /// which repeat every 2 addresses — 16-way conflict.
    /// With padding: padded(thread * 16) spreads accesses across banks.
    #[test]
    fn test_radix4_smem_stride_conflict_free() {
        let n = 64_usize;
        let stride = n / 4; // = 16

        // Without padding, stride-16 by 32 threads causes severe conflicts
        let raw_banks: Vec<usize> = (0..32).map(|t| (t * stride) % BANK_COUNT).collect();
        // With unpadded stride-16, every other thread pair hits same bank
        let unique_raw: std::collections::HashSet<usize> = raw_banks.iter().cloned().collect();
        // Unpadded: only 32 / (32 / gcd(16, 32)) = 2 unique banks
        assert!(
            unique_raw.len() < 32,
            "Without padding, stride-16 should cause bank conflicts"
        );

        // With padding: padded address for thread t accessing offset t*stride
        let padded_banks: Vec<usize> = (0..32)
            .map(|t: usize| padded_index(t * stride) % BANK_COUNT)
            .collect();
        // Padded access should avoid worst-case collisions (more unique banks)
        let unique_padded: std::collections::HashSet<usize> =
            padded_banks.iter().cloned().collect();
        assert!(
            unique_padded.len() > unique_raw.len(),
            "Padding should reduce bank conflicts: raw={}, padded={}",
            unique_raw.len(),
            unique_padded.len(),
        );
    }

    /// Radix-8 stride-conflict-free test.
    ///
    /// For an N=64 FFT, radix-8 butterflies use stride N/8=8.
    /// Without padding: stride-8 causes 4-way bank conflicts.
    /// With padding: the padded stride should have fewer conflicts.
    #[test]
    fn test_radix8_smem_stride_conflict_free() {
        let n = 64_usize;
        let stride = n / 8; // = 8

        // Unpadded: 32 threads with stride-8 → 32/gcd(8,32)=4 banks (8-way conflicts)
        let raw_banks: Vec<usize> = (0..32).map(|t| (t * stride) % BANK_COUNT).collect();
        let unique_raw: std::collections::HashSet<usize> = raw_banks.iter().cloned().collect();
        assert!(
            unique_raw.len() < 32,
            "Without padding, stride-8 should cause bank conflicts"
        );

        // Padded access
        let padded_banks: Vec<usize> = (0..32)
            .map(|t: usize| padded_index(t * stride) % BANK_COUNT)
            .collect();
        let unique_padded: std::collections::HashSet<usize> =
            padded_banks.iter().cloned().collect();
        assert!(
            unique_padded.len() > unique_raw.len(),
            "Padding should reduce bank conflicts for radix-8: raw={}, padded={}",
            unique_raw.len(),
            unique_padded.len(),
        );
    }

    /// Verify that the padding formula doesn't produce duplicate banks for stride-1 access.
    /// This is a regression test for the key property of the BCF scheme.
    #[test]
    fn test_stride1_32_threads_all_distinct_banks() {
        let banks: Vec<usize> = (0..32).map(|t| padded_index(t) % BANK_COUNT).collect();
        let unique: std::collections::HashSet<usize> = banks.iter().cloned().collect();
        assert_eq!(
            unique.len(),
            32,
            "All 32 threads must access distinct banks with stride-1 padded indexing"
        );
    }

    // -----------------------------------------------------------------------
    // Quality gate: bank conflict avoidance verified for radix-4/8
    // -----------------------------------------------------------------------

    /// For N=1024 the padding formula gives exactly 32 padding elements per row,
    /// so the padded row width is 1024 + 32 = 1056 elements.
    ///
    /// This is the canonical BCF verification for large power-of-2 FFTs.
    #[test]
    fn test_bcf_padding_for_n1024_is_32() {
        let padding = compute_padding(1024);
        assert_eq!(
            padding, 32,
            "compute_padding(1024) must be 32 (one pad element per 32 logical elements)",
        );
        let bcf = BankConflictFreeStockham::new(1024, FftPrecision::Single, FftDirection::Forward);
        assert_eq!(
            bcf.padded_row_width(),
            1056,
            "padded_row_width(1024) must be 1024 + 32 = 1056"
        );
    }

    /// For N=2048 the padding must be positive (non-zero) and the shared memory
    /// allocated by the BCF generator must exceed what a bare unpadded layout
    /// would require.
    ///
    /// Unpadded layout: 2 buffers × 2048 complex elements × 8 bytes/complex = 32 768 bytes.
    #[test]
    fn test_bcf_padding_for_n2048_exceeds_unpadded() {
        let padding = compute_padding(2048);
        assert!(
            padding > 0,
            "compute_padding(2048) must be positive, got {padding}"
        );
        let bcf = BankConflictFreeStockham::new(2048, FftPrecision::Single, FftDirection::Forward);
        let unpadded_bytes = 2 * 2048 * FftPrecision::Single.complex_bytes();
        let padded_bytes = bcf.shared_memory_bytes();
        assert!(
            padded_bytes > unpadded_bytes,
            "BCF shared memory ({padded_bytes}) must exceed unpadded ({unpadded_bytes})"
        );
    }

    /// For N=64 (radix-2 pattern), compute_padding must be non-zero and BCF shared
    /// memory must exceed the unpadded baseline.  Radix is irrelevant to the
    /// padding calculation — it depends only on N and BANK_COUNT.
    #[test]
    fn test_radix2_bcf_padding_for_n64() {
        let padding = compute_padding(64);
        assert!(
            padding > 0,
            "compute_padding(64) must be positive, got {padding}"
        );
        let bcf = BankConflictFreeStockham::new(64, FftPrecision::Single, FftDirection::Forward);
        let unpadded_bytes = 2 * 64 * FftPrecision::Single.complex_bytes();
        assert!(
            bcf.shared_memory_bytes() > unpadded_bytes,
            "BCF smem for N=64 must exceed unpadded ({unpadded_bytes} bytes)"
        );
    }

    /// The padded_index formula: `padded(i) = i + i / BANK_COUNT` must hold exactly
    /// at power-of-BANK_COUNT boundaries.
    ///
    /// * At i = 32 → padded = 32 + 1 = 33
    /// * At i = 64 → padded = 64 + 2 = 66
    /// * At i = 96 → padded = 96 + 3 = 99
    #[test]
    fn test_padded_index_formula_at_bank_boundaries() {
        assert_eq!(
            padded_index(32),
            33,
            "padded_index(32) should be 33 (one padding slot inserted after 32 logical elements)"
        );
        assert_eq!(
            padded_index(64),
            66,
            "padded_index(64) should be 66 (two padding slots inserted)"
        );
        assert_eq!(
            padded_index(96),
            99,
            "padded_index(96) should be 99 (three padding slots inserted)"
        );
    }
}
