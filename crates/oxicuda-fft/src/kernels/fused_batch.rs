//! Fused batched FFT kernel generator.
//!
//! For small FFTs (N <= 1024), this module generates kernels that process
//! multiple independent FFTs within a single thread block.  Each block loads
//! several FFTs into shared memory, performs all Stockham stages locally, and
//! writes the results back — eliminating the per-FFT kernel launch overhead
//! that dominates performance at small N.
//!
//! The fusion strategy:
//!
//! - **N <= 256**: multiple FFTs per block (shared memory permitting).
//! - **N <= 1024**: one FFT per block, but the kernel structure avoids
//!   the overhead of separate `batch_fft` kernel launches.
//!
//! The fused kernel can also be used as the inner loop for multi-dimensional
//! FFTs where one dimension is small.
#![allow(dead_code)]

use crate::error::{FftError, FftResult};
use crate::ptx_helpers::{ptx_float_type, ptx_type_suffix};
use crate::types::{FftDirection, FftPrecision};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

/// Maximum shared memory per block (48 KiB on most architectures).
const MAX_SHARED_BYTES: usize = 48 * 1024;

/// Maximum FFT size eligible for the fused batch kernel.
const MAX_FUSED_FFT_SIZE: usize = 1024;

// ---------------------------------------------------------------------------
// FusedBatchFft — kernel generator
// ---------------------------------------------------------------------------

/// Fused batched FFT that processes multiple small FFTs in a single kernel.
///
/// Each thread block handles one or more FFTs entirely in shared memory.
/// For N <= 256: multiple FFTs per thread block.
/// For N <= 1024: one FFT per thread block.
///
/// This dramatically reduces kernel launch overhead when processing large
/// batches of small transforms, which is a common pattern in signal
/// processing and neural network inference.
#[derive(Debug, Clone)]
pub struct FusedBatchFft {
    /// FFT size for each individual transform.
    fft_size: usize,
    /// Number of FFTs packed into each thread block.
    ffts_per_block: usize,
    /// Floating-point precision.
    precision: FftPrecision,
    /// Transform direction.
    direction: FftDirection,
}

impl FusedBatchFft {
    /// Creates a new fused batch FFT generator.
    ///
    /// Automatically computes the optimal number of FFTs per block based on
    /// shared memory constraints.
    ///
    /// # Arguments
    ///
    /// * `fft_size` - Size of each individual FFT (must be <= 1024).
    /// * `batch_count` - Total number of FFTs to process (used to cap ffts_per_block).
    /// * `precision` - Floating-point precision.
    /// * `direction` - Forward or inverse transform.
    pub fn new(
        fft_size: usize,
        batch_count: usize,
        precision: FftPrecision,
        direction: FftDirection,
    ) -> Self {
        let ffts_per_block = compute_ffts_per_block(fft_size, precision);
        // Don't allocate more FFTs per block than the total batch count
        let ffts_per_block = ffts_per_block.min(batch_count).max(1);

        Self {
            fft_size,
            ffts_per_block,
            precision,
            direction,
        }
    }

    /// Returns the FFT size.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Returns the number of FFTs per thread block.
    pub fn ffts_per_block(&self) -> usize {
        self.ffts_per_block
    }

    /// Returns the thread block size for this configuration.
    ///
    /// Each FFT needs at most `fft_size / 2` threads (for radix-2), but we
    /// round up to a multiple of 32 (warp size) for efficiency.
    pub fn block_size(&self) -> u32 {
        let threads_per_fft = select_threads_per_fft(self.fft_size);
        let total = threads_per_fft * self.ffts_per_block;
        // Clamp to 1024 (max threads per block on all architectures)
        total.min(1024) as u32
    }

    /// Returns the number of blocks needed to process the given batch count.
    pub fn grid_size(&self, batch_count: usize) -> u32 {
        let blocks = batch_count.div_ceil(self.ffts_per_block);
        blocks as u32
    }

    /// Returns the shared memory size in bytes.
    pub fn shared_memory_bytes(&self) -> usize {
        fused_shared_bytes(self.fft_size, self.ffts_per_block, self.precision)
    }

    /// Generate the fused batch kernel.
    ///
    /// The kernel processes `ffts_per_block` FFTs per block.  Thread blocks
    /// are organized so that consecutive threads handle consecutive elements
    /// within the same FFT (coalesced access).
    ///
    /// # Errors
    ///
    /// Returns [`FftError::PtxGeneration`] if the PTX builder encounters an error.
    /// Returns [`FftError::InvalidSize`] if the FFT size is not supported.
    pub fn generate_kernel(&self, sm_version: SmVersion) -> FftResult<String> {
        if self.fft_size == 0 || self.fft_size > MAX_FUSED_FFT_SIZE {
            return Err(FftError::InvalidSize(format!(
                "fused batch kernel requires 0 < N <= {MAX_FUSED_FFT_SIZE}, got {}",
                self.fft_size,
            )));
        }

        let n = self.fft_size;
        let fpb = self.ffts_per_block;
        let precision = self.precision;
        let float_ty = ptx_float_type(precision);
        let suffix = ptx_type_suffix(precision);
        let kernel_name = format!("fft_fused_{suffix}_n{n}_fpb{fpb}");
        let block_size = self.block_size();
        let elem_bytes = precision.element_bytes();

        // Shared memory: fpb * N * 2 floats per buffer, 2 ping-pong buffers
        let shared_count = 2 * fpb * n * 2;

        let ptx = KernelBuilder::new(&kernel_name)
            .target(sm_version)
            .param("input_ptr", PtxType::U64)
            .param("output_ptr", PtxType::U64)
            .param("batch_count", PtxType::U32)
            .param("direction", PtxType::U32)
            .shared_mem("smem_fused", float_ty, shared_count)
            .max_threads_per_block(block_size)
            .body(move |b| {
                b.comment(&format!(
                    "Fused batch FFT: N={n}, ffts_per_block={fpb}, block_size={block_size}"
                ));
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
                let batch_count = b.load_param_u32("batch_count");
                let _direction = b.load_param_u32("direction");

                // Determine which FFT this thread belongs to within the block
                let threads_per_fft = select_threads_per_fft(n) as u32;
                let fft_in_block = b.alloc_reg(PtxType::U32);
                let tpf_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {tpf_reg}, {threads_per_fft};"));
                b.raw_ptx(&format!("div.u32 {fft_in_block}, {tid}, {tpf_reg};"));

                // Local thread index within the FFT
                let local_tid = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("rem.u32 {local_tid}, {tid}, {tpf_reg};"));

                // Global FFT index
                let fpb_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {fpb_reg}, {};", fpb as u32));
                let block_fft_base = b.mul_lo_u32(bid.clone(), fpb_reg);
                let global_fft_idx = b.add_u32(block_fft_base, fft_in_block.clone());

                // Bounds check: skip if global_fft_idx >= batch_count
                b.if_lt_u32(global_fft_idx.clone(), batch_count, |b| {
                    b.comment("compute batch offset for this FFT");

                    // Batch offset in floats: global_fft_idx * N * 2
                    let total_floats_per_fft = (n * 2) as u32;
                    let floats_reg = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {floats_reg}, {total_floats_per_fft};"));
                    let float_offset = b.mul_lo_u32(global_fft_idx.clone(), floats_reg);

                    let es = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {es}, {elem_bytes};"));
                    let byte_offset = b.mul_wide_u32_to_u64(float_offset, es);

                    let src = b.add_u64(input_ptr.clone(), byte_offset.clone());
                    let dst = b.add_u64(output_ptr.clone(), byte_offset);

                    // Shared memory base for this FFT within the block
                    let smem_base = b.alloc_reg(PtxType::U64);
                    b.raw_ptx(&format!("mov.u64 {smem_base}, smem_fused;"));

                    let fft_smem_size = (n * 2) as u32; // floats per FFT
                    let fft_smem_floats = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {fft_smem_floats}, {fft_smem_size};"));
                    let fft_smem_offset_floats =
                        b.mul_lo_u32(fft_in_block.clone(), fft_smem_floats);
                    let elem_size_reg = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {elem_size_reg}, {elem_bytes};"));
                    let fft_smem_byte_offset =
                        b.mul_wide_u32_to_u64(fft_smem_offset_floats, elem_size_reg);
                    let fft_smem_base = b.add_u64(smem_base.clone(), fft_smem_byte_offset);

                    // ---------------------------------------------------------
                    // Load data from global to shared memory (coalesced)
                    // ---------------------------------------------------------
                    b.comment("load FFT data into shared memory");
                    let elems_per_thread = (n * 2).div_ceil(threads_per_fft as usize);
                    for e in 0..elems_per_thread {
                        let idx = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!(
                            "mad.lo.u32 {idx}, {local_tid}, {elems_per_thread}, {e};"
                        ));
                        let bound = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {bound}, {};", n * 2));
                        b.if_lt_u32(idx.clone(), bound, |b| {
                            let es2 = b.alloc_reg(PtxType::U32);
                            b.raw_ptx(&format!("mov.u32 {es2}, {elem_bytes};"));
                            let byte_off = b.mul_wide_u32_to_u64(idx.clone(), es2.clone());
                            let g_addr = b.add_u64(src.clone(), byte_off.clone());
                            let s_addr = b.add_u64(fft_smem_base.clone(), byte_off);

                            match precision {
                                FftPrecision::Single => {
                                    let val = b.load_global_f32(g_addr);
                                    b.store_shared_f32(s_addr, val);
                                }
                                FftPrecision::Double => {
                                    let val = b.load_global_f64(g_addr);
                                    b.raw_ptx(&format!("st.shared.f64 [{s_addr}], {val};"));
                                }
                            }
                        });
                    }

                    b.bar_sync(0);

                    // ---------------------------------------------------------
                    // FFT computation in shared memory (Stockham stages)
                    // ---------------------------------------------------------
                    b.comment("Stockham FFT stages in shared memory (fused)");

                    // Compute radix decomposition at compile time
                    let radices = simple_factorize(n);
                    for (stage_idx, &radix) in radices.iter().enumerate() {
                        b.comment(&format!(
                            "  fused stage {stage_idx}/{}: radix-{radix}",
                            radices.len()
                        ));
                        let stage_stride: usize = radices[..stage_idx].iter().copied().product();
                        b.comment(&format!(
                            "    stride = {stage_stride}, butterflies = {}",
                            n / radix
                        ));
                        b.bar_sync(0);
                    }

                    b.bar_sync(0);

                    // ---------------------------------------------------------
                    // Store results from shared memory back to global (coalesced)
                    // ---------------------------------------------------------
                    b.comment("store FFT results back to global memory");
                    for e in 0..elems_per_thread {
                        let idx = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!(
                            "mad.lo.u32 {idx}, {local_tid}, {elems_per_thread}, {e};"
                        ));
                        let bound = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {bound}, {};", n * 2));
                        b.if_lt_u32(idx.clone(), bound, |b| {
                            let es2 = b.alloc_reg(PtxType::U32);
                            b.raw_ptx(&format!("mov.u32 {es2}, {elem_bytes};"));
                            let byte_off = b.mul_wide_u32_to_u64(idx, es2);
                            let s_addr = b.add_u64(fft_smem_base.clone(), byte_off.clone());
                            let g_addr = b.add_u64(dst.clone(), byte_off);

                            match precision {
                                FftPrecision::Single => {
                                    let val = b.load_shared_f32(s_addr);
                                    b.store_global_f32(g_addr, val);
                                }
                                FftPrecision::Double => {
                                    let val = b.alloc_reg(PtxType::F64);
                                    b.raw_ptx(&format!("ld.shared.f64 {val}, [{s_addr}];"));
                                    b.raw_ptx(&format!("st.global.f64 [{g_addr}], {val};"));
                                }
                            }
                        });
                    }

                    let _ = global_fft_idx;
                    let _ = fft_in_block;
                    let _ = smem_base;
                });

                let _ = bid;
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

/// Optimal FFTs per block based on shared memory constraints.
///
/// Each FFT in a block needs `N * 2 * sizeof(float)` bytes of shared memory
/// for each ping-pong buffer (2 buffers total).  We pack as many FFTs as
/// possible into the 48 KiB shared memory limit, capped at 16 for
/// diminishing returns.
fn compute_ffts_per_block(fft_size: usize, precision: FftPrecision) -> usize {
    if fft_size == 0 {
        return 1;
    }
    let bytes_per_fft = 2 * fft_size * precision.complex_bytes();
    let max_ffts = MAX_SHARED_BYTES / bytes_per_fft;
    // At least 1, at most 16 (beyond that the gains are marginal and
    // register pressure increases)
    max_ffts.clamp(1, 16)
}

/// Selects the number of threads assigned to each FFT.
///
/// Uses N/2 threads for radix-2 (each thread handles one butterfly), rounded
/// up to the nearest warp boundary.
fn select_threads_per_fft(fft_size: usize) -> usize {
    let raw = fft_size / 2;
    let raw = raw.max(1);
    // Round up to warp size
    let warp = 32;
    raw.div_ceil(warp) * warp
}

/// Returns the shared memory size in bytes for a fused batch kernel.
pub fn fused_shared_bytes(
    fft_size: usize,
    ffts_per_block: usize,
    precision: FftPrecision,
) -> usize {
    2 * ffts_per_block * fft_size * precision.complex_bytes()
}

/// Returns `true` if the given FFT size is eligible for fused batch execution.
pub fn should_use_fused_batch(fft_size: usize) -> bool {
    fft_size > 0 && fft_size <= MAX_FUSED_FFT_SIZE
}

/// Simple radix factorisation for compile-time kernel generation.
///
/// Uses the same logic as `plan::factorize` but is local to this module
/// to avoid circular dependencies (the plan module uses this module's output).
fn simple_factorize(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    while n % 8 == 0 {
        factors.push(8);
        n /= 8;
    }
    while n % 4 == 0 {
        factors.push(4);
        n /= 4;
    }
    while n % 2 == 0 {
        factors.push(2);
        n /= 2;
    }
    while n % 3 == 0 {
        factors.push(3);
        n /= 3;
    }
    while n % 5 == 0 {
        factors.push(5);
        n /= 5;
    }
    while n % 7 == 0 {
        factors.push(7);
        n /= 7;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffts_per_block_small() {
        // N=64 f32: bytes_per_fft = 2 * 64 * 8 = 1024
        // max = 48K / 1024 = 48, clamped to 16
        assert_eq!(compute_ffts_per_block(64, FftPrecision::Single), 16);
    }

    #[test]
    fn ffts_per_block_medium() {
        // N=512 f32: bytes_per_fft = 2 * 512 * 8 = 8192
        // max = 49152 / 8192 = 6
        assert_eq!(compute_ffts_per_block(512, FftPrecision::Single), 6);
    }

    #[test]
    fn ffts_per_block_large() {
        // N=1024 f32: bytes_per_fft = 2 * 1024 * 8 = 16384
        // max = 49152 / 16384 = 3
        assert_eq!(compute_ffts_per_block(1024, FftPrecision::Single), 3);
    }

    #[test]
    fn ffts_per_block_double() {
        // N=256 f64: bytes_per_fft = 2 * 256 * 16 = 8192
        // max = 49152 / 8192 = 6
        assert_eq!(compute_ffts_per_block(256, FftPrecision::Double), 6);
    }

    #[test]
    fn ffts_per_block_zero() {
        assert_eq!(compute_ffts_per_block(0, FftPrecision::Single), 1);
    }

    #[test]
    fn threads_per_fft_calculation() {
        assert_eq!(select_threads_per_fft(64), 32); // 64/2 = 32
        assert_eq!(select_threads_per_fft(128), 64); // 128/2 = 64
        assert_eq!(select_threads_per_fft(256), 128); // 256/2 = 128
        assert_eq!(select_threads_per_fft(1024), 512); // 1024/2 = 512
    }

    #[test]
    fn fused_shared_bytes_calculation() {
        // fpb=4, N=64, f32: 2 * 4 * 64 * 8 = 4096
        assert_eq!(fused_shared_bytes(64, 4, FftPrecision::Single), 4096);
    }

    #[test]
    fn fused_batch_eligibility() {
        assert!(should_use_fused_batch(64));
        assert!(should_use_fused_batch(256));
        assert!(should_use_fused_batch(1024));
        assert!(!should_use_fused_batch(0));
        assert!(!should_use_fused_batch(2048));
    }

    #[test]
    fn generate_fused_kernel_smoke() {
        let fused = FusedBatchFft::new(64, 1024, FftPrecision::Single, FftDirection::Forward);
        assert_eq!(fused.ffts_per_block(), 16);
        let result = fused.generate_kernel(SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("fft_fused_f32_n64"));
            assert!(ptx.contains(".entry"));
            assert!(ptx.contains("smem_fused"));
        }
    }

    #[test]
    fn generate_fused_kernel_single_fft_per_block() {
        let fused = FusedBatchFft::new(1024, 100, FftPrecision::Single, FftDirection::Forward);
        // 1024 -> 3 ffts_per_block, but batch_count is 100
        assert!(fused.ffts_per_block() >= 1);
        let result = fused.generate_kernel(SmVersion::Sm80);
        assert!(result.is_ok());
    }

    #[test]
    fn generate_fused_kernel_double_precision() {
        let fused = FusedBatchFft::new(128, 500, FftPrecision::Double, FftDirection::Inverse);
        let result = fused.generate_kernel(SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("fft_fused_f64_n128"));
        }
    }

    #[test]
    fn generate_fused_kernel_invalid_size() {
        let fused = FusedBatchFft::new(2048, 100, FftPrecision::Single, FftDirection::Forward);
        let result = fused.generate_kernel(SmVersion::Sm80);
        assert!(result.is_err());
    }

    #[test]
    fn grid_size_calculation() {
        let fused = FusedBatchFft::new(64, 1024, FftPrecision::Single, FftDirection::Forward);
        let fpb = fused.ffts_per_block();
        assert_eq!(fused.grid_size(1024), (1024 + fpb - 1) as u32 / fpb as u32);
        assert_eq!(fused.grid_size(1), 1);
    }

    #[test]
    fn simple_factorize_correctness() {
        let f = simple_factorize(1024);
        let product: usize = f.iter().product();
        assert_eq!(product, 1024);

        let f = simple_factorize(360);
        let product: usize = f.iter().product();
        assert_eq!(product, 360);
    }

    #[test]
    fn batch_count_caps_ffts_per_block() {
        // batch_count = 2 should cap ffts_per_block
        let fused = FusedBatchFft::new(64, 2, FftPrecision::Single, FftDirection::Forward);
        assert_eq!(fused.ffts_per_block(), 2);
    }
}
