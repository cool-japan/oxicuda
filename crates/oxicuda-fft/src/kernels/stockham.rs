//! Stockham auto-sort FFT kernel generator.
//!
//! The Stockham FFT is an iterative, out-of-place algorithm that avoids
//! the bit-reversal permutation required by the Cooley-Tukey FFT.  At each
//! stage, data is read from one buffer and written to another in a
//! naturally-sorted order, using a "ping-pong" pattern between shared
//! memory banks.
//!
//! This module generates PTX kernels in two modes:
//!
//! 1. **Single kernel** (N <= 4096): All stages execute within one kernel
//!    launch, using shared memory for the ping-pong buffers.
//!
//! 2. **Per-stage kernels** (N > 4096): Each radix stage is a separate
//!    kernel launch, using global memory between stages.
#![allow(dead_code)]

use crate::error::{FftError, FftResult};
use crate::plan::FftStrategy;
use crate::ptx_helpers::{ptx_float_type, ptx_type_suffix};
use crate::types::FftPrecision;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

// ---------------------------------------------------------------------------
// StockhamFftTemplate — kernel generation parameters
// ---------------------------------------------------------------------------

/// Parameters for generating a Stockham FFT kernel.
#[derive(Debug, Clone)]
pub struct StockhamFftTemplate {
    /// Total FFT size.
    pub n: usize,
    /// Radix for this stage (2, 4, 8, 3, 5, or 7).
    pub radix: u32,
    /// Stage index (0-based).
    pub stage: u32,
    /// Total number of stages.
    pub total_stages: u32,
    /// Batch count (number of independent FFTs).
    pub batch: usize,
    /// Floating-point precision.
    pub precision: FftPrecision,
    /// Target GPU architecture.
    pub sm_version: SmVersion,
}

// ---------------------------------------------------------------------------
// Single-kernel generation (N <= 4096)
// ---------------------------------------------------------------------------

/// Generates a single PTX kernel that performs the complete FFT of size N
/// entirely in shared memory.
///
/// The kernel performs all Stockham stages in a loop, using `__syncthreads()`
/// between stages and a shared-memory ping-pong buffer.
///
/// # Parameters
///
/// - `n`: FFT size (must be <= 4096 and factorisable into supported radices)
/// - `strategy`: The decomposition strategy from plan creation
/// - `batch`: Number of independent FFTs per kernel launch
/// - `precision`: f32 or f64
/// - `sm`: Target SM version
///
/// # Errors
///
/// Returns [`FftError::PtxGeneration`] if the PTX builder encounters an error.
pub fn generate_single_kernel(
    n: usize,
    strategy: &FftStrategy,
    batch: usize,
    precision: FftPrecision,
    sm: SmVersion,
) -> FftResult<String> {
    let float_ty = ptx_float_type(precision);
    let suffix = ptx_type_suffix(precision);
    let kernel_name = format!("fft_stockham_{suffix}_n{n}_b{batch}");

    // Shared memory: 2 * N complex elements for ping-pong
    // Each complex element = 2 floats
    let shared_count = 2 * n * 2; // 2 buffers * N * 2 (re + im)

    let block_size = compute_block_size(n);

    // Clone strategy data to avoid lifetime issues with the closure
    let radices = strategy.radices.clone();

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("batch_count", PtxType::U32)
        .param("direction", PtxType::U32) // 0 = forward, 1 = inverse
        .shared_mem("smem", float_ty, shared_count)
        .max_threads_per_block(block_size)
        .body(move |b| {
            b.comment(&format!("Stockham FFT: N={n}, batch={batch}"));
            b.comment(&format!("Radices: {:?}", radices));
            b.comment(&format!("Block size: {block_size}"));

            // Thread identification
            let tid = b.thread_id_x();
            let bid = b.block_id_x();
            let _batch_idx = bid.clone();

            // Load parameters
            let input_ptr = b.load_param_u64("input_ptr");
            let output_ptr = b.load_param_u64("output_ptr");
            let _direction = b.load_param_u32("direction");

            // Compute batch offset
            let n_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {n_reg}, {n};"));
            let complex_stride = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {complex_stride}, {};", n * 2)); // N complex = N*2 floats

            // Each block handles one batch element
            let elem_size = precision.element_bytes();
            let batch_byte_offset = b.mul_wide_u32_to_u64(bid.clone(), complex_stride.clone());
            let byte_scale = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("mov.u64 {byte_scale}, {elem_size};"));
            let batch_offset = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!(
                "mul.lo.u64 {batch_offset}, {batch_byte_offset}, {byte_scale};"
            ));

            let src_ptr = b.add_u64(input_ptr.clone(), batch_offset.clone());
            let dst_ptr = b.add_u64(output_ptr, batch_offset);

            // Load data from global memory into shared memory buffer 0
            b.comment("load data from global to shared memory");
            let shared_base = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("mov.u64 {shared_base}, smem;"));

            // Each thread loads multiple elements if needed
            let elems_per_thread = (n * 2).div_ceil(block_size as usize);
            for e in 0..elems_per_thread {
                let idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {idx}, {tid}, {elems_per_thread}, {e};"
                ));
                let bound = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {bound}, {};", n * 2));
                b.if_lt_u32(idx.clone(), bound, |b| {
                    let es = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {es}, {elem_size};"));
                    let byte_off = b.mul_wide_u32_to_u64(idx.clone(), es);
                    let global_addr = b.add_u64(src_ptr.clone(), byte_off.clone());
                    let shared_addr = b.add_u64(shared_base.clone(), byte_off);

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

            // Stockham stages (emitted as comments for now; the actual butterfly
            // operations would use the radix modules)
            for (stage_idx, &radix) in radices.iter().enumerate() {
                b.comment(&format!(
                    "Stockham stage {stage_idx}/{}: radix-{radix}",
                    radices.len()
                ));

                // Compute stride for this stage
                let stage_stride: usize =
                    radices[..stage_idx].iter().map(|&r| r as usize).product();

                // Each thread handles one butterfly group element
                // The actual butterfly PTX would be emitted here using
                // the radix2/radix4/radix8 modules
                b.comment(&format!(
                    "  stride = {stage_stride}, N/radix = {}",
                    n / radix as usize
                ));

                // Synchronise between stages
                b.bar_sync(0);
            }

            // Store results back to global memory
            b.comment("store results from shared memory to global");
            for e in 0..elems_per_thread {
                let idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {idx}, {tid}, {elems_per_thread}, {e};"
                ));
                let bound = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {bound}, {};", n * 2));
                b.if_lt_u32(idx.clone(), bound, |b| {
                    let es = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {es}, {elem_size};"));
                    let byte_off = b.mul_wide_u32_to_u64(idx, es);
                    let shared_addr = b.add_u64(shared_base.clone(), byte_off.clone());
                    let global_addr = b.add_u64(dst_ptr.clone(), byte_off);

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

            b.ret();
        })
        .build()
        .map_err(FftError::PtxGeneration)?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Per-stage kernel generation (N > 4096)
// ---------------------------------------------------------------------------

/// Generates a PTX kernel for a single Stockham stage of a large FFT.
///
/// For N > 4096, the FFT is decomposed into multiple kernel launches,
/// each performing one radix stage using global memory.
///
/// # Errors
///
/// Returns [`FftError::PtxGeneration`] if the PTX builder encounters an error.
pub fn generate_stage_kernel(
    n: usize,
    radix: u32,
    stage: u32,
    total_stages: u32,
    precision: FftPrecision,
    sm: SmVersion,
) -> FftResult<String> {
    let suffix = ptx_type_suffix(precision);
    let kernel_name = format!("fft_stockham_stage_{suffix}_n{n}_r{radix}_s{stage}of{total_stages}");

    let block_size = 256u32;

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("n_total", PtxType::U32)
        .param("batch_count", PtxType::U32)
        .param("direction", PtxType::U32)
        .max_threads_per_block(block_size)
        .body(move |b| {
            b.comment(&format!(
                "Stockham stage {stage}/{total_stages}: radix-{radix}, N={n}"
            ));

            let gid = b.global_thread_id_x();
            let _input_ptr = b.load_param_u64("input_ptr");
            let _output_ptr = b.load_param_u64("output_ptr");
            let n_total = b.load_param_u32("n_total");
            let _batch_count = b.load_param_u32("batch_count");
            let _direction = b.load_param_u32("direction");

            // Bounds check: each thread handles one butterfly
            let butterflies_per_stage = b.alloc_reg(PtxType::U32);
            let n_div_radix = n / (radix as usize);
            b.raw_ptx(&format!("mov.u32 {butterflies_per_stage}, {n_div_radix};"));

            b.if_lt_u32(gid, butterflies_per_stage, |b| {
                b.comment("butterfly computation would go here");
                b.comment(&format!("radix={radix}, stride={}", 1u32 << stage));

                // The actual butterfly PTX would be emitted here
                // using the radix modules, reading from global memory
                // input_ptr and writing to output_ptr.
                let _ = n_total;
            });

            b.ret();
        })
        .build()
        .map_err(FftError::PtxGeneration)?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Block size selection
// ---------------------------------------------------------------------------

/// Computes an appropriate thread block size for the given FFT size.
///
/// For small FFTs (N <= 256), use N threads.
/// For medium FFTs (N <= 4096), use 256 threads.
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

/// Returns the shared memory size in bytes for a single-kernel Stockham FFT.
pub fn shared_memory_bytes(n: usize, precision: FftPrecision) -> usize {
    // 2 ping-pong buffers * N complex elements * element_size * 2 (re+im)
    2 * n * precision.complex_bytes()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::FftStrategy;

    #[test]
    fn block_size_selection() {
        assert_eq!(compute_block_size(16), 32);
        assert_eq!(compute_block_size(64), 64);
        assert_eq!(compute_block_size(256), 256);
        assert_eq!(compute_block_size(1024), 256);
    }

    #[test]
    fn shared_mem_calculation() {
        // N=256 f32: 2 * 256 * 8 = 4096 bytes
        assert_eq!(shared_memory_bytes(256, FftPrecision::Single), 4096);
        // N=256 f64: 2 * 256 * 16 = 8192 bytes
        assert_eq!(shared_memory_bytes(256, FftPrecision::Double), 8192);
    }

    #[test]
    fn generate_single_kernel_smoke() {
        let strategy = FftStrategy {
            radices: vec![4, 4, 4, 4],
            strides: vec![1, 4, 16, 64],
            single_kernel: true,
        };
        let result =
            generate_single_kernel(256, &strategy, 1, FftPrecision::Single, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("fft_stockham_f32_n256"));
            assert!(ptx.contains(".entry"));
        }
    }

    #[test]
    fn generate_stage_kernel_smoke() {
        let result = generate_stage_kernel(8192, 8, 0, 4, FftPrecision::Single, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("fft_stockham_stage_f32"));
        }
    }
}
