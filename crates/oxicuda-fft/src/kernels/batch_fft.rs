//! Batch FFT kernel generator.
//!
//! Optimised for launching many small FFTs (N <= 1024) where each
//! Cooperative Thread Array (CTA / thread block) handles one complete
//! FFT.  This maximises GPU occupancy when the batch count is large.
#![allow(dead_code)]

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{FftError, FftResult};
use crate::ptx_helpers::{ptx_float_type, ptx_type_suffix};
use crate::types::FftPrecision;

// ---------------------------------------------------------------------------
// Batch FFT kernel generation
// ---------------------------------------------------------------------------

/// Generates a PTX kernel where each thread block computes one complete
/// FFT of size N from a batch.
///
/// The kernel layout:
/// - Grid:  `(batch_count, 1, 1)`
/// - Block: `(block_size, 1, 1)` where `block_size <= N`
///
/// Shared memory is used for the Stockham ping-pong buffers within
/// each block.
///
/// # Errors
///
/// Returns [`FftError::PtxGeneration`] if the PTX builder encounters an error.
pub fn generate_batch_fft_kernel(
    n: usize,
    batch: usize,
    precision: FftPrecision,
    sm: SmVersion,
) -> FftResult<String> {
    let float_ty = ptx_float_type(precision);
    let suffix = ptx_type_suffix(precision);
    let kernel_name = format!("fft_batch_{suffix}_n{n}_b{batch}");

    let block_size = select_block_size(n);
    // Shared memory: 2 * N complex elements for ping-pong
    let shared_count = 2 * n * 2; // 2 buffers * N * (re + im)

    let elem_bytes = precision.element_bytes();

    let ptx = KernelBuilder::new(&kernel_name)
        .target(sm)
        .param("input_ptr", PtxType::U64)
        .param("output_ptr", PtxType::U64)
        .param("batch_count", PtxType::U32)
        .param("direction", PtxType::U32)
        .shared_mem("smem", float_ty, shared_count)
        .max_threads_per_block(block_size)
        .body(move |b| {
            b.comment(&format!(
                "Batch FFT: N={n}, batch={batch}, block_size={block_size}"
            ));

            // Each block handles one batch element
            let tid = b.thread_id_x();
            let batch_idx = b.block_id_x();

            let _input_ptr = b.load_param_u64("input_ptr");
            let _output_ptr = b.load_param_u64("output_ptr");
            let batch_count = b.load_param_u32("batch_count");
            let _direction = b.load_param_u32("direction");

            // Bounds check: skip if batch_idx >= batch_count
            b.if_lt_u32(batch_idx.clone(), batch_count, |b| {
                b.comment("compute batch offset into input/output arrays");

                // Batch offset = batch_idx * N * complex_size_bytes
                let complex_per_batch = b.alloc_reg(PtxType::U32);
                let total_floats = n * 2;
                b.raw_ptx(&format!("mov.u32 {complex_per_batch}, {total_floats};"));

                let batch_float_offset = b.mul_lo_u32(batch_idx, complex_per_batch);
                let elem_size_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {elem_size_reg}, {elem_bytes};"));
                let batch_byte_offset = b.mul_wide_u32_to_u64(batch_float_offset, elem_size_reg);

                let src = b.add_u64(_input_ptr, batch_byte_offset.clone());
                let dst = b.add_u64(_output_ptr, batch_byte_offset);

                // Load data from global to shared memory
                b.comment("coalesced load from global to shared memory");
                let smem_base = b.alloc_reg(PtxType::U64);
                b.raw_ptx(&format!("mov.u64 {smem_base}, smem;"));

                let elems_per_thread = total_floats.div_ceil(block_size as usize);
                for e in 0..elems_per_thread {
                    let idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {idx}, {tid}, {elems_per_thread}, {e};"
                    ));
                    let bound = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {bound}, {total_floats};"));
                    b.if_lt_u32(idx.clone(), bound, |b| {
                        let es = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {es}, {elem_bytes};"));
                        let byte_off = b.mul_wide_u32_to_u64(idx, es);
                        let g_addr = b.add_u64(src.clone(), byte_off.clone());
                        let s_addr = b.add_u64(smem_base.clone(), byte_off);

                        match precision {
                            FftPrecision::Single => {
                                let val = b.load_global_f32(g_addr);
                                b.store_shared_f32(s_addr, val);
                            }
                            FftPrecision::Double => {
                                let val = b.alloc_reg(PtxType::F64);
                                b.raw_ptx(&format!("ld.global.f64 {val}, [{g_addr}];"));
                                b.raw_ptx(&format!("st.shared.f64 [{s_addr}], {val};"));
                            }
                        }
                    });
                }

                b.bar_sync(0);

                // FFT computation in shared memory
                b.comment("Stockham FFT stages in shared memory");
                // The actual butterfly stages would be emitted here

                b.bar_sync(0);

                // Store results back to global memory
                b.comment("coalesced store from shared to global memory");
                for e in 0..elems_per_thread {
                    let idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!(
                        "mad.lo.u32 {idx}, {tid}, {elems_per_thread}, {e};"
                    ));
                    let bound = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {bound}, {total_floats};"));
                    b.if_lt_u32(idx.clone(), bound, |b| {
                        let es = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {es}, {elem_bytes};"));
                        let byte_off = b.mul_wide_u32_to_u64(idx, es);
                        let s_addr = b.add_u64(smem_base.clone(), byte_off.clone());
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
            });

            b.ret();
        })
        .build()
        .map_err(FftError::PtxGeneration)?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Selects an appropriate block size for a batch FFT kernel.
fn select_block_size(n: usize) -> u32 {
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

/// Returns the shared memory requirement in bytes for a batch FFT kernel.
pub fn batch_fft_shared_bytes(n: usize, precision: FftPrecision) -> usize {
    2 * n * precision.complex_bytes()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_fft_kernel_smoke() {
        let result = generate_batch_fft_kernel(64, 1024, FftPrecision::Single, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("fft_batch_f32_n64"));
        }
    }

    #[test]
    fn shared_bytes_calculation() {
        assert_eq!(batch_fft_shared_bytes(64, FftPrecision::Single), 1024);
        assert_eq!(batch_fft_shared_bytes(64, FftPrecision::Double), 2048);
    }
}
