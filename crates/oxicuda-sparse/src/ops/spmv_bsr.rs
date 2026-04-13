//! BSR SpMV kernel.
//!
//! Computes `y = alpha * A * x + beta * y` where `A` is in Block Sparse Row
//! (BSR) format. The kernel exploits dense sub-blocks for higher arithmetic
//! intensity compared to scalar CSR SpMV.
//!
//! Each thread block handles one block-row. Within a block-row, threads
//! cooperate to multiply the dense `block_dim x block_dim` sub-blocks by
//! the corresponding segments of `x`, accumulating into a shared-memory
//! partial result vector of length `block_dim`.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{SparseError, SparseResult};
use crate::format::BsrMatrix;
use crate::handle::SparseHandle;
use crate::ptx_helpers::{
    add_float, fma_float, load_float_imm, load_global_float, mul_float, reinterpret_bits_to_float,
    store_global_float,
};

/// Maximum threads per block for BSR SpMV.
/// Each thread block handles one block-row, with threads distributed
/// across the dense block elements.
const SPMV_BSR_MAX_BLOCK: u32 = 256;

/// BSR SpMV: `y = alpha * A * x + beta * y`.
///
/// The kernel launches one thread block per block-row. Within each thread
/// block, threads iterate over the non-zero blocks in that block-row,
/// performing dense block-vector multiplication.
///
/// # Arguments
///
/// * `handle` -- Sparse handle providing stream and device context.
/// * `bsr` -- Sparse BSR matrix `A`.
/// * `x` -- Dense input vector of length `A.cols()`.
/// * `y` -- Dense output vector of length `A.rows()`.
/// * `alpha` -- Scalar multiplier for `A * x`.
/// * `beta` -- Scalar multiplier for existing `y`.
///
/// # Errors
///
/// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
/// Returns [`SparseError::Cuda`] on kernel launch failure.
/// Returns [`SparseError::DimensionMismatch`] if vector lengths are wrong.
pub fn spmv_bsr<T: GpuFloat>(
    handle: &SparseHandle,
    bsr: &BsrMatrix<T>,
    x: &DeviceBuffer<T>,
    y: &mut DeviceBuffer<T>,
    alpha: T,
    beta: T,
) -> SparseResult<()> {
    if bsr.rows() == 0 || bsr.cols() == 0 {
        return Ok(());
    }

    if x.len() < bsr.cols() as usize {
        return Err(SparseError::DimensionMismatch(format!(
            "x length ({}) must be >= cols ({})",
            x.len(),
            bsr.cols()
        )));
    }
    if y.len() < bsr.rows() as usize {
        return Err(SparseError::DimensionMismatch(format!(
            "y length ({}) must be >= rows ({})",
            y.len(),
            bsr.rows()
        )));
    }

    let block_dim = bsr.block_dim();
    let block_rows = bsr.block_rows();

    // Threads per block: at least block_dim threads (one per row in the block),
    // rounded up to a multiple of 32 (warp size), capped at SPMV_BSR_MAX_BLOCK.
    let threads_per_block = (block_dim.div_ceil(32) * 32).min(SPMV_BSR_MAX_BLOCK);

    let ptx = emit_spmv_bsr::<T>(handle.sm_version(), block_dim)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "spmv_bsr")?;

    // One thread block per block-row
    let params = LaunchParams::new(block_rows, threads_per_block);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            bsr.row_ptr().as_device_ptr(),
            bsr.col_idx().as_device_ptr(),
            bsr.values().as_device_ptr(),
            x.as_device_ptr(),
            y.as_device_ptr(),
            alpha.to_bits_u64(),
            beta.to_bits_u64(),
            bsr.rows(),
            block_dim,
        ),
    )?;

    Ok(())
}

/// Generates PTX for BSR SpMV.
///
/// Each thread block handles one block-row. Thread `tid` within the block is
/// responsible for the `tid`-th row within the dense block (if `tid < block_dim`).
/// For each non-zero block in the block-row, the thread computes a dot product
/// of its row of the dense block with the corresponding segment of `x`.
fn emit_spmv_bsr<T: GpuFloat>(sm: SmVersion, _block_dim: u32) -> SparseResult<String> {
    let elem_bytes = T::size_u32();
    let is_f64 = T::SIZE == 8;

    KernelBuilder::new("spmv_bsr")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values_ptr", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("beta_bits", PtxType::U64)
        .param("num_rows", PtxType::U32)
        .param("block_dim", PtxType::U32)
        .body(move |b| {
            // Each thread block handles one block-row.
            // blockIdx.x = block-row index.
            // threadIdx.x = local row within the block (if < block_dim).
            let block_row = b.block_id_x();
            let tid = b.thread_id_x();
            let block_dim_reg = b.load_param_u32("block_dim");

            // Only threads with tid < block_dim participate
            let tid_inner = tid.clone();
            let block_row_inner = block_row.clone();
            b.if_lt_u32(tid, block_dim_reg, move |b| {
                let tid = tid_inner;
                let block_row = block_row_inner;
                let block_dim_reg = b.load_param_u32("block_dim");

                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values_ptr");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");
                let alpha_bits = b.load_param_u64("alpha_bits");
                let beta_bits = b.load_param_u64("beta_bits");

                let alpha = reinterpret_bits_to_float::<T>(b, alpha_bits);
                let beta = reinterpret_bits_to_float::<T>(b, beta_bits);

                // Load row_ptr[block_row] and row_ptr[block_row+1]
                let rp_addr = b.byte_offset_addr(row_ptr_base.clone(), block_row.clone(), 4);
                let blk_start_i32 = b.load_global_i32(rp_addr);
                let blk_start = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {blk_start}, {blk_start_i32};"));

                let block_row_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {block_row_plus_1}, {block_row}, 1;"));
                let rp_addr_next = b.byte_offset_addr(row_ptr_base, block_row_plus_1, 4);
                let blk_end_i32 = b.load_global_i32(rp_addr_next);
                let blk_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {blk_end}, {blk_end_i32};"));

                // Initialize accumulator for this thread's row
                let acc = load_float_imm::<T>(b, 0.0);

                // block_dim^2 = elements per dense block
                let blk_sq = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mul.lo.u32 {blk_sq}, {block_dim_reg}, {block_dim_reg};"
                ));

                // Loop over non-zero blocks: blk_idx = blk_start .. blk_end
                let blk_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {blk_idx}, {blk_start};"));

                let blk_loop = b.fresh_label("bsr_blk_loop");
                let blk_done = b.fresh_label("bsr_blk_done");

                b.label(&blk_loop);
                let pred_blk = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {pred_blk}, {blk_idx}, {blk_end};"));
                b.raw_ptx(&format!("@!{pred_blk} bra {blk_done};"));

                // Load block column index
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), blk_idx.clone(), 4);
                let blk_col_i32 = b.load_global_i32(ci_addr);
                let blk_col = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {blk_col}, {blk_col_i32};"));

                // Base offset of this block in values array:
                // values_offset = blk_idx * block_dim^2
                let val_block_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mul.lo.u32 {val_block_offset}, {blk_idx}, {blk_sq};"
                ));

                // Row offset within block: tid * block_dim
                let row_in_block_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mul.lo.u32 {row_in_block_offset}, {tid}, {block_dim_reg};"
                ));

                // x column base: blk_col * block_dim
                let x_col_base = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mul.lo.u32 {x_col_base}, {blk_col}, {block_dim_reg};"
                ));

                // Inner loop over block columns: j = 0 .. block_dim
                let j = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {j}, 0;"));

                let inner_loop = b.fresh_label("bsr_inner");
                let inner_done = b.fresh_label("bsr_inner_done");

                b.label(&inner_loop);
                let pred_j = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {pred_j}, {j}, {block_dim_reg};"));
                b.raw_ptx(&format!("@!{pred_j} bra {inner_done};"));

                // values index = val_block_offset + row_in_block_offset + j
                let val_flat = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "add.u32 {val_flat}, {val_block_offset}, {row_in_block_offset};"
                ));
                let val_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {val_idx}, {val_flat}, {j};"));

                let v_addr = b.byte_offset_addr(values_base.clone(), val_idx, elem_bytes);
                let val = load_global_float::<T>(b, v_addr);

                // x index = x_col_base + j
                let x_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {x_idx}, {x_col_base}, {j};"));

                let x_addr = b.byte_offset_addr(x_ptr.clone(), x_idx, elem_bytes);
                let x_val = load_global_float::<T>(b, x_addr);

                // acc += val * x_val
                let new_acc = fma_float::<T>(b, val, x_val, acc.clone());
                let mov_suffix = if is_f64 { "f64" } else { "f32" };
                b.raw_ptx(&format!("mov.{mov_suffix} {acc}, {new_acc};"));

                // j++
                b.raw_ptx(&format!("add.u32 {j}, {j}, 1;"));
                b.branch(&inner_loop);
                b.label(&inner_done);

                // blk_idx++
                b.raw_ptx(&format!("add.u32 {blk_idx}, {blk_idx}, 1;"));
                b.branch(&blk_loop);
                b.label(&blk_done);

                // Compute global row index = block_row * block_dim + tid
                let global_row = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {global_row}, {block_row}, {block_dim_reg}, {tid};"
                ));

                // Write y = alpha * acc + beta * y_old
                let y_addr = b.byte_offset_addr(y_ptr, global_row, elem_bytes);
                let y_old = load_global_float::<T>(b, y_addr.clone());

                let alpha_acc = mul_float::<T>(b, alpha, acc);
                let beta_y = mul_float::<T>(b, beta, y_old);
                let result = add_float::<T>(b, alpha_acc, beta_y);

                store_global_float::<T>(b, y_addr, result);
            });

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spmv_bsr_ptx_generates_f32() {
        let ptx = emit_spmv_bsr::<f32>(SmVersion::Sm80, 2);
        assert!(ptx.is_ok());
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains(".entry spmv_bsr"));
        assert!(ptx_text.contains(".target sm_80"));
    }

    #[test]
    fn spmv_bsr_ptx_generates_f64() {
        let ptx = emit_spmv_bsr::<f64>(SmVersion::Sm80, 4);
        assert!(ptx.is_ok());
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains(".entry spmv_bsr"));
    }

    #[test]
    fn spmv_bsr_ptx_block_sizes() {
        // Various common block sizes should generate valid PTX
        for bd in [2, 4, 8] {
            let ptx = emit_spmv_bsr::<f32>(SmVersion::Sm80, bd);
            assert!(ptx.is_ok(), "BSR PTX generation failed for block_dim={bd}");
        }
    }

    #[test]
    fn spmv_bsr_threads_per_block() {
        // Verify threads_per_block calculation
        for block_dim in [2u32, 4, 8, 16, 32, 64] {
            let threads = (block_dim.div_ceil(32) * 32).min(SPMV_BSR_MAX_BLOCK);
            assert!(threads >= block_dim);
            assert_eq!(threads % 32, 0);
            assert!(threads <= SPMV_BSR_MAX_BLOCK);
        }
    }

    #[test]
    fn spmv_bsr_ptx_contains_block_multiply() {
        let ptx = emit_spmv_bsr::<f32>(SmVersion::Sm80, 4);
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        // Should contain nested loop structure (block multiply)
        assert!(ptx_text.contains("bsr_blk_loop"));
        assert!(ptx_text.contains("bsr_inner"));
    }
}
