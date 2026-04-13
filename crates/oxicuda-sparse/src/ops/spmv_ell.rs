//! ELL-optimized SpMV kernel.
//!
//! Computes `y = alpha * A * x + beta * y` where `A` is in ELLPACK (ELL) format.
//!
//! The ELL format stores data in column-major order for coalesced GPU memory
//! access: `indices[row + col*num_rows]`, `values[row + col*num_rows]`.
//! Each thread handles exactly one row, iterating over at most
//! `max_nnz_per_row` elements. Rows shorter than `max_nnz_per_row` are padded
//! with sentinel column indices (-1) and zero values.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{SparseError, SparseResult};
use crate::format::EllMatrix;
use crate::handle::SparseHandle;
use crate::ptx_helpers::{
    add_float, fma_float, load_float_imm, load_global_float, mul_float, reinterpret_bits_to_float,
    store_global_float,
};

/// Default block size for ELL SpMV.
const SPMV_ELL_BLOCK: u32 = 256;

/// ELL SpMV: `y = alpha * A * x + beta * y`.
///
/// Each thread processes one row of the ELL matrix, iterating over the
/// `max_nnz_per_row` padded entries. Consecutive threads access consecutive
/// memory locations (coalesced access) due to the column-major ELL layout.
///
/// # Arguments
///
/// * `handle` -- Sparse handle providing stream and device context.
/// * `ell` -- Sparse ELL matrix `A`.
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
pub fn spmv_ell<T: GpuFloat>(
    handle: &SparseHandle,
    ell: &EllMatrix<T>,
    x: &DeviceBuffer<T>,
    y: &mut DeviceBuffer<T>,
    alpha: T,
    beta: T,
) -> SparseResult<()> {
    if ell.rows() == 0 || ell.cols() == 0 {
        return Ok(());
    }

    if x.len() < ell.cols() as usize {
        return Err(SparseError::DimensionMismatch(format!(
            "x length ({}) must be >= cols ({})",
            x.len(),
            ell.cols()
        )));
    }
    if y.len() < ell.rows() as usize {
        return Err(SparseError::DimensionMismatch(format!(
            "y length ({}) must be >= rows ({})",
            y.len(),
            ell.rows()
        )));
    }

    let ptx = emit_spmv_ell::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "spmv_ell")?;

    let block_size = SPMV_ELL_BLOCK;
    let grid_size = grid_size_for(ell.rows(), block_size);
    let params = LaunchParams::new(grid_size, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            ell.col_idx().as_device_ptr(),
            ell.values().as_device_ptr(),
            x.as_device_ptr(),
            y.as_device_ptr(),
            alpha.to_bits_u64(),
            beta.to_bits_u64(),
            ell.rows(),
            ell.max_nnz_per_row(),
        ),
    )?;

    Ok(())
}

/// Generates PTX for ELL SpMV (one thread per row, coalesced access).
///
/// The kernel iterates `k = 0..max_nnz_per_row`, loading from column-major
/// layout: `col_idx[k * num_rows + row]` and `values[k * num_rows + row]`.
/// If `col_idx` is the sentinel (-1), the entry is skipped (padded).
fn emit_spmv_ell<T: GpuFloat>(sm: SmVersion) -> SparseResult<String> {
    let elem_bytes = T::size_u32();
    let is_f64 = T::SIZE == 8;

    KernelBuilder::new("spmv_ell")
        .target(sm)
        .param("col_idx_ptr", PtxType::U64)
        .param("values_ptr", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("beta_bits", PtxType::U64)
        .param("num_rows", PtxType::U32)
        .param("max_nnz_per_row", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, num_rows, move |b| {
                let row = gid_inner;
                let col_idx_base = b.load_param_u64("col_idx_ptr");
                let values_base = b.load_param_u64("values_ptr");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");
                let alpha_bits = b.load_param_u64("alpha_bits");
                let beta_bits = b.load_param_u64("beta_bits");
                let num_rows_reg = b.load_param_u32("num_rows");
                let max_nnz = b.load_param_u32("max_nnz_per_row");

                let alpha = reinterpret_bits_to_float::<T>(b, alpha_bits);
                let beta = reinterpret_bits_to_float::<T>(b, beta_bits);

                // Initialize accumulator
                let acc = load_float_imm::<T>(b, 0.0);

                // Loop: k = 0 .. max_nnz_per_row
                let k = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {k}, 0;"));

                let loop_label = b.fresh_label("ell_loop");
                let done_label = b.fresh_label("ell_done");
                let skip_label = b.fresh_label("ell_skip");

                b.label(&loop_label);
                let pred_k = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {pred_k}, {k}, {max_nnz};"));
                b.raw_ptx(&format!("@!{pred_k} bra {done_label};"));

                // Compute index = k * num_rows + row (column-major ELL layout)
                let ell_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {ell_idx}, {k}, {num_rows_reg}, {row};"
                ));

                // Load col_idx[ell_idx] (i32 = 4 bytes)
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), ell_idx.clone(), 4);
                let col = b.load_global_i32(ci_addr);

                // Check sentinel: if col < 0 (i.e. col == -1), skip
                let is_valid = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.s32 {is_valid}, {col}, 0;"));
                b.raw_ptx(&format!("@!{is_valid} bra {skip_label};"));

                let col_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_u32}, {col};"));

                // Load values[ell_idx]
                let v_addr = b.byte_offset_addr(values_base.clone(), ell_idx, elem_bytes);
                let val = load_global_float::<T>(b, v_addr);

                // Load x[col]
                let x_addr = b.byte_offset_addr(x_ptr.clone(), col_u32, elem_bytes);
                let x_val = load_global_float::<T>(b, x_addr);

                // acc += val * x_val
                let new_acc = fma_float::<T>(b, val, x_val, acc.clone());
                let mov_suffix = if is_f64 { "f64" } else { "f32" };
                b.raw_ptx(&format!("mov.{mov_suffix} {acc}, {new_acc};"));

                b.label(&skip_label);

                // k++
                b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
                b.branch(&loop_label);
                b.label(&done_label);

                // Compute y = alpha * acc + beta * y_old
                let y_addr = b.byte_offset_addr(y_ptr, row, elem_bytes);
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
    fn spmv_ell_ptx_generates_f32() {
        let ptx = emit_spmv_ell::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains(".entry spmv_ell"));
        assert!(ptx_text.contains(".target sm_80"));
    }

    #[test]
    fn spmv_ell_ptx_generates_f64() {
        let ptx = emit_spmv_ell::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains(".entry spmv_ell"));
    }

    #[test]
    fn spmv_ell_ptx_contains_sentinel_check() {
        let ptx = emit_spmv_ell::<f32>(SmVersion::Sm80);
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        // The kernel should contain a signed comparison to detect -1 sentinel
        assert!(ptx_text.contains("setp.ge.s32"));
    }

    #[test]
    fn spmv_ell_block_size_is_reasonable() {
        let block = SPMV_ELL_BLOCK;
        assert!(block >= 128);
        assert!(block <= 1024);
        assert_eq!(SPMV_ELL_BLOCK % 32, 0);
    }

    #[test]
    fn spmv_ell_ptx_has_coalesced_pattern() {
        // The kernel should use mad.lo.u32 for computing k*num_rows+row
        let ptx = emit_spmv_ell::<f32>(SmVersion::Sm80);
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains("mad.lo.u32"));
    }
}
