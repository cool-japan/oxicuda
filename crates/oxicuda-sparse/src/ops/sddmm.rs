//! Sampled Dense-Dense Matrix Multiply (SDDMM).
//!
//! Computes `C_ij = alpha * (A @ B)_ij * spy(S)_ij + beta * S_ij`
//! where the result is only computed at positions where the sparse matrix `S`
//! has non-zero entries. This is a key primitive in graph neural networks and
//! sparse attention mechanisms.
//!
//! ## Strategy
//!
//! Each thread handles one non-zero entry of `S`. For that entry at position
//! `(row, col)`, the thread computes the dot product `A[row, :] . B[:, col]`
//! (inner dimension `K`), scales by `alpha`, and blends with the existing
//! value via `beta`.
#![allow(dead_code)]

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{SparseError, SparseResult};
use crate::format::CsrMatrix;
use crate::handle::SparseHandle;
use crate::ptx_helpers::{
    add_float, fma_float, load_float_imm, load_global_float, mul_float, reinterpret_bits_to_float,
    store_global_float,
};

/// Default block size for SDDMM kernel.
const SDDMM_BLOCK_SIZE: u32 = 256;

/// Sampled Dense-Dense Matrix Multiply.
///
/// Computes `S_ij = alpha * dot(A[i,:], B[:,j]) + beta * S_ij` for each
/// non-zero position `(i, j)` in the sparse matrix `S`.
///
/// # Arguments
///
/// * `handle` -- Sparse handle.
/// * `alpha` -- Scalar multiplier for the dense product.
/// * `a_ptr` -- Device pointer to dense matrix `A` (row-major, shape `m x k`).
/// * `a_rows` -- Number of rows of `A`.
/// * `a_cols` -- Number of columns of `A` (= inner dimension `K`).
/// * `a_ld` -- Leading dimension (row stride) of `A`.
/// * `b_ptr` -- Device pointer to dense matrix `B` (row-major, shape `k x n`).
/// * `b_cols` -- Number of columns of `B`.
/// * `b_ld` -- Leading dimension (row stride) of `B`.
/// * `beta` -- Scalar multiplier for existing `S` values.
/// * `s` -- Sparse CSR matrix `S`, updated in place.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if dimensions are incompatible.
/// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
#[allow(clippy::too_many_arguments)]
pub fn sddmm<T: GpuFloat>(
    handle: &SparseHandle,
    alpha: T,
    a_ptr: CUdeviceptr,
    a_rows: u32,
    a_cols: u32,
    a_ld: u32,
    b_ptr: CUdeviceptr,
    b_cols: u32,
    b_ld: u32,
    beta: T,
    s: &mut CsrMatrix<T>,
) -> SparseResult<()> {
    // Validate dimensions
    if s.rows() != a_rows {
        return Err(SparseError::DimensionMismatch(format!(
            "S.rows ({}) != A.rows ({})",
            s.rows(),
            a_rows
        )));
    }
    if s.cols() != b_cols {
        return Err(SparseError::DimensionMismatch(format!(
            "S.cols ({}) != B.cols ({})",
            s.cols(),
            b_cols
        )));
    }

    if s.nnz() == 0 || a_cols == 0 {
        return Ok(());
    }

    let ptx = emit_sddmm_kernel::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "sddmm")?;

    let block_size = SDDMM_BLOCK_SIZE;
    let grid_size = grid_size_for(s.nnz(), block_size);
    let params = LaunchParams::new(grid_size, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            s.row_ptr().as_device_ptr(),
            s.col_idx().as_device_ptr(),
            s.values().as_device_ptr(),
            a_ptr,
            b_ptr,
            alpha.to_bits_u64(),
            beta.to_bits_u64(),
            s.rows(),
            a_cols,
            a_ld,
            b_ld,
        ),
    )?;

    Ok(())
}

/// Generates PTX for the SDDMM kernel.
///
/// Each thread handles one non-zero of `S`. It identifies the row and column
/// from the CSR structure, then computes the dot product of `A[row, :]` and
/// `B[:, col]` over the inner dimension `K`.
fn emit_sddmm_kernel<T: GpuFloat>(sm: SmVersion) -> SparseResult<String> {
    let elem_bytes = T::size_u32();
    let is_f64 = T::SIZE == 8;

    KernelBuilder::new("sddmm")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values", PtxType::U64)
        .param("a_ptr", PtxType::U64)
        .param("b_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("beta_bits", PtxType::U64)
        .param("m", PtxType::U32)
        .param("k", PtxType::U32)
        .param("a_ld", PtxType::U32)
        .param("b_ld", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let m_param = b.load_param_u32("m");
            let mov_suffix = if is_f64 { "f64" } else { "f32" };

            // We need to find which row this non-zero belongs to.
            // Simple approach: binary search in row_ptr.
            // But for PTX simplicity, we use a linear scan.
            //
            // Actually, we launch one thread per row and iterate over that row's nnz.
            // This is simpler and avoids the row-finding problem.

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, m_param, move |b| {
                let row = gid_inner;
                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values");
                let a_ptr = b.load_param_u64("a_ptr");
                let b_ptr = b.load_param_u64("b_ptr");
                let alpha_bits = b.load_param_u64("alpha_bits");
                let beta_bits = b.load_param_u64("beta_bits");
                let k_param = b.load_param_u32("k");
                let a_ld = b.load_param_u32("a_ld");
                let b_ld = b.load_param_u32("b_ld");

                let alpha = reinterpret_bits_to_float::<T>(b, alpha_bits);
                let beta = reinterpret_bits_to_float::<T>(b, beta_bits);

                // Load row bounds
                let rp_addr = b.byte_offset_addr(row_ptr_base.clone(), row.clone(), 4);
                let rs_i32 = b.load_global_i32(rp_addr);
                let rs = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {rs}, {rs_i32};"));

                let row_p1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_p1}, {row}, 1;"));
                let re_addr = b.byte_offset_addr(row_ptr_base, row_p1, 4);
                let re_i32 = b.load_global_i32(re_addr);
                let re = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {re}, {re_i32};"));

                // For each non-zero in this row
                let nz_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {nz_idx}, {rs};"));

                let nz_loop = b.fresh_label("sddmm_nz_loop");
                let nz_done = b.fresh_label("sddmm_nz_done");

                b.label(&nz_loop);
                let nz_pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {nz_pred}, {nz_idx}, {re};"));
                b.raw_ptx(&format!("@!{nz_pred} bra {nz_done};"));

                // Load column index
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), nz_idx.clone(), 4);
                let col_i32 = b.load_global_i32(ci_addr);
                let col = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col}, {col_i32};"));

                // Compute dot product: A[row, :] . B[:, col]
                let dot = load_float_imm::<T>(b, 0.0);

                let kk = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {kk}, 0;"));

                let k_loop = b.fresh_label("sddmm_k_loop");
                let k_done = b.fresh_label("sddmm_k_done");

                b.label(&k_loop);
                let k_pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {k_pred}, {kk}, {k_param};"));
                b.raw_ptx(&format!("@!{k_pred} bra {k_done};"));

                // A[row, kk] = a_ptr + (row * a_ld + kk) * elem_bytes
                let a_row_off = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {a_row_off}, {row}, {a_ld};"));
                let a_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {a_idx}, {a_row_off}, {kk};"));
                let a_addr = b.byte_offset_addr(a_ptr.clone(), a_idx, elem_bytes);
                let a_val = load_global_float::<T>(b, a_addr);

                // B[kk, col] = b_ptr + (kk * b_ld + col) * elem_bytes
                let b_row_off = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {b_row_off}, {kk}, {b_ld};"));
                let b_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {b_idx}, {b_row_off}, {col};"));
                let b_addr = b.byte_offset_addr(b_ptr.clone(), b_idx, elem_bytes);
                let b_val = load_global_float::<T>(b, b_addr);

                // dot += a_val * b_val
                let new_dot = fma_float::<T>(b, a_val, b_val, dot.clone());
                b.raw_ptx(&format!("mov.{mov_suffix} {dot}, {new_dot};"));

                b.raw_ptx(&format!("add.u32 {kk}, {kk}, 1;"));
                b.branch(&k_loop);
                b.label(&k_done);

                // Load old S value
                let s_v_addr = b.byte_offset_addr(values_base.clone(), nz_idx.clone(), elem_bytes);
                let s_old = load_global_float::<T>(b, s_v_addr.clone());

                // result = alpha * dot + beta * s_old
                let alpha_dot = mul_float::<T>(b, alpha.clone(), dot);
                let beta_s = mul_float::<T>(b, beta.clone(), s_old);
                let result = add_float::<T>(b, alpha_dot, beta_s);

                store_global_float::<T>(b, s_v_addr, result);

                b.raw_ptx(&format!("add.u32 {nz_idx}, {nz_idx}, 1;"));
                b.branch(&nz_loop);
                b.label(&nz_done);
            });

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;

    #[test]
    fn sddmm_ptx_generates_f32() {
        let ptx = emit_sddmm_kernel::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry sddmm"));
    }

    #[test]
    fn sddmm_ptx_generates_f64() {
        let ptx = emit_sddmm_kernel::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn sddmm_ptx_has_correct_target() {
        let ptx = emit_sddmm_kernel::<f32>(SmVersion::Sm75);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".target sm_75"));
    }
}
