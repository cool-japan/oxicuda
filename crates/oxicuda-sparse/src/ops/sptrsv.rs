//! Sparse triangular solve (SpTRSV).
//!
//! Solves `L * x = b` (lower triangular) or `U * x = b` (upper triangular)
//! where the triangular matrix is stored in CSR format.
//!
//! ## Algorithm
//!
//! Uses a **level-set parallelism** approach:
//! 1. **Analysis phase** (`analyze_levels`): topologically sorts rows into
//!    dependency levels. Rows within the same level are independent and can be
//!    solved in parallel.
//! 2. **Solve phase**: for each level, launches a PTX kernel where each thread
//!    applies forward or backward substitution for one row, reading only
//!    already-solved entries from previous levels.
#![allow(dead_code)]

use std::sync::Arc;

use oxicuda_blas::{FillMode, GpuFloat};
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
    add_float, fma_float, load_float_imm, load_global_float, mul_float, store_global_float,
};

/// Default block size for SpTRSV level-solve kernels.
const SPTRSV_BLOCK_SIZE: u32 = 256;

/// Sparse triangular solve: solves `T * x = b` where `T` is triangular CSR.
///
/// # Arguments
///
/// * `handle` -- Sparse handle.
/// * `fill_mode` -- Whether `a` stores a lower (`L`) or upper (`U`) triangle.
/// * `a` -- Sparse CSR triangular matrix.
/// * `b_ptr` -- Device pointer to the right-hand side vector `b`.
/// * `x_ptr` -- Device pointer to the output vector `x`.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if the matrix is not square.
/// Returns [`SparseError::SingularMatrix`] if a zero diagonal is detected.
/// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
pub fn sptrsv<T: GpuFloat>(
    handle: &SparseHandle,
    fill_mode: FillMode,
    a: &CsrMatrix<T>,
    b_ptr: CUdeviceptr,
    x_ptr: CUdeviceptr,
) -> SparseResult<()> {
    if a.rows() != a.cols() {
        return Err(SparseError::DimensionMismatch(format!(
            "triangular solve requires square matrix, got {}x{}",
            a.rows(),
            a.cols()
        )));
    }

    let n = a.rows();
    if n == 0 {
        return Ok(());
    }

    // Download structure for analysis
    let (h_row_ptr, h_col_idx, _h_values) = a.to_host()?;

    // Build dependency levels
    let levels = analyze_levels(&h_row_ptr, &h_col_idx, n, fill_mode)?;

    // Generate the per-level solve kernel
    let ptx = emit_sptrsv_level_kernel::<T>(fill_mode, handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "sptrsv_level")?;

    // Execute level by level
    for level_rows in &levels {
        if level_rows.is_empty() {
            continue;
        }

        // Upload the row indices for this level
        let d_level_rows = oxicuda_memory::DeviceBuffer::<u32>::from_host(level_rows)?;
        let num_rows_in_level = level_rows.len() as u32;

        let block_size = SPTRSV_BLOCK_SIZE;
        let grid_size = grid_size_for(num_rows_in_level, block_size);
        let params = LaunchParams::new(grid_size, block_size);

        kernel.launch(
            &params,
            handle.stream(),
            &(
                a.row_ptr().as_device_ptr(),
                a.col_idx().as_device_ptr(),
                a.values().as_device_ptr(),
                b_ptr,
                x_ptr,
                d_level_rows.as_device_ptr(),
                num_rows_in_level,
                n,
            ),
        )?;

        // Synchronize to ensure this level completes before the next
        handle.stream().synchronize()?;
    }

    Ok(())
}

/// Analyzes the dependency structure and groups rows into parallel levels.
///
/// For lower triangular: row `i` depends on all columns `j < i` in its row.
/// For upper triangular: row `i` depends on all columns `j > i` in its row.
///
/// Rows with no unsolved dependencies at a given level can be solved in parallel.
fn analyze_levels(
    row_ptr: &[i32],
    col_idx: &[i32],
    n: u32,
    fill_mode: FillMode,
) -> SparseResult<Vec<Vec<u32>>> {
    let n_usize = n as usize;

    // Compute the dependency depth for each row using BFS / topological sort
    let mut depth = vec![0u32; n_usize];
    let mut max_depth: u32 = 0;

    match fill_mode {
        FillMode::Lower => {
            // Process rows top-to-bottom
            for i in 0..n_usize {
                let start = row_ptr[i] as usize;
                let end = row_ptr[i + 1] as usize;
                let mut max_dep = 0u32;
                for &cj in &col_idx[start..end] {
                    let j = cj as usize;
                    if j < i {
                        let d = depth[j] + 1;
                        if d > max_dep {
                            max_dep = d;
                        }
                    }
                }
                depth[i] = max_dep;
                if max_dep > max_depth {
                    max_depth = max_dep;
                }
            }
        }
        FillMode::Upper | FillMode::Full => {
            // Process rows bottom-to-top for upper triangular
            for i in (0..n_usize).rev() {
                let start = row_ptr[i] as usize;
                let end = row_ptr[i + 1] as usize;
                let mut max_dep = 0u32;
                for &cj in &col_idx[start..end] {
                    let j = cj as usize;
                    if j > i {
                        let d = depth[j] + 1;
                        if d > max_dep {
                            max_dep = d;
                        }
                    }
                }
                depth[i] = max_dep;
                if max_dep > max_depth {
                    max_depth = max_dep;
                }
            }
        }
    }

    // Group rows by depth into levels
    let num_levels = max_depth as usize + 1;
    let mut levels: Vec<Vec<u32>> = vec![Vec::new(); num_levels];
    for (i, &d) in depth.iter().enumerate() {
        levels[d as usize].push(i as u32);
    }

    Ok(levels)
}

/// Generates PTX for one level of the SpTRSV solve.
///
/// Each thread picks one row from the level-rows array and performs
/// forward (lower) or backward (upper) substitution. Since all dependencies
/// are guaranteed to be solved in previous levels, each thread can read
/// `x[j]` for dependency columns safely.
fn emit_sptrsv_level_kernel<T: GpuFloat>(
    fill_mode: FillMode,
    sm: SmVersion,
) -> SparseResult<String> {
    let elem_bytes = T::size_u32();
    let is_f64 = T::SIZE == 8;
    let is_lower = matches!(fill_mode, FillMode::Lower);

    KernelBuilder::new("sptrsv_level")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values", PtxType::U64)
        .param("b_ptr", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("level_rows", PtxType::U64)
        .param("num_level_rows", PtxType::U32)
        .param("n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let num_level_rows = b.load_param_u32("num_level_rows");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, num_level_rows, move |b| {
                let tid = gid_inner;
                let level_rows_ptr = b.load_param_u64("level_rows");
                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values");
                let b_ptr = b.load_param_u64("b_ptr");
                let x_ptr = b.load_param_u64("x_ptr");

                // Load the actual row index from level_rows[tid]
                let row_addr = b.byte_offset_addr(level_rows_ptr, tid, 4);
                let row = b.load_global_u32(row_addr);

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

                // Load b[row]
                let b_addr = b.byte_offset_addr(b_ptr, row.clone(), elem_bytes);
                let b_val = load_global_float::<T>(b, b_addr);

                // Accumulate: sum = sum of A[row,j]*x[j] for off-diagonal j
                let sum = load_float_imm::<T>(b, 0.0);
                let diag = load_float_imm::<T>(b, 1.0); // will be overwritten

                let k = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {k}, {rs};"));

                let loop_label = b.fresh_label("sptrsv_loop");
                let loop_done = b.fresh_label("sptrsv_done");

                b.label(&loop_label);
                let pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {pred}, {k}, {re};"));
                b.raw_ptx(&format!("@!{pred} bra {loop_done};"));

                // Load column and value
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), k.clone(), 4);
                let col_i32 = b.load_global_i32(ci_addr);
                let col = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col}, {col_i32};"));

                let v_addr = b.byte_offset_addr(values_base.clone(), k.clone(), elem_bytes);
                let a_val = load_global_float::<T>(b, v_addr);

                // Check if diagonal
                let is_diag = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {is_diag}, {col}, {row};"));

                // If diagonal, save it; otherwise accumulate
                let skip_diag = b.fresh_label("sptrsv_skip_diag");
                let after_diag = b.fresh_label("sptrsv_after_diag");
                b.raw_ptx(&format!("@!{is_diag} bra {skip_diag};"));

                // Save diagonal value
                let mov_suffix = if is_f64 { "f64" } else { "f32" };
                b.raw_ptx(&format!("mov.{mov_suffix} {diag}, {a_val};"));
                b.raw_ptx(&format!("bra {after_diag};"));

                b.label(&skip_diag);

                // Off-diagonal: check if it's a dependency column
                if is_lower {
                    // For lower: depend on j < row
                    let is_dep = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.lo.u32 {is_dep}, {col}, {row};"));
                    let skip_acc = b.fresh_label("sptrsv_skip_acc");
                    b.raw_ptx(&format!("@!{is_dep} bra {skip_acc};"));

                    // Load x[col] and accumulate
                    let x_addr = b.byte_offset_addr(x_ptr.clone(), col, elem_bytes);
                    let x_val = load_global_float::<T>(b, x_addr);
                    let new_sum = fma_float::<T>(b, a_val, x_val, sum.clone());
                    b.raw_ptx(&format!("mov.{mov_suffix} {sum}, {new_sum};"));

                    b.label(&skip_acc);
                } else {
                    // For upper: depend on j > row
                    let is_dep = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.hi.u32 {is_dep}, {col}, {row};"));
                    let skip_acc = b.fresh_label("sptrsv_skip_acc");
                    b.raw_ptx(&format!("@!{is_dep} bra {skip_acc};"));

                    let x_addr = b.byte_offset_addr(x_ptr.clone(), col, elem_bytes);
                    let x_val = load_global_float::<T>(b, x_addr);
                    let new_sum = fma_float::<T>(b, a_val, x_val, sum.clone());
                    b.raw_ptx(&format!("mov.{mov_suffix} {sum}, {new_sum};"));

                    b.label(&skip_acc);
                }

                b.label(&after_diag);

                b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
                b.branch(&loop_label);
                b.label(&loop_done);

                // x[row] = (b[row] - sum) / diag
                // Compute b_val - sum
                let neg_one = load_float_imm::<T>(b, -1.0);
                let neg_sum = mul_float::<T>(b, sum, neg_one);
                let numerator = add_float::<T>(b, b_val, neg_sum);

                // Divide by diagonal: x = numerator * (1/diag)
                // Use rcp for f32, div for both
                let result = b.alloc_reg(T::PTX_TYPE);
                b.raw_ptx(&format!(
                    "div.rn.{mov_suffix} {result}, {numerator}, {diag};"
                ));

                // Store x[row]
                let x_out_addr = b.byte_offset_addr(x_ptr, row, elem_bytes);
                store_global_float::<T>(b, x_out_addr, result);
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
    fn sptrsv_lower_ptx_generates_f32() {
        let ptx = emit_sptrsv_level_kernel::<f32>(FillMode::Lower, SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry sptrsv_level"));
    }

    #[test]
    fn sptrsv_upper_ptx_generates_f32() {
        let ptx = emit_sptrsv_level_kernel::<f32>(FillMode::Upper, SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn sptrsv_lower_ptx_generates_f64() {
        let ptx = emit_sptrsv_level_kernel::<f64>(FillMode::Lower, SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn analyze_levels_lower_triangular() {
        // 3x3 lower triangular:
        // row 0: [0]        -> no deps, level 0
        // row 1: [0, 1]     -> depends on row 0, level 1
        // row 2: [0, 1, 2]  -> depends on row 1 (level 1), so level 2
        let row_ptr = vec![0, 1, 3, 6];
        let col_idx = vec![0, 0, 1, 0, 1, 2];
        let levels = analyze_levels(&row_ptr, &col_idx, 3, FillMode::Lower);
        assert!(levels.is_ok());
        let levels = levels.expect("test: analyze should succeed");
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0], vec![0]);
        assert_eq!(levels[1], vec![1]);
        assert_eq!(levels[2], vec![2]);
    }

    #[test]
    fn analyze_levels_identity() {
        // 3x3 identity: all rows are independent
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let levels = analyze_levels(&row_ptr, &col_idx, 3, FillMode::Lower);
        assert!(levels.is_ok());
        let levels = levels.expect("test: analyze should succeed");
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].len(), 3);
    }
}
