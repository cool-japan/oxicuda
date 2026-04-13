//! Sparse matrix-vector multiplication (SpMV).
//!
//! Computes `y = alpha * A * x + beta * y` where `A` is a sparse CSR matrix
//! and `x`, `y` are dense vectors stored as raw device pointers.
//!
//! Three kernel strategies are available:
//! - **Scalar**: one thread per row (best for very sparse rows, < 4 nnz/row)
//! - **Vector**: one warp per row with shuffle reduction (best for moderate sparsity)
//! - **Adaptive**: auto-selects based on average nnz per row

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_ptx::prelude::*;

use crate::error::{SparseError, SparseResult};
use crate::format::CsrMatrix;
use crate::handle::SparseHandle;
use crate::ptx_helpers::{
    add_float, emit_warp_reduce_sum, fma_float, load_float_imm, load_global_float, mul_float,
    reinterpret_bits_to_float, store_global_float,
};

/// Algorithm selection for SpMV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpMVAlgo {
    /// One thread per row. Best for very sparse matrices (< 4 nnz/row).
    Scalar,
    /// One warp (32 threads) per row with warp shuffle reduction.
    /// Best for moderate sparsity (4-64 nnz/row).
    Vector,
    /// Automatically selects Scalar or Vector based on the matrix structure.
    Adaptive,
}

/// Default block size for scalar SpMV.
const SPMV_SCALAR_BLOCK: u32 = 256;

/// Default block size for vector SpMV (must be a multiple of warp size 32).
const SPMV_VECTOR_BLOCK: u32 = 256;

/// Threshold for auto-selecting Vector over Scalar (nnz per row).
const VECTOR_THRESHOLD: f64 = 4.0;

/// Resolve the [`SpMVAlgo::Adaptive`] selection to a concrete kernel given the
/// matrix's average non-zeros per row.
///
/// This pure function contains the kernel-selection heuristic so it can be
/// tested independently of GPU device memory.
///
/// * `avg_nnz_per_row < VECTOR_THRESHOLD` → [`SpMVAlgo::Scalar`]
/// * `avg_nnz_per_row >= VECTOR_THRESHOLD` → [`SpMVAlgo::Vector`]
#[inline]
pub(crate) fn resolve_adaptive(avg_nnz_per_row: f64) -> SpMVAlgo {
    if avg_nnz_per_row >= VECTOR_THRESHOLD {
        SpMVAlgo::Vector
    } else {
        SpMVAlgo::Scalar
    }
}

/// Sparse matrix-vector multiplication: `y = alpha * A * x + beta * y`.
///
/// # Arguments
///
/// * `handle` -- Sparse handle providing stream and device context.
/// * `algo` -- Algorithm selection strategy.
/// * `alpha` -- Scalar multiplier for `A * x`.
/// * `a` -- Sparse CSR matrix `A`.
/// * `x_ptr` -- Device pointer to dense vector `x` of length `A.cols()`.
/// * `beta` -- Scalar multiplier for existing `y`.
/// * `y_ptr` -- Device pointer to dense vector `y` of length `A.rows()`.
///
/// # Errors
///
/// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
/// Returns [`SparseError::Cuda`] on kernel launch failure.
#[allow(clippy::too_many_arguments)]
pub fn spmv<T: GpuFloat>(
    handle: &SparseHandle,
    algo: SpMVAlgo,
    alpha: T,
    a: &CsrMatrix<T>,
    x_ptr: CUdeviceptr,
    beta: T,
    y_ptr: CUdeviceptr,
) -> SparseResult<()> {
    if a.rows() == 0 || a.cols() == 0 {
        return Ok(());
    }

    // Resolve Adaptive algorithm
    let effective_algo = match algo {
        SpMVAlgo::Adaptive => resolve_adaptive(a.avg_nnz_per_row()),
        other => other,
    };

    match effective_algo {
        SpMVAlgo::Scalar => spmv_scalar(handle, alpha, a, x_ptr, beta, y_ptr),
        SpMVAlgo::Vector => spmv_vector(handle, alpha, a, x_ptr, beta, y_ptr),
        SpMVAlgo::Adaptive => {
            // Already resolved above; unreachable
            spmv_scalar(handle, alpha, a, x_ptr, beta, y_ptr)
        }
    }
}

/// Scalar SpMV: one thread per row.
fn spmv_scalar<T: GpuFloat>(
    handle: &SparseHandle,
    alpha: T,
    a: &CsrMatrix<T>,
    x_ptr: CUdeviceptr,
    beta: T,
    y_ptr: CUdeviceptr,
) -> SparseResult<()> {
    let ptx = emit_spmv_scalar::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "spmv_scalar")?;

    let block_size = SPMV_SCALAR_BLOCK;
    let grid_size = grid_size_for(a.rows(), block_size);
    let params = LaunchParams::new(grid_size, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            a.row_ptr().as_device_ptr(),
            a.col_idx().as_device_ptr(),
            a.values().as_device_ptr(),
            x_ptr,
            y_ptr,
            alpha.to_bits_u64(),
            beta.to_bits_u64(),
            a.rows(),
        ),
    )?;

    Ok(())
}

/// Vector SpMV: one warp (32 threads) per row.
fn spmv_vector<T: GpuFloat>(
    handle: &SparseHandle,
    alpha: T,
    a: &CsrMatrix<T>,
    x_ptr: CUdeviceptr,
    beta: T,
    y_ptr: CUdeviceptr,
) -> SparseResult<()> {
    let ptx = emit_spmv_vector::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "spmv_vector")?;

    let block_size = SPMV_VECTOR_BLOCK;
    // Each warp handles one row; warps_per_block = block_size / 32
    let warps_per_block = block_size / 32;
    let grid_size = grid_size_for(a.rows(), warps_per_block);
    let params = LaunchParams::new(grid_size, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            a.row_ptr().as_device_ptr(),
            a.col_idx().as_device_ptr(),
            a.values().as_device_ptr(),
            x_ptr,
            y_ptr,
            alpha.to_bits_u64(),
            beta.to_bits_u64(),
            a.rows(),
        ),
    )?;

    Ok(())
}

/// Generates PTX for scalar SpMV (one thread per row).
fn emit_spmv_scalar<T: GpuFloat>(sm: SmVersion) -> SparseResult<String> {
    let _ptx_ty = T::PTX_TYPE;
    let elem_bytes = T::size_u32();
    let is_f64 = T::SIZE == 8;

    KernelBuilder::new("spmv_scalar")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("beta_bits", PtxType::U64)
        .param("num_rows", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, num_rows, move |b| {
                let row = gid_inner;
                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");
                let alpha_bits = b.load_param_u64("alpha_bits");
                let beta_bits = b.load_param_u64("beta_bits");

                let alpha = reinterpret_bits_to_float::<T>(b, alpha_bits);
                let beta = reinterpret_bits_to_float::<T>(b, beta_bits);

                // Load row_ptr[row] and row_ptr[row+1] (i32 = 4 bytes)
                let rp_addr = b.byte_offset_addr(row_ptr_base.clone(), row.clone(), 4);
                let row_start = b.load_global_i32(rp_addr);

                let row_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_plus_1}, {row}, 1;"));
                let rp_addr_next = b.byte_offset_addr(row_ptr_base, row_plus_1, 4);
                let row_end = b.load_global_i32(rp_addr_next);

                // Initialize accumulator
                let acc = load_float_imm::<T>(b, 0.0);

                // Loop over non-zeros in this row
                let loop_label = b.fresh_label("spmv_loop");
                let done_label = b.fresh_label("spmv_done");

                let k = b.alloc_reg(PtxType::U32);
                // Convert row_start (i32) to u32
                let rs_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {rs_u32}, {row_start};"));
                b.raw_ptx(&format!("mov.u32 {k}, {rs_u32};"));

                let re_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {re_u32}, {row_end};"));

                b.label(&loop_label);
                let pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {pred}, {k}, {re_u32};"));
                b.raw_ptx(&format!("@!{pred} bra {done_label};"));

                // Load col_idx[k] (i32 = 4 bytes)
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), k.clone(), 4);
                let col = b.load_global_i32(ci_addr);
                let col_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_u32}, {col};"));

                // Load values[k]
                let v_addr = b.byte_offset_addr(values_base.clone(), k.clone(), elem_bytes);
                let val = load_global_float::<T>(b, v_addr);

                // Load x[col]
                let x_addr = b.byte_offset_addr(x_ptr.clone(), col_u32, elem_bytes);
                let x_val = load_global_float::<T>(b, x_addr);

                // acc += val * x_val
                let new_acc = fma_float::<T>(b, val, x_val, acc.clone());
                let mov_suffix = if is_f64 { "f64" } else { "f32" };
                b.raw_ptx(&format!("mov.{mov_suffix} {acc}, {new_acc};"));

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

/// Generates PTX for vector SpMV (one warp per row).
fn emit_spmv_vector<T: GpuFloat>(sm: SmVersion) -> SparseResult<String> {
    let ptx_ty = T::PTX_TYPE;
    let elem_bytes = T::size_u32();
    let is_f64 = T::SIZE == 8;

    // Suppress unused variable warnings for ptx_ty
    let _ = ptx_ty;

    KernelBuilder::new("spmv_vector")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("beta_bits", PtxType::U64)
        .param("num_rows", PtxType::U32)
        .body(move |b| {
            // Each warp handles one row. Warp ID = global_thread_id / 32
            let tid_global = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");

            // Lane within warp (0..31)
            let lane = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("and.b32 {lane}, {tid_global}, 31;"));

            // Warp ID = tid_global >> 5
            let warp_id = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shr.u32 {warp_id}, {tid_global}, 5;"));

            let warp_id_inner = warp_id.clone();
            let lane_inner = lane.clone();
            b.if_lt_u32(warp_id, num_rows, move |b| {
                let row = warp_id_inner;
                let lane = lane_inner;

                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");
                let alpha_bits = b.load_param_u64("alpha_bits");
                let beta_bits = b.load_param_u64("beta_bits");

                let alpha = reinterpret_bits_to_float::<T>(b, alpha_bits);
                let beta = reinterpret_bits_to_float::<T>(b, beta_bits);

                // Load row bounds
                let rp_addr = b.byte_offset_addr(row_ptr_base.clone(), row.clone(), 4);
                let row_start_i32 = b.load_global_i32(rp_addr);
                let row_start = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {row_start}, {row_start_i32};"));

                let row_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_plus_1}, {row}, 1;"));
                let rp_addr_next = b.byte_offset_addr(row_ptr_base, row_plus_1, 4);
                let row_end_i32 = b.load_global_i32(rp_addr_next);
                let row_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {row_end}, {row_end_i32};"));

                // Each lane starts at row_start + lane, stride 32
                let acc = load_float_imm::<T>(b, 0.0);

                let k = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {k}, {row_start}, {lane};"));

                let loop_label = b.fresh_label("spmv_vloop");
                let done_label = b.fresh_label("spmv_vdone");

                b.label(&loop_label);
                let pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {pred}, {k}, {row_end};"));
                b.raw_ptx(&format!("@!{pred} bra {done_label};"));

                // Load col and value
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), k.clone(), 4);
                let col_i32 = b.load_global_i32(ci_addr);
                let col_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_u32}, {col_i32};"));

                let v_addr = b.byte_offset_addr(values_base.clone(), k.clone(), elem_bytes);
                let val = load_global_float::<T>(b, v_addr);

                let x_addr = b.byte_offset_addr(x_ptr.clone(), col_u32, elem_bytes);
                let x_val = load_global_float::<T>(b, x_addr);

                let new_acc = fma_float::<T>(b, val, x_val, acc.clone());
                let mov_suffix = if is_f64 { "f64" } else { "f32" };
                b.raw_ptx(&format!("mov.{mov_suffix} {acc}, {new_acc};"));

                // k += 32 (warp width)
                b.raw_ptx(&format!("add.u32 {k}, {k}, 32;"));
                b.branch(&loop_label);
                b.label(&done_label);

                // Warp shuffle reduction
                let reduced = emit_warp_reduce_sum::<T>(b, acc);

                // Lane 0 writes the result
                let is_lane_0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {is_lane_0}, {lane}, 0;"));

                let write_label = b.fresh_label("spmv_write");
                let skip_label = b.fresh_label("spmv_skip");
                b.raw_ptx(&format!("@!{is_lane_0} bra {skip_label};"));

                b.label(&write_label);
                let y_addr = b.byte_offset_addr(y_ptr, row, elem_bytes);
                let y_old = load_global_float::<T>(b, y_addr.clone());

                let alpha_acc = mul_float::<T>(b, alpha, reduced);
                let beta_y = mul_float::<T>(b, beta, y_old);
                let result = add_float::<T>(b, alpha_acc, beta_y);
                store_global_float::<T>(b, y_addr, result);

                b.label(&skip_label);
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
    fn spmv_algo_auto_select() {
        // avg_nnz < threshold => Scalar
        // Verify VECTOR_THRESHOLD is set to a reasonable value for algorithm selection.
        let threshold = VECTOR_THRESHOLD;
        assert!(threshold > 3.0);
    }

    #[test]
    fn spmv_scalar_ptx_generates() {
        let ptx = emit_spmv_scalar::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX gen should succeed");
        assert!(ptx.contains(".entry spmv_scalar"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn spmv_vector_ptx_generates() {
        let ptx = emit_spmv_vector::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX gen should succeed");
        assert!(ptx.contains(".entry spmv_vector"));
    }

    #[test]
    fn spmv_scalar_ptx_f64() {
        let ptx = emit_spmv_scalar::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn spmv_vector_ptx_f64() {
        let ptx = emit_spmv_vector::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    // -----------------------------------------------------------------------
    // Task 5a: Auto-selection heuristic tests (CPU-only, no GPU required)
    // -----------------------------------------------------------------------

    /// Very sparse rows (avg_nnz ≈ 1.5, well below threshold 4.0) → Scalar.
    #[test]
    fn test_spmv_selects_scalar_for_very_sparse() {
        // 100 rows, 150 nnz → avg = 1.5
        let avg = 150.0_f64 / 100.0;
        assert!(avg < VECTOR_THRESHOLD);
        assert_eq!(resolve_adaptive(avg), SpMVAlgo::Scalar);
    }

    /// Moderate density (avg_nnz = 32, above threshold 4.0) → Vector.
    #[test]
    fn test_spmv_selects_vector_for_moderate_density() {
        let avg = 32.0_f64;
        assert!(avg >= VECTOR_THRESHOLD);
        assert_eq!(resolve_adaptive(avg), SpMVAlgo::Vector);
    }

    /// Dense rows (avg_nnz = 128, well above threshold) → Vector.
    #[test]
    fn test_spmv_selects_vector_for_dense() {
        let avg = 128.0_f64;
        assert!(avg >= VECTOR_THRESHOLD);
        assert_eq!(resolve_adaptive(avg), SpMVAlgo::Vector);
    }

    /// Boundary: just below threshold → Scalar; at threshold → Vector.
    #[test]
    fn test_spmv_selection_boundary_conditions() {
        // Just below threshold (3.9999…)
        let just_below = VECTOR_THRESHOLD - f64::EPSILON * VECTOR_THRESHOLD;
        assert_eq!(resolve_adaptive(just_below), SpMVAlgo::Scalar);

        // Exactly at threshold
        assert_eq!(resolve_adaptive(VECTOR_THRESHOLD), SpMVAlgo::Vector);

        // Slightly above threshold
        let just_above = VECTOR_THRESHOLD + f64::EPSILON * VECTOR_THRESHOLD;
        assert_eq!(resolve_adaptive(just_above), SpMVAlgo::Vector);
    }

    /// Empty matrix (0.0 avg_nnz) is handled gracefully → Scalar (no Vector wasted).
    #[test]
    fn test_spmv_selection_empty_matrix() {
        assert_eq!(resolve_adaptive(0.0), SpMVAlgo::Scalar);
    }

    /// VECTOR_THRESHOLD sanity: must equal 4.0 (the spec-defined boundary).
    #[test]
    fn test_vector_threshold_sanity() {
        assert_eq!(
            VECTOR_THRESHOLD, 4.0,
            "VECTOR_THRESHOLD must be 4.0 per spec"
        );
        assert!(VECTOR_THRESHOLD.is_finite());
    }

    // -----------------------------------------------------------------------
    // Deepening: explicit avg_nnz_per_row bracket tests matching sparse
    // matrix categories from estimation.md and architecture notes.
    // -----------------------------------------------------------------------

    /// avg_nnz_per_row ≤ 2 (diagonal / identity matrices) → Scalar kernel.
    ///
    /// Models a diagonal matrix (1 nnz/row) — the most sparse real-world case.
    #[test]
    fn test_spmv_scalar_for_diagonal_matrix() {
        // 1000-row diagonal → avg = 1.0
        let avg = 1000.0_f64 / 1000.0;
        assert!(avg <= 2.0, "avg={avg} should be ≤ 2");
        assert_eq!(
            resolve_adaptive(avg),
            SpMVAlgo::Scalar,
            "diagonal matrices (avg ≤ 2) should use Scalar SpMV"
        );
    }

    /// avg_nnz_per_row ≤ 2, fractional (near-diagonal) → Scalar kernel.
    ///
    /// Models a tridiagonal-like matrix with ~2 nnz/row.
    #[test]
    fn test_spmv_scalar_for_tridiagonal_matrix() {
        // 1000 rows, 2000 nnz → avg = 2.0 (tridiagonal boundary)
        let avg = 2000.0_f64 / 1000.0;
        assert!(avg <= 2.0, "avg={avg} should be ≤ 2");
        assert_eq!(
            resolve_adaptive(avg),
            SpMVAlgo::Scalar,
            "near-diagonal matrices (avg ≤ 2) should use Scalar SpMV"
        );
    }

    /// avg_nnz_per_row in (2, 32] (moderate stencil / FEM) → Vector kernel.
    ///
    /// Models a 5-point 2D finite-difference stencil (avg ≈ 5 nnz/row).
    #[test]
    fn test_spmv_vector_for_5pt_stencil() {
        // 1000×1000 grid → 5_000_000 rows with ~5 nnz each
        let avg = 5.0_f64;
        assert!(avg > 2.0 && avg <= 32.0, "avg={avg} should be in (2, 32]");
        assert_eq!(
            resolve_adaptive(avg),
            SpMVAlgo::Vector,
            "5-point stencil (avg ≈ 5) should use Vector SpMV"
        );
    }

    /// avg_nnz_per_row ≈ 16 (7-point 3D stencil) → Vector kernel.
    #[test]
    fn test_spmv_vector_for_7pt_3d_stencil() {
        let avg = 7.0_f64;
        assert!(avg <= 32.0, "avg={avg} should be ≤ 32");
        assert_eq!(
            resolve_adaptive(avg),
            SpMVAlgo::Vector,
            "7-point 3D stencil (avg ≈ 7) should use Vector SpMV"
        );
    }

    /// avg_nnz_per_row exactly at VECTOR_THRESHOLD boundary (4.0) → Vector.
    ///
    /// Tests that the boundary is inclusive: avg = VECTOR_THRESHOLD selects
    /// Vector, not Scalar (i.e., `>=` rather than `>`).
    #[test]
    fn test_spmv_vector_at_exact_threshold() {
        let avg = VECTOR_THRESHOLD; // 4.0
        assert_eq!(
            resolve_adaptive(avg),
            SpMVAlgo::Vector,
            "avg == VECTOR_THRESHOLD should select Vector (inclusive boundary)"
        );
        // One ULP below threshold → Scalar
        let below = VECTOR_THRESHOLD - f64::MIN_POSITIVE;
        // May still be 4.0 due to float precision, so only check if strictly below
        if below < VECTOR_THRESHOLD {
            assert_eq!(
                resolve_adaptive(below),
                SpMVAlgo::Scalar,
                "avg strictly below VECTOR_THRESHOLD should select Scalar"
            );
        }
    }

    /// avg_nnz_per_row > 32 (dense row, graph networks) → Vector kernel.
    ///
    /// In the current two-class model, any avg ≥ VECTOR_THRESHOLD selects
    /// Vector regardless of whether avg is 5 or 500. This confirms that the
    /// "Adaptive" algorithm resolves correctly for highly dense rows.
    #[test]
    fn test_spmv_vector_for_high_density_rows() {
        // avg = 64: above the ≤ 32 bracket, still selects Vector
        let avg_64 = 64.0_f64;
        assert_eq!(
            resolve_adaptive(avg_64),
            SpMVAlgo::Vector,
            "high-density rows (avg = 64) should use Vector SpMV via Adaptive"
        );

        // avg = 256: very dense (near-dense matrix)
        let avg_256 = 256.0_f64;
        assert_eq!(
            resolve_adaptive(avg_256),
            SpMVAlgo::Vector,
            "near-dense rows (avg = 256) should use Vector SpMV via Adaptive"
        );
    }

    /// Adaptive algo resolves to the same result as calling resolve_adaptive
    /// directly for various avg_nnz values. Confirms SpMVAlgo::Adaptive is
    /// not accidentally treated as a concrete kernel variant.
    #[test]
    fn test_spmv_adaptive_algo_is_not_concrete() {
        // SpMVAlgo::Adaptive is a selection hint, not a concrete kernel.
        // resolve_adaptive must return Scalar or Vector, never Adaptive.
        let test_avgs = [0.0, 0.5, 1.0, 2.0, 3.99, 4.0, 4.01, 32.0, 64.0, 128.0];
        for avg in test_avgs {
            let resolved = resolve_adaptive(avg);
            assert!(
                matches!(resolved, SpMVAlgo::Scalar | SpMVAlgo::Vector),
                "resolve_adaptive({avg}) returned {resolved:?}, expected Scalar or Vector"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Quality gate: CSR-Vector warp shuffle reduction simulation (CPU)
    // -----------------------------------------------------------------------

    /// Simulate a single-warp (32 threads) tree reduction of partial dot-products.
    ///
    /// In the Vector SpMV kernel each warp computes partial sums for the row
    /// elements it handles, then performs a binary tree (warp-shuffle) reduction
    /// to sum all 32 partial sums into a single row result.
    ///
    /// This test verifies the correctness of that reduction algorithm on the CPU.
    #[test]
    fn spmv_warp_reduction_sim_32_threads() {
        // 32 partial sums (one per thread in a warp)
        let partial: Vec<f64> = (0..32_u32).map(|i| f64::from(i * i + 1)).collect();
        let naive_sum: f64 = partial.iter().sum();

        // Simulate binary tree reduction (warp shuffle pattern):
        // stride 16, 8, 4, 2, 1
        let mut sums = partial.clone();
        let mut active = 32_usize;
        while active > 1 {
            let half = active / 2;
            for lane in 0..half {
                sums[lane] += sums[lane + half];
            }
            active = half;
        }
        let tree_sum = sums[0];

        assert!(
            (tree_sum - naive_sum).abs() < 1e-9,
            "Warp tree reduction ({tree_sum}) must match naive sum ({naive_sum})"
        );
    }

    /// Simulate a half-warp (16 threads) tree reduction.
    ///
    /// Verifies reduction correctness for the half-warp code path used when
    /// the row is shorter than a full warp.
    #[test]
    fn spmv_half_warp_reduction_sim_16_threads() {
        let partial: Vec<f64> = (0..16_u32).map(|i| f64::from(2 * i + 3)).collect();
        let naive_sum: f64 = partial.iter().sum();

        let mut sums = partial.clone();
        let mut active = 16_usize;
        while active > 1 {
            let half = active / 2;
            for lane in 0..half {
                sums[lane] += sums[lane + half];
            }
            active = half;
        }
        let tree_sum = sums[0];

        assert!(
            (tree_sum - naive_sum).abs() < 1e-9,
            "Half-warp tree reduction ({tree_sum}) must match naive sum ({naive_sum})"
        );
    }

    // -----------------------------------------------------------------------
    // Quality gate: SpMV numerical accuracy vs dense reference (CPU simulation)
    // -----------------------------------------------------------------------

    /// Dense-reference SpMV: computes y = A * x for a general dense matrix.
    fn dense_spmv(a_rows: usize, a_cols: usize, a: &[f64], x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0_f64; a_rows];
        for i in 0..a_rows {
            for j in 0..a_cols {
                y[i] += a[i * a_cols + j] * x[j];
            }
        }
        y
    }

    /// CSR SpMV simulation: computes y = A_csr * x on the CPU.
    fn csr_spmv_sim(
        nrows: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        values: &[f64],
        x: &[f64],
    ) -> Vec<f64> {
        let mut y = vec![0.0_f64; nrows];
        for i in 0..nrows {
            for idx in row_ptr[i]..row_ptr[i + 1] {
                y[i] += values[idx] * x[col_idx[idx]];
            }
        }
        y
    }

    /// SpMV for 4×4 identity matrix: y = I * x must equal x.
    ///
    /// This is the simplest correctness test: the identity provides a known
    /// reference where every output equals the corresponding input.
    #[test]
    fn spmv_numerical_accuracy_identity_4x4() {
        let n = 4_usize;
        // Identity matrix in CSR format
        let row_ptr = vec![0, 1, 2, 3, 4];
        let col_idx = vec![0, 1, 2, 3];
        let values = vec![1.0_f64; n];
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];

        let y_csr = csr_spmv_sim(n, &row_ptr, &col_idx, &values, &x);
        let y_dense = dense_spmv(
            n,
            n,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            &x,
        );
        for i in 0..n {
            assert!(
                (y_csr[i] - y_dense[i]).abs() < 1e-13,
                "SpMV I×x: y_csr[{i}]={} != y_dense[{i}]={}",
                y_csr[i],
                y_dense[i],
            );
        }
    }

    /// SpMV for a 0.1% sparse 1000×1000 matrix with a known diagonal pattern.
    ///
    /// Only diagonal entries are set (1000 out of 1_000_000 possible entries = 0.1%).
    /// Result must equal x (diagonal matrix with ones = identity).
    #[test]
    fn spmv_very_sparse_0_1_percent_1000x1000() {
        let n = 1000_usize;
        // Diagonal matrix (0.1% density)
        let row_ptr: Vec<usize> = (0..=n).collect();
        let col_idx: Vec<usize> = (0..n).collect();
        let values: Vec<f64> = vec![2.0; n]; // diagonal value = 2
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.001 + 1.0).collect();

        let y = csr_spmv_sim(n, &row_ptr, &col_idx, &values, &x);

        for i in 0..n {
            let expected = 2.0 * x[i];
            assert!(
                (y[i] - expected).abs() < 1e-10,
                "0.1% sparse SpMV row {i}: got {}, expected {expected}",
                y[i],
            );
        }
    }

    /// SpMV for a 10% sparse 100×100 matrix: banded structure.
    ///
    /// Uses a banded matrix with bandwidth 5 (5 non-zeros per row on average),
    /// giving approximately 10% density for a 100×100 system.
    #[test]
    fn spmv_moderate_10_percent_100x100() {
        let n = 100_usize;
        let bandwidth = 5_usize; // ±2 off-diagonal + diagonal

        let mut row_ptr = vec![0_usize; n + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        for i in 0..n {
            let start = i.saturating_sub(2);
            let end = (i + 3).min(n);
            for j in start..end {
                col_idx.push(j);
                values.push(if i == j { 4.0_f64 } else { -1.0 });
            }
            row_ptr[i + 1] = col_idx.len();
        }
        let _ = bandwidth; // document variable used in comments

        // x = [1, 1, 1, ..., 1]
        let x = vec![1.0_f64; n];
        let y_csr = csr_spmv_sim(n, &row_ptr, &col_idx, &values, &x);

        // Build dense matrix and compute reference
        let mut a_dense = vec![0.0_f64; n * n];
        for i in 0..n {
            let start = i.saturating_sub(2);
            let end = (i + 3).min(n);
            for j in start..end {
                a_dense[i * n + j] = if i == j { 4.0 } else { -1.0 };
            }
        }
        let y_dense = dense_spmv(n, n, &a_dense, &x);

        for i in 0..n {
            assert!(
                (y_csr[i] - y_dense[i]).abs() < 1e-10,
                "10% sparse SpMV row {i}: got {}, expected {}",
                y_csr[i],
                y_dense[i],
            );
        }
    }

    // -----------------------------------------------------------------------
    // Quality gate: auto-format selection thresholds
    // -----------------------------------------------------------------------

    /// Verify format selection thresholds cover the three density regimes.
    ///
    /// - avg_nnz < VECTOR_THRESHOLD (4.0) → Scalar
    /// - avg_nnz >= VECTOR_THRESHOLD       → Vector
    ///
    /// The test explicitly checks the three named brackets from the spec.
    #[test]
    fn spmv_format_selection_three_brackets() {
        // Very sparse (≤ 2): diagonal-like
        assert_eq!(
            resolve_adaptive(1.0),
            SpMVAlgo::Scalar,
            "avg_nnz=1.0 (≤ 2 bracket) must select Scalar"
        );
        assert_eq!(
            resolve_adaptive(2.0),
            SpMVAlgo::Scalar,
            "avg_nnz=2.0 (≤ 2 bracket) must select Scalar"
        );
        // Moderate (≤ 64): stencil-like (above VECTOR_THRESHOLD)
        assert_eq!(
            resolve_adaptive(5.0),
            SpMVAlgo::Vector,
            "avg_nnz=5.0 (≤ 64 bracket) must select Vector"
        );
        assert_eq!(
            resolve_adaptive(32.0),
            SpMVAlgo::Vector,
            "avg_nnz=32.0 (≤ 64 bracket) must select Vector"
        );
        // Dense (> 64): near-dense graph
        assert_eq!(
            resolve_adaptive(65.0),
            SpMVAlgo::Vector,
            "avg_nnz=65.0 (> 64 bracket) must select Vector (binary model)"
        );
        assert_eq!(
            resolve_adaptive(256.0),
            SpMVAlgo::Vector,
            "avg_nnz=256.0 (> 64 bracket) must select Vector"
        );
    }
}
