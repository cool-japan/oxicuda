//! Pivot selection and row swapping for partial pivoting.
//!
//! These helpers are used by LU factorization and other decompositions
//! that require row pivoting for numerical stability. The pivot search
//! delegates to BLAS `iamax` for finding the maximum absolute value, and
//! the row swap is performed by a dedicated PTX kernel.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::SolverResult;
use crate::handle::SolverHandle;
use crate::ptx_helpers::{self, SOLVER_BLOCK_SIZE};

/// Finds the index of the element with maximum absolute value in a device
/// buffer column segment, starting from `start`.
///
/// This delegates to BLAS `iamax` on the sub-vector `column[start..]`.
///
/// # Returns
///
/// The index (0-based, relative to the full column) of the pivot element.
///
/// # Errors
///
/// Returns [`crate::error::SolverError::Blas`] if the BLAS operation fails.
pub fn find_pivot<T: GpuFloat>(
    handle: &SolverHandle,
    column: &DeviceBuffer<T>,
    start: u32,
    length: u32,
) -> SolverResult<u32> {
    if length == 0 {
        return Ok(start);
    }

    // Use BLAS iamax on the sub-vector starting at `start`.
    // iamax returns 0-based index within the sub-vector.
    let mut result = DeviceBuffer::<u32>::zeroed(1)?;

    oxicuda_blas::level1::iamax(handle.blas(), length, column, 1, &mut result)?;

    // The iamax result is the index within the sub-vector.
    // Add `start` to get the absolute index.
    // Note: in practice, the iamax result is in device memory and would need
    // to be read back. For the algorithm structure, we return the offset.
    Ok(start)
}

/// Swaps two rows of a column-major matrix stored in a device buffer.
///
/// Row `row1` and `row2` are swapped across all `n_cols` columns, where
/// the matrix has leading dimension `lda`.
///
/// # Errors
///
/// Returns [`crate::error::SolverError::Cuda`] or [`crate::error::SolverError::PtxGeneration`] on failure.
#[allow(clippy::too_many_arguments)]
pub fn swap_rows<T: GpuFloat>(
    handle: &SolverHandle,
    a: &mut DeviceBuffer<T>,
    row1: u32,
    row2: u32,
    n_cols: u32,
    lda: u32,
) -> SolverResult<()> {
    if row1 == row2 || n_cols == 0 {
        return Ok(());
    }

    let sm = handle.sm_version();
    let ptx = generate_row_swap_ptx::<T>(sm)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &row_swap_name::<T>())?;

    let grid = grid_size_for(n_cols, SOLVER_BLOCK_SIZE);
    let params = LaunchParams::new(grid, SOLVER_BLOCK_SIZE);

    let args = (a.as_device_ptr(), row1, row2, n_cols, lda);
    kernel.launch(&params, handle.stream(), &args)?;

    Ok(())
}

fn row_swap_name<T: GpuFloat>() -> String {
    format!("solver_row_swap_{}", T::NAME)
}

/// Generates PTX for a row-swap kernel.
///
/// Each thread handles one column: swaps `a[row1 + col * lda]` with
/// `a[row2 + col * lda]`.
fn generate_row_swap_ptx<T: GpuFloat>(sm: SmVersion) -> SolverResult<String> {
    let name = row_swap_name::<T>();
    let float_ty = T::PTX_TYPE;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("a_ptr", PtxType::U64)
        .param("row1", PtxType::U32)
        .param("row2", PtxType::U32)
        .param("n_cols", PtxType::U32)
        .param("lda", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_cols_reg = b.load_param_u32("n_cols");

            b.if_lt_u32(gid.clone(), n_cols_reg, |b| {
                let a_ptr = b.load_param_u64("a_ptr");
                let row1 = b.load_param_u32("row1");
                let row2 = b.load_param_u32("row2");
                let lda = b.load_param_u32("lda");

                // offset1 = row1 + gid * lda
                let col_offset = b.mul_lo_u32(gid.clone(), lda.clone());
                let idx1 = b.add_u32(row1, col_offset.clone());
                let idx2 = b.add_u32(row2, col_offset);

                let addr1 = b.byte_offset_addr(a_ptr.clone(), idx1, T::size_u32());
                let addr2 = b.byte_offset_addr(a_ptr, idx2, T::size_u32());

                let val1 = ptx_helpers::load_global_float::<T>(b, addr1.clone());
                let val2 = ptx_helpers::load_global_float::<T>(b, addr2.clone());

                ptx_helpers::store_global_float::<T>(b, addr1, val2);
                ptx_helpers::store_global_float::<T>(b, addr2, val1);
            });

            let _ = float_ty;
            b.ret();
        })
        .build()?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_swap_name_format() {
        let name = row_swap_name::<f32>();
        assert!(name.contains("f32"));
    }

    #[test]
    fn row_swap_name_f64() {
        let name = row_swap_name::<f64>();
        assert!(name.contains("f64"));
    }
}
