//! TRSV -- Triangular solve: `op(A) * x = b`.
//!
//! Solves a triangular system of equations where `A` is upper or lower
//! triangular, with unit or non-unit diagonal. The solution `x` overwrites
//! the input `b` in-place.
//!
//! # GPU Strategy
//!
//! Triangular solve is inherently sequential along the substitution chain.
//!
//! - **Small N (< 512)**: A single thread block performs forward/back
//!   substitution sequentially. Thread 0 computes each element in order,
//!   using shared memory for the intermediate results.
//!
//! - **Large N**: A block-based approach with level-set parallelism.
//!   The matrix is partitioned into blocks along the diagonal. Within
//!   each diagonal block, a single-block solve is performed. Off-diagonal
//!   blocks contribute GEMV-like updates that can be parallelized.
//!
//! For this implementation we use the single-block sequential approach
//! which is correct for all sizes. The block-parallel optimization is
//! planned for a future iteration.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{DiagType, FillMode, GpuFloat, MatrixDesc, Transpose};

/// Maximum N for the single-block sequential kernel.
/// For larger N, a blocked approach should be used (future work).
const TRSV_SINGLE_BLOCK_MAX: u32 = 4096;

/// Solves `op(A) * x = b`, overwriting `b` with the solution `x`.
///
/// The triangular matrix `A` is specified by `uplo` (upper/lower), `trans`
/// (transpose mode), and `diag` (unit/non-unit diagonal).
///
/// # Arguments
///
/// * `handle` -- BLAS handle providing stream and device context.
/// * `uplo` -- Whether `A` is upper or lower triangular.
/// * `trans` -- Whether to use `A`, `A^T`, or `A^H`.
/// * `diag` -- Whether the diagonal is unit or non-unit.
/// * `n` -- Order of the triangular matrix `A`.
/// * `a` -- Matrix descriptor for `A`.
/// * `x` -- Input vector `b`, overwritten with the solution `x`.
/// * `incx` -- Stride between consecutive elements of `x`. Must be positive.
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if `n` exceeds single-block limit.
/// Returns [`BlasError::InvalidArgument`] if `incx` is not positive.
/// Returns [`BlasError::BufferTooSmall`] if any buffer is undersized.
#[allow(clippy::too_many_arguments)]
pub fn trsv<T: GpuFloat>(
    handle: &BlasHandle,
    uplo: FillMode,
    trans: Transpose,
    diag: DiagType,
    n: u32,
    a: &MatrixDesc<T>,
    x: &mut DeviceBuffer<T>,
    incx: i32,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }

    validate_trsv_args(n, a, x, incx)?;

    let ptx = generate_trsv_ptx::<T>(handle.sm_version(), uplo, trans, diag, n)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "trsv")?;

    // Single block with enough threads to hold the vector in shared memory.
    // Thread 0 does the sequential solve; other threads are idle but present
    // for future warp-collaborative optimization.
    let block_size = n.min(256);
    let params = LaunchParams::new(1u32, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(a.ptr, x.as_device_ptr(), n, a.ld, incx as u32),
    )?;

    Ok(())
}

/// Validates arguments for TRSV.
fn validate_trsv_args<T: GpuFloat>(
    n: u32,
    a: &MatrixDesc<T>,
    x: &DeviceBuffer<T>,
    incx: i32,
) -> BlasResult<()> {
    if incx <= 0 {
        return Err(BlasError::InvalidArgument(
            "incx must be positive".to_string(),
        ));
    }
    if n > TRSV_SINGLE_BLOCK_MAX {
        return Err(BlasError::InvalidDimension(format!(
            "n ({n}) exceeds single-block TRSV limit ({TRSV_SINGLE_BLOCK_MAX}); \
             blocked TRSV not yet implemented"
        )));
    }
    if a.rows < n || a.cols < n {
        return Err(BlasError::InvalidDimension(format!(
            "A must be at least {n}x{n}, got {}x{}",
            a.rows, a.cols
        )));
    }
    let x_req = required_elements(n, incx);
    if x.len() < x_req {
        return Err(BlasError::BufferTooSmall {
            expected: x_req,
            actual: x.len(),
        });
    }
    Ok(())
}

/// Generates PTX for the TRSV kernel (sequential single-block approach).
///
/// Thread 0 performs forward substitution (lower triangle) or back
/// substitution (upper triangle). The algorithm processes elements in order,
/// using global memory reads for both A and x.
fn generate_trsv_ptx<T: GpuFloat>(
    sm: SmVersion,
    uplo: FillMode,
    trans: Transpose,
    diag: DiagType,
    _n: u32,
) -> BlasResult<String> {
    let is_f64 = T::SIZE == 8;
    let elem_bytes = T::size_u32();
    let ptx_ty = T::PTX_TYPE;
    let is_upper = matches!(uplo, FillMode::Upper);
    let use_trans = matches!(trans, Transpose::Trans | Transpose::ConjTrans);
    let is_unit = matches!(diag, DiagType::Unit);

    // Determine iteration direction:
    // Lower + NoTrans => forward (i = 0..n)
    // Upper + NoTrans => backward (i = n-1..0)
    // Lower + Trans   => backward
    // Upper + Trans   => forward
    let forward = is_upper == use_trans;

    KernelBuilder::new("trsv")
        .target(sm)
        .param("a_ptr", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("lda", PtxType::U32)
        .param("incx", PtxType::U32)
        .body(move |b| {
            // Only thread 0 executes the sequential solve
            let tid = b.thread_id_x();
            let one_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {one_reg}, 1;"));

            b.if_lt_u32(tid, one_reg, |b| {
                let a_ptr = b.load_param_u64("a_ptr");
                let x_ptr = b.load_param_u64("x_ptr");
                let n_reg = b.load_param_u32("n");
                let lda = b.load_param_u32("lda");
                let incx = b.load_param_u32("incx");

                // Outer loop: i iterates over each element to solve
                let outer_label = b.fresh_label("trsv_outer");
                let outer_done = b.fresh_label("trsv_outer_done");
                let i = b.alloc_reg(PtxType::U32);

                if forward {
                    b.raw_ptx(&format!("mov.u32 {i}, 0;"));
                } else {
                    // i = n - 1
                    b.raw_ptx(&format!("sub.u32 {i}, {n_reg}, 1;"));
                }

                b.label(&outer_label);

                // Check bounds
                let outer_pred = b.alloc_reg(PtxType::Pred);
                if forward {
                    b.raw_ptx(&format!("setp.lo.u32 {outer_pred}, {i}, {n_reg};"));
                } else {
                    // i >= 0 is always true for u32; check i < n (handles wrap-around)
                    b.raw_ptx(&format!("setp.lo.u32 {outer_pred}, {i}, {n_reg};"));
                }
                b.raw_ptx(&format!("@!{outer_pred} bra {outer_done};"));

                // Load x[i * incx] (the current right-hand side value)
                let xi_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {xi_idx}, {i}, {incx};"));
                let xi_addr = b.byte_offset_addr(x_ptr.clone(), xi_idx, elem_bytes);
                let xi_val = load_float(b, xi_addr.clone(), is_f64);

                // Subtract contributions from previously solved elements
                // For forward: j = 0..i, for backward: j = i+1..n
                let inner_label = b.fresh_label("trsv_inner");
                let inner_done = b.fresh_label("trsv_inner_done");
                let j = b.alloc_reg(PtxType::U32);
                let sum = b.alloc_reg(ptx_ty);
                emit_zero(b, sum.clone(), is_f64);

                if forward {
                    b.raw_ptx(&format!("mov.u32 {j}, 0;"));
                } else {
                    let i_plus1 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("add.u32 {i_plus1}, {i}, 1;"));
                    b.raw_ptx(&format!("mov.u32 {j}, {i_plus1};"));
                }

                b.label(&inner_label);
                let inner_pred = b.alloc_reg(PtxType::Pred);
                if forward {
                    b.raw_ptx(&format!("setp.lo.u32 {inner_pred}, {j}, {i};"));
                } else {
                    b.raw_ptx(&format!("setp.lo.u32 {inner_pred}, {j}, {n_reg};"));
                }
                b.raw_ptx(&format!("@!{inner_pred} bra {inner_done};"));

                // A[i][j] or A[j][i] depending on transpose
                let (row, col) = if !use_trans {
                    (i.clone(), j.clone())
                } else {
                    (j.clone(), i.clone())
                };
                let row_off = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {row_off}, {row}, {lda};"));
                let a_idx = b.add_u32(row_off, col);
                let a_addr = b.byte_offset_addr(a_ptr.clone(), a_idx, elem_bytes);
                let a_val = load_float(b, a_addr, is_f64);

                // Load x[j * incx]
                let xj_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {xj_idx}, {j}, {incx};"));
                let xj_addr = b.byte_offset_addr(x_ptr.clone(), xj_idx, elem_bytes);
                let xj_val = load_float(b, xj_addr, is_f64);

                // sum += A[..][..] * x[j]
                let new_sum = if is_f64 {
                    b.fma_f64(a_val, xj_val, sum.clone())
                } else {
                    b.fma_f32(a_val, xj_val, sum.clone())
                };
                emit_mov_float(b, sum.clone(), new_sum, is_f64);

                b.raw_ptx(&format!("add.u32 {j}, {j}, 1;"));
                b.branch(&inner_label);

                b.label(&inner_done);

                // x[i] = (x[i] - sum) / A[i][i]
                let diff = if is_f64 {
                    b.sub_f64(xi_val, sum)
                } else {
                    b.sub_f32(xi_val, sum)
                };

                let result = if is_unit {
                    // Unit diagonal: x[i] = x[i] - sum (no division)
                    diff
                } else {
                    // Non-unit: divide by diagonal element A[i][i]
                    let diag_off = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {diag_off}, {i}, {lda};"));
                    let diag_idx = b.add_u32(diag_off, i.clone());
                    let diag_addr = b.byte_offset_addr(a_ptr.clone(), diag_idx, elem_bytes);
                    let diag_val = load_float(b, diag_addr, is_f64);

                    let r = b.alloc_reg(ptx_ty);
                    if is_f64 {
                        b.raw_ptx(&format!("div.rn.f64 {r}, {diff}, {diag_val};"));
                    } else {
                        b.raw_ptx(&format!("div.rn.f32 {r}, {diff}, {diag_val};"));
                    }
                    r
                };

                // Store x[i * incx] = result
                store_float(b, xi_addr, result, is_f64);

                // Advance i
                if forward {
                    b.raw_ptx(&format!("add.u32 {i}, {i}, 1;"));
                } else {
                    // i-- (subtract 1; will wrap to large value if i was 0)
                    b.raw_ptx(&format!("sub.u32 {i}, {i}, 1;"));
                }
                b.branch(&outer_label);

                b.label(&outer_done);
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))
}

// -- Shared helpers --

fn emit_zero(b: &mut BodyBuilder<'_>, reg: Register, is_f64: bool) {
    if is_f64 {
        b.raw_ptx(&format!("mov.b64 {reg}, 0d0000000000000000;"));
    } else {
        b.raw_ptx(&format!("mov.b32 {reg}, 0f00000000;"));
    }
}

fn emit_mov_float(b: &mut BodyBuilder<'_>, dst: Register, src: Register, is_f64: bool) {
    let ty = if is_f64 { "f64" } else { "f32" };
    b.raw_ptx(&format!("mov.{ty} {dst}, {src};"));
}

fn load_float(b: &mut BodyBuilder<'_>, addr: Register, is_f64: bool) -> Register {
    if is_f64 {
        b.load_global_f64(addr)
    } else {
        b.load_global_f32(addr)
    }
}

fn store_float(b: &mut BodyBuilder<'_>, addr: Register, val: Register, is_f64: bool) {
    if is_f64 {
        b.store_global_f64(addr, val);
    } else {
        b.store_global_f32(addr, val);
    }
}

fn required_elements(n: u32, inc: i32) -> usize {
    if n == 0 {
        return 0;
    }
    1 + (n as usize - 1) * inc.unsigned_abs() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trsv_ptx_generation_lower_notrans_nonunit() {
        let ptx = generate_trsv_ptx::<f32>(
            SmVersion::Sm80,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
            64,
        );
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry trsv"));
    }

    #[test]
    fn trsv_ptx_generation_upper_trans_unit() {
        let ptx = generate_trsv_ptx::<f64>(
            SmVersion::Sm80,
            FillMode::Upper,
            Transpose::Trans,
            DiagType::Unit,
            128,
        );
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry trsv"));
    }

    #[test]
    fn trsv_ptx_generation_various_sizes() {
        // Verify PTX generation succeeds for different sizes.
        for &sz in &[1, 32, 256, 512] {
            let ptx = generate_trsv_ptx::<f32>(
                SmVersion::Sm80,
                FillMode::Lower,
                Transpose::NoTrans,
                DiagType::NonUnit,
                sz,
            );
            assert!(ptx.is_ok(), "failed for n={sz}");
        }
    }
}
