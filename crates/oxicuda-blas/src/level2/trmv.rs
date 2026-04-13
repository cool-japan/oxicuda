//! TRMV -- Triangular matrix-vector multiplication.
//!
//! Computes `x = op(A) * x` where `A` is a triangular matrix (upper or lower)
//! with either unit or non-unit diagonal.
//!
//! # GPU Strategy
//!
//! Each thread computes one element of the result. The thread iterates over
//! the relevant portion of the row (for upper triangular, columns `j >= i`;
//! for lower, columns `j <= i`), accumulating the dot product with the
//! corresponding elements of `x`. The result is written back to `x` in-place.
//!
//! Because the operation is in-place, we copy the input vector to a temporary
//! buffer before launching the kernel to avoid read-after-write hazards.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{DiagType, FillMode, GpuFloat, MatrixDesc, Transpose};

/// Default block size for TRMV kernels.
const TRMV_BLOCK_SIZE: u32 = 256;

/// Computes `x = op(A) * x` where `A` is triangular.
///
/// # Arguments
///
/// * `handle` -- BLAS handle providing stream and device context.
/// * `uplo` -- Whether `A` is upper or lower triangular.
/// * `trans` -- Whether to use `A`, `A^T`, or `A^H`.
/// * `diag` -- Whether the diagonal is unit or non-unit.
/// * `n` -- Order of the triangular matrix `A`.
/// * `a` -- Matrix descriptor for `A`.
/// * `x` -- Input/output vector (modified in-place).
/// * `incx` -- Stride between consecutive elements of `x`. Must be positive.
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if `n` is zero or `A` is undersized.
/// Returns [`BlasError::InvalidArgument`] if `incx` is not positive.
/// Returns [`BlasError::BufferTooSmall`] if any buffer is undersized.
#[allow(clippy::too_many_arguments)]
pub fn trmv<T: GpuFloat>(
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

    validate_trmv_args(n, a, x, incx)?;

    // The operation is in-place (x = op(A)*x), so we need a copy of x to
    // read from while writing to x. Allocate a temporary device buffer.
    let x_copy = copy_device_buffer(x)?;

    let ptx = generate_trmv_ptx::<T>(handle.sm_version(), uplo, trans, diag)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "trmv")?;

    let block_size = TRMV_BLOCK_SIZE;
    let grid_size = grid_size_for(n, block_size);
    let params = LaunchParams::new(grid_size, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            a.ptr,
            x_copy.as_device_ptr(),
            x.as_device_ptr(),
            n,
            a.ld,
            incx as u32,
        ),
    )?;

    Ok(())
}

/// Validates arguments for TRMV.
fn validate_trmv_args<T: GpuFloat>(
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
    if a.rows < n || a.cols < n {
        return Err(BlasError::InvalidDimension(format!(
            "A must be at least {}x{}, got {}x{}",
            n, n, a.rows, a.cols
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

/// Creates a device-to-device copy of a buffer.
fn copy_device_buffer<T: GpuFloat>(src: &DeviceBuffer<T>) -> BlasResult<DeviceBuffer<T>> {
    let mut dst = DeviceBuffer::<T>::alloc(src.len())?;
    oxicuda_memory::copy::copy_dtod(&mut dst, src)?;
    Ok(dst)
}

/// Generates PTX for the TRMV kernel.
///
/// The kernel reads from `x_src` (the copy) and writes to `x_dst` (the original).
/// Each thread computes one element of the output.
fn generate_trmv_ptx<T: GpuFloat>(
    sm: SmVersion,
    uplo: FillMode,
    trans: Transpose,
    diag: DiagType,
) -> BlasResult<String> {
    let is_f64 = T::SIZE == 8;
    let elem_bytes = T::size_u32();
    let ptx_ty = T::PTX_TYPE;
    let is_upper = matches!(uplo, FillMode::Upper);
    let use_trans = matches!(trans, Transpose::Trans | Transpose::ConjTrans);
    let is_unit = matches!(diag, DiagType::Unit);

    // Effective triangle after transpose: upper+trans = lower iteration pattern
    let iter_upper = is_upper != use_trans;

    KernelBuilder::new("trmv")
        .target(sm)
        .param("a_ptr", PtxType::U64)
        .param("x_src", PtxType::U64)
        .param("x_dst", PtxType::U64)
        .param("n", PtxType::U32)
        .param("lda", PtxType::U32)
        .param("incx", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg.clone(), |b| {
                let a_ptr = b.load_param_u64("a_ptr");
                let x_src = b.load_param_u64("x_src");
                let x_dst = b.load_param_u64("x_dst");
                let lda = b.load_param_u32("lda");
                let incx = b.load_param_u32("incx");

                let acc = b.alloc_reg(ptx_ty);
                emit_zero(b, acc.clone(), is_f64);

                // Loop over j in the triangle
                let loop_label = b.fresh_label("trmv_loop");
                let done_label = b.fresh_label("trmv_done");
                let skip_diag_label = b.fresh_label("trmv_skipdiag");
                let j = b.alloc_reg(PtxType::U32);

                // Start index depends on triangle
                if iter_upper {
                    // Upper: j goes from gid to n-1
                    b.raw_ptx(&format!("mov.u32 {j}, {gid};"));
                } else {
                    // Lower: j goes from 0 to gid
                    b.raw_ptx(&format!("mov.u32 {j}, 0;"));
                }

                b.label(&loop_label);
                let pred = b.alloc_reg(PtxType::Pred);
                if iter_upper {
                    b.raw_ptx(&format!("setp.lo.u32 {pred}, {j}, {n_reg};"));
                } else {
                    // j <= gid  =>  j < gid + 1
                    let gid_plus1 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("add.u32 {gid_plus1}, {gid}, 1;"));
                    b.raw_ptx(&format!("setp.lo.u32 {pred}, {j}, {gid_plus1};"));
                }
                b.raw_ptx(&format!("@!{pred} bra {done_label};"));

                // For unit diagonal, skip the diagonal element (add x[i] later)
                if is_unit {
                    let diag_pred = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.eq.u32 {diag_pred}, {j}, {gid};"));
                    b.raw_ptx(&format!("@{diag_pred} bra {skip_diag_label};"));
                }

                // Compute A address:
                // No trans: A[gid][j] => a_ptr + (gid * lda + j) * elem_bytes
                // Trans:    A[j][gid] => a_ptr + (j * lda + gid) * elem_bytes
                let (row, col) = if !use_trans {
                    (gid.clone(), j.clone())
                } else {
                    (j.clone(), gid.clone())
                };
                let row_off = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {row_off}, {row}, {lda};"));
                let idx = b.add_u32(row_off, col);
                let a_addr = b.byte_offset_addr(a_ptr.clone(), idx, elem_bytes);

                // Load A element and x[j * incx]
                let a_val = load_float(b, a_addr, is_f64);
                let x_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {x_idx}, {j}, {incx};"));
                let x_addr = b.byte_offset_addr(x_src.clone(), x_idx, elem_bytes);
                let x_val = load_float(b, x_addr, is_f64);

                // acc += a_val * x_val
                let new_acc = if is_f64 {
                    b.fma_f64(a_val, x_val, acc.clone())
                } else {
                    b.fma_f32(a_val, x_val, acc.clone())
                };
                emit_mov_float(b, acc.clone(), new_acc, is_f64);

                if is_unit {
                    b.label(&skip_diag_label);
                }

                // j++
                b.raw_ptx(&format!("add.u32 {j}, {j}, 1;"));
                b.branch(&loop_label);

                b.label(&done_label);

                // For unit diagonal, add x[gid * incx] to acc (the implicit 1 on diagonal)
                if is_unit {
                    let xi_idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {xi_idx}, {gid}, {incx};"));
                    let xi_addr = b.byte_offset_addr(x_src.clone(), xi_idx, elem_bytes);
                    let xi_val = load_float(b, xi_addr, is_f64);
                    let new_acc = if is_f64 {
                        b.add_f64(acc.clone(), xi_val)
                    } else {
                        b.add_f32(acc.clone(), xi_val)
                    };
                    emit_mov_float(b, acc.clone(), new_acc, is_f64);
                }

                // Store result to x_dst[gid * incx]
                let out_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {out_idx}, {gid}, {incx};"));
                let out_addr = b.byte_offset_addr(x_dst, out_idx, elem_bytes);
                store_float(b, out_addr, acc, is_f64);
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))
}

// -- Shared helpers (same as in symv.rs) --

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
    fn trmv_ptx_generation_upper_notrans_nonunit() {
        let ptx = generate_trmv_ptx::<f32>(
            SmVersion::Sm80,
            FillMode::Upper,
            Transpose::NoTrans,
            DiagType::NonUnit,
        );
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry trmv"));
    }

    #[test]
    fn trmv_ptx_generation_lower_trans_unit() {
        let ptx = generate_trmv_ptx::<f64>(
            SmVersion::Sm80,
            FillMode::Lower,
            Transpose::Trans,
            DiagType::Unit,
        );
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry trmv"));
    }
}
