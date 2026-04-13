//! SYMV -- Symmetric matrix-vector multiplication.
//!
//! Computes `y = alpha * A * x + beta * y` where `A` is a symmetric matrix
//! stored in either the upper or lower triangle.
//!
//! # GPU Strategy
//!
//! Each thread computes one element of the output vector. For each output
//! element `y[i]`, the thread reads both the stored triangle element and
//! the corresponding mirror element from the other triangle. Only the
//! specified triangle (upper or lower) is actually stored in memory --
//! the mirror is obtained by swapping indices: `A[i][j] == A[j][i]`.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{FillMode, GpuFloat, MatrixDesc};

/// Default block size for SYMV kernels.
const SYMV_BLOCK_SIZE: u32 = 256;

/// Computes `y = alpha * A * x + beta * y` where A is symmetric.
///
/// Only the triangle specified by `uplo` is read from `A`. The other
/// triangle is inferred by symmetry.
///
/// # Arguments
///
/// * `handle` -- BLAS handle providing stream and device context.
/// * `uplo` -- Specifies whether the upper or lower triangle of `A` is stored.
/// * `n` -- Order of the symmetric matrix `A` (n x n).
/// * `alpha` -- Scalar multiplier for `A * x`.
/// * `a` -- Matrix descriptor for symmetric matrix `A`.
/// * `x` -- Input vector (device memory).
/// * `incx` -- Stride between consecutive elements of `x`. Must be positive.
/// * `beta` -- Scalar multiplier for the existing `y`.
/// * `y` -- Input/output vector (device memory, modified in-place).
/// * `incy` -- Stride between consecutive elements of `y`. Must be positive.
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if `n` is zero or `A` is undersized.
/// Returns [`BlasError::InvalidArgument`] if increments are not positive.
/// Returns [`BlasError::BufferTooSmall`] if any buffer is undersized.
#[allow(clippy::too_many_arguments)]
pub fn symv<T: GpuFloat>(
    handle: &BlasHandle,
    uplo: FillMode,
    n: u32,
    alpha: T,
    a: &MatrixDesc<T>,
    x: &DeviceBuffer<T>,
    incx: i32,
    beta: T,
    y: &mut DeviceBuffer<T>,
    incy: i32,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }

    validate_symv_args(n, a, x, incx, y, incy)?;

    let ptx = generate_symv_ptx::<T>(handle.sm_version(), uplo)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "symv")?;

    let block_size = SYMV_BLOCK_SIZE;
    let grid_size = grid_size_for(n, block_size);
    let params = LaunchParams::new(grid_size, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            a.ptr,
            x.as_device_ptr(),
            y.as_device_ptr(),
            alpha.to_bits_u64(),
            beta.to_bits_u64(),
            n,
            a.ld,
            incx as u32,
            incy as u32,
        ),
    )?;

    Ok(())
}

/// Validates arguments for SYMV.
fn validate_symv_args<T: GpuFloat>(
    n: u32,
    a: &MatrixDesc<T>,
    x: &DeviceBuffer<T>,
    incx: i32,
    y: &DeviceBuffer<T>,
    incy: i32,
) -> BlasResult<()> {
    if incx <= 0 {
        return Err(BlasError::InvalidArgument(
            "incx must be positive".to_string(),
        ));
    }
    if incy <= 0 {
        return Err(BlasError::InvalidArgument(
            "incy must be positive".to_string(),
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
    let y_req = required_elements(n, incy);
    if y.len() < y_req {
        return Err(BlasError::BufferTooSmall {
            expected: y_req,
            actual: y.len(),
        });
    }
    Ok(())
}

/// Generates PTX for the SYMV kernel.
///
/// Each thread `i` computes `y[i] = alpha * sum_j(A_sym[i][j] * x[j]) + beta * y[i]`
/// by iterating over `j = 0..n`. For each `(i,j)` pair, the element is loaded
/// from the stored triangle (reading `A[i][j]` if in the stored part, or
/// `A[j][i]` for the mirror).
fn generate_symv_ptx<T: GpuFloat>(sm: SmVersion, uplo: FillMode) -> BlasResult<String> {
    let is_f64 = T::size_u32() == 8;
    let elem_bytes = T::size_u32();
    let ptx_ty = T::PTX_TYPE;
    let is_upper = matches!(uplo, FillMode::Upper);

    KernelBuilder::new("symv")
        .target(sm)
        .param("a_ptr", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("beta_bits", PtxType::U64)
        .param("n", PtxType::U32)
        .param("lda", PtxType::U32)
        .param("incx", PtxType::U32)
        .param("incy", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg.clone(), |b| {
                let a_ptr = b.load_param_u64("a_ptr");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");
                let lda = b.load_param_u32("lda");
                let incx = b.load_param_u32("incx");
                let incy = b.load_param_u32("incy");
                let alpha_bits = b.load_param_u64("alpha_bits");
                let beta_bits = b.load_param_u64("beta_bits");

                let alpha = reinterpret_bits_to_float(b, alpha_bits, is_f64);
                let beta = reinterpret_bits_to_float(b, beta_bits, is_f64);

                // Initialize accumulator
                let acc = b.alloc_reg(ptx_ty);
                emit_zero(b, acc.clone(), is_f64);

                // Loop j = 0..n
                let loop_label = b.fresh_label("symv_loop");
                let done_label = b.fresh_label("symv_done");
                let j = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {j}, 0;"));

                b.label(&loop_label);
                let pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {pred}, {j}, {n_reg};"));
                b.raw_ptx(&format!("@!{pred} bra {done_label};"));

                // Determine row,col for memory access based on fill mode
                // For upper: if i <= j, read A[i][j]; else read A[j][i]
                // For lower: if i >= j, read A[i][j]; else read A[j][i]
                let row = b.alloc_reg(PtxType::U32);
                let col = b.alloc_reg(PtxType::U32);
                let swap_pred = b.alloc_reg(PtxType::Pred);

                if is_upper {
                    // Swap when i > j (read from A[j][i] instead)
                    b.raw_ptx(&format!("setp.hi.u32 {swap_pred}, {gid}, {j};"));
                } else {
                    // Swap when i < j (read from A[j][i] instead)
                    b.raw_ptx(&format!("setp.lo.u32 {swap_pred}, {gid}, {j};"));
                }

                // If swap: row=j, col=gid; else: row=gid, col=j
                b.raw_ptx(&format!("selp.u32 {row}, {j}, {gid}, {swap_pred};"));
                b.raw_ptx(&format!("selp.u32 {col}, {gid}, {j}, {swap_pred};"));

                // Address: a_ptr + (row * lda + col) * elem_bytes
                let row_off = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {row_off}, {row}, {lda};"));
                let idx = b.add_u32(row_off, col);
                let a_addr = b.byte_offset_addr(a_ptr.clone(), idx, elem_bytes);

                // Address: x_ptr + j * incx * elem_bytes
                let x_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {x_idx}, {j}, {incx};"));
                let x_addr = b.byte_offset_addr(x_ptr.clone(), x_idx, elem_bytes);

                let a_val = load_float(b, a_addr, is_f64);
                let x_val = load_float(b, x_addr, is_f64);

                // acc += a_val * x_val
                let new_acc = if is_f64 {
                    b.fma_f64(a_val, x_val, acc.clone())
                } else {
                    b.fma_f32(a_val, x_val, acc.clone())
                };
                emit_mov_float(b, acc.clone(), new_acc, is_f64);

                // j++
                b.raw_ptx(&format!("add.u32 {j}, {j}, 1;"));
                b.branch(&loop_label);

                b.label(&done_label);

                // y[gid*incy] = alpha * acc + beta * y[gid*incy]
                let y_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {y_idx}, {gid}, {incy};"));
                let y_addr = b.byte_offset_addr(y_ptr, y_idx, elem_bytes);
                let y_cur = load_float(b, y_addr.clone(), is_f64);

                let result = compute_alpha_acc_plus_beta_y(b, alpha, acc, beta, y_cur, is_f64);
                store_float(b, y_addr, result, is_f64);
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))
}

// ---------------------------------------------------------------------------
// Shared PTX helpers (used by multiple L2 kernels)
// ---------------------------------------------------------------------------

/// Reinterprets a u64 register as float (f32 or f64).
fn reinterpret_bits_to_float(b: &mut BodyBuilder<'_>, bits: Register, is_f64: bool) -> Register {
    if is_f64 {
        let r = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("mov.b64 {r}, {bits};"));
        r
    } else {
        let lo32 = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("cvt.u32.u64 {lo32}, {bits};"));
        let r = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("mov.b32 {r}, {lo32};"));
        r
    }
}

/// Emits a zero initialization for a float register.
fn emit_zero(b: &mut BodyBuilder<'_>, reg: Register, is_f64: bool) {
    if is_f64 {
        b.raw_ptx(&format!("mov.b64 {reg}, 0d0000000000000000;"));
    } else {
        b.raw_ptx(&format!("mov.b32 {reg}, 0f00000000;"));
    }
}

/// Emits a mov for float register.
fn emit_mov_float(b: &mut BodyBuilder<'_>, dst: Register, src: Register, is_f64: bool) {
    let ty = if is_f64 { "f64" } else { "f32" };
    b.raw_ptx(&format!("mov.{ty} {dst}, {src};"));
}

/// Loads a float from global memory.
fn load_float(b: &mut BodyBuilder<'_>, addr: Register, is_f64: bool) -> Register {
    if is_f64 {
        b.load_global_f64(addr)
    } else {
        b.load_global_f32(addr)
    }
}

/// Stores a float to global memory.
fn store_float(b: &mut BodyBuilder<'_>, addr: Register, val: Register, is_f64: bool) {
    if is_f64 {
        b.store_global_f64(addr, val);
    } else {
        b.store_global_f32(addr, val);
    }
}

/// Computes `alpha * acc + beta * y_cur`.
fn compute_alpha_acc_plus_beta_y(
    b: &mut BodyBuilder<'_>,
    alpha: Register,
    acc: Register,
    beta: Register,
    y_cur: Register,
    is_f64: bool,
) -> Register {
    let alpha_acc = if is_f64 {
        let r = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("mul.rn.f64 {r}, {alpha}, {acc};"));
        r
    } else {
        let r = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("mul.rn.f32 {r}, {alpha}, {acc};"));
        r
    };
    let beta_y = if is_f64 {
        let r = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("mul.rn.f64 {r}, {beta}, {y_cur};"));
        r
    } else {
        let r = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("mul.rn.f32 {r}, {beta}, {y_cur};"));
        r
    };
    if is_f64 {
        b.add_f64(alpha_acc, beta_y)
    } else {
        b.add_f32(alpha_acc, beta_y)
    }
}

/// Computes the minimum buffer elements required for a strided vector.
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
    fn symv_ptx_generation_upper_f32() {
        let ptx = generate_symv_ptx::<f32>(SmVersion::Sm80, FillMode::Upper);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry symv"));
    }

    #[test]
    fn symv_ptx_generation_lower_f64() {
        let ptx = generate_symv_ptx::<f64>(SmVersion::Sm80, FillMode::Lower);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry symv"));
    }
}
