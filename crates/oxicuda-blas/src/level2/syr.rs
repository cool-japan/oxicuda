//! SYR -- Symmetric rank-1 update.
//!
//! Computes `A = alpha * x * x^T + A` where `A` is a symmetric matrix
//! and only the specified triangle (upper or lower) is updated.
//!
//! # GPU Strategy
//!
//! Uses a 2D grid/block configuration. Each thread computes one element
//! in the specified triangle: `A[i][j] += alpha * x[i] * x[j]` where
//! (i,j) is in the upper or lower triangle depending on `uplo`.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{FillMode, GpuFloat, MatrixDescMut};

/// Default block dimensions for SYR kernels (16x16 = 256 threads).
const SYR_BLOCK_X: u32 = 16;
const SYR_BLOCK_Y: u32 = 16;

/// Computes `A = alpha * x * x^T + A` (symmetric rank-1 update).
///
/// Only the triangle specified by `uplo` is updated. The matrix `A` is
/// assumed to be symmetric, but only the stored triangle is modified.
///
/// # Arguments
///
/// * `handle` -- BLAS handle providing stream and device context.
/// * `uplo` -- Specifies which triangle of `A` to update.
/// * `n` -- Order of the symmetric matrix `A` (n x n).
/// * `alpha` -- Scalar multiplier.
/// * `x` -- Input vector of `n` elements.
/// * `incx` -- Stride for `x`. Must be positive.
/// * `a` -- Mutable matrix descriptor for `A` (modified in-place).
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if `n` is zero or `A` is undersized.
/// Returns [`BlasError::InvalidArgument`] if `incx` is not positive.
/// Returns [`BlasError::BufferTooSmall`] if `x` buffer is undersized.
pub fn syr<T: GpuFloat>(
    handle: &BlasHandle,
    uplo: FillMode,
    n: u32,
    alpha: T,
    x: &DeviceBuffer<T>,
    incx: i32,
    a: &mut MatrixDescMut<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }

    validate_syr_args(n, x, incx, a)?;

    let ptx = generate_syr_ptx::<T>(handle.sm_version(), uplo)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "syr")?;

    let grid_x = n.div_ceil(SYR_BLOCK_X);
    let grid_y = n.div_ceil(SYR_BLOCK_Y);
    let grid = Dim3::from((grid_x, grid_y));
    let block = Dim3::from((SYR_BLOCK_X, SYR_BLOCK_Y));
    let params = LaunchParams::new(grid, block);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            a.ptr,
            x.as_device_ptr(),
            alpha.to_bits_u64(),
            n,
            a.ld,
            incx as u32,
        ),
    )?;

    Ok(())
}

/// Validates arguments for SYR.
fn validate_syr_args<T: GpuFloat>(
    n: u32,
    x: &DeviceBuffer<T>,
    incx: i32,
    a: &MatrixDescMut<T>,
) -> BlasResult<()> {
    if incx <= 0 {
        return Err(BlasError::InvalidArgument(
            "incx must be positive".to_string(),
        ));
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

/// Generates PTX for the SYR kernel.
///
/// Each thread in the 2D grid computes one element update:
/// `A[row][col] += alpha * x[row] * x[col]`
/// but only if (row, col) falls within the specified triangle.
fn generate_syr_ptx<T: GpuFloat>(sm: SmVersion, uplo: FillMode) -> BlasResult<String> {
    let is_f64 = T::SIZE == 8;
    let elem_bytes = T::size_u32();
    let is_upper = matches!(uplo, FillMode::Upper);

    KernelBuilder::new("syr")
        .target(sm)
        .param("a_ptr", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("n", PtxType::U32)
        .param("lda", PtxType::U32)
        .param("incx", PtxType::U32)
        .body(move |b| {
            let (row, col) = b.global_thread_id_2d();
            let n_reg = b.load_param_u32("n");

            // Bounds check: row < n && col < n
            b.if_lt_u32(row.clone(), n_reg.clone(), |b| {
                b.if_lt_u32(col.clone(), n_reg, |b| {
                    // Triangle check: only update elements in the specified triangle
                    let in_triangle = b.alloc_reg(PtxType::Pred);
                    if is_upper {
                        // Upper: col >= row
                        b.raw_ptx(&format!("setp.hs.u32 {in_triangle}, {col}, {row};"));
                    } else {
                        // Lower: row >= col
                        b.raw_ptx(&format!("setp.hs.u32 {in_triangle}, {row}, {col};"));
                    }
                    let skip_label = b.fresh_label("syr_skip");
                    b.raw_ptx(&format!("@!{in_triangle} bra {skip_label};"));

                    let a_ptr = b.load_param_u64("a_ptr");
                    let x_ptr = b.load_param_u64("x_ptr");
                    let lda = b.load_param_u32("lda");
                    let incx = b.load_param_u32("incx");
                    let alpha_bits = b.load_param_u64("alpha_bits");

                    let alpha = reinterpret_bits_to_float(b, alpha_bits, is_f64);

                    // Load x[row * incx]
                    let xi_idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {xi_idx}, {row}, {incx};"));
                    let xi_addr = b.byte_offset_addr(x_ptr.clone(), xi_idx, elem_bytes);
                    let xi_val = load_float(b, xi_addr, is_f64);

                    // Load x[col * incx]
                    let xj_idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {xj_idx}, {col}, {incx};"));
                    let xj_addr = b.byte_offset_addr(x_ptr, xj_idx, elem_bytes);
                    let xj_val = load_float(b, xj_addr, is_f64);

                    // Compute alpha * x[i] * x[j]
                    let xixj = if is_f64 {
                        let r = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {r}, {xi_val}, {xj_val};"));
                        r
                    } else {
                        let r = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {r}, {xi_val}, {xj_val};"));
                        r
                    };
                    let alpha_xixj = if is_f64 {
                        let r = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {r}, {alpha}, {xixj};"));
                        r
                    } else {
                        let r = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {r}, {alpha}, {xixj};"));
                        r
                    };

                    // Load current A[row][col]
                    let a_row_off = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {a_row_off}, {row}, {lda};"));
                    let a_idx = b.add_u32(a_row_off, col.clone());
                    let a_addr = b.byte_offset_addr(a_ptr, a_idx, elem_bytes);
                    let a_cur = load_float(b, a_addr.clone(), is_f64);

                    // A[row][col] += alpha * x[i] * x[j]
                    let result = if is_f64 {
                        b.add_f64(a_cur, alpha_xixj)
                    } else {
                        b.add_f32(a_cur, alpha_xixj)
                    };

                    store_float(b, a_addr, result, is_f64);

                    b.label(&skip_label);
                });
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))
}

// -- Shared helpers --

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
    fn syr_ptx_generation_upper_f32() {
        let ptx = generate_syr_ptx::<f32>(SmVersion::Sm80, FillMode::Upper);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry syr"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn syr_ptx_generation_lower_f64() {
        let ptx = generate_syr_ptx::<f64>(SmVersion::Sm80, FillMode::Lower);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry syr"));
    }
}
