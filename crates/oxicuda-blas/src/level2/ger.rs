//! GER -- Rank-1 update.
//!
//! Computes `A = alpha * x * y^T + A` where `x` is an m-element vector,
//! `y` is an n-element vector, and `A` is an m-by-n matrix.
//!
//! # GPU Strategy
//!
//! This is a 2D elementwise operation: each thread computes one element
//! `A[i][j] += alpha * x[i] * y[j]`. A 2D grid/block configuration is
//! used to cover the full matrix.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{GpuFloat, MatrixDescMut};

/// Default block dimensions for GER kernels (16x16 = 256 threads).
const GER_BLOCK_X: u32 = 16;
const GER_BLOCK_Y: u32 = 16;

/// Computes `A = alpha * x * y^T + A` (rank-1 update).
///
/// # Arguments
///
/// * `handle` -- BLAS handle providing stream and device context.
/// * `m` -- Number of rows of matrix `A` and elements in `x`.
/// * `n` -- Number of columns of matrix `A` and elements in `y`.
/// * `alpha` -- Scalar multiplier.
/// * `x` -- Column vector of `m` elements.
/// * `incx` -- Stride for `x`. Must be positive.
/// * `y` -- Row vector of `n` elements.
/// * `incy` -- Stride for `y`. Must be positive.
/// * `a` -- Mutable matrix descriptor for `A` (modified in-place).
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if `m` or `n` is zero.
/// Returns [`BlasError::InvalidArgument`] if increments are not positive.
/// Returns [`BlasError::BufferTooSmall`] if any buffer is undersized.
#[allow(clippy::too_many_arguments)]
pub fn ger<T: GpuFloat>(
    handle: &BlasHandle,
    m: u32,
    n: u32,
    alpha: T,
    x: &DeviceBuffer<T>,
    incx: i32,
    y: &DeviceBuffer<T>,
    incy: i32,
    a: &mut MatrixDescMut<T>,
) -> BlasResult<()> {
    if m == 0 || n == 0 {
        return Ok(());
    }

    validate_ger_args(m, n, x, incx, y, incy, a)?;

    let ptx = generate_ger_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "ger")?;

    let grid_x = n.div_ceil(GER_BLOCK_X);
    let grid_y = m.div_ceil(GER_BLOCK_Y);
    let grid = Dim3::from((grid_x, grid_y));
    let block = Dim3::from((GER_BLOCK_X, GER_BLOCK_Y));
    let params = LaunchParams::new(grid, block);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            a.ptr,
            x.as_device_ptr(),
            y.as_device_ptr(),
            alpha.to_bits_u64(),
            m,
            n,
            a.ld,
            incx as u32,
            incy as u32,
        ),
    )?;

    Ok(())
}

/// Validates arguments for GER.
fn validate_ger_args<T: GpuFloat>(
    m: u32,
    n: u32,
    x: &DeviceBuffer<T>,
    incx: i32,
    y: &DeviceBuffer<T>,
    incy: i32,
    a: &MatrixDescMut<T>,
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
    if a.rows < m || a.cols < n {
        return Err(BlasError::InvalidDimension(format!(
            "A must be at least {}x{}, got {}x{}",
            m, n, a.rows, a.cols
        )));
    }
    let x_req = required_elements(m, incx);
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

/// Generates PTX for the GER kernel.
///
/// Each thread computes `A[row][col] += alpha * x[row] * y[col]` using
/// a 2D grid where `col = blockIdx.x * blockDim.x + threadIdx.x` and
/// `row = blockIdx.y * blockDim.y + threadIdx.y`.
fn generate_ger_ptx<T: GpuFloat>(sm: SmVersion) -> BlasResult<String> {
    let is_f64 = T::SIZE == 8;
    let elem_bytes = T::size_u32();
    let _ptx_ty = T::PTX_TYPE;

    KernelBuilder::new("ger")
        .target(sm)
        .param("a_ptr", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("m", PtxType::U32)
        .param("n", PtxType::U32)
        .param("lda", PtxType::U32)
        .param("incx", PtxType::U32)
        .param("incy", PtxType::U32)
        .body(move |b| {
            let (row, col) = b.global_thread_id_2d();
            let m_reg = b.load_param_u32("m");
            let n_reg = b.load_param_u32("n");

            // Bounds check: row < m && col < n
            b.if_lt_u32(row.clone(), m_reg, |b| {
                b.if_lt_u32(col.clone(), n_reg, |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let x_ptr = b.load_param_u64("x_ptr");
                    let y_ptr = b.load_param_u64("y_ptr");
                    let lda = b.load_param_u32("lda");
                    let incx = b.load_param_u32("incx");
                    let incy = b.load_param_u32("incy");
                    let alpha_bits = b.load_param_u64("alpha_bits");

                    let alpha = reinterpret_bits_to_float(b, alpha_bits, is_f64);

                    // Load x[row * incx]
                    let x_idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {x_idx}, {row}, {incx};"));
                    let x_addr = b.byte_offset_addr(x_ptr, x_idx, elem_bytes);
                    let x_val = load_float(b, x_addr, is_f64);

                    // Load y[col * incy]
                    let y_idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {y_idx}, {col}, {incy};"));
                    let y_addr = b.byte_offset_addr(y_ptr, y_idx, elem_bytes);
                    let y_val = load_float(b, y_addr, is_f64);

                    // Compute alpha * x[i] * y[j]
                    let xy = if is_f64 {
                        let r = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {r}, {x_val}, {y_val};"));
                        r
                    } else {
                        let r = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {r}, {x_val}, {y_val};"));
                        r
                    };
                    let alpha_xy = if is_f64 {
                        let r = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {r}, {alpha}, {xy};"));
                        r
                    } else {
                        let r = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {r}, {alpha}, {xy};"));
                        r
                    };

                    // Load current A[row][col]
                    let a_row_off = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {a_row_off}, {row}, {lda};"));
                    let a_idx = b.add_u32(a_row_off, col.clone());
                    let a_addr = b.byte_offset_addr(a_ptr, a_idx, elem_bytes);
                    let a_cur = load_float(b, a_addr.clone(), is_f64);

                    // A[row][col] += alpha * x[row] * y[col]
                    let result = if is_f64 {
                        b.add_f64(a_cur, alpha_xy)
                    } else {
                        b.add_f32(a_cur, alpha_xy)
                    };

                    store_float(b, a_addr, result, is_f64);
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
    fn ger_ptx_generation_f32() {
        let ptx = generate_ger_ptx::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry ger"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn ger_ptx_generation_f64() {
        let ptx = generate_ger_ptx::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry ger"));
    }
}
