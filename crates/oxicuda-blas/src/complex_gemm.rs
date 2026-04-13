//! Complex-number GEMM and GEMV (CGEMM / ZGEMM / CGEMV / ZGEMV).
//!
//! Supports `Complex<f32>` and `Complex<f64>` matrix multiplication using
//! interleaved (re, im) storage. Each complex element occupies 2 consecutive
//! real elements in the [`DeviceBuffer`].
//!
//! # Complex arithmetic
//!
//! The complex multiply `(a + bi)(c + di) = (ac - bd) + (ad + bc)i` is
//! implemented using 4 real multiplies and 2 additions.
//!
//! # PTX kernel
//!
//! The generated SIMT kernel assigns one output element of C per thread.
//! Each thread accumulates real and imaginary parts separately across the
//! K dimension, then applies the complex alpha/beta epilogue.

use std::fmt::Write as FmtWrite;
use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams, grid_size_for};

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{GpuFloat, Transpose};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;

/// 2D block size for the complex GEMM kernel (16×16 = 256 threads).
const CGEMM_BLOCK_X: u32 = 16;
/// 2D block size for the complex GEMM kernel (16×16 = 256 threads).
const CGEMM_BLOCK_Y: u32 = 16;
/// 1D block size for the complex GEMV kernel.
const CGEMV_BLOCK: u32 = 256;

// ---------------------------------------------------------------------------
// Public API — Complex GEMM
// ---------------------------------------------------------------------------

/// Performs complex general matrix multiplication.
///
/// Computes `C = alpha * op(A) * op(B) + beta * C`, where A, B, C contain
/// interleaved complex elements (re, im pairs). Each matrix buffer therefore
/// has `2 * rows * cols` real elements.
///
/// # Type parameters
///
/// * `T` — the underlying real type (`f32` for CGEMM, `f64` for ZGEMM).
///
/// # Arguments
///
/// * `handle` — BLAS handle.
/// * `transa` — transpose mode for A.
/// * `transb` — transpose mode for B.
/// * `m`, `n`, `k` — logical matrix dimensions of the complex matrices.
/// * `alpha_re`, `alpha_im` — real and imaginary parts of scalar alpha.
/// * `a` — device buffer holding interleaved complex A (2 * lda * cols_a elems).
/// * `lda` — leading dimension of A (in complex elements, i.e. row count).
/// * `b` — device buffer holding interleaved complex B.
/// * `ldb` — leading dimension of B.
/// * `beta_re`, `beta_im` — real and imaginary parts of scalar beta.
/// * `c` — device buffer holding interleaved complex C (output, in-place).
/// * `ldc` — leading dimension of C.
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if any dimension is zero.
/// Returns [`BlasError::BufferTooSmall`] if buffers are undersized.
/// Returns [`BlasError::PtxGeneration`] if kernel generation fails.
#[allow(clippy::too_many_arguments)]
pub fn complex_gemm<T: GpuFloat>(
    handle: &BlasHandle,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha_re: T,
    alpha_im: T,
    a: &DeviceBuffer<T>,
    lda: usize,
    b: &DeviceBuffer<T>,
    ldb: usize,
    beta_re: T,
    beta_im: T,
    c: &mut DeviceBuffer<T>,
    ldc: usize,
) -> BlasResult<()> {
    // Validate dimensions
    if m == 0 || n == 0 || k == 0 {
        return Err(BlasError::InvalidDimension(
            "complex GEMM: all dimensions must be non-zero".into(),
        ));
    }

    // Validate buffer sizes (each complex element is 2 reals)
    let a_required = match transa {
        Transpose::NoTrans => 2 * lda * k,
        Transpose::Trans | Transpose::ConjTrans => 2 * lda * m,
    };
    if a.len() < a_required {
        return Err(BlasError::BufferTooSmall {
            expected: a_required,
            actual: a.len(),
        });
    }

    let b_required = match transb {
        Transpose::NoTrans => 2 * ldb * n,
        Transpose::Trans | Transpose::ConjTrans => 2 * ldb * k,
    };
    if b.len() < b_required {
        return Err(BlasError::BufferTooSmall {
            expected: b_required,
            actual: b.len(),
        });
    }

    let c_required = 2 * ldc * n;
    if c.len() < c_required {
        return Err(BlasError::BufferTooSmall {
            expected: c_required,
            actual: c.len(),
        });
    }

    // Validate leading dimensions
    validate_complex_ld(transa, m, k, lda, "A")?;
    validate_complex_ld(transb, k, n, ldb, "B")?;
    if ldc < m {
        return Err(BlasError::InvalidDimension(format!(
            "complex GEMM: ldc ({ldc}) < m ({m})"
        )));
    }

    // Generate PTX and build the kernel module.
    let ptx = generate_complex_gemm_ptx::<T>(handle.sm_version(), transa, transb)?;
    let module = Arc::new(Module::from_ptx(&ptx).map_err(BlasError::Cuda)?);
    let kernel_name = complex_gemm_kernel_name::<T>(transa, transb);
    let kernel = Kernel::from_module(module, &kernel_name).map_err(BlasError::Cuda)?;

    // Configure 2D launch: one thread per output complex element.
    let grid_x = grid_size_for(n as u32, CGEMM_BLOCK_X);
    let grid_y = grid_size_for(m as u32, CGEMM_BLOCK_Y);
    let grid = Dim3::xy(grid_x, grid_y);
    let block = Dim3::xy(CGEMM_BLOCK_X, CGEMM_BLOCK_Y);
    let params = LaunchParams::new(grid, block);

    // Kernel signature: (a_ptr, b_ptr, c_ptr, m, n, k, lda, ldb, ldc,
    //                    alpha_re, alpha_im, beta_re, beta_im)
    let args = (
        a.as_device_ptr(),
        b.as_device_ptr(),
        c.as_device_ptr(),
        m as u32,
        n as u32,
        k as u32,
        lda as u32,
        ldb as u32,
        ldc as u32,
        alpha_re,
        alpha_im,
        beta_re,
        beta_im,
    );
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(BlasError::Cuda)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Public API — Complex GEMV
// ---------------------------------------------------------------------------

/// Performs complex matrix-vector multiplication.
///
/// Computes `y = alpha * op(A) * x + beta * y`, where A is an m x n complex
/// matrix and x, y are complex vectors stored as interleaved (re, im) pairs.
///
/// # Arguments
///
/// * `handle` — BLAS handle.
/// * `trans` — transpose mode for A.
/// * `m`, `n` — dimensions of A (in complex elements).
/// * `alpha_re`, `alpha_im` — complex scalar alpha.
/// * `a` — device buffer for matrix A (interleaved complex, 2 * lda * n elems).
/// * `lda` — leading dimension of A.
/// * `x` — input complex vector.
/// * `incx` — stride between consecutive complex elements of x (in complex units).
/// * `beta_re`, `beta_im` — complex scalar beta.
/// * `y` — input/output complex vector (modified in-place).
/// * `incy` — stride between consecutive complex elements of y (in complex units).
///
/// # Errors
///
/// Returns [`BlasError`] on invalid dimensions, buffer size, or PTX errors.
#[allow(clippy::too_many_arguments)]
pub fn complex_gemv<T: GpuFloat>(
    handle: &BlasHandle,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha_re: T,
    alpha_im: T,
    a: &DeviceBuffer<T>,
    lda: usize,
    x: &DeviceBuffer<T>,
    incx: usize,
    beta_re: T,
    beta_im: T,
    y: &mut DeviceBuffer<T>,
    incy: usize,
) -> BlasResult<()> {
    if m == 0 || n == 0 {
        return Ok(());
    }

    if incx == 0 {
        return Err(BlasError::InvalidArgument(
            "complex GEMV: incx must be positive".into(),
        ));
    }
    if incy == 0 {
        return Err(BlasError::InvalidArgument(
            "complex GEMV: incy must be positive".into(),
        ));
    }

    // Validate A buffer: 2 * lda * n reals
    let a_required = 2 * lda * n;
    if a.len() < a_required {
        return Err(BlasError::BufferTooSmall {
            expected: a_required,
            actual: a.len(),
        });
    }

    // x length depends on transpose: NoTrans => n elements, Trans => m elements
    let (x_len, y_len) = match trans {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };

    let x_required = 2 * (1 + (x_len.saturating_sub(1)) * incx);
    if x.len() < x_required {
        return Err(BlasError::BufferTooSmall {
            expected: x_required,
            actual: x.len(),
        });
    }

    let y_required = 2 * (1 + (y_len.saturating_sub(1)) * incy);
    if y.len() < y_required {
        return Err(BlasError::BufferTooSmall {
            expected: y_required,
            actual: y.len(),
        });
    }

    if lda < m {
        return Err(BlasError::InvalidDimension(format!(
            "complex GEMV: lda ({lda}) < m ({m})"
        )));
    }

    // Determine output and inner loop lengths.
    let (output_len, inner_len) = match trans {
        Transpose::NoTrans => (m, n),
        Transpose::Trans | Transpose::ConjTrans => (n, m),
    };

    // Generate PTX and build the kernel module.
    let ptx = generate_complex_gemv_ptx::<T>(handle.sm_version(), trans)?;
    let ta = trans_label(trans);
    let gemv_kernel_name = format!("complex_gemv_{}_{ta}", T::NAME);
    let module = Arc::new(Module::from_ptx(&ptx).map_err(BlasError::Cuda)?);
    let kernel = Kernel::from_module(module, &gemv_kernel_name).map_err(BlasError::Cuda)?;

    // 1D launch: one thread per output complex element.
    let grid = grid_size_for(output_len as u32, CGEMV_BLOCK);
    let params = LaunchParams::new(grid, CGEMV_BLOCK);

    // Kernel signature: (a_ptr, x_ptr, y_ptr, m, n, lda, incx, incy,
    //                    alpha_re, alpha_im, beta_re, beta_im,
    //                    output_len, inner_len)
    let args = (
        a.as_device_ptr(),
        x.as_device_ptr(),
        y.as_device_ptr(),
        m as u32,
        n as u32,
        lda as u32,
        incx as u32,
        incy as u32,
        alpha_re,
        alpha_im,
        beta_re,
        beta_im,
        output_len as u32,
        inner_len as u32,
    );
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(BlasError::Cuda)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// PTX generation — Complex GEMM kernel
// ---------------------------------------------------------------------------

/// Generates a SIMT PTX kernel for complex GEMM.
///
/// Each thread computes one complex output element of C by iterating over
/// the K dimension, accumulating real and imaginary parts separately using
/// the `(ac - bd)` and `(ad + bc)` formulas with 4 FMAs per K-iteration.
fn generate_complex_gemm_ptx<T: GpuFloat>(
    sm: SmVersion,
    transa: Transpose,
    transb: Transpose,
) -> BlasResult<String> {
    let byte_size = T::PTX_TYPE.size_bytes();
    let kernel_name = complex_gemm_kernel_name::<T>(transa, transb);
    let is_f64 = byte_size == 8;
    let (fr, ld_ty) = if is_f64 { ("fd", "f64") } else { ("f", "f32") };
    let zero_lit = if is_f64 {
        "0d0000000000000000"
    } else {
        "0f00000000"
    };

    let mut p = String::with_capacity(8192);

    // Header
    wl(&mut p, &format!(".version {}", sm.ptx_version()))?;
    wl(&mut p, &format!(".target {}", sm.as_ptx_str()))?;
    wl(&mut p, ".address_size 64")?;
    wl(&mut p, "")?;

    // Entry
    wl(&mut p, &format!(".visible .entry {kernel_name}("))?;
    wl(&mut p, "    .param .u64 %param_a,")?;
    wl(&mut p, "    .param .u64 %param_b,")?;
    wl(&mut p, "    .param .u64 %param_c,")?;
    wl(&mut p, "    .param .u32 %param_m,")?;
    wl(&mut p, "    .param .u32 %param_n,")?;
    wl(&mut p, "    .param .u32 %param_k,")?;
    wl(&mut p, "    .param .u32 %param_lda,")?;
    wl(&mut p, "    .param .u32 %param_ldb,")?;
    wl(&mut p, "    .param .u32 %param_ldc,")?;
    wl(&mut p, &format!("    .param .{ld_ty} %param_alpha_re,"))?;
    wl(&mut p, &format!("    .param .{ld_ty} %param_alpha_im,"))?;
    wl(&mut p, &format!("    .param .{ld_ty} %param_beta_re,"))?;
    wl(&mut p, &format!("    .param .{ld_ty} %param_beta_im"))?;
    wl(&mut p, ")")?;
    wl(&mut p, "{")?;

    // Register declarations
    wl(&mut p, "    .reg .b32 %r<48>;")?;
    wl(&mut p, "    .reg .b64 %rd<24>;")?;
    if is_f64 {
        wl(&mut p, "    .reg .f64 %fd<32>;")?;
    } else {
        wl(&mut p, "    .reg .f32 %f<32>;")?;
    }
    wl(&mut p, "    .reg .pred %p<8>;")?;
    wl(&mut p, "")?;

    // Thread indices
    wl(&mut p, "    mov.u32 %r0, %tid.x;")?;
    wl(&mut p, "    mov.u32 %r1, %tid.y;")?;
    wl(&mut p, "    mov.u32 %r2, %ctaid.x;")?;
    wl(&mut p, "    mov.u32 %r3, %ctaid.y;")?;
    wl(&mut p, "    mov.u32 %r4, %ntid.x;")?;
    wl(&mut p, "    mov.u32 %r5, %ntid.y;")?;
    wl(&mut p, "    mad.lo.u32 %r6, %r2, %r4, %r0;  // col")?;
    wl(&mut p, "    mad.lo.u32 %r7, %r3, %r5, %r1;  // row")?;
    wl(&mut p, "")?;

    // Load params
    wl(&mut p, "    ld.param.u64 %rd0, [%param_a];")?;
    wl(&mut p, "    ld.param.u64 %rd1, [%param_b];")?;
    wl(&mut p, "    ld.param.u64 %rd2, [%param_c];")?;
    wl(&mut p, "    ld.param.u32 %r8, [%param_m];")?;
    wl(&mut p, "    ld.param.u32 %r9, [%param_n];")?;
    wl(&mut p, "    ld.param.u32 %r10, [%param_k];")?;
    wl(&mut p, "    ld.param.u32 %r30, [%param_lda];")?;
    wl(&mut p, "    ld.param.u32 %r31, [%param_ldb];")?;
    wl(&mut p, "    ld.param.u32 %r32, [%param_ldc];")?;
    wl(
        &mut p,
        &format!("    ld.param.{ld_ty} %{fr}20, [%param_alpha_re];"),
    )?;
    wl(
        &mut p,
        &format!("    ld.param.{ld_ty} %{fr}21, [%param_alpha_im];"),
    )?;
    wl(
        &mut p,
        &format!("    ld.param.{ld_ty} %{fr}22, [%param_beta_re];"),
    )?;
    wl(
        &mut p,
        &format!("    ld.param.{ld_ty} %{fr}23, [%param_beta_im];"),
    )?;
    wl(&mut p, "")?;

    // Bounds check
    wl(&mut p, "    setp.ge.u32 %p0, %r7, %r8;")?;
    wl(&mut p, "    setp.ge.u32 %p1, %r6, %r9;")?;
    wl(&mut p, "    @%p0 bra $CGEMM_DONE;")?;
    wl(&mut p, "    @%p1 bra $CGEMM_DONE;")?;
    wl(&mut p, "")?;

    // Init accumulators
    wl(
        &mut p,
        &format!("    mov.{ld_ty} %{fr}0, {zero_lit};  // acc_re"),
    )?;
    wl(
        &mut p,
        &format!("    mov.{ld_ty} %{fr}1, {zero_lit};  // acc_im"),
    )?;
    wl(&mut p, "    mov.u32 %r11, 0;")?;
    wl(&mut p, "")?;

    // K-loop
    wl(&mut p, "$CGEMM_K_LOOP:")?;
    wl(&mut p, "    setp.ge.u32 %p2, %r11, %r10;")?;
    wl(&mut p, "    @%p2 bra $CGEMM_K_DONE;")?;

    // Load A[row,k] or A[k,row] (complex interleaved)
    let (a_maj, a_min) = match transa {
        Transpose::NoTrans => ("%r7", "%r11"),
        Transpose::Trans | Transpose::ConjTrans => ("%r11", "%r7"),
    };
    wl(
        &mut p,
        &format!("    mad.lo.u32 %r12, {a_maj}, %r30, {a_min};"),
    )?;
    wl(&mut p, "    shl.b32 %r12, %r12, 1;  // *2 for complex")?;
    wl(&mut p, "    cvt.u64.u32 %rd3, %r12;")?;
    wl(&mut p, &format!("    mul.lo.u64 %rd3, %rd3, {byte_size};"))?;
    wl(&mut p, "    add.u64 %rd4, %rd0, %rd3;")?;
    wl(&mut p, &format!("    ld.global.{ld_ty} %{fr}2, [%rd4];"))?;
    wl(
        &mut p,
        &format!("    ld.global.{ld_ty} %{fr}3, [%rd4+{byte_size}];"),
    )?;

    // Conjugate A if ConjTrans
    if transa == Transpose::ConjTrans {
        wl(&mut p, &format!("    neg.{ld_ty} %{fr}3, %{fr}3;"))?;
    }

    // Load B[k,col] or B[col,k] (complex interleaved)
    let (b_maj, b_min) = match transb {
        Transpose::NoTrans => ("%r11", "%r6"),
        Transpose::Trans | Transpose::ConjTrans => ("%r6", "%r11"),
    };
    wl(
        &mut p,
        &format!("    mad.lo.u32 %r13, {b_maj}, %r31, {b_min};"),
    )?;
    wl(&mut p, "    shl.b32 %r13, %r13, 1;")?;
    wl(&mut p, "    cvt.u64.u32 %rd5, %r13;")?;
    wl(&mut p, &format!("    mul.lo.u64 %rd5, %rd5, {byte_size};"))?;
    wl(&mut p, "    add.u64 %rd6, %rd1, %rd5;")?;
    wl(&mut p, &format!("    ld.global.{ld_ty} %{fr}4, [%rd6];"))?;
    wl(
        &mut p,
        &format!("    ld.global.{ld_ty} %{fr}5, [%rd6+{byte_size}];"),
    )?;

    // Conjugate B if ConjTrans
    if transb == Transpose::ConjTrans {
        wl(&mut p, &format!("    neg.{ld_ty} %{fr}5, %{fr}5;"))?;
    }

    // Complex multiply-accumulate using 4 FMAs:
    // acc_re = fma(a_re, b_re, acc_re)  ... acc_re += a_re * b_re
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}0, %{fr}2, %{fr}4, %{fr}0;  // acc_re += a_re*b_re"),
    )?;
    // acc_re -= a_im * b_im  =>  neg a_im, then fma
    wl(&mut p, &format!("    neg.{ld_ty} %{fr}6, %{fr}3;"))?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}0, %{fr}6, %{fr}5, %{fr}0;  // acc_re -= a_im*b_im"),
    )?;
    // acc_im = fma(a_re, b_im, acc_im)
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}1, %{fr}2, %{fr}5, %{fr}1;  // acc_im += a_re*b_im"),
    )?;
    // acc_im = fma(a_im, b_re, acc_im)
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}1, %{fr}3, %{fr}4, %{fr}1;  // acc_im += a_im*b_re"),
    )?;

    wl(&mut p, "    add.u32 %r11, %r11, 1;")?;
    wl(&mut p, "    bra $CGEMM_K_LOOP;")?;
    wl(&mut p, "$CGEMM_K_DONE:")?;
    wl(&mut p, "")?;

    // Epilogue: C[row,col] = alpha * acc + beta * C_old (complex)
    // C address
    wl(&mut p, "    mad.lo.u32 %r14, %r7, %r32, %r6;")?;
    wl(&mut p, "    shl.b32 %r14, %r14, 1;  // *2 for complex")?;
    wl(&mut p, "    cvt.u64.u32 %rd7, %r14;")?;
    wl(&mut p, &format!("    mul.lo.u64 %rd7, %rd7, {byte_size};"))?;
    wl(&mut p, "    add.u64 %rd8, %rd2, %rd7;")?;
    // Load C_old
    wl(
        &mut p,
        &format!("    ld.global.{ld_ty} %{fr}10, [%rd8];  // c_re"),
    )?;
    wl(
        &mut p,
        &format!("    ld.global.{ld_ty} %{fr}11, [%rd8+{byte_size}];  // c_im"),
    )?;

    // result_re = alpha_re * acc_re - alpha_im * acc_im + beta_re * c_re - beta_im * c_im
    // result_im = alpha_re * acc_im + alpha_im * acc_re + beta_re * c_im + beta_im * c_re
    wl(
        &mut p,
        &format!("    mul.rn.{ld_ty} %{fr}12, %{fr}20, %{fr}0;  // alpha_re * acc_re"),
    )?;
    wl(&mut p, &format!("    neg.{ld_ty} %{fr}15, %{fr}21;"))?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}12, %{fr}15, %{fr}1, %{fr}12;  // - alpha_im * acc_im"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}12, %{fr}22, %{fr}10, %{fr}12;  // + beta_re * c_re"),
    )?;
    wl(&mut p, &format!("    neg.{ld_ty} %{fr}16, %{fr}23;"))?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}12, %{fr}16, %{fr}11, %{fr}12;  // - beta_im * c_im"),
    )?;

    wl(
        &mut p,
        &format!("    mul.rn.{ld_ty} %{fr}13, %{fr}20, %{fr}1;  // alpha_re * acc_im"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}13, %{fr}21, %{fr}0, %{fr}13;  // + alpha_im * acc_re"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}13, %{fr}22, %{fr}11, %{fr}13;  // + beta_re * c_im"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}13, %{fr}23, %{fr}10, %{fr}13;  // + beta_im * c_re"),
    )?;

    // Store result
    wl(&mut p, &format!("    st.global.{ld_ty} [%rd8], %{fr}12;"))?;
    wl(
        &mut p,
        &format!("    st.global.{ld_ty} [%rd8+{byte_size}], %{fr}13;"),
    )?;
    wl(&mut p, "")?;

    wl(&mut p, "$CGEMM_DONE:")?;
    wl(&mut p, "    ret;")?;
    wl(&mut p, "}")?;

    Ok(p)
}

// ---------------------------------------------------------------------------
// PTX generation — Complex GEMV kernel
// ---------------------------------------------------------------------------

/// Generates a SIMT PTX kernel for complex GEMV.
///
/// Each thread computes one element of the output vector y by performing
/// a complex dot product of the corresponding row of op(A) with x.
fn generate_complex_gemv_ptx<T: GpuFloat>(sm: SmVersion, trans: Transpose) -> BlasResult<String> {
    let byte_size = T::PTX_TYPE.size_bytes();
    let is_f64 = byte_size == 8;
    let (fr, ld_ty) = if is_f64 { ("fd", "f64") } else { ("f", "f32") };
    let zero_lit = if is_f64 {
        "0d0000000000000000"
    } else {
        "0f00000000"
    };
    let ta = trans_label(trans);
    let kernel_name = format!("complex_gemv_{}_{ta}", T::NAME);

    let mut p = String::with_capacity(4096);

    wl(&mut p, &format!(".version {}", sm.ptx_version()))?;
    wl(&mut p, &format!(".target {}", sm.as_ptx_str()))?;
    wl(&mut p, ".address_size 64")?;
    wl(&mut p, "")?;

    wl(&mut p, &format!(".visible .entry {kernel_name}("))?;
    wl(&mut p, "    .param .u64 %param_a,")?;
    wl(&mut p, "    .param .u64 %param_x,")?;
    wl(&mut p, "    .param .u64 %param_y,")?;
    wl(&mut p, "    .param .u32 %param_m,")?;
    wl(&mut p, "    .param .u32 %param_n,")?;
    wl(&mut p, "    .param .u32 %param_lda,")?;
    wl(&mut p, "    .param .u32 %param_incx,")?;
    wl(&mut p, "    .param .u32 %param_incy,")?;
    wl(&mut p, &format!("    .param .{ld_ty} %param_alpha_re,"))?;
    wl(&mut p, &format!("    .param .{ld_ty} %param_alpha_im,"))?;
    wl(&mut p, &format!("    .param .{ld_ty} %param_beta_re,"))?;
    wl(&mut p, &format!("    .param .{ld_ty} %param_beta_im,"))?;
    wl(&mut p, "    .param .u32 %param_output_len,")?;
    wl(&mut p, "    .param .u32 %param_inner_len")?;
    wl(&mut p, ")")?;
    wl(&mut p, "{")?;

    wl(&mut p, "    .reg .b32 %r<32>;")?;
    wl(&mut p, "    .reg .b64 %rd<16>;")?;
    if is_f64 {
        wl(&mut p, "    .reg .f64 %fd<24>;")?;
    } else {
        wl(&mut p, "    .reg .f32 %f<24>;")?;
    }
    wl(&mut p, "    .reg .pred %p<4>;")?;
    wl(&mut p, "")?;

    // Global thread ID
    wl(&mut p, "    mov.u32 %r0, %tid.x;")?;
    wl(&mut p, "    mov.u32 %r1, %ctaid.x;")?;
    wl(&mut p, "    mov.u32 %r2, %ntid.x;")?;
    wl(&mut p, "    mad.lo.u32 %r3, %r1, %r2, %r0;  // gid")?;

    // Load output_len, bounds check
    wl(&mut p, "    ld.param.u32 %r4, [%param_output_len];")?;
    wl(&mut p, "    setp.ge.u32 %p0, %r3, %r4;")?;
    wl(&mut p, "    @%p0 bra $CGEMV_DONE;")?;
    wl(&mut p, "")?;

    // Load params
    wl(&mut p, "    ld.param.u64 %rd0, [%param_a];")?;
    wl(&mut p, "    ld.param.u64 %rd1, [%param_x];")?;
    wl(&mut p, "    ld.param.u64 %rd2, [%param_y];")?;
    wl(&mut p, "    ld.param.u32 %r5, [%param_lda];")?;
    wl(&mut p, "    ld.param.u32 %r6, [%param_incx];")?;
    wl(&mut p, "    ld.param.u32 %r7, [%param_incy];")?;
    wl(&mut p, "    ld.param.u32 %r8, [%param_inner_len];")?;
    wl(
        &mut p,
        &format!("    ld.param.{ld_ty} %{fr}16, [%param_alpha_re];"),
    )?;
    wl(
        &mut p,
        &format!("    ld.param.{ld_ty} %{fr}17, [%param_alpha_im];"),
    )?;
    wl(
        &mut p,
        &format!("    ld.param.{ld_ty} %{fr}18, [%param_beta_re];"),
    )?;
    wl(
        &mut p,
        &format!("    ld.param.{ld_ty} %{fr}19, [%param_beta_im];"),
    )?;
    wl(&mut p, "")?;

    // Init acc
    wl(
        &mut p,
        &format!("    mov.{ld_ty} %{fr}0, {zero_lit};  // acc_re"),
    )?;
    wl(
        &mut p,
        &format!("    mov.{ld_ty} %{fr}1, {zero_lit};  // acc_im"),
    )?;
    wl(&mut p, "    mov.u32 %r9, 0;  // k")?;
    wl(&mut p, "")?;

    // Inner loop
    wl(&mut p, "$CGEMV_LOOP:")?;
    wl(&mut p, "    setp.ge.u32 %p1, %r9, %r8;")?;
    wl(&mut p, "    @%p1 bra $CGEMV_LOOP_DONE;")?;

    // A element address depends on trans
    let use_trans = matches!(trans, Transpose::Trans | Transpose::ConjTrans);
    if !use_trans {
        // A[gid, k]: (gid * lda + k) * 2 * byte_size
        wl(&mut p, "    mad.lo.u32 %r10, %r3, %r5, %r9;")?;
    } else {
        // A[k, gid]: (k * lda + gid) * 2 * byte_size
        wl(&mut p, "    mad.lo.u32 %r10, %r9, %r5, %r3;")?;
    }
    wl(&mut p, "    shl.b32 %r10, %r10, 1;")?;
    wl(&mut p, "    cvt.u64.u32 %rd3, %r10;")?;
    wl(&mut p, &format!("    mul.lo.u64 %rd3, %rd3, {byte_size};"))?;
    wl(&mut p, "    add.u64 %rd4, %rd0, %rd3;")?;
    wl(&mut p, &format!("    ld.global.{ld_ty} %{fr}2, [%rd4];"))?;
    wl(
        &mut p,
        &format!("    ld.global.{ld_ty} %{fr}3, [%rd4+{byte_size}];"),
    )?;

    if trans == Transpose::ConjTrans {
        wl(&mut p, &format!("    neg.{ld_ty} %{fr}3, %{fr}3;"))?;
    }

    // x[k * incx]: (k * incx) * 2 * byte_size
    wl(&mut p, "    mul.lo.u32 %r11, %r9, %r6;")?;
    wl(&mut p, "    shl.b32 %r11, %r11, 1;")?;
    wl(&mut p, "    cvt.u64.u32 %rd5, %r11;")?;
    wl(&mut p, &format!("    mul.lo.u64 %rd5, %rd5, {byte_size};"))?;
    wl(&mut p, "    add.u64 %rd6, %rd1, %rd5;")?;
    wl(&mut p, &format!("    ld.global.{ld_ty} %{fr}4, [%rd6];"))?;
    wl(
        &mut p,
        &format!("    ld.global.{ld_ty} %{fr}5, [%rd6+{byte_size}];"),
    )?;

    // Complex FMA
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}0, %{fr}2, %{fr}4, %{fr}0;"),
    )?;
    wl(&mut p, &format!("    neg.{ld_ty} %{fr}6, %{fr}3;"))?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}0, %{fr}6, %{fr}5, %{fr}0;"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}1, %{fr}2, %{fr}5, %{fr}1;"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}1, %{fr}3, %{fr}4, %{fr}1;"),
    )?;

    wl(&mut p, "    add.u32 %r9, %r9, 1;")?;
    wl(&mut p, "    bra $CGEMV_LOOP;")?;
    wl(&mut p, "$CGEMV_LOOP_DONE:")?;
    wl(&mut p, "")?;

    // Load y[gid * incy]
    wl(&mut p, "    mul.lo.u32 %r12, %r3, %r7;")?;
    wl(&mut p, "    shl.b32 %r12, %r12, 1;")?;
    wl(&mut p, "    cvt.u64.u32 %rd7, %r12;")?;
    wl(&mut p, &format!("    mul.lo.u64 %rd7, %rd7, {byte_size};"))?;
    wl(&mut p, "    add.u64 %rd8, %rd2, %rd7;")?;
    wl(&mut p, &format!("    ld.global.{ld_ty} %{fr}10, [%rd8];"))?;
    wl(
        &mut p,
        &format!("    ld.global.{ld_ty} %{fr}11, [%rd8+{byte_size}];"),
    )?;

    // Complex epilogue: result = alpha * acc + beta * y_old
    wl(
        &mut p,
        &format!("    mul.rn.{ld_ty} %{fr}12, %{fr}16, %{fr}0;"),
    )?;
    wl(&mut p, &format!("    neg.{ld_ty} %{fr}14, %{fr}17;"))?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}12, %{fr}14, %{fr}1, %{fr}12;"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}12, %{fr}18, %{fr}10, %{fr}12;"),
    )?;
    wl(&mut p, &format!("    neg.{ld_ty} %{fr}15, %{fr}19;"))?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}12, %{fr}15, %{fr}11, %{fr}12;"),
    )?;

    wl(
        &mut p,
        &format!("    mul.rn.{ld_ty} %{fr}13, %{fr}16, %{fr}1;"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}13, %{fr}17, %{fr}0, %{fr}13;"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}13, %{fr}18, %{fr}11, %{fr}13;"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}13, %{fr}19, %{fr}10, %{fr}13;"),
    )?;

    // Store
    wl(&mut p, &format!("    st.global.{ld_ty} [%rd8], %{fr}12;"))?;
    wl(
        &mut p,
        &format!("    st.global.{ld_ty} [%rd8+{byte_size}], %{fr}13;"),
    )?;
    wl(&mut p, "")?;

    wl(&mut p, "$CGEMV_DONE:")?;
    wl(&mut p, "    ret;")?;
    wl(&mut p, "}")?;

    Ok(p)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validates leading dimension for a complex matrix operand.
fn validate_complex_ld(
    trans: Transpose,
    rows: usize,
    cols: usize,
    ld: usize,
    name: &str,
) -> BlasResult<()> {
    let min_ld = match trans {
        Transpose::NoTrans => rows,
        Transpose::Trans | Transpose::ConjTrans => cols,
    };
    if ld < min_ld {
        return Err(BlasError::InvalidDimension(format!(
            "complex GEMM: ld{name} ({ld}) < required ({min_ld})"
        )));
    }
    Ok(())
}

/// Constructs the kernel function name for a complex GEMM variant.
fn complex_gemm_kernel_name<T: GpuFloat>(transa: Transpose, transb: Transpose) -> String {
    let ta = trans_label(transa);
    let tb = trans_label(transb);
    format!("complex_gemm_{}_{ta}_{tb}", T::NAME)
}

/// Short label for a transpose mode.
fn trans_label(t: Transpose) -> &'static str {
    match t {
        Transpose::NoTrans => "n",
        Transpose::Trans => "t",
        Transpose::ConjTrans => "c",
    }
}

/// Writes a line to the PTX string, mapping fmt errors to `BlasError`.
fn wl(ptx: &mut String, line: &str) -> BlasResult<()> {
    writeln!(ptx, "{line}").map_err(|e| BlasError::PtxGeneration(format!("fmt error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_gemm_kernel_name_f32_nn() {
        let name = complex_gemm_kernel_name::<f32>(Transpose::NoTrans, Transpose::NoTrans);
        assert_eq!(name, "complex_gemm_f32_n_n");
    }

    #[test]
    fn complex_gemm_kernel_name_f64_tc() {
        let name = complex_gemm_kernel_name::<f64>(Transpose::Trans, Transpose::ConjTrans);
        assert_eq!(name, "complex_gemm_f64_t_c");
    }

    #[test]
    fn generate_complex_gemm_ptx_f32_nn() {
        let ptx = generate_complex_gemm_ptx::<f32>(
            SmVersion::Sm80,
            Transpose::NoTrans,
            Transpose::NoTrans,
        );
        let ptx = ptx.expect("PTX generation should succeed");
        assert!(ptx.contains(".entry complex_gemm_f32_n_n"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("$CGEMM_K_LOOP"));
        assert!(ptx.contains("$CGEMM_K_DONE"));
        assert!(ptx.contains("neg.f32"));
    }

    #[test]
    fn generate_complex_gemm_ptx_f64_tt() {
        let ptx =
            generate_complex_gemm_ptx::<f64>(SmVersion::Sm80, Transpose::Trans, Transpose::Trans);
        let ptx = ptx.expect("PTX generation should succeed");
        assert!(ptx.contains(".entry complex_gemm_f64_t_t"));
        assert!(ptx.contains("fma.rn.f64"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn generate_complex_gemm_ptx_conj_trans() {
        let ptx = generate_complex_gemm_ptx::<f32>(
            SmVersion::Sm75,
            Transpose::ConjTrans,
            Transpose::NoTrans,
        );
        let ptx = ptx.expect("PTX generation should succeed");
        assert!(ptx.contains("complex_gemm_f32_c_n"));
        // Should contain neg for conjugation
        assert!(ptx.contains("neg.f32"));
    }

    #[test]
    fn generate_complex_gemv_ptx_f32() {
        let ptx = generate_complex_gemv_ptx::<f32>(SmVersion::Sm80, Transpose::NoTrans);
        let ptx = ptx.expect("PTX generation should succeed");
        assert!(ptx.contains(".entry complex_gemv_f32_n"));
        assert!(ptx.contains("$CGEMV_LOOP"));
    }

    #[test]
    fn generate_complex_gemv_ptx_f64_trans() {
        let ptx = generate_complex_gemv_ptx::<f64>(SmVersion::Sm80, Transpose::Trans);
        let ptx = ptx.expect("PTX generation should succeed");
        assert!(ptx.contains(".entry complex_gemv_f64_t"));
        assert!(ptx.contains("fma.rn.f64"));
    }

    #[test]
    fn validate_complex_ld_ok() {
        assert!(validate_complex_ld(Transpose::NoTrans, 64, 32, 64, "A").is_ok());
        assert!(validate_complex_ld(Transpose::Trans, 64, 32, 32, "A").is_ok());
    }

    #[test]
    fn validate_complex_ld_error() {
        let err = validate_complex_ld(Transpose::NoTrans, 64, 32, 32, "A");
        assert!(err.is_err());
    }

    #[test]
    fn complex_gemm_zero_dim_error() {
        // We can't create a real BlasHandle without GPU, but we can test the
        // validation path by checking buffer sizes.
        let err =
            BlasError::InvalidDimension("complex GEMM: all dimensions must be non-zero".into());
        assert!(err.to_string().contains("non-zero"));
    }

    #[test]
    fn trans_label_values() {
        assert_eq!(trans_label(Transpose::NoTrans), "n");
        assert_eq!(trans_label(Transpose::Trans), "t");
        assert_eq!(trans_label(Transpose::ConjTrans), "c");
    }
}
