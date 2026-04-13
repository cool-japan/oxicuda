//! AXPY — `y = alpha * x + y`
//!
//! The classic BLAS Level 1 operation that scales vector `x` by scalar `alpha`
//! and adds the result to vector `y` in-place.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

use super::{L1_BLOCK_SIZE, required_elements};

/// Computes `y = alpha * x + y` (AXPY) on the GPU.
///
/// This is the fundamental scale-and-accumulate operation. If `alpha` is zero,
/// the function returns immediately without launching a kernel.
///
/// # Arguments
///
/// * `handle` — BLAS handle providing the CUDA stream and device info.
/// * `n` — number of elements to process.
/// * `alpha` — scalar multiplier for `x`.
/// * `x` — source vector (device buffer).
/// * `incx` — stride between consecutive elements of `x`. Must be positive.
/// * `y` — destination vector (device buffer), modified in-place.
/// * `incy` — stride between consecutive elements of `y`. Must be positive.
///
/// # Errors
///
/// Returns [`BlasError::BufferTooSmall`] if either buffer is too small for the
/// given `n` and stride. Returns [`BlasError::InvalidArgument`] if an increment
/// is non-positive.
pub fn axpy<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    alpha: T,
    x: &DeviceBuffer<T>,
    incx: i32,
    y: &mut DeviceBuffer<T>,
    incy: i32,
) -> BlasResult<()> {
    // Fast exit: nothing to do.
    if n == 0 {
        return Ok(());
    }
    // Skip when alpha is zero (no-op).
    if alpha == T::gpu_zero() {
        return Ok(());
    }

    validate_inc(incx, "incx")?;
    validate_inc(incy, "incy")?;

    let x_required = required_elements(n, incx);
    let y_required = required_elements(n, incy);
    if x.len() < x_required {
        return Err(BlasError::BufferTooSmall {
            expected: x_required,
            actual: x.len(),
        });
    }
    if y.len() < y_required {
        return Err(BlasError::BufferTooSmall {
            expected: y_required,
            actual: y.len(),
        });
    }

    // Generate PTX kernel.
    let ptx = generate_axpy_ptx::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &axpy_kernel_name::<T>())?;

    let grid = grid_size_for(n, L1_BLOCK_SIZE);
    let params = LaunchParams::new(grid, L1_BLOCK_SIZE);

    let args = (
        x.as_device_ptr(),
        y.as_device_ptr(),
        alpha.to_bits_u64(),
        n,
        incx as u32,
        incy as u32,
    );

    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

/// Returns the kernel function name for AXPY with the given precision.
fn axpy_kernel_name<T: GpuFloat>() -> String {
    format!("blas_axpy_{}", T::NAME)
}

/// Validates that an increment is positive.
fn validate_inc(inc: i32, name: &str) -> BlasResult<()> {
    if inc <= 0 {
        return Err(BlasError::InvalidArgument(format!(
            "{name} must be positive, got {inc}"
        )));
    }
    Ok(())
}

/// Generates PTX for the AXPY kernel: `y[i*incy] = alpha * x[i*incx] + y[i*incy]`.
///
/// The kernel takes parameters:
///   (x_ptr: u64, y_ptr: u64, alpha_bits: u64, n: u32, incx: u32, incy: u32)
fn generate_axpy_ptx<T: GpuFloat>(sm: SmVersion) -> BlasResult<String> {
    let _float_ty = T::PTX_TYPE;
    let name = axpy_kernel_name::<T>();

    // For f32, alpha_bits is the lower 32 bits of a u64.
    // We pass alpha as u64 uniformly, then reinterpret.
    let alpha_param_ty = PtxType::U64;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(L1_BLOCK_SIZE)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", alpha_param_ty)
        .param("n", PtxType::U32)
        .param("incx", PtxType::U32)
        .param("incy", PtxType::U32)
        .body(move |b| {
            // gid = blockIdx.x * blockDim.x + threadIdx.x
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            // Early exit: if gid >= n, return.
            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");
                let incx = b.load_param_u32("incx");
                let incy = b.load_param_u32("incy");

                // Compute x index: gid * incx
                let x_idx = b.mul_lo_u32(gid.clone(), incx);
                // Compute y index: gid * incy
                let y_idx = b.mul_lo_u32(gid, incy);

                // Compute addresses
                let x_addr = b.byte_offset_addr(x_ptr, x_idx, T::size_u32());
                let y_addr = b.byte_offset_addr(y_ptr.clone(), y_idx.clone(), T::size_u32());

                // Load alpha from bits
                let alpha_bits = b.load_param_u64("alpha_bits");
                let alpha = reinterpret_bits_to_float::<T>(b, alpha_bits);

                // Load x[i] and y[i]
                let x_val = load_global_float::<T>(b, x_addr);
                let y_val = load_global_float::<T>(b, y_addr);

                // result = fma(alpha, x_val, y_val) = alpha * x + y
                let result = fma_float::<T>(b, alpha, x_val, y_val);

                // Store result to y[i]
                let y_store_addr = b.byte_offset_addr(y_ptr, y_idx, T::size_u32());
                store_global_float::<T>(b, y_store_addr, result);
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Shared helpers for float operations in PTX body builders
// ---------------------------------------------------------------------------

/// Reinterprets a u64 register containing raw bits as a float register.
///
/// For f32, we truncate to the lower 32 bits, then mov to an f32 register.
/// For f64, we directly mov the u64 bits to an f64 register.
pub(crate) fn reinterpret_bits_to_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    bits: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        // Truncate u64 -> u32
        let bits32 = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("cvt.u32.u64 {bits32}, {bits};"));
        // Reinterpret u32 bits as f32
        let fval = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("mov.b32 {fval}, {bits32};"));
        fval
    } else {
        // f64: reinterpret u64 bits as f64
        let fval = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("mov.b64 {fval}, {bits};"));
        fval
    }
}

/// Loads a float value from a global memory address.
pub(crate) fn load_global_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, addr: Register) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.load_global_f32(addr)
    } else {
        b.load_global_f64(addr)
    }
}

/// Stores a float value to a global memory address.
pub(crate) fn store_global_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    addr: Register,
    val: Register,
) {
    if T::PTX_TYPE == PtxType::F32 {
        b.store_global_f32(addr, val);
    } else {
        b.store_global_f64(addr, val);
    }
}

/// Emits a fused multiply-add: `dst = a * b + c`.
pub(crate) fn fma_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
    c: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.fma_f32(a, bv, c)
    } else {
        b.fma_f64(a, bv, c)
    }
}

/// Emits a multiplication: `dst = a * b`.
pub(crate) fn mul_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        let dst = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("mul.rn.f32 {dst}, {a}, {bv};"));
        dst
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("mul.rn.f64 {dst}, {a}, {bv};"));
        dst
    }
}

/// Emits an addition: `dst = a + b`.
pub(crate) fn add_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.add_f32(a, bv)
    } else {
        b.add_f64(a, bv)
    }
}

/// Emits `abs(src)` for the given float type.
pub(crate) fn abs_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, src: Register) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.abs_f32(src)
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("abs.f64 {dst}, {src};"));
        dst
    }
}

/// Emits `max(a, b)` for the given float type.
#[allow(dead_code)]
pub(crate) fn max_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.max_f32(a, bv)
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("max.f64 {dst}, {a}, {bv};"));
        dst
    }
}

// ---------------------------------------------------------------------------
// Task 2: BLAS Level 1 mathematical formula correctness (pure Rust reference)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    /// Helper: assert two f32 slices are element-wise equal within epsilon.
    fn assert_close_f32(result: &[f32], expected: &[f32], epsilon: f32, label: &str) {
        assert_eq!(result.len(), expected.len(), "{label}: length mismatch");
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            let diff = (r - e).abs();
            assert!(
                diff <= epsilon,
                "{label}: element {i}: got {r}, expected {e}, diff={diff} > epsilon={epsilon}"
            );
        }
    }

    /// AXPY formula: y[i] = alpha * x[i] + y[i].
    /// alpha=2.0, x=[1,2,3], y=[4,5,6] → y=[6,9,12].
    #[test]
    fn axpy_formula_correctness_f32() {
        let alpha = 2.0_f32;
        let x = [1.0_f32, 2.0, 3.0];
        let y = [4.0_f32, 5.0, 6.0];
        let expected = [6.0_f32, 9.0, 12.0];
        let result: Vec<f32> = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| alpha * xi + yi)
            .collect();
        assert_close_f32(&result, &expected, 1e-7, "axpy f32");
    }

    /// AXPY with alpha=0 → no change (y unchanged).
    #[test]
    fn axpy_formula_alpha_zero_is_identity() {
        let alpha = 0.0_f32;
        let x = [1.0_f32, 2.0, 3.0];
        let y = [4.0_f32, 5.0, 6.0];
        let result: Vec<f32> = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| alpha * xi + yi)
            .collect();
        // alpha=0 → result = y unchanged
        assert_close_f32(&result, &y, 1e-7, "axpy alpha=0");
    }

    /// AXPY with alpha=-1 → y = y - x.
    #[test]
    fn axpy_formula_alpha_neg_one() {
        let alpha = -1.0_f32;
        let x = [3.0_f32, 2.0, 1.0];
        let y = [4.0_f32, 5.0, 6.0];
        // expected: [4-3, 5-2, 6-1] = [1, 3, 5]
        let expected = [1.0_f32, 3.0, 5.0];
        let result: Vec<f32> = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| alpha * xi + yi)
            .collect();
        assert_close_f32(&result, &expected, 1e-7, "axpy alpha=-1");
    }

    /// AXPY formula with f64 precision.
    #[test]
    fn axpy_formula_correctness_f64() {
        let alpha = 3.0_f64;
        let x = [1.0_f64, 0.5, -1.0];
        let y = [0.0_f64, 1.0, 2.0];
        // expected: [3*1+0, 3*0.5+1, 3*(-1)+2] = [3, 2.5, -1]
        let expected = [3.0_f64, 2.5, -1.0];
        for (i, ((&xi, &yi), &exp)) in x.iter().zip(y.iter()).zip(expected.iter()).enumerate() {
            let got = alpha * xi + yi;
            let diff = (got - exp).abs();
            assert!(
                diff <= 1e-14,
                "axpy f64 element {i}: got {got}, expected {exp}"
            );
        }
    }

    /// DOT product formula: sum(x[i] * y[i]).
    /// x=[1,2,3], y=[4,5,6] → 1*4+2*5+3*6 = 32.
    #[test]
    fn dot_product_formula_correctness() {
        let x = [1.0_f32, 2.0, 3.0];
        let y = [4.0_f32, 5.0, 6.0];
        let expected = 32.0_f32;
        let result: f32 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        assert!(
            (result - expected).abs() <= 1e-6,
            "dot: got {result}, expected {expected}"
        );
    }

    /// DOT of a vector with itself equals the squared Euclidean norm.
    #[test]
    fn dot_self_equals_squared_norm() {
        let x = [3.0_f32, 4.0];
        // ||[3,4]||^2 = 9 + 16 = 25
        let dot_self: f32 = x.iter().map(|xi| xi * xi).sum();
        assert!(
            (dot_self - 25.0).abs() <= 1e-6,
            "dot(x,x)={dot_self}, expected 25"
        );
    }

    /// DOT is commutative: dot(x, y) == dot(y, x).
    #[test]
    fn dot_product_is_commutative() {
        let x = [1.0_f32, -2.0, 3.0, -4.0];
        let y = [5.0_f32, 6.0, -7.0, 8.0];
        let dxy: f32 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let dyx: f32 = y.iter().zip(x.iter()).map(|(yi, xi)| yi * xi).sum();
        assert!(
            (dxy - dyx).abs() <= 1e-6,
            "dot must be commutative: {dxy} != {dyx}"
        );
    }

    /// NRM2 formula: sqrt(sum(x[i]^2)).
    /// x=[3,4] → sqrt(9+16) = 5.
    #[test]
    fn nrm2_formula_correctness_345() {
        let x = [3.0_f32, 4.0];
        let nrm: f32 = x.iter().map(|xi| xi * xi).sum::<f32>().sqrt();
        assert!(
            (nrm - 5.0_f32).abs() <= 1e-6,
            "nrm2([3,4])={nrm}, expected 5"
        );
    }

    /// NRM2 of a unit vector equals 1.
    #[test]
    fn nrm2_unit_vector_is_one() {
        let x = [1.0_f32, 0.0, 0.0];
        let nrm: f32 = x.iter().map(|xi| xi * xi).sum::<f32>().sqrt();
        assert!(
            (nrm - 1.0_f32).abs() <= 1e-7,
            "nrm2(unit vec)={nrm}, expected 1"
        );
    }

    /// NRM2 = sqrt(dot(x, x)).
    #[test]
    fn nrm2_equals_sqrt_of_dot_self() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let dot_self: f32 = x.iter().map(|xi| xi * xi).sum();
        let nrm_from_dot = dot_self.sqrt();
        let nrm_direct: f32 = x.iter().map(|xi| xi * xi).sum::<f32>().sqrt();
        assert!(
            (nrm_from_dot - nrm_direct).abs() <= 1e-6,
            "nrm2 should equal sqrt(dot(x,x))"
        );
    }

    /// NRM2 scaling: nrm2(alpha * x) = |alpha| * nrm2(x).
    #[test]
    fn nrm2_scales_linearly_with_alpha() {
        let x = [1.0_f32, 2.0, 3.0];
        let alpha = 3.0_f32;
        let nrm_x: f32 = x.iter().map(|xi| xi * xi).sum::<f32>().sqrt();
        let x_scaled: Vec<f32> = x.iter().map(|xi| alpha * xi).collect();
        let nrm_scaled: f32 = x_scaled.iter().map(|xi| xi * xi).sum::<f32>().sqrt();
        let expected = alpha.abs() * nrm_x;
        assert!(
            (nrm_scaled - expected).abs() <= 1e-5,
            "nrm2(alpha*x)={nrm_scaled}, expected |alpha|*nrm2(x)={expected}"
        );
    }

    /// SCAL formula: x[i] = alpha * x[i].
    #[test]
    fn scal_formula_correctness() {
        let alpha = 2.5_f32;
        let x = [1.0_f32, 2.0, -3.0, 0.0];
        let expected = [2.5_f32, 5.0, -7.5, 0.0];
        let result: Vec<f32> = x.iter().map(|xi| alpha * xi).collect();
        assert_close_f32(&result, &expected, 1e-6, "scal");
    }

    /// ASUM formula: sum(|x[i]|).
    #[test]
    fn asum_formula_correctness() {
        let x = [1.0_f32, -2.0, 3.0, -4.0];
        let expected = 10.0_f32; // 1+2+3+4
        let result: f32 = x.iter().map(|xi| xi.abs()).sum();
        assert!(
            (result - expected).abs() <= 1e-6,
            "asum: got {result}, expected {expected}"
        );
    }
}
