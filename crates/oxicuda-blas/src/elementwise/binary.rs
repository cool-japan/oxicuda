//! Binary elementwise operations on device buffers.
//!
//! Each function operates on two input arrays and produces one output array.
//! PTX is generated via [`ElementwiseTemplate`], loaded into the driver, and
//! launched on the handle's stream.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::templates::elementwise::{ElementwiseOp as PtxOp, ElementwiseTemplate};

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Standard block size for 1-D elementwise kernels.
const BLOCK_SIZE: u32 = 256;

/// Validates that all three buffers (a, b, c) have at least `n` elements.
fn validate_binary_buffers<T: Copy>(
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &DeviceBuffer<T>,
) -> BlasResult<()> {
    let n_usize = n as usize;
    if a.len() < n_usize {
        return Err(BlasError::BufferTooSmall {
            expected: n_usize,
            actual: a.len(),
        });
    }
    if b.len() < n_usize {
        return Err(BlasError::BufferTooSmall {
            expected: n_usize,
            actual: b.len(),
        });
    }
    if c.len() < n_usize {
        return Err(BlasError::BufferTooSmall {
            expected: n_usize,
            actual: c.len(),
        });
    }
    Ok(())
}

/// Generates PTX for a binary op, loads the module, and returns the kernel.
fn build_binary_kernel(
    handle: &BlasHandle,
    ptx_op: PtxOp,
    ptx_type: oxicuda_ptx::ir::PtxType,
) -> BlasResult<(Kernel, String)> {
    let template = ElementwiseTemplate::new(ptx_op, ptx_type, handle.sm_version());
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("{}: {e}", ptx_op.as_str())))?;
    let module = Arc::new(Module::from_ptx(&ptx_source).map_err(|e| {
        BlasError::LaunchFailed(format!("module load for {}: {e}", ptx_op.as_str()))
    })?);
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Element-wise addition: `C[i] = A[i] + B[i]`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle bound to a CUDA context and stream.
/// * `n` -- number of elements to process.
/// * `a` -- first input device buffer (at least `n` elements).
/// * `b` -- second input device buffer (at least `n` elements).
/// * `c` -- output device buffer for the result (at least `n` elements).
///
/// # Errors
///
/// Returns [`BlasError::BufferTooSmall`] if any buffer has fewer than `n`
/// elements, or a PTX/launch error if kernel generation or execution fails.
pub fn add<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;

    let (kernel, _name) = build_binary_kernel(handle, PtxOp::Add, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("add: {e}")))?;
    Ok(())
}

/// Element-wise multiplication (Hadamard product): `C[i] = A[i] * B[i]`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn mul<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;

    let (kernel, _name) = build_binary_kernel(handle, PtxOp::Mul, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("mul: {e}")))?;
    Ok(())
}

/// Element-wise subtraction: `C[i] = A[i] - B[i]`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn sub<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::Sub, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("sub: {e}")))?;
    Ok(())
}

/// Element-wise division: `C[i] = A[i] / B[i]`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn div<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::Div, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("div: {e}")))?;
    Ok(())
}

/// Element-wise power: `C[i] = A[i]^B[i]`.
///
/// Uses a lg2+mul+ex2 approximation.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- base input device buffer.
/// * `b` -- exponent input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn pow<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::Pow, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("pow: {e}")))?;
    Ok(())
}

/// Element-wise minimum: `C[i] = min(A[i], B[i])`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn min<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::Min, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("min: {e}")))?;
    Ok(())
}

/// Element-wise maximum: `C[i] = max(A[i], B[i])`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn max<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::Max, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("max: {e}")))?;
    Ok(())
}

/// Comparison equal: `C[i] = (A[i] == B[i]) ? 1.0 : 0.0`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn cmp_eq<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::CmpEq, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("cmp_eq: {e}")))?;
    Ok(())
}

/// Comparison not-equal: `C[i] = (A[i] != B[i]) ? 1.0 : 0.0`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn cmp_ne<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::CmpNe, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("cmp_ne: {e}")))?;
    Ok(())
}

/// Comparison less-than: `C[i] = (A[i] < B[i]) ? 1.0 : 0.0`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn cmp_lt<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::CmpLt, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("cmp_lt: {e}")))?;
    Ok(())
}

/// Comparison greater-than: `C[i] = (A[i] > B[i]) ? 1.0 : 0.0`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn cmp_gt<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::CmpGt, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("cmp_gt: {e}")))?;
    Ok(())
}

/// Comparison less-or-equal: `C[i] = (A[i] <= B[i]) ? 1.0 : 0.0`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn cmp_le<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::CmpLe, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("cmp_le: {e}")))?;
    Ok(())
}

/// Comparison greater-or-equal: `C[i] = (A[i] >= B[i]) ? 1.0 : 0.0`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn cmp_ge<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::CmpGe, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("cmp_ge: {e}")))?;
    Ok(())
}

/// Fuzzy OR via max: `C[i] = max(A[i], B[i])`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn or_max<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::OrMax, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("or_max: {e}")))?;
    Ok(())
}

/// Probabilistic OR: `C[i] = A[i] + B[i] - A[i]*B[i]`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn or_prob_sum<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::OrProbSum, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("or_prob_sum: {e}")))?;
    Ok(())
}

/// Fuzzy NAND: `C[i] = 1 - A[i]*B[i]`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn nand<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::Nand, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("nand: {e}")))?;
    Ok(())
}

/// Fuzzy NOR: `C[i] = 1 - (A[i] + B[i] - A[i]*B[i])`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn nor<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::Nor, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("nor: {e}")))?;
    Ok(())
}

/// Fuzzy XOR: `C[i] = A[i] + B[i] - 2*A[i]*B[i]`.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn xor<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;
    let (kernel, _name) = build_binary_kernel(handle, PtxOp::Xor, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);
    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("xor: {e}")))?;
    Ok(())
}

/// Fused Add + ReLU: `C[i] = max(0, A[i] + B[i])`.
///
/// Performs element-wise addition followed by ReLU in a single kernel launch,
/// avoiding an extra global memory round-trip compared to separate add and
/// relu calls.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `a` -- first input device buffer.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn fused_add_relu<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    a: &DeviceBuffer<T>,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;

    let (kernel, _name) = build_binary_kernel(handle, PtxOp::FusedAddRelu, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);
    let args = (a.as_device_ptr(), b.as_device_ptr(), c.as_device_ptr(), n);

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("fused_add_relu: {e}")))?;
    Ok(())
}

/// Fused Scale-Add: `C[i] = alpha * A[i] + beta * B[i]`.
///
/// Combines scaling of two input arrays and their addition into a single
/// kernel launch. This is equivalent to the BLAS `axpby` pattern extended
/// to output a separate result buffer.
///
/// # Arguments
///
/// * `handle` -- BLAS handle.
/// * `n` -- number of elements.
/// * `alpha` -- scalar multiplier for `A`.
/// * `a` -- first input device buffer.
/// * `beta` -- scalar multiplier for `B`.
/// * `b` -- second input device buffer.
/// * `c` -- output device buffer.
///
/// # Errors
///
/// Returns [`BlasError`] on buffer validation or kernel launch failure.
pub fn fused_scale_add<T: GpuFloat>(
    handle: &BlasHandle,
    n: u32,
    alpha: T,
    a: &DeviceBuffer<T>,
    beta: T,
    b: &DeviceBuffer<T>,
    c: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }
    validate_binary_buffers(n, a, b, c)?;

    let (kernel, _name) = build_binary_kernel(handle, PtxOp::FusedScaleAdd, T::PTX_TYPE)?;
    let grid = grid_size_for(n, BLOCK_SIZE);
    let params = LaunchParams::new(grid, BLOCK_SIZE);

    // FusedScaleAdd kernel signature:
    //   (a_ptr, b_ptr, c_ptr, alpha_bits, beta_bits, n)
    let alpha_bits = alpha.to_bits_u64();
    let beta_bits = beta.to_bits_u64();
    let args = (
        a.as_device_ptr(),
        b.as_device_ptr(),
        c.as_device_ptr(),
        alpha_bits,
        beta_bits,
        n,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("fused_scale_add: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_is_power_of_two() {
        assert!(BLOCK_SIZE.is_power_of_two());
        const { assert!(BLOCK_SIZE >= 32) };
    }

    #[test]
    fn ptx_template_generates_add_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Add,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("add PTX generation should succeed");
        assert!(ptx.contains("elementwise_add_f32"));
    }

    #[test]
    fn ptx_template_generates_mul_f64() {
        let template = ElementwiseTemplate::new(
            PtxOp::Mul,
            oxicuda_ptx::ir::PtxType::F64,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("mul PTX generation should succeed");
        assert!(ptx.contains("elementwise_mul_f64"));
    }

    #[test]
    fn ptx_template_generates_fused_add_relu_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::FusedAddRelu,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("fused_add_relu PTX generation should succeed");
        assert!(ptx.contains("elementwise_fused_add_relu_f32"));
    }

    #[test]
    fn ptx_template_generates_fused_scale_add_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::FusedScaleAdd,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("fused_scale_add PTX generation should succeed");
        assert!(ptx.contains("elementwise_fused_scale_add_f32"));
    }

    #[test]
    fn ptx_template_generates_sub_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Sub,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("sub PTX generation should succeed");
        assert!(ptx.contains("elementwise_sub_f32"));
        assert!(ptx.contains("sub.f32"));
    }

    #[test]
    fn ptx_template_generates_div_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Div,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("div PTX generation should succeed");
        assert!(ptx.contains("elementwise_div_f32"));
        assert!(ptx.contains("div.rn.f32"));
    }

    #[test]
    fn ptx_template_generates_pow_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Pow,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("pow PTX generation should succeed");
        assert!(ptx.contains("elementwise_pow_f32"));
        assert!(ptx.contains("lg2.approx.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
    }

    #[test]
    fn ptx_template_generates_min_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Min,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("min PTX generation should succeed");
        assert!(ptx.contains("elementwise_min_f32"));
        assert!(ptx.contains("min.f32"));
    }

    #[test]
    fn ptx_template_generates_max_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Max,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("max PTX generation should succeed");
        assert!(ptx.contains("elementwise_max_f32"));
        assert!(ptx.contains("max.f32"));
    }

    #[test]
    fn ptx_template_generates_cmp_eq_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::CmpEq,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("cmp_eq PTX generation should succeed");
        assert!(ptx.contains("elementwise_cmp_eq_f32"));
        assert!(ptx.contains("setp.eq.f32"));
        assert!(ptx.contains("selp.f32"));
    }

    #[test]
    fn ptx_template_generates_cmp_ne_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::CmpNe,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("cmp_ne PTX generation should succeed");
        assert!(ptx.contains("elementwise_cmp_ne_f32"));
        assert!(ptx.contains("setp.ne.f32"));
    }

    #[test]
    fn ptx_template_generates_cmp_lt_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::CmpLt,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("cmp_lt PTX generation should succeed");
        assert!(ptx.contains("elementwise_cmp_lt_f32"));
        assert!(ptx.contains("setp.lt.f32"));
    }

    #[test]
    fn ptx_template_generates_cmp_gt_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::CmpGt,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("cmp_gt PTX generation should succeed");
        assert!(ptx.contains("elementwise_cmp_gt_f32"));
        assert!(ptx.contains("setp.gt.f32"));
    }

    #[test]
    fn ptx_template_generates_cmp_le_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::CmpLe,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("cmp_le PTX generation should succeed");
        assert!(ptx.contains("elementwise_cmp_le_f32"));
        assert!(ptx.contains("setp.le.f32"));
    }

    #[test]
    fn ptx_template_generates_cmp_ge_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::CmpGe,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("cmp_ge PTX generation should succeed");
        assert!(ptx.contains("elementwise_cmp_ge_f32"));
        assert!(ptx.contains("setp.ge.f32"));
    }

    #[test]
    fn ptx_template_generates_or_max_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::OrMax,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("or_max PTX generation should succeed");
        assert!(ptx.contains("elementwise_or_max_f32"));
        assert!(ptx.contains("max.f32"));
    }

    #[test]
    fn ptx_template_generates_or_prob_sum_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::OrProbSum,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("or_prob_sum PTX generation should succeed");
        assert!(ptx.contains("elementwise_or_prob_sum_f32"));
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("sub.f32"));
        assert!(ptx.contains("add.f32"));
    }

    #[test]
    fn ptx_template_generates_nand_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Nand,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("nand PTX generation should succeed");
        assert!(ptx.contains("elementwise_nand_f32"));
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("sub.f32"));
    }

    #[test]
    fn ptx_template_generates_nor_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Nor,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("nor PTX generation should succeed");
        assert!(ptx.contains("elementwise_nor_f32"));
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("add.f32"));
    }

    #[test]
    fn ptx_template_generates_xor_f32() {
        let template = ElementwiseTemplate::new(
            PtxOp::Xor,
            oxicuda_ptx::ir::PtxType::F32,
            oxicuda_ptx::arch::SmVersion::Sm80,
        );
        let ptx = template
            .generate()
            .expect("xor PTX generation should succeed");
        assert!(ptx.contains("elementwise_xor_f32"));
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("0f40000000")); // 2.0
    }
}
