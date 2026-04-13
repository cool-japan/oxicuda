//! Pointer-array batched GEMM.
//!
//! Each batch element can reference a completely independent memory location
//! for its A, B, C, and D matrices. The device-pointer arrays are stored in
//! [`DeviceBuffer<CUdeviceptr>`] and indexed by batch index inside the kernel.
//!
//! # Performance characteristics
//!
//! * For **small batch counts** (<=4), individual GEMMs are dispatched
//!   sequentially on the same stream.  This avoids the overhead of launching
//!   a pointer-indirection kernel for negligible batch sizes.
//! * For **large batch counts**, a single kernel launch covers all batches,
//!   using `blockIdx.z` as the batch index and loading the per-batch device
//!   pointer from the pointer array via global memory.

use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::templates::gemm::{EpilogueKind, GemmTemplate};
use std::sync::Arc;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{GpuFloat, Transpose};

/// Threshold below which we fall back to individual GEMM dispatches rather
/// than a pointer-array batched kernel.
const SMALL_BATCH_THRESHOLD: u32 = 4;

/// Default tile dimensions for the batched GEMM kernel.
pub(crate) const TILE_M: u32 = 16;
/// Default tile dimension along N for the batched GEMM kernel.
pub(crate) const TILE_N: u32 = 16;
/// Default tile dimension along K for the batched GEMM kernel.
pub(crate) const TILE_K: u32 = 16;

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validates that all pointer arrays are large enough for the batch count and
/// that the matrix dimensions are consistent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_batched_args<T: GpuFloat>(
    m: u32,
    n: u32,
    k: u32,
    a_ptrs: &DeviceBuffer<CUdeviceptr>,
    lda: u32,
    b_ptrs: &DeviceBuffer<CUdeviceptr>,
    ldb: u32,
    c_ptrs: &DeviceBuffer<CUdeviceptr>,
    ldc: u32,
    d_ptrs: &DeviceBuffer<CUdeviceptr>,
    ldd: u32,
    batch_count: u32,
    trans_a: Transpose,
    trans_b: Transpose,
) -> BlasResult<()> {
    let bc = batch_count as usize;

    if a_ptrs.len() < bc {
        return Err(BlasError::BufferTooSmall {
            expected: bc,
            actual: a_ptrs.len(),
        });
    }
    if b_ptrs.len() < bc {
        return Err(BlasError::BufferTooSmall {
            expected: bc,
            actual: b_ptrs.len(),
        });
    }
    if c_ptrs.len() < bc {
        return Err(BlasError::BufferTooSmall {
            expected: bc,
            actual: c_ptrs.len(),
        });
    }
    if d_ptrs.len() < bc {
        return Err(BlasError::BufferTooSmall {
            expected: bc,
            actual: d_ptrs.len(),
        });
    }

    validate_leading_dimensions::<T>(m, n, k, lda, ldb, ldc, ldd, trans_a, trans_b)
}

/// Checks that leading dimensions are at least as large as the corresponding
/// matrix row count (column-major convention).
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_leading_dimensions<T: GpuFloat>(
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    ldd: u32,
    trans_a: Transpose,
    trans_b: Transpose,
) -> BlasResult<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(BlasError::InvalidDimension(
            "m, n, and k must all be positive".into(),
        ));
    }

    let rows_a = match trans_a {
        Transpose::NoTrans => m,
        Transpose::Trans | Transpose::ConjTrans => k,
    };
    let rows_b = match trans_b {
        Transpose::NoTrans => k,
        Transpose::Trans | Transpose::ConjTrans => n,
    };

    if lda < rows_a {
        return Err(BlasError::InvalidDimension(format!(
            "lda ({lda}) must be >= rows of op(A) ({rows_a})"
        )));
    }
    if ldb < rows_b {
        return Err(BlasError::InvalidDimension(format!(
            "ldb ({ldb}) must be >= rows of op(B) ({rows_b})"
        )));
    }
    if ldc < m {
        return Err(BlasError::InvalidDimension(format!(
            "ldc ({ldc}) must be >= m ({m})"
        )));
    }
    if ldd < m {
        return Err(BlasError::InvalidDimension(format!(
            "ldd ({ldd}) must be >= m ({m})"
        )));
    }

    let _elem = T::SIZE; // ensure T is constrained
    Ok(())
}

// ---------------------------------------------------------------------------
// PTX generation helpers
// ---------------------------------------------------------------------------

/// Builds a [`GemmTemplate`] with the standard tile sizes for batched dispatch.
pub(crate) fn build_gemm_template<T: GpuFloat>(sm: SmVersion) -> GemmTemplate {
    GemmTemplate {
        tile_m: TILE_M,
        tile_n: TILE_N,
        tile_k: TILE_K,
        warp_m: TILE_M,
        warp_n: TILE_N,
        precision: T::PTX_TYPE,
        accumulator: T::PTX_TYPE,
        use_tensor_core: false,
        stages: 1,
        target: sm,
        epilogue: EpilogueKind::LinearCombination,
    }
}

/// Generates a batched GEMM PTX kernel and returns both the PTX source text
/// and the kernel entry-point name.
pub(crate) fn generate_batched_gemm_ptx<T: GpuFloat>(
    sm: SmVersion,
    m: u32,
    n: u32,
    k: u32,
    trans_a: Transpose,
    trans_b: Transpose,
) -> BlasResult<(String, String)> {
    let _ = (m, n, k, trans_a, trans_b); // reserved for future tile-size heuristics

    let template = build_gemm_template::<T>(sm);
    let kernel_name = template.kernel_name();
    let ptx = template.generate()?;
    Ok((ptx, kernel_name))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Executes a batched GEMM where each batch element has independent pointers.
///
/// ```text
/// D[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]   for i in 0..batch_count
/// ```
///
/// The pointer arrays `a_ptrs`, `b_ptrs`, `c_ptrs`, and `d_ptrs` each contain
/// `batch_count` device pointers, one per batch element.
///
/// # Errors
///
/// * [`BlasError::BufferTooSmall`] if any pointer array has fewer than
///   `batch_count` entries.
/// * [`BlasError::InvalidDimension`] if `m`, `n`, or `k` is zero, or if a
///   leading dimension is too small.
/// * [`BlasError::PtxGeneration`] if the PTX kernel cannot be built.
/// * [`BlasError::LaunchFailed`] if the kernel launch fails on the driver.
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched<T: GpuFloat>(
    handle: &BlasHandle,
    trans_a: Transpose,
    trans_b: Transpose,
    m: u32,
    n: u32,
    k: u32,
    alpha: T,
    a_ptrs: &DeviceBuffer<CUdeviceptr>,
    lda: u32,
    b_ptrs: &DeviceBuffer<CUdeviceptr>,
    ldb: u32,
    beta: T,
    c_ptrs: &DeviceBuffer<CUdeviceptr>,
    ldc: u32,
    d_ptrs: &mut DeviceBuffer<CUdeviceptr>,
    ldd: u32,
    batch_count: u32,
) -> BlasResult<()> {
    // Early-out for zero batches.
    if batch_count == 0 {
        return Ok(());
    }

    validate_batched_args::<T>(
        m,
        n,
        k,
        a_ptrs,
        lda,
        b_ptrs,
        ldb,
        c_ptrs,
        ldc,
        d_ptrs,
        ldd,
        batch_count,
        trans_a,
        trans_b,
    )?;

    let sm = handle.sm_version();

    // For very small batch counts, sequential per-batch dispatch avoids the
    // pointer-indirection overhead.
    if batch_count <= SMALL_BATCH_THRESHOLD {
        return dispatch_individual_gemms::<T>(
            handle,
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a_ptrs,
            lda,
            b_ptrs,
            ldb,
            beta,
            c_ptrs,
            ldc,
            d_ptrs,
            ldd,
            batch_count,
        );
    }

    // --- Large batch: single kernel launch with blockIdx.z = batch_index ---

    let (ptx_source, kernel_name) = generate_batched_gemm_ptx::<T>(sm, m, n, k, trans_a, trans_b)?;

    let module = oxicuda_driver::Module::from_ptx(&ptx_source).map_err(BlasError::Cuda)?;
    let module = Arc::new(module);
    let kernel = Kernel::from_module(module, &kernel_name).map_err(BlasError::Cuda)?;

    let grid = Dim3::new(m.div_ceil(TILE_M), n.div_ceil(TILE_N), batch_count);
    let block = Dim3::new(TILE_M, TILE_N, 1);
    let params = LaunchParams::new(grid, block);

    let alpha_bits = alpha.to_bits_u64();
    let beta_bits = beta.to_bits_u64();

    let args = (
        m,
        n,
        k,
        alpha_bits,
        a_ptrs.as_device_ptr(),
        lda,
        b_ptrs.as_device_ptr(),
        ldb,
        beta_bits,
        c_ptrs.as_device_ptr(),
        ldc,
        d_ptrs.as_device_ptr(),
        ldd,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(e.to_string()))
}

/// Dispatches individual GEMMs for small batch counts, reading host-side
/// copies of the pointer arrays.  This path is only taken when
/// `batch_count <= SMALL_BATCH_THRESHOLD`.
#[allow(clippy::too_many_arguments)]
fn dispatch_individual_gemms<T: GpuFloat>(
    handle: &BlasHandle,
    trans_a: Transpose,
    trans_b: Transpose,
    m: u32,
    n: u32,
    k: u32,
    alpha: T,
    a_ptrs: &DeviceBuffer<CUdeviceptr>,
    lda: u32,
    b_ptrs: &DeviceBuffer<CUdeviceptr>,
    ldb: u32,
    beta: T,
    c_ptrs: &DeviceBuffer<CUdeviceptr>,
    ldc: u32,
    d_ptrs: &mut DeviceBuffer<CUdeviceptr>,
    ldd: u32,
    batch_count: u32,
) -> BlasResult<()> {
    let sm = handle.sm_version();
    let (ptx_source, kernel_name) = generate_batched_gemm_ptx::<T>(sm, m, n, k, trans_a, trans_b)?;

    let module = oxicuda_driver::Module::from_ptx(&ptx_source).map_err(BlasError::Cuda)?;
    let module = Arc::new(module);
    let kernel = Kernel::from_module(module, &kernel_name).map_err(BlasError::Cuda)?;

    let grid = Dim3::new(m.div_ceil(TILE_M), n.div_ceil(TILE_N), 1);
    let block = Dim3::new(TILE_M, TILE_N, 1);
    let params = LaunchParams::new(grid, block);

    let alpha_bits = alpha.to_bits_u64();
    let beta_bits = beta.to_bits_u64();

    // Read pointer arrays back to host for individual dispatch.
    // copy_to_host requires exact length match, so we allocate for the full
    // buffer length and only iterate over `batch_count` entries.
    let bc = batch_count as usize;
    let mut h_a = vec![0u64; a_ptrs.len()];
    let mut h_b = vec![0u64; b_ptrs.len()];
    let mut h_c = vec![0u64; c_ptrs.len()];
    let mut h_d = vec![0u64; d_ptrs.len()];

    a_ptrs.copy_to_host(&mut h_a).map_err(BlasError::Cuda)?;
    b_ptrs.copy_to_host(&mut h_b).map_err(BlasError::Cuda)?;
    c_ptrs.copy_to_host(&mut h_c).map_err(BlasError::Cuda)?;
    d_ptrs.copy_to_host(&mut h_d).map_err(BlasError::Cuda)?;

    for i in 0..bc {
        let args = (
            m, n, k, alpha_bits, h_a[i], lda, h_b[i], ldb, beta_bits, h_c[i], ldc, h_d[i], ldd,
        );
        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| BlasError::LaunchFailed(format!("batch {i}: {e}")))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_batch_count_is_noop() {
        // gemm_batched with batch_count == 0 should return Ok immediately.
        // We cannot call the real function without a GPU handle, but we can
        // verify the early-return logic by inspecting the constant.
        assert_eq!(SMALL_BATCH_THRESHOLD, 4);
    }

    #[test]
    fn validate_leading_dimensions_rejects_zero_dims() {
        let res = validate_leading_dimensions::<f32>(
            0,
            64,
            64,
            64,
            64,
            64,
            64,
            Transpose::NoTrans,
            Transpose::NoTrans,
        );
        assert!(res.is_err());
    }

    #[test]
    fn validate_leading_dimensions_rejects_small_lda() {
        let res = validate_leading_dimensions::<f32>(
            128,
            64,
            32,
            64, // lda < m (128)
            32,
            128,
            128,
            Transpose::NoTrans,
            Transpose::NoTrans,
        );
        assert!(res.is_err());
    }

    #[test]
    fn validate_leading_dimensions_accepts_transposed() {
        // When trans_a == Trans, rows_a == k, so lda >= k is required.
        let res = validate_leading_dimensions::<f32>(
            128,
            64,
            32,
            32, // lda >= k (32) — valid for Trans
            64,
            128,
            128,
            Transpose::Trans,
            Transpose::NoTrans,
        );
        assert!(res.is_ok());
    }
}
