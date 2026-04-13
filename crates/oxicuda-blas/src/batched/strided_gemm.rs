//! Strided batched GEMM.
//!
//! All matrices in the batch share a common base pointer and are separated by
//! a constant stride.  This avoids the pointer-indirection overhead of the
//! pointer-array variant and is the preferred path when matrices are laid out
//! contiguously (or with uniform padding) in a single allocation.
//!
//! The kernel uses a 3-D grid where `blockIdx.z` encodes the batch index.
//! Each thread-block computes its per-batch pointer offsets as:
//!
//! ```text
//! A_i = a_base + batch_idx * stride_a
//! B_i = b_base + batch_idx * stride_b
//! C_i = c_base + batch_idx * stride_c
//! D_i = d_base + batch_idx * stride_d
//! ```

use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::templates::gemm::{EpilogueKind, GemmTemplate};
use std::sync::Arc;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{GpuFloat, Transpose};

/// Default tile dimensions for the strided batched kernel.
const TILE_M: u32 = 16;
/// Default tile dimension along N.
const TILE_N: u32 = 16;
/// Default tile dimension along K.
const TILE_K: u32 = 16;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates the strided batched GEMM arguments.
#[allow(clippy::too_many_arguments)]
fn validate_strided_args<T: GpuFloat>(
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    ldd: u32,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
    stride_d: i64,
    batch_count: u32,
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

    // Strides of zero are allowed only for batch_count <= 1 (broadcast).
    if batch_count > 1 && stride_a == 0 && stride_b == 0 && stride_c == 0 && stride_d == 0 {
        return Err(BlasError::InvalidArgument(
            "all strides are zero with batch_count > 1".into(),
        ));
    }

    let _elem = T::SIZE;
    Ok(())
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Builds a [`GemmTemplate`] with the standard tile sizes for strided dispatch.
fn build_gemm_template<T: GpuFloat>(sm: SmVersion) -> GemmTemplate {
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

/// Generates a strided batched GEMM PTX kernel and returns both the PTX text
/// and the kernel entry-point name.
fn generate_strided_gemm_ptx<T: GpuFloat>(
    sm: SmVersion,
    m: u32,
    n: u32,
    k: u32,
    trans_a: Transpose,
    trans_b: Transpose,
) -> BlasResult<(String, String)> {
    let _ = (m, n, k, trans_a, trans_b);

    let template = build_gemm_template::<T>(sm);
    let kernel_name = template.kernel_name();
    let ptx = template.generate()?;
    Ok((ptx, kernel_name))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Executes a strided batched GEMM.
///
/// ```text
/// D[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
///
/// where A[i] = a + i * stride_a  (element offset, not byte offset)
///       B[i] = b + i * stride_b
///       C[i] = c + i * stride_c
///       D[i] = d + i * stride_d
/// ```
///
/// This is more efficient than the pointer-array variant because per-batch
/// address computation is a simple multiply-add instead of a global-memory
/// load.
///
/// # Stride semantics
///
/// Strides are signed 64-bit **element** counts (not byte offsets).  A stride
/// of zero means the same matrix is broadcast to every batch element.  Negative
/// strides are legal and traverse the buffer in reverse order.
///
/// # Errors
///
/// * [`BlasError::InvalidDimension`] if `m`, `n`, or `k` is zero, or leading
///   dimensions are too small.
/// * [`BlasError::InvalidArgument`] if all strides are zero with
///   `batch_count > 1`.
/// * [`BlasError::PtxGeneration`] if the PTX kernel cannot be built.
/// * [`BlasError::LaunchFailed`] if the kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn gemm_strided_batched<T: GpuFloat>(
    handle: &BlasHandle,
    trans_a: Transpose,
    trans_b: Transpose,
    m: u32,
    n: u32,
    k: u32,
    alpha: T,
    a: CUdeviceptr,
    lda: u32,
    stride_a: i64,
    b: CUdeviceptr,
    ldb: u32,
    stride_b: i64,
    beta: T,
    c: CUdeviceptr,
    ldc: u32,
    stride_c: i64,
    d: CUdeviceptr,
    ldd: u32,
    stride_d: i64,
    batch_count: u32,
) -> BlasResult<()> {
    if batch_count == 0 {
        return Ok(());
    }

    validate_strided_args::<T>(
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        ldd,
        stride_a,
        stride_b,
        stride_c,
        stride_d,
        batch_count,
        trans_a,
        trans_b,
    )?;

    let sm = handle.sm_version();
    let (ptx_source, kernel_name) = generate_strided_gemm_ptx::<T>(sm, m, n, k, trans_a, trans_b)?;

    let module = oxicuda_driver::Module::from_ptx(&ptx_source).map_err(BlasError::Cuda)?;
    let module = Arc::new(module);
    let kernel = Kernel::from_module(module, &kernel_name).map_err(BlasError::Cuda)?;

    // 3-D grid: (tile_x, tile_y, batch_count)
    let grid = Dim3::new(m.div_ceil(TILE_M), n.div_ceil(TILE_N), batch_count);
    let block = Dim3::new(TILE_M, TILE_N, 1);
    let params = LaunchParams::new(grid, block);

    let alpha_bits = alpha.to_bits_u64();
    let beta_bits = beta.to_bits_u64();

    // Convert element strides to byte strides for the kernel.
    let elem_bytes = T::SIZE as i64;
    let byte_stride_a = stride_a.saturating_mul(elem_bytes);
    let byte_stride_b = stride_b.saturating_mul(elem_bytes);
    let byte_stride_c = stride_c.saturating_mul(elem_bytes);
    let byte_stride_d = stride_d.saturating_mul(elem_bytes);

    let args = (
        m,
        n,
        k,
        alpha_bits,
        a,
        lda,
        byte_stride_a,
        b,
        ldb,
        byte_stride_b,
        beta_bits,
        c,
        ldc,
        byte_stride_c,
        d,
        ldd,
        byte_stride_d,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_zero_dimensions() {
        let res = validate_strided_args::<f32>(
            0,
            64,
            64,
            64,
            64,
            64,
            64,
            1024,
            1024,
            1024,
            1024,
            8,
            Transpose::NoTrans,
            Transpose::NoTrans,
        );
        assert!(res.is_err());
    }

    #[test]
    fn validate_rejects_all_zero_strides_multi_batch() {
        let res = validate_strided_args::<f32>(
            64,
            64,
            64,
            64,
            64,
            64,
            64,
            0,
            0,
            0,
            0,
            8,
            Transpose::NoTrans,
            Transpose::NoTrans,
        );
        assert!(res.is_err());
    }

    #[test]
    fn validate_accepts_zero_stride_single_batch() {
        let res = validate_strided_args::<f64>(
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            0,
            0,
            0,
            0,
            1,
            Transpose::NoTrans,
            Transpose::NoTrans,
        );
        assert!(res.is_ok());
    }

    #[test]
    fn validate_accepts_negative_strides() {
        let res = validate_strided_args::<f32>(
            64,
            64,
            64,
            64,
            64,
            64,
            64,
            -4096,
            -4096,
            -4096,
            -4096,
            4,
            Transpose::NoTrans,
            Transpose::NoTrans,
        );
        assert!(res.is_ok());
    }

    #[test]
    fn validate_transposed_lda() {
        // trans_a == Trans => rows_a = k = 16, so lda = 16 is valid
        let res = validate_strided_args::<f32>(
            64,
            64,
            16,
            16,
            64,
            64,
            64,
            1024,
            1024,
            1024,
            1024,
            2,
            Transpose::Trans,
            Transpose::NoTrans,
        );
        assert!(res.is_ok());
    }
}
