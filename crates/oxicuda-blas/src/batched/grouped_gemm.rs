//! Grouped GEMM — heterogeneous batch of matrix multiplications.
//!
//! Unlike the uniform-size batched and strided variants, grouped GEMM allows
//! every problem in the batch to have **different** M, N, K dimensions,
//! leading dimensions, and matrix pointers.
//!
//! # Dispatch strategy
//!
//! * **Small groups** (up to `INDIVIDUAL_DISPATCH_LIMIT` = 4): each problem is
//!   dispatched as an independent GEMM kernel launch on the same stream.
//!   This avoids the complexity of a unified kernel for negligible group sizes.
//!
//! * **Large groups**: a unified kernel reads a *problem table* from device
//!   memory.  Each thread-block determines which problem it belongs to from
//!   a prefix-sum of per-problem grid sizes, loads the corresponding row of
//!   the table, and executes the GEMM tile.

use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::templates::gemm::{EpilogueKind, GemmTemplate};
use std::sync::Arc;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{GpuFloat, Transpose};

/// Groups with at most this many problems are dispatched as individual GEMM
/// kernel launches rather than through a unified kernel.
const INDIVIDUAL_DISPATCH_LIMIT: usize = 4;

/// Default tile dimensions shared by all problems in the unified kernel.
const TILE_M: u32 = 16;
/// Default tile dimension along N.
const TILE_N: u32 = 16;
/// Default tile dimension along K.
const TILE_K: u32 = 16;

// ---------------------------------------------------------------------------
// Problem descriptor
// ---------------------------------------------------------------------------

/// Describes a single GEMM problem within a grouped batch.
///
/// Each field mirrors the corresponding parameter of a standalone GEMM call.
/// All pointer fields are raw device pointers (`CUdeviceptr`) that must be
/// valid for the lifetime of the grouped GEMM execution.
#[derive(Debug, Clone, Copy)]
pub struct GroupedGemmProblem {
    /// Transpose mode for the A operand.
    pub trans_a: Transpose,
    /// Transpose mode for the B operand.
    pub trans_b: Transpose,
    /// Number of rows of op(A) and D.
    pub m: u32,
    /// Number of columns of op(B) and D.
    pub n: u32,
    /// Inner dimension: columns of op(A) and rows of op(B).
    pub k: u32,
    /// Device pointer to matrix A.
    pub a_ptr: CUdeviceptr,
    /// Leading dimension of A.
    pub lda: u32,
    /// Device pointer to matrix B.
    pub b_ptr: CUdeviceptr,
    /// Leading dimension of B.
    pub ldb: u32,
    /// Device pointer to matrix C (input for beta scaling).
    pub c_ptr: CUdeviceptr,
    /// Leading dimension of C.
    pub ldc: u32,
    /// Device pointer to matrix D (output).
    pub d_ptr: CUdeviceptr,
    /// Leading dimension of D.
    pub ldd: u32,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates a single problem in the group.
fn validate_problem<T: GpuFloat>(problem: &GroupedGemmProblem) -> BlasResult<()> {
    if problem.m == 0 || problem.n == 0 || problem.k == 0 {
        return Err(BlasError::InvalidDimension(
            "m, n, and k must all be positive in every grouped problem".into(),
        ));
    }

    let rows_a = match problem.trans_a {
        Transpose::NoTrans => problem.m,
        Transpose::Trans | Transpose::ConjTrans => problem.k,
    };
    let rows_b = match problem.trans_b {
        Transpose::NoTrans => problem.k,
        Transpose::Trans | Transpose::ConjTrans => problem.n,
    };

    if problem.lda < rows_a {
        return Err(BlasError::InvalidDimension(format!(
            "lda ({}) must be >= rows of op(A) ({rows_a})",
            problem.lda
        )));
    }
    if problem.ldb < rows_b {
        return Err(BlasError::InvalidDimension(format!(
            "ldb ({}) must be >= rows of op(B) ({rows_b})",
            problem.ldb
        )));
    }
    if problem.ldc < problem.m {
        return Err(BlasError::InvalidDimension(format!(
            "ldc ({}) must be >= m ({})",
            problem.ldc, problem.m
        )));
    }
    if problem.ldd < problem.m {
        return Err(BlasError::InvalidDimension(format!(
            "ldd ({}) must be >= m ({})",
            problem.ldd, problem.m
        )));
    }

    let _elem = T::SIZE;
    Ok(())
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Builds a [`GemmTemplate`] with the standard tile sizes.
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

/// Generates a GEMM PTX kernel and returns both the PTX text and kernel name.
fn generate_gemm_ptx<T: GpuFloat>(sm: SmVersion) -> BlasResult<(String, String)> {
    let template = build_gemm_template::<T>(sm);
    let kernel_name = template.kernel_name();
    let ptx = template.generate()?;
    Ok((ptx, kernel_name))
}

// ---------------------------------------------------------------------------
// Per-problem dispatch (small groups)
// ---------------------------------------------------------------------------

/// Dispatches each problem individually as a separate kernel launch.
fn dispatch_individual<T: GpuFloat>(
    handle: &BlasHandle,
    problems: &[GroupedGemmProblem],
    alpha: T,
    beta: T,
) -> BlasResult<()> {
    let sm = handle.sm_version();
    let (ptx_source, kernel_name) = generate_gemm_ptx::<T>(sm)?;

    let module = oxicuda_driver::Module::from_ptx(&ptx_source).map_err(BlasError::Cuda)?;
    let module = Arc::new(module);
    let kernel = Kernel::from_module(Arc::clone(&module), &kernel_name).map_err(BlasError::Cuda)?;

    let alpha_bits = alpha.to_bits_u64();
    let beta_bits = beta.to_bits_u64();

    for (idx, p) in problems.iter().enumerate() {
        let grid = Dim3::new(p.m.div_ceil(TILE_M), p.n.div_ceil(TILE_N), 1);
        let block = Dim3::new(TILE_M, TILE_N, 1);
        let params = LaunchParams::new(grid, block);

        let args = (
            p.m, p.n, p.k, alpha_bits, p.a_ptr, p.lda, p.b_ptr, p.ldb, beta_bits, p.c_ptr, p.ldc,
            p.d_ptr, p.ldd,
        );
        kernel
            .launch(&params, handle.stream(), &args)
            .map_err(|e| BlasError::LaunchFailed(format!("grouped problem {idx}: {e}")))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Unified dispatch (large groups)
// ---------------------------------------------------------------------------

/// Packs the problem descriptors into a flat `u32` array suitable for upload
/// to device memory.  Each problem occupies a fixed-size row so the kernel
/// can index with `problem_idx * ROW_SIZE`.
///
/// Row layout (13 u32s per problem):
///   [m, n, k, lda, ldb, ldc, ldd, a_lo, a_hi, b_lo, b_hi, c_lo, c_hi, d_lo, d_hi, trans_flags]
const PROBLEM_ROW_U32S: usize = 16;

fn pack_problem_table(problems: &[GroupedGemmProblem]) -> Vec<u32> {
    let mut table = Vec::with_capacity(problems.len() * PROBLEM_ROW_U32S);

    for p in problems {
        table.push(p.m);
        table.push(p.n);
        table.push(p.k);
        table.push(p.lda);
        table.push(p.ldb);
        table.push(p.ldc);
        table.push(p.ldd);
        // Split 64-bit device pointers into two 32-bit halves.
        table.push(p.a_ptr as u32);
        table.push((p.a_ptr >> 32) as u32);
        table.push(p.b_ptr as u32);
        table.push((p.b_ptr >> 32) as u32);
        table.push(p.c_ptr as u32);
        table.push((p.c_ptr >> 32) as u32);
        table.push(p.d_ptr as u32);
        table.push((p.d_ptr >> 32) as u32);
        // Encode transpose flags: bits [0] = trans_a, bits [1] = trans_b.
        let trans_flags = encode_transpose(p.trans_a) | (encode_transpose(p.trans_b) << 2);
        table.push(trans_flags);
    }

    table
}

/// Encodes a [`Transpose`] variant into a 2-bit value.
fn encode_transpose(t: Transpose) -> u32 {
    match t {
        Transpose::NoTrans => 0,
        Transpose::Trans => 1,
        Transpose::ConjTrans => 2,
    }
}

/// Computes a prefix sum of per-problem grid sizes so the unified kernel can
/// map `blockIdx.x` to the correct problem.
fn compute_block_prefix_sums(problems: &[GroupedGemmProblem]) -> Vec<u32> {
    let mut prefix = Vec::with_capacity(problems.len() + 1);
    prefix.push(0u32);
    for p in problems {
        let blocks_m = p.m.div_ceil(TILE_M);
        let blocks_n = p.n.div_ceil(TILE_N);
        let last = prefix.last().copied().unwrap_or(0);
        prefix.push(last.saturating_add(blocks_m.saturating_mul(blocks_n)));
    }
    prefix
}

/// Dispatches all problems via a unified kernel that reads a problem table
/// from device memory.
fn dispatch_unified<T: GpuFloat>(
    handle: &BlasHandle,
    problems: &[GroupedGemmProblem],
    alpha: T,
    beta: T,
) -> BlasResult<()> {
    let sm = handle.sm_version();
    let (ptx_source, kernel_name) = generate_gemm_ptx::<T>(sm)?;

    let module = oxicuda_driver::Module::from_ptx(&ptx_source).map_err(BlasError::Cuda)?;
    let module = Arc::new(module);
    let kernel = Kernel::from_module(module, &kernel_name).map_err(BlasError::Cuda)?;

    // Upload the problem table to device memory.
    let table_host = pack_problem_table(problems);
    let mut table_device = DeviceBuffer::<u32>::alloc(table_host.len()).map_err(BlasError::Cuda)?;
    table_device
        .copy_from_host(&table_host)
        .map_err(BlasError::Cuda)?;

    // Upload the prefix-sum array so the kernel can binary-search for the
    // problem index from its global block index.
    let prefix_host = compute_block_prefix_sums(problems);
    let total_blocks = prefix_host.last().copied().unwrap_or(0);
    let mut prefix_device =
        DeviceBuffer::<u32>::alloc(prefix_host.len()).map_err(BlasError::Cuda)?;
    prefix_device
        .copy_from_host(&prefix_host)
        .map_err(BlasError::Cuda)?;

    let grid = Dim3::new(total_blocks, 1, 1);
    let block = Dim3::new(TILE_M, TILE_N, 1);
    let params = LaunchParams::new(grid, block);

    let alpha_bits = alpha.to_bits_u64();
    let beta_bits = beta.to_bits_u64();

    let args = (
        problems.len() as u32,
        alpha_bits,
        beta_bits,
        table_device.as_device_ptr(),
        prefix_device.as_device_ptr(),
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(e.to_string()))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Executes a grouped GEMM — a batch of matrix multiplications where each
/// problem may have different M, N, K, leading dimensions, and pointers.
///
/// ```text
/// D[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
/// ```
///
/// All problems share the same `alpha` and `beta` scalars.  If per-problem
/// scalars are needed, fold them into the data or use separate calls.
///
/// # Dispatch strategy
///
/// * For groups with up to 4 problems, individual kernel launches are used.
/// * For larger groups, a unified kernel reads a device-side problem table
///   and uses a prefix-sum to map thread blocks to problems.
///
/// # Errors
///
/// * [`BlasError::InvalidArgument`] if `problems` is empty.
/// * [`BlasError::InvalidDimension`] if any problem has invalid dimensions or
///   leading dimensions.
/// * [`BlasError::PtxGeneration`] if the PTX kernel cannot be built.
/// * [`BlasError::LaunchFailed`] if any kernel launch fails.
pub fn gemm_grouped<T: GpuFloat>(
    handle: &BlasHandle,
    problems: &[GroupedGemmProblem],
    alpha: T,
    beta: T,
) -> BlasResult<()> {
    if problems.is_empty() {
        return Ok(());
    }

    // Validate every problem before launching anything.
    for (idx, problem) in problems.iter().enumerate() {
        validate_problem::<T>(problem)
            .map_err(|e| BlasError::InvalidArgument(format!("grouped problem {idx}: {e}")))?;
    }

    if problems.len() <= INDIVIDUAL_DISPATCH_LIMIT {
        dispatch_individual::<T>(handle, problems, alpha, beta)
    } else {
        dispatch_unified::<T>(handle, problems, alpha, beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_problem(m: u32, n: u32, k: u32) -> GroupedGemmProblem {
        GroupedGemmProblem {
            trans_a: Transpose::NoTrans,
            trans_b: Transpose::NoTrans,
            m,
            n,
            k,
            a_ptr: 0x1000,
            lda: m,
            b_ptr: 0x2000,
            ldb: k,
            c_ptr: 0x3000,
            ldc: m,
            d_ptr: 0x4000,
            ldd: m,
        }
    }

    #[test]
    fn validate_rejects_zero_dimension() {
        let p = make_problem(0, 64, 64);
        assert!(validate_problem::<f32>(&p).is_err());
    }

    #[test]
    fn validate_accepts_valid_problem() {
        let p = make_problem(128, 64, 32);
        assert!(validate_problem::<f32>(&p).is_ok());
    }

    #[test]
    fn encode_transpose_round_trip() {
        assert_eq!(encode_transpose(Transpose::NoTrans), 0);
        assert_eq!(encode_transpose(Transpose::Trans), 1);
        assert_eq!(encode_transpose(Transpose::ConjTrans), 2);
    }

    #[test]
    fn pack_problem_table_row_size() {
        let problems = vec![make_problem(64, 64, 64)];
        let table = pack_problem_table(&problems);
        assert_eq!(table.len(), PROBLEM_ROW_U32S);
    }

    #[test]
    fn prefix_sums_correct() {
        let problems = vec![make_problem(32, 32, 16), make_problem(64, 64, 16)];
        let prefix = compute_block_prefix_sums(&problems);
        // problem 0: ceil(32/16) * ceil(32/16) = 2 * 2 = 4 blocks
        // problem 1: ceil(64/16) * ceil(64/16) = 4 * 4 = 16 blocks
        assert_eq!(prefix, vec![0, 4, 20]);
    }

    #[test]
    fn validate_transposed_problem() {
        let p = GroupedGemmProblem {
            trans_a: Transpose::Trans,
            trans_b: Transpose::Trans,
            m: 64,
            n: 32,
            k: 16,
            a_ptr: 0x1000,
            lda: 16, // rows_a = k = 16
            b_ptr: 0x2000,
            ldb: 32, // rows_b = n = 32
            c_ptr: 0x3000,
            ldc: 64,
            d_ptr: 0x4000,
            ldd: 64,
        };
        assert!(validate_problem::<f32>(&p).is_ok());
    }
}
