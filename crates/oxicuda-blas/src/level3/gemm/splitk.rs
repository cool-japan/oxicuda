//! Split-K parallelisation for GEMM.
//!
//! When the K dimension is much larger than M and N, a single thread block
//! would iterate over a very long reduction loop. Split-K decomposes the
//! K dimension into `split_k` partitions, launches one slice of the grid
//! per partition, and then performs a final reduction to sum the partial
//! results into the output matrix C.
//!
//! # Workflow
//!
//! 1. **Partitioned GEMM**: Each grid-Z slice computes a partial GEMM over
//!    `K / split_k` elements and writes to a workspace buffer.
//! 2. **Reduction**: A separate kernel sums the `split_k` partial results
//!    into the final output.

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

use crate::error::{BlasError, BlasResult};

// ---------------------------------------------------------------------------
// SplitKConfig
// ---------------------------------------------------------------------------

/// Configuration for split-K GEMM execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitKConfig {
    /// Number of K-partitions. Must be >= 2.
    pub split_factor: u32,
    /// K elements per partition (rounded up).
    pub k_per_split: u32,
    /// Total K dimension.
    pub k_total: u32,
}

impl SplitKConfig {
    /// Computes a split-K configuration for the given K dimension.
    ///
    /// # Arguments
    ///
    /// * `k` — total K dimension.
    /// * `split_factor` — number of partitions (clamped to [2, k]).
    pub fn new(k: u32, split_factor: u32) -> Self {
        let factor = split_factor.clamp(2, k.max(2));
        let k_per_split = k.div_ceil(factor);
        Self {
            split_factor: factor,
            k_per_split,
            k_total: k,
        }
    }

    /// Returns the K-range (start, end) for the given partition index.
    ///
    /// The last partition may be shorter if K is not evenly divisible.
    pub fn partition_range(&self, partition_idx: u32) -> (u32, u32) {
        let start = partition_idx * self.k_per_split;
        let end = (start + self.k_per_split).min(self.k_total);
        (start, end)
    }

    /// Returns the number of elements in the workspace buffer needed
    /// for partial results: `M * N * split_factor`.
    pub fn workspace_elements(&self, m: u32, n: u32) -> u64 {
        u64::from(m) * u64::from(n) * u64::from(self.split_factor)
    }
}

// ---------------------------------------------------------------------------
// Split-K reduction kernel generator
// ---------------------------------------------------------------------------

/// Generates a PTX kernel that sums `split_factor` partial C matrices into
/// the final output.
///
/// The kernel is a simple elementwise sum: for each `(i, j)` in `[0, M*N)`,
/// it reads `split_factor` values from the workspace at stride `M*N` and
/// writes the sum (scaled by alpha) plus `beta * C_old` to the output.
///
/// # Arguments
///
/// * `target` — SM version for PTX header.
/// * `acc_type` — accumulator element type (F32 or F64).
/// * `split_factor` — number of partitions to sum.
///
/// # Errors
///
/// Returns [`BlasError::PtxGeneration`] on formatting failure.
pub fn generate_splitk_reduction_kernel(
    target: SmVersion,
    acc_type: PtxType,
    split_factor: u32,
) -> BlasResult<(String, String)> {
    if !matches!(acc_type, PtxType::F32 | PtxType::F64) {
        return Err(BlasError::PtxGeneration(format!(
            "split-K reduction requires F32 or F64 accumulator, got {}",
            acc_type.as_ptx_str()
        )));
    }

    let ty = acc_type.as_ptx_str();
    let byte_size = acc_type.size_bytes();
    let kernel_name = format!(
        "splitk_reduce_{}_x{}",
        ty.trim_start_matches('.'),
        split_factor
    );

    let mut ptx = String::with_capacity(4096);

    write_line(&mut ptx, &format!(".version {}", target.ptx_version()))?;
    write_line(&mut ptx, &format!(".target {}", target.as_ptx_str()))?;
    write_line(&mut ptx, ".address_size 64")?;
    write_line(&mut ptx, "")?;

    // Kernel: (workspace_ptr, c_ptr, mn_count, alpha, beta)
    write_line(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
    write_line(&mut ptx, "    .param .u64 %param_ws,")?;
    write_line(&mut ptx, "    .param .u64 %param_c,")?;
    write_line(&mut ptx, "    .param .u32 %param_mn,")?;
    write_line(&mut ptx, &format!("    .param {ty} %param_alpha,"))?;
    write_line(&mut ptx, &format!("    .param {ty} %param_beta"))?;
    write_line(&mut ptx, ")")?;
    write_line(&mut ptx, "{")?;

    write_line(&mut ptx, "    .reg .b32 %r<16>;")?;
    write_line(&mut ptx, "    .reg .b64 %rd<16>;")?;
    write_line(&mut ptx, "    .reg .f32 %f<16>;")?;
    write_line(&mut ptx, "    .reg .pred %p<4>;")?;
    write_line(&mut ptx, "")?;

    // Global index
    write_line(&mut ptx, "    mov.u32 %r0, %tid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r1, %ctaid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r2, %ntid.x;")?;
    write_line(
        &mut ptx,
        "    mad.lo.u32 %r3, %r1, %r2, %r0;  // global idx",
    )?;
    write_line(&mut ptx, "")?;

    // Bounds check
    write_line(&mut ptx, "    ld.param.u32 %r4, [%param_mn];")?;
    write_line(&mut ptx, "    setp.ge.u32 %p0, %r3, %r4;")?;
    write_line(&mut ptx, "    @%p0 bra $REDUCE_DONE;")?;
    write_line(&mut ptx, "")?;

    // Load pointers and scalars
    write_line(&mut ptx, "    ld.param.u64 %rd0, [%param_ws];")?;
    write_line(&mut ptx, "    ld.param.u64 %rd1, [%param_c];")?;
    write_line(&mut ptx, &format!("    ld.param{ty} %f8, [%param_alpha];"))?;
    write_line(&mut ptx, &format!("    ld.param{ty} %f9, [%param_beta];"))?;
    write_line(&mut ptx, "")?;

    // Compute byte offset for this element
    write_line(&mut ptx, "    cvt.u64.u32 %rd2, %r3;")?;
    write_line(
        &mut ptx,
        &format!("    mul.lo.u64 %rd2, %rd2, {byte_size};"),
    )?;

    // Stride between partitions in bytes: mn_count * byte_size
    write_line(&mut ptx, "    cvt.u64.u32 %rd3, %r4;")?;
    write_line(
        &mut ptx,
        &format!("    mul.lo.u64 %rd3, %rd3, {byte_size};  // partition stride"),
    )?;
    write_line(&mut ptx, "")?;

    // Sum partitions: acc = sum of workspace[i * mn + idx] for i in 0..split_factor
    write_line(&mut ptx, &format!("    mov{ty} %f0, 0f00000000;  // acc"))?;
    write_line(&mut ptx, "    add.u64 %rd4, %rd0, %rd2;  // ws + offset")?;
    for _ in 0..split_factor {
        write_line(&mut ptx, &format!("    ld.global{ty} %f1, [%rd4];"))?;
        write_line(&mut ptx, &format!("    add{ty} %f0, %f0, %f1;"))?;
        write_line(&mut ptx, "    add.u64 %rd4, %rd4, %rd3;")?;
    }
    write_line(&mut ptx, "")?;

    // C_out = alpha * acc + beta * C_old
    write_line(&mut ptx, "    add.u64 %rd5, %rd1, %rd2;  // c + offset")?;
    write_line(&mut ptx, &format!("    ld.global{ty} %f2, [%rd5];"))?;
    write_line(&mut ptx, &format!("    mul{ty} %f0, %f0, %f8;"))?;
    write_line(&mut ptx, &format!("    fma.rn{ty} %f0, %f9, %f2, %f0;"))?;
    write_line(&mut ptx, &format!("    st.global{ty} [%rd5], %f0;"))?;
    write_line(&mut ptx, "")?;

    write_line(&mut ptx, "$REDUCE_DONE:")?;
    write_line(&mut ptx, "    ret;")?;
    write_line(&mut ptx, "}")?;

    Ok((kernel_name, ptx))
}

/// Writes a line, mapping fmt errors.
fn write_line(ptx: &mut String, line: &str) -> BlasResult<()> {
    writeln!(ptx, "{line}").map_err(|e| BlasError::PtxGeneration(format!("fmt error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitk_config_basic() {
        let cfg = SplitKConfig::new(1024, 4);
        assert_eq!(cfg.split_factor, 4);
        assert_eq!(cfg.k_per_split, 256);
        assert_eq!(cfg.partition_range(0), (0, 256));
        assert_eq!(cfg.partition_range(3), (768, 1024));
    }

    #[test]
    fn splitk_config_uneven() {
        let cfg = SplitKConfig::new(1000, 3);
        assert_eq!(cfg.split_factor, 3);
        assert_eq!(cfg.k_per_split, 334);
        // Last partition: 668..1000 (332 elements, less than k_per_split).
        assert_eq!(cfg.partition_range(2), (668, 1000));
    }

    #[test]
    fn splitk_config_clamp_low() {
        let cfg = SplitKConfig::new(100, 1);
        assert_eq!(cfg.split_factor, 2); // Clamped to minimum of 2.
    }

    #[test]
    fn splitk_workspace_size() {
        let cfg = SplitKConfig::new(1024, 4);
        assert_eq!(cfg.workspace_elements(64, 64), 64 * 64 * 4);
    }

    #[test]
    fn generate_reduction_f32() {
        let (name, ptx) = generate_splitk_reduction_kernel(SmVersion::Sm80, PtxType::F32, 4)
            .expect("reduction kernel generation failed");
        assert_eq!(name, "splitk_reduce_f32_x4");
        assert!(ptx.contains(".entry splitk_reduce_f32_x4"));
        assert!(ptx.contains("$REDUCE_DONE"));
    }

    #[test]
    fn generate_reduction_invalid_type() {
        let result = generate_splitk_reduction_kernel(SmVersion::Sm80, PtxType::U32, 4);
        assert!(result.is_err());
    }
}
