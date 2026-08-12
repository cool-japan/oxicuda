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
    let acc_zero = acc_type.zero_literal();
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

    // The value bank `%f` is declared in the accumulator precision so the f64
    // reduction (`add.f64`/`fma.rn.f64`/`ld.global.f64`/`st.global.f64`) matches
    // the register type that `ptxas` validates.
    write_line(&mut ptx, "    .reg .b32 %r<16>;")?;
    write_line(&mut ptx, "    .reg .b64 %rd<16>;")?;
    write_line(&mut ptx, &format!("    .reg {ty} %f<16>;"))?;
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
    write_line(&mut ptx, &format!("    mov{ty} %f0, {acc_zero};  // acc"))?;
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
// Split-K partial GEMM kernel generator
// ---------------------------------------------------------------------------

/// Generates a PTX kernel that computes one K-partition's worth of a GEMM
/// reduction into a scratch workspace.
///
/// This is the counterpart to [`generate_splitk_reduction_kernel`] and
/// together they form a genuine two-pass split-K GEMM, used by
/// [`super::dispatch::GemmDispatcher`] for GEMV-shaped problems (tiny `M*N`,
/// large `K` — e.g. ArcFace's `1x25088 @ 25088x512` embedding projection):
/// the single-pass tiled/naive kernel can only ever launch `M*N` total
/// threads (one per output element, each doing the *entire* `K`-length
/// reduction serially), which for a shape like that caps the launch at a
/// few hundred threads on hardware than can schedule tens of thousands
/// concurrently. This kernel instead lets `gridDim.z` threads split the
/// reduction itself: thread block `z` reduces only `A[.., k_start..k_end)`
/// against `B[k_start..k_end, ..]` and writes its partial dot product to
/// `workspace[z*M*N + row*N + col]` — `M*N` *times* `gridDim.z` independent
/// threads of work, all schedulable at once.
///
/// # Kernel signature
///
/// `(a_ptr, b_ptr, workspace_ptr, m, n, k_total, k_per_split)` — all six
/// integer parameters are unsigned 32-bit except the three pointers (64-bit).
///
/// * `a_ptr` — `M x k_total` row-major (the *full*, untruncated A; `k_total`
///   is A's row stride, so addressing a K-sub-range still needs the true K).
/// * `b_ptr` — `k_total x N` row-major.
/// * `workspace_ptr` — `gridDim.z x M x N` scratch buffer (partition-major);
///   every `(z, row, col)` triple is written by exactly one thread, so the
///   caller need not zero-initialise it first.
/// * `k_per_split` — elements of K each partition reduces; partition `z`
///   covers `[z*k_per_split, min((z+1)*k_per_split, k_total))`. The caller
///   derives this (and the `gridDim.z` partition count) from
///   [`SplitKConfig`], and must launch with `gridDim.z == split_factor`
///   exactly so every element of `k_total` is covered by exactly one
///   partition.
///
/// No `alpha`/`beta`/`C` here — this pass writes the *raw* partial dot
/// product; [`generate_splitk_reduction_kernel`] applies the epilogue once
/// all partitions are summed.
///
/// # Grid shape required at launch
///
/// `gridDim.z` **must** equal the caller's `split_factor`. `gridDim.x`,
/// `gridDim.y`, and `blockDim.x` are free to choose (`blockDim.y`/`.z` must
/// be `1`): the kernel grid-strides over the flattened `M*N` output within
/// each `z`-slice, so any positive thread count is correct, though a count
/// approaching `M*N` is what actually buys the occupancy this exists for.
///
/// # Errors
///
/// Returns [`BlasError::PtxGeneration`] if `acc_type` is not `F32`/`F64`, or
/// on formatting failure.
pub fn generate_splitk_partial_kernel(
    target: SmVersion,
    acc_type: PtxType,
) -> BlasResult<(String, String)> {
    if !matches!(acc_type, PtxType::F32 | PtxType::F64) {
        return Err(BlasError::PtxGeneration(format!(
            "split-K partial kernel requires F32 or F64 accumulator, got {}",
            acc_type.as_ptx_str()
        )));
    }

    let ty = acc_type.as_ptx_str();
    let acc_zero = acc_type.zero_literal();
    let byte_size = acc_type.size_bytes();
    let kernel_name = format!("splitk_partial_{}", ty.trim_start_matches('.'));

    let mut ptx = String::with_capacity(6144);

    write_line(&mut ptx, &format!(".version {}", target.ptx_version()))?;
    write_line(&mut ptx, &format!(".target {}", target.as_ptx_str()))?;
    write_line(&mut ptx, ".address_size 64")?;
    write_line(&mut ptx, "")?;

    write_line(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
    write_line(&mut ptx, "    .param .u64 %param_a,")?;
    write_line(&mut ptx, "    .param .u64 %param_b,")?;
    write_line(&mut ptx, "    .param .u64 %param_ws,")?;
    write_line(&mut ptx, "    .param .u32 %param_m,")?;
    write_line(&mut ptx, "    .param .u32 %param_n,")?;
    write_line(&mut ptx, "    .param .u32 %param_ktotal,")?;
    write_line(&mut ptx, "    .param .u32 %param_kpersplit")?;
    write_line(&mut ptx, ")")?;
    write_line(&mut ptx, "{")?;

    write_line(&mut ptx, "    .reg .b32 %r<40>;")?;
    write_line(&mut ptx, "    .reg .b64 %rd<24>;")?;
    write_line(&mut ptx, &format!("    .reg {ty} %f<8>;"))?;
    write_line(&mut ptx, "    .reg .pred %p<4>;")?;
    write_line(&mut ptx, "")?;

    write_line(&mut ptx, "    // Load parameters")?;
    write_line(&mut ptx, "    ld.param.u64 %rd0, [%param_a];")?;
    write_line(&mut ptx, "    ld.param.u64 %rd1, [%param_b];")?;
    write_line(&mut ptx, "    ld.param.u64 %rd2, [%param_ws];")?;
    write_line(&mut ptx, "    ld.param.u32 %r8, [%param_m];")?;
    write_line(&mut ptx, "    ld.param.u32 %r9, [%param_n];")?;
    write_line(&mut ptx, "    ld.param.u32 %r10, [%param_ktotal];")?;
    write_line(&mut ptx, "    ld.param.u32 %r11, [%param_kpersplit];")?;
    write_line(&mut ptx, "")?;

    // Partition index and its K-range: z selects a *disjoint* [k_start,
    // k_end) slice of the reduction, not an output tile — gridDim.z is the
    // split factor, one CTA-column of z per partition.
    write_line(&mut ptx, "    // Partition (K sub-range) this z-slice owns")?;
    write_line(&mut ptx, "    mov.u32 %r12, %ctaid.z;  // z")?;
    write_line(
        &mut ptx,
        "    mul.lo.u32 %r13, %r12, %r11;  // k_start = z * k_per_split",
    )?;
    write_line(&mut ptx, "    add.u32 %r14, %r13, %r11;")?;
    write_line(
        &mut ptx,
        "    min.u32 %r14, %r14, %r10;  // k_end = min(k_start + k_per_split, k_total)",
    )?;
    write_line(
        &mut ptx,
        "    setp.ge.u32 %p3, %r13, %r10;  // defensive: k_start >= k_total never happens",
    )?;
    write_line(
        &mut ptx,
        "    @%p3 bra $PARTIAL_DONE;  // for a correctly-sized launch",
    )?;
    write_line(&mut ptx, "")?;

    // Linear thread id *within this z-slice* (no ctaid.z term: z already
    // selects the K-partition, not a share of the M*N grid-stride space).
    write_line(
        &mut ptx,
        "    // linear_id = (ctaid.y*gridDim.x + ctaid.x)*blockDim.x + tid.x",
    )?;
    write_line(&mut ptx, "    mov.u32 %r0, %tid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r1, %ctaid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r2, %ctaid.y;")?;
    write_line(&mut ptx, "    mov.u32 %r4, %ntid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r5, %nctaid.x;")?;
    write_line(&mut ptx, "    mov.u32 %r6, %nctaid.y;")?;
    write_line(&mut ptx, "    mad.lo.u32 %r15, %r2, %r5, %r1;  // y*gx + x")?;
    write_line(
        &mut ptx,
        "    mad.lo.u32 %r16, %r15, %r4, %r0;  // *bdx + tid.x = linear_id",
    )?;
    write_line(
        &mut ptx,
        "    // total_threads_per_slice = gridDim.x*gridDim.y*blockDim.x",
    )?;
    write_line(&mut ptx, "    mul.lo.u32 %r17, %r5, %r6;")?;
    write_line(&mut ptx, "    mul.lo.u32 %r17, %r17, %r4;")?;
    write_line(&mut ptx, "")?;

    write_line(
        &mut ptx,
        "    // total_elems = m*n (64-bit: avoids overflow for m*n >= 2^32)",
    )?;
    write_line(&mut ptx, "    mul.wide.u32 %rd9, %r8, %r9;")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd10, %r16;  // idx = linear_id")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd11, %r17;  // stride")?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd12, %r9;   // n (64-bit)")?;
    write_line(
        &mut ptx,
        "    cvt.u64.u32 %rd20, %r12;  // z (64-bit) -> partition base offset",
    )?;
    write_line(
        &mut ptx,
        "    mul.lo.u64 %rd20, %rd20, %rd9;  // z * (m*n) elements",
    )?;
    write_line(&mut ptx, "")?;

    write_line(&mut ptx, "$PARTIAL_LOOP:")?;
    write_line(&mut ptx, "    setp.ge.u64 %p0, %rd10, %rd9;")?;
    write_line(&mut ptx, "    @%p0 bra $PARTIAL_DONE;")?;
    write_line(
        &mut ptx,
        "    div.u64 %rd13, %rd10, %rd12;  // row = idx / n",
    )?;
    write_line(
        &mut ptx,
        "    rem.u64 %rd14, %rd10, %rd12;  // col = idx % n",
    )?;
    write_line(&mut ptx, "    cvt.u32.u64 %r20, %rd13;")?;
    write_line(&mut ptx, "    cvt.u32.u64 %r21, %rd14;")?;
    write_line(&mut ptx, "")?;

    write_line(&mut ptx, &format!("    mov{ty} %f0, {acc_zero};  // acc"))?;
    write_line(&mut ptx, "    mov.u32 %r22, %r13;  // ki = k_start")?;
    write_line(&mut ptx, "")?;

    write_line(&mut ptx, "$KP_LOOP:")?;
    write_line(&mut ptx, "    setp.ge.u32 %p1, %r22, %r14;  // ki >= k_end")?;
    write_line(&mut ptx, "    @%p1 bra $KP_DONE;")?;
    write_line(
        &mut ptx,
        "    // A[row, ki] = a_ptr + (row*k_total + ki) * byte_size",
    )?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd15, %r22;")?;
    write_line(&mut ptx, "    mad.wide.u32 %rd15, %r20, %r10, %rd15;")?;
    write_line(
        &mut ptx,
        &format!("    mul.lo.u64 %rd15, %rd15, {byte_size};"),
    )?;
    write_line(&mut ptx, "    add.u64 %rd16, %rd0, %rd15;")?;
    write_line(&mut ptx, &format!("    ld.global{ty} %f1, [%rd16];"))?;
    write_line(&mut ptx, "")?;
    write_line(
        &mut ptx,
        "    // B[ki, col] = b_ptr + (ki*n + col) * byte_size",
    )?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd17, %r21;")?;
    write_line(&mut ptx, "    mad.wide.u32 %rd17, %r22, %r9, %rd17;")?;
    write_line(
        &mut ptx,
        &format!("    mul.lo.u64 %rd17, %rd17, {byte_size};"),
    )?;
    write_line(&mut ptx, "    add.u64 %rd18, %rd1, %rd17;")?;
    write_line(&mut ptx, &format!("    ld.global{ty} %f2, [%rd18];"))?;
    write_line(&mut ptx, "")?;
    write_line(&mut ptx, &format!("    fma.rn{ty} %f0, %f1, %f2, %f0;"))?;
    write_line(&mut ptx, "    add.u32 %r22, %r22, 1;")?;
    write_line(&mut ptx, "    bra $KP_LOOP;")?;
    write_line(&mut ptx, "$KP_DONE:")?;
    write_line(&mut ptx, "")?;

    write_line(
        &mut ptx,
        "    // workspace[z*m*n + row*n + col] = acc (raw partial sum, no alpha/beta)",
    )?;
    write_line(&mut ptx, "    cvt.u64.u32 %rd19, %r21;")?;
    write_line(&mut ptx, "    mad.wide.u32 %rd19, %r20, %r9, %rd19;")?;
    write_line(&mut ptx, "    add.u64 %rd19, %rd19, %rd20;")?;
    write_line(
        &mut ptx,
        &format!("    mul.lo.u64 %rd19, %rd19, {byte_size};"),
    )?;
    write_line(&mut ptx, "    add.u64 %rd21, %rd2, %rd19;")?;
    write_line(&mut ptx, &format!("    st.global{ty} [%rd21], %f0;"))?;
    write_line(&mut ptx, "")?;

    write_line(&mut ptx, "    add.u64 %rd10, %rd10, %rd11;")?;
    write_line(&mut ptx, "    bra $PARTIAL_LOOP;")?;
    write_line(&mut ptx, "")?;
    write_line(&mut ptx, "$PARTIAL_DONE:")?;
    write_line(&mut ptx, "    ret;")?;
    write_line(&mut ptx, "}")?;

    Ok((kernel_name, ptx))
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

    // ── generate_splitk_partial_kernel ──────────────────────────────────────

    #[test]
    fn generate_partial_f32() {
        let (name, ptx) =
            generate_splitk_partial_kernel(SmVersion::Sm86, PtxType::F32).expect("f32 partial");
        assert_eq!(name, "splitk_partial_f32");
        assert!(ptx.contains(".entry splitk_partial_f32"));
        assert!(ptx.contains("$PARTIAL_LOOP"));
        assert!(ptx.contains("$KP_LOOP"));
        assert!(ptx.contains("fma.rn.f32"));
        // Seven parameters: a, b, ws, m, n, k_total, k_per_split. (`.matches(".param")`
        // would also match every `ld.param.*` load instruction, so check the
        // declaration list by name instead of counting the substring.)
        for param in [
            "%param_a",
            "%param_b",
            "%param_ws",
            "%param_m",
            "%param_n",
            "%param_ktotal",
            "%param_kpersplit",
        ] {
            assert!(
                ptx.contains(&format!(".param .u64 {param}"))
                    || ptx.contains(&format!(".param .u32 {param}")),
                "missing parameter declaration for {param}"
            );
        }
        // No alpha/beta epilogue in the partial pass -- pure accumulate-and-store.
        assert!(!ptx.contains("%param_alpha"));
        assert!(!ptx.contains("%param_beta"));
    }

    #[test]
    fn generate_partial_f64() {
        let (name, ptx) =
            generate_splitk_partial_kernel(SmVersion::Sm86, PtxType::F64).expect("f64 partial");
        assert_eq!(name, "splitk_partial_f64");
        assert!(ptx.contains("fma.rn.f64"));
        assert!(ptx.contains("ld.global.f64"));
        assert!(ptx.contains("st.global.f64"));
    }

    #[test]
    fn generate_partial_invalid_type() {
        let result = generate_splitk_partial_kernel(SmVersion::Sm80, PtxType::U32);
        assert!(result.is_err());
    }

    #[test]
    fn generate_partial_is_ascii_only() {
        // Mirrors `oxicuda_ptx::templates::gemm`'s ASCII-only regression test: a
        // non-ASCII byte anywhere (even in a `//` comment) makes `ptxas` reject
        // the module on CUDA 12.9+.
        for acc in [PtxType::F32, PtxType::F64] {
            let (_, ptx) = generate_splitk_partial_kernel(SmVersion::Sm86, acc)
                .expect("partial kernel should generate");
            assert!(ptx.is_ascii(), "non-ASCII byte in generated partial PTX");
        }
    }

    #[test]
    fn generate_partial_uses_ctaid_z_as_partition_not_ctaid_x_or_y() {
        // The whole point of this kernel is that the K-partition comes from
        // `ctaid.z` (so gridDim.x/y/blockDim.x are free for M*N coverage,
        // orthogonal to gridDim.z's split-K role). Pin that down structurally.
        let (_, ptx) =
            generate_splitk_partial_kernel(SmVersion::Sm86, PtxType::F32).expect("f32 partial");
        assert!(
            ptx.contains("%ctaid.z"),
            "partition index must read %ctaid.z"
        );
    }
}
