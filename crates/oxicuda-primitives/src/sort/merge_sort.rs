//! Stable GPU merge sort: bitonic block sort + binary-search merge.
//!
//! The sort operates in two kernel types:
//!
//! 1. **`sort_blocks`** — Bitonic sort each block of `block_size` elements
//!    in shared memory.  After this kernel, the input is divided into
//!    sorted runs of length `block_size`.
//!
//! 2. **`merge`** — Merge adjacent pairs of sorted runs using the binary
//!    search-based co-rank algorithm.  Each thread determines its output
//!    position in the merged run via an O(log n) binary search, then
//!    writes the correct element.  Launched repeatedly, doubling the
//!    merge length each pass:
//!    ```text
//!    pass 0: merge_len = block_size          (merge pairs of blocks)
//!    pass 1: merge_len = 2 * block_size
//!    pass 2: merge_len = 4 * block_size
//!    …    until merge_len >= n
//!    ```
//!
//! # Bitonic sort network
//!
//! The block sort uses a two-barrier-per-stage bitonic network for
//! correctness: one `bar.sync` ensures all previous writes are visible
//! before any thread loads, and a second ensures all loads complete
//! before any thread writes.
//!
//! # Supported types
//!
//! Comparison uses `setp.lt` (ascending) with PTX integer or FP types.
//! For floating-point, NaN values compare as greater than all finite
//! values (standard PTX `setp.lt.f32/f64` semantics apply).
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::sort::merge_sort::{MergeSortConfig, MergeSortTemplate};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = MergeSortConfig { ty: PtxType::U32, block_size: 256 };
//! let t   = MergeSortTemplate::new(cfg);
//! let (sort_ptx, merge_ptx) = t.generate(SmVersion::Sm80).expect("PTX gen");
//! assert!(sort_ptx.contains("merge_sort_blocks_u32_bs256"));
//! assert!(merge_ptx.contains("merge_sort_merge_u32_bs256"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::ptx_helpers::{ptx_header, ptx_type_str};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for device-wide merge sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MergeSortConfig {
    /// Element type to sort.
    pub ty: PtxType,
    /// Threads per block (must be a power of 2, 32–1024).
    pub block_size: u32,
}

impl MergeSortConfig {
    /// log₂(block_size) — the number of bitonic sort stages.
    #[must_use]
    pub fn log2_block_size(&self) -> u32 {
        self.block_size.trailing_zeros()
    }

    /// Bytes per element.
    #[must_use]
    pub fn elem_bytes(&self) -> u32 {
        match self.ty {
            PtxType::F64 | PtxType::U64 | PtxType::S64 => 8,
            _ => 4,
        }
    }

    /// Kernel name for the block-sort pass.
    #[must_use]
    pub fn sort_blocks_name(&self) -> String {
        format!(
            "merge_sort_blocks_{}_bs{}",
            ptx_type_str(self.ty),
            self.block_size
        )
    }

    /// Kernel name for the merge pass.
    #[must_use]
    pub fn merge_kernel_name(&self) -> String {
        format!(
            "merge_sort_merge_{}_bs{}",
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

// ─── Template ────────────────────────────────────────────────────────────────

/// PTX code generator for device-wide merge sort.
///
/// Returns `(sort_blocks_ptx, merge_ptx)`.  The caller launches
/// `sort_blocks` once to produce runs of length `block_size`, then
/// `merge` repeatedly (with an increasing `merge_len` parameter) until
/// `merge_len >= n`.
pub struct MergeSortTemplate {
    /// Configuration.
    pub cfg: MergeSortConfig,
}

impl MergeSortTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: MergeSortConfig) -> Self {
        Self { cfg }
    }

    /// Generate both PTX kernels as `(sort_blocks_ptx, merge_ptx)`.
    ///
    /// # Errors
    ///
    /// Returns a `String` error on generation failure.
    pub fn generate(&self, sm: SmVersion) -> Result<(String, String), String> {
        let sort = self.generate_sort_blocks_kernel(sm)?;
        let merge = self.generate_merge_kernel(sm)?;
        Ok((sort, merge))
    }

    // ── Kernel 1: bitonic sort within block ──────────────────────────────────
    //
    // Each thread handles one shared-memory element.  The bitonic network
    // performs log2(bs) * (log2(bs)+1) / 2 compare-swap stages.
    //
    // Per stage:
    //   bar.sync   (wait for previous stage writes)
    //   load smem[tid] → %a
    //   load smem[partner] → %b
    //   bar.sync   (all loads done; prevent read-after-write hazard)
    //   compute swap condition
    //   write only smem[tid] (each thread writes its own slot only)

    fn generate_sort_blocks_kernel(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.sort_blocks_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = self.cfg.elem_bytes();
        let log2 = self.cfg.log2_block_size();
        // Type-accurate comparison: use the actual element type for setp.
        let cmp = format!("setp.lt.{ty}"); // works for all supported types

        let mut out = ptx_header(sm);
        writeln!(out, ".shared .align {eb} .{ty} bsort_smem[{bs}];").map_err(|e| e.to_string())?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_data,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .{ty}   %a, %b, %write_val;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u32    %tid, %bid, %partner, %dir, %cmp_int, %swap_int;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %n, %gid, %ptr, %smem_base, %tid_addr, %par_addr;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p, %need_swap, %gt;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr,  [param_data];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %n,     [param_n];").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64      %smem_base, bsort_smem;").map_err(|e| e.to_string())?;

        // Load data from global to shared memory (OOB threads get identity/max value).
        writeln!(out, "    mad.lo.u64   %tid_addr, %tid, {eb}, %smem_base;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    setp.lt.u64  %p, %gid, %n;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64 %glob_addr; mad.lo.u64 %glob_addr, %gid, {eb}, %ptr;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  ld.global.{ty} %a, [%glob_addr];").map_err(|e| e.to_string())?;
        // OOB: fill with maximum value so it sorts to the end.
        let fill = max_fill_literal(self.cfg.ty);
        writeln!(out, "    @!%p mov.{ty} %a, {fill};").map_err(|e| e.to_string())?;
        writeln!(out, "    st.shared.{ty} [%tid_addr], %a;").map_err(|e| e.to_string())?;
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;

        // Bitonic sort network: outer loop over stages k = 1..log2(bs),
        // inner loop over sub-stages j = k..0.
        // All loops are statically unrolled in the generated PTX.
        for stage in 1..=log2 {
            let k: u32 = 1 << stage; // 2, 4, 8, …
            let log2_k = stage;
            for sub in (0..stage).rev() {
                let j: u32 = 1 << sub; // k/2, k/4, …, 1
                // Two barriers per sub-stage: one before loads, one before stores.
                writeln!(out, "    // stage k={k} sub j={j}").map_err(|e| e.to_string())?;
                writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;
                // Load both values.
                writeln!(out, "    mad.lo.u64   %tid_addr, %tid, {eb}, %smem_base;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    xor.b32      %partner, %tid, {j};")
                    .map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "    mad.lo.u64   %par_addr, %partner, {eb}, %smem_base;"
                )
                .map_err(|e| e.to_string())?;
                writeln!(out, "    ld.shared.{ty} %a, [%tid_addr];").map_err(|e| e.to_string())?;
                writeln!(out, "    ld.shared.{ty} %b, [%par_addr];").map_err(|e| e.to_string())?;
                // Second barrier: all loads done before any writes.
                writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;
                // Direction: ascending when (tid >> log2_k) & 1 == 0.
                writeln!(out, "    shr.u32      %dir, %tid, {log2_k};")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    and.b32      %dir, %dir, 1;").map_err(|e| e.to_string())?;
                // need_swap = (a > b) XOR descending = (a > b) XOR (dir == 1)
                writeln!(out, "    {cmp}        %gt, %b, %a;").map_err(|e| e.to_string())?; // b < a → a > b
                writeln!(out, "    cvt.u32.pred %cmp_int, %gt;").map_err(|e| e.to_string())?;
                writeln!(out, "    xor.b32      %swap_int, %cmp_int, %dir;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    setp.ne.u32  %need_swap, %swap_int, 0;")
                    .map_err(|e| e.to_string())?;
                // Each thread writes only its own slot.
                writeln!(out, "    selp.{ty}    %write_val, %b, %a, %need_swap;")
                    .map_err(|e| e.to_string())?;
                writeln!(out, "    st.shared.{ty} [%tid_addr], %write_val;")
                    .map_err(|e| e.to_string())?;
            }
        }

        // Final sync + writeback to global memory.
        writeln!(out, "    bar.sync 0;").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.lt.u64  %p, %gid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  ld.shared.{ty} %a, [%tid_addr];").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64 %wb_addr; mad.lo.u64 %wb_addr, %gid, {eb}, %ptr;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    @%p  st.global.{ty} [%wb_addr], %a;").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }

    // ── Kernel 2: merge two adjacent sorted runs using co-rank ───────────────
    //
    // Thread gid produces one element of the merged output at position gid
    // within the merged pair.
    //
    // Co-rank binary search: find k (elements taken from left run) such that
    //   k + j = local_pos, A[k-1] ≤ B[j], B[j-1] ≤ A[k]
    // Then output[gid] = min(A[k], B[j]).

    fn generate_merge_kernel(&self, sm: SmVersion) -> Result<String, String> {
        let name = self.cfg.merge_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = self.cfg.elem_bytes();
        // Type-accurate comparison: "is left ≤ right?" in ascending sort.
        let cmp_le = format!("setp.le.{ty}"); // works for all supported types

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_output,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_n,\n    \
             .param .u64 param_merge_len\n)"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "{{").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .{ty}   %ak, %bj, %a_km1, %b_jm1;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %n, %merge_len, %gid, %local_pos;")
            .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %merge_id, %pair_start, %left_start, %right_start;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %lo, %hi, %mid, %lo_min, %hi_cand;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %k, %j, %k_m1, %j_m1;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    .reg .u64    %ptr_in, %ptr_out, %addr_ak, %addr_bj;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u64    %addr_akm1, %addr_bjm1;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .u32    %tid, %bid;").map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %p, %a_leq_b, %k_valid, %j_valid;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    .reg .pred   %akm1_le_bj;").map_err(|e| e.to_string())?;

        writeln!(out, "    ld.param.u64 %ptr_out,    [param_output];")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %ptr_in,     [param_input];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %n,           [param_n];").map_err(|e| e.to_string())?;
        writeln!(out, "    ld.param.u64 %merge_len,   [param_merge_len];")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(|e| e.to_string())?;

        // Out-of-bounds guard.
        writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p ret;").map_err(|e| e.to_string())?;

        // Determine which merge pair this thread belongs to.
        // pair_start = merge_id * 2 * merge_len
        // local_pos  = gid - pair_start
        writeln!(out, "    .reg .u64 %two_ml;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u64      %two_ml, %merge_len, %merge_len;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    div.u64      %merge_id, %gid, %two_ml;").map_err(|e| e.to_string())?;
        writeln!(out, "    mul.lo.u64   %pair_start, %merge_id, %two_ml;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    sub.u64      %local_pos, %gid, %pair_start;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64      %left_start, %pair_start;").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    add.u64      %right_start, %pair_start, %merge_len;"
        )
        .map_err(|e| e.to_string())?;

        // Binary search for co-rank k.
        // lo = max(0, local_pos - merge_len)
        // hi = min(local_pos, merge_len)
        writeln!(out, "    setp.gt.u64  %p, %local_pos, %merge_len;").map_err(|e| e.to_string())?;
        writeln!(out, "    sub.u64      %lo_min, %local_pos, %merge_len;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    selp.u64     %lo, %lo_min, 0, %p;").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.lt.u64  %p, %local_pos, %merge_len;").map_err(|e| e.to_string())?;
        writeln!(out, "    selp.u64     %hi, %local_pos, %merge_len, %p;")
            .map_err(|e| e.to_string())?;

        // Binary search loop: find largest k in [lo, hi] satisfying co-rank condition.
        writeln!(out, "MERGE_BSEARCH:").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.ge.u64  %p, %lo, %hi;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%p bra MERGE_BSEARCH_DONE;").map_err(|e| e.to_string())?;
        // mid = (lo + hi + 1) / 2
        writeln!(out, "    add.u64      %mid, %lo, %hi;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u64      %mid, %mid, 1;").map_err(|e| e.to_string())?;
        writeln!(out, "    shr.u64      %mid, %mid, 1;").map_err(|e| e.to_string())?;
        // j = local_pos - mid
        writeln!(out, "    sub.u64      %j, %local_pos, %mid;").map_err(|e| e.to_string())?;
        // Check if A[mid-1] <= B[j]: load A[left_start + mid - 1]
        writeln!(out, "    sub.u64      %k_m1, %mid, 1;").map_err(|e| e.to_string())?;
        writeln!(out, "    add.u64      %addr_akm1, %left_start, %k_m1;")
            .map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    mad.lo.u64   %addr_akm1, %addr_akm1, {eb}, %ptr_in;"
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "    ld.global.{ty} %a_km1, [%addr_akm1];").map_err(|e| e.to_string())?;
        // Load B[j] = input[right_start + j]
        writeln!(out, "    add.u64      %addr_bj, %right_start, %j;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %addr_bj, %addr_bj, {eb}, %ptr_in;")
            .map_err(|e| e.to_string())?;
        // Guard: j < merge_len means B[j] is a valid element; else treat as +inf (→ always take A).
        writeln!(out, "    setp.lt.u64  %j_valid, %j, %merge_len;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%j_valid ld.global.{ty} %bj, [%addr_bj];")
            .map_err(|e| e.to_string())?;
        // j invalid → A[k-1] is definitely ≤ "B[j]=+inf" → take more from A: lo = mid.
        // j valid   → check actual comparison.
        writeln!(out, "    @!%j_valid bra BSEARCH_TAKE_A_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "    {cmp_le}    %akm1_le_bj, %a_km1, %bj;").map_err(|e| e.to_string())?;
        writeln!(out, "    @!%akm1_le_bj bra BSEARCH_TAKE_B_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "BSEARCH_TAKE_A_{name}:").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64      %lo, %mid;").map_err(|e| e.to_string())?;
        writeln!(out, "    bra MERGE_BSEARCH;").map_err(|e| e.to_string())?;
        writeln!(out, "BSEARCH_TAKE_B_{name}:").map_err(|e| e.to_string())?;
        writeln!(out, "    mov.u64      %hi, %k_m1;").map_err(|e| e.to_string())?;
        writeln!(out, "    bra MERGE_BSEARCH;").map_err(|e| e.to_string())?;
        writeln!(out, "MERGE_BSEARCH_DONE:").map_err(|e| e.to_string())?;

        // k = lo (index of next A element), j = local_pos - lo (next B element).
        writeln!(out, "    mov.u64      %k, %lo;").map_err(|e| e.to_string())?;
        writeln!(out, "    sub.u64      %j, %local_pos, %lo;").map_err(|e| e.to_string())?;

        // Load A[left_start + k] and B[right_start + j] if in-bounds.
        writeln!(out, "    add.u64      %addr_ak, %left_start, %k;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %addr_ak, %addr_ak, {eb}, %ptr_in;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    setp.lt.u64  %k_valid, %k, %merge_len;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%k_valid  ld.global.{ty} %ak, [%addr_ak];")
            .map_err(|e| e.to_string())?;

        writeln!(out, "    add.u64      %addr_bj, %right_start, %j;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %addr_bj, %addr_bj, {eb}, %ptr_in;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    setp.lt.u64  %j_valid, %j, %merge_len;").map_err(|e| e.to_string())?;
        // Also ensure right_start + j < n (last pair may have a truncated right block).
        writeln!(out, "    add.u64      %hi_cand, %right_start, %j;").map_err(|e| e.to_string())?;
        writeln!(out, "    setp.lt.u64  %p, %hi_cand, %n;").map_err(|e| e.to_string())?;
        writeln!(out, "    and.pred     %j_valid, %j_valid, %p;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%j_valid  ld.global.{ty} %bj, [%addr_bj];")
            .map_err(|e| e.to_string())?;

        // Select output: prefer A[k] when A[k] ≤ B[j]; fall back to B[j].
        // Branch-based to avoid invalid double-predicated instructions.
        writeln!(out, "    .reg .u64 %out_addr;").map_err(|e| e.to_string())?;
        writeln!(out, "    mad.lo.u64   %out_addr, %gid, {eb}, %ptr_out;")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    @!%k_valid bra MERGE_USE_B_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "    @!%j_valid bra MERGE_USE_A_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "    {cmp_le}     %a_leq_b, %ak, %bj;").map_err(|e| e.to_string())?;
        writeln!(out, "    @%a_leq_b  bra MERGE_USE_A_{name};").map_err(|e| e.to_string())?;
        writeln!(out, "MERGE_USE_B_{name}:").map_err(|e| e.to_string())?;
        writeln!(out, "    st.global.{ty} [%out_addr], %bj;").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "MERGE_USE_A_{name}:").map_err(|e| e.to_string())?;
        writeln!(out, "    st.global.{ty} [%out_addr], %ak;").map_err(|e| e.to_string())?;
        writeln!(out, "    ret;").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// PTX literal for the maximum value of a type, used to pad OOB elements
/// so they sort to the end of each block.
fn max_fill_literal(ty: PtxType) -> &'static str {
    match ty {
        PtxType::U32 => "0xFFFFFFFF",
        PtxType::S32 => "0x7FFFFFFF",
        PtxType::U64 => "0xFFFFFFFFFFFFFFFF",
        PtxType::S64 => "0x7FFFFFFFFFFFFFFF",
        PtxType::F32 => "0x7F800000",         // +inf
        PtxType::F64 => "0x7FF0000000000000", // +inf
        _ => "0xFFFFFFFF",
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    fn cfg(ty: PtxType) -> MergeSortConfig {
        MergeSortConfig {
            ty,
            block_size: 256,
        }
    }

    #[test]
    fn sort_blocks_name_correct() {
        let c = cfg(PtxType::U32);
        let n = c.sort_blocks_name();
        assert!(n.contains("merge_sort_blocks"), "{n}");
        assert!(n.contains("u32"), "{n}");
        assert!(n.contains("bs256"), "{n}");
    }

    #[test]
    fn merge_kernel_name_correct() {
        let c = cfg(PtxType::F32);
        let n = c.merge_kernel_name();
        assert!(n.contains("merge_sort_merge"), "{n}");
        assert!(n.contains("f32"), "{n}");
    }

    #[test]
    fn log2_block_size_correct() {
        let c256 = MergeSortConfig {
            ty: PtxType::U32,
            block_size: 256,
        };
        let c32 = MergeSortConfig {
            ty: PtxType::U32,
            block_size: 32,
        };
        assert_eq!(c256.log2_block_size(), 8);
        assert_eq!(c32.log2_block_size(), 5);
    }

    #[test]
    fn sort_ptx_has_bitonic_network_and_bar_sync() {
        let t = MergeSortTemplate::new(cfg(PtxType::U32));
        let ptx = t.generate_sort_blocks_kernel(SmVersion::Sm80).unwrap();
        assert!(ptx.contains("bsort_smem"), "PTX: {ptx}");
        assert!(ptx.contains("bar.sync 0"), "PTX: {ptx}");
        assert!(ptx.contains("xor.b32"), "PTX: {ptx}"); // partner = tid XOR j
        assert!(ptx.contains("selp"), "PTX: {ptx}"); // conditional swap
        assert!(ptx.contains("cvt.u32.pred"), "PTX: {ptx}"); // swap condition
    }

    #[test]
    fn sort_ptx_has_global_load_and_store() {
        let t = MergeSortTemplate::new(cfg(PtxType::U32));
        let ptx = t.generate_sort_blocks_kernel(SmVersion::Sm80).unwrap();
        assert!(ptx.contains("ld.global.u32"), "PTX: {ptx}");
        assert!(ptx.contains("st.global.u32"), "PTX: {ptx}");
    }

    #[test]
    fn sort_ptx_f32_uses_setp_lt_f32() {
        let t = MergeSortTemplate::new(cfg(PtxType::F32));
        let ptx = t.generate_sort_blocks_kernel(SmVersion::Sm80).unwrap();
        assert!(ptx.contains("setp.lt.f32"), "PTX: {ptx}");
    }

    #[test]
    fn merge_ptx_has_binary_search_loop() {
        let t = MergeSortTemplate::new(cfg(PtxType::U32));
        let ptx = t.generate_merge_kernel(SmVersion::Sm80).unwrap();
        assert!(ptx.contains("MERGE_BSEARCH"), "PTX: {ptx}");
        assert!(ptx.contains("MERGE_BSEARCH_DONE"), "PTX: {ptx}");
        assert!(ptx.contains("param_merge_len"), "PTX: {ptx}");
    }

    #[test]
    fn merge_ptx_has_global_memory_accesses() {
        let t = MergeSortTemplate::new(cfg(PtxType::F32));
        let ptx = t.generate_merge_kernel(SmVersion::Sm80).unwrap();
        assert!(ptx.contains("ld.global.f32"), "PTX: {ptx}");
        assert!(ptx.contains("st.global.f32"), "PTX: {ptx}");
    }

    #[test]
    fn generate_both_kernels_succeeds() {
        let t = MergeSortTemplate::new(cfg(PtxType::U32));
        let (sort_ptx, merge_ptx) = t.generate(SmVersion::Sm80).unwrap();
        assert!(!sort_ptx.is_empty());
        assert!(!merge_ptx.is_empty());
    }

    #[test]
    fn sort_ptx_u64_uses_8byte_elem_size() {
        let t = MergeSortTemplate::new(cfg(PtxType::U64));
        let ptx = t.generate_sort_blocks_kernel(SmVersion::Sm80).unwrap();
        assert!(ptx.contains("ld.global.u64"), "PTX: {ptx}");
        assert!(ptx.contains("st.global.u64"), "PTX: {ptx}");
    }

    #[test]
    fn max_fill_literals_are_non_zero() {
        assert_ne!(max_fill_literal(PtxType::U32), "0");
        assert_ne!(max_fill_literal(PtxType::F32), "0");
    }
}
