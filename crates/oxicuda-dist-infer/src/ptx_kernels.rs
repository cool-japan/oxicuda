//! PTX kernel source strings for distributed inference primitives.
//!
//! All kernels are pure PTX strings designed to be JIT-compiled at runtime via
//! `cuModuleLoadData` (the OxiCUDA driver).  No nvcc / libcuda link-time
//! dependency.
//!
//! # Kernels
//!
//! | Function | Purpose |
//! |----------|---------|
//! | `tp_col_scatter_ptx` | Column-parallel linear scatter: write strided shard into output buffer |
//! | `tp_row_all_reduce_ptx` | Row-parallel linear all-reduce: ring-based partial-sum reduction |
//! | `sp_seq_chunk_copy_ptx` | Sequence-parallel chunk copy: extract / insert a contiguous token slice |
//! | `ep_token_scatter_ptx` | Expert-parallel token scatter: route tokens to expert-local buffers |
//! | `ep_token_gather_ptx` | Expert-parallel token gather: collect results back to original order |

use crate::handle::SmVersion;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn ptx_header(sm: SmVersion) -> String {
    format!(
        ".version {ver}\n.target {tgt}\n.address_size 64\n",
        ver = sm.ptx_version_str(),
        tgt = sm.target_str(),
    )
}

// ─── tp_col_scatter_ptx ──────────────────────────────────────────────────────

/// Generate PTX for the column-parallel scatter pass.
///
/// Each thread writes one `f32` element from the GEMM partial output (size
/// `batch × local_cols`) into the full output buffer (size `batch × total_cols`)
/// at the column offset `col_offset = tp_rank * local_cols`.
///
/// Grid: `ceil(batch * local_cols / BLOCK_SIZE)` blocks of `BLOCK_SIZE` threads.
///
/// Parameters (`.param` table):
/// * `p_src`   — pointer to local shard output (f32[batch × local_cols])
/// * `p_dst`   — pointer to full output buffer  (f32[batch × total_cols])
/// * `n`       — total elements in shard (`batch * local_cols`)
/// * `total_cols` — number of columns in full output
/// * `local_cols` — number of columns in this shard
/// * `col_offset` — starting column index for this shard (tp_rank * local_cols)
pub fn tp_col_scatter_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
.visible .entry tp_col_scatter(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u32 n,
    .param .u32 total_cols,
    .param .u32 local_cols,
    .param .u32 col_offset
) {{
    .reg .u64  %src, %dst, %ptr_s, %ptr_d;
    .reg .u32  %n, %tcols, %lcols, %off;
    .reg .u32  %tid, %ntid, %ctaid, %gid;
    .reg .u32  %row, %lcol, %gcol;
    .reg .u64  %src_idx, %dst_idx;
    .reg .f32  %val;
    .reg .pred %p;

    ld.param.u64  %src,   [p_src];
    ld.param.u64  %dst,   [p_dst];
    ld.param.u32  %n,     [n];
    ld.param.u32  %tcols, [total_cols];
    ld.param.u32  %lcols, [local_cols];
    ld.param.u32  %off,   [col_offset];

    mov.u32 %tid,   %tid.x;
    mov.u32 %ntid,  %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %gid, %ctaid, %ntid, %tid;

$LOOP:
    setp.ge.u32 %p, %gid, %n;
    @%p bra $DONE;

    // row = gid / local_cols;  local_col = gid % local_cols
    div.u32 %row,  %gid, %lcols;
    rem.u32 %lcol, %gid, %lcols;
    add.u32 %gcol, %lcol, %off;          // global column index

    // src_idx = gid;  dst_idx = row * total_cols + gcol
    cvt.u64.u32 %src_idx, %gid;
    mul.wide.u32 %dst_idx, %row, %tcols;
    cvt.u64.u32  %ptr_s, %gcol;
    add.u64 %dst_idx, %dst_idx, %ptr_s;

    // load from shard, store to full output
    mul.lo.u64  %ptr_s, %src_idx, 4;
    add.u64     %ptr_s, %src, %ptr_s;
    ld.global.f32 %val, [%ptr_s];

    mul.lo.u64  %ptr_d, %dst_idx, 4;
    add.u64     %ptr_d, %dst, %ptr_d;
    st.global.f32 [%ptr_d], %val;

    add.u32 %gid, %gid, %ntid;
    mul.lo.u32 %ntid, %ntid, 1;
    bra $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr
    )
}

// ─── tp_row_all_reduce_ptx ───────────────────────────────────────────────────

/// Ring all-reduce kernel (single-pass summation over a shared flat buffer).
///
/// This kernel implements the **accumulate** step of a simulated ring all-reduce
/// on shared host memory that has been aggregated into `p_buf`.  Each thread
/// reads its element, adds the running partial sum (provided by the caller as
/// `p_partials[rank]`), and writes back the accumulated value.
///
/// In a real multi-GPU scenario the driver layer orchestrates P2P copies
/// between GPUs before this kernel is invoked on each device.
///
/// Parameters:
/// * `p_buf`     — local partial-sum buffer (f32\[n\])
/// * `p_accum`   — already-reduced global buffer written by rank 0 (f32\[n\])
/// * `n`         — element count
pub fn tp_row_all_reduce_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
.visible .entry tp_row_all_reduce(
    .param .u64 p_buf,
    .param .u64 p_accum,
    .param .u32 n
) {{
    .reg .u64  %buf, %acc, %ptr_b, %ptr_a;
    .reg .u32  %n, %tid, %ntid, %ctaid, %gid;
    .reg .f32  %v_buf, %v_acc, %v_sum;
    .reg .pred %p;

    ld.param.u64 %buf, [p_buf];
    ld.param.u64 %acc, [p_accum];
    ld.param.u32 %n,   [n];

    mov.u32 %tid,   %tid.x;
    mov.u32 %ntid,  %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %gid, %ctaid, %ntid, %tid;

$LOOP:
    setp.ge.u32 %p, %gid, %n;
    @%p bra $DONE;

    cvt.u64.u32 %ptr_b, %gid;
    mul.lo.u64  %ptr_b, %ptr_b, 4;
    add.u64     %ptr_b, %buf, %ptr_b;
    ld.global.f32 %v_buf, [%ptr_b];

    cvt.u64.u32 %ptr_a, %gid;
    mul.lo.u64  %ptr_a, %ptr_a, 4;
    add.u64     %ptr_a, %acc, %ptr_a;
    ld.global.f32 %v_acc, [%ptr_a];

    add.f32 %v_sum, %v_buf, %v_acc;
    st.global.f32 [%ptr_b], %v_sum;

    add.u32 %gid, %gid, %ntid;
    bra $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr
    )
}

// ─── sp_seq_chunk_copy_ptx ───────────────────────────────────────────────────

/// Extract (or insert) a contiguous token chunk for sequence parallelism.
///
/// When `direction = 0` (extract): copies tokens `[chunk_start, chunk_start+chunk_len)`
/// from the full sequence buffer into a local chunk buffer.
/// When `direction = 1` (insert): copies the local chunk back into the full buffer.
///
/// Tokens are represented as flat `f32` vectors of length `hidden_dim`.
///
/// Parameters:
/// * `p_full`      — pointer to full sequence buffer (f32[total_tokens × hidden_dim])
/// * `p_chunk`     — pointer to chunk buffer          (f32[chunk_len × hidden_dim])
/// * `chunk_start` — first token index in full buffer
/// * `chunk_len`   — number of tokens in this rank's chunk
/// * `hidden_dim`  — embedding dimension
/// * `direction`   — 0=extract (full→chunk), 1=insert (chunk→full)
pub fn sp_seq_chunk_copy_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
.visible .entry sp_seq_chunk_copy(
    .param .u64 p_full,
    .param .u64 p_chunk,
    .param .u32 chunk_start,
    .param .u32 chunk_len,
    .param .u32 hidden_dim,
    .param .u32 direction
) {{
    .reg .u64  %full, %chunk, %ptr_f, %ptr_c;
    .reg .u32  %cs, %cl, %hd, %dir;
    .reg .u32  %tid, %ntid, %ctaid, %gid, %n;
    .reg .u32  %tok, %feat, %full_tok;
    .reg .u64  %idx_f, %idx_c;
    .reg .f32  %val;
    .reg .pred %p;

    ld.param.u64 %full,  [p_full];
    ld.param.u64 %chunk, [p_chunk];
    ld.param.u32 %cs,    [chunk_start];
    ld.param.u32 %cl,    [chunk_len];
    ld.param.u32 %hd,    [hidden_dim];
    ld.param.u32 %dir,   [direction];

    mov.u32 %tid,   %tid.x;
    mov.u32 %ntid,  %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %gid, %ctaid, %ntid, %tid;
    mul.lo.u32 %n, %cl, %hd;

$LOOP:
    setp.ge.u32 %p, %gid, %n;
    @%p bra $DONE;

    div.u32 %tok,  %gid, %hd;    // local token index
    rem.u32 %feat, %gid, %hd;    // feature index
    add.u32 %full_tok, %tok, %cs; // global token index

    // full buffer idx = full_tok * hidden_dim + feat
    mul.wide.u32 %idx_f, %full_tok, %hd;
    cvt.u64.u32  %ptr_f, %feat;
    add.u64      %idx_f, %idx_f, %ptr_f;
    mul.lo.u64   %idx_f, %idx_f, 4;
    add.u64      %ptr_f, %full, %idx_f;

    // chunk buffer idx = gid
    cvt.u64.u32  %idx_c, %gid;
    mul.lo.u64   %idx_c, %idx_c, 4;
    add.u64      %ptr_c, %chunk, %idx_c;

    setp.ne.u32 %p, %dir, 0;
    @%p bra $INSERT;

    // extract: full → chunk
    ld.global.f32  %val, [%ptr_f];
    st.global.f32  [%ptr_c], %val;
    bra $NEXT;

$INSERT:
    // insert: chunk → full
    ld.global.f32  %val, [%ptr_c];
    st.global.f32  [%ptr_f], %val;

$NEXT:
    add.u32 %gid, %gid, %ntid;
    bra $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr
    )
}

// ─── ep_token_scatter_ptx ────────────────────────────────────────────────────

/// Scatter tokens to expert-local input buffers.
///
/// Each token `t` is assigned to expert `expert_ids[t]`.  This kernel writes
/// token `t`'s embedding (size `hidden_dim`) into the slot reserved for it in
/// `p_expert_bufs[expert_ids[t]]`.  The `expert_offsets[e]` array gives the
/// cumulative count of tokens dispatched to experts 0..e-1 so the write address
/// can be computed without atomics.
///
/// Parameters:
/// * `p_input`        — flat token embeddings (f32[n_tokens × hidden_dim])
/// * `p_expert_buf`   — flat expert input buffer (f32[n_tokens × hidden_dim])
/// * `p_expert_ids`   — per-token expert assignment (u32\[n_tokens\])
/// * `p_expert_slots` — per-token slot within the assigned expert (u32\[n_tokens\])
/// * `n_tokens`       — total number of tokens
/// * `hidden_dim`     — embedding dimension
pub fn ep_token_scatter_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
.visible .entry ep_token_scatter(
    .param .u64 p_input,
    .param .u64 p_expert_buf,
    .param .u64 p_expert_ids,
    .param .u64 p_expert_slots,
    .param .u32 n_tokens,
    .param .u32 hidden_dim
) {{
    .reg .u64  %inp, %ebuf, %eids, %eslots;
    .reg .u32  %nt, %hd;
    .reg .u32  %tid, %ntid, %ctaid, %gid, %n;
    .reg .u32  %tok, %feat;
    .reg .u32  %eid, %slot;
    .reg .u64  %src_off, %dst_off, %ptr_s, %ptr_d;
    .reg .u64  %ptr_eid, %ptr_slot;
    .reg .f32  %val;
    .reg .pred %p;

    ld.param.u64 %inp,    [p_input];
    ld.param.u64 %ebuf,   [p_expert_buf];
    ld.param.u64 %eids,   [p_expert_ids];
    ld.param.u64 %eslots, [p_expert_slots];
    ld.param.u32 %nt,     [n_tokens];
    ld.param.u32 %hd,     [hidden_dim];

    mov.u32 %tid,   %tid.x;
    mov.u32 %ntid,  %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %gid, %ctaid, %ntid, %tid;
    mul.lo.u32 %n, %nt, %hd;

$LOOP:
    setp.ge.u32 %p, %gid, %n;
    @%p bra $DONE;

    div.u32 %tok,  %gid, %hd;
    rem.u32 %feat, %gid, %hd;

    // Load expert id and slot for this token
    cvt.u64.u32 %ptr_eid, %tok;
    mul.lo.u64  %ptr_eid, %ptr_eid, 4;
    add.u64     %ptr_eid, %eids, %ptr_eid;
    ld.global.u32 %eid, [%ptr_eid];

    cvt.u64.u32 %ptr_slot, %tok;
    mul.lo.u64  %ptr_slot, %ptr_slot, 4;
    add.u64     %ptr_slot, %eslots, %ptr_slot;
    ld.global.u32 %slot, [%ptr_slot];

    // src: input[tok, feat]
    mul.wide.u32 %src_off, %tok, %hd;
    cvt.u64.u32  %ptr_s, %feat;
    add.u64      %src_off, %src_off, %ptr_s;
    mul.lo.u64   %src_off, %src_off, 4;
    add.u64      %ptr_s, %inp, %src_off;
    ld.global.f32 %val, [%ptr_s];

    // dst: expert_buf[slot, feat] (slots are pre-computed globally)
    mul.wide.u32 %dst_off, %slot, %hd;
    cvt.u64.u32  %ptr_d, %feat;
    add.u64      %dst_off, %dst_off, %ptr_d;
    mul.lo.u64   %dst_off, %dst_off, 4;
    add.u64      %ptr_d, %ebuf, %dst_off;
    st.global.f32 [%ptr_d], %val;

    add.u32 %gid, %gid, %ntid;
    bra $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr
    )
}

// ─── ep_token_gather_ptx ────────────────────────────────────────────────────

/// Gather expert outputs back to the original token order.
///
/// The inverse of `ep_token_scatter`: for each token `t`, read its result from
/// `expert_buf[expert_slots[t], :]` and write it to `output[t, :]`.
///
/// Parameters: same layout as `ep_token_scatter`.
pub fn ep_token_gather_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
.visible .entry ep_token_gather(
    .param .u64 p_expert_buf,
    .param .u64 p_output,
    .param .u64 p_expert_slots,
    .param .u32 n_tokens,
    .param .u32 hidden_dim
) {{
    .reg .u64  %ebuf, %out, %eslots;
    .reg .u32  %nt, %hd;
    .reg .u32  %tid, %ntid, %ctaid, %gid, %n;
    .reg .u32  %tok, %feat, %slot;
    .reg .u64  %src_off, %dst_off, %ptr_s, %ptr_d, %ptr_slot;
    .reg .f32  %val;
    .reg .pred %p;

    ld.param.u64 %ebuf,   [p_expert_buf];
    ld.param.u64 %out,    [p_output];
    ld.param.u64 %eslots, [p_expert_slots];
    ld.param.u32 %nt,     [n_tokens];
    ld.param.u32 %hd,     [hidden_dim];

    mov.u32 %tid,   %tid.x;
    mov.u32 %ntid,  %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %gid, %ctaid, %ntid, %tid;
    mul.lo.u32 %n, %nt, %hd;

$LOOP:
    setp.ge.u32 %p, %gid, %n;
    @%p bra $DONE;

    div.u32 %tok,  %gid, %hd;
    rem.u32 %feat, %gid, %hd;

    // Load slot for this token
    cvt.u64.u32 %ptr_slot, %tok;
    mul.lo.u64  %ptr_slot, %ptr_slot, 4;
    add.u64     %ptr_slot, %eslots, %ptr_slot;
    ld.global.u32 %slot, [%ptr_slot];

    // src: expert_buf[slot, feat]
    mul.wide.u32 %src_off, %slot, %hd;
    cvt.u64.u32  %ptr_s, %feat;
    add.u64      %src_off, %src_off, %ptr_s;
    mul.lo.u64   %src_off, %src_off, 4;
    add.u64      %ptr_s, %ebuf, %src_off;
    ld.global.f32 %val, [%ptr_s];

    // dst: output[tok, feat]
    mul.wide.u32 %dst_off, %tok, %hd;
    cvt.u64.u32  %ptr_d, %feat;
    add.u64      %dst_off, %dst_off, %ptr_d;
    mul.lo.u64   %dst_off, %dst_off, 4;
    add.u64      %ptr_d, %out, %dst_off;
    st.global.f32 [%ptr_d], %val;

    add.u32 %gid, %gid, %ntid;
    bra $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr
    )
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sm80() -> SmVersion {
        SmVersion(80)
    }

    #[test]
    fn col_scatter_ptx_contains_key_ops() {
        let ptx = tp_col_scatter_ptx(sm80());
        assert!(ptx.contains("tp_col_scatter"), "missing kernel name");
        assert!(ptx.contains("col_offset"), "missing col_offset param");
        assert!(ptx.contains("rem.u32"), "missing modulo for column index");
        assert!(ptx.contains(".version 8.0"), "wrong ptx version");
        assert!(ptx.contains(".target sm_80"), "wrong target");
    }

    #[test]
    fn row_all_reduce_ptx_contains_key_ops() {
        let ptx = tp_row_all_reduce_ptx(sm80());
        assert!(ptx.contains("tp_row_all_reduce"));
        assert!(ptx.contains("add.f32"), "must accumulate with add.f32");
    }

    #[test]
    fn seq_chunk_copy_ptx_contains_direction() {
        let ptx = sp_seq_chunk_copy_ptx(sm80());
        assert!(ptx.contains("direction"), "missing direction param");
        assert!(ptx.contains("$INSERT"), "missing INSERT branch");
        assert!(ptx.contains("chunk_start"), "missing chunk_start param");
    }

    #[test]
    fn token_scatter_ptx_contains_key_fields() {
        let ptx = ep_token_scatter_ptx(sm80());
        assert!(ptx.contains("ep_token_scatter"));
        assert!(ptx.contains("p_expert_ids"), "missing expert ids param");
        assert!(ptx.contains("p_expert_slots"), "missing expert slots param");
    }

    #[test]
    fn token_gather_ptx_is_inverse() {
        let scatter = ep_token_scatter_ptx(sm80());
        let gather = ep_token_gather_ptx(sm80());
        // Both reference the expert buffer and slot arrays
        assert!(gather.contains("ep_token_gather"));
        assert!(gather.contains("p_expert_slots"));
        // Gather writes to p_output, scatter reads from p_input — check naming
        assert!(gather.contains("p_output"));
        assert!(scatter.contains("p_input"));
    }

    #[test]
    fn all_kernels_sm90_produce_correct_header() {
        let sm = SmVersion(90);
        for ptx in [
            tp_col_scatter_ptx(sm),
            tp_row_all_reduce_ptx(sm),
            sp_seq_chunk_copy_ptx(sm),
            ep_token_scatter_ptx(sm),
            ep_token_gather_ptx(sm),
        ] {
            assert!(ptx.contains(".version 8.4"), "sm90 should use ptx 8.4");
            assert!(ptx.contains(".target sm_90"), "sm90 target mismatch");
        }
    }
}
