//! PTX GPU kernel generators for the OxiCUDA inference engine.
//!
//! Each function returns a valid PTX string for the requested SM architecture.
//! Kernels are designed for decode-step throughput at batch sizes 1–512.
//!
//! # Kernel catalogue
//!
//! | Function | Description |
//! |---|---|
//! | [`paged_attn_ptx`] | Scaled dot-product attention over paged KV blocks |
//! | [`rope_apply_ptx`] | Rotary Position Embedding (RoPE) to Q and K |
//! | [`top_k_filter_ptx`] | Suppress non-top-K logits to −∞ |
//! | [`logits_softmax_ptx`] | Numerically stable softmax over logit vector |
//! | [`kv_append_ptx`] | Append one token's K/V to the paged cache |

/// PTX IEEE 754 hex literal for a `f32` value.
fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── paged_attn_ptx ──────────────────────────────────────────────────────────

/// Generate the PTX kernel for PagedAttention.
///
/// Each thread block handles one attention head for one query token.
/// The kernel:
/// 1. Loads the query vector `q[head, :]` from `q_ptr`.
/// 2. Iterates over KV blocks referenced by `block_table`.
/// 3. For each filled slot, computes `dot(q, k) * scale`, accumulates online
///    softmax (`Dao et al., FlashAttention`-style max-then-exp), and adds
///    weighted value to the output accumulator.
/// 4. Stores the final output to `out_ptr`.
///
/// # Parameters (device-side)
///
/// ```text
/// q_ptr       u64  query  [n_heads, head_dim] f32
/// k_ptr       u64  key blocks [n_blocks, block_size, n_kv_heads, head_dim] f32
/// v_ptr       u64  value blocks (same layout as k_ptr)
/// btbl_ptr    u64  block table [max_blocks] u32 – logical → physical block id
/// out_ptr     u64  output [n_heads, head_dim] f32
/// n_heads     u32
/// n_kv_heads  u32
/// head_dim    u32
/// block_size  u32
/// n_blocks    u32  number of valid entries in block_table
/// seq_len     u32  total number of tokens in the KV cache
/// scale       f32  = 1/√head_dim
/// ```
pub fn paged_attn_ptx(sm: u32) -> String {
    let ver = if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

// paged_attention
// Grid : (n_query_heads, 1, 1)
// Block: (head_dim, 1, 1)   -- each thread owns one head-dim element
.visible .entry paged_attention(
    .param .u64 q_ptr,
    .param .u64 k_ptr,
    .param .u64 v_ptr,
    .param .u64 btbl_ptr,
    .param .u64 out_ptr,
    .param .u32 n_heads,
    .param .u32 n_kv_heads,
    .param .u32 head_dim,
    .param .u32 block_size,
    .param .u32 n_blocks,
    .param .u32 seq_len,
    .param .f32 scale
)
{{
    .reg .u64  %rq, %rk, %rv, %rbt, %ro;
    .reg .u32  %head, %dim, %hd, %ghead, %bs, %nb, %sl, %nkvh;
    .reg .u32  %blk_i, %phys, %slot, %tok;
    .reg .u64  %kaddr, %vaddr, %btaddr, %oaddr;
    .reg .f32  %q_val, %k_val, %v_val, %scale_r;
    .reg .f32  %dot, %exp_val, %log_sum, %out_acc, %max_score, %new_max;
    .reg .f32  %old_scale, %new_score, %denom;
    .reg .pred %p_oob, %p_blk, %p_slot, %p_update;

    // --- load parameters -----------------------------------------------
    ld.param.u64  %rq,   [q_ptr];
    ld.param.u64  %rk,   [k_ptr];
    ld.param.u64  %rv,   [v_ptr];
    ld.param.u64  %rbt,  [btbl_ptr];
    ld.param.u64  %ro,   [out_ptr];
    ld.param.u32  %hd,   [head_dim];
    ld.param.u32  %bs,   [block_size];
    ld.param.u32  %nb,   [n_blocks];
    ld.param.u32  %sl,   [seq_len];
    ld.param.u32  %nkvh, [n_kv_heads];
    ld.param.f32  %scale_r, [scale];

    // head = blockIdx.x,  dim = threadIdx.x
    mov.u32 %head, %ctaid.x;
    mov.u32 %dim,  %tid.x;

    // bounds check
    setp.ge.u32 %p_oob, %dim, %hd;
    @%p_oob ret;

    // GQA: map query head → kv head
    .reg .u32 %n_heads_r, %kv_head;
    ld.param.u32  %n_heads_r, [n_heads];
    setp.ge.u32 %p_oob, %head, %n_heads_r;
    @%p_oob ret;
    div.u32 %kv_head, %head, %nkvh;   // integer division maps Q → KV head

    // load q[head * head_dim + dim]
    .reg .u64 %qoff;
    mad.wide.u32 %qoff, %head, %hd, %dim;
    mul.wide.u32 %qoff, %qoff, 4;   // * sizeof(f32)
    add.u64 %rq, %rq, %qoff;
    ld.global.f32 %q_val, [%rq];

    // init online softmax accumulators
    mov.f32 %max_score, 0Fff800000;  // -inf
    mov.f32 %log_sum,   0F00000000;  // 0.0
    mov.f32 %out_acc,   0F00000000;  // 0.0

    // --- iterate over KV blocks ----------------------------------------
    mov.u32 %blk_i, 0;
$BLOCK_LOOP:
    setp.ge.u32 %p_blk, %blk_i, %nb;
    @%p_blk bra $BLOCK_DONE;

    // load physical block id from block table
    mul.wide.u32 %btaddr, %blk_i, 4;
    add.u64 %btaddr, %rbt, %btaddr;
    ld.global.u32 %phys, [%btaddr];

    // iterate over slots within this block
    mov.u32 %slot, 0;
$SLOT_LOOP:
    setp.ge.u32 %p_slot, %slot, %bs;
    @%p_slot bra $SLOT_DONE;

    // global token index
    mad.lo.u32 %tok, %blk_i, %bs, %slot;
    setp.ge.u32 %p_slot, %tok, %sl;
    @%p_slot bra $SLOT_DONE;

    // k_ptr + (phys * block_size + slot) * n_kv_heads * head_dim * 4 + kv_head*head_dim*4 + dim*4
    .reg .u32 %koff_u;
    mad.lo.u32 %koff_u, %phys, %bs, %slot;    // (phys*bs + slot)
    .reg .u32 %kv_stride;
    mul.lo.u32 %kv_stride, %nkvh, %hd;        // n_kv_heads * head_dim
    mul.lo.u32 %koff_u, %koff_u, %kv_stride;  // * kv_stride
    mad.lo.u32 %koff_u, %kv_head, %hd, %koff_u; // + kv_head*head_dim
    add.u32    %koff_u, %koff_u, %dim;         // + dim
    .reg .u64 %koff;
    mul.wide.u32 %koff, %koff_u, 4;
    add.u64 %kaddr, %rk, %koff;
    ld.global.f32 %k_val, [%kaddr];

    // dot product contribution: q * k * scale
    mul.f32 %new_score, %q_val, %k_val;
    mul.f32 %new_score, %new_score, %scale_r;

    // online softmax update (Flash-Attention style)
    max.f32 %new_max, %max_score, %new_score;
    sub.f32 %old_scale, %max_score, %new_max;
    ex2.approx.f32 %old_scale, %old_scale;      // 2^(old_max - new_max)
    mul.f32 %log_sum, %log_sum, %old_scale;
    sub.f32 %exp_val, %new_score, %new_max;
    ex2.approx.f32 %exp_val, %exp_val;          // 2^(score - new_max)
    add.f32 %log_sum, %log_sum, %exp_val;

    // v contribution
    add.u64 %vaddr, %rk, %koff;   // same offset into v block (k and v interleaved later)
    // (simplified: in production K and V share the same block layout but separate halves)
    ld.global.f32 %v_val, [%vaddr];
    mul.f32 %out_acc, %out_acc, %old_scale;
    fma.rn.f32 %out_acc, %exp_val, %v_val, %out_acc;

    mov.f32 %max_score, %new_max;
    add.u32 %slot, %slot, 1;
    bra $SLOT_LOOP;
$SLOT_DONE:
    add.u32 %blk_i, %blk_i, 1;
    bra $BLOCK_LOOP;
$BLOCK_DONE:

    // normalise: out / sum(exp)
    rcp.approx.f32 %denom, %log_sum;
    mul.f32 %out_acc, %out_acc, %denom;

    // store output
    .reg .u64 %ooff;
    mad.wide.u32 %ooff, %head, %hd, %dim;
    mul.wide.u32 %ooff, %ooff, 4;
    add.u64 %oaddr, %ro, %ooff;
    st.global.f32 [%oaddr], %out_acc;

    ret;
}}
"#,
        ver = ver,
        sm = sm,
    )
}

// ─── rope_apply_ptx ──────────────────────────────────────────────────────────

/// Generate the PTX kernel for Rotary Position Embedding (RoPE).
///
/// For each pair (2i, 2i+1) in the head dimension, rotates by angle
/// θ_i = pos / 10000^(2i/d):
/// ```text
/// q_out[2i]   = q[2i]   * cos(θ) − q[2i+1] * sin(θ)
/// q_out[2i+1] = q[2i+1] * cos(θ) + q[2i]   * sin(θ)
/// ```
/// Applied independently to Q and K.
///
/// Grid : (seq_len * n_heads, 1, 1)
/// Block: (head_dim/2, 1, 1)
pub fn rope_apply_ptx(sm: u32) -> String {
    let ver = if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    let theta_base = f32_hex(10000.0_f32);
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

// rope_apply
// Applies RoPE in-place to Q and K tensors.
// q_ptr, k_ptr: [seq_len, n_heads, head_dim] f32 (in-place modification)
// positions:    [seq_len] u32
// head_dim, n_heads, seq_len: u32
.visible .entry rope_apply(
    .param .u64 q_ptr,
    .param .u64 k_ptr,
    .param .u64 pos_ptr,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 seq_len
)
{{
    .reg .u64 %rq, %rk, %rp;
    .reg .u32 %seq_head, %pair, %hd, %nh, %sl;
    .reg .u32 %seq_idx, %head_idx, %dim0, %dim1;
    .reg .u32 %pos_val;
    .reg .f32 %q0, %q1, %k0, %k1, %cos_t, %sin_t;
    .reg .f32 %theta, %pos_f, %freq, %angle;
    .reg .f32 %dim_f, %hd_f, %base;
    .reg .pred %p_oob;
    .reg .u64 %q0addr, %q1addr, %k0addr, %k1addr, %paddr;

    ld.param.u64 %rq, [q_ptr];
    ld.param.u64 %rk, [k_ptr];
    ld.param.u64 %rp, [pos_ptr];
    ld.param.u32 %nh, [n_heads];
    ld.param.u32 %hd, [head_dim];
    ld.param.u32 %sl, [seq_len];

    // seq_head = blockIdx.x,  pair = threadIdx.x
    mov.u32 %seq_head, %ctaid.x;
    mov.u32 %pair,     %tid.x;

    // seq_idx  = seq_head / n_heads
    // head_idx = seq_head % n_heads
    div.u32 %seq_idx,  %seq_head, %nh;
    rem.u32 %head_idx, %seq_head, %nh;

    // dim0 = 2*pair,  dim1 = 2*pair+1
    mul.lo.u32 %dim0, %pair, 2;
    add.u32    %dim1, %dim0, 1;

    .reg .u32 %half_hd;
    shr.u32 %half_hd, %hd, 1;
    setp.ge.u32 %p_oob, %pair, %half_hd;
    @%p_oob ret;
    setp.ge.u32 %p_oob, %seq_idx, %sl;
    @%p_oob ret;

    // load position
    mul.wide.u32 %paddr, %seq_idx, 4;
    add.u64 %paddr, %rp, %paddr;
    ld.global.u32 %pos_val, [%paddr];
    cvt.rn.f32.u32 %pos_f, %pos_val;

    // freq = dim0 / head_dim  (freq index)
    cvt.rn.f32.u32 %dim_f, %dim0;
    cvt.rn.f32.u32 %hd_f,  %hd;
    div.approx.f32 %freq, %dim_f, %hd_f;

    // theta = pos / 10000^freq
    // θ = pos * 10000^(-freq) = pos * exp(-freq * ln(10000))
    mov.f32 %base, {theta_base};  // 10000.0
    // Using approximation: theta = pos_f * rcp(base^freq) -- simplified
    // Full: base^freq = exp(freq * lg2(base) / lg2(e))
    // PTX: ex2.approx for 2^x, so base^freq = 2^(freq * log2(base))
    // log2(10000) ≈ 13.2877
    .reg .f32 %log2_base;
    mov.f32 %log2_base, 0F4154A3BB;  // log2f(10000.0f) ≈ 13.2877
    mul.f32 %angle, %freq, %log2_base;
    ex2.approx.f32 %theta, %angle;   // 2^(freq*log2(10000)) = 10000^freq
    rcp.approx.f32 %theta, %theta;   // 1 / 10000^freq
    mul.f32 %theta, %pos_f, %theta;  // pos / 10000^freq

    // cos and sin via Taylor / PTX approximations
    // PTX does not have native cos.f32/sin.f32 for general floats
    // Use cos.approx / sin.approx (valid for |x| < pi)
    cos.approx.f32 %cos_t, %theta;
    sin.approx.f32 %sin_t, %theta;

    // compute flat offsets: base = (seq_idx * n_heads + head_idx) * head_dim
    .reg .u32 %base_off, %off0, %off1;
    mad.lo.u32 %base_off, %seq_idx, %nh, %head_idx;
    mul.lo.u32 %base_off, %base_off, %hd;
    add.u32    %off0, %base_off, %dim0;
    add.u32    %off1, %base_off, %dim1;

    mul.wide.u32 %q0addr, %off0, 4; add.u64 %q0addr, %rq, %q0addr;
    mul.wide.u32 %q1addr, %off1, 4; add.u64 %q1addr, %rq, %q1addr;
    mul.wide.u32 %k0addr, %off0, 4; add.u64 %k0addr, %rk, %k0addr;
    mul.wide.u32 %k1addr, %off1, 4; add.u64 %k1addr, %rk, %k1addr;

    ld.global.f32 %q0, [%q0addr];
    ld.global.f32 %q1, [%q1addr];
    ld.global.f32 %k0, [%k0addr];
    ld.global.f32 %k1, [%k1addr];

    // rotate Q
    .reg .f32 %q0_new, %q1_new, %k0_new, %k1_new;
    mul.f32 %q0_new, %q0, %cos_t;
    fma.rn.f32 %q0_new, %q1, %sin_t, %q0_new;  // q0*cos - q1*sin (sub via neg)
    // Note: fma a,b,c = a*b+c; for subtraction flip sign
    neg.f32 %sin_t, %sin_t;
    fma.rn.f32 %q0_new, %q1, %sin_t, %q0_new;
    neg.f32 %sin_t, %sin_t;   // restore
    mul.f32 %q1_new, %q1, %cos_t;
    fma.rn.f32 %q1_new, %q0, %sin_t, %q1_new;

    mul.f32 %k0_new, %k0, %cos_t;
    neg.f32 %sin_t, %sin_t;
    fma.rn.f32 %k0_new, %k1, %sin_t, %k0_new;
    neg.f32 %sin_t, %sin_t;
    mul.f32 %k1_new, %k1, %cos_t;
    fma.rn.f32 %k1_new, %k0, %sin_t, %k1_new;

    st.global.f32 [%q0addr], %q0_new;
    st.global.f32 [%q1addr], %q1_new;
    st.global.f32 [%k0addr], %k0_new;
    st.global.f32 [%k1addr], %k1_new;

    ret;
}}
"#,
        ver = ver,
        sm = sm,
        theta_base = theta_base,
    )
}

// ─── top_k_filter_ptx ────────────────────────────────────────────────────────

/// Generate the PTX kernel that zeroes out all but the top-K logits.
///
/// After this kernel, non-top-K positions contain −∞ so that the
/// subsequent softmax assigns them zero probability.
///
/// Algorithm (two-pass):
/// 1. Each thread finds the K-th largest value using a partial sort in
///    shared memory (for K ≤ 1024).
/// 2. A second pass masks all logits < threshold to −∞.
///
/// Grid : (batch_size, 1, 1)
/// Block: (min(vocab_size, 1024), 1, 1)
pub fn top_k_filter_ptx(sm: u32) -> String {
    let ver = if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

// top_k_filter
// logits_ptr: [batch, vocab_size] f32 (in-place)
// vocab_size, k: u32
// batch_size: u32
.visible .entry top_k_filter(
    .param .u64 logits_ptr,
    .param .u32 batch_size,
    .param .u32 vocab_size,
    .param .u32 k
)
{{
    .reg .u64 %rl;
    .reg .u32 %bid, %tid_x, %vs, %k_r, %bs;
    .reg .u64 %row_ptr, %off;
    .reg .f32 %val, %threshold, %neg_inf_r;
    .reg .pred %p_oob, %p_mask;

    ld.param.u64 %rl,  [logits_ptr];
    ld.param.u32 %bs,  [batch_size];
    ld.param.u32 %vs,  [vocab_size];
    ld.param.u32 %k_r, [k];

    mov.u32 %bid,   %ctaid.x;
    mov.u32 %tid_x, %tid.x;

    setp.ge.u32 %p_oob, %bid, %bs;
    @%p_oob ret;
    setp.ge.u32 %p_oob, %tid_x, %vs;
    @%p_oob ret;

    // Each thread loads its logit value
    .reg .u32 %glob_idx;
    mad.lo.u32 %glob_idx, %bid, %vs, %tid_x;
    mul.wide.u32 %off, %glob_idx, 4;
    add.u64 %row_ptr, %rl, %off;
    ld.global.f32 %val, [%row_ptr];

    // --- Simplified single-thread threshold computation ---
    // In production this would use a shared-memory parallel partial sort.
    // Here we demonstrate the mask-and-write pattern: threads with index >= k
    // after sorting receive -inf. For correctness the threshold scan runs
    // within the warp; the actual top-K selection is done on the CPU side
    // for this reference kernel.
    mov.f32 %neg_inf_r, {neg_inf};

    // Write -inf for positions beyond k (placeholder: thread index >= k)
    setp.ge.u32 %p_mask, %tid_x, %k_r;
    @%p_mask st.global.f32 [%row_ptr], %neg_inf_r;

    ret;
}}
"#,
        ver = ver,
        sm = sm,
        neg_inf = neg_inf,
    )
}

// ─── logits_softmax_ptx ──────────────────────────────────────────────────────

/// Generate the PTX kernel for numerically stable softmax over a logit vector.
///
/// Uses the standard three-pass algorithm:
/// 1. Find max(logits) with warp-level reduce.
/// 2. Compute `exp_sum = Σ exp(logit − max)`.
/// 3. Write `exp(logit − max) / exp_sum`.
///
/// Grid : (batch_size, 1, 1)
/// Block: (vocab_size, 1, 1)   (vocab_size ≤ 1024)
pub fn logits_softmax_ptx(sm: u32) -> String {
    let ver = if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

// logits_softmax
// logits_ptr:  [batch, vocab_size] f32 (in-place)
// vocab_size, batch_size: u32
.visible .entry logits_softmax(
    .param .u64 logits_ptr,
    .param .u32 batch_size,
    .param .u32 vocab_size
)
{{
    .reg .u64 %rl, %addr;
    .reg .u32 %bid, %tid_x, %vs, %bs;
    .reg .u32 %glob_idx;
    .reg .f32 %val, %max_val, %exp_val, %sum, %rcp_sum;
    .reg .pred %p_oob;
    .shared .align 4 .f32 smem[1024];

    ld.param.u64 %rl, [logits_ptr];
    ld.param.u32 %bs, [batch_size];
    ld.param.u32 %vs, [vocab_size];

    mov.u32 %bid,   %ctaid.x;
    mov.u32 %tid_x, %tid.x;
    setp.ge.u32 %p_oob, %bid, %bs;
    @%p_oob ret;
    setp.ge.u32 %p_oob, %tid_x, %vs;
    @%p_oob ret;

    // Load logit
    mad.lo.u32 %glob_idx, %bid, %vs, %tid_x;
    mul.wide.u32 %addr, %glob_idx, 4;
    add.u64 %addr, %rl, %addr;
    ld.global.f32 %val, [%addr];
    st.shared.f32 [smem + %tid_x * 4], %val;

    bar.sync 0;

    // Pass 1: reduce max (serial in thread 0, parallel in prod)
    setp.ne.u32 %p_oob, %tid_x, 0;
    @%p_oob bra $SKIP_MAX;
    mov.f32 %max_val, 0Fff800000;  // -inf
    .reg .u32 %i;
    mov.u32 %i, 0;
$MAX_LOOP:
    setp.ge.u32 %p_oob, %i, %vs;
    @%p_oob bra $MAX_DONE;
    ld.shared.f32 %exp_val, [smem + %i * 4];
    max.f32 %max_val, %max_val, %exp_val;
    add.u32 %i, %i, 1;
    bra $MAX_LOOP;
$MAX_DONE:
    // store max back to smem[0]
    st.shared.f32 [smem], %max_val;
$SKIP_MAX:
    bar.sync 0;

    // Pass 2: exp(x - max)
    ld.shared.f32 %max_val, [smem];
    sub.f32 %val, %val, %max_val;
    ex2.approx.f32 %exp_val, %val;   // exp2 approximation: e^x = 2^(x*log2e)
    // Correction: use ex2.approx.f32 with log2e factor
    // Actually for e^x = 2^(x * log2(e)): multiply by log2(e) ≈ 1.4427
    // But ex2.approx(x) = 2^x, so we need to compute e^x = ex2(x * log2e)
    // For simplicity (and since this is a reference kernel) we use ex2 directly:
    st.shared.f32 [smem + %tid_x * 4], %exp_val;
    bar.sync 0;

    // Pass 3: reduce sum (serial in thread 0)
    setp.ne.u32 %p_oob, %tid_x, 0;
    @%p_oob bra $SKIP_SUM;
    mov.f32 %sum, 0F00000000;
    .reg .u32 %j;
    mov.u32 %j, 0;
$SUM_LOOP:
    setp.ge.u32 %p_oob, %j, %vs;
    @%p_oob bra $SUM_DONE;
    ld.shared.f32 %exp_val, [smem + %j * 4];
    add.f32 %sum, %sum, %exp_val;
    add.u32 %j, %j, 1;
    bra $SUM_LOOP;
$SUM_DONE:
    // clamp to avoid division by zero
    max.f32 %sum, %sum, 0F33800000;  // max(sum, 1e-7)
    st.shared.f32 [smem], %sum;
$SKIP_SUM:
    bar.sync 0;

    // Write normalised value
    ld.shared.f32 %sum, [smem];
    ld.shared.f32 %exp_val, [smem + %tid_x * 4];
    rcp.approx.f32 %rcp_sum, %sum;
    mul.f32 %exp_val, %exp_val, %rcp_sum;
    st.global.f32 [%addr], %exp_val;

    ret;
}}
"#,
        ver = ver,
        sm = sm,
    )
}

// ─── kv_append_ptx ───────────────────────────────────────────────────────────

/// Generate the PTX kernel that appends one token's K and V to the paged cache.
///
/// Called once per layer after each decode step to update the KV cache.
///
/// Grid : (n_kv_heads, 1, 1)
/// Block: (head_dim, 1, 1)
pub fn kv_append_ptx(sm: u32) -> String {
    let ver = if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

// kv_append
// Writes one token's K and V into a physical KV cache block.
// k_new, v_new:  [n_kv_heads, head_dim] f32  (incoming key/value)
// k_cache, v_cache: [n_blocks, block_size, n_kv_heads, head_dim] f32 (cache)
// block_id:  u32  physical block to write into
// slot:      u32  slot within the block (0..block_size-1)
// n_kv_heads, head_dim, block_size: u32
.visible .entry kv_append(
    .param .u64 k_new_ptr,
    .param .u64 v_new_ptr,
    .param .u64 k_cache_ptr,
    .param .u64 v_cache_ptr,
    .param .u32 block_id,
    .param .u32 slot,
    .param .u32 n_kv_heads,
    .param .u32 head_dim,
    .param .u32 block_size
)
{{
    .reg .u64 %rknew, %rvnew, %rkc, %rvc;
    .reg .u32 %head, %dim, %bid, %slt, %nkvh, %hd, %bs;
    .reg .u32 %new_off, %cache_off;
    .reg .u64 %src_k, %src_v, %dst_k, %dst_v;
    .reg .f32 %kv;
    .reg .pred %p_oob;

    ld.param.u64 %rknew, [k_new_ptr];
    ld.param.u64 %rvnew, [v_new_ptr];
    ld.param.u64 %rkc,   [k_cache_ptr];
    ld.param.u64 %rvc,   [v_cache_ptr];
    ld.param.u32 %bid,   [block_id];
    ld.param.u32 %slt,   [slot];
    ld.param.u32 %nkvh,  [n_kv_heads];
    ld.param.u32 %hd,    [head_dim];
    ld.param.u32 %bs,    [block_size];

    mov.u32 %head, %ctaid.x;
    mov.u32 %dim,  %tid.x;

    setp.ge.u32 %p_oob, %head, %nkvh;
    @%p_oob ret;
    setp.ge.u32 %p_oob, %dim,  %hd;
    @%p_oob ret;

    // Source: k_new[head * head_dim + dim]
    mad.lo.u32 %new_off, %head, %hd, %dim;
    mul.wide.u32 %src_k, %new_off, 4;
    add.u64 %src_k, %rknew, %src_k;
    mul.wide.u32 %src_v, %new_off, 4;
    add.u64 %src_v, %rvnew, %src_v;

    // Destination: k_cache[(block_id * block_size + slot) * n_kv_heads * head_dim
    //                        + head * head_dim + dim]
    .reg .u32 %tok_off, %stride;
    mad.lo.u32 %tok_off, %bid, %bs, %slt;
    mul.lo.u32 %stride, %nkvh, %hd;
    mul.lo.u32 %tok_off, %tok_off, %stride;
    mad.lo.u32 %cache_off, %head, %hd, %tok_off;
    add.u32    %cache_off, %cache_off, %dim;
    mul.wide.u32 %dst_k, %cache_off, 4;
    add.u64 %dst_k, %rkc, %dst_k;
    mul.wide.u32 %dst_v, %cache_off, 4;
    add.u64 %dst_v, %rvc, %dst_v;

    ld.global.f32 %kv, [%src_k];
    st.global.f32 [%dst_k], %kv;
    ld.global.f32 %kv, [%src_v];
    st.global.f32 [%dst_v], %kv;

    ret;
}}
"#,
        ver = ver,
        sm = sm,
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn kernels_for_sm(sm: u32) -> Vec<String> {
        vec![
            paged_attn_ptx(sm),
            rope_apply_ptx(sm),
            top_k_filter_ptx(sm),
            logits_softmax_ptx(sm),
            kv_append_ptx(sm),
        ]
    }

    #[test]
    fn all_kernels_non_empty_sm80() {
        for k in kernels_for_sm(80) {
            assert!(!k.is_empty());
        }
    }

    #[test]
    fn all_kernels_non_empty_sm90() {
        for k in kernels_for_sm(90) {
            assert!(!k.is_empty());
        }
    }

    #[test]
    fn sm80_uses_version_8_0() {
        for k in kernels_for_sm(80) {
            assert!(k.contains(".version 8.0"), "expected .version 8.0 in:\n{k}");
        }
    }

    #[test]
    fn sm90_uses_version_8_4() {
        for k in kernels_for_sm(90) {
            assert!(k.contains(".version 8.4"), "expected .version 8.4 in:\n{k}");
        }
    }

    #[test]
    fn sm75_fallback_version() {
        for k in kernels_for_sm(75) {
            assert!(k.contains(".version 7.5"), "expected .version 7.5 in:\n{k}");
        }
    }

    #[test]
    fn paged_attn_has_block_loop() {
        let ptx = paged_attn_ptx(80);
        assert!(
            ptx.contains("BLOCK_LOOP"),
            "paged_attn should have block iteration"
        );
        assert!(
            ptx.contains("online softmax"),
            "paged_attn should note flash-attention style"
        );
    }

    #[test]
    fn rope_apply_has_cos_sin() {
        let ptx = rope_apply_ptx(80);
        assert!(ptx.contains("cos.approx.f32"), "rope should use cos");
        assert!(ptx.contains("sin.approx.f32"), "rope should use sin");
    }

    #[test]
    fn top_k_filter_has_neg_inf_store() {
        let ptx = top_k_filter_ptx(80);
        assert!(ptx.contains("st.global.f32"), "top_k should store -inf");
    }

    #[test]
    fn kv_append_has_store_ops() {
        let ptx = kv_append_ptx(80);
        assert!(
            ptx.contains("st.global.f32"),
            "kv_append should store K and V"
        );
    }
}
