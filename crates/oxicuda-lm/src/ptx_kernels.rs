//! PTX GPU kernel sources for LLM operations.
//!
//! Each function returns a PTX program as a `String`.  These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`embedding_forward_ptx`] | Token embedding table lookup |
//! | [`rope_apply_ptx`] | Rotary positional embedding (in-place) |
//! | [`silu_gate_ptx`] | SiLU gating for SwiGLU feed-forward networks |
//! | [`rms_norm_ptx`] | RMSNorm forward pass with learnable scale |
//! | [`causal_attn_softmax_ptx`] | Per-head causal attention softmax |

// ─── PTX header helper ───────────────────────────────────────────────────────

fn ptx_header(sm: u32) -> String {
    // PTX language version is determined by the SM family; the `.target`
    // directive uses the exact SM version requested.
    let ptx_ver = if sm >= 100 {
        "8.7"
    } else if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    format!(".version {ptx_ver}\n.target sm_{sm}\n.address_size 64\n\n")
}

// ─── Kernel 1: embedding_forward ─────────────────────────────────────────────

/// Token embedding lookup: copies rows from the embedding table.
///
/// # Parameters (as aligned `param` entries)
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_token_ids` | `u64` (pointer to `u32`) | Token id array `[n_tokens]` |
/// | `p_embed`     | `u64` (pointer to `f32`) | Embedding table `[vocab × embed_dim]` |
/// | `p_out`       | `u64` (pointer to `f32`) | Output buffer `[n_tokens × embed_dim]` |
/// | `embed_dim`   | `u32` | Embedding dimension |
/// | `n_tokens`    | `u32` | Number of tokens |
///
/// Launch with `grid = ceil(n_tokens * embed_dim / 256)`, `block = 256`.
pub fn embedding_forward_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry embedding_forward(
    .param .u64 p_token_ids,
    .param .u64 p_embed,
    .param .u64 p_out,
    .param .u32 embed_dim,
    .param .u32 n_tokens
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<8>;
    .reg .f32  %f0;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_token_ids];
    ld.param.u64  %rd1, [p_embed];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [embed_dim];
    ld.param.u32  %r1,  [n_tokens];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

    // if tid >= n_tokens * embed_dim, exit
    mul.lo.u32    %r6, %r1, %r0;
    setp.ge.u32   %p0, %r5, %r6;
    @%p0 bra $DONE;

    // tok_idx = tid / embed_dim
    div.u32       %r6, %r5, %r0;
    // dim_idx = tid % embed_dim
    rem.u32       %r7, %r5, %r0;

    // token_id = p_token_ids[tok_idx]
    mul.wide.u32  %rd3, %r6, 4;
    add.u64       %rd3, %rd0, %rd3;
    ld.global.u32 %r6, [%rd3];

    // src = p_embed + (token_id * embed_dim + dim_idx) * 4
    mad.lo.u32    %r7, %r6, %r0, %r7;
    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd3, %rd1, %rd3;
    ld.global.f32 %f0, [%rd3];

    // dst = p_out + tid * 4
    mul.wide.u32  %rd4, %r5, 4;
    add.u64       %rd4, %rd2, %rd4;
    st.global.f32 [%rd4], %f0;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 2: rope_apply ────────────────────────────────────────────────────

/// Rotary positional embedding applied in-place to Q or K.
///
/// Operates on tensor of shape `[n_tokens × n_heads × head_dim]`.
/// Each thread handles one (dim/2) pair for one (token, head).
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_x`        | `u64` → `f32*` | Tensor to rotate in-place |
/// | `p_cos`      | `u64` → `f32*` | Cos table `[max_pos × head_dim/2]` |
/// | `p_sin`      | `u64` → `f32*` | Sin table `[max_pos × head_dim/2]` |
/// | `n_heads`    | `u32` | Number of heads |
/// | `head_dim`   | `u32` | Head dimension (even) |
/// | `n_tokens`   | `u32` | Sequence length |
/// | `pos_offset` | `u32` | Position of first token (for KV cache offset) |
///
/// Launch with `grid = ceil(n_tokens * n_heads * head_dim/2 / 256)`, `block = 256`.
pub fn rope_apply_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry rope_apply(
    .param .u64 p_x,
    .param .u64 p_cos,
    .param .u64 p_sin,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 n_tokens,
    .param .u32 pos_offset
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<16>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_cos];
    ld.param.u64  %rd2, [p_sin];
    ld.param.u32  %r0,  [n_heads];
    ld.param.u32  %r1,  [head_dim];
    ld.param.u32  %r2,  [n_tokens];
    ld.param.u32  %r3,  [pos_offset];

    // half_dim = head_dim / 2
    shr.u32       %r4, %r1, 1;

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r5, %ntid.x;
    mov.u32       %r6, %ctaid.x;
    mov.u32       %r7, %tid.x;
    mad.lo.u32    %r8, %r5, %r6, %r7;

    // total = n_tokens * n_heads * half_dim
    mul.lo.u32    %r9, %r2, %r0;
    mul.lo.u32    %r9, %r9, %r4;
    setp.ge.u32   %p0, %r8, %r9;
    @%p0 bra $DONE;

    // pair_idx        = tid % half_dim
    rem.u32       %r9,  %r8, %r4;
    // head_tok_idx    = tid / half_dim
    div.u32       %r10, %r8, %r4;
    // head_idx        = head_tok_idx % n_heads
    rem.u32       %r11, %r10, %r0;
    // tok_idx         = head_tok_idx / n_heads
    div.u32       %r12, %r10, %r0;

    // abs_pos = pos_offset + tok_idx
    add.u32       %r13, %r3, %r12;

    // base = (tok_idx * n_heads + head_idx) * head_dim
    mad.lo.u32    %r14, %r12, %r0, %r11;
    mul.lo.u32    %r14, %r14, %r1;
    // offset_x0 = (base + pair_idx*2) * 4
    shl.b32       %r15, %r9, 1;
    add.u32       %r15, %r14, %r15;
    mul.wide.u32  %rd3, %r15, 4;
    add.u64       %rd3, %rd0, %rd3;
    ld.global.f32 %f0, [%rd3];       // x0
    ld.global.f32 %f1, [%rd3 + 4];   // x1

    // cos/sin offset = (abs_pos * half_dim + pair_idx) * 4
    mad.lo.u32    %r15, %r13, %r4, %r9;
    mul.wide.u32  %rd4, %r15, 4;
    add.u64       %rd5, %rd1, %rd4;
    add.u64       %rd6, %rd2, %rd4;
    ld.global.f32 %f2, [%rd5];       // cos
    ld.global.f32 %f3, [%rd6];       // sin

    // out0 = x0*cos - x1*sin
    mul.f32       %f4, %f0, %f2;
    mul.f32       %f5, %f1, %f3;
    sub.f32       %f6, %f4, %f5;
    // out1 = x0*sin + x1*cos
    mul.f32       %f4, %f0, %f3;
    mul.f32       %f5, %f1, %f2;
    add.f32       %f7, %f4, %f5;

    st.global.f32 [%rd3],     %f6;
    st.global.f32 [%rd3 + 4], %f7;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 3: silu_gate ─────────────────────────────────────────────────────

/// SiLU-gated activation for SwiGLU feed-forward networks.
///
/// Computes `out[i] = silu(gate[i]) * up[i]` where
/// `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_gate` | `u64` → `f32*` | Gate vector `[n]` |
/// | `p_up`   | `u64` → `f32*` | Up-projection vector `[n]` |
/// | `p_out`  | `u64` → `f32*` | Output `[n]` |
/// | `n`      | `u32` | Number of elements |
///
/// Launch with `grid = ceil(n / 256)`, `block = 256`.
pub fn silu_gate_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry silu_gate(
    .param .u64 p_gate,
    .param .u64 p_up,
    .param .u64 p_out,
    .param .u32 n
)
{{
    .reg .u64  %rd<5>;
    .reg .u32  %r<5>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_gate];
    ld.param.u64  %rd1, [p_up];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $DONE;

    mul.wide.u32  %rd3, %r4, 4;
    add.u64       %rd3, %rd0, %rd3;
    ld.global.f32 %f0, [%rd3];       // gate

    add.u64       %rd4, %rd1, %rd3;
    sub.u64       %rd4, %rd4, %rd0;
    add.u64       %rd4, %rd1, %rd3;
    sub.u64       %rd4, %rd4, %rd0;

    // Recompute up offset independently
    mul.wide.u32  %rd4, %r4, 4;
    add.u64       %rd4, %rd1, %rd4;
    ld.global.f32 %f1, [%rd4];       // up

    // silu(gate) = gate * sigmoid(gate) = gate / (1 + exp2(-gate * log2e))
    // Use ex2.approx: exp2(x) ≈ e^(x * ln2); so exp(-gate) = ex2(-gate * log2e)
    // log2(e) = 1.44269504
    mul.f32       %f2, %f0, 0F3FB8AA3B; // f0 * log2e  (0x3FB8AA3B = log2(e))
    neg.f32       %f2, %f2;              // -gate * log2e
    ex2.approx.f32 %f3, %f2;            // exp(-gate)
    add.f32       %f3, %f3, 0F3F800000; // 1 + exp(-gate)  (0x3F800000 = 1.0f)
    rcp.approx.f32 %f3, %f3;            // sigmoid(gate)
    mul.f32       %f2, %f0, %f3;        // silu(gate) = gate * sigmoid(gate)
    mul.f32       %f2, %f2, %f1;        // silu(gate) * up

    mul.wide.u32  %rd3, %r4, 4;
    add.u64       %rd3, %rd2, %rd3;
    st.global.f32 [%rd3], %f2;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 4: rms_norm ──────────────────────────────────────────────────────

/// RMSNorm forward pass: `out = x / rms(x) * weight`.
///
/// One thread block processes one token.  Uses shared memory warp reduction.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_x`       | `u64` → `f32*` | Input `[n_tokens × dim]` |
/// | `p_weight`  | `u64` → `f32*` | Scale `[dim]` |
/// | `p_out`     | `u64` → `f32*` | Output `[n_tokens × dim]` |
/// | `dim`       | `u32` | Hidden dimension |
/// | `n_tokens`  | `u32` | Sequence length |
/// | `eps`       | `f32` | Stability epsilon |
///
/// Launch with `grid = n_tokens`, `block = min(dim, 256)`.
pub fn rms_norm_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry rms_norm(
    .param .u64 p_x,
    .param .u64 p_weight,
    .param .u64 p_out,
    .param .u32 dim,
    .param .u32 n_tokens,
    .param .f32 eps
)
{{
    // Shared memory for partial sums (max 256 threads per block)
    .shared .align 4 .f32 smem[256];

    .reg .u64  %rd<6>;
    .reg .u32  %r<8>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_weight];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [dim];
    ld.param.u32  %r1,  [n_tokens];
    ld.param.f32  %f0,  [eps];

    // block_id = blockIdx.x (one block per token)
    mov.u32       %r2, %ctaid.x;
    setp.ge.u32   %p0, %r2, %r1;
    @%p0 bra $DONE;

    mov.u32       %r3, %tid.x;   // local thread id
    mov.u32       %r4, %ntid.x;  // block width

    // Phase 1: grid-stride accumulate sum of squares
    mov.f32       %f1, 0F00000000; // partial_sum = 0.0
$LOOP:
    setp.ge.u32   %p0, %r3, %r0;  // if thread >= dim, skip
    @%p0 bra $SKIP_LOAD;
    // idx = block_id * dim + thread_id
    mad.lo.u32    %r5, %r2, %r0, %r3;
    mul.wide.u32  %rd3, %r5, 4;
    add.u64       %rd3, %rd0, %rd3;
    ld.global.f32 %f2, [%rd3];
    fma.rn.f32    %f1, %f2, %f2, %f1; // partial += x*x
$SKIP_LOAD:
    add.u32       %r3, %r3, %r4;  // thread_id += block_width
    setp.lt.u32   %p0, %r3, %r0;
    @%p0 bra $LOOP;

    // Store partial in smem
    mov.u32       %r3, %tid.x;
    mul.wide.u32  %rd3, %r3, 4;
    add.u64       %rd3, [smem], %rd3;
    st.shared.f32 [%rd3], %f1;
    bar.sync 0;

    // Warp-level butterfly reduction on smem (assume block <= 256)
    // stride 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    setp.lt.u32   %p0, %r3, 128;
    @%p0 ld.shared.f32 %f2, [%rd3 + 512]; // stride 128 * 4
    @%p0 add.f32       %f1, %f1, %f2;
    @%p0 st.shared.f32 [%rd3], %f1;
    bar.sync 0;

    setp.lt.u32   %p0, %r3, 64;
    @%p0 ld.shared.f32 %f2, [%rd3 + 256];
    @%p0 add.f32       %f1, %f1, %f2;
    @%p0 st.shared.f32 [%rd3], %f1;
    bar.sync 0;

    setp.lt.u32   %p0, %r3, 32;
    @%p0 ld.shared.f32 %f2, [%rd3 + 128];
    @%p0 add.f32       %f1, %f1, %f2;
    @%p0 st.shared.f32 [%rd3], %f1;
    bar.sync 0;

    setp.lt.u32   %p0, %r3, 16;
    @%p0 ld.shared.f32 %f2, [%rd3 + 64];
    @%p0 add.f32       %f1, %f1, %f2;
    @%p0 st.shared.f32 [%rd3], %f1;
    bar.sync 0;

    setp.lt.u32   %p0, %r3, 8;
    @%p0 ld.shared.f32 %f2, [%rd3 + 32];
    @%p0 add.f32       %f1, %f1, %f2;
    @%p0 st.shared.f32 [%rd3], %f1;
    bar.sync 0;

    setp.lt.u32   %p0, %r3, 4;
    @%p0 ld.shared.f32 %f2, [%rd3 + 16];
    @%p0 add.f32       %f1, %f1, %f2;
    @%p0 st.shared.f32 [%rd3], %f1;
    bar.sync 0;

    setp.lt.u32   %p0, %r3, 2;
    @%p0 ld.shared.f32 %f2, [%rd3 + 8];
    @%p0 add.f32       %f1, %f1, %f2;
    @%p0 st.shared.f32 [%rd3], %f1;
    bar.sync 0;

    setp.lt.u32   %p0, %r3, 1;
    @%p0 ld.shared.f32 %f2, [%rd3 + 4];
    @%p0 add.f32       %f1, %f1, %f2;
    @%p0 st.shared.f32 [%rd3], %f1;
    bar.sync 0;

    // Thread 0 finalises: rms = sqrt(sum/dim + eps)
    setp.ne.u32   %p1, %r3, 0;
    ld.shared.f32 %f1, [smem];
    cvt.rn.f32.u32 %f2, %r0;
    div.approx.f32 %f1, %f1, %f2;
    add.f32       %f1, %f1, %f0;    // mean_sq + eps
    sqrt.approx.f32 %f1, %f1;       // rms
    rcp.approx.f32  %f1, %f1;       // 1/rms; broadcast via smem
    st.shared.f32 [smem], %f1;
    bar.sync 0;
    ld.shared.f32 %f1, [smem];      // all threads load inv_rms

    // Phase 2: apply normalization + scale
    mov.u32       %r3, %tid.x;
$NORM_LOOP:
    setp.ge.u32   %p0, %r3, %r0;
    @%p0 bra $DONE;

    mad.lo.u32    %r5, %r2, %r0, %r3;
    mul.wide.u32  %rd3, %r5, 4;
    add.u64       %rd3, %rd0, %rd3;
    ld.global.f32 %f2, [%rd3];       // x[tok, dim]

    mul.wide.u32  %rd4, %r3, 4;
    add.u64       %rd4, %rd1, %rd4;
    ld.global.f32 %f3, [%rd4];       // weight[dim]

    mul.f32       %f2, %f2, %f1;     // x * inv_rms
    mul.f32       %f2, %f2, %f3;     // * weight

    add.u64       %rd3, %rd2, %rd3;
    sub.u64       %rd3, %rd3, %rd0;
    add.u64       %rd3, %rd2, %rd4;
    sub.u64       %rd3, %rd3, %rd1;
    // Recompute output pointer
    mul.wide.u32  %rd5, %r5, 4;
    add.u64       %rd5, %rd2, %rd5;
    st.global.f32 [%rd5], %f2;

    add.u32       %r3, %r3, %r4;
    bra $NORM_LOOP;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 5: causal_attn_softmax ───────────────────────────────────────────

/// Causal attention softmax over a single query's score vector.
///
/// Applies the causal mask (sets future positions to −∞) then computes
/// a numerically stable softmax in-place.  One thread block handles one
/// (query_position, head) pair.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_scores`   | `u64` → `f32*` | Scores `[n_q × n_heads × kv_len]` |
/// | `kv_len`     | `u32` | Total KV sequence length |
/// | `n_heads`    | `u32` | Number of attention heads |
/// | `past_len`   | `u32` | Number of past KV tokens (for absolute position) |
///
/// Launch with `grid = (n_q, n_heads)`, `block = min(kv_len, 256)`.
pub fn causal_attn_softmax_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry causal_attn_softmax(
    .param .u64 p_scores,
    .param .u32 kv_len,
    .param .u32 n_heads,
    .param .u32 past_len
)
{{
    .shared .align 4 .f32 smem[256]; // for max and sum reduction

    .reg .u64  %rd<5>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_scores];
    ld.param.u32  %r0,  [kv_len];
    ld.param.u32  %r1,  [n_heads];
    ld.param.u32  %r2,  [past_len];

    // q_pos = blockIdx.x, head_idx = blockIdx.y
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %ctaid.y;
    mov.u32       %r5, %tid.x;
    mov.u32       %r6, %ntid.x;

    // absolute query position: past_len + q_pos
    add.u32       %r7, %r2, %r3;

    // base offset for this (q_pos, head) row in scores
    // row = q_pos * n_heads + head_idx
    mad.lo.u32    %r8, %r3, %r1, %r4;
    // row_start (in elements) = row * kv_len
    mul.lo.u32    %r9, %r8, %r0;

    // Phase 1: apply causal mask and find max
    mov.f32       %f0, 0FFF800000; // -inf
$MASK_LOOP:
    add.u32       %r8, %r9, %r5;   // global element index
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $MASK_DONE;

    // kv position = r5 (thread iterates over kv positions)
    // mask: kv_pos > q_abs_pos → set -inf
    setp.gt.u32   %p1, %r5, %r7;

    mul.wide.u32  %rd1, %r8, 4;
    add.u64       %rd1, %rd0, %rd1;
    ld.global.f32 %f1, [%rd1];
    selp.f32      %f1, 0FFF800000, %f1, %p1; // select -inf if masked
    st.global.f32 [%rd1], %f1;

    // track max
    setp.gt.f32   %p0, %f1, %f0;
    selp.f32      %f0, %f1, %f0, %p0;

    add.u32       %r5, %r5, %r6;
    setp.lt.u32   %p0, %r5, %r0;
    @%p0 bra $MASK_LOOP;
$MASK_DONE:

    // Reduce max across block via smem
    mov.u32       %r5, %tid.x;
    mul.wide.u32  %rd1, %r5, 4;
    add.u64       %rd1, [smem], %rd1;
    st.shared.f32 [%rd1], %f0;
    bar.sync 0;

    // Simple sequential reduction by thread 0
    setp.ne.u32   %p0, %r5, 0;
    @%p0 bra $MAX_DONE;
    mov.f32       %f0, 0FFF800000;
    mov.u32       %r8, 0;
$MAX_RED:
    setp.ge.u32   %p0, %r8, %r6;
    @%p0 bra $MAX_DONE;
    mul.wide.u32  %rd2, %r8, 4;
    add.u64       %rd2, [smem], %rd2;
    ld.shared.f32 %f1, [%rd2];
    setp.gt.f32   %p0, %f1, %f0;
    selp.f32      %f0, %f1, %f0, %p0;
    add.u32       %r8, %r8, 1;
    bra $MAX_RED;
$MAX_DONE:
    st.shared.f32 [smem], %f0;
    bar.sync 0;
    ld.shared.f32 %f0, [smem]; // broadcast max to all threads

    // Phase 2: compute exp(score - max) and accumulate sum
    mov.u32       %r5, %tid.x;
    mov.f32       %f2, 0F00000000; // partial_sum
$EXP_LOOP:
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $EXP_DONE;

    add.u32       %r8, %r9, %r5;
    mul.wide.u32  %rd1, %r8, 4;
    add.u64       %rd1, %rd0, %rd1;
    ld.global.f32 %f1, [%rd1];

    sub.f32       %f3, %f1, %f0;    // score - max
    mul.f32       %f3, %f3, 0F3FB8AA3B; // * log2(e)
    ex2.approx.f32 %f3, %f3;        // exp(score - max)
    st.global.f32 [%rd1], %f3;
    add.f32       %f2, %f2, %f3;

    add.u32       %r5, %r5, %r6;
    setp.lt.u32   %p0, %r5, %r0;
    @%p0 bra $EXP_LOOP;
$EXP_DONE:

    // Reduce sum across block
    mov.u32       %r5, %tid.x;
    mul.wide.u32  %rd1, %r5, 4;
    add.u64       %rd1, [smem], %rd1;
    st.shared.f32 [%rd1], %f2;
    bar.sync 0;

    setp.ne.u32   %p0, %r5, 0;
    @%p0 bra $SUM_DONE;
    mov.f32       %f2, 0F00000000;
    mov.u32       %r8, 0;
$SUM_RED:
    setp.ge.u32   %p0, %r8, %r6;
    @%p0 bra $SUM_DONE;
    mul.wide.u32  %rd2, %r8, 4;
    add.u64       %rd2, [smem], %rd2;
    ld.shared.f32 %f3, [%rd2];
    add.f32       %f2, %f2, %f3;
    add.u32       %r8, %r8, 1;
    bra $SUM_RED;
$SUM_DONE:
    st.shared.f32 [smem], %f2;
    bar.sync 0;
    ld.shared.f32 %f2, [smem];      // broadcast sum
    rcp.approx.f32 %f2, %f2;        // inv_sum

    // Phase 3: normalize
    mov.u32       %r5, %tid.x;
$NORM_LOOP:
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $DONE;
    add.u32       %r8, %r9, %r5;
    mul.wide.u32  %rd1, %r8, 4;
    add.u64       %rd1, %rd0, %rd1;
    ld.global.f32 %f1, [%rd1];
    mul.f32       %f1, %f1, %f2;
    st.global.f32 [%rd1], %f1;
    add.u32       %r5, %r5, %r6;
    bra $NORM_LOOP;

$DONE:
    ret;
}}
"#
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SM_VERSIONS: &[u32] = &[75, 80, 86, 90, 100, 120];

    fn check_ptx_header(ptx: &str, sm: u32) {
        let expected_target = format!("sm_{sm}");
        assert!(
            ptx.contains(&expected_target),
            "missing target {expected_target} in:\n{ptx}"
        );
        assert!(ptx.contains(".address_size 64"), "missing .address_size 64");
        assert!(ptx.contains(".version"), "missing .version");
    }

    #[test]
    fn embedding_forward_all_sm() {
        for &sm in SM_VERSIONS {
            let ptx = embedding_forward_ptx(sm);
            check_ptx_header(&ptx, sm);
            assert!(ptx.contains("embedding_forward"), "missing entry name");
            assert!(ptx.contains("p_token_ids"), "missing param p_token_ids");
            assert!(ptx.contains("embed_dim"), "missing param embed_dim");
        }
    }

    #[test]
    fn rope_apply_all_sm() {
        for &sm in SM_VERSIONS {
            let ptx = rope_apply_ptx(sm);
            check_ptx_header(&ptx, sm);
            assert!(ptx.contains("rope_apply"), "missing entry name");
            assert!(ptx.contains("pos_offset"), "missing param pos_offset");
            // Kernel rotates using pre-computed cos/sin tables; verify the
            // multiply-subtract arithmetic that implements the rotation.
            assert!(
                ptx.contains("sub.f32"),
                "missing RoPE rotation sub instruction"
            );
        }
    }

    #[test]
    fn silu_gate_all_sm() {
        for &sm in SM_VERSIONS {
            let ptx = silu_gate_ptx(sm);
            check_ptx_header(&ptx, sm);
            assert!(ptx.contains("silu_gate"), "missing entry name");
            assert!(ptx.contains("p_gate"), "missing param p_gate");
            assert!(
                ptx.contains("ex2.approx.f32"),
                "missing SiLU exp instruction"
            );
        }
    }

    #[test]
    fn rms_norm_all_sm() {
        for &sm in SM_VERSIONS {
            let ptx = rms_norm_ptx(sm);
            check_ptx_header(&ptx, sm);
            assert!(ptx.contains("rms_norm"), "missing entry name");
            assert!(ptx.contains("sqrt.approx.f32"), "missing sqrt instruction");
            assert!(ptx.contains(".shared"), "missing shared memory declaration");
        }
    }

    #[test]
    fn causal_attn_softmax_all_sm() {
        for &sm in SM_VERSIONS {
            let ptx = causal_attn_softmax_ptx(sm);
            check_ptx_header(&ptx, sm);
            assert!(ptx.contains("causal_attn_softmax"), "missing entry name");
            assert!(ptx.contains("past_len"), "missing param past_len");
            assert!(
                ptx.contains("ex2.approx.f32"),
                "missing softmax exp instruction"
            );
        }
    }

    #[test]
    fn all_kernels_have_distinct_entry_names() {
        let sm = 80;
        let names = [
            ("embedding_forward", embedding_forward_ptx(sm)),
            ("rope_apply", rope_apply_ptx(sm)),
            ("silu_gate", silu_gate_ptx(sm)),
            ("rms_norm", rms_norm_ptx(sm)),
            ("causal_attn_softmax", causal_attn_softmax_ptx(sm)),
        ];
        for (name, ptx) in &names {
            assert!(
                ptx.contains(&format!(".entry {name}")),
                "entry .entry {name} not found"
            );
        }
    }
}
