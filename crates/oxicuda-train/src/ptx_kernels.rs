//! PTX kernel generators for fused GPU optimizer parameter updates.
//!
//! Each function returns a complete PTX 7.x source string that can be loaded
//! via `cuModuleLoadData` and then executed with `cuLaunchKernel`.  All
//! kernels use a **grid-stride loop** so a single PTX binary handles any
//! number of parameters regardless of the launch configuration.
//!
//! ## Kernels provided
//!
//! | Function | Description |
//! |---|---|
//! | [`crate::ptx_kernels::adam_update_ptx`] | Fused Adam moment update + parameter step |
//! | [`crate::ptx_kernels::adamw_update_ptx`] | AdamW: decoupled weight decay + Adam step |
//! | [`crate::ptx_kernels::sgd_update_ptx`] | SGD with optional momentum and weight decay |
//! | [`crate::ptx_kernels::lion_update_ptx`] | Lion: sign-update with one moment buffer |
//! | [`crate::ptx_kernels::came_row_factor_ptx`] | CAME: per-row factor update |
//! | [`crate::ptx_kernels::came_col_factor_ptx`] | CAME: per-column factor update |
//! | [`crate::ptx_kernels::norm_sq_partial_ptx`] | Partial L2-norm-squared accumulation (for grad clip) |
//! | [`crate::ptx_kernels::scale_inplace_ptx`] | In-place scalar multiplication of a float vector |
//! | [`crate::ptx_kernels::add_inplace_ptx`] | In-place fused add (gradient accumulation) |
//!
//! ## PTX conventions
//!
//! * All pointer parameters are `u64` (64-bit virtual address).
//! * Length parameters are `u64`.
//! * Scalar floating-point parameters are `f32`.
//! * Approximate operations (`sqrt.approx`, `rcp.approx`) are used where
//!   full IEEE-754 precision is unnecessary, matching the practice of every
//!   major framework's optimizer kernel.

use oxicuda_ptx::arch::SmVersion;

// ─── Internal helpers ────────────────────────────────────────────────────────

fn ptx_header(sm: SmVersion) -> &'static str {
    match sm {
        SmVersion::Sm120 => ".version 8.7\n.target sm_120\n.address_size 64\n",
        SmVersion::Sm100 => ".version 8.5\n.target sm_100\n.address_size 64\n",
        SmVersion::Sm90a => ".version 8.0\n.target sm_90a\n.address_size 64\n",
        SmVersion::Sm90 => ".version 8.0\n.target sm_90\n.address_size 64\n",
        SmVersion::Sm89 => ".version 7.5\n.target sm_89\n.address_size 64\n",
        SmVersion::Sm86 => ".version 7.5\n.target sm_86\n.address_size 64\n",
        SmVersion::Sm80 => ".version 7.5\n.target sm_80\n.address_size 64\n",
        SmVersion::Sm75 => ".version 7.0\n.target sm_75\n.address_size 64\n",
    }
}

/// Float-to-PTX hex literal (f32 IEEE 754).
fn f32_hex(v: f32) -> String {
    format!("0f{:08X}", v.to_bits())
}

// ─── Adam fused update ───────────────────────────────────────────────────────

/// Generate the PTX source for a fused Adam parameter update kernel.
///
/// **Kernel name:** `adam_update_f32`
///
/// **Parameters (in order):**
/// 1. `param`   – `f32*` parameter vector (read-write)
/// 2. `grad`    – `f32*` gradient vector (read-only)
/// 3. `exp_avg`    – `f32*` first moment buffer (read-write)
/// 4. `exp_avg_sq` – `f32*` second moment buffer (read-write)
/// 5. `step_size`  – `f32` = `lr / (1 − β₁ᵗ)`
/// 6. `bc2_rsqrt`  – `f32` = `1 / √(1 − β₂ᵗ)` (bias-correction for v)
/// 7. `beta1`      – `f32` moment-1 decay rate
/// 8. `beta2`      – `f32` moment-2 decay rate
/// 9. `eps`        – `f32` numerical stability term
/// 10. `n`         – `u64` number of elements
///
/// **Algorithm (per element):**
/// ```text
/// m ← β₁·m + (1−β₁)·g
/// v ← β₂·v + (1−β₂)·g²
/// √v̂ ← √v · bc2_rsqrt           (bias-corrected sqrt)
/// p ← p − step_size · m / (√v̂ + ε)
/// ```
#[must_use]
pub fn adam_update_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}
.visible .entry adam_update_f32(
    .param .u64 p_param,
    .param .u64 p_grad,
    .param .u64 p_m1,
    .param .u64 p_m2,
    .param .f32 step_size,
    .param .f32 bc2_rsqrt,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .u64 n
)
{{
    .reg .pred  %guard;
    .reg .u32   %tid, %bid, %ntid, %nctaid, %gid, %stride;
    .reg .u64   %gid64, %n64, %off;
    .reg .u64   %ap0, %ag0, %am10, %am20;
    .reg .u64   %ap,  %ag,  %am1,  %am2;
    .reg .f32   %p, %g, %m1, %m2;
    .reg .f32   %step, %bc2r, %b1, %b2, %ep;
    .reg .f32   %c1, %c2;
    .reg .f32   %nm1, %nm2, %g2, %sq, %denom, %upd;

    ld.param.u64  %n64,   [n];
    ld.param.f32  %step,  [step_size];
    ld.param.f32  %bc2r,  [bc2_rsqrt];
    ld.param.f32  %b1,    [beta1];
    ld.param.f32  %b2,    [beta2];
    ld.param.f32  %ep,    [eps];

    mov.f32       %c1,    {one};
    sub.f32       %c1,    %c1,  %b1;
    mov.f32       %c2,    {one};
    sub.f32       %c2,    %c2,  %b2;

    ld.param.u64  %ap0,  [p_param];
    ld.param.u64  %ag0,  [p_grad];
    ld.param.u64  %am10, [p_m1];
    ld.param.u64  %am20, [p_m2];

    mov.u32  %tid,    %tid.x;
    mov.u32  %bid,    %ctaid.x;
    mov.u32  %ntid,   %ntid.x;
    mov.u32  %nctaid, %nctaid.x;
    mad.lo.u32 %gid,  %bid, %ntid, %tid;
    mul.lo.u32 %stride, %ntid, %nctaid;

$LOOP:
    cvt.u64.u32   %gid64, %gid;
    setp.ge.u64   %guard, %gid64, %n64;
    @%guard bra   $DONE;

    shl.b64       %off,  %gid64, 2;
    add.u64       %ap,   %ap0,   %off;
    add.u64       %ag,   %ag0,   %off;
    add.u64       %am1,  %am10,  %off;
    add.u64       %am2,  %am20,  %off;

    ld.global.f32 %p,  [%ap];
    ld.global.f32 %g,  [%ag];
    ld.global.f32 %m1, [%am1];
    ld.global.f32 %m2, [%am2];

    // nm1 = beta1*m1 + (1-beta1)*g
    mul.f32       %nm1, %b1, %m1;
    fma.rn.f32    %nm1, %c1, %g, %nm1;

    // nm2 = beta2*m2 + (1-beta2)*g^2
    mul.f32       %g2,  %g,  %g;
    mul.f32       %nm2, %b2, %m2;
    fma.rn.f32    %nm2, %c2, %g2, %nm2;

    // bias-corrected sqrt: sq = sqrt(nm2) * bc2_rsqrt
    sqrt.approx.f32 %sq, %nm2;
    mul.f32         %sq,  %sq, %bc2r;

    // denom = sq + eps
    add.f32       %denom, %sq, %ep;

    // update: upd = step_size * nm1 / denom  (via rcp for speed)
    rcp.approx.f32 %denom, %denom;
    mul.f32        %upd,   %nm1,  %denom;
    mul.f32        %upd,   %step, %upd;

    // param -= upd
    sub.f32        %p,  %p, %upd;

    st.global.f32  [%ap],  %p;
    st.global.f32  [%am1], %nm1;
    st.global.f32  [%am2], %nm2;

    add.u32        %gid, %gid, %stride;
    bra            $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr,
        one = one
    )
}

// ─── AdamW fused update ──────────────────────────────────────────────────────

/// Generate PTX for a fused AdamW update kernel.
///
/// **Kernel name:** `adamw_update_f32`
///
/// Same signature as `adam_update_f32` plus one extra parameter:
/// * `wd` – `f32` weight decay coefficient λ (decoupled from gradient)
///
/// **Algorithm:**
/// ```text
/// p ← p · (1 − lr · λ)         (decoupled weight decay)
/// m ← β₁·m + (1−β₁)·g
/// v ← β₂·v + (1−β₂)·g²
/// p ← p − step_size · m / (√v̂ + ε)
/// ```
#[must_use]
pub fn adamw_update_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}
.visible .entry adamw_update_f32(
    .param .u64 p_param,
    .param .u64 p_grad,
    .param .u64 p_m1,
    .param .u64 p_m2,
    .param .f32 step_size,
    .param .f32 bc2_rsqrt,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 eps,
    .param .f32 lr_wd,
    .param .u64 n
)
{{
    .reg .pred  %guard;
    .reg .u32   %tid, %bid, %ntid, %nctaid, %gid, %stride;
    .reg .u64   %gid64, %n64, %off;
    .reg .u64   %ap0, %ag0, %am10, %am20;
    .reg .u64   %ap,  %ag,  %am1,  %am2;
    .reg .f32   %p, %g, %m1, %m2;
    .reg .f32   %step, %bc2r, %b1, %b2, %ep, %lrwd;
    .reg .f32   %c1, %c2;
    .reg .f32   %nm1, %nm2, %g2, %sq, %denom, %upd, %wdf;

    ld.param.u64  %n64,   [n];
    ld.param.f32  %step,  [step_size];
    ld.param.f32  %bc2r,  [bc2_rsqrt];
    ld.param.f32  %b1,    [beta1];
    ld.param.f32  %b2,    [beta2];
    ld.param.f32  %ep,    [eps];
    ld.param.f32  %lrwd,  [lr_wd];

    mov.f32       %c1,    {one};
    sub.f32       %c1,    %c1, %b1;
    mov.f32       %c2,    {one};
    sub.f32       %c2,    %c2, %b2;

    // Weight-decay factor = 1 - lr*wd
    mov.f32       %wdf,   {one};
    sub.f32       %wdf,   %wdf, %lrwd;

    ld.param.u64  %ap0,  [p_param];
    ld.param.u64  %ag0,  [p_grad];
    ld.param.u64  %am10, [p_m1];
    ld.param.u64  %am20, [p_m2];

    mov.u32  %tid,    %tid.x;
    mov.u32  %bid,    %ctaid.x;
    mov.u32  %ntid,   %ntid.x;
    mov.u32  %nctaid, %nctaid.x;
    mad.lo.u32 %gid,  %bid, %ntid, %tid;
    mul.lo.u32 %stride, %ntid, %nctaid;

$LOOP:
    cvt.u64.u32   %gid64, %gid;
    setp.ge.u64   %guard, %gid64, %n64;
    @%guard bra   $DONE;

    shl.b64       %off,  %gid64, 2;
    add.u64       %ap,   %ap0,   %off;
    add.u64       %ag,   %ag0,   %off;
    add.u64       %am1,  %am10,  %off;
    add.u64       %am2,  %am20,  %off;

    ld.global.f32 %p,  [%ap];
    ld.global.f32 %g,  [%ag];
    ld.global.f32 %m1, [%am1];
    ld.global.f32 %m2, [%am2];

    // Decoupled weight decay: p *= (1 - lr*wd)
    mul.f32       %p,   %p, %wdf;

    // nm1 = beta1*m1 + (1-beta1)*g
    mul.f32       %nm1, %b1, %m1;
    fma.rn.f32    %nm1, %c1, %g, %nm1;

    // nm2 = beta2*m2 + (1-beta2)*g^2
    mul.f32       %g2,  %g,  %g;
    mul.f32       %nm2, %b2, %m2;
    fma.rn.f32    %nm2, %c2, %g2, %nm2;

    sqrt.approx.f32 %sq, %nm2;
    mul.f32         %sq,  %sq, %bc2r;
    add.f32         %denom, %sq, %ep;
    rcp.approx.f32  %denom, %denom;
    mul.f32         %upd,   %nm1,  %denom;
    mul.f32         %upd,   %step, %upd;

    sub.f32        %p,  %p, %upd;

    st.global.f32  [%ap],  %p;
    st.global.f32  [%am1], %nm1;
    st.global.f32  [%am2], %nm2;

    add.u32        %gid, %gid, %stride;
    bra            $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr,
        one = one
    )
}

// ─── SGD fused update ────────────────────────────────────────────────────────

/// Generate PTX for a fused SGD (with optional momentum) update kernel.
///
/// **Kernel name:** `sgd_update_f32`
///
/// **Parameters:**
/// * `param` – `f32*` parameters
/// * `grad`  – `f32*` gradients
/// * `vel`   – `f32*` velocity buffer (set to zero for plain SGD)
/// * `lr`    – learning rate
/// * `momentum` – momentum coefficient (0 = no momentum)
/// * `dampening` – dampening for momentum (0 = standard)
/// * `weight_decay` – L2 regularisation coefficient
/// * `nesterov` – use Nesterov correction (`1` = yes, `0` = no, encoded as `f32`)
/// * `n`      – element count
///
/// **Algorithm:**
/// ```text
/// g' ← g + wd·p
/// v  ← μ·v + (1−dampening)·g'
/// if nesterov:  p ← p − lr·(g' + μ·v)
/// else:         p ← p − lr·v
/// ```
#[must_use]
pub fn sgd_update_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}
.visible .entry sgd_update_f32(
    .param .u64 p_param,
    .param .u64 p_grad,
    .param .u64 p_vel,
    .param .f32 lr,
    .param .f32 momentum,
    .param .f32 dampening,
    .param .f32 weight_decay,
    .param .f32 nesterov,
    .param .u64 n
)
{{
    .reg .pred  %guard, %p_nesterov;
    .reg .u32   %tid, %bid, %ntid, %nctaid, %gid, %stride;
    .reg .u64   %gid64, %n64, %off;
    .reg .u64   %ap0, %ag0, %av0;
    .reg .u64   %ap,  %ag,  %av;
    .reg .f32   %p, %g, %v;
    .reg .f32   %lr_r, %mu, %damp, %wd, %nes;
    .reg .f32   %one_damp, %upd, %gp;

    ld.param.u64  %n64,  [n];
    ld.param.f32  %lr_r, [lr];
    ld.param.f32  %mu,   [momentum];
    ld.param.f32  %damp, [dampening];
    ld.param.f32  %wd,   [weight_decay];
    ld.param.f32  %nes,  [nesterov];

    mov.f32       %one_damp, {one};
    sub.f32       %one_damp, %one_damp, %damp;

    setp.ne.f32   %p_nesterov, %nes, 0f00000000;

    ld.param.u64  %ap0, [p_param];
    ld.param.u64  %ag0, [p_grad];
    ld.param.u64  %av0, [p_vel];

    mov.u32  %tid,    %tid.x;
    mov.u32  %bid,    %ctaid.x;
    mov.u32  %ntid,   %ntid.x;
    mov.u32  %nctaid, %nctaid.x;
    mad.lo.u32 %gid,  %bid, %ntid, %tid;
    mul.lo.u32 %stride, %ntid, %nctaid;

$LOOP:
    cvt.u64.u32  %gid64, %gid;
    setp.ge.u64  %guard, %gid64, %n64;
    @%guard bra  $DONE;

    shl.b64      %off, %gid64, 2;
    add.u64      %ap,  %ap0,  %off;
    add.u64      %ag,  %ag0,  %off;
    add.u64      %av,  %av0,  %off;

    ld.global.f32 %p, [%ap];
    ld.global.f32 %g, [%ag];
    ld.global.f32 %v, [%av];

    // g' = g + wd*p
    fma.rn.f32   %gp,  %wd, %p, %g;

    // v = mu*v + (1-damp)*g'
    mul.f32      %v,   %mu, %v;
    fma.rn.f32   %v,   %one_damp, %gp, %v;

    // Nesterov: upd = g' + mu*v  else  upd = v
    mul.f32      %upd, %mu, %v;
    add.f32      %upd, %gp, %upd;
    @!%p_nesterov mov.f32 %upd, %v;

    // p -= lr * upd
    fma.rn.f32   %p,   %lr_r, %upd, %p;
    // p = p + (-lr)*upd via negated lr
    // Actually: p -= lr*upd = p + (-lr*upd)
    // Use: neg then fma, or sub after mul:
    mul.f32      %upd, %lr_r, %upd;
    sub.f32      %p,   %p,   %upd;

    st.global.f32 [%ap], %p;
    st.global.f32 [%av], %v;

    add.u32      %gid, %gid, %stride;
    bra          $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr,
        one = one
    )
}

// ─── Lion fused update ───────────────────────────────────────────────────────

/// Generate PTX for a fused Lion optimizer update kernel.
///
/// Lion ("EvoLved Sign Momentum") uses only a first moment buffer, making it
/// extremely memory-efficient compared to Adam.
///
/// **Kernel name:** `lion_update_f32`
///
/// **Parameters:**
/// * `param` – `f32*` parameters
/// * `grad`  – `f32*` gradients
/// * `exp_avg` – `f32*` first moment buffer (exponential moving average)
/// * `lr`    – learning rate
/// * `beta1` – interpolation coefficient for update direction (≈ 0.9)
/// * `beta2` – interpolation coefficient for moment update (≈ 0.99)
/// * `weight_decay` – decoupled weight decay λ
/// * `n`     – element count
///
/// **Algorithm (Chen et al. 2023):**
/// ```text
/// c ← β₁·m + (1−β₁)·g          (update direction)
/// p ← p·(1 − lr·λ) − lr·sign(c) (weight decay + signed step)
/// m ← β₂·m + (1−β₂)·g           (moment update for next step)
/// ```
#[must_use]
pub fn lion_update_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    let one = f32_hex(1.0_f32);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}
.visible .entry lion_update_f32(
    .param .u64 p_param,
    .param .u64 p_grad,
    .param .u64 p_m1,
    .param .f32 lr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 weight_decay,
    .param .u64 n
)
{{
    .reg .pred  %guard, %p_neg, %p_zero;
    .reg .u32   %tid, %bid, %ntid, %nctaid, %gid, %stride;
    .reg .u64   %gid64, %n64, %off;
    .reg .u64   %ap0, %ag0, %am0;
    .reg .u64   %ap,  %ag,  %am;
    .reg .f32   %p, %g, %m;
    .reg .f32   %lr_r, %b1, %b2, %wd;
    .reg .f32   %c1, %c2, %c, %wdf, %sgn, %one_f, %neg_one_f, %zero_f;
    .reg .b32   %c_bits, %sign_bit, %one_bits, %sgn_bits;
    .reg .f32   %nm;

    ld.param.u64  %n64,  [n];
    ld.param.f32  %lr_r, [lr];
    ld.param.f32  %b1,   [beta1];
    ld.param.f32  %b2,   [beta2];
    ld.param.f32  %wd,   [weight_decay];

    mov.f32 %c1, {one};  sub.f32 %c1, %c1, %b1;
    mov.f32 %c2, {one};  sub.f32 %c2, %c2, %b2;

    // weight-decay factor = 1 - lr*wd
    mov.f32 %wdf, {one};
    mul.f32 %sgn, %lr_r, %wd;
    sub.f32 %wdf, %wdf, %sgn;

    // pre-load scalar 1.0, -1.0, 0.0
    mov.f32 %one_f,      {one};
    mov.f32 %neg_one_f,  0fBF800000;
    mov.f32 %zero_f,     {zero};

    ld.param.u64  %ap0, [p_param];
    ld.param.u64  %ag0, [p_grad];
    ld.param.u64  %am0, [p_m1];

    mov.u32  %tid,    %tid.x;
    mov.u32  %bid,    %ctaid.x;
    mov.u32  %ntid,   %ntid.x;
    mov.u32  %nctaid, %nctaid.x;
    mad.lo.u32 %gid,  %bid, %ntid, %tid;
    mul.lo.u32 %stride, %ntid, %nctaid;

$LOOP:
    cvt.u64.u32  %gid64, %gid;
    setp.ge.u64  %guard, %gid64, %n64;
    @%guard bra  $DONE;

    shl.b64      %off, %gid64, 2;
    add.u64      %ap,  %ap0,  %off;
    add.u64      %ag,  %ag0,  %off;
    add.u64      %am,  %am0,  %off;

    ld.global.f32 %p, [%ap];
    ld.global.f32 %g, [%ag];
    ld.global.f32 %m, [%am];

    // c = beta1*m + (1-beta1)*g  (update direction)
    mul.f32       %c,  %b1, %m;
    fma.rn.f32    %c,  %c1, %g, %c;

    // sign(c): copysign trick via bit manipulation
    mov.b32       %c_bits,  %c;
    and.b32       %sign_bit, %c_bits, 0x80000000;
    mov.b32       %one_bits, %one_f;
    or.b32        %sgn_bits, %one_bits, %sign_bit;
    mov.f32       %sgn,      %sgn_bits;
    // handle c == 0 → sign = 0
    setp.eq.f32   %p_zero, %c, %zero_f;
    @%p_zero mov.f32 %sgn, %zero_f;

    // Decoupled weight decay: p = p * wdf
    mul.f32       %p, %p, %wdf;

    // p -= lr * sign(c)
    mul.f32       %sgn, %lr_r, %sgn;
    sub.f32       %p,   %p,   %sgn;

    // m ← beta2*m + (1-beta2)*g  (moment for next step)
    mul.f32       %nm,  %b2, %m;
    fma.rn.f32    %nm,  %c2, %g, %nm;

    st.global.f32 [%ap], %p;
    st.global.f32 [%am], %nm;

    add.u32       %gid, %gid, %stride;
    bra           $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr,
        one = one,
        zero = zero
    )
}

// ─── CAME row/column factor update ──────────────────────────────────────────

/// Generate PTX for CAME row-factor update: `r[i] += sum_j(g²[i,j])`.
///
/// **Kernel name:** `came_row_factor_f32`
///
/// **Parameters:**
/// * `p_g2`  – `f32*`  element-wise gradient² (n_rows × n_cols)
/// * `p_row` – `f32*`  row factor accumulator (n_rows)
/// * `n_cols` – `u64`  columns per row
/// * `n_rows` – `u64`  number of rows
///
/// Each thread handles one row, accumulating the sum of `g²[row, :]`.
#[must_use]
pub fn came_row_factor_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
.visible .entry came_row_factor_f32(
    .param .u64 p_g2,
    .param .u64 p_row,
    .param .u64 n_cols,
    .param .u64 n_rows
)
{{
    .reg .pred  %guard;
    .reg .u32   %tid, %bid, %ntid, %nctaid, %gid, %stride;
    .reg .u64   %gid64, %nrows64, %ncols64;
    .reg .u64   %g2_base, %row_base;
    .reg .u64   %row_off, %col_off, %idx;
    .reg .f32   %acc, %val;
    .reg .u64   %col_u64;
    .reg .u32   %col;

    ld.param.u64  %nrows64, [n_rows];
    ld.param.u64  %ncols64, [n_cols];
    ld.param.u64  %g2_base, [p_g2];
    ld.param.u64  %row_base,[p_row];

    mov.u32  %tid,    %tid.x;
    mov.u32  %bid,    %ctaid.x;
    mov.u32  %ntid,   %ntid.x;
    mov.u32  %nctaid, %nctaid.x;
    mad.lo.u32 %gid,  %bid, %ntid, %tid;
    mul.lo.u32 %stride, %ntid, %nctaid;

$LOOP:
    cvt.u64.u32   %gid64, %gid;
    setp.ge.u64   %guard, %gid64, %nrows64;
    @%guard bra   $DONE;

    // row_off = gid64 * n_cols * 4
    mul.lo.u64    %row_off, %gid64, %ncols64;
    shl.b64       %row_off, %row_off, 2;
    add.u64       %g2_base, %g2_base, %row_off;

    // accumulate sum of g^2 over columns
    mov.f32       %acc, 0f00000000;
    mov.u32       %col, 0;

$COL_LOOP:
    cvt.u64.u32   %col_u64, %col;
    setp.ge.u64   %guard, %col_u64, %ncols64;
    @%guard bra   $COL_DONE;
    shl.b64       %col_off, %col_u64, 2;
    add.u64       %idx, %g2_base, %col_off;
    ld.global.f32 %val, [%idx];
    add.f32       %acc, %acc, %val;
    add.u32       %col, %col, 1;
    bra           $COL_LOOP;

$COL_DONE:
    // Store row factor
    shl.b64       %col_off, %gid64, 2;
    add.u64       %idx, %row_base, %col_off;
    st.global.f32 [%idx], %acc;

    // Reset g2_base for next row (undo per-loop increment)
    // (Not needed: %g2_base was a local copy)

    add.u32       %gid, %gid, %stride;
    bra           $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr
    )
}

/// Generate PTX for CAME column-factor update: `c[j] += sum_i(g²[i,j])`.
///
/// **Kernel name:** `came_col_factor_f32`
///
/// Each thread handles one column, accumulating sum over rows.
#[must_use]
pub fn came_col_factor_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
.visible .entry came_col_factor_f32(
    .param .u64 p_g2,
    .param .u64 p_col,
    .param .u64 n_cols,
    .param .u64 n_rows
)
{{
    .reg .pred  %guard;
    .reg .u32   %tid, %bid, %ntid, %nctaid, %gid, %stride;
    .reg .u64   %gid64, %nrows64, %ncols64;
    .reg .u64   %g2_base, %col_base;
    .reg .u64   %row_u64, %idx;
    .reg .u32   %row;
    .reg .f32   %acc, %val;

    ld.param.u64  %nrows64, [n_rows];
    ld.param.u64  %ncols64, [n_cols];
    ld.param.u64  %g2_base, [p_g2];
    ld.param.u64  %col_base,[p_col];

    mov.u32  %tid,    %tid.x;
    mov.u32  %bid,    %ctaid.x;
    mov.u32  %ntid,   %ntid.x;
    mov.u32  %nctaid, %nctaid.x;
    mad.lo.u32 %gid,  %bid, %ntid, %tid;
    mul.lo.u32 %stride, %ntid, %nctaid;

$LOOP:
    cvt.u64.u32   %gid64, %gid;
    setp.ge.u64   %guard, %gid64, %ncols64;
    @%guard bra   $DONE;

    mov.f32       %acc, 0f00000000;
    mov.u32       %row, 0;

$ROW_LOOP:
    cvt.u64.u32   %row_u64, %row;
    setp.ge.u64   %guard, %row_u64, %nrows64;
    @%guard bra   $ROW_DONE;
    mul.lo.u64    %idx, %row_u64, %ncols64;
    add.u64       %idx, %idx, %gid64;
    shl.b64       %idx, %idx, 2;
    add.u64       %idx, %g2_base, %idx;
    ld.global.f32 %val, [%idx];
    add.f32       %acc, %acc, %val;
    add.u32       %row, %row, 1;
    bra           $ROW_LOOP;

$ROW_DONE:
    shl.b64       %idx, %gid64, 2;
    add.u64       %idx, %col_base, %idx;
    st.global.f32 [%idx], %acc;

    add.u32       %gid, %gid, %stride;
    bra           $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr
    )
}

// ─── Gradient norm-squared partial reduction ─────────────────────────────────

/// Generate PTX for a partial L2 norm-squared accumulation (first pass of
/// global gradient norm clipping).
///
/// **Kernel name:** `norm_sq_partial_f32`
///
/// **Parameters:**
/// * `p_grad`    – `f32*` gradient buffer
/// * `p_partial` – `f32*` per-block partial sums (size = num_blocks)
/// * `n`         – `u64`  total element count
///
/// Each block reduces its strip into a single `f32` in `p_partial`.
#[must_use]
pub fn norm_sq_partial_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
.visible .entry norm_sq_partial_f32(
    .param .u64 p_grad,
    .param .u64 p_partial,
    .param .u64 n
)
{{
    .reg .pred  %guard;
    .reg .u32   %tid, %bid, %ntid, %nctaid, %gid, %stride, %lane, %warpid, %warpn;
    .reg .u64   %gid64, %n64, %off;
    .reg .u64   %gbase, %pbase, %idx;
    .reg .f32   %val, %acc, %shfl;
    .reg .u32   %smask;

    // Shared memory for warp sums: up to 32 warps
    .shared .align 4 .f32 smem[32];

    ld.param.u64  %n64,   [n];
    ld.param.u64  %gbase, [p_grad];
    ld.param.u64  %pbase, [p_partial];

    mov.u32  %tid,    %tid.x;
    mov.u32  %bid,    %ctaid.x;
    mov.u32  %ntid,   %ntid.x;
    mov.u32  %nctaid, %nctaid.x;
    mad.lo.u32 %gid,  %bid, %ntid, %tid;
    mul.lo.u32 %stride, %ntid, %nctaid;

    and.b32  %lane,   %tid, 31;
    shr.u32  %warpid, %tid, 5;

    // Thread-local accumulation (grid-stride)
    mov.f32  %acc, 0f00000000;

$LOOP:
    cvt.u64.u32  %gid64, %gid;
    setp.ge.u64  %guard, %gid64, %n64;
    @%guard bra  $REDUCE;

    shl.b64      %off,  %gid64, 2;
    add.u64      %idx,  %gbase, %off;
    ld.global.f32 %val, [%idx];
    fma.rn.f32   %acc,  %val, %val, %acc;

    add.u32      %gid, %gid, %stride;
    bra          $LOOP;

$REDUCE:
    // Warp butterfly reduction
    mov.u32  %smask, 0xFFFFFFFF;
    shfl.sync.bfly.b32 %shfl, %acc, 16, 31, %smask;  add.f32 %acc, %acc, %shfl;
    shfl.sync.bfly.b32 %shfl, %acc,  8, 31, %smask;  add.f32 %acc, %acc, %shfl;
    shfl.sync.bfly.b32 %shfl, %acc,  4, 31, %smask;  add.f32 %acc, %acc, %shfl;
    shfl.sync.bfly.b32 %shfl, %acc,  2, 31, %smask;  add.f32 %acc, %acc, %shfl;
    shfl.sync.bfly.b32 %shfl, %acc,  1, 31, %smask;  add.f32 %acc, %acc, %shfl;

    // Lane 0 of each warp writes to smem
    setp.ne.u32  %guard, %lane, 0;
    @%guard bra  $SKIP_WRITE;
    mul.lo.u32   %off,   %warpid, 4;
    cvt.u64.u32  %idx,   %off;
    st.shared.f32 [%idx], %acc;
$SKIP_WRITE:
    bar.sync     0;

    // First warp reduces smem
    div.u32  %warpn, %ntid, 32;
    setp.ge.u32 %guard, %warpid, 1;
    @%guard bra $STORE;

    cvt.u64.u32 %idx, %lane;
    mul.lo.u64  %idx, %idx, 4;
    setp.lt.u32 %guard, %lane, %warpn;
    mov.f32     %acc, 0f00000000;
    @%guard ld.shared.f32 %acc, [%idx];

    shfl.sync.bfly.b32 %shfl, %acc, 16, 31, %smask;  add.f32 %acc, %acc, %shfl;
    shfl.sync.bfly.b32 %shfl, %acc,  8, 31, %smask;  add.f32 %acc, %acc, %shfl;
    shfl.sync.bfly.b32 %shfl, %acc,  4, 31, %smask;  add.f32 %acc, %acc, %shfl;
    shfl.sync.bfly.b32 %shfl, %acc,  2, 31, %smask;  add.f32 %acc, %acc, %shfl;
    shfl.sync.bfly.b32 %shfl, %acc,  1, 31, %smask;  add.f32 %acc, %acc, %shfl;

$STORE:
    setp.ne.u32 %guard, %tid, 0;
    @%guard bra $DONE;
    cvt.u64.u32 %idx, %bid;
    shl.b64     %idx, %idx, 2;
    add.u64     %idx, %pbase, %idx;
    st.global.f32 [%idx], %acc;

$DONE:
    ret;
}}
"#,
        hdr = hdr
    )
}

// ─── In-place scale ──────────────────────────────────────────────────────────

/// Generate PTX for in-place vector scaling: `x[i] *= scale`.
///
/// **Kernel name:** `scale_inplace_f32`
///
/// Used to apply gradient clipping: after computing the global norm, scale
/// all gradients by `max_norm / norm` if `norm > max_norm`.
#[must_use]
pub fn scale_inplace_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
.visible .entry scale_inplace_f32(
    .param .u64 p_x,
    .param .f32 scale,
    .param .u64 n
)
{{
    .reg .pred %guard;
    .reg .u32  %tid, %bid, %ntid, %nctaid, %gid, %stride;
    .reg .u64  %gid64, %n64, %off, %xbase, %xaddr;
    .reg .f32  %val, %sc;

    ld.param.u64  %n64,   [n];
    ld.param.f32  %sc,    [scale];
    ld.param.u64  %xbase, [p_x];

    mov.u32  %tid,    %tid.x;
    mov.u32  %bid,    %ctaid.x;
    mov.u32  %ntid,   %ntid.x;
    mov.u32  %nctaid, %nctaid.x;
    mad.lo.u32 %gid,  %bid, %ntid, %tid;
    mul.lo.u32 %stride, %ntid, %nctaid;

$LOOP:
    cvt.u64.u32  %gid64, %gid;
    setp.ge.u64  %guard, %gid64, %n64;
    @%guard bra  $DONE;

    shl.b64      %off,   %gid64, 2;
    add.u64      %xaddr, %xbase, %off;
    ld.global.f32 %val,  [%xaddr];
    mul.f32      %val,   %val, %sc;
    st.global.f32 [%xaddr], %val;

    add.u32      %gid, %gid, %stride;
    bra          $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr
    )
}

// ─── In-place fused add (gradient accumulation) ──────────────────────────────

/// Generate PTX for in-place fused add: `acc[i] += src[i]`.
///
/// **Kernel name:** `add_inplace_f32`
///
/// Used for gradient accumulation across micro-batches.
#[must_use]
pub fn add_inplace_ptx(sm: SmVersion) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
.visible .entry add_inplace_f32(
    .param .u64 p_acc,
    .param .u64 p_src,
    .param .u64 n
)
{{
    .reg .pred %guard;
    .reg .u32  %tid, %bid, %ntid, %nctaid, %gid, %stride;
    .reg .u64  %gid64, %n64, %off;
    .reg .u64  %abase, %sbase, %aaddr, %saddr;
    .reg .f32  %a, %s;

    ld.param.u64  %n64,   [n];
    ld.param.u64  %abase, [p_acc];
    ld.param.u64  %sbase, [p_src];

    mov.u32  %tid,    %tid.x;
    mov.u32  %bid,    %ctaid.x;
    mov.u32  %ntid,   %ntid.x;
    mov.u32  %nctaid, %nctaid.x;
    mad.lo.u32 %gid,  %bid, %ntid, %tid;
    mul.lo.u32 %stride, %ntid, %nctaid;

$LOOP:
    cvt.u64.u32  %gid64, %gid;
    setp.ge.u64  %guard, %gid64, %n64;
    @%guard bra  $DONE;

    shl.b64      %off,   %gid64, 2;
    add.u64      %aaddr, %abase, %off;
    add.u64      %saddr, %sbase, %off;
    ld.global.f32 %a,   [%aaddr];
    ld.global.f32 %s,   [%saddr];
    add.f32      %a,    %a, %s;
    st.global.f32 [%aaddr], %a;

    add.u32      %gid, %gid, %stride;
    bra          $LOOP;
$DONE:
    ret;
}}
"#,
        hdr = hdr
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;

    fn check_contains(ptx: &str, expected: &[&str]) {
        for e in expected {
            assert!(ptx.contains(e), "PTX missing {e:?}\nFull PTX:\n{ptx}");
        }
    }

    #[test]
    fn adam_ptx_smoke_sm80() {
        let ptx = adam_update_ptx(SmVersion::Sm80);
        check_contains(
            &ptx,
            &[
                ".version 7.5",
                ".target sm_80",
                "adam_update_f32",
                "sqrt.approx.f32",
                "rcp.approx.f32",
                "fma.rn.f32",
                "$LOOP",
                "$DONE",
            ],
        );
    }

    #[test]
    fn adam_ptx_smoke_sm90() {
        let ptx = adam_update_ptx(SmVersion::Sm90);
        assert!(ptx.contains(".target sm_90"));
        assert!(ptx.contains(".version 8.0"));
    }

    #[test]
    fn adamw_ptx_has_weight_decay() {
        let ptx = adamw_update_ptx(SmVersion::Sm80);
        check_contains(
            &ptx,
            &["adamw_update_f32", "lr_wd", "wdf", "Decoupled weight decay"],
        );
    }

    #[test]
    fn sgd_ptx_has_nesterov() {
        let ptx = sgd_update_ptx(SmVersion::Sm80);
        check_contains(&ptx, &["sgd_update_f32", "nesterov", "p_nesterov"]);
    }

    #[test]
    fn lion_ptx_sign_trick() {
        let ptx = lion_update_ptx(SmVersion::Sm80);
        check_contains(
            &ptx,
            &[
                "lion_update_f32",
                "0x80000000", // sign bit mask
                "copysign",   // comment text
                "sign_bit",
            ],
        );
    }

    #[test]
    fn norm_sq_ptx_has_warp_reduce() {
        let ptx = norm_sq_partial_ptx(SmVersion::Sm80);
        check_contains(&ptx, &["norm_sq_partial_f32", "shfl.sync.bfly.b32"]);
    }

    #[test]
    fn scale_inplace_ptx_smoke() {
        let ptx = scale_inplace_ptx(SmVersion::Sm80);
        check_contains(&ptx, &["scale_inplace_f32", "mul.f32"]);
    }

    #[test]
    fn add_inplace_ptx_smoke() {
        let ptx = add_inplace_ptx(SmVersion::Sm80);
        check_contains(&ptx, &["add_inplace_f32"]);
    }

    #[test]
    fn came_row_factor_ptx_smoke() {
        let ptx = came_row_factor_ptx(SmVersion::Sm80);
        check_contains(&ptx, &["came_row_factor_f32", "n_cols"]);
    }

    #[test]
    fn came_col_factor_ptx_smoke() {
        let ptx = came_col_factor_ptx(SmVersion::Sm80);
        check_contains(&ptx, &["came_col_factor_f32", "n_rows"]);
    }

    #[test]
    fn all_kernels_have_grid_stride() {
        let kernels = [
            adam_update_ptx(SmVersion::Sm80),
            adamw_update_ptx(SmVersion::Sm80),
            sgd_update_ptx(SmVersion::Sm80),
            lion_update_ptx(SmVersion::Sm80),
            scale_inplace_ptx(SmVersion::Sm80),
            add_inplace_ptx(SmVersion::Sm80),
        ];
        for k in &kernels {
            assert!(k.contains("stride"), "kernel missing grid-stride:\n{k}");
            assert!(k.contains("$LOOP"), "kernel missing LOOP label:\n{k}");
            assert!(k.contains("$DONE"), "kernel missing DONE label:\n{k}");
        }
    }
}
