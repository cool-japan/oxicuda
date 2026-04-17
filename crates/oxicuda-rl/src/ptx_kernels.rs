//! PTX kernel sources for GPU-accelerated RL operations.
//!
//! All kernels in this module are expressed as PTX source strings generated at
//! runtime.  They are designed to be loaded via `cuModuleLoadData` and
//! dispatched through `cuLaunchKernel` from the `oxicuda-driver` crate.
//!
//! The helpers follow the same convention as the rest of the OxiCUDA ecosystem:
//! * PTX `.version` / `.target` are derived from the SM version integer.
//! * Grid-stride loops are used for arbitrary input sizes.
//! * IEEE hex literals for constants ensure bit-exact values.

// ─── Helper: PTX preamble ────────────────────────────────────────────────────

/// Generate a PTX file header for the given SM version.
#[must_use]
pub fn ptx_header(sm: u32) -> String {
    let ver = if sm >= 100 {
        "8.7"
    } else if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    format!(".version {ver}\n.target sm_{sm}\n.address_size 64\n\n")
}

/// Convert an `f32` to its IEEE 754 hex literal for PTX (`0f3f800000` style).
#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0f{:08X}", v.to_bits())
}

// ─── TD-error kernel ─────────────────────────────────────────────────────────

/// PTX kernel for computing element-wise TD errors:
/// ```text
/// td_error[i] = reward[i] + gamma * next_v[i] * (1 - done[i]) - v[i]
/// ```
#[must_use]
pub fn td_error_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// td_error: delta = r + γ * next_v * (1-done) - v
// ptr_reward, ptr_next_v, ptr_done (f32 0/1), ptr_v, ptr_out  all n f32
// param: gamma
.visible .entry td_error(
    .param .u64 ptr_reward,
    .param .u64 ptr_next_v,
    .param .u64 ptr_done,
    .param .u64 ptr_v,
    .param .u64 ptr_out,
    .param .u32 n,
    .param .f32 gamma
)
{{
    .reg .u64  %r_addr, %nv_addr, %d_addr, %v_addr, %o_addr;
    .reg .u32  %tid, %ntid, %cid, %ncid, %idx, %n, %stride;
    .reg .f32  %rw, %nv, %dn, %vv, %gam, %tmp, %delta;
    .reg .pred %p;
    .reg .u64  %eaddr;

    ld.param.u64  %r_addr,  [ptr_reward];
    ld.param.u64  %nv_addr, [ptr_next_v];
    ld.param.u64  %d_addr,  [ptr_done];
    ld.param.u64  %v_addr,  [ptr_v];
    ld.param.u64  %o_addr,  [ptr_out];
    ld.param.u32  %n,       [n];
    ld.param.f32  %gam,     [gamma];

    mov.u32 %tid,  %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %cid,  %ctaid.x;
    mov.u32 %ncid, %nctaid.x;
    mad.lo.u32  %idx,    %cid, %ntid, %tid;
    mul.lo.u32  %stride, %ncid, %ntid;

LOOP:
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    mul.wide.u32 %eaddr, %idx, 4;

    add.u64 %eaddr, %r_addr,  %eaddr;
    ld.global.f32 %rw, [%eaddr];

    mul.wide.u32 %eaddr, %idx, 4;
    add.u64 %eaddr, %nv_addr, %eaddr;
    ld.global.f32 %nv, [%eaddr];

    mul.wide.u32 %eaddr, %idx, 4;
    add.u64 %eaddr, %d_addr,  %eaddr;
    ld.global.f32 %dn, [%eaddr];

    mul.wide.u32 %eaddr, %idx, 4;
    add.u64 %eaddr, %v_addr,  %eaddr;
    ld.global.f32 %vv, [%eaddr];

    // delta = r + gamma * nv * (1 - done) - v
    sub.f32       %tmp,   {one},  %dn;      // mask = 1 - done
    fma.rn.f32    %delta, %gam, %nv, {zero}; // delta = γ * nv
    mul.rn.f32    %delta, %delta, %tmp;      // delta *= mask
    add.rn.f32    %delta, %delta, %rw;       // delta += r
    sub.rn.f32    %delta, %delta, %vv;       // delta -= v

    mul.wide.u32 %eaddr, %idx, 4;
    add.u64 %eaddr, %o_addr, %eaddr;
    st.global.f32 [%eaddr], %delta;

    add.u32 %idx, %idx, %stride;
    bra LOOP;
DONE:
    ret;
}}
"#,
        hdr = hdr,
        one = f32_hex(1.0),
        zero = f32_hex(0.0),
    )
}

// ─── Advantage normalisation kernel ──────────────────────────────────────────

/// PTX kernel for normalising a flat advantage buffer in-place:
/// ```text
/// adv[i] = (adv[i] - mean) / (std + eps)
/// ```
/// `mean` and `std` must be pre-computed on the host.
#[must_use]
pub fn normalize_advantages_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// normalize_advantages: adv[i] = (adv[i] - mean) / (std + eps)
.visible .entry normalize_advantages(
    .param .u64 ptr_adv,
    .param .u32 n,
    .param .f32 mean,
    .param .f32 std_eps   // std + eps pre-summed on host
)
{{
    .reg .u64  %addr, %eaddr;
    .reg .u32  %tid, %ntid, %cid, %ncid, %idx, %n, %stride;
    .reg .f32  %val, %mu, %sig;
    .reg .pred %p;

    ld.param.u64  %addr,    [ptr_adv];
    ld.param.u32  %n,       [n];
    ld.param.f32  %mu,      [mean];
    ld.param.f32  %sig,     [std_eps];

    mov.u32 %tid,  %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %cid,  %ctaid.x;
    mov.u32 %ncid, %nctaid.x;
    mad.lo.u32  %idx,    %cid, %ntid, %tid;
    mul.lo.u32  %stride, %ncid, %ntid;

LOOP:
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    mul.wide.u32 %eaddr, %idx, 4;
    add.u64  %eaddr, %addr, %eaddr;
    ld.global.f32 %val, [%eaddr];

    sub.rn.f32 %val, %val, %mu;
    div.rn.f32 %val, %val, %sig;

    st.global.f32 [%eaddr], %val;
    add.u32 %idx, %idx, %stride;
    bra LOOP;
DONE:
    ret;
}}
"#,
        hdr = hdr,
    )
}

// ─── PPO ratio kernel ─────────────────────────────────────────────────────────

/// PTX kernel computing PPO probability ratios and clipped surrogate objective:
/// ```text
/// ratio[i] = exp(log_prob_new[i] - log_prob_old[i])
/// clipped[i] = clip(ratio[i], 1-eps, 1+eps)
/// obj[i] = min(ratio[i]*adv[i], clipped[i]*adv[i])
/// ```
#[must_use]
pub fn ppo_ratio_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// ppo_ratio: ratio, clipped ratio, surrogate objective
.visible .entry ppo_ratio(
    .param .u64 ptr_lp_new,
    .param .u64 ptr_lp_old,
    .param .u64 ptr_adv,
    .param .u64 ptr_ratio,
    .param .u64 ptr_obj,
    .param .u32 n,
    .param .f32 clip_eps
)
{{
    .reg .u64  %a0, %a1, %a2, %ar, %ao, %ea;
    .reg .u32  %tid, %ntid, %cid, %ncid, %idx, %n, %stride;
    .reg .f32  %lpn, %lpo, %adv, %ratio, %clip, %lo, %hi, %obj, %tmp;
    .reg .pred %p, %q;

    ld.param.u64  %a0,  [ptr_lp_new];
    ld.param.u64  %a1,  [ptr_lp_old];
    ld.param.u64  %a2,  [ptr_adv];
    ld.param.u64  %ar,  [ptr_ratio];
    ld.param.u64  %ao,  [ptr_obj];
    ld.param.u32  %n,   [n];
    ld.param.f32  %tmp, [clip_eps];

    // lo = 1 - eps, hi = 1 + eps
    sub.rn.f32 %lo, {one}, %tmp;
    add.rn.f32 %hi, {one}, %tmp;

    mov.u32 %tid,  %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %cid,  %ctaid.x;
    mov.u32 %ncid, %nctaid.x;
    mad.lo.u32  %idx,    %cid, %ntid, %tid;
    mul.lo.u32  %stride, %ncid, %ntid;

LOOP:
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    mul.wide.u32 %ea, %idx, 4;
    add.u64 %ea, %a0, %ea;
    ld.global.f32 %lpn, [%ea];

    mul.wide.u32 %ea, %idx, 4;
    add.u64 %ea, %a1, %ea;
    ld.global.f32 %lpo, [%ea];

    mul.wide.u32 %ea, %idx, 4;
    add.u64 %ea, %a2, %ea;
    ld.global.f32 %adv, [%ea];

    // ratio = exp(lpn - lpo)
    sub.rn.f32 %ratio, %lpn, %lpo;
    ex2.approx.f32 %ratio, %ratio;   // approximation: e^x ≈ 2^(x/ln2); PTX ex2 = 2^x
    // correct exp: multiply by ln2 first
    mul.rn.f32 %ratio, %ratio, {ln2_inv};   // use mul by 1/ln2 to go from e^x to 2^(x*log2e)

    // redo: exp(d) = ex2(d * log2e)
    sub.rn.f32 %tmp,  %lpn, %lpo;
    mul.rn.f32 %tmp,  %tmp, {log2e};
    ex2.approx.f32 %ratio, %tmp;

    // clipped ratio
    max.f32    %clip, %ratio, %lo;
    min.f32    %clip, %clip,  %hi;

    // surrogate = min(ratio*adv, clip*adv)
    mul.rn.f32 %obj, %ratio, %adv;
    mul.rn.f32 %tmp, %clip,  %adv;
    min.f32    %obj, %obj,   %tmp;

    mul.wide.u32 %ea, %idx, 4;
    add.u64 %ea, %ar, %ea;
    st.global.f32 [%ea], %ratio;

    mul.wide.u32 %ea, %idx, 4;
    add.u64 %ea, %ao, %ea;
    st.global.f32 [%ea], %obj;

    add.u32 %idx, %idx, %stride;
    bra LOOP;
DONE:
    ret;
}}
"#,
        hdr = hdr,
        one = f32_hex(1.0),
        log2e = f32_hex(std::f32::consts::LOG2_E),
        ln2_inv = f32_hex(1.0 / std::f32::consts::LN_2),
    )
}

// ─── Soft Q-value kernel (SAC) ────────────────────────────────────────────────

/// PTX kernel for SAC soft Q-value target:
/// ```text
/// target[i] = reward[i] + gamma * (1-done[i]) * (min_q[i] - alpha * log_pi[i])
/// ```
#[must_use]
pub fn sac_target_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// sac_target: r + γ*(1-d)*(min_q - α*log_π)
.visible .entry sac_target(
    .param .u64 ptr_reward,
    .param .u64 ptr_done,
    .param .u64 ptr_min_q,
    .param .u64 ptr_log_pi,
    .param .u64 ptr_out,
    .param .u32 n,
    .param .f32 gamma,
    .param .f32 alpha
)
{{
    .reg .u64  %a0, %a1, %a2, %a3, %ao, %ea;
    .reg .u32  %tid, %ntid, %cid, %ncid, %idx, %n, %stride;
    .reg .f32  %rw, %dn, %mq, %lp, %gam, %alp, %msk, %tgt;
    .reg .pred %p;

    ld.param.u64 %a0, [ptr_reward];
    ld.param.u64 %a1, [ptr_done];
    ld.param.u64 %a2, [ptr_min_q];
    ld.param.u64 %a3, [ptr_log_pi];
    ld.param.u64 %ao, [ptr_out];
    ld.param.u32 %n,  [n];
    ld.param.f32 %gam,[gamma];
    ld.param.f32 %alp,[alpha];

    mov.u32 %tid,  %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %cid,  %ctaid.x;
    mov.u32 %ncid, %nctaid.x;
    mad.lo.u32 %idx,    %cid, %ntid, %tid;
    mul.lo.u32 %stride, %ncid, %ntid;

LOOP:
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    mul.wide.u32 %ea, %idx, 4;
    add.u64 %ea, %a0, %ea; ld.global.f32 %rw, [%ea];
    mul.wide.u32 %ea, %idx, 4;
    add.u64 %ea, %a1, %ea; ld.global.f32 %dn, [%ea];
    mul.wide.u32 %ea, %idx, 4;
    add.u64 %ea, %a2, %ea; ld.global.f32 %mq, [%ea];
    mul.wide.u32 %ea, %idx, 4;
    add.u64 %ea, %a3, %ea; ld.global.f32 %lp, [%ea];

    // mask = 1 - done
    sub.rn.f32 %msk, {one}, %dn;
    // tgt = min_q - α * log_π
    fma.rn.f32 %tgt, %alp, %lp, {zero};
    sub.rn.f32 %tgt, %mq, %tgt;
    // tgt = r + γ * mask * tgt
    mul.rn.f32 %tgt, %gam, %tgt;
    mul.rn.f32 %tgt, %tgt, %msk;
    add.rn.f32 %tgt, %tgt, %rw;

    mul.wide.u32 %ea, %idx, 4;
    add.u64 %ea, %ao, %ea;
    st.global.f32 [%ea], %tgt;

    add.u32 %idx, %idx, %stride;
    bra LOOP;
DONE:
    ret;
}}
"#,
        hdr = hdr,
        one = f32_hex(1.0),
        zero = f32_hex(0.0),
    )
}

// ─── PER importance-sampling weight kernel ────────────────────────────────────

/// PTX kernel for computing importance-sampling weights in Prioritized
/// Experience Replay:
/// ```text
/// w[i] = (1/(N * P[i]))^β  (unnormalised)
/// w[i] /= max(w)           (normalised: divide by max on host after scan)
/// ```
#[must_use]
pub fn per_is_weight_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// per_is_weight: w[i] = (1 / (n * probs[i]))^beta
.visible .entry per_is_weight(
    .param .u64 ptr_probs,
    .param .u64 ptr_weights,
    .param .u32 n,
    .param .f32 n_f,
    .param .f32 beta
)
{{
    .reg .u64  %a_probs, %a_w, %ea;
    .reg .u32  %tid, %ntid, %cid, %ncid, %idx, %nn, %stride;
    .reg .f32  %prob, %w, %n_f, %beta, %tmp;
    .reg .pred %p;

    ld.param.u64 %a_probs, [ptr_probs];
    ld.param.u64 %a_w,     [ptr_weights];
    ld.param.u32 %nn,      [n];
    ld.param.f32 %n_f,     [n_f];
    ld.param.f32 %beta,    [beta];

    mov.u32 %tid,  %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %cid,  %ctaid.x;
    mov.u32 %ncid, %nctaid.x;
    mad.lo.u32 %idx,    %cid, %ntid, %tid;
    mul.lo.u32 %stride, %ncid, %ntid;

LOOP:
    setp.ge.u32 %p, %idx, %nn;
    @%p bra DONE;

    mul.wide.u32 %ea, %idx, 4;
    add.u64 %ea, %a_probs, %ea;
    ld.global.f32 %prob, [%ea];

    // w = (1 / (n_f * prob))^beta = exp(-beta * ln(n_f * prob))
    mul.rn.f32 %tmp, %n_f, %prob;
    lg2.approx.f32 %tmp, %tmp;          // log2(n_f * prob)
    mul.rn.f32 %tmp, %tmp, {ln2};       // ln(n_f * prob)
    mul.rn.f32 %tmp, %beta, %tmp;       // beta * ln(...)
    neg.f32    %tmp, %tmp;              // -beta * ln(...)
    mul.rn.f32 %tmp, %tmp, {log2e};     // convert to log2 for ex2
    ex2.approx.f32 %w, %tmp;

    mul.wide.u32 %ea, %idx, 4;
    add.u64 %ea, %a_w, %ea;
    st.global.f32 [%ea], %w;

    add.u32 %idx, %idx, %stride;
    bra LOOP;
DONE:
    ret;
}}
"#,
        hdr = hdr,
        ln2 = f32_hex(std::f32::consts::LN_2),
        log2e = f32_hex(std::f32::consts::LOG2_E),
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn td_error_ptx_contains_marker() {
        let s = td_error_ptx(80);
        assert!(s.contains("td_error"), "kernel name missing");
        assert!(s.contains("sm_80"), "sm target missing");
    }

    #[test]
    fn normalize_advantages_ptx_sm75() {
        let s = normalize_advantages_ptx(75);
        assert!(s.contains("normalize_advantages"));
        assert!(s.contains("sm_75"));
    }

    #[test]
    fn ppo_ratio_ptx_contains_ex2() {
        let s = ppo_ratio_ptx(80);
        assert!(s.contains("ex2.approx.f32"), "PPO ratio needs exp via ex2");
    }

    #[test]
    fn sac_target_ptx_sm90() {
        let s = sac_target_ptx(90);
        assert!(s.contains("sac_target"));
        assert!(s.contains("sm_90"));
    }

    #[test]
    fn per_is_weight_ptx_has_lg2() {
        let s = per_is_weight_ptx(80);
        assert!(s.contains("lg2.approx.f32"), "PER weight needs log");
    }

    #[test]
    fn f32_hex_one() {
        // 1.0f32 bits = 0x3F800000
        assert_eq!(f32_hex(1.0), "0f3F800000");
    }

    #[test]
    fn f32_hex_zero() {
        assert_eq!(f32_hex(0.0), "0f00000000");
    }

    #[test]
    fn ptx_header_versions() {
        assert!(ptx_header(75).contains(".version 7.5"));
        assert!(ptx_header(80).contains(".version 8.0"));
        assert!(ptx_header(90).contains(".version 8.4"));
        assert!(ptx_header(100).contains(".version 8.7"));
    }
}
