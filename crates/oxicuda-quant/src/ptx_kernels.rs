//! # PTX Kernel Sources for GPU-Accelerated Quantization
//!
//! These PTX snippets are generated at runtime and compiled via
//! `cuModuleLoadData` when a CUDA device is available.  They are
//! pure-string constants: no compile-time GPU toolchain is needed.
//!
//! ## Kernels
//!
//! | Function | Description |
//! |----------|-------------|
//! | `fake_quant_ptx` | Fake-quantize f32 in-place with round-and-clip (STE) |
//! | `int8_quant_ptx` | Per-tensor symmetric quantize f32 → i8 |
//! | `int8_dequant_ptx` | Dequantize i8 → f32 with scale |
//! | `nf4_dequant_ptx` | NF4 lookup-table dequantize (packed u8→f32) |
//! | `prune_mask_ptx` | Apply binary sparsity mask: `w *= mask` |

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Format an f32 constant as the IEEE 754 PTX hex literal `0fXXXXXXXX`.
#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0f{:08X}", v.to_bits())
}

/// Return the PTX `.version` / `.target` header for the given SM version.
#[must_use]
pub fn ptx_header(sm: u32) -> String {
    let ptx_ver = if sm >= 100 {
        "8.7"
    } else if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    format!(".version {ptx_ver}\n.target sm_{sm}\n.address_size 64\n")
}

// ─── Fake Quantize ────────────────────────────────────────────────────────────

/// PTX kernel: fake-quantize f32 in-place.
///
/// Each element is rounded to the nearest quantization level and clamped
/// within `[q_min, q_max]`, then scaled back:
/// ```text
/// x_q = round(x / scale)
/// x_q = clamp(x_q, q_min, q_max)
/// x   = x_q * scale
/// ```
/// The straight-through estimator (STE) gradient is implicit: the
/// rounding is non-differentiable but we pass through the gradient
/// unchanged (handled by the training framework, not this kernel).
#[must_use]
pub fn fake_quant_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    // scale = 1/scale_inv for division efficiency
    format!(
        r#"{hdr}
// fake_quant_inplace(float* data, int n, float scale, float q_min, float q_max)
.visible .entry fake_quant_inplace(
    .param .u64 p_data,
    .param .s32 p_n,
    .param .f32 p_scale,
    .param .f32 p_qmin,
    .param .f32 p_qmax
)
{{
    .reg .u64  addr;
    .reg .s32  tid, n, stride, idx;
    .reg .f32  x, scale, q_min, q_max, xq, one;
    .reg .pred done;

    ld.param.u64  addr,  [p_data];
    ld.param.s32  n,     [p_n];
    ld.param.f32  scale, [p_scale];
    ld.param.f32  q_min, [p_qmin];
    ld.param.f32  q_max, [p_qmax];

    // grid-stride loop
    mov.u32        %r0, %ctaid.x;
    mov.u32        %r1, %ntid.x;
    mov.u32        %r2, %tid.x;
    mad.lo.s32     idx, %r0, %r1, %r2;
    mov.u32        %r3, %nctaid.x;
    mul.lo.s32     stride, %r3, %r1;
    mov.f32        one, {one};

$LOOP:
    setp.ge.s32    done, idx, n;
    @done bra $DONE;

    // load
    cvt.u64.s32    %rd0, idx;
    mul.wide.s32   %rd1, idx, 4;
    add.u64        %rd2, addr, %rd1;
    ld.global.f32  x, [%rd2];

    // x_q = round(x / scale)
    div.rn.f32     xq, x, scale;
    cvt.rni.f32.f32 xq, xq;      // round to nearest integer

    // clamp
    max.f32        xq, xq, q_min;
    min.f32        xq, xq, q_max;

    // scale back
    mul.f32        x, xq, scale;
    st.global.f32  [%rd2], x;

    add.s32        idx, idx, stride;
    bra $LOOP;
$DONE:
    ret;
}}
"#,
        one = f32_hex(1.0_f32)
    )
}

// ─── INT8 Quantize ───────────────────────────────────────────────────────────

/// PTX kernel: symmetric INT8 quantization, f32 → i8.
///
/// `out[i] = clamp(round(in[i] / scale), -127, 127)`
#[must_use]
pub fn int8_quant_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// int8_quant(float* in, int8_t* out, int n, float scale)
.visible .entry int8_quant(
    .param .u64 p_in,
    .param .u64 p_out,
    .param .s32 p_n,
    .param .f32 p_scale
)
{{
    .reg .u64  ain, aout;
    .reg .s32  idx, stride, n;
    .reg .f32  x, scale, xq;
    .reg .s32  iq;
    .reg .pred done;

    ld.param.u64  ain,   [p_in];
    ld.param.u64  aout,  [p_out];
    ld.param.s32  n,     [p_n];
    ld.param.f32  scale, [p_scale];

    mov.u32    %r0, %ctaid.x;
    mov.u32    %r1, %ntid.x;
    mov.u32    %r2, %tid.x;
    mad.lo.s32 idx, %r0, %r1, %r2;
    mov.u32    %r3, %nctaid.x;
    mul.lo.s32 stride, %r3, %r1;

$LOOP:
    setp.ge.s32  done, idx, n;
    @done bra $DONE;

    cvt.u64.s32  %rd0, idx;
    mul.wide.s32 %rd1, idx, 4;
    add.u64      %rd2, ain, %rd1;
    ld.global.f32 x, [%rd2];

    div.rn.f32   xq, x, scale;
    cvt.rni.f32.f32 xq, xq;

    // clamp to [-127, 127]
    max.f32      xq, xq, {neg127};
    min.f32      xq, xq, {pos127};
    cvt.rni.s32.f32 iq, xq;

    // store as i8
    add.u64      %rd3, aout, %rd0;
    st.global.s8 [%rd3], iq;

    add.s32      idx, idx, stride;
    bra $LOOP;
$DONE:
    ret;
}}
"#,
        neg127 = f32_hex(-127.0_f32),
        pos127 = f32_hex(127.0_f32)
    )
}

// ─── INT8 Dequantize ─────────────────────────────────────────────────────────

/// PTX kernel: dequantize i8 → f32.
///
/// `out[i] = (float)in[i] * scale`
#[must_use]
pub fn int8_dequant_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// int8_dequant(int8_t* in, float* out, int n, float scale)
.visible .entry int8_dequant(
    .param .u64 p_in,
    .param .u64 p_out,
    .param .s32 p_n,
    .param .f32 p_scale
)
{{
    .reg .u64  ain, aout;
    .reg .s32  idx, stride, n, iq;
    .reg .f32  scale, x;
    .reg .pred done;

    ld.param.u64  ain,   [p_in];
    ld.param.u64  aout,  [p_out];
    ld.param.s32  n,     [p_n];
    ld.param.f32  scale, [p_scale];

    mov.u32    %r0, %ctaid.x;
    mov.u32    %r1, %ntid.x;
    mov.u32    %r2, %tid.x;
    mad.lo.s32 idx, %r0, %r1, %r2;
    mov.u32    %r3, %nctaid.x;
    mul.lo.s32 stride, %r3, %r1;

$LOOP:
    setp.ge.s32  done, idx, n;
    @done bra $DONE;

    cvt.u64.s32  %rd0, idx;
    add.u64      %rd1, ain, %rd0;
    ld.global.s8 iq, [%rd1];

    cvt.rn.f32.s32 x, iq;
    mul.f32        x, x, scale;

    mul.wide.s32   %rd2, idx, 4;
    add.u64        %rd3, aout, %rd2;
    st.global.f32  [%rd3], x;

    add.s32  idx, idx, stride;
    bra $LOOP;
$DONE:
    ret;
}}
"#
    )
}

// ─── NF4 Dequantize ──────────────────────────────────────────────────────────

/// PTX kernel: NF4 (NormalFloat4) dequantization.
///
/// Input is packed as two 4-bit indices per byte.
/// Output is f32 using the 16-entry NF4 lookup table stored in shared memory.
///
/// The NF4 lookup table values (from the QLoRA paper, Dettmers et al. 2023):
/// ```text
/// [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911,  0.0,
///   0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0]
/// ```
#[must_use]
pub fn nf4_dequant_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    // The 16 NF4 levels as hex f32 literals
    let lut: [f32; 16] = [
        -1.0,
        -0.696_192_86,
        -0.525_073_05,
        -0.394_917_5,
        -0.284_441_38,
        -0.184_773_43,
        -0.091_050_03,
        0.0,
        0.079_580_3,
        0.160_930_2,
        0.246_112_3,
        0.337_915_24,
        0.440_709_83,
        0.562_617,
        0.722_956_84,
        1.0,
    ];
    let lut_strs: Vec<String> = lut.iter().map(|&v| f32_hex(v)).collect();

    format!(
        r#"{hdr}
// nf4_dequant(uint8_t* packed, float* out, int n_floats, float absmax)
// packed[i] = (hi_nibble << 4) | lo_nibble
// out[2*i+0] = lut[lo_nibble] * absmax
// out[2*i+1] = lut[hi_nibble] * absmax
.visible .entry nf4_dequant(
    .param .u64 p_packed,
    .param .u64 p_out,
    .param .s32 p_n,
    .param .f32 p_absmax
)
{{
    .reg .u64  apk, aout;
    .reg .s32  idx, stride, n, nb;
    .reg .f32  absmax;
    .reg .pred done;
    .reg .b32  packed, lo, hi;
    .reg .f32  vlo, vhi;

    // NF4 lookup table in constant memory (16 × f32)
    .shared .align 4 .f32 lut[16];

    ld.param.u64  apk,    [p_packed];
    ld.param.u64  aout,   [p_out];
    ld.param.s32  n,      [p_n];
    ld.param.f32  absmax, [p_absmax];

    // Thread 0 initialises LUT
    mov.u32 %r0, %tid.x;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra $SKIP_LUT;
    mov.f32 %f0, {l0};  st.shared.f32 [lut+0],  %f0;
    mov.f32 %f0, {l1};  st.shared.f32 [lut+4],  %f0;
    mov.f32 %f0, {l2};  st.shared.f32 [lut+8],  %f0;
    mov.f32 %f0, {l3};  st.shared.f32 [lut+12], %f0;
    mov.f32 %f0, {l4};  st.shared.f32 [lut+16], %f0;
    mov.f32 %f0, {l5};  st.shared.f32 [lut+20], %f0;
    mov.f32 %f0, {l6};  st.shared.f32 [lut+24], %f0;
    mov.f32 %f0, {l7};  st.shared.f32 [lut+28], %f0;
    mov.f32 %f0, {l8};  st.shared.f32 [lut+32], %f0;
    mov.f32 %f0, {l9};  st.shared.f32 [lut+36], %f0;
    mov.f32 %f0, {l10}; st.shared.f32 [lut+40], %f0;
    mov.f32 %f0, {l11}; st.shared.f32 [lut+44], %f0;
    mov.f32 %f0, {l12}; st.shared.f32 [lut+48], %f0;
    mov.f32 %f0, {l13}; st.shared.f32 [lut+52], %f0;
    mov.f32 %f0, {l14}; st.shared.f32 [lut+56], %f0;
    mov.f32 %f0, {l15}; st.shared.f32 [lut+60], %f0;
$SKIP_LUT:
    bar.sync 0;

    mov.u32    %r0, %ctaid.x;
    mov.u32    %r1, %ntid.x;
    mov.u32    %r2, %tid.x;
    mad.lo.s32 idx, %r0, %r1, %r2;   // byte index
    mov.u32    %r3, %nctaid.x;
    mul.lo.s32 stride, %r3, %r1;
    shr.s32    nb, n, 1;              // n_bytes = n_floats / 2

$LOOP:
    setp.ge.s32  done, idx, nb;
    @done bra $DONE;

    // load packed byte
    cvt.u64.s32  %rd0, idx;
    add.u64      %rd1, apk, %rd0;
    ld.global.u8 packed, [%rd1];

    // extract nibbles
    and.b32  lo, packed, 15;
    shr.b32  hi, packed, 4;

    // LUT lookup (byte offset = nibble * 4)
    mul.lo.s32 %r4, lo, 4;
    ld.shared.f32 vlo, [lut + %r4];
    mul.lo.s32 %r5, hi, 4;
    ld.shared.f32 vhi, [lut + %r5];

    // scale by absmax
    mul.f32 vlo, vlo, absmax;
    mul.f32 vhi, vhi, absmax;

    // store two f32s
    mul.lo.s32   %r6, idx, 8;        // byte offset in output
    cvt.u64.s32  %rd2, %r6;
    add.u64      %rd3, aout, %rd2;
    st.global.f32 [%rd3+0], vlo;
    st.global.f32 [%rd3+4], vhi;

    add.s32  idx, idx, stride;
    bra $LOOP;
$DONE:
    ret;
}}
"#,
        l0 = lut_strs[0],
        l1 = lut_strs[1],
        l2 = lut_strs[2],
        l3 = lut_strs[3],
        l4 = lut_strs[4],
        l5 = lut_strs[5],
        l6 = lut_strs[6],
        l7 = lut_strs[7],
        l8 = lut_strs[8],
        l9 = lut_strs[9],
        l10 = lut_strs[10],
        l11 = lut_strs[11],
        l12 = lut_strs[12],
        l13 = lut_strs[13],
        l14 = lut_strs[14],
        l15 = lut_strs[15],
    )
}

// ─── Prune by Mask ───────────────────────────────────────────────────────────

/// PTX kernel: apply binary sparsity mask in-place.
///
/// `weights\[i\] *= mask\[i\]`  (mask\[i\] ∈ {0, 1})
#[must_use]
pub fn prune_mask_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// prune_by_mask(float* weights, uint8_t* mask, int n)
.visible .entry prune_by_mask(
    .param .u64 p_weights,
    .param .u64 p_mask,
    .param .s32 p_n
)
{{
    .reg .u64  aw, am;
    .reg .s32  idx, stride, n, m;
    .reg .f32  w;
    .reg .pred done;

    ld.param.u64  aw,  [p_weights];
    ld.param.u64  am,  [p_mask];
    ld.param.s32  n,   [p_n];

    mov.u32    %r0, %ctaid.x;
    mov.u32    %r1, %ntid.x;
    mov.u32    %r2, %tid.x;
    mad.lo.s32 idx, %r0, %r1, %r2;
    mov.u32    %r3, %nctaid.x;
    mul.lo.s32 stride, %r3, %r1;

$LOOP:
    setp.ge.s32  done, idx, n;
    @done bra $DONE;

    // load mask byte
    cvt.u64.s32  %rd0, idx;
    add.u64      %rd1, am, %rd0;
    ld.global.u8 m, [%rd1];

    // if mask == 0, zero the weight
    setp.ne.s32  %p0, m, 0;
    mul.wide.s32 %rd2, idx, 4;
    add.u64      %rd3, aw, %rd2;
    ld.global.f32 w, [%rd3];
    @!%p0 mov.f32 w, {zero};
    st.global.f32 [%rd3], w;

    add.s32  idx, idx, stride;
    bra $LOOP;
$DONE:
    ret;
}}
"#,
        zero = f32_hex(0.0_f32)
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_hex_zero() {
        assert_eq!(f32_hex(0.0), "0f00000000");
    }

    #[test]
    fn f32_hex_one() {
        assert_eq!(f32_hex(1.0), "0f3F800000");
    }

    #[test]
    fn ptx_header_versions() {
        assert!(ptx_header(75).contains("7.5"));
        assert!(ptx_header(80).contains("8.0"));
        assert!(ptx_header(90).contains("8.4"));
        assert!(ptx_header(100).contains("8.7"));
    }

    #[test]
    fn fake_quant_ptx_contains_rni() {
        let ptx = fake_quant_ptx(80);
        assert!(ptx.contains("cvt.rni"), "should use round-to-nearest-int");
        assert!(ptx.contains("fake_quant_inplace"));
    }

    #[test]
    fn int8_quant_ptx_clamps_to_127() {
        let ptx = int8_quant_ptx(80);
        assert!(ptx.contains("int8_quant"));
        assert!(ptx.contains(f32_hex(127.0).as_str()));
    }

    #[test]
    fn int8_dequant_ptx_has_scale() {
        let ptx = int8_dequant_ptx(80);
        assert!(ptx.contains("int8_dequant"));
        assert!(ptx.contains("mul.f32"));
    }

    #[test]
    fn nf4_dequant_ptx_has_lut() {
        let ptx = nf4_dequant_ptx(80);
        assert!(ptx.contains("nf4_dequant"));
        // LUT entry for -1.0
        assert!(ptx.contains(f32_hex(-1.0).as_str()));
        // LUT entry for +1.0
        assert!(ptx.contains(f32_hex(1.0).as_str()));
    }

    #[test]
    fn prune_mask_ptx_has_zero_store() {
        let ptx = prune_mask_ptx(80);
        assert!(ptx.contains("prune_by_mask"));
        assert!(ptx.contains(f32_hex(0.0).as_str()));
    }

    #[test]
    fn all_kernels_sm90() {
        for sm in [75_u32, 80, 86, 90, 100] {
            assert!(!fake_quant_ptx(sm).is_empty());
            assert!(!int8_quant_ptx(sm).is_empty());
            assert!(!int8_dequant_ptx(sm).is_empty());
            assert!(!nf4_dequant_ptx(sm).is_empty());
            assert!(!prune_mask_ptx(sm).is_empty());
        }
    }
}
