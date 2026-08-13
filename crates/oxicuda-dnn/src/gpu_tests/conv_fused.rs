//! On-device GPU validation for the fused conv+BN+activation epilogue paths
//! of `oxicuda-dnn`'s `conv_fprop` subsystem (RTX A4000, sm_86).
//!
//! Split out of `conv_fprop.rs` (same coverage class and CPU-oracle
//! methodology) to keep that file under the workspace's 2000-line limit.
//! Covers two related but numerically very different fusion mechanisms:
//!
//! * **`FusedConvBnAct`** (`fused.rs`) — a load/launch-only fragment. The
//!   kernel is a structural skeleton (step-marker comments + `ret`, no
//!   stores), so these tests assert it assembles, JIT-loads, launches and
//!   synchronises fault-free, and that the output buffer is left untouched —
//!   documenting the fragment status honestly rather than green-washing a
//!   wrong numeric result.
//! * **`conv_bn_relu`** (`api.rs`) — the decomposed, numerically-real
//!   fusion: a real `conv_forward` call plus a real BN-affine + activation
//!   epilogue kernel (`conv::fused::apply_fused_bn_activation`). These tests
//!   drive the public `conv::api::conv_bn_relu` entry point end-to-end and
//!   check the result against `conv2d_ref` (shared with `conv_fprop.rs`)
//!   composed with the BN-affine + activation formula.
//!
//! Shares `ConvCase`, `conv2d_ref`, `make_desc` and `make_desc_mut` with
//! `conv_fprop.rs` (`pub(super)` there, imported here via `super::conv_fprop`).

use super::conv_fprop::{ConvCase, conv2d_ref, make_desc, make_desc_mut};
use super::*;

use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

use crate::conv::algo_select::{estimate_gemm_flops, is_winograd_eligible};
use crate::conv::api::conv_bn_relu;
use crate::conv::fused::{FusedBnParams, FusedConvBnAct};
use crate::types::{Activation, ConvolutionDescriptor, TensorDesc, TensorDescMut, TensorLayout};

// ---------------------------------------------------------------------------
// FusedConvBnAct  (fused.rs — load/launch-only fragment)
// ---------------------------------------------------------------------------

fn run_fused(fx: &GpuFixture, activation: Activation) {
    let case = ConvCase {
        n: 1,
        c: 4,
        h: 5,
        w: 5,
        k: 6,
        r: 3,
        s: 3,
        pad_h: 1,
        pad_w: 1,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
        layout: TensorLayout::Nchw,
    };
    let (out_h, out_w) = case.out_hw();
    let in_n = (case.n * case.c * case.h * case.w) as usize;
    let fil_n = (case.k * case.c * case.r * case.s) as usize;
    let out_n = (case.n * case.k * out_h * out_w) as usize;
    let k = case.k as usize;

    let engine = FusedConvBnAct::new(case.problem(PtxType::F32), activation, fx.sm);
    let ptx = engine.generate_ptx().expect("fused ptx");
    let name = engine.kernel_name();
    ptxas_assembles(&ptx, &name).expect("ptxas fused");

    let mut lcg = Lcg::new(0x900d_900d_900d);
    let in32: Vec<f32> = (0..in_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let fil32: Vec<f32> = (0..fil_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let scale32: Vec<f32> = (0..k).map(|_| lcg.range_f32(0.5, 1.5)).collect();
    let bias32: Vec<f32> = (0..k).map(|_| lcg.range_f32(-0.5, 0.5)).collect();

    let in_buf = DeviceBuffer::from_host(&in32).expect("upload input");
    let fil_buf = DeviceBuffer::from_host(&fil32).expect("upload filter");
    let scale_buf = DeviceBuffer::from_host(&scale32).expect("upload scale");
    let bias_buf = DeviceBuffer::from_host(&bias32).expect("upload bias");
    let sentinel = 42.0f32;
    let out_buf = DeviceBuffer::from_host(&vec![sentinel; out_n]).expect("alloc output");

    let in_desc = make_desc(&in_buf, TensorLayout::Nchw);
    let fil_desc = make_desc(&fil_buf, TensorLayout::Nchw);
    let mut out_desc = make_desc_mut(&out_buf, TensorLayout::Nchw);
    let bn = FusedBnParams {
        fused_scale_ptr: scale_buf.as_device_ptr(),
        fused_bias_ptr: bias_buf.as_device_ptr(),
        channels: case.k,
    };

    engine
        .execute(&fx.handle, &in_desc, &fil_desc, &mut out_desc, &bn)
        .expect("fused execute");
    fx.stream().synchronize().expect("synchronize");

    // The fused kernel currently emits only step-marker comments + `ret`
    // (no stores), so the output must remain at the sentinel. This is the
    // honest assertion for the fragment; it becomes a canary once a real body
    // lands.
    let mut got = vec![0.0f32; out_n];
    out_buf.copy_to_host(&mut got).expect("copy output");
    assert!(
        got.iter().all(|&v| v == sentinel),
        "fused conv+bn+act kernel `{name}` is a no-op skeleton; output must be untouched"
    );
}

#[test]
fn fused_conv_bn_relu_f32_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_fused(&fx, Activation::Relu);
}

#[test]
fn fused_conv_bn_gelu_f32_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_fused(&fx, Activation::Gelu);
}

#[test]
fn fused_conv_bn_identity_f32_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_fused(&fx, Activation::None);
}

// ---------------------------------------------------------------------------
// conv_bn_relu  (api.rs — decomposed conv + BN-affine + activation)
// ---------------------------------------------------------------------------
//
// `conv_bn_relu` used to dispatch straight into the `FusedConvBnAct`
// fragment above and return `Ok(())` while leaving `output` completely
// untouched -- the exact same failure mode `run_fused` documents for that
// fragment directly. It now decomposes into a real convolution dispatch
// (`conv_forward`, via an internally-managed workspace) followed by a real
// BN-affine + activation epilogue kernel (`fused::apply_fused_bn_activation`,
// which reuses `moe::fused_moe::emit_activation_ptx`'s already-validated
// activation math). The tests below drive the *public* `conv_bn_relu` entry
// point end-to-end -- no direct engine construction -- and check the result
// against an independent CPU oracle: `conv2d_ref` (the same proven
// convolution reference used throughout this file) composed with the
// BN-affine + activation formula from `fused.rs`'s own module doc.

/// f64 CPU reference for [`Activation`], matching
/// `crate::moe::fused_moe::emit_activation_ptx`'s exact math (the same
/// `ex2.approx`-based transcendental approximations validated on-device by
/// `gpu_tests::moe_linear::activation_transcendental_approx_f32`, which
/// `apply_fused_bn_activation` reuses rather than re-deriving).
fn bn_epilogue_activation_oracle(act: Activation, x: f64) -> f64 {
    match act {
        Activation::None => x,
        Activation::Relu => x.max(0.0),
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::Silu => x / (1.0 + (-x).exp()),
        Activation::Tanh => x.tanh(),
        Activation::Gelu | Activation::GeluTanh => {
            let sqrt_2_over_pi = 0.797_884_560_802_865_4_f64;
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }
    }
}

/// CPU oracle for `conv_bn_relu`'s full decomposition: `conv2d_ref` followed
/// by the per-channel fused-BN affine and activation. `channels_last`
/// mirrors `apply_fused_bn_activation`'s own layout-dependent channel-index
/// recovery (`idx % channels` for NHWC, `(idx / spatial) % channels` for
/// NCHW).
fn conv_bn_act_ref(
    case: ConvCase,
    input: &[f64],
    filter: &[f64],
    fused_scale: &[f32],
    fused_bias: &[f32],
    activation: Activation,
) -> Vec<f64> {
    let conv_out = conv2d_ref(case, input, filter, None);
    let (out_h, out_w) = case.out_hw();
    let spatial = (out_h * out_w) as usize;
    let channels = case.k as usize;
    let channels_last = case.layout.is_channels_last();

    conv_out
        .iter()
        .enumerate()
        .map(|(idx, &x)| {
            let c = if channels_last {
                idx % channels
            } else {
                (idx / spatial) % channels
            };
            let affine = x * f64::from(fused_scale[c]) + f64::from(fused_bias[c]);
            bn_epilogue_activation_oracle(activation, affine)
        })
        .collect()
}

/// Drives the *public* `conv::api::conv_bn_relu` entry point end-to-end
/// (exactly as a real caller would) and checks the result against
/// [`conv_bn_act_ref`]. `tol` is `(rel, abs)`.
fn run_conv_bn_relu(
    fx: &GpuFixture,
    case: ConvCase,
    activation: Activation,
    tol: (f32, f32),
    tag: &str,
) {
    let (out_h, out_w) = case.out_hw();
    let in_n = (case.n * case.c * case.h * case.w) as usize;
    let fil_n = (case.k * case.c * case.r * case.s) as usize;
    let out_n = (case.n * case.k * out_h * out_w) as usize;
    let k = case.k as usize;

    let mut lcg = Lcg::new(0xC0FF_EE00_1357_9BDF);
    let in32: Vec<f32> = (0..in_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let fil32: Vec<f32> = (0..fil_n).map(|_| lcg.range_f32(-0.5, 0.5)).collect();
    let scale32: Vec<f32> = (0..k).map(|_| lcg.range_f32(0.5, 1.5)).collect();
    let bias32: Vec<f32> = (0..k).map(|_| lcg.range_f32(-0.5, 0.5)).collect();

    let in_buf = DeviceBuffer::from_host(&in32).expect("upload input");
    let fil_buf = DeviceBuffer::from_host(&fil32).expect("upload filter");
    let scale_buf = DeviceBuffer::from_host(&scale32).expect("upload scale");
    let bias_buf = DeviceBuffer::from_host(&bias32).expect("upload bias");
    let sentinel = -9999.5f32;
    let mut out_buf = DeviceBuffer::from_host(&vec![sentinel; out_n]).expect("alloc output");

    let (in_desc, fil_desc, mut out_desc) = match case.layout {
        TensorLayout::Nchw => (
            TensorDesc::nchw(&in_buf, case.n, case.c, case.h, case.w).expect("input desc"),
            TensorDesc::nchw(&fil_buf, case.k, case.c, case.r, case.s).expect("filter desc"),
            TensorDescMut::nchw(&mut out_buf, case.n, case.k, out_h, out_w).expect("output desc"),
        ),
        TensorLayout::Nhwc => (
            TensorDesc::nhwc(&in_buf, case.n, case.c, case.h, case.w).expect("input desc"),
            TensorDesc::nhwc(&fil_buf, case.k, case.c, case.r, case.s).expect("filter desc"),
            TensorDescMut::nhwc(&mut out_buf, case.n, case.k, out_h, out_w).expect("output desc"),
        ),
        other => panic!("run_conv_bn_relu only supports NCHW/NHWC test cases, got {other:?}"),
    };
    let conv_desc = ConvolutionDescriptor::conv2d(
        case.pad_h,
        case.pad_w,
        case.str_h,
        case.str_w,
        case.dil_h,
        case.dil_w,
        case.groups,
    )
    .expect("conv desc");
    let bn_params = FusedBnParams {
        fused_scale_ptr: scale_buf.as_device_ptr(),
        fused_bias_ptr: bias_buf.as_device_ptr(),
        channels: case.k,
    };

    conv_bn_relu(
        &fx.handle,
        &in_desc,
        &fil_desc,
        &mut out_desc,
        &conv_desc,
        &bn_params,
        activation,
    )
    .expect("conv_bn_relu");
    fx.stream().synchronize().expect("synchronize");

    let mut gpu = vec![0.0f32; out_n];
    out_buf.copy_to_host(&mut gpu).expect("copy output");

    // The pre-fix `conv_bn_relu` returned `Ok(())` after dispatching straight
    // into the `FusedConvBnAct` no-op skeleton, leaving every element at
    // whatever `output` was initialised to. Catch that failure mode
    // explicitly -- the same way the Winograd dispatcher regression test
    // does -- before the (much stricter) numeric comparison below.
    assert!(
        gpu.iter().any(|&v| v != sentinel),
        "{tag}: output buffer was never written -- conv_bn_relu regressed to \
         the silent conv+BN+act no-op"
    );

    let in_o: Vec<f64> = in32.iter().map(|&x| f64::from(x)).collect();
    let fil_o: Vec<f64> = fil32.iter().map(|&x| f64::from(x)).collect();
    let exp64 = conv_bn_act_ref(case, &in_o, &fil_o, &scale32, &bias32, activation);
    let exp32: Vec<f32> = exp64.iter().map(|&x| x as f32).collect();
    assert_close_f32(&gpu, &exp32, tol.0, tol.1, tag);
}

/// `conv_bn_relu` over a shape that is Winograd-*eligible* and clears the
/// Winograd profitability threshold -- the same shape as
/// `conv_forward_winograd_eligible_shape_matches_cpu_oracle` above, one of
/// the "ordinary mid-size 3x3 CNN layers" `algo_select`'s docs call out as
/// exactly what the pre-gate heuristic would have routed into the broken
/// `WinogradConv` engine. `conv_bn_relu` has its own dispatch path
/// (`conv_forward` via an auto-sized workspace, then
/// `apply_fused_bn_activation`), so this is not redundant with that
/// `conv_forward`-only regression test -- it proves the *fused entry point
/// itself* stays safe at this scale, on top of proving the decomposition's
/// BN + ReLU epilogue is numerically correct.
#[test]
fn conv_bn_relu_nchw_relu_matches_cpu_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 128,
        h: 80,
        w: 80,
        k: 128,
        r: 3,
        s: 3,
        pad_h: 1,
        pad_w: 1,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
        layout: TensorLayout::Nchw,
    };
    let problem = case.problem(PtxType::F32);
    assert!(
        is_winograd_eligible(&problem, case.r, case.s),
        "regression shape must be Winograd-eligible"
    );
    assert!(
        estimate_gemm_flops(&problem, case.r, case.s) > 1_000_000_000,
        "regression shape must clear the Winograd FLOP threshold"
    );
    run_conv_bn_relu(
        &fx,
        case,
        Activation::Relu,
        (2e-4, 2e-4),
        "conv_bn_relu_nchw_relu",
    );
}

/// `conv_bn_relu` over an NHWC tensor with a transcendental activation
/// (SiLU), exercising both the channels-last channel-index recovery in
/// `apply_fused_bn_activation` and the `ex2.approx`-based activation math it
/// reuses from the MoE epilogue, end-to-end through the public API.
#[test]
fn conv_bn_relu_nhwc_silu_matches_cpu_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 2,
        c: 6,
        h: 9,
        w: 11,
        k: 5,
        r: 3,
        s: 3,
        pad_h: 1,
        pad_w: 1,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
        layout: TensorLayout::Nhwc,
    };
    run_conv_bn_relu(
        &fx,
        case,
        Activation::Silu,
        (1e-2, 1e-2),
        "conv_bn_relu_nhwc_silu",
    );
}
