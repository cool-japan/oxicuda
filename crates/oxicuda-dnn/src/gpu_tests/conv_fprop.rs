//! On-device GPU validation for the `conv_fprop` subsystem of `oxicuda-dnn`.
//!
//! Coverage (RTX A4000, sm_86):
//!
//! * **Numeric (CPU-oracle)** — `Conv1x1`, `DepthwiseConv`, `ImplicitGemmConv`
//!   (all driven through their public `execute` engines) and the `im2col_expand`
//!   column-matrix kernel (driven through its public `generate_im2col_ptx` + a
//!   direct launch). Each is checked against an independent CPU re-derivation in
//!   `f32` (and `f64` where the engine supports it). NCHW and NHWC layouts,
//!   padding, stride, dilation, groups and the bias epilogue are all exercised.
//!   The full `Im2colGemmConv::execute` (im2col PTX + BLAS GEMM) is also checked
//!   end-to-end against a direct convolution.
//! * **Load / launch-only (fragments)** — the Winograd input/output transform
//!   kernels and the `FusedConvBnAct` kernel are structural skeletons (the body
//!   emits only step-marker comments and `ret`, writing nothing). For these we
//!   assert they assemble (`ptxas`), JIT-load, launch and synchronise
//!   fault-free, and — documenting the fragment status honestly rather than
//!   green-washing a wrong numeric result — that the output buffer is left
//!   untouched (the kernels perform no stores). These tests are canaries: they
//!   will (correctly) fail the moment the kernels are given a real body.
//! * **Dispatch regression (capability gate)** — `conv::algo_select` must never
//!   route a convolution into the broken `WinogradConv` engine above via the
//!   public `conv::api::conv_forward` entry point. `conv_forward_winograd_eligible_shape_matches_cpu_oracle`
//!   drives a shape that is genuinely Winograd-eligible and over the
//!   profitability threshold end-to-end through `conv_forward` (not through
//!   any single engine directly) and checks the result against `conv2d_ref`,
//!   proving the dispatcher fell back to a numerically-correct engine instead.
//! * **Decomposed fusion (`conv_bn_relu`)** — unlike the `FusedConvBnAct`
//!   fragment above, the public `conv::api::conv_bn_relu` entry point no
//!   longer dispatches into that no-op skeleton. It decomposes into a real
//!   `conv_forward` call plus a real BN-affine + activation epilogue kernel
//!   (`conv::fused::apply_fused_bn_activation`). `conv_bn_relu_nchw_relu_matches_cpu_oracle`
//!   and `conv_bn_relu_nhwc_silu_matches_cpu_oracle` drive the public entry
//!   point end-to-end (NCHW/ReLU at a Winograd-eligible scale, NHWC/SiLU for
//!   layout + transcendental-activation coverage) and check the result
//!   against `conv2d_ref` composed with the BN-affine + activation formula,
//!   proving both that the untouched-buffer failure mode is gone and that
//!   the decomposed math is numerically correct.

use super::*;

use oxicuda_blas::GpuFloat;
use oxicuda_launch::LaunchParams;
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

use crate::conv::algo_select::{estimate_gemm_flops, is_winograd_eligible};
use crate::conv::api::{conv_bn_relu, conv_forward};
use crate::conv::descriptor::ConvProblem;
use crate::conv::fprop::direct::{Conv1x1, DepthwiseConv};
use crate::conv::fprop::im2col_gemm::Im2colGemmConv;
use crate::conv::fprop::implicit_gemm::ImplicitGemmConv;
use crate::conv::fprop::winograd::{WinogradConv, WinogradTileSize};
use crate::conv::fused::{FusedBnParams, FusedConvBnAct};
use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::types::{Activation, ConvolutionDescriptor, TensorDesc, TensorDescMut, TensorLayout};

// ---------------------------------------------------------------------------
// Shared geometry + CPU oracle
// ---------------------------------------------------------------------------

/// Standard 2-D convolution output extent
/// `floor((in + 2*pad - dil*(flt-1) - 1) / stride) + 1`.
fn out_dim(inp: u32, flt: u32, pad: u32, stride: u32, dil: u32) -> u32 {
    (inp + 2 * pad - dil * (flt - 1) - 1) / stride + 1
}

/// Scalar-only (hence `Copy`) convolution geometry, mirroring the fields a
/// [`ConvProblem`] needs. Closures capture it by copy.
#[derive(Clone, Copy)]
struct ConvCase {
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    k: u32,
    r: u32,
    s: u32,
    pad_h: u32,
    pad_w: u32,
    str_h: u32,
    str_w: u32,
    dil_h: u32,
    dil_w: u32,
    groups: u32,
    layout: TensorLayout,
}

impl ConvCase {
    fn out_hw(self) -> (u32, u32) {
        (
            out_dim(self.h, self.r, self.pad_h, self.str_h, self.dil_h),
            out_dim(self.w, self.s, self.pad_w, self.str_w, self.dil_w),
        )
    }

    fn problem(self, ty: PtxType) -> ConvProblem {
        ConvProblem {
            batch: self.n,
            in_channels: self.c,
            in_dims: vec![self.h, self.w],
            out_channels: self.k,
            filter_dims: vec![self.r, self.s],
            padding: vec![self.pad_h, self.pad_w],
            stride: vec![self.str_h, self.str_w],
            dilation: vec![self.dil_h, self.dil_w],
            groups: self.groups,
            input_type: ty,
            output_type: ty,
            layout: self.layout,
        }
    }
}

/// Independent CPU cross-correlation reference. Accumulation order
/// (group-channel, then `r`, then `s`) matches the kernel exactly so the `f64`
/// path agrees to round-off. Indexing follows NCHW vs NHWC from the layout.
fn conv2d_ref(case: ConvCase, input: &[f64], filter: &[f64], bias: Option<&[f64]>) -> Vec<f64> {
    let (out_h, out_w) = case.out_hw();
    let (n, c, h, w) = (
        case.n as usize,
        case.c as usize,
        case.h as usize,
        case.w as usize,
    );
    let (k, r, s) = (case.k as usize, case.r as usize, case.s as usize);
    let (out_h, out_w) = (out_h as usize, out_w as usize);
    let groups = case.groups as usize;
    let icpg = c / groups;
    let ocpg = k / groups;
    let cl = case.layout.is_channels_last();

    let mut out = vec![0.0f64; n * k * out_h * out_w];
    for ni in 0..n {
        for ki in 0..k {
            let group = ki / ocpg;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0.0f64;
                    for cg in 0..icpg {
                        let ci = group * icpg + cg;
                        for ri in 0..r {
                            let ih = oh as isize * case.str_h as isize - case.pad_h as isize
                                + ri as isize * case.dil_h as isize;
                            for si in 0..s {
                                let iw = ow as isize * case.str_w as isize - case.pad_w as isize
                                    + si as isize * case.dil_w as isize;
                                if ih < 0 || iw < 0 || ih as usize >= h || iw as usize >= w {
                                    continue;
                                }
                                let ihu = ih as usize;
                                let iwu = iw as usize;
                                let in_idx = if cl {
                                    ((ni * h + ihu) * w + iwu) * c + ci
                                } else {
                                    ((ni * c + ci) * h + ihu) * w + iwu
                                };
                                let f_idx = if cl {
                                    ((ki * r + ri) * s + si) * icpg + cg
                                } else {
                                    ((ki * icpg + cg) * r + ri) * s + si
                                };
                                acc += input[in_idx] * filter[f_idx];
                            }
                        }
                    }
                    if let Some(bv) = bias {
                        acc += bv[ki];
                    }
                    let o_idx = if cl {
                        ((ni * out_h + oh) * out_w + ow) * k + ki
                    } else {
                        ((ni * k + ki) * out_h + oh) * out_w + ow
                    };
                    out[o_idx] = acc;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Descriptor helpers (engines consume only the device pointer)
// ---------------------------------------------------------------------------

fn make_desc<T: GpuFloat>(buf: &DeviceBuffer<T>, layout: TensorLayout) -> TensorDesc<T> {
    TensorDesc::from_raw(buf.as_device_ptr(), vec![buf.len() as u32], vec![1], layout)
        .expect("input descriptor")
}

fn make_desc_mut<T: GpuFloat>(buf: &DeviceBuffer<T>, layout: TensorLayout) -> TensorDescMut<T> {
    TensorDescMut::from_raw(buf.as_device_ptr(), vec![buf.len() as u32], vec![1], layout)
        .expect("output descriptor")
}

// ---------------------------------------------------------------------------
// Numeric runners (engine `execute` path)
// ---------------------------------------------------------------------------

type LaunchF32<'a> = dyn FnOnce(
        &DnnHandle,
        &TensorDesc<f32>,
        &TensorDesc<f32>,
        Option<&TensorDesc<f32>>,
        &mut TensorDescMut<f32>,
    ) -> DnnResult<()>
    + 'a;

fn run_conv_f32(
    fx: &GpuFixture,
    case: ConvCase,
    with_bias: bool,
    tag: &str,
    launch: Box<LaunchF32<'_>>,
) {
    let (out_h, out_w) = case.out_hw();
    let icpg = case.c / case.groups;
    let in_n = (case.n * case.c * case.h * case.w) as usize;
    let fil_n = (case.k * icpg * case.r * case.s) as usize;
    let out_n = (case.n * case.k * out_h * out_w) as usize;
    let k = case.k as usize;

    let mut lcg = Lcg::new(0x00c0_ffee_1234_5678);
    let in32: Vec<f32> = (0..in_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let fil32: Vec<f32> = (0..fil_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let bias32: Vec<f32> = if with_bias {
        (0..k).map(|_| lcg.range_f32(-0.5, 0.5)).collect()
    } else {
        Vec::new()
    };

    let in_buf = DeviceBuffer::from_host(&in32).expect("upload input");
    let fil_buf = DeviceBuffer::from_host(&fil32).expect("upload filter");
    let bias_buf = if with_bias {
        Some(DeviceBuffer::from_host(&bias32).expect("upload bias"))
    } else {
        None
    };
    let init = vec![-987.0f32; out_n];
    let out_buf = DeviceBuffer::from_host(&init).expect("alloc output");

    let in_desc = make_desc(&in_buf, case.layout);
    let fil_desc = make_desc(&fil_buf, case.layout);
    let bias_desc = bias_buf.as_ref().map(|b| make_desc(b, case.layout));
    let mut out_desc = make_desc_mut(&out_buf, case.layout);

    launch(
        &fx.handle,
        &in_desc,
        &fil_desc,
        bias_desc.as_ref(),
        &mut out_desc,
    )
    .expect("kernel launch");
    fx.stream().synchronize().expect("synchronize");

    let mut gpu = vec![0.0f32; out_n];
    out_buf.copy_to_host(&mut gpu).expect("copy output");

    let in_o: Vec<f64> = in32.iter().map(|&x| f64::from(x)).collect();
    let fil_o: Vec<f64> = fil32.iter().map(|&x| f64::from(x)).collect();
    let bias_o: Option<Vec<f64>> =
        with_bias.then(|| bias32.iter().map(|&x| f64::from(x)).collect());
    let exp64 = conv2d_ref(case, &in_o, &fil_o, bias_o.as_deref());
    let exp32: Vec<f32> = exp64.iter().map(|&x| x as f32).collect();
    assert_close_f32(&gpu, &exp32, 2e-4, 2e-4, tag);
}

type LaunchF64<'a> = dyn FnOnce(
        &DnnHandle,
        &TensorDesc<f64>,
        &TensorDesc<f64>,
        Option<&TensorDesc<f64>>,
        &mut TensorDescMut<f64>,
    ) -> DnnResult<()>
    + 'a;

fn run_conv_f64(
    fx: &GpuFixture,
    case: ConvCase,
    with_bias: bool,
    tag: &str,
    launch: Box<LaunchF64<'_>>,
) {
    let (out_h, out_w) = case.out_hw();
    let icpg = case.c / case.groups;
    let in_n = (case.n * case.c * case.h * case.w) as usize;
    let fil_n = (case.k * icpg * case.r * case.s) as usize;
    let out_n = (case.n * case.k * out_h * out_w) as usize;
    let k = case.k as usize;

    let mut lcg = Lcg::new(0x0bad_f00d_dead_beef);
    let in64: Vec<f64> = (0..in_n).map(|_| lcg.range_f64(-1.0, 1.0)).collect();
    let fil64: Vec<f64> = (0..fil_n).map(|_| lcg.range_f64(-1.0, 1.0)).collect();
    let bias64: Vec<f64> = if with_bias {
        (0..k).map(|_| lcg.range_f64(-0.5, 0.5)).collect()
    } else {
        Vec::new()
    };

    let in_buf = DeviceBuffer::from_host(&in64).expect("upload input");
    let fil_buf = DeviceBuffer::from_host(&fil64).expect("upload filter");
    let bias_buf = if with_bias {
        Some(DeviceBuffer::from_host(&bias64).expect("upload bias"))
    } else {
        None
    };
    let init = vec![-987.0f64; out_n];
    let out_buf = DeviceBuffer::from_host(&init).expect("alloc output");

    let in_desc = make_desc(&in_buf, case.layout);
    let fil_desc = make_desc(&fil_buf, case.layout);
    let bias_desc = bias_buf.as_ref().map(|b| make_desc(b, case.layout));
    let mut out_desc = make_desc_mut(&out_buf, case.layout);

    launch(
        &fx.handle,
        &in_desc,
        &fil_desc,
        bias_desc.as_ref(),
        &mut out_desc,
    )
    .expect("kernel launch");
    fx.stream().synchronize().expect("synchronize");

    let mut gpu = vec![0.0f64; out_n];
    out_buf.copy_to_host(&mut gpu).expect("copy output");

    let bias_ref = if with_bias {
        Some(bias64.as_slice())
    } else {
        None
    };
    let exp = conv2d_ref(case, &in64, &fil64, bias_ref);
    assert_close_f64(&gpu, &exp, 1e-9, 1e-10, tag);
}

// ---------------------------------------------------------------------------
// Conv1x1  (direct.rs — numeric)
// ---------------------------------------------------------------------------

#[test]
fn conv1x1_f32_nchw_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 2,
        c: 8,
        h: 5,
        w: 4,
        k: 6,
        r: 1,
        s: 1,
        pad_h: 0,
        pad_w: 0,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
        layout: TensorLayout::Nchw,
    };
    let sm = fx.sm;
    run_conv_f32(
        &fx,
        case,
        false,
        "conv1x1_f32_nchw",
        Box::new(move |h, i, f, _b, o| {
            Conv1x1::new(case.problem(PtxType::F32), sm)?.execute(h, i, f, o)
        }),
    );
}

#[test]
fn conv1x1_f32_nhwc_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 2,
        c: 5,
        h: 4,
        w: 6,
        k: 7,
        r: 1,
        s: 1,
        pad_h: 0,
        pad_w: 0,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
        layout: TensorLayout::Nhwc,
    };
    let sm = fx.sm;
    run_conv_f32(
        &fx,
        case,
        false,
        "conv1x1_f32_nhwc",
        Box::new(move |h, i, f, _b, o| {
            Conv1x1::new(case.problem(PtxType::F32), sm)?.execute(h, i, f, o)
        }),
    );
}

#[test]
fn conv1x1_f64_nchw_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 6,
        h: 3,
        w: 5,
        k: 4,
        r: 1,
        s: 1,
        pad_h: 0,
        pad_w: 0,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
        layout: TensorLayout::Nchw,
    };
    let sm = fx.sm;
    run_conv_f64(
        &fx,
        case,
        false,
        "conv1x1_f64_nchw",
        Box::new(move |h, i, f, _b, o| {
            Conv1x1::new(case.problem(PtxType::F64), sm)?.execute(h, i, f, o)
        }),
    );
}

// ---------------------------------------------------------------------------
// DepthwiseConv  (direct.rs — numeric)
// ---------------------------------------------------------------------------

#[test]
fn depthwise_3x3_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 2,
        c: 6,
        h: 7,
        w: 8,
        k: 6,
        r: 3,
        s: 3,
        pad_h: 1,
        pad_w: 1,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 6,
        layout: TensorLayout::Nchw,
    };
    let sm = fx.sm;
    run_conv_f32(
        &fx,
        case,
        false,
        "depthwise_3x3_f32",
        Box::new(move |h, i, f, _b, o| {
            DepthwiseConv::new(case.problem(PtxType::F32), sm)?.execute(h, i, f, o)
        }),
    );
}

#[test]
fn depthwise_3x3_strided_dilated_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 5,
        h: 9,
        w: 9,
        k: 5,
        r: 3,
        s: 3,
        pad_h: 2,
        pad_w: 2,
        str_h: 2,
        str_w: 1,
        dil_h: 2,
        dil_w: 2,
        groups: 5,
        layout: TensorLayout::Nchw,
    };
    let sm = fx.sm;
    run_conv_f32(
        &fx,
        case,
        false,
        "depthwise_3x3_strided_dilated_f32",
        Box::new(move |h, i, f, _b, o| {
            DepthwiseConv::new(case.problem(PtxType::F32), sm)?.execute(h, i, f, o)
        }),
    );
}

#[test]
fn depthwise_5x5_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 4,
        h: 8,
        w: 8,
        k: 4,
        r: 5,
        s: 5,
        pad_h: 2,
        pad_w: 2,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 4,
        layout: TensorLayout::Nchw,
    };
    let sm = fx.sm;
    run_conv_f32(
        &fx,
        case,
        false,
        "depthwise_5x5_f32",
        Box::new(move |h, i, f, _b, o| {
            DepthwiseConv::new(case.problem(PtxType::F32), sm)?.execute(h, i, f, o)
        }),
    );
}

#[test]
fn depthwise_3x3_f64_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 3,
        h: 6,
        w: 6,
        k: 3,
        r: 3,
        s: 3,
        pad_h: 1,
        pad_w: 1,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 3,
        layout: TensorLayout::Nchw,
    };
    let sm = fx.sm;
    run_conv_f64(
        &fx,
        case,
        false,
        "depthwise_3x3_f64",
        Box::new(move |h, i, f, _b, o| {
            DepthwiseConv::new(case.problem(PtxType::F64), sm)?.execute(h, i, f, o)
        }),
    );
}

// ---------------------------------------------------------------------------
// ImplicitGemmConv  (implicit_gemm.rs — numeric)
// ---------------------------------------------------------------------------

#[test]
fn implicit_gemm_3x3_f32_nchw_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 2,
        c: 4,
        h: 6,
        w: 6,
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
        layout: TensorLayout::Nchw,
    };
    let sm = fx.sm;
    run_conv_f32(
        &fx,
        case,
        false,
        "implicit_gemm_3x3_f32_nchw",
        Box::new(move |h, i, f, b, o| {
            ImplicitGemmConv::new(case.problem(PtxType::F32), sm).execute(h, i, f, b, o)
        }),
    );
}

#[test]
fn implicit_gemm_3x3_f32_nchw_with_bias() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 3,
        h: 5,
        w: 7,
        k: 4,
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
    let sm = fx.sm;
    run_conv_f32(
        &fx,
        case,
        true,
        "implicit_gemm_3x3_f32_nchw_bias",
        Box::new(move |h, i, f, b, o| {
            ImplicitGemmConv::new(case.problem(PtxType::F32), sm).execute(h, i, f, b, o)
        }),
    );
}

#[test]
fn implicit_gemm_strided_dilated_f32_nchw() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 3,
        h: 8,
        w: 8,
        k: 6,
        r: 3,
        s: 3,
        pad_h: 2,
        pad_w: 1,
        str_h: 2,
        str_w: 2,
        dil_h: 2,
        dil_w: 1,
        groups: 1,
        layout: TensorLayout::Nchw,
    };
    let sm = fx.sm;
    run_conv_f32(
        &fx,
        case,
        false,
        "implicit_gemm_strided_dilated_f32",
        Box::new(move |h, i, f, b, o| {
            ImplicitGemmConv::new(case.problem(PtxType::F32), sm).execute(h, i, f, b, o)
        }),
    );
}

#[test]
fn implicit_gemm_grouped_f32_nchw() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 4,
        h: 5,
        w: 5,
        k: 4,
        r: 3,
        s: 3,
        pad_h: 1,
        pad_w: 1,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 2,
        layout: TensorLayout::Nchw,
    };
    let sm = fx.sm;
    run_conv_f32(
        &fx,
        case,
        false,
        "implicit_gemm_grouped_f32",
        Box::new(move |h, i, f, b, o| {
            ImplicitGemmConv::new(case.problem(PtxType::F32), sm).execute(h, i, f, b, o)
        }),
    );
}

#[test]
fn implicit_gemm_3x3_f32_nhwc_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 2,
        c: 4,
        h: 5,
        w: 6,
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
    let sm = fx.sm;
    run_conv_f32(
        &fx,
        case,
        false,
        "implicit_gemm_3x3_f32_nhwc",
        Box::new(move |h, i, f, b, o| {
            ImplicitGemmConv::new(case.problem(PtxType::F32), sm).execute(h, i, f, b, o)
        }),
    );
}

#[test]
fn implicit_gemm_3x3_f64_nchw_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 3,
        h: 5,
        w: 5,
        k: 4,
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
    let sm = fx.sm;
    run_conv_f64(
        &fx,
        case,
        false,
        "implicit_gemm_3x3_f64_nchw",
        Box::new(move |h, i, f, b, o| {
            ImplicitGemmConv::new(case.problem(PtxType::F64), sm).execute(h, i, f, b, o)
        }),
    );
}

// ---------------------------------------------------------------------------
// im2col_expand  (im2col_gemm.rs — numeric, direct PTX launch)
// ---------------------------------------------------------------------------

/// CPU reference for the `(C*R*S) x M` row-major column matrix that
/// `im2col_expand` produces. `gid = k_idx * M + m_idx`, `M = N*out_h*out_w`,
/// groups are assumed to be 1.
fn im2col_ref(case: ConvCase, input: &[f64]) -> Vec<f64> {
    let (out_h, out_w) = case.out_hw();
    let (out_h, out_w) = (out_h as usize, out_w as usize);
    let (n, c, h, w) = (
        case.n as usize,
        case.c as usize,
        case.h as usize,
        case.w as usize,
    );
    let (r, s) = (case.r as usize, case.s as usize);
    let spatial = out_h * out_w;
    let m = n * spatial;
    let k_dim = c * r * s;
    let mut col = vec![0.0f64; k_dim * m];
    for k_idx in 0..k_dim {
        let ci = k_idx / (r * s);
        let krem = k_idx % (r * s);
        let kr = krem / s;
        let ks = krem % s;
        for m_idx in 0..m {
            let batch_n = m_idx / spatial;
            let srem = m_idx % spatial;
            let oh = srem / out_w;
            let ow = srem % out_w;
            let ih = oh as isize * case.str_h as isize - case.pad_h as isize
                + kr as isize * case.dil_h as isize;
            let iw = ow as isize * case.str_w as isize - case.pad_w as isize
                + ks as isize * case.dil_w as isize;
            if ih >= 0 && iw >= 0 && (ih as usize) < h && (iw as usize) < w {
                let in_idx = ((batch_n * c + ci) * h + ih as usize) * w + iw as usize;
                col[k_idx * m + m_idx] = input[in_idx];
            }
        }
    }
    col
}

#[test]
fn im2col_expand_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 2,
        c: 3,
        h: 6,
        w: 5,
        k: 1,
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
    let m = (case.n * out_h * out_w) as usize;
    let k_dim = (case.c * case.r * case.s) as usize;
    let total = (m * k_dim) as u32;
    let in_n = (case.n * case.c * case.h * case.w) as usize;

    let mut lcg = Lcg::new(0xfeed_face_0001);
    let in32: Vec<f32> = (0..in_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let in_buf = DeviceBuffer::from_host(&in32).expect("upload input");
    let init = vec![-321.0f32; k_dim * m];
    let col_buf = DeviceBuffer::from_host(&init).expect("alloc col");

    let engine = Im2colGemmConv::new(case.problem(PtxType::F32), fx.sm);
    let ptx = engine.generate_im2col_ptx().expect("im2col ptx");
    let entry = engine.im2col_kernel_name();
    ptxas_assembles(&ptx, "im2col_expand_f32").expect("ptxas im2col f32");
    let kernel = load_kernel(&ptx, &entry);

    let grid = ceil_div(total, 256);
    let params = LaunchParams::new(grid, 256u32);
    let args = (
        in_buf.as_device_ptr(),
        col_buf.as_device_ptr(),
        case.n,
        case.c,
        case.h,
        case.w,
        case.r,
        case.s,
        out_h,
        out_w,
        case.pad_h,
        case.pad_w,
        case.str_h,
        case.str_w,
        case.dil_h,
        case.dil_w,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch im2col");
    fx.stream().synchronize().expect("synchronize");

    let mut gpu = vec![0.0f32; k_dim * m];
    col_buf.copy_to_host(&mut gpu).expect("copy col");

    let in_o: Vec<f64> = in32.iter().map(|&x| f64::from(x)).collect();
    let exp64 = im2col_ref(case, &in_o);
    let exp32: Vec<f32> = exp64.iter().map(|&x| x as f32).collect();
    assert_close_f32(&gpu, &exp32, 1e-6, 1e-6, "im2col_expand_f32");
}

#[test]
fn im2col_expand_f64_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 2,
        h: 5,
        w: 6,
        k: 1,
        r: 3,
        s: 3,
        pad_h: 1,
        pad_w: 0,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
        layout: TensorLayout::Nchw,
    };
    let (out_h, out_w) = case.out_hw();
    let m = (case.n * out_h * out_w) as usize;
    let k_dim = (case.c * case.r * case.s) as usize;
    let total = (m * k_dim) as u32;
    let in_n = (case.n * case.c * case.h * case.w) as usize;

    let mut lcg = Lcg::new(0xfeed_face_0002);
    let in64: Vec<f64> = (0..in_n).map(|_| lcg.range_f64(-1.0, 1.0)).collect();
    let in_buf = DeviceBuffer::from_host(&in64).expect("upload input");
    let init = vec![-321.0f64; k_dim * m];
    let col_buf = DeviceBuffer::from_host(&init).expect("alloc col");

    let engine = Im2colGemmConv::new(case.problem(PtxType::F64), fx.sm);
    let ptx = engine.generate_im2col_ptx().expect("im2col ptx");
    let entry = engine.im2col_kernel_name();
    ptxas_assembles(&ptx, "im2col_expand_f64").expect("ptxas im2col f64");
    let kernel = load_kernel(&ptx, &entry);

    let grid = ceil_div(total, 256);
    let params = LaunchParams::new(grid, 256u32);
    let args = (
        in_buf.as_device_ptr(),
        col_buf.as_device_ptr(),
        case.n,
        case.c,
        case.h,
        case.w,
        case.r,
        case.s,
        out_h,
        out_w,
        case.pad_h,
        case.pad_w,
        case.str_h,
        case.str_w,
        case.dil_h,
        case.dil_w,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch im2col");
    fx.stream().synchronize().expect("synchronize");

    let mut gpu = vec![0.0f64; k_dim * m];
    col_buf.copy_to_host(&mut gpu).expect("copy col");

    let exp = im2col_ref(case, &in64);
    assert_close_f64(&gpu, &exp, 1e-12, 1e-12, "im2col_expand_f64");
}

/// Full im2col + BLAS GEMM pipeline (`Im2colGemmConv::execute`) against a direct
/// convolution. Batch is 1 so the GEMM's `[K, M]` output coincides with the
/// NCHW `[N, K, P, Q]` layout the oracle uses.
#[test]
fn im2col_gemm_execute_matches_direct_conv_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 3,
        h: 5,
        w: 5,
        k: 4,
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

    let mut lcg = Lcg::new(0x1357_9bdf_2468);
    let in32: Vec<f32> = (0..in_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let fil32: Vec<f32> = (0..fil_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();

    let in_buf = DeviceBuffer::from_host(&in32).expect("upload input");
    let fil_buf = DeviceBuffer::from_host(&fil32).expect("upload filter");
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f32; out_n]).expect("alloc output");

    let engine = Im2colGemmConv::new(case.problem(PtxType::F32), fx.sm);
    let ws_bytes = engine.workspace_bytes().expect("workspace bytes");
    let ws = DeviceBuffer::from_host(&vec![0u8; ws_bytes]).expect("alloc workspace");

    let in_desc = make_desc(&in_buf, TensorLayout::Nchw);
    let fil_desc = make_desc(&fil_buf, TensorLayout::Nchw);
    let mut out_desc = make_desc_mut(&out_buf, TensorLayout::Nchw);
    let mut ws_mut = ws;
    engine
        .execute(&fx.handle, &in_desc, &fil_desc, &mut out_desc, &mut ws_mut)
        .expect("im2col gemm execute");
    fx.stream().synchronize().expect("synchronize");

    let mut gpu = vec![0.0f32; out_n];
    out_buf.copy_to_host(&mut gpu).expect("copy output");

    let in_o: Vec<f64> = in32.iter().map(|&x| f64::from(x)).collect();
    let fil_o: Vec<f64> = fil32.iter().map(|&x| f64::from(x)).collect();
    let exp64 = conv2d_ref(case, &in_o, &fil_o, None);
    let exp32: Vec<f32> = exp64.iter().map(|&x| x as f32).collect();
    assert_close_f32(&gpu, &exp32, 2e-4, 2e-4, "im2col_gemm_execute_f32");
}

// ---------------------------------------------------------------------------
// Winograd transforms  (winograd.rs — load/launch-only fragments)
// ---------------------------------------------------------------------------

fn winograd_problem(layout: TensorLayout) -> ConvProblem {
    ConvCase {
        n: 1,
        c: 2,
        h: 8,
        w: 8,
        k: 3,
        r: 3,
        s: 3,
        pad_h: 1,
        pad_w: 1,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
        layout,
    }
    .problem(PtxType::F32)
}

fn run_winograd_input(fx: &GpuFixture, tile: WinogradTileSize) {
    let problem = winograd_problem(TensorLayout::Nchw);
    let conv = WinogradConv::with_tile_size(problem, tile, fx.sm).expect("winograd engine");
    let ptx = conv
        .generate_input_transform_ptx()
        .expect("winograd input ptx");
    let entry = format!("winograd_input_transform_f{}x3", tile.output_tile());
    ptxas_assembles(&ptx, &entry).expect("ptxas winograd input transform");
    let kernel = load_kernel(&ptx, &entry);

    let (in_h, in_w) = (8u32, 8u32);
    let (out_h, out_w) = (8u32, 8u32);
    let (batch, channels) = (1u32, 2u32);
    let ot = tile.output_tile();
    let alpha2 = tile.transform_elements();
    let num_tiles = out_h.div_ceil(ot) * out_w.div_ceil(ot) * batch * channels;

    let in_buf = DeviceBuffer::from_host(&vec![0.5f32; (batch * channels * in_h * in_w) as usize])
        .expect("alloc winograd input");
    let sentinel = 7.0f32;
    let tr_len = (num_tiles * alpha2).max(256) as usize;
    let tr_buf = DeviceBuffer::from_host(&vec![sentinel; tr_len]).expect("alloc transformed");

    let grid = ceil_div(num_tiles, 256);
    let params = LaunchParams::new(grid, 256u32);
    let args = (
        in_buf.as_device_ptr(),
        tr_buf.as_device_ptr(),
        batch,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        1u32,
        1u32,
        num_tiles,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch winograd input transform");
    fx.stream().synchronize().expect("synchronize");

    // The input-transform kernel is a structural skeleton: it bounds-checks and
    // emits step-marker comments only, performing zero stores. Confirm it ran
    // fault-free and left the workspace untouched (no green-washing).
    let mut got = vec![0.0f32; tr_len];
    tr_buf.copy_to_host(&mut got).expect("copy transformed");
    assert!(
        got.iter().all(|&v| v == sentinel),
        "winograd input transform f{}x3 is a no-op skeleton; workspace must be untouched",
        tile.output_tile()
    );
}

fn run_winograd_output(fx: &GpuFixture, tile: WinogradTileSize) {
    let problem = winograd_problem(TensorLayout::Nchw);
    let conv = WinogradConv::with_tile_size(problem, tile, fx.sm).expect("winograd engine");
    let ptx = conv
        .generate_output_transform_ptx()
        .expect("winograd output ptx");
    let entry = format!("winograd_output_transform_f{}x3", tile.output_tile());
    ptxas_assembles(&ptx, &entry).expect("ptxas winograd output transform");
    let kernel = load_kernel(&ptx, &entry);

    let (out_h, out_w) = (8u32, 8u32);
    let (batch, out_channels) = (1u32, 3u32);
    let ot = tile.output_tile();
    let alpha2 = tile.transform_elements();
    let num_tiles = out_h.div_ceil(ot) * out_w.div_ceil(ot) * batch * out_channels;

    let tr_len = (num_tiles * alpha2).max(256) as usize;
    let tr_buf = DeviceBuffer::from_host(&vec![0.25f32; tr_len]).expect("alloc transformed");
    let sentinel = -3.0f32;
    let out_len = (batch * out_channels * out_h * out_w).max(256) as usize;
    let out_buf = DeviceBuffer::from_host(&vec![sentinel; out_len]).expect("alloc output");

    let grid = ceil_div(num_tiles, 256);
    let params = LaunchParams::new(grid, 256u32);
    let args = (
        tr_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        0u64,
        batch,
        out_channels,
        out_h,
        out_w,
        num_tiles,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch winograd output transform");
    fx.stream().synchronize().expect("synchronize");

    let mut got = vec![0.0f32; out_len];
    out_buf.copy_to_host(&mut got).expect("copy output");
    assert!(
        got.iter().all(|&v| v == sentinel),
        "winograd output transform f{}x3 is a no-op skeleton; output must be untouched",
        tile.output_tile()
    );
}

#[test]
fn winograd_input_transform_f2x3_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_winograd_input(&fx, WinogradTileSize::F2x3);
}

#[test]
fn winograd_input_transform_f4x3_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_winograd_input(&fx, WinogradTileSize::F4x3);
}

#[test]
fn winograd_output_transform_f2x3_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_winograd_output(&fx, WinogradTileSize::F2x3);
}

#[test]
fn winograd_output_transform_f4x3_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_winograd_output(&fx, WinogradTileSize::F4x3);
}

// ---------------------------------------------------------------------------
// conv_forward — Winograd capability-gate regression (algo_select.rs)
// ---------------------------------------------------------------------------
//
// The Winograd fragment tests above prove `WinogradConv`'s own kernels are
// no-ops. The test below proves the *dispatcher* no longer routes anything
// there: `conv::algo_select::select_algorithm` gates Rule 3 behind
// `winograd_forward_implemented()` (currently `false`), so the public
// `conv::api::conv_forward` entry point must fall back to a real engine for
// a shape that is otherwise squarely Winograd-eligible.

/// A conv shape that is Winograd-*eligible* (3x3 filter, unit stride and
/// dilation, F32, groups=1) and clears the Winograd profitability FLOP
/// threshold: 128 in/out channels, 80x80 output, batch 1 — mirroring a
/// realistic SCRFD-scale mid-network layer (~1.89e9 estimated GEMM FLOPs).
/// This is exactly the class of shape `select_algorithm`'s Rule 3 would
/// have routed into the broken, no-op `WinogradConv` engine before
/// `winograd_forward_implemented()` gated it closed.
///
/// Regression test for the public, high-level `conv_forward` API: before
/// the gate existed, this call would have returned `Ok(())` while leaving
/// `output` at whatever value it was initialised to, because
/// `WinogradConv::execute` launches kernels that never load, transform, or
/// store anything (see the "Honesty contract" above and in
/// `gpu_tests/mod.rs`). The test drives the *actual* dispatcher
/// (`ConvProblem::from_descriptors` + `select_algorithm`, exactly as
/// `conv_forward` does internally) rather than constructing a specific
/// engine directly, so it fails the same way a real caller of
/// `oxicuda_dnn::conv::api::conv_forward` would have failed.
#[test]
fn conv_forward_winograd_eligible_shape_matches_cpu_oracle() {
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

    // Confirm this shape really is Winograd-eligible and over the
    // profitability threshold using the crate's own eligibility logic —
    // i.e. this test exercises the capability gate specifically, rather
    // than a shape that happens to fall through to Im2colGemm for some
    // unrelated reason (wrong dtype, non-unit stride, etc).
    let problem = case.problem(PtxType::F32);
    assert!(
        is_winograd_eligible(&problem, case.r, case.s),
        "regression shape must be Winograd-eligible"
    );
    assert!(
        estimate_gemm_flops(&problem, case.r, case.s) > 1_000_000_000,
        "regression shape must clear the Winograd FLOP threshold"
    );

    let (out_h, out_w) = case.out_hw();
    let in_n = (case.n * case.c * case.h * case.w) as usize;
    let fil_n = (case.k * case.c * case.r * case.s) as usize;
    let out_n = (case.n * case.k * out_h * out_w) as usize;

    let mut lcg = Lcg::new(0x9016_0729_dead_c0de);
    let in32: Vec<f32> = (0..in_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let fil32: Vec<f32> = (0..fil_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();

    let in_buf = DeviceBuffer::from_host(&in32).expect("upload input");
    let fil_buf = DeviceBuffer::from_host(&fil32).expect("upload filter");
    let sentinel = -555.0f32;
    let mut out_buf = DeviceBuffer::from_host(&vec![sentinel; out_n]).expect("alloc output");

    let in_desc = TensorDesc::nchw(&in_buf, case.n, case.c, case.h, case.w).expect("input desc");
    let fil_desc = TensorDesc::nchw(&fil_buf, case.k, case.c, case.r, case.s).expect("filter desc");
    let mut out_desc =
        TensorDescMut::nchw(&mut out_buf, case.n, case.k, out_h, out_w).expect("output desc");
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

    // Drive the *public* dispatcher exactly as a real caller would: call
    // with no workspace first so the algorithm `select_algorithm` actually
    // picked reports how much it needs. This also confirms the shape
    // selected an algorithm that requires workspace at all (Winograd and
    // Im2colGemm both do; Direct/ImplicitGemm don't), so a pass here is
    // real evidence the Winograd path specifically was avoided rather than
    // some unrelated no-workspace engine being selected by coincidence.
    let err = conv_forward(
        &fx.handle,
        &in_desc,
        &fil_desc,
        &mut out_desc,
        &conv_desc,
        None,
    )
    .expect_err("algorithm selected for this shape must require a workspace");
    let ws_bytes = match err {
        DnnError::WorkspaceRequired(n) => n,
        other => panic!("expected WorkspaceRequired, got {other:?}"),
    };
    assert!(ws_bytes > 0, "reported workspace size must be positive");
    let mut ws = DeviceBuffer::from_host(&vec![0u8; ws_bytes]).expect("alloc workspace");

    conv_forward(
        &fx.handle,
        &in_desc,
        &fil_desc,
        &mut out_desc,
        &conv_desc,
        Some(&mut ws),
    )
    .expect("conv_forward with workspace");
    fx.stream().synchronize().expect("synchronize");

    let mut gpu = vec![0.0f32; out_n];
    out_buf.copy_to_host(&mut gpu).expect("copy output");

    // Before the capability gate, `WinogradConv::execute` would have
    // returned `Ok(())` while leaving every element at the sentinel. Catch
    // that failure mode explicitly and separately from the numeric
    // comparison below (an all-sentinel buffer could in principle, if
    // astronomically unlikely with random f32 inputs, still slip past a
    // per-element tolerance check).
    assert!(
        gpu.iter().any(|&v| v != sentinel),
        "output buffer was never written -- looks like the broken no-op \
         Winograd path was selected"
    );

    let in_o: Vec<f64> = in32.iter().map(|&x| f64::from(x)).collect();
    let fil_o: Vec<f64> = fil32.iter().map(|&x| f64::from(x)).collect();
    let exp64 = conv2d_ref(case, &in_o, &fil_o, None);
    let exp32: Vec<f32> = exp64.iter().map(|&x| x as f32).collect();
    assert_close_f32(
        &gpu,
        &exp32,
        2e-4,
        2e-4,
        "conv_forward_winograd_eligible_shape",
    );
}

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

// ---------------------------------------------------------------------------
// ptxas pre-screen guarantees (numeric kernels assemble for sm_86)
// ---------------------------------------------------------------------------

#[test]
fn conv_fprop_kernels_assemble_sm86() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 4,
        h: 6,
        w: 6,
        k: 4,
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
    let one = ConvCase {
        r: 1,
        s: 1,
        groups: 1,
        ..case
    };

    let c1_f32 = Conv1x1::new(one.problem(PtxType::F32), fx.sm)
        .expect("conv1x1 f32")
        .generate_ptx()
        .expect("conv1x1 f32 ptx");
    ptxas_assembles(&c1_f32, "conv1x1_f32").expect("conv1x1 f32 assembles");
    let c1_f64 = Conv1x1::new(one.problem(PtxType::F64), fx.sm)
        .expect("conv1x1 f64")
        .generate_ptx()
        .expect("conv1x1 f64 ptx");
    ptxas_assembles(&c1_f64, "conv1x1_f64").expect("conv1x1 f64 assembles");

    let dw = ConvCase {
        groups: 4,
        k: 4,
        ..case
    };
    let dw_f32 = DepthwiseConv::new(dw.problem(PtxType::F32), fx.sm)
        .expect("depthwise f32")
        .generate_ptx()
        .expect("depthwise f32 ptx");
    ptxas_assembles(&dw_f32, "depthwise_f32").expect("depthwise f32 assembles");

    let ig_f32 = ImplicitGemmConv::new(case.problem(PtxType::F32), fx.sm)
        .generate_ptx()
        .expect("implicit gemm f32 ptx");
    ptxas_assembles(&ig_f32, "implicit_gemm_f32").expect("implicit gemm f32 assembles");
    let ig_f64 = ImplicitGemmConv::new(case.problem(PtxType::F64), fx.sm)
        .generate_ptx()
        .expect("implicit gemm f64 ptx");
    ptxas_assembles(&ig_f64, "implicit_gemm_f64").expect("implicit gemm f64 assembles");
}

// ---------------------------------------------------------------------------
// Compiled-kernel cache (handle module cache + occupancy launch config)
// ---------------------------------------------------------------------------

/// Runs one `ImplicitGemmConv` shape on `handle` and asserts the result against
/// the CPU oracle, returning nothing. Unlike [`run_conv_f32`] this takes an
/// explicit handle so several shapes can share one handle — which is exactly
/// what makes it a cache-key test.
fn assert_implicit_gemm_correct(handle: &DnnHandle, sm: SmVersion, case: ConvCase, tag: &str) {
    let (out_h, out_w) = case.out_hw();
    let icpg = case.c / case.groups;
    let in_n = (case.n * case.c * case.h * case.w) as usize;
    let fil_n = (case.k * icpg * case.r * case.s) as usize;
    let out_n = (case.n * case.k * out_h * out_w) as usize;

    let mut lcg = Lcg::new(0x05ee_d0fc_ac4e_u64);
    let in32: Vec<f32> = (0..in_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let fil32: Vec<f32> = (0..fil_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();

    let in_buf = DeviceBuffer::from_host(&in32).expect("upload input");
    let fil_buf = DeviceBuffer::from_host(&fil32).expect("upload filter");
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f32; out_n]).expect("alloc output");

    let in_desc = make_desc(&in_buf, case.layout);
    let fil_desc = make_desc(&fil_buf, case.layout);
    let mut out_desc = make_desc_mut(&out_buf, case.layout);

    ImplicitGemmConv::new(case.problem(PtxType::F32), sm)
        .execute(handle, &in_desc, &fil_desc, None, &mut out_desc)
        .expect("implicit gemm launch");
    handle.stream().synchronize().expect("synchronize");

    let mut gpu = vec![0.0f32; out_n];
    out_buf.copy_to_host(&mut gpu).expect("copy output");

    let in64: Vec<f64> = in32.iter().map(|&v| f64::from(v)).collect();
    let fil64: Vec<f64> = fil32.iter().map(|&v| f64::from(v)).collect();
    let exp: Vec<f32> = conv2d_ref(case, &in64, &fil64, None)
        .into_iter()
        .map(|v| v as f32)
        .collect();
    assert_close_f32(&gpu, &exp, 1e-5, 1e-6, tag);
}

/// The cache must actually *hit*: calling the same convolution shape many times
/// on one handle may JIT exactly one module, not one per call.
///
/// This is the whole point of the compiled-kernel cache — before it existed
/// every `execute` ran `Module::from_ptx`, a real `cuModuleLoadData` compile, on
/// every single call. A regression here is silent (results stay correct, the
/// pipeline just gets ~194 us/call slower), so it is asserted rather than
/// eyeballed.
#[test]
fn repeated_conv_compiles_exactly_one_module() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let case = ConvCase {
        n: 1,
        c: 8,
        h: 12,
        w: 10,
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

    let before = fx.handle.compiled_module_count();
    for _ in 0..16 {
        assert_implicit_gemm_correct(&fx.handle, fx.sm, case, "cache_hit");
    }
    let after = fx.handle.compiled_module_count();

    assert_eq!(
        after - before,
        1,
        "16 identical convolutions must JIT one module, not {}",
        after - before
    );
}

/// Two convolution shapes that differ *only* in what the generator bakes into
/// the PTX as an immediate must not share a cached module.
///
/// `ImplicitGemmConv` folds the filter extent and the channels-per-group counts
/// into the instruction stream. Its kernel name — which is the cache key — used
/// to encode neither, so caching by name would have served the second shape the
/// first shape's compiled module and produced silently wrong output. Both
/// shapes run on the *same handle* (so they share one cache) and both are
/// checked against the CPU oracle, so a key regression fails loudly here.
#[test]
fn distinct_conv_shapes_on_one_handle_stay_correct() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let base = ConvCase {
        n: 1,
        c: 4,
        h: 9,
        w: 7,
        k: 4,
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
    // Same precision and layout, different code-gen immediates each time.
    let wider_channels = ConvCase { c: 12, ..base };
    let bigger_filter = ConvCase {
        r: 5,
        s: 5,
        pad_h: 2,
        pad_w: 2,
        ..base
    };
    let grouped = ConvCase {
        c: 8,
        k: 8,
        groups: 2,
        ..base
    };

    // Interleave so a stale cache entry from an earlier shape would be picked
    // up by a later one.
    for (case, tag) in [
        (base, "base"),
        (wider_channels, "wider_channels"),
        (bigger_filter, "bigger_filter"),
        (grouped, "grouped"),
        (base, "base_again"),
        (bigger_filter, "bigger_filter_again"),
        (wider_channels, "wider_channels_again"),
    ] {
        assert_implicit_gemm_correct(&fx.handle, fx.sm, case, tag);
    }

    // The key must be exactly as discriminating as the generated PTX -- no
    // more, no less. `grouped` (C=8, K=8, groups=2) has the *same*
    // channels-per-group pair as `base` (C=4, K=4, groups=1), and `groups`
    // itself is never baked in (the kernel derives the group index at runtime
    // from the `ocpg` immediate), so the two emit byte-identical PTX and
    // correctly share one module. Assert that directly rather than hard-coding
    // a count nobody can check.
    let ptx_of = |case: ConvCase| {
        ImplicitGemmConv::new(case.problem(PtxType::F32), fx.sm)
            .generate_ptx()
            .expect("ptx")
    };
    assert_eq!(
        ptx_of(base),
        ptx_of(grouped),
        "same channels-per-group and filter extent must emit identical PTX"
    );
    for other in [wider_channels, bigger_filter] {
        assert_ne!(
            ptx_of(base),
            ptx_of(other),
            "differing code-gen immediates must emit different PTX"
        );
    }

    // Hence three distinct modules for the four shapes, and every repeat a hit.
    assert_eq!(
        fx.handle.compiled_module_count(),
        3,
        "one module per distinct PTX, and repeats must hit the cache"
    );
}

/// The occupancy-derived launch configuration must still cover every output
/// element, across problem sizes that straddle warp and block boundaries.
///
/// The block size is now whatever `cuOccupancyMaxPotentialBlockSize` suggests
/// for the compiled kernel rather than a hard-coded 256, so the grid math is
/// re-verified on real hardware at sizes that are deliberately *not* multiples
/// of any plausible block size. An under-covering grid would leave the output
/// buffer's sentinel fill in place, which the oracle comparison catches.
#[test]
fn occupancy_launch_covers_every_output_element() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Output extents chosen so `n*k*out_h*out_w` lands just below, on, and just
    // above 32/64/256/1024 multiples.
    for (w, k, tag) in [
        (1u32, 1u32, "tiny_1_output"),
        (5, 1, "under_one_warp"),
        (8, 4, "exactly_32"),
        (9, 4, "just_over_32"),
        (16, 16, "exactly_256"),
        (17, 16, "just_over_256"),
        (33, 31, "prime_ish"),
        (64, 16, "exactly_1024"),
        (65, 16, "just_over_1024"),
    ] {
        let case = ConvCase {
            n: 1,
            c: 3,
            h: 4,
            w: w + 2,
            k,
            r: 3,
            s: 3,
            pad_h: 1,
            pad_w: 0,
            str_h: 1,
            str_w: 1,
            dil_h: 1,
            dil_w: 1,
            groups: 1,
            layout: TensorLayout::Nchw,
        };
        assert_implicit_gemm_correct(&fx.handle, fx.sm, case, tag);
    }
}
