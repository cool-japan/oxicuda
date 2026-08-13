//! On-device numeric validation for the CTA-tiled implicit-GEMM convolution.
//!
//! This kernel replaces the scalar one on every shape that dominates the
//! face-swap pipeline's runtime, so it is checked two independent ways, and
//! both are required:
//!
//! 1. **Against an `f64` CPU re-derivation** (`conv2d_ref`, shared with
//!    `conv_fprop`) on a sweep of small shapes that covers every structural
//!    edge the tiling has: partial row tiles, partial column tiles, a `P*Q`
//!    that is not a multiple of 4 (which forces the scalar store path),
//!    padding, stride, dilation, `batch > 1`, 1x1 / 5x5 / 7x7 filter volumes,
//!    and a channel count that does not divide the k-step evenly.
//! 2. **Against the scalar engine on the device**, element for element, on the
//!    *real* face-pipeline shapes — which are far too large for a CPU oracle
//!    (38 GFLOP each) but cost milliseconds on the GPU. This is the comparison
//!    that matters for the production claim, because the scalar engine is what
//!    the pipeline used to run and what `OXIONNX_CUDA_VERIFY` shadows.
//!
//! The two kernels are expected to agree to *far* better than the 1e-5
//! relative-L2 bar: the tiled mainloop accumulates its k-slices in increasing
//! `(c, r, s)` order, which is exactly the scalar kernel's loop nesting, so the
//! two perform the same `fma.rn.f32` sequence on the same values. The
//! assertions below are set an order of magnitude tighter than the bar for
//! that reason -- a mainloop that reassociated the sum would still pass 1e-5
//! and should not.
//!
//! Every test skips cleanly when no CUDA device is present.

use super::*;

use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

use super::conv_fprop::{ConvCase, conv2d_ref, make_desc, make_desc_mut};
use crate::conv::fprop::implicit_gemm::ImplicitGemmConv;
use crate::conv::fprop::tiled_implicit_gemm::{TiledConvPlan, TiledImplicitGemmConv};
use crate::types::TensorLayout;

/// Relative-L2 bound for the tiled kernel against an `f64` CPU reference.
///
/// The task's correctness bar is 1e-5; f32 accumulation over the deepest
/// sweep shape (`C*R*S = 400`) contributes on the order of `sqrt(400) * 6e-8`,
/// so 1e-6 is comfortably above the floating-point floor and still an order of
/// magnitude inside the bar.
const CPU_ORACLE_REL_L2: f64 = 1e-6;

/// Relative-L2 bound for the tiled kernel against the scalar engine.
///
/// Same operation order (see the module docs), so this is a near-equality
/// check, not a tolerance.
const ENGINE_PARITY_REL_L2: f64 = 1e-7;

/// Builds a standard NCHW f32 case with a square `f x f` filter, unit stride
/// and unit dilation. The few cases that need a stride or dilation override
/// the field afterwards.
fn case(n: u32, c: u32, h: u32, w: u32, k: u32, f: u32, pad: u32) -> ConvCase {
    ConvCase {
        n,
        c,
        h,
        w,
        k,
        r: f,
        s: f,
        pad_h: pad,
        pad_w: pad,
        str_h: 1,
        str_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
        layout: TensorLayout::Nchw,
    }
}

// ---------------------------------------------------------------------------
// CPU-oracle sweep
// ---------------------------------------------------------------------------

/// Runs one case on the tiled engine and checks it against `conv2d_ref`.
///
/// Also asserts the engine *is* the tiled one: a silent fall-back to the
/// scalar kernel would make every numeric assertion here pass while testing
/// nothing.
fn check_against_cpu(fx: &GpuFixture, case: ConvCase, with_bias: bool, tag: &str) {
    let problem = case.problem(PtxType::F32);
    let engine = TiledImplicitGemmConv::new(problem, fx.sm)
        .unwrap_or_else(|| panic!("{tag}: the tiled engine must claim this shape"));

    let (out_h, out_w) = case.out_hw();
    let in_n = (case.n * case.c * case.h * case.w) as usize;
    let fil_n = (case.k * case.c * case.r * case.s) as usize;
    let out_n = (case.n * case.k * out_h * out_w) as usize;

    let mut lcg = Lcg::new(0x7ea1_c0de_0000_0001);
    let input: Vec<f32> = (0..in_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let filter: Vec<f32> = (0..fil_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let bias: Vec<f32> = if with_bias {
        (0..case.k as usize)
            .map(|_| lcg.range_f32(-0.5, 0.5))
            .collect()
    } else {
        Vec::new()
    };

    let in_buf = DeviceBuffer::from_host(&input).expect("upload input");
    let fil_buf = DeviceBuffer::from_host(&filter).expect("upload filter");
    let bias_buf = with_bias.then(|| DeviceBuffer::from_host(&bias).expect("upload bias"));
    // A sentinel the kernel must overwrite everywhere: an unwritten element is
    // a coverage bug the reference comparison would otherwise report as a
    // wildly wrong value without saying why.
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f32; out_n]).expect("alloc output");

    let in_desc = make_desc(&in_buf, TensorLayout::Nchw);
    let fil_desc = make_desc(&fil_buf, TensorLayout::Nchw);
    let bias_desc = bias_buf.as_ref().map(|b| make_desc(b, TensorLayout::Nchw));
    let mut out_desc = make_desc_mut(&out_buf, TensorLayout::Nchw);

    engine
        .execute(
            &fx.handle,
            &in_desc,
            &fil_desc,
            bias_desc.as_ref(),
            &mut out_desc,
        )
        .unwrap_or_else(|e| panic!("{tag}: tiled execute failed: {e}"));
    fx.stream().synchronize().expect("synchronize");

    let mut gpu = vec![0.0f32; out_n];
    out_buf.copy_to_host(&mut gpu).expect("copy output");
    assert!(
        gpu.iter().all(|&v| v != -987.0),
        "{tag}: the kernel left at least one output element unwritten"
    );

    let in64: Vec<f64> = input.iter().map(|&x| f64::from(x)).collect();
    let fil64: Vec<f64> = filter.iter().map(|&x| f64::from(x)).collect();
    let bias64: Option<Vec<f64>> = with_bias.then(|| bias.iter().map(|&x| f64::from(x)).collect());
    let want = conv2d_ref(case, &in64, &fil64, bias64.as_deref());

    let err = rel_l2_error(&gpu, &want);
    assert!(
        err < CPU_ORACLE_REL_L2,
        "{tag}: relative L2 error {err:e} exceeds {CPU_ORACLE_REL_L2:e}"
    );
}

#[test]
fn tiled_conv_matches_cpu_oracle_across_the_structural_sweep() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // (tag, case) — every entry exercises a distinct structural edge of the
    // tiling, named so a failure identifies the mechanism rather than a shape.
    let sweep: [(&str, ConvCase); 12] = [
        // Exact tiles: C_out == tile_m, P*Q a multiple of both 128 and 4.
        ("exact tiles 3x3 pad1", case(1, 8, 64, 64, 32, 3, 1)),
        // Partial row tile: 40 output channels over a 32-row tile.
        ("partial row tile", case(1, 8, 64, 64, 40, 3, 1)),
        // Partial column tile AND P*Q % 4 != 0 -> forces the scalar store path.
        (
            "partial column tile, unaligned P*Q",
            case(1, 8, 65, 67, 32, 3, 1),
        ),
        // Unpadded (valid) convolution, the InSwapper resblock convention.
        ("pad 0", case(1, 8, 66, 66, 32, 3, 0)),
        // Strided.
        ("stride 2", {
            let mut c = case(1, 8, 129, 129, 32, 3, 1);
            c.str_h = 2;
            c.str_w = 2;
            c
        }),
        // Dilated.
        ("dilation 2", {
            let mut c = case(1, 8, 68, 68, 32, 3, 2);
            c.dil_h = 2;
            c.dil_w = 2;
            c
        }),
        // Batch > 1, with P*Q a multiple of 4 (vector store path).
        ("batch 2, aligned P*Q", case(2, 8, 46, 46, 32, 3, 1)),
        // Batch > 1 with P*Q NOT a multiple of 4: a 4-wide column group would
        // straddle two batch items, which is exactly what the store-path test
        // exists to prevent.
        ("batch 2, unaligned P*Q", case(2, 8, 45, 45, 32, 3, 1)),
        // Pointwise: R*S == 1, so a k-step bundles 8 channels.
        ("1x1 pointwise", case(1, 64, 64, 64, 32, 1, 0)),
        // Pointwise with a channel count that does not divide 8: the k-step
        // bundle must shrink to a divisor.
        ("1x1 pointwise, C=68", case(1, 68, 64, 64, 32, 1, 0)),
        // Larger filter volumes: tile_k = 25 and 49.
        ("5x5", case(1, 16, 64, 64, 32, 5, 2)),
        ("7x7", case(1, 8, 64, 64, 32, 7, 3)),
    ];

    for (tag, c) in sweep {
        check_against_cpu(&fx, c, false, tag);
    }
}

#[test]
fn tiled_conv_bias_epilogue_matches_cpu_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Both store paths, both with a bias: the bias is added before either.
    check_against_cpu(
        &fx,
        case(1, 8, 64, 64, 32, 3, 1),
        true,
        "bias, vector store",
    );
    check_against_cpu(
        &fx,
        case(1, 8, 65, 67, 40, 3, 1),
        true,
        "bias, scalar store + partial tiles",
    );
}

// ---------------------------------------------------------------------------
// Scalar-engine parity on the real face-pipeline shapes
// ---------------------------------------------------------------------------

/// Runs one case on both engines and compares element for element.
fn check_against_scalar(fx: &GpuFixture, case: ConvCase, with_bias: bool, tag: &str) {
    let problem = case.problem(PtxType::F32);
    let tiled = TiledImplicitGemmConv::new(problem.clone(), fx.sm)
        .unwrap_or_else(|| panic!("{tag}: the tiled engine must claim this shape"));
    let scalar = ImplicitGemmConv::scalar_only(problem, fx.sm);
    assert!(
        !scalar.uses_tiled_kernel(),
        "{tag}: the baseline must be the scalar kernel"
    );

    let (out_h, out_w) = case.out_hw();
    let in_n = (case.n * case.c * case.h * case.w) as usize;
    let fil_n = (case.k * case.c * case.r * case.s) as usize;
    let out_n = (case.n * case.k * out_h * out_w) as usize;

    let mut lcg = Lcg::new(0x51de_0bad_0000_0007);
    let input: Vec<f32> = (0..in_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
    let filter: Vec<f32> = (0..fil_n).map(|_| lcg.range_f32(-0.5, 0.5)).collect();
    let bias: Vec<f32> = if with_bias {
        (0..case.k as usize)
            .map(|_| lcg.range_f32(-0.5, 0.5))
            .collect()
    } else {
        Vec::new()
    };

    let in_buf = DeviceBuffer::from_host(&input).expect("upload input");
    let fil_buf = DeviceBuffer::from_host(&filter).expect("upload filter");
    let bias_buf = with_bias.then(|| DeviceBuffer::from_host(&bias).expect("upload bias"));
    let tiled_buf = DeviceBuffer::from_host(&vec![-987.0f32; out_n]).expect("alloc tiled output");
    let scalar_buf = DeviceBuffer::from_host(&vec![-987.0f32; out_n]).expect("alloc scalar output");

    let in_desc = make_desc(&in_buf, TensorLayout::Nchw);
    let fil_desc = make_desc(&fil_buf, TensorLayout::Nchw);
    let bias_desc = bias_buf.as_ref().map(|b| make_desc(b, TensorLayout::Nchw));

    let mut tiled_desc = make_desc_mut(&tiled_buf, TensorLayout::Nchw);
    tiled
        .execute(
            &fx.handle,
            &in_desc,
            &fil_desc,
            bias_desc.as_ref(),
            &mut tiled_desc,
        )
        .unwrap_or_else(|e| panic!("{tag}: tiled execute failed: {e}"));

    let mut scalar_desc = make_desc_mut(&scalar_buf, TensorLayout::Nchw);
    scalar
        .execute(
            &fx.handle,
            &in_desc,
            &fil_desc,
            bias_desc.as_ref(),
            &mut scalar_desc,
        )
        .unwrap_or_else(|e| panic!("{tag}: scalar execute failed: {e}"));
    fx.stream().synchronize().expect("synchronize");

    let mut got = vec![0.0f32; out_n];
    let mut want = vec![0.0f32; out_n];
    tiled_buf.copy_to_host(&mut got).expect("copy tiled");
    scalar_buf.copy_to_host(&mut want).expect("copy scalar");
    assert!(
        got.iter().all(|&v| v != -987.0),
        "{tag}: the tiled kernel left at least one output element unwritten"
    );

    let want64: Vec<f64> = want.iter().map(|&x| f64::from(x)).collect();
    let err = rel_l2_error(&got, &want64);
    assert!(
        err < ENGINE_PARITY_REL_L2,
        "{tag}: tiled-vs-scalar relative L2 error {err:e} exceeds {ENGINE_PARITY_REL_L2:e}"
    );
    // Element-wise too: an L2 norm over millions of elements can hide a
    // single catastrophically wrong output (which is what a boundary-tile bug
    // looks like).
    assert_close_f32(&got, &want, 1e-5, 1e-5, tag);
}

#[test]
fn tiled_conv_matches_scalar_engine_on_the_face_pipeline_shapes() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // The shapes measured in this session's roofline audit of the three real
    // ONNX graphs. Sizes are the production ones: these are the convolutions
    // the tiled engine exists for, so they are checked at full size rather
    // than scaled down.
    let shapes: [(&str, ConvCase); 5] = [
        (
            "InSwapper resblock 1024->1024 3x3 pad0 @34x34",
            case(1, 1024, 34, 34, 1024, 3, 0),
        ),
        (
            "InSwapper decoder 1024->512 3x3 pad1 @64x64",
            case(1, 1024, 64, 64, 512, 3, 1),
        ),
        (
            "InSwapper decoder 512->256 3x3 pad1 @128x128",
            case(1, 512, 128, 128, 256, 3, 1),
        ),
        (
            "SCRFD 28->56 3x3 pad1 @320x320",
            case(1, 28, 320, 320, 56, 3, 1),
        ),
        (
            "ArcFace 64->64 3x3 pad1 @112x112",
            case(1, 64, 112, 112, 64, 3, 1),
        ),
    ];

    for (tag, c) in shapes {
        check_against_scalar(&fx, c, false, tag);
    }
    // ...and one of them with the bias epilogue, which production uses.
    check_against_scalar(
        &fx,
        case(1, 64, 112, 112, 64, 3, 1),
        true,
        "ArcFace 64->64 with bias",
    );
}

// ---------------------------------------------------------------------------
// Fall-back behaviour
// ---------------------------------------------------------------------------

/// Every configuration the tiling declines must still compute the right
/// answer, on the scalar kernel, through the same public entry point.
///
/// This is the safety net the whole design rests on, so it is checked on the
/// device rather than inferred from `TiledConvPlan::for_problem` returning
/// `None`.
#[test]
fn declined_shapes_fall_back_to_the_scalar_kernel_and_stay_correct() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let declined: [(&str, ConvCase); 4] = [
        // Grouped: the tiling models groups == 1 only.
        ("grouped", {
            let mut c = case(1, 16, 32, 32, 16, 3, 1);
            c.groups = 2;
            c
        }),
        // Single input channel: C*R*S = 9, far below the k-loop floor.
        ("C=1", case(1, 1, 64, 64, 32, 3, 1)),
        // Single output channel: below the row-tile floor.
        ("K=1", case(1, 64, 64, 64, 1, 3, 1)),
        // NHWC: the transposed GEMM orientation is NCHW-specific.
        ("NHWC", {
            let mut c = case(1, 64, 64, 64, 32, 3, 1);
            c.layout = TensorLayout::Nhwc;
            c
        }),
    ];

    for (tag, c) in declined {
        let problem = c.problem(PtxType::F32);
        assert!(
            TiledConvPlan::for_problem(&problem).is_none(),
            "{tag}: this shape is supposed to be declined"
        );
        let engine = ImplicitGemmConv::new(problem, fx.sm);
        assert!(
            !engine.uses_tiled_kernel(),
            "{tag}: a declined shape must not reach the tiled kernel"
        );

        let (out_h, out_w) = c.out_hw();
        let icpg = c.c / c.groups;
        let in_n = (c.n * c.c * c.h * c.w) as usize;
        let fil_n = (c.k * icpg * c.r * c.s) as usize;
        let out_n = (c.n * c.k * out_h * out_w) as usize;

        let mut lcg = Lcg::new(0x0fa1_1bac_0000_0003);
        let input: Vec<f32> = (0..in_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
        let filter: Vec<f32> = (0..fil_n).map(|_| lcg.range_f32(-1.0, 1.0)).collect();
        let in_buf = DeviceBuffer::from_host(&input).expect("upload input");
        let fil_buf = DeviceBuffer::from_host(&filter).expect("upload filter");
        let out_buf = DeviceBuffer::from_host(&vec![-987.0f32; out_n]).expect("alloc output");

        let in_desc = make_desc(&in_buf, c.layout);
        let fil_desc = make_desc(&fil_buf, c.layout);
        let mut out_desc = make_desc_mut(&out_buf, c.layout);
        engine
            .execute(&fx.handle, &in_desc, &fil_desc, None, &mut out_desc)
            .unwrap_or_else(|e| panic!("{tag}: fallback execute failed: {e}"));
        fx.stream().synchronize().expect("synchronize");

        let mut gpu = vec![0.0f32; out_n];
        out_buf.copy_to_host(&mut gpu).expect("copy output");
        let in64: Vec<f64> = input.iter().map(|&x| f64::from(x)).collect();
        let fil64: Vec<f64> = filter.iter().map(|&x| f64::from(x)).collect();
        let want = conv2d_ref(c, &in64, &fil64, None);
        let err = rel_l2_error(&gpu, &want);
        assert!(
            err < CPU_ORACLE_REL_L2,
            "{tag}: fallback relative L2 error {err:e} exceeds {CPU_ORACLE_REL_L2:e}"
        );
    }
}

/// The public engine must route the pipeline's shapes to the tiled kernel
/// without any caller opt-in — that routing is the entire deliverable, and a
/// regression in it is otherwise only visible as a performance loss.
#[test]
fn public_engine_routes_the_pipeline_shapes_to_the_tiled_kernel() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let shapes = [
        case(1, 1024, 34, 34, 1024, 3, 0),
        case(1, 1024, 64, 64, 512, 3, 1),
        case(1, 512, 128, 128, 256, 3, 1),
        case(1, 28, 320, 320, 56, 3, 1),
        case(1, 64, 112, 112, 64, 3, 1),
    ];
    for c in shapes {
        let engine = ImplicitGemmConv::new(c.problem(PtxType::F32), fx.sm);
        assert!(
            engine.uses_tiled_kernel(),
            "{}x{} {}x{} @{}x{} must route to the tiled kernel",
            c.c,
            c.k,
            c.r,
            c.s,
            c.h,
            c.w
        );
    }
}

/// A repeated call must be a cache hit: the kernel bakes a long list of
/// immediates into its entry name, and a name that failed to be stable would
/// re-JIT a large module on every convolution of every frame.
#[test]
fn repeated_execution_reuses_the_compiled_module() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let c = case(1, 64, 64, 64, 32, 3, 1);
    let engine = TiledImplicitGemmConv::new(c.problem(PtxType::F32), fx.sm).expect("claimed shape");

    let (out_h, out_w) = c.out_hw();
    let in_buf = DeviceBuffer::<f32>::zeroed((c.c * c.h * c.w) as usize).expect("input");
    let fil_buf = DeviceBuffer::<f32>::zeroed((c.k * c.c * c.r * c.s) as usize).expect("filter");
    let out_buf = DeviceBuffer::<f32>::zeroed((c.k * out_h * out_w) as usize).expect("output");
    let in_desc = make_desc(&in_buf, TensorLayout::Nchw);
    let fil_desc = make_desc(&fil_buf, TensorLayout::Nchw);
    let mut out_desc = make_desc_mut(&out_buf, TensorLayout::Nchw);

    engine
        .execute(&fx.handle, &in_desc, &fil_desc, None, &mut out_desc)
        .expect("first execute");
    let after_first = fx.handle.compiled_module_count();
    for _ in 0..4 {
        engine
            .execute(&fx.handle, &in_desc, &fil_desc, None, &mut out_desc)
            .expect("repeat execute");
    }
    fx.stream().synchronize().expect("synchronize");
    assert_eq!(
        fx.handle.compiled_module_count(),
        after_first,
        "repeated execution must not compile a second module"
    );
}
