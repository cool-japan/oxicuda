//! Host-side tests for the tiled implicit-GEMM convolution engine.
//!
//! On-device numeric parity (against both the scalar engine and an `f64` CPU
//! reference, across the full shape sweep) lives in
//! `crate::gpu_tests::conv_tiled`; everything here is checkable without a
//! device and is about the *decision* and the *generated text*:
//!
//! * which problems the engine claims and which it declines — the decline
//!   list is the safety net for every configuration the kernel does not model,
//!   so each entry is pinned by a test;
//! * the cache-key contract — the entry name must move whenever any baked-in
//!   immediate does, or a cached module would be handed to the wrong problem;
//! * the structural invariants of the emitted PTX.

use super::*;
use crate::types::TensorLayout;

fn base_problem() -> ConvProblem {
    // InSwapper decoder 512 -> 256, 3x3 same-padded at 128x128: one of the
    // three dominant shapes in the face-swap pipeline.
    ConvProblem {
        batch: 1,
        in_channels: 512,
        in_dims: vec![128, 128],
        out_channels: 256,
        filter_dims: vec![3, 3],
        padding: vec![1, 1],
        stride: vec![1, 1],
        dilation: vec![1, 1],
        groups: 1,
        input_type: PtxType::F32,
        output_type: PtxType::F32,
        layout: TensorLayout::Nchw,
    }
}

// ---------------------------------------------------------------------------
// Plan selection
// ---------------------------------------------------------------------------

/// One benchmarked/asserted convolution shape, named so a failure identifies
/// the pipeline layer rather than a tuple position.
struct Shape {
    tag: &'static str,
    cin: u32,
    h: u32,
    w: u32,
    cout: u32,
    r: u32,
    s: u32,
    pad: u32,
    stride: u32,
}

impl Shape {
    /// A square, unit-stride 3x3 layer: `[1, cin, hw, hw] * [cout, cin, 3, 3]`.
    /// Every face-pipeline shape below is one of these; only the strided test
    /// case overrides a field.
    const fn conv3x3(tag: &'static str, cin: u32, hw: u32, cout: u32, pad: u32) -> Self {
        Self {
            tag,
            cin,
            h: hw,
            w: hw,
            cout,
            r: 3,
            s: 3,
            pad,
            stride: 1,
        }
    }

    fn problem(&self) -> ConvProblem {
        let mut p = base_problem();
        p.in_channels = self.cin;
        p.in_dims = vec![self.h, self.w];
        p.out_channels = self.cout;
        p.filter_dims = vec![self.r, self.s];
        p.padding = vec![self.pad, self.pad];
        p.stride = vec![self.stride, self.stride];
        p
    }
}

/// The shapes measured in this session's roofline audit of the three real ONNX
/// graphs -- the convolutions this engine exists for.
const FACE_PIPELINE: &[Shape] = &[
    Shape::conv3x3("InSwapper resblock 1024->1024 @34x34", 1024, 34, 1024, 0),
    Shape::conv3x3("InSwapper decoder 1024->512 @64x64", 1024, 64, 512, 1),
    Shape::conv3x3("InSwapper decoder 512->256 @128x128", 512, 128, 256, 1),
    Shape::conv3x3("SCRFD 28->56 @320x320", 28, 320, 56, 1),
    Shape::conv3x3("ArcFace 64->64 @112x112", 64, 112, 64, 1),
];

#[test]
fn claims_the_dominant_face_pipeline_shapes() {
    for shape in FACE_PIPELINE {
        let p = shape.problem();
        let plan = TiledConvPlan::for_problem(&p)
            .unwrap_or_else(|| panic!("{} must be claimed by the tiled engine", shape.tag));
        assert_eq!(
            plan.cfg.tile_k,
            shape.r * shape.s,
            "{}: tile_k must be a multiple of R*S",
            shape.tag
        );
        assert_eq!(
            plan.k_tiles(shape.cin) * plan.channels_per_ktile,
            shape.cin,
            "{}: the k-loop must cover every input channel exactly",
            shape.tag
        );
    }
}

/// Row-tile height must track the output-channel count: a 56-channel SCRFD
/// layer on a 128-row tile would waste more than half of every CTA.
#[test]
fn row_tile_tracks_the_output_channel_count() {
    for (cout, want_tile_m) in [(1024u32, 128u32), (256, 128), (96, 128), (64, 64), (32, 32)] {
        let mut p = base_problem();
        p.out_channels = cout;
        let plan = TiledConvPlan::for_problem(&p).expect("claimed");
        assert_eq!(
            plan.cfg.tile_m(),
            want_tile_m,
            "out_channels={cout} should pick a {want_tile_m}-row tile"
        );
    }
}

#[test]
fn declines_grouped_convolution() {
    let mut p = base_problem();
    p.groups = 4;
    assert!(TiledConvPlan::for_problem(&p).is_none());
    // ...including the depthwise extreme.
    p.groups = p.in_channels;
    p.out_channels = p.in_channels;
    assert!(TiledConvPlan::for_problem(&p).is_none());
}

#[test]
fn declines_non_f32_and_non_nchw() {
    let mut f64_problem = base_problem();
    f64_problem.input_type = PtxType::F64;
    f64_problem.output_type = PtxType::F64;
    assert!(TiledConvPlan::for_problem(&f64_problem).is_none());

    let mut nhwc = base_problem();
    nhwc.layout = TensorLayout::Nhwc;
    assert!(TiledConvPlan::for_problem(&nhwc).is_none());
}

#[test]
fn declines_below_the_profitability_thresholds() {
    // Shallow GEMM depth: 4 channels x 3x3 = 36 < MIN_GEMM_K.
    let mut shallow = base_problem();
    shallow.in_channels = 4;
    assert!(TiledConvPlan::for_problem(&shallow).is_none());

    // Few output pixels *and* too few CTAs to fill a device: 16x16 = 256
    // pixels over 256 output channels tiles into 2 x 2 = 4 CTAs.
    let mut small = base_problem();
    small.in_dims = vec![16, 16];
    assert!(TiledConvPlan::for_problem(&small).is_none());

    // ...but a narrow output that still produces enough CTAs is claimed: this
    // is the InSwapper residual block, the pipeline's heaviest layer.
    let mut narrow_but_deep = base_problem();
    narrow_but_deep.in_channels = 1024;
    narrow_but_deep.out_channels = 1024;
    narrow_but_deep.in_dims = vec![34, 34];
    narrow_but_deep.padding = vec![0, 0];
    let plan = TiledConvPlan::for_problem(&narrow_but_deep)
        .expect("1024 output pixels over 1024 channels is 64 CTAs, plenty");
    assert_eq!(plan.cfg.tile_m(), 128);

    // Too few output channels for even the 32-row tile.
    let mut narrow = base_problem();
    narrow.out_channels = 16;
    assert!(TiledConvPlan::for_problem(&narrow).is_none());
}

/// The engine must survive a 7x7 filter by shrinking the row tile rather than
/// emitting a kernel whose staging tiles do not fit in shared memory.
#[test]
fn large_filters_shrink_the_row_tile_instead_of_overflowing_shared_memory() {
    let mut p = base_problem();
    p.filter_dims = vec![7, 7];
    p.padding = vec![3, 3];
    let plan = TiledConvPlan::for_problem(&p).expect("7x7 must still be claimed");
    assert_eq!(plan.cfg.tile_k, 49);
    assert!(plan.cfg.tile_m() <= 64, "row tile must have shrunk");
    plan.cfg.validate().expect("shrunk tile must be valid");
}

/// A 1x1 convolution has `R*S == 1`, so a k-step must bundle several channels
/// to be worth a barrier — and the bundle size must divide the channel count.
#[test]
fn pointwise_convolution_bundles_channels_per_kstep() {
    let mut p = base_problem();
    p.filter_dims = vec![1, 1];
    p.padding = vec![0, 0];
    let plan = TiledConvPlan::for_problem(&p).expect("1x1 must be claimed");
    assert_eq!(plan.channels_per_ktile, 8);
    assert_eq!(plan.cfg.tile_k, 8);
    assert_eq!(plan.k_tiles(p.in_channels) * 8, p.in_channels);
}

#[test]
fn channels_per_ktile_always_divides_the_channel_count() {
    for in_channels in [1u32, 3, 7, 8, 12, 28, 56, 64, 96, 512, 1024] {
        for rs in [1u32, 2, 3, 4, 6, 9, 25, 49] {
            if let Some(ct) = channels_per_ktile(in_channels, rs) {
                assert_eq!(
                    in_channels % ct,
                    0,
                    "ct={ct} must divide C_in={in_channels} (rs={rs})"
                );
                assert!(
                    ct * rs >= 4,
                    "ct={ct} rs={rs}: a k-step under 4 taps is not worth a barrier"
                );
            }
        }
    }
}

#[test]
fn channels_per_ktile_rejects_degenerate_inputs() {
    assert!(channels_per_ktile(0, 9).is_none());
    assert!(channels_per_ktile(64, 0).is_none());
    // 3 channels with a 1x1 filter: no admissible bundle reaches 4 taps
    // (ct must divide 3, and 3 * 1 < 4).
    assert!(channels_per_ktile(3, 1).is_none());
}

// ---------------------------------------------------------------------------
// Cache-key contract
// ---------------------------------------------------------------------------

/// Every immediate baked into the instruction stream must move the entry name,
/// because the name *is* the compiled-module cache key. A name that collided
/// across two differently-generated kernels would hand one problem the other's
/// module: wrong results, no error, exactly the failure mode this codebase has
/// already shipped once.
#[test]
fn kernel_name_discriminates_every_baked_immediate() {
    let engine = |p: ConvProblem| {
        TiledImplicitGemmConv::new(p, SmVersion::Sm86)
            .expect("claimed")
            .kernel_name()
    };
    let base = engine(base_problem());

    let mut filter = base_problem();
    filter.filter_dims = vec![1, 1];
    filter.padding = vec![0, 0];
    assert_ne!(
        base,
        engine(filter),
        "filter extent is unrolled into the PTX"
    );

    let mut channels = base_problem();
    channels.in_channels = 1024;
    assert_ne!(base, engine(channels), "C_in is an address immediate");

    let mut out_channels = base_problem();
    out_channels.out_channels = 512;
    assert_ne!(base, engine(out_channels), "C_out is a bounds immediate");

    let mut pad = base_problem();
    pad.padding = vec![0, 0];
    assert_ne!(
        base,
        engine(pad),
        "padding is folded into the coordinate math"
    );

    let mut stride = base_problem();
    stride.stride = vec![2, 2];
    assert_ne!(
        base,
        engine(stride),
        "stride selects a different address form"
    );

    let mut dilation = base_problem();
    dilation.dilation = vec![2, 2];
    dilation.padding = vec![2, 2];
    assert_ne!(
        base,
        engine(dilation),
        "dilation selects a different address form"
    );
}

#[test]
fn kernel_name_is_the_emitted_entry() {
    let engine = TiledImplicitGemmConv::new(base_problem(), SmVersion::Sm86).expect("claimed");
    let ptx = engine.generate_ptx().expect("ptx generation");
    assert!(
        ptx.contains(&format!(".visible .entry {}", engine.kernel_name())),
        "kernel_name() must name the emitted entry point"
    );
}

// ---------------------------------------------------------------------------
// Emitted PTX
// ---------------------------------------------------------------------------

#[test]
fn generated_ptx_has_the_tiled_structure() {
    let engine = TiledImplicitGemmConv::new(base_problem(), SmVersion::Sm86).expect("claimed");
    let plan = engine.plan();
    let ptx = engine.generate_ptx().expect("ptx generation");

    // Staged, register-blocked mainloop.
    assert!(
        ptx.contains("ld.shared.v4.f32"),
        "fragments must be vector loads"
    );
    assert_eq!(
        ptx.matches("bar.sync").count(),
        2,
        "two barriers per k-step"
    );
    assert_eq!(
        ptx.matches("fma.rn.f32").count() as u32,
        plan.cfg.tile_k * plan.cfg.accumulators(),
        "the register-tile update must be fully unrolled"
    );

    // Both epilogue paths present.
    assert!(
        ptx.contains("st.global.v4.f32"),
        "interior CTAs must store 128 bits at a time"
    );
    assert!(
        ptx.contains("st.global.f32"),
        "boundary CTAs must fall back to predicated scalar stores"
    );

    // Shared tiles declared at the alignment the vector loads need.
    assert!(ptx.contains(".shared .align 16 .b8 conv_smem_a"));
    assert!(ptx.contains(".shared .align 16 .b8 conv_smem_b"));
    assert!(ptx.contains(".maxntid 256, 1, 1"));

    // Guarded bias epilogue, as the scalar engine has.
    assert!(
        ptx.contains("setp.ne.u64"),
        "bias pointer must be null-tested"
    );
}

/// The k-loop must contain no convolution index arithmetic: every division,
/// every bounds test and every base address is hoisted into the prologue. A
/// regression here is invisible to correctness tests and costs most of the
/// speedup, so it is asserted structurally — no `div`/`rem` may appear after
/// the first barrier.
#[test]
fn k_loop_contains_no_index_arithmetic() {
    let engine = TiledImplicitGemmConv::new(base_problem(), SmVersion::Sm86).expect("claimed");
    let ptx = engine.generate_ptx().expect("ptx generation");
    let loop_start = ptx.find("bar.sync").expect("staging barrier");
    let loop_end = ptx.rfind("bar.sync").expect("compute barrier");
    let body = &ptx[loop_start..loop_end];
    // `setp`/`bra` for the last-iteration prefetch guard are loop *control*,
    // not index arithmetic, so they are allowed; a multiply or a division is
    // not.
    for forbidden in [
        "div.u32",
        "rem.u32",
        "mad.lo.u32",
        "mul.lo.u32",
        "cvt.u64.u32",
    ] {
        assert!(
            !body.contains(forbidden),
            "`{forbidden}` inside the k-loop: index arithmetic must be hoisted\n{body}"
        );
    }
}

#[test]
fn stride_and_dilation_are_strength_reduced() {
    let mut unit = base_problem();
    unit.stride = vec![1, 1];
    unit.dilation = vec![1, 1];
    let unit_ptx = TiledImplicitGemmConv::new(unit, SmVersion::Sm86)
        .expect("claimed")
        .generate_ptx()
        .expect("ptx");

    let mut strided = base_problem();
    strided.stride = vec![2, 2];
    let strided_ptx = TiledImplicitGemmConv::new(strided, SmVersion::Sm86)
        .expect("claimed")
        .generate_ptx()
        .expect("ptx");

    // A unit stride must not emit a multiply for the output-position term; a
    // stride of 2 must.
    assert!(
        strided_ptx.matches("mul.lo.u32").count() > unit_ptx.matches("mul.lo.u32").count(),
        "stride 2 should cost multiplies that stride 1 does not"
    );
}

#[test]
fn ptxas_assembles_every_claimed_face_pipeline_shape() {
    use std::io::Write;
    use std::process::Command;

    let strided = Shape {
        stride: 2,
        ..Shape::conv3x3("strided 64->64 @160x160", 64, 160, 64, 1)
    };
    let shapes: Vec<&Shape> = FACE_PIPELINE
        .iter()
        .chain(std::iter::once(&strided))
        .collect();
    for shape in shapes {
        let (cin, cout, h, stride) = (shape.cin, shape.cout, shape.h, shape.stride);
        let Some(engine) = TiledImplicitGemmConv::new(shape.problem(), SmVersion::Sm86) else {
            continue;
        };
        let ptx = engine.generate_ptx().expect("ptx generation");

        let dir =
            std::env::temp_dir().join(format!("oxicuda_tiled_conv_{cin}_{cout}_{h}_{stride}"));
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let src = dir.join("conv.ptx");
        let Ok(mut f) = std::fs::File::create(&src) else {
            return;
        };
        if f.write_all(ptx.as_bytes()).is_err() {
            return;
        }
        drop(f);
        let Ok(result) = Command::new("ptxas")
            .arg("-arch=sm_86")
            .arg("-v")
            .arg(&src)
            .arg("-o")
            .arg(dir.join("conv.cubin"))
            .output()
        else {
            return; // No CUDA toolkit: structural assertions above still ran.
        };
        let stderr = String::from_utf8_lossy(&result.stderr).to_string();
        assert!(
            result.status.success(),
            "ptxas rejected {}:\n{stderr}",
            engine.kernel_name()
        );
        assert!(
            stderr.contains("0 bytes spill stores"),
            "{} spills registers:\n{stderr}",
            engine.kernel_name()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
