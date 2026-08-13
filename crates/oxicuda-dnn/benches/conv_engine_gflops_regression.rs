//! GFLOPS regression harness for the CUDA conv engines oxionnx-cuda's
//! `pick_engine` actually dispatches to -- [`Conv1x1`], [`DepthwiseConv`],
//! [`ImplicitGemmConv`] -- driven directly (not through
//! [`oxicuda_dnn::conv::conv_forward`], which production bypasses; see
//! `oxionnx-cuda/src/conv.rs`'s own module doc) on the real face-swap-
//! pipeline conv shapes measured in this session's roofline audit, printed
//! next to this device's measured FP32 FFMA issue-rate peak so a
//! nonsensical result (e.g. under 1000 GFLOPS on a >10 TFLOPS device) is
//! self-evidently a bug rather than a number that has to be looked up
//! against a spec sheet to notice.
//!
//! # Shape provenance
//!
//! Every shape below (channels, spatial extent, padding, stride) is taken
//! from a per-node classification dump of the actual ONNX graphs
//! (`det_10g.onnx` / `w600k_r50.onnx` / `inswapper_128.onnx`), not
//! estimated:
//!
//! - **SCRFD 28->56 @320x320** (`Conv_6_fused_activation`) and **ArcFace
//!   64->64 @112x112** (`Conv_3`): plain "same"-padded (`pad=1, stride=1`)
//!   3x3 convolutions, `engine=ImplicitGemm`.
//! - **InSwapper resblock 1024->1024 @34x34** (e.g. `Conv_62`): the
//!   generator's residual blocks use `ReflectionPad2d` (a separate,
//!   CPU-side `Pad` op, `32x32 -> 34x34`) followed by a *valid* (`pad=0`)
//!   3x3 convolution back down to `32x32`, `engine=ImplicitGemm`.
//! - **InSwapper decoder 1024->512 @64x64** (`Conv_590`) and **512->256
//!   @128x128** (`Conv_594`): the decoder instead upsamples with a bilinear
//!   `Resize` (`32x32 -> 64x64`, `64x64 -> 128x128`) followed by an
//!   ordinary *same*-padded (`pad=1, stride=1`) convolution -- **not** the
//!   reflect-pad + valid-conv pattern the resblocks use. An earlier
//!   exploratory pass through this same audit assumed the decoder followed
//!   the resblock convention too (hence a "1024->512 @66x66" /
//!   "512->256 @130x130", `pad=0`-implying shape appearing in prior scratch
//!   notes); the node dump shows otherwise, so this file uses the verified
//!   `pad=1` / `64x64` / `128x128` shapes. (The two framings happen to cost
//!   the *same* FLOPs -- both produce a `64x64` / `128x128` output -- so
//!   this correction changes provenance and input-side memory traffic, not
//!   the GFLOPS a correct implementation should hit.)
//! - **SCRFD Conv1x1 56->88 @80x80** (`Conv_37`): an unpadded, unit-stride
//!   1x1 convolution, `engine=Conv1x1` (per `pick_engine`'s own rule and the
//!   dump's recorded classification).
//! - **Synthetic depthwise 256ch @32x32**: neither SCRFD, ArcFace, nor
//!   InSwapper contains a single genuine depthwise convolution (`groups ==
//!   in_channels == out_channels`) -- zero `engine=Depthwise` nodes appear
//!   in the dump across all three models. This shape is included purely so
//!   the harness exercises all three `pick_engine` engines, not because it
//!   was measured in a real pipeline; it is labelled SYNTHETIC in its
//!   printed output so it is never mistaken for a measured face-pipeline
//!   number.
//!
//! Every reported GFLOPS figure counts only the convolution's own
//! multiply-adds (`2 * (Cin/groups) * R * S` per output element); no bias
//! epilogue is applied (`Conv1x1`/`DepthwiseConv` do not support one
//! in-kernel at all -- production adds it as a separate host-issued
//! elementwise pass -- and `ImplicitGemmConv` is called without one here so
//! all three engines report a directly comparable "raw conv compute only"
//! number).
//!
//! On macOS / no-GPU systems this binary prints a skip notice and exits 0
//! (`init()` reports `UnsupportedPlatform` / no device found).

use std::sync::Arc;

use oxicuda_dnn::conv::descriptor::ConvProblem;
use oxicuda_dnn::conv::fprop::direct::{Conv1x1, DepthwiseConv};
use oxicuda_dnn::conv::fprop::implicit_gemm::ImplicitGemmConv;
use oxicuda_dnn::handle::DnnHandle;
use oxicuda_dnn::types::{TensorDesc, TensorDescMut, TensorLayout};
use oxicuda_driver::{Context, Device, Event, Module, Stream};
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

/// Iterations timed per shape (after a separate, untimed warm-up call).
const ITERS: u32 = 20;

// ---------------------------------------------------------------------------
// FP32 FFMA issue-rate peak probe
// ---------------------------------------------------------------------------
//
// Same technique as this session's roofline-audit tooling: `chains`
// independent FFMA accumulation chains, `inner`-times unrolled, with no
// memory access anywhere in the timed loop, so the measurement is bound
// purely by the SM's FMA issue rate rather than by memory bandwidth or
// launch overhead. Grid size scales with the *live* SM count (queried via
// the driver) rather than a hardcoded figure, so this peak is meaningful on
// whatever device the bench actually runs on.

fn ffma_peak_ptx(chains: u32, inner: u32) -> String {
    let mut p = String::from(".version 7.1\n.target sm_86\n.address_size 64\n\n");
    p.push_str(".visible .entry ffma_peak(\n .param .u64 %pc,\n .param .u32 %pn\n)\n{\n");
    p.push_str(&format!("  .reg .f32 %a<{}>;\n", chains + 4));
    p.push_str("  .reg .b32 %r<8>;\n  .reg .b64 %rd<4>;\n  .reg .pred %p<2>;\n");
    p.push_str("  ld.param.u64 %rd0, [%pc];\n  ld.param.u32 %r0, [%pn];\n");
    p.push_str("  mov.u32 %r1, %tid.x;\n  cvt.rn.f32.u32 %a0, %r1;\n");
    for i in 1..chains {
        p.push_str(&format!("  add.f32 %a{i}, %a{}, 0f3F800000;\n", i - 1));
    }
    p.push_str(&format!("  mov.f32 %a{}, 0f3F800001;\n", chains));
    p.push_str(&format!("  mov.f32 %a{}, 0f3F7FFFFF;\n", chains + 1));
    p.push_str("  mov.u32 %r2, 0;\n$L:\n");
    for _ in 0..inner {
        for i in 0..chains {
            p.push_str(&format!(
                "  fma.rn.f32 %a{i}, %a{i}, %a{}, %a{};\n",
                chains,
                chains + 1
            ));
        }
    }
    p.push_str("  add.u32 %r2, %r2, 1;\n  setp.lt.u32 %p0, %r2, %r0;\n  @%p0 bra $L;\n");
    p.push_str("  mov.f32 %a0, %a0;\n");
    for i in 1..chains {
        p.push_str(&format!("  add.f32 %a0, %a0, %a{i};\n"));
    }
    p.push_str("  mul.wide.u32 %rd1, %r1, 4;\n  add.u64 %rd2, %rd0, %rd1;\n");
    p.push_str("  st.global.f32 [%rd2], %a0;\n  ret;\n}\n");
    p
}

/// Launches `kernel` `iters` times on `stream` (after one untimed warm-up
/// launch) and returns the average wall-clock seconds per launch, measured
/// via CUDA events bracketing the whole timed batch.
fn time_kernel_avg_secs(
    kernel: &Kernel,
    params: &LaunchParams,
    stream: &Stream,
    args: &(u64, u32),
    iters: u32,
) -> f64 {
    kernel.launch(params, stream, args).expect("warm-up launch");
    stream.synchronize().expect("warm-up sync");
    let start = Event::new().expect("start event");
    let end = Event::new().expect("end event");
    start.record(stream).expect("record start");
    for _ in 0..iters {
        kernel.launch(params, stream, args).expect("timed launch");
    }
    end.record(stream).expect("record end");
    end.synchronize().expect("sync end");
    f64::from(Event::elapsed_time(&start, &end).expect("elapsed_time")) / f64::from(iters) / 1000.0
}

/// Measures this device's FP32 FFMA issue-rate peak in GFLOPS.
fn measure_ffma_peak_gflops(stream: &Stream, sm_count: u32) -> f64 {
    let chains = 8u32;
    let inner = 8u32;
    let loops: u32 = 2000;
    const WARPS_PER_SM: u32 = 8; // 256 threads/SM -- enough to saturate the FMA pipes.

    let ptx = ffma_peak_ptx(chains, inner);
    let module = Arc::new(Module::from_ptx(&ptx).expect("ffma probe module"));
    let kernel = Kernel::from_module(module, "ffma_peak").expect("ffma probe kernel");

    let out = DeviceBuffer::<f32>::zeroed(1 << 16).expect("ffma probe output buffer");
    let grid = Dim3::new(sm_count * 4, 1, 1);
    let block = Dim3::new(WARPS_PER_SM * 32 / 4, 1, 1);
    let params = LaunchParams::new(grid, block);
    let total_warps = u64::from(grid.x) * u64::from(block.x) / 32;

    let secs = time_kernel_avg_secs(&kernel, &params, stream, &(out.as_device_ptr(), loops), 5);
    let flop = total_warps * u64::from(loops) * u64::from(chains) * u64::from(inner) * 32 * 2;
    flop as f64 / secs / 1e9
}

// ---------------------------------------------------------------------------
// Conv engine shapes
// ---------------------------------------------------------------------------

/// Which of `pick_engine`'s three engines a shape is routed to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Engine {
    Conv1x1,
    Depthwise,
    ImplicitGemm,
}

impl Engine {
    const fn label(self) -> &'static str {
        match self {
            Self::Conv1x1 => "Conv1x1",
            Self::Depthwise => "DepthwiseConv",
            Self::ImplicitGemm => "ImplicitGemmConv",
        }
    }
}

/// One conv shape to benchmark: NCHW input `[1, cin, h, w]`, filter
/// `[cout, cin/groups, r, s]`, `pad`/`stride` (symmetric, both spatial
/// dims), and which engine `pick_engine` would select for it.
struct ConvShape {
    tag: &'static str,
    engine: Engine,
    cin: u32,
    h: u32,
    w: u32,
    cout: u32,
    r: u32,
    s: u32,
    pad: u32,
    stride: u32,
    groups: u32,
}

const SHAPES: &[ConvShape] = &[
    ConvShape {
        tag: "SCRFD 28->56 3x3 pad1 @320x320",
        engine: Engine::ImplicitGemm,
        cin: 28,
        h: 320,
        w: 320,
        cout: 56,
        r: 3,
        s: 3,
        pad: 1,
        stride: 1,
        groups: 1,
    },
    ConvShape {
        tag: "ArcFace 64->64 3x3 pad1 @112x112",
        engine: Engine::ImplicitGemm,
        cin: 64,
        h: 112,
        w: 112,
        cout: 64,
        r: 3,
        s: 3,
        pad: 1,
        stride: 1,
        groups: 1,
    },
    ConvShape {
        tag: "InSwapper resblock 1024->1024 3x3 pad0 @34x34",
        engine: Engine::ImplicitGemm,
        cin: 1024,
        h: 34,
        w: 34,
        cout: 1024,
        r: 3,
        s: 3,
        pad: 0,
        stride: 1,
        groups: 1,
    },
    ConvShape {
        tag: "InSwapper decoder 1024->512 3x3 pad1 @64x64",
        engine: Engine::ImplicitGemm,
        cin: 1024,
        h: 64,
        w: 64,
        cout: 512,
        r: 3,
        s: 3,
        pad: 1,
        stride: 1,
        groups: 1,
    },
    ConvShape {
        tag: "InSwapper decoder 512->256 3x3 pad1 @128x128",
        engine: Engine::ImplicitGemm,
        cin: 512,
        h: 128,
        w: 128,
        cout: 256,
        r: 3,
        s: 3,
        pad: 1,
        stride: 1,
        groups: 1,
    },
    ConvShape {
        tag: "SCRFD 56->88 1x1 pad0 @80x80",
        engine: Engine::Conv1x1,
        cin: 56,
        h: 80,
        w: 80,
        cout: 88,
        r: 1,
        s: 1,
        pad: 0,
        stride: 1,
        groups: 1,
    },
    ConvShape {
        tag: "[SYNTHETIC, not from a real pipeline] depthwise 256ch 3x3 pad1 @32x32",
        engine: Engine::Depthwise,
        cin: 256,
        h: 32,
        w: 32,
        cout: 256,
        r: 3,
        s: 3,
        pad: 1,
        stride: 1,
        groups: 256,
    },
];

impl ConvShape {
    fn out_hw(&self) -> (u32, u32) {
        let oh = (self.h + 2 * self.pad - self.r) / self.stride + 1;
        let ow = (self.w + 2 * self.pad - self.s) / self.stride + 1;
        (oh, ow)
    }

    /// Total multiply-add FLOPs (counting both the multiply and the add) for
    /// one forward pass of this shape.
    fn flops(&self) -> u64 {
        let (oh, ow) = self.out_hw();
        let cin_per_group = self.cin / self.groups;
        2 * u64::from(cin_per_group)
            * u64::from(self.r)
            * u64::from(self.s)
            * u64::from(self.cout)
            * u64::from(oh)
            * u64::from(ow)
    }

    fn problem(&self) -> ConvProblem {
        ConvProblem {
            batch: 1,
            in_channels: self.cin,
            in_dims: vec![self.h, self.w],
            out_channels: self.cout,
            filter_dims: vec![self.r, self.s],
            padding: vec![self.pad, self.pad],
            stride: vec![self.stride, self.stride],
            dilation: vec![1, 1],
            groups: self.groups,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }
}

/// Runs one shape through its `pick_engine`-selected engine, returning the
/// measured GFLOPS.
fn bench_shape(handle: &DnnHandle, sm: SmVersion, shape: &ConvShape) -> f64 {
    let problem = shape.problem();
    let (oh, ow) = shape.out_hw();

    let in_buf = DeviceBuffer::<f32>::zeroed((shape.cin * shape.h * shape.w) as usize)
        .expect("input buffer");
    let filt_elems = (shape.cout * (shape.cin / shape.groups) * shape.r * shape.s) as usize;
    let filt_buf = DeviceBuffer::<f32>::zeroed(filt_elems).expect("filter buffer");
    let mut out_buf =
        DeviceBuffer::<f32>::zeroed((shape.cout * oh * ow) as usize).expect("output buffer");

    let input = TensorDesc::<f32>::nchw(&in_buf, 1, shape.cin, shape.h, shape.w)
        .expect("input tensor desc");
    let filter = TensorDesc::<f32>::nchw(
        &filt_buf,
        shape.cout,
        shape.cin / shape.groups,
        shape.r,
        shape.s,
    )
    .expect("filter tensor desc");
    let mut output = TensorDescMut::<f32>::nchw(&mut out_buf, 1, shape.cout, oh, ow)
        .expect("output tensor desc");

    let mut run: Box<dyn FnMut()> = match shape.engine {
        Engine::Conv1x1 => {
            let engine = Conv1x1::new(problem, sm).expect("Conv1x1::new");
            engine
                .execute(handle, &input, &filter, &mut output)
                .expect("Conv1x1 warm-up execute");
            Box::new(move || {
                engine
                    .execute(handle, &input, &filter, &mut output)
                    .expect("Conv1x1 execute");
            })
        }
        Engine::Depthwise => {
            let engine = DepthwiseConv::new(problem, sm).expect("DepthwiseConv::new");
            engine
                .execute(handle, &input, &filter, &mut output)
                .expect("DepthwiseConv warm-up execute");
            Box::new(move || {
                engine
                    .execute(handle, &input, &filter, &mut output)
                    .expect("DepthwiseConv execute");
            })
        }
        Engine::ImplicitGemm => {
            let engine = ImplicitGemmConv::new(problem, sm);
            engine
                .execute(handle, &input, &filter, None, &mut output)
                .expect("ImplicitGemmConv warm-up execute");
            Box::new(move || {
                engine
                    .execute(handle, &input, &filter, None, &mut output)
                    .expect("ImplicitGemmConv execute");
            })
        }
    };

    handle.stream().synchronize().expect("post warm-up sync");

    let start = Event::new().expect("start event");
    let end = Event::new().expect("end event");
    start.record(handle.stream()).expect("record start");
    for _ in 0..ITERS {
        run();
    }
    end.record(handle.stream()).expect("record end");
    end.synchronize().expect("sync end");
    let secs = f64::from(Event::elapsed_time(&start, &end).expect("elapsed_time"))
        / f64::from(ITERS)
        / 1000.0;

    shape.flops() as f64 / secs / 1e9
}

fn main() {
    if oxicuda_driver::init().is_err() {
        eprintln!("skip: no GPU (driver init failed)");
        return;
    }
    if !matches!(Device::count(), Ok(n) if n > 0) {
        eprintln!("skip: no GPU (Device::count <= 0)");
        return;
    }
    let device = match Device::get(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: no GPU (Device::get(0) failed)");
            return;
        }
    };
    let ctx = match Context::new(&device) {
        Ok(c) => Arc::new(c),
        Err(_) => {
            eprintln!("skip: no GPU (context creation failed)");
            return;
        }
    };
    let handle = match DnnHandle::new(&ctx) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("skip: no GPU (DnnHandle init failed)");
            return;
        }
    };
    let (major, minor) = device.compute_capability().expect("compute_capability");
    let sm = SmVersion::from_compute_capability(major, minor).unwrap_or(SmVersion::Sm80);
    let sm_count = u32::try_from(device.multiprocessor_count().unwrap_or(0)).unwrap_or(0);
    let sm_count = if sm_count > 0 { sm_count } else { 48 };

    let ffma_peak = measure_ffma_peak_gflops(handle.stream(), sm_count);
    println!(
        "device: {} ({} SMs, compute capability {major}.{minor})",
        sm.as_ptx_str(),
        sm_count
    );
    println!("measured FP32 FFMA issue-rate peak: {ffma_peak:8.1} GFLOPS");
    println!(
        "{:<55} {:>10} {:>12} {:>10}",
        "shape", "engine", "GFLOPS", "% of peak"
    );

    for shape in SHAPES {
        let gflops = bench_shape(&handle, sm, shape);
        println!(
            "{:<55} {:>10} {:>12.1} {:>9.2}%",
            shape.tag,
            shape.engine.label(),
            gflops,
            100.0 * gflops / ffma_peak,
        );
    }
}
