//! Winograd F(2x2, 3x3) vs implicit GEMM on InSwapper-128's 3x3 stride-1 layers.
//!
//! This is the measurement that decides
//! [`winograd_forward_implemented`](oxicuda_dnn::conv::algo_select::winograd_forward_implemented):
//! Winograd is numerically validated (see `gpu_tests::conv_winograd`), so the
//! only remaining question is whether it is *faster* than the engine it would
//! displace. Both engines are driven directly — not through
//! `conv::api::conv_forward` — so the comparison is between the kernels, not
//! between two dispatcher decisions.
//!
//! # Methodology
//!
//! * **Same inputs, same device buffers** for both engines.
//! * **Workspace allocated from the engine's own `workspace_bytes()`.** A
//!   Winograd call with a short workspace returns `WorkspaceRequired`
//!   *before touching the GPU*, so a benchmark that ignores that error times an
//!   early return and reports a fabulous, meaningless number. This crate has
//!   made that mistake before (see `conv2d_resnet50_layer3.rs`'s comment); here
//!   the requirement is queried up front and the buffer is reused across
//!   iterations.
//! * **Kernel cache warmed** by one untimed call per engine. Without this the
//!   first sample carries a ~194 us `cuModuleLoadData` JIT that has nothing to
//!   do with the convolution.
//! * **`stream().synchronize()` inside the timed closure**, so each sample is
//!   one complete convolution's latency rather than the rate at which launches
//!   can be pushed into an async queue.
//!
//! Winograd's advantage is a 2.25x reduction in *multiplies*; the transforms
//! and the extra workspace traffic are the price. Which side wins is a property
//! of the shape (large `C` and `K` amortise the transforms; small spatial
//! extents do not), which is exactly why the sweep spans a 16x range of
//! spatial extent at a 4x range of channel count.

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_dnn::DnnHandle;
use oxicuda_dnn::conv::descriptor::ConvProblem;
use oxicuda_dnn::conv::fprop::implicit_gemm::ImplicitGemmConv;
use oxicuda_dnn::conv::fprop::winograd::WinogradConv;
use oxicuda_dnn::types::{TensorDesc, TensorDescMut, TensorLayout};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

/// One benchmarked convolution: `[1, C, H, W] * [K, C, 3, 3]`, stride 1, pad 1.
struct Layer {
    label: &'static str,
    c: u32,
    hw: u32,
    k: u32,
}

/// The InSwapper-128 3x3 stride-1 stack, plus two shapes bracketing it so the
/// crossover (if any) is visible rather than inferred.
const LAYERS: &[Layer] = &[
    Layer {
        label: "c128_128x128",
        c: 128,
        hw: 128,
        k: 128,
    },
    Layer {
        label: "c256_64x64",
        c: 256,
        hw: 64,
        k: 256,
    },
    Layer {
        label: "c512_32x32",
        c: 512,
        hw: 32,
        k: 512,
    },
    Layer {
        label: "c512_16x16",
        c: 512,
        hw: 16,
        k: 512,
    },
    Layer {
        label: "c512_8x8",
        c: 512,
        hw: 8,
        k: 512,
    },
    Layer {
        label: "c64_256x256",
        c: 64,
        hw: 256,
        k: 64,
    },
    // Small end: where the transform + workspace overhead is supposed to
    // overtake the multiply saving. The `WINOGRAD_FLOP_THRESHOLD` in
    // `algo_select` is set from these points, not from a guess.
    Layer {
        label: "c64_64x64",
        c: 64,
        hw: 64,
        k: 64,
    },
    Layer {
        label: "c32_32x32",
        c: 32,
        hw: 32,
        k: 32,
    },
    Layer {
        label: "c16_16x16",
        c: 16,
        hw: 16,
        k: 16,
    },
    Layer {
        label: "c8_8x8",
        c: 8,
        hw: 8,
        k: 8,
    },
    // Points that bracket the crossover found above, holding the FLOP count
    // roughly constant while trading channels against spatial extent — so the
    // threshold can be expressed in FLOPs rather than in any single dimension.
    Layer {
        label: "c16_32x32",
        c: 16,
        hw: 32,
        k: 16,
    },
    Layer {
        label: "c64_16x16",
        c: 64,
        hw: 16,
        k: 64,
    },
    Layer {
        label: "c256_8x8",
        c: 256,
        hw: 8,
        k: 256,
    },
];

const PAD: u32 = 1;

fn problem_for(layer: &Layer) -> ConvProblem {
    ConvProblem {
        batch: 1,
        in_channels: layer.c,
        in_dims: vec![layer.hw, layer.hw],
        out_channels: layer.k,
        filter_dims: vec![3, 3],
        padding: vec![PAD, PAD],
        stride: vec![1, 1],
        dilation: vec![1, 1],
        groups: 1,
        input_type: PtxType::F32,
        output_type: PtxType::F32,
        layout: TensorLayout::Nchw,
    }
}

/// Deterministic pseudo-random fill; the values only need to be finite and
/// non-degenerate, since neither engine's runtime depends on the data.
fn filled(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((state >> 32) as f32) / 4_294_967_296.0) - 0.5
        })
        .collect()
}

fn bench_winograd_vs_implicit_gemm(c: &mut Criterion) {
    if oxicuda_driver::init().is_err() {
        eprintln!("skip: no GPU (driver init failed)");
        return;
    }
    if !matches!(Device::count(), Ok(n) if n > 0) {
        eprintln!("skip: no GPU (Device::count <= 0)");
        return;
    }
    let Ok(device) = Device::get(0) else {
        eprintln!("skip: no GPU (Device::get(0) failed)");
        return;
    };
    let Ok(ctx) = Context::new(&device) else {
        eprintln!("skip: no GPU (context creation failed)");
        return;
    };
    let ctx = Arc::new(ctx);
    let Ok(handle) = DnnHandle::new(&ctx) else {
        eprintln!("skip: no GPU (DnnHandle init failed)");
        return;
    };
    let sm = handle.sm_version();

    let mut group = c.benchmark_group("dnn_conv3x3_winograd_vs_implicit_gemm");

    for layer in LAYERS {
        let problem = problem_for(layer);
        let out_hw = layer.hw + 2 * PAD - 2;
        let in_elems = (layer.c * layer.hw * layer.hw) as usize;
        let fil_elems = (layer.k * layer.c * 9) as usize;
        let out_elems = (layer.k * out_hw * out_hw) as usize;

        let Ok(in_buf) = DeviceBuffer::from_host(&filled(in_elems, 0x1234_5678)) else {
            eprintln!("skip {}: input alloc failed", layer.label);
            continue;
        };
        let Ok(fil_buf) = DeviceBuffer::from_host(&filled(fil_elems, 0x9abc_def0)) else {
            eprintln!("skip {}: filter alloc failed", layer.label);
            continue;
        };
        let Ok(mut out_buf) = DeviceBuffer::<f32>::zeroed(out_elems) else {
            eprintln!("skip {}: output alloc failed", layer.label);
            continue;
        };

        let input = TensorDesc::<f32>::nchw(&in_buf, 1, layer.c, layer.hw, layer.hw)
            .expect("input descriptor");
        let filter =
            TensorDesc::<f32>::nchw(&fil_buf, layer.k, layer.c, 3, 3).expect("filter descriptor");
        let mut output = TensorDescMut::<f32>::nchw(&mut out_buf, 1, layer.k, out_hw, out_hw)
            .expect("output descriptor");

        let winograd = WinogradConv::new(problem.clone(), sm).expect("winograd engine");
        let implicit = ImplicitGemmConv::new(problem, sm);

        // Query, never guess: an under-sized workspace makes `execute` return
        // `WorkspaceRequired` without launching anything.
        let ws_bytes = winograd.workspace_bytes().expect("winograd workspace size");
        let Ok(mut workspace) = DeviceBuffer::<u8>::alloc(ws_bytes) else {
            eprintln!(
                "skip {}: workspace alloc failed ({ws_bytes} bytes)",
                layer.label
            );
            continue;
        };

        // Warm the kernel cache for both engines outside the timed loop.
        winograd
            .execute(&handle, &input, &filter, &mut output, &mut workspace)
            .expect("winograd warm-up");
        implicit
            .execute(&handle, &input, &filter, None, &mut output)
            .expect("implicit gemm warm-up");
        handle.stream().synchronize().expect("warm-up synchronize");

        // Throughput in output elements/sec; multiply by 2*C*9 FLOPs per
        // element offline for the effective GFLOPS of the *direct* algorithm
        // (Winograd does the same maths with fewer multiplies, so its
        // "effective" rate is the fair cross-engine comparison).
        group.throughput(Throughput::Elements(out_elems as u64));

        group.bench_with_input(
            BenchmarkId::new("winograd_f2x3", layer.label),
            &(),
            |b, ()| {
                b.iter(|| {
                    winograd
                        .execute(&handle, &input, &filter, &mut output, &mut workspace)
                        .expect("winograd execute");
                    handle.stream().synchronize().expect("synchronize");
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("implicit_gemm", layer.label),
            &(),
            |b, ()| {
                b.iter(|| {
                    implicit
                        .execute(&handle, &input, &filter, None, &mut output)
                        .expect("implicit gemm execute");
                    handle.stream().synchronize().expect("synchronize");
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_winograd_vs_implicit_gemm);
criterion_main!(benches);
