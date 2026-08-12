//! WebGPU 2-D convolution benchmark — measures
//! `ComputeBackend::conv2d_forward` on a typical mid-size feature map: batch
//! 1, 16 input channels, 64×64 spatial, 32 output channels, a 3×3 kernel,
//! stride 1, padding 1 (so the output stays 64×64, `SAME`-style).
//!
//! This exercises the GPU dispatch path (`shader::conv2d_wgsl`), not the
//! CPU-fallback path, since batch × k_out × oh × ow = 1×32×64×64 = 131,072
//! is well within the fixed 2-D dispatch grid `conv2d_gpu_dispatch_grid`
//! supports.
//!
//! ## Platform behaviour
//!
//! * **A machine with a usable wgpu adapter** — the benchmark runs.
//! * **No adapter (headless CI)** — prints `skip: no WebGPU adapter
//!   (conv2d)` to stderr and returns before registering any benchmark
//!   closure with criterion.
//!
//! Run with:
//! ```bash
//! cargo bench -p oxicuda-webgpu --bench conv2d_bench
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_backend::ComputeBackend;
use oxicuda_webgpu::WebGpuBackend;

const N: usize = 1;
const C_IN: usize = 16;
const H: usize = 64;
const W: usize = 64;
const K_OUT: usize = 32;
const FH: usize = 3;
const FW: usize = 3;
const STRIDE: usize = 1;
const PAD: usize = 1;
// (H + 2*PAD - FH) / STRIDE + 1, same for W since the geometry is square.
const OH: usize = (H + 2 * PAD - FH) / STRIDE + 1;
const OW: usize = (W + 2 * PAD - FW) / STRIDE + 1;

/// GPU resources shared across iterations of the criterion loop.
struct Conv2dHarness {
    backend: WebGpuBackend,
    input: u64,
    filter: u64,
    output: u64,
}

fn upload_filled(backend: &WebGpuBackend, elems: usize, seed: u32) -> Option<u64> {
    let ptr = backend.alloc(elems * 4).ok()?;
    let fill: Vec<f32> = (0..elems)
        .map(|i| (((i as u32).wrapping_add(seed) % 13) as f32) * 0.01)
        .collect();
    let fill_bytes: Vec<u8> = fill.iter().flat_map(|v| v.to_le_bytes()).collect();
    backend.copy_htod(ptr, &fill_bytes).ok()?;
    Some(ptr)
}

fn try_setup() -> Option<Conv2dHarness> {
    let mut backend = WebGpuBackend::new();
    backend.init().ok()?;

    let input = upload_filled(&backend, N * C_IN * H * W, 1)?;
    let filter = upload_filled(&backend, K_OUT * C_IN * FH * FW, 7)?;
    let output = backend.alloc(N * K_OUT * OH * OW * 4).ok()?;

    Some(Conv2dHarness {
        backend,
        input,
        filter,
        output,
    })
}

fn bench_conv2d(criterion: &mut Criterion) {
    let harness = match try_setup() {
        Some(h) => h,
        None => {
            eprintln!("skip: no WebGPU adapter (conv2d)");
            return;
        }
    };

    let mut group = criterion.benchmark_group("webgpu_conv2d_1x16x64x64_k32x3x3");
    group.bench_function("conv2d_forward", |bencher| {
        bencher.iter(|| {
            let r = harness.backend.conv2d_forward(
                black_box(harness.input),
                &[N, C_IN, H, W],
                black_box(harness.filter),
                &[K_OUT, C_IN, FH, FW],
                harness.output,
                &[N, K_OUT, OH, OW],
                &[STRIDE, STRIDE],
                &[PAD, PAD],
            );
            black_box(r.ok());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_conv2d);
criterion_main!(benches);
