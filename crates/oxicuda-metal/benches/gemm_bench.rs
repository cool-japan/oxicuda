//! Metal GEMM benchmark — measures `ComputeBackend::gemm` (the tiled,
//! runtime-transpose `gemm_v2` kernel described in `oxicuda_metal::backend`)
//! at two square problem sizes, 256×256×256 and 1024×1024×1024, both
//! NoTrans/NoTrans with natural (packed) leading dimensions.
//!
//! ## Platform behaviour
//!
//! * **macOS with a Metal-capable GPU** — both groups run.
//! * **No Metal device (headless CI, non-macOS)** — each bench function
//!   prints `skip: no Metal device (gemm_<n>)` to stderr and returns before
//!   registering any benchmark closure with criterion. The skip guard is the
//!   first thing every bench function in this file does.
//!
//! Run with:
//! ```bash
//! cargo bench -p oxicuda-metal --bench gemm_bench
//! # or, to just prove the benches execute without a full measured run:
//! cargo bench -p oxicuda-metal --bench gemm_bench -- --test
//! ```

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_backend::{BackendTranspose, ComputeBackend};
use oxicuda_metal::MetalBackend;

/// GPU resources shared across iterations of the criterion loop for one
/// problem size.
struct GemmHarness {
    backend: MetalBackend,
    a: u64,
    b: u64,
    c: u64,
    n: usize,
}

/// Initialise the backend and allocate/fill three `n`×`n` `f32` row-major
/// buffers. Returns `None` on any platform without a usable Metal device.
fn try_setup(n: usize) -> Option<GemmHarness> {
    let mut backend = MetalBackend::new();
    backend.init().ok()?;

    let elems = n * n;
    let bytes = elems * 4;
    let a = backend.alloc(bytes).ok()?;
    let b = backend.alloc(bytes).ok()?;
    let c = backend.alloc(bytes).ok()?;

    // Small deterministic values so accumulation over n terms stays finite
    // and well away from f32 overflow even at n = 1024.
    let fill: Vec<f32> = (0..elems).map(|i| ((i % 17) as f32) * 0.01).collect();
    let fill_bytes: Vec<u8> = fill.iter().flat_map(|v| v.to_le_bytes()).collect();
    backend.copy_htod(a, &fill_bytes).ok()?;
    backend.copy_htod(b, &fill_bytes).ok()?;
    backend.copy_htod(c, &vec![0u8; bytes]).ok()?;

    Some(GemmHarness {
        backend,
        a,
        b,
        c,
        n,
    })
}

/// Measures one square NoTrans/NoTrans GEMM at `n`×`n`×`n`.
fn bench_gemm_n(criterion: &mut Criterion, n: usize) {
    let harness = match try_setup(n) {
        Some(h) => h,
        None => {
            eprintln!("skip: no Metal device (gemm_{n})");
            return;
        }
    };

    // 2*n^3 fused-multiply-add flops per call.
    let flops_per_call = 2u64 * (n as u64).pow(3);

    let mut group = criterion.benchmark_group(format!("metal_gemm_{n}"));
    group.throughput(Throughput::Elements(flops_per_call));
    group.bench_function("gemm", |bencher| {
        bencher.iter(|| {
            let r = harness.backend.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                harness.n,
                harness.n,
                harness.n,
                1.0,
                black_box(harness.a),
                harness.n,
                black_box(harness.b),
                harness.n,
                0.0,
                harness.c,
                harness.n,
            );
            black_box(r.ok());
        });
    });
    group.finish();
}

fn bench_gemm_256(c: &mut Criterion) {
    bench_gemm_n(c, 256);
}

fn bench_gemm_1024(c: &mut Criterion) {
    bench_gemm_n(c, 1024);
}

criterion_group!(benches, bench_gemm_256, bench_gemm_1024);
criterion_main!(benches);
