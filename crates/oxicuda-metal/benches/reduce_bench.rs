//! Metal reduction benchmark — measures `ComputeBackend::reduce` (`Sum`)
//! over 1,000,000 `f32` elements (the two-pass `chunked_reduce_msl` path,
//! since 1M elements is well past the flat-kernel/two-pass threshold).
//!
//! ## Platform behaviour
//!
//! * **macOS with a Metal-capable GPU** — the benchmark runs.
//! * **No Metal device (headless CI, non-macOS)** — prints
//!   `skip: no Metal device (reduce_1m)` to stderr and returns before
//!   registering any benchmark closure with criterion.
//!
//! Run with:
//! ```bash
//! cargo bench -p oxicuda-metal --bench reduce_bench
//! ```

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_backend::{ComputeBackend, ReduceOp};
use oxicuda_metal::MetalBackend;

/// Number of `f32` elements reduced per call.
const N: usize = 1_000_000;

/// GPU resources shared across iterations of the criterion loop.
struct ReduceHarness {
    backend: MetalBackend,
    input: u64,
    output: u64,
}

/// Initialise the backend and allocate/fill an `N`-element input plus a
/// single-element output. Returns `None` without a usable Metal device.
fn try_setup() -> Option<ReduceHarness> {
    let mut backend = MetalBackend::new();
    backend.init().ok()?;

    let input = backend.alloc(N * 4).ok()?;
    let output = backend.alloc(4).ok()?;

    let fill: Vec<f32> = (0..N).map(|i| ((i % 1000) as f32) * 0.001).collect();
    let fill_bytes: Vec<u8> = fill.iter().flat_map(|v| v.to_le_bytes()).collect();
    backend.copy_htod(input, &fill_bytes).ok()?;

    Some(ReduceHarness {
        backend,
        input,
        output,
    })
}

fn bench_reduce_1m(criterion: &mut Criterion) {
    let harness = match try_setup() {
        Some(h) => h,
        None => {
            eprintln!("skip: no Metal device (reduce_1m)");
            return;
        }
    };

    let mut group = criterion.benchmark_group("metal_reduce_1m");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("reduce_sum", |bencher| {
        bencher.iter(|| {
            let r = harness.backend.reduce(
                ReduceOp::Sum,
                black_box(harness.input),
                harness.output,
                &[N],
                0,
            );
            black_box(r.ok());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_reduce_1m);
criterion_main!(benches);
