//! WebGPU reduction benchmark — measures `ComputeBackend::reduce` (`Sum`)
//! over 1,000,000 `f32` elements via the `reduce_nd` dispatch path.
//!
//! ## Platform behaviour
//!
//! * **A machine with a usable wgpu adapter** — the benchmark runs.
//! * **No adapter (headless CI)** — prints
//!   `skip: no WebGPU adapter (reduce_1m)` to stderr and returns before
//!   registering any benchmark closure with criterion.
//!
//! Run with:
//! ```bash
//! cargo bench -p oxicuda-webgpu --bench reduce_bench
//! ```

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_backend::{ComputeBackend, ReduceOp};
use oxicuda_webgpu::WebGpuBackend;

/// Number of `f32` elements reduced per call.
const N: usize = 1_000_000;

/// GPU resources shared across iterations of the criterion loop.
struct ReduceHarness {
    backend: WebGpuBackend,
    input: u64,
    output: u64,
}

/// Initialise the backend and allocate/fill an `N`-element input plus a
/// single-element output. Returns `None` without a usable wgpu adapter.
fn try_setup() -> Option<ReduceHarness> {
    let mut backend = WebGpuBackend::new();
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
            eprintln!("skip: no WebGPU adapter (reduce_1m)");
            return;
        }
    };

    let mut group = criterion.benchmark_group("webgpu_reduce_1m");
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
            // Synchronize every iteration: `reduce` only submits (wgpu
            // executes in FIFO order with no per-op poll), so without an
            // explicit wait the measured time is submission latency, not
            // GPU execution time, and unsynchronized submissions pile up
            // across criterion's iterations — observed to overflow
            // wgpu-core's own drop-time submission wait and panic.
            black_box(r.and_then(|()| harness.backend.synchronize()).ok());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_reduce_1m);
criterion_main!(benches);
