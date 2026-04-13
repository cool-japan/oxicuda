use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_runtime::device::get_device_count;
use oxicuda_runtime::{CudaRtError, DevicePtr};

// ─── helpers ─────────────────────────────────────────────────────────────────

/// True if a real CUDA GPU is available.
fn gpu_available() -> bool {
    matches!(get_device_count(), Ok(n) if n > 0)
}

// ─── Error mapping ────────────────────────────────────────────────────────────

fn bench_from_code(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_rt_from_code");

    let codes: &[(&str, u32)] = &[
        ("success", 0),
        ("memory_alloc", 2),
        ("no_device", 100),
        ("launch_failure", 719),
        ("unknown", 99999),
    ];

    for &(name, code) in codes {
        group.bench_with_input(BenchmarkId::new("from_code", name), &code, |b, &code| {
            b.iter(|| black_box(CudaRtError::from_code(code)));
        });
    }
    group.finish();
}

// ─── DevicePtr arithmetic ─────────────────────────────────────────────────────

fn bench_device_ptr_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_ptr");

    group.bench_function("offset_forward", |b| {
        let p = DevicePtr(0x1000_0000);
        b.iter(|| black_box(p.offset(black_box(128))));
    });

    group.bench_function("offset_backward", |b| {
        let p = DevicePtr(0x1000_0000);
        b.iter(|| black_box(p.offset(black_box(-64))));
    });

    group.bench_function("is_null", |b| {
        b.iter(|| black_box(DevicePtr::NULL.is_null()));
    });

    group.finish();
}

// ─── GPU-gated: memory alloc / free ──────────────────────────────────────────

fn bench_malloc_free(c: &mut Criterion) {
    if !gpu_available() {
        return;
    }
    oxicuda_runtime::set_device(0).unwrap();

    let mut group = c.benchmark_group("cuda_malloc_free");

    let sizes: &[(&str, usize)] = &[("1kib", 1_024), ("1mib", 1 << 20), ("64mib", 64 << 20)];

    for &(name, sz) in sizes {
        group.bench_with_input(BenchmarkId::new("malloc_free", name), &sz, |b, &sz| {
            b.iter(|| {
                let ptr = oxicuda_runtime::cuda_malloc(sz).unwrap();
                black_box(ptr);
                oxicuda_runtime::cuda_free(ptr).unwrap();
            });
        });
    }
    group.finish();
}

// ─── GPU-gated: memset ────────────────────────────────────────────────────────

fn bench_memset(c: &mut Criterion) {
    if !gpu_available() {
        return;
    }
    oxicuda_runtime::set_device(0).unwrap();

    let mut group = c.benchmark_group("cuda_memset");

    let sizes: &[(&str, usize)] = &[
        ("1mib", 1 << 20),
        ("16mib", 16 << 20),
        ("128mib", 128 << 20),
    ];

    for &(name, sz) in sizes {
        let ptr = oxicuda_runtime::cuda_malloc(sz).unwrap();
        group.bench_with_input(BenchmarkId::new("memset_zero", name), &sz, |b, &sz| {
            b.iter(|| {
                oxicuda_runtime::cuda_memset(ptr, 0, sz).unwrap();
            });
        });
        oxicuda_runtime::cuda_free(ptr).unwrap();
    }
    group.finish();
}

// ─── GPU-gated: H2D / D2H transfer ───────────────────────────────────────────

fn bench_memcpy(c: &mut Criterion) {
    if !gpu_available() {
        return;
    }
    oxicuda_runtime::set_device(0).unwrap();

    let mut group = c.benchmark_group("cuda_memcpy");

    let sizes: &[(&str, usize)] = &[
        ("1mib_f32", (1 << 20) / 4), // 1 MiB as f32 elements
        ("64mib_f32", (64 << 20) / 4),
    ];

    for &(name, n) in sizes {
        let host_src: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut host_dst = vec![0.0f32; n];
        let dev_ptr = oxicuda_runtime::cuda_malloc(n * 4).unwrap();

        group.bench_with_input(BenchmarkId::new("h2d", name), &n, |b, _| {
            b.iter(|| {
                oxicuda_runtime::memcpy_h2d(dev_ptr, &host_src).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("d2h", name), &n, |b, _| {
            b.iter(|| {
                oxicuda_runtime::memcpy_d2h(&mut host_dst, dev_ptr).unwrap();
            });
        });

        oxicuda_runtime::cuda_free(dev_ptr).unwrap();
    }
    group.finish();
}

// ─── Stream create/destroy ────────────────────────────────────────────────────

fn bench_stream(c: &mut Criterion) {
    if !gpu_available() {
        return;
    }
    oxicuda_runtime::set_device(0).unwrap();

    let mut group = c.benchmark_group("cuda_stream");

    group.bench_function("create_destroy", |b| {
        b.iter(|| {
            let s = oxicuda_runtime::stream::stream_create().unwrap();
            black_box(s);
            oxicuda_runtime::stream::stream_destroy(s).unwrap();
        });
    });

    group.bench_function("synchronize_empty", |b| {
        let s = oxicuda_runtime::stream::stream_create().unwrap();
        b.iter(|| {
            oxicuda_runtime::stream::stream_synchronize(s).unwrap();
        });
        oxicuda_runtime::stream::stream_destroy(s).unwrap();
    });

    group.finish();
}

// ─── Event elapsed time ───────────────────────────────────────────────────────

fn bench_event(c: &mut Criterion) {
    if !gpu_available() {
        return;
    }
    oxicuda_runtime::set_device(0).unwrap();

    let mut group = c.benchmark_group("cuda_event");

    group.bench_function("create_record_sync_destroy", |b| {
        let s = oxicuda_runtime::stream::stream_create().unwrap();
        b.iter(|| {
            let e = oxicuda_runtime::event::event_create().unwrap();
            oxicuda_runtime::event::event_record(e, s).unwrap();
            oxicuda_runtime::event::event_synchronize(e).unwrap();
            oxicuda_runtime::event::event_destroy(e).unwrap();
        });
        oxicuda_runtime::stream::stream_destroy(s).unwrap();
    });

    group.bench_function("elapsed_time_empty_region", |b| {
        let s = oxicuda_runtime::stream::stream_create().unwrap();
        let start = oxicuda_runtime::event::event_create().unwrap();
        let end = oxicuda_runtime::event::event_create().unwrap();
        oxicuda_runtime::event::event_record(start, s).unwrap();
        oxicuda_runtime::event::event_record(end, s).unwrap();
        oxicuda_runtime::event::event_synchronize(end).unwrap();

        b.iter(|| black_box(oxicuda_runtime::event::event_elapsed_time(start, end).unwrap()));

        oxicuda_runtime::event::event_destroy(start).unwrap();
        oxicuda_runtime::event::event_destroy(end).unwrap();
        oxicuda_runtime::stream::stream_destroy(s).unwrap();
    });

    group.finish();
}

// ─── wiring ──────────────────────────────────────────────────────────────────

criterion_group!(
    runtime_benches,
    bench_from_code,
    bench_device_ptr_ops,
    bench_malloc_free,
    bench_memset,
    bench_memcpy,
    bench_stream,
    bench_event,
);
criterion_main!(runtime_benches);
