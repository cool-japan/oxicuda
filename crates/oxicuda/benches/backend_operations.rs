use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda::backend::{BackendError, BinaryOp, ComputeBackend, CudaBackend, ReduceOp, UnaryOp};

fn bench_backend_new_and_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("backend_operations");

    group.bench_function("cuda_backend_new_and_init", |b| {
        b.iter(|| {
            let mut backend = CudaBackend::new();
            black_box(backend.init())
        });
    });

    group.bench_function("cuda_backend_init_reduce", |b| {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        b.iter(|| {
            // Exercises validation path — returns Unsupported but still
            // runs the axis/shape guard logic.
            black_box(backend.reduce(ReduceOp::Sum, 1, 2, &[64, 64], 0))
        });
    });

    group.bench_function("cuda_backend_conv2d_validation", |b| {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        b.iter(|| {
            black_box(backend.conv2d_forward(
                1,
                &[1usize, 3, 224, 224],
                2,
                &[64usize, 3, 3, 3],
                3,
                &[1usize, 64, 222, 222],
                &[1usize, 1],
                &[0usize, 0],
            ))
        });
    });

    group.bench_function("cuda_backend_unary_empty", |b| {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        b.iter(|| {
            // n == 0 is a fast no-op path
            black_box(backend.unary(UnaryOp::Relu, 0, 0, 0))
        });
    });

    group.bench_function("cuda_backend_binary_empty", |b| {
        let mut backend = CudaBackend::new();
        backend.init().ok();
        b.iter(|| black_box(backend.binary(BinaryOp::Add, 0, 0, 0, 0)));
    });

    group.bench_function("backend_error_display", |b| {
        b.iter(|| black_box(BackendError::Unsupported("test operation".into()).to_string()));
    });

    group.bench_function("backend_error_not_initialized", |b| {
        b.iter(|| black_box(BackendError::NotInitialized.to_string()));
    });

    group.finish();
}

criterion_group!(backend_benches, bench_backend_new_and_init);
criterion_main!(backend_benches);
