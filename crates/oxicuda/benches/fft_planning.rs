use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_fft::{plan::FftPlan, types::FftType};

fn bench_fft_planning_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_plan_1d");
    for &n in &[256usize, 512, 1024, 4096, 8192, 65536] {
        group.bench_with_input(BenchmarkId::new("c2c", n), &n, |b, &n| {
            b.iter(|| black_box(FftPlan::new_1d(n, FftType::C2C, 1)));
        });
        group.bench_with_input(BenchmarkId::new("r2c", n), &n, |b, &n| {
            b.iter(|| black_box(FftPlan::new_1d(n, FftType::R2C, 1)));
        });
    }
    group.finish();
}

fn bench_fft_planning_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_plan_batched");
    for &batch in &[1usize, 8, 32, 64, 256] {
        group.bench_with_input(
            BenchmarkId::new("1024_c2c_batch", batch),
            &batch,
            |b, &batch| {
                b.iter(|| black_box(FftPlan::new_1d(1024, FftType::C2C, batch)));
            },
        );
    }
    group.finish();
}

fn bench_fft_planning_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_plan_2d");
    for &(nx, ny) in &[(64usize, 64usize), (128, 128), (256, 256), (512, 512)] {
        group.bench_with_input(
            BenchmarkId::new("c2c", format!("{nx}x{ny}")),
            &(nx, ny),
            |b, &(nx, ny)| {
                b.iter(|| black_box(FftPlan::new_2d(nx, ny, FftType::C2C, 1)));
            },
        );
    }
    group.finish();
}

criterion_group!(
    fft_benches,
    bench_fft_planning_1d,
    bench_fft_planning_batched,
    bench_fft_planning_2d
);
criterion_main!(fft_benches);
