use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_autotune::search_space::SearchSpace;

fn bench_search_space(c: &mut Criterion) {
    let mut group = c.benchmark_group("autotune_search_space");

    group.bench_function("gemm_default_total_configs", |b| {
        b.iter(|| {
            let ss = SearchSpace::gemm_default();
            black_box(ss.total_configs())
        });
    });

    group.bench_function("gemm_default_enumerate", |b| {
        b.iter(|| {
            let ss = SearchSpace::gemm_default();
            black_box(ss.enumerate())
        });
    });

    group.bench_function("gemm_default_prune", |b| {
        b.iter(|| {
            let ss = SearchSpace::gemm_default();
            black_box(ss.prune(48 * 1024, 255, 4))
        });
    });

    group.bench_function("minimal_enumerate", |b| {
        b.iter(|| {
            let ss = SearchSpace::minimal();
            black_box(ss.enumerate())
        });
    });

    group.bench_function("minimal_prune", |b| {
        b.iter(|| {
            let ss = SearchSpace::minimal();
            black_box(ss.prune(48 * 1024, 255, 4))
        });
    });

    group.finish();
}

criterion_group!(autotune_benches, bench_search_space);
criterion_main!(autotune_benches);
