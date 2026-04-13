use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_blas::level3::gemm::dispatch::{GemmDispatcher, GemmProblem};
use oxicuda_blas::{AlgorithmSelector, MathMode, Transpose};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

fn make_problem(m: u32, n: u32, k: u32) -> GemmProblem {
    GemmProblem {
        m,
        n,
        k,
        trans_a: Transpose::NoTrans,
        trans_b: Transpose::NoTrans,
        input_type: PtxType::F32,
        output_type: PtxType::F32,
        math_mode: MathMode::Default,
    }
}

fn bench_gemm_classify(c: &mut Criterion) {
    let mut group = c.benchmark_group("blas_gemm_classify");

    let dispatcher = GemmDispatcher::new(SmVersion::Sm80);

    let sizes: &[(&str, u32, u32, u32)] = &[
        ("standard_512", 512, 512, 512),
        ("skinny_8x512", 8, 512, 256),
        ("split_k_64x64x8192", 64, 64, 8192),
        ("large_4096", 4096, 4096, 4096),
    ];

    for &(name, m, n, k) in sizes {
        let problem = make_problem(m, n, k);
        group.bench_with_input(
            BenchmarkId::new("classify", name),
            &problem,
            |b, problem| {
                b.iter(|| black_box(dispatcher.classify(problem)));
            },
        );
    }
    group.finish();
}

fn bench_gemm_heuristic_tile(c: &mut Criterion) {
    let mut group = c.benchmark_group("blas_gemm_heuristic_tile");

    let dispatcher = GemmDispatcher::new(SmVersion::Sm80);

    let sizes: &[(&str, u32, u32, u32)] = &[
        ("standard_512", 512, 512, 512),
        ("skinny_8x512", 8, 512, 256),
        ("split_k_64x64x8192", 64, 64, 8192),
        ("large_4096", 4096, 4096, 4096),
    ];

    for &(name, m, n, k) in sizes {
        let problem = make_problem(m, n, k);
        let category = dispatcher.classify(&problem);
        group.bench_with_input(
            BenchmarkId::new("heuristic_tile", name),
            &(problem, category),
            |b, (problem, category)| {
                b.iter(|| black_box(dispatcher.heuristic_tile_config(problem, category)));
            },
        );
    }
    group.finish();
}

fn bench_algorithm_selector(c: &mut Criterion) {
    let mut group = c.benchmark_group("blas_algorithm_selector");

    let selector = AlgorithmSelector::new(SmVersion::Sm80);

    let sizes: &[(&str, u32, u32, u32)] = &[
        ("small_128", 128, 128, 128),
        ("medium_512", 512, 512, 512),
        ("large_1024", 1024, 1024, 1024),
    ];

    for &(name, m, n, k) in sizes {
        let problem = make_problem(m, n, k);
        group.bench_with_input(
            BenchmarkId::new("enumerate_algorithms", name),
            &problem,
            |b, problem| {
                b.iter(|| black_box(selector.enumerate_algorithms(problem)));
            },
        );
    }

    group.bench_function("algorithm_count", |b| {
        b.iter(|| black_box(selector.algorithm_count()));
    });

    group.bench_function("sm_version", |b| {
        b.iter(|| black_box(selector.sm_version()));
    });

    group.finish();
}

criterion_group!(
    blas_benches,
    bench_gemm_classify,
    bench_gemm_heuristic_tile,
    bench_algorithm_selector
);
criterion_main!(blas_benches);
