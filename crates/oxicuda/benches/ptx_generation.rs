use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;
use oxicuda_ptx::templates::{
    elementwise::{ElementwiseOp, ElementwiseTemplate},
    gemm::{EpilogueKind, GemmTemplate},
    reduction::{ReductionOp, ReductionTemplate},
};

fn bench_ptx_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("ptx_elementwise");

    let ops: &[(&str, ElementwiseOp, PtxType)] = &[
        ("add_f32", ElementwiseOp::Add, PtxType::F32),
        ("relu_f32", ElementwiseOp::Relu, PtxType::F32),
        ("gelu_f32", ElementwiseOp::Gelu, PtxType::F32),
        ("sigmoid_f64", ElementwiseOp::Sigmoid, PtxType::F64),
    ];

    for (name, op, ty) in ops {
        group.bench_with_input(BenchmarkId::new("generate", name), name, |b, _| {
            b.iter(|| {
                let t = ElementwiseTemplate::new(*op, *ty, SmVersion::Sm80);
                black_box(t.generate())
            });
        });
    }
    group.finish();
}

fn bench_ptx_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ptx_reduction");
    for op in [ReductionOp::Sum, ReductionOp::Max, ReductionOp::Min] {
        group.bench_with_input(
            BenchmarkId::new("generate_f32", format!("{op:?}")),
            &op,
            |b, &op| {
                b.iter(|| {
                    let t = ReductionTemplate {
                        op,
                        precision: PtxType::F32,
                        target: SmVersion::Sm80,
                        block_size: 256,
                    };
                    black_box(t.generate())
                });
            },
        );
    }
    group.finish();
}

fn bench_ptx_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("ptx_gemm");
    let configs: &[(u32, u32, u32, bool)] = &[
        (64, 64, 32, false),
        (128, 128, 32, false),
        (128, 128, 32, true),
        (128, 256, 64, true),
    ];
    for &(tm, tn, tk, tc) in configs {
        let name = format!("{tm}x{tn}x{tk}_tc{tc}");
        group.bench_with_input(
            BenchmarkId::new("generate", &name),
            &(tm, tn, tk, tc),
            |b, &(tm, tn, tk, tc)| {
                b.iter(|| {
                    let t = GemmTemplate {
                        tile_m: tm,
                        tile_n: tn,
                        tile_k: tk,
                        warp_m: 16,
                        warp_n: 16,
                        precision: PtxType::F32,
                        accumulator: PtxType::F32,
                        use_tensor_core: tc,
                        stages: 2,
                        target: SmVersion::Sm80,
                        epilogue: EpilogueKind::LinearCombination,
                    };
                    black_box(t.generate())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    ptx_benches,
    bench_ptx_elementwise,
    bench_ptx_reduction,
    bench_ptx_gemm
);
criterion_main!(ptx_benches);
