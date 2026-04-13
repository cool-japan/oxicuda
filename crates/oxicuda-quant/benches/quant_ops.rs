use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_quant::pruning::magnitude::{MagnitudeNorm, MagnitudePruner};
use oxicuda_quant::scheme::minmax::{MinMaxQuantizer, QuantGranularity, QuantScheme};
use oxicuda_quant::scheme::nf4::Nf4Quantizer;

// --- data generators ---------------------------------------------------------

fn make_weights(n: usize) -> Vec<f32> {
    // Pseudo-random normally-distributed-ish weights using simple LCG.
    let mut x: u64 = 0xdeadbeef_cafebabe;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // Box-Muller-ish: combine two uniform draws
            let u1 = (x >> 40) as f32 / (1u32 << 24) as f32;
            let u2 = ((x >> 16) & 0xFF_FFFF) as f32 / (1u32 << 24) as f32;
            (u1 - 0.5) * 0.2 + (u2 - 0.5) * 0.02
        })
        .collect()
}

// --- MinMax INT8 symmetric ---------------------------------------------------

fn bench_int8_symmetric(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_int8_symmetric");
    let q = MinMaxQuantizer::int8_symmetric();

    let sizes: &[(&str, usize)] = &[
        ("1k", 1_024),
        ("16k", 16_384),
        ("256k", 262_144),
        ("1m", 1_048_576),
        ("4m", 4_194_304),
    ];

    for &(name, n) in sizes {
        let data = make_weights(n);
        let params = q.calibrate(&data).unwrap();

        group.bench_with_input(BenchmarkId::new("calibrate", name), &n, |b, _| {
            b.iter(|| black_box(q.calibrate(&data).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("quantize", name), &n, |b, _| {
            b.iter(|| black_box(q.quantize(&data, &params).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("dequantize", name), &n, |b, _| {
            let codes = q.quantize(&data, &params).unwrap();
            b.iter(|| black_box(q.dequantize(&codes, &params)));
        });

        group.bench_with_input(BenchmarkId::new("round_trip", name), &n, |b, _| {
            b.iter(|| {
                let p = q.calibrate(&data).unwrap();
                let codes = q.quantize(&data, &p).unwrap();
                black_box(q.dequantize(&codes, &p))
            });
        });
    }
    group.finish();
}

// --- MinMax INT4 per-group ---------------------------------------------------

fn bench_int4_per_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_int4_per_group");

    let sizes: &[(&str, usize, usize)] = &[
        ("1k_g128", 1_024, 128),
        ("16k_g128", 16_384, 128),
        ("64k_g128", 65_536, 128),
        ("256k_g64", 262_144, 64),
    ];

    for &(name, n, gs) in sizes {
        let data = make_weights(n);
        let q = MinMaxQuantizer::int4_per_group(gs);

        group.bench_with_input(BenchmarkId::new("calibrate", name), &n, |b, _| {
            b.iter(|| black_box(q.calibrate(&data).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("quantize_grouped", name), &n, |b, _| {
            let params = q.calibrate(&data).unwrap();
            b.iter(|| black_box(q.quantize_grouped(&data, &params, gs).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("dequantize_grouped", name), &n, |b, _| {
            let params = q.calibrate(&data).unwrap();
            let codes = q.quantize_grouped(&data, &params, gs).unwrap();
            b.iter(|| black_box(q.dequantize_grouped(&codes, &params, gs)));
        });
    }
    group.finish();
}

// --- MinMax per-channel (asymmetric) ----------------------------------------

fn bench_per_channel_asymmetric(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_per_channel_asymmetric");

    // Simulate a weight matrix row x col, quantized per output channel (row).
    let configs: &[(&str, usize, usize, usize)] = &[
        ("768x768_ch0", 768, 768, 0),
        ("4096x1024_ch0", 4096, 1024, 0),
        ("2048x2048_ch1", 2048, 2048, 1),
    ];

    for &(name, rows, cols, axis) in configs {
        let data = make_weights(rows * cols);
        let q = MinMaxQuantizer::new(
            8,
            QuantScheme::Asymmetric,
            QuantGranularity::PerChannel { channel_axis: axis },
        );

        group.bench_with_input(
            BenchmarkId::new("calibrate_2d", name),
            &(rows, cols),
            |b, &(r, c)| {
                b.iter(|| black_box(q.calibrate_2d(&data, r, c).unwrap()));
            },
        );
    }
    group.finish();
}

// --- NF4 encode / decode (QLoRA) --------------------------------------------

fn bench_nf4(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_nf4");

    let sizes: &[(&str, usize)] = &[
        ("4k", 4_096),
        ("64k", 65_536),
        ("256k", 262_144),
        ("1m", 1_048_576),
    ];

    let qnf4 = Nf4Quantizer::new(64);

    for &(name, n) in sizes {
        let data = make_weights(n);

        group.bench_with_input(BenchmarkId::new("encode", name), &n, |b, _| {
            b.iter(|| black_box(qnf4.encode(&data).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("decode", name), &n, |b, _| {
            let (packed, absmaxs) = qnf4.encode(&data).unwrap();
            b.iter(|| black_box(qnf4.decode(&packed, &absmaxs).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("round_trip", name), &n, |b, _| {
            b.iter(|| {
                let (packed, absmaxs) = qnf4.encode(&data).unwrap();
                black_box(qnf4.decode(&packed, &absmaxs).unwrap())
            });
        });

        group.bench_with_input(BenchmarkId::new("mse", name), &n, |b, _| {
            b.iter(|| black_box(qnf4.quantization_mse(&data).unwrap()));
        });
    }
    group.finish();
}

// --- NF4 block sizes ---------------------------------------------------------

fn bench_nf4_block_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("nf4_block_size");

    const N: usize = 65_536;
    let data = make_weights(N);

    for bs in [32_usize, 64, 128, 256] {
        let q = Nf4Quantizer::new(bs);
        group.bench_with_input(BenchmarkId::new("encode_64k", bs), &bs, |b, _| {
            b.iter(|| black_box(q.encode(&data).unwrap()));
        });
    }
    group.finish();
}

// --- Magnitude pruning -------------------------------------------------------

fn bench_magnitude_pruning(c: &mut Criterion) {
    let mut group = c.benchmark_group("pruning_magnitude");

    let sizes: &[(&str, usize)] = &[("16k", 16_384), ("256k", 262_144), ("1m", 1_048_576)];

    let sparsities = [0.5_f32, 0.7, 0.9];

    for &(name, n) in sizes {
        let data = make_weights(n);

        for sp in sparsities {
            let label = format!("{name}_sp{}", (sp * 100.0) as u32);

            group.bench_with_input(
                BenchmarkId::new("compute_mask_l1", &label),
                &sp,
                |b, &sp| {
                    let pruner = MagnitudePruner::new(sp, MagnitudeNorm::L1);
                    b.iter(|| black_box(pruner.compute_mask(&data).unwrap()));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("compute_mask_l2", &label),
                &sp,
                |b, &sp| {
                    let pruner = MagnitudePruner::new(sp, MagnitudeNorm::L2);
                    b.iter(|| black_box(pruner.compute_mask(&data).unwrap()));
                },
            );
        }
    }

    // prune in-place
    let sizes_prune: &[(&str, usize)] = &[("256k", 262_144), ("1m", 1_048_576)];
    for &(name, n) in sizes_prune {
        group.bench_with_input(
            BenchmarkId::new("prune_in_place_sp50", name),
            &n,
            |b, &n| {
                b.iter_batched(
                    || make_weights(n),
                    |mut weights| {
                        let pruner = MagnitudePruner::new(0.5, MagnitudeNorm::L1);
                        black_box(pruner.prune(&mut weights).unwrap())
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

// --- criterion wiring --------------------------------------------------------

criterion_group!(
    quant_benches,
    bench_int8_symmetric,
    bench_int4_per_group,
    bench_per_channel_asymmetric,
    bench_nf4,
    bench_nf4_block_sizes,
    bench_magnitude_pruning,
);
criterion_main!(quant_benches);
