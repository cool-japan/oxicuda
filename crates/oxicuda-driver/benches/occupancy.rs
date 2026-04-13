use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_driver::occupancy_ext::{
    ClusterConfig, ClusterOccupancy, DeviceOccupancyInfo, DynamicSmemOccupancy,
    OccupancyCalculator, OccupancyGrid,
};

// --- Helpers -----------------------------------------------------------------

fn a100_info() -> DeviceOccupancyInfo {
    DeviceOccupancyInfo::for_compute_capability(8, 0)
}

fn h100_info() -> DeviceOccupancyInfo {
    DeviceOccupancyInfo::for_compute_capability(9, 0)
}

fn ada_info() -> DeviceOccupancyInfo {
    DeviceOccupancyInfo::for_compute_capability(8, 9)
}

// --- estimate_occupancy ------------------------------------------------------

fn bench_estimate_occupancy(c: &mut Criterion) {
    let mut group = c.benchmark_group("occupancy_estimate");

    // (block_size, regs_per_thread, shared_mem_bytes)
    let configs: &[(&str, u32, u32, u32)] = &[
        ("128t_32r_0s", 128, 32, 0),
        ("256t_32r_0s", 256, 32, 0),
        ("256t_64r_0s", 256, 64, 0),
        ("512t_32r_0s", 512, 32, 0),
        ("256t_32r_16384s", 256, 32, 16_384),
        ("256t_32r_49152s", 256, 32, 49_152),
        ("1024t_16r_0s", 1024, 16, 0),
    ];

    let calc_a100 = OccupancyCalculator::new(a100_info());
    let calc_h100 = OccupancyCalculator::new(h100_info());

    for &(name, bs, regs, smem) in configs {
        group.bench_with_input(
            BenchmarkId::new("a100", name),
            &(bs, regs, smem),
            |b, &(bs, regs, smem)| {
                b.iter(|| black_box(calc_a100.estimate_occupancy(bs, regs, smem)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("h100", name),
            &(bs, regs, smem),
            |b, &(bs, regs, smem)| {
                b.iter(|| black_box(calc_h100.estimate_occupancy(bs, regs, smem)));
            },
        );
    }
    group.finish();
}

// --- OccupancyGrid sweep -----------------------------------------------------

fn bench_occupancy_grid_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("occupancy_grid_sweep");

    // (label, device_info, regs_per_thread, shared_mem)
    let configs: &[(&str, DeviceOccupancyInfo, u32, u32)] = &[
        ("a100_32r_0s", a100_info(), 32, 0),
        ("a100_64r_16k_s", a100_info(), 64, 16_384),
        ("h100_32r_0s", h100_info(), 32, 0),
        ("h100_64r_49k_s", h100_info(), 64, 49_152),
        ("ada_48r_8k_s", ada_info(), 48, 8_192),
    ];

    for (name, info, regs, smem) in configs {
        let calc = OccupancyCalculator::new(*info);
        group.bench_with_input(
            BenchmarkId::new("sweep", name),
            &(*regs, *smem),
            |b, &(regs, smem)| {
                b.iter(|| {
                    // sweep returns all block sizes up to max_threads_per_sm
                    let pts = OccupancyGrid::sweep(&calc, regs, smem);
                    black_box(OccupancyGrid::best_block_size(&pts))
                });
            },
        );
    }
    group.finish();
}

// --- for_compute_capability --------------------------------------------------

fn bench_device_info_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_occupancy_info");

    let cc_pairs: &[(&str, u32, u32)] = &[
        ("sm_75_turing", 7, 5),
        ("sm_80_a100", 8, 0),
        ("sm_86_ga10x", 8, 6),
        ("sm_89_ada", 8, 9),
        ("sm_90_h100", 9, 0),
        ("sm_100_b100", 10, 0),
        ("sm_120_b200", 12, 0),
    ];

    for &(name, maj, min) in cc_pairs {
        group.bench_with_input(
            BenchmarkId::new("for_cc", name),
            &(maj, min),
            |b, &(maj, min)| {
                b.iter(|| black_box(DeviceOccupancyInfo::for_compute_capability(maj, min)));
            },
        );
    }
    group.finish();
}

// --- DynamicSmemOccupancy ----------------------------------------------------

fn bench_dynamic_smem(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_smem_occupancy");
    let calc = OccupancyCalculator::new(a100_info());

    // with_smem_function(calculator, smem_fn, registers_per_thread)
    group.bench_function("linear_smem_4bytes_32regs", |b| {
        let smem_fn = DynamicSmemOccupancy::linear_smem(4);
        b.iter(|| {
            black_box(DynamicSmemOccupancy::with_smem_function(
                &calc, &smem_fn, 32,
            ))
        });
    });

    group.bench_function("tile_smem_32tile_4elem_32regs", |b| {
        let smem_fn = DynamicSmemOccupancy::tile_smem(32, 4);
        b.iter(|| {
            black_box(DynamicSmemOccupancy::with_smem_function(
                &calc, &smem_fn, 32,
            ))
        });
    });

    // Benchmark with varying register counts.
    for &regs in &[16u32, 32, 48, 64] {
        let smem_fn = DynamicSmemOccupancy::linear_smem(16);
        group.bench_with_input(
            BenchmarkId::new("linear16_sweep", regs),
            &regs,
            |b, &regs| {
                b.iter(|| {
                    black_box(DynamicSmemOccupancy::with_smem_function(
                        &calc, &smem_fn, regs,
                    ))
                });
            },
        );
    }

    group.finish();
}

// --- ClusterOccupancy (Hopper) -----------------------------------------------

fn bench_cluster_occupancy(c: &mut Criterion) {
    let mut group = c.benchmark_group("cluster_occupancy");
    let calc_h100 = OccupancyCalculator::new(h100_info());

    // (block_size, cluster_size, regs_per_thread, smem_per_block)
    let configs: &[(&str, u32, u32, u32, u32)] = &[
        ("1blk_128t_32r_0s", 128, 1, 32, 0),
        ("2blk_128t_32r_0s", 128, 2, 32, 0),
        ("4blk_128t_32r_0s", 128, 4, 32, 0),
        ("8blk_128t_32r_0s", 128, 8, 32, 0),
        ("4blk_256t_48r_8k", 256, 4, 48, 8_192),
        ("8blk_256t_32r_16k", 256, 8, 32, 16_384),
    ];

    for &(name, bs, cs, regs, smem) in configs {
        group.bench_with_input(
            BenchmarkId::new("h100", name),
            &(bs, cs, regs, smem),
            |b, &(bs, cs, regs, smem)| {
                b.iter(|| {
                    black_box(ClusterOccupancy::estimate_cluster_occupancy(
                        &calc_h100, bs, cs, regs, smem,
                    ))
                });
            },
        );
    }

    // ClusterConfig total_blocks helper
    group.bench_function("cluster_config_total_blocks", |b| {
        b.iter(|| {
            let totals: Vec<u32> = (1..=4u32)
                .flat_map(|x| {
                    (1..=4u32).map(move |y| {
                        ClusterConfig {
                            cluster_x: x,
                            cluster_y: y,
                            cluster_z: 1,
                        }
                        .total_blocks()
                    })
                })
                .collect();
            black_box(totals)
        });
    });

    group.finish();
}

// --- Full occupancy analysis pipeline ----------------------------------------

fn bench_full_occupancy_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("occupancy_full_analysis");

    // Simulates what an autotuner would do: for each target GPU, find
    // the best block size for a kernel with given resource requirements.
    let kernels: &[(&str, u32, u32)] = &[
        ("elementwise_relu", 32, 0),
        ("gemm_32x32", 64, 32_768),
        ("reduction_sum", 48, 16_384),
        ("softmax_fused", 32, 8_192),
        ("flash_attn_v2", 64, 49_152),
    ];

    let devices = [
        ("a100", a100_info()),
        ("h100", h100_info()),
        ("ada", ada_info()),
    ];

    for &(kname, regs, smem) in kernels {
        for (dname, info) in &devices {
            let label = format!("{kname}_on_{dname}");
            let calc = OccupancyCalculator::new(*info);
            group.bench_with_input(
                BenchmarkId::new("find_best_block_size", &label),
                &(regs, smem),
                |b, &(regs, smem)| {
                    b.iter(|| {
                        let pts = OccupancyGrid::sweep(&calc, regs, smem);
                        black_box(OccupancyGrid::best_block_size(&pts))
                    });
                },
            );
        }
    }
    group.finish();
}

// --- criterion wiring --------------------------------------------------------

criterion_group!(
    occupancy_benches,
    bench_estimate_occupancy,
    bench_occupancy_grid_sweep,
    bench_device_info_construction,
    bench_dynamic_smem,
    bench_cluster_occupancy,
    bench_full_occupancy_analysis,
);
criterion_main!(occupancy_benches);
