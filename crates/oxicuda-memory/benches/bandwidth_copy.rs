use std::hint::black_box;
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_driver::{Context, Device, init};
use oxicuda_memory::copy::{copy_dtoh, copy_htod};
use oxicuda_memory::{
    BandwidthMeasurement, BandwidthProfiler, DeviceBuffer, PinnedBuffer, TransferDirection,
    bandwidth_utilization, describe_bandwidth, format_bytes, theoretical_peak_bandwidth,
};

const F32_BYTES: usize = std::mem::size_of::<f32>();

fn setup_context() -> Option<Context> {
    init().ok()?;
    if Device::count().ok()? <= 0 {
        return None;
    }
    let device = Device::get(0).ok()?;
    Context::new(&device).ok()
}

fn measure_h2d<T: Copy>(
    device: &mut DeviceBuffer<T>,
    host: &[T],
    warmup: u32,
    iters: u32,
) -> Option<BandwidthMeasurement> {
    for _ in 0..warmup {
        copy_htod(device, host).ok()?;
    }

    let start = Instant::now();
    for _ in 0..iters {
        copy_htod(device, host).ok()?;
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
    Some(BandwidthMeasurement::new(
        TransferDirection::HostToDevice,
        std::mem::size_of_val(host),
        elapsed_ms,
    ))
}

fn measure_d2h<T: Copy>(
    host: &mut [T],
    device: &DeviceBuffer<T>,
    warmup: u32,
    iters: u32,
) -> Option<BandwidthMeasurement> {
    for _ in 0..warmup {
        copy_dtoh(host, device).ok()?;
    }

    let start = Instant::now();
    for _ in 0..iters {
        copy_dtoh(host, device).ok()?;
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
    Some(BandwidthMeasurement::new(
        TransferDirection::DeviceToHost,
        std::mem::size_of_val(host),
        elapsed_ms,
    ))
}

fn report_nf2(size_elements: usize) {
    let _ctx = match setup_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("[bandwidth_copy] no CUDA device/context available, skipping NF2 report");
            return;
        }
    };

    let mut device = match DeviceBuffer::<f32>::alloc(size_elements) {
        Ok(buf) => buf,
        Err(err) => {
            eprintln!("[bandwidth_copy] device allocation failed: {err}");
            return;
        }
    };

    let host_src_pageable = vec![1.0f32; size_elements];
    let mut host_dst_pageable = vec![0.0f32; size_elements];

    let mut host_src_pinned = match PinnedBuffer::<f32>::alloc(size_elements) {
        Ok(buf) => buf,
        Err(err) => {
            eprintln!("[bandwidth_copy] pinned src allocation failed: {err}");
            return;
        }
    };
    host_src_pinned.as_mut_slice().fill(1.0);

    let mut host_dst_pinned = match PinnedBuffer::<f32>::alloc(size_elements) {
        Ok(buf) => buf,
        Err(err) => {
            eprintln!("[bandwidth_copy] pinned dst allocation failed: {err}");
            return;
        }
    };

    let mut profiler = BandwidthProfiler::with_iterations(3, 20);

    if let Some(m) = measure_h2d(
        &mut device,
        host_src_pageable.as_slice(),
        profiler.warmup_iterations,
        profiler.benchmark_iterations,
    ) {
        profiler.record(m);
    }
    if let Some(m) = measure_d2h(
        host_dst_pageable.as_mut_slice(),
        &device,
        profiler.warmup_iterations,
        profiler.benchmark_iterations,
    ) {
        profiler.record(m);
    }
    if let Some(m) = measure_h2d(
        &mut device,
        host_src_pinned.as_slice(),
        profiler.warmup_iterations,
        profiler.benchmark_iterations,
    ) {
        profiler.record(m);
    }
    if let Some(m) = measure_d2h(
        host_dst_pinned.as_mut_slice(),
        &device,
        profiler.warmup_iterations,
        profiler.benchmark_iterations,
    ) {
        profiler.record(m);
    }

    let pcie4_x16_peak = theoretical_peak_bandwidth(4, 16);
    let pcie3_x16_peak = theoretical_peak_bandwidth(3, 16);
    let summary = profiler.summary();

    eprintln!(
        "[bandwidth_copy] NF2 report for {} (peak PCIe4x16 {:.2} GB/s, PCIe3x16 {:.2} GB/s)",
        format_bytes(size_elements * F32_BYTES),
        pcie4_x16_peak,
        pcie3_x16_peak
    );

    for direction in [
        TransferDirection::HostToDevice,
        TransferDirection::DeviceToHost,
    ] {
        if let Some(dir) = summary
            .per_direction
            .iter()
            .filter(|d| d.direction == direction)
            .max_by(|a, b| a.max_bandwidth_gbps.total_cmp(&b.max_bandwidth_gbps))
        {
            let util_pcie4 = bandwidth_utilization(dir.max_bandwidth_gbps, pcie4_x16_peak) * 100.0;
            let util_pcie3 = bandwidth_utilization(dir.max_bandwidth_gbps, pcie3_x16_peak) * 100.0;
            eprintln!(
                "  {} best: {} (utilization: PCIe4x16 {:.1}%, PCIe3x16 {:.1}%)",
                direction,
                describe_bandwidth(dir.max_bandwidth_gbps),
                util_pcie4,
                util_pcie3
            );
        }
    }
}

fn bench_copy_bandwidth(c: &mut Criterion) {
    let _ctx = match setup_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("[bandwidth_copy] no CUDA device/context available, skipping benchmark");
            return;
        }
    };

    let sizes: &[(&str, usize)] = &[
        ("1mib_f32", (1 << 20) / F32_BYTES),
        ("16mib_f32", (16 << 20) / F32_BYTES),
        ("64mib_f32", (64 << 20) / F32_BYTES),
    ];

    report_nf2((64 << 20) / F32_BYTES);

    let mut group = c.benchmark_group("memory_copy_bandwidth");
    group.sample_size(20);

    for &(label, elements) in sizes {
        let mut device = match DeviceBuffer::<f32>::alloc(elements) {
            Ok(buf) => buf,
            Err(_) => continue,
        };

        let host_pageable_src = vec![1.0f32; elements];
        let mut host_pageable_dst = vec![0.0f32; elements];

        let mut host_pinned_src = match PinnedBuffer::<f32>::alloc(elements) {
            Ok(buf) => buf,
            Err(_) => continue,
        };
        host_pinned_src.as_mut_slice().fill(1.0);

        let mut host_pinned_dst = match PinnedBuffer::<f32>::alloc(elements) {
            Ok(buf) => buf,
            Err(_) => continue,
        };

        group.bench_with_input(
            BenchmarkId::new("h2d_pageable", label),
            &elements,
            |b, _| {
                b.iter(|| {
                    copy_htod(&mut device, black_box(host_pageable_src.as_slice())).ok();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("d2h_pageable", label),
            &elements,
            |b, _| {
                b.iter(|| {
                    copy_dtoh(black_box(host_pageable_dst.as_mut_slice()), &device).ok();
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("h2d_pinned", label), &elements, |b, _| {
            b.iter(|| {
                copy_htod(&mut device, black_box(host_pinned_src.as_slice())).ok();
            });
        });

        group.bench_with_input(BenchmarkId::new("d2h_pinned", label), &elements, |b, _| {
            b.iter(|| {
                copy_dtoh(black_box(host_pinned_dst.as_mut_slice()), &device).ok();
            });
        });
    }

    group.finish();
}

criterion_group!(memory_copy_benches, bench_copy_bandwidth);
criterion_main!(memory_copy_benches);
