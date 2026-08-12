//! LayerNorm — target: P5 (N=1024 rows × D=4096 hidden, memory-bandwidth bound).
//!
//! Measures throughput of [`oxicuda_dnn::norm::layer_norm`] for the canonical
//! Transformer hidden-state shape:
//!
//! - input  `[N=1024, D=4096]` (row-major)
//! - gamma  `[D=4096]`
//! - beta   `[D=4096]`
//! - output `[N=1024, D=4096]`
//!
//! Throughput is reported in **elements/sec**. Effective GB/s is recovered
//! offline by multiplying by `2 * sizeof(elem)` (one read + one write per
//! element); for f32 this is `8` bytes/element.
//!
//! Skips on any host without an NVIDIA driver / GPU.

use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_dnn::DnnHandle;
use oxicuda_dnn::norm::layer_norm;
use oxicuda_dnn::types::{TensorDesc, TensorDescMut};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

const N: u32 = 1024;
const D: u32 = 4096;
const EPS: f32 = 1e-5;

fn bench_layernorm_d4096(c: &mut Criterion) {
    if oxicuda_driver::init().is_err() {
        eprintln!("skip: no GPU (driver init failed)");
        return;
    }
    if !matches!(Device::count(), Ok(n) if n > 0) {
        eprintln!("skip: no GPU (Device::count <= 0)");
        return;
    }
    let device = match Device::get(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: no GPU (Device::get(0) failed)");
            return;
        }
    };
    let ctx = match Context::new(&device) {
        Ok(c) => Arc::new(c),
        Err(_) => {
            eprintln!("skip: no GPU (context creation failed)");
            return;
        }
    };
    let handle = match DnnHandle::new(&ctx) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("skip: no GPU (DnnHandle init failed)");
            return;
        }
    };

    let elems = (N * D) as usize;
    let in_buf = match DeviceBuffer::<f32>::zeroed(elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (input)");
            return;
        }
    };
    let mut out_buf = match DeviceBuffer::<f32>::zeroed(elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (output)");
            return;
        }
    };
    let gamma = match DeviceBuffer::<f32>::zeroed(D as usize) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (gamma)");
            return;
        }
    };
    let beta = match DeviceBuffer::<f32>::zeroed(D as usize) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (beta)");
            return;
        }
    };

    let input = match TensorDesc::<f32>::matrix(&in_buf, N, D) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: input desc failed");
            return;
        }
    };
    let mut output = match TensorDescMut::<f32>::matrix(&mut out_buf, N, D) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: output desc failed");
            return;
        }
    };

    let work = elems as u64;

    // Warm-up call outside the timed loop, plus `.expect()` inside it below:
    // a silently-discarded `Result` here would let a future regression (or a
    // future workspace requirement, mirroring the `conv_forward` bug this
    // repository has already hit) time an early-return-on-error instead of
    // the kernel this benchmark claims to measure.
    layer_norm(&handle, &input, &gamma, &beta, &mut output, EPS)
        .expect("layer_norm warm-up call must succeed");
    handle
        .stream()
        .synchronize()
        .expect("synchronize after warm-up");

    let mut group = c.benchmark_group("dnn_p5_layernorm_d4096");
    group.throughput(Throughput::Elements(work));
    group.bench_function("oxicuda_f32_n1024_d4096", |b| {
        b.iter(|| {
            layer_norm(&handle, &input, &gamma, &beta, &mut output, EPS).expect("layer_norm");
        });
    });
    group.finish();
}

criterion_group!(benches, bench_layernorm_d4096);
criterion_main!(benches);
