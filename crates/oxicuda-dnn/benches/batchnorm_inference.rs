//! BatchNorm inference — target: P7 (NCHW [64, 256, 28, 28]).
//!
//! Measures throughput of [`oxicuda_dnn::norm::batch_norm_forward`] in
//! **inference** mode (training=false) on the canonical ResNet
//! mid-network feature-map shape:
//!
//! - input  `[N=64, C=256, H=28, W=28]`
//! - gamma  `[C=256]`
//! - beta   `[C=256]`
//! - running_mean / running_var `[C=256]`
//! - output `[N=64, C=256, H=28, W=28]`
//!
//! Throughput is reported in **elements/sec**. The op is memory-bandwidth
//! bound; multiply by `2 * sizeof(elem)` for GB/s.
//!
//! Skips on any host without an NVIDIA driver / GPU.

use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_dnn::DnnHandle;
use oxicuda_dnn::norm::batch_norm_forward;
use oxicuda_dnn::types::{TensorDesc, TensorDescMut};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

const N: u32 = 64;
const C: u32 = 256;
const H: u32 = 28;
const W: u32 = 28;
const EPS: f32 = 1e-5;
const MOMENTUM: f32 = 0.1;

fn bench_batchnorm_inference(c: &mut Criterion) {
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

    let elems = (N * C * H * W) as usize;
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
    let gamma = match DeviceBuffer::<f32>::zeroed(C as usize) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (gamma)");
            return;
        }
    };
    let beta = match DeviceBuffer::<f32>::zeroed(C as usize) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (beta)");
            return;
        }
    };
    let mut running_mean = match DeviceBuffer::<f32>::zeroed(C as usize) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (running_mean)");
            return;
        }
    };
    let mut running_var = match DeviceBuffer::<f32>::zeroed(C as usize) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (running_var)");
            return;
        }
    };

    let input = match TensorDesc::<f32>::nchw(&in_buf, N, C, H, W) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: input desc failed");
            return;
        }
    };
    let mut output = match TensorDescMut::<f32>::nchw(&mut out_buf, N, C, H, W) {
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
    batch_norm_forward(
        &handle,
        &input,
        &gamma,
        &beta,
        &mut running_mean,
        &mut running_var,
        &mut output,
        EPS,
        MOMENTUM,
        false, // inference mode
        None,
        None,
    )
    .expect("batch_norm_forward warm-up call must succeed");
    handle
        .stream()
        .synchronize()
        .expect("synchronize after warm-up");

    let mut group = c.benchmark_group("dnn_p7_batchnorm_inference");
    group.throughput(Throughput::Elements(work));
    group.bench_function("oxicuda_f32_n64_c256_28x28", |b| {
        b.iter(|| {
            batch_norm_forward(
                &handle,
                &input,
                &gamma,
                &beta,
                &mut running_mean,
                &mut running_var,
                &mut output,
                EPS,
                MOMENTUM,
                false, // inference mode
                None,
                None,
            )
            .expect("batch_norm_forward");
        });
    });
    group.finish();
}

criterion_group!(benches, bench_batchnorm_inference);
criterion_main!(benches);
