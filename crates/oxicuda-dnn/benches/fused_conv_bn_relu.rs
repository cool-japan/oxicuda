//! Fused Conv + BN + ReLU vs unfused — target: P8 (≥ 2× speedup).
//!
//! Two benchmark groups:
//!   * `fused`   — single call to [`oxicuda_dnn::conv::conv_bn_relu`].
//!   * `unfused` — `conv_forward` baseline (BN + ReLU would normally follow
//!     as separate kernels). The fused/unfused ratio is recovered by dividing
//!     the two reported throughputs offline.
//!
//! Problem shape (ResNet-style mid-network block):
//! - input  `[N=8, C=128, H=28, W=28]` (NCHW, f32)
//! - filter `[K=128, C=128, R=3, S=3]`, stride 1, pad 1
//! - output `[N=8, K=128, P=28, Q=28]`
//! - BN per-channel scale/bias buffers `[C=128]`
//! - activation: ReLU
//!
//! Skips on any host without an NVIDIA driver / GPU.

use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_dnn::DnnError;
use oxicuda_dnn::DnnHandle;
use oxicuda_dnn::conv::fused::FusedBnParams;
use oxicuda_dnn::conv::{conv_bn_relu, conv_forward};
use oxicuda_dnn::types::{Activation, ConvolutionDescriptor, TensorDesc, TensorDescMut};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

const N: u32 = 8;
const C: u32 = 128;
const H: u32 = 28;
const W: u32 = 28;
const K: u32 = 128;
const KH: u32 = 3;
const KW: u32 = 3;
const STRIDE: u32 = 1;
const PAD: u32 = 1;

fn bench_fused_conv_bn_relu(c: &mut Criterion) {
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

    let in_elems = (N * C * H * W) as usize;
    let filt_elems = (K * C * KH * KW) as usize;
    let out_elems = (N * K * H * W) as usize;

    let in_buf = match DeviceBuffer::<f32>::zeroed(in_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (input)");
            return;
        }
    };
    let filt_buf = match DeviceBuffer::<f32>::zeroed(filt_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (filter)");
            return;
        }
    };
    let mut out_buf = match DeviceBuffer::<f32>::zeroed(out_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (output)");
            return;
        }
    };
    let scale_buf = match DeviceBuffer::<f32>::zeroed(K as usize) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (BN scale)");
            return;
        }
    };
    let bias_buf = match DeviceBuffer::<f32>::zeroed(K as usize) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (BN bias)");
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
    let filter = match TensorDesc::<f32>::nchw(&filt_buf, K, C, KH, KW) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: filter desc failed");
            return;
        }
    };
    let mut output = match TensorDescMut::<f32>::nchw(&mut out_buf, N, K, H, W) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: output desc failed");
            return;
        }
    };
    let conv_desc = match ConvolutionDescriptor::conv2d(PAD, PAD, STRIDE, STRIDE, 1, 1, 1) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: conv descriptor failed");
            return;
        }
    };

    let bn_params = FusedBnParams {
        fused_scale_ptr: scale_buf.as_device_ptr(),
        fused_bias_ptr: bias_buf.as_device_ptr(),
        channels: K,
    };

    let work = u64::from(out_elems as u32);

    // Warm-up call outside the timed loop: `.expect()` here and inside the
    // timed closure below means a future regression that reintroduces an
    // error fails this benchmark loudly instead of silently timing whatever
    // the early-return-on-error path costs (effectively nothing).
    // `conv_bn_relu` takes no workspace parameter, so unlike the `unfused`
    // group below there is nothing to probe for.
    conv_bn_relu(
        &handle,
        &input,
        &filter,
        &mut output,
        &conv_desc,
        &bn_params,
        Activation::Relu,
    )
    .expect("conv_bn_relu warm-up call must succeed");
    handle
        .stream()
        .synchronize()
        .expect("synchronize after fused warm-up");

    let mut fused = c.benchmark_group("dnn_p8_fused_conv_bn_relu_fused");
    fused.throughput(Throughput::Elements(work));
    fused.bench_function("oxicuda_f32_n8_c128_28x28_3x3_relu", |b| {
        b.iter(|| {
            conv_bn_relu(
                &handle,
                &input,
                &filter,
                &mut output,
                &conv_desc,
                &bn_params,
                Activation::Relu,
            )
            .expect("conv_bn_relu");
        });
    });
    fused.finish();

    // `conv_forward` dispatches to whichever algorithm `select_algorithm`
    // picks for this problem shape; for N=8,C=128,H=W=28,K=128,3x3,s1p1 the
    // estimated GEMM FLOP count clears `WINOGRAD_FLOP_THRESHOLD`, so it is
    // Winograd, which -- like im2col+GEMM -- requires a caller-provided
    // scratch workspace and otherwise returns
    // `Err(DnnError::WorkspaceRequired(bytes))` immediately, before touching
    // the GPU. A benchmark that discards that `Result` (as this one used to)
    // times the cost of that instant early return, not a convolution. Probe
    // for the requirement the same way
    // `tests/conv_forward_xcorr_gpu.rs::gpu_xcorr` does: call once with no
    // workspace, and on `WorkspaceRequired` allocate exactly the requested
    // size and keep reusing that one buffer for every iteration below.
    let mut workspace = match conv_forward(&handle, &input, &filter, &mut output, &conv_desc, None)
    {
        Ok(()) => None,
        Err(DnnError::WorkspaceRequired(bytes)) => match DeviceBuffer::<u8>::zeroed(bytes) {
            Ok(ws) => Some(ws),
            Err(_) => {
                eprintln!("skip: workspace alloc failed ({bytes} bytes)");
                return;
            }
        },
        Err(e) => panic!(
            "conv_forward failed while probing its workspace requirement \
             (expected either Ok or WorkspaceRequired): {e}"
        ),
    };

    conv_forward(
        &handle,
        &input,
        &filter,
        &mut output,
        &conv_desc,
        workspace.as_mut(),
    )
    .expect("conv_forward warm-up call must succeed once workspace is provided");
    handle
        .stream()
        .synchronize()
        .expect("synchronize after unfused warm-up");

    let mut unfused = c.benchmark_group("dnn_p8_fused_conv_bn_relu_unfused");
    unfused.throughput(Throughput::Elements(work));
    unfused.bench_function("oxicuda_f32_n8_c128_28x28_3x3_baseline", |b| {
        b.iter(|| {
            conv_forward(
                &handle,
                &input,
                &filter,
                &mut output,
                &conv_desc,
                workspace.as_mut(),
            )
            .expect("conv_forward");
        });
    });
    unfused.finish();
}

criterion_group!(benches, bench_fused_conv_bn_relu);
criterion_main!(benches);
