//! FlashAttention-2 forward pass — target: P2 (seq=2048, head_dim=128).
//!
//! Measures throughput of [`oxicuda_dnn::attn::flash_attention_forward`] for
//! the FlashAttention-2 paper's headline shape:
//!
//! - Q  `[B=1, H=16, N_q=2048, D=128]`
//! - K  `[B=1, H=16, N_kv=2048, D=128]`
//! - V  `[B=1, H=16, N_kv=2048, D=128]`
//! - O  same shape as Q
//!
//! Throughput is reported as **tokens/sec** (=`B * H * N_q` elements per call).
//! The downstream FLOP cost is `4 * B * H * N_q * N_kv * D` for non-causal
//! attention, recoverable offline.
//!
//! Skips on any host without an NVIDIA driver / GPU (macOS hits
//! `UnsupportedPlatform` from `oxicuda_driver::init`).

use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_dnn::DnnHandle;
use oxicuda_dnn::attn::{FlashAttentionConfig, flash_attention_forward};
use oxicuda_dnn::types::{TensorDesc, TensorDescMut, TensorLayout};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;

const BATCH: u32 = 1;
const NUM_HEADS: u32 = 16;
const SEQ_LEN: u32 = 2048;
const HEAD_DIM: u32 = 128;

fn bench_flash_attention_2k_d128(c: &mut Criterion) {
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

    let qkv_elems = (BATCH * NUM_HEADS * SEQ_LEN * HEAD_DIM) as usize;
    let lse_elems = (BATCH * NUM_HEADS * SEQ_LEN) as usize;

    let q_buf = match DeviceBuffer::<f32>::zeroed(qkv_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (Q)");
            return;
        }
    };
    let k_buf = match DeviceBuffer::<f32>::zeroed(qkv_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (K)");
            return;
        }
    };
    let v_buf = match DeviceBuffer::<f32>::zeroed(qkv_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (V)");
            return;
        }
    };
    let o_buf = match DeviceBuffer::<f32>::zeroed(qkv_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (O)");
            return;
        }
    };
    let mut lse_buf = match DeviceBuffer::<f32>::zeroed(lse_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (LSE)");
            return;
        }
    };

    let q_strides = vec![
        NUM_HEADS * SEQ_LEN * HEAD_DIM,
        SEQ_LEN * HEAD_DIM,
        HEAD_DIM,
        1,
    ];
    let q = match TensorDesc::<f32>::from_raw(
        q_buf.as_device_ptr(),
        vec![BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM],
        q_strides.clone(),
        TensorLayout::Nchw,
    ) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: Q desc failed");
            return;
        }
    };
    let k = match TensorDesc::<f32>::from_raw(
        k_buf.as_device_ptr(),
        vec![BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM],
        q_strides.clone(),
        TensorLayout::Nchw,
    ) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: K desc failed");
            return;
        }
    };
    let v = match TensorDesc::<f32>::from_raw(
        v_buf.as_device_ptr(),
        vec![BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM],
        q_strides.clone(),
        TensorLayout::Nchw,
    ) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: V desc failed");
            return;
        }
    };
    let mut o = match TensorDescMut::<f32>::from_raw(
        o_buf.as_device_ptr(),
        vec![BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM],
        q_strides,
        TensorLayout::Nchw,
    ) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: O desc failed");
            return;
        }
    };

    let mut config =
        FlashAttentionConfig::auto(HEAD_DIM, SEQ_LEN, SEQ_LEN, false, handle.sm_version());
    config.num_heads = NUM_HEADS;

    let tokens_per_call = u64::from(BATCH * NUM_HEADS * SEQ_LEN);

    // Warm-up call outside the timed loop, plus `.expect()` inside it below:
    // a silently-discarded `Result` here would let a future regression (or a
    // future workspace requirement, mirroring the `conv_forward` bug this
    // repository has already hit) time an early-return-on-error instead of
    // the kernel this benchmark claims to measure.
    flash_attention_forward(&handle, &q, &k, &v, &mut o, &mut lse_buf, &config)
        .expect("flash_attention_forward warm-up call must succeed");
    handle
        .stream()
        .synchronize()
        .expect("synchronize after warm-up");

    let mut group = c.benchmark_group("dnn_p2_flash_attention_2k_d128");
    group.throughput(Throughput::Elements(tokens_per_call));
    group.bench_function("oxicuda_f32_seq2048_d128_h16", |b| {
        b.iter(|| {
            flash_attention_forward(&handle, &q, &k, &v, &mut o, &mut lse_buf, &config)
                .expect("flash_attention_forward");
        });
    });
    group.finish();
}

criterion_group!(benches, bench_flash_attention_2k_d128);
criterion_main!(benches);
