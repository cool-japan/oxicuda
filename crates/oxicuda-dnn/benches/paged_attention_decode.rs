//! PagedAttention decode — target: P3 (batch=32, seq=4096 incremental decode).
//!
//! Measures throughput of [`oxicuda_dnn::attn::paged_attention_decode`] for
//! a batched single-token decode step against a paged KV-cache. Parameters
//! mirror the vLLM Llama-2-7B GQA decode path:
//!
//! - Q  `[B=32, H=32, 1, D=128]`
//! - K-cache / V-cache pool (raw `CUdeviceptr`, one physical page kept alive)
//! - `page_table` `[B * max_pages_per_seq]` of `i32`, zero-filled (every
//!   logical page maps to physical page 0). Throughput numbers measure the
//!   launch and per-token attention compute path; absolute correctness of
//!   cache contents is irrelevant for harness measurement.
//! - `seq_lengths` length 32, all 4096
//!
//! Throughput is reported as **tokens/sec** (= batch tokens decoded per call).
//!
//! Skips on any host without an NVIDIA driver / GPU.

use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_dnn::DnnHandle;
use oxicuda_dnn::attn::{PagedAttentionConfig, paged_attention_decode};
use oxicuda_dnn::types::{TensorDesc, TensorDescMut, TensorLayout};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

const BATCH: u32 = 32;
const NUM_HEADS: u32 = 32;
const NUM_KV_HEADS: u32 = 8;
const HEAD_DIM: u32 = 128;
const BLOCK_SIZE: u32 = 16;
const SEQ_LEN: u32 = 4096;

fn bench_paged_attention_decode(c: &mut Criterion) {
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

    // Q shape is [B, H, 1, D]; the seq dim is 1 for decode.
    let q_elems = (BATCH * NUM_HEADS * HEAD_DIM) as usize;
    let q_buf = match DeviceBuffer::<f32>::zeroed(q_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (Q)");
            return;
        }
    };
    let o_buf = match DeviceBuffer::<f32>::zeroed(q_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (O)");
            return;
        }
    };

    // Allocate a single physical KV page for both K and V caches. The page
    // table maps every logical page index to physical page 0; this keeps the
    // device memory footprint tractable and still exercises the launch path.
    let page_elems = (BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM) as usize;
    let k_cache_buf = match DeviceBuffer::<f32>::zeroed(page_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (K-cache)");
            return;
        }
    };
    let v_cache_buf = match DeviceBuffer::<f32>::zeroed(page_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (V-cache)");
            return;
        }
    };

    // Page table: BATCH * max_pages_per_seq entries, zero-filled.
    let max_pages_per_seq = SEQ_LEN.div_ceil(BLOCK_SIZE);
    let page_table_elems = (BATCH * max_pages_per_seq) as usize;
    let page_table = match DeviceBuffer::<i32>::zeroed(page_table_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (page table)");
            return;
        }
    };

    // seq_lengths: each batch entry has SEQ_LEN tokens decoded.
    let mut seq_lengths = match DeviceBuffer::<i32>::alloc(BATCH as usize) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (seq_lengths)");
            return;
        }
    };
    let host_lens = vec![SEQ_LEN as i32; BATCH as usize];
    if seq_lengths.copy_from_host(&host_lens).is_err() {
        eprintln!("skip: copy seq_lengths failed");
        return;
    }

    let q_strides = vec![NUM_HEADS * HEAD_DIM, HEAD_DIM, HEAD_DIM, 1];
    let q = match TensorDesc::<f32>::from_raw(
        q_buf.as_device_ptr(),
        vec![BATCH, NUM_HEADS, 1, HEAD_DIM],
        q_strides.clone(),
        TensorLayout::Nchw,
    ) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: Q desc failed");
            return;
        }
    };
    let mut output = match TensorDescMut::<f32>::from_raw(
        o_buf.as_device_ptr(),
        vec![BATCH, NUM_HEADS, 1, HEAD_DIM],
        q_strides,
        TensorLayout::Nchw,
    ) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: O desc failed");
            return;
        }
    };

    let config = PagedAttentionConfig {
        head_dim: HEAD_DIM,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        block_size: BLOCK_SIZE,
        precision: PtxType::F32,
        sm_version: handle.sm_version(),
    };

    let tokens_per_call = u64::from(BATCH);

    // Warm-up call outside the timed loop, plus `.expect()` inside it below:
    // a silently-discarded `Result` here would let a future regression (or a
    // future workspace requirement, mirroring the `conv_forward` bug this
    // repository has already hit) time an early-return-on-error instead of
    // the kernel this benchmark claims to measure.
    paged_attention_decode(
        &handle,
        &q,
        k_cache_buf.as_device_ptr(),
        v_cache_buf.as_device_ptr(),
        &page_table,
        &seq_lengths,
        &mut output,
        &config,
    )
    .expect("paged_attention_decode warm-up call must succeed");
    handle
        .stream()
        .synchronize()
        .expect("synchronize after warm-up");

    let mut group = c.benchmark_group("dnn_p3_paged_attention_decode");
    group.throughput(Throughput::Elements(tokens_per_call));
    group.bench_function("oxicuda_f32_b32_seq4096_d128_h32_kv8", |b| {
        b.iter(|| {
            paged_attention_decode(
                &handle,
                &q,
                k_cache_buf.as_device_ptr(),
                v_cache_buf.as_device_ptr(),
                &page_table,
                &seq_lengths,
                &mut output,
                &config,
            )
            .expect("paged_attention_decode");
        });
    });
    group.finish();
}

criterion_group!(benches, bench_paged_attention_decode);
criterion_main!(benches);
