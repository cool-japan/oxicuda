//! Fused Mixture-of-Experts forward — target: P4 (Mixtral-8x7B pattern).
//!
//! Measures throughput of [`oxicuda_dnn::moe::fused_moe`] with the Mixtral-8x7B
//! MoE configuration:
//!
//! - `num_experts = 8`
//! - `top_k = 2`
//! - `hidden_dim = 4096`
//! - `intermediate_dim = 14336`
//! - SiLU activation between the two FFN projections
//! - `num_tokens = 4` (decode-phase batch — selects token-parallel strategy)
//!
//! Throughput is reported as **tokens/sec**.
//!
//! Skips on any host without an NVIDIA driver / GPU.

use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_dnn::DnnHandle;
use oxicuda_dnn::moe::{MoeConfig, fused_moe};
use oxicuda_dnn::types::{Activation, TensorDesc, TensorDescMut, TensorLayout};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

const NUM_TOKENS: u32 = 4;
const NUM_EXPERTS: u32 = 8;
const TOP_K: u32 = 2;
const HIDDEN_DIM: u32 = 4096;
const INTERMEDIATE_DIM: u32 = 14336;

fn bench_moe_mixtral_8x7b(c: &mut Criterion) {
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

    let in_elems = (NUM_TOKENS * HIDDEN_DIM) as usize;
    let w1_elems = (NUM_EXPERTS * HIDDEN_DIM * INTERMEDIATE_DIM) as usize;
    let w2_elems = (NUM_EXPERTS * INTERMEDIATE_DIM * HIDDEN_DIM) as usize;
    let routing_slots = (NUM_TOKENS * TOP_K) as usize;

    let in_buf = match DeviceBuffer::<f32>::zeroed(in_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (input)");
            return;
        }
    };
    let w1_buf = match DeviceBuffer::<f32>::zeroed(w1_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!(
                "skip: device alloc failed (w1, ~{} GiB)",
                (w1_elems * 4) >> 30
            );
            return;
        }
    };
    let w2_buf = match DeviceBuffer::<f32>::zeroed(w2_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (w2)");
            return;
        }
    };
    let out_buf = match DeviceBuffer::<f32>::zeroed(in_elems) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (output)");
            return;
        }
    };

    // Expert indices/weights: top-k assignments per token.
    let expert_indices = match DeviceBuffer::<i32>::zeroed(routing_slots) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (expert_indices)");
            return;
        }
    };
    let expert_weights = match DeviceBuffer::<f32>::zeroed(routing_slots) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skip: device alloc failed (expert_weights)");
            return;
        }
    };

    let input = match TensorDesc::<f32>::from_raw(
        in_buf.as_device_ptr(),
        vec![NUM_TOKENS, HIDDEN_DIM],
        vec![HIDDEN_DIM, 1],
        TensorLayout::RowMajor,
    ) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: input desc failed");
            return;
        }
    };
    let w1 = match TensorDesc::<f32>::from_raw(
        w1_buf.as_device_ptr(),
        vec![NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM],
        vec![HIDDEN_DIM * INTERMEDIATE_DIM, INTERMEDIATE_DIM, 1],
        TensorLayout::RowMajor,
    ) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: w1 desc failed");
            return;
        }
    };
    let w2 = match TensorDesc::<f32>::from_raw(
        w2_buf.as_device_ptr(),
        vec![NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM],
        vec![INTERMEDIATE_DIM * HIDDEN_DIM, HIDDEN_DIM, 1],
        TensorLayout::RowMajor,
    ) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: w2 desc failed");
            return;
        }
    };
    let mut output = match TensorDescMut::<f32>::from_raw(
        out_buf.as_device_ptr(),
        vec![NUM_TOKENS, HIDDEN_DIM],
        vec![HIDDEN_DIM, 1],
        TensorLayout::RowMajor,
    ) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: output desc failed");
            return;
        }
    };

    let config = MoeConfig {
        num_experts: NUM_EXPERTS,
        top_k: TOP_K,
        hidden_dim: HIDDEN_DIM,
        intermediate_dim: INTERMEDIATE_DIM,
        activation: Activation::Silu,
        precision: PtxType::F32,
        sm_version: handle.sm_version(),
    };

    let tokens_per_call = u64::from(NUM_TOKENS);

    // Warm-up call outside the timed loop, plus `.expect()` inside it below:
    // a silently-discarded `Result` here would let a future regression (or a
    // future workspace requirement, mirroring the `conv_forward` bug this
    // repository has already hit) time an early-return-on-error instead of
    // the kernel this benchmark claims to measure.
    fused_moe(
        &handle,
        &input,
        &w1,
        &w2,
        &expert_indices,
        &expert_weights,
        &mut output,
        &config,
    )
    .expect("fused_moe warm-up call must succeed");
    handle
        .stream()
        .synchronize()
        .expect("synchronize after warm-up");

    let mut group = c.benchmark_group("dnn_p4_moe_mixtral_8x7b");
    group.throughput(Throughput::Elements(tokens_per_call));
    group.bench_function("oxicuda_f32_e8_topk2_h4096_i14336", |b| {
        b.iter(|| {
            fused_moe(
                &handle,
                &input,
                &w1,
                &w2,
                &expert_indices,
                &expert_weights,
                &mut output,
                &config,
            )
            .expect("fused_moe");
        });
    });
    group.finish();
}

criterion_group!(benches, bench_moe_mixtral_8x7b);
criterion_main!(benches);
