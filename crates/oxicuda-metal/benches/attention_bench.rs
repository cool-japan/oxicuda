//! Metal scaled dot-product attention benchmark — measures
//! `ComputeBackend::attention` at a typical single-layer transformer shape:
//! batch 1, 8 heads, sequence length 256 (query and key/value equal), head
//! dimension 64, causal masking on.
//!
//! ## Platform behaviour
//!
//! * **macOS with a Metal-capable GPU** — the benchmark runs.
//! * **No Metal device (headless CI, non-macOS)** — prints
//!   `skip: no Metal device (attention)` to stderr and returns before
//!   registering any benchmark closure with criterion.
//!
//! Run with:
//! ```bash
//! cargo bench -p oxicuda-metal --bench attention_bench
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_backend::ComputeBackend;
use oxicuda_metal::MetalBackend;

const BATCH: usize = 1;
const HEADS: usize = 8;
const SEQ: usize = 256;
const HEAD_DIM: usize = 64;

/// GPU resources shared across iterations of the criterion loop.
struct AttentionHarness {
    backend: MetalBackend,
    q: u64,
    k: u64,
    v: u64,
    o: u64,
}

fn upload_filled(backend: &MetalBackend, elems: usize, seed: u32) -> Option<u64> {
    let ptr = backend.alloc(elems * 4).ok()?;
    // Small values so the softmax stays numerically tame across a 256-wide row.
    let fill: Vec<f32> = (0..elems)
        .map(|i| ((((i as u32).wrapping_add(seed)) % 200) as f32 - 100.0) * 0.005)
        .collect();
    let fill_bytes: Vec<u8> = fill.iter().flat_map(|v| v.to_le_bytes()).collect();
    backend.copy_htod(ptr, &fill_bytes).ok()?;
    Some(ptr)
}

fn try_setup() -> Option<AttentionHarness> {
    let mut backend = MetalBackend::new();
    backend.init().ok()?;

    let qkv_elems = BATCH * HEADS * SEQ * HEAD_DIM;
    let q = upload_filled(&backend, qkv_elems, 1)?;
    let k = upload_filled(&backend, qkv_elems, 11)?;
    let v = upload_filled(&backend, qkv_elems, 23)?;
    let o = backend.alloc(qkv_elems * 4).ok()?;

    Some(AttentionHarness {
        backend,
        q,
        k,
        v,
        o,
    })
}

fn bench_attention(criterion: &mut Criterion) {
    let harness = match try_setup() {
        Some(h) => h,
        None => {
            eprintln!("skip: no Metal device (attention)");
            return;
        }
    };

    let scale = 1.0 / (HEAD_DIM as f64).sqrt();

    let mut group = criterion.benchmark_group("metal_attention_b1_h8_s256_d64_causal");
    group.bench_function("attention", |bencher| {
        bencher.iter(|| {
            let r = harness.backend.attention(
                black_box(harness.q),
                black_box(harness.k),
                black_box(harness.v),
                harness.o,
                BATCH,
                HEADS,
                SEQ,
                SEQ,
                HEAD_DIM,
                scale,
                true,
            );
            black_box(r.ok());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_attention);
criterion_main!(benches);
