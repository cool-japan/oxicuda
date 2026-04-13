use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_lm::config::{GptConfig, LlamaConfig};
use oxicuda_lm::layer::{RotaryEmbedding, TokenEmbedding};
use oxicuda_lm::model::{Gpt2Model, LlamaModel};
use oxicuda_lm::tokenizer::BpeBuilder;

// --- BPE tokenizer construction + encode/decode --------------------------------

/// Build a minimal BPE tokenizer with a 256-byte base vocabulary and
/// `n_merges` merge rules so we can benchmark at various tokenizer sizes.
/// `BpeBuilder::build()` calls `Vocab::gpt2_byte_vocab()` internally.
fn build_bpe(n_merges: usize) -> oxicuda_lm::tokenizer::BpeTokenizer {
    let mut builder = BpeBuilder::new();

    // Add deterministic merge rules: consecutive byte pairs.
    // Each merge produces a new token id 256, 257, ...
    for i in 0..n_merges.min(128) {
        let left = vec![(i & 0xFF) as u8];
        let right = vec![((i + 1) & 0xFF) as u8];
        builder = builder.add_merge(&left, &right);
    }
    builder.build().unwrap()
}

fn bench_bpe_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("bpe_construction");

    for n in [0usize, 16, 64, 128] {
        group.bench_with_input(BenchmarkId::new("build", n), &n, |b, &n| {
            b.iter(|| black_box(build_bpe(n)));
        });
    }
    group.finish();
}

fn bench_bpe_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("bpe_encode");

    // Use 64 merges for a realistic tokenizer.
    let tok = build_bpe(64);

    let inputs: &[(&str, &str)] = &[
        ("short_8tok", "hello world"),
        (
            "medium_64tok",
            "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
        ),
        (
            "long_256tok",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        ),
    ];

    for &(name, text) in inputs {
        group.bench_with_input(BenchmarkId::new("encode", name), &text, |b, text| {
            b.iter(|| black_box(tok.encode(text).unwrap()));
        });
    }

    group.finish();
}

fn bench_bpe_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("bpe_decode");

    let tok = build_bpe(64);
    let text =
        "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
    let ids = tok.encode(text).unwrap();

    group.bench_function("decode_medium", |b| {
        b.iter(|| black_box(tok.decode(&ids).unwrap()));
    });

    let mut long_ids = ids.clone();
    for _ in 0..8 {
        long_ids.extend_from_slice(&ids);
    }
    group.bench_function("decode_long", |b| {
        b.iter(|| black_box(tok.decode(&long_ids).unwrap()));
    });

    group.finish();
}

// --- TokenEmbedding forward --------------------------------------------------

fn bench_token_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_embedding_forward");

    let configs: &[(&str, usize, usize, usize)] = &[
        ("v512_d64_seq8", 512, 64, 8),
        ("v1024_d128_seq32", 1_024, 128, 32),
        ("v50257_d768_seq8", 50_257, 768, 8),
        ("v50257_d768_seq64", 50_257, 768, 64),
        ("v128k_d4096_seq8", 128_000, 4096, 8),
    ];

    for &(name, vocab_size, embed_dim, seq_len) in configs {
        let emb = TokenEmbedding::new(vocab_size, embed_dim).unwrap();
        let ids: Vec<u32> = (0..seq_len as u32).map(|i| i % vocab_size as u32).collect();

        group.bench_with_input(BenchmarkId::new("forward", name), &name, |b, _| {
            b.iter(|| black_box(emb.forward(&ids).unwrap()));
        });
    }
    group.finish();
}

// --- RotaryEmbedding construction + apply ------------------------------------

fn bench_rope_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope_construction");

    let configs: &[(&str, usize, usize)] = &[
        ("hd64_pos512", 64, 512),
        ("hd128_pos2048", 128, 2048),
        ("hd128_pos8192", 128, 8192),
    ];

    for &(name, head_dim, max_pos) in configs {
        group.bench_with_input(
            BenchmarkId::new("new", name),
            &(head_dim, max_pos),
            |b, &(hd, mp)| {
                b.iter(|| black_box(RotaryEmbedding::new(hd, mp, 10_000.0).unwrap()));
            },
        );
    }
    group.finish();
}

fn bench_rope_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope_apply");

    let configs: &[(&str, usize, usize, usize, usize)] = &[
        ("4h_64d_seq8", 4, 64, 8, 0),
        ("8h_128d_seq32", 8, 128, 32, 0),
        ("32h_128d_seq64", 32, 128, 64, 0),
        ("32h_128d_seq64_off256", 32, 128, 64, 256),
        ("8h_64d_seq256", 8, 64, 256, 0),
    ];

    for &(name, n_heads, head_dim, seq_len, offset) in configs {
        let rope = RotaryEmbedding::new(head_dim, seq_len + offset + 16, 10_000.0).unwrap();

        group.bench_with_input(BenchmarkId::new("apply", name), &name, |b, _| {
            b.iter_batched(
                || vec![0.5_f32; n_heads * head_dim * seq_len],
                |mut x| {
                    rope.apply(&mut x, n_heads, seq_len, offset).unwrap();
                    black_box(x)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// --- GPT-2 model forward pass ------------------------------------------------

fn bench_gpt2_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpt2_forward");

    // tiny config: 2 layers, 2 heads, hidden=8, vocab=16, max_pos=32
    let cfg_tiny = GptConfig::tiny();
    let vocab = cfg_tiny.vocab_size as u32;
    let max_pos = cfg_tiny.n_positions;
    let model_tiny = Gpt2Model::new(cfg_tiny).unwrap();

    group.bench_function("tiny_seq8_no_cache", |b| {
        let ids: Vec<u32> = (0..8u32).map(|i| i % vocab).collect();
        b.iter(|| black_box(model_tiny.forward(&ids, None).unwrap()));
    });

    group.bench_function("tiny_seq16_no_cache", |b| {
        // Fill up to max_pos/2, cycling through vocab
        let seq_len = (max_pos / 2).min(16);
        let ids: Vec<u32> = (0..seq_len as u32).map(|i| i % vocab).collect();
        b.iter(|| black_box(model_tiny.forward(&ids, None).unwrap()));
    });

    // Incremental decode: one token at a time with growing KV cache (16 steps, within max_pos=32)
    group.bench_function("tiny_incremental_decode_16steps", |b| {
        let prompt: Vec<u32> = (0..4u32).map(|i| i % vocab).collect();
        b.iter(|| {
            let (_, mut kv) = model_tiny.forward(&prompt, None).unwrap();
            for step in 4u32..20 {
                let tok = step % vocab;
                let (_, new_kv) = model_tiny.forward(&[tok], Some(&kv)).unwrap();
                kv = new_kv;
            }
            black_box(kv)
        });
    });

    // next_token (greedy decode + argmax)
    group.bench_function("tiny_next_token_seq4", |b| {
        let ids: Vec<u32> = vec![0, 1, 2, 3];
        b.iter(|| black_box(model_tiny.next_token(&ids, None).unwrap()));
    });

    group.finish();
}

// --- LLaMA model forward pass ------------------------------------------------

fn bench_llama_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("llama_forward");

    let cfg_tiny = LlamaConfig::tiny();
    let llama_vocab = cfg_tiny.vocab_size as u32;
    let model_tiny = LlamaModel::new(cfg_tiny).unwrap();

    group.bench_function("tiny_seq8_no_cache", |b| {
        let ids: Vec<u32> = (0..8u32).map(|i| i % llama_vocab).collect();
        b.iter(|| black_box(model_tiny.forward(&ids, None).unwrap()));
    });

    group.bench_function("tiny_seq15_no_cache", |b| {
        // vocab_size=16, max_pos=32; use 15 tokens to stay well within both limits
        let ids: Vec<u32> = (0..15u32).map(|i| i % llama_vocab).collect();
        b.iter(|| black_box(model_tiny.forward(&ids, None).unwrap()));
    });

    // Incremental decode with KV cache (GQA path); cycle token IDs within vocab
    group.bench_function("tiny_incremental_decode_16steps", |b| {
        let prompt: Vec<u32> = (0..4u32).map(|i| i % llama_vocab).collect();
        b.iter(|| {
            let (_, mut kv) = model_tiny.forward(&prompt, None).unwrap();
            for step in 4u32..20 {
                let tok = step % llama_vocab;
                let (_, new_kv) = model_tiny.forward(&[tok], Some(&kv)).unwrap();
                kv = new_kv;
            }
            black_box(kv)
        });
    });

    group.bench_function("tiny_next_token_seq4", |b| {
        let ids: Vec<u32> = vec![0, 1, 2, 3];
        b.iter(|| black_box(model_tiny.next_token(&ids, None).unwrap()));
    });

    group.finish();
}

// --- KV cache operations (PastKvCache) ---------------------------------------

fn bench_kv_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("past_kv_cache");

    let cfg = LlamaConfig::tiny();
    let n_layers = cfg.n_layers;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim();

    group.bench_function("past_len", |b| {
        let model = LlamaModel::new(cfg.clone()).unwrap();
        let ids: Vec<u32> = (0..8u32).collect();
        let (_, kv) = model.forward(&ids, None).unwrap();
        b.iter(|| black_box(kv.past_len()));
    });

    // Measure GPT-2 forward 1 token at a time for 8 steps, accumulating cache
    group.bench_function("gpt2_tiny_grow_kv_8steps", |b| {
        let cfg_gpt = GptConfig::tiny();
        let model_gpt = Gpt2Model::new(cfg_gpt).unwrap();
        b.iter(|| {
            let (_, mut kv) = model_gpt.forward(&[0u32], None).unwrap();
            for t in 1u32..8 {
                let (_, new_kv) = model_gpt.forward(&[t], Some(&kv)).unwrap();
                kv = new_kv;
            }
            black_box(kv.past_len())
        });
    });

    // Access hidden fields via past_len + n_layers
    let _ = (n_layers, n_kv_heads, head_dim); // suppress unused warning

    group.finish();
}

// --- criterion wiring --------------------------------------------------------

criterion_group!(
    lm_benches,
    bench_bpe_construction,
    bench_bpe_encode,
    bench_bpe_decode,
    bench_token_embedding,
    bench_rope_construction,
    bench_rope_apply,
    bench_gpt2_forward,
    bench_llama_forward,
    bench_kv_cache,
);
criterion_main!(lm_benches);
