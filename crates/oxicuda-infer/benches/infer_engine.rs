use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_infer::batch::{
    BatcherConfig, ContinuousBatcher, SamplingParams, Scheduler, SchedulerConfig,
};
use oxicuda_infer::cache::kv_cache::{BlockId, PagedKvCache};
use oxicuda_infer::executor::paged_attention_cpu;
use oxicuda_infer::sampling::{
    BeamSearchConfig, BeamSearchState, Rng, greedy_sample, greedy_sample_batch, speculative_verify,
    top_k_sample, top_p_sample,
};

// --- RNG / data helpers -------------------------------------------------------

fn make_rng() -> Rng {
    Rng::new(0x1234_5678_9abc_def0)
}

/// Create a pseudo-random logits vector of length `vocab_size`.
fn make_logits(vocab_size: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..vocab_size)
        .map(|_| rng.next_f32() * 10.0 - 5.0)
        .collect()
}

fn make_logits_batch(batch: usize, vocab_size: usize) -> Vec<Vec<f32>> {
    (0..batch)
        .map(|i| make_logits(vocab_size, i as u64 + 1))
        .collect()
}

fn make_uniform_probs(vocab_size: usize) -> Vec<f32> {
    vec![1.0 / vocab_size as f32; vocab_size]
}

// --- PagedKvCache block allocation -------------------------------------------

fn bench_paged_kv_cache_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_kv_cache_alloc");

    // (label, n_layers, n_kv_heads, head_dim, block_size, n_blocks)
    let configs: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("4l_8h_64d_bs16_512blk", 4, 8, 64, 16, 512),
        ("32l_8h_128d_bs16_1024blk", 32, 8, 128, 16, 1_024),
        ("40l_8h_128d_bs8_2048blk", 40, 8, 128, 8, 2_048),
    ];

    for &(name, nl, nkv, hd, bs, nblk) in configs {
        group.bench_with_input(
            BenchmarkId::new("construction", name),
            &(nl, nkv, hd, bs, nblk),
            |b, &(nl, nkv, hd, bs, nblk)| {
                b.iter(|| black_box(PagedKvCache::new(nl, nkv, hd, bs, nblk)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("alloc_one_block", name),
            &(nl, nkv, hd, bs, nblk),
            |b, &(nl, nkv, hd, bs, nblk)| {
                b.iter_batched(
                    || PagedKvCache::new(nl, nkv, hd, bs, nblk),
                    |mut cache| black_box(cache.alloc_block().unwrap()),
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("alloc_64_blocks", name),
            &(nl, nkv, hd, bs, nblk),
            |b, &(nl, nkv, hd, bs, nblk)| {
                b.iter_batched(
                    || PagedKvCache::new(nl, nkv, hd, bs, nblk),
                    |mut cache| black_box(cache.alloc_blocks(64).unwrap()),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

// --- paged_attention_cpu (GQA) -----------------------------------------------

fn bench_paged_attention_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_attention_cpu");

    // (label, n_heads, n_kv_heads, head_dim, seq_len, block_size)
    let configs: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("mha_8h_64d_seq32", 8, 8, 64, 32, 16),
        ("mha_8h_64d_seq128", 8, 8, 64, 128, 16),
        ("gqa_8h_2kv_64d_seq128", 8, 2, 64, 128, 16),
        ("gqa_32h_4kv_128d_seq256", 32, 4, 128, 256, 16),
        ("gqa_32h_4kv_128d_seq512", 32, 4, 128, 512, 16),
    ];

    for &(name, nh, nkv, hd, seq_len, bs) in configs {
        let n_layers = 4;
        let n_blocks = (seq_len / bs) + 4;

        let mut cache = PagedKvCache::new(n_layers, nkv, hd, bs, n_blocks);
        let block_ids: Vec<BlockId> = cache.alloc_blocks(n_blocks).unwrap();

        // Fill the cache with synthetic kv data
        let stride = nkv * hd;
        let k_tok = vec![0.1_f32; stride];
        let v_tok = vec![0.2_f32; stride];
        let mut tokens_filled = 0usize;
        for &bid in &block_ids {
            for _ in 0..bs {
                if tokens_filled >= seq_len {
                    break;
                }
                cache.append_token(bid, 0, &k_tok, &v_tok).ok();
                tokens_filled += 1;
            }
        }

        let q = vec![0.5_f32; nh * hd];
        let scale = (hd as f32).powf(-0.5);

        group.bench_with_input(
            BenchmarkId::new("gqa_attention", name),
            &(nh, nkv, hd, seq_len, bs),
            |b, _| {
                b.iter(|| {
                    black_box(
                        paged_attention_cpu(
                            &q,
                            &cache,
                            &block_ids,
                            tokens_filled,
                            0,
                            nh,
                            nkv,
                            hd,
                            bs,
                            scale,
                        )
                        .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();
}

// --- Sampling strategies -----------------------------------------------------

fn bench_sampling_greedy(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_greedy");

    for &vocab in &[512usize, 4096, 32_000, 128_000] {
        let logits = make_logits(vocab, 42);
        group.bench_with_input(BenchmarkId::new("single", vocab), &vocab, |b, _| {
            b.iter(|| black_box(greedy_sample(&logits).unwrap()))
        });
    }

    // batched greedy
    for &batch in &[1usize, 4, 16, 64] {
        let batch_logits = make_logits_batch(batch, 32_000);
        group.bench_with_input(
            BenchmarkId::new("batch_32k_vocab", batch),
            &batch,
            |b, _| b.iter(|| black_box(greedy_sample_batch(&batch_logits).unwrap())),
        );
    }
    group.finish();
}

fn bench_sampling_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_top_k");

    let vocab_sizes = [32_000usize, 128_000];
    let k_values = [10usize, 50, 200];

    for &vocab in &vocab_sizes {
        for &k in &k_values {
            let logits = make_logits(vocab, 99);
            let mut rng = make_rng();
            group.bench_with_input(
                BenchmarkId::new(format!("vocab{vocab}"), k),
                &(vocab, k),
                |b, _| b.iter(|| black_box(top_k_sample(&logits, k, &mut rng).unwrap())),
            );
        }
    }
    group.finish();
}

fn bench_sampling_top_p(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_top_p");

    let vocab_sizes = [32_000usize, 128_000];
    let p_values = [0.9_f32, 0.95, 0.99];

    for &vocab in &vocab_sizes {
        for &p in &p_values {
            let logits = make_logits(vocab, 77);
            let mut rng = make_rng();
            let label = format!("vocab{vocab}_p{}", (p * 100.0) as u32);
            group.bench_with_input(BenchmarkId::new("nucleus", &label), &(vocab, p), |b, _| {
                b.iter(|| black_box(top_p_sample(&logits, p, &mut rng).unwrap()))
            });
        }
    }
    group.finish();
}

// --- Speculative decoding verification ---------------------------------------

fn bench_speculative_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("speculative_verify");

    let vocab = 32_000usize;

    for &draft_len in &[4usize, 8, 16] {
        let draft_tokens: Vec<u32> = (0..draft_len as u32).collect();
        let draft_probs: Vec<Vec<f32>> = (0..draft_len)
            .map(|i| {
                make_uniform_probs(vocab)
                    .into_iter()
                    .enumerate()
                    .map(|(j, _)| {
                        if j == i {
                            0.5
                        } else {
                            0.5 / (vocab - 1) as f32
                        }
                    })
                    .collect()
            })
            .collect();
        let target_probs: Vec<Vec<f32>> = (0..draft_len + 1)
            .map(|_| make_uniform_probs(vocab))
            .collect();

        group.bench_with_input(BenchmarkId::new("verify", draft_len), &draft_len, |b, _| {
            let mut rng = make_rng();
            b.iter(|| {
                black_box(
                    speculative_verify(&draft_tokens, &draft_probs, &target_probs, &mut rng)
                        .unwrap(),
                )
            });
        });
    }
    group.finish();
}

// --- Beam search step --------------------------------------------------------

fn bench_beam_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("beam_search_step");

    let vocab = 32_000usize;

    for &beam_width in &[2usize, 4, 8] {
        let config = BeamSearchConfig {
            beam_width,
            length_penalty: 0.6,
            max_new_tokens: 256,
            eos_token_id: 2,
        };
        let logits_per_beam: Vec<Vec<f32>> = (0..beam_width)
            .map(|i| make_logits(vocab, i as u64 + 1))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("step_32k_vocab", beam_width),
            &beam_width,
            |b, _| {
                b.iter_batched(
                    || BeamSearchState::new(config.clone()),
                    |mut state| black_box(state.step(&logits_per_beam).unwrap()),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

// --- Scheduler (FCFS) --------------------------------------------------------

fn bench_scheduler(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler");

    let config = SchedulerConfig::new(8, 512, 16, 256);

    group.bench_function("add_request", |b| {
        b.iter_batched(
            || Scheduler::new(config.clone()),
            |mut sched| {
                let id =
                    sched.add_request(vec![1u32, 2, 3, 4, 5, 6, 7, 8], SamplingParams::greedy(128));
                black_box(id)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("schedule_16_reqs", |b| {
        b.iter_batched(
            || {
                let mut sched = Scheduler::new(config.clone());
                for i in 0..16u32 {
                    sched.add_request(vec![i; 16], SamplingParams::greedy(64));
                }
                sched
            },
            |mut sched| black_box(sched.schedule()),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("schedule_64_reqs", |b| {
        let config64 = SchedulerConfig::new(64, 4096, 16, 1024);
        b.iter_batched(
            || {
                let mut sched = Scheduler::new(config64.clone());
                for i in 0..64u32 {
                    sched.add_request(vec![i; 8], SamplingParams::greedy(32));
                }
                sched
            },
            |mut sched| black_box(sched.schedule()),
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// --- ContinuousBatcher step --------------------------------------------------

fn bench_continuous_batcher(c: &mut Criterion) {
    let mut group = c.benchmark_group("continuous_batcher");

    // Build a small batcher and measure one decode step with a trivial model.
    let vocab = 256usize;
    let eos = 1u32;

    let model_fn = |tokens: &[u32], _bt: &[Vec<u32>], _lens: &[usize]| {
        // Dummy: always softly predict EOS for each token
        Ok(tokens
            .iter()
            .map(|_| {
                let mut v = vec![0.01_f32; vocab];
                v[eos as usize] = 10.0;
                v
            })
            .collect::<Vec<_>>())
    };

    for &n_seqs in &[1usize, 4, 16] {
        let label = format!("{n_seqs}_seqs");
        group.bench_with_input(BenchmarkId::new("step", &label), &n_seqs, |b, &n_seqs| {
            b.iter_batched(
                || {
                    let kv = oxicuda_infer::PagedKvCache::new(2, 2, 32, 8, 256);
                    let cfg = BatcherConfig {
                        scheduler: SchedulerConfig::new(n_seqs + 2, 512, 8, 256),
                        vocab_size: vocab,
                        seed: 42,
                    };
                    let mut batcher = ContinuousBatcher::new(cfg, kv);
                    for i in 0..n_seqs as u32 {
                        batcher.add_request(
                            vec![10 + i; 4],
                            SamplingParams {
                                max_new_tokens: 4,
                                eos_token_id: Some(eos),
                                ..Default::default()
                            },
                        );
                    }
                    batcher
                },
                |mut batcher| black_box(batcher.step(model_fn).unwrap()),
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// --- criterion wiring --------------------------------------------------------

criterion_group!(
    infer_benches,
    bench_paged_kv_cache_alloc,
    bench_paged_attention_cpu,
    bench_sampling_greedy,
    bench_sampling_top_k,
    bench_sampling_top_p,
    bench_speculative_verify,
    bench_beam_search,
    bench_scheduler,
    bench_continuous_batcher,
);
criterion_main!(infer_benches);
