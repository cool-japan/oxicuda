# oxicuda-infer

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-infer` (Vol.11) is a production-grade GPU inference engine for large language models built on the OxiCUDA stack. It implements the core algorithms for efficient LLM serving: PagedAttention KV cache, continuous batching, speculative decoding, beam search, and a rich sampling suite — all in pure Rust with no CUDA SDK dependency at compile time.

## Features

- **PagedAttention** (Kwon et al., 2023): fixed-size physical KV blocks with per-sequence block tables for non-contiguous allocation and O(1) prefix sharing
- **Continuous batching** (vLLM, Yu et al., 2022): sequences join and leave the GPU batch at token granularity for maximum throughput
- **Speculative decoding** (Chen et al., 2023): draft-model candidate tokens verified by the target model in a single pass; provably correct output distribution
- **Sampling suite**: greedy, top-k, top-p (nucleus), beam search, and speculative verification
- **Prefix cache**: token-sequence prefix deduplication across requests to reduce KV compute
- **`#![forbid(unsafe_code)]`** — fully safe Rust implementation

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxicuda-infer = "0.1.0"
```

```rust
use oxicuda_infer::batch::{BatcherConfig, ContinuousBatcher, SamplingParams};
use oxicuda_infer::cache::kv_cache::PagedKvCache;
use oxicuda_infer::error::InferResult;

// Build a KV cache: 4 layers, 4 kv-heads, head_dim=64, block_size=16, 128 blocks.
let kv_cache = PagedKvCache::new(4, 4, 64, 16, 128);
let mut batcher = ContinuousBatcher::new(BatcherConfig::default_test(), kv_cache);

let params = SamplingParams { max_new_tokens: 64, eos_token_id: Some(2), ..Default::default() };
batcher.add_request(vec![1, 2, 3], params);

// Plug in your GPU model function here.
let model_fn = |tokens: &[u32], btables: &[Vec<u32>], lens: &[usize]| -> InferResult<Vec<Vec<f32>>> {
    // ... GPU forward pass ...
    Ok(tokens.iter().map(|_| vec![0.0_f32; 32000]).collect())
};
let outputs = batcher.step(model_fn)?;
```

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
