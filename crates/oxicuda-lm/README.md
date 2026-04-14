# oxicuda-lm

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement for the COOLJAPAN ecosystem.

## Overview

`oxicuda-lm` provides the model-layer abstractions for LLM inference: a BPE tokenizer, transformer layer building blocks with incremental KV-cache support, and complete GPT-2 and LLaMA-2/3 model implementations. All forward passes are pure-Rust CPU reference implementations suitable for testing; GPU acceleration is provided by the included PTX kernel strings once a CUDA driver is available at runtime.

## Features

- **BPE tokenizer**: `BpeBuilder` / `BpeTokenizer` with full encode/decode round-trip and special-token support
- **Transformer layers**: `TokenEmbedding`, `LearnedPositionalEmbedding`, `RotaryEmbedding` (RoPE), `MultiHeadAttention` (with GQA), `MlpFfn`, `SwiGluFfn`, `RmsNorm`, `LayerNorm`
- **KV cache**: `LayerKvCache` and `PastKvCache` for incremental (token-by-token) decoding with correct cache accumulation
- **GPT-2 and LLaMA architectures**: `Gpt2Model` and `LlamaModel` with `forward()` and `next_token()` helpers
- **PTX kernels**: five GPU kernel source strings (embedding forward, RoPE apply, SiLU gate, RMSNorm, causal attention softmax) for SM75–SM120
- Pure Rust — no CUDA SDK, no C/Fortran at compile time

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oxicuda-lm = "0.1.1"
```

```rust
use oxicuda_lm::{Gpt2Model, GptConfig, PastKvCache};

let model = Gpt2Model::new(GptConfig::tiny())?;
let prompt = vec![1u32, 42, 7];

// Prefill
let (logits, kv) = model.forward(&prompt, None)?;

// Incremental decode
let (next_token, kv2) = model.next_token(&[*prompt.last().unwrap()], Some(&kv))?;
println!("next token id: {next_token}");
```

## License

Apache-2.0 — © 2026 COOLJAPAN OU (Team KitaSan)
