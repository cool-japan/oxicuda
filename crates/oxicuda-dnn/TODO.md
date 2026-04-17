# oxicuda-dnn TODO

GPU-accelerated deep learning primitives, serving as a pure Rust equivalent
to cuDNN. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.4).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 34,711 SLoC (89 files)
- **Estimated SLoC (estimation.md):** 396K--653K (median 525K)
- **Ratio:** ~5.8% of median estimate -- compact foundation with all major DNN primitives

### Completed

#### Core Infrastructure
- [x] types.rs -- TensorDesc, TensorDescMut, TensorLayout, Activation, ConvAlgorithm, ConvolutionDescriptor, pool_output_size
- [x] handle.rs -- DnnHandle central entry point
- [x] error.rs -- DnnError, DnnResult error types
- [x] ptx_helpers.rs -- Internal PTX generation helpers
- [x] tensor_util.rs -- Tensor utility functions (stride computation, layout validation)

#### Convolution -- Forward Propagation
- [x] conv/fprop/direct.rs -- Direct convolution (1x1, depthwise)
- [x] conv/fprop/im2col_gemm.rs -- Im2col + GEMM convolution
- [x] conv/fprop/implicit_gemm.rs -- Implicit GEMM convolution (fused im2col)
- [x] conv/fprop/winograd.rs -- Winograd convolution (F(2x2, 3x3) and F(4x4, 3x3))

#### Convolution -- Backward Data Gradient
- [x] conv/dgrad/implicit_gemm.rs -- Implicit GEMM backward data gradient
- [x] conv/dgrad/mod.rs -- Backward data gradient dispatch

#### Convolution -- Backward Weight Gradient
- [x] conv/wgrad/implicit_gemm.rs -- Implicit GEMM backward weight gradient
- [x] conv/wgrad/mod.rs -- Weight gradient dispatch

#### Convolution -- Infrastructure
- [x] conv/algo_select.rs -- Algorithm selection heuristics
- [x] conv/api.rs -- Public convolution API
- [x] conv/descriptor.rs -- Convolution descriptor management
- [x] conv/fused.rs -- Fused Conv+BN+ReLU operations
- [x] conv/mod.rs -- Convolution module organization

#### Convolution -- Transposed
- [x] conv/transpose_conv.rs -- TransposeConvConfig, TransposeConvPlan, col2im PTX kernel generation, weight reshape PTX (~25 tests)

#### Convolution -- 3D
- [x] conv/conv3d/ -- 3D convolution with im2col3d + GEMM, forward/backward/wgrad for volumetric data (video, medical imaging)

#### Convolution -- Depthwise Separable
- [x] conv/depthwise_separable.rs -- Fused depthwise + pointwise convolution in single kernel

#### Convolution -- Deformable
- [x] conv/deformable.rs -- DCNv2 with learnable sampling offsets, bilinear interpolation, forward+backward

#### Convolution -- FFT-based
- [x] conv/fft_conv.rs -- Frequency-domain convolution for large kernel sizes (7x7+)

#### Attention (FlashAttention)
- [x] attn/flash_attn/forward.rs -- FlashAttention-2 forward pass
- [x] attn/flash_attn/backward.rs -- FlashAttention-2 backward pass
- [x] attn/flash_attn/decode.rs -- FlashDecoding for autoregressive inference
- [x] attn/flash_attn/paged.rs -- PagedAttention with paged KV-cache
- [x] attn/flash_attn/hopper.rs -- FlashAttention-3 Hopper warp-specialized forward+backward, TMA, ping-pong async pipeline
- [x] attn/flash_attn/mod.rs -- FlashAttention module organization

#### Attention -- Supporting Components
- [x] attn/kv_cache.rs -- KV-cache management (append, evict, paged allocation)
- [x] attn/mha.rs -- Multi-head attention API
- [x] attn/rope.rs -- Rotary Position Embedding (RoPE)
- [x] attn/fused_rope_attn.rs -- Fused RoPE+attention (single kernel RoPE + attention forward)
- [x] attn/mod.rs -- Attention module organization

#### Mixture of Experts (MoE)
- [x] moe/routing.rs -- Top-K expert routing with load balancing
- [x] moe/permute.rs -- Token permutation for expert dispatch
- [x] moe/grouped_gemm.rs -- Grouped GEMM for variable-size expert batches
- [x] moe/fused_moe.rs -- Fused MoE kernel (routing + permute + GEMM + unpermute)
- [x] moe/mod.rs -- MoE module organization

#### Normalization
- [x] norm/layer_norm.rs -- Layer normalization (forward + backward)
- [x] norm/rms_norm.rs -- RMS normalization (LLM-optimized)
- [x] norm/batch_norm.rs -- Batch normalization (forward + backward, running stats)
- [x] norm/group_norm.rs -- Group normalization
- [x] norm/fused_norm.rs -- Fused residual Add + Norm operations
- [x] norm/mod.rs -- Normalization module organization

#### Pooling
- [x] pool/max_pool.rs -- Max pooling (forward + backward)
- [x] pool/avg_pool.rs -- Average pooling (forward + backward)
- [x] pool/adaptive_pool.rs -- Adaptive pooling (variable output size)
- [x] pool/global_pool.rs -- Global average/max pooling
- [x] pool/mod.rs -- Pooling module organization

#### Resize / Interpolation
- [x] resize/nearest.rs -- Nearest neighbor interpolation
- [x] resize/bilinear.rs -- Bilinear interpolation
- [x] resize/bicubic.rs -- Bicubic interpolation
- [x] resize/mod.rs -- Resize module organization

#### Quantization
- [x] quantize/fp8_quantize.rs -- FP8 (E4M3/E5M2) quantization and dequantization
- [x] quantize/int8_quantize.rs -- INT8 symmetric/asymmetric quantization
- [x] quantize/block_scale.rs -- Block-wise scaling for quantized tensors
- [x] quantize/mod.rs -- Quantization module organization

#### RNN Module
- [x] rnn/lstm.rs -- LSTM cell forward pass (single timestep and sequence)
- [x] rnn/gru.rs -- GRU cell forward pass (single timestep and sequence)

#### INT4 Quantization
- [x] quantize/int4_quantize.rs -- INT4/NF4 quantization and dequantization with group scaling (QLoRA support)

### Future Enhancements

#### P0 -- Critical (LLM/Transformer Performance)
- [x] FlashAttention-3 (Hopper async pipeline) (attn/flash_attn/hopper.rs) -- leverage Hopper TMA and warp-specialized async pipeline for 2x attention throughput
- [x] GQA/MQA native support (attn/gqa.rs) -- grouped-query and multi-query attention without head duplication overhead
- [x] Sliding window attention (attn/sliding_window.rs) -- efficient local attention with configurable window size (Mistral-style)
- [x] Fused GEMM+bias+activation epilogue (fused_linear.rs) -- fused_linear with all activations, eliminate kernel launch overhead for FC layers
- [x] Fused RoPE+attention (attn/fused_rope_attn.rs) -- combine rotary embedding application with attention forward in a single kernel

#### P1 -- Important (Convolution and MoE Completeness)
- [x] FFT-based convolution (conv/fft_conv.rs) -- frequency-domain convolution for large kernel sizes (7x7+)
- [x] 3D convolution (conv/conv3d/) -- im2col3d + GEMM, forward/backward/wgrad for volumetric data
- [x] Deformable convolution (conv/deformable.rs) -- learnable sampling offsets (DCNv2) with bilinear interpolation, forward+backward
- [x] Transposed convolution -- fractionally-strided convolution for upsampling (conv/transpose_conv.rs; TransposeConvConfig, TransposeConvPlan with col2im and weight reshape PTX kernels; ~25 tests)
- [x] Winograd backward pass optimization -- dgrad and wgrad through Winograd domain (dgrad/winograd.rs, wgrad/winograd.rs)
- [x] Depthwise separable conv fusion (conv/depthwise_separable.rs) -- fused DW+PW kernel
- [x] MoE load balancing monitoring (moe/monitoring.rs) -- runtime expert utilization tracking and imbalance detection
- [x] Expert capacity factor tuning (moe/capacity.rs) -- dynamic capacity adjustment based on routing statistics
- [x] MoE auxiliary loss computation (moe/aux_loss.rs) -- on-device load balancing loss (Switch Transformer style) + z-loss

#### P2 -- Nice-to-Have (Advanced Features)
- [x] Block-sparse attention -- structured sparsity patterns for long-context attention (attn/block_sparse.rs)
- [x] Ring attention for sequence parallelism -- distribute long sequences across multiple GPUs (attn/ring_attention.rs)
- [x] Speculative decoding KV-cache (attn/speculative_decode.rs) -- draft/target KV-cache management with checkpoint/rollback, verification, rejection sampling PTX
- [x] Quantization-aware training (QAT) -- fake quantization with straight-through estimator (quantize/qat.rs)
- [x] GPTQ/AWQ quantization schemes -- weight-only quantization with calibration data (quantize/gptq_awq.rs)
- [x] InstanceNorm (norm/instance_norm.rs) -- per (batch, channel) normalization for style transfer and image generation
- [x] PowerNorm (norm/power_norm.rs) -- running power mean normalization for improved training stability
- [x] ScaleNorm (norm/scale_norm.rs) -- simplified L2 normalization using learned scale parameter
- [x] Dynamic batching (serving/dynamic_batching.rs) -- request-level dynamic batching with configurable max batch size, timeout, and padding strategies for inference serving (P1)
- [x] Continuous batching (serving/continuous_batching.rs) -- iteration-level continuous batching (Orca-style) with in-flight request scheduling, preemption, and KV-cache management for LLM serving (P1)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper (libloading) | Yes (runtime FFI only) |
| oxicuda-memory | Device/Host memory management | Yes |
| oxicuda-launch | Type-safe kernel launch | Yes |
| oxicuda-blas | GEMM and BLAS primitives for conv/attention | Yes |
| oxicuda-ptx | PTX code generation DSL | Yes |
| thiserror | Error derive macros | Yes |
| half | FP16/BF16 types (optional, feature-gated) | Yes |

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 960 passing
- unwrap() calls: 0 (production code)

## Performance Targets

From estimation.md -- representative benchmark sizes:

| Kernel | Sizes | Priority |
|--------|-------|----------|
| Conv2D | ResNet-50 layer sizes (all stages) | High |
| FlashAttention | seq in {512, 2048, 4096, 8192}, d in {64, 128} | Highest |
| MoE | Mixtral-8x7B pattern (8 experts, top-2) | High |

Target: 95% of cuDNN throughput for typical sizes on sm_80+.
Relaxed targets: 80% for small batch sizes, 90% for sm_75.

## Estimation vs Actual

| Metric | Estimated (estimation.md) | Actual |
|--------|---------------------------|--------|
| SLoC | 396K--653K (median 525K) | 31,293 SLoC |
| Files | ~28+ subcomponents listed | 89 |
| Development time | 13--22 days | Completed in Vol.3+4 batch |
| AI generation ratio | 62% | -- |
| Highest difficulty items | FlashAttn, Fused MoE (rated "extremely high") | Implemented |

The gap between estimate and actual reflects the estimation targeting full
production-grade cuDNN parity (all template expansions, precision variants per
architecture, and exhaustive numerical accuracy test suites), whereas the current
implementation provides complete API coverage with PTX generation delegated to
oxicuda-ptx. Key high-difficulty components (FlashAttention-2, PagedAttention,
Fused MoE, Winograd convolution) are all present.

---

## Blueprint Quality Gates (Vol.4 Sec 10)

### Functional Requirements

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| D1 | Conv2D forward — Implicit GEMM NHWC layout | P0 | [x] |
| D2 | Conv2D — 1×1 convolution and depthwise convolution | P0 | [x] |
| D3 | Conv2D backward — dgrad (input gradient) and wgrad (filter gradient) | P1 | [x] |
| D4 | Winograd 3×3 convolution — F(2×2,3×3) and F(4×4,3×3) transforms | P2 | [x] |
| D5 | FlashAttention forward — FP16 causal mask | P0 | [x] |
| D6 | FlashAttention backward | P1 | [x] |
| D7 | PagedAttention decode — variable-length KV-cache | P0 | [x] |
| D8 | RoPE (Rotary Positional Embedding) — in-place Q/K application | P0 | [x] |
| D9 | MoE routing — Top-K softmax selection | P0 | [x] |
| D10 | Fused MoE kernel — token-parallel and expert-parallel strategies | P0 | [x] |
| D11 | LayerNorm — per-row normalization with gamma/beta | P0 | [x] |
| D12 | RMSNorm — LLaMA-style without mean centering | P0 | [x] |
| D13 | BatchNorm forward — inference mode | P0 | [x] |
| D14 | BatchNorm backward — training mode gradient | P1 | [x] |
| D15 | GroupNorm | P2 | [x] |
| D16 | MaxPool2D and AvgPool2D | P0 | [x] |
| D17 | AdaptiveAvgPool2D — including global average pooling | P0 | [x] |
| D18 | FP8 quantize / dequantize (Hopper+) | P1 | [x] |
| D19 | Fused Conv + BatchNorm + ReLU | P0 | [x] |
| D20 | Fused Add + RMSNorm (Transformer Add&Norm pattern) | P0 | [x] |

### Performance Requirements

| # | Requirement | Target | Status |
|---|-------------|--------|--------|
| P1 | Conv2D ResNet-50 layer3 | ≥ 90% cuDNN throughput | [ ] |
| P2 | FlashAttention seq=2048, d=128, FP16 | ≥ 90% FlashAttention-2 throughput | [ ] |
| P3 | PagedAttention decode, batch=32, seq=4096 | ≥ 85% vLLM throughput | [ ] |
| P4 | MoE Mixtral-8x7B pattern | ≥ 90% FlashInfer throughput | [ ] |
| P5 | LayerNorm D=4096, FP16 | ≥ 95% cuDNN throughput | [ ] |
| P6 | RMSNorm D=4096, FP16 | ≥ 95% cuDNN throughput | [ ] |
| P7 | BatchNorm inference | ≥ 95% cuDNN throughput | [ ] |
| P8 | Conv + BN + ReLU fused vs unfused | ≥ 2× speedup | [ ] |

---

## Numerical Accuracy Requirements (Vol.4 Sec 10.3)

| Operation | FP16 Tolerance | FP32 Tolerance |
|-----------|---------------|----------------|
| Conv2D | < 1e-2 | < 1e-5 |
| FlashAttention | < 2e-2 | < 1e-5 |
| LayerNorm | < 1e-3 | < 1e-6 |
| Softmax | < 1e-3 | < 1e-6 |
| BatchNorm | < 1e-3 | < 1e-6 |
| RMSNorm | < 1e-3 | < 1e-6 |

---

## FlashInfer Porting Reference (Vol.4 Sec 11)

### Migration Priority Phases

| Phase | Week | Components | Use Case |
|-------|------|------------|----------|
| Phase 1 | Week 4 | FlashAttention + RoPE | TrustformeRS / oxionnx inference |
| Phase 2 | Week 5 | MoE + PagedAttention | Mixtral / DeepSeek MoE models |
| Phase 3 | Week 5–6 | Conv + Normalization + Pooling | SCRFD / INSwapper CNNs + SciRS2 |

### FlashInfer LOC Breakdown (reference: 4.8M LOC total)
- Core kernel logic: ~300K–500K unique LOC
- Template expansion variants: ~4M+ LOC
- Key template axes: head_dim (64/128/256) × precision (f16/bf16/fp8) × sm (80/89/90) × causal (true/false) × page_size (16/32/64)

---

## Architecture-Specific Deepening Opportunities

### Ampere (sm_80 / sm_86)
- [x] FlashAttention-2 tile: 128×64 or 64×128 with 4 warps — verify optimal tile selection
- [x] Implicit GEMM `cp.async` pipeline for NHWC convolution

### Ada Lovelace (sm_89)
- [x] FP8 quantization (`e4m3`/`e5m2`) for inference paths
- [x] INT8 block scaling for GPTQ/AWQ quantized model inference

### Hopper (sm_90 / sm_90a)
- [x] FlashAttention-3 with `wgmma` + TMA — 128×128 tile, 8 warps
- [x] PagedAttention with TMA-based KV-cache loading (config + layout verified; full TMA execution requires GPU)
- [x] FP8 GEMM epilogue for MoE experts

### Blackwell (sm_100 / sm_120)
- [x] FP4 / NVFP4 / OCP MXFP4 block-scaled quantization
- [x] 5th-gen Tensor Core (`tcgen05`) attention kernels

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These represent the gap between current implementation depth and blueprint-grade production requirements.

### Verification Gaps
- [ ] All D1–D20 functional requirements verified on GPU hardware
- [x] Numerical accuracy test suite for all 6 operation types × FP16/FP32
- [ ] All 8 performance requirements measured and documented

### Implementation Deepening
- [x] MoE routing strategy selection boundary verified (num_tokens < num_experts*2 → TokenParallel) — 10 tests
- [x] Top-K softmax weights verified: sum to 1.0, correct indices, deterministic
- [x] Conv+BN+ReLU coefficient folding math verified
- [x] LayerNorm formula verified: output mean≈0, var≈1
- [x] RMSNorm formula verified: shift-invariance distinction from LayerNorm confirmed
- [x] Winograd F(4×4,3×3): 4× multiplication reduction vs baseline conv measured
- [x] PagedAttention: GQA (Grouped Query Attention) with `num_kv_heads < num_heads` verified
- [x] FlashAttention causal mask: triangular mask correctness vs unfused reference < 2e-2 FP16
- [x] FlashAttention-3 Hopper kernel bodies implemented: real MMA (mma.sync.aligned.m16n8k16), ldmatrix, warp shuffle, TMA (cp.async.bulk), wgmma.mma_async instructions emitted
- [x] PTX generation verified: tests assert real instruction mnemonics present (not pseudocode)
