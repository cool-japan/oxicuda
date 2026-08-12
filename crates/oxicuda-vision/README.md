# oxicuda-vision

Vision Transformer & CLIP primitives for OxiCUDA -- ViT patch embedding,
multi-head self-attention, CLIP contrastive learning, FPN, RoI Align, and
DETR decoder, all in pure Rust.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project. See the
[workspace README](../../README.md) for the full crate map.

## Overview

`oxicuda-vision` provides the architectural pieces of modern vision deep
learning: a strided Conv2D `PatchEmbed`, sinusoidal and learnable positional
embeddings, a complete ViT block (pre-norm MHSA + MLP) plus encoder stack
and full `ViTModel`, a CLIP vision encoder + projection head + InfoNCE
contrastive loss, geometric / photometric / normalisation image
augmentations, a Feature Pyramid Network with lateral 1x1 convolutions and
top-down pathway, and a DETR-style decoder with `roi_align` and
`bipartite_match` primitives.

All forward passes operate on flat row-major `Vec<f32>` tensors so the same
code drives CPU unit tests, PTX kernel verification, and CPU-only
deployments. PTX kernels (`patch_embed`, `bilinear_interp`,
`contrastive_loss`, `roi_align`, `image_normalize`, `adaptive_avg_pool`,
`focal_loss`) are emitted for SM 7.5 through SM 12.0. The only crate
dependency is `thiserror`.

## Modules

| Module | Description |
|--------|-------------|
| `error` | `VisionError` / `VisionResult` |
| `handle` | `VisionHandle`, `SmVersion`, `LcgRng` |
| `patch_embed` | `PatchEmbed`, `PatchEmbedConfig`, `LearnablePosEmbed`, `pos_2d_sincos`, `add_pos_embed`, `prepend_cls` |
| `vit` | `ViTBlock`, `ViTEncoder`, `ViTModel`, `ViTConfig::tiny()` |
| `clip` | `ClipVisionEncoder`, `ClipVisionConfig`, `ProjectionHead`, `info_nce_loss` |
| `augment` | `AugOp`, `Pipeline`; `Resize`, `RandomCrop`, `HorizontalFlip`, ImageNet normalize |
| `fpn` | `FeatureMap`, `FpnConfig`, `Fpn`, `LateralConv1x1` |
| `detection` | `roi_align`, `bipartite_match`, `DetrDecoder`, `DetrConfig::tiny()` |
| `ptx_kernels` | PTX for the seven kernels listed above |

## Quick Start

```rust,no_run
use oxicuda_vision::prelude::*;

let mut rng = LcgRng::new(1);

// 32x32 RGB image, patch_size = 4, embed_dim = 16 -> 64 patch tokens.
let cfg = PatchEmbedConfig::new(32, 4, 3, 16)?;
let pe  = PatchEmbed::new(cfg.clone(), &mut rng);
let image = vec![0.5_f32; 3 * 32 * 32];
let tokens = pe.forward(&image)?;
assert_eq!(tokens.len(), cfg.n_patches() * cfg.embed_dim);

// Tiny ViT classifier (10-way).
let model = ViTModel::new(ViTConfig::tiny(), &mut rng)?;
let logits = model.forward(&image)?;
assert_eq!(logits.len(), 10);
# Ok::<(), VisionError>(())
```

## Status

| Item | Value |
|------|-------|
| Version | 0.5.2 |
| Release date | 2026-07-27 |
| Default features | Pure Rust (`thiserror` only) |
| `unwrap()` | 0 in production code |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
