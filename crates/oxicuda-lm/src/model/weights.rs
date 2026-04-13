//! Weight initialisation helpers for GPT-2 and LLaMA model skeletons.
//!
//! These functions populate the mutable layer fields of a model from a
//! flat [`ModelWeights`] dictionary, mirroring the HuggingFace naming
//! convention so that real checkpoints can be loaded by name.

use crate::config::{GptConfig, LlamaConfig};
use crate::error::{LmError, LmResult};
use crate::layer::transformer::{GptBlock, LlamaBlock};
use crate::weights::{ModelWeights, WeightTensor};

// ─── GPT-2 layer loading ─────────────────────────────────────────────────────

/// Load weights into a `GptBlock` from a `ModelWeights` dictionary.
///
/// Expected keys (with `prefix = "transformer.h.{layer_idx}"`):
///
/// ```text
/// {prefix}.ln_1.weight   [hidden_dim]
/// {prefix}.ln_1.bias     [hidden_dim]
/// {prefix}.attn.c_attn.weight  [3*hidden_dim × hidden_dim]  (Q,K,V packed)
/// {prefix}.attn.c_attn.bias    [3*hidden_dim]
/// {prefix}.attn.c_proj.weight  [hidden_dim × hidden_dim]
/// {prefix}.attn.c_proj.bias    [hidden_dim]
/// {prefix}.ln_2.weight   [hidden_dim]
/// {prefix}.ln_2.bias     [hidden_dim]
/// {prefix}.mlp.c_fc.weight     [ffn_intermediate × hidden_dim]
/// {prefix}.mlp.c_fc.bias       [ffn_intermediate]
/// {prefix}.mlp.c_proj.weight   [hidden_dim × ffn_intermediate]
/// {prefix}.mlp.c_proj.bias     [hidden_dim]
/// ```
pub fn load_gpt2_block(
    block: &mut GptBlock,
    mw: &ModelWeights,
    prefix: &str,
    cfg: &GptConfig,
) -> LmResult<()> {
    let hd = cfg.n_embd;
    let ffd = cfg.ffn_intermediate;

    // LayerNorm 1
    block.ln_1.weight = load_vec(mw, &format!("{prefix}.ln_1.weight"), hd)?;
    block.ln_1.bias = load_vec(mw, &format!("{prefix}.ln_1.bias"), hd)?;

    // Attention: packed c_attn weight/bias, then c_proj
    let c_attn_w = load_tensor(mw, &format!("{prefix}.attn.c_attn.weight"), &[3 * hd, hd])?;
    let c_attn_b = load_vec(mw, &format!("{prefix}.attn.c_attn.bias"), 3 * hd)?;
    // Split into Q, K, V along the first dimension
    block.attn.w_q = slice_rows(&c_attn_w, 0, hd, hd)?;
    block.attn.w_k = slice_rows(&c_attn_w, hd, hd, hd)?;
    block.attn.w_v = slice_rows(&c_attn_w, 2 * hd, hd, hd)?;
    block.attn.b_q = Some(c_attn_b[..hd].to_vec());
    block.attn.b_k = Some(c_attn_b[hd..2 * hd].to_vec());
    block.attn.b_v = Some(c_attn_b[2 * hd..].to_vec());
    block.attn.w_o = load_tensor(mw, &format!("{prefix}.attn.c_proj.weight"), &[hd, hd])?;
    block.attn.b_o = Some(load_vec(mw, &format!("{prefix}.attn.c_proj.bias"), hd)?);

    // LayerNorm 2
    block.ln_2.weight = load_vec(mw, &format!("{prefix}.ln_2.weight"), hd)?;
    block.ln_2.bias = load_vec(mw, &format!("{prefix}.ln_2.bias"), hd)?;

    // MLP
    block.ffn.w_fc = load_tensor(mw, &format!("{prefix}.mlp.c_fc.weight"), &[ffd, hd])?;
    block.ffn.b_fc = load_vec(mw, &format!("{prefix}.mlp.c_fc.bias"), ffd)?;
    block.ffn.w_proj = load_tensor(mw, &format!("{prefix}.mlp.c_proj.weight"), &[hd, ffd])?;
    block.ffn.b_proj = load_vec(mw, &format!("{prefix}.mlp.c_proj.bias"), hd)?;

    Ok(())
}

// ─── LLaMA layer loading ─────────────────────────────────────────────────────

/// Load weights into a `LlamaBlock` from a `ModelWeights` dictionary.
///
/// Expected keys (with `prefix = "model.layers.{layer_idx}"`):
///
/// ```text
/// {prefix}.input_layernorm.weight              [hidden_dim]
/// {prefix}.self_attn.q_proj.weight             [hidden_dim × hidden_dim]
/// {prefix}.self_attn.k_proj.weight             [kv_proj_dim × hidden_dim]
/// {prefix}.self_attn.v_proj.weight             [kv_proj_dim × hidden_dim]
/// {prefix}.self_attn.o_proj.weight             [hidden_dim × hidden_dim]
/// {prefix}.post_attention_layernorm.weight     [hidden_dim]
/// {prefix}.mlp.gate_proj.weight                [intermediate_dim × hidden_dim]
/// {prefix}.mlp.up_proj.weight                  [intermediate_dim × hidden_dim]
/// {prefix}.mlp.down_proj.weight                [hidden_dim × intermediate_dim]
/// ```
pub fn load_llama_block(
    block: &mut LlamaBlock,
    mw: &ModelWeights,
    prefix: &str,
    cfg: &LlamaConfig,
) -> LmResult<()> {
    let hd = cfg.hidden_dim;
    let id = cfg.intermediate_dim;
    let kv = cfg.n_kv_heads * cfg.head_dim();

    // Attention norm
    block.attn_norm.weight = load_vec(mw, &format!("{prefix}.input_layernorm.weight"), hd)?;

    // Attention projections
    block.attn.w_q = load_tensor(mw, &format!("{prefix}.self_attn.q_proj.weight"), &[hd, hd])?;
    block.attn.w_k = load_tensor(mw, &format!("{prefix}.self_attn.k_proj.weight"), &[kv, hd])?;
    block.attn.w_v = load_tensor(mw, &format!("{prefix}.self_attn.v_proj.weight"), &[kv, hd])?;
    block.attn.w_o = load_tensor(mw, &format!("{prefix}.self_attn.o_proj.weight"), &[hd, hd])?;

    // FFN norm
    block.ffn_norm.weight = load_vec(mw, &format!("{prefix}.post_attention_layernorm.weight"), hd)?;

    // FFN projections
    block.ffn.w_gate = load_tensor(mw, &format!("{prefix}.mlp.gate_proj.weight"), &[id, hd])?;
    block.ffn.w_up = load_tensor(mw, &format!("{prefix}.mlp.up_proj.weight"), &[id, hd])?;
    block.ffn.w_down = load_tensor(mw, &format!("{prefix}.mlp.down_proj.weight"), &[hd, id])?;

    Ok(())
}

// ─── Low-level helpers ────────────────────────────────────────────────────────

/// Retrieve a `Vec<f32>` of exactly `expected_len` elements from the weight
/// dictionary, validating shape `[expected_len]`.
pub fn load_vec(mw: &ModelWeights, name: &str, expected_len: usize) -> LmResult<Vec<f32>> {
    let t = mw.get_checked(name, &[expected_len])?;
    Ok(t.data.clone())
}

/// Retrieve a `WeightTensor` with the given shape.
pub fn load_tensor(
    mw: &ModelWeights,
    name: &str,
    expected_shape: &[usize],
) -> LmResult<WeightTensor> {
    let t = mw.get_checked(name, expected_shape)?;
    Ok(t.clone())
}

/// Slice `rows_len` rows starting at `row_start` from a 2-D weight.
///
/// Used to split packed QKV weight matrices.
fn slice_rows(
    w: &WeightTensor,
    row_start: usize,
    rows_len: usize,
    n_cols: usize,
) -> LmResult<WeightTensor> {
    if w.shape.len() != 2 || w.shape[1] != n_cols {
        return Err(LmError::DimensionMismatch {
            expected: n_cols,
            got: if w.shape.len() >= 2 { w.shape[1] } else { 0 },
        });
    }
    let start = row_start * n_cols;
    let end = start + rows_len * n_cols;
    if end > w.data.len() {
        return Err(LmError::DimensionMismatch {
            expected: end,
            got: w.data.len(),
        });
    }
    WeightTensor::from_data(w.data[start..end].to_vec(), vec![rows_len, n_cols])
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::ModelWeights;

    fn tiny_gpt2_cfg() -> GptConfig {
        GptConfig::tiny()
    }

    fn tiny_llama_cfg() -> LlamaConfig {
        LlamaConfig::tiny()
    }

    fn make_gpt2_weights(cfg: &GptConfig) -> ModelWeights {
        let hd = cfg.n_embd;
        let ffd = cfg.ffn_intermediate;
        let pfx = "transformer.h.0";
        let mut mw = ModelWeights::new();
        mw.insert(format!("{pfx}.ln_1.weight"), WeightTensor::ones(&[hd]));
        mw.insert(format!("{pfx}.ln_1.bias"), WeightTensor::zeros(&[hd]));
        mw.insert(
            format!("{pfx}.attn.c_attn.weight"),
            WeightTensor::zeros(&[3 * hd, hd]),
        );
        mw.insert(
            format!("{pfx}.attn.c_attn.bias"),
            WeightTensor::zeros(&[3 * hd]),
        );
        mw.insert(
            format!("{pfx}.attn.c_proj.weight"),
            WeightTensor::zeros(&[hd, hd]),
        );
        mw.insert(
            format!("{pfx}.attn.c_proj.bias"),
            WeightTensor::zeros(&[hd]),
        );
        mw.insert(format!("{pfx}.ln_2.weight"), WeightTensor::ones(&[hd]));
        mw.insert(format!("{pfx}.ln_2.bias"), WeightTensor::zeros(&[hd]));
        mw.insert(
            format!("{pfx}.mlp.c_fc.weight"),
            WeightTensor::zeros(&[ffd, hd]),
        );
        mw.insert(format!("{pfx}.mlp.c_fc.bias"), WeightTensor::zeros(&[ffd]));
        mw.insert(
            format!("{pfx}.mlp.c_proj.weight"),
            WeightTensor::zeros(&[hd, ffd]),
        );
        mw.insert(format!("{pfx}.mlp.c_proj.bias"), WeightTensor::zeros(&[hd]));
        mw
    }

    fn make_llama_weights(cfg: &LlamaConfig) -> ModelWeights {
        let hd = cfg.hidden_dim;
        let id = cfg.intermediate_dim;
        let kv = cfg.n_kv_heads * cfg.head_dim();
        let pfx = "model.layers.0";
        let mut mw = ModelWeights::new();
        mw.insert(
            format!("{pfx}.input_layernorm.weight"),
            WeightTensor::ones(&[hd]),
        );
        mw.insert(
            format!("{pfx}.self_attn.q_proj.weight"),
            WeightTensor::zeros(&[hd, hd]),
        );
        mw.insert(
            format!("{pfx}.self_attn.k_proj.weight"),
            WeightTensor::zeros(&[kv, hd]),
        );
        mw.insert(
            format!("{pfx}.self_attn.v_proj.weight"),
            WeightTensor::zeros(&[kv, hd]),
        );
        mw.insert(
            format!("{pfx}.self_attn.o_proj.weight"),
            WeightTensor::zeros(&[hd, hd]),
        );
        mw.insert(
            format!("{pfx}.post_attention_layernorm.weight"),
            WeightTensor::ones(&[hd]),
        );
        mw.insert(
            format!("{pfx}.mlp.gate_proj.weight"),
            WeightTensor::zeros(&[id, hd]),
        );
        mw.insert(
            format!("{pfx}.mlp.up_proj.weight"),
            WeightTensor::zeros(&[id, hd]),
        );
        mw.insert(
            format!("{pfx}.mlp.down_proj.weight"),
            WeightTensor::zeros(&[hd, id]),
        );
        mw
    }

    #[test]
    fn load_gpt2_block_ok() {
        let cfg = tiny_gpt2_cfg();
        let mw = make_gpt2_weights(&cfg);
        let mut block = GptBlock::new(
            cfg.n_embd,
            cfg.n_heads,
            cfg.ffn_intermediate,
            cfg.layer_norm_eps,
        )
        .unwrap();
        load_gpt2_block(&mut block, &mw, "transformer.h.0", &cfg).unwrap();
        // ln_1.weight should now be ones
        assert!(block.ln_1.weight.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn load_llama_block_ok() {
        let cfg = tiny_llama_cfg();
        let mw = make_llama_weights(&cfg);
        let mut block = LlamaBlock::new(
            cfg.hidden_dim,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.intermediate_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            cfg.rms_norm_eps,
        )
        .unwrap();
        load_llama_block(&mut block, &mw, "model.layers.0", &cfg).unwrap();
        assert!(
            block
                .attn_norm
                .weight
                .iter()
                .all(|&v| (v - 1.0).abs() < 1e-6)
        );
    }

    #[test]
    fn load_gpt2_block_missing_key_errors() {
        let cfg = tiny_gpt2_cfg();
        let mw = ModelWeights::new(); // empty
        let mut block = GptBlock::new(
            cfg.n_embd,
            cfg.n_heads,
            cfg.ffn_intermediate,
            cfg.layer_norm_eps,
        )
        .unwrap();
        assert!(load_gpt2_block(&mut block, &mw, "transformer.h.0", &cfg).is_err());
    }

    #[test]
    fn slice_rows_correct() {
        // 4×4 weight split into 2×2 top and 2×2 bottom
        let w = WeightTensor::from_data((0..16).map(|x| x as f32).collect(), vec![4, 4]).unwrap();
        let top = slice_rows(&w, 0, 2, 4).unwrap();
        assert_eq!(top.shape, vec![2, 4]);
        assert_eq!(top.data[0], 0.0);
        assert_eq!(top.data[7], 7.0);
        let bot = slice_rows(&w, 2, 2, 4).unwrap();
        assert_eq!(bot.data[0], 8.0);
    }
}
