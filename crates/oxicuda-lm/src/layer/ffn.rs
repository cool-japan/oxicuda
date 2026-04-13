//! Feed-forward network (FFN) variants.
//!
//! | Variant | Used by | Formula |
//! |---------|---------|---------|
//! | [`MlpFfn`]    | GPT-2   | `out = Linear(GELU(Linear(x)))` |
//! | [`SwiGluFfn`] | LLaMA   | `out = Linear(SiLU(gate) ⊙ up)` |

use crate::error::{LmError, LmResult};
use crate::weights::WeightTensor;

// ─── Activation functions ─────────────────────────────────────────────────────

/// GELU activation (GPT-2 tanh approximation).
///
/// `gelu(x) ≈ 0.5x (1 + tanh(√(2/π) (x + 0.044715 x³)))`
#[inline]
pub fn gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// SiLU / Swish activation: `x * sigmoid(x) = x / (1 + exp(-x))`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ─── Linear helper ────────────────────────────────────────────────────────────

/// Compute `y = W @ x + b` for a single token.
///
/// `W` is `[out_dim × in_dim]` row-major; `x` is `[in_dim]`; `b` (if present) is `[out_dim]`.
fn linear_vec(w: &WeightTensor, b: Option<&[f32]>, x: &[f32]) -> LmResult<Vec<f32>> {
    if w.shape.len() != 2 {
        return Err(LmError::DimensionMismatch {
            expected: 2,
            got: w.shape.len(),
        });
    }
    let out_dim = w.shape[0];
    let in_dim = w.shape[1];
    if x.len() != in_dim {
        return Err(LmError::DimensionMismatch {
            expected: in_dim,
            got: x.len(),
        });
    }
    let mut out = vec![0.0_f32; out_dim];
    for (i, o) in out.iter_mut().enumerate() {
        let row = &w.data[i * in_dim..(i + 1) * in_dim];
        *o = row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
    }
    if let Some(bias) = b {
        if bias.len() != out_dim {
            return Err(LmError::DimensionMismatch {
                expected: out_dim,
                got: bias.len(),
            });
        }
        for (o, &bi) in out.iter_mut().zip(bias.iter()) {
            *o += bi;
        }
    }
    Ok(out)
}

/// Compute `Y = X @ W^T + b` for a batch of tokens.
///
/// `W` is `[out_dim × in_dim]` row-major; `X` is `[n_tokens × in_dim]` flat;
/// `b` (if present) is `[out_dim]`.  Returns `[n_tokens × out_dim]` flat.
pub(crate) fn linear_batch(
    w: &WeightTensor,
    b: Option<&[f32]>,
    x: &[f32],
    n_tokens: usize,
) -> LmResult<Vec<f32>> {
    // Single-token fast path: delegate to the simpler scalar kernel.
    if n_tokens == 1 {
        return linear_vec(w, b, x);
    }
    if w.shape.len() != 2 {
        return Err(LmError::DimensionMismatch {
            expected: 2,
            got: w.shape.len(),
        });
    }
    let out_dim = w.shape[0];
    let in_dim = w.shape[1];
    if x.len() != n_tokens * in_dim {
        return Err(LmError::DimensionMismatch {
            expected: n_tokens * in_dim,
            got: x.len(),
        });
    }
    let mut out = vec![0.0_f32; n_tokens * out_dim];
    for t in 0..n_tokens {
        let x_row = &x[t * in_dim..(t + 1) * in_dim];
        let o_row = &mut out[t * out_dim..(t + 1) * out_dim];
        for (i, o) in o_row.iter_mut().enumerate() {
            let w_row = &w.data[i * in_dim..(i + 1) * in_dim];
            *o = w_row
                .iter()
                .zip(x_row.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum();
        }
        if let Some(bias) = b {
            for (o, &bi) in o_row.iter_mut().zip(bias.iter()) {
                *o += bi;
            }
        }
    }
    Ok(out)
}

// ─── MlpFfn ──────────────────────────────────────────────────────────────────

/// Standard MLP feed-forward network (GPT-2 style).
///
/// ```text
/// h = GELU(x @ W_fc^T + b_fc)
/// out = h @ W_proj^T + b_proj
/// ```
///
/// Weight shapes:
/// - `w_fc`:   `[intermediate_dim × hidden_dim]`
/// - `b_fc`:   `[intermediate_dim]`
/// - `w_proj`: `[hidden_dim × intermediate_dim]`
/// - `b_proj`: `[hidden_dim]`
#[derive(Debug, Clone)]
pub struct MlpFfn {
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Intermediate (inner) dimension (typically 4 × hidden_dim).
    pub intermediate_dim: usize,
    /// Up-projection weight: `[intermediate_dim × hidden_dim]`.
    pub w_fc: WeightTensor,
    /// Up-projection bias: `[intermediate_dim]`.
    pub b_fc: Vec<f32>,
    /// Down-projection weight: `[hidden_dim × intermediate_dim]`.
    pub w_proj: WeightTensor,
    /// Down-projection bias: `[hidden_dim]`.
    pub b_proj: Vec<f32>,
}

impl MlpFfn {
    /// Construct with zero-initialised weights and biases.
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> LmResult<Self> {
        if hidden_dim == 0 || intermediate_dim == 0 {
            return Err(LmError::InvalidConfig {
                msg: "MlpFfn: dimensions must be > 0".into(),
            });
        }
        Ok(Self {
            hidden_dim,
            intermediate_dim,
            w_fc: WeightTensor::zeros(&[intermediate_dim, hidden_dim]),
            b_fc: vec![0.0; intermediate_dim],
            w_proj: WeightTensor::zeros(&[hidden_dim, intermediate_dim]),
            b_proj: vec![0.0; hidden_dim],
        })
    }

    /// Forward pass.
    ///
    /// `x` is `[n_tokens × hidden_dim]`; returns `[n_tokens × hidden_dim]`.
    pub fn forward(&self, x: &[f32], n_tokens: usize) -> LmResult<Vec<f32>> {
        if n_tokens == 0 {
            return Err(LmError::EmptyInput {
                context: "MlpFfn::forward n_tokens",
            });
        }
        // h = gelu(W_fc @ x + b_fc)
        let mut h = linear_batch(&self.w_fc, Some(&self.b_fc), x, n_tokens)?;
        for v in &mut h {
            *v = gelu(*v);
        }
        // out = W_proj @ h + b_proj
        linear_batch(&self.w_proj, Some(&self.b_proj), &h, n_tokens)
    }
}

// ─── SwiGluFfn ───────────────────────────────────────────────────────────────

/// SwiGLU feed-forward network (LLaMA style).
///
/// ```text
/// gate = x @ W_gate^T
/// up   = x @ W_up^T
/// h    = SiLU(gate) ⊙ up
/// out  = h @ W_down^T
/// ```
///
/// Weight shapes:
/// - `w_gate`: `[intermediate_dim × hidden_dim]`
/// - `w_up`:   `[intermediate_dim × hidden_dim]`
/// - `w_down`: `[hidden_dim × intermediate_dim]`
///
/// There are no bias terms (LLaMA does not use FFN biases).
#[derive(Debug, Clone)]
pub struct SwiGluFfn {
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Intermediate dimension.
    pub intermediate_dim: usize,
    /// Gate projection weight: `[intermediate_dim × hidden_dim]`.
    pub w_gate: WeightTensor,
    /// Up projection weight: `[intermediate_dim × hidden_dim]`.
    pub w_up: WeightTensor,
    /// Down projection weight: `[hidden_dim × intermediate_dim]`.
    pub w_down: WeightTensor,
}

impl SwiGluFfn {
    /// Construct with zero-initialised weights.
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> LmResult<Self> {
        if hidden_dim == 0 || intermediate_dim == 0 {
            return Err(LmError::InvalidConfig {
                msg: "SwiGluFfn: dimensions must be > 0".into(),
            });
        }
        Ok(Self {
            hidden_dim,
            intermediate_dim,
            w_gate: WeightTensor::zeros(&[intermediate_dim, hidden_dim]),
            w_up: WeightTensor::zeros(&[intermediate_dim, hidden_dim]),
            w_down: WeightTensor::zeros(&[hidden_dim, intermediate_dim]),
        })
    }

    /// Forward pass.
    ///
    /// `x` is `[n_tokens × hidden_dim]`; returns `[n_tokens × hidden_dim]`.
    pub fn forward(&self, x: &[f32], n_tokens: usize) -> LmResult<Vec<f32>> {
        if n_tokens == 0 {
            return Err(LmError::EmptyInput {
                context: "SwiGluFfn::forward n_tokens",
            });
        }
        let gate = linear_batch(&self.w_gate, None, x, n_tokens)?;
        let up = linear_batch(&self.w_up, None, x, n_tokens)?;
        // h = silu(gate) * up
        let mut h = vec![0.0_f32; n_tokens * self.intermediate_dim];
        for (i, ((&g, &u), h_out)) in gate.iter().zip(up.iter()).zip(h.iter_mut()).enumerate() {
            let _ = i;
            *h_out = silu(g) * u;
        }
        // out = W_down @ h
        linear_batch(&self.w_down, None, &h, n_tokens)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Activations ───────────────────────────────────────────────────────

    #[test]
    fn gelu_at_zero() {
        assert!((gelu(0.0)).abs() < 1e-6);
    }

    #[test]
    fn gelu_positive_input() {
        // gelu(x) ≈ x for large positive x
        assert!((gelu(10.0) - 10.0).abs() < 1e-3);
    }

    #[test]
    fn gelu_negative_input() {
        // gelu(x) ≈ 0 for large negative x
        assert!(gelu(-10.0).abs() < 1e-3);
    }

    #[test]
    fn silu_at_zero() {
        assert!(silu(0.0).abs() < 1e-6);
    }

    #[test]
    fn silu_positive() {
        // silu(x) ≈ x for large positive x
        assert!((silu(10.0) - 10.0).abs() < 1e-3);
    }

    #[test]
    fn silu_gradient_through_zero() {
        // silu is always >= 0 for x > 0 and < 0 for x < 0
        assert!(silu(1.0) > 0.0);
        assert!(silu(-1.0) < 0.0);
    }

    // ── Linear helper ─────────────────────────────────────────────────────

    #[test]
    fn linear_vec_identity() {
        // W = I_2, x = [3,4] → out = [3,4]
        let w = WeightTensor::eye(2, 2);
        let out = linear_vec(&w, None, &[3.0, 4.0]).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn linear_vec_with_bias() {
        let w = WeightTensor::eye(2, 2);
        let bias = vec![10.0_f32, 20.0];
        let out = linear_vec(&w, Some(&bias), &[1.0, 2.0]).unwrap();
        assert!((out[0] - 11.0).abs() < 1e-6);
        assert!((out[1] - 22.0).abs() < 1e-6);
    }

    // ── MlpFfn ────────────────────────────────────────────────────────────

    #[test]
    fn mlp_ffn_zero_weights_zero_output() {
        let ffn = MlpFfn::new(4, 8).unwrap();
        let x = vec![1.0_f32; 4];
        let out = ffn.forward(&x, 1).unwrap();
        // W=0, b=0 → h=gelu(0)=0 → out=0
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn mlp_ffn_identity_chain() {
        // w_fc = I (4×4 intermediate=4), w_proj = I, biases = 0
        // → gelu applied, then back through identity
        let mut ffn = MlpFfn::new(4, 4).unwrap();
        ffn.w_fc = WeightTensor::eye(4, 4);
        ffn.w_proj = WeightTensor::eye(4, 4);
        let x = vec![1.0_f32; 4];
        let out = ffn.forward(&x, 1).unwrap();
        // gelu(1.0) ≈ 0.841
        let expected = gelu(1.0);
        for &v in &out {
            assert!((v - expected).abs() < 1e-5, "v={v} expected={expected}");
        }
    }

    #[test]
    fn mlp_ffn_batch_tokens() {
        let ffn = MlpFfn::new(4, 8).unwrap();
        let x = vec![0.0_f32; 2 * 4]; // 2 tokens
        let out = ffn.forward(&x, 2).unwrap();
        assert_eq!(out.len(), 2 * 4);
    }

    #[test]
    fn mlp_ffn_zero_dim_error() {
        assert!(MlpFfn::new(0, 8).is_err());
    }

    // ── SwiGluFfn ─────────────────────────────────────────────────────────

    #[test]
    fn swiglu_ffn_zero_weights_zero_output() {
        let ffn = SwiGluFfn::new(4, 8).unwrap();
        let x = vec![1.0_f32; 4];
        let out = ffn.forward(&x, 1).unwrap();
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn swiglu_ffn_gate_identity() {
        // w_gate = I (4×4 intermediate=4), w_up = I, w_down = I
        // gate = x, up = x → h = silu(x)*x → out = w_down @ h = h
        let mut ffn = SwiGluFfn::new(4, 4).unwrap();
        ffn.w_gate = WeightTensor::eye(4, 4);
        ffn.w_up = WeightTensor::eye(4, 4);
        ffn.w_down = WeightTensor::eye(4, 4);
        let x = vec![2.0_f32; 4];
        let out = ffn.forward(&x, 1).unwrap();
        let expected = silu(2.0) * 2.0;
        for &v in &out {
            assert!((v - expected).abs() < 1e-5, "v={v} expected={expected}");
        }
    }

    #[test]
    fn swiglu_ffn_batch_tokens() {
        let ffn = SwiGluFfn::new(4, 8).unwrap();
        let x = vec![0.0_f32; 3 * 4];
        let out = ffn.forward(&x, 3).unwrap();
        assert_eq!(out.len(), 3 * 4);
    }

    #[test]
    fn swiglu_ffn_zero_dim_error() {
        assert!(SwiGluFfn::new(4, 0).is_err());
    }
}
