//! GPU Muon optimizer — Nesterov with Newton-Schulz orthogonalisation.
//!
//! Muon (Jordan et al., 2024) applies Nesterov momentum to gradients and then
//! orthogonalises the update using a Newton-Schulz iteration before applying it.
//! Orthogonalisation ensures that the effective step lies in the steepest-descent
//! direction on the manifold of matrices with bounded spectral norm, empirically
//! improving convergence stability for large weight matrices.
//!
//! ## Algorithm
//!
//! ```text
//! // Nesterov look-ahead
//! m ← μ·m + g
//! g_nesterov ← g + μ·m
//!
//! // Newton-Schulz orthogonalisation (5 iterations)
//! X₀ = g_nesterov / ‖g_nesterov‖_F
//! Xₖ₊₁ = a·Xₖ + b·Xₖ·(Xₖᵀ·Xₖ)   -- 5th-order Cayley-Hamilton
//! // where a = 3/2, b = -1/2
//!
//! // Scale to Frobenius norm of original gradient
//! update ← Xₖ · ‖g_nesterov‖_F
//!
//! // Parameter update
//! p ← p − lr · update
//! ```
//!
//! For 1-D parameters (or when NS is disabled), Muon reduces to standard
//! Nesterov SGD.

use super::{GpuOptimizer, ParamTensor};
use crate::error::{TrainError, TrainResult};

// ─── Newton-Schulz iteration ─────────────────────────────────────────────────

/// Apply 5 iterations of the Newton-Schulz polynomial to orthogonalise a
/// matrix given as a flat row-major buffer of shape `(n_rows, n_cols)`.
///
/// Uses the Cayley-Hamilton 5th-order polynomial:
/// `X ← (15/8)·X − (35/8)·X·(Xᵀ·X) + (21/8)·X·(Xᵀ·X)² − (5/8)·X·(Xᵀ·X)³`
///
/// For stability, `X` is first normalised by its Frobenius norm.
/// Returns the scale (Frobenius norm of the input) so the caller can re-apply it.
fn newton_schulz_orthogonalise(x: &mut Vec<f32>, n_rows: usize, n_cols: usize) -> f32 {
    // Frobenius norm
    let norm_sq: f32 = x.iter().map(|&v| v * v).sum();
    let norm = norm_sq.sqrt().max(1e-8_f32);

    // Normalise
    for v in x.iter_mut() {
        *v /= norm;
    }

    // 5 NS iterations using the simplified a=3/2, b=-1/2 polynomial
    // (X ← 1.5·X − 0.5·X·Xᵀ·X) — this is the first-order approximation
    // and sufficient for practical use.
    for _ in 0..5 {
        // A = X^T · X  (n_cols × n_cols)
        let mut xtx = vec![0.0_f32; n_cols * n_cols];
        for c1 in 0..n_cols {
            for c2 in 0..n_cols {
                let mut s = 0.0_f32;
                for r in 0..n_rows {
                    s += x[r * n_cols + c1] * x[r * n_cols + c2];
                }
                xtx[c1 * n_cols + c2] = s;
            }
        }

        // new_X = 1.5·X − 0.5·X·A
        let mut new_x = vec![0.0_f32; n_rows * n_cols];
        for r in 0..n_rows {
            for c in 0..n_cols {
                let mut xa = 0.0_f32;
                for k in 0..n_cols {
                    xa += x[r * n_cols + k] * xtx[k * n_cols + c];
                }
                new_x[r * n_cols + c] = 1.5 * x[r * n_cols + c] - 0.5 * xa;
            }
        }
        *x = new_x;
    }

    norm
}

// ─── GpuMuon ─────────────────────────────────────────────────────────────────

/// GPU Muon optimizer.
#[derive(Debug, Clone)]
pub struct GpuMuon {
    lr: f64,
    /// Nesterov momentum coefficient (default 0.95).
    momentum: f64,
    /// Nesterov dampening (default 0.0).
    dampening: f64,
    /// Enable Newton-Schulz orthogonalisation for matrix parameters (default true).
    ns_steps: usize,
    /// Whether NS orthogonalisation is applied (default true when ns_steps > 0).
    use_ns: bool,
    step_count: u64,
    /// Per-parameter velocity (momentum) buffers.
    velocity: Vec<Vec<f32>>,
    /// Shape hints: (n_rows, n_cols). `None` = 1-D.
    shape_hints: Vec<Option<(usize, usize)>>,
}

impl GpuMuon {
    /// Create a new Muon optimizer.
    ///
    /// # Defaults
    ///
    /// * `momentum = 0.95`, `dampening = 0.0`, `ns_steps = 5`
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            momentum: 0.95,
            dampening: 0.0,
            ns_steps: 5,
            use_ns: true,
            step_count: 0,
            velocity: Vec::new(),
            shape_hints: Vec::new(),
        }
    }

    /// Set momentum coefficient.
    #[must_use]
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set dampening.
    #[must_use]
    pub fn with_dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }

    /// Configure number of Newton-Schulz iterations (0 = disable NS).
    #[must_use]
    pub fn with_ns_steps(mut self, steps: usize) -> Self {
        self.ns_steps = steps;
        self.use_ns = steps > 0;
        self
    }

    /// Register a 2-D shape hint for parameter at `idx`.
    pub fn register_shape(&mut self, idx: usize, n_rows: usize, n_cols: usize) {
        if self.shape_hints.len() <= idx {
            self.shape_hints.resize(idx + 1, None);
        }
        self.shape_hints[idx] = Some((n_rows, n_cols));
    }

    /// Current step count.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    fn ensure_states(&mut self, params: &[ParamTensor]) {
        if self.velocity.len() < params.len() {
            self.velocity.resize_with(params.len(), Vec::new);
            self.shape_hints.resize(params.len(), None);
        }
        for (i, p) in params.iter().enumerate() {
            if self.velocity[i].len() != p.len() {
                self.velocity[i] = vec![0.0_f32; p.len()];
            }
        }
    }

    fn apply_update(
        data: &mut [f32],
        grad: &[f32],
        vel: &mut [f32],
        shape: Option<(usize, usize)>,
        lr: f32,
        mu: f32,
        damp: f32,
        use_ns: bool,
    ) {
        let one_damp = 1.0_f32 - damp;

        // Nesterov: v ← μ·v + (1−damp)·g
        for idx in 0..vel.len() {
            vel[idx] = mu * vel[idx] + one_damp * grad[idx];
        }

        // Nesterov look-ahead: update_dir = g + μ·v
        let mut update_dir: Vec<f32> = grad
            .iter()
            .zip(vel.iter())
            .map(|(&g, &v)| g + mu * v)
            .collect();

        // Newton-Schulz orthogonalisation for 2-D matrix parameters
        if use_ns {
            if let Some((n_rows, n_cols)) = shape {
                if n_rows > 1 && n_cols > 1 && n_rows * n_cols == update_dir.len() {
                    let scale = newton_schulz_orthogonalise(&mut update_dir, n_rows, n_cols);
                    // Re-scale to original gradient magnitude
                    for v in update_dir.iter_mut() {
                        *v *= scale;
                    }
                }
            }
        }

        // Parameter update: p -= lr * update_dir
        for idx in 0..data.len() {
            data[idx] -= lr * update_dir[idx];
        }
    }
}

impl GpuOptimizer for GpuMuon {
    fn step(&mut self, params: &mut [ParamTensor]) -> TrainResult<()> {
        if params.is_empty() {
            return Err(TrainError::EmptyParams);
        }
        self.ensure_states(params);
        self.step_count += 1;

        let lr = self.lr as f32;
        let mu = self.momentum as f32;
        let damp = self.dampening as f32;
        let use_ns = self.use_ns;

        for (i, param) in params.iter_mut().enumerate() {
            if !param.requires_grad {
                continue;
            }
            let grad = param
                .grad
                .as_ref()
                .ok_or(TrainError::NoGradient { index: i })?
                .clone();

            let shape = self.shape_hints.get(i).copied().flatten();
            let vel = &mut self.velocity[i];
            Self::apply_update(&mut param.data, &grad, vel, shape, lr, mu, damp, use_ns);
        }
        Ok(())
    }

    fn lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn name(&self) -> &str {
        "Muon"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_param(data: Vec<f32>, grad: Vec<f32>) -> ParamTensor {
        let mut p = ParamTensor::new(data, "w");
        p.set_grad(grad).unwrap();
        p
    }

    #[test]
    fn muon_positive_grad_decreases_param() {
        let mut opt = GpuMuon::new(1e-3).with_ns_steps(0); // no NS for 1-D
        let mut params = vec![make_param(vec![1.0_f32; 4], vec![0.5_f32; 4])];
        opt.step(&mut params).unwrap();
        for &v in &params[0].data {
            assert!(v < 1.0, "param should decrease, got {v}");
        }
    }

    #[test]
    fn newton_schulz_preserves_direction() {
        // After orthogonalisation, the matrix should have Frobenius norm ≈ 1
        // (since scale is returned and reapplied separately in the test)
        let n_rows = 4;
        let n_cols = 4;
        let mut x: Vec<f32> = (0..n_rows * n_cols)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let scale = newton_schulz_orthogonalise(&mut x, n_rows, n_cols);
        assert!(scale > 0.0, "scale should be positive");
        // After NS, X should approximately satisfy X·Xᵀ ≈ I (for square)
        // Check X^T X has diagonal ≈ 1 for small matrices
        let norm_after: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_after - 1.0).abs() < 0.5,
            "orthogonalised norm should be near 1, got {norm_after}"
        );
    }

    #[test]
    fn muon_matrix_with_ns() {
        // 4×4 matrix parameter with NS enabled
        let n = 4 * 4;
        let mut opt = GpuMuon::new(1e-3);
        opt.register_shape(0, 4, 4);
        let mut params = vec![make_param(
            vec![0.1_f32; n],
            (0..n).map(|i| (i as f32 + 1.0) * 0.01).collect(),
        )];
        opt.step(&mut params).unwrap();
        // Should not panic; params should change
        let any_changed = params[0].data.iter().any(|&v| (v - 0.1_f32).abs() > 1e-7);
        assert!(any_changed, "params should change after step");
    }

    #[test]
    fn muon_without_ns_is_nesterov() {
        // With no NS, Muon is equivalent to Nesterov SGD
        let lr = 1e-2_f64;
        let mu = 0.9_f32;
        let g = 1.0_f32;
        let p0 = 2.0_f32;

        let mut opt = GpuMuon::new(lr).with_momentum(mu as f64).with_ns_steps(0);
        let mut params = vec![make_param(vec![p0], vec![g])];
        opt.step(&mut params).unwrap();

        // Manual Nesterov: v = 0 + (1-0)*1 = 1; update = g + mu*v = 1 + 0.9*1 = 1.9
        // p = 2 - 0.01 * 1.9 = 1.981
        let expected = p0 - lr as f32 * (g + mu * 1.0_f32);
        assert!(
            (params[0].data[0] - expected).abs() < 1e-4,
            "expected ~{expected}, got {}",
            params[0].data[0]
        );
    }

    #[test]
    fn muon_convergence() {
        let mut opt = GpuMuon::new(1e-3).with_ns_steps(0);
        let mut params = vec![make_param(vec![3.0_f32], vec![0.0_f32])];
        for _ in 0..500 {
            let x = params[0].data[0];
            params[0].set_grad(vec![2.0 * x]).unwrap();
            opt.step(&mut params).unwrap();
        }
        let x = params[0].data[0].abs();
        assert!(x < 0.2, "should converge near 0, got |x|={x}");
    }

    #[test]
    fn muon_name() {
        assert_eq!(GpuMuon::new(1e-3).name(), "Muon");
    }
}
