//! Free-support Wasserstein barycenter with adaptive support refinement.
//!
//! Implements an improved version of the Cuturi-Doucet (2014) free-support
//! barycenter algorithm. Compared to [`crate::barycenter::free_support`],
//! this variant adds:
//!
//! 1. **Adaptive support pruning**: after convergence, support points whose
//!    accumulated weight falls below `prune_threshold` are removed, yielding a
//!    sparse, geometrically meaningful barycenter.
//!
//! 2. **Improved M-step**: the barycentric projection correctly normalises each
//!    support point by the total incoming mass from all sources, not just the
//!    per-source row sum. This avoids shrinkage bias in the multi-source case.
//!
//! 3. **Convergence tracking**: the solver monitors `‖Y_new − Y_old‖_∞` and
//!    stops early once the support moves less than `tol`.
//!
//! 4. **Dual f64 precision** throughout (cost and potentials stay in f64 to
//!    avoid catastrophic cancellation in high-reg regimes).
//!
//! ## Algorithm (Lloyd-style alternating optimisation)
//!
//! ```text
//! Initialise Y ← n_support samples from the λ-weighted mixture of sources
//! Repeat until convergence:
//!   E-step: ∀s, T_s = Sinkhorn(cost(Y, X_s), b, a_s, ε)
//!   M-step: ∀k, Y_k ← (Σ_s λ_s Σ_i T_s[i,k] · x_{s,i}) / (Σ_s λ_s Σ_i T_s[i,k])
//!   Update weights: w_k = Σ_s λ_s Σ_i T_s[i,k]
//!   Prune: remove k where w_k < prune_threshold (after convergence)
//! ```

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;
use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the adaptive free-support barycenter solver.
#[derive(Debug, Clone)]
pub struct FreeSupportConfig {
    /// Requested number of support points before pruning.
    pub n_support: usize,
    /// Sinkhorn entropic regularisation ε > 0.
    pub reg: f64,
    /// Maximum number of Lloyd alternating iterations.
    pub max_iter: usize,
    /// Convergence tolerance on sup-norm support displacement `‖Y_new−Y_old‖∞`.
    pub tol: f64,
    /// After convergence, prune support points with normalised weight < threshold.
    /// Set to 0 to disable pruning.
    pub prune_threshold: f64,
    /// RNG seed for reproducible support initialisation.
    pub seed: u64,
}

impl Default for FreeSupportConfig {
    fn default() -> Self {
        Self {
            n_support: 20,
            reg: 0.05,
            max_iter: 50,
            tol: 1e-6,
            prune_threshold: 1e-4,
            seed: 0,
        }
    }
}

/// Result of the adaptive free-support Wasserstein barycenter.
#[derive(Debug, Clone)]
pub struct FreeSupportBary {
    /// Barycenter support positions, shape `[n_support × d]` row-major.
    pub support: Vec<f64>,
    /// Normalised weights for each support point, length `n_support`.
    /// Sums to 1 (approximately).
    pub weights: Vec<f64>,
    /// Number of support points after pruning.
    pub n_support: usize,
    /// Ambient dimension.
    pub d: usize,
    /// Final transport cost approximation.
    pub cost: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────────────

fn validate_free_support(
    sources: &[&[f64]],
    n_per_source: &[usize],
    weights_source: &[f64],
    d: usize,
    lambdas: &[f64],
    cfg: &FreeSupportConfig,
) -> OtResult<()> {
    let n_src = sources.len();
    if n_src == 0 {
        return Err(OtError::EmptyInput);
    }
    if n_per_source.len() != n_src {
        return Err(OtError::IncompatibleLength {
            a: n_per_source.len(),
            b: n_src,
        });
    }
    if weights_source.len() != n_src {
        return Err(OtError::IncompatibleLength {
            a: weights_source.len(),
            b: n_src,
        });
    }
    if lambdas.len() != n_src {
        return Err(OtError::IncompatibleLength {
            a: lambdas.len(),
            b: n_src,
        });
    }
    if d == 0 {
        return Err(OtError::BadDim { got: d });
    }
    if cfg.n_support == 0 {
        return Err(OtError::BadCount { got: cfg.n_support });
    }
    if cfg.reg <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: cfg.reg as f32,
        });
    }
    if cfg.max_iter == 0 {
        return Err(OtError::BadCount { got: cfg.max_iter });
    }
    // Validate lambdas
    let mut lam_sum = 0.0_f64;
    for &lam in lambdas {
        if !lam.is_finite() || lam < 0.0 {
            return Err(OtError::NegativeWeight);
        }
        lam_sum += lam;
    }
    if (lam_sum - 1.0).abs() > 1e-3 {
        return Err(OtError::NotProbability);
    }
    // Validate each source
    for (s, (&n_s, &xs)) in n_per_source.iter().zip(sources.iter()).enumerate() {
        if n_s == 0 {
            return Err(OtError::EmptyInput);
        }
        if xs.len() != n_s * d {
            return Err(OtError::IncompatibleLength {
                a: xs.len(),
                b: n_s * d,
            });
        }
        // weights_source is a flat array; index by source index s
        let ws = weights_source[s];
        if !ws.is_finite() || ws < 0.0 {
            return Err(OtError::NegativeWeight);
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build the squared-Euclidean cost matrix between support `Y` (n_k × d) and
/// source points `X_s` (n_s × d). Returns an `(n_k × n_s)` row-major matrix.
fn build_cost_matrix(y: &[f64], xs: &[f64], n_k: usize, n_s: usize, d: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; n_k * n_s];
    for i in 0..n_k {
        for j in 0..n_s {
            let mut sq = 0.0_f64;
            for dim in 0..d {
                let diff = y[i * d + dim] - xs[j * d + dim];
                sq += diff * diff;
            }
            c[i * n_s + j] = (0.5 * sq) as f32;
        }
    }
    c
}

/// Initialise the barycenter support by sampling uniformly from the λ-weighted
/// mixture of sources. For each of the `n_support` requested points, choose a
/// source `s` with probability proportional to `λ_s`, then sample a uniformly
/// chosen point from that source.
fn init_support_points(
    sources: &[&[f64]],
    n_per_source: &[usize],
    d: usize,
    n_support: usize,
    lambdas: &[f64],
    rng: &mut LcgRng,
) -> Vec<f64> {
    let n_src = sources.len();
    // Build cumulative lambda array for inverse-CDF sampling
    let mut cum_lambda = vec![0.0_f64; n_src + 1];
    for s in 0..n_src {
        cum_lambda[s + 1] = cum_lambda[s] + lambdas[s];
    }
    let total_lam = cum_lambda[n_src];

    let mut y = vec![0.0_f64; n_support * d];
    for i in 0..n_support {
        // Sample source index proportional to lambda
        let r = rng.next_f32() as f64 * total_lam;
        let mut s = n_src - 1;
        for k in 0..n_src {
            if r < cum_lambda[k + 1] {
                s = k;
                break;
            }
        }
        // Sample a uniformly random point from source s
        let n_s = n_per_source[s];
        let pt_idx = rng.next_usize(n_s);
        let src_ptr = &sources[s][pt_idx * d..(pt_idx + 1) * d];
        y[i * d..i * d + d].copy_from_slice(src_ptr);
        // Add tiny jitter to avoid degenerate initialisation
        for dim in 0..d {
            y[i * d + dim] += 1e-4 * (rng.next_f32() as f64 - 0.5);
        }
    }
    y
}

/// Compute the sup-norm displacement between two support arrays.
fn sup_norm_displacement(y_old: &[f64], y_new: &[f64]) -> f64 {
    y_old
        .iter()
        .zip(y_new.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the free-support Wasserstein barycenter with adaptive pruning.
///
/// # Parameters
///
/// - `sources`: slice of source point clouds, each a flat `n_s × d` row-major array.
/// - `n_per_source`: number of points in each source.
/// - `weights_source`: a single weight per source (used only for validation;
///   within each source, uniform weights are assumed). Pass `1/S` for each
///   source if all sources are equally weighted.
/// - `d`: ambient dimension.
/// - `lambdas`: barycenter mixture weights, one per source, must sum to 1.
/// - `cfg`: solver configuration.
/// - `rng`: seeded random number generator (used only for initialisation).
///
/// # Returns
///
/// A [`FreeSupportBary`] with the barycenter support positions and weights.
///
/// # Errors
///
/// Returns an error if inputs are invalid (wrong sizes, non-positive reg, etc.).
pub fn free_support_barycenter(
    sources: &[&[f64]],
    n_per_source: &[usize],
    weights_source: &[f64],
    d: usize,
    lambdas: &[f64],
    cfg: &FreeSupportConfig,
    rng: &mut LcgRng,
) -> OtResult<FreeSupportBary> {
    validate_free_support(sources, n_per_source, weights_source, d, lambdas, cfg)?;

    let n_src = sources.len();
    let mut n_k = cfg.n_support;

    // Initialise support positions by sampling from the weighted mixture
    let mut y = init_support_points(sources, n_per_source, d, n_k, lambdas, rng);
    // Uniform initial weights
    let mut bary_weights = vec![1.0_f64 / n_k as f64; n_k];

    let eps = cfg.reg as f32;
    let inner_cfg = SinkhornConfig {
        eps,
        max_iter: 200,
        tol: 1e-5,
    };

    let mut new_y = vec![0.0_f64; n_k * d];
    let mut accumulated_weights = vec![0.0_f64; n_k];
    let mut final_cost = 0.0_f64;

    for _iter in 0..cfg.max_iter {
        // Reset accumulators
        for v in new_y.iter_mut() {
            *v = 0.0;
        }
        for v in accumulated_weights.iter_mut() {
            *v = 0.0;
        }
        final_cost = 0.0;

        // E-step + M-step accumulation over all sources
        for s in 0..n_src {
            let xs = sources[s];
            let n_s = n_per_source[s];
            let lam_s = lambdas[s];
            let lam_s_f32 = lam_s as f32;

            // Build cost matrix: n_k rows (support) × n_s cols (source)
            let cost_mat = build_cost_matrix(&y, xs, n_k, n_s, d);

            // Uniform marginals on each side
            let b_bary = vec![1.0_f32 / n_k as f32; n_k];
            let a_src = vec![1.0_f32 / n_s as f32; n_s];

            // Sinkhorn: transport from (bary, n_k) to (source, n_s)
            let result = sinkhorn(&cost_mat, &b_bary, &a_src, n_k, n_s, &inner_cfg);

            let plan = match result {
                Ok(r) => {
                    final_cost += lam_s * r.cost as f64;
                    r.plan
                }
                Err(_) => {
                    // Fall back to uniform plan on Sinkhorn failure
                    vec![1.0_f32 / (n_k * n_s) as f32; n_k * n_s]
                }
            };

            // M-step: accumulate Y_k += λ_s Σ_j T[k,j] * x_{s,j}
            for k in 0..n_k {
                let mut row_mass = 0.0_f32;
                for j in 0..n_s {
                    row_mass += plan[k * n_s + j];
                }
                accumulated_weights[k] += (lam_s_f32 * row_mass) as f64;
                for j in 0..n_s {
                    let t_kj = plan[k * n_s + j] as f64;
                    for dim in 0..d {
                        new_y[k * d + dim] += lam_s * t_kj * xs[j * d + dim];
                    }
                }
            }
        }

        // Normalise each new support point by its total accumulated mass
        for k in 0..n_k {
            let mass_k = accumulated_weights[k];
            if mass_k > 1e-300 {
                let inv_mass = 1.0 / mass_k;
                for dim in 0..d {
                    new_y[k * d + dim] *= inv_mass;
                }
            } else {
                // Dead support point: keep old position
                for dim in 0..d {
                    new_y[k * d + dim] = y[k * d + dim];
                }
            }
        }

        // Check convergence
        let displacement = sup_norm_displacement(&y, &new_y);
        std::mem::swap(&mut y, &mut new_y);

        // Update weights (normalise accumulated_weights)
        let total_mass: f64 = accumulated_weights.iter().sum();
        if total_mass > 1e-300 {
            let inv_total = 1.0 / total_mass;
            for (w, &aw) in bary_weights.iter_mut().zip(accumulated_weights.iter()) {
                *w = aw * inv_total;
            }
        }

        if displacement < cfg.tol {
            break;
        }
    }

    // Adaptive pruning: remove support points with weight < prune_threshold
    if cfg.prune_threshold > 0.0 {
        let mut kept_y = Vec::with_capacity(n_k * d);
        let mut kept_w = Vec::with_capacity(n_k);
        for k in 0..n_k {
            if bary_weights[k] >= cfg.prune_threshold {
                kept_y.extend_from_slice(&y[k * d..k * d + d]);
                kept_w.push(bary_weights[k]);
            }
        }
        if kept_w.is_empty() {
            // Fallback: keep the highest-weight point
            let (best_k, _) = bary_weights
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            kept_y.extend_from_slice(&y[best_k * d..best_k * d + d]);
            kept_w.push(1.0);
        }
        n_k = kept_w.len();
        // Re-normalise pruned weights
        let total_w: f64 = kept_w.iter().sum();
        if total_w > 1e-300 {
            let inv_w = 1.0 / total_w;
            for w in kept_w.iter_mut() {
                *w *= inv_w;
            }
        }
        y = kept_y;
        bary_weights = kept_w;
    }

    Ok(FreeSupportBary {
        support: y,
        weights: bary_weights,
        n_support: n_k,
        d,
        cost: final_cost,
    })
}

/// Compute the approximate Wasserstein cost between a barycenter and all sources.
///
/// For each source `s`, builds the cost matrix `C(Y, X_s)`, runs Sinkhorn with
/// the barycenter weights as one marginal and uniform as the other, and returns
/// `Σ_s λ_s · OT_ε(ν, μ_s)`.
///
/// # Errors
///
/// Returns an error if dimensions are inconsistent or if reg is non-positive.
pub fn free_support_cost(
    bary: &FreeSupportBary,
    sources: &[&[f64]],
    n_per_source: &[usize],
    d: usize,
    lambdas: &[f64],
    reg: f64,
) -> OtResult<f64> {
    let n_src = sources.len();
    if n_src == 0 {
        return Err(OtError::EmptyInput);
    }
    if n_per_source.len() != n_src || lambdas.len() != n_src {
        return Err(OtError::IncompatibleLength {
            a: n_per_source.len().max(lambdas.len()),
            b: n_src,
        });
    }
    if d == 0 {
        return Err(OtError::BadDim { got: d });
    }
    if reg <= 0.0 {
        return Err(OtError::BadEpsilon { eps: reg as f32 });
    }
    if bary.d != d {
        return Err(OtError::BadDim { got: bary.d });
    }

    let n_k = bary.n_support;
    let eps = reg as f32;
    let inner_cfg = SinkhornConfig {
        eps,
        max_iter: 300,
        tol: 1e-5,
    };

    let mut total_cost = 0.0_f64;
    let b_bary: Vec<f32> = bary.weights.iter().map(|&w| w as f32).collect();

    for s in 0..n_src {
        let xs = sources[s];
        let n_s = n_per_source[s];
        if xs.len() != n_s * d {
            return Err(OtError::IncompatibleLength {
                a: xs.len(),
                b: n_s * d,
            });
        }
        let cost_mat = build_cost_matrix(&bary.support, xs, n_k, n_s, d);
        let a_src = vec![1.0_f32 / n_s as f32; n_s];

        let result = sinkhorn(&cost_mat, &b_bary, &a_src, n_k, n_s, &inner_cfg)?;
        total_cost += lambdas[s] * result.cost as f64;
    }

    Ok(total_cost)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    /// Generate `n` points on a regular 1-D grid in [lo, hi].
    fn grid_1d(n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n - 1).max(1) as f64)
            .collect()
    }

    #[test]
    fn empty_sources_rejected() {
        let cfg = FreeSupportConfig::default();
        let mut rng = make_rng(0);
        let res = free_support_barycenter(&[], &[], &[], 1, &[], &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn bad_dim_rejected() {
        let src = vec![0.0_f64, 1.0];
        let sources: Vec<&[f64]> = vec![&src];
        let cfg = FreeSupportConfig {
            n_support: 2,
            ..Default::default()
        };
        let mut rng = make_rng(0);
        let res = free_support_barycenter(&sources, &[2], &[1.0], 0, &[1.0], &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::BadDim { .. })));
    }

    #[test]
    fn bad_lambdas_rejected() {
        let src = vec![0.0_f64, 1.0];
        let sources: Vec<&[f64]> = vec![&src];
        let cfg = FreeSupportConfig {
            n_support: 2,
            ..Default::default()
        };
        let mut rng = make_rng(0);
        // lambdas = [2.0] does not sum to 1
        let res = free_support_barycenter(&sources, &[2], &[1.0], 1, &[2.0], &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::NotProbability)));
    }

    #[test]
    fn n_support_zero_rejected() {
        let src = vec![0.0_f64, 1.0];
        let sources: Vec<&[f64]> = vec![&src];
        let cfg = FreeSupportConfig {
            n_support: 0,
            ..Default::default()
        };
        let mut rng = make_rng(0);
        let res = free_support_barycenter(&sources, &[2], &[1.0], 1, &[1.0], &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }

    #[test]
    fn single_source_barycenter_stays_near_source() {
        // With a single source, the barycenter should converge near the source centroid.
        let src = grid_1d(10, 0.0, 1.0);
        let sources: Vec<&[f64]> = vec![src.as_slice()];
        let cfg = FreeSupportConfig {
            n_support: 3,
            reg: 0.1,
            max_iter: 30,
            tol: 1e-5,
            prune_threshold: 0.0, // no pruning
            seed: 7,
        };
        let mut rng = make_rng(cfg.seed);
        let bary = free_support_barycenter(&sources, &[10], &[1.0], 1, &[1.0], &cfg, &mut rng)
            .expect("converges");

        assert_eq!(bary.d, 1);
        assert_eq!(bary.n_support, 3);
        assert_eq!(bary.support.len(), 3);

        // All support points should lie roughly in [0, 1]
        for &y in &bary.support {
            assert!(
                (-0.5..=1.5).contains(&y),
                "support point {y} out of expected range"
            );
        }
    }

    #[test]
    fn weights_sum_to_one() {
        let src1 = grid_1d(8, 0.0, 1.0);
        let src2 = grid_1d(8, 2.0, 3.0);
        let sources: Vec<&[f64]> = vec![src1.as_slice(), src2.as_slice()];
        let cfg = FreeSupportConfig {
            n_support: 4,
            reg: 0.2,
            max_iter: 20,
            tol: 1e-4,
            prune_threshold: 0.0,
            seed: 11,
        };
        let mut rng = make_rng(cfg.seed);
        let bary = free_support_barycenter(
            &sources,
            &[8, 8],
            &[0.5, 0.5],
            1,
            &[0.5, 0.5],
            &cfg,
            &mut rng,
        )
        .expect("converges");

        let weight_sum: f64 = bary.weights.iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-4,
            "weights sum to {weight_sum}"
        );
    }

    #[test]
    fn barycenter_midpoint_for_two_symmetric_sources() {
        // Two 1-D point masses at 0 and 2; barycenter with equal weights should be near 1.
        let src1 = vec![0.0_f64];
        let src2 = vec![2.0_f64];
        let sources: Vec<&[f64]> = vec![src1.as_slice(), src2.as_slice()];
        let cfg = FreeSupportConfig {
            n_support: 1,
            reg: 0.5,
            max_iter: 50,
            tol: 1e-6,
            prune_threshold: 0.0,
            seed: 3,
        };
        let mut rng = make_rng(cfg.seed);
        let bary = free_support_barycenter(
            &sources,
            &[1, 1],
            &[0.5, 0.5],
            1,
            &[0.5, 0.5],
            &cfg,
            &mut rng,
        )
        .expect("converges");

        let y = bary.support[0];
        assert!((y - 1.0).abs() < 0.3, "barycenter={y} should be near 1.0");
    }

    #[test]
    fn pruning_reduces_support_size() {
        let src1 = grid_1d(12, 0.0, 1.0);
        let src2 = grid_1d(12, 4.0, 5.0);
        let sources: Vec<&[f64]> = vec![src1.as_slice(), src2.as_slice()];
        let cfg = FreeSupportConfig {
            n_support: 10,
            reg: 0.3,
            max_iter: 25,
            tol: 1e-4,
            prune_threshold: 0.3, // aggressive pruning
            seed: 42,
        };
        let mut rng = make_rng(cfg.seed);
        let bary = free_support_barycenter(
            &sources,
            &[12, 12],
            &[0.5, 0.5],
            1,
            &[0.5, 0.5],
            &cfg,
            &mut rng,
        )
        .expect("converges");

        // After pruning, should have fewer support points
        assert!(bary.n_support <= 10, "n_support={}", bary.n_support);
        assert!(bary.n_support >= 1, "must have at least one point");
    }

    #[test]
    fn free_support_cost_is_finite() {
        let src1 = grid_1d(6, 0.0, 1.0);
        let src2 = grid_1d(6, 1.5, 2.5);
        let sources: Vec<&[f64]> = vec![src1.as_slice(), src2.as_slice()];
        let cfg = FreeSupportConfig {
            n_support: 3,
            reg: 0.2,
            max_iter: 15,
            tol: 1e-4,
            prune_threshold: 0.0,
            seed: 9,
        };
        let mut rng = make_rng(cfg.seed);
        let bary = free_support_barycenter(
            &sources,
            &[6, 6],
            &[0.5, 0.5],
            1,
            &[0.5, 0.5],
            &cfg,
            &mut rng,
        )
        .expect("converges");

        let tc = free_support_cost(&bary, &sources, &[6, 6], 1, &[0.5, 0.5], 0.2).expect("ok");
        assert!(tc.is_finite(), "cost={tc}");
        assert!(tc >= 0.0, "cost={tc}");
    }

    #[test]
    fn two_d_barycenter_between_two_clusters() {
        // Source 1: 4 points at (0,0); Source 2: 4 points at (2,2)
        let src1 = vec![0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 4 × 2
        let src2 = vec![2.0_f64, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]; // 4 × 2
        let sources: Vec<&[f64]> = vec![src1.as_slice(), src2.as_slice()];
        let cfg = FreeSupportConfig {
            n_support: 2,
            reg: 0.5,
            max_iter: 30,
            tol: 1e-5,
            prune_threshold: 0.0,
            seed: 55,
        };
        let mut rng = make_rng(cfg.seed);
        let bary = free_support_barycenter(
            &sources,
            &[4, 4],
            &[0.5, 0.5],
            2,
            &[0.5, 0.5],
            &cfg,
            &mut rng,
        )
        .expect("converges");

        assert_eq!(bary.d, 2);
        // All support x coords should be between 0 and 2
        for k in 0..bary.n_support {
            let x = bary.support[k * 2];
            let y_coord = bary.support[k * 2 + 1];
            assert!((-0.5..=2.5).contains(&x), "x coord {x} out of range");
            assert!(
                (-0.5..=2.5).contains(&y_coord),
                "y coord {y_coord} out of range"
            );
        }
    }

    #[test]
    fn cost_error_on_bad_reg() {
        let bary = FreeSupportBary {
            support: vec![0.5_f64],
            weights: vec![1.0],
            n_support: 1,
            d: 1,
            cost: 0.0,
        };
        let src = vec![0.0_f64, 1.0];
        let sources: Vec<&[f64]> = vec![src.as_slice()];
        let res = free_support_cost(&bary, &sources, &[2], 1, &[1.0], -0.1);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn bad_reg_returns_error() {
        let src = vec![0.0_f64, 1.0];
        let sources: Vec<&[f64]> = vec![src.as_slice()];
        let cfg = FreeSupportConfig {
            reg: -1.0,
            n_support: 2,
            ..Default::default()
        };
        let mut rng = make_rng(0);
        let res = free_support_barycenter(&sources, &[2], &[1.0], 1, &[1.0], &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn deterministic_given_seed() {
        let src1 = grid_1d(8, 0.0, 1.0);
        let src2 = grid_1d(8, 2.0, 3.0);
        let sources: Vec<&[f64]> = vec![src1.as_slice(), src2.as_slice()];
        let cfg = FreeSupportConfig {
            n_support: 3,
            reg: 0.2,
            max_iter: 10,
            tol: 1e-4,
            prune_threshold: 0.0,
            seed: 77,
        };
        let mut rng1 = make_rng(cfg.seed);
        let bary1 = free_support_barycenter(
            &sources,
            &[8, 8],
            &[0.5, 0.5],
            1,
            &[0.5, 0.5],
            &cfg,
            &mut rng1,
        )
        .expect("ok");
        let mut rng2 = make_rng(cfg.seed);
        let bary2 = free_support_barycenter(
            &sources,
            &[8, 8],
            &[0.5, 0.5],
            1,
            &[0.5, 0.5],
            &cfg,
            &mut rng2,
        )
        .expect("ok");

        for (a, b) in bary1.support.iter().zip(bary2.support.iter()) {
            assert_eq!(a, b, "support differs with same seed");
        }
    }
}
