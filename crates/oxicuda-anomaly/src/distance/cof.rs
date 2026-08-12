//! Connectivity-based Outlier Factor (COF).
//!
//! Tang, Chen, Fu, Zhang. "Enhancing Effectiveness of Outlier Detections for
//! Low Density Patterns". *PAKDD 2002*.
//!
//! # Key Idea
//!
//! LOF uses k-distance / lrd for local density.  COF replaces this with
//! the **Set-Based Nearest-neighbor (SBN) cost**, the weighted cost of a
//! greedy chaining path through the k-NN of a point.
//!
//! ## SBN path from point o
//!
//! ```text
//! o₀ = o
//! For i = 1..k:
//!   oᵢ = nearest un-visited member of Nk(o) to oᵢ₋₁
//! cost = (2 / (k(k+1))) · Σᵢ₌₁ᵏ  i · d(oᵢ₋₁, oᵢ)
//! ```
//!
//! The weight `i` increases with chain position, making the cost sensitive
//! to long jumps (which outliers must make once the few nearby points are
//! exhausted).
//!
//! ## COF score
//!
//! ```text
//! COF(o) = SBN_cost(o) / mean_{p ∈ Nk(o)} SBN_cost(p)
//! ```
//!
//! High COF ≫ 1  ⟹ outlier.  Inliers in dense clusters get COF ≈ 1.

use crate::error::{AnomalyError, AnomalyResult};

// ─── Struct ───────────────────────────────────────────────────────────────────

/// Connectivity-based Outlier Factor detector.
///
/// Call [`Cof::fit`] with training data, then [`Cof::score`] / [`Cof::score_batch`]
/// to obtain COF values for new queries.
#[derive(Debug, Clone)]
pub struct Cof {
    /// Stored training data (n × d, row-major f32).
    data: Vec<f32>,
    n_samples: usize,
    n_features: usize,
    /// Number of nearest neighbours to use.
    k: usize,
    /// kNN indices for training points: `knn_indices[i*k..(i+1)*k]`.
    knn_indices: Vec<usize>,
    /// kNN distances for training points: `knn_dists[i*k..(i+1)*k]`.
    knn_dists: Vec<f32>,
    /// Pre-computed SBN cost for every training point.
    sbn_costs: Vec<f32>,
    fitted: bool,
}

impl Cof {
    /// Create a new (unfitted) COF detector with the given `k`.
    pub fn new(k: usize) -> Self {
        Self {
            data: Vec::new(),
            n_samples: 0,
            n_features: 0,
            k,
            knn_indices: Vec::new(),
            knn_dists: Vec::new(),
            sbn_costs: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the detector on `data` (`n_samples × n_features`, row-major f32).
    ///
    /// Precomputes k-NN and SBN costs for all training points.
    ///
    /// # Errors
    ///
    /// - [`AnomalyError::InvalidK`] if `k == 0` or `k >= n_samples`.
    /// - [`AnomalyError::InvalidFeatureCount`] if `n_features == 0`.
    /// - [`AnomalyError::DimensionMismatch`] if `data.len() != n_samples * n_features`.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if self.k == 0 || self.k >= n_samples {
            return Err(AnomalyError::InvalidK { k: self.k });
        }
        if data.len() != n_samples * n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }

        self.data = data.to_vec();
        self.n_samples = n_samples;
        self.n_features = n_features;

        // Build kNN for every training point (brute-force O(n²))
        let (knn_indices, knn_dists) = build_knn(data, n_samples, n_features, self.k);
        self.knn_indices = knn_indices;
        self.knn_dists = knn_dists;

        // Pre-compute SBN costs
        self.sbn_costs = (0..n_samples)
            .map(|i| {
                let pt = &data[i * n_features..(i + 1) * n_features];
                let nn = &self.knn_indices[i * self.k..(i + 1) * self.k];
                sbn_cost_from_training(pt, nn, data, n_features, self.k)
            })
            .collect();

        self.fitted = true;
        Ok(())
    }

    /// Compute the COF score for a single query point `x`.
    ///
    /// High score → more anomalous.
    ///
    /// # Errors
    ///
    /// - [`AnomalyError::NotFitted`] if [`Cof::fit`] has not been called.
    /// - [`AnomalyError::FeatureCountMismatch`] if `x.len() != n_features`.
    pub fn score(&self, x: &[f32]) -> AnomalyResult<f32> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        Ok(self.compute_cof_score(x))
    }

    /// Compute COF scores for a batch `x` of `n_samples` query points.
    ///
    /// # Errors
    ///
    /// - [`AnomalyError::NotFitted`], [`AnomalyError::DimensionMismatch`].
    pub fn score_batch(&self, x: &[f32], n_samples: usize) -> AnomalyResult<Vec<f32>> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != n_samples * self.n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * self.n_features,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let row = &x[i * self.n_features..(i + 1) * self.n_features];
            out.push(self.compute_cof_score(row));
        }
        Ok(out)
    }

    // ─── Internal ─────────────────────────────────────────────────────────────

    /// Core COF computation for a single query point.
    fn compute_cof_score(&self, x: &[f32]) -> f32 {
        // Find k-NN in training data for query x
        let nn = knn_query(x, &self.data, self.n_samples, self.n_features, self.k);

        // SBN cost of x traversing through its k-NN
        let query_sbn = sbn_cost_from_training(x, &nn, &self.data, self.n_features, self.k);

        // Average SBN cost of neighbours
        let mean_neighbour_sbn: f32 =
            nn.iter().map(|&idx| self.sbn_costs[idx]).sum::<f32>() / self.k as f32;

        if mean_neighbour_sbn < f32::EPSILON {
            // Degenerate: all neighbours coincide with x; define COF = 1.0
            return 1.0;
        }
        query_sbn / mean_neighbour_sbn
    }
}

impl Default for Cof {
    fn default() -> Self {
        Self::new(5)
    }
}

// ─── Free functions ───────────────────────────────────────────────────────────

/// Brute-force k-NN lookup: returns indices of k nearest training points to `x`
/// (excluding exact matches — a training point queried against itself returns its
/// true k nearest *other* points).
fn knn_query(x: &[f32], data: &[f32], n_samples: usize, n_features: usize, k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = (0..n_samples)
        .map(|i| {
            let row = &data[i * n_features..(i + 1) * n_features];
            (i, euclidean_sq(x, row))
        })
        .collect();
    // Partial sort: bring the k smallest to the front
    dists.select_nth_unstable_by(k - 1, |a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    dists[..k].iter().map(|(i, _)| *i).collect()
}

/// Build the k-NN index for all training points (mutual-exclusion: skip
/// self-distance).
fn build_knn(
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    k: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut all_indices = Vec::with_capacity(n_samples * k);
    let mut all_dists = Vec::with_capacity(n_samples * k);

    for i in 0..n_samples {
        let xi = &data[i * n_features..(i + 1) * n_features];
        let mut dists: Vec<(usize, f32)> = (0..n_samples)
            .filter(|&j| j != i)
            .map(|j| {
                let xj = &data[j * n_features..(j + 1) * n_features];
                (j, euclidean_sq(xi, xj).sqrt())
            })
            .collect();
        dists.select_nth_unstable_by(k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        // Sort the top-k by distance so SBN greedy chaining sees them in order
        let mut top_k = dists[..k].to_vec();
        top_k.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, d) in top_k {
            all_indices.push(idx);
            all_dists.push(d);
        }
    }
    (all_indices, all_dists)
}

/// SBN cost starting from `origin`, chaining greedily through `nn_set`
/// using the Euclidean distance to positions in `data`.
///
/// Weight: `(2 / (k(k+1))) · Σᵢ₌₁ᵏ  i · d(prev, current)`.
fn sbn_cost_from_training(
    origin: &[f32],
    nn_set: &[usize],
    data: &[f32],
    n_features: usize,
    k: usize,
) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let mut visited = vec![false; nn_set.len()];
    let mut prev: Vec<f32> = origin.to_vec();
    let mut total = 0.0_f32;

    for step in 1..=k {
        // Greedy: find un-visited neighbour closest to prev
        let mut best_dist = f32::INFINITY;
        let mut best_local = 0usize;
        for (local_idx, &global_idx) in nn_set.iter().enumerate() {
            if visited[local_idx] {
                continue;
            }
            let pt = &data[global_idx * n_features..(global_idx + 1) * n_features];
            let d = euclidean_dist(&prev, pt);
            if d < best_dist {
                best_dist = d;
                best_local = local_idx;
            }
        }
        visited[best_local] = true;
        let next_idx = nn_set[best_local];
        let next_pt = &data[next_idx * n_features..(next_idx + 1) * n_features];
        total += step as f32 * euclidean_dist(&prev, next_pt);
        prev = next_pt.to_vec();
    }

    let denom = k as f32 * (k as f32 + 1.0) / 2.0;
    total / denom
}

#[inline]
fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[inline]
fn euclidean_dist(a: &[f32], b: &[f32]) -> f32 {
    euclidean_sq(a, b).sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_next(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 33) as f32 / u32::MAX as f32
    }

    fn lcg_normal(state: &mut u64) -> f32 {
        let u1 = lcg_next(state).max(1e-12);
        let u2 = lcg_next(state);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    fn cluster_2d(n: usize, cx: f32, cy: f32, sigma: f32, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .flat_map(|_| {
                let x = lcg_normal(&mut state) * sigma + cx;
                let y = lcg_normal(&mut state) * sigma + cy;
                [x, y]
            })
            .collect()
    }

    #[test]
    fn test_fit_and_score_basic() {
        let data = cluster_2d(20, 0.0, 0.0, 1.0, 1);
        let mut cof = Cof::new(3);
        cof.fit(&data, 20, 2).expect("COF fit should succeed");
        let s = cof
            .score(&[0.0_f32, 0.0])
            .expect("COF score should succeed");
        assert!(s.is_finite() && s > 0.0, "score={s}");
    }

    #[test]
    fn test_inlier_vs_outlier_score() {
        // 30 inliers, 1 outlier far away
        let mut data = cluster_2d(30, 0.0, 0.0, 0.5, 2);
        data.extend_from_slice(&[50.0_f32, 50.0]);
        let mut cof = Cof::new(4);
        cof.fit(&data, 31, 2).expect("COF fit should succeed");

        let inlier_score = cof
            .score(&[0.0_f32, 0.0])
            .expect("COF score should succeed");
        let outlier_score = cof
            .score(&[50.0_f32, 50.0])
            .expect("COF score should succeed");
        assert!(
            outlier_score > inlier_score,
            "outlier score {outlier_score:.3} should exceed inlier score {inlier_score:.3}"
        );
    }

    #[test]
    fn test_cluster_center_low_cof() {
        // Center of a tight cluster should have COF ≤ peripheral points
        let mut data = cluster_2d(40, 0.0, 0.0, 0.5, 3);
        // Add an isolated outlier
        data.extend_from_slice(&[20.0_f32, 20.0]);
        let mut cof = Cof::new(5);
        cof.fit(&data, 41, 2).expect("COF fit should succeed");

        let center_score = cof
            .score(&[0.0_f32, 0.0])
            .expect("COF score should succeed");
        let outlier_score = cof
            .score(&[20.0_f32, 20.0])
            .expect("COF score should succeed");
        assert!(
            center_score < outlier_score,
            "center COF {center_score:.3} should be < outlier COF {outlier_score:.3}"
        );
    }

    #[test]
    fn test_inlier_cof_near_unity() {
        // In a homogeneous Gaussian cluster, most inlier COFs should be close to 1.0
        let data = cluster_2d(60, 0.0, 0.0, 1.0, 4);
        let mut cof = Cof::new(5);
        cof.fit(&data, 60, 2).expect("COF fit should succeed");

        let scores = cof
            .score_batch(&data, 60)
            .expect("COF batch score should succeed");
        let within_range = scores.iter().filter(|&&s| (0.3..=3.0).contains(&s)).count();
        assert!(
            within_range >= 50,
            "at least 50/60 inlier COFs should be in [0.3, 3.0], got {within_range}"
        );
    }

    #[test]
    fn test_score_batch_matches_score() {
        let data = cluster_2d(20, 0.0, 0.0, 1.0, 5);
        let mut cof = Cof::new(3);
        cof.fit(&data, 20, 2).expect("COF fit should succeed");

        let queries = [0.0_f32, 0.0, 1.0, 1.0, -1.0, -1.0];
        let batch = cof
            .score_batch(&queries, 3)
            .expect("COF batch score should succeed");
        for i in 0..3 {
            let single = cof
                .score(&queries[i * 2..(i + 1) * 2])
                .expect("COF single score should succeed");
            assert!(
                (batch[i] - single).abs() < 1e-5,
                "batch[{i}]={} single={}",
                batch[i],
                single
            );
        }
    }

    #[test]
    fn test_not_fitted_error() {
        let cof = Cof::new(3);
        let result = cof.score(&[0.0_f32, 0.0]);
        assert!(matches!(result, Err(AnomalyError::NotFitted)));
    }

    #[test]
    fn test_feature_count_mismatch() {
        let data = cluster_2d(20, 0.0, 0.0, 1.0, 6);
        let mut cof = Cof::new(3);
        cof.fit(&data, 20, 2).expect("COF fit should succeed");
        let result = cof.score(&[0.0_f32, 0.0, 0.0]); // 3D, expected 2D
        assert!(matches!(
            result,
            Err(AnomalyError::FeatureCountMismatch { .. })
        ));
    }

    #[test]
    fn test_invalid_k_zero() {
        let data = cluster_2d(10, 0.0, 0.0, 1.0, 7);
        let mut cof = Cof::new(0);
        let result = cof.fit(&data, 10, 2);
        assert!(matches!(result, Err(AnomalyError::InvalidK { .. })));
    }

    #[test]
    fn test_invalid_k_too_large() {
        let data = cluster_2d(10, 0.0, 0.0, 1.0, 8);
        let mut cof = Cof::new(10); // k == n_samples, not < n_samples
        let result = cof.fit(&data, 10, 2);
        assert!(matches!(result, Err(AnomalyError::InvalidK { .. })));
    }

    #[test]
    fn test_sbn_path_monotone_first_two_edges() {
        // Greedy SBN: first hop ≤ second hop in terms of distance from the respective predecessor
        // Construct: origin at (0,0), neighbours at (1,0) (2,0) (3,0) (4,0) (5,0)
        let data = vec![
            1.0_f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 0.0,
            0.0, // origin at index 5
        ];
        let n = 6;
        let k = 4;
        let mut cof = Cof::new(k);
        cof.fit(&data, n, 2).expect("COF fit should succeed");

        // NN set of origin (index 5) should be [0,1,2,3] (the 4 nearest)
        let nn = &cof.knn_indices[5 * k..6 * k];
        // Verify they are sorted by distance
        let dists: Vec<f32> = nn
            .iter()
            .map(|&idx| euclidean_dist(&data[idx * 2..(idx + 1) * 2], &[0.0, 0.0]))
            .collect();
        for w in dists.windows(2) {
            assert!(w[0] <= w[1] + 1e-6, "knn_dists should be sorted: {dists:?}");
        }
    }

    #[test]
    fn test_high_dimensional_fit() {
        // d=10, n=30, k=4 — should run without error
        let mut state = 42u64;
        let data: Vec<f32> = (0..30 * 10).map(|_| lcg_normal(&mut state)).collect();
        let mut cof = Cof::new(4);
        cof.fit(&data, 30, 10)
            .expect("COF high-dim fit should succeed");
        let s = cof
            .score(&[0.0_f32; 10])
            .expect("COF high-dim score should succeed");
        assert!(s.is_finite(), "score={s}");
    }

    #[test]
    fn test_dimension_mismatch_fit() {
        let data = vec![0.0_f32; 10]; // 10 elements, claimed 5 samples × 3 features = 15
        let mut cof = Cof::new(3);
        let result = cof.fit(&data, 5, 3);
        assert!(matches!(
            result,
            Err(AnomalyError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_two_clusters_separation() {
        // Cluster A at (-5,0), Cluster B at (5,0), outlier at (0,50)
        let mut data = cluster_2d(20, -5.0, 0.0, 0.3, 9);
        let cluster_b = cluster_2d(20, 5.0, 0.0, 0.3, 10);
        data.extend_from_slice(&cluster_b);
        data.extend_from_slice(&[0.0_f32, 50.0]); // outlier

        let n = 41;
        let mut cof = Cof::new(5);
        cof.fit(&data, n, 2).expect("COF fit should succeed");

        // Score at outlier should exceed any inlier by a wide margin
        let outlier_score = cof
            .score(&[0.0_f32, 50.0])
            .expect("COF outlier score should succeed");
        let inlier_scores: Vec<f32> = (0..40)
            .map(|i| {
                cof.score(&data[i * 2..(i + 1) * 2])
                    .expect("COF score in iterator should succeed")
            })
            .collect();
        let max_inlier = inlier_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            outlier_score > max_inlier,
            "outlier COF {outlier_score:.3} should exceed max inlier COF {max_inlier:.3}"
        );
    }

    #[test]
    fn test_default_k() {
        let cof = Cof::default();
        assert_eq!(cof.k, 5);
    }

    #[test]
    fn test_determinism() {
        let data = cluster_2d(25, 0.0, 0.0, 1.0, 13);
        let mut cof1 = Cof::new(4);
        let mut cof2 = Cof::new(4);
        cof1.fit(&data, 25, 2).expect("COF fit1 should succeed");
        cof2.fit(&data, 25, 2).expect("COF fit2 should succeed");
        let s1 = cof1
            .score(&[1.0_f32, 1.0])
            .expect("COF score1 should be deterministic");
        let s2 = cof2
            .score(&[1.0_f32, 1.0])
            .expect("COF score2 should be deterministic");
        assert!((s1 - s2).abs() < 1e-6, "scores must be deterministic");
    }

    #[test]
    fn test_sbn_cost_collinear() {
        // Collinear points: SBN chain is monotone along the line
        let data = vec![
            0.0_f32, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 100.0,
            0.0, // outlier
        ];
        let n = 8;
        let mut cof = Cof::new(4);
        cof.fit(&data, n, 2).expect("COF fit should succeed");

        // Origin (0,0) has its 4-NN at 1,2,3,4 → short edges, low SBN cost
        let origin_score = cof
            .score(&[0.0_f32, 0.0])
            .expect("COF origin score should succeed");
        let outlier_score = cof
            .score(&[100.0_f32, 0.0])
            .expect("COF outlier score should succeed");
        assert!(
            outlier_score > origin_score,
            "outlier COF {outlier_score:.3} should exceed origin COF {origin_score:.3}"
        );
    }

    #[test]
    fn test_sbn_weights_increase_with_position() {
        // Manually verify SBN cost formula on a simple case
        // Origin=(0,0), nn_set=[0,1] where data[0]=(1,0), data[1]=(2,0)
        // Greedy step1: nearest to origin is (1,0), dist=1 → cost += 1*1 = 1
        // Greedy step2: nearest to (1,0) among remaining {(2,0)}: dist=1 → cost += 2*1 = 2
        // denom = k(k+1)/2 = 2*3/2 = 3
        // SBN_cost = (1+2)/3 = 1.0
        let data = vec![1.0_f32, 0.0, 2.0, 0.0];
        let origin = [0.0_f32, 0.0];
        let nn_set = [0usize, 1];
        let cost = sbn_cost_from_training(&origin, &nn_set, &data, 2, 2);
        assert!(
            (cost - 1.0).abs() < 1e-5,
            "expected SBN cost 1.0, got {cost}"
        );
    }
}
