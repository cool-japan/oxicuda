//! # Prioritized Experience Replay (PER)
//!
//! Implements the proportional variant from Schaul et al. (2015), "Prioritized
//! Experience Replay", ICLR 2016.
//!
//! ## Algorithm
//!
//! Transitions are sampled with probability proportional to priority raised to
//! exponent `α`:
//! ```text
//! P(i) = p_i^α / Σ_j p_j^α
//! ```
//!
//! To correct for the sampling bias introduced by prioritised sampling, each
//! sampled transition receives an importance-sampling (IS) weight:
//! ```text
//! w_i = (1 / (N * P(i)))^β  (normalised by max_i w_i)
//! ```
//!
//! β is annealed from `beta_start` to 1 over the course of training.
//!
//! ## Implementation
//!
//! Priorities are stored in a **binary segment tree** of size `capacity` so
//! that:
//! * `O(log N)` priority update
//! * `O(log N)` stratified sampling
//! * `O(1)` total-priority query
//!
//! The tree is stored as a flat array of size `2 * capacity` (1-indexed, root
//! at index 1); leaves occupy indices `[capacity, 2*capacity)`.

use crate::buffer::replay::Transition;
use crate::error::{RlError, RlResult};
use crate::handle::RlHandle;

// ─── Segment tree ────────────────────────────────────────────────────────────

/// Min-heap-indexed sum segment tree for O(log N) priority queries.
#[derive(Debug, Clone)]
struct SumTree {
    n: usize,           // capacity (leaf count)
    tree: Vec<f64>,     // 2*n nodes, root at index 1, leaves at [n, 2n)
    min_tree: Vec<f64>, // parallel min tree for fast min-priority query
}

impl SumTree {
    fn new(n: usize) -> Self {
        Self {
            n,
            tree: vec![0.0; 2 * n],
            min_tree: vec![f64::MAX; 2 * n],
        }
    }

    /// Update priority at leaf `i` (0-indexed).
    fn update(&mut self, i: usize, priority: f64) {
        let pos = i + self.n; // leaf position in tree
        self.tree[pos] = priority;
        self.min_tree[pos] = priority;
        let mut p = pos >> 1;
        while p >= 1 {
            self.tree[p] = self.tree[2 * p] + self.tree[2 * p + 1];
            self.min_tree[p] = self.min_tree[2 * p].min(self.min_tree[2 * p + 1]);
            p >>= 1;
        }
    }

    /// Total sum of all priorities.
    #[inline]
    fn total(&self) -> f64 {
        self.tree[1]
    }

    /// Minimum priority among all leaves.
    #[inline]
    fn min_priority(&self) -> f64 {
        self.min_tree[1]
    }

    /// Find the leaf index (0-based) whose cumulative sum is just >= `value`.
    fn find(&self, value: f64) -> usize {
        let mut node = 1_usize;
        let mut v = value;
        while node < self.n {
            let left = 2 * node;
            if self.tree[left] >= v {
                node = left;
            } else {
                v -= self.tree[left];
                node = left + 1;
            }
        }
        node - self.n // convert back to 0-based leaf index
    }

    /// Priority at leaf `i` (0-indexed).
    fn priority_at(&self, i: usize) -> f64 {
        self.tree[i + self.n]
    }
}

// ─── PER buffer ──────────────────────────────────────────────────────────────

/// A sample returned by [`PrioritizedReplayBuffer::sample`].
#[derive(Debug, Clone)]
pub struct PrioritySample {
    /// The sampled transition.
    pub transition: Transition,
    /// Buffer index of this transition (needed for priority update).
    pub index: usize,
    /// Importance-sampling weight for this transition.
    pub weight: f32,
}

/// Prioritized Experience Replay buffer with proportional priority and IS
/// weights.
#[derive(Debug, Clone)]
pub struct PrioritizedReplayBuffer {
    capacity: usize,
    obs_dim: usize,
    act_dim: usize,
    // Transition storage (struct-of-arrays)
    obs: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    next_obs: Vec<f32>,
    dones: Vec<f32>,
    // Segment tree
    tree: SumTree,
    /// Priority exponent α ∈ [0, 1].  α=0 → uniform; α=1 → full priority.
    alpha: f64,
    /// IS-weight exponent β ∈ [0, 1].  Annealed from `beta_start` to 1.
    beta: f64,
    /// Maximum priority seen so far (used for new-experience priority).
    max_priority: f64,
    /// Write cursor.
    head: usize,
    /// Current size.
    size: usize,
}

impl PrioritizedReplayBuffer {
    /// Create a PER buffer.
    ///
    /// * `capacity` — maximum number of transitions.
    /// * `obs_dim`, `act_dim` — observation / action dimensions.
    /// * `alpha` — priority exponent (default 0.6).
    /// * `beta_start` — initial IS exponent (default 0.4, annealed to 1).
    pub fn new(
        capacity: usize,
        obs_dim: usize,
        act_dim: usize,
        alpha: f64,
        beta_start: f64,
    ) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        // Tree size must be a power of 2 for the segment tree to work correctly.
        let cap2 = capacity.next_power_of_two();
        Self {
            capacity,
            obs_dim,
            act_dim,
            obs: vec![0.0; capacity * obs_dim],
            actions: vec![0.0; capacity * act_dim],
            rewards: vec![0.0; capacity],
            next_obs: vec![0.0; capacity * obs_dim],
            dones: vec![0.0; capacity],
            tree: SumTree::new(cap2),
            alpha,
            beta: beta_start,
            max_priority: 1.0,
            head: 0,
            size: 0,
        }
    }

    /// Push a transition with maximum priority (so it will be sampled at least
    /// once).
    pub fn push(
        &mut self,
        obs: impl AsRef<[f32]>,
        action: impl AsRef<[f32]>,
        reward: f32,
        next_obs: impl AsRef<[f32]>,
        done: bool,
    ) {
        let obs = obs.as_ref();
        let action = action.as_ref();
        let next_obs = next_obs.as_ref();

        let i = self.head;
        self.obs[i * self.obs_dim..(i + 1) * self.obs_dim].copy_from_slice(obs);
        self.actions[i * self.act_dim..(i + 1) * self.act_dim].copy_from_slice(action);
        self.rewards[i] = reward;
        self.next_obs[i * self.obs_dim..(i + 1) * self.obs_dim].copy_from_slice(next_obs);
        self.dones[i] = if done { 1.0 } else { 0.0 };
        // Assign maximum priority to new transition
        self.tree.update(i, self.max_priority.powf(self.alpha));
        self.head = (self.head + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    /// Update priority for a transition that was previously sampled.
    ///
    /// # Arguments
    ///
    /// * `index` — the buffer index returned in [`PrioritySample::index`].
    /// * `priority` — new absolute priority (typically `|TD error| + ε`).
    pub fn update_priority(&mut self, index: usize, priority: f64) {
        let p = priority.max(1e-6);
        if p > self.max_priority {
            self.max_priority = p;
        }
        self.tree.update(index, p.powf(self.alpha));
    }

    /// Set the current β for IS weight computation.
    pub fn set_beta(&mut self, beta: f64) {
        self.beta = beta.clamp(0.0, 1.0);
    }

    /// Anneal β toward 1 by `step` (additive).
    pub fn anneal_beta(&mut self, step: f64) {
        self.beta = (self.beta + step).min(1.0);
    }

    /// Number of stored transitions.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Sample `batch_size` transitions using stratified proportional sampling.
    ///
    /// Stratified: the [0, total_priority] interval is divided into
    /// `batch_size` equal strata and one sample is drawn uniformly from each
    /// stratum.  This reduces variance compared to fully random sampling.
    ///
    /// # Errors
    ///
    /// * [`RlError::InsufficientTransitions`] if `size < batch_size`.
    /// * [`RlError::ZeroPrioritySum`] if all priorities are zero.
    pub fn sample(
        &self,
        batch_size: usize,
        handle: &mut RlHandle,
    ) -> RlResult<Vec<PrioritySample>> {
        if self.size < batch_size {
            return Err(RlError::InsufficientTransitions {
                have: self.size,
                need: batch_size,
            });
        }
        let total = self.tree.total();
        if total <= 0.0 {
            return Err(RlError::ZeroPrioritySum);
        }
        let rng = handle.rng_mut();
        let segment = total / batch_size as f64;
        let min_p = self.tree.min_priority() / total;
        let max_w = (1.0 / (self.size as f64 * min_p)).powf(self.beta) as f32;

        let mut out = Vec::with_capacity(batch_size);
        for k in 0..batch_size {
            let lo = k as f64 * segment;
            let hi = lo + segment;
            let v = lo + rng.next_f32() as f64 * (hi - lo);
            let idx = self.tree.find(v.min(total - 1e-9)).min(self.size - 1);

            let p = self.tree.priority_at(idx) / total;
            let w = ((1.0 / (self.size as f64 * p)).powf(self.beta) as f32 / max_w).min(1.0);

            let obs = self.obs[idx * self.obs_dim..(idx + 1) * self.obs_dim].to_vec();
            let action = self.actions[idx * self.act_dim..(idx + 1) * self.act_dim].to_vec();
            let reward = self.rewards[idx];
            let next_obs = self.next_obs[idx * self.obs_dim..(idx + 1) * self.obs_dim].to_vec();
            let done = self.dones[idx] > 0.5;
            out.push(PrioritySample {
                transition: Transition {
                    obs,
                    action,
                    reward,
                    next_obs,
                    done,
                },
                index: idx,
                weight: w,
            });
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_per(cap: usize) -> PrioritizedReplayBuffer {
        PrioritizedReplayBuffer::new(cap, 2, 1, 0.6, 0.4)
    }

    fn fill_per(buf: &mut PrioritizedReplayBuffer, n: usize) {
        for i in 0..n {
            buf.push(
                [i as f32, i as f32 + 1.0],
                [0.5_f32],
                i as f32 * 0.1,
                [i as f32 + 1.0, i as f32 + 2.0],
                false,
            );
        }
    }

    #[test]
    fn sum_tree_basic() {
        let mut t = SumTree::new(4);
        t.update(0, 1.0);
        t.update(1, 2.0);
        t.update(2, 3.0);
        t.update(3, 4.0);
        assert!((t.total() - 10.0).abs() < 1e-9, "total={}", t.total());
    }

    #[test]
    fn sum_tree_find() {
        let mut t = SumTree::new(4);
        t.update(0, 1.0);
        t.update(1, 2.0);
        t.update(2, 3.0);
        t.update(3, 4.0);
        // cumsum: [1, 3, 6, 10]  → value 2.5 should land in index 1 (cumsum ≤ 3)
        let idx = t.find(2.5);
        assert_eq!(idx, 1, "find(2.5) should return idx=1, got {idx}");
    }

    #[test]
    fn per_push_and_len() {
        let mut buf = make_per(32);
        fill_per(&mut buf, 20);
        assert_eq!(buf.len(), 20);
    }

    #[test]
    fn per_sample_size() {
        let mut buf = make_per(64);
        fill_per(&mut buf, 64);
        let mut handle = RlHandle::default_handle();
        let batch = buf.sample(16, &mut handle).unwrap();
        assert_eq!(batch.len(), 16);
    }

    #[test]
    fn per_weights_in_range() {
        let mut buf = make_per(64);
        fill_per(&mut buf, 64);
        let mut handle = RlHandle::default_handle();
        let batch = buf.sample(32, &mut handle).unwrap();
        for s in &batch {
            assert!(s.weight > 0.0 && s.weight <= 1.0, "weight={}", s.weight);
        }
    }

    #[test]
    fn per_update_priority() {
        let mut buf = make_per(16);
        fill_per(&mut buf, 16);
        // Update index 0 to a very high priority
        buf.update_priority(0, 100.0);
        // After many samples, index 0 should appear frequently
        let mut handle = RlHandle::default_handle();
        let mut counts = [0_usize; 16];
        for _ in 0..200 {
            let batch = buf.sample(1, &mut handle).unwrap();
            counts[batch[0].index] += 1;
        }
        // Index 0 should be sampled much more than average
        assert!(
            counts[0] > 200 / 16,
            "high-priority index should be over-sampled"
        );
    }

    #[test]
    fn per_insufficient_error() {
        let buf = make_per(16);
        let mut handle = RlHandle::default_handle();
        assert!(buf.sample(5, &mut handle).is_err());
    }

    #[test]
    fn per_anneal_beta() {
        let mut buf = make_per(16);
        buf.set_beta(0.4);
        buf.anneal_beta(0.3);
        assert!((buf.beta - 0.7).abs() < 1e-9);
        buf.anneal_beta(1.0);
        assert!((buf.beta - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sum_tree_min_priority() {
        let mut t = SumTree::new(4);
        t.update(0, 5.0);
        t.update(1, 2.0);
        t.update(2, 8.0);
        t.update(3, 3.0);
        assert!((t.min_priority() - 2.0).abs() < 1e-9);
    }
}
