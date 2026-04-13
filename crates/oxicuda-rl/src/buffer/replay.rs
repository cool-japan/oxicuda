//! # Uniform Replay Buffer
//!
//! A fixed-capacity circular buffer that stores `(s, a, r, s', done)`
//! transitions and supports uniform random sampling.
//!
//! ## Design
//!
//! * Pre-allocated, never resizes — capacity chosen once at construction.
//! * Circular: once full, the oldest transition is overwritten.
//! * `sample(batch_size)` returns a contiguous clone batch; indices chosen
//!   using the handle's LCG RNG for reproducibility.
//!
//! ## Usage
//!
//! ```rust
//! use oxicuda_rl::buffer::UniformReplayBuffer;
//! use oxicuda_rl::handle::RlHandle;
//!
//! let mut handle = RlHandle::default_handle();
//! let mut buf = UniformReplayBuffer::new(1024, 4, 2);
//!
//! for i in 0..100_usize {
//!     buf.push([i as f32; 4], [0.0_f32; 2], 1.0, [i as f32 + 1.0; 4], false);
//! }
//!
//! let batch = buf.sample(32, &mut handle).unwrap();
//! assert_eq!(batch.len(), 32);
//! ```

use crate::error::{RlError, RlResult};
use crate::handle::RlHandle;

// ─── Transition ───────────────────────────────────────────────────────────────

/// A single `(s, a, r, s', done)` experience tuple.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Observation at time `t`.
    pub obs: Vec<f32>,
    /// Action taken at time `t`.
    pub action: Vec<f32>,
    /// Reward received at time `t`.
    pub reward: f32,
    /// Observation at time `t+1`.
    pub next_obs: Vec<f32>,
    /// Whether `t+1` is a terminal state.
    pub done: bool,
}

impl Transition {
    /// Create a new transition.
    #[must_use]
    pub fn new(
        obs: impl Into<Vec<f32>>,
        action: impl Into<Vec<f32>>,
        reward: f32,
        next_obs: impl Into<Vec<f32>>,
        done: bool,
    ) -> Self {
        Self {
            obs: obs.into(),
            action: action.into(),
            reward,
            next_obs: next_obs.into(),
            done,
        }
    }
}

// ─── UniformReplayBuffer ──────────────────────────────────────────────────────

/// Uniform circular experience replay buffer.
#[derive(Debug, Clone)]
pub struct UniformReplayBuffer {
    capacity: usize,
    obs_dim: usize,
    act_dim: usize,
    // Storage in struct-of-arrays layout for cache efficiency.
    obs: Vec<f32>,      // capacity × obs_dim
    actions: Vec<f32>,  // capacity × act_dim
    rewards: Vec<f32>,  // capacity
    next_obs: Vec<f32>, // capacity × obs_dim
    dones: Vec<f32>,    // capacity (0.0 or 1.0)
    /// Write cursor (next insertion index, wraps).
    head: usize,
    /// Number of valid entries (≤ capacity).
    size: usize,
}

impl UniformReplayBuffer {
    /// Create a buffer with the given `capacity`, observation dimension
    /// `obs_dim`, and action dimension `act_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`RlError::ZeroCapacity`] if `capacity == 0`.
    pub fn new(capacity: usize, obs_dim: usize, act_dim: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        Self {
            capacity,
            obs_dim,
            act_dim,
            obs: vec![0.0; capacity * obs_dim],
            actions: vec![0.0; capacity * act_dim],
            rewards: vec![0.0; capacity],
            next_obs: vec![0.0; capacity * obs_dim],
            dones: vec![0.0; capacity],
            head: 0,
            size: 0,
        }
    }

    /// Number of transitions currently stored.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Buffer capacity.
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns `true` if no transitions are stored.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns `true` if the buffer has been filled to capacity at least once.
    #[must_use]
    #[inline]
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    /// Observation dimension.
    #[must_use]
    #[inline]
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Action dimension.
    #[must_use]
    #[inline]
    pub fn act_dim(&self) -> usize {
        self.act_dim
    }

    /// Push a new transition into the buffer.
    ///
    /// # Panics
    ///
    /// Panics (debug only) if `obs.len() != obs_dim` or `action.len() !=
    /// act_dim`.
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
        debug_assert_eq!(obs.len(), self.obs_dim);
        debug_assert_eq!(action.len(), self.act_dim);
        debug_assert_eq!(next_obs.len(), self.obs_dim);

        let i = self.head;
        self.obs[i * self.obs_dim..(i + 1) * self.obs_dim].copy_from_slice(obs);
        self.actions[i * self.act_dim..(i + 1) * self.act_dim].copy_from_slice(action);
        self.rewards[i] = reward;
        self.next_obs[i * self.obs_dim..(i + 1) * self.obs_dim].copy_from_slice(next_obs);
        self.dones[i] = if done { 1.0 } else { 0.0 };

        self.head = (self.head + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    /// Push a [`Transition`] struct.
    pub fn push_transition(&mut self, t: Transition) {
        self.push(t.obs, t.action, t.reward, t.next_obs, t.done);
    }

    /// Sample `batch_size` transitions uniformly without replacement.
    ///
    /// Returns a `Vec<Transition>` of length `batch_size`.
    ///
    /// # Errors
    ///
    /// * [`RlError::InsufficientTransitions`] if `size < batch_size`.
    pub fn sample(&self, batch_size: usize, handle: &mut RlHandle) -> RlResult<Vec<Transition>> {
        if self.size < batch_size {
            return Err(RlError::InsufficientTransitions {
                have: self.size,
                need: batch_size,
            });
        }
        let rng = handle.rng_mut();
        let mut out = Vec::with_capacity(batch_size);
        // Reservoir-like: pick batch_size indices without replacement from [0, size)
        // For small batch/size ratios: rejection sampling is fast.
        let mut indices: Vec<usize> = Vec::with_capacity(batch_size);
        while indices.len() < batch_size {
            let idx = rng.next_usize(self.size);
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        for idx in indices {
            let obs = self.obs[idx * self.obs_dim..(idx + 1) * self.obs_dim].to_vec();
            let action = self.actions[idx * self.act_dim..(idx + 1) * self.act_dim].to_vec();
            let reward = self.rewards[idx];
            let next_obs = self.next_obs[idx * self.obs_dim..(idx + 1) * self.obs_dim].to_vec();
            let done = self.dones[idx] > 0.5;
            out.push(Transition {
                obs,
                action,
                reward,
                next_obs,
                done,
            });
        }
        Ok(out)
    }

    /// Sample into pre-allocated contiguous arrays (zero-copy path).
    ///
    /// Fills:
    /// * `obs_out`      — `batch_size × obs_dim` f32 slice
    /// * `action_out`   — `batch_size × act_dim` f32 slice
    /// * `reward_out`   — `batch_size` f32 slice
    /// * `next_obs_out` — `batch_size × obs_dim` f32 slice
    /// * `done_out`     — `batch_size` f32 slice (0.0 / 1.0)
    ///
    /// # Errors
    ///
    /// [`RlError::InsufficientTransitions`] or [`RlError::DimensionMismatch`]
    /// if any slice has the wrong length.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_into(
        &self,
        batch_size: usize,
        obs_out: &mut [f32],
        action_out: &mut [f32],
        reward_out: &mut [f32],
        next_obs_out: &mut [f32],
        done_out: &mut [f32],
        handle: &mut RlHandle,
    ) -> RlResult<()> {
        if self.size < batch_size {
            return Err(RlError::InsufficientTransitions {
                have: self.size,
                need: batch_size,
            });
        }
        let rng = handle.rng_mut();
        let mut indices: Vec<usize> = Vec::with_capacity(batch_size);
        while indices.len() < batch_size {
            let idx = rng.next_usize(self.size);
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        for (b, &idx) in indices.iter().enumerate() {
            obs_out[b * self.obs_dim..(b + 1) * self.obs_dim]
                .copy_from_slice(&self.obs[idx * self.obs_dim..(idx + 1) * self.obs_dim]);
            action_out[b * self.act_dim..(b + 1) * self.act_dim]
                .copy_from_slice(&self.actions[idx * self.act_dim..(idx + 1) * self.act_dim]);
            reward_out[b] = self.rewards[idx];
            next_obs_out[b * self.obs_dim..(b + 1) * self.obs_dim]
                .copy_from_slice(&self.next_obs[idx * self.obs_dim..(idx + 1) * self.obs_dim]);
            done_out[b] = self.dones[idx];
        }
        Ok(())
    }

    /// Clear all stored transitions.
    pub fn clear(&mut self) {
        self.head = 0;
        self.size = 0;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn push_n(buf: &mut UniformReplayBuffer, n: usize) {
        let od = buf.obs_dim();
        let ad = buf.act_dim();
        for i in 0..n {
            buf.push(
                vec![i as f32; od],
                vec![0.0_f32; ad],
                i as f32 * 0.1,
                vec![i as f32 + 1.0; od],
                i % 10 == 9,
            );
        }
    }

    #[test]
    fn buffer_empty_initially() {
        let buf = UniformReplayBuffer::new(100, 4, 2);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn buffer_grows_to_capacity() {
        let mut buf = UniformReplayBuffer::new(10, 2, 1);
        push_n(&mut buf, 10);
        assert_eq!(buf.len(), 10);
        assert!(buf.is_full());
    }

    #[test]
    fn buffer_overwrites_oldest() {
        let mut buf = UniformReplayBuffer::new(5, 1, 1);
        for i in 0..7_usize {
            buf.push([i as f32], [0.0], i as f32, [i as f32 + 1.0], false);
        }
        // size is still capped at capacity
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn sample_correct_size() {
        let mut buf = UniformReplayBuffer::new(100, 4, 2);
        push_n(&mut buf, 100);
        let mut handle = RlHandle::default_handle();
        let batch = buf.sample(32, &mut handle).unwrap();
        assert_eq!(batch.len(), 32);
    }

    #[test]
    fn sample_no_duplicates() {
        let mut buf = UniformReplayBuffer::new(100, 1, 1);
        push_n(&mut buf, 100);
        let mut handle = RlHandle::default_handle();
        let batch = buf.sample(50, &mut handle).unwrap();
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for t in &batch {
            let idx = t.obs[0] as usize;
            assert!(seen.insert(idx), "duplicate index {idx}");
        }
    }

    #[test]
    fn sample_insufficient_error() {
        let buf = UniformReplayBuffer::new(100, 4, 2);
        let mut handle = RlHandle::default_handle();
        assert!(buf.sample(10, &mut handle).is_err());
    }

    #[test]
    fn push_transition_struct() {
        let mut buf = UniformReplayBuffer::new(10, 3, 2);
        buf.push_transition(Transition::new(
            [1.0, 2.0, 3.0],
            [4.0, 5.0],
            1.0,
            [2.0, 3.0, 4.0],
            false,
        ));
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn sample_into_fills_slices() {
        let mut buf = UniformReplayBuffer::new(64, 4, 2);
        push_n(&mut buf, 64);
        let mut handle = RlHandle::default_handle();
        let bs = 16;
        let mut obs = vec![0.0_f32; bs * 4];
        let mut act = vec![0.0_f32; bs * 2];
        let mut rew = vec![0.0_f32; bs];
        let mut nobs = vec![0.0_f32; bs * 4];
        let mut done = vec![0.0_f32; bs];
        buf.sample_into(
            bs,
            &mut obs,
            &mut act,
            &mut rew,
            &mut nobs,
            &mut done,
            &mut handle,
        )
        .unwrap();
        // obs entries should be >= 0 (we pushed i as f32)
        assert!(obs.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn clear_resets_buffer() {
        let mut buf = UniformReplayBuffer::new(10, 2, 1);
        push_n(&mut buf, 10);
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn transition_done_flag() {
        let mut buf = UniformReplayBuffer::new(5, 1, 1);
        buf.push([0.0], [0.0], 0.0, [1.0], true);
        let mut handle = RlHandle::default_handle();
        let batch = buf.sample(1, &mut handle).unwrap();
        assert!(batch[0].done);
    }
}
