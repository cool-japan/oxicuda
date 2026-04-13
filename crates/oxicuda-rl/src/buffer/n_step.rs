//! # N-step Return Buffer
//!
//! Accumulates `n` successive transitions and computes the n-step return:
//!
//! ```text
//! R_t^(n) = r_t + γ r_{t+1} + γ² r_{t+2} + … + γ^{n-1} r_{t+n-1}
//!           + γ^n V(s_{t+n}) * (1 - done)
//! ```
//!
//! When the trajectory terminates before `n` steps the return is truncated at
//! the terminal transition.
//!
//! This is used to improve the bias/variance tradeoff for off-policy algorithms
//! such as DQN and SAC.

use crate::error::{RlError, RlResult};

// ─── NStepTransition ─────────────────────────────────────────────────────────

/// A transition with n-step return pre-computed.
#[derive(Debug, Clone)]
pub struct NStepTransition {
    /// Observation at the start of the n-step window.
    pub obs: Vec<f32>,
    /// Action taken at the start of the window.
    pub action: Vec<f32>,
    /// Discounted n-step return `R_t^(n)`.
    pub n_step_return: f32,
    /// Observation at `t + n` (or at episode end).
    pub bootstrap_obs: Vec<f32>,
    /// Whether the episode ended within the n-step window.
    pub done: bool,
    /// Actual number of steps accumulated (≤ n, < n at episode end).
    pub actual_n: usize,
    /// `γ^actual_n` for bootstrapping.
    pub gamma_n: f32,
}

// ─── Internal single step ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Step {
    obs: Vec<f32>,
    action: Vec<f32>,
    reward: f32,
    next_obs: Vec<f32>,
    done: bool,
}

// ─── NStepBuffer ─────────────────────────────────────────────────────────────

/// Circular accumulator for n-step returns.
#[derive(Debug, Clone)]
pub struct NStepBuffer {
    n: usize,
    gamma: f32,
    steps: Vec<Option<Step>>,
    head: usize,
    count: usize,
}

impl NStepBuffer {
    /// Create an n-step buffer.
    ///
    /// * `n` — number of steps to accumulate.
    /// * `gamma` — discount factor γ ∈ (0, 1].
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    #[must_use]
    pub fn new(n: usize, gamma: f32) -> Self {
        assert!(n > 0, "n must be > 0");
        Self {
            n,
            gamma,
            steps: vec![None; n],
            head: 0,
            count: 0,
        }
    }

    /// Number of steps n.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Discount factor γ.
    #[must_use]
    #[inline]
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Current number of accumulated steps.
    #[must_use]
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Push a `(obs, action, reward, next_obs, done)` step.
    ///
    /// Returns `Some(NStepTransition)` once `n` steps have been accumulated
    /// (or immediately on terminal).  Returns `None` while the buffer is still
    /// filling up.
    pub fn push(
        &mut self,
        obs: impl Into<Vec<f32>>,
        action: impl Into<Vec<f32>>,
        reward: f32,
        next_obs: impl Into<Vec<f32>>,
        done: bool,
    ) -> Option<NStepTransition> {
        let step = Step {
            obs: obs.into(),
            action: action.into(),
            reward,
            next_obs: next_obs.into(),
            done,
        };
        self.steps[self.head] = Some(step);
        self.head = (self.head + 1) % self.n;
        if self.count < self.n {
            self.count += 1;
        }

        if self.count == self.n {
            Some(self.compute_return())
        } else if done {
            Some(self.compute_partial_return())
        } else {
            None
        }
    }

    /// Flush any remaining steps in the buffer (for end-of-episode cleanup).
    ///
    /// Returns all remaining n-step transitions.
    pub fn flush(&mut self) -> Vec<NStepTransition> {
        let mut out = Vec::new();
        while self.count > 0 {
            out.push(self.compute_partial_return());
            // Advance the oldest step pointer
            let oldest = (self.head + self.n - self.count) % self.n;
            self.steps[oldest] = None;
            self.count -= 1;
        }
        out
    }

    /// Build the full n-step transition (assumes `count == n`).
    fn compute_return(&self) -> NStepTransition {
        self.compute_n_step_return(self.n)
    }

    /// Build a partial n-step transition when episode ended early.
    fn compute_partial_return(&self) -> NStepTransition {
        self.compute_n_step_return(self.count)
    }

    fn compute_n_step_return(&self, steps: usize) -> NStepTransition {
        // Oldest step index in the circular buffer
        let oldest = (self.head + self.n - self.count) % self.n;
        let first = self.steps[oldest]
            .as_ref()
            .expect("oldest step must be Some");

        let mut cumulative = 0.0_f32;
        let mut gamma_k = 1.0_f32;
        let mut last_next_obs = first.next_obs.clone();
        let mut terminated = false;

        for k in 0..steps {
            let idx = (oldest + k) % self.n;
            let step = self.steps[idx].as_ref().expect("step must be Some");
            cumulative += gamma_k * step.reward;
            gamma_k *= self.gamma;
            last_next_obs = step.next_obs.clone();
            if step.done {
                terminated = true;
                break;
            }
        }

        NStepTransition {
            obs: first.obs.clone(),
            action: first.action.clone(),
            n_step_return: cumulative,
            bootstrap_obs: last_next_obs,
            done: terminated,
            actual_n: steps,
            gamma_n: gamma_k,
        }
    }

    /// Clear the buffer (e.g. at episode reset).
    pub fn reset(&mut self) {
        for s in self.steps.iter_mut() {
            *s = None;
        }
        self.head = 0;
        self.count = 0;
    }

    /// Attempt to produce an n-step transition from the current state.
    ///
    /// # Errors
    ///
    /// * [`RlError::NStepIncomplete`] if fewer than `n` steps have been
    ///   accumulated.
    pub fn try_get(&self) -> RlResult<NStepTransition> {
        if self.count < self.n {
            return Err(RlError::NStepIncomplete {
                have: self.count,
                need: self.n,
            });
        }
        Ok(self.compute_return())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buf(n: usize, gamma: f32) -> NStepBuffer {
        NStepBuffer::new(n, gamma)
    }

    // ── basic accumulation ───────────────────────────────────────────────────

    #[test]
    fn none_before_n_steps() {
        let mut buf = make_buf(3, 0.99);
        let r1 = buf.push([0.0], [0.0], 1.0, [1.0], false);
        let r2 = buf.push([1.0], [0.0], 1.0, [2.0], false);
        assert!(r1.is_none(), "should be None after 1 step");
        assert!(r2.is_none(), "should be None after 2 steps");
    }

    #[test]
    fn returns_transition_at_n() {
        let mut buf = make_buf(3, 0.99);
        buf.push([0.0], [0.0], 1.0, [1.0], false);
        buf.push([1.0], [0.0], 1.0, [2.0], false);
        let t = buf.push([2.0], [0.0], 1.0, [3.0], false);
        assert!(t.is_some(), "should return transition at n=3");
        let t = t.unwrap();
        // R = 1 + 0.99*1 + 0.99²*1 = 1 + 0.99 + 0.9801 = 2.9701
        assert!(
            (t.n_step_return - (1.0 + 0.99 + 0.99_f32 * 0.99)).abs() < 1e-4,
            "n_step_return={}",
            t.n_step_return
        );
        assert_eq!(t.actual_n, 3);
    }

    #[test]
    fn discount_applied_correctly() {
        let mut buf = make_buf(2, 0.5);
        buf.push([0.0], [0.0], 2.0, [1.0], false);
        let t = buf.push([1.0], [0.0], 4.0, [2.0], false);
        let t = t.unwrap();
        // R = 2 + 0.5 * 4 = 4.0
        assert!(
            (t.n_step_return - 4.0).abs() < 1e-5,
            "n_step_return={}",
            t.n_step_return
        );
        assert!((t.gamma_n - 0.25).abs() < 1e-5, "gamma_n={}", t.gamma_n);
    }

    // ── done flag ────────────────────────────────────────────────────────────

    #[test]
    fn terminal_truncates_return() {
        let mut buf = make_buf(5, 0.99);
        // Episode ends at step 2 (n < 5)
        buf.push([0.0], [0.0], 1.0, [1.0], false);
        let t = buf.push([1.0], [0.0], 2.0, [2.0], true); // done
        assert!(t.is_some(), "terminal step should emit transition early");
        let t = t.unwrap();
        assert!(t.done, "done flag should be set");
        // R = 1 + 0.99 * 2 = 2.98
        assert!(
            (t.n_step_return - (1.0 + 0.99 * 2.0)).abs() < 1e-4,
            "n_step_return={}",
            t.n_step_return
        );
    }

    // ── flush ────────────────────────────────────────────────────────────────

    #[test]
    fn flush_returns_remaining() {
        let mut buf = make_buf(3, 0.99);
        buf.push([0.0], [0.0], 1.0, [1.0], false);
        buf.push([1.0], [0.0], 2.0, [2.0], false);
        let flushed = buf.flush();
        // 2 partial returns: one 2-step, one 1-step
        assert!(
            !flushed.is_empty(),
            "flush should return partial transitions"
        );
    }

    #[test]
    fn flush_clears_buffer() {
        let mut buf = make_buf(3, 0.99);
        buf.push([0.0], [0.0], 1.0, [1.0], false);
        buf.flush();
        assert_eq!(buf.count(), 0);
    }

    // ── try_get ──────────────────────────────────────────────────────────────

    #[test]
    fn try_get_before_n_error() {
        let mut buf = make_buf(3, 0.99);
        buf.push([0.0], [0.0], 1.0, [1.0], false);
        assert!(buf.try_get().is_err());
    }

    #[test]
    fn try_get_after_n_ok() {
        let mut buf = make_buf(2, 0.99);
        buf.push([0.0], [0.0], 1.0, [1.0], false);
        buf.push([1.0], [0.0], 2.0, [2.0], false);
        assert!(buf.try_get().is_ok());
    }

    // ── reset ────────────────────────────────────────────────────────────────

    #[test]
    fn reset_clears() {
        let mut buf = make_buf(3, 0.99);
        buf.push([0.0], [0.0], 1.0, [1.0], false);
        buf.push([1.0], [0.0], 2.0, [2.0], false);
        buf.reset();
        assert_eq!(buf.count(), 0);
        assert!(buf.try_get().is_err());
    }

    // ── obs / action preservation ────────────────────────────────────────────

    #[test]
    fn obs_preserved_correctly() {
        let mut buf = make_buf(1, 0.9);
        let t = buf.push([7.0, 8.0], [3.0], 5.0, [9.0, 10.0], false);
        let t = t.unwrap();
        assert_eq!(t.obs, vec![7.0, 8.0]);
        assert_eq!(t.action, vec![3.0]);
        assert_eq!(t.bootstrap_obs, vec![9.0, 10.0]);
    }
}
