//! Experience replay buffers for off-policy RL algorithms.
//!
//! Provides three implementations:
//! * [`crate::buffer::UniformReplayBuffer`] — uniform random sampling (DQN, SAC, TD3)
//! * [`crate::buffer::PrioritizedReplayBuffer`] — proportional PER with segment
//!   tree sampling (PER-DQN, PER-SAC)
//! * [`crate::buffer::NStepBuffer`] — n-step return accumulation with discount rollout

pub mod n_step;
pub mod prioritized;
pub mod replay;

// Re-exports
pub use n_step::{NStepBuffer, NStepTransition};
pub use prioritized::{PrioritizedReplayBuffer, PrioritySample};
pub use replay::{Transition, UniformReplayBuffer};
