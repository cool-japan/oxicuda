//! Online normalization utilities for observations and rewards.
//!
//! * [`crate::normalize::RunningStats`] — Welford online mean/variance tracker
//! * [`crate::normalize::ObservationNormalizer`] — Observation normalizer
//! * [`crate::normalize::RewardNormalizer`] — Reward normalizer and clipper

pub mod obs_norm;
pub mod reward_norm;
pub mod running_stats;

pub use obs_norm::ObservationNormalizer;
pub use reward_norm::RewardNormalizer;
pub use running_stats::RunningStats;
