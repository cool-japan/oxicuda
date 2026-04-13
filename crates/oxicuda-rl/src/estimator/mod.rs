//! Return and advantage estimators for on-policy and off-policy RL algorithms.
//!
//! * [`crate::estimator::compute_gae`] — Generalized Advantage Estimation (PPO, A3C)
//! * [`crate::estimator::compute_td_lambda`] — TD(λ) multi-step returns
//! * [`crate::estimator::compute_vtrace`] — V-trace off-policy correction (IMPALA)
//! * [`crate::estimator::compute_retrace`] — Retrace(λ) safe off-policy returns

pub mod gae;
pub mod retrace;
pub mod td;
pub mod vtrace;

pub use gae::{GaeConfig, compute_gae};
pub use retrace::{RetraceConfig, compute_retrace};
pub use td::{TdConfig, compute_td_lambda};
pub use vtrace::{VtraceConfig, VtraceOutput, compute_vtrace};
