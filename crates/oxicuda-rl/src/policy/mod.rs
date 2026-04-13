//! Policy distributions for discrete and continuous action spaces.
//!
//! * [`crate::policy::CategoricalPolicy`] — discrete actions via categorical / softmax distribution
//! * [`crate::policy::GaussianPolicy`] — continuous actions via diagonal Gaussian (reparameterised)
//! * [`crate::policy::DeterministicPolicy`] — deterministic policy for DDPG/TD3

pub mod categorical;
pub mod deterministic;
pub mod gaussian;

pub use categorical::CategoricalPolicy;
pub use deterministic::DeterministicPolicy;
pub use gaussian::GaussianPolicy;
