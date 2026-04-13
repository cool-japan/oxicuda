//! RL algorithm loss functions.
//!
//! * [`crate::loss::ppo_loss`]  — PPO clip + value + entropy loss
//! * [`crate::loss::dqn_loss`]  — DQN and Double-DQN Bellman error
//! * [`crate::loss::sac_critic_loss`]  — SAC soft Q + policy + temperature loss
//! * [`crate::loss::td3_critic_loss`]  — TD3 actor-critic losses

pub mod dqn;
pub mod ppo;
pub mod sac;
pub mod td3;

pub use dqn::{DqnConfig, DqnLoss, double_dqn_loss, dqn_loss};
pub use ppo::{PpoConfig, PpoLoss, ppo_loss};
pub use sac::{SacConfig, SacLoss, sac_actor_loss, sac_critic_loss, sac_temperature_loss};
pub use td3::{Td3Config, Td3Loss, td3_actor_loss, td3_critic_loss};
