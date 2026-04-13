//! Expert parallelism for MoE inference.
//!
//! In Mixture-of-Experts architectures each token is routed to one or more
//! expert sub-networks.  When `ep > 1` experts are partitioned across GPUs:
//! rank `r` owns experts `r*(n_experts/ep) .. (r+1)*(n_experts/ep)`.
//!
//! # Communication pattern
//!
//! 1. **Token routing**: each rank determines, for each of its tokens, which
//!    expert it should go to.
//! 2. **All-to-all dispatch**: tokens are sent to the GPU that owns the target
//!    expert.  Simulated here by sorting tokens into expert-local buffers.
//! 3. **Local expert compute**: each rank runs its subset of experts on the
//!    dispatched tokens.
//! 4. **All-to-all gather**: results are sent back to the originating rank.
//!
//! # Modules
//!
//! * `router`  — `TopKRouter`: compute expert assignment scores and top-K selection.
//! * `dispatch` — `ExpertDispatcher`: all-to-all scatter/gather simulation.

pub mod dispatch;
pub mod router;

pub use dispatch::ExpertDispatcher;
pub use router::TopKRouter;
