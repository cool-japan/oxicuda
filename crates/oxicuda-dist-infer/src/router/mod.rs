//! Request routing for distributed inference.
//!
//! The `RequestRouter` selects which rank should handle a new inference request.
//! Three policies are implemented:
//!
//! * **RoundRobin** — cyclic dispatch, equal distribution by request count.
//! * **LeastLoaded** — dispatch to rank with the most available KV blocks.
//! * **PrefixAffinity** — prefer the rank that has a prefix cache hit for the
//!   request's prompt tokens; fall back to least-loaded on miss.

pub mod policy;
pub mod request;

pub use policy::{RouterMetrics, RoutingPolicy};
pub use request::{Request, RoutingDecision};
