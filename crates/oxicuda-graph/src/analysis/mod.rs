//! Analysis passes for the OxiCUDA computation graph.
//!
//! This module groups three complementary analysis algorithms:
//!
//! * [`topo`] — Topological level assignment, ASAP/ALAP scheduling, slack
//!   computation, and critical-path identification.
//! * [`liveness`] — Buffer liveness intervals, interference detection, and
//!   peak-live-memory estimation.
//! * [`dominance`] — Lengauer–Tarjan dominator tree for control-flow analysis,
//!   operator fusion eligibility, and stream partitioning.

pub mod dominance;
pub mod liveness;
pub mod topo;

pub use dominance::{DomTree, analyse as dominance_analyse};
pub use liveness::{LiveInterval, LivenessAnalysis, analyse as liveness_analyse};
pub use topo::{NodeInfo, TopoAnalysis, analyse as topo_analyse};
