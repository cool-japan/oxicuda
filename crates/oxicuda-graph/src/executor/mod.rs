//! Executor backends for the OxiCUDA computation graph.
//!
//! This module provides two execution backends:
//!
//! * [`plan`] — The [`ExecutionPlan`] data structure produced by compiling a
//!   [`ComputeGraph`] through the full analysis and optimisation pipeline.
//!
//! * [`sequential`] — A CPU-side sequential simulator that walks the plan
//!   steps in order, suitable for unit testing and performance modelling
//!   without a GPU.
//!
//! * [`cuda_graph`] — Converts an `ExecutionPlan` into an
//!   `oxicuda_driver::graph::Graph` for low-overhead CUDA graph submission.
//!
//! [`ComputeGraph`]: crate::graph::ComputeGraph
//! [`ExecutionPlan`]: plan::ExecutionPlan

pub mod cuda_graph;
pub mod plan;
pub mod sequential;

pub use plan::{ExecutionPlan, PlanStep};
pub use sequential::{ExecutionStats, SequentialExecutor};
