//! Sequential CPU-side executor for the `ExecutionPlan`.
//!
//! The sequential executor walks the plan steps in submission order on a
//! single thread, simulating the logical execution without a GPU. It is
//! primarily used for:
//!
//! * Unit testing of the compilation pipeline on machines without a GPU.
//! * Correctness checking: the executor validates that all event waits have
//!   a matching prior record on the correct stream.
//! * Profiling: it counts operations by type for performance modelling.
//!
//! # Event model
//!
//! The sequential executor simulates events using a simple boolean set.
//! `EventRecord(eid)` marks event `eid` as "fired". `EventWait(eid)` checks
//! that the event has been fired; if it has not, an error is returned.
//!
//! Because the sequential executor runs in a single-threaded, deterministic
//! order, all `EventRecord` steps necessarily precede the corresponding
//! `EventWait` steps if the plan was built correctly.

use std::collections::HashSet;

use crate::error::{GraphError, GraphResult};
use crate::executor::plan::{ExecutionPlan, PlanStep};
use crate::node::StreamId;

// ---------------------------------------------------------------------------
// ExecutionStats
// ---------------------------------------------------------------------------

/// Statistics collected during a sequential execution run.
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Number of kernel launches executed (after fusion).
    pub kernels_launched: usize,
    /// Total bytes copied (memcpy steps).
    pub bytes_copied: usize,
    /// Total bytes set (memset steps).
    pub bytes_set: usize,
    /// Number of event records issued.
    pub events_recorded: usize,
    /// Number of event waits satisfied.
    pub events_waited: usize,
    /// Number of host callbacks invoked.
    pub host_callbacks: usize,
    /// Number of barrier steps.
    pub barriers: usize,
    /// Number of steps that ran on stream 0.
    pub steps_on_stream0: usize,
    /// Number of steps that ran on non-zero streams.
    pub steps_on_other_streams: usize,
}

impl ExecutionStats {
    /// Total steps executed.
    pub fn total_steps(&self) -> usize {
        self.kernels_launched
            + self.events_recorded
            + self.events_waited
            + self.host_callbacks
            + self.barriers
            // memcpy and memset are subsumed into bytes_copied / bytes_set;
            // count separately:
            + self.steps_on_stream0
            + self.steps_on_other_streams
    }
}

// ---------------------------------------------------------------------------
// SequentialExecutor
// ---------------------------------------------------------------------------

/// Simulates execution of an [`ExecutionPlan`] sequentially on the CPU.
///
/// Instantiate with a plan, then call [`run`](Self::run) to execute it.
pub struct SequentialExecutor<'a> {
    plan: &'a ExecutionPlan,
}

impl<'a> SequentialExecutor<'a> {
    /// Creates a new executor for the given plan.
    pub fn new(plan: &'a ExecutionPlan) -> Self {
        Self { plan }
    }

    /// Executes all steps in submission order.
    ///
    /// Returns [`ExecutionStats`] summarising what was executed.
    ///
    /// # Errors
    ///
    /// * [`GraphError::InvalidPlan`] if an `EventWait` references an event
    ///   that has not yet been recorded.
    pub fn run(&self) -> GraphResult<ExecutionStats> {
        let mut stats = ExecutionStats::default();
        let mut fired_events: HashSet<usize> = HashSet::new();
        let mut memcpy_bytes = 0usize;
        let mut memset_bytes = 0usize;

        for step in &self.plan.steps {
            // Track stream statistics.
            match step.stream() {
                StreamId(0) => stats.steps_on_stream0 += 1,
                _ => stats.steps_on_other_streams += 1,
            }

            match step {
                PlanStep::KernelLaunch { .. } => {
                    stats.kernels_launched += 1;
                }
                PlanStep::Memcpy { size_bytes, .. } => {
                    memcpy_bytes += size_bytes;
                }
                PlanStep::Memset { size_bytes, .. } => {
                    memset_bytes += size_bytes;
                }
                PlanStep::EventRecord { event_id, .. } => {
                    fired_events.insert(*event_id);
                    stats.events_recorded += 1;
                }
                PlanStep::EventWait { event_id, stream } => {
                    if !fired_events.contains(event_id) {
                        return Err(GraphError::InvalidPlan(format!(
                            "EventWait for event {event_id} on stream {stream} but event was never recorded"
                        )));
                    }
                    stats.events_waited += 1;
                }
                PlanStep::HostCallback { .. } => {
                    stats.host_callbacks += 1;
                }
                PlanStep::Barrier { .. } => {
                    stats.barriers += 1;
                }
            }
        }

        stats.bytes_copied = memcpy_bytes;
        stats.bytes_set = memset_bytes;

        Ok(stats)
    }

    /// Validates the plan without executing it.
    ///
    /// Checks that:
    /// * All `EventWait`s have a preceding `EventRecord` with the same ID.
    /// * No duplicate event records exist (warns; does not error).
    ///
    /// Returns the number of validation issues found.
    pub fn validate(&self) -> GraphResult<usize> {
        let mut recorded: HashSet<usize> = HashSet::new();
        let mut issues = 0usize;
        for step in &self.plan.steps {
            match step {
                PlanStep::EventRecord { event_id, .. } if !recorded.insert(*event_id) => {
                    // Duplicate record is unusual but not fatal.
                    issues += 1;
                }
                PlanStep::EventRecord { .. } => {}
                PlanStep::EventWait { event_id, stream } if !recorded.contains(event_id) => {
                    return Err(GraphError::InvalidPlan(format!(
                        "EventWait({event_id}) on stream {stream} has no prior EventRecord"
                    )));
                }
                PlanStep::EventWait { .. } => {}
                _ => {}
            }
        }
        Ok(issues)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::executor::plan::ExecutionPlan;
    use crate::graph::ComputeGraph;
    use crate::node::MemcpyDir;

    fn simple_chain_graph() -> ComputeGraph {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let up = b.add_memcpy("up", MemcpyDir::HostToDevice, 2048);
        let k = b.add_kernel("k", 4, 256, 0).fusible(false).finish();
        let dn = b.add_memcpy("dn", MemcpyDir::DeviceToHost, 2048);
        b.chain(&[up, k, dn]);
        b.build().unwrap()
    }

    fn build_and_execute(graph: &ComputeGraph) -> GraphResult<ExecutionStats> {
        let plan = ExecutionPlan::build(graph, 4)?;
        SequentialExecutor::new(&plan).run()
    }

    #[test]
    fn seq_simple_chain_runs() {
        let g = simple_chain_graph();
        let stats = build_and_execute(&g).unwrap();
        assert_eq!(stats.kernels_launched, 1);
        assert_eq!(stats.bytes_copied, 2048 * 2);
    }

    #[test]
    fn seq_memset_counted() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        b.add_memset("zero", 8192, 0x00);
        let g = b.build().unwrap();
        let stats = build_and_execute(&g).unwrap();
        assert_eq!(stats.bytes_set, 8192);
    }

    #[test]
    fn seq_barrier_counted() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        b.add_barrier("sync");
        let g = b.build().unwrap();
        let stats = build_and_execute(&g).unwrap();
        assert_eq!(stats.barriers, 1);
    }

    #[test]
    fn seq_host_callback_counted() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        b.add_host_callback("checkpoint");
        let g = b.build().unwrap();
        let stats = build_and_execute(&g).unwrap();
        assert_eq!(stats.host_callbacks, 1);
    }

    #[test]
    fn seq_valid_event_record_wait() {
        // Manually build a plan with a record before a wait.
        let steps = vec![
            PlanStep::EventRecord {
                event_id: 0,
                stream: StreamId(0),
            },
            PlanStep::EventWait {
                event_id: 0,
                stream: StreamId(1),
            },
        ];
        let plan = ExecutionPlan {
            steps,
            num_streams: 2,
            pool_bytes: 0,
            kernel_count_original: 0,
            kernel_count_fused: 0,
            event_count: 1,
        };
        let stats = SequentialExecutor::new(&plan).run().unwrap();
        assert_eq!(stats.events_recorded, 1);
        assert_eq!(stats.events_waited, 1);
    }

    #[test]
    fn seq_event_wait_without_record_fails() {
        let steps = vec![PlanStep::EventWait {
            event_id: 99,
            stream: StreamId(0),
        }];
        let plan = ExecutionPlan {
            steps,
            num_streams: 1,
            pool_bytes: 0,
            kernel_count_original: 0,
            kernel_count_fused: 0,
            event_count: 1,
        };
        let result = SequentialExecutor::new(&plan).run();
        assert!(matches!(result, Err(GraphError::InvalidPlan(_))));
    }

    #[test]
    fn seq_validate_ok() {
        let steps = vec![
            PlanStep::EventRecord {
                event_id: 0,
                stream: StreamId(0),
            },
            PlanStep::EventWait {
                event_id: 0,
                stream: StreamId(1),
            },
        ];
        let plan = ExecutionPlan {
            steps,
            num_streams: 2,
            pool_bytes: 0,
            kernel_count_original: 0,
            kernel_count_fused: 0,
            event_count: 1,
        };
        let issues = SequentialExecutor::new(&plan).validate().unwrap();
        assert_eq!(issues, 0);
    }

    #[test]
    fn seq_validate_missing_record_fails() {
        let steps = vec![PlanStep::EventWait {
            event_id: 5,
            stream: StreamId(0),
        }];
        let plan = ExecutionPlan {
            steps,
            num_streams: 1,
            pool_bytes: 0,
            kernel_count_original: 0,
            kernel_count_fused: 0,
            event_count: 1,
        };
        assert!(matches!(
            SequentialExecutor::new(&plan).validate(),
            Err(GraphError::InvalidPlan(_))
        ));
    }

    #[test]
    fn seq_stream0_step_counted() {
        let steps = vec![PlanStep::Barrier {
            node: crate::node::NodeId(0),
            stream: StreamId(0),
        }];
        let plan = ExecutionPlan {
            steps,
            num_streams: 1,
            pool_bytes: 0,
            kernel_count_original: 0,
            kernel_count_fused: 0,
            event_count: 0,
        };
        let stats = SequentialExecutor::new(&plan).run().unwrap();
        assert_eq!(stats.steps_on_stream0, 1);
        assert_eq!(stats.steps_on_other_streams, 0);
    }

    #[test]
    fn seq_other_stream_step_counted() {
        let steps = vec![PlanStep::Barrier {
            node: crate::node::NodeId(0),
            stream: StreamId(2),
        }];
        let plan = ExecutionPlan {
            steps,
            num_streams: 3,
            pool_bytes: 0,
            kernel_count_original: 0,
            kernel_count_fused: 0,
            event_count: 0,
        };
        let stats = SequentialExecutor::new(&plan).run().unwrap();
        assert_eq!(stats.steps_on_other_streams, 1);
    }

    #[test]
    fn seq_multiple_kernels_fused_counted() {
        // Fusible chain → fused to 1 kernel.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let k0 = b.add_kernel("add", 4, 256, 0).fusible(true).finish();
        let k1 = b.add_kernel("relu", 4, 256, 0).fusible(true).finish();
        let k2 = b.add_kernel("scale", 4, 256, 0).fusible(true).finish();
        b.chain(&[k0, k1, k2]);
        let g = b.build().unwrap();
        let plan = ExecutionPlan::build(&g, 1).unwrap();
        let stats = SequentialExecutor::new(&plan).run().unwrap();
        // After fusion: 1 kernel launch.
        assert_eq!(stats.kernels_launched, 1);
    }
}
