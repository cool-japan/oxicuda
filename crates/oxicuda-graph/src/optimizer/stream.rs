//! Stream partitioning — assigns nodes to independent CUDA streams.
//!
//! Concurrent GPU execution requires independent work to be submitted on
//! different CUDA streams. This pass analyses the dependency graph and
//! assigns each node a `StreamId` such that:
//!
//! * Dependent nodes (connected by an edge) are on the **same** stream
//!   *or* the dependency is enforced by an event record/wait pair.
//! * Independent subgraphs (no dependency path between them) are placed
//!   on **different** streams.
//! * The number of streams is minimised (up to a user-specified cap).
//!
//! # Algorithm
//!
//! We use a **list-scheduling** heuristic on the topological order, guided
//! by the ASAP/ALAP slack computed in the topo analysis pass:
//!
//! 1. Obtain the topological priority order (zero-slack nodes first).
//! 2. Maintain a set of "available" streams (initially empty).
//! 3. For each node in priority order: if it has predecessors, assign it to
//!    the stream whose last node is a direct predecessor; otherwise open a
//!    new stream (up to `max_streams`). Source nodes always get a new stream.
//! 4. Record cross-stream sync points (pairs of nodes on different streams
//!    where one depends on the other).
//!
//! # Limitations
//!
//! This is a heuristic, not an optimal ILP formulation. For production use,
//! the result can be refined by a second pass that inserts event nodes.

use std::collections::HashMap;

use crate::analysis::topo_analyse;
use crate::error::{GraphError, GraphResult};
use crate::graph::ComputeGraph;
use crate::node::{NodeId, StreamId};

// ---------------------------------------------------------------------------
// StreamAssignment
// ---------------------------------------------------------------------------

/// The stream assignment for a single node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamAssignment {
    pub node: NodeId,
    pub stream: StreamId,
}

// ---------------------------------------------------------------------------
// SyncPoint — cross-stream synchronisation requirement
// ---------------------------------------------------------------------------

/// A pair of nodes on different streams where `from` must complete before `to` begins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncPoint {
    /// The producing node (on `stream_from`).
    pub from: NodeId,
    pub stream_from: StreamId,
    /// The consuming node (on `stream_to`).
    pub to: NodeId,
    pub stream_to: StreamId,
}

// ---------------------------------------------------------------------------
// StreamPlan
// ---------------------------------------------------------------------------

/// The result of stream partitioning.
#[derive(Debug, Clone)]
pub struct StreamPlan {
    /// Per-node stream assignments.
    pub assignments: Vec<StreamAssignment>,
    /// Cross-stream sync points (event record/wait pairs to insert).
    pub sync_points: Vec<SyncPoint>,
    /// Number of distinct streams used.
    pub num_streams: usize,
    /// Maximum number of streams the pass was allowed to create.
    pub max_streams: usize,
}

impl StreamPlan {
    /// Returns the stream assigned to `node`, or `StreamId(0)` if not found.
    pub fn stream_of(&self, node: NodeId) -> StreamId {
        self.assignments
            .iter()
            .find(|a| a.node == node)
            .map(|a| a.stream)
            .unwrap_or(StreamId::DEFAULT)
    }

    /// Returns all nodes assigned to `stream`.
    pub fn nodes_on(&self, stream: StreamId) -> Vec<NodeId> {
        self.assignments
            .iter()
            .filter(|a| a.stream == stream)
            .map(|a| a.node)
            .collect()
    }

    /// Returns `true` if any two nodes have been assigned to different streams.
    pub fn has_concurrency(&self) -> bool {
        let first = self.assignments.first().map(|a| a.stream);
        self.assignments.iter().any(|a| Some(a.stream) != first)
    }

    /// Returns the number of cross-stream synchronisation points.
    pub fn sync_count(&self) -> usize {
        self.sync_points.len()
    }
}

// ---------------------------------------------------------------------------
// analyse — entry point
// ---------------------------------------------------------------------------

/// Runs the stream partitioning pass.
///
/// * `max_streams` — upper bound on the number of streams to create.
///   Passing `1` disables parallelism (all nodes on stream 0).
///   Passing `0` is treated as `1`.
///
/// # Errors
///
/// Returns [`GraphError::EmptyGraph`] if there are no nodes.
/// Returns [`GraphError::StreamPartitioningFailed`] on internal errors.
pub fn analyse(graph: &ComputeGraph, max_streams: usize) -> GraphResult<StreamPlan> {
    if graph.is_empty() {
        return Err(GraphError::EmptyGraph);
    }

    let max_streams = max_streams.max(1);
    let topo = topo_analyse(graph)?;

    // Priority order: zero-slack first (most critical first).
    let priority = topo.priority_order();

    // Stream state: stream_id → last node assigned and its ALAP.
    // We use a simple vec of (last_alap, last_node, stream_id).
    let mut stream_last: Vec<(u64, NodeId, StreamId)> = Vec::new();

    // Per-node stream assignment.
    let mut node_stream: HashMap<NodeId, StreamId> = HashMap::new();

    // Assign streams.
    for &node_id in &priority {
        let preds = graph.predecessors(node_id)?;
        let node_alap = topo.node_info(node_id).map(|i| i.alap).unwrap_or(0);

        if preds.is_empty() {
            // Source node: open a new stream if capacity allows.
            let stream = if stream_last.len() < max_streams {
                let sid = StreamId(stream_last.len() as u32);
                stream_last.push((node_alap, node_id, sid));
                sid
            } else {
                // All stream slots taken: assign to the stream with the
                // lowest current ALAP (most urgent, least loaded stream).
                let best = stream_last
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, (alap, _, _))| *alap)
                    .map(|(i, (_, _, sid))| (i, *sid))
                    .unwrap();
                stream_last[best.0] = (node_alap, node_id, best.1);
                best.1
            };
            node_stream.insert(node_id, stream);
        } else {
            // Node with predecessors: find the stream whose last committed
            // node is a predecessor of `node_id` (or an ancestor via graph
            // reachability). This avoids forcing independent nodes onto the
            // same stream just because they share a common predecessor.
            //
            // Strategy: iterate over streams in order. Pick the first stream
            // whose last node is a predecessor of `node_id`. If none qualify
            // (e.g., all streams have moved on to non-predecessors), open a
            // new stream if capacity allows; otherwise fall back to the
            // stream whose predecessor has the highest ALAP.

            // Collect streams that actually carry a predecessor of node_id.
            let pred_set: std::collections::HashSet<NodeId> = preds.iter().copied().collect();
            let compatible_stream = stream_last
                .iter()
                .find(|(_, last_node, _)| pred_set.contains(last_node))
                .map(|(_, _, sid)| *sid);

            let stream = if let Some(sid) = compatible_stream {
                sid
            } else {
                // No stream has the predecessor as its last node.
                // Check if we can open a new stream.
                if stream_last.len() < max_streams {
                    // Find any predecessor's stream to base ALAP on.
                    let base_stream = preds
                        .iter()
                        .filter_map(|&p| node_stream.get(&p).copied())
                        .next()
                        .unwrap_or(StreamId::DEFAULT);
                    // Only open a new stream if the base stream is actually
                    // occupied (its last node is not our predecessor).
                    let base_last_is_pred = stream_last
                        .iter()
                        .find(|(_, _, sid)| *sid == base_stream)
                        .map(|(_, ln, _)| pred_set.contains(ln))
                        .unwrap_or(false);
                    if !base_last_is_pred {
                        let sid = StreamId(stream_last.len() as u32);
                        stream_last.push((node_alap, node_id, sid));
                        node_stream.insert(node_id, sid);
                        continue;
                    }
                    base_stream
                } else {
                    // All streams taken; use the one carrying our most urgent predecessor.
                    preds
                        .iter()
                        .filter_map(|&p| node_stream.get(&p).copied())
                        .max_by_key(|&s| {
                            stream_last
                                .iter()
                                .find(|(_, _, sid)| *sid == s)
                                .map(|(alap, _, _)| *alap)
                                .unwrap_or(0)
                        })
                        .unwrap_or(StreamId::DEFAULT)
                }
            };

            // Update stream's last ALAP.
            if let Some(entry) = stream_last.iter_mut().find(|(_, _, sid)| *sid == stream) {
                if node_alap > entry.0 {
                    entry.0 = node_alap;
                }
                entry.1 = node_id;
            }
            node_stream.insert(node_id, stream);
        }
    }

    // Collect assignments in topological order.
    let assignments: Vec<StreamAssignment> = topo
        .order
        .iter()
        .map(|&nid| StreamAssignment {
            node: nid,
            stream: node_stream.get(&nid).copied().unwrap_or(StreamId::DEFAULT),
        })
        .collect();

    // Identify cross-stream sync points.
    let mut sync_points: Vec<SyncPoint> = Vec::new();
    for edge in graph.edges() {
        let (from, to) = edge;
        let sf = node_stream.get(&from).copied().unwrap_or(StreamId::DEFAULT);
        let st = node_stream.get(&to).copied().unwrap_or(StreamId::DEFAULT);
        if sf != st {
            sync_points.push(SyncPoint {
                from,
                stream_from: sf,
                to,
                stream_to: st,
            });
        }
    }
    sync_points.sort_by_key(|sp| (sp.from.0, sp.to.0));
    sync_points.dedup_by_key(|sp| (sp.from, sp.to));

    let num_streams = stream_last.len().max(1);

    Ok(StreamPlan {
        assignments,
        sync_points,
        num_streams,
        max_streams,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;

    fn barrier(b: &mut GraphBuilder, name: &str) -> NodeId {
        b.add_barrier(name)
    }

    #[test]
    fn stream_empty_graph() {
        let g = ComputeGraph::new();
        assert!(matches!(analyse(&g, 4), Err(GraphError::EmptyGraph)));
    }

    #[test]
    fn stream_single_node_default_stream() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let n = barrier(&mut b, "n");
        let g = b.build().unwrap();
        let plan = analyse(&g, 4).unwrap();
        assert_eq!(plan.stream_of(n), StreamId(0));
    }

    #[test]
    fn stream_linear_chain_single_stream() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = barrier(&mut b, "a");
        let bnode = barrier(&mut b, "b");
        let c = barrier(&mut b, "c");
        b.chain(&[a, bnode, c]);
        let g = b.build().unwrap();
        let plan = analyse(&g, 4).unwrap();
        // All in a chain → same stream (no concurrency).
        assert!(!plan.has_concurrency());
        assert_eq!(plan.sync_count(), 0);
    }

    #[test]
    fn stream_fork_parallel_branches() {
        // src → a, src → b (a and b are independent)
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let src = barrier(&mut b, "src");
        let a = barrier(&mut b, "a");
        let bnode = barrier(&mut b, "b");
        b.fan_out(src, &[a, bnode]);
        let g = b.build().unwrap();
        let plan = analyse(&g, 4).unwrap();
        // a and b should be on different streams.
        let sa = plan.stream_of(a);
        let sb = plan.stream_of(bnode);
        assert_ne!(
            sa, sb,
            "independent branches should be on different streams"
        );
    }

    #[test]
    fn stream_max_streams_one_disables_parallelism() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = barrier(&mut b, "a");
        let bnode = barrier(&mut b, "b");
        let c = barrier(&mut b, "c");
        // No deps — all independent sources.
        let g = b.build().unwrap();
        let plan = analyse(&g, 1).unwrap();
        // All on stream 0.
        assert_eq!(plan.stream_of(a), StreamId(0));
        assert_eq!(plan.stream_of(bnode), StreamId(0));
        assert_eq!(plan.stream_of(c), StreamId(0));
        assert!(!plan.has_concurrency());
    }

    #[test]
    fn stream_cross_stream_sync_detected() {
        // a → b on stream 0, c (independent of a) → b; if a and c are on
        // different streams, there must be a sync point before b.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = barrier(&mut b, "a");
        let c = barrier(&mut b, "c");
        let bnode = barrier(&mut b, "b");
        b.dep(a, bnode).dep(c, bnode);
        let g = b.build().unwrap();
        let plan = analyse(&g, 4).unwrap();
        let sa = plan.stream_of(a);
        let sc = plan.stream_of(c);
        if sa != sc {
            // There should be sync points for the cross-stream deps.
            assert!(plan.sync_count() > 0);
        }
    }

    #[test]
    fn stream_nodes_on_stream_zero() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = barrier(&mut b, "a");
        let bnode = barrier(&mut b, "b");
        b.dep(a, bnode);
        let g = b.build().unwrap();
        let plan = analyse(&g, 4).unwrap();
        let on_s0 = plan.nodes_on(StreamId(0));
        assert!(!on_s0.is_empty());
    }

    #[test]
    fn stream_assignment_covers_all_nodes() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = barrier(&mut b, "a");
        let bnode = barrier(&mut b, "b");
        let c = barrier(&mut b, "c");
        b.chain(&[a, bnode, c]);
        let g = b.build().unwrap();
        let plan = analyse(&g, 2).unwrap();
        assert_eq!(plan.assignments.len(), 3);
    }

    #[test]
    fn stream_plan_respects_max_streams_cap() {
        // 5 independent nodes with max_streams=3 → at most 3 streams used.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        for i in 0..5 {
            barrier(&mut b, &format!("n{i}"));
        }
        let g = b.build().unwrap();
        let plan = analyse(&g, 3).unwrap();
        assert!(plan.num_streams <= 3);
    }

    #[test]
    fn stream_diamond_join_handled() {
        // a → b, a → c, b → d, c → d
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = barrier(&mut b, "a");
        let bnode = barrier(&mut b, "b");
        let c = barrier(&mut b, "c");
        let d = barrier(&mut b, "d");
        b.dep(a, bnode).dep(a, c).dep(bnode, d).dep(c, d);
        let g = b.build().unwrap();
        let plan = analyse(&g, 4).unwrap();
        // d depends on both b and c; all preds must complete before d.
        // No assertion about stream assignment (heuristic), but must not crash.
        assert_eq!(plan.assignments.len(), 4);
    }

    #[test]
    fn stream_zero_max_treated_as_one() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        barrier(&mut b, "n");
        let g = b.build().unwrap();
        let plan = analyse(&g, 0).unwrap(); // 0 → treated as 1
        assert_eq!(plan.num_streams, 1);
    }
}
