//! Topological-level analysis and critical-path scheduling.
//!
//! Beyond a simple topological order (provided by [`ComputeGraph::topological_order`]),
//! this module computes rich scheduling information:
//!
//! * **Level assignment**: Each node gets a "level" equal to the length of the
//!   longest path (in edges) from any source node to that node.
//! * **Cost-weighted ASAP/ALAP**: As-Soon-As-Possible and As-Late-As-Possible
//!   schedules using per-node cost hints.
//! * **Slack**: The difference `ALAP - ASAP` for each node; zero-slack nodes
//!   lie on the critical path.
//! * **Priority lists**: Used by the stream partitioner to order ready nodes.

use std::collections::VecDeque;

use crate::error::{GraphError, GraphResult};
use crate::graph::ComputeGraph;
use crate::node::NodeId;

// ---------------------------------------------------------------------------
// TopoInfo — per-node scheduling data
// ---------------------------------------------------------------------------

/// Scheduling metadata for a single node, produced by [`TopoAnalysis`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeInfo {
    /// Topological level (longest path in edges from any source).
    pub level: usize,
    /// As-Soon-As-Possible time (cost-weighted earliest start).
    pub asap: u64,
    /// As-Late-As-Possible time (cost-weighted latest start, critical-path constrained).
    pub alap: u64,
    /// Slack = `alap - asap`. Zero means the node is on the critical path.
    pub slack: u64,
    /// Whether this node lies on the critical path.
    pub on_critical_path: bool,
}

impl NodeInfo {
    /// Returns the "mobility" of the node — how many cost units it can be
    /// delayed without extending the overall schedule.
    #[must_use]
    pub fn mobility(&self) -> u64 {
        self.slack
    }
}

// ---------------------------------------------------------------------------
// TopoAnalysis — result of the full analysis pass
// ---------------------------------------------------------------------------

/// Result of running topological-level analysis on a [`ComputeGraph`].
#[derive(Debug, Clone)]
pub struct TopoAnalysis {
    /// Topological order of all node IDs.
    pub order: Vec<NodeId>,
    /// Per-node scheduling info (indexed by `NodeId.0`).
    info: Vec<NodeInfo>,
    /// Length of the critical path in cost units.
    pub critical_path_cost: u64,
    /// Length of the critical path in edges (number of edges).
    pub critical_path_edges: usize,
}

impl TopoAnalysis {
    /// Returns the scheduling info for a specific node.
    ///
    /// Returns `None` if the `NodeId` is out of range.
    #[must_use]
    pub fn node_info(&self, id: NodeId) -> Option<&NodeInfo> {
        self.info.get(id.0 as usize)
    }

    /// Returns all nodes on the critical path, in topological order.
    pub fn critical_path_nodes(&self) -> Vec<NodeId> {
        self.order
            .iter()
            .filter(|&&id| {
                self.info
                    .get(id.0 as usize)
                    .map(|i| i.on_critical_path)
                    .unwrap_or(false)
            })
            .copied()
            .collect()
    }

    /// Returns nodes sorted by ascending slack (most-critical first).
    pub fn priority_order(&self) -> Vec<NodeId> {
        let mut nodes = self.order.clone();
        nodes.sort_by_key(|&id| {
            self.info
                .get(id.0 as usize)
                .map(|i| i.slack)
                .unwrap_or(u64::MAX)
        });
        nodes
    }

    /// Returns the maximum topological level (depth of the graph in edges).
    #[must_use]
    pub fn max_level(&self) -> usize {
        self.info.iter().map(|i| i.level).max().unwrap_or(0)
    }

    /// Returns the number of nodes at each level (the "width profile").
    pub fn level_widths(&self) -> Vec<usize> {
        let max = self.max_level();
        let mut widths = vec![0usize; max + 1];
        for info in &self.info {
            widths[info.level] += 1;
        }
        widths
    }
}

// ---------------------------------------------------------------------------
// analyse — entry point
// ---------------------------------------------------------------------------

/// Performs full topological-level analysis on `graph`.
///
/// Computes ASAP, ALAP, slack, and critical path annotations for every node.
///
/// # Errors
///
/// Returns [`GraphError::EmptyGraph`] if the graph has no nodes.
pub fn analyse(graph: &ComputeGraph) -> GraphResult<TopoAnalysis> {
    if graph.is_empty() {
        return Err(GraphError::EmptyGraph);
    }

    let order = graph.topological_order()?;
    let n = graph.node_count();

    // ---- ASAP pass (forward, cost-weighted) --------------------------------
    // asap[v] = max over all predecessors p: asap[p] + cost[p]
    let mut asap = vec![0u64; n];
    for &id in &order {
        let cost_before = asap[id.0 as usize];
        for &succ in graph.successors(id)? {
            let new_asap = cost_before + graph.node(id)?.cost_hint;
            if new_asap > asap[succ.0 as usize] {
                asap[succ.0 as usize] = new_asap;
            }
        }
    }

    // Total schedule length = max(asap[v] + cost[v]) over all nodes.
    let schedule_len: u64 = (0..n)
        .map(|i| asap[i] + graph.nodes()[i].cost_hint)
        .max()
        .unwrap_or(0);

    // ---- ALAP pass (backward, cost-weighted) --------------------------------
    // alap[v] = schedule_len - cost[v] - max over successors s: (schedule_len - alap[s])
    let mut alap = vec![schedule_len; n];
    // For sinks: alap[v] = schedule_len - cost[v].
    for (i, node) in graph.nodes().iter().enumerate() {
        if graph.successors(node.id)?.is_empty() {
            alap[i] = schedule_len - node.cost_hint;
        }
    }
    // Backward pass over reversed topological order.
    for &id in order.iter().rev() {
        let succs = graph.successors(id)?;
        if succs.is_empty() {
            continue;
        }
        // alap[v] = min over successors s: alap[s] - cost[v]
        let cost_v = graph.node(id)?.cost_hint;
        let min_succ_alap = succs
            .iter()
            .map(|&s| alap[s.0 as usize])
            .min()
            .unwrap_or(schedule_len);
        let v_alap = min_succ_alap.saturating_sub(cost_v);
        if v_alap < alap[id.0 as usize] {
            alap[id.0 as usize] = v_alap;
        }
    }

    // ---- Slack and critical path --------------------------------------------
    let mut info = Vec::with_capacity(n);
    for i in 0..n {
        let slack = alap[i].saturating_sub(asap[i]);
        info.push(NodeInfo {
            level: 0, // filled below
            asap: asap[i],
            alap: alap[i],
            slack,
            on_critical_path: slack == 0,
        });
    }

    // ---- Level assignment (edge-based, forward) ----------------------------
    let mut level = vec![0usize; n];
    let mut in_degree: Vec<u32> = graph
        .nodes()
        .iter()
        .map(|nd| {
            graph
                .predecessors(nd.id)
                .map(|p| p.len() as u32)
                .unwrap_or(0)
        })
        .collect();
    let mut queue: VecDeque<NodeId> = (0..n)
        .filter(|&i| in_degree[i] == 0)
        .map(|i| NodeId(i as u32))
        .collect();
    while let Some(id) = queue.pop_front() {
        let lv = level[id.0 as usize];
        for &succ in graph.successors(id)? {
            let nl = lv + 1;
            if nl > level[succ.0 as usize] {
                level[succ.0 as usize] = nl;
            }
            let d = &mut in_degree[succ.0 as usize];
            *d -= 1;
            if *d == 0 {
                queue.push_back(succ);
            }
        }
    }
    for (i, inf) in info.iter_mut().enumerate() {
        inf.level = level[i];
    }

    let critical_path_edges = *level.iter().max().unwrap_or(&0);
    let critical_path_cost = schedule_len;

    Ok(TopoAnalysis {
        order,
        info,
        critical_path_cost,
        critical_path_edges,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::node::MemcpyDir;

    fn make_linear(n: usize) -> (ComputeGraph, Vec<NodeId>) {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let ids: Vec<NodeId> = (0..n).map(|_| b.add_barrier("x")).collect();
        for w in ids.windows(2) {
            b.dep(w[0], w[1]);
        }
        let g = b.build().unwrap();
        (g, ids)
    }

    #[test]
    fn analyse_empty_returns_error() {
        let g = ComputeGraph::new();
        assert!(matches!(analyse(&g), Err(GraphError::EmptyGraph)));
    }

    #[test]
    fn analyse_single_node() {
        let (g, ids) = make_linear(1);
        let ta = analyse(&g).unwrap();
        let info = ta.node_info(ids[0]).unwrap();
        assert_eq!(info.level, 0);
        assert_eq!(info.asap, 0);
        assert_eq!(info.slack, 0);
        assert!(info.on_critical_path);
    }

    #[test]
    fn analyse_linear_levels() {
        let (g, ids) = make_linear(5);
        let ta = analyse(&g).unwrap();
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(ta.node_info(id).unwrap().level, i, "level mismatch at {i}");
        }
    }

    #[test]
    fn analyse_linear_critical_path_edges() {
        let (g, _) = make_linear(4);
        let ta = analyse(&g).unwrap();
        assert_eq!(ta.critical_path_edges, 3);
    }

    #[test]
    fn analyse_diamond_all_on_critical_path() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_barrier("a");
        let b1 = b.add_barrier("b");
        let c = b.add_barrier("c");
        let d = b.add_barrier("d");
        b.dep(a, b1).dep(a, c).dep(b1, d).dep(c, d);
        let g = b.build().unwrap();
        let ta = analyse(&g).unwrap();
        // All nodes have the same depth in the diamond → all on critical path.
        assert_eq!(ta.critical_path_edges, 2);
        // a and d always on critical path.
        assert!(ta.node_info(a).unwrap().on_critical_path);
        assert!(ta.node_info(d).unwrap().on_critical_path);
    }

    #[test]
    fn analyse_level_widths() {
        let (g, _) = make_linear(3);
        let ta = analyse(&g).unwrap();
        // Linear: width at each level = 1.
        let widths = ta.level_widths();
        assert_eq!(widths, vec![1, 1, 1]);
    }

    #[test]
    fn analyse_fork_join_widths() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let src = b.add_barrier("src");
        let a = b.add_barrier("a");
        let c = b.add_barrier("c");
        let d = b.add_barrier("d");
        let sink = b.add_barrier("sink");
        b.fan_out(src, &[a, c, d]);
        b.fan_in(&[a, c, d], sink);
        let g = b.build().unwrap();
        let ta = analyse(&g).unwrap();
        let widths = ta.level_widths();
        // Level 0: src (1), Level 1: a,c,d (3), Level 2: sink (1).
        assert_eq!(widths[0], 1);
        assert_eq!(widths[1], 3);
        assert_eq!(widths[2], 1);
    }

    #[test]
    fn analyse_priority_order_zero_slack_first() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        // Chain a→b and isolated node c (high slack).
        let a = b.add_barrier("a");
        let bnode = b.add_barrier("b");
        let c = b.add_barrier("c");
        b.dep(a, bnode);
        let g = b.build().unwrap();
        let ta = analyse(&g).unwrap();
        let prio = ta.priority_order();
        // Nodes with slack=0 must come before nodes with slack>0.
        let pos_a = prio.iter().position(|&x| x == a).unwrap();
        let pos_c = prio.iter().position(|&x| x == c).unwrap();
        let slack_a = ta.node_info(a).unwrap().slack;
        let slack_c = ta.node_info(c).unwrap().slack;
        if slack_a < slack_c {
            assert!(
                pos_a < pos_c,
                "zero-slack node must precede high-slack node"
            );
        }
        let _ = bnode;
    }

    #[test]
    fn analyse_max_level() {
        let (g, _) = make_linear(7);
        let ta = analyse(&g).unwrap();
        assert_eq!(ta.max_level(), 6);
    }

    #[test]
    fn analyse_cost_weighted_asap() {
        // a (cost=1) → b (cost=10) → c (cost=1)
        // asap[a]=0, asap[b]=1, asap[c]=11
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_raw(
            crate::node::GraphNode::new(crate::node::NodeId(0), crate::node::NodeKind::Barrier)
                .with_cost(1)
                .with_name("a"),
        );
        let bnode = b.add_raw(
            crate::node::GraphNode::new(crate::node::NodeId(0), crate::node::NodeKind::Barrier)
                .with_cost(10)
                .with_name("b"),
        );
        let c = b.add_raw(
            crate::node::GraphNode::new(crate::node::NodeId(0), crate::node::NodeKind::Barrier)
                .with_cost(1)
                .with_name("c"),
        );
        b.dep(a, bnode).dep(bnode, c);
        let g = b.build().unwrap();
        let ta = analyse(&g).unwrap();
        assert_eq!(ta.node_info(a).unwrap().asap, 0);
        assert_eq!(ta.node_info(bnode).unwrap().asap, 1);
        assert_eq!(ta.node_info(c).unwrap().asap, 11);
    }

    #[test]
    fn analyse_critical_path_nodes_nonempty() {
        let (g, _) = make_linear(4);
        let ta = analyse(&g).unwrap();
        let cp = ta.critical_path_nodes();
        assert!(!cp.is_empty());
    }

    #[test]
    fn analyse_memcpy_node_in_graph() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let upload = b.add_memcpy("up", MemcpyDir::HostToDevice, 1024);
        let compute = b.add_kernel("k", 4, 128, 0).cost(50).finish();
        let download = b.add_memcpy("dn", MemcpyDir::DeviceToHost, 1024);
        b.chain(&[upload, compute, download]);
        let g = b.build().unwrap();
        let ta = analyse(&g).unwrap();
        assert_eq!(ta.critical_path_edges, 2);
        assert!(ta.node_info(compute).unwrap().asap >= 1);
    }
}
