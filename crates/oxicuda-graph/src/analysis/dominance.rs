//! Dominator tree computation using Cooper et al.'s algorithm.
//!
//! A node `d` **dominates** node `n` if every path from the graph's entry
//! (virtual root) to `n` passes through `d`. The dominator tree encodes
//! this relationship compactly: `d` is the parent of `n` in the tree iff
//! `d` is the **immediate dominator** of `n` (the closest dominator).
//!
//! # Why dominators matter for GPU graphs
//!
//! * **Operator fusion** — two nodes can be fused only if one dominates the
//!   other *and* there are no side-effecting nodes between them.
//! * **Stream partitioning** — nodes in the same dominator subtree that are
//!   independent of each other are candidates for concurrent execution.
//! * **Memory lifetime shortening** — a buffer's live range can be shortened
//!   to the subtree dominated by its definition.
//!
//! # Algorithm
//!
//! We implement Cooper, Harvey, and Kennedy (2001): "A Simple, Fast Dominance
//! Algorithm". The graph is rooted at a virtual super-source that has edges
//! to all real source nodes, giving a single entry point.
//!
//! 1. Compute a reverse-post-order (RPO) numbering via iterative DFS from
//!    the virtual root.
//! 2. Initialise `idom[entry] = entry`; all others undefined.
//! 3. Repeat until convergence: for each node `b` in RPO order (excluding
//!    entry), set `new_idom` to the intersect of all processed predecessors;
//!    update `idom[b]` if it changed.
//! 4. Strip the virtual root from the output idom array.
//!
//! Time complexity: O(n²) worst-case, O(n) in practice for structured graphs.

use crate::error::{GraphError, GraphResult};
use crate::graph::ComputeGraph;
use crate::node::NodeId;

// ---------------------------------------------------------------------------
// DomTree — the dominator tree
// ---------------------------------------------------------------------------

/// The dominator tree of a computation graph.
///
/// The tree is rooted at a virtual **root node** that is not part of the
/// original graph. Every real node has a unique immediate dominator, except
/// source nodes whose only dominator is the virtual root (represented by
/// `idom(n) == None`).
#[derive(Debug, Clone)]
pub struct DomTree {
    /// Immediate dominator of each node (indexed by NodeId.0).
    /// `None` means the node is dominated only by the virtual root.
    idom: Vec<Option<NodeId>>,
    /// Children in the dominator tree (indexed by NodeId.0 for real nodes).
    children: Vec<Vec<NodeId>>,
    /// Number of real nodes.
    n_nodes: usize,
}

impl DomTree {
    /// Returns the immediate dominator of `node`, or `None` if it is a root.
    #[must_use]
    pub fn idom(&self, node: NodeId) -> Option<NodeId> {
        self.idom.get(node.0 as usize).copied().flatten()
    }

    /// Returns the children of `node` in the dominator tree.
    pub fn children(&self, node: NodeId) -> &[NodeId] {
        self.children
            .get(node.0 as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Returns all root nodes (nodes with no immediate dominator in the real graph).
    pub fn roots(&self) -> Vec<NodeId> {
        (0..self.n_nodes)
            .filter(|&i| self.idom[i].is_none())
            .map(|i| NodeId(i as u32))
            .collect()
    }

    /// Returns `true` if `dominator` dominates `node`.
    ///
    /// A node always dominates itself.
    #[must_use]
    pub fn dominates(&self, dominator: NodeId, node: NodeId) -> bool {
        if dominator == node {
            return true;
        }
        let mut cur = node;
        loop {
            match self.idom(cur) {
                None => return false,
                Some(p) if p == dominator => return true,
                Some(p) => cur = p,
            }
        }
    }

    /// Returns the set of nodes dominated by `root` (its subtree), including `root` itself.
    pub fn dominated_by(&self, root: NodeId) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut stack = vec![root];
        while let Some(n) = stack.pop() {
            result.push(n);
            for &child in self.children(n) {
                stack.push(child);
            }
        }
        result
    }

    /// Returns the depth of `node` in the dominator tree (root = 0).
    #[must_use]
    pub fn depth(&self, node: NodeId) -> usize {
        let mut d = 0usize;
        let mut cur = node;
        loop {
            match self.idom(cur) {
                None => return d,
                Some(p) => {
                    d += 1;
                    cur = p;
                }
            }
        }
    }

    /// Returns the lowest common ancestor of two nodes in the dominator tree.
    ///
    /// If one node dominates the other, it is the LCA.
    #[must_use]
    pub fn lca(&self, a: NodeId, b: NodeId) -> NodeId {
        let da = self.depth(a);
        let db = self.depth(b);
        let mut x = a;
        let mut y = b;
        // Bring the deeper one up to the same level.
        let (shallow, deep, diff) = if da <= db {
            (x, y, db - da)
        } else {
            (y, x, da - db)
        };
        let mut deep = deep;
        for _ in 0..diff {
            deep = self.idom(deep).unwrap_or(deep);
        }
        x = shallow;
        y = deep;
        // Walk up together until they meet.
        let mut guard = self.n_nodes + 1;
        while x != y {
            x = self.idom(x).unwrap_or(x);
            y = self.idom(y).unwrap_or(y);
            if guard == 0 {
                break; // safety: disconnected components
            }
            guard -= 1;
        }
        x
    }
}

// ---------------------------------------------------------------------------
// Cooper's "intersect" helper
// ---------------------------------------------------------------------------

/// Finds the common dominator of two nodes using the idom chain.
///
/// `rpo[v]` is the reverse-post-order position of node `v` (lower = earlier).
/// `idom_raw[v]` = raw immediate dominator index (virtual-root-inclusive).
/// The virtual root is at index `vroot` with `idom_raw[vroot] = vroot`.
fn intersect(mut b1: usize, mut b2: usize, idom_raw: &[usize], rpo: &[usize]) -> usize {
    while b1 != b2 {
        while rpo[b1] > rpo[b2] {
            b1 = idom_raw[b1];
        }
        while rpo[b2] > rpo[b1] {
            b2 = idom_raw[b2];
        }
    }
    b1
}

// ---------------------------------------------------------------------------
// analyse — entry point
// ---------------------------------------------------------------------------

/// Computes the dominator tree of `graph` using Cooper's algorithm.
///
/// A virtual super-source is introduced with edges to all source nodes so
/// the graph has a single entry point. The resulting tree's roots are the
/// real source nodes (those with `idom == None`).
///
/// # Errors
///
/// Returns [`GraphError::EmptyGraph`] if there are no nodes.
pub fn analyse(graph: &ComputeGraph) -> GraphResult<DomTree> {
    if graph.is_empty() {
        return Err(GraphError::EmptyGraph);
    }

    let n_real = graph.node_count();
    let vroot = n_real; // virtual root index (beyond real nodes)
    let total = n_real + 1;

    // Build adjacency lists augmented with the virtual root.
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); total];
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); total];

    for (from, to) in graph.edges() {
        let f = from.0 as usize;
        let t = to.0 as usize;
        succ[f].push(t);
        pred[t].push(f);
    }
    for src in graph.sources() {
        succ[vroot].push(src.0 as usize);
        pred[src.0 as usize].push(vroot);
    }

    // ---- Compute reverse-post-order (RPO) via iterative DFS from vroot ----
    //
    // We push nodes onto `rpo_order` when we finish processing them
    // (post-order), then reverse to get RPO.
    let mut rpo_order: Vec<usize> = Vec::with_capacity(total);
    let mut visited = vec![false; total];
    // Stack: (node, next_successor_index).
    let mut dfs_stack: Vec<(usize, usize)> = vec![(vroot, 0)];
    visited[vroot] = true;

    while let Some((node, idx)) = dfs_stack.last_mut() {
        if *idx < succ[*node].len() {
            let child = succ[*node][*idx];
            *idx += 1;
            if !visited[child] {
                visited[child] = true;
                dfs_stack.push((child, 0));
            }
        } else {
            // All successors processed: emit node in post-order.
            let n = *node;
            dfs_stack.pop();
            rpo_order.push(n);
        }
    }

    // Reverse post-order: entry (vroot) gets rpo=0.
    rpo_order.reverse();

    // rpo[node] = position in RPO (smaller = earlier = closer to entry).
    let mut rpo = vec![usize::MAX; total];
    for (i, &node) in rpo_order.iter().enumerate() {
        rpo[node] = i;
    }

    // ---- Cooper's dominance algorithm ----------------------------------------

    const UNDEF: usize = usize::MAX;
    // idom_raw uses raw indices (including vroot).
    let mut idom_raw = vec![UNDEF; total];
    idom_raw[vroot] = vroot; // virtual root is its own immediate dominator

    let mut changed = true;
    while changed {
        changed = false;
        // Process all nodes in RPO order, skip the virtual root (rpo[vroot]=0).
        for &b in &rpo_order[1..] {
            // Find a "new_idom" that is the intersection of all processed predecessors.
            let mut new_idom = UNDEF;
            for &p in &pred[b] {
                if idom_raw[p] == UNDEF {
                    continue; // predecessor not yet processed
                }
                if new_idom == UNDEF {
                    new_idom = p;
                } else {
                    new_idom = intersect(p, new_idom, &idom_raw, &rpo);
                }
            }
            if new_idom != UNDEF && idom_raw[b] != new_idom {
                idom_raw[b] = new_idom;
                changed = true;
            }
        }
    }

    // ---- Extract real-node idom (virtual root → None) ------------------------
    let mut idom_out: Vec<Option<NodeId>> = vec![None; n_real];
    for i in 0..n_real {
        let raw = idom_raw[i];
        if raw == UNDEF || raw == vroot {
            idom_out[i] = None; // root in the dominator tree
        } else {
            idom_out[i] = Some(NodeId(raw as u32));
        }
    }

    // ---- Build children map ---------------------------------------------------
    let mut children: Vec<Vec<NodeId>> = vec![Vec::new(); n_real];
    for (i, &idom_opt) in idom_out.iter().enumerate() {
        if let Some(parent) = idom_opt {
            children[parent.0 as usize].push(NodeId(i as u32));
        }
    }

    Ok(DomTree {
        idom: idom_out,
        children,
        n_nodes: n_real,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;

    fn make_chain(n: usize) -> (ComputeGraph, Vec<NodeId>) {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let ids: Vec<NodeId> = (0..n).map(|_| b.add_barrier("x")).collect();
        for w in ids.windows(2) {
            b.dep(w[0], w[1]);
        }
        let g = b.build().unwrap();
        (g, ids)
    }

    #[test]
    fn dominance_empty_graph_error() {
        let g = ComputeGraph::new();
        assert!(matches!(analyse(&g), Err(GraphError::EmptyGraph)));
    }

    #[test]
    fn dominance_single_node_is_root() {
        let (g, ids) = make_chain(1);
        let dt = analyse(&g).unwrap();
        assert!(dt.idom(ids[0]).is_none());
        assert_eq!(dt.roots(), vec![ids[0]]);
    }

    #[test]
    fn dominance_linear_chain() {
        // a→b→c→d: each node has the previous as idom.
        let (g, ids) = make_chain(4);
        let dt = analyse(&g).unwrap();
        assert!(dt.idom(ids[0]).is_none()); // source
        assert_eq!(dt.idom(ids[1]), Some(ids[0]));
        assert_eq!(dt.idom(ids[2]), Some(ids[1]));
        assert_eq!(dt.idom(ids[3]), Some(ids[2]));
    }

    #[test]
    fn dominance_diamond() {
        // a→b, a→c, b→d, c→d
        // idom(b)=a, idom(c)=a, idom(d)=a (a is common dominator of d)
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_barrier("a");
        let bnode = b.add_barrier("b");
        let c = b.add_barrier("c");
        let d = b.add_barrier("d");
        b.dep(a, bnode).dep(a, c).dep(bnode, d).dep(c, d);
        let g = b.build().unwrap();
        let dt = analyse(&g).unwrap();
        assert!(dt.idom(a).is_none());
        assert_eq!(dt.idom(bnode), Some(a));
        assert_eq!(dt.idom(c), Some(a));
        // d is reachable from a via two paths (b and c); idom(d) = a.
        assert_eq!(dt.idom(d), Some(a));
    }

    #[test]
    fn dominance_dominates_reflexive() {
        let (g, ids) = make_chain(3);
        let dt = analyse(&g).unwrap();
        assert!(dt.dominates(ids[0], ids[0]));
        assert!(dt.dominates(ids[1], ids[1]));
    }

    #[test]
    fn dominance_dominates_transitive() {
        let (g, ids) = make_chain(4);
        let dt = analyse(&g).unwrap();
        // ids[0] dominates all subsequent nodes in a chain.
        assert!(dt.dominates(ids[0], ids[1]));
        assert!(dt.dominates(ids[0], ids[2]));
        assert!(dt.dominates(ids[0], ids[3]));
        assert!(!dt.dominates(ids[3], ids[0]));
    }

    #[test]
    fn dominance_dominated_by_subtree() {
        let (g, ids) = make_chain(4);
        let dt = analyse(&g).unwrap();
        // ids[1] dominates ids[2] and ids[3] (the rest of the chain).
        let sub = dt.dominated_by(ids[1]);
        assert!(sub.contains(&ids[1]));
        assert!(sub.contains(&ids[2]));
        assert!(sub.contains(&ids[3]));
        assert!(!sub.contains(&ids[0]));
    }

    #[test]
    fn dominance_depth_linear() {
        let (g, ids) = make_chain(4);
        let dt = analyse(&g).unwrap();
        assert_eq!(dt.depth(ids[0]), 0);
        assert_eq!(dt.depth(ids[1]), 1);
        assert_eq!(dt.depth(ids[2]), 2);
        assert_eq!(dt.depth(ids[3]), 3);
    }

    #[test]
    fn dominance_lca_same_node() {
        let (g, ids) = make_chain(3);
        let dt = analyse(&g).unwrap();
        assert_eq!(dt.lca(ids[1], ids[1]), ids[1]);
    }

    #[test]
    fn dominance_lca_diamond() {
        // Diamond: a→b, a→c, b→d, c→d
        // lca(b, c) should be a (they share idom a)
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_barrier("a");
        let bnode = b.add_barrier("b");
        let c = b.add_barrier("c");
        let d = b.add_barrier("d");
        b.dep(a, bnode).dep(a, c).dep(bnode, d).dep(c, d);
        let g = b.build().unwrap();
        let dt = analyse(&g).unwrap();
        // b and c both have idom = a, so lca(b,c) should be a.
        let result = dt.lca(bnode, c);
        // LCA of two siblings in domtree is their common idom = a.
        assert_eq!(result, a);
        // lca of a node with its ancestor:
        assert_eq!(dt.lca(bnode, d), a); // idom(d)=a, so a dominates both
    }

    #[test]
    fn dominance_children() {
        let (g, ids) = make_chain(3);
        let dt = analyse(&g).unwrap();
        // In a linear chain: children of ids[0] = [ids[1]].
        assert_eq!(dt.children(ids[0]), &[ids[1]]);
        assert_eq!(dt.children(ids[1]), &[ids[2]]);
        assert!(dt.children(ids[2]).is_empty());
    }

    #[test]
    fn dominance_fork_join_children() {
        // a→b, a→c, a→d; b→e, c→e, d→e
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_barrier("a");
        let b1 = b.add_barrier("b");
        let c = b.add_barrier("c");
        let d = b.add_barrier("d");
        let e = b.add_barrier("e");
        b.fan_out(a, &[b1, c, d]);
        b.fan_in(&[b1, c, d], e);
        let g = b.build().unwrap();
        let dt = analyse(&g).unwrap();
        // a dominates all; e is dominated by a (convergence point).
        assert!(dt.dominates(a, e));
        // b, c, d each dominated by a.
        assert_eq!(dt.idom(b1), Some(a));
        assert_eq!(dt.idom(c), Some(a));
        assert_eq!(dt.idom(d), Some(a));
    }

    #[test]
    fn dominance_two_independent_nodes() {
        // Two completely isolated nodes — both are roots.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_barrier("a");
        let bnode = b.add_barrier("b");
        let g = b.build().unwrap();
        let dt = analyse(&g).unwrap();
        assert!(dt.idom(a).is_none());
        assert!(dt.idom(bnode).is_none());
        let roots = dt.roots();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn dominance_longer_chain_all_dominated() {
        let (g, ids) = make_chain(8);
        let dt = analyse(&g).unwrap();
        // ids[0] dominates all others.
        for &id in &ids[1..] {
            assert!(dt.dominates(ids[0], id));
        }
        // ids[7] (last) does not dominate any predecessor.
        for &id in &ids[..7] {
            assert!(!dt.dominates(ids[7], id));
        }
    }

    #[test]
    fn dominance_dominated_by_includes_self() {
        let (g, ids) = make_chain(3);
        let dt = analyse(&g).unwrap();
        let sub = dt.dominated_by(ids[0]);
        assert!(sub.contains(&ids[0]));
    }
}
