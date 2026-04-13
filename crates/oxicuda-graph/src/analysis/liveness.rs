//! Buffer liveness analysis.
//!
//! For each buffer in a `ComputeGraph`, this pass computes the **live
//! interval** `[def_pos, last_use_pos]` in terms of positions in the
//! topological order. The memory planner uses these intervals to determine
//! which buffers can share the same physical memory allocation.
//!
//! # Algorithm
//!
//! 1. Run a topological sort to get a linear order of nodes.
//! 2. For each node at position `p`:
//!    - For every output buffer `b` it writes: record `def_pos[b] = p`.
//!    - For every input buffer `b` it reads: record `last_use_pos[b] = max(last_use_pos[b], p)`.
//! 3. The live interval for buffer `b` is `[def_pos[b], last_use_pos[b]]`.
//!    Buffers that are only written (no reads) have `last_use_pos = def_pos`.
//!    Buffers that are only read (external inputs) have `def_pos = 0`.
//!
//! Two buffers `a` and `b` **interfere** if their live intervals overlap, i.e.
//! `a.def <= b.last_use && b.def <= a.last_use`.

use std::collections::HashMap;

use crate::error::{GraphError, GraphResult};
use crate::graph::ComputeGraph;
use crate::node::{BufferId, NodeId};

// ---------------------------------------------------------------------------
// LiveInterval
// ---------------------------------------------------------------------------

/// The live interval of a buffer in topological-order position space.
///
/// The buffer is "live" from the step it is first written (`def_pos`) to the
/// step at which it is last read (`last_use_pos`), inclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveInterval {
    /// Buffer this interval belongs to.
    pub buf: BufferId,
    /// Topological position at which the buffer is first defined (written).
    /// `None` means the buffer is an external input (alive from the start).
    pub def_pos: Option<usize>,
    /// Topological position at which the buffer is last used (read).
    /// `None` means the buffer is never read (write-only / dead code).
    pub last_use_pos: Option<usize>,
    /// Buffer size in bytes (copied from `BufferDescriptor`).
    pub size_bytes: usize,
    /// Whether this buffer is externally managed.
    pub external: bool,
}

impl LiveInterval {
    /// Returns the effective start position (0 if never written).
    #[must_use]
    pub fn start(&self) -> usize {
        self.def_pos.unwrap_or(0)
    }

    /// Returns the effective end position (start if never read).
    #[must_use]
    pub fn end(&self) -> usize {
        self.last_use_pos.unwrap_or(self.start())
    }

    /// Returns `true` if this interval overlaps with `other`.
    ///
    /// Two intervals overlap when neither ends strictly before the other begins.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start() <= other.end() && other.start() <= self.end()
    }

    /// Returns `true` if the buffer is dead (never read after being written).
    #[must_use]
    pub fn is_dead(&self) -> bool {
        self.last_use_pos.is_none() && self.def_pos.is_some()
    }

    /// Returns the length of the live interval in steps.
    #[must_use]
    pub fn length(&self) -> usize {
        self.end() - self.start()
    }
}

// ---------------------------------------------------------------------------
// LivenessAnalysis
// ---------------------------------------------------------------------------

/// Result of running liveness analysis on a `ComputeGraph`.
#[derive(Debug, Clone)]
pub struct LivenessAnalysis {
    /// The topological order used for position assignment.
    pub order: Vec<NodeId>,
    /// Live intervals keyed by buffer ID.
    intervals: HashMap<BufferId, LiveInterval>,
}

impl LivenessAnalysis {
    /// Returns the live interval for a buffer, if it was referenced.
    #[must_use]
    pub fn interval(&self, buf: BufferId) -> Option<&LiveInterval> {
        self.intervals.get(&buf)
    }

    /// Returns all live intervals.
    pub fn all_intervals(&self) -> impl Iterator<Item = &LiveInterval> {
        self.intervals.values()
    }

    /// Returns all intervals sorted by start position (ascending).
    pub fn sorted_by_start(&self) -> Vec<&LiveInterval> {
        let mut ivs: Vec<&LiveInterval> = self.intervals.values().collect();
        ivs.sort_by_key(|i| (i.start(), i.buf.0));
        ivs
    }

    /// Returns pairs of buffers whose live intervals overlap (interference set).
    ///
    /// The result is deduplicated: each pair `(a, b)` appears at most once,
    /// with `a.0 < b.0`.
    pub fn interference_pairs(&self) -> Vec<(BufferId, BufferId)> {
        let ivs: Vec<&LiveInterval> = self.intervals.values().collect();
        let mut pairs = Vec::new();
        for i in 0..ivs.len() {
            for j in (i + 1)..ivs.len() {
                if ivs[i].overlaps(ivs[j]) {
                    let a = ivs[i].buf.min(ivs[j].buf);
                    let b = ivs[i].buf.max(ivs[j].buf);
                    pairs.push((a, b));
                }
            }
        }
        pairs.sort();
        pairs.dedup();
        pairs
    }

    /// Returns dead buffers (written but never read).
    pub fn dead_buffers(&self) -> Vec<BufferId> {
        self.intervals
            .values()
            .filter(|i| i.is_dead())
            .map(|i| i.buf)
            .collect()
    }

    /// Returns the maximum number of buffers simultaneously live at any step.
    ///
    /// This is a lower bound on the number of live allocations required.
    pub fn max_live_count(&self) -> usize {
        if self.order.is_empty() {
            return 0;
        }
        let n_steps = self.order.len();
        let mut count_at = vec![0usize; n_steps];
        for iv in self.intervals.values() {
            let range = iv.start()..=iv.end().min(n_steps - 1);
            for cnt in count_at[range].iter_mut() {
                *cnt += 1;
            }
        }
        *count_at.iter().max().unwrap_or(&0)
    }

    /// Returns the maximum total bytes simultaneously live at any step.
    pub fn max_live_bytes(&self) -> usize {
        if self.order.is_empty() {
            return 0;
        }
        let n_steps = self.order.len();
        let mut bytes_at = vec![0usize; n_steps];
        for iv in self.intervals.values() {
            if iv.external {
                continue; // externally managed, not counted in device memory
            }
            let range = iv.start()..=iv.end().min(n_steps - 1);
            let sz = iv.size_bytes;
            for b in bytes_at[range].iter_mut() {
                *b = b.saturating_add(sz);
            }
        }
        *bytes_at.iter().max().unwrap_or(&0)
    }
}

// ---------------------------------------------------------------------------
// analyse — entry point
// ---------------------------------------------------------------------------

/// Computes live intervals for all buffers referenced in `graph`.
///
/// # Errors
///
/// Returns [`GraphError::EmptyGraph`] if the graph has no nodes.
pub fn analyse(graph: &ComputeGraph) -> GraphResult<LivenessAnalysis> {
    if graph.is_empty() {
        return Err(GraphError::EmptyGraph);
    }

    let order = graph.topological_order()?;

    // Map from position in topological order.
    let pos_of: HashMap<NodeId, usize> = order.iter().enumerate().map(|(p, &id)| (id, p)).collect();

    let mut intervals: HashMap<BufferId, LiveInterval> = HashMap::new();

    // Initialise entries for all registered buffers.
    for buf in graph.buffers() {
        intervals.insert(
            buf.id,
            LiveInterval {
                buf: buf.id,
                def_pos: None,
                last_use_pos: None,
                size_bytes: buf.size_bytes,
                external: buf.external,
            },
        );
    }

    // Scan nodes in topological order.
    for &node_id in &order {
        let node = graph.node(node_id)?;
        let p = pos_of[&node_id];

        // Outputs: this node defines (writes) the buffer.
        for &buf in &node.outputs {
            let iv = intervals.entry(buf).or_insert_with(|| LiveInterval {
                buf,
                def_pos: None,
                last_use_pos: None,
                size_bytes: 0,
                external: false,
            });
            // Only record the first definition.
            if iv.def_pos.is_none() {
                iv.def_pos = Some(p);
            }
        }

        // Inputs: this node uses (reads) the buffer; update last-use.
        for &buf in &node.inputs {
            let iv = intervals.entry(buf).or_insert_with(|| LiveInterval {
                buf,
                def_pos: None,
                last_use_pos: None,
                size_bytes: 0,
                external: false,
            });
            match iv.last_use_pos {
                None => iv.last_use_pos = Some(p),
                Some(prev) if p > prev => iv.last_use_pos = Some(p),
                _ => {}
            }
        }
    }

    Ok(LivenessAnalysis { order, intervals })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;

    fn build_linear_graph() -> (ComputeGraph, BufferId, NodeId, NodeId) {
        // writer → reader, sharing buffer `buf`.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let buf = b.alloc_buffer("shared", 1024);
        let writer = b.add_barrier("writer");
        let reader = b.add_barrier("reader");
        b.set_outputs(writer, [buf]);
        b.set_inputs(reader, [buf]);
        b.dep(writer, reader);
        let g = b.build().unwrap();
        (g, buf, writer, reader)
    }

    #[test]
    fn liveness_empty_graph() {
        let g = ComputeGraph::new();
        assert!(matches!(analyse(&g), Err(GraphError::EmptyGraph)));
    }

    #[test]
    fn liveness_buffer_def_and_use() {
        let (g, buf, writer, reader) = build_linear_graph();
        let la = analyse(&g).unwrap();
        let iv = la.interval(buf).unwrap();
        let order = &la.order;
        let wpos = order.iter().position(|&x| x == writer).unwrap();
        let rpos = order.iter().position(|&x| x == reader).unwrap();
        assert_eq!(iv.def_pos, Some(wpos));
        assert_eq!(iv.last_use_pos, Some(rpos));
        assert!(!iv.is_dead());
    }

    #[test]
    fn liveness_dead_buffer() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let buf = b.alloc_buffer("dead", 512);
        let writer = b.add_barrier("w");
        b.set_outputs(writer, [buf]);
        let g = b.build().unwrap();
        let la = analyse(&g).unwrap();
        let iv = la.interval(buf).unwrap();
        assert!(iv.is_dead());
    }

    #[test]
    fn liveness_external_buffer_not_counted_in_bytes() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let ext = b.alloc_external_buffer("ext", 65536);
        let reader = b.add_barrier("r");
        b.set_inputs(reader, [ext]);
        let g = b.build().unwrap();
        let la = analyse(&g).unwrap();
        // External buffer should not count toward live bytes.
        assert_eq!(la.max_live_bytes(), 0);
    }

    #[test]
    fn liveness_overlap_detection() {
        // a writes buf0; b writes buf1; c reads buf0 and buf1.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let buf0 = b.alloc_buffer("b0", 1024);
        let buf1 = b.alloc_buffer("b1", 2048);
        let a = b.add_barrier("a");
        let bnode = b.add_barrier("b");
        let c = b.add_barrier("c");
        b.set_outputs(a, [buf0]);
        b.set_outputs(bnode, [buf1]);
        b.set_inputs(c, [buf0, buf1]);
        b.dep(a, c).dep(bnode, c);
        let g = b.build().unwrap();
        let la = analyse(&g).unwrap();
        let pairs = la.interference_pairs();
        // buf0 and buf1 both live until c, so they interfere.
        assert!(pairs.contains(&(BufferId(0), BufferId(1))));
    }

    #[test]
    fn liveness_non_overlapping_no_interference() {
        // a writes buf0, b reads buf0, b writes buf1, c reads buf1 — chain.
        // buf0 is live [0,1], buf1 is live [1,2] — they share position 1 so they DO overlap.
        // Let's test complete disjoint: buf0 live [0,0], buf1 live [2,2].
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let buf0 = b.alloc_buffer("b0", 512);
        let buf1 = b.alloc_buffer("b1", 512);
        // n0 writes buf0 (live start=0)
        // n1 reads buf0 (live end=1) and writes nothing
        // n2 writes buf1 (live start=2)
        // n3 reads buf1 (live end=3)
        let n0 = b.add_barrier("n0");
        let n1 = b.add_barrier("n1");
        let n2 = b.add_barrier("n2");
        let n3 = b.add_barrier("n3");
        b.set_outputs(n0, [buf0]);
        b.set_inputs(n1, [buf0]);
        b.set_outputs(n2, [buf1]);
        b.set_inputs(n3, [buf1]);
        b.chain(&[n0, n1, n2, n3]);
        let g = b.build().unwrap();
        let la = analyse(&g).unwrap();
        // buf0: [0,1], buf1: [2,3] — no overlap.
        let iv0 = la.interval(buf0).unwrap();
        let iv1 = la.interval(buf1).unwrap();
        assert!(!iv0.overlaps(iv1));
        assert!(la.interference_pairs().is_empty());
    }

    #[test]
    fn liveness_max_live_count() {
        let (g, _buf, _w, _r) = build_linear_graph();
        let la = analyse(&g).unwrap();
        // Only one buffer, so max live = 1.
        assert_eq!(la.max_live_count(), 1);
    }

    #[test]
    fn liveness_max_live_bytes() {
        let (g, _buf, _w, _r) = build_linear_graph();
        let la = analyse(&g).unwrap();
        // One internal buffer of 1024 bytes.
        assert_eq!(la.max_live_bytes(), 1024);
    }

    #[test]
    fn liveness_sorted_by_start() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let buf0 = b.alloc_buffer("b0", 100);
        let buf1 = b.alloc_buffer("b1", 200);
        let n0 = b.add_barrier("n0");
        let n1 = b.add_barrier("n1");
        b.set_outputs(n0, [buf0]);
        b.set_outputs(n1, [buf1]);
        b.dep(n0, n1);
        let g = b.build().unwrap();
        let la = analyse(&g).unwrap();
        let sorted = la.sorted_by_start();
        // buf0 defined at step 0, buf1 at step 1 → buf0 first.
        if sorted.len() == 2 {
            assert!(sorted[0].start() <= sorted[1].start());
        }
    }

    #[test]
    fn liveness_interval_length() {
        let (g, buf, _w, _r) = build_linear_graph();
        let la = analyse(&g).unwrap();
        let iv = la.interval(buf).unwrap();
        // writer at 0, reader at 1 → length = 1.
        assert_eq!(iv.length(), 1);
    }

    #[test]
    fn liveness_dead_buffers_list() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let dead = b.alloc_buffer("dead", 1);
        let _w = {
            let w = b.add_barrier("w");
            b.set_outputs(w, [dead]);
            w
        };
        let g = b.build().unwrap();
        let la = analyse(&g).unwrap();
        let dead_list = la.dead_buffers();
        assert!(dead_list.contains(&dead));
    }

    #[test]
    fn liveness_all_intervals_count() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        b.alloc_buffer("a", 1);
        b.alloc_buffer("b", 2);
        b.alloc_buffer("c", 4);
        b.add_barrier("n");
        let g = b.build().unwrap();
        let la = analyse(&g).unwrap();
        // 3 buffers registered.
        assert_eq!(la.all_intervals().count(), 3);
    }
}
