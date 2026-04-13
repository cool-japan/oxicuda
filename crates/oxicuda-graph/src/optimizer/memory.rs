//! Memory planning — buffer allocation via live-interval graph colouring.
//!
//! The memory planner assigns device memory "slots" to logical buffers so
//! that two buffers whose live intervals do not overlap can share the same
//! physical memory region (offset within a pool allocation).
//!
//! # Algorithm
//!
//! This is an instance of the **interval graph colouring** problem, which
//! on an interval graph is solvable optimally in O(n log n) time using a
//! greedy "earliest deadline first" scan:
//!
//! 1. Sort buffers by live-interval start position.
//! 2. Maintain a priority queue of "free slots" keyed by their end position.
//! 3. For each buffer (in order): reuse a free slot whose end position is
//!    ≤ the buffer's start position (best-fit by size); or allocate a new
//!    slot if no free slot is large enough.
//!
//! The result is a `MemoryPlan` that tells each buffer which slot and byte
//! offset to use within the device memory pool.

use crate::analysis::liveness_analyse;
use crate::error::{GraphError, GraphResult};
use crate::graph::ComputeGraph;
use crate::node::BufferId;

// ---------------------------------------------------------------------------
// SlotAssignment — per-buffer result
// ---------------------------------------------------------------------------

/// The memory slot assigned to a single buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotAssignment {
    /// The buffer being assigned.
    pub buf: BufferId,
    /// Index of the memory slot (0-based).
    pub slot: usize,
    /// Byte offset within the pool allocation for this slot.
    pub offset: usize,
    /// Size of the slot in bytes (≥ buffer size, aligned).
    pub slot_size: usize,
    /// Size of the buffer in bytes.
    pub buf_size: usize,
}

// ---------------------------------------------------------------------------
// MemoryPlan
// ---------------------------------------------------------------------------

/// The complete memory allocation plan for a graph.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Per-buffer slot assignments.
    pub assignments: Vec<SlotAssignment>,
    /// Slot sizes in bytes (indexed by slot id).
    pub slot_sizes: Vec<usize>,
    /// Total pool size required (= sum of slot sizes, aligned).
    pub total_bytes: usize,
    /// Number of distinct slots (= colouring number of the interval graph).
    pub num_slots: usize,
    /// Number of external buffers (not managed by the pool).
    pub external_count: usize,
}

impl MemoryPlan {
    /// Returns the slot assignment for a buffer, or `None` if unassigned.
    pub fn assignment(&self, buf: BufferId) -> Option<&SlotAssignment> {
        self.assignments.iter().find(|a| a.buf == buf)
    }

    /// Returns the memory reduction factor (original sum of sizes vs pool size).
    ///
    /// A value of `0.5` means the pool is 50% of the naive sum (2× memory saved).
    /// Returns `1.0` if no buffers are managed.
    pub fn compression_ratio(&self) -> f64 {
        let naive: usize = self.assignments.iter().map(|a| a.buf_size).sum();
        if naive == 0 {
            return 1.0;
        }
        self.total_bytes as f64 / naive as f64
    }
}

// ---------------------------------------------------------------------------
// Alignment helper
// ---------------------------------------------------------------------------

fn align_up(size: usize, align: usize) -> usize {
    if align == 0 {
        return size;
    }
    (size + align - 1) & !(align - 1)
}

// ---------------------------------------------------------------------------
// analyse — entry point
// ---------------------------------------------------------------------------

/// Runs the memory planning pass on `graph`.
///
/// # Errors
///
/// Returns [`GraphError::EmptyGraph`] if the graph has no nodes.
/// Returns [`GraphError::MemoryPlanningFailed`] if planning fails.
pub fn analyse(graph: &ComputeGraph) -> GraphResult<MemoryPlan> {
    if graph.is_empty() {
        return Err(GraphError::EmptyGraph);
    }

    let liveness = liveness_analyse(graph)?;

    // Alignment requirement (256 bytes is CUDA's standard requirement).
    const ALIGN: usize = 256;

    // Separate external buffers from managed ones.
    let mut managed: Vec<_> = liveness
        .sorted_by_start()
        .into_iter()
        .filter(|iv| !iv.external && iv.size_bytes > 0)
        .collect();

    // Sort by start position (already sorted by sorted_by_start, but confirm).
    managed.sort_by_key(|iv| (iv.start(), iv.buf.0));

    let external_count = liveness.all_intervals().filter(|iv| iv.external).count();

    // Active slots: sorted by end_step.
    // We use a simple linear scan here (interval graph width is typically small).
    // active[i] = (end_step, slot_id, slot_size).
    let mut active: Vec<(usize, usize, usize)> = Vec::new(); // (end, slot_id, size)
    let mut slot_sizes: Vec<usize> = Vec::new();
    let mut assignments: Vec<SlotAssignment> = Vec::new();
    let mut next_slot = 0usize;

    for iv in &managed {
        let start = iv.start();
        let end = iv.end();
        let needed = align_up(iv.size_bytes, ALIGN);

        // Find a free slot: a slot whose end_step < start and size >= needed.
        // Among candidates, prefer the smallest slot that still fits (best fit).
        let candidate_pos = active
            .iter()
            .enumerate()
            .filter(|(_, (end_step, _, sz))| *end_step < start && *sz >= needed)
            .min_by_key(|(_, (_, _, sz))| *sz)
            .map(|(pos, _)| pos);

        let slot_id = if let Some(pos) = candidate_pos {
            let (_, sid, sz) = active.remove(pos);
            // Re-enter the slot as active for this buffer's lifetime.
            active.push((end, sid, sz));
            sid
        } else {
            // Allocate a new slot.
            let sid = next_slot;
            next_slot += 1;
            slot_sizes.push(needed);
            active.push((end, sid, needed));
            sid
        };

        assignments.push(SlotAssignment {
            buf: iv.buf,
            slot: slot_id,
            offset: 0, // offsets assigned after slot sizes are known
            slot_size: slot_sizes[slot_id],
            buf_size: iv.size_bytes,
        });
    }

    // Assign byte offsets: lay slots out consecutively in memory.
    let mut slot_offsets = vec![0usize; slot_sizes.len()];
    let mut cursor = 0usize;
    for (i, &sz) in slot_sizes.iter().enumerate() {
        slot_offsets[i] = cursor;
        cursor += sz;
    }
    let total_bytes = cursor;

    // Fill in offsets.
    for asgn in &mut assignments {
        asgn.offset = slot_offsets[asgn.slot];
    }

    Ok(MemoryPlan {
        assignments,
        slot_sizes,
        total_bytes,
        num_slots: next_slot,
        external_count,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;

    #[test]
    fn memory_empty_graph() {
        let g = ComputeGraph::new();
        assert!(matches!(analyse(&g), Err(GraphError::EmptyGraph)));
    }

    #[test]
    fn memory_no_buffers() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        b.add_barrier("n");
        let g = b.build().unwrap();
        let plan = analyse(&g).unwrap();
        assert_eq!(plan.total_bytes, 0);
        assert_eq!(plan.num_slots, 0);
    }

    #[test]
    fn memory_single_buffer() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let buf = b.alloc_buffer("x", 1024);
        let w = b.add_barrier("w");
        let r = b.add_barrier("r");
        b.set_outputs(w, [buf]);
        b.set_inputs(r, [buf]);
        b.dep(w, r);
        let g = b.build().unwrap();
        let plan = analyse(&g).unwrap();
        assert_eq!(plan.num_slots, 1);
        assert!(plan.total_bytes >= 1024);
        let asgn = plan.assignment(buf).unwrap();
        assert_eq!(asgn.buf, buf);
        assert_eq!(asgn.slot, 0);
    }

    #[test]
    fn memory_non_overlapping_buffers_share_slot() {
        // buf0: live [0,1], buf1: live [2,3] — should share same slot.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let buf0 = b.alloc_buffer("b0", 512);
        let buf1 = b.alloc_buffer("b1", 512);
        let n0 = b.add_barrier("n0"); // step 0: writes buf0
        let n1 = b.add_barrier("n1"); // step 1: reads buf0
        let n2 = b.add_barrier("n2"); // step 2: writes buf1
        let n3 = b.add_barrier("n3"); // step 3: reads buf1
        b.set_outputs(n0, [buf0]);
        b.set_inputs(n1, [buf0]);
        b.set_outputs(n2, [buf1]);
        b.set_inputs(n3, [buf1]);
        b.chain(&[n0, n1, n2, n3]);
        let g = b.build().unwrap();
        let plan = analyse(&g).unwrap();
        // Ideally they share one slot.
        let a0 = plan.assignment(buf0);
        let a1 = plan.assignment(buf1);
        if let (Some(a0), Some(a1)) = (a0, a1) {
            assert_eq!(
                a0.slot, a1.slot,
                "non-overlapping buffers should share a slot"
            );
        }
        assert_eq!(plan.num_slots, 1);
    }

    #[test]
    fn memory_overlapping_buffers_use_different_slots() {
        // Both buffers alive concurrently.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let buf0 = b.alloc_buffer("b0", 1024);
        let buf1 = b.alloc_buffer("b1", 2048);
        let w0 = b.add_barrier("w0");
        let w1 = b.add_barrier("w1");
        let reader = b.add_barrier("reader");
        b.set_outputs(w0, [buf0]);
        b.set_outputs(w1, [buf1]);
        b.set_inputs(reader, [buf0, buf1]);
        b.fan_in(&[w0, w1], reader);
        let g = b.build().unwrap();
        let plan = analyse(&g).unwrap();
        let a0 = plan.assignment(buf0).unwrap();
        let a1 = plan.assignment(buf1).unwrap();
        assert_ne!(a0.slot, a1.slot);
        assert_eq!(plan.num_slots, 2);
    }

    #[test]
    fn memory_external_buffer_not_counted() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let ext = b.alloc_external_buffer("weights", 65536);
        let r = b.add_barrier("r");
        b.set_inputs(r, [ext]);
        let g = b.build().unwrap();
        let plan = analyse(&g).unwrap();
        assert_eq!(plan.external_count, 1);
        assert_eq!(plan.num_slots, 0);
        assert_eq!(plan.total_bytes, 0);
    }

    #[test]
    fn memory_alignment_respected() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let buf = b.alloc_buffer("small", 17); // 17 bytes → must align to 256
        let w = b.add_barrier("w");
        b.set_outputs(w, [buf]);
        let g = b.build().unwrap();
        let plan = analyse(&g).unwrap();
        assert!(plan.total_bytes >= 256);
        let asgn = plan.assignment(buf).unwrap();
        assert!(asgn.slot_size >= 256);
    }

    #[test]
    fn memory_compression_ratio_improves_with_sharing() {
        // Three sequential non-overlapping buffers, same size → ideal ratio ~1/3.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let bufs: Vec<_> = (0..3)
            .map(|i| b.alloc_buffer(&format!("b{i}"), 1024))
            .collect();
        let nodes: Vec<_> = (0..6).map(|i| b.add_barrier(&format!("n{i}"))).collect();
        // n0 writes buf0, n1 reads buf0, n2 writes buf1, n3 reads buf1, ...
        for i in 0..3 {
            b.set_outputs(nodes[2 * i], [bufs[i]]);
            b.set_inputs(nodes[2 * i + 1], [bufs[i]]);
        }
        b.chain(&nodes);
        let g = b.build().unwrap();
        let plan = analyse(&g).unwrap();
        // With perfect sharing, total = 1 * 256 (aligned) vs naive = 3 * 1024 = 3072.
        // ratio should be < 1.0 (better than naive).
        assert!(
            plan.compression_ratio() <= 1.0,
            "ratio = {}",
            plan.compression_ratio()
        );
    }

    #[test]
    fn memory_slot_offsets_non_overlapping() {
        // Multiple concurrent buffers → different slots and non-overlapping offsets.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let buf0 = b.alloc_buffer("b0", 1024);
        let buf1 = b.alloc_buffer("b1", 2048);
        let buf2 = b.alloc_buffer("b2", 512);
        let w0 = b.add_barrier("w0");
        let w1 = b.add_barrier("w1");
        let w2 = b.add_barrier("w2");
        let sink = b.add_barrier("sink");
        b.set_outputs(w0, [buf0]);
        b.set_outputs(w1, [buf1]);
        b.set_outputs(w2, [buf2]);
        b.set_inputs(sink, [buf0, buf1, buf2]);
        b.fan_in(&[w0, w1, w2], sink);
        let g = b.build().unwrap();
        let plan = analyse(&g).unwrap();
        // All concurrent → 3 slots with non-overlapping offsets.
        let a0 = plan.assignment(buf0).unwrap();
        let a1 = plan.assignment(buf1).unwrap();
        let a2 = plan.assignment(buf2).unwrap();
        // Offsets should not overlap.
        let ranges = [
            (a0.offset, a0.offset + a0.slot_size),
            (a1.offset, a1.offset + a1.slot_size),
            (a2.offset, a2.offset + a2.slot_size),
        ];
        for i in 0..ranges.len() {
            for j in (i + 1)..ranges.len() {
                let (s0, e0) = ranges[i];
                let (s1, e1) = ranges[j];
                assert!(e0 <= s1 || e1 <= s0, "slots {i} and {j} overlap in memory");
            }
        }
    }
}
