//! Instruction scheduling for PTX basic blocks.
//!
//! This module implements a **list-scheduling** algorithm that reorders
//! instructions within basic blocks to hide latency by scheduling independent
//! instructions between dependent ones.
//!
//! The scheduler is conservative: it never reorders instructions across basic
//! block boundaries (labels, branches, barriers), and respects all data
//! dependencies (RAW, WAR, WAW) as well as memory ordering constraints.

use std::collections::{HashMap, HashSet};

use crate::ir::{Instruction, MemorySpace, Operand, Register, WmmaOp};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Estimated latency for an instruction (in GPU clock cycles).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstructionLatency {
    /// Execution latency.
    pub execute: u32,
    /// Memory latency (0 for non-memory ops).
    pub memory: u32,
}

impl InstructionLatency {
    /// Total latency (execute + memory).
    const fn total(self) -> u32 {
        self.execute + self.memory
    }
}

/// Scheduling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingStrategy {
    /// Maximize instruction-level parallelism (default).
    MaxIlp,
    /// Minimize register pressure (for high-register-pressure kernels).
    MinRegPressure,
}

/// Result of instruction scheduling.
#[derive(Debug)]
pub struct SchedulingReport {
    /// Original instruction count.
    pub original_count: usize,
    /// Number of instructions reordered.
    pub instructions_moved: usize,
    /// Estimated stall cycles eliminated.
    pub stalls_eliminated: u32,
    /// Critical path length (in cycles) before scheduling.
    pub critical_path_before: u32,
    /// Critical path length after scheduling.
    pub critical_path_after: u32,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Schedule instructions within basic blocks to hide latency.
///
/// This uses a list-scheduling algorithm that:
/// 1. Builds a dependency DAG for each basic block
/// 2. Assigns priorities based on critical path length
/// 3. Greedily schedules ready instructions by priority
///
/// Instructions are NOT reordered across basic block boundaries
/// (labels, branches, barriers act as scheduling barriers).
pub fn schedule_instructions(
    instructions: &[Instruction],
    strategy: SchedulingStrategy,
) -> (Vec<Instruction>, SchedulingReport) {
    if instructions.is_empty() {
        return (
            Vec::new(),
            SchedulingReport {
                original_count: 0,
                instructions_moved: 0,
                stalls_eliminated: 0,
                critical_path_before: 0,
                critical_path_after: 0,
            },
        );
    }

    let blocks = split_basic_blocks(instructions);
    let mut result = Vec::with_capacity(instructions.len());
    let mut total_moved: usize = 0;
    let mut total_stalls_eliminated: u32 = 0;
    let mut total_cp_before: u32 = 0;
    let mut total_cp_after: u32 = 0;

    for block in &blocks {
        let (scheduled, report) = schedule_block(block, strategy);
        total_moved += report.instructions_moved;
        total_stalls_eliminated += report.stalls_eliminated;
        total_cp_before += report.critical_path_before;
        total_cp_after += report.critical_path_after;
        result.extend(scheduled);
    }

    let report = SchedulingReport {
        original_count: instructions.len(),
        instructions_moved: total_moved,
        stalls_eliminated: total_stalls_eliminated,
        critical_path_before: total_cp_before,
        critical_path_after: total_cp_after,
    };
    (result, report)
}

// ---------------------------------------------------------------------------
// Basic block splitting
// ---------------------------------------------------------------------------

/// Returns `true` if the instruction is a scheduling barrier (block boundary).
const fn is_scheduling_barrier(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Label(_)
            | Instruction::Branch { .. }
            | Instruction::Return
            | Instruction::BarSync { .. }
            | Instruction::BarArrive { .. }
            | Instruction::FenceAcqRel { .. }
    )
}

/// Split the instruction stream into basic blocks.
///
/// Each barrier instruction forms a single-element block by itself. Consecutive
/// non-barrier instructions form a schedulable block.
fn split_basic_blocks(instructions: &[Instruction]) -> Vec<Vec<Instruction>> {
    let mut blocks: Vec<Vec<Instruction>> = Vec::new();
    let mut current_block: Vec<Instruction> = Vec::new();

    for inst in instructions {
        if is_scheduling_barrier(inst) {
            // Flush the current block (if any) before the barrier.
            if !current_block.is_empty() {
                blocks.push(std::mem::take(&mut current_block));
            }
            // The barrier itself is its own block (it won't be reordered).
            blocks.push(vec![inst.clone()]);
        } else {
            current_block.push(inst.clone());
        }
    }
    // Flush trailing instructions.
    if !current_block.is_empty() {
        blocks.push(current_block);
    }
    blocks
}

// ---------------------------------------------------------------------------
// Dependency DAG
// ---------------------------------------------------------------------------

/// Edge in the dependency DAG: (`predecessor_index`, `latency_weight`).
type DepEdge = (usize, u32);

/// Dependency DAG for a basic block.
struct DependencyDag {
    /// Number of nodes (== number of instructions in the block).
    len: usize,
    /// Predecessors: `predecessors[i]` = list of (`pred_idx`, latency).
    /// Instruction i cannot be scheduled until all predecessors complete.
    predecessors: Vec<Vec<DepEdge>>,
    /// Successors: `successors[i]` = list of (`succ_idx`, latency).
    successors: Vec<Vec<DepEdge>>,
}

/// Build the dependency DAG for a basic block.
#[allow(clippy::too_many_lines)]
fn build_dependency_dag(block: &[Instruction]) -> DependencyDag {
    let n = block.len();
    let mut preds: Vec<Vec<DepEdge>> = vec![Vec::new(); n];
    let mut succs: Vec<Vec<DepEdge>> = vec![Vec::new(); n];

    // Pre-compute defs, uses, latencies, side-effect flags, and memory info.
    let defs_vec: Vec<Vec<String>> = block
        .iter()
        .map(|inst| defs(inst).into_iter().map(|r| r.name.clone()).collect())
        .collect();
    let uses_vec: Vec<Vec<String>> = block
        .iter()
        .map(|inst| uses(inst).into_iter().map(|r| r.name.clone()).collect())
        .collect();
    let latencies: Vec<InstructionLatency> = block.iter().map(estimate_latency).collect();
    let side_effects: Vec<bool> = block.iter().map(has_side_effects).collect();
    let is_mem_read: Vec<bool> = block.iter().map(is_memory_read).collect();
    let is_mem_write: Vec<bool> = block.iter().map(is_memory_write).collect();

    // Track the last writer for each register (for RAW and WAW).
    let mut last_writer: HashMap<String, usize> = HashMap::new();
    // Track readers since the last write for each register (for WAR).
    let mut readers_since_write: HashMap<String, Vec<usize>> = HashMap::new();
    // Track last memory writing instruction (conservative: all alias).
    let mut last_mem_write: Option<usize> = None;
    // Track last memory reading instruction (conservative: all alias).
    let mut last_mem_read: Option<usize> = None;
    // Track last side-effect instruction.
    let mut last_side_effect: Option<usize> = None;

    let add_edge = |pred_list: &mut Vec<Vec<DepEdge>>,
                    succ_list: &mut Vec<Vec<DepEdge>>,
                    from: usize,
                    to: usize,
                    lat: u32| {
        // Avoid duplicate edges: keep the maximum latency.
        if let Some(existing) = pred_list[to].iter_mut().find(|(src, _)| *src == from) {
            if lat > existing.1 {
                existing.1 = lat;
            }
            // Also update the successor entry.
            if let Some(s) = succ_list[from].iter_mut().find(|(d, _)| *d == to) {
                if lat > s.1 {
                    s.1 = lat;
                }
            }
            return;
        }
        pred_list[to].push((from, lat));
        succ_list[from].push((to, lat));
    };

    for (i, _lat_i) in latencies
        .iter()
        .enumerate()
        .map(|(idx, l)| (idx, l.total()))
    {
        // RAW: for each register this instruction reads, if a prior instruction
        // wrote it, then we depend on that writer.
        for reg in &uses_vec[i] {
            if let Some(&writer) = last_writer.get(reg) {
                let dep_lat = latencies[writer].total();
                add_edge(&mut preds, &mut succs, writer, i, dep_lat);
            }
        }

        // WAW: for each register this instruction writes, if a prior instruction
        // also wrote it, we must come after.
        for reg in &defs_vec[i] {
            if let Some(&prev_writer) = last_writer.get(reg) {
                add_edge(&mut preds, &mut succs, prev_writer, i, 1);
            }
        }

        // WAR: for each register this instruction writes, all prior readers
        // (since the last write) must complete before we overwrite.
        for reg in &defs_vec[i] {
            if let Some(readers) = readers_since_write.get(reg) {
                for &reader in readers {
                    if reader != i {
                        add_edge(&mut preds, &mut succs, reader, i, 1);
                    }
                }
            }
        }

        // Memory ordering (conservative: all memory ops may alias).
        if is_mem_read[i] {
            // Read must wait for prior write.
            if let Some(prev_w) = last_mem_write {
                let dep_lat = latencies[prev_w].total();
                add_edge(&mut preds, &mut succs, prev_w, i, dep_lat);
            }
        }
        if is_mem_write[i] {
            // Write must wait for prior read and prior write.
            if let Some(prev_r) = last_mem_read {
                add_edge(&mut preds, &mut succs, prev_r, i, 1);
            }
            if let Some(prev_w) = last_mem_write {
                add_edge(&mut preds, &mut succs, prev_w, i, 1);
            }
        }

        // Side-effect ordering: side-effect instructions maintain relative order.
        if side_effects[i] {
            if let Some(prev_se) = last_side_effect {
                add_edge(&mut preds, &mut succs, prev_se, i, 1);
            }
            last_side_effect = Some(i);
        }

        // Update tracking maps.
        for reg in &defs_vec[i] {
            // Clear readers since we're the new writer.
            readers_since_write.remove(reg);
            last_writer.insert(reg.clone(), i);
        }
        for reg in &uses_vec[i] {
            readers_since_write.entry(reg.clone()).or_default().push(i);
        }
        if is_mem_read[i] {
            last_mem_read = Some(i);
        }
        if is_mem_write[i] {
            last_mem_write = Some(i);
        }
    }

    DependencyDag {
        len: n,
        predecessors: preds,
        successors: succs,
    }
}

// ---------------------------------------------------------------------------
// Priority calculation (critical path)
// ---------------------------------------------------------------------------

/// Compute the critical path length from each node to any sink node.
///
/// Uses reverse topological order traversal. Returns priority per node.
fn compute_priorities(dag: &DependencyDag, latencies: &[InstructionLatency]) -> Vec<u32> {
    let n = dag.len;
    let mut priority = vec![0u32; n];

    // Initialize each node's priority to its own latency.
    for i in 0..n {
        priority[i] = latencies[i].total();
    }

    // Process in reverse order. For each node, the priority is:
    //   priority[i] = latency[i] + max over successors j of (edge_latency(i,j) + priority[j] - latency[i])
    // Simplified: priority[i] = max over successors j of (edge_weight(i,j) + priority[j])
    // But we also include the node's own latency as the minimum.
    //
    // We iterate until fixed point since the DAG is acyclic by construction.
    // A reverse-topological sweep suffices. Since nodes are added in
    // program order and edges go from lower to higher indices, we can
    // simply sweep from n-1 down to 0.
    for i in (0..n).rev() {
        let mut max_path = 0u32;
        for &(succ, edge_lat) in &dag.successors[i] {
            let candidate = edge_lat.saturating_add(priority[succ]);
            if candidate > max_path {
                max_path = candidate;
            }
        }
        // The node's priority is at least its own latency.
        priority[i] = latencies[i].total().max(max_path);
    }

    priority
}

/// Compute the critical path of the entire DAG (maximum priority among all nodes).
fn critical_path_length(priorities: &[u32]) -> u32 {
    priorities.iter().copied().max().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// List scheduling
// ---------------------------------------------------------------------------

/// Schedule one basic block using list scheduling.
fn schedule_block(
    block: &[Instruction],
    strategy: SchedulingStrategy,
) -> (Vec<Instruction>, SchedulingReport) {
    let n = block.len();

    // Single or zero instruction blocks: nothing to schedule.
    if n <= 1 {
        return (
            block.to_vec(),
            SchedulingReport {
                original_count: n,
                instructions_moved: 0,
                stalls_eliminated: 0,
                critical_path_before: block.first().map_or(0, |i| estimate_latency(i).total()),
                critical_path_after: block.first().map_or(0, |i| estimate_latency(i).total()),
            },
        );
    }

    // If the block is a single barrier, just pass it through.
    if n == 1 && is_scheduling_barrier(&block[0]) {
        return (
            block.to_vec(),
            SchedulingReport {
                original_count: 1,
                instructions_moved: 0,
                stalls_eliminated: 0,
                critical_path_before: 1,
                critical_path_after: 1,
            },
        );
    }

    let dag = build_dependency_dag(block);
    let latencies: Vec<InstructionLatency> = block.iter().map(estimate_latency).collect();
    let priorities = compute_priorities(&dag, &latencies);
    let cp_before = critical_path_length(&priorities);

    // Count how many predecessors each node has (in-degree).
    let mut in_degree: Vec<usize> = dag.predecessors.iter().map(Vec::len).collect();

    // Compute defs count for MinRegPressure: prefer instructions that define
    // fewer registers and whose uses are about to expire.
    let def_counts: Vec<usize> = block.iter().map(|inst| defs(inst).len()).collect();
    let use_counts: Vec<usize> = block.iter().map(|inst| uses(inst).len()).collect();

    // Predecessor counts (total, not in-degree) for tie-breaking in MaxIlp:
    // instructions with fewer total predecessors are more independent.
    let pred_counts: Vec<usize> = dag.predecessors.iter().map(Vec::len).collect();

    // Ready set: instructions with all dependencies satisfied.
    let mut ready: Vec<usize> = Vec::new();
    for (i, &deg) in in_degree.iter().enumerate().take(n) {
        if deg == 0 {
            ready.push(i);
        }
    }

    let mut scheduled_order: Vec<usize> = Vec::with_capacity(n);
    let mut scheduled_set: HashSet<usize> = HashSet::with_capacity(n);

    // Greedy scheduling loop.
    while !ready.is_empty() {
        // Pick the highest-priority ready instruction.
        let best_idx = select_best(
            &ready,
            &priorities,
            &def_counts,
            &use_counts,
            &pred_counts,
            strategy,
        );
        let chosen = ready.swap_remove(best_idx);

        scheduled_order.push(chosen);
        scheduled_set.insert(chosen);

        // Update successors' in-degrees.
        for &(succ, _) in &dag.successors[chosen] {
            if !scheduled_set.contains(&succ) {
                in_degree[succ] = in_degree[succ].saturating_sub(1);
                if in_degree[succ] == 0 {
                    ready.push(succ);
                }
            }
        }
    }

    // If some instructions were not scheduled (shouldn't happen with a
    // correct DAG), append them in original order for safety.
    if scheduled_order.len() < n {
        for i in 0..n {
            if !scheduled_set.contains(&i) {
                scheduled_order.push(i);
            }
        }
    }

    // Build the output.
    let scheduled_insts: Vec<Instruction> =
        scheduled_order.iter().map(|&i| block[i].clone()).collect();

    // Count moved instructions: compare with original order.
    let moved = count_moved(&scheduled_order);

    // Estimate stalls eliminated by comparing adjacent latency gaps.
    let stalls_before = estimate_stalls(block, &(0..n).collect::<Vec<_>>(), &dag, &latencies);
    let stalls_after = estimate_stalls(block, &scheduled_order, &dag, &latencies);
    let stalls_eliminated = stalls_before.saturating_sub(stalls_after);

    // Recompute critical path after scheduling (same DAG, same priorities).
    let cp_after = cp_before; // Critical path doesn't change with reordering.

    let report = SchedulingReport {
        original_count: n,
        instructions_moved: moved,
        stalls_eliminated,
        critical_path_before: cp_before,
        critical_path_after: cp_after,
    };
    (scheduled_insts, report)
}

/// Select the best instruction from the ready set.
///
/// Returns the *index into the ready vector* (not the instruction index).
fn select_best(
    ready: &[usize],
    priorities: &[u32],
    def_counts: &[usize],
    use_counts: &[usize],
    pred_counts: &[usize],
    strategy: SchedulingStrategy,
) -> usize {
    let mut best = 0usize;

    for i in 1..ready.len() {
        let should_prefer = match strategy {
            SchedulingStrategy::MaxIlp => {
                // Prefer higher priority (longer critical path).
                // Break ties by fewer predecessors (more independent = better
                // for filling latency gaps), then by original order.
                let prio_new = priorities[ready[i]];
                let prio_best = priorities[ready[best]];
                if prio_new == prio_best {
                    let pcount_new = pred_counts[ready[i]];
                    let pcount_best = pred_counts[ready[best]];
                    if pcount_new == pcount_best {
                        ready[i] < ready[best]
                    } else {
                        pcount_new < pcount_best
                    }
                } else {
                    prio_new > prio_best
                }
            }
            SchedulingStrategy::MinRegPressure => {
                // Prefer instructions that consume registers (high use count)
                // and produce fewer registers (low def count).
                // This helps release registers sooner.
                #[allow(clippy::cast_possible_wrap)]
                let score_new = use_counts[ready[i]] as isize - def_counts[ready[i]] as isize;
                #[allow(clippy::cast_possible_wrap)]
                let score_best =
                    use_counts[ready[best]] as isize - def_counts[ready[best]] as isize;
                score_new > score_best
                    || (score_new == score_best && priorities[ready[i]] > priorities[ready[best]])
                    || (score_new == score_best
                        && priorities[ready[i]] == priorities[ready[best]]
                        && ready[i] < ready[best])
            }
        };
        if should_prefer {
            best = i;
        }
    }
    best
}

/// Count how many instructions are not in their original position.
fn count_moved(order: &[usize]) -> usize {
    order
        .iter()
        .enumerate()
        .filter(|&(pos, orig)| pos != *orig)
        .count()
}

/// Estimate total stall cycles for a given scheduling order.
///
/// For each instruction in the order, if it depends on a predecessor that
/// was scheduled recently (within its latency window), the difference is
/// counted as a stall.
fn estimate_stalls(
    _block: &[Instruction],
    order: &[usize],
    dag: &DependencyDag,
    latencies: &[InstructionLatency],
) -> u32 {
    let n = order.len();
    // Map from instruction index to its scheduled position.
    let mut position: Vec<usize> = vec![0; n];
    for (pos, &inst_idx) in order.iter().enumerate() {
        if inst_idx < n {
            position[inst_idx] = pos;
        }
    }

    let mut total_stalls: u32 = 0;
    for (pos, &inst_idx) in order.iter().enumerate() {
        if inst_idx >= dag.len {
            continue;
        }
        for &(pred_idx, edge_lat) in &dag.predecessors[inst_idx] {
            if pred_idx >= n {
                continue;
            }
            let pred_pos = position[pred_idx];
            // The number of intervening instructions between pred and this.
            let gap = pos.saturating_sub(pred_pos);
            let lat = latencies[pred_idx].total().max(edge_lat);
            let gap_u32 = u32::try_from(gap).unwrap_or(u32::MAX);
            total_stalls += lat.saturating_sub(gap_u32);
        }
    }
    total_stalls
}

// ---------------------------------------------------------------------------
// Latency model
// ---------------------------------------------------------------------------

/// Estimate the execution latency of a PTX instruction.
#[allow(clippy::too_many_lines)]
const fn estimate_latency(inst: &Instruction) -> InstructionLatency {
    match inst {
        // Arithmetic, stores, compare/conversion, reductions, bit manipulation: 4 cycles
        Instruction::Add { .. }
        | Instruction::Sub { .. }
        | Instruction::Min { .. }
        | Instruction::Max { .. }
        | Instruction::Neg { .. }
        | Instruction::Abs { .. }
        | Instruction::Shl { .. }
        | Instruction::Shr { .. }
        | Instruction::And { .. }
        | Instruction::Or { .. }
        | Instruction::Xor { .. }
        | Instruction::Store { .. }
        | Instruction::SetP { .. }
        | Instruction::Cvt { .. }
        | Instruction::Red { .. }
        | Instruction::Brev { .. }
        | Instruction::Clz { .. }
        | Instruction::Popc { .. }
        | Instruction::Bfind { .. }
        | Instruction::Bfe { .. }
        | Instruction::Bfi { .. }
        // Dot-product accumulate: 4-cycle arithmetic
        | Instruction::Dp4a { .. }
        | Instruction::Dp2a { .. } => InstructionLatency {
            execute: 4,
            memory: 0,
        },

        // Multiply/FMA: 8 cycles
        Instruction::Mul { .. }
        | Instruction::Mad { .. }
        | Instruction::MadLo { .. }
        | Instruction::MadHi { .. }
        | Instruction::MadWide { .. }
        | Instruction::Fma { .. }
        | Instruction::Div { .. }
        | Instruction::Rem { .. } => InstructionLatency {
            execute: 8,
            memory: 0,
        },

        // Special math: 16 cycles
        Instruction::Rcp { .. } | Instruction::Rsqrt { .. } | Instruction::Sqrt { .. } => {
            InstructionLatency {
                execute: 16,
                memory: 0,
            }
        }

        // Transcendentals: 24 cycles
        Instruction::Ex2 { .. }
        | Instruction::Lg2 { .. }
        | Instruction::Sin { .. }
        | Instruction::Cos { .. } => InstructionLatency {
            execute: 24,
            memory: 0,
        },

        // Global memory loads, global atomics, TMA, bulk async copy: high latency (200 cycles)
        Instruction::Load {
            space: MemorySpace::Global,
            ..
        }
        | Instruction::Atom {
            space: MemorySpace::Global,
            ..
        }
        | Instruction::AtomCas {
            space: MemorySpace::Global,
            ..
        }
        | Instruction::TmaLoad { .. }
        | Instruction::CpAsyncBulk { .. } => InstructionLatency {
            execute: 4,
            memory: 200,
        },

        // Shared memory loads, shared atomics, and ldmatrix: moderate latency (20 cycles)
        // ldmatrix is a warp-cooperative shared memory load — same latency class
        Instruction::Load {
            space: MemorySpace::Shared,
            ..
        }
        | Instruction::Atom { .. }
        | Instruction::AtomCas { .. }
        | Instruction::Ldmatrix { .. } => InstructionLatency {
            execute: 4,
            memory: 20,
        },

        // Other loads: moderate latency
        Instruction::Load { .. } => InstructionLatency {
            execute: 4,
            memory: 50,
        },

        // Texture/surface operations: very high latency (~400 cycles)
        Instruction::Tex1d { .. }
        | Instruction::Tex2d { .. }
        | Instruction::Tex3d { .. }
        | Instruction::SurfLoad { .. }
        | Instruction::SurfStore { .. } => InstructionLatency {
            execute: 4,
            memory: 400,
        },

        // Tensor core: 32 cycles
        Instruction::Mma { .. } | Instruction::Wmma { .. } | Instruction::Wgmma { .. } => {
            InstructionLatency {
                execute: 32,
                memory: 0,
            }
        }

        // Async copy: moderate latency
        Instruction::CpAsync { .. } => InstructionLatency {
            execute: 4,
            memory: 100,
        },

        // tcgen05 MMA: Blackwell 5th-gen Tensor Core — very high latency
        Instruction::Tcgen05Mma { .. } => InstructionLatency {
            execute: 64,
            memory: 0,
        },

        // Everything else (barriers, branches, labels, comments, etc.): 1 cycle
        Instruction::CpAsyncCommit
        | Instruction::CpAsyncWait { .. }
        | Instruction::Branch { .. }
        | Instruction::Label(_)
        | Instruction::Return
        | Instruction::BarSync { .. }
        | Instruction::BarArrive { .. }
        | Instruction::FenceAcqRel { .. }
        | Instruction::MovSpecial { .. }
        | Instruction::LoadParam { .. }
        | Instruction::Comment(_)
        | Instruction::Raw(_)
        | Instruction::Pragma(_)
        // PTX 8.x: barriers, fences, control flow — ~1 cycle
        | Instruction::Redux { .. }
        | Instruction::Stmatrix { .. }
        | Instruction::ElectSync { .. }
        | Instruction::Setmaxnreg { .. }
        | Instruction::Griddepcontrol { .. }
        | Instruction::FenceProxy { .. }
        | Instruction::MbarrierInit { .. }
        | Instruction::MbarrierArrive { .. }
        | Instruction::MbarrierWait { .. }
        | Instruction::BarrierCluster
        | Instruction::FenceCluster => InstructionLatency {
            execute: 1,
            memory: 0,
        },
    }
}

// ---------------------------------------------------------------------------
// Register extraction helpers (same patterns as dead_code.rs)
// ---------------------------------------------------------------------------

/// Extract registers defined (written to) by an instruction.
fn defs(inst: &Instruction) -> Vec<&Register> {
    match inst {
        Instruction::Add { dst, .. }
        | Instruction::Sub { dst, .. }
        | Instruction::Mul { dst, .. }
        | Instruction::Mad { dst, .. }
        | Instruction::Fma { dst, .. }
        | Instruction::Neg { dst, .. }
        | Instruction::Abs { dst, .. }
        | Instruction::Min { dst, .. }
        | Instruction::Max { dst, .. }
        | Instruction::Brev { dst, .. }
        | Instruction::Clz { dst, .. }
        | Instruction::Popc { dst, .. }
        | Instruction::Bfind { dst, .. }
        | Instruction::Bfe { dst, .. }
        | Instruction::Bfi { dst, .. }
        | Instruction::Shl { dst, .. }
        | Instruction::Shr { dst, .. }
        | Instruction::Div { dst, .. }
        | Instruction::Rem { dst, .. }
        | Instruction::And { dst, .. }
        | Instruction::Or { dst, .. }
        | Instruction::Xor { dst, .. }
        | Instruction::MadLo { dst, .. }
        | Instruction::MadHi { dst, .. }
        | Instruction::MadWide { dst, .. }
        | Instruction::Rcp { dst, .. }
        | Instruction::Rsqrt { dst, .. }
        | Instruction::Sqrt { dst, .. }
        | Instruction::Ex2 { dst, .. }
        | Instruction::Lg2 { dst, .. }
        | Instruction::Sin { dst, .. }
        | Instruction::Cos { dst, .. }
        | Instruction::SetP { dst, .. }
        | Instruction::Load { dst, .. }
        | Instruction::Cvt { dst, .. }
        | Instruction::MovSpecial { dst, .. }
        | Instruction::LoadParam { dst, .. }
        | Instruction::Atom { dst, .. }
        | Instruction::AtomCas { dst, .. }
        | Instruction::Dp4a { dst, .. }
        | Instruction::Dp2a { dst, .. }
        | Instruction::Tex1d { dst, .. }
        | Instruction::Tex2d { dst, .. }
        | Instruction::Tex3d { dst, .. }
        | Instruction::SurfLoad { dst, .. }
        | Instruction::Redux { dst, .. }
        | Instruction::ElectSync { dst, .. } => vec![dst],

        Instruction::Ldmatrix { dst_regs, .. } => dst_regs.iter().collect(),

        Instruction::Store { .. }
        | Instruction::CpAsync { .. }
        | Instruction::CpAsyncCommit
        | Instruction::CpAsyncWait { .. }
        | Instruction::Branch { .. }
        | Instruction::Label(_)
        | Instruction::Return
        | Instruction::BarSync { .. }
        | Instruction::BarArrive { .. }
        | Instruction::FenceAcqRel { .. }
        | Instruction::TmaLoad { .. }
        | Instruction::Red { .. }
        | Instruction::SurfStore { .. }
        | Instruction::Comment(_)
        | Instruction::Raw(_)
        | Instruction::Pragma(_)
        | Instruction::Stmatrix { .. }
        | Instruction::Setmaxnreg { .. }
        | Instruction::Griddepcontrol { .. }
        | Instruction::FenceProxy { .. }
        | Instruction::MbarrierInit { .. }
        | Instruction::MbarrierArrive { .. }
        | Instruction::MbarrierWait { .. }
        | Instruction::Tcgen05Mma { .. }
        | Instruction::BarrierCluster
        | Instruction::FenceCluster
        | Instruction::CpAsyncBulk { .. } => vec![],

        Instruction::Wmma { op, fragments, .. } => match op {
            WmmaOp::LoadA | WmmaOp::LoadB | WmmaOp::Mma => fragments.iter().collect(),
            WmmaOp::StoreD => vec![],
        },
        Instruction::Mma { d_regs, .. } | Instruction::Wgmma { d_regs, .. } => {
            d_regs.iter().collect()
        }
    }
}

/// Extract registers used (read from) by an instruction.
#[allow(clippy::too_many_lines)]
fn uses(inst: &Instruction) -> Vec<&Register> {
    match inst {
        Instruction::Add { a, b, .. }
        | Instruction::Sub { a, b, .. }
        | Instruction::Mul { a, b, .. }
        | Instruction::Min { a, b, .. }
        | Instruction::Max { a, b, .. }
        | Instruction::Div { a, b, .. }
        | Instruction::Rem { a, b, .. }
        | Instruction::And { a, b, .. }
        | Instruction::Or { a, b, .. }
        | Instruction::Xor { a, b, .. }
        | Instruction::SetP { a, b, .. } => {
            let mut regs = operand_regs(a);
            regs.extend(operand_regs(b));
            regs
        }

        Instruction::Shl { src, amount, .. } | Instruction::Shr { src, amount, .. } => {
            let mut regs = operand_regs(src);
            regs.extend(operand_regs(amount));
            regs
        }

        Instruction::Mad { a, b, c, .. }
        | Instruction::MadLo { a, b, c, .. }
        | Instruction::MadHi { a, b, c, .. }
        | Instruction::MadWide { a, b, c, .. }
        | Instruction::Fma { a, b, c, .. }
        | Instruction::Dp4a { a, b, c, .. }
        | Instruction::Dp2a { a, b, c, .. } => {
            let mut regs = operand_regs(a);
            regs.extend(operand_regs(b));
            regs.extend(operand_regs(c));
            regs
        }

        Instruction::Neg { src, .. }
        | Instruction::Abs { src, .. }
        | Instruction::Brev { src, .. }
        | Instruction::Clz { src, .. }
        | Instruction::Popc { src, .. }
        | Instruction::Bfind { src, .. }
        | Instruction::Rcp { src, .. }
        | Instruction::Rsqrt { src, .. }
        | Instruction::Sqrt { src, .. }
        | Instruction::Ex2 { src, .. }
        | Instruction::Lg2 { src, .. }
        | Instruction::Sin { src, .. }
        | Instruction::Cos { src, .. }
        | Instruction::Cvt { src, .. }
        | Instruction::Redux { src, .. } => operand_regs(src),

        Instruction::Bfe {
            src, start, len, ..
        } => {
            let mut regs = operand_regs(src);
            regs.extend(operand_regs(start));
            regs.extend(operand_regs(len));
            regs
        }

        Instruction::Bfi {
            insert,
            base,
            start,
            len,
            ..
        } => {
            let mut regs = operand_regs(insert);
            regs.extend(operand_regs(base));
            regs.extend(operand_regs(start));
            regs.extend(operand_regs(len));
            regs
        }

        Instruction::Load { addr, .. } | Instruction::MbarrierArrive { addr, .. } => {
            operand_regs(addr)
        }

        Instruction::Store { addr, src, .. } => {
            let mut regs = operand_regs(addr);
            regs.push(src);
            regs
        }

        Instruction::CpAsync {
            dst_shared,
            src_global,
            ..
        } => {
            let mut regs = operand_regs(dst_shared);
            regs.extend(operand_regs(src_global));
            regs
        }

        Instruction::CpAsyncCommit
        | Instruction::CpAsyncWait { .. }
        | Instruction::Label(_)
        | Instruction::Return
        | Instruction::BarSync { .. }
        | Instruction::BarArrive { .. }
        | Instruction::FenceAcqRel { .. }
        | Instruction::MovSpecial { .. }
        | Instruction::LoadParam { .. }
        | Instruction::ElectSync { .. }
        | Instruction::Setmaxnreg { .. }
        | Instruction::Griddepcontrol { .. }
        | Instruction::FenceProxy { .. }
        | Instruction::BarrierCluster
        | Instruction::FenceCluster
        | Instruction::Comment(_)
        | Instruction::Raw(_)
        | Instruction::Pragma(_) => vec![],

        Instruction::Branch { predicate, .. } => {
            if let Some((reg, _)) = predicate {
                vec![reg]
            } else {
                vec![]
            }
        }

        Instruction::Wmma {
            op,
            fragments,
            addr,
            stride,
            ..
        } => {
            let mut regs: Vec<&Register> = Vec::new();
            match op {
                WmmaOp::LoadA | WmmaOp::LoadB => {
                    if let Some(a) = addr {
                        regs.extend(operand_regs(a));
                    }
                    if let Some(s) = stride {
                        regs.extend(operand_regs(s));
                    }
                }
                WmmaOp::StoreD => {
                    regs.extend(fragments.iter());
                    if let Some(a) = addr {
                        regs.extend(operand_regs(a));
                    }
                    if let Some(s) = stride {
                        regs.extend(operand_regs(s));
                    }
                }
                WmmaOp::Mma => {
                    regs.extend(fragments.iter());
                }
            }
            regs
        }

        Instruction::Mma {
            a_regs,
            b_regs,
            c_regs,
            ..
        } => {
            let mut regs: Vec<&Register> = Vec::new();
            regs.extend(a_regs.iter());
            regs.extend(b_regs.iter());
            regs.extend(c_regs.iter());
            regs
        }

        Instruction::Wgmma { desc_a, desc_b, .. } => vec![desc_a, desc_b],

        Instruction::TmaLoad {
            dst_shared,
            desc,
            coords,
            barrier,
            ..
        } => {
            let mut regs = operand_regs(dst_shared);
            regs.push(desc);
            regs.extend(coords.iter());
            regs.push(barrier);
            regs
        }

        Instruction::Atom { addr, src, .. } | Instruction::Red { addr, src, .. } => {
            let mut regs = operand_regs(addr);
            regs.extend(operand_regs(src));
            regs
        }

        Instruction::AtomCas {
            addr,
            compare,
            value,
            ..
        } => {
            let mut regs = operand_regs(addr);
            regs.extend(operand_regs(compare));
            regs.extend(operand_regs(value));
            regs
        }

        // Texture: coord registers are used
        Instruction::Tex1d { coord, .. } | Instruction::SurfLoad { coord, .. } => {
            operand_regs(coord)
        }
        Instruction::Tex2d {
            coord_x, coord_y, ..
        } => {
            let mut regs = operand_regs(coord_x);
            regs.extend(operand_regs(coord_y));
            regs
        }
        Instruction::Tex3d {
            coord_x,
            coord_y,
            coord_z,
            ..
        } => {
            let mut regs = operand_regs(coord_x);
            regs.extend(operand_regs(coord_y));
            regs.extend(operand_regs(coord_z));
            regs
        }
        Instruction::SurfStore { coord, src, .. } => {
            let mut regs = operand_regs(coord);
            regs.push(src);
            regs
        }

        // PTX 8.x
        Instruction::Stmatrix { dst_addr, src, .. } => {
            let mut regs = operand_regs(dst_addr);
            regs.push(src);
            regs
        }
        Instruction::MbarrierInit { addr, count, .. } => {
            let mut regs = operand_regs(addr);
            regs.extend(operand_regs(count));
            regs
        }
        Instruction::MbarrierWait { addr, phase, .. } => {
            let mut regs = operand_regs(addr);
            regs.extend(operand_regs(phase));
            regs
        }

        Instruction::Tcgen05Mma { a_desc, b_desc } => vec![a_desc, b_desc],

        Instruction::CpAsyncBulk {
            dst_smem,
            src_gmem,
            desc,
        } => vec![dst_smem, src_gmem, desc],

        Instruction::Ldmatrix { src_addr, .. } => operand_regs(src_addr),
    }
}

/// Extract register references from an operand.
fn operand_regs(op: &Operand) -> Vec<&Register> {
    match op {
        Operand::Register(reg) => vec![reg],
        Operand::Address { base, .. } => vec![base],
        Operand::Immediate(_) | Operand::Symbol(_) => vec![],
    }
}

// ---------------------------------------------------------------------------
// Side-effect and memory classification helpers
// ---------------------------------------------------------------------------

/// Check if an instruction has side effects.
const fn has_side_effects(inst: &Instruction) -> bool {
    match inst {
        Instruction::Store { .. }
        | Instruction::CpAsync { .. }
        | Instruction::CpAsyncCommit
        | Instruction::CpAsyncWait { .. }
        | Instruction::Branch { .. }
        | Instruction::Label(_)
        | Instruction::Return
        | Instruction::BarSync { .. }
        | Instruction::BarArrive { .. }
        | Instruction::FenceAcqRel { .. }
        | Instruction::TmaLoad { .. }
        | Instruction::Atom { .. }
        | Instruction::AtomCas { .. }
        | Instruction::Red { .. }
        | Instruction::SurfStore { .. }
        | Instruction::Comment(_)
        | Instruction::Raw(_)
        | Instruction::Stmatrix { .. }
        | Instruction::Setmaxnreg { .. }
        | Instruction::Griddepcontrol { .. }
        | Instruction::FenceProxy { .. }
        | Instruction::MbarrierInit { .. }
        | Instruction::MbarrierArrive { .. }
        | Instruction::MbarrierWait { .. }
        | Instruction::Tcgen05Mma { .. }
        | Instruction::BarrierCluster
        | Instruction::FenceCluster
        | Instruction::CpAsyncBulk { .. } => true,

        Instruction::Wmma { op, .. } => matches!(op, WmmaOp::StoreD),

        Instruction::Add { .. }
        | Instruction::Sub { .. }
        | Instruction::Mul { .. }
        | Instruction::Mad { .. }
        | Instruction::MadLo { .. }
        | Instruction::MadHi { .. }
        | Instruction::MadWide { .. }
        | Instruction::Fma { .. }
        | Instruction::Neg { .. }
        | Instruction::Abs { .. }
        | Instruction::Min { .. }
        | Instruction::Max { .. }
        | Instruction::Brev { .. }
        | Instruction::Clz { .. }
        | Instruction::Popc { .. }
        | Instruction::Bfind { .. }
        | Instruction::Bfe { .. }
        | Instruction::Bfi { .. }
        | Instruction::Shl { .. }
        | Instruction::Shr { .. }
        | Instruction::Div { .. }
        | Instruction::Rem { .. }
        | Instruction::And { .. }
        | Instruction::Or { .. }
        | Instruction::Xor { .. }
        | Instruction::Rcp { .. }
        | Instruction::Rsqrt { .. }
        | Instruction::Sqrt { .. }
        | Instruction::Ex2 { .. }
        | Instruction::Lg2 { .. }
        | Instruction::Sin { .. }
        | Instruction::Cos { .. }
        | Instruction::SetP { .. }
        | Instruction::Load { .. }
        | Instruction::Cvt { .. }
        | Instruction::Mma { .. }
        | Instruction::Wgmma { .. }
        | Instruction::MovSpecial { .. }
        | Instruction::LoadParam { .. }
        | Instruction::Pragma(_)
        | Instruction::Dp4a { .. }
        | Instruction::Dp2a { .. }
        | Instruction::Tex1d { .. }
        | Instruction::Tex2d { .. }
        | Instruction::Tex3d { .. }
        | Instruction::SurfLoad { .. }
        | Instruction::Redux { .. }
        | Instruction::ElectSync { .. }
        // ldmatrix: warp-cooperative shared memory load — pure read, no write side effect
        | Instruction::Ldmatrix { .. } => false,
    }
}

/// Check if an instruction reads from memory.
const fn is_memory_read(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Load { .. }
            | Instruction::TmaLoad { .. }
            | Instruction::CpAsync { .. }
            | Instruction::Atom { .. }
            | Instruction::AtomCas { .. }
            | Instruction::Tex1d { .. }
            | Instruction::Tex2d { .. }
            | Instruction::Tex3d { .. }
            | Instruction::SurfLoad { .. }
    )
}

/// Check if an instruction writes to memory.
const fn is_memory_write(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Store { .. }
            | Instruction::TmaLoad { .. }
            | Instruction::CpAsync { .. }
            | Instruction::Atom { .. }
            | Instruction::AtomCas { .. }
            | Instruction::Red { .. }
            | Instruction::SurfStore { .. }
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        CacheQualifier, CmpOp, FenceScope, MemorySpace, MulMode, PtxType, RoundingMode, SpecialReg,
        VectorWidth,
    };

    fn reg(name: &str, ty: PtxType) -> Register {
        Register {
            name: name.to_string(),
            ty,
        }
    }

    fn reg_op(name: &str, ty: PtxType) -> Operand {
        Operand::Register(reg(name, ty))
    }

    fn addr_op(name: &str) -> Operand {
        Operand::Address {
            base: reg(name, PtxType::U64),
            offset: None,
        }
    }

    fn make_add(dst: &str, a: &str, b: &str) -> Instruction {
        Instruction::Add {
            ty: PtxType::U32,
            dst: reg(dst, PtxType::U32),
            a: reg_op(a, PtxType::U32),
            b: reg_op(b, PtxType::U32),
        }
    }

    fn make_mul(dst: &str, a: &str, b: &str) -> Instruction {
        Instruction::Mul {
            ty: PtxType::U32,
            mode: MulMode::Lo,
            dst: reg(dst, PtxType::U32),
            a: reg_op(a, PtxType::U32),
            b: reg_op(b, PtxType::U32),
        }
    }

    fn make_load_global(dst: &str, addr: &str) -> Instruction {
        Instruction::Load {
            space: MemorySpace::Global,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            dst: reg(dst, PtxType::F32),
            addr: addr_op(addr),
        }
    }

    fn make_store_global(addr: &str, src: &str) -> Instruction {
        Instruction::Store {
            space: MemorySpace::Global,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            addr: addr_op(addr),
            src: reg(src, PtxType::F32),
        }
    }

    fn make_label(name: &str) -> Instruction {
        Instruction::Label(name.to_string())
    }

    fn make_branch(target: &str) -> Instruction {
        Instruction::Branch {
            target: target.to_string(),
            predicate: None,
        }
    }

    fn make_bar_sync(id: u32) -> Instruction {
        Instruction::BarSync { id }
    }

    fn make_fence() -> Instruction {
        Instruction::FenceAcqRel {
            scope: FenceScope::Cta,
        }
    }

    // === Test: empty input ===
    #[test]
    fn test_empty_input() {
        let (result, report) = schedule_instructions(&[], SchedulingStrategy::MaxIlp);
        assert!(result.is_empty());
        assert_eq!(report.original_count, 0);
        assert_eq!(report.instructions_moved, 0);
        assert_eq!(report.stalls_eliminated, 0);
    }

    // === Test: single instruction ===
    #[test]
    fn test_single_instruction() {
        let insts = vec![make_add("r0", "r1", "r2")];
        let (result, report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 1);
        assert_eq!(report.instructions_moved, 0);
    }

    // === Test: independent instructions remain valid ===
    #[test]
    fn test_independent_instructions() {
        // Two independent adds: both should appear, order may change.
        let insts = vec![make_add("r0", "r1", "r2"), make_add("r3", "r4", "r5")];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 2);
        // Both instructions should be present (check dsts).
        let dst_names: Vec<&str> = result
            .iter()
            .filter_map(|inst| match inst {
                Instruction::Add { dst, .. } => Some(dst.name.as_str()),
                _ => None,
            })
            .collect();
        assert!(dst_names.contains(&"r0"));
        assert!(dst_names.contains(&"r3"));
    }

    // === Test: RAW dependency respected ===
    #[test]
    fn test_raw_dependency() {
        // r0 = r1 + r2; r3 = r0 + r4  (r3 depends on r0)
        let insts = vec![make_add("r0", "r1", "r2"), make_add("r3", "r0", "r4")];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 2);
        // Find positions.
        let pos_r0 = result
            .iter()
            .position(|i| matches!(i, Instruction::Add { dst, .. } if dst.name == "r0"));
        let pos_r3 = result
            .iter()
            .position(|i| matches!(i, Instruction::Add { dst, .. } if dst.name == "r3"));
        assert!(pos_r0.is_some());
        assert!(pos_r3.is_some());
        // r0 must come before r3.
        assert!(pos_r0 < pos_r3);
    }

    // === Test: WAR dependency respected ===
    #[test]
    fn test_war_dependency() {
        // r3 = r0 + r1; r0 = r4 + r5  (second write to r0 must come after first reads r0)
        let insts = vec![make_add("r3", "r0", "r1"), make_add("r0", "r4", "r5")];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        let pos_r3 = result
            .iter()
            .position(|i| matches!(i, Instruction::Add { dst, .. } if dst.name == "r3"));
        let pos_r0 = result
            .iter()
            .position(|i| matches!(i, Instruction::Add { dst, .. } if dst.name == "r0"));
        assert!(pos_r3.is_some());
        assert!(pos_r0.is_some());
        assert!(
            pos_r3 < pos_r0,
            "reader of r0 must come before writer of r0"
        );
    }

    // === Test: WAW dependency respected ===
    #[test]
    fn test_waw_dependency() {
        // r0 = r1 + r2; r0 = r3 + r4  (second write to r0 must come after first)
        let insts = vec![make_add("r0", "r1", "r2"), make_add("r0", "r3", "r4")];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        // Both should be present and in order.
        let first_src = result.first().and_then(|i| match i {
            Instruction::Add { a, .. } => Some(a),
            _ => None,
        });
        // First should use r1 (the original first instruction).
        assert!(matches!(first_src, Some(Operand::Register(r)) if r.name == "r1"));
    }

    // === Test: scheduling barriers are not reordered ===
    #[test]
    fn test_scheduling_barriers_not_reordered() {
        let insts = vec![
            make_add("r0", "r1", "r2"),
            make_label("L1"),
            make_add("r3", "r4", "r5"),
        ];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 3);
        // Label must be in position 1.
        assert!(matches!(&result[1], Instruction::Label(l) if l == "L1"));
    }

    // === Test: branch is a barrier ===
    #[test]
    fn test_branch_is_barrier() {
        let insts = vec![
            make_add("r0", "r1", "r2"),
            make_branch("target"),
            make_add("r3", "r4", "r5"),
        ];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 3);
        assert!(matches!(&result[1], Instruction::Branch { .. }));
    }

    // === Test: bar_sync is a barrier ===
    #[test]
    fn test_bar_sync_is_barrier() {
        let insts = vec![
            make_add("r0", "r1", "r2"),
            make_bar_sync(0),
            make_add("r3", "r4", "r5"),
        ];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 3);
        assert!(matches!(&result[1], Instruction::BarSync { id: 0 }));
    }

    // === Test: fence is a barrier ===
    #[test]
    fn test_fence_is_barrier() {
        let insts = vec![
            make_add("r0", "r1", "r2"),
            make_fence(),
            make_add("r3", "r4", "r5"),
        ];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 3);
        assert!(matches!(&result[1], Instruction::FenceAcqRel { .. }));
    }

    // === Test: side-effect instructions maintain relative order ===
    #[test]
    fn test_side_effect_ordering() {
        let insts = vec![
            make_store_global("addr1", "r0"),
            make_store_global("addr2", "r1"),
        ];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 2);
        // First store must reference addr1, second addr2.
        let addrs: Vec<&str> = result
            .iter()
            .filter_map(|i| match i {
                Instruction::Store {
                    addr: Operand::Address { base, .. },
                    ..
                } => Some(base.name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(addrs, vec!["addr1", "addr2"]);
    }

    // === Test: stores not reordered past stores ===
    #[test]
    fn test_stores_not_reordered() {
        let insts = vec![
            make_store_global("addr_a", "src1"),
            make_store_global("addr_b", "src2"),
            make_store_global("addr_c", "src3"),
        ];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        let addrs: Vec<&str> = result
            .iter()
            .filter_map(|i| match i {
                Instruction::Store {
                    addr: Operand::Address { base, .. },
                    ..
                } => Some(base.name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(addrs, vec!["addr_a", "addr_b", "addr_c"]);
    }

    // === Test: load followed by use gets independent instruction between ===
    #[test]
    fn test_load_then_use_interleaving() {
        // load r0 (200 cycle latency), add r3 = r0 + r1, add r5 = r6 + r7 (independent)
        // Scheduler should place the independent add between load and its consumer.
        let insts = vec![
            make_load_global("r0", "addr1"),
            make_add("r3", "r0", "r1"),
            make_add("r5", "r6", "r7"),
        ];
        let (result, report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 3);
        // The load should still be first (highest critical path).
        assert!(matches!(&result[0], Instruction::Load { dst, .. } if dst.name == "r0"));
        // The independent add should come between load and its consumer.
        assert!(
            matches!(&result[1], Instruction::Add { dst, .. } if dst.name == "r5"),
            "independent instruction should be scheduled between load and consumer"
        );
        assert!(report.instructions_moved > 0);
    }

    // === Test: critical path computation ===
    #[test]
    fn test_critical_path() {
        // Chain: load -> add (uses load result) -> store
        let insts = vec![
            make_load_global("r0", "addr1"),
            make_add("r1", "r0", "r2"),
            make_store_global("addr2", "r1"),
        ];
        let (_, report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        // Critical path: load (204) + add (4) + store (4).
        // The critical path should be ≥ 204 (load latency dominates).
        assert!(report.critical_path_before >= 200);
    }

    // === Test: report shows stalls eliminated ===
    #[test]
    fn test_stalls_eliminated() {
        // load r0 -> use r0 (stall without interleaving)
        // + independent instruction that can fill the gap.
        let insts = vec![
            make_load_global("r0", "addr1"),
            make_add("r3", "r0", "r1"), // depends on load
            make_add("r5", "r6", "r7"), // independent
        ];
        let (_, report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        // With scheduling, the independent add fills some of the load latency gap.
        // Stalls should be reduced (possibly to 0 if fully hidden).
        // We just check the report is populated sensibly.
        assert!(
            report.stalls_eliminated > 0 || report.instructions_moved > 0,
            "scheduling should show some benefit"
        );
    }

    // === Test: all-barriers input unchanged ===
    #[test]
    fn test_all_barriers_unchanged() {
        let insts = vec![make_label("L0"), make_bar_sync(0), make_label("L1")];
        let (result, report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 3);
        assert_eq!(report.instructions_moved, 0);
        assert!(matches!(&result[0], Instruction::Label(l) if l == "L0"));
        assert!(matches!(&result[1], Instruction::BarSync { id: 0 }));
        assert!(matches!(&result[2], Instruction::Label(l) if l == "L1"));
    }

    // === Test: report shows moved instructions count ===
    #[test]
    fn test_report_moved_count() {
        // Load with high latency, consumer, independent instructions.
        let insts = vec![
            make_load_global("r0", "addr1"),
            make_add("r3", "r0", "r1"),  // depends on load
            make_add("r5", "r6", "r7"),  // independent
            make_add("r8", "r9", "r10"), // independent
        ];
        let (result, report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 4);
        // At least some instructions should have been moved.
        assert!(
            report.instructions_moved > 0,
            "expected instructions to be reordered"
        );
    }

    // === Test: MinRegPressure strategy ===
    #[test]
    fn test_min_reg_pressure_strategy() {
        // Create a scenario where MinRegPressure makes different choices.
        let insts = vec![
            make_load_global("r0", "addr1"),
            make_load_global("r1", "addr2"),
            make_add("r2", "r0", "r1"), // consumes both loads
            make_add("r3", "r4", "r5"), // independent, introduces new regs
        ];
        let (result_ilp, _) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        let (result_regp, _) = schedule_instructions(&insts, SchedulingStrategy::MinRegPressure);
        // Both should produce valid output.
        assert_eq!(result_ilp.len(), 4);
        assert_eq!(result_regp.len(), 4);
    }

    // === Test: Return is a barrier ===
    #[test]
    fn test_return_is_barrier() {
        let insts = vec![make_add("r0", "r1", "r2"), Instruction::Return];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 2);
        assert!(matches!(&result[1], Instruction::Return));
    }

    // === Test: latency model covers all categories ===
    #[test]
    fn test_latency_model() {
        // Arithmetic
        let add = make_add("r0", "r1", "r2");
        assert_eq!(estimate_latency(&add).execute, 4);
        assert_eq!(estimate_latency(&add).memory, 0);

        // Multiply
        let mul = make_mul("r0", "r1", "r2");
        assert_eq!(estimate_latency(&mul).execute, 8);

        // Global load
        let ld = make_load_global("r0", "addr1");
        assert_eq!(estimate_latency(&ld).memory, 200);

        // Store
        let st = make_store_global("addr1", "r0");
        assert_eq!(estimate_latency(&st).memory, 0);

        // Shared load
        let ld_shared = Instruction::Load {
            space: MemorySpace::Shared,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            dst: reg("r0", PtxType::F32),
            addr: addr_op("addr1"),
        };
        assert_eq!(estimate_latency(&ld_shared).memory, 20);

        // Special math
        let sqrt = Instruction::Sqrt {
            rnd: None,
            ty: PtxType::F32,
            dst: reg("r0", PtxType::F32),
            src: reg_op("r1", PtxType::F32),
        };
        assert_eq!(estimate_latency(&sqrt).execute, 16);

        // Transcendental
        let sin = Instruction::Sin {
            approx: true,
            ty: PtxType::F32,
            dst: reg("r0", PtxType::F32),
            src: reg_op("r1", PtxType::F32),
        };
        assert_eq!(estimate_latency(&sin).execute, 24);

        // Bit manipulation
        let brev = Instruction::Brev {
            ty: PtxType::B32,
            dst: reg("r0", PtxType::B32),
            src: reg_op("r1", PtxType::B32),
        };
        assert_eq!(estimate_latency(&brev).execute, 4);

        // SetP
        let setp = Instruction::SetP {
            cmp: CmpOp::Lt,
            ty: PtxType::U32,
            dst: reg("p0", PtxType::Pred),
            a: reg_op("r0", PtxType::U32),
            b: reg_op("r1", PtxType::U32),
        };
        assert_eq!(estimate_latency(&setp).execute, 4);

        // Cvt
        let cvt = Instruction::Cvt {
            rnd: Some(RoundingMode::Rn),
            dst_ty: PtxType::F32,
            src_ty: PtxType::U32,
            dst: reg("f0", PtxType::F32),
            src: reg_op("r0", PtxType::U32),
        };
        assert_eq!(estimate_latency(&cvt).execute, 4);

        // MovSpecial: 1 cycle
        let movs = Instruction::MovSpecial {
            dst: reg("r0", PtxType::U32),
            special: SpecialReg::TidX,
        };
        assert_eq!(estimate_latency(&movs).execute, 1);

        // Barrier: 1 cycle
        let bar = make_bar_sync(0);
        assert_eq!(estimate_latency(&bar).execute, 1);
    }

    // === Test: complex interleaving scenario ===
    #[test]
    fn test_complex_interleaving() {
        // Two independent load-use chains.
        // Optimal: interleave loads and their consumers.
        let insts = vec![
            make_load_global("a", "addr1"), // load a (high latency)
            make_add("c", "a", "x"),        // use a -> depends on load a
            make_load_global("b", "addr2"), // load b (high latency, independent of a)
            make_add("d", "b", "y"),        // use b -> depends on load b
        ];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 4);

        // Both loads should appear before their respective consumers.
        let pos_load_a = result
            .iter()
            .position(|i| matches!(i, Instruction::Load { dst, .. } if dst.name == "a"));
        let pos_use_a = result
            .iter()
            .position(|i| matches!(i, Instruction::Add { dst, .. } if dst.name == "c"));
        let pos_load_b = result
            .iter()
            .position(|i| matches!(i, Instruction::Load { dst, .. } if dst.name == "b"));
        let pos_use_b = result
            .iter()
            .position(|i| matches!(i, Instruction::Add { dst, .. } if dst.name == "d"));
        assert!(pos_load_a < pos_use_a);
        assert!(pos_load_b < pos_use_b);
    }

    // === Test: schedule output is a permutation (no instructions added or dropped) ===
    #[test]
    fn test_schedule_is_permutation() {
        let insts = vec![
            make_load_global("r0", "addr1"),
            make_add("r3", "r0", "r1"),
            make_add("r5", "r6", "r7"),
            make_mul("r8", "r9", "r10"),
            make_add("r11", "r12", "r13"),
        ];
        let n = insts.len();
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        // Output must be same length.
        assert_eq!(
            result.len(),
            n,
            "scheduled output must have same instruction count"
        );
        // Collect destination register names from original and scheduled output.
        // Since every instruction in this test writes a distinct register, checking
        // the multiset of written registers is sufficient to verify it's a permutation.
        let mut orig_dsts: Vec<String> = insts
            .iter()
            .flat_map(|inst| defs(inst).into_iter().map(|r| r.name.clone()))
            .collect();
        let mut sched_dsts: Vec<String> = result
            .iter()
            .flat_map(|inst| defs(inst).into_iter().map(|r| r.name.clone()))
            .collect();
        orig_dsts.sort();
        sched_dsts.sort();
        assert_eq!(
            orig_dsts, sched_dsts,
            "scheduled output must be a permutation of the input"
        );
    }

    // === Test: scheduling an already-optimal sequence is idempotent ===
    #[test]
    fn test_schedule_idempotent_on_optimal() {
        // A fully sequential chain: each instruction depends on the previous one.
        // The scheduler cannot reorder anything; re-scheduling must produce the same order.
        let insts = vec![
            make_add("r0", "r1", "r2"),
            make_add("r3", "r0", "r4"), // depends on r0
            make_add("r5", "r3", "r6"), // depends on r3
            make_add("r7", "r5", "r8"), // depends on r5
        ];
        let (first_pass, _) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        let (second_pass, _) = schedule_instructions(&first_pass, SchedulingStrategy::MaxIlp);
        // Both passes must produce identical instruction sequences.
        assert_eq!(first_pass.len(), second_pass.len());
        for (a, b) in first_pass.iter().zip(second_pass.iter()) {
            // Compare by the destination register name (unique per instruction here).
            let dst_a: Vec<_> = defs(a).into_iter().map(|r| r.name.clone()).collect();
            let dst_b: Vec<_> = defs(b).into_iter().map(|r| r.name.clone()).collect();
            assert_eq!(
                dst_a, dst_b,
                "idempotent: second scheduling pass must match first"
            );
        }
    }

    // === Test: comment and raw instructions preserved ===
    #[test]
    fn test_comment_and_raw_preserved() {
        let insts = vec![
            Instruction::Comment("hello".to_string()),
            make_add("r0", "r1", "r2"),
            Instruction::Raw("nop;".to_string()),
        ];
        let (result, _report) = schedule_instructions(&insts, SchedulingStrategy::MaxIlp);
        assert_eq!(result.len(), 3);
        // Comments and raw instructions have side effects, so they maintain
        // relative ordering among side-effect instructions.
        let has_comment = result
            .iter()
            .any(|i| matches!(i, Instruction::Comment(c) if c == "hello"));
        let has_raw = result
            .iter()
            .any(|i| matches!(i, Instruction::Raw(s) if s == "nop;"));
        assert!(has_comment);
        assert!(has_raw);
    }
}
