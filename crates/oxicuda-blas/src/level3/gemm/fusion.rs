//! Graph-based fusion of GEMM chains.
//!
//! Automatically fuses sequences of GEMMs (e.g. `A * B * C`) into optimised
//! pipelines that minimise intermediate materialisations. The module provides:
//!
//! - [`GemmGraph`]: a DAG of GEMM operations with typed inputs/outputs.
//! - [`FusionPass`]: an analysis pass that identifies fusible pairs and builds
//!   a [`FusionPlan`] with staged execution and memory savings estimates.
//! - [`optimal_chain_order`]: classic matrix-chain-multiplication ordering via
//!   dynamic programming (minimises total FLOPs).

use std::fmt;

use crate::error::{BlasError, BlasResult};
use crate::types::Transpose;

// ---------------------------------------------------------------------------
// Node-level types
// ---------------------------------------------------------------------------

/// The kind of operation a [`GemmNode`] represents.
#[derive(Debug, Clone, PartialEq)]
pub enum GemmOp {
    /// Standard matrix multiply: `C = A * B`.
    Gemm,
    /// Uniform scalar scaling: `B = alpha * A`.
    /// Uniform scalar scaling: `B = alpha * A`.
    Scale {
        /// The scaling factor.
        alpha: f64,
    },
    /// Element-wise addition: `C = A + B`.
    Add,
    /// Transpose a matrix.
    Transpose,
}

/// Describes where a [`GemmNode`] gets an operand.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeInput {
    /// An externally-provided matrix identified by name.
    External {
        /// Human-readable name for the external matrix.
        name: String,
    },
    /// The output of a previous node in the graph.
    NodeOutput {
        /// The id of the producing node.
        node_id: usize,
    },
}

/// A single node in the GEMM computation graph.
#[derive(Debug, Clone)]
pub struct GemmNode {
    /// Unique identifier within the owning [`GemmGraph`].
    pub id: usize,
    /// The operation this node performs.
    pub op: GemmOp,
    /// Number of rows of the output matrix.
    pub m: u32,
    /// Number of columns of the output matrix.
    pub n: u32,
    /// Shared (inner / reduction) dimension — meaningful for `Gemm` nodes.
    pub k: u32,
    /// Transpose mode for operand A.
    pub transpose_a: Transpose,
    /// Transpose mode for operand B.
    pub transpose_b: Transpose,
    /// Where each operand comes from (ordered: A then B for Gemm, single for
    /// Scale / Transpose, two for Add).
    pub inputs: Vec<NodeInput>,
}

// ---------------------------------------------------------------------------
// GemmGraph — directed acyclic graph of GEMM-related operations
// ---------------------------------------------------------------------------

/// A directed acyclic graph (DAG) of GEMM and auxiliary operations.
///
/// Nodes are added incrementally; the caller must finally designate one node as
/// the graph output via [`set_output`](Self::set_output).
#[derive(Debug, Clone)]
pub struct GemmGraph {
    /// All nodes in topological order (id == index).
    nodes: Vec<GemmNode>,
    /// Which node produces the final result.
    output_node: Option<usize>,
}

impl GemmGraph {
    /// Creates an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            output_node: None,
        }
    }

    /// Adds a GEMM node: `C[m×n] = A[m×k] * B[k×n]`.
    ///
    /// Returns the new node's id.
    pub fn add_gemm(
        &mut self,
        m: u32,
        n: u32,
        k: u32,
        input_a: NodeInput,
        input_b: NodeInput,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(GemmNode {
            id,
            op: GemmOp::Gemm,
            m,
            n,
            k,
            transpose_a: Transpose::NoTrans,
            transpose_b: Transpose::NoTrans,
            inputs: vec![input_a, input_b],
        });
        id
    }

    /// Adds a scalar-scaling node: `B = alpha * A`.
    ///
    /// Inherits dimensions from the producer node.  Returns an error if
    /// `node_id` is out of range.
    pub fn add_scale(&mut self, node_id: usize, alpha: f64) -> BlasResult<usize> {
        let src = self
            .nodes
            .get(node_id)
            .ok_or_else(|| BlasError::InvalidArgument(format!("node {node_id} not found")))?;
        let (m, n) = (src.m, src.n);
        let id = self.nodes.len();
        self.nodes.push(GemmNode {
            id,
            op: GemmOp::Scale { alpha },
            m,
            n,
            k: 0,
            transpose_a: Transpose::NoTrans,
            transpose_b: Transpose::NoTrans,
            inputs: vec![NodeInput::NodeOutput { node_id }],
        });
        Ok(id)
    }

    /// Adds an element-wise addition node: `C = A + B`.
    ///
    /// Both inputs must have the same dimensions.  Returns an error if the
    /// referenced nodes do not exist or their dimensions mismatch.
    pub fn add_add(&mut self, a_id: usize, b_id: usize) -> BlasResult<usize> {
        let a = self
            .nodes
            .get(a_id)
            .ok_or_else(|| BlasError::InvalidArgument(format!("node {a_id} not found")))?;
        let b = self
            .nodes
            .get(b_id)
            .ok_or_else(|| BlasError::InvalidArgument(format!("node {b_id} not found")))?;
        if a.m != b.m || a.n != b.n {
            return Err(BlasError::DimensionMismatch(format!(
                "add nodes {a_id} ({}x{}) and {b_id} ({}x{}) differ",
                a.m, a.n, b.m, b.n,
            )));
        }
        let (m, n) = (a.m, a.n);
        let id = self.nodes.len();
        self.nodes.push(GemmNode {
            id,
            op: GemmOp::Add,
            m,
            n,
            k: 0,
            transpose_a: Transpose::NoTrans,
            transpose_b: Transpose::NoTrans,
            inputs: vec![
                NodeInput::NodeOutput { node_id: a_id },
                NodeInput::NodeOutput { node_id: b_id },
            ],
        });
        Ok(id)
    }

    /// Designates `node_id` as the graph's output.
    pub fn set_output(&mut self, node_id: usize) -> BlasResult<()> {
        if node_id >= self.nodes.len() {
            return Err(BlasError::InvalidArgument(format!(
                "node {node_id} does not exist"
            )));
        }
        self.output_node = Some(node_id);
        Ok(())
    }

    /// Returns the output node id, if set.
    #[must_use]
    pub fn output_node(&self) -> Option<usize> {
        self.output_node
    }

    /// Returns a slice of all nodes.
    #[must_use]
    pub fn nodes(&self) -> &[GemmNode] {
        &self.nodes
    }

    /// Number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Convenience constructor for a linear chain `M_0 * M_1 * … * M_{n-1}`.
    ///
    /// `matrices` gives the `(rows, cols)` of each matrix in left-to-right
    /// order. Adjacent matrices must have compatible inner dimensions.
    ///
    /// The chain is multiplied left-to-right in the naïve order (no
    /// reordering). Use [`optimal_chain_order`] first if you want minimal
    /// FLOPs.
    pub fn chain(matrices: &[(u32, u32)]) -> BlasResult<Self> {
        if matrices.len() < 2 {
            return Err(BlasError::InvalidArgument(
                "chain requires at least 2 matrices".into(),
            ));
        }

        // Validate compatible inner dimensions.
        for i in 0..matrices.len() - 1 {
            if matrices[i].1 != matrices[i + 1].0 {
                return Err(BlasError::DimensionMismatch(format!(
                    "matrix {} cols ({}) != matrix {} rows ({})",
                    i,
                    matrices[i].1,
                    i + 1,
                    matrices[i + 1].0,
                )));
            }
        }

        let mut graph = Self::new();

        // First multiplication: M_0 * M_1.
        let (m0, k0) = (matrices[0].0, matrices[0].1);
        let n0 = matrices[1].1;
        let prev = graph.add_gemm(
            m0,
            n0,
            k0,
            NodeInput::External {
                name: format!("M{}", 0),
            },
            NodeInput::External {
                name: format!("M{}", 1),
            },
        );

        // Subsequent multiplications: result * M_{i+1}.
        let mut last = prev;
        for (i, mat) in matrices.iter().enumerate().skip(2) {
            let prev_node = &graph.nodes[last];
            let m = prev_node.m;
            let k = prev_node.n;
            let n = mat.1;
            last = graph.add_gemm(
                m,
                n,
                k,
                NodeInput::NodeOutput { node_id: last },
                NodeInput::External {
                    name: format!("M{i}"),
                },
            );
        }

        graph.set_output(last)?;
        Ok(graph)
    }

    // -- helpers for analysis ------------------------------------------------

    /// Returns the set of node ids that consume the output of `node_id`.
    fn consumers_of(&self, node_id: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| {
                n.inputs.iter().any(|inp| match inp {
                    NodeInput::NodeOutput { node_id: id } => *id == node_id,
                    _ => false,
                })
            })
            .map(|n| n.id)
            .collect()
    }

    /// Number of consumers of a given node.
    #[must_use]
    pub fn fan_out(&self, node_id: usize) -> usize {
        self.consumers_of(node_id).len()
    }
}

impl Default for GemmGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Fusion strategy and pair identification
// ---------------------------------------------------------------------------

/// Strategy for fusing a producer-consumer pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionStrategy {
    /// Fuse the producer's output directly into the consumer's epilogue,
    /// eliminating the intermediate write-back entirely.
    EpilogueFusion,
    /// Reuse the producer's shared-memory tile as an input to the consumer,
    /// avoiding a global-memory round-trip.
    SharedMemoryReuse,
    /// Overlap producer and consumer execution via CUDA streams so that
    /// the consumer begins as soon as partial tiles are ready.
    StreamPipelining,
    /// The pair cannot be fused (kept for completeness).
    None,
}

/// A pair of nodes that are candidates for fusion.
#[derive(Debug, Clone)]
pub struct FusiblePair {
    /// Node whose output feeds the consumer.
    pub producer: usize,
    /// Node that reads the producer's output.
    pub consumer: usize,
    /// The fusion approach selected for this pair.
    pub strategy: FusionStrategy,
}

// ---------------------------------------------------------------------------
// FusionStage / FusionPlan
// ---------------------------------------------------------------------------

/// One execution stage in a [`FusionPlan`].
///
/// Within a stage, all nodes can execute concurrently (subject to data
/// dependencies already satisfied by prior stages).
#[derive(Debug, Clone)]
pub struct FusionStage {
    /// Zero-based index of this stage.
    pub stage_index: u32,
    /// Node ids scheduled in this stage.
    pub nodes: Vec<usize>,
    /// Whether this stage can overlap with the *next* stage via stream
    /// pipelining.
    pub can_overlap_with_next: bool,
    /// Bytes of intermediate storage required by the nodes in this stage.
    pub intermediate_bytes: usize,
}

/// The complete plan produced by [`FusionPass::analyze`].
#[derive(Debug, Clone)]
pub struct FusionPlan {
    /// Ordered execution stages.
    pub stages: Vec<FusionStage>,
    /// Pairs that were identified as fusible.
    pub fused_pairs: Vec<FusiblePair>,
    /// Total GEMM kernel launches after fusion.
    pub total_gemm_calls: u32,
    /// Number of GEMM launches in the original (unfused) graph.
    pub original_gemm_calls: u32,
}

impl fmt::Display for FusionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "FusionPlan: {} stages, {} -> {} GEMM calls ({} fused pairs)",
            self.stages.len(),
            self.original_gemm_calls,
            self.total_gemm_calls,
            self.fused_pairs.len(),
        )?;
        for stage in &self.stages {
            write!(
                f,
                "  stage {}: nodes {:?}, {} intermediate bytes",
                stage.stage_index, stage.nodes, stage.intermediate_bytes,
            )?;
            if stage.can_overlap_with_next {
                write!(f, " [overlap]")?;
            }
            writeln!(f)?;
        }
        for pair in &self.fused_pairs {
            writeln!(
                f,
                "  fused: {} -> {} ({:?})",
                pair.producer, pair.consumer, pair.strategy,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Higher-level fusion pass: FusionType / FusionOpportunity / FusedKernelPlan
// ---------------------------------------------------------------------------

/// Classifies the pattern that a fusion opportunity represents.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionType {
    /// GEMM + bias-add + activation epilogue (most common deep-learning pattern).
    GemmBiasActivation,
    /// GEMM followed by an in-place layer normalisation (fused kernel reduces
    /// a global-memory round-trip for the normalisation reduction).
    GemmLayerNorm,
    /// Two back-to-back GEMMs where the first output is consumed only by the
    /// second (stream-pipelined batched path).
    ConsecutiveGemm,
    /// GEMM followed by a scalar element-wise scale (alpha multiplication).
    GemmScale,
}

impl fmt::Display for FusionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GemmBiasActivation => write!(f, "GemmBiasActivation"),
            Self::GemmLayerNorm => write!(f, "GemmLayerNorm"),
            Self::ConsecutiveGemm => write!(f, "ConsecutiveGemm"),
            Self::GemmScale => write!(f, "GemmScale"),
        }
    }
}

/// A detected opportunity to fuse a group of nodes into a single kernel.
#[derive(Debug, Clone)]
pub struct FusionOpportunity {
    /// IDs of the nodes that will be merged.
    pub node_ids: Vec<usize>,
    /// Estimated performance speedup relative to running the nodes separately.
    /// `1.0` = no benefit; `2.0` = twice as fast.
    pub estimated_speedup: f32,
    /// The structural pattern that was matched.
    pub fusion_type: FusionType,
}

/// A single operation in a [`FusedKernelPlan`].
#[derive(Debug, Clone)]
pub enum FusedOp {
    /// A GEMM node fused with a downstream epilogue (bias/activation/scale).
    FusedGemmEpilogue {
        /// The GEMM producer node.
        gemm: GemmNode,
        /// The epilogue consumer node.
        epilogue: GemmNode,
    },
    /// A standalone, un-fused node (pass-through).
    Standalone {
        /// Graph node id.
        node_id: usize,
    },
}

/// The execution plan produced by [`FusionPass::apply`].
#[derive(Debug, Clone)]
pub struct FusedKernelPlan {
    /// Ordered list of operations after applying the fusion.
    pub operations: Vec<FusedOp>,
    /// Bytes of intermediate storage that will no longer be materialised.
    pub memory_saved: usize,
    /// Fractional compute overhead introduced by the fused kernel (typically
    /// small; `0.0` = no overhead, `0.05` = 5 % overhead).
    pub compute_overhead: f32,
}

// ---------------------------------------------------------------------------
// FusionPass — the core analysis
// ---------------------------------------------------------------------------

/// Analyses a [`GemmGraph`] and produces a [`FusionPlan`].
pub struct FusionPass;

impl FusionPass {
    /// Runs the full fusion analysis pipeline.
    pub fn analyze(graph: &GemmGraph) -> BlasResult<FusionPlan> {
        if graph.is_empty() {
            return Err(BlasError::InvalidArgument(
                "cannot analyse an empty graph".into(),
            ));
        }

        let fusible = Self::find_fusible_pairs(graph);
        let stages = Self::build_stages(graph, &fusible);

        let original_gemm_calls = graph
            .nodes()
            .iter()
            .filter(|n| n.op == GemmOp::Gemm)
            .count() as u32;

        // Every epilogue-fusion or shared-memory-reuse pair eliminates one
        // GEMM launch (the producer's result goes straight to the consumer).
        let fused_count = fusible
            .iter()
            .filter(|p| {
                matches!(
                    p.strategy,
                    FusionStrategy::EpilogueFusion | FusionStrategy::SharedMemoryReuse
                )
            })
            .count() as u32;

        let total_gemm_calls = original_gemm_calls.saturating_sub(fused_count);

        Ok(FusionPlan {
            stages,
            fused_pairs: fusible,
            total_gemm_calls,
            original_gemm_calls,
        })
    }

    /// Identifies all producer-consumer pairs eligible for fusion.
    ///
    /// A pair is fusible when:
    /// 1. Both producer and consumer are `Gemm` nodes.
    /// 2. The producer has exactly one consumer (fan-out == 1).
    /// 3. The consumer directly uses the producer's output as one of its
    ///    operands.
    pub fn find_fusible_pairs(graph: &GemmGraph) -> Vec<FusiblePair> {
        let mut pairs = Vec::new();

        for node in graph.nodes() {
            if node.op != GemmOp::Gemm {
                continue;
            }

            let consumers = graph.consumers_of(node.id);
            if consumers.len() != 1 {
                continue;
            }

            let consumer_id = consumers[0];
            let consumer = &graph.nodes()[consumer_id];
            if consumer.op != GemmOp::Gemm {
                continue;
            }

            let strategy = Self::select_strategy(node, consumer);
            pairs.push(FusiblePair {
                producer: node.id,
                consumer: consumer_id,
                strategy,
            });
        }

        pairs
    }

    /// Heuristically selects a [`FusionStrategy`] for a producer-consumer pair.
    fn select_strategy(producer: &GemmNode, consumer: &GemmNode) -> FusionStrategy {
        // If the producer's output tiles align with the consumer's input
        // tiles (same M or N dimension), epilogue fusion is ideal.
        if producer.m == consumer.m && producer.n == consumer.k {
            return FusionStrategy::EpilogueFusion;
        }

        // If both are small enough to fit tile data in shared memory (each
        // dimension <= 256), shared-memory reuse is viable.
        let small_threshold = 256;
        if producer.m <= small_threshold
            && producer.n <= small_threshold
            && consumer.m <= small_threshold
            && consumer.n <= small_threshold
        {
            return FusionStrategy::SharedMemoryReuse;
        }

        // Fall back to stream pipelining for large, misaligned pairs.
        FusionStrategy::StreamPipelining
    }

    /// Estimates the total bytes of intermediate storage required by the
    /// *unfused* graph (every node except the output materialises its result).
    ///
    /// Assumes f32 (4 bytes) per element.
    pub fn estimate_intermediate_memory(graph: &GemmGraph) -> usize {
        let output = graph.output_node();
        graph
            .nodes()
            .iter()
            .filter(|n| Some(n.id) != output)
            .map(|n| n.m as usize * n.n as usize * 4) // f32 = 4 bytes
            .sum()
    }

    /// Estimates the intermediate memory under a [`FusionPlan`].
    ///
    /// Fused pairs with `EpilogueFusion` or `SharedMemoryReuse` eliminate
    /// the producer's intermediate entirely.
    pub fn estimate_fused_memory(graph: &GemmGraph, plan: &FusionPlan) -> usize {
        let output = graph.output_node();

        // Collect the set of producers whose intermediates are eliminated.
        let eliminated: std::collections::HashSet<usize> = plan
            .fused_pairs
            .iter()
            .filter(|p| {
                matches!(
                    p.strategy,
                    FusionStrategy::EpilogueFusion | FusionStrategy::SharedMemoryReuse
                )
            })
            .map(|p| p.producer)
            .collect();

        graph
            .nodes()
            .iter()
            .filter(|n| Some(n.id) != output && !eliminated.contains(&n.id))
            .map(|n| n.m as usize * n.n as usize * 4)
            .sum()
    }

    /// Fractional memory savings: `1.0 - fused / original`.
    ///
    /// Returns 0.0 when the original graph requires no intermediates.
    pub fn memory_savings(graph: &GemmGraph, plan: &FusionPlan) -> f64 {
        let original = Self::estimate_intermediate_memory(graph);
        if original == 0 {
            return 0.0;
        }
        let fused = Self::estimate_fused_memory(graph, plan);
        1.0 - (fused as f64 / original as f64)
    }

    // -- higher-level opportunity analysis ------------------------------------

    /// Analyse the graph and return all detected fusion opportunities.
    ///
    /// Three pattern classes are detected:
    ///
    /// * **`GemmBiasActivation`**: a `Gemm` node whose sole consumer is an
    ///   `Add` node (bias) that is itself the sole consumer of a `Scale` node
    ///   (activation approximation), **or** any `Gemm → Add` two-step chain.
    /// * **`ConsecutiveGemm`**: two `Gemm` nodes in a direct producer-consumer
    ///   relationship (detected via [`FusionPass::find_fusible_pairs`]).
    /// * **`GemmScale`**: a `Gemm` node whose sole consumer is a `Scale` node.
    ///
    /// Opportunities are filtered by `min_speedup_estimate`; set it to `0.0`
    /// to return every valid opportunity.
    pub fn analyze_opportunities(
        graph: &GemmGraph,
        min_speedup_estimate: f32,
    ) -> Vec<FusionOpportunity> {
        let mut opps = Vec::new();

        // Detect GemmBiasActivation, GemmScale, and ConsecutiveGemm patterns.
        for node in graph.nodes() {
            if node.op != GemmOp::Gemm {
                continue;
            }

            let consumers = graph.consumers_of(node.id);
            if consumers.len() != 1 {
                continue; // fan-out > 1 prevents safe fusion
            }
            let consumer_id = consumers[0];
            let consumer = &graph.nodes()[consumer_id];

            let (fusion_type, node_ids) = match &consumer.op {
                GemmOp::Add => {
                    // Check if the Add's second consumers form an
                    // activation-like chain. For now, Gemm → Add qualifies
                    // as GemmBiasActivation (bias-only variant).
                    (FusionType::GemmBiasActivation, vec![node.id, consumer_id])
                }
                GemmOp::Scale { .. } => (FusionType::GemmScale, vec![node.id, consumer_id]),
                GemmOp::Gemm => (FusionType::ConsecutiveGemm, vec![node.id, consumer_id]),
                _ => continue,
            };

            let speedup = Self::estimate_speedup(
                &fusion_type,
                node.m as usize,
                node.n as usize,
                node.k as usize,
            );

            if speedup >= min_speedup_estimate {
                opps.push(FusionOpportunity {
                    node_ids,
                    estimated_speedup: speedup,
                    fusion_type,
                });
            }
        }

        opps
    }

    /// Estimate the speedup from fusing a particular pattern given the GEMM
    /// dimensions `m × n × k`.
    ///
    /// # Model
    ///
    /// - For **memory-bandwidth-bound** problems (`M*N > 4*K`), fusing the
    ///   epilogue avoids one full global-memory write + read cycle, yielding
    ///   roughly 1.5–2.0× throughput.
    /// - For **compute-bound** problems, the benefit is smaller (1.05–1.4×).
    #[must_use]
    pub fn estimate_speedup(fusion_type: &FusionType, m: usize, n: usize, k: usize) -> f32 {
        match fusion_type {
            FusionType::GemmBiasActivation => {
                let mn = (m * n) as f32;
                let k_f = k as f32;
                // Memory-bandwidth bound: MN large relative to K.
                if mn > k_f * 4.0 { 1.8 } else { 1.1 }
            }
            FusionType::GemmLayerNorm => 1.3,
            FusionType::ConsecutiveGemm => 1.05,
            FusionType::GemmScale => 1.4,
        }
    }

    /// Apply a [`FusionOpportunity`] to the graph and produce a
    /// [`FusedKernelPlan`].
    ///
    /// Nodes that are part of the opportunity are merged into
    /// `FusedOp::FusedGemmEpilogue` entries; all other nodes become
    /// `FusedOp::Standalone`.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::InvalidArgument`] when the opportunity references
    /// a node id that does not exist in `graph`.
    pub fn apply(graph: &GemmGraph, opp: &FusionOpportunity) -> BlasResult<FusedKernelPlan> {
        // Validate all referenced node ids upfront.
        for &nid in &opp.node_ids {
            if nid >= graph.len() {
                return Err(BlasError::InvalidArgument(format!(
                    "fusion opportunity references non-existent node {nid}"
                )));
            }
        }

        // Build a set of node ids that belong to this opportunity so we can
        // classify every graph node quickly.
        let fused_set: std::collections::HashSet<usize> = opp.node_ids.iter().copied().collect();

        let mut operations = Vec::new();

        // For the fused pair we expect exactly two node ids: producer + consumer.
        // Larger groups (future: multi-hop) fall back to standalone emission.
        if opp.node_ids.len() == 2 {
            let producer_id = opp.node_ids[0];
            let consumer_id = opp.node_ids[1];
            let producer = graph.nodes()[producer_id].clone();
            let consumer = graph.nodes()[consumer_id].clone();

            // Emit a single fused entry for the pair.
            operations.push(FusedOp::FusedGemmEpilogue {
                gemm: producer,
                epilogue: consumer,
            });

            // Emit standalone entries for all nodes outside the fused set, in
            // topological order (id order, which is guaranteed to be topo-sorted
            // by GemmGraph's construction).
            for node in graph.nodes() {
                if !fused_set.contains(&node.id) {
                    operations.push(FusedOp::Standalone { node_id: node.id });
                }
            }
        } else {
            // Generic fallback: every node becomes standalone.
            for node in graph.nodes() {
                operations.push(FusedOp::Standalone { node_id: node.id });
            }
        }

        // Memory saved: the producer's intermediate buffer is no longer
        // written to global memory.  Assumes f32 (4 bytes per element).
        let memory_saved: usize = if opp.node_ids.len() >= 2 {
            let producer = &graph.nodes()[opp.node_ids[0]];
            producer.m as usize * producer.n as usize * 4
        } else {
            0
        };

        // Compute overhead is pattern-specific: fused kernels typically have a
        // small register-pressure penalty.
        let compute_overhead = match opp.fusion_type {
            FusionType::GemmBiasActivation => 0.02,
            FusionType::GemmLayerNorm => 0.05,
            FusionType::ConsecutiveGemm => 0.01,
            FusionType::GemmScale => 0.01,
        };

        Ok(FusedKernelPlan {
            operations,
            memory_saved,
            compute_overhead,
        })
    }

    // -- internal stage builder -----------------------------------------------

    /// Builds execution stages by topological level.
    ///
    /// Nodes are placed in the earliest stage whose predecessors are all in
    /// prior stages. Fused pairs that use stream pipelining get the
    /// `can_overlap_with_next` flag.
    fn build_stages(graph: &GemmGraph, fusible: &[FusiblePair]) -> Vec<FusionStage> {
        let n = graph.len();
        if n == 0 {
            return Vec::new();
        }

        // Compute the topological level (longest path from a root) for each
        // node.
        let mut level = vec![0u32; n];
        for node in graph.nodes() {
            for inp in &node.inputs {
                if let NodeInput::NodeOutput { node_id } = inp {
                    let candidate = level[*node_id] + 1;
                    if candidate > level[node.id] {
                        level[node.id] = candidate;
                    }
                }
            }
        }

        let max_level = level.iter().copied().max().unwrap_or(0);

        // Build stream-pipelining lookup for overlap annotation.
        let stream_pipeline_stages: std::collections::HashSet<u32> = fusible
            .iter()
            .filter(|p| p.strategy == FusionStrategy::StreamPipelining)
            .filter_map(|p| level.get(p.producer).copied())
            .collect();

        let output = graph.output_node();

        let mut stages = Vec::new();
        for lv in 0..=max_level {
            let nodes_in_level: Vec<usize> = (0..n).filter(|&i| level[i] == lv).collect();
            let intermediate_bytes: usize = nodes_in_level
                .iter()
                .filter(|&&nid| Some(nid) != output)
                .map(|&nid| {
                    let nd = &graph.nodes()[nid];
                    nd.m as usize * nd.n as usize * 4
                })
                .sum();

            let can_overlap = stream_pipeline_stages.contains(&lv);

            stages.push(FusionStage {
                stage_index: lv,
                nodes: nodes_in_level,
                can_overlap_with_next: can_overlap,
                intermediate_bytes,
            });
        }

        stages
    }
}

// ---------------------------------------------------------------------------
// Optimal matrix chain ordering (dynamic programming)
// ---------------------------------------------------------------------------

/// Computes the optimal parenthesisation for a matrix chain product.
///
/// `dimensions` is a slice of `(rows, cols)` for each matrix. Adjacent
/// matrices must have compatible inner dimensions (`dims[i].1 == dims[i+1].0`).
///
/// Returns a list of `(i, j)` pairs describing the order of multiplications,
/// where each pair means "multiply the current result of range `[i]` with
/// range `[j]`" from innermost to outermost.
///
/// The algorithm minimises total scalar multiply-add operations (FLOPs).
pub fn optimal_chain_order(dimensions: &[(u32, u32)]) -> BlasResult<Vec<(usize, usize)>> {
    let n = dimensions.len();
    if n < 2 {
        return Err(BlasError::InvalidArgument(
            "optimal_chain_order requires at least 2 matrices".into(),
        ));
    }

    // Validate inner-dimension compatibility.
    for i in 0..n - 1 {
        if dimensions[i].1 != dimensions[i + 1].0 {
            return Err(BlasError::DimensionMismatch(format!(
                "matrix {} cols ({}) != matrix {} rows ({})",
                i,
                dimensions[i].1,
                i + 1,
                dimensions[i + 1].0,
            )));
        }
    }

    // Build the classic DP table.
    // `p` is the dimension array of length n+1 where matrix i has dimensions
    // p[i] × p[i+1].
    let mut p = Vec::with_capacity(n + 1);
    p.push(dimensions[0].0 as u64);
    for d in dimensions {
        p.push(d.1 as u64);
    }

    // cost[i][j] = minimum scalar multiplications for matrices i..=j.
    let mut cost = vec![vec![0u64; n]; n];
    // split[i][j] = optimal split point k such that we multiply
    //               (i..=k) * (k+1..=j).
    let mut split = vec![vec![0usize; n]; n];

    // chain_len = number of matrices in the sub-chain minus 1.
    for chain_len in 1..n {
        for i in 0..n - chain_len {
            let j = i + chain_len;
            cost[i][j] = u64::MAX;
            for k in i..j {
                let q = cost[i][k] + cost[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
                if q < cost[i][j] {
                    cost[i][j] = q;
                    split[i][j] = k;
                }
            }
        }
    }

    // Reconstruct the multiplication order.
    let mut order = Vec::new();
    reconstruct_order(&split, 0, n - 1, &mut order);
    Ok(order)
}

/// Recursively reconstruct the multiplication order from the split table.
fn reconstruct_order(split: &[Vec<usize>], i: usize, j: usize, order: &mut Vec<(usize, usize)>) {
    if i == j {
        return;
    }
    let k = split[i][j];
    reconstruct_order(split, i, k, order);
    reconstruct_order(split, k + 1, j, order);
    order.push((i, j));
}

/// Estimates the total FLOPs (scalar multiply-adds) for a chain product
/// executed in the given `order`.
///
/// `dimensions` has the same meaning as in [`optimal_chain_order`]. `order`
/// is a sequence of `(i, j)` pairs from innermost to outermost (as returned
/// by [`optimal_chain_order`]).
pub fn estimate_chain_flops(
    dimensions: &[(u32, u32)],
    order: &[(usize, usize)],
) -> BlasResult<f64> {
    let n = dimensions.len();
    if n < 2 {
        return Err(BlasError::InvalidArgument(
            "need at least 2 matrices".into(),
        ));
    }

    // Build dimension array.
    let mut p = Vec::with_capacity(n + 1);
    p.push(dimensions[0].0 as f64);
    for d in dimensions {
        p.push(d.1 as f64);
    }

    // Track the effective outer dimensions of each intermediate result.
    // result_rows[i] = rows of intermediate covering matrices[i..=?].
    // result_cols[i] = cols of that intermediate.
    let mut result_rows: Vec<f64> = dimensions.iter().map(|d| d.0 as f64).collect();
    let mut result_cols: Vec<f64> = dimensions.iter().map(|d| d.1 as f64).collect();

    let mut total = 0.0f64;

    for &(i, j) in order {
        if i == j {
            continue;
        }
        // When we multiply the sub-chain [i..=k] with [k+1..=j], the cost is
        // rows_i * cols_j * inner, where inner = cols of [i..=k] = rows of
        // [k+1..=j]. After the split table, the product of [i..=j] has
        // dimensions rows_i × cols_j.
        //
        // We approximate using the dimension-array representation:
        // cost = p[i] * p[split+1] * p[j+1], but here we just use the
        // tracked result dimensions.
        let rows = result_rows[i];
        let cols = result_cols[j];
        // Inner dimension: the first sub-product ends at some column which
        // equals the left sub-product's cols. For a two-matrix product
        // this is p[k+1]. We use the dp dimension array directly:
        // for (i, j) the flops are p[i] * (some inner) * p[j+1].
        // Since we've already computed them via DP, use the DP formula.
        let inner = p[split_point_for(i, j, dimensions)];
        total += 2.0 * rows * cols * inner; // 2× for multiply + add

        // Update tracked dimensions: the result of [i..=j] is rows_i × cols_j.
        result_rows[j] = rows;
        result_cols[i] = cols;
    }

    Ok(total)
}

/// Re-derive the split point for a given (i, j) pair so we can look up the
/// inner dimension. This re-runs the DP for the sub-problem.
fn split_point_for(i: usize, j: usize, dimensions: &[(u32, u32)]) -> usize {
    if j == i + 1 {
        // Exactly two matrices: inner dim is dimensions[i].1 (== dimensions[j].0).
        return i + 1; // index into p: p[i+1]
    }

    let n = dimensions.len();
    let mut p = Vec::with_capacity(n + 1);
    p.push(dimensions[0].0 as u64);
    for d in dimensions {
        p.push(d.1 as u64);
    }

    let mut cost = vec![vec![0u64; n]; n];
    let mut split = vec![vec![0usize; n]; n];

    for chain_len in 1..n {
        for ii in 0..n - chain_len {
            let jj = ii + chain_len;
            cost[ii][jj] = u64::MAX;
            for k in ii..jj {
                let q = cost[ii][k] + cost[k + 1][jj] + p[ii] * p[k + 1] * p[jj + 1];
                if q < cost[ii][jj] {
                    cost[ii][jj] = q;
                    split[ii][jj] = k;
                }
            }
        }
    }

    split[i][j] + 1 // index into p
}

/// A simpler FLOP estimator that uses the DP cost table directly.
///
/// Returns the minimum total scalar multiplications (not multiply-adds)
/// for the chain product.
pub fn minimum_chain_flops(dimensions: &[(u32, u32)]) -> BlasResult<u64> {
    let n = dimensions.len();
    if n < 2 {
        return Err(BlasError::InvalidArgument(
            "need at least 2 matrices".into(),
        ));
    }

    for i in 0..n - 1 {
        if dimensions[i].1 != dimensions[i + 1].0 {
            return Err(BlasError::DimensionMismatch(format!(
                "matrix {} cols ({}) != matrix {} rows ({})",
                i,
                dimensions[i].1,
                i + 1,
                dimensions[i + 1].0,
            )));
        }
    }

    let mut p = Vec::with_capacity(n + 1);
    p.push(dimensions[0].0 as u64);
    for d in dimensions {
        p.push(d.1 as u64);
    }

    let mut cost = vec![vec![0u64; n]; n];

    for chain_len in 1..n {
        for i in 0..n - chain_len {
            let j = i + chain_len;
            cost[i][j] = u64::MAX;
            for k in i..j {
                let q = cost[i][k] + cost[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
                if q < cost[i][j] {
                    cost[i][j] = q;
                }
            }
        }
    }

    Ok(cost[0][n - 1])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- graph construction --------------------------------------------------

    #[test]
    fn empty_graph() {
        let g = GemmGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert_eq!(g.output_node(), None);
    }

    #[test]
    fn single_gemm_node() {
        let mut g = GemmGraph::new();
        let id = g.add_gemm(
            128,
            256,
            64,
            NodeInput::External { name: "A".into() },
            NodeInput::External { name: "B".into() },
        );
        assert_eq!(id, 0);
        assert_eq!(g.len(), 1);
        assert_eq!(g.nodes()[0].m, 128);
        assert_eq!(g.nodes()[0].n, 256);
        assert_eq!(g.nodes()[0].k, 64);
        assert!(matches!(g.nodes()[0].op, GemmOp::Gemm));
    }

    #[test]
    fn chain_two_matrices() {
        // A(10×30) * B(30×5) => single GEMM node.
        let g = GemmGraph::chain(&[(10, 30), (30, 5)]);
        assert!(g.is_ok());
        let g = g.ok().filter(|_| true).unwrap_or_default();
        assert_eq!(g.len(), 1);
        assert_eq!(g.output_node(), Some(0));
        let n = &g.nodes()[0];
        assert_eq!((n.m, n.n, n.k), (10, 5, 30));
    }

    #[test]
    fn chain_three_matrices() {
        // A(10×30) * B(30×5) * C(5×60)
        let g = GemmGraph::chain(&[(10, 30), (30, 5), (5, 60)]);
        assert!(g.is_ok());
        let g = g.ok().filter(|_| true).unwrap_or_default();
        assert_eq!(g.len(), 2);
        assert_eq!(g.output_node(), Some(1));
        // First node: 10×5, k=30.
        assert_eq!(
            (g.nodes()[0].m, g.nodes()[0].n, g.nodes()[0].k),
            (10, 5, 30)
        );
        // Second node: 10×60, k=5.
        assert_eq!(
            (g.nodes()[1].m, g.nodes()[1].n, g.nodes()[1].k),
            (10, 60, 5)
        );
    }

    #[test]
    fn chain_dimension_mismatch() {
        let r = GemmGraph::chain(&[(10, 30), (20, 5)]); // 30 != 20
        assert!(r.is_err());
    }

    #[test]
    fn chain_too_few_matrices() {
        let r = GemmGraph::chain(&[(10, 30)]);
        assert!(r.is_err());
    }

    #[test]
    fn add_scale_node() {
        let mut g = GemmGraph::new();
        let id0 = g.add_gemm(
            64,
            64,
            32,
            NodeInput::External { name: "A".into() },
            NodeInput::External { name: "B".into() },
        );
        let id1 = g.add_scale(id0, 2.5);
        assert!(id1.is_ok());
        let id1 = id1.unwrap_or(0);
        assert_eq!(g.nodes()[id1].m, 64);
        assert_eq!(g.nodes()[id1].n, 64);
        assert!(
            matches!(g.nodes()[id1].op, GemmOp::Scale { alpha } if (alpha - 2.5).abs() < 1e-12)
        );
    }

    #[test]
    fn set_output_invalid() {
        let mut g = GemmGraph::new();
        assert!(g.set_output(0).is_err());
    }

    // -- optimal chain ordering (DP) -----------------------------------------

    #[test]
    fn optimal_order_three_matrices() {
        // Classic textbook: A(10×30), B(30×5), C(5×60).
        // Naïve left-to-right: (A*B)*C => 10*30*5 + 10*5*60 = 1500+3000 = 4500.
        // Optimal: A*(B*C) => 30*5*60 + 10*30*60 = 9000+18000 = 27000 — no,
        // that's worse. Actually left-to-right IS optimal here: 4500.
        let dims = [(10, 30), (30, 5), (5, 60)];
        let order = optimal_chain_order(&dims);
        assert!(order.is_ok());
        let cost = minimum_chain_flops(&dims);
        assert!(cost.is_ok());
        assert_eq!(cost.unwrap_or(0), 4500);
    }

    #[test]
    fn optimal_order_four_matrices() {
        // A(40×20), B(20×30), C(30×10), D(10×30).
        // Optimal cost = 40*20*10 + 40*10*30 + 20*30*10 = 8000+12000+6000=26000.
        let dims = [(40, 20), (20, 30), (30, 10), (10, 30)];
        let cost = minimum_chain_flops(&dims);
        assert!(cost.is_ok());
        let c = cost.unwrap_or(0);
        // The DP should find something <= the naïve cost.
        // Naïve left-to-right: 40*20*30 + 40*30*10 + 40*10*30
        //   = 24000 + 12000 + 12000 = 48000.
        assert!(c < 48000, "optimal cost {c} should be < 48000");
        assert_eq!(c, 26000);
    }

    #[test]
    fn optimal_order_dimension_mismatch() {
        let r = optimal_chain_order(&[(10, 30), (20, 5)]);
        assert!(r.is_err());
    }

    #[test]
    fn optimal_order_too_few() {
        let r = optimal_chain_order(&[(10, 30)]);
        assert!(r.is_err());
    }

    // -- fusion pair detection -----------------------------------------------

    #[test]
    fn fusible_pairs_linear_chain() {
        // A * B * C => two GEMM nodes, first feeds second.
        let g = GemmGraph::chain(&[(64, 64), (64, 64), (64, 64)]);
        let g = g.ok().filter(|_| true).unwrap_or_default();
        let pairs = FusionPass::find_fusible_pairs(&g);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].producer, 0);
        assert_eq!(pairs[0].consumer, 1);
    }

    #[test]
    fn fusible_pairs_no_fusion_fanout() {
        // Producer has two consumers => not fusible.
        let mut g = GemmGraph::new();
        let n0 = g.add_gemm(
            64,
            64,
            64,
            NodeInput::External { name: "A".into() },
            NodeInput::External { name: "B".into() },
        );
        let _n1 = g.add_gemm(
            64,
            32,
            64,
            NodeInput::NodeOutput { node_id: n0 },
            NodeInput::External { name: "C".into() },
        );
        let _n2 = g.add_gemm(
            64,
            16,
            64,
            NodeInput::NodeOutput { node_id: n0 },
            NodeInput::External { name: "D".into() },
        );
        let pairs = FusionPass::find_fusible_pairs(&g);
        assert!(pairs.is_empty());
    }

    // -- memory estimation ---------------------------------------------------

    #[test]
    fn intermediate_memory_chain() {
        // A(64×64) * B(64×64) * C(64×64).
        // Node 0: 64×64 intermediate (not output). Node 1: output.
        let g = GemmGraph::chain(&[(64, 64), (64, 64), (64, 64)]);
        let g = g.ok().filter(|_| true).unwrap_or_default();
        let mem = FusionPass::estimate_intermediate_memory(&g);
        // Node 0 = 64*64*4 = 16384 bytes. Node 1 is output => excluded.
        assert_eq!(mem, 64 * 64 * 4);
    }

    #[test]
    fn memory_savings_with_fusion() {
        let g = GemmGraph::chain(&[(64, 64), (64, 64), (64, 64)]);
        let g = g.ok().filter(|_| true).unwrap_or_default();
        let plan = FusionPass::analyze(&g);
        assert!(plan.is_ok());
        let plan = plan.unwrap_or_else(|_| FusionPlan {
            stages: Vec::new(),
            fused_pairs: Vec::new(),
            total_gemm_calls: 0,
            original_gemm_calls: 0,
        });
        let savings = FusionPass::memory_savings(&g, &plan);
        // The single intermediate should be eliminated by epilogue fusion.
        assert!(savings > 0.0, "expected positive savings, got {savings}");
    }

    // -- stage planning ------------------------------------------------------

    #[test]
    fn stages_linear_chain() {
        let g = GemmGraph::chain(&[(128, 64), (64, 32), (32, 256)]);
        let g = g.ok().filter(|_| true).unwrap_or_default();
        let plan = FusionPass::analyze(&g);
        assert!(plan.is_ok());
        let plan = plan.unwrap_or_else(|_| FusionPlan {
            stages: Vec::new(),
            fused_pairs: Vec::new(),
            total_gemm_calls: 0,
            original_gemm_calls: 0,
        });
        // Two GEMM nodes => two stages (topological levels 0 and 1).
        assert_eq!(plan.stages.len(), 2);
        assert_eq!(plan.stages[0].nodes, vec![0]);
        assert_eq!(plan.stages[1].nodes, vec![1]);
        assert_eq!(plan.original_gemm_calls, 2);
    }

    // -- Display impl --------------------------------------------------------

    #[test]
    fn display_fusion_plan() {
        let g = GemmGraph::chain(&[(64, 64), (64, 64), (64, 64)]);
        let g = g.ok().filter(|_| true).unwrap_or_default();
        let plan = FusionPass::analyze(&g);
        assert!(plan.is_ok());
        let plan = plan.unwrap_or_else(|_| FusionPlan {
            stages: Vec::new(),
            fused_pairs: Vec::new(),
            total_gemm_calls: 0,
            original_gemm_calls: 0,
        });
        let display = format!("{plan}");
        assert!(display.contains("FusionPlan"));
        assert!(display.contains("stage"));
        assert!(display.contains("GEMM calls"));
    }

    #[test]
    fn analyze_empty_graph_errors() {
        let g = GemmGraph::new();
        assert!(FusionPass::analyze(&g).is_err());
    }

    // -- estimate_chain_flops ------------------------------------------------

    #[test]
    fn chain_flops_two_matrices() {
        let dims = [(10, 30), (30, 5)];
        let order = optimal_chain_order(&dims);
        assert!(order.is_ok());
        let order = order.unwrap_or_default();
        let flops = estimate_chain_flops(&dims, &order);
        assert!(flops.is_ok());
        // 2 * 10 * 30 * 5 = 3000.
        let f = flops.unwrap_or(0.0);
        assert!((f - 3000.0).abs() < 1e-6, "expected 3000, got {f}");
    }

    // -- FusionType / FusionOpportunity / FusedKernelPlan -------------------

    #[test]
    fn test_fusion_pass_identifies_gemm_bias_relu() {
        // Builds a graph: GEMM (node 0) → Add (node 2), where the Add's
        // second operand is a Scale node (node 1) acting as a bias proxy.
        // The Add is the sole consumer of the GEMM, so GemmBiasActivation
        // should be detected.
        let mut g = GemmGraph::new();
        let gemm_id = g.add_gemm(
            128,
            256,
            64,
            NodeInput::External { name: "A".into() },
            NodeInput::External { name: "B".into() },
        );
        // Create a second GEMM-output node that acts as a bias source (external path).
        let bias_gemm_id = g.add_gemm(
            128,
            256,
            1,
            NodeInput::External {
                name: "bias".into(),
            },
            NodeInput::External {
                name: "ones".into(),
            },
        );
        let add_id = g.add_add(gemm_id, bias_gemm_id).unwrap_or(gemm_id);
        g.set_output(add_id).unwrap_or_default();

        let opps = FusionPass::analyze_opportunities(&g, 0.0);
        // The GEMM→Add chain must produce a GemmBiasActivation opportunity.
        assert!(
            opps.iter()
                .any(|o| o.fusion_type == FusionType::GemmBiasActivation),
            "expected GemmBiasActivation opportunity, got: {opps:?}"
        );
    }

    #[test]
    fn test_fusion_pass_identifies_gemm_scale() {
        // GEMM → Scale: should produce a GemmScale opportunity.
        let mut g = GemmGraph::new();
        let gemm_id = g.add_gemm(
            128,
            256,
            64,
            NodeInput::External { name: "A".into() },
            NodeInput::External { name: "B".into() },
        );
        let scale_id = g.add_scale(gemm_id, 0.5).unwrap_or(gemm_id);
        g.set_output(scale_id).unwrap_or_default();

        let opps = FusionPass::analyze_opportunities(&g, 0.0);
        assert!(!opps.is_empty(), "expected at least one opportunity");
        assert!(
            opps.iter().any(|o| o.fusion_type == FusionType::GemmScale),
            "expected GemmScale opportunity"
        );
    }

    #[test]
    fn test_fusion_pass_identifies_consecutive_gemm() {
        // Linear chain: GEMM → GEMM — should detect ConsecutiveGemm.
        let g = GemmGraph::chain(&[(64, 64), (64, 64), (64, 64)]).unwrap_or_default();

        let opps = FusionPass::analyze_opportunities(&g, 0.0);
        assert!(!opps.is_empty(), "expected ConsecutiveGemm opportunity");
        assert!(
            opps.iter()
                .any(|o| o.fusion_type == FusionType::ConsecutiveGemm),
            "expected at least one ConsecutiveGemm opportunity"
        );
    }

    #[test]
    fn test_fusion_pass_no_opportunities_for_standalone_gemm() {
        // Single GEMM with no consumers → no fusion opportunity.
        let mut g = GemmGraph::new();
        let id = g.add_gemm(
            64,
            64,
            64,
            NodeInput::External { name: "A".into() },
            NodeInput::External { name: "B".into() },
        );
        g.set_output(id).unwrap_or_default();

        let opps = FusionPass::analyze_opportunities(&g, 0.0);
        assert!(
            opps.is_empty(),
            "expected no opportunities for standalone GEMM"
        );
    }

    #[test]
    fn test_fusion_speedup_large_mn_is_higher_than_small() {
        // Large MN (memory-bandwidth bound) should give higher speedup than
        // small MN (compute bound).
        let large = FusionPass::estimate_speedup(&FusionType::GemmBiasActivation, 1024, 1024, 64);
        let small = FusionPass::estimate_speedup(&FusionType::GemmBiasActivation, 16, 16, 1024);
        assert!(
            large > small,
            "large-MN speedup {large} should exceed small-MN speedup {small}"
        );
    }

    #[test]
    fn test_fused_plan_memory_savings_nonzero() {
        // GEMM → Scale: the producer's M×N f32 buffer should be saved.
        let mut g = GemmGraph::new();
        let gemm_id = g.add_gemm(
            128,
            256,
            64,
            NodeInput::External { name: "A".into() },
            NodeInput::External { name: "B".into() },
        );
        let scale_id = g.add_scale(gemm_id, 2.0).unwrap_or(gemm_id);
        g.set_output(scale_id).unwrap_or_default();

        let opps = FusionPass::analyze_opportunities(&g, 0.0);
        assert!(!opps.is_empty());

        let plan = FusionPass::apply(&g, &opps[0]);
        assert!(plan.is_ok(), "apply should succeed: {:?}", plan.err());
        let plan = plan.unwrap_or_else(|_| FusedKernelPlan {
            operations: vec![],
            memory_saved: 0,
            compute_overhead: 0.0,
        });
        // 128 × 256 × 4 bytes = 131_072
        assert_eq!(plan.memory_saved, 128 * 256 * 4);
    }

    #[test]
    fn test_fusion_type_display() {
        assert_eq!(
            FusionType::GemmBiasActivation.to_string(),
            "GemmBiasActivation"
        );
        assert_eq!(FusionType::GemmLayerNorm.to_string(), "GemmLayerNorm");
        assert_eq!(FusionType::ConsecutiveGemm.to_string(), "ConsecutiveGemm");
        assert_eq!(FusionType::GemmScale.to_string(), "GemmScale");
    }

    #[test]
    fn test_apply_invalid_node_id_errors() {
        let g = GemmGraph::new();
        let opp = FusionOpportunity {
            node_ids: vec![99],
            estimated_speedup: 1.5,
            fusion_type: FusionType::GemmScale,
        };
        assert!(FusionPass::apply(&g, &opp).is_err());
    }

    #[test]
    fn test_apply_produces_fused_gemm_epilogue() {
        let mut g = GemmGraph::new();
        let gemm_id = g.add_gemm(
            64,
            64,
            32,
            NodeInput::External { name: "A".into() },
            NodeInput::External { name: "B".into() },
        );
        let scale_id = g.add_scale(gemm_id, 1.5).unwrap_or(gemm_id);
        g.set_output(scale_id).unwrap_or_default();

        let opps = FusionPass::analyze_opportunities(&g, 0.0);
        assert!(!opps.is_empty());

        let plan = FusionPass::apply(&g, &opps[0]).unwrap_or_else(|_| FusedKernelPlan {
            operations: vec![],
            memory_saved: 0,
            compute_overhead: 0.0,
        });

        // The first operation should be a FusedGemmEpilogue.
        assert!(
            matches!(
                plan.operations.first(),
                Some(FusedOp::FusedGemmEpilogue { .. })
            ),
            "expected FusedGemmEpilogue as first operation"
        );
    }

    #[test]
    fn test_gemm_layernorm_speedup() {
        let speedup = FusionPass::estimate_speedup(&FusionType::GemmLayerNorm, 512, 512, 256);
        // Should be exactly 1.3 regardless of dimensions.
        assert!((speedup - 1.3).abs() < 1e-6, "expected 1.3, got {speedup}");
    }
}
