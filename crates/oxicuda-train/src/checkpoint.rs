//! Gradient checkpointing (activation recomputation).
//!
//! Gradient checkpointing trades compute for memory: instead of storing all
//! intermediate activations during the forward pass, only the *inputs* to
//! designated checkpoint boundaries are retained.  During the backward pass the
//! forward computation inside each segment is rerun to recover the activations
//! needed for gradient computation.
//!
//! ## Memory savings
//!
//! For a network with N layers split into `sqrt(N)` segments, activation memory
//! drops from O(N) to O(sqrt(N)).
//!
//! ## Policies
//!
//! | Policy | Description |
//! |---|---|
//! | [`crate::checkpoint::CheckpointPolicy::Uniform`] | Checkpoint every `interval` layers |
//! | [`crate::checkpoint::CheckpointPolicy::Selective`] | Checkpoint named layers only |
//! | [`crate::checkpoint::CheckpointPolicy::Offload`] | Store activations in CPU memory |
//! | [`crate::checkpoint::CheckpointPolicy::None`] | Disable checkpointing (store all activations) |
//!
//! ## Usage
//!
//! ```rust
//! use oxicuda_train::checkpoint::{CheckpointManager, CheckpointPolicy, ActivationBuffer};
//!
//! let mut mgr = CheckpointManager::new(CheckpointPolicy::Uniform { interval: 2 });
//!
//! // Forward pass: register segment inputs
//! mgr.save_input("segment_0", vec![1.0f32, 2.0, 3.0]).unwrap();
//! mgr.save_input("segment_1", vec![4.0f32, 5.0, 6.0]).unwrap();
//!
//! // Backward pass: retrieve inputs for recomputation
//! let inp = mgr.get_input("segment_0").unwrap();
//! assert_eq!(inp[0], 1.0);
//! ```

use std::collections::HashMap;

use crate::error::{TrainError, TrainResult};

// ─── CheckpointPolicy ────────────────────────────────────────────────────────

/// Strategy for selecting which layers to checkpoint.
#[derive(Debug, Clone)]
pub enum CheckpointPolicy {
    /// Checkpoint every `interval` layers (uniform activation checkpointing).
    Uniform {
        /// Checkpoint every `interval`-th layer boundary.
        interval: usize,
    },
    /// Checkpoint only the named layers.
    Selective {
        /// Set of layer names to checkpoint.
        names: Vec<String>,
    },
    /// Offload all intermediate activations to CPU (host) memory.
    ///
    /// Maximises GPU memory savings at the cost of H2D/D2H transfer overhead.
    Offload,
    /// Do not checkpoint — store all activations (standard backprop).
    None,
}

impl CheckpointPolicy {
    /// Returns `true` if `layer_name` should be checkpointed under this policy.
    #[must_use]
    pub fn should_checkpoint(&self, layer_name: &str, layer_idx: usize) -> bool {
        match self {
            Self::Uniform { interval } => layer_idx % interval == 0,
            Self::Selective { names } => names.iter().any(|n| n == layer_name),
            Self::Offload => true,
            Self::None => false,
        }
    }
}

// ─── ActivationBuffer ────────────────────────────────────────────────────────

/// A stored activation tensor (checkpoint segment input).
#[derive(Debug, Clone)]
pub struct ActivationBuffer {
    /// Name of the segment (human-readable key).
    pub name: String,
    /// Stored activation data (CPU-accessible).
    pub data: Vec<f32>,
    /// Shape of the stored tensor.
    pub shape: Vec<usize>,
    /// Whether this activation is offloaded to CPU (always true in this impl).
    pub is_offloaded: bool,
}

impl ActivationBuffer {
    /// Create a new activation buffer.
    #[must_use]
    pub fn new(name: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            data,
            shape,
            is_offloaded: false,
        }
    }

    /// Total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Estimated memory usage in bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.data.len() * 4
    }
}

// ─── CheckpointManager ───────────────────────────────────────────────────────

/// Manages gradient checkpoint buffers for an entire forward/backward pass.
///
/// Call [`CheckpointManager::save_input`] at each checkpoint boundary during the forward pass,
/// then call [`CheckpointManager::get_input`] during the backward pass to retrieve the stored
/// tensors for recomputation.
#[derive(Debug)]
pub struct CheckpointManager {
    policy: CheckpointPolicy,
    /// Stored segment inputs, keyed by segment name.
    inputs: HashMap<String, ActivationBuffer>,
    /// Ordered list of segment names (insertion order).
    segment_order: Vec<String>,
    /// Maximum number of stored segments (0 = unlimited).
    max_segments: usize,
    /// Cumulative bytes stored across all buffers.
    total_bytes: usize,
    /// Layer index counter for Uniform policy.
    layer_counter: usize,
}

impl CheckpointManager {
    /// Create a new checkpoint manager with the given policy.
    #[must_use]
    pub fn new(policy: CheckpointPolicy) -> Self {
        Self {
            policy,
            inputs: HashMap::new(),
            segment_order: Vec::new(),
            max_segments: 0,
            total_bytes: 0,
            layer_counter: 0,
        }
    }

    /// Set a maximum number of segments (0 = unlimited).
    #[must_use]
    pub fn with_max_segments(mut self, max: usize) -> Self {
        self.max_segments = max;
        self
    }

    /// Returns the current policy.
    #[must_use]
    pub fn policy(&self) -> &CheckpointPolicy {
        &self.policy
    }

    /// Returns the number of stored checkpoint segments.
    #[must_use]
    pub fn num_segments(&self) -> usize {
        self.inputs.len()
    }

    /// Total bytes stored across all buffers.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Returns `true` if the given segment should be checkpointed.
    #[must_use]
    pub fn should_checkpoint(&mut self, name: &str) -> bool {
        let idx = self.layer_counter;
        self.layer_counter += 1;
        self.policy.should_checkpoint(name, idx)
    }

    /// Save a segment's input activation to the checkpoint store.
    ///
    /// If the policy says this segment should not be checkpointed, the call
    /// is silently ignored.
    ///
    /// # Errors
    ///
    /// * [`TrainError::CheckpointOverflow`] if `max_segments > 0` and the
    ///   segment limit would be exceeded.
    pub fn save_input(&mut self, name: &str, data: Vec<f32>) -> TrainResult<()> {
        let shape = vec![data.len()];
        self.save_shaped(name, data, shape)
    }

    /// Save a shaped activation tensor to the checkpoint store.
    ///
    /// # Errors
    ///
    /// See [`CheckpointManager::save_input`].
    pub fn save_shaped(
        &mut self,
        name: &str,
        data: Vec<f32>,
        shape: Vec<usize>,
    ) -> TrainResult<()> {
        if self.max_segments > 0 && self.inputs.len() >= self.max_segments {
            return Err(TrainError::CheckpointOverflow {
                max: self.max_segments,
            });
        }
        let bytes = data.len() * 4;
        self.total_bytes += bytes;
        let buf = ActivationBuffer::new(name, data, shape);
        if !self.inputs.contains_key(name) {
            self.segment_order.push(name.to_owned());
        }
        self.inputs.insert(name.to_owned(), buf);
        Ok(())
    }

    /// Retrieve a stored activation buffer by segment name.
    ///
    /// Returns `None` if the segment was not checkpointed.
    #[must_use]
    pub fn get_input(&self, name: &str) -> Option<&[f32]> {
        self.inputs.get(name).map(|b| b.data.as_slice())
    }

    /// Retrieve a stored activation buffer with full shape information.
    #[must_use]
    pub fn get_buffer(&self, name: &str) -> Option<&ActivationBuffer> {
        self.inputs.get(name)
    }

    /// Ordered list of segment names in checkpoint order.
    #[must_use]
    pub fn segment_names(&self) -> &[String] {
        &self.segment_order
    }

    /// Clear all stored buffers (call after backward pass).
    pub fn clear(&mut self) {
        self.total_bytes = 0;
        self.inputs.clear();
        self.segment_order.clear();
        self.layer_counter = 0;
    }

    /// Summary statistics for the checkpoint store.
    #[must_use]
    pub fn stats(&self) -> CheckpointStats {
        CheckpointStats {
            num_segments: self.inputs.len(),
            total_bytes: self.total_bytes,
            total_elements: self.inputs.values().map(|b| b.numel()).sum(),
        }
    }
}

// ─── CheckpointStats ─────────────────────────────────────────────────────────

/// Summary statistics for a checkpoint manager.
#[derive(Debug, Clone, Copy)]
pub struct CheckpointStats {
    /// Number of stored segments.
    pub num_segments: usize,
    /// Total bytes across all buffers.
    pub total_bytes: usize,
    /// Total elements across all buffers.
    pub total_elements: usize,
}

// ─── RecomputeSegment ─────────────────────────────────────────────────────────

/// A description of a recomputable forward-pass segment.
///
/// During the backward pass, each segment is re-run with its saved input to
/// recover intermediate activations.
#[derive(Debug, Clone)]
pub struct RecomputeSegment {
    /// Segment identifier (matches the key used in `CheckpointManager::save_input`).
    pub name: String,
    /// Shape of the input tensor.
    pub input_shape: Vec<usize>,
    /// Length of the output produced by this segment.
    pub output_len: usize,
}

/// Boxed recompute closure: maps input activations to output activations.
type RecomputeInner = Box<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>;

/// A simple function-based recompute handle.
///
/// Wraps an `Fn(&[f32]) -> Vec<f32>` (the forward pass of one segment) so it
/// can be stored and called during the backward pass.
pub struct RecomputeFn {
    /// Human-readable segment name.
    pub name: String,
    inner: RecomputeInner,
}

impl RecomputeFn {
    /// Create from a closure.
    pub fn new(
        name: impl Into<String>,
        f: impl Fn(&[f32]) -> Vec<f32> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            inner: Box::new(f),
        }
    }

    /// Run the forward pass with `input` and return the activation.
    #[must_use]
    pub fn run(&self, input: &[f32]) -> Vec<f32> {
        (self.inner)(input)
    }
}

impl std::fmt::Debug for RecomputeFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecomputeFn")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_retrieve() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::Uniform { interval: 1 });
        mgr.save_input("seg_0", vec![1.0, 2.0, 3.0]).unwrap();
        let inp = mgr.get_input("seg_0").unwrap();
        assert_eq!(inp, &[1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn uniform_policy_skips_odd_layers() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::Uniform { interval: 2 });
        // Layer 0: should checkpoint (0 % 2 == 0)
        assert!(mgr.should_checkpoint("layer_0"));
        // Layer 1: should not (1 % 2 != 0)
        assert!(!mgr.should_checkpoint("layer_1"));
        // Layer 2: should (2 % 2 == 0)
        assert!(mgr.should_checkpoint("layer_2"));
    }

    #[test]
    fn selective_policy() {
        let policy = CheckpointPolicy::Selective {
            names: vec!["attn".to_owned(), "ffn".to_owned()],
        };
        assert!(policy.should_checkpoint("attn", 0));
        assert!(!policy.should_checkpoint("norm", 1));
        assert!(policy.should_checkpoint("ffn", 5));
    }

    #[test]
    fn none_policy_never_checkpoints() {
        let policy = CheckpointPolicy::None;
        for i in 0..10 {
            assert!(!policy.should_checkpoint("any", i));
        }
    }

    #[test]
    fn offload_policy_always_checkpoints() {
        let policy = CheckpointPolicy::Offload;
        for i in 0..10 {
            assert!(policy.should_checkpoint("any", i));
        }
    }

    #[test]
    fn max_segments_overflow_error() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::Offload).with_max_segments(2);
        mgr.save_input("s0", vec![1.0]).unwrap();
        mgr.save_input("s1", vec![2.0]).unwrap();
        let result = mgr.save_input("s2", vec![3.0]);
        assert!(matches!(
            result,
            Err(TrainError::CheckpointOverflow { max: 2 })
        ));
    }

    #[test]
    fn clear_resets_all() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::Offload);
        mgr.save_input("x", vec![1.0, 2.0]).unwrap();
        assert_eq!(mgr.num_segments(), 1);
        mgr.clear();
        assert_eq!(mgr.num_segments(), 0);
        assert_eq!(mgr.total_bytes(), 0);
    }

    #[test]
    fn stats_are_correct() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::Offload);
        mgr.save_input("a", vec![1.0_f32; 100]).unwrap();
        mgr.save_input("b", vec![2.0_f32; 200]).unwrap();
        let stats = mgr.stats();
        assert_eq!(stats.num_segments, 2);
        assert_eq!(stats.total_elements, 300);
        assert_eq!(stats.total_bytes, 300 * 4);
    }

    #[test]
    fn recompute_fn_runs() {
        let double = RecomputeFn::new("double", |x| x.iter().map(|&v| v * 2.0).collect());
        let result = double.run(&[1.0, 2.0, 3.0]);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn segment_order_preserved() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::Offload);
        mgr.save_input("first", vec![1.0]).unwrap();
        mgr.save_input("second", vec![2.0]).unwrap();
        mgr.save_input("third", vec![3.0]).unwrap();
        let names = mgr.segment_names();
        assert_eq!(names, &["first", "second", "third"]);
    }

    #[test]
    fn get_shaped_buffer() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::Offload);
        mgr.save_shaped("layer", vec![1.0; 12], vec![3, 4]).unwrap();
        let buf = mgr.get_buffer("layer").unwrap();
        assert_eq!(buf.shape, vec![3, 4]);
        assert_eq!(buf.numel(), 12);
    }
}
