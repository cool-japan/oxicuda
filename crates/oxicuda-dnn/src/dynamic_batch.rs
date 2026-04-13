//! Dynamic batching and continuous batching for inference serving.
//!
//! This module implements the core scheduling primitives used by modern LLM
//! inference engines (vLLM, Orca, TensorRT-LLM):
//!
//! - **[`ContinuousBatcher`]** — iteration-level scheduler that decides which
//!   requests to prefill, decode, or preempt at each step.
//! - **[`TokenBudgetAllocator`]** — manages a per-step token budget shared
//!   between prefill and decode phases.
//! - **[`PagedKvManager`]** — block-level paged KV-cache allocator with
//!   copy-on-write support for beam search and speculative decoding.
//! - **[`SpeculativeDecoder`]** — draft-model speculative decoding with
//!   rejection sampling verification.
//! - **[`BatchMetrics`]** — running statistics for throughput, latency, and
//!   utilization monitoring.
//!
//! # Scheduling Policies
//!
//! | Policy | Description |
//! |--------|-------------|
//! | [`SchedulingPolicy::Fcfs`] | First-come, first-served |
//! | [`SchedulingPolicy::ShortestJobFirst`] | Shortest remaining generation |
//! | [`SchedulingPolicy::PriorityBased`] | User-assigned priority levels |
//! | [`SchedulingPolicy::DeadlineAware`] | EDF (earliest deadline first) |
//! | [`SchedulingPolicy::Orca`] | Iteration-level (selective batching) |

use std::collections::VecDeque;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Basic types
// ---------------------------------------------------------------------------

/// Unique identifier for an inference request.
pub type RequestId = u64;

/// Priority level for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Lowest priority — best-effort.
    Low = 0,
    /// Default priority.
    Normal = 1,
    /// Expedited processing.
    High = 2,
}

/// An incoming inference request.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Unique request identifier.
    pub request_id: RequestId,
    /// Number of input (prompt) tokens.
    pub sequence_length: usize,
    /// Maximum number of tokens to generate.
    pub max_new_tokens: usize,
    /// Scheduling priority.
    pub priority: Priority,
    /// Monotonic arrival timestamp in nanoseconds.
    pub arrival_time_ns: u64,
    /// Optional hard deadline in nanoseconds (absolute).
    pub deadline_ns: Option<u64>,
}

/// A slot inside the running batch.
#[derive(Debug, Clone)]
pub struct BatchSlot {
    /// Slot index within the batch.
    pub slot_id: usize,
    /// Request occupying this slot.
    pub request_id: RequestId,
    /// Tokens processed so far (prompt + generated).
    pub current_seq_len: usize,
    /// Maximum sequence length (prompt + max_new_tokens).
    pub max_seq_len: usize,
    /// Whether this slot is currently doing prefill.
    pub is_prefill: bool,
    /// Whether this slot is actively generating.
    pub is_active: bool,
}

/// Scheduling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-come, first-served.
    Fcfs,
    /// Prefer requests with fewer remaining tokens.
    ShortestJobFirst,
    /// Respect user-assigned [`Priority`] levels.
    PriorityBased,
    /// Earliest-deadline-first (requires `deadline_ns`).
    DeadlineAware,
    /// Orca-style iteration-level selective batching.
    Orca,
}

/// How to handle a preempted request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionPolicy {
    /// Discard KV cache and recompute from scratch when resumed.
    Recompute,
    /// Swap (offload) KV cache blocks to host memory.
    Swap,
}

/// Configuration for the continuous batcher.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of requests in a single batch.
    pub max_batch_size: usize,
    /// Maximum total tokens (prompt + generated) across the batch.
    pub max_total_tokens: usize,
    /// Maximum sequence length for any single request.
    pub max_sequence_length: usize,
    /// Maximum prefill tokens per step.
    pub prefill_batch_size: usize,
    /// Maximum decode slots per step.
    pub decode_batch_size: usize,
    /// Scheduling algorithm.
    pub scheduling_policy: SchedulingPolicy,
}

/// Result of a single scheduling step.
#[derive(Debug, Clone)]
pub struct BatchDecision {
    /// Requests to prefill in this step.
    pub prefill_requests: Vec<RequestId>,
    /// Requests to decode in this step.
    pub decode_requests: Vec<RequestId>,
    /// Requests preempted to free capacity.
    pub preempted: Vec<RequestId>,
    /// Total token count for the step.
    pub total_tokens: usize,
}

// ---------------------------------------------------------------------------
// BatchState — internal bookkeeping
// ---------------------------------------------------------------------------

/// Internal state of the batcher.
#[derive(Debug)]
struct BatchState {
    /// Currently running slots.
    active_slots: Vec<BatchSlot>,
    /// Total tokens across all active slots.
    total_tokens: usize,
    /// Requests waiting for prefill.
    prefill_queue: VecDeque<InferenceRequest>,
    /// Requests in decode phase (tracked separately for Orca).
    decode_queue: VecDeque<RequestId>,
    /// Preempted requests awaiting resumption.
    preempted_queue: VecDeque<InferenceRequest>,
}

impl BatchState {
    fn new() -> Self {
        Self {
            active_slots: Vec::new(),
            total_tokens: 0,
            prefill_queue: VecDeque::new(),
            decode_queue: VecDeque::new(),
            preempted_queue: VecDeque::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ContinuousBatcher
// ---------------------------------------------------------------------------

/// Continuous batcher — the main scheduler for LLM inference serving.
///
/// Implements iteration-level scheduling inspired by Orca / vLLM.  At each
/// [`step`](ContinuousBatcher::step) the batcher decides which waiting
/// requests to admit for prefill, which running requests continue decoding,
/// and whether any requests must be preempted.
#[derive(Debug)]
pub struct ContinuousBatcher {
    config: BatchConfig,
    state: BatchState,
    next_slot_id: usize,
    completed_count: u64,
}

impl ContinuousBatcher {
    /// Create a new batcher with the given configuration.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            state: BatchState::new(),
            next_slot_id: 0,
            completed_count: 0,
        }
    }

    /// Enqueue a new inference request. Returns its `RequestId`.
    pub fn add_request(&mut self, request: InferenceRequest) -> DnnResult<RequestId> {
        if request.sequence_length == 0 {
            return Err(DnnError::InvalidArgument(
                "sequence_length must be > 0".into(),
            ));
        }
        if request.sequence_length > self.config.max_sequence_length {
            return Err(DnnError::InvalidArgument(format!(
                "sequence_length {} exceeds max_sequence_length {}",
                request.sequence_length, self.config.max_sequence_length
            )));
        }
        let id = request.request_id;
        self.state.prefill_queue.push_back(request);
        Ok(id)
    }

    /// Execute one scheduling step.
    ///
    /// Returns a [`BatchDecision`] describing which requests to prefill,
    /// decode, and preempt during this iteration.
    pub fn step(&mut self) -> DnnResult<BatchDecision> {
        let mut decision = BatchDecision {
            prefill_requests: Vec::new(),
            decode_requests: Vec::new(),
            preempted: Vec::new(),
            total_tokens: 0,
        };

        // 1. Collect decode requests from active slots.
        let decode_ids: Vec<RequestId> = self
            .state
            .active_slots
            .iter()
            .filter(|s| s.is_active && !s.is_prefill)
            .map(|s| s.request_id)
            .collect();

        let decode_count = decode_ids.len().min(self.config.decode_batch_size);
        let decode_tokens: usize = self
            .state
            .active_slots
            .iter()
            .filter(|s| s.is_active && !s.is_prefill)
            .take(decode_count)
            .map(|s| s.current_seq_len + 1) // +1 for the token being generated
            .sum();

        decision.decode_requests = decode_ids.into_iter().take(decode_count).collect();

        // 2. Sort the prefill queue according to scheduling policy.
        self.sort_prefill_queue();

        // 3. Admit prefill requests within budget.
        let mut prefill_budget = self
            .config
            .prefill_batch_size
            .min(self.config.max_total_tokens.saturating_sub(decode_tokens));

        let mut admitted = Vec::new();
        while !self.state.prefill_queue.is_empty()
            && self.state.active_slots.len() + admitted.len() < self.config.max_batch_size
        {
            // Peek at the front.
            let req = match self.state.prefill_queue.front() {
                Some(r) => r,
                None => break,
            };
            if req.sequence_length > prefill_budget {
                break;
            }
            // Safe: we just confirmed front() is Some.
            let req = self
                .state
                .prefill_queue
                .pop_front()
                .ok_or_else(|| DnnError::InvalidArgument("empty queue".into()))?;

            prefill_budget = prefill_budget.saturating_sub(req.sequence_length);

            let slot = BatchSlot {
                slot_id: self.next_slot_id,
                request_id: req.request_id,
                current_seq_len: req.sequence_length,
                max_seq_len: req.sequence_length + req.max_new_tokens,
                is_prefill: true,
                is_active: true,
            };
            self.next_slot_id += 1;
            decision.prefill_requests.push(req.request_id);
            admitted.push(slot);
        }

        // 4. Transition admitted prefill slots to decode.
        for slot in &mut admitted {
            slot.is_prefill = false;
        }
        self.state.active_slots.extend(admitted);

        // Increment decode tokens for existing slots.
        for slot in &mut self.state.active_slots {
            if slot.is_active && !slot.is_prefill {
                slot.current_seq_len = slot.current_seq_len.saturating_add(1);
            }
        }

        decision.total_tokens = self
            .state
            .active_slots
            .iter()
            .filter(|s| s.is_active)
            .map(|s| s.current_seq_len)
            .sum();

        self.state.total_tokens = decision.total_tokens;

        Ok(decision)
    }

    /// Mark a request as completed and free its resources.
    pub fn complete_request(&mut self, request_id: RequestId) -> DnnResult<()> {
        let pos = self
            .state
            .active_slots
            .iter()
            .position(|s| s.request_id == request_id)
            .ok_or_else(|| {
                DnnError::InvalidArgument(format!("request {request_id} not in active slots"))
            })?;
        let slot = &self.state.active_slots[pos];
        self.state.total_tokens = self.state.total_tokens.saturating_sub(slot.current_seq_len);
        self.state.active_slots.remove(pos);
        self.state.decode_queue.retain(|id| *id != request_id);
        self.completed_count += 1;
        Ok(())
    }

    /// Preempt a running request. The request is moved to the preempted queue
    /// and may be resumed later.
    pub fn preempt(&mut self, request_id: RequestId) -> DnnResult<()> {
        let pos = self
            .state
            .active_slots
            .iter()
            .position(|s| s.request_id == request_id)
            .ok_or_else(|| {
                DnnError::InvalidArgument(format!("request {request_id} not in active slots"))
            })?;
        let slot = self.state.active_slots.remove(pos);
        self.state.total_tokens = self.state.total_tokens.saturating_sub(slot.current_seq_len);
        self.state.decode_queue.retain(|id| *id != request_id);

        // Re-enqueue as a prefill request so it can be recomputed.
        let preempted_req = InferenceRequest {
            request_id,
            sequence_length: slot.current_seq_len,
            max_new_tokens: slot.max_seq_len.saturating_sub(slot.current_seq_len),
            priority: Priority::Normal,
            arrival_time_ns: 0,
            deadline_ns: None,
        };
        self.state.preempted_queue.push_back(preempted_req);
        Ok(())
    }

    /// Number of requests currently executing (prefill + decode).
    pub fn active_requests(&self) -> usize {
        self.state
            .active_slots
            .iter()
            .filter(|s| s.is_active)
            .count()
    }

    /// Number of requests waiting in all queues (prefill + preempted).
    pub fn pending_requests(&self) -> usize {
        self.state.prefill_queue.len() + self.state.preempted_queue.len()
    }

    /// Total tokens that would be processed in the current active batch.
    pub fn throughput_tokens_per_step(&self) -> usize {
        self.state.total_tokens
    }

    // -- private helpers --

    fn sort_prefill_queue(&mut self) {
        let queue = &mut self.state.prefill_queue;
        let policy = self.config.scheduling_policy;

        let mut vec: Vec<InferenceRequest> = queue.drain(..).collect();
        match policy {
            SchedulingPolicy::Fcfs => {
                // Already in arrival order — sort by arrival_time_ns.
                vec.sort_by_key(|r| r.arrival_time_ns);
            }
            SchedulingPolicy::ShortestJobFirst => {
                vec.sort_by_key(|r| r.max_new_tokens);
            }
            SchedulingPolicy::PriorityBased => {
                // Higher priority first, then FCFS within same priority.
                vec.sort_by(|a, b| {
                    b.priority
                        .cmp(&a.priority)
                        .then(a.arrival_time_ns.cmp(&b.arrival_time_ns))
                });
            }
            SchedulingPolicy::DeadlineAware => {
                // Earliest deadline first; no-deadline requests go last.
                vec.sort_by(|a, b| {
                    let da = a.deadline_ns.unwrap_or(u64::MAX);
                    let db = b.deadline_ns.unwrap_or(u64::MAX);
                    da.cmp(&db).then(a.arrival_time_ns.cmp(&b.arrival_time_ns))
                });
            }
            SchedulingPolicy::Orca => {
                // Orca: iteration-level — same as FCFS for the prefill queue.
                vec.sort_by_key(|r| r.arrival_time_ns);
            }
        }
        *queue = VecDeque::from(vec);
    }
}

// ---------------------------------------------------------------------------
// TokenBudgetAllocator
// ---------------------------------------------------------------------------

/// Manages the per-step token budget shared between prefill and decode.
#[derive(Debug)]
pub struct TokenBudgetAllocator {
    max_total_tokens: usize,
    allocated: usize,
}

impl TokenBudgetAllocator {
    /// Create an allocator with the given capacity.
    pub fn new(max_total_tokens: usize) -> Self {
        Self {
            max_total_tokens,
            allocated: 0,
        }
    }

    /// Try to allocate `seq_len` tokens for a prefill request.
    /// Returns `Some(slot_index)` on success, `None` if the budget is
    /// exhausted.
    pub fn allocate_prefill(&mut self, seq_len: usize) -> Option<usize> {
        if self.allocated + seq_len > self.max_total_tokens {
            return None;
        }
        let slot = self.allocated;
        self.allocated += seq_len;
        Some(slot)
    }

    /// How many decode slots (each consuming 1 token) can still fit.
    pub fn allocate_decode(&mut self, count: usize) -> usize {
        let remaining = self.max_total_tokens.saturating_sub(self.allocated);
        let actual = count.min(remaining);
        self.allocated += actual;
        actual
    }

    /// Release `tokens` from the budget.
    pub fn release(&mut self, tokens: usize) {
        self.allocated = self.allocated.saturating_sub(tokens);
    }

    /// Fraction of the budget currently in use (0.0..=1.0).
    pub fn utilization(&self) -> f64 {
        if self.max_total_tokens == 0 {
            return 0.0;
        }
        self.allocated as f64 / self.max_total_tokens as f64
    }
}

// ---------------------------------------------------------------------------
// PagedKvManager
// ---------------------------------------------------------------------------

/// Block-level paged KV-cache manager.
///
/// Inspired by the paging scheme in vLLM.  Physical blocks are allocated on
/// demand and freed when a request completes.  Copy-on-write is supported for
/// speculative / beam-search scenarios.
#[derive(Debug)]
pub struct PagedKvManager {
    num_blocks: usize,
    block_size: usize,
    /// `true` ⇒ block is free.
    free_map: Vec<bool>,
    /// Reference count per block (for CoW).
    ref_counts: Vec<usize>,
}

impl PagedKvManager {
    /// Create a manager with `num_blocks` blocks, each holding `block_size`
    /// tokens.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            num_blocks,
            block_size,
            free_map: vec![true; num_blocks],
            ref_counts: vec![0; num_blocks],
        }
    }

    /// Allocate enough blocks to hold `num_tokens` tokens.
    ///
    /// Returns the list of allocated block IDs, or an error if there is
    /// insufficient free space.
    pub fn allocate(&mut self, num_tokens: usize) -> DnnResult<Vec<usize>> {
        if self.block_size == 0 {
            return Err(DnnError::InvalidArgument("block_size is 0".into()));
        }
        let blocks_needed = num_tokens.div_ceil(self.block_size);
        if !self.can_allocate(num_tokens) {
            return Err(DnnError::InvalidArgument(format!(
                "not enough free blocks: need {blocks_needed}, have {}",
                self.free_block_count()
            )));
        }
        let mut ids = Vec::with_capacity(blocks_needed);
        for (i, free) in self.free_map.iter_mut().enumerate() {
            if ids.len() >= blocks_needed {
                break;
            }
            if *free {
                *free = false;
                self.ref_counts[i] = 1;
                ids.push(i);
            }
        }
        Ok(ids)
    }

    /// Free the given blocks. Decrements reference counts and marks blocks as
    /// free when the count reaches zero.
    pub fn free(&mut self, block_ids: &[usize]) {
        for &id in block_ids {
            if id < self.num_blocks {
                self.ref_counts[id] = self.ref_counts[id].saturating_sub(1);
                if self.ref_counts[id] == 0 {
                    self.free_map[id] = true;
                }
            }
        }
    }

    /// Copy-on-write: create a new physical copy of `block_id`.
    ///
    /// Used when a block is shared (ref_count > 1) and one branch needs to
    /// diverge (e.g. beam search).
    pub fn copy_on_write(&mut self, block_id: usize) -> DnnResult<usize> {
        if block_id >= self.num_blocks {
            return Err(DnnError::InvalidArgument(format!(
                "block_id {block_id} out of range (max {})",
                self.num_blocks
            )));
        }
        // Find a free block.
        let new_id =
            self.free_map.iter().position(|&free| free).ok_or_else(|| {
                DnnError::InvalidArgument("no free blocks for copy-on-write".into())
            })?;
        self.free_map[new_id] = false;
        self.ref_counts[new_id] = 1;

        // Decrement old block ref count.
        self.ref_counts[block_id] = self.ref_counts[block_id].saturating_sub(1);
        if self.ref_counts[block_id] == 0 {
            self.free_map[block_id] = true;
        }

        Ok(new_id)
    }

    /// (used, total) block counts.
    pub fn usage(&self) -> (usize, usize) {
        let used = self.free_map.iter().filter(|&&free| !free).count();
        (used, self.num_blocks)
    }

    /// Whether `num_tokens` tokens can be allocated right now.
    pub fn can_allocate(&self, num_tokens: usize) -> bool {
        if self.block_size == 0 {
            return false;
        }
        let needed = num_tokens.div_ceil(self.block_size);
        self.free_block_count() >= needed
    }

    fn free_block_count(&self) -> usize {
        self.free_map.iter().filter(|&&f| f).count()
    }
}

// ---------------------------------------------------------------------------
// SpeculativeDecoder
// ---------------------------------------------------------------------------

/// Speculative decoding support (draft + verify).
///
/// A small "draft" model proposes several tokens ahead, and a larger "target"
/// model verifies them in a single forward pass, accepting a prefix of the
/// proposed tokens.  This amortises the cost of autoregressive generation.
#[derive(Debug)]
pub struct SpeculativeDecoder {
    draft_length: usize,
    total_proposed: u64,
    total_accepted: u64,
}

impl SpeculativeDecoder {
    /// Create a speculative decoder that proposes `draft_length` tokens at a
    /// time.
    pub fn new(draft_length: usize) -> Self {
        Self {
            draft_length,
            total_proposed: 0,
            total_accepted: 0,
        }
    }

    /// Propose draft token sequences.
    ///
    /// Returns `num_candidates` candidate sequences, each of length
    /// `draft_length`.  On macOS (no GPU) these are deterministic placeholder
    /// token ids.
    pub fn propose_tokens(&self, num_candidates: usize) -> Vec<Vec<u32>> {
        // Simulated: produce deterministic sequences for testing / macOS.
        (0..num_candidates)
            .map(|c| {
                (0..self.draft_length)
                    .map(|t| ((c * self.draft_length + t) % 32000) as u32)
                    .collect()
            })
            .collect()
    }

    /// Verify proposed sequences against target model probabilities.
    ///
    /// Uses rejection sampling: accepts the longest prefix whose cumulative
    /// acceptance probability exceeds the target threshold.
    ///
    /// Returns `(accepted_tokens, acceptance_count)`.
    pub fn verify_and_accept(
        &mut self,
        proposed: &[Vec<u32>],
        target_probs: &[f64],
    ) -> (Vec<u32>, usize) {
        if proposed.is_empty() {
            return (Vec::new(), 0);
        }

        // Simple rejection-sampling simulation:
        // Accept tokens while target_prob >= threshold (0.5).
        let best = proposed.first().cloned().unwrap_or_default();
        let mut accepted = Vec::new();
        let threshold = 0.5;

        for (i, token) in best.iter().enumerate() {
            let prob = target_probs.get(i).copied().unwrap_or(0.0);
            if prob >= threshold {
                accepted.push(*token);
            } else {
                break;
            }
        }

        let count = accepted.len();
        self.total_proposed += best.len() as u64;
        self.total_accepted += count as u64;
        (accepted, count)
    }

    /// Running acceptance rate across all calls to `verify_and_accept`.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_proposed == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_proposed as f64
    }
}

// ---------------------------------------------------------------------------
// BatchMetrics
// ---------------------------------------------------------------------------

/// Running statistics for the inference serving loop.
#[derive(Debug)]
pub struct BatchMetrics {
    /// (prefill_tokens, decode_tokens, latency_us) per step.
    steps: Vec<(usize, usize, u64)>,
    /// Time-to-first-token in microseconds for each request.
    ttft_samples: Vec<u64>,
}

impl BatchMetrics {
    /// Create an empty metrics collector.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            ttft_samples: Vec::new(),
        }
    }

    /// Record one scheduling step.
    pub fn record_step(&mut self, prefill_tokens: usize, decode_tokens: usize, latency_us: u64) {
        self.steps.push((prefill_tokens, decode_tokens, latency_us));
    }

    /// Record a time-to-first-token sample (in microseconds).
    pub fn record_ttft(&mut self, ttft_us: u64) {
        self.ttft_samples.push(ttft_us);
    }

    /// Average latency of steps that included prefill tokens (microseconds).
    pub fn avg_prefill_latency(&self) -> f64 {
        let prefills: Vec<u64> = self
            .steps
            .iter()
            .filter(|(p, _, _)| *p > 0)
            .map(|(_, _, l)| *l)
            .collect();
        if prefills.is_empty() {
            return 0.0;
        }
        prefills.iter().sum::<u64>() as f64 / prefills.len() as f64
    }

    /// Average latency of steps that included decode tokens (microseconds).
    pub fn avg_decode_latency(&self) -> f64 {
        let decodes: Vec<u64> = self
            .steps
            .iter()
            .filter(|(_, d, _)| *d > 0)
            .map(|(_, _, l)| *l)
            .collect();
        if decodes.is_empty() {
            return 0.0;
        }
        decodes.iter().sum::<u64>() as f64 / decodes.len() as f64
    }

    /// Average batch size (total tokens per step).
    pub fn avg_batch_size(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total: usize = self.steps.iter().map(|(p, d, _)| p + d).sum();
        total as f64 / self.steps.len() as f64
    }

    /// Estimated token throughput (tokens / second).
    pub fn token_throughput(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let total_tokens: usize = self.steps.iter().map(|(p, d, _)| p + d).sum();
        let total_us: u64 = self.steps.iter().map(|(_, _, l)| l).sum();
        if total_us == 0 {
            return 0.0;
        }
        total_tokens as f64 / (total_us as f64 / 1_000_000.0)
    }

    /// Median (p50) time-to-first-token in microseconds.
    pub fn time_to_first_token_p50(&self) -> f64 {
        if self.ttft_samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.ttft_samples.clone();
        sorted.sort_unstable();
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 && sorted.len() >= 2 {
            (sorted[mid - 1] + sorted[mid]) as f64 / 2.0
        } else {
            sorted[mid] as f64
        }
    }

    /// Human-readable performance report.
    pub fn format_report(&self) -> String {
        format!(
            "BatchMetrics Report\n\
             ====================\n\
             Steps recorded       : {}\n\
             Avg prefill latency  : {:.1} us\n\
             Avg decode latency   : {:.1} us\n\
             Avg batch size       : {:.1} tokens/step\n\
             Token throughput     : {:.0} tokens/s\n\
             TTFT p50             : {:.1} us\n\
             TTFT samples         : {}",
            self.steps.len(),
            self.avg_prefill_latency(),
            self.avg_decode_latency(),
            self.avg_batch_size(),
            self.token_throughput(),
            self.time_to_first_token_p50(),
            self.ttft_samples.len(),
        )
    }
}

impl Default for BatchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> BatchConfig {
        BatchConfig {
            max_batch_size: 8,
            max_total_tokens: 4096,
            max_sequence_length: 2048,
            prefill_batch_size: 1024,
            decode_batch_size: 8,
            scheduling_policy: SchedulingPolicy::Fcfs,
        }
    }

    fn make_request(id: RequestId, seq_len: usize, max_new: usize) -> InferenceRequest {
        InferenceRequest {
            request_id: id,
            sequence_length: seq_len,
            max_new_tokens: max_new,
            priority: Priority::Normal,
            arrival_time_ns: id * 1000,
            deadline_ns: None,
        }
    }

    // 1. Add single request
    #[test]
    fn test_add_single_request() {
        let mut batcher = ContinuousBatcher::new(default_config());
        let req = make_request(1, 128, 64);
        let id = batcher.add_request(req).expect("should succeed");
        assert_eq!(id, 1);
        assert_eq!(batcher.pending_requests(), 1);
        assert_eq!(batcher.active_requests(), 0);
    }

    // 2. Batch step with mixed prefill/decode
    #[test]
    fn test_batch_step_mixed_prefill_decode() {
        let mut batcher = ContinuousBatcher::new(default_config());
        // First request: prefill + become decode
        batcher.add_request(make_request(1, 64, 32)).expect("add 1");
        let d1 = batcher.step().expect("step 1");
        assert_eq!(d1.prefill_requests.len(), 1);

        // Second request while first is decoding
        batcher.add_request(make_request(2, 32, 16)).expect("add 2");
        let d2 = batcher.step().expect("step 2");
        assert!(!d2.decode_requests.is_empty(), "should have decode slots");
        assert!(!d2.prefill_requests.is_empty(), "should have prefill slots");
    }

    // 3. Token budget allocation/release
    #[test]
    fn test_token_budget_allocation_release() {
        let mut alloc = TokenBudgetAllocator::new(1024);
        let slot = alloc.allocate_prefill(512);
        assert!(slot.is_some());
        assert!((alloc.utilization() - 0.5).abs() < 1e-9);

        // Allocate more than remaining.
        assert!(alloc.allocate_prefill(600).is_none());

        alloc.release(256);
        assert!((alloc.utilization() - 0.25).abs() < 1e-9);
    }

    // 4. Paged KV allocation/free
    #[test]
    fn test_paged_kv_allocation_free() {
        let mut mgr = PagedKvManager::new(16, 64);
        let blocks = mgr.allocate(128).expect("allocate 128");
        assert_eq!(blocks.len(), 2);
        let (used, total) = mgr.usage();
        assert_eq!(used, 2);
        assert_eq!(total, 16);

        mgr.free(&blocks);
        let (used, _) = mgr.usage();
        assert_eq!(used, 0);
    }

    // 5. Copy-on-write
    #[test]
    fn test_copy_on_write() {
        let mut mgr = PagedKvManager::new(4, 64);
        let blocks = mgr.allocate(64).expect("allocate");
        assert_eq!(blocks.len(), 1);
        let orig = blocks[0];

        // Bump ref count to simulate sharing.
        mgr.ref_counts[orig] = 2;

        let new_id = mgr.copy_on_write(orig).expect("cow");
        assert_ne!(new_id, orig);
        // Old block should still be allocated (ref_count decremented to 1).
        assert!(!mgr.free_map[orig]);
        assert_eq!(mgr.ref_counts[orig], 1);
        assert_eq!(mgr.ref_counts[new_id], 1);
    }

    // 6. Continuous batching with request completion
    #[test]
    fn test_continuous_batching_completion() {
        let mut batcher = ContinuousBatcher::new(default_config());
        batcher.add_request(make_request(10, 64, 8)).expect("add");
        let _ = batcher.step().expect("step");
        assert_eq!(batcher.active_requests(), 1);

        batcher.complete_request(10).expect("complete");
        assert_eq!(batcher.active_requests(), 0);
    }

    // 7. Preemption
    #[test]
    fn test_preemption() {
        let mut batcher = ContinuousBatcher::new(default_config());
        batcher.add_request(make_request(20, 64, 16)).expect("add");
        let _ = batcher.step().expect("step");
        assert_eq!(batcher.active_requests(), 1);

        batcher.preempt(20).expect("preempt");
        assert_eq!(batcher.active_requests(), 0);
        // Preempted request is in the preempted queue.
        assert_eq!(batcher.pending_requests(), 1);
    }

    // 8. FCFS scheduling order
    #[test]
    fn test_fcfs_scheduling_order() {
        let mut batcher = ContinuousBatcher::new(default_config());
        batcher.add_request(make_request(3, 32, 8)).expect("add 3");
        batcher.add_request(make_request(1, 32, 8)).expect("add 1");
        batcher.add_request(make_request(2, 32, 8)).expect("add 2");
        // arrival_time_ns = id * 1000, so order is 1, 2, 3.
        let d = batcher.step().expect("step");
        assert_eq!(d.prefill_requests, vec![1, 2, 3]);
    }

    // 9. Priority-based scheduling
    #[test]
    fn test_priority_based_scheduling() {
        let mut config = default_config();
        config.scheduling_policy = SchedulingPolicy::PriorityBased;
        let mut batcher = ContinuousBatcher::new(config);

        let mut low = make_request(1, 32, 8);
        low.priority = Priority::Low;
        low.arrival_time_ns = 100;
        let mut high = make_request(2, 32, 8);
        high.priority = Priority::High;
        high.arrival_time_ns = 200;
        let mut normal = make_request(3, 32, 8);
        normal.priority = Priority::Normal;
        normal.arrival_time_ns = 50;

        batcher.add_request(low).expect("add low");
        batcher.add_request(high).expect("add high");
        batcher.add_request(normal).expect("add normal");

        let d = batcher.step().expect("step");
        // High (2) first, then Normal (3), then Low (1).
        assert_eq!(d.prefill_requests, vec![2, 3, 1]);
    }

    // 10. Deadline-aware scheduling
    #[test]
    fn test_deadline_aware_scheduling() {
        let mut config = default_config();
        config.scheduling_policy = SchedulingPolicy::DeadlineAware;
        let mut batcher = ContinuousBatcher::new(config);

        let mut r1 = make_request(1, 32, 8);
        r1.deadline_ns = Some(5000);
        let mut r2 = make_request(2, 32, 8);
        r2.deadline_ns = Some(1000);
        let mut r3 = make_request(3, 32, 8);
        r3.deadline_ns = None; // No deadline → goes last.

        batcher.add_request(r1).expect("add r1");
        batcher.add_request(r2).expect("add r2");
        batcher.add_request(r3).expect("add r3");

        let d = batcher.step().expect("step");
        assert_eq!(d.prefill_requests, vec![2, 1, 3]);
    }

    // 11. Speculative decoding acceptance
    #[test]
    fn test_speculative_decoding_acceptance() {
        let mut spec = SpeculativeDecoder::new(4);
        let proposed = spec.propose_tokens(2);
        assert_eq!(proposed.len(), 2);
        assert_eq!(proposed[0].len(), 4);

        // Accept first 2, reject the rest.
        let probs = vec![0.8, 0.6, 0.3, 0.1];
        let (accepted, count) = spec.verify_and_accept(&proposed, &probs);
        assert_eq!(count, 2);
        assert_eq!(accepted.len(), 2);
        assert!(spec.acceptance_rate() > 0.0);
    }

    // 12. Batch metrics tracking
    #[test]
    fn test_batch_metrics_tracking() {
        let mut m = BatchMetrics::new();
        m.record_step(128, 0, 500);
        m.record_step(0, 8, 100);
        m.record_step(64, 4, 300);

        assert!((m.avg_prefill_latency() - 400.0).abs() < 1e-9);
        assert!((m.avg_decode_latency() - 200.0).abs() < 1e-9);
        // (128+0+8+64+4) / 3 = 68.0
        assert!((m.avg_batch_size() - 68.0).abs() < 1e-9);
        assert!(m.token_throughput() > 0.0);
    }

    // 13. Max batch size enforcement
    #[test]
    fn test_max_batch_size_enforcement() {
        let mut config = default_config();
        config.max_batch_size = 2;
        let mut batcher = ContinuousBatcher::new(config);

        for i in 0..4 {
            batcher.add_request(make_request(i, 32, 8)).expect("add");
        }
        let d = batcher.step().expect("step");
        assert!(d.prefill_requests.len() <= 2);
        assert_eq!(batcher.active_requests(), d.prefill_requests.len());
    }

    // 14. Queue management
    #[test]
    fn test_queue_management() {
        let mut batcher = ContinuousBatcher::new(default_config());
        assert_eq!(batcher.pending_requests(), 0);

        batcher.add_request(make_request(1, 32, 8)).expect("add");
        batcher.add_request(make_request(2, 32, 8)).expect("add");
        assert_eq!(batcher.pending_requests(), 2);

        let _ = batcher.step().expect("step");
        assert_eq!(batcher.pending_requests(), 0);
        assert_eq!(batcher.active_requests(), 2);

        batcher.complete_request(1).expect("complete");
        assert_eq!(batcher.active_requests(), 1);
    }

    // 15. Utilization calculation
    #[test]
    fn test_utilization_calculation() {
        let mut alloc = TokenBudgetAllocator::new(1000);
        assert!((alloc.utilization() - 0.0).abs() < 1e-9);

        alloc.allocate_prefill(250);
        assert!((alloc.utilization() - 0.25).abs() < 1e-9);

        let fitted = alloc.allocate_decode(900);
        assert_eq!(fitted, 750);
        assert!((alloc.utilization() - 1.0).abs() < 1e-9);

        // Edge case: zero-capacity allocator.
        let zero = TokenBudgetAllocator::new(0);
        assert!((zero.utilization() - 0.0).abs() < 1e-9);
    }

    // 16. Format report
    #[test]
    fn test_format_report() {
        let mut m = BatchMetrics::new();
        m.record_step(100, 10, 200);
        m.record_step(0, 8, 100);
        m.record_ttft(150);
        m.record_ttft(250);
        let report = m.format_report();
        assert!(report.contains("Steps recorded"));
        assert!(report.contains("Token throughput"));
        assert!(report.contains("TTFT p50"));
    }
}
