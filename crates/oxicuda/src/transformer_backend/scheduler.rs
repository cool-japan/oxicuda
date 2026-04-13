//! Continuous batching scheduler for transformer inference.
//!
//! Implements iteration-level scheduling that re-evaluates the batch
//! composition at every decoding step. Supports:
//!
//! - FCFS scheduling with priority override
//! - Preemption of low-priority sequences when memory is tight
//! - Prefill/decode separation (prefill sequences batched separately)
//! - Token budget management
//! - Prefill chunking (split long prompts across iterations)

use std::collections::VecDeque;

use super::{TransformerError, TransformerResult};

/// Priority level for scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Priority {
    /// Low priority — can be preempted.
    Low = 0,
    /// Normal priority — default.
    #[default]
    Normal = 1,
    /// High priority — preempts lower priority.
    High = 2,
    /// Critical priority — never preempted.
    Critical = 3,
}

/// Status of a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Waiting to be scheduled for initial prefill.
    Waiting,
    /// Currently running (prefill or decode).
    Running,
    /// Preempted — removed from running set, waiting to resume.
    Preempted,
    /// Finished — reached stop condition or max tokens.
    Finished,
}

/// A single sequence within a sequence group.
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Unique sequence ID.
    pub seq_id: u64,
    /// Input token IDs (prompt).
    pub prompt_tokens: Vec<u32>,
    /// Generated output token IDs so far.
    pub output_tokens: Vec<u32>,
    /// Current status.
    pub status: SequenceStatus,
    /// Number of prompt tokens processed so far (for chunked prefill).
    pub num_prefilled: usize,
}

impl Sequence {
    /// Create a new sequence.
    pub fn new(seq_id: u64, prompt_tokens: Vec<u32>) -> Self {
        Self {
            seq_id,
            prompt_tokens,
            output_tokens: Vec::new(),
            status: SequenceStatus::Waiting,
            num_prefilled: 0,
        }
    }

    /// Total number of tokens (prompt + output).
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    /// Number of tokens remaining in prefill.
    pub fn remaining_prefill(&self) -> usize {
        self.prompt_tokens.len().saturating_sub(self.num_prefilled)
    }

    /// Whether prefill is complete.
    pub fn is_prefill_complete(&self) -> bool {
        self.num_prefilled >= self.prompt_tokens.len()
    }
}

/// Sampling parameters for a sequence group.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Temperature for sampling.
    pub temperature: f64,
    /// Number of output sequences to generate per prompt.
    pub n: usize,
    /// Whether this is a beam search.
    pub use_beam_search: bool,
    /// Stop token IDs.
    pub stop_token_ids: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            n: 1,
            use_beam_search: false,
            stop_token_ids: Vec::new(),
        }
    }
}

/// A group of sequences sharing the same prompt and sampling params.
#[derive(Debug, Clone)]
pub struct SequenceGroup {
    /// Unique request ID.
    pub request_id: u64,
    /// Sequences in this group.
    pub sequences: Vec<Sequence>,
    /// Arrival time (monotonic clock seconds).
    pub arrival_time: f64,
    /// Priority level.
    pub priority: Priority,
    /// Sampling parameters.
    pub sampling_params: SamplingParams,
}

impl SequenceGroup {
    /// Create a new sequence group with a single sequence.
    pub fn new(
        request_id: u64,
        prompt_tokens: Vec<u32>,
        arrival_time: f64,
        sampling_params: SamplingParams,
    ) -> Self {
        let n = sampling_params.n.max(1);
        let mut sequences = Vec::with_capacity(n);
        for i in 0..n {
            sequences.push(Sequence::new(
                request_id * 1000 + i as u64,
                prompt_tokens.clone(),
            ));
        }
        Self {
            request_id,
            sequences,
            arrival_time,
            priority: Priority::Normal,
            sampling_params,
        }
    }

    /// Whether all sequences are finished.
    pub fn is_finished(&self) -> bool {
        self.sequences
            .iter()
            .all(|s| s.status == SequenceStatus::Finished)
    }

    /// Number of tokens needed for one decoding step across all sequences.
    pub fn num_decode_tokens(&self) -> usize {
        self.sequences
            .iter()
            .filter(|s| s.status == SequenceStatus::Running && s.is_prefill_complete())
            .count()
    }

    /// Total KV cache tokens across all running sequences.
    pub fn total_cached_tokens(&self) -> usize {
        self.sequences
            .iter()
            .filter(|s| s.status == SequenceStatus::Running)
            .map(|s| s.total_tokens())
            .sum()
    }

    /// Number of tokens remaining in prefill for this group.
    pub fn remaining_prefill_tokens(&self) -> usize {
        self.sequences
            .iter()
            .filter(|s| s.status == SequenceStatus::Running || s.status == SequenceStatus::Waiting)
            .map(|s| s.remaining_prefill())
            .sum()
    }
}

/// Token budget for scheduling.
#[derive(Debug, Clone)]
pub struct SchedulerBudget {
    /// Tokens remaining in budget for this iteration.
    pub token_budget: usize,
    /// Sequences remaining in budget.
    pub seq_budget: usize,
}

impl SchedulerBudget {
    /// Create a new budget.
    pub fn new(max_tokens: usize, max_seqs: usize) -> Self {
        Self {
            token_budget: max_tokens,
            seq_budget: max_seqs,
        }
    }

    /// Whether any budget remains.
    pub fn has_budget(&self) -> bool {
        self.token_budget > 0 && self.seq_budget > 0
    }

    /// Consume budget.
    pub fn consume(&mut self, tokens: usize, seqs: usize) {
        self.token_budget = self.token_budget.saturating_sub(tokens);
        self.seq_budget = self.seq_budget.saturating_sub(seqs);
    }
}

/// Scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of tokens in a single batch.
    pub max_batch_tokens: usize,
    /// Maximum concurrent sequences.
    pub max_num_seqs: usize,
    /// Maximum tokens to prefill per chunk (0 = unlimited).
    pub max_prefill_chunk: usize,
    /// Whether to enable chunked prefill.
    pub enable_chunked_prefill: bool,
    /// Whether to enable preemption.
    pub enable_preemption: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_tokens: 4096,
            max_num_seqs: 256,
            max_prefill_chunk: 512,
            enable_chunked_prefill: true,
            enable_preemption: true,
        }
    }
}

/// Output of a scheduling step.
#[derive(Debug, Clone)]
pub enum SchedulerOutput {
    /// Scheduled sequences for this iteration.
    Schedule {
        /// Sequences needing prefill.
        prefill: Vec<SequenceGroup>,
        /// Sequences in decode phase.
        decode: Vec<SequenceGroup>,
        /// Sequences that were preempted.
        preempted: Vec<SequenceGroup>,
        /// Total tokens in this batch.
        num_batched_tokens: usize,
    },
    /// Nothing to schedule.
    Empty,
}

/// Continuous batching scheduler.
///
/// Re-evaluates batch composition at every decoding step. New requests
/// can join the batch at any time, and finished/preempted sequences
/// free their slots immediately.
#[derive(Debug)]
pub struct ContinuousBatchScheduler {
    /// Currently running sequence groups.
    running: Vec<SequenceGroup>,
    /// Waiting queue (FCFS order).
    waiting: VecDeque<SequenceGroup>,
    /// Preempted sequence groups.
    preempted: Vec<SequenceGroup>,
    /// Configuration.
    config: SchedulerConfig,
    /// Next request ID.
    next_request_id: u64,
}

impl ContinuousBatchScheduler {
    /// Create a new scheduler.
    pub fn new(config: SchedulerConfig) -> TransformerResult<Self> {
        if config.max_batch_tokens == 0 {
            return Err(TransformerError::SchedulerError(
                "max_batch_tokens must be > 0".to_string(),
            ));
        }
        if config.max_num_seqs == 0 {
            return Err(TransformerError::SchedulerError(
                "max_num_seqs must be > 0".to_string(),
            ));
        }
        Ok(Self {
            running: Vec::new(),
            waiting: VecDeque::new(),
            preempted: Vec::new(),
            config,
            next_request_id: 1,
        })
    }

    /// Add a new request to the waiting queue.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        arrival_time: f64,
    ) -> u64 {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let sg = SequenceGroup::new(request_id, prompt_tokens, arrival_time, sampling_params);
        self.waiting.push_back(sg);
        request_id
    }

    /// Add a request with a specific priority.
    pub fn add_request_with_priority(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        arrival_time: f64,
        priority: Priority,
    ) -> u64 {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let mut sg = SequenceGroup::new(request_id, prompt_tokens, arrival_time, sampling_params);
        sg.priority = priority;

        // Insert based on priority (higher priority goes earlier)
        let insert_pos = self
            .waiting
            .iter()
            .position(|w| w.priority < priority)
            .unwrap_or(self.waiting.len());
        self.waiting.insert(insert_pos, sg);

        request_id
    }

    /// Run one scheduling step.
    ///
    /// Returns the set of sequences to run this iteration, split into
    /// prefill and decode groups.
    #[allow(clippy::too_many_lines)]
    pub fn schedule(&mut self, available_blocks: usize) -> SchedulerOutput {
        let mut budget =
            SchedulerBudget::new(self.config.max_batch_tokens, self.config.max_num_seqs);

        let mut prefill_out = Vec::new();
        let mut decode_out = Vec::new();
        let mut preempted_out = Vec::new();
        let mut num_batched_tokens = 0usize;

        // Step 1: Remove finished sequences from running set
        self.running.retain(|sg| !sg.is_finished());

        // Step 2: Try to schedule running (decode) sequences first
        let mut keep_running = Vec::new();
        let mut to_preempt = Vec::new();

        for sg in self.running.drain(..) {
            let decode_tokens = sg.num_decode_tokens().max(1);
            if budget.has_budget() {
                budget.consume(decode_tokens, 1);
                num_batched_tokens += decode_tokens;
                keep_running.push(sg);
            } else if self.config.enable_preemption {
                to_preempt.push(sg);
            } else {
                keep_running.push(sg);
            }
        }

        // Preempt from lowest priority first
        to_preempt.sort_by_key(|sg| sg.priority);
        for mut sg in to_preempt {
            for seq in &mut sg.sequences {
                if seq.status == SequenceStatus::Running {
                    seq.status = SequenceStatus::Preempted;
                }
            }
            preempted_out.push(sg.clone());
            self.preempted.push(sg);
        }

        // Step 3: Try to schedule preempted sequences (higher priority first)
        self.preempted
            .sort_by_key(|b| std::cmp::Reverse(b.priority));
        let mut still_preempted = Vec::new();

        for mut sg in self.preempted.drain(..) {
            if !budget.has_budget() || available_blocks == 0 {
                still_preempted.push(sg);
                continue;
            }
            let tokens_needed = sg.num_decode_tokens().max(1);
            budget.consume(tokens_needed, 1);
            num_batched_tokens += tokens_needed;
            for seq in &mut sg.sequences {
                if seq.status == SequenceStatus::Preempted {
                    seq.status = SequenceStatus::Running;
                }
            }
            keep_running.push(sg);
        }
        self.preempted = still_preempted;

        // Step 4: Schedule waiting (prefill) sequences
        let mut still_waiting = VecDeque::new();

        while let Some(mut sg) = self.waiting.pop_front() {
            if !budget.has_budget() {
                still_waiting.push_back(sg);
                continue;
            }

            let prefill_tokens =
                if self.config.enable_chunked_prefill && self.config.max_prefill_chunk > 0 {
                    sg.remaining_prefill_tokens()
                        .min(self.config.max_prefill_chunk)
                } else {
                    sg.remaining_prefill_tokens()
                };

            if prefill_tokens == 0 {
                // No prefill needed, treat as decode
                for seq in &mut sg.sequences {
                    seq.status = SequenceStatus::Running;
                    seq.num_prefilled = seq.prompt_tokens.len();
                }
                budget.consume(1, 1);
                num_batched_tokens += 1;
                keep_running.push(sg);
                continue;
            }

            if prefill_tokens > budget.token_budget {
                still_waiting.push_back(sg);
                continue;
            }

            // Schedule prefill
            for seq in &mut sg.sequences {
                if seq.status == SequenceStatus::Waiting {
                    seq.status = SequenceStatus::Running;
                }
                let chunk =
                    if self.config.enable_chunked_prefill && self.config.max_prefill_chunk > 0 {
                        seq.remaining_prefill().min(self.config.max_prefill_chunk)
                    } else {
                        seq.remaining_prefill()
                    };
                seq.num_prefilled += chunk;
            }

            budget.consume(prefill_tokens, 1);
            num_batched_tokens += prefill_tokens;

            if sg.sequences.iter().all(|s| s.is_prefill_complete()) {
                keep_running.push(sg.clone());
            }
            prefill_out.push(sg);
        }

        // Remaining waiting sequences stay in the queue
        while let Some(sg) = still_waiting.pop_front() {
            self.waiting.push_back(sg);
        }

        // Move decode sequences to output
        for sg in &keep_running {
            if sg.sequences.iter().any(|s| s.is_prefill_complete()) {
                decode_out.push(sg.clone());
            }
        }

        self.running = keep_running;

        if num_batched_tokens == 0 && prefill_out.is_empty() && decode_out.is_empty() {
            return SchedulerOutput::Empty;
        }

        SchedulerOutput::Schedule {
            prefill: prefill_out,
            decode: decode_out,
            preempted: preempted_out,
            num_batched_tokens,
        }
    }

    /// Mark a sequence group as finished.
    pub fn finish_request(&mut self, request_id: u64) {
        for sg in &mut self.running {
            if sg.request_id == request_id {
                for seq in &mut sg.sequences {
                    seq.status = SequenceStatus::Finished;
                }
            }
        }
    }

    /// Abort a request (remove from all queues).
    pub fn abort_request(&mut self, request_id: u64) {
        self.running.retain(|sg| sg.request_id != request_id);
        self.waiting.retain(|sg| sg.request_id != request_id);
        self.preempted.retain(|sg| sg.request_id != request_id);
    }

    /// Number of running sequence groups.
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Number of waiting requests.
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Number of preempted requests.
    pub fn num_preempted(&self) -> usize {
        self.preempted.len()
    }

    /// Whether the scheduler has any pending work.
    pub fn has_pending(&self) -> bool {
        !self.running.is_empty() || !self.waiting.is_empty() || !self.preempted.is_empty()
    }

    /// Get the scheduler configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_scheduler() -> ContinuousBatchScheduler {
        ContinuousBatchScheduler::new(SchedulerConfig::default()).unwrap()
    }

    fn make_prompt(len: usize) -> Vec<u32> {
        (0..len as u32).collect()
    }

    #[test]
    fn test_add_request() {
        let mut sched = default_scheduler();
        let id = sched.add_request(make_prompt(10), SamplingParams::default(), 0.0);
        assert_eq!(id, 1);
        assert_eq!(sched.num_waiting(), 1);
    }

    #[test]
    fn test_schedule_single_prefill() {
        let mut sched = default_scheduler();
        sched.add_request(make_prompt(10), SamplingParams::default(), 0.0);

        let output = sched.schedule(100);
        match output {
            SchedulerOutput::Schedule {
                prefill,
                num_batched_tokens,
                ..
            } => {
                assert_eq!(prefill.len(), 1);
                assert!(num_batched_tokens > 0);
            }
            SchedulerOutput::Empty => panic!("expected schedule output"),
        }
    }

    #[test]
    fn test_schedule_fcfs_order() {
        let mut sched = default_scheduler();
        let id1 = sched.add_request(make_prompt(5), SamplingParams::default(), 0.0);
        let id2 = sched.add_request(make_prompt(5), SamplingParams::default(), 1.0);

        let output = sched.schedule(100);
        match output {
            SchedulerOutput::Schedule { prefill, .. } => {
                assert!(!prefill.is_empty());
                assert_eq!(prefill[0].request_id, id1);
                if prefill.len() > 1 {
                    assert_eq!(prefill[1].request_id, id2);
                }
            }
            SchedulerOutput::Empty => panic!("expected schedule output"),
        }
    }

    #[test]
    fn test_priority_ordering() {
        let mut sched = default_scheduler();
        let _low = sched.add_request_with_priority(
            make_prompt(5),
            SamplingParams::default(),
            0.0,
            Priority::Low,
        );
        let high = sched.add_request_with_priority(
            make_prompt(5),
            SamplingParams::default(),
            1.0,
            Priority::High,
        );

        // High priority should be scheduled first
        let output = sched.schedule(100);
        if let SchedulerOutput::Schedule { prefill, .. } = output {
            assert!(!prefill.is_empty());
            assert_eq!(prefill[0].request_id, high);
        }
    }

    #[test]
    fn test_budget_limit() {
        let config = SchedulerConfig {
            max_batch_tokens: 10,
            max_num_seqs: 1,
            ..Default::default()
        };
        let mut sched = ContinuousBatchScheduler::new(config).unwrap();
        sched.add_request(make_prompt(5), SamplingParams::default(), 0.0);
        sched.add_request(make_prompt(5), SamplingParams::default(), 1.0);

        let output = sched.schedule(100);
        match output {
            SchedulerOutput::Schedule { prefill, .. } => {
                // Only one should be scheduled due to seq budget
                assert_eq!(prefill.len(), 1);
            }
            SchedulerOutput::Empty => panic!("expected schedule output"),
        }
        // Second request should still be waiting
        assert_eq!(sched.num_waiting(), 1);
    }

    #[test]
    fn test_finish_request() {
        let mut sched = default_scheduler();
        let id = sched.add_request(make_prompt(5), SamplingParams::default(), 0.0);
        sched.schedule(100); // prefill
        sched.finish_request(id);

        // Next schedule should clean up
        let output = sched.schedule(100);
        matches!(output, SchedulerOutput::Empty);
    }

    #[test]
    fn test_abort_request() {
        let mut sched = default_scheduler();
        sched.add_request(make_prompt(5), SamplingParams::default(), 0.0);
        sched.abort_request(1);
        assert_eq!(sched.num_waiting(), 0);
    }

    #[test]
    fn test_empty_schedule() {
        let mut sched = default_scheduler();
        let output = sched.schedule(100);
        matches!(output, SchedulerOutput::Empty);
    }

    #[test]
    fn test_has_pending() {
        let mut sched = default_scheduler();
        assert!(!sched.has_pending());
        sched.add_request(make_prompt(5), SamplingParams::default(), 0.0);
        assert!(sched.has_pending());
    }

    #[test]
    fn test_sequence_status_transitions() {
        let mut seq = Sequence::new(1, vec![1, 2, 3, 4, 5]);
        assert_eq!(seq.status, SequenceStatus::Waiting);
        assert_eq!(seq.total_tokens(), 5);
        assert_eq!(seq.remaining_prefill(), 5);
        assert!(!seq.is_prefill_complete());

        seq.num_prefilled = 5;
        assert!(seq.is_prefill_complete());
        assert_eq!(seq.remaining_prefill(), 0);
    }

    #[test]
    fn test_sequence_group_new() {
        let params = SamplingParams {
            n: 3,
            ..Default::default()
        };
        let sg = SequenceGroup::new(1, vec![1, 2, 3], 0.0, params);
        assert_eq!(sg.sequences.len(), 3);
        assert!(!sg.is_finished());
    }

    #[test]
    fn test_invalid_config() {
        assert!(
            ContinuousBatchScheduler::new(SchedulerConfig {
                max_batch_tokens: 0,
                ..Default::default()
            })
            .is_err()
        );

        assert!(
            ContinuousBatchScheduler::new(SchedulerConfig {
                max_num_seqs: 0,
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    fn test_scheduler_budget() {
        let mut budget = SchedulerBudget::new(100, 10);
        assert!(budget.has_budget());
        budget.consume(50, 5);
        assert!(budget.has_budget());
        assert_eq!(budget.token_budget, 50);
        assert_eq!(budget.seq_budget, 5);
        budget.consume(100, 100); // saturates at 0
        assert!(!budget.has_budget());
    }

    #[test]
    fn test_decode_after_prefill() {
        let mut sched = default_scheduler();
        sched.add_request(make_prompt(5), SamplingParams::default(), 0.0);

        // First schedule: prefill
        let output = sched.schedule(100);
        assert!(matches!(output, SchedulerOutput::Schedule { .. }));

        // Second schedule: decode (the sequence is now running)
        let output = sched.schedule(100);
        if let SchedulerOutput::Schedule { decode, .. } = output {
            assert!(!decode.is_empty());
        }
    }
}
