//! Batching and scheduling subsystem.
//!
//! * [`sequence`]          — Sequence state machine, SamplingParams.
//! * [`scheduler`]         — FCFS scheduler with preemption.
//! * [`continuous_batcher`] — vLLM-style continuous batching orchestrator.

pub mod continuous_batcher;
pub mod scheduler;
pub mod sequence;

pub use continuous_batcher::{BatcherConfig, ContinuousBatcher, GenerationOutput};
pub use scheduler::{ScheduledBatch, Scheduler, SchedulerConfig, StepResult};
pub use sequence::{FinishReason, SamplingParams, Sequence, SequenceId, SequenceStatus};
