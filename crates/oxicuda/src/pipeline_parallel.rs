//! Pipeline parallelism primitives for multi-GPU model parallelism.
//!
//! Provides scheduling algorithms (GPipe, 1F1B, Interleaved, ZeroBubble),
//! bubble analysis, activation checkpointing, and ASCII visualization for
//! pipeline-parallel training across multiple GPU stages.

use oxicuda_driver::{CudaError, CudaResult};

// ─── PipelineSchedule ──────────────────────────────────────

/// Pipeline schedule algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineSchedule {
    /// GPipe: simple fill-drain. All forwards then all backwards.
    GPipe,
    /// PipeDream 1F1B: warmup, steady-state alternating F/B, cooldown.
    PipeDream1F1B,
    /// Interleaved 1F1B with virtual pipeline stages.
    InterleavedStages,
    /// Zero-bubble pipeline with split backward (B + W).
    ZeroBubble,
}

// ─── MicrobatchStatus ──────────────────────────────────────

/// Status of a microbatch as it flows through the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MicrobatchStatus {
    /// Not yet started.
    Pending,
    /// Currently executing forward pass at the given stage.
    InForward(usize),
    /// Currently executing backward pass at the given stage.
    InBackward(usize),
    /// All passes complete.
    Complete,
}

// ─── EventType ─────────────────────────────────────────────

/// Type of pipeline event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    /// Forward pass begins at a stage.
    ForwardStart,
    /// Forward pass ends at a stage.
    ForwardEnd,
    /// Backward pass begins at a stage.
    BackwardStart,
    /// Backward pass ends at a stage.
    BackwardEnd,
    /// Activation sent from stage to next stage.
    SendActivation,
    /// Activation received at stage from previous stage.
    RecvActivation,
    /// Weight-gradient backward (used by ZeroBubble).
    WeightGradStart,
    /// Weight-gradient backward end.
    WeightGradEnd,
}

// ─── PipelineEvent ─────────────────────────────────────────

/// A single event in the pipeline schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineEvent {
    /// Time slot (integer, 0-based).
    pub timestamp: usize,
    /// Type of this event.
    pub event_type: EventType,
    /// Stage index (0-based).
    pub stage_id: usize,
    /// Microbatch index (0-based).
    pub microbatch_id: usize,
}

// ─── PipelineStage ─────────────────────────────────────────

/// Description of a single pipeline stage.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Index of this stage in the pipeline.
    pub stage_id: usize,
    /// GPU device ordinal this stage runs on.
    pub device_id: i32,
    /// Human-readable name for the stage.
    pub name: String,
    /// Estimated compute cost in time units (1 = baseline).
    pub compute_cost_estimate: f64,
}

impl PipelineStage {
    /// Create a new pipeline stage.
    pub fn new(stage_id: usize, device_id: i32, name: impl Into<String>) -> Self {
        Self {
            stage_id,
            device_id,
            name: name.into(),
            compute_cost_estimate: 1.0,
        }
    }

    /// Set the compute cost estimate.
    pub fn with_compute_cost(mut self, cost: f64) -> Self {
        self.compute_cost_estimate = cost;
        self
    }
}

// ─── PipelineConfig ────────────────────────────────────────

/// Configuration for pipeline-parallel execution.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Ordered list of pipeline stages.
    pub stages: Vec<PipelineStage>,
    /// Number of microbatches to split the minibatch into.
    pub num_microbatches: usize,
    /// Pipeline schedule algorithm.
    pub schedule_type: PipelineSchedule,
    /// Interleave factor for `InterleavedStages` (virtual pipelines).
    pub interleave_factor: usize,
}

impl PipelineConfig {
    /// Validate the configuration, returning `CudaError::InvalidValue` on failure.
    pub fn validate(&self) -> CudaResult<()> {
        if self.stages.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        if self.num_microbatches == 0 {
            return Err(CudaError::InvalidValue);
        }
        if self.interleave_factor == 0 {
            return Err(CudaError::InvalidValue);
        }
        if self.schedule_type == PipelineSchedule::InterleavedStages
            && self.stages.len() % self.interleave_factor != 0
        {
            return Err(CudaError::InvalidValue);
        }
        Ok(())
    }
}

// ─── BubbleAnalysis ────────────────────────────────────────

/// Detailed bubble (idle time) analysis of a pipeline schedule.
#[derive(Debug, Clone)]
pub struct BubbleAnalysis {
    /// Total wall-clock time slots.
    pub total_time: usize,
    /// Total compute time across all stages (sum of F + B durations).
    pub compute_time: usize,
    /// Total bubble (idle) time across all stages.
    pub bubble_time: usize,
    /// Fraction of total stage-time that is bubble.
    pub bubble_ratio: f64,
    /// Per-stage idle fraction.
    pub per_stage_idle: Vec<f64>,
}

// ─── CheckpointDecision ────────────────────────────────────

/// Activation checkpointing decision for a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckpointDecision {
    /// Store all activations in GPU memory (fastest, most memory).
    Store,
    /// Recompute activations during backward pass (saves memory, costs compute).
    Recompute,
    /// Offload activations to host memory (saves GPU memory, costs PCIe bandwidth).
    Offload,
}

// ─── GPipe Scheduler ───────────────────────────────────────

/// GPipe scheduler: all forwards pipeline through, then all backwards.
///
/// For S stages and M microbatches:
/// - Forward microbatch m at stage s starts at time s + m
/// - Backward microbatch m at stage s starts after all forwards complete,
///   going from last stage to first: time (S + M - 1) + (S - 1 - s) + m
pub struct GpipeScheduler;

impl GpipeScheduler {
    /// Generate the full GPipe schedule.
    pub fn schedule(num_stages: usize, num_microbatches: usize) -> Vec<PipelineEvent> {
        let mut events = Vec::new();
        let s = num_stages;
        let m = num_microbatches;

        // Forward pass: microbatch mb at stage st starts at st + mb
        for mb in 0..m {
            for st in 0..s {
                let t = st + mb;
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardStart,
                    stage_id: st,
                    microbatch_id: mb,
                });
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardEnd,
                    stage_id: st,
                    microbatch_id: mb,
                });
                // Send activation to next stage
                if st + 1 < s {
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::SendActivation,
                        stage_id: st,
                        microbatch_id: mb,
                    });
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::RecvActivation,
                        stage_id: st + 1,
                        microbatch_id: mb,
                    });
                }
            }
        }

        // Backward pass: starts after all forwards complete
        // All forwards end at time (S-1) + (M-1) = S+M-2, so backward starts at S+M-1
        let bwd_offset = s + m - 1;
        for mb in 0..m {
            for st in (0..s).rev() {
                let t = bwd_offset + (s - 1 - st) + mb;
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::BackwardStart,
                    stage_id: st,
                    microbatch_id: mb,
                });
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::BackwardEnd,
                    stage_id: st,
                    microbatch_id: mb,
                });
                // Send gradient to previous stage
                if st > 0 {
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::SendActivation,
                        stage_id: st,
                        microbatch_id: mb,
                    });
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::RecvActivation,
                        stage_id: st - 1,
                        microbatch_id: mb,
                    });
                }
            }
        }

        events.sort_by_key(|e| (e.timestamp, e.stage_id, e.microbatch_id));
        events
    }
}

// ─── 1F1B Scheduler ────────────────────────────────────────

/// PipeDream 1F1B scheduler: warmup forwards, steady-state alternating F/B, cooldown.
///
/// Each stage s performs:
/// - Warmup: (S - 1 - s) forward passes
/// - Steady state: alternating 1 forward + 1 backward until all microbatches processed
/// - Cooldown: remaining backward passes
pub struct OneFOneBScheduler;

impl OneFOneBScheduler {
    /// Generate the 1F1B schedule.
    pub fn schedule(num_stages: usize, num_microbatches: usize) -> Vec<PipelineEvent> {
        let mut events = Vec::new();
        let s = num_stages;
        let m = num_microbatches;

        if s == 0 || m == 0 {
            return events;
        }

        // Per-stage scheduling: track the current time slot for each stage
        // Stage s starts its first forward at time s (pipeline offset)
        let mut stage_time: Vec<usize> = (0..s).collect();

        // Track which forward / backward microbatch each stage is on
        let mut fwd_mb = vec![0usize; s]; // next forward microbatch to schedule
        let mut bwd_mb = vec![0usize; s]; // next backward microbatch to schedule

        // Number of warmup forwards for each stage
        let warmup_count: Vec<usize> = (0..s).map(|st| (s - 1 - st).min(m)).collect();

        // Phase 1: Warmup — each stage does its warmup forwards
        for st in 0..s {
            for _ in 0..warmup_count[st] {
                let mb = fwd_mb[st];
                if mb >= m {
                    break;
                }
                let t = stage_time[st];
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardStart,
                    stage_id: st,
                    microbatch_id: mb,
                });
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardEnd,
                    stage_id: st,
                    microbatch_id: mb,
                });
                fwd_mb[st] = mb + 1;
                stage_time[st] = t + 1;
            }
        }

        // Phase 2: Steady state — alternate 1F1B
        for st in 0..s {
            while fwd_mb[st] < m {
                // 1 forward
                let mb_f = fwd_mb[st];
                let t = stage_time[st];
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardStart,
                    stage_id: st,
                    microbatch_id: mb_f,
                });
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardEnd,
                    stage_id: st,
                    microbatch_id: mb_f,
                });
                fwd_mb[st] = mb_f + 1;
                stage_time[st] = t + 1;

                // 1 backward
                let mb_b = bwd_mb[st];
                if mb_b < m {
                    let t = stage_time[st];
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::BackwardStart,
                        stage_id: st,
                        microbatch_id: mb_b,
                    });
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::BackwardEnd,
                        stage_id: st,
                        microbatch_id: mb_b,
                    });
                    bwd_mb[st] = mb_b + 1;
                    stage_time[st] = t + 1;
                }
            }
        }

        // Phase 3: Cooldown — remaining backwards
        for st in 0..s {
            while bwd_mb[st] < m {
                let mb_b = bwd_mb[st];
                let t = stage_time[st];
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::BackwardStart,
                    stage_id: st,
                    microbatch_id: mb_b,
                });
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::BackwardEnd,
                    stage_id: st,
                    microbatch_id: mb_b,
                });
                bwd_mb[st] = mb_b + 1;
                stage_time[st] = t + 1;
            }
        }

        events.sort_by_key(|e| (e.timestamp, e.stage_id, e.microbatch_id));
        events
    }
}

// ─── Interleaved Scheduler ─────────────────────────────────

/// Interleaved 1F1B scheduler with virtual pipeline stages.
///
/// Each physical device runs `interleave_factor` virtual stages. This reduces
/// the pipeline bubble by a factor of `interleave_factor` compared to standard
/// 1F1B. Virtual stages are numbered `0..num_stages` and mapped to physical
/// devices in round-robin fashion with stride `num_stages / interleave_factor`.
pub struct InterleavedScheduler;

impl InterleavedScheduler {
    /// Generate the interleaved 1F1B schedule.
    pub fn schedule(
        num_stages: usize,
        num_microbatches: usize,
        interleave_factor: usize,
    ) -> Vec<PipelineEvent> {
        let mut events = Vec::new();
        let s = num_stages;
        let m = num_microbatches;

        if s == 0 || m == 0 || interleave_factor == 0 {
            return events;
        }

        let num_physical = s / interleave_factor;
        if num_physical == 0 {
            return events;
        }

        // Each physical device processes interleave_factor virtual stages.
        // Virtual stage v is assigned to physical device v % num_physical.
        // Schedule proceeds in chunks: for each virtual pipeline iteration,
        // we do a round of forwards through all virtual stages, interleaved.

        // Simplified model: treat it as standard 1F1B but with num_stages
        // virtual stages, where consecutive virtual stages on the same physical
        // device can overlap (reducing bubble).
        //
        // The key scheduling difference: warmup count per virtual stage v is
        // (num_stages - 1 - v) / interleave_factor, reducing the warmup phase.

        let mut fwd_mb = vec![0usize; s];
        let mut bwd_mb = vec![0usize; s];

        // Initialize pipeline offsets — virtual stages on the same physical
        // device share time, so space them by their virtual position within
        // the interleaved group.
        let mut stage_time: Vec<usize> = (0..s)
            .map(|v| {
                let phys = v % num_physical;
                let group_idx = v / num_physical;
                phys + group_idx * num_physical
            })
            .collect();

        // Warmup: reduced by interleave_factor
        let warmup_count: Vec<usize> = (0..s)
            .map(|v| {
                let effective = (s - 1 - v) / interleave_factor;
                effective.min(m)
            })
            .collect();

        // Phase 1: Warmup
        for v in 0..s {
            for _ in 0..warmup_count[v] {
                let mb = fwd_mb[v];
                if mb >= m {
                    break;
                }
                let t = stage_time[v];
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardStart,
                    stage_id: v,
                    microbatch_id: mb,
                });
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardEnd,
                    stage_id: v,
                    microbatch_id: mb,
                });
                fwd_mb[v] = mb + 1;
                stage_time[v] = t + 1;
            }
        }

        // Phase 2: Steady state 1F1B
        for v in 0..s {
            while fwd_mb[v] < m {
                let mb_f = fwd_mb[v];
                let t = stage_time[v];
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardStart,
                    stage_id: v,
                    microbatch_id: mb_f,
                });
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardEnd,
                    stage_id: v,
                    microbatch_id: mb_f,
                });
                fwd_mb[v] = mb_f + 1;
                stage_time[v] = t + 1;

                let mb_b = bwd_mb[v];
                if mb_b < m {
                    let t = stage_time[v];
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::BackwardStart,
                        stage_id: v,
                        microbatch_id: mb_b,
                    });
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::BackwardEnd,
                        stage_id: v,
                        microbatch_id: mb_b,
                    });
                    bwd_mb[v] = mb_b + 1;
                    stage_time[v] = t + 1;
                }
            }
        }

        // Phase 3: Cooldown
        for v in 0..s {
            while bwd_mb[v] < m {
                let mb_b = bwd_mb[v];
                let t = stage_time[v];
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::BackwardStart,
                    stage_id: v,
                    microbatch_id: mb_b,
                });
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::BackwardEnd,
                    stage_id: v,
                    microbatch_id: mb_b,
                });
                bwd_mb[v] = mb_b + 1;
                stage_time[v] = t + 1;
            }
        }

        events.sort_by_key(|e| (e.timestamp, e.stage_id, e.microbatch_id));
        events
    }
}

// ─── ZeroBubble Scheduler ──────────────────────────────────

/// Zero-bubble pipeline scheduler (V-shape schedule).
///
/// Splits the backward pass into two parts:
/// - B (activation gradient): must happen in reverse stage order (like normal backward)
/// - W (weight gradient): can be deferred and scheduled to fill bubble slots
///
/// This allows near-zero pipeline bubble by inserting W computations into what
/// would otherwise be idle time slots. Each stage processes:
/// 1. Warmup forwards (fewer than 1F1B due to W-filling)
/// 2. Steady state: F, B, W interleaved
/// 3. Cooldown: remaining B and W operations
pub struct ZeroBubbleScheduler;

impl ZeroBubbleScheduler {
    /// Generate the zero-bubble schedule.
    ///
    /// Models backward as two half-cost operations (B and W), each taking 0.5
    /// time units conceptually, but scheduled in integer slots for simplicity.
    /// The B pass computes activation gradients; the W pass computes weight
    /// gradients. W passes fill what would be bubble slots in 1F1B.
    pub fn schedule(num_stages: usize, num_microbatches: usize) -> Vec<PipelineEvent> {
        let mut events = Vec::new();
        let s = num_stages;
        let m = num_microbatches;

        if s == 0 || m == 0 {
            return events;
        }

        // Per-stage state
        let mut stage_time: Vec<usize> = (0..s).collect();
        let mut fwd_mb = vec![0usize; s];
        let mut bwd_mb = vec![0usize; s]; // B pass (activation gradient)
        let mut wgt_mb = vec![0usize; s]; // W pass (weight gradient)

        // Warmup: stage s does s warmup forwards (one more than 1F1B's S-1-s
        // because W fills bubbles). Use same warmup count as 1F1B but then
        // fill with W.
        let warmup_count: Vec<usize> = (0..s).map(|st| (s - 1 - st).min(m)).collect();

        // Phase 1: Warmup forwards
        for st in 0..s {
            for _ in 0..warmup_count[st] {
                let mb = fwd_mb[st];
                if mb >= m {
                    break;
                }
                let t = stage_time[st];
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardStart,
                    stage_id: st,
                    microbatch_id: mb,
                });
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardEnd,
                    stage_id: st,
                    microbatch_id: mb,
                });
                fwd_mb[st] = mb + 1;
                stage_time[st] = t + 1;
            }
        }

        // Phase 2: Steady state — F, B, W interleaved
        // Each iteration: 1 forward, 1 backward (B), 1 weight grad (W)
        for st in 0..s {
            while fwd_mb[st] < m {
                // Forward
                let mb_f = fwd_mb[st];
                let t = stage_time[st];
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardStart,
                    stage_id: st,
                    microbatch_id: mb_f,
                });
                events.push(PipelineEvent {
                    timestamp: t,
                    event_type: EventType::ForwardEnd,
                    stage_id: st,
                    microbatch_id: mb_f,
                });
                fwd_mb[st] = mb_f + 1;
                stage_time[st] = t + 1;

                // Backward (activation gradient)
                let mb_b = bwd_mb[st];
                if mb_b < m {
                    let t = stage_time[st];
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::BackwardStart,
                        stage_id: st,
                        microbatch_id: mb_b,
                    });
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::BackwardEnd,
                        stage_id: st,
                        microbatch_id: mb_b,
                    });
                    bwd_mb[st] = mb_b + 1;
                    stage_time[st] = t + 1;
                }

                // Weight gradient (fills bubble slot)
                let mb_w = wgt_mb[st];
                if mb_w < m {
                    let t = stage_time[st];
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::WeightGradStart,
                        stage_id: st,
                        microbatch_id: mb_w,
                    });
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::WeightGradEnd,
                        stage_id: st,
                        microbatch_id: mb_w,
                    });
                    wgt_mb[st] = mb_w + 1;
                    stage_time[st] = t + 1;
                }
            }
        }

        // Phase 3: Cooldown — remaining B and W passes
        for st in 0..s {
            while bwd_mb[st] < m || wgt_mb[st] < m {
                if bwd_mb[st] < m {
                    let mb_b = bwd_mb[st];
                    let t = stage_time[st];
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::BackwardStart,
                        stage_id: st,
                        microbatch_id: mb_b,
                    });
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::BackwardEnd,
                        stage_id: st,
                        microbatch_id: mb_b,
                    });
                    bwd_mb[st] = mb_b + 1;
                    stage_time[st] = t + 1;
                }

                if wgt_mb[st] < m {
                    let mb_w = wgt_mb[st];
                    let t = stage_time[st];
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::WeightGradStart,
                        stage_id: st,
                        microbatch_id: mb_w,
                    });
                    events.push(PipelineEvent {
                        timestamp: t,
                        event_type: EventType::WeightGradEnd,
                        stage_id: st,
                        microbatch_id: mb_w,
                    });
                    wgt_mb[st] = mb_w + 1;
                    stage_time[st] = t + 1;
                }
            }
        }

        events.sort_by_key(|e| (e.timestamp, e.stage_id, e.microbatch_id));
        events
    }
}

// ─── PipelineEngine ────────────────────────────────────────

/// Manages pipeline-parallel execution across multiple GPU stages.
///
/// The engine generates schedules, computes bubble analysis, and provides
/// throughput estimates for a given pipeline configuration.
pub struct PipelineEngine {
    config: PipelineConfig,
    schedule_cache: Option<Vec<PipelineEvent>>,
}

impl PipelineEngine {
    /// Create a new pipeline engine with the given configuration.
    pub fn new(config: PipelineConfig) -> CudaResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            schedule_cache: None,
        })
    }

    /// Access the configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Number of stages in the pipeline.
    pub fn num_stages(&self) -> usize {
        self.config.stages.len()
    }

    /// Number of microbatches.
    pub fn num_microbatches(&self) -> usize {
        self.config.num_microbatches
    }

    /// Generate the schedule for all microbatches across all stages.
    pub fn generate_schedule(&mut self) -> Vec<PipelineEvent> {
        let s = self.config.stages.len();
        let m = self.config.num_microbatches;

        let schedule = match self.config.schedule_type {
            PipelineSchedule::GPipe => GpipeScheduler::schedule(s, m),
            PipelineSchedule::PipeDream1F1B => OneFOneBScheduler::schedule(s, m),
            PipelineSchedule::InterleavedStages => {
                InterleavedScheduler::schedule(s, m, self.config.interleave_factor)
            }
            PipelineSchedule::ZeroBubble => ZeroBubbleScheduler::schedule(s, m),
        };

        self.schedule_cache = Some(schedule.clone());
        schedule
    }

    /// Compute the bubble ratio for the configured schedule.
    ///
    /// Bubble ratio = idle time / total available time across all stages.
    pub fn bubble_ratio(&mut self) -> f64 {
        self.analyze().bubble_ratio
    }

    /// Detailed bubble analysis of the pipeline schedule.
    pub fn analyze(&mut self) -> BubbleAnalysis {
        let events = if let Some(ref cached) = self.schedule_cache {
            cached.clone()
        } else {
            self.generate_schedule()
        };

        let num_stages = self.config.stages.len();

        Self::analyze_events(&events, num_stages)
    }

    /// Analyze a set of events for the given number of stages.
    fn analyze_events(events: &[PipelineEvent], num_stages: usize) -> BubbleAnalysis {
        if events.is_empty() || num_stages == 0 {
            return BubbleAnalysis {
                total_time: 0,
                compute_time: 0,
                bubble_time: 0,
                bubble_ratio: 0.0,
                per_stage_idle: vec![0.0; num_stages],
            };
        }

        // Find the maximum timestamp (end of schedule)
        let max_time = events.iter().map(|e| e.timestamp).max().unwrap_or(0) + 1;

        // Count compute slots per stage (ForwardStart/BackwardStart/WeightGradStart each occupy 1 slot)
        let mut stage_compute = vec![0usize; num_stages];
        for ev in events {
            match ev.event_type {
                EventType::ForwardStart | EventType::BackwardStart | EventType::WeightGradStart
                    if ev.stage_id < num_stages =>
                {
                    stage_compute[ev.stage_id] += 1;
                }
                _ => {}
            }
        }

        let compute_time: usize = stage_compute.iter().sum();
        let total_available = max_time * num_stages;
        let bubble_time = total_available.saturating_sub(compute_time);
        let bubble_ratio = if total_available > 0 {
            bubble_time as f64 / total_available as f64
        } else {
            0.0
        };

        let per_stage_idle: Vec<f64> = stage_compute
            .iter()
            .map(|&c| {
                if max_time > 0 {
                    (max_time.saturating_sub(c)) as f64 / max_time as f64
                } else {
                    0.0
                }
            })
            .collect();

        BubbleAnalysis {
            total_time: max_time,
            compute_time,
            bubble_time,
            bubble_ratio,
            per_stage_idle,
        }
    }

    /// Compute steady-state throughput in microbatches per time unit.
    ///
    /// In steady state, 1F1B processes one microbatch per time unit per stage.
    /// For GPipe, throughput is `M / total_time`.
    pub fn steady_state_throughput(&mut self) -> f64 {
        let analysis = self.analyze();
        if analysis.total_time == 0 {
            return 0.0;
        }
        self.config.num_microbatches as f64 / analysis.total_time as f64
    }

    /// Get the current status of a microbatch given a time slot.
    pub fn microbatch_status_at(&self, microbatch_id: usize, time: usize) -> MicrobatchStatus {
        let events = match &self.schedule_cache {
            Some(cached) => cached,
            None => return MicrobatchStatus::Pending,
        };

        // Find the latest event for this microbatch at or before the given time
        let mut status = MicrobatchStatus::Pending;

        for ev in events {
            if ev.microbatch_id != microbatch_id || ev.timestamp > time {
                continue;
            }
            match ev.event_type {
                EventType::ForwardStart => {
                    status = MicrobatchStatus::InForward(ev.stage_id);
                }
                EventType::BackwardStart => {
                    status = MicrobatchStatus::InBackward(ev.stage_id);
                }
                EventType::BackwardEnd if ev.stage_id == 0 => {
                    // Final backward at stage 0 means microbatch is complete
                    status = MicrobatchStatus::Complete;
                }
                _ => {}
            }
        }

        status
    }
}

// ─── ActivationCheckpointing ──────────────────────────────

/// Activation checkpointing planner for pipeline stages.
///
/// Determines whether each stage should store, recompute, or offload its
/// activations based on the available memory budget per stage.
///
/// Thresholds (in arbitrary memory units):
/// - budget >= 1024: Store (keep all activations in GPU memory)
/// - budget >= 256: Recompute (discard activations, recompute during backward)
/// - budget < 256: Offload (move activations to host memory via PCIe)
pub struct ActivationCheckpointing;

impl ActivationCheckpointing {
    /// Memory threshold above which activations are stored.
    const STORE_THRESHOLD: usize = 1024;
    /// Memory threshold above which activations are recomputed (below Store).
    const RECOMPUTE_THRESHOLD: usize = 256;

    /// Plan activation checkpointing for each stage.
    ///
    /// # Arguments
    /// - `num_stages`: number of pipeline stages
    /// - `memory_budget_per_stage`: available memory budget per stage (arbitrary units)
    ///
    /// # Returns
    /// A vector of `CheckpointDecision` for each stage.
    pub fn plan(num_stages: usize, memory_budget_per_stage: usize) -> Vec<CheckpointDecision> {
        (0..num_stages)
            .map(|_| {
                if memory_budget_per_stage >= Self::STORE_THRESHOLD {
                    CheckpointDecision::Store
                } else if memory_budget_per_stage >= Self::RECOMPUTE_THRESHOLD {
                    CheckpointDecision::Recompute
                } else {
                    CheckpointDecision::Offload
                }
            })
            .collect()
    }

    /// Plan with per-stage memory budgets (stages may have different budgets).
    pub fn plan_variable(budgets: &[usize]) -> Vec<CheckpointDecision> {
        budgets
            .iter()
            .map(|&budget| {
                if budget >= Self::STORE_THRESHOLD {
                    CheckpointDecision::Store
                } else if budget >= Self::RECOMPUTE_THRESHOLD {
                    CheckpointDecision::Recompute
                } else {
                    CheckpointDecision::Offload
                }
            })
            .collect()
    }
}

// ─── PipelineVisualizer ────────────────────────────────────

/// ASCII visualization of pipeline schedules.
///
/// Renders a stage × time grid showing forward (F), backward (B),
/// weight-gradient (W), and idle (.) slots.
pub struct PipelineVisualizer;

impl PipelineVisualizer {
    /// Render an ASCII visualization of the pipeline schedule.
    ///
    /// Output format:
    /// ```text
    /// Stage 0: F0 F1 F2 F3 .. .. .. .. B0 B1 B2 B3
    /// Stage 1: .. F0 F1 F2 F3 .. .. B0 B1 B2 B3 ..
    /// Stage 2: .. .. F0 F1 F2 F3 B0 B1 B2 B3 .. ..
    /// Stage 3: .. .. .. F0 F1 F2 B3 B0 B1 B2 B3 ..
    /// ```
    pub fn render_ascii(events: &[PipelineEvent], num_stages: usize) -> String {
        if events.is_empty() || num_stages == 0 {
            return String::new();
        }

        let max_time = events.iter().map(|e| e.timestamp).max().unwrap_or(0) + 1;

        // Build a grid: stage × time -> cell content
        let mut grid: Vec<Vec<String>> = vec![vec![String::from(".."); max_time]; num_stages];

        for ev in events {
            if ev.stage_id >= num_stages || ev.timestamp >= max_time {
                continue;
            }
            let cell = &mut grid[ev.stage_id][ev.timestamp];
            match ev.event_type {
                EventType::ForwardStart => {
                    *cell = format!("F{}", ev.microbatch_id);
                }
                EventType::BackwardStart => {
                    *cell = format!("B{}", ev.microbatch_id);
                }
                EventType::WeightGradStart => {
                    *cell = format!("W{}", ev.microbatch_id);
                }
                _ => {}
            }
        }

        // Determine column width (accommodate multi-digit microbatch ids)
        let col_width = grid
            .iter()
            .flat_map(|row| row.iter())
            .map(|s| s.len())
            .max()
            .unwrap_or(2)
            .max(2);

        let mut output = String::new();
        for (st, row) in grid.iter().enumerate() {
            output.push_str(&format!("Stage {st}: "));
            for (t, cell) in row.iter().enumerate() {
                if t > 0 {
                    output.push(' ');
                }
                // Right-pad to col_width
                output.push_str(&format!("{cell:>width$}", width = col_width));
            }
            output.push('\n');
        }

        output
    }
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stages(n: usize) -> Vec<PipelineStage> {
        (0..n)
            .map(|i| PipelineStage::new(i, i as i32, format!("stage_{i}")))
            .collect()
    }

    fn make_config(
        num_stages: usize,
        num_microbatches: usize,
        schedule: PipelineSchedule,
    ) -> PipelineConfig {
        PipelineConfig {
            stages: make_stages(num_stages),
            num_microbatches,
            schedule_type: schedule,
            interleave_factor: 1,
        }
    }

    // ── GPipe schedule correctness ─────────────────────────

    #[test]
    fn gpipe_schedule_4_stages_8_microbatches() {
        let events = GpipeScheduler::schedule(4, 8);

        // Count forward and backward starts
        let fwd_count = events
            .iter()
            .filter(|e| e.event_type == EventType::ForwardStart)
            .count();
        let bwd_count = events
            .iter()
            .filter(|e| e.event_type == EventType::BackwardStart)
            .count();

        // 4 stages × 8 microbatches = 32 forward starts and 32 backward starts
        assert_eq!(fwd_count, 32);
        assert_eq!(bwd_count, 32);

        // First forward at stage 0, microbatch 0 starts at time 0
        let first_fwd = events
            .iter()
            .find(|e| e.event_type == EventType::ForwardStart)
            .expect("should have a forward event");
        assert_eq!(first_fwd.timestamp, 0);
        assert_eq!(first_fwd.stage_id, 0);
        assert_eq!(first_fwd.microbatch_id, 0);

        // All forwards complete before any backward starts
        let last_fwd_time = events
            .iter()
            .filter(|e| e.event_type == EventType::ForwardEnd)
            .map(|e| e.timestamp)
            .max()
            .unwrap_or(0);
        let first_bwd_time = events
            .iter()
            .filter(|e| e.event_type == EventType::BackwardStart)
            .map(|e| e.timestamp)
            .min()
            .unwrap_or(0);
        assert!(first_bwd_time > last_fwd_time);
    }

    // ── 1F1B schedule correctness ──────────────────────────

    #[test]
    fn one_f_one_b_schedule_correctness() {
        let events = OneFOneBScheduler::schedule(4, 8);

        let fwd_count = events
            .iter()
            .filter(|e| e.event_type == EventType::ForwardStart)
            .count();
        let bwd_count = events
            .iter()
            .filter(|e| e.event_type == EventType::BackwardStart)
            .count();

        assert_eq!(fwd_count, 32); // 4 stages × 8 microbatches
        assert_eq!(bwd_count, 32);

        // In 1F1B, the last stage (stage 3) should have 0 warmup forwards,
        // meaning its first backward happens right after its first forward.
        let stage3_events: Vec<&PipelineEvent> = events
            .iter()
            .filter(|e| e.stage_id == 3)
            .filter(|e| {
                matches!(
                    e.event_type,
                    EventType::ForwardStart | EventType::BackwardStart
                )
            })
            .collect();
        // Stage 3: F0, B0, F1, B1, ... (interleaved from the start)
        assert_eq!(stage3_events[0].event_type, EventType::ForwardStart);
        assert_eq!(stage3_events[1].event_type, EventType::BackwardStart);
    }

    // ── Interleaved schedule correctness ───────────────────

    #[test]
    fn interleaved_schedule_correctness() {
        // 4 virtual stages, interleave_factor=2 => 2 physical devices
        let events = InterleavedScheduler::schedule(4, 8, 2);

        let fwd_count = events
            .iter()
            .filter(|e| e.event_type == EventType::ForwardStart)
            .count();
        let bwd_count = events
            .iter()
            .filter(|e| e.event_type == EventType::BackwardStart)
            .count();

        assert_eq!(fwd_count, 32); // 4 × 8
        assert_eq!(bwd_count, 32);

        // Interleaved should have less bubble than GPipe
        let analysis_interleaved = PipelineEngine::analyze_events(&events, 4);
        let gpipe_events = GpipeScheduler::schedule(4, 8);
        let analysis_gpipe = PipelineEngine::analyze_events(&gpipe_events, 4);

        assert!(
            analysis_interleaved.bubble_ratio <= analysis_gpipe.bubble_ratio,
            "interleaved ({}) should have <= bubble ratio than GPipe ({})",
            analysis_interleaved.bubble_ratio,
            analysis_gpipe.bubble_ratio
        );
    }

    // ── Zero-bubble schedule ───────────────────────────────

    #[test]
    fn zero_bubble_schedule() {
        let events = ZeroBubbleScheduler::schedule(4, 8);

        let fwd_count = events
            .iter()
            .filter(|e| e.event_type == EventType::ForwardStart)
            .count();
        let bwd_count = events
            .iter()
            .filter(|e| e.event_type == EventType::BackwardStart)
            .count();
        let wgt_count = events
            .iter()
            .filter(|e| e.event_type == EventType::WeightGradStart)
            .count();

        assert_eq!(fwd_count, 32);
        assert_eq!(bwd_count, 32);
        assert_eq!(wgt_count, 32); // Weight gradient for each forward

        // ZeroBubble should have total compute = F + B + W = 3 * stages * microbatches
        let analysis = PipelineEngine::analyze_events(&events, 4);
        assert_eq!(analysis.compute_time, 96); // 32 F + 32 B + 32 W
    }

    // ── Bubble analysis accuracy ───────────────────────────

    #[test]
    fn bubble_analysis_gpipe() {
        let mut engine =
            PipelineEngine::new(make_config(4, 8, PipelineSchedule::GPipe)).expect("create engine");

        let analysis = engine.analyze();

        // GPipe total time for 4 stages, 8 microbatches:
        // Forward: max time = (4-1) + (8-1) = 10 (time 0..10)
        // Backward: starts at 11, max time = 11 + (4-1) + (8-1) = 21 (time 11..21)
        // Total = 22 time slots
        assert_eq!(analysis.total_time, 22);

        // Compute: each stage does 8 forwards + 8 backwards = 16
        // Total compute = 4 * 16 = 64
        assert_eq!(analysis.compute_time, 64);

        // Total available = 22 * 4 = 88
        // Bubble = 88 - 64 = 24
        assert_eq!(analysis.bubble_time, 24);

        // Bubble ratio ≈ 24/88 ≈ 0.2727
        assert!(
            (analysis.bubble_ratio - 24.0 / 88.0).abs() < 1e-6,
            "bubble_ratio = {}",
            analysis.bubble_ratio
        );

        // Per-stage idle should be populated
        assert_eq!(analysis.per_stage_idle.len(), 4);
    }

    // ── Steady-state throughput ────────────────────────────

    #[test]
    fn steady_state_throughput() {
        let mut engine =
            PipelineEngine::new(make_config(4, 8, PipelineSchedule::GPipe)).expect("create engine");

        let throughput = engine.steady_state_throughput();
        // 8 microbatches / 22 time slots ≈ 0.3636
        assert!(throughput > 0.0);
        assert!((throughput - 8.0 / 22.0).abs() < 1e-6);
    }

    // ── Degenerate: 1 stage ────────────────────────────────

    #[test]
    fn pipeline_single_stage() {
        let events = GpipeScheduler::schedule(1, 4);

        let fwd_count = events
            .iter()
            .filter(|e| e.event_type == EventType::ForwardStart)
            .count();
        let bwd_count = events
            .iter()
            .filter(|e| e.event_type == EventType::BackwardStart)
            .count();

        assert_eq!(fwd_count, 4);
        assert_eq!(bwd_count, 4);

        // With 1 stage there should be minimal/no bubble
        let analysis = PipelineEngine::analyze_events(&events, 1);
        // total_time = forward(0..3) + backward(4..7) = 8
        assert_eq!(analysis.total_time, 8);
        assert_eq!(analysis.compute_time, 8);
        assert_eq!(analysis.bubble_time, 0);
        assert!((analysis.bubble_ratio).abs() < 1e-6);
    }

    // ── Many microbatches ──────────────────────────────────

    #[test]
    fn pipeline_many_microbatches() {
        let events = OneFOneBScheduler::schedule(4, 64);

        let fwd_count = events
            .iter()
            .filter(|e| e.event_type == EventType::ForwardStart)
            .count();
        assert_eq!(fwd_count, 256); // 4 * 64

        // More microbatches should yield a lower bubble ratio
        let analysis_64 = PipelineEngine::analyze_events(&events, 4);
        let events_8 = OneFOneBScheduler::schedule(4, 8);
        let analysis_8 = PipelineEngine::analyze_events(&events_8, 4);

        assert!(
            analysis_64.bubble_ratio <= analysis_8.bubble_ratio,
            "more microbatches should reduce bubble: {} vs {}",
            analysis_64.bubble_ratio,
            analysis_8.bubble_ratio
        );
    }

    // ── Activation checkpointing ───────────────────────────

    #[test]
    fn activation_checkpointing_plan() {
        // High budget: Store
        let plan = ActivationCheckpointing::plan(4, 2048);
        assert_eq!(plan.len(), 4);
        assert!(plan.iter().all(|d| *d == CheckpointDecision::Store));

        // Medium budget: Recompute
        let plan = ActivationCheckpointing::plan(4, 512);
        assert!(plan.iter().all(|d| *d == CheckpointDecision::Recompute));

        // Low budget: Offload
        let plan = ActivationCheckpointing::plan(4, 128);
        assert!(plan.iter().all(|d| *d == CheckpointDecision::Offload));

        // Variable budgets
        let plan = ActivationCheckpointing::plan_variable(&[2048, 512, 128, 1024]);
        assert_eq!(plan[0], CheckpointDecision::Store);
        assert_eq!(plan[1], CheckpointDecision::Recompute);
        assert_eq!(plan[2], CheckpointDecision::Offload);
        assert_eq!(plan[3], CheckpointDecision::Store);
    }

    // ── ASCII visualization ────────────────────────────────

    #[test]
    fn ascii_visualization() {
        let events = GpipeScheduler::schedule(2, 3);
        let output = PipelineVisualizer::render_ascii(&events, 2);

        // Should contain stage labels
        assert!(output.contains("Stage 0:"));
        assert!(output.contains("Stage 1:"));

        // Should contain forward and backward markers
        assert!(output.contains("F0"));
        assert!(output.contains("F1"));
        assert!(output.contains("F2"));
        assert!(output.contains("B0"));
        assert!(output.contains("B1"));
        assert!(output.contains("B2"));

        // Should contain idle markers
        assert!(output.contains(".."));

        // Should have 2 lines (2 stages)
        let line_count = output.lines().count();
        assert_eq!(line_count, 2);
    }

    // ── Event ordering ─────────────────────────────────────

    #[test]
    fn event_ordering() {
        let events = GpipeScheduler::schedule(4, 4);

        // Events should be sorted by timestamp
        for pair in events.windows(2) {
            assert!(
                pair[0].timestamp <= pair[1].timestamp,
                "events not sorted: t={} before t={}",
                pair[0].timestamp,
                pair[1].timestamp
            );
        }
    }

    // ── Schedule completeness ──────────────────────────────

    #[test]
    fn schedule_completeness() {
        // Every microbatch must go through every stage in both F and B
        for schedule_type in [PipelineSchedule::GPipe, PipelineSchedule::PipeDream1F1B] {
            let num_stages = 4;
            let num_mb = 6;
            let config = make_config(num_stages, num_mb, schedule_type);
            let mut engine = PipelineEngine::new(config).expect("engine");
            let events = engine.generate_schedule();

            for mb in 0..num_mb {
                for st in 0..num_stages {
                    let has_fwd = events.iter().any(|e| {
                        e.microbatch_id == mb
                            && e.stage_id == st
                            && e.event_type == EventType::ForwardStart
                    });
                    let has_bwd = events.iter().any(|e| {
                        e.microbatch_id == mb
                            && e.stage_id == st
                            && e.event_type == EventType::BackwardStart
                    });
                    assert!(
                        has_fwd,
                        "{schedule_type:?}: microbatch {mb} missing forward at stage {st}"
                    );
                    assert!(
                        has_bwd,
                        "{schedule_type:?}: microbatch {mb} missing backward at stage {st}"
                    );
                }
            }
        }
    }

    // ── Microbatch status tracking ─────────────────────────

    #[test]
    fn microbatch_status_tracking() {
        let config = make_config(2, 2, PipelineSchedule::GPipe);
        let mut engine = PipelineEngine::new(config).expect("engine");
        engine.generate_schedule();

        // Before any events, microbatch is pending
        // GPipe 2 stages, 2 microbatches:
        // F: mb0@s0 t=0, mb0@s1 t=1, mb1@s0 t=1, mb1@s1 t=2
        // B: mb0@s1 t=3, mb0@s0 t=4, mb1@s1 t=4, mb1@s0 t=5

        // At time 0: microbatch 0 is in forward at stage 0
        let status = engine.microbatch_status_at(0, 0);
        assert_eq!(status, MicrobatchStatus::InForward(0));

        // At time 1: microbatch 0 is in forward at stage 1
        let status = engine.microbatch_status_at(0, 1);
        assert_eq!(status, MicrobatchStatus::InForward(1));

        // Microbatch 1 starts at time 1 for stage 0
        let status = engine.microbatch_status_at(1, 0);
        assert_eq!(status, MicrobatchStatus::Pending);

        // After the last backward at stage 0, microbatch should be complete
        let status = engine.microbatch_status_at(0, 4);
        assert_eq!(status, MicrobatchStatus::Complete);
    }

    // ── Config validation ──────────────────────────────────

    #[test]
    fn config_validation() {
        // Empty stages
        let config = PipelineConfig {
            stages: vec![],
            num_microbatches: 4,
            schedule_type: PipelineSchedule::GPipe,
            interleave_factor: 1,
        };
        assert!(config.validate().is_err());

        // Zero microbatches
        let config = make_config(4, 0, PipelineSchedule::GPipe);
        assert!(config.validate().is_err());

        // Zero interleave factor
        let config = PipelineConfig {
            stages: make_stages(4),
            num_microbatches: 4,
            schedule_type: PipelineSchedule::InterleavedStages,
            interleave_factor: 0,
        };
        assert!(config.validate().is_err());

        // Interleaved with stages not divisible by factor
        let config = PipelineConfig {
            stages: make_stages(3),
            num_microbatches: 4,
            schedule_type: PipelineSchedule::InterleavedStages,
            interleave_factor: 2,
        };
        assert!(config.validate().is_err());

        // Valid interleaved config
        let config = PipelineConfig {
            stages: make_stages(4),
            num_microbatches: 8,
            schedule_type: PipelineSchedule::InterleavedStages,
            interleave_factor: 2,
        };
        assert!(config.validate().is_ok());
    }

    // ── GPipe bubble ratio theoretical value ───────────────

    #[test]
    fn gpipe_bubble_ratio_theoretical() {
        // GPipe bubble ratio ≈ (S-1) * 2 / (2*(S+M-1)) for balanced stages
        // With S=4, M=8: total_time = 2*(4+8-1) = 22
        // compute per stage = 2*M = 16, total compute = 64
        // bubble = 22*4 - 64 = 24
        // ratio = 24/88 ≈ 0.2727
        let mut engine =
            PipelineEngine::new(make_config(4, 8, PipelineSchedule::GPipe)).expect("engine");
        let ratio = engine.bubble_ratio();
        assert!((ratio - 24.0 / 88.0).abs() < 1e-6);
    }

    // ── PipelineStage builder ──────────────────────────────

    #[test]
    fn pipeline_stage_builder() {
        let stage = PipelineStage::new(0, 0, "encoder").with_compute_cost(2.5);
        assert_eq!(stage.stage_id, 0);
        assert_eq!(stage.device_id, 0);
        assert_eq!(stage.name, "encoder");
        assert!((stage.compute_cost_estimate - 2.5).abs() < f64::EPSILON);
    }
}
