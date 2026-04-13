//! Distributed autotune coordination for multi-node GPU tuning.
//!
//! This module provides a coordination protocol layer for distributing
//! autotuning work across multiple GPU nodes. Actual networking is
//! abstracted behind the [`Transport`] trait, allowing pluggable
//! backends (TCP, MPI, gRPC, etc.) as well as an [`InMemoryTransport`]
//! for testing.
//!
//! # Architecture
//!
//! ```text
//!  ┌──────────────────────────────────────────────────────────────┐
//!  │                 DistributedCoordinator                       │
//!  │                                                              │
//!  │  configs ──► WorkDistributor ──► [TuneTask₁..TuneTaskₙ]     │
//!  │                                       │                      │
//!  │                      Transport::send ─┘                      │
//!  │                                                              │
//!  │  Node₁ ──► TaskComplete(results)  ┐                          │
//!  │  Node₂ ──► TaskComplete(results)  ├──► merge_results()       │
//!  │  Node₃ ──► TaskComplete(results)  ┘                          │
//!  └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::benchmark::BenchmarkResult;
use crate::config::Config;
use crate::error::AutotuneError;

// ---------------------------------------------------------------------------
// NodeId
// ---------------------------------------------------------------------------

/// Identifies a node in the distributed tuning cluster.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId {
    /// Unique numeric identifier for this node.
    pub id: u64,
    /// Human-readable hostname of the node.
    pub hostname: String,
}

impl NodeId {
    /// Creates a `NodeId` for the current process.
    ///
    /// The hostname is read from the `HOSTNAME` environment variable
    /// (falling back to `"unknown"`), and the id is derived from the
    /// current process ID.
    #[must_use]
    pub fn local() -> Self {
        let hostname = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string());
        let id = std::process::id() as u64;
        Self { id, hostname }
    }

    /// Creates a `NodeId` with explicit values.
    #[must_use]
    pub fn new(id: u64, hostname: impl Into<String>) -> Self {
        Self {
            id,
            hostname: hostname.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// TuneTask / TuneTaskResult
// ---------------------------------------------------------------------------

/// A unit of tuning work assigned to a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneTask {
    /// Unique task identifier within a job.
    pub task_id: u64,
    /// Name of the kernel being tuned (e.g. `"sgemm"`).
    pub kernel_name: String,
    /// Opaque problem key (e.g. `"1024x1024x1024"`).
    pub problem_key: String,
    /// Configurations this task should benchmark.
    pub configs: Vec<Config>,
    /// The node this task is assigned to, if any.
    pub assigned_node: Option<NodeId>,
}

/// Results produced by a node for a single [`TuneTask`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneTaskResult {
    /// The task these results correspond to.
    pub task_id: u64,
    /// The node that produced the results.
    pub node_id: NodeId,
    /// Benchmark results for each configuration in the task.
    pub results: Vec<BenchmarkResult>,
    /// Wall-clock time the node spent on this task (milliseconds).
    pub elapsed_ms: u64,
}

// ---------------------------------------------------------------------------
// WorkDistributor
// ---------------------------------------------------------------------------

/// Partitions configurations across nodes for distributed benchmarking.
pub struct WorkDistributor;

impl WorkDistributor {
    /// Distributes configs round-robin across `num_nodes` nodes.
    ///
    /// Each node gets roughly `configs.len() / num_nodes` configurations.
    /// Returns one [`TuneTask`] per node (empty tasks are omitted).
    #[must_use]
    pub fn distribute(
        kernel_name: &str,
        problem_key: &str,
        configs: &[Config],
        num_nodes: usize,
    ) -> Vec<TuneTask> {
        if num_nodes == 0 || configs.is_empty() {
            return Vec::new();
        }

        let mut buckets: Vec<Vec<Config>> = (0..num_nodes).map(|_| Vec::new()).collect();

        for (i, cfg) in configs.iter().enumerate() {
            buckets[i % num_nodes].push(cfg.clone());
        }

        buckets
            .into_iter()
            .enumerate()
            .filter(|(_, b)| !b.is_empty())
            .map(|(idx, cfgs)| TuneTask {
                task_id: idx as u64,
                kernel_name: kernel_name.to_string(),
                problem_key: problem_key.to_string(),
                configs: cfgs,
                assigned_node: None,
            })
            .collect()
    }

    /// Distributes configs proportionally to each node's capacity.
    ///
    /// `node_capacities` contains a relative weight for each node
    /// (e.g. `[1.0, 2.0]` gives the second node twice as many configs).
    /// Returns one [`TuneTask`] per node with configs.
    #[must_use]
    pub fn distribute_weighted(
        kernel_name: &str,
        problem_key: &str,
        configs: &[Config],
        node_capacities: &[f64],
    ) -> Vec<TuneTask> {
        if node_capacities.is_empty() || configs.is_empty() {
            return Vec::new();
        }

        let total_capacity: f64 = node_capacities.iter().sum();
        if total_capacity <= 0.0 {
            return Vec::new();
        }

        let n = configs.len();
        let num_nodes = node_capacities.len();

        // Compute target counts, distributing remainder to highest-capacity nodes.
        let mut counts: Vec<usize> = node_capacities
            .iter()
            .map(|c| ((c / total_capacity) * n as f64).floor() as usize)
            .collect();

        let assigned: usize = counts.iter().sum();
        let mut remainder = n.saturating_sub(assigned);

        // Distribute remainder by descending capacity.
        let mut order: Vec<usize> = (0..num_nodes).collect();
        order.sort_by(|&a, &b| {
            node_capacities[b]
                .partial_cmp(&node_capacities[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &idx in &order {
            if remainder == 0 {
                break;
            }
            counts[idx] += 1;
            remainder -= 1;
        }

        let mut offset = 0;
        let mut tasks = Vec::new();

        for (idx, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let end = (offset + count).min(n);
            let cfgs: Vec<Config> = configs[offset..end].to_vec();
            offset = end;

            tasks.push(TuneTask {
                task_id: idx as u64,
                kernel_name: kernel_name.to_string(),
                problem_key: problem_key.to_string(),
                configs: cfgs,
                assigned_node: None,
            });
        }

        tasks
    }
}

// ---------------------------------------------------------------------------
// DistributedTuneConfig
// ---------------------------------------------------------------------------

/// Configuration for the distributed tuning coordinator.
#[derive(Debug, Clone)]
pub struct DistributedTuneConfig {
    /// Total number of nodes in the cluster.
    pub num_nodes: usize,
    /// Identity of this node.
    pub node_id: NodeId,
    /// Identity of the coordinator node.
    pub coordinator_id: NodeId,
    /// Heartbeat interval in milliseconds.
    pub heartbeat_interval_ms: u64,
    /// Task timeout in milliseconds — how long to wait for a node
    /// to complete a task before considering it failed.
    pub task_timeout_ms: u64,
}

// ---------------------------------------------------------------------------
// TuneMessage
// ---------------------------------------------------------------------------

/// Messages exchanged between nodes in the distributed tuning protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TuneMessage {
    /// Coordinator assigns a task to a worker node.
    AssignTask(TuneTask),
    /// Worker reports completed benchmark results.
    TaskComplete(TuneTaskResult),
    /// Periodic liveness signal from a node.
    Heartbeat(NodeId),
    /// Worker requests additional work from coordinator.
    RequestWork(NodeId),
    /// Coordinator signals all workers to shut down.
    ShutdownSignal,
}

// ---------------------------------------------------------------------------
// Transport trait
// ---------------------------------------------------------------------------

/// Abstraction over network communication for distributed tuning.
///
/// Implementations can wrap TCP sockets, MPI communicators, gRPC
/// channels, or in-memory queues for testing.
pub trait Transport: Send + Sync {
    /// Sends a message to a specific node.
    fn send(&self, target: &NodeId, message: &TuneMessage) -> Result<(), AutotuneError>;

    /// Receives the next incoming message (blocking).
    fn recv(&self) -> Result<TuneMessage, AutotuneError>;

    /// Broadcasts a message to all nodes in the cluster.
    fn broadcast(&self, message: &TuneMessage) -> Result<(), AutotuneError>;
}

// ---------------------------------------------------------------------------
// InMemoryTransport
// ---------------------------------------------------------------------------

/// In-memory transport for testing distributed coordination logic.
///
/// Messages are stored in a shared queue (`Arc<Mutex<VecDeque>>`).
/// All `send` and `broadcast` calls push to the queue; `recv` pops
/// from the front.
#[derive(Debug, Clone)]
pub struct InMemoryTransport {
    queue: Arc<Mutex<VecDeque<TuneMessage>>>,
}

impl InMemoryTransport {
    /// Creates a new empty in-memory transport.
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Returns the number of messages currently in the queue.
    pub fn len(&self) -> Result<usize, AutotuneError> {
        let guard = self
            .queue
            .lock()
            .map_err(|e| AutotuneError::BenchmarkFailed(format!("lock poisoned: {e}")))?;
        Ok(guard.len())
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> Result<bool, AutotuneError> {
        Ok(self.len()? == 0)
    }
}

impl Default for InMemoryTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for InMemoryTransport {
    fn send(&self, _target: &NodeId, message: &TuneMessage) -> Result<(), AutotuneError> {
        let serialized = serde_json::to_string(message)?;
        let deserialized: TuneMessage = serde_json::from_str(&serialized)?;
        let mut guard = self
            .queue
            .lock()
            .map_err(|e| AutotuneError::BenchmarkFailed(format!("lock poisoned: {e}")))?;
        guard.push_back(deserialized);
        Ok(())
    }

    fn recv(&self) -> Result<TuneMessage, AutotuneError> {
        let mut guard = self
            .queue
            .lock()
            .map_err(|e| AutotuneError::BenchmarkFailed(format!("lock poisoned: {e}")))?;
        guard
            .pop_front()
            .ok_or_else(|| AutotuneError::BenchmarkFailed("no messages available".to_string()))
    }

    fn broadcast(&self, message: &TuneMessage) -> Result<(), AutotuneError> {
        // In-memory: just enqueue once (no real multi-node fanout).
        let serialized = serde_json::to_string(message)?;
        let deserialized: TuneMessage = serde_json::from_str(&serialized)?;
        let mut guard = self
            .queue
            .lock()
            .map_err(|e| AutotuneError::BenchmarkFailed(format!("lock poisoned: {e}")))?;
        guard.push_back(deserialized);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DistributedTuneJob
// ---------------------------------------------------------------------------

/// A distributed tuning job — tracks the state of a multi-node
/// benchmarking session.
#[derive(Debug, Clone)]
pub struct DistributedTuneJob {
    /// Unique job identifier.
    pub job_id: u64,
    /// Kernel being tuned.
    pub kernel_name: String,
    /// Problem size key.
    pub problem_key: String,
    /// Total number of configurations across all tasks.
    pub total_configs: usize,
    /// Individual tasks distributed to nodes.
    pub tasks: Vec<TuneTask>,
}

// ---------------------------------------------------------------------------
// DistributedCoordinator
// ---------------------------------------------------------------------------

/// Orchestrates multi-node distributed autotuning.
///
/// The coordinator partitions configurations across nodes using
/// [`WorkDistributor`], sends tasks via the [`Transport`], collects
/// results, and merges them to find the global best configuration.
pub struct DistributedCoordinator {
    config: DistributedTuneConfig,
    transport: Box<dyn Transport>,
    next_job_id: u64,
}

impl DistributedCoordinator {
    /// Creates a new coordinator with the given configuration and transport.
    pub fn new(config: DistributedTuneConfig, transport: Box<dyn Transport>) -> Self {
        Self {
            config,
            transport,
            next_job_id: 0,
        }
    }

    /// Returns a reference to the coordinator's configuration.
    #[must_use]
    pub fn config(&self) -> &DistributedTuneConfig {
        &self.config
    }

    /// Submits a tuning job: partitions configs and sends tasks to workers.
    ///
    /// Returns a [`DistributedTuneJob`] handle for tracking and
    /// collecting results.
    pub fn submit_tuning_job(
        &mut self,
        kernel: &str,
        problem: &str,
        configs: Vec<Config>,
    ) -> Result<DistributedTuneJob, AutotuneError> {
        let job_id = self.next_job_id;
        self.next_job_id += 1;

        let total_configs = configs.len();
        let mut tasks =
            WorkDistributor::distribute(kernel, problem, &configs, self.config.num_nodes);

        // Assign task IDs relative to the job and send to workers.
        for (i, task) in tasks.iter_mut().enumerate() {
            task.task_id = job_id * 1000 + i as u64;
            let node = NodeId::new(i as u64, format!("worker-{i}"));
            task.assigned_node = Some(node.clone());

            let msg = TuneMessage::AssignTask(task.clone());
            self.transport.send(&node, &msg)?;
        }

        Ok(DistributedTuneJob {
            job_id,
            kernel_name: kernel.to_string(),
            problem_key: problem.to_string(),
            total_configs,
            tasks,
        })
    }

    /// Collects results for a distributed tuning job.
    ///
    /// Receives one [`TuneMessage::TaskComplete`] per task and merges
    /// all benchmark results.
    pub fn collect_results(
        &mut self,
        job: &DistributedTuneJob,
    ) -> Result<Vec<BenchmarkResult>, AutotuneError> {
        let mut all_results: Vec<Vec<BenchmarkResult>> = Vec::new();
        let mut completed = 0;
        let expected = job.tasks.len();

        while completed < expected {
            let msg = self.transport.recv()?;
            match msg {
                TuneMessage::TaskComplete(result) => {
                    all_results.push(result.results);
                    completed += 1;
                }
                TuneMessage::Heartbeat(_) => {
                    // Heartbeats are acknowledged but don't count as task completion.
                    // In a real implementation we'd track liveness here.
                }
                _ => {
                    return Err(AutotuneError::BenchmarkFailed(
                        "unexpected message type while collecting results".to_string(),
                    ));
                }
            }
        }

        Ok(merge_results(&all_results))
    }
}

// ---------------------------------------------------------------------------
// merge_results
// ---------------------------------------------------------------------------

/// Merges benchmark results from multiple nodes.
///
/// When multiple results exist for the same configuration (identified
/// by the `Config` value), only the result with the lowest
/// `median_us` is kept.
#[must_use]
pub fn merge_results(results: &[Vec<BenchmarkResult>]) -> Vec<BenchmarkResult> {
    use std::collections::HashMap;

    let mut best_map: HashMap<Config, BenchmarkResult> = HashMap::new();

    for batch in results {
        for result in batch {
            let entry = best_map.entry(result.config.clone());
            match entry {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(result.clone());
                }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    if result.median_us < e.get().median_us {
                        e.insert(result.clone());
                    }
                }
            }
        }
    }

    let mut merged: Vec<BenchmarkResult> = best_map.into_values().collect();
    // Sort by median_us for deterministic output.
    merged.sort_by(|a, b| {
        a.median_us
            .partial_cmp(&b.median_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merged
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(tile_m: u32) -> Config {
        Config::new().with_tile_m(tile_m)
    }

    fn make_benchmark_result(tile_m: u32, median_us: f64) -> BenchmarkResult {
        BenchmarkResult {
            config: make_config(tile_m),
            median_us,
            min_us: median_us * 0.95,
            max_us: median_us * 1.05,
            stddev_us: median_us * 0.02,
            gflops: None,
            efficiency: None,
        }
    }

    #[test]
    fn node_id_local_creation() {
        let node = NodeId::local();
        assert!(node.id > 0);
        // Hostname may be "unknown" in CI environments, just check it's non-empty.
        assert!(!node.hostname.is_empty());
    }

    #[test]
    fn node_id_new_creation() {
        let node = NodeId::new(42, "gpu-server-1");
        assert_eq!(node.id, 42);
        assert_eq!(node.hostname, "gpu-server-1");
    }

    #[test]
    fn distribute_round_robin() {
        let configs: Vec<Config> = (0..6).map(|i| make_config(32 * (i + 1))).collect();
        let tasks = WorkDistributor::distribute("sgemm", "1024x1024", &configs, 3);

        assert_eq!(tasks.len(), 3);
        // Round-robin: 0,3 | 1,4 | 2,5
        assert_eq!(tasks[0].configs.len(), 2);
        assert_eq!(tasks[1].configs.len(), 2);
        assert_eq!(tasks[2].configs.len(), 2);
        assert_eq!(tasks[0].configs[0].tile_m, 32);
        assert_eq!(tasks[0].configs[1].tile_m, 128); // 32*4
    }

    #[test]
    fn distribute_weighted() {
        let configs: Vec<Config> = (0..10).map(|i| make_config(32 * (i + 1))).collect();
        let tasks =
            WorkDistributor::distribute_weighted("sgemm", "2048x2048", &configs, &[1.0, 3.0]);

        assert_eq!(tasks.len(), 2);
        // With capacities 1:3, expect roughly 2-3 and 7-8 configs.
        let total: usize = tasks.iter().map(|t| t.configs.len()).sum();
        assert_eq!(total, 10);
        assert!(tasks[1].configs.len() > tasks[0].configs.len());
    }

    #[test]
    fn distribute_empty_configs() {
        let tasks = WorkDistributor::distribute("sgemm", "empty", &[], 4);
        assert!(tasks.is_empty());
    }

    #[test]
    fn distribute_single_config() {
        let configs = vec![make_config(64)];
        let tasks = WorkDistributor::distribute("sgemm", "small", &configs, 4);

        // Only 1 config, so only 1 task (empty ones are omitted).
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].configs.len(), 1);
        assert_eq!(tasks[0].configs[0].tile_m, 64);
    }

    #[test]
    fn tune_message_serialization() {
        let task = TuneTask {
            task_id: 1,
            kernel_name: "sgemm".to_string(),
            problem_key: "1024x1024".to_string(),
            configs: vec![make_config(128)],
            assigned_node: Some(NodeId::new(0, "node-0")),
        };

        let msg = TuneMessage::AssignTask(task);
        let json = serde_json::to_string(&msg).expect("serialize TuneMessage");
        let restored: TuneMessage = serde_json::from_str(&json).expect("deserialize TuneMessage");

        if let TuneMessage::AssignTask(t) = restored {
            assert_eq!(t.task_id, 1);
            assert_eq!(t.kernel_name, "sgemm");
            assert_eq!(t.configs.len(), 1);
        } else {
            panic!("expected AssignTask variant");
        }
    }

    #[test]
    fn tune_message_heartbeat_serialization() {
        let msg = TuneMessage::Heartbeat(NodeId::new(5, "worker-5"));
        let json = serde_json::to_string(&msg).expect("serialize");
        let restored: TuneMessage = serde_json::from_str(&json).expect("deserialize");

        if let TuneMessage::Heartbeat(node) = restored {
            assert_eq!(node.id, 5);
            assert_eq!(node.hostname, "worker-5");
        } else {
            panic!("expected Heartbeat variant");
        }
    }

    #[test]
    fn in_memory_transport_send_recv() {
        let transport = InMemoryTransport::new();
        let target = NodeId::new(1, "target");
        let msg = TuneMessage::Heartbeat(NodeId::new(0, "sender"));

        transport.send(&target, &msg).expect("send should succeed");
        assert_eq!(transport.len().expect("len"), 1);

        let received = transport.recv().expect("recv should succeed");
        if let TuneMessage::Heartbeat(node) = received {
            assert_eq!(node.id, 0);
        } else {
            panic!("expected Heartbeat");
        }

        assert!(transport.is_empty().expect("is_empty"));
    }

    #[test]
    fn in_memory_transport_recv_empty_returns_error() {
        let transport = InMemoryTransport::new();
        let result = transport.recv();
        assert!(result.is_err());
    }

    #[test]
    fn merge_results_deduplication() {
        let batch1 = vec![make_benchmark_result(64, 100.0)];
        let batch2 = vec![make_benchmark_result(64, 90.0)];

        let merged = merge_results(&[batch1, batch2]);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].median_us - 90.0).abs() < f64::EPSILON);
    }

    #[test]
    fn merge_results_keep_best() {
        let batch1 = vec![
            make_benchmark_result(64, 100.0),
            make_benchmark_result(128, 80.0),
        ];
        let batch2 = vec![
            make_benchmark_result(64, 50.0),   // better
            make_benchmark_result(128, 120.0), // worse
        ];

        let merged = merge_results(&[batch1, batch2]);
        assert_eq!(merged.len(), 2);

        // Results are sorted by median_us.
        let r64 = merged.iter().find(|r| r.config.tile_m == 64).expect("64");
        let r128 = merged.iter().find(|r| r.config.tile_m == 128).expect("128");
        assert!((r64.median_us - 50.0).abs() < f64::EPSILON);
        assert!((r128.median_us - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn job_creation_via_coordinator() {
        let transport = InMemoryTransport::new();
        let config = DistributedTuneConfig {
            num_nodes: 2,
            node_id: NodeId::new(0, "coordinator"),
            coordinator_id: NodeId::new(0, "coordinator"),
            heartbeat_interval_ms: 1000,
            task_timeout_ms: 30000,
        };

        let mut coordinator = DistributedCoordinator::new(config, Box::new(transport));

        let configs: Vec<Config> = (0..4).map(|i| make_config(32 * (i + 1))).collect();
        let job = coordinator
            .submit_tuning_job("dgemm", "512x512", configs)
            .expect("submit should succeed");

        assert_eq!(job.job_id, 0);
        assert_eq!(job.kernel_name, "dgemm");
        assert_eq!(job.total_configs, 4);
        assert_eq!(job.tasks.len(), 2);
    }

    #[test]
    fn task_assignment_sets_node() {
        let transport = InMemoryTransport::new();
        let config = DistributedTuneConfig {
            num_nodes: 2,
            node_id: NodeId::new(0, "coordinator"),
            coordinator_id: NodeId::new(0, "coordinator"),
            heartbeat_interval_ms: 1000,
            task_timeout_ms: 30000,
        };

        let mut coordinator = DistributedCoordinator::new(config, Box::new(transport));

        let configs = vec![make_config(64), make_config(128)];
        let job = coordinator
            .submit_tuning_job("sgemm", "256x256", configs)
            .expect("submit");

        for task in &job.tasks {
            assert!(
                task.assigned_node.is_some(),
                "each task should have an assigned node"
            );
        }
    }

    #[test]
    fn heartbeat_handling_in_coordinator() {
        let transport = InMemoryTransport::new();
        let node = NodeId::new(1, "worker-1");

        // Pre-populate the transport with heartbeat + task_complete messages.
        let heartbeat = TuneMessage::Heartbeat(node.clone());
        let task_complete = TuneMessage::TaskComplete(TuneTaskResult {
            task_id: 0,
            node_id: node,
            results: vec![make_benchmark_result(64, 42.0)],
            elapsed_ms: 100,
        });

        transport
            .send(&NodeId::new(0, "coord"), &heartbeat)
            .expect("send heartbeat");
        transport
            .send(&NodeId::new(0, "coord"), &task_complete)
            .expect("send result");

        let config = DistributedTuneConfig {
            num_nodes: 1,
            node_id: NodeId::new(0, "coordinator"),
            coordinator_id: NodeId::new(0, "coordinator"),
            heartbeat_interval_ms: 1000,
            task_timeout_ms: 30000,
        };

        // Build a minimal job with 1 task to collect.
        let job = DistributedTuneJob {
            job_id: 0,
            kernel_name: "sgemm".to_string(),
            problem_key: "test".to_string(),
            total_configs: 1,
            tasks: vec![TuneTask {
                task_id: 0,
                kernel_name: "sgemm".to_string(),
                problem_key: "test".to_string(),
                configs: vec![make_config(64)],
                assigned_node: None,
            }],
        };

        let mut coordinator = DistributedCoordinator::new(config, Box::new(transport));
        // collect_results should skip the heartbeat and pick up the TaskComplete.
        let results = coordinator.collect_results(&job).expect("collect");
        assert_eq!(results.len(), 1);
        assert!((results[0].median_us - 42.0).abs() < f64::EPSILON);
    }
}
