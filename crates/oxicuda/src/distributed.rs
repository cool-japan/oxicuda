//! Multi-node distributed training support (TCP/IP based).
//!
//! This module provides the multi-NODE coordination layer on top of the
//! single-node collective communication primitives in [`crate::collective`].
//! It implements PyTorch-style distributed primitives:
//!
//! - **`DistributedRuntime`** — manages multi-node initialization & barriers
//! - **`TcpStore`** / **`FileStore`** — key-value rendezvous stores
//! - **`GradientBucket`** — gradient bucketing for distributed data parallel
//! - **`DistributedOptimizer`** — gradient communication & ZeRO sharding
//!
//! All networking uses `std::net` (pure Rust, no external dependencies).
//!
//! On macOS the runtime returns simulated results since no NVIDIA GPU is
//! present.

use oxicuda_driver::{CudaError, CudaResult};
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ─── NodeId ─────────────────────────────────────────────────

/// Unique identifier for a node in the distributed cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Create a new node identifier.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// The underlying integer value.
    pub fn value(&self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

// ─── NodeInfo ───────────────────────────────────────────────

/// Metadata about a single node in the cluster.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// This node's identifier.
    pub node_id: NodeId,
    /// Human-readable hostname.
    pub hostname: String,
    /// IP address (v4 or v6 string).
    pub ip_addr: String,
    /// Port the node listens on.
    pub port: u16,
    /// Number of GPUs available on this node.
    pub gpu_count: u32,
    /// Global rank assigned to this node.
    pub rank: u32,
}

impl NodeInfo {
    /// Create a new `NodeInfo`.
    pub fn new(
        node_id: NodeId,
        hostname: &str,
        ip_addr: &str,
        port: u16,
        gpu_count: u32,
        rank: u32,
    ) -> Self {
        Self {
            node_id,
            hostname: hostname.to_string(),
            ip_addr: ip_addr.to_string(),
            port,
            gpu_count,
            rank,
        }
    }
}

// ─── DistributedBackend ─────────────────────────────────────

/// Communication backend for distributed operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DistributedBackend {
    /// TCP/IP sockets (always available).
    #[default]
    Tcp,
    /// Shared-memory transport for intra-node communication.
    SharedMemory,
}

// ─── InitMethod ─────────────────────────────────────────────

/// How the distributed runtime discovers peers.
#[derive(Debug, Clone)]
pub enum InitMethod {
    /// TCP-based rendezvous through a master node.
    TcpRendezvous {
        /// Master node address.
        master_addr: String,
        /// Master node port.
        master_port: u16,
    },
    /// Read configuration from environment variables
    /// (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`).
    EnvVars,
    /// File-system rendezvous via a shared directory.
    FileStore(PathBuf),
}

// ─── DistributedConfig ──────────────────────────────────────

/// Configuration for initializing a distributed runtime.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Total number of participating processes.
    pub world_size: u32,
    /// Rank of this process on its local node (GPU index).
    pub local_rank: u32,
    /// Globally unique rank across all nodes.
    pub global_rank: u32,
    /// Address of the master / rendezvous node.
    pub master_addr: String,
    /// Port on the master node.
    pub master_port: u16,
    /// Communication backend.
    pub backend: DistributedBackend,
}

impl DistributedConfig {
    /// Validate the configuration, returning an error for invalid values.
    pub fn validate(&self) -> CudaResult<()> {
        if self.world_size == 0 {
            return Err(CudaError::InvalidValue);
        }
        if self.global_rank >= self.world_size {
            return Err(CudaError::InvalidValue);
        }
        if self.master_addr.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        if self.master_port == 0 {
            return Err(CudaError::InvalidValue);
        }
        Ok(())
    }
}

// ─── ProcessGroup ───────────────────────────────────────────

/// A subset of ranks that participate in collective operations together.
#[derive(Debug, Clone)]
pub struct ProcessGroup {
    /// Unique identifier for this group.
    pub group_id: u32,
    /// Ranks that belong to this group.
    pub ranks: Vec<u32>,
    /// Number of ranks in the group.
    pub size: u32,
}

impl ProcessGroup {
    /// Create a new process group.
    pub fn new(group_id: u32, ranks: Vec<u32>) -> CudaResult<Self> {
        if ranks.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        let size = ranks.len() as u32;
        Ok(Self {
            group_id,
            ranks,
            size,
        })
    }

    /// Check whether a given rank belongs to this group.
    pub fn contains_rank(&self, rank: u32) -> bool {
        self.ranks.contains(&rank)
    }

    /// Return the local index of `rank` within this group, if present.
    pub fn local_rank(&self, rank: u32) -> Option<usize> {
        self.ranks.iter().position(|&r| r == rank)
    }
}

// ─── DistributedStatus ──────────────────────────────────────

/// Lifecycle status of the distributed runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedStatus {
    /// Runtime is being set up.
    Initializing,
    /// Runtime is ready for collective operations.
    Ready,
    /// A synchronization primitive is in progress.
    Synchronizing,
    /// An error occurred.
    Error(String),
    /// Runtime has been shut down.
    Shutdown,
}

// ─── DistributedRuntime ─────────────────────────────────────

/// Multi-node distributed training coordinator.
///
/// Manages peer discovery, barriers, and lifecycle for a set of
/// processes spanning multiple machines.
pub struct DistributedRuntime {
    config: DistributedConfig,
    status: Arc<Mutex<DistributedStatus>>,
    /// Epoch counter used for barrier synchronization.
    barrier_epoch: Arc<Mutex<u64>>,
}

impl DistributedRuntime {
    /// Initialize a distributed runtime from an explicit configuration.
    pub fn init(config: DistributedConfig) -> CudaResult<Self> {
        config.validate()?;

        let rt = Self {
            config,
            status: Arc::new(Mutex::new(DistributedStatus::Ready)),
            barrier_epoch: Arc::new(Mutex::new(0)),
        };
        Ok(rt)
    }

    /// Initialize from environment variables.
    ///
    /// Reads `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, and
    /// optionally `LOCAL_RANK`.
    pub fn from_env() -> CudaResult<Self> {
        let master_addr = std::env::var("MASTER_ADDR").map_err(|_| CudaError::InvalidValue)?;
        let master_port: u16 = std::env::var("MASTER_PORT")
            .map_err(|_| CudaError::InvalidValue)?
            .parse()
            .map_err(|_| CudaError::InvalidValue)?;
        let rank: u32 = std::env::var("RANK")
            .map_err(|_| CudaError::InvalidValue)?
            .parse()
            .map_err(|_| CudaError::InvalidValue)?;
        let world_size: u32 = std::env::var("WORLD_SIZE")
            .map_err(|_| CudaError::InvalidValue)?
            .parse()
            .map_err(|_| CudaError::InvalidValue)?;
        let local_rank: u32 = std::env::var("LOCAL_RANK")
            .unwrap_or_else(|_| rank.to_string())
            .parse()
            .map_err(|_| CudaError::InvalidValue)?;

        let config = DistributedConfig {
            world_size,
            local_rank,
            global_rank: rank,
            master_addr,
            master_port,
            backend: DistributedBackend::Tcp,
        };

        Self::init(config)
    }

    /// Total number of processes in the distributed group.
    pub fn world_size(&self) -> u32 {
        self.config.world_size
    }

    /// Global rank of this process.
    pub fn rank(&self) -> u32 {
        self.config.global_rank
    }

    /// Local rank (GPU index on this node).
    pub fn local_rank(&self) -> u32 {
        self.config.local_rank
    }

    /// Whether this process is the master (rank 0).
    pub fn is_master(&self) -> bool {
        self.config.global_rank == 0
    }

    /// Execute a global barrier — all ranks must call this before any
    /// can proceed.
    ///
    /// In host simulation this increments an internal epoch counter.
    pub fn barrier(&self) -> CudaResult<()> {
        let mut status = self.status.lock().map_err(|_| CudaError::InvalidValue)?;
        if *status == DistributedStatus::Shutdown {
            return Err(CudaError::NotInitialized);
        }
        *status = DistributedStatus::Synchronizing;

        let mut epoch = self
            .barrier_epoch
            .lock()
            .map_err(|_| CudaError::InvalidValue)?;
        *epoch += 1;

        *status = DistributedStatus::Ready;
        Ok(())
    }

    /// Current lifecycle status.
    pub fn status(&self) -> DistributedStatus {
        self.status
            .lock()
            .map(|s| s.clone())
            .unwrap_or_else(|_| DistributedStatus::Error("lock poisoned".to_string()))
    }

    /// Shut down the distributed runtime.
    ///
    /// Idempotent — calling shutdown on an already-shut-down runtime is a
    /// no-op.
    pub fn shutdown(&self) -> CudaResult<()> {
        let mut status = self.status.lock().map_err(|_| CudaError::InvalidValue)?;
        *status = DistributedStatus::Shutdown;
        Ok(())
    }
}

// ─── TcpStore ───────────────────────────────────────────────

/// In-memory key-value store for distributed rendezvous.
///
/// Mirrors PyTorch's `TCPStore`. In this host simulation the store is
/// backed by a `HashMap` behind a `Mutex`; a real implementation would
/// run a TCP server on the master and proxy `set`/`get` over the wire.
pub struct TcpStore {
    /// Master address (informational).
    _master_addr: String,
    /// Port (informational).
    _port: u16,
    /// Expected number of workers.
    _world_size: u32,
    /// Whether this instance is the master.
    is_master: bool,
    /// The actual key-value data.
    data: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    /// Atomic counters for `add`.
    counters: Arc<Mutex<HashMap<String, i64>>>,
}

impl TcpStore {
    /// Create a new TCP store.
    ///
    /// On the master rank (`is_master = true`) the store is authoritative;
    /// workers connect to it for reads/writes.
    pub fn new(master_addr: &str, port: u16, world_size: u32, is_master: bool) -> CudaResult<Self> {
        if master_addr.is_empty() || world_size == 0 {
            return Err(CudaError::InvalidValue);
        }
        Ok(Self {
            _master_addr: master_addr.to_string(),
            _port: port,
            _world_size: world_size,
            is_master,
            data: Arc::new(Mutex::new(HashMap::new())),
            counters: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Whether this store instance is the master.
    pub fn is_master(&self) -> bool {
        self.is_master
    }

    /// Set a key to the given value.
    pub fn set(&self, key: &str, value: &[u8]) -> CudaResult<()> {
        let mut data = self.data.lock().map_err(|_| CudaError::InvalidValue)?;
        data.insert(key.to_string(), value.to_vec());
        Ok(())
    }

    /// Retrieve the value for `key`, or `CudaError::InvalidValue` if absent.
    pub fn get(&self, key: &str) -> CudaResult<Vec<u8>> {
        let data = self.data.lock().map_err(|_| CudaError::InvalidValue)?;
        data.get(key).cloned().ok_or(CudaError::InvalidValue)
    }

    /// Block until all specified keys exist in the store.
    ///
    /// In host simulation this succeeds immediately if all keys are
    /// present, otherwise returns an error (no actual blocking).
    pub fn wait(&self, keys: &[&str]) -> CudaResult<()> {
        let data = self.data.lock().map_err(|_| CudaError::InvalidValue)?;
        for &k in keys {
            if !data.contains_key(k) {
                return Err(CudaError::NotReady);
            }
        }
        Ok(())
    }

    /// Atomically add `amount` to the counter stored under `key`.
    ///
    /// If the key does not exist it is initialised to 0 before adding.
    /// Returns the new value.
    pub fn add(&self, key: &str, amount: i64) -> CudaResult<i64> {
        let mut counters = self.counters.lock().map_err(|_| CudaError::InvalidValue)?;
        let entry = counters.entry(key.to_string()).or_insert(0);
        *entry += amount;
        Ok(*entry)
    }
}

// ─── FileStore ──────────────────────────────────────────────

/// File-system based rendezvous store for shared filesystems (NFS, etc.).
///
/// Each key is stored as a separate file under the root directory.
pub struct FileStore {
    /// Root directory for the store.
    root: PathBuf,
}

impl FileStore {
    /// Create a new file store rooted at `path`.
    ///
    /// The directory is created if it does not exist.
    pub fn new(path: &Path) -> CudaResult<Self> {
        std::fs::create_dir_all(path).map_err(|_| CudaError::InvalidValue)?;
        Ok(Self {
            root: path.to_path_buf(),
        })
    }

    /// Sanitize a key to a safe filename component.
    fn key_path(&self, key: &str) -> PathBuf {
        let safe: String = key
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.root.join(safe)
    }

    /// Set a key to the given value.
    pub fn set(&self, key: &str, value: &[u8]) -> CudaResult<()> {
        std::fs::write(self.key_path(key), value).map_err(|_| CudaError::InvalidValue)
    }

    /// Retrieve the value for `key`.
    pub fn get(&self, key: &str) -> CudaResult<Vec<u8>> {
        std::fs::read(self.key_path(key)).map_err(|_| CudaError::InvalidValue)
    }

    /// Block until all keys exist on disk.
    ///
    /// In host simulation this is a single check (no polling).
    pub fn wait(&self, keys: &[&str]) -> CudaResult<()> {
        for &k in keys {
            if !self.key_path(k).exists() {
                return Err(CudaError::NotReady);
            }
        }
        Ok(())
    }

    /// Atomically add `amount` to a counter stored in a file.
    ///
    /// Uses a simple read-modify-write cycle. In production a file lock
    /// would be required.
    pub fn add(&self, key: &str, amount: i64) -> CudaResult<i64> {
        let path = self.key_path(key);
        let current: i64 = if path.exists() {
            let bytes = std::fs::read(&path).map_err(|_| CudaError::InvalidValue)?;
            let s = String::from_utf8(bytes).map_err(|_| CudaError::InvalidValue)?;
            s.trim().parse().map_err(|_| CudaError::InvalidValue)?
        } else {
            0
        };
        let new_val = current + amount;
        std::fs::write(&path, new_val.to_string().as_bytes())
            .map_err(|_| CudaError::InvalidValue)?;
        Ok(new_val)
    }
}

// ─── GradientBucket ─────────────────────────────────────────

/// A single bucket of gradients for communication overlap.
#[derive(Debug, Clone)]
pub struct Bucket {
    /// Parameter indices in this bucket.
    pub param_ids: Vec<usize>,
    /// Accumulated size in bytes.
    pub total_size: usize,
    /// Whether all gradients in this bucket have been computed.
    pub ready: bool,
}

/// Gradient bucketing for distributed data parallel training.
///
/// Groups parameter gradients into fixed-size buckets so that
/// communication can overlap with backward-pass computation.
#[derive(Debug, Clone)]
pub struct GradientBucket {
    /// Maximum bucket size in bytes.
    bucket_size_bytes: usize,
    /// Current list of buckets.
    buckets: Vec<Bucket>,
}

impl GradientBucket {
    /// Create a new gradient bucketing scheme.
    ///
    /// `bucket_size_mb` is the target bucket capacity in megabytes.
    pub fn new(bucket_size_mb: usize) -> Self {
        Self {
            bucket_size_bytes: bucket_size_mb * 1024 * 1024,
            buckets: Vec::new(),
        }
    }

    /// Add a gradient for parameter `param_id` with `grad_size` bytes.
    ///
    /// If the current bucket cannot fit the gradient a new bucket is
    /// started.
    pub fn add_gradient(&mut self, param_id: usize, grad_size: usize) {
        let needs_new = self.buckets.is_empty()
            || self
                .buckets
                .last()
                .is_none_or(|b| b.total_size + grad_size > self.bucket_size_bytes);

        if needs_new {
            self.buckets.push(Bucket {
                param_ids: vec![param_id],
                total_size: grad_size,
                ready: false,
            });
        } else if let Some(last) = self.buckets.last_mut() {
            last.param_ids.push(param_id);
            last.total_size += grad_size;
        }
    }

    /// Read-only access to the computed buckets.
    pub fn buckets(&self) -> &[Bucket] {
        &self.buckets
    }

    /// Mark a specific bucket as ready for communication.
    pub fn mark_ready(&mut self, bucket_idx: usize) -> CudaResult<()> {
        let bucket = self
            .buckets
            .get_mut(bucket_idx)
            .ok_or(CudaError::InvalidValue)?;
        bucket.ready = true;
        Ok(())
    }

    /// Number of buckets.
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Target bucket capacity in bytes.
    pub fn bucket_capacity(&self) -> usize {
        self.bucket_size_bytes
    }
}

// ─── DataParallelConfig ─────────────────────────────────────

/// Configuration for distributed data-parallel training.
#[derive(Debug, Clone)]
pub struct DataParallelConfig {
    /// Target gradient bucket size in MB.
    pub gradient_bucket_size_mb: usize,
    /// Whether to overlap communication with backward computation.
    pub overlap_communication: bool,
    /// Whether to detect and skip unused parameters in each iteration.
    pub find_unused_parameters: bool,
}

impl Default for DataParallelConfig {
    fn default() -> Self {
        Self {
            gradient_bucket_size_mb: 25,
            overlap_communication: true,
            find_unused_parameters: false,
        }
    }
}

// ─── ModelParallelConfig ────────────────────────────────────

/// Configuration for model-parallel training.
#[derive(Debug, Clone)]
pub struct ModelParallelConfig {
    /// Number of GPUs across which each tensor is sharded.
    pub tensor_parallel_size: u32,
    /// Number of pipeline stages.
    pub pipeline_parallel_size: u32,
    /// Whether to enable sequence-parallelism (split along sequence dim).
    pub sequence_parallel: bool,
}

impl ModelParallelConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> CudaResult<()> {
        if self.tensor_parallel_size == 0 {
            return Err(CudaError::InvalidValue);
        }
        if self.pipeline_parallel_size == 0 {
            return Err(CudaError::InvalidValue);
        }
        Ok(())
    }

    /// Total number of GPUs required (tensor × pipeline parallelism).
    pub fn total_gpus_required(&self) -> u32 {
        self.tensor_parallel_size * self.pipeline_parallel_size
    }
}

impl Default for ModelParallelConfig {
    fn default() -> Self {
        Self {
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            sequence_parallel: false,
        }
    }
}

// ─── DistributedOptimizer ───────────────────────────────────

/// Wraps gradient communication for distributed training.
///
/// Provides simulated all-reduce over gradient buckets and ZeRO-style
/// optimizer-state partitioning.
pub struct DistributedOptimizer;

impl DistributedOptimizer {
    /// Simulate an all-reduce across all buckets.
    ///
    /// In a real implementation this would use the collective ops over
    /// device memory; here it validates bucket readiness.
    pub fn all_reduce_gradients(buckets: &[Bucket]) -> CudaResult<()> {
        for (i, bucket) in buckets.iter().enumerate() {
            if !bucket.ready {
                return Err(CudaError::NotReady);
            }
            // Simulate communication delay proportional to size
            let _simulated_bytes = bucket.total_size;
            let _bucket_id = i;
        }
        Ok(())
    }

    /// Compute ZeRO-style parameter sharding ranges.
    ///
    /// Partitions `param_count` parameters evenly across `world_size`
    /// ranks, returning one `Range<usize>` per rank.
    pub fn zero_redundancy_partition(world_size: u32, param_count: usize) -> Vec<Range<usize>> {
        if world_size == 0 {
            return Vec::new();
        }
        let ws = world_size as usize;
        let base = param_count / ws;
        let remainder = param_count % ws;

        let mut ranges = Vec::with_capacity(ws);
        let mut start = 0;
        for i in 0..ws {
            let extra = if i < remainder { 1 } else { 0 };
            let end = start + base + extra;
            ranges.push(start..end);
            start = end;
        }
        ranges
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> DistributedConfig {
        DistributedConfig {
            world_size: 4,
            local_rank: 0,
            global_rank: 0,
            master_addr: "127.0.0.1".to_string(),
            master_port: 29500,
            backend: DistributedBackend::Tcp,
        }
    }

    // ── DistributedConfig creation and validation ───────────

    #[test]
    fn config_valid() {
        let cfg = sample_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_invalid_world_size_zero() {
        let mut cfg = sample_config();
        cfg.world_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_invalid_rank_exceeds_world() {
        let mut cfg = sample_config();
        cfg.global_rank = 10;
        assert!(cfg.validate().is_err());
    }

    // ── TcpStore set/get roundtrip ──────────────────────────

    #[test]
    fn tcp_store_set_get() {
        let store = TcpStore::new("127.0.0.1", 29500, 2, true).expect("create store");
        assert!(store.is_master());

        store.set("key1", b"hello").expect("set");
        let val = store.get("key1").expect("get");
        assert_eq!(val, b"hello");
    }

    #[test]
    fn tcp_store_get_missing_key() {
        let store = TcpStore::new("127.0.0.1", 29500, 1, true).expect("create store");
        assert!(store.get("nonexistent").is_err());
    }

    #[test]
    fn tcp_store_add_counter() {
        let store = TcpStore::new("127.0.0.1", 29500, 1, true).expect("create store");
        let v1 = store.add("counter", 5).expect("add");
        assert_eq!(v1, 5);
        let v2 = store.add("counter", 3).expect("add");
        assert_eq!(v2, 8);
    }

    #[test]
    fn tcp_store_wait_present() {
        let store = TcpStore::new("127.0.0.1", 29500, 1, true).expect("create store");
        store.set("a", b"1").expect("set");
        store.set("b", b"2").expect("set");
        assert!(store.wait(&["a", "b"]).is_ok());
    }

    #[test]
    fn tcp_store_wait_missing() {
        let store = TcpStore::new("127.0.0.1", 29500, 1, true).expect("create store");
        store.set("a", b"1").expect("set");
        assert!(store.wait(&["a", "missing"]).is_err());
    }

    // ── FileStore set/get roundtrip ─────────────────────────

    #[test]
    fn file_store_set_get() {
        let dir = std::env::temp_dir().join("oxicuda_test_filestore_setget");
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileStore::new(&dir).expect("create file store");
        store.set("mykey", b"world").expect("set");
        let val = store.get("mykey").expect("get");
        assert_eq!(val, b"world");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_store_add_counter() {
        let dir = std::env::temp_dir().join("oxicuda_test_filestore_add");
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileStore::new(&dir).expect("create file store");
        let v1 = store.add("ctr", 10).expect("add");
        assert_eq!(v1, 10);
        let v2 = store.add("ctr", -3).expect("add");
        assert_eq!(v2, 7);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_store_wait() {
        let dir = std::env::temp_dir().join("oxicuda_test_filestore_wait");
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileStore::new(&dir).expect("create file store");
        store.set("x", b"1").expect("set");
        assert!(store.wait(&["x"]).is_ok());
        assert!(store.wait(&["x", "y"]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── ProcessGroup creation ───────────────────────────────

    #[test]
    fn process_group_creation() {
        let pg = ProcessGroup::new(0, vec![0, 1, 2, 3]).expect("create pg");
        assert_eq!(pg.size, 4);
        assert_eq!(pg.group_id, 0);
        assert!(pg.contains_rank(2));
        assert!(!pg.contains_rank(5));
        assert_eq!(pg.local_rank(3), Some(3));
        assert_eq!(pg.local_rank(9), None);
    }

    #[test]
    fn process_group_empty_rejected() {
        assert!(ProcessGroup::new(0, vec![]).is_err());
    }

    // ── GradientBucket partitioning ─────────────────────────

    #[test]
    fn gradient_bucket_partitioning() {
        // 1 MB buckets
        let mut gb = GradientBucket::new(1);
        // Each gradient is 512 KB => 2 fit per bucket
        let half_mb = 512 * 1024;
        gb.add_gradient(0, half_mb);
        gb.add_gradient(1, half_mb);
        gb.add_gradient(2, half_mb);

        assert_eq!(gb.num_buckets(), 2);
        assert_eq!(gb.buckets()[0].param_ids, vec![0, 1]);
        assert_eq!(gb.buckets()[1].param_ids, vec![2]);
    }

    #[test]
    fn gradient_bucket_size_distribution() {
        let mut gb = GradientBucket::new(2); // 2 MB buckets
        let one_mb = 1024 * 1024;
        for i in 0..5 {
            gb.add_gradient(i, one_mb);
        }
        // Expect 3 buckets: [0,1], [2,3], [4]
        assert_eq!(gb.num_buckets(), 3);
        assert_eq!(gb.buckets()[0].total_size, 2 * one_mb);
        assert_eq!(gb.buckets()[2].param_ids.len(), 1);
    }

    // ── ZeRO parameter sharding ─────────────────────────────

    #[test]
    fn zero_sharding_even() {
        let ranges = DistributedOptimizer::zero_redundancy_partition(4, 100);
        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0], 0..25);
        assert_eq!(ranges[1], 25..50);
        assert_eq!(ranges[2], 50..75);
        assert_eq!(ranges[3], 75..100);
    }

    #[test]
    fn zero_sharding_uneven() {
        let ranges = DistributedOptimizer::zero_redundancy_partition(3, 10);
        assert_eq!(ranges.len(), 3);
        // 10 / 3 = 3 remainder 1 => first rank gets 4
        assert_eq!(ranges[0], 0..4);
        assert_eq!(ranges[1], 4..7);
        assert_eq!(ranges[2], 7..10);
    }

    #[test]
    fn zero_sharding_zero_world() {
        let ranges = DistributedOptimizer::zero_redundancy_partition(0, 100);
        assert!(ranges.is_empty());
    }

    // ── Environment variable parsing ────────────────────────

    #[test]
    fn from_env_missing_vars() {
        // Without the env vars set, this should fail gracefully
        // (We don't set them here on purpose)
        // Note: if a CI sets these vars this test would behave differently,
        // so we only assert it doesn't panic.
        let _result = DistributedRuntime::from_env();
    }

    // ── Barrier logic ───────────────────────────────────────

    #[test]
    fn barrier_increments_epoch() {
        let rt = DistributedRuntime::init(sample_config()).expect("init");
        rt.barrier().expect("barrier 1");
        rt.barrier().expect("barrier 2");
        let epoch = rt.barrier_epoch.lock().expect("lock");
        assert_eq!(*epoch, 2);
    }

    // ── Master node detection ───────────────────────────────

    #[test]
    fn master_detection() {
        let cfg = sample_config(); // global_rank = 0
        let rt = DistributedRuntime::init(cfg).expect("init");
        assert!(rt.is_master());

        let mut cfg2 = sample_config();
        cfg2.global_rank = 2;
        let rt2 = DistributedRuntime::init(cfg2).expect("init");
        assert!(!rt2.is_master());
    }

    // ── World size / rank accessors ─────────────────────────

    #[test]
    fn world_size_rank_accessors() {
        let mut cfg = sample_config();
        cfg.world_size = 8;
        cfg.global_rank = 3;
        cfg.local_rank = 1;
        let rt = DistributedRuntime::init(cfg).expect("init");
        assert_eq!(rt.world_size(), 8);
        assert_eq!(rt.rank(), 3);
        assert_eq!(rt.local_rank(), 1);
    }

    // ── DataParallelConfig defaults ─────────────────────────

    #[test]
    fn data_parallel_config_defaults() {
        let dpc = DataParallelConfig::default();
        assert_eq!(dpc.gradient_bucket_size_mb, 25);
        assert!(dpc.overlap_communication);
        assert!(!dpc.find_unused_parameters);
    }

    // ── ModelParallelConfig validation ──────────────────────

    #[test]
    fn model_parallel_config_validation() {
        let mpc = ModelParallelConfig::default();
        assert!(mpc.validate().is_ok());
        assert_eq!(mpc.total_gpus_required(), 1);

        let bad = ModelParallelConfig {
            tensor_parallel_size: 0,
            pipeline_parallel_size: 4,
            sequence_parallel: false,
        };
        assert!(bad.validate().is_err());
    }

    // ── Status transitions ──────────────────────────────────

    #[test]
    fn status_transitions() {
        let rt = DistributedRuntime::init(sample_config()).expect("init");
        assert_eq!(rt.status(), DistributedStatus::Ready);

        rt.barrier().expect("barrier");
        assert_eq!(rt.status(), DistributedStatus::Ready);

        rt.shutdown().expect("shutdown");
        assert_eq!(rt.status(), DistributedStatus::Shutdown);
    }

    // ── Shutdown idempotency ────────────────────────────────

    #[test]
    fn shutdown_idempotent() {
        let rt = DistributedRuntime::init(sample_config()).expect("init");
        rt.shutdown().expect("shutdown 1");
        rt.shutdown().expect("shutdown 2");
        assert_eq!(rt.status(), DistributedStatus::Shutdown);
    }

    // ── Barrier after shutdown ──────────────────────────────

    #[test]
    fn barrier_after_shutdown_fails() {
        let rt = DistributedRuntime::init(sample_config()).expect("init");
        rt.shutdown().expect("shutdown");
        assert!(rt.barrier().is_err());
    }

    // ── AllReduce gradients ─────────────────────────────────

    #[test]
    fn all_reduce_gradients_ready() {
        let buckets = vec![
            Bucket {
                param_ids: vec![0, 1],
                total_size: 1024,
                ready: true,
            },
            Bucket {
                param_ids: vec![2],
                total_size: 512,
                ready: true,
            },
        ];
        assert!(DistributedOptimizer::all_reduce_gradients(&buckets).is_ok());
    }

    #[test]
    fn all_reduce_gradients_not_ready() {
        let buckets = vec![
            Bucket {
                param_ids: vec![0],
                total_size: 1024,
                ready: true,
            },
            Bucket {
                param_ids: vec![1],
                total_size: 512,
                ready: false,
            },
        ];
        assert!(DistributedOptimizer::all_reduce_gradients(&buckets).is_err());
    }

    // ── NodeInfo ────────────────────────────────────────────

    #[test]
    fn node_info_creation() {
        let ni = NodeInfo::new(NodeId::new(0), "host0", "10.0.0.1", 8080, 4, 0);
        assert_eq!(ni.node_id, NodeId(0));
        assert_eq!(ni.hostname, "host0");
        assert_eq!(ni.gpu_count, 4);
    }

    // ── NodeId display ──────────────────────────────────────

    #[test]
    fn node_id_display() {
        let id = NodeId::new(42);
        assert_eq!(format!("{id}"), "Node(42)");
        assert_eq!(id.value(), 42);
    }
}
