//! NCCL-equivalent collective communication primitives for multi-GPU training.
//!
//! Provides host-side simulations of collective operations (AllReduce, AllGather,
//! ReduceScatter, Broadcast, Reduce, AllToAll) using standard algorithms
//! (ring, tree, recursive halving). The API mirrors NCCL so it can be upgraded
//! to real GPU transfers when running on NVIDIA hardware.

use oxicuda_driver::{CudaError, CudaResult};

// ─── ReduceOp ───────────────────────────────────────────────

/// Reduction operation applied element-wise across ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Summation (identity: 0).
    Sum,
    /// Multiplication (identity: 1).
    Product,
    /// Minimum (identity: +inf).
    Min,
    /// Maximum (identity: -inf).
    Max,
    /// Average (sum / world_size, identity: 0).
    Avg,
}

impl ReduceOp {
    /// Apply the reduction to two `f32` operands.
    pub fn apply_f32(&self, a: f32, b: f32) -> f32 {
        match self {
            Self::Sum | Self::Avg => a + b,
            Self::Product => a * b,
            Self::Min => a.min(b),
            Self::Max => a.max(b),
        }
    }

    /// Apply the reduction to two `f64` operands.
    pub fn apply_f64(&self, a: f64, b: f64) -> f64 {
        match self {
            Self::Sum | Self::Avg => a + b,
            Self::Product => a * b,
            Self::Min => a.min(b),
            Self::Max => a.max(b),
        }
    }

    /// Identity element for `f32` (neutral under the operation).
    pub fn identity_f32(&self) -> f32 {
        match self {
            Self::Sum | Self::Avg => 0.0,
            Self::Product => 1.0,
            Self::Min => f32::INFINITY,
            Self::Max => f32::NEG_INFINITY,
        }
    }

    /// Identity element for `f64`.
    pub fn identity_f64(&self) -> f64 {
        match self {
            Self::Sum | Self::Avg => 0.0,
            Self::Product => 1.0,
            Self::Min => f64::INFINITY,
            Self::Max => f64::NEG_INFINITY,
        }
    }
}

// ─── DataType ───────────────────────────────────────────────

/// Supported data types for collective operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// 16-bit IEEE 754 half-precision float.
    F16,
    /// 16-bit brain float.
    BF16,
}

impl DataType {
    /// Size of a single element in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }
}

// ─── Communicator ───────────────────────────────────────────

/// A communicator spanning a set of GPU devices.
///
/// Mirrors `ncclComm_t` — each communicator tracks which devices
/// participate and assigns sequential ranks.
#[derive(Debug, Clone)]
pub struct Communicator {
    /// Ordered device ordinals.
    devices: Vec<i32>,
    /// This process's rank (index into `devices`).
    my_rank: usize,
}

impl Communicator {
    /// Create a communicator over the given device ordinals.
    ///
    /// `device_ordinals` must be non-empty. The first device is rank 0.
    pub fn new(device_ordinals: &[i32]) -> CudaResult<Self> {
        if device_ordinals.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        Ok(Self {
            devices: device_ordinals.to_vec(),
            my_rank: 0,
        })
    }

    /// Create a communicator with a specific rank selected.
    pub fn with_rank(device_ordinals: &[i32], rank: usize) -> CudaResult<Self> {
        if device_ordinals.is_empty() || rank >= device_ordinals.len() {
            return Err(CudaError::InvalidValue);
        }
        Ok(Self {
            devices: device_ordinals.to_vec(),
            my_rank: rank,
        })
    }

    /// This device's rank in the communicator.
    pub fn rank(&self) -> usize {
        self.my_rank
    }

    /// Total number of devices in the communicator.
    pub fn world_size(&self) -> usize {
        self.devices.len()
    }

    /// Device ordinal for the given rank, if valid.
    pub fn device_ordinal(&self, rank: usize) -> Option<i32> {
        self.devices.get(rank).copied()
    }
}

// ─── CollectiveConfig ───────────────────────────────────────

/// Configuration for a collective operation.
#[derive(Debug, Clone)]
pub struct CollectiveConfig {
    /// Stream index to use (`None` = default stream).
    pub stream: Option<usize>,
    /// Whether the operation is non-blocking.
    pub async_op: bool,
    /// Chunk size for pipelining large transfers (bytes).
    pub chunk_size: Option<usize>,
}

#[allow(clippy::derivable_impls)]
impl Default for CollectiveConfig {
    fn default() -> Self {
        Self {
            stream: None,
            async_op: false,
            chunk_size: None,
        }
    }
}

// ─── AllReduceAlgorithm ─────────────────────────────────────

/// Algorithm selection for AllReduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllReduceAlgorithm {
    /// Ring-based — bandwidth-optimal for large messages.
    Ring,
    /// Tree-based — latency-optimal for small messages.
    Tree,
    /// Recursive halving-doubling — for power-of-2 rank counts.
    RecursiveHalving,
    /// Automatic selection based on message size and rank count.
    Auto,
}

impl AllReduceAlgorithm {
    /// Heuristic: pick the best algorithm for the given parameters.
    fn select(world_size: usize, msg_len: usize) -> Self {
        if msg_len < 256 {
            Self::Tree
        } else if world_size.is_power_of_two() && msg_len < 4096 {
            Self::RecursiveHalving
        } else {
            Self::Ring
        }
    }
}

// ─── Ring AllReduce ─────────────────────────────────────────

/// Ring-based all-reduce implementation.
///
/// Two phases: scatter-reduce (each rank reduces one chunk from its
/// left neighbour) then all-gather (each rank copies the final chunk
/// from its left neighbour).
pub struct RingAllReduce;

impl RingAllReduce {
    /// Execute ring all-reduce on host-side buffers (one per rank).
    pub fn execute(buffers: &mut [Vec<f32>], op: ReduceOp) -> CudaResult<()> {
        let n_ranks = buffers.len();
        if n_ranks < 2 {
            return Ok(());
        }
        let buf_len = buffers[0].len();
        for b in buffers.iter() {
            if b.len() != buf_len {
                return Err(CudaError::InvalidValue);
            }
        }
        if buf_len == 0 {
            return Ok(());
        }

        let chunk_size = buf_len.div_ceil(n_ranks);

        // Phase 1: scatter-reduce — n_ranks-1 steps
        for step in 0..n_ranks - 1 {
            // Collect sends first to avoid aliasing
            let sends: Vec<(usize, usize, Vec<f32>)> = (0..n_ranks)
                .map(|rank| {
                    let send_chunk = (rank + n_ranks - 1 - step) % n_ranks;
                    let start = send_chunk * chunk_size;
                    let end = (start + chunk_size).min(buf_len);
                    let data = buffers[rank][start..end].to_vec();
                    (rank, send_chunk, data)
                })
                .collect();

            for (rank, send_chunk, data) in sends {
                let recv_rank = (rank + 1) % n_ranks;
                let start = send_chunk * chunk_size;
                let end = (start + chunk_size).min(buf_len);
                for (i, idx) in (start..end).enumerate() {
                    buffers[recv_rank][idx] = op.apply_f32(buffers[recv_rank][idx], data[i]);
                }
            }
        }

        // Phase 2: all-gather — n_ranks-1 steps
        for step in 0..n_ranks - 1 {
            let sends: Vec<(usize, Vec<f32>)> = (0..n_ranks)
                .map(|rank| {
                    let send_chunk = (rank + n_ranks - step) % n_ranks;
                    let start = send_chunk * chunk_size;
                    let end = (start + chunk_size).min(buf_len);
                    let data = buffers[rank][start..end].to_vec();
                    (send_chunk, data)
                })
                .collect();

            for (rank, (send_chunk, data)) in sends.into_iter().enumerate() {
                let recv_rank = (rank + 1) % n_ranks;
                let start = send_chunk * chunk_size;
                let end = (start + chunk_size).min(buf_len);
                buffers[recv_rank][start..end].copy_from_slice(&data[..end - start]);
            }
        }

        // For Avg: divide by world_size after summation
        if op == ReduceOp::Avg {
            let divisor = n_ranks as f32;
            for buf in buffers.iter_mut() {
                for v in buf.iter_mut() {
                    *v /= divisor;
                }
            }
        }

        Ok(())
    }
}

// ─── Tree AllReduce ─────────────────────────────────────────

/// Binary-tree all-reduce implementation.
///
/// Phase 1: reduce up the tree to root (rank 0).
/// Phase 2: broadcast the result back down.
pub struct TreeAllReduce;

impl TreeAllReduce {
    /// Execute tree all-reduce on host-side buffers.
    pub fn execute(buffers: &mut [Vec<f32>], op: ReduceOp) -> CudaResult<()> {
        let n_ranks = buffers.len();
        if n_ranks < 2 {
            return Ok(());
        }
        let buf_len = buffers[0].len();
        for b in buffers.iter() {
            if b.len() != buf_len {
                return Err(CudaError::InvalidValue);
            }
        }

        // Phase 1: reduce to root (rank 0)
        let mut stride = 1;
        while stride < n_ranks {
            let mut rank = 0;
            while rank + stride < n_ranks {
                let child = rank + stride;
                let child_data = buffers[child].clone();
                for (i, v) in child_data.into_iter().enumerate() {
                    buffers[rank][i] = op.apply_f32(buffers[rank][i], v);
                }
                rank += stride * 2;
            }
            stride *= 2;
        }

        // For Avg: divide root result by world_size
        if op == ReduceOp::Avg {
            let divisor = n_ranks as f32;
            for v in buffers[0].iter_mut() {
                *v /= divisor;
            }
        }

        // Phase 2: broadcast from root
        let root_data = buffers[0].clone();
        for buf in buffers.iter_mut().skip(1) {
            buf.copy_from_slice(&root_data);
        }

        Ok(())
    }
}

// ─── CollectiveOps ──────────────────────────────────────────

/// Host-side simulation of NCCL-style collective operations.
///
/// All operations work on `f32` slices. The `comm` parameter
/// determines world size; the actual multi-buffer work is done by
/// constructing per-rank buffers internally when needed.
pub struct CollectiveOps;

impl CollectiveOps {
    /// AllReduce: every rank ends up with the reduced result.
    ///
    /// `sendbuf` is treated as data from rank 0; the result is
    /// replicated across all simulated ranks (returned in `recvbuf`).
    pub fn all_reduce(
        sendbuf: &[f32],
        recvbuf: &mut [f32],
        op: ReduceOp,
        comm: &Communicator,
        algo: AllReduceAlgorithm,
    ) -> CudaResult<()> {
        if sendbuf.len() != recvbuf.len() {
            return Err(CudaError::InvalidValue);
        }

        let n = comm.world_size();
        let mut buffers: Vec<Vec<f32>> = (0..n).map(|_| sendbuf.to_vec()).collect();

        let resolved = match algo {
            AllReduceAlgorithm::Auto => AllReduceAlgorithm::select(n, sendbuf.len()),
            other => other,
        };

        match resolved {
            AllReduceAlgorithm::Ring => RingAllReduce::execute(&mut buffers, op)?,
            AllReduceAlgorithm::Tree => TreeAllReduce::execute(&mut buffers, op)?,
            AllReduceAlgorithm::RecursiveHalving => {
                Self::recursive_halving(&mut buffers, op)?;
            }
            AllReduceAlgorithm::Auto => unreachable!(),
        }

        recvbuf.copy_from_slice(&buffers[0]);
        Ok(())
    }

    /// Recursive halving-doubling all-reduce for power-of-2 rank counts.
    fn recursive_halving(buffers: &mut [Vec<f32>], op: ReduceOp) -> CudaResult<()> {
        let n = buffers.len();
        if n < 2 {
            return Ok(());
        }
        // Fall back to ring for non-power-of-2
        if !n.is_power_of_two() {
            return RingAllReduce::execute(buffers, op);
        }
        let buf_len = buffers[0].len();

        let mut distance = 1;
        while distance < n {
            let pairs: Vec<(usize, Vec<f32>)> = (0..n)
                .map(|rank| {
                    let partner = rank ^ distance;
                    (partner, buffers[rank].clone())
                })
                .collect();

            for (rank, (partner, data)) in pairs.into_iter().enumerate() {
                if partner < n {
                    for i in 0..buf_len {
                        buffers[rank][i] = op.apply_f32(buffers[rank][i], data[i]);
                    }
                }
            }
            distance *= 2;
        }

        // For Avg: divide by world_size
        if op == ReduceOp::Avg {
            let divisor = n as f32;
            for buf in buffers.iter_mut() {
                for v in buf.iter_mut() {
                    *v /= divisor;
                }
            }
        }

        Ok(())
    }

    /// AllGather: concatenate each rank's send buffer into every rank's recv buffer.
    pub fn all_gather(sendbuf: &[f32], recvbuf: &mut [f32], comm: &Communicator) -> CudaResult<()> {
        let n = comm.world_size();
        let send_len = sendbuf.len();
        if recvbuf.len() != send_len * n {
            return Err(CudaError::InvalidValue);
        }
        // Each rank contributes the same sendbuf (host simulation)
        for rank in 0..n {
            let start = rank * send_len;
            recvbuf[start..start + send_len].copy_from_slice(sendbuf);
        }
        Ok(())
    }

    /// ReduceScatter: reduce across ranks then scatter chunks.
    ///
    /// `recvbuf` gets `sendbuf.len() / world_size` elements — the
    /// chunk assigned to this communicator's rank.
    pub fn reduce_scatter(
        sendbuf: &[f32],
        recvbuf: &mut [f32],
        op: ReduceOp,
        comm: &Communicator,
    ) -> CudaResult<()> {
        let n = comm.world_size();
        let total = sendbuf.len();
        let chunk = total / n;
        if chunk == 0 || recvbuf.len() != chunk {
            return Err(CudaError::InvalidValue);
        }

        // Simulate: all ranks have same data, reduce then take this rank's chunk
        let my_rank = comm.rank();
        let start = my_rank * chunk;
        let end = start + chunk;

        // With identical data: reduce n copies element-wise
        for (i, idx) in (start..end).enumerate() {
            let mut acc = sendbuf[idx];
            for _ in 1..n {
                acc = op.apply_f32(acc, sendbuf[idx]);
            }
            if op == ReduceOp::Avg {
                acc /= n as f32;
            }
            recvbuf[i] = acc;
        }

        Ok(())
    }

    /// Broadcast: root rank's data is sent to all ranks.
    pub fn broadcast(_buf: &mut [f32], root: usize, comm: &Communicator) -> CudaResult<()> {
        if root >= comm.world_size() {
            return Err(CudaError::InvalidValue);
        }
        // In host simulation, buf already contains root's data — no-op
        // (would copy from root's device buffer to all others on real hardware)
        Ok(())
    }

    /// Reduce: all ranks send to root, which holds the reduced result.
    pub fn reduce(
        sendbuf: &[f32],
        recvbuf: &mut [f32],
        op: ReduceOp,
        root: usize,
        comm: &Communicator,
    ) -> CudaResult<()> {
        if root >= comm.world_size() {
            return Err(CudaError::InvalidValue);
        }
        if sendbuf.len() != recvbuf.len() {
            return Err(CudaError::InvalidValue);
        }

        let n = comm.world_size();
        // Simulate: n copies of sendbuf reduced element-wise
        for (i, &v) in sendbuf.iter().enumerate() {
            let mut acc = v;
            for _ in 1..n {
                acc = op.apply_f32(acc, v);
            }
            if op == ReduceOp::Avg {
                acc /= n as f32;
            }
            recvbuf[i] = acc;
        }
        Ok(())
    }

    /// AllToAll: personalized exchange — each rank sends distinct chunks.
    ///
    /// `sendbuf` has `world_size` equal chunks; chunk `j` goes to rank `j`.
    /// `recvbuf` collects chunk `rank` from every other rank.
    /// In host simulation with identical data, recv chunk `i` = send chunk `rank`.
    pub fn all_to_all(sendbuf: &[f32], recvbuf: &mut [f32], comm: &Communicator) -> CudaResult<()> {
        let n = comm.world_size();
        let total = sendbuf.len();
        if total != recvbuf.len() {
            return Err(CudaError::InvalidValue);
        }
        let chunk = total / n;
        if chunk == 0 {
            return Err(CudaError::InvalidValue);
        }

        // Host simulation: all ranks have same sendbuf
        // Rank r receives chunk[r] from each sender
        // Since all senders are identical, recvbuf gets chunk[my_rank] repeated n times
        let my_rank = comm.rank();
        let src_start = my_rank * chunk;
        for r in 0..n {
            let dst_start = r * chunk;
            recvbuf[dst_start..dst_start + chunk]
                .copy_from_slice(&sendbuf[src_start..src_start + chunk]);
        }
        Ok(())
    }
}

// ─── CommGroup ──────────────────────────────────────────────

/// Named communicator group utilities.
pub struct CommGroup;

impl CommGroup {
    /// Create a communicator spanning all available GPUs (simulated).
    ///
    /// In host simulation this defaults to 4 virtual GPUs.
    pub fn world() -> Communicator {
        Communicator {
            devices: vec![0, 1, 2, 3],
            my_rank: 0,
        }
    }

    /// Split a communicator by colour — ranks with the same colour are
    /// grouped together, preserving their relative order.
    pub fn split(comm: &Communicator, color: usize, rank: usize) -> CudaResult<Communicator> {
        // In a real implementation, each rank calls split with its own colour.
        // Here we just return a sub-communicator for the given rank.
        if rank >= comm.world_size() {
            return Err(CudaError::InvalidValue);
        }

        // Build a sub-group: take every `color+1`-th device starting from the
        // matching offset, clamped to valid indices.
        let step = color.max(1);
        let sub: Vec<i32> = comm
            .devices
            .iter()
            .copied()
            .skip(rank % step)
            .step_by(step)
            .collect();

        if sub.is_empty() {
            return Err(CudaError::InvalidValue);
        }

        Ok(Communicator {
            devices: sub,
            my_rank: 0,
        })
    }

    /// Duplicate a communicator (independent copy with same membership).
    pub fn dup(comm: &Communicator) -> Communicator {
        Communicator {
            devices: comm.devices.clone(),
            my_rank: comm.my_rank,
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ReduceOp ────────────────────────────────────────────

    #[test]
    fn reduce_op_sum() {
        assert!((ReduceOp::Sum.apply_f32(3.0, 4.0) - 7.0).abs() < f32::EPSILON);
        assert!((ReduceOp::Sum.identity_f32()).abs() < f32::EPSILON);
    }

    #[test]
    fn reduce_op_product() {
        assert!((ReduceOp::Product.apply_f32(3.0, 4.0) - 12.0).abs() < f32::EPSILON);
        assert!((ReduceOp::Product.identity_f32() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn reduce_op_min_max() {
        assert!((ReduceOp::Min.apply_f32(3.0, 4.0) - 3.0).abs() < f32::EPSILON);
        assert!((ReduceOp::Max.apply_f32(3.0, 4.0) - 4.0).abs() < f32::EPSILON);
        assert!(ReduceOp::Min.identity_f32().is_infinite());
        assert!(ReduceOp::Max.identity_f32().is_infinite());
    }

    #[test]
    fn reduce_op_avg() {
        // Avg accumulates via sum, then divides — apply is same as sum
        assert!((ReduceOp::Avg.apply_f64(3.0, 4.0) - 7.0).abs() < f64::EPSILON);
        assert!((ReduceOp::Avg.identity_f64()).abs() < f64::EPSILON);
    }

    #[test]
    fn reduce_op_f64() {
        assert!((ReduceOp::Product.apply_f64(2.5, 4.0) - 10.0).abs() < f64::EPSILON);
        assert!((ReduceOp::Min.apply_f64(1.0, 2.0) - 1.0).abs() < f64::EPSILON);
    }

    // ── DataType ────────────────────────────────────────────

    #[test]
    fn data_type_sizes() {
        assert_eq!(DataType::F16.size_bytes(), 2);
        assert_eq!(DataType::BF16.size_bytes(), 2);
        assert_eq!(DataType::F32.size_bytes(), 4);
        assert_eq!(DataType::I32.size_bytes(), 4);
        assert_eq!(DataType::U32.size_bytes(), 4);
        assert_eq!(DataType::F64.size_bytes(), 8);
        assert_eq!(DataType::I64.size_bytes(), 8);
        assert_eq!(DataType::U64.size_bytes(), 8);
    }

    // ── Communicator ────────────────────────────────────────

    #[test]
    fn communicator_basics() {
        let comm = Communicator::new(&[0, 1, 2]).expect("create comm");
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.world_size(), 3);
        assert_eq!(comm.device_ordinal(1), Some(1));
        assert_eq!(comm.device_ordinal(5), None);
    }

    #[test]
    fn communicator_empty_rejected() {
        assert!(Communicator::new(&[]).is_err());
    }

    #[test]
    fn communicator_with_rank() {
        let comm = Communicator::with_rank(&[10, 20, 30], 2).expect("rank 2");
        assert_eq!(comm.rank(), 2);
        assert_eq!(comm.device_ordinal(0), Some(10));
    }

    // ── Ring AllReduce ──────────────────────────────────────

    #[test]
    fn ring_all_reduce_2_ranks_sum() {
        let mut bufs = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        RingAllReduce::execute(&mut bufs, ReduceOp::Sum).expect("ring 2");
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        for (a, b) in bufs[0].iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5, "got {a}, expected {b}");
        }
        assert_eq!(bufs[0], bufs[1]);
    }

    #[test]
    fn ring_all_reduce_4_ranks_sum() {
        let mut bufs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 3.0, 0.0],
            vec![0.0, 0.0, 0.0, 4.0],
        ];
        RingAllReduce::execute(&mut bufs, ReduceOp::Sum).expect("ring 4");
        let expected = vec![1.0, 2.0, 3.0, 4.0];
        for buf in &bufs {
            for (a, b) in buf.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    // ── Tree AllReduce ──────────────────────────────────────

    #[test]
    fn tree_all_reduce_sum() {
        let mut bufs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        TreeAllReduce::execute(&mut bufs, ReduceOp::Sum).expect("tree");
        let expected = vec![9.0, 12.0];
        for buf in &bufs {
            for (a, b) in buf.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn tree_all_reduce_product() {
        let mut bufs = vec![vec![2.0, 3.0], vec![4.0, 5.0]];
        TreeAllReduce::execute(&mut bufs, ReduceOp::Product).expect("tree prod");
        assert!((bufs[0][0] - 8.0).abs() < 1e-5);
        assert!((bufs[0][1] - 15.0).abs() < 1e-5);
        assert_eq!(bufs[0], bufs[1]);
    }

    // ── CollectiveOps ───────────────────────────────────────

    #[test]
    fn all_gather_correctness() {
        let comm = Communicator::new(&[0, 1, 2]).expect("comm");
        let send = [10.0, 20.0];
        let mut recv = vec![0.0f32; 6];
        CollectiveOps::all_gather(&send, &mut recv, &comm).expect("all_gather");
        assert_eq!(recv, vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0]);
    }

    #[test]
    fn reduce_scatter_correctness() {
        let comm = Communicator::new(&[0, 1]).expect("comm");
        let send = [1.0, 2.0, 3.0, 4.0];
        let mut recv = vec![0.0f32; 2];
        CollectiveOps::reduce_scatter(&send, &mut recv, ReduceOp::Sum, &comm)
            .expect("reduce_scatter");
        // 2 ranks, same data => each element summed 2×, rank 0 gets first half
        assert!((recv[0] - 2.0).abs() < 1e-5);
        assert!((recv[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn broadcast_from_root() {
        let comm = Communicator::new(&[0, 1, 2]).expect("comm");
        let mut buf = [42.0f32, 99.0];
        CollectiveOps::broadcast(&mut buf, 0, &comm).expect("broadcast");
        assert_eq!(buf, [42.0, 99.0]);
    }

    #[test]
    fn broadcast_invalid_root() {
        let comm = Communicator::new(&[0]).expect("comm");
        let mut buf = [1.0f32];
        assert!(CollectiveOps::broadcast(&mut buf, 5, &comm).is_err());
    }

    #[test]
    fn reduce_to_root() {
        let comm = Communicator::new(&[0, 1, 2, 3]).expect("comm");
        let send = [1.0f32, 2.0];
        let mut recv = vec![0.0f32; 2];
        CollectiveOps::reduce(&send, &mut recv, ReduceOp::Sum, 0, &comm).expect("reduce");
        // 4 ranks of identical data summed
        assert!((recv[0] - 4.0).abs() < 1e-5);
        assert!((recv[1] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn all_to_all_exchange() {
        let comm = Communicator::new(&[0, 1]).expect("comm");
        let send = [1.0f32, 2.0, 3.0, 4.0];
        let mut recv = vec![0.0f32; 4];
        CollectiveOps::all_to_all(&send, &mut recv, &comm).expect("a2a");
        // rank 0 receives chunk[0] from all senders
        assert_eq!(recv, vec![1.0, 2.0, 1.0, 2.0]);
    }

    // ── CommGroup ───────────────────────────────────────────

    #[test]
    fn comm_group_world() {
        let w = CommGroup::world();
        assert_eq!(w.world_size(), 4);
        assert_eq!(w.rank(), 0);
    }

    #[test]
    fn comm_group_dup() {
        let comm = Communicator::with_rank(&[0, 1, 2], 1).expect("comm");
        let dup = CommGroup::dup(&comm);
        assert_eq!(dup.world_size(), comm.world_size());
        assert_eq!(dup.rank(), comm.rank());
    }

    #[test]
    fn comm_group_split() {
        let comm = Communicator::new(&[0, 1, 2, 3]).expect("comm");
        let sub = CommGroup::split(&comm, 2, 0).expect("split");
        // step=2, skip 0 => devices [0, 2]
        assert_eq!(sub.world_size(), 2);
    }

    // ── AllReduce via CollectiveOps (integration) ───────────

    #[test]
    fn all_reduce_auto_algorithm() {
        let comm = Communicator::new(&[0, 1]).expect("comm");
        let send = vec![1.0f32; 8];
        let mut recv = vec![0.0f32; 8];
        CollectiveOps::all_reduce(
            &send,
            &mut recv,
            ReduceOp::Sum,
            &comm,
            AllReduceAlgorithm::Auto,
        )
        .expect("auto");
        for v in &recv {
            assert!((*v - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn collective_config_defaults() {
        let cfg = CollectiveConfig::default();
        assert!(cfg.stream.is_none());
        assert!(!cfg.async_op);
        assert!(cfg.chunk_size.is_none());
    }
}
