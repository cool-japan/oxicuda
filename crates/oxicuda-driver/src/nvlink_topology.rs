//! NVLink/NVSwitch topology-aware communication.
//!
//! This module provides topology discovery and routing primitives for
//! multi-GPU systems connected via NVLink, NVSwitch, or PCIe.  It enables
//! topology-aware collective communication scheduling (ring AllReduce,
//! tree broadcast/reduce) and task placement that minimises inter-GPU
//! communication cost.
//!
//! On macOS, where NVIDIA GPUs are not available, this module returns
//! synthetic topology data for a 4-GPU NVLink mesh so that algorithm
//! logic can be tested without hardware.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxicuda_driver::nvlink_topology::GpuTopology;
//!
//! let topo = GpuTopology::discover()?;
//! println!("topology type: {:?}", topo.topology_type());
//! let ring = topo.optimal_ring_order()?;
//! println!("ring order: {ring:?}");
//! # Ok::<(), oxicuda_driver::error::CudaError>(())
//! ```

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::error::{CudaError, CudaResult};

// ---------------------------------------------------------------------------
// NVLink version & status enums
// ---------------------------------------------------------------------------

/// NVLink generation/version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NvLinkVersion {
    /// NVLink 1.0 — Pascal architecture (P100), 20 GB/s per link.
    V1,
    /// NVLink 2.0 — Volta architecture (V100), 25 GB/s per link.
    V2,
    /// NVLink 3.0 — Ampere architecture (A100), 25 GB/s per link (wider).
    V3,
    /// NVLink 4.0 — Hopper architecture (H100), 25 GB/s per sub-link.
    V4,
    /// NVSwitch-mediated all-to-all fabric.
    NvSwitch,
}

impl NvLinkVersion {
    /// Per-link bandwidth in GB/s for this NVLink generation.
    #[inline]
    pub fn per_link_bandwidth_gbps(self) -> f64 {
        match self {
            Self::V1 => 20.0,
            Self::V2 => 25.0,
            Self::V3 => 25.0,
            Self::V4 => 25.0,
            Self::NvSwitch => 25.0,
        }
    }
}

/// Status of an NVLink connection between two devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NvLinkStatus {
    /// Link is active and usable.
    Active,
    /// Link exists but is currently inactive.
    Inactive,
    /// NVLink not supported between these devices.
    Unsupported,
}

// ---------------------------------------------------------------------------
// NVLink info per peer
// ---------------------------------------------------------------------------

/// Information about the NVLink connections to a specific peer device.
#[derive(Debug, Clone)]
pub struct NvLinkInfo {
    /// NVLink generation.
    pub version: NvLinkVersion,
    /// Aggregate bidirectional bandwidth in GB/s across all active links.
    pub bandwidth_gbps: f64,
    /// Number of active NVLink connections.
    pub link_count: u32,
    /// Ordinal of the peer device.
    pub peer_device_id: i32,
}

// ---------------------------------------------------------------------------
// Link type between devices
// ---------------------------------------------------------------------------

/// The physical interconnect type between two devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinkType {
    /// Direct NVLink connection.
    NvLink,
    /// PCIe connection (possibly through a switch).
    PCIe,
    /// NVSwitch-mediated connection (full bisection bandwidth).
    NvSwitch,
}

// ---------------------------------------------------------------------------
// Topology link
// ---------------------------------------------------------------------------

/// A directed link between two GPU devices in the topology graph.
#[derive(Debug, Clone)]
pub struct TopologyLink {
    /// Source device ordinal.
    pub from_device: i32,
    /// Destination device ordinal.
    pub to_device: i32,
    /// Physical interconnect type.
    pub link_type: LinkType,
    /// Aggregate bandwidth in GB/s.
    pub bandwidth_gbps: f64,
    /// Estimated one-way latency in nanoseconds.
    pub latency_ns: f64,
    /// Number of hops (1 for direct NVLink, 1 for NVSwitch, 2+ for PCIe).
    pub hop_count: u32,
}

// ---------------------------------------------------------------------------
// Topology classification
// ---------------------------------------------------------------------------

/// High-level classification of the detected GPU topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TopologyType {
    /// Only one GPU detected.
    SingleGpu,
    /// Two GPUs connected by NVLink.
    NvLinkPair,
    /// GPUs form a ring via NVLink.
    NvLinkRing,
    /// All GPUs connected to all others via NVLink (full mesh).
    NvLinkMesh,
    /// NVSwitch fabric connecting all GPUs.
    NvSwitchFabric,
    /// GPUs connected only via PCIe.
    PcieOnly,
}

// ---------------------------------------------------------------------------
// Topology tree (for broadcast / reduce)
// ---------------------------------------------------------------------------

/// A tree structure rooted at a single device, used for broadcast/reduce
/// collectives.
#[derive(Debug, Clone)]
pub struct TopologyTree {
    /// Root device ordinal.
    pub root: i32,
    /// Maps each device to its list of child devices.
    pub children: HashMap<i32, Vec<i32>>,
    /// Maps each device to its parent (root has no entry).
    pub parent: HashMap<i32, i32>,
}

impl TopologyTree {
    /// Returns all devices in the tree.
    pub fn devices(&self) -> Vec<i32> {
        let mut devs: Vec<i32> = self.children.keys().copied().collect();
        devs.sort_unstable();
        devs
    }

    /// Returns the depth of the tree.
    pub fn depth(&self) -> usize {
        fn walk_depth(node: i32, children: &HashMap<i32, Vec<i32>>) -> usize {
            match children.get(&node) {
                Some(kids) if !kids.is_empty() => {
                    1 + kids
                        .iter()
                        .map(|&c| walk_depth(c, children))
                        .max()
                        .unwrap_or(0)
                }
                _ => 1,
            }
        }
        walk_depth(self.root, &self.children)
    }
}

// ---------------------------------------------------------------------------
// Communication schedule
// ---------------------------------------------------------------------------

/// A single data transfer in a communication schedule.
#[derive(Debug, Clone)]
pub struct Transfer {
    /// Source device ordinal.
    pub src: i32,
    /// Destination device ordinal.
    pub dst: i32,
    /// Data size in bytes.
    pub data_size: u64,
}

/// An ordered list of transfers optimised for the topology.
#[derive(Debug, Clone)]
pub struct CommunicationSchedule {
    /// Ordered transfers.
    pub transfers: Vec<Transfer>,
    /// Estimated total time in microseconds.
    pub estimated_time_us: f64,
}

// ---------------------------------------------------------------------------
// Task placement
// ---------------------------------------------------------------------------

/// A communication demand between two logical tasks.
#[derive(Debug, Clone)]
pub struct TaskCommunication {
    /// First task index.
    pub task_a: usize,
    /// Second task index.
    pub task_b: usize,
    /// Communication volume in bytes.
    pub volume_bytes: u64,
}

/// Result of topology-aware task placement.
#[derive(Debug, Clone)]
pub struct TopologyAwarePlacement {
    /// Maps task index to device ordinal.
    pub assignment: HashMap<usize, i32>,
    /// Estimated total communication cost (lower is better).
    pub total_cost: f64,
}

// ---------------------------------------------------------------------------
// GpuTopology
// ---------------------------------------------------------------------------

/// Complete GPU topology graph with adjacency information.
///
/// Holds all devices, inter-device links, and a dense adjacency matrix for
/// fast bandwidth/latency lookups.
#[derive(Debug, Clone)]
pub struct GpuTopology {
    /// Device ordinals present in the topology.
    pub devices: Vec<i32>,
    /// All directed links between devices.
    pub links: Vec<TopologyLink>,
    /// Dense adjacency matrix: `adj[i][j]` = bandwidth in GB/s from
    /// device `devices[i]` to device `devices[j]`.  Diagonal is `f64::INFINITY`.
    adj_bandwidth: Vec<Vec<f64>>,
    /// Dense latency matrix in nanoseconds.
    adj_latency: Vec<Vec<f64>>,
}

impl GpuTopology {
    // -- Discovery ----------------------------------------------------------

    /// Discover the GPU topology of the current system.
    ///
    /// On Linux/Windows with NVIDIA drivers, this queries the driver for
    /// device count and peer-access capabilities.  On macOS, a synthetic
    /// 4-GPU NVLink mesh is returned for testing purposes.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NoDevice`] if no devices are found.
    pub fn discover() -> CudaResult<Self> {
        #[cfg(target_os = "macos")]
        {
            Self::synthetic_mesh(4)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self::discover_real()
        }
    }

    /// Build a synthetic N-GPU NVLink mesh (used on macOS and in tests).
    #[cfg(any(target_os = "macos", test))]
    fn synthetic_mesh(n: usize) -> CudaResult<Self> {
        if n == 0 {
            return Err(CudaError::NoDevice);
        }
        let devices: Vec<i32> = (0..n as i32).collect();
        let mut links = Vec::new();
        let mut adj_bandwidth = vec![vec![0.0; n]; n];
        let mut adj_latency = vec![vec![f64::MAX; n]; n];

        for i in 0..n {
            adj_bandwidth[i][i] = f64::INFINITY;
            adj_latency[i][i] = 0.0;
        }

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let bw = 100.0; // Synthetic 100 GB/s NVLink
                let lat = 500.0; // 500 ns latency
                links.push(TopologyLink {
                    from_device: i as i32,
                    to_device: j as i32,
                    link_type: LinkType::NvLink,
                    bandwidth_gbps: bw,
                    latency_ns: lat,
                    hop_count: 1,
                });
                adj_bandwidth[i][j] = bw;
                adj_latency[i][j] = lat;
            }
        }

        Ok(Self {
            devices,
            links,
            adj_bandwidth,
            adj_latency,
        })
    }

    /// Real topology discovery using the CUDA driver.
    #[cfg(not(target_os = "macos"))]
    fn discover_real() -> CudaResult<Self> {
        use crate::device::Device;

        crate::init()?;
        let count = Device::count()?;
        if count <= 0 {
            return Err(CudaError::NoDevice);
        }
        let n = count as usize;
        let devices: Vec<i32> = (0..count).collect();
        let mut links = Vec::new();
        let mut adj_bandwidth = vec![vec![0.0; n]; n];
        let mut adj_latency = vec![vec![f64::MAX; n]; n];

        for i in 0..n {
            adj_bandwidth[i][i] = f64::INFINITY;
            adj_latency[i][i] = 0.0;
        }

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                // Check peer access to determine link type
                let can_peer = crate::device::can_access_peer(
                    &Device::get(i as i32)?,
                    &Device::get(j as i32)?,
                )?;

                let (lt, bw, lat, hops) = if can_peer {
                    // Peer access available — likely NVLink
                    (LinkType::NvLink, 50.0, 800.0, 1)
                } else {
                    // Fallback to PCIe
                    (LinkType::PCIe, 16.0, 2000.0, 2)
                };

                links.push(TopologyLink {
                    from_device: i as i32,
                    to_device: j as i32,
                    link_type: lt,
                    bandwidth_gbps: bw,
                    latency_ns: lat,
                    hop_count: hops,
                });
                adj_bandwidth[i][j] = bw;
                adj_latency[i][j] = lat;
            }
        }

        Ok(Self {
            devices,
            links,
            adj_bandwidth,
            adj_latency,
        })
    }

    // -- Topology classification --------------------------------------------

    /// Classify the topology into a high-level category.
    pub fn topology_type(&self) -> TopologyType {
        let n = self.devices.len();
        if n == 0 {
            return TopologyType::SingleGpu;
        }
        if n == 1 {
            return TopologyType::SingleGpu;
        }

        let has_nvswitch = self.links.iter().any(|l| l.link_type == LinkType::NvSwitch);
        if has_nvswitch {
            return TopologyType::NvSwitchFabric;
        }

        let nvlink_links: Vec<&TopologyLink> = self
            .links
            .iter()
            .filter(|l| l.link_type == LinkType::NvLink)
            .collect();

        if nvlink_links.is_empty() {
            return TopologyType::PcieOnly;
        }

        // Count NVLink neighbors per device
        let mut nvlink_neighbors: HashMap<i32, Vec<i32>> = HashMap::new();
        for link in &nvlink_links {
            nvlink_neighbors
                .entry(link.from_device)
                .or_default()
                .push(link.to_device);
        }

        // Check if all devices have NVLink neighbors
        let all_have_nvlink = self
            .devices
            .iter()
            .all(|d| nvlink_neighbors.contains_key(d));

        if !all_have_nvlink {
            // Some devices only on PCIe
            if n == 2 {
                return TopologyType::NvLinkPair;
            }
            return TopologyType::PcieOnly;
        }

        // Check for full mesh: every device connected to every other
        let is_full_mesh = self.devices.iter().all(|d| {
            let neighbors = nvlink_neighbors.get(d).map(|v| v.len()).unwrap_or(0);
            neighbors == n - 1
        });

        if is_full_mesh {
            return TopologyType::NvLinkMesh;
        }

        if n == 2 {
            return TopologyType::NvLinkPair;
        }

        // Check for ring: every device has exactly 2 NVLink neighbors
        // and they form a single cycle
        let is_ring = self
            .devices
            .iter()
            .all(|d| nvlink_neighbors.get(d).map(|v| v.len()).unwrap_or(0) == 2);

        if is_ring && self.verify_ring(&nvlink_neighbors) {
            return TopologyType::NvLinkRing;
        }

        // Default: partial NVLink connectivity
        TopologyType::NvLinkMesh
    }

    /// Verify that the given adjacency forms a single Hamiltonian cycle.
    fn verify_ring(&self, neighbors: &HashMap<i32, Vec<i32>>) -> bool {
        if self.devices.is_empty() {
            return false;
        }
        let start = self.devices[0];
        let mut visited = vec![false; self.devices.len()];
        let mut current = start;
        let mut prev = -1_i32;

        for step in 0..self.devices.len() {
            let idx = match self.device_index(current) {
                Some(i) => i,
                None => return false,
            };
            if visited[idx] {
                return false;
            }
            visited[idx] = true;

            let nbrs = match neighbors.get(&current) {
                Some(v) => v,
                None => return false,
            };

            if step < self.devices.len() - 1 {
                // Move to the neighbor that isn't `prev`
                let next = nbrs.iter().find(|&&n| n != prev);
                match next {
                    Some(&n) => {
                        prev = current;
                        current = n;
                    }
                    None => return false,
                }
            } else {
                // Last step: must connect back to start
                return nbrs.contains(&start);
            }
        }
        false
    }

    // -- Path finding -------------------------------------------------------

    /// Find the fastest path between two devices (maximising bandwidth).
    ///
    /// Uses Dijkstra on inverse-bandwidth weights so that the path with the
    /// highest minimum-bandwidth bottleneck is found.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidDevice`] if either device is not in
    /// the topology.
    pub fn best_path(&self, src: i32, dst: i32) -> CudaResult<Vec<i32>> {
        let src_idx = self.device_index(src).ok_or(CudaError::InvalidDevice)?;
        let dst_idx = self.device_index(dst).ok_or(CudaError::InvalidDevice)?;

        if src_idx == dst_idx {
            return Ok(vec![src]);
        }

        let n = self.devices.len();
        // Use inverse bandwidth as weight for Dijkstra
        let mut dist = vec![f64::INFINITY; n];
        let mut prev: Vec<Option<usize>> = vec![None; n];
        dist[src_idx] = 0.0;

        let mut heap: BinaryHeap<Reverse<(OrderedF64, usize)>> = BinaryHeap::new();
        heap.push(Reverse((OrderedF64(0.0), src_idx)));

        while let Some(Reverse((OrderedF64(cost), u))) = heap.pop() {
            if u == dst_idx {
                break;
            }
            if cost > dist[u] {
                continue;
            }
            for v in 0..n {
                if u == v {
                    continue;
                }
                let bw = self.adj_bandwidth[u][v];
                if bw <= 0.0 {
                    continue;
                }
                let edge_cost = 1.0 / bw;
                let new_dist = dist[u] + edge_cost;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    prev[v] = Some(u);
                    heap.push(Reverse((OrderedF64(new_dist), v)));
                }
            }
        }

        // Reconstruct path
        if prev[dst_idx].is_none() {
            return Err(CudaError::InvalidValue);
        }

        let mut path = Vec::new();
        let mut cur = dst_idx;
        while let Some(p) = prev[cur] {
            path.push(self.devices[cur]);
            cur = p;
        }
        path.push(self.devices[src_idx]);
        path.reverse();
        Ok(path)
    }

    // -- Bandwidth & latency queries ----------------------------------------

    /// Returns the direct bandwidth between two devices in GB/s.
    ///
    /// Returns 0.0 if the devices are not directly connected, and
    /// `f64::INFINITY` for same-device queries.
    pub fn bandwidth_between(&self, src: i32, dst: i32) -> f64 {
        let src_idx = match self.device_index(src) {
            Some(i) => i,
            None => return 0.0,
        };
        let dst_idx = match self.device_index(dst) {
            Some(i) => i,
            None => return 0.0,
        };
        self.adj_bandwidth[src_idx][dst_idx]
    }

    /// Returns the estimated latency between two devices in microseconds.
    ///
    /// Returns 0.0 for same-device and `f64::MAX` (converted to us) if
    /// no connection exists.
    pub fn latency_between(&self, src: i32, dst: i32) -> f64 {
        let src_idx = match self.device_index(src) {
            Some(i) => i,
            None => return f64::MAX / 1000.0,
        };
        let dst_idx = match self.device_index(dst) {
            Some(i) => i,
            None => return f64::MAX / 1000.0,
        };
        self.adj_latency[src_idx][dst_idx] / 1000.0 // ns -> us
    }

    // -- Ring order for AllReduce -------------------------------------------

    /// Find an optimal ring ordering for AllReduce collective.
    ///
    /// Uses a greedy nearest-bandwidth-neighbor heuristic: starting from
    /// device 0, always pick the unvisited neighbor with the highest
    /// bandwidth.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NoDevice`] if the topology has no devices.
    pub fn optimal_ring_order(&self) -> CudaResult<Vec<i32>> {
        let n = self.devices.len();
        if n == 0 {
            return Err(CudaError::NoDevice);
        }
        if n == 1 {
            return Ok(vec![self.devices[0]]);
        }

        let mut visited = vec![false; n];
        let mut ring = Vec::with_capacity(n);

        // Start from device 0
        let mut current = 0_usize;
        visited[current] = true;
        ring.push(self.devices[current]);

        for _ in 1..n {
            // Pick unvisited neighbor with highest bandwidth
            let mut best_bw = -1.0_f64;
            let mut best_idx = None;
            for (j, &is_visited) in visited.iter().enumerate() {
                if is_visited {
                    continue;
                }
                let bw = self.adj_bandwidth[current][j];
                if bw > best_bw {
                    best_bw = bw;
                    best_idx = Some(j);
                }
            }
            match best_idx {
                Some(idx) => {
                    visited[idx] = true;
                    ring.push(self.devices[idx]);
                    current = idx;
                }
                None => {
                    // No reachable unvisited device — shouldn't happen in connected graph
                    return Err(CudaError::InvalidValue);
                }
            }
        }

        Ok(ring)
    }

    // -- Optimal tree for broadcast/reduce ----------------------------------

    /// Build an optimal spanning tree for broadcast/reduce collectives.
    ///
    /// Constructs a maximum-bandwidth spanning tree rooted at the device
    /// with the best aggregate bandwidth to all others (Prim-style).
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NoDevice`] if the topology has no devices.
    pub fn optimal_tree(&self) -> CudaResult<TopologyTree> {
        let n = self.devices.len();
        if n == 0 {
            return Err(CudaError::NoDevice);
        }

        // Pick root: device with highest total outgoing bandwidth
        let root_idx = (0..n)
            .max_by(|&a, &b| {
                let sum_a: f64 = (0..n)
                    .filter(|&j| j != a)
                    .map(|j| {
                        let bw = self.adj_bandwidth[a][j];
                        if bw.is_infinite() { 0.0 } else { bw }
                    })
                    .sum();
                let sum_b: f64 = (0..n)
                    .filter(|&j| j != b)
                    .map(|j| {
                        let bw = self.adj_bandwidth[b][j];
                        if bw.is_infinite() { 0.0 } else { bw }
                    })
                    .sum();
                sum_a
                    .partial_cmp(&sum_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        let root = self.devices[root_idx];
        let mut children: HashMap<i32, Vec<i32>> = HashMap::new();
        let mut parent: HashMap<i32, i32> = HashMap::new();

        // Initialize children for all devices
        for &d in &self.devices {
            children.insert(d, Vec::new());
        }

        if n == 1 {
            return Ok(TopologyTree {
                root,
                children,
                parent,
            });
        }

        // Prim's algorithm for maximum spanning tree
        let mut in_tree = vec![false; n];
        let mut best_edge: Vec<(f64, Option<usize>)> = vec![(0.0, None); n];
        in_tree[root_idx] = true;

        // Initialize edges from root
        for (j, edge) in best_edge.iter_mut().enumerate() {
            if j != root_idx {
                let bw = self.adj_bandwidth[root_idx][j];
                let bw_val = if bw.is_infinite() { 0.0 } else { bw };
                *edge = (bw_val, Some(root_idx));
            }
        }

        for _ in 1..n {
            // Find the not-in-tree node with maximum bandwidth edge
            let mut best_bw = -1.0_f64;
            let mut best_node = None;
            for j in 0..n {
                if !in_tree[j] && best_edge[j].0 > best_bw {
                    best_bw = best_edge[j].0;
                    best_node = Some(j);
                }
            }

            let node = match best_node {
                Some(j) => j,
                None => break,
            };

            in_tree[node] = true;
            let parent_idx = match best_edge[node].1 {
                Some(p) => p,
                None => continue,
            };

            let parent_dev = self.devices[parent_idx];
            let child_dev = self.devices[node];
            parent.insert(child_dev, parent_dev);
            children.entry(parent_dev).or_default().push(child_dev);

            // Update best edges for remaining nodes
            for j in 0..n {
                if !in_tree[j] {
                    let bw = self.adj_bandwidth[node][j];
                    let bw_val = if bw.is_infinite() { 0.0 } else { bw };
                    if bw_val > best_edge[j].0 {
                        best_edge[j] = (bw_val, Some(node));
                    }
                }
            }
        }

        Ok(TopologyTree {
            root,
            children,
            parent,
        })
    }

    // -- Task placement -----------------------------------------------------

    /// Assign tasks to devices minimising total communication cost.
    ///
    /// For small task counts (<=8), tries all permutations. For larger counts,
    /// uses a greedy assignment heuristic.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if there are more tasks than devices,
    /// or if the topology has no devices.
    pub fn optimal_placement(
        &self,
        num_tasks: usize,
        communications: &[TaskCommunication],
    ) -> CudaResult<TopologyAwarePlacement> {
        let n = self.devices.len();
        if num_tasks == 0 || n == 0 {
            return Err(CudaError::InvalidValue);
        }
        if num_tasks > n {
            return Err(CudaError::InvalidValue);
        }

        if num_tasks <= 8 && n <= 8 {
            self.placement_exhaustive(num_tasks, communications)
        } else {
            self.placement_greedy(num_tasks, communications)
        }
    }

    /// Exhaustive (permutation-based) placement for small instances.
    fn placement_exhaustive(
        &self,
        num_tasks: usize,
        communications: &[TaskCommunication],
    ) -> CudaResult<TopologyAwarePlacement> {
        let n = self.devices.len();
        let mut best_cost = f64::INFINITY;
        let mut best_assignment: HashMap<usize, i32> = HashMap::new();

        // Generate permutations of device indices (up to num_tasks)
        let mut perm: Vec<usize> = (0..n).collect();
        let mut found = false;

        // Heap's algorithm for permutations
        Self::for_each_permutation(&mut perm, n, &mut |p| {
            let assignment: HashMap<usize, i32> =
                (0..num_tasks).map(|t| (t, self.devices[p[t]])).collect();
            let cost = Self::compute_placement_cost(
                &assignment,
                communications,
                &self.adj_bandwidth,
                &self.devices,
            );
            if cost < best_cost {
                best_cost = cost;
                best_assignment = assignment;
                found = true;
            }
        });

        if !found {
            // Fallback: identity mapping
            best_assignment = (0..num_tasks).map(|t| (t, self.devices[t])).collect();
            best_cost = Self::compute_placement_cost(
                &best_assignment,
                communications,
                &self.adj_bandwidth,
                &self.devices,
            );
        }

        Ok(TopologyAwarePlacement {
            assignment: best_assignment,
            total_cost: best_cost,
        })
    }

    /// Heap's algorithm to iterate over all permutations.
    fn for_each_permutation(arr: &mut [usize], k: usize, callback: &mut impl FnMut(&[usize])) {
        if k == 1 {
            callback(arr);
            return;
        }
        Self::for_each_permutation(arr, k - 1, callback);
        for i in 0..k - 1 {
            if k % 2 == 0 {
                arr.swap(i, k - 1);
            } else {
                arr.swap(0, k - 1);
            }
            Self::for_each_permutation(arr, k - 1, callback);
        }
    }

    /// Greedy placement for larger instances.
    fn placement_greedy(
        &self,
        num_tasks: usize,
        communications: &[TaskCommunication],
    ) -> CudaResult<TopologyAwarePlacement> {
        // Sort tasks by total communication volume (descending)
        let mut task_volume: Vec<(usize, u64)> = (0..num_tasks).map(|t| (t, 0u64)).collect();
        for comm in communications {
            if comm.task_a < num_tasks {
                task_volume[comm.task_a].1 += comm.volume_bytes;
            }
            if comm.task_b < num_tasks {
                task_volume[comm.task_b].1 += comm.volume_bytes;
            }
        }
        task_volume.sort_by_key(|entry| std::cmp::Reverse(entry.1));

        let mut assignment: HashMap<usize, i32> = HashMap::new();
        let mut used_devices: Vec<bool> = vec![false; self.devices.len()];

        for &(task, _) in &task_volume {
            // Find best device for this task given current assignments
            let mut best_cost = f64::INFINITY;
            let mut best_dev_idx = None;

            for (dev_idx, &used) in used_devices.iter().enumerate() {
                if used {
                    continue;
                }
                // Compute cost of placing this task on this device
                let mut cost = 0.0_f64;
                for comm in communications {
                    let (peer_task, is_relevant) = if comm.task_a == task {
                        (comm.task_b, true)
                    } else if comm.task_b == task {
                        (comm.task_a, true)
                    } else {
                        (0, false)
                    };
                    if !is_relevant {
                        continue;
                    }
                    if let Some(&peer_dev) = assignment.get(&peer_task) {
                        let peer_dev_idx = match self.device_index(peer_dev) {
                            Some(i) => i,
                            None => continue,
                        };
                        let bw = self.adj_bandwidth[dev_idx][peer_dev_idx];
                        let bw_val = if bw.is_infinite() || bw <= 0.0 {
                            1e-9
                        } else {
                            bw
                        };
                        cost += comm.volume_bytes as f64 / bw_val;
                    }
                }
                if cost < best_cost {
                    best_cost = cost;
                    best_dev_idx = Some(dev_idx);
                }
            }

            let dev_idx =
                best_dev_idx.unwrap_or_else(|| used_devices.iter().position(|&u| !u).unwrap_or(0));
            used_devices[dev_idx] = true;
            assignment.insert(task, self.devices[dev_idx]);
        }

        let total_cost = Self::compute_placement_cost(
            &assignment,
            communications,
            &self.adj_bandwidth,
            &self.devices,
        );

        Ok(TopologyAwarePlacement {
            assignment,
            total_cost,
        })
    }

    /// Compute total communication cost for a given placement.
    fn compute_placement_cost(
        assignment: &HashMap<usize, i32>,
        communications: &[TaskCommunication],
        adj_bandwidth: &[Vec<f64>],
        devices: &[i32],
    ) -> f64 {
        let dev_to_idx: HashMap<i32, usize> =
            devices.iter().enumerate().map(|(i, &d)| (d, i)).collect();

        let mut cost = 0.0_f64;
        for comm in communications {
            let dev_a = match assignment.get(&comm.task_a) {
                Some(&d) => d,
                None => continue,
            };
            let dev_b = match assignment.get(&comm.task_b) {
                Some(&d) => d,
                None => continue,
            };
            if dev_a == dev_b {
                continue; // Same device, no cost
            }
            let idx_a = match dev_to_idx.get(&dev_a) {
                Some(&i) => i,
                None => continue,
            };
            let idx_b = match dev_to_idx.get(&dev_b) {
                Some(&i) => i,
                None => continue,
            };
            let bw = adj_bandwidth[idx_a][idx_b];
            let bw_val = if bw.is_infinite() || bw <= 0.0 {
                1e-9
            } else {
                bw
            };
            cost += comm.volume_bytes as f64 / bw_val;
        }
        cost
    }

    // -- Communication schedule ---------------------------------------------

    /// Build an optimised communication schedule for a set of transfers.
    ///
    /// Sorts transfers by bandwidth (highest-bandwidth links first) so that
    /// high-bandwidth transfers are scheduled early.
    pub fn build_schedule(&self, transfers: &[(i32, i32, u64)]) -> CommunicationSchedule {
        let mut entries: Vec<(f64, Transfer)> = transfers
            .iter()
            .map(|&(src, dst, size)| {
                let bw = self.bandwidth_between(src, dst);
                let bw_val = if bw.is_infinite() || bw <= 0.0 {
                    1e-9
                } else {
                    bw
                };
                (
                    bw_val,
                    Transfer {
                        src,
                        dst,
                        data_size: size,
                    },
                )
            })
            .collect();

        // Sort by bandwidth descending (schedule high-bandwidth transfers first)
        entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut total_time_us = 0.0_f64;
        let transfers_out: Vec<Transfer> = entries
            .into_iter()
            .map(|(bw, t)| {
                // time = size_bytes / (bw_gbps * 1e9) * 1e6 = size / (bw * 1000)
                total_time_us += t.data_size as f64 / (bw * 1000.0);
                t
            })
            .collect();

        CommunicationSchedule {
            transfers: transfers_out,
            estimated_time_us: total_time_us,
        }
    }

    // -- NVLink info query --------------------------------------------------

    /// Query NVLink information for a specific device pair.
    pub fn nvlink_info(&self, device: i32, peer: i32) -> Option<NvLinkInfo> {
        let link = self
            .links
            .iter()
            .find(|l| l.from_device == device && l.to_device == peer)?;

        if link.link_type == LinkType::PCIe {
            return None;
        }

        let version = match link.link_type {
            LinkType::NvSwitch => NvLinkVersion::NvSwitch,
            _ => NvLinkVersion::V3, // Default assumption
        };

        Some(NvLinkInfo {
            version,
            bandwidth_gbps: link.bandwidth_gbps,
            link_count: if link.bandwidth_gbps > 0.0 {
                (link.bandwidth_gbps / version.per_link_bandwidth_gbps()).ceil() as u32
            } else {
                0
            },
            peer_device_id: peer,
        })
    }

    // -- Helpers ------------------------------------------------------------

    /// Find the index of a device ordinal in `self.devices`.
    fn device_index(&self, device_id: i32) -> Option<usize> {
        self.devices.iter().position(|&d| d == device_id)
    }

    /// Return the adjacency bandwidth matrix (read-only).
    pub fn adjacency_matrix(&self) -> &Vec<Vec<f64>> {
        &self.adj_bandwidth
    }
}

// ---------------------------------------------------------------------------
// OrderedF64 — wrapper for use in BinaryHeap
// ---------------------------------------------------------------------------

/// Wrapper around f64 that implements Ord for use in priority queues.
///
/// NaN values are treated as equal and greater than all finite values.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_4gpu() -> GpuTopology {
        GpuTopology::synthetic_mesh(4).expect("synthetic mesh should not fail")
    }

    fn make_synthetic_1gpu() -> GpuTopology {
        GpuTopology::synthetic_mesh(1).expect("single GPU mesh should not fail")
    }

    #[test]
    fn test_topology_discovery_macos_synthetic() {
        let topo = GpuTopology::discover();
        // On macOS this should succeed with synthetic data
        #[cfg(target_os = "macos")]
        {
            let topo = topo.expect("discover should succeed on macOS");
            assert_eq!(topo.devices.len(), 4);
            // 4 devices * 3 peers = 12 links
            assert_eq!(topo.links.len(), 12);
        }
        // On other platforms, result depends on hardware
        #[cfg(not(target_os = "macos"))]
        {
            let _ = topo; // May fail if no GPU
        }
    }

    #[test]
    fn test_topology_type_single_gpu() {
        let topo = make_synthetic_1gpu();
        assert_eq!(topo.topology_type(), TopologyType::SingleGpu);
    }

    #[test]
    fn test_topology_type_nvlink_mesh() {
        let topo = make_synthetic_4gpu();
        assert_eq!(topo.topology_type(), TopologyType::NvLinkMesh);
    }

    #[test]
    fn test_topology_type_pcie_only() {
        let mut topo = make_synthetic_4gpu();
        // Replace all NVLink with PCIe
        for link in &mut topo.links {
            link.link_type = LinkType::PCIe;
        }
        assert_eq!(topo.topology_type(), TopologyType::PcieOnly);
    }

    #[test]
    fn test_topology_type_nvlink_pair() {
        let topo = GpuTopology::synthetic_mesh(2).expect("pair mesh");
        assert_eq!(topo.topology_type(), TopologyType::NvLinkMesh);
        // But with only 2 and partial connectivity:
        let mut partial = GpuTopology {
            devices: vec![0, 1],
            links: vec![TopologyLink {
                from_device: 0,
                to_device: 1,
                link_type: LinkType::NvLink,
                bandwidth_gbps: 50.0,
                latency_ns: 500.0,
                hop_count: 1,
            }],
            adj_bandwidth: vec![vec![f64::INFINITY, 50.0], vec![0.0, f64::INFINITY]],
            adj_latency: vec![vec![0.0, 500.0], vec![f64::MAX, 0.0]],
        };
        // Only one direction — not full mesh
        assert_eq!(partial.topology_type(), TopologyType::NvLinkPair);

        // Add reverse link to make it full mesh
        partial.links.push(TopologyLink {
            from_device: 1,
            to_device: 0,
            link_type: LinkType::NvLink,
            bandwidth_gbps: 50.0,
            latency_ns: 500.0,
            hop_count: 1,
        });
        partial.adj_bandwidth[1][0] = 50.0;
        partial.adj_latency[1][0] = 500.0;
        assert_eq!(partial.topology_type(), TopologyType::NvLinkMesh);
    }

    #[test]
    fn test_best_path_direct() {
        let topo = make_synthetic_4gpu();
        let path = topo.best_path(0, 3).expect("path should exist");
        // In a full mesh, direct path is optimal
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], 0);
        assert_eq!(path[1], 3);
    }

    #[test]
    fn test_best_path_same_device() {
        let topo = make_synthetic_4gpu();
        let path = topo.best_path(1, 1).expect("same device path");
        assert_eq!(path, vec![1]);
    }

    #[test]
    fn test_best_path_invalid_device() {
        let topo = make_synthetic_4gpu();
        let result = topo.best_path(0, 99);
        assert!(result.is_err());
    }

    #[test]
    fn test_bandwidth_between() {
        let topo = make_synthetic_4gpu();
        let bw = topo.bandwidth_between(0, 1);
        assert!((bw - 100.0).abs() < 1e-6);

        // Same device
        let bw_self = topo.bandwidth_between(0, 0);
        assert!(bw_self.is_infinite());

        // Invalid device
        let bw_invalid = topo.bandwidth_between(0, 99);
        assert!((bw_invalid - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_latency_between() {
        let topo = make_synthetic_4gpu();
        // 500 ns = 0.5 us
        let lat = topo.latency_between(0, 1);
        assert!((lat - 0.5).abs() < 1e-6);

        // Same device: 0.0
        let lat_self = topo.latency_between(0, 0);
        assert!((lat_self - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ring_order() {
        let topo = make_synthetic_4gpu();
        let ring = topo.optimal_ring_order().expect("ring order");
        // All devices present
        assert_eq!(ring.len(), 4);
        let mut sorted = ring.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_ring_order_single() {
        let topo = make_synthetic_1gpu();
        let ring = topo.optimal_ring_order().expect("ring order");
        assert_eq!(ring, vec![0]);
    }

    #[test]
    fn test_tree_construction() {
        let topo = make_synthetic_4gpu();
        let tree = topo.optimal_tree().expect("tree");
        assert!(topo.devices.contains(&tree.root));
        // All non-root devices have a parent
        assert_eq!(tree.parent.len(), 3);
        // All devices are in the tree
        let devs = tree.devices();
        assert_eq!(devs.len(), 4);
        // Tree depth is at least 1
        assert!(tree.depth() >= 1);
    }

    #[test]
    fn test_adjacency_matrix_correctness() {
        let topo = make_synthetic_4gpu();
        let adj = topo.adjacency_matrix();
        assert_eq!(adj.len(), 4);
        for (i, adj_row) in adj.iter().enumerate().take(4) {
            assert!(adj_row[i].is_infinite()); // Self-bandwidth is infinity
            for (j, adj_val) in adj_row.iter().enumerate().take(4) {
                if i != j {
                    assert!((*adj_val - 100.0).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_path_symmetry() {
        let topo = make_synthetic_4gpu();
        for i in 0..4_i32 {
            for j in 0..4_i32 {
                if i == j {
                    continue;
                }
                let bw_ij = topo.bandwidth_between(i, j);
                let bw_ji = topo.bandwidth_between(j, i);
                assert!(
                    (bw_ij - bw_ji).abs() < 1e-6,
                    "bandwidth {i}->{j} ({bw_ij}) != {j}->{i} ({bw_ji})"
                );
            }
        }
    }

    #[test]
    fn test_communication_schedule() {
        let topo = make_synthetic_4gpu();
        let transfers = vec![(0, 1, 1_000_000u64), (2, 3, 2_000_000), (0, 3, 500_000)];
        let schedule = topo.build_schedule(&transfers);
        assert_eq!(schedule.transfers.len(), 3);
        assert!(schedule.estimated_time_us > 0.0);
    }

    #[test]
    fn test_placement_optimization() {
        let topo = make_synthetic_4gpu();
        let comms = vec![
            TaskCommunication {
                task_a: 0,
                task_b: 1,
                volume_bytes: 1_000_000,
            },
            TaskCommunication {
                task_a: 1,
                task_b: 2,
                volume_bytes: 2_000_000,
            },
        ];
        let placement = topo
            .optimal_placement(3, &comms)
            .expect("placement should succeed");
        assert_eq!(placement.assignment.len(), 3);
        // All tasks assigned to different devices
        let mut devs: Vec<i32> = placement.assignment.values().copied().collect();
        devs.sort_unstable();
        devs.dedup();
        assert_eq!(devs.len(), 3);
    }

    #[test]
    fn test_nvswitch_fabric_detection() {
        let mut topo = make_synthetic_4gpu();
        for link in &mut topo.links {
            link.link_type = LinkType::NvSwitch;
        }
        assert_eq!(topo.topology_type(), TopologyType::NvSwitchFabric);
    }

    #[test]
    fn test_empty_topology_errors() {
        let result = GpuTopology::synthetic_mesh(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_nvlink_info_link_count_correct() {
        // Synthetic 4-GPU mesh has 100 GB/s links using NvLink V3 (25 GB/s per link).
        // Expected link_count = ceil(100 / 25) = 4.
        let topo = make_synthetic_4gpu();
        let info = topo
            .nvlink_info(0, 1)
            .expect("NVLink info should exist for NvLink topology");
        assert_eq!(info.peer_device_id, 1);
        assert!((info.bandwidth_gbps - 100.0).abs() < 1e-6);
        let per_link = info.version.per_link_bandwidth_gbps();
        let expected_count = (100.0_f64 / per_link).ceil() as u32;
        assert_eq!(
            info.link_count, expected_count,
            "link count should match bandwidth / per-link rate"
        );
    }

    #[test]
    fn test_nvlink_info_returns_none_for_pcie_link() {
        let mut topo = make_synthetic_4gpu();
        // Force all links to PCIe
        for link in &mut topo.links {
            link.link_type = LinkType::PCIe;
        }
        // nvlink_info returns None for PCIe links
        let info = topo.nvlink_info(0, 1);
        assert!(info.is_none(), "should return None for PCIe links");
    }

    #[test]
    fn test_nvlink_info_nonexistent_peer() {
        let topo = make_synthetic_4gpu();
        // Device 99 doesn't exist
        let info = topo.nvlink_info(0, 99);
        assert!(info.is_none(), "should return None for nonexistent peer");
    }

    #[test]
    fn test_best_path_finds_intermediate_hop() {
        // Build a chain topology: 0 -- 1 -- 2 (no direct 0->2 link)
        let bw_direct = 100.0_f64;
        let lat = 500.0_f64;
        let topo = GpuTopology {
            devices: vec![0, 1, 2],
            links: vec![
                TopologyLink {
                    from_device: 0,
                    to_device: 1,
                    link_type: LinkType::NvLink,
                    bandwidth_gbps: bw_direct,
                    latency_ns: lat,
                    hop_count: 1,
                },
                TopologyLink {
                    from_device: 1,
                    to_device: 0,
                    link_type: LinkType::NvLink,
                    bandwidth_gbps: bw_direct,
                    latency_ns: lat,
                    hop_count: 1,
                },
                TopologyLink {
                    from_device: 1,
                    to_device: 2,
                    link_type: LinkType::NvLink,
                    bandwidth_gbps: bw_direct,
                    latency_ns: lat,
                    hop_count: 1,
                },
                TopologyLink {
                    from_device: 2,
                    to_device: 1,
                    link_type: LinkType::NvLink,
                    bandwidth_gbps: bw_direct,
                    latency_ns: lat,
                    hop_count: 1,
                },
            ],
            adj_bandwidth: vec![
                vec![f64::INFINITY, bw_direct, 0.0],
                vec![bw_direct, f64::INFINITY, bw_direct],
                vec![0.0, bw_direct, f64::INFINITY],
            ],
            adj_latency: vec![
                vec![0.0, lat, f64::MAX],
                vec![lat, 0.0, lat],
                vec![f64::MAX, lat, 0.0],
            ],
        };

        // No direct 0->2 link: path must go through 1
        let path = topo.best_path(0, 2).expect("path should exist via hop");
        assert_eq!(path.len(), 3, "chain topology requires intermediate hop");
        assert_eq!(path[0], 0);
        assert_eq!(path[1], 1);
        assert_eq!(path[2], 2);
    }

    #[test]
    fn test_build_schedule_sorted_by_bandwidth_desc() {
        let topo = make_synthetic_4gpu();
        // All links have equal 100 GB/s bandwidth in synthetic mesh,
        // so verify the schedule is built for all transfers and time is positive.
        let transfers = vec![(0, 1, 1_000_000_u64), (1, 2, 500_000), (2, 3, 2_000_000)];
        let schedule = topo.build_schedule(&transfers);
        assert_eq!(
            schedule.transfers.len(),
            3,
            "all transfers should appear in schedule"
        );
        assert!(
            schedule.estimated_time_us > 0.0,
            "total time should be positive"
        );
        // The 2 MB transfer should take longer than the 0.5 MB one at the same BW
        let time_2mb = 2_000_000.0_f64 / (100.0 * 1000.0);
        let time_1mb = 1_000_000.0_f64 / (100.0 * 1000.0);
        let time_0_5mb = 500_000.0_f64 / (100.0 * 1000.0);
        let expected_total = time_2mb + time_1mb + time_0_5mb;
        assert!(
            (schedule.estimated_time_us - expected_total).abs() < 1e-6,
            "estimated time should match sum of individual transfer times"
        );
    }

    #[test]
    fn test_adjacency_matrix_chain_topology() {
        // 3-node chain: 0-1-2 with no direct 0-2 link
        let topo = GpuTopology {
            devices: vec![0, 1, 2],
            links: vec![],
            adj_bandwidth: vec![
                vec![f64::INFINITY, 50.0, 0.0],
                vec![50.0, f64::INFINITY, 50.0],
                vec![0.0, 50.0, f64::INFINITY],
            ],
            adj_latency: vec![
                vec![0.0, 500.0, f64::MAX],
                vec![500.0, 0.0, 500.0],
                vec![f64::MAX, 500.0, 0.0],
            ],
        };
        let adj = topo.adjacency_matrix();
        assert!(
            (adj[0][1] - 50.0).abs() < 1e-6,
            "0->1 bandwidth should be 50 GB/s"
        );
        assert!((adj[0][2] - 0.0).abs() < 1e-6, "0->2 has no direct link");
        assert!(adj[1][1].is_infinite(), "self-bandwidth is infinite");
    }

    #[test]
    fn test_optimal_placement_all_same_device_no_cost() {
        // With only 1 task, placement should succeed with no communication cost.
        let topo = make_synthetic_4gpu();
        let comms = vec![];
        let placement = topo
            .optimal_placement(1, &comms)
            .expect("single-task placement should succeed");
        assert_eq!(placement.assignment.len(), 1);
        assert_eq!(placement.total_cost, 0.0, "no comms means no cost");
    }

    #[test]
    fn test_topology_tree_single_device() {
        let topo = make_synthetic_1gpu();
        let tree = topo.optimal_tree().expect("single-device tree");
        assert_eq!(tree.root, 0);
        assert_eq!(tree.parent.len(), 0, "root has no parent");
        assert_eq!(tree.depth(), 1, "single-node tree has depth 1");
    }
}
