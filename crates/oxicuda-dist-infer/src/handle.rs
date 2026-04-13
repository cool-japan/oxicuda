//! Session handle carrying distributed topology metadata.
//!
//! `DistInferHandle` is a lightweight, cheaply-cloned descriptor that tells
//! every sub-system the following about the current process:
//!
//! * Which GPU device it controls (`device`).
//! * Its rank within each parallelism axis (tensor / sequence / expert).
//! * The degrees of parallelism on each axis and their product = world_size.
//! * A PTX SM version string for JIT kernel generation.

use crate::error::{DistInferError, DistInferResult};

// ─── SmVersion ──────────────────────────────────────────────────────────────

/// CUDA SM architecture version, e.g. `SmVersion(80)` for Ampere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmVersion(pub u32);

impl SmVersion {
    /// Map to the PTX ISA version string used in `.version` directives.
    pub fn ptx_version_str(self) -> &'static str {
        match self.0 {
            v if v >= 120 => "8.7",
            v if v >= 100 => "8.7",
            v if v >= 90 => "8.4",
            v if v >= 80 => "8.0",
            _ => "7.5",
        }
    }

    /// PTX target string, e.g. `"sm_80"`.
    pub fn target_str(self) -> String {
        format!("sm_{}", self.0)
    }
}

// ─── ParallelismConfig ──────────────────────────────────────────────────────

/// Describes a three-way parallelism decomposition:
/// **TP × SP × EP = world_size**.
///
/// Axes:
/// * **tp** — Tensor parallelism: weight matrices are column- or row-sharded.
/// * **sp** — Sequence parallelism: the token sequence is chunked.
/// * **ep** — Expert parallelism: MoE experts are partitioned across ranks.
///
/// Any axis may be set to 1 (disabled). The product must equal `world_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelismConfig {
    /// Tensor-parallel degree (1 = disabled).
    pub tp: usize,
    /// Sequence-parallel degree (1 = disabled).
    pub sp: usize,
    /// Expert-parallel degree (1 = disabled).
    pub ep: usize,
}

impl ParallelismConfig {
    /// Data-parallel only (all axes = 1).
    pub const DATA_PARALLEL: Self = Self {
        tp: 1,
        sp: 1,
        ep: 1,
    };

    /// Total number of participating ranks: `tp * sp * ep`.
    pub fn world_size(&self) -> usize {
        self.tp * self.sp * self.ep
    }

    /// Validate that all degrees are ≥ 1.
    pub fn validate(&self) -> DistInferResult<()> {
        if self.tp == 0 || self.sp == 0 || self.ep == 0 {
            return Err(DistInferError::InvalidWorldSize {
                world_size: self.tp * self.sp * self.ep,
                reason: "all parallelism degrees must be ≥ 1",
            });
        }
        Ok(())
    }
}

// ─── RankCoordinates ────────────────────────────────────────────────────────

/// The 3-D coordinates of this rank in `(tp_rank, sp_rank, ep_rank)` space.
///
/// Given `parallelism = ParallelismConfig { tp, sp, ep }` and a flat
/// `global_rank`, coordinates are computed as:
///
/// ```text
/// tp_rank = global_rank % tp
/// sp_rank = (global_rank / tp) % sp
/// ep_rank = global_rank / (tp * sp)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankCoordinates {
    /// This rank's position in the tensor-parallel group (0..tp).
    pub tp_rank: usize,
    /// This rank's position in the sequence-parallel group (0..sp).
    pub sp_rank: usize,
    /// This rank's position in the expert-parallel group (0..ep).
    pub ep_rank: usize,
    /// Flat global rank across all axes.
    pub global_rank: usize,
}

impl RankCoordinates {
    /// Compute 3-D coordinates from `global_rank` and `config`.
    pub fn from_global(global_rank: usize, config: &ParallelismConfig) -> DistInferResult<Self> {
        let ws = config.world_size();
        if global_rank >= ws {
            return Err(DistInferError::RankOutOfRange {
                rank: global_rank,
                world_size: ws,
            });
        }
        let tp_rank = global_rank % config.tp;
        let sp_rank = (global_rank / config.tp) % config.sp;
        let ep_rank = global_rank / (config.tp * config.sp);
        Ok(Self {
            tp_rank,
            sp_rank,
            ep_rank,
            global_rank,
        })
    }

    /// Recover the flat global rank from 3-D coordinates.
    pub fn to_global(&self, config: &ParallelismConfig) -> usize {
        self.ep_rank * (config.tp * config.sp) + self.sp_rank * config.tp + self.tp_rank
    }

    /// The global rank of the peer that shares the same SP+EP group but has
    /// `tp_rank = r`.  Used for tensor-parallel all-reduce target lookup.
    pub fn peer_tp(&self, tp_rank: usize, config: &ParallelismConfig) -> usize {
        self.ep_rank * (config.tp * config.sp) + self.sp_rank * config.tp + tp_rank
    }

    /// The global rank of the peer that shares the same TP+EP group but has
    /// `sp_rank = r`.  Used for sequence-parallel gather target lookup.
    pub fn peer_sp(&self, sp_rank: usize, config: &ParallelismConfig) -> usize {
        self.ep_rank * (config.tp * config.sp) + sp_rank * config.tp + self.tp_rank
    }

    /// The global rank of the peer that shares the same TP+SP group but has
    /// `ep_rank = r`.  Used for expert-parallel all-to-all target lookup.
    pub fn peer_ep(&self, ep_rank: usize, config: &ParallelismConfig) -> usize {
        ep_rank * (config.tp * config.sp) + self.sp_rank * config.tp + self.tp_rank
    }
}

// ─── DistInferHandle ────────────────────────────────────────────────────────

/// Central session handle for distributed inference.
///
/// All sub-systems (tensor_parallel, sequence_parallel, expert_parallel,
/// distributed_cache, router) hold a clone of this lightweight handle.
#[derive(Debug, Clone)]
pub struct DistInferHandle {
    /// CUDA device ordinal controlled by this rank.
    pub device: i32,
    /// SM version of the device, used for PTX generation.
    pub sm_version: SmVersion,
    /// Parallelism decomposition: tp × sp × ep.
    pub config: ParallelismConfig,
    /// This rank's 3-D coordinates.
    pub coords: RankCoordinates,
}

impl DistInferHandle {
    /// Construct a handle for `global_rank` in `config`.
    ///
    /// `device` is the CUDA ordinal for this rank (caller ensures mapping).
    pub fn new(
        device: i32,
        sm_version: SmVersion,
        global_rank: usize,
        config: ParallelismConfig,
    ) -> DistInferResult<Self> {
        config.validate()?;
        let coords = RankCoordinates::from_global(global_rank, &config)?;
        Ok(Self {
            device,
            sm_version,
            config,
            coords,
        })
    }

    /// Build a single-rank (pure local) handle — convenient for tests.
    pub fn single_rank(sm: u32) -> Self {
        Self {
            device: 0,
            sm_version: SmVersion(sm),
            config: ParallelismConfig::DATA_PARALLEL,
            coords: RankCoordinates {
                tp_rank: 0,
                sp_rank: 0,
                ep_rank: 0,
                global_rank: 0,
            },
        }
    }

    /// World size = tp × sp × ep.
    pub fn world_size(&self) -> usize {
        self.config.world_size()
    }

    /// This rank's flat global rank.
    pub fn global_rank(&self) -> usize {
        self.coords.global_rank
    }

    /// This rank's position within the tensor-parallel group.
    pub fn tp_rank(&self) -> usize {
        self.coords.tp_rank
    }

    /// This rank's position within the sequence-parallel group.
    pub fn sp_rank(&self) -> usize {
        self.coords.sp_rank
    }

    /// This rank's position within the expert-parallel group.
    pub fn ep_rank(&self) -> usize {
        self.coords.ep_rank
    }

    /// PTX ISA version string, e.g. `"8.0"`.
    pub fn ptx_version_str(&self) -> &'static str {
        self.sm_version.ptx_version_str()
    }

    /// PTX target string, e.g. `"sm_80"`.
    pub fn ptx_target(&self) -> String {
        self.sm_version.target_str()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_rank_handle() {
        let h = DistInferHandle::single_rank(80);
        assert_eq!(h.world_size(), 1);
        assert_eq!(h.global_rank(), 0);
        assert_eq!(h.tp_rank(), 0);
        assert_eq!(h.sp_rank(), 0);
        assert_eq!(h.ep_rank(), 0);
        assert_eq!(h.ptx_version_str(), "8.0");
        assert_eq!(h.ptx_target(), "sm_80");
    }

    #[test]
    fn rank_coordinates_tp2_sp2_ep2() {
        // 8 ranks: tp=2, sp=2, ep=2 → world_size=8
        let cfg = ParallelismConfig {
            tp: 2,
            sp: 2,
            ep: 2,
        };
        // rank 0 → (tp=0, sp=0, ep=0)
        let c0 = RankCoordinates::from_global(0, &cfg).unwrap();
        assert_eq!((c0.tp_rank, c0.sp_rank, c0.ep_rank), (0, 0, 0));
        // rank 1 → (tp=1, sp=0, ep=0)
        let c1 = RankCoordinates::from_global(1, &cfg).unwrap();
        assert_eq!((c1.tp_rank, c1.sp_rank, c1.ep_rank), (1, 0, 0));
        // rank 2 → (tp=0, sp=1, ep=0)
        let c2 = RankCoordinates::from_global(2, &cfg).unwrap();
        assert_eq!((c2.tp_rank, c2.sp_rank, c2.ep_rank), (0, 1, 0));
        // rank 7 → (tp=1, sp=1, ep=1)
        let c7 = RankCoordinates::from_global(7, &cfg).unwrap();
        assert_eq!((c7.tp_rank, c7.sp_rank, c7.ep_rank), (1, 1, 1));
        // round-trip
        assert_eq!(c7.to_global(&cfg), 7);
    }

    #[test]
    fn rank_out_of_range() {
        let cfg = ParallelismConfig {
            tp: 2,
            sp: 2,
            ep: 1,
        };
        let err = RankCoordinates::from_global(4, &cfg).unwrap_err();
        assert!(matches!(
            err,
            DistInferError::RankOutOfRange {
                rank: 4,
                world_size: 4
            }
        ));
    }

    #[test]
    fn config_validation_zero_degree() {
        let cfg = ParallelismConfig {
            tp: 0,
            sp: 1,
            ep: 1,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn peer_tp_lookup() {
        let cfg = ParallelismConfig {
            tp: 4,
            sp: 1,
            ep: 1,
        };
        let coords = RankCoordinates::from_global(0, &cfg).unwrap();
        // tp_rank=0 peer with tp_rank=3 should be global rank 3
        assert_eq!(coords.peer_tp(3, &cfg), 3);
    }

    #[test]
    fn peer_sp_lookup() {
        let cfg = ParallelismConfig {
            tp: 2,
            sp: 4,
            ep: 1,
        };
        // rank 0 → tp=0, sp=0. peer_sp(3) → ep=0, sp=3, tp=0 → global=6
        let coords = RankCoordinates::from_global(0, &cfg).unwrap();
        assert_eq!(coords.peer_sp(3, &cfg), 6);
    }

    #[test]
    fn handle_new_tp4_world() {
        let cfg = ParallelismConfig {
            tp: 4,
            sp: 1,
            ep: 1,
        };
        for rank in 0..4 {
            let h = DistInferHandle::new(rank as i32, SmVersion(90), rank, cfg).unwrap();
            assert_eq!(h.tp_rank(), rank);
            assert_eq!(h.sp_rank(), 0);
            assert_eq!(h.ep_rank(), 0);
        }
    }

    #[test]
    fn sm_version_ptx_strings() {
        assert_eq!(SmVersion(75).ptx_version_str(), "7.5");
        assert_eq!(SmVersion(80).ptx_version_str(), "8.0");
        assert_eq!(SmVersion(90).ptx_version_str(), "8.4");
        assert_eq!(SmVersion(100).ptx_version_str(), "8.7");
        assert_eq!(SmVersion(120).ptx_version_str(), "8.7");
    }
}
