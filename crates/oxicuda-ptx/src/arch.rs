//! NVIDIA GPU architecture definitions and capability queries.
//!
//! This module provides [`SmVersion`] to identify target architectures from
//! Turing (sm_75) through Blackwell (sm_120), and [`ArchCapabilities`] for
//! querying hardware features such as tensor core availability, async copy
//! support, and maximum thread counts.

use std::fmt;

/// NVIDIA GPU Streaming Multiprocessor version.
///
/// Each variant corresponds to a CUDA compute capability and determines
/// the PTX ISA version, available instructions, and hardware features.
///
/// # Ordering
///
/// `SmVersion` derives `Ord` so that newer architectures compare greater
/// than older ones: `Sm80 > Sm75`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SmVersion {
    /// Turing (compute capability 7.5).
    Sm75,
    /// Ampere (compute capability 8.0).
    Sm80,
    /// Ampere `GA10x` (compute capability 8.6).
    Sm86,
    /// Ada Lovelace (compute capability 8.9).
    Sm89,
    /// Hopper (compute capability 9.0).
    Sm90,
    /// Hopper with accelerated features (compute capability 9.0a).
    Sm90a,
    /// Blackwell (compute capability 10.0).
    Sm100,
    /// Blackwell B200 / next-gen (compute capability 12.0).
    Sm120,
}

impl SmVersion {
    /// Returns the PTX target string (e.g. `"sm_80"`, `"sm_90a"`).
    #[must_use]
    pub const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::Sm75 => "sm_75",
            Self::Sm80 => "sm_80",
            Self::Sm86 => "sm_86",
            Self::Sm89 => "sm_89",
            Self::Sm90 => "sm_90",
            Self::Sm90a => "sm_90a",
            Self::Sm100 => "sm_100",
            Self::Sm120 => "sm_120",
        }
    }

    /// Returns the PTX ISA version string appropriate for this architecture.
    ///
    /// The PTX version determines which instructions and features are available.
    /// Later architectures require higher PTX versions.
    #[must_use]
    pub const fn ptx_version(self) -> &'static str {
        match self {
            Self::Sm75 => "6.4",
            Self::Sm80 => "7.0",
            Self::Sm86 => "7.1",
            Self::Sm89 => "7.8",
            Self::Sm90 | Self::Sm90a => "8.0",
            Self::Sm100 => "8.5",
            Self::Sm120 => "8.7",
        }
    }

    /// Returns the PTX ISA version as a `(major, minor)` pair.
    ///
    /// This is useful for programmatic version comparisons rather than
    /// string parsing.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxicuda_ptx::arch::SmVersion;
    ///
    /// assert_eq!(SmVersion::Sm80.ptx_isa_version(), (7, 0));
    /// assert_eq!(SmVersion::Sm90.ptx_isa_version(), (8, 0));
    /// assert_eq!(SmVersion::Sm120.ptx_isa_version(), (8, 7));
    /// ```
    #[must_use]
    pub const fn ptx_isa_version(self) -> (u32, u32) {
        match self {
            Self::Sm75 => (6, 4),
            Self::Sm80 => (7, 0),
            Self::Sm86 => (7, 1),
            Self::Sm89 => (7, 8),
            Self::Sm90 | Self::Sm90a => (8, 0),
            Self::Sm100 => (8, 5),
            Self::Sm120 => (8, 7),
        }
    }

    /// Returns the architecture capabilities for this SM version.
    #[must_use]
    pub const fn capabilities(self) -> ArchCapabilities {
        ArchCapabilities::for_sm(self)
    }

    /// Converts a CUDA compute capability pair to an `SmVersion`.
    ///
    /// Returns `None` if the compute capability is not recognized.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxicuda_ptx::arch::SmVersion;
    ///
    /// assert_eq!(SmVersion::from_compute_capability(8, 0), Some(SmVersion::Sm80));
    /// assert_eq!(SmVersion::from_compute_capability(7, 5), Some(SmVersion::Sm75));
    /// assert_eq!(SmVersion::from_compute_capability(6, 0), None);
    /// ```
    #[must_use]
    pub const fn from_compute_capability(major: i32, minor: i32) -> Option<Self> {
        match (major, minor) {
            (7, 5) => Some(Self::Sm75),
            (8, 0) => Some(Self::Sm80),
            (8, 6) => Some(Self::Sm86),
            (8, 9) => Some(Self::Sm89),
            (9, 0) => Some(Self::Sm90),
            (10, 0) => Some(Self::Sm100),
            (12, 0) => Some(Self::Sm120),
            _ => None,
        }
    }

    /// Returns the maximum number of threads per block for this architecture.
    #[must_use]
    pub const fn max_threads_per_block(self) -> u32 {
        1024
    }

    /// Returns the maximum number of threads per SM for this architecture.
    #[must_use]
    pub const fn max_threads_per_sm(self) -> u32 {
        match self {
            Self::Sm75 => 1024,
            Self::Sm89 => 1536,
            Self::Sm80 | Self::Sm86 | Self::Sm90 | Self::Sm90a | Self::Sm100 | Self::Sm120 => 2048,
        }
    }

    /// Returns the warp size for this architecture (always 32).
    #[must_use]
    pub const fn warp_size(self) -> u32 {
        32
    }

    /// Returns the maximum shared memory per block in bytes.
    #[must_use]
    pub const fn max_shared_mem_per_block(self) -> u32 {
        match self {
            Self::Sm75 => 65536,
            Self::Sm80 | Self::Sm86 => 163_840,
            Self::Sm89 => 101_376,
            Self::Sm90 | Self::Sm90a | Self::Sm100 | Self::Sm120 => 232_448,
        }
    }
}

impl fmt::Display for SmVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_ptx_str())
    }
}

/// Hardware capabilities for a specific GPU architecture.
///
/// Query this struct to determine whether a given feature (tensor cores,
/// async copy, TMA, etc.) is available on the target architecture before
/// emitting instructions that require it.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchCapabilities {
    /// Whether the architecture supports `mma.sync` tensor core instructions.
    pub has_tensor_cores: bool,
    /// Whether `cp.async` (asynchronous global-to-shared copy) is supported.
    pub has_cp_async: bool,
    /// Whether `ldmatrix` (warp-cooperative shared memory load) is supported.
    pub has_ldmatrix: bool,
    /// Whether `mma.sync.aligned.m16n8k16` (Ampere MMA shapes) is supported.
    pub has_ampere_mma: bool,
    /// Whether WGMMA (warp-group MMA, Hopper) instructions are supported.
    pub has_wgmma: bool,
    /// Whether TMA (Tensor Memory Accelerator, Hopper) is supported.
    pub has_tma: bool,
    /// Whether FP8 (E4M3/E5M2) data types are supported.
    pub has_fp8: bool,
    /// Whether FP6/FP4 narrow floating-point types are supported (Blackwell).
    pub has_fp6_fp4: bool,
    /// Whether dynamic shared memory (`extern __shared__`) is supported.
    pub has_dynamic_smem: bool,
    /// Whether `bar.sync` with named barriers is supported.
    pub has_named_barriers: bool,
    /// Whether `fence.mbarrier` and related cluster barriers are supported.
    pub has_cluster_barriers: bool,
    /// Whether `stmatrix` (store matrix to shared memory) is supported (SM >= 90).
    pub has_stmatrix: bool,
    /// Whether `redux.sync` (warp-level reduction) is supported (SM >= 80).
    pub has_redux: bool,
    /// Whether `elect.sync` (warp leader election) is supported (SM >= 90).
    pub has_elect_one: bool,
    /// Whether `griddepcontrol` (grid dependency control) is supported (SM >= 90).
    pub has_griddepcontrol: bool,
    /// Whether `setmaxnreg` (set max register count) is supported (SM >= 90).
    pub has_setmaxnreg: bool,
    /// Whether bulk async copy operations are supported (SM >= 90).
    pub has_bulk_copy: bool,
    /// Whether SM 120 (Rubin) specific features are available.
    pub has_sm120_features: bool,
}

impl ArchCapabilities {
    /// Returns the capabilities for the given SM version.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub const fn for_sm(sm: SmVersion) -> Self {
        match sm {
            SmVersion::Sm75 => Self {
                has_tensor_cores: true,
                has_cp_async: false,
                has_ldmatrix: true,
                has_ampere_mma: false,
                has_wgmma: false,
                has_tma: false,
                has_fp8: false,
                has_fp6_fp4: false,
                has_dynamic_smem: true,
                has_named_barriers: true,
                has_cluster_barriers: false,
                has_stmatrix: false,
                has_redux: false,
                has_elect_one: false,
                has_griddepcontrol: false,
                has_setmaxnreg: false,
                has_bulk_copy: false,
                has_sm120_features: false,
            },
            SmVersion::Sm80 | SmVersion::Sm86 => Self {
                has_tensor_cores: true,
                has_cp_async: true,
                has_ldmatrix: true,
                has_ampere_mma: true,
                has_wgmma: false,
                has_tma: false,
                has_fp8: false,
                has_fp6_fp4: false,
                has_dynamic_smem: true,
                has_named_barriers: true,
                has_cluster_barriers: false,
                has_stmatrix: false,
                has_redux: true,
                has_elect_one: false,
                has_griddepcontrol: false,
                has_setmaxnreg: false,
                has_bulk_copy: false,
                has_sm120_features: false,
            },
            SmVersion::Sm89 => Self {
                has_tensor_cores: true,
                has_cp_async: true,
                has_ldmatrix: true,
                has_ampere_mma: true,
                has_wgmma: false,
                has_tma: false,
                has_fp8: true,
                has_fp6_fp4: false,
                has_dynamic_smem: true,
                has_named_barriers: true,
                has_cluster_barriers: false,
                has_stmatrix: false,
                has_redux: true,
                has_elect_one: false,
                has_griddepcontrol: false,
                has_setmaxnreg: false,
                has_bulk_copy: false,
                has_sm120_features: false,
            },
            SmVersion::Sm90 | SmVersion::Sm90a => Self {
                has_tensor_cores: true,
                has_cp_async: true,
                has_ldmatrix: true,
                has_ampere_mma: true,
                has_wgmma: true,
                has_tma: true,
                has_fp8: true,
                has_fp6_fp4: false,
                has_dynamic_smem: true,
                has_named_barriers: true,
                has_cluster_barriers: true,
                has_stmatrix: true,
                has_redux: true,
                has_elect_one: true,
                has_griddepcontrol: true,
                has_setmaxnreg: true,
                has_bulk_copy: true,
                has_sm120_features: false,
            },
            SmVersion::Sm100 => Self {
                has_tensor_cores: true,
                has_cp_async: true,
                has_ldmatrix: true,
                has_ampere_mma: true,
                has_wgmma: true,
                has_tma: true,
                has_fp8: true,
                has_fp6_fp4: true,
                has_dynamic_smem: true,
                has_named_barriers: true,
                has_cluster_barriers: true,
                has_stmatrix: true,
                has_redux: true,
                has_elect_one: true,
                has_griddepcontrol: true,
                has_setmaxnreg: true,
                has_bulk_copy: true,
                has_sm120_features: false,
            },
            SmVersion::Sm120 => Self {
                has_tensor_cores: true,
                has_cp_async: true,
                has_ldmatrix: true,
                has_ampere_mma: true,
                has_wgmma: true,
                has_tma: true,
                has_fp8: true,
                has_fp6_fp4: true,
                has_dynamic_smem: true,
                has_named_barriers: true,
                has_cluster_barriers: true,
                has_stmatrix: true,
                has_redux: true,
                has_elect_one: true,
                has_griddepcontrol: true,
                has_setmaxnreg: true,
                has_bulk_copy: true,
                has_sm120_features: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_version_ordering() {
        assert!(SmVersion::Sm80 > SmVersion::Sm75);
        assert!(SmVersion::Sm90a > SmVersion::Sm90);
        assert!(SmVersion::Sm120 > SmVersion::Sm100);
    }

    #[test]
    fn ptx_version_strings() {
        assert_eq!(SmVersion::Sm75.ptx_version(), "6.4");
        assert_eq!(SmVersion::Sm80.ptx_version(), "7.0");
        assert_eq!(SmVersion::Sm86.ptx_version(), "7.1");
        assert_eq!(SmVersion::Sm90.ptx_version(), "8.0");
        assert_eq!(SmVersion::Sm100.ptx_version(), "8.5");
        assert_eq!(SmVersion::Sm120.ptx_version(), "8.7");
    }

    #[test]
    fn from_compute_capability_valid() {
        assert_eq!(
            SmVersion::from_compute_capability(7, 5),
            Some(SmVersion::Sm75)
        );
        assert_eq!(
            SmVersion::from_compute_capability(8, 0),
            Some(SmVersion::Sm80)
        );
        assert_eq!(
            SmVersion::from_compute_capability(9, 0),
            Some(SmVersion::Sm90)
        );
    }

    #[test]
    fn from_compute_capability_unknown() {
        assert_eq!(SmVersion::from_compute_capability(6, 0), None);
        assert_eq!(SmVersion::from_compute_capability(5, 2), None);
    }

    #[test]
    fn capabilities_turing() {
        let caps = SmVersion::Sm75.capabilities();
        assert!(caps.has_tensor_cores);
        assert!(!caps.has_cp_async);
        assert!(!caps.has_ampere_mma);
        assert!(!caps.has_wgmma);
    }

    #[test]
    fn capabilities_ampere() {
        let caps = SmVersion::Sm80.capabilities();
        assert!(caps.has_tensor_cores);
        assert!(caps.has_cp_async);
        assert!(caps.has_ampere_mma);
        assert!(!caps.has_wgmma);
        assert!(!caps.has_fp8);
    }

    #[test]
    fn capabilities_hopper() {
        let caps = SmVersion::Sm90a.capabilities();
        assert!(caps.has_wgmma);
        assert!(caps.has_tma);
        assert!(caps.has_fp8);
        assert!(!caps.has_fp6_fp4);
        assert!(caps.has_cluster_barriers);
    }

    #[test]
    fn capabilities_blackwell() {
        let caps = SmVersion::Sm100.capabilities();
        assert!(caps.has_fp6_fp4);
        assert!(caps.has_wgmma);
        assert!(caps.has_tma);
    }

    #[test]
    fn display_sm_version() {
        assert_eq!(format!("{}", SmVersion::Sm80), "sm_80");
        assert_eq!(format!("{}", SmVersion::Sm90a), "sm_90a");
    }

    #[test]
    fn shared_memory_limits() {
        assert_eq!(SmVersion::Sm75.max_shared_mem_per_block(), 65536);
        assert_eq!(SmVersion::Sm80.max_shared_mem_per_block(), 163_840);
        assert_eq!(SmVersion::Sm90.max_shared_mem_per_block(), 232_448);
    }

    #[test]
    fn ptx_isa_version_all_sm() {
        assert_eq!(SmVersion::Sm75.ptx_isa_version(), (6, 4));
        assert_eq!(SmVersion::Sm80.ptx_isa_version(), (7, 0));
        assert_eq!(SmVersion::Sm86.ptx_isa_version(), (7, 1));
        assert_eq!(SmVersion::Sm89.ptx_isa_version(), (7, 8));
        assert_eq!(SmVersion::Sm90.ptx_isa_version(), (8, 0));
        assert_eq!(SmVersion::Sm90a.ptx_isa_version(), (8, 0));
        assert_eq!(SmVersion::Sm100.ptx_isa_version(), (8, 5));
        assert_eq!(SmVersion::Sm120.ptx_isa_version(), (8, 7));
    }

    #[test]
    fn capabilities_new_fields_turing() {
        let caps = SmVersion::Sm75.capabilities();
        assert!(!caps.has_redux);
        assert!(!caps.has_stmatrix);
        assert!(!caps.has_elect_one);
        assert!(!caps.has_griddepcontrol);
        assert!(!caps.has_setmaxnreg);
        assert!(!caps.has_bulk_copy);
        assert!(!caps.has_sm120_features);
    }

    #[test]
    fn capabilities_new_fields_ampere() {
        let caps = SmVersion::Sm80.capabilities();
        assert!(caps.has_redux);
        assert!(!caps.has_stmatrix);
        assert!(!caps.has_elect_one);
        assert!(!caps.has_griddepcontrol);
        assert!(!caps.has_sm120_features);
    }

    #[test]
    fn capabilities_new_fields_hopper() {
        let caps = SmVersion::Sm90.capabilities();
        assert!(caps.has_redux);
        assert!(caps.has_stmatrix);
        assert!(caps.has_elect_one);
        assert!(caps.has_griddepcontrol);
        assert!(caps.has_setmaxnreg);
        assert!(caps.has_bulk_copy);
        assert!(!caps.has_sm120_features);
    }

    #[test]
    fn capabilities_new_fields_sm120() {
        let caps = SmVersion::Sm120.capabilities();
        assert!(caps.has_redux);
        assert!(caps.has_stmatrix);
        assert!(caps.has_elect_one);
        assert!(caps.has_griddepcontrol);
        assert!(caps.has_setmaxnreg);
        assert!(caps.has_bulk_copy);
        assert!(caps.has_sm120_features);
    }
}
