//! Metal GPU family and feature-set capability tables.
//!
//! Apple exposes hardware capabilities through `MTLGPUFamily` (e.g. `Apple5`,
//! `Apple7`, `Mac2`).  At runtime the actual check is `[MTLDevice
//! supportsFamily:]`, but the *gating logic* — which features each family
//! unlocks — is pure data and is fully unit-testable on any host.
//!
//! This module mirrors the family hierarchy and answers questions such as
//! "does this GPU support `simdgroup_matrix`?", "what is the threadgroup
//! memory budget?", and "is dynamic caching available?".  The backend uses it
//! to decide whether to dispatch the tiled GEMM or the
//! [`crate::msl_nn::simdgroup_gemm_msl`] MMA kernel.

use std::fmt;

// ─── MetalGpuFamily ────────────────────────────────────────────────────────────

/// A Metal GPU family, mirroring `MTLGPUFamily`.
///
/// The `Apple*` families are ordered: a higher number is a strict superset of
/// the features of the lower numbers (for the Apple Silicon line).  `Mac2` is
/// the discrete/Intel-Mac family and is treated separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetalGpuFamily {
    /// `MTLGPUFamilyApple4` — A11/A12 class.
    Apple4,
    /// `MTLGPUFamilyApple5` — A13 class.
    Apple5,
    /// `MTLGPUFamilyApple6` — A14 / M1 class.
    Apple6,
    /// `MTLGPUFamilyApple7` — A15 / M1 (Metal 3 `simdgroup_matrix`).
    Apple7,
    /// `MTLGPUFamilyApple8` — M2 / A16 class.
    Apple8,
    /// `MTLGPUFamilyApple9` — M3 / A17 (Dynamic Caching, mesh shaders).
    Apple9,
    /// `MTLGPUFamilyMac2` — Intel Mac / discrete AMD GPUs (managed storage).
    Mac2,
}

impl MetalGpuFamily {
    /// `true` when this family supports `simdgroup_matrix` MMA instructions.
    ///
    /// Available from Apple GPU family 7 (Metal 3) onward.
    pub fn supports_simdgroup_matrix(self) -> bool {
        matches!(
            self,
            Self::Apple7 | Self::Apple8 | Self::Apple9 | Self::Mac2
        )
    }

    /// `true` when this family supports M3-era Dynamic Caching.
    pub fn supports_dynamic_caching(self) -> bool {
        self == Self::Apple9
    }

    /// `true` when this family supports compute mesh shaders.
    pub fn supports_mesh_shaders(self) -> bool {
        self == Self::Apple9
    }

    /// `true` when this family natively supports unified (zero-copy) memory.
    ///
    /// All Apple Silicon families do; `Mac2` (discrete) does not and requires
    /// the `Managed` storage mode with explicit synchronisation.
    pub fn supports_unified_memory(self) -> bool {
        !matches!(self, Self::Mac2)
    }

    /// `true` when the family supports tier-2 argument buffers (bindless).
    ///
    /// Apple GPU family 6+ and Mac2 expose argument-buffer tier 2.
    pub fn supports_argument_buffers_tier2(self) -> bool {
        match self {
            Self::Apple4 | Self::Apple5 => false,
            Self::Apple6 | Self::Apple7 | Self::Apple8 | Self::Apple9 | Self::Mac2 => true,
        }
    }

    /// Maximum threadgroup memory in bytes for this family.
    ///
    /// Apple Silicon exposes 32 KiB; the discrete Mac2 path conservatively
    /// reports the same documented minimum.
    pub fn max_threadgroup_memory(self) -> usize {
        match self {
            Self::Apple4 | Self::Apple5 => 16 * 1024,
            _ => 32 * 1024,
        }
    }

    /// Maximum number of threads per threadgroup for this family.
    pub fn max_threads_per_threadgroup(self) -> usize {
        match self {
            Self::Apple4 => 512,
            _ => 1024,
        }
    }

    /// SIMD-group (warp) width.  Apple GPUs use 32-wide SIMD groups.
    pub fn simd_width(self) -> usize {
        32
    }

    /// The numeric generation in the family's Metal name: `Apple7` → `7`,
    /// `Mac2` → `2`.
    ///
    /// Metal has no CUDA-style *compute capability*, so this is what
    /// `MetalBackend::available_devices` reports in
    /// [`oxicuda_backend::DeviceInfo::compute_capability`]'s major slot — a real
    /// driver-probed number rather than an invented version. The minor slot is
    /// always `0`; the two families are not comparable across the `Apple`/`Mac`
    /// split, so do not order devices by this value alone.
    pub fn generation(self) -> u32 {
        match self {
            Self::Apple4 => 4,
            Self::Apple5 => 5,
            Self::Apple6 => 6,
            Self::Apple7 => 7,
            Self::Apple8 => 8,
            Self::Apple9 => 9,
            Self::Mac2 => 2,
        }
    }
}

impl fmt::Display for MetalGpuFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Apple4 => "Apple4 (A11-A12)",
            Self::Apple5 => "Apple5 (A13)",
            Self::Apple6 => "Apple6 (A14/M1)",
            Self::Apple7 => "Apple7 (A15/M1, Metal 3)",
            Self::Apple8 => "Apple8 (M2/A16)",
            Self::Apple9 => "Apple9 (M3/A17)",
            Self::Mac2 => "Mac2 (Intel/discrete)",
        };
        f.write_str(s)
    }
}

// ─── MetalDeviceCapabilities ───────────────────────────────────────────────────

/// A resolved capability snapshot for a device.
///
/// Built either from a [`MetalGpuFamily`] directly or heuristically from a
/// device-name string via [`MetalDeviceCapabilities::from_device_name`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalDeviceCapabilities {
    /// The detected GPU family.
    pub family: MetalGpuFamily,
    /// Whether `simdgroup_matrix` GEMM may be dispatched.
    pub simdgroup_matrix: bool,
    /// Whether unified (shared) memory is available.
    pub unified_memory: bool,
    /// Threadgroup memory budget in bytes.
    pub threadgroup_memory: usize,
    /// Maximum threads per threadgroup.
    pub max_threads_per_threadgroup: usize,
}

impl MetalDeviceCapabilities {
    /// Construct capabilities from a known GPU family.
    pub fn from_family(family: MetalGpuFamily) -> Self {
        Self {
            family,
            simdgroup_matrix: family.supports_simdgroup_matrix(),
            unified_memory: family.supports_unified_memory(),
            threadgroup_memory: family.max_threadgroup_memory(),
            max_threads_per_threadgroup: family.max_threads_per_threadgroup(),
        }
    }

    /// Heuristically detect the GPU family from a device-name string.
    ///
    /// Accepts strings like `"Apple M3 Max"`, `"Apple A15"`, or `"AMD Radeon
    /// Pro 580"`.  Unknown Apple devices default to `Apple7` (the first
    /// `simdgroup_matrix`-capable family); unknown non-Apple devices map to
    /// `Mac2`.
    pub fn from_device_name(name: &str) -> Self {
        let lname = name.to_ascii_lowercase();
        let family = if lname.contains("m3") || lname.contains("m4") || lname.contains("a17") {
            MetalGpuFamily::Apple9
        } else if lname.contains("m2") || lname.contains("a16") {
            MetalGpuFamily::Apple8
        } else if lname.contains("m1") || lname.contains("a15") || lname.contains("a14") {
            MetalGpuFamily::Apple7
        } else if lname.contains("a13") {
            MetalGpuFamily::Apple5
        } else if lname.contains("a11") || lname.contains("a12") {
            MetalGpuFamily::Apple4
        } else if lname.contains("amd")
            || lname.contains("radeon")
            || lname.contains("intel")
            || lname.contains("vega")
            || lname.contains("navi")
        {
            MetalGpuFamily::Mac2
        } else if lname.contains("apple") {
            // Unknown Apple device — assume modern Metal-3 family.
            MetalGpuFamily::Apple7
        } else {
            MetalGpuFamily::Mac2
        };
        Self::from_family(family)
    }

    /// Recommend whether the `simdgroup_matrix` GEMM should be used for the
    /// given problem size, accounting for hardware support and tile alignment.
    ///
    /// The MMA kernel works on `8×8` tiles, so it is only beneficial when all
    /// three dimensions are multiples of 8 (otherwise the tiled scalar kernel
    /// avoids ragged-edge handling) and the family actually supports it.
    pub fn prefer_simdgroup_gemm(&self, m: usize, n: usize, k: usize) -> bool {
        self.simdgroup_matrix
            && m % 8 == 0
            && n % 8 == 0
            && k % 8 == 0
            && m >= 8
            && n >= 8
            && k >= 8
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simdgroup_matrix_gated_on_apple7_plus() {
        assert!(!MetalGpuFamily::Apple4.supports_simdgroup_matrix());
        assert!(!MetalGpuFamily::Apple6.supports_simdgroup_matrix());
        assert!(MetalGpuFamily::Apple7.supports_simdgroup_matrix());
        assert!(MetalGpuFamily::Apple9.supports_simdgroup_matrix());
        assert!(MetalGpuFamily::Mac2.supports_simdgroup_matrix());
    }

    #[test]
    fn dynamic_caching_only_apple9() {
        assert!(MetalGpuFamily::Apple9.supports_dynamic_caching());
        assert!(!MetalGpuFamily::Apple8.supports_dynamic_caching());
        assert!(MetalGpuFamily::Apple9.supports_mesh_shaders());
        assert!(!MetalGpuFamily::Apple7.supports_mesh_shaders());
    }

    #[test]
    fn unified_memory_for_apple_only() {
        assert!(MetalGpuFamily::Apple6.supports_unified_memory());
        assert!(MetalGpuFamily::Apple9.supports_unified_memory());
        assert!(!MetalGpuFamily::Mac2.supports_unified_memory());
    }

    #[test]
    fn argument_buffer_tier2_gating() {
        assert!(!MetalGpuFamily::Apple4.supports_argument_buffers_tier2());
        assert!(!MetalGpuFamily::Apple5.supports_argument_buffers_tier2());
        assert!(MetalGpuFamily::Apple6.supports_argument_buffers_tier2());
        assert!(MetalGpuFamily::Mac2.supports_argument_buffers_tier2());
    }

    #[test]
    fn threadgroup_memory_budget() {
        assert_eq!(MetalGpuFamily::Apple5.max_threadgroup_memory(), 16 * 1024);
        assert_eq!(MetalGpuFamily::Apple7.max_threadgroup_memory(), 32 * 1024);
        assert_eq!(MetalGpuFamily::Apple9.max_threadgroup_memory(), 32 * 1024);
    }

    #[test]
    fn thread_and_simd_limits() {
        assert_eq!(MetalGpuFamily::Apple4.max_threads_per_threadgroup(), 512);
        assert_eq!(MetalGpuFamily::Apple9.max_threads_per_threadgroup(), 1024);
        assert_eq!(MetalGpuFamily::Apple9.simd_width(), 32);
    }

    #[test]
    fn family_ordering_is_monotone() {
        assert!(MetalGpuFamily::Apple9 > MetalGpuFamily::Apple7);
        assert!(MetalGpuFamily::Apple4 < MetalGpuFamily::Apple6);
    }

    #[test]
    fn capabilities_from_family() {
        let caps = MetalDeviceCapabilities::from_family(MetalGpuFamily::Apple8);
        assert_eq!(caps.family, MetalGpuFamily::Apple8);
        assert!(caps.simdgroup_matrix);
        assert!(caps.unified_memory);
        assert_eq!(caps.threadgroup_memory, 32 * 1024);
    }

    #[test]
    fn detect_from_device_name() {
        assert_eq!(
            MetalDeviceCapabilities::from_device_name("Apple M3 Max").family,
            MetalGpuFamily::Apple9
        );
        assert_eq!(
            MetalDeviceCapabilities::from_device_name("Apple M2 Pro").family,
            MetalGpuFamily::Apple8
        );
        assert_eq!(
            MetalDeviceCapabilities::from_device_name("Apple M1").family,
            MetalGpuFamily::Apple7
        );
        assert_eq!(
            MetalDeviceCapabilities::from_device_name("AMD Radeon Pro 580").family,
            MetalGpuFamily::Mac2
        );
        assert_eq!(
            MetalDeviceCapabilities::from_device_name("Intel Iris Xe").family,
            MetalGpuFamily::Mac2
        );
    }

    #[test]
    fn unknown_apple_defaults_to_apple7() {
        let caps = MetalDeviceCapabilities::from_device_name("Apple Future GPU");
        assert_eq!(caps.family, MetalGpuFamily::Apple7);
        assert!(caps.simdgroup_matrix);
    }

    #[test]
    fn prefer_simdgroup_gemm_alignment() {
        let m3 = MetalDeviceCapabilities::from_family(MetalGpuFamily::Apple9);
        // Aligned and supported → prefer MMA kernel.
        assert!(m3.prefer_simdgroup_gemm(64, 64, 64));
        // Ragged dimension → fall back to tiled scalar GEMM.
        assert!(!m3.prefer_simdgroup_gemm(63, 64, 64));
        // Too small → not worthwhile.
        assert!(!m3.prefer_simdgroup_gemm(4, 4, 4));

        // Older family never prefers MMA even if aligned.
        let old = MetalDeviceCapabilities::from_family(MetalGpuFamily::Apple6);
        assert!(!old.prefer_simdgroup_gemm(64, 64, 64));
    }

    #[test]
    fn display_family() {
        assert!(MetalGpuFamily::Apple9.to_string().contains("M3"));
        assert!(MetalGpuFamily::Mac2.to_string().contains("discrete"));
    }
}
