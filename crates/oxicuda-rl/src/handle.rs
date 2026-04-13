//! # RlHandle — RL Session Handle
//!
//! [`crate::handle::RlHandle`] is a lightweight context object that carries device information
//! needed by RL kernels (SM version for PTX header selection, random seed, and
//! an optional stream reference).

use crate::error::{RlError, RlResult};

// ─── SmVersion (local mirror of driver's u32 SM version) ────────────────────

/// GPU SM (Streaming Multiprocessor) version as a single integer.
///
/// Examples: 75 = SM 7.5 (Turing), 80 = SM 8.0 (Ampere), 90 = SM 9.0
/// (Hopper).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmVersion(pub u32);

impl SmVersion {
    /// Return the version number (e.g. 80 for Ampere).
    #[must_use]
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// PTX `.version` string for this SM.
    #[must_use]
    pub fn ptx_version_str(self) -> &'static str {
        match self.0 {
            v if v >= 100 => "8.7",
            v if v >= 90 => "8.4",
            v if v >= 80 => "8.0",
            _ => "7.5",
        }
    }
}

impl std::fmt::Display for SmVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sm_{}", self.0)
    }
}

// ─── RandomState ─────────────────────────────────────────────────────────────

/// Minimal LCG random state for CPU-side sampling in tests and buffers.
///
/// Uses the same multiplier/increment as `glibc`'s `rand()`:
/// `x_{n+1} = 1103515245 * x_n + 12345 (mod 2³¹)`.
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// Create a new LCG with the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance one step and return a `u32` in `[0, 2³¹)`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as u32
    }

    /// Return a `f32` in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    /// Return a `usize` in `[0, n)`.
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
}

// ─── RlHandle ────────────────────────────────────────────────────────────────

/// RL session handle: carries GPU metadata and a CPU-side RNG for buffer
/// operations.
#[derive(Debug, Clone)]
pub struct RlHandle {
    sm: SmVersion,
    rng: LcgRng,
    /// Device ordinal (0-indexed).
    device: u32,
}

impl RlHandle {
    /// Create a new handle for the given SM version and device.
    #[must_use]
    pub fn new(sm: u32, device: u32, seed: u64) -> Self {
        Self {
            sm: SmVersion(sm),
            rng: LcgRng::new(seed),
            device,
        }
    }

    /// Create a default handle (SM 8.0 / Ampere, device 0, seed 42).
    #[must_use]
    pub fn default_handle() -> Self {
        Self::new(80, 0, 42)
    }

    /// SM version.
    #[must_use]
    #[inline]
    pub fn sm(&self) -> SmVersion {
        self.sm
    }

    /// Device ordinal.
    #[must_use]
    #[inline]
    pub fn device(&self) -> u32 {
        self.device
    }

    /// Mutable access to the internal RNG (used by replay buffer sampling).
    #[inline]
    pub fn rng_mut(&mut self) -> &mut LcgRng {
        &mut self.rng
    }

    /// Validate that `batch_size > 0` and `batch_size <= capacity`.
    pub fn validate_batch(batch_size: usize, capacity: usize) -> RlResult<()> {
        if batch_size == 0 {
            return Err(RlError::InvalidHyperparameter {
                name: "batch_size".into(),
                msg: "must be > 0".into(),
            });
        }
        if batch_size > capacity {
            return Err(RlError::InsufficientTransitions {
                have: capacity,
                need: batch_size,
            });
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_different_values() {
        let mut rng = LcgRng::new(123);
        let v1 = rng.next_u32();
        let v2 = rng.next_u32();
        assert_ne!(v1, v2, "LCG should produce different values");
    }

    #[test]
    fn lcg_f32_in_range() {
        let mut rng = LcgRng::new(0);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "f32 out of [0,1): {v}");
        }
    }

    #[test]
    fn lcg_usize_in_range() {
        let mut rng = LcgRng::new(7);
        for _ in 0..1000 {
            let v = rng.next_usize(10);
            assert!(v < 10, "usize out of [0,10): {v}");
        }
    }

    #[test]
    fn sm_version_ordering() {
        assert!(SmVersion(80) > SmVersion(75));
        assert!(SmVersion(90) > SmVersion(80));
    }

    #[test]
    fn sm_version_ptx_str() {
        assert_eq!(SmVersion(75).ptx_version_str(), "7.5");
        assert_eq!(SmVersion(80).ptx_version_str(), "8.0");
        assert_eq!(SmVersion(90).ptx_version_str(), "8.4");
    }

    #[test]
    fn rl_handle_default() {
        let h = RlHandle::default_handle();
        assert_eq!(h.sm().as_u32(), 80);
        assert_eq!(h.device(), 0);
    }

    #[test]
    fn validate_batch_ok() {
        RlHandle::validate_batch(32, 1024).unwrap();
    }

    #[test]
    fn validate_batch_zero_error() {
        assert!(RlHandle::validate_batch(0, 100).is_err());
    }

    #[test]
    fn validate_batch_too_large_error() {
        assert!(RlHandle::validate_batch(200, 100).is_err());
    }
}
