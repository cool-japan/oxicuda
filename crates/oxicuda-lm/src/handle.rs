//! Session handle for `oxicuda-lm`.
//!
//! `LmHandle` stores the compute device index and the GPU SM version,
//! which is used by the PTX kernel generators to emit the correct
//! `.target` and `.address_size` directives.

// ─── SmVersion ───────────────────────────────────────────────────────────────

/// SM (Streaming Multiprocessor) version encoded as `major*10 + minor`.
///
/// Examples: 80 = SM 8.0 (Ampere A100), 90 = SM 9.0 (Hopper H100),
/// 120 = SM 12.0 (Blackwell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmVersion(pub u32);

impl SmVersion {
    /// PTX `.version` directive string for this SM.
    ///
    /// PTX ISA 8.7 covers SM 12.x, 8.4 covers SM 9.x,
    /// 8.0 covers SM 8.x, 7.5 covers SM 7.x.
    pub fn ptx_version_str(self) -> &'static str {
        if self.0 >= 100 {
            "8.7"
        } else if self.0 >= 90 {
            "8.4"
        } else if self.0 >= 80 {
            "8.0"
        } else {
            "7.5"
        }
    }

    /// PTX `.target` string for this SM (e.g., `"sm_80"`).
    pub fn target_str(self) -> String {
        format!("sm_{}", self.0)
    }
}

impl std::fmt::Display for SmVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SM {}.{}", self.0 / 10, self.0 % 10)
    }
}

// ─── LmHandle ────────────────────────────────────────────────────────────────

/// Lightweight session descriptor for LLM inference operations.
///
/// An `LmHandle` does **not** open a CUDA context; it merely records which
/// device and SM version are targeted so that PTX kernel generators can
/// emit architecture-appropriate code.
#[derive(Debug, Clone)]
pub struct LmHandle {
    /// GPU device ordinal (0-indexed).
    pub device: i32,
    /// GPU SM version.
    pub sm_version: SmVersion,
}

impl LmHandle {
    /// Create a new handle.
    pub fn new(device: i32, sm_version: SmVersion) -> Self {
        Self { device, sm_version }
    }

    /// Convenience constructor for unit-test / CPU environments (device 0, SM 8.0).
    pub fn default_handle() -> Self {
        Self::new(0, SmVersion(80))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_version_ptx_strings() {
        assert_eq!(SmVersion(75).ptx_version_str(), "7.5");
        assert_eq!(SmVersion(80).ptx_version_str(), "8.0");
        assert_eq!(SmVersion(86).ptx_version_str(), "8.0");
        assert_eq!(SmVersion(90).ptx_version_str(), "8.4");
        assert_eq!(SmVersion(100).ptx_version_str(), "8.7");
        assert_eq!(SmVersion(120).ptx_version_str(), "8.7");
    }

    #[test]
    fn sm_version_target_str() {
        assert_eq!(SmVersion(80).target_str(), "sm_80");
        assert_eq!(SmVersion(90).target_str(), "sm_90");
        assert_eq!(SmVersion(120).target_str(), "sm_120");
    }

    #[test]
    fn sm_version_display() {
        assert_eq!(SmVersion(80).to_string(), "SM 8.0");
        assert_eq!(SmVersion(90).to_string(), "SM 9.0");
    }

    #[test]
    fn sm_version_ordering() {
        assert!(SmVersion(80) < SmVersion(90));
        assert!(SmVersion(100) > SmVersion(90));
        assert_eq!(SmVersion(80), SmVersion(80));
    }

    #[test]
    fn lm_handle_default() {
        let h = LmHandle::default_handle();
        assert_eq!(h.device, 0);
        assert_eq!(h.sm_version, SmVersion(80));
    }

    #[test]
    fn lm_handle_custom() {
        let h = LmHandle::new(2, SmVersion(90));
        assert_eq!(h.device, 2);
        assert_eq!(h.sm_version, SmVersion(90));
    }
}
