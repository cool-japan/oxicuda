//! Apple Neural Engine (ANE) scheduler hints and integration.
//!
//! The Apple Neural Engine is a dedicated ML accelerator present on Apple
//! Silicon (A11 and later, M-series).  Unlike the GPU, ANE is accessed
//! indirectly through Apple's CoreML and Metal Performance Shaders Graph
//! frameworks — there is no public direct-dispatch API.
//!
//! This module provides:
//!
//! 1. **Detection** — heuristic detection of ANE availability from device /
//!    chip names.
//! 2. **Operation classification** — which operations are good ANE candidates
//!    versus GPU-preferred operations.
//! 3. **Dispatch hints** — `AneDispatchHint` tells the caller whether to route
//!    an operation to ANE (via CoreML) or keep it on the GPU Metal path.
//!
//! # Note
//!
//! ANE offload requires the `coremltools` / `coreml` Objective-C bridge to be
//! invoked at the application layer. This crate provides the *decision layer*
//! only; the actual ANE execution is outside the scope of `oxicuda-metal`.

use std::fmt;

// ─── AneGeneration ────────────────────────────────────────────────────────────

/// Apple Neural Engine generation, tied to chip family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AneGeneration {
    /// ANE not present (Intel Mac, simulator).
    None,
    /// First-generation ANE (A11–A12, limited ML ops).
    Gen1,
    /// Second-generation ANE (A13–A15, improved throughput).
    Gen2,
    /// A16 / A17 and M1 series (16 TOPS).
    Gen3,
    /// M2 / A15 Pro / A16 class (enhanced inter-op fusion).
    Gen4,
    /// M3 and later (hardware ray tracing + ML fusion).
    Gen5,
}

impl AneGeneration {
    /// Maximum INT8 throughput in TOPS (tera-operations per second).
    ///
    /// These are approximate published figures.
    pub fn tops(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Gen1 => 0.6,
            Self::Gen2 => 6.0,
            Self::Gen3 => 15.8,
            Self::Gen4 => 15.8,
            Self::Gen5 => 38.0,
        }
    }

    /// Return `true` if this generation supports INT8 matrix multiply.
    pub fn supports_int8_matmul(self) -> bool {
        self >= Self::Gen2
    }

    /// Return `true` if this generation supports FP16 operations.
    pub fn supports_fp16(self) -> bool {
        self >= Self::Gen1
    }
}

impl fmt::Display for AneGeneration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::None => "No ANE",
            Self::Gen1 => "ANE Gen1 (A11-A12)",
            Self::Gen2 => "ANE Gen2 (A13-A15)",
            Self::Gen3 => "ANE Gen3 (M1/A16)",
            Self::Gen4 => "ANE Gen4 (M2/A17)",
            Self::Gen5 => "ANE Gen5 (M3+)",
        };
        f.write_str(s)
    }
}

// ─── AneDetector ─────────────────────────────────────────────────────────────

/// Heuristic ANE capability detector based on chip/device name strings.
pub struct AneDetector;

impl AneDetector {
    /// Detect the ANE generation from a chip or device name string.
    ///
    /// Accepts strings like `"Apple M2"`, `"A17 Pro"`, `"arm64"`, etc.
    pub fn detect(chip_name: &str) -> AneGeneration {
        let name = chip_name.to_ascii_lowercase();

        // M3+ / A17 class
        if name.contains("m3") || name.contains("m4") || name.contains("a17") {
            return AneGeneration::Gen5;
        }
        // M2 / A16
        if name.contains("m2") || name.contains("a16") {
            return AneGeneration::Gen4;
        }
        // M1 / A15 / A14
        if name.contains("m1") || name.contains("a15") || name.contains("a14") {
            return AneGeneration::Gen3;
        }
        // A13
        if name.contains("a13") {
            return AneGeneration::Gen2;
        }
        // A11 / A12
        if name.contains("a11") || name.contains("a12") {
            return AneGeneration::Gen1;
        }
        // arm64 without version → assume at least Gen3 (modern Apple Silicon)
        if name.contains("apple") || name.contains("arm64") {
            return AneGeneration::Gen3;
        }

        AneGeneration::None
    }
}

// ─── AneOperation ────────────────────────────────────────────────────────────

/// Operations that can be routed to the ANE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AneOperation {
    /// Matrix multiplication (GEMM).
    MatMul,
    /// Convolution (2-D).
    Conv2d,
    /// Batch normalisation.
    BatchNorm,
    /// Multi-head attention.
    Attention,
    /// SoftMax.
    Softmax,
    /// Layer normalisation.
    LayerNorm,
    /// Rectified linear unit (element-wise).
    Relu,
    /// GELU activation.
    Gelu,
}

// ─── AneDispatchHint ─────────────────────────────────────────────────────────

/// Routing hint returned by [`AneScheduler::recommend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AneDispatchHint {
    /// Route through CoreML / MPS-Graph for potential ANE execution.
    PreferAne,
    /// Keep on the GPU Metal kernel path.
    PreferGpu,
    /// ANE not available; always use GPU.
    AneUnavailable,
}

impl fmt::Display for AneDispatchHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreferAne => f.write_str("prefer ANE"),
            Self::PreferGpu => f.write_str("prefer GPU"),
            Self::AneUnavailable => f.write_str("ANE unavailable"),
        }
    }
}

// ─── AneScheduler ────────────────────────────────────────────────────────────

/// Routes ML operations between the ANE and the Metal GPU path.
///
/// Instantiate with the chip name from `MTLDevice.name` or similar.
///
/// ```rust
/// use oxicuda_metal::ane::{AneScheduler, AneOperation, AneDispatchHint};
///
/// let scheduler = AneScheduler::new("Apple M2 Pro");
/// let hint = scheduler.recommend(AneOperation::MatMul, 512 * 512 * 4);
/// assert_eq!(hint, AneDispatchHint::PreferAne);
/// ```
#[derive(Debug)]
pub struct AneScheduler {
    /// Detected ANE generation.
    pub generation: AneGeneration,
    /// Minimum tensor element count for ANE offload to be worthwhile.
    /// Below this threshold the GPU path has lower launch overhead.
    pub min_elements_for_ane: usize,
}

impl AneScheduler {
    /// Create a scheduler for the given chip name.
    pub fn new(chip_name: &str) -> Self {
        let generation = AneDetector::detect(chip_name);
        // For Gen1 ANE the overhead is higher; use a larger threshold.
        let min_elements_for_ane = match generation {
            AneGeneration::None => usize::MAX,
            AneGeneration::Gen1 => 1024 * 1024,
            _ => 64 * 1024,
        };
        Self {
            generation,
            min_elements_for_ane,
        }
    }

    /// Recommend whether to route `op` to the ANE or GPU.
    ///
    /// `tensor_elements` is the total number of scalar elements in the primary
    /// input tensor (used to assess whether launch overhead is justified).
    pub fn recommend(&self, op: AneOperation, tensor_elements: usize) -> AneDispatchHint {
        if self.generation == AneGeneration::None {
            return AneDispatchHint::AneUnavailable;
        }
        if tensor_elements < self.min_elements_for_ane {
            return AneDispatchHint::PreferGpu;
        }
        // ANE excels at fixed-topology inference ops; dynamic shapes favour GPU.
        match op {
            AneOperation::MatMul
            | AneOperation::Conv2d
            | AneOperation::BatchNorm
            | AneOperation::Attention
            | AneOperation::Softmax
            | AneOperation::LayerNorm => AneDispatchHint::PreferAne,
            // Element-wise ops are fast on GPU and have low overhead.
            AneOperation::Relu | AneOperation::Gelu => AneDispatchHint::PreferGpu,
        }
    }

    /// Return `true` when the ANE is present and functional.
    pub fn ane_available(&self) -> bool {
        self.generation != AneGeneration::None
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_m3() {
        assert_eq!(AneDetector::detect("Apple M3 Max"), AneGeneration::Gen5);
    }

    #[test]
    fn detect_m2() {
        assert_eq!(AneDetector::detect("Apple M2 Pro"), AneGeneration::Gen4);
    }

    #[test]
    fn detect_m1() {
        assert_eq!(AneDetector::detect("Apple M1 Ultra"), AneGeneration::Gen3);
    }

    #[test]
    fn detect_a17() {
        assert_eq!(AneDetector::detect("A17 Pro"), AneGeneration::Gen5);
    }

    #[test]
    fn detect_a13() {
        assert_eq!(AneDetector::detect("A13 Bionic"), AneGeneration::Gen2);
    }

    #[test]
    fn detect_a11() {
        assert_eq!(AneDetector::detect("A11 Bionic"), AneGeneration::Gen1);
    }

    #[test]
    fn detect_intel_mac() {
        assert_eq!(
            AneDetector::detect("Intel HD Graphics 630"),
            AneGeneration::None
        );
    }

    #[test]
    fn detect_generic_arm64() {
        // arm64 without model → assume modern Gen3
        assert_eq!(AneDetector::detect("arm64"), AneGeneration::Gen3);
    }

    #[test]
    fn tops_ordering() {
        assert!(AneGeneration::Gen5.tops() > AneGeneration::Gen4.tops());
        assert!(AneGeneration::Gen3.tops() > AneGeneration::Gen1.tops());
        assert_eq!(AneGeneration::None.tops(), 0.0);
    }

    #[test]
    fn int8_support() {
        assert!(AneGeneration::Gen2.supports_int8_matmul());
        assert!(AneGeneration::Gen5.supports_int8_matmul());
        assert!(!AneGeneration::Gen1.supports_int8_matmul());
        assert!(!AneGeneration::None.supports_int8_matmul());
    }

    #[test]
    fn fp16_support() {
        assert!(AneGeneration::Gen1.supports_fp16());
        assert!(AneGeneration::Gen5.supports_fp16());
        assert!(!AneGeneration::None.supports_fp16());
    }

    #[test]
    fn scheduler_recommends_ane_for_large_matmul() {
        let sched = AneScheduler::new("Apple M2");
        let hint = sched.recommend(AneOperation::MatMul, 512 * 512);
        assert_eq!(hint, AneDispatchHint::PreferAne);
    }

    #[test]
    fn scheduler_prefers_gpu_for_small_tensor() {
        let sched = AneScheduler::new("Apple M2");
        // 100 elements is below the 64K threshold.
        let hint = sched.recommend(AneOperation::MatMul, 100);
        assert_eq!(hint, AneDispatchHint::PreferGpu);
    }

    #[test]
    fn scheduler_prefers_gpu_for_relu() {
        let sched = AneScheduler::new("Apple M2");
        let hint = sched.recommend(AneOperation::Relu, 1024 * 1024);
        assert_eq!(hint, AneDispatchHint::PreferGpu);
    }

    #[test]
    fn scheduler_unavailable_on_intel() {
        let sched = AneScheduler::new("Intel HD Graphics");
        assert!(!sched.ane_available());
        let hint = sched.recommend(AneOperation::Conv2d, usize::MAX);
        assert_eq!(hint, AneDispatchHint::AneUnavailable);
    }

    #[test]
    fn display_hints() {
        assert_eq!(AneDispatchHint::PreferAne.to_string(), "prefer ANE");
        assert_eq!(AneDispatchHint::PreferGpu.to_string(), "prefer GPU");
        assert_eq!(
            AneDispatchHint::AneUnavailable.to_string(),
            "ANE unavailable"
        );
    }

    #[test]
    fn display_generation() {
        assert!(AneGeneration::Gen3.to_string().contains("M1/A16"));
        assert!(AneGeneration::None.to_string().contains("No ANE"));
    }

    #[test]
    fn ane_ordering() {
        assert!(AneGeneration::Gen5 > AneGeneration::Gen1);
        assert!(AneGeneration::Gen1 < AneGeneration::Gen3);
    }
}
