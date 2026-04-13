//! Convolution algorithm selection logic.
//!
//! Implements a decision tree that selects the optimal convolution algorithm
//! based on problem dimensions, filter size, data type, layout, and target
//! GPU architecture. The logic mirrors cuDNN's heuristic selection with
//! adjustments for OxiCUDA's kernel performance characteristics.
//!
//! # Decision tree
//!
//! 1. **1x1 kernels** with unit stride/dilation -> [`Direct`](ConvAlgorithm::Direct)
//!    (reduces to plain GEMM)
//! 2. **Depthwise convolutions** -> [`Direct`](ConvAlgorithm::Direct) (specialised kernel)
//! 3. **3x3 kernels** with unit stride/dilation and FP32 on large inputs
//!    -> [`Winograd`](ConvAlgorithm::Winograd) (2.25x multiplication reduction)
//! 4. **Large kernels** (7x7+) -> [`FftConv`](ConvAlgorithm::FftConv)
//! 5. **Ampere+ with NHWC** -> [`ImplicitGemm`](ConvAlgorithm::ImplicitGemm)
//! 6. **Default** -> [`Im2colGemm`](ConvAlgorithm::Im2colGemm)

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

use crate::types::ConvAlgorithm;

use super::descriptor::ConvProblem;

/// Minimum GEMM FLOP count for Winograd to be profitable.
///
/// Below this threshold the Winograd transform overhead dominates the
/// multiplication savings, so we fall back to implicit GEMM or im2col.
const WINOGRAD_FLOP_THRESHOLD: u64 = 1_000_000_000;

/// Minimum filter spatial size to consider FFT-based convolution.
const FFT_FILTER_MIN: u32 = 7;

/// Selects the best convolution algorithm for the given problem and SM version.
///
/// This is a heuristic selection — for maximum performance, the caller can
/// override the result or use the autotuner from `oxicuda-autotune` to
/// empirically benchmark all candidate algorithms.
#[must_use]
pub fn select_algorithm(problem: &ConvProblem, sm: SmVersion) -> ConvAlgorithm {
    // Rule 1: 1x1 convolutions reduce directly to GEMM.
    if problem.is_1x1() {
        return ConvAlgorithm::Direct;
    }

    // Rule 2: Depthwise convolutions need a specialised kernel.
    if problem.is_depthwise() {
        return ConvAlgorithm::Direct;
    }

    let r = problem.filter_dims.first().copied().unwrap_or(1);
    let s = problem.filter_dims.get(1).copied().unwrap_or(1);

    // Rule 3: 3x3 Winograd when conditions are met.
    if is_winograd_eligible(problem, r, s) {
        let flops = estimate_gemm_flops(problem, r, s);
        if flops > WINOGRAD_FLOP_THRESHOLD {
            return ConvAlgorithm::Winograd;
        }
    }

    // Rule 4: Large kernels benefit from FFT.
    if r >= FFT_FILTER_MIN && s >= FFT_FILTER_MIN {
        return ConvAlgorithm::FftConv;
    }

    // Rule 5: Ampere+ with NHWC layout -> implicit GEMM is best.
    if sm >= SmVersion::Sm80 && problem.layout.is_channels_last() {
        return ConvAlgorithm::ImplicitGemm;
    }

    // Rule 6: Default fallback — im2col + GEMM.
    ConvAlgorithm::Im2colGemm
}

/// Returns `true` if Winograd is applicable.
///
/// Winograd requires:
/// - 3x3 filter
/// - Unit stride and dilation
/// - FP32 precision (FP16 Winograd has excessive numerical error)
/// - Non-grouped convolution (or exact depthwise, handled above)
fn is_winograd_eligible(problem: &ConvProblem, r: u32, s: u32) -> bool {
    r == 3
        && s == 3
        && problem.stride.iter().all(|&v| v == 1)
        && problem.dilation.iter().all(|&v| v == 1)
        && problem.input_type == PtxType::F32
        && problem.groups == 1
}

/// Estimates the number of multiply-accumulate operations for a standard
/// conv GEMM approach (used to decide Winograd profitability).
fn estimate_gemm_flops(problem: &ConvProblem, r: u32, s: u32) -> u64 {
    let out_h = problem.output_h().unwrap_or(1);
    let out_w = problem.output_w().unwrap_or(1);
    2 * problem.batch as u64
        * problem.out_channels as u64
        * problem.in_channels as u64
        * out_h as u64
        * out_w as u64
        * r as u64
        * s as u64
}

/// Returns a list of candidate algorithms for autotuning, ordered by
/// expected performance (best first).
///
/// Unlike [`select_algorithm`] which returns a single heuristic pick,
/// this function returns all applicable algorithms so that the autotuner
/// can empirically benchmark them.
#[must_use]
pub fn candidate_algorithms(problem: &ConvProblem, sm: SmVersion) -> Vec<ConvAlgorithm> {
    let mut candidates = Vec::with_capacity(5);

    // Always include the heuristic winner first.
    let best = select_algorithm(problem, sm);
    candidates.push(best);

    // 1x1 and depthwise only make sense with Direct.
    if problem.is_1x1() || problem.is_depthwise() {
        return candidates;
    }

    // Add other applicable algorithms.
    if sm >= SmVersion::Sm80 {
        push_if_absent(&mut candidates, ConvAlgorithm::ImplicitGemm);
    }
    push_if_absent(&mut candidates, ConvAlgorithm::Im2colGemm);

    let r = problem.filter_dims.first().copied().unwrap_or(1);
    let s = problem.filter_dims.get(1).copied().unwrap_or(1);

    if is_winograd_eligible(problem, r, s) {
        push_if_absent(&mut candidates, ConvAlgorithm::Winograd);
    }
    if r >= FFT_FILTER_MIN && s >= FFT_FILTER_MIN {
        push_if_absent(&mut candidates, ConvAlgorithm::FftConv);
    }

    candidates
}

/// Pushes `algo` into `vec` only if it is not already present.
fn push_if_absent(vec: &mut Vec<ConvAlgorithm>, algo: ConvAlgorithm) {
    if !vec.contains(&algo) {
        vec.push(algo);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    fn problem_3x3_nchw() -> ConvProblem {
        ConvProblem {
            batch: 32,
            in_channels: 256,
            in_dims: vec![56, 56],
            out_channels: 256,
            filter_dims: vec![3, 3],
            padding: vec![1, 1],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }

    fn problem_1x1() -> ConvProblem {
        ConvProblem {
            batch: 1,
            in_channels: 64,
            in_dims: vec![32, 32],
            out_channels: 128,
            filter_dims: vec![1, 1],
            padding: vec![0, 0],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }

    fn problem_depthwise() -> ConvProblem {
        ConvProblem {
            batch: 1,
            in_channels: 64,
            in_dims: vec![32, 32],
            out_channels: 64,
            filter_dims: vec![3, 3],
            padding: vec![1, 1],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 64,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }

    #[test]
    fn select_1x1_direct() {
        let algo = select_algorithm(&problem_1x1(), SmVersion::Sm80);
        assert_eq!(algo, ConvAlgorithm::Direct);
    }

    #[test]
    fn select_depthwise_direct() {
        let algo = select_algorithm(&problem_depthwise(), SmVersion::Sm80);
        assert_eq!(algo, ConvAlgorithm::Direct);
    }

    #[test]
    fn select_3x3_large_winograd() {
        // Large enough for Winograd threshold
        let algo = select_algorithm(&problem_3x3_nchw(), SmVersion::Sm80);
        assert_eq!(algo, ConvAlgorithm::Winograd);
    }

    #[test]
    fn select_3x3_fp16_not_winograd() {
        let mut p = problem_3x3_nchw();
        p.input_type = PtxType::F16;
        let algo = select_algorithm(&p, SmVersion::Sm80);
        // FP16 should not select Winograd
        assert_ne!(algo, ConvAlgorithm::Winograd);
    }

    #[test]
    fn select_7x7_fft() {
        let mut p = problem_3x3_nchw();
        p.filter_dims = vec![7, 7];
        let algo = select_algorithm(&p, SmVersion::Sm80);
        assert_eq!(algo, ConvAlgorithm::FftConv);
    }

    #[test]
    fn select_nhwc_ampere_implicit_gemm() {
        let mut p = problem_3x3_nchw();
        p.layout = TensorLayout::Nhwc;
        p.batch = 1;
        p.in_channels = 4;
        p.out_channels = 4;
        p.in_dims = vec![8, 8]; // Small dims -> low FLOPs -> no Winograd
        let algo = select_algorithm(&p, SmVersion::Sm80);
        assert_eq!(algo, ConvAlgorithm::ImplicitGemm);
    }

    #[test]
    fn select_nchw_turing_im2col() {
        let mut p = problem_3x3_nchw();
        p.batch = 1;
        p.in_channels = 4;
        p.out_channels = 4;
        p.in_dims = vec![8, 8]; // Small dims -> low FLOPs -> no Winograd
        let algo = select_algorithm(&p, SmVersion::Sm75);
        assert_eq!(algo, ConvAlgorithm::Im2colGemm);
    }

    #[test]
    fn candidates_include_heuristic_first() {
        let p = problem_1x1();
        let cands = candidate_algorithms(&p, SmVersion::Sm80);
        assert_eq!(cands[0], ConvAlgorithm::Direct);
    }

    #[test]
    fn candidates_no_duplicates() {
        let p = problem_3x3_nchw();
        let cands = candidate_algorithms(&p, SmVersion::Sm80);
        let mut seen = std::collections::HashSet::new();
        for c in &cands {
            assert!(seen.insert(c), "duplicate algorithm: {c:?}");
        }
    }
}
