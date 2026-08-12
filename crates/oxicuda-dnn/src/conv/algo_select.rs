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
//!    -> [`Winograd`](ConvAlgorithm::Winograd) (2.25x multiplication reduction),
//!    **gated behind [`winograd_forward_implemented`]**. `WinogradConv`'s
//!    forward kernels are currently load/launch-only skeletons that perform
//!    no numeric work (see that engine's module docs and the "Honesty
//!    contract" in `gpu_tests/conv_fprop.rs`), so this rule always falls
//!    through to rule 4/5/6 today rather than silently handing the caller
//!    an untouched output buffer.
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

/// Capability gate: `true` once [`WinogradConv`](super::fprop::winograd::WinogradConv)'s
/// forward kernels compute a real numeric result.
///
/// # Why this exists
///
/// `WinogradConv::execute` currently launches three stages and returns
/// `Ok(())`, but none of them do any real work:
///
/// - `generate_input_transform_ptx`'s body emits only `comment()` calls
///   narrating steps 1-4, then `ret` -- no load, transform, or store.
/// - `launch_winograd_gemm` is literally `let _ = handle; Ok(())` -- it
///   launches no kernel at all.
/// - `generate_output_transform_ptx` is the same comment-only skeleton --
///   the output pointer is never touched.
///
/// Net effect: any convolution routed to `WinogradConv` leaves the
/// caller's output buffer **completely untouched** (whatever device memory
/// happened to already be there) rather than merely computing a wrong
/// answer. This is intentionally documented, not hidden -- see the
/// "Honesty contract" and the `winograd_*_transform_*_launches` canary
/// tests in `gpu_tests/conv_fprop.rs`, which assert the untouched-buffer
/// behaviour as their PASS condition.
///
/// Before real F(2,3)/F(4,3) kernels existed, [`select_algorithm`]'s Rule 3
/// would route *any* eligible shape over `WINOGRAD_FLOP_THRESHOLD` FLOPs
/// (ordinary mid-size 3x3 CNN layers, not an exotic edge case) into this
/// silently-broken path. This gate keeps [`select_algorithm`] and
/// [`candidate_algorithms`] from ever returning
/// [`ConvAlgorithm::Winograd`] while it returns `false`, so callers fall
/// back to the numerically-verified [`Im2colGemm`](ConvAlgorithm::Im2colGemm)
/// / [`ImplicitGemm`](ConvAlgorithm::ImplicitGemm) engines instead.
///
/// # Flipping this on
///
/// Set this to `true` only once `WinogradConv`'s three launch stages have
/// real kernel bodies verified against the `conv2d_ref` CPU oracle in
/// `gpu_tests/conv_fprop.rs` (the same way `Im2colGemmConv` and
/// `ImplicitGemmConv` already are), and update that file's untouched-buffer
/// canary tests to real numeric-oracle assertions accordingly.
#[must_use]
#[inline]
pub const fn winograd_forward_implemented() -> bool {
    false
}

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

    // Rule 3: 3x3 Winograd when conditions are met -- gated behind
    // `winograd_forward_implemented()` until `WinogradConv` has real
    // kernels (see that function's docs). While the gate is closed this
    // always falls through to rule 4/5/6 rather than handing the caller a
    // silently-untouched output buffer.
    if winograd_forward_implemented() && is_winograd_eligible(problem, r, s) {
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
///
/// This is the *shape* eligibility test only -- it is independent of
/// [`winograd_forward_implemented`], which separately gates whether
/// `select_algorithm`/`candidate_algorithms` are actually allowed to act on
/// it. `pub(crate)` so `gpu_tests` can assert a regression shape is
/// genuinely eligible (not merely below the FLOP threshold).
pub(crate) fn is_winograd_eligible(problem: &ConvProblem, r: u32, s: u32) -> bool {
    r == 3
        && s == 3
        && problem.stride.iter().all(|&v| v == 1)
        && problem.dilation.iter().all(|&v| v == 1)
        && problem.input_type == PtxType::F32
        && problem.groups == 1
}

/// Estimates the number of multiply-accumulate operations for a standard
/// conv GEMM approach (used to decide Winograd profitability).
///
/// `pub(crate)` for the same reason as [`is_winograd_eligible`].
pub(crate) fn estimate_gemm_flops(problem: &ConvProblem, r: u32, s: u32) -> u64 {
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

    // Same gate as Rule 3 in `select_algorithm`: don't offer the autotuner
    // a "candidate" whose engine silently no-ops (it would look like the
    // fastest option by benchmarking as literally instantaneous).
    if winograd_forward_implemented() && is_winograd_eligible(problem, r, s) {
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
    fn winograd_forward_not_yet_implemented() {
        // Sanity-check the capability gate is still closed. If this starts
        // failing because someone flipped `winograd_forward_implemented`,
        // every assertion below (and the `conv_forward` regression test in
        // `gpu_tests::conv_fprop`) must be re-validated against the CPU
        // oracle for a real Winograd engine before shipping.
        assert!(!winograd_forward_implemented());
    }

    #[test]
    fn select_3x3_large_shape_is_winograd_eligible_but_gate_is_closed() {
        // `problem_3x3_nchw()` genuinely satisfies `is_winograd_eligible`
        // and clears `WINOGRAD_FLOP_THRESHOLD` -- i.e. this is exactly the
        // shape Rule 3 would route to the broken `WinogradConv` engine were
        // the gate not in place. Proves the fallback below is caused solely
        // by `winograd_forward_implemented() == false`, not by the shape
        // failing Winograd's own eligibility criteria.
        let p = problem_3x3_nchw();
        assert!(is_winograd_eligible(&p, 3, 3));
        assert!(estimate_gemm_flops(&p, 3, 3) > WINOGRAD_FLOP_THRESHOLD);
    }

    #[test]
    fn select_3x3_large_falls_back_while_winograd_unimplemented() {
        // Same eligible/over-threshold shape as the test above. NCHW layout
        // doesn't qualify for Rule 5 (ImplicitGemm requires channels-last),
        // so with Rule 3 gated closed this must land on the Rule 6 default:
        // Im2colGemm, whose `execute` path is verified correct end-to-end
        // against the CPU oracle in `gpu_tests::conv_fprop`.
        let algo = select_algorithm(&problem_3x3_nchw(), SmVersion::Sm80);
        assert_eq!(algo, ConvAlgorithm::Im2colGemm);
        assert_ne!(algo, ConvAlgorithm::Winograd);
    }

    #[test]
    fn candidates_exclude_unimplemented_winograd() {
        // The autotune candidate list must not offer Winograd either --
        // an autotuner benchmarking a no-op kernel would see it as
        // infinitely fast and always "win".
        let cands = candidate_algorithms(&problem_3x3_nchw(), SmVersion::Sm80);
        assert!(
            !cands.contains(&ConvAlgorithm::Winograd),
            "Winograd must not be offered as an autotune candidate while its \
             forward kernels are load/launch-only skeletons: {cands:?}"
        );
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
