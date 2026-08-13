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
//! 3. **Anything the CTA-tiled implicit-GEMM kernel claims**
//!    ([`TiledConvPlan::for_problem`]) -> [`ImplicitGemm`](ConvAlgorithm::ImplicitGemm).
//!    Measured 5.7-8.0 TFLOPS against Winograd's 1.7-3.5 on the same shapes,
//!    which is why this rule precedes the Winograd one.
//! 4. **3x3 NCHW FP32 kernels** with unit stride/dilation, `groups == 1` and
//!    padding <= 1, above [`WINOGRAD_FLOP_THRESHOLD`], *that the tiling
//!    declined* -> [`Winograd`](ConvAlgorithm::Winograd). Eligibility is
//!    [`WinogradConv::supports`] verbatim (via [`is_winograd_eligible`]), so
//!    this rule can never select an engine that would then refuse the
//!    problem; profitability is the separate FLOP test, calibrated from the
//!    measurements quoted on [`WINOGRAD_FLOP_THRESHOLD`].
//! 4. **Large kernels** (7x7+) -> [`FftConv`](ConvAlgorithm::FftConv)
//! 5. **Ampere+ with NHWC** -> [`ImplicitGemm`](ConvAlgorithm::ImplicitGemm)
//! 6. **Default** -> [`Im2colGemm`](ConvAlgorithm::Im2colGemm)

use oxicuda_ptx::arch::SmVersion;

use crate::types::ConvAlgorithm;

use super::descriptor::ConvProblem;
use super::fprop::tiled_implicit_gemm::TiledConvPlan;

/// Minimum [`estimate_gemm_flops`] count for Winograd to be profitable.
///
/// # Calibration
///
/// Measured on an RTX A4000 (sm_86, driver 550.144.03) with
/// `benches/winograd_vs_implicit_gemm.rs` — `WinogradConv` against
/// `ImplicitGemmConv`, same buffers, kernel cache warm, one
/// `stream().synchronize()` per sample. Median times:
///
/// ```text
///   est. FLOPs   shape           winograd   implicit   speedup
///    7.373e+04   c8_8x8            12.1 us     7.9 us     0.65x
///    1.180e+06   c16_16x16         12.2 us    10.1 us     0.83x
///    4.719e+06   c16_32x32         13.1 us    11.7 us     0.89x
///    1.887e+07   c32_32x32         17.2 us    29.1 us     1.69x
///    1.887e+07   c64_16x16         18.1 us    29.4 us     1.62x
///    7.550e+07   c256_8x8          55.0 us   101.8 us     1.85x
///    3.020e+08   c512_8x8         176.0 us   434.5 us     2.47x
///    3.020e+08   c64_64x64        127.6 us   355.5 us     2.79x
///    1.208e+09   c512_16x16       450.6 us  1527.8 us     3.39x
///    4.832e+09   c128_128x128    1542.9 us  5959.9 us     3.86x
///    4.832e+09   c256_64x64      1482.5 us  5489.6 us     3.70x
///    4.832e+09   c512_32x32      1513.8 us  5591.4 us     3.69x
///    4.832e+09   c64_256x256     2019.4 us  6900.9 us     3.42x
/// ```
///
/// The GPU is shared, so these are medians from a warm device; the sub-30 us
/// rows — the ones that actually set the threshold — were re-measured across
/// three separate runs and agree to within 2%.
///
/// Below ~5e6 FLOPs every configuration is launch-bound — Winograd issues
/// four kernels where implicit GEMM issues one, and no amount of saved
/// multiplies pays for three extra launches. Above ~1.9e7 Winograd wins by
/// 1.5x or more, and the margin grows monotonically with problem size. The
/// threshold sits between the two measured brackets, slightly above the
/// interpolated break-even (~8e6) so the un-measured band is conservative.
///
/// Note that the two 1.887e7-FLOP points trade channels against spatial
/// extent (`c32_32x32` vs `c64_16x16`) and land within 10% of each other,
/// which is what justifies expressing the threshold in FLOPs at all rather
/// than in any single dimension.
const WINOGRAD_FLOP_THRESHOLD: u64 = 10_000_000;

/// Minimum filter spatial size to consider FFT-based convolution.
const FFT_FILTER_MIN: u32 = 7;

/// Capability gate: whether [`select_algorithm`] and [`candidate_algorithms`]
/// may return [`ConvAlgorithm::Winograd`].
///
/// # History
///
/// This started life as a *correctness* gate. `WinogradConv`'s three stages
/// used to be comment-only PTX skeletons and a `launch_winograd_gemm` that was
/// literally `let _ = handle; Ok(())`, so a convolution routed there returned
/// `Ok(())` with the output buffer **completely untouched** — not wrong, never
/// written. The gate existed to keep ordinary mid-size 3x3 layers out of that
/// path.
///
/// # Why it is now `true`
///
/// [`WinogradConv`](super::fprop::winograd::WinogradConv) is a real
/// F(2x2,3x3) implementation: input transform, filter transform, a
/// shared-memory-tiled batched GEMM over the 16 transform-domain positions,
/// and an output transform with bias. Two independent on-device validations
/// gate this flag (`gpu_tests::conv_winograd`, RTX A4000 / sm_86):
///
/// * against an `f64` CPU oracle — relative L2 error `1.0e-7` to `1.7e-7`
///   across 15 shapes covering partial tiles, odd extents, `C == 1`, `K == 1`,
///   multi-batch and both supported paddings;
/// * against `ImplicitGemmConv` on device — relative L2 `7.9e-8` to `1.4e-6`,
///   including all four InSwapper-128-scale layers.
///
/// Both are three to four orders of magnitude inside the documented `1e-4`
/// budget. Performance is measured, not assumed: see
/// [`WINOGRAD_FLOP_THRESHOLD`] for the full table, which is also what defines
/// the problem-size region this rule applies to (Winograd is 1.5x to 3.8x
/// faster above the threshold and *slower* below it, so the flag alone is not
/// the whole decision).
///
/// # If you need to turn it back off
///
/// Setting this to `false` restores the previous behaviour exactly — every
/// eligible shape falls through to the `ImplicitGemm` / `Im2colGemm` engines,
/// which remain fully verified. Nothing else in the crate needs changing; the
/// dispatch regression test in `gpu_tests::conv_fprop` asserts a correct
/// numeric result either way.
#[must_use]
#[inline]
pub const fn winograd_forward_implemented() -> bool {
    true
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

    // Rule 3: the CTA-tiled implicit-GEMM kernel, wherever it claims the
    // shape. This rule sits *above* Winograd because that is what the
    // measurement says: on this workspace's RTX A4000, over the five real
    // face-pipeline 3x3 shapes, the tiled kernel runs at 5.7-8.0 TFLOPS while
    // Winograd -- whose transform-domain GEMM is itself an untiled 16x16
    // kernel -- runs at 1.7-3.5 TFLOPS on the same shapes:
    //
    // ```text
    //   shape                                  scalar    tiled  winograd
    //   SCRFD    28->56  3x3 pad1 @320x320       905     6317      1720
    //   ArcFace  64->64  3x3 pad1 @112x112       938     5702      2503
    //   InSwap 1024->1024 3x3 pad0 @34x34        941     6124      3530
    //   InSwap 1024->512  3x3 pad1 @64x64        931     7976      3498
    //   InSwap  512->256  3x3 pad1 @128x128      886     7856      3468
    //                                        (GFLOPS, benches/conv_engine_gflops_regression)
    // ```
    //
    // Winograd's 2.25x multiply reduction cannot make up a 2x deficit in the
    // GEMM it reduces *to*, so it stays a fallback for the shapes the tiling
    // declines (`groups > 1`, f64, NHWC, or too small) -- where it is still
    // 1.5-3.9x ahead of the scalar kernel and therefore still worth selecting.
    if TiledConvPlan::for_problem(problem).is_some() {
        return ConvAlgorithm::ImplicitGemm;
    }

    // Rule 4: 3x3 Winograd when the engine supports the shape and the problem
    // is large enough for the transform overhead to pay for itself. Both
    // conditions are load-bearing: `is_winograd_eligible` mirrors the engine's
    // own `supports`, and the FLOP threshold is calibrated from measurements
    // (see its docs) -- below it Winograd is measurably *slower*, because it
    // issues four kernels where implicit GEMM issues one.
    if winograd_forward_implemented() && is_winograd_eligible(problem, r, s) {
        let flops = estimate_gemm_flops(problem, r, s);
        if flops > WINOGRAD_FLOP_THRESHOLD {
            return ConvAlgorithm::Winograd;
        }
    }

    // Rule 5: Large kernels benefit from FFT.
    if r >= FFT_FILTER_MIN && s >= FFT_FILTER_MIN {
        return ConvAlgorithm::FftConv;
    }

    // Rule 6: Ampere+ with NHWC layout -> implicit GEMM is best.
    if sm >= SmVersion::Sm80 && problem.layout.is_channels_last() {
        return ConvAlgorithm::ImplicitGemm;
    }

    // Rule 7: Default fallback — im2col + GEMM.
    ConvAlgorithm::Im2colGemm
}

/// Returns `true` if the Winograd engine can compute `problem`.
///
/// Delegates to [`WinogradConv::supports`](super::fprop::winograd::WinogradConv::supports)
/// rather than restating the conditions, because the two must agree *exactly*:
/// a selector that is more permissive than the engine turns a working
/// convolution into a hard error at dispatch time (the engine refuses the
/// problem the selector just handed it), and one that is less permissive
/// silently leaves performance on the table. `r`/`s` are the caller's
/// already-extracted filter extent and are checked for consistency with the
/// problem so a stale pair cannot widen the test.
///
/// This is the *shape* eligibility test only — profitability is
/// [`WINOGRAD_FLOP_THRESHOLD`], and whether the rule is consulted at all is
/// [`winograd_forward_implemented`]. `pub(crate)` so `gpu_tests` can assert a
/// regression shape is genuinely eligible (not merely below the FLOP
/// threshold).
pub(crate) fn is_winograd_eligible(problem: &ConvProblem, r: u32, s: u32) -> bool {
    r == 3 && s == 3 && super::fprop::winograd::WinogradConv::supports(problem)
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

    // Same eligibility test as Rule 3 in `select_algorithm`, but deliberately
    // *without* its FLOP threshold: the threshold is a heuristic about where
    // Winograd usually wins, and an autotuner exists precisely to measure that
    // for the shape in hand rather than trust the heuristic. The capability
    // gate still applies -- an engine that cannot compute the problem must
    // never be offered, since the autotuner would time its immediate error
    // return as an infinitely fast kernel.
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
    use oxicuda_ptx::ir::PtxType;

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

    /// The selector's eligibility test and the engine's own `supports` must be
    /// the same predicate. If the selector is the wider of the two, Rule 3
    /// hands `conv_forward` a problem `WinogradConv::new` then refuses — a
    /// working convolution turned into a hard error.
    /// A named mutation of the baseline problem, for table-driven tests.
    type ProblemMutation = (&'static str, fn(&mut ConvProblem));

    #[test]
    fn eligibility_matches_the_engine_exactly() {
        use crate::conv::fprop::winograd::WinogradConv;
        let mutations: [ProblemMutation; 10] = [
            ("baseline", |_| {}),
            ("5x5 filter", |p| p.filter_dims = vec![5, 5]),
            ("stride 2", |p| p.stride = vec![2, 2]),
            ("dilation 2", |p| p.dilation = vec![2, 2]),
            ("groups 2", |p| p.groups = 2),
            ("padding 2", |p| p.padding = vec![2, 2]),
            ("padding 0", |p| p.padding = vec![0, 0]),
            ("NHWC", |p| p.layout = TensorLayout::Nhwc),
            ("f16", |p| p.input_type = PtxType::F16),
            ("3-D", |p| p.in_dims = vec![4, 8, 8]),
        ];
        for (label, mutate) in mutations {
            let mut p = problem_3x3_nchw();
            mutate(&mut p);
            let r = p.filter_dims.first().copied().unwrap_or(1);
            let s = p.filter_dims.get(1).copied().unwrap_or(1);
            assert_eq!(
                is_winograd_eligible(&p, r, s),
                WinogradConv::supports(&p),
                "{label}: selector and engine disagree about eligibility"
            );
        }
    }

    /// A large 3x3 NCHW f32 convolution satisfies *both* the Winograd rule and
    /// the tiled-implicit-GEMM rule; the tiled kernel must win, because it is
    /// 1.7-2.3x faster than Winograd on every measured shape (see the table on
    /// Rule 3).
    #[test]
    fn select_3x3_large_nchw_prefers_the_tiled_kernel_over_winograd() {
        let p = problem_3x3_nchw();
        assert!(
            is_winograd_eligible(&p, 3, 3)
                && estimate_gemm_flops(&p, 3, 3) > WINOGRAD_FLOP_THRESHOLD,
            "this shape must be Winograd-eligible, so the tiled rule is what decides it"
        );
        assert!(
            TiledConvPlan::for_problem(&p).is_some(),
            "...and claimed by the tiling"
        );
        assert_eq!(
            select_algorithm(&p, SmVersion::Sm80),
            ConvAlgorithm::ImplicitGemm
        );
    }

    /// ...and Winograd is still selected for a Winograd-eligible shape the
    /// tiling declines, which is the only reason Rule 4 still exists. A
    /// grouped convolution is the cleanest such shape: `WinogradConv::supports`
    /// requires `groups == 1`, so the decline has to come from somewhere the
    /// two rules disagree -- here, an f32 NCHW 3x3 whose GEMM depth is below
    /// the tiling's floor but whose FLOP count clears Winograd's.
    #[test]
    fn select_3x3_falls_back_to_winograd_where_the_tiling_declines() {
        let mut p = problem_3x3_nchw();
        p.in_channels = 4; // C*R*S = 36, under the tiling's MIN_GEMM_K of 64.
        assert!(
            TiledConvPlan::for_problem(&p).is_none(),
            "the tiling must decline this shape"
        );
        assert!(is_winograd_eligible(&p, 3, 3));
        assert!(estimate_gemm_flops(&p, 3, 3) > WINOGRAD_FLOP_THRESHOLD);
        assert_eq!(
            select_algorithm(&p, SmVersion::Sm80),
            ConvAlgorithm::Winograd
        );
    }

    /// Below the measured break-even Winograd is *slower*, so an eligible but
    /// small shape must not be routed there however capable the engine is.
    #[test]
    fn select_small_3x3_stays_off_winograd() {
        let mut p = problem_3x3_nchw();
        p.batch = 1;
        p.in_channels = 16;
        p.out_channels = 16;
        p.in_dims = vec![16, 16];
        assert!(
            is_winograd_eligible(&p, 3, 3),
            "shape must be eligible, so the FLOP threshold is what excludes it"
        );
        assert!(estimate_gemm_flops(&p, 3, 3) < WINOGRAD_FLOP_THRESHOLD);
        assert_ne!(
            select_algorithm(&p, SmVersion::Sm80),
            ConvAlgorithm::Winograd
        );
    }

    /// A shape the engine cannot compute must never be offered to the
    /// autotuner: it would time the engine's immediate error return as an
    /// infinitely fast kernel and always "win".
    #[test]
    fn candidates_exclude_unsupported_winograd_shapes() {
        let mut p = problem_3x3_nchw();
        p.padding = vec![2, 2];
        let cands = candidate_algorithms(&p, SmVersion::Sm80);
        assert!(
            !cands.contains(&ConvAlgorithm::Winograd),
            "padding 2 is outside the engine's supported region: {cands:?}"
        );
    }

    /// The candidate list deliberately ignores the FLOP threshold — measuring
    /// is the autotuner's whole job — but still respects the capability gate.
    #[test]
    fn candidates_offer_winograd_below_the_flop_threshold() {
        let mut p = problem_3x3_nchw();
        p.batch = 1;
        p.in_channels = 16;
        p.out_channels = 16;
        p.in_dims = vec![16, 16];
        assert!(estimate_gemm_flops(&p, 3, 3) < WINOGRAD_FLOP_THRESHOLD);
        let cands = candidate_algorithms(&p, SmVersion::Sm80);
        assert_eq!(
            cands.contains(&ConvAlgorithm::Winograd),
            winograd_forward_implemented()
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

    /// The FFT rule now sits below the tiled rule, and that ordering is a
    /// deliberate improvement rather than an accident of rule numbering: the
    /// `FftConv` arm of `conv::api::conv_forward` does not actually run an
    /// FFT convolution -- it validates an `FftConv2dPlan`, then executes
    /// `Im2colGemmConv` -- *and* demands a workspace, returning
    /// `WorkspaceRequired` without one. Routing a 7x7 the tiling can claim to
    /// the tiled kernel replaces an im2col materialisation plus a BLAS GEMM
    /// with a single zero-workspace kernel.
    #[test]
    fn select_7x7_prefers_the_tiled_kernel_over_the_fft_placeholder() {
        let mut p = problem_3x3_nchw();
        p.filter_dims = vec![7, 7];
        p.padding = vec![3, 3];
        assert!(TiledConvPlan::for_problem(&p).is_some());
        assert_eq!(
            select_algorithm(&p, SmVersion::Sm80),
            ConvAlgorithm::ImplicitGemm
        );
    }

    /// ...and a 7x7 the tiling declines still reaches the FFT rule.
    #[test]
    fn select_7x7_fft_when_the_tiling_declines() {
        let mut p = problem_3x3_nchw();
        p.filter_dims = vec![7, 7];
        p.in_channels = 1; // C*R*S = 49, under the tiling's floor.
        assert!(TiledConvPlan::for_problem(&p).is_none());
        assert_eq!(
            select_algorithm(&p, SmVersion::Sm80),
            ConvAlgorithm::FftConv
        );
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
