//! # Top-K Sampling
//!
//! Restricts sampling to the K most probable tokens, setting all other logit
//! values to −∞ before applying softmax and sampling.
//!
//! ## Algorithm
//!
//! 1. Compute the K-th largest value (partial sort / selection).
//! 2. Set all logits strictly below the threshold to `f32::NEG_INFINITY`.
//! 3. Compute softmax probabilities over the remaining logits.
//! 4. Sample from the resulting categorical distribution.
//!
//! ## Notes
//!
//! * Ties at the threshold are handled by counting: we keep exactly K tokens,
//!   preferring lower indices among equals.
//! * K = 1 is equivalent to greedy decoding.
//! * K ≥ vocab_size disables top-K filtering.

use crate::error::{InferError, InferResult};
use crate::sampling::{Rng, categorical_sample, softmax};

// ─── top_k_filter ────────────────────────────────────────────────────────────

/// Set all but the top-K logits to `f32::NEG_INFINITY` **in-place**.
///
/// After this call the input slice contains at most `k` finite values;
/// the rest are `−∞`.
///
/// # Errors
///
/// * [`InferError::EmptyBatch`]   — `logits` is empty.
/// * [`InferError::NanLogits`]    — any logit is NaN.
/// * [`InferError::SamplingError`] — `k == 0`.
pub fn top_k_filter(logits: &mut [f32], k: usize) -> InferResult<()> {
    if logits.is_empty() {
        return Err(InferError::EmptyBatch);
    }
    if k == 0 {
        return Err(InferError::SamplingError("top_k: k must be ≥ 1".to_owned()));
    }
    for &v in logits.iter() {
        if v.is_nan() {
            return Err(InferError::NanLogits);
        }
    }
    if k >= logits.len() {
        return Ok(()); // No filtering needed.
    }

    // Find the k-th largest value via a partial sort of a copy.
    let mut sorted: Vec<f32> = logits.to_vec();
    // Sort descending (NaN-free after check above).
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).expect("no NaN after check"));
    let threshold = sorted[k - 1];

    // Mask: keep at most k values that are ≥ threshold.
    let mut kept = 0_usize;
    for v in logits.iter_mut() {
        if *v >= threshold && kept < k {
            kept += 1;
        } else {
            *v = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

// ─── top_k_sample ────────────────────────────────────────────────────────────

/// Sample a token from the top-K distribution.
///
/// Applies [`top_k_filter`] to a copy of `logits`, then computes softmax and
/// draws from the resulting categorical distribution.
///
/// # Errors
///
/// Propagates errors from [`top_k_filter`].  Also returns
/// [`InferError::SamplingError`] if all probabilities are zero after filtering
/// (can happen only if all logits are `−∞`, which is unusual).
pub fn top_k_sample(logits: &[f32], k: usize, rng: &mut Rng) -> InferResult<u32> {
    if logits.is_empty() {
        return Err(InferError::EmptyBatch);
    }
    let mut filtered = logits.to_vec();
    top_k_filter(&mut filtered, k)?;

    // Replace remaining -inf with a very negative number to avoid NaN in exp.
    let probs = softmax(&filtered);

    if probs.iter().sum::<f32>() < 1e-10 {
        return Err(InferError::SamplingError(
            "top_k_sample: all probabilities are zero".to_owned(),
        ));
    }
    Ok(categorical_sample(&probs, rng) as u32)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_keeps_exactly_k() {
        let mut logits = vec![1.0_f32, 4.0, 2.0, 3.0, 0.5];
        top_k_filter(&mut logits, 2).unwrap();
        let finite_count = logits.iter().filter(|&&v| v.is_finite()).count();
        assert_eq!(finite_count, 2);
    }

    #[test]
    fn filter_top1_is_greedy() {
        let mut logits = vec![0.0_f32, 9.0, 1.0];
        top_k_filter(&mut logits, 1).unwrap();
        assert!(logits[1].is_finite());
        assert!(logits[0].is_infinite());
        assert!(logits[2].is_infinite());
    }

    #[test]
    fn filter_k_gte_vocab_is_noop() {
        let orig = vec![1.0_f32, 2.0, 3.0];
        let mut logits = orig.clone();
        top_k_filter(&mut logits, 10).unwrap();
        assert_eq!(logits, orig);
    }

    #[test]
    fn filter_empty_error() {
        assert!(matches!(
            top_k_filter(&mut [], 2),
            Err(InferError::EmptyBatch)
        ));
    }

    #[test]
    fn filter_k_zero_error() {
        let mut l = vec![1.0_f32];
        assert!(matches!(
            top_k_filter(&mut l, 0),
            Err(InferError::SamplingError(_))
        ));
    }

    #[test]
    fn sample_k1_always_picks_argmax() {
        let logits = vec![0.0_f32, 0.0, 10.0, 0.0];
        let mut rng = Rng::new(0);
        for _ in 0..20 {
            assert_eq!(top_k_sample(&logits, 1, &mut rng).unwrap(), 2);
        }
    }

    #[test]
    fn sample_k2_stays_in_top2() {
        let logits = vec![0.0_f32, 5.0, 4.0, -100.0];
        let mut rng = Rng::new(99);
        for _ in 0..100 {
            let t = top_k_sample(&logits, 2, &mut rng).unwrap();
            assert!(t == 1 || t == 2, "expected token 1 or 2, got {t}");
        }
    }
}
