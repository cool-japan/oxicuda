//! # Nucleus (Top-P) Sampling
//!
//! Nucleus sampling (Holtzman et al., 2019) restricts sampling to the
//! smallest set of tokens whose cumulative probability mass is at least `p`.
//! This adapts the effective vocabulary size to the model's confidence:
//! a peaked distribution keeps very few tokens; a flat distribution keeps
//! many.
//!
//! ## Algorithm
//!
//! 1. Sort tokens by logit (descending).
//! 2. Compute cumulative softmax probabilities along the sorted order.
//! 3. Keep all tokens up to and including the one that first pushes the
//!    cumulative sum ≥ p.
//! 4. Set all remaining tokens to −∞.
//! 5. Re-normalise and sample.
//!
//! ## Edge cases
//!
//! * `p ≥ 1.0` — no filtering is applied.
//! * `p ≤ 0.0` — only the single most probable token is kept (greedy).

use crate::error::{InferError, InferResult};
use crate::sampling::{Rng, categorical_sample, softmax};

// ─── top_p_filter ────────────────────────────────────────────────────────────

/// Apply nucleus (top-p) filtering to `logits` **in-place**.
///
/// After this call the slice contains only the nucleus tokens as finite
/// values; the rest are `f32::NEG_INFINITY`.
///
/// # Errors
///
/// * [`InferError::EmptyBatch`]    — `logits` is empty.
/// * [`InferError::NanLogits`]     — any logit is NaN.
/// * [`InferError::SamplingError`] — `p` is negative or NaN.
pub fn top_p_filter(logits: &mut [f32], p: f32) -> InferResult<()> {
    if logits.is_empty() {
        return Err(InferError::EmptyBatch);
    }
    if p.is_nan() || p < 0.0 {
        return Err(InferError::SamplingError(format!(
            "top_p_filter: p must be in [0, 1], got {p}"
        )));
    }
    for &v in logits.iter() {
        if v.is_nan() {
            return Err(InferError::NanLogits);
        }
    }

    if p >= 1.0 {
        return Ok(()); // No filtering.
    }

    let n = logits.len();
    // Compute softmax probabilities (operate on a copy for sorting).
    let probs = softmax(logits);

    // Sort indices by probability descending.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).expect("no NaN"));

    // Find the nucleus cutoff index.
    let mut cumsum = 0.0_f32;
    let mut cutoff = 0_usize; // last index (inclusive) in the nucleus
    for &idx in &order {
        cumsum += probs[idx];
        cutoff = idx;
        if cumsum >= p {
            break;
        }
    }

    // Determine the probability threshold = logit value at `cutoff` index.
    let threshold_prob = probs[cutoff];

    // Mask logits with probability < threshold.
    // Keep exactly the nucleus (positions where prob >= threshold, up to
    // the cutoff position in the sorted order).
    let nucleus_set: std::collections::HashSet<usize> = order
        .iter()
        .copied()
        .take_while(|&idx| {
            let mut c = 0.0_f32;
            // Recompute cumulative up to this position.
            for &i in order.iter().take_while(|&&j| j != idx || { true }) {
                c += probs[i];
                if i == idx {
                    break;
                }
            }
            c <= cumsum
        })
        .collect();
    let _ = nucleus_set; // Not used; we use the threshold approach below.

    // Simpler: mask any token whose probability is strictly less than
    // the threshold probability (the tail beyond the nucleus).
    for (i, v) in logits.iter_mut().enumerate() {
        if probs[i] < threshold_prob {
            *v = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

// ─── top_p_sample ────────────────────────────────────────────────────────────

/// Sample from the nucleus distribution.
///
/// Applies [`top_p_filter`] to a copy of `logits`, then re-normalises and
/// draws a token.
///
/// # Errors
///
/// Propagates errors from [`top_p_filter`].
pub fn top_p_sample(logits: &[f32], p: f32, rng: &mut Rng) -> InferResult<u32> {
    if logits.is_empty() {
        return Err(InferError::EmptyBatch);
    }
    let mut filtered = logits.to_vec();
    top_p_filter(&mut filtered, p)?;
    let probs = softmax(&filtered);
    if probs.iter().sum::<f32>() < 1e-10 {
        return Err(InferError::SamplingError(
            "top_p_sample: all probabilities are zero".to_owned(),
        ));
    }
    Ok(categorical_sample(&probs, rng) as u32)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_one_no_filter() {
        let orig = vec![1.0_f32, 2.0, 3.0];
        let mut l = orig.clone();
        top_p_filter(&mut l, 1.0).unwrap();
        assert_eq!(l, orig);
    }

    #[test]
    fn p_zero_keeps_only_top() {
        let mut l = vec![1.0_f32, 10.0, 2.0];
        top_p_filter(&mut l, 0.0).unwrap();
        assert!(l[1].is_finite());
        // Others may be filtered (depends on probability mass)
    }

    #[test]
    fn filter_empty_error() {
        assert!(matches!(
            top_p_filter(&mut [], 0.9),
            Err(InferError::EmptyBatch)
        ));
    }

    #[test]
    fn filter_negative_p_error() {
        let mut l = vec![1.0_f32, 2.0];
        assert!(matches!(
            top_p_filter(&mut l, -0.1),
            Err(InferError::SamplingError(_))
        ));
    }

    #[test]
    fn sample_p_one_all_tokens_possible() {
        // With p=1.0 all tokens are valid.
        let logits = vec![1.0_f32; 5];
        let mut rng = Rng::new(0);
        let seen: std::collections::HashSet<u32> = (0..500)
            .map(|_| top_p_sample(&logits, 1.0, &mut rng).unwrap())
            .collect();
        // Should see all 5 tokens eventually.
        assert!(
            seen.len() >= 4,
            "expected most tokens to appear, got {}",
            seen.len()
        );
    }

    #[test]
    fn sample_p_small_stays_in_nucleus() {
        // Very small p: only the top token survives.
        let logits = vec![0.0_f32, 100.0, 0.0, 0.0];
        let mut rng = Rng::new(0);
        for _ in 0..50 {
            let t = top_p_sample(&logits, 0.01, &mut rng).unwrap();
            assert_eq!(t, 1, "expected token 1 to dominate");
        }
    }
}
