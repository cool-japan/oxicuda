//! # Greedy (Argmax) Sampling
//!
//! Always selects the token with the highest logit.  Equivalent to a
//! temperature of 0 (or top-K with K=1 after sorting).
//!
//! Greedy decoding is deterministic and produces the single most probable
//! continuation according to the model, which is often sufficient for tasks
//! that have a clear single best answer.

use crate::error::{InferError, InferResult};

// ─── greedy_sample ────────────────────────────────────────────────────────────

/// Return the index of the maximum logit.
///
/// # Errors
///
/// * [`InferError::EmptyBatch`]  — `logits` is empty.
/// * [`InferError::NanLogits`]   — at least one logit is NaN.
pub fn greedy_sample(logits: &[f32]) -> InferResult<u32> {
    if logits.is_empty() {
        return Err(InferError::EmptyBatch);
    }
    let mut best_idx = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            return Err(InferError::NanLogits);
        }
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as u32)
}

// ─── greedy_sample_batch ─────────────────────────────────────────────────────

/// Batch variant: returns one argmax per row.
///
/// `logits` has shape `[n_seqs][vocab_size]`; all rows must be non-empty and
/// have the same length.
///
/// # Errors
///
/// * [`InferError::EmptyBatch`]         — `logits` is empty.
/// * [`InferError::DimensionMismatch`]  — rows have different lengths.
/// * [`InferError::NanLogits`]          — any logit is NaN.
pub fn greedy_sample_batch(logits: &[Vec<f32>]) -> InferResult<Vec<u32>> {
    if logits.is_empty() {
        return Err(InferError::EmptyBatch);
    }
    let width = logits[0].len();
    let mut out = Vec::with_capacity(logits.len());
    for (row_idx, row) in logits.iter().enumerate() {
        if row.len() != width {
            return Err(InferError::DimensionMismatch {
                expected: width,
                got: row.len(),
            });
        }
        let _ = row_idx;
        out.push(greedy_sample(row)?);
    }
    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_basic() {
        let logits = vec![0.1_f32, 5.0, 2.0, -1.0];
        assert_eq!(greedy_sample(&logits).unwrap(), 1);
    }

    #[test]
    fn argmax_first_element() {
        let logits = vec![10.0_f32, 1.0, 2.0];
        assert_eq!(greedy_sample(&logits).unwrap(), 0);
    }

    #[test]
    fn argmax_last_element() {
        let logits = vec![0.0_f32, 0.0, 7.0];
        assert_eq!(greedy_sample(&logits).unwrap(), 2);
    }

    #[test]
    fn empty_returns_error() {
        assert!(matches!(greedy_sample(&[]), Err(InferError::EmptyBatch)));
    }

    #[test]
    fn nan_returns_error() {
        let logits = vec![1.0_f32, f32::NAN, 2.0];
        assert!(matches!(greedy_sample(&logits), Err(InferError::NanLogits)));
    }

    #[test]
    fn batch_argmax() {
        let rows = vec![vec![0.0_f32, 3.0, 1.0], vec![5.0_f32, 1.0, 2.0]];
        let tokens = greedy_sample_batch(&rows).unwrap();
        assert_eq!(tokens, vec![1, 0]);
    }

    #[test]
    fn batch_empty_returns_error() {
        assert!(matches!(
            greedy_sample_batch(&[]),
            Err(InferError::EmptyBatch)
        ));
    }
}
