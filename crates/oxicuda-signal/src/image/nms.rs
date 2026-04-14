//! Non-Maximum Suppression (NMS) for object detection.
//!
//! NMS is the standard post-processing step in object detection to eliminate
//! duplicate bounding box predictions, retaining only the highest-confidence
//! detection within each spatial region.
//!
//! ## Greedy NMS Algorithm
//!
//! 1. Sort detections by score in descending order.
//! 2. For each un-suppressed detection, suppress all others with IoU > threshold.
//!
//! ## Soft-NMS (Bodla et al. 2017)
//!
//! Instead of hard suppression, reduce the score of overlapping boxes:
//! ```text
//! score_j *= f(IoU(box_i, box_j))
//! ```
//! where `f` is a Gaussian or linear decay function.

use crate::error::{SignalError, SignalResult};

// --------------------------------------------------------------------------- //
//  Bounding box type
// --------------------------------------------------------------------------- //

/// Axis-aligned bounding box in `[x1, y1, x2, y2]` format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// Left edge (x-coordinate of top-left corner).
    pub x1: f32,
    /// Top edge (y-coordinate of top-left corner).
    pub y1: f32,
    /// Right edge (x-coordinate of bottom-right corner).
    pub x2: f32,
    /// Bottom edge (y-coordinate of bottom-right corner).
    pub y2: f32,
}

impl BBox {
    /// Create a new bounding box, ensuring `x1 ≤ x2` and `y1 ≤ y2`.
    #[must_use]
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            x1: x1.min(x2),
            y1: y1.min(y2),
            x2: x1.max(x2),
            y2: y1.max(y2),
        }
    }

    /// Area of the bounding box.
    #[must_use]
    pub fn area(self) -> f32 {
        let w = (self.x2 - self.x1).max(0.0);
        let h = (self.y2 - self.y1).max(0.0);
        w * h
    }
}

/// Intersection over Union (IoU) between two bounding boxes.
#[must_use]
pub fn iou(a: BBox, b: BBox) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let inter_w = (ix2 - ix1).max(0.0);
    let inter_h = (iy2 - iy1).max(0.0);
    let inter = inter_w * inter_h;
    let union = a.area() + b.area() - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

// --------------------------------------------------------------------------- //
//  Greedy NMS
// --------------------------------------------------------------------------- //

/// Perform greedy Non-Maximum Suppression.
///
/// Returns the indices (into `boxes` / `scores`) of the surviving detections.
///
/// # Arguments
/// - `boxes` — bounding boxes
/// - `scores` — confidence score for each box
/// - `iou_thresh` — IoU threshold above which a box is suppressed
///
/// # Errors
/// Returns `SignalError::DimensionMismatch` if `boxes.len() != scores.len()`.
pub fn nms_greedy(boxes: &[BBox], scores: &[f32], iou_thresh: f32) -> SignalResult<Vec<usize>> {
    if boxes.len() != scores.len() {
        return Err(SignalError::DimensionMismatch {
            expected: format!("scores.len() = {}", boxes.len()),
            got: format!("{}", scores.len()),
        });
    }
    // Sort indices by score descending.
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for &j in &indices {
            if suppressed[j] || i == j {
                continue;
            }
            if iou(boxes[i], boxes[j]) > iou_thresh {
                suppressed[j] = true;
            }
        }
    }
    Ok(keep)
}

// --------------------------------------------------------------------------- //
//  Soft-NMS
// --------------------------------------------------------------------------- //

/// Soft-NMS decay function.
#[derive(Debug, Clone, Copy)]
pub enum SoftNmsDecay {
    /// Linear decay: `score *= (1 - IoU)`.
    Linear,
    /// Gaussian decay: `score *= exp(-IoU² / σ²)`.
    Gaussian {
        /// Gaussian sigma² parameter.
        sigma_sq: f32,
    },
}

/// Perform Soft-NMS (Bodla et al. 2017).
///
/// Returns `(kept_indices, final_scores)` after applying soft suppression.
/// All detections with final score ≥ `score_thresh` are returned.
///
/// # Errors
/// Returns `SignalError::DimensionMismatch` if `boxes.len() != scores.len()`.
pub fn nms_soft(
    boxes: &[BBox],
    scores: &[f32],
    iou_thresh: f32,
    score_thresh: f32,
    decay: SoftNmsDecay,
) -> SignalResult<Vec<(usize, f32)>> {
    if boxes.len() != scores.len() {
        return Err(SignalError::DimensionMismatch {
            expected: format!("scores.len() = {}", boxes.len()),
            got: format!("{}", scores.len()),
        });
    }
    let n = boxes.len();
    let mut soft_scores = scores.to_vec();
    // Sort by score descending initially.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        soft_scores[b]
            .partial_cmp(&soft_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut results = Vec::new();
    let mut remaining: Vec<usize> = order.clone();

    while !remaining.is_empty() {
        // Pick the highest-score remaining box.
        let best_idx = remaining
            .iter()
            .cloned()
            .max_by(|&a, &b| {
                soft_scores[a]
                    .partial_cmp(&soft_scores[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| {
                SignalError::InvalidParameter(
                    "remaining boxes is empty during soft-NMS iteration".to_owned(),
                )
            })?;
        if soft_scores[best_idx] < score_thresh {
            break;
        }
        results.push((best_idx, soft_scores[best_idx]));
        remaining.retain(|&i| i != best_idx);

        // Decay scores of remaining boxes based on IoU.
        for &j in &remaining {
            let overlap = iou(boxes[best_idx], boxes[j]);
            if overlap > iou_thresh {
                soft_scores[j] *= match decay {
                    SoftNmsDecay::Linear => 1.0 - overlap,
                    SoftNmsDecay::Gaussian { sigma_sq } => (-overlap * overlap / sigma_sq).exp(),
                };
            }
        }
        // Re-sort remaining by updated scores.
        remaining.sort_by(|&a, &b| {
            soft_scores[b]
                .partial_cmp(&soft_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    Ok(results)
}

// --------------------------------------------------------------------------- //
//  2D score-map NMS (for keypoint detection / heatmaps)
// --------------------------------------------------------------------------- //

/// Apply 2D NMS to a score heatmap.
///
/// Suppresses any pixel that is not a local maximum within a `(2r+1)×(2r+1)`
/// neighbourhood.  Returns a binary mask (1 = kept, 0 = suppressed).
///
/// - `heatmap` — flat score map of shape `[height, width]`
/// - `height`, `width` — spatial dimensions
/// - `radius` — half-width of suppression neighbourhood
/// - `threshold` — minimum score to be considered
///
/// # Errors
/// Returns `SignalError::InvalidSize` if `heatmap.len() != height * width`.
pub fn nms_heatmap(
    heatmap: &[f32],
    height: usize,
    width: usize,
    radius: usize,
    threshold: f32,
) -> SignalResult<Vec<u8>> {
    if heatmap.len() != height * width {
        return Err(SignalError::InvalidSize(format!(
            "heatmap length {} ≠ height ({height}) × width ({width}) = {}",
            heatmap.len(),
            height * width
        )));
    }
    let mut mask = vec![0u8; height * width];
    for r in 0..height {
        for c in 0..width {
            let v = heatmap[r * width + c];
            if v < threshold {
                continue;
            }
            // Check if v is local maximum in (2r+1)×(2r+1) neighbourhood.
            let r0 = r.saturating_sub(radius);
            let r1 = (r + radius + 1).min(height);
            let c0 = c.saturating_sub(radius);
            let c1 = (c + radius + 1).min(width);
            let is_max = (r0..r1).all(|ri| (c0..c1).all(|ci| heatmap[ri * width + ci] <= v));
            if is_max {
                mask[r * width + c] = 1;
            }
        }
    }
    Ok(mask)
}

// --------------------------------------------------------------------------- //
//  Tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_perfect_overlap() {
        let b = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((iou(b, b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = BBox::new(0.0, 0.0, 5.0, 5.0);
        let b = BBox::new(10.0, 10.0, 15.0, 15.0);
        assert_eq!(iou(a, b), 0.0);
    }

    #[test]
    fn test_iou_half_overlap() {
        let a = BBox::new(0.0, 0.0, 4.0, 4.0);
        let b = BBox::new(2.0, 0.0, 6.0, 4.0); // 2×4 = 8 inter, 16+16-8=24 union
        let iou_val = iou(a, b);
        assert!((iou_val - 8.0 / 24.0).abs() < 1e-5);
    }

    #[test]
    fn test_nms_greedy_single_box() {
        let boxes = vec![BBox::new(0.0, 0.0, 10.0, 10.0)];
        let scores = vec![0.9_f32];
        let keep = nms_greedy(&boxes, &scores, 0.5).unwrap();
        assert_eq!(keep, vec![0]);
    }

    #[test]
    fn test_nms_greedy_suppresses_duplicate() {
        let boxes = vec![
            BBox::new(0.0, 0.0, 10.0, 10.0),
            BBox::new(0.5, 0.5, 10.5, 10.5), // nearly identical → should be suppressed
        ];
        let scores = vec![0.9_f32, 0.8_f32];
        let keep = nms_greedy(&boxes, &scores, 0.5).unwrap();
        assert_eq!(keep, vec![0], "expected only box 0 kept");
    }

    #[test]
    fn test_nms_greedy_keeps_non_overlapping() {
        let boxes = vec![
            BBox::new(0.0, 0.0, 5.0, 5.0),
            BBox::new(10.0, 10.0, 15.0, 15.0),
        ];
        let scores = vec![0.9_f32, 0.8_f32];
        let keep = nms_greedy(&boxes, &scores, 0.5).unwrap();
        assert_eq!(keep.len(), 2);
    }

    #[test]
    fn test_nms_greedy_dimension_mismatch() {
        let boxes = vec![BBox::new(0.0, 0.0, 1.0, 1.0)];
        let scores = vec![0.9_f32, 0.8_f32];
        assert!(nms_greedy(&boxes, &scores, 0.5).is_err());
    }

    #[test]
    fn test_soft_nms_returns_results() {
        let boxes = vec![
            BBox::new(0.0, 0.0, 10.0, 10.0),
            BBox::new(1.0, 1.0, 11.0, 11.0),
            BBox::new(20.0, 20.0, 30.0, 30.0),
        ];
        let scores = vec![0.9_f32, 0.85_f32, 0.8_f32];
        let results = nms_soft(
            &boxes,
            &scores,
            0.3,
            0.5,
            SoftNmsDecay::Gaussian { sigma_sq: 0.5 },
        )
        .unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_nms_heatmap_local_max() {
        // 3×3 heatmap with maximum at centre.
        let h = vec![0.1, 0.2, 0.1, 0.2, 0.9, 0.2, 0.1, 0.2, 0.1_f32];
        let mask = nms_heatmap(&h, 3, 3, 1, 0.5).unwrap();
        // Only centre should be marked.
        assert_eq!(mask[4], 1, "centre should be kept");
        for i in [0, 1, 2, 3, 5, 6, 7, 8] {
            assert_eq!(mask[i], 0, "non-max at {i} should be suppressed");
        }
    }

    #[test]
    fn test_nms_heatmap_threshold() {
        let h = vec![0.1_f32; 9];
        let mask = nms_heatmap(&h, 3, 3, 1, 0.5).unwrap(); // all below threshold
        assert!(mask.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_bbox_area() {
        let b = BBox::new(0.0, 0.0, 4.0, 5.0);
        assert!((b.area() - 20.0).abs() < 1e-6);
    }
}
