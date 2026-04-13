//! # Response-Based Knowledge Distillation
//!
//! Distills the teacher's output logit distribution to the student.  The total
//! training loss combines a hard-label cross-entropy term with a soft-label
//! distillation term:
//!
//! ```text
//! L = α × CE(student_logits, hard_labels) + β × distil_loss(teacher, student)
//! ```
//!
//! Setting `hard_label_weight = 0` performs pure distillation without ground-truth
//! labels.

use crate::distill::loss::{DistilLoss, softmax};
use crate::error::{QuantError, QuantResult};

// ─── ResponseDistiller ───────────────────────────────────────────────────────

/// Response-based knowledge distillation.
///
/// Combines a hard-label cross-entropy loss with a soft-label distillation loss.
#[derive(Debug, Clone)]
pub struct ResponseDistiller {
    /// Distillation loss applied to soft targets.
    pub soft_loss: DistilLoss,
    /// Weight for the hard-label (cross-entropy) term.
    pub hard_label_weight: f32,
    /// Weight for the soft-label (distillation) term.
    pub soft_label_weight: f32,
}

impl ResponseDistiller {
    /// Create a response distiller.
    ///
    /// # Parameters
    ///
    /// * `soft_loss`          — distillation loss (e.g., KL divergence with temperature).
    /// * `hard_label_weight`  — weight α for cross-entropy term.
    /// * `soft_label_weight`  — weight β for distillation term.
    #[must_use]
    pub fn new(soft_loss: DistilLoss, hard_label_weight: f32, soft_label_weight: f32) -> Self {
        Self {
            soft_loss,
            hard_label_weight,
            soft_label_weight,
        }
    }

    /// Pure distillation (no hard labels): KL divergence at temperature `tau`.
    #[must_use]
    pub fn pure_kl(temperature: f32) -> Self {
        Self::new(DistilLoss::kl_divergence(temperature), 0.0, 1.0)
    }

    /// Combined distillation: `0.5 × CE + 0.5 × KL(τ=4)`.
    #[must_use]
    pub fn balanced() -> Self {
        Self::new(DistilLoss::kl_divergence(4.0), 0.5, 0.5)
    }

    /// Compute the combined distillation loss.
    ///
    /// # Parameters
    ///
    /// * `student_logits`  — unnormalised student output (length = n_classes).
    /// * `teacher_logits`  — unnormalised teacher output (same length).
    /// * `hard_label`      — integer ground-truth class index.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`]            — either logit slice is empty.
    /// * [`QuantError::TeacherStudentMismatch`] — logit slices differ in length.
    /// * [`QuantError::DimensionMismatch`]     — `hard_label` ≥ `n_classes`.
    pub fn compute_loss(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
        hard_label: usize,
    ) -> QuantResult<f32> {
        if student_logits.is_empty() {
            return Err(QuantError::EmptyInput(
                "ResponseDistiller: empty student logits",
            ));
        }
        if student_logits.len() != teacher_logits.len() {
            return Err(QuantError::TeacherStudentMismatch {
                teacher: teacher_logits.len(),
                student: student_logits.len(),
            });
        }
        if hard_label >= student_logits.len() {
            return Err(QuantError::DimensionMismatch {
                expected: student_logits.len(),
                got: hard_label + 1,
            });
        }

        // Soft loss component.
        let soft = self.soft_loss.compute(teacher_logits, student_logits)?;

        // Hard cross-entropy: -log(softmax(student)[hard_label])
        let probs = softmax(student_logits);
        let ce = -(probs[hard_label].max(1e-12).ln());

        Ok(self.hard_label_weight * ce + self.soft_label_weight * soft)
    }

    /// Compute distillation loss over a batch of examples.
    ///
    /// Returns the average loss over all examples in the batch.
    ///
    /// # Parameters
    ///
    /// * `student_batch`   — `[batch_size, n_classes]` row-major student logits.
    /// * `teacher_batch`   — `[batch_size, n_classes]` row-major teacher logits.
    /// * `hard_labels`     — `[batch_size]` integer class labels.
    /// * `n_classes`       — number of output classes.
    ///
    /// # Errors
    ///
    /// Propagates dimension and empty-input errors.
    pub fn compute_batch_loss(
        &self,
        student_batch: &[f32],
        teacher_batch: &[f32],
        hard_labels: &[usize],
        n_classes: usize,
    ) -> QuantResult<f32> {
        let batch_size = hard_labels.len();
        if batch_size == 0 {
            return Err(QuantError::EmptyInput(
                "ResponseDistiller::compute_batch_loss",
            ));
        }
        if student_batch.len() != batch_size * n_classes {
            return Err(QuantError::DimensionMismatch {
                expected: batch_size * n_classes,
                got: student_batch.len(),
            });
        }
        if teacher_batch.len() != batch_size * n_classes {
            return Err(QuantError::DimensionMismatch {
                expected: batch_size * n_classes,
                got: teacher_batch.len(),
            });
        }
        let total: f32 = (0..batch_size)
            .map(|b| {
                let s = &student_batch[b * n_classes..(b + 1) * n_classes];
                let t = &teacher_batch[b * n_classes..(b + 1) * n_classes];
                self.compute_loss(s, t, hard_labels[b]).unwrap_or(0.0)
            })
            .sum();
        Ok(total / batch_size as f32)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn pure_kl_zero_when_student_equals_teacher() {
        let d = ResponseDistiller::pure_kl(1.0);
        let logits = vec![1.0_f32, 2.0, 3.0];
        // With identical teacher/student and no hard labels, loss = 0.
        let loss = d.compute_loss(&logits, &logits, 0).unwrap();
        assert!(
            loss.abs() < 1e-3,
            "pure KL with equal logits ≈ 0, got {loss}"
        );
    }

    #[test]
    fn hard_label_only() {
        // hard_label_weight=1, soft_label_weight=0
        let d = ResponseDistiller::new(DistilLoss::mse(), 1.0, 0.0);
        let student = vec![0.0_f32, 0.0, 10.0]; // strongly predicts class 2
        let teacher = vec![0.0_f32, 0.0, 0.0]; // doesn't matter
        // CE = -log(softmax(student)[2]) ≈ 0 (class 2 is dominant)
        let loss = d.compute_loss(&student, &teacher, 2).unwrap();
        assert!(
            loss < 0.1,
            "CE for correct confident prediction ≈ 0, got {loss}"
        );
    }

    #[test]
    fn hard_label_out_of_range_error() {
        let d = ResponseDistiller::pure_kl(1.0);
        let logits = vec![1.0_f32, 2.0, 3.0];
        assert!(matches!(
            d.compute_loss(&logits, &logits, 3), // 3 >= n_classes=3
            Err(QuantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn teacher_student_mismatch_error() {
        let d = ResponseDistiller::pure_kl(1.0);
        let t = vec![1.0_f32; 3];
        let s = vec![1.0_f32; 4];
        assert!(matches!(
            d.compute_loss(&s, &t, 0),
            Err(QuantError::TeacherStudentMismatch { .. })
        ));
    }

    #[test]
    fn batch_loss_average() {
        let d = ResponseDistiller::new(DistilLoss::mse(), 0.0, 1.0);
        let n_classes = 3;
        let batch_size = 4;
        let teacher = vec![1.0_f32; batch_size * n_classes];
        let student = vec![1.0_f32; batch_size * n_classes];
        let labels = vec![0_usize; batch_size];
        let loss = d
            .compute_batch_loss(&student, &teacher, &labels, n_classes)
            .unwrap();
        assert_abs_diff_eq!(loss, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn balanced_distiller_construction() {
        let d = ResponseDistiller::balanced();
        assert_abs_diff_eq!(d.hard_label_weight, 0.5, epsilon = 1e-7);
        assert_abs_diff_eq!(d.soft_label_weight, 0.5, epsilon = 1e-7);
    }
}
