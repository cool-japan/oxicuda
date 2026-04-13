//! # Distillation Loss Functions
//!
//! Loss functions used to transfer knowledge from a teacher model to a
//! student model during knowledge distillation.
//!
//! | Loss type              | Description                                |
//! |------------------------|--------------------------------------------|
//! | `KLDivergence`         | KL(softmax(T/τ) ‖ softmax(S/τ)) × τ²     |
//! | `Mse`                  | Mean squared error between outputs         |
//! | `Cosine`               | 1 − cos_sim(teacher, student)              |
//! | `CombinedKlMse`        | Weighted combination of KL and MSE         |

use crate::error::{QuantError, QuantResult};

// ─── DistilLossType ───────────────────────────────────────────────────────────

/// Distillation loss variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistilLossType {
    /// KL divergence between temperature-scaled teacher and student logit distributions.
    ///
    /// `loss = τ² × KL(softmax(teacher/τ) ‖ softmax(student/τ))`
    KlDivergence {
        /// Distillation temperature (τ > 0).  Higher values produce softer targets.
        temperature: f32,
    },
    /// Mean squared error between teacher and student outputs.
    Mse,
    /// Cosine distance: `1 − cosine_similarity(teacher, student)`.
    Cosine,
    /// Weighted combination: `kl_weight × KL + mse_weight × MSE`.
    CombinedKlMse {
        /// Weight for KL divergence term.
        kl_weight: f32,
        /// Weight for MSE term.
        mse_weight: f32,
        /// Temperature for KL divergence.
        temperature: f32,
    },
}

// ─── DistilLoss ───────────────────────────────────────────────────────────────

/// Distillation loss calculator.
#[derive(Debug, Clone, Copy)]
pub struct DistilLoss {
    /// Type of distillation loss.
    pub loss_type: DistilLossType,
}

impl DistilLoss {
    /// Create a KL-divergence distillation loss with the given temperature.
    #[must_use]
    pub fn kl_divergence(temperature: f32) -> Self {
        Self {
            loss_type: DistilLossType::KlDivergence { temperature },
        }
    }

    /// Create an MSE distillation loss.
    #[must_use]
    pub fn mse() -> Self {
        Self {
            loss_type: DistilLossType::Mse,
        }
    }

    /// Create a cosine-distance loss.
    #[must_use]
    pub fn cosine() -> Self {
        Self {
            loss_type: DistilLossType::Cosine,
        }
    }

    /// Create a combined KL + MSE loss.
    #[must_use]
    pub fn combined(kl_weight: f32, mse_weight: f32, temperature: f32) -> Self {
        Self {
            loss_type: DistilLossType::CombinedKlMse {
                kl_weight,
                mse_weight,
                temperature,
            },
        }
    }

    /// Compute the distillation loss between `teacher` and `student` logits.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`]           — either slice is empty.
    /// * [`QuantError::TeacherStudentMismatch`] — different lengths.
    pub fn compute(&self, teacher: &[f32], student: &[f32]) -> QuantResult<f32> {
        if teacher.is_empty() {
            return Err(QuantError::EmptyInput(
                "DistilLoss::compute: teacher is empty",
            ));
        }
        if teacher.len() != student.len() {
            return Err(QuantError::TeacherStudentMismatch {
                teacher: teacher.len(),
                student: student.len(),
            });
        }
        match self.loss_type {
            DistilLossType::KlDivergence { temperature } => {
                Ok(kl_divergence_loss(teacher, student, temperature))
            }
            DistilLossType::Mse => Ok(mse_loss(teacher, student)),
            DistilLossType::Cosine => Ok(cosine_distance(teacher, student)),
            DistilLossType::CombinedKlMse {
                kl_weight,
                mse_weight,
                temperature,
            } => {
                let kl = kl_divergence_loss(teacher, student, temperature);
                let mse = mse_loss(teacher, student);
                Ok(kl_weight * kl + mse_weight * mse)
            }
        }
    }
}

// ─── Loss implementations ─────────────────────────────────────────────────────

/// Compute softmax of a logit vector.
pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
    exps.iter().map(|&e| e / sum).collect()
}

/// KL divergence: τ² × KL(P ‖ Q) where P = softmax(teacher/τ), Q = softmax(student/τ).
fn kl_divergence_loss(teacher: &[f32], student: &[f32], temperature: f32) -> f32 {
    let tau = temperature.max(1e-6);
    let t_scaled: Vec<f32> = teacher.iter().map(|&x| x / tau).collect();
    let s_scaled: Vec<f32> = student.iter().map(|&x| x / tau).collect();
    let p = softmax(&t_scaled);
    let q = softmax(&s_scaled);
    let kl: f32 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi < 1e-12 {
                0.0
            } else {
                pi * (pi.ln() - qi.max(1e-12).ln())
            }
        })
        .sum();
    // τ² factor comes from the chain rule of the temperature scaling.
    tau * tau * kl
}

/// Mean squared error between teacher and student outputs.
fn mse_loss(teacher: &[f32], student: &[f32]) -> f32 {
    let n = teacher.len() as f32;
    teacher
        .iter()
        .zip(student.iter())
        .map(|(t, s)| (t - s).powi(2))
        .sum::<f32>()
        / n
}

/// Cosine distance: 1 − cosine_similarity.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = (na * nb).max(1e-12);
    1.0 - (dot / denom).clamp(-1.0, 1.0)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn kl_zero_when_identical() {
        // KL(P ‖ P) = 0
        let logits = vec![1.0_f32, 2.0, 3.0];
        let loss = DistilLoss::kl_divergence(1.0)
            .compute(&logits, &logits)
            .unwrap();
        assert!(loss.abs() < 1e-4, "KL(P‖P) should be ~0, got {loss}");
    }

    #[test]
    fn kl_nonzero_when_different() {
        let teacher = vec![1.0_f32, 2.0, 3.0];
        let student = vec![3.0_f32, 2.0, 1.0]; // reversed
        let loss = DistilLoss::kl_divergence(1.0)
            .compute(&teacher, &student)
            .unwrap();
        assert!(
            loss > 0.0,
            "KL(P‖Q) with different distributions should be > 0"
        );
    }

    #[test]
    fn kl_temperature_scaling() {
        // Higher temperature → softer targets → smaller KL
        let teacher = vec![0.0_f32, 2.0, 4.0];
        let student = vec![0.0_f32, 1.0, 2.0];
        let loss_t1 = DistilLoss::kl_divergence(1.0)
            .compute(&teacher, &student)
            .unwrap();
        let loss_t4 = DistilLoss::kl_divergence(4.0)
            .compute(&teacher, &student)
            .unwrap();
        // At higher T, logits are compressed → less KL (before τ² scaling)
        assert!(
            loss_t1 != loss_t4,
            "Different temperatures should give different losses"
        );
    }

    #[test]
    fn mse_zero_when_identical() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let loss = DistilLoss::mse().compute(&x, &x).unwrap();
        assert_abs_diff_eq!(loss, 0.0, epsilon = 1e-7);
    }

    #[test]
    fn mse_correct() {
        let teacher = vec![0.0_f32, 0.0];
        let student = vec![1.0_f32, 1.0];
        let loss = DistilLoss::mse().compute(&teacher, &student).unwrap();
        assert_abs_diff_eq!(loss, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn cosine_zero_when_identical() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let loss = DistilLoss::cosine().compute(&x, &x).unwrap();
        assert!(
            loss.abs() < 1e-5,
            "cosine distance between equal vectors = 0, got {loss}"
        );
    }

    #[test]
    fn cosine_two_when_opposite() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        let loss = DistilLoss::cosine().compute(&a, &b).unwrap();
        // 1 - (-1) = 2
        assert_abs_diff_eq!(loss, 2.0, epsilon = 1e-5);
    }

    #[test]
    fn combined_loss() {
        let teacher = vec![1.0_f32, 2.0];
        let student = vec![1.5_f32, 1.5];
        let loss = DistilLoss::combined(0.5, 0.5, 1.0)
            .compute(&teacher, &student)
            .unwrap();
        assert!(loss >= 0.0, "combined loss must be non-negative");
    }

    #[test]
    fn mismatch_error() {
        let a = vec![1.0_f32; 3];
        let b = vec![1.0_f32; 4];
        assert!(matches!(
            DistilLoss::mse().compute(&a, &b),
            Err(QuantError::TeacherStudentMismatch { .. })
        ));
    }

    #[test]
    fn empty_input_error() {
        assert!(matches!(
            DistilLoss::mse().compute(&[], &[]),
            Err(QuantError::EmptyInput(_))
        ));
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, -1.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }
}
