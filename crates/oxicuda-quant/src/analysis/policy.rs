//! # Mixed-Precision Policy
//!
//! Assigns per-layer bit-widths to meet a target average bits-per-parameter
//! budget, while respecting per-layer sensitivity.  More sensitive layers
//! receive higher bit-widths.
//!
//! ## Greedy algorithm
//!
//! 1. Initialise every layer to the minimum bit-width.
//! 2. While the current average bits < target:
//!    a. Find the layer whose upgrade (to the next bit-width) yields the
//!    largest marginal sensitivity reduction per extra bit spent.
//!    b. Upgrade that layer.
//! 3. Return the final assignment.

use crate::analysis::sensitivity::LayerSensitivity;
use crate::error::{QuantError, QuantResult};

// ─── MixedPrecisionPolicy ────────────────────────────────────────────────────

/// Per-layer bit-width assignment produced by the greedy sensitivity policy.
#[derive(Debug, Clone)]
pub struct MixedPrecisionPolicy {
    /// Bit-width assigned to each layer (same order as the input sensitivity list).
    pub layer_bits: Vec<u32>,
    /// Layer names (mirrors the order of `layer_bits`).
    pub layer_names: Vec<String>,
    /// Target average bits-per-parameter (budget constraint).
    pub target_avg_bits: f32,
}

impl MixedPrecisionPolicy {
    /// Compute a mixed-precision policy from layer sensitivity profiles.
    ///
    /// # Parameters
    ///
    /// * `sensitivities`   — per-layer [`LayerSensitivity`] from `SensitivityAnalyzer`.
    /// * `target_avg_bits` — target average bit-width (e.g., `4.0` for ~4-bit).
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`]               — `sensitivities` is empty.
    /// * [`QuantError::InfeasibleCompressionTarget`] — target cannot be met even at max bits.
    pub fn from_sensitivity(
        sensitivities: &[LayerSensitivity],
        target_avg_bits: f32,
    ) -> QuantResult<Self> {
        if sensitivities.is_empty() {
            return Err(QuantError::EmptyInput(
                "MixedPrecisionPolicy::from_sensitivity",
            ));
        }

        // Verify target feasibility.
        let max_bits = sensitivities
            .iter()
            .map(|s| s.bits_range.iter().copied().max().unwrap_or(0))
            .max()
            .unwrap_or(0) as f32;
        let min_bits = sensitivities
            .iter()
            .map(|s| s.bits_range.iter().copied().min().unwrap_or(32))
            .min()
            .unwrap_or(32) as f32;

        if target_avg_bits > max_bits {
            return Err(QuantError::InfeasibleCompressionTarget {
                target: target_avg_bits,
            });
        }

        let n = sensitivities.len();
        // Start with minimum bit-width for each layer.
        let mut bits: Vec<u32> = sensitivities
            .iter()
            .map(|s| s.bits_range.iter().copied().min().unwrap_or(4))
            .collect();

        // Greedy upgrade loop.
        loop {
            let avg = bits.iter().sum::<u32>() as f32 / n as f32;
            if avg >= target_avg_bits {
                break;
            }

            // Find the layer that benefits most from upgrading one step.
            let mut best_layer = None;
            let mut best_gain = f32::NEG_INFINITY;

            for i in 0..n {
                let sens = &sensitivities[i];
                let cur_bits = bits[i];
                // Find next higher bit-width in this layer's range.
                let next = sens
                    .bits_range
                    .iter()
                    .copied()
                    .filter(|&b| b > cur_bits)
                    .min();
                let Some(next_bits) = next else { continue };

                // Sensitivity gain = reduction in MSE per extra bit used.
                let mse_cur = sens.mse_at(cur_bits).unwrap_or(0.0);
                let mse_next = sens.mse_at(next_bits).unwrap_or(0.0);
                let delta_mse = mse_cur - mse_next; // positive = improvement
                let delta_bits = (next_bits - cur_bits) as f32;
                let gain = delta_mse / delta_bits.max(1.0);

                if gain > best_gain {
                    best_gain = gain;
                    best_layer = Some((i, next_bits));
                }
            }

            match best_layer {
                Some((i, b)) => bits[i] = b,
                None => break, // All layers at maximum bits.
            }
        }

        // Check that minimum is actually achievable (edge case: target < min_bits).
        let actual_avg = bits.iter().sum::<u32>() as f32 / n as f32;
        if actual_avg < target_avg_bits - min_bits && target_avg_bits > min_bits {
            // Unable to reach target even at maximum.
            return Err(QuantError::InfeasibleCompressionTarget {
                target: target_avg_bits,
            });
        }

        let layer_names = sensitivities.iter().map(|s| s.name.clone()).collect();
        Ok(Self {
            layer_bits: bits,
            layer_names,
            target_avg_bits,
        })
    }

    /// Effective average bits per parameter across all layers.
    #[must_use]
    pub fn effective_average_bits(&self) -> f32 {
        if self.layer_bits.is_empty() {
            return 0.0;
        }
        self.layer_bits.iter().sum::<u32>() as f32 / self.layer_bits.len() as f32
    }

    /// Return the bit-width assigned to a layer by name.
    ///
    /// Returns `None` if the name is not found.
    #[must_use]
    pub fn bits_for_layer(&self, name: &str) -> Option<u32> {
        self.layer_names
            .iter()
            .position(|n| n == name)
            .map(|i| self.layer_bits[i])
    }

    /// Number of layers in the policy.
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.layer_bits.len()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::sensitivity::LayerSensitivity;
    use approx::assert_abs_diff_eq;

    fn make_sensitivity(name: &str, bits: &[u32], mse: &[f32]) -> LayerSensitivity {
        LayerSensitivity {
            bits_range: bits.to_vec(),
            mse_per_bits: mse.to_vec(),
            name: name.to_string(),
        }
    }

    #[test]
    fn greedy_assigns_more_bits_to_sensitive_layer() {
        // Layer 0 is very sensitive (high MSE at low bits), layer 1 is not.
        let s0 = make_sensitivity("l0", &[2, 4, 8], &[0.5, 0.05, 0.001]);
        let s1 = make_sensitivity("l1", &[2, 4, 8], &[0.01, 0.005, 0.001]);
        let policy = MixedPrecisionPolicy::from_sensitivity(&[s0, s1], 5.0).unwrap();
        // l0 should get more bits than l1.
        assert!(
            policy.bits_for_layer("l0").unwrap() >= policy.bits_for_layer("l1").unwrap(),
            "l0 (sensitive) should get >= bits than l1"
        );
    }

    #[test]
    fn target_average_bits_met() {
        let s0 = make_sensitivity("l0", &[2, 4, 8], &[0.5, 0.05, 0.001]);
        let s1 = make_sensitivity("l1", &[2, 4, 8], &[0.5, 0.05, 0.001]);
        let target = 4.0_f32;
        let policy = MixedPrecisionPolicy::from_sensitivity(&[s0, s1], target).unwrap();
        let avg = policy.effective_average_bits();
        assert!(
            avg >= target,
            "average bits {avg} should be >= target {target}"
        );
    }

    #[test]
    fn single_layer_policy() {
        let s = make_sensitivity("only", &[2, 4, 8], &[0.3, 0.02, 0.001]);
        let policy = MixedPrecisionPolicy::from_sensitivity(&[s], 4.0).unwrap();
        assert_eq!(policy.n_layers(), 1);
        assert_abs_diff_eq!(policy.effective_average_bits(), 4.0, epsilon = 1.0);
    }

    #[test]
    fn infeasible_target_error() {
        let s = make_sensitivity("l", &[2, 4], &[0.5, 0.01]);
        // Target 16 bits but max is 4 → infeasible.
        assert!(matches!(
            MixedPrecisionPolicy::from_sensitivity(&[s], 16.0),
            Err(QuantError::InfeasibleCompressionTarget { .. })
        ));
    }

    #[test]
    fn empty_sensitivities_error() {
        assert!(matches!(
            MixedPrecisionPolicy::from_sensitivity(&[], 4.0),
            Err(QuantError::EmptyInput(_))
        ));
    }

    #[test]
    fn bits_for_layer_lookup() {
        let s0 = make_sensitivity("attn", &[2, 4, 8], &[0.5, 0.05, 0.001]);
        let s1 = make_sensitivity("ffn", &[2, 4, 8], &[0.1, 0.01, 0.001]);
        let policy = MixedPrecisionPolicy::from_sensitivity(&[s0, s1], 4.0).unwrap();
        assert!(policy.bits_for_layer("attn").is_some());
        assert!(policy.bits_for_layer("ffn").is_some());
        assert!(policy.bits_for_layer("unknown").is_none());
    }

    #[test]
    fn all_layers_get_minimum_at_low_target() {
        // target = 2.0 = minimum → all layers should stay at 2 bits.
        let s0 = make_sensitivity("l0", &[2, 4, 8], &[0.5, 0.05, 0.001]);
        let s1 = make_sensitivity("l1", &[2, 4, 8], &[0.4, 0.04, 0.001]);
        let policy = MixedPrecisionPolicy::from_sensitivity(&[s0, s1], 2.0).unwrap();
        for &b in &policy.layer_bits {
            assert!(b >= 2, "all layers should be at minimum bits");
        }
    }
}
