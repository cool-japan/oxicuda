//! # Quantization Sensitivity Analysis
//!
//! Measures how sensitive each layer is to quantization at different bit-widths.
//! More sensitive layers should be assigned higher bit-widths in a mixed-precision
//! quantization scheme.
//!
//! ## Sensitivity metric
//!
//! For each layer and each candidate bit-width, we quantize the weights with a
//! MinMax symmetric scheme and compute the mean squared error between the
//! original and dequantized weights:
//!
//! ```text
//! sensitivity(layer, bits) = MSE(W, dequant(quant(W, bits)))
//! ```

use crate::error::{QuantError, QuantResult};
use crate::scheme::minmax::{MinMaxQuantizer, QuantGranularity, QuantScheme};

// ─── LayerSensitivity ────────────────────────────────────────────────────────

/// Sensitivity scores for one layer across multiple bit-widths.
#[derive(Debug, Clone)]
pub struct LayerSensitivity {
    /// Candidate bit-widths tested (sorted ascending).
    pub bits_range: Vec<u32>,
    /// Quantization MSE at each bit-width (same order as `bits_range`).
    pub mse_per_bits: Vec<f32>,
    /// Layer name or identifier (optional).
    pub name: String,
}

impl LayerSensitivity {
    /// Return the sensitivity (MSE) for a specific bit-width.
    ///
    /// Returns `None` if the bit-width was not tested.
    #[must_use]
    pub fn mse_at(&self, bits: u32) -> Option<f32> {
        self.bits_range
            .iter()
            .position(|&b| b == bits)
            .map(|i| self.mse_per_bits[i])
    }

    /// Mean sensitivity across all tested bit-widths.
    #[must_use]
    pub fn mean_sensitivity(&self) -> f32 {
        if self.mse_per_bits.is_empty() {
            return 0.0;
        }
        self.mse_per_bits.iter().sum::<f32>() / self.mse_per_bits.len() as f32
    }

    /// Returns `true` if higher bit-widths give lower MSE (monotone sensitivity).
    #[must_use]
    pub fn is_monotone(&self) -> bool {
        self.mse_per_bits.windows(2).all(|w| w[0] >= w[1])
    }
}

// ─── SensitivityAnalyzer ─────────────────────────────────────────────────────

/// Analyses per-layer quantization sensitivity.
#[derive(Debug, Clone, Default)]
pub struct SensitivityAnalyzer;

impl SensitivityAnalyzer {
    /// Create a new sensitivity analyser.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compute quantization sensitivity for one layer across `bits_range`.
    ///
    /// # Parameters
    ///
    /// * `weights`    — flat weight tensor (any layout).
    /// * `bits_range` — candidate bit-widths (e.g., `&[2, 3, 4, 8]`).
    /// * `name`       — optional layer label.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`] — `weights` is empty.
    /// * [`QuantError::InvalidBitWidth`] — any bit-width in `bits_range` is 0 or > 16.
    pub fn analyze_layer(
        &self,
        weights: &[f32],
        bits_range: &[u32],
        name: impl Into<String>,
    ) -> QuantResult<LayerSensitivity> {
        if weights.is_empty() {
            return Err(QuantError::EmptyInput("SensitivityAnalyzer::analyze_layer"));
        }
        for &b in bits_range {
            if b == 0 || b > 16 {
                return Err(QuantError::InvalidBitWidth { bits: b });
            }
        }

        let mut mse_per_bits = Vec::with_capacity(bits_range.len());
        for &bits in bits_range {
            let q = MinMaxQuantizer::new(bits, QuantScheme::Symmetric, QuantGranularity::PerTensor);
            let p = q.calibrate(weights)?;
            let qw = q.quantize(weights, &p)?;
            let dqw = q.dequantize(&qw, &p);
            let mse = weights
                .iter()
                .zip(dqw.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / weights.len() as f32;
            mse_per_bits.push(mse);
        }

        Ok(LayerSensitivity {
            bits_range: bits_range.to_vec(),
            mse_per_bits,
            name: name.into(),
        })
    }

    /// Analyse multiple layers and return their sensitivity profiles.
    ///
    /// # Parameters
    ///
    /// * `layers`     — list of `(name, weights)` pairs.
    /// * `bits_range` — candidate bit-widths to test for each layer.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`analyze_layer`](Self::analyze_layer).
    pub fn analyze_multiple<'a>(
        &self,
        layers: &[(&'a str, &'a [f32])],
        bits_range: &[u32],
    ) -> QuantResult<Vec<LayerSensitivity>> {
        layers
            .iter()
            .map(|(name, weights)| self.analyze_layer(weights, bits_range, *name))
            .collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_weights(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 / n as f32) * 2.0 - 1.0).collect()
    }

    #[test]
    fn higher_bits_lower_mse() {
        let a = SensitivityAnalyzer::new();
        let w = make_weights(64);
        let sens = a.analyze_layer(&w, &[2, 4, 8], "test_layer").unwrap();
        assert!(
            sens.mse_per_bits[0] >= sens.mse_per_bits[2],
            "MSE at 2 bits ({}) should be >= MSE at 8 bits ({})",
            sens.mse_per_bits[0],
            sens.mse_per_bits[2]
        );
    }

    #[test]
    fn int8_very_low_mse() {
        let a = SensitivityAnalyzer::new();
        let w = make_weights(128);
        let sens = a.analyze_layer(&w, &[8], "layer0").unwrap();
        assert!(sens.mse_at(8).unwrap() < 1e-4, "INT8 MSE should be tiny");
    }

    #[test]
    fn mse_at_missing_bits_returns_none() {
        let a = SensitivityAnalyzer::new();
        let w = make_weights(16);
        let sens = a.analyze_layer(&w, &[4, 8], "l").unwrap();
        assert!(sens.mse_at(2).is_none());
        assert!(sens.mse_at(4).is_some());
    }

    #[test]
    fn monotone_sensitivity() {
        let a = SensitivityAnalyzer::new();
        let w = make_weights(64);
        let sens = a.analyze_layer(&w, &[2, 4, 8], "l").unwrap();
        assert!(
            sens.is_monotone(),
            "MSE should decrease with increasing bits"
        );
    }

    #[test]
    fn analyze_multiple_layers() {
        let a = SensitivityAnalyzer::new();
        let w0 = make_weights(32);
        let w1 = make_weights(64);
        let result = a
            .analyze_multiple(&[("layer0", &w0), ("layer1", &w1)], &[4, 8])
            .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "layer0");
        assert_eq!(result[1].name, "layer1");
    }

    #[test]
    fn empty_input_error() {
        let a = SensitivityAnalyzer::new();
        assert!(matches!(
            a.analyze_layer(&[], &[4], "l"),
            Err(QuantError::EmptyInput(_))
        ));
    }

    #[test]
    fn invalid_bit_width_error() {
        let a = SensitivityAnalyzer::new();
        let w = make_weights(16);
        assert!(matches!(
            a.analyze_layer(&w, &[0], "l"),
            Err(QuantError::InvalidBitWidth { bits: 0 })
        ));
    }

    #[test]
    fn mean_sensitivity_nonzero() {
        let a = SensitivityAnalyzer::new();
        let w = make_weights(32);
        let sens = a.analyze_layer(&w, &[2, 4], "l").unwrap();
        assert!(sens.mean_sensitivity() > 0.0);
        assert_abs_diff_eq!(
            sens.mean_sensitivity(),
            (sens.mse_per_bits[0] + sens.mse_per_bits[1]) / 2.0,
            epsilon = 1e-6
        );
    }
}
