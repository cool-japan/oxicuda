//! # Compression Metrics
//!
//! Tracks the compression statistics for quantized and pruned models:
//! effective bits per parameter, compression ratio, and sparsity.

use crate::error::QuantResult;

// ─── CompressionMetrics ───────────────────────────────────────────────────────

/// Compression statistics for a single layer or entire model.
#[derive(Debug, Clone, Default)]
pub struct CompressionMetrics {
    /// Number of parameters.
    pub n_parameters: u64,
    /// Bit-width of the original (full-precision) representation.
    pub original_bits_per_param: f32,
    /// Effective bit-width used in the compressed representation.
    /// For quantized weights this equals `quant_bits`; for pruned weights
    /// the effective bits = `quant_bits × (1 − sparsity)`.
    pub effective_bits_per_param: f32,
    /// Fraction of weights that are pruned ∈ [0, 1].
    pub sparsity: f32,
    /// Mean squared quantization error (0 if no quantization was applied).
    pub quantization_mse: f32,
}

impl CompressionMetrics {
    /// Create metrics for a layer quantized to `quant_bits` bits with no pruning.
    #[must_use]
    pub fn quantized_only(n_parameters: u64, quant_bits: u32, quant_mse: f32) -> Self {
        Self {
            n_parameters,
            original_bits_per_param: 32.0,
            effective_bits_per_param: quant_bits as f32,
            sparsity: 0.0,
            quantization_mse: quant_mse,
        }
    }

    /// Create metrics for a layer pruned to `sparsity` with FP32 weights.
    #[must_use]
    pub fn pruned_only(n_parameters: u64, sparsity: f32) -> Self {
        Self {
            n_parameters,
            original_bits_per_param: 32.0,
            effective_bits_per_param: 32.0 * (1.0 - sparsity),
            sparsity,
            quantization_mse: 0.0,
        }
    }

    /// Create metrics for a layer that is both quantized and pruned.
    #[must_use]
    pub fn quantized_and_pruned(
        n_parameters: u64,
        quant_bits: u32,
        sparsity: f32,
        quant_mse: f32,
    ) -> Self {
        Self {
            n_parameters,
            original_bits_per_param: 32.0,
            effective_bits_per_param: quant_bits as f32 * (1.0 - sparsity),
            sparsity,
            quantization_mse: quant_mse,
        }
    }

    /// Ratio of original to compressed storage: `original_bits / effective_bits`.
    ///
    /// Returns `f32::INFINITY` if the effective bits per param is 0.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        if self.effective_bits_per_param <= 0.0 {
            return f32::INFINITY;
        }
        self.original_bits_per_param / self.effective_bits_per_param
    }

    /// Total original bits for this layer.
    #[must_use]
    pub fn total_original_bits(&self) -> f64 {
        self.n_parameters as f64 * self.original_bits_per_param as f64
    }

    /// Total compressed bits for this layer.
    #[must_use]
    pub fn total_compressed_bits(&self) -> f64 {
        self.n_parameters as f64 * self.effective_bits_per_param as f64
    }
}

// ─── ModelCompressionMetrics ─────────────────────────────────────────────────

/// Aggregated compression statistics over an entire model.
#[derive(Debug, Clone, Default)]
pub struct ModelCompressionMetrics {
    /// Per-layer metrics.
    pub layers: Vec<CompressionMetrics>,
    /// Per-layer names.
    pub names: Vec<String>,
}

impl ModelCompressionMetrics {
    /// Create an empty model metrics container.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a layer's metrics.
    pub fn add_layer(&mut self, name: impl Into<String>, m: CompressionMetrics) {
        self.names.push(name.into());
        self.layers.push(m);
    }

    /// Total number of parameters across all layers.
    #[must_use]
    pub fn total_parameters(&self) -> u64 {
        self.layers.iter().map(|m| m.n_parameters).sum()
    }

    /// Model-wide compression ratio (total original bits / total compressed bits).
    #[must_use]
    pub fn model_compression_ratio(&self) -> f32 {
        let orig: f64 = self.layers.iter().map(|m| m.total_original_bits()).sum();
        let comp: f64 = self.layers.iter().map(|m| m.total_compressed_bits()).sum();
        if comp <= 0.0 {
            return f32::INFINITY;
        }
        (orig / comp) as f32
    }

    /// Weighted average quantization MSE across all layers.
    #[must_use]
    pub fn mean_quantization_mse(&self) -> f32 {
        let total_n: u64 = self.total_parameters();
        if total_n == 0 {
            return 0.0;
        }
        let weighted: f32 = self
            .layers
            .iter()
            .map(|m| m.quantization_mse * m.n_parameters as f32)
            .sum();
        weighted / total_n as f32
    }

    /// Average effective bits per parameter across all layers (parameter-weighted).
    #[must_use]
    pub fn average_effective_bits(&self) -> f32 {
        let total_n = self.total_parameters();
        if total_n == 0 {
            return 0.0;
        }
        let weighted: f32 = self
            .layers
            .iter()
            .map(|m| m.effective_bits_per_param * m.n_parameters as f32)
            .sum();
        weighted / total_n as f32
    }

    /// Compute the MSE for a quantized layer inline and add it.
    ///
    /// This is a convenience method to avoid pre-computing MSE externally.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::QuantError::EmptyInput`] if `weights` is empty.
    pub fn add_quantized_layer(
        &mut self,
        name: impl Into<String>,
        weights: &[f32],
        quant_bits: u32,
        quantization_mse: f32,
    ) -> QuantResult<()> {
        if weights.is_empty() {
            return Err(crate::error::QuantError::EmptyInput(
                "ModelCompressionMetrics::add_quantized_layer",
            ));
        }
        let m =
            CompressionMetrics::quantized_only(weights.len() as u64, quant_bits, quantization_mse);
        self.add_layer(name, m);
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn int8_compression_ratio() {
        let m = CompressionMetrics::quantized_only(1024, 8, 0.0);
        assert_abs_diff_eq!(m.compression_ratio(), 4.0, epsilon = 1e-5);
    }

    #[test]
    fn int4_compression_ratio() {
        let m = CompressionMetrics::quantized_only(1024, 4, 0.0);
        assert_abs_diff_eq!(m.compression_ratio(), 8.0, epsilon = 1e-5);
    }

    #[test]
    fn pruned_50_percent_fp32_ratio() {
        let m = CompressionMetrics::pruned_only(1024, 0.5);
        // effective = 32 * 0.5 = 16, ratio = 32/16 = 2
        assert_abs_diff_eq!(m.compression_ratio(), 2.0, epsilon = 1e-5);
    }

    #[test]
    fn quantized_and_pruned_metrics() {
        // INT4 + 50% sparsity: effective = 4 * 0.5 = 2 bits/param
        let m = CompressionMetrics::quantized_and_pruned(1024, 4, 0.5, 0.001);
        assert_abs_diff_eq!(m.effective_bits_per_param, 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(m.compression_ratio(), 16.0, epsilon = 1e-5);
    }

    #[test]
    fn model_compression_ratio_weighted() {
        let mut model = ModelCompressionMetrics::new();
        model.add_layer("l0", CompressionMetrics::quantized_only(100, 8, 0.0));
        model.add_layer("l1", CompressionMetrics::quantized_only(900, 4, 0.0));
        // total orig = 1000 * 32 = 32000 bits
        // total comp = 100*8 + 900*4 = 800 + 3600 = 4400 bits
        // ratio = 32000 / 4400 ≈ 7.27
        let ratio = model.model_compression_ratio();
        assert!(ratio > 7.0 && ratio < 8.0, "ratio = {ratio}");
    }

    #[test]
    fn average_effective_bits() {
        let mut model = ModelCompressionMetrics::new();
        model.add_layer("l0", CompressionMetrics::quantized_only(100, 4, 0.0));
        model.add_layer("l1", CompressionMetrics::quantized_only(100, 8, 0.0));
        // weighted avg = (100*4 + 100*8) / 200 = 6
        assert_abs_diff_eq!(model.average_effective_bits(), 6.0, epsilon = 1e-5);
    }

    #[test]
    fn total_bits_correct() {
        let m = CompressionMetrics::quantized_only(1000, 8, 0.0);
        assert_abs_diff_eq!(m.total_original_bits(), 32_000.0, epsilon = 1.0);
        assert_abs_diff_eq!(m.total_compressed_bits(), 8_000.0, epsilon = 1.0);
    }

    #[test]
    fn zero_effective_bits_gives_infinity() {
        let m = CompressionMetrics {
            effective_bits_per_param: 0.0,
            ..Default::default()
        };
        assert!(m.compression_ratio().is_infinite());
    }
}
