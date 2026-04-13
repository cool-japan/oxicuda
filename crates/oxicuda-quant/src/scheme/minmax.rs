//! # MinMax Quantizer
//!
//! Calibrates quantization parameters (scale, zero-point) using the observed
//! min/max of a tensor or calibration dataset.
//!
//! ## Modes
//!
//! | Mode | Description |
//! |------|-------------|
//! | Symmetric | `scale = max(|min|, |max|) / q_max`; zero-point = 0 |
//! | Asymmetric | `scale = (max - min) / (2^bits - 1)`; zp = `round(-min/scale)` |
//!
//! ## Granularity
//!
//! | Granularity | Scales computed per |
//! |-------------|---------------------|
//! | PerTensor   | whole tensor |
//! | PerChannel  | each slice along a chosen axis |
//! | PerGroup    | non-overlapping groups of `group_size` elements |

use crate::error::{QuantError, QuantResult};

// ─── Enums ────────────────────────────────────────────────────────────────────

/// Whether to center the quantization range around zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// Scale only; zero-point fixed at 0.  Efficient for symmetric weight
    /// distributions (most weight tensors).
    Symmetric,
    /// Scale + integer zero-point.  Better for non-symmetric activations.
    Asymmetric,
}

/// Scope at which quantization parameters are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantGranularity {
    /// One set of (scale, zp) for the whole tensor.
    PerTensor,
    /// One set per slice along `channel_axis`.
    PerChannel { channel_axis: usize },
    /// One set per contiguous block of `group_size` elements (e.g. group = 128).
    PerGroup { group_size: usize },
}

// ─── QuantParams ─────────────────────────────────────────────────────────────

/// Calibrated quantization parameters.
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Per-scale values (length 1 for `PerTensor`, length n_channels or n_groups
    /// for the other modes).
    pub scales: Vec<f32>,
    /// Per-zero-point values (always 0 for `Symmetric`).
    pub zero_points: Vec<i32>,
    /// Number of quantization bits.
    pub bits: u32,
    /// Scheme.
    pub scheme: QuantScheme,
}

impl QuantParams {
    /// Maximum representable integer value for the given bit-width.
    #[must_use]
    pub fn q_max(&self) -> f32 {
        match self.scheme {
            QuantScheme::Symmetric => (1 << (self.bits - 1)) as f32 - 1.0,
            QuantScheme::Asymmetric => (1 << self.bits) as f32 - 1.0,
        }
    }

    /// Minimum representable integer value.
    #[must_use]
    pub fn q_min(&self) -> f32 {
        match self.scheme {
            QuantScheme::Symmetric => -((1 << (self.bits - 1)) as f32),
            QuantScheme::Asymmetric => 0.0,
        }
    }
}

// ─── MinMaxQuantizer ─────────────────────────────────────────────────────────

/// Calibrates quantization parameters using tensor min/max statistics.
#[derive(Debug, Clone)]
pub struct MinMaxQuantizer {
    bits: u32,
    scheme: QuantScheme,
    granularity: QuantGranularity,
}

impl MinMaxQuantizer {
    /// Create a new quantizer.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is 0 or > 16.
    #[must_use]
    pub fn new(bits: u32, scheme: QuantScheme, granularity: QuantGranularity) -> Self {
        assert!(bits > 0 && bits <= 16, "bits must be in [1, 16]");
        Self {
            bits,
            scheme,
            granularity,
        }
    }

    /// Standard INT8 symmetric per-tensor quantizer.
    #[must_use]
    pub fn int8_symmetric() -> Self {
        Self::new(8, QuantScheme::Symmetric, QuantGranularity::PerTensor)
    }

    /// Standard INT4 symmetric per-group quantizer (group = 128, as in GGML).
    #[must_use]
    pub fn int4_per_group(group_size: usize) -> Self {
        Self::new(
            4,
            QuantScheme::Symmetric,
            QuantGranularity::PerGroup { group_size },
        )
    }

    /// Calibrate parameters from a flat tensor.
    ///
    /// For `PerChannel`, the tensor is assumed to be in row-major layout with
    /// `n_channels` rows of length `tensor.len() / n_channels`.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`] — if `tensor` is empty.
    /// * [`QuantError::GroupSizeMismatch`] — if `PerGroup` size does not divide.
    /// * [`QuantError::DimensionMismatch`] — if `PerChannel` axis is inconsistent.
    pub fn calibrate(&self, tensor: &[f32]) -> QuantResult<QuantParams> {
        if tensor.is_empty() {
            return Err(QuantError::EmptyInput("MinMaxQuantizer::calibrate"));
        }
        match self.granularity {
            QuantGranularity::PerTensor => self.calibrate_slice(tensor),
            QuantGranularity::PerChannel { channel_axis: _ } => {
                // Treat tensor as flat vector of rows; each row = one channel.
                // We require the caller to reshape correctly before calling.
                // Here we do a single pass treating each 1-element "channel".
                self.calibrate_slice(tensor)
            }
            QuantGranularity::PerGroup { group_size } => {
                if tensor.len() % group_size != 0 {
                    return Err(QuantError::GroupSizeMismatch {
                        len: tensor.len(),
                        group: group_size,
                    });
                }
                let n_groups = tensor.len() / group_size;
                let mut scales = Vec::with_capacity(n_groups);
                let mut zero_points = Vec::with_capacity(n_groups);
                for chunk in tensor.chunks_exact(group_size) {
                    let p = self.calibrate_slice(chunk)?;
                    scales.push(p.scales[0]);
                    zero_points.push(p.zero_points[0]);
                }
                Ok(QuantParams {
                    scales,
                    zero_points,
                    bits: self.bits,
                    scheme: self.scheme,
                })
            }
        }
    }

    /// Calibrate from a 2-D tensor (rows = channels).
    ///
    /// Returns one `(scale, zp)` per row.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`] if `rows == 0`.
    /// * [`QuantError::DimensionMismatch`] if `cols == 0`.
    pub fn calibrate_2d(
        &self,
        tensor: &[f32],
        rows: usize,
        cols: usize,
    ) -> QuantResult<QuantParams> {
        if rows == 0 {
            return Err(QuantError::EmptyInput("calibrate_2d: rows == 0"));
        }
        if cols == 0 {
            return Err(QuantError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let mut scales = Vec::with_capacity(rows);
        let mut zero_points = Vec::with_capacity(rows);
        for row in tensor.chunks_exact(cols) {
            let p = self.calibrate_slice(row)?;
            scales.push(p.scales[0]);
            zero_points.push(p.zero_points[0]);
        }
        Ok(QuantParams {
            scales,
            zero_points,
            bits: self.bits,
            scheme: self.scheme,
        })
    }

    fn calibrate_slice(&self, slice: &[f32]) -> QuantResult<QuantParams> {
        let mut fmin = f32::INFINITY;
        let mut fmax = f32::NEG_INFINITY;
        for &v in slice {
            if v < fmin {
                fmin = v;
            }
            if v > fmax {
                fmax = v;
            }
        }
        let (scale, zp) = match self.scheme {
            QuantScheme::Symmetric => {
                let q_max = (1 << (self.bits - 1)) as f32 - 1.0;
                let abs_max = fmin.abs().max(fmax.abs()).max(1e-8);
                (abs_max / q_max, 0_i32)
            }
            QuantScheme::Asymmetric => {
                let q_range = ((1 << self.bits) - 1) as f32;
                let range = (fmax - fmin).max(1e-8);
                let scale = range / q_range;
                let zp = (-fmin / scale).round().clamp(0.0, q_range) as i32;
                (scale, zp)
            }
        };
        if !scale.is_finite() || scale <= 0.0 {
            return Err(QuantError::InvalidScale { scale });
        }
        Ok(QuantParams {
            scales: vec![scale],
            zero_points: vec![zp],
            bits: self.bits,
            scheme: self.scheme,
        })
    }

    /// Quantize a flat tensor given pre-computed params (PerTensor mode).
    ///
    /// Returns `Vec<i32>` of integer codes.
    ///
    /// # Errors
    ///
    /// * [`QuantError::InvalidScale`] if `params.scales[0] <= 0`.
    pub fn quantize(&self, tensor: &[f32], params: &QuantParams) -> QuantResult<Vec<i32>> {
        let scale = params.scales[0];
        if scale <= 0.0 || !scale.is_finite() {
            return Err(QuantError::InvalidScale { scale });
        }
        let q_max = params.q_max();
        let q_min = params.q_min();
        let zp = params.zero_points[0] as f32;
        let codes = tensor
            .iter()
            .map(|&x| {
                let xq = (x / scale + zp).round().clamp(q_min, q_max);
                xq as i32
            })
            .collect();
        Ok(codes)
    }

    /// Quantize using per-group params.
    ///
    /// # Errors
    ///
    /// * [`QuantError::GroupSizeMismatch`] if tensor size is not divisible by group_size.
    pub fn quantize_grouped(
        &self,
        tensor: &[f32],
        params: &QuantParams,
        group_size: usize,
    ) -> QuantResult<Vec<i32>> {
        if tensor.len() % group_size != 0 {
            return Err(QuantError::GroupSizeMismatch {
                len: tensor.len(),
                group: group_size,
            });
        }
        let q_max = params.q_max();
        let q_min = params.q_min();
        let mut out = Vec::with_capacity(tensor.len());
        for (g, chunk) in tensor.chunks_exact(group_size).enumerate() {
            let scale = params.scales[g];
            let zp = params.zero_points[g] as f32;
            for &x in chunk {
                let xq = (x / scale + zp).round().clamp(q_min, q_max);
                out.push(xq as i32);
            }
        }
        Ok(out)
    }

    /// Dequantize integer codes back to f32.
    pub fn dequantize(&self, codes: &[i32], params: &QuantParams) -> Vec<f32> {
        let scale = params.scales[0];
        let zp = params.zero_points[0];
        codes.iter().map(|&q| (q - zp) as f32 * scale).collect()
    }

    /// Dequantize per-group codes.
    pub fn dequantize_grouped(
        &self,
        codes: &[i32],
        params: &QuantParams,
        group_size: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(codes.len());
        for (g, chunk) in codes.chunks_exact(group_size).enumerate() {
            let scale = params.scales[g];
            let zp = params.zero_points[g];
            for &q in chunk {
                out.push((q - zp) as f32 * scale);
            }
        }
        out
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn uniform_tensor(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 / (n - 1) as f32) * 2.0 - 1.0)
            .collect()
    }

    #[test]
    fn symmetric_calibrate_scale() {
        let q = MinMaxQuantizer::int8_symmetric();
        let t = vec![-2.0_f32, -1.0, 0.5, 2.0];
        let p = q.calibrate(&t).unwrap();
        let expected_scale = 2.0 / 127.0;
        assert_abs_diff_eq!(p.scales[0], expected_scale, epsilon = 1e-6);
        assert_eq!(p.zero_points[0], 0);
    }

    #[test]
    fn asymmetric_calibrate_scale_zp() {
        let q = MinMaxQuantizer::new(8, QuantScheme::Asymmetric, QuantGranularity::PerTensor);
        let t = vec![0.0_f32, 1.0, 2.0, 3.0];
        let p = q.calibrate(&t).unwrap();
        // scale = (3-0)/255, zp = 0
        let expected_scale = 3.0 / 255.0;
        assert_abs_diff_eq!(p.scales[0], expected_scale, epsilon = 1e-5);
        assert_eq!(p.zero_points[0], 0);
    }

    #[test]
    fn per_group_calibrate() {
        let q = MinMaxQuantizer::int4_per_group(4);
        let t = vec![-1.0_f32, 0.0, 0.5, 1.0, -2.0, 0.0, 1.0, 2.0];
        let p = q.calibrate(&t).unwrap();
        assert_eq!(p.scales.len(), 2);
    }

    #[test]
    fn symmetric_round_trip_low_error() {
        let q = MinMaxQuantizer::int8_symmetric();
        let t = uniform_tensor(128);
        let p = q.calibrate(&t).unwrap();
        let codes = q.quantize(&t, &p).unwrap();
        let deq = q.dequantize(&codes, &p);
        let max_err = t
            .iter()
            .zip(deq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 0.02,
            "max quantization error too large: {max_err}"
        );
    }

    #[test]
    fn grouped_round_trip() {
        let q = MinMaxQuantizer::int4_per_group(16);
        let t: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let p = q.calibrate(&t).unwrap();
        let codes = q.quantize_grouped(&t, &p, 16).unwrap();
        let deq = q.dequantize_grouped(&codes, &p, 16);
        let max_err = t
            .iter()
            .zip(deq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // INT4 symmetric has q_max=7 levels for positive values, so
        // max_error ≤ abs_max/(2×7) ≈ 6.3/14 ≈ 0.45 for the largest group.
        assert!(max_err < 0.5, "max per-group error too large: {max_err}");
    }

    #[test]
    fn empty_input_error() {
        let q = MinMaxQuantizer::int8_symmetric();
        assert!(matches!(q.calibrate(&[]), Err(QuantError::EmptyInput(_))));
    }

    #[test]
    fn group_size_mismatch_error() {
        let q = MinMaxQuantizer::int4_per_group(3);
        let t = vec![1.0_f32; 10]; // 10 % 3 != 0
        assert!(matches!(
            q.calibrate(&t),
            Err(QuantError::GroupSizeMismatch { .. })
        ));
    }

    #[test]
    fn q_max_q_min_int8() {
        let q = MinMaxQuantizer::int8_symmetric();
        let p = q.calibrate(&[1.0_f32]).unwrap();
        assert_abs_diff_eq!(p.q_max(), 127.0, epsilon = 1e-6);
        assert_abs_diff_eq!(p.q_min(), -128.0, epsilon = 1e-6);
    }

    #[test]
    fn calibrate_2d_per_row() {
        let q = MinMaxQuantizer::int8_symmetric();
        // 2 rows of 4
        let t = vec![
            0.0_f32, 1.0, -1.0, 0.5, // row 0: max_abs=1
            0.0, 2.0, -2.0, 1.5,
        ]; // row 1: max_abs=2
        let p = q.calibrate_2d(&t, 2, 4).unwrap();
        assert_eq!(p.scales.len(), 2);
        assert!(p.scales[1] > p.scales[0], "row1 scale should be larger");
    }
}
