//! # NF4 — NormalFloat4 Quantization
//!
//! NF4 (Dettmers et al., 2023 — "QLoRA: Efficient Finetuning of Quantized LLMs")
//! is a data type that is **information-theoretically optimal** for normally
//! distributed weights.  It stores 4-bit indices into a 16-entry lookup table
//! whose values are the quantiles of N(0, 1) at equal probability mass points.
//!
//! ## Encoding
//!
//! ```text
//! absmax = max(|W|)
//! W_norm = W / absmax            ∈ [-1, 1]
//! code   = argmin_{v ∈ LUT} |W_norm - v|   (nearest-neighbour in LUT)
//! ```
//!
//! Two codes are packed per byte (lo nibble = first element).
//!
//! ## Decoding
//!
//! ```text
//! W_approx = LUT[code] * absmax
//! ```

use crate::error::{QuantError, QuantResult};

// ─── NF4 Lookup Table ────────────────────────────────────────────────────────

/// The 16 NF4 quantization levels (sorted ascending).
///
/// These are the quantiles of the standard normal distribution at probability
/// mass points `{0.5/16, 1.5/16, ..., 15.5/16}`, scaled so that the extreme
/// values are exactly ±1.
pub const NF4_LUT: [f32; 16] = [
    -1.0,
    -0.696_192_86,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_03,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

// ─── Nf4Quantizer ─────────────────────────────────────────────────────────────

/// NF4 quantizer — encodes tensors to packed 4-bit NF4 codes.
///
/// Blocks of `block_size` elements share an `absmax` scaling factor.
/// The default `block_size` of 64 matches the QLoRA paper.
#[derive(Debug, Clone)]
pub struct Nf4Quantizer {
    /// Number of elements per absmax scaling block.
    pub block_size: usize,
}

impl Default for Nf4Quantizer {
    fn default() -> Self {
        Self { block_size: 64 }
    }
}

impl Nf4Quantizer {
    /// Create an NF4 quantizer with the given block size.
    ///
    /// # Panics
    ///
    /// Panics if `block_size == 0`.
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        Self { block_size }
    }

    /// Encode a flat tensor to packed NF4 bytes and per-block absmax values.
    ///
    /// The number of elements must be a multiple of `block_size`, and
    /// `block_size` must be even (pairs of nibbles pack into bytes).
    ///
    /// Returns `(packed_bytes, absmax_per_block)`.
    ///
    /// # Errors
    ///
    /// * [`QuantError::GroupSizeMismatch`] — if `len` is not divisible by `block_size`.
    /// * [`QuantError::EmptyInput`] — if `tensor` is empty.
    pub fn encode(&self, tensor: &[f32]) -> QuantResult<(Vec<u8>, Vec<f32>)> {
        if tensor.is_empty() {
            return Err(QuantError::EmptyInput("Nf4Quantizer::encode"));
        }
        if tensor.len() % self.block_size != 0 {
            return Err(QuantError::GroupSizeMismatch {
                len: tensor.len(),
                group: self.block_size,
            });
        }
        let n_blocks = tensor.len() / self.block_size;
        let n_bytes = tensor.len() / 2; // 2 codes per byte
        let mut packed = vec![0u8; n_bytes];
        let mut absmaxs = Vec::with_capacity(n_blocks);

        for (blk_idx, block) in tensor.chunks_exact(self.block_size).enumerate() {
            // Compute absmax for this block.
            let absmax = block.iter().map(|&v| v.abs()).fold(0.0_f32, f32::max);
            let absmax = if absmax < 1e-8 { 1e-8 } else { absmax };
            absmaxs.push(absmax);

            // Encode each element.
            let base_byte = blk_idx * self.block_size / 2;
            for (i, &v) in block.iter().enumerate() {
                let normed = (v / absmax).clamp(-1.0, 1.0);
                let code = nearest_nf4(normed) as u8;
                let byte_idx = base_byte + i / 2;
                if i % 2 == 0 {
                    packed[byte_idx] = code; // lo nibble
                } else {
                    packed[byte_idx] |= code << 4; // hi nibble
                }
            }
        }
        Ok((packed, absmaxs))
    }

    /// Decode packed NF4 bytes back to f32 using the stored absmax values.
    ///
    /// # Errors
    ///
    /// * [`QuantError::DimensionMismatch`] — if packed/absmax lengths are inconsistent.
    pub fn decode(&self, packed: &[u8], absmaxs: &[f32]) -> QuantResult<Vec<f32>> {
        let n_floats = packed.len() * 2;
        let n_blocks_expected = n_floats / self.block_size;
        if absmaxs.len() != n_blocks_expected {
            return Err(QuantError::DimensionMismatch {
                expected: n_blocks_expected,
                got: absmaxs.len(),
            });
        }
        let mut out = Vec::with_capacity(n_floats);
        for (blk_idx, block_bytes) in packed.chunks_exact(self.block_size / 2).enumerate() {
            let absmax = absmaxs[blk_idx];
            for &byte in block_bytes {
                let lo = (byte & 0x0F) as usize;
                let hi = (byte >> 4) as usize;
                out.push(NF4_LUT[lo] * absmax);
                out.push(NF4_LUT[hi] * absmax);
            }
        }
        Ok(out)
    }

    /// Estimate the quantization error (mean squared error) for a tensor.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`encode`](Self::encode) / [`decode`](Self::decode).
    pub fn quantization_mse(&self, tensor: &[f32]) -> QuantResult<f32> {
        let (packed, absmaxs) = self.encode(tensor)?;
        let decoded = self.decode(&packed, &absmaxs)?;
        let mse = tensor
            .iter()
            .zip(decoded.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / tensor.len() as f32;
        Ok(mse)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Find the index of the nearest NF4 level using binary search.
///
/// Since `NF4_LUT` is sorted, we use the standard partition-point approach
/// and compare both neighbours.
fn nearest_nf4(v: f32) -> usize {
    // Binary search for the insertion point.
    let mut lo = 0_usize;
    let mut hi = NF4_LUT.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if NF4_LUT[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // `lo` is the first index where NF4_LUT[lo] >= v.
    if lo == 0 {
        return 0;
    }
    if lo == NF4_LUT.len() {
        return NF4_LUT.len() - 1;
    }
    // Compare the two neighbours.
    let d_lo = (v - NF4_LUT[lo - 1]).abs();
    let d_hi = (NF4_LUT[lo] - v).abs();
    if d_lo <= d_hi { lo - 1 } else { lo }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn lut_is_sorted_ascending() {
        for w in NF4_LUT.windows(2) {
            assert!(w[0] < w[1], "LUT must be sorted: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn lut_endpoints() {
        assert_abs_diff_eq!(NF4_LUT[0], -1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(NF4_LUT[15], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(NF4_LUT[7], 0.0, epsilon = 1e-9);
    }

    #[test]
    fn nearest_nf4_endpoints() {
        assert_eq!(nearest_nf4(-1.0), 0, "exactly -1 → index 0");
        assert_eq!(nearest_nf4(1.0), 15, "exactly 1 → index 15");
        assert_eq!(nearest_nf4(0.0), 7, "exactly 0 → index 7");
    }

    #[test]
    fn nearest_nf4_midpoint() {
        // Between LUT[7]=0 and LUT[8]=0.0796: midpoint ≈ 0.0398
        let mid = (NF4_LUT[7] + NF4_LUT[8]) / 2.0;
        let idx = nearest_nf4(mid);
        assert!(idx == 7 || idx == 8, "midpoint should map to 7 or 8");
    }

    #[test]
    fn encode_decode_round_trip() {
        let q = Nf4Quantizer::new(64);
        let t: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0) - 1.0).collect();
        let (packed, absmaxs) = q.encode(&t).unwrap();
        assert_eq!(packed.len(), 64);
        assert_eq!(absmaxs.len(), 2);
        let decoded = q.decode(&packed, &absmaxs).unwrap();
        // NF4 is lossy; error should be small but non-zero.
        let mse = t
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 128.0;
        assert!(mse < 0.01, "MSE too large: {mse}");
    }

    #[test]
    fn all_zeros_encodes_cleanly() {
        let q = Nf4Quantizer::default();
        let t = vec![0.0_f32; 64];
        let (packed, absmaxs) = q.encode(&t).unwrap();
        // With all-zero input, absmax = 1e-8, codes all map to index 7 (value 0).
        assert_eq!(absmaxs.len(), 1);
        let decoded = q.decode(&packed, &absmaxs).unwrap();
        for v in decoded {
            assert!(v.abs() < 1e-5, "decoded zero should be near zero");
        }
    }

    #[test]
    fn mse_within_nf4_theory() {
        // Theory: NF4 should give ~0.3% relative MSE for normal data.
        // Use a larger random-ish sample.
        let q = Nf4Quantizer::new(64);
        // Approximate N(0,1) using sum of uniforms (CLT)
        let t: Vec<f32> = (0..1024)
            .map(|i| {
                let u = (i % 64) as f32 / 64.0;
                2.0 * u - 1.0
            })
            .collect();
        let mse = q.quantization_mse(&t).unwrap();
        assert!(mse < 0.05, "NF4 MSE unexpectedly large: {mse}");
    }

    #[test]
    fn group_size_mismatch_error() {
        let q = Nf4Quantizer::new(64);
        let t = vec![0.5_f32; 65]; // 65 % 64 != 0
        assert!(matches!(
            q.encode(&t),
            Err(QuantError::GroupSizeMismatch { .. })
        ));
    }

    #[test]
    fn decode_length_mismatch_error() {
        let q = Nf4Quantizer::new(64);
        let packed = vec![0u8; 32];
        let absmaxs = vec![1.0_f32; 5]; // wrong: expected 1
        assert!(matches!(
            q.decode(&packed, &absmaxs),
            Err(QuantError::DimensionMismatch { .. })
        ));
    }
}
