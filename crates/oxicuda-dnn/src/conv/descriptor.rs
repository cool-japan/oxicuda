//! Convolution problem descriptor.
//!
//! [`ConvProblem`] captures all parameters of a convolution operation in a
//! single struct, enabling algorithm selection, GEMM dimension mapping, and
//! PTX kernel generation. It can be constructed manually or derived from
//! tensor and convolution descriptors via [`from_descriptors`](ConvProblem::from_descriptors).

use oxicuda_blas::GpuFloat;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};
use crate::types::{ConvAlgorithm, ConvolutionDescriptor, TensorDesc, TensorDescMut, TensorLayout};

use super::algo_select;

// ---------------------------------------------------------------------------
// ConvProblem
// ---------------------------------------------------------------------------

/// Complete description of a convolution problem.
///
/// Contains input/filter dimensions, spatial parameters (padding, stride,
/// dilation), grouping, data types, and memory layout. This struct serves
/// as the single source of truth for all downstream code (algorithm
/// selection, PTX generation, launch configuration).
#[derive(Debug, Clone)]
pub struct ConvProblem {
    /// Batch size (N).
    pub batch: u32,
    /// Input channel count (C).
    pub in_channels: u32,
    /// Spatial dimensions of the input (e.g. `[H, W]` or `[D, H, W]`).
    pub in_dims: Vec<u32>,
    /// Output channel count (K = number of filters).
    pub out_channels: u32,
    /// Spatial dimensions of the filter (e.g. `[R, S]` or `[T, R, S]`).
    pub filter_dims: Vec<u32>,
    /// Zero-padding per spatial dimension.
    pub padding: Vec<u32>,
    /// Stride per spatial dimension.
    pub stride: Vec<u32>,
    /// Dilation per spatial dimension.
    pub dilation: Vec<u32>,
    /// Number of groups (1 = standard conv, `in_channels` = depthwise).
    pub groups: u32,
    /// PTX data type for the input tensor.
    pub input_type: PtxType,
    /// PTX data type for the output tensor.
    pub output_type: PtxType,
    /// Memory layout of the tensors.
    pub layout: TensorLayout,
}

impl ConvProblem {
    /// Computes the output height.
    ///
    /// Uses the standard convolution output formula:
    /// `floor((H + 2*pad_h - dilation_h*(R-1) - 1) / stride_h) + 1`
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] if the padded input is smaller
    /// than the effective kernel size.
    pub fn output_h(&self) -> DnnResult<u32> {
        let h_idx = self.h_index();
        ConvolutionDescriptor::output_size(
            self.in_dims[h_idx],
            self.filter_dims[h_idx],
            self.padding[h_idx],
            self.stride[h_idx],
            self.dilation[h_idx],
        )
    }

    /// Computes the output width.
    ///
    /// # Errors
    ///
    /// Same as [`output_h`](Self::output_h).
    pub fn output_w(&self) -> DnnResult<u32> {
        let w_idx = self.w_index();
        ConvolutionDescriptor::output_size(
            self.in_dims[w_idx],
            self.filter_dims[w_idx],
            self.padding[w_idx],
            self.stride[w_idx],
            self.dilation[w_idx],
        )
    }

    /// Computes output spatial dimensions for all spatial axes.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension's computation fails.
    pub fn output_dims(&self) -> DnnResult<Vec<u32>> {
        self.in_dims
            .iter()
            .zip(self.filter_dims.iter())
            .zip(self.padding.iter())
            .zip(self.stride.iter())
            .zip(self.dilation.iter())
            .map(|((((&inp, &flt), &pad), &str_), &dil)| {
                ConvolutionDescriptor::output_size(inp, flt, pad, str_, dil)
            })
            .collect()
    }

    /// Returns `true` if this is a 1x1 convolution with unit stride and
    /// unit dilation (reduces to a plain GEMM).
    #[must_use]
    pub fn is_1x1(&self) -> bool {
        self.filter_dims.iter().all(|&d| d == 1)
            && self.stride.iter().all(|&s| s == 1)
            && self.dilation.iter().all(|&d| d == 1)
    }

    /// Returns `true` if this is a depthwise convolution
    /// (`groups == in_channels == out_channels`).
    #[must_use]
    pub fn is_depthwise(&self) -> bool {
        self.groups == self.in_channels && self.groups == self.out_channels
    }

    /// Returns `true` if this is a grouped (but not depthwise) convolution.
    #[must_use]
    pub fn is_grouped(&self) -> bool {
        self.groups > 1 && !self.is_depthwise()
    }

    /// Maps convolution dimensions to GEMM dimensions (M, N, K).
    ///
    /// - M = batch * product(output_spatial_dims) — output spatial points
    /// - N = out_channels — number of filters
    /// - K = (in_channels / groups) * product(filter_spatial_dims) — filter volume
    ///
    /// # Errors
    ///
    /// Returns an error if output dimension computation fails.
    pub fn conv_to_gemm_dims(&self) -> DnnResult<(u32, u32, u32)> {
        let out_dims = self.output_dims()?;
        let spatial_product: u32 = out_dims.iter().product();
        let gemm_m = self.batch.saturating_mul(spatial_product);
        let gemm_n = self.out_channels;
        let filter_volume: u32 = self.filter_dims.iter().product();
        let channels_per_group = self.in_channels / self.groups;
        let gemm_k = channels_per_group.saturating_mul(filter_volume);
        Ok((gemm_m, gemm_n, gemm_k))
    }

    /// Selects the best convolution algorithm for the given SM version.
    ///
    /// Delegates to [`algo_select::select_algorithm`].
    #[must_use]
    pub fn select_algorithm(&self, sm: SmVersion) -> ConvAlgorithm {
        algo_select::select_algorithm(self, sm)
    }

    /// Constructs a [`ConvProblem`] from tensor and convolution descriptors.
    ///
    /// Extracts dimensions from the NCHW or NHWC tensor descriptors and
    /// maps them into the canonical `ConvProblem` representation.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidDimension`] if tensor shapes are
    /// inconsistent (e.g. wrong number of dimensions for the layout).
    pub fn from_descriptors<T: GpuFloat>(
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
        output: &TensorDescMut<T>,
        conv_desc: &ConvolutionDescriptor,
    ) -> DnnResult<Self> {
        let layout = input.layout;
        let ndim = layout.expected_ndim();
        let spatial = layout.spatial_dims();

        if input.dims.len() != ndim {
            return Err(DnnError::InvalidDimension(format!(
                "input has {} dims, expected {ndim} for {:?} layout",
                input.dims.len(),
                layout
            )));
        }
        if filter.dims.len() != ndim {
            return Err(DnnError::InvalidDimension(format!(
                "filter has {} dims, expected {ndim} for {:?} layout",
                filter.dims.len(),
                layout
            )));
        }
        if output.dims.len() != ndim {
            return Err(DnnError::InvalidDimension(format!(
                "output has {} dims, expected {ndim} for {:?} layout",
                output.dims.len(),
                layout
            )));
        }
        if conv_desc.padding.len() != spatial {
            return Err(DnnError::InvalidDimension(format!(
                "conv_desc padding length {} != spatial dims {spatial}",
                conv_desc.padding.len()
            )));
        }

        // Extract dimensions based on layout.
        // NCHW: [N, C, H, W], filter: [K, C/g, R, S]
        // NHWC: [N, H, W, C], filter: [K, R, S, C/g]  (but we store canonically)
        let (batch, in_channels, in_dims) = Self::extract_input_dims(input)?;
        let (out_channels, filter_dims) = Self::extract_filter_dims(filter, spatial)?;

        Ok(Self {
            batch,
            in_channels,
            in_dims,
            out_channels,
            filter_dims,
            padding: conv_desc.padding.clone(),
            stride: conv_desc.stride.clone(),
            dilation: conv_desc.dilation.clone(),
            groups: conv_desc.groups,
            input_type: T::PTX_TYPE,
            output_type: T::PTX_TYPE,
            layout,
        })
    }

    /// Validates that all problem parameters are consistent.
    ///
    /// Checks channel divisibility by groups, matching spatial dim counts,
    /// and non-zero values for stride/dilation.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] or [`DnnError::InvalidDimension`]
    /// on inconsistency.
    pub fn validate(&self) -> DnnResult<()> {
        if self.groups == 0 {
            return Err(DnnError::InvalidArgument("groups must be >= 1".into()));
        }
        if self.in_channels % self.groups != 0 {
            return Err(DnnError::InvalidArgument(format!(
                "in_channels ({}) not divisible by groups ({})",
                self.in_channels, self.groups
            )));
        }
        if self.out_channels % self.groups != 0 {
            return Err(DnnError::InvalidArgument(format!(
                "out_channels ({}) not divisible by groups ({})",
                self.out_channels, self.groups
            )));
        }
        let n_spatial = self.in_dims.len();
        if self.filter_dims.len() != n_spatial {
            return Err(DnnError::InvalidDimension(format!(
                "filter spatial dims ({}) != input spatial dims ({n_spatial})",
                self.filter_dims.len()
            )));
        }
        if self.padding.len() != n_spatial
            || self.stride.len() != n_spatial
            || self.dilation.len() != n_spatial
        {
            return Err(DnnError::InvalidDimension(
                "padding/stride/dilation length mismatch with spatial dims".into(),
            ));
        }
        for (i, &s) in self.stride.iter().enumerate() {
            if s == 0 {
                return Err(DnnError::InvalidArgument(format!("stride[{i}] is zero")));
            }
        }
        for (i, &d) in self.dilation.iter().enumerate() {
            if d == 0 {
                return Err(DnnError::InvalidArgument(format!("dilation[{i}] is zero")));
            }
        }
        // Verify output dimensions are computable
        let _out_dims = self.output_dims()?;
        Ok(())
    }

    // -- Private helpers ------------------------------------------------------

    /// Index of height in the spatial dims vector (always 0 for 2D, 1 for 3D).
    fn h_index(&self) -> usize {
        if self.in_dims.len() == 3 { 1 } else { 0 }
    }

    /// Index of width in the spatial dims vector (always 1 for 2D, 2 for 3D).
    fn w_index(&self) -> usize {
        if self.in_dims.len() == 3 { 2 } else { 1 }
    }

    /// Extracts (batch, in_channels, spatial_dims) from an input tensor descriptor.
    fn extract_input_dims<T: GpuFloat>(input: &TensorDesc<T>) -> DnnResult<(u32, u32, Vec<u32>)> {
        // For both NCHW and NHWC, dims[0] = N, dims[1] = C (in our canonical form)
        let batch = input.dims[0];
        let in_channels = input.dims[1];
        let spatial = input.dims[2..].to_vec();
        Ok((batch, in_channels, spatial))
    }

    /// Extracts (out_channels, filter_spatial_dims) from a filter descriptor.
    fn extract_filter_dims<T: GpuFloat>(
        filter: &TensorDesc<T>,
        spatial_count: usize,
    ) -> DnnResult<(u32, Vec<u32>)> {
        // Filter dims: [K, C/g, R, S, ...] — first dim is out_channels
        if filter.dims.len() < 2 + spatial_count {
            return Err(DnnError::InvalidDimension(format!(
                "filter has {} dims, expected at least {}",
                filter.dims.len(),
                2 + spatial_count
            )));
        }
        let out_channels = filter.dims[0];
        let filter_spatial = filter.dims[2..2 + spatial_count].to_vec();
        Ok((out_channels, filter_spatial))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_problem_3x3() -> ConvProblem {
        ConvProblem {
            batch: 1,
            in_channels: 64,
            in_dims: vec![32, 32],
            out_channels: 128,
            filter_dims: vec![3, 3],
            padding: vec![1, 1],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }

    fn make_problem_1x1() -> ConvProblem {
        ConvProblem {
            batch: 2,
            in_channels: 256,
            in_dims: vec![16, 16],
            out_channels: 512,
            filter_dims: vec![1, 1],
            padding: vec![0, 0],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }

    fn make_depthwise() -> ConvProblem {
        ConvProblem {
            batch: 1,
            in_channels: 64,
            in_dims: vec![32, 32],
            out_channels: 64,
            filter_dims: vec![3, 3],
            padding: vec![1, 1],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 64,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        }
    }

    #[test]
    fn output_h_basic() {
        let p = make_problem_3x3();
        assert_eq!(p.output_h().ok(), Some(32));
    }

    #[test]
    fn output_w_basic() {
        let p = make_problem_3x3();
        assert_eq!(p.output_w().ok(), Some(32));
    }

    #[test]
    fn is_1x1_true() {
        assert!(make_problem_1x1().is_1x1());
    }

    #[test]
    fn is_1x1_false() {
        assert!(!make_problem_3x3().is_1x1());
    }

    #[test]
    fn is_depthwise_true() {
        assert!(make_depthwise().is_depthwise());
    }

    #[test]
    fn is_depthwise_false() {
        assert!(!make_problem_3x3().is_depthwise());
    }

    #[test]
    fn conv_to_gemm_dims_3x3() {
        let p = make_problem_3x3();
        let (m, n, k) = p.conv_to_gemm_dims().ok().unwrap_or((0, 0, 0));
        // M = 1 * 32 * 32 = 1024
        assert_eq!(m, 1024);
        // N = 128
        assert_eq!(n, 128);
        // K = 64 * 3 * 3 = 576
        assert_eq!(k, 576);
    }

    #[test]
    fn conv_to_gemm_dims_1x1() {
        let p = make_problem_1x1();
        let (m, n, k) = p.conv_to_gemm_dims().ok().unwrap_or((0, 0, 0));
        // M = 2 * 16 * 16 = 512
        assert_eq!(m, 512);
        // N = 512
        assert_eq!(n, 512);
        // K = 256 * 1 * 1 = 256
        assert_eq!(k, 256);
    }

    #[test]
    fn validate_ok() {
        assert!(make_problem_3x3().validate().is_ok());
    }

    #[test]
    fn validate_zero_groups() {
        let mut p = make_problem_3x3();
        p.groups = 0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_channels_not_divisible() {
        let mut p = make_problem_3x3();
        p.groups = 3; // 64 not divisible by 3
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_zero_stride() {
        let mut p = make_problem_3x3();
        p.stride[0] = 0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn output_dims_strided() {
        let mut p = make_problem_3x3();
        p.stride = vec![2, 2];
        let out = p.output_dims().ok().unwrap_or_default();
        // (32 + 2*1 - 1*(3-1) - 1)/2 + 1 = (32+2-2-1)/2+1 = 31/2+1 = 16
        assert_eq!(out, vec![16, 16]);
    }
}
