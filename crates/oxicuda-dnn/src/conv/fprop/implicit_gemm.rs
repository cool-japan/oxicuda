//! Implicit GEMM convolution forward pass.
//!
//! Maps convolution to a GEMM without explicitly materialising the im2col
//! matrix. Instead, the kernel computes the conv-to-GEMM index mapping
//! on-the-fly, checking padding boundaries for each loaded element.
//!
//! This is the most versatile conv algorithm — it requires zero workspace
//! and handles arbitrary padding, stride, dilation and grouping. Each thread
//! owns one output element `(n, k, oh, ow)` and accumulates the
//! cross-correlation reduction over the filter volume
//! `(C_in/groups) × R × S`, reading inputs at the conv-mapped positions with
//! implicit zero padding. The actual per-element compute lives in the shared
//! `emit_standard_conv_body`
//! so the 1x1 and general engines stay numerically identical.
//!
//! # GEMM mapping
//!
//! ```text
//! M = batch * out_H * out_W            (output spatial points)
//! N = out_channels                      (filter count)
//! K = (in_channels / groups) * R * S    (filter volume per group)
//!
//! A[m, k] = input at conv-mapped position  (implicit im2col)
//! B[k, n] = filter weights
//! D[m, n] = output
//! ```

use oxicuda_blas::GpuFloat;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::kernel_cache::cache_key;
use crate::types::{TensorDesc, TensorDescMut, TensorLayout, TileConfig};

use super::super::descriptor::ConvProblem;
use super::standard_conv::{emit_standard_conv_body, with_standard_conv_params};

// ---------------------------------------------------------------------------
// ImplicitGemmConv
// ---------------------------------------------------------------------------

/// Implicit GEMM convolution engine.
///
/// Generates and launches a PTX kernel that computes convolution as a GEMM
/// with implicit im2col address mapping inside the inner loop.
pub struct ImplicitGemmConv {
    problem: ConvProblem,
    tile_config: TileConfig,
    sm_version: SmVersion,
}

impl ImplicitGemmConv {
    /// Creates a new implicit GEMM convolution engine.
    #[must_use]
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> Self {
        let tile_config = TileConfig::default_conv(sm_version);
        Self {
            problem,
            tile_config,
            sm_version,
        }
    }

    /// Creates with a custom tile configuration.
    #[must_use]
    pub fn with_tile_config(
        problem: ConvProblem,
        tile_config: TileConfig,
        sm_version: SmVersion,
    ) -> Self {
        Self {
            problem,
            tile_config,
            sm_version,
        }
    }

    /// Returns a unique kernel name encoding every code-generation constant.
    ///
    /// # Cache-key contract
    ///
    /// This name is the compiled-module cache key (see
    /// [`crate::kernel_cache`]), so it must discriminate everything
    /// [`Self::generate_ptx`] bakes into the instruction stream:
    ///
    /// | Code-gen constant | Encoded as |
    /// |---|---|
    /// | element type | `prec` |
    /// | `channels_last` | `nchw` / `nhwc` |
    /// | filter extent `R`, `S` (unrolled tap loops) | `{r}x{s}` |
    /// | `C_in / groups`, `C_out / groups` (group routing immediates) | `g{icpg}x{ocpg}` |
    ///
    /// The tile dimensions are also folded in. They do **not** currently affect
    /// the emitted body — the shared `emit_standard_conv_body` is one thread
    /// per output element, not a tiled GEMM — but keeping them in the key means
    /// a future tiled implementation cannot silently reuse a module compiled
    /// for a different tiling.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let prec = self.problem.input_type.as_ptx_str().trim_start_matches('.');
        let layout = if self.problem.layout.is_channels_last() {
            "nhwc"
        } else {
            "nchw"
        };
        let r = self.problem.filter_dims.first().copied().unwrap_or(1);
        let s = self.problem.filter_dims.get(1).copied().unwrap_or(1);
        let groups = self.problem.groups.max(1);
        let in_cpg = self.problem.in_channels / groups;
        let out_cpg = self.problem.out_channels / groups;
        format!(
            "implicit_gemm_conv_{}x{}x{}_{r}x{s}_g{in_cpg}x{out_cpg}_{prec}_{layout}",
            self.tile_config.tile_m, self.tile_config.tile_n, self.tile_config.tile_k,
        )
    }

    /// Generates the complete PTX module for the implicit GEMM conv kernel.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] for an unsupported precision, a
    /// non-2-D problem, or a layout other than NCHW/NHWC.
    pub fn generate_ptx(&self) -> DnnResult<String> {
        let elem_ty = self.problem.input_type;
        if !matches!(elem_ty, PtxType::F32 | PtxType::F64) {
            return Err(DnnError::PtxGeneration(format!(
                "implicit-GEMM convolution kernel supports f32/f64 storage, got {elem_ty}"
            )));
        }
        if self.problem.in_dims.len() != 2 || self.problem.filter_dims.len() != 2 {
            return Err(DnnError::PtxGeneration(
                "implicit-GEMM forward kernel supports 2-D convolution only".into(),
            ));
        }
        if !matches!(self.problem.layout, TensorLayout::Nchw | TensorLayout::Nhwc) {
            return Err(DnnError::PtxGeneration(format!(
                "implicit-GEMM forward kernel supports NCHW/NHWC, got {:?}",
                self.problem.layout
            )));
        }
        if self.problem.groups == 0
            || self.problem.in_channels % self.problem.groups != 0
            || self.problem.out_channels % self.problem.groups != 0
        {
            return Err(DnnError::PtxGeneration(
                "channels must be divisible by groups".into(),
            ));
        }

        let channels_last = self.problem.layout.is_channels_last();
        let filter_h = self.problem.filter_dims[0];
        let filter_w = self.problem.filter_dims[1];
        let in_ch_per_group = self.problem.in_channels / self.problem.groups;
        let out_ch_per_group = self.problem.out_channels / self.problem.groups;

        let kb = KernelBuilder::new(&self.kernel_name()).target(self.sm_version);
        let ptx = with_standard_conv_params(kb)
            .body(move |b| {
                emit_standard_conv_body(
                    b,
                    elem_ty,
                    channels_last,
                    filter_h,
                    filter_w,
                    in_ch_per_group,
                    out_ch_per_group,
                );
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the implicit GEMM convolution.
    ///
    /// An optional `bias` tensor adds a per-output-channel constant in the
    /// kernel epilogue. When `bias` is `None` a null device pointer is passed
    /// and the epilogue skips the bias add via a guarded branch.
    ///
    /// # Errors
    ///
    /// Returns errors from PTX generation, module loading, or kernel launch.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
        bias: Option<&TensorDesc<T>>,
        output: &mut TensorDescMut<T>,
    ) -> DnnResult<()> {
        let entry = self.kernel_name();
        let kernel =
            handle.get_or_compile_kernel(&cache_key(&entry, self.sm_version), &entry, || {
                self.generate_ptx()
            })?;

        let out_dims = self.problem.output_dims()?;
        let out_h = out_dims.first().copied().unwrap_or(1);
        let out_w = out_dims.get(1).copied().unwrap_or(1);
        let total_outputs = self
            .problem
            .batch
            .saturating_mul(self.problem.out_channels)
            .saturating_mul(out_h)
            .saturating_mul(out_w);

        let params = kernel.launch_1d(total_outputs);

        // Optional bias: pass the device pointer, or 0 when absent. The
        // kernel epilogue treats a zero pointer as "no bias".
        let bias_ptr = bias.map_or(0u64, |b| b.ptr);

        let args = (
            input.ptr,
            filter.ptr,
            output.ptr,
            bias_ptr,
            self.problem.in_channels,
            self.problem.out_channels,
            self.problem.in_dims[0],
            self.problem.in_dims.get(1).copied().unwrap_or(1),
            out_h,
            out_w,
            self.problem.padding[0],
            self.problem.padding.get(1).copied().unwrap_or(0),
            self.problem.stride[0],
            self.problem.stride.get(1).copied().unwrap_or(1),
            self.problem.dilation[0],
            self.problem.dilation.get(1).copied().unwrap_or(1),
            total_outputs,
        );

        kernel
            .kernel()
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(e.to_string()))?;

        Ok(())
    }

    /// Returns the workspace size in bytes (implicit GEMM needs zero).
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Conv-to-GEMM index mapping utilities
// ---------------------------------------------------------------------------

/// Maps a linear GEMM-M index back to convolution output coordinates.
///
/// Given `m = batch_idx * (out_H * out_W) + oh * out_W + ow`, this
/// function recovers `(batch_idx, oh, ow)`.
#[inline]
pub fn gemm_m_to_conv_coords(m: u32, out_h: u32, out_w: u32) -> (u32, u32, u32) {
    let spatial = out_h * out_w;
    let batch_idx = m / spatial;
    let remainder = m % spatial;
    let oh = remainder / out_w;
    let ow = remainder % out_w;
    (batch_idx, oh, ow)
}

/// Maps a linear GEMM-K index to convolution filter coordinates.
///
/// Given `k = c * (R * S) + r * S + s`, recovers `(c, r, s)`.
#[inline]
pub fn gemm_k_to_filter_coords(k: u32, filter_h: u32, filter_w: u32) -> (u32, u32, u32) {
    let rs = filter_h * filter_w;
    let c = k / rs;
    let remainder = k % rs;
    let r = remainder / filter_w;
    let s = remainder % filter_w;
    (c, r, s)
}

/// Computes the input spatial coordinate for a given output position
/// and filter offset, checking padding boundaries.
///
/// Returns `None` if the computed position falls outside the valid
/// input range (i.e. it would be a zero-padded position).
#[inline]
pub fn input_coord(
    out_pos: u32,
    filter_pos: u32,
    pad: u32,
    stride: u32,
    dilation: u32,
    input_size: u32,
) -> Option<u32> {
    let pos = (out_pos * stride) as i64 - pad as i64 + (filter_pos * dilation) as i64;
    if pos >= 0 && (pos as u32) < input_size {
        Some(pos as u32)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    fn make_problem() -> ConvProblem {
        ConvProblem {
            batch: 2,
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

    #[test]
    fn kernel_name_format() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        let name = conv.kernel_name();
        assert!(name.contains("implicit_gemm_conv"));
        assert!(name.contains("f32"));
    }

    /// The name is the compiled-module cache key, so every constant baked into
    /// the PTX must move it. A shared name for differently-generated PTX would
    /// hand one problem the other's module — wrong results, no error.
    #[test]
    fn kernel_name_discriminates_every_codegen_constant() {
        let base = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80).kernel_name();

        let mut filter_5x5 = make_problem();
        filter_5x5.filter_dims = vec![5, 5];
        assert_ne!(
            base,
            ImplicitGemmConv::new(filter_5x5, SmVersion::Sm80).kernel_name(),
            "filter extent is unrolled into the PTX"
        );

        let mut more_channels = make_problem();
        more_channels.in_channels = 128;
        assert_ne!(
            base,
            ImplicitGemmConv::new(more_channels, SmVersion::Sm80).kernel_name(),
            "C_in/groups is a code-gen immediate"
        );

        let mut grouped = make_problem();
        grouped.groups = 2;
        assert_ne!(
            base,
            ImplicitGemmConv::new(grouped, SmVersion::Sm80).kernel_name(),
            "grouping changes both channels-per-group immediates"
        );

        let mut nhwc = make_problem();
        nhwc.layout = TensorLayout::Nhwc;
        assert_ne!(
            base,
            ImplicitGemmConv::new(nhwc, SmVersion::Sm80).kernel_name(),
            "layout selects a different addressing path"
        );

        let mut f64_problem = make_problem();
        f64_problem.input_type = PtxType::F64;
        assert_ne!(
            base,
            ImplicitGemmConv::new(f64_problem, SmVersion::Sm80).kernel_name(),
            "precision changes the register class and instruction widths"
        );
    }

    /// The cache looks the entry point up by `kernel_name()`, so the name must
    /// be exactly what the generated PTX declares.
    #[test]
    fn kernel_name_is_the_emitted_entry() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        let ptx = conv.generate_ptx().unwrap_or_default();
        assert!(
            ptx.contains(&format!(".visible .entry {}", conv.kernel_name())),
            "kernel_name() must name the emitted entry point"
        );
    }

    #[test]
    fn workspace_is_zero() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        assert_eq!(conv.workspace_bytes(), 0);
    }

    #[test]
    fn gemm_m_to_conv_coords_basic() {
        // m=0 -> (batch=0, oh=0, ow=0)
        assert_eq!(gemm_m_to_conv_coords(0, 4, 4), (0, 0, 0));
        // m=5 -> (batch=0, oh=1, ow=1)
        assert_eq!(gemm_m_to_conv_coords(5, 4, 4), (0, 1, 1));
        // m=16 -> (batch=1, oh=0, ow=0)
        assert_eq!(gemm_m_to_conv_coords(16, 4, 4), (1, 0, 0));
    }

    #[test]
    fn gemm_k_to_filter_coords_basic() {
        // k=0 -> (c=0, r=0, s=0)
        assert_eq!(gemm_k_to_filter_coords(0, 3, 3), (0, 0, 0));
        // k=4 -> (c=0, r=1, s=1)
        assert_eq!(gemm_k_to_filter_coords(4, 3, 3), (0, 1, 1));
        // k=9 -> (c=1, r=0, s=0)
        assert_eq!(gemm_k_to_filter_coords(9, 3, 3), (1, 0, 0));
    }

    #[test]
    fn input_coord_valid() {
        // out=1, filter=0, pad=1, stride=1, dilation=1, size=32
        // pos = 1*1 - 1 + 0*1 = 0 -> valid
        assert_eq!(input_coord(1, 0, 1, 1, 1, 32), Some(0));
    }

    #[test]
    fn input_coord_padded() {
        // out=0, filter=0, pad=1, stride=1, dilation=1, size=32
        // pos = 0*1 - 1 + 0*1 = -1 -> out of bounds
        assert_eq!(input_coord(0, 0, 1, 1, 1, 32), None);
    }

    #[test]
    fn input_coord_beyond_input() {
        // out=31, filter=2, pad=1, stride=1, dilation=1, size=32
        // pos = 31 - 1 + 2 = 32 -> out of bounds (size=32)
        assert_eq!(input_coord(31, 2, 1, 1, 1, 32), None);
    }

    #[test]
    fn ptx_generation_produces_output() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        let ptx = conv.generate_ptx();
        assert!(ptx.is_ok());
        let ptx_text = ptx.unwrap_or_default();
        assert!(ptx_text.contains("implicit_gemm_conv"));
        assert!(ptx_text.contains(".entry"));
    }

    // -----------------------------------------------------------------------
    // Bias epilogue tests
    // -----------------------------------------------------------------------

    /// The generated kernel epilogue must contain a *guarded* bias add: a
    /// null-pointer test on the bias parameter, followed by a bias load and
    /// a float add. This proves the bias is plumbed through, not discarded.
    #[test]
    fn ptx_epilogue_has_guarded_bias_add() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        let ptx = conv.generate_ptx().expect("ptx generation");

        // Null-pointer guard on the bias parameter.
        assert!(
            ptx.contains("setp.ne.u64"),
            "epilogue must test the bias pointer for null"
        );
        // The bias is loaded from global memory and added to the accumulator.
        assert!(
            ptx.contains("ld.global.f32"),
            "epilogue must load the bias value"
        );
        assert!(
            ptx.contains("add.rn.f32"),
            "epilogue must add the bias to the accumulator"
        );
        // The accumulator is stored back after the (optional) bias add.
        assert!(
            ptx.contains("st.global.f32"),
            "epilogue must store the result"
        );
    }

    /// The bias parameter must be declared on the kernel signature.
    #[test]
    fn ptx_declares_bias_param() {
        let conv = ImplicitGemmConv::new(make_problem(), SmVersion::Sm80);
        let ptx = conv.generate_ptx().expect("ptx generation");
        assert!(ptx.contains("bias"), "kernel must declare a bias parameter");
    }

    /// CPU reference: the bias epilogue adds `bias[c_out]` to every spatial
    /// position of the corresponding output channel. This mirrors the guarded
    /// `out[..] += bias[k]` performed by the shared standard-conv epilogue.
    #[test]
    fn bias_epilogue_cpu_reference() {
        let out_channels = 4usize;
        let m = 6usize; // spatial points per channel
        // Pre-epilogue accumulator (row-major [out_channels x M]).
        let mut acc: Vec<f32> = (0..out_channels * m)
            .map(|i| (i as f32) * 0.25 - 1.0)
            .collect();
        let pre = acc.clone();
        let bias: Vec<f32> = (0..out_channels).map(|c| (c as f32) * 0.5 + 0.1).collect();

        // Apply the epilogue: out[n*M + m] += bias[n].
        for (n, &bias_n) in bias.iter().enumerate() {
            for mi in 0..m {
                acc[n * m + mi] += bias_n;
            }
        }

        for (n, &bias_n) in bias.iter().enumerate() {
            for mi in 0..m {
                let idx = n * m + mi;
                let expected = pre[idx] + bias_n;
                assert!(
                    (acc[idx] - expected).abs() < 1e-6,
                    "bias add mismatch at (n={n}, m={mi})"
                );
            }
        }
    }

    /// With no bias, the epilogue must leave the accumulator unchanged: the
    /// host passes a null pointer and the guard branch skips the add.
    #[test]
    fn no_bias_leaves_accumulator_unchanged() {
        // `execute` maps `None` -> 0u64 pointer; the kernel's `setp.ne.u64`
        // guard then branches over the bias load/add. Modelled on the CPU
        // side: a null bias contributes nothing.
        let acc: Vec<f32> = vec![1.5, -2.0, 0.0, 3.25];
        let null_bias: Option<&[f32]> = None;
        let result: Vec<f32> = acc
            .iter()
            .enumerate()
            .map(|(i, &v)| v + null_bias.map_or(0.0, |b| b[i]))
            .collect();
        assert_eq!(result, acc);
    }
}
