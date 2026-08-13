//! Direct convolution kernels.
//!
//! Provides optimised implementations for two special cases:
//!
//! 1. **1x1 convolution** — reduces to a plain GEMM since there is no
//!    spatial filtering (every input pixel maps directly to one output pixel).
//!
//! 2. **Depthwise convolution** — each input channel is convolved
//!    independently with its own filter. There is no cross-channel mixing,
//!    so this cannot be expressed as a single GEMM. Instead, a dedicated
//!    kernel assigns one thread per output pixel per channel.
//!
//! Both cases are common in modern architectures (MobileNet, EfficientNet,
//! ResNet bottleneck blocks).

use oxicuda_blas::GpuFloat;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::{BodyBuilder, KernelBuilder};
use oxicuda_ptx::ir::{PtxType, Register};

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::kernel_cache::cache_key;
use crate::types::{TensorDesc, TensorDescMut, TensorLayout};

use super::super::descriptor::ConvProblem;
use super::standard_conv::{emit_standard_conv_body, with_standard_conv_params};

// ---------------------------------------------------------------------------
// 1x1 Convolution (= GEMM)
// ---------------------------------------------------------------------------

/// 1x1 convolution engine.
///
/// Reshapes the problem as a pure matrix multiply:
/// - A: input reshaped to `[N*H*W, C]`
/// - B: filter reshaped to `[C, K]`
/// - C: output reshaped to `[N*H*W, K]`
pub struct Conv1x1 {
    problem: ConvProblem,
    sm_version: SmVersion,
}

impl Conv1x1 {
    /// Creates a new 1x1 convolution engine.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if the filter is not 1x1.
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> DnnResult<Self> {
        if !problem.is_1x1() {
            return Err(DnnError::InvalidArgument(
                "Conv1x1 requires 1x1 filter with unit stride/dilation".into(),
            ));
        }
        Ok(Self {
            problem,
            sm_version,
        })
    }

    /// Returns the kernel name encoding precision, layout, and the
    /// channels-per-group counts.
    ///
    /// # Why the group counts are in the name
    ///
    /// [`Self::generate_ptx`] bakes `C_in / groups` and `C_out / groups` into
    /// the instruction stream as immediates (they drive the group routing and
    /// the filter row stride), so two problems that differ only in channel
    /// count produce *different* PTX. The name is the compiled-module cache key
    /// (see `crate::kernel_cache`), so leaving them out would let one
    /// problem's module be handed to another shape — a silent wrong-results
    /// bug, not a compile error. Encoding them here keeps the key faithful and
    /// makes the emitted PTX self-describing.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let prec = self.problem.input_type.as_ptx_str().trim_start_matches('.');
        let layout = if self.problem.layout.is_channels_last() {
            "nhwc"
        } else {
            "nchw"
        };
        let (in_cpg, out_cpg) = self.channels_per_group();
        format!("conv1x1_{prec}_{layout}_g{in_cpg}x{out_cpg}")
    }

    /// `(C_in / groups, C_out / groups)`, saturating at zero groups so the
    /// name is always well-formed; `generate_ptx` rejects the degenerate case
    /// with a proper error.
    fn channels_per_group(&self) -> (u32, u32) {
        let groups = self.problem.groups.max(1);
        (
            self.problem.in_channels / groups,
            self.problem.out_channels / groups,
        )
    }

    /// Generates the PTX for the 1x1 convolution kernel.
    ///
    /// A 1x1 convolution is the `R = S = 1` special case of the standard
    /// per-output-element cross-correlation: each output `(n, k, oh, ow)` is the
    /// channel-reduced dot product `Σ_c in[n, c, oh, ow] * filter[k, c]`. It
    /// shares the exact compute path with the general implicit-GEMM kernel.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] for an unsupported precision, a
    /// non-2-D problem, or a layout other than NCHW/NHWC.
    pub fn generate_ptx(&self) -> DnnResult<String> {
        let elem_ty = self.problem.input_type;
        if !matches!(elem_ty, PtxType::F32 | PtxType::F64) {
            return Err(DnnError::PtxGeneration(format!(
                "1x1 convolution kernel supports f32/f64 storage, got {elem_ty}"
            )));
        }
        if self.problem.in_dims.len() != 2 || self.problem.filter_dims.len() != 2 {
            return Err(DnnError::PtxGeneration(
                "1x1 forward kernel supports 2-D convolution only".into(),
            ));
        }
        if !matches!(self.problem.layout, TensorLayout::Nchw | TensorLayout::Nhwc) {
            return Err(DnnError::PtxGeneration(format!(
                "1x1 forward kernel supports NCHW/NHWC, got {:?}",
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
        let in_ch_per_group = self.problem.in_channels / self.problem.groups;
        let out_ch_per_group = self.problem.out_channels / self.problem.groups;

        let kb = KernelBuilder::new(&self.kernel_name()).target(self.sm_version);
        let ptx = with_standard_conv_params(kb)
            .body(move |b| {
                emit_standard_conv_body(
                    b,
                    elem_ty,
                    channels_last,
                    1,
                    1,
                    in_ch_per_group,
                    out_ch_per_group,
                );
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the 1x1 convolution.
    ///
    /// Launches one thread per output element; each computes the per-spatial
    /// channel dot product `out[n, k, oh, ow] = Σ_c in[n, c, oh, ow]·filter[k, c]`
    /// (cross-correlation, no kernel flip), honouring padding/groups exactly as
    /// the descriptor specifies.
    ///
    /// # Performance
    ///
    /// The module is fetched from the handle's compiled-kernel cache, so a
    /// repeated call (the common case in a per-frame inference pipeline) skips
    /// both PTX generation and the `cuModuleLoadData` JIT. The launch geometry
    /// comes from an occupancy query against the compiled function rather than
    /// a hard-coded block size; the grid still covers every output element
    /// (see `CachedKernel::launch_1d`).
    ///
    /// # Errors
    ///
    /// Returns errors from PTX generation, module loading, or kernel launch.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
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

        let args = (
            input.ptr,
            filter.ptr,
            output.ptr,
            0u64, // bias: 1x1 forward has no bias term
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

    /// Workspace required (zero for 1x1).
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Depthwise Convolution
// ---------------------------------------------------------------------------

/// Depthwise convolution engine.
///
/// Each channel has its own independent filter. The kernel assigns one
/// thread per output pixel per channel, with filter weights stored in
/// registers (for small filters like 3x3 = 9 values).
pub struct DepthwiseConv {
    problem: ConvProblem,
    sm_version: SmVersion,
}

impl DepthwiseConv {
    /// Creates a new depthwise convolution engine.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::InvalidArgument`] if the problem is not depthwise.
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> DnnResult<Self> {
        if !problem.is_depthwise() {
            return Err(DnnError::InvalidArgument(
                "DepthwiseConv requires groups == in_channels == out_channels".into(),
            ));
        }
        Ok(Self {
            problem,
            sm_version,
        })
    }

    /// Returns the kernel name.
    ///
    /// [`Self::generate_ptx`] bakes exactly three things into the instruction
    /// stream — the filter extent `R`, `S` (the tap loops are unrolled) and the
    /// element type — and all three appear here, so this name is a complete
    /// compiled-module cache key for a given target architecture. Everything
    /// else (dims, padding, stride, dilation) is a runtime kernel parameter.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let prec = self.problem.input_type.as_ptx_str().trim_start_matches('.');
        let r = self.problem.filter_dims.first().copied().unwrap_or(0);
        let s = self.problem.filter_dims.get(1).copied().unwrap_or(0);
        format!("depthwise_conv_{r}x{s}_{prec}")
    }

    /// Generates PTX for the depthwise convolution kernel.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on failure.
    pub fn generate_ptx(&self) -> DnnResult<String> {
        let elem_ty = self.problem.input_type;
        if !matches!(elem_ty, PtxType::F32 | PtxType::F64) {
            return Err(DnnError::PtxGeneration(format!(
                "depthwise convolution kernel supports f32/f64 storage, got {elem_ty}"
            )));
        }
        let filter_h = self.problem.filter_dims.first().copied().unwrap_or(1);
        let filter_w = self.problem.filter_dims.get(1).copied().unwrap_or(1);

        let ptx = KernelBuilder::new(&self.kernel_name())
            .target(self.sm_version)
            .param("input", PtxType::U64)
            .param("filter", PtxType::U64)
            .param("output", PtxType::U64)
            .param("bias", PtxType::U64)
            .param("batch_size", PtxType::U32)
            .param("channels", PtxType::U32)
            .param("in_h", PtxType::U32)
            .param("in_w", PtxType::U32)
            .param("filter_h", PtxType::U32)
            .param("filter_w", PtxType::U32)
            .param("out_h", PtxType::U32)
            .param("out_w", PtxType::U32)
            .param("pad_h", PtxType::U32)
            .param("pad_w", PtxType::U32)
            .param("stride_h", PtxType::U32)
            .param("stride_w", PtxType::U32)
            .param("dilation_h", PtxType::U32)
            .param("dilation_w", PtxType::U32)
            .param("total_outputs", PtxType::U32)
            .body(move |b| {
                emit_depthwise_body(b, filter_h, filter_w, elem_ty);
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the depthwise convolution.
    ///
    /// # Errors
    ///
    /// Returns errors from PTX generation, module loading, or launch.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
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
        let total_outputs = self.problem.batch * self.problem.in_channels * out_h * out_w;

        let params = kernel.launch_1d(total_outputs);

        let args = (
            input.ptr,
            filter.ptr,
            output.ptr,
            0u64, // bias
            self.problem.batch,
            self.problem.in_channels,
            self.problem.in_dims[0],
            self.problem.in_dims.get(1).copied().unwrap_or(1),
            self.problem.filter_dims[0],
            self.problem.filter_dims.get(1).copied().unwrap_or(1),
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

    /// Workspace required (zero for depthwise).
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        0
    }
}

/// Standalone depthwise body emitter for the `'static` closure requirement.
///
/// Implements a real per-thread depthwise cross-correlation (the cuDNN
/// convention — **no** 180° kernel flip): each thread owns one output pixel
/// `(n, c, oh, ow)` and accumulates
///
/// ```text
/// out[n, c, oh, ow] = Σ_r Σ_s in[n, c, ih, iw] * filter[c, r, s]
/// ih = oh*stride_h - pad_h + r*dilation_h
/// iw = ow*stride_w - pad_w + s*dilation_w
/// ```
///
/// over the in-bounds `(ih, iw)` positions (out-of-range taps contribute
/// zero, i.e. implicit zero padding). The filter spatial extent
/// (`filter_h × filter_w`) is known at PTX-generation time, so the tap loops
/// are unrolled here; all remaining geometry (dims, padding, stride,
/// dilation) is read from kernel parameters so the padded / strided / dilated
/// paths stay consistent with the pad=0 / stride=1 / dilation=1 case.
fn emit_depthwise_body(b: &mut BodyBuilder<'_>, filter_h: u32, filter_w: u32, elem_ty: PtxType) {
    b.comment("=== Depthwise Convolution (cross-correlation, forward) ===");
    b.comment("Each thread computes one output pixel for one channel.");

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_outputs");
    let gid_cmp = gid.clone();
    b.if_lt_u32(gid_cmp, total, move |b| {
        emit_depthwise_pixel(b, filter_h, filter_w, elem_ty, gid);
    });

    b.ret();
}

/// Emits the body executed by a single in-range depthwise thread.
fn emit_depthwise_pixel(
    b: &mut BodyBuilder<'_>,
    filter_h: u32,
    filter_w: u32,
    elem_ty: PtxType,
    gid: Register,
) {
    let elem_bytes = elem_ty.size_bytes() as u32;

    // Tensor base pointers (bias is intentionally not consumed: conv_forward
    // has no bias term; the parameter is retained for ABI stability).
    let input_ptr = b.load_param_u64("input");
    let filter_ptr = b.load_param_u64("filter");
    let output_ptr = b.load_param_u64("output");

    // Geometry parameters.
    let channels = b.load_param_u32("channels");
    let in_h = b.load_param_u32("in_h");
    let in_w = b.load_param_u32("in_w");
    let out_h = b.load_param_u32("out_h");
    let out_w = b.load_param_u32("out_w");
    let pad_h = b.load_param_u32("pad_h");
    let pad_w = b.load_param_u32("pad_w");
    let stride_h = b.load_param_u32("stride_h");
    let stride_w = b.load_param_u32("stride_w");
    let dil_h = b.load_param_u32("dilation_h");
    let dil_w = b.load_param_u32("dilation_w");

    // Decompose the linear output index (NCHW order) into (n, c, oh, ow):
    //   gid = ((n*C + c)*P + oh)*Q + ow
    b.comment("Decompose linear output index -> (n, c, oh, ow)");
    let ow = b.alloc_reg(PtxType::U32);
    let t1 = b.alloc_reg(PtxType::U32);
    let oh = b.alloc_reg(PtxType::U32);
    let t2 = b.alloc_reg(PtxType::U32);
    let c = b.alloc_reg(PtxType::U32);
    let n = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {ow}, {gid}, {out_w};"));
    b.raw_ptx(&format!("div.u32 {t1}, {gid}, {out_w};"));
    b.raw_ptx(&format!("rem.u32 {oh}, {t1}, {out_h};"));
    b.raw_ptx(&format!("div.u32 {t2}, {t1}, {out_h};"));
    b.raw_ptx(&format!("rem.u32 {c}, {t2}, {channels};"));
    b.raw_ptx(&format!("div.u32 {n}, {t2}, {channels};"));

    // nc = n*C + c : channel-major offset shared by every input tap.
    let nc = b.mad_lo_u32(n, channels, c.clone());

    // c_base = c * (R*S) : start of this channel's filter in [C, 1, R, S].
    let rxs = filter_h.saturating_mul(filter_w);
    let rxs_reg = b.mov_imm_u32(rxs);
    let c_base = b.mul_lo_u32(c, rxs_reg);

    // Accumulator and a reusable zero of the same float width (a single float
    // type per kernel — F32 and F64 share the %f register class but differ in
    // declared width, so they must never be mixed).
    let mut acc = zero_float(b, elem_ty);
    let fzero = zero_float(b, elem_ty);

    b.comment("Unrolled cross-correlation over the filter window");
    for r in 0..filter_h {
        // ih = oh*stride_h - pad_h + r*dilation_h  (signed).
        let r_reg = b.mov_imm_u32(r);
        let rdil = b.mul_lo_u32(dil_h.clone(), r_reg);
        let ih_pos = b.mad_lo_u32(oh.clone(), stride_h.clone(), rdil);
        let ih = b.alloc_reg(PtxType::S32);
        b.raw_ptx(&format!("sub.s32 {ih}, {ih_pos}, {pad_h};"));
        // Unsigned compare doubles as a 0 <= ih < H bounds check: a negative
        // ih reinterpreted as u32 is huge and fails the upper bound.
        let p_ih = b.alloc_reg(PtxType::Pred);
        b.raw_ptx(&format!("setp.lo.u32 {p_ih}, {ih}, {in_h};"));

        for s in 0..filter_w {
            let s_reg = b.mov_imm_u32(s);
            let sdil = b.mul_lo_u32(dil_w.clone(), s_reg);
            let iw_pos = b.mad_lo_u32(ow.clone(), stride_w.clone(), sdil);
            let iw = b.alloc_reg(PtxType::S32);
            b.raw_ptx(&format!("sub.s32 {iw}, {iw_pos}, {pad_w};"));
            let p_iw = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lo.u32 {p_iw}, {iw}, {in_w};"));
            let p_valid = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("and.pred {p_valid}, {p_ih}, {p_iw};"));

            // in_idx = (nc*H + ih)*W + iw, clamped to 0 when the tap is out of
            // range so the load address is always valid; the value is masked.
            let row = b.mad_lo_u32(nc.clone(), in_h.clone(), ih.clone());
            let in_idx = b.mad_lo_u32(row, in_w.clone(), iw.clone());
            let zero_idx = b.mov_imm_u32(0);
            let safe_idx = b.selp(PtxType::U32, in_idx, zero_idx, p_valid.clone());
            let in_addr = b.byte_offset_addr(input_ptr.clone(), safe_idx, elem_bytes);
            let raw_val = load_float(b, elem_ty, in_addr);
            let val = b.selp(elem_ty, raw_val, fzero.clone(), p_valid);

            // filter index = c_base + r*S + s (always in range).
            let f_off = r.saturating_mul(filter_w).saturating_add(s);
            let f_off_reg = b.mov_imm_u32(f_off);
            let filt_idx = b.add_u32(c_base.clone(), f_off_reg);
            let filt_addr = b.byte_offset_addr(filter_ptr.clone(), filt_idx, elem_bytes);
            let weight = load_float(b, elem_ty, filt_addr);

            acc = fma_float(b, elem_ty, val, weight, acc);
        }
    }

    // out[gid] = acc (the linear gid already addresses the NCHW output).
    let out_addr = b.byte_offset_addr(output_ptr, gid, elem_bytes);
    store_float(b, elem_ty, out_addr, acc);
}

/// Allocates a fresh `+0.0` register of the kernel's float width.
fn zero_float(b: &mut BodyBuilder<'_>, elem_ty: PtxType) -> Register {
    let z = b.alloc_reg(elem_ty);
    if elem_ty == PtxType::F64 {
        b.raw_ptx(&format!("mov.f64 {z}, 0d0000000000000000;"));
    } else {
        b.raw_ptx(&format!("mov.f32 {z}, 0f00000000;"));
    }
    z
}

/// Loads one scalar of the kernel's float width from global memory.
fn load_float(b: &mut BodyBuilder<'_>, elem_ty: PtxType, addr: Register) -> Register {
    if elem_ty == PtxType::F64 {
        b.load_global_f64(addr)
    } else {
        b.load_global_f32(addr)
    }
}

/// Stores one scalar of the kernel's float width to global memory.
fn store_float(b: &mut BodyBuilder<'_>, elem_ty: PtxType, addr: Register, val: Register) {
    if elem_ty == PtxType::F64 {
        b.store_global_f64(addr, val);
    } else {
        b.store_global_f32(addr, val);
    }
}

/// Fused multiply-add `acc + a*w` at the kernel's float width.
fn fma_float(
    b: &mut BodyBuilder<'_>,
    elem_ty: PtxType,
    a: Register,
    w: Register,
    acc: Register,
) -> Register {
    if elem_ty == PtxType::F64 {
        b.fma_f64(a, w, acc)
    } else {
        b.fma_f32(a, w, acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    fn make_1x1_problem() -> ConvProblem {
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

    fn make_depthwise_problem() -> ConvProblem {
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
    fn conv1x1_rejects_non_1x1() {
        let mut p = make_1x1_problem();
        p.filter_dims = vec![3, 3];
        assert!(Conv1x1::new(p, SmVersion::Sm80).is_err());
    }

    #[test]
    fn conv1x1_workspace_zero() {
        let c = Conv1x1::new(make_1x1_problem(), SmVersion::Sm80);
        assert!(c.is_ok());
        if let Ok(conv) = c {
            assert_eq!(conv.workspace_bytes(), 0);
        }
    }

    /// The kernel name is the compiled-module cache key. Two 1x1 problems that
    /// share precision and layout but differ in channels-per-group emit
    /// *different* PTX (the group counts are code-gen immediates), so they must
    /// not share a name — otherwise one problem's compiled module would be
    /// handed to the other and silently produce wrong results.
    #[test]
    fn conv1x1_name_discriminates_channels_per_group() {
        let base = make_1x1_problem();
        let mut wider = base.clone();
        wider.in_channels = 128;
        let mut grouped = base.clone();
        grouped.groups = 2;

        let n_base = Conv1x1::new(base, SmVersion::Sm80).map(|c| c.kernel_name());
        let n_wider = Conv1x1::new(wider, SmVersion::Sm80).map(|c| c.kernel_name());
        let n_grouped = Conv1x1::new(grouped, SmVersion::Sm80).map(|c| c.kernel_name());

        assert!(n_base.is_ok() && n_wider.is_ok() && n_grouped.is_ok());
        assert_ne!(n_base.as_ref().ok(), n_wider.as_ref().ok());
        assert_ne!(n_base.as_ref().ok(), n_grouped.as_ref().ok());
    }

    /// Whatever the name encodes, it must still be the entry point actually
    /// emitted into the PTX — the cache looks the function up by this name.
    #[test]
    fn conv1x1_name_is_the_emitted_entry() {
        let conv = Conv1x1::new(make_1x1_problem(), SmVersion::Sm80);
        assert!(conv.is_ok());
        if let Ok(conv) = conv {
            let ptx = conv.generate_ptx().unwrap_or_default();
            assert!(
                ptx.contains(&format!(".visible .entry {}", conv.kernel_name())),
                "kernel_name() must name the emitted entry point"
            );
        }
    }

    #[test]
    fn depthwise_rejects_non_depthwise() {
        let mut p = make_depthwise_problem();
        p.groups = 1;
        assert!(DepthwiseConv::new(p, SmVersion::Sm80).is_err());
    }

    #[test]
    fn depthwise_kernel_name() {
        let d = DepthwiseConv::new(make_depthwise_problem(), SmVersion::Sm80);
        assert!(d.is_ok());
        if let Ok(conv) = d {
            assert_eq!(conv.kernel_name(), "depthwise_conv_3x3_f32");
        }
    }

    #[test]
    fn depthwise_workspace_zero() {
        let d = DepthwiseConv::new(make_depthwise_problem(), SmVersion::Sm80);
        assert!(d.is_ok());
        if let Ok(conv) = d {
            assert_eq!(conv.workspace_bytes(), 0);
        }
    }

    #[test]
    fn depthwise_ptx_generation() {
        let d = DepthwiseConv::new(make_depthwise_problem(), SmVersion::Sm80);
        assert!(d.is_ok());
        if let Ok(conv) = d {
            let ptx = conv.generate_ptx();
            assert!(ptx.is_ok());
            let text = ptx.unwrap_or_default();
            assert!(text.contains("depthwise_conv"));
        }
    }
}
