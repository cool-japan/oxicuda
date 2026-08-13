//! Im2col + GEMM convolution forward pass.
//!
//! This algorithm explicitly expands input patches into a column matrix
//! (im2col) and then calls the Vol.3 GEMM routine. It requires workspace
//! memory for the expanded matrix but benefits from highly-tuned GEMM
//! implementations.
//!
//! # Memory layout
//!
//! ```text
//! im2col matrix: M x K  where M = out_H * out_W, K = C * R * S
//! filter matrix: K x N  where N = out_channels
//! output matrix: M x N
//! ```
//!
//! The im2col expansion is performed by a separate GPU kernel before
//! invoking the GEMM.

use oxicuda_blas::GpuFloat;
use oxicuda_blas::level3::gemm_api;
use oxicuda_blas::types::{Layout, MatrixDesc, MatrixDescMut, Transpose};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};
use crate::handle::DnnHandle;
use crate::kernel_cache::cache_key;
use crate::types::{TensorDesc, TensorDescMut};

use super::super::descriptor::ConvProblem;

// ---------------------------------------------------------------------------
// Im2colGemmConv
// ---------------------------------------------------------------------------

/// Im2col + GEMM convolution engine.
///
/// Generates an im2col expansion kernel that rearranges input patches into
/// a matrix, then dispatches a BLAS GEMM for the actual multiplication.
pub struct Im2colGemmConv {
    problem: ConvProblem,
    sm_version: SmVersion,
}

impl Im2colGemmConv {
    /// Creates a new im2col + GEMM convolution engine.
    #[must_use]
    pub fn new(problem: ConvProblem, sm_version: SmVersion) -> Self {
        Self {
            problem,
            sm_version,
        }
    }

    /// Returns the kernel name for the im2col expansion.
    ///
    /// The element width is the *only* thing
    /// [`Self::generate_im2col_ptx`] bakes into the instruction stream (it
    /// selects the load/store width); every dimension, pad, stride and
    /// dilation is a runtime kernel parameter. The precision is encoded here,
    /// so this name is a complete compiled-module cache key for a given target
    /// architecture.
    #[must_use]
    pub fn im2col_kernel_name(&self) -> String {
        let prec = self.problem.input_type.as_ptx_str().trim_start_matches('.');
        format!("im2col_expand_{prec}")
    }

    /// Computes the workspace size in bytes required for the im2col matrix.
    ///
    /// The workspace stores the expanded M x K matrix where:
    /// - M = batch * out_H * out_W
    /// - K = (in_channels / groups) * R * S
    ///
    /// # Errors
    ///
    /// Returns an error if output dimension computation fails.
    pub fn workspace_bytes(&self) -> DnnResult<usize> {
        let out_dims = self.problem.output_dims()?;
        let spatial_product: u64 = out_dims.iter().map(|&d| d as u64).product();
        let m = self.problem.batch as u64 * spatial_product;
        let channels_per_group = self.problem.in_channels as u64 / self.problem.groups as u64;
        let filter_volume: u64 = self.problem.filter_dims.iter().map(|&d| d as u64).product();
        let k = channels_per_group * filter_volume;
        let elements = m * k;
        let bytes = elements * self.problem.input_type.size_bytes() as u64;
        Ok(bytes as usize)
    }

    /// Generates PTX for the im2col expansion kernel.
    ///
    /// Each thread processes one element of the `(C*R*S) x M` column matrix:
    /// it decodes the matrix coordinate, computes the corresponding input
    /// pixel position, and copies the input value (or zero-fills for padded
    /// positions) into the workspace.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::PtxGeneration`] on code generation failure.
    pub fn generate_im2col_ptx(&self) -> DnnResult<String> {
        // Element bit-width is fixed at PTX-generation time by the input type.
        let elem_bytes = self.problem.input_type.size_bytes() as u32;
        let ptx = KernelBuilder::new(&self.im2col_kernel_name())
            .target(self.sm_version)
            .param("input", PtxType::U64)
            .param("col_matrix", PtxType::U64)
            .param("batch_size", PtxType::U32)
            .param("in_channels", PtxType::U32)
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
            .param("total_elements", PtxType::U32)
            .body(move |b| {
                emit_im2col_body(b, elem_bytes);
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(e.to_string()))?;

        Ok(ptx)
    }

    /// Executes the im2col + GEMM convolution.
    ///
    /// 1. Launches the im2col kernel to expand input into the workspace.
    /// 2. Calls Vol.3 GEMM: output = filter^T * col_matrix.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::WorkspaceRequired`] if workspace is `None` or
    /// too small. Other errors from PTX generation, module loading, or
    /// kernel launch.
    pub fn execute<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        filter: &TensorDesc<T>,
        output: &mut TensorDescMut<T>,
        workspace: &mut oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        let required = self.workspace_bytes()?;
        if workspace.len() < required {
            return Err(DnnError::WorkspaceRequired(required));
        }

        // Phase 1: im2col expansion
        self.launch_im2col(handle, input, workspace)?;

        // The GEMM (phase 2) runs on the BLAS sub-handle's own stream, which
        // is a separate non-blocking stream from `handle.stream()` (where the
        // im2col kernel above was launched) with no implicit ordering between
        // the two. Record an event after the im2col launch and make the BLAS
        // stream wait on it, so the GEMM cannot start reading the workspace
        // before im2col has finished writing it.
        let im2col_done = oxicuda_driver::Event::new()?;
        im2col_done.record(handle.stream())?;
        handle.blas().stream().wait_event(&im2col_done)?;

        // Phase 2: GEMM
        // output = filter * col_matrix
        // filter: [K, C*R*S] (out_channels x filter_volume)
        // col:    [C*R*S, M] (filter_volume x spatial_points)
        // output: [K, M]     (out_channels x spatial_points)
        self.launch_gemm(handle, filter, output, workspace)?;

        // Join the GEMM's completion back into `handle.stream()` so that any
        // caller who only tracks `handle.stream()` (the documented primary
        // stream) observes `output` as ready once that stream is synchronised
        // or otherwise waited on, without needing to know the GEMM ran on the
        // BLAS sub-handle's separate stream internally.
        let gemm_done = oxicuda_driver::Event::new()?;
        gemm_done.record(handle.blas().stream())?;
        handle.stream().wait_event(&gemm_done)?;

        Ok(())
    }

    // -- Private helpers -----------------------------------------------------

    /// Launches the im2col expansion kernel.
    fn launch_im2col<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        input: &TensorDesc<T>,
        workspace: &mut oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        let entry = self.im2col_kernel_name();
        let kernel =
            handle.get_or_compile_kernel(&cache_key(&entry, self.sm_version), &entry, || {
                self.generate_im2col_ptx()
            })?;

        let out_dims = self.problem.output_dims()?;
        let out_h = out_dims.first().copied().unwrap_or(1);
        let out_w = out_dims.get(1).copied().unwrap_or(1);
        let channels_per_group = self.problem.in_channels / self.problem.groups;
        let filter_volume: u64 = self
            .problem
            .filter_dims
            .iter()
            .map(|&d| u64::from(d))
            .product();
        let total_elements_u64 = u64::from(self.problem.batch)
            * u64::from(out_h)
            * u64::from(out_w)
            * u64::from(channels_per_group)
            * filter_volume;
        let total_elements = u32::try_from(total_elements_u64).map_err(|_| {
            DnnError::InvalidDimension(format!(
                "im2col element count {total_elements_u64} exceeds u32::MAX (32-bit kernel indexing limit)"
            ))
        })?;

        let params = kernel.launch_1d(total_elements);
        let args = (
            input.ptr,
            workspace.as_device_ptr(),
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
            total_elements,
        );

        kernel
            .kernel()
            .launch(&params, handle.stream(), &args)
            .map_err(|e| DnnError::LaunchFailed(e.to_string()))?;

        Ok(())
    }

    /// Launches the GEMM phase.
    ///
    /// Uses the Vol.3 BLAS handle to compute:
    ///   `output[K x M] = filter[K x (C*R*S)] * col[(C*R*S) x M]`
    ///
    /// where `K = out_channels`, `C*R*S` is the filter volume, and
    /// `M = batch * out_spatial` is the number of output spatial points.
    ///
    /// # Dimension mapping
    ///
    /// [`ConvProblem::conv_to_gemm_dims`] returns `(gemm_m, gemm_n, gemm_k)`
    /// with `gemm_m = M`, `gemm_n = out_channels`, `gemm_k = C*R*S`. The BLAS
    /// `gemm` computes `C[m x n] = op(A)[m x k] * op(B)[k x n]`, so the natural
    /// orientation here uses BLAS dimensions `(blas_m, blas_n, blas_k) =
    /// (out_channels, M, C*R*S)`:
    ///
    /// - `A` = `filter`, an `out_channels x (C*R*S)` row-major matrix
    ///   (filter is stored `[K, C, R, S]` contiguously → leading dim `C*R*S`).
    /// - `B` = `col`, the `(C*R*S) x M` row-major workspace produced by the
    ///   im2col kernel → leading dim `M`.
    /// - `C` = `output`, an `out_channels x M` row-major matrix → leading
    ///   dim `M`.
    ///
    /// No transpose is required on either operand. `alpha = 1`, `beta = 0`.
    ///
    /// # Errors
    ///
    /// Returns [`DnnError::Blas`] if the BLAS GEMM dispatch fails, or an
    /// error from [`ConvProblem::conv_to_gemm_dims`].
    fn launch_gemm<T: GpuFloat>(
        &self,
        handle: &DnnHandle,
        filter: &TensorDesc<T>,
        output: &mut TensorDescMut<T>,
        workspace: &oxicuda_memory::DeviceBuffer<u8>,
    ) -> DnnResult<()> {
        let (gemm_m, gemm_n, gemm_k) = self.problem.conv_to_gemm_dims()?;

        // BLAS orientation: blas_m = out_channels, blas_n = spatial points,
        // blas_k = filter volume. (gemm_n is out_channels, gemm_m is M.)
        let blas_m = gemm_n;
        let blas_n = gemm_m;
        let blas_k = gemm_k;

        // A = filter: [out_channels x (C*R*S)] row-major, ld = C*R*S.
        let a = MatrixDesc::<T>::from_raw(filter.ptr, blas_m, blas_k, blas_k, Layout::RowMajor);

        // B = col workspace: [(C*R*S) x M] row-major, ld = M. The workspace
        // is an opaque byte buffer; reinterpret its base pointer as `T`.
        let b = MatrixDesc::<T>::from_raw(
            workspace.as_device_ptr(),
            blas_k,
            blas_n,
            blas_n,
            Layout::RowMajor,
        );

        // C = output: [out_channels x M] row-major, ld = M.
        let mut c =
            MatrixDescMut::<T>::from_raw(output.ptr, blas_m, blas_n, blas_n, Layout::RowMajor);

        gemm_api::gemm(
            handle.blas(),
            Transpose::NoTrans,
            Transpose::NoTrans,
            T::gpu_one(),
            &a,
            &b,
            T::gpu_zero(),
            &mut c,
        )?;

        Ok(())
    }
}

/// Standalone im2col body emitter for the `'static` closure requirement.
///
/// Each thread expands one element of the `(C*R*S) x M` column matrix that
/// the GEMM phase consumes as operand `B` (row-major, leading dimension
/// `M = batch * out_h * out_w`). The thread index `gid` is interpreted as
/// `k_idx * M + m_idx`; the thread decodes the convolution coordinates,
/// computes the source input pixel (zero for padded positions), and stores
/// it into `col_matrix[gid]`.
///
/// `elem_bytes` is the element width fixed at PTX-generation time (4 for
/// `f32`, 8 for `f64`, 2 for `f16`).
fn emit_im2col_body(b: &mut oxicuda_ptx::builder::BodyBuilder<'_>, elem_bytes: u32) {
    b.comment("=== Im2col Expansion Kernel ===");
    b.comment("Each thread expands one element of the (C*R*S) x M column matrix.");

    let load_ty = match elem_bytes {
        2 => "b16",
        8 => "f64",
        _ => "f32",
    };

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_elements");

    // Bounds check on the linear element index.
    let p_in = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {p_in}, {gid}, {total};"));
    let exit = b.fresh_label("im2col_exit");
    b.raw_ptx(&format!("@!{p_in} bra {exit};"));

    // Parameters.
    let input_ptr = b.load_param_u64("input");
    let col_ptr = b.load_param_u64("col_matrix");
    let batch = b.load_param_u32("batch_size");
    let in_channels = b.load_param_u32("in_channels");
    let in_h = b.load_param_u32("in_h");
    let in_w = b.load_param_u32("in_w");
    let filter_h = b.load_param_u32("filter_h");
    let filter_w = b.load_param_u32("filter_w");
    let out_h = b.load_param_u32("out_h");
    let out_w = b.load_param_u32("out_w");
    let pad_h = b.load_param_u32("pad_h");
    let pad_w = b.load_param_u32("pad_w");
    let stride_h = b.load_param_u32("stride_h");
    let stride_w = b.load_param_u32("stride_w");
    let dilation_h = b.load_param_u32("dilation_h");
    let dilation_w = b.load_param_u32("dilation_w");

    // M = batch * out_h * out_w  (the column-matrix leading dimension).
    let out_hw = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {out_hw}, {out_h}, {out_w};"));
    let m_total = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {m_total}, {batch}, {out_hw};"));

    // Decode gid = k_idx * M + m_idx.
    let m_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {m_idx}, {gid}, {m_total};"));
    let k_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {k_idx}, {gid}, {m_total};"));

    // m_idx -> (batch_n, oh, ow).
    let batch_n = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {batch_n}, {m_idx}, {out_hw};"));
    let m_rem = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {m_rem}, {m_idx}, {out_hw};"));
    let oh = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {oh}, {m_rem}, {out_w};"));
    let ow = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {ow}, {m_rem}, {out_w};"));

    // k_idx -> (c, kr, ks).
    let rs = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {rs}, {filter_h}, {filter_w};"));
    let c = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {c}, {k_idx}, {rs};"));
    let k_rem = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {k_rem}, {k_idx}, {rs};"));
    let kr = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {kr}, {k_rem}, {filter_w};"));
    let ks = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {ks}, {k_rem}, {filter_w};"));

    // ih = oh*stride_h - pad_h + kr*dilation_h  (signed: padding may go < 0).
    let ih = b.alloc_reg(PtxType::S32);
    let tmp = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {tmp}, {oh}, {stride_h};"));
    b.raw_ptx(&format!("mov.b32 {ih}, {tmp};"));
    let pad_h_s = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {pad_h_s}, {pad_h};"));
    b.raw_ptx(&format!("sub.s32 {ih}, {ih}, {pad_h_s};"));
    b.raw_ptx(&format!("mul.lo.u32 {tmp}, {kr}, {dilation_h};"));
    let dh_s = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {dh_s}, {tmp};"));
    b.raw_ptx(&format!("add.s32 {ih}, {ih}, {dh_s};"));

    // iw = ow*stride_w - pad_w + ks*dilation_w.
    let iw = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mul.lo.u32 {tmp}, {ow}, {stride_w};"));
    b.raw_ptx(&format!("mov.b32 {iw}, {tmp};"));
    let pad_w_s = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {pad_w_s}, {pad_w};"));
    b.raw_ptx(&format!("sub.s32 {iw}, {iw}, {pad_w_s};"));
    b.raw_ptx(&format!("mul.lo.u32 {tmp}, {ks}, {dilation_w};"));
    let dw_s = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {dw_s}, {tmp};"));
    b.raw_ptx(&format!("add.s32 {iw}, {iw}, {dw_s};"));

    // Boundary check: 0 <= ih < in_h && 0 <= iw < in_w.
    let in_h_s = b.alloc_reg(PtxType::S32);
    let in_w_s = b.alloc_reg(PtxType::S32);
    b.raw_ptx(&format!("mov.b32 {in_h_s}, {in_h};"));
    b.raw_ptx(&format!("mov.b32 {in_w_s}, {in_w};"));
    let p_h0 = b.alloc_reg(PtxType::Pred);
    let p_h1 = b.alloc_reg(PtxType::Pred);
    let p_w0 = b.alloc_reg(PtxType::Pred);
    let p_w1 = b.alloc_reg(PtxType::Pred);
    let p_valid = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ge.s32 {p_h0}, {ih}, 0;"));
    b.raw_ptx(&format!("setp.lt.s32 {p_h1}, {ih}, {in_h_s};"));
    b.raw_ptx(&format!("setp.ge.s32 {p_w0}, {iw}, 0;"));
    b.raw_ptx(&format!("setp.lt.s32 {p_w1}, {iw}, {in_w_s};"));
    b.raw_ptx(&format!("and.pred {p_valid}, {p_h0}, {p_h1};"));
    b.raw_ptx(&format!("and.pred {p_valid}, {p_valid}, {p_w0};"));
    b.raw_ptx(&format!("and.pred {p_valid}, {p_valid}, {p_w1};"));

    // Compute the destination column address: col_matrix + gid * elem_bytes.
    let col_addr = b.byte_offset_addr(col_ptr, gid.clone(), elem_bytes);

    // Out-of-bounds path: store a zero (zero padding).
    let zero_store = b.fresh_label("im2col_zero");
    let store_done = b.fresh_label("im2col_store_done");
    b.raw_ptx(&format!("@!{p_valid} bra {zero_store};"));

    // In-bounds: input NCHW index = ((batch_n*C + c)*in_h + ih)*in_w + iw.
    let ih_u = b.alloc_reg(PtxType::U32);
    let iw_u = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.b32 {ih_u}, {ih};"));
    b.raw_ptx(&format!("mov.b32 {iw_u}, {iw};"));
    let in_idx = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {in_idx}, {batch_n}, {in_channels};"));
    b.raw_ptx(&format!("add.u32 {in_idx}, {in_idx}, {c};"));
    b.raw_ptx(&format!("mul.lo.u32 {in_idx}, {in_idx}, {in_h};"));
    b.raw_ptx(&format!("add.u32 {in_idx}, {in_idx}, {ih_u};"));
    b.raw_ptx(&format!("mul.lo.u32 {in_idx}, {in_idx}, {in_w};"));
    b.raw_ptx(&format!("add.u32 {in_idx}, {in_idx}, {iw_u};"));
    let in_addr = b.byte_offset_addr(input_ptr, in_idx, elem_bytes);

    let val = b.alloc_reg(if elem_bytes == 8 {
        PtxType::F64
    } else if elem_bytes == 2 {
        PtxType::B16
    } else {
        PtxType::F32
    });
    b.raw_ptx(&format!("ld.global.{load_ty} {val}, [{in_addr}];"));
    b.raw_ptx(&format!("st.global.{load_ty} [{col_addr}], {val};"));
    b.raw_ptx(&format!("bra {store_done};"));

    // Zero-padding path.
    b.raw_ptx(&format!("{zero_store}:"));
    let zero = b.alloc_reg(if elem_bytes == 8 {
        PtxType::F64
    } else if elem_bytes == 2 {
        PtxType::B16
    } else {
        PtxType::F32
    });
    match elem_bytes {
        2 => b.raw_ptx(&format!("mov.b16 {zero}, 0x0000;")),
        8 => b.raw_ptx(&format!("mov.f64 {zero}, 0d0000000000000000;")),
        _ => b.raw_ptx(&format!("mov.f32 {zero}, 0f00000000;")),
    }
    b.raw_ptx(&format!("st.global.{load_ty} [{col_addr}], {zero};"));

    b.raw_ptx(&format!("{store_done}:"));
    b.raw_ptx(&format!("{exit}:"));
    b.ret();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TensorLayout;

    fn make_problem() -> ConvProblem {
        ConvProblem {
            batch: 1,
            in_channels: 3,
            in_dims: vec![8, 8],
            out_channels: 16,
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
    fn workspace_bytes_calculation() {
        let conv = Im2colGemmConv::new(make_problem(), SmVersion::Sm80);
        let ws = conv.workspace_bytes();
        assert!(ws.is_ok());
        // M = 1 * 8 * 8 = 64, K = 3 * 3 * 3 = 27
        // elements = 64 * 27 = 1728, bytes = 1728 * 4 = 6912
        assert_eq!(ws.unwrap_or(0), 6912);
    }

    #[test]
    fn im2col_kernel_name() {
        let conv = Im2colGemmConv::new(make_problem(), SmVersion::Sm80);
        assert_eq!(conv.im2col_kernel_name(), "im2col_expand_f32");
    }

    #[test]
    fn ptx_generation() {
        let conv = Im2colGemmConv::new(make_problem(), SmVersion::Sm80);
        let ptx = conv.generate_im2col_ptx();
        assert!(ptx.is_ok());
        let text = ptx.unwrap_or_default();
        assert!(text.contains("im2col_expand"));
    }

    // -----------------------------------------------------------------------
    // im2col + GEMM correctness: CPU reference for the launch_gemm orientation
    // -----------------------------------------------------------------------

    /// Convolution geometry for the CPU reference helpers (NCHW, batch 1).
    struct ConvGeom {
        in_c: usize,
        in_h: usize,
        in_w: usize,
        out_c: usize,
        r: usize,
        s: usize,
        pad: usize,
        out_h: usize,
        out_w: usize,
    }

    /// CPU im2col expansion producing a `(C*R*S) x M` column matrix in
    /// row-major order, matching the layout `launch_gemm` expects for the
    /// `B` operand. `input` is NCHW with `batch == 1`.
    fn cpu_im2col(input: &[f32], g: &ConvGeom) -> Vec<f32> {
        let k_dim = g.in_c * g.r * g.s;
        let m = g.out_h * g.out_w;
        let mut col = vec![0.0f32; k_dim * m];
        for c in 0..g.in_c {
            for kr in 0..g.r {
                for ks in 0..g.s {
                    let k_idx = (c * g.r + kr) * g.s + ks;
                    for oh in 0..g.out_h {
                        for ow in 0..g.out_w {
                            let ih = oh as isize + kr as isize - g.pad as isize;
                            let iw = ow as isize + ks as isize - g.pad as isize;
                            let m_idx = oh * g.out_w + ow;
                            if ih >= 0
                                && iw >= 0
                                && (ih as usize) < g.in_h
                                && (iw as usize) < g.in_w
                            {
                                let in_idx = (c * g.in_h + ih as usize) * g.in_w + iw as usize;
                                col[k_idx * m + m_idx] = input[in_idx];
                            }
                        }
                    }
                }
            }
        }
        col
    }

    /// Reference row-major GEMM `C[m x n] = A[m x k] * B[k x n]`.
    fn cpu_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    /// Direct convolution reference (NCHW, batch 1, zero padding).
    fn cpu_conv2d(input: &[f32], filter: &[f32], g: &ConvGeom) -> Vec<f32> {
        let mut out = vec![0.0f32; g.out_c * g.out_h * g.out_w];
        for oc in 0..g.out_c {
            for oh in 0..g.out_h {
                for ow in 0..g.out_w {
                    let mut acc = 0.0f32;
                    for c in 0..g.in_c {
                        for kr in 0..g.r {
                            for ks in 0..g.s {
                                let ih = oh as isize + kr as isize - g.pad as isize;
                                let iw = ow as isize + ks as isize - g.pad as isize;
                                if ih >= 0
                                    && iw >= 0
                                    && (ih as usize) < g.in_h
                                    && (iw as usize) < g.in_w
                                {
                                    let in_idx = (c * g.in_h + ih as usize) * g.in_w + iw as usize;
                                    let w_idx = ((oc * g.in_c + c) * g.r + kr) * g.s + ks;
                                    acc += input[in_idx] * filter[w_idx];
                                }
                            }
                        }
                    }
                    out[(oc * g.out_h + oh) * g.out_w + ow] = acc;
                }
            }
        }
        out
    }

    /// The `launch_gemm` orientation (`filter * col`) must reproduce a direct
    /// convolution. This validates the `K x M` vs `M x K` reasoning encoded in
    /// `launch_gemm`'s BLAS dimension mapping on the CPU.
    #[test]
    fn im2col_gemm_matches_direct_conv() {
        let pad = 1usize;
        let geom = ConvGeom {
            in_c: 3,
            in_h: 5,
            in_w: 5,
            out_c: 4,
            r: 3,
            s: 3,
            pad,
            out_h: 5 + 2 * pad - 3 + 1,
            out_w: 5 + 2 * pad - 3 + 1,
        };

        // Deterministic pseudo-random inputs.
        let input: Vec<f32> = (0..geom.in_c * geom.in_h * geom.in_w)
            .map(|i| ((i * 37 % 17) as f32) * 0.1 - 0.8)
            .collect();
        let filter: Vec<f32> = (0..geom.out_c * geom.in_c * geom.r * geom.s)
            .map(|i| ((i * 23 % 13) as f32) * 0.05 - 0.3)
            .collect();

        // GEMM path: A = filter [out_c x (C*R*S)], B = col [(C*R*S) x M].
        let k_dim = geom.in_c * geom.r * geom.s;
        let m = geom.out_h * geom.out_w;
        let col = cpu_im2col(&input, &geom);
        let gemm_out = cpu_gemm(&filter, &col, geom.out_c, m, k_dim);

        // Direct convolution reference.
        let direct = cpu_conv2d(&input, &filter, &geom);

        assert_eq!(gemm_out.len(), direct.len());
        for (g, d) in gemm_out.iter().zip(direct.iter()) {
            assert!(
                (g - d).abs() < 1e-4,
                "im2col-GEMM output {g} != direct conv {d}"
            );
        }
    }

    /// The BLAS dimension mapping inside `launch_gemm` derives
    /// `(blas_m, blas_n, blas_k)` from `conv_to_gemm_dims`. Verify the
    /// values for the canonical 3x3 problem.
    #[test]
    fn launch_gemm_dimension_mapping() {
        let problem = make_problem();
        let (gemm_m, gemm_n, gemm_k) = problem.conv_to_gemm_dims().expect("gemm dims must compute");
        // M = batch * out_h * out_w = 1 * 8 * 8 = 64
        assert_eq!(gemm_m, 64);
        // N = out_channels = 16
        assert_eq!(gemm_n, 16);
        // K = C*R*S = 3*3*3 = 27
        assert_eq!(gemm_k, 27);

        // launch_gemm uses blas_m = out_channels, blas_n = M, blas_k = K.
        let blas_m = gemm_n;
        let blas_n = gemm_m;
        let blas_k = gemm_k;
        assert_eq!((blas_m, blas_n, blas_k), (16, 64, 27));
    }

    // -----------------------------------------------------------------------
    // Regression for F019: 32-bit element-count overflow must be rejected,
    // not silently truncated into an under-launched kernel.
    // -----------------------------------------------------------------------

    /// `launch_im2col` must reject (rather than silently truncate) an
    /// element count that overflows `u32`, computing the product in `u64`
    /// first. The problem below is built from plain integer dimensions (no
    /// multi-gigabyte allocation is needed) whose product
    /// `batch * out_h * out_w * channels_per_group * filter_volume` exceeds
    /// `u32::MAX`; the check must fire before any device memory is touched
    /// by the launch, so tiny 1-element `input`/`workspace` buffers suffice.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn launch_im2col_rejects_element_count_overflow() {
        use std::sync::Arc;

        use oxicuda_driver::{Context, Device};
        use oxicuda_memory::DeviceBuffer;

        use crate::types::TensorLayout;

        if oxicuda_driver::init().is_err() {
            eprintln!("skipping launch_im2col_rejects_element_count_overflow: no CUDA driver");
            return;
        }
        let Ok(device) = Device::get(0) else {
            eprintln!("skipping launch_im2col_rejects_element_count_overflow: no CUDA device");
            return;
        };
        let Ok(context) = Context::new(&device) else {
            eprintln!("skipping launch_im2col_rejects_element_count_overflow: no context");
            return;
        };
        let ctx = Arc::new(context);
        let Ok(handle) = DnnHandle::new(&ctx) else {
            eprintln!("skipping launch_im2col_rejects_element_count_overflow: no DnnHandle");
            return;
        };

        // filter_volume alone (65536 * 65536 = 2^32) already exceeds
        // u32::MAX; in_dims match filter_dims exactly (pad=0, stride=1,
        // dilation=1) so output_dims() resolves to [1, 1].
        let problem = ConvProblem {
            batch: 1,
            in_channels: 1,
            in_dims: vec![65536, 65536],
            out_channels: 1,
            filter_dims: vec![65536, 65536],
            padding: vec![0, 0],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            input_type: PtxType::F32,
            output_type: PtxType::F32,
            layout: TensorLayout::Nchw,
        };
        let filter_volume: u64 = problem.filter_dims.iter().map(|&d| u64::from(d)).product();
        assert!(
            filter_volume > u64::from(u32::MAX),
            "test fixture must actually overflow u32"
        );

        let engine = Im2colGemmConv::new(problem, handle.sm_version());

        let d_in = DeviceBuffer::<f32>::from_host(&[0.0f32]).expect("alloc input");
        let input = TensorDesc::<f32>::from_raw(
            d_in.as_device_ptr(),
            vec![1, 1],
            vec![1, 1],
            TensorLayout::Nchw,
        )
        .expect("build input descriptor");
        let mut workspace = DeviceBuffer::<u8>::alloc(1).expect("alloc workspace");

        match engine.launch_im2col(&handle, &input, &mut workspace) {
            Err(DnnError::InvalidDimension(msg)) => {
                assert!(
                    msg.contains("u32::MAX"),
                    "expected overflow message, got: {msg}"
                );
            }
            other => {
                panic!("expected DnnError::InvalidDimension overflow rejection, got {other:?}")
            }
        }
    }
}
