//! Per-channel / scalar broadcast kernel templates: **channel-broadcast
//! binary arithmetic** and **`PRelu`**.
//!
//! Both kernels share one addressing trick, so they live in one file: every
//! thread owns one element `i` of the *larger* ("full") operand, and reads a
//! second, much smaller ("small") operand at
//!
//! ```text
//! channel   = (i / spatial) % channels
//! small_idx = small_len == 1 ? 0 : channel
//! ```
//!
//! `channels`/`spatial` describe the *full* operand's shape as `[outer,
//! channels, spatial]` (outer folded into the flat index, since nothing below
//! needs it separately) — the row-major layout of an ONNX `[N, C, H, W]`
//! tensor with `spatial = H*W`. `small_len == 1` additionally covers a plain
//! scalar broadcast (a `[]`/`[1]` operand) with the exact same kernel body:
//! `channels` can be passed as `1` for that case (making `channel` always
//! `0`), or as the real channel count with `small_len` at `1` — either way
//! `small_idx` resolves to `0`.
//!
//! # Kernels
//!
//! - [`ChannelBroadcastTemplate`] — `out[i] = full[i] OP small[small_idx]`
//!   (or the operands swapped, for `Sub`/`Div`'s [`ChannelBroadcastTemplate::reverse`]).
//!   Covers ONNX `Add`/`Sub`/`Mul`/`Div` against a `[1,C,1,1]`-vs-`[1,C,H,W]`
//!   or scalar-vs-tensor operand pair.
//! - [`PReluTemplate`] — `out[i] = full[i] >= 0 ? full[i] : slope[small_idx] *
//!   full[i]`. The per-channel-slope generalisation of `LeakyRelu`.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::channel_broadcast::{ChannelBroadcastOp, ChannelBroadcastTemplate};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let t = ChannelBroadcastTemplate {
//!     op: ChannelBroadcastOp::Mul,
//!     reverse: false,
//!     precision: PtxType::F32,
//!     target: SmVersion::Sm86,
//! };
//! let ptx = t.generate().expect("PTX generation failed");
//! assert!(ptx.contains("channel_broadcast_mul_f32"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Which arithmetic op a [`ChannelBroadcastTemplate`] kernel computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelBroadcastOp {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
    /// `a / b`.
    Div,
}

impl ChannelBroadcastOp {
    /// Short lowercase name, used in the kernel's entry-point symbol.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
        }
    }

    /// The PTX instruction mnemonic (sans type suffix) for this op.
    ///
    /// `Div` uses the rounded form (`div.rn`), matching this crate's other
    /// elementwise division kernel (`elementwise::generate_div`) rather than
    /// the bare `div`, which PTX does not define for floating point.
    const fn instr(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div.rn",
        }
    }
}

/// Emits the shared "channel index" PTX body common to both kernels in this
/// module: the global thread id, the bounds check, and `small_idx = (small_len
/// == 1) ? 0 : (tid / spatial) % channels`.
///
/// Leaves behind, for the caller to consume:
/// - `%r3` — the global thread id (== the flat index into the full/output
///   operand).
/// - `%r10` — `small_idx`.
/// - `%rd0`/`%rd1`/`%rd2` — the three `u64` pointer params, loaded (named
///   generically as `full_ptr`/`small_ptr`/`out_ptr` regardless of which
///   kernel calls this).
/// - `%rd3` — the full/output operand's byte offset (`tid * byte_size`),
///   reusable by the caller for both the full-operand load and the
///   output store.
///
/// The bounds-check failure branches to a caller-supplied label so each
/// kernel can name its own exit point.
#[allow(clippy::too_many_arguments)]
fn emit_channel_index_prologue(
    ptx: &mut String,
    byte_size: usize,
    done_label: &str,
) -> Result<(), PtxGenError> {
    writeln!(ptx, "    // global thread id = flat index into full/output")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx).map_err(PtxGenError::FormatError)?;

    writeln!(ptx, "    ld.param.u32 %r4, [%param_total_len];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    setp.ge.u32 %p0, %r3, %r4;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    @%p0 bra {done_label};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx).map_err(PtxGenError::FormatError)?;

    writeln!(ptx, "    ld.param.u32 %r5, [%param_channels];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.param.u32 %r6, [%param_spatial];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.param.u32 %r7, [%param_small_len];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx).map_err(PtxGenError::FormatError)?;

    // q = tid / spatial ; channel = q - (q / channels) * channels  (== q % channels)
    writeln!(ptx, "    // channel = (tid / spatial) % channels")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    div.u32 %r8, %r3, %r6;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    div.u32 %r9, %r8, %r5;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u32 %r9, %r9, %r5;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    sub.u32 %r9, %r8, %r9;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx).map_err(PtxGenError::FormatError)?;

    // small_idx = (small_len == 1) ? 0 : channel
    writeln!(ptx, "    // small_idx = (small_len == 1) ? 0 : channel")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    setp.eq.u32 %p1, %r7, 1;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    selp.u32 %r10, 0, %r9, %p1;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx).map_err(PtxGenError::FormatError)?;

    writeln!(ptx, "    ld.param.u64 %rd0, [%param_full_ptr];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.param.u64 %rd1, [%param_small_ptr];")
        .map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    ld.param.u64 %rd2, [%param_out_ptr];").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    cvt.u64.u32 %rd3, %r3;").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    mul.lo.u64 %rd3, %rd3, {byte_size};").map_err(PtxGenError::FormatError)?;
    writeln!(ptx).map_err(PtxGenError::FormatError)?;
    Ok(())
}

/// Declares the parameter list common to both kernels: `full_ptr`,
/// `small_ptr`, `out_ptr` (all `.u64`), then `total_len`, `channels`,
/// `spatial`, `small_len` (all `.u32`).
fn write_params(ptx: &mut String) -> Result<(), PtxGenError> {
    writeln!(ptx, "    .param .u64 %param_full_ptr,").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    .param .u64 %param_small_ptr,").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    .param .u64 %param_out_ptr,").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    .param .u32 %param_total_len,").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    .param .u32 %param_channels,").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    .param .u32 %param_spatial,").map_err(PtxGenError::FormatError)?;
    writeln!(ptx, "    .param .u32 %param_small_len").map_err(PtxGenError::FormatError)?;
    Ok(())
}

// ─── ChannelBroadcastTemplate ───────────────────────────────────────────────

/// Template for a channel/scalar-broadcast binary arithmetic PTX kernel.
///
/// See the [module docs](self) for the addressing scheme. Kernel signature:
///
/// ```text
/// kernel(.u64 full_ptr, .u64 small_ptr, .u64 out_ptr,
///        .u32 total_len, .u32 channels, .u32 spatial, .u32 small_len)
/// ```
///
/// computing, for every `tid` in `[0, total_len)`:
///
/// ```text
/// small_idx = small_len == 1 ? 0 : (tid / spatial) % channels
/// out[tid]  = reverse ? small[small_idx] OP full[tid] : full[tid] OP small[small_idx]
/// ```
pub struct ChannelBroadcastTemplate {
    /// The arithmetic operation.
    pub op: ChannelBroadcastOp,
    /// When `true`, the *small* (broadcast) operand is the left-hand side of
    /// `op` and `full` the right-hand side — needed for `Sub`/`Div`, whose
    /// result depends on operand order, when the ONNX node's first input is
    /// the small operand. Irrelevant for `Add`/`Mul` (commutative); callers
    /// are expected to always pass `false` for those so the two share one
    /// compiled kernel.
    pub reverse: bool,
    /// The data precision.
    pub precision: PtxType,
    /// The target GPU architecture.
    pub target: SmVersion,
}

impl ChannelBroadcastTemplate {
    /// Returns the kernel function name, e.g. `channel_broadcast_mul_f32` or
    /// `channel_broadcast_sub_rev_f32`.
    ///
    /// Every value baked into the generated PTX text (`op`, `reverse`,
    /// `precision`) appears in this name, so it is safe to use unmodified as
    /// both the `CudaContext::module` cache key and the PTX entry-point
    /// symbol — the two-part discipline every other kernel in this crate
    /// follows (contrast [`crate::templates::batch_norm::BatchNormTemplate`],
    /// whose `epsilon` is baked in but *not* named, which is why
    /// `oxionnx-cuda`'s dispatcher must not use it unmodified as a cache
    /// key).
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        let dir = if self.reverse { "_rev" } else { "" };
        format!("channel_broadcast_{}{dir}_{type_str}", self.op.as_str())
    }

    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(self.precision, PtxType::F32 | PtxType::F64) {
            return Err(PtxGenError::InvalidType(format!(
                "channel_broadcast requires F32 or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }
        Ok(())
    }

    /// Generates the complete PTX module text.
    ///
    /// # Errors
    /// Returns [`PtxGenError`] if the precision is not `F32`/`F64`, or if
    /// text formatting fails.
    pub fn generate(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();
        let instr = self.op.instr();

        let mut ptx = String::with_capacity(3072);
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        write_params(&mut ptx)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    .reg .b32 %r<11>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {ty} %f<3>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<2>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        emit_channel_index_prologue(&mut ptx, byte_size, "$CB_DONE")?;

        writeln!(ptx, "    // load full[tid], small[small_idx]")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd0, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f0, [%rd4];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd1, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd6];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(
            ptx,
            "    // out = {}",
            if self.reverse {
                "small OP full"
            } else {
                "full OP small"
            }
        )
        .map_err(PtxGenError::FormatError)?;
        if self.reverse {
            writeln!(ptx, "    {instr}{ty} %f2, %f1, %f0;").map_err(PtxGenError::FormatError)?;
        } else {
            writeln!(ptx, "    {instr}{ty} %f2, %f0, %f1;").map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx, "    add.u64 %rd7, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd7], %f2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$CB_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }
}

// ─── PReluTemplate ──────────────────────────────────────────────────────────

/// Template for a per-channel `PRelu` PTX kernel:
/// `out[i] = x[i] >= 0 ? x[i] : slope[small_idx] * x[i]`.
///
/// A one-line variant of [`ChannelBroadcastTemplate`]'s `Mul` body with the
/// binary op replaced by a compare-and-select, and of
/// `elementwise::generate_leaky_relu`'s compare-and-select with the scalar
/// `alpha` uniform replaced by the same per-channel/scalar `slope` indexing
/// [`ChannelBroadcastTemplate`] uses. Reuses the identical parameter layout
/// (`full_ptr` names the activation `x`, `small_ptr` names `slope`) so
/// `oxionnx-cuda`'s dispatcher can share its channel/spatial-shape
/// bookkeeping between the two kernels.
pub struct PReluTemplate {
    /// The data precision.
    pub precision: PtxType,
    /// The target GPU architecture.
    pub target: SmVersion,
}

impl PReluTemplate {
    /// Returns the kernel function name, e.g. `prelu_channel_f32`.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        format!("prelu_channel_{type_str}")
    }

    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(self.precision, PtxType::F32 | PtxType::F64) {
            return Err(PtxGenError::InvalidType(format!(
                "prelu_channel requires F32 or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }
        Ok(())
    }

    /// Generates the complete PTX module text.
    ///
    /// Kernel signature is identical to [`ChannelBroadcastTemplate`]'s
    /// (`full_ptr` is `x`, `small_ptr` is `slope`, `out_ptr` is the result):
    ///
    /// ```text
    /// kernel(.u64 full_ptr, .u64 small_ptr, .u64 out_ptr,
    ///        .u32 total_len, .u32 channels, .u32 spatial, .u32 small_len)
    /// ```
    ///
    /// # Errors
    /// Returns [`PtxGenError`] if the precision is not `F32`/`F64`, or if
    /// text formatting fails.
    pub fn generate(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let kernel_name = self.kernel_name();
        let zero_lit = match self.precision {
            PtxType::F64 => "0d0000000000000000",
            _ => "0f00000000",
        };

        let mut ptx = String::with_capacity(3072);
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        write_params(&mut ptx)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    .reg .b32 %r<11>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg {ty} %f<4>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<3>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        emit_channel_index_prologue(&mut ptx, byte_size, "$PR_DONE")?;

        writeln!(ptx, "    // load x[tid], slope[small_idx]").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd4, %rd0, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f0, [%rd4];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd5, %r10;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd5, %rd5, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd6, %rd1, %rd5;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f1, [%rd6];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "    // y = x >= 0 ? x : slope * x").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul{ty} %f2, %f1, %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.ge{ty} %p2, %f0, {zero_lit};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    selp{ty} %f3, %f0, %f2, %p2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd7, %rd2, %rd3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd7], %f3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$PR_DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;

    fn cb(op: ChannelBroadcastOp, reverse: bool) -> ChannelBroadcastTemplate {
        ChannelBroadcastTemplate {
            op,
            reverse,
            precision: PtxType::F32,
            target: SmVersion::Sm86,
        }
    }

    #[test]
    fn kernel_names_encode_op_and_direction() {
        assert_eq!(
            cb(ChannelBroadcastOp::Add, false).kernel_name(),
            "channel_broadcast_add_f32"
        );
        assert_eq!(
            cb(ChannelBroadcastOp::Sub, false).kernel_name(),
            "channel_broadcast_sub_f32"
        );
        assert_eq!(
            cb(ChannelBroadcastOp::Sub, true).kernel_name(),
            "channel_broadcast_sub_rev_f32"
        );
        assert_eq!(
            cb(ChannelBroadcastOp::Mul, false).kernel_name(),
            "channel_broadcast_mul_f32"
        );
        assert_eq!(
            cb(ChannelBroadcastOp::Div, true).kernel_name(),
            "channel_broadcast_div_rev_f32"
        );
    }

    #[test]
    fn forward_and_reverse_names_differ() {
        assert_ne!(
            cb(ChannelBroadcastOp::Div, false).kernel_name(),
            cb(ChannelBroadcastOp::Div, true).kernel_name(),
        );
    }

    #[test]
    fn rejects_integer_precision() {
        let t = ChannelBroadcastTemplate {
            op: ChannelBroadcastOp::Add,
            reverse: false,
            precision: PtxType::U32,
            target: SmVersion::Sm86,
        };
        assert!(t.generate().is_err());
    }

    #[test]
    fn generates_valid_headers_and_params() {
        let t = cb(ChannelBroadcastOp::Add, false);
        let ptx = t.generate().expect("generate");
        assert!(ptx.contains(".target sm_86"));
        assert!(ptx.contains(".entry channel_broadcast_add_f32("));
        assert!(ptx.contains("%param_full_ptr"));
        assert!(ptx.contains("%param_small_ptr"));
        assert!(ptx.contains("%param_out_ptr"));
        assert!(ptx.contains("%param_total_len"));
        assert!(ptx.contains("%param_channels"));
        assert!(ptx.contains("%param_spatial"));
        assert!(ptx.contains("%param_small_len"));
    }

    #[test]
    fn forward_add_computes_full_op_small() {
        let ptx = cb(ChannelBroadcastOp::Add, false)
            .generate()
            .expect("generate");
        assert!(ptx.contains("add.f32 %f2, %f0, %f1;"));
    }

    #[test]
    fn reverse_sub_computes_small_op_full() {
        let ptx = cb(ChannelBroadcastOp::Sub, true)
            .generate()
            .expect("generate");
        assert!(ptx.contains("sub.f32 %f2, %f1, %f0;"));
    }

    #[test]
    fn forward_sub_computes_full_op_small() {
        let ptx = cb(ChannelBroadcastOp::Sub, false)
            .generate()
            .expect("generate");
        assert!(ptx.contains("sub.f32 %f2, %f0, %f1;"));
    }

    #[test]
    fn div_uses_rounded_division() {
        let ptx = cb(ChannelBroadcastOp::Div, false)
            .generate()
            .expect("generate");
        assert!(ptx.contains("div.rn.f32 %f2, %f0, %f1;"));
    }

    #[test]
    fn small_index_uses_div_rem_and_selp() {
        let ptx = cb(ChannelBroadcastOp::Mul, false)
            .generate()
            .expect("generate");
        assert!(ptx.contains("div.u32 %r8, %r3, %r6;"), "tid / spatial");
        assert!(ptx.contains("div.u32 %r9, %r8, %r5;"), "q / channels");
        assert!(
            ptx.contains("sub.u32 %r9, %r8, %r9;"),
            "channel = q - (q/channels)*channels"
        );
        assert!(
            ptx.contains("setp.eq.u32 %p1, %r7, 1;"),
            "small_len == 1 guard"
        );
        assert!(
            ptx.contains("selp.u32 %r10, 0, %r9, %p1;"),
            "small_idx select"
        );
    }

    #[test]
    fn bounds_guard_exits_before_any_load() {
        let ptx = cb(ChannelBroadcastOp::Add, false)
            .generate()
            .expect("generate");
        let bounds_pos = ptx.find("setp.ge.u32 %p0").expect("bounds check present");
        let first_load = ptx.find("ld.global.f32").expect("a load exists");
        assert!(
            bounds_pos < first_load,
            "bounds check must precede every load"
        );
    }

    #[test]
    fn generates_f64_kernel() {
        let t = ChannelBroadcastTemplate {
            op: ChannelBroadcastOp::Mul,
            reverse: false,
            precision: PtxType::F64,
            target: SmVersion::Sm90,
        };
        let ptx = t.generate().expect("generate f64");
        assert!(ptx.contains("channel_broadcast_mul_f64"));
        assert!(ptx.contains("mul.f64 %f2, %f0, %f1;"));
        assert!(ptx.contains("mul.lo.u64 %rd3, %rd3, 8;"));
    }

    // ── PReluTemplate ───────────────────────────────────────────────────────

    fn pr() -> PReluTemplate {
        PReluTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm86,
        }
    }

    #[test]
    fn prelu_kernel_name() {
        assert_eq!(pr().kernel_name(), "prelu_channel_f32");
    }

    #[test]
    fn prelu_rejects_integer_precision() {
        let t = PReluTemplate {
            precision: PtxType::S32,
            target: SmVersion::Sm86,
        };
        assert!(t.generate().is_err());
    }

    #[test]
    fn prelu_generates_valid_headers_and_params() {
        let ptx = pr().generate().expect("generate");
        assert!(ptx.contains(".entry prelu_channel_f32("));
        assert!(ptx.contains("%param_full_ptr"));
        assert!(ptx.contains("%param_small_ptr"));
        assert!(ptx.contains("%param_small_len"));
    }

    #[test]
    fn prelu_body_computes_compare_and_select() {
        let ptx = pr().generate().expect("generate");
        assert!(ptx.contains("mul.f32 %f2, %f1, %f0;"), "slope * x");
        assert!(ptx.contains("setp.ge.f32 %p2, %f0, 0f00000000;"), "x >= 0");
        assert!(
            ptx.contains("selp.f32 %f3, %f0, %f2, %p2;"),
            "select x vs slope*x"
        );
    }

    #[test]
    fn prelu_shares_the_channel_index_addressing_with_channel_broadcast() {
        let prelu_ptx = pr().generate().expect("generate");
        let mul_ptx = cb(ChannelBroadcastOp::Mul, false)
            .generate()
            .expect("generate");
        for needle in [
            "div.u32 %r8, %r3, %r6;",
            "div.u32 %r9, %r8, %r5;",
            "setp.eq.u32 %p1, %r7, 1;",
            "selp.u32 %r10, 0, %r9, %p1;",
        ] {
            assert!(prelu_ptx.contains(needle), "prelu missing {needle}");
            assert!(
                mul_ptx.contains(needle),
                "channel_broadcast missing {needle}"
            );
        }
    }

    #[test]
    fn prelu_generates_f64_kernel() {
        let t = PReluTemplate {
            precision: PtxType::F64,
            target: SmVersion::Sm90,
        };
        let ptx = t.generate().expect("generate f64");
        assert!(ptx.contains("prelu_channel_f64"));
        assert!(ptx.contains("setp.ge.f64"));
    }
}
