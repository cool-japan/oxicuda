//! GPU kernel template for axis-broadcast operations.
//!
//! Generates a PTX kernel that replicates a reduced tensor back to a larger
//! shape using the stride-zero trick: for each reduced axis, the corresponding
//! entry in `src_strides_padded` is set to zero, so the flat-index unravel
//! naturally maps multiple output elements to the same source element.
//!
//! # Stride-zero trick
//!
//! Let `dst` have rank `R` with shape `ds[0..R]` and row-major strides
//! `dst_s[0..R]` where `dst_s[d] = product(ds[d+1..R])`.
//!
//! For a source reduced along some axes, we build `src_strides_padded[0..R]`
//! where `src_strides_padded[d] = 0` if axis `d` was reduced, and the
//! corresponding source row-major stride otherwise.
//!
//! Each GPU thread handling output element `tid` computes:
//! ```text
//! flat_src = Σ_{d=0}^{R-1} ((tid / dst_s[d]) % ds[d]) * src_strides_padded[d]
//! ```
//! Because reduced-axis strides are zero, those dimensions contribute nothing,
//! mapping many output positions to the same source element.
//!
//! The kernel is unrolled to `MAX_BROADCAST_RANK` (8) dimensions and uses
//! `setp.lt.u32` predicates to mask out dimensions above the actual rank.

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Maximum rank supported by the broadcast kernel.
///
/// All shape/stride arrays are padded to this width. Output elements in
/// dimensions beyond the actual rank are masked out via predicate guards.
pub const MAX_BROADCAST_RANK: usize = 8;

/// Template for generating a GPU broadcast-axes PTX kernel.
///
/// The generated kernel expands a `src` tensor to `dst` by replicating
/// values along axes listed in `reduced_axes`. Uses the stride-zero trick:
/// strides for reduced axes are set to 0 in `src_strides_padded`.
///
/// # Kernel signature
///
/// ```ptx
/// .visible .entry broadcast_axes_f32(
///     .param .u64 src_ptr,
///     .param .u64 dst_ptr,
///     .param .u32 rank,
///     .param .u32 ds0, .param .u32 ds1, ..., .param .u32 ds7,       // dst shape
///     .param .u32 dst_s0, ..., .param .u32 dst_s7,                   // dst row-major strides
///     .param .u32 ss0, ..., .param .u32 ss7,                         // src strides (0 for reduced)
///     .param .u32 n_dst                                               // total output elements
/// )
/// ```
pub struct BroadcastTemplate {
    /// The data precision for the broadcast operation.
    pub precision: PtxType,
    /// The target GPU architecture.
    pub target: SmVersion,
}

impl BroadcastTemplate {
    /// Creates a new broadcast template with the given precision and target.
    #[must_use]
    pub const fn new(precision: PtxType, target: SmVersion) -> Self {
        Self { precision, target }
    }

    /// Returns the kernel function name.
    ///
    /// Follows the pattern `broadcast_axes_{type}`, e.g. `broadcast_axes_f32`.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        format!("broadcast_axes_{type_str}")
    }

    /// Generates the complete PTX module text for the broadcast-axes kernel.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if the precision type is unsupported (non-float)
    /// or if string formatting fails.
    pub fn generate(&self) -> Result<String, PtxGenError> {
        self.validate_precision()?;
        Ok(self.generate_raw_ptx())
    }

    /// Validates that the precision is a supported floating-point type.
    fn validate_precision(&self) -> Result<(), PtxGenError> {
        if !matches!(
            self.precision,
            PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64
        ) {
            return Err(PtxGenError::InvalidType(format!(
                "broadcast_axes requires F16, BF16, F32, or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }
        Ok(())
    }

    /// Generates raw PTX text via `format!()`.
    ///
    /// The broadcast kernel has 28 parameters (`src_ptr`, `dst_ptr`, `rank`, 8×`ds`,
    /// 8×`dst_strides`, 8×`src_strides_padded`, `n_dst`) which would be unwieldy
    /// through the `KernelBuilder` API, so we generate the text directly.
    ///
    /// The computation per output thread is:
    /// ```text
    /// flat_src = Σ_d ( (tid / dst_s[d]) % ds[d] ) * ss[d]
    /// ```
    /// This is the standard "unravel + re-index with zero-stride" algorithm.
    /// Each of the 8 dimensions is unrolled with a predicate guard (`d < rank`).
    #[allow(clippy::too_many_lines)]
    fn generate_raw_ptx(&self) -> String {
        let kernel_name = self.kernel_name();
        let ptx_ver = self.target.ptx_version();
        let sm_target = self.target.as_ptx_str();
        let byte_size = self.precision.size_bytes();

        // Float register type for ld/st instructions.
        // For f16/bf16, we use b16 for the load/store (bit-width match) since
        // PTX does not allow direct ld.global.f16 in all contexts, but for
        // standard BLAS usage the underlying bits are what we copy.
        let (float_reg, ld_ty, st_ty) = float_reg_info(self.precision);

        // Build the unrolled dimension accumulation body.
        // For each dim d=0..7:
        //   setp.lt.u32 %p_dN, %rank, N+1       // active if d < rank
        // @%p_dN:
        //   div.u32 %q, %rem, %dst_sN            // q = rem / dst_s[d]
        //   rem.u32 %rem, %rem, %dst_sN           // not exactly, see below
        //   rem.u32 %r, %q, %dsN                  // r = q % ds[d]
        //   mul.lo.u32 %contrib, %r, %ssN         // contrib = r * ss[d]
        //   add.u32 %flat_src, %flat_src, %contrib
        //
        // However, the standard "unravel" uses the row-major strides:
        //   coord[d] = (tid / dst_strides[d]) % dst_shape[d]
        // This is different from peeling dimensions with remainder tracking.
        // The stride-based approach is simpler and maps directly to:
        //   coord_d = (tid / dst_s[d]) % ds[d]
        // We use this approach (not the remainder-chain approach) because we
        // already have precomputed `dst_strides` from the host side.
        let dim_body = Self::generate_dim_accumulation();

        format!(
            ".version {ptx_ver}\n\
             .target {sm_target}\n\
             .address_size 64\n\
             \n\
             .visible .entry {kernel_name}(\n\
                 .param .u64 %param_src_ptr,\n\
                 .param .u64 %param_dst_ptr,\n\
                 .param .u32 %param_rank,\n\
                 .param .u32 %param_ds0,\n\
                 .param .u32 %param_ds1,\n\
                 .param .u32 %param_ds2,\n\
                 .param .u32 %param_ds3,\n\
                 .param .u32 %param_ds4,\n\
                 .param .u32 %param_ds5,\n\
                 .param .u32 %param_ds6,\n\
                 .param .u32 %param_ds7,\n\
                 .param .u32 %param_dst_s0,\n\
                 .param .u32 %param_dst_s1,\n\
                 .param .u32 %param_dst_s2,\n\
                 .param .u32 %param_dst_s3,\n\
                 .param .u32 %param_dst_s4,\n\
                 .param .u32 %param_dst_s5,\n\
                 .param .u32 %param_dst_s6,\n\
                 .param .u32 %param_dst_s7,\n\
                 .param .u32 %param_ss0,\n\
                 .param .u32 %param_ss1,\n\
                 .param .u32 %param_ss2,\n\
                 .param .u32 %param_ss3,\n\
                 .param .u32 %param_ss4,\n\
                 .param .u32 %param_ss5,\n\
                 .param .u32 %param_ss6,\n\
                 .param .u32 %param_ss7,\n\
                 .param .u32 %param_n_dst\n\
             )\n\
             {{\n\
                 .reg .u32 %tid, %n_dst, %rank;\n\
                 .reg .u64 %src_ptr, %dst_ptr, %off;\n\
                 .reg .pred %p_oob;\n\
                 .reg .u32 %flat_src;\n\
                 .reg .u32 %q, %r;\n\
                 .reg .u32 %ds0, %ds1, %ds2, %ds3, %ds4, %ds5, %ds6, %ds7;\n\
                 .reg .u32 %dst_s0, %dst_s1, %dst_s2, %dst_s3, %dst_s4, %dst_s5, %dst_s6, %dst_s7;\n\
                 .reg .u32 %ss0, %ss1, %ss2, %ss3, %ss4, %ss5, %ss6, %ss7;\n\
                 .reg .pred %p_d0, %p_d1, %p_d2, %p_d3, %p_d4, %p_d5, %p_d6, %p_d7;\n\
                 .reg .u32 %contrib;\n\
                 .reg .{float_reg} %val;\n\
             \n\
                 // Compute global thread id\n\
                 mov.u32 %tid, %ctaid.x;\n\
                 mad.lo.u32 %tid, %ntid.x, %tid, %tid.x;\n\
             \n\
                 // Bounds check\n\
                 ld.param.u32 %n_dst, [%param_n_dst];\n\
                 setp.ge.u32 %p_oob, %tid, %n_dst;\n\
                 @%p_oob bra done;\n\
             \n\
                 // Load rank and all shape/stride arrays\n\
                 ld.param.u32 %rank, [%param_rank];\n\
                 ld.param.u32 %ds0, [%param_ds0];\n\
                 ld.param.u32 %ds1, [%param_ds1];\n\
                 ld.param.u32 %ds2, [%param_ds2];\n\
                 ld.param.u32 %ds3, [%param_ds3];\n\
                 ld.param.u32 %ds4, [%param_ds4];\n\
                 ld.param.u32 %ds5, [%param_ds5];\n\
                 ld.param.u32 %ds6, [%param_ds6];\n\
                 ld.param.u32 %ds7, [%param_ds7];\n\
                 ld.param.u32 %dst_s0, [%param_dst_s0];\n\
                 ld.param.u32 %dst_s1, [%param_dst_s1];\n\
                 ld.param.u32 %dst_s2, [%param_dst_s2];\n\
                 ld.param.u32 %dst_s3, [%param_dst_s3];\n\
                 ld.param.u32 %dst_s4, [%param_dst_s4];\n\
                 ld.param.u32 %dst_s5, [%param_dst_s5];\n\
                 ld.param.u32 %dst_s6, [%param_dst_s6];\n\
                 ld.param.u32 %dst_s7, [%param_dst_s7];\n\
                 ld.param.u32 %ss0, [%param_ss0];\n\
                 ld.param.u32 %ss1, [%param_ss1];\n\
                 ld.param.u32 %ss2, [%param_ss2];\n\
                 ld.param.u32 %ss3, [%param_ss3];\n\
                 ld.param.u32 %ss4, [%param_ss4];\n\
                 ld.param.u32 %ss5, [%param_ss5];\n\
                 ld.param.u32 %ss6, [%param_ss6];\n\
                 ld.param.u32 %ss7, [%param_ss7];\n\
             \n\
                 // Compute flat_src via stride-zero trick (unrolled 8 dims)\n\
                 mov.u32 %flat_src, 0;\n\
             {dim_body}\n\
                 // Load src[flat_src]\n\
                 ld.param.u64 %src_ptr, [%param_src_ptr];\n\
                 cvt.u64.u32 %off, %flat_src;\n\
                 mul.lo.u64 %off, %off, {byte_size};\n\
                 add.u64 %src_ptr, %src_ptr, %off;\n\
                 ld.global{ld_ty} %val, [%src_ptr];\n\
             \n\
                 // Store dst[tid]\n\
                 ld.param.u64 %dst_ptr, [%param_dst_ptr];\n\
                 cvt.u64.u32 %off, %tid;\n\
                 mul.lo.u64 %off, %off, {byte_size};\n\
                 add.u64 %dst_ptr, %dst_ptr, %off;\n\
                 st.global{st_ty} [%dst_ptr], %val;\n\
             \n\
             done:\n\
                 ret;\n\
             }}\n"
        )
    }

    /// Generates the unrolled per-dimension `flat_src` accumulation body.
    ///
    /// For each dimension `d` in `0..MAX_BROADCAST_RANK`:
    ///   1. Guard: `setp.lt.u32 %p_dN, %rank, N+1` — false when `d >= rank`
    ///   2. `@%p_dN` compute `coord_d = (tid / dst_s[d]) % ds[d]`
    ///   3. `@%p_dN` accumulate `flat_src += coord_d * ss[d]`
    ///
    /// Using `setp.lt` with negation: `p_dN = !(rank <= d)` — we set
    /// `%p_dN` to true when `d < rank`, i.e. `rank > d`, i.e. NOT (rank <= d).
    fn generate_dim_accumulation() -> String {
        use std::fmt::Write as _;
        let mut body = String::with_capacity(2048);
        let dim_names = ["0", "1", "2", "3", "4", "5", "6", "7"];

        for (d, &dn) in dim_names.iter().enumerate() {
            // p_dN is TRUE when d < rank, i.e. when rank > d.
            // setp.gt.u32 %p_dN, %rank, d  (true when rank > d)
            let _ = writeln!(body, "    setp.gt.u32 %p_d{dn}, %rank, {d};");
            // coord_d = (tid / dst_s[d]) % ds[d]
            // @pred: div.u32 %q, %tid, %dst_sN
            let _ = writeln!(body, "    @%p_d{dn} div.u32 %q, %tid, %dst_s{dn};");
            // @pred: rem.u32 %r, %q, %dsN  (r = q % ds[d])
            let _ = writeln!(body, "    @%p_d{dn} rem.u32 %r, %q, %ds{dn};");
            // @pred: mul.lo.u32 %contrib, %r, %ssN
            let _ = writeln!(body, "    @%p_d{dn} mul.lo.u32 %contrib, %r, %ss{dn};");
            // @pred: add.u32 %flat_src, %flat_src, %contrib
            let _ = writeln!(
                body,
                "    @%p_d{dn} add.u32 %flat_src, %flat_src, %contrib;"
            );
        }
        body
    }
}

/// Returns the float register class, load type suffix, and store type suffix
/// for the given precision.
///
/// For f16/bf16, PTX uses `.b16` typed loads/stores when working at the bit
/// level (copy without conversion). Since broadcast is a pure copy, this
/// preserves bit patterns correctly across precision widths.
const fn float_reg_info(precision: PtxType) -> (&'static str, &'static str, &'static str) {
    match precision {
        PtxType::F16 | PtxType::BF16 => ("b16", ".b16", ".b16"),
        PtxType::F32 => ("f32", ".f32", ".f32"),
        PtxType::F64 => ("f64", ".f64", ".f64"),
        _ => ("b32", ".b32", ".b32"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;

    #[test]
    fn broadcast_kernel_name_f32() {
        let t = BroadcastTemplate::new(PtxType::F32, SmVersion::Sm80);
        assert_eq!(t.kernel_name(), "broadcast_axes_f32");
    }

    #[test]
    fn broadcast_kernel_name_f16() {
        let t = BroadcastTemplate::new(PtxType::F16, SmVersion::Sm80);
        assert_eq!(t.kernel_name(), "broadcast_axes_f16");
    }

    #[test]
    fn broadcast_invalid_precision_rejected() {
        let t = BroadcastTemplate::new(PtxType::U32, SmVersion::Sm80);
        assert!(t.generate().is_err());
    }

    #[test]
    fn broadcast_generates_valid_ptx_headers_f32() {
        let t = BroadcastTemplate::new(PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("broadcast_axes_f32 generation failed");
        assert!(
            ptx.contains(".version 7.0"),
            "must have correct PTX version"
        );
        assert!(ptx.contains(".target sm_80"), "must have correct target");
        assert!(
            ptx.contains(".address_size 64"),
            "must have 64-bit addressing"
        );
        assert!(
            ptx.contains(".entry broadcast_axes_f32"),
            "must have correct kernel name"
        );
    }

    #[test]
    fn broadcast_ptx_contains_stride_zero_trick_f32() {
        let t = BroadcastTemplate::new(PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("broadcast_axes_f32 generation failed");
        // Must load src with global load and store to dst
        assert!(ptx.contains("ld.global.f32"), "must have global load");
        assert!(ptx.contains("st.global.f32"), "must have global store");
        // Must have the dim-accumulation via div+rem for striding
        assert!(ptx.contains("div.u32"), "must divide for coord extraction");
        assert!(ptx.contains("rem.u32"), "must mod for coord extraction");
        // Must have predicated dimension guards
        assert!(
            ptx.contains("setp.gt.u32 %p_d0"),
            "must guard dim 0 with predicate"
        );
        assert!(
            ptx.contains("setp.gt.u32 %p_d7"),
            "must guard dim 7 with predicate"
        );
        // flat_src accumulation
        assert!(
            ptx.contains("add.u32 %flat_src"),
            "must accumulate flat_src"
        );
    }

    #[test]
    fn broadcast_ptx_contains_all_param_slots() {
        let t = BroadcastTemplate::new(PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("broadcast_axes_f32 generation failed");
        // Verify all 28 parameter slots are declared
        assert!(ptx.contains("%param_src_ptr"), "must have src_ptr param");
        assert!(ptx.contains("%param_dst_ptr"), "must have dst_ptr param");
        assert!(ptx.contains("%param_rank"), "must have rank param");
        for d in 0..MAX_BROADCAST_RANK {
            assert!(
                ptx.contains(&format!("%param_ds{d}")),
                "must have ds{d} param"
            );
            assert!(
                ptx.contains(&format!("%param_dst_s{d}")),
                "must have dst_s{d} param"
            );
            assert!(
                ptx.contains(&format!("%param_ss{d}")),
                "must have ss{d} param"
            );
        }
        assert!(ptx.contains("%param_n_dst"), "must have n_dst param");
    }

    #[test]
    fn broadcast_generates_for_f64() {
        let t = BroadcastTemplate::new(PtxType::F64, SmVersion::Sm90);
        let ptx = t.generate().expect("broadcast_axes_f64 generation failed");
        assert!(
            ptx.contains("broadcast_axes_f64"),
            "must have f64 kernel name"
        );
        assert!(ptx.contains("ld.global.f64"), "must have f64 load");
        assert!(ptx.contains("st.global.f64"), "must have f64 store");
    }

    #[test]
    fn broadcast_max_rank_is_eight() {
        assert_eq!(MAX_BROADCAST_RANK, 8);
    }
}
