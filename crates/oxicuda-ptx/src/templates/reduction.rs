//! Parallel reduction kernel templates.
//!
//! This module generates PTX kernels for block-level parallel reductions
//! over device arrays. The reduction uses a two-phase approach:
//!
//! 1. **Shared memory tree reduction**: Each thread loads one element into shared
//!    memory, then threads cooperatively reduce using a binary tree pattern
//!    with barrier synchronization at each level.
//! 2. **Warp shuffle final reduction**: The last 32 elements are reduced using
//!    `shfl.sync.down.b32` instructions, avoiding shared memory bank conflicts.
//!
//! Supported operations: sum, max, min, product, L1 norm, and L2 norm.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::reduction::{ReductionTemplate, ReductionOp};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let template = ReductionTemplate {
//!     op: ReductionOp::Sum,
//!     precision: PtxType::F32,
//!     target: SmVersion::Sm80,
//!     block_size: 256,
//! };
//! let ptx = template.generate().expect("PTX generation failed");
//! assert!(ptx.contains("shfl.sync.down"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;
use crate::ir::PtxType;

/// Reduction operation type.
///
/// Each variant determines the identity element and the combining operation
/// used during the tree reduction and warp shuffle phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    /// Summation: identity = 0, combine = add.
    Sum,
    /// Maximum: identity = -INF, combine = max.
    Max,
    /// Minimum: identity = +INF, combine = min.
    Min,
    /// Product: identity = 1, combine = mul.
    Prod,
    /// L1 norm: sum of absolute values, identity = 0, combine = add(abs).
    L1Norm,
    /// L2 norm: sqrt of sum of squares, identity = 0, combine = add(mul).
    L2Norm,
}

impl ReductionOp {
    /// Returns a short lowercase name for kernel naming.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Max => "max",
            Self::Min => "min",
            Self::Prod => "prod",
            Self::L1Norm => "l1norm",
            Self::L2Norm => "l2norm",
        }
    }

    /// Returns the PTX instruction that combines two values for this reduction.
    fn combine_instruction(self, ty_str: &str) -> String {
        match self {
            Self::Sum | Self::L1Norm | Self::L2Norm => format!("add{ty_str}"),
            Self::Max => format!("max{ty_str}"),
            Self::Min => format!("min{ty_str}"),
            Self::Prod => format!("mul{ty_str}"),
        }
    }

    /// Returns the PTX hex literal for the identity element of this operation.
    const fn identity_literal(self, precision: PtxType) -> &'static str {
        match (self, precision) {
            // Sum, L1Norm, L2Norm: identity = 0.0
            (Self::Sum | Self::L1Norm | Self::L2Norm, PtxType::F64) => "0d0000000000000000",
            (Self::Sum | Self::L1Norm | Self::L2Norm, _) => "0f00000000",
            // Prod: identity = 1.0
            (Self::Prod, PtxType::F64) => "0d3FF0000000000000",
            (Self::Prod, _) => "0f3F800000",
            // Max: identity = -INF
            (Self::Max, PtxType::F64) => "0dFFF0000000000000",
            (Self::Max, _) => "0fFF800000",
            // Min: identity = +INF
            (Self::Min, PtxType::F64) => "0d7FF0000000000000",
            (Self::Min, _) => "0f7F800000",
        }
    }
}

/// Template for generating parallel reduction PTX kernels.
///
/// The generated kernel performs a block-level reduction over an input array.
/// Each block produces a single partial result written to the output array.
/// For a complete reduction over large arrays, launch multiple blocks and
/// then reduce the partial results with a second kernel invocation.
///
/// # Block size requirements
///
/// The block size must be a power of 2 and at least 32 (one warp). Typical
/// values are 256 or 512.
pub struct ReductionTemplate {
    /// The reduction operation.
    pub op: ReductionOp,
    /// The data precision.
    pub precision: PtxType,
    /// The target GPU architecture.
    pub target: SmVersion,
    /// Number of threads per block (must be a power of 2, >= 32).
    pub block_size: u32,
}

impl ReductionTemplate {
    /// Returns the kernel function name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        let type_str = self.precision.as_ptx_str().trim_start_matches('.');
        format!(
            "reduce_{}_{}_bs{}",
            self.op.as_str(),
            type_str,
            self.block_size
        )
    }

    /// Generates the complete PTX module text for this reduction kernel.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if:
    /// - The precision type is not a supported floating-point type
    /// - The block size is not a power of 2 or is less than 32
    /// - PTX text formatting fails
    #[allow(clippy::too_many_lines)]
    pub fn generate(&self) -> Result<String, PtxGenError> {
        self.validate()?;

        let ty = self.precision.as_ptx_str();
        let byte_size = self.precision.size_bytes();
        let identity = self.op.identity_literal(self.precision);
        let combine = self.op.combine_instruction(ty);
        let smem_bytes = (self.block_size as usize) * byte_size;
        let kernel_name = self.kernel_name();

        let mut ptx = String::with_capacity(4096);

        // Header
        writeln!(ptx, ".version {}", self.target.ptx_version())
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".target {}", self.target.as_ptx_str()).map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ".address_size 64").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Kernel signature
        writeln!(ptx, ".visible .entry {kernel_name}(").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_input,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u64 %param_output,").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .param .u32 %param_n").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, ")").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "{{").map_err(PtxGenError::FormatError)?;

        // Directives and declarations
        writeln!(ptx, "    .maxntid {}, 1, 1;", self.block_size)
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b32 %r<16>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .b64 %rd<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .f32 %f<8>;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    .reg .pred %p<4>;").map_err(PtxGenError::FormatError)?;
        writeln!(
            ptx,
            "    .shared .align {} .b8 smem_reduce[{}];",
            byte_size.max(4),
            smem_bytes
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Compute global thread ID
        writeln!(ptx, "    // Compute global thread index").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r0, %tid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r1, %ctaid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u32 %r2, %ntid.x;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mad.lo.u32 %r3, %r1, %r2, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Load parameters
        writeln!(ptx, "    // Load parameters").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd0, [%param_input];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u64 %rd1, [%param_output];")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.param.u32 %r4, [%param_n];").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Bounds check: load element or identity
        writeln!(
            ptx,
            "    // Load element or use identity for out-of-bounds threads"
        )
        .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.u32 %p0, %r3, %r4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov{ty} %f0, {identity};").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p0 bra $SKIP_LOAD;").map_err(PtxGenError::FormatError)?;

        // Load from global memory
        writeln!(ptx, "    cvt.u64.u32 %rd2, %r3;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd2, %rd2, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd3, %rd0, %rd2;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ld.global{ty} %f0, [%rd3];").map_err(PtxGenError::FormatError)?;

        // Apply pre-processing for L1Norm (abs) or L2Norm (square)
        match self.op {
            ReductionOp::L1Norm => {
                writeln!(ptx, "    abs{ty} %f0, %f0;").map_err(PtxGenError::FormatError)?;
            }
            ReductionOp::L2Norm => {
                writeln!(ptx, "    mul{ty} %f0, %f0, %f0;").map_err(PtxGenError::FormatError)?;
            }
            _ => {}
        }

        writeln!(ptx, "$SKIP_LOAD:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Store to shared memory
        writeln!(ptx, "    // Store value to shared memory").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    cvt.u64.u32 %rd4, %r0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd4, %rd4, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mov.u64 %rd5, smem_reduce;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd5, %rd5, %rd4;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.shared{ty} [%rd5], %f0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Shared memory tree reduction (down to warp size)
        writeln!(ptx, "    // Shared memory tree reduction").map_err(PtxGenError::FormatError)?;
        let mut stride = self.block_size / 2;
        while stride > 16 {
            writeln!(ptx, "    setp.lt.u32 %p1, %r0, {stride};")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    @!%p1 bra $SKIP_S{stride};").map_err(PtxGenError::FormatError)?;

            // Load the partner element from shared memory
            let partner_offset = stride as usize * byte_size;
            writeln!(ptx, "    ld.shared{ty} %f1, [%rd5+{partner_offset}];")
                .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    ld.shared{ty} %f2, [%rd5];").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    {combine} %f2, %f2, %f1;").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    st.shared{ty} [%rd5], %f2;").map_err(PtxGenError::FormatError)?;

            writeln!(ptx, "$SKIP_S{stride}:").map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    bar.sync 0;").map_err(PtxGenError::FormatError)?;
            stride /= 2;
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Warp shuffle reduction for the final 32 elements
        writeln!(ptx, "    // Warp shuffle reduction (final 32 elements)")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.lt.u32 %p2, %r0, 32;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p2 bra $DONE;").map_err(PtxGenError::FormatError)?;

        // Load from shared memory into register for warp shuffle
        writeln!(ptx, "    ld.shared{ty} %f3, [%rd5];").map_err(PtxGenError::FormatError)?;

        // Warp shuffle down for offsets 16, 8, 4, 2, 1
        for shfl_offset in [16u32, 8, 4, 2, 1] {
            writeln!(
                ptx,
                "    shfl.sync.down.b32 %f4, %f3, {shfl_offset}, 31, 0xFFFFFFFF;"
            )
            .map_err(PtxGenError::FormatError)?;
            writeln!(ptx, "    {combine} %f3, %f3, %f4;").map_err(PtxGenError::FormatError)?;
        }
        writeln!(ptx).map_err(PtxGenError::FormatError)?;

        // Thread 0 writes block result
        writeln!(ptx, "    // Thread 0 writes block result").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    setp.eq.u32 %p3, %r0, 0;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    @!%p3 bra $DONE;").map_err(PtxGenError::FormatError)?;

        // For L2Norm, apply sqrt to the final result
        if self.op == ReductionOp::L2Norm {
            writeln!(ptx, "    sqrt.rn{ty} %f3, %f3;").map_err(PtxGenError::FormatError)?;
        }

        // Compute output address: output[blockIdx.x]
        writeln!(ptx, "    cvt.u64.u32 %rd6, %r1;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    mul.lo.u64 %rd6, %rd6, {byte_size};")
            .map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    add.u64 %rd7, %rd1, %rd6;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    st.global{ty} [%rd7], %f3;").map_err(PtxGenError::FormatError)?;

        writeln!(ptx, "$DONE:").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "    ret;").map_err(PtxGenError::FormatError)?;
        writeln!(ptx, "}}").map_err(PtxGenError::FormatError)?;

        Ok(ptx)
    }

    /// Validates template parameters.
    fn validate(&self) -> Result<(), PtxGenError> {
        if !matches!(
            self.precision,
            PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64
        ) {
            return Err(PtxGenError::InvalidType(format!(
                "reduction requires F16, BF16, F32, or F64, got {}",
                self.precision.as_ptx_str()
            )));
        }

        if self.block_size < 32 || !self.block_size.is_power_of_two() {
            return Err(PtxGenError::GenerationFailed(format!(
                "block_size must be a power of 2 >= 32, got {}",
                self.block_size
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SmVersion;

    #[test]
    fn reduction_op_names() {
        assert_eq!(ReductionOp::Sum.as_str(), "sum");
        assert_eq!(ReductionOp::Max.as_str(), "max");
        assert_eq!(ReductionOp::L2Norm.as_str(), "l2norm");
    }

    #[test]
    fn kernel_name_format() {
        let t = ReductionTemplate {
            op: ReductionOp::Sum,
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            block_size: 256,
        };
        assert_eq!(t.kernel_name(), "reduce_sum_f32_bs256");
    }

    #[test]
    fn invalid_block_size() {
        let t = ReductionTemplate {
            op: ReductionOp::Sum,
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            block_size: 100,
        };
        assert!(t.generate().is_err());
    }

    #[test]
    fn invalid_precision() {
        let t = ReductionTemplate {
            op: ReductionOp::Sum,
            precision: PtxType::U32,
            target: SmVersion::Sm80,
            block_size: 256,
        };
        assert!(t.generate().is_err());
    }

    #[test]
    fn generate_sum_f32() {
        let t = ReductionTemplate {
            op: ReductionOp::Sum,
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            block_size: 256,
        };
        let ptx = t.generate().expect("should generate sum kernel");
        assert!(ptx.contains(".entry reduce_sum_f32_bs256"));
        assert!(ptx.contains("shfl.sync.down"));
        assert!(ptx.contains("bar.sync 0"));
        assert!(ptx.contains(".shared"));
    }

    #[test]
    fn generate_max_f32() {
        let t = ReductionTemplate {
            op: ReductionOp::Max,
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            block_size: 256,
        };
        let ptx = t.generate().expect("should generate max kernel");
        assert!(ptx.contains("max.f32"));
    }

    #[test]
    fn generate_l2norm_f32() {
        let t = ReductionTemplate {
            op: ReductionOp::L2Norm,
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            block_size: 256,
        };
        let ptx = t.generate().expect("should generate l2norm kernel");
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("sqrt.rn.f32"));
    }

    #[test]
    fn generate_small_block() {
        let t = ReductionTemplate {
            op: ReductionOp::Sum,
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            block_size: 32,
        };
        let ptx = t.generate().expect("should generate with block_size=32");
        assert!(ptx.contains("shfl.sync.down"));
    }
}
