//! Epilogue operations for GEMM kernels.
//!
//! After the matrix multiplication accumulation phase, an epilogue operation
//! is applied to the result before writing to the output matrix. This module
//! defines the [`EpilogueOp`] enum and provides PTX code generation helpers
//! for fusing post-GEMM operations into the kernel.
//!
//! # Supported epilogues
//!
//! | Variant | Formula |
//! |---------|---------|
//! | `LinearCombination` | `C = alpha * acc + beta * C` |
//! | `LinearCombinationRelu` | `C = max(0, alpha * acc + beta * C)` |
//! | `LinearCombinationGelu` | `C = gelu(alpha * acc + beta * C)` |
//! | `LinearCombinationBias` | `C = alpha * acc + beta * C + bias` |
//! | `LinearCombinationBiasRelu` | `C = max(0, alpha * acc + beta * C + bias)` |

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::ir::PtxType;

use crate::error::{BlasError, BlasResult};

// ---------------------------------------------------------------------------
// EpilogueOp
// ---------------------------------------------------------------------------

/// Post-GEMM fused operation applied to the accumulator before writing to C.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpilogueOp {
    /// `C = alpha * acc + beta * C`.
    LinearCombination,
    /// `C = max(0, alpha * acc + beta * C)`.
    LinearCombinationRelu,
    /// `C = gelu(alpha * acc + beta * C)`.
    LinearCombinationGelu,
    /// `C = alpha * acc + beta * C + bias[col]`.
    LinearCombinationBias,
    /// `C = max(0, alpha * acc + beta * C + bias[col])`.
    LinearCombinationBiasRelu,
}

impl EpilogueOp {
    /// Returns a short label for kernel naming.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LinearCombination => "lincomb",
            Self::LinearCombinationRelu => "lincomb_relu",
            Self::LinearCombinationGelu => "lincomb_gelu",
            Self::LinearCombinationBias => "lincomb_bias",
            Self::LinearCombinationBiasRelu => "lincomb_bias_relu",
        }
    }

    /// Returns `true` if this epilogue requires a bias vector parameter.
    pub fn needs_bias(self) -> bool {
        matches!(
            self,
            Self::LinearCombinationBias | Self::LinearCombinationBiasRelu
        )
    }

    /// Returns `true` if this epilogue applies a ReLU activation.
    pub fn has_relu(self) -> bool {
        matches!(
            self,
            Self::LinearCombinationRelu | Self::LinearCombinationBiasRelu
        )
    }

    /// Returns `true` if this epilogue applies a GELU activation.
    pub fn has_gelu(self) -> bool {
        matches!(self, Self::LinearCombinationGelu)
    }

    /// Converts to the corresponding [`oxicuda_ptx::templates::gemm::EpilogueKind`] used by `oxicuda-ptx`.
    pub fn to_ptx_kind(self) -> oxicuda_ptx::templates::gemm::EpilogueKind {
        match self {
            Self::LinearCombination => {
                oxicuda_ptx::templates::gemm::EpilogueKind::LinearCombination
            }
            Self::LinearCombinationRelu => {
                oxicuda_ptx::templates::gemm::EpilogueKind::LinearCombinationRelu
            }
            Self::LinearCombinationGelu => {
                oxicuda_ptx::templates::gemm::EpilogueKind::LinearCombinationGelu
            }
            Self::LinearCombinationBias => {
                oxicuda_ptx::templates::gemm::EpilogueKind::LinearCombinationBias
            }
            Self::LinearCombinationBiasRelu => {
                oxicuda_ptx::templates::gemm::EpilogueKind::LinearCombinationBiasRelu
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PTX epilogue snippet generation
// ---------------------------------------------------------------------------

/// Generates PTX instructions for the linear combination epilogue.
///
/// Assumes the accumulator value is in `%f_acc`, the alpha scalar in
/// `%f_alpha`, the beta scalar in `%f_beta`, and the old C value in
/// `%f_cold`. The result is written to `%f_result`.
///
/// # Arguments
///
/// * `acc_type` — the accumulator's PTX type (F32 or F64).
/// * `op` — the epilogue operation to generate.
///
/// # Returns
///
/// A string of PTX instructions implementing the epilogue.
pub fn generate_epilogue_ptx(acc_type: PtxType, op: EpilogueOp) -> BlasResult<String> {
    let ty = acc_type.as_ptx_str();
    let mut ptx = String::with_capacity(512);

    // Base: result = alpha * acc + beta * c_old
    write_line(
        &mut ptx,
        &format!("    mul{ty} %f_result, %f_acc, %f_alpha;"),
    )?;
    write_line(
        &mut ptx,
        &format!("    fma.rn{ty} %f_result, %f_beta, %f_cold, %f_result;"),
    )?;

    // Bias addition (if needed).
    if op.needs_bias() {
        write_line(
            &mut ptx,
            &format!("    add{ty} %f_result, %f_result, %f_bias;"),
        )?;
    }

    // Activation functions.
    if op.has_relu() {
        // ReLU: result = max(0, result)
        write_line(
            &mut ptx,
            &format!("    max{ty} %f_result, %f_result, 0f00000000;"),
        )?;
    } else if op.has_gelu() {
        // GELU approximation: result = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // For GPU, we use the fast approximation: x * sigmoid(1.702 * x)
        // But in PTX we approximate with: result * 0.5 * (1 + tanh(0.7978845608 * (result + 0.044715 * result^3)))
        // Simplified: use the PTX `ex2` and division approach.
        //
        // For now, use the simpler sigmoid-based GELU:
        //   gelu(x) ~= x * 0.5 * (1 + erf(x / sqrt(2)))
        // Approximated as: x * sigmoid(1.702 * x)
        write_line(
            &mut ptx,
            &format!("    mul{ty} %f_gelu_s, %f_result, 0f3FDA6286;  // 1.702"),
        )?;
        // sigmoid(x) = 1 / (1 + exp(-x)) ~= use neg + ex2 + add + rcp
        write_line(&mut ptx, &format!("    neg{ty} %f_gelu_s, %f_gelu_s;"))?;
        // exp(x) = 2^(x * log2(e)) = 2^(x * 1.4426950408...)
        write_line(
            &mut ptx,
            &format!("    mul{ty} %f_gelu_s, %f_gelu_s, 0f3FB8AA3B;  // log2(e)"),
        )?;
        write_line(
            &mut ptx,
            &format!("    ex2.approx{ty} %f_gelu_s, %f_gelu_s;"),
        )?;
        write_line(
            &mut ptx,
            &format!("    add{ty} %f_gelu_s, %f_gelu_s, 0f3F800000;  // +1.0"),
        )?;
        write_line(
            &mut ptx,
            &format!("    rcp.approx{ty} %f_gelu_s, %f_gelu_s;"),
        )?;
        write_line(
            &mut ptx,
            &format!("    mul{ty} %f_result, %f_result, %f_gelu_s;"),
        )?;
    }

    Ok(ptx)
}

/// Writes a line, mapping fmt errors.
fn write_line(ptx: &mut String, line: &str) -> BlasResult<()> {
    writeln!(ptx, "{line}").map_err(|e| BlasError::PtxGeneration(format!("fmt error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epilogue_op_labels() {
        assert_eq!(EpilogueOp::LinearCombination.as_str(), "lincomb");
        assert_eq!(EpilogueOp::LinearCombinationRelu.as_str(), "lincomb_relu");
        assert_eq!(EpilogueOp::LinearCombinationGelu.as_str(), "lincomb_gelu");
        assert_eq!(EpilogueOp::LinearCombinationBias.as_str(), "lincomb_bias");
        assert_eq!(
            EpilogueOp::LinearCombinationBiasRelu.as_str(),
            "lincomb_bias_relu"
        );
    }

    #[test]
    fn epilogue_needs_bias() {
        assert!(!EpilogueOp::LinearCombination.needs_bias());
        assert!(!EpilogueOp::LinearCombinationRelu.needs_bias());
        assert!(EpilogueOp::LinearCombinationBias.needs_bias());
        assert!(EpilogueOp::LinearCombinationBiasRelu.needs_bias());
    }

    #[test]
    fn epilogue_has_relu() {
        assert!(!EpilogueOp::LinearCombination.has_relu());
        assert!(EpilogueOp::LinearCombinationRelu.has_relu());
        assert!(EpilogueOp::LinearCombinationBiasRelu.has_relu());
    }

    #[test]
    fn generate_linear_combination() {
        let ptx = generate_epilogue_ptx(PtxType::F32, EpilogueOp::LinearCombination)
            .expect("epilogue generation failed");
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(!ptx.contains("max.f32"));
    }

    #[test]
    fn generate_relu_epilogue() {
        let ptx = generate_epilogue_ptx(PtxType::F32, EpilogueOp::LinearCombinationRelu)
            .expect("relu epilogue generation failed");
        assert!(ptx.contains("max.f32"));
    }

    #[test]
    fn generate_bias_epilogue() {
        let ptx = generate_epilogue_ptx(PtxType::F32, EpilogueOp::LinearCombinationBias)
            .expect("bias epilogue generation failed");
        assert!(ptx.contains("add.f32 %f_result, %f_result, %f_bias"));
    }

    #[test]
    fn generate_gelu_epilogue() {
        let ptx = generate_epilogue_ptx(PtxType::F32, EpilogueOp::LinearCombinationGelu)
            .expect("gelu epilogue generation failed");
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
    }

    #[test]
    fn to_ptx_kind_roundtrip() {
        // Verify the mapping doesn't panic for all variants.
        let _ = EpilogueOp::LinearCombination.to_ptx_kind();
        let _ = EpilogueOp::LinearCombinationRelu.to_ptx_kind();
        let _ = EpilogueOp::LinearCombinationGelu.to_ptx_kind();
        let _ = EpilogueOp::LinearCombinationBias.to_ptx_kind();
        let _ = EpilogueOp::LinearCombinationBiasRelu.to_ptx_kind();
    }
}
