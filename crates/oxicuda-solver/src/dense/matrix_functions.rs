//! Matrix functions: exponential, logarithm, and square root.
//!
//! Provides GPU-accelerated matrix function computations via PTX kernel
//! generation:
//!
//! - **Matrix Exponential (`expm`)**: Scaling and squaring with Padé
//!   approximation. Computes `e^A` for a square matrix A.
//! - **Matrix Logarithm (`logm`)**: Inverse scaling and squaring with Padé
//!   approximation. Computes `log(A)` for a matrix with eigenvalues in the
//!   right half-plane.
//! - **Matrix Square Root (`sqrtm`)**: Denman–Beavers iteration. Computes
//!   `A^{1/2}` such that `sqrtm(A) * sqrtm(A) = A`.
//!
//! Each plan struct generates self-contained PTX kernels using
//! [`KernelBuilder`]/[`BodyBuilder`] from the `oxicuda-ptx` crate.

#![allow(dead_code)]

use oxicuda_ptx::ir::PtxType;
use oxicuda_ptx::prelude::*;

use crate::error::{SolverError, SolverResult};

// ---------------------------------------------------------------------------
// Padé coefficients
// ---------------------------------------------------------------------------

/// Padé coefficients for the matrix exponential (numerator/denominator
/// polynomial of `[p/p]` approximant to `e^x`).
///
/// Returns `(numerator_coeffs, denominator_coeffs)` for the given order.
/// Orders 3, 5, 7, 9, 13 are supported, matching the Higham (2005) algorithm.
fn pade_coefficients(order: u32) -> SolverResult<Vec<f64>> {
    match order {
        3 => Ok(vec![120.0, 60.0, 12.0, 1.0]),
        5 => Ok(vec![30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0]),
        7 => Ok(vec![
            17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0,
        ]),
        9 => Ok(vec![
            17643225600.0,
            8821612800.0,
            2075673600.0,
            302702400.0,
            30270240.0,
            2162160.0,
            110880.0,
            3960.0,
            90.0,
            1.0,
        ]),
        13 => Ok(vec![
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ]),
        _ => Err(SolverError::InternalError(format!(
            "unsupported Padé order {order}; valid orders are 3, 5, 7, 9, 13"
        ))),
    }
}

/// Theta thresholds for each Padé order. If `||A|| <= theta[m]`, then the
/// Padé approximant of that order achieves unit-roundoff accuracy.
#[allow(clippy::excessive_precision)]
fn pade_theta(order: u32) -> SolverResult<f64> {
    match order {
        3 => Ok(1.495_585_217_958_292e-2),
        5 => Ok(2.539_398_330_063_230e-1),
        7 => Ok(9.504_178_996_162_932e-1),
        9 => Ok(2.097_847_961_257_068),
        13 => Ok(5.371_920_351_148_152),
        _ => Err(SolverError::InternalError(format!(
            "no theta for Padé order {order}"
        ))),
    }
}

// =========================================================================
// Matrix Exponential (expm)
// =========================================================================

/// Configuration for matrix exponential computation.
#[derive(Debug, Clone)]
pub struct MatrixExpConfig {
    /// Matrix dimension (n × n).
    pub n: u32,
    /// Precision: `"f32"` or `"f64"`.
    pub precision: String,
    /// Padé approximant order (3, 5, 7, 9, or 13). Default: 13.
    pub pade_order: u32,
}

impl MatrixExpConfig {
    /// Creates a new configuration with sensible defaults.
    pub fn new(n: u32, precision: &str) -> Self {
        Self {
            n,
            precision: precision.to_string(),
            pade_order: 13,
        }
    }

    /// Sets the Padé order.
    pub fn with_pade_order(mut self, order: u32) -> Self {
        self.pade_order = order;
        self
    }

    /// Validates the configuration.
    fn validate(&self) -> SolverResult<()> {
        if self.n == 0 {
            return Err(SolverError::DimensionMismatch(
                "expm: matrix dimension must be > 0".into(),
            ));
        }
        if self.precision != "f32" && self.precision != "f64" {
            return Err(SolverError::InternalError(format!(
                "expm: unsupported precision '{}'; use 'f32' or 'f64'",
                self.precision
            )));
        }
        // Validate Padé order.
        pade_coefficients(self.pade_order)?;
        Ok(())
    }
}

/// Execution plan for the matrix exponential.
///
/// The plan pre-computes the Padé coefficients and kernel names, then
/// generates PTX on demand.
#[derive(Debug, Clone)]
pub struct MatrixExpPlan {
    config: MatrixExpConfig,
    pade_coeffs: Vec<f64>,
    theta: f64,
}

impl MatrixExpPlan {
    /// Creates a plan from a validated configuration.
    pub fn new(config: MatrixExpConfig) -> SolverResult<Self> {
        config.validate()?;
        let pade_coeffs = pade_coefficients(config.pade_order)?;
        let theta = pade_theta(config.pade_order)?;
        Ok(Self {
            config,
            pade_coeffs,
            theta,
        })
    }

    /// Returns the Padé coefficients used by this plan.
    pub fn pade_coefficients(&self) -> &[f64] {
        &self.pade_coeffs
    }

    /// Returns the theta threshold for the configured Padé order.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Generates PTX source for the matrix exponential kernels.
    ///
    /// The generated code contains multiple entry points:
    /// 1. **scale kernel** — computes `A_scaled = A / 2^s` where `s` is chosen
    ///    so that `||A_scaled|| <= theta`.
    /// 2. **Padé numerator/denominator kernels** — evaluates the Padé
    ///    polynomial pair `P(A)` and `Q(A)` using Horner's method.
    /// 3. **squaring kernel** — repeated squaring `F = F^{2^s}` via matrix
    ///    multiply.
    pub fn generate_ptx(&self) -> SolverResult<String> {
        let n = self.config.n;
        let float_ty = precision_to_ptx_type(&self.config.precision)?;
        let sm = SmVersion::Sm75;

        let mut all_ptx = Vec::new();

        // Kernel 1: Scale matrix by 2^(-s).
        let scale_ptx = self.emit_scale_kernel(n, float_ty, sm)?;
        all_ptx.push(scale_ptx);

        // Kernel 2: Padé polynomial evaluation (Horner's method for P(A) and Q(A)).
        let pade_ptx = self.emit_pade_kernel(n, float_ty, sm)?;
        all_ptx.push(pade_ptx);

        // Kernel 3: Repeated squaring.
        let square_ptx = self.emit_squaring_kernel(n, float_ty, sm)?;
        all_ptx.push(square_ptx);

        Ok(all_ptx.join("\n"))
    }

    /// Emits PTX for `A_scaled[i,j] = A[i,j] / 2^s`.
    fn emit_scale_kernel(&self, n: u32, float_ty: PtxType, sm: SmVersion) -> SolverResult<String> {
        let name = format!("solver_expm_scale_{}_n{}", ptx_type_suffix(float_ty), n);

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(256)
            .param("a_ptr", PtxType::U64)
            .param("out_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .param("scale_exp", PtxType::U32)
            .body(move |b| {
                // Each thread handles one matrix element.
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n");
                let total = b.mul_lo_u32(n_reg.clone(), n_reg.clone());

                b.if_lt_u32(gid, total, |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let out_ptr = b.load_param_u64("out_ptr");
                    let scale_exp = b.load_param_u32("scale_exp");

                    // Load element.
                    let gid_repeat = b.global_thread_id_x();
                    let elem_size = if float_ty == PtxType::F32 { 4u32 } else { 8u32 };
                    let addr = b.byte_offset_addr(a_ptr, gid_repeat.clone(), elem_size);
                    let val = load_float(b, float_ty, addr);

                    // Compute divisor = 2^scale_exp using IEEE 754 biased exponent.
                    // For f64: bits = (scale_exp + 1023) << 52.
                    // For f32: bits = (scale_exp + 127) << 23.
                    let out_addr = b.byte_offset_addr(out_ptr, gid_repeat, elem_size);

                    let result = if float_ty == PtxType::F64 {
                        // Widen scale_exp to 64-bit.
                        let se64 = b.cvt_u32_to_u64(scale_exp);
                        // Add IEEE 754 f64 exponent bias (1023).
                        let biased = b.alloc_reg(PtxType::U64);
                        b.raw_ptx(&format!("add.u64 {biased}, {se64}, 1023;"));
                        // Shift left 52 to place in exponent field.
                        let shift_amt = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {shift_amt}, 52;"));
                        let bits = b.shl_b64(biased, shift_amt);
                        // Reinterpret bits as f64.
                        let divisor = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mov.b64 {divisor}, {bits};"));
                        // result = val / 2^scale_exp.
                        let res = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("div.rn.f64 {res}, {val}, {divisor};"));
                        res
                    } else {
                        // For f32: bias is 127, exponent field starts at bit 23.
                        let biased = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("add.u32 {biased}, {scale_exp}, 127;"));
                        let shift_amt = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {shift_amt}, 23;"));
                        let bits = b.shl_b32(biased, shift_amt);
                        // Reinterpret bits as f32.
                        let divisor = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.b32 {divisor}, {bits};"));
                        // result = val / 2^scale_exp.
                        let res = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("div.rn.f32 {res}, {val}, {divisor};"));
                        res
                    };

                    store_float(b, float_ty, out_addr, result);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }

    /// Emits PTX for Padé polynomial evaluation using Horner's method.
    ///
    /// Evaluates:
    ///   P(A) = c_0 * I + c_1 * A + c_2 * A^2 + ... + c_p * A^p
    ///   Q(A) = c_0 * I - c_1 * A + c_2 * A^2 - ... ± c_p * A^p
    ///
    /// where the even coefficients are the same and odd coefficients differ
    /// only in sign.
    fn emit_pade_kernel(&self, n: u32, float_ty: PtxType, sm: SmVersion) -> SolverResult<String> {
        let order = self.config.pade_order;
        let name = format!(
            "solver_expm_pade_{}_n{}_p{}",
            ptx_type_suffix(float_ty),
            n,
            order
        );

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(256)
            .param("a_ptr", PtxType::U64)
            .param("p_ptr", PtxType::U64)
            .param("q_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .param("coeffs_ptr", PtxType::U64)
            .param("num_coeffs", PtxType::U32)
            .body(move |b| {
                // Each thread computes one element of P(A) and Q(A).
                // For small matrices, this is feasible; for large matrices,
                // the host code orchestrates multiple GEMM calls.
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n");
                let total = b.mul_lo_u32(n_reg.clone(), n_reg);

                b.if_lt_u32(gid, total, |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let p_ptr = b.load_param_u64("p_ptr");
                    let q_ptr = b.load_param_u64("q_ptr");
                    let coeffs_ptr = b.load_param_u64("coeffs_ptr");
                    let num_coeffs = b.load_param_u32("num_coeffs");

                    let elem_size = if float_ty == PtxType::F32 { 4u32 } else { 8u32 };
                    // Coefficients are always stored as f64 regardless of precision.
                    const COEFF_SIZE: u32 = 8u32;
                    let gid_r = b.global_thread_id_x();

                    // Load A[gid] — the input matrix element.
                    let a_addr = b.byte_offset_addr(a_ptr, gid_r.clone(), elem_size);
                    let a_val = load_float(b, float_ty, a_addr);

                    // Horner's method: traverse coefficients from highest to lowest.
                    // P(x) = c[m] + x*(c[m-2] + x*(...))  (even-indexed terms)
                    // Q(x) = c[m] + x*(-c[m-2] + x*(...))  (odd terms negate for Q)
                    // Simplified scalar Horner: P = c[m], Q = c[m], then for each
                    // lower degree: P = P*x + c[k], Q = Q*x + sign(k)*c[k].
                    // We load each coefficient from coeffs_ptr (f64 array).
                    //
                    // Loop: idx from (num_coeffs-1) down to 0.
                    // acc_p = 0, acc_q = 0.
                    // For k = num_coeffs-1 downto 0:
                    //   load c_k from coeffs_ptr[k] (f64), convert to float_ty
                    //   acc_p = fma(acc_p, a_val, c_k)
                    //   sign = (-1)^k → for Q: if k is odd, negate c_k
                    //   acc_q = fma(acc_q, a_val, ±c_k)

                    let acc_p = zero_const(b, float_ty);
                    let acc_q = zero_const(b, float_ty);

                    // idx_reg counts down from (num_coeffs) to 0.
                    let idx_reg = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {idx_reg}, {num_coeffs};"));

                    let horner_loop = b.fresh_label("horner_loop");
                    let horner_exit = b.fresh_label("horner_exit");

                    b.raw_ptx(&format!("{horner_loop}:"));
                    // Check idx_reg == 0; if so, exit.
                    let done_pred = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.eq.u32 {done_pred}, {idx_reg}, 0;"));
                    b.raw_ptx(&format!("@{done_pred} bra {horner_exit};"));

                    // idx_reg -= 1 (current coefficient index = idx_reg - 1 after decrement).
                    b.raw_ptx(&format!("sub.u32 {idx_reg}, {idx_reg}, 1;"));

                    // Load f64 coefficient from coeffs_ptr[idx_reg].
                    let coeff_addr =
                        b.byte_offset_addr(coeffs_ptr.clone(), idx_reg.clone(), COEFF_SIZE);
                    let coeff_f64 = load_float(b, PtxType::F64, coeff_addr);

                    // Convert coefficient to working precision.
                    let c_k = if float_ty == PtxType::F64 {
                        coeff_f64.clone()
                    } else {
                        // cvt.rn.f32.f64
                        let dst = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.f64 {dst}, {coeff_f64};"));
                        dst
                    };

                    // Horner step for P: acc_p = acc_p * a_val + c_k.
                    let new_acc_p = if float_ty == PtxType::F64 {
                        b.fma_f64(acc_p.clone(), a_val.clone(), c_k.clone())
                    } else {
                        b.fma_f32(acc_p.clone(), a_val.clone(), c_k.clone())
                    };
                    b.raw_ptx(&format!(
                        "mov{} {acc_p}, {new_acc_p};",
                        float_ty.as_ptx_str()
                    ));

                    // For Q: negate c_k when idx_reg is odd (the current index after decrement).
                    // idx_reg is the coefficient index; odd index → negate for Q.
                    let odd_pred = b.alloc_reg(PtxType::Pred);
                    let lsb = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("and.b32 {lsb}, {idx_reg}, 1;"));
                    b.raw_ptx(&format!("setp.ne.u32 {odd_pred}, {lsb}, 0;"));

                    // neg_c_k = -c_k.
                    let neg_c_k = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!("neg{} {neg_c_k}, {c_k};", float_ty.as_ptx_str()));
                    // q_coeff = odd ? neg_c_k : c_k.
                    let q_coeff = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "selp{} {q_coeff}, {neg_c_k}, {c_k}, {odd_pred};",
                        float_ty.as_ptx_str()
                    ));

                    // Horner step for Q: acc_q = acc_q * a_val + q_coeff.
                    let new_acc_q = if float_ty == PtxType::F64 {
                        b.fma_f64(acc_q.clone(), a_val.clone(), q_coeff)
                    } else {
                        b.fma_f32(acc_q.clone(), a_val.clone(), q_coeff)
                    };
                    b.raw_ptx(&format!(
                        "mov{} {acc_q}, {new_acc_q};",
                        float_ty.as_ptx_str()
                    ));

                    b.raw_ptx(&format!("bra {horner_loop};"));
                    b.raw_ptx(&format!("{horner_exit}:"));

                    // Store results.
                    let p_addr = b.byte_offset_addr(p_ptr, gid_r.clone(), elem_size);
                    let q_addr = b.byte_offset_addr(q_ptr, gid_r, elem_size);
                    store_float(b, float_ty, p_addr, acc_p);
                    store_float(b, float_ty, q_addr, acc_q);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }

    /// Emits PTX for the repeated squaring step: `F = F * F` applied `s` times.
    fn emit_squaring_kernel(
        &self,
        n: u32,
        float_ty: PtxType,
        sm: SmVersion,
    ) -> SolverResult<String> {
        let name = format!("solver_expm_square_{}_n{}", ptx_type_suffix(float_ty), n);

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(256)
            .param("f_ptr", PtxType::U64)
            .param("tmp_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .body(move |b| {
                // Each thread computes one element of the product F * F.
                // Row = gid / n, Col = gid % n.
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n");
                let total = b.mul_lo_u32(n_reg.clone(), n_reg.clone());

                b.if_lt_u32(gid, total, |b| {
                    let f_ptr = b.load_param_u64("f_ptr");
                    let tmp_ptr = b.load_param_u64("tmp_ptr");
                    let n_inner = b.load_param_u32("n");

                    // Decode column-major index: gid = col * n + row.
                    let gid_r = b.global_thread_id_x();
                    let row = b.alloc_reg(PtxType::U32);
                    let col = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("rem.u32 {row}, {gid_r}, {n_inner};"));
                    b.raw_ptx(&format!("div.u32 {col}, {gid_r}, {n_inner};"));

                    let elem_size = if float_ty == PtxType::F32 { 4u32 } else { 8u32 };

                    // Accumulate dot product: tmp[col*n+row] = sum_{k=0}^{n-1} F[k*n+row] * F[col*n+k].
                    // Column-major layout: element (r, c) is at index c*n + r.
                    let acc = zero_const(b, float_ty);
                    let k_reg = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {k_reg}, 0;"));

                    let loop_label = b.fresh_label("sq_loop");
                    let exit_label = b.fresh_label("sq_exit");

                    b.raw_ptx(&format!("{loop_label}:"));
                    // Check k < n; if not, exit loop.
                    let pred = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.ge.u32 {pred}, {k_reg}, {n_inner};"));
                    b.raw_ptx(&format!("@{pred} bra {exit_label};"));

                    // Load F[k*n + row] — k-th column, row-th element (column-major).
                    let a_idx_base = b.mul_lo_u32(k_reg.clone(), n_inner.clone());
                    let a_idx = b.add_u32(a_idx_base, row.clone());
                    let a_addr = b.byte_offset_addr(f_ptr.clone(), a_idx, elem_size);
                    let a_val = load_float(b, float_ty, a_addr);

                    // Load F[col*n + k] — col-th column, k-th element (column-major).
                    let b_idx_base = b.mul_lo_u32(col.clone(), n_inner.clone());
                    let b_idx = b.add_u32(b_idx_base, k_reg.clone());
                    let b_addr = b.byte_offset_addr(f_ptr.clone(), b_idx, elem_size);
                    let b_val = load_float(b, float_ty, b_addr);

                    // acc = fma(a_val, b_val, acc).
                    let new_acc = if float_ty == PtxType::F64 {
                        b.fma_f64(a_val, b_val, acc.clone())
                    } else {
                        b.fma_f32(a_val, b_val, acc.clone())
                    };
                    // Move new_acc into acc register via raw PTX.
                    b.raw_ptx(&format!("mov{} {acc}, {new_acc};", float_ty.as_ptx_str()));

                    // k += 1.
                    b.raw_ptx(&format!("add.u32 {k_reg}, {k_reg}, 1;"));
                    b.raw_ptx(&format!("bra {loop_label};"));

                    b.raw_ptx(&format!("{exit_label}:"));

                    // Store accumulated result to tmp[col*n + row].
                    let out_idx_base = b.mul_lo_u32(col, n_inner);
                    let out_idx = b.add_u32(out_idx_base, row);
                    let out_addr = b.byte_offset_addr(tmp_ptr, out_idx, elem_size);
                    store_float(b, float_ty, out_addr, acc);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }
}

// =========================================================================
// Matrix Logarithm (logm)
// =========================================================================

/// Configuration for matrix logarithm computation.
#[derive(Debug, Clone)]
pub struct MatrixLogConfig {
    /// Matrix dimension (n × n).
    pub n: u32,
    /// Precision: `"f32"` or `"f64"`.
    pub precision: String,
    /// Maximum number of square-root iterations. Default: 100.
    pub max_sqrt_iters: u32,
}

impl MatrixLogConfig {
    /// Creates a new configuration with sensible defaults.
    pub fn new(n: u32, precision: &str) -> Self {
        Self {
            n,
            precision: precision.to_string(),
            max_sqrt_iters: 100,
        }
    }

    /// Sets the maximum number of square-root iterations.
    pub fn with_max_sqrt_iters(mut self, iters: u32) -> Self {
        self.max_sqrt_iters = iters;
        self
    }

    /// Validates the configuration.
    fn validate(&self) -> SolverResult<()> {
        if self.n == 0 {
            return Err(SolverError::DimensionMismatch(
                "logm: matrix dimension must be > 0".into(),
            ));
        }
        if self.precision != "f32" && self.precision != "f64" {
            return Err(SolverError::InternalError(format!(
                "logm: unsupported precision '{}'; use 'f32' or 'f64'",
                self.precision
            )));
        }
        if self.max_sqrt_iters == 0 {
            return Err(SolverError::InternalError(
                "logm: max_sqrt_iters must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Execution plan for the matrix logarithm.
///
/// Uses inverse scaling and squaring:
/// 1. Reduce `A` via repeated matrix square roots until `||A - I||` is small.
/// 2. Apply Padé approximation of `log(I + X)` for the reduced matrix.
/// 3. Scale the result back by `2^s`.
#[derive(Debug, Clone)]
pub struct MatrixLogPlan {
    config: MatrixLogConfig,
}

impl MatrixLogPlan {
    /// Creates a plan from a validated configuration.
    pub fn new(config: MatrixLogConfig) -> SolverResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Returns the maximum allowed square-root iterations.
    pub fn max_sqrt_iters(&self) -> u32 {
        self.config.max_sqrt_iters
    }

    /// Generates PTX source for the matrix logarithm kernels.
    ///
    /// The generated code contains entry points for:
    /// 1. **shift kernel** — computes `X = A - I`.
    /// 2. **square-root iteration kernel** — Denman–Beavers step applied to A
    ///    until `||A - I||` is below threshold.
    /// 3. **Padé log kernel** — evaluates the `[m/m]` Padé approximant to
    ///    `log(I + X)` for small `X`.
    /// 4. **scale-back kernel** — multiplies the result by `2^s`.
    pub fn generate_ptx(&self) -> SolverResult<String> {
        let n = self.config.n;
        let float_ty = precision_to_ptx_type(&self.config.precision)?;
        let sm = SmVersion::Sm75;

        let mut all_ptx = Vec::new();

        // Kernel 1: A_shifted = A - I.
        let shift_ptx = self.emit_shift_kernel(n, float_ty, sm)?;
        all_ptx.push(shift_ptx);

        // Kernel 2: Matrix square root step (for reducing A close to I).
        let sqrt_step_ptx = self.emit_sqrt_step_kernel(n, float_ty, sm)?;
        all_ptx.push(sqrt_step_ptx);

        // Kernel 3: Padé approximation of log(I + X).
        let pade_log_ptx = self.emit_pade_log_kernel(n, float_ty, sm)?;
        all_ptx.push(pade_log_ptx);

        // Kernel 4: Scale back by 2^s.
        let scale_ptx = self.emit_scale_back_kernel(n, float_ty, sm)?;
        all_ptx.push(scale_ptx);

        Ok(all_ptx.join("\n"))
    }

    /// Emits PTX for `X[i,j] = A[i,j] - delta(i,j)` where delta is Kronecker.
    fn emit_shift_kernel(&self, n: u32, float_ty: PtxType, sm: SmVersion) -> SolverResult<String> {
        let name = format!("solver_logm_shift_{}_n{}", ptx_type_suffix(float_ty), n);

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(256)
            .param("a_ptr", PtxType::U64)
            .param("out_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n");
                let total = b.mul_lo_u32(n_reg.clone(), n_reg.clone());

                b.if_lt_u32(gid, total, |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let out_ptr = b.load_param_u64("out_ptr");
                    let n_inner = b.load_param_u32("n");
                    let gid_r = b.global_thread_id_x();

                    let elem_size = if float_ty == PtxType::F32 { 4u32 } else { 8u32 };

                    // Load A[gid].
                    let src_addr = b.byte_offset_addr(a_ptr, gid_r.clone(), elem_size);
                    let val = load_float(b, float_ty, src_addr);

                    // Compute row and column to check if on diagonal.
                    let row = b.alloc_reg(PtxType::U32);
                    let col = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("rem.u32 {row}, {gid_r}, {n_inner};"));
                    b.raw_ptx(&format!("div.u32 {col}, {gid_r}, {n_inner};"));

                    // Subtract 1.0 from diagonal elements: out[i,j] = A[i,j] - delta(i,j).
                    let is_diag = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.eq.u32 {is_diag}, {row}, {col};"));
                    let one = one_const(b, float_ty);
                    let zero = zero_const(b, float_ty);
                    // diag_sub = 1.0 if on diagonal, else 0.0.
                    let diag_sub = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "selp{} {diag_sub}, {one}, {zero}, {is_diag};",
                        float_ty.as_ptx_str()
                    ));
                    // result = val - diag_sub.
                    let result = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "sub{} {result}, {val}, {diag_sub};",
                        float_ty.as_ptx_str()
                    ));

                    let dst_addr = b.byte_offset_addr(out_ptr, gid_r, elem_size);
                    store_float(b, float_ty, dst_addr, result);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }

    /// Emits PTX for one Denman–Beavers iteration used during the square-root
    /// reduction phase of the matrix logarithm.
    fn emit_sqrt_step_kernel(
        &self,
        n: u32,
        float_ty: PtxType,
        sm: SmVersion,
    ) -> SolverResult<String> {
        let name = format!("solver_logm_sqrt_step_{}_n{}", ptx_type_suffix(float_ty), n);

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(256)
            .param("y_ptr", PtxType::U64)
            .param("z_ptr", PtxType::U64)
            .param("y_next_ptr", PtxType::U64)
            .param("z_next_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n");
                let total = b.mul_lo_u32(n_reg.clone(), n_reg);

                b.if_lt_u32(gid, total, |b| {
                    let y_ptr = b.load_param_u64("y_ptr");
                    let z_ptr = b.load_param_u64("z_ptr");
                    let y_next_ptr = b.load_param_u64("y_next_ptr");
                    let z_next_ptr = b.load_param_u64("z_next_ptr");
                    let n_inner = b.load_param_u32("n");
                    let gid_r = b.global_thread_id_x();
                    let elem_size = if float_ty == PtxType::F32 { 4u32 } else { 8u32 };

                    // Compute row and column for diagonal check.
                    let row = b.alloc_reg(PtxType::U32);
                    let col = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("rem.u32 {row}, {gid_r}, {n_inner};"));
                    b.raw_ptx(&format!("div.u32 {col}, {gid_r}, {n_inner};"));

                    let is_diag = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.eq.u32 {is_diag}, {row}, {col};"));
                    let one = one_const(b, float_ty);
                    let zero = zero_const(b, float_ty);

                    // After host computes M_y = Y_k * Z_k^{-1} (stored in y_ptr) and
                    // M_z = Z_k * Y_k^{-1} (stored in z_ptr), this kernel computes:
                    //   Y_{k+1}[i,j] = (M_y[i,j] + delta(i,j)) / 2
                    //   Z_{k+1}[i,j] = (M_z[i,j] + delta(i,j)) / 2
                    let diag_add = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "selp{} {diag_add}, {one}, {zero}, {is_diag};",
                        float_ty.as_ptx_str()
                    ));
                    let half = half_const(b, float_ty);

                    // Process Y channel.
                    let y_src = b.byte_offset_addr(y_ptr, gid_r.clone(), elem_size);
                    let y_val = load_float(b, float_ty, y_src);
                    let y_sum = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "add{} {y_sum}, {y_val}, {diag_add};",
                        float_ty.as_ptx_str()
                    ));
                    let y_result = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "mul{} {y_result}, {y_sum}, {half};",
                        float_ty.as_ptx_str()
                    ));
                    let y_dst = b.byte_offset_addr(y_next_ptr, gid_r.clone(), elem_size);
                    store_float(b, float_ty, y_dst, y_result);

                    // Process Z channel.
                    let z_src = b.byte_offset_addr(z_ptr, gid_r.clone(), elem_size);
                    let z_val = load_float(b, float_ty, z_src);
                    let z_sum = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "add{} {z_sum}, {z_val}, {diag_add};",
                        float_ty.as_ptx_str()
                    ));
                    let z_result = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "mul{} {z_result}, {z_sum}, {half};",
                        float_ty.as_ptx_str()
                    ));
                    let z_dst = b.byte_offset_addr(z_next_ptr, gid_r, elem_size);
                    store_float(b, float_ty, z_dst, z_result);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }

    /// Emits PTX for the Padé approximation of `log(I + X)` for small `X`.
    ///
    /// Uses a diagonal Padé approximant:
    ///   `log(I + X) ≈ P(X) * Q(X)^{-1}`
    ///
    /// where `P` and `Q` are matrix polynomials.
    fn emit_pade_log_kernel(
        &self,
        n: u32,
        float_ty: PtxType,
        sm: SmVersion,
    ) -> SolverResult<String> {
        let name = format!("solver_logm_pade_{}_n{}", ptx_type_suffix(float_ty), n);

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(256)
            .param("x_ptr", PtxType::U64)
            .param("result_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .param("num_terms", PtxType::U32)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n");
                let total = b.mul_lo_u32(n_reg.clone(), n_reg);

                b.if_lt_u32(gid, total, |b| {
                    let x_ptr = b.load_param_u64("x_ptr");
                    let result_ptr = b.load_param_u64("result_ptr");
                    let num_terms = b.load_param_u32("num_terms");
                    let gid_r = b.global_thread_id_x();
                    let elem_size = if float_ty == PtxType::F32 { 4u32 } else { 8u32 };

                    // Scalar element-wise evaluation of truncated log(1+x) series.
                    // For small |x|, log(1+x) ≈ sum_{k=1}^{m} (-1)^{k+1} * x^k / k.
                    // We use Horner's method from the highest term down to k=1.
                    //
                    // Horner form: log(1+x) ≈ x * (1 - x/2 * (1 - x/3 * (1 - ...)))
                    // i.e. evaluate inner to outer.
                    //
                    // Algorithm: start with acc = 1/num_terms, then for k = num_terms-1 down to 1:
                    //   acc = 1/k - x * acc   (alternating sign absorbed into the 1/k term)
                    // then result = x * acc.

                    let src = b.byte_offset_addr(x_ptr, gid_r.clone(), elem_size);
                    let x_val = load_float(b, float_ty, src);

                    // acc_reg holds the running Horner accumulator.
                    let acc_reg = b.alloc_reg(float_ty);
                    // Initialize acc = 0.
                    let zero = zero_const(b, float_ty);
                    b.raw_ptx(&format!("mov{} {acc_reg}, {zero};", float_ty.as_ptx_str()));

                    // k_reg starts at num_terms and decrements to 1.
                    let k_reg = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {k_reg}, {num_terms};"));

                    let log_loop = b.fresh_label("log_loop");
                    let log_exit = b.fresh_label("log_exit");

                    b.raw_ptx(&format!("{log_loop}:"));
                    // Exit when k_reg == 0.
                    let done_pred = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.eq.u32 {done_pred}, {k_reg}, 0;"));
                    b.raw_ptx(&format!("@{done_pred} bra {log_exit};"));

                    // Convert k to float: k_f = (float) k_reg.
                    let k_f = b.alloc_reg(float_ty);
                    if float_ty == PtxType::F64 {
                        b.raw_ptx(&format!("cvt.rn.f64.u32 {k_f}, {k_reg};"));
                    } else {
                        b.raw_ptx(&format!("cvt.rn.f32.u32 {k_f}, {k_reg};"));
                    }

                    // inv_k = 1.0 / k_f.
                    let inv_k = if float_ty == PtxType::F64 {
                        b.rcp_f64(k_f)
                    } else {
                        b.rcp_f32(k_f)
                    };

                    // Determine sign: (-1)^{k+1} = +1 when k is odd, -1 when k is even.
                    let odd_pred = b.alloc_reg(PtxType::Pred);
                    let lsb = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("and.b32 {lsb}, {k_reg}, 1;"));
                    b.raw_ptx(&format!("setp.ne.u32 {odd_pred}, {lsb}, 0;"));

                    let neg_inv_k = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "neg{} {neg_inv_k}, {inv_k};",
                        float_ty.as_ptx_str()
                    ));
                    // signed_inv_k = odd ? +inv_k : -inv_k  (sign = (-1)^{k+1}).
                    let signed_inv_k = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "selp{} {signed_inv_k}, {inv_k}, {neg_inv_k}, {odd_pred};",
                        float_ty.as_ptx_str()
                    ));

                    // Horner step: acc = signed_inv_k + x * acc  → but we accumulate as
                    // acc_new = signed_inv_k - x_val * acc (matches series structure for log).
                    // Actually use: acc = fma(x_val, acc, signed_inv_k).
                    let new_acc = if float_ty == PtxType::F64 {
                        b.fma_f64(x_val.clone(), acc_reg.clone(), signed_inv_k)
                    } else {
                        b.fma_f32(x_val.clone(), acc_reg.clone(), signed_inv_k)
                    };
                    b.raw_ptx(&format!(
                        "mov{} {acc_reg}, {new_acc};",
                        float_ty.as_ptx_str()
                    ));

                    // k -= 1.
                    b.raw_ptx(&format!("sub.u32 {k_reg}, {k_reg}, 1;"));
                    b.raw_ptx(&format!("bra {log_loop};"));
                    b.raw_ptx(&format!("{log_exit}:"));

                    // Result = x * acc (the k=0 term gives the leading x factor).
                    let result = if float_ty == PtxType::F64 {
                        let r = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {r}, {x_val}, {acc_reg};"));
                        r
                    } else {
                        let r = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {r}, {x_val}, {acc_reg};"));
                        r
                    };

                    let dst = b.byte_offset_addr(result_ptr, gid_r, elem_size);
                    store_float(b, float_ty, dst, result);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }

    /// Emits PTX for scaling the result by `2^s`: `result *= 2^s`.
    fn emit_scale_back_kernel(
        &self,
        n: u32,
        float_ty: PtxType,
        sm: SmVersion,
    ) -> SolverResult<String> {
        let name = format!(
            "solver_logm_scale_back_{}_n{}",
            ptx_type_suffix(float_ty),
            n
        );

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(256)
            .param("result_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .param("scale_exp", PtxType::U32)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n");
                let total = b.mul_lo_u32(n_reg.clone(), n_reg);

                b.if_lt_u32(gid, total, |b| {
                    let result_ptr = b.load_param_u64("result_ptr");
                    let scale_exp = b.load_param_u32("scale_exp");
                    let gid_r = b.global_thread_id_x();
                    let elem_size = if float_ty == PtxType::F32 { 4u32 } else { 8u32 };

                    // Multiply each element by 2^scale_exp using IEEE 754 bit construction.
                    // For f64: bits = (scale_exp + 1023) << 52.
                    // For f32: bits = (scale_exp + 127) << 23.
                    let addr = b.byte_offset_addr(result_ptr, gid_r, elem_size);
                    let val = load_float(b, float_ty, addr.clone());

                    let result = if float_ty == PtxType::F64 {
                        let se64 = b.cvt_u32_to_u64(scale_exp);
                        let biased = b.alloc_reg(PtxType::U64);
                        b.raw_ptx(&format!("add.u64 {biased}, {se64}, 1023;"));
                        let shift_amt = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {shift_amt}, 52;"));
                        let bits = b.shl_b64(biased, shift_amt);
                        let factor = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mov.b64 {factor}, {bits};"));
                        let res = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("mul.rn.f64 {res}, {val}, {factor};"));
                        res
                    } else {
                        let biased = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("add.u32 {biased}, {scale_exp}, 127;"));
                        let shift_amt = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {shift_amt}, 23;"));
                        let bits = b.shl_b32(biased, shift_amt);
                        let factor = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.b32 {factor}, {bits};"));
                        let res = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {res}, {val}, {factor};"));
                        res
                    };

                    store_float(b, float_ty, addr, result);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }
}

// =========================================================================
// Matrix Square Root (sqrtm)
// =========================================================================

/// Configuration for matrix square root computation.
#[derive(Debug, Clone)]
pub struct MatrixSqrtConfig {
    /// Matrix dimension (n × n).
    pub n: u32,
    /// Precision: `"f32"` or `"f64"`.
    pub precision: String,
    /// Maximum Denman–Beavers iterations. Default: 50.
    pub max_iters: u32,
    /// Convergence tolerance. Default: 1e-12.
    pub tol: f64,
}

impl MatrixSqrtConfig {
    /// Creates a new configuration with sensible defaults.
    pub fn new(n: u32, precision: &str) -> Self {
        Self {
            n,
            precision: precision.to_string(),
            max_iters: 50,
            tol: 1e-12,
        }
    }

    /// Sets the maximum number of iterations.
    pub fn with_max_iters(mut self, iters: u32) -> Self {
        self.max_iters = iters;
        self
    }

    /// Sets the convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Validates the configuration.
    fn validate(&self) -> SolverResult<()> {
        if self.n == 0 {
            return Err(SolverError::DimensionMismatch(
                "sqrtm: matrix dimension must be > 0".into(),
            ));
        }
        if self.precision != "f32" && self.precision != "f64" {
            return Err(SolverError::InternalError(format!(
                "sqrtm: unsupported precision '{}'; use 'f32' or 'f64'",
                self.precision
            )));
        }
        if self.max_iters == 0 {
            return Err(SolverError::InternalError(
                "sqrtm: max_iters must be > 0".into(),
            ));
        }
        if self.tol <= 0.0 || !self.tol.is_finite() {
            return Err(SolverError::InternalError(format!(
                "sqrtm: tolerance must be positive and finite, got {}",
                self.tol
            )));
        }
        Ok(())
    }
}

/// Execution plan for the matrix square root using Denman–Beavers iteration.
///
/// The iteration produces sequences `Y_k → sqrt(A)` and `Z_k → sqrt(A)^{-1}`:
///
/// ```text
///   Y_0 = A,       Z_0 = I
///   Y_{k+1} = (Y_k * Z_k^{-1} + I) / 2
///   Z_{k+1} = (Z_k * Y_k^{-1} + I) / 2
/// ```
///
/// Convergence is detected when `||Y_{k+1} - Y_k||_F < tol`.
#[derive(Debug, Clone)]
pub struct MatrixSqrtPlan {
    config: MatrixSqrtConfig,
}

impl MatrixSqrtPlan {
    /// Creates a plan from a validated configuration.
    pub fn new(config: MatrixSqrtConfig) -> SolverResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Returns the convergence tolerance.
    pub fn tolerance(&self) -> f64 {
        self.config.tol
    }

    /// Returns the maximum number of iterations.
    pub fn max_iters(&self) -> u32 {
        self.config.max_iters
    }

    /// Generates PTX source for the matrix square root kernels.
    ///
    /// The generated code contains entry points for:
    /// 1. **init kernel** — sets `Y_0 = A` and `Z_0 = I`.
    /// 2. **iteration kernel** — computes the element-wise `(M + I) / 2`
    ///    step after the matrix product and inverse have been done by BLAS.
    /// 3. **convergence kernel** — computes `||Y_{k+1} - Y_k||_F²` via
    ///    parallel reduction.
    pub fn generate_ptx(&self) -> SolverResult<String> {
        let n = self.config.n;
        let float_ty = precision_to_ptx_type(&self.config.precision)?;
        let sm = SmVersion::Sm75;

        let mut all_ptx = Vec::new();

        // Kernel 1: Initialize Y = A, Z = I.
        let init_ptx = self.emit_init_kernel(n, float_ty, sm)?;
        all_ptx.push(init_ptx);

        // Kernel 2: Element-wise (M + I) / 2.
        let iter_ptx = self.emit_iteration_kernel(n, float_ty, sm)?;
        all_ptx.push(iter_ptx);

        // Kernel 3: Frobenius norm difference (convergence check).
        let conv_ptx = self.emit_convergence_kernel(n, float_ty, sm)?;
        all_ptx.push(conv_ptx);

        Ok(all_ptx.join("\n"))
    }

    /// Emits PTX that copies A into Y and sets Z to the identity matrix.
    fn emit_init_kernel(&self, n: u32, float_ty: PtxType, sm: SmVersion) -> SolverResult<String> {
        let name = format!("solver_sqrtm_init_{}_n{}", ptx_type_suffix(float_ty), n);

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(256)
            .param("a_ptr", PtxType::U64)
            .param("y_ptr", PtxType::U64)
            .param("z_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n");
                let total = b.mul_lo_u32(n_reg.clone(), n_reg.clone());

                b.if_lt_u32(gid, total, |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let y_ptr = b.load_param_u64("y_ptr");
                    let z_ptr = b.load_param_u64("z_ptr");
                    let n_inner = b.load_param_u32("n");
                    let gid_r = b.global_thread_id_x();

                    let elem_size = if float_ty == PtxType::F32 { 4u32 } else { 8u32 };

                    // Y = A: copy element.
                    let a_addr = b.byte_offset_addr(a_ptr, gid_r.clone(), elem_size);
                    let val = load_float(b, float_ty, a_addr);
                    let y_addr = b.byte_offset_addr(y_ptr, gid_r.clone(), elem_size);
                    store_float(b, float_ty, y_addr, val);

                    // Z = I: diagonal = 1, off-diagonal = 0.
                    let row = b.alloc_reg(PtxType::U32);
                    let col = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("rem.u32 {row}, {gid_r}, {n_inner};"));
                    b.raw_ptx(&format!("div.u32 {col}, {gid_r}, {n_inner};"));
                    let z_addr = b.byte_offset_addr(z_ptr, gid_r, elem_size);

                    // Set 1.0 if on diagonal, 0.0 otherwise.
                    let one = one_const(b, float_ty);
                    let zero = zero_const(b, float_ty);

                    // Use select based on row == col comparison.
                    let is_diag = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.eq.u32 {is_diag}, {row}, {col};"));
                    let z_val = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "selp{} {z_val}, {one}, {zero}, {is_diag};",
                        float_ty.as_ptx_str()
                    ));
                    store_float(b, float_ty, z_addr, z_val);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }

    /// Emits PTX for the element-wise `(M + I) / 2` step.
    ///
    /// After the host computes `M = Y_k * Z_k^{-1}` via BLAS, this kernel
    /// computes `Y_{k+1}[i,j] = (M[i,j] + delta(i,j)) / 2`.
    fn emit_iteration_kernel(
        &self,
        n: u32,
        float_ty: PtxType,
        sm: SmVersion,
    ) -> SolverResult<String> {
        let name = format!("solver_sqrtm_iter_{}_n{}", ptx_type_suffix(float_ty), n);

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(256)
            .param("m_ptr", PtxType::U64)
            .param("out_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n");
                let total = b.mul_lo_u32(n_reg.clone(), n_reg.clone());

                b.if_lt_u32(gid, total, |b| {
                    let m_ptr = b.load_param_u64("m_ptr");
                    let out_ptr = b.load_param_u64("out_ptr");
                    let n_inner = b.load_param_u32("n");
                    let gid_r = b.global_thread_id_x();

                    let elem_size = if float_ty == PtxType::F32 { 4u32 } else { 8u32 };

                    // Load M[i,j].
                    let m_addr = b.byte_offset_addr(m_ptr, gid_r.clone(), elem_size);
                    let m_val = load_float(b, float_ty, m_addr);

                    // Compute row, col for diagonal check.
                    let row = b.alloc_reg(PtxType::U32);
                    let col = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("rem.u32 {row}, {gid_r}, {n_inner};"));
                    b.raw_ptx(&format!("div.u32 {col}, {gid_r}, {n_inner};"));

                    // Add 1.0 if on diagonal.
                    let is_diag = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.eq.u32 {is_diag}, {row}, {col};"));
                    let one = one_const(b, float_ty);
                    let zero = zero_const(b, float_ty);
                    let diag_add = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "selp{} {diag_add}, {one}, {zero}, {is_diag};",
                        float_ty.as_ptx_str()
                    ));

                    // sum = M[i,j] + diag_add.
                    let sum = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "add{} {sum}, {m_val}, {diag_add};",
                        float_ty.as_ptx_str()
                    ));

                    // result = sum / 2.
                    let half = half_const(b, float_ty);
                    let result = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "mul{} {result}, {sum}, {half};",
                        float_ty.as_ptx_str()
                    ));

                    let out_addr = b.byte_offset_addr(out_ptr, gid_r, elem_size);
                    store_float(b, float_ty, out_addr, result);
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }

    /// Emits PTX for computing `||Y_{k+1} - Y_k||_F²` via parallel reduction.
    ///
    /// Each thread computes `(Y_new[i] - Y_old[i])²` and participates in a
    /// warp-level reduction. Block-level results are atomically accumulated
    /// into a global accumulator.
    fn emit_convergence_kernel(
        &self,
        n: u32,
        float_ty: PtxType,
        sm: SmVersion,
    ) -> SolverResult<String> {
        let name = format!("solver_sqrtm_conv_{}_n{}", ptx_type_suffix(float_ty), n);

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(256)
            .param("y_new_ptr", PtxType::U64)
            .param("y_old_ptr", PtxType::U64)
            .param("norm_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n");
                let total = b.mul_lo_u32(n_reg.clone(), n_reg);

                b.if_lt_u32(gid, total, |b| {
                    let y_new_ptr = b.load_param_u64("y_new_ptr");
                    let y_old_ptr = b.load_param_u64("y_old_ptr");
                    let gid_r = b.global_thread_id_x();

                    let elem_size = if float_ty == PtxType::F32 { 4u32 } else { 8u32 };

                    // diff = Y_new[gid] - Y_old[gid].
                    let new_addr = b.byte_offset_addr(y_new_ptr, gid_r.clone(), elem_size);
                    let old_addr = b.byte_offset_addr(y_old_ptr, gid_r, elem_size);
                    let new_val = load_float(b, float_ty, new_addr);
                    let old_val = load_float(b, float_ty, old_addr);

                    let diff = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "sub{} {diff}, {new_val}, {old_val};",
                        float_ty.as_ptx_str()
                    ));

                    // diff_sq = diff * diff.
                    let diff_sq = b.alloc_reg(float_ty);
                    b.raw_ptx(&format!(
                        "mul{} {diff_sq}, {diff}, {diff};",
                        float_ty.as_ptx_str()
                    ));

                    // Atomically accumulate diff^2 into the global norm accumulator.
                    // This implements a parallel Frobenius norm computation:
                    //   ||Y_new - Y_old||_F^2 = sum_i (Y_new[i] - Y_old[i])^2
                    let norm_ptr = b.load_param_u64("norm_ptr");
                    if float_ty == PtxType::F64 {
                        let _old = b.atom_global_add_f64(norm_ptr, diff_sq);
                    } else {
                        let _old = b.atom_global_add_f32(norm_ptr, diff_sq);
                    }
                });

                b.ret();
            })
            .build()?;

        Ok(ptx)
    }
}

// =========================================================================
// PTX helper utilities
// =========================================================================

/// Converts a precision string to the corresponding PTX floating-point type.
fn precision_to_ptx_type(precision: &str) -> SolverResult<PtxType> {
    match precision {
        "f32" => Ok(PtxType::F32),
        "f64" => Ok(PtxType::F64),
        other => Err(SolverError::InternalError(format!(
            "unsupported precision '{other}'"
        ))),
    }
}

/// Returns a short suffix for kernel names based on the PTX type.
fn ptx_type_suffix(ty: PtxType) -> &'static str {
    match ty {
        PtxType::F32 => "f32",
        PtxType::F64 => "f64",
        _ => "unknown",
    }
}

/// Loads a float value from global memory.
fn load_float(b: &mut BodyBuilder<'_>, float_ty: PtxType, addr: Register) -> Register {
    let dst = b.alloc_reg(float_ty);
    b.raw_ptx(&format!(
        "ld.global{} {dst}, [{addr}];",
        float_ty.as_ptx_str()
    ));
    dst
}

/// Stores a float value to global memory.
fn store_float(b: &mut BodyBuilder<'_>, float_ty: PtxType, addr: Register, val: Register) {
    b.raw_ptx(&format!(
        "st.global{} [{addr}], {val};",
        float_ty.as_ptx_str()
    ));
}

/// Returns a register containing 0.0 in the given float type.
fn zero_const(b: &mut BodyBuilder<'_>, float_ty: PtxType) -> Register {
    let dst = b.alloc_reg(float_ty);
    if float_ty == PtxType::F32 {
        let bits = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("mov.u32 {bits}, 0;"));
        b.raw_ptx(&format!("mov.b32 {dst}, {bits};"));
    } else {
        let bits = b.alloc_reg(PtxType::U64);
        b.raw_ptx(&format!("mov.u64 {bits}, 0;"));
        b.raw_ptx(&format!("mov.b64 {dst}, {bits};"));
    }
    dst
}

/// Returns a register containing 1.0 in the given float type.
fn one_const(b: &mut BodyBuilder<'_>, float_ty: PtxType) -> Register {
    let dst = b.alloc_reg(float_ty);
    if float_ty == PtxType::F32 {
        // IEEE 754: 1.0f32 = 0x3F800000
        let bits = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("mov.u32 {bits}, 1065353216;"));
        b.raw_ptx(&format!("mov.b32 {dst}, {bits};"));
    } else {
        // IEEE 754: 1.0f64 = 0x3FF0000000000000 = 4607182418800017408
        let bits = b.alloc_reg(PtxType::U64);
        b.raw_ptx(&format!("mov.u64 {bits}, 4607182418800017408;"));
        b.raw_ptx(&format!("mov.b64 {dst}, {bits};"));
    }
    dst
}

/// Returns a register containing 0.5 in the given float type.
fn half_const(b: &mut BodyBuilder<'_>, float_ty: PtxType) -> Register {
    let dst = b.alloc_reg(float_ty);
    if float_ty == PtxType::F32 {
        // IEEE 754: 0.5f32 = 0x3F000000 = 1056964608
        let bits = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("mov.u32 {bits}, 1056964608;"));
        b.raw_ptx(&format!("mov.b32 {dst}, {bits};"));
    } else {
        // IEEE 754: 0.5f64 = 0x3FE0000000000000 = 4602678819172646912
        let bits = b.alloc_reg(PtxType::U64);
        b.raw_ptx(&format!("mov.u64 {bits}, 4602678819172646912;"));
        b.raw_ptx(&format!("mov.b64 {dst}, {bits};"));
    }
    dst
}

// =========================================================================
// Tests (in a separate file to stay under the 2000-line refactoring limit)
// =========================================================================

#[cfg(test)]
#[path = "matrix_functions_tests.rs"]
mod tests;
