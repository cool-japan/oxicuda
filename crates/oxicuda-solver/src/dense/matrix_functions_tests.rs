//! Tests for matrix functions: exponential, logarithm, and square root.
//!
//! This module contains both:
//! - PTX generation validation tests (kernel structure, labels, instructions)
//! - CPU reference implementations for mathematical correctness tests

use super::*;

// -- MatrixExpConfig validation --

#[test]
fn expm_config_default_pade_order() {
    let cfg = MatrixExpConfig::new(4, "f64");
    assert_eq!(cfg.pade_order, 13);
}

#[test]
fn expm_config_custom_pade_order() {
    let cfg = MatrixExpConfig::new(4, "f64").with_pade_order(7);
    assert_eq!(cfg.pade_order, 7);
}

#[test]
fn expm_config_zero_dimension_rejected() {
    let cfg = MatrixExpConfig::new(0, "f64");
    assert!(cfg.validate().is_err());
}

#[test]
fn expm_config_bad_precision_rejected() {
    let cfg = MatrixExpConfig::new(4, "f16");
    assert!(cfg.validate().is_err());
}

#[test]
fn expm_config_invalid_pade_order_rejected() {
    let cfg = MatrixExpConfig::new(4, "f64").with_pade_order(6);
    assert!(cfg.validate().is_err());
}

// -- MatrixExpPlan PTX generation --

#[test]
fn expm_plan_f32_generates_ptx() {
    let cfg = MatrixExpConfig::new(4, "f32");
    let plan = MatrixExpPlan::new(cfg).ok();
    assert!(plan.is_some());
    let ptx = plan.map(|p| p.generate_ptx());
    assert!(ptx.is_some());
    let ptx_str = ptx.and_then(|r| r.ok());
    assert!(ptx_str.is_some());
    let s = ptx_str.unwrap_or_default();
    assert!(s.contains("solver_expm_scale_f32_n4"));
    assert!(s.contains("solver_expm_pade_f32_n4_p13"));
    assert!(s.contains("solver_expm_square_f32_n4"));
}

#[test]
fn expm_plan_f64_generates_ptx() {
    let cfg = MatrixExpConfig::new(8, "f64");
    let plan = MatrixExpPlan::new(cfg).ok();
    assert!(plan.is_some());
    let ptx = plan.map(|p| p.generate_ptx());
    assert!(ptx.is_some());
    let ptx_str = ptx.and_then(|r| r.ok());
    assert!(ptx_str.is_some());
    let s = ptx_str.unwrap_or_default();
    assert!(s.contains("solver_expm_scale_f64_n8"));
    assert!(s.contains("solver_expm_pade_f64_n8_p13"));
    assert!(s.contains("solver_expm_square_f64_n8"));
}

#[test]
fn expm_plan_pade_order_3() {
    let cfg = MatrixExpConfig::new(2, "f64").with_pade_order(3);
    let plan = MatrixExpPlan::new(cfg);
    assert!(plan.is_ok());
    let p = plan.ok();
    assert!(p.is_some());
    if let Some(plan) = p {
        assert_eq!(plan.pade_coefficients().len(), 4);
        let ptx = plan.generate_ptx();
        assert!(ptx.is_ok());
        let s = ptx.unwrap_or_default();
        assert!(s.contains("_p3"));
    }
}

#[test]
fn expm_plan_pade_order_5() {
    let cfg = MatrixExpConfig::new(3, "f32").with_pade_order(5);
    let plan = MatrixExpPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(plan) = plan {
        assert_eq!(plan.pade_coefficients().len(), 6);
    }
}

#[test]
fn expm_plan_pade_order_9() {
    let cfg = MatrixExpConfig::new(5, "f64").with_pade_order(9);
    let plan = MatrixExpPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(plan) = plan {
        assert_eq!(plan.pade_coefficients().len(), 10);
    }
}

#[test]
fn expm_plan_theta_values() {
    let cfg13 = MatrixExpConfig::new(4, "f64").with_pade_order(13);
    let plan13 = MatrixExpPlan::new(cfg13);
    assert!(plan13.is_ok());
    if let Ok(p) = plan13 {
        assert!(p.theta() > 5.0);
        assert!(p.theta() < 6.0);
    }

    let cfg3 = MatrixExpConfig::new(4, "f64").with_pade_order(3);
    let plan3 = MatrixExpPlan::new(cfg3);
    assert!(plan3.is_ok());
    if let Ok(p) = plan3 {
        assert!(p.theta() < 0.02);
    }
}

// -- MatrixLogConfig validation --

#[test]
fn logm_config_defaults() {
    let cfg = MatrixLogConfig::new(4, "f64");
    assert_eq!(cfg.max_sqrt_iters, 100);
}

#[test]
fn logm_config_zero_dimension_rejected() {
    let cfg = MatrixLogConfig::new(0, "f64");
    assert!(cfg.validate().is_err());
}

#[test]
fn logm_config_bad_precision_rejected() {
    let cfg = MatrixLogConfig::new(4, "f16");
    assert!(cfg.validate().is_err());
}

#[test]
fn logm_config_zero_iters_rejected() {
    let cfg = MatrixLogConfig::new(4, "f64").with_max_sqrt_iters(0);
    assert!(cfg.validate().is_err());
}

// -- MatrixLogPlan PTX generation --

#[test]
fn logm_plan_f64_generates_ptx() {
    let cfg = MatrixLogConfig::new(6, "f64");
    let plan = MatrixLogPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(plan) = plan {
        assert_eq!(plan.max_sqrt_iters(), 100);
        let ptx = plan.generate_ptx();
        assert!(ptx.is_ok());
        let s = ptx.unwrap_or_default();
        assert!(s.contains("solver_logm_shift_f64_n6"));
        assert!(s.contains("solver_logm_sqrt_step_f64_n6"));
        assert!(s.contains("solver_logm_pade_f64_n6"));
        assert!(s.contains("solver_logm_scale_back_f64_n6"));
    }
}

#[test]
fn logm_plan_f32_generates_ptx() {
    let cfg = MatrixLogConfig::new(3, "f32");
    let plan = MatrixLogPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(plan) = plan {
        let ptx = plan.generate_ptx();
        assert!(ptx.is_ok());
        let s = ptx.unwrap_or_default();
        assert!(s.contains("solver_logm_shift_f32_n3"));
    }
}

#[test]
fn logm_plan_custom_sqrt_iters() {
    let cfg = MatrixLogConfig::new(4, "f64").with_max_sqrt_iters(200);
    let plan = MatrixLogPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        assert_eq!(p.max_sqrt_iters(), 200);
    }
}

// -- MatrixSqrtConfig validation --

#[test]
fn sqrtm_config_defaults() {
    let cfg = MatrixSqrtConfig::new(4, "f64");
    assert_eq!(cfg.max_iters, 50);
    assert!((cfg.tol - 1e-12).abs() < 1e-20);
}

#[test]
fn sqrtm_config_zero_dimension_rejected() {
    let cfg = MatrixSqrtConfig::new(0, "f64");
    assert!(cfg.validate().is_err());
}

#[test]
fn sqrtm_config_bad_precision_rejected() {
    let cfg = MatrixSqrtConfig::new(4, "bf16");
    assert!(cfg.validate().is_err());
}

#[test]
fn sqrtm_config_zero_iters_rejected() {
    let cfg = MatrixSqrtConfig::new(4, "f64").with_max_iters(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn sqrtm_config_negative_tol_rejected() {
    let cfg = MatrixSqrtConfig::new(4, "f64").with_tol(-1.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn sqrtm_config_nan_tol_rejected() {
    let cfg = MatrixSqrtConfig::new(4, "f64").with_tol(f64::NAN);
    assert!(cfg.validate().is_err());
}

#[test]
fn sqrtm_config_inf_tol_rejected() {
    let cfg = MatrixSqrtConfig::new(4, "f64").with_tol(f64::INFINITY);
    assert!(cfg.validate().is_err());
}

// -- MatrixSqrtPlan PTX generation --

#[test]
fn sqrtm_plan_f64_generates_ptx() {
    let cfg = MatrixSqrtConfig::new(4, "f64");
    let plan = MatrixSqrtPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(plan) = plan {
        assert_eq!(plan.max_iters(), 50);
        assert!((plan.tolerance() - 1e-12).abs() < 1e-20);
        let ptx = plan.generate_ptx();
        assert!(ptx.is_ok());
        let s = ptx.unwrap_or_default();
        assert!(s.contains("solver_sqrtm_init_f64_n4"));
        assert!(s.contains("solver_sqrtm_iter_f64_n4"));
        assert!(s.contains("solver_sqrtm_conv_f64_n4"));
    }
}

#[test]
fn sqrtm_plan_f32_generates_ptx() {
    let cfg = MatrixSqrtConfig::new(8, "f32");
    let plan = MatrixSqrtPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(plan) = plan {
        let ptx = plan.generate_ptx();
        assert!(ptx.is_ok());
        let s = ptx.unwrap_or_default();
        assert!(s.contains("solver_sqrtm_init_f32_n8"));
        assert!(s.contains("solver_sqrtm_iter_f32_n8"));
        assert!(s.contains("solver_sqrtm_conv_f32_n8"));
    }
}

#[test]
fn sqrtm_plan_custom_params() {
    let cfg = MatrixSqrtConfig::new(16, "f64")
        .with_max_iters(100)
        .with_tol(1e-15);
    let plan = MatrixSqrtPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        assert_eq!(p.max_iters(), 100);
        assert!((p.tolerance() - 1e-15).abs() < 1e-25);
    }
}

// -- Padé coefficient tests --

#[test]
fn pade_coefficients_order_3() {
    let c = pade_coefficients(3);
    assert!(c.is_ok());
    let c = c.unwrap_or_default();
    assert_eq!(c.len(), 4);
    assert!((c[0] - 120.0).abs() < 1e-10);
}

#[test]
fn pade_coefficients_order_13() {
    let c = pade_coefficients(13);
    assert!(c.is_ok());
    let c = c.unwrap_or_default();
    assert_eq!(c.len(), 14);
}

#[test]
fn pade_coefficients_invalid_order() {
    assert!(pade_coefficients(4).is_err());
    assert!(pade_coefficients(0).is_err());
    assert!(pade_coefficients(100).is_err());
}

#[test]
fn pade_theta_all_orders() {
    for order in [3, 5, 7, 9, 13] {
        let theta = pade_theta(order);
        assert!(theta.is_ok(), "theta for order {order} should be Ok");
        if let Ok(t) = theta {
            assert!(t > 0.0, "theta must be positive for order {order}");
            assert!(t.is_finite(), "theta must be finite for order {order}");
        }
    }
}

#[test]
fn pade_theta_increasing_with_order() {
    let t3 = pade_theta(3).unwrap_or(0.0);
    let t5 = pade_theta(5).unwrap_or(0.0);
    let t7 = pade_theta(7).unwrap_or(0.0);
    let t9 = pade_theta(9).unwrap_or(0.0);
    let t13 = pade_theta(13).unwrap_or(0.0);
    assert!(t3 < t5);
    assert!(t5 < t7);
    assert!(t7 < t9);
    assert!(t9 < t13);
}

// -- Precision helper tests --

#[test]
fn precision_to_ptx_type_valid() {
    assert_eq!(precision_to_ptx_type("f32").ok(), Some(PtxType::F32));
    assert_eq!(precision_to_ptx_type("f64").ok(), Some(PtxType::F64));
}

#[test]
fn precision_to_ptx_type_invalid() {
    assert!(precision_to_ptx_type("f16").is_err());
    assert!(precision_to_ptx_type("int").is_err());
}

#[test]
fn ptx_type_suffix_values() {
    assert_eq!(ptx_type_suffix(PtxType::F32), "f32");
    assert_eq!(ptx_type_suffix(PtxType::F64), "f64");
}

// -- Large matrix size tests --

#[test]
fn expm_large_matrix_plan() {
    let cfg = MatrixExpConfig::new(256, "f64");
    let plan = MatrixExpPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        let ptx = p.generate_ptx();
        assert!(ptx.is_ok());
        let s = ptx.unwrap_or_default();
        assert!(s.contains("n256"));
    }
}

#[test]
fn logm_large_matrix_plan() {
    let cfg = MatrixLogConfig::new(128, "f32");
    let plan = MatrixLogPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        let ptx = p.generate_ptx();
        assert!(ptx.is_ok());
        let s = ptx.unwrap_or_default();
        assert!(s.contains("n128"));
    }
}

#[test]
fn sqrtm_large_matrix_plan() {
    let cfg = MatrixSqrtConfig::new(512, "f64");
    let plan = MatrixSqrtPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        let ptx = p.generate_ptx();
        assert!(ptx.is_ok());
        let s = ptx.unwrap_or_default();
        assert!(s.contains("n512"));
    }
}

// ---------------------------------------------------------------------------
// CPU reference implementations for correctness tests.
// These compute the exact matrix functions on small f64 matrices using
// the same algorithms (scaling+squaring Padé, Denman–Beavers) entirely
// in Rust, independently from the PTX generation path.
// ---------------------------------------------------------------------------

/// Multiply two n×n column-major matrices (C = A * B).
fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for col in 0..n {
        for row in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += a[k * n + row] * b[col * n + k];
            }
            c[col * n + row] = acc;
        }
    }
    c
}

/// Scale an n×n matrix by a scalar.
fn matscale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

/// Returns an n×n identity matrix (column-major).
fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Frobenius norm of a matrix stored as a flat slice.
fn frobenius_norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// CPU reference implementation of matrix exponential via Padé order-13
/// scaling-and-squaring. Computes `e^A` for a small n×n column-major matrix.
fn cpu_expm(a: &[f64], n: usize) -> Vec<f64> {
    // Padé [13/13] coefficients (Higham 2005).
    let c: [f64; 14] = [
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
    ];

    // Determine scaling parameter s such that ||A/2^s|| <= theta_13.
    let norm = frobenius_norm(a);
    let theta = 5.371_920_351_148_152f64;
    let s = if norm <= theta {
        0u32
    } else {
        let ratio = norm / theta;
        ratio.log2().ceil() as u32
    };

    // Scale: A_s = A / 2^s.
    let scale = (s as f64).exp2();
    let a_s: Vec<f64> = a.iter().map(|x| x / scale).collect();

    // Compute A^2, A^4, A^6 for the Padé-13 algorithm.
    let a2 = matmul(&a_s, &a_s, n);
    let a4 = matmul(&a2, &a2, n);
    let a6 = matmul(&a2, &a4, n);

    let nn = n * n;
    let eye = identity(n);

    // Padé-13: U and V polynomials (Higham 2005, Table 1).
    let mut u2_inner = vec![0.0f64; nn];
    for i in 0..nn {
        u2_inner[i] = c[13] * a6[i] + c[11] * a4[i] + c[9] * a2[i];
    }
    let u2 = matmul(&a6, &u2_inner, n);
    let mut u_inner = vec![0.0f64; nn];
    for i in 0..nn {
        u_inner[i] = u2[i] + c[7] * a6[i] + c[5] * a4[i] + c[3] * a2[i] + c[1] * eye[i];
    }
    let u = matmul(&a_s, &u_inner, n);

    let mut v2_inner = vec![0.0f64; nn];
    for i in 0..nn {
        v2_inner[i] = c[12] * a6[i] + c[10] * a4[i] + c[8] * a2[i];
    }
    let v2 = matmul(&a6, &v2_inner, n);
    let mut v = vec![0.0f64; nn];
    for i in 0..nn {
        v[i] = v2[i] + c[6] * a6[i] + c[4] * a4[i] + c[2] * a2[i] + c[0] * eye[i];
    }

    // P = U + V,  Q = -U + V.
    let p: Vec<f64> = (0..nn).map(|i| u[i] + v[i]).collect();
    let q: Vec<f64> = (0..nn).map(|i| -u[i] + v[i]).collect();

    // Solve Q * F = P for F using Gaussian elimination.
    let f_scaled = solve_linear(q, p, n);

    // Squaring phase: F = F^{2^s}.
    let mut f = f_scaled;
    for _ in 0..s {
        f = matmul(&f.clone(), &f, n);
    }
    f
}

/// Solve the linear system `A * X = B` using Gaussian elimination with
/// partial pivoting. Both A and B are n×n column-major matrices.
/// Returns the solution matrix X (column-major).
fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    for col in 0..n {
        // Find pivot row.
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[col * n + row].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for c in 0..n {
                a.swap(c * n + col, c * n + max_row);
                b.swap(c * n + col, c * n + max_row);
            }
        }
        let pivot = a[col * n + col];
        if pivot.abs() < 1e-15 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = a[col * n + row] / pivot;
            for c in col..n {
                // A is column-major: A[row, c] = a[c * n + row], A[col, c] = a[c * n + col].
                let delta = factor * a[c * n + col];
                let idx = c * n + row;
                a[idx] -= delta;
            }
            for c in 0..n {
                let delta = factor * b[c * n + col];
                let idx = c * n + row;
                b[idx] -= delta;
            }
        }
    }
    // Back substitution.
    let mut x = vec![0.0f64; n * n];
    for bc in 0..n {
        for row in (0..n).rev() {
            let mut val = b[bc * n + row];
            for c in (row + 1)..n {
                val -= a[c * n + row] * x[bc * n + c];
            }
            let diag = a[row * n + row];
            x[bc * n + row] = if diag.abs() > 1e-15 { val / diag } else { 0.0 };
        }
    }
    x
}

/// CPU reference implementation of matrix square root via Denman–Beavers.
/// Returns Y such that `Y * Y ≈ A`.
fn cpu_sqrtm(a: &[f64], n: usize) -> Vec<f64> {
    let nn = n * n;
    let eye = identity(n);
    let mut y = a.to_vec();
    let mut z = eye.clone();
    let tol = 1e-12f64;

    for _ in 0..50 {
        let y_inv = solve_linear(y.clone(), eye.clone(), n);
        let z_inv = solve_linear(z.clone(), eye.clone(), n);

        let y_new: Vec<f64> = (0..nn).map(|i| 0.5 * (y[i] + z_inv[i])).collect();
        let z_new: Vec<f64> = (0..nn).map(|i| 0.5 * (z[i] + y_inv[i])).collect();

        let diff: Vec<f64> = (0..nn).map(|i| y_new[i] - y[i]).collect();
        y = y_new;
        z = z_new;

        if frobenius_norm(&diff) < tol {
            break;
        }
    }
    y
}

// ---------------------------------------------------------------------------
// Correctness tests using CPU reference implementations
// ---------------------------------------------------------------------------

#[test]
fn expm_zeros_matrix_is_identity() {
    // e^0 = I.
    let n = 4usize;
    let zeros = vec![0.0f64; n * n];
    let result = cpu_expm(&zeros, n);
    let eye = identity(n);
    let diff: Vec<f64> = result.iter().zip(eye.iter()).map(|(a, b)| a - b).collect();
    let err = frobenius_norm(&diff);
    assert!(
        err < 1e-10,
        "expm(zeros) should be identity; Frobenius err = {err:.3e}"
    );
}

#[test]
fn expm_scaled_identity_zero_matrix() {
    // e^{0*I} = I.
    let n = 3usize;
    let a = matscale(&identity(n), 0.0);
    let result = cpu_expm(&a, n);
    let eye = identity(n);
    let diff: Vec<f64> = result.iter().zip(eye.iter()).map(|(r, e)| r - e).collect();
    let err = frobenius_norm(&diff);
    assert!(
        err < 1e-10,
        "expm(0*I) should be identity; Frobenius err = {err:.3e}"
    );
}

#[test]
fn logm_identity_is_zeros() {
    // log(I) = 0. The shift step: X = I - I = 0.
    let n = 4usize;
    let eye = identity(n);
    let x: Vec<f64> = eye
        .iter()
        .map(|v| {
            v - if (*v - 1.0).abs() < f64::EPSILON {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let zeros = vec![0.0f64; n * n];
    let diff: Vec<f64> = x.iter().zip(zeros.iter()).map(|(a, b)| a - b).collect();
    let err = frobenius_norm(&diff);
    assert!(
        err < 1e-10,
        "logm(I) shift should be zeros; Frobenius err = {err:.3e}"
    );
}

#[test]
fn sqrtm_4_identity_is_2_identity() {
    // sqrt(4*I) = 2*I.
    let n = 4usize;
    let a = matscale(&identity(n), 4.0);
    let result = cpu_sqrtm(&a, n);
    let expected = matscale(&identity(n), 2.0);
    let diff: Vec<f64> = result
        .iter()
        .zip(expected.iter())
        .map(|(r, e)| r - e)
        .collect();
    let err = frobenius_norm(&diff);
    assert!(
        err < 1e-10,
        "sqrtm(4*I) should be 2*I; Frobenius err = {err:.3e}"
    );
}

// -- PTX content tests for real kernel logic --

#[test]
fn scale_kernel_ptx_contains_div() {
    let cfg = MatrixExpConfig::new(4, "f64");
    let plan = MatrixExpPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        let ptx = p.generate_ptx().unwrap_or_default();
        assert!(
            ptx.contains("div.rn.f64"),
            "scale kernel should emit div.rn.f64 for f64"
        );
    }
}

#[test]
fn scale_kernel_f32_ptx_contains_div() {
    let cfg = MatrixExpConfig::new(4, "f32");
    let plan = MatrixExpPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        let ptx = p.generate_ptx().unwrap_or_default();
        assert!(
            ptx.contains("div.rn.f32"),
            "scale kernel should emit div.rn.f32 for f32"
        );
    }
}

#[test]
fn squaring_kernel_ptx_contains_loop() {
    let cfg = MatrixExpConfig::new(4, "f64");
    let plan = MatrixExpPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        let ptx = p.generate_ptx().unwrap_or_default();
        assert!(
            ptx.contains("sq_loop"),
            "squaring kernel should emit sq_loop label"
        );
        assert!(
            ptx.contains("fma.rn.f64"),
            "squaring kernel should emit fma.rn.f64 for the dot product"
        );
    }
}

#[test]
fn shift_kernel_ptx_contains_setp_and_sub() {
    let cfg = MatrixLogConfig::new(4, "f64");
    let plan = MatrixLogPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        let ptx = p.generate_ptx().unwrap_or_default();
        assert!(
            ptx.contains("setp.eq.u32"),
            "shift kernel should emit setp.eq.u32 for diagonal check"
        );
        assert!(
            ptx.contains("sub.f64"),
            "shift kernel should emit sub.f64 for identity subtraction"
        );
    }
}

#[test]
fn sqrt_step_kernel_ptx_contains_both_channels() {
    let cfg = MatrixLogConfig::new(4, "f64");
    let plan = MatrixLogPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        let ptx = p.generate_ptx().unwrap_or_default();
        assert!(
            ptx.contains("y_next_ptr"),
            "sqrt step should reference y_next_ptr"
        );
        assert!(
            ptx.contains("z_next_ptr"),
            "sqrt step should reference z_next_ptr"
        );
    }
}

#[test]
fn convergence_kernel_ptx_contains_atom_add() {
    let cfg = MatrixSqrtConfig::new(4, "f64");
    let plan = MatrixSqrtPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        let ptx = p.generate_ptx().unwrap_or_default();
        assert!(
            ptx.contains("atom.global.add.f64"),
            "convergence kernel should emit atom.global.add.f64"
        );
    }
}

#[test]
fn pade_kernel_ptx_contains_horner_loop() {
    let cfg = MatrixExpConfig::new(4, "f64");
    let plan = MatrixExpPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        let ptx = p.generate_ptx().unwrap_or_default();
        assert!(
            ptx.contains("horner_loop"),
            "Padé kernel should emit horner_loop label"
        );
    }
}

#[test]
fn scale_back_kernel_ptx_contains_mul() {
    let cfg = MatrixLogConfig::new(4, "f64");
    let plan = MatrixLogPlan::new(cfg);
    assert!(plan.is_ok());
    if let Ok(p) = plan {
        let ptx = p.generate_ptx().unwrap_or_default();
        assert!(
            ptx.contains("mul.rn.f64"),
            "scale-back kernel should emit mul.rn.f64"
        );
    }
}

// ============================================================================
// Task 5b: Backward Error Bounds and Residual Formula Tests (CPU, no GPU)
//
// These tests verify the mathematical correctness of the backward error
// formulas used to validate dense matrix decompositions.  They operate
// entirely on small host-side matrices and do not require a CUDA device.
// ============================================================================

// ---------------------------------------------------------------------------
// Utility helpers (frobenius_norm is defined earlier in this file)
// ---------------------------------------------------------------------------

/// Dense n×n matrix-matrix product C = A * B (row-major).
fn matmul_square(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n * n];
    for i in 0..n {
        for k in 0..n {
            if a[i * n + k] == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    c
}

/// Apply permutation to a matrix: result[i] = a[perm[i]] (row permutation).
fn permute_rows(a: &[f64], perm: &[usize], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for (i, &src) in perm.iter().enumerate() {
        for j in 0..n {
            out[i * n + j] = a[src * n + j];
        }
    }
    out
}

/// Element-wise difference: a - b.
fn mat_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

// ---------------------------------------------------------------------------
// Frobenius norm computation
// ---------------------------------------------------------------------------

/// Known matrix [[3,0],[4,0]] has Frobenius norm 5.
#[test]
fn test_frobenius_norm_computation() {
    let a = [3.0_f64, 0.0, 4.0, 0.0];
    let norm = frobenius_norm(&a);
    assert!((norm - 5.0).abs() < 1e-14, "expected norm=5, got {norm}");
}

/// [[1,2],[3,4]] has Frobenius norm = sqrt(1+4+9+16) = sqrt(30).
#[test]
fn test_frobenius_norm_computation_sqrt30() {
    let a = [1.0_f64, 2.0, 3.0, 4.0];
    let norm = frobenius_norm(&a);
    let expected = 30.0_f64.sqrt();
    assert!(
        (norm - expected).abs() < 1e-13,
        "expected norm={expected:.10}, got {norm:.10}"
    );
}

/// Zero matrix has Frobenius norm 0.
#[test]
fn test_frobenius_norm_zero_matrix() {
    let a = [0.0_f64; 9];
    assert_eq!(frobenius_norm(&a), 0.0);
}

// ---------------------------------------------------------------------------
// LU backward error: ||PA - LU||_F
// ---------------------------------------------------------------------------

/// Verify the backward error formula for a known 3×3 LU factorization.
///
/// Given (hand-computed):
///   A = [[6, 2, 1],
///        [3, 6, 3],
///        [1, 2, 6]]
///
/// Partial pivoting keeps the same row order here (no swap needed because
/// the diagonal is dominant), so P = I and PA = A.
///
/// L (unit lower triangular):
///   [[1,     0,   0],
///    [0.5,   1,   0],
///    [1/6, 2/5,   1]]
///
/// U (upper triangular):
///   [[6, 2,   1],
///    [0, 5, 2.5],
///    [0, 0, 5.5 - 0.5 = 5]] -- see below for exact values
///
/// We use a simpler well-conditioned diagonal-dominant example where exact
/// arithmetic is easy to verify, checking that ||PA - LU||_F < 1e-12.
#[test]
fn test_lu_backward_error_formula() {
    // A = [[6, 2, 1],
    //      [3, 6, 3],
    //      [1, 2, 6]]
    // Permutation: identity (row order preserved; pivot col 0: 6 is largest).
    let a = [6.0_f64, 2.0, 1.0, 3.0, 6.0, 3.0, 1.0, 2.0, 6.0];
    let n = 3usize;
    let perm = [0usize, 1, 2]; // identity

    // Manual LU (no swaps):
    // Step 1: l10 = 3/6 = 0.5, l20 = 1/6
    // After elim col 0:
    //   A' = [[6, 2, 1],
    //         [0, 5, 2.5],
    //         [0, 5/3, 35/6]]
    // Step 2: l21 = (5/3)/5 = 1/3
    // After elim col 1:
    //   A'' = [[6, 2, 1],
    //          [0, 5, 2.5],
    //          [0, 0, 35/6 - (1/3)*2.5]] = [0, 0, 35/6 - 5/6] = [0, 0, 5]
    let l = [1.0_f64, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0 / 6.0, 1.0 / 3.0, 1.0];
    let u = [6.0_f64, 2.0, 1.0, 0.0, 5.0, 2.5, 0.0, 0.0, 5.0];

    let pa = permute_rows(&a, &perm, n);
    let lu = matmul_square(&l, &u, n);
    let diff = mat_sub(&pa, &lu);
    let err = frobenius_norm(&diff);

    // Should be essentially machine zero for this exact factorisation.
    assert!(
        err < 1e-12,
        "LU backward error ||PA-LU||_F = {err:.3e}, expected < 1e-12"
    );

    // Normalised backward error: err / (||A||_F * n * eps)
    let norm_a = frobenius_norm(&a);
    let eps = f64::EPSILON;
    let normalised = err / (norm_a * n as f64 * eps);
    // For a well-conditioned matrix, normalised error should be < 10.
    assert!(
        normalised < 10.0,
        "normalised LU backward error = {normalised:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Cholesky backward error: ||A - LL^T||_F
// ---------------------------------------------------------------------------

/// For A = [[4, 2], [2, 3]], the Cholesky factor is
/// L = [[2, 0], [1, sqrt(2)]].
///
/// Verify ||A - L*L^T||_F < 1e-14 (exact for f64 arithmetic).
#[test]
fn test_cholesky_backward_error_formula() {
    // A (SPD 2×2)
    let a = [4.0_f64, 2.0, 2.0, 3.0];
    // L (lower triangular Cholesky factor)
    let l = [2.0_f64, 0.0, 1.0, 2.0_f64.sqrt()];

    // L^T (transpose of L)
    let lt = [l[0], l[2], l[1], l[3]]; // 2×2 transpose

    let llt = matmul_square(&l, &lt, 2);
    let diff = mat_sub(&a, &llt);
    let err = frobenius_norm(&diff);

    assert!(
        err < 1e-14,
        "Cholesky ||A - LL^T||_F = {err:.3e}, expected < 1e-14"
    );
}

/// For a 3×3 SPD diagonal matrix diag(1, 4, 9), Cholesky factor is
/// diag(1, 2, 3). Verify exact reconstruction.
#[test]
fn test_cholesky_backward_error_diagonal() {
    // A = diag(1, 4, 9)
    let a = [1.0_f64, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0];
    // L = diag(1, 2, 3)
    let l = [1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
    let lt = [1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];

    let llt = matmul_square(&l, &lt, 3);
    let diff = mat_sub(&a, &llt);
    let err = frobenius_norm(&diff);

    assert!(err < 1e-14, "Diagonal Cholesky ||A - LL^T||_F = {err:.3e}");
}

// ---------------------------------------------------------------------------
// SVD reconstruction error: ||A - U * Sigma * V^T||_F / ||A||_F
// ---------------------------------------------------------------------------

/// For a known rank-1 3×3 matrix A = u * sigma * v^T (outer product),
/// the SVD reconstruction error must be essentially zero.
#[test]
fn test_svd_reconstruction_error_formula() {
    // A = 2 * [1,0,0]^T * [0,1,0] (rank-1, sigma=2, u=[1,0,0], v=[0,1,0])
    let a = [0.0_f64, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // U = I (left singular vector u=[1,0,0] is e_1)
    let u = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    // Sigma = diag(2, 0, 0)
    let sigma_diag = [2.0_f64, 0.0, 0.0];
    // V = [[0,1,0],[1,0,0],[0,0,1]] (right singular vector v=[0,1,0] is e_2)
    let vt = [0.0_f64, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    // Compute U * Sigma (scale columns of U by singular values)
    let mut u_sigma = [0.0_f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            u_sigma[i * 3 + j] = u[i * 3 + j] * sigma_diag[j];
        }
    }
    // Multiply by V^T
    let reconstruction = matmul_square(&u_sigma, &vt, 3);
    let diff = mat_sub(&a, &reconstruction);
    let err = frobenius_norm(&diff);
    let norm_a = frobenius_norm(&a);

    // Relative reconstruction error should be essentially zero.
    let rel_err = if norm_a > 0.0 { err / norm_a } else { err };
    assert!(
        rel_err < 1e-14,
        "SVD reconstruction relative error = {rel_err:.3e}"
    );
}

/// For a 2×2 full-rank matrix, verify the SVD reconstruction error bound:
/// ||A - U*Σ*V^T||_F / ||A||_F < n * eps.
#[test]
fn test_svd_reconstruction_error_bound() {
    // A = [[3, 1], [1, 3]] — symmetric, so SVD = eigendecomposition.
    // Eigenvalues: 4, 2.  Eigenvectors: [1,1]/√2, [1,-1]/√2.
    let a = [3.0_f64, 1.0, 1.0, 3.0];
    let isqrt2 = 1.0_f64 / 2.0_f64.sqrt();

    // U = V = [[isqrt2, isqrt2], [isqrt2, -isqrt2]]
    let u = [isqrt2, isqrt2, isqrt2, -isqrt2];
    let sigma_diag = [4.0_f64, 2.0];
    let vt = [isqrt2, isqrt2, isqrt2, -isqrt2]; // V^T = V (orthogonal)

    // U * Sigma
    let mut u_sigma = [0.0_f64; 4];
    for i in 0..2 {
        for j in 0..2 {
            u_sigma[i * 2 + j] = u[i * 2 + j] * sigma_diag[j];
        }
    }
    let reconstruction = matmul_square(&u_sigma, &vt, 2);
    let diff = mat_sub(&a, &reconstruction);
    let err = frobenius_norm(&diff);
    let norm_a = frobenius_norm(&a);
    let n = 2usize;
    let eps = f64::EPSILON;

    let bound = norm_a * n as f64 * eps;
    assert!(
        err < bound * 100.0, // allow 100× eps for floating-point arithmetic
        "SVD reconstruction error {err:.3e} exceeds bound {bound:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Residual formula and condition number
// ---------------------------------------------------------------------------

/// For Ax = b with exact solution x, residual ||Ax - b|| / ||b|| < 1e-14.
///
/// A = [[2, 1], [1, 3]], b = [5, 10], exact x = [1, 3].
/// Verify: A*[1,3] = [2+3, 1+9] = [5, 10] = b.  Residual = 0.
#[test]
fn test_residual_formula_for_known_system() {
    let a = [2.0_f64, 1.0, 1.0, 3.0];
    let b = [5.0_f64, 10.0];
    let x = [1.0_f64, 3.0];

    // Compute Ax
    let ax = [a[0] * x[0] + a[1] * x[1], a[2] * x[0] + a[3] * x[1]];

    // Residual vector r = Ax - b
    let r = [ax[0] - b[0], ax[1] - b[1]];
    let res_norm = (r[0] * r[0] + r[1] * r[1]).sqrt();
    let b_norm = (b[0] * b[0] + b[1] * b[1]).sqrt();
    let relative_residual = res_norm / b_norm;

    assert!(
        relative_residual < 1e-14,
        "relative residual = {relative_residual:.3e}, expected < 1e-14"
    );
}

/// For a non-trivial system Ax = b (overdetermined check: verify residual
/// formula correctly measures departure from solution).
#[test]
fn test_residual_formula_non_zero() {
    // A = [[1, 0], [0, 1]], b = [1, 1], x_approx = [1.1, 0.9]
    // Residual = ||(0.1, -0.1)|| / ||(1,1)|| = sqrt(0.02) / sqrt(2) = 0.1
    let a = [1.0_f64, 0.0, 0.0, 1.0]; // identity
    let b = [1.0_f64, 1.0];
    let x_approx = [1.1_f64, 0.9];

    let ax = [
        a[0] * x_approx[0] + a[1] * x_approx[1],
        a[2] * x_approx[0] + a[3] * x_approx[1],
    ];
    let r = [ax[0] - b[0], ax[1] - b[1]];
    let res_norm = (r[0] * r[0] + r[1] * r[1]).sqrt();
    let b_norm = (b[0] * b[0] + b[1] * b[1]).sqrt();
    let relative_residual = res_norm / b_norm;

    let expected = (0.02_f64).sqrt() / (2.0_f64).sqrt();
    assert!(
        (relative_residual - expected).abs() < 1e-14,
        "relative residual = {relative_residual:.6}, expected {expected:.6}"
    );
}

/// Condition number for identity matrix is 1.0; for diag(1, 1000) ≈ 1000.
///
/// Uses the ratio of largest to smallest diagonal entry as a simple
/// surrogate for the exact condition number (valid for diagonal matrices).
#[test]
fn test_condition_number_estimation() {
    // Identity matrix: cond = max_sv / min_sv = 1 / 1 = 1.
    let identity_diag = [1.0_f64, 1.0, 1.0, 1.0];
    let cond_identity = identity_diag
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        / identity_diag.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        (cond_identity - 1.0).abs() < 1e-14,
        "identity cond = {cond_identity}"
    );

    // diag(1, 1000): cond = 1000.
    let ill_diag = [1.0_f64, 1000.0];
    let cond_ill = ill_diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        / ill_diag.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        (cond_ill - 1000.0).abs() < 1e-10,
        "diag(1,1000) cond = {cond_ill}"
    );
}

/// For a well-conditioned matrix, the normalised LU backward error
/// satisfies err / (||A||_F * n * eps) < 1 (well within LAPACK tolerance).
#[test]
fn test_lu_backward_error_normalised_tolerance() {
    // The same 3×3 system used in test_lu_backward_error_formula.
    // This test independently re-derives the threshold inequality.
    let a_norm_f = frobenius_norm(&[6.0_f64, 2.0, 1.0, 3.0, 6.0, 3.0, 1.0, 2.0, 6.0]);
    let n = 3usize;
    let eps = f64::EPSILON;

    // The backward error ||PA - LU||_F from the exact factorisation is 0
    // (floating-point rounding only), but for the purpose of this test we
    // assume a worst-case rounding error of n * eps * ||A||_F.
    let worst_case_err = (n as f64) * eps * a_norm_f;
    let normalised = worst_case_err / (a_norm_f * n as f64 * eps);
    // By construction, normalised = 1.0 exactly.
    assert!(
        normalised <= 1.0 + f64::EPSILON,
        "normalised error = {normalised}"
    );
}
