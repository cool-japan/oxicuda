//! Explicit ODE solvers: Euler, RK4, RK45.

use crate::error::{SolverError, SolverResult};

use super::types::{OdeConfig, OdeSolution, OdeSystem};
use super::utils::validate_ode_inputs;

// =========================================================================
// Forward Euler
// =========================================================================

/// Forward Euler solver.
pub struct EulerSolver;

impl EulerSolver {
    /// Integrate the ODE system using the forward Euler method.
    pub fn solve(
        system: &dyn OdeSystem,
        y0: &[f64],
        config: &OdeConfig,
    ) -> SolverResult<OdeSolution> {
        let n = system.dim();
        validate_ode_inputs(n, y0, config)?;

        let mut t = config.t_start;
        let dt = config.dt;
        let mut y = y0.to_vec();
        let mut k = vec![0.0; n];

        let mut times = vec![t];
        let mut states = vec![y.clone()];
        let mut num_steps = 0_usize;
        let mut num_rhs = 0_usize;

        while t < config.t_end - dt * 1e-10 && num_steps < config.max_steps {
            let h = dt.min(config.t_end - t);
            system.rhs(t, &y, &mut k)?;
            num_rhs += 1;

            for i in 0..n {
                y[i] += h * k[i];
            }
            t += h;
            num_steps += 1;

            times.push(t);
            states.push(y.clone());
        }

        Ok(OdeSolution {
            times,
            states,
            num_steps,
            num_rejected: 0,
            num_rhs_evals: num_rhs,
        })
    }
}

// =========================================================================
// Classical RK4
// =========================================================================

/// Classical fourth-order Runge-Kutta solver.
pub struct Rk4Solver;

impl Rk4Solver {
    /// Integrate the ODE system using classical RK4.
    pub fn solve(
        system: &dyn OdeSystem,
        y0: &[f64],
        config: &OdeConfig,
    ) -> SolverResult<OdeSolution> {
        let n = system.dim();
        validate_ode_inputs(n, y0, config)?;

        let mut t = config.t_start;
        let dt = config.dt;
        let mut y = y0.to_vec();

        let mut k1 = vec![0.0; n];
        let mut k2 = vec![0.0; n];
        let mut k3 = vec![0.0; n];
        let mut k4 = vec![0.0; n];
        let mut tmp = vec![0.0; n];

        let mut times = vec![t];
        let mut states = vec![y.clone()];
        let mut num_steps = 0_usize;
        let mut num_rhs = 0_usize;

        while t < config.t_end - dt * 1e-10 && num_steps < config.max_steps {
            let h = dt.min(config.t_end - t);

            // k1
            system.rhs(t, &y, &mut k1)?;
            num_rhs += 1;

            // k2
            for i in 0..n {
                tmp[i] = y[i] + 0.5 * h * k1[i];
            }
            system.rhs(t + 0.5 * h, &tmp, &mut k2)?;
            num_rhs += 1;

            // k3
            for i in 0..n {
                tmp[i] = y[i] + 0.5 * h * k2[i];
            }
            system.rhs(t + 0.5 * h, &tmp, &mut k3)?;
            num_rhs += 1;

            // k4
            for i in 0..n {
                tmp[i] = y[i] + h * k3[i];
            }
            system.rhs(t + h, &tmp, &mut k4)?;
            num_rhs += 1;

            // Combine
            for i in 0..n {
                y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
            t += h;
            num_steps += 1;

            times.push(t);
            states.push(y.clone());
        }

        Ok(OdeSolution {
            times,
            states,
            num_steps,
            num_rejected: 0,
            num_rhs_evals: num_rhs,
        })
    }
}

// =========================================================================
// Dormand-Prince RK45 (adaptive)
// =========================================================================

/// Dormand-Prince 4(5) adaptive solver (RK45).
pub struct Rk45Solver;

impl Rk45Solver {
    // Dormand-Prince Butcher tableau (DOPRI5) coefficients.
    const A21: f64 = 1.0 / 5.0;
    const A31: f64 = 3.0 / 40.0;
    const A32: f64 = 9.0 / 40.0;
    const A41: f64 = 44.0 / 45.0;
    const A42: f64 = -56.0 / 15.0;
    const A43: f64 = 32.0 / 9.0;
    const A51: f64 = 19372.0 / 6561.0;
    const A52: f64 = -25360.0 / 2187.0;
    const A53: f64 = 64448.0 / 6561.0;
    const A54: f64 = -212.0 / 729.0;
    const A61: f64 = 9017.0 / 3168.0;
    const A62: f64 = -355.0 / 33.0;
    const A63: f64 = 46732.0 / 5247.0;
    const A64: f64 = 49.0 / 176.0;
    const A65: f64 = -5103.0 / 18656.0;

    // 5th-order weights (for the solution)
    const B1: f64 = 35.0 / 384.0;
    // B2 = 0
    const B3: f64 = 500.0 / 1113.0;
    const B4: f64 = 125.0 / 192.0;
    const B5: f64 = -2187.0 / 6784.0;
    const B6: f64 = 11.0 / 84.0;

    // 4th-order weights (for error estimation)
    const E1: f64 = 71.0 / 57600.0;
    // E2 = 0
    const E3: f64 = -71.0 / 16695.0;
    const E4: f64 = 71.0 / 1920.0;
    const E5: f64 = -17253.0 / 339200.0;
    const E6: f64 = 22.0 / 525.0;
    const E7: f64 = -1.0 / 40.0;

    /// Integrate with adaptive step-size control.
    pub fn solve(
        system: &dyn OdeSystem,
        y0: &[f64],
        config: &OdeConfig,
    ) -> SolverResult<OdeSolution> {
        let n = system.dim();
        validate_ode_inputs(n, y0, config)?;

        let mut t = config.t_start;
        let mut h = config.dt;
        let mut y = y0.to_vec();

        let mut k1 = vec![0.0; n];
        let mut k2 = vec![0.0; n];
        let mut k3 = vec![0.0; n];
        let mut k4 = vec![0.0; n];
        let mut k5 = vec![0.0; n];
        let mut k6 = vec![0.0; n];
        let mut k7 = vec![0.0; n];
        let mut tmp = vec![0.0; n];
        let mut y_new = vec![0.0; n];

        let mut times = vec![t];
        let mut states = vec![y.clone()];
        let mut num_steps = 0_usize;
        let mut num_rejected = 0_usize;
        let mut num_rhs = 0_usize;

        // Safety factor and step size bounds
        let safety = 0.9;
        let min_factor = 0.2;
        let max_factor = 5.0;

        system.rhs(t, &y, &mut k1)?;
        num_rhs += 1;

        while t < config.t_end - 1e-14 * config.t_end.abs().max(1.0)
            && num_steps + num_rejected < config.max_steps
        {
            h = h.min(config.t_end - t);

            // Stage 2
            for i in 0..n {
                tmp[i] = y[i] + h * Self::A21 * k1[i];
            }
            system.rhs(t + h / 5.0, &tmp, &mut k2)?;

            // Stage 3
            for i in 0..n {
                tmp[i] = y[i] + h * (Self::A31 * k1[i] + Self::A32 * k2[i]);
            }
            system.rhs(t + 3.0 / 10.0 * h, &tmp, &mut k3)?;

            // Stage 4
            for i in 0..n {
                tmp[i] = y[i] + h * (Self::A41 * k1[i] + Self::A42 * k2[i] + Self::A43 * k3[i]);
            }
            system.rhs(t + 4.0 / 5.0 * h, &tmp, &mut k4)?;

            // Stage 5
            for i in 0..n {
                tmp[i] = y[i]
                    + h * (Self::A51 * k1[i]
                        + Self::A52 * k2[i]
                        + Self::A53 * k3[i]
                        + Self::A54 * k4[i]);
            }
            system.rhs(t + 8.0 / 9.0 * h, &tmp, &mut k5)?;

            // Stage 6
            for i in 0..n {
                tmp[i] = y[i]
                    + h * (Self::A61 * k1[i]
                        + Self::A62 * k2[i]
                        + Self::A63 * k3[i]
                        + Self::A64 * k4[i]
                        + Self::A65 * k5[i]);
            }
            system.rhs(t + h, &tmp, &mut k6)?;

            num_rhs += 5;

            // 5th-order solution
            for i in 0..n {
                y_new[i] = y[i]
                    + h * (Self::B1 * k1[i]
                        + Self::B3 * k3[i]
                        + Self::B4 * k4[i]
                        + Self::B5 * k5[i]
                        + Self::B6 * k6[i]);
            }

            // Error estimate (difference between 5th and 4th order)
            // We need k7 for the error estimate
            system.rhs(t + h, &y_new, &mut k7)?;
            num_rhs += 1;

            let mut err_norm = 0.0;
            for i in 0..n {
                let err_i = h
                    * (Self::E1 * k1[i]
                        + Self::E3 * k3[i]
                        + Self::E4 * k4[i]
                        + Self::E5 * k5[i]
                        + Self::E6 * k6[i]
                        + Self::E7 * k7[i]);
                let scale = config.atol + config.rtol * y_new[i].abs().max(y[i].abs());
                err_norm += (err_i / scale).powi(2);
            }
            err_norm = (err_norm / n as f64).sqrt();

            if err_norm <= 1.0 {
                // Accept step
                t += h;
                y.copy_from_slice(&y_new);
                num_steps += 1;

                times.push(t);
                states.push(y.clone());

                // FSAL: reuse k7 as k1 for next step
                k1.copy_from_slice(&k7);

                // Increase step size
                let factor = if err_norm > 1e-15 {
                    (safety / err_norm.powf(0.2)).clamp(min_factor, max_factor)
                } else {
                    max_factor
                };
                h *= factor;
            } else {
                // Reject step
                num_rejected += 1;
                let factor = (safety / err_norm.powf(0.2)).clamp(min_factor, 1.0);
                h *= factor;
            }
        }

        if num_steps + num_rejected >= config.max_steps && t < config.t_end - 1e-10 {
            return Err(SolverError::ConvergenceFailure {
                iterations: config.max_steps as u32,
                residual: (config.t_end - t).abs(),
            });
        }

        Ok(OdeSolution {
            times,
            states,
            num_steps,
            num_rejected,
            num_rhs_evals: num_rhs,
        })
    }
}
