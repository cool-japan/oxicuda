//! Implicit ODE solvers: Backward Euler, BDF2.

use crate::error::{SolverError, SolverResult};

use super::types::{OdeConfig, OdeSolution, OdeSystem};
use super::utils::{numerical_jacobian, solve_dense_system, validate_ode_inputs, vec_norm};

// =========================================================================
// Implicit Euler
// =========================================================================

/// Backward (implicit) Euler solver — suitable for stiff systems.
pub struct ImplicitEulerSolver;

impl ImplicitEulerSolver {
    /// Maximum Newton iterations per time step.
    const MAX_NEWTON: usize = 50;
    /// Newton convergence tolerance.
    const NEWTON_TOL: f64 = 1e-12;

    /// Integrate using the backward Euler method with Newton iteration.
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

        let mut times = vec![t];
        let mut states = vec![y.clone()];
        let mut num_steps = 0_usize;
        let mut num_rhs = 0_usize;

        while t < config.t_end - dt * 1e-10 && num_steps < config.max_steps {
            let h = dt.min(config.t_end - t);
            let t_new = t + h;

            // Newton iteration to solve: y_new - y - h*f(t_new, y_new) = 0
            // Initial guess: forward Euler
            let mut y_new = y.clone();
            let mut f_val = vec![0.0; n];
            system.rhs(t, &y, &mut f_val)?;
            num_rhs += 1;
            for i in 0..n {
                y_new[i] = y[i] + h * f_val[i];
            }

            let mut converged = false;
            for _ in 0..Self::MAX_NEWTON {
                // Evaluate residual: G(y_new) = y_new - y - h*f(t_new, y_new)
                system.rhs(t_new, &y_new, &mut f_val)?;
                num_rhs += 1;

                let mut residual = vec![0.0; n];
                for i in 0..n {
                    residual[i] = y_new[i] - y[i] - h * f_val[i];
                }

                let res_norm = vec_norm(&residual);
                if res_norm < Self::NEWTON_TOL {
                    converged = true;
                    break;
                }

                // Numerical Jacobian of G: J_G = I - h * J_f
                let jac_f = numerical_jacobian(system, t_new, &y_new, 1e-8)?;
                num_rhs += n; // Jacobian evaluates rhs n times

                // Build J_G = I - h*J_f
                let mut jac_g = vec![vec![0.0; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        jac_g[i][j] = -h * jac_f[i][j];
                        if i == j {
                            jac_g[i][j] += 1.0;
                        }
                    }
                }

                // Solve J_G * delta = -residual using Gaussian elimination
                let delta =
                    solve_dense_system(&jac_g, &residual.iter().map(|r| -r).collect::<Vec<_>>())?;

                for i in 0..n {
                    y_new[i] += delta[i];
                }
            }

            if !converged {
                return Err(SolverError::ConvergenceFailure {
                    iterations: Self::MAX_NEWTON as u32,
                    residual: {
                        system.rhs(t_new, &y_new, &mut f_val)?;
                        let mut res = vec![0.0; n];
                        for i in 0..n {
                            res[i] = y_new[i] - y[i] - h * f_val[i];
                        }
                        vec_norm(&res)
                    },
                });
            }

            y.copy_from_slice(&y_new);
            t = t_new;
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
// BDF2
// =========================================================================

/// Second-order backward differentiation formula (BDF2) solver.
pub struct Bdf2Solver;

impl Bdf2Solver {
    /// Maximum Newton iterations per step.
    const MAX_NEWTON: usize = 50;
    /// Newton convergence tolerance.
    const NEWTON_TOL: f64 = 1e-12;

    /// Integrate using BDF2. The first step is bootstrapped with implicit Euler.
    ///
    /// BDF2 formula: y_{n+1} = (4/3)*y_n - (1/3)*y_{n-1} + (2/3)*h*f(t_{n+1}, y_{n+1})
    pub fn solve(
        system: &dyn OdeSystem,
        y0: &[f64],
        config: &OdeConfig,
    ) -> SolverResult<OdeSolution> {
        let n = system.dim();
        validate_ode_inputs(n, y0, config)?;

        let dt = config.dt;
        let mut t = config.t_start;
        let mut y_prev = y0.to_vec();

        let mut times = vec![t];
        let mut states = vec![y_prev.clone()];
        let mut num_steps = 0_usize;
        let mut num_rhs = 0_usize;

        // Bootstrap: take one implicit Euler step to get y_1
        let h = dt.min(config.t_end - t);
        if t >= config.t_end - dt * 1e-10 {
            return Ok(OdeSolution {
                times,
                states,
                num_steps: 0,
                num_rejected: 0,
                num_rhs_evals: 0,
            });
        }

        let mut y_cur = y_prev.clone();
        let mut f_val = vec![0.0; n];
        {
            // Forward Euler initial guess
            system.rhs(t, &y_prev, &mut f_val)?;
            num_rhs += 1;
            for i in 0..n {
                y_cur[i] = y_prev[i] + h * f_val[i];
            }

            let t_new = t + h;
            let mut converged = false;
            for _ in 0..Self::MAX_NEWTON {
                system.rhs(t_new, &y_cur, &mut f_val)?;
                num_rhs += 1;

                let mut residual = vec![0.0; n];
                for i in 0..n {
                    residual[i] = y_cur[i] - y_prev[i] - h * f_val[i];
                }

                if vec_norm(&residual) < Self::NEWTON_TOL {
                    converged = true;
                    break;
                }

                let jac_f = numerical_jacobian(system, t_new, &y_cur, 1e-8)?;
                num_rhs += n;

                let mut jac_g = vec![vec![0.0; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        jac_g[i][j] = -h * jac_f[i][j];
                        if i == j {
                            jac_g[i][j] += 1.0;
                        }
                    }
                }

                let neg_res: Vec<f64> = residual.iter().map(|r| -r).collect();
                let delta = solve_dense_system(&jac_g, &neg_res)?;
                for i in 0..n {
                    y_cur[i] += delta[i];
                }
            }

            if !converged {
                return Err(SolverError::ConvergenceFailure {
                    iterations: Self::MAX_NEWTON as u32,
                    residual: 1.0,
                });
            }

            t = t_new;
            num_steps += 1;
            times.push(t);
            states.push(y_cur.clone());
        }

        // BDF2 steps
        while t < config.t_end - dt * 1e-10 && num_steps < config.max_steps {
            let h_step = dt.min(config.t_end - t);
            let t_new = t + h_step;

            // BDF2: y_{n+1} = (4/3)*y_n - (1/3)*y_{n-1} + (2/3)*h*f(t_{n+1}, y_{n+1})
            // Residual: y_{n+1} - (4/3)*y_n + (1/3)*y_{n-1} - (2/3)*h*f(t_{n+1}, y_{n+1}) = 0
            let coeff_h = 2.0 / 3.0 * h_step;

            // Initial guess: extrapolation
            let mut y_new = vec![0.0; n];
            for i in 0..n {
                y_new[i] = 2.0 * y_cur[i] - y_prev[i];
            }

            let mut converged = false;
            for _ in 0..Self::MAX_NEWTON {
                system.rhs(t_new, &y_new, &mut f_val)?;
                num_rhs += 1;

                let mut residual = vec![0.0; n];
                for i in 0..n {
                    residual[i] = y_new[i] - (4.0 / 3.0) * y_cur[i] + (1.0 / 3.0) * y_prev[i]
                        - coeff_h * f_val[i];
                }

                if vec_norm(&residual) < Self::NEWTON_TOL {
                    converged = true;
                    break;
                }

                let jac_f = numerical_jacobian(system, t_new, &y_new, 1e-8)?;
                num_rhs += n;

                let mut jac_g = vec![vec![0.0; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        jac_g[i][j] = -coeff_h * jac_f[i][j];
                        if i == j {
                            jac_g[i][j] += 1.0;
                        }
                    }
                }

                let neg_res: Vec<f64> = residual.iter().map(|r| -r).collect();
                let delta = solve_dense_system(&jac_g, &neg_res)?;
                for i in 0..n {
                    y_new[i] += delta[i];
                }
            }

            if !converged {
                return Err(SolverError::ConvergenceFailure {
                    iterations: Self::MAX_NEWTON as u32,
                    residual: 1.0,
                });
            }

            y_prev.copy_from_slice(&y_cur);
            y_cur.copy_from_slice(&y_new);
            t = t_new;
            num_steps += 1;
            times.push(t);
            states.push(y_cur.clone());
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
