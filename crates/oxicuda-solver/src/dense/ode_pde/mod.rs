//! ODE and PDE solver kernels.
//!
//! Provides CPU-side implementations of numerical methods for ordinary and
//! partial differential equations. In production these algorithms would generate
//! PTX kernels for GPU execution; the CPU reference implementations here ensure
//! correctness and serve as the algorithmic specification.
//!
//! ## ODE solvers
//!
//! - **Euler** — first-order forward Euler
//! - **RK4** — classical fourth-order Runge-Kutta
//! - **RK45** — Dormand-Prince 4(5) with adaptive step-size control
//! - **Implicit Euler** — backward Euler for stiff systems (Newton iteration)
//! - **BDF2** — second-order backward differentiation formula
//!
//! ## PDE solvers
//!
//! - **Heat equation (1-D)** — explicit FTCS and implicit Crank-Nicolson
//! - **Wave equation (1-D)** — leapfrog / Störmer-Verlet
//! - **Poisson equation (1-D)** — direct tridiagonal solve
//! - **Advection equation (1-D)** — first-order upwind and Lax-Wendroff

mod explicit;
mod implicit;
mod pde;
mod types;
mod utils;

// Re-export public API
pub use explicit::{EulerSolver, Rk4Solver, Rk45Solver};
pub use implicit::{Bdf2Solver, ImplicitEulerSolver};
pub use pde::{
    AdvectionEquation1D, BoundaryCondition, Grid1D, Grid2D, HeatEquation1D, PdeConfig, Poisson1D,
    WaveEquation1D,
};
pub use types::{OdeConfig, OdeMethod, OdeSolution, OdeSystem, StepResult};
pub use utils::{numerical_jacobian, solve_tridiagonal};

#[cfg(test)]
mod tests {
    use super::utils::apply_bc_1d;
    use super::*;
    use std::f64::consts::PI;

    // --- Test ODE systems ---

    /// Exponential decay: y' = -y, solution y(t) = y0 * exp(-t).
    struct ExponentialDecay;

    impl OdeSystem for ExponentialDecay {
        fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) -> crate::error::SolverResult<()> {
            dydt[0] = -y[0];
            Ok(())
        }
        fn dim(&self) -> usize {
            1
        }
    }

    /// Harmonic oscillator: y'' + y = 0, as system y' = v, v' = -y.
    /// Solution: y(t) = cos(t), v(t) = -sin(t) with y(0)=1, v(0)=0.
    struct HarmonicOscillator;

    impl OdeSystem for HarmonicOscillator {
        fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) -> crate::error::SolverResult<()> {
            dydt[0] = y[1]; // dy/dt = v
            dydt[1] = -y[0]; // dv/dt = -y
            Ok(())
        }
        fn dim(&self) -> usize {
            2
        }
    }

    /// Stiff system: y' = -1000*(y - sin(t)) + cos(t).
    struct StiffSystem;

    impl OdeSystem for StiffSystem {
        fn rhs(&self, t: f64, y: &[f64], dydt: &mut [f64]) -> crate::error::SolverResult<()> {
            dydt[0] = -1000.0 * (y[0] - t.sin()) + t.cos();
            Ok(())
        }
        fn dim(&self) -> usize {
            1
        }
    }

    /// Van der Pol oscillator: y'' - mu*(1-y²)*y' + y = 0.
    struct VanDerPol {
        mu: f64,
    }

    impl OdeSystem for VanDerPol {
        fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) -> crate::error::SolverResult<()> {
            dydt[0] = y[1];
            dydt[1] = self.mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
            Ok(())
        }
        fn dim(&self) -> usize {
            2
        }
    }

    // --- ODE tests ---

    #[test]
    fn euler_exponential_decay() {
        let sys = ExponentialDecay;
        let config = OdeConfig {
            t_start: 0.0,
            t_end: 1.0,
            dt: 0.001,
            method: OdeMethod::Euler,
            ..OdeConfig::default()
        };
        let sol = EulerSolver::solve(&sys, &[1.0], &config);
        assert!(sol.is_ok());
        let sol = sol.ok().filter(|s| !s.states.is_empty());
        assert!(sol.is_some());
        let sol = sol.as_ref().and_then(|s| s.states.last());
        assert!(sol.is_some());
        let y_final = sol.map(|s| s[0]).unwrap_or(0.0);
        let expected = (-1.0_f64).exp();
        // Euler is only first-order, so ~0.001 accuracy with dt=0.001
        assert!(
            (y_final - expected).abs() < 0.01,
            "Euler: y(1) = {y_final}, expected {expected}"
        );
    }

    #[test]
    fn rk4_harmonic_oscillator() {
        let sys = HarmonicOscillator;
        let config = OdeConfig {
            t_start: 0.0,
            t_end: 2.0 * PI,
            dt: 0.01,
            method: OdeMethod::Rk4,
            ..OdeConfig::default()
        };
        let sol = Rk4Solver::solve(&sys, &[1.0, 0.0], &config);
        assert!(sol.is_ok());
        let sol = sol.ok().filter(|s| !s.states.is_empty());
        assert!(sol.is_some());
        let last = sol.as_ref().and_then(|s| s.states.last());
        assert!(last.is_some());
        let y = last.map(|s| s[0]).unwrap_or(0.0);
        let v = last.map(|s| s[1]).unwrap_or(0.0);
        // After one full period, should return to (1, 0)
        assert!((y - 1.0).abs() < 1e-6, "RK4: y(2pi) = {y}, expected 1.0");
        assert!(v.abs() < 1e-6, "RK4: v(2pi) = {v}, expected 0.0");
    }

    #[test]
    fn rk45_adaptive_step() {
        let sys = HarmonicOscillator;
        let config = OdeConfig {
            t_start: 0.0,
            t_end: 2.0 * PI,
            dt: 0.1,
            rtol: 1e-8,
            atol: 1e-10,
            max_steps: 10_000,
            method: OdeMethod::Rk45,
        };
        let sol = Rk45Solver::solve(&sys, &[1.0, 0.0], &config);
        assert!(sol.is_ok());
        let sol_data = sol.ok().filter(|s| !s.states.is_empty());
        assert!(sol_data.is_some());
        let sd = sol_data.as_ref();
        let last = sd.and_then(|s| s.states.last());
        let y = last.map(|s| s[0]).unwrap_or(0.0);
        let v = last.map(|s| s[1]).unwrap_or(0.0);
        assert!((y - 1.0).abs() < 1e-6, "RK45: y(2pi) = {y}, expected 1.0");
        assert!(v.abs() < 1e-6, "RK45: v(2pi) = {v}, expected 0.0");
        // Adaptive should take fewer steps than fixed RK4 with same accuracy
        let num = sd.map(|s| s.num_steps).unwrap_or(0);
        assert!(
            num < 1000,
            "RK45 should take fewer than 1000 steps, took {num}"
        );
    }

    #[test]
    fn implicit_euler_stiff() {
        let sys = StiffSystem;
        let config = OdeConfig {
            t_start: 0.0,
            t_end: 0.1,
            dt: 0.01,
            method: OdeMethod::ImplicitEuler,
            ..OdeConfig::default()
        };
        // Start near the exact solution sin(0) = 0
        let sol = ImplicitEulerSolver::solve(&sys, &[0.0], &config);
        assert!(sol.is_ok());
        let sol_data = sol.ok().filter(|s| !s.states.is_empty());
        assert!(sol_data.is_some());
        let last = sol_data.as_ref().and_then(|s| s.states.last());
        let y = last.map(|s| s[0]).unwrap_or(f64::NAN);
        let expected = 0.1_f64.sin();
        // Implicit Euler is first-order, but it should handle the stiffness
        assert!(
            (y - expected).abs() < 0.05,
            "ImplicitEuler: y(0.1) = {y}, expected {expected}"
        );
    }

    #[test]
    fn bdf2_van_der_pol() {
        let sys = VanDerPol { mu: 1.0 };
        let config = OdeConfig {
            t_start: 0.0,
            t_end: 1.0,
            dt: 0.01,
            method: OdeMethod::Bdf2,
            ..OdeConfig::default()
        };
        let sol = Bdf2Solver::solve(&sys, &[2.0, 0.0], &config);
        assert!(sol.is_ok());
        let sol_data = sol.ok().filter(|s| !s.states.is_empty());
        assert!(sol_data.is_some());
        // Just verify it completed and states are finite
        let sd = sol_data.as_ref();
        let all_finite = sd
            .map(|s| s.states.iter().all(|st| st.iter().all(|v| v.is_finite())))
            .unwrap_or(false);
        assert!(all_finite, "BDF2: all states should be finite");
    }

    #[test]
    fn ode_convergence_order() {
        // Test that RK4 achieves ~4th order convergence
        let sys = ExponentialDecay;
        let t_end = 1.0;
        let exact = (-1.0_f64).exp();

        let mut errors = Vec::new();
        for &dt in &[0.1, 0.05, 0.025] {
            let config = OdeConfig {
                t_start: 0.0,
                t_end,
                dt,
                method: OdeMethod::Rk4,
                ..OdeConfig::default()
            };
            let sol = Rk4Solver::solve(&sys, &[1.0], &config);
            let sol_data = sol.ok().filter(|s| !s.states.is_empty());
            let y = sol_data
                .as_ref()
                .and_then(|s| s.states.last())
                .map(|s| s[0])
                .unwrap_or(0.0);
            errors.push((y - exact).abs());
        }

        // When dt halves, error should decrease by ~16x for 4th order
        if errors[0] > 1e-15 && errors[1] > 1e-15 {
            let ratio = errors[0] / errors[1];
            assert!(
                ratio > 10.0,
                "RK4 convergence ratio should be ~16, got {ratio}"
            );
        }
    }

    // --- PDE tests ---

    #[test]
    fn heat_explicit_gaussian() {
        // Gaussian diffusion: u(x,0) = exp(-x²), alpha = 0.01
        let alpha = 0.01;
        let nx = 101;
        let grid = Grid1D::new(-5.0, 5.0, nx);
        let heat = HeatEquation1D { alpha };

        let dt = 0.4 * heat.stability_limit(grid.dx); // below stability limit
        let config = PdeConfig {
            grid: grid.clone(),
            dt,
            num_steps: 10,
            bc_left: BoundaryCondition::Dirichlet(0.0),
            bc_right: BoundaryCondition::Dirichlet(0.0),
        };

        let u0: Vec<f64> = (0..nx)
            .map(|i| {
                let x = grid.point(i);
                (-x * x).exp()
            })
            .collect();

        let result = heat.solve_explicit(&u0, &config);
        assert!(result.is_ok());
        let data = result.ok().filter(|d| !d.is_empty());
        assert!(data.is_some());
        let last = data.as_ref().and_then(|d| d.last());
        assert!(last.is_some());

        // Solution should still be peaked at center but broader
        let u_final = last.as_ref().map(|u| u.as_slice()).unwrap_or(&[]);
        if u_final.len() == nx {
            let mid = nx / 2;
            // Peak should be lower than initial (diffusion spreads it)
            assert!(u_final[mid] < u0[mid], "Heat diffusion should reduce peak");
            // Solution should remain positive
            assert!(
                u_final.iter().all(|&v| v >= -1e-10),
                "Heat solution should remain non-negative"
            );
        }
    }

    #[test]
    fn heat_crank_nicolson_stability() {
        // Crank-Nicolson should be stable even with large dt
        let alpha = 1.0;
        let nx = 51;
        let grid = Grid1D::new(0.0, 1.0, nx);
        let heat = HeatEquation1D { alpha };

        // Use dt >> stability limit for explicit
        let dt = 10.0 * heat.stability_limit(grid.dx);
        let config = PdeConfig {
            grid: grid.clone(),
            dt,
            num_steps: 20,
            bc_left: BoundaryCondition::Dirichlet(0.0),
            bc_right: BoundaryCondition::Dirichlet(0.0),
        };

        let u0: Vec<f64> = (0..nx)
            .map(|i| {
                let x = grid.point(i);
                (PI * x).sin()
            })
            .collect();

        let result = heat.solve_implicit(&u0, &config);
        assert!(result.is_ok());
        let data = result.ok().filter(|d| !d.is_empty());
        assert!(data.is_some());
        let last = data.as_ref().and_then(|d| d.last());

        // Solution should be bounded (no blow-up)
        let u_final = last.as_ref().map(|u| u.as_slice()).unwrap_or(&[]);
        let max_val = u_final.iter().copied().fold(0.0_f64, f64::max);
        assert!(
            max_val < 2.0,
            "Crank-Nicolson should be stable; max = {max_val}"
        );
    }

    #[test]
    fn wave_energy_conservation() {
        let c = 1.0;
        let nx = 101;
        let grid = Grid1D::new(0.0, 1.0, nx);
        let dx = grid.dx;
        let wave = WaveEquation1D { c };

        let dt = 0.5 * dx / c; // CFL < 1
        let config = PdeConfig {
            grid: grid.clone(),
            dt,
            num_steps: 50,
            bc_left: BoundaryCondition::Dirichlet(0.0),
            bc_right: BoundaryCondition::Dirichlet(0.0),
        };

        // Initial displacement: sin(pi*x), zero velocity
        let u0: Vec<f64> = (0..nx)
            .map(|i| {
                let x = grid.point(i);
                (PI * x).sin()
            })
            .collect();
        let v0 = vec![0.0; nx];

        let result = wave.solve(&u0, &v0, &config);
        assert!(result.is_ok());
        let data = result.ok().filter(|d| d.len() > 2);
        assert!(data.is_some());

        // Compute energy at first and last time step
        let states = data.as_deref().unwrap_or(&[]);
        if states.len() >= 3 {
            let energy = |u: &[f64], u_prev: &[f64]| -> f64 {
                let mut ke = 0.0;
                let mut pe = 0.0;
                for i in 1..nx - 1 {
                    let v = (u[i] - u_prev[i]) / dt;
                    ke += 0.5 * v * v * dx;
                    let ux = (u[i + 1] - u[i - 1]) / (2.0 * dx);
                    pe += 0.5 * c * c * ux * ux * dx;
                }
                ke + pe
            };

            let e_initial = energy(&states[1], &states[0]);
            let e_final = energy(&states[states.len() - 1], &states[states.len() - 2]);

            // Energy should be approximately conserved
            if e_initial > 1e-10 {
                let rel_change = (e_final - e_initial).abs() / e_initial;
                assert!(
                    rel_change < 0.1,
                    "Wave energy changed by {:.2}%",
                    rel_change * 100.0
                );
            }
        }
    }

    #[test]
    fn poisson_analytical() {
        // Solve -u'' = 2 on [0,1] with u(0) = 0, u(1) = 0
        // Exact: u(x) = x*(1-x)
        let nx = 101;
        let grid = Grid1D::new(0.0, 1.0, nx);
        let poisson = Poisson1D;

        let f_rhs: Vec<f64> = vec![2.0; nx];
        let config = PdeConfig {
            grid: grid.clone(),
            dt: 0.0, // not used
            num_steps: 0,
            bc_left: BoundaryCondition::Dirichlet(0.0),
            bc_right: BoundaryCondition::Dirichlet(0.0),
        };

        let result = poisson.solve(&f_rhs, &config);
        assert!(result.is_ok());
        let u = result.ok().unwrap_or_default();

        let mut max_err = 0.0_f64;
        for (i, u_val) in u.iter().enumerate().take(nx) {
            let x = grid.point(i);
            let exact = x * (1.0 - x);
            max_err = max_err.max((u_val - exact).abs());
        }

        // Second-order scheme: error ~ O(dx²) ~ O(1e-4)
        assert!(
            max_err < 1e-3,
            "Poisson max error = {max_err}, expected < 1e-3"
        );
    }

    #[test]
    fn advection_upwind() {
        let a = 1.0;
        let nx = 101;
        let grid = Grid1D::new(0.0, 2.0, nx);
        let adv = AdvectionEquation1D { a };

        let dt = 0.5 * grid.dx / a.abs(); // CFL = 0.5
        let config = PdeConfig {
            grid: grid.clone(),
            dt,
            num_steps: 10,
            bc_left: BoundaryCondition::Dirichlet(0.0),
            bc_right: BoundaryCondition::Dirichlet(0.0),
        };

        // Step function initial condition
        let u0: Vec<f64> = (0..nx)
            .map(|i| {
                let x = grid.point(i);
                if (0.5..=1.0).contains(&x) { 1.0 } else { 0.0 }
            })
            .collect();

        let result = adv.solve_upwind(&u0, &config);
        assert!(result.is_ok());
        let data = result.ok().filter(|d| !d.is_empty());
        assert!(data.is_some());

        // After advection, the pulse should have moved to the right
        let last = data
            .as_ref()
            .and_then(|d| d.last())
            .cloned()
            .unwrap_or_default();
        // Solution should remain bounded
        let max_val = last.iter().copied().fold(0.0_f64, f64::max);
        assert!(
            max_val <= 1.0 + 1e-10,
            "Upwind should be monotone, max = {max_val}"
        );
    }

    #[test]
    fn lax_wendroff_accuracy() {
        let a = 1.0;
        let nx = 201;
        let grid = Grid1D::new(0.0, 4.0, nx);
        let adv = AdvectionEquation1D { a };

        let dt = 0.5 * grid.dx / a.abs();
        let num_steps = 20;
        let config = PdeConfig {
            grid: grid.clone(),
            dt,
            num_steps,
            bc_left: BoundaryCondition::Dirichlet(0.0),
            bc_right: BoundaryCondition::Dirichlet(0.0),
        };

        // Smooth Gaussian initial condition
        let u0: Vec<f64> = (0..nx)
            .map(|i| {
                let x = grid.point(i);
                (-(x - 1.0).powi(2) / 0.1).exp()
            })
            .collect();

        let result_lw = adv.solve_lax_wendroff(&u0, &config);
        let result_up = adv.solve_upwind(&u0, &config);
        assert!(result_lw.is_ok());
        assert!(result_up.is_ok());

        let lw_last = result_lw
            .ok()
            .and_then(|d| d.last().cloned())
            .unwrap_or_default();
        let up_last = result_up
            .ok()
            .and_then(|d| d.last().cloned())
            .unwrap_or_default();

        // Lax-Wendroff should preserve the pulse shape better (less diffusion)
        let lw_max = lw_last.iter().copied().fold(0.0_f64, f64::max);
        let up_max = up_last.iter().copied().fold(0.0_f64, f64::max);

        assert!(
            lw_max >= up_max - 1e-10,
            "Lax-Wendroff should have less diffusion: LW max = {lw_max}, upwind max = {up_max}"
        );
    }

    #[test]
    fn boundary_condition_enforcement() {
        let nx = 11;
        let mut u = vec![1.0_f64; nx];

        apply_bc_1d(
            &mut u,
            &BoundaryCondition::Dirichlet(0.0),
            &BoundaryCondition::Dirichlet(2.0),
            nx,
        );
        assert!((u[0] - 0.0_f64).abs() < 1e-15);
        assert!((u[nx - 1] - 2.0_f64).abs() < 1e-15);

        // Periodic
        let mut u2 = vec![0.0_f64; nx];
        u2[1] = 3.0;
        u2[nx - 2] = 5.0;
        apply_bc_1d(
            &mut u2,
            &BoundaryCondition::Periodic,
            &BoundaryCondition::Periodic,
            nx,
        );
        assert!((u2[0] - 5.0_f64).abs() < 1e-15);
        assert!((u2[nx - 1] - 3.0_f64).abs() < 1e-15);
    }

    #[test]
    fn stability_limit_calculation() {
        let heat = HeatEquation1D { alpha: 0.5 };
        let dx = 0.1;
        let limit = heat.stability_limit(dx);
        // dt <= dx² / (2*alpha) = 0.01 / 1.0 = 0.01
        assert!((limit - 0.01).abs() < 1e-15, "stability limit = {limit}");
    }

    #[test]
    fn grid_construction() {
        let g = Grid1D::new(0.0, 1.0, 11);
        assert_eq!(g.nx, 11);
        assert!((g.dx - 0.1).abs() < 1e-15);
        assert!((g.point(0) - 0.0).abs() < 1e-15);
        assert!((g.point(5) - 0.5).abs() < 1e-15);
        assert!((g.point(10) - 1.0).abs() < 1e-15);

        let g2 = Grid2D::new(0.0, 1.0, 11, -1.0, 1.0, 21);
        assert_eq!(g2.nx, 11);
        assert_eq!(g2.ny, 21);
        assert!((g2.dx - 0.1).abs() < 1e-15);
        assert!((g2.dy - 0.1).abs() < 1e-15);
    }

    #[test]
    fn numerical_jacobian_accuracy() {
        // For y' = -y, the Jacobian is J = [-1]
        let sys = ExponentialDecay;
        let jac = numerical_jacobian(&sys, 0.0, &[1.0], 1e-8);
        assert!(jac.is_ok());
        let j = jac.ok().unwrap_or_default();
        if !j.is_empty() && !j[0].is_empty() {
            assert!(
                (j[0][0] - (-1.0)).abs() < 1e-5,
                "Jacobian J[0][0] = {}, expected -1.0",
                j[0][0]
            );
        }

        // Harmonic oscillator: J = [[0, 1], [-1, 0]]
        let sys2 = HarmonicOscillator;
        let jac2 = numerical_jacobian(&sys2, 0.0, &[1.0, 0.0], 1e-8);
        assert!(jac2.is_ok());
        let j2 = jac2.ok().unwrap_or_default();
        if j2.len() >= 2 && j2[0].len() >= 2 {
            assert!((j2[0][0]).abs() < 1e-5, "J[0][0] should be ~0");
            assert!((j2[0][1] - 1.0).abs() < 1e-5, "J[0][1] should be ~1");
            assert!((j2[1][0] - (-1.0)).abs() < 1e-5, "J[1][0] should be ~-1");
            assert!((j2[1][1]).abs() < 1e-5, "J[1][1] should be ~0");
        }
    }
}
