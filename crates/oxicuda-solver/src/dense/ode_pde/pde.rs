//! PDE types and solvers: Heat, Wave, Poisson, Advection (1-D).

use crate::error::{SolverError, SolverResult};

use super::utils::{apply_bc_1d, solve_tridiagonal};

// =========================================================================
// PDE types
// =========================================================================

/// One-dimensional uniform grid.
#[derive(Debug, Clone)]
pub struct Grid1D {
    /// Left boundary.
    pub x_min: f64,
    /// Right boundary.
    pub x_max: f64,
    /// Number of grid points.
    pub nx: usize,
    /// Grid spacing (computed).
    pub dx: f64,
}

impl Grid1D {
    /// Create a uniform 1-D grid with `nx` points from `x_min` to `x_max`.
    pub fn new(x_min: f64, x_max: f64, nx: usize) -> Self {
        let dx = if nx > 1 {
            (x_max - x_min) / (nx - 1) as f64
        } else {
            0.0
        };
        Self {
            x_min,
            x_max,
            nx,
            dx,
        }
    }

    /// Return the coordinate of grid point `i`.
    pub fn point(&self, i: usize) -> f64 {
        self.x_min + i as f64 * self.dx
    }
}

/// Two-dimensional uniform grid.
#[derive(Debug, Clone)]
pub struct Grid2D {
    /// X-direction range.
    pub x_min: f64,
    /// X-direction range.
    pub x_max: f64,
    /// Y-direction range.
    pub y_min: f64,
    /// Y-direction range.
    pub y_max: f64,
    /// Number of grid points in x.
    pub nx: usize,
    /// Number of grid points in y.
    pub ny: usize,
    /// Spacing in x.
    pub dx: f64,
    /// Spacing in y.
    pub dy: f64,
}

impl Grid2D {
    /// Create a uniform 2-D grid.
    pub fn new(x_min: f64, x_max: f64, nx: usize, y_min: f64, y_max: f64, ny: usize) -> Self {
        let dx = if nx > 1 {
            (x_max - x_min) / (nx - 1) as f64
        } else {
            0.0
        };
        let dy = if ny > 1 {
            (y_max - y_min) / (ny - 1) as f64
        } else {
            0.0
        };
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
            nx,
            ny,
            dx,
            dy,
        }
    }
}

/// Boundary condition type.
#[derive(Debug, Clone, Copy)]
pub enum BoundaryCondition {
    /// Fixed value at the boundary.
    Dirichlet(f64),
    /// Fixed derivative at the boundary.
    Neumann(f64),
    /// Periodic boundary (left = right).
    Periodic,
}

/// Configuration for a 1-D PDE solve.
#[derive(Debug, Clone)]
pub struct PdeConfig {
    /// Spatial grid.
    pub grid: Grid1D,
    /// Time step.
    pub dt: f64,
    /// Number of time steps.
    pub num_steps: usize,
    /// Left boundary condition.
    pub bc_left: BoundaryCondition,
    /// Right boundary condition.
    pub bc_right: BoundaryCondition,
}

// =========================================================================
// PDE solvers
// =========================================================================

/// 1-D heat equation solver: du/dt = alpha * d²u/dx².
pub struct HeatEquation1D {
    /// Thermal diffusivity.
    pub alpha: f64,
}

impl HeatEquation1D {
    /// Maximum stable time step for the explicit (FTCS) scheme.
    ///
    /// For stability we need dt <= dx² / (2 * alpha).
    pub fn stability_limit(&self, dx: f64) -> f64 {
        dx * dx / (2.0 * self.alpha)
    }

    /// Solve using Forward-Time Central-Space (FTCS) explicit scheme.
    pub fn solve_explicit(&self, u0: &[f64], config: &PdeConfig) -> SolverResult<Vec<Vec<f64>>> {
        let nx = config.grid.nx;
        if u0.len() != nx {
            return Err(SolverError::DimensionMismatch(format!(
                "heat_explicit: u0 length ({}) != nx ({nx})",
                u0.len()
            )));
        }

        let dx = config.grid.dx;
        let dt = config.dt;
        let r = self.alpha * dt / (dx * dx);

        let mut u = u0.to_vec();
        let mut results = vec![u.clone()];

        for _ in 0..config.num_steps {
            let mut u_new = u.clone();

            // Interior points
            for i in 1..nx - 1 {
                u_new[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
            }

            // Boundary conditions
            apply_bc_1d(&mut u_new, &config.bc_left, &config.bc_right, nx);

            u = u_new;
            results.push(u.clone());
        }

        Ok(results)
    }

    /// Solve using Crank-Nicolson (implicit) scheme.
    ///
    /// Unconditionally stable, second-order in both time and space.
    /// Reduces to a tridiagonal system at each time step.
    pub fn solve_implicit(&self, u0: &[f64], config: &PdeConfig) -> SolverResult<Vec<Vec<f64>>> {
        let nx = config.grid.nx;
        if u0.len() != nx {
            return Err(SolverError::DimensionMismatch(format!(
                "heat_implicit: u0 length ({}) != nx ({nx})",
                u0.len()
            )));
        }
        if nx < 3 {
            return Err(SolverError::DimensionMismatch(
                "heat_implicit: need at least 3 grid points".to_string(),
            ));
        }

        let dx = config.grid.dx;
        let dt = config.dt;
        let r = self.alpha * dt / (dx * dx);

        let mut u = u0.to_vec();
        let mut results = vec![u.clone()];

        // Interior system size
        let m = nx - 2;

        for _ in 0..config.num_steps {
            // Build RHS from explicit half: (I + r/2 * A) * u_interior
            let mut rhs = vec![0.0; m];
            for (i, rhs_i) in rhs.iter_mut().enumerate() {
                let idx = i + 1; // grid index
                *rhs_i = u[idx] + 0.5 * r * (u[idx + 1] - 2.0 * u[idx] + u[idx - 1]);
            }

            // Add boundary contributions
            match config.bc_left {
                BoundaryCondition::Dirichlet(val) => {
                    rhs[0] += 0.5 * r * val;
                }
                BoundaryCondition::Neumann(_) | BoundaryCondition::Periodic => {}
            }
            match config.bc_right {
                BoundaryCondition::Dirichlet(val) => {
                    if m > 0 {
                        rhs[m - 1] += 0.5 * r * val;
                    }
                }
                BoundaryCondition::Neumann(_) | BoundaryCondition::Periodic => {}
            }

            // Tridiagonal system: (I - r/2 * A) * u_new = rhs
            // sub-diag: -r/2, main: 1+r, super-diag: -r/2
            let sub = vec![-0.5 * r; m.saturating_sub(1)];
            let main = vec![1.0 + r; m];
            let sup = vec![-0.5 * r; m.saturating_sub(1)];

            let interior = solve_tridiagonal(&sub, &main, &sup, &rhs)?;

            // Assemble full solution
            let mut u_new = vec![0.0; nx];
            u_new[1..(m + 1)].copy_from_slice(&interior[..m]);

            apply_bc_1d(&mut u_new, &config.bc_left, &config.bc_right, nx);

            u = u_new;
            results.push(u.clone());
        }

        Ok(results)
    }
}

/// 1-D wave equation solver: d²u/dt² = c² * d²u/dx².
pub struct WaveEquation1D {
    /// Wave speed.
    pub c: f64,
}

impl WaveEquation1D {
    /// Compute the Courant number: c * dt / dx.
    pub fn courant_number(&self, dx: f64, dt: f64) -> f64 {
        self.c * dt / dx
    }

    /// Solve using the leapfrog / Störmer-Verlet scheme.
    ///
    /// `u0` is the initial displacement, `v0` the initial velocity.
    /// Stability requires Courant number <= 1.
    pub fn solve(&self, u0: &[f64], v0: &[f64], config: &PdeConfig) -> SolverResult<Vec<Vec<f64>>> {
        let nx = config.grid.nx;
        if u0.len() != nx || v0.len() != nx {
            return Err(SolverError::DimensionMismatch(format!(
                "wave_solve: u0/v0 length mismatch with nx ({nx})"
            )));
        }
        if nx < 3 {
            return Err(SolverError::DimensionMismatch(
                "wave_solve: need at least 3 grid points".to_string(),
            ));
        }

        let dx = config.grid.dx;
        let dt = config.dt;
        let cfl = self.c * dt / dx;
        let cfl2 = cfl * cfl;

        // u^{n-1} and u^{n}
        let mut u_prev = u0.to_vec();
        let mut u_cur = vec![0.0; nx];

        // First step uses Taylor expansion: u^1 = u^0 + dt*v0 + 0.5*dt²*c²*u''
        for i in 1..nx - 1 {
            let d2u = (u0[i + 1] - 2.0 * u0[i] + u0[i - 1]) / (dx * dx);
            u_cur[i] = u0[i] + dt * v0[i] + 0.5 * dt * dt * self.c * self.c * d2u;
        }
        apply_bc_1d(&mut u_cur, &config.bc_left, &config.bc_right, nx);

        let mut results = vec![u_prev.clone(), u_cur.clone()];

        // Leapfrog: u^{n+1} = 2*u^n - u^{n-1} + cfl² * (u_{i+1} - 2*u_i + u_{i-1})
        for _ in 1..config.num_steps {
            let mut u_next = vec![0.0; nx];
            for i in 1..nx - 1 {
                u_next[i] = 2.0 * u_cur[i] - u_prev[i]
                    + cfl2 * (u_cur[i + 1] - 2.0 * u_cur[i] + u_cur[i - 1]);
            }
            apply_bc_1d(&mut u_next, &config.bc_left, &config.bc_right, nx);

            u_prev = u_cur;
            u_cur = u_next;
            results.push(u_cur.clone());
        }

        Ok(results)
    }
}

/// 1-D Poisson equation solver: -u'' = f, with boundary conditions.
pub struct Poisson1D;

impl Poisson1D {
    /// Solve -u'' = f on the grid with specified boundary conditions.
    ///
    /// Uses a tridiagonal direct solve (Thomas algorithm).
    pub fn solve(&self, f: &[f64], config: &PdeConfig) -> SolverResult<Vec<f64>> {
        let nx = config.grid.nx;
        if f.len() != nx {
            return Err(SolverError::DimensionMismatch(format!(
                "poisson: f length ({}) != nx ({nx})",
                f.len()
            )));
        }
        if nx < 3 {
            return Err(SolverError::DimensionMismatch(
                "poisson: need at least 3 grid points".to_string(),
            ));
        }

        let dx = config.grid.dx;
        let dx2 = dx * dx;
        let m = nx - 2; // interior points

        // Build tridiagonal system: -u_{i-1} + 2*u_i - u_{i+1} = dx²*f_i
        let sub = vec![-1.0; m.saturating_sub(1)];
        let main = vec![2.0; m];
        let sup = vec![-1.0; m.saturating_sub(1)];

        let mut rhs = vec![0.0; m];
        for i in 0..m {
            rhs[i] = dx2 * f[i + 1];
        }

        // Add boundary contributions
        match config.bc_left {
            BoundaryCondition::Dirichlet(val) => {
                rhs[0] += val;
            }
            BoundaryCondition::Neumann(val) => {
                // Ghost point approach: u_{-1} = u_1 - 2*dx*val
                rhs[0] += -2.0 * dx * val; // approximate
            }
            BoundaryCondition::Periodic => {}
        }
        match config.bc_right {
            BoundaryCondition::Dirichlet(val) => {
                if m > 0 {
                    rhs[m - 1] += val;
                }
            }
            BoundaryCondition::Neumann(val) => {
                if m > 0 {
                    rhs[m - 1] += 2.0 * dx * val;
                }
            }
            BoundaryCondition::Periodic => {}
        }

        let interior = solve_tridiagonal(&sub, &main, &sup, &rhs)?;

        // Assemble full solution
        let mut u = vec![0.0; nx];
        u[1..(m + 1)].copy_from_slice(&interior[..m]);
        apply_bc_1d(&mut u, &config.bc_left, &config.bc_right, nx);

        Ok(u)
    }
}

/// 1-D advection equation solver: du/dt + a * du/dx = 0.
pub struct AdvectionEquation1D {
    /// Advection velocity.
    pub a: f64,
}

impl AdvectionEquation1D {
    /// Solve using the first-order upwind scheme.
    pub fn solve_upwind(&self, u0: &[f64], config: &PdeConfig) -> SolverResult<Vec<Vec<f64>>> {
        let nx = config.grid.nx;
        if u0.len() != nx {
            return Err(SolverError::DimensionMismatch(format!(
                "advection_upwind: u0 length ({}) != nx ({nx})",
                u0.len()
            )));
        }

        let dx = config.grid.dx;
        let dt = config.dt;
        let cfl = self.a * dt / dx;

        let mut u = u0.to_vec();
        let mut results = vec![u.clone()];

        for _ in 0..config.num_steps {
            let mut u_new = u.clone();

            for i in 1..nx - 1 {
                if self.a >= 0.0 {
                    // Upwind from left
                    u_new[i] = u[i] - cfl * (u[i] - u[i - 1]);
                } else {
                    // Upwind from right
                    u_new[i] = u[i] - cfl * (u[i + 1] - u[i]);
                }
            }

            apply_bc_1d(&mut u_new, &config.bc_left, &config.bc_right, nx);
            u = u_new;
            results.push(u.clone());
        }

        Ok(results)
    }

    /// Solve using the Lax-Wendroff scheme (second-order).
    pub fn solve_lax_wendroff(
        &self,
        u0: &[f64],
        config: &PdeConfig,
    ) -> SolverResult<Vec<Vec<f64>>> {
        let nx = config.grid.nx;
        if u0.len() != nx {
            return Err(SolverError::DimensionMismatch(format!(
                "advection_lw: u0 length ({}) != nx ({nx})",
                u0.len()
            )));
        }

        let dx = config.grid.dx;
        let dt = config.dt;
        let cfl = self.a * dt / dx;
        let cfl2 = cfl * cfl;

        let mut u = u0.to_vec();
        let mut results = vec![u.clone()];

        for _ in 0..config.num_steps {
            let mut u_new = u.clone();

            for i in 1..nx - 1 {
                u_new[i] = u[i] - 0.5 * cfl * (u[i + 1] - u[i - 1])
                    + 0.5 * cfl2 * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
            }

            apply_bc_1d(&mut u_new, &config.bc_left, &config.bc_right, nx);
            u = u_new;
            results.push(u.clone());
        }

        Ok(results)
    }
}
