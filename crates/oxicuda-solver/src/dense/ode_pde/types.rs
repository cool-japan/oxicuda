//! ODE types and configuration.

use crate::error::SolverResult;

// =========================================================================
// ODE types
// =========================================================================

/// Right-hand side of an ODE system dy/dt = f(t, y).
pub trait OdeSystem {
    /// Evaluate the right-hand side at `(t, y)`, writing into `dydt`.
    fn rhs(&self, t: f64, y: &[f64], dydt: &mut [f64]) -> SolverResult<()>;
    /// Dimension of the state vector.
    fn dim(&self) -> usize;
}

/// Integration method selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeMethod {
    /// Forward Euler (first order, explicit).
    Euler,
    /// Classical fourth-order Runge-Kutta.
    Rk4,
    /// Dormand-Prince 4(5) with adaptive step-size control.
    Rk45,
    /// Backward Euler (first order, implicit — for stiff systems).
    ImplicitEuler,
    /// Second-order backward differentiation formula (for stiff systems).
    Bdf2,
}

/// Configuration for an ODE integration.
#[derive(Debug, Clone)]
pub struct OdeConfig {
    /// Start time.
    pub t_start: f64,
    /// End time.
    pub t_end: f64,
    /// Initial step size.
    pub dt: f64,
    /// Relative tolerance (used by adaptive methods).
    pub rtol: f64,
    /// Absolute tolerance (used by adaptive methods).
    pub atol: f64,
    /// Maximum number of steps before giving up.
    pub max_steps: usize,
    /// Which integration method to use.
    pub method: OdeMethod,
}

impl Default for OdeConfig {
    fn default() -> Self {
        Self {
            t_start: 0.0,
            t_end: 1.0,
            dt: 1e-3,
            rtol: 1e-6,
            atol: 1e-9,
            max_steps: 10_000,
            method: OdeMethod::Rk4,
        }
    }
}

/// Solution returned by an ODE solver.
#[derive(Debug, Clone)]
pub struct OdeSolution {
    /// Time points at which the state was recorded.
    pub times: Vec<f64>,
    /// State vector at each recorded time point.
    pub states: Vec<Vec<f64>>,
    /// Total number of accepted steps.
    pub num_steps: usize,
    /// Number of rejected steps (adaptive methods only).
    pub num_rejected: usize,
    /// Total number of right-hand side evaluations.
    pub num_rhs_evals: usize,
}

/// Result of a single integration step (adaptive methods).
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Whether the step was accepted.
    pub accepted: bool,
    /// Suggested next step size.
    pub dt_next: f64,
    /// Estimated local truncation error.
    pub error_estimate: f64,
}
