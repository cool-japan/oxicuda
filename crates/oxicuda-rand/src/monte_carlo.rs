//! Monte Carlo simulation primitives.
//!
//! Provides CPU-side Monte Carlo integration, variance reduction techniques,
//! MCMC samplers (Metropolis-Hastings and Hamiltonian Monte Carlo), and
//! financial option pricing via geometric Brownian motion.
//!
//! All methods use an internal Xoshiro256**-style PRNG so they do not
//! require a CUDA context.

use crate::error::{RandError, RandResult};

// ---------------------------------------------------------------------------
// Internal PRNG -- Xoshiro256**
// ---------------------------------------------------------------------------

/// Minimal Xoshiro256** PRNG for CPU-side Monte Carlo.
#[derive(Clone, Debug)]
struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    /// Seed the generator using SplitMix64 to fill the state.
    fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            sm = sm.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            *slot = z;
        }
        Self { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Returns a uniform f64 in [0, 1).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Returns a standard normal variate via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 1e-300 {
                return (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration & Results
// ---------------------------------------------------------------------------

/// Configuration for Monte Carlo simulations.
#[derive(Clone, Debug)]
pub struct MonteCarloConfig {
    /// Number of samples to draw.
    pub num_samples: usize,
    /// PRNG seed.
    pub seed: u64,
    /// Confidence level for the confidence interval (default 0.95).
    pub confidence_level: f64,
    /// Enable antithetic variates when applicable.
    pub use_antithetic: bool,
    /// Enable control variates when applicable.
    pub use_control_variate: bool,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            num_samples: 10_000,
            seed: 42,
            confidence_level: 0.95,
            use_antithetic: false,
            use_control_variate: false,
        }
    }
}

impl MonteCarloConfig {
    /// Validate the configuration, returning an error on bad values.
    fn validate(&self) -> RandResult<()> {
        if self.num_samples == 0 {
            return Err(RandError::InvalidSize(
                "num_samples must be positive".to_string(),
            ));
        }
        if self.confidence_level <= 0.0 || self.confidence_level >= 1.0 {
            return Err(RandError::InvalidSize(
                "confidence_level must be in (0, 1)".to_string(),
            ));
        }
        Ok(())
    }
}

/// Result of a Monte Carlo estimation.
#[derive(Clone, Debug)]
pub struct MonteCarloResult {
    /// Point estimate.
    pub estimate: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
    /// Confidence interval `(lower, upper)`.
    pub confidence_interval: (f64, f64),
    /// Number of samples used.
    pub num_samples: usize,
    /// Sample variance.
    pub variance: f64,
}

/// Result of an MCMC sampling run.
#[derive(Clone, Debug)]
pub struct McmcResult {
    /// Collected samples (each inner Vec is one sample in R^d).
    pub samples: Vec<Vec<f64>>,
    /// Fraction of proposals accepted.
    pub acceptance_rate: f64,
    /// Estimated effective sample size.
    pub effective_sample_size: f64,
}

/// Internal state for MCMC chains.
#[derive(Clone, Debug)]
pub struct SamplerState {
    /// Current position in parameter space.
    pub position: Vec<f64>,
    /// Current log-density value.
    pub log_density: f64,
}

// ---------------------------------------------------------------------------
// Helper: z-quantile for confidence intervals
// ---------------------------------------------------------------------------

/// Approximate inverse of the standard normal CDF using rational approximation
/// (Abramowitz & Stegun 26.2.23).
fn normal_quantile(p: f64) -> f64 {
    // For p in (0, 0.5] we use the left tail; symmetry for p > 0.5.
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    let (sign, pp) = if p < 0.5 { (-1.0, 1.0 - p) } else { (1.0, p) };
    let t = (-2.0 * (1.0 - pp).ln()).sqrt();
    let c0 = 2.515_517;
    let c1 = 0.802_853;
    let c2 = 0.010_328;
    let d1 = 1.432_788;
    let d2 = 0.189_269;
    let d3 = 0.001_308;
    let numerator = c0 + c1 * t + c2 * t * t;
    let denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    sign * (t - numerator / denominator)
}

/// Build a [`MonteCarloResult`] from a slice of values.
fn build_result(values: &[f64], config: &MonteCarloConfig) -> MonteCarloResult {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var / n).sqrt();
    let z = normal_quantile(0.5 + config.confidence_level / 2.0);
    MonteCarloResult {
        estimate: mean,
        std_error: se,
        confidence_interval: (mean - z * se, mean + z * se),
        num_samples: values.len(),
        variance: var,
    }
}

// ---------------------------------------------------------------------------
// Core Integration
// ---------------------------------------------------------------------------

/// Simple Monte Carlo integration of `f` on \[0, 1\].
///
/// Estimates `\int_0^1 f(x) dx` by averaging `f(U_i)` over uniform samples.
///
/// # Errors
///
/// Returns an error if the configuration is invalid.
pub fn mc_integrate<F>(f: F, config: &MonteCarloConfig) -> RandResult<MonteCarloResult>
where
    F: Fn(f64) -> f64,
{
    config.validate()?;
    let mut rng = Xoshiro256::new(config.seed);
    let values: Vec<f64> = (0..config.num_samples).map(|_| f(rng.next_f64())).collect();
    Ok(build_result(&values, config))
}

/// Multi-dimensional Monte Carlo integration of `f` on \[0, 1\]^d.
///
/// # Errors
///
/// Returns an error if `dim` is zero or the configuration is invalid.
pub fn mc_integrate_nd<F>(
    f: F,
    dim: usize,
    config: &MonteCarloConfig,
) -> RandResult<MonteCarloResult>
where
    F: Fn(&[f64]) -> f64,
{
    config.validate()?;
    if dim == 0 {
        return Err(RandError::InvalidSize("dim must be >= 1".to_string()));
    }
    let mut rng = Xoshiro256::new(config.seed);
    let mut point = vec![0.0f64; dim];
    let values: Vec<f64> = (0..config.num_samples)
        .map(|_| {
            for x in point.iter_mut() {
                *x = rng.next_f64();
            }
            f(&point)
        })
        .collect();
    Ok(build_result(&values, config))
}

/// Importance sampling Monte Carlo integration.
///
/// Estimates `\int f(x) dx` using a proposal distribution `g` (pdf) from
/// which we can draw samples (`proposal_sample`). The estimator is
/// `(1/N) \sum f(x_i)/g(x_i)`.
///
/// # Errors
///
/// Returns an error if the configuration is invalid.
pub fn mc_integrate_importance<F, G, S>(
    f: F,
    proposal_pdf: G,
    proposal_sample: S,
    config: &MonteCarloConfig,
) -> RandResult<MonteCarloResult>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
    S: Fn(&mut dyn FnMut() -> f64) -> f64,
{
    config.validate()?;
    let mut rng = Xoshiro256::new(config.seed);
    let mut uniform = || rng.next_f64();
    let values: Vec<f64> = (0..config.num_samples)
        .map(|_| {
            let x = proposal_sample(&mut uniform);
            let g = proposal_pdf(x);
            if g.abs() < 1e-300 { 0.0 } else { f(x) / g }
        })
        .collect();
    Ok(build_result(&values, config))
}

// ---------------------------------------------------------------------------
// Variance Reduction
// ---------------------------------------------------------------------------

/// Antithetic variates integration of `f` on \[0, 1\].
///
/// Pairs each uniform sample `u` with `1 - u` to reduce variance.
///
/// # Errors
///
/// Returns an error if the configuration is invalid.
pub fn antithetic_integrate<F>(f: F, config: &MonteCarloConfig) -> RandResult<MonteCarloResult>
where
    F: Fn(f64) -> f64,
{
    config.validate()?;
    let mut rng = Xoshiro256::new(config.seed);
    // Each pair produces one averaged value.
    let half_n = config.num_samples / 2;
    let values: Vec<f64> = (0..half_n)
        .map(|_| {
            let u = rng.next_f64();
            0.5 * (f(u) + f(1.0 - u))
        })
        .collect();
    Ok(build_result(&values, config))
}

/// Control variate integration of `f` on \[0, 1\].
///
/// Uses a control function `c` with known mean `control_mean` to reduce
/// variance via `f(x) - beta * (c(x) - control_mean)`.
///
/// # Errors
///
/// Returns an error if the configuration is invalid.
pub fn control_variate_integrate<F, C>(
    f: F,
    control: C,
    control_mean: f64,
    config: &MonteCarloConfig,
) -> RandResult<MonteCarloResult>
where
    F: Fn(f64) -> f64,
    C: Fn(f64) -> f64,
{
    config.validate()?;
    let mut rng = Xoshiro256::new(config.seed);
    // First pass: collect raw f and c values to estimate optimal beta.
    let mut f_vals = Vec::with_capacity(config.num_samples);
    let mut c_vals = Vec::with_capacity(config.num_samples);
    for _ in 0..config.num_samples {
        let u = rng.next_f64();
        f_vals.push(f(u));
        c_vals.push(control(u));
    }

    let f_mean = f_vals.iter().sum::<f64>() / f_vals.len() as f64;
    let c_mean_sample = c_vals.iter().sum::<f64>() / c_vals.len() as f64;

    let mut cov = 0.0;
    let mut var_c = 0.0;
    for i in 0..f_vals.len() {
        let df = f_vals[i] - f_mean;
        let dc = c_vals[i] - c_mean_sample;
        cov += df * dc;
        var_c += dc * dc;
    }

    let beta = if var_c.abs() < 1e-300 {
        0.0
    } else {
        cov / var_c
    };

    let adjusted: Vec<f64> = f_vals
        .iter()
        .zip(c_vals.iter())
        .map(|(&fv, &cv)| fv - beta * (cv - control_mean))
        .collect();

    Ok(build_result(&adjusted, config))
}

/// Stratified sampling integration of `f` on \[0, 1\].
///
/// Divides \[0, 1\] into `num_strata` equal bins and samples within each.
///
/// # Errors
///
/// Returns an error if `num_strata` is zero or the configuration is invalid.
pub fn stratified_integrate<F>(
    f: F,
    num_strata: usize,
    config: &MonteCarloConfig,
) -> RandResult<MonteCarloResult>
where
    F: Fn(f64) -> f64,
{
    config.validate()?;
    if num_strata == 0 {
        return Err(RandError::InvalidSize(
            "num_strata must be >= 1".to_string(),
        ));
    }
    let mut rng = Xoshiro256::new(config.seed);
    let samples_per_stratum = config.num_samples / num_strata;
    let samples_per_stratum = if samples_per_stratum == 0 {
        1
    } else {
        samples_per_stratum
    };

    let mut values = Vec::with_capacity(num_strata * samples_per_stratum);
    let stratum_width = 1.0 / num_strata as f64;

    for i in 0..num_strata {
        let lo = i as f64 * stratum_width;
        for _ in 0..samples_per_stratum {
            let u = lo + rng.next_f64() * stratum_width;
            values.push(f(u));
        }
    }

    Ok(build_result(&values, config))
}

// ---------------------------------------------------------------------------
// MCMC -- Metropolis-Hastings
// ---------------------------------------------------------------------------

/// Metropolis-Hastings sampler with symmetric Gaussian proposals.
#[derive(Clone, Debug)]
pub struct MetropolisHastings {
    log_target: fn(&[f64]) -> f64,
    proposal_std: f64,
    dim: usize,
}

impl MetropolisHastings {
    /// Create a new Metropolis-Hastings sampler.
    ///
    /// - `log_target`: log of the unnormalized target density.
    /// - `proposal_std`: standard deviation of the isotropic Gaussian proposal.
    /// - `dim`: dimensionality of the parameter space.
    pub fn new(log_target: fn(&[f64]) -> f64, proposal_std: f64, dim: usize) -> Self {
        Self {
            log_target,
            proposal_std,
            dim,
        }
    }

    /// Run the sampler, discarding `burn_in` initial samples.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_samples` is zero.
    pub fn sample(
        &mut self,
        num_samples: usize,
        burn_in: usize,
        seed: u64,
    ) -> RandResult<McmcResult> {
        if num_samples == 0 {
            return Err(RandError::InvalidSize(
                "num_samples must be positive".to_string(),
            ));
        }
        let mut rng = Xoshiro256::new(seed);
        let mut current = vec![0.0f64; self.dim];
        let mut current_log_p = (self.log_target)(&current);
        let mut accepted = 0u64;
        let total = burn_in + num_samples;
        let mut samples = Vec::with_capacity(num_samples);

        for step in 0..total {
            // Propose
            let proposal: Vec<f64> = current
                .iter()
                .map(|&x| x + self.proposal_std * rng.next_normal())
                .collect();
            let proposal_log_p = (self.log_target)(&proposal);
            let log_alpha = proposal_log_p - current_log_p;

            if log_alpha >= 0.0 || rng.next_f64().ln() < log_alpha {
                current = proposal;
                current_log_p = proposal_log_p;
                if step >= burn_in {
                    accepted += 1;
                }
            } else if step >= burn_in {
                // rejection after burn-in still counts toward denominator but
                // not accepted count
            }

            if step >= burn_in {
                samples.push(current.clone());
            }
        }

        let ess = if self.dim > 0 {
            let first_dim: Vec<f64> = samples.iter().map(|s| s[0]).collect();
            effective_sample_size(&first_dim)
        } else {
            num_samples as f64
        };

        Ok(McmcResult {
            samples,
            acceptance_rate: accepted as f64 / num_samples as f64,
            effective_sample_size: ess,
        })
    }
}

// ---------------------------------------------------------------------------
// MCMC -- Hamiltonian Monte Carlo
// ---------------------------------------------------------------------------

/// Hamiltonian Monte Carlo (HMC) sampler.
#[derive(Clone, Debug)]
pub struct HamiltonianMC {
    log_target: fn(&[f64]) -> f64,
    grad_log_target: fn(&[f64]) -> Vec<f64>,
    step_size: f64,
    num_leapfrog: usize,
    dim: usize,
}

impl HamiltonianMC {
    /// Create a new HMC sampler.
    ///
    /// - `log_target`: log of the unnormalized target density.
    /// - `grad_log_target`: gradient of `log_target`.
    /// - `step_size`: leapfrog integrator step size (epsilon).
    /// - `num_leapfrog`: number of leapfrog steps per proposal.
    /// - `dim`: dimensionality of the parameter space.
    pub fn new(
        log_target: fn(&[f64]) -> f64,
        grad_log_target: fn(&[f64]) -> Vec<f64>,
        step_size: f64,
        num_leapfrog: usize,
        dim: usize,
    ) -> Self {
        Self {
            log_target,
            grad_log_target,
            step_size,
            num_leapfrog,
            dim,
        }
    }

    /// Run the HMC sampler, discarding `burn_in` initial samples.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_samples` is zero.
    pub fn sample(
        &mut self,
        num_samples: usize,
        burn_in: usize,
        seed: u64,
    ) -> RandResult<McmcResult> {
        if num_samples == 0 {
            return Err(RandError::InvalidSize(
                "num_samples must be positive".to_string(),
            ));
        }
        let mut rng = Xoshiro256::new(seed);
        let mut q = vec![0.0f64; self.dim];
        let total = burn_in + num_samples;
        let mut samples = Vec::with_capacity(num_samples);
        let mut accepted = 0u64;

        for step in 0..total {
            // Sample momentum
            let p: Vec<f64> = (0..self.dim).map(|_| rng.next_normal()).collect();
            let current_k: f64 = p.iter().map(|pi| 0.5 * pi * pi).sum();
            let current_u = -(self.log_target)(&q);

            let mut q_new = q.clone();
            let mut p_new = p.clone();

            // Leapfrog integration
            let grad = (self.grad_log_target)(&q_new);
            for (pi, gi) in p_new.iter_mut().zip(grad.iter()) {
                *pi += 0.5 * self.step_size * gi;
            }
            for _ in 0..self.num_leapfrog.saturating_sub(1) {
                for (qi, pi) in q_new.iter_mut().zip(p_new.iter()) {
                    *qi += self.step_size * pi;
                }
                let grad = (self.grad_log_target)(&q_new);
                for (pi, gi) in p_new.iter_mut().zip(grad.iter()) {
                    *pi += self.step_size * gi;
                }
            }
            // Final full position step and half momentum step
            for (qi, pi) in q_new.iter_mut().zip(p_new.iter()) {
                *qi += self.step_size * pi;
            }
            let grad = (self.grad_log_target)(&q_new);
            for (pi, gi) in p_new.iter_mut().zip(grad.iter()) {
                *pi += 0.5 * self.step_size * gi;
            }
            // Negate momentum (for reversibility, not strictly necessary for accept/reject)
            for pi in &mut p_new {
                *pi = -*pi;
            }

            let proposed_u = -(self.log_target)(&q_new);
            let proposed_k: f64 = p_new.iter().map(|pi| 0.5 * pi * pi).sum();

            let log_alpha = current_u + current_k - proposed_u - proposed_k;
            if log_alpha >= 0.0 || rng.next_f64().ln() < log_alpha {
                q = q_new;
                if step >= burn_in {
                    accepted += 1;
                }
            }

            if step >= burn_in {
                samples.push(q.clone());
            }
        }

        let ess = if self.dim > 0 {
            let first_dim: Vec<f64> = samples.iter().map(|s| s[0]).collect();
            effective_sample_size(&first_dim)
        } else {
            num_samples as f64
        };

        Ok(McmcResult {
            samples,
            acceptance_rate: accepted as f64 / num_samples as f64,
            effective_sample_size: ess,
        })
    }
}

// ---------------------------------------------------------------------------
// Financial Monte Carlo
// ---------------------------------------------------------------------------

/// Parameters for Black-Scholes style option pricing.
#[derive(Clone, Debug)]
pub struct BlackScholesParams {
    /// Current spot price.
    pub spot: f64,
    /// Strike price.
    pub strike: f64,
    /// Risk-free interest rate (annualized).
    pub risk_free_rate: f64,
    /// Volatility (annualized).
    pub volatility: f64,
    /// Time to maturity in years.
    pub time_to_maturity: f64,
}

/// Analytical Black-Scholes price for a European option.
///
/// Useful for validating Monte Carlo estimates against the closed-form
/// solution.
pub fn bs_analytical(params: &BlackScholesParams, is_call: bool) -> f64 {
    let s = params.spot;
    let k = params.strike;
    let r = params.risk_free_rate;
    let sigma = params.volatility;
    let t = params.time_to_maturity;

    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);

    if is_call {
        s * nd1 - k * (-r * t).exp() * nd2
    } else {
        k * (-r * t).exp() * normal_cdf(-d2) - s * normal_cdf(-d1)
    }
}

/// Approximate standard normal CDF (Abramowitz & Stegun 26.2.17).
///
/// Accurate to approximately 1e-7.
pub fn normal_cdf(x: f64) -> f64 {
    let b1 = 0.319_381_530;
    let b2 = -0.356_563_782;
    let b3 = 1.781_477_937;
    let b4 = -1.821_255_978;
    let b5 = 1.330_274_429;
    let pp = 0.231_641_9;

    if x >= 0.0 {
        let t = 1.0 / (1.0 + pp * x);
        let poly = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t;
        let pdf = (-0.5 * x * x).exp() / (2.0 * core::f64::consts::PI).sqrt();
        1.0 - pdf * poly
    } else {
        1.0 - normal_cdf(-x)
    }
}

/// Monte Carlo pricing of a European call or put option.
///
/// Simulates geometric Brownian motion terminal prices and averages the
/// discounted payoff.
///
/// # Errors
///
/// Returns an error if the configuration is invalid.
pub fn mc_european_option(
    params: &BlackScholesParams,
    is_call: bool,
    config: &MonteCarloConfig,
) -> RandResult<MonteCarloResult> {
    config.validate()?;
    let mut rng = Xoshiro256::new(config.seed);
    let r = params.risk_free_rate;
    let sigma = params.volatility;
    let t = params.time_to_maturity;
    let discount = (-r * t).exp();
    let drift = (r - 0.5 * sigma * sigma) * t;
    let vol_sqrt_t = sigma * t.sqrt();

    let values: Vec<f64> = (0..config.num_samples)
        .map(|_| {
            let z = rng.next_normal();
            let s_t = params.spot * (drift + vol_sqrt_t * z).exp();
            let payoff = if is_call {
                (s_t - params.strike).max(0.0)
            } else {
                (params.strike - s_t).max(0.0)
            };
            discount * payoff
        })
        .collect();

    Ok(build_result(&values, config))
}

/// Monte Carlo pricing of an arithmetic Asian option.
///
/// The payoff depends on the arithmetic average of the price at
/// `num_time_steps` equally spaced monitoring dates.
///
/// # Errors
///
/// Returns an error if the configuration is invalid or `num_time_steps` is zero.
pub fn mc_asian_option(
    params: &BlackScholesParams,
    num_time_steps: usize,
    is_call: bool,
    config: &MonteCarloConfig,
) -> RandResult<MonteCarloResult> {
    config.validate()?;
    if num_time_steps == 0 {
        return Err(RandError::InvalidSize(
            "num_time_steps must be positive".to_string(),
        ));
    }
    let mut rng = Xoshiro256::new(config.seed);
    let r = params.risk_free_rate;
    let sigma = params.volatility;
    let t = params.time_to_maturity;
    let dt = t / num_time_steps as f64;
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let vol_sqrt_dt = sigma * dt.sqrt();
    let discount = (-r * t).exp();

    let values: Vec<f64> = (0..config.num_samples)
        .map(|_| {
            let mut s = params.spot;
            let mut sum = 0.0;
            for _ in 0..num_time_steps {
                let z = rng.next_normal();
                s *= (drift + vol_sqrt_dt * z).exp();
                sum += s;
            }
            let avg = sum / num_time_steps as f64;
            let payoff = if is_call {
                (avg - params.strike).max(0.0)
            } else {
                (params.strike - avg).max(0.0)
            };
            discount * payoff
        })
        .collect();

    Ok(build_result(&values, config))
}

/// Monte Carlo pricing of a barrier option (up/down, knock-in/knock-out).
///
/// Uses discrete monitoring at each time step.
///
/// # Errors
///
/// Returns an error if the configuration is invalid.
pub fn mc_barrier_option(
    params: &BlackScholesParams,
    barrier: f64,
    is_up: bool,
    is_knock_out: bool,
    config: &MonteCarloConfig,
) -> RandResult<MonteCarloResult> {
    config.validate()?;
    let mut rng = Xoshiro256::new(config.seed);
    let r = params.risk_free_rate;
    let sigma = params.volatility;
    let t = params.time_to_maturity;
    let num_steps = 252; // daily monitoring
    let dt = t / num_steps as f64;
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let vol_sqrt_dt = sigma * dt.sqrt();
    let discount = (-r * t).exp();

    let values: Vec<f64> = (0..config.num_samples)
        .map(|_| {
            let mut s = params.spot;
            let mut hit_barrier = false;
            for _ in 0..num_steps {
                let z = rng.next_normal();
                s *= (drift + vol_sqrt_dt * z).exp();
                if (is_up && s >= barrier) || (!is_up && s <= barrier) {
                    hit_barrier = true;
                }
            }
            let payoff = (s - params.strike).max(0.0); // call payoff
            let active = if is_knock_out {
                !hit_barrier
            } else {
                hit_barrier
            };
            if active { discount * payoff } else { 0.0 }
        })
        .collect();

    Ok(build_result(&values, config))
}

// ---------------------------------------------------------------------------
// Convergence Analysis
// ---------------------------------------------------------------------------

/// Run Monte Carlo integration at multiple sample sizes to observe convergence.
///
/// Returns one [`MonteCarloResult`] per entry in `sample_sizes`.
///
/// # Errors
///
/// Returns an error if any sample size is zero.
pub fn convergence_analysis<F>(
    f: F,
    sample_sizes: &[usize],
    seed: u64,
) -> RandResult<Vec<MonteCarloResult>>
where
    F: Fn(f64) -> f64,
{
    let mut results = Vec::with_capacity(sample_sizes.len());
    for &n in sample_sizes {
        let config = MonteCarloConfig {
            num_samples: n,
            seed,
            confidence_level: 0.95,
            use_antithetic: false,
            use_control_variate: false,
        };
        results.push(mc_integrate(&f, &config)?);
    }
    Ok(results)
}

/// Estimate the effective sample size from autocorrelation.
///
/// Uses the initial positive sequence estimator: sums autocorrelations
/// until they become negative, then ESS = N / (1 + 2 * sum).
pub fn effective_sample_size(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n < 2 {
        return n as f64;
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-300 {
        return n as f64;
    }

    let max_lag = n / 2;
    let mut sum_rho = 0.0;
    for lag in 1..max_lag {
        let mut autocov = 0.0;
        for i in 0..(n - lag) {
            autocov += (samples[i] - mean) * (samples[i + lag] - mean);
        }
        autocov /= n as f64;
        let rho = autocov / var;
        if rho < 0.0 {
            break;
        }
        sum_rho += rho;
    }

    let tau = 1.0 + 2.0 * sum_rho;
    n as f64 / tau
}

/// Compute the Gelman-Rubin R-hat statistic for convergence of multiple
/// MCMC chains.
///
/// Values close to 1.0 indicate convergence; > 1.1 suggests further
/// sampling is needed.
///
/// Returns `NaN` if fewer than 2 chains or any chain has length < 2.
pub fn gelman_rubin(chains: &[Vec<f64>]) -> f64 {
    if chains.len() < 2 {
        return f64::NAN;
    }
    let m = chains.len() as f64;
    let n = chains[0].len();
    if n < 2 {
        return f64::NAN;
    }

    // Chain means and overall mean.
    let chain_means: Vec<f64> = chains
        .iter()
        .map(|c| c.iter().sum::<f64>() / c.len() as f64)
        .collect();
    let overall_mean = chain_means.iter().sum::<f64>() / m;

    // Between-chain variance B.
    let b = (n as f64 / (m - 1.0))
        * chain_means
            .iter()
            .map(|&cm| (cm - overall_mean).powi(2))
            .sum::<f64>();

    // Within-chain variance W.
    let w: f64 = chains
        .iter()
        .zip(chain_means.iter())
        .map(|(c, &cm)| c.iter().map(|&x| (x - cm).powi(2)).sum::<f64>() / (n as f64 - 1.0))
        .sum::<f64>()
        / m;

    // Pooled variance and R-hat.
    let var_hat = (1.0 - 1.0 / n as f64) * w + b / n as f64;
    if w < 1e-300 {
        return f64::NAN;
    }
    (var_hat / w).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(n: usize) -> MonteCarloConfig {
        MonteCarloConfig {
            num_samples: n,
            seed: 12345,
            confidence_level: 0.95,
            use_antithetic: false,
            use_control_variate: false,
        }
    }

    // 1. Integrate sqrt(1-x^2) on [0,1] -> pi/4
    #[test]
    fn test_mc_integrate_pi_quarter() {
        let config = default_config(200_000);
        let result = mc_integrate(|x| (1.0 - x * x).sqrt(), &config).expect("integration failed");
        let expected = core::f64::consts::FRAC_PI_4;
        assert!(
            (result.estimate - expected).abs() < 0.01,
            "estimate {} not close to pi/4 = {}",
            result.estimate,
            expected
        );
    }

    // 2. ND integration -- volume of unit sphere in 3D via indicator
    //    V_3 = (4/3)*pi, fraction in [0,1]^3 octant = (4/3)*pi/8 = pi/6
    #[test]
    fn test_mc_integrate_nd_sphere() {
        let config = default_config(500_000);
        let result = mc_integrate_nd(
            |x| {
                let r2: f64 = x.iter().map(|xi| xi * xi).sum();
                if r2 <= 1.0 { 1.0 } else { 0.0 }
            },
            3,
            &config,
        )
        .expect("nd integration failed");
        let expected = core::f64::consts::PI / 6.0;
        assert!(
            (result.estimate - expected).abs() < 0.01,
            "estimate {} not close to pi/6 = {}",
            result.estimate,
            expected
        );
    }

    // 3. Antithetic variates reduce variance
    #[test]
    fn test_antithetic_reduces_variance() {
        let config = default_config(100_000);
        let naive = mc_integrate(|x| (1.0 - x * x).sqrt(), &config).expect("naive failed");
        let anti = antithetic_integrate(|x| (1.0 - x * x).sqrt(), &config).expect("anti failed");
        assert!(
            anti.variance < naive.variance,
            "antithetic variance {} should be < naive variance {}",
            anti.variance,
            naive.variance
        );
    }

    // 4. Control variate accuracy
    #[test]
    fn test_control_variate() {
        let config = default_config(100_000);
        // Integrate x^2 using x as control (known mean = 0.5).
        let result = control_variate_integrate(|x| x * x, |x| x, 0.5, &config).expect("cv failed");
        let expected = 1.0 / 3.0;
        assert!(
            (result.estimate - expected).abs() < 0.005,
            "estimate {} not close to 1/3",
            result.estimate
        );
    }

    // 5. Stratified sampling
    #[test]
    fn test_stratified_sampling() {
        let config = default_config(100_000);
        let result =
            stratified_integrate(|x| (1.0 - x * x).sqrt(), 100, &config).expect("strat failed");
        let expected = core::f64::consts::FRAC_PI_4;
        assert!(
            (result.estimate - expected).abs() < 0.005,
            "estimate {} not close to pi/4",
            result.estimate
        );
    }

    // 6. Metropolis-Hastings on standard Gaussian target
    #[test]
    fn test_metropolis_hastings_gaussian() {
        fn log_normal(x: &[f64]) -> f64 {
            -0.5 * x[0] * x[0]
        }
        let mut mh = MetropolisHastings::new(log_normal, 1.0, 1);
        let result = mh.sample(50_000, 5_000, 42).expect("mh failed");
        // Mean should be near 0.
        let mean: f64 =
            result.samples.iter().map(|s| s[0]).sum::<f64>() / result.samples.len() as f64;
        assert!(mean.abs() < 0.1, "MH mean {} should be near 0", mean);
        assert!(
            result.acceptance_rate > 0.1 && result.acceptance_rate < 0.95,
            "acceptance rate {} out of range",
            result.acceptance_rate
        );
    }

    // 7. HMC on 2D Gaussian
    #[test]
    fn test_hmc_2d_gaussian() {
        fn log_target(x: &[f64]) -> f64 {
            -0.5 * (x[0] * x[0] + x[1] * x[1])
        }
        fn grad_log_target(x: &[f64]) -> Vec<f64> {
            vec![-x[0], -x[1]]
        }
        let mut hmc = HamiltonianMC::new(log_target, grad_log_target, 0.1, 20, 2);
        let result = hmc.sample(10_000, 2_000, 99).expect("hmc failed");
        let mean_x: f64 =
            result.samples.iter().map(|s| s[0]).sum::<f64>() / result.samples.len() as f64;
        let mean_y: f64 =
            result.samples.iter().map(|s| s[1]).sum::<f64>() / result.samples.len() as f64;
        assert!(mean_x.abs() < 0.1, "HMC mean_x {} should be near 0", mean_x);
        assert!(mean_y.abs() < 0.1, "HMC mean_y {} should be near 0", mean_y);
        assert!(
            result.acceptance_rate > 0.5,
            "HMC acceptance {} too low",
            result.acceptance_rate
        );
    }

    // 8. European call vs Black-Scholes analytical
    #[test]
    fn test_european_call_bs() {
        let params = BlackScholesParams {
            spot: 100.0,
            strike: 100.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
            time_to_maturity: 1.0,
        };
        let config = default_config(1_000_000);
        let mc = mc_european_option(&params, true, &config).expect("eu call failed");
        let analytical = bs_analytical(&params, true);
        assert!(
            (mc.estimate - analytical).abs() < 0.5,
            "MC price {} not close to BS price {}",
            mc.estimate,
            analytical
        );
    }

    // 9. Asian option convergence (just check it runs and is positive)
    #[test]
    fn test_asian_option() {
        let params = BlackScholesParams {
            spot: 100.0,
            strike: 100.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
            time_to_maturity: 1.0,
        };
        let config = default_config(50_000);
        let result = mc_asian_option(&params, 50, true, &config).expect("asian failed");
        assert!(result.estimate > 0.0, "Asian call price should be positive");
        // Asian call should be cheaper than European call.
        let eu = mc_european_option(&params, true, &config).expect("eu failed");
        assert!(
            result.estimate < eu.estimate + 2.0,
            "Asian call {} should be <= European call {}",
            result.estimate,
            eu.estimate
        );
    }

    // 10. Barrier option
    #[test]
    fn test_barrier_option() {
        let params = BlackScholesParams {
            spot: 100.0,
            strike: 100.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
            time_to_maturity: 1.0,
        };
        let config = default_config(100_000);
        let result =
            mc_barrier_option(&params, 130.0, true, true, &config).expect("barrier failed");
        // Up-and-out call: price should be less than vanilla European call.
        let eu = mc_european_option(&params, true, &config).expect("eu failed");
        assert!(
            result.estimate < eu.estimate + 0.5,
            "Barrier {} should be <= European {}",
            result.estimate,
            eu.estimate
        );
        assert!(result.estimate >= 0.0, "Price should be non-negative");
    }

    // 11. Convergence analysis -- error decreases with N
    #[test]
    fn test_convergence_analysis() {
        let sizes = [1_000, 10_000, 100_000];
        let results = convergence_analysis(|x| x * x, &sizes, 42).expect("conv failed");
        assert_eq!(results.len(), 3);
        // Standard error should generally decrease.
        assert!(
            results[2].std_error < results[0].std_error,
            "SE at 100k ({}) should be < SE at 1k ({})",
            results[2].std_error,
            results[0].std_error
        );
    }

    // 12. Effective sample size
    #[test]
    fn test_effective_sample_size() {
        // Uncorrelated samples => ESS ~= N.
        let mut rng = Xoshiro256::new(42);
        let samples: Vec<f64> = (0..10_000).map(|_| rng.next_f64()).collect();
        let ess = effective_sample_size(&samples);
        assert!(
            ess > 5_000.0,
            "ESS {} should be high for uncorrelated samples",
            ess
        );

        // Highly correlated samples => ESS << N.
        let mut corr = Vec::with_capacity(10_000);
        let mut x = 0.0f64;
        for _ in 0..10_000 {
            x = 0.999 * x + 0.001 * rng.next_normal();
            corr.push(x);
        }
        let ess_corr = effective_sample_size(&corr);
        assert!(
            ess_corr < 1_000.0,
            "ESS {} should be low for correlated samples",
            ess_corr
        );
    }

    // 13. Gelman-Rubin R-hat close to 1 for converged chains
    #[test]
    fn test_gelman_rubin() {
        let mut rng1 = Xoshiro256::new(1);
        let mut rng2 = Xoshiro256::new(2);
        let mut rng3 = Xoshiro256::new(3);
        let chain1: Vec<f64> = (0..5_000).map(|_| rng1.next_normal()).collect();
        let chain2: Vec<f64> = (0..5_000).map(|_| rng2.next_normal()).collect();
        let chain3: Vec<f64> = (0..5_000).map(|_| rng3.next_normal()).collect();
        let rhat = gelman_rubin(&[chain1, chain2, chain3]);
        assert!(
            (rhat - 1.0).abs() < 0.05,
            "R-hat {} should be close to 1.0",
            rhat
        );
    }

    // 14. Importance sampling
    #[test]
    fn test_importance_sampling() {
        // Integrate x^2 on [0,1] using uniform(0,1) proposal (trivial case).
        let config = default_config(100_000);
        let result = mc_integrate_importance(
            |x| x * x,
            |_x| 1.0,    // uniform pdf
            |rng| rng(), // sample from uniform
            &config,
        )
        .expect("importance sampling failed");
        let expected = 1.0 / 3.0;
        assert!(
            (result.estimate - expected).abs() < 0.01,
            "estimate {} not close to 1/3",
            result.estimate
        );
    }

    // 15. Config validation
    #[test]
    fn test_config_validation() {
        let bad_size = MonteCarloConfig {
            num_samples: 0,
            ..MonteCarloConfig::default()
        };
        assert!(mc_integrate(|x| x, &bad_size).is_err());

        let bad_confidence = MonteCarloConfig {
            confidence_level: 1.5,
            ..MonteCarloConfig::default()
        };
        assert!(mc_integrate(|x| x, &bad_confidence).is_err());

        let bad_confidence2 = MonteCarloConfig {
            confidence_level: 0.0,
            ..MonteCarloConfig::default()
        };
        assert!(mc_integrate(|x| x, &bad_confidence2).is_err());
    }
}
