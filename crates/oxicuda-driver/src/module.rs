//! PTX module loading and kernel function management.
//!
//! Modules are created from PTX source code and contain one or more
//! kernel functions that can be launched on the GPU.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_driver::module::{Module, JitOptions};
//! # fn main() -> Result<(), oxicuda_driver::error::CudaError> {
//! let ptx = r#"
//! .version 7.0
//! .target sm_70
//! .address_size 64
//! .visible .entry my_kernel() { ret; }
//! "#;
//!
//! let module = Module::from_ptx(ptx)?;
//! let func = module.get_function("my_kernel")?;
//!
//! // Or with JIT options and compilation logs:
//! let opts = JitOptions { optimization_level: 4, ..Default::default() };
//! let (module2, log) = Module::from_ptx_with_options(ptx, &opts)?;
//! if !log.info.is_empty() {
//!     println!("JIT info: {}", log.info);
//! }
//! # Ok(())
//! # }
//! ```

use std::ffi::{CString, c_void};

use crate::error::{CudaError, CudaResult};
use crate::ffi::{CUfunction, CUjit_option, CUmodule};
use crate::loader::try_driver;

// ---------------------------------------------------------------------------
// JitOptions
// ---------------------------------------------------------------------------

/// Options for JIT compilation of PTX to GPU binary.
///
/// These options control the behaviour of the CUDA JIT compiler when
/// loading PTX source via [`Module::from_ptx_with_options`].
#[derive(Debug, Clone)]
pub struct JitOptions {
    /// Maximum number of registers per thread (0 = no limit).
    ///
    /// Limiting register usage can increase occupancy at the cost of
    /// potential register spilling to local memory.
    pub max_registers: u32,
    /// Optimisation level (0--4, default 4).
    ///
    /// Higher levels produce faster code but take longer to compile.
    pub optimization_level: u32,
    /// Whether to generate debug information in the compiled binary.
    pub generate_debug_info: bool,
    /// If `true`, the JIT compiler determines the target compute
    /// capability from the current CUDA context.
    pub target_from_context: bool,
}

impl Default for JitOptions {
    /// Returns sensible defaults: no register limit, maximum
    /// optimisation, no debug info, target derived from context.
    fn default() -> Self {
        Self {
            max_registers: 0,
            optimization_level: 4,
            generate_debug_info: false,
            target_from_context: true,
        }
    }
}

// ---------------------------------------------------------------------------
// JitLog
// ---------------------------------------------------------------------------

/// Severity of a JIT compiler diagnostic message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JitSeverity {
    /// A fatal error that prevents PTX compilation.
    Fatal,
    /// A non-fatal error.
    Error,
    /// A compiler warning (compilation may still succeed).
    Warning,
    /// An informational message (e.g. register usage).
    Info,
}

impl std::fmt::Display for JitSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fatal => f.write_str("fatal"),
            Self::Error => f.write_str("error"),
            Self::Warning => f.write_str("warning"),
            Self::Info => f.write_str("info"),
        }
    }
}

/// A single structured diagnostic emitted by the JIT compiler.
///
/// Parsed from the raw `ptxas` log lines that look like:
///
/// ```text
/// ptxas error   : 'kernel', line 10; error   : Unknown instruction 'xyz'
/// ptxas warning : 'kernel', line 15; warning : double-precision is slow
/// ptxas info    : 'kernel' used 16 registers, 0 bytes smem
/// ```
#[derive(Debug, Clone)]
pub struct JitDiagnostic {
    /// Severity level.
    pub severity: JitSeverity,
    /// Kernel function name, if the message is function-scoped.
    pub kernel: Option<String>,
    /// Source line number, if present.
    pub line: Option<u32>,
    /// Human-readable message text.
    pub message: String,
}

/// Log output from JIT compilation.
///
/// After calling [`Module::from_ptx_with_options`], this struct
/// contains any informational or error messages emitted by the
/// JIT compiler.
///
/// Use [`JitLog::parse_diagnostics`] to obtain structured
/// [`JitDiagnostic`] entries instead of parsing the raw strings.
#[derive(Debug, Clone, Default)]
pub struct JitLog {
    /// Informational messages from the JIT compiler.
    pub info: String,
    /// Error messages from the JIT compiler.
    pub error: String,
}

impl JitLog {
    /// Returns `true` if there are no messages in either buffer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.info.is_empty() && self.error.is_empty()
    }

    /// Returns `true` if the error buffer is non-empty.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.error.is_empty()
    }

    /// Parse both log buffers into a `Vec` of structured [`JitDiagnostic`]
    /// entries.
    ///
    /// Lines that do not match the `ptxas` diagnostic format are included as
    /// [`JitSeverity::Info`] diagnostics with no kernel or line information,
    /// unless they are entirely blank.
    ///
    /// # Message format
    ///
    /// The CUDA JIT compiler emits lines in one of these formats:
    ///
    /// ```text
    /// ptxas {severity}   : '{kernel}', line {n}; {type}   : {message}
    /// ptxas {severity}   : '{kernel}' {message}
    /// ptxas {severity}   : {message}
    /// ```
    ///
    /// This method normalises all of those into [`JitDiagnostic`] values.
    #[must_use]
    pub fn parse_diagnostics(&self) -> Vec<JitDiagnostic> {
        let mut out = Vec::new();
        for line in self.error.lines().chain(self.info.lines()) {
            if let Some(d) = parse_ptxas_line(line) {
                out.push(d);
            }
        }
        out
    }

    /// Return only the [`JitDiagnostic`] entries whose severity is
    /// [`JitSeverity::Error`] or [`JitSeverity::Fatal`].
    #[must_use]
    pub fn errors(&self) -> Vec<JitDiagnostic> {
        self.parse_diagnostics()
            .into_iter()
            .filter(|d| matches!(d.severity, JitSeverity::Error | JitSeverity::Fatal))
            .collect()
    }

    /// Return only the [`JitDiagnostic`] entries whose severity is
    /// [`JitSeverity::Warning`].
    #[must_use]
    pub fn warnings(&self) -> Vec<JitDiagnostic> {
        self.parse_diagnostics()
            .into_iter()
            .filter(|d| matches!(d.severity, JitSeverity::Warning))
            .collect()
    }
}

// ── ptxas log line parser ─────────────────────────────────────────────────────

/// Parse a single `ptxas` log line into a [`JitDiagnostic`], returning
/// `None` for blank lines.
///
/// Handles these representative patterns:
///
/// ```text
/// ptxas error   : 'vec_add', line 10; error   : Unknown instruction 'xyz'
/// ptxas warning : 'vec_add', line 15; warning : slow double-precision
/// ptxas info    : 'vec_add' used 16 registers, 0 bytes smem, 0 bytes cmem[0]
/// ptxas fatal   : Unresolved extern function 'foo'
/// ```
fn parse_ptxas_line(line: &str) -> Option<JitDiagnostic> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }

    // Must start with "ptxas " (case-insensitive is not needed — ptxas always
    // uses lower-case).
    let rest = line.strip_prefix("ptxas ")?;

    // Extract severity word (first whitespace-delimited token).
    let (sev_str, after_sev) = split_first_word(rest.trim_start());
    let severity = match sev_str.to_ascii_lowercase().trim_end_matches(':') {
        "fatal" => JitSeverity::Fatal,
        "error" => JitSeverity::Error,
        "warning" => JitSeverity::Warning,
        "info" => JitSeverity::Info,
        _ => JitSeverity::Info,
    };

    // Skip past `: ` after the severity keyword.
    let body = skip_colon(after_sev.trim_start());

    // Try to extract kernel name from `'kernel_name'` at the start.
    let (kernel, after_kernel) = extract_kernel_name(body);

    // Try to extract line number: `, line N;` or `, line N,`.
    let (line_no, after_line) = extract_line_number(after_kernel);

    // The remaining text — skip a leading type word if present (e.g. `error   : `).
    let message = extract_message(after_line.trim());

    Some(JitDiagnostic {
        severity,
        kernel,
        line: line_no,
        message: message.to_string(),
    })
}

/// Split a `&str` at the first whitespace boundary; returns `("", s)` if
/// there is no whitespace.
fn split_first_word(s: &str) -> (&str, &str) {
    match s.find(|c: char| c.is_whitespace()) {
        Some(pos) => (&s[..pos], &s[pos..]),
        None => (s, ""),
    }
}

/// Skip past the first `: ` (colon + optional spaces) in `s`.
fn skip_colon(s: &str) -> &str {
    if let Some(pos) = s.find(':') {
        s[pos + 1..].trim_start()
    } else {
        s
    }
}

/// Attempt to extract `'kernel_name'` from the beginning of `s`.
/// Returns `(Some(name), rest_after_name)` or `(None, s)`.
fn extract_kernel_name(s: &str) -> (Option<String>, &str) {
    let s = s.trim_start();
    if !s.starts_with('\'') {
        return (None, s);
    }
    let inner = &s[1..];
    if let Some(end) = inner.find('\'') {
        let name = inner[..end].to_string();
        let after = &inner[end + 1..];
        (Some(name), after)
    } else {
        (None, s)
    }
}

/// Attempt to extract `, line N;` or `, line N,` from the start of `s`.
/// Returns `(Some(n), rest)` or `(None, s)`.
fn extract_line_number(s: &str) -> (Option<u32>, &str) {
    // Accept `, line N` (with optional trailing `;` or `,`)
    let s_trim = s.trim_start_matches([',', ' ', ';']);
    let lower = s_trim.to_ascii_lowercase();
    if !lower.starts_with("line ") {
        return (None, s);
    }
    let after_line = &s_trim[5..]; // skip "line "
    let (num_str, rest) = split_first_word(after_line.trim_start());
    let num_clean: String = num_str.chars().filter(|c| c.is_ascii_digit()).collect();
    if let Ok(n) = num_clean.parse::<u32>() {
        (Some(n), rest)
    } else {
        (None, s)
    }
}

/// Strip a leading `type   : ` prefix (e.g. `error   : ` or `warning : `)
/// from a message if present; return the remaining text.
fn extract_message(s: &str) -> &str {
    // Pattern: word followed by optional spaces and `:`.
    let (word, rest) = split_first_word(s);
    let word_clean = word.trim_end_matches(':');
    if matches!(
        word_clean.to_ascii_lowercase().as_str(),
        "error" | "warning" | "info" | "fatal"
    ) {
        skip_colon(rest.trim_start())
    } else {
        s
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// A loaded CUDA module containing one or more kernel functions.
///
/// Modules are typically created from PTX source via [`Module::from_ptx`]
/// or [`Module::from_ptx_with_options`]. Individual kernel functions
/// are retrieved by name with [`Module::get_function`].
///
/// The module is unloaded when this struct is dropped.
pub struct Module {
    /// Raw CUDA module handle.
    raw: CUmodule,
}

// SAFETY: CUDA modules are safe to send between threads when properly
// synchronised via the driver API.
unsafe impl Send for Module {}

/// Size of the JIT log buffers in bytes.
const JIT_LOG_BUFFER_SIZE: usize = 4096;

impl Module {
    /// Loads a module from PTX source with default JIT options.
    ///
    /// The PTX string is automatically null-terminated before being
    /// passed to the driver.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidImage`] if
    /// the PTX is malformed, or another [`CudaError`] if the driver
    /// call fails (e.g. no current context).
    pub fn from_ptx(ptx: &str) -> CudaResult<Self> {
        let api = try_driver()?;
        let c_ptx = CString::new(ptx).map_err(|_| CudaError::InvalidValue)?;
        let mut raw = CUmodule::default();
        crate::cuda_call!((api.cu_module_load_data)(
            &mut raw,
            c_ptx.as_ptr().cast::<c_void>()
        ))?;
        Ok(Self { raw })
    }

    /// Loads a module from PTX source with explicit JIT compiler options.
    ///
    /// Returns the loaded module together with a [`JitLog`] containing
    /// any informational or error messages from the JIT compiler.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] if JIT compilation fails or the driver
    /// call otherwise errors.
    pub fn from_ptx_with_options(ptx: &str, options: &JitOptions) -> CudaResult<(Self, JitLog)> {
        let api = try_driver()?;
        let c_ptx = CString::new(ptx).map_err(|_| CudaError::InvalidValue)?;

        // Allocate log buffers on the heap.
        let mut info_buf: Vec<u8> = vec![0u8; JIT_LOG_BUFFER_SIZE];
        let mut error_buf: Vec<u8> = vec![0u8; JIT_LOG_BUFFER_SIZE];

        // Build the parallel option-key and option-value arrays.
        //
        // Each option is a (CUjit_option, *mut c_void) pair. The value
        // pointer is reinterpreted according to the option key — scalar
        // values are cast directly to pointer-width integers.
        let mut opt_keys: Vec<CUjit_option> = Vec::with_capacity(8);
        let mut opt_vals: Vec<*mut c_void> = Vec::with_capacity(8);

        // Info log buffer.
        opt_keys.push(CUjit_option::InfoLogBuffer);
        opt_vals.push(info_buf.as_mut_ptr().cast::<c_void>());

        opt_keys.push(CUjit_option::InfoLogBufferSizeBytes);
        opt_vals.push(JIT_LOG_BUFFER_SIZE as *mut c_void);

        // Error log buffer.
        opt_keys.push(CUjit_option::ErrorLogBuffer);
        opt_vals.push(error_buf.as_mut_ptr().cast::<c_void>());

        opt_keys.push(CUjit_option::ErrorLogBufferSizeBytes);
        opt_vals.push(JIT_LOG_BUFFER_SIZE as *mut c_void);

        // Optimisation level.
        opt_keys.push(CUjit_option::OptimizationLevel);
        opt_vals.push(options.optimization_level as *mut c_void);

        // Max registers (only if non-zero to avoid overriding defaults).
        if options.max_registers > 0 {
            opt_keys.push(CUjit_option::MaxRegisters);
            opt_vals.push(options.max_registers as *mut c_void);
        }

        // Generate debug info.
        if options.generate_debug_info {
            opt_keys.push(CUjit_option::GenerateDebugInfo);
            opt_vals.push(core::ptr::without_provenance_mut::<c_void>(1));
        }

        // Target from context.
        if options.target_from_context {
            opt_keys.push(CUjit_option::TargetFromCuContext);
            opt_vals.push(core::ptr::without_provenance_mut::<c_void>(1));
        }

        let num_options = opt_keys.len() as u32;

        let mut raw = CUmodule::default();
        let result = crate::cuda_call!((api.cu_module_load_data_ex)(
            &mut raw,
            c_ptx.as_ptr().cast::<c_void>(),
            num_options,
            opt_keys.as_mut_ptr(),
            opt_vals.as_mut_ptr(),
        ));

        // Extract log strings regardless of success or failure.
        let log = JitLog {
            info: buf_to_string(&info_buf),
            error: buf_to_string(&error_buf),
        };

        result?;
        Ok((Self { raw }, log))
    }

    /// Retrieves a kernel function by name from this module.
    ///
    /// The returned [`Function`] is a lightweight handle. The caller
    /// must ensure that this `Module` outlives any `Function` handles
    /// obtained from it.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotFound`] if no
    /// function with the given name exists in the module, or another
    /// [`CudaError`] on driver failure.
    pub fn get_function(&self, name: &str) -> CudaResult<Function> {
        let api = try_driver()?;
        let c_name = CString::new(name).map_err(|_| CudaError::InvalidValue)?;
        let mut raw = CUfunction::default();
        crate::cuda_call!((api.cu_module_get_function)(
            &mut raw,
            self.raw,
            c_name.as_ptr()
        ))?;
        Ok(Function { raw })
    }

    /// Returns the raw [`CUmodule`] handle.
    ///
    /// # Safety (caller)
    ///
    /// The caller must not unload or otherwise invalidate the handle
    /// while this `Module` is still alive.
    #[inline]
    pub fn raw(&self) -> CUmodule {
        self.raw
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if let Ok(api) = try_driver() {
            let rc = unsafe { (api.cu_module_unload)(self.raw) };
            if rc != 0 {
                tracing::warn!(
                    cuda_error = rc,
                    module = ?self.raw,
                    "cuModuleUnload failed during drop"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Function
// ---------------------------------------------------------------------------

/// A kernel function handle within a loaded module.
///
/// Functions are lightweight handles (a single pointer) — the lifetime
/// is tied to the parent [`Module`]. The caller is responsible for
/// ensuring the `Module` outlives any `Function` handles obtained
/// from it.
///
/// Occupancy query methods are provided in the [`crate::occupancy`]
/// module via an `impl Function` block.
#[derive(Debug, Clone, Copy)]
pub struct Function {
    /// Raw CUDA function handle.
    raw: CUfunction,
}

impl Function {
    /// Returns the raw [`CUfunction`] handle.
    ///
    /// This is needed for kernel launches and occupancy queries
    /// at the FFI level.
    #[inline]
    pub fn raw(&self) -> CUfunction {
        self.raw
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Converts a null-terminated C buffer to a Rust [`String`], trimming
/// trailing null bytes and whitespace.
fn buf_to_string(buf: &[u8]) -> String {
    // Find the first null byte (or use the whole buffer).
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..len]).trim().to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_ptxas_line ──────────────────────────────────────────────────────

    #[test]
    fn parse_blank_line_returns_none() {
        assert!(parse_ptxas_line("").is_none());
        assert!(parse_ptxas_line("   ").is_none());
    }

    #[test]
    fn parse_non_ptxas_line_returns_none() {
        // Lines not starting with "ptxas " are ignored.
        assert!(parse_ptxas_line("nvcc error: something").is_none());
        assert!(parse_ptxas_line("  error: foo").is_none());
    }

    #[test]
    fn parse_standard_error_with_kernel_and_line() {
        let line = "ptxas error   : 'vec_add', line 42; error   : Unknown instruction 'xyz.f32'";
        let d = parse_ptxas_line(line).expect("should parse");
        assert_eq!(d.severity, JitSeverity::Error);
        assert_eq!(d.kernel.as_deref(), Some("vec_add"));
        assert_eq!(d.line, Some(42));
        assert!(
            d.message.contains("Unknown instruction"),
            "msg: {}",
            d.message
        );
    }

    #[test]
    fn parse_warning_with_kernel_and_line() {
        let line = "ptxas warning : 'my_kernel', line 7; warning : Double-precision instructions will be slow";
        let d = parse_ptxas_line(line).expect("should parse");
        assert_eq!(d.severity, JitSeverity::Warning);
        assert_eq!(d.kernel.as_deref(), Some("my_kernel"));
        assert_eq!(d.line, Some(7));
        assert!(d.message.contains("Double-precision"), "msg: {}", d.message);
    }

    #[test]
    fn parse_info_register_usage() {
        let line =
            "ptxas info    : 'reduce_kernel' used 32 registers, 0 bytes smem, 0 bytes cmem[0]";
        let d = parse_ptxas_line(line).expect("should parse");
        assert_eq!(d.severity, JitSeverity::Info);
        assert_eq!(d.kernel.as_deref(), Some("reduce_kernel"));
        assert!(d.message.contains("32 registers"), "msg: {}", d.message);
        assert!(d.line.is_none());
    }

    #[test]
    fn parse_fatal_no_kernel() {
        let line = "ptxas fatal   : Unresolved extern function 'missing_func'";
        let d = parse_ptxas_line(line).expect("should parse");
        assert_eq!(d.severity, JitSeverity::Fatal);
        assert!(d.kernel.is_none());
        assert!(d.message.contains("Unresolved"), "msg: {}", d.message);
    }

    #[test]
    fn parse_error_no_kernel_no_line() {
        let line = "ptxas error : syntax error near token ';'";
        let d = parse_ptxas_line(line).expect("should parse");
        assert_eq!(d.severity, JitSeverity::Error);
        assert!(d.kernel.is_none());
        assert!(d.line.is_none());
        assert!(d.message.contains("syntax error"), "msg: {}", d.message);
    }

    // ── JitLog helpers ────────────────────────────────────────────────────────

    #[test]
    fn jitlog_is_empty_for_default() {
        let log = JitLog::default();
        assert!(log.is_empty());
        assert!(!log.has_errors());
    }

    #[test]
    fn jitlog_has_errors_when_error_buf_nonempty() {
        let log = JitLog {
            info: String::new(),
            error: "ptxas error : something went wrong".to_string(),
        };
        assert!(log.has_errors());
        assert!(!log.is_empty());
    }

    #[test]
    fn jitlog_parse_diagnostics_multiline() {
        let log = JitLog {
            error: concat!(
                "ptxas error   : 'k1', line 5; error   : bad opcode\n",
                "ptxas warning : 'k1', line 8; warning : slow path\n",
            )
            .to_string(),
            info: "ptxas info    : 'k1' used 8 registers, 0 bytes smem\n".to_string(),
        };
        let diags = log.parse_diagnostics();
        assert_eq!(diags.len(), 3);
        assert_eq!(diags[0].severity, JitSeverity::Error);
        assert_eq!(diags[1].severity, JitSeverity::Warning);
        assert_eq!(diags[2].severity, JitSeverity::Info);
    }

    #[test]
    fn jitlog_errors_filter() {
        let log = JitLog {
            error: concat!(
                "ptxas error   : 'k', line 1; error : bad\n",
                "ptxas warning : 'k', line 2; warning : slow\n",
            )
            .to_string(),
            info: "ptxas info    : 'k' used 4 registers\n".to_string(),
        };
        let errs = log.errors();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].severity, JitSeverity::Error);
    }

    #[test]
    fn jitlog_warnings_filter() {
        let log = JitLog {
            error: "ptxas warning : 'k', line 3; warning : something slow\n".to_string(),
            info: String::new(),
        };
        let warns = log.warnings();
        assert_eq!(warns.len(), 1);
        assert_eq!(warns[0].severity, JitSeverity::Warning);
        assert_eq!(warns[0].line, Some(3));
    }

    // ── buf_to_string ─────────────────────────────────────────────────────────

    #[test]
    fn buf_to_string_null_terminated() {
        let mut buf = b"hello\0\0\0".to_vec();
        buf.extend_from_slice(&[0u8; 100]);
        assert_eq!(buf_to_string(&buf), "hello");
    }

    #[test]
    fn buf_to_string_empty() {
        assert_eq!(buf_to_string(&[0u8; 10]), "");
    }

    #[test]
    fn buf_to_string_no_null() {
        let buf = b"abc".to_vec();
        assert_eq!(buf_to_string(&buf), "abc");
    }

    // ── JitSeverity Display ───────────────────────────────────────────────────

    #[test]
    fn jit_severity_display() {
        assert_eq!(JitSeverity::Fatal.to_string(), "fatal");
        assert_eq!(JitSeverity::Error.to_string(), "error");
        assert_eq!(JitSeverity::Warning.to_string(), "warning");
        assert_eq!(JitSeverity::Info.to_string(), "info");
    }
}
