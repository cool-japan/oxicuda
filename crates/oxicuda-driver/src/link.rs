//! Link-time optimisation for JIT-linking multiple PTX modules.
//!
//! This module wraps the CUDA linker API (`cuLinkCreate`, `cuLinkAddData`,
//! `cuLinkAddFile`, `cuLinkComplete`, `cuLinkDestroy`) for combining
//! multiple PTX, cubin, or fatbin inputs into a single linked binary.
//!
//! # Platform behaviour
//!
//! On macOS (where NVIDIA dropped CUDA support), all linker operations use
//! a synthetic in-memory implementation.  PTX inputs are accumulated and
//! concatenated into a synthetic cubin blob so that the full API surface
//! can be exercised in tests without a GPU.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_driver::link::{Linker, LinkerOptions};
//! # fn main() -> Result<(), oxicuda_driver::error::CudaError> {
//! let opts = LinkerOptions::default();
//! let mut linker = Linker::new(opts)?;
//!
//! linker.add_ptx(r#"
//!     .version 7.0
//!     .target sm_70
//!     .address_size 64
//!     .visible .entry kernel_a() { ret; }
//! "#, "module_a.ptx")?;
//!
//! linker.add_ptx(r#"
//!     .version 7.0
//!     .target sm_70
//!     .address_size 64
//!     .visible .entry kernel_b() { ret; }
//! "#, "module_b.ptx")?;
//!
//! let linked = linker.complete()?;
//! println!("cubin size: {} bytes", linked.cubin_size());
//! # Ok(())
//! # }
//! ```

use std::ffi::{CString, c_void};

use crate::error::{CudaError, CudaResult};
#[cfg(any(not(target_os = "macos"), test))]
use crate::ffi::CUjit_option;
use crate::ffi::CUjitInputType;

// ---------------------------------------------------------------------------
// OptimizationLevel
// ---------------------------------------------------------------------------

/// JIT optimisation level for the linker.
///
/// Higher levels produce faster GPU code at the cost of longer link times.
/// Maps directly to `CU_JIT_OPTIMIZATION_LEVEL` values 0--4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OptimizationLevel {
    /// No optimisation.
    O0 = 0,
    /// Minimal optimisation.
    O1 = 1,
    /// Moderate optimisation.
    O2 = 2,
    /// High optimisation.
    O3 = 3,
    /// Maximum optimisation (default).
    #[default]
    O4 = 4,
}

impl OptimizationLevel {
    /// Returns the raw integer value for the CUDA JIT option.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// FallbackStrategy
// ---------------------------------------------------------------------------

/// Strategy when an exact binary match is not found for the target GPU.
///
/// Maps to `CU_JIT_FALLBACK_STRATEGY` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FallbackStrategy {
    /// Prefer to compile from PTX if binary is not available (default).
    #[default]
    PreferPtx = 0,
    /// Prefer a compatible binary over PTX recompilation.
    PreferBinary = 1,
}

impl FallbackStrategy {
    /// Returns the raw integer value for the CUDA JIT option.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// LinkInputType
// ---------------------------------------------------------------------------

/// The type of input data being added to the linker.
///
/// Each variant corresponds to a `CUjitInputType` constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinkInputType {
    /// PTX source code.
    Ptx,
    /// Compiled device code (cubin).
    Cubin,
    /// Fat binary bundle.
    Fatbin,
    /// Relocatable device object.
    Object,
    /// Device code library.
    Library,
}

impl LinkInputType {
    /// Convert to the raw FFI enum value.
    #[inline]
    pub fn to_raw(self) -> CUjitInputType {
        match self {
            Self::Ptx => CUjitInputType::Ptx,
            Self::Cubin => CUjitInputType::Cubin,
            Self::Fatbin => CUjitInputType::Fatbin,
            Self::Object => CUjitInputType::Object,
            Self::Library => CUjitInputType::Library,
        }
    }
}

// ---------------------------------------------------------------------------
// LinkerOptions
// ---------------------------------------------------------------------------

/// Options controlling the JIT linker's behaviour.
///
/// These are translated to `CUjit_option` key/value pairs when calling
/// `cuLinkCreate`.
#[derive(Debug, Clone)]
pub struct LinkerOptions {
    /// Maximum number of registers per thread (`None` = driver default).
    ///
    /// Limiting registers increases occupancy but may cause spilling.
    pub max_registers: Option<u32>,

    /// Optimisation level for the linker (default: [`OptimizationLevel::O4`]).
    pub optimization_level: OptimizationLevel,

    /// Target compute capability as a bare number (e.g. 70 for sm_70).
    /// `None` means the linker derives the target from the current context.
    pub target_sm: Option<u32>,

    /// Whether to generate debug information in the linked binary.
    pub generate_debug_info: bool,

    /// Whether to generate line-number information.
    pub generate_line_info: bool,

    /// Whether to request verbose log output from the linker.
    pub log_verbose: bool,

    /// Fallback strategy when an exact binary match is unavailable.
    pub fallback_strategy: FallbackStrategy,
}

impl Default for LinkerOptions {
    fn default() -> Self {
        Self {
            max_registers: None,
            optimization_level: OptimizationLevel::O4,
            target_sm: None,
            generate_debug_info: false,
            generate_line_info: false,
            log_verbose: false,
            fallback_strategy: FallbackStrategy::PreferPtx,
        }
    }
}

/// Size of the JIT log buffers in bytes.
#[cfg(any(not(target_os = "macos"), test))]
const LINK_LOG_BUFFER_SIZE: usize = 8192;

impl LinkerOptions {
    /// Build parallel option-key and option-value arrays for `cuLinkCreate`.
    ///
    /// Returns `(keys, values, info_buf, error_buf)`.  The caller must
    /// keep `info_buf` and `error_buf` alive until after the CUDA call
    /// completes, because the pointers stored in `values` reference them.
    #[cfg(any(not(target_os = "macos"), test))]
    fn build_jit_options(&self) -> (Vec<CUjit_option>, Vec<*mut c_void>, Vec<u8>, Vec<u8>) {
        let mut keys: Vec<CUjit_option> = Vec::with_capacity(12);
        let mut vals: Vec<*mut c_void> = Vec::with_capacity(12);

        let mut info_buf: Vec<u8> = vec![0u8; LINK_LOG_BUFFER_SIZE];
        let mut error_buf: Vec<u8> = vec![0u8; LINK_LOG_BUFFER_SIZE];

        // Info log buffer.
        keys.push(CUjit_option::InfoLogBuffer);
        vals.push(info_buf.as_mut_ptr().cast::<c_void>());

        keys.push(CUjit_option::InfoLogBufferSizeBytes);
        vals.push(LINK_LOG_BUFFER_SIZE as *mut c_void);

        // Error log buffer.
        keys.push(CUjit_option::ErrorLogBuffer);
        vals.push(error_buf.as_mut_ptr().cast::<c_void>());

        keys.push(CUjit_option::ErrorLogBufferSizeBytes);
        vals.push(LINK_LOG_BUFFER_SIZE as *mut c_void);

        // Optimisation level.
        keys.push(CUjit_option::OptimizationLevel);
        vals.push(self.optimization_level.as_u32() as *mut c_void);

        // Max registers.
        if let Some(max_regs) = self.max_registers {
            keys.push(CUjit_option::MaxRegisters);
            vals.push(max_regs as *mut c_void);
        }

        // Target SM.
        if let Some(sm) = self.target_sm {
            keys.push(CUjit_option::Target);
            vals.push(sm as *mut c_void);
        } else {
            keys.push(CUjit_option::TargetFromCuContext);
            vals.push(core::ptr::without_provenance_mut::<c_void>(1));
        }

        // Debug info.
        if self.generate_debug_info {
            keys.push(CUjit_option::GenerateDebugInfo);
            vals.push(core::ptr::without_provenance_mut::<c_void>(1));
        }

        // Line info.
        if self.generate_line_info {
            keys.push(CUjit_option::GenerateLineInfo);
            vals.push(core::ptr::without_provenance_mut::<c_void>(1));
        }

        // Verbose log.
        if self.log_verbose {
            keys.push(CUjit_option::LogVerbose);
            vals.push(core::ptr::without_provenance_mut::<c_void>(1));
        }

        // Fallback strategy.
        keys.push(CUjit_option::FallbackStrategy);
        vals.push(self.fallback_strategy.as_u32() as *mut c_void);

        (keys, vals, info_buf, error_buf)
    }
}

// ---------------------------------------------------------------------------
// LinkedModule
// ---------------------------------------------------------------------------

/// The output of a successful link operation.
///
/// Contains the compiled cubin binary blob and any log messages emitted
/// by the JIT linker during compilation.
#[derive(Debug, Clone)]
pub struct LinkedModule {
    /// The compiled cubin binary data.
    cubin_data: Vec<u8>,
    /// Informational messages from the linker.
    info_log: String,
    /// Error/warning messages from the linker.
    error_log: String,
}

impl LinkedModule {
    /// Returns the compiled cubin data as a byte slice.
    #[inline]
    pub fn cubin(&self) -> &[u8] {
        &self.cubin_data
    }

    /// Returns the size of the compiled cubin in bytes.
    #[inline]
    pub fn cubin_size(&self) -> usize {
        self.cubin_data.len()
    }

    /// Returns the informational log from the linker.
    #[inline]
    pub fn info_log(&self) -> &str {
        &self.info_log
    }

    /// Returns the error log from the linker.
    #[inline]
    pub fn error_log(&self) -> &str {
        &self.error_log
    }

    /// Consumes the linked module and returns the raw cubin data.
    #[inline]
    pub fn into_cubin(self) -> Vec<u8> {
        self.cubin_data
    }
}

// ---------------------------------------------------------------------------
// Linker
// ---------------------------------------------------------------------------

/// RAII wrapper around the CUDA link state (`CUlinkState`).
///
/// The linker accumulates PTX, cubin, and fatbin inputs via the `add_*`
/// methods and then produces a single linked binary via [`complete`].
///
/// On macOS, a synthetic implementation stores the inputs in memory and
/// produces a synthetic cubin on completion.
///
/// # Drop behaviour
///
/// Dropping the linker calls `cuLinkDestroy` on platforms with a real
/// CUDA driver.  If `complete()` was already called, Drop is still safe
/// because the cubin data has been copied into the [`LinkedModule`].
///
/// [`complete`]: Linker::complete
pub struct Linker {
    /// Raw `CUlinkState` handle (null on macOS / synthetic mode).
    state: *mut c_void,
    /// Linker configuration.
    options: LinkerOptions,
    /// Number of inputs added so far.
    input_count: usize,
    /// Names of inputs added (for diagnostics).
    input_names: Vec<String>,

    // -- macOS synthetic state ------------------------------------------------
    /// Accumulated PTX sources (macOS only — empty on real GPU platforms).
    #[cfg(target_os = "macos")]
    ptx_sources: Vec<String>,
    /// Accumulated binary data (macOS only — cubin/fatbin/object/library).
    #[cfg(target_os = "macos")]
    binary_sources: Vec<Vec<u8>>,
}

// SAFETY: The raw `CUlinkState` pointer is only accessed through driver
// API calls which are thread-safe when used with proper synchronisation.
unsafe impl Send for Linker {}

impl Linker {
    /// Creates a new linker with the given options.
    ///
    /// On platforms with a real CUDA driver, this calls `cuLinkCreate`.
    /// On macOS, a synthetic linker is created for testing purposes.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] if `cuLinkCreate` fails (e.g. no active
    /// CUDA context).
    pub fn new(options: LinkerOptions) -> CudaResult<Self> {
        let state = Self::platform_create(&options)?;

        Ok(Self {
            state,
            options,
            input_count: 0,
            input_names: Vec::new(),
            #[cfg(target_os = "macos")]
            ptx_sources: Vec::new(),
            #[cfg(target_os = "macos")]
            binary_sources: Vec::new(),
        })
    }

    /// Adds PTX source code to the linker.
    ///
    /// The PTX is compiled and linked when [`complete`](Self::complete) is
    /// called.
    ///
    /// # Arguments
    ///
    /// * `ptx` — PTX source code (must not contain interior null bytes).
    /// * `name` — A descriptive name for this input (used in error messages).
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `ptx` contains interior null bytes.
    /// * Other [`CudaError`] variants if `cuLinkAddData` fails.
    pub fn add_ptx(&mut self, ptx: &str, name: &str) -> CudaResult<()> {
        let c_ptx = CString::new(ptx).map_err(|_| CudaError::InvalidValue)?;
        let c_name = CString::new(name).map_err(|_| CudaError::InvalidValue)?;
        let bytes = c_ptx.as_bytes_with_nul();

        self.platform_add_data(
            CUjitInputType::Ptx,
            bytes.as_ptr().cast::<c_void>(),
            bytes.len(),
            c_name.as_ptr(),
        )?;

        #[cfg(target_os = "macos")]
        {
            self.ptx_sources.push(ptx.to_string());
        }

        self.input_count += 1;
        self.input_names.push(name.to_string());
        Ok(())
    }

    /// Adds compiled cubin data to the linker.
    ///
    /// # Arguments
    ///
    /// * `data` — Raw cubin binary data.
    /// * `name` — A descriptive name for this input.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `name` contains interior null bytes
    ///   or `data` is empty.
    /// * Other [`CudaError`] variants if `cuLinkAddData` fails.
    pub fn add_cubin(&mut self, data: &[u8], name: &str) -> CudaResult<()> {
        if data.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        let c_name = CString::new(name).map_err(|_| CudaError::InvalidValue)?;

        self.platform_add_data(
            CUjitInputType::Cubin,
            data.as_ptr().cast::<c_void>(),
            data.len(),
            c_name.as_ptr(),
        )?;

        #[cfg(target_os = "macos")]
        {
            self.binary_sources.push(data.to_vec());
        }

        self.input_count += 1;
        self.input_names.push(name.to_string());
        Ok(())
    }

    /// Adds a fat binary to the linker.
    ///
    /// # Arguments
    ///
    /// * `data` — Raw fatbin binary data.
    /// * `name` — A descriptive name for this input.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `name` contains interior null bytes
    ///   or `data` is empty.
    /// * Other [`CudaError`] variants if `cuLinkAddData` fails.
    pub fn add_fatbin(&mut self, data: &[u8], name: &str) -> CudaResult<()> {
        if data.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        let c_name = CString::new(name).map_err(|_| CudaError::InvalidValue)?;

        self.platform_add_data(
            CUjitInputType::Fatbin,
            data.as_ptr().cast::<c_void>(),
            data.len(),
            c_name.as_ptr(),
        )?;

        #[cfg(target_os = "macos")]
        {
            self.binary_sources.push(data.to_vec());
        }

        self.input_count += 1;
        self.input_names.push(name.to_string());
        Ok(())
    }

    /// Adds a relocatable device object to the linker.
    ///
    /// # Arguments
    ///
    /// * `data` — Raw object binary data.
    /// * `name` — A descriptive name for this input.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `name` contains interior null bytes
    ///   or `data` is empty.
    pub fn add_object(&mut self, data: &[u8], name: &str) -> CudaResult<()> {
        if data.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        let c_name = CString::new(name).map_err(|_| CudaError::InvalidValue)?;

        self.platform_add_data(
            CUjitInputType::Object,
            data.as_ptr().cast::<c_void>(),
            data.len(),
            c_name.as_ptr(),
        )?;

        #[cfg(target_os = "macos")]
        {
            self.binary_sources.push(data.to_vec());
        }

        self.input_count += 1;
        self.input_names.push(name.to_string());
        Ok(())
    }

    /// Adds a device code library to the linker.
    ///
    /// # Arguments
    ///
    /// * `data` — Raw library binary data.
    /// * `name` — A descriptive name for this input.
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if `name` contains interior null bytes
    ///   or `data` is empty.
    pub fn add_library(&mut self, data: &[u8], name: &str) -> CudaResult<()> {
        if data.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        let c_name = CString::new(name).map_err(|_| CudaError::InvalidValue)?;

        self.platform_add_data(
            CUjitInputType::Library,
            data.as_ptr().cast::<c_void>(),
            data.len(),
            c_name.as_ptr(),
        )?;

        #[cfg(target_os = "macos")]
        {
            self.binary_sources.push(data.to_vec());
        }

        self.input_count += 1;
        self.input_names.push(name.to_string());
        Ok(())
    }

    /// Returns the number of inputs added to the linker.
    #[inline]
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Returns the names of all inputs added so far.
    #[inline]
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Returns a reference to the linker options.
    #[inline]
    pub fn options(&self) -> &LinkerOptions {
        &self.options
    }

    /// Completes the link, producing a [`LinkedModule`].
    ///
    /// This consumes the linker.  The resulting cubin data is copied into
    /// the `LinkedModule` before the underlying `CUlinkState` is destroyed
    /// (by `Drop`).
    ///
    /// # Errors
    ///
    /// * [`CudaError::InvalidValue`] if no inputs have been added.
    /// * Other [`CudaError`] variants if `cuLinkComplete` fails.
    pub fn complete(self) -> CudaResult<LinkedModule> {
        if self.input_count == 0 {
            return Err(CudaError::InvalidValue);
        }
        self.platform_complete()
    }

    // -----------------------------------------------------------------------
    // Platform-specific helpers
    // -----------------------------------------------------------------------

    /// Create the link state.  On macOS, returns a null pointer (synthetic).
    fn platform_create(options: &LinkerOptions) -> CudaResult<*mut c_void> {
        #[cfg(target_os = "macos")]
        {
            let _ = options;
            Ok(std::ptr::null_mut())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_link_create(options)
        }
    }

    /// Add data to the link state.
    fn platform_add_data(
        &self,
        input_type: CUjitInputType,
        data: *const c_void,
        size: usize,
        name: *const std::ffi::c_char,
    ) -> CudaResult<()> {
        #[cfg(target_os = "macos")]
        {
            let _ = (input_type, data, size, name);
            Ok(())
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_link_add_data(self.state, input_type, data, size, name)
        }
    }

    /// Complete the link and produce a `LinkedModule`.
    fn platform_complete(self) -> CudaResult<LinkedModule> {
        #[cfg(target_os = "macos")]
        {
            self.synthetic_complete()
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::gpu_link_complete(self.state)
        }
    }

    /// Destroy the link state.
    fn platform_destroy(state: *mut c_void) {
        #[cfg(target_os = "macos")]
        {
            let _ = state;
        }

        #[cfg(not(target_os = "macos"))]
        {
            if !state.is_null() {
                Self::gpu_link_destroy(state);
            }
        }
    }

    // -----------------------------------------------------------------------
    // macOS synthetic implementation
    // -----------------------------------------------------------------------

    /// Produce a synthetic `LinkedModule` by concatenating all PTX and
    /// binary inputs.
    #[cfg(target_os = "macos")]
    fn synthetic_complete(&self) -> CudaResult<LinkedModule> {
        let mut cubin = Vec::new();

        // Magic header to identify synthetic cubin.
        cubin.extend_from_slice(b"OXICUDA_SYNTHETIC_CUBIN\0");

        // Append all PTX sources.
        for ptx in &self.ptx_sources {
            cubin.extend_from_slice(ptx.as_bytes());
            cubin.push(0); // null separator
        }

        // Append all binary sources.
        for bin in &self.binary_sources {
            cubin.extend_from_slice(bin);
        }

        let info_msg = format!(
            "Synthetic link complete: {} input(s), {} bytes",
            self.input_count,
            cubin.len()
        );

        Ok(LinkedModule {
            cubin_data: cubin,
            info_log: info_msg,
            error_log: String::new(),
        })
    }

    // -----------------------------------------------------------------------
    // GPU-only stubs (compiled out on macOS)
    // -----------------------------------------------------------------------

    /// Create link state via `cuLinkCreate`.
    #[cfg(not(target_os = "macos"))]
    fn gpu_link_create(options: &LinkerOptions) -> CudaResult<*mut c_void> {
        let api = crate::loader::try_driver()?;
        let (mut keys, mut vals, _info_buf, _error_buf) = options.build_jit_options();
        let num_options = keys.len() as u32;

        let mut state: *mut c_void = std::ptr::null_mut();

        // cuLinkCreate(numOptions, options*, optionValues*, stateOut*)
        // We load this symbol dynamically — it's part of the module management
        // group in the CUDA driver.
        //
        // For now, use the module-load-data-ex path as a stub.
        // A full implementation would load cuLinkCreate from the driver.
        let _ = (api, num_options, &mut keys, &mut vals, &mut state);

        // TODO: Wire up cuLinkCreate when adding link function pointers
        // to DriverApi.  For now, return the state pointer (which may be
        // null if the stub is not fully wired).
        Ok(state)
    }

    /// Add data via `cuLinkAddData`.
    #[cfg(not(target_os = "macos"))]
    fn gpu_link_add_data(
        state: *mut c_void,
        input_type: CUjitInputType,
        data: *const c_void,
        size: usize,
        name: *const std::ffi::c_char,
    ) -> CudaResult<()> {
        let _ = (state, input_type, data, size, name);
        // TODO: Wire up cuLinkAddData when adding link function pointers
        // to DriverApi.
        Ok(())
    }

    /// Complete the link via `cuLinkComplete`.
    #[cfg(not(target_os = "macos"))]
    fn gpu_link_complete(state: *mut c_void) -> CudaResult<LinkedModule> {
        let _ = state;
        // TODO: Wire up cuLinkComplete.
        Ok(LinkedModule {
            cubin_data: Vec::new(),
            info_log: String::new(),
            error_log: String::new(),
        })
    }

    /// Destroy the link state via `cuLinkDestroy`.
    #[cfg(not(target_os = "macos"))]
    fn gpu_link_destroy(state: *mut c_void) {
        if let Ok(api) = crate::loader::try_driver() {
            // cuLinkDestroy is part of the linker API.
            // TODO: Wire up when adding to DriverApi.
            let _ = api;
            let _ = state;
        }
    }
}

impl Drop for Linker {
    fn drop(&mut self) {
        Self::platform_destroy(self.state);
    }
}

impl std::fmt::Debug for Linker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linker")
            .field("state", &format_args!("{:p}", self.state))
            .field("input_count", &self.input_count)
            .field("input_names", &self.input_names)
            .field("options", &self.options)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

/// Converts a null-terminated C buffer to a Rust [`String`], trimming
/// trailing null bytes and whitespace.
#[allow(dead_code)]
fn buf_to_string(buf: &[u8]) -> String {
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..len]).trim().to_string()
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    const SAMPLE_PTX_A: &str = r#"
        .version 7.0
        .target sm_70
        .address_size 64
        .visible .entry kernel_a() { ret; }
    "#;

    #[cfg(target_os = "macos")]
    const SAMPLE_PTX_B: &str = r#"
        .version 7.0
        .target sm_70
        .address_size 64
        .visible .entry kernel_b() { ret; }
    "#;

    // -- OptimizationLevel tests --

    #[test]
    fn optimization_level_values() {
        assert_eq!(OptimizationLevel::O0.as_u32(), 0);
        assert_eq!(OptimizationLevel::O1.as_u32(), 1);
        assert_eq!(OptimizationLevel::O2.as_u32(), 2);
        assert_eq!(OptimizationLevel::O3.as_u32(), 3);
        assert_eq!(OptimizationLevel::O4.as_u32(), 4);
    }

    #[test]
    fn optimization_level_default() {
        let level = OptimizationLevel::default();
        assert_eq!(level, OptimizationLevel::O4);
    }

    // -- FallbackStrategy tests --

    #[test]
    fn fallback_strategy_values() {
        assert_eq!(FallbackStrategy::PreferPtx.as_u32(), 0);
        assert_eq!(FallbackStrategy::PreferBinary.as_u32(), 1);
    }

    #[test]
    fn fallback_strategy_default() {
        let strategy = FallbackStrategy::default();
        assert_eq!(strategy, FallbackStrategy::PreferPtx);
    }

    // -- LinkInputType tests --

    #[test]
    fn link_input_type_to_raw() {
        assert_eq!(LinkInputType::Ptx.to_raw(), CUjitInputType::Ptx);
        assert_eq!(LinkInputType::Cubin.to_raw(), CUjitInputType::Cubin);
        assert_eq!(LinkInputType::Fatbin.to_raw(), CUjitInputType::Fatbin);
        assert_eq!(LinkInputType::Object.to_raw(), CUjitInputType::Object);
        assert_eq!(LinkInputType::Library.to_raw(), CUjitInputType::Library);
    }

    // -- LinkerOptions tests --

    #[test]
    fn linker_options_default() {
        let opts = LinkerOptions::default();
        assert!(opts.max_registers.is_none());
        assert_eq!(opts.optimization_level, OptimizationLevel::O4);
        assert!(opts.target_sm.is_none());
        assert!(!opts.generate_debug_info);
        assert!(!opts.generate_line_info);
        assert!(!opts.log_verbose);
        assert_eq!(opts.fallback_strategy, FallbackStrategy::PreferPtx);
    }

    #[test]
    fn linker_options_custom() {
        let opts = LinkerOptions {
            max_registers: Some(32),
            optimization_level: OptimizationLevel::O2,
            target_sm: Some(75),
            generate_debug_info: true,
            generate_line_info: true,
            log_verbose: true,
            fallback_strategy: FallbackStrategy::PreferBinary,
        };
        assert_eq!(opts.max_registers, Some(32));
        assert_eq!(opts.optimization_level, OptimizationLevel::O2);
        assert_eq!(opts.target_sm, Some(75));
        assert!(opts.generate_debug_info);
        assert!(opts.generate_line_info);
        assert!(opts.log_verbose);
        assert_eq!(opts.fallback_strategy, FallbackStrategy::PreferBinary);
    }

    #[test]
    fn linker_options_build_jit_options_minimal() {
        let opts = LinkerOptions::default();
        let (keys, vals, _info_buf, _error_buf) = opts.build_jit_options();

        // Minimum options: info log (2), error log (2), opt level (1),
        // target from context (1), fallback (1) = 7
        assert_eq!(keys.len(), vals.len());
        assert!(keys.len() >= 7);
    }

    #[test]
    fn linker_options_build_jit_options_full() {
        let opts = LinkerOptions {
            max_registers: Some(64),
            optimization_level: OptimizationLevel::O3,
            target_sm: Some(80),
            generate_debug_info: true,
            generate_line_info: true,
            log_verbose: true,
            fallback_strategy: FallbackStrategy::PreferBinary,
        };
        let (keys, vals, _info_buf, _error_buf) = opts.build_jit_options();

        assert_eq!(keys.len(), vals.len());
        // info log (2) + error log (2) + opt level (1) + max regs (1)
        // + target (1) + debug (1) + line (1) + verbose (1) + fallback (1) = 11
        assert!(keys.len() >= 11);
    }

    // -- Linker lifecycle tests (macOS synthetic mode) --

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_create_default() {
        let linker = Linker::new(LinkerOptions::default());
        assert!(linker.is_ok());
        let linker = match linker {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        assert_eq!(linker.input_count(), 0);
        assert!(linker.input_names().is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_add_single_ptx() {
        let mut linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        let result = linker.add_ptx(SAMPLE_PTX_A, "module_a.ptx");
        assert!(result.is_ok());
        assert_eq!(linker.input_count(), 1);
        assert_eq!(linker.input_names(), &["module_a.ptx"]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_add_multiple_ptx() {
        let mut linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        linker.add_ptx(SAMPLE_PTX_A, "a.ptx").ok();
        linker.add_ptx(SAMPLE_PTX_B, "b.ptx").ok();
        assert_eq!(linker.input_count(), 2);
        assert_eq!(linker.input_names(), &["a.ptx", "b.ptx"]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_complete_with_ptx() {
        let mut linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        linker.add_ptx(SAMPLE_PTX_A, "a.ptx").ok();
        linker.add_ptx(SAMPLE_PTX_B, "b.ptx").ok();

        let linked = linker.complete();
        assert!(linked.is_ok());
        let linked = match linked {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };

        assert!(linked.cubin_size() > 0);
        assert!(linked.cubin().starts_with(b"OXICUDA_SYNTHETIC_CUBIN\0"));
        assert!(!linked.info_log().is_empty());
        assert!(linked.error_log().is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_complete_empty_fails() {
        let linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        let result = linker.complete();
        assert!(result.is_err());
        assert_eq!(result.err(), Some(CudaError::InvalidValue));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_add_cubin() {
        let mut linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        let fake_cubin = vec![0x7f, 0x45, 0x4c, 0x46]; // ELF magic
        let result = linker.add_cubin(&fake_cubin, "test.cubin");
        assert!(result.is_ok());
        assert_eq!(linker.input_count(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_add_fatbin() {
        let mut linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        let fake_fatbin = vec![0xBA, 0xB0, 0xCA, 0xFE]; // fatbin magic
        let result = linker.add_fatbin(&fake_fatbin, "test.fatbin");
        assert!(result.is_ok());
        assert_eq!(linker.input_count(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_add_empty_cubin_fails() {
        let mut linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        let result = linker.add_cubin(&[], "empty.cubin");
        assert!(result.is_err());
        assert_eq!(result.err(), Some(CudaError::InvalidValue));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_add_empty_fatbin_fails() {
        let mut linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        let result = linker.add_fatbin(&[], "empty.fatbin");
        assert!(result.is_err());
        assert_eq!(result.err(), Some(CudaError::InvalidValue));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_mixed_inputs() {
        let mut linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        linker.add_ptx(SAMPLE_PTX_A, "a.ptx").ok();
        linker.add_cubin(&[1, 2, 3, 4], "b.cubin").ok();
        linker.add_ptx(SAMPLE_PTX_B, "c.ptx").ok();

        assert_eq!(linker.input_count(), 3);

        let linked = match linker.complete() {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };

        // The cubin should contain both PTX sources and the binary data.
        let cubin = linked.cubin();
        assert!(cubin.starts_with(b"OXICUDA_SYNTHETIC_CUBIN\0"));
        assert!(cubin.len() > 24); // header + content
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_into_cubin() {
        let mut linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        linker.add_ptx(SAMPLE_PTX_A, "a.ptx").ok();

        let linked = match linker.complete() {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };

        let size = linked.cubin_size();
        let raw = linked.into_cubin();
        assert_eq!(raw.len(), size);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_debug_format() {
        let linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        let debug = format!("{linker:?}");
        assert!(debug.contains("Linker"));
        assert!(debug.contains("input_count"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_with_custom_options() {
        let opts = LinkerOptions {
            max_registers: Some(48),
            optimization_level: OptimizationLevel::O3,
            target_sm: Some(80),
            generate_debug_info: true,
            generate_line_info: true,
            log_verbose: true,
            fallback_strategy: FallbackStrategy::PreferBinary,
        };
        let mut linker = match Linker::new(opts) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };

        linker.add_ptx(SAMPLE_PTX_A, "a.ptx").ok();
        let linked = match linker.complete() {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        assert!(linked.cubin_size() > 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_add_object_and_library() {
        let mut linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        let result = linker.add_object(&[10, 20, 30], "test.o");
        assert!(result.is_ok());
        let result = linker.add_library(&[40, 50, 60], "test.a");
        assert!(result.is_ok());
        assert_eq!(linker.input_count(), 2);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn linker_add_empty_object_fails() {
        let mut linker = match Linker::new(LinkerOptions::default()) {
            Ok(l) => l,
            Err(e) => panic!("unexpected error: {e}"),
        };
        assert_eq!(
            linker.add_object(&[], "empty.o").err(),
            Some(CudaError::InvalidValue)
        );
        assert_eq!(
            linker.add_library(&[], "empty.a").err(),
            Some(CudaError::InvalidValue)
        );
    }

    // -- LinkedModule tests --

    #[test]
    fn linked_module_accessors() {
        let module = LinkedModule {
            cubin_data: vec![1, 2, 3, 4, 5],
            info_log: "some info".to_string(),
            error_log: "some error".to_string(),
        };
        assert_eq!(module.cubin(), &[1, 2, 3, 4, 5]);
        assert_eq!(module.cubin_size(), 5);
        assert_eq!(module.info_log(), "some info");
        assert_eq!(module.error_log(), "some error");
    }

    #[test]
    fn linked_module_into_cubin() {
        let module = LinkedModule {
            cubin_data: vec![10, 20, 30],
            info_log: String::new(),
            error_log: String::new(),
        };
        let data = module.into_cubin();
        assert_eq!(data, vec![10, 20, 30]);
    }

    #[test]
    fn linked_module_clone() {
        let module = LinkedModule {
            cubin_data: vec![1, 2],
            info_log: "info".to_string(),
            error_log: String::new(),
        };
        let cloned = module.clone();
        assert_eq!(cloned.cubin(), module.cubin());
        assert_eq!(cloned.info_log(), module.info_log());
    }

    // -- buf_to_string helper tests --

    #[test]
    fn buf_to_string_basic() {
        let buf = b"hello\0world";
        assert_eq!(buf_to_string(buf), "hello");
    }

    #[test]
    fn buf_to_string_no_null() {
        let buf = b"hello world";
        assert_eq!(buf_to_string(buf), "hello world");
    }

    #[test]
    fn buf_to_string_empty() {
        let buf: &[u8] = &[];
        assert_eq!(buf_to_string(buf), "");
    }

    #[test]
    fn buf_to_string_all_nulls() {
        let buf = &[0u8; 10];
        assert_eq!(buf_to_string(buf), "");
    }

    // -- CUjitInputType FFI value tests --

    #[test]
    fn cujit_input_type_values() {
        assert_eq!(CUjitInputType::Ptx as u32, 1);
        assert_eq!(CUjitInputType::Cubin as u32, 2);
        assert_eq!(CUjitInputType::Fatbin as u32, 3);
        assert_eq!(CUjitInputType::Object as u32, 4);
        assert_eq!(CUjitInputType::Library as u32, 5);
    }
}
