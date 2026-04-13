//! Metal compute pipeline wrapper.
//!
//! A [`MetalComputePipeline`] compiles MSL source into a
//! `metal::ComputePipelineState` and owns the associated `metal::CommandQueue`.
//! On non-macOS platforms every constructor returns
//! [`MetalError::UnsupportedPlatform`].

use crate::{
    device::MetalDevice,
    error::{MetalError, MetalResult},
};

// ─── MetalComputePipeline ─────────────────────────────────────────────────────

/// A compiled Metal compute pipeline together with its command queue.
///
/// Created by compiling an MSL source string through
/// [`MetalComputePipeline::new`].  The pipeline state and command queue are
/// kept together so that callers can dispatch work without needing to manage
/// them separately.
pub struct MetalComputePipeline {
    /// The compiled pipeline state — only present on macOS.
    /// Kept alive for future kernel dispatch wiring.
    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    pub(crate) pipeline_state: metal::ComputePipelineState,
    /// The command queue used to create command buffers — only present on macOS.
    /// Kept alive for future kernel dispatch wiring.
    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    pub(crate) command_queue: metal::CommandQueue,
    /// The MSL entry-point function name (kept for diagnostics).
    function_name: String,
}

impl MetalComputePipeline {
    /// Compile `msl_source` and look up `function_name` inside the resulting
    /// library, then create a compute pipeline state.
    ///
    /// Returns:
    /// * [`MetalError::ShaderCompilation`] if the MSL fails to compile.
    /// * [`MetalError::PipelineCreation`] if the PSO cannot be created.
    /// * [`MetalError::UnsupportedPlatform`] on non-macOS.
    pub fn new(device: &MetalDevice, msl_source: &str, function_name: &str) -> MetalResult<Self> {
        #[cfg(target_os = "macos")]
        {
            let opts = metal::CompileOptions::new();
            let library = device
                .device
                .new_library_with_source(msl_source, &opts)
                .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

            let function = library
                .get_function(function_name, None)
                .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

            let pipeline_state = device
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MetalError::PipelineCreation(e.to_string()))?;

            let command_queue = device.device.new_command_queue();

            Ok(Self {
                pipeline_state,
                command_queue,
                function_name: function_name.to_string(),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (device, msl_source, function_name);
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// The MSL function name this pipeline was compiled for.
    pub fn function_name(&self) -> &str {
        &self.function_name
    }
}

impl std::fmt::Debug for MetalComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalComputePipeline(fn={})", self.function_name)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    #[cfg(target_os = "macos")]
    fn try_device() -> Option<MetalDevice> {
        MetalDevice::new().ok()
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipeline_compile_valid_msl() {
        let Some(dev) = try_device() else {
            return;
        };
        let src = crate::msl::gemm_msl();
        let p = MetalComputePipeline::new(&dev, src, "gemm_f32")
            .expect("pipeline creation from valid MSL should succeed");
        assert_eq!(p.function_name(), "gemm_f32");
        let dbg = format!("{p:?}");
        assert!(dbg.contains("gemm_f32"));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipeline_bad_msl_returns_shader_error() {
        let Some(dev) = try_device() else {
            return;
        };
        let bad_src = "this is not valid MSL !!!";
        let err = MetalComputePipeline::new(&dev, bad_src, "nope").unwrap_err();
        assert!(matches!(err, MetalError::ShaderCompilation(_)));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipeline_missing_function_returns_error() {
        let Some(dev) = try_device() else {
            return;
        };
        let src = crate::msl::gemm_msl();
        let err = MetalComputePipeline::new(&dev, src, "nonexistent_function").unwrap_err();
        assert!(matches!(err, MetalError::ShaderCompilation(_)));
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn pipeline_unsupported_on_non_macos() {
        // On non-macOS we can't even construct a MetalDevice, so just verify
        // the UnsupportedPlatform error is what MetalDevice returns.
        let result = MetalDevice::new();
        assert!(matches!(result, Err(MetalError::UnsupportedPlatform)));
    }
}
