//! Error types for the Vulkan compute backend.

use oxicuda_backend::BackendError;

/// Errors that can occur in the Vulkan backend.
#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    /// Vulkan shared library could not be loaded (e.g. no driver installed).
    #[error("Vulkan library not found: {0}")]
    LibraryNotFound(String),

    /// A Vulkan API call returned a non-success result code.
    #[error("Vulkan error code {0}: {1}")]
    VkError(i32, String),

    /// No physical device with a compute queue was found.
    #[error("no suitable GPU found")]
    NoSuitableDevice,

    /// Device memory allocation failed.
    #[error("out of device memory")]
    OutOfMemory,

    /// SPIR-V shader creation or validation failed.
    #[error("SPIR-V shader error: {0}")]
    ShaderError(String),

    /// Compute pipeline creation failed.
    #[error("pipeline creation failed: {0}")]
    PipelineError(String),

    /// An operation was attempted before `init()` was called.
    #[error("not initialized")]
    NotInitialized,

    /// The requested operation is not yet supported.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// An invalid argument was passed to an operation.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A command buffer operation failed.
    #[error("command buffer error: {0}")]
    CommandBufferError(String),

    /// Memory mapping failed.
    #[error("memory mapping failed: {0}")]
    MemoryMapError(String),
}

/// Convenience `Result` alias for Vulkan backend operations.
pub type VulkanResult<T> = Result<T, VulkanError>;

impl From<VulkanError> for BackendError {
    fn from(e: VulkanError) -> Self {
        match e {
            VulkanError::LibraryNotFound(s) => BackendError::DeviceError(s),
            VulkanError::VkError(code, msg) => {
                BackendError::DeviceError(format!("vk({code}): {msg}"))
            }
            VulkanError::NoSuitableDevice => {
                BackendError::DeviceError("no suitable Vulkan GPU found".into())
            }
            VulkanError::OutOfMemory => BackendError::OutOfMemory,
            VulkanError::ShaderError(s) => BackendError::DeviceError(format!("shader: {s}")),
            VulkanError::PipelineError(s) => BackendError::DeviceError(format!("pipeline: {s}")),
            VulkanError::NotInitialized => BackendError::NotInitialized,
            VulkanError::Unsupported(s) => BackendError::Unsupported(s),
            VulkanError::InvalidArgument(s) => BackendError::InvalidArgument(s),
            VulkanError::CommandBufferError(s) => {
                BackendError::DeviceError(format!("cmd_buf: {s}"))
            }
            VulkanError::MemoryMapError(s) => BackendError::DeviceError(format!("mmap: {s}")),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_error_display() {
        assert_eq!(
            VulkanError::LibraryNotFound("no libvulkan".into()).to_string(),
            "Vulkan library not found: no libvulkan"
        );
        assert_eq!(
            VulkanError::VkError(-4, "out of host memory".into()).to_string(),
            "Vulkan error code -4: out of host memory"
        );
        assert_eq!(
            VulkanError::NoSuitableDevice.to_string(),
            "no suitable GPU found"
        );
        assert_eq!(VulkanError::OutOfMemory.to_string(), "out of device memory");
        assert_eq!(
            VulkanError::ShaderError("bad spv".into()).to_string(),
            "SPIR-V shader error: bad spv"
        );
        assert_eq!(
            VulkanError::PipelineError("layout".into()).to_string(),
            "pipeline creation failed: layout"
        );
        assert_eq!(VulkanError::NotInitialized.to_string(), "not initialized");
        assert_eq!(
            VulkanError::Unsupported("atomics".into()).to_string(),
            "unsupported: atomics"
        );
        assert_eq!(
            VulkanError::InvalidArgument("n==0".into()).to_string(),
            "invalid argument: n==0"
        );
        assert_eq!(
            VulkanError::CommandBufferError("alloc".into()).to_string(),
            "command buffer error: alloc"
        );
        assert_eq!(
            VulkanError::MemoryMapError("host-visible".into()).to_string(),
            "memory mapping failed: host-visible"
        );
    }

    #[test]
    fn backend_error_from_vulkan_error() {
        assert_eq!(
            BackendError::from(VulkanError::OutOfMemory),
            BackendError::OutOfMemory
        );
        assert_eq!(
            BackendError::from(VulkanError::NotInitialized),
            BackendError::NotInitialized
        );
        assert!(matches!(
            BackendError::from(VulkanError::NoSuitableDevice),
            BackendError::DeviceError(_)
        ));
        assert!(matches!(
            BackendError::from(VulkanError::LibraryNotFound("x".into())),
            BackendError::DeviceError(_)
        ));
        assert!(matches!(
            BackendError::from(VulkanError::Unsupported("x".into())),
            BackendError::Unsupported(_)
        ));
        assert!(matches!(
            BackendError::from(VulkanError::InvalidArgument("x".into())),
            BackendError::InvalidArgument(_)
        ));
    }
}
