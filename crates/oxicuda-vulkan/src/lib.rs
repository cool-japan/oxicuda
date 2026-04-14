//! # oxicuda-vulkan
//!
//! Vulkan compute backend for the OxiCUDA ecosystem.
//!
//! This crate implements the [`ComputeBackend`] trait from `oxicuda-backend`
//! using the Vulkan GPU API through the [`ash`] crate's runtime loader.
//! No compile-time dependency on `libvulkan` is required: the library is
//! loaded dynamically at runtime via `ash::Entry::load`.
//!
//! # Platform support
//!
//! | Platform | Status |
//! |----------|--------|
//! | Linux + NVIDIA/AMD/Intel | Full (with Vulkan driver 1.2+) |
//! | Windows + discrete GPU  | Full (with Vulkan driver 1.2+) |
//! | macOS                   | `init()` returns `Err` (no native Vulkan) |
//!
//! # Quick start
//!
//! ```no_run
//! use oxicuda_vulkan::VulkanBackend;
//! use oxicuda_backend::ComputeBackend;
//!
//! let mut backend = VulkanBackend::new();
//! match backend.init() {
//!     Ok(()) => println!("Vulkan backend ready"),
//!     Err(e) => println!("Vulkan not available: {e}"),
//! }
//! ```
//!
//! [`ComputeBackend`]: oxicuda_backend::ComputeBackend

pub mod async_compute;
pub mod backend;
pub mod command;
pub mod device;
pub mod error;
pub mod memory;
pub mod pipeline;
pub mod spirv;

pub use async_compute::{AsyncComputeManager, VulkanFence, VulkanSemaphore};
pub use backend::VulkanBackend;
pub use error::{VulkanError, VulkanResult};
