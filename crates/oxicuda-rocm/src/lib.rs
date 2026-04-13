//! OxiCUDA ROCm backend — GPU compute via AMD HIP runtime.
//!
//! # Platform Support
//!
//! | Platform | Status |
//! |----------|--------|
//! | Linux (AMD GPU) | Full support via libamdhip64.so |
//! | Windows | Not supported (`UnsupportedPlatform`) |
//! | macOS | Not supported (`UnsupportedPlatform`) |

pub mod backend;
pub mod device;
pub mod error;
pub mod memory;

pub use backend::RocmBackend;
pub use error::{RocmError, RocmResult};
