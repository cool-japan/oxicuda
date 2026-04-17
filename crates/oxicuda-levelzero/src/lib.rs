//! OxiCUDA Level Zero backend — GPU compute via Intel oneAPI/Level Zero.
//!
//! # Platform Support
//!
//! | Platform | Status |
//! |----------|--------|
//! | Linux (Intel GPU) | Full support via libze_loader.so |
//! | Windows (Intel GPU) | Full support via ze_loader.dll |
//! | macOS | Not supported (`UnsupportedPlatform`) |

pub mod backend;
pub mod device;
pub mod error;
pub mod memory;
pub mod multi_tile;
pub mod spirv;
pub mod spirv_nn;
pub mod spirv_subgroup;
pub mod spirv_xmx;

pub use backend::LevelZeroBackend;
pub use error::{LevelZeroError, LevelZeroResult};
