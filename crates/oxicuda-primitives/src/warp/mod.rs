//! Warp-level parallel primitives using PTX shuffle instructions.
//!
//! Warp primitives operate within a single 32-thread warp without shared
//! memory, relying entirely on register-to-register shuffle operations
//! (`shfl.sync`).  They are the building block for block- and device-level
//! primitives.
//!
//! # Modules
//!
//! * [`reduce`] — warp-wide reduction (sum, min, max, product, …)
//! * [`scan`] — warp-wide prefix scan (inclusive / exclusive)

pub mod reduce;
pub mod scan;

pub use reduce::{WarpReduceConfig, WarpReduceTemplate};
pub use scan::{ScanKind, WarpScanConfig, WarpScanTemplate};
