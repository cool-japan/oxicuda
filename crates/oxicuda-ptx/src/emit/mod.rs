//! PTX text emission and validation.
//!
//! This module handles the final stage of the PTX pipeline: converting the
//! in-memory IR structures ([`PtxModule`], [`PtxFunction`]) into textual PTX
//! assembly, and performing basic validation on the generated output.
//!
//! - [`crate::emit::printer`]: Converts IR structures to PTX text
//! - [`crate::emit::validator`]: Basic PTX validation checks
//!
//! [`PtxModule`]: crate::ir::PtxModule
//! [`PtxFunction`]: crate::ir::PtxFunction

pub mod printer;
pub mod validator;
