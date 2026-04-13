//! Error types for PTX code generation.
//!
//! Provides the [`PtxGenError`] enum which covers all failure modes
//! encountered during PTX kernel construction and text emission.

/// Errors that can occur during PTX code generation.
///
/// This enum covers validation failures during kernel building, register
/// allocation issues, missing required fields, and I/O or formatting errors
/// encountered while emitting PTX text.
#[derive(Debug, thiserror::Error)]
pub enum PtxGenError {
    /// General PTX generation failure with a descriptive message.
    #[error("PTX generation failed: {0}")]
    GenerationFailed(String),

    /// An invalid or unsupported PTX type was encountered.
    #[error("invalid PTX type: {0}")]
    InvalidType(String),

    /// A register allocation error (e.g. exceeding limits).
    #[error("register allocation failed: {0}")]
    RegisterError(String),

    /// The kernel builder was finalized without a body function.
    #[error("missing body function")]
    MissingBody,

    /// A `std::fmt::Write` formatting error during PTX text emission.
    #[error("format error: {0}")]
    FormatError(#[from] std::fmt::Error),

    /// An I/O error during PTX text emission or file writing.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// A required kernel parameter was referenced but not declared.
    #[error("unknown parameter: {0}")]
    UnknownParam(String),

    /// The target architecture does not support the requested feature.
    #[error("unsupported on {arch}: {feature}")]
    UnsupportedFeature {
        /// The target architecture name.
        arch: String,
        /// The feature that is not supported.
        feature: String,
    },
}
