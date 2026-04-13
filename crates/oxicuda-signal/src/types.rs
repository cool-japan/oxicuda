//! Core types shared across the signal processing subsystems.

use num_complex::Complex;

/// Floating-point precision for signal operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalPrecision {
    /// Single-precision (f32).
    F32,
    /// Double-precision (f64).
    F64,
}

impl SignalPrecision {
    /// Returns the byte size of the scalar type.
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// PTX type string for load/store instructions.
    #[must_use]
    pub const fn ptx_type(self) -> &'static str {
        match self {
            Self::F32 => ".f32",
            Self::F64 => ".f64",
        }
    }
}

/// Transform direction (forward / inverse).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformDirection {
    /// Forward transform (analysis).
    Forward,
    /// Inverse transform (synthesis).
    Inverse,
}

/// Window function types for time-frequency analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Rectangular (no windowing).
    Rectangular,
    /// Hann (raised cosine).
    Hann,
    /// Hamming (raised cosine with pedestal).
    Hamming,
    /// Blackman (3-term cosine).
    Blackman,
    /// Blackman-Harris (4-term cosine, excellent sidelobe attenuation).
    BlackmanHarris,
    /// Kaiser with shape parameter β.
    Kaiser {
        /// Kaiser window β shape parameter; typical range 0 (rectangular) – 14.
        beta: f64,
    },
    /// Gaussian with standard deviation σ (normalised to window length).
    Gaussian {
        /// Standard deviation normalised to window half-length; typical range 0.1–0.5.
        sigma: f64,
    },
    /// Bartlett (triangular with zero endpoints).
    Bartlett,
    /// Flat-top (minimal amplitude ripple; useful for calibration).
    FlatTop,
    /// Dolph-Chebyshev with sidelobe attenuation in dB.
    DolphChebyshev {
        /// Desired sidelobe attenuation in dB (positive); typical range 40–200.
        attenuation_db: f64,
    },
}

/// Padding mode for convolution / correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    /// Zero-pad to avoid circular wrap-around.
    Zero,
    /// Circular (periodic) boundary.
    Circular,
    /// Reflect at boundaries (mirror padding).
    Reflect,
    /// Replicate border values.
    Replicate,
}

/// Morphological structuring-element shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuringElement {
    /// Rectangular (full) structuring element.
    Rectangle {
        /// Kernel height.
        height: u32,
        /// Kernel width.
        width: u32,
    },
    /// Elliptical (disk) structuring element.
    Ellipse {
        /// Kernel height.
        height: u32,
        /// Kernel width.
        width: u32,
    },
    /// Cross-shaped structuring element (3×3 cross).
    Cross {
        /// Arm length in each direction.
        radius: u32,
    },
}

/// Normalisation mode for the inverse DCT / DWT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMode {
    /// No normalisation (unnormalized).
    None,
    /// Orthonormal normalisation (1/√N factor).
    Orthonormal,
    /// Forward scaling by 1/N.
    Forward,
}

/// Complex sample type alias (f32 precision).
pub type Complex32 = Complex<f32>;
/// Complex sample type alias (f64 precision).
pub type Complex64 = Complex<f64>;

/// Wavelet family identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletFamily {
    /// Haar wavelet (db1 — simplest orthonormal wavelet).
    Haar,
    /// Daubechies wavelet of order N (vanishing moments).
    Daubechies(u8),
    /// Symlet of order N (near-symmetric Daubechies).
    Symlet(u8),
    /// Coiflet of order N.
    Coiflet(u8),
    /// Biorthogonal wavelet (decomposition order, reconstruction order).
    Biorthogonal(u8, u8),
}

impl WaveletFamily {
    /// Filter length (number of coefficients in the low-pass decomposition filter).
    #[must_use]
    pub const fn filter_len(self) -> usize {
        match self {
            Self::Haar => 2,
            Self::Daubechies(n) | Self::Symlet(n) => 2 * n as usize,
            Self::Coiflet(n) => 6 * n as usize,
            Self::Biorthogonal(d, _) => 2 * d as usize + 2,
        }
    }
}
