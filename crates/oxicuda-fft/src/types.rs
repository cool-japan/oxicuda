//! Core FFT types: complex numbers, transform descriptors, and precision modes.
#![allow(dead_code)]

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// Complex<T> — lightweight repr(C) complex number
// ---------------------------------------------------------------------------

/// A complex number with real and imaginary parts.
///
/// This is a minimal, `#[repr(C)]` compatible complex type used for FFT
/// data layout.  It intentionally avoids pulling in `num-complex` so that
/// kernel-level code can depend only on plain `Copy` types.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Complex<T: Copy> {
    /// Real component.
    pub re: T,
    /// Imaginary component.
    pub im: T,
}

// -- Construction helpers for f32 ------------------------------------------

impl Complex<f32> {
    /// Creates a new complex number.
    #[inline]
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// The additive identity `(0 + 0i)`.
    #[inline]
    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    /// The multiplicative identity `(1 + 0i)`.
    #[inline]
    pub fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    /// Complex conjugate `(re - im*i)`.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared magnitude `re^2 + im^2`.
    #[inline]
    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude (absolute value).
    #[inline]
    pub fn abs(self) -> f32 {
        self.norm_sqr().sqrt()
    }
}

// -- Construction helpers for f64 ------------------------------------------

impl Complex<f64> {
    /// Creates a new complex number.
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// The additive identity `(0 + 0i)`.
    #[inline]
    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    /// The multiplicative identity `(1 + 0i)`.
    #[inline]
    pub fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    /// Complex conjugate `(re - im*i)`.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared magnitude `re^2 + im^2`.
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude (absolute value).
    #[inline]
    pub fn abs(self) -> f64 {
        self.norm_sqr().sqrt()
    }
}

// -- Display ---------------------------------------------------------------

impl<T: Copy + fmt::Display> fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {}i)", self.re, self.im)
    }
}

// -- Arithmetic: Add -------------------------------------------------------

impl Add for Complex<f32> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Add for Complex<f64> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

// -- Arithmetic: Sub -------------------------------------------------------

impl Sub for Complex<f32> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Sub for Complex<f64> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

// -- Arithmetic: Mul (complex multiplication) ------------------------------

impl Mul for Complex<f32> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Mul for Complex<f64> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

// -- Arithmetic: Neg -------------------------------------------------------

impl Neg for Complex<f32> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Neg for Complex<f64> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

// ---------------------------------------------------------------------------
// FftType — transform category
// ---------------------------------------------------------------------------

/// Type of FFT transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FftType {
    /// Complex-to-complex transform.
    C2C,
    /// Real-to-complex (forward) transform.
    R2C,
    /// Complex-to-real (inverse) transform.
    C2R,
}

impl fmt::Display for FftType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::C2C => write!(f, "C2C"),
            Self::R2C => write!(f, "R2C"),
            Self::C2R => write!(f, "C2R"),
        }
    }
}

// ---------------------------------------------------------------------------
// FftDirection — forward / inverse
// ---------------------------------------------------------------------------

/// Direction of the FFT transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FftDirection {
    /// Forward transform (time -> frequency), applies `exp(-2*pi*i*k*n/N)`.
    Forward,
    /// Inverse transform (frequency -> time), applies `exp(+2*pi*i*k*n/N)`.
    Inverse,
}

impl FftDirection {
    /// Returns the twiddle sign: `-1.0` for forward, `+1.0` for inverse.
    #[inline]
    pub fn sign(self) -> f64 {
        match self {
            Self::Forward => -1.0,
            Self::Inverse => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// FftPrecision — floating-point width
// ---------------------------------------------------------------------------

/// Floating-point precision for the FFT computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FftPrecision {
    /// Single precision (`f32` / Complex\<`f32`\>).
    Single,
    /// Double precision (`f64` / Complex\<`f64`\>).
    Double,
}

impl FftPrecision {
    /// Size of one real element in bytes.
    #[inline]
    pub fn element_bytes(self) -> usize {
        match self {
            Self::Single => 4,
            Self::Double => 8,
        }
    }

    /// Size of one complex element in bytes (2 * real).
    #[inline]
    pub fn complex_bytes(self) -> usize {
        self.element_bytes() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_f32_arithmetic() {
        let a = Complex::<f32>::new(1.0, 2.0);
        let b = Complex::<f32>::new(3.0, 4.0);
        let sum = a + b;
        assert_eq!(sum, Complex::<f32>::new(4.0, 6.0));
        let prod = a * b;
        assert_eq!(prod, Complex::<f32>::new(-5.0, 10.0));
    }

    #[test]
    fn complex_f64_conj() {
        let z = Complex::<f64>::new(3.0, -4.0);
        assert_eq!(z.conj(), Complex::<f64>::new(3.0, 4.0));
        assert!((z.abs() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn direction_sign() {
        assert_eq!(FftDirection::Forward.sign(), -1.0);
        assert_eq!(FftDirection::Inverse.sign(), 1.0);
    }

    #[test]
    fn precision_bytes() {
        assert_eq!(FftPrecision::Single.element_bytes(), 4);
        assert_eq!(FftPrecision::Double.complex_bytes(), 16);
    }
}
