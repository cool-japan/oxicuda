//! RNG engine implementations.
//!
//! Each engine produces raw pseudorandom bits using a different algorithm.
//! The engines are:
//!
//! - [`philox`] -- Philox-4x32-10, a counter-based PRNG (cuRAND default)
//! - [`xorwow`] -- XORWOW, XORshift with Weyl sequence addition
//! - [`mrg32k3a`] -- MRG32k3a, highest statistical quality
//! - [`aes_ctr`] -- AES-256-CTR CSPRNG for security-sensitive applications

pub mod aes_ctr;
pub mod mrg32k3a;
pub mod philox;
pub mod philox_optimized;
pub mod xorwow;
