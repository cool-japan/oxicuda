//! ToRSh GPU Tensor Backend for OxiCUDA.
//!
//! This module provides a PyTorch-compatible GPU tensor backend with
//! autograd support, suitable for use by the ToRSh project and other
//! COOLJAPAN ecosystem consumers.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │            ToRSh / TrustformeRS             │
//! │              (consumers)                    │
//! └──────────────┬──────────────────────────────┘
//!                │
//! ┌──────────────▼──────────────────────────────┐
//! │          tensor_backend (this module)        │
//! │  ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
//! │  │ GpuTensor│ │ Autograd │ │  Optimizer   │  │
//! │  │  dtype   │ │   tape   │ │ SGD,Adam,... │  │
//! │  └──────────┘ └──────────┘ └─────────────┘  │
//! │  ┌──────────┐ ┌──────────────────────────┐  │
//! │  │   Ops    │ │   Mixed Precision        │  │
//! │  │ matmul,  │ │ GradScaler, Autocast     │  │
//! │  │ conv2d,  │ └──────────────────────────┘  │
//! │  │ softmax  │                               │
//! │  └──────────┘                               │
//! └──────────────┬──────────────────────────────┘
//!                │  (uses)
//! ┌──────────────▼──────────────────────────────┐
//! │   oxicuda-driver / oxicuda-memory / blas     │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```rust
//! use oxicuda::tensor_backend::*;
//!
//! # fn main() -> Result<(), oxicuda::tensor_backend::TensorError> {
//! // Create tensors
//! let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0], &[3], 0)?;
//! let b = GpuTensor::from_host_f64(&[4.0, 5.0, 6.0], &[3], 0)?;
//!
//! // Element-wise add (no autograd)
//! let c = ops::add(&a, &b, None)?;
//! assert!((c.item().unwrap_or(0.0) - 5.0).abs() > 0.0); // c has 3 elements
//! # Ok(())
//! # }
//! ```
//!
//! # Autograd Example
//!
//! ```rust
//! use oxicuda::tensor_backend::*;
//! use oxicuda::tensor_backend::autograd::AutogradTape;
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), oxicuda::tensor_backend::TensorError> {
//! let mut tape = AutogradTape::new();
//! let mut x = GpuTensor::from_host_f64(&[3.0], &[1], 0)?;
//! x.set_requires_grad(true);
//! let mut w = GpuTensor::from_host_f64(&[2.0], &[1], 0)?;
//! w.set_requires_grad(true);
//!
//! // y = w * x
//! let y = ops::mul(&w, &x, Some(&mut tape))?;
//! // loss = sum(y)
//! let loss = ops::sum(&y, Some(&mut tape))?;
//!
//! let mut tensors = HashMap::new();
//! tensors.insert(x.id(), x);
//! tensors.insert(w.id(), w);
//! tensors.insert(y.id(), y);
//! let loss_id = loss.id();
//! tensors.insert(loss_id, loss);
//!
//! tape.backward(loss_id, &mut tensors)?;
//! # Ok(())
//! # }
//! ```

/// Error types for tensor operations.
pub mod error;

/// Data type system for GPU tensors.
pub mod dtype;

/// GPU Tensor type with shape, stride, dtype, and autograd support.
pub mod tensor;

/// Autograd tape, backward pass, and gradient computation.
pub mod autograd;

/// Forward/backward implementations for tensor operations.
pub mod ops;

/// GPU-accelerated optimizers (SGD, Adam, AdaGrad, RMSProp, LAMB).
pub mod optimizer;

/// Mixed-precision training with loss scaling and autocast.
pub mod mixed_precision;

// ─── Public re-exports ──────────────────────────────────────

pub use dtype::{PrecisionCategory, TensorDtype};
pub use error::TensorError;
pub use tensor::{GpuTensor, SavedTensor, TensorId};

pub use autograd::{
    AutogradTape, CheckpointStrategy, GradFn, GradientCheckpointing, NoGradGuard, is_grad_enabled,
    no_grad,
};

pub use optimizer::{AdaGrad, Adam, Lamb, Optimizer, RmsProp, Sgd};

pub use mixed_precision::{Autocast, AutocastGuard, GradScaler, current_autocast, enter_autocast};
