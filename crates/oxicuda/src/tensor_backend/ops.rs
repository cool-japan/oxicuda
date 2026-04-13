//! Forward and backward implementations for tensor operations.
//!
//! Each operation computes the forward result and registers itself
//! on the autograd tape for gradient computation.

use super::autograd::{AutogradTape, GradFn, is_grad_enabled};
use super::error::TensorError;
use super::tensor::{GpuTensor, SavedTensor, TensorId, shape_numel};

// ─── Elementwise binary ops ─────────────────────────────────

/// Element-wise addition: `c = a + b`.
pub fn add(
    a: &GpuTensor,
    b: &GpuTensor,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    check_shapes_match(a, b)?;
    let data: Vec<f64> = a
        .host_data()
        .iter()
        .zip(b.host_data().iter())
        .map(|(&x, &y)| x + y)
        .collect();
    let requires_grad = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    let grad_fn = if requires_grad {
        Some(GradFn::Add {
            lhs: a.id(),
            rhs: b.id(),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Element-wise subtraction: `c = a - b`.
pub fn sub(
    a: &GpuTensor,
    b: &GpuTensor,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    check_shapes_match(a, b)?;
    let data: Vec<f64> = a
        .host_data()
        .iter()
        .zip(b.host_data().iter())
        .map(|(&x, &y)| x - y)
        .collect();
    let requires_grad = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    let grad_fn = if requires_grad {
        Some(GradFn::Sub {
            lhs: a.id(),
            rhs: b.id(),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Element-wise multiplication: `c = a * b`.
pub fn mul(
    a: &GpuTensor,
    b: &GpuTensor,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    check_shapes_match(a, b)?;
    let data: Vec<f64> = a
        .host_data()
        .iter()
        .zip(b.host_data().iter())
        .map(|(&x, &y)| x * y)
        .collect();
    let requires_grad = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    let grad_fn = if requires_grad {
        Some(GradFn::Mul {
            lhs: a.id(),
            rhs: b.id(),
            lhs_data: SavedTensor::from_tensor(a),
            rhs_data: SavedTensor::from_tensor(b),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Element-wise division: `c = a / b`.
pub fn div(
    a: &GpuTensor,
    b: &GpuTensor,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    check_shapes_match(a, b)?;
    let data: Vec<f64> = a
        .host_data()
        .iter()
        .zip(b.host_data().iter())
        .map(|(&x, &y)| if y.abs() > 1e-30 { x / y } else { 0.0 })
        .collect();
    let requires_grad = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    let grad_fn = if requires_grad {
        Some(GradFn::Div {
            lhs: a.id(),
            rhs: b.id(),
            lhs_data: SavedTensor::from_tensor(a),
            rhs_data: SavedTensor::from_tensor(b),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

// ─── Elementwise unary ops ──────────────────────────────────

/// Negation: `-a`.
pub fn neg(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    let data: Vec<f64> = a.host_data().iter().map(|&x| -x).collect();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::Neg { input: a.id() })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Absolute value.
pub fn abs(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    let data: Vec<f64> = a.host_data().iter().map(|&x| x.abs()).collect();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::Abs {
            input: a.id(),
            input_data: SavedTensor::from_tensor(a),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// ReLU activation: `max(0, x)`.
pub fn relu(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    let data: Vec<f64> = a.host_data().iter().map(|&x| x.max(0.0)).collect();
    let mask_data: Vec<f64> = a
        .host_data()
        .iter()
        .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
        .collect();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::Relu {
            input: a.id(),
            mask: SavedTensor {
                id: TensorId::new(),
                shape: a.shape().to_vec(),
                dtype: a.dtype(),
                data: mask_data,
            },
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Leaky ReLU: `x if x > 0 else alpha * x`.
pub fn leaky_relu(
    a: &GpuTensor,
    alpha: f64,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    let data: Vec<f64> = a
        .host_data()
        .iter()
        .map(|&x| if x > 0.0 { x } else { alpha * x })
        .collect();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::LeakyRelu {
            input: a.id(),
            alpha,
            input_data: SavedTensor::from_tensor(a),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Sigmoid: `1 / (1 + exp(-x))`.
pub fn sigmoid(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    let data: Vec<f64> = a
        .host_data()
        .iter()
        .map(|&x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        None,
    );
    let grad_fn = if requires_grad {
        Some(GradFn::Sigmoid {
            input: a.id(),
            output: SavedTensor::from_tensor(&out),
        })
    } else {
        None
    };
    let mut result = out;
    if let Some(ref gf) = grad_fn {
        result.set_grad_fn(gf.clone());
    }
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(result.id(), gf);
    }
    Ok(result)
}

/// Hyperbolic tangent.
pub fn tanh_op(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    let data: Vec<f64> = a.host_data().iter().map(|&x| x.tanh()).collect();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        None,
    );
    let grad_fn = if requires_grad {
        Some(GradFn::Tanh {
            input: a.id(),
            output: SavedTensor::from_tensor(&out),
        })
    } else {
        None
    };
    let mut result = out;
    if let Some(ref gf) = grad_fn {
        result.set_grad_fn(gf.clone());
    }
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(result.id(), gf);
    }
    Ok(result)
}

/// GELU activation (Gaussian Error Linear Unit).
pub fn gelu(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
    let data: Vec<f64> = a
        .host_data()
        .iter()
        .map(|&x| 0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh()))
        .collect();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::Gelu {
            input: a.id(),
            input_data: SavedTensor::from_tensor(a),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// SiLU (Swish): `x * sigmoid(x)`.
pub fn silu(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    let data: Vec<f64> = a
        .host_data()
        .iter()
        .map(|&x| x / (1.0 + (-x).exp()))
        .collect();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::Silu {
            input: a.id(),
            input_data: SavedTensor::from_tensor(a),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Exponential.
pub fn exp(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    let data: Vec<f64> = a.host_data().iter().map(|&x| x.exp()).collect();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        None,
    );
    let grad_fn = if requires_grad {
        Some(GradFn::Exp {
            input: a.id(),
            output: SavedTensor::from_tensor(&out),
        })
    } else {
        None
    };
    let mut result = out;
    if let Some(ref gf) = grad_fn {
        result.set_grad_fn(gf.clone());
    }
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(result.id(), gf);
    }
    Ok(result)
}

/// Natural logarithm.
pub fn log(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    let data: Vec<f64> = a.host_data().iter().map(|&x| x.ln()).collect();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::Log {
            input: a.id(),
            input_data: SavedTensor::from_tensor(a),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Square root.
pub fn sqrt(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    let data: Vec<f64> = a.host_data().iter().map(|&x| x.sqrt()).collect();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        None,
    );
    let grad_fn = if requires_grad {
        Some(GradFn::Sqrt {
            input: a.id(),
            output: SavedTensor::from_tensor(&out),
        })
    } else {
        None
    };
    let mut result = out;
    if let Some(ref gf) = grad_fn {
        result.set_grad_fn(gf.clone());
    }
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(result.id(), gf);
    }
    Ok(result)
}

/// Power: `x^p`.
pub fn pow(
    a: &GpuTensor,
    exponent: f64,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    let data: Vec<f64> = a.host_data().iter().map(|&x| x.powf(exponent)).collect();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::Pow {
            input: a.id(),
            exponent,
            input_data: SavedTensor::from_tensor(a),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

// ─── Matrix operations ──────────────────────────────────────

/// Matrix multiplication: `C = A @ B` where A is (M, K) and B is (K, N).
pub fn matmul(
    a: &GpuTensor,
    b: &GpuTensor,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(TensorError::InvalidOperation(
            "matmul requires 2D tensors".into(),
        ));
    }
    let m = a.shape()[0];
    let k_a = a.shape()[1];
    let k_b = b.shape()[0];
    let n = b.shape()[1];
    if k_a != k_b {
        return Err(TensorError::ShapeMismatch {
            expected: k_a,
            got: k_b,
        });
    }
    let k = k_a;

    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a.host_data()[i * k + l] * b.host_data()[l * n + j];
            }
            data[i * n + j] = sum;
        }
    }

    let requires_grad = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    let grad_fn = if requires_grad {
        Some(GradFn::MatMul {
            lhs: a.id(),
            rhs: b.id(),
            lhs_data: SavedTensor::from_tensor(a),
            rhs_data: SavedTensor::from_tensor(b),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        vec![m, n],
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

// ─── Reduction ops ──────────────────────────────────────────

/// Sum all elements to a scalar.
pub fn sum(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    let total: f64 = a.host_data().iter().sum();
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::Sum {
            input: a.id(),
            input_shape: a.shape().to_vec(),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        vec![1],
        a.dtype(),
        a.device_id(),
        vec![total],
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Mean of all elements.
pub fn mean(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    let n = a.numel();
    let total: f64 = a.host_data().iter().sum();
    let mean_val = if n > 0 { total / n as f64 } else { 0.0 };
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::Mean {
            input: a.id(),
            input_shape: a.shape().to_vec(),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        vec![1],
        a.dtype(),
        a.device_id(),
        vec![mean_val],
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Max of all elements.
pub fn max(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    if a.numel() == 0 {
        return Err(TensorError::InvalidOperation("empty tensor".into()));
    }
    let mut max_val = f64::NEG_INFINITY;
    let mut max_idx = 0;
    for (i, &v) in a.host_data().iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::Max {
            input: a.id(),
            indices: vec![max_idx],
            input_shape: a.shape().to_vec(),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        vec![1],
        a.dtype(),
        a.device_id(),
        vec![max_val],
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Min of all elements.
pub fn min(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    if a.numel() == 0 {
        return Err(TensorError::InvalidOperation("empty tensor".into()));
    }
    let mut min_val = f64::INFINITY;
    let mut min_idx = 0;
    for (i, &v) in a.host_data().iter().enumerate() {
        if v < min_val {
            min_val = v;
            min_idx = i;
        }
    }
    let requires_grad = is_grad_enabled() && a.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::Min {
            input: a.id(),
            indices: vec![min_idx],
            input_shape: a.shape().to_vec(),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        vec![1],
        a.dtype(),
        a.device_id(),
        vec![min_val],
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

// ─── Normalization ops ──────────────────────────────────────

/// Batch normalization forward.
///
/// Input shape: `[N, C, ...]` — normalizes over the batch (N) and spatial dims.
#[allow(clippy::needless_range_loop)]
pub fn batch_norm(
    input: &GpuTensor,
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    if input.ndim() < 2 {
        return Err(TensorError::InvalidOperation(
            "batch_norm requires at least 2D input".into(),
        ));
    }
    let num_channels = input.shape()[1];
    let spatial: usize = input.shape()[2..].iter().product();
    let batch = input.shape()[0];
    let per_channel = batch * spatial;

    let mut output_data = vec![0.0; input.numel()];
    let mut normalized_data = vec![0.0; input.numel()];
    let mut std_inv = vec![0.0; num_channels];

    for c in 0..num_channels {
        // Compute mean
        let mut channel_mean = 0.0;
        for n in 0..batch {
            for s in 0..spatial {
                let idx = (n * num_channels + c) * spatial + s;
                channel_mean += input.host_data()[idx];
            }
        }
        channel_mean /= per_channel as f64;

        // Compute variance
        let mut channel_var = 0.0;
        for n in 0..batch {
            for s in 0..spatial {
                let idx = (n * num_channels + c) * spatial + s;
                let diff = input.host_data()[idx] - channel_mean;
                channel_var += diff * diff;
            }
        }
        channel_var /= per_channel as f64;

        let inv_std = 1.0 / (channel_var + eps).sqrt();
        std_inv[c] = inv_std;
        let g = gamma.get(c).copied().unwrap_or(1.0);
        let b = beta.get(c).copied().unwrap_or(0.0);

        for n in 0..batch {
            for s in 0..spatial {
                let idx = (n * num_channels + c) * spatial + s;
                let normed = (input.host_data()[idx] - channel_mean) * inv_std;
                normalized_data[idx] = normed;
                output_data[idx] = g * normed + b;
            }
        }
    }

    let requires_grad = is_grad_enabled() && input.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::BatchNorm {
            input: input.id(),
            normalized: SavedTensor {
                id: TensorId::new(),
                shape: input.shape().to_vec(),
                dtype: input.dtype(),
                data: normalized_data,
            },
            std_inv,
            gamma: gamma.to_vec(),
            num_channels,
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        input.shape().to_vec(),
        input.dtype(),
        input.device_id(),
        output_data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Layer normalization forward.
#[allow(clippy::needless_range_loop)]
pub fn layer_norm(
    input: &GpuTensor,
    norm_shape: &[usize],
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    let norm_size: usize = norm_shape.iter().product();
    if norm_size == 0 || input.numel() % norm_size != 0 {
        return Err(TensorError::InvalidOperation(
            "layer_norm: incompatible norm_shape".into(),
        ));
    }
    let num_instances = input.numel() / norm_size;
    let mut output_data = vec![0.0; input.numel()];
    let mut normalized_data = vec![0.0; input.numel()];
    let mut std_inv = vec![0.0; num_instances];

    for inst in 0..num_instances {
        let start = inst * norm_size;
        let end = start + norm_size;
        let slice = &input.host_data()[start..end];

        let mean_val: f64 = slice.iter().sum::<f64>() / norm_size as f64;
        let var: f64 =
            slice.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / norm_size as f64;
        let inv = 1.0 / (var + eps).sqrt();
        std_inv[inst] = inv;

        for i in 0..norm_size {
            let normed = (slice[i] - mean_val) * inv;
            normalized_data[start + i] = normed;
            let g = gamma.get(i).copied().unwrap_or(1.0);
            let b = beta.get(i).copied().unwrap_or(0.0);
            output_data[start + i] = g * normed + b;
        }
    }

    let requires_grad = is_grad_enabled() && input.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::LayerNorm {
            input: input.id(),
            normalized: SavedTensor {
                id: TensorId::new(),
                shape: input.shape().to_vec(),
                dtype: input.dtype(),
                data: normalized_data,
            },
            std_inv,
            gamma: gamma.to_vec(),
            norm_shape: norm_shape.to_vec(),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        input.shape().to_vec(),
        input.dtype(),
        input.device_id(),
        output_data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Group normalization forward.
pub fn group_norm(
    input: &GpuTensor,
    num_groups: usize,
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    if input.ndim() < 2 {
        return Err(TensorError::InvalidOperation(
            "group_norm requires at least 2D".into(),
        ));
    }
    let num_channels = input.shape()[1];
    if num_channels % num_groups != 0 {
        return Err(TensorError::InvalidOperation(
            "channels must be divisible by num_groups".into(),
        ));
    }
    let batch = input.shape()[0];
    let spatial: usize = input.shape()[2..].iter().product();
    let channels_per_group = num_channels / num_groups;
    let group_size = channels_per_group * spatial;

    let mut output_data = vec![0.0; input.numel()];
    let mut normalized_data = vec![0.0; input.numel()];
    let mut std_inv = vec![0.0; batch * num_groups];

    for n in 0..batch {
        for g in 0..num_groups {
            let gi = n * num_groups + g;
            // Gather group elements
            let mut group_mean = 0.0;
            for c_off in 0..channels_per_group {
                let c = g * channels_per_group + c_off;
                for s in 0..spatial {
                    let idx = (n * num_channels + c) * spatial + s;
                    group_mean += input.host_data()[idx];
                }
            }
            group_mean /= group_size as f64;

            let mut group_var = 0.0;
            for c_off in 0..channels_per_group {
                let c = g * channels_per_group + c_off;
                for s in 0..spatial {
                    let idx = (n * num_channels + c) * spatial + s;
                    let diff = input.host_data()[idx] - group_mean;
                    group_var += diff * diff;
                }
            }
            group_var /= group_size as f64;
            let inv = 1.0 / (group_var + eps).sqrt();
            std_inv[gi] = inv;

            for c_off in 0..channels_per_group {
                let c = g * channels_per_group + c_off;
                let gm = gamma.get(c).copied().unwrap_or(1.0);
                let b = beta.get(c).copied().unwrap_or(0.0);
                for s in 0..spatial {
                    let idx = (n * num_channels + c) * spatial + s;
                    let normed = (input.host_data()[idx] - group_mean) * inv;
                    normalized_data[idx] = normed;
                    output_data[idx] = gm * normed + b;
                }
            }
        }
    }

    let requires_grad = is_grad_enabled() && input.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::GroupNorm {
            input: input.id(),
            normalized: SavedTensor {
                id: TensorId::new(),
                shape: input.shape().to_vec(),
                dtype: input.dtype(),
                data: normalized_data,
            },
            std_inv,
            gamma: gamma.to_vec(),
            num_groups,
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        input.shape().to_vec(),
        input.dtype(),
        input.device_id(),
        output_data,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

// ─── Activation / softmax ───────────────────────────────────

/// Numerically stable softmax along the last dimension.
pub fn softmax(a: &GpuTensor, tape: Option<&mut AutogradTape>) -> Result<GpuTensor, TensorError> {
    if a.numel() == 0 {
        return Err(TensorError::InvalidOperation("empty tensor".into()));
    }
    let max_val = a
        .host_data()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = a.host_data().iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f64 = exps.iter().sum();
    let data: Vec<f64> = exps.iter().map(|&e| e / sum_exp).collect();

    let requires_grad = is_grad_enabled() && a.requires_grad();
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        None,
    );
    let grad_fn = if requires_grad {
        Some(GradFn::Softmax {
            input: a.id(),
            output: SavedTensor::from_tensor(&out),
            dim: a.ndim().saturating_sub(1),
        })
    } else {
        None
    };
    let mut result = out;
    if let Some(ref gf) = grad_fn {
        result.set_grad_fn(gf.clone());
    }
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(result.id(), gf);
    }
    Ok(result)
}

/// Numerically stable log-softmax along the last dimension.
pub fn log_softmax(
    a: &GpuTensor,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    if a.numel() == 0 {
        return Err(TensorError::InvalidOperation("empty tensor".into()));
    }
    let max_val = a
        .host_data()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let shifted: Vec<f64> = a.host_data().iter().map(|&x| x - max_val).collect();
    let log_sum_exp = shifted.iter().map(|&x| x.exp()).sum::<f64>().ln();
    let data: Vec<f64> = shifted.iter().map(|&x| x - log_sum_exp).collect();

    let requires_grad = is_grad_enabled() && a.requires_grad();
    let out = GpuTensor::from_parts(
        a.shape().to_vec(),
        a.dtype(),
        a.device_id(),
        data,
        requires_grad,
        None,
    );
    let grad_fn = if requires_grad {
        Some(GradFn::LogSoftmax {
            input: a.id(),
            output: SavedTensor::from_tensor(&out),
            dim: a.ndim().saturating_sub(1),
        })
    } else {
        None
    };
    let mut result = out;
    if let Some(ref gf) = grad_fn {
        result.set_grad_fn(gf.clone());
    }
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(result.id(), gf);
    }
    Ok(result)
}

// ─── Loss functions (in ops_loss.rs) ────────────────────────

/// Cross-entropy loss: `-mean(log(softmax(logits))[target])`.
///
/// * `logits`: `[N, C]` — raw unnormalized predictions.
/// * `targets`: `[N]` — class indices.
#[allow(clippy::needless_range_loop)]
pub fn cross_entropy_loss(
    logits: &GpuTensor,
    targets: &[usize],
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    if logits.ndim() != 2 {
        return Err(TensorError::InvalidOperation(
            "cross_entropy requires [N, C] input".into(),
        ));
    }
    let batch_size = logits.shape()[0];
    let num_classes = logits.shape()[1];
    if targets.len() != batch_size {
        return Err(TensorError::ShapeMismatch {
            expected: batch_size,
            got: targets.len(),
        });
    }

    // Compute softmax probabilities per sample
    let mut probs = vec![0.0; batch_size * num_classes];
    let mut loss_total = 0.0;
    for b in 0..batch_size {
        let start = b * num_classes;
        let end = start + num_classes;
        let row = &logits.host_data()[start..end];
        let max_val = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        for c in 0..num_classes {
            probs[start + c] = exps[c] / sum_exp;
        }
        let t = targets[b];
        if t < num_classes {
            loss_total -= (probs[start + t]).max(1e-30).ln();
        }
    }
    let loss_val = loss_total / batch_size as f64;

    let requires_grad = is_grad_enabled() && logits.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::CrossEntropy {
            input: logits.id(),
            probs: SavedTensor {
                id: TensorId::new(),
                shape: logits.shape().to_vec(),
                dtype: logits.dtype(),
                data: probs,
            },
            targets: targets.to_vec(),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        vec![1],
        logits.dtype(),
        logits.device_id(),
        vec![loss_val],
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// MSE loss: `mean((pred - target)^2)`.
pub fn mse_loss(
    pred: &GpuTensor,
    target: &GpuTensor,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    check_shapes_match(pred, target)?;
    let diff: Vec<f64> = pred
        .host_data()
        .iter()
        .zip(target.host_data().iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let n = diff.len();
    let loss_val: f64 = diff.iter().map(|&d| d * d).sum::<f64>() / n as f64;

    let requires_grad = is_grad_enabled() && pred.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::MseLoss {
            input: pred.id(),
            target: target.id(),
            diff: SavedTensor {
                id: TensorId::new(),
                shape: pred.shape().to_vec(),
                dtype: pred.dtype(),
                data: diff,
            },
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        vec![1],
        pred.dtype(),
        pred.device_id(),
        vec![loss_val],
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// L1 loss: `mean(|pred - target|)`.
pub fn l1_loss(
    pred: &GpuTensor,
    target: &GpuTensor,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    check_shapes_match(pred, target)?;
    let diff: Vec<f64> = pred
        .host_data()
        .iter()
        .zip(target.host_data().iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let n = diff.len();
    let loss_val: f64 = diff.iter().map(|&d| d.abs()).sum::<f64>() / n as f64;
    let sign_diff: Vec<f64> = diff
        .iter()
        .map(|&d| {
            if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect();

    let requires_grad = is_grad_enabled() && pred.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::L1Loss {
            input: pred.id(),
            sign_diff: SavedTensor {
                id: TensorId::new(),
                shape: pred.shape().to_vec(),
                dtype: pred.dtype(),
                data: sign_diff,
            },
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        vec![1],
        pred.dtype(),
        pred.device_id(),
        vec![loss_val],
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Smooth L1 loss (Huber loss).
pub fn smooth_l1_loss(
    pred: &GpuTensor,
    target: &GpuTensor,
    beta: f64,
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    check_shapes_match(pred, target)?;
    let diff: Vec<f64> = pred
        .host_data()
        .iter()
        .zip(target.host_data().iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let n = diff.len();
    let loss_val: f64 = diff
        .iter()
        .map(|&d| {
            if d.abs() < beta {
                0.5 * d * d / beta
            } else {
                d.abs() - 0.5 * beta
            }
        })
        .sum::<f64>()
        / n as f64;

    let requires_grad = is_grad_enabled() && pred.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::SmoothL1Loss {
            input: pred.id(),
            beta,
            diff: SavedTensor {
                id: TensorId::new(),
                shape: pred.shape().to_vec(),
                dtype: pred.dtype(),
                data: diff,
            },
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        vec![1],
        pred.dtype(),
        pred.device_id(),
        vec![loss_val],
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Negative log-likelihood loss.
#[allow(clippy::needless_range_loop)]
pub fn nll_loss(
    log_probs: &GpuTensor,
    targets: &[usize],
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    if log_probs.ndim() != 2 {
        return Err(TensorError::InvalidOperation(
            "nll_loss requires [N, C] input".into(),
        ));
    }
    let batch_size = log_probs.shape()[0];
    let num_classes = log_probs.shape()[1];
    if targets.len() != batch_size {
        return Err(TensorError::ShapeMismatch {
            expected: batch_size,
            got: targets.len(),
        });
    }
    let mut loss_total = 0.0;
    for b in 0..batch_size {
        let t = targets[b];
        if t < num_classes {
            loss_total -= log_probs.host_data()[b * num_classes + t];
        }
    }
    let loss_val = loss_total / batch_size as f64;

    let requires_grad = is_grad_enabled() && log_probs.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::NllLoss {
            input: log_probs.id(),
            targets: targets.to_vec(),
            batch_size,
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        vec![1],
        log_probs.dtype(),
        log_probs.device_id(),
        vec![loss_val],
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

// ─── Convolution & pooling ──────────────────────────────────

/// Conv2D forward using im2col + matmul.
///
/// * `input`: `[N, C_in, H, W]`
/// * `weight`: `[C_out, C_in, kH, kW]`
/// * `bias`: optional `[C_out]`
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    input: &GpuTensor,
    weight: &GpuTensor,
    bias: Option<&[f64]>,
    stride: (usize, usize),
    padding: (usize, usize),
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    if input.ndim() != 4 || weight.ndim() != 4 {
        return Err(TensorError::InvalidOperation(
            "conv2d requires 4D input and weight".into(),
        ));
    }
    let (n, c_in, h, w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let (c_out, wc_in, kh, kw) = (
        weight.shape()[0],
        weight.shape()[1],
        weight.shape()[2],
        weight.shape()[3],
    );
    if c_in != wc_in {
        return Err(TensorError::ShapeMismatch {
            expected: c_in,
            got: wc_in,
        });
    }
    let out_h = (h + 2 * padding.0 - kh) / stride.0 + 1;
    let out_w = (w + 2 * padding.1 - kw) / stride.1 + 1;

    // im2col: build column matrix for each batch
    let col_rows = c_in * kh * kw;
    let col_cols = out_h * out_w;

    let mut output = vec![0.0; n * c_out * out_h * out_w];
    let mut all_col_data = Vec::new();

    for batch in 0..n {
        let col = im2col(input, batch, c_in, h, w, kh, kw, stride, padding);
        // matmul: weight_reshaped (c_out, col_rows) @ col (col_rows, col_cols)
        for co in 0..c_out {
            for oc in 0..col_cols {
                let mut val = 0.0;
                for cr in 0..col_rows {
                    val += weight.host_data()[co * col_rows + cr] * col[cr * col_cols + oc];
                }
                if let Some(b) = bias {
                    val += b.get(co).copied().unwrap_or(0.0);
                }
                output[(batch * c_out + co) * col_cols + oc] = val;
            }
        }
        all_col_data.extend_from_slice(&col);
    }

    let out_shape = vec![n, c_out, out_h, out_w];
    let requires_grad = is_grad_enabled() && input.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::Conv2d {
            input: input.id(),
            col_data: SavedTensor {
                id: TensorId::new(),
                shape: vec![n, col_rows, col_cols],
                dtype: input.dtype(),
                data: all_col_data,
            },
            weight_data: SavedTensor::from_tensor(weight),
            input_shape: input.shape().to_vec(),
            kernel_size: (kh, kw),
            stride,
            padding,
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        out_shape,
        input.dtype(),
        input.device_id(),
        output,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// im2col helper: extract columns for one batch sample.
#[allow(clippy::too_many_arguments)]
fn im2col(
    input: &GpuTensor,
    batch: usize,
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Vec<f64> {
    let out_h = (h + 2 * padding.0 - kh) / stride.0 + 1;
    let out_w = (w + 2 * padding.1 - kw) / stride.1 + 1;
    let col_rows = c_in * kh * kw;
    let col_cols = out_h * out_w;
    let mut col = vec![0.0; col_rows * col_cols];

    for c in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let row = c * kh * kw + ki * kw + kj;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih = oh * stride.0 + ki;
                        let iw = ow * stride.1 + kj;
                        let ih_orig = ih as isize - padding.0 as isize;
                        let iw_orig = iw as isize - padding.1 as isize;
                        let val = if ih_orig >= 0
                            && ih_orig < h as isize
                            && iw_orig >= 0
                            && iw_orig < w as isize
                        {
                            let idx =
                                ((batch * c_in + c) * h + ih_orig as usize) * w + iw_orig as usize;
                            input.host_data()[idx]
                        } else {
                            0.0
                        };
                        col[row * col_cols + oh * out_w + ow] = val;
                    }
                }
            }
        }
    }
    col
}

/// Max pooling 2D.
///
/// * `input`: `[N, C, H, W]`
pub fn max_pool2d(
    input: &GpuTensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    if input.ndim() != 4 {
        return Err(TensorError::InvalidOperation(
            "max_pool2d requires 4D input".into(),
        ));
    }
    let (n, c, h, w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let out_h = (h + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let out_w = (w + 2 * padding.1 - kernel_size.1) / stride.1 + 1;
    let out_numel = n * c * out_h * out_w;
    let mut output = vec![0.0; out_numel];
    let mut indices = vec![0usize; out_numel];

    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let out_idx = ((batch * c + ch) * out_h + oh) * out_w + ow;
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_input_idx = 0;
                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = (oh * stride.0 + kh) as isize - padding.0 as isize;
                            let iw = (ow * stride.1 + kw) as isize - padding.1 as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let in_idx = ((batch * c + ch) * h + ih as usize) * w + iw as usize;
                                if input.host_data()[in_idx] > max_val {
                                    max_val = input.host_data()[in_idx];
                                    max_input_idx = in_idx;
                                }
                            }
                        }
                    }
                    output[out_idx] = max_val;
                    indices[out_idx] = max_input_idx;
                }
            }
        }
    }

    let requires_grad = is_grad_enabled() && input.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::MaxPool2d {
            input: input.id(),
            indices: indices.clone(),
            input_shape: input.shape().to_vec(),
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        vec![n, c, out_h, out_w],
        input.dtype(),
        input.device_id(),
        output,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

/// Average pooling 2D.
pub fn avg_pool2d(
    input: &GpuTensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    tape: Option<&mut AutogradTape>,
) -> Result<GpuTensor, TensorError> {
    if input.ndim() != 4 {
        return Err(TensorError::InvalidOperation(
            "avg_pool2d requires 4D input".into(),
        ));
    }
    let (n, c, h, w) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let out_h = (h + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let out_w = (w + 2 * padding.1 - kernel_size.1) / stride.1 + 1;
    let pool_size = (kernel_size.0 * kernel_size.1) as f64;
    let mut output = vec![0.0; n * c * out_h * out_w];

    for batch in 0..n {
        for ch in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let out_idx = ((batch * c + ch) * out_h + oh) * out_w + ow;
                    let mut sum_val = 0.0;
                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = (oh * stride.0 + kh) as isize - padding.0 as isize;
                            let iw = (ow * stride.1 + kw) as isize - padding.1 as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let in_idx = ((batch * c + ch) * h + ih as usize) * w + iw as usize;
                                sum_val += input.host_data()[in_idx];
                            }
                        }
                    }
                    output[out_idx] = sum_val / pool_size;
                }
            }
        }
    }

    let requires_grad = is_grad_enabled() && input.requires_grad();
    let grad_fn = if requires_grad {
        Some(GradFn::AvgPool2d {
            input: input.id(),
            input_shape: input.shape().to_vec(),
            kernel_size,
            stride,
            padding,
        })
    } else {
        None
    };
    let out = GpuTensor::from_parts(
        vec![n, c, out_h, out_w],
        input.dtype(),
        input.device_id(),
        output,
        requires_grad,
        grad_fn.clone(),
    );
    if let (Some(tape), Some(gf)) = (tape, grad_fn) {
        tape.record(out.id(), gf);
    }
    Ok(out)
}

// ─── Helpers ────────────────────────────────────────────────

fn check_shapes_match(a: &GpuTensor, b: &GpuTensor) -> Result<(), TensorError> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: shape_numel(a.shape()),
            got: shape_numel(b.shape()),
        });
    }
    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_forward() {
        let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0], &[3], 0).unwrap();
        let b = GpuTensor::from_host_f64(&[4.0, 5.0, 6.0], &[3], 0).unwrap();
        let c = add(&a, &b, None).unwrap();
        assert!((c.host_data()[0] - 5.0).abs() < 1e-10);
        assert!((c.host_data()[1] - 7.0).abs() < 1e-10);
        assert!((c.host_data()[2] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_sub_forward() {
        let a = GpuTensor::from_host_f64(&[5.0, 3.0], &[2], 0).unwrap();
        let b = GpuTensor::from_host_f64(&[2.0, 1.0], &[2], 0).unwrap();
        let c = sub(&a, &b, None).unwrap();
        assert!((c.host_data()[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mul_forward() {
        let a = GpuTensor::from_host_f64(&[2.0, 3.0], &[2], 0).unwrap();
        let b = GpuTensor::from_host_f64(&[4.0, 5.0], &[2], 0).unwrap();
        let c = mul(&a, &b, None).unwrap();
        assert!((c.host_data()[0] - 8.0).abs() < 1e-10);
        assert!((c.host_data()[1] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_div_forward() {
        let a = GpuTensor::from_host_f64(&[10.0, 6.0], &[2], 0).unwrap();
        let b = GpuTensor::from_host_f64(&[2.0, 3.0], &[2], 0).unwrap();
        let c = div(&a, &b, None).unwrap();
        assert!((c.host_data()[0] - 5.0).abs() < 1e-10);
        assert!((c.host_data()[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_relu_forward() {
        let a = GpuTensor::from_host_f64(&[-1.0, 0.0, 2.0, -3.0], &[4], 0).unwrap();
        let c = relu(&a, None).unwrap();
        assert!((c.host_data()[0] - 0.0).abs() < 1e-10);
        assert!((c.host_data()[1] - 0.0).abs() < 1e-10);
        assert!((c.host_data()[2] - 2.0).abs() < 1e-10);
        assert!((c.host_data()[3] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_forward() {
        let a = GpuTensor::from_host_f64(&[0.0], &[1], 0).unwrap();
        let c = sigmoid(&a, None).unwrap();
        assert!((c.host_data()[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_forward() {
        // [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
        let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 0).unwrap();
        let b = GpuTensor::from_host_f64(&[5.0, 6.0, 7.0, 8.0], &[2, 2], 0).unwrap();
        let c = matmul(&a, &b, None).unwrap();
        assert!((c.host_data()[0] - 19.0).abs() < 1e-10);
        assert!((c.host_data()[1] - 22.0).abs() < 1e-10);
        assert!((c.host_data()[2] - 43.0).abs() < 1e-10);
        assert!((c.host_data()[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_sum_forward() {
        let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0], &[3], 0).unwrap();
        let s = sum(&a, None).unwrap();
        assert!((s.item().unwrap() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_forward() {
        let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0], &[3], 0).unwrap();
        let m = mean(&a, None).unwrap();
        assert!((m.item().unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_forward() {
        let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0], &[3], 0).unwrap();
        let s = softmax(&a, None).unwrap();
        let total: f64 = s.host_data().iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
        // Values should be monotonically increasing
        assert!(s.host_data()[0] < s.host_data()[1]);
        assert!(s.host_data()[1] < s.host_data()[2]);
    }

    #[test]
    fn test_cross_entropy_loss() {
        // 2 samples, 3 classes
        let logits = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3], 0).unwrap();
        let targets = vec![2, 0]; // class 2, class 0
        let loss = cross_entropy_loss(&logits, &targets, None).unwrap();
        // loss > 0
        assert!(loss.item().unwrap() > 0.0);
    }

    #[test]
    fn test_mse_loss() {
        let pred = GpuTensor::from_host_f64(&[1.0, 2.0], &[2], 0).unwrap();
        let target = GpuTensor::from_host_f64(&[1.0, 2.0], &[2], 0).unwrap();
        let loss = mse_loss(&pred, &target, None).unwrap();
        assert!((loss.item().unwrap() - 0.0).abs() < 1e-10);

        let target2 = GpuTensor::from_host_f64(&[2.0, 3.0], &[2], 0).unwrap();
        let loss2 = mse_loss(&pred, &target2, None).unwrap();
        assert!((loss2.item().unwrap() - 1.0).abs() < 1e-10); // mean((1)^2, (1)^2) = 1
    }

    #[test]
    fn test_conv2d_forward() {
        // Simple 1x1x3x3 input, 1x1x2x2 kernel, stride=1, padding=0
        let input_data: Vec<f64> = (1..=9).map(|x| x as f64).collect();
        let input = GpuTensor::from_host_f64(&input_data, &[1, 1, 3, 3], 0).unwrap();
        let weight = GpuTensor::from_host_f64(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2], 0).unwrap();
        let out = conv2d(&input, &weight, None, (1, 1), (0, 0), None).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        // kernel picks top-left and bottom-right of each 2x2 window
        // window (0,0): 1+5=6, (0,1): 2+6=8, (1,0): 4+8=12, (1,1): 5+9=14
        assert!((out.host_data()[0] - 6.0).abs() < 1e-10);
        assert!((out.host_data()[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_pool2d() {
        let input_data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let input = GpuTensor::from_host_f64(&input_data, &[1, 1, 4, 4], 0).unwrap();
        let out = max_pool2d(&input, (2, 2), (2, 2), (0, 0), None).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert!((out.host_data()[0] - 6.0).abs() < 1e-10); // max of [1,2,5,6]
        assert!((out.host_data()[1] - 8.0).abs() < 1e-10); // max of [3,4,7,8]
    }

    #[test]
    fn test_avg_pool2d() {
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = GpuTensor::from_host_f64(&input_data, &[1, 1, 2, 2], 0).unwrap();
        let out = avg_pool2d(&input, (2, 2), (2, 2), (0, 0), None).unwrap();
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert!((out.host_data()[0] - 2.5).abs() < 1e-10); // mean(1,2,3,4)
    }

    #[test]
    fn test_layer_norm_forward() {
        let a = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 0).unwrap();
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let out = layer_norm(&a, &[2], &gamma, &beta, 1e-5, None).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        // Each pair should be normalized to ~[-1, 1]
        assert!((out.host_data()[0] + out.host_data()[1]).abs() < 1e-5);
    }

    #[test]
    fn test_numerical_gradient_exp() {
        // Verify exp backward with finite differences
        let eps = 1e-5;
        let x_val = 1.5;
        let a = GpuTensor::from_host_f64(&[x_val], &[1], 0).unwrap();
        let fwd = exp(&a, None).unwrap();
        let analytical = fwd.host_data()[0]; // exp'(x) = exp(x)

        let a_plus = GpuTensor::from_host_f64(&[x_val + eps], &[1], 0).unwrap();
        let a_minus = GpuTensor::from_host_f64(&[x_val - eps], &[1], 0).unwrap();
        let f_plus = exp(&a_plus, None).unwrap();
        let f_minus = exp(&a_minus, None).unwrap();
        let numerical = (f_plus.host_data()[0] - f_minus.host_data()[0]) / (2.0 * eps);

        assert!((analytical - numerical).abs() < 1e-5);
    }

    #[test]
    fn test_gelu_forward() {
        let a = GpuTensor::from_host_f64(&[0.0, 1.0, -1.0], &[3], 0).unwrap();
        let c = gelu(&a, None).unwrap();
        // GELU(0) ≈ 0
        assert!((c.host_data()[0]).abs() < 1e-5);
        // GELU(1) ≈ 0.8412
        assert!((c.host_data()[1] - 0.8412).abs() < 0.01);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let a = GpuTensor::from_host_f64(&[1.0, 2.0], &[2], 0).unwrap();
        let b = GpuTensor::from_host_f64(&[1.0, 2.0, 3.0], &[3], 0).unwrap();
        assert!(add(&a, &b, None).is_err());
    }

    #[test]
    fn test_batch_norm_forward() {
        // 1 batch, 2 channels, 2x2 spatial
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = GpuTensor::from_host_f64(&data, &[1, 2, 2, 2], 0).unwrap();
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let out = batch_norm(&input, &gamma, &beta, 1e-5, None).unwrap();
        assert_eq!(out.shape(), &[1, 2, 2, 2]);
        // Each channel should be zero-mean (approximately)
        let ch0_mean: f64 = out.host_data()[0..4].iter().sum::<f64>() / 4.0;
        assert!(ch0_mean.abs() < 1e-5);
    }

    #[test]
    fn test_l1_loss() {
        let pred = GpuTensor::from_host_f64(&[1.0, 3.0], &[2], 0).unwrap();
        let target = GpuTensor::from_host_f64(&[2.0, 1.0], &[2], 0).unwrap();
        let loss = l1_loss(&pred, &target, None).unwrap();
        // mean(|1-2|, |3-1|) = mean(1, 2) = 1.5
        assert!((loss.item().unwrap() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_smooth_l1_loss() {
        let pred = GpuTensor::from_host_f64(&[0.5], &[1], 0).unwrap();
        let target = GpuTensor::from_host_f64(&[0.0], &[1], 0).unwrap();
        // beta=1.0: diff=0.5, |0.5| < 1.0 => 0.5*0.25/1.0 = 0.125
        let loss = smooth_l1_loss(&pred, &target, 1.0, None).unwrap();
        assert!((loss.item().unwrap() - 0.125).abs() < 1e-10);
    }
}
