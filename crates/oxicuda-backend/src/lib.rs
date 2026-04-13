//! Abstract compute backend for GPU-accelerated operations.
//!
//! The [`ComputeBackend`] trait defines the interface for GPU computation,
//! allowing higher-level crates (SciRS2, oxionnx, ToRSh, TrustformeRS)
//! to use GPU acceleration without coupling to specific GPU APIs.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────┐
//! │  SciRS2 / ToRSh / oxionnx  │
//! │         (consumers)         │
//! └─────────────┬───────────────┘
//!               │  dyn ComputeBackend
//! ┌─────────────▼───────────────┐
//! │       ComputeBackend        │
//! │     (trait definition)      │
//! └─────────────┬───────────────┘
//!               │
//! ┌─────────────▼───────────────┐
//! │  CudaBackend / MetalBackend │
//! │    (concrete impls)         │
//! └─────────────────────────────┘
//! ```

use std::fmt;

// ─── Error types ────────────────────────────────────────────

/// Error type for backend operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// The requested operation is not supported by this backend.
    Unsupported(String),
    /// A GPU/device error occurred.
    DeviceError(String),
    /// Invalid argument to an operation.
    InvalidArgument(String),
    /// Out of device memory.
    OutOfMemory,
    /// Backend not initialized — call [`ComputeBackend::init`] first.
    NotInitialized,
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(msg) => write!(f, "unsupported operation: {msg}"),
            Self::DeviceError(msg) => write!(f, "device error: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::OutOfMemory => write!(f, "out of device memory"),
            Self::NotInitialized => write!(f, "backend not initialized"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Result type for backend operations.
pub type BackendResult<T> = Result<T, BackendError>;

// ─── Operation enums ────────────────────────────────────────

/// Transpose mode for matrix operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendTranspose {
    /// No transpose — use the matrix as-is.
    NoTrans,
    /// Transpose (swap rows and columns).
    Trans,
    /// Conjugate transpose (Hermitian).
    ConjTrans,
}

impl fmt::Display for BackendTranspose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTrans => write!(f, "N"),
            Self::Trans => write!(f, "T"),
            Self::ConjTrans => write!(f, "C"),
        }
    }
}

/// Reduction operation applied along an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Summation.
    Sum,
    /// Maximum value.
    Max,
    /// Minimum value.
    Min,
    /// Arithmetic mean.
    Mean,
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sum => write!(f, "sum"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
            Self::Mean => write!(f, "mean"),
        }
    }
}

/// Element-wise unary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Rectified linear unit: max(0, x).
    Relu,
    /// Sigmoid: 1 / (1 + exp(-x)).
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Exponential.
    Exp,
    /// Natural logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// Absolute value.
    Abs,
    /// Negation.
    Neg,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relu => write!(f, "relu"),
            Self::Sigmoid => write!(f, "sigmoid"),
            Self::Tanh => write!(f, "tanh"),
            Self::Exp => write!(f, "exp"),
            Self::Log => write!(f, "log"),
            Self::Sqrt => write!(f, "sqrt"),
            Self::Abs => write!(f, "abs"),
            Self::Neg => write!(f, "neg"),
        }
    }
}

/// Element-wise binary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Element-wise maximum.
    Max,
    /// Element-wise minimum.
    Min,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "add"),
            Self::Sub => write!(f, "sub"),
            Self::Mul => write!(f, "mul"),
            Self::Div => write!(f, "div"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
        }
    }
}

// ─── ComputeBackend trait ───────────────────────────────────

/// Abstract compute backend trait.
///
/// Implementations provide GPU-accelerated compute operations.
/// All operations work with opaque device memory pointers (`u64`)
/// and explicit shape/stride information, making the trait
/// independent of any particular memory management scheme.
///
/// # Object Safety
///
/// This trait is object-safe and can be used as `Box<dyn ComputeBackend>`
/// or `&dyn ComputeBackend` for dynamic dispatch.
///
/// # Lifecycle
///
/// 1. Create the backend (`CudaBackend::new()`).
/// 2. Call [`init`](ComputeBackend::init) to select a device and create a context.
/// 3. Allocate memory with [`alloc`](ComputeBackend::alloc).
/// 4. Transfer data with [`copy_htod`](ComputeBackend::copy_htod).
/// 5. Run compute operations ([`gemm`](ComputeBackend::gemm), [`conv2d_forward`](ComputeBackend::conv2d_forward), etc.).
/// 6. Read results with [`copy_dtoh`](ComputeBackend::copy_dtoh).
/// 7. Free memory with [`free`](ComputeBackend::free).
pub trait ComputeBackend: Send + Sync + fmt::Debug {
    /// Backend name (e.g., `"cuda"`, `"rocm"`, `"metal"`).
    fn name(&self) -> &str;

    /// Initialize the backend (select device, create context).
    ///
    /// Must be called before any other operation. Calling `init` on an
    /// already-initialized backend is a no-op.
    fn init(&mut self) -> BackendResult<()>;

    /// Returns `true` if the backend is ready for operations.
    fn is_initialized(&self) -> bool;

    /// General matrix multiply: `C = alpha * op(A) * op(B) + beta * C`.
    ///
    /// # Arguments
    ///
    /// * `trans_a`, `trans_b` — transpose modes for A and B.
    /// * `m`, `n`, `k` — matrix dimensions (C is m×n, A is m×k, B is k×n after transpose).
    /// * `alpha`, `beta` — scaling factors.
    /// * `a_ptr`, `b_ptr`, `c_ptr` — device pointers to column-major f64 matrices.
    /// * `lda`, `ldb`, `ldc` — leading dimensions.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        trans_a: BackendTranspose,
        trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        lda: usize,
        b_ptr: u64,
        ldb: usize,
        beta: f64,
        c_ptr: u64,
        ldc: usize,
    ) -> BackendResult<()>;

    /// 2D convolution forward pass.
    ///
    /// # Arguments
    ///
    /// * `input_ptr` — device pointer to input tensor (NCHW layout).
    /// * `input_shape` — `[N, C, H, W]`.
    /// * `filter_ptr` — device pointer to filter tensor.
    /// * `filter_shape` — `[K, C, Fh, Fw]`.
    /// * `output_ptr` — device pointer to output tensor.
    /// * `output_shape` — `[N, K, Oh, Ow]`.
    /// * `stride` — `[sh, sw]`.
    /// * `padding` — `[ph, pw]`.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_forward(
        &self,
        input_ptr: u64,
        input_shape: &[usize],
        filter_ptr: u64,
        filter_shape: &[usize],
        output_ptr: u64,
        output_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> BackendResult<()>;

    /// Scaled dot-product attention.
    ///
    /// Computes `softmax(Q * K^T / scale) * V` with optional causal masking.
    ///
    /// # Arguments
    ///
    /// * `q_ptr`, `k_ptr`, `v_ptr` — device pointers to query, key, value tensors.
    /// * `o_ptr` — device pointer to output tensor.
    /// * `batch`, `heads` — batch size and number of attention heads.
    /// * `seq_q`, `seq_kv` — query and key/value sequence lengths.
    /// * `head_dim` — dimension of each attention head.
    /// * `scale` — attention scale factor (typically `1 / sqrt(head_dim)`).
    /// * `causal` — if `true`, apply causal (lower-triangular) mask.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        o_ptr: u64,
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_kv: usize,
        head_dim: usize,
        scale: f64,
        causal: bool,
    ) -> BackendResult<()>;

    /// Reduction along an axis.
    ///
    /// Reduces `input` along `axis` using the specified `op` and writes to `output`.
    fn reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()>;

    /// Element-wise unary operation.
    ///
    /// Applies `op` to each of the `n` elements at `input_ptr` and writes to `output_ptr`.
    fn unary(&self, op: UnaryOp, input_ptr: u64, output_ptr: u64, n: usize) -> BackendResult<()>;

    /// Element-wise binary operation.
    ///
    /// Applies `op` element-wise: `output[i] = op(a[i], b[i])` for `n` elements.
    fn binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()>;

    /// Synchronize all pending operations on this backend.
    ///
    /// Blocks the host until all previously submitted GPU work completes.
    fn synchronize(&self) -> BackendResult<()>;

    /// Allocate device memory.
    ///
    /// Returns an opaque device pointer. The caller is responsible for
    /// eventually calling [`free`](ComputeBackend::free).
    fn alloc(&self, bytes: usize) -> BackendResult<u64>;

    /// Free device memory previously allocated with [`alloc`](ComputeBackend::alloc).
    fn free(&self, ptr: u64) -> BackendResult<()>;

    /// Copy data from host memory to device memory.
    ///
    /// * `dst` — device pointer (destination).
    /// * `src` — host byte slice (source).
    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()>;

    /// Copy data from device memory to host memory.
    ///
    /// * `dst` — host byte slice (destination).
    /// * `src` — device pointer (source).
    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()>;
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_error_display() {
        assert_eq!(
            BackendError::Unsupported("foo".into()).to_string(),
            "unsupported operation: foo"
        );
        assert_eq!(
            BackendError::DeviceError("bar".into()).to_string(),
            "device error: bar"
        );
        assert_eq!(
            BackendError::InvalidArgument("baz".into()).to_string(),
            "invalid argument: baz"
        );
        assert_eq!(
            BackendError::OutOfMemory.to_string(),
            "out of device memory"
        );
        assert_eq!(
            BackendError::NotInitialized.to_string(),
            "backend not initialized"
        );
    }

    #[test]
    fn backend_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BackendError::DeviceError("test".into()));
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn backend_transpose_display_and_values() {
        assert_eq!(BackendTranspose::NoTrans.to_string(), "N");
        assert_eq!(BackendTranspose::Trans.to_string(), "T");
        assert_eq!(BackendTranspose::ConjTrans.to_string(), "C");

        // Equality
        assert_eq!(BackendTranspose::NoTrans, BackendTranspose::NoTrans);
        assert_ne!(BackendTranspose::NoTrans, BackendTranspose::Trans);
    }

    #[test]
    fn reduce_op_display_and_coverage() {
        let ops = [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min, ReduceOp::Mean];
        let names = ["sum", "max", "min", "mean"];
        for (op, name) in ops.iter().zip(names.iter()) {
            assert_eq!(op.to_string(), *name);
        }
    }

    #[test]
    fn unary_op_display_and_coverage() {
        let ops = [
            UnaryOp::Relu,
            UnaryOp::Sigmoid,
            UnaryOp::Tanh,
            UnaryOp::Exp,
            UnaryOp::Log,
            UnaryOp::Sqrt,
            UnaryOp::Abs,
            UnaryOp::Neg,
        ];
        let names = [
            "relu", "sigmoid", "tanh", "exp", "log", "sqrt", "abs", "neg",
        ];
        for (op, name) in ops.iter().zip(names.iter()) {
            assert_eq!(op.to_string(), *name);
        }
    }

    #[test]
    fn binary_op_display_and_coverage() {
        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Max,
            BinaryOp::Min,
        ];
        let names = ["add", "sub", "mul", "div", "max", "min"];
        for (op, name) in ops.iter().zip(names.iter()) {
            assert_eq!(op.to_string(), *name);
        }
    }

    #[test]
    fn enum_clone_and_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(ReduceOp::Sum);
        set.insert(ReduceOp::Max);
        assert!(set.contains(&ReduceOp::Sum));
        assert!(!set.contains(&ReduceOp::Min));

        // Clone
        let op = UnaryOp::Relu;
        let cloned = op;
        assert_eq!(op, cloned);

        let bop = BinaryOp::Add;
        let bcloned = bop;
        assert_eq!(bop, bcloned);

        let trans = BackendTranspose::ConjTrans;
        let tcloned = trans;
        assert_eq!(trans, tcloned);
    }
}
