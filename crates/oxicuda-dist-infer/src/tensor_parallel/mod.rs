//! Tensor parallelism (Megatron-LM style).
//!
//! A linear layer `y = x W^T + b` of shape `[batch, out]` is split across `tp`
//! GPUs so that each GPU holds `out / tp` output columns.
//!
//! # Column-parallel linear (forward)
//!
//! ```text
//! W  ∈ ℝ^{out × in}   split as  W_i ∈ ℝ^{(out/tp) × in}  for i = 0..tp
//! y_i = x W_i^T + b_i          (local GEMM on rank i)
//! y = concat(y_0, …, y_{tp-1}) over output dimension  (all-gather)
//! ```
//!
//! # Row-parallel linear (forward)
//!
//! ```text
//! W  ∈ ℝ^{out × in}   split as  W_i ∈ ℝ^{out × (in/tp)}  for i = 0..tp
//! x_i = x[:, i*(in/tp) : (i+1)*(in/tp)]
//! y_i = x_i W_i^T                           (local GEMM on rank i)
//! y   = Σ_i y_i                              (all-reduce over ranks)
//! ```

pub mod column_parallel;
pub mod row_parallel;

pub use column_parallel::{ColumnLinear, ColumnLinearShard};
pub use row_parallel::{RowLinear, RowLinearShard};
