//! ONNX GPU Inference Backend for OxiCUDA.
//!
//! Provides a complete ONNX operator runtime with:
//! - IR types for representing ONNX graphs ([`ir`](crate::onnx_backend::ir))
//! - 60+ operator implementations ([`ops`](crate::onnx_backend::ops))
//! - Graph executor with topological ordering ([`executor`](crate::onnx_backend::executor))
//! - Memory planning with buffer reuse ([`planner`](crate::onnx_backend::planner))
//! - Operator fusion passes ([`fusion`](crate::onnx_backend::fusion))
//! - Shape inference engine ([`shape_inference`](crate::onnx_backend::shape_inference))

pub mod executor;
pub mod fusion;
pub mod ir;
pub mod ops;
pub mod planner;
pub mod shape_inference;

pub use executor::GraphExecutor;
pub use fusion::{FusionPass, FusionResult};
pub use ir::{
    AttributeValue, DataType, Graph, Node, OnnxError, OnnxResult, OnnxTensor, TensorInfo,
    TensorShape,
};
pub use ops::OpRegistry;
pub use planner::{AllocationInfo, MemoryPlan, MemoryPlanner};
pub use shape_inference::ShapeInference;
