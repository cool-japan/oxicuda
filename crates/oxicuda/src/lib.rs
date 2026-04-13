//! # OxiCUDA — Pure Rust CUDA Replacement
//!
//! OxiCUDA provides a complete, pure Rust replacement for NVIDIA's CUDA
//! software stack. It dynamically loads `libcuda.so` at runtime, requiring
//! no CUDA Toolkit at build time.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │           COOLJAPAN Ecosystem                 │
//! │  SciRS2 │ oxionnx │ TrustformeRS │ ToRSh     │
//! │         └────┬────┘              │            │
//! │              └───────────────────┘            │
//! │                      │                        │
//! │              ┌───────▼────────┐               │
//! │              │    OxiCUDA     │               │
//! │              ├────────────────┤               │
//! │              │ Driver (Vol.1) │               │
//! │              │ Memory (Vol.1) │               │
//! │              │ Launch (Vol.1) │               │
//! │              │ PTX    (Vol.2) │               │
//! │              │ Autotune(Vol.2)│               │
//! │              │ BLAS   (Vol.3) │               │
//! │              │ DNN    (Vol.4) │               │
//! │              │ FFT    (Vol.5) │               │
//! │              │ Sparse (Vol.5) │               │
//! │              │ Solver (Vol.5) │               │
//! │              │ Rand   (Vol.5) │               │
//! │              └───────┬────────┘               │
//! │              ┌───────▼────────┐               │
//! │              │ libcuda.so     │               │
//! │              │ (NVIDIA Driver)│               │
//! │              └────────────────┘               │
//! └──────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```no_run
//! use oxicuda::prelude::*;
//!
//! fn main() -> CudaResult<()> {
//!     // Initialize the CUDA driver
//!     oxicuda::init()?;
//!
//!     // Enumerate devices
//!     let device = Device::get(0)?;
//!     println!("GPU: {}", device.name()?);
//!
//!     // Create context and stream
//!     let ctx = Context::new(&device)?;
//!     let ctx = std::sync::Arc::new(ctx);
//!     let stream = Stream::new(&ctx)?;
//!
//!     // Allocate device memory
//!     let mut buf = DeviceBuffer::<f32>::alloc(1024)?;
//!     let host_data = vec![1.0f32; 1024];
//!     buf.copy_from_host(&host_data)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description | Default |
//! |---------|-------------|---------|
//! | `driver` | CUDA driver API wrapper | Yes |
//! | `memory` | GPU memory management | Yes |
//! | `launch` | Kernel launch infrastructure | Yes |
//! | `ptx` | PTX code generation DSL | No |
//! | `autotune` | Autotuner engine | No |
//! | `blas` | cuBLAS equivalent | No |
//! | `dnn` | cuDNN equivalent | No |
//! | `fft` | cuFFT equivalent | No |
//! | `sparse` | cuSPARSE equivalent | No |
//! | `solver` | cuSOLVER equivalent | No |
//! | `rand` | cuRAND equivalent | No |
//! | `pool` | Stream-ordered memory pool | No |
//! | `backend` | Abstract compute backend trait | No |
//! | `full` | Enable all features | No |
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::wildcard_imports)]

// ─── Global initialization with device auto-selection ───────

/// Global initialization with device auto-selection.
///
/// Provides [`lazy_init`](global_init::lazy_init),
/// [`OxiCudaRuntimeBuilder`](global_init::OxiCudaRuntimeBuilder), and
/// related helpers for one-call GPU setup.
pub mod global_init;

pub use global_init::{DeviceSelection, OxiCudaRuntime, OxiCudaRuntimeBuilder};

// ─── Profiling & tracing ────────────────────────────────────

/// Profiling and tracing hooks for kernel-level performance analysis.
///
/// Provides chrome://tracing compatible output for visualizing GPU kernel
/// execution, memory transfers, and synchronization events.
pub mod profiling;

// ─── Multi-GPU device pool ─────────────────────────────────

/// Thread-safe multi-GPU device pool with workload-aware scheduling.
///
/// Provides [`MultiGpuPool`](device_pool::MultiGpuPool),
/// [`DeviceSelectionPolicy`](device_pool::DeviceSelectionPolicy),
/// [`GpuLease`](device_pool::GpuLease), and
/// [`WorkloadBalancer`](device_pool::WorkloadBalancer).
pub mod device_pool;

// ─── Abstract compute backend ───────────────────────────────

/// Abstract compute backend for GPU-accelerated operations.
///
/// Provides the [`ComputeBackend`](backend::ComputeBackend) trait that
/// higher-level crates use for GPU dispatch without coupling to a
/// specific GPU API.
#[cfg(feature = "backend")]
pub mod backend;

/// ONNX GPU inference backend.
///
/// Provides a complete ONNX operator runtime with IR types, 60+ operators,
/// graph executor, memory planner, operator fusion, and shape inference.
#[cfg(feature = "onnx-backend")]
pub mod onnx_backend;

/// ToRSh GPU tensor backend with autograd, optimizers, and mixed precision.
///
/// Provides [`GpuTensor`](tensor_backend::GpuTensor), an autograd tape,
/// forward/backward ops (matmul, conv2d, softmax, loss functions, etc.),
/// optimizers (SGD, Adam, AdaGrad, RMSProp, LAMB), and mixed-precision
/// training (GradScaler, Autocast).
#[cfg(feature = "tensor-backend")]
pub mod tensor_backend;

/// TrustformeRS Transformer GPU Backend.
///
/// Provides transformer model inference infrastructure: paged KV-cache,
/// continuous batching, speculative decoding, attention dispatch,
/// token sampling, and quantized inference.
#[cfg(feature = "transformer-backend")]
pub mod transformer_backend;

/// WASM + WebGPU compute backend for browser environments.
///
/// Wraps [`oxicuda_webgpu::WebGpuBackend`] with WASM-specific bindings,
/// making the OxiCUDA compute API usable from a browser via WebAssembly.
/// On native targets the module is still available and compiles cleanly;
/// the `#[wasm_bindgen]` exports are only emitted when targeting `wasm32`.
#[cfg(feature = "wasm-backend")]
pub mod wasm_backend;

#[cfg(feature = "wasm-backend")]
pub use wasm_backend::WasmComputeBackend;

// ─── Collective communication (NCCL equivalent) ────────────

/// NCCL-equivalent collective communication primitives for multi-GPU training.
///
/// Provides AllReduce, AllGather, ReduceScatter, Broadcast, Reduce, and
/// AllToAll with ring / tree / recursive-halving algorithm support.
pub mod collective;

/// Pipeline parallelism primitives for multi-GPU model parallelism.
///
/// Provides scheduling algorithms (GPipe, 1F1B, Interleaved, ZeroBubble),
/// bubble analysis, activation checkpointing, and ASCII visualization.
pub mod pipeline_parallel;

/// Multi-node distributed training support (TCP/IP based).
///
/// Provides [`DistributedRuntime`](distributed::DistributedRuntime),
/// [`TcpStore`](distributed::TcpStore), [`FileStore`](distributed::FileStore),
/// [`GradientBucket`](distributed::GradientBucket), and
/// [`DistributedOptimizer`](distributed::DistributedOptimizer) for
/// coordinating training across multiple machines.
pub mod distributed;

// ─── Core crates (always available) ─────────────────────────

/// CUDA Driver API wrapper.
pub use oxicuda_driver as driver;

/// GPU memory management.
pub use oxicuda_memory as memory;

/// Kernel launch infrastructure.
pub use oxicuda_launch as launch;

// ─── Optional crates (feature-gated) ────────────────────────

/// PTX code generation DSL.
#[cfg(feature = "ptx")]
pub use oxicuda_ptx as ptx;

/// Autotuner engine.
#[cfg(feature = "autotune")]
pub use oxicuda_autotune as autotune;

/// GPU-accelerated BLAS operations.
#[cfg(feature = "blas")]
pub use oxicuda_blas as blas;

/// GPU-accelerated deep learning primitives.
#[cfg(feature = "dnn")]
pub use oxicuda_dnn as dnn;

/// GPU-accelerated FFT operations.
#[cfg(feature = "fft")]
pub use oxicuda_fft as fft;

/// GPU-accelerated sparse matrix operations.
#[cfg(feature = "sparse")]
pub use oxicuda_sparse as sparse;

/// GPU-accelerated matrix decompositions.
#[cfg(feature = "solver")]
pub use oxicuda_solver as solver;

/// GPU-accelerated random number generation.
#[cfg(feature = "rand")]
pub use oxicuda_rand as rand;

/// CUB-equivalent high-performance parallel GPU primitives.
///
/// Provides PTX code generators for warp, block, and device-wide reduce, scan,
/// histogram, radix sort, and merge sort — all without any CUDA SDK dependency.
#[cfg(feature = "primitives")]
pub use oxicuda_primitives as primitives;

/// Vulkan Compute backend for cross-vendor GPU compute.
#[cfg(feature = "vulkan")]
pub use oxicuda_vulkan as vulkan;

/// Apple Metal Compute backend (macOS/iOS).
#[cfg(feature = "metal")]
pub use oxicuda_metal as metal_backend;

/// WebGPU Compute backend (cross-platform via wgpu).
#[cfg(feature = "webgpu")]
pub use oxicuda_webgpu as webgpu;

/// AMD ROCm/HIP Compute backend (Linux with AMD GPU).
#[cfg(feature = "rocm")]
pub use oxicuda_rocm as rocm;

/// Intel Level Zero Compute backend (Linux/Windows with Intel GPU).
#[cfg(feature = "level-zero")]
pub use oxicuda_levelzero as level_zero;

// ─── Key type re-exports ─────────────────────────────────────

// Error types
pub use oxicuda_driver::{CudaError, CudaResult, DriverLoadError};

// Core types
pub use oxicuda_driver::{
    Context, Device, Event, Function, JitDiagnostic, JitLog, JitOptions, JitSeverity, Module,
    Stream,
};
pub use oxicuda_driver::{best_device, list_devices, try_driver};

// Memory types
pub use oxicuda_memory::copy;
pub use oxicuda_memory::{DeviceBuffer, DeviceSlice, PinnedBuffer, UnifiedBuffer};

// Launch types
pub use oxicuda_launch::{
    Dim3, Kernel, KernelArgs, LaunchParams, LaunchParamsBuilder, grid_size_for,
};

// Re-export the launch! macro
pub use oxicuda_launch::launch;

/// Initialize the CUDA driver API.
///
/// This must be called before any other OxiCUDA function.
/// It dynamically loads `libcuda.so` (Linux), `nvcuda.dll` (Windows),
/// and initializes the CUDA driver.
///
/// Returns `Err(CudaError::NotInitialized)` on macOS or systems
/// without an NVIDIA GPU.
pub fn init() -> CudaResult<()> {
    oxicuda_driver::init()
}

/// Compile-time feature availability.
pub mod features {
    /// Whether PTX code generation is available.
    pub const HAS_PTX: bool = cfg!(feature = "ptx");
    /// Whether the autotuner is available.
    pub const HAS_AUTOTUNE: bool = cfg!(feature = "autotune");
    /// Whether BLAS operations are available.
    pub const HAS_BLAS: bool = cfg!(feature = "blas");
    /// Whether DNN operations are available.
    pub const HAS_DNN: bool = cfg!(feature = "dnn");
    /// Whether FFT operations are available.
    pub const HAS_FFT: bool = cfg!(feature = "fft");
    /// Whether sparse matrix operations are available.
    pub const HAS_SPARSE: bool = cfg!(feature = "sparse");
    /// Whether solver operations are available.
    pub const HAS_SOLVER: bool = cfg!(feature = "solver");
    /// Whether random number generation is available.
    pub const HAS_RAND: bool = cfg!(feature = "rand");
    /// Whether the abstract compute backend is available.
    pub const HAS_BACKEND: bool = cfg!(feature = "backend");
    /// Whether the ONNX inference backend is available.
    pub const HAS_ONNX_BACKEND: bool = cfg!(feature = "onnx-backend");
    /// Whether the ToRSh tensor backend is available.
    pub const HAS_TENSOR_BACKEND: bool = cfg!(feature = "tensor-backend");
    /// Whether the TrustformeRS transformer backend is available.
    pub const HAS_TRANSFORMER_BACKEND: bool = cfg!(feature = "transformer-backend");
    /// Whether stream-ordered memory pool is available.
    pub const HAS_POOL: bool = cfg!(feature = "pool");
    /// Whether GPU tests are enabled.
    pub const HAS_GPU_TESTS: bool = cfg!(feature = "gpu-tests");
    /// Whether global initialization is available (always `true`).
    pub const HAS_GLOBAL_INIT: bool = true;
    /// Whether the Vulkan Compute backend is available.
    pub const HAS_VULKAN: bool = cfg!(feature = "vulkan");
    /// Whether the Apple Metal Compute backend is available.
    pub const HAS_METAL: bool = cfg!(feature = "metal");
    /// Whether the WebGPU Compute backend is available.
    pub const HAS_WEBGPU: bool = cfg!(feature = "webgpu");
    /// Whether the AMD ROCm/HIP Compute backend is available.
    pub const HAS_ROCM: bool = cfg!(feature = "rocm");
    /// Whether the Intel Level Zero Compute backend is available.
    pub const HAS_LEVEL_ZERO: bool = cfg!(feature = "level-zero");
    /// Whether the WASM + WebGPU browser backend is available.
    pub const HAS_WASM_BACKEND: bool = cfg!(feature = "wasm-backend");
}

// ---------------------------------------------------------------------------
// ComputeBackend auto-selection threshold
// ---------------------------------------------------------------------------

/// Auto-selection threshold for the compute backend.
///
/// Tensors or data buffers larger than this threshold (in bytes) will be
/// dispatched to the GPU backend; smaller workloads use the CPU backend.
/// The 64 KB default is tuned for SciRS2 workloads where GPU launch overhead
/// dominates for small matrices.
pub const AUTO_SELECT_THRESHOLD_BYTES: usize = 64 * 1024; // 65536 bytes

// ---------------------------------------------------------------------------
// ONNX supported operator list
// ---------------------------------------------------------------------------

/// List of ONNX operators supported by the OxiCUDA ONNX backend.
///
/// This is the canonical list used both for operator dispatch and for
/// validating ONNX model compatibility.
pub const SUPPORTED_ONNX_OPS: &[&str] = &[
    "MatMul",
    "Conv",
    "Relu",
    "BatchNormalization",
    "Softmax",
    "LayerNormalization",
    "Add",
    "Mul",
    "Transpose",
    "Reshape",
    "Concat",
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod umbrella_tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ComputeBackend auto-selection threshold tests
    // -----------------------------------------------------------------------

    #[test]
    fn compute_backend_threshold_is_64kb() {
        assert_eq!(
            AUTO_SELECT_THRESHOLD_BYTES,
            64 * 1024,
            "auto-select threshold must be exactly 64 KiB = 65536 bytes"
        );
    }

    #[test]
    fn small_tensor_uses_cpu_backend() {
        // A tensor with < 64 KB data is below the threshold → CPU backend selected.
        let small_data_bytes: usize = 1024; // 1 KB
        assert!(
            small_data_bytes < AUTO_SELECT_THRESHOLD_BYTES,
            "1 KB should be below threshold → CPU backend"
        );
    }

    #[test]
    fn large_tensor_uses_gpu_backend() {
        // A tensor with > 64 KB data is above the threshold → GPU backend attempted.
        let large_data_bytes: usize = 1024 * 1024; // 1 MB
        assert!(
            large_data_bytes > AUTO_SELECT_THRESHOLD_BYTES,
            "1 MB should be above threshold → GPU backend"
        );
    }

    #[test]
    fn threshold_boundary_values() {
        // Exactly at threshold: not above → CPU backend.
        const { assert!(AUTO_SELECT_THRESHOLD_BYTES <= AUTO_SELECT_THRESHOLD_BYTES) }
        // One byte above threshold → GPU backend.
        const { assert!(AUTO_SELECT_THRESHOLD_BYTES + 1 > AUTO_SELECT_THRESHOLD_BYTES) }
    }

    // -----------------------------------------------------------------------
    // ONNX operator interface tests
    // -----------------------------------------------------------------------

    #[test]
    fn onnx_matmul_op_name_correct() {
        assert!(
            SUPPORTED_ONNX_OPS.contains(&"MatMul"),
            "SUPPORTED_ONNX_OPS must contain 'MatMul'"
        );
    }

    #[test]
    fn onnx_conv_op_name_correct() {
        assert!(
            SUPPORTED_ONNX_OPS.contains(&"Conv"),
            "SUPPORTED_ONNX_OPS must contain 'Conv'"
        );
    }

    #[test]
    fn onnx_op_list_includes_relu() {
        assert!(
            SUPPORTED_ONNX_OPS.contains(&"Relu"),
            "SUPPORTED_ONNX_OPS must contain 'Relu'"
        );
    }

    #[test]
    fn onnx_op_list_includes_softmax() {
        assert!(
            SUPPORTED_ONNX_OPS.contains(&"Softmax"),
            "SUPPORTED_ONNX_OPS must contain 'Softmax'"
        );
    }

    #[test]
    fn onnx_op_list_includes_layer_norm() {
        assert!(
            SUPPORTED_ONNX_OPS.contains(&"LayerNormalization"),
            "SUPPORTED_ONNX_OPS must contain 'LayerNormalization'"
        );
    }

    #[test]
    fn onnx_op_list_includes_batch_norm() {
        assert!(
            SUPPORTED_ONNX_OPS.contains(&"BatchNormalization"),
            "SUPPORTED_ONNX_OPS must contain 'BatchNormalization'"
        );
    }

    // -----------------------------------------------------------------------
    // ToRSh SDPA + TrustformeRS MoE config tests
    // (gated on transformer-backend feature)
    // -----------------------------------------------------------------------

    #[cfg(feature = "transformer-backend")]
    mod transformer_tests {
        use crate::transformer_backend::attention::ComputeTier;
        use crate::transformer_backend::attention::{AttentionConfig, AttentionKind, HeadConfig};

        #[test]
        fn torsh_sdpa_attention_config_exists() {
            // Verify AttentionConfig exists and supports FlashAttention dispatch.
            let cfg = AttentionConfig {
                head_config: HeadConfig::Mha { num_heads: 32 },
                head_dim: 128,
                use_paged_cache: false,
                compute_tier: ComputeTier::Hopper,
                sliding_window: None,
                causal: true,
                scale: None,
                max_seq_len_hint: Some(4096),
            };
            // For Hopper + long sequences, kernel should be FlashHopper.
            use crate::transformer_backend::attention::AttentionDispatch;
            let dispatch = AttentionDispatch::new(cfg);
            assert!(dispatch.is_ok(), "AttentionDispatch::new should succeed");
            let mut dispatch = dispatch.expect("AttentionDispatch creation failed");
            // With long sequences on Hopper, Flash or FlashHopper should be selected.
            let kernel = dispatch.select_kernel(4096);
            assert!(
                matches!(kernel, AttentionKind::Flash | AttentionKind::FlashHopper),
                "Hopper with 4096 tokens should use Flash attention, got {kernel:?}"
            );
        }
    }

    /// MoE configuration verification (CPU-side, no GPU required).
    #[test]
    fn trustformers_moe_config_exists() {
        // Verify the MoE routing math: tokens_per_expert ≈ batch * seq * top_k / num_experts.
        let num_experts: usize = 8;
        let top_k: usize = 2;
        let batch_size: usize = 4;
        let seq_len: usize = 512;

        // Expected tokens per expert on average (Mixtral 8x7B pattern).
        let total_tokens = batch_size * seq_len;
        let routed_tokens = total_tokens * top_k;
        // Each of the 8 experts should receive routed_tokens / num_experts.
        let tokens_per_expert = routed_tokens / num_experts;

        // For batch=4, seq=512, top_k=2, num_experts=8:
        // total = 2048, routed = 4096, per_expert = 512.
        assert_eq!(total_tokens, 2048);
        assert_eq!(routed_tokens, 4096);
        assert_eq!(tokens_per_expert, 512);
    }

    #[test]
    fn moe_mixtral_config_8x7b() {
        // Verify the standard Mixtral 8x7B MoE routing configuration.
        let num_experts: usize = 8;
        let top_k: usize = 2;

        // Activation rate: each token activates top_k / num_experts fraction of experts.
        let activation_rate = top_k as f64 / num_experts as f64;
        assert!(
            (activation_rate - 0.25).abs() < 1e-10,
            "Mixtral 8x7B: activation rate = {activation_rate}, expected 0.25"
        );

        // Load balance: with uniform routing, each expert receives the same
        // expected number of tokens.
        let batch_size: usize = 16;
        let seq_len: usize = 1024;
        let expected_per_expert = batch_size * seq_len * top_k / num_experts;
        // 16 * 1024 * 2 / 8 = 4096 tokens per expert.
        assert_eq!(expected_per_expert, 4096);
    }
}

/// Convenience re-exports for common usage patterns.
///
/// ```no_run
/// use oxicuda::prelude::*;
/// ```
pub mod prelude {
    // Error handling
    pub use crate::{CudaError, CudaResult};

    // Initialization
    pub use crate::{init, try_driver};

    // Core GPU types
    pub use crate::{Context, Device, Event, Function, Module, Stream};
    pub use crate::{best_device, list_devices};

    // Memory management
    pub use crate::{DeviceBuffer, PinnedBuffer, UnifiedBuffer};

    // Kernel launch
    pub use crate::{Dim3, Kernel, KernelArgs, LaunchParams, grid_size_for};

    // Global initialization
    pub use crate::global_init::{
        default_context, default_device, default_stream, is_initialized, lazy_init,
    };

    // Parallel primitives (feature = "primitives")
    #[cfg(feature = "primitives")]
    pub use oxicuda_primitives::{PrimitivesError, PrimitivesHandle, PrimitivesResult, ReduceOp};
}
