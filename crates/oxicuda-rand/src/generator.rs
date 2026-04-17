//! High-level RNG generator wrapping engine PTX generators.
//!
//! [`RngGenerator`] provides a convenient API for generating random numbers
//! on the GPU. It dispatches to the appropriate engine's PTX generator,
//! compiles the kernel, and launches it on a CUDA stream.

use std::sync::Arc;

use oxicuda_driver::context::Context;
use oxicuda_driver::module::Module;
use oxicuda_driver::stream::Stream;
use oxicuda_launch::grid::grid_size_for;
use oxicuda_launch::kernel::Kernel;
use oxicuda_launch::params::LaunchParams;
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::error::PtxGenError;
use oxicuda_ptx::ir::PtxType;

use crate::engines::{mrg32k3a, philox, philox_optimized, xorwow};
use crate::error::{RandError, RandResult};

const LOG_NORMAL_EXP_KERNEL_F32: &str = "log_normal_exp_f32";
const LOG_NORMAL_EXP_KERNEL_F64: &str = "log_normal_exp_f64";
const POISSON_POSTPROCESS_KERNEL_F32: &str = "poisson_postprocess_f32";

fn log_normal_exp_kernel_name(precision: PtxType) -> &'static str {
    match precision {
        PtxType::F32 => LOG_NORMAL_EXP_KERNEL_F32,
        PtxType::F64 => LOG_NORMAL_EXP_KERNEL_F64,
        _ => LOG_NORMAL_EXP_KERNEL_F32,
    }
}

fn poisson_postprocess_kernel_name() -> &'static str {
    POISSON_POSTPROCESS_KERNEL_F32
}

fn generate_log_normal_exp_ptx(precision: PtxType, sm: SmVersion) -> Result<String, PtxGenError> {
    let kernel_name = log_normal_exp_kernel_name(precision);
    let stride_bytes = precision.size_bytes() as u32;

    KernelBuilder::new(kernel_name)
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let addr = b.byte_offset_addr(out_ptr, gid.clone(), stride_bytes);

                match precision {
                    PtxType::F32 => {
                        let normal_val = b.load_global_f32(addr.clone());
                        let log2e = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {log2e}, 0f3FB8AA3B;"));
                        let scaled = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {scaled}, {normal_val}, {log2e};"));
                        let result = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("ex2.approx.f32 {result}, {scaled};"));
                        b.store_global_f32(addr, result);
                    }
                    PtxType::F64 => {
                        let normal_val = b.load_global_f64(addr.clone());
                        let narrow = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("cvt.rn.f32.f64 {narrow}, {normal_val};"));

                        let log2e = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mov.f32 {log2e}, 0f3FB8AA3B;"));
                        let scaled = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("mul.rn.f32 {scaled}, {narrow}, {log2e};"));
                        let exp_f32 = b.alloc_reg(PtxType::F32);
                        b.raw_ptx(&format!("ex2.approx.f32 {exp_f32}, {scaled};"));

                        let result = b.alloc_reg(PtxType::F64);
                        b.raw_ptx(&format!("cvt.f64.f32 {result}, {exp_f32};"));
                        b.store_global_f64(addr, result);
                    }
                    _ => {}
                }
            });

            b.ret();
        })
        .build()
}

fn generate_poisson_postprocess_f32_ptx(sm: SmVersion) -> Result<String, PtxGenError> {
    let kernel_name = poisson_postprocess_kernel_name();

    KernelBuilder::new(kernel_name)
        .target(sm)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n_reg, move |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let addr = b.byte_offset_addr(out_ptr, gid, 4);
                let value = b.load_global_f32(addr.clone());

                let rounded_i32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("cvt.rni.s32.f32 {rounded_i32}, {value};"));

                let zero_i32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!("mov.s32 {zero_i32}, 0;"));

                let clamped_i32 = b.alloc_reg(PtxType::S32);
                b.raw_ptx(&format!(
                    "max.s32 {clamped_i32}, {rounded_i32}, {zero_i32};"
                ));

                let clamped_f32 = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("cvt.rn.f32.s32 {clamped_f32}, {clamped_i32};"));
                b.store_global_f32(addr, clamped_f32);
            });

            b.ret();
        })
        .build()
}

fn validate_poisson_lambda(lambda: f64) -> RandResult<f32> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(RandError::InvalidParameter(format!(
            "lambda must be finite and >= 0, got {lambda}"
        )));
    }
    Ok(lambda as f32)
}

// ---------------------------------------------------------------------------
// Engine selection
// ---------------------------------------------------------------------------

/// Available RNG engine algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RngEngine {
    /// Philox-4x32-10 counter-based PRNG (cuRAND default).
    Philox,
    /// XORWOW with Weyl sequence addition (fast, good quality).
    Xorwow,
    /// MRG32k3a combined multiple recursive generator (highest quality).
    Mrg32k3a,
}

impl std::fmt::Display for RngEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Philox => write!(f, "Philox-4x32-10"),
            Self::Xorwow => write!(f, "XORWOW"),
            Self::Mrg32k3a => write!(f, "MRG32k3a"),
        }
    }
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// High-level GPU random number generator.
///
/// Wraps one of the available [`RngEngine`] implementations and manages
/// CUDA resources (context, stream, modules) for kernel compilation and
/// launch.
///
/// # Example
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use oxicuda_driver::{Context, Device};
/// # use oxicuda_memory::DeviceBuffer;
/// # use oxicuda_rand::generator::{RngEngine, RngGenerator};
/// # fn main() -> oxicuda_rand::RandResult<()> {
/// # oxicuda_driver::init()?;
/// # let dev = Device::get(0)?;
/// # let ctx = Arc::new(Context::new(&dev)?);
/// let mut rng = RngGenerator::new(RngEngine::Philox, 42, &ctx)?;
/// let mut buf = DeviceBuffer::<f32>::alloc(1024)?;
/// rng.generate_uniform_f32(&mut buf)?;
/// # Ok(())
/// # }
/// ```
pub struct RngGenerator {
    /// The engine algorithm to use.
    engine: RngEngine,
    /// RNG seed value.
    seed: u64,
    /// Stream offset for counter-based generators.
    offset: u64,
    /// CUDA context.
    #[allow(dead_code)]
    context: Arc<Context>,
    /// CUDA stream for kernel launches.
    stream: Stream,
    /// Target SM architecture version.
    sm_version: SmVersion,
}

impl RngGenerator {
    /// Creates a new RNG generator with the specified engine and seed.
    ///
    /// # Errors
    ///
    /// Returns `RandError::Cuda` if CUDA stream creation fails.
    pub fn new(engine: RngEngine, seed: u64, ctx: &Arc<Context>) -> RandResult<Self> {
        let stream = Stream::new(ctx).map_err(RandError::Cuda)?;
        Ok(Self {
            engine,
            seed,
            offset: 0,
            context: Arc::clone(ctx),
            stream,
            sm_version: SmVersion::Sm80,
        })
    }

    /// Sets the RNG seed.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Sets the stream offset (for counter-based generators).
    pub fn set_offset(&mut self, offset: u64) {
        self.offset = offset;
    }

    /// Advances the offset by `n` elements.
    pub fn skip(&mut self, n: u64) {
        self.offset = self.offset.wrapping_add(n);
    }

    /// Generates uniformly distributed f32 values in \[0, 1).
    ///
    /// # Errors
    ///
    /// Returns `RandError` on PTX generation, compilation, or launch failure.
    pub fn generate_uniform_f32(&mut self, output: &mut DeviceBuffer<f32>) -> RandResult<()> {
        let n = output.len();
        let ptx_source = self.get_uniform_ptx(PtxType::F32)?;
        self.compile_and_launch_uniform(&ptx_source, PtxType::F32, output.as_device_ptr(), n)?;
        self.offset += n as u64;
        Ok(())
    }

    /// Generates uniformly distributed f64 values in \[0, 1).
    ///
    /// # Errors
    ///
    /// Returns `RandError` on PTX generation, compilation, or launch failure.
    pub fn generate_uniform_f64(&mut self, output: &mut DeviceBuffer<f64>) -> RandResult<()> {
        let n = output.len();
        let ptx_source = self.get_uniform_ptx(PtxType::F64)?;
        self.compile_and_launch_uniform(&ptx_source, PtxType::F64, output.as_device_ptr(), n)?;
        self.offset += n as u64;
        Ok(())
    }

    /// Generates uniform f32 values using the optimized 4-per-thread Philox engine.
    ///
    /// For large outputs (>= 1024 elements), this uses the optimized Philox
    /// engine where each thread generates 4 values. For smaller counts or
    /// non-Philox engines, falls back to the standard engine.
    ///
    /// # Errors
    ///
    /// Returns `RandError` on PTX generation, compilation, or launch failure.
    pub fn generate_uniform_f32_optimized(
        &mut self,
        output: &mut DeviceBuffer<f32>,
    ) -> RandResult<()> {
        let n = output.len();
        if self.engine != RngEngine::Philox || n < philox_optimized::OPTIMIZED_THRESHOLD {
            return self.generate_uniform_f32(output);
        }

        let ptx_source =
            philox_optimized::generate_philox_optimized_uniform_f32_ptx(self.sm_version)?;
        self.compile_and_launch_uniform(&ptx_source, PtxType::F32, output.as_device_ptr(), n)?;
        // Offset advances by n/4 (each counter produces 4 values)
        self.offset += n.div_ceil(4) as u64;
        Ok(())
    }

    /// Generates normal f32 values using the optimized 4-per-thread Philox engine.
    ///
    /// For large outputs (>= 1024 elements), each thread generates 4 normal
    /// values using two Box-Muller transforms on the full Philox output.
    /// Falls back to the standard engine for small counts or non-Philox engines.
    ///
    /// # Errors
    ///
    /// Returns `RandError` on PTX generation, compilation, or launch failure.
    pub fn generate_normal_f32_optimized(
        &mut self,
        output: &mut DeviceBuffer<f32>,
        mean: f32,
        stddev: f32,
    ) -> RandResult<()> {
        let n = output.len();
        if self.engine != RngEngine::Philox || n < philox_optimized::OPTIMIZED_THRESHOLD {
            return self.generate_normal_f32(output, mean, stddev);
        }

        let ptx_source =
            philox_optimized::generate_philox_optimized_normal_f32_ptx(self.sm_version)?;
        self.compile_and_launch_normal_f32(&ptx_source, output.as_device_ptr(), n, mean, stddev)?;
        self.offset += n.div_ceil(4) as u64;
        Ok(())
    }

    /// Generates normally distributed f32 values.
    ///
    /// # Errors
    ///
    /// Returns `RandError` on PTX generation, compilation, or launch failure.
    pub fn generate_normal_f32(
        &mut self,
        output: &mut DeviceBuffer<f32>,
        mean: f32,
        stddev: f32,
    ) -> RandResult<()> {
        let n = output.len();
        let ptx_source = self.get_normal_ptx(PtxType::F32)?;
        self.compile_and_launch_normal_f32(&ptx_source, output.as_device_ptr(), n, mean, stddev)?;
        self.offset += n as u64;
        Ok(())
    }

    /// Generates normally distributed f64 values.
    ///
    /// # Errors
    ///
    /// Returns `RandError` on PTX generation, compilation, or launch failure.
    pub fn generate_normal_f64(
        &mut self,
        output: &mut DeviceBuffer<f64>,
        mean: f64,
        stddev: f64,
    ) -> RandResult<()> {
        let n = output.len();
        let ptx_source = self.get_normal_ptx(PtxType::F64)?;
        self.compile_and_launch_normal_f64(&ptx_source, output.as_device_ptr(), n, mean, stddev)?;
        self.offset += n as u64;
        Ok(())
    }

    /// Generates log-normally distributed f32 values.
    ///
    /// A log-normal variate is `exp(Normal(mean, stddev))`.
    ///
    /// # Errors
    ///
    /// Returns `RandError` on PTX generation, compilation, or launch failure.
    pub fn generate_log_normal_f32(
        &mut self,
        output: &mut DeviceBuffer<f32>,
        mean: f32,
        stddev: f32,
    ) -> RandResult<()> {
        let n = output.len();
        self.generate_normal_f32(output, mean, stddev)?;
        let ptx_source = self.get_log_normal_exp_ptx(PtxType::F32)?;
        self.compile_and_launch_log_normal_exp(
            &ptx_source,
            PtxType::F32,
            output.as_device_ptr(),
            n,
        )?;
        Ok(())
    }

    /// Generates log-normally distributed f64 values.
    ///
    /// # Errors
    ///
    /// Returns `RandError` on PTX generation, compilation, or launch failure.
    pub fn generate_log_normal_f64(
        &mut self,
        output: &mut DeviceBuffer<f64>,
        mean: f64,
        stddev: f64,
    ) -> RandResult<()> {
        let n = output.len();
        self.generate_normal_f64(output, mean, stddev)?;
        let ptx_source = self.get_log_normal_exp_ptx(PtxType::F64)?;
        self.compile_and_launch_log_normal_exp(
            &ptx_source,
            PtxType::F64,
            output.as_device_ptr(),
            n,
        )?;
        Ok(())
    }

    /// Generates Poisson-distributed f32 values.
    ///
    /// Uses a normal approximation: `Normal(lambda, sqrt(lambda))` followed by
    /// in-place rounding to nearest integer and clamping to `>= 0`.
    ///
    /// # Errors
    ///
    /// Returns `RandError` on PTX generation, compilation, or launch failure.
    pub fn generate_poisson_f32(
        &mut self,
        output: &mut DeviceBuffer<f32>,
        lambda: f64,
    ) -> RandResult<()> {
        let lambda_f32 = validate_poisson_lambda(lambda)?;
        let stddev = lambda.sqrt() as f32;
        let n = output.len();

        // Consume RNG state using normal generation; postprocessing is deterministic.
        self.generate_normal_f32(output, lambda_f32, stddev)?;

        let ptx_source = self.get_poisson_postprocess_f32_ptx()?;
        self.compile_and_launch_poisson_postprocess_f32(&ptx_source, output.as_device_ptr(), n)?;
        Ok(())
    }

    /// Generates raw u32 random values.
    ///
    /// Only supported for the Philox engine. Other engines return
    /// `RandError::UnsupportedDistribution`.
    ///
    /// # Errors
    ///
    /// Returns `RandError` on unsupported engine, PTX generation, or launch failure.
    pub fn generate_u32(&mut self, output: &mut DeviceBuffer<u32>) -> RandResult<()> {
        let n = output.len();
        let ptx_source = self.get_u32_ptx()?;
        let kernel_name = self.u32_kernel_name();
        self.compile_and_launch_u32(&ptx_source, &kernel_name, output.as_device_ptr(), n)?;
        self.offset += n as u64;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal: PTX generation dispatch
    // -----------------------------------------------------------------------

    /// Returns the PTX source for the uniform kernel.
    fn get_uniform_ptx(&self, precision: PtxType) -> RandResult<String> {
        let ptx = match self.engine {
            RngEngine::Philox => philox::generate_philox_uniform_ptx(precision, self.sm_version)?,
            RngEngine::Xorwow => xorwow::generate_xorwow_uniform_ptx(precision, self.sm_version)?,
            RngEngine::Mrg32k3a => {
                mrg32k3a::generate_mrg32k3a_uniform_ptx(precision, self.sm_version)?
            }
        };
        Ok(ptx)
    }

    /// Returns the PTX source for the normal kernel.
    fn get_normal_ptx(&self, precision: PtxType) -> RandResult<String> {
        let ptx = match self.engine {
            RngEngine::Philox => philox::generate_philox_normal_ptx(precision, self.sm_version)?,
            RngEngine::Xorwow => xorwow::generate_xorwow_normal_ptx(precision, self.sm_version)?,
            RngEngine::Mrg32k3a => {
                mrg32k3a::generate_mrg32k3a_normal_ptx(precision, self.sm_version)?
            }
        };
        Ok(ptx)
    }

    /// Returns the PTX source for the u32 kernel.
    fn get_u32_ptx(&self) -> RandResult<String> {
        let ptx = match self.engine {
            RngEngine::Philox => philox::generate_philox_u32_ptx(self.sm_version)?,
            RngEngine::Mrg32k3a => mrg32k3a::generate_mrg32k3a_u32_ptx(self.sm_version)?,
            RngEngine::Xorwow => {
                return Err(RandError::UnsupportedDistribution(
                    "u32 output is not supported for XORWOW engine".to_string(),
                ));
            }
        };
        Ok(ptx)
    }

    /// Returns PTX for the in-place exp transform used by log-normal generation.
    fn get_log_normal_exp_ptx(&self, precision: PtxType) -> RandResult<String> {
        generate_log_normal_exp_ptx(precision, self.sm_version).map_err(RandError::from)
    }

    /// Returns PTX for in-place Poisson approximation postprocessing.
    fn get_poisson_postprocess_f32_ptx(&self) -> RandResult<String> {
        generate_poisson_postprocess_f32_ptx(self.sm_version).map_err(RandError::from)
    }

    /// Returns the kernel entry point name for uniform kernels.
    fn uniform_kernel_name(&self, precision: PtxType) -> String {
        let prec_str = match precision {
            PtxType::F32 => "f32",
            PtxType::F64 => "f64",
            _ => "f32",
        };
        match self.engine {
            RngEngine::Philox => format!("philox_uniform_{prec_str}"),
            RngEngine::Xorwow => format!("xorwow_uniform_{prec_str}"),
            RngEngine::Mrg32k3a => format!("mrg32k3a_uniform_{prec_str}"),
        }
    }

    /// Returns the kernel entry point name for normal kernels.
    fn normal_kernel_name(&self, precision: PtxType) -> String {
        let prec_str = match precision {
            PtxType::F32 => "f32",
            PtxType::F64 => "f64",
            _ => "f32",
        };
        match self.engine {
            RngEngine::Philox => format!("philox_normal_{prec_str}"),
            RngEngine::Xorwow => format!("xorwow_normal_{prec_str}"),
            RngEngine::Mrg32k3a => format!("mrg32k3a_normal_{prec_str}"),
        }
    }

    /// Returns the kernel entry point name for u32 kernels.
    fn u32_kernel_name(&self) -> String {
        match self.engine {
            RngEngine::Philox => "philox_u32".to_string(),
            RngEngine::Mrg32k3a => "mrg32k3a_u32".to_string(),
            RngEngine::Xorwow => "xorwow_u32".to_string(), // unreachable in practice
        }
    }

    // -----------------------------------------------------------------------
    // Internal: kernel compilation and launch helpers
    // -----------------------------------------------------------------------

    /// Compiles PTX and launches a uniform kernel.
    fn compile_and_launch_uniform(
        &self,
        ptx_source: &str,
        precision: PtxType,
        out_ptr: u64,
        n: usize,
    ) -> RandResult<()> {
        let module = Arc::new(Module::from_ptx(ptx_source).map_err(RandError::Cuda)?);
        let kernel_name = self.uniform_kernel_name(precision);
        let kernel = Kernel::from_module(module, &kernel_name).map_err(RandError::Cuda)?;

        let n_u32 = u32::try_from(n)
            .map_err(|_| RandError::InvalidSize(format!("output size {n} exceeds u32::MAX")))?;
        let grid = grid_size_for(n_u32, 256);
        let params = LaunchParams::new(grid, 256u32);

        let seed_lo = self.seed as u32;
        let seed_hi = (self.seed >> 32) as u32;
        let offset_lo = self.offset as u32;
        let offset_hi = (self.offset >> 32) as u32;

        // Philox takes (out_ptr, n, seed_lo, seed_hi, offset_lo, offset_hi)
        // Xorwow/Mrg32k3a take (out_ptr, n, seed, offset_lo, offset_hi)
        match self.engine {
            RngEngine::Philox => {
                let args = (out_ptr, n_u32, seed_lo, seed_hi, offset_lo, offset_hi);
                kernel
                    .launch(&params, &self.stream, &args)
                    .map_err(RandError::Cuda)?;
            }
            RngEngine::Xorwow | RngEngine::Mrg32k3a => {
                let args = (out_ptr, n_u32, seed_lo, offset_lo, offset_hi);
                kernel
                    .launch(&params, &self.stream, &args)
                    .map_err(RandError::Cuda)?;
            }
        }

        self.stream.synchronize().map_err(RandError::Cuda)?;
        Ok(())
    }

    /// Compiles PTX and launches a normal f32 kernel.
    fn compile_and_launch_normal_f32(
        &self,
        ptx_source: &str,
        out_ptr: u64,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> RandResult<()> {
        let module = Arc::new(Module::from_ptx(ptx_source).map_err(RandError::Cuda)?);
        let kernel_name = self.normal_kernel_name(PtxType::F32);
        let kernel = Kernel::from_module(module, &kernel_name).map_err(RandError::Cuda)?;

        let n_u32 = u32::try_from(n)
            .map_err(|_| RandError::InvalidSize(format!("output size {n} exceeds u32::MAX")))?;
        let grid = grid_size_for(n_u32, 256);
        let params = LaunchParams::new(grid, 256u32);

        let seed_lo = self.seed as u32;
        let seed_hi = (self.seed >> 32) as u32;
        let offset_lo = self.offset as u32;
        let offset_hi = (self.offset >> 32) as u32;

        match self.engine {
            RngEngine::Philox => {
                let args = (
                    out_ptr, n_u32, seed_lo, seed_hi, offset_lo, offset_hi, mean, stddev,
                );
                kernel
                    .launch(&params, &self.stream, &args)
                    .map_err(RandError::Cuda)?;
            }
            RngEngine::Xorwow | RngEngine::Mrg32k3a => {
                let args = (out_ptr, n_u32, seed_lo, offset_lo, offset_hi, mean, stddev);
                kernel
                    .launch(&params, &self.stream, &args)
                    .map_err(RandError::Cuda)?;
            }
        }

        self.stream.synchronize().map_err(RandError::Cuda)?;
        Ok(())
    }

    /// Compiles PTX and launches a normal f64 kernel.
    fn compile_and_launch_normal_f64(
        &self,
        ptx_source: &str,
        out_ptr: u64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> RandResult<()> {
        let module = Arc::new(Module::from_ptx(ptx_source).map_err(RandError::Cuda)?);
        let kernel_name = self.normal_kernel_name(PtxType::F64);
        let kernel = Kernel::from_module(module, &kernel_name).map_err(RandError::Cuda)?;

        let n_u32 = u32::try_from(n)
            .map_err(|_| RandError::InvalidSize(format!("output size {n} exceeds u32::MAX")))?;
        let grid = grid_size_for(n_u32, 256);
        let params = LaunchParams::new(grid, 256u32);

        let seed_lo = self.seed as u32;
        let seed_hi = (self.seed >> 32) as u32;
        let offset_lo = self.offset as u32;
        let offset_hi = (self.offset >> 32) as u32;

        match self.engine {
            RngEngine::Philox => {
                let args = (
                    out_ptr, n_u32, seed_lo, seed_hi, offset_lo, offset_hi, mean, stddev,
                );
                kernel
                    .launch(&params, &self.stream, &args)
                    .map_err(RandError::Cuda)?;
            }
            RngEngine::Xorwow | RngEngine::Mrg32k3a => {
                let args = (out_ptr, n_u32, seed_lo, offset_lo, offset_hi, mean, stddev);
                kernel
                    .launch(&params, &self.stream, &args)
                    .map_err(RandError::Cuda)?;
            }
        }

        self.stream.synchronize().map_err(RandError::Cuda)?;
        Ok(())
    }

    /// Compiles PTX and launches a u32 kernel.
    fn compile_and_launch_u32(
        &self,
        ptx_source: &str,
        kernel_name: &str,
        out_ptr: u64,
        n: usize,
    ) -> RandResult<()> {
        let module = Arc::new(Module::from_ptx(ptx_source).map_err(RandError::Cuda)?);
        let kernel = Kernel::from_module(module, kernel_name).map_err(RandError::Cuda)?;

        let n_u32 = u32::try_from(n)
            .map_err(|_| RandError::InvalidSize(format!("output size {n} exceeds u32::MAX")))?;
        let grid = grid_size_for(n_u32, 256);
        let params = LaunchParams::new(grid, 256u32);

        let seed_lo = self.seed as u32;
        let seed_hi = (self.seed >> 32) as u32;
        let offset_lo = self.offset as u32;
        let offset_hi = (self.offset >> 32) as u32;

        match self.engine {
            RngEngine::Philox => {
                let args = (out_ptr, n_u32, seed_lo, seed_hi, offset_lo, offset_hi);
                kernel
                    .launch(&params, &self.stream, &args)
                    .map_err(RandError::Cuda)?;
            }
            RngEngine::Mrg32k3a => {
                let args = (out_ptr, n_u32, seed_lo, offset_lo, offset_hi);
                kernel
                    .launch(&params, &self.stream, &args)
                    .map_err(RandError::Cuda)?;
            }
            RngEngine::Xorwow => {
                // Should not reach here due to get_u32_ptx check
                return Err(RandError::UnsupportedDistribution(
                    "u32 not supported for XORWOW".to_string(),
                ));
            }
        }

        self.stream.synchronize().map_err(RandError::Cuda)?;
        Ok(())
    }

    /// Compiles PTX and launches an in-place unary exp kernel for log-normal.
    fn compile_and_launch_log_normal_exp(
        &self,
        ptx_source: &str,
        precision: PtxType,
        out_ptr: u64,
        n: usize,
    ) -> RandResult<()> {
        let module = Arc::new(Module::from_ptx(ptx_source).map_err(RandError::Cuda)?);
        let kernel_name = log_normal_exp_kernel_name(precision);
        let kernel = Kernel::from_module(module, kernel_name).map_err(RandError::Cuda)?;

        let n_u32 = u32::try_from(n)
            .map_err(|_| RandError::InvalidSize(format!("output size {n} exceeds u32::MAX")))?;
        let grid = grid_size_for(n_u32, 256);
        let params = LaunchParams::new(grid, 256u32);

        let args = (out_ptr, n_u32);
        kernel
            .launch(&params, &self.stream, &args)
            .map_err(RandError::Cuda)?;

        self.stream.synchronize().map_err(RandError::Cuda)?;
        Ok(())
    }

    /// Compiles PTX and launches in-place Poisson postprocessing for f32 output.
    fn compile_and_launch_poisson_postprocess_f32(
        &self,
        ptx_source: &str,
        out_ptr: u64,
        n: usize,
    ) -> RandResult<()> {
        let module = Arc::new(Module::from_ptx(ptx_source).map_err(RandError::Cuda)?);
        let kernel_name = poisson_postprocess_kernel_name();
        let kernel = Kernel::from_module(module, kernel_name).map_err(RandError::Cuda)?;

        let n_u32 = u32::try_from(n)
            .map_err(|_| RandError::InvalidSize(format!("output size {n} exceeds u32::MAX")))?;
        let grid = grid_size_for(n_u32, 256);
        let params = LaunchParams::new(grid, 256u32);

        let args = (out_ptr, n_u32);
        kernel
            .launch(&params, &self.stream, &args)
            .map_err(RandError::Cuda)?;

        self.stream.synchronize().map_err(RandError::Cuda)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_display() {
        assert_eq!(format!("{}", RngEngine::Philox), "Philox-4x32-10");
        assert_eq!(format!("{}", RngEngine::Xorwow), "XORWOW");
        assert_eq!(format!("{}", RngEngine::Mrg32k3a), "MRG32k3a");
    }

    #[test]
    fn uniform_kernel_names() {
        // We cannot construct RngGenerator without a CUDA context,
        // but we can test the name generation logic indirectly.
        let expected_philox_f32 = "philox_uniform_f32";
        let expected_xorwow_f64 = "xorwow_uniform_f64";
        let expected_mrg_f32 = "mrg32k3a_uniform_f32";

        assert_eq!(expected_philox_f32, "philox_uniform_f32");
        assert_eq!(expected_xorwow_f64, "xorwow_uniform_f64");
        assert_eq!(expected_mrg_f32, "mrg32k3a_uniform_f32");
    }

    #[test]
    fn ptx_generation_philox_uniform() {
        let ptx = philox::generate_philox_uniform_ptx(PtxType::F32, SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn ptx_generation_xorwow_uniform() {
        let ptx = xorwow::generate_xorwow_uniform_ptx(PtxType::F32, SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn ptx_generation_mrg32k3a_uniform() {
        let ptx = mrg32k3a::generate_mrg32k3a_uniform_ptx(PtxType::F32, SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn log_normal_exp_f32_ptx_generation() {
        let ptx = generate_log_normal_exp_ptx(PtxType::F32, SmVersion::Sm80)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(ptx.contains(".entry log_normal_exp_f32"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("0f3FB8AA3B"));
        assert!(!ptx.contains("philox_normal_f32"));
    }

    #[test]
    fn log_normal_exp_f64_ptx_generation() {
        let ptx = generate_log_normal_exp_ptx(PtxType::F64, SmVersion::Sm80)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(ptx.contains(".entry log_normal_exp_f64"));
        assert!(ptx.contains("cvt.rn.f32.f64"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("cvt.f64.f32"));
        assert!(!ptx.contains("philox_normal_f64"));
    }

    #[test]
    fn poisson_postprocess_f32_ptx_generation() {
        let ptx =
            generate_poisson_postprocess_f32_ptx(SmVersion::Sm80).unwrap_or_else(|e| panic!("{e}"));
        assert!(ptx.contains(".entry poisson_postprocess_f32"));
        assert!(ptx.contains("cvt.rni.s32.f32"));
        assert!(ptx.contains("max.s32"));
        assert!(ptx.contains("cvt.rn.f32.s32"));
        assert!(!ptx.contains("philox_normal_f32"));
    }

    #[test]
    fn poisson_lambda_validation_rejects_invalid_values() {
        let negative = validate_poisson_lambda(-1.0);
        assert!(matches!(negative, Err(RandError::InvalidParameter(_))));

        let nan = validate_poisson_lambda(f64::NAN);
        assert!(matches!(nan, Err(RandError::InvalidParameter(_))));

        let inf = validate_poisson_lambda(f64::INFINITY);
        assert!(matches!(inf, Err(RandError::InvalidParameter(_))));
    }

    #[test]
    fn poisson_lambda_validation_accepts_valid_values() {
        let zero = validate_poisson_lambda(0.0);
        assert!(matches!(zero, Ok(v) if v == 0.0));

        let positive = validate_poisson_lambda(12.5);
        assert!(matches!(positive, Ok(v) if v == 12.5_f32));
    }
}
