//! Quantization-Aware Training (QAT) support.
//!
//! QAT uses fake quantization during training: quantize then dequantize in a
//! single forward pass, with Straight-Through Estimator (STE) for backward
//! gradients.  This module provides configuration, observer state tracking,
//! quantization parameter computation, and PTX kernel generation for:
//!
//! - **Fake quantize** (forward): `out = (clamp(round(x/scale) + zp, qmin, qmax) - zp) * scale`
//! - **STE backward**: `grad_out = grad_in * (x >= qmin_f && x <= qmax_f ? 1.0 : 0.0)`
//! - **Observer** (min/max reduction): parallel reduction to find tensor min/max

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{DnnError, DnnResult};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Quantization bit width for QAT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QatBitWidth {
    /// 4-bit integer quantization (signed): range [-8, 7].
    Int4,
    /// 8-bit integer quantization (signed): range [-128, 127].
    Int8,
}

impl QatBitWidth {
    /// Minimum representable quantized value.
    #[inline]
    pub fn quant_min(self) -> i32 {
        match self {
            Self::Int4 => -8,
            Self::Int8 => -128,
        }
    }

    /// Maximum representable quantized value.
    #[inline]
    pub fn quant_max(self) -> i32 {
        match self {
            Self::Int4 => 7,
            Self::Int8 => 127,
        }
    }

    /// Total number of distinct quantization levels.
    #[inline]
    pub fn num_levels(self) -> u32 {
        match self {
            Self::Int4 => 16,
            Self::Int8 => 256,
        }
    }
}

/// Whether quantization is symmetric (zero-point = 0) or asymmetric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QatSymmetry {
    /// Symmetric quantization: `zero_point = 0`, scale based on max absolute value.
    Symmetric,
    /// Asymmetric quantization: zero-point is computed to map the full [min, max]
    /// range to [quant_min, quant_max].
    Asymmetric,
}

/// Granularity of quantization parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QatGranularity {
    /// One scale/zero-point pair for the entire tensor.
    PerTensor,
    /// One scale/zero-point pair per slice along the given axis (e.g. per
    /// output channel).
    PerChannel {
        /// The axis along which separate parameters are computed.
        axis: u32,
    },
}

/// Observer algorithm for tracking tensor statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObserverMode {
    /// Track absolute min/max across all observed batches.
    MinMax,
    /// Exponential moving average of min/max with given momentum.
    MovingAverage {
        /// Momentum coefficient in (0, 1). Higher values weight history more.
        momentum: f32,
    },
    /// Use a percentile of the distribution for range estimation.
    Percentile {
        /// Percentile in (0, 100].
        percentile: f32,
    },
}

// ---------------------------------------------------------------------------
// QatConfig
// ---------------------------------------------------------------------------

/// Complete configuration for a QAT pass.
#[derive(Debug, Clone)]
pub struct QatConfig {
    /// Bit width for quantization.
    pub bit_width: QatBitWidth,
    /// Symmetric vs asymmetric quantization.
    pub symmetry: QatSymmetry,
    /// Per-tensor or per-channel granularity.
    pub granularity: QatGranularity,
    /// Observer algorithm for range estimation.
    pub observer: ObserverMode,
    /// Target SM version for PTX generation.
    pub sm_version: SmVersion,
    /// Floating-point type used by the tensor data.
    pub float_type: PtxType,
}

impl QatConfig {
    /// Validates the configuration, returning an error for invalid combinations.
    pub fn validate(&self) -> DnnResult<()> {
        // float_type must actually be a float
        match self.float_type {
            PtxType::F16 | PtxType::F32 | PtxType::F64 | PtxType::BF16 => {}
            other => {
                return Err(DnnError::InvalidArgument(format!(
                    "QAT float_type must be a floating-point type, got {other:?}"
                )));
            }
        }

        // Moving average momentum must be in (0, 1)
        if let ObserverMode::MovingAverage { momentum } = self.observer {
            if momentum <= 0.0 || momentum >= 1.0 {
                return Err(DnnError::InvalidArgument(format!(
                    "MovingAverage momentum must be in (0, 1), got {momentum}"
                )));
            }
        }

        // Percentile must be in (0, 100]
        if let ObserverMode::Percentile { percentile } = self.observer {
            if percentile <= 0.0 || percentile > 100.0 {
                return Err(DnnError::InvalidArgument(format!(
                    "Percentile must be in (0, 100], got {percentile}"
                )));
            }
        }

        // PerChannel with Int4 is allowed but axis must be reasonable
        // (we cannot fully validate axis without tensor shape, so we just check
        // for a very large axis that is almost certainly wrong)
        if let QatGranularity::PerChannel { axis } = self.granularity {
            if axis > 1024 {
                return Err(DnnError::InvalidArgument(format!(
                    "PerChannel axis {axis} is unreasonably large (> 1024)"
                )));
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// QuantParams
// ---------------------------------------------------------------------------

/// Computed quantization parameters (scale and zero-point).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantParams {
    /// Scale factor: real_value = (quantized - zero_point) * scale.
    pub scale: f32,
    /// Zero-point offset (0 for symmetric quantization).
    pub zero_point: i32,
}

impl QuantParams {
    /// Computes quantization parameters from observed min/max and configuration.
    ///
    /// For **symmetric** quantization:
    ///   - `scale = max(|min|, |max|) / quant_max`
    ///   - `zero_point = 0`
    ///
    /// For **asymmetric** quantization:
    ///   - `scale = (max - min) / (quant_max - quant_min)`
    ///   - `zero_point = round(quant_min - min / scale)`
    pub fn compute_from_min_max(min: f32, max: f32, config: &QatConfig) -> Self {
        let qmin = config.bit_width.quant_min();
        let qmax = config.bit_width.quant_max();

        // Handle degenerate case where min == max
        let (adj_min, adj_max) = if (max - min).abs() < f32::EPSILON {
            // Ensure we have a non-zero range
            (min - 1.0, max + 1.0)
        } else {
            (min, max)
        };

        match config.symmetry {
            QatSymmetry::Symmetric => {
                let abs_max = adj_min.abs().max(adj_max.abs());
                let scale = abs_max / (qmax as f32);
                // Guard against zero scale
                let safe_scale = if scale.abs() < f32::EPSILON {
                    f32::EPSILON
                } else {
                    scale
                };
                Self {
                    scale: safe_scale,
                    zero_point: 0,
                }
            }
            QatSymmetry::Asymmetric => {
                let range = adj_max - adj_min;
                let qrange = (qmax - qmin) as f32;
                let scale = range / qrange;
                let safe_scale = if scale.abs() < f32::EPSILON {
                    f32::EPSILON
                } else {
                    scale
                };
                let zero_point = (qmin as f32 - adj_min / safe_scale).round() as i32;
                // Clamp zero_point to quantized range
                let zero_point = zero_point.clamp(qmin, qmax);
                Self {
                    scale: safe_scale,
                    zero_point,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ObserverState
// ---------------------------------------------------------------------------

/// Internal state for a MinMax observer.
#[derive(Debug, Clone)]
pub struct MinMaxState {
    /// Running minimum.
    pub min: f32,
    /// Running maximum.
    pub max: f32,
    /// Number of batches observed.
    pub num_batches: u64,
}

/// Internal state for a MovingAverage observer.
#[derive(Debug, Clone)]
pub struct MovingAverageState {
    /// Exponential moving average of the minimum.
    pub min_avg: f32,
    /// Exponential moving average of the maximum.
    pub max_avg: f32,
}

/// Tracks running statistics for quantization range estimation.
#[derive(Debug, Clone)]
pub enum ObserverState {
    /// Absolute min/max tracking.
    MinMax(MinMaxState),
    /// Exponential moving average tracking.
    MovingAverage(MovingAverageState),
}

impl ObserverState {
    /// Creates a new observer state for the given mode.
    pub fn new(mode: &ObserverMode) -> Self {
        match mode {
            ObserverMode::MinMax | ObserverMode::Percentile { .. } => Self::MinMax(MinMaxState {
                min: f32::MAX,
                max: f32::MIN,
                num_batches: 0,
            }),
            ObserverMode::MovingAverage { .. } => Self::MovingAverage(MovingAverageState {
                min_avg: 0.0,
                max_avg: 0.0,
            }),
        }
    }

    /// Updates the observer with newly observed min/max from a batch.
    pub fn update(&mut self, observed_min: f32, observed_max: f32) {
        match self {
            Self::MinMax(state) => {
                state.min = state.min.min(observed_min);
                state.max = state.max.max(observed_max);
                state.num_batches += 1;
            }
            Self::MovingAverage(state) => {
                // On the first observation, initialize directly
                if state.min_avg == 0.0 && state.max_avg == 0.0 {
                    state.min_avg = observed_min;
                    state.max_avg = observed_max;
                } else {
                    // We use a default momentum of 0.1 for the update;
                    // the caller can set this via the ObserverMode config.
                    // Here we use a simple EMA with alpha = 0.1.
                    let alpha = 0.1_f32;
                    state.min_avg = state.min_avg * (1.0 - alpha) + observed_min * alpha;
                    state.max_avg = state.max_avg * (1.0 - alpha) + observed_max * alpha;
                }
            }
        }
    }

    /// Updates the moving-average observer with a specific momentum.
    pub fn update_with_momentum(&mut self, observed_min: f32, observed_max: f32, momentum: f32) {
        if let Self::MovingAverage(state) = self {
            if state.min_avg == 0.0 && state.max_avg == 0.0 {
                state.min_avg = observed_min;
                state.max_avg = observed_max;
            } else {
                let alpha = 1.0 - momentum;
                state.min_avg = state.min_avg * momentum + observed_min * alpha;
                state.max_avg = state.max_avg * momentum + observed_max * alpha;
            }
        }
    }

    /// Computes quantization parameters from the accumulated observer state.
    pub fn compute_qparams(&self, config: &QatConfig) -> QuantParams {
        let (min, max) = match self {
            Self::MinMax(state) => (state.min, state.max),
            Self::MovingAverage(state) => (state.min_avg, state.max_avg),
        };
        QuantParams::compute_from_min_max(min, max, config)
    }
}

// ---------------------------------------------------------------------------
// FakeQuantize — PTX kernel generation
// ---------------------------------------------------------------------------

/// Block size for QAT kernels.
const QAT_BLOCK_SIZE: u32 = 256;

/// Fake quantization operator for QAT.
///
/// Generates PTX kernels for the forward (fake quantize), backward (STE), and
/// observer (min/max reduction) passes.
#[derive(Debug, Clone)]
pub struct FakeQuantize {
    config: QatConfig,
}

impl FakeQuantize {
    /// Creates a new `FakeQuantize` operator from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: QatConfig) -> DnnResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Returns a reference to the underlying configuration.
    pub fn config(&self) -> &QatConfig {
        &self.config
    }

    /// Generates PTX for the fake-quantize forward kernel.
    ///
    /// Each element is processed as:
    /// ```text
    /// q = clamp(round(x / scale) + zero_point, qmin, qmax)
    /// out = (q - zero_point) * scale
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if PTX generation fails.
    pub fn generate_fake_quantize_ptx(&self) -> DnnResult<String> {
        let qmin = self.config.bit_width.quant_min();
        let qmax = self.config.bit_width.quant_max();
        let sm = self.config.sm_version;

        let name = format!(
            "qat_fake_quantize_{:?}_{:?}",
            self.config.bit_width, self.config.symmetry
        );

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(QAT_BLOCK_SIZE)
            .param("in_ptr", PtxType::U64)
            .param("out_ptr", PtxType::U64)
            .param("scale", PtxType::F32)
            .param("zero_point", PtxType::S32)
            .param("n", PtxType::U32)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n");

                b.if_lt_u32(gid.clone(), n_reg, move |b| {
                    let in_ptr = b.load_param_u64("in_ptr");
                    let scale = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("ld.param.f32 {scale}, [param_scale];"));
                    let zp_s32 = b.alloc_reg(PtxType::S32);
                    b.raw_ptx(&format!("ld.param.s32 {zp_s32}, [param_zero_point];"));

                    // Load input
                    let addr = b.byte_offset_addr(in_ptr, gid.clone(), 4u32);
                    let x = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("ld.global.f32 {x}, [{addr}];"));

                    // q_float = round(x / scale) + zero_point
                    let div_val = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("div.rn.f32 {div_val}, {x}, {scale};"));
                    let rounded = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("cvt.rni.f32.f32 {rounded}, {div_val};"));

                    // Convert zero_point to float and add
                    let zp_f = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("cvt.rn.f32.s32 {zp_f}, {zp_s32};"));
                    let q_float = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("add.rn.f32 {q_float}, {rounded}, {zp_f};"));

                    // Clamp to [qmin, qmax]
                    let qmin_f = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!(
                        "mov.f32 {qmin_f}, 0f{:08X};",
                        (qmin as f32).to_bits()
                    ));
                    let qmax_f = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!(
                        "mov.f32 {qmax_f}, 0f{:08X};",
                        (qmax as f32).to_bits()
                    ));
                    let cl1 = b.max_f32(q_float, qmin_f);
                    let clamped = b.min_f32(cl1, qmax_f);

                    // Dequantize: out = (clamped - zp) * scale
                    let sub_val = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("sub.rn.f32 {sub_val}, {clamped}, {zp_f};"));
                    let result = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {result}, {sub_val}, {scale};"));

                    // Store
                    let out_ptr = b.load_param_u64("out_ptr");
                    let out_addr = b.byte_offset_addr(out_ptr, gid, 4u32);
                    b.raw_ptx(&format!("st.global.f32 [{out_addr}], {result};"));
                });

                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(format!("qat fake_quantize: {e}")))?;

        Ok(ptx)
    }

    /// Generates PTX for the STE backward kernel.
    ///
    /// The straight-through estimator passes gradients through only where the
    /// input was within the quantized range:
    /// ```text
    /// qmin_float = (qmin - zero_point) * scale
    /// qmax_float = (qmax - zero_point) * scale
    /// grad_out = grad_in * (x >= qmin_float && x <= qmax_float ? 1.0 : 0.0)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if PTX generation fails.
    pub fn generate_ste_backward_ptx(&self) -> DnnResult<String> {
        let sm = self.config.sm_version;

        let name = format!(
            "qat_ste_backward_{:?}_{:?}",
            self.config.bit_width, self.config.symmetry
        );

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(QAT_BLOCK_SIZE)
            .param("x_ptr", PtxType::U64)
            .param("grad_in_ptr", PtxType::U64)
            .param("grad_out_ptr", PtxType::U64)
            .param("qmin_float", PtxType::F32)
            .param("qmax_float", PtxType::F32)
            .param("n", PtxType::U32)
            .body(move |b| {
                let gid = b.global_thread_id_x();
                let n_reg = b.load_param_u32("n");

                b.if_lt_u32(gid.clone(), n_reg, move |b| {
                    let x_ptr = b.load_param_u64("x_ptr");
                    let grad_in_ptr = b.load_param_u64("grad_in_ptr");

                    let qmin_f = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("ld.param.f32 {qmin_f}, [param_qmin_float];"));
                    let qmax_f = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("ld.param.f32 {qmax_f}, [param_qmax_float];"));

                    // Load x[i]
                    let x_addr = b.byte_offset_addr(x_ptr, gid.clone(), 4u32);
                    let x_val = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("ld.global.f32 {x_val}, [{x_addr}];"));

                    // Load grad_in[i]
                    let g_addr = b.byte_offset_addr(grad_in_ptr, gid.clone(), 4u32);
                    let grad_in = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("ld.global.f32 {grad_in}, [{g_addr}];"));

                    // Check x >= qmin_float && x <= qmax_float
                    let p_ge = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.ge.f32 {p_ge}, {x_val}, {qmin_f};"));
                    let p_le = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.le.f32 {p_le}, {x_val}, {qmax_f};"));
                    let p_in_range = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("and.pred {p_in_range}, {p_ge}, {p_le};"));

                    // Select: in_range ? grad_in : 0.0
                    let zero = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.f32 {zero}, 0f00000000;"));
                    let result = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!(
                        "selp.f32 {result}, {grad_in}, {zero}, {p_in_range};"
                    ));

                    // Store
                    let grad_out_ptr = b.load_param_u64("grad_out_ptr");
                    let out_addr = b.byte_offset_addr(grad_out_ptr, gid, 4u32);
                    b.raw_ptx(&format!("st.global.f32 [{out_addr}], {result};"));
                });

                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(format!("qat ste_backward: {e}")))?;

        Ok(ptx)
    }

    /// Generates PTX for the observer reduction kernel.
    ///
    /// A parallel reduction that computes the min and max of the input tensor.
    /// The results are stored in `out[0] = min`, `out[1] = max`.
    ///
    /// # Errors
    ///
    /// Returns an error if PTX generation fails.
    pub fn generate_observer_ptx(&self) -> DnnResult<String> {
        let sm = self.config.sm_version;

        let name = format!("qat_observer_minmax_{:?}", self.config.bit_width);

        let ptx = KernelBuilder::new(&name)
            .target(sm)
            .max_threads_per_block(QAT_BLOCK_SIZE)
            .shared_mem("smem_min", PtxType::F32, QAT_BLOCK_SIZE as usize)
            .shared_mem("smem_max", PtxType::F32, QAT_BLOCK_SIZE as usize)
            .param("in_ptr", PtxType::U64)
            .param("out_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .body(move |b| {
                let tid = b.thread_id_x();
                let bdim = b.block_dim_x();
                let n_reg = b.load_param_u32("n");
                let in_ptr = b.load_param_u64("in_ptr");

                // Initialize with +INF / -INF
                let partial_min = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.f32 {partial_min}, 0f7F800000;")); // +INF
                let partial_max = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("mov.f32 {partial_max}, 0fFF800000;")); // -INF

                // Grid-stride loop
                let i = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {i}, {tid};"));

                let loop_lbl = b.fresh_label("obs_loop");
                let end_lbl = b.fresh_label("obs_end");
                b.label(&loop_lbl);
                let p_done = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {p_done}, {i}, {n_reg};"));
                b.branch_if(p_done, &end_lbl);

                let addr = b.byte_offset_addr(in_ptr.clone(), i.clone(), 4u32);
                let val = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("ld.global.f32 {val}, [{addr}];"));

                let new_min = b.min_f32(partial_min.clone(), val.clone());
                b.raw_ptx(&format!("mov.f32 {partial_min}, {new_min};"));
                let new_max = b.max_f32(partial_max.clone(), val);
                b.raw_ptx(&format!("mov.f32 {partial_max}, {new_max};"));

                b.raw_ptx(&format!("add.u32 {i}, {i}, {bdim};"));
                b.branch(&loop_lbl);
                b.label(&end_lbl);

                // Store partials to shared memory
                b.raw_ptx(&format!(
                    "st.shared.f32 [smem_min + {tid} * 4], {partial_min};"
                ));
                b.raw_ptx(&format!(
                    "st.shared.f32 [smem_max + {tid} * 4], {partial_max};"
                ));
                b.bar_sync(0);

                // Tree reduction
                let stride = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shr.u32 {stride}, {bdim}, 1;"));

                let red_loop = b.fresh_label("obs_red");
                let red_end = b.fresh_label("obs_red_end");
                b.label(&red_loop);
                let p_s = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {p_s}, {stride}, 0;"));
                b.branch_if(p_s, &red_end);

                let p_active = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lt.u32 {p_active}, {tid}, {stride};"));
                let skip = b.fresh_label("obs_skip");
                let p_not = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {p_not}, {p_active};"));
                b.branch_if(p_not, &skip);

                let other = b.add_u32(tid.clone(), stride.clone());
                let a_min = b.alloc_reg(PtxType::F32);
                let b_min = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("ld.shared.f32 {a_min}, [smem_min + {tid} * 4];"));
                b.raw_ptx(&format!("ld.shared.f32 {b_min}, [smem_min + {other} * 4];"));
                let m_min = b.min_f32(a_min, b_min);
                b.raw_ptx(&format!("st.shared.f32 [smem_min + {tid} * 4], {m_min};"));

                let a_max = b.alloc_reg(PtxType::F32);
                let b_max = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("ld.shared.f32 {a_max}, [smem_max + {tid} * 4];"));
                b.raw_ptx(&format!("ld.shared.f32 {b_max}, [smem_max + {other} * 4];"));
                let m_max = b.max_f32(a_max, b_max);
                b.raw_ptx(&format!("st.shared.f32 [smem_max + {tid} * 4], {m_max};"));

                b.label(&skip);
                b.bar_sync(0);
                b.raw_ptx(&format!("shr.u32 {stride}, {stride}, 1;"));
                b.branch(&red_loop);
                b.label(&red_end);

                // Thread 0 writes out the result
                let p_t0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.eq.u32 {p_t0}, {tid}, 0;"));
                let skip_w = b.fresh_label("obs_skip_w");
                let p_not_t0 = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("not.pred {p_not_t0}, {p_t0};"));
                b.branch_if(p_not_t0, &skip_w);

                let final_min = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("ld.shared.f32 {final_min}, [smem_min];"));
                let final_max = b.alloc_reg(PtxType::F32);
                b.raw_ptx(&format!("ld.shared.f32 {final_max}, [smem_max];"));

                let out_ptr = b.load_param_u64("out_ptr");
                b.raw_ptx(&format!("st.global.f32 [{out_ptr}], {final_min};"));
                let out_off = b.alloc_reg(PtxType::U64);
                b.raw_ptx(&format!("add.u64 {out_off}, {out_ptr}, 4;"));
                b.raw_ptx(&format!("st.global.f32 [{out_off}], {final_max};"));

                b.label(&skip_w);
                b.ret();
            })
            .build()
            .map_err(|e| DnnError::PtxGeneration(format!("qat observer: {e}")))?;

        Ok(ptx)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> QatConfig {
        QatConfig {
            bit_width: QatBitWidth::Int8,
            symmetry: QatSymmetry::Symmetric,
            granularity: QatGranularity::PerTensor,
            observer: ObserverMode::MinMax,
            sm_version: SmVersion::Sm80,
            float_type: PtxType::F32,
        }
    }

    // -- QatBitWidth --

    #[test]
    fn int4_range() {
        assert_eq!(QatBitWidth::Int4.quant_min(), -8);
        assert_eq!(QatBitWidth::Int4.quant_max(), 7);
        assert_eq!(QatBitWidth::Int4.num_levels(), 16);
    }

    #[test]
    fn int8_range() {
        assert_eq!(QatBitWidth::Int8.quant_min(), -128);
        assert_eq!(QatBitWidth::Int8.quant_max(), 127);
        assert_eq!(QatBitWidth::Int8.num_levels(), 256);
    }

    // -- Symmetric qparams --

    #[test]
    fn symmetric_qparams_positive_range() {
        let config = default_config();
        let params = QuantParams::compute_from_min_max(-1.0, 1.0, &config);
        assert_eq!(params.zero_point, 0);
        let expected_scale = 1.0 / 127.0;
        assert!((params.scale - expected_scale).abs() < 1e-6);
    }

    #[test]
    fn symmetric_qparams_asymmetric_input() {
        let config = default_config();
        let params = QuantParams::compute_from_min_max(-0.5, 2.0, &config);
        assert_eq!(params.zero_point, 0);
        // scale = max(0.5, 2.0) / 127 = 2.0 / 127
        let expected_scale = 2.0 / 127.0;
        assert!((params.scale - expected_scale).abs() < 1e-6);
    }

    // -- Asymmetric qparams --

    #[test]
    fn asymmetric_qparams() {
        let mut config = default_config();
        config.symmetry = QatSymmetry::Asymmetric;
        let params = QuantParams::compute_from_min_max(0.0, 1.0, &config);
        // scale = (1.0 - 0.0) / (127 - (-128)) = 1.0 / 255
        let expected_scale = 1.0 / 255.0;
        assert!((params.scale - expected_scale).abs() < 1e-5);
        // zero_point = round(-128 - 0.0/scale) = -128
        assert_eq!(params.zero_point, -128);
    }

    // -- Observer state --

    #[test]
    fn observer_minmax_update() {
        let mut obs = ObserverState::new(&ObserverMode::MinMax);
        obs.update(-1.0, 2.0);
        obs.update(-3.0, 1.5);
        if let ObserverState::MinMax(ref state) = obs {
            assert_eq!(state.min, -3.0);
            assert_eq!(state.max, 2.0);
            assert_eq!(state.num_batches, 2);
        } else {
            panic!("Expected MinMax state");
        }
    }

    #[test]
    fn observer_moving_average() {
        let mut obs = ObserverState::new(&ObserverMode::MovingAverage { momentum: 0.9 });
        obs.update_with_momentum(-1.0, 1.0, 0.9);
        // First update initializes directly
        if let ObserverState::MovingAverage(ref state) = obs {
            assert!((state.min_avg - (-1.0)).abs() < 1e-6);
            assert!((state.max_avg - 1.0).abs() < 1e-6);
        }
        obs.update_with_momentum(-2.0, 3.0, 0.9);
        // Second: min = -1.0 * 0.9 + -2.0 * 0.1 = -1.1
        // max = 1.0 * 0.9 + 3.0 * 0.1 = 1.2
        if let ObserverState::MovingAverage(ref state) = obs {
            assert!((state.min_avg - (-1.1)).abs() < 1e-5);
            assert!((state.max_avg - 1.2).abs() < 1e-5);
        }
    }

    #[test]
    fn observer_compute_qparams() {
        let mut obs = ObserverState::new(&ObserverMode::MinMax);
        obs.update(-1.0, 1.0);
        let config = default_config();
        let params = obs.compute_qparams(&config);
        assert_eq!(params.zero_point, 0);
        assert!((params.scale - 1.0 / 127.0).abs() < 1e-6);
    }

    // -- Config validation --

    #[test]
    fn config_valid() {
        let config = default_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_invalid_float_type() {
        let mut config = default_config();
        config.float_type = PtxType::U32;
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_invalid_momentum() {
        let mut config = default_config();
        config.observer = ObserverMode::MovingAverage { momentum: 1.5 };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_invalid_percentile() {
        let mut config = default_config();
        config.observer = ObserverMode::Percentile { percentile: 0.0 };
        assert!(config.validate().is_err());
    }

    // -- PTX generation --

    #[test]
    fn fake_quantize_ptx_generates() {
        let fq = FakeQuantize::new(default_config());
        assert!(fq.is_ok());
        let fq = fq.unwrap();
        let ptx = fq.generate_fake_quantize_ptx();
        assert!(ptx.is_ok());
        let ptx_str = ptx.unwrap();
        assert!(ptx_str.contains("qat_fake_quantize"));
        assert!(ptx_str.contains(".entry"));
    }

    #[test]
    fn ste_backward_ptx_generates() {
        let fq = FakeQuantize::new(default_config());
        assert!(fq.is_ok());
        let ptx = fq.unwrap().generate_ste_backward_ptx();
        assert!(ptx.is_ok());
        let ptx_str = ptx.unwrap();
        assert!(ptx_str.contains("qat_ste_backward"));
        assert!(ptx_str.contains("selp.f32"));
    }

    #[test]
    fn observer_ptx_generates() {
        let fq = FakeQuantize::new(default_config());
        assert!(fq.is_ok());
        let ptx = fq.unwrap().generate_observer_ptx();
        assert!(ptx.is_ok());
        let ptx_str = ptx.unwrap();
        assert!(ptx_str.contains("qat_observer_minmax"));
        assert!(ptx_str.contains("smem_min"));
    }

    // -- Per-channel granularity --

    #[test]
    fn per_channel_config_valid() {
        let mut config = default_config();
        config.granularity = QatGranularity::PerChannel { axis: 0 };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn per_channel_large_axis_rejected() {
        let mut config = default_config();
        config.granularity = QatGranularity::PerChannel { axis: 2000 };
        assert!(config.validate().is_err());
    }

    // -- Edge cases --

    #[test]
    fn zero_range_qparams() {
        let config = default_config();
        let params = QuantParams::compute_from_min_max(5.0, 5.0, &config);
        // Should not produce scale = 0
        assert!(params.scale > 0.0);
    }

    #[test]
    fn negative_only_qparams() {
        let config = default_config();
        let params = QuantParams::compute_from_min_max(-10.0, -1.0, &config);
        assert_eq!(params.zero_point, 0);
        // scale = max(10.0, 1.0) / 127 = 10.0 / 127
        let expected = 10.0 / 127.0;
        assert!((params.scale - expected).abs() < 1e-6);
    }

    // -- Int4 PTX generation --

    #[test]
    fn int4_fake_quantize_ptx() {
        let mut config = default_config();
        config.bit_width = QatBitWidth::Int4;
        let fq = FakeQuantize::new(config);
        assert!(fq.is_ok());
        let ptx = fq.unwrap().generate_fake_quantize_ptx();
        assert!(ptx.is_ok());
        let ptx_str = ptx.unwrap();
        assert!(ptx_str.contains("Int4"));
    }

    // -- Asymmetric PTX generation --

    #[test]
    fn asymmetric_fake_quantize_ptx() {
        let mut config = default_config();
        config.symmetry = QatSymmetry::Asymmetric;
        let fq = FakeQuantize::new(config);
        assert!(fq.is_ok());
        let ptx = fq.unwrap().generate_fake_quantize_ptx();
        assert!(ptx.is_ok());
        let ptx_str = ptx.unwrap();
        assert!(ptx_str.contains("Asymmetric"));
    }
}
