//! Metal GPU FFT — Cooley-Tukey radix-2 DIT FFT executed on Apple Metal.
//!
//! Provides [`MetalFftPlan`] for planning and executing FFTs over power-of-2
//! sizes, and [`MetalFftBuffer`] as a convenient host-side buffer type.
//!
//! On non-macOS platforms every operation returns
//! [`MetalError::UnsupportedPlatform`] — the crate still compiles cleanly.

use crate::error::{MetalError, MetalResult};
use num_complex::Complex;

// ─── MSL Shader Source ────────────────────────────────────────────────────────

/// MSL source for the radix-2 DIT FFT kernels.
///
/// Two entry points are provided:
/// - `fft_butterfly`: one Cooley-Tukey butterfly stage in-place.
/// - `bit_reverse`: bit-reversal permutation (input → separate output buffer).
///
/// `inverse` is passed as a `uint` (0 = forward, 1 = inverse) to avoid
/// Metal `bool` ABI ambiguity when using `set_bytes`.
#[cfg(target_os = "macos")]
const FFT_MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Complex {
    float re;
    float im;
};

static Complex cmul(Complex a, Complex b) {
    return Complex{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
static Complex cadd(Complex a, Complex b) {
    return Complex{a.re + b.re, a.im + b.im};
}
static Complex csub(Complex a, Complex b) {
    return Complex{a.re - b.re, a.im - b.im};
}

/// Cooley-Tukey butterfly stage, batched over `gid.y`.  Called once per
/// stage with `stage` in [0, log2(n)).  Each thread handles one butterfly
/// pair of ONE batch row, selected by `gid.y`; `gid.x` is the pair index
/// within that row exactly as in the single-transform kernel.  Folding the
/// batch dimension into the grid lets one command-buffer submission cover
/// every batch row instead of looping and resubmitting per row on the CPU.
///
/// `twiddles` is a precomputed base table of `n/2` FORWARD twiddle factors
/// W_n^j = exp(-2πi·j/n) for j in [0, n/2), uploaded once at plan creation
/// (computed host-side in f64 and rounded to f32 — see
/// `MetalFftPlan::new_macos`). The twiddle needed for a butterfly of size
/// `butterfly_size` at position `pair` is W_n^{pair·(n/butterfly_size)} —
/// the standard stride-indexed table lookup — so no per-thread `cos`/`sin`
/// is evaluated on the GPU. `inverse` (0 = forward, 1 = inverse) negates
/// the imaginary part to obtain the conjugate (inverse) twiddle from the
/// same forward table.
kernel void fft_butterfly(
    device Complex* data              [[ buffer(0) ]],
    constant uint& stage               [[ buffer(1) ]],
    constant uint& n                   [[ buffer(2) ]],
    constant uint& inverse             [[ buffer(3) ]],
    device const Complex* twiddles     [[ buffer(4) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint tid            = gid.x;
    uint batch_row      = gid.y;
    uint butterfly_size = 1u << (stage + 1u);
    uint half_size      = butterfly_size >> 1u;
    uint group          = tid / half_size;
    uint pair           = tid % half_size;
    uint i              = group * butterfly_size + pair;
    uint j              = i + half_size;

    if (j >= n) return;

    uint stride = n / butterfly_size;
    Complex w = twiddles[pair * stride];
    if (inverse != 0u) {
        w.im = -w.im;
    }

    device Complex* row = data + batch_row * n;
    Complex u = row[i];
    Complex t = cmul(w, row[j]);
    row[i]    = cadd(u, t);
    row[j]    = csub(u, t);
}

/// Bit-reversal permutation, batched over `gid.y`: reads from `input`,
/// writes to `output`.  `log2n` is the bit width (e.g. 10 for n=1024);
/// `n = 1 << log2n` is derived in-kernel so no extra buffer is needed to
/// carry the batch row stride.
kernel void bit_reverse(
    device const Complex* input  [[ buffer(0) ]],
    device Complex*       output [[ buffer(1) ]],
    constant uint&        log2n  [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint n         = 1u << log2n;
    uint tid       = gid.x;
    uint batch_row = gid.y;
    uint rev = 0u;
    uint idx = tid;
    for (uint i = 0u; i < log2n; i++) {
        rev = (rev << 1u) | (idx & 1u);
        idx >>= 1u;
    }
    output[batch_row * n + rev] = input[batch_row * n + tid];
}
"#;

// ─── Public types ─────────────────────────────────────────────────────────────

/// FFT transform direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalFftDirection {
    /// Forward DFT: X\[k\] = Σ x\[n\]·e^{-2πi·kn/N}
    Forward,
    /// Inverse DFT (normalised by 1/N): x\[n\] = (1/N) Σ X\[k\]·e^{2πi·kn/N}
    Inverse,
}

/// A Metal FFT plan for a fixed size and batch count.
///
/// Create with [`MetalFftPlan::new`], then call [`MetalFftPlan::execute`]
/// to run the transform on Apple GPU hardware.
///
/// The MSL shaders are compiled **once at creation time** and the resulting
/// `ComputePipelineState` objects are cached in the plan.  Successive calls to
/// [`execute`][MetalFftPlan::execute] reuse the cached pipelines, eliminating
/// the 100 ms+ per-call shader compilation overhead.
///
/// Each call to [`execute`][MetalFftPlan::execute] encodes the bit-reversal
/// pass and every butterfly stage for **all** batch rows into a single
/// command buffer and performs exactly one `commit` + `wait_until_completed`
/// — the batch dimension is folded into the dispatch grid (`gid.y`) rather
/// than looped on the CPU side, so cost no longer scales with `batch` in
/// the number of CPU↔GPU round trips.
///
/// # Examples
///
/// ```rust,no_run
/// use oxicuda_metal::fft::{MetalFftDirection, MetalFftPlan};
/// use num_complex::Complex;
///
/// let plan = MetalFftPlan::new(1024, 1).expect("valid n and batch");
/// let input: Vec<Complex<f32>> = (0..1024)
///     .map(|i| Complex::new(i as f32, 0.0))
///     .collect();
/// let mut output = vec![Complex::new(0.0f32, 0.0); 1024];
/// plan.execute(&input, &mut output, MetalFftDirection::Forward).expect("execute should succeed");
/// ```
pub struct MetalFftPlan {
    /// FFT size — always a power of 2.
    n: usize,
    /// log₂(n).
    log2n: u32,
    /// Number of transforms to execute in a single call.
    batch: usize,
    /// Metal device — cached for buffer allocation in `execute`.
    #[cfg(target_os = "macos")]
    device: metal::Device,
    /// Command queue — created once, reused across calls.
    #[cfg(target_os = "macos")]
    command_queue: metal::CommandQueue,
    /// Compiled compute pipeline for the butterfly stage (cached).
    #[cfg(target_os = "macos")]
    butterfly_pipeline: metal::ComputePipelineState,
    /// Compiled compute pipeline for the bit-reversal permutation (cached).
    #[cfg(target_os = "macos")]
    bit_reverse_pipeline: metal::ComputePipelineState,
    /// Precomputed base table of `n/2` forward twiddle factors
    /// (`W_n^j = exp(-2πi·j/n)`), computed once in f64 and rounded to f32
    /// at plan creation (see [`MetalFftPlan::new_macos`]).
    ///
    /// Written exactly once, at construction, and never mutated again — it
    /// is safe to share read-only across concurrent [`execute`][MetalFftPlan::execute]
    /// calls on `&self`. Unlike this field, the per-call input/output
    /// scratch buffers are deliberately NOT stored here: `execute` takes
    /// `&self`, so plan-owned *mutable* scratch would race under concurrent
    /// calls without additional locking. Keep it that way — see
    /// `execute_macos`, which allocates its input/output buffers fresh
    /// (but only once each, not once per batch element) on every call.
    #[cfg(target_os = "macos")]
    twiddle_buffer: metal::Buffer,
}

impl std::fmt::Debug for MetalFftPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalFftPlan")
            .field("n", &self.n)
            .field("log2n", &self.log2n)
            .field("batch", &self.batch)
            .finish_non_exhaustive()
    }
}

impl MetalFftPlan {
    /// Create a new FFT plan, compiling the MSL shaders eagerly.
    ///
    /// On macOS this acquires a Metal device, compiles the FFT shaders, and
    /// creates the compute pipeline states — all at construction time so that
    /// subsequent [`execute`][MetalFftPlan::execute] calls pay no compilation cost.
    ///
    /// # Errors
    ///
    /// Returns [`MetalError::InvalidArgument`] when:
    /// - `n == 0`
    /// - `n` is not a power of 2
    /// - `batch == 0`
    ///
    /// On macOS, also returns [`MetalError::NoDevice`],
    /// [`MetalError::ShaderCompilation`], or [`MetalError::PipelineCreation`]
    /// if Metal initialisation fails.
    pub fn new(n: usize, batch: usize) -> MetalResult<Self> {
        if n == 0 {
            return Err(MetalError::InvalidArgument("n must be > 0".into()));
        }
        if !n.is_power_of_two() {
            return Err(MetalError::InvalidArgument(format!(
                "n ({n}) must be a power of 2"
            )));
        }
        if batch == 0 {
            return Err(MetalError::InvalidArgument("batch must be > 0".into()));
        }
        let log2n = n.trailing_zeros();

        #[cfg(target_os = "macos")]
        {
            Self::new_macos(n, log2n, batch)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self { n, log2n, batch })
        }
    }

    /// Execute the FFT.
    ///
    /// - `input`  — `batch * n` complex samples (row-major, batch-first).
    /// - `output` — must have the same length as `input`.
    /// - `direction` — [`MetalFftDirection::Forward`] or [`MetalFftDirection::Inverse`].
    ///
    /// On macOS the transform is dispatched to the GPU via Metal using
    /// pre-compiled pipeline states (no shader recompilation per call).
    /// On other platforms returns [`MetalError::UnsupportedPlatform`].
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer sizes are wrong, the Metal device is
    /// unavailable, or a command-buffer error occurs.
    pub fn execute(
        &self,
        input: &[Complex<f32>],
        output: &mut [Complex<f32>],
        direction: MetalFftDirection,
    ) -> MetalResult<()> {
        let expected = self
            .n
            .checked_mul(self.batch)
            .ok_or_else(|| MetalError::InvalidArgument("batch * n overflows".into()))?;
        if input.len() != expected {
            return Err(MetalError::InvalidArgument(format!(
                "input length {} != batch({}) * n({})",
                input.len(),
                self.batch,
                self.n
            )));
        }
        if output.len() != expected {
            return Err(MetalError::InvalidArgument(format!(
                "output length {} != batch({}) * n({})",
                output.len(),
                self.batch,
                self.n
            )));
        }

        #[cfg(target_os = "macos")]
        {
            self.execute_macos(input, output, direction)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (input, output, direction);
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// FFT size.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// log₂(n).
    #[inline]
    pub fn log2n(&self) -> u32 {
        self.log2n
    }

    /// Batch count.
    #[inline]
    pub fn batch(&self) -> usize {
        self.batch
    }
}

// ─── macOS GPU dispatch ───────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
impl MetalFftPlan {
    /// Initialise the plan on macOS: acquire a Metal device, compile the MSL
    /// library, and create the two compute pipeline states.  Called once from
    /// [`MetalFftPlan::new`]; pipelines are then cached for the plan's lifetime.
    fn new_macos(n: usize, log2n: u32, batch: usize) -> MetalResult<Self> {
        use crate::device::MetalDevice;
        use metal::MTLResourceOptions;

        let metal_device = MetalDevice::new()?;
        let device = metal_device.device;

        // ── Compile the MSL library once — at plan creation time.
        let compile_opts = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(FFT_MSL_SOURCE, &compile_opts)
            .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

        // Retrieve both kernel functions from the library.
        let fn_bit_reverse = library
            .get_function("bit_reverse", None)
            .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;
        let fn_butterfly = library
            .get_function("fft_butterfly", None)
            .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

        // Build and cache compute pipeline states.
        let bit_reverse_pipeline = device
            .new_compute_pipeline_state_with_function(&fn_bit_reverse)
            .map_err(|e| MetalError::PipelineCreation(e.to_string()))?;
        let butterfly_pipeline = device
            .new_compute_pipeline_state_with_function(&fn_butterfly)
            .map_err(|e| MetalError::PipelineCreation(e.to_string()))?;

        // Create the command queue once; it is reused across execute() calls.
        let command_queue = device.new_command_queue();

        // ── Precompute the twiddle-factor table ONCE, in f64, rounded to f32.
        //
        // Stage `s` (butterfly_size = 2^(s+1)) needs W_{butterfly_size}^{pair}
        // for pair in [0, butterfly_size/2). Since
        // W_{butterfly_size}^{pair} = W_n^{pair·(n/butterfly_size)}, a single
        // base table of n/2 entries W_n^{j} = exp(-2πi·j/n), j in [0, n/2),
        // covers every stage via the stride `n / butterfly_size` computed
        // in-kernel (see `fft_butterfly` above) — no per-thread `cos`/`sin`.
        // Computing the angle in f64 and rounding only the final (cos, sin)
        // pair to f32 avoids accumulating f32 rounding error in the angle
        // itself, which is the classic source of FFT accuracy loss at large
        // n when the angle is formed directly in f32 on the GPU.
        let half_n = n / 2;
        let mut twiddle_host: Vec<Complex<f32>> = Vec::with_capacity(half_n.max(1));
        for j in 0..half_n {
            let theta = -2.0_f64 * std::f64::consts::PI * (j as f64) / (n as f64);
            let (sin_theta, cos_theta) = theta.sin_cos();
            twiddle_host.push(Complex::new(cos_theta as f32, sin_theta as f32));
        }
        if twiddle_host.is_empty() {
            // n == 1: log2n == 0, so `fft_butterfly` is never dispatched and
            // this entry is never read — but a zero-length Metal buffer is
            // avoided by keeping the allocation non-empty regardless.
            twiddle_host.push(Complex::new(0.0, 0.0));
        }
        let twiddle_bytes = std::mem::size_of_val(twiddle_host.as_slice()) as u64;
        // SAFETY: `twiddle_host` is a valid, fully-initialised `Vec<Complex<f32>>`;
        // `twiddle_bytes` is exactly its byte length (`size_of_val`), matching
        // the MSL `Complex{float re, im}` layout (two adjacent f32 values, no
        // padding) used by the kernel.
        let twiddle_buffer = device.new_buffer_with_data(
            twiddle_host.as_ptr().cast::<std::ffi::c_void>(),
            twiddle_bytes,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            n,
            log2n,
            batch,
            device,
            command_queue,
            butterfly_pipeline,
            bit_reverse_pipeline,
            twiddle_buffer,
        })
    }

    /// Core GPU dispatch — only compiled on macOS.
    ///
    /// Uses the pre-compiled [`Self::butterfly_pipeline`] and
    /// [`Self::bit_reverse_pipeline`] cached at construction; no shader
    /// recompilation occurs here.
    fn execute_macos(
        &self,
        input: &[Complex<f32>],
        output: &mut [Complex<f32>],
        direction: MetalFftDirection,
    ) -> MetalResult<()> {
        use metal::{MTLResourceOptions, MTLSize};
        use std::mem::size_of;

        let n = self.n;
        let log2n = self.log2n;
        let batch = self.batch;
        // `execute()` already validated `input.len() == output.len() ==
        // batch * n` (including the overflow check), so both are exact —
        // and, since the lengths match, so are their byte sizes.
        let total = input.len();
        let total_bytes = std::mem::size_of_val(input) as u64;
        let inverse_flag: u32 = if direction == MetalFftDirection::Inverse {
            1
        } else {
            0
        };

        // ── Allocate the whole-batch buffers ONCE for this call — not once
        // per batch element as before. One host→device copy seeds the
        // input buffer for every row; the output buffer is a single fresh
        // allocation sized for the whole batch.
        //
        // SAFETY: `input` is a valid slice of exactly `total` elements;
        // `total_bytes` is exactly its byte length (`size_of_val`), which
        // is therefore the correct length of the region pointed to by
        // `input.as_ptr()`.
        let buf_input = self.device.new_buffer_with_data(
            input.as_ptr().cast::<std::ffi::c_void>(),
            total_bytes,
            MTLResourceOptions::StorageModeShared,
        );

        // Allocate an output buffer sized for the whole batch (contents
        // initialised by Metal; bit_reverse writes every element of every row).
        let buf_output = self
            .device
            .new_buffer(total_bytes, MTLResourceOptions::StorageModeShared);

        // ── Single command buffer for the ENTIRE batch ────────────────────
        //
        // The whole transform — bit-reversal plus every butterfly stage,
        // across every batch row — is encoded into ONE command buffer with
        // a separate serial compute encoder per pass. Metal automatically
        // tracks read/write hazards between successive serial encoders in a
        // command buffer (and executes them in encode order on a serial
        // queue), so the in-place stages stay correctly ordered.  Each pass
        // is dispatched over a 2-D grid (row width × `batch`) so the batch
        // dimension is folded into `gid.y` in the shader instead of being a
        // CPU-side loop that re-encodes and resubmits per row — we commit +
        // wait exactly ONCE per `execute()` call, regardless of `batch`.
        let cmd_buf = self.command_queue.new_command_buffer();

        // Pass 0: bit-reversal permutation (buf_input → buf_output), all rows.
        {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.bit_reverse_pipeline);
            // buffer(0) = input (read-only in shader)
            encoder.set_buffer(0, Some(&buf_input), 0);
            // buffer(1) = output (write-only in shader)
            encoder.set_buffer(1, Some(&buf_output), 0);
            // buffer(2) = log2n
            let log2n_val = log2n;
            // SAFETY: &log2n_val is a valid u32 reference; we pass its
            // size correctly.
            encoder.set_bytes(
                2,
                size_of::<u32>() as u64,
                (&log2n_val as *const u32).cast::<std::ffi::c_void>(),
            );

            let tg_size = determine_threadgroup_size(
                self.bit_reverse_pipeline
                    .max_total_threads_per_threadgroup(),
                n as u64,
            );
            encoder.dispatch_threads(
                MTLSize {
                    width: n as u64,
                    height: batch as u64,
                    depth: 1,
                },
                MTLSize {
                    width: tg_size,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
        }

        // After bit_reverse the data lives in buf_output.
        // fft_butterfly operates in-place on buf_output.

        // ── Butterfly stages ───────────────────────────────────────────────
        //
        // Each stage is dispatched as n/2 threads per row (one per butterfly
        // pair) × `batch` rows. `stage_vals` keeps every per-stage constant
        // alive until commit, since `set_bytes` copies at encode time but the
        // encoders are only submitted when the shared command buffer is
        // committed below.
        let half_n = (n / 2) as u64;
        let n_val = n as u32;
        let stage_vals: Vec<u32> = (0..log2n).collect();

        for &stage_val in &stage_vals {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.butterfly_pipeline);
            // buffer(0) = data (in-place)
            encoder.set_buffer(0, Some(&buf_output), 0);
            // buffer(1) = stage index
            // SAFETY: pointer is valid for the byte size passed.
            encoder.set_bytes(
                1,
                size_of::<u32>() as u64,
                (&stage_val as *const u32).cast::<std::ffi::c_void>(),
            );
            // buffer(2) = n
            // SAFETY: pointer is valid for the byte size passed.
            encoder.set_bytes(
                2,
                size_of::<u32>() as u64,
                (&n_val as *const u32).cast::<std::ffi::c_void>(),
            );
            // buffer(3) = inverse flag (0 or 1)
            // SAFETY: pointer is valid for the byte size passed.
            encoder.set_bytes(
                3,
                size_of::<u32>() as u64,
                (&inverse_flag as *const u32).cast::<std::ffi::c_void>(),
            );
            // buffer(4) = precomputed twiddle-factor table (read-only, plan-owned)
            encoder.set_buffer(4, Some(&self.twiddle_buffer), 0);

            let tg_size = determine_threadgroup_size(
                self.butterfly_pipeline.max_total_threads_per_threadgroup(),
                half_n,
            );
            encoder.dispatch_threads(
                MTLSize {
                    width: half_n,
                    height: batch as u64,
                    depth: 1,
                },
                MTLSize {
                    width: tg_size,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
        }

        // Submit the whole batch's transform once and wait for completion —
        // exactly one commit + wait per `execute()` call, regardless of batch.
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        check_command_buffer_status(cmd_buf)?;

        // ── Read back GPU results — one copy for the whole batch ──────────
        //
        // SAFETY: `buf_output` is a shared-mode Metal buffer; `contents()`
        // returns a valid CPU-accessible pointer covering the full buffer
        // length `total_bytes` bytes.  We cast to `*const Complex<f32>`,
        // which has the same memory layout as the MSL `Complex{float re, im}`
        // struct (two adjacent f32 values, no padding).
        unsafe {
            let src_ptr = buf_output.contents().cast::<Complex<f32>>();
            std::ptr::copy_nonoverlapping(src_ptr, output.as_mut_ptr(), total);
        }

        // ── Inverse-transform normalisation (1/N), over the whole batch ───
        if direction == MetalFftDirection::Inverse {
            let norm = 1.0_f32 / n as f32;
            for elem in output.iter_mut() {
                *elem = Complex::new(elem.re * norm, elem.im * norm);
            }
        }

        Ok(())
    }
}

// ─── Helper functions (macOS only) ───────────────────────────────────────────

/// Choose the largest threadgroup size that is ≤ both `max_tg` and `work_items`.
/// Falls back to 1 if both are 0.
#[cfg(target_os = "macos")]
fn determine_threadgroup_size(max_tg: u64, work_items: u64) -> u64 {
    if max_tg == 0 || work_items == 0 {
        return 1;
    }
    // Round `max_tg` down to the largest power of 2 that is ≤ `work_items`.
    //
    // `next_power_of_two()` is the IDENTITY when its input is already a
    // power of two, so a naive `next_power_of_two() >> 1` HALVES `capped`
    // even when `capped` needed no rounding at all (e.g. capped=512 would
    // wrongly become 256). Only shift down when `capped` is not already a
    // power of two.
    let capped = max_tg.min(work_items);
    if capped.is_power_of_two() {
        capped
    } else {
        (capped.next_power_of_two() >> 1).max(1)
    }
}

/// Return an error if the command buffer completed with an error status.
#[cfg(target_os = "macos")]
fn check_command_buffer_status(cmd_buf: &metal::CommandBufferRef) -> MetalResult<()> {
    use metal::MTLCommandBufferStatus;
    match cmd_buf.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        status => Err(MetalError::CommandBufferError(format!(
            "command buffer finished with status: {status:?}"
        ))),
    }
}

// ─── MetalFftBuffer ───────────────────────────────────────────────────────────

/// A host-side complex buffer sized for `batch * n` elements.
///
/// Convenience type for preparing input and collecting output for
/// [`MetalFftPlan::execute`].
pub struct MetalFftBuffer {
    data: Vec<Complex<f32>>,
}

impl MetalFftBuffer {
    /// Allocate a zero-initialised buffer for `batch * n` complex f32 values.
    pub fn new(n: usize, batch: usize) -> Self {
        let len = n.saturating_mul(batch);
        Self {
            data: vec![Complex::new(0.0, 0.0); len],
        }
    }

    /// View as an immutable slice of complex values.
    pub fn as_slice(&self) -> &[Complex<f32>] {
        &self.data
    }

    /// View as a mutable slice of complex values.
    pub fn as_mut_slice(&mut self) -> &mut [Complex<f32>] {
        &mut self.data
    }

    /// Total number of complex elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl std::fmt::Debug for MetalFftBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalFftBuffer(len={})", self.data.len())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MetalFftPlan::new ─────────────────────────────────────────────────────

    #[test]
    fn plan_new_valid() {
        let plan = MetalFftPlan::new(1024, 2).expect("valid plan");
        assert_eq!(plan.n(), 1024);
        assert_eq!(plan.log2n(), 10);
        assert_eq!(plan.batch(), 2);
    }

    #[test]
    fn plan_new_zero_n_errors() {
        assert!(matches!(
            MetalFftPlan::new(0, 1),
            Err(MetalError::InvalidArgument(_))
        ));
    }

    #[test]
    fn plan_new_non_power_of_two_errors() {
        assert!(matches!(
            MetalFftPlan::new(1000, 1),
            Err(MetalError::InvalidArgument(_))
        ));
    }

    #[test]
    fn plan_new_zero_batch_errors() {
        assert!(matches!(
            MetalFftPlan::new(64, 0),
            Err(MetalError::InvalidArgument(_))
        ));
    }

    #[test]
    fn plan_new_n_equals_one() {
        let plan = MetalFftPlan::new(1, 1).expect("n=1 is power of 2");
        assert_eq!(plan.n(), 1);
        assert_eq!(plan.log2n(), 0);
    }

    // ── Debug impl ───────────────────────────────────────────────────────────

    #[test]
    fn plan_debug_contains_fields() {
        // Validation-only test — plan_new_n_equals_one already covers n=1 which
        // is cheapest to compile on macOS.  On non-macOS new() is always cheap.
        // We only call debug formatting here; execution is not needed.
        //
        // On macOS this WILL compile the Metal shaders. Accept a NoDevice error
        // gracefully so CI without a GPU still passes.
        match MetalFftPlan::new(8, 1) {
            Ok(plan) => {
                let s = format!("{plan:?}");
                assert!(s.contains("MetalFftPlan"), "debug: {s}");
                assert!(s.contains('8') || s.contains("n:"), "debug: {s}");
            }
            #[cfg(target_os = "macos")]
            Err(MetalError::NoDevice) => {
                // No GPU on CI — acceptable.
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    // ── MetalFftPlan::execute input validation ────────────────────────────────

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn execute_wrong_input_length() {
        let plan = MetalFftPlan::new(8, 1)
            .expect("FFT plan creation should succeed for valid power-of-2 size");
        let input = vec![Complex::new(0.0f32, 0.0); 7]; // wrong: should be 8
        let mut output = vec![Complex::new(0.0f32, 0.0); 8];
        assert!(matches!(
            plan.execute(&input, &mut output, MetalFftDirection::Forward),
            Err(MetalError::InvalidArgument(_))
        ));
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn execute_wrong_output_length() {
        let plan = MetalFftPlan::new(8, 1)
            .expect("FFT plan creation should succeed for valid power-of-2 size");
        let input = vec![Complex::new(0.0f32, 0.0); 8];
        let mut output = vec![Complex::new(0.0f32, 0.0); 7]; // wrong: should be 8
        assert!(matches!(
            plan.execute(&input, &mut output, MetalFftDirection::Forward),
            Err(MetalError::InvalidArgument(_))
        ));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn execute_wrong_input_length_macos() {
        match MetalFftPlan::new(8, 1) {
            Ok(plan) => {
                let input = vec![Complex::new(0.0f32, 0.0); 7]; // wrong: should be 8
                let mut output = vec![Complex::new(0.0f32, 0.0); 8];
                assert!(matches!(
                    plan.execute(&input, &mut output, MetalFftDirection::Forward),
                    Err(MetalError::InvalidArgument(_))
                ));
            }
            Err(MetalError::NoDevice) => {} // CI without GPU
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn execute_wrong_output_length_macos() {
        match MetalFftPlan::new(8, 1) {
            Ok(plan) => {
                let input = vec![Complex::new(0.0f32, 0.0); 8];
                let mut output = vec![Complex::new(0.0f32, 0.0); 7]; // wrong: should be 8
                assert!(matches!(
                    plan.execute(&input, &mut output, MetalFftDirection::Forward),
                    Err(MetalError::InvalidArgument(_))
                ));
            }
            Err(MetalError::NoDevice) => {} // CI without GPU
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    // ── Platform-specific execution tests ────────────────────────────────────

    #[test]
    #[cfg(target_os = "macos")]
    fn execute_impulse_response_forward() {
        // x = [1, 0, 0, ..., 0] → X[k] = 1 for all k
        let n = 8usize;
        let plan = match MetalFftPlan::new(n, 1) {
            Ok(p) => p,
            Err(MetalError::NoDevice) => return, // no GPU on CI
            Err(e) => panic!("plan creation error: {e}"),
        };
        let mut input = vec![Complex::new(0.0f32, 0.0); n];
        input[0] = Complex::new(1.0, 0.0);
        let mut output = vec![Complex::new(0.0f32, 0.0); n];

        if plan
            .execute(&input, &mut output, MetalFftDirection::Forward)
            .is_ok()
        {
            // All output bins should equal 1+0i within floating-point tolerance.
            for (i, c) in output.iter().enumerate() {
                assert!(
                    (c.re - 1.0).abs() < 1e-4,
                    "output[{i}].re = {} expected ~1.0",
                    c.re
                );
                assert!(c.im.abs() < 1e-4, "output[{i}].im = {} expected ~0.0", c.im);
            }
        }
        // If no Metal device, the test is silently skipped (NoDevice returns err).
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn execute_roundtrip_forward_inverse() {
        // FFT then IFFT should recover the original signal.
        let n = 16usize;
        let plan = match MetalFftPlan::new(n, 1) {
            Ok(p) => p,
            Err(MetalError::NoDevice) => return, // no GPU on CI
            Err(e) => panic!("plan creation error: {e}"),
        };
        let input: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new(i as f32, (n - i) as f32))
            .collect();
        let mut spectrum = vec![Complex::new(0.0f32, 0.0); n];
        let mut recovered = vec![Complex::new(0.0f32, 0.0); n];

        if plan
            .execute(&input, &mut spectrum, MetalFftDirection::Forward)
            .is_ok()
            && plan
                .execute(&spectrum, &mut recovered, MetalFftDirection::Inverse)
                .is_ok()
        {
            for (i, (orig, rec)) in input.iter().zip(recovered.iter()).enumerate() {
                assert!(
                    (orig.re - rec.re).abs() < 1e-3,
                    "recovered[{i}].re = {} expected {}",
                    rec.re,
                    orig.re
                );
                assert!(
                    (orig.im - rec.im).abs() < 1e-3,
                    "recovered[{i}].im = {} expected {}",
                    rec.im,
                    orig.im
                );
            }
        }
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn execute_unsupported_platform() {
        let plan = MetalFftPlan::new(8, 1)
            .expect("FFT plan creation should succeed for valid power-of-2 size");
        let input = vec![Complex::new(0.0f32, 0.0); 8];
        let mut output = vec![Complex::new(0.0f32, 0.0); 8];
        assert!(matches!(
            plan.execute(&input, &mut output, MetalFftDirection::Forward),
            Err(MetalError::UnsupportedPlatform)
        ));
    }

    // ── MetalFftBuffer ────────────────────────────────────────────────────────

    #[test]
    fn fft_buffer_new_and_len() {
        let buf = MetalFftBuffer::new(128, 4);
        assert_eq!(buf.len(), 512);
        assert!(!buf.is_empty());
    }

    #[test]
    fn fft_buffer_zero_initialised() {
        let buf = MetalFftBuffer::new(8, 2);
        for c in buf.as_slice() {
            assert_eq!(*c, Complex::new(0.0f32, 0.0));
        }
    }

    #[test]
    fn fft_buffer_mut_slice() {
        let mut buf = MetalFftBuffer::new(4, 1);
        buf.as_mut_slice()[0] = Complex::new(1.0, 2.0);
        assert_eq!(buf.as_slice()[0], Complex::new(1.0f32, 2.0));
    }

    #[test]
    fn fft_buffer_empty() {
        let buf = MetalFftBuffer::new(0, 1);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn fft_buffer_debug() {
        let buf = MetalFftBuffer::new(8, 2);
        let s = format!("{buf:?}");
        assert!(s.contains("MetalFftBuffer"));
        assert!(s.contains("16"));
    }

    // ── Helper: determine_threadgroup_size ────────────────────────────────────

    #[test]
    #[cfg(target_os = "macos")]
    fn threadgroup_size_basic() {
        // capped is already a power of two in these two cases → returned as-is
        // (no halving): the bug this pins down halved 512→256 and 1024→512.
        assert_eq!(super::determine_threadgroup_size(1024, 512), 512);
        assert_eq!(super::determine_threadgroup_size(1024, 1500), 1024);
        assert_eq!(super::determine_threadgroup_size(256, 300), 256);
        // capped=100 is NOT a power of two → rounds down to 64.
        assert_eq!(super::determine_threadgroup_size(1024, 100), 64);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn threadgroup_size_zero_work() {
        assert_eq!(super::determine_threadgroup_size(256, 0), 1);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn threadgroup_size_zero_max() {
        assert_eq!(super::determine_threadgroup_size(0, 256), 1);
    }

    // ── Correctness tests ─────────────────────────────────────────────────────

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_fft_impulse_spectrum_is_constant() {
        // FFT of [1, 0, 0, ...] should produce [1, 1, 1, ...] (all ones).
        let n = 64;
        let plan = match MetalFftPlan::new(n, 1) {
            Ok(p) => p,
            Err(MetalError::NoDevice) => return, // no GPU on CI
            Err(e) => panic!("plan creation failed: {e}"),
        };
        let mut input = vec![num_complex::Complex::<f32>::new(0.0, 0.0); n];
        input[0] = num_complex::Complex::new(1.0, 0.0);
        let mut output = vec![num_complex::Complex::<f32>::new(0.0, 0.0); n];

        plan.execute(&input, &mut output, MetalFftDirection::Forward)
            .expect("FFT execute failed");

        // Every output bin should have magnitude 1.0 (within f32 tolerance).
        for (i, c) in output.iter().enumerate() {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-4,
                "bin {i}: expected magnitude 1.0, got {mag}"
            );
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_fft_round_trip() {
        // Forward FFT followed by inverse FFT should recover the original signal.
        let n = 256;
        let plan = match MetalFftPlan::new(n, 1) {
            Ok(p) => p,
            Err(MetalError::NoDevice) => return, // no GPU on CI
            Err(e) => panic!("plan creation failed: {e}"),
        };
        let input: Vec<num_complex::Complex<f32>> = (0..n)
            .map(|k| {
                let t = k as f32 / n as f32;
                num_complex::Complex::new(t.sin() + 0.5 * (2.0 * t).cos(), 0.0)
            })
            .collect();

        // Forward
        let mut freq = vec![num_complex::Complex::<f32>::new(0.0, 0.0); n];
        plan.execute(&input, &mut freq, MetalFftDirection::Forward)
            .expect("forward FFT failed");

        // Inverse — oxicuda-metal applies 1/N normalization internally
        let mut recovered = vec![num_complex::Complex::<f32>::new(0.0, 0.0); n];
        plan.execute(&freq, &mut recovered, MetalFftDirection::Inverse)
            .expect("inverse FFT failed");

        // Compare (f32 precision, ~1e-4 tolerance)
        for i in 0..n {
            let err = ((recovered[i].re - input[i].re).powi(2)
                + (recovered[i].im - input[i].im).powi(2))
            .sqrt();
            assert!(
                err < 1e-4,
                "sample {i}: input=({}, {}), recovered=({}, {}), error={err}",
                input[i].re,
                input[i].im,
                recovered[i].re,
                recovered[i].im
            );
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_fft_batch_correctness() {
        // Batch of 2 identical signals should produce identical spectra.
        let n = 32;
        let batch = 2;
        let plan = match MetalFftPlan::new(n, batch) {
            Ok(p) => p,
            Err(MetalError::NoDevice) => return, // no GPU on CI
            Err(e) => panic!("plan creation failed: {e}"),
        };

        let single: Vec<num_complex::Complex<f32>> = (0..n)
            .map(|k| num_complex::Complex::new((k as f32 / n as f32).sin(), 0.0))
            .collect();

        // Replicate the same signal twice for batch input
        let input: Vec<num_complex::Complex<f32>> =
            single.iter().chain(single.iter()).copied().collect();

        let mut output = vec![num_complex::Complex::<f32>::new(0.0, 0.0); n * batch];
        plan.execute(&input, &mut output, MetalFftDirection::Forward)
            .expect("batch FFT failed");

        // Both halves should be identical
        for i in 0..n {
            let a = output[i];
            let b = output[n + i];
            let err = ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt();
            assert!(
                err < 1e-5,
                "batch mismatch at bin {i}: ({}, {}) vs ({}, {})",
                a.re,
                a.im,
                b.re,
                b.im
            );
        }
    }

    // ── DFT oracle: naive O(n²) reference, independent of any FFT algorithm ──
    //
    // The impulse/round-trip tests above pass under a wrong twiddle sign OR
    // an index permutation, because both directions would still be exact
    // conjugates of each other and an impulse's spectrum is convention-
    // invariant. These tests instead compare the GPU FFT against a
    // from-scratch O(n²) DFT — no butterfly structure, no bit-reversal, no
    // twiddle table — computed in f64, so it cannot share a sign or
    // permutation bug with the code under test.

    /// Naive O(n²) discrete Fourier transform, accumulated in f64. Mirrors
    /// the sign convention documented on [`MetalFftDirection`]: forward
    /// uses `exp(-2πi·kn/N)`; inverse uses `exp(+2πi·kn/N)` and is
    /// normalised by `1/N`.
    #[cfg(target_os = "macos")]
    fn naive_dft(input: &[Complex<f32>], inverse: bool) -> Vec<Complex<f32>> {
        let n = input.len();
        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
        let mut output = vec![Complex::new(0.0f32, 0.0f32); n];
        for (k, out_k) in output.iter_mut().enumerate() {
            let mut acc_re = 0.0_f64;
            let mut acc_im = 0.0_f64;
            for (t, x_t) in input.iter().enumerate() {
                let angle =
                    sign * 2.0_f64 * std::f64::consts::PI * (k as f64) * (t as f64) / (n as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                let xr = x_t.re as f64;
                let xi = x_t.im as f64;
                acc_re += xr * cos_a - xi * sin_a;
                acc_im += xr * sin_a + xi * cos_a;
            }
            if inverse {
                acc_re /= n as f64;
                acc_im /= n as f64;
            }
            *out_k = Complex::new(acc_re as f32, acc_im as f32);
        }
        output
    }

    /// Minimal deterministic PRNG (xorshift64*), so test signals are
    /// reproducible without adding a `rand` dependency to the crate.
    #[cfg(target_os = "macos")]
    fn xorshift64star(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// One pseudo-random f32 in `[-1, 1)`, drawn from the top 24 bits of the
    /// generator's output (a full float mantissa's worth of entropy).
    #[cfg(target_os = "macos")]
    fn rand_unit(state: &mut u64) -> f32 {
        let raw = (xorshift64star(state) >> 40) as f32; // in [0, 2^24)
        (raw / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    /// Deterministic pseudo-random complex signal, both parts in `[-1, 1)`.
    #[cfg(target_os = "macos")]
    fn pseudo_random_signal(n: usize, seed: u64) -> Vec<Complex<f32>> {
        let mut state = seed | 1; // xorshift64* requires a nonzero state
        (0..n)
            .map(|_| Complex::new(rand_unit(&mut state), rand_unit(&mut state)))
            .collect()
    }

    /// Assert two complex slices match within a tolerance that scales with
    /// the expected magnitude (`1e-3 * |expected|`) plus a small absolute
    /// floor (`1e-3`) for near-zero bins — tight enough that a twiddle
    /// stride/indexing bug (which perturbs only some bins, by less than the
    /// full signal energy) cannot slip through, while still tolerant of
    /// ordinary f32 GPU rounding.
    #[cfg(target_os = "macos")]
    fn assert_complex_slices_close(actual: &[Complex<f32>], expected: &[Complex<f32>], ctx: &str) {
        assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let err = ((a.re - e.re).powi(2) + (a.im - e.im).powi(2)).sqrt();
            let mag = (e.re * e.re + e.im * e.im).sqrt();
            let tol = 1e-3 * mag + 1e-3;
            assert!(
                err < tol,
                "{ctx}: bin {i}: got ({}, {}), expected ({}, {}), err={err}, tol={tol}",
                a.re,
                a.im,
                e.re,
                e.im
            );
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_fft_forward_matches_naive_dft_oracle_multi_size() {
        // Forward FFT vs. an independent O(n²) DFT oracle at several sizes,
        // using a non-trivial signal (a pure tone plus a DC offset) so
        // every bin has a distinct, mostly-nonzero expected value — unlike
        // the impulse test, a wrong twiddle sign or an index permutation
        // cannot pass this by accident.
        for &n in &[8usize, 16, 64, 256] {
            let plan = match MetalFftPlan::new(n, 1) {
                Ok(p) => p,
                Err(MetalError::NoDevice) => return, // no GPU on CI
                Err(e) => panic!("plan creation failed (n={n}): {e}"),
            };
            let k0 = (n / 4).max(1); // an arbitrary non-DC, non-Nyquist tone
            let input: Vec<Complex<f32>> = (0..n)
                .map(|t| {
                    let angle =
                        2.0_f64 * std::f64::consts::PI * (k0 as f64) * (t as f64) / (n as f64);
                    Complex::new(angle.cos() as f32 + 0.3, angle.sin() as f32 - 0.1)
                })
                .collect();

            let expected = naive_dft(&input, false);
            let mut actual = vec![Complex::new(0.0f32, 0.0f32); n];
            plan.execute(&input, &mut actual, MetalFftDirection::Forward)
                .unwrap_or_else(|e| panic!("execute failed (n={n}): {e}"));

            assert_complex_slices_close(&actual, &expected, &format!("forward FFT n={n}"));
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_fft_forward_matches_naive_dft_oracle_random_signal() {
        // Forward FFT vs. the O(n²) oracle on a pseudo-random (non-tone)
        // signal, so energy is spread across every bin instead of
        // concentrated at one frequency.
        let n = 64usize;
        let plan = match MetalFftPlan::new(n, 1) {
            Ok(p) => p,
            Err(MetalError::NoDevice) => return, // no GPU on CI
            Err(e) => panic!("plan creation failed: {e}"),
        };
        let input = pseudo_random_signal(n, 0xC0FF_EE12_3456_789A);
        let expected = naive_dft(&input, false);
        let mut actual = vec![Complex::new(0.0f32, 0.0f32); n];
        plan.execute(&input, &mut actual, MetalFftDirection::Forward)
            .expect("execute failed");
        assert_complex_slices_close(&actual, &expected, "forward FFT random signal n=64");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_fft_inverse_matches_naive_dft_oracle_with_normalization() {
        // Inverse FFT vs. the O(n²) inverse-DFT oracle (which applies its
        // own 1/N normalisation), starting from a forward spectrum that is
        // ALSO cross-checked against the oracle — so both directions, and
        // the normalisation factor, are pinned independently of the
        // round-trip self-consistency tests above (which would still pass
        // if forward and inverse shared a compensating bug).
        let n = 128usize;
        let plan = match MetalFftPlan::new(n, 1) {
            Ok(p) => p,
            Err(MetalError::NoDevice) => return, // no GPU on CI
            Err(e) => panic!("plan creation failed: {e}"),
        };
        let input = pseudo_random_signal(n, 0x1357_9BDF_2468_ACE0);

        let expected_forward = naive_dft(&input, false);
        let mut spectrum = vec![Complex::new(0.0f32, 0.0f32); n];
        plan.execute(&input, &mut spectrum, MetalFftDirection::Forward)
            .expect("forward execute failed");
        assert_complex_slices_close(&spectrum, &expected_forward, "forward stage");

        let expected_inverse = naive_dft(&spectrum, true);
        let mut recovered = vec![Complex::new(0.0f32, 0.0f32); n];
        plan.execute(&spectrum, &mut recovered, MetalFftDirection::Inverse)
            .expect("inverse execute failed");
        assert_complex_slices_close(&recovered, &expected_inverse, "inverse stage");

        // ...and recovered should match the ORIGINAL input: this is the
        // normalisation check — a missing or doubled 1/N would fail here
        // even if both DFT-oracle comparisons above happened to pass.
        assert_complex_slices_close(&recovered, &input, "round trip after normalization");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_fft_batch_different_signals_match_oracle_individually() {
        // A batch of DIFFERENT signals — not replicated copies — each
        // checked against its OWN oracle transform. Row 0 and row 1 are
        // impulses at different (nonzero) positions: an impulse at index d
        // has a distinctive per-bin phase ramp (exp(-2πi·k·d/n)), so a
        // batch-offset bug (e.g. row 1 reading row 0's data, or a
        // `gid.y`/stride mistake) produces an obviously wrong spectrum
        // rather than merely "different but plausible" numbers. Row 2 is
        // pseudo-random for good measure. This is the discriminating case
        // `metal_fft_batch_correctness` (identical rows) cannot exercise.
        let n = 32usize;
        let batch = 3usize;
        let plan = match MetalFftPlan::new(n, batch) {
            Ok(p) => p,
            Err(MetalError::NoDevice) => return, // no GPU on CI
            Err(e) => panic!("plan creation failed: {e}"),
        };

        let impulse_at = |d: usize| -> Vec<Complex<f32>> {
            let mut row = vec![Complex::new(0.0f32, 0.0f32); n];
            row[d] = Complex::new(1.0, 0.0);
            row
        };
        let rows: Vec<Vec<Complex<f32>>> = vec![
            impulse_at(0),
            impulse_at(5),
            pseudo_random_signal(n, 0x9999_AAAA_BBBB_CCCC),
        ];
        let input: Vec<Complex<f32>> = rows.iter().flatten().copied().collect();

        let mut output = vec![Complex::new(0.0f32, 0.0f32); n * batch];
        plan.execute(&input, &mut output, MetalFftDirection::Forward)
            .expect("batch execute failed");

        for (row_idx, row) in rows.iter().enumerate() {
            let expected = naive_dft(row, false);
            let actual = &output[row_idx * n..(row_idx + 1) * n];
            assert_complex_slices_close(actual, &expected, &format!("batch row {row_idx}"));
        }
    }
}
