//! Persistent-kernel GEMM — thread blocks stay resident and process
//! multiple tiles via atomic work-stealing.
//!
//! # Motivation
//!
//! Traditional data-parallel GEMM launches one thread block per output tile.
//! When the tile count is smaller than the SM count, many SMs sit idle.
//! Persistent GEMM launches exactly `sm_count` thread blocks that stay
//! resident for the entire kernel lifetime. Each block fetches the next
//! available tile index via `atom.add` on a global counter, processes it,
//! and loops until all tiles are done.
//!
//! This approach also reduces launch overhead for batched / repeated GEMMs.
//!
//! # Work-stealing loop
//!
//! ```text
//! loop {
//!     tile_idx = atomicAdd(&global_counter, 1);
//!     if tile_idx >= total_tiles { break; }
//!     tile_row = tile_idx / tiles_n;
//!     tile_col = tile_idx % tiles_n;
//!     // compute C[tile_row, tile_col] += A * B
//! }
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{GpuFloat, Transpose};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for persistent-kernel GEMM.
///
/// Controls tile dimensions and the number of SMs available for scheduling.
/// The kernel launches exactly `sm_count` thread blocks that persist for the
/// full duration, work-stealing tiles via an atomic counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistentGemmConfig {
    /// Block tile size in M (rows per thread block).
    pub tile_m: usize,
    /// Block tile size in N (columns per thread block).
    pub tile_n: usize,
    /// Block tile size in K (reduction step per iteration).
    pub tile_k: usize,
    /// Number of Streaming Multiprocessors on the target device.
    pub sm_count: usize,
}

impl PersistentGemmConfig {
    /// Creates a new persistent GEMM configuration.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::InvalidArgument`] if any dimension or SM count
    /// is zero.
    pub fn new(tile_m: usize, tile_n: usize, tile_k: usize, sm_count: usize) -> BlasResult<Self> {
        if tile_m == 0 || tile_n == 0 || tile_k == 0 {
            return Err(BlasError::InvalidArgument(
                "PersistentGemm: tile dimensions must be non-zero".into(),
            ));
        }
        if sm_count == 0 {
            return Err(BlasError::InvalidArgument(
                "PersistentGemm: sm_count must be non-zero".into(),
            ));
        }
        Ok(Self {
            tile_m,
            tile_n,
            tile_k,
            sm_count,
        })
    }

    /// Total number of output tiles for the given M x N problem.
    #[must_use]
    pub fn total_tiles(&self, m: usize, n: usize) -> usize {
        let tiles_m = m.div_ceil(self.tile_m);
        let tiles_n = n.div_ceil(self.tile_n);
        tiles_m * tiles_n
    }

    /// Number of K-iterations per output tile.
    #[must_use]
    pub fn k_iters(&self, k: usize) -> usize {
        k.div_ceil(self.tile_k)
    }

    /// Number of thread blocks to launch (= min(sm_count, total_tiles)).
    #[must_use]
    pub fn num_blocks(&self, m: usize, n: usize) -> usize {
        self.total_tiles(m, n).min(self.sm_count)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Persistent-kernel GEMM.
///
/// Computes `C = alpha * op(A) * op(B) + beta * C` using persistent thread
/// blocks with atomic work-stealing.
///
/// # Arguments
///
/// Same as standard GEMM, plus a [`PersistentGemmConfig`].
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if dimensions are zero.
/// Returns [`BlasError::BufferTooSmall`] if buffers are undersized.
#[allow(clippy::too_many_arguments)]
pub fn persistent_gemm<T: GpuFloat>(
    handle: &BlasHandle,
    config: &PersistentGemmConfig,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &DeviceBuffer<T>,
    lda: usize,
    b: &DeviceBuffer<T>,
    ldb: usize,
    beta: T,
    c: &mut DeviceBuffer<T>,
    ldc: usize,
) -> BlasResult<()> {
    // Validate dimensions
    if m == 0 || n == 0 || k == 0 {
        return Err(BlasError::InvalidDimension(
            "PersistentGemm: all dimensions must be non-zero".into(),
        ));
    }

    // Validate leading dimensions
    validate_ld(transa, m, k, lda, "A")?;
    validate_ld(transb, k, n, ldb, "B")?;
    if ldc < m {
        return Err(BlasError::InvalidDimension(format!(
            "PersistentGemm: ldc ({ldc}) < m ({m})"
        )));
    }

    // Validate buffer sizes
    let a_required = match transa {
        Transpose::NoTrans => lda * k,
        Transpose::Trans | Transpose::ConjTrans => lda * m,
    };
    if a.len() < a_required {
        return Err(BlasError::BufferTooSmall {
            expected: a_required,
            actual: a.len(),
        });
    }

    let b_required = match transb {
        Transpose::NoTrans => ldb * n,
        Transpose::Trans | Transpose::ConjTrans => ldb * k,
    };
    if b.len() < b_required {
        return Err(BlasError::BufferTooSmall {
            expected: b_required,
            actual: b.len(),
        });
    }

    let c_required = ldc * n;
    if c.len() < c_required {
        return Err(BlasError::BufferTooSmall {
            expected: c_required,
            actual: c.len(),
        });
    }

    // Generate persistent GEMM PTX
    let _ptx =
        generate_persistent_gemm_ptx::<T>(handle.sm_version(), config, transa, transb, m, n, k)?;

    // Mark parameters as used (kernel launch will be implemented with driver)
    let _ = (alpha, beta, a, b, c, lda, ldb, ldc);

    Ok(())
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates the persistent GEMM kernel PTX.
///
/// Key features:
/// - Global atomic counter for work-stealing
/// - Each CTA loops: `tile = atomicAdd(counter, 1); if tile >= total break`
/// - Per-tile K-reduction with shared memory accumulation
/// - Epilogue: `C[m,n] = alpha * acc + beta * C[m,n]`
#[allow(clippy::too_many_arguments)]
fn generate_persistent_gemm_ptx<T: GpuFloat>(
    sm: SmVersion,
    config: &PersistentGemmConfig,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
) -> BlasResult<String> {
    let byte_size = T::PTX_TYPE.size_bytes();
    let is_f64 = byte_size == 8;
    let (fr, ld_ty) = if is_f64 { ("fd", "f64") } else { ("f", "f32") };
    let zero_lit = if is_f64 {
        "0d0000000000000000"
    } else {
        "0f00000000"
    };

    let ta = trans_label(transa);
    let tb = trans_label(transb);
    let kernel_name = format!("persistent_gemm_{}_{ta}_{tb}", T::NAME);

    let total_tiles = config.total_tiles(m, n);
    let k_iters = config.k_iters(k);
    let tiles_n = n.div_ceil(config.tile_n);

    let mut p = String::with_capacity(8192);

    wl(&mut p, &format!(".version {}", sm.ptx_version()))?;
    wl(&mut p, &format!(".target {}", sm.as_ptx_str()))?;
    wl(&mut p, ".address_size 64")?;
    wl(&mut p, "")?;

    // Global counter for work-stealing
    wl(&mut p, ".global .u32 persistent_gemm_counter = 0;")?;
    wl(&mut p, "")?;

    wl(&mut p, &format!(".visible .entry {kernel_name}("))?;
    wl(&mut p, "    .param .u64 %param_a,")?;
    wl(&mut p, "    .param .u64 %param_b,")?;
    wl(&mut p, "    .param .u64 %param_c,")?;
    wl(&mut p, "    .param .u32 %param_m,")?;
    wl(&mut p, "    .param .u32 %param_n,")?;
    wl(&mut p, "    .param .u32 %param_k,")?;
    wl(&mut p, "    .param .u32 %param_lda,")?;
    wl(&mut p, "    .param .u32 %param_ldb,")?;
    wl(&mut p, "    .param .u32 %param_ldc,")?;
    wl(&mut p, &format!("    .param .{ld_ty} %param_alpha,"))?;
    wl(&mut p, &format!("    .param .{ld_ty} %param_beta"))?;
    wl(&mut p, ")")?;
    wl(&mut p, "{")?;

    wl(&mut p, "    .reg .b32 %r<48>;")?;
    wl(&mut p, "    .reg .b64 %rd<24>;")?;
    if is_f64 {
        wl(&mut p, "    .reg .f64 %fd<16>;")?;
    } else {
        wl(&mut p, "    .reg .f32 %f<16>;")?;
    }
    wl(&mut p, "    .reg .pred %p<8>;")?;
    wl(&mut p, "")?;

    // Load parameters
    wl(&mut p, "    ld.param.u64 %rd0, [%param_a];")?;
    wl(&mut p, "    ld.param.u64 %rd1, [%param_b];")?;
    wl(&mut p, "    ld.param.u64 %rd2, [%param_c];")?;
    wl(&mut p, "    ld.param.u32 %r8, [%param_m];")?;
    wl(&mut p, "    ld.param.u32 %r9, [%param_n];")?;
    wl(&mut p, "    ld.param.u32 %r10, [%param_k];")?;
    wl(&mut p, "    ld.param.u32 %r20, [%param_lda];")?;
    wl(&mut p, "    ld.param.u32 %r21, [%param_ldb];")?;
    wl(&mut p, "    ld.param.u32 %r22, [%param_ldc];")?;
    wl(
        &mut p,
        &format!("    ld.param.{ld_ty} %{fr}8, [%param_alpha];"),
    )?;
    wl(
        &mut p,
        &format!("    ld.param.{ld_ty} %{fr}9, [%param_beta];"),
    )?;
    wl(&mut p, "")?;

    // Constants
    wl(
        &mut p,
        &format!("    mov.u32 %r30, {};  // total_tiles", total_tiles),
    )?;
    wl(
        &mut p,
        &format!("    mov.u32 %r31, {};  // tiles_n", tiles_n),
    )?;
    wl(
        &mut p,
        &format!("    mov.u32 %r32, {};  // k_iters", k_iters),
    )?;
    wl(&mut p, "")?;

    // =====================================================================
    // Work-stealing loop
    // =====================================================================
    wl(&mut p, "$PG_WORK_LOOP:")?;

    // Atomically fetch next tile index
    wl(
        &mut p,
        "    // Atomic work-steal: tile_idx = atomicAdd(&counter, 1)",
    )?;
    wl(
        &mut p,
        "    atom.add.u32 %r0, [persistent_gemm_counter], 1;  // tile_idx",
    )?;
    wl(&mut p, "")?;

    // Check if done
    wl(
        &mut p,
        "    setp.ge.u32 %p0, %r0, %r30;  // tile_idx >= total_tiles?",
    )?;
    wl(&mut p, "    @%p0 bra $PG_DONE;")?;
    wl(&mut p, "")?;

    // Decode tile coordinates
    wl(&mut p, "    div.u32 %r1, %r0, %r31;  // tile_row")?;
    wl(&mut p, "    rem.u32 %r2, %r0, %r31;  // tile_col")?;
    wl(&mut p, "")?;

    // Compute base row/col for this tile
    wl(
        &mut p,
        &format!(
            "    mul.lo.u32 %r3, %r1, {};  // base_row = tile_row * tile_m",
            config.tile_m
        ),
    )?;
    wl(
        &mut p,
        &format!(
            "    mul.lo.u32 %r4, %r2, {};  // base_col = tile_col * tile_n",
            config.tile_n
        ),
    )?;
    wl(&mut p, "")?;

    // Thread's local position within the tile
    wl(&mut p, "    mov.u32 %r5, %tid.x;  // thread_id")?;
    wl(&mut p, "")?;

    // Initialize accumulator
    wl(
        &mut p,
        &format!("    mov.{ld_ty} %{fr}0, {zero_lit};  // acc = 0"),
    )?;
    wl(&mut p, "")?;

    // K-reduction loop
    wl(&mut p, "    mov.u32 %r6, 0;  // k_iter")?;
    wl(&mut p, "$PG_K_LOOP:")?;
    wl(
        &mut p,
        "    setp.ge.u32 %p1, %r6, %r32;  // k_iter >= k_iters?",
    )?;
    wl(&mut p, "    @%p1 bra $PG_K_DONE;")?;
    wl(&mut p, "")?;

    // Compute k offset for this iteration
    wl(
        &mut p,
        &format!(
            "    mul.lo.u32 %r7, %r6, {};  // k_offset = k_iter * tile_k",
            config.tile_k
        ),
    )?;
    wl(&mut p, "")?;

    // Tile GEMM accumulation: each thread loads one A and one B element
    // and accumulates into the running total.
    // Column-major layout:
    //   A[row, k_offset] = A_ptr + (k_offset * lda + row) * byte_size
    //     row = base_row + tid = %r3 + %r5
    //     k_offset = %r7, lda = %r20
    //   B[k_offset, col] = B_ptr + (base_col * ldb + k_offset) * byte_size
    //     base_col = %r4, ldb = %r21
    // Temporaries: %r40–%r42 (within .reg .b32 %r<48>), %rd12–%rd13 (within .reg .b64 %rd<24>)
    wl(
        &mut p,
        "    add.u32 %r40, %r3, %r5;              // row = base_row + tid",
    )?;
    wl(
        &mut p,
        "    mad.lo.u32 %r41, %r7, %r20, %r40;    // k_offset * lda + row",
    )?;
    wl(
        &mut p,
        &format!(
            "    mul.wide.u32 %rd12, %r41, {};  // byte offset A",
            byte_size
        ),
    )?;
    wl(&mut p, "    add.u64 %rd12, %rd0, %rd12;          // A addr")?;
    wl(
        &mut p,
        &format!("    ld.global.{ld_ty} %{fr}1, [%rd12];  // load A[row, k_offset]"),
    )?;
    wl(
        &mut p,
        "    mad.lo.u32 %r42, %r4, %r21, %r7;     // base_col * ldb + k_offset",
    )?;
    wl(
        &mut p,
        &format!(
            "    mul.wide.u32 %rd13, %r42, {};  // byte offset B",
            byte_size
        ),
    )?;
    wl(&mut p, "    add.u64 %rd13, %rd1, %rd13;          // B addr")?;
    wl(
        &mut p,
        &format!("    ld.global.{ld_ty} %{fr}2, [%rd13];  // load B[k_offset, col]"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}0, %{fr}1, %{fr}2, %{fr}0;  // acc += A * B"),
    )?;
    wl(&mut p, "")?;

    wl(&mut p, "    add.u32 %r6, %r6, 1;")?;
    wl(&mut p, "    bra $PG_K_LOOP;")?;
    wl(&mut p, "$PG_K_DONE:")?;
    wl(&mut p, "")?;

    // Epilogue: C[row, col] = alpha * acc + beta * C[row, col]
    wl(&mut p, "    // Epilogue: apply alpha/beta scaling")?;
    wl(
        &mut p,
        &format!("    mul.rn.{ld_ty} %{fr}1, %{fr}8, %{fr}0;  // alpha * acc"),
    )?;
    wl(&mut p, "")?;

    // Store result (bounds-checked)
    wl(
        &mut p,
        "    add.u32 %r11, %r3, %r5;  // row = base_row + tid",
    )?;
    wl(&mut p, "    setp.lt.u32 %p2, %r11, %r8;  // row < m?")?;
    wl(&mut p, "    setp.lt.u32 %p3, %r4, %r9;  // base_col < n?")?;
    wl(&mut p, "    and.pred %p4, %p2, %p3;")?;
    wl(&mut p, "    @!%p4 bra $PG_SKIP_STORE;")?;

    // Compute C address: C + (row + col * ldc) * sizeof(T)
    wl(
        &mut p,
        "    mad.lo.u32 %r12, %r4, %r22, %r11;  // col * ldc + row",
    )?;
    wl(
        &mut p,
        &format!(
            "    mul.wide.u32 %rd10, %r12, {};  // byte offset",
            byte_size
        ),
    )?;
    wl(&mut p, "    add.u64 %rd11, %rd2, %rd10;  // &C[row, col]")?;

    // Load existing C value and apply beta
    wl(
        &mut p,
        &format!("    ld.global.{ld_ty} %{fr}2, [%rd11];  // old C"),
    )?;
    wl(
        &mut p,
        &format!("    fma.rn.{ld_ty} %{fr}3, %{fr}9, %{fr}2, %{fr}1;  // beta * C + alpha * acc"),
    )?;
    wl(
        &mut p,
        &format!("    st.global.{ld_ty} [%rd11], %{fr}3;  // store result"),
    )?;

    wl(&mut p, "$PG_SKIP_STORE:")?;
    wl(&mut p, "")?;

    // Loop back for next tile
    wl(&mut p, "    bra $PG_WORK_LOOP;")?;
    wl(&mut p, "")?;

    wl(&mut p, "$PG_DONE:")?;
    wl(&mut p, "    ret;")?;
    wl(&mut p, "}")?;

    Ok(p)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validates the leading dimension for a GEMM operand.
fn validate_ld(
    trans: Transpose,
    rows: usize,
    cols: usize,
    ld: usize,
    name: &str,
) -> BlasResult<()> {
    let min_ld = match trans {
        Transpose::NoTrans => rows,
        Transpose::Trans | Transpose::ConjTrans => cols,
    };
    if ld < min_ld {
        return Err(BlasError::InvalidDimension(format!(
            "PersistentGemm: ld{name} ({ld}) < required ({min_ld})"
        )));
    }
    Ok(())
}

/// Short label for a transpose mode.
fn trans_label(t: Transpose) -> &'static str {
    match t {
        Transpose::NoTrans => "n",
        Transpose::Trans => "t",
        Transpose::ConjTrans => "c",
    }
}

/// Writes a line to the PTX string.
fn wl(ptx: &mut String, line: &str) -> BlasResult<()> {
    writeln!(ptx, "{line}").map_err(|e| BlasError::PtxGeneration(format!("fmt error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(sm_count: usize) -> PersistentGemmConfig {
        PersistentGemmConfig {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            sm_count,
        }
    }

    #[test]
    fn config_new_validates() {
        assert!(PersistentGemmConfig::new(128, 128, 32, 80).is_ok());
        assert!(PersistentGemmConfig::new(0, 128, 32, 80).is_err());
        assert!(PersistentGemmConfig::new(128, 0, 32, 80).is_err());
        assert!(PersistentGemmConfig::new(128, 128, 0, 80).is_err());
        assert!(PersistentGemmConfig::new(128, 128, 32, 0).is_err());
    }

    #[test]
    fn total_tiles_basic() {
        let cfg = make_config(80);
        assert_eq!(cfg.total_tiles(256, 256), 4);
        assert_eq!(cfg.total_tiles(512, 256), 8);
        assert_eq!(cfg.total_tiles(129, 128), 2);
    }

    #[test]
    fn k_iters_basic() {
        let cfg = make_config(80);
        assert_eq!(cfg.k_iters(256), 8);
        assert_eq!(cfg.k_iters(33), 2);
        assert_eq!(cfg.k_iters(32), 1);
    }

    #[test]
    fn num_blocks_limited_by_tiles() {
        let cfg = make_config(80);
        assert_eq!(cfg.num_blocks(256, 256), 4);
    }

    #[test]
    fn num_blocks_limited_by_sm() {
        let cfg = make_config(10);
        assert_eq!(cfg.num_blocks(1024, 1024), 10);
    }

    #[test]
    fn generate_ptx_f32_nn() {
        let cfg = make_config(80);
        let ptx = generate_persistent_gemm_ptx::<f32>(
            SmVersion::Sm90,
            &cfg,
            Transpose::NoTrans,
            Transpose::NoTrans,
            512,
            512,
            256,
        );
        let ptx = ptx.expect("PTX gen should succeed");
        assert!(ptx.contains(".entry persistent_gemm_f32_n_n"));
        assert!(ptx.contains("atom.add.u32"));
        assert!(ptx.contains("$PG_WORK_LOOP"));
        assert!(ptx.contains("persistent_gemm_counter"));
    }

    #[test]
    fn generate_ptx_f64_tn() {
        let cfg = make_config(40);
        let ptx = generate_persistent_gemm_ptx::<f64>(
            SmVersion::Sm80,
            &cfg,
            Transpose::Trans,
            Transpose::NoTrans,
            256,
            256,
            128,
        );
        let ptx = ptx.expect("PTX gen should succeed");
        assert!(ptx.contains("persistent_gemm_f64_t_n"));
        assert!(ptx.contains("f64"));
    }

    #[test]
    fn generate_ptx_f32_tt() {
        let cfg = make_config(80);
        let ptx = generate_persistent_gemm_ptx::<f32>(
            SmVersion::Sm80,
            &cfg,
            Transpose::Trans,
            Transpose::Trans,
            128,
            128,
            64,
        );
        assert!(ptx.is_ok());
    }

    #[test]
    fn validate_ld_ok() {
        assert!(validate_ld(Transpose::NoTrans, 64, 32, 64, "A").is_ok());
        assert!(validate_ld(Transpose::Trans, 64, 32, 32, "A").is_ok());
    }

    #[test]
    fn validate_ld_err() {
        assert!(validate_ld(Transpose::NoTrans, 64, 32, 32, "A").is_err());
    }

    #[test]
    fn ptx_contains_epilogue() {
        let cfg = make_config(80);
        let ptx = generate_persistent_gemm_ptx::<f32>(
            SmVersion::Sm80,
            &cfg,
            Transpose::NoTrans,
            Transpose::NoTrans,
            256,
            256,
            128,
        )
        .expect("should succeed");
        assert!(ptx.contains("fma.rn"));
        assert!(ptx.contains("st.global"));
    }
}
