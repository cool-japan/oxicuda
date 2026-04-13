//! Matrix Multiplication Example
//!
//! Demonstrates GEMM (C = A × B) using a PTX kernel built programmatically
//! with the OxiCUDA PTX DSL via `KernelBuilder` and `BodyBuilder`.
//!
//! The kernel implements a simple SIMT GEMM where each thread computes one
//! output element C[row][col] = sum_k A[row][k] * B[k][col].
//! A 16×16 shared-memory staging tile is used to amortise global-memory
//! bandwidth, with `bar.sync 0` barriers between stages.
//!
//! # Running
//!
//! Requires a system with an NVIDIA GPU and driver installed.
//! ```bash
//! cargo run --example matrix_multiply -p oxicuda --features "ptx"
//! ```
//!
//! On macOS or systems without a GPU the PTX generation step runs and
//! prints the generated PTX to stdout, then fails gracefully.

use std::sync::Arc;

use oxicuda::prelude::*;
use oxicuda::{DeviceBuffer, LaunchParams};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

/// Thread-block tile size (TILE × TILE threads per block).
const TILE: u32 = 16;

/// Build the PTX text for a tiled SGEMM kernel using the `KernelBuilder` DSL.
///
/// Each thread computes `C[row][col] = sum_k A[row][k] * B[k][col]`
/// for row-major matrices of size M×K, K×N, M×N respectively.
///
/// The inner-product loop over K is split into tiles of size `TILE`
/// staged through shared memory.  Register naming uses temporaries to
/// avoid Rust's double-borrow restriction on `&mut BodyBuilder`.
///
/// # Errors
///
/// Returns `PtxGenError` if the DSL builder fails.
fn build_sgemm_ptx() -> Result<String, oxicuda_ptx::error::PtxGenError> {
    let tile = TILE;

    KernelBuilder::new("sgemm_tiled")
        .target(SmVersion::Sm80)
        .param("a", PtxType::U64) // const float* A (row-major, M×K)
        .param("b", PtxType::U64) // const float* B (row-major, K×N)
        .param("c", PtxType::U64) // float*       C (row-major, M×N)
        .param("m", PtxType::U32) // rows of A and C
        .param("n", PtxType::U32) // cols of B and C
        .param("k", PtxType::U32) // inner dimension
        // Two shared-memory tiles (A and B), each TILE×TILE f32 elements.
        .shared_mem("smem_a", PtxType::F32, (tile * tile) as usize)
        .shared_mem("smem_b", PtxType::F32, (tile * tile) as usize)
        .max_threads_per_block(tile * tile)
        .body(move |b| {
            b.comment("=== sgemm_tiled: one thread → one C element ===");

            // Load matrix dimensions from kernel parameters.
            let m_reg = b.load_param_u32("m");
            let n_reg = b.load_param_u32("n");
            let k_reg = b.load_param_u32("k");

            // Load base device pointers.
            let a_base = b.load_param_u64("a");
            let b_base = b.load_param_u64("b");
            let c_base = b.load_param_u64("c");

            // This thread's global row and column (2-D grid).
            let row = b.global_thread_id_y();
            let col = b.global_thread_id_x();

            // Thread-local tile indices (tid.x = column, tid.y via raw PTX).
            let tx = b.thread_id_x();
            // tid.y — BodyBuilder only exposes tid.x directly.
            let ty_local = b.mov_imm_u32(0);
            b.raw_ptx(&format!("mov.u32 {ty_local}, %tid.y;"));

            // Initialise the per-thread accumulator to 0.0.
            let acc = b.mov_imm_u32(0); // allocate a u32 slot then reuse as f32
            b.raw_ptx(&format!("mov.f32 {acc}, 0f00000000;"));

            // Outer loop: iterate over K in steps of TILE.
            //
            // We use a manual PTX loop since BodyBuilder has no general
            // counted-loop primitive.  The loop counter `tile_col` tracks
            // the starting column of the current A-tile (= starting row of
            // the current B-tile).
            let tile_col = b.mov_imm_u32(0); // loop counter: 0, TILE, 2*TILE, …
            b.raw_ptx("TILE_LOOP:");

            b.comment("--- load A[row][tile_col + tx] into smem_a[ty_local][tx] ---");
            // A element: row-major index = row * K + (tile_col + tx)
            let a_inner = {
                let tc_plus_tx = b.add_u32(tile_col.clone(), tx.clone());
                b.mad_lo_u32(row.clone(), k_reg.clone(), tc_plus_tx)
            };
            let a_elem_addr = b.f32_elem_addr(a_base.clone(), a_inner);
            let a_val = b.load_global_f32(a_elem_addr);
            // smem_a[ty_local * TILE + tx]  — evaluate all sub-expressions before mad_lo_u32
            let smem_a_flat = {
                let tile_imm = b.mov_imm_u32(tile);
                b.mad_lo_u32(ty_local.clone(), tile_imm, tx.clone())
            };
            let smem_a_ptr = {
                let ptr = b.mov_imm_u32(0);
                b.raw_ptx(&format!("cvta.to.shared.u64 {ptr}, smem_a;"));
                b.f32_elem_addr(ptr, smem_a_flat)
            };
            b.store_shared_f32(smem_a_ptr, a_val);

            b.comment("--- load B[tile_col + ty_local][col] into smem_b[ty_local][tx] ---");
            // B element: row-major index = (tile_col + ty_local) * N + col
            let b_inner = {
                let tc_plus_ty = b.add_u32(tile_col.clone(), ty_local.clone());
                b.mad_lo_u32(tc_plus_ty, n_reg.clone(), col.clone())
            };
            let b_elem_addr = b.f32_elem_addr(b_base.clone(), b_inner);
            let b_val = b.load_global_f32(b_elem_addr);
            // smem_b[ty_local * TILE + tx]
            let smem_b_flat = {
                let tile_imm = b.mov_imm_u32(tile);
                b.mad_lo_u32(ty_local.clone(), tile_imm, tx.clone())
            };
            let smem_b_ptr = {
                let ptr = b.mov_imm_u32(0);
                b.raw_ptx(&format!("cvta.to.shared.u64 {ptr}, smem_b;"));
                b.f32_elem_addr(ptr, smem_b_flat)
            };
            b.store_shared_f32(smem_b_ptr, b_val);

            // Synchronise to ensure all threads have loaded their tile elements.
            b.bar_sync(0);

            b.comment("--- inner product over the tile (unrolled TILE iterations) ---");
            b.unroll(tile, |b, ki| {
                // smem_a[ty_local * TILE + ki]
                let sa_idx = {
                    let tile_imm = b.mov_imm_u32(tile);
                    let ki_imm = b.mov_imm_u32(ki);
                    b.mad_lo_u32(ty_local.clone(), tile_imm, ki_imm)
                };
                let sa_ptr = {
                    let ptr = b.mov_imm_u32(0);
                    b.raw_ptx(&format!("cvta.to.shared.u64 {ptr}, smem_a;"));
                    b.f32_elem_addr(ptr, sa_idx)
                };
                let av = b.load_shared_f32(sa_ptr);

                // smem_b[ki * TILE + tx]
                let sb_idx = {
                    let ki_imm = b.mov_imm_u32(ki);
                    let tile_imm = b.mov_imm_u32(tile);
                    b.mad_lo_u32(ki_imm, tile_imm, tx.clone())
                };
                let sb_ptr = {
                    let ptr = b.mov_imm_u32(0);
                    b.raw_ptx(&format!("cvta.to.shared.u64 {ptr}, smem_b;"));
                    b.f32_elem_addr(ptr, sb_idx)
                };
                let bv = b.load_shared_f32(sb_ptr);

                // acc = fma(av, bv, acc)
                let new_acc = b.fma_f32(av, bv, acc.clone());
                b.raw_ptx(&format!("mov.f32 {acc}, {new_acc};"));
            });

            // Synchronise before loading the next tile.
            b.bar_sync(0);

            // Advance the tile counter: tile_col += TILE.
            let next_col = {
                let step = b.mov_imm_u32(tile);
                b.add_u32(tile_col.clone(), step)
            };
            b.raw_ptx(&format!("mov.u32 {tile_col}, {next_col};"));

            // Loop back if tile_col < K.
            b.raw_ptx(&format!("setp.lt.u32 %p_tl, {tile_col}, {k_reg};"));
            b.raw_ptx("@%p_tl bra TILE_LOOP;");

            // Write back C[row][col] = acc — guard against out-of-bounds threads.
            b.if_lt_u32(row.clone(), m_reg.clone(), |b| {
                b.if_lt_u32(col.clone(), n_reg.clone(), |b| {
                    let c_flat = b.mad_lo_u32(row.clone(), n_reg.clone(), col.clone());
                    let c_ptr = b.f32_elem_addr(c_base.clone(), c_flat);
                    b.store_global_f32(c_ptr, acc.clone());
                });
            });

            b.ret();
        })
        .build()
}

/// CPU reference GEMM for correctness verification.
fn cpu_gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a[row * k + ki] * b[ki * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiCUDA Matrix Multiplication (SGEMM) Example ===\n");

    // Step 1: generate the PTX (no GPU required).
    println!("Building PTX for sgemm_tiled (tile={}×{})...", TILE, TILE);
    let ptx = build_sgemm_ptx()?;
    println!("PTX generated successfully ({} bytes).", ptx.len());
    println!(
        "Shared memory per block: {} bytes (2 × {}×{} f32 tiles)\n",
        2 * TILE * TILE * 4,
        TILE,
        TILE
    );

    // Print the first 800 characters of the generated PTX.
    let preview_len = ptx.len().min(800);
    println!(
        "--- PTX preview (first {} of {} chars) ---",
        preview_len,
        ptx.len()
    );
    println!("{}", &ptx[..preview_len]);
    if ptx.len() > preview_len {
        println!("... ({} more chars)", ptx.len() - preview_len);
    }
    println!("---\n");

    // Step 2: optionally run on GPU.
    match try_gpu_gemm(&ptx) {
        Ok(()) => println!("\nGPU GEMM completed successfully."),
        Err(e) => println!(
            "\nGPU not available: {} (expected on macOS / no-GPU systems)",
            e
        ),
    }

    Ok(())
}

/// Attempt to run the GEMM kernel on an available GPU.
fn try_gpu_gemm(ptx: &str) -> Result<(), Box<dyn std::error::Error>> {
    oxicuda::init()?;

    let device = Device::get(0)?;
    println!("Using GPU: {}", device.name()?);
    let (maj, min) = device.compute_capability()?;
    println!("  Compute capability: {maj}.{min}");
    println!(
        "  Total memory: {} MiB",
        device.total_memory()? / (1024 * 1024)
    );

    let ctx = Arc::new(Context::new(&device)?);
    let stream = Stream::new(&ctx)?;

    // Small problem size so we can verify on CPU quickly.
    let m: usize = 32;
    let n: usize = 32;
    let k: usize = 32;

    let host_a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let host_b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.1).collect();
    let mut host_c_gpu = vec![0.0f32; m * n];
    let mut host_c_cpu = vec![0.0f32; m * n];

    cpu_gemm(&host_a, &host_b, &mut host_c_cpu, m, n, k);

    let dev_a = DeviceBuffer::<f32>::from_host(&host_a)?;
    let dev_b = DeviceBuffer::<f32>::from_host(&host_b)?;
    let dev_c = DeviceBuffer::<f32>::alloc(m * n)?;

    let module = Arc::new(oxicuda::Module::from_ptx(ptx)?);
    let kernel = oxicuda::Kernel::from_module(module, "sgemm_tiled")?;

    // Grid: ceil(N/TILE) × ceil(M/TILE).
    let grid_x = (n as u32).div_ceil(TILE);
    let grid_y = (m as u32).div_ceil(TILE);
    let params = LaunchParams::builder()
        .grid(oxicuda::Dim3::new(grid_x, grid_y, 1))
        .block(oxicuda::Dim3::new(TILE, TILE, 1))
        .build();

    println!("\nLaunching sgemm_tiled: M={m}, N={n}, K={k}");
    println!("  Grid: {grid_x}×{grid_y}, Block: {TILE}×{TILE}");

    let args = (
        dev_a.as_device_ptr(),
        dev_b.as_device_ptr(),
        dev_c.as_device_ptr(),
        m as u32,
        n as u32,
        k as u32,
    );
    kernel.launch(&params, &stream, &args)?;
    stream.synchronize()?;

    dev_c.copy_to_host(&mut host_c_gpu)?;

    let mut max_err = 0.0f32;
    let mut mismatches = 0usize;
    for (i, (&gv, &cv)) in host_c_gpu.iter().zip(host_c_cpu.iter()).enumerate() {
        let err = (gv - cv).abs();
        if err > max_err {
            max_err = err;
        }
        if err > 1e-3 {
            mismatches += 1;
            if mismatches <= 3 {
                eprintln!("  Mismatch [{i}]: GPU={gv:.6}, CPU={cv:.6}");
            }
        }
    }

    if mismatches == 0 {
        println!("SUCCESS: All {m}×{n} elements correct (max_err={max_err:.2e})");
    } else {
        eprintln!("FAILED: {mismatches} mismatches (max_err={max_err:.2e})");
    }

    Ok(())
}
