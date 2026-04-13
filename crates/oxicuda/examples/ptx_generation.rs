//! PTX Code Generation Demo
//!
//! Demonstrates the OxiCUDA PTX generation DSL (`oxicuda_ptx`) without
//! requiring a GPU.  The program:
//!
//! 1. Generates a ReLU elementwise kernel via `KernelBuilder`.
//! 2. Generates a sigmoid elementwise kernel (using ex2/approx ops).
//! 3. Generates a parallel-reduction (sum) kernel.
//! 4. Prints the generated PTX source for each kernel.
//! 5. Optionally launches the ReLU kernel on GPU if one is available.
//!
//! The PTX generation steps (1–4) run entirely on the CPU — no NVIDIA
//! hardware or driver is needed.
//!
//! # Running
//!
//! ```bash
//! cargo run --example ptx_generation -p oxicuda --features "ptx"
//! ```

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

// ─── ReLU kernel ────────────────────────────────────────────────────────────

/// Generate PTX for an elementwise ReLU kernel:  `y[i] = max(0.0, x[i])`.
///
/// Thread layout: 1-D grid, one thread per element.
fn generate_relu_kernel() -> Result<String, oxicuda_ptx::error::PtxGenError> {
    KernelBuilder::new("relu_f32")
        .target(SmVersion::Sm80)
        .param("x", PtxType::U64) // const float* input
        .param("y", PtxType::U64) // float*       output
        .param("n", PtxType::U32) // element count
        .max_threads_per_block(256)
        .body(|b| {
            let gid = b.global_thread_id_x();
            let n = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n, |b| {
                let x_base = b.load_param_u64("x");
                let y_base = b.load_param_u64("y");

                // Compute element address and load x[gid].
                let x_addr = b.f32_elem_addr(x_base, gid.clone());
                let xi = b.load_global_f32(x_addr);

                // Load a zero immediate into an f32 register.
                let zero = b.mov_imm_u32(0); // allocate register slot
                b.raw_ptx(&format!("mov.f32 {zero}, 0f00000000;"));

                // ReLU = max(xi, 0.0).
                let yi = b.max_f32(xi, zero);

                // Store result at y[gid].
                let y_addr = b.f32_elem_addr(y_base, gid.clone());
                b.store_global_f32(y_addr, yi);
            });

            b.ret();
        })
        .build()
}

// ─── Sigmoid kernel ─────────────────────────────────────────────────────────

/// Generate PTX for an elementwise sigmoid kernel: `y[i] = 1 / (1 + exp(-x[i]))`.
///
/// Implemented via the PTX approximate `ex2.approx.f32` (base-2 exponent):
/// ```text
/// sigmoid(x) = 1 / (1 + 2^(−x · log2(e)))
/// ```
///
/// All immediate constants use PTX hex float literals (IEEE 754 bit patterns).
fn generate_sigmoid_kernel() -> Result<String, oxicuda_ptx::error::PtxGenError> {
    // log2(e) ≈ 1.44269504 — used to convert natural-log exponent to base-2.
    let log2e: f32 = std::f32::consts::LOG2_E;
    let log2e_bits = log2e.to_bits();
    // 1.0f32 bit pattern = 0x3F800000.
    let one_bits: u32 = 1.0f32.to_bits();

    KernelBuilder::new("sigmoid_f32")
        .target(SmVersion::Sm80)
        .param("x", PtxType::U64)
        .param("y", PtxType::U64)
        .param("n", PtxType::U32)
        .max_threads_per_block(256)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let n = b.load_param_u32("n");

            b.if_lt_u32(gid.clone(), n, |b| {
                let x_base = b.load_param_u64("x");
                let y_base = b.load_param_u64("y");

                let x_addr = b.f32_elem_addr(x_base, gid.clone());
                let xi = b.load_global_f32(x_addr);

                // neg_x = −xi
                let neg_x = b.neg_f32(xi);

                // neg_x_log2e = neg_x × log2(e)   (constant mul via raw PTX).
                let neg_x_log2e = b.mov_imm_u32(0);
                b.raw_ptx(&format!(
                    "mul.f32 {neg_x_log2e}, {neg_x}, 0f{log2e_bits:08X};"
                ));

                // ex2_val = 2^(neg_x_log2e)  ≈ exp(−xi)
                let ex2_val = b.ex2_approx_f32(neg_x_log2e);

                // one_plus = 1.0 + ex2_val
                let one_plus = b.mov_imm_u32(0);
                b.raw_ptx(&format!("add.f32 {one_plus}, {ex2_val}, 0f{one_bits:08X};"));

                // sigmoid = 1 / (1 + exp(−x))
                let sig = b.rcp_approx_f32(one_plus);

                let y_addr = b.f32_elem_addr(y_base, gid.clone());
                b.store_global_f32(y_addr, sig);
            });

            b.ret();
        })
        .build()
}

// ─── Parallel reduction (sum) kernel ────────────────────────────────────────

/// Generate PTX for a block-level parallel tree-reduction sum kernel.
///
/// Algorithm:
/// 1. Each thread loads `x[gid]` into shared memory (0.0 if out of bounds).
/// 2. `bar.sync 0` ensures the tile is fully populated.
/// 3. Iterative halving: at each step, the first `stride` threads add the
///    element at `smem[tid + stride]` into `smem[tid]`.
/// 4. Thread 0 atomically adds the block sum into `result[0]`.
fn generate_reduction_kernel() -> Result<String, oxicuda_ptx::error::PtxGenError> {
    const BLOCK: u32 = 256;

    KernelBuilder::new("sum_reduce_f32")
        .target(SmVersion::Sm80)
        .param("x", PtxType::U64) // const float* input
        .param("result", PtxType::U64) // float*       output (one element)
        .param("n", PtxType::U32) // element count
        .shared_mem("smem", PtxType::F32, BLOCK as usize)
        .max_threads_per_block(BLOCK)
        .body(|b| {
            let gid = b.global_thread_id_x();
            let tid = b.thread_id_x();
            let n = b.load_param_u32("n");
            let x_base = b.load_param_u64("x");

            // Compute smem[tid] address once.
            let smem_tid_addr = {
                let ptr = b.mov_imm_u32(0);
                b.raw_ptx(&format!("cvta.to.shared.u64 {ptr}, smem;"));
                b.f32_elem_addr(ptr, tid.clone())
            };

            // Load x[gid] into smem[tid], or 0.0 if gid >= n.
            let zero_f32 = b.mov_imm_u32(0);
            b.raw_ptx(&format!("mov.f32 {zero_f32}, 0f00000000;"));
            b.store_shared_f32(smem_tid_addr.clone(), zero_f32);
            b.if_lt_u32(gid.clone(), n.clone(), |b| {
                let x_addr = b.f32_elem_addr(x_base, gid.clone());
                let xv = b.load_global_f32(x_addr);
                b.store_shared_f32(smem_tid_addr.clone(), xv);
            });
            b.bar_sync(0);

            b.comment("tree reduction: strides 128, 64, 32, 16, 8, 4, 2, 1");
            for stride in [128u32, 64, 32, 16, 8, 4, 2, 1] {
                // Only threads with tid < stride participate.
                // Evaluate the upper-bound immediate BEFORE the if_lt_u32 call
                // to avoid a double-mutable-borrow of `b`.
                let stride_upper = b.mov_imm_u32(stride);
                b.if_lt_u32(tid.clone(), stride_upper, |b| {
                    // partner index = tid + stride
                    let stride_reg = b.mov_imm_u32(stride);
                    let partner_idx = b.add_u32(tid.clone(), stride_reg);
                    let partner_addr = {
                        let ptr = b.mov_imm_u32(0);
                        b.raw_ptx(&format!("cvta.to.shared.u64 {ptr}, smem;"));
                        b.f32_elem_addr(ptr, partner_idx)
                    };
                    let pv = b.load_shared_f32(partner_addr);
                    let sv = b.load_shared_f32(smem_tid_addr.clone());
                    let sum = b.add_f32(sv, pv);
                    b.store_shared_f32(smem_tid_addr.clone(), sum);
                });
                b.bar_sync(0);
            }

            // Thread 0 atomically accumulates the block sum into result[0].
            let one_bound = b.mov_imm_u32(1);
            b.if_lt_u32(tid.clone(), one_bound, |b| {
                let block_sum = b.load_shared_f32(smem_tid_addr.clone());
                let res_base = b.load_param_u64("result");
                let _old = b.atom_global_add_f32(res_base, block_sum);
            });

            b.ret();
        })
        .build()
}

// ─── Optional GPU launch ────────────────────────────────────────────────────

/// Attempt to launch the ReLU kernel on an available GPU.
fn try_gpu_relu(relu_ptx: &str) -> Result<(), Box<dyn std::error::Error>> {
    use oxicuda::prelude::*;
    use oxicuda::{DeviceBuffer, LaunchParams};
    use std::sync::Arc;

    oxicuda::init()?;

    let device = Device::get(0)?;
    let (maj, min) = device.compute_capability()?;
    println!("  GPU: {} ({maj}.{min})", device.name()?);

    let ctx = Arc::new(Context::new(&device)?);
    let stream = Stream::new(&ctx)?;

    let n: u32 = 1024;
    // Input: values ranging from −512 to +511.
    let host_x: Vec<f32> = (0..n).map(|i| (i as f32) - (n / 2) as f32).collect();
    let mut host_y = vec![0.0f32; n as usize];

    let dev_x = DeviceBuffer::<f32>::from_host(&host_x)?;
    let dev_y = DeviceBuffer::<f32>::alloc(n as usize)?;

    let module = Arc::new(oxicuda::Module::from_ptx(relu_ptx)?);
    let kernel = oxicuda::Kernel::from_module(module, "relu_f32")?;

    let block = 256u32;
    let grid = n.div_ceil(block);
    let params = LaunchParams::new(grid, block);
    let args = (dev_x.as_device_ptr(), dev_y.as_device_ptr(), n);
    kernel.launch(&params, &stream, &args)?;
    stream.synchronize()?;

    dev_y.copy_to_host(&mut host_y)?;

    let mismatches = host_x
        .iter()
        .zip(host_y.iter())
        .filter(|&(&x, &y)| {
            let expected = x.max(0.0);
            (y - expected).abs() > 1e-6
        })
        .count();

    if mismatches == 0 {
        println!("  ReLU GPU kernel: PASSED ({n} elements)");
    } else {
        eprintln!("  ReLU GPU kernel: FAILED ({mismatches} mismatches)");
    }

    Ok(())
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiCUDA PTX Generation Demo ===\n");

    // 1. ReLU kernel — print a preview.
    println!("--- 1. ReLU (elementwise max(0, x)) ---");
    let relu_ptx = generate_relu_kernel()?;
    println!("Generated {} bytes of PTX.", relu_ptx.len());
    let preview = relu_ptx.len().min(500);
    println!("{}", &relu_ptx[..preview]);
    if relu_ptx.len() > preview {
        println!("... ({} more chars)", relu_ptx.len() - preview);
    }
    println!();

    // 2. Sigmoid kernel.
    println!("--- 2. Sigmoid (1 / (1 + exp(−x))) ---");
    let sigmoid_ptx = generate_sigmoid_kernel()?;
    println!("Generated {} bytes of PTX.\n", sigmoid_ptx.len());

    // 3. Reduction kernel.
    println!("--- 3. Parallel sum reduction (block-level) ---");
    let reduce_ptx = generate_reduction_kernel()?;
    println!("Generated {} bytes of PTX.\n", reduce_ptx.len());

    // Summary of generated kernels.
    println!("Kernels generated:");
    println!("  relu_f32          : {} bytes", relu_ptx.len());
    println!("  sigmoid_f32       : {} bytes", sigmoid_ptx.len());
    println!("  sum_reduce_f32    : {} bytes", reduce_ptx.len());
    println!();

    // 4. Optional GPU launch.
    println!("--- 4. Attempting GPU launch of relu_f32 ---");
    match try_gpu_relu(&relu_ptx) {
        Ok(()) => println!("GPU launch succeeded!"),
        Err(e) => println!(
            "GPU not available: {} (expected on macOS / no-GPU systems)",
            e
        ),
    }

    Ok(())
}
