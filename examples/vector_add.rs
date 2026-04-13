//! Vector Addition Example
//!
//! Demonstrates the complete OxiCUDA workflow:
//! 1. Initialize CUDA driver
//! 2. Create context and stream
//! 3. Allocate device memory
//! 4. Load PTX kernel
//! 5. Launch kernel
//! 6. Read back results
//!
//! # Running
//!
//! This example requires a system with an NVIDIA GPU and driver installed.
//! ```bash
//! cargo run --example vector_add
//! ```

use std::sync::Arc;

use oxicuda::prelude::*;
use oxicuda::{DeviceBuffer, JitOptions, LaunchParams};

/// PTX source for a simple vector addition kernel.
///
/// This is hand-written PTX that adds two f32 vectors element-wise:
/// c[i] = a[i] + b[i]
const VECTOR_ADD_PTX: &str = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry vector_add(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
)
{
    .reg .pred %p0;
    .reg .f32 %f<3>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<7>;

    // Global thread index
    mov.u32         %r0, %ctaid.x;
    mov.u32         %r1, %ntid.x;
    mov.u32         %r2, %tid.x;
    mad.lo.u32      %r3, %r0, %r1, %r2;

    // Bounds check
    ld.param.u32    %r1, [n];
    setp.ge.u32     %p0, %r3, %r1;
    @%p0 bra        DONE;

    // Load parameters
    ld.param.u64    %rd0, [a];
    ld.param.u64    %rd1, [b];
    ld.param.u64    %rd2, [c];

    // Compute byte offset: idx * 4
    mul.wide.u32    %rd3, %r3, 4;

    // Load a[i] and b[i]
    add.u64         %rd4, %rd0, %rd3;
    add.u64         %rd5, %rd1, %rd3;
    ld.global.f32   %f0, [%rd4];
    ld.global.f32   %f1, [%rd5];

    // c[i] = a[i] + b[i]
    add.f32         %f2, %f0, %f1;

    // Store result
    add.u64         %rd6, %rd2, %rd3;
    st.global.f32   [%rd6], %f2;

DONE:
    ret;
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    oxicuda::init()?;

    // Get the first GPU device
    let device = Device::get(0)?;
    println!("Using GPU: {}", device.name()?);
    println!("  Compute capability: {}.{}",
        device.compute_capability()?.0,
        device.compute_capability()?.1);
    println!("  Total memory: {} MB", device.total_memory()? / (1024 * 1024));

    // Create context and stream
    let ctx = Arc::new(Context::new(&device)?);
    let stream = Stream::new(&ctx)?;

    // Problem size
    let n: u32 = 1024 * 1024; // 1M elements

    // Prepare host data
    let host_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let host_b: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();
    let mut host_c = vec![0.0f32; n as usize];

    // Allocate device memory and copy input data
    let mut dev_a = DeviceBuffer::<f32>::from_host(&host_a)?;
    let mut dev_b = DeviceBuffer::<f32>::from_host(&host_b)?;
    let mut dev_c = DeviceBuffer::<f32>::alloc(n as usize)?;

    // Load PTX module and get kernel function
    let module = Arc::new(Module::from_ptx(VECTOR_ADD_PTX)?);
    let kernel = Kernel::from_module(module, "vector_add")?;

    // Configure launch parameters
    let block_size = 256u32;
    let grid_size = grid_size_for(n, block_size);
    let params = LaunchParams::new(grid_size, block_size);

    println!("\nLaunching kernel: grid={}, block={}, n={}", grid_size, block_size, n);

    // Launch kernel
    // Arguments: (a_ptr, b_ptr, c_ptr, n)
    let args = (
        dev_a.as_device_ptr(),
        dev_b.as_device_ptr(),
        dev_c.as_device_ptr(),
        n,
    );
    kernel.launch(&params, &stream, &args)?;

    // Wait for completion
    stream.synchronize()?;

    // Copy results back to host
    dev_c.copy_to_host(&mut host_c)?;

    // Verify results
    let mut errors = 0u32;
    for i in 0..n as usize {
        let expected = host_a[i] + host_b[i];
        if (host_c[i] - expected).abs() > 1e-5 {
            errors += 1;
            if errors <= 5 {
                eprintln!("Mismatch at {}: expected {}, got {}", i, expected, host_c[i]);
            }
        }
    }

    if errors == 0 {
        println!("SUCCESS: All {} elements verified correctly!", n);
    } else {
        eprintln!("FAILED: {} mismatches out of {} elements", errors, n);
    }

    Ok(())
}
