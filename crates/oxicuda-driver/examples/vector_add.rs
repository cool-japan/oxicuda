//! `vector_add` — end-to-end PTX kernel launch example.
//!
//! This example demonstrates the full OxiCUDA pipeline:
//! 1. Load the CUDA driver at runtime
//! 2. Create a device context
//! 3. Write a `vector_add` PTX kernel as a string literal
//! 4. JIT-compile the PTX via `Module::from_ptx_with_options`
//! 5. Allocate device buffers for inputs A, B and output C
//! 6. Copy host data to device (H2D)
//! 7. Launch the kernel
//! 8. Copy results back to host (D2H)
//! 9. Verify correctness and print the first few results
//!
//! Run with:
//! ```sh
//! cargo run --example vector_add -p oxicuda-driver
//! ```
//!
//! Expected output on a machine with a GPU:
//! ```text
//! Device: NVIDIA A100-SXM4-80GB  (sm_80)
//! JIT info: ptxas info    : 'vector_add' used 8 registers, 0 bytes smem ...
//! C[0] = 1.0 + 2.0 = 3.0  ✓
//! C[1] = 2.0 + 4.0 = 6.0  ✓
//! C[2] = 3.0 + 6.0 = 9.0  ✓
//! C[3] = 4.0 + 8.0 = 12.0 ✓
//! All 1024 elements verified.
//! ```

use std::ffi::c_void;
use std::sync::Arc;

use oxicuda_driver::{
    Context, CudaResult, Device, JitOptions, Module, Stream, ffi::CUdeviceptr, loader::try_driver,
};

// ── PTX source builder ────────────────────────────────────────────────────────

/// Map `(major, minor)` compute capability to a `(ptx_major, ptx_minor, sm_target)` triple.
fn ptx_version_for_cc(major: i32, minor: i32) -> (u32, u32, String) {
    let sm_str = format!("sm_{major}{minor}");
    let (pm, pn) = match (major, minor) {
        (7, 5) => (7, 4),
        (8, 0) | (8, 6) => (7, 5),
        (8, 9) => (8, 0),
        (9, 0) | (9, 1) => (8, 0),
        (10, _) => (8, 5),
        (12, _) => (8, 7),
        _ => (7, 4), // conservative fallback
    };
    (pm, pn, sm_str)
}

/// Build a PTX source string for the `vector_add` kernel targeting the device.
///
/// Computes `C[i] = A[i] + B[i]` for `f32` arrays; one thread per element.
fn make_vector_add_ptx(cc_major: i32, cc_minor: i32) -> String {
    let (ptx_major, ptx_minor, sm_str) = ptx_version_for_cc(cc_major, cc_minor);

    format!(
        r#".version {ptx_major}.{ptx_minor}
.target {sm_str}
.address_size 64

// vector_add: C[tid] = A[tid] + B[tid]  (f32, one thread per element)
.visible .entry vector_add(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{{
    .reg .u32  %r<4>;
    .reg .u64  %rd<5>;
    .reg .f32  %f<3>;
    .reg .pred %p0;

    // tid = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32     %r0, %tid.x;
    mov.u32     %r1, %ntid.x;
    mov.u32     %r2, %ctaid.x;
    mad.lo.u32  %r0, %r2, %r1, %r0;

    // Bounds check.
    ld.param.u32    %r3, [param_n];
    setp.ge.u32     %p0, %r0, %r3;
    @%p0 bra        $exit;

    // Load base pointers.
    ld.param.u64    %rd0, [param_a];
    ld.param.u64    %rd1, [param_b];
    ld.param.u64    %rd2, [param_c];

    // Byte offset = tid * sizeof(f32) = tid << 2.
    cvt.u64.u32     %rd3, %r0;
    shl.b64         %rd3, %rd3, 2;

    // A[tid]
    add.u64         %rd4, %rd0, %rd3;
    ld.global.f32   %f0, [%rd4];

    // B[tid]
    add.u64         %rd4, %rd1, %rd3;
    ld.global.f32   %f1, [%rd4];

    // C[tid] = A[tid] + B[tid]
    add.f32         %f2, %f0, %f1;
    add.u64         %rd4, %rd2, %rd3;
    st.global.f32   [%rd4], %f2;

$exit:
    ret;
}}
"#
    )
}

// ── Low-level device memory helpers ──────────────────────────────────────────
//
// We use `oxicuda_driver::cuda_call!` (which is `#[macro_export]`) to wrap
// raw `DriverApi` calls in a safe `check(unsafe { ... })` invocation.

/// Allocate `count * size_of::<T>()` bytes of device memory.
fn device_alloc<T>(count: usize) -> CudaResult<CUdeviceptr> {
    let api = try_driver()?;
    let bytes = count * std::mem::size_of::<T>();
    let mut ptr: CUdeviceptr = 0;
    oxicuda_driver::cuda_call!((api.cu_mem_alloc_v2)(&mut ptr, bytes))?;
    Ok(ptr)
}

/// Synchronously copy `src` (host slice) → `dst` (device pointer).
fn htod<T>(dst: CUdeviceptr, src: &[T]) -> CudaResult<()> {
    let api = try_driver()?;
    let bytes = std::mem::size_of_val(src);
    oxicuda_driver::cuda_call!((api.cu_memcpy_htod_v2)(
        dst,
        src.as_ptr().cast::<c_void>(),
        bytes
    ))
}

/// Synchronously copy `src` (device pointer) → `dst` (host slice).
fn dtoh<T>(dst: &mut [T], src: CUdeviceptr) -> CudaResult<()> {
    let api = try_driver()?;
    let bytes = std::mem::size_of_val(dst);
    oxicuda_driver::cuda_call!((api.cu_memcpy_dtoh_v2)(
        dst.as_mut_ptr().cast::<c_void>(),
        src,
        bytes
    ))
}

/// Free a raw device pointer; ignores errors (safe to call from cleanup paths).
fn device_free(ptr: CUdeviceptr) {
    if let Ok(api) = try_driver() {
        let _ = unsafe { (api.cu_mem_free_v2)(ptr) };
    }
}

// ── Kernel launch helper ──────────────────────────────────────────────────────

/// Launch `func` over a `(grid_x × 1 × 1)` grid of `(block_x × 1 × 1)` blocks.
///
/// `params` is a slice of `*mut c_void` where each element points to the
/// corresponding kernel argument value (the standard CUDA launch convention).
fn launch(
    func: &oxicuda_driver::Function,
    grid_x: u32,
    block_x: u32,
    stream: &Stream,
    params: &[*mut c_void],
) -> CudaResult<()> {
    let api = try_driver()?;
    oxicuda_driver::cuda_call!((api.cu_launch_kernel)(
        func.raw(),
        grid_x,
        1,
        1,
        block_x,
        1,
        1,
        0, // shared memory bytes
        stream.raw(),
        params.as_ptr() as *mut *mut c_void,
        std::ptr::null_mut(), // extra
    ))
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    // ── 1. Initialise the driver ─────────────────────────────────────────────
    if let Err(e) = oxicuda_driver::init() {
        eprintln!("Failed to load CUDA driver: {e}");
        eprintln!("Is an NVIDIA GPU driver installed?");
        std::process::exit(1);
    }

    // ── 2. Select device and create context ──────────────────────────────────
    let dev = match Device::get(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("No GPU found: {e}");
            std::process::exit(1);
        }
    };

    let (cc_major, cc_minor) = dev.compute_capability().unwrap_or((7, 5));
    println!(
        "Device: {}  (sm_{cc_major}{cc_minor})",
        dev.name().unwrap_or_else(|_| "<unknown>".into())
    );

    let ctx = match Context::new(&dev) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("Failed to create CUDA context: {e}");
            std::process::exit(1);
        }
    };

    // ── 3. Generate and JIT-compile the PTX ─────────────────────────────────
    let ptx = make_vector_add_ptx(cc_major, cc_minor);

    let opts = JitOptions {
        optimization_level: 4,
        target_from_context: true,
        ..Default::default()
    };

    let (module, log) = match Module::from_ptx_with_options(&ptx, &opts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("PTX JIT compilation failed: {e}");
            std::process::exit(1);
        }
    };

    if !log.info.is_empty() {
        println!("JIT info: {}", log.info);
    }
    if log.has_errors() {
        eprintln!("JIT errors:\n{}", log.error);
        for d in log.errors() {
            eprintln!(
                "  [{sev}]{kern} {msg}",
                sev = d.severity,
                kern = d
                    .kernel
                    .as_deref()
                    .map(|k| format!(" ({k})"))
                    .unwrap_or_default(),
                msg = d.message,
            );
        }
        std::process::exit(1);
    }

    let func = match module.get_function("vector_add") {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Kernel lookup failed: {e}");
            std::process::exit(1);
        }
    };

    // ── 4. Prepare host data ─────────────────────────────────────────────────
    const N: usize = 1024;
    const BLOCK: u32 = 256;
    const GRID: u32 = (N as u32).div_ceil(BLOCK);

    let h_a: Vec<f32> = (1..=N).map(|i| i as f32).collect();
    let h_b: Vec<f32> = (1..=N).map(|i| (i * 2) as f32).collect();
    let mut h_c = vec![0.0f32; N];

    // ── 5. Allocate device memory ────────────────────────────────────────────
    let d_a = match device_alloc::<f32>(N) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Device alloc A: {e}");
            std::process::exit(1);
        }
    };
    let d_b = match device_alloc::<f32>(N) {
        Ok(p) => p,
        Err(e) => {
            device_free(d_a);
            eprintln!("Device alloc B: {e}");
            std::process::exit(1);
        }
    };
    let d_c = match device_alloc::<f32>(N) {
        Ok(p) => p,
        Err(e) => {
            device_free(d_a);
            device_free(d_b);
            eprintln!("Device alloc C: {e}");
            std::process::exit(1);
        }
    };

    // ── 6. Copy inputs H2D ───────────────────────────────────────────────────
    if let Err(e) = htod(d_a, &h_a) {
        eprintln!("H2D A: {e}");
        std::process::exit(1);
    }
    if let Err(e) = htod(d_b, &h_b) {
        eprintln!("H2D B: {e}");
        std::process::exit(1);
    }

    // ── 7. Create a stream and launch the kernel ──────────────────────────────
    let stream = match Stream::new(&ctx) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Stream: {e}");
            std::process::exit(1);
        }
    };

    // cuLaunchKernel expects `*mut *mut c_void` where each element is
    // a pointer to the kernel argument value.
    let n_val: u32 = N as u32;
    let mut arg_a = d_a;
    let mut arg_b = d_b;
    let mut arg_c = d_c;
    let mut arg_n = n_val;
    let params: [*mut c_void; 4] = [
        (&raw mut arg_a) as *mut c_void,
        (&raw mut arg_b) as *mut c_void,
        (&raw mut arg_c) as *mut c_void,
        (&raw mut arg_n) as *mut c_void,
    ];

    if let Err(e) = launch(&func, GRID, BLOCK, &stream, &params) {
        eprintln!("Kernel launch: {e}");
        std::process::exit(1);
    }

    // ── 8. Synchronise and copy results D2H ─────────────────────────────────
    if let Err(e) = stream.synchronize() {
        eprintln!("Sync: {e}");
        std::process::exit(1);
    }
    if let Err(e) = dtoh(&mut h_c, d_c) {
        eprintln!("D2H: {e}");
        std::process::exit(1);
    }

    // ── 9. Verify results ────────────────────────────────────────────────────
    let mut fail = 0usize;
    for i in 0..N {
        let expected = h_a[i] + h_b[i];
        if (h_c[i] - expected).abs() > 1e-5 {
            eprintln!("Mismatch [{i}]: got {}, expected {}", h_c[i], expected);
            fail += 1;
        }
    }

    for i in 0..4.min(N) {
        let ok = (h_c[i] - (h_a[i] + h_b[i])).abs() < 1e-5;
        println!(
            "C[{i}] = {} + {} = {}  {}",
            h_a[i],
            h_b[i],
            h_c[i],
            if ok { "\u{2713}" } else { "\u{2717}" }
        );
    }

    if fail == 0 {
        println!("All {N} elements verified.");
    } else {
        eprintln!("{fail} verification failure(s)!");
        std::process::exit(1);
    }

    // ── 10. Cleanup ──────────────────────────────────────────────────────────
    device_free(d_a);
    device_free(d_b);
    device_free(d_c);
}
