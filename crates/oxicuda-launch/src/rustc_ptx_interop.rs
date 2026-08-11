//! Foreign-compiler PTX interop: launching `rustc`-generated kernels.
//!
//! Everything else in this workspace launches PTX produced by `oxicuda-ptx`'s
//! own code generator. These tests instead launch PTX emitted by **upstream
//! `rustc`'s LLVM NVPTX backend** (the `nvptx64-nvidia-cuda` target), proving
//! that the driver/launch stack is a general PTX host — including for the
//! Rust-kernels-on-GPU ecosystem (Rust-CUDA / Rust-GPU / VectorWare's
//! portable-SIMD work), whose output is exactly this kind of module.
//!
//! ## Fixture provenance
//!
//! Both fixtures were compiled from the sources below with:
//!
//! ```text
//! rustc +nightly (1.99.0-nightly 2026-08-10) \
//!     --target nvptx64-nvidia-cuda -O \
//!     -C target-cpu=sm_86 -C unsafe-allow-abi-mismatch=target-cpu \
//!     --crate-type rlib --emit asm=<name>.ptx <name>.rs
//! ```
//!
//! (The ABI-mismatch escape is sound here: the kernels are fully
//! monomorphized in-crate and cross the boundary only through the C-like
//! `ptx-kernel` ABI.)
//!
//! `rustc_saxpy.ptx` — scalar SIMT:
//!
//! ```ignore
//! #![no_std]
//! #![feature(abi_ptx, stdarch_nvptx)]
//! use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
//! #[panic_handler]
//! fn panic(_: &core::panic::PanicInfo<'_>) -> ! { loop {} }
//! #[no_mangle]
//! pub unsafe extern "ptx-kernel" fn rust_saxpy(y: *mut f32, x: *const f32, a: f32, n: u32) {
//!     let i = (_block_idx_x() as u32) * (_block_dim_x() as u32) + (_thread_idx_x() as u32);
//!     if i < n {
//!         let xi = unsafe { *x.add(i as usize) };
//!         let yi = unsafe { y.add(i as usize) };
//!         unsafe { *yi = a * xi + *yi };
//!     }
//! }
//! ```
//!
//! `rustc_simd_relu_dot.ptx` — `core::simd` (portable SIMD), one `f32x4`
//! relu-dot per thread:
//!
//! ```ignore
//! #![no_std]
//! #![feature(abi_ptx, stdarch_nvptx, portable_simd)]
//! use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
//! use core::simd::f32x4;
//! use core::simd::num::SimdFloat;
//! #[panic_handler]
//! fn panic(_: &core::panic::PanicInfo<'_>) -> ! { loop {} }
//! #[no_mangle]
//! pub unsafe extern "ptx-kernel" fn rust_simd_relu_dot(
//!     out: *mut f32, x: *const f32, y: *const f32, n_chunks: u32,
//! ) {
//!     let t = (_block_idx_x() as u32) * (_block_dim_x() as u32) + (_thread_idx_x() as u32);
//!     if t < n_chunks {
//!         let base = (t as usize) * 4;
//!         let xv = f32x4::from_array(unsafe { *x.add(base).cast::<[f32; 4]>() });
//!         let yv = f32x4::from_array(unsafe { *y.add(base).cast::<[f32; 4]>() });
//!         let relu = (xv * yv).simd_max(f32x4::splat(0.0));
//!         unsafe { *out.add(t as usize) = relu.reduce_sum() };
//!     }
//! }
//! ```
//!
//! ## What the portable-SIMD fixture demonstrates (and what it does not)
//!
//! Upstream rustc compiles `core::simd` for nvptx64 today, but lowers the
//! vector **within a single thread**: the fixture PTX contains four scalar
//! `mul.rn.f32` / `max.f32` per chunk and a sequential `add.rn.f32` tree for
//! `reduce_sum` — no `shfl.sync`, no lane-to-warp mapping. Mapping
//! `Simd<T, 32>` onto the 32 warp lanes (one lane per hardware thread, with
//! reductions as warp shuffles) is downstream compiler work (VectorWare),
//! not upstream behavior. The warp-mapped programming model is available in
//! this workspace on stable Rust via `oxicuda_ptx::builder::warp_vec`.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_memory::DeviceBuffer;

use crate::kernel::Kernel;
use crate::params::LaunchParams;

/// Scalar SIMT saxpy compiled by upstream rustc (see module docs).
const RUSTC_SAXPY_PTX: &str = include_str!("fixtures/rustc_saxpy.ptx");

/// Thread-level `core::simd` relu-dot compiled by upstream rustc.
const RUSTC_SIMD_RELU_DOT_PTX: &str = include_str!("fixtures/rustc_simd_relu_dot.ptx");

/// Acquires a context, or `None` when no GPU / driver is present (the tests
/// self-skip on CPU-only machines; JIT or launch failures beyond this point
/// are real failures and panic).
fn gpu_context() -> Option<Arc<Context>> {
    oxicuda_driver::init().ok()?;
    if Device::count().ok()? == 0 {
        return None;
    }
    let dev = Device::get(0).ok()?;
    Context::new(&dev).ok().map(Arc::new)
}

/// `rustc`-compiled scalar SIMT saxpy: JIT, launch across two blocks, and
/// match the CPU oracle exactly.
#[test]
fn rustc_scalar_saxpy_launches_and_matches_oracle() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    const N: usize = 256;
    let a = 2.5_f32;
    let x: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();
    let expected: Vec<f32> = x.iter().zip(&y).map(|(&xi, &yi)| a * xi + yi).collect();

    let module = Module::from_ptx(RUSTC_SAXPY_PTX)
        .unwrap_or_else(|e| panic!("driver JIT rejected rustc PTX: {e}"));
    let kernel = Kernel::from_module(Arc::new(module), "rust_saxpy").expect("entry symbol");
    let stream = Stream::new(&ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_y = DeviceBuffer::<f32>::from_host(&y).expect("d_y");
    // Two blocks of 128 so `%ctaid.x` participates in the indexing.
    kernel
        .launch(
            &LaunchParams::new(2, 128),
            &stream,
            &(d_y.as_device_ptr(), d_x.as_device_ptr(), a, N as u32),
        )
        .expect("launch rust_saxpy");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; N];
    d_y.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "rustc saxpy mismatch");
}

/// `rustc`-compiled `core::simd` kernel: JIT, launch, and match the CPU
/// oracle exactly (inputs are small integers, so every product, `max`, and
/// 4-term sum is exact in f32).
#[test]
fn rustc_portable_simd_kernel_launches_and_matches_oracle() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    const CHUNKS: usize = 64;
    const N: usize = CHUNKS * 4;
    let x: Vec<f32> = (0..N).map(|i| ((i % 13) as f32) - 6.0).collect();
    let y: Vec<f32> = (0..N).map(|i| ((i % 7) as f32) - 3.0).collect();
    let expected: Vec<f32> = (0..CHUNKS)
        .map(|c| {
            (0..4)
                .map(|j| (x[c * 4 + j] * y[c * 4 + j]).max(0.0))
                .sum()
        })
        .collect();

    let module = Module::from_ptx(RUSTC_SIMD_RELU_DOT_PTX)
        .unwrap_or_else(|e| panic!("driver JIT rejected rustc portable-SIMD PTX: {e}"));
    let kernel =
        Kernel::from_module(Arc::new(module), "rust_simd_relu_dot").expect("entry symbol");
    let stream = Stream::new(&ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_y = DeviceBuffer::<f32>::from_host(&y).expect("d_y");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32; CHUNKS]).expect("d_out");
    kernel
        .launch(
            &LaunchParams::new(1, CHUNKS as u32),
            &stream,
            &(
                d_out.as_device_ptr(),
                d_x.as_device_ptr(),
                d_y.as_device_ptr(),
                CHUNKS as u32,
            ),
        )
        .expect("launch rust_simd_relu_dot");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; CHUNKS];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "rustc portable-SIMD relu_dot mismatch");
}
