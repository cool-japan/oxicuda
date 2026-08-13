//! On-device numeric coverage for [`super::vector_mem_ops`]: vectorized
//! shared-memory load/store and TF32 rounding, JIT-compiled and launched on
//! the real CUDA device rather than only checked as emitted PTX text.
//!
//! Requires the `gpu-tests` feature and, to do anything beyond a no-op, a
//! live CUDA driver + device; every test returns early (skips) when neither
//! is present, matching every other `gpu-tests`-gated suite in this
//! workspace (see `crate::arch::gpu_tests` for the sibling convention this
//! mirrors).

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::arch::SmVersion;
use crate::builder::KernelBuilder;
use crate::ir::PtxType;

// ---------------------------------------------------------------------------
// Fixture & helpers
// ---------------------------------------------------------------------------

/// A live CUDA context + stream, plus the device's SM version.
struct GpuFixture {
    _ctx: Arc<Context>,
    stream: Stream,
    sm: SmVersion,
}

/// Acquires a GPU fixture, or `None` when no driver / device is present.
fn gpu_fixture() -> Option<GpuFixture> {
    oxicuda_driver::init().ok()?;
    if Device::count().ok()? == 0 {
        return None;
    }
    let dev = Device::get(0).ok()?;
    let (major, minor) = dev.compute_capability().ok()?;
    let sm = SmVersion::from_compute_capability(major, minor).unwrap_or(SmVersion::Sm80);
    let ctx = Arc::new(Context::new(&dev).ok()?);
    let stream = Stream::new(&ctx).ok()?;
    Some(GpuFixture {
        _ctx: ctx,
        stream,
        sm,
    })
}

/// JIT-compiles `ptx` and looks up `entry`, panicking (with the PTX text)
/// on failure -- a compile/lookup failure here is a real bug in the emitted
/// PTX, not something a test should silently skip past.
fn load_kernel(ptx: &str, entry: &str) -> Kernel {
    let module = Module::from_ptx(ptx).unwrap_or_else(|e| {
        panic!("PTX JIT compile failed for `{entry}`: {e}\n--- PTX ---\n{ptx}")
    });
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel lookup failed for `{entry}`: {e}"))
}

/// A single-block, single-thread launch -- every kernel in this file does a
/// fixed amount of work with no dependence on thread/block index, so there
/// is nothing to gain from a wider launch.
fn single_thread_params() -> LaunchParams {
    LaunchParams::new(Dim3::new(1, 1, 1), Dim3::new(1, 1, 1))
}

// ═══════════════════════════════════════════════════════════════════════
//  store_global_f32x4 (+ load_global_f32x4) round trip
// ═══════════════════════════════════════════════════════════════════════

/// `load_global_f32x4` then `store_global_f32x4` must reproduce the input
/// bit-for-bit: pure data movement, no arithmetic transformation.
#[test]
fn global_f32x4_roundtrip_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let ptx = KernelBuilder::new("test_global_f32x4_roundtrip")
        .target(fx.sm)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .body(|b| {
            let in_ptr = b.load_param_u64("in_ptr");
            let out_ptr = b.load_param_u64("out_ptr");
            let vals = b.load_global_f32x4(&in_ptr);
            b.store_global_f32x4(&out_ptr, &vals);
            b.ret();
        })
        .build()
        .expect("PTX generation must succeed");

    let kernel = load_kernel(&ptx, "test_global_f32x4_roundtrip");

    let input = [1.0f32, -2.5, 3.0e10, 1.0 / 3.0];
    let in_buf = DeviceBuffer::<f32>::from_host(&input).expect("in_buf alloc");
    let out_buf = DeviceBuffer::<f32>::zeroed(4).expect("out_buf alloc");

    let params = single_thread_params();
    let args = (in_buf.as_device_ptr(), out_buf.as_device_ptr());
    kernel
        .launch(&params, &fx.stream, &args)
        .expect("kernel launch");
    fx.stream.synchronize().expect("synchronize");

    let mut output = [0.0f32; 4];
    out_buf.copy_to_host(&mut output).expect("copy_to_host");

    // Compare IEEE-754 bit patterns rather than `f32 == f32`: this is
    // deliberately an exact reproduction check (pure data movement, no
    // arithmetic), and bit-pattern equality is the precise way to express
    // that intent (and sidesteps `clippy::float_cmp`, which exists to catch
    // the *unintentional* use of `==` after a floating-point computation).
    let output_bits = output.map(f32::to_bits);
    let input_bits = input.map(f32::to_bits);
    assert_eq!(
        output_bits, input_bits,
        "load_global_f32x4 + store_global_f32x4 must reproduce the input \
         bit-for-bit; got {output:?}, expected {input:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  load_shared_f32x4 / store_shared_f32x4 round trip
// ═══════════════════════════════════════════════════════════════════════

/// Global -> shared (`store_shared_f32x4`) -> shared (`load_shared_f32x4`)
/// -> global round trip through a 16-byte-aligned `shared_mem_aligned`
/// allocation must reproduce the input bit-for-bit.
#[test]
fn shared_f32x4_roundtrip_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let ptx = KernelBuilder::new("test_shared_f32x4_roundtrip")
        .target(fx.sm)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        // 16-byte-aligned -- the alignment `ld`/`st.shared.v4.f32` require;
        // the default `shared_mem`'s 4-byte alignment would be insufficient.
        .shared_mem_aligned("tile", PtxType::F32, 4, 16)
        .body(|b| {
            let in_ptr = b.load_param_u64("in_ptr");
            let out_ptr = b.load_param_u64("out_ptr");
            let vals = b.load_global_f32x4(&in_ptr);

            // Address of the `tile` shared array -- the same `mov.u64 dst,
            // <symbol>;` idiom `oxicuda-blas`'s hand-written kernels use
            // (see `oxicuda_blas::level1::dot::shared_mem_base_addr`).
            let tile_addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("mov.u64 {tile_addr}, tile;"));

            b.store_shared_f32x4(&tile_addr, &vals);
            let roundtripped = b.load_shared_f32x4(&tile_addr);
            b.store_global_f32x4(&out_ptr, &roundtripped);
            b.ret();
        })
        .build()
        .expect("PTX generation must succeed");

    let kernel = load_kernel(&ptx, "test_shared_f32x4_roundtrip");

    let input = [1.0f32, -2.5, 3.0e10, 1.0 / 3.0];
    let in_buf = DeviceBuffer::<f32>::from_host(&input).expect("in_buf alloc");
    let out_buf = DeviceBuffer::<f32>::zeroed(4).expect("out_buf alloc");

    let params = single_thread_params();
    let args = (in_buf.as_device_ptr(), out_buf.as_device_ptr());
    kernel
        .launch(&params, &fx.stream, &args)
        .expect("kernel launch");
    fx.stream.synchronize().expect("synchronize");

    let mut output = [0.0f32; 4];
    out_buf.copy_to_host(&mut output).expect("copy_to_host");

    // See `global_f32x4_roundtrip_matches_host` for why this compares bit
    // patterns rather than `f32 == f32`.
    let output_bits = output.map(f32::to_bits);
    let input_bits = input.map(f32::to_bits);
    assert_eq!(
        output_bits, input_bits,
        "round-tripping through shared memory via store_shared_f32x4 + \
         load_shared_f32x4 must reproduce the input bit-for-bit; got \
         {output:?}, expected {input:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  cvt_f32_to_tf32
// ═══════════════════════════════════════════════════════════════════════

/// CPU reference for `cvt.rna.tf32.f32`: round the low 13 discarded
/// mantissa bits to the nearest 10-bit-mantissa TF32 value, ties away from
/// zero. This is the textbook "add half an ULP, then truncate" integer
/// trick applied to the IEEE-754 bit pattern (monotonic with magnitude for
/// either sign, since none of this file's test values carry the addition's
/// carry as far as the sign bit).
const fn tf32_round_reference(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x0000_1000) & 0xFFFF_E000;
    f32::from_bits(rounded)
}

/// `cvt_f32_to_tf32` must match [`tf32_round_reference`] bit-for-bit --
/// including at an *exact* halfway point, which is the one input that
/// distinguishes RNA (ties away from zero, what this method emits) from
/// both plain truncation (a `mov.b32` reinterpret, which this method's doc
/// comment explicitly warns against) and from RN / round-to-nearest-even
/// (what `cvt.rn.tf32.f32` would compute, were it legal on this
/// architecture): at the exact tie `f32::from_bits(0x3F80_1000)`,
/// truncation stays at `0x3F80_0000` and round-to-even also lands on
/// `0x3F80_0000` (its kept mantissa LSB is 0, i.e. even), while RNA must
/// round up to `0x3F80_2000`. A regression to either alternative changes
/// this one case's result even though every non-tie case would still agree
/// with this test.
#[test]
fn cvt_f32_to_tf32_matches_bit_exact_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let ptx = KernelBuilder::new("test_cvt_f32_to_tf32")
        .target(fx.sm)
        .param("in_ptr", PtxType::U64)
        .param("out_ptr", PtxType::U64)
        .body(|b| {
            let in_ptr = b.load_param_u64("in_ptr");
            let out_ptr = b.load_param_u64("out_ptr");
            let x = b.load_global_f32(in_ptr);
            let y = b.cvt_f32_to_tf32(x);
            b.store_global_f32(out_ptr, y);
            b.ret();
        })
        .build()
        .expect("PTX generation must succeed");

    let kernel = load_kernel(&ptx, "test_cvt_f32_to_tf32");

    let test_values: [f32; 13] = [
        1.0,
        -1.0,
        0.0,
        -0.0,
        123.456,
        -123.456,
        1.0 / 3.0,
        1.0e10,
        -1.0e-10,
        f32::from_bits(0x3F80_1000), // exact tie -> RNA rounds AWAY from zero (up)
        f32::from_bits(0xBF80_1000), // exact tie, negative -> larger magnitude
        f32::from_bits(0x3F80_0FFF), // just below the tie -> rounds down (stays)
        f32::from_bits(0x3F80_1001), // just above the tie -> rounds up
    ];

    for &x in &test_values {
        let in_buf = DeviceBuffer::<f32>::from_host(&[x]).expect("in_buf alloc");
        let out_buf = DeviceBuffer::<f32>::zeroed(1).expect("out_buf alloc");

        let params = single_thread_params();
        let args = (in_buf.as_device_ptr(), out_buf.as_device_ptr());
        kernel
            .launch(&params, &fx.stream, &args)
            .expect("kernel launch");
        fx.stream.synchronize().expect("synchronize");

        let mut out = [0.0f32; 1];
        out_buf.copy_to_host(&mut out).expect("copy_to_host");

        let expected = tf32_round_reference(x);
        assert_eq!(
            out[0].to_bits(),
            expected.to_bits(),
            "cvt.rna.tf32.f32({x:e}) = {:e} (bits {:#010x}) on-device, but \
             the RNA reference gives {:e} (bits {:#010x}) -- a different \
             rounding mode (e.g. round-to-even) or a truncating mov.b32 \
             would diverge exactly here",
            out[0],
            out[0].to_bits(),
            expected,
            expected.to_bits(),
        );
    }
}
