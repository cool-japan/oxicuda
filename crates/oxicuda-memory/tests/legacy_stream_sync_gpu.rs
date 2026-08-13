//! On-device proof that `DeviceBuffer::zeroed` / `copy_from_host` synchronise
//! the **legacy default stream specifically**, not the whole CUDA context.
//!
//! # What is being tested
//!
//! Both constructors issue a non-async driver primitive (`cuMemsetD8_v2`,
//! `cuMemcpyHtoD_v2`) that is documented as *asynchronous with respect to the
//! host*: the call returns before the device-side work has landed, with the
//! work queued on the legacy default (`NULL`) stream. Every OxiCUDA
//! [`Stream`](oxicuda_driver::Stream) is created `CU_STREAM_NON_BLOCKING`,
//! which by definition does **not** implicitly synchronise with the legacy
//! stream — so without an explicit wait, a kernel on such a stream can read the
//! buffer before the zero-fill / upload lands. That is a real data race.
//!
//! It used to be closed with `cuCtxSynchronize()`, which blocks "until the
//! device has completed all preceding requested tasks" in the *context* — every
//! stream, including ones with nothing to do with this buffer. It is now closed
//! with `cuStreamSynchronize(NULL)`, which waits only for "all operations in the
//! stream specified by hStream". This file proves the swap kept the correctness
//! guarantee and bought the concurrency:
//!
//! * [`zeroed_still_waits_for_the_legacy_stream`] — work queued on the legacy
//!   stream **is** still awaited. Fails outright if the synchronisation were
//!   removed.
//! * [`copy_from_host_result_is_visible_to_a_non_blocking_stream`] — the actual
//!   race: a consumer kernel on a non-blocking stream never observes a
//!   half-landed upload. Fails if the synchronisation were removed.
//! * [`zeroed_no_longer_waits_for_unrelated_streams`] — the payoff: an
//!   independent non-blocking stream is **not** awaited. This test *fails*
//!   against the old `cuCtxSynchronize` implementation, and is the live
//!   demonstration that the new synchronisation is genuinely narrower.
//!
//! Every test skips cleanly when no CUDA device is present.

#![cfg(feature = "gpu-tests")]

use std::ffi::c_void;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use oxicuda_driver::ffi::{CUdeviceptr, CUstream};
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::{Context, Device, Function, Module, Stream};
use oxicuda_memory::DeviceBuffer;

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

/// Occupies a stream for a wall-clock duration without pinning the whole GPU.
///
/// A single thread polls `%globaltimer` (a nanosecond-resolution device clock,
/// sm_30+) until the deadline. One thread is deliberate: it keeps the stream
/// busy for a known, tunable time while leaving essentially every SM free, so a
/// concurrent copy on another stream is limited by synchronisation semantics
/// rather than by contention for compute resources — exactly the property these
/// timing tests need to be meaningful.
const SPIN_PTX: &str = r"
.version 6.0
.target sm_52
.address_size 64

.visible .entry spin_ns(
    .param .u64 spin_dur,
    .param .u64 spin_out
)
{
    .reg .b64 %rd<8>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [spin_dur];
    ld.param.u64 %rd2, [spin_out];
    mov.u64 %rd3, %globaltimer;
    add.s64 %rd4, %rd3, %rd1;
$SPIN:
    mov.u64 %rd5, %globaltimer;
    setp.lt.u64 %p0, %rd5, %rd4;
    @%p0 bra $SPIN;
    st.global.u64 [%rd2], %rd5;
    ret;
}
";

/// Counts elements of an `f32` buffer that differ from an expected constant,
/// accumulating into a `u32` counter with `atom.global.add`.
///
/// Used as the *consumer* half of the race test: if a host→device upload has
/// not fully landed when this runs, the stale prior contents are observed and
/// the counter comes back non-zero.
const CHECK_PTX: &str = r"
.version 6.0
.target sm_52
.address_size 64

.visible .entry count_mismatch(
    .param .u64 cm_data,
    .param .u32 cm_n,
    .param .u64 cm_out,
    .param .f32 cm_expect
)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;
    .reg .pred %p<4>;
    ld.param.u64 %rd1, [cm_data];
    ld.param.u32 %r1, [cm_n];
    ld.param.u64 %rd2, [cm_out];
    ld.param.f32 %f1, [cm_expect];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    mov.u32 %r6, %nctaid.x;
    mul.lo.u32 %r7, %r6, %r3;
$LOOP:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra $DONE;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];
    setp.eq.f32 %p2, %f2, %f1;
    @%p2 bra $NEXT;
    mov.u32 %r8, 1;
    atom.global.add.u32 %r9, [%rd2], %r8;
$NEXT:
    add.s32 %r5, %r5, %r7;
    bra $LOOP;
$DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// One CUDA context shared by every test in this binary.
///
/// Timing tests must not compete with each other: two live contexts on a
/// non-MPS GPU are time-sliced by the driver, which would add scheduling noise
/// to the very measurements under test. A single context plus [`serial`] keeps
/// each measurement alone on the device.
static CONTEXT: OnceLock<Option<Arc<Context>>> = OnceLock::new();

/// Serialises the timing-sensitive sections across the harness's test threads.
static GPU_SERIAL: Mutex<()> = Mutex::new(());

/// Acquires the device-exclusive lock (poison-tolerant: a panicking test must
/// not cascade into spurious failures in the others).
fn serial() -> MutexGuard<'static, ()> {
    GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The shared context, or `None` when no CUDA driver/device is available.
fn context() -> Option<&'static Arc<Context>> {
    CONTEXT
        .get_or_init(|| {
            oxicuda_driver::init().ok()?;
            if Device::count().ok()? == 0 {
                return None;
            }
            let dev = Device::get(0).ok()?;
            Some(Arc::new(Context::new(&dev).ok()?))
        })
        .as_ref()
}

/// Skips the test when there is no GPU to run it on; otherwise yields the
/// shared context **made current on the calling thread**.
///
/// A CUDA context is current per *thread*, and the test harness runs each test
/// on its own thread — so the thread that happened to initialise [`CONTEXT`] is
/// the only one that would otherwise have a current context, and every other
/// test would fail with `InvalidContext` on its first driver call.
macro_rules! gpu_or_skip {
    () => {
        match context() {
            Some(ctx) => {
                ctx.set_current().expect("make shared context current");
                ctx
            }
            None => {
                eprintln!("skipping: no CUDA driver/device");
                return;
            }
        }
    };
}

/// JIT-compiles `ptx` and resolves `entry`.
fn load(ptx: &str, entry: &str) -> (Module, Function) {
    let module = Module::from_ptx(ptx).expect("ptxas rejected the test kernel");
    let func = module.get_function(entry).expect("entry point not found");
    (module, func)
}

/// Launches `spin_ns` on `stream` for `duration`, writing a timestamp to `out`.
fn launch_spin(func: &Function, stream: CUstream, duration: Duration, out: CUdeviceptr) {
    let api = try_driver().expect("driver present");
    let mut ns_arg: u64 = duration.as_nanos() as u64;
    let mut out_arg: CUdeviceptr = out;
    let mut params: [*mut c_void; 2] = [
        std::ptr::from_mut(&mut ns_arg).cast(),
        std::ptr::from_mut(&mut out_arg).cast(),
    ];
    // SAFETY: `func` is a live entry point taking exactly these two parameters,
    // `stream` is a valid handle in the current context, and `params` outlives
    // the (asynchronous) launch call itself, which is all `cuLaunchKernel`
    // requires of the argument array.
    let rc = unsafe {
        (api.cu_launch_kernel)(
            func.raw(),
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            stream,
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    oxicuda_driver::check(rc).expect("spin kernel launch");
}

/// Launches `count_mismatch` over `n` f32 elements of `data` on `stream`.
fn launch_check(
    func: &Function,
    stream: CUstream,
    data: CUdeviceptr,
    n: u32,
    out: CUdeviceptr,
    expect: f32,
) {
    let api = try_driver().expect("driver present");
    let mut data_arg = data;
    let mut n_arg = n;
    let mut out_arg = out;
    let mut expect_arg = expect;
    let mut params: [*mut c_void; 4] = [
        std::ptr::from_mut(&mut data_arg).cast(),
        std::ptr::from_mut(&mut n_arg).cast(),
        std::ptr::from_mut(&mut out_arg).cast(),
        std::ptr::from_mut(&mut expect_arg).cast(),
    ];
    // SAFETY: as in `launch_spin` -- matching signature, valid stream, argument
    // array live across the launch call.
    let rc = unsafe {
        (api.cu_launch_kernel)(
            func.raw(),
            256,
            1,
            1,
            256,
            1,
            1,
            0,
            stream,
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    oxicuda_driver::check(rc).expect("check kernel launch");
}

/// Blocks until `stream` has drained.
fn sync_raw(stream: CUstream) {
    let api = try_driver().expect("driver present");
    // SAFETY: `stream` is a valid handle (or NULL for the legacy stream).
    oxicuda_driver::check(unsafe { (api.cu_stream_synchronize)(stream) }).expect("stream sync");
}

/// How long the spin kernels run. Long enough that "waited for it" and "did not
/// wait for it" are separated by orders of magnitude, short enough to keep the
/// suite fast.
const SPIN: Duration = Duration::from_millis(400);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// **The guarantee must survive.** With work already queued on the legacy
/// default stream, `DeviceBuffer::zeroed` must not return until that stream has
/// drained — because its own `cuMemsetD8_v2` is queued behind that work on that
/// same stream, and returning early is precisely the race the synchronisation
/// exists to prevent.
///
/// This is the test that fails if the synchronisation is removed entirely:
/// without it `zeroed` returns in microseconds instead of waiting out the spin.
#[test]
fn zeroed_still_waits_for_the_legacy_stream() {
    let _ctx = gpu_or_skip!();
    let _guard = serial();
    let (_module, spin) = load(SPIN_PTX, "spin_ns");
    let stamp = DeviceBuffer::<u64>::zeroed(1).expect("stamp buffer");

    // Queue the spin on the LEGACY stream (a NULL handle) -- the same stream
    // `zeroed`'s memset will be queued on, and therefore ordered ahead of it.
    launch_spin(&spin, CUstream::default(), SPIN, stamp.as_device_ptr());

    let started = Instant::now();
    let buf = DeviceBuffer::<f32>::zeroed(1024).expect("zeroed");
    let elapsed = started.elapsed();
    eprintln!("zeroed() with {SPIN:?} queued on the legacy stream: {elapsed:?}");

    assert!(
        elapsed >= SPIN.mul_f64(0.8),
        "zeroed() returned after {elapsed:?} while {SPIN:?} of work was still \
         queued on the legacy default stream -- the memset it issues is queued \
         behind that work on that same stream, so returning early means the \
         zero-fill had NOT landed and any consumer stream could race it"
    );

    // And the postcondition itself still holds.
    let mut host = vec![1.0f32; 1024];
    buf.copy_to_host(&mut host).expect("readback");
    assert!(host.iter().all(|&v| v == 0.0), "buffer was not zero-filled");
}

/// **The payoff.** An independent `CU_STREAM_NON_BLOCKING` stream is *not*
/// awaited: `zeroed` returns as soon as the legacy stream is clear, while
/// unrelated work continues on the other stream.
///
/// This is the live demonstration that the new synchronisation is narrower than
/// the old one. Against the previous `cuCtxSynchronize()` implementation --
/// documented as blocking until the device has completed *all* preceding tasks
/// in the context -- this test fails, because `zeroed` would sit out the whole
/// 400 ms spin belonging to a stream it has nothing to do with. That is the
/// collateral serialisation a multi-stream pipeline (independent models on
/// independent streams) paid on every buffer it allocated.
#[test]
fn zeroed_no_longer_waits_for_unrelated_streams() {
    let ctx = gpu_or_skip!();
    let _guard = serial();
    let other = Stream::new(ctx).expect("non-blocking stream");
    let (_module, spin) = load(SPIN_PTX, "spin_ns");
    let stamp = DeviceBuffer::<u64>::zeroed(1).expect("stamp buffer");

    // Occupy an unrelated non-blocking stream for `SPIN`.
    launch_spin(&spin, other.raw(), SPIN, stamp.as_device_ptr());

    let started = Instant::now();
    let buf = DeviceBuffer::<f32>::zeroed(1024).expect("zeroed");
    let elapsed = started.elapsed();
    eprintln!("zeroed() with {SPIN:?} queued on an UNRELATED stream: {elapsed:?}");

    assert!(
        elapsed < SPIN.mul_f64(0.5),
        "zeroed() blocked for {elapsed:?} while {SPIN:?} of unrelated work ran \
         on an independent non-blocking stream -- that is the whole-context \
         barrier `cuCtxSynchronize` imposed; the legacy-stream-scoped \
         `cuStreamSynchronize(NULL)` must not wait for it"
    );

    // Correctness is unaffected by the narrowing: the buffer really is zeroed
    // even though the other stream is still busy.
    let mut host = vec![1.0f32; 1024];
    buf.copy_to_host(&mut host).expect("readback");
    assert!(host.iter().all(|&v| v == 0.0), "buffer was not zero-filled");

    other.synchronize().expect("drain spin");
}

/// **The original race, closed.** A consumer kernel on a `CU_STREAM_NON_BLOCKING`
/// stream, launched immediately after `copy_from_host` with no further
/// synchronisation, must observe the *complete* upload.
///
/// A non-blocking stream does not implicitly wait on the legacy stream, so if
/// `copy_from_host` returned while its DMA was still in flight, this kernel
/// would read the buffer's previous contents (zeros) and count mismatches. The
/// buffer is deliberately large (16 MiB) so the DMA takes milliseconds while the
/// kernel launch takes microseconds — the race window is enormous, and a
/// regression here would be loud rather than intermittent.
#[test]
fn copy_from_host_result_is_visible_to_a_non_blocking_stream() {
    let ctx = gpu_or_skip!();
    let _guard = serial();
    let consumer = Stream::new(ctx).expect("non-blocking stream");
    let (_module, check) = load(CHECK_PTX, "count_mismatch");

    const N: usize = 4 * 1024 * 1024; // 16 MiB of f32
    const SENTINEL: f32 = 3.25;

    // Start from zeros, fully landed, so a too-early read is unmistakable.
    let mut buf = DeviceBuffer::<f32>::zeroed(N).expect("zeroed");
    let mismatches = DeviceBuffer::<u32>::zeroed(1).expect("counter");

    let host = vec![SENTINEL; N];
    buf.copy_from_host(&host).expect("copy_from_host");

    // No synchronisation of our own here: `copy_from_host`'s postcondition is
    // the only thing standing between this launch and stale data.
    launch_check(
        &check,
        consumer.raw(),
        buf.as_device_ptr(),
        N as u32,
        mismatches.as_device_ptr(),
        SENTINEL,
    );
    consumer.synchronize().expect("drain consumer");

    let mut count = [0u32; 1];
    mismatches.copy_to_host(&mut count).expect("read counter");
    assert_eq!(
        count[0], 0,
        "a kernel on a non-blocking stream saw {} of {N} elements still holding \
         pre-upload data -- copy_from_host returned before its DMA landed",
        count[0]
    );
}

/// The same race, for `zeroed`: a consumer on a non-blocking stream must see a
/// fully zero-filled buffer with no synchronisation of its own.
#[test]
fn zeroed_result_is_visible_to_a_non_blocking_stream() {
    let ctx = gpu_or_skip!();
    let _guard = serial();
    let consumer = Stream::new(ctx).expect("non-blocking stream");
    let (_module, check) = load(CHECK_PTX, "count_mismatch");

    const N: usize = 4 * 1024 * 1024; // 16 MiB of f32

    // Pre-dirty a region, then allocate over it: `cuMemAlloc_v2` frequently
    // hands back a just-freed address, so the zero-fill has something non-zero
    // to overwrite and an early read is detectable rather than accidentally
    // correct.
    {
        let mut dirty = DeviceBuffer::<f32>::alloc(N).expect("dirty alloc");
        dirty.copy_from_host(&vec![7.5f32; N]).expect("dirty fill");
    }

    let buf = DeviceBuffer::<f32>::zeroed(N).expect("zeroed");
    let mismatches = DeviceBuffer::<u32>::zeroed(1).expect("counter");

    launch_check(
        &check,
        consumer.raw(),
        buf.as_device_ptr(),
        N as u32,
        mismatches.as_device_ptr(),
        0.0,
    );
    consumer.synchronize().expect("drain consumer");

    let mut count = [0u32; 1];
    mismatches.copy_to_host(&mut count).expect("read counter");
    assert_eq!(
        count[0], 0,
        "a kernel on a non-blocking stream saw {} of {N} elements not yet zeroed \
         -- zeroed() returned before its memset landed",
        count[0]
    );
}

/// Sanity check on the instrument itself: the spin kernel really does occupy a
/// stream for the requested wall-clock time.
///
/// Without this, a spin kernel that silently returned immediately (wrong PTX,
/// unsupported `%globaltimer`) would make
/// [`zeroed_no_longer_waits_for_unrelated_streams`] pass vacuously.
#[test]
fn spin_kernel_actually_spins() {
    let ctx = gpu_or_skip!();
    let _guard = serial();
    let stream = Stream::new(ctx).expect("stream");
    let (_module, spin) = load(SPIN_PTX, "spin_ns");
    let stamp = DeviceBuffer::<u64>::zeroed(1).expect("stamp buffer");

    let started = Instant::now();
    launch_spin(&spin, stream.raw(), SPIN, stamp.as_device_ptr());
    stream.synchronize().expect("drain spin");
    let elapsed = started.elapsed();

    assert!(
        elapsed >= SPIN.mul_f64(0.8),
        "spin kernel returned after {elapsed:?}, expected ~{SPIN:?} -- the \
         timing tests that rely on it would be vacuous"
    );

    let mut stamped = [0u64; 1];
    stamp.copy_to_host(&mut stamped).expect("read stamp");
    assert_ne!(stamped[0], 0, "spin kernel never reached its final store");
}

/// The legacy stream really is what a NULL handle selects: work queued on the
/// legacy stream is drained by `cuStreamSynchronize(NULL)`.
///
/// This anchors the driver-semantics claim the implementation rests on, rather
/// than taking it on faith from the documentation alone.
#[test]
fn null_handle_synchronizes_the_legacy_stream() {
    let _ctx = gpu_or_skip!();
    let _guard = serial();
    let (_module, spin) = load(SPIN_PTX, "spin_ns");
    let stamp = DeviceBuffer::<u64>::zeroed(1).expect("stamp buffer");

    let started = Instant::now();
    launch_spin(&spin, CUstream::default(), SPIN, stamp.as_device_ptr());
    sync_raw(CUstream::default());
    let elapsed = started.elapsed();

    assert!(
        elapsed >= SPIN.mul_f64(0.8),
        "cuStreamSynchronize(NULL) returned after {elapsed:?} without waiting \
         for work launched on the NULL stream -- the legacy-stream \
         interpretation this optimisation relies on would not hold"
    );

    let mut stamped = [0u64; 1];
    stamp.copy_to_host(&mut stamped).expect("read stamp");
    assert_ne!(stamped[0], 0, "spin kernel did not complete");
}
