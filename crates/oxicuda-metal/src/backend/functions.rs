//! Helper utilities used by the Metal backend, plus the integration test
//! suite (gated on `#[cfg(test)]`).
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(target_os = "macos")]
use oxicuda_backend::{BackendError, BackendResult};

/// Round up to the next power of 2 (minimum 1), saturating instead of panicking.
///
/// The previous hand-rolled `1 << (usize::BITS - (n - 1).leading_zeros())` shifted
/// by 64 for any `n > 2^63` — a shift-overflow panic in debug builds and a
/// silently wrong value in release. `checked_next_power_of_two` handles the edge
/// correctly; the `unwrap_or` saturates to the largest representable power of two
/// so this stays panic-free for every input.
#[cfg(target_os = "macos")]
pub(super) fn next_power_of_2(n: usize) -> usize {
    n.checked_next_power_of_two()
        .unwrap_or(1usize << (usize::BITS - 1))
}

/// Largest power of two that is `>= 1` and no greater than either `desired` or
/// `max_threads`.
///
/// The MSL tree reductions fold with `for (s = tg_size / 2; s > 0; s >>= 1)`,
/// which only visits every lane when `tg_size` is a power of two — a bare
/// `min(max_threads)` clamp can produce e.g. 10 and silently drop lanes. Use
/// this wherever a threadgroup width feeds a tree reduction.
#[cfg(target_os = "macos")]
pub(super) fn pow2_threadgroup(desired: u64, max_threads: u64) -> u64 {
    let cap = desired.min(max_threads).max(1);
    // Largest power of two <= cap.
    1u64 << (u64::BITS - 1 - cap.leading_zeros())
}

/// Clamp a 1-D threadgroup width to what the compiled pipeline actually allows,
/// then align it down to a whole number of SIMD groups.
///
/// A threadgroup wider than the PSO's `maxTotalThreadsPerThreadgroup` is
/// rejected by Metal at dispatch; a width that is not a multiple of
/// `threadExecutionWidth` leaves lanes idle in the trailing SIMD group.
#[cfg(target_os = "macos")]
pub(super) fn clamp_1d_threadgroup(
    desired: u64,
    max_threads: u64,
    execution_width: u64,
    total_threads: u64,
) -> u64 {
    let mut tg = desired.min(max_threads.max(1)).min(total_threads).max(1);
    let width = execution_width.max(1);
    if tg > width {
        tg -= tg % width;
    }
    tg.max(1)
}

/// Narrow a `usize` to the `u32` the MSL kernel parameter structs use, failing
/// loudly instead of wrapping.
///
/// Every kernel parameter (element counts, `m`/`n`/`k`, batch strides) is a
/// 32-bit unsigned integer in MSL. An unchecked `as u32` on a 64-bit host turns
/// an 8 GiB batch stride into a small number and makes the kernel read and write
/// the wrong addresses while still returning `Ok`.
#[cfg(target_os = "macos")]
pub(super) fn to_u32(value: usize, what: &str) -> BackendResult<u32> {
    u32::try_from(value).map_err(|_| {
        BackendError::InvalidArgument(format!(
            "{what} = {value} exceeds the u32 range accepted by the Metal kernels"
        ))
    })
}

/// Resolve device handles to independent `metal::Buffer` retains.
///
/// The buffer-map mutex is held **only** for the lookup: each retain keeps its
/// buffer alive independently of the map, so encoding and the blocking GPU wait
/// happen with the lock released and unrelated `alloc`/`free`/`copy_*` calls on
/// other threads are not serialised behind the running kernel.
#[cfg(target_os = "macos")]
pub(super) fn resolve_buffers<const N: usize>(
    memory: &crate::memory::MetalMemoryManager,
    handles: [u64; N],
) -> BackendResult<[metal::Buffer; N]> {
    let bound: Vec<metal::Buffer> = {
        let buffers = memory.lock_buffers().map_err(BackendError::from)?;
        let mut bound = Vec::with_capacity(N);
        for handle in handles {
            let info = buffers
                .get(&handle)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {handle}")))?;
            bound.push(info.buffer.to_owned());
        }
        bound
    };
    // `bound` always has exactly N entries (one per handle, or an early return),
    // so this conversion cannot fail; mapping the error keeps the path panic-free.
    bound.try_into().map_err(|_: Vec<metal::Buffer>| {
        BackendError::DeviceError("internal error: buffer resolution arity mismatch".into())
    })
}

/// MSL entry-point name for the chunked (two-pass) reduction kernel of `op`.
///
/// `mean` deliberately maps to the `sum` kernel: the mean is a sum whose final
/// pass divides by the original element count, so both share one PSO.
#[cfg(target_os = "macos")]
pub(super) fn chunked_reduce_function_name(op: &str) -> &'static str {
    match op {
        "max" => "chunk_reduce_max_f32",
        "min" => "chunk_reduce_min_f32",
        _ => "chunk_reduce_sum_f32",
    }
}

/// MSL source for a chunked 1-D reduction kernel.
///
/// Unlike [`crate::msl::reduction_msl`] — which assigns exactly one threadgroup
/// per output element and therefore reduces a whole flat tensor with a single
/// threadgroup — this kernel spreads the work over `num_groups` threadgroups
/// with a grid-stride loop and writes one partial per group. Running it twice
/// (input → partials, partials → output) reduces an arbitrarily large 1-D
/// tensor at full device occupancy.
///
/// Bindings: `input(0)`, `output(1)`, `count(2)`, `num_groups(3)`,
/// `divisor(4)`, `threadgroup(0)` scratch. `divisor` is `1.0` for every pass
/// except the final pass of a `Mean`, which divides by the original element
/// count — matching `reduction_msl`'s `sdata[0] / float(reduce_size)` exactly so
/// the one-pass and two-pass paths agree bit-for-bit on small inputs.
///
/// `num_groups` is passed explicitly rather than read from
/// `[[threadgroups_per_grid]]` so the host stays the single source of truth for
/// the grid shape.
#[cfg(target_os = "macos")]
pub(super) fn chunked_reduce_msl(op: &str) -> String {
    let (identity, reduce_body) = match op {
        "max" => ("-INFINITY", "return (a > b) ? a : b;"),
        "min" => ("INFINITY", "return (a < b) ? a : b;"),
        // "sum" and "mean" share the additive kernel.
        _ => ("0.0f", "return a + b;"),
    };
    let kernel_name = chunked_reduce_function_name(op);
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

inline float chunk_reduce_fn(float a, float b) {{
    {reduce_body}
}}

kernel void {kernel_name}(
    device const float* input   [[buffer(0)]],
    device float*       output  [[buffer(1)]],
    constant uint&  count       [[buffer(2)]],
    constant uint&  num_groups  [[buffer(3)]],
    constant float& divisor     [[buffer(4)]],
    threadgroup float* sdata    [[threadgroup(0)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint lid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {{
    float acc = {identity};
    uint stride = tg_size * num_groups;
    for (uint i = tg_id * tg_size + lid; i < count; i += stride) {{
        acc = chunk_reduce_fn(acc, input[i]);
    }}
    sdata[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Width-agnostic tree fold: collapse the upper ceil(n/2) lanes into the
    // lower half each round. Unlike `for (s = tg_size/2; s > 0; s >>= 1)` this
    // is correct for a threadgroup width that is not a power of two, and the
    // loop condition stays uniform so the barrier is uniformly executed.
    for (uint n_active = tg_size; n_active > 1u; ) {{
        uint half_n = (n_active + 1u) / 2u;
        if (lid + half_n < n_active) {{
            sdata[lid] = chunk_reduce_fn(sdata[lid], sdata[lid + half_n]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        n_active = half_n;
    }}

    if (lid == 0) {{
        output[tg_id] = sdata[0] / divisor;
    }}
}}
"#
    )
}

/// Interpret a byte slice as little-endian `f32` values.
///
/// Only reachable through [`super::nn::attention_host`], which is itself only
/// live on macOS (off macOS no dispatch ever runs), so the helper is
/// deliberately dead there.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn read_f32_le(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Encode `f32` values as little-endian bytes.
///
/// See [`read_f32_le`] for why this is dead off macOS.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn write_f32_le(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use oxicuda_backend::{
        BackendError, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
    };

    use super::super::types::MetalBackend;

    // ─── Pure helper unit tests ──────────────────────────────────────────────

    #[test]
    #[cfg(target_os = "macos")]
    fn next_power_of_2_is_panic_free_at_the_top_of_the_range() {
        use super::next_power_of_2;
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(1024), 1024);
        assert_eq!(next_power_of_2(1025), 2048);
        // The old `1 << (usize::BITS - (n - 1).leading_zeros())` shifted by 64
        // here: a panic in debug, a masked wrong value in release.
        let top = next_power_of_2(usize::MAX);
        assert_eq!(top, 1usize << (usize::BITS - 1));
        assert_eq!(next_power_of_2((1usize << 63) + 1), 1usize << 63);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pow2_threadgroup_never_returns_a_non_power_of_two() {
        use super::pow2_threadgroup;
        // The MSL tree reduction only visits every lane for power-of-two widths.
        for desired in [1u64, 2, 3, 10, 100, 256, 257, 4096] {
            for max in [64u64, 256, 1024] {
                let tg = pow2_threadgroup(desired, max);
                assert!(tg.is_power_of_two(), "{tg} from ({desired}, {max})");
                assert!(tg <= max && tg <= desired.max(1));
            }
        }
        assert_eq!(pow2_threadgroup(0, 1024), 1);
        assert_eq!(pow2_threadgroup(300, 1024), 256);
        assert_eq!(pow2_threadgroup(1024, 256), 256);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn clamp_1d_threadgroup_respects_pipeline_limit_and_simd_width() {
        use super::clamp_1d_threadgroup;
        // Clamped by the pipeline's own maximum.
        assert_eq!(clamp_1d_threadgroup(1024, 512, 32, 100_000), 512);
        // Aligned down to whole SIMD groups.
        assert_eq!(clamp_1d_threadgroup(100, 1024, 32, 100_000), 96);
        // Never wider than the work itself.
        assert_eq!(clamp_1d_threadgroup(256, 1024, 32, 10), 10);
        // Degenerate inputs stay >= 1 instead of producing a zero-wide group.
        assert_eq!(clamp_1d_threadgroup(0, 0, 0, 1), 1);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn to_u32_rejects_values_the_kernels_cannot_represent() {
        use super::to_u32;
        assert_eq!(to_u32(0, "x"), Ok(0));
        assert_eq!(to_u32(u32::MAX as usize, "x"), Ok(u32::MAX));
        let err = to_u32(u32::MAX as usize + 1, "batched gemm stride_a")
            .expect_err("a value past u32::MAX must not wrap");
        match err {
            BackendError::InvalidArgument(msg) => {
                assert!(msg.contains("batched gemm stride_a"), "{msg}");
                assert!(msg.contains("4294967296"), "{msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn chunked_reduce_msl_emits_one_kernel_per_op() {
        use super::{chunked_reduce_function_name, chunked_reduce_msl};
        assert_eq!(chunked_reduce_function_name("sum"), "chunk_reduce_sum_f32");
        // Mean shares the additive kernel; only the final divisor differs.
        assert_eq!(chunked_reduce_function_name("mean"), "chunk_reduce_sum_f32");
        assert_eq!(chunked_reduce_function_name("max"), "chunk_reduce_max_f32");
        assert_eq!(chunked_reduce_function_name("min"), "chunk_reduce_min_f32");
        for op in ["sum", "mean", "max", "min"] {
            let src = chunked_reduce_msl(op);
            assert!(src.contains(chunked_reduce_function_name(op)), "{op}");
            assert!(src.contains("threadgroup_barrier"), "{op}");
            // Width-agnostic fold, not the power-of-two-only `s >>= 1` form.
            assert!(
                src.contains("for (uint n_active = tg_size; n_active > 1u; )"),
                "{op}"
            );
            // num_groups is passed explicitly rather than read from the grid.
            assert!(src.contains("constant uint&  num_groups"), "{op}");
            assert!(src.contains("output[tg_id] = sdata[0] / divisor;"), "{op}");
        }
        assert!(chunked_reduce_msl("max").contains("-INFINITY"));
        assert!(chunked_reduce_msl("min").contains("INFINITY"));
        assert!(chunked_reduce_msl("sum").contains("return a + b;"));
    }

    #[test]
    fn metal_backend_new_uninitialized() {
        let b = MetalBackend::new();
        assert!(!b.is_initialized());
    }
    #[test]
    fn metal_backend_name() {
        let b = MetalBackend::new();
        assert_eq!(b.name(), "metal");
    }
    #[test]
    fn metal_backend_default() {
        let b = MetalBackend::default();
        assert!(!b.is_initialized());
        assert_eq!(b.name(), "metal");
    }
    #[test]
    fn backend_debug_impl() {
        let b = MetalBackend::new();
        let s = format!("{b:?}");
        assert!(s.contains("MetalBackend"));
    }
    #[test]
    fn backend_object_safe() {
        let b: Box<dyn ComputeBackend> = Box::new(MetalBackend::new());
        assert_eq!(b.name(), "metal");
    }
    #[test]
    fn backend_not_initialized_gemm() {
        let b = MetalBackend::new();
        let result = b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            0,
            4,
            0.0,
            0,
            4,
        );
        assert_eq!(result, Err(BackendError::NotInitialized));
    }
    #[test]
    fn backend_not_initialized_alloc() {
        let b = MetalBackend::new();
        assert_eq!(b.alloc(1024), Err(BackendError::NotInitialized));
    }
    #[test]
    fn backend_not_initialized_synchronize() {
        let b = MetalBackend::new();
        assert_eq!(b.synchronize(), Err(BackendError::NotInitialized));
    }
    #[test]
    fn backend_not_initialized_free() {
        let b = MetalBackend::new();
        assert_eq!(b.free(1), Err(BackendError::NotInitialized));
    }
    #[test]
    fn backend_not_initialized_copy_htod() {
        let b = MetalBackend::new();
        assert_eq!(b.copy_htod(1, b"hello"), Err(BackendError::NotInitialized));
    }
    #[test]
    fn backend_not_initialized_copy_dtoh() {
        let b = MetalBackend::new();
        let mut buf = [0u8; 4];
        assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
    }
    #[test]
    fn batched_gemm_not_initialized() {
        let b = MetalBackend::new();
        let result = b.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            16,
            0,
            4,
            16,
            0.0,
            0,
            4,
            16,
            2,
        );
        assert_eq!(result, Err(BackendError::NotInitialized));
    }
    #[test]
    fn batched_gemm_zero_batch_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.batched_gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                4,
                4,
                4,
                1.0,
                0,
                4,
                16,
                0,
                4,
                16,
                0.0,
                0,
                4,
                16,
                0,
            ),
            Ok(())
        );
    }
    #[test]
    fn batched_gemm_zero_dims_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.batched_gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                0,
                0,
                1.0,
                0,
                1,
                0,
                0,
                1,
                0,
                0.0,
                0,
                1,
                0,
                3,
            ),
            Ok(())
        );
    }
    #[test]
    fn gemm_f16_not_initialized() {
        let b = MetalBackend::new();
        let result = b.gemm_f16(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            0,
            4,
            0.0,
            0,
            4,
        );
        assert_eq!(result, Err(BackendError::NotInitialized));
    }
    #[test]
    fn gemm_f16_zero_dims_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.gemm_f16(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                0,
                0,
                1.0,
                0,
                1,
                0,
                1,
                0.0,
                0,
                1,
            ),
            Ok(())
        );
    }
    fn try_init() -> Option<MetalBackend> {
        let mut b = MetalBackend::new();
        match b.init() {
            Ok(()) => Some(b),
            Err(_) => None,
        }
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_backend_init_on_macos() {
        let mut backend = MetalBackend::new();
        match backend.init() {
            Ok(()) => {
                assert!(backend.is_initialized());
                assert_eq!(backend.init(), Ok(()));
                assert!(backend.is_initialized());
                let result = backend.alloc(64);
                match result {
                    Ok(handle) => {
                        assert!(handle > 0);
                        backend.free(handle).expect("free should succeed");
                    }
                    Err(e) => {
                        let _ = e;
                    }
                }
            }
            Err(e) => {
                let _ = e;
            }
        }
    }
    #[test]
    fn alloc_zero_bytes_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.alloc(0),
            Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into()
            ))
        );
    }
    #[test]
    fn copy_htod_empty_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.copy_htod(0, &[]), Ok(()));
    }
    #[test]
    fn copy_dtoh_empty_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.copy_dtoh(&mut [], 0), Ok(()));
    }
    #[test]
    fn gemm_zero_dims_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                0,
                0,
                1.0,
                0,
                1,
                0,
                1,
                0.0,
                0,
                1
            ),
            Ok(())
        );
    }
    #[test]
    fn unary_zero_n_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.unary(UnaryOp::Relu, 0, 0, 0), Ok(()));
    }
    #[test]
    fn binary_zero_n_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.binary(BinaryOp::Add, 0, 0, 0, 0), Ok(()));
    }
    #[test]
    fn synchronize_after_init() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.synchronize(), Ok(()));
    }
    #[test]
    fn reduce_empty_shape_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.reduce(ReduceOp::Sum, 0, 0, &[], 0),
            Err(BackendError::InvalidArgument(
                "shape must not be empty".into()
            ))
        );
    }
    #[test]
    fn reduce_axis_out_of_bounds_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.reduce(ReduceOp::Sum, 0, 0, &[4, 4], 5),
            Err(BackendError::InvalidArgument(
                "axis 5 is out of bounds for shape of length 2".into()
            ))
        );
    }
    #[test]
    fn attention_zero_seq_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 0, 8, 64, 0.125, false),
            Err(BackendError::InvalidArgument(
                "seq_q, seq_kv, and head_dim must all be > 0".into()
            ))
        );
    }
    #[test]
    fn attention_invalid_scale_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, 0.0, false),
            Err(BackendError::InvalidArgument(
                "scale must be a positive finite number, got 0".into()
            ))
        );
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, -1.0, false),
            Err(BackendError::InvalidArgument(
                "scale must be a positive finite number, got -1".into()
            ))
        );
        assert!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, f64::INFINITY, false)
                .is_err()
        );
    }
    #[test]
    fn conv2d_wrong_input_rank() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.conv2d_forward(
                0,
                &[1, 3, 32],
                0,
                &[16, 3, 3, 3],
                0,
                &[1, 16, 30, 30],
                &[1, 1],
                &[0, 0]
            ),
            Err(BackendError::InvalidArgument(
                "input_shape must have 4 elements (NCHW)".into()
            ))
        );
    }
    #[test]
    fn conv2d_wrong_filter_rank() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.conv2d_forward(
                0,
                &[1, 3, 32, 32],
                0,
                &[16, 3, 3],
                0,
                &[1, 16, 30, 30],
                &[1, 1],
                &[0, 0]
            ),
            Err(BackendError::InvalidArgument(
                "filter_shape must have 4 elements (KCFHFW)".into()
            ))
        );
    }
    #[test]
    fn init_idempotent() {
        let Some(mut b) = try_init() else {
            return;
        };
        assert_eq!(b.init(), Ok(()));
        assert!(b.is_initialized());
    }
    #[test]
    fn metal_init_graceful_failure() {
        let mut b = MetalBackend::new();
        let _result = b.init();
    }
    #[test]
    fn alloc_copy_roundtrip() {
        let Some(b) = try_init() else {
            return;
        };
        let src: Vec<u8> = (0u8..64).collect();
        let handle = match b.alloc(src.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(handle, &src).expect("copy_htod");
        let mut dst = vec![0u8; src.len()];
        b.copy_dtoh(&mut dst, handle).expect("copy_dtoh");
        assert_eq!(src, dst);
        b.free(handle).expect("free");
    }
    /// Helper: encode f32 slice to bytes (little-endian).
    #[cfg(target_os = "macos")]
    fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
        let mut bytes = vec![0u8; std::mem::size_of_val(data)];
        for (i, &val) in data.iter().enumerate() {
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
        }
        bytes
    }
    /// Helper: decode bytes to f32 vec (little-endian).
    #[cfg(target_os = "macos")]
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn unary_relu_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![-1.0f32, 0.0, 1.0, 2.0];
        let n = input.len();
        let bytes_in = f32_to_bytes(&input);
        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.unary(UnaryOp::Relu, ih, oh, n).expect("unary relu");
        let mut out = vec![0u8; bytes_in.len()];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert_eq!(result, vec![0.0f32, 0.0, 1.0, 2.0]);
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn unary_neg_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![1.0f32, -2.0, 3.0, 0.0];
        let n = input.len();
        let bytes_in = f32_to_bytes(&input);
        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.unary(UnaryOp::Neg, ih, oh, n).expect("unary neg");
        let mut out = vec![0u8; bytes_in.len()];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert_eq!(result, vec![-1.0f32, 2.0, -3.0, -0.0]);
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn binary_add_compute() {
        let Some(b) = try_init() else { return };
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let bv = vec![10.0f32, 20.0, 30.0, 40.0];
        let n = a.len();
        let ba = f32_to_bytes(&a);
        let bb = f32_to_bytes(&bv);
        let ah = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let bh = match b.alloc(bb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ah, &ba).expect("htod a");
        b.copy_htod(bh, &bb).expect("htod b");
        b.binary(BinaryOp::Add, ah, bh, oh, n).expect("binary add");
        let mut out = vec![0u8; ba.len()];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert_eq!(result, vec![11.0f32, 22.0, 33.0, 44.0]);
        b.free(ah).expect("free");
        b.free(bh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn binary_mul_compute() {
        let Some(b) = try_init() else { return };
        let a = vec![2.0f32, 3.0, 4.0, 5.0];
        let bv = vec![10.0f32, 10.0, 10.0, 10.0];
        let n = a.len();
        let ba = f32_to_bytes(&a);
        let bb = f32_to_bytes(&bv);
        let ah = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let bh = match b.alloc(bb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ah, &ba).expect("htod a");
        b.copy_htod(bh, &bb).expect("htod b");
        b.binary(BinaryOp::Mul, ah, bh, oh, n).expect("binary mul");
        let mut out = vec![0u8; ba.len()];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert_eq!(result, vec![20.0f32, 30.0, 40.0, 50.0]);
        b.free(ah).expect("free");
        b.free(bh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_sum_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes_in = f32_to_bytes(&input);
        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(4) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 4]).expect("zero output");
        b.reduce(ReduceOp::Sum, ih, oh, &[4], 0)
            .expect("reduce sum");
        let mut out = vec![0u8; 4];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 10.0).abs() < 1e-5,
            "expected 10.0, got {}",
            result[0]
        );
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_max_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![3.0f32, 1.0, 4.0, 1.5];
        let bytes_in = f32_to_bytes(&input);
        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(4) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 4]).expect("zero output");
        b.reduce(ReduceOp::Max, ih, oh, &[4], 0)
            .expect("reduce max");
        let mut out = vec![0u8; 4];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 4.0).abs() < 1e-5,
            "expected 4.0, got {}",
            result[0]
        );
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_mean_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![2.0f32, 4.0, 6.0, 8.0];
        let bytes_in = f32_to_bytes(&input);
        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(4) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 4]).expect("zero output");
        b.reduce(ReduceOp::Mean, ih, oh, &[4], 0)
            .expect("reduce mean");
        let mut out = vec![0u8; 4];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 5.0).abs() < 1e-5,
            "expected 5.0, got {}",
            result[0]
        );
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_2d_axis1_compute() {
        let Some(b) = try_init() else { return };
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes_in = f32_to_bytes(&input);
        let ih = match b.alloc(bytes_in.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(8) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 8]).expect("zero output");
        b.reduce(ReduceOp::Sum, ih, oh, &[2, 3], 1)
            .expect("reduce sum axis=1");
        let mut out = vec![0u8; 8];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 6.0).abs() < 1e-5,
            "expected 6.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 15.0).abs() < 1e-5,
            "expected 15.0, got {}",
            result[1]
        );
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn gemm_simple_compute() {
        let Some(b) = try_init() else { return };
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let bm = vec![5.0f32, 6.0, 7.0, 8.0];
        let c_init = vec![0.0f32; 4];
        let ba = f32_to_bytes(&a);
        let bb = f32_to_bytes(&bm);
        let bc = f32_to_bytes(&c_init);
        let ah = match b.alloc(ba.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let bh = match b.alloc(bb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let ch = match b.alloc(bc.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ah, &ba).expect("htod a");
        b.copy_htod(bh, &bb).expect("htod b");
        b.copy_htod(ch, &bc).expect("htod c");
        b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            ah,
            2,
            bh,
            2,
            0.0,
            ch,
            2,
        )
        .expect("gemm");
        let mut out = vec![0u8; bc.len()];
        b.copy_dtoh(&mut out, ch).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!(
            (result[0] - 19.0).abs() < 1e-4,
            "C[0,0]={}, expected 19",
            result[0]
        );
        assert!(
            (result[1] - 22.0).abs() < 1e-4,
            "C[0,1]={}, expected 22",
            result[1]
        );
        assert!(
            (result[2] - 43.0).abs() < 1e-4,
            "C[1,0]={}, expected 43",
            result[2]
        );
        assert!(
            (result[3] - 50.0).abs() < 1e-4,
            "C[1,1]={}, expected 50",
            result[3]
        );
        b.free(ah).expect("free");
        b.free(bh).expect("free");
        b.free(ch).expect("free");
    }

    /// Zero-copy import: register pre-existing external `metal::Buffer`s and run
    /// `gemm` directly on them (no host round-trip), then verify the result
    /// matches a CPU triple-loop and that freeing/dropping the backend does NOT
    /// deallocate the caller-owned buffers.
    #[test]
    #[cfg(target_os = "macos")]
    fn import_external_gemm_zero_copy() {
        let Some(backend) = try_init() else { return };
        let Some(device) = metal::Device::system_default() else {
            return;
        };

        // Row-major operands: A is 3x2, B is 2x3, C is 3x3.
        let m = 3usize;
        let k = 2usize;
        let n = 3usize;
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let bmat: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 2x3
        let a_bytes = f32_to_bytes(&a);
        let b_bytes = f32_to_bytes(&bmat);
        let c_len_bytes = m * n * std::mem::size_of::<f32>();

        // Build EXTERNAL buffers the way a consumer's cache would: the test (not
        // oxicuda) owns these `metal::Buffer`s for the whole function.
        let a_buf = device.new_buffer_with_data(
            a_bytes.as_ptr() as *const std::ffi::c_void,
            a_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let b_buf = device.new_buffer_with_data(
            b_bytes.as_ptr() as *const std::ffi::c_void,
            b_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let c_buf = device.new_buffer(
            c_len_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Register the external buffers as zero-copy handles.
        let a_h = backend
            .register_external(&a_buf, a_bytes.len())
            .expect("register external A");
        let b_h = backend
            .register_external(&b_buf, b_bytes.len())
            .expect("register external B");
        // Exercise the by-value convenience wrapper for the resident output C.
        let c_h = backend
            .import_buffer(c_buf.clone(), c_len_bytes)
            .expect("import external C");

        // All three handles must be flagged as imported (external).
        assert_eq!(backend.is_imported(a_h), Ok(Some(true)));
        assert_eq!(backend.is_imported(b_h), Ok(Some(true)));
        assert_eq!(backend.is_imported(c_h), Ok(Some(true)));

        // C(3x3) = 1.0 * A(3x2) * B(2x3) + 0.0 * C, all row-major.
        backend
            .gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                m,
                n,
                k,
                1.0,
                a_h,
                k,
                b_h,
                n,
                0.0,
                c_h,
                n,
            )
            .expect("gemm on imported handles");

        // Read the result back via oxicuda and via the caller's own buffer
        // pointer — both must agree (proves the GPU wrote the caller's memory).
        let mut out = vec![0u8; c_len_bytes];
        backend.copy_dtoh(&mut out, c_h).expect("dtoh C");
        let got = bytes_to_f32(&out);

        let direct = unsafe { std::slice::from_raw_parts(c_buf.contents() as *const f32, m * n) };

        // CPU reference triple-loop.
        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * bmat[p * n + j];
                }
                want[i * n + j] = acc;
            }
        }

        for idx in 0..(m * n) {
            let tol = 1e-3 * want[idx].abs().max(1.0);
            assert!(
                (got[idx] - want[idx]).abs() <= tol,
                "dtoh[{idx}]={} vs cpu {}",
                got[idx],
                want[idx]
            );
            assert!(
                (direct[idx] - want[idx]).abs() <= tol,
                "caller-buffer[{idx}]={} vs cpu {} (zero-copy write mismatch)",
                direct[idx],
                want[idx]
            );
        }

        // Also exercise the mixed case: external A·B → oxicuda-OWNED output.
        let owned_c = backend.alloc(c_len_bytes).expect("alloc owned C");
        assert_eq!(backend.is_imported(owned_c), Ok(Some(false)));
        backend
            .gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                m,
                n,
                k,
                1.0,
                a_h,
                k,
                b_h,
                n,
                0.0,
                owned_c,
                n,
            )
            .expect("gemm external A,B -> owned C");
        let mut owned_out = vec![0u8; c_len_bytes];
        backend
            .copy_dtoh(&mut owned_out, owned_c)
            .expect("dtoh owned C");
        let owned_got = bytes_to_f32(&owned_out);
        for idx in 0..(m * n) {
            let tol = 1e-3 * want[idx].abs().max(1.0);
            assert!(
                (owned_got[idx] - want[idx]).abs() <= tol,
                "owned-out[{idx}]={} vs cpu {}",
                owned_got[idx],
                want[idx]
            );
        }

        // Free everything. Imported handles must NOT deallocate the caller's
        // buffers; the owned handle releases normally.
        backend.free(a_h).expect("free imported A");
        backend.free(b_h).expect("free imported B");
        backend.free(c_h).expect("free imported C");
        backend.free(owned_c).expect("free owned C");
        // Handles are gone now.
        assert_eq!(backend.is_imported(a_h), Ok(None));

        // The external buffers are STILL alive and readable — the test owns
        // them. Reading them after oxicuda freed its handles proves there was no
        // double-free / premature deallocation.
        let still_a =
            unsafe { std::slice::from_raw_parts(a_buf.contents() as *const f32, a.len()) };
        assert_eq!(still_a, a.as_slice(), "A survived oxicuda free");
        let still_c = unsafe { std::slice::from_raw_parts(c_buf.contents() as *const f32, m * n) };
        assert_eq!(still_c, want.as_slice(), "C result survived oxicuda free");

        // Drop the backend explicitly while the external buffers are still held;
        // backend drop must not touch them either.
        drop(backend);
        let after_drop =
            unsafe { std::slice::from_raw_parts(b_buf.contents() as *const f32, bmat.len()) };
        assert_eq!(after_drop, bmat.as_slice(), "B survived backend drop");

        // a_buf / b_buf / c_buf drop here (the test's sole remaining retains).
    }

    /// Device-to-device copy with no host round-trip, mixing an oxicuda-owned
    /// source with an imported external destination (the residency scenario:
    /// land a resident result into a consumer's cached buffer).
    #[test]
    #[cfg(target_os = "macos")]
    fn copy_dtod_owned_to_external() {
        let Some(backend) = try_init() else { return };
        let Some(device) = metal::Device::system_default() else {
            return;
        };

        let src_vals: Vec<f32> = vec![1.5, -2.0, 3.25, 4.0, 5.0, 6.5];
        let src_bytes = f32_to_bytes(&src_vals);
        let len = src_bytes.len();

        // Owned source (filled from host once), external destination.
        let src_h = match backend.alloc(len) {
            Ok(h) => h,
            Err(_) => return,
        };
        backend.copy_htod(src_h, &src_bytes).expect("htod src");

        let dst_buf = device.new_buffer(len as u64, metal::MTLResourceOptions::StorageModeShared);
        let dst_h = backend
            .import_buffer(dst_buf.clone(), len)
            .expect("import dst");

        backend.copy_dtod(dst_h, src_h, len).expect("copy_dtod");

        // Verify via the caller's own buffer pointer (zero-copy landed there).
        let landed =
            unsafe { std::slice::from_raw_parts(dst_buf.contents() as *const f32, src_vals.len()) };
        for (i, (&g, &w)) in landed.iter().zip(src_vals.iter()).enumerate() {
            assert!((g - w).abs() <= 1e-6, "dtod[{i}]={g} vs {w}");
        }

        // src == dst must be rejected.
        assert!(matches!(
            backend.copy_dtod(src_h, src_h, len),
            Err(oxicuda_backend::BackendError::InvalidArgument(_))
        ));

        backend.free(src_h).expect("free src");
        backend.free(dst_h).expect("free dst");
        // dst_buf still owned by the test; drop here is safe.
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_conv2d_identity_1x1() {
        let Some(b) = try_init() else { return };
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let filter = vec![2.0f32];
        let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();
        let ib = f32_to_bytes(&input);
        let fb = f32_to_bytes(&filter);
        let out_size = expected.len() * 4;
        let ih = match b.alloc(ib.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let fh = match b.alloc(fb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &ib).expect("htod input");
        b.copy_htod(fh, &fb).expect("htod filter");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero output");
        b.conv2d_forward(
            ih,
            &[1, 1, 3, 3],
            fh,
            &[1, 1, 1, 1],
            oh,
            &[1, 1, 3, 3],
            &[1, 1],
            &[0, 0],
        )
        .expect("conv2d 1x1");
        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "1x1 mismatch at {i}: {r} vs {e}");
        }
        b.free(ih).expect("free");
        b.free(fh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_conv2d_3x3_basic() {
        let Some(b) = try_init() else { return };
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let filter = vec![1.0f32; 9];
        let expected = [54.0f32, 63.0, 90.0, 99.0];
        let ib = f32_to_bytes(&input);
        let fb = f32_to_bytes(&filter);
        let out_size = expected.len() * 4;
        let ih = match b.alloc(ib.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let fh = match b.alloc(fb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &ib).expect("htod");
        b.copy_htod(fh, &fb).expect("htod");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");
        b.conv2d_forward(
            ih,
            &[1, 1, 4, 4],
            fh,
            &[1, 1, 3, 3],
            oh,
            &[1, 1, 2, 2],
            &[1, 1],
            &[0, 0],
        )
        .expect("conv2d 3x3");
        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "3x3 mismatch at {i}: {r} vs {e}");
        }
        b.free(ih).expect("free");
        b.free(fh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_conv2d_with_padding() {
        let Some(b) = try_init() else { return };
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let filter = vec![1.0f32; 9];
        let expected = vec![12.0, 21.0, 16.0, 27.0, 45.0, 33.0, 24.0, 39.0, 28.0];
        let ib = f32_to_bytes(&input);
        let fb = f32_to_bytes(&filter);
        let out_size = expected.len() * 4;
        let ih = match b.alloc(ib.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let fh = match b.alloc(fb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(ih, &ib).expect("htod");
        b.copy_htod(fh, &fb).expect("htod");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");
        b.conv2d_forward(
            ih,
            &[1, 1, 3, 3],
            fh,
            &[1, 1, 3, 3],
            oh,
            &[1, 1, 3, 3],
            &[1, 1],
            &[1, 1],
        )
        .expect("conv2d padded");
        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "pad mismatch at {i}: {r} vs {e}");
        }
        b.free(ih).expect("free");
        b.free(fh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_attention_uniform() {
        let Some(b) = try_init() else { return };
        let q = vec![1.0f32; 2 * 2];
        let k = vec![1.0f32; 3 * 2];
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let qb = f32_to_bytes(&q);
        let kb = f32_to_bytes(&k);
        let vb = f32_to_bytes(&v);
        let out_size = q.len() * 4;
        let qh = match b.alloc(qb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let kh = match b.alloc(kb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let vh = match b.alloc(vb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(qh, &qb).expect("htod q");
        b.copy_htod(kh, &kb).expect("htod k");
        b.copy_htod(vh, &vb).expect("htod v");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");
        let scale = 1.0 / (2.0f64).sqrt();
        b.attention(qh, kh, vh, oh, 1, 1, 2, 3, 2, scale, false)
            .expect("attention uniform");
        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for sq in 0..2 {
            let base = sq * 2;
            assert!(
                (result[base] - 3.0).abs() < 0.1,
                "sq={sq} d=0: {} vs 3.0",
                result[base]
            );
            assert!(
                (result[base + 1] - 4.0).abs() < 0.1,
                "sq={sq} d=1: {} vs 4.0",
                result[base + 1]
            );
        }
        b.free(qh).expect("free");
        b.free(kh).expect("free");
        b.free(vh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_attention_causal() {
        let Some(b) = try_init() else { return };
        let q = vec![1.0f32; 3 * 2];
        let k = vec![1.0f32; 3 * 2];
        let v = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let expected = [10.0f32, 20.0, 20.0, 30.0, 30.0, 40.0];
        let qb = f32_to_bytes(&q);
        let kb = f32_to_bytes(&k);
        let vb = f32_to_bytes(&v);
        let out_size = expected.len() * 4;
        let qh = match b.alloc(qb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let kh = match b.alloc(kb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let vh = match b.alloc(vb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(qh, &qb).expect("htod");
        b.copy_htod(kh, &kb).expect("htod");
        b.copy_htod(vh, &vb).expect("htod");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");
        let scale = 1.0 / (2.0f64).sqrt();
        b.attention(qh, kh, vh, oh, 1, 1, 3, 3, 2, scale, true)
            .expect("attention causal");
        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 0.5, "causal idx {i}: {r} vs {e}");
        }
        b.free(qh).expect("free");
        b.free(kh).expect("free");
        b.free(vh).expect("free");
        b.free(oh).expect("free");
    }
    #[test]
    #[cfg(target_os = "macos")]
    fn metal_attention_dominant_key() {
        let Some(b) = try_init() else { return };
        let q = vec![1.0f32, 0.0];
        let k = vec![0.0f32, 0.0, 0.0, 0.0, 10.0, 0.0];
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let qb = f32_to_bytes(&q);
        let kb = f32_to_bytes(&k);
        let vb = f32_to_bytes(&v);
        let out_size = q.len() * 4;
        let qh = match b.alloc(qb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let kh = match b.alloc(kb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let vh = match b.alloc(vb.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        let oh = match b.alloc(out_size) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(qh, &qb).expect("htod");
        b.copy_htod(kh, &kb).expect("htod");
        b.copy_htod(vh, &vb).expect("htod");
        b.copy_htod(oh, &vec![0u8; out_size]).expect("zero");
        b.attention(qh, kh, vh, oh, 1, 1, 1, 3, 2, 1.0, false)
            .expect("attention dominant");
        let mut out = vec![0u8; out_size];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        let result = bytes_to_f32(&out);
        assert!((result[0] - 5.0).abs() < 0.01, "d=0: {} vs 5.0", result[0]);
        assert!((result[1] - 6.0).abs() < 0.01, "d=1: {} vs 6.0", result[1]);
        b.free(qh).expect("free");
        b.free(kh).expect("free");
        b.free(vh).expect("free");
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn launch_custom_kernel_fma_roundtrip() -> oxicuda_backend::BackendResult<()> {
        use super::{read_f32_le, write_f32_le};

        let mut backend = MetalBackend::new();
        if backend.init().is_err() {
            // No Metal device available (e.g. headless CI) — skip gracefully.
            return Ok(());
        }

        // Fused multiply-add kernel with an explicit bounds check.
        let msl = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fma_kernel(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      n   [[buffer(3)]],
    constant float&     c   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] * b[gid] + c;
}
"#;

        let count: usize = 1024;
        let byte_len = count * std::mem::size_of::<f32>();
        // Deterministic inputs — no randomness.
        let a: Vec<f32> = (0..count).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..count).map(|i| i as f32 - 100.0).collect();
        let c: f32 = 3.25;

        let a_h = backend.alloc(byte_len)?;
        let b_h = backend.alloc(byte_len)?;
        let out_h = backend.alloc(byte_len)?;
        backend.copy_htod(a_h, &write_f32_le(&a))?;
        backend.copy_htod(b_h, &write_f32_le(&b))?;

        let n_le = (count as u32).to_le_bytes();
        let c_le = c.to_le_bytes();
        backend.launch_custom_kernel(
            msl,
            "fma_kernel",
            &[a_h, b_h, out_h],
            &[&n_le, &c_le],
            count,
        )?;

        let mut out_bytes = vec![0u8; byte_len];
        backend.copy_dtoh(&mut out_bytes, out_h)?;
        let out = read_f32_le(&out_bytes);

        for i in 0..count {
            let want = a[i] * b[i] + c;
            let tol = 1e-3 * want.abs().max(1.0);
            assert!(
                (out[i] - want).abs() <= tol,
                "index {i}: got {}, want {want}",
                out[i]
            );
        }

        // A second launch must reuse the cached pipeline and stay correct.
        backend.launch_custom_kernel(
            msl,
            "fma_kernel",
            &[a_h, b_h, out_h],
            &[&n_le, &c_le],
            count,
        )?;
        backend.copy_dtoh(&mut out_bytes, out_h)?;
        let out2 = read_f32_le(&out_bytes);
        for i in 0..count {
            let want = a[i] * b[i] + c;
            let tol = 1e-3 * want.abs().max(1.0);
            assert!((out2[i] - want).abs() <= tol, "second launch index {i}");
        }

        backend.free(a_h)?;
        backend.free(b_h)?;
        backend.free(out_h)?;
        Ok(())
    }

    // ─── Reduction: degenerate shapes and the two-pass path ──────────────────

    #[test]
    fn reduce_rejects_zero_length_dimensions() {
        let Some(b) = try_init() else { return };
        // Previously `.max(1)` rewrote the empty products to 1 and the kernel
        // wrote a garbage value with `Ok(())`.
        for (shape, axis) in [
            (vec![0usize, 4], 1usize),
            (vec![2, 0], 1),
            (vec![0], 0),
            (vec![3, 0, 5], 0),
        ] {
            let err = b
                .reduce(ReduceOp::Sum, 0, 0, &shape, axis)
                .expect_err("a zero-length dimension must be rejected");
            match err {
                BackendError::InvalidArgument(msg) => {
                    assert!(msg.contains("zero-length dimension"), "{msg}");
                }
                other => panic!("expected InvalidArgument, got {other:?}"),
            }
        }
    }

    #[test]
    fn reduce_mean_over_zero_extent_errors_instead_of_returning_nan() {
        let Some(b) = try_init() else { return };
        // `Mean` used to evaluate 0.0 / 0.0 in the kernel and silently store NaN.
        let err = b
            .reduce(ReduceOp::Mean, 0, 0, &[2, 0], 1)
            .expect_err("mean over a zero-length axis must be rejected");
        assert!(matches!(err, BackendError::InvalidArgument(_)), "{err:?}");
    }

    /// Run `op` over `count` elements of the deterministic ramp
    /// `v[i] = (i % 97) as f32 - 48.0` and return the GPU result.
    #[cfg(target_os = "macos")]
    fn reduce_ramp(b: &MetalBackend, op: ReduceOp, count: usize) -> Option<f32> {
        let input: Vec<f32> = (0..count).map(|i| (i % 97) as f32 - 48.0).collect();
        let bytes_in = f32_to_bytes(&input);
        let ih = b.alloc(bytes_in.len()).ok()?;
        let oh = b.alloc(4).ok()?;
        b.copy_htod(ih, &bytes_in).expect("htod");
        b.copy_htod(oh, &[0u8; 4]).expect("zero output");
        b.reduce(op, ih, oh, &[count], 0).expect("reduce");
        let mut out = vec![0u8; 4];
        b.copy_dtoh(&mut out, oh).expect("dtoh");
        b.free(ih).expect("free");
        b.free(oh).expect("free");
        Some(bytes_to_f32(&out)[0])
    }

    /// CPU oracle for [`reduce_ramp`].
    #[cfg(target_os = "macos")]
    fn reduce_ramp_expected(op: ReduceOp, count: usize) -> f32 {
        let input: Vec<f32> = (0..count).map(|i| (i % 97) as f32 - 48.0).collect();
        match op {
            ReduceOp::Sum => input.iter().sum(),
            ReduceOp::Mean => input.iter().sum::<f32>() / count as f32,
            ReduceOp::Max => input.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            ReduceOp::Min => input.iter().copied().fold(f32::INFINITY, f32::min),
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_flat_matches_cpu_on_both_sides_of_the_two_pass_threshold() {
        let Some(b) = try_init() else { return };
        // 1024 stays on the single-threadgroup kernel; 65_536 crosses the
        // two-pass threshold, so the same data checks both code paths against
        // the same oracle. 100_000 is the awkward geometry: it is not a multiple
        // of the 256-wide threadgroup, so pass 1 produces 391 partials (not a
        // power of two, and below the MAX_REDUCE_GROUPS clamp) which pass 2 then
        // folds with a 256-lane strided load.
        for count in [1024usize, 65_536, 100_000] {
            for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min, ReduceOp::Mean] {
                let Some(got) = reduce_ramp(&b, op, count) else {
                    return;
                };
                let want = reduce_ramp_expected(op, count);
                let tol = 1e-3 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= tol,
                    "count={count} op={op:?}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn reduce_flat_two_pass_is_exact_for_a_large_uniform_tensor() {
        let Some(b) = try_init() else { return };
        // 2^20 ones: the sum (1048576) and the mean (1.0) are both exactly
        // representable in f32, so any threadgroup that silently dropped lanes
        // would show up immediately.
        let count = 1usize << 20;
        let bytes_in = f32_to_bytes(&vec![1.0f32; count]);
        let Ok(ih) = b.alloc(bytes_in.len()) else {
            return;
        };
        let Ok(oh) = b.alloc(4) else { return };
        b.copy_htod(ih, &bytes_in).expect("htod");
        for (op, want) in [
            (ReduceOp::Sum, count as f32),
            (ReduceOp::Mean, 1.0f32),
            (ReduceOp::Max, 1.0f32),
            (ReduceOp::Min, 1.0f32),
        ] {
            b.copy_htod(oh, &[0u8; 4]).expect("zero output");
            b.reduce(op, ih, oh, &[count], 0).expect("reduce");
            let mut out = vec![0u8; 4];
            b.copy_dtoh(&mut out, oh).expect("dtoh");
            let got = bytes_to_f32(&out)[0];
            assert!((got - want).abs() < 1e-3, "{op:?}: got {got}, want {want}");
        }
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }

    // ─── Pipeline cache ──────────────────────────────────────────────────────

    #[test]
    #[cfg(target_os = "macos")]
    fn repeated_builtin_dispatches_reuse_one_cached_pipeline() {
        let Some(b) = try_init() else { return };
        let input = vec![-1.0f32, 2.0, -3.0, 4.0];
        let bytes_in = f32_to_bytes(&input);
        let Ok(ih) = b.alloc(bytes_in.len()) else {
            return;
        };
        let Ok(oh) = b.alloc(bytes_in.len()) else {
            return;
        };
        b.copy_htod(ih, &bytes_in).expect("htod");
        assert_eq!(b.pipeline_cache_len(), 0);
        for _ in 0..5 {
            b.unary(UnaryOp::Relu, ih, oh, input.len())
                .expect("relu dispatch");
        }
        // Five dispatches of the same op must compile exactly one pipeline —
        // the semantic key means the MSL source is not even regenerated.
        assert_eq!(b.pipeline_cache_len(), 1);
        // A different op is a different key.
        b.unary(UnaryOp::Abs, ih, oh, input.len())
            .expect("abs dispatch");
        assert_eq!(b.pipeline_cache_len(), 2);
        b.free(ih).expect("free");
        b.free(oh).expect("free");
    }

    /// Guards the [`super::super::types::PipelineKey`] `Builtin` invariant for
    /// the one dispatch path whose threadgroup width is **data dependent**.
    ///
    /// `dispatch_reduce` derives its width from the reduce extent
    /// (`pow2_threadgroup(next_power_of_2(n).min(REDUCE_TARGET_THREADS))`), so
    /// n=8 runs 8 lanes and n=1024 runs 256 — yet both map to the single key
    /// `{kind: "reduce", op: "sum", dtype: "f32"}` and therefore share one
    /// compiled pipeline. That is only sound because `reduction_msl` bakes
    /// nothing but `op` into the source and reads the width at runtime via
    /// `[[threads_per_threadgroup]]`.
    ///
    /// If anyone ever bakes a width (or a math mode, or a shape constant) into
    /// that generator without extending `PipelineKey`, the second reduce here
    /// silently executes the *first* one's kernel and this test fails. Both
    /// extents stay below `TWO_PASS_REDUCE_THRESHOLD` so neither is diverted to
    /// the separately-keyed chunked path.
    #[test]
    #[cfg(target_os = "macos")]
    fn one_cached_reduce_pipeline_serves_two_different_threadgroup_widths() {
        let Some(b) = try_init() else { return };
        assert_eq!(b.pipeline_cache_len(), 0);
        let Ok(oh) = b.alloc(4) else { return };
        // Sums of all-ones are exactly representable in f32, so a dropped lane
        // is visible rather than hidden in rounding.
        for count in [8usize, 1024] {
            let bytes_in = f32_to_bytes(&vec![1.0f32; count]);
            let Ok(ih) = b.alloc(bytes_in.len()) else {
                return;
            };
            b.copy_htod(ih, &bytes_in).expect("htod");
            b.copy_htod(oh, &[0u8; 4]).expect("zero output");
            b.reduce(ReduceOp::Sum, ih, oh, &[count], 0)
                .expect("reduce sum");
            let mut out = vec![0u8; 4];
            b.copy_dtoh(&mut out, oh).expect("dtoh");
            let got = bytes_to_f32(&out)[0];
            assert!(
                (got - count as f32).abs() < 1e-3,
                "count={count}: got {got}, want {count}"
            );
            b.free(ih).expect("free");
        }
        // The decisive assertion: one entry, not two — both widths really did
        // go through the same cached pipeline.
        assert_eq!(b.pipeline_cache_len(), 1);
        b.free(oh).expect("free");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipeline_cache_evicts_least_recently_used_past_capacity() {
        let mut b = MetalBackend::with_pipeline_cache_capacity(2);
        if b.init().is_err() {
            return;
        }
        let Ok(h) = b.alloc(std::mem::size_of::<f32>()) else {
            return;
        };
        // Three distinct sources sharing one entry-point name: with a 64-bit
        // source hash a collision would silently reuse the wrong pipeline, and
        // with an unbounded cache all three would be retained.
        let sources: Vec<String> = (0..3)
            .map(|i| {
                format!(
                    "#include <metal_stdlib>\nusing namespace metal;\n\
                     kernel void tiny(device float* out [[buffer(0)]], uint gid \
                     [[thread_position_in_grid]]) {{ out[gid] = {i}.0f; }}\n"
                )
            })
            .collect();
        for (i, src) in sources.iter().enumerate() {
            b.launch_custom_kernel(src, "tiny", &[h], &[], 1)
                .expect("custom launch");
            assert_eq!(b.pipeline_cache_len(), (i + 1).min(2));
        }
        // The cache is bounded; the last write still produced the right value.
        let mut out = vec![0u8; 4];
        b.copy_dtoh(&mut out, h).expect("dtoh");
        assert!((bytes_to_f32(&out)[0] - 2.0).abs() < 1e-6);
        // Re-running source 0 recompiles it (it was evicted) and must produce
        // *its* value, not the surviving entry's.
        b.launch_custom_kernel(&sources[0], "tiny", &[h], &[], 1)
            .expect("relaunch");
        b.copy_dtoh(&mut out, h).expect("dtoh");
        assert!((bytes_to_f32(&out)[0]).abs() < 1e-6);
        assert_eq!(b.pipeline_cache_len(), 2);
        b.free(h).expect("free");
    }

    // ─── Non-macOS stub parity ───────────────────────────────────────────────

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn external_import_api_exists_off_macos_and_reports_unsupported() {
        use super::super::types::MetalExternalBuffer;
        let b = MetalBackend::new();
        // Uninitialised (always the case off macOS) reports NotInitialized …
        assert_eq!(
            b.register_external(&MetalExternalBuffer, 16),
            Err(BackendError::NotInitialized)
        );
        assert_eq!(
            b.import_buffer(MetalExternalBuffer, 16),
            Err(BackendError::NotInitialized)
        );
    }
}
