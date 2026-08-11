//! On-device validation of the `oxicuda-ptx` `WarpVec` / `WarpMask` layer.
//!
//! Each test builds a kernel through the SIMD-flavored warp-vector API,
//! JIT-compiles it for the live device, launches it, and asserts numerical
//! equivalence against a scalar CPU oracle — covering every lowering path:
//! `redux.sync` fast path, `shfl.sync.bfly` butterfly (32- and 64-bit
//! composite), segmented widths, guarded scans, votes/ballot/popc, and the
//! full shuffle family.

use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::builder::warp_vec::{WarpScanMode, WarpVec};
use oxicuda_ptx::ir::PtxType;

use super::{Lcg, gpu_fixture, load_kernel, params};

// ---------------------------------------------------------------------------
// relu_dot — the VectorWare-article example, one warp per 32 elements
// ---------------------------------------------------------------------------

/// `out[w] = Σ_{i ∈ warp w} max(x[i]·y[i], 0)` — eight warps in one block.
#[test]
fn warp_vec_relu_dot_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    const N: usize = 256;
    const WARPS: usize = N / 32;

    let mut rng = Lcg::new(0xD07_F32);
    // Small integers as f32: every product and 32-term sum is exact.
    let x: Vec<f32> = (0..N).map(|_| rng.below(15) as f32 - 7.0).collect();
    let y: Vec<f32> = (0..N).map(|_| rng.below(15) as f32 - 7.0).collect();
    let expected: Vec<f32> = (0..WARPS)
        .map(|w| {
            (0..32)
                .map(|i| (x[w * 32 + i] * y[w * 32 + i]).max(0.0))
                .sum()
        })
        .collect();

    let ptx = KernelBuilder::new("warp_relu_dot")
        .target(fx.sm)
        .param("out", PtxType::U64)
        .param("x", PtxType::U64)
        .param("y", PtxType::U64)
        .body(|b| {
            let tid = b.thread_id_x();
            let x_ptr = b.load_param_u64("x");
            let y_ptr = b.load_param_u64("y");
            let x_addr = b.f32_elem_addr(x_ptr, tid.clone());
            let y_addr = b.f32_elem_addr(y_ptr, tid);
            let x_val = b.load_global_f32(x_addr);
            let y_val = b.load_global_f32(y_addr);
            let xv = WarpVec::from_register(x_val).expect("x vec");
            let yv = WarpVec::from_register(y_val).expect("y vec");

            let dot = xv
                .mul(b, &yv)
                .expect("mul")
                .relu(b)
                .expect("relu")
                .reduce_sum(b)
                .expect("reduce");

            // Lane 0 of each warp stores the warp's aggregate.
            let warp_idx = b.warp_index_x();
            let out_ptr = b.load_param_u64("out");
            let out_addr = b.f32_elem_addr(out_ptr, warp_idx);
            let lane = b.lane_id();
            let one = b.mov_imm_u32(1);
            b.if_lt_u32(lane, one, |b| {
                b.store_global_f32(out_addr.clone(), dot.register().clone());
            });
            b.ret();
        })
        .build()
        .expect("gen relu_dot");

    let kernel = load_kernel(&ptx, "warp_relu_dot");
    let stream = fx.stream();
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_y = DeviceBuffer::<f32>::from_host(&y).expect("d_y");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32; WARPS]).expect("d_out");
    kernel
        .launch(
            &params(1, N as u32),
            &stream,
            &(
                d_out.as_device_ptr(),
                d_x.as_device_ptr(),
                d_y.as_device_ptr(),
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; WARPS];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "relu_dot per-warp partials");
}

// ---------------------------------------------------------------------------
// Reductions — redux fast path, butterfly, f64 composite, segmented
// ---------------------------------------------------------------------------

/// Builds a one-warp kernel reducing 32 lanes of `ty` and storing every
/// lane's (all-reduce) result.
fn reduce_kernel(sm: oxicuda_ptx::arch::SmVersion, ty: PtxType, width: u32) -> String {
    let name = "warp_vec_reduce";
    KernelBuilder::new(name)
        .target(sm)
        .param("out", PtxType::U64)
        .param("x", PtxType::U64)
        .body(move |b| {
            let lane = b.lane_id();
            let x_ptr = b.load_param_u64("x");
            let out_ptr = b.load_param_u64("out");
            match ty {
                PtxType::F64 => {
                    let addr = b.f64_elem_addr(x_ptr, lane.clone());
                    let val = b.load_global_f64(addr);
                    let vec = WarpVec::from_register_segmented(val, width).expect("vec");
                    let sum = vec.reduce_sum(b).expect("reduce");
                    let out_addr = b.f64_elem_addr(out_ptr, lane);
                    b.store_global_f64(out_addr, sum.into_register());
                }
                PtxType::S32 => {
                    let addr = b.f32_elem_addr(x_ptr, lane.clone());
                    let val = b.load_global_i32(addr);
                    let vec = WarpVec::from_register_segmented(val, width).expect("vec");
                    let sum = vec.reduce_sum(b).expect("reduce");
                    let out_addr = b.f32_elem_addr(out_ptr, lane);
                    b.store_global_i32(out_addr, sum.into_register());
                }
                _ => {
                    let addr = b.f32_elem_addr(x_ptr, lane.clone());
                    let val = b.load_global_u32(addr);
                    let vec = WarpVec::from_register_segmented(val, width).expect("vec");
                    let sum = vec.reduce_sum(b).expect("reduce");
                    let out_addr = b.f32_elem_addr(out_ptr, lane);
                    b.store_global_u32(out_addr, sum.into_register());
                }
            }
            b.ret();
        })
        .build()
        .expect("gen reduce")
}

/// u32 full-warp sum — the `redux.sync` fast path on `sm_80`+ devices.
#[test]
fn warp_vec_reduce_sum_u32_full() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x0EDD);
    let input: Vec<u32> = (0..32).map(|_| rng.below(100_000)).collect();
    let expected: u32 = input.iter().sum();

    let ptx = reduce_kernel(fx.sm, PtxType::U32, 32);
    if fx.sm.capabilities().has_redux {
        assert!(
            ptx.contains("redux.sync.add.u32"),
            "sm_80+ u32 sum must take the redux path:\n{ptx}"
        );
    }
    let kernel = load_kernel(&ptx, "warp_vec_reduce");
    let stream = fx.stream();
    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<u32>::from_host(&[0_u32; 32]).expect("d_out");
    kernel
        .launch(
            &params(1, 32),
            &stream,
            &(d_out.as_device_ptr(), d_in.as_device_ptr()),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = [0_u32; 32];
    d_out.copy_to_host(&mut got).expect("copy");
    assert!(
        got.iter().all(|&v| v == expected),
        "all-reduce: every lane must hold {expected}, got {got:?}"
    );
}

/// s32 full-warp sum with negatives — must take the butterfly (no s32 redux).
#[test]
fn warp_vec_reduce_sum_s32_butterfly() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5163 ^ 0xFFFF);
    let input: Vec<i32> = (0..32).map(|_| rng.below(2000) as i32 - 1000).collect();
    let expected: i32 = input.iter().sum();

    let ptx = reduce_kernel(fx.sm, PtxType::S32, 32);
    assert!(
        !ptx.contains("redux.sync"),
        "s32 has no redux fast path:\n{ptx}"
    );
    assert!(ptx.contains("shfl.sync.bfly.b32"), "{ptx}");
    let kernel = load_kernel(&ptx, "warp_vec_reduce");
    let stream = fx.stream();
    let d_in = DeviceBuffer::<i32>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<i32>::from_host(&[0_i32; 32]).expect("d_out");
    kernel
        .launch(
            &params(1, 32),
            &stream,
            &(d_out.as_device_ptr(), d_in.as_device_ptr()),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = [0_i32; 32];
    d_out.copy_to_host(&mut got).expect("copy");
    assert!(
        got.iter().all(|&v| v == expected),
        "s32 all-reduce: every lane must hold {expected}, got {got:?}"
    );
}

/// f64 full-warp sum — exercises the 64-bit unpack/shuffle/pack datapath.
#[test]
fn warp_vec_reduce_sum_f64_composite() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0xF64);
    // Halves are exactly representable; 32-term sums stay exact.
    let input: Vec<f64> = (0..32)
        .map(|_| f64::from(rng.below(1000)) - 500.0 + 0.5)
        .collect();
    let expected: f64 = input.iter().sum();

    let ptx = reduce_kernel(fx.sm, PtxType::F64, 32);
    assert!(ptx.contains("mov.b64 {"), "composite unpack expected:\n{ptx}");
    let kernel = load_kernel(&ptx, "warp_vec_reduce");
    let stream = fx.stream();
    let d_in = DeviceBuffer::<f64>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<f64>::from_host(&[0.0_f64; 32]).expect("d_out");
    kernel
        .launch(
            &params(1, 32),
            &stream,
            &(d_out.as_device_ptr(), d_in.as_device_ptr()),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = [0.0_f64; 32];
    d_out.copy_to_host(&mut got).expect("copy");
    assert!(
        got.iter().all(|&v| (v - expected).abs() < 1e-9),
        "f64 all-reduce: every lane must hold {expected}, got {got:?}"
    );
}

/// Segmented width-16 u32 sum: the two half-warps reduce independently.
#[test]
fn warp_vec_reduce_sum_u32_segmented_width16() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5E6);
    let input: Vec<u32> = (0..32).map(|_| rng.below(10_000)).collect();
    let lo: u32 = input[..16].iter().sum();
    let hi: u32 = input[16..].iter().sum();

    let ptx = reduce_kernel(fx.sm, PtxType::U32, 16);
    assert!(
        !ptx.contains("redux.sync"),
        "segmented reduce must not take the full-warp redux path:\n{ptx}"
    );
    let kernel = load_kernel(&ptx, "warp_vec_reduce");
    let stream = fx.stream();
    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<u32>::from_host(&[0_u32; 32]).expect("d_out");
    kernel
        .launch(
            &params(1, 32),
            &stream,
            &(d_out.as_device_ptr(), d_in.as_device_ptr()),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = [0_u32; 32];
    d_out.copy_to_host(&mut got).expect("copy");
    assert!(
        got[..16].iter().all(|&v| v == lo),
        "low segment must hold {lo}, got {:?}",
        &got[..16]
    );
    assert!(
        got[16..].iter().all(|&v| v == hi),
        "high segment must hold {hi}, got {:?}",
        &got[16..]
    );
}

/// f32 min/max with negatives in one kernel (two outputs).
#[test]
fn warp_vec_reduce_min_max_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x314159);
    let input: Vec<f32> = (0..32).map(|_| rng.f32_in(-100.0, 100.0)).collect();
    let expected_min = input.iter().copied().fold(f32::INFINITY, f32::min);
    let expected_max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let ptx = KernelBuilder::new("warp_vec_minmax")
        .target(fx.sm)
        .param("out_min", PtxType::U64)
        .param("out_max", PtxType::U64)
        .param("x", PtxType::U64)
        .body(|b| {
            let lane = b.lane_id();
            let x_ptr = b.load_param_u64("x");
            let addr = b.f32_elem_addr(x_ptr, lane.clone());
            let val = b.load_global_f32(addr);
            let vec = WarpVec::from_register(val).expect("vec");
            let lo = vec.reduce_min(b).expect("min");
            let hi = vec.reduce_max(b).expect("max");
            let min_ptr = b.load_param_u64("out_min");
            let min_addr = b.f32_elem_addr(min_ptr, lane.clone());
            b.store_global_f32(min_addr, lo.into_register());
            let max_ptr = b.load_param_u64("out_max");
            let max_addr = b.f32_elem_addr(max_ptr, lane);
            b.store_global_f32(max_addr, hi.into_register());
            b.ret();
        })
        .build()
        .expect("gen minmax");

    let kernel = load_kernel(&ptx, "warp_vec_minmax");
    let stream = fx.stream();
    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_min = DeviceBuffer::<f32>::from_host(&[0.0_f32; 32]).expect("d_min");
    let d_max = DeviceBuffer::<f32>::from_host(&[0.0_f32; 32]).expect("d_max");
    kernel
        .launch(
            &params(1, 32),
            &stream,
            &(
                d_min.as_device_ptr(),
                d_max.as_device_ptr(),
                d_in.as_device_ptr(),
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got_min = [0.0_f32; 32];
    let mut got_max = [0.0_f32; 32];
    d_min.copy_to_host(&mut got_min).expect("copy min");
    d_max.copy_to_host(&mut got_max).expect("copy max");
    assert!(
        got_min.iter().all(|&v| (v - expected_min).abs() < f32::EPSILON),
        "min: expected {expected_min}, got {got_min:?}"
    );
    assert!(
        got_max.iter().all(|&v| (v - expected_max).abs() < f32::EPSILON),
        "max: expected {expected_max}, got {got_max:?}"
    );
}

// ---------------------------------------------------------------------------
// Scans
// ---------------------------------------------------------------------------

fn scan_kernel(sm: oxicuda_ptx::arch::SmVersion, mode: WarpScanMode) -> String {
    KernelBuilder::new("warp_vec_scan")
        .target(sm)
        .param("out", PtxType::U64)
        .param("x", PtxType::U64)
        .body(move |b| {
            let lane = b.lane_id();
            let x_ptr = b.load_param_u64("x");
            let addr = b.f32_elem_addr(x_ptr, lane.clone());
            let val = b.load_global_u32(addr);
            let vec = WarpVec::from_register(val).expect("vec");
            let scanned = vec.scan_sum(b, mode).expect("scan");
            let out_ptr = b.load_param_u64("out");
            let out_addr = b.f32_elem_addr(out_ptr, lane);
            b.store_global_u32(out_addr, scanned.into_register());
            b.ret();
        })
        .build()
        .expect("gen scan")
}

fn run_scan(mode: WarpScanMode) {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5CA2 ^ matches!(mode, WarpScanMode::Exclusive) as u64);
    let input: Vec<u32> = (0..32).map(|_| rng.below(1000)).collect();
    let mut expected = Vec::with_capacity(32);
    let mut acc = 0_u32;
    for &v in &input {
        match mode {
            WarpScanMode::Inclusive => {
                acc += v;
                expected.push(acc);
            }
            WarpScanMode::Exclusive => {
                expected.push(acc);
                acc += v;
            }
        }
    }

    let ptx = scan_kernel(fx.sm, mode);
    let kernel = load_kernel(&ptx, "warp_vec_scan");
    let stream = fx.stream();
    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<u32>::from_host(&[0_u32; 32]).expect("d_out");
    kernel
        .launch(
            &params(1, 32),
            &stream,
            &(d_out.as_device_ptr(), d_in.as_device_ptr()),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = [0_u32; 32];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got.to_vec(), expected, "{mode:?} scan mismatch");
}

#[test]
fn warp_vec_scan_sum_inclusive_u32() {
    run_scan(WarpScanMode::Inclusive);
}

#[test]
fn warp_vec_scan_sum_exclusive_u32() {
    run_scan(WarpScanMode::Exclusive);
}

// ---------------------------------------------------------------------------
// Masks: votes, ballot count, segmented any
// ---------------------------------------------------------------------------

/// Per-lane `count`, warp `any`/`all` (as 0/1), and segmented-width-8 `any`.
#[test]
fn warp_vec_mask_votes_and_counts() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0xBA110);
    let input: Vec<f32> = (0..32).map(|_| rng.f32_in(0.0, 1.0)).collect();
    let threshold = 0.5_f32;
    let expected_count = input.iter().filter(|&&v| v > threshold).count() as u32;
    let expected_any = u32::from(input.iter().any(|&v| v > threshold));
    let expected_all = u32::from(input.iter().all(|&v| v > threshold));
    let expected_seg_any: Vec<u32> = (0..32)
        .map(|i| {
            let seg = i / 8;
            u32::from(input[seg * 8..(seg + 1) * 8].iter().any(|&v| v > threshold))
        })
        .collect();

    let ptx = KernelBuilder::new("warp_vec_masks")
        .target(fx.sm)
        .param("out_count", PtxType::U64)
        .param("out_any", PtxType::U64)
        .param("out_all", PtxType::U64)
        .param("out_seg", PtxType::U64)
        .param("x", PtxType::U64)
        .body(move |b| {
            let lane = b.lane_id();
            let x_ptr = b.load_param_u64("x");
            let addr = b.f32_elem_addr(x_ptr, lane.clone());
            let val = b.load_global_f32(addr);
            let vec = WarpVec::from_register(val).expect("vec");
            let thresh = WarpVec::splat_f32(b, threshold);
            let mask = vec.gt(b, &thresh).expect("gt");

            let count = mask.count(b).expect("count");
            let one = WarpVec::splat_u32(b, 1);
            let zero = WarpVec::splat_u32(b, 0);
            let any_mask = mask.any(b).expect("any");
            let any_flag = any_mask.select(b, &one, &zero).expect("select");
            let all_mask = mask.all(b).expect("all");
            let all_flag = all_mask.select(b, &one, &zero).expect("select");

            // Segmented: reinterpret the comparison at width 8.
            let vec8 = vec.with_width(8).expect("w8");
            let thresh8 = thresh.with_width(8).expect("w8");
            let mask8 = vec8.gt(b, &thresh8).expect("gt8");
            let seg_any = mask8.any(b).expect("seg any");
            let one8 = one.with_width(8).expect("w8");
            let zero8 = zero.with_width(8).expect("w8");
            let seg_flag = seg_any.select(b, &one8, &zero8).expect("select");

            for (param, value) in [
                ("out_count", count),
                ("out_any", any_flag),
                ("out_all", all_flag),
                ("out_seg", seg_flag),
            ] {
                let ptr = b.load_param_u64(param);
                let out_addr = b.f32_elem_addr(ptr, lane.clone());
                b.store_global_u32(out_addr, value.into_register());
            }
            b.ret();
        })
        .build()
        .expect("gen masks");

    let kernel = load_kernel(&ptx, "warp_vec_masks");
    let stream = fx.stream();
    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_count = DeviceBuffer::<u32>::from_host(&[0_u32; 32]).expect("d_count");
    let d_any = DeviceBuffer::<u32>::from_host(&[0_u32; 32]).expect("d_any");
    let d_all = DeviceBuffer::<u32>::from_host(&[0_u32; 32]).expect("d_all");
    let d_seg = DeviceBuffer::<u32>::from_host(&[0_u32; 32]).expect("d_seg");
    kernel
        .launch(
            &params(1, 32),
            &stream,
            &(
                d_count.as_device_ptr(),
                d_any.as_device_ptr(),
                d_all.as_device_ptr(),
                d_seg.as_device_ptr(),
                d_in.as_device_ptr(),
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = [0_u32; 32];
    d_count.copy_to_host(&mut got).expect("copy count");
    assert!(
        got.iter().all(|&v| v == expected_count),
        "count: expected {expected_count} in every lane, got {got:?}"
    );
    d_any.copy_to_host(&mut got).expect("copy any");
    assert!(
        got.iter().all(|&v| v == expected_any),
        "any: expected {expected_any}, got {got:?}"
    );
    d_all.copy_to_host(&mut got).expect("copy all");
    assert!(
        got.iter().all(|&v| v == expected_all),
        "all: expected {expected_all}, got {got:?}"
    );
    d_seg.copy_to_host(&mut got).expect("copy seg");
    assert_eq!(got.to_vec(), expected_seg_any, "segmented any per lane");
}

// ---------------------------------------------------------------------------
// Shuffles: reverse, broadcast, dynamic rotation, f64 shuffle_down
// ---------------------------------------------------------------------------

#[test]
fn warp_vec_shuffles_on_device() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5FF1E);
    let input: Vec<u32> = (0..32).map(|_| rng.below(1_000_000)).collect();
    let expected_rev: Vec<u32> = (0..32).map(|i| input[31 - i]).collect();
    let expected_bc: Vec<u32> = vec![input[7]; 32];
    // Dynamic rotation by +1: lane i receives input[(i + 1) % 32] (the
    // hardware masks the index to its low 5 bits at width 32).
    let expected_rot: Vec<u32> = (0..32).map(|i| input[(i + 1) % 32]).collect();

    let ptx = KernelBuilder::new("warp_vec_shuffles")
        .target(fx.sm)
        .param("out_rev", PtxType::U64)
        .param("out_bc", PtxType::U64)
        .param("out_rot", PtxType::U64)
        .param("x", PtxType::U64)
        .body(|b| {
            let lane = b.lane_id();
            let x_ptr = b.load_param_u64("x");
            let addr = b.f32_elem_addr(x_ptr, lane.clone());
            let val = b.load_global_u32(addr);
            let vec = WarpVec::from_register(val).expect("vec");

            let rev = vec.reverse(b).expect("reverse");
            let bc = vec.broadcast(b, 7).expect("broadcast");
            let lanes = WarpVec::lane_ids(b);
            let one = WarpVec::splat_u32(b, 1);
            let next = lanes.add(b, &one).expect("lane+1");
            let rot = vec.shuffle_idx(b, &next).expect("rotate");

            for (param, value) in [("out_rev", rev), ("out_bc", bc), ("out_rot", rot)] {
                let ptr = b.load_param_u64(param);
                let out_addr = b.f32_elem_addr(ptr, lane.clone());
                b.store_global_u32(out_addr, value.into_register());
            }
            b.ret();
        })
        .build()
        .expect("gen shuffles");

    let kernel = load_kernel(&ptx, "warp_vec_shuffles");
    let stream = fx.stream();
    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    let d_rev = DeviceBuffer::<u32>::from_host(&[0_u32; 32]).expect("d_rev");
    let d_bc = DeviceBuffer::<u32>::from_host(&[0_u32; 32]).expect("d_bc");
    let d_rot = DeviceBuffer::<u32>::from_host(&[0_u32; 32]).expect("d_rot");
    kernel
        .launch(
            &params(1, 32),
            &stream,
            &(
                d_rev.as_device_ptr(),
                d_bc.as_device_ptr(),
                d_rot.as_device_ptr(),
                d_in.as_device_ptr(),
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = [0_u32; 32];
    d_rev.copy_to_host(&mut got).expect("copy rev");
    assert_eq!(got.to_vec(), expected_rev, "reverse");
    d_bc.copy_to_host(&mut got).expect("copy bc");
    assert_eq!(got.to_vec(), expected_bc, "broadcast lane 7");
    d_rot.copy_to_host(&mut got).expect("copy rot");
    assert_eq!(got.to_vec(), expected_rot, "dynamic rotation");
}

/// f64 `shuffle_down(1)` through the composite 64-bit datapath; the last
/// lane keeps its own value (hardware clamp).
#[test]
fn warp_vec_shuffle_down_f64() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0xD0F64);
    let input: Vec<f64> = (0..32)
        .map(|_| f64::from(rng.below(1_000_000)) + 0.25)
        .collect();
    let expected: Vec<f64> = (0..32).map(|i| input[(i + 1).min(31)]).collect();

    let ptx = KernelBuilder::new("warp_vec_f64_down")
        .target(fx.sm)
        .param("out", PtxType::U64)
        .param("x", PtxType::U64)
        .body(|b| {
            let lane = b.lane_id();
            let x_ptr = b.load_param_u64("x");
            let addr = b.f64_elem_addr(x_ptr, lane.clone());
            let val = b.load_global_f64(addr);
            let vec = WarpVec::from_register(val).expect("vec");
            let down = vec.shuffle_down(b, 1).expect("down");
            let out_ptr = b.load_param_u64("out");
            let out_addr = b.f64_elem_addr(out_ptr, lane);
            b.store_global_f64(out_addr, down.into_register());
            b.ret();
        })
        .build()
        .expect("gen f64 down");

    let kernel = load_kernel(&ptx, "warp_vec_f64_down");
    let stream = fx.stream();
    let d_in = DeviceBuffer::<f64>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<f64>::from_host(&[0.0_f64; 32]).expect("d_out");
    kernel
        .launch(
            &params(1, 32),
            &stream,
            &(d_out.as_device_ptr(), d_in.as_device_ptr()),
        )
        .expect("launch");
    stream.synchronize().expect("sync");

    let mut got = [0.0_f64; 32];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got.to_vec(), expected, "f64 shuffle_down(1)");
}
