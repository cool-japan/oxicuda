# oxicuda-ptx

Pure Rust PTX code generation DSL and intermediate representation for GPU kernel development.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-ptx` provides a complete Rust-native DSL and intermediate representation
for generating NVIDIA PTX (Parallel Thread Execution) assembly code at runtime.
It eliminates the dependency on `nvcc`, the proprietary CUDA Toolkit, or any
C/C++ compiler toolchain -- PTX text is constructed entirely from safe Rust code.

The crate spans every stage of kernel authoring: a typed IR that catches operand
mismatches at construction time, a fluent builder API for rapid kernel prototyping,
high-level templates for common workloads (GEMM, reduction, softmax, elementwise),
and Tensor Core helpers covering three hardware generations (WMMA, MMA, WGMMA).
A disk-based cache avoids redundant regeneration of identical kernels.

## Architecture

| Module         | Purpose                                                    |
|----------------|------------------------------------------------------------|
| `ir`           | Typed intermediate representation -- registers, ~40 opcodes, basic blocks, functions, modules |
| `builder`      | Fluent builder API -- `KernelBuilder` (params, shared mem, target) and `BodyBuilder` (register alloc, arithmetic, control flow, sync) |
| `templates`    | Parameterized kernel templates -- elementwise, GEMM, reduction, softmax |
| `tensor_core`  | Tensor Core instruction generation -- WMMA (sm_70+), MMA (sm_80+), WGMMA (sm_90+) |
| `emit`         | PTX text printer and structural validator                  |
| `arch`         | Architecture definitions and capability queries (sm_75 through sm_120) |
| `cache`        | Disk-based content-addressable PTX kernel cache            |
| `error`        | Error types for all PTX generation failure modes           |

### Supported Data Types

F16, BF16, F32, F64, U8--U64, S8--S64, Pred, B16--B64.

### Supported Architectures

sm_75 (Turing), sm_80/sm_86 (Ampere), sm_89 (Ada Lovelace), sm_90/sm_90a (Hopper), sm_100 (Blackwell), sm_120 (Next-gen Blackwell).

## Quick Start

```rust,no_run
use oxicuda_ptx::prelude::*;

// Build a vector-add kernel targeting Ampere
let ptx = KernelBuilder::new("vector_add")
    .target(SmVersion::Sm80)
    .param("a_ptr", PtxType::U64)
    .param("b_ptr", PtxType::U64)
    .param("c_ptr", PtxType::U64)
    .param("n", PtxType::U32)
    .body(|b| {
        let gid = b.global_thread_id_x();
        // ... load, add, store ...
    })
    .build()
    .expect("PTX generation failed");
```

### Low-Level IR

```rust
use oxicuda_ptx::ir::*;

let mut alloc = RegisterAllocator::new();
let tid = alloc.alloc(PtxType::U32);

let inst = Instruction::MovSpecial {
    dst: tid,
    special: SpecialReg::TidX,
};
assert!(inst.emit().contains("%tid.x"));
```

### Templates

High-level templates handle shared memory layout, thread coordination, and
architecture-specific optimizations automatically:

- `ElementwiseTemplate` -- unary/binary ops (add, relu, sigmoid)
- `ReductionTemplate` -- parallel block-level reductions (sum, max, min)
- `GemmTemplate` -- tiled matrix multiplication with epilogue support
- `SoftmaxTemplate` -- numerically stable row-wise softmax
- `templates::tiled_mainloop` -- reusable CTA-tiled f32 GEMM mainloop emitter (256-thread CTA, 128x128x8 tile, 8x8 register tile/thread); operand-agnostic via a `GlobalTap` callback so one emitter serves both a plain GEMM and a convolution's implicit im2col. Backs `oxicuda-dnn`'s CTA-tiled implicit-GEMM convolution engine.
- `ChannelBroadcastTemplate` / `PReluTemplate` (`templates::channel_broadcast`) -- ONNX-style `[1,C,1,1]`-vs-`[1,C,H,W]` channel-broadcast elementwise ops and the per-channel-slope generalization of `LeakyRelu`

## Features

All functionality is available by default -- no optional feature flags.
The crate is 100% pure Rust with zero external tool requirements for PTX
text generation.

A SIMD-flavored warp-vector expression layer (`WarpVec`/`WarpMask`, in
`builder::warp_vec`, re-exported from the prelude) treats a CUDA warp's
32 lanes as a first-class vector value -- elementwise arithmetic,
comparisons, reductions, scans, and the full shuffle family, generated
as PTX `shfl.sync`/`vote.sync`/`redux.sync` instructions. See the
crate's rustdoc for `WarpVec` for a worked example.

`BodyBuilder` also exposes vectorized shared-memory and TF32-rounding
primitives (`builder::body_builder::vector_mem_ops`): `load_shared_f32x4`/
`store_shared_f32x4`/`store_global_f32x4` (`ld`/`st.v4.f32`) and
`cvt_f32_to_tf32` (`cvt.rna.tf32.f32`), plus a matching
`KernelBuilder::shared_mem_aligned` for the 16-byte-aligned shared arrays
`.v4` accesses require.

## Status

| Version | Date       | Tests        |
|---------|------------|--------------|
| 0.5.5   | 2026-08-13 | 1064 passing |
| 0.5.4   | 2026-08-11 | 1061 passing |
| 0.5.2   | 2026-07-27 | 1035 passing |
| 0.3.0   | 2026-06-25 | 1006 passing |
| 0.2.0   | 2026-06-16 | 934 passing  |
| 0.1.4   | 2026-04-18 | 916 passing  |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
