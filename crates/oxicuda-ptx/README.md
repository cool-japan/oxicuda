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

## Features

All functionality is available by default -- no optional feature flags.
The crate is 100% pure Rust with zero external tool requirements for PTX
text generation.

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
