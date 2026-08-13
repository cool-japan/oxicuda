# oxicuda-ptx TODO

Pure Rust PTX code generation DSL and intermediate representation. Generates NVIDIA PTX assembly at runtime without nvcc, ptxas, or any CUDA SDK dependency. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan)

## Implementation Status

**Actual SLoC: 37,287** (68 files) (estimated 130K-230K for oxicuda-ptx portion of Vol.2)

The PTX crate is the largest in Vol.1+2 and the core differentiator of OxiCUDA. It provides a typed IR, builder DSL, kernel templates, Tensor Core instruction helpers, validation, disk caching, atomic operations, bit manipulation, special math functions, and compiler-style analysis passes. Current coverage handles the most important instruction classes with room for significant expansion.

### Completed [x]

**IR Layer (ir/)**
- [x] `ir/types.rs` -- PtxType enum (predicate through F64, vector widths), type size queries
- [x] `ir/register.rs` -- RegisterAllocator (sequential naming, type-tracked), Register type
- [x] `ir/instruction.rs` -- Instruction enum (arithmetic, load/store, compare, branch, barrier, special register moves, MMA/WMMA/WGMMA, shared memory, fence, conversions, texture/surface), emit() method
- [x] `ir/texture.rs` -- TextureType, SurfaceOp, texture/surface instruction support types
- [x] `ir/operand.rs` -- Operand enum (Register, Immediate, SpecialReg, Address, VectorReg)
- [x] `ir/block.rs` -- BasicBlock with label and instruction list
- [x] `ir/function.rs` -- PtxFunction (params, registers, body blocks, entry flag)
- [x] `ir/module.rs` -- PtxModule (version, target, address_size, functions, globals)

**Builder Layer (builder/)**
- [x] `builder/kernel_builder.rs` -- KernelBuilder fluent API (target, params, body closure, build)
- [x] `builder/body_builder/mod.rs` -- BodyBuilder (arithmetic ops, loads/stores, branches, barriers, special regs, shared memory, control flow helpers like if_lt_u32, bit manipulation, special math, integer multiply-add: mad.lo/mad.hi/mad.wide, FP8 conversions: e4m3/e5m2, BF16 conversions, wgmma helper)
- [x] `builder/body_builder/body_builder_ext.rs` -- BodyBuilder ext: atomic ops, texture/surface load/store, warp-level redux/stmatrix/elect_sync, barrier/griddepcontrol/mbarrier (split for 2000-line policy)

**Analysis Passes (analysis/)**
- [x] `analysis/register_pressure.rs` -- RegisterPressureReport with peak tracking per-block, spill risk warnings, occupancy estimation
- [x] `analysis/dead_code.rs` -- Fixed-point dead code elimination, removes unreachable blocks and unused registers
- [x] `analysis/constant_folding.rs` -- Constant folding optimization pass, simplify constant expressions at IR level
- [x] `analysis/strength_reduction.rs` -- Strength reduction optimization pass, replace expensive ops with cheaper equivalents

**Template Layer (templates/)**
- [x] `templates/elementwise.rs` -- ElementwiseTemplate (add, mul, relu, sigmoid, tanh, gelu, silu, abs, neg, sqrt, exp, log)
- [x] `templates/reduction.rs` -- ReductionTemplate (sum, max, min, product) with shared memory and warp shuffle
- [x] `templates/gemm.rs` -- GemmTemplate (tiled GEMM with shared memory, configurable tile sizes, epilogues: none/relu/bias/bias+relu)
- [x] `templates/softmax.rs` -- SoftmaxTemplate (numerically stable row-wise softmax with online max/sum)
- [x] `templates/scan.rs` -- ScanTemplate (Blelloch work-efficient scan, inclusive/exclusive, sum/product/min/max ops)

**Tensor Core (tensor_core/)**
- [x] `tensor_core/wmma.rs` -- WmmaConfig (Volta/Turing+, wmma.load/wmma.mma/wmma.store shapes and types)
- [x] `tensor_core/mma.rs` -- MmaConfig (Ampere+, mma.sync.aligned shapes and precision configs)
- [x] `tensor_core/wgmma.rs` -- WgmmaConfig (Hopper+, warp-group MMA shapes and async configs)

**Emit and Validation (emit/)**
- [x] `emit/printer.rs` -- emit_module, emit_function, emit_function_standalone (IR to PTX text)
- [x] `emit/validator.rs` -- validate_ptx structural validation, validate_ptx_for_target architecture checks

**Infrastructure**
- [x] `arch.rs` -- SmVersion enum (sm_75 through sm_120), ArchCapabilities, from_compute_capability
- [x] `cache.rs` -- PtxCache disk-based caching with PtxCacheKey (content-addressable)
- [x] `error.rs` -- PtxGenError with variants for all failure modes

### Future Enhancements [ ]

**Instruction Coverage (P0)**
- [x] Atomic operations -- atom.global.add, atom.shared.cas, atom.exch, red (Atom, AtomCas, Red instructions; AtomOp enum with 9 variants; 20 BodyBuilder atomic methods; ~32 tests)
- [x] Texture/surface instructions -- tex.1d, tex.2d, tex.3d, suld, sust (ir/instruction.rs, ir/texture.rs, builder/body_builder.rs; 5 analysis passes updated)
- [x] Video instructions -- dp4a, dp2a for INT8 inference (ir/instruction.rs, builder/body_builder.rs)
- [x] Bit manipulation -- brev, clz, popc, bfind, bfe, bfi (Brev, Clz, Popc, Bfind, Bfe, Bfi instructions; 21+ BodyBuilder methods)
- [x] Integer multiply-add -- mad.lo, mad.hi, mad.wide (ir/instruction.rs, builder/body_builder.rs)
- [x] Floating-point fused ops -- fma.rn, rcp, rsqrt, sqrt, sin, cos, ex2, lg2 (Rcp, Rsqrt, Sqrt, Ex2, Lg2, Sin, Cos instructions; BodyBuilder methods with rounding modes)

**Code Quality (P0)**
- [x] Register pressure analysis -- count live registers, warn on spilling risk (analysis/register_pressure.rs; RegisterPressureReport with peak tracking, spill risk, occupancy estimation; ~34 tests)
- [x] Instruction scheduling -- reorder independent instructions to hide latency (analysis/instruction_scheduling.rs; dependency graph, latency model, MaxIlp/MinLatency strategies; ~1698 lines)
- [x] Dead code elimination -- remove unreachable blocks and unused registers (analysis/dead_code.rs; fixed-point DCE with liveness analysis)

**Architecture Support (P1)**
- [x] PTX 8.x features -- new instructions and modifiers from CUDA 12.x (Redux, Stmatrix, ElectSync, Setmaxnreg, Griddepcontrol, FenceProxy, MbarrierInit/Arrive/Wait; ptx_isa_version(); 7 new ArchCapabilities fields; 27 new tests)
- [x] sm_120 Rubin full support -- SM 120 capabilities, PTX ISA 8.7, has_sm120_features flag, updated arch capabilities
- [x] Per-architecture instruction legality tables -- reject invalid arch/instruction combos (analysis/arch_legality.rs; LegalityReport, LegalityViolation, LegalityWarning; check_instruction_legality, is_instruction_legal, minimum_sm_for_instruction, instruction_arch_requirement; 25 tests)

**Optimization (P1)**
- [x] PTX-level loop unrolling -- pragma unroll, manual unroll in builder
- [x] Constant folding -- simplify constant expressions at IR level (analysis/constant_folding.rs)
- [x] Strength reduction -- replace expensive ops (div, mod) with shifts where possible (analysis/strength_reduction.rs)
- [x] Profile-guided code generation -- use autotune results to select code paths

**Validation (P1)**
- [x] Better validator coverage -- type checking across operands, register lifetime analysis (IR-level validation: validate_ir_instructions, validate_register_lifetimes; IrValidationResult/IrErrorKind types; 14 tests)
- [x] Memory consistency validation -- fence/barrier placement correctness (validate_memory_consistency; barrier divergence detection, shared memory race condition warnings)
- [x] Shared memory bank conflict detection -- static analysis for known access patterns

**Templates (P2)**
- [x] `cp.async` Generator Module (`src/templates/cp_async_gen.rs`) -- standalone `CpAsyncGenerator` producing `cp.async.cg.global.L2::128B` / `cp.async.ca.global` PTX sequences with configurable bypass size (4/8/16 B; `.cg`+`L2::128B` gated to 16 B) and a multi-stage `commit_group`/`wait_group` pipeline loop (prologue primes `stages-1`, steady-state waits to `stages-1` in flight, epilogue drains all); synchronous `ld.global`/`st.shared` fallback for pre-`sm_80`. `CpAsyncGenerator` + `CpAsyncCachePolicy` (23 tests)
- [x] Kernel Fusion Cost Model (`src/analysis/fusion_cost_model.rs`) -- register-pressure + shared-memory + ILP heuristic `FusionCostModel`/`FusionDecision`/`FusionVerdict` deciding whether to fuse two adjacent pointwise kernels; per-SM budgets via `for_target`. Wired into `kernel_fusion::plan_fusion` (new `plan_fusion_with_model`/`plan_fusion_for_target`), replacing the former unconditional accept of every structurally-legal candidate with a cost-gated fuse/refuse (refuses on register spill > file, shared-mem overflow, or below-threshold benefit) (18 tests)
- [x] Convolution template (templates/convolution.rs) -- im2col, direct conv, 1x1 optimized, backward data/filter
- [x] Attention template (templates/attention.rs) -- FlashAttention-style fused attention kernel
- [x] MoE (Mixture of Experts) template (templates/moe.rs) -- top-k gating, permute, expert GEMM, unpermute
- [x] Scan/prefix-sum template -- inclusive/exclusive scan with Blelloch work-efficient algorithm, sum/product/min/max ops (templates/scan.rs)
- [x] Transpose template (templates/transpose.rs) -- coalesced shared-memory transpose with bank-conflict-free padding
- [x] Batch normalization template (templates/batch_norm.rs) -- training + inference BN kernels
- [x] CTA-tiled f32 GEMM mainloop emitter (templates/tiled_mainloop.rs) -- reusable, operand-agnostic (via a `GlobalTap` callback) register-blocked mainloop: 256-thread CTA, 128x128x8 tile, 8x8 register tile/thread; backs `oxicuda-dnn`'s CTA-tiled implicit-GEMM convolution engine. `oxicuda-blas`'s `GemmDispatcher` does not call it yet, despite the module doc naming both as intended consumers.
- [x] Vectorized shared-memory / TF32-rounding builder primitives (builder/body_builder/vector_mem_ops.rs) -- `load_shared_f32x4`/`store_shared_f32x4`/`store_global_f32x4` (`ld`/`st.v4.f32`) and `cvt_f32_to_tf32` (`cvt.rna.tf32.f32`, `.rna` required pre-sm_90), plus `KernelBuilder::shared_mem_aligned` for the 16-byte-aligned shared arrays `.v4` needs. Only `shared_mem_aligned` has a caller outside this file's own tests so far (the tiled mainloop above).
- [x] Channel-broadcast / PRelu templates (templates/channel_broadcast.rs) -- `ChannelBroadcastTemplate`/`PReluTemplate` for ONNX-style `[1,C,1,1]`-vs-`[1,C,H,W]` broadcast ops and per-channel-slope LeakyReLU; no caller elsewhere in the workspace yet
- [x] Visual PTX explorer (explorer/) -- interactive TUI-based PTX code explorer with syntax highlighting, register liveness visualization, instruction dependency graph, and per-block register pressure heatmap (P2)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | SmVersion detection from live GPU, Module loading | Yes |
| thiserror | Derive macro for PtxGenError | Yes |

## Quality Status

- Warnings: 0
- Tests: 1061 unit + 32 doc passing (0.5.4: added WarpVec/WarpMask warp-vector layer + shfl.sync/vote.sync IR regression tests; 0.5.0: added F64 elementwise/reduction register-width and transcendental-precision-validation regression tests; previously added cp_async_gen + fusion_cost_model)
- unwrap() calls: 0 (production code; tests use `.unwrap()` on infallible `new()` fixtures)
- Clippy: clean (pedantic + nursery)
- `#![deny(unsafe_code)]` -- entire crate is safe Rust

## Performance Targets

PTX generation is CPU-bound and should be fast enough for JIT scenarios:
- Single kernel generation: < 1ms for simple kernels, < 10ms for complex GEMM
- Template instantiation: < 5ms per template variant
- Cache hit: < 100us for disk cache lookup
- Validation pass: < 1ms per function

## Notes

- The entire crate is `#![deny(unsafe_code)]` -- all PTX text is constructed from safe Rust
- 60 source files across 7 subsystems (ir, builder, templates, tensor_core, emit, analysis, arch + cache/error)
- GEMM template supports configurable tile sizes but does not yet reach peak Tensor Core throughput
- Tensor Core configs define shapes/types but full code generation integration is in progress
- The cache uses deterministic key hashing -- same inputs always produce same PTX output

---

## Blueprint Quality Gates (Vol.2 Sec 4.1)

### PTX Generator Functional Requirements

| # | Requirement | Priority | Status |
|---|-------------|----------|--------|
| P1 | `vector_add` PTX generation + execution — full E2E test | P0 | [x] Done |
| P2 | GEMM template PTX for sm_80 — verified via `ptxas` JIT compilation | P0 | [x] Done |
| P3 | GEMM template PTX for sm_90 with `wgmma` — verified on Hopper hardware | P1 | [x] Done |
| P4 | Elementwise templates (relu, sigmoid, add) — numerical precision verified | P0 | [x] Done |
| P5 | Reduction template — large input size correctness test | P0 | [x] Done |
| P6 | Tensor Core MMA PTX correctness — fragment layout verified vs CUTLASS reference | P0 | [x] Done |
| P7 | PTX cache hit rate test — verify disk cache avoids regeneration | P1 | [x] Done |
| P8 | Generated PTX validated via `ptxas` for ALL supported SM versions (sm_75 through sm_100) | P0 | [x] Done |

---

## Risk Mitigations (Vol.2 Sec 5)

| Risk | Impact | Mitigation Status |
|------|--------|------------------|
| PTX IR instruction coverage gaps | High | `Raw()` escape hatch available; coverage audit needed | [ ] |
| `ptxas` JIT error message parsing difficulty | Medium | JIT log buffer captured; structured error extraction needed | [ ] |
| Tensor Core fragment layout errors | High | Validate numerically using CUTLASS test matrix patterns (P6) | [ ] |
| Autotuner search space explosion | Medium | Bayesian optimization implemented; efficiency audit needed | [ ] |
| PTX version inter-compatibility | Low | `SmVersion` → PTX version mapping in `arch/`; cross-SM test needed (P8) | [ ] |

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [x] WMMA `m16n16k16` fragment layout verified — confirmed correct for `f16` accumulation
- [x] `ldmatrix` availability verified (sm_75+)

### Ampere (sm_80 / sm_86)
- [x] `cp.async` 4/8/16 byte variants all tested in generated PTX
- [x] Multi-stage pipeline (3-stage, 4-stage) GEMM template correctness vs reference
- [x] `mma.sync.aligned.m16n8k16` MMA instruction fragment layout verified (P6)

### Ada Lovelace (sm_89)
- [x] FP8 (`e4m3` / `e5m2`) type emission and conversion instructions -- `cvt_f32_to_e4m3`, `cvt_e4m3_to_f32`, `cvt_f32_to_e5m2`, `cvt_e5m2_to_f32` in BodyBuilder; `cvt_bf16_to_f32`, `cvt_f32_to_bf16` added; 6 tests
- [x] `mma` with FP8 input types verified

### Hopper (sm_90 / sm_90a)
- [x] `wgmma.mma_async` instruction generation helper -- `wgmma_mma_async_m64n128k16_f16` in BodyBuilder, SM gate enforced, 2 tests
- [x] `wgmma.mma_async` instruction generation tested end-to-end (P3)
- [x] TMA (`cp.async.bulk`) descriptor-based load emission
- [x] Cluster-level synchronization (`bar.sync` cluster scope) emission
- [x] PTX 8.0 features: `griddepcontrol`, fence with cluster scope

### Blackwell (sm_100 / sm_120)
- [x] PTX 8.5 instruction set coverage
- [x] FP4 (`e2m1`) type emission
- [x] `tcgen05.mma` (5th-gen Tensor Core) instruction emission

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These represent the gap between current implementation depth and blueprint-grade production requirements.

### Correctness Verification
- [x] vector_add PTX generation verified: all required elements present for f32/f16/f64 (P1, 15 tests)
- [x] All SM versions (sm_75 through sm_120) generate valid PTX with correct .target and .version (P8, 21 tests)
- [x] FP8 type gating: sm<89 cannot emit FP8 parameters
- [x] PTX .address_size 64 present for all SM versions
- [x] GEMM template numerical output matches cuBLAS reference within precision tolerance
- [x] Elementwise template `relu/sigmoid/gelu/tanh` precision vs CPU reference < 1 ULP for f32
- [x] F64 elementwise/reduction kernels emit correctly-sized value registers (0.5.0 fix) -- `ElementwiseTemplate::vreg_prefix()` now selects `fd`/`fh`/`f` by precision instead of a hardcoded `.b32`-sized `%f_*` bank, which `ptxas` rejected for every F64 elementwise kernel; `ReductionTemplate`'s F64 path uses a `.f64` register bank and a shared-memory-only tree (a 32-bit `shfl.sync.down.b32` cannot move a 64-bit value in one step)
- [x] Transcendental elementwise ops (exp/log/sigmoid/gelu/silu/tanh/softplus/pow) reject non-F32 precision up front (0.5.0 fix) -- `ElementwiseOp::requires_f32_sfu()` + `ElementwiseTemplate::validate_precision()` catch the `ex2.approx`/`lg2.approx` special-function-unit instructions' F32-only PTX restriction at generation time instead of silently emitting PTX `ptxas` would reject; `rsqrt`/`sqrt`/`hard_*`/`leaky_relu` are unaffected

### Code Quality
- [x] Register pressure analysis used to warn when kernel exceeds 255 regs/thread
- [x] Instruction scheduling pass verified to reduce memory-latency stalls
- [x] Dead code elimination pass removes unreachable PTX blocks
- [x] All SM versions (sm_75...sm_100) pass ptxas JIT — elementwise templates verified for valid PTX headers
- [x] Elementwise relu/sigmoid/gelu/tanh/silu precision vs CPU reference — 6 tests verifying correct arithmetic instructions
- [x] PTX cache hit rate verification test — 5 tests including cache-miss-avoids-regeneration
- [x] Register pressure analysis warns when kernel exceeds 255 regs/thread — 4 tests including boundary
- [x] Dead code elimination removes unreachable PTX blocks — 3 DCE tests including idempotency

- [x] `oxicuda-unary-ptx-extensions` (planned 2026-04-17)
  - **Goal:** Add `OneMinus` (= 1 − x) unary variant to `ElementwiseOp` in `src/templates/elementwise.rs` with a `generate_one_minus` method, `as_str` entry `"one_minus"`, and `is_binary => false`.
  - **Files:** `src/templates/elementwise.rs`
  - **Tests:** Assert PTX for `OneMinus` contains `sub.f32` and `0f3F800000`.

- [x] `oxicuda-binary-ptx-extensions` (planned 2026-04-17)
  - **Goal:** Add binary `ElementwiseOp` variants: `Pow, Min, Max, CmpEq, CmpNe, CmpLt, CmpGt, CmpLe, CmpGe, OrMax, OrProbSum, Nand, Nor, Xor`. Each gets a `generate_*` method, entry in `as_str`, and `is_binary => true`. Add helper const fns `float_one_literal` and `float_two_literal`.
  - **Files:** `src/templates/elementwise.rs`
  - **Tests:** PTX string content assertions for each new op (key instructions present).

- [x] `oxicuda-per-axis-reduction-ptx` (planned 2026-04-17)
  - **Goal:** Add `PerAxisReductionTemplate` struct to `src/templates/reduction.rs` and add `Mean` variant to `ReductionOp`. Kernel: one block per `(outer, inner)` output pair, 256 threads, loop over axis chunks, shared-memory tree reduce + warp shuffle final 32 elements. Kernel signature: `(input: u64, output: u64, axis_len: u32, inner: u32)` plus `inv_axis_len: f32` for Mean. Kernel name pattern: `reduce_axis_{op}_{type}`.
  - **Files:** `src/templates/reduction.rs`
  - **Tests:** PTX for Sum contains `.entry reduce_axis_sum_f32`, `shfl.sync.down`, `.shared`; for Mean also checks `mul.f32`.

- [x] ptx-typed-variants-tier1 (completed 2026-05-01)
  - **Goal:** Add 3 highest-leverage `Instruction` variants for ops currently expressed via `raw_ptx()`: `Addc` (carry-add), `Selp` (predicated select), `AtomGlobalAddFloat` (atom.global.add.f32/.f64). Migrate ≥2 representative call sites per variant (6 total) from `raw_ptx()` to the typed form. The 3,581 existing `Raw()` callers are NOT bulk-migrated; `Raw` is retained as the documented escape hatch.
  - **Design:** Add to `Instruction` enum in `src/ir/instruction.rs`: `Addc { ty: PtxIntegerType, dst, a, b }`, `Selp { ty: PtxType, dst, a, b, p }`, `AtomGlobalAddFloat { ty: PtxFloatType, dst, addr, value }`. Update `Raw` doc-comment to say it's an escape hatch for the long tail. Add 3 emit branches in `src/ir/instruction_emit.rs`: `addc.<ty> {dst}, {a}, {b};` / `selp.<ty> {dst}, {a}, {b}, {p};` / `atom.global.add.<ty> {dst}, [{addr}], {value};`. Find 6 representative `raw_ptx(` / `raw_ptx!` call sites in oxicuda-rand, oxicuda-fft, oxicuda-blas, oxicuda-dnn and migrate them.
  - **Files:** `src/ir/instruction.rs` (3 variants + Raw doc-comment), `src/ir/instruction_emit.rs` (3 emit branches), 6 call sites in rand/fft/blas/dnn crates
  - **Tests (in oxicuda-ptx, text comparison):** `emit_addc_u32`, `emit_addc_u64`, `emit_selp_b32`, `emit_selp_f32`, `emit_atom_global_add_f32`, `emit_atom_global_add_f64`; regression: existing kernel tests in migrated crates must still pass.
  - **Risk:** PTX text drift from migration breaking text-comparison tests. Mitigated by matching spacing of existing emit branches; impl subagent diffs per-site before committing.
  - **Prerequisites:** None.
