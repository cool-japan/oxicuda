# oxicuda-primitives TODO

CUB-equivalent parallel GPU primitives with zero CUDA SDK dependency. All kernels are generated as PTX source strings at runtime and JIT-compiled via `cuModuleLoadData`. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/C++, no nvcc, no CUB headers.

## Implementation Status

**Actual: ~10,712 non-blank SLoC across 33 files** (sub-modules `warp/`, `block/`, `device/`, `sort/`, `gpu_tests/` + shared `ptx_helpers.rs`, `handle.rs`, `host_reference.rs`, `error.rs`)

Coverage spans the full CUB hierarchy: warp-level reductions / scans built on `shfl.sync.*`, block-level reductions / scans built on shared memory + warp shuffles, device-wide reduce / scan / decoupled-lookback-scan / stream-compaction / partition / run-length-encode / segmented-reduce+scan / select-unique / histogram pipelines, and the sort family (4-bit & 8-bit LSD radix, key+value/descending/float radix, onesweep, bitonic-block + co-rank merge keys and pairs). A `host_reference` module mirrors every algorithm on the CPU (slice-based, no `ndarray`). Every public template generates target-specific PTX for sm_75 through sm_120 via `oxicuda-ptx` and exposes a `workspace_bytes(input_len)` scratch query.

### Completed [x]

#### Core infrastructure
- [x] `error.rs` (207 SLoC) -- `PrimitivesError` with 9 variants (`Cuda(#[from] CudaError)`, `BufferTooSmall`, `InputTooLarge`, `InvalidArgument`, `UnsupportedOperation`, `PtxGeneration`, `KernelLoad`, `KernelLaunch`, `DimensionMismatch`, `WorkspaceTooSmall`); convenience constructors `ptx()`, `load()`, `launch()`; `PrimitivesResult<T>` alias; 8 unit tests
- [x] `handle.rs` (160 SLoC) -- `PrimitivesHandle { ctx: Arc<Context>, stream: Arc<Stream>, sm: SmVersion }`; `new()`, `from_arc()`, `sm_version()`, `context()`, `stream()`; `cc_to_sm()` mapping (7.5/8.0/8.6/8.9/9.0/10.0/12.0 → Sm75/Sm80/Sm86/Sm89/Sm90/Sm100/Sm120, fallback Sm80); 3 unit tests
- [x] `lib.rs` (46 SLoC) -- module wiring and top-level re-exports (`PrimitivesError`, `PrimitivesResult`, `PrimitivesHandle`, `PrimitiveType`, `ReduceOp`) plus a doc-test that round-trips a `DeviceReduceTemplate` PTX generation

#### PTX code-generation helpers (`ptx_helpers.rs`, 499 SLoC)
- [x] `ptx_header(sm: SmVersion)` -- `.version` / `.target` lines including the Sm90a, Sm100, Sm120 cases
- [x] `ptx_type_str(PtxType)` / `ptx_type_bytes(PtxType)` -- full coverage of every variant defined in `oxicuda-ptx::ir::PtxType`: B8/B16/B32/B64/B128, U8/U16/U32/U64, S8/S16/S32/S64, F16/F16x2, BF16/BF16x2, F32, TF32, F64, E4M3/E5M2 (FP8), E2M3/E3M2 (FP6 variants), E2M1 (FP4), Pred
- [x] `reg_decl(reg_prefix, ty, count)` -- helper for emitting `.reg .ty %reg<N>;` blocks
- [x] `ReduceOp` enum: Sum, Product, Min, Max, And, Or, Xor (7 ops); `mnemonic_for(ty)`, `identity_literal(ty)` per op × type
- [x] `PrimitiveType` trait -- bridges Rust scalar types to `PtxType` and byte-width for templated kernel generators
- [x] 16 unit tests covering every SmVersion target string, every PtxType byte width, and every ReduceOp identity / mnemonic

#### Warp-level primitives (`warp/`, 861 SLoC)
- [x] `warp/reduce.rs` (439 SLoC) -- `WarpReduceConfig`, `WarpReduceTemplate`; `shfl.sync.bfly.b32` butterfly tree (5 steps for 32-lane warp); f64 split into lo/hi 32-bit registers and recombined; optional broadcast lane mask to write the result back to a chosen lane; all 7 reduce ops × all numeric types
- [x] `warp/scan.rs` (405 SLoC) -- `WarpScanConfig`, `WarpScanTemplate`, `ScanKind { Inclusive, Exclusive }`; `shfl.sync.up.b32` shift-and-combine (5 steps); f64 lo/hi split; exclusive variant subtracts the seed
- [x] 22 unit tests across reduce + scan (PTX string content checks, every op × type × SmVersion)

#### Block-level primitives (`block/`, 940 SLoC)
- [x] `block/reduce.rs` (476 SLoC) -- `BlockReduceConfig`, `BlockReduceTemplate`, `MAX_BLOCK_SIZE`; two-stage algorithm: per-warp reduction via `shfl.sync.bfly`, partial sums spilled to shared memory, warp 0 reduces the partials, broadcast back to all threads
- [x] `block/scan.rs` (464 SLoC) -- `BlockScanConfig`, `BlockScanTemplate`; work-efficient Blelloch algorithm with explicit up-sweep / down-sweep over shared memory; inclusive + exclusive; f64 split throughout
- [x] 26 unit tests across reduce + scan (block sizes 32/64/128/256/512/1024, every op × type)

#### Device-wide primitives (`device/`, 2,256 SLoC)
- [x] `device/reduce.rs` (579 SLoC) -- `DeviceReduceConfig`, `DeviceReduceTemplate`, `DEFAULT_BLOCK_SIZE`; 2-pass pipeline -- pass 1 emits per-block partials, pass 2 reduces the partials to a single scalar; all 7 ReduceOps × all numeric types × all SM versions; 11 unit tests
- [x] `device/scan.rs` (645 SLoC) -- `DeviceScanConfig`, `DeviceScanTemplate`; 3-kernel pipeline -- (1) per-block exclusive scan, (2) propagate block aggregates, (3) apply block prefix to local results; inclusive + exclusive; 8 unit tests
- [x] `device/select.rs` (487 SLoC) -- `DeviceSelectConfig`, `DeviceSelectTemplate`, `SelectPredicate { NonZero, Positive, Negative, FlagArray }`; 2-kernel flag+gather pipeline implemented on top of exclusive scan; type-correct `setp.{lt,gt,ne}.{ty}` per predicate × element type; unsigned types short-circuit `Negative` to always-false; 10 unit tests
- [x] `device/histogram.rs` (645 SLoC) -- `DeviceHistogramConfig`, `DeviceHistogramMode { Modulo, EvenRange }`, `DeviceHistogramTemplate`; 2-kernel init+count pipeline; per-block shared-memory privatized histogram with `atom.shared.add.u32` for collision-free per-bin updates; strided global merge; integer (`rem.u32` / `rem.u64`) and fp/integer linear `EvenRange` mappings; 11 unit tests
- [x] `device/run_length_encode.rs` -- `DeviceRunLengthEncodeConfig`/`Template`; head-flag → scan → gather → lengths pipeline; CPU `reference_run_length_encode`; 10 unit tests
- [x] `device/segmented.rs` -- `SegmentedReduceTemplate` (one block / segment, shared-memory tree reduce) + `SegmentedScanTemplate` (one thread / segment, inclusive/exclusive); CPU references; 10 unit tests
- [x] `device/partition.rs` -- `DevicePartitionTemplate` (flag → scatter to buf A / buf B) + `DeviceSelectUniqueTemplate` (head-flag → scan → gather); CPU references; 11 unit tests
- [x] `device/decoupled_scan.rs` -- `DecoupledScanTemplate`; single-kernel decoupled-lookback scan with X/A/P descriptors, `membar.gl` + `atom.global.exch.b32` publish, serial lookback loop; inclusive + exclusive; CPU reference; 8 unit tests

#### Sort primitives (`sort/`, 1,102 SLoC)
- [x] `sort/radix_sort.rs` (524 SLoC) -- `RadixSortConfig`, `RadixSortTemplate`; 4-bit LSD radix sort; 8 passes for u32 keys, 16 for u64; three kernels per pass:
  - **Count** -- private `cnt_hist[16]` in shared memory + `atom.shared.add.u32`
  - **Scan** -- single block × 16 threads doing a sequential column scan for the exclusive prefix
  - **Scatter** -- `block_offs[16]` pre-loaded, `atom.shared.add.u32` for unique output positions, ping-pong between two key buffers between passes
  - 12 unit tests
- [x] `sort/merge_sort.rs` (578 SLoC) -- `MergeSortConfig`, `MergeSortTemplate`; stable bitonic block sort + co-rank merge sort:
  - **Bitonic sort** -- 2-barrier-per-stage correctness pattern (pre-load + pre-write); `selp.{ty}` for type-correct compare-swap
  - **Merge kernel** -- O(log n) co-rank binary search per output element; branch-based output selection
  - All 11 unit tests passing including 142-thread blocks and odd-length inputs
- [x] `sort/radix_sort_8bit.rs` -- `RadixSort8Config`/`Template`; 256-bin shared histogram, 4 passes (u32) / 8 (u64); strided init/flush count + 256-thread scan + scatter; CPU `reference_radix8_sort_u32`; 7 unit tests
- [x] `sort/radix_sort_pairs.rs` -- `RadixPairsTemplate` (key+value scatter, `SortOrder::{Ascending,Descending}` digit inversion) + `FloatTwiddleTemplate` (order-preserving f32/f64→unsigned bijection + inverse); CPU references + round-trip tests; 12 unit tests
- [x] `sort/onesweep.rs` -- `OnesweepTemplate`; single-kernel-per-pass decoupled-lookback over per-(block,digit) aggregates, `global_base + block_prefix + local_rank` scatter; 4/8-bit; CPU pass + full-sort references; 6 unit tests
- [x] `sort/merge_pairs.rs` -- `MergePairsTemplate`; value-carrying co-rank merge (merge-sort pass + standalone `DeviceMergeKeysValues`); CPU `reference_merge_pairs`; 5 unit tests

#### Host reference (`host_reference.rs`)
- [x] Slice-based CPU references (`reference_reduce`, `reference_scan`, `reference_histogram_modulo`/`_even`, `reference_select`, `reference_reduce_f64`) plus re-exports of every per-module reference; warp/block/cross-block boundary-size coverage; 8 unit tests. No `ndarray` (crate no-extra-deps policy).

### Future Enhancements [ ]

#### P0 -- Pipeline depth and bandwidth (CUB parity)
- [x] Privatized histogram bin contention -- `atom.shared.add.u32` already in place
- [x] Radix sort scatter unique-position guarantee via `block_offs[16]` + atomic
- [x] Decoupled-lookback scan -- single-kernel decoupled-lookback algorithm with X/A/P partition descriptors and a lookback loop (`device/decoupled_scan.rs`); inclusive + exclusive; CPU reference + 8 unit tests
- [x] 8-bit radix per pass -- 256-bin shared histogram, 4 passes for u32 / 8 for u64 (`sort/radix_sort_8bit.rs`); count/scan/scatter trio + CPU reference + 7 unit tests
- [x] Onesweep radix sort -- single-kernel-per-pass global decoupled-lookback over per-(block,digit) aggregates; `global_base + block_prefix + local_rank` offset decomposition (`sort/onesweep.rs`); CPU pass + full-sort reference + 6 unit tests

#### P1 -- Algorithmic coverage
- [x] `DeviceRunLengthEncode` -- head-flag + scan + gather + lengths pipeline (`device/run_length_encode.rs`); CPU reference + 10 unit tests
- [x] `DeviceSegmentedReduce` / `DeviceSegmentedScan` -- one-block-per-segment reduce + one-thread-per-segment scan given offsets (`device/segmented.rs`); CPU references + 10 unit tests
- [x] `DeviceSegmentedSort` / `DeviceSegmentedRadixSort` -- one-block-per-segment bitonic sort (segments ≤ block_size) with max-value tail padding (`sort/segmented_sort.rs`); CPU `reference_segmented_sort_u64`; 5 unit tests. (Segments longer than one block still need the multi-pass merge path -- GPU-gated.)
- [x] `DeviceSelectUnique` -- consecutive-duplicate compaction via head-flag + scan + gather (`device/partition.rs`); CPU reference
- [x] `DevicePartition` -- two-output flag-based partition (match → buf A, non-match → buf B in stable order) (`device/partition.rs`); CPU reference
- [x] `DeviceMergeKeysValues` -- key+value co-rank merge for stable joins (`sort/merge_pairs.rs`); CPU reference + 5 unit tests

#### P1 -- Sort completeness
- [x] Key+value variants for both radix and merge sort (`sort/radix_sort_pairs.rs` value-carrying scatter; `sort/merge_pairs.rs` value-carrying co-rank merge)
- [x] Descending order option -- digit-inverting (`d → 15-d`) count + scatter under `SortOrder::Descending` (`sort/radix_sort_pairs.rs`)
- [x] f32/f64 radix sort via sign-flipping pre-pass -- order-preserving twiddle bijection + exact inverse (`sort/radix_sort_pairs.rs::FloatTwiddleTemplate`); round-trip + ordering tests
- [x] Pair-key sort -- covered by the key+value radix scatter (joint sort of a key channel with a co-moved payload), `sort/radix_sort_pairs.rs`

#### P2 -- Architecture-aware templates (GPU-gated -- require Hopper/Blackwell hardware to verify)
- [ ] sm_90 cluster-launch reductions -- use `cp.async.bulk` to stage block partials directly into a peer block's shared memory (requires GPU hardware)
- [ ] sm_100 / sm_120 thread-block clusters for device-wide scan -- single-cluster scan with `barrier.cluster.sync` (requires GPU hardware)
- [ ] Tensor Memory Accelerator (TMA) descriptors for histogram bulk-loads on Hopper+ (requires GPU hardware)

#### P2 -- Quality of life
- [ ] `PrimitivesHandle::auto()` -- single-call helper that drives `oxicuda_driver::init()`, picks the best device, creates a context + stream, and returns a handle (requires GPU hardware / live driver for device enumeration)
- [x] A "host reference" mode that runs the same algorithm on the CPU for cross-checking GPU output during development -- slice-based (no `ndarray`, per crate SCIRS2/no-extra-deps policy) reduce / scan / histogram / select references plus re-exports of every per-module reference (`host_reference.rs`); 8 unit tests
- [x] Public `workspace_bytes(input_len)` query on every device template -- added to reduce / scan / select / histogram and to every new device + sort config (RLE, segmented, partition, decoupled-scan, onesweep, radix-8bit)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API loader (`Context`, `Stream`, module loading) | Yes (runtime FFI only) |
| oxicuda-memory | Device buffer allocation for partials / scratch / output | Yes |
| oxicuda-launch | Type-safe kernel launch with `Dim3` grid/block | Yes |
| oxicuda-ptx | PTX IR + code emitter (`SmVersion`, `PtxType`, instruction builders) | Yes |
| thiserror | `#[derive(Error)]` for `PrimitivesError` | Yes |
| tracing | Structured logging on PTX generation + launch | Yes |
| approx (dev) | Floating-point comparison in tests | Yes |

## Quality Status

- Warnings: 0
- Tests: 271 unit tests (231 with default features) + 20 doctests (291 total with --all-features, all passing)
- unwrap() calls: 0 (production code; `expect` only inside `#[cfg(test)]` and doc-example doctests)
- clippy: clean (`cargo clippy -p oxicuda-primitives --all-features --all-targets -- -D warnings`)
- ptx_helpers coverage: all `PtxType` variants (including B128, E4M3/E5M2/E2M3/E3M2/E2M1, F16x2, BF16x2, TF32), all `SmVersion` variants (Sm90a, Sm100, Sm120), all `ReduceOp` identities and instruction mnemonics

## Performance Targets

| Algorithm | Sizes | sm_80 (A100) Target |
|-----------|-------|---------------------|
| Device reduce, u32 | 1M, 16M, 256M | ≥ 90% of CUB `DeviceReduce::Sum` bandwidth |
| Device scan (inclusive), u32 | 1M, 16M | ≥ 80% CUB `DeviceScan::InclusiveSum` (will rise to ≥ 90% once decoupled-lookback lands) |
| Stream compaction, u32, density 50% | 16M | ≥ 80% CUB `DeviceSelect::If` |
| Histogram, 256 bins, u8 input | 256M | ≥ 90% CUB `DeviceHistogram::HistogramEven` (privatized shared) |
| Radix sort, u32 keys | 8M, 64M | ≥ 75% CUB `DeviceRadixSort::SortKeys` (will rise once onesweep lands) |
| Radix sort, u64 keys | 8M | ≥ 70% CUB |
| Merge sort, u32 keys | 1M | ≥ 60% CUB `DeviceMergeSort::SortKeys` (bitonic+co-rank is asymptotically slower than CUB's tile-based merge but is stable and simpler) |

## Notes

- All kernels are emitted as PTX strings, never as C++ source; nothing in this crate touches `nvcc`, `libnvrtc`, or CUB headers
- The runtime path is `oxicuda-ptx` → PTX string → `oxicuda-driver::Module::from_ptx` → `cuModuleLoadData` → `Function::from_module` → `oxicuda-launch::Kernel::launch`
- macOS builds compile but every GPU-touching call returns `PrimitivesError::Cuda(CudaError::DriverNotAvailable)` at runtime
- f64 is supported throughout via lo/hi 32-bit register splits because `shfl.sync.*` only operates on 32-bit lanes
- The crate intentionally does not depend on `oxicuda-blas` -- primitives are the building blocks for it, not the other way around

---

## Architecture-Specific Deepening Opportunities

### Ampere (sm_80) / Ada (sm_89) -- GPU-gated (require on-device latency/occupancy measurement to validate)
- [ ] `cp.async.cg` (cache global) for staging input tiles into shared memory in the histogram and scan kernels (requires GPU hardware)
- [ ] 4-stage software pipelining of load / count / scatter in the radix sort scatter kernel to overlap memory and compute (requires GPU hardware)
- [ ] Bank-conflict-free shared layout for a 256-bin `HistogramEven` private histogram needing explicit padding (requires GPU hardware to measure conflict regime)

### Hopper (sm_90) / Blackwell (sm_100, sm_120) -- GPU-gated (require Hopper+/Blackwell silicon)
- [ ] `cp.async.bulk` TMA bulk-load of 1 KiB tiles in the sort kernels (requires GPU hardware)
- [ ] Distributed shared memory across a thread-block cluster for single-cluster device-wide reduce (requires GPU hardware)
- [ ] Sm90a `setmaxnreg.async` to widen the register budget in the scatter kernel for u64 sorts (requires GPU hardware)

---

## Deepening Opportunities

> Items marked `[x]` represent API surface coverage. The items below represent the gap between the present generator-based implementation and the depth of NVIDIA's CUB.

### Test Coverage Gaps
- [x] Property-based test: for every `(ReduceOp, PtxType)` pair the mnemonic encodes the op family and the identity literal is the algebraic unit (`ptx_helpers.rs::every_op_type_pair_has_consistent_mnemonic_and_identity`)
- [x] Round-trip correctness via the CPU reference -- `host_reference::reference_reduce` / `reference_scan` exercised across the warp/block/cross-block boundary sizes `n ∈ {1, 31, 32, 33, 1023, 1024, 1025}` (`host_reference.rs::scan_boundary_sizes`); the 1M and GPU-comparison legs remain GPU-gated
- [ ] Stress test: 10⁶ randomized radix-sort runs with random distribution comparing GPU output (requires GPU hardware; the CPU radix/onesweep references are randomized-tested at smaller n)
- [ ] Histogram density sweep on-device to catch the shared-memory atomic contention regime (requires GPU hardware)

### Implementation Deepening
- [x] Decoupled-lookback scan rewrite -- `device/decoupled_scan.rs`
- [x] Onesweep radix sort rewrite -- `sort/onesweep.rs`
- [ ] Fused select+scan single-kernel `DeviceSelect` -- folding the flag/scan/gather launches into one kernel requires the decoupled-lookback chained-scan to run inline; correctness depends on inter-block ordering that is only observable on hardware (requires GPU hardware)
- [ ] Templated tile size autotune via `oxicuda-autotune` -- per (algorithm × SM × element type × size class) block size / items-per-thread (requires on-device timing -- GPU-gated)
- [x] Documentation: a numbered "CUB → oxicuda-primitives mapping" table -- added to the crate root docs in `lib.rs` (18 rows covering every template)
