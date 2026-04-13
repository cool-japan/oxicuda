# oxicuda-autotune

Automatic GPU kernel parameter optimization engine with measurement-based benchmarking and persistent result caching.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-autotune` provides a measurement-based autotuning engine for GPU kernel
parameters. Given a search space of candidate configurations (tile sizes,
pipeline depths, thread counts, etc.), the engine benchmarks each viable
configuration via CUDA events and persists the best result for future lookups.

The tuning workflow follows a define-prune-benchmark-persist-dispatch pipeline.
Search spaces are pruned against hardware limits (shared memory, register count)
before benchmarking, eliminating infeasible configurations early. At runtime, a
three-tier dispatcher selects the optimal configuration: exact match from the
result database, nearest-neighbor interpolation for unseen problem sizes, or a
sensible default fallback.

## Architecture

| Type                | Purpose                                              |
|---------------------|------------------------------------------------------|
| `TunableKernel`     | Trait for kernels that expose a tunable search space  |
| `SearchSpace`       | Defines candidate values per tuning dimension         |
| `SearchSpaceBuilder`| Builder API for constructing custom search spaces     |
| `Config`            | One point in the search space (tile sizes, threads, pipeline depth) |
| `BenchmarkEngine`   | Measures kernel execution time via CUDA events        |
| `BenchmarkConfig`   | Warmup count, iteration count, and timing parameters  |
| `BenchmarkResult`   | Timing statistics (median, min, max, GFLOPS)          |
| `ResultDb`          | Persistent JSON-backed storage keyed by (GPU, kernel, problem) |
| `Dispatcher`        | Runtime config selection with 3-tier fallback         |
| `DispatchTier`      | Indicates which fallback tier was used (Cached, Measured, Default) |

### Dispatch Tiers

1. **Cached** -- exact match found in `ResultDb` for this GPU + kernel + problem size
2. **Measured** -- nearest-neighbor interpolation from neighboring problem sizes
3. **Default** -- conservative default configuration as a last resort

## Quick Start

```rust,no_run
use oxicuda_autotune::prelude::*;

fn example() -> Result<(), AutotuneError> {
    // 1. Define and prune the search space
    let space = SearchSpace::gemm_default();
    let configs = space.prune(48 * 1024, 255, 4); // 48 KiB smem, 255 regs, f32

    // 2. At runtime, select the best config
    let dispatcher = Dispatcher::new("NVIDIA RTX 4090".to_string())?;
    let config = dispatcher.select_config("sgemm", "1024x1024x1024");
    println!("Using tile_m={}, tile_n={}", config.tile_m, config.tile_n);
    Ok(())
}
```

## Features

| Feature      | Default | Description                                    |
|--------------|---------|------------------------------------------------|
| `ptx`        | off     | Enables `oxicuda-ptx` integration for kernel generation during tuning |
| `gpu-tests`  | off     | Enables GPU-dependent integration tests (requires NVIDIA driver) |

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
