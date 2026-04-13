//! Autotune Demo
//!
//! Demonstrates the OxiCUDA autotuner (`oxicuda_autotune`) without requiring
//! a GPU.  The demo runs entirely on the CPU, simulating benchmark timing with
//! a synthetic performance model.
//!
//! Steps covered:
//!
//! 1. **SearchSpace construction** — build a custom GEMM search space with
//!    `SearchSpaceBuilder`.
//! 2. **Pruning** — eliminate architecturally infeasible configurations.
//! 3. **Simulated benchmarking** — score each configuration using a CPU
//!    heuristic (larger tiles = faster, up to a shared-memory limit).
//! 4. **TemplateAutotuner** — record results and find the best config.
//! 5. **ResultDb persistence** — save the best result to a temp-file database
//!    and reload it to verify round-trip correctness.
//! 6. **Dispatcher 3-tier fallback** — demonstrate exact-match, nearest-
//!    neighbor, and default-fallback tiers.
//! 7. **TunableKernel trait** — show how to implement the trait for a custom
//!    kernel type.
//!
//! # Running (no GPU needed)
//!
//! ```bash
//! cargo run --example autotune_demo -p oxicuda --features "autotune"
//! ```

use oxicuda_autotune::{
    BenchmarkResult, Config, DispatchTier, Dispatcher, ResultDb, SearchSpace, SearchSpaceBuilder,
    TemplateAutotuner, TunableKernel,
    ptx_integration::{elementwise_search_space, gemm_search_space, reduction_search_space},
};

// ─── Step 7: TunableKernel implementation ───────────────────────────────────

/// A GEMM problem descriptor (row, column, inner dimensions).
struct GemmProblem {
    m: u32,
    n: u32,
    k: u32,
}

/// A minimal GEMM kernel that implements [`TunableKernel`].
///
/// This is a CPU-side descriptor — it does not launch any GPU code.
/// The autotune engine uses it to generate database keys, estimate
/// resource requirements, and validate configurations.
struct SgemmKernel;

impl TunableKernel for SgemmKernel {
    type Problem = GemmProblem;

    fn problem_key(&self, p: &GemmProblem) -> String {
        format!("{}x{}x{}", p.m, p.n, p.k)
    }

    fn kernel_name(&self) -> &str {
        "sgemm"
    }

    fn compute_flops(&self, p: &GemmProblem) -> f64 {
        2.0 * f64::from(p.m) * f64::from(p.n) * f64::from(p.k)
    }

    fn shared_mem_bytes(&self, config: &Config) -> u32 {
        // Double-buffered A + B tiles in f32.
        config.estimated_shared_mem(4) as u32
    }
}

// ─── Synthetic benchmark model ───────────────────────────────────────────────

/// Simulate a benchmark result for a configuration using a heuristic model.
///
/// Performance model (synthetic, CPU-side only):
/// - Larger tile M × N → lower latency (up to a sweet spot).
/// - More pipeline stages → lower latency (diminishing returns).
/// - Tensor Core reduces latency by ~30 %.
/// - Oversized shared memory → penalty.
///
/// Returns a (median_us, gflops) pair.
fn simulate_benchmark(config: &Config, m: u32, n: u32, k: u32) -> (f64, f64) {
    let tile_area = f64::from(config.tile_m * config.tile_n);
    let sweet_spot = 128.0 * 128.0_f64;
    let tile_score = (tile_area / sweet_spot).min(1.0);

    let stage_bonus = 1.0 - 0.05 * f64::from(config.stages - 1).min(3.0);
    let tc_bonus = if config.use_tensor_core { 0.70 } else { 1.0 };

    // Rough latency in microseconds.
    let base_us = 1000.0;
    let median_us = base_us * (1.0 - tile_score * 0.5) * stage_bonus * tc_bonus;

    let flops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
    let gflops = flops / (median_us * 1e-6) / 1e9;

    (median_us, gflops)
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiCUDA Autotune Demo ===\n");

    // ── Step 1: Custom SearchSpace via builder ──────────────────────────────
    println!("Step 1: Building custom GEMM search space");
    let space = SearchSpaceBuilder::new()
        .tile_m(vec![32, 64, 128])
        .tile_n(vec![32, 64, 128])
        .tile_k(vec![16, 32])
        .warp_m(vec![32, 64])
        .warp_n(vec![32, 64])
        .stages(vec![2, 3])
        .use_tensor_core(vec![false, true])
        .block_size(vec![128, 256])
        .build();

    println!(
        "  Total configurations (Cartesian product): {}",
        space.total_configs()
    );

    // Default space for comparison.
    let default_space = SearchSpace::gemm_default();
    println!(
        "  Default GEMM space total configs: {}",
        default_space.total_configs()
    );

    // ── Step 2: Prune for architecture limits ───────────────────────────────
    println!("\nStep 2: Pruning for sm_80 architecture limits");
    let max_smem: usize = 48 * 1024; // 48 KiB shared memory per block
    let max_regs: u32 = 255; // maximum registers per thread
    let viable = space.prune(max_smem, max_regs, 4 /* f32 = 4 bytes */);
    println!(
        "  Surviving configurations: {} / {}",
        viable.len(),
        space.total_configs()
    );

    // ── Step 3: Simulated benchmarking via TemplateAutotuner ───────────────
    println!(
        "\nStep 3: Simulated benchmarking ({} configs)",
        viable.len()
    );
    let m: u32 = 1024;
    let n: u32 = 1024;
    let k: u32 = 1024;

    let mut autotuner = TemplateAutotuner::new(space.clone());
    for config in &viable {
        let (_median_us, gflops) = simulate_benchmark(config, m, n, k);
        autotuner.record_result(config.clone(), gflops);
    }
    println!("  Recorded {} results.", autotuner.num_results());

    let (best_cfg, best_gflops) = autotuner.best_config().ok_or("no results recorded")?;
    println!(
        "  Best config: tile_m={}, tile_n={}, tile_k={}, stages={}, tc={}, block={} \
         → {:.1} GFLOPS",
        best_cfg.tile_m,
        best_cfg.tile_n,
        best_cfg.tile_k,
        best_cfg.stages,
        best_cfg.use_tensor_core,
        best_cfg.block_size,
        best_gflops
    );

    let top3 = autotuner.top_n(3);
    println!("  Top-3 configurations:");
    for (i, (cfg, gflops)) in top3.iter().enumerate() {
        println!(
            "    #{}: tile_m={}, tile_n={}, tc={} → {:.1} GFLOPS",
            i + 1,
            cfg.tile_m,
            cfg.tile_n,
            cfg.use_tensor_core,
            gflops
        );
    }

    // ── Step 4: ResultDb persistence ───────────────────────────────────────
    println!("\nStep 4: Persisting best result to ResultDb (temp file)");
    let db_dir = std::env::temp_dir().join("oxicuda_autotune_demo");
    std::fs::create_dir_all(&db_dir)?;
    let db_path = db_dir.join("results.json");

    let (median_us, gflops) = simulate_benchmark(best_cfg, m, n, k);
    let bench_result = BenchmarkResult {
        config: best_cfg.clone(),
        median_us,
        min_us: median_us * 0.95,
        max_us: median_us * 1.05,
        stddev_us: median_us * 0.02,
        gflops: Some(gflops),
        efficiency: None,
    };

    let gpu_name = "Simulated GPU A100";
    let kernel_name = "sgemm";
    let problem_key = format!("{m}x{n}x{k}");

    {
        let mut db = ResultDb::open_at(db_path.clone())?;
        db.save(gpu_name, kernel_name, &problem_key, bench_result)?;
        println!(
            "  Saved result: {} | {} | {} (median={:.1} us)",
            gpu_name, kernel_name, problem_key, median_us
        );

        // Verify round-trip.
        let loaded = db
            .lookup(gpu_name, kernel_name, &problem_key)
            .ok_or("lookup failed after save")?;
        println!("  Verified round-trip: median={:.1} us", loaded.median_us);
        println!("  DB total entries: {}", db.total_entries());
    }

    // Save a second entry to demonstrate nearest-neighbor lookup.
    {
        let mut db = ResultDb::open_at(db_path.clone())?;
        let small_result = BenchmarkResult {
            config: Config::new().with_tile_m(64).with_tile_n(64),
            median_us: 2000.0,
            min_us: 1900.0,
            max_us: 2100.0,
            stddev_us: 50.0,
            gflops: Some(500.0),
            efficiency: None,
        };
        db.save(gpu_name, kernel_name, "512x512x512", small_result)?;
    }

    // ── Step 5: Dispatcher 3-tier fallback ─────────────────────────────────
    println!("\nStep 5: Dispatcher 3-tier fallback");
    let db = ResultDb::open_at(db_path.clone())?;
    let dispatcher = Dispatcher::with_db_and_default(
        db,
        gpu_name.to_string(),
        Config::new().with_tile_m(32).with_tile_n(32), // fallback default
    );

    // Tier 1: exact match.
    let (cfg1, tier1) = dispatcher.select_config_with_tier(kernel_name, &problem_key);
    println!(
        "  Tier 1 (exact '{}')    : {} → tile_m={}, tile_n={}",
        problem_key, tier1, cfg1.tile_m, cfg1.tile_n
    );
    assert_eq!(tier1, DispatchTier::ExactMatch);

    // Tier 2: nearest neighbor (query a size not in the DB).
    let (cfg2, tier2) = dispatcher.select_config_with_tier(kernel_name, "768x768x768");
    println!(
        "  Tier 2 (near '768x768x768'): {} → tile_m={}, tile_n={}",
        tier2, cfg2.tile_m, cfg2.tile_n
    );
    assert_eq!(tier2, DispatchTier::NearestNeighbor);

    // Tier 3: default (unknown kernel).
    let (cfg3, tier3) = dispatcher.select_config_with_tier("unknown_kernel", "1x1x1");
    println!(
        "  Tier 3 (unknown kernel): {} → tile_m={}, tile_n={}",
        tier3, cfg3.tile_m, cfg3.tile_n
    );
    assert_eq!(tier3, DispatchTier::Default);
    assert_eq!(cfg3.tile_m, 32); // our custom default

    // ── Step 6: TunableKernel trait ────────────────────────────────────────
    println!("\nStep 6: TunableKernel trait");
    let kernel = SgemmKernel;
    let problem = GemmProblem { m, n, k };

    println!("  kernel_name : {}", kernel.kernel_name());
    println!("  problem_key : {}", kernel.problem_key(&problem));
    println!(
        "  compute_flops: {:.2e} FLOP",
        kernel.compute_flops(&problem)
    );

    let smem = kernel.shared_mem_bytes(best_cfg);
    println!(
        "  shared_mem (best config): {} bytes ({:.1} KiB)",
        smem,
        smem as f64 / 1024.0
    );

    let is_valid = kernel.is_valid_config(best_cfg, max_smem);
    println!("  is_valid (48 KiB limit): {}", is_valid);

    // ── Step 7: PTX-integration search spaces ──────────────────────────────
    println!("\nStep 7: PTX-integration search space helpers");
    let gemm_space = gemm_search_space(&[64, 128], &[64, 128], &[16, 32], &[2, 3], &[128, 256]);
    println!(
        "  gemm_search_space configs:        {}",
        gemm_space.total_configs()
    );

    let elem_space = elementwise_search_space(&[128, 256], &[4, 8]);
    println!(
        "  elementwise_search_space configs: {}",
        elem_space.total_configs()
    );

    let red_space = reduction_search_space(&[64, 128, 256], &[1, 2, 4]);
    println!(
        "  reduction_search_space configs:   {}",
        red_space.total_configs()
    );

    // ── Cleanup ─────────────────────────────────────────────────────────────
    std::fs::remove_dir_all(&db_dir)?;
    println!("\nDemo complete. Temp files cleaned up.");

    Ok(())
}
