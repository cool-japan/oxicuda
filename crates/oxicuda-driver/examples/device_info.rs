//! `device_info` — enumerate GPUs and print their capabilities.
//!
//! Run with:
//! ```sh
//! cargo run --example device_info -p oxicuda-driver
//! ```
//!
//! Output on a machine with one A100 might look like:
//! ```text
//! OxiCUDA device info (1 device found)
//! ────────────────────────────────────────
//! [0] NVIDIA A100-SXM4-80GB
//!     Compute capability : 8.0  (sm_80)
//!     Multiprocessors    : 108
//!     Max threads/block  : 1024
//!     Global memory      : 81920 MiB
//!     Driver version     : 12040
//! ```

use oxicuda_driver::Device;

fn main() {
    // ── 1. Load the CUDA driver library ──────────────────────────────────────
    if let Err(e) = oxicuda_driver::init() {
        eprintln!("Failed to load CUDA driver: {e}");
        eprintln!("Is an NVIDIA GPU driver installed?");
        std::process::exit(1);
    }

    // ── 2. Enumerate devices ──────────────────────────────────────────────────
    let count = match Device::count() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("cuDeviceGetCount failed: {e}");
            std::process::exit(1);
        }
    };

    if count == 0 {
        println!("No CUDA-capable GPUs found.");
        return;
    }

    println!(
        "OxiCUDA device info ({count} device{} found)",
        if count == 1 { "" } else { "s" }
    );
    println!("{}", "─".repeat(40));

    for idx in 0..count {
        let dev = match Device::get(idx) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[{idx}] Failed to get device: {e}");
                continue;
            }
        };

        let name = dev.name().unwrap_or_else(|_| "<unknown>".into());
        let (cc_major, cc_minor) = dev.compute_capability().unwrap_or((0, 0));

        // Derive the PTX sm_ target string from compute capability.
        // sm_90a is a special case — all others follow "sm_<major><minor>" convention.
        let sm_str = compute_capability_to_sm_str(cc_major, cc_minor);

        let mp_count = dev.multiprocessor_count().unwrap_or(0);
        let max_threads = dev.max_threads_per_block().unwrap_or(0);
        let total_mem_mib = dev.total_memory().map(|b| b / (1024 * 1024)).unwrap_or(0);
        let driver_ver = oxicuda_driver::driver_version().unwrap_or(0);

        println!(
            "[{idx}] {name}\n\
             \x20    Compute capability : {cc_major}.{cc_minor}  ({sm_str})\n\
             \x20    Multiprocessors    : {mp_count}\n\
             \x20    Max threads/block  : {max_threads}\n\
             \x20    Global memory      : {total_mem_mib} MiB\n\
             \x20    Driver version     : {driver_ver}",
        );
    }
}

/// Convert a `(major, minor)` compute capability to a PTX SM target string.
///
/// Handles the special `sm_90a` case; for everything else follows the
/// `sm_<major><minor>` convention (e.g. `(8, 0)` → `"sm_80"`).
fn compute_capability_to_sm_str(major: i32, minor: i32) -> String {
    match (major, minor) {
        (9, 0) => "sm_90".to_string(), // standard Hopper
        _ => format!("sm_{major}{minor}"),
    }
}
