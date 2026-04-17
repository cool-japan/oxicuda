//! `jit_logging` — structured JIT compiler diagnostic parsing.
//!
//! This example demonstrates how to use [`JitLog`] and [`JitDiagnostic`] to
//! parse the raw `ptxas` output emitted by the CUDA JIT compiler into
//! structured, machine-readable diagnostics.
//!
//! The example has two sections:
//!
//! 1. **CPU-only (always runs)**: Parse a hardcoded log string that mimics
//!    real `ptxas` output.  No GPU required.
//!
//! 2. **GPU section (skipped when no GPU is present)**: Load a valid PTX
//!    kernel with `JitOptions` and show the register-usage info line
//!    parsed as a structured [`JitDiagnostic`].
//!
//! Run with:
//! ```sh
//! cargo run --example jit_logging -p oxicuda-driver
//! ```
//!
//! Expected output on a machine with a GPU:
//! ```text
//! ── Part 1: Parse simulated ptxas log (no GPU required) ──────────────────────
//! Parsed 5 diagnostic(s):
//!   [error] (my_kernel, line 10) Unknown instruction 'xyz.f32'
//!   [warning] (my_kernel, line 15) Double-precision will be slow on this device
//!   [fatal] <no kernel> Unresolved extern function 'missing_symbol'
//!   [error] <no kernel> syntax error near token ';'
//!   [info] (my_kernel) used 16 registers, 0 bytes smem, 0 bytes cmem[0]
//! Errors:  3
//! Warnings: 1
//!
//! ── Part 2: Real JIT compilation (requires GPU) ───────────────────────────────
//! Device: NVIDIA A100-SXM4-80GB (sm_80)
//! JIT compile succeeded.
//! Info diagnostics (1):
//!   [info] (vector_add) used 8 registers, 0 bytes smem, 0 bytes cmem[0]
//! ```

use std::sync::Arc;

use oxicuda_driver::{Context, Device, JitDiagnostic, JitLog, JitOptions, JitSeverity, Module};

// ── Part 1: CPU-only diagnostic parsing ──────────────────────────────────────

fn demo_parse_hardcoded_log() {
    println!("── Part 1: Parse simulated ptxas log (no GPU required) ──────────────────────");

    // Simulate the two-buffer output that cuModuleLoadDataEx populates.
    let simulated_error_log = concat!(
        "ptxas error   : 'my_kernel', line 10; error   : Unknown instruction 'xyz.f32'\n",
        "ptxas warning : 'my_kernel', line 15; warning : Double-precision will be slow on this device\n",
        "ptxas fatal   : Unresolved extern function 'missing_symbol'\n",
        "ptxas error : syntax error near token ';'\n",
    )
    .to_string();

    let simulated_info_log =
        "ptxas info    : 'my_kernel' used 16 registers, 0 bytes smem, 0 bytes cmem[0]\n"
            .to_string();

    let log = JitLog {
        error: simulated_error_log,
        info: simulated_info_log,
    };

    let diags = log.parse_diagnostics();
    println!("Parsed {} diagnostic(s):", diags.len());

    for d in &diags {
        let kernel_str = d
            .kernel
            .as_deref()
            .map(|k| {
                let mut s = format!("({k}");
                if let Some(ln) = d.line {
                    s.push_str(&format!(", line {ln}"));
                }
                s.push(')');
                s
            })
            .unwrap_or_else(|| "<no kernel>".into());
        println!(
            "  [{severity}] {kernel_str} {msg}",
            severity = d.severity,
            msg = d.message
        );
    }

    let error_count = log.errors().len();
    let warn_count = log.warnings().len();
    println!("Errors:  {error_count}");
    println!("Warnings: {warn_count}");

    // Verify expected structure.
    assert_eq!(diags.len(), 5, "expected 5 diagnostics");
    assert_eq!(error_count, 3, "expected 3 errors (2x error + 1x fatal)");
    assert_eq!(warn_count, 1, "expected 1 warning");

    let first = &diags[0];
    assert_eq!(first.severity, JitSeverity::Error);
    assert_eq!(first.kernel.as_deref(), Some("my_kernel"));
    assert_eq!(first.line, Some(10));
    assert!(
        first.message.contains("Unknown instruction"),
        "message: {}",
        first.message
    );

    println!();
}

// ── Part 2: Real JIT compilation ─────────────────────────────────────────────

/// Build a valid PTX kernel targeting the given compute capability.
fn make_valid_ptx(cc_major: i32, cc_minor: i32) -> String {
    let (ptx_major, ptx_minor) = match (cc_major, cc_minor) {
        (7, 5) => (7, 4),
        (8, 0) | (8, 6) => (7, 5),
        (8, 9) => (8, 0),
        (9, _) => (8, 0),
        (10, _) => (8, 5),
        (12, _) => (8, 7),
        _ => (7, 4),
    };
    let sm_str = format!("sm_{cc_major}{cc_minor}");

    format!(
        r#".version {ptx_major}.{ptx_minor}
.target {sm_str}
.address_size 64

// Simple kernel: C[tid] = A[tid] + B[tid]  (f32)
.visible .entry vector_add(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{{
    .reg .u32  %r<4>;
    .reg .u64  %rd<5>;
    .reg .f32  %f<3>;
    .reg .pred %p0;

    mov.u32     %r0, %tid.x;
    mov.u32     %r1, %ntid.x;
    mov.u32     %r2, %ctaid.x;
    mad.lo.u32  %r0, %r2, %r1, %r0;

    ld.param.u32    %r3, [param_n];
    setp.ge.u32     %p0, %r0, %r3;
    @%p0 bra        $exit;

    ld.param.u64    %rd0, [param_a];
    ld.param.u64    %rd1, [param_b];
    ld.param.u64    %rd2, [param_c];

    cvt.u64.u32     %rd3, %r0;
    shl.b64         %rd3, %rd3, 2;

    add.u64         %rd4, %rd0, %rd3;
    ld.global.f32   %f0, [%rd4];
    add.u64         %rd4, %rd1, %rd3;
    ld.global.f32   %f1, [%rd4];
    add.f32         %f2, %f0, %f1;
    add.u64         %rd4, %rd2, %rd3;
    st.global.f32   [%rd4], %f2;

$exit:
    ret;
}}
"#
    )
}

fn demo_real_jit() {
    println!("── Part 2: Real JIT compilation (requires GPU) ───────────────────────────────");

    // Initialise the driver — skip GPU section on failure.
    if let Err(e) = oxicuda_driver::init() {
        println!("Skipping GPU section: {e}");
        return;
    }

    let dev = match Device::get(0) {
        Ok(d) => d,
        Err(e) => {
            println!("Skipping GPU section (no device): {e}");
            return;
        }
    };

    let name = dev.name().unwrap_or_else(|_| "<unknown>".into());
    let (cc_major, cc_minor) = dev.compute_capability().unwrap_or((7, 5));
    println!("Device: {name} (sm_{cc_major}{cc_minor})");

    let ctx = match Context::new(&dev) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            println!("Skipping GPU section (context failed): {e}");
            return;
        }
    };
    let _ = &ctx; // keep ctx alive

    let ptx = make_valid_ptx(cc_major, cc_minor);

    let opts = JitOptions {
        optimization_level: 4,
        generate_debug_info: false,
        target_from_context: true,
        ..Default::default()
    };

    match Module::from_ptx_with_options(&ptx, &opts) {
        Ok((_module, log)) => {
            println!("JIT compile succeeded.");

            if !log.error.is_empty() {
                println!("Raw error log:\n{}", log.error);
            }

            // Collect info diagnostics into an owned Vec.
            let all_diags = log.parse_diagnostics();
            let info_diags: Vec<&JitDiagnostic> = all_diags
                .iter()
                .filter(|d| d.severity == JitSeverity::Info)
                .collect();

            println!("Info diagnostics ({}):", info_diags.len());
            for d in &info_diags {
                let kernel_str = d
                    .kernel
                    .as_deref()
                    .map(|k| format!("({k}) "))
                    .unwrap_or_default();
                println!("  [info] {kernel_str}{msg}", msg = d.message);
            }

            let warn_diags = log.warnings();
            if !warn_diags.is_empty() {
                println!("Warnings ({}):", warn_diags.len());
                for w in &warn_diags {
                    println!("  [warning] {}", w.message);
                }
            }
        }
        Err(e) => {
            println!("JIT compile failed (expected on some configs): {e}");
        }
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    demo_parse_hardcoded_log();
    demo_real_jit();
}
