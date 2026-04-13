//! Additional tests for the PTX validator — SM compatibility and register
//! pressure checks introduced in the second implementation pass.

use std::fmt::Write as _;

use super::{ValidationError, validate_ptx, validate_ptx_for_target};
use crate::arch::SmVersion;

// ---------------------------------------------------------------------------
// SM compatibility tests
// ---------------------------------------------------------------------------

#[test]
fn test_valid_ptx_passes_validation() {
    let ptx = ".version 7.0\n.target sm_80\n.address_size 64\n\
               .visible .entry test_kernel(.param .u32 %param_n) {\n\
               .reg .u32 %r<4>;\n\
               mov.u32 %r0, %tid.x;\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm80);
    assert!(
        result.is_ok(),
        "expected no errors for valid sm_80 PTX, got: {:?}",
        result.errors
    );
}

#[test]
fn test_cp_async_requires_sm80() {
    // cp.async on sm_75 should trigger SmIncompatibleInstruction
    let ptx = ".version 6.4\n.target sm_75\n.address_size 64\n\
               .visible .entry k() {\n\
               cp.async.ca.shared.global [%r0], [%rd0], 4;\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm75);
    let compat_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| matches!(e, ValidationError::SmIncompatibleInstruction { .. }))
        .collect();
    assert!(
        !compat_errors.is_empty(),
        "expected SmIncompatibleInstruction for cp.async on sm_75"
    );
    // Verify the error names the required SM correctly
    let msg = format!("{}", compat_errors[0]);
    assert!(
        msg.contains("sm_80"),
        "expected required_sm=sm_80, got: {msg}"
    );
}

#[test]
fn test_wgmma_requires_sm90() {
    let ptx = ".version 7.0\n.target sm_80\n.address_size 64\n\
               .visible .entry k() {\n\
               wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {...};\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm80);
    let compat_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| matches!(e, ValidationError::SmIncompatibleInstruction { .. }))
        .collect();
    assert!(
        !compat_errors.is_empty(),
        "expected SmIncompatibleInstruction for wgmma on sm_80"
    );
    let msg = format!("{}", compat_errors[0]);
    assert!(
        msg.contains("sm_90"),
        "expected required_sm=sm_90, got: {msg}"
    );
}

#[test]
fn test_fp8_requires_sm89() {
    // Using .e4m3 type in an sm_80 kernel
    let ptx = ".version 7.0\n.target sm_80\n.address_size 64\n\
               .visible .entry k() {\n\
               cvt.rn.e4m3x2.f32 %r0, %f0, %f1;\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm80);
    let compat_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| matches!(e, ValidationError::SmIncompatibleInstruction { .. }))
        .collect();
    assert!(
        !compat_errors.is_empty(),
        "expected SmIncompatibleInstruction for fp8 e4m3 on sm_80"
    );
    let msg = format!("{}", compat_errors[0]);
    assert!(
        msg.contains("sm_89"),
        "expected required_sm=sm_89, got: {msg}"
    );
}

#[test]
fn test_too_many_registers_error() {
    // Build a PTX with more than 255 distinct registers (%f0..%f259 = 260 distinct)
    let mut ptx_body = String::from(
        ".version 7.0\n.target sm_80\n.address_size 64\n\
         .visible .entry k() {\n\
         .reg .f32 %f<260>;\n",
    );
    for i in 0..260_usize {
        let _ = writeln!(ptx_body, "    mov.f32 %f{i}, 0f00000000;");
    }
    ptx_body.push_str("    ret;\n}\n");

    let result = validate_ptx_for_target(&ptx_body, SmVersion::Sm80);
    assert!(
        result
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::RegisterPressureExceeded { .. })),
        "expected RegisterPressureExceeded for 260 registers, errors: {:?}",
        result.errors
    );
}

#[test]
fn test_shared_memory_over_limit_error() {
    // sm_75 allows 65536 bytes; declare 70000 bytes
    let ptx = ".version 6.4\n.target sm_75\n.address_size 64\n\
               .visible .entry k() {\n\
               .shared .align 4 .b8 smem[70000];\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm75);
    assert!(
        result.has_errors(),
        "expected error for shared memory exceeding sm_75 limit"
    );
    assert!(
        result.errors.iter().any(|e| matches!(
            e,
            ValidationError::InvalidSharedMemSize {
                declared: 70000,
                ..
            }
        )),
        "expected InvalidSharedMemSize error, got: {:?}",
        result.errors
    );
}

#[test]
fn test_wgmma_valid_on_sm90() {
    // wgmma on sm_90 should pass the SM compatibility check
    let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
               .visible .entry k() {\n\
               wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {...};\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm90);
    // Should not have any SM-compatibility errors
    let compat_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| matches!(e, ValidationError::SmIncompatibleInstruction { .. }))
        .collect();
    assert!(
        compat_errors.is_empty(),
        "wgmma should be valid on sm_90, got errors: {compat_errors:?}"
    );
}

#[test]
fn test_empty_ptx_valid() {
    // Empty PTX should report missing directives but no SM/register errors
    let result = validate_ptx("");
    // Should have MissingVersionDirective and MissingTargetDirective errors
    assert!(result.has_errors());
    assert!(
        result
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingVersionDirective))
    );
    assert!(
        result
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingTargetDirective))
    );
    // No SM or register pressure errors (nothing to scan)
    assert!(!result.errors.iter().any(|e| matches!(
        e,
        ValidationError::SmIncompatibleInstruction { .. }
            | ValidationError::RegisterPressureExceeded { .. }
    )));
}

#[test]
fn test_sm_compat_error_display() {
    let err = ValidationError::SmIncompatibleInstruction {
        instruction: "cp.async".to_string(),
        required_sm: "sm_80".to_string(),
        found_sm: "sm_75".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("cp.async"));
    assert!(msg.contains("sm_80"));
    assert!(msg.contains("sm_75"));
}

#[test]
fn test_register_pressure_exceeded_display() {
    let err = ValidationError::RegisterPressureExceeded {
        count: 300,
        max_allowed: 255,
    };
    let msg = format!("{err}");
    assert!(msg.contains("300"));
    assert!(msg.contains("255"));
}

#[test]
fn test_mma_sync_requires_sm75() {
    // mma.sync is available on sm_75+ — verify sm_75 target passes.
    let ptx = ".version 6.4\n.target sm_75\n.address_size 64\n\
               .visible .entry k() {\n\
               mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...};\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm75);
    let compat_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| {
            matches!(e, ValidationError::SmIncompatibleInstruction { instruction, .. }
                if instruction.contains("mma.sync"))
        })
        .collect();
    // sm_75 supports mma.sync, so no error expected
    assert!(
        compat_errors.is_empty(),
        "mma.sync should be valid on sm_75: {compat_errors:?}"
    );
}
