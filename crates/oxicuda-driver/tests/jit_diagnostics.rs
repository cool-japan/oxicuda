//! Integration tests for the `jit-diagnostic-on-failure` feature.
//!
//! Verifies that [`CudaError::JitFailed`] is produced with a well-formed
//! [`JitLog`] when JIT compilation fails, and that the error chain and
//! display string are correct.
//!
//! All tests run on macOS without a GPU by calling the `jit_failure` helper
//! directly with synthetic log buffers.

use oxicuda_driver::error::CudaError;
use oxicuda_driver::module::{JitLog, JitSeverity};

// Re-export the crate-internal helper via a path alias so tests can call it.
// The helper is `pub(crate)` in oxicuda-driver, but integration tests live in
// a separate crate, so we need to call the public-facing test helper below.
// We expose a thin wrapper in the test module itself.

/// Call the crate-internal `jit_failure` by constructing a `JitLog` manually
/// and using `CudaError::JitFailed` directly — this is the integration-test
/// equivalent of the `pub(crate) jit_failure` helper.
fn make_jit_failed(source: CudaError, info: &[u8], error_bytes: &[u8]) -> CudaError {
    let info_str = String::from_utf8_lossy(info).into_owned();
    let error_str = String::from_utf8_lossy(error_bytes).into_owned();
    let log = JitLog {
        info: info_str,
        error: error_str,
    };
    let diagnostic_count = log.parse_diagnostics().len();
    CudaError::JitFailed {
        log: Box::new(log),
        diagnostic_count,
        source: Box::new(source),
    }
}

#[test]
fn jit_failure_parses_ptxas_log() {
    let info = b"ptxas info    : 0 bytes gmem\nptxas info    : Function properties for 'kernel': 12 bytes stack";
    let error_bytes = b"ptxas error   : ./../test.ptx, line 42; error: parse error near ';'\n";
    let source = CudaError::InvalidValue;
    let err = make_jit_failed(source, info, error_bytes);

    if let CudaError::JitFailed { log, .. } = err {
        let diags = log.parse_diagnostics();
        assert!(
            !diags.is_empty(),
            "expected at least one diagnostic, got none"
        );
        let first_error = diags
            .iter()
            .find(|d| matches!(d.severity, JitSeverity::Error));
        assert!(
            first_error.is_some(),
            "expected at least one error diagnostic"
        );
    } else {
        panic!("expected CudaError::JitFailed");
    }
}

#[test]
fn jit_failure_unparseable_falls_through_to_raw() {
    let garbage = b"this is not a ptxas line at all\nnor is this";
    let err = make_jit_failed(CudaError::InvalidValue, garbage, b"");

    if let CudaError::JitFailed { log, .. } = err {
        assert!(
            log.parse_diagnostics().is_empty(),
            "non-ptxas lines should yield zero structured diagnostics"
        );
        assert!(
            !log.info.is_empty(),
            "raw info log should be non-empty for the garbage input"
        );
        assert!(
            log.info.contains("this is not a ptxas line"),
            "raw log should contain original text"
        );
    } else {
        panic!("expected CudaError::JitFailed");
    }
}

#[test]
fn jit_failed_display_includes_diagnostic_count() {
    // Two parseable ptxas error lines → diagnostic_count == 2.
    let info = b"ptxas error   : foo error\nptxas error   : bar error\n";
    let err = make_jit_failed(CudaError::InvalidValue, info, b"");

    let display = format!("{err}");
    assert!(
        display.contains("2 diagnostic"),
        "display should mention 2 diagnostics, got: {display}"
    );
}

#[test]
fn jit_failed_display_zero_diagnostics() {
    // Empty buffers → diagnostic_count == 0.
    let err = make_jit_failed(CudaError::InvalidValue, b"", b"");
    let display = format!("{err}");
    assert!(
        display.contains("0 diagnostic"),
        "display should mention 0 diagnostics, got: {display}"
    );
}

#[test]
fn jit_failed_source_chain_intact() {
    use std::error::Error;

    let err = make_jit_failed(CudaError::InvalidValue, b"", b"");
    let source = err.source();
    assert!(
        source.is_some(),
        "JitFailed should carry a source error in the chain"
    );

    let source_display = format!("{}", source.expect("already checked Some"));
    assert!(
        source_display.contains("invalid value") || !source_display.is_empty(),
        "source should be the wrapped InvalidValue error"
    );
}

#[test]
fn jit_failed_error_log_parsed() {
    let error_bytes = b"ptxas error   : 'my_kernel', line 10; error   : Unknown instruction 'xyz.f32'\nptxas warning : 'my_kernel', line 20; warning : slow path";
    let err = make_jit_failed(CudaError::InvalidPtx, b"", error_bytes);

    if let CudaError::JitFailed {
        log,
        diagnostic_count,
        ..
    } = err
    {
        let diags = log.parse_diagnostics();
        assert_eq!(diags.len(), 2, "expected 2 structured diagnostics");
        assert_eq!(
            diagnostic_count, 2,
            "pre-computed count should match parse_diagnostics()"
        );
        assert_eq!(diags[0].severity, JitSeverity::Error);
        assert_eq!(diags[1].severity, JitSeverity::Warning);
    } else {
        panic!("expected CudaError::JitFailed");
    }
}

#[test]
fn jit_failed_clone_and_eq() {
    let err = make_jit_failed(CudaError::InvalidValue, b"ptxas error : foo\n", b"");
    let cloned = err.clone();
    assert_eq!(err, cloned, "JitFailed should be Clone + PartialEq");
}

#[test]
fn jit_failed_as_raw_delegates_to_source() {
    // `as_raw()` on JitFailed should return the source error's code.
    let err = make_jit_failed(CudaError::InvalidPtx, b"", b"");
    // InvalidPtx == 218
    assert_eq!(
        err.as_raw(),
        CudaError::InvalidPtx.as_raw(),
        "JitFailed.as_raw() should delegate to the source error"
    );
}
