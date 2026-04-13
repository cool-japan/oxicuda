//! Graph-based kernel launch capture and replay.
//!
//! CUDA Graphs allow recording a sequence of operations (kernel launches,
//! memory copies, etc.) and replaying them as a single unit with reduced
//! launch overhead. This module provides a lightweight capture facility
//! that records kernel launch configurations for later replay or
//! analysis.
//!
//! # Example
//!
//! ```rust,no_run
//! # use oxicuda_launch::graph_launch::{GraphLaunchCapture, LaunchRecord};
//! # use oxicuda_launch::{LaunchParams, Dim3};
//! let mut capture = GraphLaunchCapture::begin();
//! // In a real scenario you would record actual kernel launches:
//! // capture.record_launch(&kernel, &params);
//! let records = capture.end();
//! println!("captured {} launches", records.len());
//! ```

use crate::kernel::Kernel;
use crate::params::LaunchParams;

// ---------------------------------------------------------------------------
// LaunchRecord
// ---------------------------------------------------------------------------

/// A recorded kernel launch operation.
///
/// Captures the kernel name and launch parameters at the time of
/// recording. This is a lightweight snapshot that does not retain
/// references to the kernel or its module.
#[derive(Debug, Clone)]
pub struct LaunchRecord {
    /// The name of the kernel function that was recorded.
    kernel_name: String,
    /// The launch configuration (grid, block, shared memory).
    params: LaunchParams,
}

impl LaunchRecord {
    /// Returns the kernel function name.
    #[inline]
    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    /// Returns the recorded launch parameters.
    #[inline]
    pub fn params(&self) -> &LaunchParams {
        &self.params
    }
}

impl std::fmt::Display for LaunchRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LaunchRecord(kernel={}, grid={}x{}x{}, block={}x{}x{})",
            self.kernel_name,
            self.params.grid.x,
            self.params.grid.y,
            self.params.grid.z,
            self.params.block.x,
            self.params.block.y,
            self.params.block.z,
        )
    }
}

// ---------------------------------------------------------------------------
// GraphLaunchCapture
// ---------------------------------------------------------------------------

/// Captures a sequence of kernel launches for graph-based replay.
///
/// Create with [`begin`](Self::begin), record launches with
/// [`record_launch`](Self::record_launch), and finalise with
/// [`end`](Self::end) to obtain the list of recorded operations.
///
/// This is a host-side recording facility. On systems with GPU support,
/// the recorded operations can be converted into a CUDA graph for
/// optimised replay via the `oxicuda-driver` graph API.
#[derive(Debug)]
pub struct GraphLaunchCapture {
    /// The sequence of recorded kernel launches.
    stream_nodes: Vec<LaunchRecord>,
    /// Whether the capture is currently active.
    active: bool,
}

impl GraphLaunchCapture {
    /// Begins a new graph launch capture session.
    ///
    /// Returns a capture object in the active state. Record launches
    /// using [`record_launch`](Self::record_launch) and finalise
    /// with [`end`](Self::end).
    pub fn begin() -> Self {
        Self {
            stream_nodes: Vec::new(),
            active: true,
        }
    }

    /// Records a kernel launch into the capture sequence.
    ///
    /// The kernel name and launch parameters are snapshot at the time
    /// of recording. If the capture is not active (i.e., [`end`](Self::end)
    /// has already been called), this method is a no-op.
    pub fn record_launch(&mut self, kernel: &Kernel, params: &LaunchParams) {
        if !self.active {
            return;
        }
        self.stream_nodes.push(LaunchRecord {
            kernel_name: kernel.name().to_owned(),
            params: *params,
        });
    }

    /// Records a kernel launch directly from a kernel name and params.
    ///
    /// This is a lower-level variant of [`record_launch`](Self::record_launch)
    /// that does not require a [`Kernel`] handle — useful for testing and
    /// for replaying recorded descriptions.
    pub fn record_raw(&mut self, kernel_name: impl Into<String>, params: LaunchParams) {
        if !self.active {
            return;
        }
        self.stream_nodes.push(LaunchRecord {
            kernel_name: kernel_name.into(),
            params,
        });
    }

    /// Ends the capture session and returns all recorded launches.
    ///
    /// After calling this method, the capture is no longer active and
    /// further calls to [`record_launch`](Self::record_launch) are
    /// ignored.
    pub fn end(mut self) -> Vec<LaunchRecord> {
        self.active = false;
        std::mem::take(&mut self.stream_nodes)
    }

    /// Returns the number of launches recorded so far.
    #[inline]
    pub fn len(&self) -> usize {
        self.stream_nodes.len()
    }

    /// Returns `true` if no launches have been recorded.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.stream_nodes.is_empty()
    }

    /// Returns `true` if the capture is currently active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Dim3;

    #[test]
    fn capture_begin_is_active() {
        let capture = GraphLaunchCapture::begin();
        assert!(capture.is_active());
        assert!(capture.is_empty());
        assert_eq!(capture.len(), 0);
    }

    #[test]
    fn capture_end_returns_empty_vec() {
        let capture = GraphLaunchCapture::begin();
        let records = capture.end();
        assert!(records.is_empty());
    }

    #[test]
    fn launch_record_display() {
        let record = LaunchRecord {
            kernel_name: "vector_add".to_owned(),
            params: LaunchParams::new(Dim3::x(4), Dim3::x(256)),
        };
        let s = format!("{record}");
        assert!(s.contains("vector_add"));
        assert!(s.contains("4x1x1"));
        assert!(s.contains("256x1x1"));
    }

    #[test]
    fn launch_record_accessors() {
        let record = LaunchRecord {
            kernel_name: "my_kernel".to_owned(),
            params: LaunchParams::new(8u32, 128u32),
        };
        assert_eq!(record.kernel_name(), "my_kernel");
        assert_eq!(record.params().grid.x, 8);
        assert_eq!(record.params().block.x, 128);
    }

    #[test]
    fn capture_debug() {
        let capture = GraphLaunchCapture::begin();
        let dbg = format!("{capture:?}");
        assert!(dbg.contains("GraphLaunchCapture"));
        assert!(dbg.contains("active: true"));
    }

    #[test]
    fn launch_record_clone() {
        let record = LaunchRecord {
            kernel_name: "clone_test".to_owned(),
            params: LaunchParams::new(2u32, 64u32),
        };
        let cloned = record.clone();
        assert_eq!(cloned.kernel_name(), record.kernel_name());
        assert_eq!(cloned.params().grid.x, record.params().grid.x);
    }

    // ---------------------------------------------------------------------------
    // Quality gate tests (CPU-only)
    // ---------------------------------------------------------------------------

    #[test]
    fn graph_capture_records_launches() {
        // GraphLaunchCapture::begin() creates an empty capture; after pushing one
        // record via record_raw, len() == 1.
        let mut capture = GraphLaunchCapture::begin();
        assert_eq!(capture.len(), 0);
        assert!(capture.is_empty());

        capture.record_raw("vector_add", LaunchParams::new(Dim3::x(4), Dim3::x(256)));
        assert_eq!(capture.len(), 1);
        assert!(!capture.is_empty());
    }

    #[test]
    fn graph_record_contains_params() {
        // A LaunchRecord stores grid/block dims accurately.
        let params = LaunchParams::new(Dim3::new(8, 2, 1), Dim3::new(32, 8, 1));
        let record = LaunchRecord {
            kernel_name: "my_kernel".to_owned(),
            params,
        };
        assert_eq!(record.params().grid.x, 8);
        assert_eq!(record.params().grid.y, 2);
        assert_eq!(record.params().grid.z, 1);
        assert_eq!(record.params().block.x, 32);
        assert_eq!(record.params().block.y, 8);
        assert_eq!(record.params().block.z, 1);
    }

    #[test]
    fn graph_replay_count() {
        // After recording 3 launches via record_raw, the records vector has length 3.
        let mut capture = GraphLaunchCapture::begin();
        let params = LaunchParams::new(Dim3::x(4), Dim3::x(128));
        capture.record_raw("kernel_a", params);
        capture.record_raw("kernel_b", params);
        capture.record_raw("kernel_c", params);

        assert_eq!(capture.len(), 3);

        // end() consumes the capture and returns all records
        let records = capture.end();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].kernel_name(), "kernel_a");
        assert_eq!(records[1].kernel_name(), "kernel_b");
        assert_eq!(records[2].kernel_name(), "kernel_c");
    }
}
