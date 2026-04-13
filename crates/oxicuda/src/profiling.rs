//! Profiling and tracing hooks for kernel-level performance analysis.
//!
//! Provides chrome://tracing compatible output for visualizing GPU kernel
//! execution, memory transfers, and synchronization events.
//!
//! # Example
//!
//! ```
//! use oxicuda::profiling::*;
//! use std::collections::HashMap;
//!
//! let session = ProfilingSession::new("my_session");
//! session.begin();
//!
//! {
//!     let _guard = session.kernel("matmul_f32");
//!     // ... kernel would execute here ...
//! }
//!
//! session.end();
//! let trace = session.export_chrome_trace();
//! assert!(trace.contains("traceEvents"));
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Phase of a trace event in chrome://tracing format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracePhase {
    /// Begin event ("B") — marks the start of a duration.
    Begin,
    /// End event ("E") — marks the end of a duration.
    End,
    /// Complete event ("X") — self-contained duration.
    Complete,
    /// Instant event ("i") — a single point in time.
    Instant,
    /// Counter event ("C") — tracks a numeric value over time.
    Counter,
}

impl TracePhase {
    /// Returns the chrome://tracing phase character.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Begin => "B",
            Self::End => "E",
            Self::Complete => "X",
            Self::Instant => "i",
            Self::Counter => "C",
        }
    }
}

/// A single trace event compatible with chrome://tracing JSON format.
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Event name (kernel name, memcpy, etc.).
    pub name: String,
    /// Category string ("kernel", "memcpy", "sync", "api").
    pub category: String,
    /// Event phase (Begin/End/Complete/Instant/Counter).
    pub phase: TracePhase,
    /// Microsecond timestamp relative to trace start.
    pub timestamp_us: f64,
    /// Duration in microseconds (for Complete events).
    pub duration_us: Option<f64>,
    /// Thread ID.
    pub tid: u64,
    /// Process ID (device ordinal for GPU events).
    pub pid: u32,
    /// Additional metadata arguments.
    pub args: HashMap<String, serde_json::Value>,
}

impl TraceEvent {
    /// Serialize this event to a JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        obj.insert("name".into(), serde_json::Value::String(self.name.clone()));
        obj.insert(
            "cat".into(),
            serde_json::Value::String(self.category.clone()),
        );
        obj.insert(
            "ph".into(),
            serde_json::Value::String(self.phase.as_str().to_string()),
        );
        obj.insert(
            "ts".into(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(self.timestamp_us)
                    .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
        obj.insert(
            "tid".into(),
            serde_json::Value::Number(serde_json::Number::from(self.tid)),
        );
        obj.insert(
            "pid".into(),
            serde_json::Value::Number(serde_json::Number::from(self.pid)),
        );

        if let Some(dur) = self.duration_us {
            if let Some(n) = serde_json::Number::from_f64(dur) {
                obj.insert("dur".into(), serde_json::Value::Number(n));
            }
        }

        if !self.args.is_empty() {
            obj.insert(
                "args".into(),
                serde_json::Value::Object(
                    self.args
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                ),
            );
        }

        serde_json::Value::Object(obj)
    }
}

/// Thread-safe trace event recorder.
///
/// Records [`TraceEvent`]s into a shared buffer. Safe to clone and use
/// across multiple threads.
#[derive(Debug, Clone)]
pub struct TraceRecorder {
    events: Arc<Mutex<Vec<TraceEvent>>>,
    epoch: Instant,
}

impl TraceRecorder {
    /// Create a new empty recorder.
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            epoch: Instant::now(),
        }
    }

    /// Record a trace event.
    pub fn record(&self, event: TraceEvent) {
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }
    }

    /// Start a span that records Begin now and End on drop.
    pub fn start_span(&self, name: &str, category: &str) -> SpanGuard {
        let ts = self.elapsed_us();
        let tid = current_thread_id();
        let begin = TraceEvent {
            name: name.to_string(),
            category: category.to_string(),
            phase: TracePhase::Begin,
            timestamp_us: ts,
            duration_us: None,
            tid,
            pid: 0,
            args: HashMap::new(),
        };
        self.record(begin);

        SpanGuard {
            recorder: self.clone(),
            name: name.to_string(),
            category: category.to_string(),
            tid,
            pid: 0,
            _start_us: ts,
            args: HashMap::new(),
        }
    }

    /// Record a complete (self-contained) event.
    pub fn record_complete(
        &self,
        name: &str,
        category: &str,
        duration_us: f64,
        args: HashMap<String, serde_json::Value>,
    ) {
        let ts = self.elapsed_us();
        let event = TraceEvent {
            name: name.to_string(),
            category: category.to_string(),
            phase: TracePhase::Complete,
            timestamp_us: ts - duration_us,
            duration_us: Some(duration_us),
            tid: current_thread_id(),
            pid: 0,
            args,
        };
        self.record(event);
    }

    /// Record a counter event with one or more named values.
    pub fn record_counter(&self, name: &str, values: &[(&str, f64)]) {
        let mut args = HashMap::new();
        for &(key, val) in values {
            if let Some(n) = serde_json::Number::from_f64(val) {
                args.insert(key.to_string(), serde_json::Value::Number(n));
            }
        }
        let event = TraceEvent {
            name: name.to_string(),
            category: "counter".to_string(),
            phase: TracePhase::Counter,
            timestamp_us: self.elapsed_us(),
            duration_us: None,
            tid: current_thread_id(),
            pid: 0,
            args,
        };
        self.record(event);
    }

    /// Return a snapshot of all recorded events.
    pub fn events(&self) -> Vec<TraceEvent> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Clear all recorded events.
    pub fn clear(&self) {
        if let Ok(mut events) = self.events.lock() {
            events.clear();
        }
    }

    /// Return the number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Microseconds elapsed since this recorder was created.
    fn elapsed_us(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64() * 1_000_000.0
    }
}

impl Default for TraceRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII span guard that records a Begin event on creation and an End event on drop.
pub struct SpanGuard {
    recorder: TraceRecorder,
    name: String,
    category: String,
    tid: u64,
    pid: u32,
    /// Start timestamp, available for callers to compute duration.
    _start_us: f64,
    args: HashMap<String, serde_json::Value>,
}

impl SpanGuard {
    /// Add metadata to the End event that will be emitted on drop.
    pub fn add_arg(&mut self, key: &str, value: serde_json::Value) {
        self.args.insert(key.to_string(), value);
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        let end = TraceEvent {
            name: self.name.clone(),
            category: self.category.clone(),
            phase: TracePhase::End,
            timestamp_us: self.recorder.elapsed_us(),
            duration_us: None,
            tid: self.tid,
            pid: self.pid,
            args: std::mem::take(&mut self.args),
        };
        self.recorder.record(end);
    }
}

/// Exports trace events to chrome://tracing JSON format.
pub struct ChromeTraceExporter;

impl ChromeTraceExporter {
    /// Export events as a chrome://tracing JSON string.
    pub fn export(events: &[TraceEvent]) -> String {
        let trace_events: Vec<serde_json::Value> = events.iter().map(|e| e.to_json()).collect();
        let obj = serde_json::json!({ "traceEvents": trace_events });
        // serde_json::to_string_pretty on a Value is infallible in practice,
        // but we handle the error path to satisfy no-unwrap policy.
        serde_json::to_string_pretty(&obj).unwrap_or_else(|_| String::from("{\"traceEvents\":[]}"))
    }

    /// Export events to a file at the given path.
    pub fn export_to_file(events: &[TraceEvent], path: &str) -> Result<(), std::io::Error> {
        let json = Self::export(events);
        std::fs::write(path, json)
    }
}

/// High-level kernel profiling helper.
///
/// Records kernel executions, memory transfers, and synchronization
/// events into a [`TraceRecorder`].
pub struct KernelProfiler {
    recorder: TraceRecorder,
}

impl KernelProfiler {
    /// Create a new kernel profiler backed by the given recorder.
    pub fn new(recorder: &TraceRecorder) -> Self {
        Self {
            recorder: recorder.clone(),
        }
    }

    /// Record a completed kernel execution.
    pub fn profile_kernel(
        &self,
        name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        duration_us: f64,
    ) {
        let mut args = HashMap::new();
        args.insert(
            "grid".to_string(),
            serde_json::json!(format!("({},{},{})", grid.0, grid.1, grid.2)),
        );
        args.insert(
            "block".to_string(),
            serde_json::json!(format!("({},{},{})", block.0, block.1, block.2)),
        );
        args.insert("shared_mem".to_string(), serde_json::json!(shared_mem));
        self.recorder
            .record_complete(name, "kernel", duration_us, args);
    }

    /// Record a completed memory copy.
    pub fn profile_memcpy(&self, direction: &str, bytes: usize, duration_us: f64) {
        let mut args = HashMap::new();
        args.insert("direction".to_string(), serde_json::json!(direction));
        args.insert("bytes".to_string(), serde_json::json!(bytes));
        let bandwidth_gbps = if duration_us > 0.0 {
            (bytes as f64) / duration_us / 1000.0
        } else {
            0.0
        };
        args.insert(
            "bandwidth_GB_s".to_string(),
            serde_json::json!(bandwidth_gbps),
        );
        self.recorder
            .record_complete(&format!("memcpy_{direction}"), "memcpy", duration_us, args);
    }

    /// Record a completed stream synchronization.
    pub fn profile_sync(&self, stream_id: u64, duration_us: f64) {
        let mut args = HashMap::new();
        args.insert("stream_id".to_string(), serde_json::json!(stream_id));
        self.recorder
            .record_complete("stream_sync", "sync", duration_us, args);
    }
}

/// Summary statistics from a profiling session.
#[derive(Debug, Clone)]
pub struct ProfileSummary {
    /// Total session duration in microseconds.
    pub total_duration_us: f64,
    /// Number of kernel events.
    pub kernel_count: usize,
    /// Total kernel execution time in microseconds.
    pub kernel_time_us: f64,
    /// Number of memcpy events.
    pub memcpy_count: usize,
    /// Total memcpy time in microseconds.
    pub memcpy_time_us: f64,
    /// Number of sync events.
    pub sync_count: usize,
    /// Total sync time in microseconds.
    pub sync_time_us: f64,
    /// Per-kernel breakdown: (name, total_us, invocation_count).
    pub kernel_breakdown: Vec<(String, f64, usize)>,
    /// Fraction of total time spent in kernels.
    pub compute_utilization: f64,
}

impl ProfileSummary {
    /// Format a human-readable profiling report.
    pub fn format_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Profile Summary ===\n");
        report.push_str(&format!(
            "Total duration: {:.2} us ({:.4} ms)\n",
            self.total_duration_us,
            self.total_duration_us / 1000.0
        ));
        report.push_str(&format!(
            "Kernels: {} calls, {:.2} us total\n",
            self.kernel_count, self.kernel_time_us
        ));
        report.push_str(&format!(
            "Memcpy:  {} calls, {:.2} us total\n",
            self.memcpy_count, self.memcpy_time_us
        ));
        report.push_str(&format!(
            "Sync:    {} calls, {:.2} us total\n",
            self.sync_count, self.sync_time_us
        ));
        report.push_str(&format!(
            "Compute utilization: {:.1}%\n",
            self.compute_utilization * 100.0
        ));

        if !self.kernel_breakdown.is_empty() {
            report.push_str("\nKernel Breakdown:\n");
            for (name, total_us, count) in &self.kernel_breakdown {
                let avg = if *count > 0 {
                    total_us / (*count as f64)
                } else {
                    0.0
                };
                report.push_str(&format!(
                    "  {name}: {count} calls, {total_us:.2} us total, {avg:.2} us avg\n",
                ));
            }
        }

        report
    }
}

/// A profiling session that manages event recording and summary generation.
pub struct ProfilingSession {
    name: String,
    recorder: TraceRecorder,
    start_time: Mutex<Option<f64>>,
}

impl ProfilingSession {
    /// Create a new profiling session with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            recorder: TraceRecorder::new(),
            start_time: Mutex::new(None),
        }
    }

    /// Mark the beginning of the session.
    pub fn begin(&self) {
        let ts = self.recorder.elapsed_us();
        if let Ok(mut start) = self.start_time.lock() {
            *start = Some(ts);
        }
        let event = TraceEvent {
            name: format!("session:{}", self.name),
            category: "api".to_string(),
            phase: TracePhase::Begin,
            timestamp_us: ts,
            duration_us: None,
            tid: current_thread_id(),
            pid: 0,
            args: HashMap::new(),
        };
        self.recorder.record(event);
    }

    /// Mark the end of the session.
    pub fn end(&self) {
        let ts = self.recorder.elapsed_us();
        let event = TraceEvent {
            name: format!("session:{}", self.name),
            category: "api".to_string(),
            phase: TracePhase::End,
            timestamp_us: ts,
            duration_us: None,
            tid: current_thread_id(),
            pid: 0,
            args: HashMap::new(),
        };
        self.recorder.record(event);
    }

    /// Start a kernel span (returns RAII guard).
    pub fn kernel(&self, name: &str) -> SpanGuard {
        self.recorder.start_span(name, "kernel")
    }

    /// Start a memcpy span (returns RAII guard).
    pub fn memcpy(&self, direction: &str, bytes: usize) -> SpanGuard {
        let mut guard = self
            .recorder
            .start_span(&format!("memcpy_{direction}"), "memcpy");
        guard.add_arg("direction", serde_json::json!(direction));
        guard.add_arg("bytes", serde_json::json!(bytes));
        guard
    }

    /// Export the recorded events as a chrome://tracing JSON string.
    pub fn export_chrome_trace(&self) -> String {
        ChromeTraceExporter::export(&self.recorder.events())
    }

    /// Compute a summary of the profiling session.
    pub fn summary(&self) -> ProfileSummary {
        let events = self.recorder.events();
        compute_summary(&events, &self.start_time)
    }

    /// Access the underlying recorder.
    pub fn recorder(&self) -> &TraceRecorder {
        &self.recorder
    }
}

/// Compute [`ProfileSummary`] from a list of events.
fn compute_summary(events: &[TraceEvent], start_time: &Mutex<Option<f64>>) -> ProfileSummary {
    let mut kernel_count = 0usize;
    let mut kernel_time = 0.0f64;
    let mut memcpy_count = 0usize;
    let mut memcpy_time = 0.0f64;
    let mut sync_count = 0usize;
    let mut sync_time = 0.0f64;
    let mut kernel_map: HashMap<String, (f64, usize)> = HashMap::new();

    // Find total duration from session begin/end or from first/last event timestamps
    let session_start = start_time
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .unwrap_or(0.0);
    let session_end = events
        .iter()
        .map(|e| e.timestamp_us + e.duration_us.unwrap_or(0.0))
        .fold(0.0f64, f64::max);
    let total_duration = if session_end > session_start {
        session_end - session_start
    } else {
        0.0
    };

    for event in events {
        if event.phase != TracePhase::Complete {
            continue;
        }
        let dur = event.duration_us.unwrap_or(0.0);
        match event.category.as_str() {
            "kernel" => {
                kernel_count += 1;
                kernel_time += dur;
                let entry = kernel_map.entry(event.name.clone()).or_insert((0.0, 0));
                entry.0 += dur;
                entry.1 += 1;
            }
            "memcpy" => {
                memcpy_count += 1;
                memcpy_time += dur;
            }
            "sync" => {
                sync_count += 1;
                sync_time += dur;
            }
            _ => {}
        }
    }

    let mut kernel_breakdown: Vec<(String, f64, usize)> = kernel_map
        .into_iter()
        .map(|(name, (total, count))| (name, total, count))
        .collect();
    kernel_breakdown.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let compute_utilization = if total_duration > 0.0 {
        kernel_time / total_duration
    } else {
        0.0
    };

    ProfileSummary {
        total_duration_us: total_duration,
        kernel_count,
        kernel_time_us: kernel_time,
        memcpy_count,
        memcpy_time_us: memcpy_time,
        sync_count,
        sync_time_us: sync_time,
        kernel_breakdown,
        compute_utilization,
    }
}

/// Opt-in global profiler using static storage.
pub struct GlobalProfiler;

static GLOBAL_ENABLED: AtomicBool = AtomicBool::new(false);
static GLOBAL_RECORDER: OnceLock<TraceRecorder> = OnceLock::new();

impl GlobalProfiler {
    /// Enable the global profiler.
    pub fn enable() {
        GLOBAL_ENABLED.store(true, Ordering::SeqCst);
    }

    /// Disable the global profiler.
    pub fn disable() {
        GLOBAL_ENABLED.store(false, Ordering::SeqCst);
    }

    /// Check whether the global profiler is enabled.
    pub fn is_enabled() -> bool {
        GLOBAL_ENABLED.load(Ordering::SeqCst)
    }

    /// Access the global trace recorder (created on first call).
    pub fn global_recorder() -> &'static TraceRecorder {
        GLOBAL_RECORDER.get_or_init(TraceRecorder::new)
    }
}

/// Get the current thread's numeric ID.
///
/// Extracts a numeric ID from the debug representation of `ThreadId`
/// since `ThreadId::as_u64()` is not yet stable.
fn current_thread_id() -> u64 {
    let id = std::thread::current().id();
    let debug_str = format!("{id:?}");
    // ThreadId debug format is "ThreadId(N)"
    debug_str
        .trim_start_matches("ThreadId(")
        .trim_end_matches(')')
        .parse::<u64>()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_event_json_serialization() {
        let event = TraceEvent {
            name: "kernel_add".to_string(),
            category: "kernel".to_string(),
            phase: TracePhase::Complete,
            timestamp_us: 100.0,
            duration_us: Some(50.0),
            tid: 1,
            pid: 0,
            args: HashMap::new(),
        };
        let json = event.to_json();
        assert_eq!(json["name"], "kernel_add");
        assert_eq!(json["cat"], "kernel");
        assert_eq!(json["ph"], "X");
        assert_eq!(json["ts"], 100.0);
        assert_eq!(json["dur"], 50.0);
    }

    #[test]
    fn test_trace_event_with_args() {
        let mut args = HashMap::new();
        args.insert("grid".to_string(), serde_json::json!("(256,1,1)"));
        args.insert("shared_mem".to_string(), serde_json::json!(4096));
        let event = TraceEvent {
            name: "matmul".to_string(),
            category: "kernel".to_string(),
            phase: TracePhase::Complete,
            timestamp_us: 200.0,
            duration_us: Some(1500.0),
            tid: 1,
            pid: 0,
            args,
        };
        let json = event.to_json();
        assert!(json["args"]["grid"].as_str().is_some());
        assert_eq!(json["args"]["shared_mem"], 4096);
    }

    #[test]
    fn test_trace_phase_as_str() {
        assert_eq!(TracePhase::Begin.as_str(), "B");
        assert_eq!(TracePhase::End.as_str(), "E");
        assert_eq!(TracePhase::Complete.as_str(), "X");
        assert_eq!(TracePhase::Instant.as_str(), "i");
        assert_eq!(TracePhase::Counter.as_str(), "C");
    }

    #[test]
    fn test_recorder_event_recording() {
        let recorder = TraceRecorder::new();
        assert_eq!(recorder.event_count(), 0);

        recorder.record(TraceEvent {
            name: "test".to_string(),
            category: "api".to_string(),
            phase: TracePhase::Instant,
            timestamp_us: 0.0,
            duration_us: None,
            tid: 1,
            pid: 0,
            args: HashMap::new(),
        });
        assert_eq!(recorder.event_count(), 1);

        let events = recorder.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "test");
    }

    #[test]
    fn test_recorder_clear() {
        let recorder = TraceRecorder::new();
        recorder.record(TraceEvent {
            name: "a".to_string(),
            category: "api".to_string(),
            phase: TracePhase::Instant,
            timestamp_us: 0.0,
            duration_us: None,
            tid: 1,
            pid: 0,
            args: HashMap::new(),
        });
        assert_eq!(recorder.event_count(), 1);
        recorder.clear();
        assert_eq!(recorder.event_count(), 0);
    }

    #[test]
    fn test_span_guard_raii() {
        let recorder = TraceRecorder::new();
        {
            let _guard = recorder.start_span("my_kernel", "kernel");
            // Begin should be recorded immediately
            assert_eq!(recorder.event_count(), 1);
        }
        // End should be recorded on drop
        assert_eq!(recorder.event_count(), 2);
        let events = recorder.events();
        assert_eq!(events[0].phase, TracePhase::Begin);
        assert_eq!(events[0].name, "my_kernel");
        assert_eq!(events[1].phase, TracePhase::End);
        assert_eq!(events[1].name, "my_kernel");
    }

    #[test]
    fn test_span_guard_add_arg() {
        let recorder = TraceRecorder::new();
        {
            let mut guard = recorder.start_span("kern", "kernel");
            guard.add_arg("flops", serde_json::json!(1_000_000));
        }
        let events = recorder.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].args["flops"], 1_000_000);
    }

    #[test]
    fn test_chrome_trace_exporter() {
        let events = vec![TraceEvent {
            name: "k1".to_string(),
            category: "kernel".to_string(),
            phase: TracePhase::Complete,
            timestamp_us: 10.0,
            duration_us: Some(90.0),
            tid: 1,
            pid: 0,
            args: HashMap::new(),
        }];
        let json_str = ChromeTraceExporter::export(&events);
        assert!(json_str.contains("traceEvents"));
        assert!(json_str.contains("\"k1\""));
        assert!(json_str.contains("\"X\""));
    }

    #[test]
    fn test_chrome_trace_export_to_file() {
        let events = vec![TraceEvent {
            name: "file_test".to_string(),
            category: "api".to_string(),
            phase: TracePhase::Instant,
            timestamp_us: 0.0,
            duration_us: None,
            tid: 1,
            pid: 0,
            args: HashMap::new(),
        }];
        let path = std::env::temp_dir().join("oxicuda_trace_test.json");
        let path_str = path.to_string_lossy().to_string();
        let result = ChromeTraceExporter::export_to_file(&events, &path_str);
        assert!(result.is_ok());

        let contents = std::fs::read_to_string(&path).expect("read temp file");
        assert!(contents.contains("file_test"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_kernel_profiler() {
        let recorder = TraceRecorder::new();
        let profiler = KernelProfiler::new(&recorder);

        profiler.profile_kernel("gemm_f32", (128, 1, 1), (256, 1, 1), 8192, 450.0);
        profiler.profile_memcpy("HtoD", 4096, 12.5);
        profiler.profile_sync(0, 5.0);

        assert_eq!(recorder.event_count(), 3);
        let events = recorder.events();
        assert_eq!(events[0].category, "kernel");
        assert_eq!(events[1].category, "memcpy");
        assert_eq!(events[2].category, "sync");
    }

    #[test]
    fn test_profiling_session_workflow() {
        let session = ProfilingSession::new("test_session");
        session.begin();

        // Simulate kernel work
        {
            let _g = session.kernel("conv2d");
        }
        {
            let _g = session.memcpy("HtoD", 1024);
        }

        session.end();

        let trace = session.export_chrome_trace();
        assert!(trace.contains("traceEvents"));
        assert!(trace.contains("conv2d"));
        assert!(trace.contains("memcpy_HtoD"));
    }

    #[test]
    fn test_profile_summary_calculation() {
        let recorder = TraceRecorder::new();
        let profiler = KernelProfiler::new(&recorder);

        profiler.profile_kernel("matmul", (1, 1, 1), (1, 1, 1), 0, 100.0);
        profiler.profile_kernel("matmul", (1, 1, 1), (1, 1, 1), 0, 200.0);
        profiler.profile_kernel("relu", (1, 1, 1), (1, 1, 1), 0, 50.0);
        profiler.profile_memcpy("HtoD", 1024, 30.0);
        profiler.profile_sync(0, 10.0);

        let start_time = Mutex::new(Some(0.0));
        let summary = compute_summary(&recorder.events(), &start_time);

        assert_eq!(summary.kernel_count, 3);
        assert!((summary.kernel_time_us - 350.0).abs() < 1e-6);
        assert_eq!(summary.memcpy_count, 1);
        assert!((summary.memcpy_time_us - 30.0).abs() < 1e-6);
        assert_eq!(summary.sync_count, 1);
        assert!((summary.sync_time_us - 10.0).abs() < 1e-6);
        assert_eq!(summary.kernel_breakdown.len(), 2);

        let report = summary.format_report();
        assert!(report.contains("Profile Summary"));
        assert!(report.contains("matmul"));
    }

    #[test]
    fn test_global_profiler_enable_disable() {
        // Reset state
        GlobalProfiler::disable();
        assert!(!GlobalProfiler::is_enabled());

        GlobalProfiler::enable();
        assert!(GlobalProfiler::is_enabled());

        let recorder = GlobalProfiler::global_recorder();
        recorder.record(TraceEvent {
            name: "global_event".to_string(),
            category: "api".to_string(),
            phase: TracePhase::Instant,
            timestamp_us: 0.0,
            duration_us: None,
            tid: current_thread_id(),
            pid: 0,
            args: HashMap::new(),
        });
        assert!(recorder.event_count() >= 1);

        GlobalProfiler::disable();
        assert!(!GlobalProfiler::is_enabled());
    }

    #[test]
    fn test_counter_events() {
        let recorder = TraceRecorder::new();
        recorder.record_counter("gpu_memory", &[("used_mb", 512.0), ("free_mb", 1536.0)]);

        let events = recorder.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].phase, TracePhase::Counter);
        assert_eq!(events[0].name, "gpu_memory");
        assert_eq!(events[0].args["used_mb"], 512.0);
        assert_eq!(events[0].args["free_mb"], 1536.0);
    }

    #[test]
    fn test_record_complete() {
        let recorder = TraceRecorder::new();
        let mut args = HashMap::new();
        args.insert("info".to_string(), serde_json::json!("test_val"));
        recorder.record_complete("op", "api", 42.0, args);

        let events = recorder.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].phase, TracePhase::Complete);
        assert_eq!(events[0].duration_us, Some(42.0));
        assert_eq!(events[0].args["info"], "test_val");
    }
}
