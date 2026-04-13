//! Visualization and charting for autotuning results.
//!
//! Generates text-based charts, ASCII tables, and data files for external
//! plotting tools (gnuplot, spreadsheets).  All rendering is pure Rust
//! with no external plotting dependencies.
//!
//! (C) 2026 COOLJAPAN OU (Team KitaSan)

use std::fmt::Write as FmtWrite;

use crate::error::AutotuneError;
use crate::result_db::ResultDb;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// A single data point in a chart series.
#[derive(Debug, Clone)]
pub struct DataPoint {
    /// X-axis value.
    pub x: f64,
    /// Y-axis value.
    pub y: f64,
    /// Optional annotation label.
    pub label: Option<String>,
    /// Lower error bound (absolute distance below `y`).
    pub error_low: Option<f64>,
    /// Upper error bound (absolute distance above `y`).
    pub error_high: Option<f64>,
}

/// Visual style for a data series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesStyle {
    /// Connected line chart.
    Line,
    /// Horizontal bar chart.
    Bar,
    /// Scatter plot (individual points).
    Scatter,
    /// Frequency histogram.
    Histogram,
}

/// A named series of data points with a rendering style.
#[derive(Debug, Clone)]
pub struct DataSeries {
    /// Series name (used in legends).
    pub name: String,
    /// Ordered data points.
    pub points: Vec<DataPoint>,
    /// Visual style hint.
    pub style: SeriesStyle,
}

/// Complete chart data with title, axis labels, and one or more series.
#[derive(Debug, Clone)]
pub struct ChartData {
    /// Chart title.
    pub title: String,
    /// X-axis label.
    pub x_label: String,
    /// Y-axis label.
    pub y_label: String,
    /// Data series to render.
    pub series: Vec<DataSeries>,
}

// ---------------------------------------------------------------------------
// ASCII chart renderer
// ---------------------------------------------------------------------------

/// Text-based chart renderer producing ASCII / Unicode output.
pub struct AsciiChart {
    /// Chart width in characters.
    width: usize,
    /// Chart height in characters (rows).
    height: usize,
}

/// Unicode block characters for sparkline rendering (8 levels).
const SPARK_BLOCKS: [char; 8] = [
    '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}',
];

impl AsciiChart {
    /// Creates a new chart renderer with the given dimensions.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Returns the configured chart height.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the configured chart width.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Renders a horizontal bar chart from [`ChartData`].
    ///
    /// Each data point in the first series becomes a labeled bar.
    /// The bar length is proportional to the y-value relative to the
    /// maximum y across all points.  The chart is capped to [`height`](Self::height)
    /// rows; excess data points are truncated.
    pub fn render_bar_chart(&self, data: &ChartData) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "{}", data.title);
        let _ = writeln!(out, "{}", "-".repeat(self.width.min(data.title.len() + 4)));

        let all_points: Vec<&DataPoint> = data
            .series
            .iter()
            .flat_map(|s| s.points.iter())
            .take(self.height.saturating_sub(4)) // reserve header/footer rows
            .collect();

        if all_points.is_empty() {
            let _ = writeln!(out, "(no data)");
            return out;
        }

        let max_y = all_points
            .iter()
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max);

        // Determine label width from labels or indices.
        let labels: Vec<String> = all_points
            .iter()
            .enumerate()
            .map(|(i, p)| p.label.clone().unwrap_or_else(|| format!("{i}")))
            .collect();

        let label_width = labels.iter().map(String::len).max().unwrap_or(1);
        let bar_area = self.width.saturating_sub(label_width + 3 + 10);

        for (i, point) in all_points.iter().enumerate() {
            let frac = if max_y > 0.0 { point.y / max_y } else { 0.0 };
            let bar_len = ((frac * bar_area as f64) as usize).min(bar_area);
            let bar = "\u{2588}".repeat(bar_len);
            let _ = writeln!(
                out,
                "{:>lw$} | {:<bw$} {:.1}",
                labels[i],
                bar,
                point.y,
                lw = label_width,
                bw = bar_area,
            );
        }

        let _ = writeln!(out);
        let _ = writeln!(out, "  X: {}", data.x_label);
        let _ = writeln!(out, "  Y: {}", data.y_label);

        out
    }

    /// Renders a frequency histogram from raw values.
    ///
    /// Values are binned into `bins` equal-width buckets.  Each bin
    /// is rendered as a horizontal bar showing frequency count.
    pub fn render_histogram(&self, values: &[f64], bins: usize) -> String {
        let mut out = String::new();

        if values.is_empty() || bins == 0 {
            let _ = writeln!(out, "(no data)");
            return out;
        }

        let min_v = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_v = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max_v - min_v;
        let bin_width = if range > 0.0 {
            range / bins as f64
        } else {
            1.0
        };

        let mut counts = vec![0usize; bins];
        for &v in values {
            let idx = if range > 0.0 {
                ((v - min_v) / bin_width) as usize
            } else {
                0
            };
            let idx = idx.min(bins - 1);
            counts[idx] += 1;
        }

        let max_count = counts.iter().copied().max().unwrap_or(1);
        let label_width = 16;
        let bar_area = self.width.saturating_sub(label_width + 10);

        for (i, &count) in counts.iter().enumerate() {
            let lo = min_v + i as f64 * bin_width;
            let hi = lo + bin_width;
            let frac = if max_count > 0 {
                count as f64 / max_count as f64
            } else {
                0.0
            };
            let bar_len = ((frac * bar_area as f64) as usize).min(bar_area);
            let bar = "\u{2588}".repeat(bar_len);
            let _ = writeln!(
                out,
                "[{:>6.1},{:>6.1}) | {:<bw$} {count}",
                lo,
                hi,
                bar,
                bw = bar_area,
            );
        }

        out
    }

    /// Renders a single-line sparkline using Unicode block characters.
    ///
    /// Each value maps to one of eight block heights (▁▂▃▄▅▆▇█).
    /// Returns an empty string for empty input.
    pub fn render_sparkline(&self, values: &[f64]) -> String {
        if values.is_empty() {
            return String::new();
        }

        let min_v = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_v = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max_v - min_v;

        let mut out = String::with_capacity(values.len() * 3);
        for &v in values {
            let norm = if range > 0.0 {
                (v - min_v) / range
            } else {
                0.5
            };
            let idx = ((norm * 7.0) as usize).min(7);
            out.push(SPARK_BLOCKS[idx]);
        }
        out
    }

    /// Renders an aligned ASCII table.
    ///
    /// Each column is padded to the maximum width of its header or data.
    pub fn render_table(&self, headers: &[&str], rows: &[Vec<String>]) -> String {
        let ncols = headers.len();
        let mut col_widths = vec![0usize; ncols];

        for (i, h) in headers.iter().enumerate() {
            col_widths[i] = col_widths[i].max(h.len());
        }
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                if i < ncols {
                    col_widths[i] = col_widths[i].max(cell.len());
                }
            }
        }

        let mut out = String::new();

        // Header row.
        let _ = write!(out, "|");
        for (h, &w) in headers.iter().zip(col_widths.iter()) {
            let _ = write!(out, " {:>w$} |", h);
        }
        let _ = writeln!(out);

        // Separator.
        let _ = write!(out, "|");
        for &w in &col_widths {
            let _ = write!(out, "-{}-|", "-".repeat(w));
        }
        let _ = writeln!(out);

        // Data rows.
        for row in rows {
            let _ = write!(out, "|");
            for (i, &w) in col_widths.iter().enumerate() {
                let cell = row.get(i).map_or("", String::as_str);
                let _ = write!(out, " {:>w$} |", cell);
            }
            let _ = writeln!(out);
        }

        out
    }
}

impl Default for AsciiChart {
    fn default() -> Self {
        Self::new(80, 24)
    }
}

// ---------------------------------------------------------------------------
// VisualizationBuilder — build ChartData from ResultDb
// ---------------------------------------------------------------------------

/// Builds visualization-ready chart data from a [`ResultDb`].
pub struct VisualizationBuilder<'a> {
    db: &'a ResultDb,
}

impl<'a> VisualizationBuilder<'a> {
    /// Creates a builder referencing the given result database.
    #[must_use]
    pub fn from_db(db: &'a ResultDb) -> Self {
        Self { db }
    }

    /// GFLOPS vs problem size for a specific kernel and GPU.
    ///
    /// Each stored problem key becomes a data point, sorted by the
    /// numeric portion of the key (or alphabetically as fallback).
    pub fn performance_by_problem(
        &self,
        kernel: &str,
        gpu: &str,
    ) -> Result<ChartData, AutotuneError> {
        let entries = self.db.list_gpu(gpu);
        let mut points: Vec<DataPoint> = entries
            .iter()
            .filter(|(k, _, _)| *k == kernel)
            .map(|(_, problem, result)| {
                let x = extract_numeric(problem);
                DataPoint {
                    x,
                    y: result.gflops.unwrap_or(0.0),
                    label: Some(problem.to_string()),
                    error_low: None,
                    error_high: None,
                }
            })
            .collect();

        if points.is_empty() {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "no results for kernel={kernel}, gpu={gpu}"
            )));
        }

        points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        Ok(ChartData {
            title: format!("{kernel} Performance on {gpu}"),
            x_label: "Problem Size".to_string(),
            y_label: "GFLOPS".to_string(),
            series: vec![DataSeries {
                name: kernel.to_string(),
                points,
                style: SeriesStyle::Line,
            }],
        })
    }

    /// GFLOPS vs tile configuration for a specific problem.
    ///
    /// Each result's tile_m * tile_n product becomes the x-axis value.
    pub fn performance_by_tile_size(
        &self,
        kernel: &str,
        gpu: &str,
        problem: &str,
    ) -> Result<ChartData, AutotuneError> {
        let entries = self.db.list_gpu(gpu);
        let mut points: Vec<DataPoint> = entries
            .iter()
            .filter(|(k, p, _)| *k == kernel && *p == problem)
            .map(|(_, _, result)| {
                let tile_area = (result.config.tile_m * result.config.tile_n) as f64;
                DataPoint {
                    x: tile_area,
                    y: result.gflops.unwrap_or(0.0),
                    label: Some(format!("{}x{}", result.config.tile_m, result.config.tile_n)),
                    error_low: None,
                    error_high: None,
                }
            })
            .collect();

        if points.is_empty() {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "no results for kernel={kernel}, gpu={gpu}, problem={problem}"
            )));
        }

        points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        Ok(ChartData {
            title: format!("{kernel} Tile Performance ({problem})"),
            x_label: "Tile Area (tile_m * tile_n)".to_string(),
            y_label: "GFLOPS".to_string(),
            series: vec![DataSeries {
                name: kernel.to_string(),
                points,
                style: SeriesStyle::Scatter,
            }],
        })
    }

    /// Histogram of median execution times across all problems for a kernel.
    pub fn latency_distribution(
        &self,
        kernel: &str,
        gpu: &str,
    ) -> Result<ChartData, AutotuneError> {
        let entries = self.db.list_gpu(gpu);
        let medians: Vec<f64> = entries
            .iter()
            .filter(|(k, _, _)| *k == kernel)
            .map(|(_, _, r)| r.median_us)
            .collect();

        if medians.is_empty() {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "no results for kernel={kernel}, gpu={gpu}"
            )));
        }

        let points = medians
            .iter()
            .enumerate()
            .map(|(i, &v)| DataPoint {
                x: i as f64,
                y: v,
                label: None,
                error_low: None,
                error_high: None,
            })
            .collect();

        Ok(ChartData {
            title: format!("{kernel} Latency Distribution on {gpu}"),
            x_label: "Sample".to_string(),
            y_label: "Median (us)".to_string(),
            series: vec![DataSeries {
                name: kernel.to_string(),
                points,
                style: SeriesStyle::Histogram,
            }],
        })
    }

    /// Renders a tile_m x tile_n heatmap showing GFLOPS as ASCII.
    ///
    /// Returns a formatted string with rows = tile_m values and
    /// columns = tile_n values, cells showing GFLOPS.
    pub fn config_heatmap(
        &self,
        kernel: &str,
        gpu: &str,
        problem: &str,
    ) -> Result<String, AutotuneError> {
        let entries = self.db.list_gpu(gpu);
        let filtered: Vec<_> = entries
            .iter()
            .filter(|(k, p, _)| *k == kernel && *p == problem)
            .collect();

        if filtered.is_empty() {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "no results for kernel={kernel}, gpu={gpu}, problem={problem}"
            )));
        }

        // Collect unique tile_m and tile_n values.
        let mut tile_ms: Vec<u32> = filtered.iter().map(|(_, _, r)| r.config.tile_m).collect();
        let mut tile_ns: Vec<u32> = filtered.iter().map(|(_, _, r)| r.config.tile_n).collect();
        tile_ms.sort();
        tile_ms.dedup();
        tile_ns.sort();
        tile_ns.dedup();

        // Build lookup.
        let mut grid = std::collections::HashMap::new();
        for (_, _, result) in &filtered {
            let key = (result.config.tile_m, result.config.tile_n);
            let gf = result.gflops.unwrap_or(0.0);
            grid.insert(key, gf);
        }

        let mut out = String::new();
        let _ = writeln!(out, "Heatmap: {kernel} on {gpu} ({problem})");
        let _ = writeln!(out, "Rows=tile_m, Cols=tile_n, Values=GFLOPS");
        let _ = writeln!(out);

        // Header.
        let _ = write!(out, "{:>8}", "tile_m\\n");
        for &tn in &tile_ns {
            let _ = write!(out, " {:>8}", tn);
        }
        let _ = writeln!(out);

        let _ = write!(out, "{:>8}", "--------");
        for _ in &tile_ns {
            let _ = write!(out, " {:>8}", "--------");
        }
        let _ = writeln!(out);

        for &tm in &tile_ms {
            let _ = write!(out, "{:>8}", tm);
            for &tn in &tile_ns {
                if let Some(&gf) = grid.get(&(tm, tn)) {
                    let _ = write!(out, " {:>8.1}", gf);
                } else {
                    let _ = write!(out, " {:>8}", "-");
                }
            }
            let _ = writeln!(out);
        }

        Ok(out)
    }

    /// Side-by-side kernel comparison table for a specific GPU and problem.
    pub fn comparison_table(
        &self,
        kernels: &[&str],
        gpu: &str,
        problem: &str,
    ) -> Result<String, AutotuneError> {
        let entries = self.db.list_gpu(gpu);
        let chart = AsciiChart::default();

        let headers: Vec<&str> = vec![
            "Kernel",
            "Median(us)",
            "Min(us)",
            "Max(us)",
            "StdDev",
            "GFLOPS",
            "Tile",
        ];

        let mut rows: Vec<Vec<String>> = Vec::new();
        let mut found_any = false;

        for &kernel in kernels {
            for (k, p, result) in &entries {
                if *k == kernel && *p == problem {
                    found_any = true;
                    let tile = format!("{}x{}", result.config.tile_m, result.config.tile_n);
                    rows.push(vec![
                        kernel.to_string(),
                        format!("{:.1}", result.median_us),
                        format!("{:.1}", result.min_us),
                        format!("{:.1}", result.max_us),
                        format!("{:.2}", result.stddev_us),
                        result.gflops.map_or("-".to_string(), |g| format!("{g:.1}")),
                        tile,
                    ]);
                }
            }
        }

        if !found_any {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "no results for kernels={kernels:?}, gpu={gpu}, problem={problem}"
            )));
        }

        Ok(chart.render_table(&headers, &rows))
    }
}

// ---------------------------------------------------------------------------
// GnuplotExporter
// ---------------------------------------------------------------------------

/// Exports chart data in gnuplot-compatible formats.
pub struct GnuplotExporter;

impl GnuplotExporter {
    /// Writes a `.dat` file with tab-separated columns.
    ///
    /// Columns: x, y, error_low (or 0), error_high (or 0), label (if any).
    pub fn export_dat(data: &ChartData, path: &str) -> Result<(), AutotuneError> {
        let mut content = String::new();
        let _ = writeln!(content, "# {}", data.title);
        let _ = writeln!(content, "# x\ty\terror_low\terror_high\tlabel");

        for series in &data.series {
            let _ = writeln!(content, "# Series: {}", series.name);
            for pt in &series.points {
                let label = pt.label.as_deref().unwrap_or("");
                let _ = writeln!(
                    content,
                    "{}\t{}\t{}\t{}\t{}",
                    pt.x,
                    pt.y,
                    pt.error_low.unwrap_or(0.0),
                    pt.error_high.unwrap_or(0.0),
                    label,
                );
            }
            let _ = writeln!(content);
        }

        std::fs::write(path, content)?;
        Ok(())
    }

    /// Generates a gnuplot script string for rendering the chart.
    ///
    /// The script reads data from `dat_path` and outputs to `output_path`
    /// (PNG format).
    pub fn export_script(
        data: &ChartData,
        dat_path: &str,
        output_path: &str,
    ) -> Result<String, AutotuneError> {
        let mut script = String::new();
        let _ = writeln!(script, "set terminal png size 1024,768");
        let _ = writeln!(script, "set output '{output_path}'");
        let _ = writeln!(script, "set title '{}'", data.title);
        let _ = writeln!(script, "set xlabel '{}'", data.x_label);
        let _ = writeln!(script, "set ylabel '{}'", data.y_label);
        let _ = writeln!(script, "set grid");
        let _ = writeln!(script, "set key outside right top");
        let _ = writeln!(script);

        if data.series.is_empty() {
            let _ = writeln!(
                script,
                "plot '{dat_path}' using 1:2 with linespoints title 'data'"
            );
        } else {
            let mut plot_parts = Vec::new();
            for (i, series) in data.series.iter().enumerate() {
                let style = match series.style {
                    SeriesStyle::Line => "linespoints",
                    SeriesStyle::Bar => "boxes",
                    SeriesStyle::Scatter => "points",
                    SeriesStyle::Histogram => "boxes",
                };
                let idx_offset = i; // gnuplot index for multi-dataset
                plot_parts.push(format!(
                    "'{dat_path}' index {idx_offset} using 1:2 with {style} title '{}'",
                    series.name,
                ));
            }
            let _ = writeln!(script, "plot {}", plot_parts.join(", \\\n     "));
        }

        Ok(script)
    }
}

// ---------------------------------------------------------------------------
// CsvVisualizationExporter
// ---------------------------------------------------------------------------

/// Exports pivot-table CSV data for spreadsheet tools.
pub struct CsvVisualizationExporter;

impl CsvVisualizationExporter {
    /// Generates a CSV pivot table: problem x config -> GFLOPS.
    ///
    /// Rows are problem keys, columns are config descriptors
    /// (tile_m x tile_n), and cell values are GFLOPS.
    pub fn export_pivot_table(
        db: &ResultDb,
        kernel: &str,
        gpu: &str,
    ) -> Result<String, AutotuneError> {
        let entries = db.list_gpu(gpu);
        let filtered: Vec<_> = entries.iter().filter(|(k, _, _)| *k == kernel).collect();

        if filtered.is_empty() {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "no results for kernel={kernel}, gpu={gpu}"
            )));
        }

        // Collect unique problems and configs.
        let mut problems: Vec<String> = filtered.iter().map(|(_, p, _)| p.to_string()).collect();
        problems.sort();
        problems.dedup();

        let mut configs: Vec<String> = filtered
            .iter()
            .map(|(_, _, r)| format!("{}x{}", r.config.tile_m, r.config.tile_n))
            .collect();
        configs.sort();
        configs.dedup();

        // Build lookup.
        let mut lookup = std::collections::HashMap::new();
        for (_, problem, result) in &filtered {
            let cfg_key = format!("{}x{}", result.config.tile_m, result.config.tile_n);
            let gf = result.gflops.unwrap_or(0.0);
            lookup.insert((problem.to_string(), cfg_key), gf);
        }

        let mut out = String::new();
        // Header.
        let _ = write!(out, "problem");
        for cfg in &configs {
            let _ = write!(out, ",{cfg}");
        }
        let _ = writeln!(out);

        // Data rows.
        for problem in &problems {
            let _ = write!(out, "{problem}");
            for cfg in &configs {
                if let Some(&gf) = lookup.get(&(problem.clone(), cfg.clone())) {
                    let _ = write!(out, ",{gf:.1}");
                } else {
                    let _ = write!(out, ",");
                }
            }
            let _ = writeln!(out);
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extracts the first numeric value from a string for sorting.
///
/// Falls back to 0.0 if no digits are found.
fn extract_numeric(s: &str) -> f64 {
    let num_str: String = s
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    if num_str.is_empty() {
        // Try to find first digit sequence anywhere.
        let digits: String = s
            .chars()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        digits.parse::<f64>().unwrap_or(0.0)
    } else {
        num_str.parse::<f64>().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkResult;
    use crate::config::Config;

    fn make_result_with(
        tile_m: u32,
        tile_n: u32,
        median_us: f64,
        gflops: Option<f64>,
    ) -> BenchmarkResult {
        BenchmarkResult {
            config: Config::new().with_tile_m(tile_m).with_tile_n(tile_n),
            median_us,
            min_us: median_us - 1.0,
            max_us: median_us + 1.0,
            stddev_us: 0.5,
            gflops,
            efficiency: None,
        }
    }

    fn temp_db(name: &str) -> (ResultDb, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("oxicuda_viz_test_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("results.json");
        let db = ResultDb::open_at(path).expect("open db");
        (db, dir)
    }

    fn populated_db(name: &str) -> (ResultDb, std::path::PathBuf) {
        let (mut db, dir) = temp_db(name);
        db.save(
            "RTX4090",
            "sgemm",
            "512x512",
            make_result_with(64, 64, 10.0, Some(500.0)),
        )
        .expect("save");
        db.save(
            "RTX4090",
            "sgemm",
            "1024x1024",
            make_result_with(128, 128, 25.0, Some(1200.0)),
        )
        .expect("save");
        db.save(
            "RTX4090",
            "sgemm",
            "2048x2048",
            make_result_with(128, 256, 80.0, Some(1800.0)),
        )
        .expect("save");
        db.save(
            "RTX4090",
            "dgemm",
            "1024x1024",
            make_result_with(64, 64, 50.0, Some(400.0)),
        )
        .expect("save");
        (db, dir)
    }

    #[test]
    fn bar_chart_renders_with_data() {
        let data = ChartData {
            title: "Test Chart".to_string(),
            x_label: "Config".to_string(),
            y_label: "GFLOPS".to_string(),
            series: vec![DataSeries {
                name: "series1".to_string(),
                points: vec![
                    DataPoint {
                        x: 1.0,
                        y: 100.0,
                        label: Some("A".to_string()),
                        error_low: None,
                        error_high: None,
                    },
                    DataPoint {
                        x: 2.0,
                        y: 200.0,
                        label: Some("B".to_string()),
                        error_low: None,
                        error_high: None,
                    },
                    DataPoint {
                        x: 3.0,
                        y: 150.0,
                        label: Some("C".to_string()),
                        error_low: None,
                        error_high: None,
                    },
                ],
                style: SeriesStyle::Bar,
            }],
        };

        let chart = AsciiChart::new(60, 20);
        let rendered = chart.render_bar_chart(&data);
        assert!(rendered.contains("Test Chart"));
        assert!(rendered.contains("A"));
        assert!(rendered.contains("B"));
        assert!(rendered.contains("200.0"));
    }

    #[test]
    fn bar_chart_empty_data() {
        let data = ChartData {
            title: "Empty".to_string(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            series: vec![],
        };
        let chart = AsciiChart::default();
        let rendered = chart.render_bar_chart(&data);
        assert!(rendered.contains("(no data)"));
    }

    #[test]
    fn sparkline_basic() {
        let chart = AsciiChart::default();
        let spark = chart.render_sparkline(&[1.0, 3.0, 5.0, 2.0, 8.0, 4.0]);
        assert!(!spark.is_empty());
        assert_eq!(spark.chars().count(), 6);
        // The highest value (8.0) should produce the full block.
        let chars: Vec<char> = spark.chars().collect();
        assert_eq!(chars[4], '\u{2588}');
        // The lowest value (1.0) should produce the smallest block.
        assert_eq!(chars[0], '\u{2581}');
    }

    #[test]
    fn sparkline_empty() {
        let chart = AsciiChart::default();
        let spark = chart.render_sparkline(&[]);
        assert!(spark.is_empty());
    }

    #[test]
    fn sparkline_constant_values() {
        let chart = AsciiChart::default();
        let spark = chart.render_sparkline(&[5.0, 5.0, 5.0]);
        assert_eq!(spark.chars().count(), 3);
        // All same value => all map to middle block.
    }

    #[test]
    fn table_alignment() {
        let chart = AsciiChart::default();
        let headers = vec!["Name", "Value", "Unit"];
        let rows = vec![
            vec![
                "alpha".to_string(),
                "1234.5".to_string(),
                "GFLOPS".to_string(),
            ],
            vec!["beta".to_string(), "42.0".to_string(), "us".to_string()],
        ];
        let table = chart.render_table(&headers, &rows);
        assert!(table.contains("Name"));
        assert!(table.contains("alpha"));
        assert!(table.contains("1234.5"));
        // All lines should have the same number of '|' separators.
        let lines: Vec<&str> = table.lines().collect();
        let pipe_count = lines[0].matches('|').count();
        for line in &lines {
            assert_eq!(line.matches('|').count(), pipe_count);
        }
    }

    #[test]
    fn histogram_binning() {
        let chart = AsciiChart::new(60, 20);
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let hist = chart.render_histogram(&values, 5);
        // Should have 5 bin lines.
        let data_lines: Vec<&str> = hist.lines().filter(|l| l.starts_with('[')).collect();
        assert_eq!(data_lines.len(), 5);
    }

    #[test]
    fn histogram_empty() {
        let chart = AsciiChart::default();
        let hist = chart.render_histogram(&[], 10);
        assert!(hist.contains("(no data)"));
    }

    #[test]
    fn performance_by_problem_from_db() {
        let (db, dir) = populated_db("perf_problem");
        let builder = VisualizationBuilder::from_db(&db);
        let chart_data = builder.performance_by_problem("sgemm", "RTX4090");
        assert!(chart_data.is_ok());
        let cd = chart_data.expect("chart data");
        assert_eq!(cd.series.len(), 1);
        assert_eq!(cd.series[0].points.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn performance_by_problem_missing_kernel() {
        let (db, dir) = populated_db("perf_missing");
        let builder = VisualizationBuilder::from_db(&db);
        let result = builder.performance_by_problem("nonexistent", "RTX4090");
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn latency_distribution_from_db() {
        let (db, dir) = populated_db("latency_dist");
        let builder = VisualizationBuilder::from_db(&db);
        let chart_data = builder.latency_distribution("sgemm", "RTX4090");
        assert!(chart_data.is_ok());
        let cd = chart_data.expect("chart data");
        assert_eq!(cd.series[0].points.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_heatmap_from_db() {
        let (db, dir) = populated_db("heatmap");
        let builder = VisualizationBuilder::from_db(&db);
        let heatmap = builder.config_heatmap("sgemm", "RTX4090", "1024x1024");
        assert!(heatmap.is_ok());
        let text = heatmap.expect("heatmap text");
        assert!(text.contains("Heatmap"));
        assert!(text.contains("tile_m"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn comparison_table_from_db() {
        let (mut db, dir) = populated_db("comparison");
        db.save(
            "RTX4090",
            "dgemm",
            "1024x1024",
            make_result_with(64, 64, 50.0, Some(400.0)),
        )
        .expect("save");
        let builder = VisualizationBuilder::from_db(&db);
        let table = builder.comparison_table(&["sgemm", "dgemm"], "RTX4090", "1024x1024");
        assert!(table.is_ok());
        let text = table.expect("table");
        assert!(text.contains("sgemm"));
        assert!(text.contains("dgemm"));
        assert!(text.contains("Kernel"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gnuplot_dat_export() {
        let data = ChartData {
            title: "Test".to_string(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            series: vec![DataSeries {
                name: "s1".to_string(),
                points: vec![
                    DataPoint {
                        x: 1.0,
                        y: 10.0,
                        label: Some("pt1".to_string()),
                        error_low: None,
                        error_high: None,
                    },
                    DataPoint {
                        x: 2.0,
                        y: 20.0,
                        label: None,
                        error_low: Some(1.0),
                        error_high: Some(2.0),
                    },
                ],
                style: SeriesStyle::Line,
            }],
        };

        let path = std::env::temp_dir().join("oxicuda_viz_test_gnuplot.dat");
        let path_str = path.to_string_lossy().to_string();
        let result = GnuplotExporter::export_dat(&data, &path_str);
        assert!(result.is_ok());

        let content = std::fs::read_to_string(&path).expect("read dat");
        assert!(content.contains("# Test"));
        assert!(content.contains("1\t10"));
        assert!(content.contains("pt1"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gnuplot_script_generation() {
        let data = ChartData {
            title: "Perf Chart".to_string(),
            x_label: "Size".to_string(),
            y_label: "GFLOPS".to_string(),
            series: vec![DataSeries {
                name: "sgemm".to_string(),
                points: vec![],
                style: SeriesStyle::Line,
            }],
        };

        let script = GnuplotExporter::export_script(&data, "data.dat", "output.png");
        assert!(script.is_ok());
        let s = script.expect("script");
        assert!(s.contains("set terminal png"));
        assert!(s.contains("set title 'Perf Chart'"));
        assert!(s.contains("data.dat"));
        assert!(s.contains("output.png"));
        assert!(s.contains("linespoints"));
    }

    #[test]
    fn csv_pivot_table() {
        let (db, dir) = populated_db("pivot");
        let csv = CsvVisualizationExporter::export_pivot_table(&db, "sgemm", "RTX4090");
        assert!(csv.is_ok());
        let text = csv.expect("csv");
        assert!(text.contains("problem"));
        assert!(text.contains("512x512"));
        assert!(text.contains("1024x1024"));
        // Should contain GFLOPS values.
        assert!(text.contains("500.0") || text.contains("1200.0"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn csv_pivot_table_no_data() {
        let (db, dir) = temp_db("pivot_empty");
        let result = CsvVisualizationExporter::export_pivot_table(&db, "sgemm", "GPU0");
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_numeric_cases() {
        assert!((extract_numeric("1024x1024") - 1024.0).abs() < 1e-9);
        assert!((extract_numeric("abc512def") - 512.0).abs() < 1e-9);
        assert!((extract_numeric("nodigits") - 0.0).abs() < 1e-9);
    }
}
