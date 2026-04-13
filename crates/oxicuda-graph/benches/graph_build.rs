use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_graph::builder::GraphBuilder;
use oxicuda_graph::executor::{ExecutionPlan, SequentialExecutor};
use oxicuda_graph::node::MemcpyDir;

// --- Graph construction helpers -----------------------------------------------

/// Build a linear pipeline: upload -> N fusible kernels -> download.
fn build_linear_pipeline(n_kernels: usize) -> GraphBuilder {
    let mut b = GraphBuilder::new();

    let buf_in = b.alloc_buffer("input", 4096);
    let upload = b.add_memcpy("upload", MemcpyDir::HostToDevice, 4096);

    let mut prev_buf = buf_in;
    let mut prev_node = upload;

    for i in 0..n_kernels {
        let buf_out = b.alloc_buffer(&format!("mid_{i}"), 4096);
        let kernel = b
            .add_kernel(&format!("k{i}"), 16, 256, 0)
            .fusible(true)
            .inputs([prev_buf])
            .outputs([buf_out])
            .finish();
        b.dep(prev_node, kernel);
        prev_buf = buf_out;
        prev_node = kernel;
    }

    let download = b.add_memcpy("download", MemcpyDir::DeviceToHost, 4096);
    b.dep(prev_node, download);
    b
}

/// Build a wide fan-out / fan-in pattern (one upload -> N parallel kernels -> merge).
fn build_wide_fanout(n_branches: usize) -> GraphBuilder {
    let mut b = GraphBuilder::new();

    let buf_in = b.alloc_buffer("input", 4096);
    let upload = b.add_memcpy("upload", MemcpyDir::HostToDevice, 4096);

    let mut branch_nodes = Vec::with_capacity(n_branches);
    let mut branch_bufs = Vec::with_capacity(n_branches);

    for i in 0..n_branches {
        let buf_out = b.alloc_buffer(&format!("branch_{i}"), 4096);
        let k = b
            .add_kernel(&format!("branch_{i}"), 16, 256, 0)
            .fusible(false)
            .inputs([buf_in])
            .outputs([buf_out])
            .finish();
        b.dep(upload, k);
        branch_nodes.push(k);
        branch_bufs.push(buf_out);
    }

    // Merge kernel reads all branch outputs.
    let buf_merged = b.alloc_buffer("merged", 4096);
    let merge = b
        .add_kernel("merge", 16, 256, 0)
        .fusible(false)
        .inputs(branch_bufs)
        .outputs([buf_merged])
        .finish();

    for bn in branch_nodes {
        b.dep(bn, merge);
    }

    let download = b.add_memcpy("download", MemcpyDir::DeviceToHost, 4096);
    b.dep(merge, download);
    b
}

/// Transformer-block-like graph: LayerNorm -> QKV proj -> attention -> FFN.
fn build_transformer_block_graph(n_layers: usize) -> GraphBuilder {
    let mut b = GraphBuilder::new();

    let mut prev_buf = b.alloc_buffer("residual_0", 4096);
    let upload = b.add_memcpy("upload", MemcpyDir::HostToDevice, 4096);
    let mut prev_node = upload;

    for layer in 0..n_layers {
        // LayerNorm
        let buf_ln = b.alloc_buffer(&format!("ln_{layer}"), 4096);
        let k_ln = b
            .add_kernel(&format!("layernorm_{layer}"), 16, 128, 0)
            .fusible(false)
            .inputs([prev_buf])
            .outputs([buf_ln])
            .finish();
        b.dep(prev_node, k_ln);

        // Q, K, V projections (parallel fan-out)
        let buf_q = b.alloc_buffer(&format!("q_{layer}"), 4096);
        let buf_k = b.alloc_buffer(&format!("k_{layer}"), 4096);
        let buf_v = b.alloc_buffer(&format!("v_{layer}"), 4096);

        let k_q = b
            .add_kernel(&format!("proj_q_{layer}"), 16, 256, 0)
            .fusible(false)
            .inputs([buf_ln])
            .outputs([buf_q])
            .finish();
        let k_k = b
            .add_kernel(&format!("proj_k_{layer}"), 16, 256, 0)
            .fusible(false)
            .inputs([buf_ln])
            .outputs([buf_k])
            .finish();
        let k_v = b
            .add_kernel(&format!("proj_v_{layer}"), 16, 256, 0)
            .fusible(false)
            .inputs([buf_ln])
            .outputs([buf_v])
            .finish();

        b.dep(k_ln, k_q).dep(k_ln, k_k).dep(k_ln, k_v);

        // Attention
        let buf_attn = b.alloc_buffer(&format!("attn_{layer}"), 4096);
        let k_attn = b
            .add_kernel(&format!("attention_{layer}"), 32, 256, 32_768)
            .fusible(false)
            .inputs([buf_q, buf_k, buf_v])
            .outputs([buf_attn])
            .finish();
        b.dep(k_q, k_attn).dep(k_k, k_attn).dep(k_v, k_attn);

        // FFN: two fused kernels
        let buf_ffn1 = b.alloc_buffer(&format!("ffn1_{layer}"), 16384);
        let buf_ffn2 = b.alloc_buffer(&format!("ffn2_{layer}"), 4096);
        let k_ffn1 = b
            .add_kernel(&format!("ffn_up_{layer}"), 32, 256, 0)
            .fusible(true)
            .inputs([buf_attn])
            .outputs([buf_ffn1])
            .finish();
        let k_ffn2 = b
            .add_kernel(&format!("ffn_down_{layer}"), 32, 256, 0)
            .fusible(true)
            .inputs([buf_ffn1])
            .outputs([buf_ffn2])
            .finish();
        b.dep(k_attn, k_ffn1).dep(k_ffn1, k_ffn2);

        prev_buf = buf_ffn2;
        prev_node = k_ffn2;
    }

    let download = b.add_memcpy("download", MemcpyDir::DeviceToHost, 4096);
    b.dep(prev_node, download);
    b
}

// --- Benchmarks ---------------------------------------------------------------

fn bench_graph_build_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_build_linear");

    for n in [4, 16, 64, 256] {
        group.bench_with_input(BenchmarkId::new("build", n), &n, |b, &n| {
            b.iter(|| {
                let builder = build_linear_pipeline(n);
                black_box(builder.build().unwrap())
            });
        });
    }
    group.finish();
}

fn bench_graph_build_fanout(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_build_fanout");

    for n in [4, 8, 16, 32] {
        group.bench_with_input(BenchmarkId::new("build", n), &n, |b, &n| {
            b.iter(|| {
                let builder = build_wide_fanout(n);
                black_box(builder.build().unwrap())
            });
        });
    }
    group.finish();
}

fn bench_graph_build_transformer(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_build_transformer");

    for n_layers in [1, 4, 12, 32] {
        group.bench_with_input(
            BenchmarkId::new("build", n_layers),
            &n_layers,
            |b, &n_layers| {
                b.iter(|| {
                    let builder = build_transformer_block_graph(n_layers);
                    black_box(builder.build().unwrap())
                });
            },
        );
    }
    group.finish();
}

fn bench_execution_plan_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_plan_compile");

    // Pre-build graphs outside timing loop; measure only ExecutionPlan::build
    let graph_linear_64 = build_linear_pipeline(64).build().unwrap();
    let graph_fanout_16 = build_wide_fanout(16).build().unwrap();
    let graph_transformer_12 = build_transformer_block_graph(12).build().unwrap();
    let graph_transformer_32 = build_transformer_block_graph(32).build().unwrap();

    group.bench_function("linear_64_nodes_4streams", |b| {
        b.iter(|| black_box(ExecutionPlan::build(&graph_linear_64, 4).unwrap()));
    });

    group.bench_function("fanout_16_branches_8streams", |b| {
        b.iter(|| black_box(ExecutionPlan::build(&graph_fanout_16, 8).unwrap()));
    });

    group.bench_function("transformer_12l_4streams", |b| {
        b.iter(|| black_box(ExecutionPlan::build(&graph_transformer_12, 4).unwrap()));
    });

    group.bench_function("transformer_32l_8streams", |b| {
        b.iter(|| black_box(ExecutionPlan::build(&graph_transformer_32, 8).unwrap()));
    });

    group.finish();
}

fn bench_sequential_executor_validate(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_executor");

    let graph_12l = build_transformer_block_graph(12).build().unwrap();
    let plan_12l = ExecutionPlan::build(&graph_12l, 4).unwrap();

    let graph_32l = build_transformer_block_graph(32).build().unwrap();
    let plan_32l = ExecutionPlan::build(&graph_32l, 8).unwrap();

    group.bench_function("validate_12l", |b| {
        b.iter(|| {
            let exec = SequentialExecutor::new(&plan_12l);
            black_box(exec.validate().unwrap())
        });
    });

    group.bench_function("validate_32l", |b| {
        b.iter(|| {
            let exec = SequentialExecutor::new(&plan_32l);
            black_box(exec.validate().unwrap())
        });
    });

    group.bench_function("run_12l_cpu_sim", |b| {
        b.iter(|| {
            let exec = SequentialExecutor::new(&plan_12l);
            black_box(exec.run().unwrap())
        });
    });

    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_full_pipeline");

    let sizes: &[(&str, usize)] = &[("4_kernels", 4), ("16_kernels", 16), ("64_kernels", 64)];

    for &(name, n) in sizes {
        group.bench_with_input(
            BenchmarkId::new("build_compile_validate", name),
            &n,
            |b, &n| {
                b.iter(|| {
                    let graph = build_linear_pipeline(n).build().unwrap();
                    let plan = ExecutionPlan::build(&graph, 4).unwrap();
                    let exec = SequentialExecutor::new(&plan);
                    black_box(exec.validate().unwrap())
                });
            },
        );
    }
    group.finish();
}

// --- criterion wiring ---------------------------------------------------------

criterion_group!(
    graph_benches,
    bench_graph_build_linear,
    bench_graph_build_fanout,
    bench_graph_build_transformer,
    bench_execution_plan_compile,
    bench_sequential_executor_validate,
    bench_full_pipeline,
);
criterion_main!(graph_benches);
