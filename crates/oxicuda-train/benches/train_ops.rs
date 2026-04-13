use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxicuda_train::ema::ExponentialMovingAverage;
use oxicuda_train::gpu_optimizer::adamw::GpuAdamW;
use oxicuda_train::gpu_optimizer::lion::GpuLion;
use oxicuda_train::gpu_optimizer::{GpuOptimizer, ParamTensor};
use oxicuda_train::grad_clip::{GlobalNormClip, GradientClipper, PerLayerNormClip, ValueClip};
use oxicuda_train::lr_scheduler::{
    ConstantLR, CosineAnnealingLR, ExponentialLR, LrScheduler, OneCycleLR, ReduceLROnPlateau,
    StepLR, WarmupCosine,
};

// --- helpers -----------------------------------------------------------------

fn make_params(n_params: usize, param_size: usize) -> Vec<ParamTensor> {
    (0..n_params)
        .map(|i| {
            let mut p = ParamTensor::new(vec![0.1_f32; param_size], format!("w{i}"));
            p.set_grad(vec![0.01_f32; param_size]).unwrap();
            p
        })
        .collect()
}

// Optimizer step: build fresh params each iteration so the state grows
// naturally (first step = initialise moment buffers).  We bench the
// steady-state step by running one warm-up step outside the timing loop
// and cloning the warmed-up optimizer + params each iteration.
fn run_adamw_step(params: &mut [ParamTensor], opt: &mut GpuAdamW) {
    // refresh grads so the step has meaningful input
    for p in params.iter_mut() {
        if let Some(g) = p.grad.as_mut() {
            g.iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = (i as f32 + 1.0) * 0.001);
        }
    }
    opt.step(params).unwrap();
}

// --- AdamW optimizer step ----------------------------------------------------

fn bench_adamw_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("adamw_step");

    // (label, n_params, param_size)
    let sizes: &[(&str, usize, usize)] = &[
        ("1k_total", 1, 1_024),
        ("16k_total", 4, 4_096),
        ("256k_total", 4, 65_536),
        ("1m_total", 4, 262_144),
        ("4m_total", 8, 524_288),
    ];

    for &(name, n_params, param_size) in sizes {
        group.bench_with_input(
            BenchmarkId::new("steady_state", name),
            &(n_params, param_size),
            |b, &(np, ps)| {
                // warm-up: initialise moment buffers
                let mut params = make_params(np, ps);
                let mut opt = GpuAdamW::new(3e-4).with_weight_decay(0.01);
                run_adamw_step(&mut params, &mut opt);

                b.iter(|| {
                    run_adamw_step(&mut params, &mut opt);
                    black_box(opt.step_count())
                });
            },
        );
    }
    group.finish();
}

// --- Lion optimizer step -----------------------------------------------------

fn bench_lion_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("lion_step");

    let sizes: &[(&str, usize, usize)] = &[
        ("1k_total", 1, 1_024),
        ("16k_total", 4, 4_096),
        ("256k_total", 4, 65_536),
        ("1m_total", 4, 262_144),
    ];

    for &(name, n_params, param_size) in sizes {
        group.bench_with_input(
            BenchmarkId::new("steady_state", name),
            &(n_params, param_size),
            |b, &(np, ps)| {
                let mut params = make_params(np, ps);
                let mut opt = GpuLion::new(1e-4).with_weight_decay(0.01);
                // warm-up
                opt.step(&mut params).unwrap();

                b.iter(|| {
                    for p in params.iter_mut() {
                        if let Some(g) = p.grad.as_mut() {
                            g.iter_mut().for_each(|v| *v = 0.01_f32);
                        }
                    }
                    opt.step(&mut params).unwrap();
                    black_box(opt.step_count())
                });
            },
        );
    }
    group.finish();
}

// --- LR schedulers -----------------------------------------------------------

fn bench_lr_schedulers(c: &mut Criterion) {
    let mut group = c.benchmark_group("lr_scheduler_step");

    // Single step latency for each scheduler type.
    group.bench_function("constant_step", |b| {
        let mut sched = ConstantLR::new(1e-3);
        b.iter(|| black_box(sched.step()));
    });

    group.bench_function("step_lr_step", |b| {
        let mut sched = StepLR::new(1e-3, 1000, 0.1);
        b.iter(|| black_box(sched.step()));
    });

    group.bench_function("cosine_annealing_step", |b| {
        let mut sched = CosineAnnealingLR::new(1e-3, 10_000);
        b.iter(|| black_box(sched.step()));
    });

    group.bench_function("warmup_cosine_step", |b| {
        let mut sched = WarmupCosine::new(3e-4, 500, 10_000);
        b.iter(|| black_box(sched.step()));
    });

    group.bench_function("exponential_lr_step", |b| {
        let mut sched = ExponentialLR::new(1e-3, 0.99);
        b.iter(|| black_box(sched.step()));
    });

    group.bench_function("one_cycle_lr_step", |b| {
        let mut sched = OneCycleLR::new(1e-3, 10_000);
        b.iter(|| black_box(sched.step()));
    });

    group.bench_function("reduce_on_plateau_step", |b| {
        let mut sched = ReduceLROnPlateau::new(1e-3, 0.5, 10);
        b.iter(|| {
            // simulate alternating improvement / stagnation
            let loss: f64 = 1.0 / (sched.steps_done() as f64 + 1.0);
            let _ = sched.step_metric(loss);
            black_box(sched.get_lr())
        });
    });

    // Throughput: how fast can we step 10k times?
    group.bench_function("warmup_cosine_10k_steps", |b| {
        b.iter(|| {
            let mut sched = WarmupCosine::new(3e-4, 500, 10_000);
            let final_lr = (0..10_000).map(|_| sched.step()).last().unwrap_or(0.0);
            black_box(final_lr)
        });
    });

    group.finish();
}

// --- Gradient clipping -------------------------------------------------------

fn bench_grad_clip(c: &mut Criterion) {
    let mut group = c.benchmark_group("grad_clip");

    let sizes: &[(&str, usize, usize)] = &[
        ("8params_1k", 8, 1_024),
        ("8params_16k", 8, 16_384),
        ("16params_64k", 16, 65_536),
        ("32params_128k", 32, 131_072),
    ];

    for &(name, n_params, param_size) in sizes {
        // GlobalNormClip
        group.bench_with_input(
            BenchmarkId::new("global_norm_clip", name),
            &(n_params, param_size),
            |b, &(np, ps)| {
                b.iter_batched(
                    || make_params(np, ps),
                    |mut params| {
                        let clipper = GlobalNormClip::new(1.0);
                        black_box(clipper.clip(&mut params).unwrap())
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // PerLayerNormClip
        group.bench_with_input(
            BenchmarkId::new("per_layer_norm_clip", name),
            &(n_params, param_size),
            |b, &(np, ps)| {
                b.iter_batched(
                    || make_params(np, ps),
                    |mut params| {
                        let clipper = PerLayerNormClip::new(1.0);
                        black_box(clipper.clip(&mut params).unwrap())
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // ValueClip
        group.bench_with_input(
            BenchmarkId::new("value_clip", name),
            &(n_params, param_size),
            |b, &(np, ps)| {
                b.iter_batched(
                    || make_params(np, ps),
                    |mut params| {
                        let clipper = ValueClip::new(0.1);
                        black_box(clipper.clip(&mut params).unwrap())
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// --- EMA update --------------------------------------------------------------

fn bench_ema_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("ema_update");

    let sizes: &[(&str, usize, usize)] = &[
        ("8params_1k", 8, 1_024),
        ("8params_16k", 8, 16_384),
        ("16params_64k", 16, 65_536),
        ("32params_128k", 32, 131_072),
    ];

    for &(name, n_params, param_size) in sizes {
        group.bench_with_input(
            BenchmarkId::new("bias_correct_decay_0.999", name),
            &(n_params, param_size),
            |b, &(np, ps)| {
                let params = make_params(np, ps);
                let mut ema = ExponentialMovingAverage::new(0.999);
                // warm up
                ema.update(&params).unwrap();
                b.iter(|| {
                    ema.update(&params).unwrap();
                    black_box(ema.step())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fixed_decay_0.999", name),
            &(n_params, param_size),
            |b, &(np, ps)| {
                let params = make_params(np, ps);
                let mut ema = ExponentialMovingAverage::new(0.999).with_fixed_decay();
                ema.update(&params).unwrap();
                b.iter(|| {
                    ema.update(&params).unwrap();
                    black_box(ema.step())
                });
            },
        );
    }

    group.finish();
}

// --- criterion wiring --------------------------------------------------------

criterion_group!(
    train_benches,
    bench_adamw_step,
    bench_lion_step,
    bench_lr_schedulers,
    bench_grad_clip,
    bench_ema_update,
);
criterion_main!(train_benches);
