//! GPU dispatch paths for `conv2d_forward` / `attention`, their CPU-reference
//! oracles, and the dispatch-grid planning that decides between the two.
//!
//! Split out of `backend.rs` (as `backend::gpu_ops`, via `#[path = ...] mod
//! gpu_ops;`) purely to keep that file under the workspace's 2 000-line
//! refactoring policy — everything here is still logically part of the
//! `WebGpuBackend` implementation and freely uses `backend`'s private helpers
//! (`gpu_limits`, `bytes_to_f32_vec`, …) via ordinary Rust module-privacy
//! rules (a private item is visible to its defining module *and all
//! descendants*, and `gpu_ops` is a child of `backend`).  Symmetrically, the
//! items here that `backend.rs` and `backend_tests.rs` (its sibling `tests`
//! submodule) need to call are marked `pub(super)`.

use oxicuda_backend::{BackendError, BackendResult};

use super::{WebGpuBackend, gpu_limits};
use crate::shader;

// ─── Conv2D: CPU oracle + GPU dispatch planning ──────────────────────────────

/// CPU reference implementation of 2-D convolution (NCHW).
///
/// This is both the fallback path for configurations too large for
/// [`shader::conv2d_wgsl`]'s fixed 2-D dispatch (see
/// [`conv2d_gpu_dispatch_grid`]) *and* the correctness oracle GPU results are
/// checked against in `backend_tests.rs` — keeping a single implementation
/// means the two paths cannot silently drift apart.
#[allow(clippy::too_many_arguments)]
pub(super) fn conv2d_cpu_reference(
    in_f32: &[f32],
    f_f32: &[f32],
    batch: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    k_out: usize,
    fh: usize,
    fw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let mut out_f32 = vec![0.0f32; batch * k_out * oh * ow];
    for b in 0..batch {
        for kf in 0..k_out {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0.0f32;
                    for ci in 0..c_in {
                        for fy in 0..fh {
                            for fx in 0..fw {
                                let iy = (oy * sh + fy) as isize - ph as isize;
                                let ix = (ox * sw + fx) as isize - pw as isize;
                                if iy >= 0
                                    && (iy as usize) < h_in
                                    && ix >= 0
                                    && (ix as usize) < w_in
                                {
                                    let in_idx =
                                        ((b * c_in + ci) * h_in + iy as usize) * w_in + ix as usize;
                                    let f_idx = ((kf * c_in + ci) * fh + fy) * fw + fx;
                                    acc += in_f32[in_idx] * f_f32[f_idx];
                                }
                            }
                        }
                    }
                    out_f32[((b * k_out + kf) * oh + oy) * ow + ox] = acc;
                }
            }
        }
    }
    out_f32
}

/// Checked-`u32` shape parameters for [`shader::conv2d_wgsl`]'s baked-literal
/// generator.  `None` from [`conv2d_u32_dims`] means at least one dimension
/// exceeds `u32::MAX`, so the GPU path cannot express this shape at all.
#[derive(Debug, Clone, Copy)]
pub(super) struct Conv2dU32Dims {
    pub(super) n: u32,
    pub(super) c_in: u32,
    pub(super) h_in: u32,
    pub(super) w_in: u32,
    pub(super) k_out: u32,
    pub(super) fh: u32,
    pub(super) fw: u32,
    pub(super) oh: u32,
    pub(super) ow: u32,
    pub(super) sh: u32,
    pub(super) sw: u32,
    pub(super) ph: u32,
    pub(super) pw: u32,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn conv2d_u32_dims(
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    k_out: usize,
    fh: usize,
    fw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> Option<Conv2dU32Dims> {
    Some(Conv2dU32Dims {
        n: u32::try_from(n).ok()?,
        c_in: u32::try_from(c_in).ok()?,
        h_in: u32::try_from(h_in).ok()?,
        w_in: u32::try_from(w_in).ok()?,
        k_out: u32::try_from(k_out).ok()?,
        fh: u32::try_from(fh).ok()?,
        fw: u32::try_from(fw).ok()?,
        oh: u32::try_from(oh).ok()?,
        ow: u32::try_from(ow).ok()?,
        sh: u32::try_from(sh).ok()?,
        sw: u32::try_from(sw).ok()?,
        ph: u32::try_from(ph).ok()?,
        pw: u32::try_from(pw).ok()?,
    })
}

/// Compute the `(wg_x, wg_y)` dispatch-workgroup counts for
/// [`shader::conv2d_wgsl`]'s fixed `@workgroup_size(8, 8)`, or `None` if
/// either axis would exceed the portable per-axis workgroup cap.
///
/// Unlike `reduction_nd_wgsl`, this shader has no 2-D dispatch fold — `gid.x`
/// addresses `ow` directly and `gid.y` addresses `batch * k_out * oh`
/// directly — so a shape whose grid would overflow genuinely cannot be
/// expressed by this kernel and must fall back to the CPU reference.
pub(super) fn conv2d_gpu_dispatch_grid(
    batch: usize,
    k_out: usize,
    oh: usize,
    ow: usize,
) -> Option<(u32, u32)> {
    let cap = u64::from(gpu_limits().max_workgroups_per_dim);
    let rows = (batch as u64)
        .checked_mul(k_out as u64)?
        .checked_mul(oh as u64)?;
    let wg_x = (ow as u64).div_ceil(8);
    let wg_y = rows.div_ceil(8);
    if wg_x > cap || wg_y > cap {
        return None;
    }
    Some((u32::try_from(wg_x).ok()?, u32::try_from(wg_y).ok()?))
}

// ─── Attention: CPU oracle + GPU dispatch planning ───────────────────────────

/// CPU reference implementation of scaled dot-product attention with stable
/// softmax and optional causal masking.
///
/// Serves the same dual role as [`conv2d_cpu_reference`]: the production
/// fallback for shapes too large for the fixed-`@workgroup_size(64)`
/// [`shader::attention_wgsl`] dispatch, and the oracle `backend_tests.rs`
/// checks GPU results against.
#[allow(clippy::too_many_arguments)]
pub(super) fn attention_cpu_reference(
    q_f32: &[f32],
    k_f32: &[f32],
    v_f32: &[f32],
    batch_heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale_f32: f32,
    causal: bool,
) -> Vec<f32> {
    let mut o_f32 = vec![0.0f32; batch_heads * seq_q * head_dim];

    for bh in 0..batch_heads {
        let q_off = bh * seq_q * head_dim;
        let k_off = bh * seq_kv * head_dim;
        let v_off = k_off;

        for sq in 0..seq_q {
            let kv_limit = if causal { (sq + 1).min(seq_kv) } else { seq_kv };

            // Pass 1: find max score for numerical stability
            let mut max_score = f32::NEG_INFINITY;
            for sk in 0..kv_limit {
                let mut dot = 0.0f32;
                for dd in 0..head_dim {
                    dot += q_f32[q_off + sq * head_dim + dd] * k_f32[k_off + sk * head_dim + dd];
                }
                let s = dot * scale_f32;
                if s > max_score {
                    max_score = s;
                }
            }

            // Pass 2: exp(score - max), accumulate weighted V
            let mut sum_exp = 0.0f32;
            let mut acc = vec![0.0f32; head_dim];
            for sk in 0..kv_limit {
                let mut dot = 0.0f32;
                for dd in 0..head_dim {
                    dot += q_f32[q_off + sq * head_dim + dd] * k_f32[k_off + sk * head_dim + dd];
                }
                let w = (dot * scale_f32 - max_score).exp();
                sum_exp += w;
                for dd in 0..head_dim {
                    acc[dd] += w * v_f32[v_off + sk * head_dim + dd];
                }
            }

            // Normalise
            let o_base = q_off + sq * head_dim;
            if sum_exp > 0.0 {
                for dd in 0..head_dim {
                    o_f32[o_base + dd] = acc[dd] / sum_exp;
                }
            }
        }
    }

    o_f32
}

/// Compute the 1-D dispatch-workgroup count for
/// [`shader::attention_wgsl`]'s fixed `@workgroup_size(64)`, or `None` if it
/// would exceed the portable per-axis workgroup cap (this shader, like
/// `conv2d_wgsl`, has no 2-D dispatch fold).
pub(super) fn attention_gpu_dispatch_grid(batch_heads: usize, seq_q: usize) -> Option<u32> {
    let cap = u64::from(gpu_limits().max_workgroups_per_dim);
    let total = (batch_heads as u64).checked_mul(seq_q as u64)?;
    let wg = total.div_ceil(64);
    if wg > cap {
        return None;
    }
    u32::try_from(wg).ok()
}

// ─── Dispatch methods on WebGpuBackend ───────────────────────────────────────

impl WebGpuBackend {
    /// GPU dispatch path for `conv2d_forward`.
    ///
    /// The caller (`ComputeBackend::conv2d_forward`) has already validated
    /// shapes and confirmed `dims`/`(wg_x, wg_y)` fit the shader; this method
    /// just binds, validates buffer sizes, and dispatches.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn conv2d_forward_gpu(
        &self,
        input_ptr: u64,
        filter_ptr: u64,
        output_ptr: u64,
        dims: Conv2dU32Dims,
        in_elems: usize,
        f_elems: usize,
        o_elems: usize,
        wg_x: u32,
        wg_y: u32,
    ) -> BackendResult<()> {
        let dev = self.device()?;
        let mem = self.memory()?;

        // `conv2d_wgsl` bakes every shape parameter as a WGSL literal, so the
        // cache key must encode the full shape tuple — otherwise a second
        // distinct shape would silently reuse the first shape's pipeline.
        // This means each distinct (shape, stride, padding) tuple compiles
        // its own pipeline on first use; the alternative (a `Conv2dParams`
        // uniform, one pipeline for all shapes) is a larger refactor of the
        // generator, out of scope here.
        //
        // Two consequences worth naming explicitly, since neither shows up
        // in this crate's tests today (they all reuse a handful of fixed
        // shapes): (1) `pipeline_cache` grows without bound for a caller
        // that dispatches many distinct shapes (e.g. varying batch size
        // every call) — there is no eviction; (2) the *first* dispatch of a
        // never-before-seen shape pays a full `create_shader_module` +
        // `create_compute_pipeline` (naga parse + Metal/Vulkan/D3D12 ISA
        // codegen, commonly tens of milliseconds), so a workload that
        // constantly varies its conv2d shape can measure slower overall
        // than the CPU fallback it replaced. Fine for the steady-shape
        // (training/inference-loop) case this backend targets; a caller
        // with per-call-varying shapes should prefer the `Conv2dParams`
        // uniform refactor noted above.
        let key = format!(
            "conv2d:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
            dims.n,
            dims.c_in,
            dims.h_in,
            dims.w_in,
            dims.k_out,
            dims.fh,
            dims.fw,
            dims.oh,
            dims.ow,
            dims.sh,
            dims.sw,
            dims.ph,
            dims.pw,
        );
        let cached = self.cached_pipeline(&key, "oxicuda-conv2d", || {
            shader::conv2d_wgsl(
                dims.n, dims.c_in, dims.h_in, dims.w_in, dims.k_out, dims.fh, dims.fw, dims.oh,
                dims.ow, dims.sh, dims.sw, dims.ph, dims.pw,
            )
        })?;

        let bgl = &cached.bind_group_layout;
        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let in_info = buffers.get(&input_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {input_ptr}"))
            })?;
            let f_info = buffers.get(&filter_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {filter_ptr}"))
            })?;
            let out_info = buffers.get(&output_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {output_ptr}"))
            })?;

            // The shader has no bounds contract with the caller beyond "the
            // buffers are at least this many f32 elements" — an undersized
            // buffer would otherwise silently drop writes / read garbage
            // rather than error.
            let need_in = (in_elems as u64) * 4;
            let need_f = (f_elems as u64) * 4;
            let need_out = (o_elems as u64) * 4;
            if in_info.size < need_in {
                return Err(BackendError::InvalidArgument(format!(
                    "conv2d_forward: input buffer holds {} bytes, need {need_in}",
                    in_info.size
                )));
            }
            if f_info.size < need_f {
                return Err(BackendError::InvalidArgument(format!(
                    "conv2d_forward: filter buffer holds {} bytes, need {need_f}",
                    f_info.size
                )));
            }
            if out_info.size < need_out {
                return Err(BackendError::InvalidArgument(format!(
                    "conv2d_forward: output buffer holds {} bytes, need {need_out}",
                    out_info.size
                )));
            }

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-conv2d"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: f_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_info.buffer.as_entire_binding(),
                    },
                ],
            })
        };

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-conv2d"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-conv2d"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        // No per-op poll — see the identical comment in `backend.rs`'s
        // `gemm`/`unary`/etc.: queue-FIFO ordering plus the indexed wait in
        // `copy_from_device` / `synchronize()` make it unnecessary and it was
        // a full pipeline stall on every dispatch.

        Ok(())
    }

    /// GPU dispatch path for `attention`.
    ///
    /// The caller (`ComputeBackend::attention`) has already validated shapes
    /// and confirmed the dispatch fits (`attention_gpu_dispatch_grid`); this
    /// method binds, validates buffer sizes, and dispatches.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_gpu(
        &self,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        o_ptr: u64,
        batch_heads: u32,
        seq_q: u32,
        seq_kv: u32,
        head_dim: u32,
        scale: f32,
        causal: bool,
        q_elems: usize,
        kv_elems: usize,
        o_elems: usize,
        wg: u32,
    ) -> BackendResult<()> {
        let dev = self.device()?;
        let mem = self.memory()?;

        // `attention_wgsl` bakes every shape/scale/causal parameter as a WGSL
        // literal (same trade-off, and the same unbounded-`pipeline_cache`-
        // growth / slower-than-CPU-on-a-fresh-shape consequences, as conv2d
        // above), so the cache key encodes the full tuple.  `scale`'s bit
        // pattern (not its decimal text) is used to avoid float-formatting
        // ambiguity in the key.
        let key = format!(
            "attention:{batch_heads}:{seq_q}:{seq_kv}:{head_dim}:{:08x}:{causal}",
            scale.to_bits()
        );
        let cached = self.cached_pipeline(&key, "oxicuda-attention", || {
            shader::attention_wgsl(batch_heads, seq_q, seq_kv, head_dim, scale, causal)
        })?;

        let bgl = &cached.bind_group_layout;
        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let q_info = buffers
                .get(&q_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {q_ptr}")))?;
            let k_info = buffers
                .get(&k_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {k_ptr}")))?;
            let v_info = buffers
                .get(&v_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {v_ptr}")))?;
            let o_info = buffers
                .get(&o_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {o_ptr}")))?;

            let need_q = (q_elems as u64) * 4;
            let need_kv = (kv_elems as u64) * 4;
            let need_o = (o_elems as u64) * 4;
            if q_info.size < need_q {
                return Err(BackendError::InvalidArgument(format!(
                    "attention: q buffer holds {} bytes, need {need_q}",
                    q_info.size
                )));
            }
            if k_info.size < need_kv {
                return Err(BackendError::InvalidArgument(format!(
                    "attention: k buffer holds {} bytes, need {need_kv}",
                    k_info.size
                )));
            }
            if v_info.size < need_kv {
                return Err(BackendError::InvalidArgument(format!(
                    "attention: v buffer holds {} bytes, need {need_kv}",
                    v_info.size
                )));
            }
            if o_info.size < need_o {
                return Err(BackendError::InvalidArgument(format!(
                    "attention: o buffer holds {} bytes, need {need_o}",
                    o_info.size
                )));
            }

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-attention"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: o_info.buffer.as_entire_binding(),
                    },
                ],
            })
        };

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-attention"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-attention"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        // No per-op poll — see the identical comment in `backend.rs`'s
        // `gemm`/`unary`/etc.: queue-FIFO ordering plus the indexed wait in
        // `copy_from_device` / `synchronize()` make it unnecessary and it was
        // a full pipeline stall on every dispatch.

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `conv2d_gpu_dispatch_grid` / `attention_gpu_dispatch_grid` are pure
    // host-side arithmetic (no GPU, no allocation) — exactly the boundary
    // logic that decides whether `conv2d_forward` / `attention` dispatch to
    // the GPU kernel or fall back to `conv2d_cpu_reference` /
    // `attention_cpu_reference`.  Every GPU-vs-CPU test in
    // `backend_tests_gpu_ops.rs` uses shapes small enough that this always
    // returns `Some`, so the `None` branch — the actual trigger for the
    // production CPU-fallback path — is otherwise never exercised by this
    // crate's test suite.  These tests cover the boundary directly, without
    // needing a live GPU or a many-hundred-thousand-element buffer.

    #[test]
    fn conv2d_dispatch_grid_accepts_at_the_per_axis_cap() {
        let cap = u64::from(gpu_limits().max_workgroups_per_dim);
        // wg_y = rows.div_ceil(8) == cap exactly, at the boundary.
        let rows = cap * 8;
        let grid = conv2d_gpu_dispatch_grid(1, 1, rows as usize, 8);
        assert_eq!(grid, Some((1, u32::try_from(cap).expect("cap fits u32"))));
    }

    #[test]
    fn conv2d_dispatch_grid_rejects_one_past_the_cap_on_y() {
        let cap = u64::from(gpu_limits().max_workgroups_per_dim);
        // One more row than the previous test pushes wg_y to cap + 1.
        let rows = cap * 8 + 1;
        assert_eq!(conv2d_gpu_dispatch_grid(1, 1, rows as usize, 8), None);
    }

    #[test]
    fn conv2d_dispatch_grid_rejects_one_past_the_cap_on_x() {
        let cap = u64::from(gpu_limits().max_workgroups_per_dim);
        // wg_x = ow.div_ceil(8); push it to cap + 1.
        let ow = cap * 8 + 1;
        assert_eq!(conv2d_gpu_dispatch_grid(1, 1, 1, ow as usize), None);
    }

    #[test]
    fn attention_dispatch_grid_accepts_at_the_per_axis_cap() {
        let cap = u64::from(gpu_limits().max_workgroups_per_dim);
        // wg = (batch_heads * seq_q).div_ceil(64) == cap exactly.
        let seq_q = cap * 64;
        assert_eq!(
            attention_gpu_dispatch_grid(1, seq_q as usize),
            Some(u32::try_from(cap).expect("cap fits u32"))
        );
    }

    #[test]
    fn attention_dispatch_grid_rejects_one_past_the_cap() {
        let cap = u64::from(gpu_limits().max_workgroups_per_dim);
        let seq_q = cap * 64 + 1;
        assert_eq!(attention_gpu_dispatch_grid(1, seq_q as usize), None);
    }

    #[test]
    fn conv2d_dispatch_grid_none_on_row_product_overflow() {
        // batch * k_out * oh overflowing u64 must return None, not panic.
        assert_eq!(
            conv2d_gpu_dispatch_grid(usize::MAX, 2, 2, 8),
            None,
            "checked_mul overflow in the row-count product must yield None"
        );
    }
}
