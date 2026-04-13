//! P1 and P8 quality-gate tests for the PTX builder pipeline.
//!
//! - **P1**: Complete `vector_add` PTX generation for f32, f16 (via `raw_ptx`), and f64.
//! - **P8**: Every [`SmVersion`] variant produces a valid PTX header whose `.target`
//!   and `.version` directives match the expected values from [`SmVersion`].

#[cfg(test)]
mod tests {
    use crate::arch::SmVersion;
    use crate::builder::KernelBuilder;
    use crate::ir::PtxType;

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Generates a `vector_add` kernel for the given SM version and element type.
    ///
    /// The kernel signature is `(a_ptr: u64, b_ptr: u64, c_ptr: u64, n: u32)`.
    /// For f32 and f64 the `BodyBuilder` typed methods are used; for f16 the
    /// elementwise template approach (`raw_ptx`) is used because there are no
    /// dedicated `load_global_f16` / `add_f16` helpers.
    /// Generates a `vector_add` kernel for f32 using raw PTX exclusively.
    fn generate_vector_add_ptx_f32_raw(sm: SmVersion) -> Result<String, crate::error::PtxGenError> {
        KernelBuilder::new("vector_add")
            .target(sm)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .body(|b| {
                // Global thread id
                let gid = b.global_thread_id_x();
                let gid_name = gid.to_string();
                let n_reg = b.load_param_u32("n");

                // Bounds check
                b.if_lt_u32(gid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    // Byte-offset computation + load + add + store in raw PTX
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {gid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, 4;\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;\n    \
                         ld.global.f32 %f_a, [%rd_a];\n    \
                         ld.global.f32 %f_b, [%rd_b];\n    \
                         add.f32 %f_c, %f_a, %f_b;\n    \
                         st.global.f32 [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a `vector_add_f16` kernel using raw PTX (f16 has no typed helpers).
    fn generate_vector_add_ptx_f16(sm: SmVersion) -> Result<String, crate::error::PtxGenError> {
        KernelBuilder::new("vector_add")
            .target(sm)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .body(|b| {
                let gid = b.global_thread_id_x();
                let gid_name = gid.to_string();
                let n_reg = b.load_param_u32("n");

                b.if_lt_u32(gid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    // f16 is 2 bytes
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {gid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, 2;\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;\n    \
                         ld.global.f16 %f_a, [%rd_a];\n    \
                         ld.global.f16 %f_b, [%rd_b];\n    \
                         add.f16 %f_c, %f_a, %f_b;\n    \
                         st.global.f16 [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a `vector_add_f64` kernel.
    fn generate_vector_add_ptx_f64(sm: SmVersion) -> Result<String, crate::error::PtxGenError> {
        KernelBuilder::new("vector_add")
            .target(sm)
            .param("a_ptr", PtxType::U64)
            .param("b_ptr", PtxType::U64)
            .param("c_ptr", PtxType::U64)
            .param("n", PtxType::U32)
            .body(|b| {
                let gid = b.global_thread_id_x();
                let gid_name = gid.to_string();
                let n_reg = b.load_param_u32("n");

                b.if_lt_u32(gid, n_reg, move |b| {
                    let a_ptr = b.load_param_u64("a_ptr");
                    let b_ptr = b.load_param_u64("b_ptr");
                    let c_ptr = b.load_param_u64("c_ptr");

                    // f64 is 8 bytes
                    b.raw_ptx(&format!(
                        "cvt.u64.u32 %rd_off, {gid_name};\n    \
                         mul.lo.u64 %rd_off, %rd_off, 8;\n    \
                         add.u64 %rd_a, {a_ptr}, %rd_off;\n    \
                         add.u64 %rd_b, {b_ptr}, %rd_off;\n    \
                         add.u64 %rd_c, {c_ptr}, %rd_off;\n    \
                         ld.global.f64 %f_a, [%rd_a];\n    \
                         ld.global.f64 %f_b, [%rd_b];\n    \
                         add.f64 %f_c, %f_a, %f_b;\n    \
                         st.global.f64 [%rd_c], %f_c;"
                    ));
                });
                b.ret();
            })
            .build()
    }

    /// Generates a minimal kernel (body: just `ret`) for the given SM version.
    fn generate_simple_kernel_ptx(sm: SmVersion) -> Result<String, crate::error::PtxGenError> {
        KernelBuilder::new("simple_kernel")
            .target(sm)
            .param("n", PtxType::U32)
            .body(|b| {
                b.ret();
            })
            .build()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // P1 — vector_add PTX generation tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn p1_vector_add_f32_contains_version_header() {
        let ptx =
            generate_vector_add_ptx_f32_raw(SmVersion::Sm80).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".version"),
            "PTX must contain .version directive; got:\n{ptx}"
        );
    }

    #[test]
    fn p1_vector_add_f32_contains_correct_target() {
        let ptx =
            generate_vector_add_ptx_f32_raw(SmVersion::Sm80).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".target sm_80"),
            "PTX must contain '.target sm_80'; got:\n{ptx}"
        );
    }

    #[test]
    fn p1_vector_add_f32_contains_entry_name() {
        let ptx =
            generate_vector_add_ptx_f32_raw(SmVersion::Sm80).expect("PTX generation must succeed");
        assert!(
            ptx.contains("vector_add"),
            "PTX must contain kernel name 'vector_add'; got:\n{ptx}"
        );
    }

    #[test]
    fn p1_vector_add_f32_contains_parameter_loads() {
        let ptx =
            generate_vector_add_ptx_f32_raw(SmVersion::Sm80).expect("PTX generation must succeed");
        // Parameters are loaded with ld.param
        assert!(
            ptx.contains("ld.param"),
            "PTX must contain parameter loading instructions; got:\n{ptx}"
        );
    }

    #[test]
    fn p1_vector_add_f32_contains_global_thread_id_computation() {
        let ptx =
            generate_vector_add_ptx_f32_raw(SmVersion::Sm80).expect("PTX generation must succeed");
        // global_thread_id_x() emits mad.lo.u32
        assert!(
            ptx.contains("mad.lo.u32"),
            "PTX must contain global thread ID computation (mad.lo.u32); got:\n{ptx}"
        );
    }

    #[test]
    fn p1_vector_add_f32_contains_global_load() {
        let ptx =
            generate_vector_add_ptx_f32_raw(SmVersion::Sm80).expect("PTX generation must succeed");
        assert!(
            ptx.contains("ld.global"),
            "PTX must contain global memory load; got:\n{ptx}"
        );
    }

    #[test]
    fn p1_vector_add_f32_contains_add_operation() {
        let ptx =
            generate_vector_add_ptx_f32_raw(SmVersion::Sm80).expect("PTX generation must succeed");
        assert!(
            ptx.contains("add.f32") || ptx.contains("fma.rn.f32"),
            "PTX must contain add.f32 or fma.rn.f32 operation; got:\n{ptx}"
        );
    }

    #[test]
    fn p1_vector_add_f32_contains_global_store() {
        let ptx =
            generate_vector_add_ptx_f32_raw(SmVersion::Sm80).expect("PTX generation must succeed");
        assert!(
            ptx.contains("st.global"),
            "PTX must contain global memory store; got:\n{ptx}"
        );
    }

    #[test]
    fn p1_vector_add_f32_contains_bounds_check() {
        let ptx =
            generate_vector_add_ptx_f32_raw(SmVersion::Sm80).expect("PTX generation must succeed");
        // if_lt_u32 emits setp.lo.u32 + conditional branch
        assert!(
            ptx.contains("setp") || ptx.contains("bra"),
            "PTX must contain bounds check (setp/bra); got:\n{ptx}"
        );
    }

    #[test]
    fn p1_vector_add_f32_well_formed_structure() {
        let ptx =
            generate_vector_add_ptx_f32_raw(SmVersion::Sm80).expect("PTX generation must succeed");
        // Must contain a complete entry block with braces
        assert!(
            ptx.contains(".visible .entry vector_add"),
            "PTX must contain .visible .entry vector_add; got:\n{ptx}"
        );
        assert!(ptx.contains("ret;"), "PTX must contain ret; got:\n{ptx}");
    }

    #[test]
    fn p1_vector_add_f16_contains_required_elements() {
        let ptx =
            generate_vector_add_ptx_f16(SmVersion::Sm80).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".version"),
            "f16 PTX must contain .version; got:\n{ptx}"
        );
        assert!(
            ptx.contains("vector_add"),
            "f16 PTX must contain kernel name; got:\n{ptx}"
        );
        assert!(
            ptx.contains("ld.global"),
            "f16 PTX must contain global load; got:\n{ptx}"
        );
        assert!(
            ptx.contains("st.global"),
            "f16 PTX must contain global store; got:\n{ptx}"
        );
        assert!(
            ptx.contains("add.f16") || ptx.contains("fma.rn.f16"),
            "f16 PTX must contain f16 add; got:\n{ptx}"
        );
    }

    #[test]
    fn p1_vector_add_f64_contains_required_elements() {
        let ptx =
            generate_vector_add_ptx_f64(SmVersion::Sm80).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".version"),
            "f64 PTX must contain .version; got:\n{ptx}"
        );
        assert!(
            ptx.contains("vector_add"),
            "f64 PTX must contain kernel name; got:\n{ptx}"
        );
        assert!(
            ptx.contains("ld.global"),
            "f64 PTX must contain global load; got:\n{ptx}"
        );
        assert!(
            ptx.contains("st.global"),
            "f64 PTX must contain global store; got:\n{ptx}"
        );
        assert!(
            ptx.contains("add.f64") || ptx.contains("fma.rn.f64"),
            "f64 PTX must contain f64 add; got:\n{ptx}"
        );
    }

    #[test]
    fn p1_vector_add_f32_address_size_64() {
        let ptx =
            generate_vector_add_ptx_f32_raw(SmVersion::Sm80).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".address_size 64"),
            "PTX must contain .address_size 64; got:\n{ptx}"
        );
    }

    #[test]
    fn p1_vector_add_f32_has_u64_params() {
        let ptx =
            generate_vector_add_ptx_f32_raw(SmVersion::Sm80).expect("PTX generation must succeed");
        // Pointer params are .param .u64
        assert!(
            ptx.contains(".param .u64"),
            "PTX must contain .param .u64 for pointer args; got:\n{ptx}"
        );
        // Length param is .param .u32
        assert!(
            ptx.contains(".param .u32"),
            "PTX must contain .param .u32 for n; got:\n{ptx}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // P8 — All SM versions produce valid PTX headers
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn p8_ptx_target_sm75() {
        let ptx = generate_simple_kernel_ptx(SmVersion::Sm75).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".target sm_75"),
            "Expected .target sm_75 in PTX; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_target_sm80() {
        let ptx = generate_simple_kernel_ptx(SmVersion::Sm80).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".target sm_80"),
            "Expected .target sm_80 in PTX; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_target_sm86() {
        let ptx = generate_simple_kernel_ptx(SmVersion::Sm86).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".target sm_86"),
            "Expected .target sm_86 in PTX; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_target_sm89() {
        let ptx = generate_simple_kernel_ptx(SmVersion::Sm89).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".target sm_89"),
            "Expected .target sm_89 in PTX; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_target_sm90() {
        let ptx = generate_simple_kernel_ptx(SmVersion::Sm90).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".target sm_90"),
            "Expected .target sm_90 in PTX; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_target_sm90a() {
        let ptx =
            generate_simple_kernel_ptx(SmVersion::Sm90a).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".target sm_90a"),
            "Expected .target sm_90a in PTX; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_target_sm100() {
        let ptx =
            generate_simple_kernel_ptx(SmVersion::Sm100).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".target sm_100"),
            "Expected .target sm_100 in PTX; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_target_sm120() {
        let ptx =
            generate_simple_kernel_ptx(SmVersion::Sm120).expect("PTX generation must succeed");
        assert!(
            ptx.contains(".target sm_120"),
            "Expected .target sm_120 in PTX; got:\n{ptx}"
        );
    }

    /// Verify that the `.version` directive in the generated PTX matches
    /// the expected PTX ISA version for each SM variant.
    #[test]
    fn p8_ptx_version_matches_sm75() {
        let ptx = generate_simple_kernel_ptx(SmVersion::Sm75).expect("PTX generation must succeed");
        let expected_version = SmVersion::Sm75.ptx_version();
        assert!(
            ptx.contains(&format!(".version {expected_version}")),
            "Expected .version {expected_version} for sm_75; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_version_matches_sm80() {
        let ptx = generate_simple_kernel_ptx(SmVersion::Sm80).expect("PTX generation must succeed");
        let expected_version = SmVersion::Sm80.ptx_version();
        assert!(
            ptx.contains(&format!(".version {expected_version}")),
            "Expected .version {expected_version} for sm_80; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_version_matches_sm86() {
        let ptx = generate_simple_kernel_ptx(SmVersion::Sm86).expect("PTX generation must succeed");
        let expected_version = SmVersion::Sm86.ptx_version();
        assert!(
            ptx.contains(&format!(".version {expected_version}")),
            "Expected .version {expected_version} for sm_86; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_version_matches_sm89() {
        let ptx = generate_simple_kernel_ptx(SmVersion::Sm89).expect("PTX generation must succeed");
        let expected_version = SmVersion::Sm89.ptx_version();
        assert!(
            ptx.contains(&format!(".version {expected_version}")),
            "Expected .version {expected_version} for sm_89; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_version_matches_sm90() {
        let ptx = generate_simple_kernel_ptx(SmVersion::Sm90).expect("PTX generation must succeed");
        let expected_version = SmVersion::Sm90.ptx_version();
        assert!(
            ptx.contains(&format!(".version {expected_version}")),
            "Expected .version {expected_version} for sm_90; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_version_matches_sm90a() {
        let ptx =
            generate_simple_kernel_ptx(SmVersion::Sm90a).expect("PTX generation must succeed");
        // sm_90a shares the same PTX version as sm_90 (8.0)
        let expected_version = SmVersion::Sm90a.ptx_version();
        assert!(
            ptx.contains(&format!(".version {expected_version}")),
            "Expected .version {expected_version} for sm_90a; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_version_matches_sm100() {
        let ptx =
            generate_simple_kernel_ptx(SmVersion::Sm100).expect("PTX generation must succeed");
        let expected_version = SmVersion::Sm100.ptx_version();
        assert!(
            ptx.contains(&format!(".version {expected_version}")),
            "Expected .version {expected_version} for sm_100; got:\n{ptx}"
        );
    }

    #[test]
    fn p8_ptx_version_matches_sm120() {
        let ptx =
            generate_simple_kernel_ptx(SmVersion::Sm120).expect("PTX generation must succeed");
        let expected_version = SmVersion::Sm120.ptx_version();
        assert!(
            ptx.contains(&format!(".version {expected_version}")),
            "Expected .version {expected_version} for sm_120; got:\n{ptx}"
        );
    }

    /// Cross-check: the `.version` value reflects the SM version ordering —
    /// higher SM → newer (≥) PTX ISA version.
    #[test]
    fn p8_ptx_version_ordering_respects_sm_ordering() {
        let versions_in_order = [
            SmVersion::Sm75,
            SmVersion::Sm80,
            SmVersion::Sm86,
            SmVersion::Sm89,
            SmVersion::Sm90,
            SmVersion::Sm90a,
            SmVersion::Sm100,
            SmVersion::Sm120,
        ];

        let isa_versions: Vec<(u32, u32)> = versions_in_order
            .iter()
            .map(|sm| sm.ptx_isa_version())
            .collect();

        // Each entry must be >= the previous one.
        for window in isa_versions.windows(2) {
            let (prev_major, prev_minor) = window[0];
            let (curr_major, curr_minor) = window[1];
            assert!(
                (curr_major, curr_minor) >= (prev_major, prev_minor),
                "PTX ISA version must be non-decreasing across SM versions: \
                 {prev_major}.{prev_minor} -> {curr_major}.{curr_minor}"
            );
        }
    }

    /// FP8 types (E4M3/E5M2) require SM 89 or higher.
    ///
    /// On SM 80 (no FP8 support), verify capability flag is false.
    /// On SM 89+ (FP8 supported), verify capability flag is true.
    #[test]
    fn p8_fp8_types_require_sm89_or_higher() {
        // Architectures without FP8
        let no_fp8 = [SmVersion::Sm75, SmVersion::Sm80, SmVersion::Sm86];
        for sm in no_fp8 {
            assert!(
                !sm.capabilities().has_fp8,
                "{sm} should not have FP8 support"
            );
        }

        // Architectures with FP8
        let has_fp8 = [
            SmVersion::Sm89,
            SmVersion::Sm90,
            SmVersion::Sm90a,
            SmVersion::Sm100,
            SmVersion::Sm120,
        ];
        for sm in has_fp8 {
            assert!(sm.capabilities().has_fp8, "{sm} should have FP8 support");
        }
    }

    /// FP8 param type `.b8` appears in PTX only when explicitly used;
    /// verify the `param_type_str` mapping is consistent (via an actual kernel build).
    #[test]
    fn p8_fp8_param_type_in_ptx_sm89() {
        // On SM 89, a kernel with an E4M3 param should emit `.param .b8`
        let ptx = KernelBuilder::new("fp8_kernel")
            .target(SmVersion::Sm89)
            .param("scale", PtxType::E4M3)
            .body(|b| {
                b.ret();
            })
            .build()
            .expect("PTX generation must succeed");

        assert!(
            ptx.contains(".param .b8"),
            "SM 89 kernel with E4M3 param must have .param .b8; got:\n{ptx}"
        );
        assert!(
            ptx.contains(".target sm_89"),
            "Must target sm_89; got:\n{ptx}"
        );
    }

    /// On architectures that do NOT support FP8, the param type mapping still
    /// works (it's a code-gen gate, not a hardware check at PTX-text level).
    /// Verify SM 80 doesn't gain FP8 capability even if we generate the text.
    #[test]
    fn p8_sm80_has_no_fp8_capability() {
        let caps = SmVersion::Sm80.capabilities();
        assert!(!caps.has_fp8, "SM 80 must not advertise FP8 capability");
        assert!(
            !caps.has_fp6_fp4,
            "SM 80 must not advertise FP6/FP4 capability"
        );
    }

    /// All SM versions must produce a kernel with `.address_size 64`.
    #[test]
    fn p8_all_sm_produce_address_size_64() {
        let all_sm = [
            SmVersion::Sm75,
            SmVersion::Sm80,
            SmVersion::Sm86,
            SmVersion::Sm89,
            SmVersion::Sm90,
            SmVersion::Sm90a,
            SmVersion::Sm100,
            SmVersion::Sm120,
        ];
        for sm in all_sm {
            let ptx = generate_simple_kernel_ptx(sm)
                .unwrap_or_else(|e| panic!("PTX generation failed for {sm}: {e}"));
            assert!(
                ptx.contains(".address_size 64"),
                "{sm} PTX must contain .address_size 64; got:\n{ptx}"
            );
        }
    }

    /// Every SM variant must produce a non-empty PTX string.
    #[test]
    fn p8_all_sm_produce_non_empty_ptx() {
        let all_sm = [
            SmVersion::Sm75,
            SmVersion::Sm80,
            SmVersion::Sm86,
            SmVersion::Sm89,
            SmVersion::Sm90,
            SmVersion::Sm90a,
            SmVersion::Sm100,
            SmVersion::Sm120,
        ];
        for sm in all_sm {
            let ptx = generate_simple_kernel_ptx(sm)
                .unwrap_or_else(|e| panic!("PTX generation failed for {sm}: {e}"));
            assert!(!ptx.is_empty(), "PTX for {sm} must not be empty");
            assert!(
                ptx.len() > 50,
                "PTX for {sm} is suspiciously short: {ptx:?}"
            );
        }
    }
}
